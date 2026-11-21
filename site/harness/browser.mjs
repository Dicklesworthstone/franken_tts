// The real thing: the real site, in a real browser, over a real server.
//
// Run:  node site/harness/browser.mjs [--headed] [--keep]
//
// There are NO shims here, and that is the entire point. A previous Node harness stubbed fetch,
// OPFS, Worker and navigator, passed all eight of its cases, and the deployed site still hung on
// every browser — because the bug lived in the gap between the stubs and reality. Anything mocked
// is untested by definition, so nothing is mocked: real Chromium, real OPFS, real Workers, real
// COOP/COEP, real WebAssembly, real byte-Range downloads of the real model.
//
// What it asserts, in order, because each failure looked identical from outside ("it just sits
// there"):
//   1. the page becomes crossOriginIsolated (decides which build loads)
//   2. the engine Worker reports init, and says which build and how many threads
//   3. hydration walks its stages and REACHES THE END rather than stalling
//   4. the synthesize control actually becomes enabled
//
// Every console message and page error is captured, so a silent failure in a Worker — invisible in
// the page UI, which is how the last one escaped — shows up in the transcript.

import { chromium, webkit } from "playwright";
import path from "node:path";
import fs from "node:fs/promises";
import os from "node:os";
import { fileURLToPath } from "node:url";
import { serve } from "./serve.mjs";
import { sweepStaleWebKitOrigins, WEBKIT_HARNESS_PORT } from "./webkit_storage.mjs";
import {
  EXPECTED_SAMPLE_RATE,
  wav16,
  readWav16,
  validateBrowserPcm,
  compareBrowserToGolden,
} from "./parity.mjs";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelDir = path.join(process.env.HOME, ".cache/franken_tts/model");
const HEADED = process.argv.includes("--headed");
// `--webkit` runs the SAME assertions in Safari's engine family.
//
// This is the closest reachable proxy for iOS: mobile Safari cannot be driven programmatically
// (Apple dropped iOS from safaridriver), but WebKit exercises the branch that matters — the
// engine-worker's Safari detection routes to the UNSHARED serial build, whose growable memory is
// the configuration the iPhone probe showed to be the safe one. A pass here does not prove the
// phone works; it proves the serial path is not broken in the engine the phone runs.
const ENGINE = process.argv.includes("--webkit") ? webkit : chromium;
const ENGINE_NAME = process.argv.includes("--webkit") ? "webkit" : "chromium";
const KEEP = process.argv.includes("--keep");
// Conformance mode: compare the browser's own PCM against a WAV the CLI produced for the SAME
// text, voice and seed. This is the only comparison that means anything — the browser and the CLI
// share every line of the engine, so any divergence is a wasm-vs-native fork, not a model
// difference, and there are known forks (codec int8 arm, f32 accumulation order).
const argValue = (flag) => {
  const at = process.argv.indexOf(flag);
  return at >= 0 ? process.argv[at + 1] : null;
};
const CLI_GOLDEN = argValue("--cli-golden");
const PCM_OUT = argValue("--pcm-out");
const CONFORMANCE_TEXT = "The quick brown fox jumps over the lazy dog.";
// `--packet-frames N` overrides the codec's packet size for this run. Output is bit-identical
// under every schedule (the streaming==batch gate), so a sweep measures speed and memory only —
// and the CLI-parity check still has to pass at each size, which is what proves that claim here
// rather than assuming it.
const PACKET_FRAMES = Number(argValue("--packet-frames") ?? 0);
// `--debug-taps` turns on the DISC-006 cross-target seam taps (frankentts-p16p): one
// console line per frame whose hashes must match the native run's `FTTS_DEBUG_TAPS` lines
// exactly until the first divergent operator.
const DEBUG_TAPS = process.argv.includes("--debug-taps");
// `--canonical-math` turns on the canonical cross-target RMSNorm (frankentts-p16p): the
// native side must run with FTTS_CANONICAL_NORM=1 for the comparison to be apples-to-apples.
const CANONICAL_MATH = process.argv.includes("--canonical-math");
// `--transcript-out <path>` writes the COMPLETE captured console/page/worker transcript
// (not the tail-only dump this script prints) — DISC-006 tap runs need every
// `ftts-tap` line, and long downloads push early ones out of the printed window.
const TRANSCRIPT_OUT = argValue("--transcript-out");

const modelFiles = {
  "qwen3-tts-12hz-0.6b-base.fttsq": path.join(modelDir, "qwen3-tts-12hz-0.6b-base.fttsq"),
  "speech_tokenizer_model.safetensors": path.join(modelDir, "speech_tokenizer/model.safetensors"),
  "vocab.json": path.join(modelDir, "vocab.json"),
  "merges.txt": path.join(modelDir, "merges.txt"),
  "tokenizer_config.json": path.join(modelDir, "tokenizer_config.json"),
};

// WebKit OPFS is per origin, and the origin includes the port. A random port
// stored another tokenizer under ~/Library/WebKit on every run; the temp
// profile removed below does not contain that directory. Pin the port and
// drop every other loopback origin before launch.
const webkitRun = ENGINE_NAME === "webkit";
let served;
try {
  served = await serve({
    siteDir,
    modelFiles,
    port: webkitRun ? WEBKIT_HARNESS_PORT : 0,
  });
} catch (error) {
  if (webkitRun && error?.code === "EADDRINUSE") {
    console.error(
      `WebKit harness port ${WEBKIT_HARNESS_PORT} is already in use. ` +
        "Not falling back to a random port: each port keeps another OPFS copy in ~/Library/WebKit.",
    );
    process.exit(1);
  }
  throw error;
}
const { server, port, headers } = served;
const transcript = [];
const pageErrors = [];
const webErrors = [];
let profile = null;
let browser = null;

let failures = 0;
const check = (label, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
  return ok;
};

try {
  if (webkitRun) {
    const swept = await sweepStaleWebKitOrigins({ keepPort: port });
    if (swept.removed) {
      console.log(
        `removed ${swept.removed} stale WebKit loopback origin(s), ${(swept.bytes / 1024 ** 3).toFixed(2)} GB`,
      );
    }
  }
  console.log(`serving ${siteDir} on :${port}`);
  console.log(`  COOP=${headers["Cross-Origin-Opener-Policy"]} COEP=${headers["Cross-Origin-Embedder-Policy"]}`);

  console.log(`browser engine: ${ENGINE_NAME}`);
  // PERSISTENT profile, and it is not optional. WebKit's OPFS needs a real storage directory: in an
  // ephemeral context the first `getFileHandle` fails with "operation failed for an unknown transient
  // reason", the download never starts, and the run looks like a site bug. Chromium tolerates the
  // ephemeral case, so testing only Chromium hides the difference — which is exactly how a WebKit
  // "stall" got mistaken for a real defect once already.
  profile = await fs.mkdtemp(path.join(os.tmpdir(), `ftts-harness-${ENGINE_NAME}-`));
  browser = await ENGINE.launchPersistentContext(profile, { headless: !HEADED });
  const page = browser.pages()[0] ?? (await browser.newPage());

  page.on("console", (message) => {
    const text = message.text();
    transcript.push(`[${message.type()}] ${text}`);
    // The engine's own stage timing goes to the worker console, which never reaches the page.
    // Surfacing it here is how the codec-vs-talker split becomes visible.
    if (text.includes("ftts-wasm timing")) console.log(`\n      ${text}`);
  });
  page.on("pageerror", (error) => {
    pageErrors.push(error.message);
    transcript.push(`[pageerror] ${error.message}`);
  });
  // Worker consoles and errors do NOT surface on the page by default, which is exactly how a fatal
  // error inside engine-worker.js looked like an innocent stall. Attach to every worker, including
  // the kernel workers it spawns.
  page.on("worker", (worker) => {
    transcript.push(`[worker started] ${worker.url()}`);
    worker.on("close", () => transcript.push(`[worker CLOSED] ${worker.url()}`));
  });
  browser.on("weberror", (error) => {
    const msg = error.error().message;
    webErrors.push(msg);
    transcript.push(`[weberror] ${msg}`);
  });
  page.on("requestfailed", (request) =>
    transcript.push(`[requestfailed] ${request.url()} ${request.failure()?.errorText}`),
  );
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "domcontentloaded" });

  const isolated = await page.evaluate(() => self.crossOriginIsolated);
  check("page is crossOriginIsolated", isolated, `crossOriginIsolated=${isolated}`);

  // Drive the real UI in the real order: "Download & load model…" reveals the consent panel, and
  // only "Start the … download" begins the transfer. Clicking blindly in one pass silently did
  // nothing and produced a "stall" at "Model not loaded." that was the harness's fault, not the
  // site's — the same class of false result this whole harness exists to eliminate.
  await page.click("#load-model");
  await page.waitForSelector("#consent-yes", { state: "visible", timeout: 15_000 });
  await page.click("#consent-yes");

  // Follow the live stage text rather than a fixed sleep: a stall is a stage that stops advancing,
  // which is precisely what "stuck indefinitely at staging-new" was.
  const seen = [];
  let lastChange = Date.now();
  let status = "";
  const deadline = Date.now() + 15 * 60 * 1000;
  let stalled = null;
  // One constant, used by both the detector and the report: they disagreed once, and a report
  // that says "180s" after waiting 420 sends the next reader down the wrong path entirely.
  const STALL_MS = 420_000;

  while (Date.now() < deadline) {
    const next = await page.evaluate(() => {
      const node = document.getElementById("dl-status");
      return node ? node.textContent.trim() : "";
    });
    if (next !== status) {
      status = next;
      seen.push(status);
      lastChange = Date.now();
      console.log(`      stage: ${status}`);
    }
    const enabled = await page.evaluate(() => {
      const button = document.getElementById("speak");
      return button ? !button.disabled : false;
    });
    if (enabled) break;
    // Hydration is one synchronous wasm call that posts no progress while it runs, so the stage
    // text is legitimately frozen for its whole duration. The window has to exceed the slowest
    // real hydration or the harness invents a stall — which it did, at "hydrate-talker", for a
    // build that was merely working.
    if (Date.now() - lastChange > STALL_MS) {
      stalled = status;
      break;
    }
    await page.waitForTimeout(1000);
  }

  const history = await page.evaluate(() => globalThis.__fttsStages ?? []);
  console.log("\n--- stage history ---");
  for (const entry of history) console.log(`      ${entry}`);
  check("hydration advanced past staging", history.length > 1, `${history.length} stages`);
  check("no stall", !stalled, stalled ? `stuck ${STALL_MS / 1000}s at "${stalled}"` : "");

  const ready = await page.evaluate(() => {
    const button = document.getElementById("speak");
    return button ? !button.disabled : null;
  });
  check("synthesize control is enabled", ready === true, `enabled=${ready}`);

  // Read the page's own error banner. The wasm build sets `panic = "abort"`, which means
  // `set_hook` never runs and a Rust panic is a SILENT trap — the only surviving evidence is the
  // RuntimeError the worker forwards here. Watching only the stage text reported that as a
  // "stall", which sent the diagnosis after a phantom performance problem instead of a real
  // error that the page had been displaying the whole time.
  const pageError = await page.evaluate(
    () => document.getElementById("error")?.textContent?.trim() ?? "",
  );
  if (pageError) console.log(`\nPAGE ERROR: ${pageError}`);
  check("no hydration error banner", pageError.length === 0, pageError);

  // The number that actually matters. Everything above only proves the page loads; this measures
  // whether the kernel work bought anything, in the browser, on the real model.
  if (ready) {
    // Set the inputs, then READ THEM BACK. Every one of these was previously set with the error
    // swallowed, which means a selector that matched the wrong element — or nothing — produced a
    // parity comparison against different text or a different voice, and reported it as a
    // divergence in the engine. Three separate harness lies this session started exactly here.
    await page.fill("#text", CONFORMANCE_TEXT);
    // Pinned so the CLI golden is reproducible: the sampler is seeded, and an unpinned seed would
    // make every comparison a fresh coin flip rather than a parity test.
    await page.fill("#seed", "0");
    await page.selectOption("#voice", "matt");
    if (PACKET_FRAMES > 0) {
      await page.evaluate((n) => {
        globalThis.__fttsPacketFrames = n;
      }, PACKET_FRAMES);
      console.log(`      packet frames: ${PACKET_FRAMES}`);
    }
    if (DEBUG_TAPS) {
      await page.evaluate(() => {
        globalThis.__fttsDebugTaps = true;
      });
      console.log("      debug taps: on");
    }
    if (CANONICAL_MATH) {
      await page.evaluate(() => {
        globalThis.__fttsCanonicalNorm = true;
      });
      console.log("      canonical math: on");
    }
    const inputs = await page.evaluate(() => ({
      text: document.getElementById("text")?.value ?? null,
      seed: document.getElementById("seed")?.value ?? null,
      voice: document.getElementById("voice")?.value ?? null,
    }));
    console.log(`      inputs: voice=${inputs.voice} seed=${inputs.seed} text=${JSON.stringify(inputs.text)}`);
    check(
      "conformance inputs are what the CLI golden was made with",
      inputs.text === CONFORMANCE_TEXT && inputs.seed === "0" && inputs.voice === "matt",
      `voice=${inputs.voice} seed=${inputs.seed}`,
    );
    await page.evaluate(() => {
      globalThis.__fttsLastPcm = null;
      globalThis.__fttsLastSampleRate = null;
    });
    const started = Date.now();
    await page.click("#speak");
    // Wait for the control to go DOWN before waiting for it to come back up. Checking only for
    // "enabled" races the click handler: the button is still enabled for the instant between the
    // click and the handler disabling it, so the wait returned immediately, reported 0.0 s, and
    // the run exited while synthesis was still going — a pass that measured nothing.
    let synthStarted = true;
    try {
      await page.waitForFunction(() => document.getElementById("speak")?.disabled === true, null, {
        timeout: 30_000,
      });
    } catch (error) {
      synthStarted = false;
      check("synthesis started (speak button disabled)", false, error.message);
    }
    if (synthStarted) {
      check("synthesis started (speak button disabled)", true);
    }
    // Synthesis is minutes at wasm speed; now wait for it to re-enable.
    let synthCompleted = true;
    try {
      await page.waitForFunction(() => document.getElementById("speak")?.disabled === false, null, {
        timeout: 900_000,
      });
    } catch (error) {
      synthCompleted = false;
      check("synthesis completed within timeout", false, error.message);
    }
    if (synthCompleted) {
      check("synthesis completed within timeout", true);
    }
    const seconds = (Date.now() - started) / 1000;
    const status = await page.evaluate(
      () => document.getElementById("synth-status")?.textContent?.trim() ?? "",
    );
    console.log(`\nSYNTHESIS: ${seconds.toFixed(1)} s wall — status: ${status}`);
    // The error banner AFTER synthesis, not just before it. A synthesis that fails leaves the
    // status line empty and the button enabled, which is indistinguishable from one that never
    // started — and the only thing that tells them apart is this element.
    const synthError = await page.evaluate(
      () => document.getElementById("error")?.textContent?.trim() ?? "",
    );
    if (synthError) console.log(`SYNTH ERROR: ${synthError}`);
    check("no synthesis error banner", synthError.length === 0, synthError);
    check(
      "no uncaught page or worker exceptions",
      pageErrors.length === 0 && webErrors.length === 0,
      [...pageErrors, ...webErrors].join("; "),
    );

    // ── conformance: require fresh finite 24 kHz PCM and full-length CLI parity ─────────────
    const pcm = await page.evaluate(() =>
      globalThis.__fttsLastPcm ? Array.from(globalThis.__fttsLastPcm) : null,
    );
    const rate = await page.evaluate(() => globalThis.__fttsLastSampleRate ?? null);
    const pcmCheck = validateBrowserPcm(pcm, rate, EXPECTED_SAMPLE_RATE);
    check(
      "browser produced fresh non-empty finite 24 kHz PCM",
      pcmCheck.ok,
      pcmCheck.ok ? `${pcmCheck.samples} samples @ ${pcmCheck.rate} Hz` : pcmCheck.reason,
    );
    if (pcmCheck.ok && PCM_OUT) {
      await fs.writeFile(PCM_OUT, wav16(pcm, rate));
      console.log(`wrote ${PCM_OUT} (${pcm.length} samples @ ${rate} Hz)`);
    }
    if (CLI_GOLDEN) {
      if (pcmCheck.ok) {
        const goldenWav = readWav16(await fs.readFile(CLI_GOLDEN), EXPECTED_SAMPLE_RATE);
        const cmp = compareBrowserToGolden(pcm, rate, goldenWav.samples, EXPECTED_SAMPLE_RATE);
        console.log(`      raw f32 deltas at differing samples: ${JSON.stringify(cmp.rawDiffs)}`);
        console.log(`\n--- CLI PARITY (${CONFORMANCE_TEXT.slice(0, 32)}…, voice matt, seed 0) ---`);
        console.log(`      lengths: browser ${cmp.browserLen}, cli ${cmp.goldenLen}, compared ${cmp.compared}`);
        const pct = cmp.compared ? ((100 * cmp.identical) / cmp.compared).toFixed(2) : "0.00";
        console.log(`      identical samples: ${cmp.identical}/${cmp.compared} (${pct}%)`);
        console.log(`      max |diff|: ${cmp.maxDiff} LSB of 32767`);
        console.log(`      SNR of the difference: ${cmp.snr.toFixed(1)} dB`);
        check("browser output matches the CLI sample-for-sample (full length)", cmp.ok, cmp.reason);
      } else {
        check("browser output matches the CLI sample-for-sample (full length)", false, pcmCheck.reason);
      }
    }
    // Stages recorded DURING synthesis. The history dump above happens at "ready", so anything
    // synthesis reports — including how much linear memory it claims, which is now the number
    // that decides whether a phone survives pressing the button — was never printed.
    const after = await page.evaluate(() => globalThis.__fttsStages ?? []);
    for (const entry of after.slice(history.length)) console.log(`      ${entry}`);
  }
} finally {
  console.log("\n--- browser transcript ---");
  for (const line of transcript.slice(-60)) console.log(line);
  if (TRANSCRIPT_OUT) {
    await fs.writeFile(TRANSCRIPT_OUT, transcript.join("\n")).catch((error) => {
      console.error(`warning: could not write transcript ${TRANSCRIPT_OUT}: ${error.message}`);
      failures += 1;
    });
    console.log(`full transcript: ${TRANSCRIPT_OUT} (${transcript.length} lines)`);
  }
  if (browser) {
    await browser.close().catch((error) => {
      console.error(`warning: could not close browser context: ${error.message}`);
    });
  }
  if (server) {
    await new Promise((resolve) => server.close(() => resolve()));
  }
  if (profile) {
    if (KEEP) {
      console.log(`retained browser profile: ${profile}`);
    } else {
      // Remove persistent profile only AFTER browser closure.
      await fs.rm(profile, { recursive: true, force: true }).catch((error) => {
        console.log(`warning: could not remove profile ${profile}: ${error.message}`);
      });
    }
  }
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exitCode = failures ? 1 : 0;
