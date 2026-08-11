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

const modelFiles = {
  "qwen3-tts-12hz-0.6b-base.fttsq": path.join(modelDir, "qwen3-tts-12hz-0.6b-base.fttsq"),
  "speech_tokenizer_model.safetensors": path.join(modelDir, "speech_tokenizer/model.safetensors"),
  "vocab.json": path.join(modelDir, "vocab.json"),
  "merges.txt": path.join(modelDir, "merges.txt"),
  "tokenizer_config.json": path.join(modelDir, "tokenizer_config.json"),
};

const { server, port, headers } = await serve({ siteDir, modelFiles });
console.log(`serving ${siteDir} on :${port}`);
console.log(`  COOP=${headers["Cross-Origin-Opener-Policy"]} COEP=${headers["Cross-Origin-Embedder-Policy"]}`);

console.log(`browser engine: ${ENGINE_NAME}`);
// PERSISTENT profile, and it is not optional. WebKit's OPFS needs a real storage directory: in an
// ephemeral context the first `getFileHandle` fails with "operation failed for an unknown transient
// reason", the download never starts, and the run looks like a site bug. Chromium tolerates the
// ephemeral case, so testing only Chromium hides the difference — which is exactly how a WebKit
// "stall" got mistaken for a real defect once already.
const profile = await fs.mkdtemp(path.join(os.tmpdir(), `ftts-harness-${ENGINE_NAME}-`));
const browser = await ENGINE.launchPersistentContext(profile, { headless: !HEADED });
const page = browser.pages()[0] ?? (await browser.newPage());

const transcript = [];
page.on("console", (message) => {
  const text = message.text();
  transcript.push(`[${message.type()}] ${text}`);
  // The engine's own stage timing goes to the worker console, which never reaches the page.
  // Surfacing it here is how the codec-vs-talker split becomes visible.
  if (text.includes("ftts-wasm timing")) console.log(`\n      ${text}`);
});
page.on("pageerror", (error) => transcript.push(`[pageerror] ${error.message}`));
// Worker consoles and errors do NOT surface on the page by default, which is exactly how a fatal
// error inside engine-worker.js looked like an innocent stall. Attach to every worker, including
// the kernel workers it spawns.
page.on("worker", (worker) => {
  transcript.push(`[worker started] ${worker.url()}`);
  worker.on("close", () => transcript.push(`[worker CLOSED] ${worker.url()}`));
});
browser.on("weberror", (error) => transcript.push(`[weberror] ${error.error().message}`));
page.on("requestfailed", (request) =>
  transcript.push(`[requestfailed] ${request.url()} ${request.failure()?.errorText}`),
);

let failures = 0;
const check = (label, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) failures += 1;
  return ok;
};

try {
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

  // The number that actually matters. Everything above only proves the page loads; this measures
  // whether the kernel work bought anything, in the browser, on the real model.
  if (ready) {
    await page.fill("#text, textarea", "The quick brown fox jumps over the lazy dog.").catch(() => {});
    const started = Date.now();
    await page.click("#speak");
    // Wait for the control to go DOWN before waiting for it to come back up. Checking only for
    // "enabled" races the click handler: the button is still enabled for the instant between the
    // click and the handler disabling it, so the wait returned immediately, reported 0.0 s, and
    // the run exited while synthesis was still going — a pass that measured nothing.
    await page
      .waitForFunction(() => document.getElementById("speak").disabled, null, { timeout: 30_000 })
      .catch(() => {});
    // Synthesis is minutes at wasm speed; now wait for it to re-enable.
    await page
      .waitForFunction(() => !document.getElementById("speak").disabled, null, { timeout: 900_000 })
      .catch(() => {});
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
    // Stages recorded DURING synthesis. The history dump above happens at "ready", so anything
    // synthesis reports — including how much linear memory it claims, which is now the number
    // that decides whether a phone survives pressing the button — was never printed.
    const after = await page.evaluate(() => globalThis.__fttsStages ?? []);
    for (const entry of after.slice(history.length)) console.log(`      ${entry}`);
  }
} finally {
  console.log("\n--- browser transcript ---");
  for (const line of transcript.slice(-60)) console.log(line);
  if (!KEEP) {
    await browser.close();
    server.close();
  }
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
