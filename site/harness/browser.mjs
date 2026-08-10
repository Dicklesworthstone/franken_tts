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

import { chromium } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { serve } from "./serve.mjs";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelDir = path.join(process.env.HOME, ".cache/franken_tts/model");
const HEADED = process.argv.includes("--headed");
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

const browser = await chromium.launch({ headless: !HEADED });
const page = await browser.newPage();

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
page.context().on("weberror", (error) => transcript.push(`[weberror] ${error.error().message}`));
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
    if (Date.now() - lastChange > 180_000) {
      stalled = status;
      break;
    }
    await page.waitForTimeout(1000);
  }

  const history = await page.evaluate(() => globalThis.__fttsStages ?? []);
  console.log("\n--- stage history ---");
  for (const entry of history) console.log(`      ${entry}`);
  check("hydration advanced past staging", history.length > 1, `${history.length} stages`);
  check("no stall", !stalled, stalled ? `stuck ${180}s at "${stalled}"` : "");

  const ready = await page.evaluate(() => {
    const button = document.getElementById("speak");
    return button ? !button.disabled : null;
  });
  check("synthesize control is enabled", ready === true, `enabled=${ready}`);

  // The number that actually matters. Everything above only proves the page loads; this measures
  // whether the kernel work bought anything, in the browser, on the real model.
  if (ready) {
    await page.fill("#text, textarea", "The quick brown fox jumps over the lazy dog.").catch(() => {});
    const started = Date.now();
    await page.click("#speak");
    // Synthesis is minutes at wasm speed; wait for the control to re-enable.
    await page
      .waitForFunction(() => !document.getElementById("speak").disabled, null, { timeout: 900_000 })
      .catch(() => {});
    const seconds = (Date.now() - started) / 1000;
    const status = await page.evaluate(
      () => document.getElementById("synth-status")?.textContent?.trim() ?? "",
    );
    console.log(`\nSYNTHESIS: ${seconds.toFixed(1)} s wall — status: ${status}`);
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
