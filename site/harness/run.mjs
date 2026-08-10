// Drives the real engine-worker.js through the same postMessage protocol app.js uses.
//
// Run:  node site/harness/run.mjs [--full]
//
// Without --full it runs init only, across every (userAgent x crossOriginIsolated) combination —
// seconds, no model needed, and it is the matrix that would have caught the build-selection bug
// that bricked the site. With --full it also streams the real model in and hydrates, per case.
//
// Exit code is 0 only if every case reaches its expected state; a case that HANGS is reported as a
// timeout rather than being waited on forever, because "stuck indefinitely" is precisely the
// failure mode being hunted.

import { Worker } from "node:worker_threads";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelDir = path.join(process.env.HOME, ".cache/franken_tts/model");
const FULL = process.argv.includes("--full");

const assets = {
  "qwen3-tts-12hz-0.6b-base.fttsq": path.join(modelDir, "qwen3-tts-12hz-0.6b-base.fttsq"),
  "speech_tokenizer_model.safetensors": path.join(modelDir, "speech_tokenizer/model.safetensors"),
};

const UA = {
  chrome:
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0 Safari/537.36",
  safari:
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
  iphone:
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
  firefox:
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0",
};

const TIMEOUT_INIT = 60_000;
const TIMEOUT_LOAD = 600_000;

function runCase({ label, userAgent, isolated, full }) {
  return new Promise((resolve) => {
    const worker = new Worker(path.join(siteDir, "harness", "worker-host.mjs"), {
      workerData: { modelDir, siteDir, userAgent, isolated, assets },
    });
    const stages = [];
    let settled = false;
    const finish = (outcome) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      worker.terminate();
      resolve({ label, stages, ...outcome });
    };
    let timer = setTimeout(
      () => finish({ ok: false, why: `TIMEOUT after init (last stage: ${stages.at(-1) ?? "none"})` }),
      TIMEOUT_INIT,
    );

    worker.on("message", (data) => {
      if (data.type === "stage") {
        stages.push(data.stage);
        return;
      }
      if (data.type === "loadProgress") return;
      if (data.type === "init") {
        if (!data.ok) return finish({ ok: false, why: `init failed: ${data.error}` });
        if (!full) return finish({ ok: true, threads: data.threads, route: data.route });
        clearTimeout(timer);
        timer = setTimeout(
          () => finish({ ok: false, why: `TIMEOUT during load (last stage: ${stages.at(-1)})` }),
          TIMEOUT_LOAD,
        );
        worker.postMessage({
          type: "load",
          fttsq: { asset: "qwen3-tts-12hz-0.6b-base.fttsq", bytes: 1312015713 },
          codec: { asset: "speech_tokenizer_model.safetensors", bytes: 682293092 },
          // The real tokenizer files, not placeholders: hydration validates them, so stubs turn a
          // hydration test into a JSON-parsing test and hide whatever comes after.
          vocab: fs.readFileSync(path.join(modelDir, "vocab.json"), "utf8"),
          merges: fs.readFileSync(path.join(modelDir, "merges.txt"), "utf8"),
          tokenizerConfig: fs.readFileSync(path.join(modelDir, "tokenizer_config.json"), "utf8"),
        });
        return;
      }
      if (data.type === "load") {
        finish(data.ok ? { ok: true } : { ok: false, why: `load failed: ${data.error}` });
      }
    });
    worker.on("error", (error) => finish({ ok: false, why: `threw: ${error.message}` }));

    // Exactly what app.js does the instant the Worker is constructed. Posting immediately is also
    // the real race: engine-worker.js only installs its handler after a top-level `await import`,
    // so this message must survive being sent before anyone is listening.
    worker.postMessage({ type: "init" });
  });
}

const cases = [];
for (const [name, userAgent] of Object.entries(UA)) {
  for (const isolated of [true, false]) {
    cases.push({ label: `${name}/${isolated ? "isolated" : "not-isolated"}`, userAgent, isolated, full: FULL });
  }
}

let failures = 0;
for (const testCase of cases) {
  const result = await runCase(testCase);
  const detail = result.ok
    ? `threads=${result.threads ?? "-"} route=${result.route ?? "-"}`
    : result.why;
  console.log(`${result.ok ? "PASS" : "FAIL"}  ${result.label.padEnd(26)} ${detail}`);
  console.log(`      stages: ${result.stages.join(" -> ") || "(none)"}`);
  if (!result.ok) failures += 1;
}

console.log(failures ? `\n${failures}/${cases.length} cases failed` : `\nall ${cases.length} cases passed`);
process.exit(failures ? 1 : 0);
