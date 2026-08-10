// Runs the REAL site/engine-worker.js inside a Node worker thread.
//
// Every earlier Node test imported the wasm glue directly and re-implemented the worker's logic in
// the test file. That design cannot find worker bugs by construction — it never loads the worker —
// and it is why a build-selection bug that bricked every browser got shipped with "tests passing".
// This harness inverts that: the worker source is imported UNMODIFIED, and only the browser
// globals it reaches for are shimmed.
//
// Shimmed, in order of how much they matter:
//   crossOriginIsolated  — decides which build engine-worker.js loads. Forceable, so BOTH paths
//                          are testable on a machine where isolation would otherwise always hold.
//   navigator.userAgent  — the other half of that decision. Forceable, so the iOS/Safari branch is
//                          testable without an iPhone.
//   navigator.storage    — an OPFS stand-in over real files on disk, slice-faithful so the
//                          streaming path is exercised for real rather than mocked out.
//   fetch/compileStreaming — file reads, since there is no server.
//   Worker               — kernel workers become Node worker threads against the same memory.

import { parentPort, workerData } from "node:worker_threads";
import { Worker as NodeWorker } from "node:worker_threads";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const { modelDir, siteDir, userAgent, isolated, assets } = workerData;

// ── globals the worker expects ───────────────────────────────────────────────────────────────
globalThis.self = globalThis;

// `self.onmessage = fn` must actually receive the host's messages.
let messageHandler = null;
Object.defineProperty(globalThis, "onmessage", {
  get: () => messageHandler,
  set: (fn) => {
    messageHandler = fn;
    parentPort.on("message", (data) => fn({ data }));
  },
  configurable: true,
});

globalThis.postMessage = (message) => parentPort.postMessage(message);

Object.defineProperty(globalThis, "crossOriginIsolated", {
  value: isolated,
  configurable: true,
});

// Node defines `navigator` as a getter-only global, so it must be redefined rather than assigned.
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  writable: true,
  value: {
  userAgent,
  hardwareConcurrency: 8,
  // Deliberately absent in real WorkerNavigator; kept undefined so the worker's Apple detection
  // is exercised exactly as it behaves in a browser worker.
  storage: {
    getDirectory: async () => ({
      getFileHandle: async (name) => {
        const file = assets[name];
        if (!file) throw new Error(`no such OPFS entry: ${name}`);
        const { size } = fs.statSync(file);
        return {
          getFile: async () => ({
            size,
            // Slice-faithful and lazy: reads only the requested window, so a 1.3 GB asset streams
            // exactly as it does from OPFS instead of being materialized up front.
            slice: (start, end) => ({
              arrayBuffer: async () => {
                const length = end - start;
                const buffer = Buffer.allocUnsafe(length);
                const fd = fs.openSync(file, "r");
                try {
                  fs.readSync(fd, buffer, 0, length, start);
                } finally {
                  fs.closeSync(fd);
                }
                return buffer.buffer.slice(
                  buffer.byteOffset,
                  buffer.byteOffset + buffer.byteLength,
                );
              },
            }),
          }),
        };
      },
    }),
  },
  },
});

// The worker fetches its own wasm; serve it off disk, ignoring the cache-buster query.
globalThis.fetch = async (input) => {
  const url = typeof input === "string" ? input : input.href ?? String(input);
  const file = fileURLToPath(new URL(url).href.split("?")[0]);
  return { ok: true, arrayBuffer: async () => fs.readFileSync(file).buffer };
};

globalThis.WebAssembly.compileStreaming = async (response) => {
  const resolved = await response;
  return new WebAssembly.Module(await resolved.arrayBuffer());
};

// Kernel workers: same module, same memory, as in the browser.
globalThis.Worker = class {
  constructor(url) {
    const file = fileURLToPath(new URL(String(url)).href.split("?")[0]);
    this.inner = null;
    this.file = file;
    this.listeners = {};
  }
  postMessage(data) {
    this.inner = new NodeWorker(
      path.join(siteDir, "harness", "kernel-host.mjs"),
      { workerData: { ...data, siteDir } },
    );
    this.inner.unref();
    this.inner.on("message", (m) => this.onmessage?.({ data: m }));
    this.inner.on("error", (e) => this.onerror?.(e));
  }
  addEventListener(type, fn) {
    if (type === "message") this.onmessage = (event) => fn(event);
    if (type === "error") this.onerror = fn;
  }
  terminate() {
    this.inner?.terminate();
  }
};

// Import the worker UNMODIFIED. Everything above exists so this line needs no changes.
await import(pathToFileURL(path.join(siteDir, "engine-worker.js")).href);
