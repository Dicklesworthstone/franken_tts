// Node stand-in for a browser kernel Worker: same globals shim, then the REAL kernel-worker.js.
//
// Loading the real file matters. The bug that armed the team at width 1 for weeks was a
// default-vs-named import inside kernel-worker.js — invisible to any test that re-implemented it.

import { parentPort, workerData } from "node:worker_threads";
import path from "node:path";
import { pathToFileURL } from "node:url";

globalThis.self = globalThis;

let messageHandler = null;
Object.defineProperty(globalThis, "onmessage", {
  get: () => messageHandler,
  set: (fn) => {
    messageHandler = fn;
    // The engine posts exactly once, before this module finishes evaluating, so replay the
    // payload the host already captured rather than waiting for a message that has been and gone.
    queueMicrotask(() => fn({ data: workerData }));
  },
  configurable: true,
});

globalThis.postMessage = (message) => parentPort.postMessage(message);

await import(pathToFileURL(path.join(workerData.siteDir, "kernel-worker.js")).href);
