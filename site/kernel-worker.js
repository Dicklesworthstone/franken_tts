// One int8 partition worker.
//
// These are not task runners. Each instantiates the SAME compiled module against the SAME shared
// `WebAssembly.Memory` as the engine Worker, then parks inside Rust on the team's condvar for the
// life of the page — no postMessage in the hot path, no per-dispatch handshake. Work arrives as a
// generation bump on shared memory and is picked up by `atomic.wait` waking.
//
// That is also why this is a Worker rather than anything on the main thread: `atomic.wait` traps
// on a browser's main thread. A parked partition there would abort the page.

// NAMED import, and the distinction is load-bearing rather than stylistic. wasm-bindgen's web
// target exports `initSync` by name and the ASYNC `__wbg_init` as the default, so
// `import initSync from ...` silently binds the async one. It then takes `module_or_path`, not
// `module`, so it received undefined, went looking for a URL to fetch, and returned an unawaited
// promise — after which `worker_loop_entry` ran against uninitialized wasm and threw. Every
// partition reported `ready: false`, the team armed at width 1, and the engine ran serially while
// still paying to spawn the Workers. That is why threads measured as no faster, or slower.
import { initSync, worker_loop_entry } from "./pkg/ftts_wasm.js?v=@SITEV@";

self.onmessage = ({ data }) => {
  try {
    // Synchronous init on purpose: the engine has already compiled the module and created the
    // memory, so there is nothing to await, and staying synchronous means the worker is parked in
    // Rust before the engine's first dispatch can look for it.
    initSync({ module: data.module, memory: data.memory });
    self.postMessage({ ready: true, index: data.index });
    // Never returns. The team owns this thread until the page goes away.
    worker_loop_entry(data.index);
  } catch (error) {
    // A partition that failed to start must say so: the engine sizes the team from the count of
    // workers that reported ready, so a silent failure would leave it waiting on a partition that
    // will never report done.
    self.postMessage({ ready: false, index: data.index, error: String(error) });
  }
};
