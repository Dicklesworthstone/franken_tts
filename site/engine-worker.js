// The engine lives in a Worker: hydration takes seconds and synthesis tens of seconds
// at wasm speed, and none of that may block the UI thread. Protocol: postMessage
// {type, ...}; replies mirror the request type with `ok` or `error`.

/// Whether this engine may use the THREADED build, whose memory is shared and imported.
///
/// Threads are opt-IN rather than opt-out, and the default is the safe one, because getting this
/// wrong does not merely slow the page down — it kills the tab.
///
/// Measured on an iPhone 17 Pro Max: growing a SHARED wasm memory past ~1 GB reclaims the tab,
/// while growing an UNSHARED one to 2.75 GB is fine and flat allocations of either kind are fine
/// to 3.5 GB. Rust's allocator grows linear memory on every heap request, so a 2 GB model
/// guarantees growth, so a shared memory guarantees the crash on that device.
///
/// This cannot be feature-detected: the only test IS the crash. So it is a capability allow-list —
/// browsers known to grow shared memory safely get threads, everything else gets the serial build
/// and still works. Since the team only covers the ~8% of frame time that is talker+microdecoder
/// (the codec is 92% and single-threaded), the cost of guessing "serial" wrongly is small, and the
/// cost of guessing "threaded" wrongly is a dead page.
function threadsAreSafeHere() {
  if (typeof SharedArrayBuffer === "undefined" || !self.crossOriginIsolated) return false;
  const ua = navigator.userAgent;
  // Every iOS browser is WebKit underneath, so Chrome/Firefox on iOS inherit the same behavior.
  const isApple = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const isSafari = /Safari/.test(ua) && !/Chrome|Chromium|Edg\//.test(ua);
  return !isApple && !isSafari;
}

const THREADED = threadsAreSafeHere();

/// The package directory for this engine, used for BOTH the glue and the binary.
///
/// One constant, deliberately: loading serial glue against the threaded binary instantiates a
/// module that imports a shared memory without one being supplied. Instantiation fails, the glue's
/// `wasm` binding is never assigned, and the first call surfaces as
/// `undefined is not an object (evaluating 'wasm.modelstaging_new')` — a confusing error a long
/// way from its cause. The two must never be able to disagree.
const PKG_DIR = THREADED ? "./pkg" : "./pkg-serial";
const PKG = `${PKG_DIR}/ftts_wasm.js?v=@SITEV@`;

// ── the inbox MUST be installed before the first `await` in this module ───────────────────────
//
// A module worker starts running its event loop while the module body is still evaluating. A
// `message` event that arrives before `self.onmessage` exists is dispatched to nothing and is GONE
// — it is not queued for a listener attached later. The dynamic `import` below is an await, so any
// message posted during it lands in exactly that window.
//
// app.js posts `init` the instant it constructs the Worker, which is inside the window; `load`
// arrives seconds later, after the handler exists, and survives. That asymmetry is the whole bug:
// the engine only ever saw `load`, so the wasm glue was never bound, and `new ModelStaging(...)`
// then hung on Chrome and threw `undefined is not an object (evaluating 'wasm.modelstaging_new')`
// on iOS. One lost message, two unrecognizable symptoms, and a green Node test suite that never
// loaded this file.
//
// So the handler goes in FIRST and buffers. Nothing here touches wasm; it only remembers.
const inbox = [];
let deliver = (event) => inbox.push(event);
self.onmessage = (event) => deliver(event);

const {
  default: init,
  WasmEngine,
  ModelStaging,
  presets,
  preset_vector,
  publish_team_block,
  arm_worker_team,
  worker_team_width,
  int8_route,
} = await import(PKG);

// Slice size for streaming OPFS into wasm. Big enough that per-call overhead is noise, small
// enough that the JS heap never holds a meaningful fraction of the model.
const INGEST_SLICE = 16 * 1024 * 1024;

let engine = null;

/// A shared memory for the threaded build, or null when this engine runs the serial one.
///
/// # Two things that were tried here and do NOT work
///
/// **Pre-reserving the whole model up front** (`initial` = 2.75 GB) so growth never happens: Rust's
/// `dlmalloc` calls `memory.grow` for every heap request and never reuses space it did not itself
/// request, so the pre-reserved region below the heap is simply skipped and it grows on top anyway.
///
/// **Pinning `maximum = initial`** to forbid growth outright: the first allocation then fails,
/// `dlmalloc` returns null, and the module aborts with `unreachable` — verified in Node, not
/// theorized. That briefly shipped, and it made the crash worse rather than better.
///
/// So growth is unavoidable, and growth of a SHARED memory is what kills iOS. The fix is not here;
/// it is [`threadsAreSafeHere`], which routes those devices to the unshared serial build entirely.
/// This function only ever runs when threads are already known to be safe.
function createSharedMemory() {
  if (!THREADED) return null;
  try {
    // The maximum matches --max-memory in the build. Growth within it is the normal path.
    return new WebAssembly.Memory({ initial: 512, maximum: 65536, shared: true });
  } catch {
    return null;
  }
}

/// Announces the stage about to be entered, for crash forensics on the page side.
///
/// Posted BEFORE the work starts, never after: the entire point is that this survives a stage
/// that does not.
function stage(name, detail) {
  self.postMessage({ type: "stage", stage: name, detail });
}

/// Bytes of wasm linear memory currently committed, for stage telemetry.
function memoryBytes() {
  try {
    return wasmMemory?.buffer.byteLength ?? 0;
  } catch {
    return 0;
  }
}

let wasmMemory = null;

/// The module+memory captured at init, so the team can be armed AFTER hydration instead of before.
/// See the note in the "init" case: arming first deadlocks shared-memory growth.
let teamPending = null;

/// Spawns the kernel Workers and arms the team with however many actually parked.
///
/// The order matters and is the whole reason this is not one call: the control block is published
/// first so a Worker can park before a team exists, and the team is sized afterwards from the
/// count that confirmed. Sizing it up front would make one Worker that failed to boot into a
/// dispatcher that waits forever for a partition that will never report done.
async function startTeam(module, memory) {
  // Leave a core for the UI thread and the codec worker; more partitions than cores is pure
  // contention on a memory-bound kernel.
  const desired = Math.max(1, Math.min((navigator.hardwareConcurrency || 2) - 1, 6));
  if (desired <= 1) return 1;
  publish_team_block();

  const workers = [];
  const parked = await Promise.all(
    Array.from({ length: desired - 1 }, (_, slot) => {
      const index = slot + 1;
      const worker = new Worker(new URL("./kernel-worker.js?v=@SITEV@", import.meta.url), {
        type: "module",
      });
      workers.push(worker);
      return new Promise((resolve) => {
        // A Worker that never answers must not hang startup; it costs its partition instead.
        const timer = setTimeout(() => resolve(false), 5000);
        worker.onmessage = ({ data }) => {
          clearTimeout(timer);
          resolve(Boolean(data.ready));
        };
        worker.onerror = () => {
          clearTimeout(timer);
          resolve(false);
        };
        worker.postMessage({ module, memory, index });
      });
    }),
  );

  const ready = parked.filter(Boolean).length;
  arm_worker_team(ready + 1);
  return worker_team_width();
}

function reply(type, payload, transfer = []) {
  self.postMessage({ type, ...payload }, transfer);
}

/// Messages are handled STRICTLY ONE AT A TIME, in arrival order.
///
/// `self.onmessage` being `async` means the runtime happily starts a second handler while the
/// first is still awaiting. app.js posts `load` as soon as it finds a complete cache, which races
/// `init` — and hydration then runs against a module whose glue has not finished binding. The
/// symptoms were maddening and platform-dependent: iOS threw
/// `undefined is not an object (evaluating 'wasm.modelstaging_new')`, while Chrome simply hung
/// forever inside `new ModelStaging(...)` with `wasmMemory` still null (reported as `mem 0.00 GB`,
/// the clue that finally identified this).
///
/// Chaining onto the previous handler's promise makes the ordering guarantee explicit rather than
/// depending on which message happens to win.
// Real delivery, now that the glue is bound: strictly one message at a time, in arrival order.
//
// Serialization matters independently of the lost-message bug. `handleMessage` is async, so the
// runtime would otherwise happily start `load` while `init` is still awaiting, and hydration would
// run against a half-initialized module.
//
// `.catch` is not optional: a rejected link poisons the chain and every later `.then` is skipped
// silently, turning one failed message into a worker that ignores all subsequent ones forever.
// `handleMessage` reports its own errors, so swallowing here only keeps the chain alive.
let queue = Promise.resolve();
deliver = (event) => {
  queue = queue.then(() => handleMessage(event)).catch(() => {});
};
// Drain whatever arrived while the module was still evaluating, in the order it arrived.
for (const event of inbox.splice(0)) deliver(event);

async function handleMessage({ data }) {
  // Announced before anything is dispatched, so the page can see WHICH messages the worker
  // actually received and in what order — the fact that no amount of reasoning could supply.
  stage(`msg:${data.type}`);
  try {
    switch (data.type) {
      case "init": {
        const wasmUrl = new URL(`${PKG_DIR}/ftts_wasm_bg.wasm?v=@SITEV@`, import.meta.url);
        // Compile once and keep the module: kernel Workers must instantiate THE SAME module
        // against THE SAME memory, or they get their own linear memory and the team's shared
        // control block means nothing to them.
        stage("compile-module");
        const module = await WebAssembly.compileStreaming(fetch(wasmUrl));
        stage("create-memory", THREADED ? "shared/threaded" : "unshared/serial");
        const memory = createSharedMemory();
        wasmMemory = memory;
        stage(
          "instantiate",
          `threaded=${THREADED} pkg=${PKG_DIR} memory=${memory ? "made" : "null"} sab=${
            typeof SharedArrayBuffer !== "undefined"
          } isolated=${self.crossOriginIsolated}`,
        );
        const exports = await init({ module_or_path: module, memory: memory ?? undefined });
        wasmMemory = exports?.memory ?? memory;
        stage(
          "instantiated",
          `exports=${typeof exports} exportsMemory=${exports?.memory ? "yes" : "no"} bytes=${
            wasmMemory?.buffer?.byteLength ?? -1
          }`,
        );
        // Arm ONLY if the memory the module actually instantiated against is shared. A build whose
        // memory is defined-and-exported rather than shared-and-imported silently ignores the
        // memory passed above, and every Worker would then get its own linear memory — the team's
        // control block would be a different object in each, the dispatcher would wait forever on
        // partitions that cannot report, and the page would hang instead of merely being slow.
        // Checking the instantiated buffer is the only honest test; the build flags are not.
        // The team is NOT armed here, and the ordering is the fix for a hard deadlock.
        //
        // Arming spawns kernel Workers that park inside Rust on `atomic.wait` for the life of the
        // page. Hydration then asks dlmalloc for ~2 GB, which grows linear memory. Growing a
        // SHARED memory has to coordinate every thread holding a view of it — and a thread parked
        // in `atomic.wait` never reaches that point, so the grow never completes. The page hung
        // forever inside `new ModelStaging(...)` with no error: reproduced in real Chromium, where
        // `staging-detail` printed and `staging-ok` never did.
        //
        // It only ever bit threaded browsers, which is why the serial iOS build was unaffected and
        // why every Node test missed it — those armed the team AFTER hydrating.
        //
        // So: all growth happens first, and the team is armed at the end of `load`, once the
        // model is resident and linear memory has stopped moving.
        teamPending = { module, memory: exports?.memory ?? memory };
        reply("init", {
          ok: true,
          presets: JSON.parse(presets()),
          threads: 1,
          route: int8_route(),
        });
        break;
      }
      case "load": {
        // The large files are streamed OUT of OPFS and straight INTO wasm linear memory. They are
        // never materialized as JS ArrayBuffers: doing that put a 1.3 GB artifact in memory twice
        // (once for JS, once for wasm-bindgen's copy) and is what reclaimed the tab on iOS.
        // The engine still re-verifies the artifact's own digests in wasm before reading a tensor.
        const root = await navigator.storage.getDirectory();

        // CODEC FIRST, and hydrated before the artifact is even reserved.
        //
        // The order is load-bearing, not stylistic. `CodecCheckpoint` widens every BF16 tensor
        // into owned f32, so hydrating it while the 1.31 GB artifact is also resident puts three
        // large allocations in flight at once (~3.35 GB) — past what iOS Safari grants a tab, and
        // the reason the page died the moment it reported "hydrating". Draining the codec's source
        // bytes first drops the high-water mark to ~2.67 GB with nothing read twice. See the
        // ModelStaging docs in crates/ftts-wasm/src/lib.rs for the arithmetic.
        stage("staging-new", `mem ${(memoryBytes() / 1e9).toFixed(2)} GB`);
        const staging = new ModelStaging(data.codec.bytes);
        const drain = async (meta, push) => {
          const blob = await (await root.getFileHandle(meta.asset)).getFile();
          for (let offset = 0; offset < blob.size; offset += INGEST_SLICE) {
            const end = Math.min(offset + INGEST_SLICE, blob.size);
            // One slice live at a time; the previous is collectable before the next is read.
            push(new Uint8Array(await blob.slice(offset, end).arrayBuffer()));
            reply("loadProgress", { bytesDone: staging.filled() });
          }
        };

        stage("stream-codec", `${(data.codec.bytes / 1e9).toFixed(2)} GB`);
        await drain(data.codec, (chunk) => staging.push_codec(chunk));
        stage("widen-codec", `mem ${(memoryBytes() / 1e9).toFixed(2)} GB`);
        staging.finish_codec();

        stage("reserve-artifact", `${(data.fttsq.bytes / 1e9).toFixed(2)} GB`);
        staging.reserve_fttsq(data.fttsq.bytes);
        stage("stream-artifact");
        await drain(data.fttsq, (chunk) => staging.push_fttsq(chunk));

        stage("hydrate-talker", `mem ${(memoryBytes() / 1e9).toFixed(2)} GB`);
        engine = WasmEngine.from_staging(
          staging,
          data.vocab,
          data.merges,
          data.tokenizerConfig,
        );
        // Now that linear memory has finished growing, it is safe to park worker threads on it.
        let threads = 1;
        if (teamPending) {
          const shared =
            typeof SharedArrayBuffer !== "undefined" &&
            teamPending.memory?.buffer instanceof SharedArrayBuffer;
          if (shared) {
            stage("arm-team");
            threads = await startTeam(teamPending.module, teamPending.memory);
          }
          teamPending = null;
        }
        stage("ready", `threads=${threads}`);
        reply("load", { ok: true, threads });
        break;
      }
      case "synthesize": {
        if (!engine) throw new Error("engine not loaded");
        const voice =
          data.voiceVector ?? Float32Array.from(preset_vector(data.voiceName ?? "matt"));
        const started = performance.now();
        const pcm = engine.synthesize(data.text, voice, BigInt(data.seed ?? 0), 0);
        const elapsedMs = performance.now() - started;
        reply(
          "synthesize",
          { ok: true, pcm: pcm.buffer, sampleRate: 24000, elapsedMs, requestId: data.requestId },
          [pcm.buffer],
        );
        break;
      }
      case "enroll": {
        if (!engine) throw new Error("engine not loaded");
        const vector = engine.enroll(new Float32Array(data.pcm));
        reply("enroll", { ok: true, vector: vector.buffer, requestId: data.requestId }, [
          vector.buffer,
        ]);
        break;
      }
      default:
        throw new Error(`unknown message type ${data.type}`);
    }
  } catch (error) {
    reply(data.type, { ok: false, error: String(error), requestId: data.requestId });
  }
}
