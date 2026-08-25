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
  // The engine token, not just the browser token: `navigator.maxTouchPoints` does not
  // exist on WorkerNavigator, so the "Macintosh + touch" iPad test above is dead here, and
  // an iPadOS browser in desktop mode without a `Safari` UA token (Firefox) would slip
  // through both checks onto the exact WebKit shared-memory-growth crash this allow-list
  // exists to prevent. Any WebKit engine that is not really Blink is treated as unsafe.
  const isWebKitEngine = /AppleWebKit/.test(ua) && !/Chrome|Chromium|Edg\//.test(ua);
  return !isApple && !isSafari && !isWebKitEngine;
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
const INGEST_SLICE = 8 * 1024 * 1024;

let engine = null;
// The in-flight load's staging handle, held here so the error path can free its wasm-side
// buffers (~2 GB) immediately instead of waiting on the FinalizationRegistry.
let loadStaging = null;
// Non-null once any armed kernel Worker has died (its partition can never report done, so
// every future team dispatch would hang). Set from the Worker's onerror; checked before
// any message that could dispatch. The value is the reason shown to the user.
let teamDead = null;

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

  // All-or-nothing, and the reason is stripe arithmetic, not tidiness. A worker computes its
  // partition stripe FROM ITS SPAWN INDEX, and every parked worker decrements the join counter
  // on every dispatch. Arm width 5 with worker 2 dead and stripe 2 is silently never written —
  // the join still completes and the output carries a hole. Arm a smaller width with orphaned
  // parked workers and the counter is decremented more times than it was set, which underflows.
  // A tail failure would happen to stay contiguous, but distinguishing that from a middle
  // failure buys little: boot failures are all-or-nothing in practice, so any failure means
  // terminate the whole set and run serial, which is always correct.
  const ready = parked.filter(Boolean).length;
  if (ready === desired - 1) {
    // Post-arm liveness: `panic = "abort"` means a trap in a parked kernel Worker kills
    // ONLY that Worker; the join counter is never decremented and the next dispatch parks
    // this engine Worker forever inside `memory.atomic.wait` — where no JS can run. So the
    // rescue cannot live here: mark the team dead so every message AFTER the wedged one
    // fails fast with a reason, and the page-side watchdog (app.js) converts the wedged
    // in-flight call itself into an error instead of an eternal spinner.
    for (const worker of workers) {
      worker.onerror = (event) => {
        teamDead = event?.message
          ? `a kernel worker crashed: ${event.message}`
          : "a kernel worker crashed";
        stage("team-dead", teamDead);
      };
    }
    arm_worker_team(desired);
    return worker_team_width();
  }
  for (const worker of workers) worker.terminate();
  arm_worker_team(1);
  return 1;
}

function reply(type, payload, transfer = []) {
  self.postMessage({ type, ...payload }, transfer);
}

// Mirror this worker's console.error to the page.
//
// A Worker's console is not the page's console, and Playwright exposes no way to subscribe to it,
// so ANYTHING logged in here — including the Rust panic hook, which is the one thing that explains
// a dead engine — is invisible to the harness. That blindness has now cost two separate
// diagnoses: a fatal worker error read as an innocent stall, and a wasm trap that produced no
// output at all. Forwarding costs one postMessage per error and makes the panic hook worth having.
{
  const passthrough = console.error.bind(console);
  console.error = (...args) => {
    passthrough(...args);
    try {
      self.postMessage({ type: "workerLog", text: args.map(String).join(" ") });
    } catch {
      /* an unpostable argument must never mask the error being reported */
    }
  };
}

/// Messages are handled STRICTLY ONE AT A TIME, in arrival order.
///
/// `self.onmessage` being `async` means the runtime happily starts a second handler while the
/// first is still awaiting. app.js posts `load` as soon as it finds a complete cache, which races
/// `init` — and hydration then runs against a module whose glue has not finished binding. The
/// symptoms were maddening and platform-dependent: iOS threw
/// `undefined is not an object (evaluating 'wasm.modelstaging_new')`, while Chrome simply hung
/// A plan for streaming ONLY the codec's decoder into wasm, rebuilt as a valid safetensors file.
///
/// # Why the whole file is the wrong thing to stage
///
/// `CodecCheckpoint` is decoder-only — every tensor it reads is named `decoder.*`, with no
/// exceptions — but the file is 0.68 GB of which 0.225 GB is the encoder, used only during voice
/// enrollment. Staging all of it allocated 0.68 GB of wasm linear memory to build an object that
/// measured ~38 MB (committed memory right after widening was 0.72 GB, and wasm memory can only
/// grow, so source + widened peaked there).
///
/// That buffer is then freed — and wasm can never return memory to the OS, so it becomes a
/// permanent 0.68 GB hole. Worse, it lands 7 MB short of the 0.69 GB artifact reservation, so the
/// artifact cannot reuse it and grows on top instead. Roughly two thirds of the page's committed
/// memory was dead space, which is why an iPhone died during streaming while carrying far less
/// live data than a desktop that survived.
///
/// So the header is rewritten here to describe the decoder alone, and only those payload bytes are
/// streamed. Nothing is dropped that synthesis reads, nothing is reordered within a tensor, and
/// the engine parses an ordinary safetensors file that happens to be smaller.
async function planCodecDecoderOnly(root, asset) {
  const handle = await root.getFileHandle(asset);
  const sync = handle.createSyncAccessHandle ? await handle.createSyncAccessHandle() : null;
  const at = async (offset, length) => {
    const buffer = new Uint8Array(length);
    if (sync) {
      sync.read(buffer, { at: offset });
      return buffer;
    }
    const file = await handle.getFile();
    return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
  };
  try {
    const lengthBytes = await at(0, 8);
    const headerLength = Number(
      new DataView(lengthBytes.buffer, lengthBytes.byteOffset, 8).getBigUint64(0, true),
    );
    const header = JSON.parse(new TextDecoder().decode(await at(8, headerLength)));
    const payloadBase = 8 + headerLength;

    // Sorted by source offset so the reads stay sequential across a 0.68 GB file.
    const kept = Object.entries(header)
      .filter(([name, entry]) => name.startsWith("decoder.") && entry?.data_offsets)
      .sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
    if (kept.length === 0) throw new Error("codec file declares no decoder tensors");

    // Reassign offsets contiguously in the order the bytes will be pushed, so the rewritten
    // header describes exactly the stream the engine is about to receive.
    // Just where each tensor lives. Nothing is rewritten into a new container: the engine takes
    // tensors by name now, so the header this used to synthesize was a step that existed only to
    // be parsed straight back apart.
    const tensors = [];
    let cursor = 0;
    for (const [name, entry] of kept) {
      const [start, end] = entry.data_offsets;
      const length = end - start;
      if (entry.dtype !== "F32") {
        throw new Error(`codec tensor ${name} is ${entry.dtype}; the engine widens f32 only`);
      }
      tensors.push({ name, at: payloadBase + start, length });
      cursor += length;
    }

    return { tensors, totalBytes: cursor, asset };
  } finally {
    sync?.close();
  }
}

/// Where the cold text embedding lives inside the artifact, read from the artifact itself.
///
/// The `.fttsq` header is `magic(8) | version(4) | directoryLength(8) | directoryJSON`, all at the
/// head of the file, so this costs one small read regardless of how large the artifact is.
///
/// Nothing here is hardcoded on purpose. The byte offset where staging stops and the offset each
/// embedding row is read from both come from the directory the artifact declares about itself, so
/// a re-exported model with a different layout cannot silently desynchronize the two sides — and
/// the engine independently re-derives the same prefix length at hydration and refuses a mismatch.
let coldLayout = null;

async function readColdLayout(root, asset) {
  const handle = await root.getFileHandle(asset);
  const sync = handle.createSyncAccessHandle ? await handle.createSyncAccessHandle() : null;
  const at = async (offset, length) => {
    const buffer = new Uint8Array(length);
    if (sync) {
      sync.read(buffer, { at: offset });
      return buffer;
    }
    const file = await handle.getFile();
    return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
  };
  let head;
  let directory;
  try {
    head = await at(0, 20);
    if (new TextDecoder().decode(head.subarray(0, 5)) !== "FTTSQ") {
      throw new Error("not a .fttsq artifact");
    }
    // getBigUint64 because the directory length is a u64; Number() is safe afterwards because a
    // directory that exceeded 2^53 bytes would not fit in the file it describes.
    const directoryLength = Number(
      new DataView(head.buffer, head.byteOffset, head.byteLength).getBigUint64(12, true),
    );
    directory = JSON.parse(new TextDecoder().decode(await at(20, directoryLength)));
  } finally {
    sync?.close();
  }

  const cold = directory.sections.find((s) => s.access_class === "COLD_TEXT_EMBEDDING");
  if (!cold) throw new Error("artifact declares no COLD_TEXT_EMBEDDING section");
  // The saving depends on the cold section being last: everything before it keeps its true file
  // offset, so truncating needs no remapping at all. Refuse rather than assume.
  const later = directory.sections.find(
    (s) => s.access_class !== "COLD_TEXT_EMBEDDING" && s.offset >= cold.offset,
  );
  if (later) throw new Error(`section ${later.name} sits after the cold section; cannot elide`);

  const tensor = directory.tensors.find((t) => t.name === "talker.model.text_embedding.weight");
  if (!tensor || tensor.dtype !== "bf16") {
    throw new Error(`cold text embedding must be bf16, found ${tensor?.dtype}`);
  }
  const width = tensor.shape[1];
  return {
    hotBytes: cold.offset,
    // Rows are read straight out of the file, so the offset is the section's plus the tensor's
    // offset within it plus the row stride.
    rowBase: cold.offset + tensor.offset,
    rowBytes: width * 2,
    asset,
  };
}

/// Read one bf16 embedding row per id, concatenated in the order given.
///
/// `ids` arrives from `engine.text_row_ids`, which returns them ascending and distinct — the exact
/// order the Rust side reconstructs independently when it rebuilds the table.
async function readColdRows(root, layout, ids) {
  const handle = await root.getFileHandle(layout.asset);
  const out = new Uint8Array(ids.length * layout.rowBytes);
  // Sync access handle for the same reason as staging: this runs once per utterance and would
  // otherwise re-materialize a File over a 1.3 GB entry every time somebody presses speak.
  const sync = handle.createSyncAccessHandle ? await handle.createSyncAccessHandle() : null;
  try {
    for (let i = 0; i < ids.length; i += 1) {
      const at = layout.rowBase + ids[i] * layout.rowBytes;
      // Read straight into the destination slice; no intermediate buffer, no copy.
      const view = out.subarray(i * layout.rowBytes, (i + 1) * layout.rowBytes);
      const read = sync
        ? sync.read(view, { at })
        : await (async () => {
            const file = await handle.getFile();
            const bytes = new Uint8Array(await file.slice(at, at + layout.rowBytes).arrayBuffer());
            view.set(bytes);
            return bytes.length;
          })();
      if (read !== layout.rowBytes) {
        throw new Error(`short cold row for id ${ids[i]}: ${read}/${layout.rowBytes}`);
      }
    }
  } finally {
    sync?.close();
  }
  return out;
}

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
          requestId: data.requestId,
        });
        break;
      }
      case "load": {
        // A superseded engine's wasm-side buffers (~2 GB) must not wait for the
        // FinalizationRegistry: a second load alongside the orphan doubles peak memory,
        // which is exactly the reclaim threshold on iOS.
        if (engine) {
          const superseded = engine;
          engine = null;
          try {
            superseded.free();
          } catch {
            /* already freed */
          }
        }
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
        // Freed explicitly on ANY failure below: the handle owns up to ~2 GB of wasm-side
        // staging buffers, and leaving it to the FinalizationRegistry means the retry the
        // error message invites allocates a second 2 GB alongside the orphan → tab death.
        // (`from_staging` consumes the handle on success, making the later free a no-op.)
        const codecPlan = await planCodecDecoderOnly(root, data.codec.asset);
        const staging = new ModelStaging();
        loadStaging = staging;
        // What will actually be ingested: the decoder-only codec, plus the artifact's hot prefix
        // once the layout is known. Before that it is the codec alone, which is all that streamed.
        const stagedTotal = () => codecPlan.totalBytes + (coldLayout?.hotBytes ?? 0);
        // Read through ONE reused buffer, never through Blob.
        //
        // The previous shape — `getFileHandle().getFile()` then `blob.slice(a, b).arrayBuffer()`
        // per chunk — asks the browser for a fresh 16 MB ArrayBuffer on every iteration, on top of
        // whatever the File object itself materializes for a 1.3 GB OPFS entry. On a phone that is
        // allocation pressure layered directly on the largest allocation the page ever makes, and
        // it is where an iPhone died: during streaming, at roughly 1.4 GB, comfortably BELOW the
        // 1.86 GB peak the same build reaches on a desktop. Dying under your own high-water mark
        // is the signature of the reader, not the totals.
        //
        // `createSyncAccessHandle` reads straight into a buffer we own and reuse, so the steady
        // state is one 8 MB staging buffer regardless of file size and the GC has nothing to
        // chase. It is worker-only, which is fine because this IS the worker — and it is what the
        // browser-side LLM runtimes settled on for the same reason.
        const scratch = new Uint8Array(INGEST_SLICE);
        const drain = async (meta, push, limit) => {
          const handle = await root.getFileHandle(meta.asset);
          const sync = handle.createSyncAccessHandle
            ? await handle.createSyncAccessHandle()
            : null;
          try {
            const size = sync ? sync.getSize() : (await handle.getFile()).size;
            const stop = Math.min(limit ?? size, size);
            for (let offset = 0; offset < stop; offset += INGEST_SLICE) {
              const want = Math.min(INGEST_SLICE, stop - offset);
              if (sync) {
                // subarray is a VIEW, not a copy: nothing is allocated per chunk.
                const view = scratch.subarray(0, want);
                const read = sync.read(view, { at: offset });
                if (read !== want) throw new Error(`short read at ${offset}: ${read}/${want}`);
                push(view);
              } else {
                // Fallback for engines without sync access handles. Same bytes, more garbage.
                const file = await handle.getFile();
                push(new Uint8Array(await file.slice(offset, offset + want).arrayBuffer()));
              }
              // The total is reported alongside, because staging no longer ingests the whole
              // artifact: the elided cold section would otherwise leave the bar stalled at ~85%
              // with nothing wrong. Memory rides along so a crash breadcrumb records the size the
              // tab actually died at, which is the number that decides what to cut next.
              reply("loadProgress", {
                bytesDone: staging.filled(),
                bytesTotal: stagedTotal(),
                wasmBytes: memoryBytes(),
              });
            }
          } finally {
            // The handle holds an EXCLUSIVE lock on the file; leaving it open makes every later
            // read — including the cold embedding rows — fail for the life of the page.
            sync?.close();
          }
        };

        // Hand the codec over one tensor at a time, widening each on arrival.
        //
        // A tensor is read whole because the engine widens whole tensors; the largest is ~75 MB,
        // so the transient is that rather than the 0.46 GB the staged file used to cost. Nothing
        // accumulates on this side: each buffer is dropped as soon as wasm has copied it.
        const pushCodecTensors = async (asset, tensors, staging) => {
          const handle = await root.getFileHandle(asset);
          const sync = handle.createSyncAccessHandle
            ? await handle.createSyncAccessHandle()
            : null;
          // ONE buffer, sized to the largest tensor, reused for all 271 of them.
          //
          // Allocating per tensor asked the browser for 271 fresh buffers totalling 0.46 GB, the
          // largest 75.5 MB, right at the page's high-water mark. That is the same per-chunk
          // allocation pattern already removed from the artifact drain, reintroduced here — and
          // it is what put a phone back over its limit after the peak had come DOWN. The wasm
          // side copies out of this synchronously, so one buffer is safe to reuse.
          const widest = tensors.reduce((most, t) => Math.max(most, t.length), 0);
          let pushedSinceReport = 0;
          const scratchTensor = sync ? new Uint8Array(widest) : null;
          try {
            for (const tensor of tensors) {
              let bytes;
              if (sync) {
                // A VIEW, not a copy: nothing is allocated per tensor.
                bytes = scratchTensor.subarray(0, tensor.length);
                const read = sync.read(bytes, { at: tensor.at });
                if (read !== tensor.length) {
                  throw new Error(`short codec tensor ${tensor.name}: ${read}/${tensor.length}`);
                }
              } else {
                const file = await handle.getFile();
                bytes = new Uint8Array(
                  await file.slice(tensor.at, tensor.at + tensor.length).arrayBuffer(),
                );
              }
              staging.push_codec_tensor(tensor.name, bytes);
              // Report on a byte cadence rather than per tensor: 271 postMessages, each
              // allocating a structured-clone payload, is churn of exactly the kind this loop
              // was just cleaned of.
              pushedSinceReport += tensor.length;
              if (pushedSinceReport >= INGEST_SLICE) {
                pushedSinceReport = 0;
                reply("loadProgress", {
                  bytesDone: staging.filled(),
                  bytesTotal: stagedTotal(),
                  wasmBytes: memoryBytes(),
                });
              }
            }
          } finally {
            sync?.close();
          }
        };

        // ARTIFACT FIRST, codec second — the reverse of the original order, and worth ~0.45 GB.
        //
        // wasm memory only ever grows, so a freed buffer is not returned to anyone; it is a hole,
        // and a hole is only useful if a later allocation can land in it. The codec's staged bytes
        // are freed the instant the checkpoint is built, and that checkpoint is ~0.04 GB against
        // ~0.46 GB of source — so the codec's buffer is the hole, and it wants to be the LAST big
        // claim, where the talker's 0.34 GB of widened tensors can reuse it.
        //
        // Codec-first committed that hole before the artifact ever grew, and the artifact (0.69 GB)
        // could not fit in it, so the page paid for both and streamed the artifact while already
        // carrying ~1.19 GB. That is the phase an iPhone died in, which is the whole reason the
        // order changed.
        //
        // The prefix leaves the cold text embedding on disk: 622 MB, 47% of the artifact, read a
        // few hundred 4 KB rows at a time. A native host pays only for the rows it touches because
        // it maps the file; wasm cannot map and its heap is grow-only, so every staged byte is
        // resident for the life of the page.
        //
        // Both of these run on EVERY browser rather than only the small ones. A path that runs
        // exclusively on phones is a path nothing tests, and this session already paid for that
        // lesson when a Chromium-only harness hid a WebKit-only failure (NE-007).
        coldLayout = await readColdLayout(root, data.fttsq.asset);
        stage("reserve-artifact", `${(coldLayout.hotBytes / 1e9).toFixed(2)} GB hot prefix`);
        // The full length rides along with the prefix length: the directory's declared ranges are
        // validated against the REAL artifact, since the section left behind necessarily runs past
        // what is staged. Passing the prefix as if it were the whole file is what made the first
        // attempt reject its own artifact.
        staging.reserve_fttsq_hot_prefix(coldLayout.hotBytes, BigInt(data.fttsq.bytes));
        stage("stream-artifact");
        await drain(data.fttsq, (chunk) => staging.push_fttsq(chunk), coldLayout.hotBytes);

        // Hydrate the talker and RELEASE the artifact before the codec is staged at all.
        //
        // This is the ordering the whole memory story turns on. Fusing the int8 tables makes the
        // artifact's 0.69 GB hot prefix dead, and wasm memory only grows — so the hole that
        // release leaves is worth whatever gets allocated next. Staging the codec after it means
        // the codec's source and checkpoint land inside that hole. The other way round, they land
        // on top and the peak was 2.10 GB.
        stage("hydrate-talker", `mem ${(memoryBytes() / 1e9).toFixed(2)} GB`);
        staging.finish_artifact();

        stage(
          "stream-codec",
          `${(codecPlan.totalBytes / 1e9).toFixed(2)} GB decoder of ${(data.codec.bytes / 1e9).toFixed(2)} GB`,
        );
        await pushCodecTensors(data.codec.asset, codecPlan.tensors, staging);
        stage("widen-codec", `mem ${(memoryBytes() / 1e9).toFixed(2)} GB`);
        staging.finish_codec();

        stage("assemble-engine", `mem ${(memoryBytes() / 1e9).toFixed(2)} GB`);
        engine = WasmEngine.from_staging(
          staging,
          data.vocab,
          data.merges,
          data.tokenizerConfig,
        );
        loadStaging = null; // consumed by from_staging
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
        reply("load", { ok: true, threads, requestId: data.requestId });
        break;
      }
      case "synthesize": {
        if (!engine) throw new Error("engine not loaded");
        if (teamDead) {
          throw new Error(`${teamDead}; reload the page to rebuild the worker team`);
        }
        const voice =
          data.voiceVector ?? Float32Array.from(preset_vector(data.voiceName ?? "matt"));
        // The cold embedding rows this utterance needs, read on demand from OPFS. The ids come
        // from the engine's own tokenizer rather than from anything here: the two sides must agree
        // exactly, and asking the tokenizer that will actually run is the only way to be sure.
        const rowIds = engine.text_row_ids(data.text);
        const rows = await readColdRows(
          await navigator.storage.getDirectory(),
          coldLayout,
          rowIds,
        );
        // Synthesis has its own high-water mark, separate from hydration's, and it is now the one
        // that matters: a phone that reaches "ready to speak" and dies on the first press is being
        // killed by what THIS allocates, not by the model. Recorded before and after so a crash
        // leaves the before-number behind, and the pair says how much synthesis costs.
        const memBefore = memoryBytes();
        stage("synthesize", `mem ${(memBefore / 1e9).toFixed(2)} GB`);
        const started = performance.now();
        // DISC-006 seam taps (frankentts-p16p): per-request, off unless the page asked.
        if (typeof engine.set_debug_taps === "function") {
          engine.set_debug_taps(Boolean(data.debugTaps));
        }
        // Packet size is a pure speed/memory dial: the streaming==batch gate makes output
        // bit-identical under every schedule, so 0 (the engine's default of 4) and any override
        // produce the same samples. Bigger packets mean a larger `m` in every codec GEMM.
        const pcm = engine.synthesize_with_text_rows(
          data.text,
          voice,
          BigInt(data.seed ?? 0),
          0,
          rows,
          data.packetFrames ?? 0,
        );
        const elapsedMs = performance.now() - started;
        stage(
          "synthesized",
          `mem ${(memoryBytes() / 1e9).toFixed(2)} GB (+${((memoryBytes() - memBefore) / 1e9).toFixed(2)} GB)`,
        );
        reply(
          "synthesize",
          { ok: true, pcm: pcm.buffer, sampleRate: 24000, elapsedMs, requestId: data.requestId },
          [pcm.buffer],
        );
        break;
      }
      case "enroll": {
        if (!engine) throw new Error("engine not loaded");
        if (teamDead) {
          throw new Error(`${teamDead}; reload the page to rebuild the worker team`);
        }
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
    // Free an abandoned load's staging buffers before inviting a retry (see `loadStaging`).
    if (loadStaging) {
      try {
        loadStaging.free();
      } catch {
        /* already consumed */
      }
      loadStaging = null;
    }
    reply(data.type, { ok: false, error: String(error), requestId: data.requestId });
  }
}
