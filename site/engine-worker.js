// The engine lives in a Worker: hydration takes seconds and synthesis tens of seconds
// at wasm speed, and none of that may block the UI thread. Protocol: postMessage
// {type, ...}; replies mirror the request type with `ok` or `error`.

import init, {
  WasmEngine,
  ModelStaging,
  presets,
  preset_vector,
  publish_team_block,
  arm_worker_team,
  worker_team_width,
  int8_route,
} from "./pkg/ftts_wasm.js?v=@SITEV@";

// Slice size for streaming OPFS into wasm. Big enough that per-call overhead is noise, small
// enough that the JS heap never holds a meaningful fraction of the model.
const INGEST_SLICE = 16 * 1024 * 1024;

let engine = null;

/// A shared memory, or null when this browser cannot give us one.
///
/// Threads are all-or-nothing here and the fallback is silent by design: iOS Safari, any page
/// served without COOP/COEP, and any engine without `SharedArrayBuffer` land on the serial path
/// from the same bytes rather than failing to start. The module is built with atomics either way —
/// unshared memory runs it fine as long as nothing ever parks on the team.
function createSharedMemory() {
  if (typeof SharedArrayBuffer === "undefined" || !self.crossOriginIsolated) return null;
  try {
    // 4 GB ceiling matches --max-memory in the build; the initial is deliberately small because
    // the artifact is streamed in later and growth is cheap where it is supported at all.
    return new WebAssembly.Memory({ initial: 512, maximum: 65536, shared: true });
  } catch {
    return null;
  }
}

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

self.onmessage = async ({ data }) => {
  try {
    switch (data.type) {
      case "init": {
        const wasmUrl = new URL("./pkg/ftts_wasm_bg.wasm?v=@SITEV@", import.meta.url);
        // Compile once and keep the module: kernel Workers must instantiate THE SAME module
        // against THE SAME memory, or they get their own linear memory and the team's shared
        // control block means nothing to them.
        const module = await WebAssembly.compileStreaming(fetch(wasmUrl));
        const memory = createSharedMemory();
        await init({ module_or_path: module, memory: memory ?? undefined });
        const threads = memory ? await startTeam(module, memory) : 1;
        reply("init", {
          ok: true,
          presets: JSON.parse(presets()),
          threads,
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
        const staging = new ModelStaging(data.fttsq.bytes, data.codec.bytes);
        for (const [meta, push] of [
          [data.fttsq, (chunk) => staging.push_fttsq(chunk)],
          [data.codec, (chunk) => staging.push_codec(chunk)],
        ]) {
          const blob = await (await root.getFileHandle(meta.asset)).getFile();
          for (let offset = 0; offset < blob.size; offset += INGEST_SLICE) {
            const end = Math.min(offset + INGEST_SLICE, blob.size);
            // One slice live at a time; the previous is collectable before the next is read.
            push(new Uint8Array(await blob.slice(offset, end).arrayBuffer()));
            reply("loadProgress", { bytesDone: staging.filled() });
          }
        }
        engine = WasmEngine.from_staging(
          staging,
          data.vocab,
          data.merges,
          data.tokenizerConfig,
        );
        reply("load", { ok: true });
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
};
