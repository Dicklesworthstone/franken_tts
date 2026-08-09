// The engine lives in a Worker: hydration takes seconds and synthesis tens of seconds
// at wasm speed, and none of that may block the UI thread. Protocol: postMessage
// {type, ...}; replies mirror the request type with `ok` or `error`.

import init, {
  WasmEngine,
  ModelStaging,
  presets,
  preset_vector,
} from "./pkg/ftts_wasm.js?v=@SITEV@";

// Slice size for streaming OPFS into wasm. Big enough that per-call overhead is noise, small
// enough that the JS heap never holds a meaningful fraction of the model.
const INGEST_SLICE = 16 * 1024 * 1024;

let engine = null;

function reply(type, payload, transfer = []) {
  self.postMessage({ type, ...payload }, transfer);
}

self.onmessage = async ({ data }) => {
  try {
    switch (data.type) {
      case "init": {
        await init({ module_or_path: "./pkg/ftts_wasm_bg.wasm?v=@SITEV@" });
        reply("init", { ok: true, presets: JSON.parse(presets()) });
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
