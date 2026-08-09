// The engine lives in a Worker: hydration takes seconds and synthesis tens of seconds
// at wasm speed, and none of that may block the UI thread. Protocol: postMessage
// {type, ...}; replies mirror the request type with `ok` or `error`.

import init, { WasmEngine, presets, preset_vector } from "./pkg/ftts_wasm.js";

let engine = null;

function reply(type, payload, transfer = []) {
  self.postMessage({ type, ...payload }, transfer);
}

self.onmessage = async ({ data }) => {
  try {
    switch (data.type) {
      case "init": {
        await init({ module_or_path: "./pkg/ftts_wasm_bg.wasm" });
        reply("init", { ok: true, presets: JSON.parse(presets()) });
        break;
      }
      case "load": {
        // Buffers arrive transferred (zero-copy); the engine verifies the artifact's
        // digests again in wasm before reading a single tensor.
        engine = new WasmEngine(
          new Uint8Array(data.fttsq),
          new Uint8Array(data.codec),
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
