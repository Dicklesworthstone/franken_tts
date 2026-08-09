// Playground orchestration: loader → worker-hosted engine → WebAudio playback.

import { ensureModel, clearCache, cachedBytes } from "./loader.js";

const ui = Object.fromEntries(
  [
    "load-model",
    "dl-bar",
    "dl-status",
    "voice",
    "voice-character",
    "record",
    "record-status",
    "text",
    "seed",
    "speak",
    "synth-status",
    "synth-bar",
    "player",
    "download",
    "share",
    "error",
    "clear-cache",
  ].map((id) => [id.replace(/-([a-z])/g, (_, c) => c.toUpperCase()), document.getElementById(id)]),
);

const worker = new Worker("./engine-worker.js", { type: "module" });
const pending = new Map();
let requestCounter = 0;

function call(type, payload = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const requestId = ++requestCounter;
    pending.set(`${type}:${requestId}`, { resolve, reject });
    worker.postMessage({ type, requestId, ...payload }, transfer);
  });
}

worker.onmessage = ({ data }) => {
  const key = `${data.type}:${data.requestId ?? ""}`;
  // init/load replies carry no requestId; match by type alone.
  const entry =
    pending.get(key) ?? pending.get([...pending.keys()].find((k) => k.startsWith(`${data.type}:`)));
  if (!entry) return;
  pending.delete(key);
  for (const k of [...pending.keys()]) {
    if (k.startsWith(`${data.type}:`) && (data.requestId === undefined || k === key)) {
      pending.delete(k);
    }
  }
  if (data.ok) entry.resolve(data);
  else entry.reject(new Error(data.error));
};

function showError(error) {
  ui.error.textContent = String(error);
  ui.error.classList.remove("hidden");
}

function clearError() {
  ui.error.classList.add("hidden");
}

const gigabytes = (bytes) => (bytes / 1024 ** 3).toFixed(2);

let presetList = [];
let clonedVector = null;
let lastWavBlob = null;

async function boot() {
  const { presets } = await call("init");
  presetList = presets;
  for (const preset of presets) {
    const option = document.createElement("option");
    option.value = preset.name;
    option.textContent = preset.name;
    ui.voice.appendChild(option);
  }
  const cloned = document.createElement("option");
  cloned.value = "__cloned__";
  cloned.textContent = "my cloned voice";
  cloned.disabled = true;
  ui.voice.appendChild(cloned);
  updateVoiceCharacter();

  const cached = await cachedBytes();
  if (cached > 0) {
    ui.dlStatus.textContent = `${gigabytes(cached)} GB cached — click to verify & load.`;
  }
}

function updateVoiceCharacter() {
  const preset = presetList.find((p) => p.name === ui.voice.value);
  ui.voiceCharacter.textContent =
    ui.voice.value === "__cloned__" ? "your locally cloned voice" : (preset?.character ?? "");
}
ui.voice.addEventListener("change", updateVoiceCharacter);

ui.loadModel.addEventListener("click", async () => {
  clearError();
  ui.loadModel.disabled = true;
  try {
    const files = await ensureModel(({ bytesDone, bytesTotal, phase, asset }) => {
      ui.dlBar.style.width = `${((bytesDone / bytesTotal) * 100).toFixed(1)}%`;
      ui.dlStatus.textContent = `${phase} ${asset ?? ""} — ${gigabytes(bytesDone)} / ${gigabytes(bytesTotal)} GB`;
    });
    ui.dlStatus.textContent = "Hydrating the engine (this takes a minute at wasm speed)…";
    await call(
      "load",
      {
        fttsq: files.fttsq,
        codec: files.codec,
        vocab: files.vocab,
        merges: files.merges,
        tokenizerConfig: files.tokenizerConfig,
      },
      [files.fttsq, files.codec],
    );
    ui.dlBar.style.width = "100%";
    ui.dlStatus.textContent = "Model loaded. Ready to speak.";
    ui.speak.disabled = false;
  } catch (error) {
    showError(error);
    ui.loadModel.disabled = false;
  }
});

ui.speak.addEventListener("click", async () => {
  clearError();
  ui.speak.disabled = true;
  ui.synthBar.style.width = "10%";
  const startedAt = performance.now();
  const ticker = setInterval(() => {
    const seconds = ((performance.now() - startedAt) / 1000).toFixed(0);
    ui.synthStatus.textContent = `synthesizing… ${seconds}s (single-thread wasm is ~0.2–0.3× real time)`;
    ui.synthBar.style.width = `${Math.min(90, 10 + seconds * 2)}%`;
  }, 500);
  try {
    const payload = {
      text: ui.text.value,
      seed: Number.parseInt(ui.seed.value, 10) || 0,
    };
    if (ui.voice.value === "__cloned__" && clonedVector) {
      payload.voiceVector = clonedVector;
    } else {
      payload.voiceName = ui.voice.value;
    }
    const { pcm, sampleRate, elapsedMs } = await call("synthesize", payload);
    const samples = new Float32Array(pcm);
    lastWavBlob = pcmToWavBlob(samples, sampleRate);
    ui.player.src = URL.createObjectURL(lastWavBlob);
    ui.player.classList.remove("hidden");
    ui.download.classList.remove("hidden");
    if (navigator.canShare) ui.share.classList.remove("hidden");
    const audioSeconds = samples.length / sampleRate;
    ui.synthStatus.textContent = `${audioSeconds.toFixed(1)}s of audio in ${(elapsedMs / 1000).toFixed(1)}s (${(audioSeconds / (elapsedMs / 1000)).toFixed(2)}× real time)`;
    ui.synthBar.style.width = "100%";
    ui.player.play().catch(() => {});
  } catch (error) {
    showError(error);
  } finally {
    clearInterval(ticker);
    ui.speak.disabled = false;
  }
});

ui.record.addEventListener("click", async () => {
  clearError();
  ui.record.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const context = new AudioContext({ sampleRate: 24000 });
    const source = context.createMediaStreamSource(stream);
    const recorderNode = context.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    recorderNode.onaudioprocess = (event) => {
      chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    source.connect(recorderNode);
    recorderNode.connect(context.destination);

    const seconds = 10;
    for (let remaining = seconds; remaining > 0; remaining -= 1) {
      ui.recordStatus.textContent = `recording… ${remaining}s (read a couple of sentences)`;
      await new Promise((r) => setTimeout(r, 1000));
    }
    recorderNode.disconnect();
    source.disconnect();
    for (const track of stream.getTracks()) track.stop();
    await context.close();

    const total = chunks.reduce((n, c) => n + c.length, 0);
    const pcm = new Float32Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      pcm.set(chunk, offset);
      offset += chunk.length;
    }
    ui.recordStatus.textContent = "computing your voice vector…";
    const { vector } = await call("enroll", { pcm: pcm.buffer }, [pcm.buffer]);
    clonedVector = new Float32Array(vector);
    const clonedOption = [...ui.voice.options].find((o) => o.value === "__cloned__");
    clonedOption.disabled = false;
    ui.voice.value = "__cloned__";
    updateVoiceCharacter();
    ui.recordStatus.textContent = "cloned — your voice is selected.";
  } catch (error) {
    showError(error);
    ui.recordStatus.textContent = "";
  } finally {
    ui.record.disabled = false;
  }
});

ui.download.addEventListener("click", () => {
  if (!lastWavBlob) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(lastWavBlob);
  link.download = "franken_tts.wav";
  link.click();
});

ui.share.addEventListener("click", async () => {
  if (!lastWavBlob) return;
  const file = new File([lastWavBlob], "franken_tts.wav", { type: "audio/wav" });
  try {
    await navigator.share({ files: [file], title: "franken_tts" });
  } catch {
    /* user cancelled */
  }
});

ui.clearCache.addEventListener("click", async () => {
  await clearCache();
  ui.dlStatus.textContent = "Model cache cleared.";
  ui.dlBar.style.width = "0%";
  ui.speak.disabled = true;
  ui.loadModel.disabled = false;
});

function pcmToWavBlob(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) {
    view.setInt16(44 + i * 2, Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767))), true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

boot().catch(showError);
