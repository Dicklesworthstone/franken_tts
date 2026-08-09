// Playground orchestration: loader → worker-hosted engine → WebAudio playback.

import { ensureModel, clearCache, cachedBytes } from "./loader.js";
import { TOTAL_BYTES } from "./model-manifest.js";

const ui = Object.fromEntries(
  [
    "consent",
    "consent-yes",
    "consent-no",
    "load-model",
    "dl-bar",
    "dl-status",
    "voice",
    "voice-character",
    "record",
    "record-status",
    "script",
    "script-wrap",
    "clone-name",
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
let clonedName = "my voice";
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
  applySharedFragment();
  updateVoiceCharacter();

  // Persistence contract: the model downloads ONCE and stays in this browser's storage until
  // "Clear model cache". When a complete cache exists, hydrate immediately on page open —
  // consent is for the download, not for using what was already approved and downloaded.
  const cached = await cachedBytes();
  if (cached >= TOTAL_BYTES) {
    ui.dlStatus.textContent = "Model cached. Verifying and loading…";
    await loadFromStore();
  } else if (cached > 0) {
    ui.dlStatus.textContent = `${gigabytes(cached)} GB of a partial download cached; click to resume.`;
  }
}

async function loadFromStore() {
  ui.loadModel.disabled = true;
  ui.loadModel.classList.add("hidden");
  try {
    const files = await ensureModel(({ bytesDone, bytesTotal, phase, asset }) => {
      ui.dlBar.style.width = `${((bytesDone / bytesTotal) * 100).toFixed(1)}%`;
      ui.dlStatus.textContent = `${phase} ${asset ?? ""}: ${gigabytes(bytesDone)} / ${gigabytes(bytesTotal)} GB`;
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
    ui.loadModel.classList.remove("hidden");
  }
}

function updateVoiceCharacter() {
  const preset = presetList.find((p) => p.name === ui.voice.value);
  ui.voiceCharacter.textContent =
    ui.voice.value === "__cloned__" ? `“${clonedName}”, locally cloned` : (preset?.character ?? "");
}
ui.voice.addEventListener("change", updateVoiceCharacter);

// The download is 2 GB: never start it on a bare click. The first button only reveals the
// consent panel (size, storage location, resumability, removal); the download starts from
// explicit consent, and progress then reports rate and estimated time remaining from a
// rolling throughput window.
ui.loadModel.addEventListener("click", () => {
  clearError();
  ui.consent.classList.remove("hidden");
  ui.loadModel.classList.add("hidden");
});

ui.consentNo.addEventListener("click", () => {
  ui.consent.classList.add("hidden");
  ui.loadModel.classList.remove("hidden");
});

ui.consentYes.addEventListener("click", async () => {
  ui.consent.classList.add("hidden");
  ui.loadModel.disabled = true;
  const samples = [];
  const eta = (bytesDone, bytesTotal) => {
    const now = performance.now();
    samples.push([now, bytesDone]);
    while (samples.length > 2 && now - samples[0][0] > 10_000) samples.shift();
    const [t0, b0] = samples[0];
    const rate = (bytesDone - b0) / Math.max(now - t0, 1) * 1000; // bytes/s
    if (rate < 1024) return "";
    const remaining = (bytesTotal - bytesDone) / rate;
    const minutes = Math.floor(remaining / 60);
    const seconds = Math.round(remaining % 60);
    const speed = rate >= 1024 ** 2 ? `${(rate / 1024 ** 2).toFixed(1)} MB/s` : `${(rate / 1024).toFixed(0)} kB/s`;
    return ` · ${speed} · ~${minutes ? `${minutes}m ` : ""}${seconds}s left`;
  };
  try {
    const files = await ensureModel(({ bytesDone, bytesTotal, phase, asset }) => {
      ui.dlBar.style.width = `${((bytesDone / bytesTotal) * 100).toFixed(1)}%`;
      const tail = phase === "downloading" ? eta(bytesDone, bytesTotal) : "";
      ui.dlStatus.textContent = `${phase} ${asset ?? ""}: ${gigabytes(bytesDone)} / ${gigabytes(bytesTotal)} GB${tail}`;
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
    ui.loadModel.classList.remove("hidden");
  }
});

ui.speak.addEventListener("click", async () => {
  clearError();
  ui.speak.disabled = true;
  ui.synthBar.style.width = "10%";
  const startedAt = performance.now();
  const ticker = setInterval(() => {
    const seconds = ((performance.now() - startedAt) / 1000).toFixed(0);
    ui.synthStatus.textContent = `synthesizing… ${seconds}s (single-thread wasm runs ~0.03–0.05× real time, so a sentence takes a couple of minutes)`;
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

// The enrollment script, verbatim from the README: the Rainbow Passage + "Please call
// Stella" — phonetically rich, and a known transcript is future-proof for ICL cloning.
const ENROLLMENT_SCRIPT =
  "Please call Stella. Ask her to bring these things with her from the store: six spoons " +
  "of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother " +
  "Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop " +
  "these things into three red bags, and we will go meet her Wednesday at the train " +
  "station.\n\nWhen the sunlight strikes raindrops in the air, they act as a prism and " +
  "form a rainbow. The rainbow is a division of white light into many beautiful colors. " +
  "These take the shape of a long round arch, with its path high above, and its two ends " +
  "apparently beyond the horizon. There is, according to legend, a boiling pot of gold at " +
  "one end. People look, but no one ever finds it. When a man looks for something beyond " +
  "his reach, his friends say he is looking for the pot of gold at the end of the rainbow.";

let recorder = null; // {stop: () => void} while a recording is live

ui.record.addEventListener("click", async () => {
  clearError();
  if (recorder) {
    recorder.stop();
    return;
  }
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

    ui.script.textContent = ENROLLMENT_SCRIPT;
    ui.scriptWrap.classList.remove("hidden");
    ui.record.textContent = "⏹ Stop recording";
    const startedAt = performance.now();
    const ticker = setInterval(() => {
      ui.recordStatus.textContent = `recording… ${((performance.now() - startedAt) / 1000).toFixed(0)}s. Read the script aloud, then press Stop`;
    }, 500);

    const finish = async () => {
      clearInterval(ticker);
      recorder = null;
      ui.record.textContent = "🎙 Record & clone my voice";
      ui.scriptWrap.classList.add("hidden");
      recorderNode.disconnect();
      source.disconnect();
      for (const track of stream.getTracks()) track.stop();
      await context.close();

      const total = chunks.reduce((n, c) => n + c.length, 0);
      if (total < 24000 * 3) {
        ui.recordStatus.textContent = "";
        throw new Error("recording too short; read at least a few seconds of the script");
      }
      const pcm = new Float32Array(total);
      let offset = 0;
      for (const chunk of chunks) {
        pcm.set(chunk, offset);
        offset += chunk.length;
      }
      ui.recordStatus.textContent = "computing your voice vector…";
      const { vector } = await call("enroll", { pcm: pcm.buffer }, [pcm.buffer]);
      clonedVector = new Float32Array(vector);
      clonedName = (ui.cloneName.value || "my voice").trim().slice(0, 40);
      const clonedOption = [...ui.voice.options].find((o) => o.value === "__cloned__");
      clonedOption.disabled = false;
      clonedOption.textContent = clonedName;
      ui.voice.value = "__cloned__";
      updateVoiceCharacter();
      ui.recordStatus.textContent = `cloned: “${clonedName}” is selected.`;
    };
    recorder = { stop: () => finish().catch(showError) };
    // Backstop: the full script takes under a minute; stop automatically at 60 s.
    setTimeout(() => recorder?.stop(), 60_000);
  } catch (error) {
    showError(error);
    ui.recordStatus.textContent = "";
    recorder = null;
    ui.record.textContent = "🎙 Record & clone my voice";
  }
});

ui.download.addEventListener("click", () => {
  if (!lastWavBlob) return;
  const link = document.createElement("a");
  link.href = URL.createObjectURL(lastWavBlob);
  link.download = "franken_tts.wav";
  link.click();
});

// Share = a stateless URL. Everything needed to reproduce the utterance rides in the
// FRAGMENT (never sent to any server): the text, the seed, and the voice — a preset by
// name, or a custom clone as its full 1,024-float vector plus the name its owner gave it.
function base64UrlEncode(bytes) {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

function base64UrlDecode(text) {
  const binary = atob(text.replaceAll("-", "+").replaceAll("_", "/"));
  return Uint8Array.from(binary, (c) => c.charCodeAt(0));
}

function buildShareUrl() {
  const params = new URLSearchParams();
  params.set("v", "1");
  params.set("t", base64UrlEncode(new TextEncoder().encode(ui.text.value)));
  params.set("s", String(Number.parseInt(ui.seed.value, 10) || 0));
  if (ui.voice.value === "__cloned__" && clonedVector) {
    params.set("vec", base64UrlEncode(new Uint8Array(clonedVector.buffer)));
    params.set("name", encodeURIComponent(clonedName));
  } else {
    params.set("voice", ui.voice.value);
  }
  return `${location.origin}${location.pathname}#${params.toString()}`;
}

ui.share.addEventListener("click", async () => {
  const url = buildShareUrl();
  try {
    await navigator.clipboard.writeText(url);
    ui.synthStatus.textContent = "share link copied; it carries the text, seed, and voice.";
  } catch (error) {
    showError(error);
  }
});

function applySharedFragment() {
  if (!location.hash.startsWith("#v=")) return;
  try {
    const params = new URLSearchParams(location.hash.slice(1));
    const text = params.get("t");
    if (text) ui.text.value = new TextDecoder().decode(base64UrlDecode(text));
    if (params.get("s")) ui.seed.value = params.get("s");
    const vec = params.get("vec");
    if (vec) {
      const bytes = base64UrlDecode(vec);
      if (bytes.length === 1024 * 4) {
        clonedVector = new Float32Array(bytes.buffer);
        clonedName = decodeURIComponent(params.get("name") ?? "shared voice").slice(0, 40);
        const clonedOption = [...ui.voice.options].find((o) => o.value === "__cloned__");
        clonedOption.disabled = false;
        clonedOption.textContent = clonedName;
        ui.voice.value = "__cloned__";
      }
    } else if (params.get("voice")) {
      ui.voice.value = params.get("voice");
    }
    updateVoiceCharacter();
    ui.synthStatus.textContent = "loaded a shared utterance. Load the model, then Synthesize.";
  } catch {
    /* a malformed fragment is ignored, never fatal */
  }
}

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
