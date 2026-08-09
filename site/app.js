// Playground orchestration: loader → worker-hosted engine → WebAudio playback.

import { ensureModel, clearCache, cachedBytes } from "./loader.js?v=@SITEV@";
import { TOTAL_BYTES } from "./model-manifest.js?v=@SITEV@";

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
    "char-count",
    "dice",
    "wave-canvas",
    "voice-cards",
    "play-again",
    "download-mp3",
  ].map((id) => [id.replace(/-([a-z])/g, (_, c) => c.toUpperCase()), document.getElementById(id)]),
);

const worker = new Worker("./engine-worker.js?v=@SITEV@", { type: "module" });
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

// Mirrors PRESET_VOICES in crates/ftts-wasm/src/lib.rs (names and characters only; the
// vectors stay inside the wasm module). Static so the voice UI renders instantly and the
// voices section works even before the worker finishes booting.
const PRESETS = [
  { name: "matt", character: "warm, easy, masculine; the out-of-box default" },
  { name: "james", character: "natural, conversational, masculine" },
  { name: "leo", character: "relaxed, resonant, masculine" },
  { name: "robert", character: "steady, measured, masculine" },
  { name: "judy", character: "bright, articulate, feminine" },
  { name: "aria", character: "clear, warm, feminine" },
  { name: "ember", character: "aria's character, a few semitones deeper" },
];

let clonedVector = null;
let clonedName = "my voice";
let lastWavBlob = null;
let lastWavUrl = null;
let lastSamples = null; // Float32Array of the latest synthesis, for MP3 encoding
let lastSampleRate = 24000;
let lastMp3Blob = null; // encoded lazily, once per synthesis

// One object URL per synthesis, shared by the player and the download link; the
// previous one is revoked so repeated syntheses don't leak blobs.
function wavObjectUrl() {
  return lastWavUrl;
}
function setWavBlob(blob) {
  if (lastWavUrl) URL.revokeObjectURL(lastWavUrl);
  lastWavBlob = blob;
  lastWavUrl = URL.createObjectURL(blob);
}

for (const preset of PRESETS) {
  const option = document.createElement("option");
  option.value = preset.name;
  option.textContent = preset.name;
  ui.voice.appendChild(option);
}
{
  const cloned = document.createElement("option");
  cloned.value = "__cloned__";
  cloned.textContent = "my cloned voice";
  cloned.disabled = true;
  ui.voice.appendChild(cloned);
}
buildVoiceCards(PRESETS);
applySharedFragment();
updateVoiceCharacter();
updateCharCount();

async function boot() {
  await call("init");

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
  const preset = PRESETS.find((p) => p.name === ui.voice.value);
  ui.voiceCharacter.textContent =
    ui.voice.value === "__cloned__" ? `“${clonedName}”, locally cloned` : (preset?.character ?? "");
}
ui.voice.addEventListener("change", updateVoiceCharacter);

// The voices section: one card per preset, plus a card for cloning your own. Picking a
// card selects that voice in the playground and jumps there.
function buildVoiceCards(presets) {
  if (!ui.voiceCards) return;
  const jumpToPlayground = (focusTarget) => {
    document.getElementById("playground").scrollIntoView();
    if (focusTarget) focusTarget.focus({ preventScroll: true });
  };
  for (const preset of presets) {
    const card = document.createElement("div");
    card.className = "pg-card rounded-2xl border border-white/5 bg-black/40 p-6";
    const name = document.createElement("div");
    name.className = "pg-voice-name";
    name.textContent = preset.name;
    const character = document.createElement("p");
    character.className = "pg-note pg-voice-char mb-3";
    character.textContent = preset.character ?? "";
    const use = document.createElement("button");
    use.className = "pg-btn";
    use.textContent = "Use this voice";
    use.addEventListener("click", () => {
      ui.voice.value = preset.name;
      updateVoiceCharacter();
      jumpToPlayground(ui.text);
    });
    card.append(name, character, use);
    ui.voiceCards.appendChild(card);
  }
  const cloneCard = document.createElement("div");
  cloneCard.className = "pg-card rounded-2xl border border-emerald-500/25 bg-black/40 p-6";
  const cloneName = document.createElement("div");
  cloneName.className = "pg-voice-name";
  cloneName.textContent = "your voice";
  const cloneNote = document.createElement("p");
  cloneNote.className = "pg-note pg-voice-char mb-3";
  cloneNote.textContent =
    "Read a short script into your microphone; the speaker encoder runs locally and the recording is discarded.";
  const cloneBtn = document.createElement("button");
  cloneBtn.className = "pg-btn";
  cloneBtn.textContent = "Clone it in the playground";
  cloneBtn.addEventListener("click", () => jumpToPlayground(ui.record));
  cloneCard.append(cloneName, cloneNote, cloneBtn);
  ui.voiceCards.appendChild(cloneCard);
}

function updateCharCount() {
  if (!ui.charCount) return;
  ui.charCount.textContent = `${ui.text.value.length} / ${ui.text.maxLength}`;
}
ui.text.addEventListener("input", updateCharCount);

// One-click sample texts; short on purpose, since a wasm sentence costs minutes.
const SAMPLE_TEXTS = [
  "Now is the time for all good men to come to the aid of the agents.",
  "When sunlight strikes raindrops in the air, they act as a prism and form a rainbow.",
  "I am a voice model running in a browser tab, with no server behind me.",
];
{
  const samplesWrap = document.getElementById("samples");
  if (samplesWrap) {
    for (const sample of SAMPLE_TEXTS) {
      const chip = document.createElement("button");
      chip.className = "pg-chip";
      chip.style.cursor = "pointer";
      chip.textContent = `${sample.slice(0, 34)}…`;
      chip.title = sample;
      chip.addEventListener("click", () => {
        ui.text.value = sample;
        updateCharCount();
        ui.text.focus();
      });
      samplesWrap.appendChild(chip);
    }
  }
}

ui.dice.addEventListener("click", () => {
  ui.seed.value = String(Math.floor(Math.random() * 100000));
});

// Ctrl+Enter / Cmd+Enter synthesizes from anywhere in the playground.
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !ui.speak.disabled) {
    event.preventDefault();
    ui.speak.click();
  }
});

// Min/max envelope of the finished PCM, drawn once per synthesis.
function drawWaveform(samples) {
  const canvas = ui.waveCanvas;
  if (!canvas) return;
  const width = Math.max(300, Math.floor(canvas.clientWidth || canvas.parentElement.clientWidth));
  canvas.width = width * (window.devicePixelRatio || 1);
  canvas.height = 72 * (window.devicePixelRatio || 1);
  canvas.style.width = `${width}px`;
  canvas.style.height = "72px";
  const ctx = canvas.getContext("2d");
  ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
  ctx.clearRect(0, 0, width, 72);
  const mid = 36;
  const perPixel = Math.max(1, Math.floor(samples.length / width));
  ctx.fillStyle = "rgba(52, 211, 153, 0.75)";
  for (let x = 0; x < width; x += 1) {
    let lo = 0;
    let hi = 0;
    const start = x * perPixel;
    for (let i = start; i < Math.min(start + perPixel, samples.length); i += 1) {
      if (samples[i] < lo) lo = samples[i];
      if (samples[i] > hi) hi = samples[i];
    }
    const top = mid - hi * 32;
    const bottom = mid - lo * 32;
    ctx.fillRect(x, top, 1, Math.max(1, bottom - top));
  }
  canvas.classList.remove("hidden");
}

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
    setWavBlob(pcmToWavBlob(samples, sampleRate));
    lastSamples = samples;
    lastSampleRate = sampleRate;
    lastMp3Blob = null;
    drawWaveform(samples);
    ui.player.src = wavObjectUrl();
    ui.player.classList.remove("hidden");
    ui.playAgain.classList.remove("hidden");
    ui.download.classList.remove("hidden");
    ui.downloadMp3.classList.remove("hidden");
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
    const session = { stop: () => finish().catch(showError) };
    recorder = session;
    // Backstop: the full script takes under a minute; stop automatically at 60 s. Guarded
    // so a stale timer from an earlier recording can never stop a later one.
    setTimeout(() => {
      if (recorder === session) session.stop();
    }, 60_000);
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
  link.href = wavObjectUrl();
  link.download = "franken_tts.wav";
  link.click();
});

ui.playAgain.addEventListener("click", () => {
  ui.player.currentTime = 0;
  ui.player.play().catch(showError);
});

// MP3 export: the vendored lamejs encoder (LGPL, loaded as its own classic script) is
// fetched lazily on the first click, and each synthesis is encoded at most once.
let lameLoading = null;
function loadLame() {
  if (globalThis.lamejs) return Promise.resolve();
  lameLoading ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "vendor/lame.min.js";
    script.onload = () => resolve();
    script.onerror = () => {
      lameLoading = null;
      reject(new Error("failed to load the MP3 encoder"));
    };
    document.head.appendChild(script);
  });
  return lameLoading;
}

function encodeMp3(samples, sampleRate) {
  const encoder = new globalThis.lamejs.Mp3Encoder(1, sampleRate, 128);
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    pcm[i] = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32767)));
  }
  const parts = [];
  const BLOCK = 1152; // one MPEG frame of samples
  for (let i = 0; i < pcm.length; i += BLOCK) {
    const chunk = encoder.encodeBuffer(pcm.subarray(i, Math.min(i + BLOCK, pcm.length)));
    if (chunk.length) parts.push(chunk);
  }
  const tail = encoder.flush();
  if (tail.length) parts.push(tail);
  return new Blob(parts, { type: "audio/mpeg" });
}

ui.downloadMp3.addEventListener("click", async () => {
  if (!lastSamples) return;
  ui.downloadMp3.disabled = true;
  const originalLabel = ui.downloadMp3.textContent;
  ui.downloadMp3.textContent = "Encoding…";
  try {
    await loadLame();
    lastMp3Blob ??= encodeMp3(lastSamples, lastSampleRate);
    const url = URL.createObjectURL(lastMp3Blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "franken_tts.mp3";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  } catch (error) {
    showError(error);
  } finally {
    ui.downloadMp3.disabled = false;
    ui.downloadMp3.textContent = originalLabel;
  }
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
