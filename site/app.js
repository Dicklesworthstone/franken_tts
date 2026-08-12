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
    "memory-warning",
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

// The last line of defense against a wedged engine Worker. A dead kernel Worker strands the
// engine Worker inside `memory.atomic.wait`, where it can run no JS ever again — no reply,
// no error, nothing (`panic = "abort"` on the wasm build makes in-wasm recovery impossible).
// Only THIS thread can still speak, so a call that outlives its generous ceiling is failed
// here with a reload suggestion instead of spinning forever. Ceilings are deliberately far
// past slow-phone reality (browser synthesis measured 0.31-0.43x realtime; hydration runs
// minutes on a cold phone): the watchdog exists to catch "never", not "slow".
const CALL_CEILINGS_MS = { init: 120_000, load: 20 * 60_000, synthesize: 15 * 60_000, enroll: 5 * 60_000 };

function call(type, payload = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const requestId = ++requestCounter;
    const key = `${type}:${requestId}`;
    const ceiling = CALL_CEILINGS_MS[type];
    const timer = ceiling
      ? setTimeout(() => {
          if (!pending.has(key)) return;
          pending.delete(key);
          recordStage(`watchdog:${type}`);
          reject(
            new Error(
              `the engine did not answer "${type}" within ${Math.round(ceiling / 60_000)} minutes — ` +
                "it is likely wedged (a crashed worker thread); reload the page",
            ),
          );
        }, ceiling)
      : null;
    pending.set(key, {
      resolve: (value) => {
        if (timer) clearTimeout(timer);
        resolve(value);
      },
      reject: (error) => {
        if (timer) clearTimeout(timer);
        reject(error);
      },
    });
    worker.postMessage({ type, requestId, ...payload }, transfer);
  });
}

// ── crash breadcrumbs ────────────────────────────────────────────────────────────────────────
//
// When iOS reclaims the tab there is no error, no unload event, and no console: the diagnosis dies
// with the page. The only evidence that survives a kill is what reached durable storage BEFORE the
// dangerous step ran. So the engine Worker announces each hydration stage as it ENTERS it, the
// page commits that synchronously to localStorage, and a stage still marked "entered" on the next
// load is — by construction — the one that killed us.
//
// It lives here rather than in the Worker because Workers have no localStorage at all.
const CRASH_KEY = "ftts-last-stage";

const stageStarted = new Map();

/// Update ONLY the crash breadcrumb, without touching the stage history.
///
/// Streaming reports progress many times a second. Routing those through `recordStage` pushed a
/// history entry per tick — 179 of them in one run — burying the init stages that say which build
/// loaded, which is the history's whole reason for existing. The breadcrumb is a single
/// last-known-position that wants overwriting; the history is a log that does not.
function markPosition(stage, detail) {
  try {
    localStorage.setItem(CRASH_KEY, JSON.stringify({ stage, detail, at: Date.now() }));
  } catch {
    /* private mode: diagnosis is best-effort, never a reason to fail the load */
  }
}

function recordStage(stage, detail) {
  try {
    localStorage.setItem(CRASH_KEY, JSON.stringify({ stage, detail, at: Date.now() }));
  } catch {
    /* private mode: diagnosis is best-effort, never a reason to fail the load */
  }
  // Show it too, not just record it. A page that says only "hydrating" for minutes is
  // indistinguishable from a hung one — which is exactly the report this is here to answer.
  // Naming the live stage turns "it never finishes" into "it is stuck in widen-codec".
  stageStarted.set(stage, performance.now());
  // Full history, not just the latest. Download progress rewrites the status line many times a
  // second, so the init stages — the ones that say which build actually loaded — were erased
  // before anyone could read them. The harness reads this array.
  (globalThis.__fttsStages ??= []).push(detail ? `${stage} [${detail}]` : stage);
  // Message-arrival pings (`msg:synthesize`, …) are forensics, not status: painting them
  // here replaced "Model loaded — 6 kernel threads" with the literal text "msg:synthesize"
  // on every click. They still land in localStorage and the stage history above.
  if (stage.startsWith("msg:")) return;
  const status = document.getElementById("dl-status");
  if (status) {
    status.textContent = detail ? `${stage} — ${detail}` : stage;
  }
}

function clearStage() {
  try {
    localStorage.removeItem(CRASH_KEY);
  } catch {
    /* as above */
  }
}

/// Surfaces the stage that killed the previous visit, if there was one.
export function reportPreviousCrash() {
  let noted;
  try {
    noted = JSON.parse(localStorage.getItem(CRASH_KEY) ?? "null");
  } catch {
    return null;
  }
  return noted?.stage ? noted : null;
}

worker.onmessage = ({ data }) => {
  // Stage pings are telemetry, not request replies: they carry no requestId and must not be
  // matched against the pending map or they would resolve someone else's promise.
  if (data.type === "stage") {
    recordStage(data.stage, data.detail);
    return;
  }
  // Streaming-into-wasm progress: also telemetry. The worker emits one of these per slice
  // precisely so the bar keeps moving during the multi-ten-second hydration phase; they
  // were silently discarded before, freezing the bar at 100%-downloaded.
  // Re-emit the worker's errors on the page console, which is the only one anything outside the
  // browser can observe. See the mirror in engine-worker.js for why this exists.
  if (data.type === "workerLog") {
    console.error(`[engine-worker] ${data.text}`);
    return;
  }
  if (data.type === "loadProgress") {
    // The worker's own total wins when it sends one: it stages the artifact's hot prefix rather
    // than the whole file, so the download total would leave the bar short of 100% forever.
    const total = Number.isFinite(data.bytesTotal) && data.bytesTotal > 0
      ? data.bytesTotal
      : hydrateTotalBytes;
    if (total > 0 && Number.isFinite(data.bytesDone)) {
      const percent = Math.min(100, (data.bytesDone / total) * 100);
      ui.dlBar.style.width = `${percent.toFixed(1)}%`;
      ui.dlStatus.textContent = `Streaming into the engine: ${gigabytes(data.bytesDone)} / ${gigabytes(total)} GB`;
      // Overwrite the breadcrumb with the live wasm size. A tab killed by the OS gets no
      // callback, no error and no unload, so the ONLY way to learn how big linear memory was at
      // the moment of death is to have written it down beforehand. Streaming is where a phone
      // died, and "it crashed somewhere in streaming" is not a number anyone can act on.
      if (Number.isFinite(data.wasmBytes)) {
        markPosition(
          "stream-artifact",
          `${gigabytes(data.bytesDone)} GB in, wasm ${gigabytes(data.wasmBytes)} GB`,
        );
      }
    }
    return;
  }
  const key = `${data.type}:${data.requestId ?? ""}`;
  // Every reply now echoes its requestId; the by-type fallback survives only for an older
  // worker script paired with a newer page mid-deploy. It settles exactly ONE promise —
  // the earlier version deleted every same-type key, so a second in-flight call of the
  // same type was dropped without ever settling.
  const fallbackKey =
    data.requestId === undefined
      ? [...pending.keys()].find((k) => k.startsWith(`${data.type}:`))
      : undefined;
  const matched = pending.has(key) ? key : fallbackKey;
  const entry = matched === undefined ? undefined : pending.get(matched);
  if (!entry) return;
  pending.delete(matched);
  if (data.ok) {
    // The last recorded stage completed without killing the tab; clearing here is what
    // makes the crash-breadcrumb contract true — a stage still marked "entered" on the
    // next visit is one that genuinely never finished.
    clearStage();
    entry.resolve(data);
  } else entry.reject(new Error(data.error));
};

// A worker that dies before (or instead of) replying must fail the calls, not strand them:
// a top-level import error in the worker script previously left `call("init")` pending
// forever and the page stuck on "Model not loaded." with zero error surface.
function failAllPending(reason) {
  const error = reason instanceof Error ? reason : new Error(String(reason));
  for (const [, entry] of pending) entry.reject(error);
  pending.clear();
  showError(error);
}
worker.onerror = (event) => {
  failAllPending(event.message ? `engine worker error: ${event.message}` : "engine worker crashed");
};
worker.onmessageerror = () => {
  failAllPending("engine worker message could not be deserialized");
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

/// Partitions the engine actually armed, reported by the worker after hydration.
///
/// Read rather than assumed: the team is sized from the workers that genuinely parked, so a build
/// where they failed to start runs serially and must say so. A page that cannot tell 1 from 6 is
/// how a fully-serial run went unnoticed for a whole session.
let engineThreads = 1;

// The armed int8/fast-math route, straight from the engine. Surfaced in the ready line
// because the wasm build arms quantized codec paths by default while native does not:
// same seed + same artifact can differ in PCM bits by platform, and that must be
// discoverable without reading source (parity doctrine: no silent numerics switches).
let engineRoute = "";

// Total bytes the load call will stream into wasm, for the hydration progress bar.
let hydrateTotalBytes = 0;

async function boot() {
  // Surface, don't just record: the breadcrumb only helps if somebody reads it back.
  const crash = reportPreviousCrash();
  clearStage();

  engineRoute = (await call("init"))?.route ?? "";

  // Persistence contract: the model downloads ONCE and stays in this browser's storage until
  // "Clear model cache". When a complete cache exists the page normally hydrates on open —
  // consent is for the download, not for using what was already approved and downloaded.
  //
  // UNLESS the previous visit died during hydration. Auto-loading into a hydration that kills the
  // tab makes the page unreachable: every visit re-crashes before anything can be clicked, so the
  // user cannot even reach the button that clears the cache. That happened on an iPhone, and it is
  // a trap of our own making — the cached model turns the site into a brick.
  //
  // A crash breadcrumb that names a hydration stage is therefore treated as a refusal to retry
  // automatically. The user can still load deliberately, and is pointed at the standalone reset
  // page, which loads no engine at all and so cannot crash for the same reason.
  const crashedHydrating =
    crash?.stage && !["compile-module", "create-memory", "instantiate"].includes(crash.stage);
  const cached = await cachedBytes();

  if (crashedHydrating) {
    // The detail is the whole point on a device that was killed by the OS. It carries how far
    // staging got and how large linear memory was at that instant, and a tab reclaimed by iOS
    // reports nothing else at all — no error, no unload, no callback. Showing only the stage name
    // threw away the one measurement that says whether the ceiling was hit and where.
    ui.dlStatus.innerHTML =
      `The previous visit stopped during <b>${crash.stage}</b>` +
      `${crash.detail ? ` — <code>${crash.detail}</code>` : ""}, so the model was not loaded ` +
      `automatically this time. Press “Download &amp; load model…” to try again, or ` +
      `<a href="reset.html">clear the cached model</a> if it keeps failing.`;
    ui.loadModel.classList.remove("hidden");
    ui.loadModel.disabled = false;
    return;
  }
  if (crash) {
    ui.dlStatus.textContent = `Note: the previous visit ended during “${crash.stage}”.`;
  }

  // Say the hard thing BEFORE the download, not after the tab dies.
  //
  // Hydration's resident set is ~2.45 GB, and that is not a tuning problem: 1.31 GB is the q8
  // artifact (read in place, zero-copy), 0.70 GB the widened codec, and the remaining 0.34 GB is
  // spread across 477 small f32 tensors with no dominant one. iOS Safari caps a tab well below
  // that once its own overhead is counted, so on an iPhone this reliably kills the tab — after a
  // 1.86 GB download the visitor has already paid for.
  //
  // The page still ALLOWS the attempt: it is the visitor's device, and the ceiling is a measured
  // expectation rather than a certainty. It just refuses to spend someone's bandwidth on a likely
  // crash without saying so first.
  if (isMemoryConstrainedDevice()) {
    ui.memoryWarning.innerHTML =
      `<b>This may not work on a phone.</b> Loading the model needs about 1.9 GB of memory at ` +
      `once, and mobile Safari allows a tab roughly 2 GB — so it is close, and it depends on the ` +
      `device. The download is 1.86 GB. If the tab crashes while loading, that is why, and a ` +
      `desktop browser will work. You can also <a href="reset.html">clear the cached model</a> ` +
      `afterwards.`;
    ui.memoryWarning.classList.remove("hidden");
  }
  if (cached >= TOTAL_BYTES) {
    ui.dlStatus.textContent = "Model cached. Verifying and loading…";
    await loadFromStore();
  } else if (cached > 0) {
    ui.dlStatus.textContent = `${gigabytes(cached)} GB of a partial download cached; click to resume.`;
  }
}

/// True on devices whose per-tab memory ceiling sits below what hydration needs.
///
/// Sniffs the platform rather than measuring, because there is no way to ASK a browser for its
/// tab ceiling, and the one measurement available — allocating until it fails — IS the crash
/// being avoided. iPadOS is caught by the touch test rather than the userAgent: it reports itself
/// as "MacIntel", so a string check alone would miss every iPad.
function isMemoryConstrainedDevice() {
  const ua = navigator.userAgent;
  const iOS =
    /iPhone|iPad|iPod/.test(ua) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  if (iOS) return true;
  // Android and small laptops. deviceMemory is coarse (rounded, capped at 8) and absent in
  // Safari, so it can only ever ADD a warning here, never clear one.
  return typeof navigator.deviceMemory === "number" && navigator.deviceMemory <= 4;
}

/// The visualization containers, by id. Kept here rather than imported from viz.js because this
/// must work even if viz.js failed to load at all.
const VIZ_CONTAINERS = [
  "anatomy-viz",
  "rvq-viz",
  "seams-viz",
  "speed-ladder",
  "wasm-stepper",
  "rms-bars",
];

/// Stop the page's decorative DOM from competing with the engine for a small device's memory.
///
/// Two things, and the teardown is not redundant with the flag. viz.js builds lazily on scroll but
/// ALSO on a 3.5-second timeout, so on a phone the visualizations are already built and animating
/// long before anyone presses load — the flag only prevents future builds. An iPhone that reached
/// "ready to speak" died on the very next scroll, with the engine resident at 1.61 GB; these
/// scenes are live SVG with their own animation loops, and they are what scrolling was touching.
///
/// Only on devices that need it, and only for this page view: a reload brings the whole page back.
function releasePageMemoryForEngine() {
  globalThis.__fttsEngineResident = true;
  if (!isMemoryConstrainedDevice()) return;
  for (const id of VIZ_CONTAINERS) {
    const node = document.getElementById(id);
    if (!node) continue;
    // replaceChildren() drops the subtree, its animation loops, and its compositor layers.
    node.replaceChildren();
    node.classList.add("hidden");
  }
  // Then the rest of the page below the playground.
  //
  // Scrolling is what kills it. iOS Safari synthesized successfully and then died on the scroll
  // down to the player, and the engine is not what moved — compositing is. Ten sections of
  // marketing page with 27 reveal-animated blocks each want a layer as they come into view, and
  // at ~1.6 GB resident there is nothing left to give them.
  //
  // Only the playground survives, and only on a device that needs it, and only until reload.
  // Someone who loaded a 1.86 GB model came for the playground; the prose can wait for a tab
  // that is not holding a neural network.
  for (const section of document.querySelectorAll("section")) {
    if (section.id === "playground" || section.id === "top") continue;
    section.remove();
  }
  // Reveal observers on surviving nodes have nothing left to reveal, and the animations
  // themselves are layer-creating. Show everything, animate nothing.
  for (const revealed of document.querySelectorAll(".reveal")) {
    revealed.classList.remove("reveal");
  }
  const note = document.getElementById("viz-note");
  if (note) {
    note.textContent =
      "The rest of this page is hidden while the model is loaded, so the engine has this " +
      "device's memory to itself. Reload without loading the model to read it.";
    note.classList.remove("hidden");
  }
}

async function loadFromStore() {
  releasePageMemoryForEngine();
  ui.loadModel.disabled = true;
  ui.loadModel.classList.add("hidden");
  try {
    const files = await ensureModel(({ bytesDone, bytesTotal, phase, asset }) => {
      ui.dlBar.style.width = `${((bytesDone / bytesTotal) * 100).toFixed(1)}%`;
      ui.dlStatus.textContent = `${phase} ${asset ?? ""}: ${gigabytes(bytesDone)} / ${gigabytes(bytesTotal)} GB`;
    });
    ui.dlStatus.textContent = "Hydrating the engine (this takes a minute at wasm speed)…";
    hydrateTotalBytes = (files.codec?.bytes ?? 0) + (files.fttsq?.bytes ?? 0);
    const loaded = await call(
      "load",
      {
        fttsq: files.fttsq,
        codec: files.codec,
        vocab: files.vocab,
        merges: files.merges,
        tokenizerConfig: files.tokenizerConfig,
      },
    );
    engineThreads = loaded?.threads ?? 1;
    ui.dlBar.style.width = "100%";
    ui.dlStatus.textContent =
      (engineThreads > 1
        ? `Model loaded - ${engineThreads} kernel threads.`
        : "Model loaded (single thread).") +
      (engineRoute ? ` int8 route: ${engineRoute}.` : "") +
      " Ready to speak.";
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

// One-click sample texts, kept short so a first try returns quickly. At 0.31x real time a
// sentence is seconds of compute rather than the minutes the single-threaded build cost.
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
    hydrateTotalBytes = (files.codec?.bytes ?? 0) + (files.fttsq?.bytes ?? 0);
    const loaded = await call(
      "load",
      {
        fttsq: files.fttsq,
        codec: files.codec,
        vocab: files.vocab,
        merges: files.merges,
        tokenizerConfig: files.tokenizerConfig,
      },
    );
    engineThreads = loaded?.threads ?? 1;
    ui.dlBar.style.width = "100%";
    ui.dlStatus.textContent =
      (engineThreads > 1
        ? `Model loaded - ${engineThreads} kernel threads.`
        : "Model loaded (single thread).") +
      (engineRoute ? ` int8 route: ${engineRoute}.` : "") +
      " Ready to speak.";
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
    // The engine reports its own team width at load; say what THIS session is actually running
    // rather than a figure baked in at authoring time, which is how the previous copy ended up
    // claiming "single-thread ... a couple of minutes" on a six-partition build ~6x faster.
    const how = engineThreads > 1 ? `${engineThreads} threads` : "single thread";
    ui.synthStatus.textContent = `synthesizing… ${seconds}s (${how}, ~0.3× real time)`;
    // Progress paced to the measured rate: roughly 3.2 s of compute per second of speech.
    ui.synthBar.style.width = `${Math.min(90, 10 + seconds * 8)}%`;
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
    // Sweepable without a rebuild, for the harness. Unset means the engine's own default.
    if (Number.isFinite(globalThis.__fttsPacketFrames)) {
      payload.packetFrames = globalThis.__fttsPacketFrames;
    }
    const { pcm, sampleRate, elapsedMs } = await call("synthesize", payload);
    const samples = new Float32Array(pcm);
    setWavBlob(pcmToWavBlob(samples, sampleRate));
    lastSamples = samples;
    lastSampleRate = sampleRate;
    // Expose the raw PCM for the conformance harness, which compares these exact samples against
    // the CLI's for the same text/voice/seed. Reading the <audio> element instead would measure
    // the WAV encoder and the browser's decoder rather than the engine.
    globalThis.__fttsLastPcm = samples;
    globalThis.__fttsLastSampleRate = sampleRate;
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

// The enrollment script, verbatim from the README: the "Please call Stella" elicitation
// paragraph (the phonetically densest part) plus the Rainbow Passage's opening for flowing
// prosody. The encoder pools over everything recorded with no truncation, but its voice
// information saturates well before a minute, so ~30 s of reading is the sweet spot; the
// known transcript also stays future-proof for ICL cloning.
const ENROLLMENT_SCRIPT =
  "Please call Stella. Ask her to bring these things with her from the store: six spoons " +
  "of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother " +
  "Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop " +
  "these things into three red bags, and we will go meet her Wednesday at the train " +
  "station.\n\nWhen the sunlight strikes raindrops in the air, they act as a prism and " +
  "form a rainbow. The rainbow is a division of white light into many beautiful colors.";

let recorder = null; // {stop: () => void} while a recording is live

ui.record.addEventListener("click", async () => {
  clearError();
  if (recorder) {
    recorder.stop();
    return;
  }
  // Hoisted out of the `try` so the catch can release them: without this, a failure after
  // `getUserMedia` (an AudioContext that rejects the 24 kHz rate, a retired API) left the
  // microphone LIVE — OS recording indicator on — until reload, on a page whose whole
  // pitch is that nothing leaves the tab.
  let stream = null;
  let context = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    context = new AudioContext({ sampleRate: 24000 });
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
    // Backstop: the script reads in about half a minute; stop automatically at 60 s, which
    // leaves room for slow readers. Guarded so a stale timer from an earlier recording can
    // never stop a later one.
    setTimeout(() => {
      if (recorder === session) session.stop();
    }, 60_000);
  } catch (error) {
    for (const track of stream?.getTracks() ?? []) track.stop();
    await context?.close().catch(() => {});
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
