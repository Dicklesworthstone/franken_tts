// Interactive anatomy of one 80 ms frame, plus count-up receipts.
//
// Pedagogy target: the one fact the README says nobody sees from the outside — the
// "12.5 Hz model" story hides a 5-layer residual decoder that runs FIFTEEN sequential
// times inside every frame. The animation makes that loop visible: the talker fires
// once, the microdecoder drum spins fifteen times filling fifteen code slots, the
// codec turns the full set into 1,920 samples. Real numbers throughout; the byte
// counter is the first-order Q8 weight-traffic figure from the plan.
//
// Vanilla JS + SVG, lazy-started by IntersectionObserver, honoring
// prefers-reduced-motion by jumping to the completed state.

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Build lazily on approach, but never rely on intersection alone: an instant scroll
// jump can skip right past an element without a single intersection callback, so a
// timeout fallback guarantees every visualization exists within a few seconds.
// Same guarantee for one-shot entrance animations (count-ups, bar fills): run when
// the element becomes visible, or after a timeout if an instant jump skipped it.
function onceVisible(el, run, threshold = 0.4, timeout = 3500) {
  let done = false;
  const fire = () => {
    if (done) return;
    done = true;
    run();
  };
  const obs = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      obs.disconnect();
      fire();
    }
  }, { threshold });
  obs.observe(el);
  setTimeout(fire, timeout);
}

function lazyBuild(el, build) {
  let built = false;
  const once = () => {
    if (built) return;
    built = true;
    build();
  };
  new IntersectionObserver((entries, observer) => {
    if (entries.some((e) => e.isIntersecting)) {
      observer.disconnect();
      once();
    }
  }, { rootMargin: "150px" }).observe(el);
  setTimeout(once, 3000);
}

/* ---------------------------------------------------------------- frame anatomy */

const anatomy = document.getElementById("anatomy-viz");

function buildAnatomy() {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 900 360");
  svg.setAttribute("role", "img");
  svg.setAttribute(
    "aria-label",
    "One 80 millisecond frame: the talker runs once, the microdecoder runs fifteen times, the codec emits 1,920 samples",
  );
  svg.style.width = "100%";
  svg.style.height = "auto";
  anatomy.appendChild(svg);

  const make = (tag, attrs, parent = svg) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    parent.appendChild(el);
    return el;
  };
  const label = (x, y, text, size = 13, fill = "#94a3b8", weight = "600", anchor = "middle") => {
    const t = make("text", {
      x, y, fill,
      "font-size": size,
      "font-weight": weight,
      "text-anchor": anchor,
      "font-family": "Inter, sans-serif",
    });
    t.textContent = text;
    return t;
  };

  // ---- talker: 28 thin layers
  const talkerX = 30, talkerY = 60, talkerW = 150, layerH = 6.5;
  make("rect", { x: talkerX - 12, y: talkerY - 34, width: talkerW + 24, height: 268, rx: 14, fill: "rgba(255,255,255,0.02)", stroke: "rgba(255,255,255,0.07)" });
  label(talkerX + talkerW / 2, talkerY - 12, "TALKER", 12, "#10b981", "900");
  label(talkerX + talkerW / 2, talkerY + 220, "28 layers · runs once", 13, "#94a3b8");
  label(talkerX + talkerW / 2, talkerY + 238, "predicts code #0", 13, "#94a3b8");
  const talkerLayers = [];
  for (let i = 0; i < 28; i++) {
    talkerLayers.push(make("rect", {
      x: talkerX, y: talkerY + i * (layerH + 0.9), width: talkerW, height: layerH,
      rx: 2, fill: "rgba(52,211,153,0.10)", stroke: "rgba(52,211,153,0.20)", "stroke-width": 0.6,
    }));
  }

  // ---- microdecoder: 5-layer drum + 15 code slots
  const microX = 300, microY = 78, microW = 170;
  make("rect", { x: microX - 12, y: talkerY - 34, width: microW + 24, height: 268, rx: 14, fill: "rgba(16,185,129,0.035)", stroke: "rgba(52,211,153,0.22)" });
  label(microX + microW / 2, talkerY - 12, "MICRODECODER", 12, "#34d399", "900");
  const passCounter = label(microX + microW / 2, microY + 22, "", 30, "#34d399", "900");
  label(microX + microW / 2, microY + 44, "sequential passes", 13, "#94a3b8");
  const microLayers = [];
  for (let i = 0; i < 5; i++) {
    microLayers.push(make("rect", {
      x: microX + 10, y: microY + 60 + i * 15, width: microW - 20, height: 11,
      rx: 3, fill: "rgba(52,211,153,0.12)", stroke: "rgba(52,211,153,0.3)", "stroke-width": 0.7,
    }));
  }
  label(microX + microW / 2, microY + 152, "5 layers, rerun per pass", 13, "#94a3b8");
  label(microX + microW / 2, microY + 196, "the body is reread 15×:", 13, "#94a3b8");
  label(microX + microW / 2, microY + 214, "≈1.18 GB of the frame's traffic", 13, "#cbd5e1");

  // ---- code slots between micro and codec
  const slotX = 520, slotY = 52;
  label(slotX + 24, slotY - 14, "16 codes", 13, "#94a3b8");
  const slots = [];
  for (let i = 0; i < 16; i++) {
    const s = make("rect", {
      x: slotX, y: slotY + i * 16.4, width: 48, height: 12.5,
      rx: 3, fill: "rgba(255,255,255,0.03)", stroke: "rgba(255,255,255,0.10)", "stroke-width": 0.7,
    });
    slots.push(s);
  }

  // ---- codec + waveform
  const codecX = 640, codecY = 60, codecW = 230;
  make("rect", { x: codecX - 12, y: talkerY - 34, width: codecW + 24, height: 268, rx: 14, fill: "rgba(255,255,255,0.02)", stroke: "rgba(255,255,255,0.07)" });
  label(codecX + codecW / 2, talkerY - 12, "CODEC", 12, "#10b981", "900");
  label(codecX + codecW / 2, codecY + 30, "causal decoder", 13, "#94a3b8");
  label(codecX + codecW / 2, codecY + 48, "upsample 8·5·4·3", 13, "#94a3b8");
  const wave = make("polyline", {
    points: "", fill: "none", stroke: "#34d399", "stroke-width": 2, "stroke-linejoin": "round",
    opacity: 0,
  });
  const samplesLabel = label(codecX + codecW / 2, codecY + 218, "", 13.5, "#cbd5e1", "700");

  // ---- traffic counter
  const traffic = label(450, 348, "", 15, "#e2e8f0", "700");

  const wavePoints = (progress) => {
    const points = [];
    const n = 64;
    for (let i = 0; i <= n * progress; i++) {
      const x = codecX + 12 + (i / n) * (codecW - 24);
      const t = i / n;
      const env = Math.sin(Math.PI * t);
      const y = codecY + 140 - env * (Math.sin(i * 0.9) * 24 + Math.sin(i * 2.3) * 9);
      points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return points.join(" ");
  };

  const setState = (state) => {
    // state: {talker: 0..1, pass: 0..15, wave: 0..1, gb: number}
    talkerLayers.forEach((l, i) => {
      const on = i / 28 < state.talker;
      l.setAttribute("fill", on ? "rgba(52,211,153,0.45)" : "rgba(52,211,153,0.10)");
    });
    passCounter.textContent = state.pass > 0 ? `×${state.pass}` : "×15";
    slots.forEach((s, i) => {
      const filled = (i === 0 && state.talker >= 1) || (i > 0 && i <= state.pass);
      s.setAttribute("fill", filled ? "rgba(52,211,153,0.55)" : "rgba(255,255,255,0.03)");
      s.setAttribute("stroke", filled ? "#34d399" : "rgba(255,255,255,0.10)");
    });
    microLayers.forEach((l) => {
      l.setAttribute("fill", state.microHot ? "rgba(52,211,153,0.5)" : "rgba(52,211,153,0.12)");
    });
    wave.setAttribute("opacity", state.wave > 0 ? 1 : 0);
    wave.setAttribute("points", wavePoints(state.wave));
    samplesLabel.textContent = state.wave >= 1 ? "1,920 samples · 24 kHz" : "";
    traffic.textContent = `Q8 weight bytes touched this frame: ${state.gb.toFixed(2)} GB of ≈1.65 GB`;
  };

  if (reducedMotion) {
    setState({ talker: 1, pass: 15, microHot: false, wave: 1, gb: 1.65 });
    return;
  }

  // Animation loop: talker cascade (0.9s) → 15 micro pulses (~3.4s) → wave (0.9s), repeat.
  let start = null;
  const TALKER_MS = 900, PASS_MS = 225, WAVE_MS = 900, HOLD_MS = 2200;
  const TOTAL = TALKER_MS + 15 * PASS_MS + WAVE_MS + HOLD_MS;
  const TALKER_GB = 0.30, MICRO_GB = 1.18, CODEC_GB = 0.17;
  const tick = (now) => {
    if (start === null) start = now;
    const t = (now - start) % TOTAL;
    if (t < TALKER_MS) {
      setState({ talker: t / TALKER_MS, pass: 0, microHot: false, wave: 0, gb: (t / TALKER_MS) * TALKER_GB });
    } else if (t < TALKER_MS + 15 * PASS_MS) {
      const p = (t - TALKER_MS) / PASS_MS;
      setState({
        talker: 1, pass: Math.min(15, Math.floor(p) + 1),
        microHot: p % 1 < 0.5, wave: 0,
        gb: TALKER_GB + (p / 15) * MICRO_GB,
      });
    } else if (t < TALKER_MS + 15 * PASS_MS + WAVE_MS) {
      const w = (t - TALKER_MS - 15 * PASS_MS) / WAVE_MS;
      setState({ talker: 1, pass: 15, microHot: false, wave: w, gb: TALKER_GB + MICRO_GB + w * CODEC_GB });
    } else {
      setState({ talker: 1, pass: 15, microHot: false, wave: 1, gb: 1.65 });
    }
    raf = requestAnimationFrame(tick);
  };
  let raf = requestAnimationFrame(tick);
  // Stop the loop when scrolled far away; restart when back.
  new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && raf === null) { start = null; raf = requestAnimationFrame(tick); }
      if (!e.isIntersecting && raf !== null) { cancelAnimationFrame(raf); raf = null; }
    }
  }, { rootMargin: "200px" }).observe(anatomy);
}

if (anatomy) lazyBuild(anatomy, buildAnatomy);

/* ---------------------------------------------------------------- receipts count-up */

for (const el of document.querySelectorAll("[data-countup]")) {
  const target = Number.parseFloat(el.dataset.countup);
  const decimals = Number.parseInt(el.dataset.decimals ?? "0", 10);
  const suffix = el.dataset.suffix ?? "";
  if (reducedMotion) {
    el.textContent = target.toFixed(decimals) + suffix;
    continue;
  }
  el.textContent = (0).toFixed(decimals) + suffix;
  onceVisible(el, () => {
    const started = performance.now();
    const duration = 1100;
    const step = (now) => {
      const t = Math.min(1, (now - started) / duration);
      const eased = 1 - (1 - t) ** 3;
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

/* ------------------------------------------- pipeline seams comparison */
// Two lanes: a classic multi-model TTS stack with three lossy seams, and the
// codec-LM shape with one seam. Flow dots ride each lane; seams get amber ticks.

const seams = document.getElementById("seams-viz");

function buildSeams() {
  const NS = "http://www.w3.org/2000/svg";
  const W = 900, H = 312;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "A classic TTS pipeline with three lossy seams compared to a codec language model with one seam");
  svg.style.width = "100%";
  svg.style.height = "auto";
  seams.appendChild(svg);

  const make = (tag, attrs, parent = svg) => {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    parent.appendChild(el);
    return el;
  };
  const label = (x, y, text, size = 12.5, fill = "#94a3b8", weight = "600", anchor = "middle") => {
    const t = make("text", {
      x, y, fill, "font-size": size, "font-weight": weight,
      "text-anchor": anchor, "font-family": "Inter, sans-serif",
    });
    t.textContent = text;
    return t;
  };

  // A lane: boxes at given x centers, connectors between them, seam ticks on marked joints.
  const lane = (y, boxes, tone) => {
    const boxH = 46, centers = [];
    const stroke = tone === "us" ? "rgba(52,211,153,0.45)" : "rgba(148,163,184,0.3)";
    const fill = tone === "us" ? "rgba(16,185,129,0.08)" : "rgba(255,255,255,0.03)";
    const textFill = tone === "us" ? "#a7f3d0" : "#cbd5e1";
    for (const box of boxes) {
      const w = box.w;
      const x = box.x - w / 2;
      make("rect", { x, y: y - boxH / 2, width: w, height: boxH, rx: 10, fill, stroke, "stroke-width": 1.2 });
      label(box.x, y - (box.sub ? 4 : -5), box.name, 14.5, textFill, "700");
      if (box.sub) label(box.x, y + 15, box.sub, 12, "#94a3b8");
      centers.push({ x: box.x, w });
    }
    for (let i = 0; i < centers.length - 1; i += 1) {
      const from = centers[i].x + centers[i].w / 2;
      const to = centers[i + 1].x - centers[i + 1].w / 2;
      make("line", { x1: from, y1: y, x2: to - 6, y2: y, stroke: "rgba(148,163,184,0.35)", "stroke-width": 1.5 });
      make("path", { d: `M ${to - 6} ${y - 4} L ${to} ${y} L ${to - 6} ${y + 4}`, fill: "none", stroke: "rgba(148,163,184,0.5)", "stroke-width": 1.5 });
      if (boxes[i].seamAfter) {
        const mid = (from + to) / 2;
        make("line", { x1: mid, y1: y - 14, x2: mid, y2: y + 14, stroke: "#fbbf24", "stroke-width": 2, opacity: 0.85 });
        label(mid, y - 20, "seam", 11.5, "#fbbf24", "800");
      }
    }
    return centers;
  };

  label(16, 30, "MOST TTS STACKS", 12, "#94a3b8", "900", "start");
  label(16, 190, "A CODEC LANGUAGE MODEL", 12, "#34d399", "900", "start");
  const y1 = 85, y2 = 245;
  lane(y1, [
    { x: 90, w: 120, name: "text" },
    { x: 265, w: 160, name: "acoustic model", seamAfter: true },
    { x: 470, w: 170, name: "mel spectrogram", sub: "a lossy picture of sound", seamAfter: true },
    { x: 668, w: 130, name: "vocoder", seamAfter: true },
    { x: 830, w: 100, name: "audio" },
  ], "them");
  label(450, 137, "three seams, three separately trained models, errors compound left to right", 13, "#94a3b8");
  lane(y2, [
    { x: 118, w: 176, name: "text + voice vector" },
    { x: 388, w: 236, name: "one transformer", sub: "talker + microdecoder", seamAfter: true },
    { x: 640, w: 156, name: "discrete codes", sub: "the codec's native input" },
    { x: 830, w: 100, name: "audio" },
  ], "us");
  label(450, 297, "one seam, and the tokens crossing it are exactly what the codec was trained on", 13, "#94a3b8");

  // Flow dots (skipped under reduced motion).
  if (!reducedMotion) {
    const dot1 = make("circle", { r: 4, cy: y1, fill: "#94a3b8" });
    const dot2 = make("circle", { r: 4, cy: y2, fill: "#34d399" });
    const X0 = 40, X1 = 860, PERIOD = 5200;
    let raf = null;
    const tick = (now) => {
      const t = (now % PERIOD) / PERIOD;
      dot1.setAttribute("cx", X0 + t * (X1 - X0));
      dot2.setAttribute("cx", X0 + ((t + 0.5) % 1) * (X1 - X0));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting && raf === null) raf = requestAnimationFrame(tick);
        if (!e.isIntersecting && raf !== null) { cancelAnimationFrame(raf); raf = null; }
      }
    }, { rootMargin: "150px" }).observe(seams);
  }
}

if (seams) lazyBuild(seams, buildSeams);

/* -------------------------------------------- RVQ coarse-to-fine slider */
// Illustrative residual refinement: the target curve is a fixed sum of 16
// components; the reconstruction uses only the first N. Dragging the slider is
// the whole lesson: early codes sketch, later codes correct.

const rvq = document.getElementById("rvq-viz");

function buildRvq() {
  const NS = "http://www.w3.org/2000/svg";
  // MID sits below center: the component sum's worst-case upward excursion is larger
  // than its downward one (measured across all 16 depths), and the legend needs y<30.
  const W = 900, H = 240, MID = 124, AMP = 78;
  const COMPONENTS = 16;
  const comp = [];
  for (let k = 0; k < COMPONENTS; k++) {
    comp.push({
      amp: 1.0 / (k + 1) ** 1.15,
      freq: 1.5 + k * 1.9,
      phase: (k * 2.399963) % (2 * Math.PI), // golden-angle spacing, deterministic
    });
  }
  const totalEnergy = comp.reduce((s, c) => s + c.amp * c.amp, 0);
  const sample = (t, depth) => {
    let v = 0;
    for (let k = 0; k < depth; k++) v += comp[k].amp * Math.sin(2 * Math.PI * comp[k].freq * t + comp[k].phase);
    return v;
  };
  const norm = 1 / comp.reduce((s, c) => s + c.amp, 0) * 1.55;
  const path = (depth) => {
    const pts = [];
    const n = 220;
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      pts.push(`${(t * W).toFixed(1)},${(MID - sample(t, depth) * AMP * norm).toFixed(1)}`);
    }
    return pts.join(" ");
  };

  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Waveform rebuilt from a growing number of residual codes");
  svg.style.width = "100%";
  svg.style.height = "auto";

  const target = document.createElementNS(NS, "polyline");
  target.setAttribute("points", path(COMPONENTS));
  target.setAttribute("fill", "none");
  target.setAttribute("stroke", "#475569");
  target.setAttribute("stroke-width", "1.5");
  target.setAttribute("stroke-dasharray", "5 5");
  svg.appendChild(target);

  const recon = document.createElementNS(NS, "polyline");
  recon.setAttribute("fill", "none");
  recon.setAttribute("stroke", "#34d399");
  recon.setAttribute("stroke-width", "2.5");
  recon.setAttribute("stroke-linejoin", "round");
  svg.appendChild(recon);

  const legend = document.createElementNS(NS, "text");
  legend.setAttribute("x", 10);
  legend.setAttribute("y", 22);
  legend.setAttribute("text-anchor", "start");
  legend.setAttribute("fill", "#94a3b8");
  legend.setAttribute("font-size", "13.5");
  legend.setAttribute("font-family", "Inter, sans-serif");
  legend.textContent = "dashed: the target · solid: what N codes rebuild";
  svg.appendChild(legend);

  const controls = document.createElement("div");
  controls.className = "pg-rvq-controls";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "1";
  slider.max = String(COMPONENTS);
  slider.value = "3";
  slider.className = "pg-slider";
  slider.setAttribute("aria-label", "Number of residual codes used");
  const readout = document.createElement("div");
  readout.className = "pg-rvq-readout";

  const update = () => {
    const depth = Number.parseInt(slider.value, 10);
    recon.setAttribute("points", path(depth));
    const used = comp.slice(0, depth).reduce((s, c) => s + c.amp * c.amp, 0);
    const residual = Math.max(0, (1 - used / totalEnergy) * 100);
    readout.innerHTML =
      `<span class="pg-rvq-big">${depth}</span><span class="pg-stat-dim">/16 codes</span>` +
      `<span class="pg-rvq-res">residual: ${residual.toFixed(1)}%</span>`;
  };
  slider.addEventListener("input", update);

  controls.append(slider, readout);
  rvq.append(svg, controls);
  update();
}

if (rvq) lazyBuild(rvq, buildRvq);

/* ---------------------------------------------------- speed ladder bars */
// Measured real-time factors, linear scale, with the 1x line marked. Bars
// animate to width on reveal; reduced motion renders them at rest.

const ladder = document.getElementById("speed-ladder");

function buildLadder() {
  const MAX = 1.8; // x real time, right edge of the scale
  const rows = [
    { name: "f32 reference", value: 0.15, label: "≈0.15×", cls: "pg-lad-ref" },
    { name: "int8 + team", value: 1.5, label: "1.4–1.6×", cls: "" },
    { name: "browser wasm", value: 0.31, label: "0.31×", cls: "pg-lad-ref" },
  ];
  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "pg-bar-row";
    const name = document.createElement("span");
    name.className = "pg-bar-name";
    name.textContent = row.name;
    const track = document.createElement("div");
    track.className = "pg-bar-track pg-lad-track";
    const fill = document.createElement("div");
    fill.className = `pg-bar-fill ${row.cls}`.trim();
    fill.dataset.width = `${(row.value / MAX) * 100}%`;
    fill.style.width = "0";
    const marker = document.createElement("div");
    marker.className = "pg-lad-marker";
    marker.style.left = `${(1.0 / MAX) * 100}%`;
    track.append(fill, marker);
    const val = document.createElement("span");
    val.className = "pg-bar-val pg-lad-val";
    val.textContent = row.label;
    line.append(name, track, val);
    ladder.appendChild(line);
  }
  const key = document.createElement("p");
  key.className = "pg-note";
  key.style.marginTop = "0.6rem";
  key.innerHTML = 'The <span style="color:#fbbf24">amber line</span> is 1× real time: audio produced exactly as fast as it plays. ' +
    'The f32 route is 6–7× slower by design. The browser figure is a six-partition wasm build with a packed GEMM.';
  ladder.appendChild(key);

  onceVisible(ladder, () => {
    for (const bar of ladder.querySelectorAll("[data-width]")) {
      bar.style.transition = reducedMotion ? "none" : "width 1.2s cubic-bezier(.2,.8,.2,1)";
      requestAnimationFrame(() => { bar.style.width = bar.dataset.width; });
    }
  });
}

if (ladder) buildLadder();

/* ----------------------------------------------- wasm debugging stepper */
// Four real problems, in the order they were hit. Each step: what broke, what the
// diagnosis showed, what fixed it, with the measured before/after where one exists.

const stepper = document.getElementById("wasm-stepper");

function buildStepper() {
  const STEPS = [
    {
      title: "The 1 GB ceiling",
      stat: "1 GB → 4 GB",
      symptom: "The engine died before finishing its first load. The model artifact alone is 2 GB.",
      diagnosis: "rustc caps a wasm module's linear memory at 1 GB unless told otherwise. The " +
        "runtime ceiling for wasm32 is 4 GB, but nothing asks for it by default.",
      fix: "Link with the memory maximum raised to the full 4 GB and budget everything else " +
        "against what remains after the model and its working set.",
    },
    {
      title: "The 1.24 GB gather",
      stat: "1.24 GB → under 1 MB",
      symptom: "First synthesis crashed with an opaque RuntimeError: unreachable near 3.5 GB resident.",
      diagnosis: "Every synthesis materialized a text-embedding buffer strided for the full " +
        "151,936-token vocabulary: 1.24 GB for a one-line prompt, against a memory model where " +
        "every allocated page is committed for real.",
      fix: "Store only the rows the prompt actually uses and find them by binary search over " +
        "sorted token ids. For a typical sentence that is under a megabyte, and the result was " +
        "verified against the reference model's own embedding output.",
    },
    {
      title: "The clock that isn't there",
      stat: "Instant::now() → fixed plan",
      symptom: "Still RuntimeError: unreachable, now with plenty of free memory. Same button, same crash.",
      diagnosis: "A panic hook that routes Rust panics to the console showed the real message: " +
        "wasm's standard library has no monotonic clock, and the kernel autotuner benchmarks " +
        "candidate kernels by timing them at load.",
      fix: "On wasm the autotuner is skipped entirely in favor of a precomputed kernel plan. " +
        "The whole pipeline was then re-run in Node against the real 2 GB model to prove the " +
        "browser path clean end to end.",
    },
    {
      title: "The 71% dot product",
      stat: "codec 119.9 s → 9.5 s",
      symptom: "It finally spoke, at 0.012× real time. Four seconds of audio took over five minutes.",
      diagnosis: "A V8 profile plus wasm disassembly pinned 71% of synthesis time on one f32 " +
        "dot-product routine. Floating-point addition cannot be reordered by the compiler, so " +
        "a single running sum compiles to a scalar chain no SIMD unit can help.",
      fix: "Restructure the sum into eight independent partial lanes the compiler can " +
        "vectorize, and route the codec through an int8 arm whose integer sums reorder freely. " +
        "Codec decode fell from 119.9 s to 9.5 s.",
    },
    {
      title: "The codec was 92% of the frame",
      stat: "codec 89.1 s → 6.8 s",
      symptom: "Threads landed and nothing got faster. The kernel team covered the talker and " +
        "microdecoder, which together account for 8% of the time.",
      diagnosis: "Per-stage timing in a real browser put 89.1 s of a 97.3 s frame in the codec. " +
        "Off macOS there is no BLAS, so every convolution and projection fell through to a " +
        "dot product per output element: the activation row re-read once per column, no weight " +
        "reused, five worker threads parked throughout.",
      fix: "A register-tiled GEMM with packed weight panels, holding an MR×NR output tile in " +
        "registers while one k-panel streams past, then partitioned across six threads by " +
        "output column. Every codec matmul reaches one function, so both changes landed " +
        "everywhere at once. Each element still sums in ascending k, so the result is " +
        "bit-identical to the scalar reference.",
    },
    {
      title: "One message, silently dropped",
      stat: "every browser, no error",
      symptom: "The page downloaded 1.86 GB, printed one hydration stage, and sat there. " +
        "Safari threw a null-property error instead. Neither had a stack trace worth reading.",
      diagnosis: "A module worker starts its event loop while the module is still evaluating. " +
        "Splitting the build for iOS added a top-level await, and the init message posted at " +
        "construction landed in that window, dispatched to no listener and gone. Messages " +
        "sent later arrived normally, so the engine only ever saw the second one.",
      fix: "Install the handler before the first await and buffer until the glue is bound. " +
        "Found by printing every message type on arrival, after five wrong theories: the log " +
        "read msg:load and nothing else.",
    },
  ];

  let index = 0;
  const header = document.createElement("div");
  header.className = "pg-step-header";
  const counter = document.createElement("span");
  counter.className = "pg-step-counter";
  const title = document.createElement("span");
  title.className = "pg-step-title";
  const stat = document.createElement("span");
  stat.className = "pg-step-stat";
  header.append(counter, title, stat);

  const body = document.createElement("div");
  body.className = "pg-step-body";
  body.setAttribute("aria-live", "polite");

  const nav = document.createElement("div");
  nav.className = "pg-step-nav";
  const prev = document.createElement("button");
  prev.className = "pg-btn";
  prev.textContent = "← Previous";
  const dots = document.createElement("div");
  dots.className = "pg-step-dots";
  const dotEls = STEPS.map((_, i) => {
    const dot = document.createElement("button");
    dot.className = "pg-step-dot";
    dot.setAttribute("aria-label", `Step ${i + 1}: ${STEPS[i].title}`);
    dot.addEventListener("click", () => render(i));
    dots.appendChild(dot);
    return dot;
  });
  const next = document.createElement("button");
  next.className = "pg-btn primary";
  next.textContent = "Next →";
  nav.append(prev, dots, next);

  const row = (kind, text) => `
    <div class="pg-step-row">
      <span class="pg-step-kind pg-step-kind-${kind.toLowerCase()}">${kind}</span>
      <span class="pg-note">${text}</span>
    </div>`;

  function render(i) {
    index = i;
    counter.textContent = `${i + 1} / ${STEPS.length}`;
    title.textContent = STEPS[i].title;
    stat.textContent = STEPS[i].stat;
    body.innerHTML =
      row("Symptom", STEPS[i].symptom) +
      row("Diagnosis", STEPS[i].diagnosis) +
      row("Fix", STEPS[i].fix);
    prev.disabled = i === 0;
    next.disabled = i === STEPS.length - 1;
    dotEls.forEach((d, j) => d.classList.toggle("active", j === i));
  }
  prev.addEventListener("click", () => index > 0 && render(index - 1));
  next.addEventListener("click", () => index < STEPS.length - 1 && render(index + 1));
  stepper.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" && index > 0) { event.preventDefault(); render(index - 1); }
    if (event.key === "ArrowRight" && index < STEPS.length - 1) { event.preventDefault(); render(index + 1); }
  });

  stepper.append(header, body, nav);
  render(0);
}

if (stepper) buildStepper();

/* ------------------------------------------------- RMS juxtaposition bars */

const rms = document.getElementById("rms-bars");
if (rms) {
  onceVisible(rms, () => {
    for (const bar of rms.querySelectorAll("[data-width]")) {
      bar.style.transition = reducedMotion ? "none" : "width 1.2s cubic-bezier(.2,.8,.2,1)";
      requestAnimationFrame(() => { bar.style.width = bar.dataset.width; });
    }
  });
}
