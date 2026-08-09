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
  label(talkerX + talkerW / 2, talkerY + 220, "28 layers · runs once", 11.5, "#64748b");
  label(talkerX + talkerW / 2, talkerY + 238, "predicts code #0", 11.5, "#64748b");
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
  label(microX + microW / 2, microY + 44, "sequential passes", 11.5, "#64748b");
  const microLayers = [];
  for (let i = 0; i < 5; i++) {
    microLayers.push(make("rect", {
      x: microX + 10, y: microY + 60 + i * 15, width: microW - 20, height: 11,
      rx: 3, fill: "rgba(52,211,153,0.12)", stroke: "rgba(52,211,153,0.3)", "stroke-width": 0.7,
    }));
  }
  label(microX + microW / 2, microY + 152, "5 layers, rerun per pass", 11.5, "#64748b");
  label(microX + microW / 2, microY + 196, "the body is reread 15×:", 11.5, "#64748b");
  label(microX + microW / 2, microY + 214, "≈1.18 GB of the frame's traffic", 11.5, "#94a3b8");

  // ---- code slots between micro and codec
  const slotX = 520, slotY = 52;
  label(slotX + 24, slotY - 14, "16 codes", 11.5, "#64748b");
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
  label(codecX + codecW / 2, codecY + 30, "causal decoder", 11.5, "#64748b");
  label(codecX + codecW / 2, codecY + 48, "upsample 8·5·4·3", 11.5, "#64748b");
  const wave = make("polyline", {
    points: "", fill: "none", stroke: "#34d399", "stroke-width": 2, "stroke-linejoin": "round",
    opacity: 0,
  });
  const samplesLabel = label(codecX + codecW / 2, codecY + 218, "", 12.5, "#94a3b8", "700");

  // ---- traffic counter
  const traffic = label(450, 348, "", 14, "#cbd5e1", "700");

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

if (anatomy) {
  new IntersectionObserver((entries, observer) => {
    if (entries.some((e) => e.isIntersecting)) {
      observer.disconnect();
      buildAnatomy();
    }
  }, { rootMargin: "150px" }).observe(anatomy);
}

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
  new IntersectionObserver((entries, observer) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    observer.disconnect();
    const started = performance.now();
    const duration = 1100;
    const step = (now) => {
      const t = Math.min(1, (now - started) / duration);
      const eased = 1 - (1 - t) ** 3;
      el.textContent = (target * eased).toFixed(decimals) + suffix;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }, { threshold: 0.4 }).observe(el);
}

/* ------------------------------------------------- RMS juxtaposition bars */

const rms = document.getElementById("rms-bars");
if (rms) {
  new IntersectionObserver((entries, observer) => {
    if (!entries.some((e) => e.isIntersecting)) return;
    observer.disconnect();
    for (const bar of rms.querySelectorAll("[data-width]")) {
      bar.style.transition = reducedMotion ? "none" : "width 1.2s cubic-bezier(.2,.8,.2,1)";
      requestAnimationFrame(() => { bar.style.width = bar.dataset.width; });
    }
  }, { threshold: 0.4 }).observe(rms);
}
