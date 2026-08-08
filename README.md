# franken_tts

**A pure-Rust, memory-safe, CPU-only runtime for one text-to-speech model: [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base). Zero-shot voice cloning with no Python, no ML framework, and no GPU at inference.**

[![License: MIT + rider](https://img.shields.io/badge/license-MIT%20%2B%20rider-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/Dicklesworthstone/franken_tts/releases)

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.sh | bash
```

Sibling of [franken_ocr](https://github.com/Dicklesworthstone/franken_ocr) and franken_whisper: one fixed model revision, model-specific kernels, and no pretense of being a general speech framework.

---

## TL;DR

**The problem.** Running a modern voice-cloning TTS model normally means dragging in Python, PyTorch, and a GPU for a 0.6B-parameter model. And Qwen3-TTS's public story of "a 12.5 Hz model" hides its real CPU cost: a **15-step autoregressive residual-code microdecoder inside every 80 ms frame**. The 28-layer talker runs once, then a 5-layer code predictor runs *fifteen sequential times* (per-depth embeddings, per-depth 2,048-way heads), then a causal codec decodes. First-order Q8 weight traffic is ≈1.65 GB per frame; the microdecoder body accounts for ~1.18 GB of it, reread 15×.

**The solution.** `franken_tts` reimplements the whole pipeline in safe Rust, verified seam-by-seam against the upstream PyTorch model, and treats that hidden microdecoder as the primary optimization target rather than a liability: cache-resident hot-packing, per-depth quantization, and **FrankenMTP**, speculative block drafting that the 5-layer predictor itself verifies exactly in a single causal pass. Those three are the roadmap; the f32 reference engine ships today.

| Why franken_tts | |
|---|---|
| **One binary, no runtime deps** | `ftts say "text" --voice v.spk -o out.wav`. No Python, no PyTorch, no CUDA. |
| **Memory-safe** | Pure Rust, end to end. |
| **Verified against the oracle** | Argmax-exact talker parity vs oracle through all 28 layers; whole-utterance codec codes exact vs oracle; audio envelope checked against the pinned PyTorch reference. |
| **Deterministic streaming** | Codec streaming output is bit-identical to offline decoding under every packet schedule. |
| **Agent-first** | `ftts robot schema` emits NDJSON events with stable exit codes. |

## Quick example

```bash
# Fetch the pinned model snapshot from Hugging Face (weights are not bundled)
# into a directory of your choice, then:

FTTS_MAX_FRAMES=120 ftts say \
  "Now is the time for all good men to come to the aid of the agents" \
  --model /path/to/qwen3-tts-12hz-0.6b-base \
  --voice voice.spk \
  -o out.wav
```

That produces real cloned speech on Apple Silicon today. `--voice` takes a 1,024-float x-vector file; enrollment from reference audio is in progress.

## Status: v0.1.0, a working f32 reference engine

This is the first release. What works now:

- **Real speech out.** `ftts say` synthesizes complete utterances on Apple Silicon with the production sampling stack matching upstream runtime defaults: do_sample, temperature 0.9, top-k 50, repetition penalty 1.05, sampled subtalker.
- **Parity receipts.**
  - Talker: argmax-exact vs captured oracle activations through all 28 layers.
  - Codec: whole-utterance codes exact vs oracle at the engine level.
  - Audio: peak frame RMS 0.08578 vs 0.0859745 for the pinned PyTorch reference.
  - Streaming: codec streaming == offline, bit-identical, under all packet schedules.
- **Honest performance.** A 55-frame utterance takes ~60 s wall in a release build (~0.5 s/frame, plus a one-time ~30 s model load). That is 6–7× slower than real time, and deliberately so: v0.1.0 is the unoptimized f32 reference. The quantization and hand-kernel phases (int8 SDOT/VNNI) are the roadmap to >1× real time.

## Verification

Every seam of the pipeline is checked against captured ground truth, not judged by ear:

- **Pinned truth pack** with SHA-256 manifests for model weights and oracle captures.
- **Teacher-forced parity** against captured oracle activations at every seam of the graph.
- **Tolerances derived from the measured nondeterminism floor**, not picked to make tests pass.
- **DISCREPANCIES / NEGATIVE_EVIDENCE / PERF ledgers** recording what disagrees, what was ruled out, and what it costs.
- **Skip-honest test receipts**: a skipped test says why, on the record.

## Installation

**Homebrew (macOS/Linux):**

```bash
brew install dicklesworthstone/tap/franken-tts
```

**Install script:**

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.sh | bash
```

**Cargo:**

```bash
cargo +nightly install ftts-cli --locked   # installs the `ftts` and `franken_tts` binaries
```

Prebuilt binaries cover macOS (arm64, x86_64), Linux (x86_64, arm64), and Windows (x86_64).

### Getting the model

Weights are **not** bundled. Download the pinned Qwen3-TTS-12Hz-0.6B-Base snapshot from [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (Apache-2.0) into a directory, then point the CLI at it with `ftts say --model <dir>`.

## Quick start

1. Install `ftts` (above).
2. Download the pinned model snapshot into `<model-dir>`.
3. Get a voice: `--voice` takes a 1,024-float x-vector file (`.spk`). Enrollment from reference audio is in progress; for now, bring your own x-vector.
4. Synthesize:

   ```bash
   FTTS_MAX_FRAMES=120 ftts say "Hello from safe Rust" \
     --model <model-dir> --voice voice.spk -o hello.wav
   ```

## Robot mode

`franken_tts` is built agent-first. `ftts robot schema` prints the NDJSON event schema; synthesis runs emit machine-readable events and exit with stable, documented codes, so orchestrators and coding agents can drive it without scraping human-facing text.

## Architecture

```
text ──► 28-layer talker (runs once per 80 ms frame)
              │
              ▼
     5-layer code predictor ── runs 15× per frame
     (per-depth embeddings,    (autoregressive residual
      per-depth 2,048-way       -code microdecoder)
      heads)
              │
              ▼
       causal codec decoder ──► 24 kHz WAV
```

Workspace crates: `ftts-core` (engine), `ftts-model-qwen` (model graph), `ftts-kernels` (compute kernels), `ftts-artifacts` (weight/artifact handling), `ftts-conformance` (oracle parity suites), `ftts-cli` (the `ftts` binary).

## Known limitations

- **6–7× slower than real time** on M-series today. v0.1.0 is the f32 reference engine; speed comes from the quantization and kernel phases.
- **f32 only.** No quantized artifacts ship yet.
- **Enrollment pipeline in progress.** You must supply your own 1,024-float x-vector today.
- **Voice quality depends on the reference voice** you enroll.
- **EOS stop timing is sampling-dependent.** Cap generation with `FTTS_MAX_FRAMES`.

## Responsible use

Voice cloning is dual-use. This project records consent attestation and provenance in voice packs, ships no audio-acquisition features, preserves any upstream watermarking, and treats identity claims as settled by blind listening — not embedding cosines. Clone voices you have the right to clone.

## About Contributions

*About Contributions:* Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

Runtime code: MIT with an OpenAI/Anthropic rider (see [LICENSE](LICENSE)). Model weights: Apache-2.0 by Qwen; the Apache attribution text is embedded in the binary and preserved in converted artifacts.
