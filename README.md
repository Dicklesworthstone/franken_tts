# franken_tts

![franken_tts](assets/frankentts_illustration.webp)

**A pure-Rust, memory-safe, CPU-only runtime for one text-to-speech model: [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base). Zero-shot voice cloning with no Python, no ML framework, and no GPU at inference.**

[![License: MIT + rider](https://img.shields.io/badge/license-MIT%20%2B%20rider-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-nightly-orange.svg)](https://www.rust-lang.org/)
[![Version](https://img.shields.io/badge/version-0.1.9-green.svg)](https://github.com/Dicklesworthstone/franken_tts/releases)

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.sh | bash

# macOS / Linux, via Homebrew
brew install dicklesworthstone/tap/franken-tts
```

```powershell
# Windows
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.ps1"))) -EasyMode
```

Sibling of [franken_ocr](https://github.com/Dicklesworthstone/franken_ocr) and franken_whisper: one fixed model revision, model-specific kernels, and no pretense of being a general speech framework.

---

## TL;DR

**The problem.** Running a modern voice-cloning TTS model normally means dragging in Python, PyTorch, and a GPU for a 0.6B-parameter model. And Qwen3-TTS's public story of "a 12.5 Hz model" hides its real CPU cost: a **15-step autoregressive residual-code microdecoder inside every 80 ms frame**. The 28-layer talker runs once, then a 5-layer code predictor runs *fifteen sequential times* (per-depth embeddings, per-depth 2,048-way heads), then a causal codec decodes. First-order Q8 weight traffic is ≈1.65 GB per frame; the microdecoder body accounts for ~1.18 GB of it, reread 15×.

**The solution.** `franken_tts` reimplements the whole pipeline in safe Rust, verified seam-by-seam against the upstream PyTorch model, and treats that hidden microdecoder as the primary optimization target rather than a liability: cache-resident hot-packing, per-depth quantization, and **FrankenMTP**, speculative block drafting that the 5-layer predictor itself verifies exactly in a single causal pass. Per-depth quantization ships today as the default int8 route. The other two are still ahead: the cache-resident hot pack is designed but unimplemented, and FrankenMTP's drafter and block-verification primitives exist in-tree without being wired into the generation path. The bit-exact f32 reference engine stays one environment variable away.

| Why franken_tts | |
|---|---|
| **Faster than real time on CPU** | Speech synthesizes faster than it plays on a Mac Mini M4 Pro — no GPU, and the biggest optimization is still ahead. |
| **One binary, no runtime deps** | `ftts say "text" --voice v.spk -o out.wav`. No Python, no PyTorch, no CUDA. |
| **Memory-safe** | Pure Rust, end to end. |
| **Verified against the oracle** | Argmax-exact talker parity vs oracle through all 28 layers; whole-utterance codec codes exact vs oracle; audio envelope checked against the pinned PyTorch reference. |
| **Deterministic streaming** | Codec streaming output is bit-identical to offline decoding under every packet schedule. |
| **Agent-first** | `ftts robot schema` emits NDJSON events with stable exit codes. |

## Quick example

Three commands, no configuration:

```bash
ftts pull                              # one-time: fetch the quantized model (~2.0 GB, SHA-256 verified)
ftts say "Now is the time for all good men to come to the aid of the agents" out.m4a
```

That speaks immediately with **matt**, the built-in default voice. Seven built-ins ship in the binary — `matt` (the default), `james`, `leo`, and `robert` (masculine), plus `judy`, `aria`, and `ember` (feminine) — selectable by name with `--voice`:

```bash
ftts say --voice james "Uh, excuse me, I was here first." test.m4a && afplay test.m4a
ftts say --voice judy  "Same sentence, different voice."   judy.m4a && afplay judy.m4a
```

Note that `--voice` is an option of `ftts say` (two dashes, before the text), not of your audio player. Clone your own voice from any recording you have the right to use, and it becomes the default:

```bash
ftts enroll voice_memo.m4a --default   # your voice replaces the built-in default
ftts say "Hello in my own voice" hello.m4a
ftts say --voice matt "And back to a built-in one" matt.m4a
```

The built-ins are ordinary enrolled x-vectors, each approved by listening before it shipped. The model installs into `~/.cache/franken_tts/model` and every command finds it there automatically; `--model` and `FTTS_MODEL_DIR` remain available to point elsewhere.

The output format follows the extension. `.wav` comes straight from the built-in pure-Rust encoder; `.m4a`, `.mp3`, and `.flac` are converted from that WAV by whichever system encoder is present (`afconvert` on macOS, `ffmpeg`, `lame`, `flac`), and if none is found you get an error naming the tools rather than a silently different format. Generation stops at the model's EOS, with a text-proportional frame cap as a backstop; set `FTTS_MAX_FRAMES` only when you want an exact cap. `--model`, `--voice`, and `-o` remain available for explicit control.

### Share videos: `ftts make-video`

One command turns a sentence into a share-ready 1080p video — the FrankenTTS artwork with a slow Ken Burns drift, a live waveform of the speech, and the voice's name on a pill:

```bash
ftts make-video "FrankenTTS renders its own share videos now." demo.mp4 --voice aria
ftts make-video --audio existing.wav --label "My Voice" clip.mp4   # render audio you already have
```

Every frame is drawn by memory-safe Rust inside the binary — the embedded illustration (QOI), the waveform, and the text (IBM Plex via `fmd-font`, rasterized in-process). Only the final H.264+AAC encode uses a system encoder, under the same contract as `.m4a` output: `.mp4` needs `ffmpeg`, while a `.y4m` output path renders natively with a `.wav` beside it and needs no encoder at all. Custom voices work exactly as in `say`, so a clip of your own cloned voice carries the project's look when you share it.

### Voice cards: `ftts card`

A voice card is a picture that carries the voice itself. The green mosaic in the image IS the 1,024-float speaker embedding, written at two bits per cell across a 144×144 grid with QR-style finder patterns and interleaved Reed-Solomon error correction, so the card survives screenshots and messaging-app recompression; a lossless PNG chunk in the same file is used first when the bytes arrive intact. The format is shared with the iOS app — a card exported here imports on a phone from Photos, and a card shared from a phone imports here:

```bash
ftts card export aria -o aria-card.png          # any preset, or a .spk file
ftts card export my_voice.spk --name "Jeff"     # name is written into the card
ftts card import aria-card.png                  # writes aria.spk next to the image
ftts say "Hello from a picture." --voice aria-card.png   # or skip the import entirely
```

The two encoders are bit-identical by pinned test (`crates/ftts-voicecard`), and import decodes PNG and JPEG. Cards hold only the small voiceprint, never a recording; share only voices that are yours to share.

### Warm starts: the resident engine

The first `ftts say` of a session pays the model load; when it finishes, the loaded model stays behind in a small helper process, and the next `say` starts synthesizing at once instead of spending seconds reloading weights. The helper is spawned from the same binary, listens only on a loopback port recorded in a file that only your user can read, keeps no record of what it spoke, and exits on its own after ten minutes without a request (`FTTS_RESIDENT_IDLE_SECS` adjusts the window). Output is byte-identical with and without it, which the e2e suite asserts; a re-pulled model or an upgraded binary retires the old helper automatically. Opt out per run with `--no-resident`, or globally with `FTTS_NO_RESIDENT=1`.

## The iOS app

`ios/` holds a native SwiftUI version of the playground, built on the same Rust engine
through a small C ABI (`crates/ftts-ffi`): on-device model download with pinned-digest
verification, the seven preset voices, microphone enrollment, and on-device synthesis.
Design, phone-vs-Mac arithmetic, and memory strategy live in `docs/IOS_APP_PLAN.md`;
build instructions in `ios/README.md`. No speed figure is claimed until one is measured
on A18-class hardware.

## Try it in your browser

**[frankentts.com](https://frankentts.com)** (also [frankentts.pages.dev](https://frankentts.pages.dev)) runs the whole pipeline as WebAssembly — type, pick a voice, clone your own by reading a half-minute script into the mic, listen, download. Nothing you type or record leaves the tab: the model downloads once (~1.77 GB — the artifact plus only the decoder half of the codec checkpoint; the encoder side is enrollment weight the browser never reads), resumable and SHA-256-verified into browser storage, and inference is local. Honest speed note: the single-threaded wasm build ran ~0.2–0.3× real time; with codec decode fanned across a worker team the spike receipt measured ~0.4× (`frankentts-kkuq`), and this repo's own Chromium gate currently measures ≈0.5× on the threaded build — a five-second line takes about ten seconds with a progress bar. Browser threads everywhere + relaxed-SIMD int8 dot are the tracked path to real time. Desktop browsers only (~3.5 GB peak memory). The site lives in `site/`; the bindings in `cra…

## Status: faster than real time, realtime-capable end to end

The release log: v0.1.0 shipped the f32 reference engine, v0.1.1 made the model a one-command download, v0.1.2 moved that download to a pre-quantized artifact, v0.1.3 made the optimized int8 route the library-wide default, v0.1.4 crossed real time and shipped seven built-in voices, and v0.1.5 landed the realtime conversation layer — `ftts talk` streaming sessions with continuation, live PCM emission during synthesis, and a proven cancellation contract (SIGINT yields the cancelled exit code 6 with the partial audio already delivered: `say` emits a `run_error{kind:"cancelled"}` terminal event; `talk` settles the in-flight utterance with its `speak_cancelled` receipt; a second signal force-exits). Releases v0.1.6–v0.1.8 added voice cards, browser download resilience, and grouped-Q8 experiments — see the releases page for the running list. v0.1.9 (2026-08-25) shipped the post-v0.1.8 backlog — ICL QUALITY voice-pack say-side consumption groundwork, metamorphic/prosody conformance harnesses, HF-mirror download-resilience refinements — cut from green main after CI-runner starvation was root-caused and remedied. What works now:

- **Real speech out.** `ftts say` synthesizes complete utterances on Apple Silicon with the production sampling stack matching upstream runtime defaults: do_sample, temperature 0.9, top-k 50, repetition penalty 1.05, sampled subtalker.
- **Parity receipts.**
  - Talker: argmax-exact vs captured oracle activations through all 28 layers.
  - Codec: whole-utterance codes exact vs oracle at the engine level.
  - Audio: peak frame RMS 0.08578 vs 0.0859745 for the pinned PyTorch reference.
  - Streaming: codec streaming == offline, bit-identical, under all packet schedules.
- **Faster than real time on a Mac Mini M4 Pro, and getting faster quickly.** The optimized int8 route (below) is the default everywhere, library included — owner-side runs typically measure 1.4–1.6× real time on an unloaded machine (heavily loaded runs have measured 0.66–1.05×); no certified RTF row exists yet in `docs/PERF_LEDGER.md`, and first audio measures 200–225 ms after synthesis starts in the latest bench (`ttfa_ms` in robot output; PERF-007, indicative pending quiet-window re-certification). Warm model load is ~3.7 s certified (PERF-003), down from ~9 s plus a hidden requantize in v0.1.3, by widening tensors concurrently, hydrating int8 weights natively from the artifact, and loading the codec and tokenizer in parallel with the talker. Short one-shot utterances are still dominated by that load — judge throughput on a real paragraph, not on `"Hi."`. Headroom remains: the microdecoder's cache-resident hot pack and p…

### The optimized route (the default in the `ftts` binary)

Since v0.1.3 the quantized stack is the library-wide default (`ftts_kernels::route::optimized_default`), not just a CLI setting, so library consumers get the same speed path as the binary; conformance and oracle entry points pin the f32 reference route explicitly, so parity suites never measure the optimized numerics. `FTTS_INT8=0` is the master switch back to the bit-exact f32 reference. What the default route runs, with its measured evidence (one M4 Pro, this tree's own f32 reference as the comparison — local evidence, not a cross-project benchmark):

- **W8A8 int8 talker + microdecoder** (symmetric per-channel weights, dynamic per-row activations, exact i32 accumulation), fanned across a six-thread persistent worker team whose partitioning is bit-identical to serial execution at every thread count.
- **A BLAS-form f32 codec** that runs the reference's own reduction order through Accelerate: measured faster than the int8 codec arm while tracking the oracle more closely, so the codec stays full-precision by default and the int8 codec arms (`FTTS_INT8_CODEC=convnext|all`) remain opt-in A/B routes. Codec decode is pipelined with generation, and streamed output is bit-identical to offline decode.
- **SLEEF SnakeBeta** (129.6 dB SNR against libm — inaudible by a wide margin) and a startup autotuner that picks the fastest proven kernel tier per regime, cached in `~/.cache/franken_tts/`.
- Every int8 kernel tier is proven exactly equal to the scalar reference in i32 by `ftts robot selftest`, on your machine, at this model's real reduction lengths.

What the default route does not promise yet: sampled outputs are *different valid renditions*, not the f32 waveform — with sixteen sampled draws per frame, any lossy weight change alters the token stream within a frame or two (measured). Objective spectral checks pass for the codec and SnakeBeta pieces; a listening-based evaluation of the full sampled path against the reference is still open (`docs/DISCREPANCIES.md`, DISC-003), which is why the reference route stays one environment variable away.

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

**Install script (macOS/Linux):**

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.sh | bash
```

**Windows (PowerShell):**

```powershell
& ([scriptblock]::Create((irm "https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.ps1"))) -EasyMode
```

Downloads the release zip, verifies it against the release's own `SHA256SUMS`, installs both
binaries under `%LOCALAPPDATA%\Programs\franken_tts\bin`, and with `-EasyMode` adds that to your
user PATH (no administrator rights needed). `-Version`, `-InstallDir`, `-Force` and `-Quiet` are
also accepted.

**Cargo:**

```bash
cargo +nightly install ftts-cli --locked   # installs the `ftts` and `franken_tts` binaries
```

Prebuilt binaries cover macOS (arm64, x86_64), Linux (x86_64, arm64), and Windows (x86_64).

### Getting the model

Weights are not bundled with the binary; `ftts pull` fetches this project's pre-quantized `.fttsq` artifact plus the codec checkpoint and tokenizer files (~2.0 GB total, derived from Qwen3-TTS-12Hz-0.6B-Base, Apache-2.0 by Qwen) from the [model release](https://github.com/Dicklesworthstone/franken_tts/releases/tag/model-qwen3-tts-v1) into `~/.cache/franken_tts/model`, verifying every file's SHA-256 against the manifest embedded in the binary. The artifact packs the talker's hot projections as int8 per-output-channel and everything else verbatim, so it is 30% smaller than the raw checkpoint and is the same file `ftts convert` produces. Re-running `ftts pull` verifies and skips files already present. You can instead download the same snapshot from [Hugging Face](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) manually and point the CLI at it with `--model <dir>` or `FTTS_MODEL_DIR`.

## Cloning your voice, step by step

1. Install `ftts` (above), then fetch the model once: `ftts pull`.
2. **Record about 30 to 50 seconds of yourself reading the passage below**, in a quiet room, at your natural pace, on any device — a phone voice memo is fine. Any common format works (`.m4a`, `.mp3`, `.wav`, `.flac`).
3. Enroll it as your default voice:

   ```bash
   ftts enroll my_recording.m4a --default
   ```

   Enrollment computes a 1,024-float speaker embedding from the audio alone — **no transcript is needed**. Any container at any sample rate works — 44.1 and 48 kHz phone and Mac voice memos included. Compressed references are converted by the system decoder; `.wav` and `.flac` are read directly and resampled in-process by a windowed-sinc (Lanczos-6) kernel. Audio already at 24 kHz is passed through untouched. The enroll step warns about recordings that will clone poorly (clipping, whispering, multiple speakers). Background noise is handled for you: enrollment automatically denoises the reference with a neural model before the embedding is computed. The difference this makes is dramatic. An ordinary iPhone voice memo, recorded in a normal room with a fan and a laptop nearby, enrolls sounding like it was captured in a treated studio; there is no audible noise floor at all, and because the clone inherits its sound from the enrollment, every utterance it ever speaks keeps that cleanliness. Pass `--no-denoise` to enroll a recording exactly as given.

   **How the denoiser works.** The engine is a pure-Rust port of FastEnhancer-S (ICASSP 2026), a 207,000-parameter speech-enhancement network whose 0.8 MB weight file rides along with `ftts pull`. It works on the spectrogram: the recording is resampled to the network's native 48 kHz, split into overlapping 1,024-sample frames, and each frame's spectrum is multiplied by a complex ratio mask the network predicts as it listens (small GRUs carry memory across frames; a tiny attention layer compares frequency bands within each frame). Steady noise such as tape hiss, fan rumble, or mic self-noise gets a mask near zero; speech gets a mask near one, phase included. The cleaned spectrum is turned back into audio, resampled to 24 kHz, and only then does enrollment compute the speaker embedding. Since a cloned voice inherits its noise character from that embedding, cleaning the reference once cleans every future utterance spoken in it.

   The port is verified against its PyTorch reference at 114–125 dB SNR on every pinned fixture, which is float rounding rather than approximation ([docs/DENOISER.md](docs/DENOISER.md) has the receipts). On a test reference with white static at 15 dB SNR, the pause floor drops from −51.8 to −117.4 dBFS and the enrolled voice lands measurably closer to a clean-source enrollment than the raw recording does. Cleanup runs by default whenever the weight file is present, which a normal `ftts pull` guarantees. When it has not been pulled, the default skips cleanup rather than substituting a different engine unannounced; passing `--denoise` explicitly engages the in-tree OM-LSA spectral subtraction in that case, which needs no weights, and `FTTS_DENOISE_ENGINE=omlsa` forces that engine outright. `--no-denoise` turns all of it off. The browser playground applies the same automatic cleanup to mic enrollments, with the weights embedded in the wasm module itself.

   Re-enrolling over an existing voice asks for confirmation in a terminal, or proceeds when you pass `--overwrite`; either way the displaced voice is saved to `<name>.spk.bak` and the write is staged-then-renamed so a crash cannot leave a half-written voice. (`--force` is separate: it only proceeds past *quality* warnings.) Use `-o name.spk` instead of `--default` to keep several voices and select one per run with `--voice name.spk`.
4. Speak:

   ```bash
   ftts say "Hello from my cloned voice" hello.m4a
   ```

### The enrollment passage

Any natural speech works, but a phonetically rich passage measurably beats casual filler; this is to voices what "the quick brown fox" is to fonts. The script below is the Speech Accent Archive's **"Please call Stella"** elicitation paragraph, which packs most English phonemes and the discriminative consonant clusters into four sentences, followed by the opening of the **Rainbow Passage** for connected, flowing prosody. The speaker encoder pools over the whole recording with no truncation, but the voice information it extracts saturates well before a minute, so the first few sentences already make a good clone and about thirty seconds of reading is the polished version:

> Please call Stella. Ask her to bring these things with her from the store: six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three red bags, and we will go meet her Wednesday at the train station.
>
> When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.

Keep a note of exactly what you read: the upcoming higher-quality ICL cloning mode conditions on the reference audio *plus its verbatim transcript*, so a recording of a known passage is already future-proof.

## Robot mode

`franken_tts` is built agent-first. `ftts robot schema` prints the NDJSON event schema; synthesis runs emit machine-readable events and exit with stable, documented codes, so orchestrators and coding agents can drive it without scraping human-facing text. Two subcommands report the kernel state of the machine they run on: `ftts robot selftest` executes the integer-overflow proof rows through every dispatchable kernel tier and reports per-row verdicts, and `ftts robot backends` lists available tiers, the dispatched route, detected ISA features, and the autotuned kernel plan.

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

## Documentation

- [`docs/VOICE_COMPILER_DESIGN.md`](docs/VOICE_COMPILER_DESIGN.md) — how enrollment becomes a
  voice: the `.ftvoice` pack, its privacy profiles, and the derived `.ftvoice-cache`.
- [`docs/truth-pack/`](docs/truth-pack/) — pinned sources, oracle provenance, and the
  nondeterminism-floor measurements behind every conformance claim.

## Known limitations

- **Real time is load-dependent.** The default optimized route runs faster than real time on an unloaded Mac Mini M4 Pro; on the same machine saturated with concurrent build jobs it measured 0.66–1.05× real time. The f32 reference route (`FTTS_INT8=0`) runs 6–7× slower than real time by design.
- **The optimized default trades exactness for speed.** Kernel integer math is exactly proven, and the codec and SnakeBeta pieces pass objective spectral gates, but listening-based evaluation of the full sampled path is still open (DISC-003); `FTTS_INT8=0` restores the reference route.
- **Model load costs ~4 s before the first sample.** `ftts pull` ships pre-quantized weights as of v0.1.2, so this is artifact load rather than a quantization pass, but it is still paid once per process — short one-shot utterances are dominated by it.
- **The two biggest optimizations are not built yet.** The microdecoder's 5-layer body is re-read 15× per frame (~1.18 GB of the ~1.65 GB per-frame weight traffic), and neither planned fix has landed: the cache-resident MTP hot pack is designed but unimplemented, and FrankenMTP's drafter and block-verification primitives are in-tree without being wired into the generation path, so decode still runs all 15 steps sequentially.
- **Voice quality depends on the reference voice** you enroll.
- **EOS stop timing is sampling-dependent.** A text-proportional frame cap backstops it by default; set `FTTS_MAX_FRAMES` for an exact cap.
- **Compressed output needs a system encoder.** `.m4a`, `.mp3`, and `.flac` shell out to `afconvert`, `ffmpeg`, `lame`, or `flac`; `.wav` needs nothing.

## Responsible use

Voice cloning is dual-use. This project records consent attestation and provenance in voice packs, ships no audio-acquisition features, preserves any upstream watermarking, and treats identity claims as settled by blind listening, not embedding cosines. Clone voices you have the right to clone.

## About Contributions

*About Contributions:* Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

## License

Runtime code: MIT with an OpenAI/Anthropic rider (see [LICENSE](LICENSE)). Model weights: Apache-2.0 by Qwen; the Apache attribution text is embedded in the binary and preserved in converted artifacts.
