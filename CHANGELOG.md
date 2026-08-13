# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Scope window: full project history, 2026-08-05 through 2026-08-08 (~240 commits).
Sources: git history, the beads tracker (`.beads/issues.jsonl`, 47 closed
workstreams at release time), and the checked-in truth-pack/ledger docs. As the
first release, v0.1.0 is organized by capability theme rather than by commit;
individual commits are intentionally not cited.

## Version Timeline

| Version | Date | Status | Summary |
|---------|------|--------|---------|
| 0.1.8 | 2026-08-12 | current | model downloads survive GitHub throttling: Hugging Face is the primary mirror everywhere |
| 0.1.7 | 2026-08-12 | superseded | browser memory diet (no double-resident model), native↔browser parity harness, w8a16 multicore |
| 0.1.6 | 2026-08-10 | superseded | voice cards (a picture that IS the voice), phone↔CLI interop; iOS video export unstuck; output denoising |
| 0.1.5 | 2026-08-10 | superseded | browser engine 9.4× faster; iOS survives; enrollment denoises itself; Windows installer |
| 0.1.4 | 2026-08-09 | superseded | faster than real time: worker team, pipelined codec, 2.5× faster startup |
| 0.1.3 | 2026-08-09 | superseded | optimized route becomes the library-wide default; 48 kHz enrollment fix |
| 0.1.2 | 2026-08-08 | superseded | `ftts convert` works on the real checkpoint; `ftts pull` ships the quantized `.fttsq` |
| 0.1.1 | 2026-08-08 | superseded | zero-config UX: `ftts pull`, default model cache, m4a/mp3 enrollment |
| 0.1.0 | 2026-08-08 | first release | f32 reference engine, CLI, conformance ladder, artifact groundwork |

## [Unreleased]

## [0.1.8] - 2026-08-12

The model download stops depending on GitHub's goodwill. Within hours of real
traffic, GitHub's release-asset limiter started returning 503s for the chunked
downloads both `ftts pull` and the browser playground perform — and each had a
single source, so a throttled host meant a dead download.

### Fixed

- **`ftts pull` tries ordered mirrors**: the Hugging Face model repo
  ([Dicklesworthstone/franken-tts-qwen3-tts-12hz-0.6b-base](https://huggingface.co/Dicklesworthstone/franken-tts-qwen3-tts-12hz-0.6b-base))
  first, the GitHub release second. Every asset carries its release name on
  both hosts, digest verification decides acceptance exactly as before, and
  only the last mirror's error surfaces when everything fails.
- The playground's `/model` proxy serves the same chain (HF, then the site's
  R2 bucket, then GitHub), so any single host failing degrades the download
  to a slower one instead of a dead one.

### Added

- `ftts convert --embed-q8` (EXPERIMENTAL, native-only): stores the 622 MB
  cold text embedding as Q8 with one scale per 64-element group, producing a
  1.02 GB artifact instead of 1.31 GB. Grouped scales exist because per-row
  scales measured 23.8 dB SQNR on the most common tokens' rows; groups lift
  the floor to 35.0 dB (zero rows below 30 dB). Off by default until the
  artifact-v2 listening and logit-parity gates pass; binaries at or below
  0.1.7 refuse grouped artifacts loudly by name.

## [0.1.7] - 2026-08-12

The browser stops paying for the model twice, proves its output against the
CLI sample by sample, and the kernel team picks up the last two routes that
were still running single-core.

### Added

- **CLI-golden browser conformance harness**: the playground exposes raw PCM,
  and `site/harness/browser.mjs` compares a real in-browser synthesis against
  a CLI-rendered golden of the same text and voice. The first codec frames are
  mirrored on both sides for token-level triage when samples diverge. The
  measured native↔browser gap is recorded as DISC-006.
- `packet_frames` dial through the wasm synthesize ABI: codec packet size is
  sweepable from the page (default stays 4). Output is bit-identical under
  every schedule — the streaming==batch gate holds — so it is purely a
  speed/memory dial.

### Changed

- **Browser memory diet** (recorded as PERF-006): the codec now streams into
  the engine as widened tensors instead of staging a second safetensors copy,
  the artifact hot prefix is released before codec staging begins, staging is
  decoder-only (the encoder never crosses the wire), codec tensor ingestion
  reuses one buffer instead of 271 allocations, and the page tears down its
  animated diagrams and non-playground chrome while an engine is resident.
  Net effect: the 2 GB model fits comfortably inside the browser tab's budget
  instead of brushing against it.
- The codec derives its transposed-conv column layout once per utterance
  instead of once per frame, and one-shot artifact digests go through `sha2`
  (streaming digests stay ours).
- **Kernel team covers every quantized route**: `FTTS_INT8=w8a16` now fans out
  across the persistent worker team exactly like the default w8a8 route
  (3.6–5.0x per projection at the model's decode shapes in a kernel-level
  A/B, provisional pending a quiet-host run); batched calls (prefill, the
  seq-16 verify) route to the autotuned plan's measured batch-regime winner
  instead of inheriting the single-row winner; and the armed W8A8 quant path
  no longer allocates per call (thread-local scratch, in keeping with the
  no-allocator-activity steady-state doctrine).
- The FrankenMTP speculation track is now labeled unshipped at its code sites,
  cross-referencing the measured negative evidence, so the in-tree primitives
  cannot be mistaken for a realized traffic win.

### Fixed

- **Native↔browser parity**: wasm now uses the native f32 reduction order for
  sample-level parity, and every target ships the same int8 default, so the
  browser and the CLI run the same numerics route out of the box.
- The installer accepts the checksum manifest we actually ship and tolerates
  AppleDouble members in the archive.

## [0.1.6] - 2026-08-10

Voices become pictures: a card exported anywhere imports everywhere, phone and
CLI alike. The iOS video exporter stops deadlocking, synthesis output gets the
same denoiser enrollment uses, and the end-of-utterance trim removes the noise
it was built to remove.

### Added

- **Voice cards**: a picture that carries the voice itself, interchangeable
  between the CLI and the iOS app. The green mosaic in the image is the full
  1,024-float speaker embedding, written at two bits per cell across a 144×144
  grid with QR-style finder patterns, calibration strips, and interleaved
  Reed-Solomon (255,223) error correction, so the card survives screenshots
  and messaging-app recompression; a lossless private PNG chunk in the same
  file is used first when the bytes arrive intact. The codec lives in the new
  `ftts-voicecard` crate, and the CLI and Swift encoders are bit-identical —
  proven byte-for-byte against real artifacts and pinned by a CRC test so the
  implementations cannot drift apart silently.
- `ftts card export <preset|.spk>` renders a voice as a shareable card PNG;
  `ftts card import <image>` reads a card (PNG or JPEG) back into a `.spk`
  file, printing the name written into it.
- `ftts say --voice card.png` (and `make-video --voice`) accept a voice-card
  image directly; no import step needed.
- iOS: enrolled voices export as voice cards (share sheet or straight to
  Photos, with the exact bytes preserved so the lossless layer survives), and
  "Add a voice from a picture" imports one from the photo library. Importing
  the same card twice selects the existing voice instead of duplicating it.
- iOS: the voice constellation draws every voice as a glyph whose shape
  amplifies its difference from the average voice, positioned by
  multidimensional scaling over pairwise similarity, colored from the map
  itself — neighbors share a hue, outliers go vivid.
- iOS: synthesis shows a progress percentage estimated from the text length
  and the last measured speed on the device, plus a "waking the model" state
  for the first run.

### Fixed

- iOS video export no longer sticks partway. Two stacked bugs: feeding all
  video frames before any audio deadlocked `AVAssetWriter`'s interleaving at
  its buffer depth, and behind that, mono AAC at 24 kHz rejects a 96 kbps
  bitrate outright, killing the export the moment the deadlock was cured.
  Audio and video now feed concurrently and the encoder picks its own
  bitrate. Export is also much faster: fixed-point Ken Burns resampling,
  frames rendered in parallel chunks, and the BGRA conversion moved into
  Rust — a 20-second clip that previously never finished exports in about
  15 seconds.
- The end-of-utterance tail trim removed the wrong samples whenever the noise
  burst was followed by inaudible silence: it kept the audible artifact and
  deleted the harmless silence. The detector now reports where the noise run
  sits and the writer removes exactly that window; a content-asserting test
  pins the behavior (sample counts alone cannot distinguish right from wrong
  here). The holdback window also derives from the writer's own sample rate
  instead of a hardcoded 24 kHz.
- iOS kernel worker threads request an elevated QoS class so the scheduler
  stops parking them on efficiency cores; the barrier waits for its slowest
  member, so one demoted worker set the pace of every dispatch.

### Changed

- The FastEnhancer denoiser is now a required model file on iOS: enrollment
  refuses to run without it (a profile built from un-denoised audio carries
  the recording's noise into every synthesis), synthesis output is denoised
  with the same network, and installs that predate the file complete the
  missing 0.8 MB silently at launch.

## [0.1.5] - 2026-08-10

The browser engine gets 9.4x faster, Apple devices stop crashing, and enrollment
cleans its own reference audio.

### Added

- The browser runs at **0.31–0.43x real time**, measured, against 0.05x before.
  Three seconds of speech now costs seconds of compute rather than minutes. Two
  levers did it, and both are bit-identical to the scalar reference rather than
  approximations traded for speed:
  - A register-tiled, panel-packed f32 GEMM (`ftts-kernels/src/packed_gemm.rs`),
    ported from `franken_numpy`'s `fnp-linalg` and adapted to this project's
    `[n, k]` weight layout so no transpose is ever materialized. WebAssembly has
    no BLAS, so every codec convolution and projection had been falling through
    to a dot product per output element. All six codec GEMM sites reach one
    function, so one kernel upgraded every convolution, ConvNeXt pointwise pair
    and transformer projection at once.
  - The codec's dense route now dispatches to the `KernelTeam` across six
    partitions. It had been 92% of frame time running on a single thread while
    every worker sat parked.

  Per-stage, in a real browser against the real model: codec 89.1 s -> 6.8 s
  (13.2x), talker+microdecoder 7.6 s -> 3.4 s (2.2x), total 97.3 s -> 10.35 s.
  Recorded as `PERF-005`, labelled PROVISIONAL_LOCAL_WIN: single runs, unequal
  utterance lengths, no interleaved thermal pairs, and the incumbent is our own
  previous build rather than a pinned external one.

- Symmetric int4 (W4A8) weight quantization for the microdecoder
  (`ftts-kernels/src/int4.rs`): two biased nibbles per byte with the +8 bias
  cancelled by a single correction term, so the inner loop is mask/shift/MAC
  with no per-element sign extension. **Nothing routes to it yet** — doctrine #2
  requires both a per-ISA speed test including unpack cost and a blind-listening
  equivalence test first, and neither has been run. See `NE-005`.

- A real-browser test harness (`site/harness/`): the actual site in actual
  Chromium, with real OPFS, real Workers, real COOP/COEP from the shipped
  `_headers`, and real byte-Range downloads of the real model. No mocks, because
  a previous shim-based Node harness passed every one of its cases while the
  deployed site was broken in every browser. It now runs as a `check.sh` stage.

- Enrollment denoises automatically. Every `ftts enroll` now cleans the reference with
  a neural denoiser before computing the embedding — no flag to remember — and the
  browser playground applies the identical cleanup to mic enrollments, with the weights
  embedded in the wasm module. `ftts say --voice <audio-file>` (the ephemeral one-off
  enrollment) gets the same treatment, so the one-off form no longer sounds worse than
  the saved form; `.spk` vectors and presets never enter the cleanup path. The result
  on an ordinary phone voice memo is a voice that enrolls sounding studio-recorded,
  and the clone keeps that cleanliness in everything it speaks. Opt-outs and levers:
  `--no-denoise` enrolls the recording untouched, an explicit `--denoise` additionally
  engages the classic no-weights spectral subtraction when the neural weights are
  absent (the automatic path skips cleanup rather than swapping engines unannounced),
  and `FTTS_DENOISE_ENGINE=omlsa` forces the classic engine outright.
- The denoiser is a pure-Rust port of FastEnhancer-S 48 kHz (207 K parameters, MIT),
  proven to 114–125 dB SNR parity against its pinned PyTorch reference on every fixture.
  `ftts pull` fetches the 0.8 MB weight artifact (sha256-pinned) alongside the model.
  On a 15 dB-SNR static-hiss reference it lowers the pause floor by 65.5 dB (classic
  spectral subtraction: 24.6 dB) and moves the enrolled x-vector closer to the
  clean-source enrollment (cosine 0.9613 vs 0.9548). The engine is thread-free and
  mmap-free and compiles unchanged for wasm32. See `docs/DENOISER.md` for the truth pack.

### Windows installer and human console output

Windows gets a real installer, and the terminal stops printing NDJSON at people.

### Added

- `install.ps1`: a PowerShell one-liner for Windows, the platform `install.sh`
  documents but explicitly declines to cover. It resolves the latest release,
  downloads the zip, verifies it against the release's own `SHA256SUMS`,
  installs both binaries under `%LOCALAPPDATA%` with no administrator rights,
  and optionally adds them to the user PATH (`-EasyMode`). Validated end to end
  on Windows PowerShell 5.1 against a real published release: download,
  checksum, extract, install, PATH, already-installed short-circuit, and
  `ftts --version` responding.
- `ftts say` renders a human summary when stdout is a terminal — voice, model
  load, frames, destination, and a real-time factor — instead of the NDJSON
  event stream. The machine contract is unchanged for every non-terminal
  consumer, which is the entire point; `--robot` forces NDJSON back on a
  terminal.
- `ftts enroll --dereverb`: blind dereverberation by Weighted Prediction Error,
  for a reference recorded in a live room. Reverb is convolutive, so `--denoise`
  cannot touch it; the speaker encoder cannot separate voice from room, so a wet
  reference enrolls the room as part of the speaker.

### Fixed

- **Apple devices no longer crash loading the model.** Measured on an iPhone 17
  Pro Max with a crash-persistent probe: growing a *shared* wasm memory reclaims
  the tab past ~1 GB, while growing an unshared one to 2.75 GB is fine and flat
  allocations of either kind are fine to 3.5 GB. Rust's allocator grows linear
  memory on every heap request, so a 2 GB model guarantees growth and a threaded
  build could never work there. The site now ships two builds and picks at
  runtime: shared/threaded where it is known safe, unshared/serial on WebKit.
  Two approaches that do *not* work are recorded in `engine-worker.js` so they
  are not retried — pre-reserving (the allocator ignores it and grows on top)
  and pinning `maximum = initial` (the first allocation fails and the module
  aborts).
- **A dropped message bricked the engine in every browser.** A module worker
  starts its event loop while the module is still evaluating, so the `init`
  message posted at construction was dispatched to no listener and lost — never
  queued. Chrome hung inside `ModelStaging`; Safari threw
  `undefined is not an object (evaluating 'wasm.modelstaging_new')`. Same cause,
  two unrecognizable symptoms. The handler is now installed before the first
  `await` and buffers until the wasm glue is bound, and messages are serialized
  so `load` can no longer race `init`.
- **Resident daemon hardening**, three defects, all locally reachable and the
  first needing no token at all: the pre-auth `read_line` was unbounded (a peer
  that never sends a newline grew the buffer until the process died, now capped
  at 1 MiB); speaker vectors were parsed with `filter_map`, silently *dropping*
  non-numeric entries so a malformed request quietly became a well-formed one,
  and non-finite values reached the quantizer's assertion and panicked; and
  `handle_connection` ran bare in the accept loop, so any panic killed the
  process holding the only warm model. All three now refuse or contain.
- `LICENSE` shipped correctly. It had been truncated to zero bytes in the
  working tree; release artifacts are built from a pinned worktree, which is
  what kept the published packages intact.
- Undefined behaviour in the worker team: every worker materialized a `&mut`
  over the whole output buffer. The writes were disjoint but the references
  overlapped, which is UB regardless — and `rustc` marks `&mut` `noalias`, so
  the optimizer was entitled to act on it. Both the linear and attention paths
  now write through raw pointers or a narrowed per-head span.

## [0.1.4] - 2026-08-09

The speed release: synthesis now runs faster than real time on an M4 Pro
(typically 1.4–1.6× on a machine that is also doing other work), model load is
2.5× faster, and the first audio arrives about half a second after synthesis
starts. Every optimization ships with a bit-identity or ledgered-equivalence
proof; the f32 reference route remains one variable away (`FTTS_INT8=0`).

### Added

- **Built-in voices, selectable by name.** A fresh install speaks out of the
  box: `ftts pull` then `ftts say "hello" out.wav` uses `matt`, the default.
  Seven voices ship in the binary — `matt`, `james`, `leo`, `robert`
  (masculine) and `judy`, `aria`, `ember` (feminine) — each an ordinary
  enrolled x-vector approved
  by listening before it shipped, chosen with `--voice NAME`. An enrolled
  `default.spk`, `FTTS_DEFAULT_VOICE`, or explicit `--voice` always wins; an
  unknown bare name lists the built-ins. (A first cut of tone-enrolled voices
  rendered inconsistently across texts and was replaced before release.)

### Performance

- **Persistent int8 worker team.** The talker and microdecoder projection
  matmuls fan out across a pool of persistent workers (six by default,
  `FTTS_INT8_THREADS` to change) with output-column partitioning that is
  bit-identical to serial execution at every thread count. GQA attention now
  partitions across query heads on the same team, through a head-range loop
  extracted from — not duplicated from — the f32 reference.
- **Codec decode overlaps generation.** A tee on the frame generator feeds a
  streaming codec worker on its own core, so codec time hides behind talker
  time instead of adding to it. Streamed output is bit-identical to offline
  decode; `run_complete` now reports `ttfa_ms` (time to first audio,
  ~450 ms typical).
- **Startup 2.5× faster** (PERF-003, quiet-window certified): checkpoint
  tensors widen concurrently, the codec and tokenizer load in parallel with
  the talker, the quantized artifact hydrates int8 weights natively instead
  of widening and requantizing (PERF-002, byte-identical output proof), and a
  duplicated 1.3 GB artifact hash was eliminated. Warm load is ~3.7 s where
  v0.1.3 took ~9.2 s plus a hidden ~1.1 s requantize.
- **The codec runs the reference's own BLAS form** (PERF-004): dense codec
  projections now use the bias-seeded GEMM arithmetic the oracle itself uses,
  which measured equal-or-faster than the int8 codec arm while tightening
  parity at all eight transformer seams. The int8 codec arms are demoted to
  opt-in A/B routes (`FTTS_INT8_CODEC=convnext|all`).
- Smaller levers: int8 per-depth heads with exact top-96 f32 refinement
  (byte-identical sampler inputs), a packed seq-16 microdecoder block
  schedule for draft verification, allocation-free microdecoder steps,
  prefill computes the final norm and primary head on the last row only,
  the SLEEF SnakeBeta path (129.6 dB SNR vs libm), and a persisted per-machine
  kernel plan (`~/.cache/franken_tts/kernel_plan_v0.txt`).

### Added

- `ftts enroll --denoise`: opt-in OM-LSA spectral subtraction for noisy
  reference recordings, with an offline-quantile noise floor.
- Enrollment accepts any input rate; audio is resampled to 24 kHz in-process
  with a Lanczos-3 kernel (m4a/mp3/wav/flac).
- `scripts/audio_ab_report.py`: the objective audio quality gate — SNR,
  log-spectral distance, mel-band distance, and ffmpeg spectrograms, with an
  aligned mode for same-codes comparisons and a stats mode for sampled runs.
- `FTTS_SPEC_PROBE=1` measures speculative-draft acceptance rates in place;
  its first use ruled OUT speculative sampling with the current drafter
  (NE-002: ~1% per-depth acceptance — measured dead before implementation).
- `FTTS_INT8_SCOPE=talker|micro` narrows the int8 lever for sensitivity
  attribution; `FTTS_INT8=w8a16` selects the weight-only quantization arm.

### Fixed

- The worker team survives panics: a panicking partition previously hung the
  caller forever, a panicking caller could free buffers workers still held
  pointers into, and workers materialized aliasing `&mut` views of the shared
  output buffer. All three are closed, with a panic-injection regression test.
- Enrollment denoise correctness: IMCRA speech presence matches Cohen eq. 29,
  the Nyquist bin is no longer double-scaled in the STFT mirror pass, and
  references that resample to empty PCM are refused instead of enrolled.
- Generation errors take precedence over codec-worker errors when both fail.

## [0.1.3] - 2026-08-09

### Changed

- The optimized int8 route is now the **library-wide** default
  (`ftts_kernels::route::optimized_default`), not just a CLI entry-point
  environment default, so library consumers get the same speed path as the
  binary. Conformance and oracle entry points pin the f32 reference route
  explicitly, so parity suites never measure the optimized numerics.
  `FTTS_INT8=0` remains the master switch back to the bit-exact reference
  (DISC-003, amended).
- Talker QKV and gate‖up projections fuse into single int8 dispatches.

### Fixed

- Enrollment from 44.1/48 kHz recordings (the default for phone and Mac voice
  memos): the system-decoder transcode now resamples to the speaker encoder's
  pinned 24 kHz mono instead of preserving the source rate and failing the
  rate check (`frankentts-gra`).
- The codec int8 quantization memo keys on shape as well as pointer/length, so
  an allocator-reused address can never replay a memo entry under the wrong
  matrix geometry.

## [0.1.2] - 2026-08-08

### Fixed

- `ftts convert` completes on the real pinned checkpoint. The converter
  emitted one container section per tensor (478) against the format's
  64-section cap; sections are now one per access class — the page-in policy
  unit the format was designed around — with tensors located by offset inside
  them (`frankentts-zm5`).

### Changed

- `ftts pull` now fetches the pre-quantized `.fttsq` artifact (1.3 GB) instead
  of the raw 1.7 GB main checkpoint, shrinking the download to ~2.0 GB total.
  The artifact is byte-for-byte what `ftts convert` produces from the pinned
  snapshot: talker/microdecoder hot projections stored int8
  per-output-channel, everything else verbatim.
- Enrollment hydrates the speaker encoder from the canonical `.fttsq` when the
  model directory carries one. Speaker-encoder tensors are stored verbatim, so
  artifact enrollment is bit-identical to raw-checkpoint enrollment (verified:
  identical `.spk` SHA-256 from a real recording through both paths).

## [0.1.1] - 2026-08-08

### Added

- `ftts pull`: one-command model download from the project's GitHub model
  release, SHA-256-verified against a manifest embedded in the binary, into
  `~/.cache/franken_tts/model` — no environment variable needed afterwards.
- Enrollment accepts m4a/mp3/aac/mp4/ogg/opus references, transcoded through
  the system decoder (afconvert or ffmpeg); `ftts enroll REF --default` stores
  the voice the model directory picks up automatically.
- Positional output path for `ftts say`, with m4a/flac encoding via the system
  encoder; text-proportional EOS frame backstop.
- Experimental `FTTS_INT8` W8A8 route with process-local autotune
  (`KernelPlanV0`), off by default pending quality gates.

### Fixed

- Production sampling drives the microdecoder's residual selection through the
  sampler instead of a hard-wired greedy argmax, eliminating the
  silence-attractor failure on sampled runs.

## [0.1.0] - 2026-08-08

First release: a pure-Rust, CPU-only f32 reference implementation of the pinned
Qwen3-TTS talker/codec stack, built commit-by-commit against a CPU-fp32
oracle. Everything below landed between 2026-08-05 and 2026-08-08 across a
six-crate workspace (`ftts-core`, `ftts-cli`, `ftts-kernels`, `ftts-model-qwen`,
`ftts-artifacts`, `ftts-conformance`); workstream IDs (e.g. `frankentts-p1-talker-z2w`)
refer to the checked-in beads tracker at `.beads/issues.jsonl`.

### Added

#### Engine

- f32 reference forward pass for the full pinned model: 28-layer talker with
  mRoPE position schedule, grouped-query attention, and QK-Norm.
- Sequential 15-step residual microdecoder (authoritative baseline), argmax-exact
  against oracle fixtures (60/60 code ids).
- Causal codec decoder (transformer + ConvNeXt/SnakeBeta vocoder path) with
  streaming decoder state and ring buffers; streaming and offline decode are
  bit-identical.
- Exact Qwen tokenizer (id-exact, NFC normalization modes, verbatim default)
  and token-exact ICL prompt assembly (4 variants, provable target-independent
  prefix).
- Production sampling stack matching upstream runtime defaults, plus a
  canonical greedy decoder and seeded two-level sampler contract.
- Checkpoint hydration: talker, codec, speaker, and prompt-header derivation
  from pinned weights; KV-cache replay proven bit-identical to full recompute
  across a whole utterance.
- ECAPA-TDNN speaker encoder in safe Rust with per-block L2 parity gates
  (enrollment integration still in progress).
- FrankenMTP speculative-decoding scaffolding: microdecoder verifier draft
  validation and rejection-prefix guards (`frankentts-k-verify-kernel-0i0`).
- `TtsEngine` shell with cancellation, bounded queues, health/observer events,
  resource admission for predicted peak utterance memory, and per-frame
  synthesis budgets.

#### CLI

- `ftts` / `franken_tts` binary shims with stable exit codes and stateless
  profiles.
- Subcommands: `say` (text to codes to PCM to a playable 16-bit WAV, or
  streamed PCM), `enroll`, `voice`, `convert` (pinned streaming .fttsq
  conversion), `robot`, and `doctor`.
- Robot mode: frozen-schema NDJSON contract with lifecycle events
  (`text_prepared`, `health_violation`, ...), schema validation, and export;
  guarded by a frozen-fixture contract test.
- `say --check` admission preflight backed by the real engine admission path.

#### Conformance & verification

- Truth pack: all upstream sources pinned, hashed (SHA-256), and snapshotted;
  17 open questions (license, tokenizer, mRoPE schedule, codec details,
  streaming internals, GGML quant recipe, watermarking, ...) adjudicated and
  recorded before implementation.
- Teacher-forced seam parity harnesses against CPU-fp32 oracle captures
  (`frankentts-ft7`) at every stage seam: talker layers, microdecoder layers,
  codec, prompt header, and end-to-end first frame.
- ConformanceExact ladder runner covering rungs L0-L5 with skip-honest
  receipts, oracle-tier receipt honesty enforced in CI, and a receipt auditor
  for model-gated skips.
- Standing gates: streaming==offline bit-identity, talker primary-code argmax
  EXACT through the full stack, whole-utterance code parity with exact
  decode-loop stop assertions, and left-padded==unpadded prompt bit-identity.
- Codec parity ratchet: successive bisect rounds (RoPE inv_freq form, Snake pow-first
  association, Lanes4 RMSNorm, divide-form softmax, GELU alpha, im2col GEMM,
  RVQ Conv1d projections) with measured attribution of remaining residuals to
  libm/Accelerate rounding, not wiring.
- Checked-in evidence ledgers: DISCREPANCIES, NEGATIVE_EVIDENCE, and
  PERF_LEDGER, seeded with inherited priors.
- Fuzzing workspace (cargo-fuzz) for the safetensors reader, .fttsq parser
  (structure-aware mutations), and tokenizer.
- CI: blocking local gate script, five-target build enforcement, execution
  census drift guard, and working-tree provenance on red gates.

#### Artifacts

- `.fttsq` container groundwork: canonical format with access classes,
  hardened reader, atomic writes (temp + fsync + rename), manifest-driven
  streaming conversion, and bounded Q8 matrix-row streaming quantization.
- Hand-parsed safetensors index with zero-copy mmap-backed loading (audited
  unsafe island), page-in advice, and residency observation.
- Load-time execution census: shapes, contexts, buffers, and hot-working-set
  bytes, regenerated and drift-checked in CI.

### Known limitations

- Performance: the f32 reference path runs roughly 6-7x slower than real time
  on CPU; it is a correctness baseline, not the optimized engine.
- Precision: f32 only; int8/SIMD kernels and the quantized `.fttsq` runtime
  path are groundwork, not wired into synthesis.
- Speaker enrollment is still in progress; `enroll`/`voice` surfaces exist but
  the enrollment parity gate is not yet closed.
- CUDA and long-form/chunked synthesis are out of scope for this release.

[Unreleased]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Dicklesworthstone/franken_tts/releases/tag/v0.1.0
