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
| 0.1.3 | 2026-08-09 | current | optimized route becomes the library-wide default; 48 kHz enrollment fix |
| 0.1.2 | 2026-08-08 | superseded | `ftts convert` works on the real checkpoint; `ftts pull` ships the quantized `.fttsq` |
| 0.1.1 | 2026-08-08 | superseded | zero-config UX: `ftts pull`, default model cache, m4a/mp3 enrollment |
| 0.1.0 | 2026-08-08 | first release | f32 reference engine, CLI, conformance ladder, artifact groundwork |

## [Unreleased]

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

[Unreleased]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Dicklesworthstone/franken_tts/releases/tag/v0.1.0
