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
| 0.1.0 | 2026-08-08 | first release | f32 reference engine, CLI, conformance ladder, artifact groundwork |

## [Unreleased]

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

[Unreleased]: https://github.com/Dicklesworthstone/franken_tts/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Dicklesworthstone/franken_tts/releases/tag/v0.1.0
