# COMPREHENSIVE PLAN FOR franken_tts

**Master engineering plan — v2.1 (round 1: GPT-Pro — microdecoder correction, promoted facts, re-ranked program · round 2: Grok hardness patch — traffic-model honesty, FrankenMTP exactness tiers + ragged batching, residency operationalization, AF gate-binding)**
**Status:** architecture proposal / pre-Phase −1A (two external review rounds integrated + three fresh-eyes audit passes; next gate is the §16.4 evidence gates, then beads)
**Audience:** implementing agents (CPU-kernel, model-forward, codec, CLI, conformance) and the lead architect
**Target model:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (post-cutoff)
**Challenger model:** `kyutai/pocket-tts` (100M) — bakeoff challenger and ultra-edge second model

> **The thesis, stated correctly (v2).** Qwen3-TTS's public story is "a 12.5 Hz model." Its *actual* CPU execution graph hides a **15-step autoregressive residual-code microdecoder inside every 80 ms frame**: per frame, the 28-layer talker runs once, then the 5-layer code predictor runs **fifteen sequential times** (with per-depth embeddings and per-depth 2,048-way heads), then the codec decodes. First-order Q8 weight traffic is ≈ **1.65 GB per frame** — ≈ 20.7 GB/s just to hit real time — and **the microdecoder body accounts for ~1.18 GB of it**. `franken_tts` is a model-specific native runtime that **turns this hidden microdecoder from the model's largest CPU liability into its largest optimization advantage**: cache-resident hot-packing, per-depth quantization, and speculative block drafting that the 5-layer predictor itself verifies exactly in a single causal pass.

> **How to read this document.** Claims are tagged **[VERIFIED]** (confirmed from a source we have pinned and hashed), **[SOURCE]** (confirmed from the official *live* repo/config/generation-config during the v2 review — links in §17 — accurate but pending hash-pinning in Phase −1A, where each is re-asserted and promoted to [VERIFIED]), **[REPORTED]** (from the model-selection dossier/paper; not yet line-verified), or **[OPEN]** (must be resolved from the pinned source before the dependent kernel ships). §14 indexes every [OPEN]. **Truth gating is per-component (v2 change):** a kernel is blocked only by the unresolved [OPEN]s *it* depends on — a watermark question does not block a verified talker GEMV.

---

## Table of contents

1. [Mission & non-negotiable goals](#1-mission--non-negotiable-goals)
2. [Target model dossier — Qwen3-TTS-12Hz-0.6B-Base](#2-target-model-dossier)
3. [Challenger dossier — Kyutai Pocket TTS 100M](#3-challenger-dossier--kyutai-pocket-tts-100m)
4. [Why pure-Rust + frankentorch + asupersync](#4-why-pure-rust--frankentorch--asupersync)
5. [System architecture — workspace layout & the nested-decode runtime](#5-system-architecture)
6. [Artifacts — .fttsq / .fttspack / .ftvoice / .ftvoice-cache / .fttsdraft](#6-artifacts)
7. [Model-specific CPU kernel strategy (re-ranked)](#7-model-specific-cpu-kernel-strategy)
8. [The `ftts` CLI, execution profiles & the voice compiler](#8-the-ftts-cli-execution-profiles--the-voice-compiler)
9. [Verification: two conformance contracts + reliability gates](#9-verification)
10. [Performance methodology](#10-performance-methodology)
11. [The bakeoff — three sequential gates](#11-the-bakeoff--three-sequential-gates)
12. [Phased roadmap](#12-phased-roadmap)
13. [Risks & mitigations](#13-risks--mitigations)
14. [Open research questions register](#14-open-research-questions-register)
15. [Success metrics](#15-success-metrics)
16. [Skills, methodology, companion documents & the path to beads](#16-skills-methodology-companion-documents--the-path-to-beads)
17. [Source links (promoted facts)](#17-source-links-promoted-facts)

---

## 1. Mission & non-negotiable goals

**Mission.** `franken_tts` is a pure-Rust (Rust 2024, nightly), memory-safe, CPU-hyper-optimized **library + single-binary CLI (`ftts`)** that runs the **Qwen3-TTS-12Hz-0.6B-Base** zero-shot voice-cloning TTS model **with no general ML framework**. We transform the model's bf16 weights into a canonical quantized artifact (**int8 first**; int4 only where it passes the double gate of §6.4 — and, per the v2 re-ranking, **tried on the microdecoder before the talker**) and write **model-specific kernels** whose only job is to run *this one model* as fast as possible on:

- **Apple Silicon / ARM64** — NEON, FEAT_DotProd (SDOT), FEAT_MATMUL_INT8 (SMMLA / i8mm)
- **Intel / AMD x86-64** — AVX2, AVX-VNNI, AVX-512-VNNI (and AMX tiles where present, *chosen per shape/regime, never by a fixed hierarchy*)

A **Metal feasibility spike for the microdecoder runs early** (Phase −1B; §7.12); Metal *productization* is Phase 6; CUDA sits behind that. **CPU is the priority.**

The engine is built on **frankentorch** (custom CPU tensor kernels — consumed at the *kernel* level through one facade) and **asupersync** (structured concurrency — orchestration / cancellation / streaming IO only; the hot loop runs on our own fixed `KernelTeam`, §7.10). It is **agent-ergonomic** (robot/NDJSON, stable schema, explicit exit codes) and **embeddable** (blocking sync API; streaming via bounded callbacks). **Stateless by default** — no synthesis history is persisted unless explicitly requested (§8.5).

**This is a by-the-book instantiation of `/ai-model-into-rust-mega-fused-hyper-kernel`** — fixed model revision, custom packed formats, model-specific kernels, no pretense of being a general speech framework.

### 1.1 Non-negotiable goals

| # | Goal | Operational definition | Owner |
|---|------|------------------------|-------|
| **G1** | **Audio fidelity matches the reference stack** | Two contracts (§9): **ConformanceExact** — token/activation/codec parity under canonical greedy decode, exact where the reference reproduces itself; **ProductionQuality** — the shipping sampler + quant hold WER, speaker identity, prosody, and long-form drift within budgets anchored by a powered listening protocol (§9.4). | §9 |
| **G2** | **CPU speed beats the proven CPU baseline — per stage, per profile** | With fairness controls (§10.3): warm **TTFA** and **RTF** beat the Phase −1B proven CPU baseline per stage (prefill / talker / microdecoder / codec) under the `interactive` profile; **aggregate RTF (audio-seconds/wall-second per socket), streams/socket, p95 queueing latency, and joules/generated-minute** beat it under the `throughput` profile. Beating the incumbent is necessary but does not certify the thesis — the *structural* "liability→advantage" claim is certified by **roofline efficiency**: each hot stage's measured bytes/frame and time/frame within a stated factor of its −1B floor, published per SKU. Numeric targets are set **only after the Phase −1B cost model** (§10.1); until then, all speed numbers are hypotheses. | §10 |
| **G3** | **Self-contained single binary, cross-platform** | One `ftts` executable per target (linux x86-64/arm64, darwin x86-64/arm64, windows-msvc x86-64): **no Python, no foreign ML/DSP runtime, no C++ ABI dependency, no non-system shared libraries, no network at inference time, no GPU required.** Tiny audited OS-interface islands (mmap/madvise/affinity) are permitted — that is the property users care about, and it is achievable. | §8, §12 |
| **G4** | **Memory-safe** | `unsafe_code = "forbid"` in every crate **except** `ftts-kernels`, the single audited kernel crate, where `unsafe` lives in named islands, each load carrying `// SAFETY:` and each kernel a bit-identical scalar fallback. (v2 fix: `forbid` cannot be locally `allow`ed — the workspace split in §5.1 is the compile-clean design.) | §5, §7 |
| **G5** | **Agent-ergonomic** | Versioned NDJSON robot mode, self-describing `robot schema`, stable exit codes, `--json` everywhere. | §8 |
| **G6** | **Embeddable** | `TtsEngine::synthesize(...)` / `enroll(...)` sync + blocking; streaming via bounded channels; caller-owned telemetry hooks; no global state leaks. | §4, §8 |
| **G7** | **Honest** | Every accepted divergence → `DISCREPANCIES.md` (kill-switch + measured impact + review date); every rejected lever → `NEGATIVE_EVIDENCE.md`; listening evidence over embedding metrics; claims stated at their equivalence tier; NO ADMISSIBLE RATIO over guesses. | §9, §10 |
| **G8** | **Streaming-first, packet-aware** | First PCM packet before the utterance completes; **packet size is an execution-policy parameter** (1/2/4 frames/auto — official stack uses 4-frame 320 ms packets **[SOURCE]**), never silently equated with the 80 ms model frame; streamed output equals offline decode of the same tokens in strict mode; bounded buffers; cancellation at the next frame boundary even when packets are larger. | §7.9, §8, §9 |

> **G1 over G2, always.** And the verdict's closing rule stands: real-time speech does not become more useful at 20× real time if it sounds like the wrong person.

### 1.2 Explicit non-goals (product v1) — with one relaxation

- Not a general TTS framework; not a multi-model zoo (until the gated Pocket track); not voice conversion or singing; not ASR (WER scorers are eval-side only).
- **The shipping runtime is inference-only.** *v2 relaxation:* a separate **`franken_tts_lab` model-surgery track** is explicitly permitted for the speculative-microdecoder program (§7.5): draft-model distillation, residual-group parallelization, adaptive microdecoder depth, per-depth quantization, predictor pruning. That is not scope dilution — it attacks the architecture's actual bottleneck. Lab outputs enter the shipping runtime only through the full conformance + listening gates.
- Not leaderboard SOTA vs 1.7B/hosted models (quality-ceiling references only).
- No GPU in v1 (early Metal *spike* ≠ v1 dependency).
- Not a voice-surveillance or impersonation toolkit (doctrine #10; §8.6).
- **No default persistence of synthesis history** (texts and voices are sensitive; stateless by default, §8.5).

---

## 2. Target model dossier

### 2.1 One-paragraph orientation — corrected (v2)

Qwen3-TTS-12Hz-0.6B-Base is a **hierarchical autoregressive codec-token TTS model** with zero-shot cloning. Speech is 12.5 frames/s (80 ms/frame), 16 code groups per frame. **The real per-frame execution graph [SOURCE]:**

```
for each 80 ms audio frame:
    run the 28-layer main talker once            # → semantic/primary code (1 token, 3072-way head)
    sample the primary code
    reset the tiny fixed microdecoder KV state
    for residual_depth in 0..14:                 # 15 SEQUENTIAL 5-layer forwards
        run the 5-layer residual-code microdecoder one step
        run the residual_depth-specific 2048-way head
        sample the residual code                 # conditions the NEXT depth
    # --- the talker's next input, which v2 of this plan omitted [VERIFIED @pin, C-1] ---
    talker_next_input = sum(talker_emb(code0),
                            depth_emb[j](code_{j+1}) for j in 0..14)   # all 16 codes SUMMED
                      + (trailing_text_hidden[step] if step < len(text) else tts_pad_embed)
    enqueue the 16-code frame for the codec
```

**The text stream is consumed one hidden per frame, interleaved with audio generation — not as a
prefill-only prefix.** Any forward that treats text as fully consumed at prefill desyncs after the
first frame. Per-depth embedding/head list index `j` serves code **`j+1`**; code 0 is embedded by the
*talker's* `codec_embedding`, not by any microdecoder embedding (C-2 — a silent-wrongness trap).
Citations: `docs/truth-pack/FACT_DISPOSITIONS.md` (C-1, C-2).

Serial work per second of speech: **12.5 × 28 = 350 talker-layer evaluations** plus **12.5 × 15 × 5 = 937.5 microdecoder-layer evaluations**, plus the codec. The 12.5 Hz frame rate remains highly attractive — but "only 12.5 heavyweight sequential steps per second" materially understates the work, and v1 of this plan repeated that understatement. Throughout this document the module is called the **Residual-Code Microdecoder** (never "the MTP module") precisely so its behavior cannot be mentally flattened into a single call. Residual codes are autoregressively dependent within the frame **[SOURCE]** — any "the groups are independent" optimization is invalid (v1's batch-of-frames MTP idea is deleted).

Cloning has two paths: **x-vector (speaker-embedding)** — no transcript needed, upstream documents it as potentially lower quality — and **ICL** (reference speech + transcript), the quality path **[SOURCE]**. Streaming, ten languages, ~3 s references.

### 2.2 Identity, format, size, license

| Field | Value | Tag |
|-------|-------|-----|
| HF repo | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | [SOURCE] |
| Siblings | 1.7B-Base (upper-bound reference only); CustomVoice (out of scope); **25Hz variants (see §2.8 long-form)** | [SOURCE] |
| Dtype / size | bf16; repo ≈ 2.52 GB (≈1.83 GB talker + speech tokenizer); system ≈ 0.9B params; community Q8 ≈ 993 MB talker + 291 MB codec | [REPORTED] |
| License | Apache-2.0 — **code repo** ships verbatim stock Apache-2.0 (`Copyright 2026 Alibaba Cloud`); **weights repo ships NO `LICENSE` and NO `NOTICE`**, only the model-card `license:` tag (C-4, OQ-1) | [VERIFIED code / METADATA-ONLY weights] |
| Ecosystem | official repo (cloning/streaming/eval/finetune), GGML (CPU/Metal/CUDA/Vulkan), mlx-audio; executable oracles, not performance ceilings | [REPORTED] |

### 2.3 The main talker — **[SOURCE]**

| Field | Value |
|-------|-------|
| Layers / hidden / intermediate | 28 / 1024 / 3072 |
| Heads | 16 Q / 8 KV (GQA), head_dim 128 → attention width 2048 > hidden (q_proj 1024→2048, o_proj 2048→1024) |
| **RoPE** | **theta 1,000,000; interleaved *multimodal* RoPE, sections `[24, 20, 20]`; 3-D position IDs; `position_id_per_seconds = 13`; attention calls `apply_multimodal_rotary_pos_emb`** |
| Sampling defaults | temperature 0.9, top_k 50, top_p 1.0, repetition_penalty 1.05 |
| Primary-code head | ≈ 3 MB Q8 |

The remaining talker [OPEN]s narrow to: the exact position *schedule* (`rope_deltas`, prompt→audio position transitions, the 3-D id assignment over a real prompt) — **OQ-4**; QK-Norm presence/eps and MLP gating details — **OQ-3**.

### 2.4 The text path is bigger than v1 said — **[SOURCE]**

- Text embedding: **151,936 × 2048** — ≈ **622 MB in BF16**, a major artifact component. It is **cold** (sparsely accessed rows during prefill only).
- A learned **two-layer projection `2048 → 2048 → 1024` with SiLU** sits between text embedding and talker width; it belongs in the prefill cost model.
- Consequences (§6.2, §7.13): the text embedding is a **separately mapped cold section**, never `WILLNEED`d wholesale; only referenced rows are fetched and widened; **Q8 text embedding is a legitimate footprint experiment, tested separately** from the perceptually sensitive acoustic codebooks. v1's "keep all embeddings high precision" was too coarse — text embeddings and RVQ codebooks have different risk profiles and access patterns.

### 2.5 The Residual-Code Microdecoder — **[SOURCE]**

- Five layers, 1024-wide geometry (same shape family as a talker layer).
- Invoked with `max_new_tokens = num_code_groups − 1 = 15` — **fifteen sequential 5-layer forwards per frame**.
- **Per-depth embeddings and per-depth 2,048-way output heads**; each sampled residual token conditions the next depth.
- **Ordinary RoPE** (`apply_rotary_pos_emb`, no scaling) — **a different rotary kernel than the talker's mRoPE**; the two are separate, independently conformed kernels (§5.3). **Its `rope_theta` is 1e6, the SAME as the talker's** — "plain" means *no mRoPE sectioning*, **not** a different theta; a port defaulting it to 1e4 is silently wrong (C-3). The codec decoder is the one at theta 1e4.
- **QK-Norm confirmed present** (RMSNorm over `head_dim` only, before RoPE, eps 1e-6) on both this module and the talker; `attention_bias: false` everywhere ⇒ no QKV/O biases (E-1).
- Sampling defaults: temperature 0.9, top_k 50, top_p 1.0.
- Its KV state for one frame (≤16 positions × 5 layers) is tiny — comfortably cache-resident. **Its weights, not its attention history, are the problem.**
- Training/finetuning forward computes all 15 residual positions in **one causal sequence pass** — which is exactly what makes it usable as a **block verifier** for speculative decoding (§7.5).

### 2.6 The sequential-baseline traffic model (the planning centerpiece — read the labels carefully)

Per 80 ms frame, **as the official sequential path executes** (each matrix read once per *use*, served from whatever level of the hierarchy holds it):

| Component | Q8 traffic / frame |
|---|---:|
| 28-layer talker body | ≈ 440 MB |
| Talker primary head | ≈ 3 MB |
| Microdecoder body × **15 steps** | ≈ 78.6 MB × 15 = **1.18 GB** |
| 15 per-depth heads | ≈ 31.5 MB |
| **Total before codec/ancillary** | **≈ 1.65 GB/frame** |

⇒ **sequential-baseline demand** ≈ 20.7 GB/s at 1× real time, 41.4 at 2×, 103.4 at 5× — excluding scales, activations, KV, embeddings, sampling, codec, and imperfect caching. Two labels that must never be confused:

- **This is the sequential-execution baseline, NOT a hierarchy-agnostic physics floor.** The microdecoder body's true one-read floor is ≈79 MB/frame; the all-components one-read floor is ≈ **0.55 GB/frame ⇒ ≈6.9 GB/s at 1×**. Cache residency moves the 15× reread off DRAM; FrankenMTP (§7.5) removes it altogether. **Never quote 20.7 GB/s as the bound "even after" those levers.**
- **The residency target is the hot working set, not "the body."** It is body ≈79 MB + 15 per-depth heads ≈31.5 MB + per-depth embeddings + scales + the RoPE table + KV/scratch ≈ **~110 MB-class** (exact bytes forced by the OQ-2 census — a single number, not a soft target). And the operative question is not "does ~110 MB fit in cache?" but **"does it survive the ≈440 MB talker weight stream that runs between microdecoder bursts every frame?"** — cross-stage interference is a first-class −1B experiment (OQ-18), never an isolated-microdecoder benchmark.

Everything in §7's ranking follows from this table — as re-priced per SKU and per profile by −1B.

### 2.7 The codec (speech tokenizer) — geometry now largely known **[SOURCE]**

| Field | Value |
|-------|-------|
| Sample rate | 24 kHz in/out; **1,920 samples per 12.5 Hz frame** |
| Quantizers | 16 |
| Decoder | 8 layers; hidden 512; intermediate 1024; 16 heads × head_dim 64; **sliding window 72**; upsample rates `[8, 5, 4, 3]` (product 480; ×4 base hop ⇒ 1,920 — confirm exact hop math at pin, OQ-7) |
| Encoder | separate geometry + causal-conv settings (enrollment-only build) |
| Streaming | official system groups **4 frames into a 320 ms packet** to avoid scheduling overhead — packet size is an execution-policy parameter for us (§7.9) |

Remaining [OPEN]s narrow to: exact semantic-ID mapping, implementation-level conv details/receptive fields, true causality/lookahead — **OQ-7**; watermarking — **OQ-8**.

### 2.8 Long-form is a distinct quality regime — **[SOURCE, paper]**

The paper reports the **25Hz variant outperformed 12Hz in its long-speech evaluation** (attributed partly to semantic-token stability). Not disqualifying for 12Hz long-form cloning, but a real warning. Decision (v2): **12Hz interactive mode is the primary champion; long-form document mode gets an explicit drift gate** (§9.4, §15); a same-family 25Hz engine becomes eligible only if a Base/cloning comparison confirms a substantial long-form advantage. **The trigger is predefined so the decision is evidence-driven, not re-litigated**: if the 12Hz drift gate *fails* on the long-form corpus AND a cloning-capable 25Hz Base checkpoint exists, the 25Hz document-mode engine is promoted from "eligible" to a **scheduled Phase-5-class epic** sharing the artifact/kernel machinery — a dual-rate product (interactive 12Hz + document 25Hz) inside one CLI. We do not claim one checkpoint is uniformly optimal for dialogue and 2,000-word narration.

### 2.9 The speaker encoder — **[SOURCE]**

ECAPA-TDNN-style: **128 mel bins, 24 kHz input, 1,024-d output; channels `[512, 512, 512, 512, 1536]`; kernels `[5, 3, 3, 3, 1]`; dilations `[1, 2, 3, 4, 1]`.** OQ-9 narrows to exact feature-extraction parity (mel parameters, normalization) and the downstream conditioning injection point.

### 2.10 Required neural-op set

Text tokenization + normalization modes (§8.7); cold-row text embedding + the 2048→2048→1024 SiLU projection; **two rotary kernels over three configs** (talker mRoPE sections [24,20,20] θ=1e6; microdecoder plain RoPE **θ=1e6**; codec decoder θ=1e4 windowed — C-3); RMSNorm + **QK-Norm (confirmed present, head_dim-only, pre-RoPE, eps 1e-6 — E-1)**; GQA decode attention + KV cache; SwiGLU-class MLP; the fixed 15-step microdecoder loop with per-depth embeddings/heads; per-level samplers (talker: T0.9/k50/rep1.05; residual: T0.9/k50) + canonical greedy mode; stateful causal Conv1d + windowed small attention + upsampling for the codec decoder; codec encoder + ECAPA speaker encoder (enrollment build); STFT/mel; audio decode/resample/VAD/diagnostics; WAV/PCM emission.

---

## 3. Challenger dossier — Kyutai Pocket TTS 100M

All [REPORTED] from the model-selection dossier: 6-layer ≈90M generator + 20M causal VAE, distilled from a 24-layer teacher; continuous VAE frames + one-step consistency head; 12.5 frames/s; ≈6× real time on two M4 CPU cores; ≈200 ms first chunk; reusable voice-state safetensors; EN/FR/DE/PT/IT/ES; Elo: audio quality 2016±25 (near Chatterbox Turbo 2055), **speaker similarity 1898±26 (−114 vs Chatterbox, −139 vs DSM)** — the identity deficit is why it is not primary. MIT code, CC-BY-4.0 gated weights. Upstream found no batch-one GPU advantage — CPU-only for any Pocket work.

Its disposition is now governed by the **three-gate bakeoff (§11)**, which fixes v1's circular "after equivalent optimization" kill rule. Architecturally, nothing in §5 hard-codes discrete-16-group decoding so deeply that a continuous-frame model is unrepresentable — but no `ModelArch` abstraction ships before the Qwen exemplar is certified (skill Phase 5 discipline).

---

## 4. Why pure-Rust + frankentorch + asupersync

### 4.1 The generality-tax wedge — now with the correct target

Every dimension is compile-time-known: 1024/2048/3072/128, 28+5 layers, 16 code groups, 15 residual depths, codec 512/1024/64/72/[8,5,4,3]. We specialize exact-shape kernels, pre-pack weights offline, pre-allocate everything (KV census; microdecoder KV ≤16 positions; codec ring buffers from receptive fields), skip autograd, and fuse aggressively. **The wedge's biggest payoff is the microdecoder**: 187.5 sequential microdecoder steps per second of speech (12.5 × 15), each small — per-step overhead (dispatch, cache objects, generation machinery, allocation) is a *visible* fraction of each step in a generic runtime, and the official path runs it through full generic generation machinery. A fixed-topology, zero-allocation, cache-resident `ResidualCodeDecoder` (§7.4) is the single most model-specific thing this project builds. The Luce-megakernel precedent (eliminating ~100 dispatches/token → 1.55× over llama.cpp) is the same systems principle.

### 4.2 frankentorch (kernel level, one facade)

The crown asset remains `linear_int8_dynamic_f32` + `quantize_per_output_channel_i8` + bit-exact SDOT/VNNI int8 dot with scalar fallback, plus f32 building blocks and `ft-serialize` BF16 loading. **v2 caveat: it is the incumbent, not the assumed winner** — §7.7 benchmarks W8A8 vs W8A16 vs BF16 (and later Q4A8) per operation class and ISA at real shapes; the runtime selects a `KernelPlan` by exact shape and regime, never a fixed ISA hierarchy (AMX can be excellent for big tiles and poor for batch-one GEMV). Inherited priors (kernels-below-peak not framework-overhead; unblocked-SMMLA trap; hand-SIMD-glue ~5× slower; autovec beat hand-SDOT at m=1 on M4) are imported as `inherited (pre-truth-pack)` and re-confirmed locally.

Gaps we build: audio I/O + DSP; stateful causal Conv1d + upsampling; the two rotary kernels; the `ResidualCodeDecoder`; GQA decode attention with our KV layout; RVQ machinery; samplers; the register-blocked tiled SMMLA/VNNI GEMM (prefill/verification/batched regimes); Metal (Phase 6).

### 4.3 asupersync — orchestration only; the hot loop is a `KernelTeam` (v2)

asupersync keeps: sync-shell `main`, engine-owned runtime, `spawn_blocking` + budget timeouts, cancellation checkpoints, bounded streaming channels, external I/O. **The Phase-3 hot decode loop does NOT run on rayon** (v1 had three scheduling systems; that is one too many). Division of labor:

- **asupersync** — process orchestration, cancellation, deadlines, robot event pump, file/socket I/O.
- **`KernelTeam` (§7.10)** — fixed, long-lived workers executing the hot math: static output-channel partitions, sense-reversing barrier / generation counter, per-op active-worker counts, no closures, no work-stealing, no per-layer task submission in steady state, affinity from the hardware plan, an explicit *one parallel owner at a time* invariant.
- **rayon** — retained for the Phase-1 f32 port, converters, enrollment batch work, and as the correctness/performance incumbent the `KernelTeam` must beat to land.
- **One bounded stream-queue abstraction** for PCM/events (backpressure; consumer stall parks the producer).

> Hard rules stand: never nest a runtime; never parallelism under a held lock; one live synthesis fan-out per engine; `many_utterances_without_deadlock` CI watchdog.

---

## 5. System architecture

### 5.1 Workspace layout (v2 — fixes the forbid/allow contradiction)

`#![forbid(unsafe_code)]` cannot be locally `allow`ed (a `forbid` is un-overridable by design), so v1's "single crate with unsafe islands under forbid" would not compile. **One product, tiny workspace:**

```
frankentts/                            (workspace)
├── Cargo.toml                         # workspace; release profile: LTO fat, codegen-units=1, panic=abort
├── rust-toolchain.toml                # nightly (stdarch i8mm/dotprod, portable_simd)
├── crates/
│   ├── ftts-core/                     # forbid(unsafe_code). Engine, orchestrator, KernelTeam protocol,
│   │                                  #   pipeline, streaming queues, robot events, error/exit codes,
│   │                                  #   audio front/back end (decode/resample/DSP/VAD), text front end
│   ├── ftts-model-qwen/               # forbid(unsafe_code). The model package: talker.rs, microdecoder.rs,
│   │                                  #   sampler.rs, codec.rs, codec_encode.rs, speaker.rs, decode.rs,
│   │                                  #   prompt.rs, weights facade — all math through ftts-kernels' safe API
│   ├── ftts-artifacts/                # forbid(unsafe_code). .fttsq/.fttspack/.ftvoice/.ftvoice-cache
│   │                                  #   readers/writers, hardened parsers (§6.6), streaming converter
│   ├── ftts-kernels/                  # deny-by-default; audited unsafe PERMITTED HERE ONLY, in named
│   │                                  #   feature-gated islands w/ // SAFETY: notes + bit-identical scalar
│   │                                  #   fallbacks; the frankentorch facade lives here (grep-enforceable)
│   ├── ftts-cli/                      # forbid(unsafe_code). clap; pub fn cli_main() -> ExitCode in its lib;
│   │                                  #   two thin [[bin]] shims: franken_tts (src/main.rs) + ftts
│   │                                  #   (src/bin/ftts.rs), each `fn main() { ftts_cli::cli_main() }`-shaped
│   └── ftts-conformance/              # dev-only workspace member, never shipped: integration tests
│                                      #   (conformance_exact, production_quality, robot_contract,
│                                      #   streaming_equals_batch, many_utterances watchdog, fuzz targets)
│                                      #   + criterion benches (prefill, talker_frame, microdecoder_step,
│                                      #   microdecoder_frame15, codec_packet_1/2/4, e2e, gauntlet,
│                                      #   sustained_thermal) — a virtual workspace root does NOT
│                                      #   auto-discover tests/ or benches/, so they live in a crate
├── scripts/                           # fetch_model.sh, gen_reference_fixtures.py, cost_model.py, check.sh
└── docs/                              # DISCREPANCIES.md, NEGATIVE_EVIDENCE.md, PERF_LEDGER.md, truth-pack/,
                                       #   QWEN3_TTS_EXECUTABLE_SPEC.md, PERFORMANCE_ARCHITECTURE.md,
                                       #   CONFORMANCE_AND_LISTENING.md, VOICE_COMPILER_DESIGN.md (§16.2)
```

Alternative considered: put all new unsafe kernels in frankentorch and keep this repo 100% safe — rejected for now (model-specific kernels don't belong in the general substrate), revisit if kernels prove reusable. Two binaries stay (owner convention; thin byte-equivalent shims cost nothing).

### 5.2 The nested-decode runtime (four stages, one corrected inner loop)

```
[A] VOICE COMPILATION (offline — `ftts enroll`, §8.6)
[B] TEXT PREPARATION & PREFILL: normalize (per §8.7 mode) → tokenize → prompt/ICL assembly
    → cold-row text embedding + 2048→2048→1024 SiLU projection → talker KV prefill
[C] PER-FRAME NESTED DECODE (the §2.1 graph):
    talker step → primary code → RESET microdecoder KV →
    15 × (microdecoder step → depth head → sample) → 16-code frame
[D] STREAMING CODEC DECODE: packets of 1/2/4 frames per the execution profile;
    ring-buffer state advances; PCM emitted through the bounded queue
```

C/D overlap and the codec's thread share are **measured decisions** from the −1B cost model (codec overlap may merely contend with the talker/microdecoder for bandwidth — §10.1 Q5). Cancellation checkpoints at every talker frame regardless of packet size.

### 5.3 Op → kernel map (self-contained)

| Op | frankentorch status | Plan |
|---|---|---|
| int8 dynamic-quant linear (bit-exact SDOT/VNNI dot + scalar fallback) | **EXISTS** | reuse as the incumbent; §7.7 decides W8A8 vs W8A16 per op |
| f32 linear / SDPA(+GQA) / RMSNorm / SiLU / softmax / argmax | **EXISTS** | reuse for the Phase-1 f32 forward + prefill basis |
| safetensors BF16 load | **EXISTS** | reuse inside the streaming converter (§6.5) |
| register-blocked tiled SMMLA/VNNI GEMM (prefill / seq-16 verification / batched) | named, unbuilt | **BUILD** — the wedge |
| talker **mRoPE** (theta 1e6, sections [24,20,20], 3-D ids) | — | **BUILD** (rotary kernel #1) |
| microdecoder **plain RoPE** (16-position precomputed table) | — | **BUILD** (rotary kernel #2 — never one generic "RoPE") |
| GQA decode attention + KV cache (16Q/8KV, head_dim 128) | — | **BUILD** |
| QK-Norm per-head variant (if OQ-3 confirms) | — | **BUILD** thin |
| **`ResidualCodeDecoder`** (fixed 15-step engine, per-depth embeddings/heads) | — | **BUILD** — the flagship (§7.4) |
| verification-mode seq-16 causal pass | — | **BUILD** alongside the step kernel (§7.5) |
| stateful causal Conv1d + transposed-conv upsampling (streaming) | — | **BUILD** (codec, §7.9) |
| codec small attention (hidden 512, 16 heads × 64, window 72) | — | **BUILD** |
| RVQ codebook lookups/sums (high-precision codebooks) | — | **BUILD** (small) |
| two-level sampler (T0.9/k50, +rep 1.05 talker) + canonical greedy + seeded RNG | — | **BUILD** (our RNG — a documented DISC vs torch's stream) |
| ECAPA-TDNN speaker encoder (dilated Conv1d + SE blocks + attentive stat pooling) | — | **BUILD** (enrollment build; geometry known §2.9) |
| codec encoder | — | **BUILD** (enrollment build) |
| STFT/mel (fixed sizes); audio decode/resample/VAD/diagnostics; WAV/PCM emission | — | **BUILD** (pure Rust) |
| BPE tokenizer (id-exact) + §8.7 normalization modes | — | **BUILD** |
| text projection MLP 2048→2048→1024 (SiLU) | EXISTS (linears) | reuse; part of prefill |

---

## 6. Artifacts

### 6.1 Two-layer split: canonical model vs machine-specific packing (v2)

v1 conflated logical content with ISA layout ("one canonical artifact, many packings" while emitting per-arch conversions). v2 separates:

**`.fttsq` — immutable canonical model artifact (portable).** Logical Q8/BF16(/Q4) tensors in a canonical layout; quantization recipe; tensor manifest; source revision + per-section SHA-256; model config; license/NOTICE. **No machine-specific tiling.** One download serves every machine; kernel layouts can evolve without republishing GB-scale model files.

**`.fttspack` — regenerable local execution cache.** Keyed by `{model_content_hash, kernel_abi_version, cpu vendor/model, ISA feature set, op-shape plan, packing version, quant execution mode, autotune-result hash}`. Contains SDOT/i8mm/AVX-VNNI/AVX-512/AMX layouts, the **microdecoder hot-pack** (§7.4), and the selected `KernelPlan`. Safe to delete and rebuild; enables install-time autotuning; makes rollback clean.

**`.fttsdraft` — versioned lab-output artifact (Phase 5)**: draft/surgery models (distilled drafters, parallel heads, adaptive-depth variants) ship beside `.fttsq`, format-versioned, compatibility-keyed to model + engine ABI, kill-switched — the lab→runtime ABI that lets structural wins land as product (§12 Phase 5). Parsed by the same hardened `ftts-artifacts` machinery (§6.6).

### 6.2 Access-class sections (physical layout of `.fttsq`)

```
HOT_RECURRENT_MICRODECODER   # ~79 MB Q8 body + 15 heads — the cache-residency target
HOT_RECURRENT_TALKER         # ~440 MB Q8
HOT_CODEC_DECODER
COLD_TEXT_EMBEDDING          # ~622 MB BF16 (or Q8 after §6.3 experiment) — lazy, row-granular, never WILLNEED'd wholesale
ENROLLMENT_SPEAKER_ENCODER   # optional section (--with-enrollment)
ENROLLMENT_CODEC_ENCODER     # optional section
METADATA
```

Page-in policy per class: hot recurrent sections get `WILLNEED`/huge pages; cold text embedding stays sparse and lazy.

### 6.3 Quantization policy — INT8 first; embeddings differentiated (v2)

Unchanged spine: per-output-channel INT8 weights on talker + microdecoder GEMMs; INT32 accumulation; fused requant; norms/sampling-logits/speaker-conditioning/RVQ codebooks high-precision; codec Q8 with high-precision codebooks (validate the GGML kept-high set, OQ-13). **v2 refinements:**

- **W8A8 vs W8A16 is an open benchmark per op class** (§7.7), not a settled default.
- **Text embedding (622 MB) is a separately gated Q8 experiment** — cold, row-wise access, different risk profile than acoustic codebooks. "All embeddings high-precision" was too coarse.
- **Per-depth quantization is a first-class axis**: the 15 microdecoder depths may tolerate different precision (later acoustic codebooks plausibly less sensitive than the first semantic-rich code) — AF-1 allocates per (tensor × residual-depth), not per tensor alone.

### 6.4 int4 — MTP-first, double-gated (v2 re-ranking)

The double gate stands (faster end-to-end per actual ISA including unpack; blind-listening clean per §9.4). **The placement order is inverted from v1:** once a genuinely native packed Q4 kernel exists, test (1) later microdecoder residual depths, (2) the whole microdecoder body, (3) microdecoder heads, (4) talker MLPs, (5) talker attention projections. Why: the body is reread **15× per frame** — Q4 shrinks ≈79 MB toward ≈40 MB, making cache residency possible on materially more machines, and an MTP-only rollback is clean. Inherited prior (int4 unpack cost 5.8× slower in a sibling) must be re-proven at these shapes.

### 6.5 Streaming, bounded-memory conversion (v2)

The converter must NOT load the whole BF16 model into a `BTreeMap<String, DenseTensor>` widened to f32. Instead: mmap/stream one safetensors tensor at a time → validate vs manifest → quantize in bounded tiles → hash + write its canonical section → release → atomically finalize the index. Runtime quantization still ships first and the converter reuses the identical quantizer function (byte-identical by construction, plus the asserting test).

### 6.6 Hardened artifact parsing (concrete acceptance criteria)

Maximum tensor count/rank/dims; checked arithmetic on every offset/length; rejection of overlapping ranges; per-section digest verification; atomic temp-write + rename; truncated-file and bit-flip tests; parser fuzzing (`.fttsq`/`.fttspack`/`.ftvoice`/`.ftvoice-cache`); header size caps; no decompression without bomb limits. These are CI gates on `ftts-artifacts`, not general security ceremony.

### 6.7 `.ftvoice` split: portable voice recipe vs derived caches (v2)

**Portable `.ftvoice`** (stable identity data): reference hashes, provenance + consent attestation, transcript + alignment, selected speech regions, speaker embedding, reference codec tokens, diagnostics + quality scores, language metadata, the reproducible preprocessing recipe, optionally embedded lossless reference audio. Profiles: `--voice-pack portable` (embeds normalized lossless reference) / `private` (derived features only) / `minimal` (embedding only) — the privacy default is a deliberate product choice, surfaced in docs.

**Derived `.ftvoice-cache`** (recomputable only): static prompt embeddings, verified static-prefix KV, backend layouts, quantized cached states, packet-mode-specific codec context. Keyed by `{voice_recipe_hash, model_hash, prompt_builder_version, streaming_mode, quant_recipe, math_mode, engine_abi}`.

**Prompt-KV caching is NOT assumed straightforward (v2):** official ICL prompt construction interleaves reference text, target text, reference codec tokens, and mode-dependent alignment; streaming and non-streaming prompts differ structurally **[SOURCE]**. Cache only the **maximal prefix proven independent of the target text** (OQ-10 resolves what that is); keep all source primitives so caches regenerate after any bump.

### 6.8 Determinism & proof obligations (restored — over-compressed in the v2 rewrite)

- **Bit-exact round-trip**: BF16/F32 high-precision tensors stored verbatim; a `convert→load→re-serialize` test asserts byte identity.
- **Deterministic quantization**: pure function of the pinned source bytes — same source → same `.fttsq` (content hash asserted; no RNG, no calibration data in v1). `.fttspack` regeneration is deterministic given its full key tuple; CI verifies every packing dequantizes to identical logical weights.
- **Artifact == runtime quantization byte-identical by construction** (§6.5) plus the test that asserts it.
- **i32-overflow is a proof obligation at THIS model's real worst-case K, per arch, forever**: talker `down_proj` **K = 3072** (U8S8 ≤ ~99.5M — ≥21× i32 headroom), `o_proj` **K = 2048**; the microdecoder shares the same K bounds; the **seq-16 verification GEMM** (§7.5) uses the same weights at larger m — same per-element K bound, re-proven anyway; **the codec convs' reduction length (C_in·kernel_width — the K of the conv-as-dot-product, whether or not an im2col buffer is ever materialized) comes from the −1A census (OQ-7) and must be recomputed then**. A unit test multiplies worst-case saturated operands at each real worst-case K on every kernel tier and asserts i32 == i64 reference; it lives in the **shipped selftest**, not just CI. The AVX2 `vpmaddubsw` path carries its own saturation proof (split-accumulate) or a ledgered DISC. **This is permanent law for every kernel class that ever lands** — the Q4 unpack path, the codec convs, the seq-16 verify pass, batched variants: each new class ships its own worst-case-K selftest row before it can become a default.

---

## 7. Model-specific CPU kernel strategy

### 7.1 The hot ops — corrected ranking (hypothesis → replaced by the −1B cost model)

| Op | Stage | Traffic/frame (Q8, first-order) | Priority |
|----|-------|---:|----------|
| **Microdecoder: 15 × (5-layer step + depth head)** | C | **≈ 1.21 GB** | **#1 — the serial monster** |
| Talker frame (28 layers + head) | C | ≈ 443 MB | #2 |
| Codec packet decode | D | census (OQ-7) | #3 — measure before ranking below talker |
| Prefill (prompt + ICL tokens + text projection) | B | per-utterance | #4 (TTFA) |
| Sampling (two levels, top-k) | C | small | #5 |
| Enrollment encoders | A | offline | #6 |
| Norm/rotary/activation glue | all | small | autovectorize |

The ranking is **per profile, not global**: the TTFA-critical path is prefill + ICL assembly + cold text-embedding rows + first packet; steady-state RTF is microdecoder/talker/codec; long-form adds growing talker-attention cost and drift management. A short "hey" is prefill-dominated while its RTF is microdecoder-dominated. **−1B re-ranks this entire table from measurements, per profile** — including the live possibility that codec or prefill tops the interactive ranking once the residual work lands.

### 7.2 The inherited optimization constraints (unchanged, binding)

No hand-wide-SIMD over glue (≈5× slower, measured); native int8 matmul intrinsics + full-core parallelism are the levers; register/cache blocking mandatory for SMMLA; re-prove hand-SIMD vs autovec per toolchain at real shapes; m=1 GEMV is bandwidth-bound — structural levers only near the roofline. Two standing riders: **(a)** the one measured hand-SIMD *exception* is **vectorized polynomial transcendentals** (`exp`/`sigmoid`/`tanh` — LLVM emits scalar libm calls; a range-reduced minimax polynomial is a known multi-× win for softmax/SiLU and any hot codec activation) — numerics-affecting, so it ships behind a `FTTS_*` switch, defaults ON only after a measured A/B + passing parity gate, and is ledgered; **(b)** **every fusion is a byte-identity-preserving refactor** (same arithmetic, fewer materializations) gated by the parity proof, and parallelism runs over *independence* dimensions only (heads, output channels, disjoint chunks, streams) — never over reductions.

### 7.3 Per-arch int8 GEMM/GEMV plan

U8S8 (+128 fold) for VNNI, S8S8 for SDOT/SMMLA — converter and kernel agree per arch; i32 accumulation with scales applied post-accumulation and fused bias/activation; the AVX2 `vpmaddubsw` path carries its own saturation proof or a ledgered DISC; the scalar fallback is bit-identical and cross-compiles to every target. **Tier selection is per (shape, regime) via the autotuned `KernelPlan` in `.fttspack`** — no fixed `AMX > AVX-512 > …` hierarchy.

### 7.4 Lever #1 — the `ResidualCodeDecoder` (cache residency + inner-loop specialization)

A dedicated microdecoder engine, not a generic mini-transformer:

- fixed 15-step loop, fixed 5-layer topology, compile-time shapes;
- **tiny fixed-size KV arrays reset once per frame** (≤16 positions; lives in private cache);
- **precomputed plain-RoPE values for positions 0–15** (a 16-row table — no trig in the loop);
- direct per-depth embedding + head selection (indexed arrays, no dispatch);
- no dynamic cache object, no generation machinery, no allocation, no full probability vectors outside the top-k/sampler path;
- **the MTP hot pack**: the full **~110 MB-class hot working set** (§2.6: body + per-depth heads + per-depth embeddings + scales + RoPE table + KV/scratch; exact bytes from OQ-2) physically separated in `.fttspack`, layout tuned so it stays resident across the 15 steps. Be precise about what is and isn't available here: in exact sequential mode there is **no cross-step scheduling freedom** — step *t+1*'s input depends on step *t*'s sampled token, so the win is pure cache residency + layout, not weight-stationary scheduling. True weight-stationary execution (each matrix read once per frame) exists only in the FrankenMTP verification pass (§7.5) and in cross-stream batching (§7.11).
- **Residency is a measured quantity, not a vibe**: operationally defined as **measured DRAM bytes/frame for the microdecoder sections (PMCs / Instruments) approaching the one-read floor**, with body-tile hit rates as the diagnostic — and measured **under the real talker→microdecoder→codec schedule** (§2.6 interference), never in isolation. OQ-18 fails *cleanly* when DRAM bytes/frame stay near the sequential baseline despite packing.
- **The failure-mode playbook is engineered, not hoped**: if residency fails on a SKU — Q4 depth waterfall (§6.4), head/embedding packing, software-prefetch injection at layer boundaries, huge pages, CCD/SLC pinning, and ultimately FrankenMTP as the traffic lever (§7.5). The best measured combination ships per SKU via `.fttspack`; *benchmarked against* the talker's generic layout either way.

### 7.5 Lever #2 (flagship radical) — `FrankenMTP`: speculative block verification

The microdecoder's training-mode forward evaluates all 15 residual positions in **one causal sequence pass** **[SOURCE]** — the basis for using the full 5-layer predictor as a **block verifier**. Precision about what this kernel actually is:

- **The frame sequence is 1 + 15 = 16 positions**: one conditioning/primary position (from the talker, **frozen** during verification) plus 15 residual positions; the verifier scores exactly the 15 residual positions. Masking, position ids, and the depth-index → sequence-index → head-index mapping must be **bit-equivalent to the training-mode forward** (OQ-5) — an off-by-one here invalidates the entire epic.
- **This is NOT a stock seq-16 LM pass.** Each residual position applies its **own depth-specific embedding** to the drafted id and its **own 2,048-way head**. "One body read" is real, but the verify kernel is a custom **multi-head-per-position engine** (shared 5-layer body over the sequence + per-position embed/head application), and the per-depth head + embedding traffic (~31.5 MB + embeds) is part of its cost model — not "a healthier GEMM and we're done."

The loop:

1. a cheap drafter proposes all 15 residual codes;
2. the full microdecoder verifies the block in one causal forward (one body read instead of 15; healthier matrix utilization and output-channel parallelism);
3. accept the valid prefix under the named speculative rule (see exactness tiers below; OQ-19);
4. at the first rejection, sample the corrected token from the **verifier's own adjusted distribution** at that position; because the suffix's context has now changed, the remaining depths are **regenerated** — sequentially, or by re-draft + re-verify (a measured policy choice), never by "verifying" the stale draft;
5. fall back to exact sequential decode when acceptance is poor (the sequential path is always authoritative).

**Exactness is a two-tier claim — and AF-3 is neither tier:**

- **Strict/greedy mode**: acceptance = "draft id equals the verifier's argmax at that position" — token-identical to sequential greedy *by construction*; the §9.5 invariant `greedy FrankenMTP ≡ greedy sequential` enforces it.
- **Sampled mode** (T0.9 / top-k 50 at both levels, + talker repetition penalty): requires a **named speculative-sampling algorithm with a distributional-correctness proof** — accepted prefixes distributed exactly as the sequential sampler's output; rejections resampled from the verifier's *adjusted* conditional (never the draft's); the top-k-truncation and repetition-penalty interactions resolved explicitly. **This is OQ-19 — a research deliverable, not a checklist row.**
- **AF-3's e-value bound is a runtime reliability monitor / kill-switch** (it bounds how often the speculative path misbehaves in production). It is **not** a substitute for distributional exactness — "disagreement is rare" and "samples are legal" are different claims, and conflating them is a category error this plan explicitly forbids.

**The expected-case surface, not the asymptote, is what gets ranked.** Per frame:
`T(α) = T_draft + T_verify + (1 − α_full) · T_repair` — where repair is sequential-suffix, re-draft, or hybrid, and a partial accept at depth *k* of 15 can be **worse than pure sequential** (draft + full verify + residual sequential all paid). **−1B measures the primitives (T_draft candidates, T_verify, T_repair per SKU) and publishes the break-even acceptance curve α\*(SKU); Phase 3A measures real α per drafter.** The high-acceptance asymptote is the ceiling, not the plan.

Drafters, in increasing ambition: previous-frame residuals + a small transition model; a 1–2-layer distilled microdecoder; 15 parallel residual heads off the talker hidden state; a quantized/layer-skipped microdecoder copy; a learned block predictor (groups of 3–5). At high acceptance, recurrent pre-codec traffic falls from ≈1.65 GB/frame toward ≈**0.55 GB/frame + draft cost** — the 5×-real-time first-order bound drops from ≈103 GB/s toward the mid-30s.

**How the two flagship levers compose (per SKU, not additively):** the traffic model counts bytes touched at *whatever* level serves them. On SKUs where the §7.4 hot pack achieves cache residency, DRAM already sees the body only ~once per frame and FrankenMTP's win is **serial latency + compute shape** (15 dependent GEMVs collapse into one block pass + a cheap draft); on SKUs where residency fails, FrankenMTP is the **traffic** lever that makes the roofline reachable at all. They are complements, partially substitutes — the −1B cost model prices both per SKU rather than assuming their savings add.

**Dual claim tiers, pre-written (so a sour OQ-5 cannot collapse the epic into silence):** **Tier 1** — OQ-5 resolves optimistically (the training-mode forward is mask- and distribution-equivalent to the 15-step inference loop): FrankenMTP is an **exact** verifier; strict-mode token identity and sampled-mode distributional exactness are both claimable. **Tier 2** — OQ-5 resolves sour (not equivalent): FrankenMTP downgrades to **approximate draft assistance** — Contract-B-only claims, the AF-3 monitor always on, the sequential path remaining the only exact mode. Still valuable; a different epic with different claims, chosen by evidence rather than discovered by surprise. The fixed 15-length makes acceptance machinery far simpler than open-ended LLM speculation. Drafter training lives in **`franken_tts_lab`** (§1.2); the runtime integration ships behind the AF-3 reliability monitor (§10.7) with α→0 disabling it. **FrankenMTP owns the flagship slot; talker-level speculation is re-sequenced to Phase 5, not deleted (§10.5).**

### 7.6 The talker persistent loop

On the `KernelTeam`: per-layer fused pass 1 (norm → QKV → QK-Norm? → **mRoPE** → GQA attention → O → residual) and pass 2 (norm → gate‖up → SiLU⊙ → down → residual, with the §7.8 quant fusion); workers persist across all 28 layers, the 15 microdecoder steps, and successive frames; zero steady-state allocation/dispatch; KV preallocated from the census.

### 7.7 `KernelPlan`: W8A8 vs W8A16 vs BF16 per op class (v2)

For each op class × ISA, benchmark at the real shapes: W8A8 (dynamic per-row activation quant), W8A16/BF16 weight-only, BF16/F16, and later Q4A8. The winner may differ among: single-row talker GEMV; 15-step microdecoder GEMV; **seq-16 microdecoder verification GEMM**; text-prefill GEMM; codec conv; multi-stream batched GEMM. The autotuned `KernelPlan` (persisted in `.fttspack`) binds each op to its winner; `robot backends` reports the plan.

### 7.8 Correct RMSNorm → activation-quant fusion (v2 fix)

RMSNorm needs a global reduction before any normalized value exists — v1's "RMSNorm→QKV one pass" overstated. The exact two-pass form:

1. pass 1: sum of squares **and** `max|x_i·γ_i|` in one sweep;
2. derive the RMS factor and the dynamic quant scale;
3. pass 2: emit **quantized** normalized activations directly into aligned scratch — no materialized f32 normalized vector;
4. Q/K/V reuse that one quantized input; gate/up reuse the post-attention norm's.

Also fuse residual-add with collection of the *next* norm's statistics where the arithmetic-order contract permits.

### 7.9 The codec: co-equal, packet-adaptive

Stateful causal-conv ring buffers; fused conv+bias+norm+act; specialized small attention at the codec's exact shape (hidden 512, **16 heads × head_dim 64**, sliding window 72 — note attention width 1024 = 2× hidden, the same wider-than-hidden signature as the talker); no im2col materialization; upsample `[8,5,4,3]` streaming straight into PCM; distinct encode/decode builds; streaming==batch standing gate — **plus packet adaptivity**: `--packet-frames {1,2,4,auto}`; profiles map to packets (`interactive`→1–2, `balanced`→4, `throughput`→larger batches, `strict`→conformance packetization). Cancellation still checks every 80 ms talker frame.

### 7.10 `KernelTeam` (the hot-loop execution architecture — v2)

Fixed long-lived workers; static output-channel partitions per op; sense-reversing barrier; per-op active-worker counts (a 1024-row GEMV may want 4 workers while a seq-16 block pass wants 12 — from the USL sweep); no closures/work-stealing/task submission in steady state; affinity from the hardware plan (P-cores on Apple; CCD-aware on AMD); *one parallel owner at a time* invariant enforced by construction. **Variable batch size is a designed case, not an afterthought**: under continuous batching the active stream count N changes every scheduling quantum, so static partitions must **rebind, not degrade** — precomputed partition tables per N-bucket, rebinding only at quantum boundaries, hysteresis so churn doesn't thrash the plan. Rayon remains for f32 port/converter/enrollment and as the incumbent to beat.

### 7.11 Continuous batching — the AMD throughput architecture (v2)

Isolated per-stream engines reread the same weights N times. For high-core AMD, a central frame scheduler:

```
collect streams ready for their next frame
batch the talker step across them          (GEMV → GEMM; weights read once)
for residual_depth in 0..15:
    batch that microdecoder depth across all active streams
batch/pipeline codec packets
return each stream to its own sequential state
```

Preserves each utterance's autoregressive dependency while amortizing every matrix read across streams — the primary throughput axis on Threadripper/EPYC. Two explicit policies: **`Latency`** (one stream, small static team, minimal queueing) and **`Throughput`** (continuous batching, configurable queue delay, weight-stationary scheduling, NUMA-local packs, admission control).

**Ragged FrankenMTP × batching is a designed artifact, not emergent behavior.** Under speculation, streams reject at different depths, so depth-synchronized batching goes ragged. Candidate policies — designed and measured jointly by 3A + 3D, one scheduler for both flagships: pad-and-mask cohorts; cohort splitting by accept depth; a **dual-lane scheduler** (block-verify lane + sequential-repair lane, streams migrating between them per frame); per-cohort re-drafting. The two flagship systems must share one scheduler design — they are not two independent diagrams that meet in production.

**Ownership note**: the continuous-batching scheduler **is** the engine's single parallel owner (doctrine #5) — batching is multi-stream *inside* one fan-out, never N concurrent fan-outs; `fttsd` admission rides the same capacity model and certificate.

### 7.12 Early Metal microdecoder spike (v2 — moved from Phase 6 to −1B)

One narrow question, answered cheaply right after the execution graph exists: **can a persistent Metal kernel/command-graph run the 15 residual steps — including sampling and KV updates — without 15 host round trips?** Positive ⇒ the microdecoder becomes vastly more attractive on M5 and artifact/quant choices stay Metal-compatible; negative ⇒ path closed cheaply, documented. Metal *productization* remains Phase 6 (integer cross-backend contract; CPU stays default; negative go/no-go is a valid outcome).

### 7.13 Memory, allocator, layout

System allocator by default (mimalloc is an opt-in perf feature; any head-to-head claim uses the same allocator on both sides); zero steady-state allocation; 64-byte-aligned activation and packed-weight buffers; scales in struct-of-arrays beside their tiles. Plus: **page-in policy per access class** (§6.2) — hot recurrent sections `WILLNEED`/huge-paged, **cold text embedding lazy and row-granular, never wholesale**; the microdecoder hot pack placed for maximal L2/L3/SLC residency.

### 7.14 Build-time optimization

Unchanged: LTO fat + codegen-units=1 + panic=abort shipping profile; separate release-perf profile; PGO on the golden corpus (the nested frame loop is what PGO straightens); BOLT; `target-feature` + runtime dispatch, never `target-cpu=native`.

---

## 8. The `ftts` CLI, execution profiles & the voice compiler

### 8.1 Binaries & entrypoint

Two `[[bin]]` shims (`ftts` + `franken_tts`) over `ftts-cli`'s `cli_main()`; sync `main`; engine-owned runtime.

### 8.2 Execution profiles (v2 — two products, one runtime)

```
ftts say --profile interactive   # p50/p99 TTFA, single-stream RTF, small packets, low memory, thermals, cancel latency
ftts say --profile balanced      # default; 4-frame packets
ftts say --profile throughput    # continuous batching, larger codec batches, streams/socket, joules/minute
ftts say --profile strict        # canonical math mode + conformance packetization (the reproducibility contract)
```

Interactive edge (M4/M5 laptops) and throughput host (many-core AMD servers) are different products with different honest optimization targets; profiles make each claim precise. A thin optional **`fttsd`** adapter (OpenAI-compatible speech endpoint) is a post-v1 companion binary — HTTP concerns never enter the core crates.

The library backs all of this with a **caller-owned observer contract** (`SynthesisObserver`: stage timings, frame progress, admission decisions, packet emissions, health events) — `--trace`, robot NDJSON, and the bench harness are all thin consumers of the same hook; no global state, no default persistence (G6 made concrete).

### 8.3 Subcommands

`say` (WAV out or `--stream raw`), `enroll`, `voice inspect`, `convert`, `robot schema/health/backends`, `doctor` — with flags: `--profile`, `--packet-frames`, `--math-mode {strict,fast}`, `--voice-pack {portable,private,minimal}`, `--normalize {verbatim,conservative,locale-aware}`; **removed (v2): `ftts runs`, `sync export/import`** — stateless by default; `--trace <dir>` opts into structured NDJSON traces; benchmark evidence is written by the bench harness, not a database. (If durable local state is ever reintroduced it uses fsqlite per owner convention — never rusqlite.)

### 8.4 Robot / NDJSON contract

Versioned NDJSON events, one JSON object per line, every line carrying `schema_version`: `run_start`, `stage` (name, seq, elapsed, budget), `frame` (coarse progress, throttled), `audio_chunk` (byte offset, duration, packet size, sink), `health`, `run_complete`, `run_error` (carries the exit code). **PCM bytes are never interleaved with NDJSON on the same stream** — events on stdout with audio to `-o`/fd, or `--stream raw` PCM on stdout with events on stderr; one documented contract. `robot schema` self-describes all event types; a frozen-JSON-schema contract test validates emitted events. Deterministic under `strict` math mode + fixed seed, with the scope stated in §9.3.

### 8.5 Exit codes, model resolution & env

**Exit codes (stable, documented in `error.rs`, mirrored in `run_error`):** `0` success · `1` generic error · `2` usage/CLI error · `3` model not found/resolvable · `4` input error (bad text encoding / unreadable reference audio) · `5` budget/timeout exceeded · `6` cancelled (Ctrl+C) · `7` format/version mismatch (`.fttsq`/`.fttspack`/`.ftvoice`/`.ftvoice-cache`) · `8` enrollment-quality refusal (reference unusable; details in payload; `--force` for consenting adults).

**Model resolution (no network at runtime):** `resolve_model(spec)` accepts an explicit path or a short name searched over an ordered, env-driven list (`$FTTS_MODEL_DIR`, `~/.cache/franken_tts/models`, …), failing with an actionable error that lists every searched directory; availability checks use a cheap header sniff (magic bytes), never a tensor load.

**Env:** `FTTS_MODEL_DIR`, `FTTS_THREADS`, `FTTS_PROFILE`, `FTTS_PACKET_FRAMES`, `FTTS_MATH_MODE`, `FTTS_QUANT`, `FTTS_FORCE_ARCH`, `FTTS_NUMA`, `FTTS_STAGE_BUDGET_*_MS`, plus per-lever kill-switches — each read once via `OnceLock`, documented, defaulted by a measured A/B, every lever's unarmed path structurally unchanged (the kill-switch convention).

### 8.6 The voice compiler (enrollment as the second central thesis — v2 expansion)

> `ftts enroll` compiles messy real-world recordings into a reproducible, scored, portable voice asset.

- **Automatic reference-segment discovery**: detect speech regions → reject overlap/multi-speaker → score clipping/SNR/reverb/music/stationarity → score phonetic coverage from the transcript when present → penalize extreme emotion/whisper/atypical register unless requested → select several candidate 3–12 s segments → synthesize a fixed audition set from each → rank by objective metrics (+ optional listener preference) → preserve the chosen segment and all alternatives in the enrollment report. This will often buy more perceived quality than another kernel lever.
- **ICL is the quality default**: modes are `quality` (transcript-backed ICL), `quick` (x-vector only — upstream documents possible quality reduction), `auto` (ICL when the transcript verifies, else x-vector **with a warning**). Never presented as interchangeable equals.
- **Transcript verification**: duration consistency, alignment confidence, omission/repetition detection, optional external ASR verifier hook (a clean plugin contract — the shipping binary contains no ASR), and a forced choice between corrected-ICL and x-vector fallback on poor confidence.
- **Multi-reference: selection before averaging** — rank and pick high-quality homogeneous references, evaluate individually, only then test embedding aggregation; preserve per-reference embeddings + provenance; never irreversibly average away style variation.
- **Consent & provenance** (doctrine #10): attestation recorded; no acquisition features; watermark preserved (OQ-8).

### 8.7 Explicit text-transformation modes (v2)

Qwen advertises robustness to noisy text — aggressive silent rewriting is wrong. Modes: **`verbatim`** (exact upstream behavior; the conformance path and default), **`conservative`** (unambiguous normalization only), **`locale-aware`** (explicit locale-dependent expansion), plus a pronunciation lexicon, per-span language overrides, and an **emitted normalization trace** (what changed, why). This makes names, technical prose, math, code, currencies, and multilingual text tractable.

---

## 9. Verification

Governed by `/ai-model-into-rust-mega-fused-hyper-kernel` (oracle before engine; seam-ordered proofs; measured tolerances; skip-honest receipts) with the v2 correction that **one ladder cannot serve two masters**: the official model *samples at both autoregressive levels* (T=0.9/k=50 defaults **[SOURCE]**), so a tiny legitimate logit perturbation can flip a token and diverge the whole sequence without any perceptual harm. Two contracts:

### 9.1 Contract A — `ConformanceExact` (implementation correctness)

Exact text token ids; exact prompt assembly (both modes, streaming + non-streaming); exact speaker-encoder preprocessing; **teacher-forced** per-layer activations and logits (talker, all 15 microdecoder depths, codec blocks, ECAPA); **canonical greedy generation** (argmax both levels) with exact code ids where the reference reproduces them; codec decode of *fixed* tokens; scalar==SIMD; cached==uncached; streaming==offline for fixed tokens; `.ftvoice`-idempotence. May use slower canonical arithmetic and fixed/no RNG. This is the kernel-development ladder, instantiated for TTS as: **L0** text normalization + tokenizer + prompt assembly + audio preprocessing EXACT → **L1** per-op activations → **L2** per-layer/stage (talker layers, each microdecoder depth, codec blocks, ECAPA) → **L3** per-frame per-level logits (argmax exact where the oracle reproduces itself; else within *measured* tolerance derived from the oracle's own nondeterminism floor) → **L4** full codec-token stream exact under canonical greedy → **L5** end-to-end audio + metrics within documented budget. Teacher-forced wherever sampling would confound. **The OQ-12 fork is pre-committed both ways**: if canonical greedy audio proves quality-viable, L5 means perceptual goldens + metrics over greedy audio; if it does not, L5 means exactness over codec-fixed audio (decode of frozen token streams) and the perceptual duty transfers wholly to Contract B — agents never invent tolerances to bridge that gap.

### 9.2 Contract B — `ProductionQuality` (shipping sampler + quantization)

Two metric families, precisely separated: **teacher-forced distributional metrics** (logit KL/JS divergence per level and depth; top-k set overlap; reference rank of the selected token; first-divergence position distributions) and **free-running metrics under the default production sampler** (sequence WER; repeated/skipped-word rate; stop-token errors; speaker identity; prosody; **long-form drift — the §2.8 gate**; blind-listening equivalence per §9.4). Free-running comparisons are **distribution-level by design** (we do not replicate torch's RNG stream — a standing DISC): many seeds × paired texts, rank statistics and equivalence bounds over the metric distributions, never single-seed A/B. Full generated-token equality is a diagnostic here, **not** the INT8 shipping gate. Phase-5 **model surgery gets surgery-specific canaries** (high-frequency loss, identity, sibilance under shallow/adaptive residual depth), never only the generic listening battery.

### 9.3 Math modes & determinism scope

`strict`: canonical reductions, deterministic sampler, reproducible on a **fixed engine build + ISA path + sampler version + seed + artifact** (the full stated scope — cross-architecture byte identity is NOT claimed and would forbid useful fusions). `fast`: fused reductions, certified vector approximations, platform-optimal kernels, perceptually equivalent per Contract B. Cross-build outputs compare by content metric, never byte identity (equivalence-tier rule).

### 9.4 The listening protocol (replaces "blind listening finds no degradation")

- **ABX speaker-identity** (reference A; candidate B/C) and **pairwise/MUSHRA-style naturalness**;
- separate tests for model selection (§11), quantization gates, and enrollment preprocessing;
- **predefined smallest effect size of interest + equivalence bounds (TOST-style)** — we demonstrate equivalence, not failure-to-reject;
- hierarchical analysis over speaker, text, language, listener;
- **tail reporting** on the canary axes: noisy references, sibilants, breaths, code-switching, numbers, long form.

A powered design with correct grouping beats "several hundred undifferentiated judgments."

**Operationalized before first use** (fixed in `docs/CONFORMANCE_AND_LISTENING.md`, not improvised per test): the smallest effect size of interest + equivalence margin **per metric family**; the listener-pool recruitment plan and a named owner for the power analysis; the **automation boundary** — objective proxies (WER, spectral, secondary embeddings) run CI-nightly, human panels run at quantization gates and releases; and the **release binding** — AF-2's CVaR bound is a named bit in the ship gate, not advisory commentary. Without these, the boldest quality apparatus cannot actually gate a lever.

### 9.5 Differential, metamorphic, golden

Differential vs the pinned oracle and community conversions (frozen fixtures). **Valid metamorphic invariants (v2 — the trailing-punctuation relation is deleted; punctuation legitimately alters preceding prosody in a text-conditioned AR model; it becomes a behavior study):** batch==singleton (strict mode); packet-1 == packet-4 for the same codec tokens; prompt-cache == full prefill; streaming codec == offline codec; identical voice input → identical portable `.ftvoice`; scalar==SIMD; thread-count invariance (strict); corrupted/truncated artifacts always fail before inference; **greedy FrankenMTP ids ≡ greedy sequential ids; AF-3 α→0 ≡ the sequential path byte-for-byte in strict mode; verify-then-repair can never emit a depth sequence impossible under the training-mode causal mask**. Every optimized path (each packet size, batched execution, FrankenMTP) carries its **own** greedy token-identity proof against sequential — the gate is per-path, not global. Golden artifacts: exact codec-token streams; fuzzy/ULP logits; scrubbed robot NDJSON; per-platform PCM content hashes; `UPDATE_GOLDENS=1` with mandatory diff review, `*.actual` gitignored, CI never auto-updates.

### 9.6 Reliability & hostile-input gates (v2 addition)

- **Resource admission**: before synthesis, compute and validate token count, predicted max frames, KV allocation, codec state, voice-cache footprint, concurrency budget, hard duration limit, total memory reservation — reject before partial allocation.
- **Runtime health**: NaN/Inf checks at configurable seams; no-progress frame watchdog; EOS/max-duration consistency; repeated-token runaway detector; output-silence detector; **fallback from an optimized ISA kernel to the certified scalar baseline after selftest failure**; thermal-degradation reporting on sustained runs.
- **Streaming failure semantics** (defined, tested): consumer stops reading (producer parks; budget then cancels); sink write error (drain + structured error); event-sink blocking (audio and event channels may not deadlock each other — separate bounded queues, independent shutdown); cancellation during codec emission (finalize at frame boundary); disk-full (partial-WAV finalization with valid header).
- **Fuzzing**: all four artifact parsers, tokenizer metadata, WAV/FLAC, malformed UTF-8/extreme text, chunk-boundary schedules.
- **Sustained-performance gate**: a 30-minute (or fixed-long-corpus) thermal run in the scorecard — a turbo-window laptop number is not an interactive-TTS claim.

### 9.7 Ledgers & release certification

Artifact-graph `NEGATIVE_EVIDENCE.md` / `PERF_LEDGER.md` / `DISCREPANCIES.md` — per entry: claim_id, evidence_id, model-source commit + fixture hash, CPU feature string, exact command + env, kill-switch state, W/L/N tally (seeded day 0 with the sibling graveyard as `inherited (pre-truth-pack)` priors); `/running-the-gauntlet-on-your-rust-port` as the release back-end (three pillars, conformal lower bounds, e-processes over the standing invariants — now including *streaming==batch*, *admission-before-allocation*, *i32-overflow* at the census worst-case K — skip-honest receipts, convergence rounds).

---

## 10. Performance methodology

### 10.1 Phase −1B: the executable cost model comes FIRST (v2 centerpiece)

Before heroic kernels, generate from the pinned source + checkpoint a **costed execution graph**: per op — component, layer, residual depth, shape, dtype, executions/frame, weight bytes, activation bytes, MACs, KV bytes, predicted cache level, parallel dimension, kernel candidates. The generated report explicitly contains: talker cost/frame; **each of the 15 microdecoder depths**; total serial depth; codec cost per 1- and 4-frame packet; x-vector and ICL prefill costs; cold text-embedding traffic; max KV by duration; single- and N-stream traffic lower bounds. Then **instrument the official implementation and the best MLX/GGML ports** per stage (talker alone; every microdecoder depth; the 15 steps as a unit; codec packets; prefill; both encoders). The phase answers, with measurements:

1. Is the Q8 microdecoder cache-resident on each actual M4/M5 and AMD target?
2. Which hardware tier limits 1×/2×/5× real time?
3. W8A8 or W8A16 — which is faster at each real shape?
4. At what stream count does batched GEMM overtake per-stream GEMV?
5. Does codec overlap help or merely contend for bandwidth?
6. **Is 5× real time on M4 physically plausible without speculative MTP or Q4?**
7. **What is FrankenMTP's break-even acceptance α\*(SKU)** — from measured T_draft / T_verify / T_repair primitives (§7.5)?

Two experiment classes are mandatory, not optional: the **cross-stage cache-interference schedule** (microdecoder residency measured under the real talker→microdecoder→codec loop, §2.6/OQ-18) and the **acceptance-surface primitives** for question 7. Until this phase exists, `≥5× RT on M4` remains an ambitious stretch, **not a release promise**. Deliverable: `docs/PERFORMANCE_ARCHITECTURE.md` (§16.2).

### 10.2 The optimization ritual

Unchanged (the skill's Phase-4 loop: graveyard sweep → re-profile → one lever behind `FTTS_*` → bit-identical proof before speed → interleaved thermal-paired A/B, cv%≤5 → keep/revert + ledger → evidence bundle → equivalence class in the commit subject; PROVISIONAL_LOCAL_WIN discipline).

### 10.3 Head-to-head gauntlet

Unchanged fairness controls (thread/allocator/precision parity; pinned incumbent contract; per-stage AND per-profile rows; TTFA and RTF separate; NO ADMISSIBLE RATIO over guesses) — now also **per packet size** and **per profile**.

### 10.4 The re-ranked lever list (v2)

1. **Microdecoder cache residency + `ResidualCodeDecoder` specialization** (§7.4).
2. **`FrankenMTP` speculative block verification** (§7.5) — the flagship radical epic.
3. **Persistent talker loop on the `KernelTeam`** (§7.6, §7.10).
4. **Codec stateful streaming kernels + packet autotuning** (§7.9).
5. **Continuous batching for AMD throughput** (§7.11).
6. **`KernelPlan` autotuning (W8A8 vs W8A16 vs BF16 per op)** (§7.7).
7. **MTP-first Q4** (§6.4).
8. **Prefill tiled GEMM + prompt partial evaluation / voice-cache prefixes** (§6.7).
9. Norm→quant fusion (§7.8); argmax-only heads under greedy; PGO/BOLT.

### 10.5 Removed levers (ledgered as invalid, not merely deprioritized)

- ~~Batch-of-frames MTP evaluation~~ — residual codes are autoregressively dependent within the frame [SOURCE].
- ~~Speculative main-talker decode as the flagship~~ — **re-sequenced, not deleted**: FrankenMTP owns the flagship slot (better traffic math; a natural block verifier). After the residual layer is won, the talker still streams ≈440 MB/frame forever — talker-level speculation returns as the Phase-5 sibling epic (dual-level speculation over the whole hierarchical stack).

### 10.6 Honest perf hygiene

Unchanged (self-speedups are maintenance; claim-coverage audits; model-gated benches skip green; guardrail vs frozen baseline) + the sustained-thermal gate (§9.6).

### 10.7 Alien-artifact families (≤5, each with fallback)

Per Doctrine #0, **every AF names its consumer, its shipping gate, and its deletion condition** — otherwise it is unfalsifiable ambition decoration:

- **AF-1 — per-(tensor × residual-depth) bit allocation** (rate-distortion water-filling over the §6.3/§6.4 axes; residual depth is a semantic axis v1 missed). Fallback: uniform Q8. *Consumer*: Phase-4 `ftts convert --optimize-bits`. *Gate*: the §6.4 double gate. *Delete if*: Q4 never passes and uniform Q8 ships permanently.
- **AF-2 — tail-risk gates (CVaR/EVT) on WER + identity + drift**, gating releases on the worst α-fraction, with the §9.4 canary axes. Fallback: lever off / one tier higher. *Consumer*: the release ship gate — **a named bit** (§9.4). *Delete if*: never, while any lossy lever ships (it IS the tail gate).
- **AF-3 — sequential-test (e-value) reliability monitor for FrankenMTP**: bounds the in-production draft-misbehavior rate at risk α; α→0 disables speculation. **Explicitly NOT the exactness proof — that is OQ-19** (§7.5). Fallback: exact sequential microdecoder. *Consumer*: the FrankenMTP kill-switch. *Delete if*: FrankenMTP is abandoned.
- **AF-4 — voice-compiler enrollment optimization** (the kernel AFs' peer — the second thesis gets equal formal treatment): submodular segment / multi-reference selection under a quality-coverage objective, plus an explicit **attribute→quality loss model** (reference SNR/reverb/clipping/phonetic coverage → predicted identity + naturalness, calibrated on the audition benchmark). Fallback: whole-file enrollment. *Consumer*: `ftts enroll` ranking (§8.6). *Gate*: the audition benchmark. *Delete if*: selection never beats whole-file.
- **AF-5 — USL pool/team sizing** per (arch, op-class) feeding the `KernelTeam` per-op worker counts, the N-bucket partition tables (§7.10), and the admission policy. Fallback: physical-core count. *Consumer*: `.fttspack` `pool_sizing` + `robot backends`. *Delete if*: fixed counts measure equal.

Each family ships a transparency card + assumptions ledger + deterministic fallback wired first.

---

## 11. The bakeoff — three sequential gates

v1's single kill rule was circular ("2× after equivalent optimization" judged before any optimization). Replaced:

- **Gate A — upstream quality** (runs during Phases −1A/−1B, official implementations only): identity, naturalness, prosody, intelligibility, multilingual behavior, enrollment sensitivity, under the §9.4 protocol on the full corpus (50–100 speakers; 3/10/30 s references; clean/phone/reverberant/noisy; prose through numbers/URLs; emotional + neutral; same- and cross-language). **Pocket is eliminated as primary if it fails a predeclared quality floor** (identity equivalence bound vs Qwen on the intended workload).
- **Gate B — architectural systems potential** (uses the −1B cost model, no full ports): exact op counts, measured reference profiles, representative GEMV/GEMM microbenchmarks, codec microbenchmarks, cache-fit models, footprints, rooflines — organized around an explicit **stage isomorphism map** between the two architectures (text/prefill ↔ text/prefill; talker backbone ↔ Pocket backbone; microdecoder ↔ consistency frame head; RVQ codec ↔ causal VAE; streaming ↔ streaming), so the continuous-frame vs discrete-16-group comparison is structural, not rhetorical. Answers whether Pocket's structure justifies parallel investment — without pretending either side is "equivalently optimized."
- **Gate C — optimized confirmation** (only if A and B both pass): after the champion's microdecoder/talker path and Pocket's corresponding core path have *representative* optimized implementations, the final quality/performance decision under the original kill rule's spirit — **displacement requires BOTH** ≥ ~95% human-preference retention on speaker identity + naturalness on the intended workload **AND** a ≥2× end-to-end systems advantage. Failing either, Qwen stays champion. The timebox bounds the *decision evidence* only: **a mixed verdict (Pocket wins some regimes) hard-requires the dual-engine product** — Pocket as the ultra-edge profile inside the same CLI, onboarded through the skill's full add-model recipe with complete certification, never a handicap-match second engine.

Qwen3-TTS 1.7B remains the upper-bound reference. Artifacts under `docs/bakeoff/`; Gate A+B verdicts are the Phase-3 entry evidence.

---

## 12. Phased roadmap

Per-component truth gates throughout (v2): each component's kernels are blocked only by *its* unresolved [OPEN]s.

### Phase −1A — Exact model truth
Pin source + weights (README-derived runtime pins, asserted at oracle runtime); hash everything; resolve the tensor inventory (OQ-2); capture prompt construction for **both cloning modes × both streaming modes**; resolve ID mappings (semantic/codec token spaces); produce the exact operator DAG; **promote every §2 [SOURCE] row to [VERIFIED] or correct it**; extract `docs/QWEN3_TTS_EXECUTABLE_SPEC.md` (per `/porting-to-rust`: implement from the spec, never line-by-line). Seed ledgers with inherited priors. **Oracle before engine (the skill's Phase 0 — it lives here, not implicitly)**: stand up the pinned oracle environment (`gen_reference_fixtures.py` — per-stage activation dumps and teacher-forced fixtures for talker, every microdecoder depth, codec blocks, both encoders); **measure the oracle's OWN nondeterminism floor** (two runs × two thread counts; every Contract-A tolerance derives from that committed envelope, never imported from a sibling project); freeze the conformance golden corpus; decide the canonical conformance decoder (OQ-12). **Exit**: per-component truth gates green for talker, microdecoder, codec, encoders, prompt builder; oracle fixtures + the nondeterminism-floor envelope committed.

### Phase −1B — Cost & feasibility (new)
The §10.1 costed execution graph + official/MLX/GGML stage instrumentation + M4/M5/AMD memory-hierarchy measurements + microdecoder cache-residency experiments + packet-size experiments + W8A8-vs-W8A16 microbenchmarks + **the Metal microdecoder spike (§7.12)** + revised performance targets. Bakeoff **Gate A** (needs only the upstream stacks — may start alongside −1A) concludes here; **Gate B** runs on the −1B cost model. **Exit**: the seven §10.1 questions answered; `docs/PERFORMANCE_ARCHITECTURE.md` committed; numeric G2 targets (including the roofline-efficiency factors) set.

### Phase 0 — Minimal executable skeleton
Workspace (§5.1) compiles on all 5 targets; library API; one CLI with both shims; artifact stubs; robot mode + schema; error contract; cancellation; model-gated tests; `scripts/check.sh`. **No database history, no sync.** **Exit**: builds green on all 5 targets; `robot schema/health/backends` emit valid versioned JSON; the empty pipeline skips-green without weights.

### Phase 1 — Exact safe forward
Upstream-faithful text/prompt path (verbatim mode); talker (mRoPE); **sequential 15-step microdecoder**; codec (streaming + offline); both clone modes; ECAPA + codec encoders; ConformanceExact green end-to-end; **zero-copy BF16-resident weights** (widen at the accessor — no whole-model f32 expansion). Rayon-parallel is fine here; `KernelTeam` comes in Phase 3.

### Phase 2 — Canonical Q8 artifact
Streaming bounded-memory converter (§6.5); runtime quant first, converter reuses it; W8A8/W8A16 candidates wired; canonical `.fttsq` + local `.fttspack`; hardened parsers + fuzzing (§6.6); staged levers (talker → microdecoder → heads gated → codec Q8) each with ConformanceExact + Contract-B gates; text-embedding Q8 experiment.

### Phase 3A — Microdecoder engine
`ResidualCodeDecoder` + MTP hot pack + per-depth profiling + `KernelTeam` bring-up + **FrankenMTP spike** (drafter #1: previous-frame + transition model — no lab training needed) + MTP-first Q4 experiments (double-gated).

### Phase 3B — Talker & prefill engine
Exact-shape GEMV; §7.8 norm/quant fusion; text projection; prompt partial evaluation + voice-cache prefixes (§6.7); long-context KV layout.

### Phase 3C — Codec & streaming
Packet autotuning; stateful conv; streaming==offline proof; backpressure + cancellation semantics (§9.6); sustained streaming tests.

### Phase 3D — Server throughput
Continuous batching (§7.11); NUMA-local packs; admission control; queueing policy; AMD capacity curves; the capacity certificate.

*(3A–3D have independent entry gates; 3A is the critical path. Bakeoff Gate C runs — only if Gates A and B both passed — once 3A exists plus a **timeboxed, representative** optimized Pocket core path whose scope Gate B's findings define; it is explicitly bounded work, not a second full port.)*

### Phase 4 — The voice compiler
Segment discovery; transcript verification; multi-reference selection-then-aggregation; portable/cache artifact split; privacy modes; `docs/VOICE_COMPILER_DESIGN.md` finalized.

### Phase 5 — Structural compression & the full hierarchical-AR assault (`franken_tts_lab` outputs land here)
Speculative-MTP drafters #2–#5 — including **oracle residual-trajectory distillation** (teacher-forced residual trajectories from the pinned oracle are free supervision), cross-frame residual dynamics beyond prev-frame copy, and **block drafts of 3/5 with tree verify**; **talker-level speculation as the sibling epic** (dual-level speculation over the whole hierarchical stack — the talker's ≈440 MB/frame is forever otherwise); joint talker→residual conditioning compression; adaptive/depth-conditional early exit **with the §9.2 surgery canaries**; per-depth precision via AF-1; talker Q4 if still worthwhile after MTP-first results; predictor pruning. **Lab → runtime ABI**: draft/surgery models ship as a versioned **`.fttsdraft`** artifact beside `.fttsq` (format-versioned, compatibility-keyed to model + engine ABI, kill-switched) — so structural wins land as product, not notebooks. Every lab output re-enters through the full §9 gates. If §2.8's dual-rate trigger fired, the 25Hz document-mode engine runs here as its own epic.

### Phase 6 — Metal productization
Only if the −1B spike was positive, under a **pre-stated full-path contract** (so Metal becomes a second full path, not a residual-only toy): the integer cross-backend contract (int8×int8→i32, post-accumulation scales ⇒ exact CPU↔GPU equality); a talker/codec co-residency plan alongside the persistent command-graph microdecoder; CPU-fallback invariants (CPU stays default, every Metal path has a CPU twin, selftest covers both); install-time CPU-vs-Metal policy. Negative verdicts validly end it.

### Phase 7 — Certify + ship
Gauntlet convergence; ladder receipts (skip-honest); 5-target release + installer + checksums; distribution proofs (convert byte-parity vs published hash; clean-cache pull + real synthesis); sustained-thermal scorecard row; selftest on user silicon.

---

## 13. Risks & mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Microdecoder bandwidth wall** — 1.18 GB/frame × 12.5 breaks real-time on weak memory systems | **HIGH** | It is now the #1 design center: cache-residency measurement (−1B), hot-pack, MTP-first Q4, FrankenMTP; per-SKU honesty in the cost model — no blanket RTF promises. |
| **FrankenMTP acceptance too low** (draft quality) | MED-HIGH | Exact sequential path always authoritative; AF-3 bounds disagreement; drafter ladder from free (prev-frame) to trained (lab); a negative verdict is a valid ledgered outcome. |
| **[SOURCE] facts drift before pinning** | MED | Phase −1A re-asserts every row against the pinned revision; per-component truth gates. |
| **Sampled reference vs exact parity** | HIGH → managed | The two-contract split (§9.1/§9.2) is the fix; canonical greedy for Contract A; distributional metrics for B. |
| **Codec dominates after talker+MTP work** | MED-HIGH | Co-equal from day one; −1B measures it; packet adaptivity. |
| **Quantization audibly damages identity (tails)** | HIGH | §9.4 equivalence protocol + AF-2 tail gates + canary axes; staged levers; per-depth allocation; f32 escape hatch. |
| **12Hz long-form drift** (25Hz won the paper's long-speech eval) | MED-HIGH | Long-form is a distinct gated regime (§2.8); drift metric in Contract B; 25Hz engine eligibility rule predefined. |
| **Streaming state bugs** | MED-HIGH | streaming==batch gate; chunk-boundary fuzzing; §9.6 failure semantics. |
| **ICL prompt-cache invalidation subtleties** | MED | §6.7: cache only the proven target-independent prefix; regenerable caches keyed by full compatibility tuple. |
| **Enrollment variance dominates perceived quality** | MED | The voice compiler (§8.6) is the second thesis; audition-set ranking. |
| **`forbid`/unsafe architecture contradiction** | resolved | Workspace split (§5.1). |
| **Scheduler complexity (3 systems)** | resolved | `KernelTeam` owns the hot loop; asupersync orchestrates; rayon retained off the hot path. |
| **License/attribution slip** | MED | OQ-1 verbatim verification; NOTICE embedded in `.fttsq`; release checklist. |
| **Misuse (non-consensual cloning)** | MED | Doctrine #10; consent attestation; no acquisition features; watermark preserved (OQ-8). |
| **Deadlock** | MED | Architectural fix + watchdog. |
| **Scope creep** | MED | Non-goals §1.2; the lab track is fenced to §7.5's program; zoo behind the three gates. |

---

## 14. Open research questions register

Per-component blocking (each OQ blocks only its dependents). Promoted-to-[SOURCE] items from v1 are gone from this list — they are §2 facts pending only hash-pin re-assertion in −1A.

| ID | Question | Blocks |
|----|----------|--------|
| **OQ-1** | Pinned revision; LICENSE verbatim; NOTICE obligations | artifact publishing |
| **OQ-2** | Full tensor inventory (names/shapes/dtypes) incl. per-depth embeddings/heads, codec, encoders — **and the exact microdecoder hot-working-set census in bytes** (body + depth embeds + depth heads + scales + RoPE table + KV/scratch; one number, §2.6) | converter, manifest census, §7.4 hot pack |
| **OQ-3** | Talker residuals: QK-Norm presence/eps, MLP gating details, biases | talker.rs L1/L2 |
| **OQ-4** | The exact mRoPE position *schedule*: `rope_deltas`, 3-D id assignment over real prompts, prompt→audio transitions, `position_id_per_seconds` semantics | talker attention, prefill |
| **OQ-5** | Microdecoder wiring details: exact conditioning inputs per depth (talker hidden? primary-code embedding?), KV reset semantics, head weight layout, the training-mode causal pass's exact masking + position ids + depth→sequence→head index map. **Training-mask/distribution equivalence decides the FrankenMTP claim tier (§7.5: exact verifier vs approximate assistance)** — the highest-blast-radius OQ in the register | microdecoder.rs, §7.5, the flagship epic |
| **OQ-6** | Context limit; long-form chunking policy; EOS/stop semantics; max-duration behavior | decode.rs, admission (§9.6) |
| **OQ-7** | Codec implementation details: exact hop/upsample math (→1,920), conv receptive fields, causality/lookahead, semantic-ID mapping, activations | codec.rs, ring buffers, overflow census |
| **OQ-8** | Output watermarking? (preserve if present) | doctrine #10 |
| **OQ-9** | ECAPA feature-extraction parity (mel params, normalization) + conditioning injection point | speaker.rs, x-vector path |
| **OQ-10** | ICL prompt structure per (clone mode × streaming mode); the maximal target-text-independent prefix | prompt.rs, `.ftvoice-cache` |
| **OQ-11** | Text tokenizer files; upstream normalization responsibilities (feeds §8.7 `verbatim`) | text front end, L0 |
| **OQ-12** | Define the **canonical conformance decoder** (greedy at both levels — is greedy audio quality-viable for goldens?) and the production sampler contract (defaults are known [SOURCE]) | §9.1/§9.2 definitions |
| **OQ-13** | GGML conversion's exact kept-high-precision set | §6.3 validation |
| **OQ-14** | Official streaming internals: first-packet path, flush behavior, prompt differences vs non-streaming | stream contract, TTFA |
| **OQ-15** | Oracle runtime pins; does official CPU reproduce its own GPU tokens? | §9 oracle, G2 baseline |
| **OQ-16 [SOURCE; partially resolved — Gate A access BLOCKED 2026-08-05]** | Pocket: weights gating/licensing (needed just to RUN it in the bakeoff); voice-state format. See the resolution immediately below. | bakeoff Gate A; Phase-5 track |
| **OQ-17** | Energy/thermal measurement method (joules/generated-minute) | §15, bakeoff harness |
| **OQ-18** | Per-SKU cache-residency of the **~110 MB-class hot working set under talker interference** (M4/M5 SLC; AMD L3-per-CCD; Intel) — operationalized as measured DRAM bytes/frame vs the one-read floor (§7.4) | §7.4 hot-pack design, G2 targets |
| **OQ-19** | The **named speculative-sampling algorithm + distributional-correctness proof** for top-k/top-p + repetition-penalty at both levels (accepted prefixes ≡ sequential sampler's distribution; rejections from the verifier's adjusted conditional). A research deliverable; **AF-3 is reliability-only and never substitutes** | §7.5 sampled mode |

### OQ-16 resolution — Pocket access, license, and voice state **[SOURCE; checked 2026-08-05]**

- **Access / Gate A status:** the official `kyutai/pocket-tts` HF API declares `gated: "auto"` and `license: "cc-by-4.0"`; its anonymous weight resolver returns `401`, `GatedRepo`, and says that authentication plus model access are required. This workspace has neither an HF credential nor a local Pocket cache, so **the weights have not been obtained and Gate A remains BLOCKED**. A human with authority to accept the model gate must authenticate to HF, accept the displayed terms, download the required voice-cloning revision, and record its immutable revision plus SHA-256 before Gate A runs. The displayed gate separately prohibits unlawful/non-consensual cloning and other harmful/deceptive use; it does not relax the project's consent rule.
- **Redistribution:** CC BY 4.0 permits reproduction, sharing, and adapted material, but sharing the original or a converted/quantized artifact must retain supplied creator/copyright/notice/URI information, identify modifications, and state/link the CC BY 4.0 license. Therefore a hypothetical Pocket `.fttsq` / dual-engine distribution needs a Pocket-specific NOTICE containing Kyutai attribution, the model URL and pinned source revision, `CC-BY-4.0`, the conversion/quantization statement, and the original gate-use notice when supplied. This is a license-obligation summary, not legal advice; re-check the gated model card and included notices at the accepted pin.
- **Voice-state format / Gate B mapping:** upstream `export_model_state` writes a flat safetensors map with keys `<StatefulModule-name>/<tensor-key>`. For each FlowLM attention module, the reusable state is `cache` shaped `[2, batch, T, heads, head_dim]` (K/V) plus `offset`; current export writes `<module>/offset`. Import also accepts legacy `<module>/current_end` and reconstructs `offset` from `current_end.shape[0]`, then loads the remaining tensors and expands a sliced cache with NaN capacity before generation. The state is created by Mimi-encoding the reference audio, projecting it into FlowLM conditioning, then prompting the FlowLM; it is thus a **derived prompt/KV cache**, not an audio/provenance-bearing portable voice recipe.
- **Compatibility consequence:** upstream stores no model revision, model/config hash, dtype declaration, prompt-builder version, or engine ABI in this safetensors payload, and its `current_end` import is already a compatibility migration. Treat Pocket voice-state files as untrusted, engine-specific `.ftvoice-cache` inputs: reject malformed names/shapes/dtypes, require an external full compatibility key, and regenerate from an owned, consented source voice on any mismatch. Do not map them directly to portable `.ftvoice`.
- **Sources:** HF model metadata/gate: `https://huggingface.co/api/models/kyutai/pocket-tts`; official resolver endpoint: `https://huggingface.co/kyutai/pocket-tts/resolve/main/tts_b6369a24.safetensors`; Pocket source pinned for this reading: `https://github.com/kyutai-labs/pocket-tts/commit/d108410d23eef7e01db282f9442891162dbc3db6` (`README.md`, `pocket_tts/models/tts_model.py`, `pocket_tts/modules/transformer.py`, `pocket_tts/modules/stateful_module.py`); CC BY 4.0 legal code: `https://creativecommons.org/licenses/by/4.0/legalcode`.

---

## 15. Success metrics

**Correctness (G1)** — ConformanceExact all green (teacher-forced ladders; canonical greedy exact; streaming==offline; scalar==SIMD; artifact idempotence). ProductionQuality within budgets: WER band, ABX identity equivalence, prosody, **long-form drift within its gate**, tail (CVaR) bounds on the canary axes.

**Performance (G2/G8)** — set numerically by the −1B cost model, then gated: per-stage TTFA + RTF vs the proven baseline under `interactive`; under `throughput`: **aggregate RTF (generated-audio-seconds per wall-clock second per socket)** as the headline, plus streams/socket, **p95 admission-to-first-packet queueing latency under load**, and joules/generated-minute; sustained-thermal row green; **roofline-efficiency rows per hot stage per SKU** (the structural claim — measured distance from the −1B floor, not just incumbent deltas). Standing stretch (not a promise until −1B says plausible): first packet ≲ 200 ms warm and ≥5× RT single-stream on M4-class; the honest fallbacks are whatever the roofline permits, published per SKU.

**Footprint (G3/G4)** — one self-contained binary per target (per the corrected G3 wording); `.fttsq` ≈ **1.6 GB-class at the §6.3 default** (Q8 talker + microdecoder + codec with the text embedding kept BF16 — careful with the community figure: its 993 MB Q8 talker *includes* a Q8 text embedding (~311 MB of the ~311M embedding params), so keeping the embedding BF16 adds ≈ 310 MB on top), dropping toward ≈ **1.3 GB-class** if the Q8 text-embedding experiment passes; `.fttspack` regenerable; workspace safety story per §5.1.

**Cloning quality** — voice compiler: enrollment from 3 s references produces ABX-recognizable identity vs the reference stack at equal quant; segment discovery beats naive whole-file enrollment on the audition benchmark; both modes honest about their quality difference.

**Agent ergonomics & honesty (G5/G7)** — stable robot schema (contract-tested); stateless default; every divergence/lever ledgered; claims at their equivalence tier; gauntlet green before ship.

---

## 16. Skills, methodology, companion documents & the path to beads

### 16.1 Governing skill

`/ai-model-into-rust-mega-fused-hyper-kernel` remains the spine: its Loop maps to §12 (−1A/−1B are its Phase −1/0 with the v2 cost-model extension; 3A–3D its Phases 3–4; the lab track its innovation harness; 7 its certify-ship). Its Nine-Law Doctrine + Doctrine #0 (anti-ceremony) live in `AGENTS.md`. Agents route through its First-30-Seconds table; this plan is the *what*, the skill is the *how*.

### 16.2 Companion documents (deliverables, not prose-splitting)

The v2 review recommends splitting the plan; we adopt it as **generated deliverables with owners and phases** (this file stays the master thesis/roadmap):

| Document | Content | Produced by |
|---|---|---|
| `docs/QWEN3_TTS_EXECUTABLE_SPEC.md` | the exact pinned model graph, prompt formats, ID maps | Phase −1A |
| `docs/PERFORMANCE_ARCHITECTURE.md` | cost model, measured stage profiles, kernel plans, execution policies | Phase −1B, living |
| `docs/CONFORMANCE_AND_LISTENING.md` | Contract A/B details, math modes, the listening protocol | Phase −1A/0 |
| `docs/VOICE_COMPILER_DESIGN.md` | enrollment pipeline, artifact split, privacy modes | Phase 4 (skeleton earlier) |

### 16.3 Supporting skills

`/porting-to-rust` (spec-first), `/running-the-gauntlet-on-your-rust-port` (release), testing skills (conformance/golden/metamorphic/fuzzing), profiling + extreme-optimization (under the Phase-4 ritual), `/alien-artifact-coding` + `/alien-graveyard` (AF families), unsafe/UB exorcists (after intrinsics), asupersync-mega-skill, beads/bv/cass.

### 16.4 The path to beads — evidence gates, not round counts (v2)

**STATUS UPDATE (2026-08-05): the conversion has been executed** — by owner decision, the full graph now lives in `.beads/` (15 epics + 106 tasks, slug-embedded IDs prefixed `frankentts-`, multiple polish rounds applied; start with `br ready`). The evidence gates below therefore now gate the **execution of kernel-implementation beads**, not the conversion itself — a kernel bead may be *picked up* only when its component's gates are green (the graph encodes this as dependencies on the −1A/−1B beads):

- model graph reconciled with pinned source (−1A green for the component);
- cost model generated (−1B);
- no unresolved contradiction in the component being implemented;
- oracle fixture exists for it;
- acceptance criteria are executable.

Bead granularity (v2): **an implementation issue owns its test and benchmark criteria**; separate test/bench beads only for independently reusable infrastructure or genuine cross-cutting gates (the ladder runner, the listening harness, the fuzz corpus). Epics: −1A truth, −1B cost model, bakeoff (Gates A/B fences), skeleton, exact forward, Q8 artifact, 3A–3D engines, voice compiler, lab, certification. Every §14 OQ is a research bead blocking only its dependents. Validate `bv --robot-insights | jq '.Cycles'` empty before implementation.

---

## 17. Source links (promoted facts)

Recorded by the v2 review; re-fetch, hash, and cite line-level in the −1A truth pack:

- Official modeling source (microdecoder loop, `max_new_tokens = num_code_groups − 1`, per-depth heads, both rotary paths, ICL prompt construction): `https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/qwen_tts/core/models/modeling_qwen3_tts.py`
- Talker config (mRoPE theta 1e6, sections [24,20,20], `position_id_per_seconds` 13, 151,936×2048 text embedding, projection MLP): `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/config.json`
- Speech-tokenizer config (24 kHz, 1,920 samples/frame, 16 quantizers, 8-layer decoder, 512/1024, 16×64 heads, window 72, upsample [8,5,4,3]): `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/speech_tokenizer/config.json`
- Generation config (talker T0.9/k50/p1.0/rep1.05; residual T0.9/k50/p1.0): `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/main/generation_config.json`
- Paper (4-frame/320 ms packets; 25Hz-vs-12Hz long-speech result): `https://arxiv.org/html/2601.15621v1`
- Official repo (x-vector vs ICL quality note, robustness claims): `https://github.com/QwenLM/Qwen3-TTS`
- Rust lint levels (`forbid` un-overridable): `https://doc.rust-lang.org/stable/rustc/lints/levels.html`

---

*End of plan v2. The thesis in one line: turn the hidden 15-step residual-code microdecoder from Qwen3-TTS's largest CPU liability into its largest optimization advantage — cache-resident, per-depth-quantized, speculatively block-verified — while the two-contract conformance regime and the listening protocol keep every claim honest. The working engine is the deliverable; the machinery exists to keep it honest, never to replace it.*
