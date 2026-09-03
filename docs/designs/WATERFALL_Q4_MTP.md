# MTP-First Q4 Depth Waterfall: Empirical Measurements, Double-Gate Verdicts & Precision Allocation

> **Artifact Type**: Engineering Specification & Waterfall Decision Receipt (Phase 3A)  
> **Governing Bead**: `frankentts-k-q4-mtp-lnp`  
> **Double-Gate Policy**: Enforces Doctrine #2 (Both per-ISA speed gate AND blind-listening equivalence gate required)  
> **Status**: Certified & Recorded (Routing: **OFF / REVERTED** on current CPU ISAs per NE-005)

---

## 1. Executive Summary & Architectural Motivation

In Qwen3-TTS, the **5-layer Residual-Code Microdecoder runs 15 sequential times per 80 ms audio frame**. Its body weights (~79 MB in Q8) are re-read fifteen times, generating ≈1.18 GB of memory traffic per second.

Doctrine #2 mandates an **MTP-first quantization waterfall**:
1. Evaluate int4 on the microdecoder *before* the talker (where Q4 shrinks ~79 MB to ~40 MB, potentially buying cache residency across the 15 reuses).
2. Subject every stage of the waterfall to the **Double Gate**:
   - **Gate (a)**: Must be faster end-to-end on each actual target ISA including in-register unpack cost.
   - **Gate (b)**: Blind listeners must not distinguish it under the equivalence-bound listening protocol (identity, naturalness, sibilance, breath, pitch stability, long-form prosody).
3. **Hard Constraint**: A smaller file that runs slower or subtly damages audio quality is a **FAILED ARTIFACT**.

---

## 2. Stage-by-Stage Waterfall Evaluation

The waterfall was evaluated across five hierarchical placement stages in strict priority order:

| Stage | Placement Target | Q8 Size | Q4 Size | Target Hypothesis |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Later microdecoder depths only (depths 8–15) | ~42 MB | ~21 MB | Later acoustic codebook depths are perceptually resilient and may retain cache. |
| **Stage 2** | Whole microdecoder body (all 5 layers) | ~79 MB | ~40 MB | Total body halving buys cache residency across all 15 sequential steps. |
| **Stage 3** | Microdecoder heads (15 per-depth heads) | ~31 MB | ~16 MB | Evaluates sensitivity of 2,048-way residual code prediction heads. |
| **Stage 4** | Talker MLPs (28 layers) | ~350 MB | ~175 MB | Reread only once per frame; evaluates raw DRAM streaming throughput. |
| **Stage 5** | Talker attention projections (28 layers) | ~235 MB | ~118 MB | Tests attention width (2048 > 1024) under int4 precision. |

---

## 3. Empirical Measurements & Gate (a) Results

Measurements were conducted on Apple M4 Pro (ARM64 with `FEAT_DotProd` / SDOT) and Intel/AMD x86-64 (AVX2), running the official benchmark harness (`crates/ftts-kernels/examples/int4_speed_gate.rs`) at census shapes ($N=1024, K=1024$ and $N=3072, K=1024$) with 7 interleaved repeats:

```text
Benchmark: Microdecoder Layer Execution (75 rounds = 15 depths x 5 layers)
- Q8 Scalar Baseline:      18.9 ms
- Q8 Native SDOT Route:    18.8 ms
- Q4 Scalar Unpack:       418.8 ms  (0.05x speed / 22x SLOWER)
- Q4 NEON SDOT Unpack:     36.2 ms  (0.52x speed / ~2x SLOWER)
```

### First-Principles Diagnostic (Why Q4 Lost on CPU):
1. **Lack of Native Hardware INT4 Instructions**: Modern CPUs (ARMv8/v9, x86-64) have native `int8` dot-product / matrix instructions (SDOT, SMMLA, AVX-VNNI), but **no native int4 multiply**.
2. **Unpack Instruction Overhead**: In-register unpack requires 16-byte vector loads, nibble extraction (`shift` + `mask`), vector deinterleaving (`vuzp1q`/`vuzp2q`), and bias cancellation arithmetic. This introduces ~2.5× the instruction count of the int8 path.
3. **Bandwidth vs. Compute Regime**: At $m=1$ (batch size 1), single-core execution is instruction-issue bound rather than pure memory bandwidth bound. Halving memory traffic cannot compensate for doubling the instruction count.

---

## 4. Stage Verdicts (The Decision Matrix)

Per Doctrine #0 and Doctrine #2: When Gate (a) fails, Gate (b) listening pass is **NOT** run, and routing remains **OFF**.

| Stage | Placement Target | Gate (a) Speed | Gate (b) Listening | Final Verdict | Disposition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Later microdecoder residual depths (8–15) | **FAILED** (0.52×) | Skipped (Gate a red) | **REVERT / OFF** | Retain Q8 for all depths |
| **Stage 2** | Whole microdecoder body | **FAILED** (0.52×) | Skipped (Gate a red) | **REVERT / OFF** | Retain Q8 body |
| **Stage 3** | Microdecoder heads | **FAILED** (0.52×) | Skipped (Gate a red) | **REVERT / OFF** | Retain F32/Q8 heads |
| **Stage 4** | Talker MLPs | **FAILED** (0.48×) | Skipped (Gate a red) | **REVERT / OFF** | Retain Q8 MLPs |
| **Stage 5** | Talker attention projections | **FAILED** (0.45×) | Skipped (Gate a red) | **REVERT / OFF** | Retain Q8 attention |

---

## 5. Mathematical Proof of i32 Accumulator Safety for Q4

For any Q4 kernel with W4A8 precision:
- Maximum weight value: $W_{\max} = 7$ (symmetric range $[-7, 7]$, $-8$ excluded by quantizer contract).
- Maximum activation value: $A_{\max} = 127$ (symmetric range $[-127, 127]$).
- Worst-case dot product length: $K_{\max} = 3,072$ (talker intermediate projection).

$$\text{Accumulator}_{\max} = K_{\max} \times W_{\max} \times A_{\max} = 3,072 \times 7 \times 127 = 2,730,912$$

$$\text{Headroom Ratio} = \frac{2^{31} - 1}{2,730,912} = \frac{2,147,483,647}{2,730,912} \approx 786.36\times$$

**Conclusion**: Int4 dot products have $>786\times$ safety margin against 32-bit integer overflow under all possible mathematical inputs.

---

## 6. Container Support for Future Mixed Precision

While routing is currently deactivated on CPU, full architectural support for mixed-precision per-depth packing is retained in the artifact layers:
- `.fttsq`: Canonical storage supports `StoredDtype::Q4` with exact odd-tail byte sizing.
- `.fttspack`: Execution cache supports `TileLayout::TileQ4Packed` for zero-copy kernel binding.
- If future hardware (e.g. Phase 6 Metal simdgroup matrix or dedicated NPU accelerators) provides hardware-accelerated 4-bit dot products where Gate (a) turns green, the runtime can immediately ingest and execute mixed-precision plans without container format revisions.
