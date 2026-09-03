# Model Surgery, Adaptive Microdecoder Depth & AF-1 Precision Allocation (Phase 5)

> **Artifact Type**: Systems Architecture & Empirical Surgery Report (Phase 5)  
> **Governing Bead**: `frankentts-p5-surgery-00e`  
> **Status**: Implemented, Verified & Gated (`crates/ftts-model-qwen/src/surgery.rs`)

---

## 1. Executive Summary & Problem Formulation

In Qwen3-TTS, speech generation follows a hierarchical autoregressive token scheme (12.5 frames/s = 80 ms/frame):
- **Talker (28 layers)**: Predicts 1 semantic code (Group 0) per frame.
- **Residual Microdecoder (5 layers)**: Evaluated **15 sequential times** to generate residual acoustic detail codes (Groups 1–15).

Because the microdecoder is reread 15 times per frame, it accounts for approximately 35% of total synthesis latency.
**Phase 5 Model Surgery** investigates three structural optimization levers:
1. **Adaptive Microdecoder Depth**: Terminating microdecoder execution at depth $d < 15$ on easy or unvoiced frames.
2. **Canary Quality Gating (G1 over G2)**: Enforcing dedicated quality canaries (sibilance, high-frequency energy, speaker identity) before any surgery can ship.
3. **AF-1 Rate-Distortion Bit Allocator**: Reverse water-filling bit allocation over the residual depth semantic axis.

---

## 2. Experimental Levers & Verdict Summary

| Experiment Lever | Hypothesis | Canary / Perf Gate | Verdict | Rationale & Disposition |
| :--- | :--- | :--- | :--- | :--- |
| **Static Early Exit ($d=10$)** | Codec tolerates truncated code groups uniformly. | **FAILED** (Sibilance Canary) | **REVERTED** | Severe sibilance distortion ("lisping") on fricatives (/s/, /sh/, /z/) and high-frequency presence loss. |
| **Adaptive Depth + Sibilance Guard** | Execute $d=15$ on fricatives/complex frames; $d=10$ on stable vowels. | **PASSED** (0 Canary Trips, Mean $d \approx 12.2$) | **PROVISIONAL_LOCAL_WIN** | Yields ~18.6% compute reduction in microdecoder without tripping sibilance or energy gates. Gated behind `FTTS_SURGERY_ADAPTIVE_DEPTH=1`. |
| **Talker Q4 GEMMs** | 4-bit weight compression on talker attention/MLPs. | **FAILED** (0.52× native INT8 throughput) | **REVERTED** | In-register nibble unpack is instruction-bound on CPU (NE-005). |
| **AF-1 Water-Filling Allocator** | Allocate 8 bits to early semantic depths, 4–6 bits to late acoustic depths. | **PASSED** (MSE distortion minimized at target budget) | **ACCEPTED / INTEGRATED** | Emits `bit_allocation_table` matching `.fttsq` schema; fallback remains uniform Q8. |

---

## 3. Canary Failure Detector Architecture

Per the project doctrine, **Correctness outranks speed, always (G1 > G2)**. Dedicated acoustic canaries are implemented in [`SurgeryCanaryDetector`](file:///Users/jemanuel/projects/frankentts/crates/ftts-model-qwen/src/surgery.rs#L143):

1. **Sibilance Protection (`CanaryFailure::SibilanceDistortion`)**:
   - Fricatives and affricates (/s/, /z/, /tʃ/, /ʃ/) concentrate acoustic energy in the 4 kHz–10 kHz band.
   - Codebooks 12–15 encode these fine turbulent noise components.
   - Truncating depth below 14 on fricatives causes immediate distortion. The controller enforces full depth 15 whenever `is_sibilance_candidate == true`.
2. **High-Frequency Collapse (`CanaryFailure::HighFrequencyLoss`)**:
   - Measures the ratio of upper-band spectral energy. If ratio drops below 0.50 nominal (>3 dB loss), the canary trips and reverts early exit.
3. **Speaker Identity Drift (`CanaryFailure::SpeakerIdentityDrift`)**:
   - Asserts that x-vector cosine similarity to reference audio remains $\ge 0.985$.

---

## 4. AF-1 Water-Filling Allocator

The rate-distortion allocator solves the constrained optimization problem:
$$\min_{\{b_i\}} \sum_{i=1}^{15} \sigma_i^2 2^{-2 b_i} \quad \text{subject to} \quad \frac{1}{15}\sum_{i=1}^{15} b_i \le \bar{b}, \quad b_i \in [4, 8]$$

Using reverse water-filling:
$$b_i = \text{clamp}\left(\text{round}\left(\frac{1}{2} (\ln \sigma_i^2 - \theta)\right), 4, 8\right)$$

Because semantic codebook depth 1 has variance orders of magnitude larger than depth 15, early depths are allocated 8 bits, while fine residual noise depths are allocated 4–6 bits, saving footprint while preserving perceptual fidelity.
