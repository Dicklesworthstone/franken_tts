# Drafter Ladder (#2–#5): Distillation, 15 Parallel Heads & Tree Verification (Phase 5)

> **Artifact Type**: Systems Architecture & Empirical Research Report (Phase 5)  
> **Governing Bead**: `frankentts-p5-drafter-ladder-0wk`  
> **Status**: Implemented, Verified & Integrated (`crates/ftts-model-qwen/src/drafter_ladder.rs`)

---

## 1. Executive Summary & Objective

In Qwen3-TTS, speech generation is governed by a 28-layer talker followed by a **15-step Residual-Code Microdecoder**.
Because each residual code conditions the subsequent depth, the microdecoder is reread 15 sequential times per frame, creating the project's primary serial CPU bottleneck.

The **Drafter Ladder** evaluates an escalating series of speculative proposal architectures designed to propose residual tokens cheaply, verified in a single teacher-forced block pass by the seq-16 FrankenMTP verifier.

---

## 2. The Drafter Ladder Rungs

| Rung | Engine Architecture | Complexity | Theoretical Speedup Ceiling | Status & Disposition |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | **Transition Sketch** (`FrankenMtpDrafter`) | 64 buckets/depth, Markov copy | Up to $15\times$ (greedy) | Measured ~0.01 acceptance under sampling (NE-004); greedy-only. |
| **#2** | **1-Layer Distilled Student** (`DistilledMicroDrafter`) | 1-layer transformer (80% fewer FLOPs) | $5\times$ over microdecoder | **Implemented & Gated** via `.fttsdraft` (`DrafterType::DistilledMtp`). |
| **#3** | **15 Parallel Heads** (`ParallelHeadsDrafter`) | 15 linear projections off `talker_hidden` | $15\times$ (one-shot proposal, $O(1)$) | **Implemented & Gated** via `.fttsdraft` (`DrafterType::ParallelHeads`). |
| **#5** | **Tree Verification** (`TreeVerifyController`) | Cascaded $3 \to 5 \to 7$ block checks | Amortizes failed verifications | **Implemented & Verified** in scheduler. |

---

## 3. Mathematical Decision Frame: $T(\alpha)$ & Break-Even $\alpha^*(\text{SKU})$

For a speculative proposal engine to deliver wall-clock speedup, its mean per-depth token acceptance probability $\bar{\alpha}$ must exceed the machine's break-even threshold $\alpha^*(\text{SKU})$:

$$T(\alpha) = T_{\text{draft}} + T_{\text{verify}} + (1 - \alpha) \cdot (15 - L_{\text{accepted}}) \cdot T_{\text{seq}}$$

- **Drafter #3 (Parallel Heads)** incurs virtually zero sequential cost ($T_{\text{draft}} \approx 0.05 T_{\text{seq}}$ because it computes 15 projections in a single fused GEMV).
- **Tree Verification** mitigates the penalty of early mispredictions: by checking depths 1..=3 first, verification aborts immediately if early formants are rejected, preventing wasted teacher passes on depths 4..=15.

---

## 4. Integration with `.fttsdraft` ABI

All drafter implementations implement [`SpeculativeDrafter`](file:///Users/jemanuel/projects/frankentts/crates/ftts-model-qwen/src/drafter_ladder.rs#L29) and cleanly export to the `.fttsdraft` container format:
- Cryptographic binding to base `.fttsq` hash.
- Strict ABI version matching (`CURRENT_ENGINE_ABI_VERSION = 1`).
- Dynamic kill-switch support (`is_kill_switched = true`).
