# Alien-Artifact Transparency Card: AF-3 FrankenMTP E-Process Reliability Monitor

> **Artifact ID**: `AF-3`  
> **Component**: `crates/ftts-model-qwen/src/af3.rs`  
> **Consumer**: The FrankenMTP speculative decode loop (`FTTS_SPEC_MTP` kill-switch)  
> **Deletion Condition**: FrankenMTP abandoned or removed  
> **Fallback**: Authoritative sequential microdecoder (exactness Tier 1)  
> **Calibration Fixture**: [`docs/truth-pack/AF3_CALIBRATION.json`](file:///Users/jemanuel/projects/frankentts/docs/truth-pack/AF3_CALIBRATION.json)

---

## 1. Purpose and Operational Context

In speculative decoding for the 15-step microdecoder (FrankenMTP), drafting attempts to predict future residual codes before full verification. While strict-greedy verification guarantees token-for-token identity by construction (any rejected suffix is repaired sequentially), draft-path misbehavior (e.g., zero prefix acceptance, distribution drift, or corrupted transition sketches) can cause net computational regression:

$$T_{\text{spec}} = T_{\text{draft}} + T_{\text{verify}} + T_{\text{repair}} > T_{\text{seq}}$$

AF-3 implements an **anytime-valid sequential test (e-process)** that monitors the speculative path in real-time. It provides a mathematically rigorous upper bound on false alarms while rapidly detecting misbehavior. If the e-process exceeds its decision boundary ($1/\alpha$), the monitor triggers an automatic, irrevocable demotion to the authoritative sequential microdecoder and reports a structured health event (`HealthViolation::SpeculationDemoted`).

---

## 2. Mathematical Foundation: Ville's Inequality

An e-process $(E_t)_{t \ge 0}$ is a non-negative supermartingale with $E_0 = 1$ under the null hypothesis $H_0$. By Ville's maximal inequality:

$$\mathbb{P}_{H_0}\left(\exists t \ge 1: E_t \ge \frac{1}{\alpha}\right) \le \alpha \cdot \mathbb{E}[E_0] = \alpha$$

### Anytime Validity
Unlike fixed-horizon hypothesis testing (which suffers from $p$-hacking or multiple-testing penalties under continuous monitoring), the e-process bound holds at **any arbitrary stopping time** $\tau$:
- Across individual token predictions
- Across 80 ms audio frames
- Across long streaming utterances or continuous batching sessions

No Bonferroni correction or horizon truncation is needed.

---

## 3. Explicit State Space, Action Space, and Loss Matrix

### State Space $\mathcal{S}$
$$\mathcal{S} = (E_t, t, N_{\text{anomaly}}, \text{alarmed}) \in [0, \infty) \times \mathbb{N} \times \mathbb{N} \times \{\text{false}, \text{true}\}$$
- $E_t$: current accumulated e-value ($E_0 = 1.0$)
- $t$: total speculative verification steps observed
- $N_{\text{anomaly}}$: total anomaly events observed
- $\text{alarmed}$: boolean latched state

### Action Space $\mathcal{A}$
1. `Healthy`: $E_t < 1/\alpha$. Speculative drafting continues.
2. `Alarm`: $E_t \ge 1/\alpha$. Latches `alarmed = true`; demotes to sequential execution immediately; emits `HealthViolation::SpeculationDemoted`.
3. `Demoted`: Subsequent steps run exclusively on the authoritative sequential microdecoder.
4. `Disabled`: Speculation disabled by configuration ($\alpha \le 0$ or `FTTS_SPEC_MTP=0`).

### Loss Matrix $L(a, \theta)$

| True State $\theta$ | Action: `Healthy` (Speculate) | Action: `Alarm`/`Demote` (Sequential) |
| :--- | :--- | :--- |
| **$H_0$ (Drafter Well-Behaved)** | $0$ (Optimal execution) | $L_{\text{type I}}$ (Foregone speculative speedup, at most $\alpha$) |
| **$H_1$ (Drafter Broken/Misbehaving)** | $L_{\text{type II}}$ (Wasteful $T_{\text{draft}} + T_{\text{verify}}$) | $0$ (Optimal fallback to sequential) |

Because $L_{\text{type II}}$ scales linearly with every uncontained frame whereas $L_{\text{type I}}$ simply matches standard sequential decode time, the loss function strongly favors conservative, rapid demotion.

---

## 4. Parameter Choices and Calibration

| Parameter | Symbol | Default Value | Rationale |
| :--- | :--- | :--- | :--- |
| **Nominal Anomaly Bound** | $p_0$ | `0.10` | Tolerates up to 10% isolated prefix misses under nominal drafting. |
| **Betting Fraction** | $\lambda$ | `2.0` | Sized to ensure $1 + \lambda(0 - p_0) = 0.8 > 0$ and $1 + \lambda(1 - p_0) = 2.8$. |
| **Significance Level** | $\alpha$ | `0.01` | Guarantees $\le 1\%$ false alarm rate across infinite operational steps. |
| **Alarm Threshold** | $1/\alpha$ | `100.0` | $E_t \ge 100.0$ trips demotion. |

### Trajectory Under Complete Failure ($Y_t = 1$)
When an anomalous drafter produces zero useful proposals:
- Step 1: $E_1 = 1.0 \times 2.8 = 2.8$
- Step 2: $E_2 = 2.8 \times 2.8 = 7.84$
- Step 3: $E_3 = 7.84 \times 2.8 \approx 21.95$
- Step 4: $E_4 \approx 21.95 \times 2.8 \approx 61.47$
- Step 5: $E_5 \approx 61.47 \times 2.8 \approx 172.10 \ge 100.0$ $\rightarrow$ **Alarm trips at step 5**.

The broken drafter is contained within exactly 5 frames ($\le 400$ ms of audio), preventing runaway latency regression.

---

## 5. Deterministic Fallback Trigger

In accordance with Doctrine 0 and the Alien-Artifact Engineering Contract:
1. **$\alpha \le 0$ Fallback**: Setting $\alpha \le 0.0$ (via API or `FTTS_AF3_ALPHA=0`) causes `is_demoted()` to evaluate to `true` prior to any model execution. Speculation is bypassed entirely and the engine runs sequential decode with zero draft overhead.
2. **Environment Kill-Switch**: `FTTS_SPEC_MTP=0` or unsetting the variable keeps the speculative branch structurally inactive.
3. **Equivalence Invariant**:
   $$\text{AF-3}(\alpha \to 0) \equiv \text{Sequential Decode (bit-for-bit)}$$
   This is verified continuously in test `af3_deterministic_fallback_when_alpha_zero_matches_sequential_exactly`.

---

## 6. Assumptions Ledger

1. **Exchangeability under $H_0$**: Under nominal operation, the occurrence of random draft misses is conditionally bounded by $p_0$. If model prompt shifts violate $p_0$, the monitor correctly demotes.
2. **Authoritative Repair**: The sequential microdecoder repair step is assumed mathematically exact and side-effect-free (proven by `speculative_greedy_mode_produces_identical_code_streams_to_sequential`).
3. **Fail-Closed Demotion**: Once tripped, `alarmed` remains latched for the entire utterance to prevent flapping or thrashing between modes.
