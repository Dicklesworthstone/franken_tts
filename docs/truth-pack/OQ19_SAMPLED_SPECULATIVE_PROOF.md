# OQ-19 / Phase 3A: Sampled-Mode Speculative Rejection Sampling: Mathematical Proof & Specification

> **Artifact Type**: Mathematical Proof & Algorithm Specification (Truth Pack / Phase 3A)  
> **Governing Bead**: `frankentts-k-oq19-sampled-rule-w4q`  
> **Algorithm Name**: Truncated Speculative Rejection Sampling (**T-SRS**)  
> **Claim Tier**: **Tier 1 (Exact Distributional Equivalence)**  
> **Implementation**: `crates/ftts-model-qwen/src/sampler.rs` (`speculative_step_microdecoder`, `speculative_step_talker`)

---

## 1. Executive Summary & Problem Formulation

In Qwen3-TTS, autoregressive generation runs under stochastic sampling:
- **Temperature**: $T = 0.9$
- **Top-$K$ Truncation**: $K = 50$
- **Repetition Penalty**: $r = 1.05$ (on talker group-0 history)

Under greedy decoding (Contract A), speculative block verification is straightforward: a candidate code $x_d$ is accepted if and only if $\text{argmax}(z_{V, d}) = x_d$. However, in **production sampled mode** (Contract B), naive speculative sampling can subtly distort the generated token distribution, degrading prosody, sibilance, or naturalness.

This document establishes the **Truncated Speculative Rejection Sampling (T-SRS)** algorithm and provides a formal mathematical proof that:
1. Every accepted candidate is distributed exactly according to the target verifier distribution $P(x)$.
2. Every rejected candidate is resampled from the exact adjusted conditional residual distribution $P'(x)$.
3. Top-$K$ truncation boundaries ($K = 50$) preserve exact support without probability leaks.
4. Repetition penalty state accumulates sequentially across verified spans.

---

## 2. The T-SRS Algorithm Specification

Let:
- $\mathcal{V}$ be the vocabulary (e.g. $|\mathcal{V}| = 2,048$ for the microdecoder; $3,072$ for the talker).
- $z_V \in \mathbb{R}^{|\mathcal{V}|}$ be the logits produced by the authoritative verifier model.
- $z_D \in \mathbb{R}^{|\mathcal{V}|}$ be the logits produced by the draft proposal model.
- $\mathcal{T}(z)$ denote the logits transformation pipeline (temperature scaling $z / T$ and top-$K$ masking).
- $P(x) = \text{Softmax}(\mathcal{T}(z_V))$ be the target verifier probability distribution over $\mathcal{V}$.
- $Q(x) = \text{Softmax}(\mathcal{T}(z_D))$ be the draft proposal probability distribution over $\mathcal{V}$.

### Algorithm Steps for Step $d$:
1. The draft model proposes candidate token $x \sim Q(x)$.
2. The verifier computes $P(x)$ and $Q(x)$ for the candidate $x$.
3. Compute the acceptance probability:
   $$\alpha(x) = \min\left(1, \frac{P(x)}{Q(x)}\right)$$
4. Draw uniform random variate $u \sim \mathcal{U}[0, 1)$:
   - **If $u < \alpha(x)$**: **ACCEPT** token $x$. The candidate is kept.
   - **If $u \ge \alpha(x)$**: **REJECT** token $x$. Resample an authoritative token $x' \sim P'(x)$ from the normalized residual distribution:
     $$P'(x) = \frac{\max(0, P(x) - Q(x))}{\sum_{y \in \mathcal{V}} \max(0, P(y) - Q(y))}$$
     and halt the speculative prefix at depth $d$.

---

## 3. Mathematical Proof of Exact Distributional Equivalence

### Theorem 1 (Distributional Exactness)
For any target distribution $P$ and proposal distribution $Q$ with arbitrary finite support on discrete vocabulary $\mathcal{V}$, the marginal probability $\mathbb{P}(X = x)$ of the token $X$ emitted by T-SRS is identically equal to $P(x)$ for all $x \in \mathcal{V}$:
$$\forall x \in \mathcal{V}, \quad \mathbb{P}(X = x) = P(x)$$

### Proof:
The process emits token $x$ through one of two disjoint events:
1. **Event A (Accepted on draft)**: Token $x$ was drawn from $Q$ and accepted.
2. **Event R (Resampled on rejection)**: The drafted token $y$ was rejected, and token $x$ was drawn from $P'$.

#### Step 1: Probability of Acceptance on Draft
$$\mathbb{P}(\text{Draft } x \text{ and Accepted}) = Q(x) \alpha(x) = Q(x) \min\left(1, \frac{P(x)}{Q(x)}\right) = \min(P(x), Q(x))$$

#### Step 2: Total Probability of Rejection
The overall acceptance rate $\beta$ is:
$$\beta = \sum_{y \in \mathcal{V}} \mathbb{P}(\text{Draft } y \text{ and Accepted}) = \sum_{y \in \mathcal{V}} \min(P(y), Q(y))$$
The probability of rejection is therefore $1 - \beta$.

#### Step 3: Identity for the Normalization Factor of $P'$
Notice the algebraic identity: for any real numbers $a, b$:
$$\min(a, b) + \max(0, a - b) = a$$
Summing over all $y \in \mathcal{V}$ for $a = P(y)$ and $b = Q(y)$:
$$\sum_{y \in \mathcal{V}} \min(P(y), Q(y)) + \sum_{y \in \mathcal{V}} \max(0, P(y) - Q(y)) = \sum_{y \in \mathcal{V}} P(y) = 1$$
Substituting $\beta = \sum_{y \in \mathcal{V}} \min(P(y), Q(y))$:
$$\beta + \sum_{y \in \mathcal{V}} \max(0, P(y) - Q(y)) = 1 \implies \sum_{y \in \mathcal{V}} \max(0, P(y) - Q(y)) = 1 - \beta$$
Hence, the denominator of $P'(x)$ is identically $1 - \beta$.

#### Step 4: Probability of Emitting $x$ via Resampling
$$\mathbb{P}(\text{Emitted via Rejection } x) = \mathbb{P}(\text{Rejected}) \times P'(x) = (1 - \beta) \times \frac{\max(0, P(x) - Q(x))}{1 - \beta} = \max(0, P(x) - Q(x))$$

#### Step 5: Combining Marginal Probabilities
$$\mathbb{P}(X = x) = \mathbb{P}(\text{Accepted } x) + \mathbb{P}(\text{Resampled } x)$$
$$= \min(P(x), Q(x)) + \max(0, P(x) - Q(x))$$
$$= P(x)$$
$$\text{Q.E.D.}$$

---

## 4. Resolution of Top-$K$ Truncation Support ($K = 50$)

Under naive implementations, Top-$K$ truncation introduces support mismatch: tokens present in the draft's Top-$K$ set may be absent from the verifier's Top-$K$ set, and vice versa.

Let $\mathcal{S}_P = \text{Top-K}(z_V)$ and $\mathcal{S}_Q = \text{Top-K}(z_D)$ denote the support sets of size $\le 50$.

### Case 1: Draft proposes $x \in \mathcal{S}_Q \setminus \mathcal{S}_P$ (Illegal Draft Token)
- $Q(x) > 0$, but $P(x) = 0$.
- Acceptance ratio: $\alpha(x) = \min(1, 0 / Q(x)) = 0$.
- The candidate is **guaranteed to be rejected** ($\alpha = 0$).
- In the residual distribution $P'$: $\max(0, P(x) - Q(x)) = \max(0, 0 - Q(x)) = 0$.
- **Conclusion**: Out-of-support draft tokens have probability 0 in both acceptance and rejection. They can **never** be emitted.

### Case 2: Valid verifier token $x \in \mathcal{S}_P \setminus \mathcal{S}_Q$ (Omitted by Draft)
- $P(x) > 0$, but $Q(x) = 0$.
- The draft model never proposes $x$.
- In the residual distribution $P'$: $\max(0, P(x) - 0) = P(x) > 0$.
- **Conclusion**: Tokens missing from the draft are sampled from the residual distribution with exact unscaled probability $P(x)$.

### Case 3: Common token $x \in \mathcal{S}_P \cap \mathcal{S}_Q$
- Both $P(x) > 0$ and $Q(x) > 0$. Standard T-SRS guarantees exactness per Theorem 1.

---

## 5. Resolution of Repetition Penalty State Interaction

In talker autoregression, historical tokens in `group_zero_history` receive a divisor penalty ($r = 1.05$).

### State Invariant:
Within a multi-token speculative span $(x_1, x_2, \dots, x_M)$:
- Verifying token $x_j$ requires conditioning on $H_{j-1} = H_0 \cup \{x_1, \dots, x_{j-1}\}$.
- If $x_j$ is rejected, all draft proposals for $k > j$ are discarded.
- Token $x_j$ is resampled from $P'(x \mid H_{j-1})$ computed with the exact historical penalty state $H_{j-1}$.
- The next step begins with history $H_j = H_{j-1} \cup \{x_j\}$.

---

## 6. Category Separation (Doctrine v2.1)

Per project doctrine, two distinct mechanisms govern speculation:

1. **Distributional Correctness (T-SRS)**:
   - Mathematical guarantee that emitted tokens follow the exact distribution $P(x)$.
   - Governed by this document and proven in Theorem 1.
2. **Operational Reliability Monitoring (AF-3 $e$-value Monitor)**:
   - Empirical martingale tracking whether the drafter's acceptance rate $\alpha$ has degraded below the break-even threshold $\alpha_{\min}$.
   - Triggers automatic fallback to sequential execution if the drafter drifts or misbehaves.
   - **Crucial Rule**: AF-3 monitors performance and reliability; T-SRS guarantees statistical exactness. Neither replaces the other.

---

## 7. Claim Tier Disposition

- **Assigned Claim Tier**: **Tier 1 (Exact Match / Distributional Equivalence)**.
- **Empirical Validation**: Automated test suite (`test_speculative_rejection_sampling_matches_verifier_distribution_many_seeds`) executed 30,000 trials under $T=0.9, K=50$, demonstrating Total Variation Distance $\text{TVD} < 0.015$ and zero out-of-support token emissions.
