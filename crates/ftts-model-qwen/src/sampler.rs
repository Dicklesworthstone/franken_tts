//! Qwen3-TTS's two distinct autoregressive sampling contracts.
//!
//! The talker and residual-code microdecoder intentionally have separate entry points.  The
//! former owns the Hugging Face logits-processor stack, whereas the latter has no processors at
//! all.  See `docs/truth-pack/OQ12_SAMPLER_CONTRACT.md` sections 1, 2, and 4, derived from the
//! pinned transformers 4.57.3 source and Qwen snapshot.

use std::fmt;

/// The talker's full output-head width.
pub const TALKER_VOCAB_SIZE: usize = 3_072;
/// The residual-code microdecoder's output-head width.
pub const MICRODECODER_VOCAB_SIZE: usize = 2_048;
/// The only talker stop token.
pub const CODEC_EOS_TOKEN_ID: u32 = 2_150;
/// Talker tokens generated before EOS may be selected.
pub const TALKER_MIN_NEW_TOKENS: usize = 2;
/// Effective default from the pinned `generation_config.json`, not either code default.
pub const TALKER_MAX_NEW_TOKENS: usize = 8_192;
/// The residual decoder emits exactly this many non-primary codes per frame.
pub const MICRODECODER_MAX_NEW_TOKENS: usize = 15;

const REPETITION_PENALTY: f32 = 1.05;
const TEMPERATURE: f32 = 0.9;
pub const TOP_K: usize = 50;

/// Selects either the Contract-A canonical decoder or the production distribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplingMode {
    /// Apply non-warper processors, then take the maximum without consuming RNG state.
    CanonicalGreedy,
    /// Apply the complete production processor/warper stack and draw categorically.
    Production,
}

/// Outcome of a Truncated Speculative Rejection Sampling (T-SRS) step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeculativeDecision {
    /// Draft candidate accepted; sample matches verifier distribution.
    Accepted(u32),
    /// Draft candidate rejected; resampled from the adjusted conditional residual distribution.
    Rejected {
        resampled_token: u32,
    },
}

impl SpeculativeDecision {
    /// Authoritative token selected by the verifier (either accepted candidate or resampled token).
    #[must_use]
    pub const fn token(self) -> u32 {
        match self {
            Self::Accepted(tok) | Self::Rejected { resampled_token: tok } => tok,
        }
    }

    /// Whether the proposed candidate was accepted.
    #[must_use]
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted(_))
    }
}

/// A sampler failure caused by an invalid model boundary or logits tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplerError {
    UnexpectedLogitCount {
        level: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidTalkerHistoryToken(u32),
    NonFiniteLogit {
        index: usize,
    },
    NoCandidate,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedLogitCount {
                level,
                expected,
                actual,
            } => write!(
                formatter,
                "{level} sampler expected {expected} logits, received {actual}"
            ),
            Self::InvalidTalkerHistoryToken(token) => {
                write!(
                    formatter,
                    "talker group-0 history contains invalid token {token}"
                )
            }
            Self::NonFiniteLogit { index } => write!(formatter, "logit {index} is not finite"),
            Self::NoCandidate => formatter.write_str("all sampler candidates were masked"),
        }
    }
}

impl std::error::Error for SamplerError {}

/// Stateful, deterministic production sampler.
///
/// Its own PRNG deliberately models the categorical distribution rather than PyTorch's RNG
/// stream.  Determinism is consequently scoped to this engine build, ISA route, sampler version,
/// seed, and model artifact.
#[derive(Clone, Debug)]
pub struct QwenSampler {
    rng: SplitMix64,
}

impl QwenSampler {
    /// Creates a sampler whose production draws are reproducible for `seed`.
    pub const fn seeded(seed: u64) -> Self {
        Self {
            rng: SplitMix64::seeded(seed),
        }
    }

    /// Selects a primary talker code.
    ///
    /// `group_zero_history` contains *only* codes previously emitted by the talker.  Residual
    /// codes are intentionally absent: applying the repetition penalty to a whole 16-code frame
    /// would diverge from the reference processor input.
    pub fn select_talker(
        &mut self,
        logits: &[f32],
        group_zero_history: &[u32],
        mode: SamplingMode,
    ) -> Result<u32, SamplerError> {
        ensure_logits("talker", logits, TALKER_VOCAB_SIZE)?;
        let mut scores = logits.to_vec();
        apply_talker_processors(&mut scores, group_zero_history)?;

        match mode {
            SamplingMode::CanonicalGreedy => Ok(argmax(&scores)? as u32),
            SamplingMode::Production => {
                apply_sampling_warpers(&mut scores);
                self.sample(&scores)
            }
        }
    }

    /// Selects one residual-code microdecoder token.
    ///
    /// Unlike [`Self::select_talker`], this has neither history nor EOS parameters.  That shape
    /// is intentional: transformers applies no penalty, min-length, suppression, or EOS logic to
    /// the fixed 15-step microdecoder generation call.
    pub fn select_microdecoder(
        &mut self,
        logits: &[f32],
        mode: SamplingMode,
    ) -> Result<u32, SamplerError> {
        ensure_logits("microdecoder", logits, MICRODECODER_VOCAB_SIZE)?;
        let mut scores = logits.to_vec();

        match mode {
            SamplingMode::CanonicalGreedy => Ok(argmax(&scores)? as u32),
            SamplingMode::Production => {
                apply_sampling_warpers(&mut scores);
                self.sample(&scores)
            }
        }
    }

    fn sample(&mut self, scores: &[f32]) -> Result<u32, SamplerError> {
        let probabilities = softmax_finite(scores)?;
        let draw = self.rng.next_unit_interval();
        let mut cumulative = 0.0_f64;
        let mut last = None;

        for (token, probability) in probabilities {
            cumulative += probability;
            last = Some(token);
            if draw < cumulative {
                return Ok(token as u32);
            }
        }

        // Rounding can leave a sub-ULP interval at the top of [0, 1).  It still belongs to the
        // final nonzero-probability candidate.
        last.map(|token| token as u32)
            .ok_or(SamplerError::NoCandidate)
    }

    /// Executes Truncated Speculative Rejection Sampling (T-SRS) for one microdecoder step.
    ///
    /// Proves that the marginal distribution of the emitted token (whether accepted or resampled)
    /// equals the verifier's target categorical distribution with T=0.9 and top_k=50.
    pub fn speculative_step_microdecoder(
        &mut self,
        verifier_logits: &[f32],
        draft_logits: &[f32],
        draft_token: u32,
    ) -> Result<SpeculativeDecision, SamplerError> {
        ensure_logits("microdecoder verifier", verifier_logits, MICRODECODER_VOCAB_SIZE)?;
        ensure_logits("microdecoder draft", draft_logits, MICRODECODER_VOCAB_SIZE)?;

        let mut v_scores = verifier_logits.to_vec();
        apply_sampling_warpers(&mut v_scores);
        let p_dist = softmax_finite(&v_scores)?;

        let mut d_scores = draft_logits.to_vec();
        apply_sampling_warpers(&mut d_scores);
        let q_dist = softmax_finite(&d_scores)?;

        self.speculative_sample_core(&p_dist, &q_dist, draft_token)
    }

    /// Executes Truncated Speculative Rejection Sampling (T-SRS) for one talker step.
    ///
    /// Evaluates repetition penalty and top_k=50 truncation before computing acceptance.
    pub fn speculative_step_talker(
        &mut self,
        verifier_logits: &[f32],
        draft_logits: &[f32],
        group_zero_history: &[u32],
        draft_token: u32,
    ) -> Result<SpeculativeDecision, SamplerError> {
        ensure_logits("talker verifier", verifier_logits, TALKER_VOCAB_SIZE)?;
        ensure_logits("talker draft", draft_logits, TALKER_VOCAB_SIZE)?;

        let mut v_scores = verifier_logits.to_vec();
        apply_talker_processors(&mut v_scores, group_zero_history)?;
        apply_sampling_warpers(&mut v_scores);
        let p_dist = softmax_finite(&v_scores)?;

        let mut d_scores = draft_logits.to_vec();
        apply_talker_processors(&mut d_scores, group_zero_history)?;
        apply_sampling_warpers(&mut d_scores);
        let q_dist = softmax_finite(&d_scores)?;

        self.speculative_sample_core(&p_dist, &q_dist, draft_token)
    }

    /// Core Truncated Speculative Rejection Sampling algorithm (T-SRS).
    fn speculative_sample_core(
        &mut self,
        p_dist: &[(usize, f64)],
        q_dist: &[(usize, f64)],
        draft_token: u32,
    ) -> Result<SpeculativeDecision, SamplerError> {
        let draft_idx = draft_token as usize;
        let p_draft = p_dist
            .iter()
            .find(|(tok, _)| *tok == draft_idx)
            .map_or(0.0, |(_, prob)| *prob);
        let q_draft = q_dist
            .iter()
            .find(|(tok, _)| *tok == draft_idx)
            .map_or(0.0, |(_, prob)| *prob);

        let alpha = if q_draft > 0.0 {
            (p_draft / q_draft).min(1.0)
        } else {
            0.0
        };

        let u = self.rng.next_unit_interval();
        if u < alpha {
            return Ok(SpeculativeDecision::Accepted(draft_token));
        }

        // Rejection branch: resample from normalized residual distribution max(0, P(x) - Q(x))
        let max_vocab = p_dist
            .iter()
            .map(|(t, _)| *t)
            .chain(q_dist.iter().map(|(t, _)| *t))
            .max()
            .unwrap_or(0)
            + 1;
        let mut q_map = vec![0.0_f64; max_vocab];
        for &(tok, prob) in q_dist {
            if tok < q_map.len() {
                q_map[tok] = prob;
            }
        }

        let mut residual = Vec::with_capacity(p_dist.len());
        let mut residual_sum = 0.0_f64;
        for &(tok, p_prob) in p_dist {
            let q_prob = if tok < q_map.len() { q_map[tok] } else { 0.0 };
            let diff = (p_prob - q_prob).max(0.0);
            if diff > 0.0 {
                residual.push((tok, diff));
                residual_sum += diff;
            }
        }

        if residual_sum <= 0.0 || residual.is_empty() {
            let resampled = self.sample_from_dist(p_dist)?;
            return Ok(SpeculativeDecision::Rejected {
                resampled_token: resampled,
            });
        }

        // Sample from normalized residual
        let draw = self.rng.next_unit_interval();
        let mut cumulative = 0.0_f64;
        let mut last = None;
        for (tok, diff) in residual {
            cumulative += diff / residual_sum;
            last = Some(tok);
            if draw < cumulative {
                return Ok(SpeculativeDecision::Rejected {
                    resampled_token: tok as u32,
                });
            }
        }

        let final_tok = last.ok_or(SamplerError::NoCandidate)? as u32;
        Ok(SpeculativeDecision::Rejected {
            resampled_token: final_tok,
        })
    }

    fn sample_from_dist(&mut self, dist: &[(usize, f64)]) -> Result<u32, SamplerError> {
        let draw = self.rng.next_unit_interval();
        let mut cumulative = 0.0_f64;
        let mut last = None;
        for &(tok, prob) in dist {
            cumulative += prob;
            last = Some(tok);
            if draw < cumulative {
                return Ok(tok as u32);
            }
        }
        last.map(|t| t as u32).ok_or(SamplerError::NoCandidate)
    }
}

fn ensure_logits(level: &'static str, logits: &[f32], expected: usize) -> Result<(), SamplerError> {
    if logits.len() != expected {
        return Err(SamplerError::UnexpectedLogitCount {
            level,
            expected,
            actual: logits.len(),
        });
    }

    for (index, score) in logits.iter().enumerate() {
        if score.is_nan() || *score == f32::INFINITY {
            return Err(SamplerError::NonFiniteLogit { index });
        }
    }
    Ok(())
}

/// Mirrors the always-active portion of transformers 4.57.3's processor stack:
/// repetition penalty, then min-new-tokens, then the Qwen suppress mask.
fn apply_talker_processors(
    scores: &mut [f32],
    group_zero_history: &[u32],
) -> Result<(), SamplerError> {
    // Once per *distinct* token, never once per occurrence.
    //
    // The reference processor is gather → transform → scatter over `input_ids`. Duplicate ids all
    // gather the same original score and scatter writes that one value, so a code appearing four
    // times is penalized exactly as much as a code appearing once. Applying the penalty per
    // occurrence instead compounds it (1.05^n) and progressively crushes any repeated code until
    // EOS out-scores it — the talker then ends the utterance early, mid-speech, on nothing but
    // repetition. Observed against the ft7 oracle: code 1657 held for four frames fell 28.425 →
    // 23.385 under compounding and lost to EOS at 24.146, stopping us at frame 20 where the
    // reference ran on to 255.
    let mut penalized = vec![false; scores.len()];
    for &token in group_zero_history {
        let index =
            usize::try_from(token).map_err(|_| SamplerError::InvalidTalkerHistoryToken(token))?;
        let Some(already) = penalized.get_mut(index) else {
            return Err(SamplerError::InvalidTalkerHistoryToken(token));
        };
        if std::mem::replace(already, true) {
            continue;
        }
        let Some(score) = scores.get_mut(index) else {
            return Err(SamplerError::InvalidTalkerHistoryToken(token));
        };
        *score = if *score > 0.0 {
            *score / REPETITION_PENALTY
        } else {
            *score * REPETITION_PENALTY
        };
    }

    if group_zero_history.len() < TALKER_MIN_NEW_TOKENS {
        scores[CODEC_EOS_TOKEN_ID as usize] = f32::NEG_INFINITY;
    }

    for (token, score) in scores.iter_mut().enumerate().skip(MICRODECODER_VOCAB_SIZE) {
        if token != CODEC_EOS_TOKEN_ID as usize {
            *score = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

/// Applies the sampling-only transformers warpers in their pinned order.
/// The production sampling probability of one token: exactly the mass [`QwenSampler::sample`]
/// would assign it after the T/top-k warpers. Diagnostic support for the speculative-sampling
/// probe (bead w4q); consumes no RNG.
pub(crate) fn production_probability(logits: &[f32], token: usize) -> f64 {
    let mut scores = logits.to_vec();
    apply_sampling_warpers(&mut scores);
    match softmax_finite(&scores) {
        Ok(probabilities) => probabilities
            .iter()
            .find(|(candidate, _)| *candidate == token)
            .map_or(0.0, |(_, probability)| *probability),
        Err(_) => 0.0,
    }
}

fn apply_sampling_warpers(scores: &mut [f32]) {
    for score in scores.iter_mut().filter(|score| score.is_finite()) {
        *score /= TEMPERATURE;
    }
    apply_top_k(scores, TOP_K);
    apply_top_p(scores, 1.0);
}

/// Implements `TopKLogitsWarper` without sorting the full vocabulary.
fn apply_top_k(scores: &mut [f32], top_k: usize) {
    let finite_count = scores.iter().filter(|score| score.is_finite()).count();
    if finite_count <= top_k {
        return;
    }

    let mut candidates: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(token, score)| score.is_finite().then_some((token, *score)))
        .collect();
    let (_, cutoff, _) = candidates.select_nth_unstable_by(top_k - 1, |left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let cutoff = cutoff.1;
    for score in scores {
        if *score < cutoff {
            *score = f32::NEG_INFINITY;
        }
    }
}

/// Implements `TopPLogitsWarper` after top-k.  With Qwen's pinned p=1.0 this is a no-op, but
/// keeping the operation here preserves the executable production contract.
fn apply_top_p(scores: &mut [f32], top_p: f64) {
    if top_p >= 1.0 {
        return;
    }

    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .filter_map(|(token, score)| score.is_finite().then_some((token, *score)))
        .collect();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let Some(max_score) = ranked
        .iter()
        .map(|(_, score)| *score)
        .max_by(f32::total_cmp)
    else {
        return;
    };
    let denominator: f64 = ranked
        .iter()
        .map(|(_, score)| {
            let shifted = f64::from(*score - max_score);
            if ftts_kernels::f32ref::canonical_norm_requested() {
                ftts_kernels::f32ref::canonical_exp_wide(shifted)
            } else {
                shifted.exp()
            }
        })
        .sum();

    let mut cumulative = 0.0;
    for (rank, (token, score)) in ranked.into_iter().enumerate() {
        let shifted = f64::from(score - max_score);
        let numerator = if ftts_kernels::f32ref::canonical_norm_requested() {
            ftts_kernels::f32ref::canonical_exp_wide(shifted)
        } else {
            shifted.exp()
        };
        let probability = numerator / denominator;
        if rank > 0 && cumulative >= top_p {
            scores[token] = f32::NEG_INFINITY;
        }
        cumulative += probability;
    }
}

fn argmax(scores: &[f32]) -> Result<usize, SamplerError> {
    scores
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(index, _)| index)
        .ok_or(SamplerError::NoCandidate)
}

fn softmax_finite(scores: &[f32]) -> Result<Vec<(usize, f64)>, SamplerError> {
    let max_score = scores
        .iter()
        .copied()
        .filter(|score| score.is_finite())
        .max_by(f32::total_cmp)
        .ok_or(SamplerError::NoCandidate)?;
    let mut unnormalized = Vec::new();
    let mut denominator = 0.0_f64;
    for (token, score) in scores.iter().enumerate() {
        if score.is_finite() {
            let shifted = f64::from(*score - max_score);
            let value = if ftts_kernels::f32ref::canonical_norm_requested() {
                ftts_kernels::f32ref::canonical_exp_wide(shifted)
            } else {
                shifted.exp()
            };
            denominator += value;
            unnormalized.push((token, value));
        }
    }
    if !denominator.is_finite() || denominator == 0.0 {
        return Err(SamplerError::NoCandidate);
    }
    Ok(unnormalized
        .into_iter()
        .map(|(token, value)| (token, value / denominator))
        .collect())
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_unit_interval(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn talker_logits() -> Vec<f32> {
        vec![0.0; TALKER_VOCAB_SIZE]
    }

    #[test]
    fn production_draws_concentrate_on_a_peaked_mode() {
        // The captured talker logits are sharply peaked (winning code ~26.5 vs a ~5.5 background
        // — engine_e2e's recorded step-1 row). After T = 0.9 that peak holds ~all probability
        // mass, so a correct categorical draw selects it essentially always. A sampler whose
        // draws scatter over the top-k set on THIS distribution has a warper/normalization/draw
        // bug — this is the unit-level discriminator for silence-shaped production failures.
        let mut logits = vec![5.5_f32; TALKER_VOCAB_SIZE];
        logits[125] = 26.5;
        let mut sampler = QwenSampler::seeded(0);
        let mut peak_hits = 0usize;
        for _ in 0..200 {
            let pick = sampler
                .select_talker(&logits, &[], SamplingMode::Production)
                .expect("valid row");
            assert!(
                pick < MICRODECODER_VOCAB_SIZE as u32 || pick == CODEC_EOS_TOKEN_ID,
                "suppressed special {pick} must never be drawn"
            );
            if pick == 125 {
                peak_hits += 1;
            }
        }
        assert!(
            peak_hits >= 195,
            "a 21-logit peak after T=0.9 holds ~all mass; drawing it only {peak_hits}/200 times \
             means the production warper/draw stack is broken"
        );

        // And on a FLAT top-k the draws must scatter rather than collapse onto one token —
        // the opposite failure (always returning the first candidate) also produces
        // degenerate audio.
        let flat = vec![5.5_f32; TALKER_VOCAB_SIZE];
        let mut distinct = std::collections::BTreeSet::new();
        for _ in 0..200 {
            distinct.insert(
                sampler
                    .select_talker(&flat, &[], SamplingMode::Production)
                    .expect("valid row"),
            );
        }
        assert!(
            distinct.len() >= 20,
            "200 draws over a flat top-{TOP_K} produced only {} distinct tokens; the draw is \
             not exploring the candidate set",
            distinct.len()
        );
    }

    fn microdecoder_logits() -> Vec<f32> {
        vec![0.0; MICRODECODER_VOCAB_SIZE]
    }

    #[test]
    fn contract_a_l3_talker_suppress_mask_keeps_only_codes_and_eos() {
        let mut logits = talker_logits();
        logits[2_148] = 100.0;
        logits[CODEC_EOS_TOKEN_ID as usize] = 99.0;
        logits[2_047] = 98.0;

        let token = QwenSampler::seeded(1)
            .select_talker(&logits, &[10, 11], SamplingMode::CanonicalGreedy)
            .expect("valid talker logits");

        assert_eq!(
            token, CODEC_EOS_TOKEN_ID,
            "EOS alone survives the tail suppress mask"
        );
    }

    #[test]
    fn contract_a_l3_talker_repetition_penalty_is_sign_dependent_for_negative_scores() {
        let mut logits = vec![f32::NEG_INFINITY; TALKER_VOCAB_SIZE];
        logits[7] = -1.0;
        logits[8] = -1.04;

        let token = QwenSampler::seeded(1)
            .select_talker(&logits, &[7, 12], SamplingMode::CanonicalGreedy)
            .expect("valid talker logits");

        assert_eq!(token, 8, "a repeated negative score is multiplied by 1.05");
    }

    /// A repeated code is penalized once, not once per repeat.
    ///
    /// The reference gathers/scatters over `input_ids`, so duplicates collapse. Compounding the
    /// penalty instead makes any held code decay by 1.05^n; the practical symptom is the talker
    /// choosing EOS over a legitimately sustained code and truncating speech.
    #[test]
    fn contract_a_l3_talker_repetition_penalty_applies_once_per_distinct_token() {
        let mut logits = vec![f32::NEG_INFINITY; TALKER_VOCAB_SIZE];
        // Held four times. Penalized once, 10.0 becomes 9.52 and stays ahead of 9.0; compounded by
        // 1.05^4 it becomes 8.23 and loses. The rival sits between the two outcomes on purpose.
        logits[7] = 10.0;
        logits[8] = 9.0;

        let token = QwenSampler::seeded(1)
            .select_talker(&logits, &[7, 7, 7, 7], SamplingMode::CanonicalGreedy)
            .expect("valid talker logits");

        assert_eq!(
            token, 7,
            "four repeats must penalize 10.0 exactly once (9.52), keeping it above the 9.0 rival; \
             compounding would give 8.23 and hand the frame to the wrong code"
        );
    }

    /// The same compounding bug, at the boundary that actually bit: EOS winning by attrition.
    #[test]
    fn contract_a_l3_a_sustained_code_does_not_decay_into_a_spurious_eos() {
        let mut logits = vec![f32::NEG_INFINITY; TALKER_VOCAB_SIZE];
        // The ft7 numbers: a code held four frames, against an EOS that never moves.
        logits[1657] = 28.425;
        logits[CODEC_EOS_TOKEN_ID as usize] = 24.146;

        let token = QwenSampler::seeded(1)
            .select_talker(
                &logits,
                &[1221, 1014, 1657, 1657, 1657, 1657],
                SamplingMode::CanonicalGreedy,
            )
            .expect("valid talker logits");

        assert_eq!(
            token, 1657,
            "a sustained code must not be penalized below EOS by its own repetition; \
             compounding drops it to 23.385 and ends the utterance mid-speech"
        );
    }

    #[test]
    fn contract_a_l3_talker_min_new_tokens_masks_eos_before_two_group_zero_codes() {
        let mut logits = talker_logits();
        logits[CODEC_EOS_TOKEN_ID as usize] = 100.0;
        logits[7] = 99.0;

        let token = QwenSampler::seeded(1)
            .select_talker(&logits, &[6], SamplingMode::CanonicalGreedy)
            .expect("valid talker logits");

        assert_eq!(
            token, 7,
            "EOS must remain unavailable until two codes were generated"
        );
    }

    #[test]
    fn contract_a_l3_talker_penalty_accepts_group_zero_history_only() {
        let mut logits = talker_logits();
        logits[9] = 5.05;
        logits[10] = 5.2;

        let token = QwenSampler::seeded(1)
            .select_talker(&logits, &[10, 11], SamplingMode::CanonicalGreedy)
            .expect("valid talker logits");

        assert_eq!(
            token, 9,
            "only the supplied group-0 code is penalized; residual history has no API path"
        );
    }

    #[test]
    fn contract_a_l3_effective_talker_max_new_tokens_is_generation_config_value() {
        assert_eq!(TALKER_MAX_NEW_TOKENS, 8_192);
    }

    #[test]
    fn contract_a_l3_microdecoder_raw_argmax_has_no_talker_processors() {
        let mut logits = microdecoder_logits();
        logits[2_047] = 10.0;

        let token = QwenSampler::seeded(1)
            .select_microdecoder(&logits, SamplingMode::CanonicalGreedy)
            .expect("valid microdecoder logits");

        assert_eq!(
            token, 2_047,
            "microdecoder support is its full raw 2048-token head"
        );
    }

    #[test]
    fn production_sampling_is_deterministic_for_our_seeded_rng() {
        let mut logits = microdecoder_logits();
        logits[2] = 1.0;
        logits[3] = 0.5;
        let mut left = QwenSampler::seeded(42);
        let mut right = QwenSampler::seeded(42);

        let left_tokens: Vec<_> = (0..32)
            .map(|_| left.select_microdecoder(&logits, SamplingMode::Production))
            .collect::<Result<_, _>>()
            .expect("valid logits");
        let right_tokens: Vec<_> = (0..32)
            .map(|_| right.select_microdecoder(&logits, SamplingMode::Production))
            .collect::<Result<_, _>>()
            .expect("valid logits");

        assert_eq!(left_tokens, right_tokens);
    }

    #[test]
    fn test_speculative_rejection_sampling_matches_verifier_distribution_many_seeds() {
        let mut v_logits = microdecoder_logits();
        let mut d_logits = microdecoder_logits();

        // Create realistic non-trivial probability distribution
        for i in 0..100 {
            v_logits[i] = 10.0 - (i as f32 * 0.15);
            d_logits[i] = 9.8 - (i as f32 * 0.14) + ((i % 5) as f32 * 0.1);
        }

        // Compute true target distribution P(x)
        let mut v_scores = v_logits.clone();
        apply_sampling_warpers(&mut v_scores);
        let p_dist = softmax_finite(&v_scores).expect("valid p_dist");

        let mut p_true_map = vec![0.0_f64; MICRODECODER_VOCAB_SIZE];
        for &(tok, prob) in &p_dist {
            p_true_map[tok] = prob;
        }

        let num_trials = 30_000;
        let mut draft_sampler = QwenSampler::seeded(12345);
        let mut spec_sampler = QwenSampler::seeded(67890);

        let mut counts = vec![0usize; MICRODECODER_VOCAB_SIZE];
        let mut accepted_count = 0usize;

        for _ in 0..num_trials {
            // 1. Propose draft candidate from Q(x)
            let draft_cand = draft_sampler
                .select_microdecoder(&d_logits, SamplingMode::Production)
                .expect("valid draft draw");

            // 2. Step speculative rejection sampler
            let decision = spec_sampler
                .speculative_step_microdecoder(&v_logits, &d_logits, draft_cand)
                .expect("valid spec step");

            if decision.is_accepted() {
                accepted_count += 1;
            }
            counts[decision.token() as usize] += 1;
        }

        // Statistical assertion 1: Accept rate should be non-trivial (draft and verifier are close)
        let accept_rate = accepted_count as f64 / num_trials as f64;
        assert!(
            accept_rate > 0.60 && accept_rate < 0.99,
            "acceptance rate {accept_rate:.2} should be realistic"
        );

        // Statistical assertion 2: No tokens outside verifier Top-K are ever drawn
        for tok in 0..MICRODECODER_VOCAB_SIZE {
            if p_true_map[tok] == 0.0 {
                assert_eq!(
                    counts[tok], 0,
                    "token {tok} has 0 probability in verifier Top-K but was drawn {} times!",
                    counts[tok]
                );
            }
        }

        // Statistical assertion 3: Total Variation Distance (TVD) between empirical and true distribution
        let mut tvd = 0.0_f64;
        for tok in 0..MICRODECODER_VOCAB_SIZE {
            let p_emp = counts[tok] as f64 / num_trials as f64;
            tvd += (p_emp - p_true_map[tok]).abs();
        }
        tvd *= 0.5;

        assert!(
            tvd < 0.02,
            "Total variation distance {tvd:.4} must be < 0.02 for 30k trials (distributional equivalence)"
        );
    }

    #[test]
    fn test_speculative_rejection_sampling_with_top_k_truncation() {
        let mut v_logits = microdecoder_logits();
        let mut d_logits = microdecoder_logits();

        // Token 999 has high probability in draft but is completely absent from verifier Top-K
        v_logits[10] = 20.0;
        v_logits[20] = 19.0;
        d_logits[999] = 25.0; // dominant in draft

        let mut spec_sampler = QwenSampler::seeded(42);

        for _ in 0..100 {
            let decision = spec_sampler
                .speculative_step_microdecoder(&v_logits, &d_logits, 999)
                .expect("valid spec step");

            // Candidate 999 must ALWAYS be rejected because P(999) == 0
            assert!(!decision.is_accepted(), "candidate 999 must be rejected");
            assert!(
                decision.token() == 10 || decision.token() == 20,
                "resampled token must come from verifier's Top-K support"
            );
        }
    }

    #[test]
    fn test_talker_repetition_penalty_interaction_under_speculation() {
        let mut v_logits = talker_logits();
        let mut d_logits = talker_logits();

        v_logits[50] = 15.0;
        v_logits[51] = 15.0;
        d_logits[50] = 15.0;
        d_logits[51] = 15.0;

        let mut spec_sampler = QwenSampler::seeded(101);

        // Without history, 50 and 51 have equal probability
        let dec_no_history = spec_sampler
            .speculative_step_talker(&v_logits, &d_logits, &[], 50)
            .expect("valid step");
        assert!(dec_no_history.is_accepted());

        // With token 50 in history, repetition penalty reduces token 50's logit
        let dec_with_history = spec_sampler
            .speculative_step_talker(&v_logits, &d_logits, &[50], 50)
            .expect("valid step with history");
        assert!(dec_with_history.token() < TALKER_VOCAB_SIZE as u32);
    }
}
