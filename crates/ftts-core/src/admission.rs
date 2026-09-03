//! Resource admission: decide before allocating, and report truncation instead of hiding it.
//!
//! Distinct from the engine's *concurrency* admission (the one-live-synthesis lease in
//! [`crate::TtsEngine`]). This module answers a different question: **will this request fit?**
//!
//! # Reject before partial allocation
//!
//! The failure this prevents is dying halfway through a long generation, after the caller has
//! waited a minute and after we have already committed hundreds of megabytes. Peak memory for an
//! utterance is *predictable* from the prompt length and the frame cap, so it is computed up front
//! and the request is refused whole or admitted whole. There is no middle state.
//!
//! # The rule (OQ-6, `docs/QWEN3_TTS_DECODE_AND_ADMISSION.md` §4–5)
//!
//! ```text
//! predicted_max_frames = min(max_new_tokens, MAX_CONTEXT_TOKENS - prompt_tokens)
//! predicted_peak_bytes = KV_talker(prompt_tokens, predicted_max_frames)
//!                      + MICRODECODER_KV_BYTES + CODEC_DECODER_KV_BYTES
//!                      + ring_buffer_bytes + weights_resident_bytes
//! admit iff predicted_peak_bytes <= budget_bytes
//!
//! KV_talker(L, N) = (L + N) * TALKER_KV_VALUES_PER_TOKEN * sizeof(dtype)
//! ```
//!
//! Only the talker KV grows with duration. The microdecoder KV is per-frame-reset, the codec
//! decoder KV is a fixed 72-frame window, and the conv rings are a function of receptive fields —
//! all bounded, which is why a long utterance is affordable at all.
//!
//! # Truncation is an outcome, not a silence
//!
//! When the frame cap is reached without an end-of-speech token, the reference implementation
//! returns the truncated audio with no exception, no warning, and no flag — the caller cannot tell
//! "the model finished" from "the model was cut off mid-word". `ftts` is agent-facing and an agent
//! cannot *hear* the difference, so [`StopReason::FrameCapReached`] is a distinct, reported
//! outcome. Per Doctrine #0.4, returning a cut-off utterance as a plain success is a counterfeit
//! green.
//!
//! Bead: `frankentts-v-reliability-d65`.

use core::fmt;

pub use crate::server_admission::{
    AdmissionTicket, CapacityCertificate, QueueingLatencySummary, ServerAdmissionController,
    ServerAdmissionRejection, ServerAdmissionRequest, ServerCapacityModel, ServerQueueingConfig,
    SocketTopology,
};

/// Talker KV values retained per token.
///
/// 28 layers × 2 (key and value) × 8 KV heads × 128 head_dim = 57,344. Grouped-query attention is
/// why this is 8 KV heads and not 16 — the KV cache is half what the query head count suggests.
pub const TALKER_KV_VALUES_PER_TOKEN: u64 = 57_344;

/// Microdecoder KV footprint: 5 layers × ≤16 positions, reset every frame. Does **not** grow.
pub const MICRODECODER_KV_BYTES: u64 = 320 * 1024;

/// Codec decoder KV footprint: 8 layers × window 72 × 16 × 64 × 2. Fixed window, does **not** grow.
pub const CODEC_DECODER_KV_BYTES: u64 = 2_359_296;

/// Hard context ceiling in tokens.
///
/// In practice `max_new_tokens` binds first: this ceiling only becomes the constraint above roughly
/// 24,500 prompt tokens, which is unreachable under the 8,192-frame cap.
pub const MAX_CONTEXT_TOKENS: u64 = 32_768;

/// Precision the KV cache is held at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum KvDtype {
    /// bfloat16 — 2 bytes per value.
    Bf16,
    /// f32 — 4 bytes per value.
    F32,
}

impl KvDtype {
    /// Bytes per stored value.
    #[must_use]
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    /// The stable wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }
}

impl fmt::Display for KvDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which limit determined the predicted frame count.
///
/// Reported because the two have different remedies: a caller hitting the frame cap should raise
/// it or chunk the text, while one hitting the context ceiling must shorten the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingConstraint {
    /// `max_new_tokens` bound the generation. The usual case.
    FrameCap,
    /// The context ceiling bound it — the prompt is long enough to crowd out generation.
    ContextCeiling,
    /// The text-derived EOS backstop bound it (`prompt_tokens * 4 + 64` frames).
    ///
    /// The sampled EOS is a stochastic stop (README: "EOS stop timing is sampling-dependent"),
    /// so a bare `ftts say` without `FTTS_MAX_FRAMES` needs a cap proportional to the text
    /// rather than the flat 8,192-frame (≈11 minute) policy default. Setting `FTTS_MAX_FRAMES`
    /// disables this backstop: an explicit cap is obeyed exactly.
    TextHeuristic,
}

impl BindingConstraint {
    /// The stable wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FrameCap => "frame_cap",
            Self::ContextCeiling => "context_ceiling",
            Self::TextHeuristic => "text_heuristic",
        }
    }
}

/// Frames granted per prompt token by the EOS backstop.
///
/// Calibrated on the demo utterance: 28 prompt tokens (with wrapper) produced 55 frames of real
/// speech, ≈2 frames/token; 4 leaves room for slow prosody and pauses without permitting a
/// runaway. An engineering backstop, not a physics claim.
pub const HEURISTIC_FRAMES_PER_PROMPT_TOKEN: u64 = 4;

/// Flat headroom the EOS backstop adds for leading/trailing silence.
pub const HEURISTIC_FRAME_HEADROOM: u64 = 64;

/// Everything admission needs to know before any allocation happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionRequest {
    /// Prompt length in tokens, after normalization and prompt assembly.
    pub prompt_tokens: u64,
    /// Caller's frame cap.
    pub max_new_tokens: u64,
    /// Whether the text-derived EOS backstop also bounds the generation.
    ///
    /// False when the caller set an explicit cap (`FTTS_MAX_FRAMES`), which is then obeyed
    /// exactly.
    pub heuristic_eos_backstop: bool,
    /// Precision the KV cache is held at.
    pub kv_dtype: KvDtype,
    /// Codec conv ring buffers, from receptive fields.
    pub ring_buffer_bytes: u64,
    /// Resident model weights.
    pub weights_resident_bytes: u64,
    /// The ceiling this request must fit under.
    pub budget_bytes: u64,
}

/// The computed prediction. Produced whether or not the request is admitted.
///
/// A rejection carries its plan too: a caller told only "no" cannot tell whether to shorten the
/// text, lower the cap, or raise the budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionPlan {
    /// Frames the request may generate.
    pub predicted_max_frames: u64,
    /// Which limit produced that number.
    pub binding_constraint: BindingConstraint,
    /// Talker KV bytes — the only term that grows with duration.
    pub kv_talker_bytes: u64,
    /// Microdecoder KV + codec decoder KV + conv rings. Bounded regardless of duration.
    pub bounded_state_bytes: u64,
    /// Resident weights.
    pub weights_resident_bytes: u64,
    /// The total that must fit.
    pub predicted_peak_bytes: u64,
    /// The ceiling it was compared against.
    pub budget_bytes: u64,
}

impl AdmissionPlan {
    /// Bytes by which the prediction exceeds the budget; zero when it fits.
    #[must_use]
    pub const fn shortfall_bytes(&self) -> u64 {
        self.predicted_peak_bytes.saturating_sub(self.budget_bytes)
    }

    /// Whether the prediction fits.
    #[must_use]
    pub const fn fits(&self) -> bool {
        self.predicted_peak_bytes <= self.budget_bytes
    }
}

/// Why a request could not be admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionRejection {
    /// The prompt alone meets or exceeds the context ceiling, leaving no room to generate.
    ///
    /// Separate from a budget shortfall because no amount of memory fixes it.
    PromptExceedsContext {
        /// Prompt length.
        prompt_tokens: u64,
        /// The ceiling.
        ceiling: u64,
    },
    /// The caller asked for zero frames; there is nothing to synthesize.
    NoFramesRequested,
    /// Predicted peak memory exceeds the budget.
    BudgetExceeded {
        /// The full prediction, so the caller can act on it.
        plan: AdmissionPlan,
    },
    /// The prediction overflowed `u64`.
    ///
    /// A wrapped total would be a small, plausible-looking number that admits a request certain to
    /// die mid-generation — the precise failure admission exists to prevent, so it is its own
    /// refusal rather than a saturating clamp.
    Overflow {
        /// Which term overflowed.
        term: &'static str,
    },
}

impl fmt::Display for AdmissionRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PromptExceedsContext {
                prompt_tokens,
                ceiling,
            } => write!(
                f,
                "prompt is {prompt_tokens} tokens but the context ceiling is {ceiling}; \
                 no frames could be generated. Shorten the text or chunk it — raising the memory \
                 budget cannot help"
            ),
            Self::NoFramesRequested => {
                f.write_str("max_new_tokens is 0; there is nothing to synthesize")
            }
            Self::BudgetExceeded { plan } => write!(
                f,
                "predicted peak {} bytes exceeds the {} byte budget by {} \
                 (talker KV {} over {} frames, bounded state {}, weights {}; binding constraint: {}). \
                 Rejected before allocating, so nothing was committed",
                plan.predicted_peak_bytes,
                plan.budget_bytes,
                plan.shortfall_bytes(),
                plan.kv_talker_bytes,
                plan.predicted_max_frames,
                plan.bounded_state_bytes,
                plan.weights_resident_bytes,
                plan.binding_constraint.as_str(),
            ),
            Self::Overflow { term } => write!(
                f,
                "admission arithmetic overflowed computing `{term}`; refusing rather than \
                 admitting on a wrapped total"
            ),
        }
    }
}

impl core::error::Error for AdmissionRejection {}

/// Talker KV bytes for a prompt of `prompt_tokens` generating `frames` frames.
///
/// # Errors
///
/// Returns [`AdmissionRejection::Overflow`] rather than wrapping.
pub fn talker_kv_bytes(
    prompt_tokens: u64,
    frames: u64,
    dtype: KvDtype,
) -> Result<u64, AdmissionRejection> {
    prompt_tokens
        .checked_add(frames)
        .and_then(|tokens| tokens.checked_mul(TALKER_KV_VALUES_PER_TOKEN))
        .and_then(|values| values.checked_mul(dtype.size_bytes()))
        .ok_or(AdmissionRejection::Overflow {
            term: "talker_kv_bytes",
        })
}

/// The engine-held half of an admission decision: everything known before the text arrives.
///
/// Split from [`AdmissionRequest`] because the two halves are known at different times. The budget,
/// frame cap, and resident footprint are properties of the *engine*; only `prompt_tokens` depends on
/// the request, and it is not known until after tokenization. Keeping them apart is what lets the
/// engine run admission at the one correct seam — after `prepare`, before any allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionPolicy {
    /// Ceiling on predicted peak memory for one utterance.
    pub budget_bytes: u64,
    /// Frame cap applied to every request.
    pub max_new_tokens: u64,
    /// Whether the text-derived EOS backstop also applies (disabled by an explicit
    /// `FTTS_MAX_FRAMES`).
    pub heuristic_eos_backstop: bool,
    /// Precision the KV cache is held at.
    pub kv_dtype: KvDtype,
    /// Codec conv ring buffers.
    pub ring_buffer_bytes: u64,
    /// Resident model weights.
    pub weights_resident_bytes: u64,
}

/// Default utterance memory budget: 2 GiB.
///
/// Chosen so the common 8,192-frame cap (952 MiB of talker KV at a 512-token prompt) fits with
/// room for weights, rather than as a round number. Override with `FTTS_MEMORY_BUDGET_MB`.
pub const DEFAULT_BUDGET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

/// Default frame cap: 8,192 frames ≈ 655 seconds at 12.5 frames/s.
pub const DEFAULT_MAX_NEW_TOKENS: u64 = 8_192;

impl Default for AdmissionPolicy {
    fn default() -> Self {
        Self {
            budget_bytes: DEFAULT_BUDGET_BYTES,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            heuristic_eos_backstop: true,
            kv_dtype: KvDtype::Bf16,
            // Phase 0 has no codec rings and no resident weights yet. Zero is the honest value:
            // an invented placeholder would make the prediction look complete while being wrong,
            // and these terms are bounded, so they are added when the components that own them land.
            ring_buffer_bytes: 0,
            weights_resident_bytes: 0,
        }
    }
}

impl AdmissionPolicy {
    /// Completes the policy into a decidable request, given the tokenized prompt length.
    #[must_use]
    pub const fn request_for(&self, prompt_tokens: u64) -> AdmissionRequest {
        AdmissionRequest {
            prompt_tokens,
            max_new_tokens: self.max_new_tokens,
            heuristic_eos_backstop: self.heuristic_eos_backstop,
            kv_dtype: self.kv_dtype,
            ring_buffer_bytes: self.ring_buffer_bytes,
            weights_resident_bytes: self.weights_resident_bytes,
            budget_bytes: self.budget_bytes,
        }
    }

    /// Runs admission for a tokenized prompt.
    ///
    /// # Errors
    ///
    /// Returns the [`AdmissionRejection`]; the caller must then allocate nothing.
    pub fn admit(&self, prompt_tokens: u64) -> Result<AdmissionPlan, AdmissionRejection> {
        admit(&self.request_for(prompt_tokens))
    }
}

/// Decides whether a request may proceed, computing the full prediction either way.
///
/// # Errors
///
/// Returns the specific [`AdmissionRejection`]; the request must then allocate nothing.
pub fn admit(request: &AdmissionRequest) -> Result<AdmissionPlan, AdmissionRejection> {
    if request.prompt_tokens >= MAX_CONTEXT_TOKENS {
        return Err(AdmissionRejection::PromptExceedsContext {
            prompt_tokens: request.prompt_tokens,
            ceiling: MAX_CONTEXT_TOKENS,
        });
    }
    if request.max_new_tokens == 0 {
        return Err(AdmissionRejection::NoFramesRequested);
    }

    let headroom = MAX_CONTEXT_TOKENS - request.prompt_tokens;
    let heuristic_cap = if request.heuristic_eos_backstop {
        request
            .prompt_tokens
            .saturating_mul(HEURISTIC_FRAMES_PER_PROMPT_TOKEN)
            .saturating_add(HEURISTIC_FRAME_HEADROOM)
    } else {
        u64::MAX
    };
    let predicted_max_frames = request.max_new_tokens.min(headroom).min(heuristic_cap);
    let binding_constraint = if predicted_max_frames == heuristic_cap
        && heuristic_cap < request.max_new_tokens.min(headroom)
    {
        BindingConstraint::TextHeuristic
    } else if request.max_new_tokens <= headroom {
        BindingConstraint::FrameCap
    } else {
        BindingConstraint::ContextCeiling
    };

    let kv_talker_bytes = talker_kv_bytes(
        request.prompt_tokens,
        predicted_max_frames,
        request.kv_dtype,
    )?;

    let bounded_state_bytes = MICRODECODER_KV_BYTES
        .checked_add(CODEC_DECODER_KV_BYTES)
        .and_then(|sum| sum.checked_add(request.ring_buffer_bytes))
        .ok_or(AdmissionRejection::Overflow {
            term: "bounded_state_bytes",
        })?;

    let predicted_peak_bytes = kv_talker_bytes
        .checked_add(bounded_state_bytes)
        .and_then(|sum| sum.checked_add(request.weights_resident_bytes))
        .ok_or(AdmissionRejection::Overflow {
            term: "predicted_peak_bytes",
        })?;

    let plan = AdmissionPlan {
        predicted_max_frames,
        binding_constraint,
        kv_talker_bytes,
        bounded_state_bytes,
        weights_resident_bytes: request.weights_resident_bytes,
        predicted_peak_bytes,
        budget_bytes: request.budget_bytes,
    };

    if plan.fits() {
        Ok(plan)
    } else {
        Err(AdmissionRejection::BudgetExceeded { plan })
    }
}

/// Why a generation stopped.
///
/// The reason travels with every result because two of these produce *audio that sounds finished*
/// and are not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The model emitted end-of-speech. The only clean completion.
    EndOfSpeech,
    /// The frame cap was reached without end-of-speech — **the audio is cut off mid-utterance**.
    FrameCapReached,
    /// A hard duration limit stopped generation — also a cut-off.
    DurationLimitReached,
    /// The caller cancelled.
    Cancelled,
}

impl StopReason {
    /// The stable wire string, for robot mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EndOfSpeech => "end_of_speech",
            Self::FrameCapReached => "frame_cap_reached",
            Self::DurationLimitReached => "duration_limit_reached",
            Self::Cancelled => "cancelled",
        }
    }

    /// Whether the audio was cut off rather than completed.
    ///
    /// The predicate the CLI branches its exit code on. An agent cannot hear a truncated word, so
    /// this must be inspectable rather than audible.
    #[must_use]
    pub const fn is_truncated(self) -> bool {
        matches!(self, Self::FrameCapReached | Self::DurationLimitReached)
    }

    /// Whether this outcome may be reported as an unqualified success.
    ///
    /// Only [`StopReason::EndOfSpeech`] may. Anything else is either truncated or cancelled, and
    /// reporting it as plain success is the counterfeit green Doctrine #0.4 forbids.
    #[must_use]
    pub const fn is_clean_completion(self) -> bool {
        matches!(self, Self::EndOfSpeech)
    }

    /// A caller-facing explanation of what to do about it.
    #[must_use]
    pub const fn remedy(self) -> Option<&'static str> {
        match self {
            Self::EndOfSpeech | Self::Cancelled => None,
            Self::FrameCapReached => Some(
                "the utterance hit the frame cap before the model finished speaking; raise \
                 --max-frames or split the text into shorter chunks",
            ),
            Self::DurationLimitReached => Some(
                "the utterance hit the hard duration limit; raise it or split the text into \
                 shorter chunks",
            ),
        }
    }
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * 1024 * 1024;

    fn request(prompt_tokens: u64, max_new_tokens: u64, budget_bytes: u64) -> AdmissionRequest {
        AdmissionRequest {
            prompt_tokens,
            max_new_tokens,
            // These tests pin the explicit-cap arithmetic; the backstop has its own tests below.
            heuristic_eos_backstop: false,
            kv_dtype: KvDtype::Bf16,
            ring_buffer_bytes: 0,
            weights_resident_bytes: 0,
            budget_bytes,
        }
    }

    #[test]
    fn the_eos_backstop_binds_a_short_prompt_under_the_flat_default_cap() {
        let mut with_backstop = request(28, DEFAULT_MAX_NEW_TOKENS, 2 * GIB);
        with_backstop.heuristic_eos_backstop = true;
        let plan = admit(&with_backstop).expect("fits easily");
        assert_eq!(
            plan.predicted_max_frames,
            28 * HEURISTIC_FRAMES_PER_PROMPT_TOKEN + HEURISTIC_FRAME_HEADROOM
        );
        assert_eq!(plan.binding_constraint, BindingConstraint::TextHeuristic);
    }

    #[test]
    fn an_explicit_cap_disables_the_eos_backstop_exactly() {
        // FTTS_MAX_FRAMES semantics: the explicit value is obeyed even when the heuristic would
        // have been smaller.
        let explicit = request(28, 2_000, 2 * GIB);
        let plan = admit(&explicit).expect("fits");
        assert_eq!(plan.predicted_max_frames, 2_000);
        assert_eq!(plan.binding_constraint, BindingConstraint::FrameCap);
    }

    #[test]
    fn the_backstop_never_raises_a_smaller_explicit_cap() {
        let mut small = request(1_000, 32, 2 * GIB);
        small.heuristic_eos_backstop = true;
        let plan = admit(&small).expect("fits");
        assert_eq!(plan.predicted_max_frames, 32);
        assert_eq!(plan.binding_constraint, BindingConstraint::FrameCap);
    }

    /// The three worked points recorded in OQ-6. These are the numbers the rule was derived from;
    /// if the formula drifts, these are what catch it.
    #[test]
    fn talker_kv_matches_the_oq6_worked_points_exactly() {
        assert_eq!(
            talker_kv_bytes(512, 2048, KvDtype::Bf16).expect("no overflow"),
            280 * MIB,
            "512-token prompt + 2048-frame cap must be exactly 280 MiB"
        );
        assert_eq!(
            talker_kv_bytes(512, 8192, KvDtype::Bf16).expect("no overflow"),
            952 * MIB,
            "512-token prompt + 8192-frame cap must be exactly 952 MiB"
        );
        // The full context, however it is split between prompt and generation.
        assert_eq!(
            talker_kv_bytes(0, MAX_CONTEXT_TOKENS, KvDtype::Bf16).expect("no overflow"),
            7 * GIB / 2,
            "the full 32768-token context must be exactly 3.50 GiB"
        );
        // 112 KiB per token at BF16, the figure the whole budget rests on.
        assert_eq!(
            talker_kv_bytes(1, 0, KvDtype::Bf16).expect("no overflow"),
            112 * 1024
        );
        // F32 is exactly double.
        assert_eq!(
            talker_kv_bytes(512, 2048, KvDtype::F32).expect("no overflow"),
            560 * MIB
        );
    }

    #[test]
    fn a_request_that_fits_is_admitted_with_its_full_prediction() {
        let plan = admit(&request(512, 2048, 2 * GIB)).expect("must be admitted");
        assert_eq!(plan.predicted_max_frames, 2048);
        assert_eq!(plan.binding_constraint, BindingConstraint::FrameCap);
        assert_eq!(plan.kv_talker_bytes, 280 * MIB);
        assert_eq!(
            plan.bounded_state_bytes,
            MICRODECODER_KV_BYTES + CODEC_DECODER_KV_BYTES
        );
        assert!(plan.fits());
        assert_eq!(plan.shortfall_bytes(), 0);
    }

    /// The core promise: over budget means refused, and refused means nothing was committed.
    #[test]
    fn an_over_budget_request_is_rejected_before_any_allocation_and_says_by_how_much() {
        let error = admit(&request(512, 8192, 512 * MIB)).expect_err("must be rejected");
        let AdmissionRejection::BudgetExceeded { plan } = error else {
            panic!("expected a budget rejection, got {error}");
        };
        assert!(!plan.fits());
        assert_eq!(plan.kv_talker_bytes, 952 * MIB);
        assert!(plan.shortfall_bytes() > 0);

        // A rejection a caller cannot act on is only half a refusal.
        let rendered = error.to_string();
        for expected in ["predicted peak", "budget", "exceeds", "before allocating"] {
            assert!(
                rendered.contains(expected),
                "rejection is not actionable, missing `{expected}`: {rendered}"
            );
        }
    }

    #[test]
    fn the_binding_constraint_is_reported_because_the_two_have_different_remedies() {
        // Frame cap binds: the ordinary case at any realistic prompt length.
        let plan = admit(&request(512, 8192, 8 * GIB)).expect("admitted");
        assert_eq!(plan.binding_constraint, BindingConstraint::FrameCap);
        assert_eq!(plan.predicted_max_frames, 8192);

        // Context ceiling binds only when the prompt crowds out generation.
        let prompt = MAX_CONTEXT_TOKENS - 100;
        let plan = admit(&request(prompt, 8192, 8 * GIB)).expect("admitted");
        assert_eq!(plan.binding_constraint, BindingConstraint::ContextCeiling);
        assert_eq!(plan.predicted_max_frames, 100);

        // OQ-6's claim that the ceiling is unreachable under an 8192 cap below ~24,500 prompt
        // tokens: at 24,000 the frame cap still binds.
        let plan = admit(&request(24_000, 8192, 8 * GIB)).expect("admitted");
        assert_eq!(plan.binding_constraint, BindingConstraint::FrameCap);
    }

    #[test]
    fn a_prompt_at_or_past_the_ceiling_is_refused_as_unfixable_by_memory() {
        for prompt in [MAX_CONTEXT_TOKENS, MAX_CONTEXT_TOKENS + 1, u64::MAX] {
            let error = admit(&request(prompt, 1024, u64::MAX)).expect_err("must be rejected");
            assert!(
                matches!(error, AdmissionRejection::PromptExceedsContext { .. }),
                "got {error}"
            );
            // Even with an unlimited budget: more memory cannot buy context.
            assert!(error.to_string().contains("cannot help"));
        }
    }

    #[test]
    fn zero_frames_is_refused_rather_than_admitted_as_a_no_op() {
        let error = admit(&request(512, 0, u64::MAX)).expect_err("must be rejected");
        assert_eq!(error, AdmissionRejection::NoFramesRequested);
    }

    /// A wrapped total would admit a request certain to die mid-generation.
    #[test]
    fn arithmetic_overflow_is_refused_never_wrapped_into_a_plausible_total() {
        assert!(matches!(
            talker_kv_bytes(u64::MAX, u64::MAX, KvDtype::F32),
            Err(AdmissionRejection::Overflow { .. })
        ));

        let over = AdmissionRequest {
            prompt_tokens: 512,
            max_new_tokens: 2048,
            heuristic_eos_backstop: false,
            kv_dtype: KvDtype::Bf16,
            ring_buffer_bytes: u64::MAX,
            weights_resident_bytes: u64::MAX,
            budget_bytes: u64::MAX,
        };
        let error = admit(&over).expect_err("overflow must not be admitted");
        assert!(
            matches!(error, AdmissionRejection::Overflow { .. }),
            "a wrapped total is exactly the failure admission exists to prevent, got {error}"
        );
    }

    #[test]
    fn admission_is_exactly_at_the_boundary_not_off_by_one() {
        let peak = admit(&request(512, 2048, u64::MAX))
            .expect("admitted")
            .predicted_peak_bytes;
        // Exactly the budget admits; one byte less does not.
        assert!(admit(&request(512, 2048, peak)).is_ok());
        assert!(admit(&request(512, 2048, peak - 1)).is_err());
    }

    #[test]
    fn only_end_of_speech_counts_as_a_clean_completion() {
        assert!(StopReason::EndOfSpeech.is_clean_completion());
        assert!(!StopReason::EndOfSpeech.is_truncated());

        // The two that produce audio which *sounds* finished but is not.
        for cut in [
            StopReason::FrameCapReached,
            StopReason::DurationLimitReached,
        ] {
            assert!(cut.is_truncated(), "{cut} must be reported as truncated");
            assert!(
                !cut.is_clean_completion(),
                "{cut} must never be reported as an unqualified success — an agent cannot hear \
                 that the audio stopped mid-word"
            );
            assert!(
                cut.remedy().is_some(),
                "{cut} must tell the caller what to do"
            );
        }

        // Cancellation is neither clean nor truncated-by-the-model: the caller already knows.
        assert!(!StopReason::Cancelled.is_clean_completion());
        assert!(!StopReason::Cancelled.is_truncated());
    }

    #[test]
    fn stop_reason_wire_strings_are_distinct_and_stable() {
        let all = [
            StopReason::EndOfSpeech,
            StopReason::FrameCapReached,
            StopReason::DurationLimitReached,
            StopReason::Cancelled,
        ];
        let mut seen: Vec<&str> = all.iter().map(|reason| reason.as_str()).collect();
        let count = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), count, "two stop reasons share a wire string");
        assert_eq!(StopReason::FrameCapReached.as_str(), "frame_cap_reached");
    }
}
