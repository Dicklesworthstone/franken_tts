//! Shipped integer-overflow proofs for the kernels this crate can dispatch.
//!
//! The canonical Q8 converter emits symmetric signed weights in `[-127, 127]`; `-128` is
//! deliberately excluded. Dynamic unsigned activations can span `[0, 255]`. Every dot-product
//! route therefore has to prove its i32 accumulator against `255 * 127 * K` at this checkpoint's
//! real reduction lengths, not a bound borrowed from another model.
//!
//! The current executable route is only [`KernelTier::Scalar`]. It is intentionally the only tier
//! this module advertises or marks as passed: native ISA tiers must add their implementation and a
//! scalar-equality proof row before they become dispatchable. The rows mirror the binding component
//! maxima in `docs/truth-pack/EXECUTION_CENSUS.json`, plus the seq-16 microdecoder verifier, whose
//! larger M does not alter its per-output reduction length.

/// Largest unsigned activation byte accepted by the U8S8 contract.
pub const U8_MAX: u8 = u8::MAX;
/// Largest absolute signed Q8 weight byte under the canonical symmetric recipe.
pub const S8_MAX_ABS: i8 = 127;

/// A route that this build can actually execute and certify.
///
/// Do not add an ISA variant merely because the CPU can report that feature. A variant belongs
/// here only after the corresponding implementation exists and its exact scalar comparison is
/// wired into [`run_selftest`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelTier {
    /// The portable, safe integer reference route.
    Scalar,
}

impl KernelTier {
    /// Stable machine-readable route name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
        }
    }
}

/// The model component whose maximum reduction length a proof row represents.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionScope {
    /// Enrollment-only codec encoder.
    Enrollment,
    /// Per-frame decode path.
    Decode,
    /// Prompt-time text projection/embedding path.
    Prefill,
    /// The residual-code microdecoder's one-step execution.
    Microdecoder,
    /// The seq-16 residual-code verifier; it shares the microdecoder's per-output K.
    MicrodecoderVerify,
    /// Main Qwen talker.
    Talker,
}

impl ExecutionScope {
    /// Stable machine-readable scope name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Enrollment => "enrollment",
            Self::Decode => "decode",
            Self::Prefill => "prefill",
            Self::Microdecoder => "microdecoder",
            Self::MicrodecoderVerify => "microdecoder_verify_seq16",
            Self::Talker => "talker",
        }
    }
}

/// One permanent, model-specific i32-overflow obligation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OverflowProofRow {
    /// Stable row identifier consumed by selftest output and future release receipts.
    pub id: &'static str,
    /// Execution regime covered by this row.
    pub scope: ExecutionScope,
    /// Pinned census tensor that established this component maximum.
    pub census_tensor: &'static str,
    /// Actual reduction length for one output element.
    pub reduction_k: u32,
}

/// Component maxima generated from the pinned execution census.
///
/// The global 8192 row is enrollment-only; the 7168 codec-decoder row is the binding decode
/// maximum. Keeping both prevents the larger enrollment shape from accidentally shadowing the
/// actual real-time requirement.
pub const OVERFLOW_PROOF_ROWS: &[OverflowProofRow] = &[
    OverflowProofRow {
        id: "codec_encoder_global_k8192",
        scope: ExecutionScope::Enrollment,
        census_tensor: "encoder.encoder.layers.12.conv.weight",
        reduction_k: 8192,
    },
    OverflowProofRow {
        id: "codec_decoder_decode_k7168",
        scope: ExecutionScope::Decode,
        census_tensor: "decoder.decoder.0.conv.weight",
        reduction_k: 7168,
    },
    OverflowProofRow {
        id: "speaker_encoder_k4608",
        scope: ExecutionScope::Enrollment,
        census_tensor: "speaker_encoder.asp.tdnn.conv.weight",
        reduction_k: 4608,
    },
    OverflowProofRow {
        id: "microdecoder_step_k3072",
        scope: ExecutionScope::Microdecoder,
        census_tensor: "talker.code_predictor.model.layers.0.mlp.down_proj.weight",
        reduction_k: 3072,
    },
    OverflowProofRow {
        id: "microdecoder_verify_seq16_k3072",
        scope: ExecutionScope::MicrodecoderVerify,
        census_tensor: "talker.code_predictor.model.layers.0.mlp.down_proj.weight",
        reduction_k: 3072,
    },
    OverflowProofRow {
        id: "talker_down_proj_k3072",
        scope: ExecutionScope::Talker,
        census_tensor: "talker.model.layers.0.mlp.down_proj.weight",
        reduction_k: 3072,
    },
    OverflowProofRow {
        id: "text_projection_k2048",
        scope: ExecutionScope::Prefill,
        census_tensor: "talker.model.text_embedding.weight",
        reduction_k: 2048,
    },
];

/// One completed proof result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelftestCheck {
    /// Row selected from [`OVERFLOW_PROOF_ROWS`].
    pub row: OverflowProofRow,
    /// Route that executed the check.
    pub tier: KernelTier,
    /// Accumulation performed by that route using all-extreme operands.
    ///
    /// `None` means the route overflowed before producing an i32 result; that is a failed proof,
    /// not a panic or an implicitly widened success.
    pub accumulator_i32: Option<i32>,
    /// Independent widened reference for the same dot product.
    pub reference_i64: i64,
    /// Whether the route retained exact i32 equality and fit in range.
    pub passed: bool,
}

/// A complete selftest result for the dispatched routes in this build.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SelftestReport {
    /// Currently selected executable route.
    pub dispatched: KernelTier,
    /// Every binding row executed against that route.
    pub checks: Vec<SelftestCheck>,
}

impl SelftestReport {
    /// Whether every executed check has exact i32 equality with its widened reference.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.checks.iter().all(|check| check.passed)
    }
}

/// Runs the permanent overflow proof against every route this build can dispatch.
///
/// This is deliberately not a compile-time arithmetic assertion. The deployed route executes the
/// actual all-extreme dot loop and compares that i32 result with a separately widened i64 oracle,
/// so the same entrypoint can later compare an intrinsic backend before dispatch enables it.
#[must_use]
pub fn run_selftest() -> SelftestReport {
    run_selftest_inner(None)
}

fn run_selftest_inner(fault_row: Option<&str>) -> SelftestReport {
    let dispatched = KernelTier::Scalar;
    let checks = OVERFLOW_PROOF_ROWS
        .iter()
        .copied()
        .map(|row| {
            let accumulator_i32 = scalar_all_extreme_dot_i32(row.reduction_k);
            let reference_i64 = all_extreme_dot_i64(row.reduction_k);
            let accumulator_i32 = if fault_row == Some(row.id) {
                accumulator_i32.map(|accumulator| accumulator.saturating_sub(1))
            } else {
                accumulator_i32
            };
            SelftestCheck {
                row,
                tier: dispatched,
                accumulator_i32,
                reference_i64,
                passed: accumulator_i32
                    .is_some_and(|accumulator| i64::from(accumulator) == reference_i64),
            }
        })
        .collect();
    SelftestReport { dispatched, checks }
}

fn scalar_all_extreme_dot_i32(reduction_k: u32) -> Option<i32> {
    let term = i32::from(U8_MAX) * i32::from(S8_MAX_ABS);
    (0..reduction_k).try_fold(0_i32, |accumulator, _| accumulator.checked_add(term))
}

fn all_extreme_dot_i64(reduction_k: u32) -> i64 {
    i64::from(U8_MAX) * i64::from(S8_MAX_ABS) * i64::from(reduction_k)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CENSUS: &str = include_str!("../../../docs/truth-pack/EXECUTION_CENSUS.json");

    #[test]
    fn every_deployed_scalar_row_equals_its_i64_reference() {
        let report = run_selftest();
        assert_eq!(report.dispatched, KernelTier::Scalar);
        assert!(report.passed(), "{report:#?}");
        assert_eq!(report.checks.len(), OVERFLOW_PROOF_ROWS.len());
        for check in report.checks {
            assert_eq!(check.tier, KernelTier::Scalar, "{}", check.row.id);
            assert_eq!(
                check.accumulator_i32.map(i64::from),
                Some(check.reference_i64),
                "{}",
                check.row.id
            );
        }
    }

    #[test]
    fn census_binding_rows_are_not_replaced_by_a_stale_talker_only_bound() {
        for (tensor, reduction_k) in [
            ("encoder.encoder.layers.12.conv.weight", 8192),
            ("decoder.decoder.0.conv.weight", 7168),
            ("speaker_encoder.asp.tdnn.conv.weight", 4608),
            (
                "talker.code_predictor.model.layers.0.mlp.down_proj.weight",
                3072,
            ),
            ("talker.model.layers.0.mlp.down_proj.weight", 3072),
            ("talker.model.text_embedding.weight", 2048),
        ] {
            assert!(
                OVERFLOW_PROOF_ROWS
                    .iter()
                    .any(|row| { row.census_tensor == tensor && row.reduction_k == reduction_k }),
                "proof row missing for {tensor} K={reduction_k}"
            );
            assert!(
                CENSUS.contains(&format!("\"tensor\": \"{tensor}\"")),
                "pinned census no longer contains {tensor}; regenerate the proof table"
            );
            assert!(
                CENSUS.split('{').any(|object| {
                    object.contains(&format!("\"tensor\": \"{tensor}\""))
                        && object.contains(&format!("\"k\": {reduction_k}"))
                }),
                "pinned census no longer gives {tensor} reduction K={reduction_k}; regenerate the proof table"
            );
        }
        assert!(
            CENSUS.contains("\"decode_path_binding_row\""),
            "proof table requires a separately named decode binding"
        );
    }

    #[test]
    fn a_corrupted_route_fails_the_selftest_instead_of_reporting_green() {
        let report = run_selftest_inner(Some("codec_decoder_decode_k7168"));
        assert!(
            !report.passed(),
            "fault injection must fail the aggregate verdict"
        );
        assert!(
            report
                .checks
                .iter()
                .any(|check| { check.row.id == "codec_decoder_decode_k7168" && !check.passed })
        );
    }

    #[test]
    fn i32_bound_remains_strictly_below_the_widened_limit() {
        for row in OVERFLOW_PROOF_ROWS {
            let reference = all_extreme_dot_i64(row.reduction_k);
            assert!(
                reference < i64::from(i32::MAX),
                "{} no longer fits i32: {reference}",
                row.id
            );
        }
    }
}
