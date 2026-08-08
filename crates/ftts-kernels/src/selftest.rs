//! Shipped integer-overflow proofs for the kernels this crate can dispatch.
//!
//! The canonical Q8 converter emits symmetric signed weights in `[-127, 127]`; `-128` is
//! deliberately excluded. Dynamic unsigned activations can span `[0, 255]`. Every dot-product
//! route therefore has to prove its i32 accumulator against `255 * 127 * K` at this checkpoint's
//! real reduction lengths, not a bound borrowed from another model.
//!
//! Two proof families run at every census binding K:
//!
//! - **U8S8 envelope** (`255 * 127 * K`): the contract ceiling for a future unsigned-activation
//!   route (x86 VNNI's +128 fold), executed on the checked scalar path. It strictly dominates the
//!   S8S8 magnitude, so it remains the conservative bound for every row.
//! - **S8S8 kernel** (`±127 * 127 * K`): executed through the *real* [`crate::int8::dot_i32`]
//!   kernel on every tier this build can dispatch ([`crate::int8::Int8Tier::available`]), each
//!   result compared against the independent i64 oracle and the scalar route's i32.
//!
//! A native tier must appear here, through its real kernel function, before it may be selected.
//! The rows mirror the binding component maxima in `docs/truth-pack/EXECUTION_CENSUS.json`, plus
//! the seq-16 microdecoder verifier, whose larger M does not alter its per-output reduction
//! length.

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
    /// The portable eight-lane route shaped for LLVM autovectorization.
    Autovec,
    /// The aarch64 SDOT island (`neon-dotprod` feature + runtime FEAT_DotProd).
    NeonSdot,
}

impl KernelTier {
    /// Stable machine-readable route name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Autovec => "autovec",
            Self::NeonSdot => "neon-sdot",
        }
    }

    const fn from_int8(tier: crate::int8::Int8Tier) -> Self {
        match tier {
            crate::int8::Int8Tier::Scalar => Self::Scalar,
            crate::int8::Int8Tier::Autovec => Self::Autovec,
            crate::int8::Int8Tier::NeonSdot => Self::NeonSdot,
        }
    }
}

/// Which numeric contract a proof check exercised.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DotContract {
    /// `255 * 127 * K` on the checked scalar path — the conservative envelope for a future
    /// unsigned-activation (VNNI +128 fold) route.
    U8S8Envelope,
    /// `±127 * 127 * K` executed through the real [`crate::int8::dot_i32`] kernel.
    S8S8Kernel,
}

impl DotContract {
    /// Stable machine-readable contract name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8S8Envelope => "u8s8-envelope",
            Self::S8S8Kernel => "s8s8-kernel",
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
    /// Numeric contract this check exercised.
    pub contract: DotContract,
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
    let dispatched = KernelTier::from_int8(crate::int8::Int8Tier::dispatch());
    let mut checks = Vec::new();
    for row in OVERFLOW_PROOF_ROWS.iter().copied() {
        // U8S8 envelope: the contract ceiling, on the checked scalar path.
        let accumulator_i32 = scalar_all_extreme_dot_i32(row.reduction_k);
        let reference_i64 = all_extreme_dot_i64(row.reduction_k);
        let accumulator_i32 = if fault_row == Some(row.id) {
            accumulator_i32.map(|accumulator| accumulator.saturating_sub(1))
        } else {
            accumulator_i32
        };
        checks.push(SelftestCheck {
            row,
            tier: KernelTier::Scalar,
            contract: DotContract::U8S8Envelope,
            accumulator_i32,
            reference_i64,
            passed: accumulator_i32
                .is_some_and(|accumulator| i64::from(accumulator) == reference_i64),
        });

        // S8S8 kernel proof: the real dot kernel, on every tier this build can dispatch, at both
        // all-extreme signs, against the independent i64 oracle and the scalar route's i32.
        let k = row.reduction_k as usize;
        let positive = vec![crate::int8::Q8_MAX_ABS; k];
        let negative = vec![-crate::int8::Q8_MAX_ABS; k];
        let s8_reference_i64 =
            i64::from(S8_MAX_ABS) * i64::from(S8_MAX_ABS) * i64::from(row.reduction_k);
        let scalar_positive =
            crate::int8::dot_i32(&positive, &positive, crate::int8::Int8Tier::Scalar);
        for tier in crate::int8::Int8Tier::available() {
            let up = crate::int8::dot_i32(&positive, &positive, tier);
            let down = crate::int8::dot_i32(&positive, &negative, tier);
            let up = if fault_row == Some(row.id) {
                up.saturating_sub(1)
            } else {
                up
            };
            let passed = i64::from(up) == s8_reference_i64
                && i64::from(down) == -s8_reference_i64
                && up == scalar_positive;
            checks.push(SelftestCheck {
                row,
                tier: KernelTier::from_int8(tier),
                contract: DotContract::S8S8Kernel,
                accumulator_i32: Some(up),
                reference_i64: s8_reference_i64,
                passed,
            });
        }
    }
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

    //  Crate-local pinned copy; byte-identity with the truth-pack canonical is asserted by a
    //  unit test whenever the repository checkout is present (crates.io builds have no repo).
    const CENSUS: &str = include_str!("../pinned/EXECUTION_CENSUS.json");

    #[test]
    fn every_deployed_row_equals_its_i64_reference_on_every_tier() {
        let report = run_selftest();
        assert!(report.passed(), "{report:#?}");

        let envelope: Vec<_> = report
            .checks
            .iter()
            .filter(|check| check.contract == DotContract::U8S8Envelope)
            .collect();
        assert_eq!(envelope.len(), OVERFLOW_PROOF_ROWS.len());
        for check in &envelope {
            assert_eq!(check.tier, KernelTier::Scalar, "{}", check.row.id);
            assert_eq!(
                check.accumulator_i32.map(i64::from),
                Some(check.reference_i64),
                "{}",
                check.row.id
            );
        }

        let tiers = crate::int8::Int8Tier::available();
        let s8s8: Vec<_> = report
            .checks
            .iter()
            .filter(|check| check.contract == DotContract::S8S8Kernel)
            .collect();
        assert_eq!(s8s8.len(), OVERFLOW_PROOF_ROWS.len() * tiers.len());
        for check in &s8s8 {
            assert_eq!(
                check.accumulator_i32.map(i64::from),
                Some(check.reference_i64),
                "{} on {}",
                check.row.id,
                check.tier.as_str()
            );
        }

        assert!(
            tiers
                .iter()
                .any(|tier| KernelTier::from_int8(*tier) == report.dispatched),
            "dispatched route {:?} is not among the available tiers",
            report.dispatched
        );
    }

    #[test]
    fn the_sdot_island_is_proven_on_this_silicon_when_present() {
        // On an Apple Silicon dev host the island must actually run — a silently absent
        // FEAT_DotProd would turn every SDOT proof row into vacuous truth.
        if cfg!(all(target_arch = "aarch64", feature = "neon-dotprod"))
            && crate::int8::neon_sdot_available()
        {
            let report = run_selftest();
            assert!(
                report.checks.iter().any(|check| {
                    check.tier == KernelTier::NeonSdot
                        && check.contract == DotContract::S8S8Kernel
                        && check.passed
                }),
                "FEAT_DotProd reported but no SDOT proof row executed"
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

    #[test]
    fn pinned_census_copy_matches_the_truth_pack_canonical() {
        //  The crate-local copy exists because `cargo package` cannot ship the truth pack; the
        //  truth pack stays canonical. A drifted copy silently pins the selftest to a stale
        //  census, so equality is asserted byte-for-byte whenever the repo checkout is present.
        let canonical = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/truth-pack/EXECUTION_CENSUS.json");
        match std::fs::read_to_string(&canonical) {
            Ok(bytes) => assert_eq!(
                bytes, CENSUS,
                "pinned/EXECUTION_CENSUS.json drifted from the truth-pack canonical; re-copy it"
            ),
            Err(_) => eprintln!(
                "SKIP pinned_census_copy_matches_the_truth_pack_canonical: no repo checkout at {}",
                canonical.display()
            ),
        }
    }
}
