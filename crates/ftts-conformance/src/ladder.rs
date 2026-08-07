//! The reusable, skip-honest Contract-A conformance ladder.
//!
//! The floor is an input to this runner, never a suggestion. In particular, a zero or absent
//! `max_abs` envelope means exact comparison; there is deliberately no default epsilon. Each
//! emitted rung receipt carries the oracle tier and SHA-256 digests of both artifacts needed to
//! re-derive the decision.

use std::{error::Error, fmt, fs, path::Path};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    compare::{Comparison, ExactComparison, compare_exact, compare_f32},
    report::{FixtureProvenance, OracleTier, Outcome, Receipt},
};

/// The committed CPU FP32 fallback nondeterminism floor.
pub const CPU_FP32_FLOOR_PATH: &str = "docs/truth-pack/nondeterminism-floor.json";

/// The pinned CPU FP32 fallback fixture manifest supplied with the Phase-0 oracle capture.
pub const CPU_FP32_FIXTURE_MANIFEST_PATH: &str =
    "/Users/jemanuel/.cache/frankentts/oracle-fixtures/ft7-cpu-fp32-r1/fixture_manifest.json";

/// Stable labels for Contract-A ladder rungs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LadderRung {
    /// Prompt and preprocessing IDs.
    L0PromptTokenIds,
    /// Per-operator seam proof.
    L1OperatorSeams,
    /// Layer and component activations.
    L2LayerAndComponentActivations,
    /// Logits and canonical argmax boundaries.
    L3Logits,
    /// Canonical greedy codec-token stream.
    L4GreedyCodecTokens,
    /// Codec waveform output.
    L5CodecWaveform,
}

impl LadderRung {
    /// The stable Contract-A receipt label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::L0PromptTokenIds => "L0",
            Self::L1OperatorSeams => "L1",
            Self::L2LayerAndComponentActivations => "L2",
            Self::L3Logits => "L3",
            Self::L4GreedyCodecTokens => "L4",
            Self::L5CodecWaveform => "L5",
        }
    }

    const fn floor_key(self) -> &'static str {
        match self {
            Self::L0PromptTokenIds => "L0_prompt_token_ids",
            Self::L1OperatorSeams => "L1_operator_seams",
            Self::L2LayerAndComponentActivations => "L2_layer_and_component_activations",
            Self::L3Logits => "L3_logits",
            Self::L4GreedyCodecTokens => "L4_greedy_codec_tokens",
            Self::L5CodecWaveform => "L5_codec_waveform",
        }
    }
}

/// The numeric policy obtained from the committed oracle floor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ComparisonPolicy {
    /// Every element must compare exactly; no epsilon is permitted.
    Exact,
    /// A measured maximum absolute envelope permits this difference.
    MaximumAbsolute(f64),
}

impl ComparisonPolicy {
    /// The elementwise absolute tolerance handed to the comparator.
    #[must_use]
    pub const fn tolerance(self) -> f64 {
        match self {
            Self::Exact => 0.0,
            Self::MaximumAbsolute(tolerance) => tolerance,
        }
    }
}

/// Errors that make a ladder decision non-re-derivable.
#[derive(Debug)]
pub enum LadderError {
    /// A floor or manifest artifact could not be read.
    ReadArtifact {
        /// Which artifact was being read.
        artifact: String,
        /// The operating-system error.
        source: std::io::Error,
    },
    /// An artifact was not valid JSON.
    ParseArtifact {
        /// Which artifact was parsed.
        artifact: String,
        /// The JSON error.
        source: serde_json::Error,
    },
    /// A required floor field is absent or malformed.
    InvalidFloor {
        /// An actionable explanation of the invalid field.
        reason: String,
    },
    /// The fixture manifest does not describe the requested oracle tier.
    TierManifestMismatch {
        /// The requested receipt tier.
        requested: OracleTier,
        /// The manifest's declared oracle class.
        manifest: String,
    },
}

impl fmt::Display for LadderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadArtifact { artifact, source } => {
                write!(
                    formatter,
                    "failed to read ladder artifact `{artifact}`: {source}"
                )
            }
            Self::ParseArtifact { artifact, source } => {
                write!(
                    formatter,
                    "failed to parse ladder artifact `{artifact}`: {source}"
                )
            }
            Self::InvalidFloor { reason } => {
                write!(formatter, "invalid nondeterminism floor: {reason}")
            }
            Self::TierManifestMismatch {
                requested,
                manifest,
            } => write!(
                formatter,
                "fixture manifest declares oracle tier `{manifest}`, not requested `{}`",
                requested.as_str()
            ),
        }
    }
}

impl Error for LadderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ReadArtifact { source, .. } => Some(source),
            Self::ParseArtifact { source, .. } => Some(source),
            Self::InvalidFloor { .. } | Self::TierManifestMismatch { .. } => None,
        }
    }
}

/// One emitted rung decision plus the receipt that made it auditable.
#[derive(Clone, Debug)]
pub struct RungResult {
    /// The executed (or explicitly skipped) rung.
    pub rung: LadderRung,
    /// The outcome written to the receipt stream.
    pub outcome: Outcome,
    /// The policy applied when this was a numeric comparison.
    pub comparison_policy: Option<ComparisonPolicy>,
    /// The numeric comparison when this was a numeric comparison.
    pub comparison: Option<Comparison>,
    /// The exact sequence comparison when this was a token or ID rung.
    pub exact_comparison: Option<ExactComparison>,
    /// The complete machine-readable receipt.
    pub receipt: Value,
}

/// The one-command scorecard result that callers write into an evidence directory.
#[derive(Clone, Debug)]
pub struct LadderScorecard {
    results: Vec<RungResult>,
}

impl LadderScorecard {
    /// Whether every recorded rung passed. Empty scorecards are not green.
    #[must_use]
    pub fn all_green(&self) -> bool {
        !self.results.is_empty()
            && self
                .results
                .iter()
                .all(|result| result.outcome == Outcome::Passed)
    }

    /// Rung decisions in execution order.
    #[must_use]
    pub fn results(&self) -> &[RungResult] {
        &self.results
    }

    /// Stable JSON suitable for an evidence bundle's scorecard entry.
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "contract": "ConformanceExact",
            "all_green": self.all_green(),
            "rungs": self.results.iter().map(|result| &result.receipt).collect::<Vec<_>>(),
        })
    }
}

/// A hash-bound floor and fixture-manifest pair used to run Contract A.
#[derive(Debug)]
pub struct LadderRunner {
    tier: OracleTier,
    floor_source: String,
    floor_sha256: String,
    floor: Value,
    fixture_manifest_source: String,
    fixture_manifest_sha256: String,
    results: Vec<RungResult>,
}

impl LadderRunner {
    /// Opens the supplied Phase-0 CPU FP32 fallback fixture pack and its committed floor.
    ///
    /// This is the standard local entry point for the currently available oracle evidence. It
    /// cannot be used to manufacture a Native CUDA claim; [`Self::from_paths`] verifies the
    /// manifest tier before any comparison can run.
    ///
    /// # Errors
    ///
    /// Returns an error when the local fixture cache is absent, either artifact is malformed, or
    /// the manifest is not the declared CPU FP32 fallback tier.
    pub fn cpu_fp32_fixture() -> Result<Self, LadderError> {
        Self::from_paths(
            OracleTier::CpuFp32Fallback,
            Path::new(CPU_FP32_FLOOR_PATH),
            Path::new(CPU_FP32_FIXTURE_MANIFEST_PATH),
        )
    }

    /// Loads the floor and fixture manifest from their canonical artifact paths.
    ///
    /// The manifest must describe `tier`; this prevents a CPU fallback capture from being emitted
    /// as Native CUDA evidence merely because the caller chose a different enum value.
    ///
    /// # Errors
    ///
    /// Returns an error when an artifact cannot be read or parsed, the manifest's tier differs,
    /// or a later floor policy is malformed.
    pub fn from_paths(
        tier: OracleTier,
        floor_path: &Path,
        fixture_manifest_path: &Path,
    ) -> Result<Self, LadderError> {
        let floor_source = floor_path.display().to_string();
        let floor_bytes = fs::read(floor_path).map_err(|source| LadderError::ReadArtifact {
            artifact: floor_source.clone(),
            source,
        })?;
        let manifest_source = fixture_manifest_path.display().to_string();
        let manifest_bytes =
            fs::read(fixture_manifest_path).map_err(|source| LadderError::ReadArtifact {
                artifact: manifest_source.clone(),
                source,
            })?;
        Self::from_bytes(
            tier,
            floor_source,
            &floor_bytes,
            manifest_source,
            &manifest_bytes,
        )
    }

    /// Creates a runner from artifact bytes; useful for hermetic tests and embedded evidence.
    ///
    /// # Errors
    ///
    /// Returns an error when either artifact is malformed or the manifest's tier differs.
    pub fn from_bytes(
        tier: OracleTier,
        floor_source: impl Into<String>,
        floor_bytes: &[u8],
        fixture_manifest_source: impl Into<String>,
        fixture_manifest_bytes: &[u8],
    ) -> Result<Self, LadderError> {
        let floor_source = floor_source.into();
        let floor: Value =
            serde_json::from_slice(floor_bytes).map_err(|source| LadderError::ParseArtifact {
                artifact: floor_source.clone(),
                source,
            })?;
        let fixture_manifest_source = fixture_manifest_source.into();
        let fixture_manifest: Value =
            serde_json::from_slice(fixture_manifest_bytes).map_err(|source| {
                LadderError::ParseArtifact {
                    artifact: fixture_manifest_source.clone(),
                    source,
                }
            })?;
        let manifest_tier = fixture_manifest
            .get("oracle_class")
            .and_then(Value::as_str)
            .ok_or_else(|| LadderError::InvalidFloor {
                reason: format!(
                    "fixture manifest `{fixture_manifest_source}` has no string oracle_class"
                ),
            })?;
        if manifest_tier != tier.as_str() {
            return Err(LadderError::TierManifestMismatch {
                requested: tier,
                manifest: manifest_tier.to_owned(),
            });
        }

        Ok(Self {
            tier,
            floor_source,
            floor_sha256: sha256(floor_bytes),
            floor,
            fixture_manifest_source,
            fixture_manifest_sha256: sha256(fixture_manifest_bytes),
            results: Vec::new(),
        })
    }

    /// Returns the SHA-256 hash of the exact floor artifact this runner loaded.
    #[must_use]
    pub fn floor_sha256(&self) -> &str {
        &self.floor_sha256
    }

    /// Returns the SHA-256 hash of the exact fixture manifest this runner loaded.
    #[must_use]
    pub fn fixture_manifest_sha256(&self) -> &str {
        &self.fixture_manifest_sha256
    }

    /// Runs one numeric rung and immediately emits its receipt.
    ///
    /// A floor entry with `status != observed` becomes an explicit skip using the artifact's own
    /// reason. A zero or absent `max_abs` envelope becomes an exact comparison; no fallback
    /// tolerance exists in this API.
    ///
    /// # Errors
    ///
    /// Returns an error only when the committed floor is malformed. A missing observation is an
    /// honest `Skipped` result, not an error and not a pass.
    pub fn compare_f32(
        &mut self,
        test: &str,
        rung: LadderRung,
        seam: &str,
        expected: &[f32],
        actual: &[f32],
    ) -> Result<&RungResult, LadderError> {
        let outcome = match self.policy_for(rung)? {
            FloorDecision::Skip(reason) => self.skip(test, rung, seam, reason),
            FloorDecision::Compare(policy) => {
                self.compare(test, rung, seam, expected, actual, policy)
            }
        };
        let index = self.results.len();
        self.results.push(outcome);
        Ok(&self.results[index])
    }

    /// Runs one token/ID rung with exact sequence comparison and emits its receipt.
    ///
    /// Exact Contract-A rungs such as L0 prompt IDs and L4 greedy codec tokens must have a zero
    /// or absent `max_abs` envelope. A nonzero numeric envelope cannot be repurposed as token
    /// tolerance, because doing so would conceal a discrete semantic change.
    ///
    /// # Errors
    ///
    /// Returns an error when the floor is malformed or tries to relax an exact sequence rung.
    pub fn compare_exact<T>(
        &mut self,
        test: &str,
        rung: LadderRung,
        seam: &str,
        expected: &[T],
        actual: &[T],
    ) -> Result<&RungResult, LadderError>
    where
        T: PartialEq + fmt::Debug,
    {
        let outcome = match self.policy_for(rung)? {
            FloorDecision::Skip(reason) => self.skip(test, rung, seam, reason),
            FloorDecision::Compare(ComparisonPolicy::Exact) => {
                self.compare_sequence(test, rung, seam, expected, actual)
            }
            FloorDecision::Compare(ComparisonPolicy::MaximumAbsolute(_)) => {
                return Err(LadderError::InvalidFloor {
                    reason: format!(
                        "contract_a.{} has a nonzero numeric envelope but {seam} is an exact sequence rung",
                        rung.floor_key()
                    ),
                });
            }
        };
        let index = self.results.len();
        self.results.push(outcome);
        Ok(&self.results[index])
    }

    /// Finalizes the current scorecard. A skipped L1 therefore forces `all_green: false`.
    #[must_use]
    pub fn scorecard(&self) -> LadderScorecard {
        LadderScorecard {
            results: self.results.clone(),
        }
    }

    fn policy_for(&self, rung: LadderRung) -> Result<FloorDecision, LadderError> {
        let entry = self
            .floor
            .get("contract_a")
            .and_then(|contract| contract.get(rung.floor_key()))
            .ok_or_else(|| LadderError::InvalidFloor {
                reason: format!("missing contract_a.{}", rung.floor_key()),
            })?;
        let status = entry.get("status").and_then(Value::as_str).ok_or_else(|| {
            LadderError::InvalidFloor {
                reason: format!("contract_a.{} has no string status", rung.floor_key()),
            }
        })?;
        if status != "observed" {
            let reason = entry.get("reason").and_then(Value::as_str).ok_or_else(|| {
                LadderError::InvalidFloor {
                    reason: format!(
                        "contract_a.{} is `{status}` but has no reason string",
                        rung.floor_key()
                    ),
                }
            })?;
            return Ok(FloorDecision::Skip(reason.to_owned()));
        }

        // A CPU floor may only justify CPU-FP32-fallback comparisons. Native CUDA needs its own
        // captured envelope; it must not inherit a permissive or exact CPU number.
        if self.tier != OracleTier::CpuFp32Fallback {
            return Ok(FloorDecision::Skip(format!(
                "{} is a CPU FP32 fallback floor and cannot supply a {} tolerance",
                self.floor_source,
                self.tier.as_str()
            )));
        }

        match entry.get("max_abs") {
            None | Some(Value::Null) => Ok(FloorDecision::Compare(ComparisonPolicy::Exact)),
            Some(Value::Number(number)) if number.as_f64() == Some(0.0) => {
                Ok(FloorDecision::Compare(ComparisonPolicy::Exact))
            }
            Some(Value::Number(number)) => {
                let tolerance = number.as_f64().ok_or_else(|| LadderError::InvalidFloor {
                    reason: format!(
                        "contract_a.{}.max_abs is not representable as f64",
                        rung.floor_key()
                    ),
                })?;
                if tolerance.is_sign_negative() || !tolerance.is_finite() {
                    return Err(LadderError::InvalidFloor {
                        reason: format!(
                            "contract_a.{}.max_abs must be finite and non-negative",
                            rung.floor_key()
                        ),
                    });
                }
                Ok(FloorDecision::Compare(ComparisonPolicy::MaximumAbsolute(
                    tolerance,
                )))
            }
            Some(_) => Err(LadderError::InvalidFloor {
                reason: format!("contract_a.{}.max_abs is not a number", rung.floor_key()),
            }),
        }
    }

    fn compare(
        &self,
        test: &str,
        rung: LadderRung,
        seam: &str,
        expected: &[f32],
        actual: &[f32],
        policy: ComparisonPolicy,
    ) -> RungResult {
        let comparison = compare_f32(expected, actual, policy.tolerance());
        let outcome = if comparison.holds() {
            Outcome::Passed
        } else {
            Outcome::Failed
        };
        let receipt = self
            .receipt(test, outcome, rung, seam)
            .tolerance(policy.tolerance(), self.floor_source.clone())
            .detail(json!({
                "comparison_policy": match policy {
                    ComparisonPolicy::Exact => "exact",
                    ComparisonPolicy::MaximumAbsolute(_) => "max_abs",
                },
                "comparison": comparison.to_json(),
            }));
        let receipt_json = receipt.to_json();
        receipt.emit();
        RungResult {
            rung,
            outcome,
            comparison_policy: Some(policy),
            comparison: Some(comparison),
            exact_comparison: None,
            receipt: receipt_json,
        }
    }

    fn compare_sequence<T>(
        &self,
        test: &str,
        rung: LadderRung,
        seam: &str,
        expected: &[T],
        actual: &[T],
    ) -> RungResult
    where
        T: PartialEq + fmt::Debug,
    {
        let comparison = compare_exact(expected, actual);
        let outcome = if comparison.holds() {
            Outcome::Passed
        } else {
            Outcome::Failed
        };
        let receipt = self
            .receipt(test, outcome, rung, seam)
            .tolerance(0.0, self.floor_source.clone())
            .detail(json!({
                "comparison_policy": "exact",
                "comparison": comparison.to_json(),
            }));
        let receipt_json = receipt.to_json();
        receipt.emit();
        RungResult {
            rung,
            outcome,
            comparison_policy: Some(ComparisonPolicy::Exact),
            comparison: None,
            exact_comparison: Some(comparison),
            receipt: receipt_json,
        }
    }

    fn skip(&self, test: &str, rung: LadderRung, seam: &str, reason: String) -> RungResult {
        let receipt = self
            .receipt(test, Outcome::Skipped, rung, seam)
            .reason(reason);
        let receipt_json = receipt.to_json();
        receipt.emit();
        RungResult {
            rung,
            outcome: Outcome::Skipped,
            comparison_policy: None,
            comparison: None,
            exact_comparison: None,
            receipt: receipt_json,
        }
    }

    fn receipt(&self, test: &str, outcome: Outcome, rung: LadderRung, seam: &str) -> Receipt {
        Receipt::new(test, outcome)
            .contract(format!("ConformanceExact/{}", rung.as_str()))
            .seam(seam)
            .provenance(
                FixtureProvenance::new(self.fixture_manifest_source.as_str())
                    .with_sha256(self.fixture_manifest_sha256.as_str())
                    .with_capture_provenance(self.tier.as_str(), self.tier_device(), "float32"),
            )
            .oracle_tier(self.tier)
            .ladder_artifacts(
                self.floor_sha256.as_str(),
                self.fixture_manifest_sha256.as_str(),
            )
    }

    const fn tier_device(&self) -> &'static str {
        match self.tier {
            OracleTier::CpuFp32Fallback => "cpu",
            OracleTier::NativeCuda => "cuda",
        }
    }
}

enum FloorDecision {
    Skip(String),
    Compare(ComparisonPolicy),
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOOR: &str = r#"{
        "contract_a": {
            "L0_prompt_token_ids": {"status": "observed", "max_abs": 0.0},
            "L1_operator_seams": {
                "status": "not_observed",
                "reason": "the fixture generator records L2+ named module seams, not individual operators"
            },
            "L2_layer_and_component_activations": {"status": "observed", "max_abs": 0.0},
            "L3_logits": {"status": "observed"}
        }
    }"#;
    const CPU_MANIFEST: &str = r#"{"oracle_class":"cpu_fp32_fallback"}"#;

    fn cpu_runner() -> LadderRunner {
        LadderRunner::from_bytes(
            OracleTier::CpuFp32Fallback,
            "floor.json",
            FLOOR.as_bytes(),
            "fixture_manifest.json",
            CPU_MANIFEST.as_bytes(),
        )
        .expect("valid CPU floor and manifest")
    }

    #[test]
    fn zero_envelope_is_exact_and_receipt_embeds_tier_and_hashes() {
        let mut runner = cpu_runner();
        let result = runner
            .compare_f32(
                "contract_a_l2_zero_envelope",
                LadderRung::L2LayerAndComponentActivations,
                "talker.layer_00.output",
                &[1.0],
                &[1.000_001],
            )
            .expect("floor is valid");

        assert_eq!(result.comparison_policy, Some(ComparisonPolicy::Exact));
        assert_eq!(
            result.outcome,
            Outcome::Failed,
            "no default epsilon is permitted"
        );
        assert_eq!(result.receipt["oracle_tier"], "cpu_fp32_fallback");
        assert_eq!(result.receipt["floor_sha256"], sha256(FLOOR.as_bytes()));
        assert_eq!(
            result.receipt["fixture_manifest_sha256"],
            sha256(CPU_MANIFEST.as_bytes())
        );
    }

    #[test]
    fn absent_envelope_is_also_exact() {
        let mut runner = cpu_runner();
        let result = runner
            .compare_f32(
                "contract_a_l3_absent_envelope",
                LadderRung::L3Logits,
                "talker.codec_head.output",
                &[1.0],
                &[1.000_001],
            )
            .expect("floor is valid");

        assert_eq!(result.comparison_policy, Some(ComparisonPolicy::Exact));
        assert_eq!(result.outcome, Outcome::Failed);
    }

    #[test]
    fn exact_sequence_rungs_do_not_accept_numeric_tolerance() {
        let mut runner = cpu_runner();
        let result = runner
            .compare_exact(
                "contract_a_l0_prompt_token_ids",
                LadderRung::L0PromptTokenIds,
                "prompt_build.text_ids",
                &[1_u32, 2],
                &[1_u32, 3],
            )
            .expect("zero floor supports exact IDs");

        assert_eq!(result.outcome, Outcome::Failed);
        assert!(result.exact_comparison.is_some());
        assert_eq!(result.receipt["tolerance"], 0.0);
    }

    #[test]
    fn l1_is_an_explicit_floor_reasoned_skip_and_blocks_all_green() {
        let mut runner = cpu_runner();
        let result = runner
            .compare_f32(
                "contract_a_l1_operator_seams",
                LadderRung::L1OperatorSeams,
                "talker.layer_00.rms_norm",
                &[1.0],
                &[1.0],
            )
            .expect("not-observed L1 is a skip, not a floor parse error");

        assert_eq!(result.outcome, Outcome::Skipped);
        assert_eq!(
            result.receipt["reason"],
            "the fixture generator records L2+ named module seams, not individual operators"
        );
        assert!(
            !runner.scorecard().all_green(),
            "a skipped L1 cannot be green"
        );
    }

    #[test]
    fn native_cuda_cannot_relabel_cpu_fixture_evidence() {
        let error = LadderRunner::from_bytes(
            OracleTier::NativeCuda,
            "floor.json",
            FLOOR.as_bytes(),
            "fixture_manifest.json",
            CPU_MANIFEST.as_bytes(),
        )
        .expect_err("CPU fixture cannot become native CUDA evidence");

        assert!(matches!(error, LadderError::TierManifestMismatch { .. }));
    }
}
