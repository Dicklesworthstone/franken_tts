//! Navigating the oracle fixture packs, and comparing a Rust stage against the captured one.
//!
//! # What a fixture pack looks like
//!
//! ```text
//! ft7-cpu-fp32-r1/
//!   fixture_manifest.json          cases, modes, per-mode manifest digests, oracle_class
//!   provenance.json
//!   <case>/<mode>/stages/<group>/<seam>/<slot>.<step>.npy
//! ```
//!
//! A seam directory holds one file per captured slot per decode step, e.g.
//! `talker.layer_13.input/args.0.000.npy` (the layer's hidden-state argument at step 0) beside
//! `kwargs.position_ids.000.npy`. Output seams use bare slot names: `talker.layer_16.output/0.000.npy`.
//!
//! # Why teacher-forced comparison needs the input seams too
//!
//! The parity discipline is to feed each Rust stage **the oracle's exact input tensor**, so a layer
//! is judged on its own arithmetic rather than on error accumulated upstream. That only works if the
//! harness can read the input seam as readily as the output seam — hence
//! [`OracleFixtures::seam`] takes the slot, not just the seam.
//!
//! # Exact compare at the CPU tier
//!
//! `docs/truth-pack/nondeterminism-floor.json` measured **max_abs 0.0 at every observed CPU-tier
//! seam**, so Contract A at this tier is an *exact* comparison. [`compare_exactly`] therefore uses a
//! zero tolerance. Any nonzero epsilon must be a ledgered DISC entry, never an inline constant
//! quietly introduced here to make a stage pass.
//!
//! Bead: `frankentts-p1-talker-z2w`.

use std::{
    fmt, fs,
    path::{Path, PathBuf},
};

use crate::{
    compare::{Comparison, compare_f32, describe_failure},
    npy::{self, NpyArray},
};

/// Environment variable overriding the fixture-pack location.
pub const FIXTURES_ENV: &str = "FTTS_ORACLE_FIXTURES";

/// Where the CPU-fp32 pack is cached by default, relative to `$HOME`.
pub const DEFAULT_RELATIVE_PATH: &str = ".cache/frankentts/oracle-fixtures/ft7-cpu-fp32-r1";

/// The oracle tier this pack was captured at.
///
/// CPU-fp32 is a *fallback* tier: green here is real evidence, but it is not the native-CUDA golden
/// proof, and a receipt must not let one be read as the other.
pub const CPU_FP32_ORACLE_CLASS: &str = "cpu_fp32_fallback";

/// Tolerance for a CPU-tier Contract A comparison: exact.
pub const CPU_TIER_TOLERANCE: f64 = 0.0;

/// Provenance string for that tolerance, carried on every receipt.
pub const CPU_TIER_TOLERANCE_SOURCE: &str =
    "docs/truth-pack/nondeterminism-floor.json (max_abs 0.0 at every observed CPU-tier seam)";

/// Why a fixture could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixtureError {
    /// The pack directory is absent. Expected on a machine that has not fetched it.
    PackAbsent {
        /// Where the pack was looked for.
        path: String,
    },
    /// The pack is present but a requested seam is not.
    SeamAbsent {
        /// The seam directory.
        seam: String,
        /// The specific file.
        path: String,
    },
    /// The manifest could not be read or parsed.
    Manifest {
        /// What went wrong.
        detail: String,
    },
    /// The pack declares a different oracle tier than the caller required.
    WrongOracleClass {
        /// Tier the pack declares.
        found: String,
        /// Tier the caller required.
        required: String,
    },
    /// A fixture file was unreadable.
    Npy {
        /// The file.
        path: String,
        /// The reader's complaint.
        detail: String,
    },
}

impl fmt::Display for FixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PackAbsent { path } => write!(
                f,
                "oracle fixture pack not found at `{path}`; set {FIXTURES_ENV} or fetch the pack"
            ),
            Self::SeamAbsent { seam, path } => {
                write!(f, "seam `{seam}` has no fixture at `{path}`")
            }
            Self::Manifest { detail } => write!(f, "fixture manifest is unusable: {detail}"),
            Self::WrongOracleClass { found, required } => write!(
                f,
                "fixture pack declares oracle tier `{found}`, not the required `{required}`; \
                 a lower-tier pass must never be recorded as higher-tier evidence"
            ),
            Self::Npy { path, detail } => write!(f, "cannot decode `{path}`: {detail}"),
        }
    }
}

impl std::error::Error for FixtureError {}

/// A located oracle fixture pack.
#[derive(Clone, Debug)]
pub struct OracleFixtures {
    root: PathBuf,
    oracle_class: String,
}

impl OracleFixtures {
    /// Locates the pack from the environment, or the default cache path.
    ///
    /// Returns `Err(FixtureError::PackAbsent)` when it is simply not on this machine — the caller
    /// is expected to turn that into an **honest skip receipt**, not a failure. Absent weights and
    /// absent fixtures are the same situation: CI stays green without them, and the receipt says
    /// `skipped`.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError`] when the pack is absent or its manifest is unusable.
    pub fn open_default() -> Result<Self, FixtureError> {
        let root = match std::env::var(FIXTURES_ENV) {
            Ok(value) if !value.trim().is_empty() => PathBuf::from(value),
            _ => {
                let home = std::env::var("HOME").unwrap_or_default();
                Path::new(&home).join(DEFAULT_RELATIVE_PATH)
            }
        };
        Self::open(&root)
    }

    /// Opens a pack at an explicit path.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError`] when the directory or its manifest is missing or malformed.
    pub fn open(root: &Path) -> Result<Self, FixtureError> {
        let manifest_path = root.join("fixture_manifest.json");
        if !manifest_path.is_file() {
            return Err(FixtureError::PackAbsent {
                path: root.display().to_string(),
            });
        }
        let bytes = fs::read(&manifest_path).map_err(|error| FixtureError::Manifest {
            detail: error.to_string(),
        })?;
        let manifest: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|error| FixtureError::Manifest {
                detail: error.to_string(),
            })?;
        let oracle_class = manifest["oracle_class"]
            .as_str()
            .ok_or_else(|| FixtureError::Manifest {
                detail: "no `oracle_class` field".to_owned(),
            })?
            .to_owned();

        Ok(Self {
            root: root.to_path_buf(),
            oracle_class,
        })
    }

    /// The oracle tier this pack was captured at.
    #[must_use]
    pub fn oracle_class(&self) -> &str {
        &self.oracle_class
    }

    /// The pack root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Asserts the pack is the tier the caller requires.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError::WrongOracleClass`] on a mismatch, so a CPU-tier pass can never be
    /// filed as native-tier evidence.
    pub fn require_oracle_class(&self, required: &str) -> Result<(), FixtureError> {
        if self.oracle_class == required {
            Ok(())
        } else {
            Err(FixtureError::WrongOracleClass {
                found: self.oracle_class.clone(),
                required: required.to_owned(),
            })
        }
    }

    /// Path of one captured slot: `<case>/<mode>/stages/<group>/<seam>/<slot>.<step:03>.npy`.
    #[must_use]
    pub fn seam_path(&self, seam: &SeamRef<'_>, slot: &str, step: usize) -> PathBuf {
        self.root
            .join(seam.case)
            .join(seam.mode)
            .join("stages")
            .join(seam.group)
            .join(seam.seam)
            .join(format!("{slot}.{step:03}.npy"))
    }

    /// Loads one captured slot.
    ///
    /// # Errors
    ///
    /// Returns [`FixtureError::SeamAbsent`] when the file is missing, or [`FixtureError::Npy`] when
    /// it cannot be decoded.
    pub fn seam(
        &self,
        seam: &SeamRef<'_>,
        slot: &str,
        step: usize,
    ) -> Result<NpyArray, FixtureError> {
        let path = self.seam_path(seam, slot, step);
        if !path.is_file() {
            return Err(FixtureError::SeamAbsent {
                seam: seam.describe(),
                path: path.display().to_string(),
            });
        }
        npy::read(&path).map_err(|error| FixtureError::Npy {
            path: path.display().to_string(),
            detail: error.to_string(),
        })
    }

    /// Whether a seam directory exists at all, for probing which layers were captured.
    #[must_use]
    pub fn has_seam(&self, seam: &SeamRef<'_>) -> bool {
        self.root
            .join(seam.case)
            .join(seam.mode)
            .join("stages")
            .join(seam.group)
            .join(seam.seam)
            .is_dir()
    }
}

/// Names one captured seam within a pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeamRef<'a> {
    /// Corpus case, e.g. `synthetic-tone-en`.
    pub case: &'a str,
    /// Prompt mode, e.g. `icl_non_streaming`.
    pub mode: &'a str,
    /// Stage group, e.g. `talker_free_running`.
    pub group: &'a str,
    /// Seam name, e.g. `talker.layer_13.input`.
    pub seam: &'a str,
}

impl SeamRef<'_> {
    /// A single string naming this seam, for receipts and failure messages.
    #[must_use]
    pub fn describe(&self) -> String {
        format!("{}/{}/{}/{}", self.case, self.mode, self.group, self.seam)
    }
}

/// Compares a Rust stage output against the oracle's, at the CPU tier's exact tolerance.
///
/// Shape is compared before values: a shape disagreement is a wiring bug that no tolerance can
/// express, and reporting it as "4,194,304 elements over tolerance" hides the actual cause.
///
/// # Errors
///
/// Returns the self-localizing report when the stage diverges, naming the first divergent element
/// with its coordinates.
pub fn compare_exactly(
    seam: &str,
    expected: &NpyArray,
    actual: &[f32],
) -> Result<Comparison, String> {
    if expected.data.len() != actual.len() {
        return Err(format!(
            "seam `{seam}` shape mismatch: oracle has {} elements with shape {}, ours has {} — \
             this is a wiring bug, not a tolerance question",
            expected.data.len(),
            expected.shape_string(),
            actual.len()
        ));
    }

    let comparison = compare_f32(&expected.data, actual, CPU_TIER_TOLERANCE);
    if comparison.holds() {
        Ok(comparison)
    } else {
        Err(describe_failure(
            seam,
            &comparison,
            CPU_TIER_TOLERANCE,
            CPU_TIER_TOLERANCE_SOURCE,
            Some(&expected.shape),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pack lives outside the repo, so every fixture-backed test skips honestly without it.
    fn pack() -> Option<OracleFixtures> {
        OracleFixtures::open_default().ok()
    }

    const TALKER: SeamRef<'static> = SeamRef {
        case: "synthetic-tone-en",
        mode: "icl_non_streaming",
        group: "talker_free_running",
        seam: "talker.layer_13.input",
    };

    #[test]
    fn an_absent_pack_is_a_named_condition_a_caller_can_skip_on() {
        let error = OracleFixtures::open(Path::new("/nonexistent/ftts-oracle-pack"))
            .expect_err("an absent pack must be reported, not panicked on");
        assert!(matches!(error, FixtureError::PackAbsent { .. }));
        // The message has to say where it looked, or the reader cannot fix it.
        assert!(error.to_string().contains("/nonexistent/ftts-oracle-pack"));
        assert!(error.to_string().contains(FIXTURES_ENV));
    }

    #[test]
    fn a_shape_disagreement_is_reported_as_wiring_not_as_tolerance() {
        let expected = NpyArray {
            shape: vec![2, 3],
            data: vec![1.0; 6],
        };
        let error = compare_exactly("talker.layer_13.output", &expected, &[1.0; 5])
            .expect_err("a length mismatch must fail");
        assert!(error.contains("shape mismatch"), "{error}");
        assert!(error.contains("[2, 3]"), "{error}");
        assert!(error.contains("wiring bug"), "{error}");
    }

    #[test]
    fn identical_values_hold_and_a_one_ulp_difference_does_not() {
        let expected = NpyArray {
            shape: vec![4],
            data: vec![0.1, 0.2, 0.3, 0.4],
        };
        assert!(compare_exactly("seam", &expected, &expected.data).is_ok());

        // The CPU tier is an EXACT compare: the smallest representable difference must fail.
        let mut perturbed = expected.data.clone();
        perturbed[2] = f32::from_bits(perturbed[2].to_bits() + 1);
        let error = compare_exactly("seam", &expected, &perturbed)
            .expect_err("one ULP must fail an exact comparison");
        assert!(error.contains("flat[2]"), "{error}");
        assert!(
            error.contains("nondeterminism-floor.json"),
            "the tolerance must name its source: {error}"
        );
    }

    /// Reads a real captured seam, when the pack is present.
    #[test]
    fn a_real_ft7_seam_decodes_with_the_expected_geometry() {
        let Some(pack) = pack() else {
            // Honest skip: the pack is not on this machine. Never a silent pass.
            eprintln!("SKIP: oracle pack absent; set {FIXTURES_ENV} to run this");
            return;
        };
        pack.require_oracle_class(CPU_FP32_ORACLE_CLASS)
            .expect("the cached pack is the CPU-fp32 tier");

        assert!(
            pack.has_seam(&TALKER),
            "expected {} in the pack",
            TALKER.describe()
        );
        let hidden = pack
            .seam(&TALKER, "args.0", 0)
            .expect("the layer input is captured");

        // [batch, seq, hidden] with the pinned hidden size of 1024.
        assert_eq!(hidden.shape.len(), 3, "shape was {}", hidden.shape_string());
        assert_eq!(hidden.shape[0], 1, "batch of one");
        assert_eq!(
            hidden.shape[2],
            1024,
            "talker hidden size is 1024; got {}",
            hidden.shape_string()
        );
        assert_eq!(hidden.data.len(), hidden.shape.iter().product::<usize>());
        assert!(
            hidden.data.iter().all(|value| value.is_finite()),
            "a captured activation containing NaN/Inf would poison every downstream comparison"
        );
    }

    /// A seam compared against itself must hold exactly — the harness's own sanity check.
    #[test]
    fn a_captured_seam_compares_exactly_against_itself() {
        let Some(pack) = pack() else {
            eprintln!("SKIP: oracle pack absent; set {FIXTURES_ENV} to run this");
            return;
        };
        let hidden = pack
            .seam(&TALKER, "args.0", 0)
            .expect("the layer input is captured");
        let comparison = compare_exactly(&TALKER.describe(), &hidden, &hidden.data)
            .expect("a seam must equal itself at zero tolerance");
        assert!(comparison.holds());
        assert_eq!(comparison.over_tolerance, 0);
        assert_eq!(comparison.non_finite, 0);
    }

    #[test]
    fn a_missing_seam_names_both_the_seam_and_the_file() {
        let Some(pack) = pack() else {
            eprintln!("SKIP: oracle pack absent; set {FIXTURES_ENV} to run this");
            return;
        };
        let absent = SeamRef {
            seam: "talker.layer_999.input",
            ..TALKER
        };
        let error = pack
            .seam(&absent, "args.0", 0)
            .expect_err("a layer that was never captured must be a named refusal");
        let FixtureError::SeamAbsent { seam, path } = &error else {
            panic!("expected SeamAbsent, got {error}");
        };
        assert!(seam.contains("talker.layer_999.input"));
        assert!(path.ends_with("args.0.000.npy"));
    }

    #[test]
    fn a_lower_tier_pack_cannot_be_passed_off_as_a_higher_tier_one() {
        let Some(pack) = pack() else {
            eprintln!("SKIP: oracle pack absent; set {FIXTURES_ENV} to run this");
            return;
        };
        assert_eq!(pack.oracle_class(), CPU_FP32_ORACLE_CLASS);
        let error = pack
            .require_oracle_class("native_cuda_golden")
            .expect_err("CPU-tier evidence must not satisfy a native-tier requirement");
        assert!(matches!(error, FixtureError::WrongOracleClass { .. }));
    }
}
