//! The model gate: how a test that needs multi-GB weights behaves when they are absent.
//!
//! Two rules, both non-negotiable:
//!
//! 1. **A skip is never a pass.** Absent weights must not turn a suite red, but the receipt has to
//!    say `skipped`, not `passed`. `libtest` has no first-class skip — an early `return` is
//!    reported as a pass — so the honest signal lives in the emitted receipt, and the ladder runner
//!    aggregates receipts rather than `libtest`'s pass count.
//! 2. **A present model must prove the native path ran.** The classic failure is a fallback that
//!    quietly substitutes for the real path, leaving a green suite that tested nothing. Point every
//!    fallback at a nonexistent location and make its use a loud error.

use std::{
    env,
    path::{Path, PathBuf},
};

/// Environment variable naming the directory holding the quantized model artifact.
///
/// Shared with the CLI (`ftts-cli`) so tests and the binary never disagree about where weights are.
pub const MODEL_DIR_ENV: &str = "FTTS_MODEL_DIR";

/// The canonical artifact filename inside [`MODEL_DIR_ENV`].
pub const MODEL_BASENAME: &str = "qwen3-tts-12hz-0.6b-base.fttsq";

/// A path that must never exist, for aiming fallbacks at so their use is detectable.
///
/// Handing this to a fallback turns "silently degraded" into "loudly broken".
pub const NONEXISTENT_FALLBACK: &str = "/nonexistent/ftts-fallback-must-not-be-used";

/// Whether the weights needed by a model-gated test are available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelGate {
    /// The artifact exists; the test must run for real and prove the native path executed.
    Present {
        /// Absolute path to the resolved artifact.
        artifact: PathBuf,
    },
    /// The artifact is unavailable; the test records an honest skip.
    Absent {
        /// Why the gate is closed, phrased so a CI reader needs no follow-up.
        reason: String,
    },
}

impl ModelGate {
    /// Resolves the gate from the process environment.
    ///
    /// Prefer [`ModelGate::resolve_from`] in tests: it takes the directory explicitly, so no test
    /// mutates process-global state. (Edition 2024 makes `env::set_var` `unsafe`, which this crate
    /// forbids outright — injecting the directory is the only way to test both branches.)
    #[must_use]
    pub fn resolve() -> Self {
        match env::var(MODEL_DIR_ENV) {
            Ok(dir) if !dir.trim().is_empty() => Self::resolve_from(Some(Path::new(&dir))),
            _ => Self::resolve_from(None),
        }
    }

    /// Resolves the gate against an explicitly supplied model directory.
    #[must_use]
    pub fn resolve_from(model_dir: Option<&Path>) -> Self {
        let Some(dir) = model_dir else {
            return Self::Absent {
                reason: format!("{MODEL_DIR_ENV} is unset or empty"),
            };
        };
        if !dir.is_dir() {
            return Self::Absent {
                reason: format!("{MODEL_DIR_ENV}={} is not a directory", dir.display()),
            };
        }
        let artifact = dir.join(MODEL_BASENAME);
        if artifact.is_file() {
            Self::Present { artifact }
        } else {
            Self::Absent {
                reason: format!("{MODEL_BASENAME} not found in {}", dir.display()),
            }
        }
    }

    /// Returns the artifact path when the gate is open.
    #[must_use]
    pub fn artifact(&self) -> Option<&Path> {
        match self {
            Self::Present { artifact } => Some(artifact.as_path()),
            Self::Absent { .. } => None,
        }
    }

    /// Returns whether weights are available.
    #[must_use]
    pub const fn is_present(&self) -> bool {
        matches!(self, Self::Present { .. })
    }
}

/// Which code path actually produced a result.
///
/// A model-gated test asserts on this rather than on the result alone: a plausible-looking output
/// from a fallback is exactly the outcome the gate exists to catch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionPath {
    /// The real, model-backed implementation ran.
    Native,
    /// A substitute ran. Never acceptable under an open gate.
    Fallback,
}

/// Rejects a result that came from a fallback while the gate was open.
///
/// # Errors
///
/// Returns a message naming the seam and both paths when `observed` is [`ExecutionPath::Fallback`],
/// so the failure localizes without a debugger rerun.
pub fn require_native_execution(seam: &str, observed: ExecutionPath) -> Result<(), String> {
    match observed {
        ExecutionPath::Native => Ok(()),
        ExecutionPath::Fallback => Err(format!(
            "seam `{seam}` ran the FALLBACK path while the model gate was OPEN. \
             A model-gated assertion passed without exercising the native implementation, \
             which makes the result meaningless. Aim fallbacks at `{NONEXISTENT_FALLBACK}` \
             so this fails at construction instead of silently substituting."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = env::temp_dir().join(format!("ftts-gate-{tag}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir is creatable");
        dir
    }

    #[test]
    fn gate_is_absent_without_a_model_directory() {
        let gate = ModelGate::resolve_from(None);
        assert!(!gate.is_present());
        assert!(matches!(gate, ModelGate::Absent { ref reason } if reason.contains(MODEL_DIR_ENV)));
    }

    #[test]
    fn gate_is_absent_when_the_directory_lacks_the_artifact() {
        let dir = temp_dir("empty");
        let gate = ModelGate::resolve_from(Some(&dir));
        assert!(
            matches!(gate, ModelGate::Absent { ref reason } if reason.contains(MODEL_BASENAME))
        );
    }

    #[test]
    fn gate_is_present_when_the_artifact_exists() {
        let dir = temp_dir("present");
        let artifact = dir.join(MODEL_BASENAME);
        fs::write(
            &artifact,
            b"not a real artifact; the gate only checks presence",
        )
        .expect("artifact file is writable");
        let gate = ModelGate::resolve_from(Some(&dir));
        assert!(gate.is_present());
        assert_eq!(gate.artifact(), Some(artifact.as_path()));
        fs::remove_file(&artifact).expect("test artifact is removable");
    }

    #[test]
    fn the_nonexistent_fallback_path_really_does_not_exist() {
        assert!(
            !Path::new(NONEXISTENT_FALLBACK).exists(),
            "the sentinel fallback path must never exist, or fallback detection is defeated"
        );
    }

    #[test]
    fn native_execution_is_accepted_and_fallback_is_rejected_loudly() {
        assert!(require_native_execution("talker.prefill", ExecutionPath::Native).is_ok());
        let error = require_native_execution("talker.prefill", ExecutionPath::Fallback)
            .expect_err("a fallback under an open gate must be an error");
        assert!(error.contains("talker.prefill"), "names the seam: {error}");
        assert!(error.contains("FALLBACK"), "names the path taken: {error}");
        assert!(
            error.contains(NONEXISTENT_FALLBACK),
            "says how to fix: {error}"
        );
    }
}
