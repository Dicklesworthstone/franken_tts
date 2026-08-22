//! Stable CLI error and exit-code contract.

use std::fmt;
use std::process::ExitCode;

/// The process exit codes promised by the `ftts` robot and human interfaces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum FttsExitCode {
    Success = 0,
    Generic = 1,
    Usage = 2,
    ModelNotFound = 3,
    Input = 4,
    BudgetTimeout = 5,
    Cancelled = 6,
    ArtifactFormat = 7,
    EnrollmentQualityRefusal = 8,
    SessionTransport = 9,
}

impl FttsExitCode {
    /// Stable, machine-readable explanation of this process status.
    pub const fn description(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Generic => "generic error",
            Self::Usage => "usage or CLI error",
            Self::ModelNotFound => "model not found or not resolvable",
            Self::Input => "input error",
            Self::BudgetTimeout => "budget or timeout exceeded",
            Self::Cancelled => "cancelled",
            Self::ArtifactFormat => "artifact format or version mismatch",
            Self::EnrollmentQualityRefusal => "enrollment-quality refusal",
            Self::SessionTransport => "talk-session transport failed",
        }
    }

    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn as_exit_code(self) -> ExitCode {
        ExitCode::from(self.as_u8())
    }
}

/// An error that can be presented by the Phase-0 CLI without fabricating engine work.
#[derive(Debug)]
pub enum FttsError {
    Generic(String),
    Usage(String),
    ModelNotFound(String),
    Input(String),
    BudgetTimeout(String),
    ArtifactFormat(String),
    EnrollmentQualityRefusal(String),
    /// The run stopped because SIGINT/SIGTERM asked it to. The message states what the
    /// cancelled run left behind — the finalized partial artifact, or the encoding that
    /// was skipped — so the existing `run_error` shape carries the whole disposition
    /// without any new fields.
    Cancelled(String),
    /// A `ftts talk` transport death no op could cause and no op can recover from.
    SessionTransport(String),
}

impl FttsError {
    /// A stable, actionable next step for this failure class.
    ///
    /// Kept separate from the message so an agent can branch on remediation without parsing prose,
    /// and so the message stays free to name the specific offending path or value.
    pub const fn remediation(&self) -> &'static str {
        match self {
            Self::Generic(_) => {
                "see the message; if it names an unimplemented phase, the capability is not built yet"
            }
            Self::Usage(_) => "re-run with --help to see the accepted argument shapes",
            Self::ModelNotFound(_) => {
                "run `ftts pull` to fetch the model (~2.0 GB), or pass --model PATH or set FTTS_MODEL_DIR; `ftts robot health` lists every directory searched"
            }
            Self::Input(_) => {
                "check the input text or file encoding; input must be non-empty UTF-8"
            }
            Self::BudgetTimeout(_) => {
                "raise the budget, shorten the text, or choose a faster profile"
            }
            Self::ArtifactFormat(_) => {
                "regenerate the artifact with a matching ftts version; `ftts robot health` \
                 reports the expected format"
            }
            Self::EnrollmentQualityRefusal(_) => {
                "supply a cleaner reference, or pass --force to accept the warned-about quality"
            }
            Self::Cancelled(_) => {
                "the run stopped because a signal asked it to; re-run the command to \
                 synthesize again"
            }
            Self::SessionTransport(_) => {
                "the session process is exiting; restart `ftts talk`. If it \
                                           recurs, check that the PCM consumer is still draining \
                                           the audio channel - a wedged reader is the one failure \
                                           the session cannot absorb"
            }
        }
    }

    pub const fn exit_code(&self) -> FttsExitCode {
        match self {
            Self::Generic(_) => FttsExitCode::Generic,
            Self::Usage(_) => FttsExitCode::Usage,
            Self::ModelNotFound(_) => FttsExitCode::ModelNotFound,
            Self::Input(_) => FttsExitCode::Input,
            Self::BudgetTimeout(_) => FttsExitCode::BudgetTimeout,
            Self::ArtifactFormat(_) => FttsExitCode::ArtifactFormat,
            Self::EnrollmentQualityRefusal(_) => FttsExitCode::EnrollmentQualityRefusal,
            Self::Cancelled(_) => FttsExitCode::Cancelled,
            Self::SessionTransport(_) => FttsExitCode::SessionTransport,
        }
    }
}

impl fmt::Display for FttsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generic(message)
            | Self::Usage(message)
            | Self::ModelNotFound(message)
            | Self::Input(message)
            | Self::BudgetTimeout(message)
            | Self::ArtifactFormat(message)
            | Self::EnrollmentQualityRefusal(message)
            | Self::Cancelled(message)
            | Self::SessionTransport(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for FttsError {}

#[cfg(test)]
mod tests {
    use super::FttsExitCode;

    #[test]
    fn exit_codes_are_stable() {
        let cases = [
            (FttsExitCode::Success, 0),
            (FttsExitCode::Generic, 1),
            (FttsExitCode::Usage, 2),
            (FttsExitCode::ModelNotFound, 3),
            (FttsExitCode::Input, 4),
            (FttsExitCode::BudgetTimeout, 5),
            (FttsExitCode::Cancelled, 6),
            (FttsExitCode::ArtifactFormat, 7),
            (FttsExitCode::EnrollmentQualityRefusal, 8),
            (FttsExitCode::SessionTransport, 9),
        ];

        for (code, expected) in cases {
            assert_eq!(code.as_u8(), expected, "{}", code.description());
        }
    }

    /// The cancellation contract: an interrupted run reports exit code 6 under kind
    /// "cancelled" through the same mapping every other failure class uses.
    #[test]
    fn cancelled_errors_carry_the_documented_terminal_shape() {
        let error = super::FttsError::Cancelled("partial WAV kept at out.wav".to_owned());
        assert_eq!(error.exit_code().as_u8(), 6);
        assert_eq!(error.exit_code().description(), "cancelled");
        assert_eq!(error.to_string(), "partial WAV kept at out.wav");
        assert!(!error.remediation().is_empty());
    }
}
