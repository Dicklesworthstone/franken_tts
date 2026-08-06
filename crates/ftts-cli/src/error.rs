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
}

impl FttsError {
    pub const fn exit_code(&self) -> FttsExitCode {
        match self {
            Self::Generic(_) => FttsExitCode::Generic,
            Self::Usage(_) => FttsExitCode::Usage,
            Self::ModelNotFound(_) => FttsExitCode::ModelNotFound,
            Self::Input(_) => FttsExitCode::Input,
            Self::BudgetTimeout(_) => FttsExitCode::BudgetTimeout,
            Self::ArtifactFormat(_) => FttsExitCode::ArtifactFormat,
            Self::EnrollmentQualityRefusal(_) => FttsExitCode::EnrollmentQualityRefusal,
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
            | Self::EnrollmentQualityRefusal(message) => formatter.write_str(message),
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
        ];

        for (code, expected) in cases {
            assert_eq!(code.as_u8(), expected, "{}", code.description());
        }
    }
}
