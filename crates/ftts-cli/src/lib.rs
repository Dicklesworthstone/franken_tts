#![forbid(unsafe_code)]

//! Shared command-line dispatch for both FrankenTTS binaries.

use std::process::ExitCode;

/// Runs the command-line interface.
pub fn cli_main() -> ExitCode {
    ExitCode::SUCCESS
}
