//! Human-facing console output: colored status lines and interactive confirmation.
//!
//! # Why this is not `gum`
//!
//! The shell-installer house style reaches for [`gum`](https://github.com/charmbracelet/gum) and
//! falls back to raw ANSI when it is absent. A CLI cannot borrow that directly: shelling out to a
//! formatter would make a *runtime dependency* out of pretty output, and this project ships one
//! binary with no runtime dependencies at all. What transfers is the shape of the output stack —
//! `info` / `ok` / `warn`, one visual grammar, and graceful degradation — so it is reimplemented
//! here in a dozen lines of ANSI instead.
//!
//! # Degradation is the contract
//!
//! Color is emitted only when stdout is a terminal and `NO_COLOR` is unset. Everything here is
//! therefore inert under a pipe, a file, a test harness capturing to a `Vec<u8>`, and robot mode —
//! which matters more than the color does: NDJSON consumers and golden-output tests must never
//! have to strip escape sequences. Prompts follow the same rule and refuse to block when there is
//! no human attached.

use std::io::{IsTerminal, Write};

/// Whether human-facing decoration should be emitted at all.
///
/// Honors the [NO_COLOR convention](https://no-color.org): any non-empty value disables color.
#[must_use]
pub fn decorate() -> bool {
    if std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty()) {
        return false;
    }
    std::io::stdout().is_terminal()
}

/// Paints `text` with an SGR code, or returns it unchanged when decoration is off.
fn paint(code: &str, text: &str) -> String {
    if decorate() {
        format!("\u{1b}[{code}m{text}\u{1b}[0m")
    } else {
        text.to_owned()
    }
}

/// A completed step.
///
/// # Errors
///
/// When the sink cannot be written.
pub fn ok(out: &mut dyn Write, message: &str) -> std::io::Result<()> {
    writeln!(out, "{} {message}", paint("32;1", "✓"))
}

/// A step in progress, or a neutral fact worth showing.
///
/// # Errors
///
/// When the sink cannot be written.
pub fn info(out: &mut dyn Write, message: &str) -> std::io::Result<()> {
    writeln!(out, "{} {message}", paint("34;1", "→"))
}

/// Something the user should notice but which is not a failure.
///
/// # Errors
///
/// When the sink cannot be written.
pub fn warn(out: &mut dyn Write, message: &str) -> std::io::Result<()> {
    writeln!(out, "{} {message}", paint("33;1", "!"))
}

/// Dims secondary detail so it reads as subordinate to the line above it.
#[must_use]
pub fn detail(text: &str) -> String {
    paint("2", text)
}

/// Emphasizes a path or value inside a sentence.
#[must_use]
pub fn emphasis(text: &str) -> String {
    paint("1", text)
}

/// Asks a yes/no question, defaulting to no.
///
/// Returns `None` when there is no human to ask — stdin or stdout is not a terminal — so callers
/// can keep their non-interactive behavior (a clear error) instead of blocking a script or an
/// agent forever on a prompt nothing will answer. That distinction is the whole reason this
/// returns an `Option` rather than a `bool`.
///
/// # Errors
///
/// When the prompt cannot be written or stdin cannot be read.
pub fn confirm(out: &mut dyn Write, question: &str) -> std::io::Result<Option<bool>> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Ok(None);
    }
    write!(
        out,
        "{} {question} {} ",
        paint("33;1", "?"),
        detail("[y/N]")
    )?;
    out.flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let answer = answer.trim();
    Ok(Some(
        answer.eq_ignore_ascii_case("y") || answer.eq_ignore_ascii_case("yes"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Captured output must be free of escape sequences: golden tests and NDJSON consumers read
    /// this same text, and a stray SGR code would be a parsing bug rather than a cosmetic one.
    #[test]
    fn status_lines_carry_no_escapes_when_the_sink_is_not_a_terminal() {
        let mut buffer: Vec<u8> = Vec::new();
        ok(&mut buffer, "enrolled").expect("write");
        info(&mut buffer, "loading").expect("write");
        warn(&mut buffer, "noisy reference").expect("write");
        let text = String::from_utf8(buffer).expect("utf8");
        assert!(
            !text.contains('\u{1b}'),
            "decoration leaked into a captured sink: {text:?}"
        );
        assert!(text.contains("enrolled") && text.contains("loading"));
    }

    /// A prompt with no terminal attached must not block; it reports "nobody to ask".
    #[test]
    fn confirm_declines_to_block_without_a_terminal() {
        let mut buffer: Vec<u8> = Vec::new();
        let answer = confirm(&mut buffer, "Overwrite?").expect("confirm");
        assert_eq!(
            answer, None,
            "a non-interactive run must fall through to the caller's own policy"
        );
        assert!(
            buffer.is_empty(),
            "nothing should be printed with no reader"
        );
    }
}
