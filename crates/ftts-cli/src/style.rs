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

/// Whether a human is reading stdout, as opposed to a pipe, a file, an agent, or CI.
///
/// Deliberately distinct from [`decorate`]: `NO_COLOR` means "no color", not "give me JSON", so a
/// user who sets it still gets human output, just uncolored.
#[must_use]
pub fn is_interactive() -> bool {
    std::io::stdout().is_terminal()
}

/// Renders the `say` lifecycle as something a person wants to read.
///
/// The NDJSON stream is a machine contract — stable schema, one event per line, `audio_chunk` per
/// packet — and it is exactly the wrong thing to put in front of somebody who typed a sentence and
/// wants a file. Rather than weaken that contract, this consumes the same events and prints a
/// different view of them; the stream is unchanged for every non-terminal consumer.
#[derive(Default)]
pub struct SayPresenter {
    destination: Option<String>,
    load_started_ms: u64,
    synthesis_started_ms: u64,
    load_ms: u64,
    synthesis_ms: u64,
    frames: u64,
}

impl SayPresenter {
    /// Names the file the audio lands in, so the summary can say where it went.
    ///
    /// The event stream never carries this — a machine consumer passed the path in and knows it —
    /// but a person reading four lines of output should not have to look back at their own command
    /// to find out what was written.
    #[must_use]
    pub fn writing_to(destination: Option<String>) -> Self {
        Self {
            destination,
            ..Self::default()
        }
    }

    /// Feed one lifecycle event. Unknown events are ignored, so a schema addition cannot break
    /// human output — it simply will not be narrated until someone teaches this about it.
    ///
    /// # Errors
    ///
    /// When the sink cannot be written.
    pub fn event(&mut self, event: &serde_json::Value, out: &mut dyn Write) -> std::io::Result<()> {
        let kind = event.get("event").and_then(serde_json::Value::as_str);
        let elapsed = event
            .get("elapsed_ms")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        match kind {
            Some("run_start") => {
                let voice = event
                    .get("voice")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("default");
                writeln!(out, "{} {}", detail("voice"), emphasis(voice))?;
            }
            Some("stage") => {
                let name = event
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                let state = event
                    .get("state")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                match (name, state) {
                    ("load", "begin") => self.load_started_ms = elapsed,
                    ("load", "end") => {
                        self.load_ms = elapsed.saturating_sub(self.load_started_ms);
                        ok(
                            out,
                            &format!("model loaded {}", detail(&secs(self.load_ms))),
                        )?;
                    }
                    ("synthesis", "begin") => self.synthesis_started_ms = elapsed,
                    ("synthesis", "end") => {
                        self.synthesis_ms = elapsed.saturating_sub(self.synthesis_started_ms);
                    }
                    _ => {}
                }
            }
            // Per-packet chunks are the machine contract's business. A person watching a file get
            // written does not want one line per 320 ms of audio.
            Some("audio_chunk") => {}
            Some("run_complete") => {
                let audio_ms = event
                    .get("duration_ms")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                self.frames = event
                    .get("frames")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                // `--check` completes without synthesizing anything; announcing "0 frames" there
                // would describe work that was never attempted.
                if self.frames > 0 {
                    let where_to = self
                        .destination
                        .as_deref()
                        .map_or_else(String::new, |path| format!(" → {}", emphasis(path)));
                    ok(
                        out,
                        &format!(
                            "synthesized {} frames {}{where_to}",
                            self.frames,
                            detail(&secs(self.synthesis_ms))
                        ),
                    )?;
                }
                let mut tail = format!("{} of audio in {} total", secs(audio_ms), secs(elapsed));
                if let Some(ttfa) = event.get("ttfa_ms").and_then(serde_json::Value::as_u64) {
                    tail.push_str(&format!(" · first audio {ttfa} ms"));
                }
                // Report synthesis against real time, not the whole run: model load is a fixed
                // one-off, and folding it in makes a short sentence look slow for a reason that
                // has nothing to do with how fast the engine generates.
                if audio_ms > 0 && self.synthesis_ms > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let ratio = audio_ms as f64 / self.synthesis_ms as f64;
                    tail.push_str(&format!(" · synthesis {ratio:.2}× real time"));
                }
                writeln!(out, "  {}", detail(&tail))?;
            }
            _ => {}
        }
        Ok(())
    }
}

/// Milliseconds as a short human duration.
fn secs(ms: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let seconds = ms as f64 / 1000.0;
    if seconds < 10.0 {
        format!("{seconds:.2} s")
    } else {
        format!("{seconds:.1} s")
    }
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
