//! The system-encoder boundary for MP4 output.
//!
//! Rendering is pure Rust up to and including the Y4M stream; H.264+AAC in
//! MP4 is an encoder post-step under the same contract as `ftts say`'s
//! `.m4a` path: the first available system encoder does it, and no encoder
//! found is a refusal that names the native alternative (a `.y4m` output
//! path), never a silent format switch.

use std::io::Write;
use std::path::Path;
use std::process::{Child, Command, Stdio};

/// A running encoder accepting Y4M on stdin.
///
/// Dropping the sink without calling [`EncoderSink::finish`] (the render
/// error path) kills and reaps the child so no half-fed ffmpeg lingers.
pub struct EncoderSink {
    child: Option<Child>,
}

impl Drop for EncoderSink {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub enum Encoder {
    /// Pipe Y4M frames to ffmpeg, muxing `audio` and writing `output`.
    Ffmpeg,
}

/// Probe for a usable encoder without running it.
pub fn available() -> Option<Encoder> {
    // `-version` exits 0 fast; NotFound means the tool is absent. ffmpeg is
    // the one tool that can take Y4M + WAV to H.264+AAC MP4 in a single
    // pass; afconvert is audio-only so it cannot serve here.
    match Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(status) if status.success() => Some(Encoder::Ffmpeg),
        _ => None,
    }
}

/// The refusal text when no encoder exists, naming the native way out.
pub fn refusal() -> String {
    "no system video encoder found (looked for: ffmpeg); \
     either install ffmpeg, or pass a .y4m output path to render natively \
     without any encoder (the audio stays alongside as a .wav)"
        .to_owned()
}

impl EncoderSink {
    /// Spawn ffmpeg reading Y4M from stdin and `audio` as the second input.
    ///
    /// Encoding parameters target X/Twitter's published limits: H.264 high
    /// profile level 4.2, yuv420p, ≤25 Mb/s, AAC 192k, faststart.
    pub fn spawn(audio: &Path, output: &Path) -> Result<Self, String> {
        let child = Command::new("ffmpeg")
            .args([
                "-y",
                "-loglevel",
                "error",
                "-f",
                "yuv4mpegpipe",
                "-i",
                "pipe:0",
                "-i",
            ])
            .arg(audio)
            .args([
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "19",
                "-profile:v",
                "high",
                "-level:v",
                "4.2",
                "-pix_fmt",
                "yuv420p",
                "-maxrate",
                "20M",
                "-bufsize",
                "25M",
                "-colorspace",
                "bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
            ])
            .arg(output)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .map_err(|error| format!("cannot start ffmpeg: {error}"))?;
        Ok(Self { child: Some(child) })
    }

    pub fn stdin(&mut self) -> Result<&mut (dyn Write + Send), String> {
        self.child
            .as_mut()
            .and_then(|child| child.stdin.as_mut())
            .map(|stdin| stdin as &mut (dyn Write + Send))
            .ok_or_else(|| "ffmpeg stdin unavailable".to_owned())
    }

    /// Close stdin and wait for the encoder to finish.
    pub fn finish(mut self) -> Result<(), String> {
        let mut child = self
            .child
            .take()
            .ok_or_else(|| "ffmpeg already finished".to_owned())?;
        drop(child.stdin.take());
        let status = child
            .wait()
            .map_err(|error| format!("waiting for ffmpeg: {error}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("ffmpeg exited with {status}"))
        }
    }
}
