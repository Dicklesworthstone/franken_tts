//! ftts-video — the branded renderer behind `ftts make-video`.
//!
//! Takes synthesized speech (a WAV) and produces a share-ready 1920×1080
//! video: a Ken Burns pass over the FrankenTTS illustration, a live waveform
//! lane, and the project's overlay text with the voice's name. Every frame
//! is rendered by memory-safe Rust in this crate; the only external step is
//! the optional H.264+AAC encode through the first available system encoder
//! (the same contract as `ftts say`'s `.m4a` path). Without an encoder the
//! renderer still ships its native pair: YUV4MPEG2 video + WAV audio.
//!
//! Design lineage: the glyph pipeline consumes `fmd-font` outlines
//! (franken_markdown's clean-room font subsystem, the same engine
//! franken_manim's Scribe builds on), and the native-output-or-typed-refusal
//! encoder boundary mirrors franken_manim's ffmpeg protocol.

pub mod encode;
pub mod kenburns;
pub mod overlay;
pub mod qoi;
pub mod raster;
pub mod wav;
pub mod waveform;
pub mod yuv;

use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

/// The illustration from the repository README, compiled in as QOI.
const ILLUSTRATION_QOI: &[u8] = include_bytes!("../assets/illustration.qoi");

pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;
pub const FPS: u32 = 30;

/// What to render.
pub struct VideoRequest<'a> {
    /// The finished speech audio (PCM WAV path).
    pub audio: &'a Path,
    /// Destination: `.mp4` (system encoder) or `.y4m` (native).
    pub output: &'a Path,
    /// Voice name shown in the pill, e.g. `Matt` or a custom voice's stem.
    pub voice_label: &'a str,
}

/// Rendering progress, reported once per percent step.
pub struct Progress {
    pub frame: usize,
    pub total_frames: usize,
}

enum Sink {
    Encoder(encode::EncoderSink),
    Y4m(BufWriter<fs::File>),
}

/// Everything needed to draw any frame of the branded video, host-agnostic.
///
/// The CLI drives this through [`render`] into Y4M/ffmpeg; the iOS app drives it through
/// `ftts-ffi` into `AVAssetWriter`, where the system's hardware H.264 encoder replaces
/// ffmpeg. One implementation, two encoders — the frames are identical by construction.
pub struct FrameRenderer {
    background: qoi::Image,
    overlay: raster::Surface,
    samples: Vec<f32>,
    sample_rate: u32,
    total_frames: usize,
}

// Frames are pure functions of (frame index, immutable renderer state), so hosts may
// render several frames concurrently over one renderer. This assertion keeps that
// property load-bearing: a future field with interior mutability must break the build,
// not the iOS exporter that renders chunks in parallel.
const _: fn() = || {
    fn requires_sync<T: Sync>() {}
    requires_sync::<FrameRenderer>();
};

impl FrameRenderer {
    /// Build a renderer over finished speech PCM (mono, any positive sample rate).
    ///
    /// # Errors
    ///
    /// When the embedded assets fail to decode or the overlay cannot be built.
    pub fn new(samples: Vec<f32>, sample_rate: u32, voice_label: &str) -> Result<Self, String> {
        if samples.is_empty() || sample_rate == 0 {
            return Err("video needs non-empty audio and a positive sample rate".to_owned());
        }
        let duration = samples.len() as f64 / f64::from(sample_rate);
        let total_frames = (duration * f64::from(FPS)).ceil().max(1.0) as usize;
        Ok(Self {
            background: qoi::decode(ILLUSTRATION_QOI)?,
            overlay: overlay::build(voice_label)?,
            samples,
            sample_rate,
            total_frames,
        })
    }

    #[must_use]
    pub fn total_frames(&self) -> usize {
        self.total_frames
    }

    /// Draw frame `frame` as BGRA32 with a caller-chosen row stride — the layout
    /// CoreVideo pixel buffers want, so the iOS exporter copies rows instead of
    /// swizzling two million pixels per frame in Swift.
    ///
    /// # Panics
    ///
    /// If `stride < WIDTH * 4` or `bgra` is shorter than `stride * HEIGHT`.
    pub fn render_into_bgra(&self, frame: usize, bgra: &mut [u8], stride: usize) {
        assert!(stride >= WIDTH * 4, "stride must cover a BGRA row");
        assert!(bgra.len() >= stride * HEIGHT, "bgra must hold HEIGHT rows");
        let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];
        self.render_into(frame, &mut rgb);
        for (source_row, target_row) in rgb
            .as_chunks::<{ WIDTH * 3 }>()
            .0
            .iter()
            .zip(bgra.chunks_mut(stride))
        {
            for (source, target) in source_row
                .as_chunks::<3>()
                .0
                .iter()
                .zip(target_row.as_chunks_mut::<4>().0.iter_mut())
            {
                *target = [source[2], source[1], source[0], 255];
            }
        }
    }

    /// Draw frame `frame` into `rgb`, which must be `WIDTH * HEIGHT * 3` bytes.
    ///
    /// # Panics
    ///
    /// If `rgb` has the wrong length.
    pub fn render_into(&self, frame: usize, rgb: &mut [u8]) {
        assert_eq!(rgb.len(), WIDTH * HEIGHT * 3, "rgb must be WIDTH*HEIGHT*3");
        let camera = kenburns::Camera::new(&self.background);
        camera.render(frame, self.total_frames, WIDTH, HEIGHT, rgb);
        composite_overlay(rgb, &self.overlay);
        waveform::draw(
            rgb,
            WIDTH,
            HEIGHT,
            &self.samples,
            self.sample_rate,
            FPS,
            frame,
            885,
            135.0,
        );
    }
}

/// Render the video. Calls `progress` as frames complete.
///
/// With a `.y4m` output the WAV is left beside it (same stem) so the pair
/// stays playable; with `.mp4` the audio is muxed in by the encoder.
pub fn render(
    request: &VideoRequest<'_>,
    progress: &mut dyn FnMut(Progress),
) -> Result<(), String> {
    let extension = request
        .output
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let is_mp4 = match extension.as_deref() {
        Some("mp4") => true,
        Some("y4m") => false,
        other => {
            return Err(format!(
                "unsupported video extension {:?}; use .mp4 (system encoder) or .y4m (native)",
                other.unwrap_or("<none>")
            ));
        }
    };

    let audio_bytes = fs::read(request.audio)
        .map_err(|error| format!("cannot read audio {}: {error}", request.audio.display()))?;
    let audio = wav::decode(&audio_bytes)?;
    let renderer = FrameRenderer::new(audio.samples, audio.sample_rate, request.voice_label)?;
    let total_frames = renderer.total_frames();

    let mut sink = if is_mp4 {
        if encode::available().is_none() {
            return Err(encode::refusal());
        }
        Sink::Encoder(encode::EncoderSink::spawn(request.audio, request.output)?)
    } else {
        let file = fs::File::create(request.output)
            .map_err(|error| format!("cannot create {}: {error}", request.output.display()))?;
        Sink::Y4m(BufWriter::new(file))
    };

    {
        let stream: &mut dyn Write = match &mut sink {
            Sink::Encoder(encoder) => encoder.stdin()?,
            Sink::Y4m(file) => file,
        };
        yuv::write_y4m_header(stream, WIDTH, HEIGHT, FPS)
            .map_err(|error| format!("writing video header: {error}"))?;

        let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];
        let mut y_plane = vec![0u8; WIDTH * HEIGHT];
        let mut u_plane = vec![0u8; WIDTH * HEIGHT / 4];
        let mut v_plane = vec![0u8; WIDTH * HEIGHT / 4];

        for frame in 0..total_frames {
            renderer.render_into(frame, &mut rgb);
            yuv::rgb_to_i420(
                &rgb,
                WIDTH,
                HEIGHT,
                &mut y_plane,
                &mut u_plane,
                &mut v_plane,
            );
            yuv::write_y4m_frame(stream, &y_plane, &u_plane, &v_plane)
                .map_err(|error| format!("writing frame {frame}: {error}"))?;
            progress(Progress {
                frame: frame + 1,
                total_frames,
            });
        }
        stream
            .flush()
            .map_err(|error| format!("flushing video stream: {error}"))?;
    }

    match sink {
        Sink::Encoder(encoder) => encoder.finish()?,
        Sink::Y4m(file) => {
            drop(file);
            // Keep the audio playable next to the native video. Comparing
            // canonical paths (not spellings) keeps `fs::copy` from ever
            // truncating the source when both names reach the same file.
            let wav_sibling = request.output.with_extension("wav");
            let same_file = match (
                fs::canonicalize(request.audio),
                fs::canonicalize(&wav_sibling),
            ) {
                (Ok(audio), Ok(sibling)) => audio == sibling,
                _ => false,
            };
            if !same_file {
                fs::copy(request.audio, &wav_sibling).map_err(|error| {
                    format!("cannot place audio at {}: {error}", wav_sibling.display())
                })?;
            }
        }
    }
    Ok(())
}

/// Composite the premultiplied-free (straight alpha) overlay onto RGB24.
fn composite_overlay(rgb: &mut [u8], overlay: &raster::Surface) {
    for (pixel, source) in rgb
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(overlay.rgba.as_chunks::<4>().0.iter())
    {
        // Almost every overlay pixel is fully transparent; skip before any float work.
        if source[3] == 0 {
            continue;
        }
        let alpha = f32::from(source[3]) / 255.0;
        for c in 0..3 {
            let dst = f32::from(pixel[c]);
            let src = f32::from(source[c]);
            pixel[c] = (src * alpha + dst * (1.0 - alpha))
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
}
