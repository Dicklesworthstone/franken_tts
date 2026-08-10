//! The audio output tail: decoded PCM to a playable file.
//!
//! The codec hands back `f32` samples in `[-1, 1]`; everything downstream of that is this module.
//! It is deliberately small and pure-Rust (AGENTS.md toolchain: no libsndfile, no FFmpeg FFI), and
//! it owns exactly two conversions that are easy to get subtly wrong:
//!
//! 1. **f32 to 16-bit PCM.** Clamping happens *before* scaling. A sample of `1.2` scaled first and
//!    clamped second wraps to a large negative value — an audible click that a duration check and a
//!    byte-count check both call success.
//! 2. **The RIFF header's two length fields.** They must describe the bytes actually present. A
//!    header claiming more data than the file holds is corrupt, and players disagree about how to
//!    fail on it, so [`WavWriter::finish`] patches both from the real sample count. The same rule
//!    is what makes a disk-full partial file still playable.
//!
//! # Format
//!
//! 16-bit signed PCM, mono, 24 kHz — [`SAMPLE_RATE_HZ`], the codec's native rate (plan §2.7:
//! 24 kHz, 1,920 samples per 80 ms frame). Nothing here resamples: emitting the codec's own rate
//! keeps the output bit-faithful to what the model produced, and resampling is a separate,
//! explicitly-requested operation.

use std::io::{self, Seek, SeekFrom, Write};

/// The codec's native output rate.
pub const SAMPLE_RATE_HZ: u32 = 24_000;

/// Samples the codec emits per 80 ms frame: `SAMPLE_RATE_HZ / 12.5`.
pub const SAMPLES_PER_FRAME: usize = 1_920;

/// Channel count. The model is mono; stereo would be a fabrication.
pub const CHANNELS: u16 = 1;

/// Bits per sample in the emitted WAV.
pub const BITS_PER_SAMPLE: u16 = 16;

/// Bytes in a canonical 44-byte RIFF/WAVE header for uncompressed PCM.
pub const WAV_HEADER_BYTES: usize = 44;

/// Samples produced by `frames` codec frames.
#[must_use]
pub const fn samples_for_frames(frames: usize) -> usize {
    frames * SAMPLES_PER_FRAME
}

/// Convert one `f32` sample in `[-1, 1]` to signed 16-bit PCM.
///
/// Clamp first, then scale. The reverse order wraps on overshoot: `1.2 * 32767.0` is `39320`,
/// which truncates to a large negative `i16` and produces a click exactly where the audio was
/// loudest. Non-finite input becomes silence rather than an arbitrary bit pattern — a NaN reaching
/// here is a bug upstream, and the runtime-health `NonFinite` seam is what reports it; this
/// conversion's job is to not turn it into noise.
///
/// The scale is `32767.0` (not `32768.0`) so that `+1.0` maps to `i16::MAX` exactly and the
/// mapping stays symmetric about zero.
#[must_use]
pub fn sample_to_i16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    let clamped = sample.clamp(-1.0, 1.0);
    // `round()` gives round-half-away-from-zero, matching the reference converters; truncation
    // would bias every sample toward zero and quietly lower the output level.
    (clamped * 32_767.0).round() as i16
}

/// Convert a decoded `f32` buffer to 16-bit PCM.
#[must_use]
pub fn pcm_f32_to_i16(pcm: &[f32]) -> Vec<i16> {
    pcm.iter().copied().map(sample_to_i16).collect()
}

/// Mean square energy of a PCM buffer, in `[0, 1]`.
///
/// Used to answer "did we actually produce audio?". A silent result is a failure that every
/// byte-count and duration check reports as success, so energy is the check that distinguishes
/// them. Returns `0.0` for an empty buffer rather than dividing by zero.
#[must_use]
pub fn mean_square_energy(pcm: &[f32]) -> f64 {
    if pcm.is_empty() {
        return 0.0;
    }
    let total: f64 = pcm.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
    total / pcm.len() as f64
}

/// Build a 44-byte RIFF/WAVE header for `sample_count` mono 16-bit samples.
///
/// Both length fields are derived from `sample_count` so they can never disagree with each other
/// or with the payload.
#[must_use]
pub fn wav_header(sample_rate: u32, sample_count: usize) -> [u8; WAV_HEADER_BYTES] {
    let data_bytes = (sample_count * usize::from(BITS_PER_SAMPLE / 8)) as u32;
    let byte_rate = sample_rate * u32::from(CHANNELS) * u32::from(BITS_PER_SAMPLE / 8);
    let block_align = CHANNELS * (BITS_PER_SAMPLE / 8);

    let mut header = [0u8; WAV_HEADER_BYTES];
    header[0..4].copy_from_slice(b"RIFF");
    // RIFF size counts everything after this field: 36 header bytes plus the payload.
    header[4..8].copy_from_slice(&(36u32.saturating_add(data_bytes)).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    header[20..22].copy_from_slice(&1u16.to_le_bytes()); // format 1 = uncompressed PCM
    header[22..24].copy_from_slice(&CHANNELS.to_le_bytes());
    header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&block_align.to_le_bytes());
    header[34..36].copy_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    header
}

/// Encode a whole PCM buffer as a WAV file in memory.
///
/// For the offline path, where the sample count is known before writing. Streaming synthesis uses
/// [`WavWriter`], which does not need to know it up front.
#[must_use]
pub fn encode_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(WAV_HEADER_BYTES + pcm.len() * 2);
    bytes.extend_from_slice(&wav_header(sample_rate, pcm.len()));
    for sample in pcm {
        bytes.extend_from_slice(&sample_to_i16(*sample).to_le_bytes());
    }
    bytes
}

/// Samples of low-level, high-frequency junk at the very end of an utterance.
///
/// # The artifact
///
/// The model reliably ends an utterance with a short burst of broadband noise after speech has
/// decayed and before its own trailing silence. Measured on real synthesis at 24 kHz, the burst
/// sits around RMS 20-90 against speech at 500-1200, and its first-difference energy ratio (a cheap
/// high-frequency proxy) runs 0.3-0.9 where speech runs 0.02-0.10. It is what a listener hears as
/// "a little noise right at the end".
///
/// It is NOT a quantization artifact: an interleaved A/B against the f32 reference route on the
/// same text and seed showed the reference producing MORE of it (sustained ~90 ms at ratio up to
/// 0.94) than the int8 route. So it comes from the model's own final frames, and no kernel change
/// removes it.
///
/// # What this does, and what it deliberately does not
///
/// Returns how many samples to drop from the end. The rule is conjunctive on purpose, because each
/// condition alone has a false positive that would eat real audio:
///
/// * **quiet** relative to this utterance's own speech level, so a loud ending is never touched;
/// * **high-frequency dominated**, so a soft voiced ending (a low hum, a sustained vowel) is never
///   touched — those are the tonal opposite of this artifact;
/// * **contiguous from the end**, so noise in the middle of a sentence is left alone entirely.
///
/// Trailing pure silence is skipped before analysis and kept afterwards: it is genuine model output
/// (measured runs carry 0-71 ms of it), and trimming it would change utterance timing.
///
/// Returns 0 whenever the input is too short to judge, which keeps every caller total.
#[must_use]
pub fn trailing_noise_samples(pcm: &[f32], sample_rate: u32) -> usize {
    /// Analysis window. 10 ms is short enough to localize the burst and long enough that a
    /// single glottal pulse does not dominate the statistic.
    const WINDOW_MILLIS: usize = 10;
    /// Never remove more than this. Measured artifact runs: 30 ms (nz5), 80 ms (f32 reference),
    /// and just over 200 ms on the preset voice samples. 250 ms covers the observed range while
    /// still bounding the damage if the rule ever misfires.
    const MAX_TRIM_MILLIS: usize = 250;
    /// A window must be under this fraction of the utterance's speech level to be a candidate.
    /// Artifact windows measured at most 0.08 of the utterance peak, so 0.15 leaves headroom
    /// without reaching the level a real final consonant occupies.
    const QUIET_FRACTION: f32 = 0.15;
    /// First-difference energy ratio above which a window is high-frequency dominated. Voiced
    /// speech measured 0.005-0.10; the artifact measured 0.31-1.92. 0.25 sits in the empty gap
    /// between those two populations.
    const HF_RATIO: f32 = 0.25;
    /// The whole trimmed run must also be this quiet on average. A sustained final fricative is
    /// high-frequency too, and this is what separates it from the artifact: /s/ carries real
    /// level, the artifact does not.
    const RUN_MEAN_FRACTION: f32 = 0.10;

    let window = (sample_rate as usize).saturating_mul(WINDOW_MILLIS) / 1000;
    if window < 2 || pcm.len() < window * 4 {
        return 0;
    }

    // Trailing exact silence is model output, not artifact; keep it, and analyze what precedes it.
    let voiced_end = pcm
        .iter()
        .rposition(|sample| *sample != 0.0)
        .map_or(0, |i| i + 1);
    if voiced_end < window * 4 {
        return 0;
    }

    let rms = |seg: &[f32]| -> f32 {
        if seg.is_empty() {
            return 0.0;
        }
        (seg.iter().map(|s| s * s).sum::<f32>() / seg.len() as f32).sqrt()
    };
    // First-difference energy over signal energy: high for broadband noise, low for voiced speech.
    let hf = |seg: &[f32]| -> f32 {
        let energy: f32 = seg.iter().map(|s| s * s).sum();
        if energy <= f32::MIN_POSITIVE {
            return 0.0;
        }
        let diff: f32 = seg.windows(2).map(|p| (p[1] - p[0]) * (p[1] - p[0])).sum();
        diff / energy
    };

    // The utterance's own speech level, taken as the loudest window so the threshold scales with
    // the recording rather than assuming an absolute amplitude.
    let speech_level = pcm[..voiced_end]
        .chunks(window)
        .map(|chunk| rms(chunk))
        .fold(0.0_f32, f32::max);
    if speech_level <= 0.0 {
        return 0;
    }
    let quiet_ceiling = speech_level * QUIET_FRACTION;

    let max_trim = (sample_rate as usize).saturating_mul(MAX_TRIM_MILLIS) / 1000;
    let mut trimmed = 0_usize;
    let mut end = voiced_end;
    while end >= window && trimmed + window <= max_trim {
        let start = end - window;
        let segment = &pcm[start..end];
        if rms(segment) < quiet_ceiling && hf(segment) > HF_RATIO {
            trimmed += window;
            end = start;
        } else {
            break;
        }
    }

    // Final guard on the run as a whole. Each window passing individually is not enough: a
    // sustained final fricative is quiet-ish AND high-frequency window by window, and would walk
    // the loop above backwards through real speech. The artifact's run mean sits far below a
    // fricative's, so this is the condition that separates them.
    if trimmed > 0 {
        let run = &pcm[voiced_end - trimmed..voiced_end];
        if rms(run) >= speech_level * RUN_MEAN_FRACTION {
            return 0;
        }
    }
    trimmed
}

/// A streaming WAV writer that finalises a correct header.
///
/// Streaming synthesis does not know the sample count until the run ends, so a provisional header
/// is written first and patched by [`WavWriter::finish`]. That seek-back is why the sink must be
/// `Seek`: an unseekable sink cannot carry a correct length, and silently emitting a wrong one is
/// the corruption this type exists to prevent.
///
/// If the run is cut short — cancellation, a full disk — calling `finish` still yields a valid
/// file describing the samples that made it, which is the partial-output promise in plan §9.6.
/// Samples held back when tail trimming is armed: the detector's own ceiling, so the buffer always
/// holds every sample the trim could possibly want.
const TAIL_HOLDBACK_SAMPLES: usize = SAMPLE_RATE_HZ as usize * 250 / 1000;

pub struct WavWriter<W: Write + Seek> {
    /// `None` once [`WavWriter::finish`] has handed the sink back.
    ///
    /// An `Option` rather than a bare `W` because a type with a `Drop` impl cannot be moved out
    /// of, and `ftts-core` forbids the `unsafe` that `ManuallyDrop` would need. The `None` state
    /// also tells `Drop` that finalisation already happened, so it is not attempted twice.
    sink: Option<W>,
    sample_rate: u32,
    samples_written: usize,
    /// Samples withheld from the file so the end-of-utterance trim can still see them.
    ///
    /// Writing is delayed by at most `TAIL_HOLDBACK_SAMPLES`, which is why this is opt-in: for a
    /// file that delay is invisible, but on `--stream raw` it would add latency to a path whose
    /// whole contract is time-to-first-audio. `None` means trimming is off and every sample goes
    /// straight through.
    holdback: Option<Vec<f32>>,
}

impl<W: Write + Seek> WavWriter<W> {
    /// Begin a file, writing a provisional header.
    ///
    /// # Errors
    ///
    /// If the provisional header cannot be written.
    pub fn new(mut sink: W, sample_rate: u32) -> io::Result<Self> {
        sink.write_all(&wav_header(sample_rate, 0))?;
        Ok(Self {
            sink: Some(sink),
            sample_rate,
            samples_written: 0,
            holdback: None,
        })
    }

    /// As [`WavWriter::new`], but drops the model's end-of-utterance noise burst.
    ///
    /// See [`trailing_noise_samples`] for what is removed and why it is safe. The cost is that the
    /// last quarter second is held in memory until [`WavWriter::finish`], so this is for file
    /// output only; a raw PCM stream must keep its latency and stays untrimmed.
    ///
    /// # Errors
    ///
    /// If the provisional header cannot be written.
    pub fn new_trimming_tail(sink: W, sample_rate: u32) -> io::Result<Self> {
        let mut writer = Self::new(sink, sample_rate)?;
        writer.holdback = Some(Vec::with_capacity(TAIL_HOLDBACK_SAMPLES * 2));
        Ok(writer)
    }

    /// Append one packet of decoded `f32` samples.
    ///
    /// # Errors
    ///
    /// If the sink rejects the write. The count of samples already accepted stays accurate, so a
    /// later [`WavWriter::finish`] still describes the file truthfully.
    pub fn write_samples(&mut self, pcm: &[f32]) -> io::Result<()> {
        // With trimming armed, keep the newest TAIL_HOLDBACK_SAMPLES back and emit only what has
        // aged out. Those held samples are the only ones the trim can ever remove, so nothing that
        // reaches the file here can need taking back.
        if self.holdback.is_some() {
            let mut pending = self.holdback.take().unwrap_or_default();
            pending.extend_from_slice(pcm);
            let releasable = pending.len().saturating_sub(TAIL_HOLDBACK_SAMPLES);
            let released: Vec<f32> = pending.drain(..releasable).collect();
            self.holdback = Some(pending);
            if released.is_empty() {
                return Ok(());
            }
            return self.write_through(&released);
        }
        self.write_through(pcm)
    }

    /// Writes samples straight to the sink, bypassing the hold-back.
    fn write_through(&mut self, pcm: &[f32]) -> io::Result<()> {
        // Buffer the packet so one short write cannot leave half a sample in the file, which
        // would desynchronise every subsequent frame by one byte.
        let mut bytes = Vec::with_capacity(pcm.len() * 2);
        for sample in pcm {
            bytes.extend_from_slice(&sample_to_i16(*sample).to_le_bytes());
        }
        let sink = self
            .sink
            .as_mut()
            .ok_or_else(|| io::Error::other("WavWriter already finished"))?;
        sink.write_all(&bytes)?;
        self.samples_written += pcm.len();
        Ok(())
    }

    /// Samples accepted so far.
    #[must_use]
    pub const fn samples_written(&self) -> usize {
        self.samples_written
    }

    /// Duration of the audio written so far, in milliseconds.
    #[must_use]
    pub const fn duration_millis(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        (self.samples_written as u64) * 1000 / (self.sample_rate as u64)
    }

    /// Patch the header to the real length and flush.
    ///
    /// # Errors
    ///
    /// If seeking back to the header, rewriting it, or flushing fails.
    pub fn finish(mut self) -> io::Result<W> {
        // Release the held tail, minus whatever the detector identifies as the model's end-of-
        // utterance noise. Done before the header is patched so the length describes what landed.
        if let Some(pending) = self.holdback.take() {
            let drop = trailing_noise_samples(&pending, self.sample_rate);
            let keep = pending.len().saturating_sub(drop);
            if keep > 0 {
                self.write_through(&pending[..keep])?;
            }
        }
        self.finalize_header()?;
        self.sink
            .take()
            .ok_or_else(|| io::Error::other("WavWriter already finished"))
    }

    fn finalize_header(&mut self) -> io::Result<()> {
        let header = wav_header(self.sample_rate, self.samples_written);
        let Some(sink) = self.sink.as_mut() else {
            return Ok(());
        };
        sink.seek(SeekFrom::Start(0))?;
        sink.write_all(&header)?;
        sink.seek(SeekFrom::End(0))?;
        sink.flush()
    }
}

impl<W: Write + Seek> Drop for WavWriter<W> {
    /// Best-effort finalisation for a writer dropped without [`WavWriter::finish`].
    ///
    /// A dropped writer means an abnormal end — a panic, an early return, a cancelled run. Leaving
    /// the provisional zero-length header would make the file claim it contains no audio while
    /// holding a megabyte of it. The error is deliberately swallowed because `Drop` cannot report,
    /// which is exactly why `finish` exists and should be called explicitly.
    fn drop(&mut self) {
        // `finish` takes the sink, so a still-present sink means an abnormal end.
        if self.sink.is_some() {
            let _ = self.finalize_header();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn frame_count_maps_to_the_codec_sample_rate() {
        // 24 kHz at 12.5 frames/s is 1,920 samples per 80 ms frame; the two constants must agree
        // or every duration we report is wrong.
        assert_eq!(SAMPLES_PER_FRAME * 25, SAMPLE_RATE_HZ as usize * 2);
        assert_eq!(samples_for_frames(1), 1_920);
        assert_eq!(samples_for_frames(125), SAMPLE_RATE_HZ as usize * 10);
    }

    #[test]
    fn conversion_clamps_before_scaling() {
        // The bug this prevents: scaling 1.2 first gives 39_320, which wraps to a large negative
        // i16 — a click at the loudest moment, which every byte-count check calls success.
        assert_eq!(sample_to_i16(1.2), i16::MAX);
        assert_eq!(sample_to_i16(-1.2), -i16::MAX);
        assert_eq!(sample_to_i16(1.0), i16::MAX);
        assert_eq!(sample_to_i16(-1.0), -i16::MAX);
        assert_eq!(sample_to_i16(0.0), 0);
    }

    #[test]
    fn conversion_rounds_rather_than_truncates() {
        // Truncation biases every sample toward zero and quietly lowers the output level.
        let half_step = 0.5 / 32_767.0;
        assert_eq!(sample_to_i16(half_step), 1);
        assert_eq!(sample_to_i16(-half_step), -1);
    }

    #[test]
    fn non_finite_becomes_silence_not_noise() {
        assert_eq!(sample_to_i16(f32::NAN), 0);
        assert_eq!(sample_to_i16(f32::INFINITY), 0);
        assert_eq!(sample_to_i16(f32::NEG_INFINITY), 0);
    }

    #[test]
    fn energy_separates_silence_from_audio() {
        assert_eq!(mean_square_energy(&[]), 0.0);
        assert_eq!(mean_square_energy(&[0.0; 64]), 0.0);
        assert!(mean_square_energy(&[0.5; 64]) > 0.2);
    }

    #[test]
    fn the_header_describes_exactly_the_payload() {
        let pcm = vec![0.25f32; 1_920];
        let wav = encode_wav(&pcm, SAMPLE_RATE_HZ);
        assert_eq!(wav.len(), WAV_HEADER_BYTES + pcm.len() * 2);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");

        let declared_data = u32::from_le_bytes(wav[40..44].try_into().expect("data size"));
        let actual_data = (wav.len() - WAV_HEADER_BYTES) as u32;
        assert_eq!(
            declared_data, actual_data,
            "data size must match the payload"
        );

        let declared_riff = u32::from_le_bytes(wav[4..8].try_into().expect("riff size"));
        assert_eq!(declared_riff, 36 + actual_data, "RIFF size must agree");

        let rate = u32::from_le_bytes(wav[24..28].try_into().expect("rate"));
        assert_eq!(rate, SAMPLE_RATE_HZ);
        let channels = u16::from_le_bytes(wav[22..24].try_into().expect("channels"));
        assert_eq!(channels, 1, "the model is mono");
        let bits = u16::from_le_bytes(wav[34..36].try_into().expect("bits"));
        assert_eq!(bits, 16);
    }

    #[test]
    fn a_streamed_file_is_byte_identical_to_the_offline_encoding() {
        // Streaming and offline must produce the same file for the same samples, or "streaming ==
        // batch" fails at the very last stage of the pipeline.
        let pcm: Vec<f32> = (0..1_920)
            .map(|i| (i as f32 / 1_920.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();

        let mut writer = WavWriter::new(Cursor::new(Vec::new()), SAMPLE_RATE_HZ).expect("header");
        for packet in pcm.chunks(480) {
            writer.write_samples(packet).expect("packet");
        }
        assert_eq!(writer.samples_written(), pcm.len());
        assert_eq!(writer.duration_millis(), 80);
        let streamed = writer.finish().expect("finish").into_inner();

        assert_eq!(streamed, encode_wav(&pcm, SAMPLE_RATE_HZ));
    }

    #[test]
    fn a_truncated_run_still_finalises_a_valid_header() {
        // The partial-output promise: a run cut short produces a file describing the samples that
        // actually landed, not the zero-length provisional header.
        let mut writer = WavWriter::new(Cursor::new(Vec::new()), SAMPLE_RATE_HZ).expect("header");
        writer.write_samples(&[0.5f32; 960]).expect("packet");
        let file = writer.finish().expect("finish").into_inner();

        let declared = u32::from_le_bytes(file[40..44].try_into().expect("data size"));
        assert_eq!(declared, 960 * 2);
        assert_eq!(file.len(), WAV_HEADER_BYTES + 960 * 2);
    }

    #[test]
    fn dropping_without_finish_still_patches_the_length() {
        // A panic or early return must not leave a file claiming it holds no audio.
        let mut sink = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut sink, SAMPLE_RATE_HZ).expect("header");
            writer.write_samples(&[0.25f32; 128]).expect("packet");
            // dropped here without finish()
        }
        let file = sink.into_inner();
        let declared = u32::from_le_bytes(file[40..44].try_into().expect("data size"));
        assert_eq!(declared, 128 * 2, "Drop must finalise the length");
    }

    #[test]
    fn an_empty_run_is_a_valid_zero_length_wav() {
        let wav = encode_wav(&[], SAMPLE_RATE_HZ);
        assert_eq!(wav.len(), WAV_HEADER_BYTES);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().expect("size")), 0);
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().expect("riff")), 36);
    }
}

#[cfg(test)]
mod tail_tests {
    use super::*;

    const SR: u32 = 24_000;

    fn noise(len: usize, amplitude: f32) -> Vec<f32> {
        // Alternating sign: maximal first-difference energy, i.e. the broadband shape the
        // artifact has.
        (0..len)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    fn tone(len: usize, amplitude: f32) -> Vec<f32> {
        // 200 Hz: voiced, low first-difference energy.
        (0..len)
            .map(|i| amplitude * (i as f32 * 2.0 * std::f32::consts::PI * 200.0 / SR as f32).sin())
            .collect()
    }

    #[test]
    fn a_quiet_high_frequency_tail_is_trimmed() {
        let mut pcm = tone(SR as usize / 2, 0.5);
        pcm.extend(noise(SR as usize * 40 / 1000, 0.02));
        let trimmed = trailing_noise_samples(&pcm, SR);
        assert!(trimmed > 0, "the artifact shape must be detected");
        assert!(
            trimmed <= SR as usize * 40 / 1000 + SR as usize / 100,
            "trim {trimmed} reached past the noise into speech"
        );
    }

    #[test]
    fn a_loud_ending_is_never_trimmed() {
        // Speech that simply stops at full level: nothing to remove, however abrupt.
        let pcm = tone(SR as usize / 2, 0.5);
        assert_eq!(trailing_noise_samples(&pcm, SR), 0);
    }

    #[test]
    fn a_quiet_voiced_ending_is_never_trimmed() {
        // The dangerous false positive: a soft sustained vowel is quiet but TONAL, so the
        // high-frequency condition must save it.
        let mut pcm = tone(SR as usize / 2, 0.5);
        pcm.extend(tone(SR as usize * 60 / 1000, 0.02));
        assert_eq!(
            trailing_noise_samples(&pcm, SR),
            0,
            "a soft voiced ending must survive"
        );
    }

    #[test]
    fn trailing_silence_is_preserved_and_the_noise_before_it_is_found() {
        let mut pcm = tone(SR as usize / 2, 0.5);
        pcm.extend(noise(SR as usize * 30 / 1000, 0.02));
        let silence = SR as usize * 50 / 1000;
        pcm.extend(std::iter::repeat_n(0.0_f32, silence));
        let trimmed = trailing_noise_samples(&pcm, SR);
        assert!(
            trimmed > 0,
            "silence after the burst must not hide the burst"
        );
        // The reported count covers only the noise; the caller keeps the silence.
        assert!(trimmed <= SR as usize * 40 / 1000);
    }

    #[test]
    fn noise_in_the_middle_is_left_alone() {
        let mut pcm = tone(SR as usize / 4, 0.5);
        pcm.extend(noise(SR as usize * 30 / 1000, 0.02));
        pcm.extend(tone(SR as usize / 4, 0.5));
        assert_eq!(
            trailing_noise_samples(&pcm, SR),
            0,
            "only a tail contiguous with the end is in scope"
        );
    }

    #[test]
    fn short_and_empty_inputs_are_total() {
        assert_eq!(trailing_noise_samples(&[], SR), 0);
        assert_eq!(trailing_noise_samples(&[0.1; 16], SR), 0);
        assert_eq!(trailing_noise_samples(&[0.0; 4096], SR), 0);
        assert_eq!(trailing_noise_samples(&tone(4096, 0.5), 0), 0);
    }
}
