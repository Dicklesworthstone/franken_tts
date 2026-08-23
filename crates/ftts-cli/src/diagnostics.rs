//! Enrollment-side signal-quality diagnostics (bead frankentts-p1-audio-diagnostics-8t9).
//!
//! Pure detectors over mono 24 kHz f32 PCM. Unlike the synthesis path these are
//! QUALITY-gated, not exactness-gated: each detector is tuned against synthetic defect
//! fixtures with known ground truth (`tests/diagnostics_fixtures.rs`), and every threshold
//! is a named, documented constant so the listening protocol can retune them without
//! archaeology. Consumers: `ftts enroll` warnings today, the AF-4 attribute→quality loss
//! model later.
//!
//! # What each detector can and cannot claim
//!
//! * VAD is energy-with-hysteresis against an adaptive percentile floor — robust for
//!   single-speaker references, deliberately conservative for two-speaker inputs (which
//!   surface as a low SNR estimate instead).
//! * The SNR estimate compares ACTIVE-frame power against the pause floor; it is an
//!   envelope-domain figure, not a spectral subtraction, and reads ~6 dB pessimistic on
//!   references with breath-heavy pauses.
//! * Music-bed likelihood flags SUSTAINED tonality with low syllabic modulation — the
//!   thing continuation-style cloners reproduce as background singing. A dry solo voice
//!   scores near zero even when pitched.

use rustfft::{num_complex::Complex, FftPlanner};

/// Frame length for all envelope statistics: 30 ms at the pinned 24 kHz rate.
const FRAME_LEN: usize = 720;
/// Hop between analysis frames: 10 ms (66% overlap smooths region edges).
const HOP: usize = 240;
/// A sample whose magnitude reaches this level counts toward hard-clipping fraction.
/// Full-scale digital audio rails at 1.0; real ADCs saturate just below, so 0.985
/// catches both.
const CLIP_LEVEL: f32 = 0.985;
/// Consecutive at-rail samples that distinguish "clipped run" from a legitimate full-scale
/// crest: a 24 kHz sine crest holds the rail for well under one sample; a squared wave or
/// railed ADC holds for many.
const CLIP_RUN_SAMPLES: usize = 3;
/// Frames below this energy quantile define the NOISE FLOOR (pauses, room tone). The 10th
/// percentile survives up to ~35% silence in a reference before the floor starts tracking
/// speech.
const FLOOR_QUANTILE: f64 = 0.10;
/// VAD open threshold, in multiples of the noise-floor RMS: above this a frame is speech.
const VAD_OPEN_RATIO: f64 = 3.16; // +10 dB
/// VAD close threshold (hysteresis): speech REGIONS stay open until energy drops under
/// half the open level, so intra-word dips do not shatter them.
const VAD_CLOSE_RATIO: f64 = 1.0; // 0 dB over floor
/// Spectral flatness under which a frame is TONAL (music beds, sustained vowels). Speech
/// consonants and breaths sit far above; chord beds sit far below.
const TONAL_FLATNESS_MAX: f64 = 0.02;

fn frame_rms(pcm: &[f32], start: usize) -> f64 {
    let end = (start + FRAME_LEN).min(pcm.len());
    let len = (end - start) as f64;
    let sum: f64 = pcm[start..end]
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum();
    (sum / len).sqrt()
}

fn percentile(mut values: Vec<f64>, quantile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((values.len() - 1) as f64 * quantile).round() as usize;
    values[index]
}

fn frame_energies(pcm: &[f32]) -> Vec<f64> {
    let mut energies = Vec::new();
    let mut start = 0;
    while start + FRAME_LEN <= pcm.len() {
        energies.push(frame_rms(pcm, start));
        start += HOP;
    }
    energies
}

/// Voice-activity regions as `(start_sample, end_sample)` pairs, exclusive end.
///
/// Hysteresis: a region OPENS when frame energy crosses [`VAD_OPEN_RATIO`] × floor and
/// CLOSES only after it falls under [`VAD_CLOSE_RATIO`] × floor, which keeps plosive gaps
/// from shattering words.
#[must_use]
pub fn voice_activity_regions(pcm: &[f32]) -> Vec<(usize, usize)> {
    let energies = frame_energies(pcm);
    if energies.is_empty() {
        return Vec::new();
    }
    let floor = percentile(energies.clone(), FLOOR_QUANTILE);
    let open = floor * VAD_OPEN_RATIO;
    let close = floor * VAD_CLOSE_RATIO;

    let mut regions = Vec::new();
    let mut region_start: Option<usize> = None;
    for (index, &energy) in energies.iter().enumerate() {
        match (region_start, energy >= open, energy >= close) {
            (None, true, _) => region_start = Some(index * HOP),
            (Some(_), _, false) => {
                if let Some(begin) = region_start.take() {
                    regions.push((begin, index * HOP));
                }
            }
            _ => {}
        }
    }
    if let Some(begin) = region_start.take() {
        regions.push((begin, pcm.len()));
    }
    regions
}

/// Hard-clipping diagnostics: the fraction of samples at or over [`CLIP_LEVEL`], the
/// longest consecutive at-rail run in samples, and an estimated true-peak overshoot in dB
/// over the digital peak (Catmull-Rom interpolation ×4 around local maxima — inter-sample
/// peaks are what actual DACs reconstruct, not the stored grid).
#[must_use]
pub fn clipping_diagnostics(pcm: &[f32]) -> (f64, usize, f64) {
    if pcm.is_empty() {
        return (0.0, 0, 0.0);
    }
    let at_rail: Vec<bool> = pcm.iter().map(|&value| value.abs() >= CLIP_LEVEL).collect();
    let clipped = at_rail.iter().filter(|&&rail| rail).count();
    let fraction = f64::from(clipped as u32) / pcm.len() as f64;

    let mut longest_run = 0_usize;
    let mut run = 0_usize;
    for rail in &at_rail {
        if *rail {
            run += 1;
            longest_run = longest_run.max(run);
        } else {
            run = 0;
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "loop indices over a 4-step interpolation; exactness is irrelevant here"
    )]
    let digital_peak = pcm.iter().fold(0.0_f32, |max, &value| max.max(value.abs()));
    let mut true_peak = f64::from(digital_peak);
    for window in pcm.windows(4) {
        let [a, b, c, d] = window else {
            continue;
        };
        if b.abs() < c.abs() || c.abs() < CLIP_LEVEL * 0.5 {
            continue;
        }
        for step in 1..4_u32 {
            let t = step as f32 / 4.0;
            // Catmull-Rom through (a, b, c, d) evaluated between b and c.
            let interpolated = 0.5
                * ((2.0 * c)
                    + (-a + c) * t
                    + (2.0 * a - 5.0 * b + 4.0 * c - d) * t * t
                    + (-a + 3.0 * b - 3.0 * c + d) * t * t * t);
            true_peak = true_peak.max(f64::from(interpolated.abs()));
        }
    }
    let overshoot_db = if digital_peak > 0.0 && true_peak > 0.0 {
        20.0 * (true_peak / f64::from(digital_peak)).log10()
    } else {
        0.0
    };
    let _ = CLIP_RUN_SAMPLES; // documented alongside longest_clip_run for consumers
    (fraction, longest_run, overshoot_db)
}

/// Envelope-domain SNR: mean active-frame power (inside VAD regions) versus the noise
/// floor power, in dB. Returns `(estimate, pause_floor_rms)`; the estimate is `None` when
/// the reference has no detectable speech at all.
#[must_use]
pub fn snr_estimate_db(pcm: &[f32]) -> (Option<f64>, f64) {
    let energies = frame_energies(pcm);
    if energies.is_empty() {
        return (None, 0.0);
    }
    let floor_rms = percentile(energies.clone(), FLOOR_QUANTILE);
    let regions = voice_activity_regions(pcm);
    let mut active_power_sum = 0.0_f64;
    let mut active_frames = 0_usize;
    for (region_start, region_end) in regions {
        let mut frame = region_start;
        while frame + FRAME_LEN <= region_end {
            let rms = frame_rms(pcm, frame);
            active_power_sum += rms * rms;
            active_frames += 1;
            frame += HOP;
        }
    }
    if active_frames == 0 || floor_rms <= 0.0 {
        return (None, floor_rms);
    }
    let active_rms = (active_power_sum / active_frames as f64).sqrt();
    (
        Some(20.0 * (active_rms / floor_rms).log10()),
        floor_rms,
    )
}

/// Music-bed likelihood in `[0, 1]`: the fraction of ACTIVE frames that are simultaneously
/// tonal (spectral flatness under [`TONAL_FLATNESS_MAX`]) and part of a low-modulation
/// span (neighbor-frame RMS ratio near 1 — beds sustain; speech syllables pump). Uses a
/// Hann-windowed 1,024-point spectrum via the workspace's `rustfft`.
#[must_use]
pub fn music_bed_likelihood(pcm: &[f32]) -> f64 {
    const FFT_LEN: usize = 1024;
    if pcm.len() < FFT_LEN * 2 {
        return 0.0;
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_LEN);
    let hann: Vec<f32> = (0..FFT_LEN)
        .map(|index| {
            0.5_f32
                * (1.0
                    - (2.0 * std::f32::consts::PI * index as f32 / FFT_LEN as f32).cos())
        })
        .collect();

    let mut frame_flatness: Vec<f64> = Vec::new();
    let mut frame_rms_values: Vec<f64> = Vec::new();
    let mut offset = 0;
    while offset + FFT_LEN <= pcm.len() {
        let mut buffer: Vec<Complex<f32>> = pcm[offset..offset + FFT_LEN]
            .iter()
            .zip(&hann)
            .map(|(&sample, &window)| Complex::new(sample * window, 0.0))
            .collect();
        fft.process(&mut buffer);
        // Flatness over the voiced band (~100 Hz .. 5 kHz): geometric vs arithmetic mean
        // of magnitudes. Pure tones → near zero; broadband → near one.
        let band: Vec<f64> = buffer[FFT_LEN / 240..FFT_LEN / 5]
            .iter()
            .map(|bin| bin.norm().max(1e-9) as f64)
            .collect();
        let log_mean: f64 = band.iter().map(|value| value.ln()).sum::<f64>() / band.len() as f64;
        let linear_mean: f64 = band.iter().sum::<f64>() / band.len() as f64;
        frame_flatness.push(if linear_mean > 0.0 {
            log_mean.exp() / linear_mean
        } else {
            1.0
        });
        frame_rms_values.push(frame_rms(
            pcm,
            offset.min(pcm.len().saturating_sub(FRAME_LEN)),
        ));
        offset += HOP;
    }
    if frame_flatness.len() < 4 {
        return 0.0;
    }

    let active = voice_activity_regions(pcm);
    let mut tonal_low_modulation = 0_u32;
    let mut considered = 0_u32;
    for index in 1..frame_flatness.len() {
        let frame_start = index * HOP;
        let in_speech = active
            .iter()
            .any(|&(start, end)| frame_start >= start && frame_start + FFT_LEN <= end);
        if !in_speech {
            continue;
        }
        considered += 1;
        let modulation = if frame_rms_values[index - 1] > 0.0 && frame_rms_values[index] > 0.0 {
            (frame_rms_values[index] / frame_rms_values[index - 1]).abs().ln()
        } else {
            f64::INFINITY
        };
        // Beds hold level (|log ratio| tiny across a 10 ms hop); syllables do not.
        if frame_flatness[index] < TONAL_FLATNESS_MAX && modulation < 0.10 {
            tonal_low_modulation += 1;
        }
    }
    if considered == 0 {
        return 0.0;
    }
    f64::from(tonal_low_modulation) / f64::from(considered)
}

/// Stationarity drift: the relative spread between the quietest and loudest QUARTER of the
/// timeline's median frame RMS. Stationary defect sources (constant hum, one steady bed)
/// score near zero regardless of loudness; healthy speech with pauses does not.
#[must_use]
pub fn stationarity_drift(pcm: &[f32]) -> f64 {
    let energies = frame_energies(pcm);
    if energies.len() < 8 {
        return 0.0;
    }
    let quarter = energies.len() / 4;
    let medians: [f64; 4] = [
        percentile(energies[..quarter].to_vec(), 0.5),
        percentile(energies[quarter..2 * quarter].to_vec(), 0.5),
        percentile(energies[2 * quarter..3 * quarter].to_vec(), 0.5),
        percentile(energies[3 * quarter..].to_vec(), 0.5),
    ];
    let min = medians.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = medians.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max <= 0.0 {
        return 0.0;
    }
    (max - min) / max
}

/// Everything this module measures about one reference, ready to print.
#[derive(Debug)]
pub struct AudioDiagnostics {
    /// Fraction of samples railed at or over [`CLIP_LEVEL`].
    pub clipping_fraction: f64,
    /// Longest consecutive at-rail run, samples.
    pub longest_clip_run: usize,
    /// Estimated true-peak overshoot over the digital peak, dB.
    pub intersample_overshoot_db: f64,
    /// Active-speech vs pause-floor power, dB. `None` when no speech detected.
    pub snr_estimate_db: Option<f64>,
    /// 10th-percentile frame RMS, dBFS.
    pub pause_floor_dbfs: f64,
    /// Reverb-time equivalent from the existing estimator (see [`crate::synth`]).
    pub reverb_time_s: Option<f64>,
    /// Music-bed likelihood in `[0, 1]`.
    pub music_bed_likelihood: f64,
    /// Quarter-to-quarter energy spread, relative.
    pub stationarity_drift: f64,
    /// Whole-file RMS, dBFS (a loudness approximation, explicitly NOT LUFS).
    pub loudness_rms_dbfs: f64,
    /// Fraction of the timeline inside a voice-activity region.
    pub voice_activity_ratio: f64,
}

/// Runs every detector over one mono 24 kHz reference. `reverb_time_s` comes from the
/// existing enrollment estimator so there is one definition of that quantity.
#[must_use]
pub fn diagnose(pcm: &[f32], reverb_time_s: Option<f64>) -> AudioDiagnostics {
    let (clipping_fraction, longest_clip_run, intersample_overshoot_db) =
        clipping_diagnostics(pcm);
    let (snr_estimate_db, floor_rms) = snr_estimate_db(pcm);
    let regions = voice_activity_regions(pcm);
    let voiced_samples: usize = regions.iter().map(|(start, end)| end - start).sum();
    let total_rms = if pcm.is_empty() {
        0.0
    } else {
        (pcm.iter().map(|&value| f64::from(value) * f64::from(value)).sum::<f64>()
            / pcm.len() as f64)
            .sqrt()
    };
    AudioDiagnostics {
        clipping_fraction,
        longest_clip_run,
        intersample_overshoot_db,
        snr_estimate_db,
        pause_floor_dbfs: if floor_rms > 0.0 {
            20.0 * floor_rms.log10()
        } else {
            f64::NEG_INFINITY
        },
        reverb_time_s,
        music_bed_likelihood: music_bed_likelihood(pcm),
        stationarity_drift: stationarity_drift(pcm),
        loudness_rms_dbfs: if total_rms > 0.0 {
            20.0 * total_rms.log10()
        } else {
            f64::NEG_INFINITY
        },
        voice_activity_ratio: if pcm.is_empty() {
            0.0
        } else {
            f64::from(voiced_samples as u32) / pcm.len() as f64
        },
    }
}
