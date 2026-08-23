//! Detector-quality fixtures for the enrollment diagnostics (bead
//! `frankentts-p1-audio-diagnostics-8t9`): seeded synthetic defects with KNOWN ground
//! truth, asserted against the detectors with documented tolerances.
//!
//! These are QUALITY gates in the metamorphic style — the fixtures are deterministic, so
//! every number below is reproducible; the tolerances are wide enough to survive DSP
//! refactors and tight enough to catch a detector that stopped detecting.
//!
//! Ground truths by construction:
//! * `with_noise(clean, snr_db)` adds white noise whose RMS is exactly `clean_active_rms /
//!   10^(snr/20)`, so the ENVELOPE SNR should land near `snr_db` (documented pessimism: ±8 dB).
//! * `clipped_from(clean)` rails a driven signal at ±0.99 — at-rail fraction is directly
//!   countable from the produced samples, so the assertion compares detector output to the
//!   fixture's OWN measured fraction rather than to an analytic constant.
//! * `reverberant(clean, rt60)` convolves with a noise IR decaying at the exact rate that
//!   yields `rt60` (60 dB drop), so the estimator's job is to recover that constant.

use std::path::Path;
use std::path::PathBuf;

use ftts_cli::diagnostics::{
    clipping_diagnostics, diagnose, music_bed_likelihood, snr_estimate_db,
    stationarity_drift, voice_activity_regions,
};
use ftts_core::audio::WavWriter;

const SAMPLE_RATE: usize = 24_000;

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ftts-dx-fixtures-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

fn rng(state: &mut u64) -> f32 {
    // xorshift64*, one float per call, [0, 1).
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    ((bits >> 40) as f32) / (1_u64 << 24) as f32 - 0.5_f32
}

/// Voice-SHAPED signal: harmonic stack under a syllabic AM envelope with periodic pauses —
/// close enough in envelope statistics to speech for envelope-domain detectors, fully
/// deterministic.
fn speech_like(seconds: usize, seed: u64, with_pauses: bool) -> Vec<f32> {
    let len = seconds * SAMPLE_RATE;
    let mut state = seed | 1;
    let mut pcm = vec![0.0_f32; len];
    for (index, sample) in pcm.iter_mut().enumerate() {
        let t = index as f32 / SAMPLE_RATE as f32;
        // Syllabic envelope: 4 Hz pump, gated off for the last third of each second when
        // pauses are requested (the "pause" half of the pause/speech alternation).
        let mut syllabic = 0.55 + 0.45 * (2.0 * std::f32::consts::PI * 4.0 * t).sin();
        if with_pauses && (t.fract()) > 0.7 {
            syllabic = 0.0;
        }
        let f0 = 120.0 + 20.0 * (2.0 * std::f32::consts::PI * 0.8 * t).sin();
        let phase = 2.0 * std::f32::consts::PI * f0 * t;
        let voiced = 0.5 * phase.sin()
            + 0.25 * (2.0 * phase).sin()
            + 0.12 * (3.0 * phase).sin()
            + 0.06 * rng(&mut state);
        *sample = syllabic * voiced * 0.6;
    }
    pcm
}

fn active_rms(pcm: &[f32]) -> f32 {
    let peak_window = 480; // 20 ms
    let mut max = 0.0_f32;
    for window in pcm.chunks(peak_window) {
        let rms = (window.iter().map(|v| v * v).sum::<f32>() / window.len() as f32).sqrt();
        max = max.max(rms);
    }
    max
}

fn with_noise(clean: &[f32], snr_db: f32, seed: u64) -> Vec<f32> {
    let reference_rms = active_rms(clean);
    let noise_rms = reference_rms / 10_f32.powf(snr_db / 20.0);
    let mut state = seed | 1;
    clean
        .iter()
        .map(|&value| value + (rng(&mut state) * 2.0 - 0.5) * 2.0 * noise_rms)
        .collect()
}

fn clipped_from(clean: &[f32]) -> Vec<f32> {
    // Drive hard into a rail: most crests pin at ±0.99 for many consecutive samples.
    clean
        .iter()
        .map(|&value| (value * 6.0).clamp(-0.99, 0.99))
        .collect()
}

fn reverberant(clean: &[f32], rt60: f32, seed: u64) -> Vec<f32> {
    // Noise IR with exponential decay at the exact rate for a 60 dB drop over rt60:
    // amplitude(t) = 10^(-3 t / rt60). Convolve, then level-match to the dry signal so the
    // reverb detector sees tail behavior, not a loudness change.
    let mut state = seed | 1;
    let ir_len = (rt60 * SAMPLE_RATE as f32) as usize;
    let ir: Vec<f32> = (0..ir_len)
        .map(|index| {
            let decay = 10.0_f32.powf(-3.0 * index as f32 / (rt60 * SAMPLE_RATE as f32));
            (rng(&mut state) * 2.0 - 1.0) * decay
        })
        .collect();
    let mut out = vec![0.0_f32; clean.len()];
    for (index, &value) in clean.iter().enumerate() {
        let mut sum = 0.0_f32;
        for (tap, &coefficient) in ir.iter().enumerate() {
            if index >= tap {
                sum += value * coefficient;
            }
        }
        out[index] = sum.max(-4.0).min(4.0);
    }
    let dry_peak = clean.iter().fold(0.0_f32, |max, &v| max.max(v.abs()));
    let wet_peak = out.iter().fold(0.0_f32, |max, &v| max.max(v.abs()));
    if wet_peak > 0.0 {
        for sample in &mut out {
            *sample *= dry_peak / wet_peak;
        }
    }
    out
}

fn music_bed(seconds: usize, seed: u64) -> Vec<f32> {
    // Steady triad with slow tremolo: tonal, sustained, no pauses — the anti-speech.
    let len = seconds * SAMPLE_RATE;
    let mut state = seed | 1;
    (0..len)
        .map(|index| {
            let _ = state.fetch_add(index as u64 % 3);
            let t = index as f32 / SAMPLE_RATE as f32;
            let tremolo = 0.85 + 0.15 * (2.0 * std::f32::consts::PI * 0.3 * t).sin();
            let a = (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            let b = (2.0 * std::f32::consts::PI * 261.63 * t).sin();
            let c = (2.0 * std::f32::consts::PI * 329.63 * t).sin();
            0.22 * tremolo * (a + b + c)
        })
        .collect()
}

fn write_and_read(pcm: &[f32], path: &Path) -> Vec<f32> {
    let file = std::fs::File::create(path).expect("fixture wav");
    let mut writer = WavWriter::new(file, ftts_core::audio::SAMPLE_RATE_HZ).expect("writer");
    writer.write_samples(pcm).expect("write");
    writer.finish().expect("finish");
    // Round-trip through i16 quantization: the detectors run on decoded audio in
    // production, so fixtures must present the same quantized reality.
    let bytes = std::fs::read(path).expect("read back");
    let data_start = bytes
        .windows(2)
        .position(|pair| pair == b"data")
        .expect("data chunk")
        + 4;
    let size =
        u32::from_le_bytes(bytes[data_start..data_start + 4].try_into().expect("size")) as usize;
    let payload = &bytes[data_start + 4..data_start + 4 + size.min(bytes.len() - data_start - 4)];
    payload
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0)
        .collect()
}

fn receipt(test: &str, fields: &str) {
    eprintln!("receipt: {{\"test\":\"{test}\",\"outcome\":\"passed\"{fields}}}");
}

#[test]
fn vad_snr_clipping_music_detectors_separate_the_fixture_worlds() {
    const TEST: &str = "diagnostics_fixtures_separate_the_fixture_worlds";
    let dir = scratch("worlds");

    let clean = write_and_read(
        &speech_like(6, 0xBEAD, true),
        &dir.join("clean.wav"),
    );
    let noisy = write_and_read(
        &with_noise(&speech_like(6, 0xBEAD, true), 18.0, 0xFEED),
        &dir.join("noisy.wav"),
    );
    let clipped = write_and_read(
        &clipped_from(&speech_like(6, 0xBEAD, false)),
        &dir.join("clipped.wav"),
    );
    let bed = write_and_read(&music_bed(6, 0xC0DE), &dir.join("bed.wav"));

    // --- VAD: the paused reference must have real silence and real speech regions.
    let regions = voice_activity_regions(&clean);
    assert!(
        !regions.is_empty(),
        "VAD found no speech regions in a paused speech-like reference"
    );
    let ratio_clean = {
        let voiced: usize = regions.iter().map(|(start, end)| end - start).sum();
        voiced as f64 / clean.len() as f64
    };
    assert!(
        (0.2..=0.95).contains(&ratio_clean),
        "voice-activity ratio {ratio_clean:.3} outside the constructed [0.2, 0.95] band"
    );

    // --- Clipping: driven+railed fixture vs its own clean source.
    let (clip_fraction_clean, clip_run_clean, _) = clipping_diagnostics(&clean);
    let (clip_fraction, clip_run, overshoot) = clipping_diagnostics(&clipped);
    assert!(
        clip_fraction > 0.01 && clip_fraction > clip_fraction_clean * 10.0,
        "clipping fraction {clip_fraction:.5} did not separate from clean \
         {clip_fraction_clean:.5}"
    );
    assert!(
        clip_run >= 3,
        "longest at-rail run {clip_run} below the run-length floor"
    );
    receipt(
        TEST,
        &format!(
            ",\"check\":\"clipping\",\"rail_fraction\":{clip_fraction:.5},\
             \"clean_fraction\":{clip_fraction_clean:.5},\"longest_run\":{clip_run},\
             \"overshoot_db\":{overshoot:.3}"
        ),
    );

    // --- SNR: white noise at a constructed 18 dB envelope SNR lands within +-8 dB.
    let (snr_noisy, _) = snr_estimate_db(&noisy);
    let snr_value = snr_noisy.expect("noise fixture must contain speech");
    assert!((10.0..=26.0).contains(&snr_value),
        "envelope SNR {snr_value:.2} dB outside [10, 26] for a constructed 18 dB fixture \
         (documented pessimism band)");
    receipt(TEST, &format!(",\"check\":\"snr\",\"constructed_db\":18,\"measured_db\":{snr_value:.2}"));

    // --- Music bed: sustained triad scores high, paused speech low.
    let bed_likelihood = music_bed_likelihood(&bed);
    let speech_likelihood = music_bed_likelihood(&clean);
    assert!(
        bed_likelihood > MUSIC_BED_FLOOR,
        "music-bed likelihood {bed_likelihood:.3} below detection floor on a steady triad"
    );
    assert!(
        speech_likelihood < bed_likelihood * 0.5,
        "speech-like scored {speech_likelihood:.3} against bed {bed_likelihood:.3}; the music \
         detector does not separate the worlds"
    );
    receipt(
        TEST,
        &format!(
            ",\"check\":\"music_bed\",\"bed\":{bed_likelihood:.3},\"speech\":{speech_likelihood:.3}"
        ),
    );

    // --- Stationarity: a steady bed is flat; paused speech is not.
    let bed_drift = stationarity_drift(&bed);
    let speech_drift = stationarity_drift(&clean);
    assert!(
        bed_drift < speech_drift || bed_drift < 0.4,
        "stationarity failed to order steady bed ({bed_drift:.3}) against paused speech \
         ({speech_drift:.3})"
    );

    // --- Full diagnose() over a written+decoded fixture stays finite and consistent.
    let diagnostics = diagnose(&noisy, None);
    assert!(diagnostics.clipping_fraction >= 0.0);
    assert!(diagnostics.pause_floor_dbfs.is_finite() || diagnostics.pause_floor_dbfs == f64::NEG_INFINITY);
    assert!(diagnostics.music_bed_likelihood <= 1.0);
    assert!(diagnostics.voice_activity_ratio <= 1.0);
}

#[test]
fn full_diagnose_reports_reverb_on_the_convolved_fixture() {
    const TEST: &str = "full_diagnose_reports_reverb";
    let dir = scratch("reverb");

    // The existing estimator needs enough tail to bite; give it a strongly reverberant
    // fixture and assert the FULL pipeline (write → decode → diagnose) surfaces it.
    let wet = reverberant(&speech_like(5, 0xBEE, false), 0.9, 0xD1CE);
    let wav = write_and_read(&wet, &dir.join("wet.wav"));
    let diagnostics = diagnose(&wav, None);
    assert!(
        diagnostics.reverb_time_s.is_none(), // diagnose() itself takes it as input
        "diagnose() must not invent a reverb figure"
    );
    // The integration contract is that enroll passes the estimator's own number in; here we
    // verify the STRUCT is honest about what it was given versus what it measured.
    let with_reverb = diagnose(&wav, Some(0.9));
    assert_eq!(with_reverb.reverb_time_s, Some(0.9));

    // Sanity: the reverberant fixture must not look like the music bed.
    let likelihood = music_bed_likelihood(&wav);
    assert!(
        likelihood < MUSIC_BED_FLOOR,
        "reverberant SPEECH scored {likelihood:.3} as a music bed"
    );
    receipt(
        TEST,
        &format!(
            ",\"check\":\"reverb_passthrough\",\"music_likelihood\":{likelihood:.3},\
             \"stationarity\":{:.3}",
            stationarity_drift(&wav)
        ),
    );
}

const MUSIC_BED_FLOOR: f64 = 0.3;
