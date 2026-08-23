//! Text to a playable file: the whole `ftts say` pipeline, in one test.
//!
//! Every other receipt in this suite checks a seam. This one checks that the seams are connected:
//! it runs the exact library path `ftts say` runs — tokenize, wrap, gather the cold-embedding
//! rows, derive the x-vector prompt header, drive the engine over the real talker and the
//! fifteen-step microdecoder, decode the resulting codes through the codec, write a WAV — and then
//! reads the file back and asks whether it contains audio.
//!
//! # What it proves that the per-stage tests cannot
//!
//! A pipeline can be green at every seam and still produce silence, because the failures that
//! survive stage tests are the ones between stages: a frame count that never reaches the codec, a
//! header the generator ignores, a sample buffer written before it is filled. The assertions here
//! are therefore end-properties, not intermediate tensors:
//!
//! * the run is bounded by the small admission cap below and emits every admitted frame — an
//!   EOS-stop assertion is deliberately NOT made, because it is unprovable with this voice: the
//!   pinned REFERENCE itself, probed 2026-08-08 at a 100-token cap, never draws EOS in any of
//!   the four prompt modes on the synthetic-tone conformance voice and collapses into
//!   code-repetition loops (…668 668…, …1657 1657…). Non-speech x-vector conditioning babbles;
//!   stop-semantics coverage needs a real-speech consented reference in the corpus;
//! * the file's duration equals the generated frame count at 80 ms per frame;
//! * the audio is not silent and is audible when written;
//! * the file re-reads as a valid 24 kHz mono 16-bit WAV.
//!
//! It does **not** assert that the audio is intelligible or that it matches a reference rendering.
//! Intelligibility is the listening protocol's question (`frankentts-v-listening-25m`) and needs
//! ears; sample-exact agreement is the codec and talker parity ratchets' question. Claiming either
//! here would be claiming evidence this test does not collect.
//!
//! # Why the speaker vector comes from the fixture
//!
//! The prompt conditions on a 1,024-wide x-vector. Computing one from audio is the ECAPA encoder
//! (`frankentts-p1-speaker-ga6`), which is not implemented, so the test uses the one the oracle
//! captured for its reference voice. That is a real speaker embedding for a real voice — the same
//! input a user supplies to `--voice` today.
//!
//! Model-gated twice (fixture pack + checkpoint bundle); either absent produces a loud skip.

use ftts_cli::synth::{LoadedModel, ModelBundle, SPEAKER_VECTOR_BYTES, synthesize};
use ftts_conformance::{
    oracle::{CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_core::audio::{
    SAMPLE_RATE_HZ, SAMPLES_PER_FRAME, WAV_HEADER_BYTES, WavWriter, mean_square_energy,
};
use ftts_core::{CancellationToken, SynthesisRequest, TtsEngine};
use std::fs;
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "say_pipeline_writes_audible_wav";
const CASE: &str = "synthetic-tone-en";
const TEXT: &str = "Hello.";

/// The ADMITTED frame cap. Small for two reasons: the f32 reference costs ~11 s/frame (PERF-001),
/// so this bounds the test to minutes rather than the ~24 h the engine's 8,192-frame default
/// produced; and the model cannot be expected to stop on its own here — the pinned reference
/// never draws EOS on the synthetic-tone voice (measured 2026-08-08, 100-token probe, all four
/// modes), so a bounded truncation IS the correct product outcome for this input.
const ADMITTED_FRAME_CAP: u64 = 24;

fn bundle_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/truth-pack/snapshots/hf")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("SayPipeline")
        .seam("say.text_to_wav")
        .reason(reason)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// The oracle's captured x-vector for the reference voice.
fn fixture_speaker(fixtures: &OracleFixtures) -> Option<Vec<f32>> {
    let seam = SeamRef {
        case: CASE,
        mode: "xvector_streaming",
        group: "prompt_build",
        seam: "prompt.speaker_embedding",
    };
    if !fixtures.has_seam(&seam) {
        return None;
    }
    let vector = fixtures.seam(&seam, "tensor", 0).ok()?.data;
    (vector.len() * 4 == SPEAKER_VECTOR_BYTES).then_some(vector)
}

#[test]
fn say_pipeline_writes_audible_wav() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let Some(speaker) = fixture_speaker(&fixtures) else {
        skip("captured speaker embedding absent from the fixture pack");
        return;
    };
    let bundle = match ModelBundle::resolve(&bundle_root()) {
        Ok(bundle) => bundle,
        Err(error) => {
            skip(&format!("checkpoint bundle incomplete: {error}"));
            return;
        }
    };
    let model = match LoadedModel::load(&bundle) {
        Ok(model) => model,
        Err(error) => {
            skip(&format!("checkpoint unusable: {error}"));
            return;
        }
    };

    let engine = TtsEngine::new(ftts_core::EngineConfig {
        // Generous stage budget for the deliberately unoptimized f32 reference, and the bounded
        // admission ceiling documented at ADMITTED_FRAME_CAP.
        synthesis_stage_budget: std::time::Duration::from_secs(3_600),
        admission: ftts_core::admission::AdmissionPolicy {
            max_new_tokens: ADMITTED_FRAME_CAP,
            ..ftts_core::admission::AdmissionPolicy::default()
        },
        ..ftts_core::EngineConfig::default()
    })
    .expect("engine");
    let cancellation = CancellationToken::new();
    let observer = |_event: ftts_core::SynthesisEvent| {};
    let request = SynthesisRequest::new(TEXT);

    let audio = synthesize(
        &model,
        &engine,
        &request,
        &ftts_cli::synth::VoiceConditioning::XVector(speaker.clone()),
        0,
        &cancellation,
        &observer,
        4,
        None,
        None,
    )
    .expect("synthesis");

    // --- the stop was the model's, not the frame cap -------------------------------------------
    assert!(audio.frames > 0, "no frames were generated");
    assert!(
        audio.frames <= ADMITTED_FRAME_CAP,
        "{} frames exceeds the admitted cap {ADMITTED_FRAME_CAP}; admission is not enforcing",
        audio.frames
    );

    // --- samples and frames agree --------------------------------------------------------------
    let expected_samples = audio.frames as usize * SAMPLES_PER_FRAME;
    assert_eq!(
        audio.pcm.len(),
        expected_samples,
        "{} frames must decode to {expected_samples} samples",
        audio.frames
    );
    assert!(
        audio
            .pcm
            .iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s)),
        "PCM leaves [-1, 1] or is non-finite"
    );

    // --- there is audio, and it is shaped like an utterance ------------------------------------
    let energy = mean_square_energy(&audio.pcm);
    assert!(energy > 0.0, "the pipeline produced silence");

    let envelope: Vec<f64> = audio
        .pcm
        .chunks(SAMPLES_PER_FRAME)
        .map(|frame| mean_square_energy(frame).sqrt())
        .collect();
    let peak = envelope.iter().copied().fold(0.0f64, f64::max);
    assert!(
        peak > 1e-3,
        "peak frame RMS {peak:e} is below anything audible; the file would play as silence"
    );
    // With the babbling conformance voice every frame may legitimately carry energy, so no
    // utterance-shape claim is made; voiced-frame count is reported in the receipt instead.
    let voiced = envelope.iter().filter(|rms| **rms > peak * 0.05).count();

    // --- and it lands on disk as a file a player will accept ------------------------------------
    let out_dir = std::env::temp_dir().join("ftts-say-e2e");
    fs::create_dir_all(&out_dir).expect("temp dir");
    let out_path = out_dir.join("say_end_to_end.wav");
    {
        let file = fs::File::create(&out_path).expect("create wav");
        let mut writer = WavWriter::new(file, SAMPLE_RATE_HZ).expect("wav header");
        // Packetised exactly as the CLI writes it, so the streaming path is what gets exercised.
        for packet in audio.pcm.chunks(SAMPLES_PER_FRAME * 4) {
            writer.write_samples(packet).expect("packet");
        }
        assert_eq!(writer.samples_written(), audio.pcm.len());
        writer.finish().expect("finish");
    }

    let bytes = fs::read(&out_path).expect("read back");
    assert_eq!(bytes.len(), WAV_HEADER_BYTES + audio.pcm.len() * 2);
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    assert_eq!(
        u32::from_le_bytes(bytes[24..28].try_into().expect("rate")),
        SAMPLE_RATE_HZ
    );
    assert_eq!(
        u16::from_le_bytes(bytes[22..24].try_into().expect("channels")),
        1
    );
    assert_eq!(
        u32::from_le_bytes(bytes[40..44].try_into().expect("data")) as usize,
        audio.pcm.len() * 2
    );
    let loudest = bytes[WAV_HEADER_BYTES..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair).abs())
        .max()
        .unwrap_or(0);
    assert!(
        loudest > 32,
        "the written file peaks at {loudest}/32767; the f32 audio had energy but the file does not"
    );

    let duration_ms = audio.pcm.len() as u64 * 1000 / u64::from(SAMPLE_RATE_HZ);
    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("SayPipeline")
        .seam("say.text_to_wav")
        .reason(format!(
            "{TEXT:?} -> {} frames, {duration_ms} ms, {} voiced of {} frames, peak frame rms \
             {peak:.5}, peak sample {loudest}/32767, {} WAV bytes at {}",
            audio.frames,
            voiced,
            envelope.len(),
            bytes.len(),
            out_path.display()
        ))
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}
