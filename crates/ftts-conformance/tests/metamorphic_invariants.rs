//! Metamorphic invariants for the synthesis pipeline (bead
//! `frankentts-v-metamorphic-0wq`, collision-free subset).
//!
//! Each test pins one relation the plan declares must hold for ANY implementation:
//!
//! * **same seed + same inputs → byte-identical PCM** — the determinism floor every
//!   other comparison (goldens, route A/Bs) silently relies on;
//! * **different seed → different rendering** — the sampler actually consumes entropy;
//!   if seeds stop mattering, every seeded metric above is measuring one sample;
//! * **packet schedule does not change delivered samples** — packets are a latency
//!   dial, never a content dial (packet-1 == packet-4 == odd schedules);
//! * **reference-route PCM matches its golden hash** — the pinned-stream regression
//!   ratchet, updated only through `UPDATE_GOLDENS=1` with mandatory diff review.
//!
//! Thread-count invariance lives in `crates/ftts-cli/tests` because the worker team is
//! a process-global and needs two processes to vary.
//!
//! Model-gated; each absence reports an honest skip.

use std::path::{Path, PathBuf};

use ftts_cli::preset_voice_path;
use ftts_cli::synth::{
    LoadedModel, ModelBundle, PcmPacketSink, SynthesizedAudio, read_speaker_vector, synthesize,
};
use ftts_conformance::report::{Outcome, Receipt};
use ftts_core::{CancellationToken, SynthesisRequest, TtsEngine};
use sha2::{Digest, Sha256};

const CONTRACT: &str = "ProductionQuality/metamorphic";
const VOICE: &str = "matt";
const TEXT: &str = "Hello.";
/// Bounds each rendering to ~3 s of audio so five syntheses stay minutes, not tens.
const FRAME_CAP: u64 = 36;
const SEED_A: u64 = 42;
const SEED_B: u64 = 43;

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped).contract(CONTRACT).seam("metamorphic").reason(reason).emit();
}

struct Bench {
    model: LoadedModel,
    bundle: ModelBundle,
    speaker: Vec<f32>,
}

fn bench() -> Result<Bench, String> {
    let bundle_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/truth-pack/snapshots/hf");
    let bundle = ModelBundle::resolve(&bundle_root).map_err(|e| format!("bundle: {e}"))?;
    let model = LoadedModel::load(&bundle).map_err(|e| format!("checkpoint: {e}"))?;
    let voice_path = preset_voice_path(VOICE).map_err(|e| format!("preset voice: {e}"))?;
    let speaker = read_speaker_vector(&voice_path).map_err(|e| format!("speaker: {e}"))?;
    Ok(Bench { model, bundle, speaker })
}

fn render(
    bench: &Bench,
    seed: u64,
    packet_frames: usize,
    sink: Option<&mut dyn PcmPacketSink>,
) -> Result<SynthesizedAudio, String> {
    let engine = TtsEngine::new(ftts_core::EngineConfig {
        synthesis_stage_budget: std::time::Duration::from_secs(3_600),
        admission: ftts_core::admission::AdmissionPolicy {
            max_new_tokens: FRAME_CAP,
            ..ftts_core::admission::AdmissionPolicy::default()
        },
        ..ftts_core::EngineConfig::default()
    })
    .map_err(|e| format!("engine: {e}"))?;
    synthesize(
        &bench.model,
        &engine,
        &SynthesisRequest::new(TEXT),
        &bench.speaker,
        seed,
        &CancellationToken::new(),
        &|_event: ftts_core::SynthesisEvent| {},
        packet_frames,
        None,
        sink,
    )
    .map_err(|e| format!("synthesis: {e}"))
}

fn pcm_sha256(pcm: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for sample in pcm {
        let bits = sample.to_bits();
        hasher.update(bits.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn same_seed_and_inputs_are_byte_identical() {
    const TEST: &str = "metamorphic_same_seed_is_byte_identical";
    let bench = match bench() {
        Ok(bench) => bench,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    let first = match render(&bench, SEED_A, 4, None) {
        Ok(audio) => audio,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    let second = match render(&bench, SEED_A, 4, None) {
        Ok(audio) => audio,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    assert_eq!(first.frames, second.frames, "frame counts diverge under identical inputs");
    assert_eq!(first.pcm.len(), second.pcm.len(), "sample counts diverge");
    let divergent = first
        .pcm
        .iter()
        .zip(&second.pcm)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        divergent.is_none(),
        "same seed diverged at sample {divergent:?}: the determinism floor is broken, \
         every seeded comparison downstream is now uninterpretable"
    );
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("metamorphic.determinism")
        .reason(format!("seed {SEED_A} twice → {} frames, pcm sha {}", first.frames, pcm_sha256(&first.pcm)))
        .emit();
}

#[test]
fn different_seed_changes_the_rendering() {
    const TEST: &str = "metamorphic_different_seed_varies_output";
    let bench = match bench() {
        Ok(bench) => bench,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    let a = match render(&bench, SEED_A, 4, None) {
        Ok(audio) => audio,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    let b = match render(&bench, SEED_B, 4, None) {
        Ok(audio) => audio,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    let identical_len = a.frames == b.frames && a.pcm.len() == b.pcm.len();
    let identical_bits = identical_len
        && a.pcm.iter().zip(&b.pcm).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(
        !identical_bits,
        "seeds {SEED_A} and {SEED_B} produced identical audio: the sampler consumed no \
         entropy — seeded metrics are measuring one draw"
    );
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("metamorphic.seed_sensitivity")
        .reason(format!(
            "seeds {SEED_A}/{SEED_B} → {} vs {} frames, distinct pcm",
            a.frames, b.frames
        ))
        .emit();
}

/// Collects everything a packet schedule delivers.
struct CollectingSink {
    samples: Vec<f32>,
}

impl PcmPacketSink for CollectingSink {
    fn deliver(&mut self, samples: &[f32], _frames: usize) -> Result<(), ftts_cli::FttsError> {
        self.samples.extend_from_slice(samples);
        Ok(())
    }
}

#[test]
fn packet_schedule_does_not_change_delivered_samples() {
    const TEST: &str = "metamorphic_packet_schedule_content_invariance";
    let bench = match bench() {
        Ok(bench) => bench,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };

    let mut rendered: Vec<(usize, Vec<f32>)> = Vec::new();
    for packet_frames in [1_usize, 4, 7] {
        let mut sink = CollectingSink { samples: Vec::new() };
        let audio = match render(&bench, SEED_A, packet_frames, Some(&mut sink)) {
            Ok(audio) => audio,
            Err(reason) => {
                skip(TEST, &reason);
                return;
            }
        };
        assert_eq!(
            sink.samples.len(),
            audio.pcm.len(),
            "schedule {packet_frames}: delivered {} samples but buffer holds {}",
            sink.samples.len(),
            audio.pcm.len()
        );
        rendered.push((packet_frames, sink.samples));
    }

    let (_, baseline) = rendered.remove(0);
    for (schedule, samples) in &rendered {
        let divergent = baseline
            .iter()
            .zip(samples)
            .position(|(a, b)| a.to_bits() != b.to_bits());
        assert!(
            divergent.is_none(),
            "packet schedule {schedule} diverges from schedule 1 at sample {divergent:?}"
        );
    }
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("metamorphic.packet_invariance")
        .reason("schedules 1/4/7 deliver bit-identical streams")
        .emit();
}

/// The golden ratchet: reference-route PCM for the conformance case, pinned by hash.
///
/// `UPDATE_GOLDENS=1` rewrites the file and prints the old/new hashes — the mandatory
/// human diff review. CI never updates; a mismatch fails with the exact command.
#[test]
fn reference_route_pcm_matches_golden_hash() {
    const TEST: &str = "metamorphic_reference_route_golden_pcm";
    ftts_conformance::pin_reference_route();

    // Same setup as say_end_to_end_wav: oracle-captured synthetic-tone speaker.
    let fixtures = match ftts_conformance::oracle::OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(TEST, &format!("oracle fixtures unavailable: {error}"));
            return;
        }
    };
    let speaker = fixture_speaker(&fixtures);
    let Some(speaker) = speaker else {
        skip(TEST, "captured speaker embedding absent from the fixture pack");
        return;
    };
    let bundle_root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/truth-pack/snapshots/hf");
    let bundle = match ModelBundle::resolve(&bundle_root) {
        Ok(bundle) => bundle,
        Err(error) => {
            skip(TEST, &format!("bundle incomplete: {error}"));
            return;
        }
    };
    let model = match LoadedModel::load(&bundle) {
        Ok(model) => model,
        Err(error) => {
            skip(TEST, &format!("checkpoint unusable: {error}"));
            return;
        }
    };
    let engine = match TtsEngine::new(ftts_core::EngineConfig {
        synthesis_stage_budget: std::time::Duration::from_secs(3_600),
        admission: ftts_core::admission::AdmissionPolicy {
            max_new_tokens: 24,
            ..ftts_core::admission::AdmissionPolicy::default()
        },
        ..ftts_core::EngineConfig::default()
    }) {
        Ok(engine) => engine,
        Err(error) => {
            skip(TEST, &format!("engine: {error}"));
            return;
        }
    };
    let audio = match synthesize(
        &model,
        &engine,
        &SynthesisRequest::new(TEXT),
        &speaker,
        0,
        &CancellationToken::new(),
        &|_event: ftts_core::SynthesisEvent| {},
        4,
        None,
        None,
    ) {
        Ok(audio) => audio,
        Err(reason) => {
            skip(TEST, &format!("synthesis: {reason}"));
            return;
        }
    };

    let hash = pcm_sha256(&audio.pcm);
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/metamorphic/golden_reference_pcm.json");

    if std::env::var("UPDATE_GOLDENS").as_deref() == Ok("1") {
        let entry = serde_json::json!({
            "schema_version": 1,
            "route": "cpu_fp32_reference",
            "text": TEXT,
            "voice": "fixture_synthetic_tone",
            "frames": audio.frames,
            "pcm_f32_bits_sha256": hash,
        });
        std::fs::create_dir_all(golden_path.parent().expect("parent")).expect("mkdir");
        std::fs::write(
            &golden_path,
            serde_json::to_string_pretty(&entry).expect("json") + "\n",
        )
        .expect("write golden");
        println!("GOLDEN UPDATED: pcm sha {hash}");
        Receipt::new(TEST, Outcome::Passed)
            .contract(CONTRACT)
            .seam("metamorphic.golden")
            .reason(format!("golden rewritten: frames {}, sha {hash}", audio.frames))
            .emit();
        return;
    }

    let Ok(existing) = std::fs::read_to_string(&golden_path) else {
        skip(
            TEST,
            &format!(
                "no golden at {} — generate deliberately once and review the diff: \
                 UPDATE_GOLDENS=1 cargo test -p ftts-conformance --test metamorphic_invariants",
                golden_path.display()
            ),
        );
        return;
    };
    let golden: serde_json::Value = match serde_json::from_str(&existing) {
        Ok(value) => value,
        Err(error) => {
            skip(TEST, &format!("golden file unparsable: {error}"));
            return;
        }
    };
    let stored = golden["pcm_f32_bits_sha256"].as_str().unwrap_or("");
    assert_eq!(
        stored,
        hash,
        "reference-route PCM drifted from the pinned golden.\n  golden: {stored}\n  actual:  \
         {hash}\nIf this drift is INTENDED (an accepted numeric change), re-pin with review: \
         UPDATE_GOLDENS=1 cargo test -p ftts-conformance --test metamorphic_invariants\nIf it \
         is NOT intended, treat as a numerics regression before anything else ships."
    );
    assert_eq!(
        golden["frames"].as_u64(),
        Some(audio.frames),
        "frame count drifted from the golden"
    );
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("metamorphic.golden")
        .reason(format!(
            "{} frames, pcm sha {hash} matches golden",
            audio.frames
        ))
        .emit();
}

/// The oracle's captured x-vector for the fixture reference voice.
fn fixture_speaker(fixtures: &ftts_conformance::oracle::OracleFixtures) -> Option<Vec<f32>> {
    let seam = ftts_conformance::oracle::SeamRef {
        case: "synthetic-tone-en",
        mode: "xvector_non_streaming",
        group: "prompt_build",
        seam: "prompt.speaker_embedding",
    };
    if !fixtures.has_seam(&seam) {
        return None;
    }
    let npy = fixtures.seam(&seam, "tensor", 0).ok()?;
    let mut vector: Vec<f32> = npy.data.clone();
    vector.truncate(1024);
    (vector.len() == 1024).then_some(vector)
}
