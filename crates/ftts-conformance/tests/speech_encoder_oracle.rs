//! The speech-tokenizer ENCODER against the pinned reference, EXACT (bead
//! frankentts-p1-codec-encoder-snt).
//!
//! The fixture (`fixtures/speech_encoder_oracle.json`) was captured by
//! `scripts/capture_speech_encoder_oracle.py`: the pinned CPU float32 stack
//! (transformers 4.57.3, snapshot qwen_tts wrapper) encoding deterministic synthetic
//! waveforms. The waveform is REGENERATED here from the recorded `(samples, seed)` —
//! integer-seeded xorshift64 noise plus an integer-phase sawtooth, all arithmetic in
//! f64 then one cast to f32, so both languages produce identical bits and no audio
//! ships in the repo.
//!
//! Codes are discrete, so the comparison is exact equality over every `[frame, group]`
//! id — the strongest claim tier the artifact class allows, and the roundtrip gate
//! spec §8 asks for before the encoder rows may graduate to a kernel contract.
//!
//! Model-gated: without `speech_tokenizer/model.safetensors` the test reports the skip
//! and passes (skip-honest, like every model-backed suite here).

use std::path::PathBuf;

use ftts_model_qwen::speech_encoder::SpeechEncoder;

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var("FTTS_MODEL_DIR").map_or_else(
        |_| {
            #[allow(deprecated)]
            std::env::home_dir().map(|home| home.join(".cache/franken_tts/model"))
        },
        |dir| Some(PathBuf::from(dir)),
    )?;
    root.join("speech_tokenizer/model.safetensors")
        .is_file()
        .then_some(root)
}

/// The capture script's waveform, bit for bit (see its `synthetic_wave`).
fn synthetic_wave(samples: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut wave = Vec::with_capacity(samples);
    for index in 0..samples {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = ((state >> 40) as f64) / f64::from(1u32 << 24) - 0.5;
        let saw = ((index % 240) as f64) / 240.0 - 0.5;
        wave.push((noise * 0.3 + saw * 0.4) as f32);
    }
    wave
}

#[test]
fn encoded_codes_match_the_pinned_oracle_exactly() {
    let fixture_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/speech_encoder_oracle.json");
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|error| panic!("fixture {}: {error}", fixture_path.display())),
    )
    .expect("fixture parses");

    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"speech_encoder_oracle\",\"outcome\":\"skipped\",\
             \"reason\":\"speech_tokenizer checkpoint unavailable\"}}"
        );
        return;
    };
    let encoder = SpeechEncoder::load(&root.join("speech_tokenizer/model.safetensors"))
        .expect("encoder loads from the shipped checkpoint");

    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let samples = usize::try_from(case["samples"].as_u64().expect("samples")).expect("usize");
        let seed = case["seed"].as_u64().expect("seed");
        let frames = usize::try_from(case["frames"].as_u64().expect("frames")).expect("usize");
        let expected = case["codes"].as_array().expect("codes");
        assert_eq!(expected.len(), frames, "{name}: fixture self-consistency");

        let wave = synthetic_wave(samples, seed);
        let codes = encoder
            .encode_24khz_pcm(&wave)
            .unwrap_or_else(|error| panic!("{name}: encode failed: {error}"));
        assert_eq!(codes.len(), frames * 16, "{name}: frame count");

        let mut mismatches = 0usize;
        let mut first: Option<(usize, usize, u32, u64)> = None;
        for (frame, row) in expected.iter().enumerate() {
            for (group, id) in row.as_array().expect("row").iter().enumerate() {
                let oracle = id.as_u64().expect("id");
                let ours = codes[frame * 16 + group];
                if u64::from(ours) != oracle {
                    mismatches += 1;
                    first.get_or_insert((frame, group, ours, oracle));
                }
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{name}: {mismatches}/{} ids diverge; first at frame {} group {} (ours {} oracle {})",
            frames * 16,
            first.map_or(0, |f| f.0),
            first.map_or(0, |f| f.1),
            first.map_or(0, |f| f.2),
            first.map_or(0, |f| f.3),
        );
        eprintln!(
            "receipt: {{\"test\":\"speech_encoder_oracle\",\"case\":\"{name}\",\
             \"outcome\":\"passed\",\"frames\":{frames},\"ids_compared\":{}}}",
            frames * 16
        );
    }
}
