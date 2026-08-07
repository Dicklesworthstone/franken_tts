//! Codes to PCM to a playable file, through **our** codec decoder and **our** WAV writer.
//!
//! `audio_output_tail` covers the second half of that sentence: it takes the oracle's already-
//! decoded waveform and proves the file we write around it is honest. That leaves the first half
//! — code groups becoming samples — untested outside the codec's own arithmetic-parity ratchet,
//! which compares intermediate tensors and never reaches a file.
//!
//! This test closes the seam. It feeds the fixture's own `[1, 16, frames]` codec codes to
//! [`CodecCheckpoint::decode`] with the pinned speech-tokenizer weights, and takes the result all
//! the way to WAV bytes. It is the path `ftts say` runs after the talker stops.
//!
//! # What it asserts, and what it deliberately does not
//!
//! Asserted: the sample count is exactly `frames * 1920` (tied to the shape of the input codes, so
//! a frame-to-sample error fails here rather than shipping audio of quietly wrong length); every
//! sample is finite and inside `[-1, 1]`; the PCM is not silent; the 16-bit conversion preserves
//! that energy; and the emitted file's two RIFF length fields describe exactly the payload.
//!
//! **Not** asserted: sample-exact agreement with the oracle's waveform. The codec's exact-parity
//! ratchet (`frankentts-p1-codec-hu7`) is open, and `codec_decode_l2` is the test that owns it.
//! Claiming parity here would either duplicate that gate or, worse, quietly weaken it to a
//! tolerance this file chose. What this test does record is the RMS ratio against the oracle
//! waveform, as a number in the receipt — so the gap is visible and tracked rather than unstated.
//!
//! Model-gated twice (fixture pack + speech-tokenizer checkpoint); either absent produces a loud
//! skip receipt, never a silent green.

use ftts_conformance::{
    npy,
    oracle::{CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_core::audio::{
    SAMPLE_RATE_HZ, SAMPLES_PER_FRAME, WAV_HEADER_BYTES, encode_wav, mean_square_energy,
    pcm_f32_to_i16, samples_for_frames,
};
use ftts_model_qwen::checkpoint::CodecCheckpoint;
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "audio_codes_decode_to_a_playable_wav";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "xvector_non_streaming";

fn codec_checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/speech_tokenizer/model.safetensors")
}

fn seam<'a>(name: &'a str) -> SeamRef<'a> {
    SeamRef {
        case: CASE,
        mode: MODE,
        group: "codec_decode",
        seam: name,
    }
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("AudioTail")
        .seam("codec_decoder.input.input")
        .reason(reason)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

fn rms(values: &[f32]) -> f64 {
    mean_square_energy(values).sqrt()
}

#[test]
fn audio_codes_decode_to_a_playable_wav() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let codes_seam = seam("codec_decoder.input.input");
    let waveform_seam = seam("codec.generated_waveform");
    if !fixtures.has_seam(&codes_seam) {
        skip("codec_decoder input codes absent from the fixture pack");
        return;
    }
    let checkpoint_path = codec_checkpoint_path();
    if !checkpoint_path.is_file() {
        skip(&format!(
            "speech-tokenizer checkpoint absent at {}",
            checkpoint_path.display()
        ));
        return;
    }

    // --- the fixture's own codes, group-major [1, 16, frames] -> our frame-major layout ---------
    let codes_path = fixtures.seam_path(&codes_seam, "args.0", 0);
    let raw = npy::read_i64(&codes_path).expect("codec input codes read");
    let (groups, frames) = match raw.shape.as_slice() {
        [1, groups, frames] => (*groups, *frames),
        other => panic!("expected codes [1, 16, frames], got {other:?}"),
    };
    assert_eq!(groups, 16, "the codec always carries 16 code groups");
    assert!(frames > 0, "the fixture must carry at least one frame");
    let mut codes = vec![0i32; frames * groups];
    for group in 0..groups {
        for frame in 0..frames {
            codes[frame * groups + group] =
                i32::try_from(raw.data[group * frames + frame]).expect("codec id fits i32");
        }
    }

    // --- our decoder, real pinned weights ------------------------------------------------------
    let codec = match CodecCheckpoint::load(&checkpoint_path) {
        Ok(codec) => codec,
        Err(error) => {
            skip(&format!("codec checkpoint unusable: {error}"));
            return;
        }
    };
    let pcm = codec.decode(&codes, frames).expect("codec decode");

    // --- exact sample count, tied to the frame count of the input codes -------------------------
    assert_eq!(
        pcm.len(),
        samples_for_frames(frames),
        "{frames} frames of codes must decode to exactly {} samples at 24 kHz / 12.5 fps, got {}",
        samples_for_frames(frames),
        pcm.len()
    );
    assert_eq!(pcm.len() % SAMPLES_PER_FRAME, 0, "whole frames only");

    // --- the samples are audio, not NaNs and not silence ---------------------------------------
    assert!(
        pcm.iter().all(|s| s.is_finite()),
        "decoded PCM contains a non-finite sample; the WAV writer would turn it into silence and \
         hide the bug"
    );
    assert!(
        pcm.iter().all(|s| (-1.0..=1.0).contains(s)),
        "decoded PCM leaves [-1, 1]; the codec's final clamp did not run"
    );
    let energy = mean_square_energy(&pcm);
    assert!(
        energy > 0.0,
        "decoded PCM is silent (mean square energy {energy:e}); a silent result has the right \
         length, the right header, and the right byte count, and is still a failure"
    );

    // --- the file describes exactly what it holds, and the payload survives 16-bit --------------
    let wav = encode_wav(&pcm, SAMPLE_RATE_HZ);
    assert_eq!(wav.len(), WAV_HEADER_BYTES + pcm.len() * 2);
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    let declared_data = u32::from_le_bytes(wav[40..44].try_into().expect("data size"));
    assert_eq!(declared_data as usize, pcm.len() * 2);
    let declared_riff = u32::from_le_bytes(wav[4..8].try_into().expect("riff size"));
    assert_eq!(declared_riff as usize, 36 + pcm.len() * 2);
    assert_eq!(
        u32::from_le_bytes(wav[24..28].try_into().expect("rate")),
        SAMPLE_RATE_HZ
    );

    let quantised = pcm_f32_to_i16(&pcm);
    let decoded: Vec<i16> = wav[WAV_HEADER_BYTES..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect();
    assert_eq!(decoded, quantised, "WAV payload must round-trip exactly");
    assert!(
        decoded.iter().any(|s| *s != 0),
        "quantised PCM is all zeros: the f32 waveform had energy but 16-bit conversion lost it, \
         which is silence in the file the user receives"
    );

    // --- level against the oracle: recorded, not gated (see the module docs) --------------------
    let level = if fixtures.has_seam(&waveform_seam) {
        match fixtures.seam(&waveform_seam, "tensor", 0) {
            Ok(reference) => {
                let ours = rms(&pcm);
                let theirs = rms(&reference.data);
                if theirs > 0.0 {
                    format!(", rms ratio vs oracle {:.4}", ours / theirs)
                } else {
                    ", oracle waveform is silent".to_owned()
                }
            }
            Err(_) => String::new(),
        }
    } else {
        String::new()
    };

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("AudioTail")
        .seam("codec_decoder.input.input")
        .reason(format!(
            "{frames} frame(s) of codes -> {} samples at {SAMPLE_RATE_HZ} Hz through our codec, \
             mean square energy {energy:e}, {} WAV bytes{level}",
            pcm.len(),
            wav.len()
        ))
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}
