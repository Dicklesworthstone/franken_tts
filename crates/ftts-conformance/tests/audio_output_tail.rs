//! The audio output tail: real model PCM through the WAV writer.
//!
//! This is the last stage of the pipeline and the one a user actually receives. It is deliberately
//! separate from `codec_decode_l2`, which owns codec *arithmetic* parity against the oracle seams:
//! this file owns the conversion from decoded samples to a playable file, and it must keep passing
//! even while the codec's exact-parity ratchet is still open.
//!
//! The PCM here is the oracle's own `codec.generated_waveform` — the waveform the reference
//! produced from the fixture's codec codes — so the assertions run against genuine model output
//! rather than a synthetic ramp. Two properties are checked, and both are ones that a byte count
//! alone would call success:
//!
//! * **Exact sample count.** The waveform length must be `frames * 1920`, with `frames` read from
//!   the shape of the codec's own input codes. Tying the two together means a frame-to-sample
//!   mistake fails here rather than producing audio of quietly wrong duration.
//! * **Nonzero energy.** A silent result has the right length, the right header, and the right byte
//!   count. Energy is the only one of these checks that can tell audio from silence.
//!
//! Model-gated: skips with SUCCESS and a named reason when the fixture pack is absent.

use ftts_conformance::{
    npy,
    oracle::{CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_core::audio::{
    SAMPLE_RATE_HZ, SAMPLES_PER_FRAME, WAV_HEADER_BYTES, WavWriter, encode_wav, mean_square_energy,
    pcm_f32_to_i16, samples_for_frames,
};
use std::io::Cursor;

const TEST_NAME: &str = "audio_tail_fixture_pcm_to_wav";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "xvector_non_streaming";

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
        .seam("codec.generated_waveform")
        .reason(reason)
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

#[test]
fn audio_tail_fixture_pcm_to_wav() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let waveform_seam = seam("codec.generated_waveform");
    let codes_seam = seam("codec_decoder.input.input");
    if !fixtures.has_seam(&waveform_seam) || !fixtures.has_seam(&codes_seam) {
        skip("codec_decode waveform/codes seams absent from the fixture pack");
        return;
    }

    // Frames come from the codes the codec was actually given: [1, 16, frames].
    let codes_path = fixtures.seam_path(&codes_seam, "args.0", 0);
    let codes = npy::read_i64(&codes_path).expect("codec input codes read");
    let frames = match codes.shape.as_slice() {
        [1, groups, frames] => {
            assert_eq!(*groups, 16, "the codec always carries 16 code groups");
            *frames
        }
        other => panic!("expected codes [1, 16, frames], got {other:?}"),
    };
    assert!(frames > 0, "the fixture must carry at least one frame");

    let waveform = fixtures
        .seam(&waveform_seam, "tensor", 0)
        .expect("generated waveform");
    let pcm = waveform.data;

    // --- exact sample count, tied to the frame count -----------------------------------------
    assert_eq!(
        pcm.len(),
        samples_for_frames(frames),
        "{frames} frames must decode to exactly {} samples at 24 kHz / 12.5 fps, got {}",
        samples_for_frames(frames),
        pcm.len()
    );
    assert_eq!(pcm.len() % SAMPLES_PER_FRAME, 0, "whole frames only");

    // --- nonzero energy: the check that distinguishes audio from a correctly-shaped silence ---
    let energy = mean_square_energy(&pcm);
    assert!(
        energy > 0.0,
        "decoded PCM is silent (mean square energy {energy:e}); a silent result has the right \
         length and the right byte count and is still a failure"
    );
    assert!(
        pcm.iter().all(|s| s.is_finite()),
        "decoded PCM contains a non-finite sample"
    );

    // --- the emitted file describes exactly what it contains ---------------------------------
    let wav = encode_wav(&pcm, SAMPLE_RATE_HZ);
    assert_eq!(
        wav.len(),
        WAV_HEADER_BYTES + pcm.len() * 2,
        "16-bit mono: two bytes per sample after a 44-byte header"
    );
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[36..40], b"data");
    let declared_data = u32::from_le_bytes(wav[40..44].try_into().expect("data size"));
    assert_eq!(
        declared_data as usize,
        pcm.len() * 2,
        "a header claiming more data than the file holds is corrupt"
    );
    let declared_riff = u32::from_le_bytes(wav[4..8].try_into().expect("riff size"));
    assert_eq!(declared_riff as usize, 36 + pcm.len() * 2);
    assert_eq!(
        u32::from_le_bytes(wav[24..28].try_into().expect("rate")),
        SAMPLE_RATE_HZ,
        "the file must declare the codec's native rate; nothing resamples"
    );

    // --- the payload round-trips: no sample is lost or reordered ------------------------------
    let expected: Vec<i16> = pcm_f32_to_i16(&pcm);
    let decoded: Vec<i16> = wav[WAV_HEADER_BYTES..]
        .chunks_exact(2)
        .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    assert_eq!(decoded, expected, "WAV payload must round-trip exactly");
    assert!(
        decoded.iter().any(|s| *s != 0),
        "quantised PCM is all zeros: the f32 waveform had energy but 16-bit conversion lost it"
    );

    // --- streaming the same audio in packets yields a byte-identical file ---------------------
    // The streaming==batch rule applied to the last stage: packetised writing must not change the
    // file, or a streamed run and an offline run of the same tokens would differ on disk.
    let mut writer = WavWriter::new(Cursor::new(Vec::new()), SAMPLE_RATE_HZ).expect("wav header");
    for packet in pcm.chunks(SAMPLES_PER_FRAME) {
        writer.write_samples(packet).expect("packet write");
    }
    assert_eq!(writer.samples_written(), pcm.len());
    let streamed = writer.finish().expect("finish").into_inner();
    assert_eq!(
        streamed, wav,
        "streamed and offline encodings of the same PCM must be byte-identical"
    );

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("AudioTail")
        .seam("codec.generated_waveform")
        .reason(&format!(
            "{frames} frame(s) -> {} samples at {SAMPLE_RATE_HZ} Hz, mean square energy {energy:e}",
            pcm.len()
        ))
        .tolerance(CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}
