//! ICL reference-prefix decode: priming equals the official concatenate-then-cut, BIT FOR BIT
//! (bead frankentts-5yl, codec spec §5.3).
//!
//! The official ICL path prepends the reference codec codes to the generated codes, decodes the
//! concatenation as one sequence, and cuts `round(ref_len / total_len * wav_len)` samples in the
//! waveform domain. Spec §5.3's derived paragraph shows that on this strictly causal
//! 1,920-samples-per-frame decoder the proportional rule collapses to `ref_len * 1920` exactly —
//! so priming a streaming state with the reference and pushing only the generated frames must
//! reproduce the concatenated decode's tail with zero divergence. That equivalence is what this
//! suite pins, plus the `.ftvoice-cache` seam: a primed state, cloned, keeps working — the
//! reference decode happens once per voice, not once per utterance.
//!
//! Codes are deterministic synthetic ids (xorshift over the 2,048-way codebooks): the identity
//! under test is a property of the DECODER's state machine, not of any particular utterance, and
//! synthetic ids exercise it as strongly as sampled ones. Model-gated: without the pinned
//! speech-tokenizer checkpoint the tests report the skip and pass.

use std::path::PathBuf;

use ftts_model_qwen::checkpoint::CodecCheckpoint;

const CODE_GROUPS: usize = 16;
const SAMPLES_PER_FRAME: usize = 1_920;
const REFERENCE_FRAMES: usize = 8;
const GENERATED_FRAMES: usize = 12;

fn checkpoint_path() -> Option<PathBuf> {
    let root = std::env::var("FTTS_MODEL_DIR").map_or_else(
        |_| {
            #[allow(deprecated)]
            std::env::home_dir().map(|home| home.join(".cache/franken_tts/model"))
        },
        |dir| Some(PathBuf::from(dir)),
    )?;
    let path = root.join("speech_tokenizer/model.safetensors");
    path.is_file().then_some(path)
}

/// Deterministic valid code ids, frames-major `[frames * 16]`.
fn synthetic_codes(frames: usize, seed: u64) -> Vec<i32> {
    let mut state = seed;
    (0..frames * CODE_GROUPS)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 33) % 2_048) as i32
        })
        .collect()
}

fn first_divergence(left: &[f32], right: &[f32]) -> Option<usize> {
    assert_eq!(
        left.len(),
        right.len(),
        "lengths must match before comparing"
    );
    left.iter()
        .zip(right.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits())
}

#[test]
fn primed_stream_equals_the_concatenated_decode_tail_bit_for_bit() {
    let Some(path) = checkpoint_path() else {
        eprintln!(
            "receipt: {{\"test\":\"icl_prefix_decode\",\"outcome\":\"skipped\",\
             \"reason\":\"speech_tokenizer checkpoint unavailable\"}}"
        );
        return;
    };
    let checkpoint = CodecCheckpoint::load(&path).expect("codec checkpoint loads");

    let reference = synthetic_codes(REFERENCE_FRAMES, 0xA5A5_5A5A_DEAD_BEEF);
    let generated = synthetic_codes(GENERATED_FRAMES, 0x0123_4567_89AB_CDEF);
    let mut concatenated = reference.clone();
    concatenated.extend_from_slice(&generated);

    // The official shape: decode the concatenation whole, cut ref_len * 1920 in the waveform.
    let whole = checkpoint
        .decode(&concatenated, REFERENCE_FRAMES + GENERATED_FRAMES)
        .expect("concatenated decode");
    assert_eq!(
        whole.len(),
        (REFERENCE_FRAMES + GENERATED_FRAMES) * SAMPLES_PER_FRAME,
        "spec §5.3 premise: exactly 1920 samples per frame, so the proportional cut is exact"
    );
    let official_tail = &whole[REFERENCE_FRAMES * SAMPLES_PER_FRAME..];

    // Ours: prime with the reference, then push the generated frames in uneven packets.
    let mut state = checkpoint.stream_state();
    let discarded = checkpoint
        .stream_prime_reference(&mut state, &reference, REFERENCE_FRAMES)
        .expect("prime");
    assert_eq!(discarded, REFERENCE_FRAMES * SAMPLES_PER_FRAME, "cut size");

    let mut ours = Vec::new();
    let mut packet = Vec::new();
    for (start, len) in [(0usize, 1usize), (1, 4), (5, 7)] {
        packet.clear();
        checkpoint
            .stream_push(
                &mut state,
                &generated[start * CODE_GROUPS..(start + len) * CODE_GROUPS],
                len,
                &mut packet,
            )
            .expect("push");
        ours.extend_from_slice(&packet);
    }

    assert_eq!(ours.len(), official_tail.len(), "sample counts");
    let divergence = first_divergence(official_tail, &ours);
    assert_eq!(divergence, None, "first divergent sample at {divergence:?}");
    eprintln!(
        "receipt: {{\"test\":\"icl_prefix_decode\",\"case\":\"prefix_identity\",\
         \"outcome\":\"passed\",\"samples_compared\":{},\"discarded\":{discarded}}}",
        ours.len()
    );
}

#[test]
fn a_cloned_primed_state_serves_repeated_utterances_identically() {
    let Some(path) = checkpoint_path() else {
        eprintln!(
            "receipt: {{\"test\":\"icl_prefix_decode\",\"outcome\":\"skipped\",\
             \"reason\":\"speech_tokenizer checkpoint unavailable\"}}"
        );
        return;
    };
    let checkpoint = CodecCheckpoint::load(&path).expect("codec checkpoint loads");

    let reference = synthetic_codes(REFERENCE_FRAMES, 0xFEED_FACE_CAFE_F00D);
    let mut primed = checkpoint.stream_state();
    checkpoint
        .stream_prime_reference(&mut primed, &reference, REFERENCE_FRAMES)
        .expect("prime");

    // Two different "utterances" decoded from clones of ONE primed snapshot must each equal a
    // freshly primed decode — the .ftvoice-cache contract: prime once per voice, clone per
    // utterance, never re-decode the reference.
    for seed in [0x1111_2222_3333_4444u64, 0x5555_6666_7777_8888] {
        let generated = synthetic_codes(GENERATED_FRAMES, seed);

        let mut from_snapshot = primed.clone();
        let mut snapshot_pcm = Vec::new();
        checkpoint
            .stream_push(
                &mut from_snapshot,
                &generated,
                GENERATED_FRAMES,
                &mut snapshot_pcm,
            )
            .expect("push from snapshot");

        let mut fresh = checkpoint.stream_state();
        checkpoint
            .stream_prime_reference(&mut fresh, &reference, REFERENCE_FRAMES)
            .expect("fresh prime");
        let mut fresh_pcm = Vec::new();
        checkpoint
            .stream_push(&mut fresh, &generated, GENERATED_FRAMES, &mut fresh_pcm)
            .expect("push from fresh");

        assert_eq!(snapshot_pcm.len(), fresh_pcm.len(), "sample counts");
        let divergence = first_divergence(&fresh_pcm, &snapshot_pcm);
        assert_eq!(
            divergence, None,
            "snapshot decode diverged at {divergence:?}"
        );
    }
    eprintln!(
        "receipt: {{\"test\":\"icl_prefix_decode\",\"case\":\"snapshot_reuse\",\
         \"outcome\":\"passed\",\"utterances\":2}}"
    );
}
