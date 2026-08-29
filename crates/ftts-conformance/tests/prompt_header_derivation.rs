//! The x-vector prompt header, derived from checkpoint tensors rather than stubbed.
//!
//! `engine_e2e_ft7` proves the decode loop by feeding it the oracle's *captured* prefill; its
//! header is `vec![0.0; HIDDEN]` and it says so. That is the right scope for a decode-loop
//! receipt, but it leaves the question `ftts say` actually has to answer unasked: can we build
//! that prefill ourselves, from the pinned checkpoint, without a capture to copy?
//!
//! This test answers it. [`TalkerCheckpoint::xvector_header`] plus [`assemble_prompt`] must
//! reproduce the reference's own `talker.input.input/kwargs.inputs_embeds`, row for row, given
//! only:
//!
//! * pinned checkpoint tensors (`text_embedding`, `text_projection`, `codec_embedding`),
//! * pinned token ids from `config.json`, and
//! * the fixture's captured speaker embedding — the one input a checkpoint cannot supply, because
//!   it is a property of the reference *voice*. The ECAPA encoder that computes it from audio is
//!   `frankentts-p1-speaker-ga6` and is not implemented; until it is, a caller supplies the vector.
//!
//! The header composition being checked is `[think, think_bos, language, think_eos, speaker,
//! codec_pad, codec_bos]`. Each element was identified by solving the captured prefill against
//! candidate atoms, not by reading it off a spec — so this test is the evidence for that claim and
//! must fail if any element drifts.
//!
//! The trailing text stream is checked too. It is what feeds one hidden per generated frame; a
//! prefill that matched while the trailing stream did not would produce a correct first frame and
//! then diverge, which is the most expensive kind of wrong.
//!
//! Model-gated twice (fixture pack + main checkpoint); either absent produces a loud skip receipt.

#![cfg(feature = "ultra-tests")]

use ftts_conformance::{
    npy,
    oracle::{CPU_TIER_TOLERANCE, CPU_TIER_TOLERANCE_SOURCE, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_model_qwen::checkpoint::{CODEC_LANGUAGE_ENGLISH_ID, TALKER_HIDDEN, TalkerCheckpoint};
use ftts_model_qwen::prompt::{
    CloneMode, PromptAssemblyInput, PromptMode, assemble_prompt, extract_prompt_text_ids,
};
use std::path::{Path, PathBuf};

const TEST_NAME: &str = "prompt_header_derives_the_reference_prefill";
const CASE: &str = "synthetic-tone-en";

/// Both x-vector geometries: the header is shared, the tail differs.
const MODES: [(&str, bool); 2] = [
    ("xvector_streaming", false),
    ("xvector_non_streaming", true),
];

/// The reference's captured tensors are bf16-widened f32; our projection recomputes them in f32.
/// Rounding differences of a few ulps at 1e-3 magnitudes are expected and are not the drift this
/// test is looking for — a wrong *atom* is off by whole units, as the probe that identified them
/// showed (nearest wrong candidate: 0.097 against 7e-7 for the right one).
const ROW_TOLERANCE: f32 = 2e-3;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("PromptAssembly")
        .seam("talker.input.input")
        .reason(reason)
        .tolerance(f64::from(ROW_TOLERANCE), CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

#[test]
fn prompt_header_derives_the_reference_prefill() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("fixtures unavailable: {error}"));
            return;
        }
    };
    let path = checkpoint_path();
    if !path.is_file() {
        skip(&format!("checkpoint absent at {}", path.display()));
        return;
    }

    let first_mode = SeamRef {
        case: CASE,
        mode: MODES[0].0,
        group: "prompt_build",
        seam: "prompt.text_ids",
    };
    if !fixtures.has_seam(&first_mode) {
        skip("prompt_build seams absent from the fixture pack");
        return;
    }

    let checkpoint = match TalkerCheckpoint::load(&path) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            skip(&format!("checkpoint unusable: {error}"));
            return;
        }
    };

    let mut summaries = Vec::new();

    for (mode_name, non_streaming) in MODES {
        let ids_seam = SeamRef {
            case: CASE,
            mode: mode_name,
            group: "prompt_build",
            seam: "prompt.text_ids",
        };
        let speaker_seam = SeamRef {
            case: CASE,
            mode: mode_name,
            group: "prompt_build",
            seam: "prompt.speaker_embedding",
        };
        let input_seam = SeamRef {
            case: CASE,
            mode: mode_name,
            group: "talker_free_running",
            seam: "talker.input.input",
        };

        // The reference's own wrapped ids. Using these rather than re-tokenizing keeps this test
        // about prompt *assembly*; tokenizer parity is `prompt_l0`'s job.
        let wrapped: Vec<u32> = npy::read_i64(&fixtures.seam_path(&ids_seam, "tensor", 0))
            .expect("prompt text ids")
            .data
            .iter()
            .map(|id| u32::try_from(*id).expect("token id fits u32"))
            .collect();
        let speaker = fixtures
            .seam(&speaker_seam, "tensor", 0)
            .expect("speaker embedding")
            .data;
        assert_eq!(
            speaker.len(),
            TALKER_HIDDEN,
            "the captured x-vector must be one talker-hidden row"
        );
        let expected = fixtures
            .seam(&input_seam, "kwargs.inputs_embeds", 0)
            .expect("reference prefill");
        let expected_rows: Vec<&[f32]> = expected
            .data
            .as_chunks::<TALKER_HIDDEN>()
            .0
            .iter()
            .map(<[f32; TALKER_HIDDEN]>::as_slice)
            .collect();

        // --- our derivation ---------------------------------------------------------------
        let table = checkpoint
            .gather_text_rows(&TalkerCheckpoint::utterance_text_ids(&wrapped))
            .expect("gather text rows");
        assert!(
            table.covers(&wrapped),
            "every wrapped id must be materialized; an ungathered row reads as a zero embedding"
        );
        let header = checkpoint
            .xvector_header(&table, &speaker, CODEC_LANGUAGE_ENGLISH_ID)
            .expect("x-vector header");
        let tts_eos = checkpoint.tts_eos(&table);

        let ids = extract_prompt_text_ids(&wrapped, None).expect("wrapper strip");
        let target_text = checkpoint.project_text_ids(&table, &ids.target);

        let assembly = assemble_prompt(PromptAssemblyInput {
            mode: PromptMode {
                clone_mode: CloneMode::XVector,
                non_streaming_mode: non_streaming,
            },
            header,
            target_text,
            reference_text: None,
            reference_codec: None,
            tts_eos,
            hold_tts_eos: false,
        })
        .expect("assemble prompt");

        // --- row-for-row comparison -------------------------------------------------------
        assert_eq!(
            assembly.prefill.len(),
            expected_rows.len(),
            "{mode_name}: derived prefill has {} positions, the reference has {}",
            assembly.prefill.len(),
            expected_rows.len()
        );
        let mut worst = 0.0f32;
        for (index, (ours, theirs)) in assembly.prefill.iter().zip(&expected_rows).enumerate() {
            assert_eq!(ours.len(), TALKER_HIDDEN, "{mode_name}: row {index} width");
            let error = ours
                .iter()
                .zip(theirs.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                error <= ROW_TOLERANCE,
                "{mode_name}: prefill row {index} differs from the reference by {error:e}; a \
                 wrong header atom is off by whole units, so this is a composition error, not \
                 rounding"
            );
            worst = worst.max(error);
        }

        // --- the trailing stream, which feeds one hidden per generated frame ---------------
        let expected_trailing = fixtures
            .seam(&input_seam, "kwargs.trailing_text_hidden", 0)
            .expect("reference trailing text hidden");
        let trailing_rows: Vec<&[f32]> = expected_trailing
            .data
            .as_chunks::<TALKER_HIDDEN>()
            .0
            .iter()
            .map(<[f32; TALKER_HIDDEN]>::as_slice)
            .collect();
        assert_eq!(
            assembly.trailing_text_hidden.len(),
            trailing_rows.len(),
            "{mode_name}: trailing text stream length"
        );
        for (index, (ours, theirs)) in assembly
            .trailing_text_hidden
            .iter()
            .zip(&trailing_rows)
            .enumerate()
        {
            let error = ours
                .iter()
                .zip(theirs.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                error <= ROW_TOLERANCE,
                "{mode_name}: trailing row {index} differs by {error:e}; the prefill matching \
                 while this does not means a correct first frame and divergence after it"
            );
            worst = worst.max(error);
        }

        summaries.push(format!(
            "{mode_name}: {} prefill + {} trailing rows, max abs {worst:e}",
            assembly.prefill.len(),
            assembly.trailing_text_hidden.len()
        ));
    }

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("PromptAssembly")
        .seam("talker.input.input")
        .reason(summaries.join("; "))
        .tolerance(f64::from(ROW_TOLERANCE), CPU_TIER_TOLERANCE_SOURCE)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();

    // Keep the shared tolerance constant referenced so a floor change is a compile-time visit.
    let _ = CPU_TIER_TOLERANCE;
}
