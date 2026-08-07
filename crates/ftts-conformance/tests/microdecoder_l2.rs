//! L2 parity: the microdecoder's OQ-5 index map against the CPU-fp32 oracle seams.
//!
//! A stage without a fixture comparison is not done. This is that comparison for the piece of
//! `ftts_model_qwen::microdecoder` that can be checked **without the checkpoint**: the head-to-code
//! wiring and the greedy selection rule.
//!
//! # What this proves, and why it is worth a test of its own
//!
//! The oracle captures, per frame, fifteen per-depth head outputs (`microdecoder.head_01..head_15`)
//! and the sixteen codes the frame produced (`talker.codec_codes` = `c0..c15`). The OQ-5 contract
//! says head `d - 1` scores position `d` and produces `c_d` — so `argmax(head_d.output)` must equal
//! `codec_codes[d]`, for every depth and every prompt mode.
//!
//! That single relation pins the two traps the contract calls out:
//!
//! * **Position 1 is scored.** If the port treated `c0`'s slot as a second conditioning position,
//!   every head would be off by one and `head_01` would have to match `c0` (1995) instead of `c1`
//!   (910). It does not.
//! * **Head placement.** An off-by-one in either direction breaks the relation at every depth at
//!   once, so this fails loudly rather than degrading audio quietly.
//!
//! It also confirms the per-depth head vocabulary is 2048, not the talker's 3072 — the two tables
//! are different widths and conflating them is silent.
//!
//! # Claim tier
//!
//! This is an **exact token-id comparison**, which is precisely what OQ-5 §6 licenses: the
//! sequential loop and a batched forward agree in exact arithmetic only, so strict acceptance
//! compares argmax/token ids, never logit bits. No tolerance is involved and none is needed.
//!
//! What this test does **not** prove: that our layers reproduce the oracle's hidden states. That
//! needs the checkpoint and is the weight-gated half of `frankentts-p1-microdecoder-xst`.

use ftts_conformance::{
    npy,
    oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_model_qwen::microdecoder::{
    FRAME_POSITIONS, PositionRole, RESIDUAL_DEPTHS, RESIDUAL_VOCAB, argmax, position_role,
};

const TEST_NAME: &str = "contract_a_l2_microdecoder_head_to_code_map_cpu_fp32_exact";
const CONTRACT: &str = "ConformanceExact/L2";
const CASE: &str = "synthetic-tone-en";
const MODES: [&str; 4] = [
    "xvector_non_streaming",
    "xvector_streaming",
    "icl_non_streaming",
    "icl_streaming",
];

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("microdecoder.head_NN.output")
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// `argmax(head_d.output) == codec_codes[d]` for every depth, in every captured prompt mode.
#[test]
fn contract_a_l2_microdecoder_head_to_code_map_cpu_fp32_exact() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(error) => {
            skip(&format!("oracle fixtures unavailable: {error}"));
            return;
        }
    };
    if let Err(error) = fixtures.require_oracle_class(CPU_FP32_ORACLE_CLASS) {
        skip(&format!("fixture pack is not the CPU-fp32 tier: {error}"));
        return;
    }

    let mut checked = 0_usize;
    for mode in MODES {
        let codes_seam = SeamRef {
            case: CASE,
            mode,
            group: "talker_free_running",
            seam: "talker.codec_codes",
        };
        if !fixtures.has_seam(&codes_seam) {
            skip(&format!("`{}` is not in this pack", codes_seam.describe()));
            return;
        }
        // Code ids are captured as int64, not float32 — they need the integer reader.
        let codes_path = fixtures.seam_path(&codes_seam, "tensor", 0);
        let codes = match npy::read_i64(&codes_path) {
            Ok(codes) => codes,
            Err(error) => {
                skip(&format!("cannot read {}: {error}", codes_seam.describe()));
                return;
            }
        };
        assert_eq!(
            codes.data.len(),
            FRAME_POSITIONS,
            "a frame is {FRAME_POSITIONS} codes (c0..c15); `{}` has {} — \
             the frame geometry disagrees with the oracle",
            codes_seam.describe(),
            codes.data.len()
        );

        for depth in 1..=RESIDUAL_DEPTHS {
            let head_seam_name = format!("microdecoder.head_{depth:02}.output");
            let head_seam = SeamRef {
                case: CASE,
                mode,
                group: "teacher_forced_frame_0000",
                seam: &head_seam_name,
            };
            if !fixtures.has_seam(&head_seam) {
                skip(&format!("`{}` is not in this pack", head_seam.describe()));
                return;
            }
            let logits = match fixtures.seam(&head_seam, "tensor", 0) {
                Ok(logits) => logits,
                Err(error) => {
                    skip(&format!("cannot read {}: {error}", head_seam.describe()));
                    return;
                }
            };

            assert_eq!(
                logits.data.len(),
                RESIDUAL_VOCAB,
                "per-depth heads are {RESIDUAL_VOCAB}-way, not the talker's 3072; `{}` has {}",
                head_seam.describe(),
                logits.data.len()
            );

            // Our own greedy rule, over the oracle's own logits: this isolates the selection and
            // the wiring from any arithmetic of ours.
            let ours = argmax(&logits.data);

            let raw = codes.data[depth];
            assert!(raw >= 0, "code id {raw} at depth {depth} is negative");
            let expected = raw as usize;

            assert_eq!(
                ours, expected,
                "OQ-5 index map broken at depth {depth} in mode `{mode}`: \
                 argmax(head_{depth:02}.output) = {ours} but codec_codes[{depth}] = {expected}. \
                 Head d-1 must score position d and produce c_d; an off-by-one here means \
                 position 1 was treated as unscored, or head placement is shifted."
            );

            // The map our implementation exposes must agree with what the fixture just proved.
            let position = depth;
            let head_from_map = match position_role(position) {
                PositionRole::Conditioning => panic!(
                    "position {position} must be scored — position 0 is the only conditioning slot"
                ),
                PositionRole::PrimaryCodeEmbedding { head }
                | PositionRole::ResidualEmbedding { head, .. } => head,
            };
            assert_eq!(
                head_from_map,
                depth - 1,
                "position_role({position}) claims head {head_from_map}, but the oracle shows \
                 head {} scores this position",
                depth - 1
            );

            checked += 1;
        }
    }

    assert_eq!(
        checked,
        MODES.len() * RESIDUAL_DEPTHS,
        "every depth in every mode must be checked"
    );

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract(CONTRACT)
        .seam("microdecoder.head_NN.output")
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({
            "modes": MODES.len(),
            "depths_per_mode": RESIDUAL_DEPTHS,
            "comparisons": checked,
            "comparison": "exact token id (argmax), per OQ-5 §6",
        }))
        .emit();
}

/// Position 0 conditions the frame and is never scored — the trap stated as its own check.
#[test]
fn contract_a_l2_microdecoder_position_zero_is_never_scored() {
    assert_eq!(position_role(0), PositionRole::Conditioning);
    let scored = (0..FRAME_POSITIONS)
        .filter(|p| !matches!(position_role(*p), PositionRole::Conditioning))
        .count();
    assert_eq!(
        scored, RESIDUAL_DEPTHS,
        "exactly {RESIDUAL_DEPTHS} of the {FRAME_POSITIONS} positions are scored"
    );
}
