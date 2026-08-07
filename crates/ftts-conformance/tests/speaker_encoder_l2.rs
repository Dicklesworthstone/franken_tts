//! L2 parity: the ECAPA-TDNN speaker encoder against the CPU-fp32 oracle, with real weights.
//!
//! The oracle captured this stack at six seams — the mel it was handed, each of the four blocks'
//! input *and* output, and the 1024-d embedding — so every block is fed the **oracle's own input**
//! and compared against the oracle's own output. That decoupling is the point: a block-2 failure
//! means block 2, not accumulated drift from block 0. The whole-stack run is then checked
//! separately, from the mel alone, which is what actually ships.
//!
//! # Why this seam is graded harder than the talker's
//!
//! `docs/QWEN3_TTS_SPEAKER_ENCODER_SPEC.md` §3: the x-vector is injected into the talker as one
//! raw token position — **no projection, no normalization, no scaling**. There is nothing
//! downstream to absorb error, so drift here lands in the talker's input sequence at full scale.
//! The embedding is therefore checked three ways: absolute divergence, cosine (which is what
//! speaker identity actually is), and against `prompt.speaker_embedding`, proving the vector the
//! prompt consumes is the vector this encoder produces.
//!
//! # Claim tier
//!
//! Whether this reaches bit-exact is *measured*, not assumed: `..._bitwise_cpu_fp32` asserts
//! exactness under [`xfail`], so a regression away from exact and an unexpected arrival at exact
//! are both caught. The graded gate below it is scale-relative and tight enough that a wiring
//! error — a reordered MFA concatenation, a Res2Net chain carrying the wrong term, a dropped
//! squeeze-excitation gate — cannot pass it, because every one of those is an O(1) relative move
//! while f32 reduction-order drift over these K values is not.

use ftts_conformance::{
    compare::compare_f32,
    oracle::{CPU_FP32_ORACLE_CLASS, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
    xfail,
};
use ftts_model_qwen::speaker::{ENC_CHANNELS, ENC_DIM, Encoder, MEL_DIM, SE_RES2NET_BLOCKS};
use std::path::{Path, PathBuf};

const BLOCK_TEST: &str = "contract_a_l2_speaker_encoder_blocks_00_03_cpu_fp32";
const EMBEDDING_TEST: &str = "contract_a_l2_speaker_encoder_embedding_cpu_fp32";
const PROMPT_TEST: &str = "contract_a_l2_speaker_encoder_feeds_prompt_speaker_embedding";
const REORDER_TEST: &str = "contract_a_l2_speaker_encoder_gate_rejects_a_reordered_mfa";
const BITWISE_TEST: &str = "contract_a_l2_speaker_encoder_bitwise_cpu_fp32";
const CONTRACT: &str = "ConformanceExact/L2";
const CASE: &str = "synthetic-tone-en";
const GROUP: &str = "prompt_build";

/// The encoder runs identically in every mode; `icl_*` is where an x-vector is actually derived
/// from reference audio rather than supplied precomputed, so it is the binding capture.
const MODE: &str = "icl_non_streaming";

const LEDGER: &str =
    "frankentts-p1-speaker-ga6 (bead comment: ECAPA f32 reduction order vs CPU tier)";

/// Scale-relative bound, ratcheted to the measured floor rather than to the rounding budget.
///
/// The budget would allow far more: the widest reduction in the stack is `asp.tdnn` at K = 4608,
/// where `sqrt(K) * eps` is about 8.1e-6 relative. What is actually measured is two decades below
/// that, because the bias-seeded im2col GEMM this engine issues is the reference's own:
///
/// ```text
///   block_0     max_abs 0.0        relative 0.0        0 of 47_616 inexact   <- bit-exact
///   block_1     max_abs 1.192e-6   relative 5.610e-8   6_601 of 47_616
///   block_2     max_abs 9.537e-7   relative 4.488e-8   2_150 of 47_616
///   block_3     max_abs 3.815e-6   relative 1.515e-7   3_770 of 47_616
///   embedding   max_abs 4.768e-7   relative 8.159e-8     941 of 1_024
/// ```
///
/// `block_0` reaching exact is the load-bearing row: it is the only block that is a bare
/// convolution, so it isolates the padding split, the reflect indexing and the GEMM seeding from
/// everything the SE-Res2Net blocks add. The residual divergence in blocks 1-3 therefore belongs
/// to the reductions those blocks introduce — the squeeze-excitation channel mean and the Res2Net
/// chain — not to the convolution path.
///
/// So the gate is set at 1e-6: about 7x the worst measured row, still six decades below the O(1)
/// relative move any wiring error in this stack produces. Off macOS the GEMM request degrades to
/// the scalar reduction and these numbers loosen; the fixture pack this test needs is only
/// captured for the CPU-fp32 tier, so that configuration skips rather than fails.
const RELATIVE_BOUND: f64 = 1e-6;

/// A wiring error collapses cosine; f32 rounding does not move it below this.
const COSINE_FLOOR: f64 = 0.999_999;

/// The embedding is consumed unnormalized, so its cosine is held tighter than the blocks'.
const EMBEDDING_COSINE_FLOOR: f64 = 0.999_999_9;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn seam(name: &str) -> SeamRef<'_> {
    SeamRef {
        case: CASE,
        mode: MODE,
        group: GROUP,
        seam: name,
    }
}

fn skip(test: &str, seam_name: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam(seam_name)
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// PyTorch-major `(1, channels, frames)` to this engine's time-major `[frames, channels]`.
///
/// Every fixture in this stage except `speaker_encoder.input` — which the reference captures
/// *before* its own `transpose(1, 2)` and so is already time-major — needs this.
fn to_time_major(data: &[f32], channels: usize, frames: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; frames * channels];
    for channel in 0..channels {
        for frame in 0..frames {
            out[frame * channels + channel] = data[channel * frames + frame];
        }
    }
    out
}

/// One seam's measured divergence, expressed against the magnitude of what it was compared to.
struct Divergence {
    label: String,
    max_abs: f64,
    scale: f64,
    cosine: f64,
    over_tolerance: usize,
    len: usize,
}

impl Divergence {
    fn measure(label: impl Into<String>, expected: &[f32], actual: &[f32]) -> Self {
        let comparison = compare_f32(expected, actual, f64::INFINITY);
        Self {
            label: label.into(),
            max_abs: comparison.max_abs_diff,
            scale: expected
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(f64::from(value.abs()))),
            cosine: comparison.cosine,
            over_tolerance: compare_f32(expected, actual, 0.0).over_tolerance,
            len: comparison.len,
        }
    }

    fn relative(&self) -> f64 {
        if self.scale > 0.0 {
            self.max_abs / self.scale
        } else {
            self.max_abs
        }
    }

    fn describe(&self) -> String {
        format!(
            "{}: max_abs {:.3e} over scale {:.3e} = {:.3e} relative, cosine {:.9}, {} of {} inexact",
            self.label,
            self.max_abs,
            self.scale,
            self.relative(),
            self.cosine,
            self.over_tolerance,
            self.len
        )
    }
}

/// Everything the gates below need, or the reason it could not be produced.
struct Measured {
    blocks: Vec<Divergence>,
    embedding: Divergence,
    prompt: Divergence,
    /// The same stack with the MFA concatenation reversed — the negative control for the gate.
    mfa_reordered: Divergence,
}

fn measure() -> Result<Measured, String> {
    let fixtures = OracleFixtures::open_default()
        .map_err(|error| format!("oracle fixtures unavailable: {error}"))?;
    fixtures
        .require_oracle_class(CPU_FP32_ORACLE_CLASS)
        .map_err(|error| format!("fixture pack is not the CPU-fp32 tier: {error}"))?;
    let path = checkpoint_path();
    if !path.is_file() {
        return Err(format!("checkpoint absent at {}", path.display()));
    }
    let encoder = Encoder::load(&path)
        .map_err(|error| format!("cannot hydrate the speaker encoder: {error}"))?;

    // The mel, as the reference received it: `(1, frames, 128)`, already time-major.
    let mel_seam = seam("speaker_encoder.input");
    if !fixtures.has_seam(&mel_seam) {
        return Err(format!("`{}` is not in this pack", mel_seam.describe()));
    }
    let mel = fixtures
        .seam(&mel_seam, "args.0", 0)
        .map_err(|error| format!("cannot read {}: {error}", mel_seam.describe()))?;
    if mel.data.len() % MEL_DIM != 0 {
        return Err(format!(
            "{} holds {} floats, not a multiple of {MEL_DIM}",
            mel_seam.describe(),
            mel.data.len()
        ));
    }
    let frames = mel.data.len() / MEL_DIM;
    if frames == 0 {
        return Err("the captured mel is empty".to_owned());
    }

    // Each block against the oracle's own input, so failures localize.
    let mut blocks = Vec::with_capacity(SE_RES2NET_BLOCKS + 1);
    // `index` is the block number the oracle named its seams after, not merely a cursor into
    // `ENC_CHANNELS` — it selects the entry point too, so iterating the channel array would lose
    // the thing the loop is actually keyed on.
    #[allow(clippy::needless_range_loop)]
    for index in 0..=SE_RES2NET_BLOCKS {
        let input_name = format!("speaker_encoder.block_{index}.input");
        let output_name = format!("speaker_encoder.block_{index}.output");
        let input_seam = seam(&input_name);
        let output_seam = seam(&output_name);
        for reference in [&input_seam, &output_seam] {
            if !fixtures.has_seam(reference) {
                return Err(format!("`{}` is not in this pack", reference.describe()));
            }
        }
        let input = fixtures
            .seam(&input_seam, "args.0", 0)
            .map_err(|error| format!("cannot read {}: {error}", input_seam.describe()))?;
        let expected = fixtures
            .seam(&output_seam, "tensor", 0)
            .map_err(|error| format!("cannot read {}: {error}", output_seam.describe()))?;

        // Block 0 consumes mel bins; the SE-Res2Net blocks consume 512 channels.
        let in_channels = if index == 0 {
            MEL_DIM
        } else {
            ENC_CHANNELS[index]
        };
        let expected_len = frames * in_channels;
        if input.data.len() != expected_len {
            return Err(format!(
                "{} holds {} floats, expected {expected_len}",
                input_seam.describe(),
                input.data.len()
            ));
        }
        let time_major = to_time_major(&input.data, in_channels, frames);
        let ours = if index == 0 {
            encoder.initial_block(&time_major, frames)
        } else {
            encoder.se_res2net_block(index, &time_major, frames)
        };
        let expected = to_time_major(&expected.data, ENC_CHANNELS[index], frames);
        blocks.push(Divergence::measure(
            format!("block_{index}"),
            &expected,
            &ours,
        ));
    }

    // The whole stack, from the mel alone — the path that actually ships.
    let ours = encoder.encode(&mel.data, frames);
    let output_seam = seam("speaker_encoder.output");
    let expected = fixtures
        .seam(&output_seam, "tensor", 0)
        .map_err(|error| format!("cannot read {}: {error}", output_seam.describe()))?;
    if expected.data.len() != ENC_DIM {
        return Err(format!(
            "{} holds {} floats, expected {ENC_DIM}",
            output_seam.describe(),
            expected.data.len()
        ));
    }
    let embedding = Divergence::measure("embedding", &expected.data, &ours);

    // And that this is the vector the prompt actually consumes.
    let prompt_seam = seam("prompt.speaker_embedding");
    let prompt_expected = fixtures
        .seam(&prompt_seam, "tensor", 0)
        .map_err(|error| format!("cannot read {}: {error}", prompt_seam.describe()))?;
    let prompt = Divergence::measure("prompt.speaker_embedding", &prompt_expected.data, &ours);

    // Negative control. Reversing the MFA concatenation is the trap §2 of the spec records: it
    // still loads, still produces a well-scaled 1024-d vector, and is wrong. Measuring it here is
    // what proves the gate above discriminates rather than merely being satisfiable.
    let h0 = encoder.initial_block(&mel.data, frames);
    let h1 = encoder.se_res2net_block(1, &h0, frames);
    let h2 = encoder.se_res2net_block(2, &h1, frames);
    let h3 = encoder.se_res2net_block(3, &h2, frames);
    let reordered = encoder.aggregate(&h3, &h2, &h1, frames);
    let mfa_reordered = Divergence::measure("mfa_reordered", &expected.data, &reordered);

    Ok(Measured {
        blocks,
        embedding,
        prompt,
        mfa_reordered,
    })
}

fn emit(test: &str, seam_name: &str, outcome: Outcome, detail: &[&Divergence]) {
    let rows: Vec<serde_json::Value> = detail
        .iter()
        .map(|divergence| {
            serde_json::json!({
                "seam": divergence.label,
                "max_abs_diff": divergence.max_abs,
                "scale": divergence.scale,
                "relative": divergence.relative(),
                "cosine": divergence.cosine,
                "inexact": divergence.over_tolerance,
                "compared": divergence.len,
            })
        })
        .collect();
    Receipt::new(test, outcome)
        .contract(CONTRACT)
        .seam(seam_name)
        .tolerance(
            RELATIVE_BOUND,
            "scale-relative, ratcheted to the measured floor (worst row 1.515e-7 at block_3)",
        )
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .detail(serde_json::json!({ "mode": MODE, "seams": rows }))
        .emit();
}

/// Every block, fed the oracle's own input.
#[test]
fn contract_a_l2_speaker_encoder_blocks_00_03_cpu_fp32() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(BLOCK_TEST, "speaker_encoder.block_N.output", &reason);
            return;
        }
    };

    let failures: Vec<String> = measured
        .blocks
        .iter()
        .filter(|block| block.relative() > RELATIVE_BOUND || block.cosine < COSINE_FLOOR)
        .map(Divergence::describe)
        .collect();

    let outcome = if failures.is_empty() {
        Outcome::Passed
    } else {
        Outcome::Failed
    };
    emit(
        BLOCK_TEST,
        "speaker_encoder.block_N.output",
        outcome,
        &measured.blocks.iter().collect::<Vec<_>>(),
    );

    assert!(
        failures.is_empty(),
        "speaker encoder blocks diverged beyond a rounding scale:\n  {}",
        failures.join("\n  ")
    );
}

/// The whole stack, mel to x-vector.
#[test]
fn contract_a_l2_speaker_encoder_embedding_cpu_fp32() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(EMBEDDING_TEST, "speaker_encoder.output", &reason);
            return;
        }
    };

    let embedding = &measured.embedding;
    let passed =
        embedding.relative() <= RELATIVE_BOUND && embedding.cosine >= EMBEDDING_COSINE_FLOOR;
    emit(
        EMBEDDING_TEST,
        "speaker_encoder.output",
        if passed {
            Outcome::Passed
        } else {
            Outcome::Failed
        },
        &[embedding],
    );

    assert!(
        passed,
        "the x-vector is injected with no normalization downstream, so this bound is the \
         identity itself — {}",
        embedding.describe()
    );
}

/// The embedding this encoder produces is the one the prompt consumes.
///
/// `speaker_encoder.output` and `prompt.speaker_embedding` are captured at different points; that
/// they agree is what proves no reshape, cast or normalization sits between them, which §3 of the
/// spec claims and this asserts.
#[test]
fn contract_a_l2_speaker_encoder_feeds_prompt_speaker_embedding() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(PROMPT_TEST, "prompt.speaker_embedding", &reason);
            return;
        }
    };

    let prompt = &measured.prompt;
    let embedding = &measured.embedding;
    // The two captures must be the same tensor, so our divergence against them must match.
    let passed = prompt.relative() <= RELATIVE_BOUND
        && prompt.cosine >= EMBEDDING_COSINE_FLOOR
        && (prompt.max_abs - embedding.max_abs).abs() < f64::EPSILON;
    emit(
        PROMPT_TEST,
        "prompt.speaker_embedding",
        if passed {
            Outcome::Passed
        } else {
            Outcome::Failed
        },
        &[prompt, embedding],
    );

    assert!(
        passed,
        "the prompt's speaker embedding is not this encoder's output unchanged — {} vs {}",
        prompt.describe(),
        embedding.describe()
    );
}

/// The gate discriminates: the MFA-order trap must fail it by decades.
///
/// A tolerance is only evidence if something realistic fails it. Reversing `cat(h1, h2, h3)` is
/// the specific error §2 of the spec warns about — it loads, it runs, it produces a 1024-d vector
/// of ordinary magnitude — so if the bound above could not tell that apart from the real
/// embedding, the bound would be decoration.
#[test]
fn contract_a_l2_speaker_encoder_gate_rejects_a_reordered_mfa() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(REORDER_TEST, "speaker_encoder.output", &reason);
            return;
        }
    };

    let reordered = &measured.mfa_reordered;
    // Two independent ways of being wrong: past the scale-relative bound, and off the cosine
    // floor. Cosine is the one that matters — it is speaker identity — and a permutation of the
    // aggregated channels moves it far more than any rounding can.
    let rejected = reordered.relative() > RELATIVE_BOUND && reordered.cosine < COSINE_FLOOR;
    emit(
        REORDER_TEST,
        "speaker_encoder.output",
        if rejected {
            Outcome::Passed
        } else {
            Outcome::Failed
        },
        &[reordered, &measured.embedding],
    );

    assert!(
        rejected,
        "a reversed MFA concatenation passed the parity gate, so the gate proves nothing — {}\n\
         (the correctly ordered stack measures {})",
        reordered.describe(),
        measured.embedding.describe()
    );
}

/// Whether this stack is bit-exact is measured, not assumed.
#[test]
fn contract_a_l2_speaker_encoder_bitwise_cpu_fp32() {
    let measured = match measure() {
        Ok(measured) => measured,
        Err(reason) => {
            skip(BITWISE_TEST, "speaker_encoder.output", &reason);
            return;
        }
    };

    // `xfail` panics if this ever returns `Ok` — arriving at exact is a claim upgrade that must
    // retire the ledger entry, not a silent pass.
    xfail(BITWISE_TEST, CONTRACT, LEDGER, || {
        let inexact: Vec<String> = measured
            .blocks
            .iter()
            .chain(std::iter::once(&measured.embedding))
            .filter(|divergence| divergence.over_tolerance > 0)
            .map(Divergence::describe)
            .collect();
        if inexact.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "the speaker encoder is not bit-exact against the CPU-tier oracle. The widest \
                 reduction in the stack is `asp.tdnn` at K=4608, where an f32 accumulation costs \
                 about sqrt(K)*eps = {:.3e} relative:\n  {}",
                4608.0_f64.sqrt() * f64::from(f32::EPSILON),
                inexact.join("\n  ")
            ))
        }
    });
}
