//! The one-command Contract-A ladder scorecard (bead `frankentts-v-ladder-runner-zmk`).
//!
//! One `cargo test -p ftts-conformance --test ladder_runner` runs every ladder rung the harness
//! can currently drive against real Phase-1 components, emits one skip-honest receipt per rung,
//! and writes the evidence bundle — `scorecard.json`, `receipts.ndjson`, `MANIFEST.sha256` —
//! under `target/evidence/ladder-scorecard/`.
//!
//! # What each rung does today
//!
//! - **L0** — subject: our tokenizer (official regex, DISC-001 default) over the pinned corpus
//!   text, wrapped with our prompt geometry; oracle: the captured `prompt.text_ids`. Exact.
//! - **L1** — the committed floor records `not_observed`, so the runner emits the floor's own
//!   reasoned skip. The sides passed in are deliberately mismatched sentinels: if the floor ever
//!   flips to `observed` without real per-operator wiring, this rung fails loudly instead of
//!   passing vacuously.
//! - **L4** — subject: our production sampler stack (canonical greedy, repetition/min-new-tokens
//!   processors) over the oracle's own captured per-step logits; oracle: the captured primary
//!   code stream. Exact sequence compare.
//! - **L2 / L3 / L5** — explicit skips with named owners. These seams run every gate in their
//!   per-seam suites (`talker_layer_l2`, `codec_decode_l2`, …) under measured relative bounds,
//!   and are **measurably non-exact** under the committed zero envelope (talker layer 00:
//!   max_abs ≈ 9.5e-6, attributed to RMSNorm). Driving them through the ladder is the owning
//!   component beads' exit work (`frankentts-p1-talker-z2w`, `frankentts-p1-codec-hu7`,
//!   `frankentts-p1-microdecoder-xst`); recording them as skipped-with-reason keeps
//!   `all_green=false` — which is the honest Contract-A state — rather than leaving them
//!   invisible.
//!
//! The scorecard is therefore **expected non-green**; the assertions below pin each rung's
//! outcome so any drift (a floor change, a sampler regression, a tokenizer change) fails this
//! test loudly.

use std::{
    fs,
    path::{Path, PathBuf},
};

use ftts_conformance::{
    ladder::{
        EngineOutput, LadderRung, LadderRunner, cpu_fp32_fixture_manifest_path, cpu_fp32_floor_path,
    },
    npy,
    oracle::{FixtureError, OracleFixtures, SeamRef},
    report::{OracleTier, Outcome, Receipt},
};
use ftts_model_qwen::prompt::ROLE_PREFIX_IDS;
use ftts_model_qwen::sampler::{QwenSampler, SamplingMode};
use ftts_model_qwen::tokenizer::{QwenTokenizer, TokenizerFiles, TokenizerRegex};

const TEST_NAME: &str = "contract_a_ladder_scorecard_cpu_fp32";
const CASE: &str = "synthetic-tone-en";
const MODE: &str = "xvector_streaming";
const GROUP: &str = "talker_free_running";

/// `<|im_end|>` and the newline that closes a chat turn; the wrap suffix between the target text
/// and the reply-role prefix. Pinned by `prompt_l0`'s `TARGET_WRAPPED` geometry.
const IM_END_NL: [u32; 2] = [151_645, 198];

fn workspace_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn skip(reason: &str) {
    Receipt::new(TEST_NAME, Outcome::Skipped)
        .contract("ConformanceExact/ladder")
        .seam("ladder.scorecard")
        .reason(reason)
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}

/// The corpus text the capture was generated from, so the L0 subject starts from the same input
/// the oracle did rather than from a constant that could drift away from the pack.
fn corpus_text(case: &str) -> Option<String> {
    let bytes = fs::read(workspace_path("docs/conformance/oracle_corpus.json")).ok()?;
    let corpus: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    corpus["cases"]
        .as_array()?
        .iter()
        .find(|entry| entry["id"].as_str() == Some(case))?["text"]
        .as_str()
        .map(str::to_owned)
}

#[test]
fn ladder_scorecard_runs_and_writes_evidence() {
    let fixtures = match OracleFixtures::open_default() {
        Ok(fixtures) => fixtures,
        Err(FixtureError::PackAbsent { path }) => {
            skip(&format!("oracle fixture pack absent at {path}"));
            return;
        }
        Err(error) => panic!("fixture pack unusable: {error}"),
    };
    let mut runner = LadderRunner::cpu_fp32_fixture().unwrap_or_else(|error| {
        panic!(
            "fixtures are present, so the floor ({}) and manifest ({}) must both load: {error}",
            cpu_fp32_floor_path().display(),
            cpu_fp32_fixture_manifest_path().display()
        )
    });

    // L0 — tokenizer + prompt geometry vs the captured wrapped prompt ids.
    let oracle_prompt_ids: Vec<u32> = npy::read_i64(&fixtures.seam_path(
        &SeamRef {
            case: CASE,
            mode: MODE,
            group: "prompt_build",
            seam: "prompt.text_ids",
        },
        "tensor",
        0,
    ))
    .expect("wrapped prompt ids captured")
    .data
    .iter()
    .map(|&id| u32::try_from(id).expect("prompt ids are non-negative token ids"))
    .collect();
    let vocab = fs::read_to_string(workspace_path("docs/truth-pack/snapshots/hf/vocab.json"));
    let merges = fs::read_to_string(workspace_path("docs/truth-pack/snapshots/hf/merges.txt"));
    let config = fs::read_to_string(workspace_path(
        "docs/truth-pack/snapshots/hf/tokenizer_config.json",
    ));
    let text = corpus_text(CASE);
    match (&vocab, &merges, &config, &text) {
        (Ok(vocab_json), Ok(merges_txt), Ok(config_json), Some(text)) => {
            let tokenizer = QwenTokenizer::from_files(
                TokenizerFiles {
                    vocab_json,
                    merges_txt,
                    tokenizer_config_json: config_json,
                },
                TokenizerRegex::Official,
            )
            .expect("pinned tokenizer files parse");
            let text_ids = tokenizer.encode(text).expect("corpus text tokenizes");
            let mut subject_wrapped = ROLE_PREFIX_IDS.to_vec();
            subject_wrapped.extend_from_slice(&text_ids);
            subject_wrapped.extend_from_slice(&IM_END_NL);
            subject_wrapped.extend_from_slice(&ROLE_PREFIX_IDS);
            let result = runner
                .compare_exact(
                    TEST_NAME,
                    LadderRung::L0PromptTokenIds,
                    "prompt_build.prompt.text_ids",
                    EngineOutput::oracle(&oracle_prompt_ids),
                    EngineOutput::subject(&subject_wrapped),
                )
                .expect("floor supports exact L0");
            assert_eq!(
                result.outcome,
                Outcome::Passed,
                "tokenizer + prompt wrap must reproduce the captured prompt ids exactly"
            );
        }
        _ => {
            runner.skip_rung(
                TEST_NAME,
                LadderRung::L0PromptTokenIds,
                "prompt_build.prompt.text_ids",
                "pinned tokenizer snapshot or corpus absent; run docs/truth-pack/fetch-truth-pack.sh",
            );
        }
    }

    // L1 — the floor records `not_observed`; the runner must turn that into its reasoned skip.
    // Sentinel-mismatched sides guarantee that a floor flipped to `observed` without real
    // per-operator capture fails here instead of silently passing.
    let l1 = runner
        .compare_f32(
            TEST_NAME,
            LadderRung::L1OperatorSeams,
            "operators.unwired",
            EngineOutput::oracle(&[0.0]),
            EngineOutput::subject(&[f32::NAN]),
        )
        .expect("floor is well-formed");
    assert_eq!(
        l1.outcome,
        Outcome::Skipped,
        "L1 must stay an explicit floor-reasoned skip until per-operator capture exists \
         (frankentts-gdq defined it as scalar==dispatched-kernel equivalence)"
    );

    // L2 / L3 / L5 — measurably non-exact under the zero envelope; owned by their component
    // beads. Skipped-with-reason so the scorecard carries them visibly instead of omitting them.
    runner.skip_rung(
        TEST_NAME,
        LadderRung::L2LayerAndComponentActivations,
        "talker.layer_00.output",
        "seam runs in talker_layer_l2 under a measured relative bound; non-exact under the zero \
         envelope (max_abs ~9.5e-6, RMSNorm-attributed) — ladder wiring is frankentts-p1-talker-z2w exit work",
    );
    runner.skip_rung(
        TEST_NAME,
        LadderRung::L3Logits,
        "talker.codec_head.output",
        "logit-level exactness inherits the talker activation residual; per-seam coverage in \
         talker_all_layers_l2 — ladder wiring is frankentts-p1-talker-z2w exit work",
    );

    // L4 — our production sampler stack over the oracle's own captured per-step logits must
    // reproduce the captured primary-code stream exactly.
    let manifest = fixtures
        .mode_manifest(CASE, MODE)
        .expect("per-mode manifest readable");
    let expected_codes = npy::read_i64(&fixtures.seam_path(
        &SeamRef {
            case: CASE,
            mode: MODE,
            group: GROUP,
            seam: "talker.codec_codes",
        },
        "tensor",
        0,
    ))
    .expect("oracle utterance codes captured");
    let frames = manifest.generated_frames;
    assert!(frames > 0, "the oracle captured no frames");
    let code_groups = expected_codes.data.len() / frames;
    let oracle_primary: Vec<u32> = expected_codes
        .data
        .chunks(code_groups)
        .map(|frame| u32::try_from(frame[0]).expect("primary codes are non-negative"))
        .collect();
    let mut subject_primary = Vec::with_capacity(frames);
    let mut history: Vec<u32> = Vec::with_capacity(frames);
    for (step, &oracle_code) in oracle_primary.iter().enumerate() {
        let logits = fixtures
            .seam(
                &SeamRef {
                    case: CASE,
                    mode: MODE,
                    group: GROUP,
                    seam: "talker.codec_head.output",
                },
                "tensor",
                step,
            )
            .expect("per-step talker logits captured");
        // The prefill step captures logits for every prompt position ([1, seq, vocab]); the code
        // decision reads the final position's row, which is also the whole tensor once seq is 1.
        let vocab = *logits.shape.last().expect("logits have a vocab axis");
        let row = &logits.data[logits.data.len() - vocab..];
        let code = QwenSampler::seeded(0)
            .select_talker(row, &history, SamplingMode::CanonicalGreedy)
            .expect("captured logits are a well-formed talker row");
        subject_primary.push(code);
        history.push(oracle_code);
    }
    let l4 = runner
        .compare_exact(
            TEST_NAME,
            LadderRung::L4GreedyCodecTokens,
            "talker.codec_codes.primary",
            EngineOutput::oracle(&oracle_primary),
            EngineOutput::subject(&subject_primary),
        )
        .expect("floor supports exact L4");
    assert_eq!(
        l4.outcome,
        Outcome::Passed,
        "canonical greedy over the oracle's own logits must reproduce the captured primary codes"
    );

    runner.skip_rung(
        TEST_NAME,
        LadderRung::L5CodecWaveform,
        "codec.generated_waveform",
        "codec decode runs in codec_decode_l2 under measured bounds; snake-activation \
         transcendental parity is in flight (codec_snake_bisect) — ladder wiring is \
         frankentts-p1-codec-hu7 exit work",
    );

    // The scorecard: six rungs recorded, honestly non-green, bundle on disk with verifying hashes.
    let scorecard = runner.scorecard();
    assert_eq!(scorecard.results().len(), 6, "every rung must be recorded");
    assert!(
        !scorecard.all_green(),
        "L1/L2/L3/L5 are skipped today; a green scorecard here would be counterfeit"
    );
    let evidence_dir = workspace_path("target/evidence/ladder-scorecard");
    let bundle = scorecard
        .write_evidence(&evidence_dir)
        .expect("evidence bundle writes");
    let manifest_text =
        fs::read_to_string(&bundle.manifest_path).expect("evidence manifest readable");
    for (line, path) in manifest_text
        .lines()
        .zip([&bundle.scorecard_path, &bundle.receipts_path])
    {
        let (recorded, name) = line
            .split_once("  ")
            .expect("`<sha256>  <file>` manifest lines");
        let bytes = fs::read(path).expect("bundle file readable");
        let actual = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(&bytes))
        };
        assert_eq!(recorded, actual, "manifest hash must match `{name}`");
    }

    Receipt::new(TEST_NAME, Outcome::Passed)
        .contract("ConformanceExact/ladder")
        .seam("ladder.scorecard")
        .reason(&*format!(
            "scorecard recorded 6 rungs (L0 pass, L4 pass, L1 floor-skip, L2/L3/L5 owned-skip), \
             all_green=false as expected; evidence bundle at {}",
            evidence_dir.display()
        ))
        .oracle_tier(OracleTier::CpuFp32Fallback)
        .emit();
}
