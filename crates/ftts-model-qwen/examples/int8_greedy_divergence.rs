//! Prints the canonical-greedy code stream for one utterance, for f32-vs-int8 comparison.
//!
//! Contract-B DIAGNOSTIC, not the listening gate: run this twice — once plain, once with
//! `FTTS_INT8=1` — and diff the lines. Greedy decode consumes no RNG, so any difference is
//! quantization-induced token drift at the argmax boundary, the cheapest observable slice of
//! what Contract B measures properly (logit KL, WER, blind listening).
//!
//! ```sh
//! cargo run --release -p ftts-model-qwen --example int8_greedy_divergence -- \
//!     docs/truth-pack/snapshots/hf voice.spk "some text" 24 > f32.codes
//! FTTS_INT8=1 cargo run --release -p ftts-model-qwen --example int8_greedy_divergence -- \
//!     docs/truth-pack/snapshots/hf voice.spk "some text" 24 > int8.codes
//! diff f32.codes int8.codes
//! ```

use ftts_core::{FrameGenerator, NormalizationOptions, PreparedText, TextPreparer};
use ftts_model_qwen::checkpoint::{CODEC_LANGUAGE_ENGLISH_ID, TalkerCheckpoint};
use ftts_model_qwen::generate::{QwenGenerator, QwenGeneratorConfig};
use ftts_model_qwen::microdecoder::MicrodecoderConfig;
use ftts_model_qwen::prompt::{CloneMode, PromptMode};
use ftts_model_qwen::sampler::SamplingMode;
use ftts_model_qwen::talker::TalkerConfig;
use ftts_model_qwen::tokenizer::{QwenTokenizer, TokenizerFiles};
use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let model_dir = args
        .next()
        .expect("usage: MODEL_DIR VOICE.spk TEXT [MAX_FRAMES]");
    let voice = args.next().expect("VOICE.spk");
    let text = args.next().expect("TEXT");
    let max_frames: usize = args.next().map_or(24, |v| v.parse().expect("MAX_FRAMES"));
    // `production` switches to the shipped sampler (seeded, deterministic per build+seed). The
    // greedy default is the divergence diagnostic; production mode exists to dump codes with a
    // real speech envelope — greedy-under-greedy sits in the measured silence attractor (p7r).
    let sampling = match args.next().as_deref() {
        Some("production") => SamplingMode::Production,
        _ => SamplingMode::CanonicalGreedy,
    };
    let root = Path::new(&model_dir);

    let bytes = std::fs::read(&voice).expect("voice file");
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(remainder.is_empty(), "voice file is not whole f32s");
    let speaker: Vec<f32> = words.iter().map(|word| f32::from_le_bytes(*word)).collect();
    assert_eq!(
        speaker.len(),
        1024,
        "voice must be 1,024 little-endian f32s"
    );

    let read = |name: &str| std::fs::read_to_string(root.join(name)).expect(name);
    let (vocab, merges, config) = (
        read("vocab.json"),
        read("merges.txt"),
        read("tokenizer_config.json"),
    );
    let tokenizer = QwenTokenizer::from_files_using_environment(TokenizerFiles {
        vocab_json: &vocab,
        merges_txt: &merges,
        tokenizer_config_json: &config,
    })
    .expect("tokenizer");

    let talker = TalkerCheckpoint::load(&root.join("model.safetensors")).expect("checkpoint");

    let prepared_raw = tokenizer
        .prepare(&text, &NormalizationOptions::default())
        .expect("text");
    let wrapped = TalkerCheckpoint::wrap_target_ids(&prepared_raw.token_ids);
    let prepared = PreparedText::new(wrapped.clone(), prepared_raw.normalization_trace);

    let ids = TalkerCheckpoint::utterance_text_ids(&wrapped);
    let table = talker.gather_text_rows(&ids).expect("text rows");
    let header = talker
        .xvector_header(&table, &speaker, CODEC_LANGUAGE_ENGLISH_ID)
        .expect("header");
    let tts_eos = talker.tts_eos(&table);

    let talker_layers = talker.talker_layer_weights();
    let micro_layers = talker.microdecoder_layer_weights();
    let residual = talker.residual_embedding_slices();
    let heads = talker.microdecoder_head_slices();
    let micro_residual = &residual[..residual.len() - 1];

    let mut generator = QwenGenerator::new(QwenGeneratorConfig {
        talker_config: TalkerConfig::default(),
        talker_weights: talker.talker_weights(&talker_layers),
        text: talker.text_weights(&table),
        feedback: talker.feedback_tables(&residual),
        microdecoder_config: MicrodecoderConfig::default(),
        microdecoder_weights: talker.microdecoder_weights(&micro_layers, micro_residual, &heads),
        prompt_mode: PromptMode {
            clone_mode: CloneMode::XVector,
            non_streaming_mode: false,
        },
        header,
        tts_eos,
        reference: None,
        // Canonical greedy by default: zero RNG, so two runs differ only through the kernels.
        sampling_mode: sampling,
        seed: 0,
    });

    eprintln!(
        "int8 route: {}",
        if std::env::var("FTTS_INT8").as_deref() == Ok("1") {
            "ARMED"
        } else {
            "off (f32 reference)"
        }
    );

    generator
        .begin_utterance(&prepared, ftts_core::UtteranceStart::Fresh)
        .expect("prefill");
    for frame in 0..max_frames {
        match generator.next_frame().expect("frame") {
            ftts_core::FrameStep::Frame(code_frame) => {
                let rendered: Vec<String> =
                    code_frame.codes.iter().map(ToString::to_string).collect();
                println!("frame {frame:03}: {}", rendered.join(" "));
            }
            ftts_core::FrameStep::Finished => {
                println!("frame {frame:03}: EOS");
                break;
            }
            ftts_core::FrameStep::AwaitingText => unreachable!("fresh utterance"),
        }
    }
}
