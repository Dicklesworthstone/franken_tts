#![no_main]

use ftts_model_qwen::tokenizer::{
    NormalizationOptions, QwenTokenizer, TokenizerFiles, TokenizerRegex,
};
use libfuzzer_sys::fuzz_target;

const VOCAB: &str = r#"{"a":0,"b":1,"ab":2,"ÿ":3}"#;
const MERGES: &str = "#version: 0.2\na b\n";
const CONFIG: &str = r#"{"added_tokens_decoder":{"4":{"content":"<s>"}}}"#;

fuzz_target!(|data: &[u8]| {
    let input = String::from_utf8_lossy(data);
    let tokenizer = QwenTokenizer::from_files(
        TokenizerFiles {
            vocab_json: VOCAB,
            merges_txt: MERGES,
            tokenizer_config_json: CONFIG,
        },
        TokenizerRegex::Official,
    )
    .expect("static fuzz fixture is valid");

    let _ = tokenizer.normalize(&input, &NormalizationOptions::default());
    let _ = tokenizer.encode(&input);
});
