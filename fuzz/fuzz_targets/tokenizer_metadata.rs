#![no_main]

use ftts_model_qwen::tokenizer::{QwenTokenizer, TokenizerFiles, TokenizerRegex};
use libfuzzer_sys::fuzz_target;

fn split_metadata(data: &[u8]) -> (&str, &str, &str) {
    let first = data
        .iter()
        .position(|byte| *byte == b'|')
        .unwrap_or(data.len());
    let second_start = first.saturating_add(1).min(data.len());
    let second = data[second_start..]
        .iter()
        .position(|byte| *byte == b'|')
        .map(|offset| second_start + offset)
        .unwrap_or(data.len());
    let third_start = second.saturating_add(1).min(data.len());
    (
        std::str::from_utf8(&data[..first]).unwrap_or(""),
        std::str::from_utf8(&data[second_start..second]).unwrap_or(""),
        std::str::from_utf8(&data[third_start..]).unwrap_or(""),
    )
}

fuzz_target!(|data: &[u8]| {
    let (vocab_json, merges_txt, tokenizer_config_json) = split_metadata(data);
    let files = TokenizerFiles {
        vocab_json,
        merges_txt,
        tokenizer_config_json,
    };
    let _ = QwenTokenizer::from_files(files, TokenizerRegex::Official);
    let _ = QwenTokenizer::from_files(files, TokenizerRegex::Native);
});
