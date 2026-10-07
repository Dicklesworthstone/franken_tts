#![no_main]

//! `.ftvoice-cache` container parser. `ftts say` reads these from a user-writable cache
//! directory (the ICL primed-codec-state blob), so the container is untrusted input: truncation,
//! bit flips, lying offsets, and a key digest that disagrees with its components must all be
//! refused without panicking. A valid cache with a multi-kilobyte blob is exercised and corrupted
//! every run.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use ftts_artifacts::voice::{FtVoiceCacheKey, parse_ftvoice_cache, serialize_ftvoice_cache};
use libfuzzer_sys::fuzz_target;

fn valid_seed() -> &'static [u8] {
    static SEED: OnceLock<Vec<u8>> = OnceLock::new();
    SEED.get_or_init(|| {
        let key = FtVoiceCacheKey {
            voice_recipe_hash: "ab".repeat(32),
            model_hash: "cd".repeat(32),
            prompt_builder_version: 0,
            streaming_mode: "streaming".to_owned(),
            quant_recipe: "codec-primed-state/fuzz".to_owned(),
            math_mode: "default".to_owned(),
            engine_abi: 1,
            language_id: String::new(),
            speaker_embed_sha256: String::new(),
            ref_transcript_tokens_sha256: None,
            ref_codec_codes_sha256: Some("ab".repeat(32)),
        };
        let mut blobs = BTreeMap::new();
        blobs.insert(
            "codec_primed_state".to_owned(),
            (0..4096_u32).flat_map(u32::to_le_bytes).collect(),
        );
        blobs.insert("header_kv".to_owned(), vec![7; 64]);
        serialize_ftvoice_cache(&key, &blobs).expect("the fixed fuzz seed must be a valid cache")
    })
}

fn exercise(bytes: &[u8]) {
    let Ok(cache) = parse_ftvoice_cache(bytes) else {
        return;
    };
    let reserialized =
        serialize_ftvoice_cache(&cache.key, &cache.blobs).expect("a parsed cache re-serializes");
    let reparsed = parse_ftvoice_cache(&reserialized).expect("a re-serialized cache parses");
    assert_eq!(reparsed.key, cache.key, "round trip changed the cache key");
    assert_eq!(reparsed.blobs, cache.blobs, "round trip changed a blob");
}

fuzz_target!(|data: &[u8]| {
    exercise(data);
    let seed = valid_seed();
    exercise(seed);
    let mut mutated = seed.to_vec();
    if let Some((&offset, mutations)) = data.split_first() {
        let start = usize::from(offset) % mutated.len();
        for (index, byte) in mutations.iter().enumerate().take(mutated.len() - start) {
            mutated[start + index] ^= byte;
        }
    }
    exercise(&mutated);
});
