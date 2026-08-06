#![no_main]

use std::sync::OnceLock;

use ftts_artifacts::fttsq::{
    AccessClass, FttsqReader, FttsqWriter, StoredDtype, TensorEntry,
};
use libfuzzer_sys::fuzz_target;

const NOTICE: &str = "Apache-2.0 fuzz seed";

fn valid_seed() -> &'static [u8] {
    static SEED: OnceLock<Vec<u8>> = OnceLock::new();

    SEED.get_or_init(|| {
        FttsqWriter::new("qwen3-tts-12hz-0.6b-base", "0".repeat(64))
            .license_notice(NOTICE)
            .section(
                "microdecoder",
                AccessClass::HotRecurrentMicrodecoder,
                vec![0x55; 2],
            )
            .section(
                "text_embedding",
                AccessClass::ColdTextEmbedding,
                vec![0xAA; 2],
            )
            .tensor(TensorEntry {
                name: "microdecoder.odd_q4".to_owned(),
                section: "microdecoder".to_owned(),
                dtype: StoredDtype::Q4,
                shape: vec![3],
                offset: 0,
                length: 2,
                scales: None,
            })
            .tensor(TensorEntry {
                name: "text_embedding.weight".to_owned(),
                section: "text_embedding".to_owned(),
                dtype: StoredDtype::Bf16,
                shape: vec![1],
                offset: 0,
                length: 2,
                scales: None,
            })
            .finish()
            .expect("the fixed fuzz seed must be a valid .fttsq artifact")
    })
}

fn exercise(bytes: &[u8]) {
    let Ok(reader) = FttsqReader::parse_directory(bytes) else {
        return;
    };

    for tensor in reader.tensors() {
        let _ = reader.tensor_bytes(&tensor.name, bytes);
    }
    let _ = reader.verify_digests(bytes);
    let _ = FttsqReader::open(bytes);
}

fuzz_target!(|data: &[u8]| {
    exercise(data);

    // Keep one known-valid artifact on every execution, then apply fuzzer-directed corruption to
    // it. A magic-only corpus otherwise almost never reaches the directory/digest/tensor paths.
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
