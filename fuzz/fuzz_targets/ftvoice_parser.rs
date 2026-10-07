#![no_main]

//! `.ftvoice` voice-pack parser: hostile bytes must be refused with a named error, never panic,
//! and a pack that parses must re-serialize and re-parse to the same recipe (byte idempotence is
//! the format's metamorphic gate). A valid portable pack (embedding, transcript, ICL codec codes,
//! embedded audio — every section kind) is exercised every run and corrupted by fuzzer input, so
//! the directory/digest/profile paths are reached, not just the magic check.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use ftts_artifacts::voice::{
    ConsentAttestation, ConsentMethod, Provenance, SPEAKER_EMBEDDING_LEN, VoicePack, VoiceProfile,
    parse_voice_pack, serialize_voice_pack,
};
use libfuzzer_sys::fuzz_target;

fn valid_seed() -> &'static [u8] {
    static SEED: OnceLock<Vec<u8>> = OnceLock::new();
    SEED.get_or_init(|| {
        let pack = VoicePack {
            profile: VoiceProfile::Portable,
            consent: ConsentAttestation {
                attested: true,
                method: ConsentMethod::Interactive,
            },
            language: Some("en".to_owned()),
            transcript: Some("Please call Stella.".to_owned()),
            speech_regions: vec![(0, 24_000)],
            diagnostics: None,
            preprocessing: None,
            provenance: Provenance {
                engine: "fuzz seed".to_owned(),
                source_audio_sha256: None,
                source_frames: None,
            },
            embedding: (0..SPEAKER_EMBEDDING_LEN)
                .map(|index| index as f32 / SPEAKER_EMBEDDING_LEN as f32)
                .collect(),
            codec_codes: Some((0..32).map(|code| code * 63).collect()),
            reference_audio: Some(vec![1, 2, 3, 4]),
            section_digests: BTreeMap::new(),
        };
        serialize_voice_pack(&pack).expect("the fixed fuzz seed must be a valid pack")
    })
}

fn exercise(bytes: &[u8]) {
    let Ok(pack) = parse_voice_pack(bytes) else {
        return;
    };
    let _ = pack.recipe_hash();
    let reserialized = serialize_voice_pack(&pack).expect("a parsed pack must re-serialize");
    let reparsed = parse_voice_pack(&reserialized).expect("a re-serialized pack must parse");
    assert_eq!(
        pack.recipe_hash().ok(),
        reparsed.recipe_hash().ok(),
        "parse -> serialize -> parse changed the voice recipe"
    );
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
