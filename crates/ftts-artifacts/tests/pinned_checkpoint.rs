//! Model-gated: load the real pinned checkpoints through the mmap path and run the census.
//!
//! Skips with SUCCESS when the truth-pack snapshots are absent (they are gitignored, ~2.5 GB), and
//! says so out loud — a skipped test is never reported as a pass. When the weights *are* present
//! these assertions are the real thing: the actual 1.7 GB and 651 MB files, mapped rather than
//! read, with every number below taken from the pinned bytes rather than from the plan.

use ftts_artifacts::census::{ExpectedTensor, WeightsManifest};
use ftts_artifacts::safetensors::{Dtype, SafetensorsFile};
use std::path::{Path, PathBuf};

/// Where the truth pack puts the pinned snapshots.
fn snapshot(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf")
        .join(relative)
}

/// Report a skip loudly rather than silently passing.
fn skip(what: &str, path: &Path) -> bool {
    eprintln!(
        "SKIP[model-gated]: {what} — pinned weights absent at {}; run \
         docs/truth-pack/fetch-truth-pack.sh --with-weights to enable",
        path.display()
    );
    true
}

#[test]
fn talker_checkpoint_maps_and_passes_census() {
    let path = snapshot("model.safetensors");
    if !path.exists() {
        assert!(skip("talker checkpoint", &path));
        return;
    }

    let file = SafetensorsFile::open(&path).expect("pinned talker checkpoint maps and parses");
    file.advise_random();
    let index = file.index();

    // Counts and geometry come from the pinned bytes (OQ-2/OQ-7 census).
    assert_eq!(
        index.len(),
        478,
        "tensor count changed — is this the pinned revision?"
    );
    assert_eq!(file.mapped_len(), 1_829_344_272);

    // The cold text embedding is the headline case for the whole design (plan 2.4).
    let embedding = index
        .entry("talker.model.text_embedding.weight")
        .expect("cold text embedding present");
    assert_eq!(embedding.shape, vec![151_936, 2048]);
    assert_eq!(embedding.dtype, Dtype::Bf16);
    assert_eq!(embedding.byte_len(), 622_329_856);

    // Cold-row access: read scattered rows only, exactly as prefill would.
    let view = file
        .view("talker.model.text_embedding.weight")
        .expect("view");
    let row_len = view.row_len();
    assert_eq!(row_len, 2048);
    let mut row = vec![0.0f32; row_len];
    for index_of_row in [0usize, 1, 75_968, 151_935] {
        assert!(
            view.copy_row_f32(index_of_row, &mut row),
            "row {index_of_row} must read"
        );
        assert!(
            row.iter().all(|value| value.is_finite()),
            "row {index_of_row} widened to a non-finite value"
        );
    }
    // One past the last row must refuse rather than read past the table.
    assert!(!view.copy_row_f32(151_936, &mut row));

    // A census built from the file itself is green by construction; the point is that the
    // comparison machinery agrees with reality on a 478-tensor checkpoint.
    let manifest = WeightsManifest::from_expectations(
        "talker",
        index
            .entries()
            .map(|entry| ExpectedTensor::new(entry.name.clone(), entry.shape.clone(), entry.dtype)),
    );
    let report = manifest.audit(index);
    assert!(report.is_green(), "{}", report.render());
}

#[test]
fn codec_checkpoint_maps_and_passes_census() {
    let path = snapshot("speech_tokenizer/model.safetensors");
    if !path.exists() {
        assert!(skip("codec checkpoint", &path));
        return;
    }

    let file = SafetensorsFile::open(&path).expect("pinned codec checkpoint maps and parses");
    let index = file.index();

    assert_eq!(
        index.len(),
        496,
        "tensor count changed — is this the pinned revision?"
    );

    // OQ-7: all 16 decoder codebooks are [2048, 256] and the codec ships F32, not BF16.
    let semantic = index
        .entry("decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")
        .expect("semantic codebook present");
    assert_eq!(semantic.shape, vec![2048, 256]);
    assert_eq!(semantic.dtype, Dtype::F32);
    assert!(
        index.entries().all(|entry| entry.dtype == Dtype::F32),
        "codec is expected to be entirely F32"
    );

    // The binding overflow case from OQ-7: decoder.0 conv is [1536, 1024, 7], K = 1024*7 = 7168.
    let widest = index
        .entry("decoder.decoder.0.conv.weight")
        .expect("decoder.0 conv present");
    assert_eq!(widest.shape, vec![1536, 1024, 7]);
    let reduction_k = widest.shape[1] * widest.shape[2];
    assert_eq!(reduction_k, 7168);

    let manifest = WeightsManifest::from_expectations(
        "codec",
        index
            .entries()
            .map(|entry| ExpectedTensor::new(entry.name.clone(), entry.shape.clone(), entry.dtype)),
    );
    assert!(manifest.verify(index).is_ok());
}

#[test]
fn a_wrong_manifest_is_refused_against_the_real_checkpoint() {
    let path = snapshot("model.safetensors");
    if !path.exists() {
        assert!(skip("wrong-manifest refusal", &path));
        return;
    }

    let file = SafetensorsFile::open(&path).expect("maps");
    let mut manifest = WeightsManifest::new("deliberately-wrong");
    manifest.expect(ExpectedTensor::new(
        "talker.model.text_embedding.weight",
        vec![151_936, 1024], // wrong: real is 2048 wide
        Dtype::F32,          // wrong: real is BF16
    ));

    let report = manifest
        .verify(file.index())
        .expect_err("a wrong manifest must be refused");
    assert_eq!(report.count_of("SHAPE-MISMATCH"), 1);
    assert_eq!(report.count_of("DTYPE-MISMATCH"), 1);
    assert_eq!(report.count_of("EXTRA"), 477);

    let rendered = report.render();
    assert!(rendered.contains("REFUSED"));
    assert!(rendered.contains("talker.model.text_embedding.weight"));
}
