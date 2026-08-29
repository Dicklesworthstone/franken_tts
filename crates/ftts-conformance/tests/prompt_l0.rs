//! Contract-A L0 checks for the Base voice-clone prompt boundary.
//!
//! The external oracle fixture is intentionally not committed: it contains sensitive reference
//! material.  Its absence emits an honest skip receipt; when supplied, every token comparison here
//! has tolerance zero because wrapper ids and prompt geometry are discrete contracts.

#![cfg(feature = "ultra-tests")]

use std::{
    env, fs,
    path::{Path, PathBuf},
};

use ftts_conformance::{
    assert_exact,
    prelude::{Outcome, Receipt},
    test_name,
};
use ftts_model_qwen::prompt::extract_prompt_text_ids;

const FIXTURE_REVISION: &str = "ft7-cpu-fp32-r1";
const TARGET_WRAPPED: [u32; 10] = [
    151_644, 77_091, 198, 9707, 13, 151_645, 198, 151_644, 77_091, 198,
];
const REFERENCE_WRAPPED: [u32; 7] = [151_644, 77_091, 198, 9707, 13, 151_645, 198];
const TARGET_TEXT: [u32; 2] = [9707, 13];

fn fixture_root() -> PathBuf {
    let configured = env::var_os("FTTS_ORACLE_FIXTURES").map(PathBuf::from);
    let parent = configured.unwrap_or_else(|| {
        env::var_os("HOME").map_or_else(PathBuf::new, |home| {
            PathBuf::from(home).join(".cache/frankentts/oracle-fixtures")
        })
    });
    if parent
        .file_name()
        .is_some_and(|name| name == FIXTURE_REVISION)
    {
        parent
    } else {
        parent.join(FIXTURE_REVISION)
    }
}

fn read_npy_i64(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if bytes.get(..6) != Some(b"\x93NUMPY".as_slice()) || bytes.get(6..8) != Some([1, 0].as_slice())
    {
        return Err(format!("{} is not a NumPy v1 file", path.display()));
    }
    let header_len_bytes = bytes
        .get(8..10)
        .ok_or_else(|| format!("{} has a truncated NumPy preamble", path.display()))?;
    let header_len =
        usize::from(u16::from_le_bytes(header_len_bytes.try_into().map_err(
            |_| format!("{} has an invalid NumPy preamble", path.display()),
        )?));
    let payload = bytes
        .get(10 + header_len..)
        .ok_or_else(|| format!("{} has a truncated NumPy header", path.display()))?;
    if payload.len() % 8 != 0 {
        return Err(format!("{} has a non-i64 payload", path.display()));
    }
    payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| {
            let id = i64::from_le_bytes(*chunk);
            u32::try_from(id).map_err(|_| format!("{} contains token id {id}", path.display()))
        })
        .collect()
}

fn exact_fixture_ids(path: &Path, mode: &str, kind: &str) -> Vec<u32> {
    match read_npy_i64(path) {
        Ok(ids) => ids,
        Err(error) => {
            Receipt::new(test_name!(), Outcome::Failed)
                .contract("ConformanceExact/L0")
                .seam("prompt.fixture_load")
                .reason(format!("{mode} {kind} fixture is unreadable: {error}"))
                .emit();
            Vec::new()
        }
    }
}

fn prompt_ids_path(root: &Path, mode: &str, kind: &str) -> PathBuf {
    root.join("synthetic-tone-en")
        .join(mode)
        .join("stages/prompt_build")
        .join(format!("prompt.{kind}/tensor.000.npy"))
}

#[test]
fn contract_a_l0_prompt_wrappers_all_four_variants_zero_tolerance() {
    let root = fixture_root();
    let streaming_target = prompt_ids_path(&root, "xvector_streaming", "text_ids");
    if !streaming_target.is_file() {
        Receipt::new(test_name!(), Outcome::Skipped)
            .contract("ConformanceExact/L0")
            .seam("prompt.wrapper_ids")
            .reason(format!(
                "oracle fixtures unavailable; expected {} or set FTTS_ORACLE_FIXTURES",
                streaming_target.display()
            ))
            .emit();
        return;
    }

    for mode in [
        "xvector_streaming",
        "xvector_non_streaming",
        "icl_streaming",
        "icl_non_streaming",
    ] {
        let actual = exact_fixture_ids(&prompt_ids_path(&root, mode, "text_ids"), mode, "target");
        assert_exact!(
            contract = "ConformanceExact/L0",
            seam = "prompt.text_ids",
            expected = &TARGET_WRAPPED,
            actual = &actual,
        );
    }

    for mode in ["icl_streaming", "icl_non_streaming"] {
        let actual = exact_fixture_ids(
            &prompt_ids_path(&root, mode, "reference_ids"),
            mode,
            "reference",
        );
        assert_exact!(
            contract = "ConformanceExact/L0",
            seam = "prompt.reference_ids",
            expected = &REFERENCE_WRAPPED,
            actual = &actual,
        );
    }
}

#[test]
fn contract_a_l0_trap_3_asymmetric_slices_fixture_confirmed_zero_tolerance() {
    let root = fixture_root();
    let target_path = prompt_ids_path(&root, "icl_streaming", "text_ids");
    let reference_path = prompt_ids_path(&root, "icl_streaming", "reference_ids");
    if !target_path.is_file() || !reference_path.is_file() {
        Receipt::new(test_name!(), Outcome::Skipped)
            .contract("ConformanceExact/L0")
            .seam("prompt.trap_3_wrapper_slices")
            .reason("oracle fixture required to confirm the previously inferred asymmetric slices")
            .emit();
        return;
    }
    let target = read_npy_i64(&target_path).expect("read target wrapper fixture");
    let reference = read_npy_i64(&reference_path).expect("read reference wrapper fixture");
    let extracted = extract_prompt_text_ids(&target, Some(&reference))
        .expect("fixture must carry the official assistant wrappers");
    assert_exact!(
        contract = "ConformanceExact/L0",
        seam = "prompt.target_ids.input_3_to_minus_5",
        expected = &TARGET_TEXT,
        actual = &extracted.target,
    );
    assert_exact!(
        contract = "ConformanceExact/L0",
        seam = "prompt.reference_ids.input_3_to_minus_2",
        expected = &TARGET_TEXT,
        actual = extracted
            .reference
            .as_deref()
            .expect("ICL fixture has reference ids"),
    );
}
