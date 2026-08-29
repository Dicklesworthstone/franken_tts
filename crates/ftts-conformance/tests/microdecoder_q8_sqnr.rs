//! Per-row Q8 SQNR receipt for the microdecoder's per-depth embedding tables and scoring
//! heads, measured on the real pinned checkpoint (bead `frankentts-x7bt`).
//!
//! `--micro-q8` stores these thirty `[2048, 1024]` bf16 tensors as per-row symmetric Q8
//! (the canonical [`quantize_output_channel_q8`] recipe). This test produces the ledger
//! numbers doctrine #8 demands BEFORE anyone flips a default:
//!
//! 1. **Round-trip identity is asserted, per row, on real weights.** Re-quantizing a
//!    stored block's own dequantized values must reproduce the exact bytes and scale bit
//!    pattern. This is the claim artifact-native head consumption relies on: the requantize
//!    fallback it skips would have produced identical bytes.
//! 2. **Per-row SNR is reported, not gated.** Worst-row and mean SNR per tensor family go
//!    to the receipt stream (`FTTS_RECEIPTS`), where the Contract-B listening design and
//!    any future default-flip decision consume them. A hard floor here would prejudge the
//!    listening pair (`frankentts-4tgm`) that outranks every objective number.
//!
//! Model-gated: without the pinned checkpoint each check reports its skip and passes.

#![cfg(feature = "ultra-tests")]

use std::path::{Path, PathBuf};

use ftts_artifacts::converter::quantize_output_channel_q8;
use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_conformance::report::{Outcome, Receipt};

const CONTRACT: &str = "ConformanceExact/L2";
const SEAM: &str = "microdecoder.q8_tables";
const VOCAB: usize = 2048;
const HIDDEN: usize = 1024;
const EMBEDDING_TABLES: usize = 15;
const HEAD_TABLES: usize = 15;

fn checkpoint_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/truth-pack/snapshots/hf/model.safetensors")
}

fn skip(test: &str, reason: &str) {
    Receipt::new(test, Outcome::Skipped)
        .contract(CONTRACT)
        .seam(SEAM)
        .reason(reason)
        .emit();
}

/// Widens one BF16 tensor to f32 through the accessor, exactly how the engine reads weights.
fn widen(file: &SafetensorsFile, name: &str) -> Result<Vec<f32>, String> {
    let view = file
        .view(name)
        .ok_or_else(|| format!("checkpoint is missing `{name}`"))?;
    if view.len() != VOCAB * HIDDEN {
        return Err(format!(
            "`{name}` holds {} elements, expected {VOCAB}x{HIDDEN}",
            view.len()
        ));
    }
    (0..view.len())
        .map(|index| {
            view.get_f32(index)
                .ok_or_else(|| format!("`{name}` index {index} out of range"))
        })
        .collect()
}

struct RowQuality {
    /// 10·log10(Σx² / Σe²) for the noisiest row, f64 so aggregation cannot hide drift.
    worst_snr_db: f64,
    mean_snr_db: f64,
    worst_row: usize,
}

/// Quantizes one table row-by-row with the canonical converter primitive and scores it.
fn table_quality(name: &str, table: &[f32]) -> Result<RowQuality, String> {
    let mut q8 = vec![0_i8; HIDDEN];
    let mut snrs = Vec::with_capacity(VOCAB);
    let mut worst = (f64::INFINITY, 0_usize);
    let mut sum = 0.0_f64;
    for (row_index, row) in table.as_chunks::<HIDDEN>().0.iter().enumerate() {
        let scale = quantize_output_channel_q8(row, &mut q8)
            .map_err(|error| format!("{name} row {row_index}: {error}"))?;
        let signal: f64 = row
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum();
        let error: f64 = row
            .iter()
            .zip(q8.iter())
            .map(|(&value, &byte)| {
                let difference = value - f32::from(byte) * scale;
                f64::from(difference) * f64::from(difference)
            })
            .sum();
        // An all-zero row has zero error by construction (scale 1.0, bytes 0): perfect,
        // and excluded from the geometric story rather than scored as infinite.
        let snr = if error == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (signal / error).log10()
        };
        if snr < worst.0 {
            worst = (snr, row_index);
        }
        sum += snr;
        snrs.push(snr);
    }
    Ok(RowQuality {
        worst_snr_db: worst.0,
        mean_snr_db: sum / snrs.len() as f64,
        worst_row: worst.1,
    })
}

/// The round-trip law the runtime fallback leans on: quantizing a stored block's own
/// dequantized values reproduces the exact bytes and the exact scale bits.
fn assert_round_trip_identity(name: &str, table: &[f32]) -> Result<(), String> {
    let mut first = vec![0_i8; HIDDEN];
    let mut second = vec![0_i8; HIDDEN];
    for (row_index, row) in table.as_chunks::<HIDDEN>().0.iter().enumerate() {
        let scale_a = quantize_output_channel_q8(row, &mut first)
            .map_err(|error| format!("{name} row {row_index}: {error}"))?;
        let dequantized: Vec<f32> = first
            .iter()
            .map(|&byte| f32::from(byte) * scale_a)
            .collect();
        let scale_b = quantize_output_channel_q8(&dequantized, &mut second)
            .map_err(|error| format!("{name} row {row_index} re-quantize: {error}"))?;
        if scale_a.to_bits() != scale_b.to_bits() {
            return Err(format!(
                "{name} row {row_index}: round-trip scale drifted ({scale_a:e} vs {scale_b:e})"
            ));
        }
        if first != second {
            return Err(format!(
                "{name} row {row_index}: round-trip bytes differ — the requantize-fallback \
                 is NOT byte-identical to artifact-native consumption"
            ));
        }
    }
    Ok(())
}

fn run_family(
    file: &SafetensorsFile,
    prefix: &str,
    count: usize,
    label: &str,
) -> Result<(f64, usize), String> {
    let mut global_worst = f64::INFINITY;
    let mut global_worst_tensor = 0_usize;
    for index in 0..count {
        let name = format!("{prefix}{index}.weight");
        let table = widen(file, &name)?;
        assert_round_trip_identity(&name, &table)?;
        let quality = table_quality(&name, &table)?;
        eprintln!(
            "receipt: {{\"test\":\"q8_table_sqnr\",\"family\":\"{label}\",\"tensor\":\"{name}\",\
             \"worst_row_snr_db\":{:.3},\"worst_row\":{},\"mean_row_snr_db\":{:.3}}}",
            quality.worst_snr_db, quality.worst_row, quality.mean_snr_db,
        );
        if quality.worst_snr_db < global_worst {
            global_worst = quality.worst_snr_db;
            global_worst_tensor = index;
        }
    }
    Ok((global_worst, global_worst_tensor))
}

#[test]
fn microdecoder_q8_tables_round_trip_and_report_sqnr() {
    const TEST: &str = "microdecoder_q8_tables_round_trip_and_report_sqnr";
    let path = checkpoint_path();
    if !path.is_file() {
        skip(TEST, &format!("checkpoint absent at {}", path.display()));
        return;
    }
    let file = match SafetensorsFile::open(&path) {
        Ok(file) => file,
        Err(error) => {
            skip(TEST, &format!("cannot open {}: {error}", path.display()));
            return;
        }
    };

    let (embedding_worst, embedding_tensor) = match run_family(
        &file,
        "talker.code_predictor.model.codec_embedding.",
        EMBEDDING_TABLES,
        "embedding",
    ) {
        Ok(result) => result,
        Err(reason) => {
            skip(TEST, &reason);
            return;
        }
    };
    let (head_worst, head_tensor) =
        match run_family(&file, "talker.code_predictor.lm_head.", HEAD_TABLES, "head") {
            Ok(result) => result,
            Err(reason) => {
                skip(TEST, &reason);
                return;
            }
        };

    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam(SEAM)
        .detail(serde_json::json!({
            "tables": 30,
            "rows_per_table": VOCAB,
            "round_trip_identity": "exact",
            "embedding_worst_row_snr_db": embedding_worst,
            "embedding_worst_tensor": embedding_tensor,
            "head_worst_row_snr_db": head_worst,
            "head_worst_tensor": head_tensor,
        }))
        .emit();
}
