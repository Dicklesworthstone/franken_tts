//! `.fttspack` — the regenerable per-machine execution cache.
//!
//! # Purpose & Architecture Split
//!
//! `.fttsq` is canonical and machine-independent.
//! `.fttspack` is **per-machine and regenerable**: it holds arch-specific packed layouts
//! (e.g. SDOT, i8mm, AVX-VNNI, AMX tile interleavings), the physically separated
//! **Microdecoder Hot Pack** (tuned for cache residency across the 15 sequential steps),
//! and the persisted [`PersistedKernelPlan`] (which kernel tier wins per op/shape/regime).
//!
//! # 8-Tuple Cache Key
//!
//! A `.fttspack` cache entry is strictly invalidated if ANY component of the 8-tuple changes:
//! 1. `model_content_hash` — SHA-256 hex of the source `.fttsq`
//! 2. `kernel_abi_version` — integer ABI epoch
//! 3. `cpu_vendor_model` — CPU identifier string
//! 4. `isa_features` — detected hardware feature set
//! 5. `op_shape_plan` — shape mapping schema version
//! 6. `packing_version` — tile packing layout version
//! 7. `quant_execution_mode` — execution regime (e.g. `"w8a8_dynamic"`)
//! 8. `autotune_result_hash` — hash of install-time benchmark winner timings
//!
//! Bead: `frankentts-p2-fttspack-b4t`.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde_json::{Value, json};

use crate::sha256::hex_digest;

/// Magic bytes identifying a `.fttspack` container.
pub const PACK_MAGIC: &[u8; 8] = b"FTTSPACK";

/// Format version for `.fttspack`.
pub const PACK_FORMAT_VERSION: u32 = 1;

/// Errors arising from `.fttspack` encoding, decoding, and validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    BadMagic([u8; 8]),
    UnsupportedVersion(u32),
    CorruptHeader(String),
    KeyMismatch {
        expected: String,
        actual: String,
    },
    TruncatedPayload {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    ChecksumMismatch {
        expected: String,
        actual: String,
    },
    Io(String),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic(bytes) => write!(f, "bad fttspack magic: {bytes:?}"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported fttspack version: {v}"),
            Self::CorruptHeader(msg) => write!(f, "corrupt fttspack header: {msg}"),
            Self::KeyMismatch { expected, actual } => {
                write!(
                    f,
                    "fttspack key mismatch (expected '{expected}', got '{actual}')"
                )
            }
            Self::TruncatedPayload {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "truncated payload: expected {expected_bytes} bytes, found {actual_bytes}"
            ),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected}, actual {actual}")
            }
            Self::Io(msg) => write!(f, "fttspack I/O error: {msg}"),
        }
    }
}

impl std::error::Error for PackError {}

/// The 8-tuple cache key governing cache validation and regeneration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackKey {
    pub model_content_hash: String,
    pub kernel_abi_version: u32,
    pub cpu_vendor_model: String,
    pub isa_features: Vec<String>,
    pub op_shape_plan: String,
    pub packing_version: u32,
    pub quant_execution_mode: String,
    pub autotune_result_hash: String,
}

impl PackKey {
    /// Computes a deterministic canonical digest of the 8-tuple key.
    #[must_use]
    pub fn compute_cache_key(&self) -> String {
        let mut sorted_features = self.isa_features.clone();
        sorted_features.sort();
        let payload = format!(
            "model={}|abi={}|cpu={}|isa={}|shape={}|packver={}|mode={}|autotune={}",
            self.model_content_hash,
            self.kernel_abi_version,
            self.cpu_vendor_model,
            sorted_features.join(","),
            self.op_shape_plan,
            self.packing_version,
            self.quant_execution_mode,
            self.autotune_result_hash,
        );
        hex_digest(payload.as_bytes())
    }

    /// Returns `true` if all 8 components match the other key.
    #[must_use]
    pub fn matches(&self, other: &Self) -> bool {
        self.compute_cache_key() == other.compute_cache_key()
    }

    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "model_content_hash": self.model_content_hash,
            "kernel_abi_version": self.kernel_abi_version,
            "cpu_vendor_model": self.cpu_vendor_model,
            "isa_features": self.isa_features,
            "op_shape_plan": self.op_shape_plan,
            "packing_version": self.packing_version,
            "quant_execution_mode": self.quant_execution_mode,
            "autotune_result_hash": self.autotune_result_hash,
        })
    }

    /// Parses a `PackKey` from a JSON `Value`.
    ///
    /// # Errors
    ///
    /// Returns `PackError::CorruptHeader` if any field is missing or invalid.
    pub fn from_json(val: &Value) -> Result<Self, PackError> {
        let obj = val
            .as_object()
            .ok_or_else(|| PackError::CorruptHeader("expected key object".into()))?;
        let get_str = |k: &str| {
            obj.get(k)
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .ok_or_else(|| PackError::CorruptHeader(format!("missing {k}")))
        };
        let get_u32 = |k: &str| {
            obj.get(k)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| PackError::CorruptHeader(format!("missing {k}")))
        };
        let isa_features = obj
            .get("isa_features")
            .and_then(Value::as_array)
            .ok_or_else(|| PackError::CorruptHeader("missing isa_features array".into()))?
            .iter()
            .filter_map(Value::as_str)
            .map(ToString::to_string)
            .collect();

        Ok(Self {
            model_content_hash: get_str("model_content_hash")?,
            kernel_abi_version: get_u32("kernel_abi_version")?,
            cpu_vendor_model: get_str("cpu_vendor_model")?,
            isa_features,
            op_shape_plan: get_str("op_shape_plan")?,
            packing_version: get_u32("packing_version")?,
            quant_execution_mode: get_str("quant_execution_mode")?,
            autotune_result_hash: get_str("autotune_result_hash")?,
        })
    }
}

/// Packing mode used for an individual weight matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileLayout {
    /// Standard row-major [N, K].
    RowMajor,
    /// Interleaved 4x16 block layout for ARM NEON SDOT (`i8mm` / `sdot` friendly).
    Tile4x16Sdot,
    /// AVX-VNNI 4x32 block layout.
    Tile4x32Vnni,
}

impl TileLayout {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RowMajor => "row_major",
            Self::Tile4x16Sdot => "tile_4x16_sdot",
            Self::Tile4x32Vnni => "tile_4x32_vnni",
        }
    }

    /// Parses a layout from string.
    ///
    /// # Errors
    ///
    /// Returns `PackError::CorruptHeader` on unknown layout string.
    pub fn parse_layout(s: &str) -> Result<Self, PackError> {
        match s {
            "row_major" => Ok(Self::RowMajor),
            "tile_4x16_sdot" => Ok(Self::Tile4x16Sdot),
            "tile_4x32_vnni" => Ok(Self::Tile4x32Vnni),
            other => Err(PackError::CorruptHeader(format!("unknown layout: {other}"))),
        }
    }
}

impl std::str::FromStr for TileLayout {
    type Err = PackError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_layout(s)
    }
}
/// A packed weight tensor ready for zero-copy kernel execution.
#[derive(Debug, Clone, PartialEq)]
pub struct PackedTensor {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub layout: TileLayout,
    pub scales: Vec<f32>,
    pub data: Vec<i8>,
}

impl PackedTensor {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "layout": self.layout.as_str(),
            "scales": self.scales,
            "data": self.data.iter().map(|&b| b as i32).collect::<Vec<_>>(),
        })
    }

    /// Parses a `PackedTensor` from a JSON `Value`.
    ///
    /// # Errors
    ///
    /// Returns `PackError::CorruptHeader` if fields are invalid.
    pub fn from_json(val: &Value) -> Result<Self, PackError> {
        let obj = val
            .as_object()
            .ok_or_else(|| PackError::CorruptHeader("expected tensor object".into()))?;
        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| PackError::CorruptHeader("missing name".into()))?
            .to_string();
        let rows = obj
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| PackError::CorruptHeader("missing rows".into()))?
            as usize;
        let cols = obj
            .get("cols")
            .and_then(Value::as_u64)
            .ok_or_else(|| PackError::CorruptHeader("missing cols".into()))?
            as usize;
        let layout_str = obj
            .get("layout")
            .and_then(Value::as_str)
            .ok_or_else(|| PackError::CorruptHeader("missing layout".into()))?;
        let layout = TileLayout::parse_layout(layout_str)?;

        let scales: Vec<f32> = obj
            .get("scales")
            .and_then(Value::as_array)
            .ok_or_else(|| PackError::CorruptHeader("missing scales".into()))?
            .iter()
            .filter_map(Value::as_f64)
            .map(|f| f as f32)
            .collect();

        let data: Vec<i8> = obj
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| PackError::CorruptHeader("missing data".into()))?
            .iter()
            .filter_map(Value::as_i64)
            .map(|i| i as i8)
            .collect();

        Ok(Self {
            name,
            rows,
            cols,
            layout,
            scales,
            data,
        })
    }
}

/// The physically separated Microdecoder Hot Pack, tuned for cache residency.
#[derive(Debug, Clone, PartialEq)]
pub struct MicrodecoderHotPack {
    pub body_layers: Vec<PackedTensor>,
    pub per_depth_heads: Vec<PackedTensor>,
    pub per_depth_embeddings: Vec<PackedTensor>,
    pub footprint_bytes: usize,
}

impl MicrodecoderHotPack {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "body_layers": self.body_layers.iter().map(PackedTensor::to_json).collect::<Vec<_>>(),
            "per_depth_heads": self.per_depth_heads.iter().map(PackedTensor::to_json).collect::<Vec<_>>(),
            "per_depth_embeddings": self.per_depth_embeddings.iter().map(PackedTensor::to_json).collect::<Vec<_>>(),
            "footprint_bytes": self.footprint_bytes,
        })
    }

    /// Parses a `MicrodecoderHotPack` from JSON.
    ///
    /// # Errors
    ///
    /// Returns `PackError::CorruptHeader` on invalid fields.
    pub fn from_json(val: &Value) -> Result<Self, PackError> {
        let obj = val
            .as_object()
            .ok_or_else(|| PackError::CorruptHeader("expected hot pack object".into()))?;
        let parse_tensors = |key: &str| -> Result<Vec<PackedTensor>, PackError> {
            obj.get(key)
                .and_then(Value::as_array)
                .ok_or_else(|| PackError::CorruptHeader(format!("missing {key}")))?
                .iter()
                .map(PackedTensor::from_json)
                .collect()
        };
        let body_layers = parse_tensors("body_layers")?;
        let per_depth_heads = parse_tensors("per_depth_heads")?;
        let per_depth_embeddings = parse_tensors("per_depth_embeddings")?;
        let footprint_bytes = obj
            .get("footprint_bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| PackError::CorruptHeader("missing footprint_bytes".into()))?
            as usize;

        Ok(Self {
            body_layers,
            per_depth_heads,
            per_depth_embeddings,
            footprint_bytes,
        })
    }
}

/// Persisted autotuner winner mapping per op / shape / regime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedKernelPlan {
    pub decode_gemv_winner: String,
    pub batch_gemm_winner: String,
    pub per_op_winners: BTreeMap<String, String>,
}

impl PersistedKernelPlan {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "decode_gemv_winner": self.decode_gemv_winner,
            "batch_gemm_winner": self.batch_gemm_winner,
            "per_op_winners": self.per_op_winners,
        })
    }

    /// Parses a `PersistedKernelPlan` from JSON.
    ///
    /// # Errors
    ///
    /// Returns `PackError::CorruptHeader` on invalid fields.
    pub fn from_json(val: &Value) -> Result<Self, PackError> {
        let obj = val
            .as_object()
            .ok_or_else(|| PackError::CorruptHeader("expected plan object".into()))?;
        let decode_gemv_winner = obj
            .get("decode_gemv_winner")
            .and_then(Value::as_str)
            .ok_or_else(|| PackError::CorruptHeader("missing decode_gemv_winner".into()))?
            .to_string();
        let batch_gemm_winner = obj
            .get("batch_gemm_winner")
            .and_then(Value::as_str)
            .ok_or_else(|| PackError::CorruptHeader("missing batch_gemm_winner".into()))?
            .to_string();

        let per_op_winners = obj
            .get("per_op_winners")
            .and_then(Value::as_object)
            .ok_or_else(|| PackError::CorruptHeader("missing per_op_winners".into()))?
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();

        Ok(Self {
            decode_gemv_winner,
            batch_gemm_winner,
            per_op_winners,
        })
    }
}

/// The full `.fttspack` execution cache container.
#[derive(Debug, Clone, PartialEq)]
pub struct FttsPack {
    pub key: PackKey,
    pub plan: PersistedKernelPlan,
    pub microdecoder_hot_pack: MicrodecoderHotPack,
    pub talker_tensors: Vec<PackedTensor>,
}

impl FttsPack {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "key": self.key.to_json(),
            "plan": self.plan.to_json(),
            "microdecoder_hot_pack": self.microdecoder_hot_pack.to_json(),
            "talker_tensors": self.talker_tensors.iter().map(PackedTensor::to_json).collect::<Vec<_>>(),
        })
    }

    /// Parses an `FttsPack` from JSON.
    ///
    /// # Errors
    ///
    /// Returns `PackError::CorruptHeader` on invalid fields.
    pub fn from_json(val: &Value) -> Result<Self, PackError> {
        let obj = val
            .as_object()
            .ok_or_else(|| PackError::CorruptHeader("expected pack object".into()))?;
        let key = PackKey::from_json(
            obj.get("key")
                .ok_or_else(|| PackError::CorruptHeader("missing key".into()))?,
        )?;
        let plan = PersistedKernelPlan::from_json(
            obj.get("plan")
                .ok_or_else(|| PackError::CorruptHeader("missing plan".into()))?,
        )?;
        let microdecoder_hot_pack = MicrodecoderHotPack::from_json(
            obj.get("microdecoder_hot_pack")
                .ok_or_else(|| PackError::CorruptHeader("missing hot pack".into()))?,
        )?;
        let talker_tensors = obj
            .get("talker_tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| PackError::CorruptHeader("missing talker_tensors".into()))?
            .iter()
            .map(PackedTensor::from_json)
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            key,
            plan,
            microdecoder_hot_pack,
            talker_tensors,
        })
    }

    /// Encodes the pack into a binary byte vector with header, JSON metadata, and payload.
    ///
    /// # Errors
    ///
    /// Returns `PackError::CorruptHeader` if serialization fails.
    pub fn encode(&self) -> Result<Vec<u8>, PackError> {
        let json_val = self.to_json();
        let json_bytes =
            serde_json::to_vec(&json_val).map_err(|e| PackError::CorruptHeader(e.to_string()))?;
        let json_len = json_bytes.len() as u64;
        let checksum = hex_digest(&json_bytes);

        let mut output = Vec::with_capacity(8 + 4 + 8 + 64 + json_bytes.len());
        output.extend_from_slice(PACK_MAGIC);
        output.extend_from_slice(&PACK_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&json_len.to_le_bytes());
        output.extend_from_slice(checksum.as_bytes()); // 64 ASCII hex bytes
        output.extend_from_slice(&json_bytes);

        Ok(output)
    }

    /// Decodes a pack from binary bytes, verifying magic, version, length, and checksum.
    ///
    /// # Errors
    ///
    /// Returns `PackError` on any corrupt, truncated, or mismatched bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, PackError> {
        if bytes.len() < 8 + 4 + 8 + 64 {
            return Err(PackError::TruncatedPayload {
                expected_bytes: 84,
                actual_bytes: bytes.len(),
            });
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if &magic != PACK_MAGIC {
            return Err(PackError::BadMagic(magic));
        }

        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| PackError::CorruptHeader("bad version offset".into()))?,
        );
        if version != PACK_FORMAT_VERSION {
            return Err(PackError::UnsupportedVersion(version));
        }

        let json_len_u64 = u64::from_le_bytes(
            bytes[12..20]
                .try_into()
                .map_err(|_| PackError::CorruptHeader("bad json length offset".into()))?,
        );
        let json_len = usize::try_from(json_len_u64).map_err(|_| {
            PackError::CorruptHeader("json length exceeds platform address space".into())
        })?;

        let stored_checksum = std::str::from_utf8(&bytes[20..84])
            .map_err(|_| PackError::CorruptHeader("non-utf8 checksum".into()))?;

        let payload_start: usize = 84;
        let payload_end = payload_start
            .checked_add(json_len)
            .ok_or_else(|| PackError::CorruptHeader("json payload offset overflow".into()))?;
        if bytes.len() < payload_end {
            return Err(PackError::TruncatedPayload {
                expected_bytes: payload_end,
                actual_bytes: bytes.len(),
            });
        }

        let json_slice = &bytes[payload_start..payload_end];
        let calculated_checksum = hex_digest(json_slice);
        if calculated_checksum != stored_checksum {
            return Err(PackError::ChecksumMismatch {
                expected: stored_checksum.to_string(),
                actual: calculated_checksum,
            });
        }

        let val: Value = serde_json::from_slice(json_slice)
            .map_err(|e| PackError::CorruptHeader(e.to_string()))?;
        Self::from_json(&val)
    }

    /// Writes the pack atomically to the given file path.
    ///
    /// # Errors
    ///
    /// Returns `PackError` if file write or encoding fails.
    pub fn write_to_file(&self, path: &Path) -> Result<(), PackError> {
        let bytes = self.encode()?;
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let temp_path = path.with_extension("tmp");
        {
            let mut file = File::create(&temp_path)
                .map_err(|e| PackError::Io(format!("create {}: {e}", temp_path.display())))?;
            file.write_all(&bytes)
                .map_err(|e| PackError::Io(format!("write {}: {e}", temp_path.display())))?;
            file.flush()
                .map_err(|e| PackError::Io(format!("flush {}: {e}", temp_path.display())))?;
        }
        std::fs::rename(&temp_path, path)
            .map_err(|e| PackError::Io(format!("rename to {}: {e}", path.display())))?;
        Ok(())
    }

    /// Reads and validates a pack from the given file path against an expected key.
    ///
    /// # Errors
    ///
    /// Returns `PackError` if reading, decoding, or key verification fails.
    pub fn read_from_file_checked(path: &Path, expected_key: &PackKey) -> Result<Self, PackError> {
        let mut file =
            File::open(path).map_err(|e| PackError::Io(format!("open {}: {e}", path.display())))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| PackError::Io(format!("read {}: {e}", path.display())))?;
        let pack = Self::decode(&bytes)?;
        if !pack.key.matches(expected_key) {
            return Err(PackError::KeyMismatch {
                expected: expected_key.compute_cache_key(),
                actual: pack.key.compute_cache_key(),
            });
        }
        Ok(pack)
    }
}

/// Packs a row-major [N, K] matrix into 4x16 SDOT-interleaved tiles.
#[must_use]
pub fn pack_matrix_sdot_4x16(matrix: &[i8], rows: usize, cols: usize) -> Vec<i8> {
    assert_eq!(matrix.len(), rows * cols);
    assert_eq!(rows % 4, 0, "rows must be a multiple of 4 for 4x16 tiling");
    assert_eq!(
        cols % 16,
        0,
        "cols must be a multiple of 16 for 4x16 tiling"
    );
    let mut packed = vec![0i8; rows * cols];
    let row_blocks = rows / 4;
    let col_blocks = cols / 16;

    let mut dst_idx = 0;
    for rb in 0..row_blocks {
        let r_base = rb * 4;
        for cb in 0..col_blocks {
            let c_base = cb * 16;
            for r_offset in 0..4 {
                let r = r_base + r_offset;
                for c_offset in 0..16 {
                    let c = c_base + c_offset;
                    packed[dst_idx] = matrix[r * cols + c];
                    dst_idx += 1;
                }
            }
        }
    }
    debug_assert_eq!(dst_idx, rows * cols);
    packed
}

/// Unpacks a 4x16 SDOT-interleaved matrix back into canonical row-major [N, K].
#[must_use]
pub fn unpack_matrix_sdot_4x16(packed: &[i8], rows: usize, cols: usize) -> Vec<i8> {
    assert_eq!(packed.len(), rows * cols);
    assert_eq!(rows % 4, 0, "rows must be a multiple of 4 for 4x16 tiling");
    assert_eq!(
        cols % 16,
        0,
        "cols must be a multiple of 16 for 4x16 tiling"
    );
    let mut unpacked = vec![0i8; rows * cols];
    let row_blocks = rows / 4;
    let col_blocks = cols / 16;

    let mut src_idx = 0;
    for rb in 0..row_blocks {
        let r_base = rb * 4;
        for cb in 0..col_blocks {
            let c_base = cb * 16;
            for r_offset in 0..4 {
                let r = r_base + r_offset;
                for c_offset in 0..16 {
                    let c = c_base + c_offset;
                    unpacked[r * cols + c] = packed[src_idx];
                    src_idx += 1;
                }
            }
        }
    }
    debug_assert_eq!(src_idx, rows * cols);
    unpacked
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_key() -> PackKey {
        PackKey {
            model_content_hash: "a1b2c3d4e5f67890".into(),
            kernel_abi_version: 1,
            cpu_vendor_model: "Apple M4 Pro".into(),
            isa_features: vec!["neon".into(), "dotprod".into(), "i8mm".into()],
            op_shape_plan: "qwen3_tts_v1".into(),
            packing_version: 1,
            quant_execution_mode: "w8a8_dynamic".into(),
            autotune_result_hash: "deadbeefcafe".into(),
        }
    }

    fn sample_pack() -> FttsPack {
        let key = sample_key();
        let plan = PersistedKernelPlan {
            decode_gemv_winner: "neon-sdot".into(),
            batch_gemm_winner: "neon-sdot".into(),
            per_op_winners: BTreeMap::from([
                ("q_proj".into(), "neon-sdot".into()),
                ("down_proj".into(), "scalar".into()),
            ]),
        };
        let micro_tensor = PackedTensor {
            name: "microdecoder.l0.q_proj".into(),
            rows: 2048,
            cols: 1024,
            layout: TileLayout::Tile4x16Sdot,
            scales: vec![0.05; 2048],
            data: vec![7i8; 2048 * 1024],
        };
        let hot_pack = MicrodecoderHotPack {
            body_layers: vec![micro_tensor],
            per_depth_heads: vec![],
            per_depth_embeddings: vec![],
            footprint_bytes: 2048 * 1024,
        };
        FttsPack {
            key,
            plan,
            microdecoder_hot_pack: hot_pack,
            talker_tensors: vec![],
        }
    }

    #[test]
    fn roundtrip_pack_encode_decode() {
        let pack = sample_pack();
        let encoded = pack.encode().expect("encode pack");
        let decoded = FttsPack::decode(&encoded).expect("decode pack");
        assert_eq!(pack, decoded);
    }

    #[test]
    fn key_invalidation_detects_all_8_tuple_mutations() {
        let base = sample_key();
        let base_digest = base.compute_cache_key();

        // 1. Model content hash
        let mut k1 = base.clone();
        k1.model_content_hash = "mutated_hash".into();
        assert_ne!(base_digest, k1.compute_cache_key());
        assert!(!base.matches(&k1));

        // 2. Kernel ABI version
        let mut k2 = base.clone();
        k2.kernel_abi_version = 2;
        assert_ne!(base_digest, k2.compute_cache_key());
        assert!(!base.matches(&k2));

        // 3. CPU vendor / model
        let mut k3 = base.clone();
        k3.cpu_vendor_model = "AMD Ryzen 9".into();
        assert_ne!(base_digest, k3.compute_cache_key());
        assert!(!base.matches(&k3));

        // 4. ISA feature set
        let mut k4 = base.clone();
        k4.isa_features = vec!["neon".into()];
        assert_ne!(base_digest, k4.compute_cache_key());
        assert!(!base.matches(&k4));

        // 5. Op shape plan
        let mut k5 = base.clone();
        k5.op_shape_plan = "qwen3_tts_v2".into();
        assert_ne!(base_digest, k5.compute_cache_key());
        assert!(!base.matches(&k5));

        // 6. Packing version
        let mut k6 = base.clone();
        k6.packing_version = 2;
        assert_ne!(base_digest, k6.compute_cache_key());
        assert!(!base.matches(&k6));

        // 7. Quant execution mode
        let mut k7 = base.clone();
        k7.quant_execution_mode = "w8a16".into();
        assert_ne!(base_digest, k7.compute_cache_key());
        assert!(!base.matches(&k7));

        // 8. Autotune result hash
        let mut k8 = base.clone();
        k8.autotune_result_hash = "different_timings".into();
        assert_ne!(base_digest, k8.compute_cache_key());
        assert!(!base.matches(&k8));
    }

    #[test]
    fn sdot_4x16_packing_roundtrip_is_exact() {
        let rows = 128;
        let cols = 64;
        let original: Vec<i8> = (0..(rows * cols)).map(|i| (i % 127) as i8).collect();
        let packed = pack_matrix_sdot_4x16(&original, rows, cols);
        let unpacked = unpack_matrix_sdot_4x16(&packed, rows, cols);
        assert_eq!(original, unpacked);
    }

    #[test]
    fn corrupted_checksum_is_rejected() {
        let pack = sample_pack();
        let mut encoded = pack.encode().expect("encode pack");
        // Tamper with one byte in the payload
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;
        let err = FttsPack::decode(&encoded).expect_err("must reject tampered payload");
        assert!(matches!(
            err,
            PackError::ChecksumMismatch { .. } | PackError::CorruptHeader(_)
        ));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut encoded = sample_pack().encode().expect("encode");
        encoded[0] = b'X';
        let err = FttsPack::decode(&encoded).expect_err("bad magic");
        assert!(matches!(err, PackError::BadMagic(_)));
    }

    #[test]
    fn file_atomic_write_and_checked_read() {
        let dir = std::env::temp_dir().join(format!("fttspack_test_{}", std::process::id()));
        let file_path = dir.join("cache.fttspack");
        let pack = sample_pack();

        pack.write_to_file(&file_path).expect("write pack to file");
        assert!(file_path.exists());

        // Valid read
        let loaded =
            FttsPack::read_from_file_checked(&file_path, &pack.key).expect("read valid pack");
        assert_eq!(pack, loaded);

        // Mismatched key read fails
        let mut invalid_key = pack.key.clone();
        invalid_key.packing_version += 1;
        let err = FttsPack::read_from_file_checked(&file_path, &invalid_key)
            .expect_err("must fail on mismatched key");
        assert!(matches!(err, PackError::KeyMismatch { .. }));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
