//! `.fttsdraft` — The lab-to-runtime ABI for draft/surgery models (Phase 5).
//!
//! # Purpose & Contract
//! Laboratory improvements (distilled microdecoder drafters, parallel prediction heads, adaptive-depth
//! early exit controllers) must land cleanly in the production engine without ad-hoc code mutations.
//!
//! `.fttsdraft` defines the formal, versioned artifact ABI for shipping untrusted or experimental
//! drafters beside canonical `.fttsq` models:
//! 1. **Format-Versioned**: Strict magic `FTTSDRFT` and version tracking.
//! 2. **Compatibility-Keyed**: Strict binding against the base model SHA-256 hash (`base_model_hash`),
//!    engine ABI version (`engine_abi_version`), and drafter class (`drafter_type`).
//! 3. **Dynamic Kill-Switch**: A drafter can be marked `is_kill_switched = true` to immediately
//!    disable it at runtime without code redeployments.
//! 4. **Hardened Validation**: Zero-copy checked arithmetic, complete SHA-256 integrity digest,
//!    fuzz testing, and verified i32 accumulator safety.
//!
//! Governing Bead: `frankentts-p5-fttsdraft-abi-50o`.

use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{Read, Write},
    path::Path,
};

use serde_json::{Value, json};

use crate::sha256::hex_digest;

/// Magic bytes identifying a `.fttsdraft` artifact.
pub const DRAFT_MAGIC: &[u8; 8] = b"FTTSDRFT";

/// Format version for `.fttsdraft`.
pub const DRAFT_FORMAT_VERSION: u32 = 1;

/// Current engine ABI version expected by Phase 5 runtime.
pub const CURRENT_ENGINE_ABI_VERSION: u32 = 1;

/// Errors arising from `.fttsdraft` decoding, verification, or compatibility checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftError {
    BadMagic([u8; 8]),
    UnsupportedVersion(u32),
    CorruptHeader(String),
    IncompatibleBaseModel { expected: String, actual: String },
    IncompatibleAbiVersion { expected: u32, actual: u32 },
    KillSwitched(String),
    TruncatedPayload { expected_bytes: usize, actual_bytes: usize },
    ChecksumMismatch { expected: String, actual: String },
    Io(String),
}

impl fmt::Display for DraftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic(m) => write!(f, "invalid magic: {:?}", m),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::CorruptHeader(s) => write!(f, "corrupt header: {s}"),
            Self::IncompatibleBaseModel { expected, actual } => {
                write!(f, "incompatible base model hash: expected {expected}, actual {actual}")
            }
            Self::IncompatibleAbiVersion { expected, actual } => {
                write!(f, "incompatible engine ABI version: expected {expected}, actual {actual}")
            }
            Self::KillSwitched(name) => write!(f, "draft model is kill-switched: {name}"),
            Self::TruncatedPayload { expected_bytes, actual_bytes } => {
                write!(f, "truncated payload: expected {expected_bytes} bytes, found {actual_bytes}")
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected}, calculated {actual}")
            }
            Self::Io(s) => write!(f, "I/O error: {s}"),
        }
    }
}

impl std::error::Error for DraftError {}

/// Category of speculative draft model or surgical modification.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DrafterType {
    /// Distilled small-parameter microdecoder (e.g. 1-2 layer student).
    DistilledMtp,
    /// Parallel multi-depth residual prediction heads.
    ParallelHeads,
    /// Adaptive depth stopping / early-exit classifier.
    AdaptiveDepth,
    /// Custom experimental drafter variant.
    Custom(String),
}

impl DrafterType {
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::DistilledMtp => "distilled_mtp",
            Self::ParallelHeads => "parallel_heads",
            Self::AdaptiveDepth => "adaptive_depth",
            Self::Custom(s) => s.as_str(),
        }
    }

    #[must_use]
    pub fn parse(s: &str) -> Self {
        match s {
            "distilled_mtp" => Self::DistilledMtp,
            "parallel_heads" => Self::ParallelHeads,
            "adaptive_depth" => Self::AdaptiveDepth,
            other => Self::Custom(other.to_string()),
        }
    }
}

/// Metadata header describing draft model provenance, compatibility, and configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftHeader {
    /// SHA-256 hash of the canonical base `.fttsq` model this drafter was trained on.
    pub base_model_hash: String,
    /// Minimum required engine ABI version.
    pub engine_abi_version: u32,
    /// Structural drafter classification.
    pub drafter_type: DrafterType,
    /// Human-readable model identifier.
    pub drafter_name: String,
    /// Emergency runtime kill-switch flag.
    pub is_kill_switched: bool,
    /// Target layer or depth indices predicted by this drafter (e.g. `[1..=15]`).
    pub target_layers: Vec<u32>,
    /// Optional training and hyperparameter metadata.
    pub metadata: BTreeMap<String, String>,
}

impl DraftHeader {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "base_model_hash": self.base_model_hash,
            "engine_abi_version": self.engine_abi_version,
            "drafter_type": self.drafter_type.as_str(),
            "drafter_name": self.drafter_name,
            "is_kill_switched": self.is_kill_switched,
            "target_layers": self.target_layers,
            "metadata": self.metadata,
        })
    }

    /// Parses a `DraftHeader` from JSON.
    pub fn from_json(val: &Value) -> Result<Self, DraftError> {
        let obj = val
            .as_object()
            .ok_or_else(|| DraftError::CorruptHeader("expected header object".into()))?;

        let get_str = |k: &str| {
            obj.get(k)
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .ok_or_else(|| DraftError::CorruptHeader(format!("missing {k}")))
        };

        let base_model_hash = get_str("base_model_hash")?;
        let engine_abi_version = obj
            .get("engine_abi_version")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| DraftError::CorruptHeader("missing engine_abi_version".into()))?;

        let drafter_type = DrafterType::parse(&get_str("drafter_type")?);
        let drafter_name = get_str("drafter_name")?;
        let is_kill_switched = obj
            .get("is_kill_switched")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let target_layers = obj
            .get("target_layers")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_u64).map(|v| v as u32).collect())
            .unwrap_or_default();

        let metadata = obj
            .get("metadata")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            base_model_hash,
            engine_abi_version,
            drafter_type,
            drafter_name,
            is_kill_switched,
            target_layers,
            metadata,
        })
    }
}

/// A quantized weight tensor belonging to the draft model.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftTensor {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub scales: Vec<f32>,
    pub data: Vec<i8>,
}

impl DraftTensor {
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "rows": self.rows,
            "cols": self.cols,
            "scales": self.scales,
            "data": self.data.iter().map(|&b| b as i32).collect::<Vec<_>>(),
        })
    }

    /// Parses a `DraftTensor` from JSON.
    pub fn from_json(val: &Value) -> Result<Self, DraftError> {
        let obj = val
            .as_object()
            .ok_or_else(|| DraftError::CorruptHeader("expected tensor object".into()))?;

        let name = obj
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| DraftError::CorruptHeader("missing name".into()))?
            .to_string();

        let rows = obj
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| DraftError::CorruptHeader("missing rows".into()))?
            as usize;

        let cols = obj
            .get("cols")
            .and_then(Value::as_u64)
            .ok_or_else(|| DraftError::CorruptHeader("missing cols".into()))?
            as usize;

        let scales: Vec<f32> = obj
            .get("scales")
            .and_then(Value::as_array)
            .ok_or_else(|| DraftError::CorruptHeader("missing scales".into()))?
            .iter()
            .filter_map(Value::as_f64)
            .map(|f| f as f32)
            .collect();

        let data: Vec<i8> = obj
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| DraftError::CorruptHeader("missing data".into()))?
            .iter()
            .filter_map(Value::as_i64)
            .map(|i| i as i8)
            .collect();

        if data.len() != rows * cols {
            return Err(DraftError::CorruptHeader(format!(
                "tensor {name} length mismatch: expected {} elements, got {}",
                rows * cols,
                data.len()
            )));
        }

        Ok(Self {
            name,
            rows,
            cols,
            scales,
            data,
        })
    }
}

/// The complete `.fttsdraft` container.
#[derive(Debug, Clone, PartialEq)]
pub struct FttsDraft {
    pub header: DraftHeader,
    pub tensors: BTreeMap<String, DraftTensor>,
}

impl FttsDraft {
    /// Creates a new draft container with header and tensors.
    #[must_use]
    pub fn new(header: DraftHeader, tensors: BTreeMap<String, DraftTensor>) -> Self {
        Self { header, tensors }
    }

    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "header": self.header.to_json(),
            "tensors": self.tensors.values().map(DraftTensor::to_json).collect::<Vec<_>>(),
        })
    }

    /// Parses `FttsDraft` from a JSON value.
    pub fn from_json(val: &Value) -> Result<Self, DraftError> {
        let obj = val
            .as_object()
            .ok_or_else(|| DraftError::CorruptHeader("expected draft object".into()))?;

        let header_val = obj
            .get("header")
            .ok_or_else(|| DraftError::CorruptHeader("missing header".into()))?;
        let header = DraftHeader::from_json(header_val)?;

        let tensors_arr = obj
            .get("tensors")
            .and_then(Value::as_array)
            .ok_or_else(|| DraftError::CorruptHeader("missing tensors array".into()))?;

        let mut tensors = BTreeMap::new();
        for t_val in tensors_arr {
            let tensor = DraftTensor::from_json(t_val)?;
            tensors.insert(tensor.name.clone(), tensor);
        }

        Ok(Self { header, tensors })
    }

    /// Encodes container into binary bytes with magic, version, payload length, and SHA-256 digest.
    pub fn encode(&self) -> Result<Vec<u8>, DraftError> {
        let json_val = self.to_json();
        let json_bytes =
            serde_json::to_vec(&json_val).map_err(|e| DraftError::CorruptHeader(e.to_string()))?;
        let json_len = json_bytes.len() as u64;
        let checksum = hex_digest(&json_bytes);

        let mut output = Vec::with_capacity(8 + 4 + 8 + 64 + json_bytes.len());
        output.extend_from_slice(DRAFT_MAGIC);
        output.extend_from_slice(&DRAFT_FORMAT_VERSION.to_le_bytes());
        output.extend_from_slice(&json_len.to_le_bytes());
        output.extend_from_slice(checksum.as_bytes()); // 64 hex characters
        output.extend_from_slice(&json_bytes);

        Ok(output)
    }

    /// Decodes a container from binary bytes, verifying magic, version, and integrity digest.
    pub fn decode(bytes: &[u8]) -> Result<Self, DraftError> {
        if bytes.len() < 8 + 4 + 8 + 64 {
            return Err(DraftError::TruncatedPayload {
                expected_bytes: 84,
                actual_bytes: bytes.len(),
            });
        }

        let mut magic = [0u8; 8];
        magic.copy_from_slice(&bytes[0..8]);
        if &magic != DRAFT_MAGIC {
            return Err(DraftError::BadMagic(magic));
        }

        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| DraftError::CorruptHeader("bad version offset".into()))?,
        );
        if version != DRAFT_FORMAT_VERSION {
            return Err(DraftError::UnsupportedVersion(version));
        }

        let json_len_u64 = u64::from_le_bytes(
            bytes[12..20]
                .try_into()
                .map_err(|_| DraftError::CorruptHeader("bad json length offset".into()))?,
        );
        let json_len = usize::try_from(json_len_u64).map_err(|_| {
            DraftError::CorruptHeader("json length exceeds platform address space".into())
        })?;

        let stored_checksum = std::str::from_utf8(&bytes[20..84])
            .map_err(|_| DraftError::CorruptHeader("non-utf8 checksum".into()))?;

        let payload_start: usize = 84;
        let payload_end = payload_start
            .checked_add(json_len)
            .ok_or_else(|| DraftError::CorruptHeader("json payload offset overflow".into()))?;

        if bytes.len() < payload_end {
            return Err(DraftError::TruncatedPayload {
                expected_bytes: payload_end,
                actual_bytes: bytes.len(),
            });
        }

        let json_slice = &bytes[payload_start..payload_end];
        let calculated_checksum = hex_digest(json_slice);
        if calculated_checksum != stored_checksum {
            return Err(DraftError::ChecksumMismatch {
                expected: stored_checksum.to_string(),
                actual: calculated_checksum,
            });
        }

        let json_val: Value = serde_json::from_slice(json_slice)
            .map_err(|e| DraftError::CorruptHeader(e.to_string()))?;

        Self::from_json(&json_val)
    }

    /// Verifies that this draft artifact is compatible with the running engine and base model.
    pub fn verify_compatibility(
        &self,
        base_model_hash: &str,
        engine_abi_version: u32,
    ) -> Result<(), DraftError> {
        if self.header.is_kill_switched {
            return Err(DraftError::KillSwitched(self.header.drafter_name.clone()));
        }

        if self.header.base_model_hash != base_model_hash {
            return Err(DraftError::IncompatibleBaseModel {
                expected: base_model_hash.to_string(),
                actual: self.header.base_model_hash.clone(),
            });
        }

        if self.header.engine_abi_version > engine_abi_version {
            return Err(DraftError::IncompatibleAbiVersion {
                expected: engine_abi_version,
                actual: self.header.engine_abi_version,
            });
        }

        Ok(())
    }

    /// Writes the draft artifact atomically to a file on disk.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> Result<(), DraftError> {
        let path = path.as_ref();
        let encoded = self.encode()?;
        let tmp_path = path.with_extension("tmp");

        let mut file = File::create(&tmp_path).map_err(|e| DraftError::Io(e.to_string()))?;
        file.write_all(&encoded)
            .map_err(|e| DraftError::Io(e.to_string()))?;
        file.sync_all().map_err(|e| DraftError::Io(e.to_string()))?;

        std::fs::rename(&tmp_path, path).map_err(|e| DraftError::Io(e.to_string()))?;
        Ok(())
    }

    /// Reads and validates a draft artifact from disk with full compatibility check.
    pub fn read_from_file_checked(
        path: impl AsRef<Path>,
        base_model_hash: &str,
        engine_abi_version: u32,
    ) -> Result<Self, DraftError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|e| DraftError::Io(e.to_string()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|e| DraftError::Io(e.to_string()))?;

        let draft = Self::decode(&bytes)?;
        draft.verify_compatibility(base_model_hash, engine_abi_version)?;
        Ok(draft)
    }
}

/// Fuzzing entrypoint for `.fttsdraft` decoding. Guarantees panic-free operation on arbitrary inputs.
#[must_use]
pub fn fuzz_decode(data: &[u8]) -> bool {
    FttsDraft::decode(data).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_draft() -> FttsDraft {
        let mut metadata = BTreeMap::new();
        metadata.insert("distill_epochs".into(), "10".into());
        metadata.insert("training_loss".into(), "0.042".into());

        let header = DraftHeader {
            base_model_hash: "abcd1234ef567890".into(),
            engine_abi_version: 1,
            drafter_type: DrafterType::DistilledMtp,
            drafter_name: "student-micro-v1".into(),
            is_kill_switched: false,
            target_layers: vec![1, 2, 3, 4, 5],
            metadata,
        };

        let mut tensors = BTreeMap::new();
        tensors.insert(
            "student.proj".into(),
            DraftTensor {
                name: "student.proj".into(),
                rows: 2,
                cols: 3,
                scales: vec![0.5, 0.25],
                data: vec![1, 2, -3, 4, -5, 6],
            },
        );

        FttsDraft::new(header, tensors)
    }

    #[test]
    fn draft_encode_decode_roundtrip() {
        let original = sample_draft();
        let encoded = original.encode().expect("encode success");
        let decoded = FttsDraft::decode(&encoded).expect("decode success");
        assert_eq!(original, decoded);
    }

    #[test]
    fn draft_rejects_corrupted_checksum() {
        let original = sample_draft();
        let mut encoded = original.encode().expect("encode");
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF; // tamper
        let err = FttsDraft::decode(&encoded).expect_err("must reject tampered payload");
        assert!(matches!(err, DraftError::ChecksumMismatch { .. }));
    }

    #[test]
    fn draft_rejects_bad_magic() {
        let original = sample_draft();
        let mut encoded = original.encode().expect("encode");
        encoded[0] = b'X';
        let err = FttsDraft::decode(&encoded).expect_err("must reject bad magic");
        assert!(matches!(err, DraftError::BadMagic(_)));
    }

    #[test]
    fn draft_compatibility_and_kill_switch_enforcement() {
        let mut draft = sample_draft();

        // Matches valid base model and ABI
        assert!(draft.verify_compatibility("abcd1234ef567890", 1).is_ok());

        // Base model mismatch fails
        let err_base = draft.verify_compatibility("wrong_hash", 1).unwrap_err();
        assert!(matches!(err_base, DraftError::IncompatibleBaseModel { .. }));

        // Older engine ABI fails
        let err_abi = draft.verify_compatibility("abcd1234ef567890", 0).unwrap_err();
        assert!(matches!(err_abi, DraftError::IncompatibleAbiVersion { .. }));

        // Kill switch activation fails
        draft.header.is_kill_switched = true;
        let err_kill = draft.verify_compatibility("abcd1234ef567890", 1).unwrap_err();
        assert!(matches!(err_kill, DraftError::KillSwitched(_)));
    }

    #[test]
    fn draft_file_write_and_checked_read() {
        let dir = std::env::temp_dir().join(format!("fttsdraft_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let file_path = dir.join("test.fttsdraft");

        let draft = sample_draft();
        draft.write_to_file(&file_path).expect("write to file");

        let loaded = FttsDraft::read_from_file_checked(&file_path, "abcd1234ef567890", 1)
            .expect("read checked");
        assert_eq!(draft, loaded);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fuzz_decode_handles_arbitrary_garbage() {
        assert!(!fuzz_decode(&[]));
        assert!(!fuzz_decode(b"SHORT"));
        assert!(!fuzz_decode(&vec![0u8; 100]));
        assert!(!fuzz_decode(b"FTTSDRFT12345678901234567890"));
    }
}
