//! `.fttsq` — the canonical, portable, quantized model container.
//!
//! # What this format is, and what it deliberately is not
//!
//! `.fttsq` is **portable and machine-independent**: tensor payloads, quantization scales, the
//! frozen model config, provenance, and the license notice. It carries **no machine-specific
//! tiling** — packed kernel layouts and the autotuned plan live in `.fttspack`, which is a
//! regenerable per-machine cache. That split is the whole point: kernel layouts change often, and
//! republishing a multi-gigabyte artifact every time a tile shape improves is not a cost we are
//! willing to pay.
//!
//! # Access classes
//!
//! Sections are grouped by how the runtime *touches* them, because the paging policy differs by an
//! order of magnitude between them:
//!
//! | Class | Rough size | Access pattern |
//! |---|---|---|
//! | [`AccessClass::HotRecurrentMicrodecoder`] | ~110 MB | reread 15× per frame — the residency target |
//! | [`AccessClass::HotRecurrentTalker`] | ~440 MB | 28 layers, once per frame |
//! | [`AccessClass::HotCodecDecoder`] | smaller | once per frame, latency-critical |
//! | [`AccessClass::ColdTextEmbedding`] | ~622 MB | **row-granular**; never paged in wholesale |
//! | [`AccessClass::EnrollmentSpeakerEncoder`] / [`AccessClass::EnrollmentCodecEncoder`] | optional | enrollment only |
//! | [`AccessClass::Metadata`] | tiny | config, manifests, license |
//!
//! The cold text embedding is 622 MB that a synthesis run touches a few kilobytes of. Faulting it
//! in as a unit would dominate startup and evict the microdecoder pack that the whole optimization
//! program is built around, so its class is a load-bearing declaration, not a label.
//!
//! # Hardening
//!
//! This reader parses attacker-influenced binary blobs — an artifact is just a file someone gives
//! you. Every length and offset is checked arithmetic against the real file length, section ranges
//! may not overlap, tensor ranges must lie inside their section, counts and dimensions are capped,
//! and every section is digest-verified before its bytes are handed out. A malformed artifact is a
//! **named refusal**, never a partial load that resurfaces later as garbage audio.
//!
//! # Layout
//!
//! ```text
//! [0  .. 8 )   magic          b"FTTSQ\0\0\0"
//! [8  ..12 )   format_version u32 little-endian
//! [12 ..20 )   directory_len  u64 little-endian
//! [20 ..20+D)  directory      UTF-8 JSON
//! [20+D..   )  section payloads, at absolute offsets named in the directory
//! ```
//!
//! Bead: `frankentts-p2-fttsq-format-wsa`.

use std::{collections::BTreeMap, fmt};

use serde_json::{Value, json};

use crate::sha256::{Sha256, hex_digest, to_hex};

/// File magic. Eight bytes so the directory length lands 8-byte aligned.
pub const MAGIC: &[u8; 8] = b"FTTSQ\0\0\0";

/// The format version this binary writes and is the newest it can read.
///
/// A reader **refuses** anything newer: a future version may relocate bytes this binary would
/// otherwise misinterpret, and "read it anyway and hope" is how a container format acquires
/// silent, version-dependent corruption.
pub const FORMAT_VERSION: u32 = 1;

/// Fixed prefix length: magic + version + directory length.
pub const HEADER_PREFIX_BYTES: u64 = 20;

/// Largest directory we will parse, guarding against a hostile length prefix.
///
/// The real checkpoint's directory is a few hundred KiB. Matches `safetensors::MAX_HEADER_BYTES`
/// deliberately — two artifact readers with different limits is a bug waiting to be found.
pub const MAX_DIRECTORY_BYTES: u64 = 64 * 1024 * 1024;

/// Most sections an artifact may declare. Seven access classes, with headroom for splitting.
pub const MAX_SECTIONS: usize = 64;

/// Most tensors an artifact may declare. The pinned checkpoint has 974.
pub const MAX_TENSORS: usize = 16_384;

/// Most dimensions one tensor may have. The pinned checkpoint's maximum rank is 4.
pub const MAX_RANK: usize = 8;

/// Largest single dimension. The largest real one is the 151,936-row text embedding.
pub const MAX_DIM: u64 = 1 << 32;

/// How the runtime touches a section. Drives the page-in policy.
///
/// Unknown values are refused rather than defaulted: a section whose access class we do not
/// understand is one we cannot page correctly, and guessing "probably hot" for a 622 MB cold
/// section is exactly the mistake that costs the microdecoder its cache residency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessClass {
    /// The ~110 MB microdecoder hot pack, reread 15× per frame.
    HotRecurrentMicrodecoder,
    /// The ~440 MB talker body, read once per frame across 28 layers.
    HotRecurrentTalker,
    /// The codec decoder, read once per frame.
    HotCodecDecoder,
    /// The ~622 MB text embedding: lazy, row-granular, never paged in wholesale.
    ColdTextEmbedding,
    /// Speaker encoder, needed only during enrollment.
    EnrollmentSpeakerEncoder,
    /// Codec encoder, needed only during enrollment.
    EnrollmentCodecEncoder,
    /// Config, manifests, and the license notice.
    Metadata,
}

impl AccessClass {
    /// The stable wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HotRecurrentMicrodecoder => "HOT_RECURRENT_MICRODECODER",
            Self::HotRecurrentTalker => "HOT_RECURRENT_TALKER",
            Self::HotCodecDecoder => "HOT_CODEC_DECODER",
            Self::ColdTextEmbedding => "COLD_TEXT_EMBEDDING",
            Self::EnrollmentSpeakerEncoder => "ENROLLMENT_SPEAKER_ENCODER",
            Self::EnrollmentCodecEncoder => "ENROLLMENT_CODEC_ENCODER",
            Self::Metadata => "METADATA",
        }
    }

    /// Parses a wire string, refusing anything unrecognized.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "HOT_RECURRENT_MICRODECODER" => Self::HotRecurrentMicrodecoder,
            "HOT_RECURRENT_TALKER" => Self::HotRecurrentTalker,
            "HOT_CODEC_DECODER" => Self::HotCodecDecoder,
            "COLD_TEXT_EMBEDDING" => Self::ColdTextEmbedding,
            "ENROLLMENT_SPEAKER_ENCODER" => Self::EnrollmentSpeakerEncoder,
            "ENROLLMENT_CODEC_ENCODER" => Self::EnrollmentCodecEncoder,
            "METADATA" => Self::Metadata,
            _ => return None,
        })
    }

    /// Whether this section should be resident during steady-state decode.
    ///
    /// Consumed by the page-in policy: hot classes are advised resident, the cold embedding is
    /// explicitly not, and enrollment sections are only touched by the voice compiler.
    #[must_use]
    pub const fn is_hot(self) -> bool {
        matches!(
            self,
            Self::HotRecurrentMicrodecoder | Self::HotRecurrentTalker | Self::HotCodecDecoder
        )
    }

    /// Whether the runtime must access this section row-granularly rather than as a unit.
    #[must_use]
    pub const fn is_row_granular(self) -> bool {
        matches!(self, Self::ColdTextEmbedding)
    }
}

impl fmt::Display for AccessClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a tensor's elements are stored in the container.
///
/// Narrow on purpose, and unknown values are refused. The high-precision variants exist because
/// the quant recipe protects norms, codebooks, and the speaker path — an artifact that claims a
/// dtype we have never conformed is a refusal, not a best-effort read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StoredDtype {
    /// bfloat16, kept verbatim from the checkpoint.
    Bf16,
    /// 32-bit float.
    F32,
    /// int8 with separate per-group scales.
    Q8,
    /// int4, two elements per byte, with separate per-group scales.
    Q4,
}

impl StoredDtype {
    /// The stable wire string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::Q8 => "q8",
            Self::Q4 => "q4",
        }
    }

    /// Parses a wire string, refusing anything unrecognized.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "bf16" => Self::Bf16,
            "f32" => Self::F32,
            "q8" => Self::Q8,
            "q4" => Self::Q4,
            _ => return None,
        })
    }

    /// Storage bytes for `elements` values, or `None` on overflow.
    ///
    /// Q4 packs two elements per byte and rounds up, so an odd element count still occupies a whole
    /// trailing byte.
    #[must_use]
    pub const fn storage_bytes(self, elements: u64) -> Option<u64> {
        match self {
            Self::Bf16 => elements.checked_mul(2),
            Self::F32 => elements.checked_mul(4),
            Self::Q8 => Some(elements),
            Self::Q4 => match elements.checked_add(1) {
                Some(padded) => Some(padded / 2),
                None => None,
            },
        }
    }
}

impl fmt::Display for StoredDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One access-class section: a contiguous, digest-verified byte range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionEntry {
    /// Section name, unique within the artifact.
    pub name: String,
    /// How the runtime touches it.
    pub access_class: AccessClass,
    /// Absolute offset into the file.
    pub offset: u64,
    /// Byte length.
    pub length: u64,
    /// Lowercase-hex SHA-256 of the section bytes.
    pub sha256: String,
}

impl SectionEntry {
    /// Exclusive end offset, or `None` on overflow.
    #[must_use]
    pub const fn end(&self) -> Option<u64> {
        self.offset.checked_add(self.length)
    }
}

/// One tensor, located relative to the start of its section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    /// Tensor name, unique within the artifact.
    pub name: String,
    /// Name of the section holding it.
    pub section: String,
    /// Storage dtype.
    pub dtype: StoredDtype,
    /// Logical shape.
    pub shape: Vec<u64>,
    /// Offset **relative to the section start**, so a section can move without rewriting tensors.
    pub offset: u64,
    /// Byte length; must equal the size implied by shape and dtype.
    pub length: u64,
    /// Name of the tensor holding this one's quantization scales, when quantized.
    pub scales: Option<String>,
}

impl TensorEntry {
    /// Element count, or `None` on overflow.
    #[must_use]
    pub fn elements(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1_u64, |acc, &d| acc.checked_mul(d))
    }
}

/// What went wrong reading an artifact.
///
/// Every variant names the offending section, tensor, or offset: a refusal that does not say what
/// it refused costs an hour of bisecting a multi-gigabyte file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FttsqError {
    /// File is shorter than the fixed header prefix.
    TooShort {
        /// Bytes actually present.
        length: u64,
    },
    /// Magic bytes are not `FTTSQ`.
    BadMagic {
        /// The first eight bytes found.
        found: [u8; 8],
    },
    /// The artifact is newer than this binary understands.
    UnsupportedVersion {
        /// Version recorded in the file.
        found: u32,
        /// Newest version this binary reads.
        supported: u32,
    },
    /// The directory length is implausible or does not fit in the file.
    DirectoryLength {
        /// Declared length.
        declared: u64,
        /// Cap or file length it violated.
        limit: u64,
    },
    /// The directory is not valid UTF-8 JSON, or is not an object.
    DirectoryMalformed {
        /// What the parser said.
        detail: String,
    },
    /// A required field is missing or the wrong type.
    Field {
        /// JSON path of the field.
        path: String,
        /// What was expected.
        expected: String,
    },
    /// An enum carried a value this binary does not know.
    UnknownValue {
        /// JSON path of the field.
        path: String,
        /// The unrecognized value.
        found: String,
    },
    /// A declared count or dimension exceeds its cap.
    LimitExceeded {
        /// What was being counted.
        what: String,
        /// Declared value.
        found: u64,
        /// The cap.
        limit: u64,
    },
    /// A byte range overflowed or ran past the end of its container.
    RangeOutOfBounds {
        /// What the range belongs to.
        what: String,
        /// Start offset.
        offset: u64,
        /// Length.
        length: u64,
        /// The bound it violated.
        bound: u64,
    },
    /// Two sections claim overlapping bytes.
    SectionOverlap {
        /// The earlier section.
        first: String,
        /// The later section.
        second: String,
    },
    /// Two tensors in one section claim overlapping bytes.
    TensorOverlap {
        /// The earlier tensor.
        first: String,
        /// The later tensor.
        second: String,
    },
    /// A name is declared twice.
    DuplicateName {
        /// What kind of thing.
        what: String,
        /// The repeated name.
        name: String,
    },
    /// A tensor references a section that does not exist.
    UnknownSection {
        /// The tensor.
        tensor: String,
        /// The section it named.
        section: String,
    },
    /// A tensor's declared length disagrees with its shape and dtype.
    LengthMismatch {
        /// The tensor.
        tensor: String,
        /// Length the directory declared.
        declared: u64,
        /// Length implied by shape and dtype.
        implied: u64,
    },
    /// A section's bytes do not match its recorded digest.
    DigestMismatch {
        /// The section.
        section: String,
        /// Digest the directory recorded.
        expected: String,
        /// Digest the bytes actually produce.
        actual: String,
    },
    /// The mandatory license notice is absent or empty.
    ///
    /// Apache-2.0 §4 attaches to every artifact we publish; an artifact without the notice must not
    /// be readable, or the obligation becomes advisory in practice.
    LicenseNoticeMissing,
    /// A filesystem operation failed.
    ///
    /// Carries the rendered message rather than [`std::io::Error`] so this enum stays `Clone` and
    /// `PartialEq` — properties the tests and the fuzz target rely on.
    Io {
        /// What was being attempted.
        operation: String,
        /// The path involved.
        path: String,
        /// The OS error text.
        detail: String,
    },
}

impl fmt::Display for FttsqError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { length } => write!(
                f,
                "not a .fttsq artifact: {length} bytes is shorter than the {HEADER_PREFIX_BYTES}-byte header"
            ),
            Self::BadMagic { found } => {
                write!(f, "not a .fttsq artifact: magic {found:?} is not {MAGIC:?}")
            }
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "artifact format version {found} is newer than this binary supports ({supported}); \
                 upgrade ftts rather than reading it with a stale layout"
            ),
            Self::DirectoryLength { declared, limit } => {
                write!(f, "directory length {declared} exceeds {limit}")
            }
            Self::DirectoryMalformed { detail } => write!(f, "directory is malformed: {detail}"),
            Self::Field { path, expected } => {
                write!(f, "directory field `{path}` is missing or not {expected}")
            }
            Self::UnknownValue { path, found } => write!(
                f,
                "directory field `{path}` has unknown value `{found}`; this artifact needs a newer ftts"
            ),
            Self::LimitExceeded { what, found, limit } => {
                write!(f, "{what} count {found} exceeds the cap of {limit}")
            }
            Self::RangeOutOfBounds {
                what,
                offset,
                length,
                bound,
            } => write!(
                f,
                "{what} range [{offset}, {offset}+{length}) runs past its bound {bound}"
            ),
            Self::SectionOverlap { first, second } => write!(
                f,
                "sections `{first}` and `{second}` claim overlapping bytes"
            ),
            Self::TensorOverlap { first, second } => write!(
                f,
                "tensors `{first}` and `{second}` claim overlapping bytes"
            ),
            Self::DuplicateName { what, name } => write!(f, "{what} `{name}` is declared twice"),
            Self::UnknownSection { tensor, section } => write!(
                f,
                "tensor `{tensor}` names section `{section}`, which is not declared"
            ),
            Self::LengthMismatch {
                tensor,
                declared,
                implied,
            } => write!(
                f,
                "tensor `{tensor}` declares {declared} bytes but its shape and dtype imply {implied}"
            ),
            Self::DigestMismatch {
                section,
                expected,
                actual,
            } => write!(
                f,
                "section `{section}` is corrupt: recorded sha256 {expected}, computed {actual}"
            ),
            Self::LicenseNoticeMissing => f.write_str(
                "artifact carries no license_notice; Apache-2.0 §4 requires it on every published \
                 artifact, so an artifact without one is refused rather than silently accepted",
            ),
            Self::Io {
                operation,
                path,
                detail,
            } => write!(f, "{operation} failed for `{path}`: {detail}"),
        }
    }
}

impl std::error::Error for FttsqError {}

/// A parsed, validated `.fttsq` directory over a buffer the reader does not own.
#[derive(Clone, Debug)]
pub struct FttsqReader {
    format_version: u32,
    model_family: String,
    source_sha256: String,
    license_notice: String,
    model_config: Value,
    quantization_manifest: Value,
    sections: Vec<SectionEntry>,
    tensors: Vec<TensorEntry>,
    section_index: BTreeMap<String, usize>,
    tensor_index: BTreeMap<String, usize>,
}

impl FttsqReader {
    /// Parses and fully validates an artifact, **including** every section digest.
    ///
    /// Digest verification is not optional here. A reader that can be asked to skip it grows a
    /// caller that always skips it, and then corruption surfaces as audio rather than as an error.
    ///
    /// # Errors
    ///
    /// Returns a named [`FttsqError`] for any structural, range, or integrity violation.
    pub fn open(bytes: &[u8]) -> Result<Self, FttsqError> {
        let reader = Self::parse_directory(bytes)?;
        reader.verify_digests(bytes)?;
        Ok(reader)
    }

    /// Parses and structurally validates without computing digests.
    ///
    /// For inspection tooling (`ftts inspect`) over an artifact whose bytes are not all present —
    /// listing a remote artifact's tensors, say. Never use this to load weights: it does not prove
    /// the payload is intact.
    ///
    /// # Errors
    ///
    /// Returns a named [`FttsqError`] for any structural or range violation.
    pub fn parse_directory(bytes: &[u8]) -> Result<Self, FttsqError> {
        let file_len = bytes.len() as u64;
        if file_len < HEADER_PREFIX_BYTES {
            return Err(FttsqError::TooShort { length: file_len });
        }

        let mut magic = [0_u8; 8];
        magic.copy_from_slice(&bytes[..8]);
        if &magic != MAGIC {
            return Err(FttsqError::BadMagic { found: magic });
        }

        let format_version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if format_version > FORMAT_VERSION {
            return Err(FttsqError::UnsupportedVersion {
                found: format_version,
                supported: FORMAT_VERSION,
            });
        }

        let mut length_bytes = [0_u8; 8];
        length_bytes.copy_from_slice(&bytes[12..20]);
        let directory_len = u64::from_le_bytes(length_bytes);
        if directory_len > MAX_DIRECTORY_BYTES {
            return Err(FttsqError::DirectoryLength {
                declared: directory_len,
                limit: MAX_DIRECTORY_BYTES,
            });
        }
        let directory_end =
            HEADER_PREFIX_BYTES
                .checked_add(directory_len)
                .ok_or(FttsqError::DirectoryLength {
                    declared: directory_len,
                    limit: u64::MAX,
                })?;
        if directory_end > file_len {
            return Err(FttsqError::DirectoryLength {
                declared: directory_len,
                limit: file_len,
            });
        }

        // `directory_end` is bounded by `file_len` above, so both casts are in range.
        let directory_bytes = &bytes[HEADER_PREFIX_BYTES as usize..directory_end as usize];
        let directory: Value = serde_json::from_slice(directory_bytes).map_err(|error| {
            FttsqError::DirectoryMalformed {
                detail: error.to_string(),
            }
        })?;
        let object = directory
            .as_object()
            .ok_or_else(|| FttsqError::DirectoryMalformed {
                detail: "top level is not a JSON object".to_owned(),
            })?;

        let model_family = required_str(object.get("model_family"), "model_family")?.to_owned();
        let source_sha256 = required_str(object.get("source_sha256"), "source_sha256")?.to_owned();

        // Apache-2.0 §4 compliance is a read-time gate, not a writer-side courtesy.
        let license_notice = object
            .get("license_notice")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if license_notice.trim().is_empty() {
            return Err(FttsqError::LicenseNoticeMissing);
        }

        let model_config = object.get("model_config").cloned().unwrap_or(Value::Null);
        let quantization_manifest = object
            .get("quantization_manifest")
            .cloned()
            .unwrap_or(Value::Null);

        let sections = parse_sections(object.get("sections"), file_len)?;
        let section_index: BTreeMap<String, usize> = sections
            .iter()
            .enumerate()
            .map(|(index, section)| (section.name.clone(), index))
            .collect();
        let tensors = parse_tensors(object.get("tensors"), &sections, &section_index)?;
        let tensor_index: BTreeMap<String, usize> = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| (tensor.name.clone(), index))
            .collect();

        Ok(Self {
            format_version,
            model_family,
            source_sha256,
            license_notice,
            model_config,
            quantization_manifest,
            sections,
            tensors,
            section_index,
            tensor_index,
        })
    }

    /// Recomputes and checks every section digest against `bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`FttsqError::DigestMismatch`] naming the first corrupt section.
    pub fn verify_digests(&self, bytes: &[u8]) -> Result<(), FttsqError> {
        for section in &self.sections {
            let payload = self.section_bytes(section, bytes)?;
            let mut hasher = Sha256::new();
            hasher.update(payload);
            let actual = to_hex(&hasher.finish());
            if actual != section.sha256 {
                return Err(FttsqError::DigestMismatch {
                    section: section.name.clone(),
                    expected: section.sha256.clone(),
                    actual,
                });
            }
        }
        Ok(())
    }

    fn section_bytes<'a>(
        &self,
        section: &SectionEntry,
        bytes: &'a [u8],
    ) -> Result<&'a [u8], FttsqError> {
        let end = section.end().ok_or_else(|| FttsqError::RangeOutOfBounds {
            what: format!("section `{}`", section.name),
            offset: section.offset,
            length: section.length,
            bound: bytes.len() as u64,
        })?;
        if end > bytes.len() as u64 {
            return Err(FttsqError::RangeOutOfBounds {
                what: format!("section `{}`", section.name),
                offset: section.offset,
                length: section.length,
                bound: bytes.len() as u64,
            });
        }
        Ok(&bytes[section.offset as usize..end as usize])
    }

    /// The format version the artifact declares.
    #[must_use]
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// The model family the artifact was built from.
    #[must_use]
    pub fn model_family(&self) -> &str {
        &self.model_family
    }

    /// SHA-256 of the upstream checkpoint this artifact was converted from.
    #[must_use]
    pub fn source_sha256(&self) -> &str {
        &self.source_sha256
    }

    /// The Apache-2.0 §4 attribution notice. Guaranteed non-empty.
    #[must_use]
    pub fn license_notice(&self) -> &str {
        &self.license_notice
    }

    /// The frozen copy of the upstream model config.
    #[must_use]
    pub const fn model_config(&self) -> &Value {
        &self.model_config
    }

    /// The per-tensor quantization policy actually applied, which the license notice references.
    #[must_use]
    pub const fn quantization_manifest(&self) -> &Value {
        &self.quantization_manifest
    }

    /// Every declared section, in declaration order.
    #[must_use]
    pub fn sections(&self) -> &[SectionEntry] {
        &self.sections
    }

    /// Every declared tensor, in declaration order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorEntry] {
        &self.tensors
    }

    /// Looks a section up by name.
    #[must_use]
    pub fn section(&self, name: &str) -> Option<&SectionEntry> {
        self.section_index
            .get(name)
            .and_then(|&index| self.sections.get(index))
    }

    /// Looks a tensor up by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensor_index
            .get(name)
            .and_then(|&index| self.tensors.get(index))
    }

    /// Sections belonging to one access class.
    #[must_use]
    pub fn sections_in_class(&self, class: AccessClass) -> Vec<&SectionEntry> {
        self.sections
            .iter()
            .filter(|section| section.access_class == class)
            .collect()
    }

    /// The byte span of one tensor within `bytes`.
    ///
    /// # Errors
    ///
    /// Returns [`FttsqError::UnknownSection`] when the tensor's section is missing, or
    /// [`FttsqError::RangeOutOfBounds`] when the resolved span leaves the buffer.
    pub fn tensor_bytes<'a>(&self, name: &str, bytes: &'a [u8]) -> Result<&'a [u8], FttsqError> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| FttsqError::UnknownSection {
                tensor: name.to_owned(),
                section: "<unknown tensor>".to_owned(),
            })?;
        let section = self
            .section(&tensor.section)
            .ok_or_else(|| FttsqError::UnknownSection {
                tensor: tensor.name.clone(),
                section: tensor.section.clone(),
            })?;
        let payload = self.section_bytes(section, bytes)?;
        let end = tensor.offset.checked_add(tensor.length).ok_or_else(|| {
            FttsqError::RangeOutOfBounds {
                what: format!("tensor `{}`", tensor.name),
                offset: tensor.offset,
                length: tensor.length,
                bound: payload.len() as u64,
            }
        })?;
        if end > payload.len() as u64 {
            return Err(FttsqError::RangeOutOfBounds {
                what: format!("tensor `{}`", tensor.name),
                offset: tensor.offset,
                length: tensor.length,
                bound: payload.len() as u64,
            });
        }
        Ok(&payload[tensor.offset as usize..end as usize])
    }
}

fn required_str<'a>(value: Option<&'a Value>, path: &str) -> Result<&'a str, FttsqError> {
    value
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| FttsqError::Field {
            path: path.to_owned(),
            expected: "a non-empty string".to_owned(),
        })
}

fn required_u64(value: Option<&Value>, path: &str) -> Result<u64, FttsqError> {
    value
        .and_then(Value::as_u64)
        .ok_or_else(|| FttsqError::Field {
            path: path.to_owned(),
            expected: "a non-negative integer".to_owned(),
        })
}

fn parse_sections(value: Option<&Value>, file_len: u64) -> Result<Vec<SectionEntry>, FttsqError> {
    let array = value
        .and_then(Value::as_array)
        .ok_or_else(|| FttsqError::Field {
            path: "sections".to_owned(),
            expected: "an array".to_owned(),
        })?;
    if array.len() > MAX_SECTIONS {
        return Err(FttsqError::LimitExceeded {
            what: "section".to_owned(),
            found: array.len() as u64,
            limit: MAX_SECTIONS as u64,
        });
    }

    let mut sections = Vec::with_capacity(array.len());
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for (index, entry) in array.iter().enumerate() {
        let path = |field: &str| format!("sections[{index}].{field}");
        let name = required_str(entry.get("name"), &path("name"))?.to_owned();
        if seen.insert(name.clone(), ()).is_some() {
            return Err(FttsqError::DuplicateName {
                what: "section".to_owned(),
                name,
            });
        }
        let class_text = required_str(entry.get("access_class"), &path("access_class"))?;
        let access_class =
            AccessClass::parse(class_text).ok_or_else(|| FttsqError::UnknownValue {
                path: path("access_class"),
                found: class_text.to_owned(),
            })?;
        let offset = required_u64(entry.get("offset"), &path("offset"))?;
        let length = required_u64(entry.get("length"), &path("length"))?;
        let sha256 = required_str(entry.get("sha256"), &path("sha256"))?.to_owned();

        let end = offset
            .checked_add(length)
            .ok_or_else(|| FttsqError::RangeOutOfBounds {
                what: format!("section `{name}`"),
                offset,
                length,
                bound: file_len,
            })?;
        if end > file_len {
            return Err(FttsqError::RangeOutOfBounds {
                what: format!("section `{name}`"),
                offset,
                length,
                bound: file_len,
            });
        }

        sections.push(SectionEntry {
            name,
            access_class,
            offset,
            length,
            sha256,
        });
    }

    // Overlap is checked on a sorted copy so declaration order stays free.
    let mut ordered: Vec<&SectionEntry> = sections.iter().collect();
    ordered.sort_by_key(|section| section.offset);
    for pair in ordered.windows(2) {
        let (first, second) = (pair[0], pair[1]);
        let first_end = first.end().unwrap_or(u64::MAX);
        if first_end > second.offset {
            return Err(FttsqError::SectionOverlap {
                first: first.name.clone(),
                second: second.name.clone(),
            });
        }
    }

    Ok(sections)
}

fn parse_tensors(
    value: Option<&Value>,
    sections: &[SectionEntry],
    section_index: &BTreeMap<String, usize>,
) -> Result<Vec<TensorEntry>, FttsqError> {
    let array = value
        .and_then(Value::as_array)
        .ok_or_else(|| FttsqError::Field {
            path: "tensors".to_owned(),
            expected: "an array".to_owned(),
        })?;
    if array.len() > MAX_TENSORS {
        return Err(FttsqError::LimitExceeded {
            what: "tensor".to_owned(),
            found: array.len() as u64,
            limit: MAX_TENSORS as u64,
        });
    }

    let mut tensors = Vec::with_capacity(array.len());
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for (index, entry) in array.iter().enumerate() {
        let path = |field: &str| format!("tensors[{index}].{field}");
        let name = required_str(entry.get("name"), &path("name"))?.to_owned();
        if seen.insert(name.clone(), ()).is_some() {
            return Err(FttsqError::DuplicateName {
                what: "tensor".to_owned(),
                name,
            });
        }
        let section = required_str(entry.get("section"), &path("section"))?.to_owned();
        let dtype_text = required_str(entry.get("dtype"), &path("dtype"))?;
        let dtype = StoredDtype::parse(dtype_text).ok_or_else(|| FttsqError::UnknownValue {
            path: path("dtype"),
            found: dtype_text.to_owned(),
        })?;

        let shape_array = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| FttsqError::Field {
                path: path("shape"),
                expected: "an array".to_owned(),
            })?;
        if shape_array.len() > MAX_RANK {
            return Err(FttsqError::LimitExceeded {
                what: format!("tensor `{name}` rank"),
                found: shape_array.len() as u64,
                limit: MAX_RANK as u64,
            });
        }
        let mut shape = Vec::with_capacity(shape_array.len());
        for (axis, dim) in shape_array.iter().enumerate() {
            let dim = dim.as_u64().ok_or_else(|| FttsqError::Field {
                path: format!("{}[{axis}]", path("shape")),
                expected: "a non-negative integer".to_owned(),
            })?;
            if dim > MAX_DIM {
                return Err(FttsqError::LimitExceeded {
                    what: format!("tensor `{name}` dimension {axis}"),
                    found: dim,
                    limit: MAX_DIM,
                });
            }
            shape.push(dim);
        }

        let offset = required_u64(entry.get("offset"), &path("offset"))?;
        let length = required_u64(entry.get("length"), &path("length"))?;
        let scales = entry
            .get("scales")
            .and_then(Value::as_str)
            .map(str::to_owned);

        let tensor = TensorEntry {
            name,
            section,
            dtype,
            shape,
            offset,
            length,
            scales,
        };

        // The declared length must equal what shape and dtype imply. Trusting the declared length
        // alone would let a directory hand out a window that does not match the tensor a kernel
        // then indexes by shape.
        let elements = tensor.elements().ok_or_else(|| FttsqError::LimitExceeded {
            what: format!("tensor `{}` element count", tensor.name),
            found: u64::MAX,
            limit: MAX_DIM,
        })?;
        let implied = dtype
            .storage_bytes(elements)
            .ok_or_else(|| FttsqError::LimitExceeded {
                what: format!("tensor `{}` storage size", tensor.name),
                found: u64::MAX,
                limit: MAX_DIM,
            })?;
        if implied != tensor.length {
            return Err(FttsqError::LengthMismatch {
                tensor: tensor.name.clone(),
                declared: tensor.length,
                implied,
            });
        }

        let owner = section_index
            .get(&tensor.section)
            .and_then(|&index| sections.get(index))
            .ok_or_else(|| FttsqError::UnknownSection {
                tensor: tensor.name.clone(),
                section: tensor.section.clone(),
            })?;
        let end = tensor.offset.checked_add(tensor.length).ok_or_else(|| {
            FttsqError::RangeOutOfBounds {
                what: format!("tensor `{}`", tensor.name),
                offset: tensor.offset,
                length: tensor.length,
                bound: owner.length,
            }
        })?;
        if end > owner.length {
            return Err(FttsqError::RangeOutOfBounds {
                what: format!("tensor `{}`", tensor.name),
                offset: tensor.offset,
                length: tensor.length,
                bound: owner.length,
            });
        }

        tensors.push(tensor);
    }

    // Overlap within each section. Two tensors sharing bytes means one of them is wrong, and which
    // one is not knowable at read time — so both are refused.
    let mut by_section: BTreeMap<&str, Vec<&TensorEntry>> = BTreeMap::new();
    for tensor in &tensors {
        by_section
            .entry(tensor.section.as_str())
            .or_default()
            .push(tensor);
    }
    for group in by_section.values_mut() {
        group.sort_by_key(|tensor| tensor.offset);
        for pair in group.windows(2) {
            let (first, second) = (pair[0], pair[1]);
            let first_end = first.offset.saturating_add(first.length);
            if first_end > second.offset {
                return Err(FttsqError::TensorOverlap {
                    first: first.name.clone(),
                    second: second.name.clone(),
                });
            }
        }
    }

    Ok(tensors)
}

/// Builds a `.fttsq` artifact.
///
/// Sections are appended in the order given; the writer computes each digest and lays out absolute
/// offsets, so a caller cannot produce an artifact whose directory disagrees with its payload.
#[derive(Debug, Default)]
pub struct FttsqWriter {
    model_family: String,
    source_sha256: String,
    license_notice: String,
    model_config: Value,
    quantization_manifest: Value,
    sections: Vec<(SectionEntry, Vec<u8>)>,
    tensors: Vec<TensorEntry>,
}

impl FttsqWriter {
    /// Starts an artifact for one model family, converted from a checkpoint with `source_sha256`.
    #[must_use]
    pub fn new(model_family: impl Into<String>, source_sha256: impl Into<String>) -> Self {
        Self {
            model_family: model_family.into(),
            source_sha256: source_sha256.into(),
            license_notice: String::new(),
            model_config: Value::Null,
            quantization_manifest: Value::Null,
            sections: Vec::new(),
            tensors: Vec::new(),
        }
    }

    /// Sets the Apache-2.0 §4 attribution notice. Required — [`FttsqWriter::finish`] refuses without it.
    #[must_use]
    pub fn license_notice(mut self, notice: impl Into<String>) -> Self {
        self.license_notice = notice.into();
        self
    }

    /// Attaches the frozen upstream model config.
    #[must_use]
    pub fn model_config(mut self, config: Value) -> Self {
        self.model_config = config;
        self
    }

    /// Attaches the per-tensor quantization policy the license notice refers to.
    #[must_use]
    pub fn quantization_manifest(mut self, manifest: Value) -> Self {
        self.quantization_manifest = manifest;
        self
    }

    /// Appends a section with its payload. Offset and digest are computed at [`FttsqWriter::finish`].
    #[must_use]
    pub fn section(
        mut self,
        name: impl Into<String>,
        access_class: AccessClass,
        payload: Vec<u8>,
    ) -> Self {
        let entry = SectionEntry {
            name: name.into(),
            access_class,
            offset: 0,
            length: payload.len() as u64,
            sha256: String::new(),
        };
        self.sections.push((entry, payload));
        self
    }

    /// Declares a tensor located inside an already-added section.
    #[must_use]
    pub fn tensor(mut self, tensor: TensorEntry) -> Self {
        self.tensors.push(tensor);
        self
    }

    /// Serializes the artifact.
    ///
    /// The result is re-parsed before being returned, so a writer bug surfaces here rather than as
    /// an unreadable multi-gigabyte file discovered hours later.
    ///
    /// # Errors
    ///
    /// Returns [`FttsqError::LicenseNoticeMissing`] without a notice, or whatever the validating
    /// re-parse rejects.
    pub fn finish(mut self) -> Result<Vec<u8>, FttsqError> {
        if self.license_notice.trim().is_empty() {
            return Err(FttsqError::LicenseNoticeMissing);
        }

        for (entry, payload) in &mut self.sections {
            entry.length = payload.len() as u64;
            entry.sha256 = hex_digest(payload);
        }

        // Two passes: the directory's size depends on the offsets, and the offsets depend on the
        // directory's size. Serialize once with placeholder offsets to learn the exact directory
        // length, then again with the real ones. The placeholder pass uses u64::MAX-width numbers
        // so the second directory can only be the same size or smaller — and we pad to match.
        let probe = self.directory_json(u64::MAX / 2);
        let probe_len = serde_json::to_vec(&probe)
            .map_err(|error| FttsqError::DirectoryMalformed {
                detail: error.to_string(),
            })?
            .len() as u64;

        let payload_start = HEADER_PREFIX_BYTES + probe_len;
        let directory = self.directory_json(payload_start);
        let mut directory_bytes =
            serde_json::to_vec(&directory).map_err(|error| FttsqError::DirectoryMalformed {
                detail: error.to_string(),
            })?;
        // Pad with trailing spaces so the directory occupies exactly `probe_len` bytes and the
        // offsets computed above stay correct. JSON tolerates trailing whitespace.
        while (directory_bytes.len() as u64) < probe_len {
            directory_bytes.push(b' ');
        }

        let mut out = Vec::with_capacity(payload_start as usize);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(directory_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&directory_bytes);
        for (_, payload) in &self.sections {
            out.extend_from_slice(payload);
        }

        // Prove what we just wrote is readable, digests and all.
        FttsqReader::open(&out)?;
        Ok(out)
    }

    /// Serializes and writes the artifact to `path` **atomically**.
    ///
    /// Writes to a temporary file beside the destination, fsyncs it, then renames over the target.
    /// A reader therefore observes either the previous artifact or the complete new one, never a
    /// half-written prefix — which matters because a truncated `.fttsq` is exactly the shape that
    /// digest verification would report as *corruption* rather than as an interrupted write.
    ///
    /// The temporary lives in the destination's own directory so the rename stays within one
    /// filesystem; `rename` across mount points is not atomic and would silently degrade to a copy.
    /// On failure the temporary is removed.
    ///
    /// # Errors
    ///
    /// Returns [`FttsqError::Io`] naming the operation and path, or whatever
    /// [`FttsqWriter::finish`] rejects.
    pub fn write_to_path(self, path: &std::path::Path) -> Result<(), FttsqError> {
        use std::io::Write as _;

        let bytes = self.finish()?;

        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        // Unique per process so two concurrent converters cannot collide on the temporary.
        let file_name = path.file_name().map_or_else(
            || std::ffi::OsString::from("artifact.fttsq"),
            std::ffi::OsStr::to_os_string,
        );
        let mut temp_name = file_name;
        temp_name.push(format!(".tmp.{}", std::process::id()));
        let temp_path = parent.join(temp_name);

        let io =
            |operation: &str, target: &std::path::Path, error: &std::io::Error| FttsqError::Io {
                operation: operation.to_owned(),
                path: target.display().to_string(),
                detail: error.to_string(),
            };

        // Any failure past this point must not leave the temporary behind.
        let result = (|| -> Result<(), FttsqError> {
            let mut file = std::fs::File::create(&temp_path)
                .map_err(|error| io("create", &temp_path, &error))?;
            file.write_all(&bytes)
                .map_err(|error| io("write", &temp_path, &error))?;
            // fsync before rename: without it the rename can land while the data is still in the
            // page cache, so a crash leaves a correctly-named file full of zeros.
            file.sync_all()
                .map_err(|error| io("fsync", &temp_path, &error))?;
            drop(file);
            std::fs::rename(&temp_path, path).map_err(|error| io("rename", path, &error))
        })();

        if result.is_err() {
            let _ = std::fs::remove_file(&temp_path);
        }
        result
    }

    fn directory_json(&self, payload_start: u64) -> Value {
        let mut cursor = payload_start;
        let sections: Vec<Value> = self
            .sections
            .iter()
            .map(|(entry, _)| {
                let offset = cursor;
                cursor += entry.length;
                json!({
                    "name": entry.name,
                    "access_class": entry.access_class.as_str(),
                    "offset": offset,
                    "length": entry.length,
                    "sha256": entry.sha256,
                })
            })
            .collect();

        let tensors: Vec<Value> = self
            .tensors
            .iter()
            .map(|tensor| {
                json!({
                    "name": tensor.name,
                    "section": tensor.section,
                    "dtype": tensor.dtype.as_str(),
                    "shape": tensor.shape,
                    "offset": tensor.offset,
                    "length": tensor.length,
                    "scales": tensor.scales,
                })
            })
            .collect();

        json!({
            "format_version": FORMAT_VERSION,
            "model_family": self.model_family,
            "source_sha256": self.source_sha256,
            "license_notice": self.license_notice,
            "model_config": self.model_config,
            "quantization_manifest": self.quantization_manifest,
            "sections": sections,
            "tensors": tensors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The §3 notice from `docs/LICENSE_AND_ATTRIBUTION.md`, abbreviated for tests.
    const NOTICE: &str = "Copyright 2026 Alibaba Cloud\nApache-2.0\nCHANGES: requantized to .fttsq";

    fn artifact() -> Vec<u8> {
        FttsqWriter::new("qwen3-tts-12hz-0.6b-base", "a".repeat(64))
            .license_notice(NOTICE)
            .model_config(json!({ "hidden_size": 1024 }))
            .quantization_manifest(json!({ "talker": "q8" }))
            .section(
                "microdecoder",
                AccessClass::HotRecurrentMicrodecoder,
                vec![7_u8; 64],
            )
            .section(
                "text_embedding",
                AccessClass::ColdTextEmbedding,
                vec![9_u8; 32],
            )
            .tensor(TensorEntry {
                name: "microdecoder.body".to_owned(),
                section: "microdecoder".to_owned(),
                dtype: StoredDtype::Q8,
                shape: vec![8, 8],
                offset: 0,
                length: 64,
                scales: Some("microdecoder.body.scales".to_owned()),
            })
            .tensor(TensorEntry {
                name: "text_embedding.weight".to_owned(),
                section: "text_embedding".to_owned(),
                dtype: StoredDtype::Bf16,
                shape: vec![4, 4],
                offset: 0,
                length: 32,
                scales: None,
            })
            .finish()
            .expect("the fixture artifact is writable")
    }

    #[test]
    fn round_trips_through_write_and_read() {
        let bytes = artifact();
        let reader =
            FttsqReader::open(&bytes).expect("the artifact we just wrote must be readable");

        assert_eq!(reader.format_version(), FORMAT_VERSION);
        assert_eq!(reader.model_family(), "qwen3-tts-12hz-0.6b-base");
        assert!(reader.license_notice().contains("Alibaba Cloud"));
        assert_eq!(reader.model_config()["hidden_size"], 1024);
        assert_eq!(reader.sections().len(), 2);
        assert_eq!(reader.tensors().len(), 2);

        // Tensor payloads must come back byte-identical, through the section indirection.
        assert_eq!(
            reader
                .tensor_bytes("microdecoder.body", &bytes)
                .expect("tensor resolves"),
            &vec![7_u8; 64][..]
        );
        assert_eq!(
            reader
                .tensor_bytes("text_embedding.weight", &bytes)
                .expect("tensor resolves"),
            &vec![9_u8; 32][..]
        );
    }

    #[test]
    fn bf16_payload_is_byte_identical_across_the_round_trip() {
        // Verbatim BF16 carriage is the property the converter's parity argument rests on: if the
        // container perturbs a single byte, every downstream parity claim is about the wrong bytes.
        let payload: Vec<u8> = (0..=255_u8).cycle().take(4096).collect();
        let bytes = FttsqWriter::new("qwen3-tts-12hz-0.6b-base", "b".repeat(64))
            .license_notice(NOTICE)
            .section("talker", AccessClass::HotRecurrentTalker, payload.clone())
            .tensor(TensorEntry {
                name: "talker.weight".to_owned(),
                section: "talker".to_owned(),
                dtype: StoredDtype::Bf16,
                shape: vec![64, 32],
                offset: 0,
                length: 4096,
                scales: None,
            })
            .finish()
            .expect("writable");
        let reader = FttsqReader::open(&bytes).expect("readable");
        assert_eq!(
            reader
                .tensor_bytes("talker.weight", &bytes)
                .expect("resolves"),
            &payload[..]
        );
    }

    #[test]
    fn access_classes_drive_the_page_in_policy() {
        let bytes = artifact();
        let reader = FttsqReader::open(&bytes).expect("readable");

        let hot = reader.sections_in_class(AccessClass::HotRecurrentMicrodecoder);
        assert_eq!(hot.len(), 1);
        assert!(hot[0].access_class.is_hot());
        assert!(!hot[0].access_class.is_row_granular());

        let cold = reader.sections_in_class(AccessClass::ColdTextEmbedding);
        assert_eq!(cold.len(), 1);
        assert!(
            !cold[0].access_class.is_hot(),
            "the 622 MB embedding must never be advised resident"
        );
        assert!(
            cold[0].access_class.is_row_granular(),
            "the cold embedding is accessed a row at a time, never as a unit"
        );
    }

    #[test]
    fn a_newer_format_version_is_refused_rather_than_guessed_at() {
        let mut bytes = artifact();
        bytes[8..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        let error = FttsqReader::parse_directory(&bytes).expect_err("must refuse");
        assert_eq!(
            error,
            FttsqError::UnsupportedVersion {
                found: FORMAT_VERSION + 1,
                supported: FORMAT_VERSION,
            }
        );
    }

    #[test]
    fn bad_magic_and_truncation_are_named_refusals() {
        assert!(matches!(
            FttsqReader::parse_directory(&[]),
            Err(FttsqError::TooShort { .. })
        ));
        let mut bytes = artifact();
        bytes[0] = b'X';
        assert!(matches!(
            FttsqReader::parse_directory(&bytes),
            Err(FttsqError::BadMagic { .. })
        ));
    }

    #[test]
    fn a_truncated_file_never_yields_a_partial_load() {
        let full = artifact();
        // Cut into the payload: the directory still parses, but a section runs past the end.
        for cut in [full.len() - 1, full.len() - 40, full.len() - 90] {
            let error = FttsqReader::open(&full[..cut]).expect_err("truncation must be refused");
            assert!(
                matches!(
                    error,
                    FttsqError::RangeOutOfBounds { .. } | FttsqError::DirectoryLength { .. }
                ),
                "unexpected error for cut at {cut}: {error}"
            );
        }
    }

    #[test]
    fn a_single_flipped_payload_bit_fails_digest_verification() {
        let mut bytes = artifact();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        let error = FttsqReader::open(&bytes).expect_err("a bit flip must be caught");
        assert!(
            matches!(
                &error,
                FttsqError::DigestMismatch { section, .. } if section == "text_embedding"
            ),
            "expected a digest mismatch for text_embedding, got {error}"
        );
        // Structure alone still parses — which is exactly why the digest gate has to exist.
        assert!(FttsqReader::parse_directory(&bytes).is_ok());
    }

    #[test]
    fn a_hostile_directory_length_cannot_provoke_a_huge_read() {
        let mut bytes = artifact();
        bytes[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        let error = FttsqReader::parse_directory(&bytes).expect_err("must refuse");
        assert!(matches!(error, FttsqError::DirectoryLength { .. }));
    }

    /// Directory-level defects, each of which would otherwise become a bad read at runtime.
    #[test]
    fn structural_violations_are_each_refused_by_name() {
        type StructuralCase = (&'static str, Value, fn(&FttsqError) -> bool);
        let cases: Vec<StructuralCase> = vec![
            (
                "overlapping sections",
                json!([
                    {"name": "a", "access_class": "METADATA", "offset": 100, "length": 50, "sha256": "x"},
                    {"name": "b", "access_class": "METADATA", "offset": 120, "length": 10, "sha256": "x"},
                ]),
                |e| matches!(e, FttsqError::SectionOverlap { .. }),
            ),
            (
                "a section running past the file",
                json!([
                    {"name": "a", "access_class": "METADATA", "offset": 100, "length": u64::MAX, "sha256": "x"},
                ]),
                |e| matches!(e, FttsqError::RangeOutOfBounds { .. }),
            ),
            (
                "a duplicate section name",
                json!([
                    {"name": "a", "access_class": "METADATA", "offset": 100, "length": 10, "sha256": "x"},
                    {"name": "a", "access_class": "METADATA", "offset": 200, "length": 10, "sha256": "x"},
                ]),
                |e| matches!(e, FttsqError::DuplicateName { .. }),
            ),
            (
                "an unknown access class",
                json!([
                    {"name": "a", "access_class": "PROBABLY_HOT", "offset": 100, "length": 10, "sha256": "x"},
                ]),
                |e| matches!(e, FttsqError::UnknownValue { .. }),
            ),
        ];

        for (description, sections, matches_expected) in cases {
            let error = parse_sections(Some(&sections), 4096)
                .expect_err(&format!("`{description}` must be refused"));
            assert!(
                matches_expected(&error),
                "`{description}` produced the wrong error: {error}"
            );
        }
    }

    #[test]
    fn a_tensor_whose_length_disagrees_with_its_shape_is_refused() {
        let sections = vec![SectionEntry {
            name: "s".to_owned(),
            access_class: AccessClass::Metadata,
            offset: 0,
            length: 4096,
            sha256: String::new(),
        }];
        let index: BTreeMap<String, usize> = [("s".to_owned(), 0)].into_iter().collect();

        // 8x8 bf16 is 128 bytes, not 64.
        let tensors = json!([
            {"name": "t", "section": "s", "dtype": "bf16", "shape": [8, 8], "offset": 0, "length": 64},
        ]);
        let error = parse_tensors(Some(&tensors), &sections, &index).expect_err("must refuse");
        assert_eq!(
            error,
            FttsqError::LengthMismatch {
                tensor: "t".to_owned(),
                declared: 64,
                implied: 128,
            }
        );
    }

    #[test]
    fn tensors_may_not_overlap_within_a_section() {
        let sections = vec![SectionEntry {
            name: "s".to_owned(),
            access_class: AccessClass::Metadata,
            offset: 0,
            length: 4096,
            sha256: String::new(),
        }];
        let index: BTreeMap<String, usize> = [("s".to_owned(), 0)].into_iter().collect();
        let tensors = json!([
            {"name": "a", "section": "s", "dtype": "q8", "shape": [64], "offset": 0, "length": 64},
            {"name": "b", "section": "s", "dtype": "q8", "shape": [64], "offset": 32, "length": 64},
        ]);
        let error = parse_tensors(Some(&tensors), &sections, &index).expect_err("must refuse");
        assert!(matches!(error, FttsqError::TensorOverlap { .. }), "{error}");
    }

    #[test]
    fn a_tensor_leaving_its_section_is_refused() {
        let sections = vec![SectionEntry {
            name: "s".to_owned(),
            access_class: AccessClass::Metadata,
            offset: 0,
            length: 64,
            sha256: String::new(),
        }];
        let index: BTreeMap<String, usize> = [("s".to_owned(), 0)].into_iter().collect();
        let tensors = json!([
            {"name": "a", "section": "s", "dtype": "q8", "shape": [64], "offset": 32, "length": 64},
        ]);
        let error = parse_tensors(Some(&tensors), &sections, &index).expect_err("must refuse");
        assert!(
            matches!(error, FttsqError::RangeOutOfBounds { .. }),
            "{error}"
        );
    }

    #[test]
    fn an_artifact_without_a_license_notice_cannot_be_written_or_read() {
        // Writer side.
        let error = FttsqWriter::new("qwen3-tts-12hz-0.6b-base", "c".repeat(64))
            .section("m", AccessClass::Metadata, vec![1, 2, 3])
            .finish()
            .expect_err("Apache-2.0 §4 makes the notice mandatory");
        assert_eq!(error, FttsqError::LicenseNoticeMissing);

        // Reader side: an artifact whose notice was stripped after the fact is still refused.
        let mut bytes = artifact();
        let directory_len = u64::from_le_bytes(bytes[12..20].try_into().expect("header length"));
        let directory_start = HEADER_PREFIX_BYTES as usize;
        let directory_end = directory_start + directory_len as usize;
        let mut directory: Value = serde_json::from_slice(&bytes[directory_start..directory_end])
            .expect("fixture directory");
        directory["license_notice"] = Value::String(String::new());
        let mut replacement = serde_json::to_vec(&directory).expect("serializes directory");
        assert!(
            replacement.len() <= directory_len as usize,
            "removing a notice cannot grow it"
        );
        replacement.resize(directory_len as usize, b' ');
        bytes[directory_start..directory_end].copy_from_slice(&replacement);
        assert_eq!(
            FttsqReader::open(&bytes).expect_err("must refuse a missing notice"),
            FttsqError::LicenseNoticeMissing
        );
    }

    #[test]
    fn write_to_path_lands_a_complete_readable_artifact_and_leaves_no_temporary() {
        let dir = std::env::temp_dir().join(format!("ftts-fttsq-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("model.fttsq");

        FttsqWriter::new("qwen3-tts-12hz-0.6b-base", "d".repeat(64))
            .license_notice(NOTICE)
            .section("m", AccessClass::HotRecurrentMicrodecoder, vec![3_u8; 128])
            .tensor(TensorEntry {
                name: "m.w".to_owned(),
                section: "m".to_owned(),
                dtype: StoredDtype::Q8,
                shape: vec![128],
                offset: 0,
                length: 128,
                scales: None,
            })
            .write_to_path(&path)
            .expect("artifact is writable");

        let bytes = std::fs::read(&path).expect("artifact is readable");
        let reader = FttsqReader::open(&bytes).expect("what landed on disk must verify");
        assert_eq!(
            reader.tensor_bytes("m.w", &bytes).expect("resolves"),
            &vec![3_u8; 128][..]
        );

        // The temporary must not survive a successful write.
        let strays: Vec<_> = std::fs::read_dir(&dir)
            .expect("dir is listable")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.contains(".tmp."))
            .collect();
        assert!(strays.is_empty(), "temporary files left behind: {strays:?}");

        std::fs::remove_file(&path).expect("cleanup");
    }

    #[test]
    fn write_to_path_refuses_before_touching_the_filesystem_when_the_notice_is_missing() {
        let dir = std::env::temp_dir().join(format!("ftts-fttsq-refuse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let path = dir.join("model.fttsq");

        let error = FttsqWriter::new("qwen3-tts-12hz-0.6b-base", "e".repeat(64))
            .section("m", AccessClass::Metadata, vec![1, 2, 3])
            .write_to_path(&path)
            .expect_err("a notice-less artifact must never reach disk");
        assert_eq!(error, FttsqError::LicenseNoticeMissing);
        assert!(
            !path.exists(),
            "a refused artifact must not leave a file behind"
        );
    }

    #[test]
    fn quantized_dtype_sizes_are_exact_including_the_odd_q4_tail() {
        assert_eq!(StoredDtype::Bf16.storage_bytes(10), Some(20));
        assert_eq!(StoredDtype::F32.storage_bytes(10), Some(40));
        assert_eq!(StoredDtype::Q8.storage_bytes(10), Some(10));
        // Two elements per byte, rounding up: an odd count still occupies a whole trailing byte.
        assert_eq!(StoredDtype::Q4.storage_bytes(10), Some(5));
        assert_eq!(StoredDtype::Q4.storage_bytes(11), Some(6));
        // Overflow is reported, never wrapped into a small, plausible-looking size.
        assert_eq!(StoredDtype::F32.storage_bytes(u64::MAX), None);
    }

    #[test]
    fn wire_strings_round_trip_for_every_enum_value() {
        for class in [
            AccessClass::HotRecurrentMicrodecoder,
            AccessClass::HotRecurrentTalker,
            AccessClass::HotCodecDecoder,
            AccessClass::ColdTextEmbedding,
            AccessClass::EnrollmentSpeakerEncoder,
            AccessClass::EnrollmentCodecEncoder,
            AccessClass::Metadata,
        ] {
            assert_eq!(AccessClass::parse(class.as_str()), Some(class));
        }
        for dtype in [
            StoredDtype::Bf16,
            StoredDtype::F32,
            StoredDtype::Q8,
            StoredDtype::Q4,
        ] {
            assert_eq!(StoredDtype::parse(dtype.as_str()), Some(dtype));
        }
        assert_eq!(AccessClass::parse("HOT_SOMETHING"), None);
        assert_eq!(StoredDtype::parse("f16"), None);
    }
}
