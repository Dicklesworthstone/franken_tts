//! Hand-parsed safetensors reading into a byte-range index.
//!
//! The format is three parts: a little-endian `u64` header length, a JSON directory of that many
//! bytes, then an opaque payload. We parse it ourselves rather than pulling a dependency because
//! the checkpoint is the one input we accept from outside our own toolchain, and every field in it
//! is attacker-reachable if a user points `ftts` at a hostile file. Every offset is bounds-checked
//! against the payload before any read, and every declared shape is cross-checked against the byte
//! span it claims (see [`SafetensorsIndex::parse`]).
//!
//! **BF16 stays resident.** The index is a map of byte ranges over a borrowed buffer; nothing is
//! copied or widened at load. Widening to `f32` happens per element at the accessor
//! ([`TensorView::get_f32`]) or per row ([`TensorView::copy_row_f32`]). Materializing a whole-model
//! `f32` copy would more than double residency — the text embedding alone is ~622 MB in BF16 — for
//! no benefit, since every consumer reads it a row or a tile at a time.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::Value;

/// Largest JSON directory we will parse, as a guard against a hostile length prefix.
///
/// The real checkpoint's directory is a few hundred KiB; 64 MiB is far above any legitimate value
/// while keeping a corrupt `u64` from provoking a huge allocation.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// Element types we accept from a checkpoint.
///
/// Deliberately narrow: the pinned Qwen3-TTS checkpoint is BF16 (talker) and F32 (speech
/// tokenizer). An unknown dtype is a refusal, not a silent skip — a checkpoint carrying types we
/// have never conformed is exactly the "wrong or stale weights" case the census exists to catch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dtype {
    /// bfloat16 — the talker checkpoint's storage type.
    Bf16,
    /// IEEE-754 binary32 — the speech tokenizer's storage type.
    F32,
}

impl Dtype {
    /// Bytes per element.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    /// The safetensors spelling of this dtype.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything the directory declares about one tensor, plus its resolved absolute byte span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorEntry {
    /// Tensor name as it appears in the directory.
    pub name: String,
    /// Storage dtype.
    pub dtype: Dtype,
    /// Logical shape, outermost dimension first.
    pub shape: Vec<usize>,
    /// Absolute start offset into the whole file.
    pub begin: usize,
    /// Absolute end offset (exclusive) into the whole file.
    pub end: usize,
}

impl TensorEntry {
    /// Total element count.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Byte length of the payload span.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.end - self.begin
    }

    /// Elements per row, i.e. the product of every dimension after the first.
    ///
    /// Used by the cold-row text-embedding path, where a "row" is one vocabulary entry.
    #[must_use]
    pub fn row_len(&self) -> usize {
        self.shape.iter().skip(1).product()
    }
}

/// What went wrong reading a checkpoint.
///
/// Every variant names the offending tensor or offset. A checkpoint that fails to parse is a
/// loud, specific refusal — never a partial load that surfaces later as garbage audio.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightsError {
    /// File is too short to contain even the 8-byte header length.
    TooShortForHeader {
        /// Bytes actually available.
        len: usize,
    },
    /// The declared header length is implausible or does not fit in the file.
    HeaderLengthOutOfRange {
        /// Length the file declared.
        declared: u64,
        /// Bytes actually available.
        available: usize,
    },
    /// The header is not valid JSON.
    HeaderNotJson {
        /// Parser message.
        detail: String,
    },
    /// The header parsed but is not a JSON object.
    HeaderNotObject,
    /// A tensor's directory entry is malformed.
    MalformedEntry {
        /// Tensor name.
        name: String,
        /// What was wrong.
        detail: String,
    },
    /// A tensor declares a dtype we do not accept.
    UnsupportedDtype {
        /// Tensor name.
        name: String,
        /// Raw dtype string from the directory.
        raw: String,
    },
    /// A tensor's byte span lies outside the payload.
    SpanOutOfBounds {
        /// Tensor name.
        name: String,
        /// Declared start, relative to the payload.
        begin: usize,
        /// Declared end, relative to the payload.
        end: usize,
        /// Payload length.
        payload_len: usize,
    },
    /// A tensor's declared shape does not match the size of its byte span.
    ShapeSpanMismatch {
        /// Tensor name.
        name: String,
        /// Declared shape.
        shape: Vec<usize>,
        /// Bytes the shape implies.
        expected_bytes: usize,
        /// Bytes the span actually covers.
        actual_bytes: usize,
    },
    /// A shape's element count overflowed `usize`.
    ShapeOverflow {
        /// Tensor name.
        name: String,
        /// Declared shape.
        shape: Vec<usize>,
    },
}

impl fmt::Display for WeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShortForHeader { len } => {
                write!(f, "not a safetensors file: {len} bytes, need at least 8")
            }
            Self::HeaderLengthOutOfRange {
                declared,
                available,
            } => write!(
                f,
                "header length {declared} is out of range (file has {available} bytes, cap is \
                 {MAX_HEADER_BYTES})"
            ),
            Self::HeaderNotJson { detail } => write!(f, "header is not valid JSON: {detail}"),
            Self::HeaderNotObject => f.write_str("header JSON is not an object"),
            Self::MalformedEntry { name, detail } => {
                write!(f, "tensor `{name}`: {detail}")
            }
            Self::UnsupportedDtype { name, raw } => write!(
                f,
                "tensor `{name}`: unsupported dtype `{raw}` (accepted: BF16, F32)"
            ),
            Self::SpanOutOfBounds {
                name,
                begin,
                end,
                payload_len,
            } => write!(
                f,
                "tensor `{name}`: byte span {begin}..{end} escapes the {payload_len}-byte payload"
            ),
            Self::ShapeSpanMismatch {
                name,
                shape,
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "tensor `{name}`: shape {shape:?} implies {expected_bytes} bytes but the span \
                 covers {actual_bytes}"
            ),
            Self::ShapeOverflow { name, shape } => {
                write!(f, "tensor `{name}`: shape {shape:?} overflows usize")
            }
        }
    }
}

impl std::error::Error for WeightsError {}

/// A parsed directory: names to byte ranges, over a buffer we do not own.
///
/// Construct with [`SafetensorsIndex::parse`], then pair it with the same buffer via
/// [`SafetensorsIndex::view`] to read tensors.
#[derive(Clone, Debug)]
pub struct SafetensorsIndex {
    entries: BTreeMap<String, TensorEntry>,
    payload_begin: usize,
}

impl SafetensorsIndex {
    /// Parse the header and resolve every tensor's absolute byte span.
    ///
    /// # Errors
    ///
    /// Returns a [`WeightsError`] naming the specific tensor or offset at fault. Nothing is read
    /// from the payload here; this validates that every later read is in bounds by construction.
    pub fn parse(bytes: &[u8]) -> Result<Self, WeightsError> {
        let Some(len_prefix) = bytes.get(..8) else {
            return Err(WeightsError::TooShortForHeader { len: bytes.len() });
        };
        // The prefix is exactly 8 bytes, so the conversion cannot fail.
        let header_len = u64::from_le_bytes(
            len_prefix
                .try_into()
                .expect("slice of 8 bytes converts to [u8; 8]"),
        );

        if header_len > MAX_HEADER_BYTES {
            return Err(WeightsError::HeaderLengthOutOfRange {
                declared: header_len,
                available: bytes.len(),
            });
        }
        // `header_len` is now <= 64 MiB, so this cast is lossless on every target we build for.
        let header_len_usize =
            usize::try_from(header_len).map_err(|_| WeightsError::HeaderLengthOutOfRange {
                declared: header_len,
                available: bytes.len(),
            })?;
        let payload_begin =
            8usize
                .checked_add(header_len_usize)
                .ok_or(WeightsError::HeaderLengthOutOfRange {
                    declared: header_len,
                    available: bytes.len(),
                })?;
        let Some(header_bytes) = bytes.get(8..payload_begin) else {
            return Err(WeightsError::HeaderLengthOutOfRange {
                declared: header_len,
                available: bytes.len(),
            });
        };

        let parsed: Value =
            serde_json::from_slice(header_bytes).map_err(|error| WeightsError::HeaderNotJson {
                detail: error.to_string(),
            })?;
        let Value::Object(directory) = parsed else {
            return Err(WeightsError::HeaderNotObject);
        };

        let payload_len = bytes.len() - payload_begin;
        let mut entries = BTreeMap::new();
        for (name, value) in directory {
            // `__metadata__` is a free-form string map, not a tensor; skipping it is part of the
            // format, not a leniency.
            if name == "__metadata__" {
                continue;
            }
            let entry = parse_entry(&name, &value, payload_begin, payload_len)?;
            entries.insert(name, entry);
        }

        Ok(Self {
            entries,
            payload_begin,
        })
    }

    /// Absolute offset where the payload begins.
    #[must_use]
    pub const fn payload_begin(&self) -> usize {
        self.payload_begin
    }

    /// Number of tensors in the directory.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the directory is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up one tensor's entry.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<&TensorEntry> {
        self.entries.get(name)
    }

    /// Every entry, in name order.
    pub fn entries(&self) -> impl Iterator<Item = &TensorEntry> {
        self.entries.values()
    }

    /// Tensor names, in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Total payload bytes claimed by all tensors.
    ///
    /// The census reports this so a wrong-checkpoint diagnosis can lead with the size mismatch.
    #[must_use]
    pub fn total_tensor_bytes(&self) -> usize {
        self.entries.values().map(TensorEntry::byte_len).sum()
    }

    /// Borrow one tensor's bytes for reading.
    ///
    /// `bytes` must be the same buffer that was passed to [`SafetensorsIndex::parse`]; passing a
    /// shorter one returns `None` rather than reading out of bounds.
    #[must_use]
    pub fn view<'a>(&self, name: &str, bytes: &'a [u8]) -> Option<TensorView<'a>> {
        let entry = self.entries.get(name)?;
        let raw = bytes.get(entry.begin..entry.end)?;
        Some(TensorView {
            dtype: entry.dtype,
            shape: entry.shape.clone(),
            raw,
        })
    }
}

fn parse_entry(
    name: &str,
    value: &Value,
    payload_begin: usize,
    payload_len: usize,
) -> Result<TensorEntry, WeightsError> {
    let object = value
        .as_object()
        .ok_or_else(|| WeightsError::MalformedEntry {
            name: name.to_owned(),
            detail: "entry is not a JSON object".to_owned(),
        })?;

    let raw_dtype = object.get("dtype").and_then(Value::as_str).ok_or_else(|| {
        WeightsError::MalformedEntry {
            name: name.to_owned(),
            detail: "missing string field `dtype`".to_owned(),
        }
    })?;
    let dtype = Dtype::parse(raw_dtype).ok_or_else(|| WeightsError::UnsupportedDtype {
        name: name.to_owned(),
        raw: raw_dtype.to_owned(),
    })?;

    let raw_shape = object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| WeightsError::MalformedEntry {
            name: name.to_owned(),
            detail: "missing array field `shape`".to_owned(),
        })?;
    let mut shape = Vec::with_capacity(raw_shape.len());
    for dim in raw_shape {
        let dim = dim
            .as_u64()
            .and_then(|d| usize::try_from(d).ok())
            .ok_or_else(|| WeightsError::MalformedEntry {
                name: name.to_owned(),
                detail: "shape contains a non-usize dimension".to_owned(),
            })?;
        shape.push(dim);
    }

    let offsets = object
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| WeightsError::MalformedEntry {
            name: name.to_owned(),
            detail: "missing array field `data_offsets`".to_owned(),
        })?;
    if offsets.len() != 2 {
        return Err(WeightsError::MalformedEntry {
            name: name.to_owned(),
            detail: format!("`data_offsets` has {} entries, expected 2", offsets.len()),
        });
    }
    let mut bound = [0usize; 2];
    for (slot, raw) in bound.iter_mut().zip(offsets) {
        *slot = raw
            .as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| WeightsError::MalformedEntry {
                name: name.to_owned(),
                detail: "`data_offsets` contains a non-usize value".to_owned(),
            })?;
    }
    let [begin, end] = bound;

    // Ordering first, so the subtraction below cannot wrap.
    if begin > end || end > payload_len {
        return Err(WeightsError::SpanOutOfBounds {
            name: name.to_owned(),
            begin,
            end,
            payload_len,
        });
    }

    // A shape whose product overflows would otherwise wrap into a small, plausible-looking count.
    let mut elements = 1usize;
    for dim in &shape {
        elements = elements
            .checked_mul(*dim)
            .ok_or_else(|| WeightsError::ShapeOverflow {
                name: name.to_owned(),
                shape: shape.clone(),
            })?;
    }
    let expected_bytes =
        elements
            .checked_mul(dtype.size())
            .ok_or_else(|| WeightsError::ShapeOverflow {
                name: name.to_owned(),
                shape: shape.clone(),
            })?;
    let actual_bytes = end - begin;
    if expected_bytes != actual_bytes {
        return Err(WeightsError::ShapeSpanMismatch {
            name: name.to_owned(),
            shape,
            expected_bytes,
            actual_bytes,
        });
    }

    Ok(TensorEntry {
        name: name.to_owned(),
        dtype,
        shape,
        begin: payload_begin + begin,
        end: payload_begin + end,
    })
}

/// A borrowed window onto one tensor's bytes, widening to `f32` on read.
///
/// Holds no owned element storage: the BF16 (or F32) bytes stay exactly where they were mapped.
#[derive(Clone, Copy, Debug)]
pub struct TensorViewRef<'a> {
    dtype: Dtype,
    raw: &'a [u8],
}

/// A borrowed window onto one tensor, carrying its shape.
#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    raw: &'a [u8],
}

/// Widen one bfloat16, given as its raw bits, to `f32`.
///
/// BF16 is the top 16 bits of an `f32` with the same exponent layout, so widening is exact for
/// every value including subnormals, infinities, and NaN payloads — a left shift, never a
/// computation. This is the only conversion direction we need: nothing writes BF16.
#[must_use]
pub const fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

impl<'a> TensorView<'a> {
    /// Storage dtype.
    #[must_use]
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Logical shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Element count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.raw.len() / self.dtype.size()
    }

    /// Whether the tensor has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elements per row (product of all dimensions after the first).
    #[must_use]
    pub fn row_len(&self) -> usize {
        self.shape.iter().skip(1).product()
    }

    /// Read one element, widened to `f32`.
    ///
    /// Returns `None` if `index` is past the end. This is the accessor-level widening the design
    /// calls for: no whole-tensor `f32` buffer is ever built.
    #[must_use]
    pub fn get_f32(&self, index: usize) -> Option<f32> {
        let size = self.dtype.size();
        let start = index.checked_mul(size)?;
        let chunk = self.raw.get(start..start.checked_add(size)?)?;
        Some(match self.dtype {
            Dtype::Bf16 => bf16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            Dtype::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
        })
    }

    /// Copy one row, widening to `f32`, into `out`.
    ///
    /// This is the cold-row path for the 151 936 × 2048 text embedding: prefill touches only the
    /// rows its token ids name, so only those rows are ever fetched and widened. Nothing advises
    /// the whole section resident.
    ///
    /// Returns `false` (writing nothing) if the row is out of range or `out` is the wrong length,
    /// so a caller cannot silently consume a partially-filled buffer.
    #[must_use]
    pub fn copy_row_f32(&self, row: usize, out: &mut [f32]) -> bool {
        let row_len = self.row_len();
        if row_len == 0 || out.len() != row_len {
            return false;
        }
        let Some(base) = row.checked_mul(row_len) else {
            return false;
        };
        if base.checked_add(row_len).is_none_or(|end| end > self.len()) {
            return false;
        }
        for (offset, slot) in out.iter_mut().enumerate() {
            // Bounds were proven above, so this cannot miss.
            match self.get_f32(base + offset) {
                Some(value) => *slot = value,
                None => return false,
            }
        }
        true
    }

    /// Borrow the raw, un-widened bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.raw
    }

    /// A shape-less reference to the same bytes.
    #[must_use]
    pub const fn as_ref(&self) -> TensorViewRef<'a> {
        TensorViewRef {
            dtype: self.dtype,
            raw: self.raw,
        }
    }
}

impl TensorViewRef<'_> {
    /// Storage dtype.
    #[must_use]
    pub const fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// Element count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.raw.len() / self.dtype.size()
    }

    /// Whether the tensor has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a well-formed safetensors buffer from `(name, dtype, shape, payload)` parts.
    fn build(parts: &[(&str, Dtype, &[usize], &[u8])]) -> Vec<u8> {
        let mut directory = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in parts {
            let begin = payload.len();
            payload.extend_from_slice(bytes);
            directory.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype.as_str(),
                    "shape": shape,
                    "data_offsets": [begin, payload.len()],
                }),
            );
        }
        assemble(&Value::Object(directory), &payload)
    }

    fn assemble(header: &Value, payload: &[u8]) -> Vec<u8> {
        let header_bytes = serde_json::to_vec(header).expect("header serializes");
        let mut out = Vec::new();
        out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(payload);
        out
    }

    fn bf16_payload(values: &[u16]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn parses_a_two_tensor_directory() {
        let buffer = build(&[
            ("a", Dtype::Bf16, &[2, 2], &bf16_payload(&[0, 1, 2, 3])),
            ("b", Dtype::F32, &[2], &1.0f32.to_le_bytes().repeat(2)),
        ]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");

        assert_eq!(index.len(), 2);
        assert_eq!(index.names().collect::<Vec<_>>(), vec!["a", "b"]);
        let a = index.entry("a").expect("entry a");
        assert_eq!(a.dtype, Dtype::Bf16);
        assert_eq!(a.shape, vec![2, 2]);
        assert_eq!(a.element_count(), 4);
        assert_eq!(a.byte_len(), 8);
        assert_eq!(a.row_len(), 2);
        assert_eq!(index.total_tensor_bytes(), 16);
    }

    #[test]
    fn metadata_key_is_not_a_tensor() {
        let mut directory = serde_json::Map::new();
        directory.insert(
            "__metadata__".to_owned(),
            serde_json::json!({"format": "pt"}),
        );
        directory.insert(
            "w".to_owned(),
            serde_json::json!({"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}),
        );
        let buffer = assemble(&Value::Object(directory), &1.0f32.to_le_bytes());
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        assert_eq!(index.len(), 1);
        assert!(index.entry("__metadata__").is_none());
    }

    #[test]
    fn widening_bf16_is_exact_for_representable_values() {
        // Every BF16 bit pattern is the top half of an f32, so round-tripping an f32 whose low 16
        // mantissa bits are zero must be lossless.
        for bits in [0x0000u16, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x0001, 0x8000] {
            let widened = bf16_bits_to_f32(bits);
            assert_eq!(widened.to_bits() >> 16, u32::from(bits));
            assert_eq!(widened.to_bits() & 0x0000_ffff, 0);
        }
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_bits_to_f32(0xbf80), -1.0);
        assert_eq!(bf16_bits_to_f32(0x0000), 0.0);
        assert!(bf16_bits_to_f32(0x7f80).is_infinite());
        assert!(bf16_bits_to_f32(0x7fc0).is_nan());
    }

    #[test]
    fn bf16_widening_round_trips_every_bit_pattern() {
        // Exhaustive over the whole 16-bit space: widening must never lose the pattern, and must
        // preserve NaN-ness and sign rather than silently canonicalizing.
        for bits in 0..=u16::MAX {
            let widened = bf16_bits_to_f32(bits);
            assert_eq!(
                (widened.to_bits() >> 16) as u16,
                bits,
                "bit pattern {bits:#06x} did not survive widening"
            );
            let exponent = bits & 0x7f80;
            let mantissa = bits & 0x007f;
            if exponent == 0x7f80 && mantissa != 0 {
                assert!(widened.is_nan(), "{bits:#06x} should widen to NaN");
            } else {
                assert!(!widened.is_nan(), "{bits:#06x} should not widen to NaN");
            }
        }
    }

    #[test]
    fn reads_elements_and_rows_without_materializing() {
        let payload = bf16_payload(&[0x3f80, 0xbf80, 0x4000, 0xc000]);
        let buffer = build(&[("w", Dtype::Bf16, &[2, 2], &payload)]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let view = index.view("w", &buffer).expect("view");

        assert_eq!(view.len(), 4);
        assert_eq!(view.row_len(), 2);
        assert_eq!(view.get_f32(0), Some(1.0));
        assert_eq!(view.get_f32(1), Some(-1.0));
        assert_eq!(view.get_f32(3), Some(-2.0));
        assert_eq!(view.get_f32(4), None);

        let mut row = [0.0f32; 2];
        assert!(view.copy_row_f32(1, &mut row));
        assert_eq!(row, [2.0, -2.0]);

        // Out-of-range row and wrong-sized buffer both refuse rather than partially fill.
        assert!(!view.copy_row_f32(2, &mut row));
        let mut wrong = [0.0f32; 3];
        assert!(!view.copy_row_f32(0, &mut wrong));
    }

    #[test]
    fn refuses_a_truncated_file() {
        assert_eq!(
            SafetensorsIndex::parse(&[0u8; 4]),
            Err(WeightsError::TooShortForHeader { len: 4 })
        );
    }

    #[test]
    fn refuses_an_absurd_header_length() {
        let mut buffer = u64::MAX.to_le_bytes().to_vec();
        buffer.extend_from_slice(b"{}");
        let error = SafetensorsIndex::parse(&buffer).expect_err("must refuse");
        assert!(matches!(error, WeightsError::HeaderLengthOutOfRange { .. }));
    }

    #[test]
    fn refuses_a_header_longer_than_the_file() {
        let mut buffer = 4096u64.to_le_bytes().to_vec();
        buffer.extend_from_slice(b"{}");
        let error = SafetensorsIndex::parse(&buffer).expect_err("must refuse");
        assert!(matches!(error, WeightsError::HeaderLengthOutOfRange { .. }));
    }

    #[test]
    fn refuses_malformed_json_and_non_objects() {
        let mut buffer = 5u64.to_le_bytes().to_vec();
        buffer.extend_from_slice(b"{ not");
        assert!(matches!(
            SafetensorsIndex::parse(&buffer).expect_err("must refuse"),
            WeightsError::HeaderNotJson { .. }
        ));

        let buffer = assemble(&serde_json::json!([1, 2]), &[]);
        assert_eq!(
            SafetensorsIndex::parse(&buffer),
            Err(WeightsError::HeaderNotObject)
        );
    }

    #[test]
    fn refuses_an_unsupported_dtype() {
        let buffer = assemble(
            &serde_json::json!({"w": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}),
            &[0u8; 8],
        );
        assert_eq!(
            SafetensorsIndex::parse(&buffer),
            Err(WeightsError::UnsupportedDtype {
                name: "w".to_owned(),
                raw: "I64".to_owned(),
            })
        );
    }

    #[test]
    fn refuses_a_span_past_the_payload() {
        // The directory claims 64 bytes but only 8 follow the header.
        let buffer = assemble(
            &serde_json::json!({"w": {"dtype": "F32", "shape": [16], "data_offsets": [0, 64]}}),
            &[0u8; 8],
        );
        let error = SafetensorsIndex::parse(&buffer).expect_err("must refuse");
        assert!(matches!(error, WeightsError::SpanOutOfBounds { .. }));
    }

    #[test]
    fn refuses_reversed_offsets() {
        let buffer = assemble(
            &serde_json::json!({"w": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}}),
            &[0u8; 8],
        );
        let error = SafetensorsIndex::parse(&buffer).expect_err("must refuse");
        assert!(matches!(error, WeightsError::SpanOutOfBounds { .. }));
    }

    #[test]
    fn refuses_a_shape_that_disagrees_with_its_span() {
        // Shape says 4 F32 elements (16 bytes); the span covers 8.
        let buffer = assemble(
            &serde_json::json!({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}),
            &[0u8; 8],
        );
        let error = SafetensorsIndex::parse(&buffer).expect_err("must refuse");
        match error {
            WeightsError::ShapeSpanMismatch {
                expected_bytes,
                actual_bytes,
                ..
            } => {
                assert_eq!(expected_bytes, 16);
                assert_eq!(actual_bytes, 8);
            }
            other => panic!("wrong error: {other}"),
        }
    }

    #[test]
    fn refuses_a_shape_that_overflows() {
        let huge = usize::MAX;
        let buffer = assemble(
            &serde_json::json!({
                "w": {"dtype": "F32", "shape": [huge, huge], "data_offsets": [0, 8]}
            }),
            &[0u8; 8],
        );
        let error = SafetensorsIndex::parse(&buffer).expect_err("must refuse");
        assert!(matches!(error, WeightsError::ShapeOverflow { .. }));
    }

    #[test]
    fn view_refuses_a_buffer_that_is_not_the_parsed_one() {
        let buffer = build(&[("w", Dtype::F32, &[1], &1.0f32.to_le_bytes())]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        assert!(index.view("w", &buffer[..4]).is_none());
        assert!(index.view("missing", &buffer).is_none());
    }
}
