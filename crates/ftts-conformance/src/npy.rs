//! A minimal NumPy `.npy` reader, for the oracle activation dumps.
//!
//! The fixture packs are trees of `.npy` files — one per captured seam, per decode step. Nothing in
//! this workspace could read them, so Contract A's per-layer comparisons had no way to obtain the
//! expected side. This is that piece.
//!
//! # Deliberately narrow
//!
//! It reads exactly what the CPU-fp32 oracle writes: version 1.0/2.0 headers, C order, little-endian
//! `float32` (`<f4`). Every other dtype, byte order, and Fortran ordering is a **named refusal**
//! rather than a best-effort read. That is not laziness — a reader that silently accepts `>f4` or
//! `fortran_order: True` produces a plausible tensor with transposed axes or byte-swapped values,
//! and the resulting parity failure looks like a kernel bug for a day before anyone suspects the
//! loader. Narrow and loud beats general and quiet here.
//!
//! Bead: `frankentts-p1-talker-z2w` (the parity harness that gates the talker).

use std::{fmt, fs, path::Path};

/// Magic prefix every `.npy` file starts with.
const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// The only dtype the oracle emits, and so the only one accepted.
const ACCEPTED_DESCR: &str = "<f4";

/// Refuse a header longer than this. The real ones are ~118 bytes.
const MAX_HEADER_BYTES: usize = 64 * 1024;

/// What went wrong reading a `.npy` file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NpyError {
    /// The file could not be read.
    Io {
        /// Path involved.
        path: String,
        /// OS error text.
        detail: String,
    },
    /// The file is shorter than a valid header.
    TooShort {
        /// Bytes present.
        length: usize,
    },
    /// The magic prefix is absent.
    BadMagic,
    /// The format version is not one this reader understands.
    UnsupportedVersion {
        /// Major version found.
        major: u8,
        /// Minor version found.
        minor: u8,
    },
    /// The declared header length is implausible or runs past the file.
    HeaderLength {
        /// Declared length.
        declared: usize,
        /// The bound it violated.
        limit: usize,
    },
    /// The header dictionary could not be parsed.
    HeaderMalformed {
        /// What was missing or unreadable.
        detail: String,
    },
    /// The array is not little-endian `float32`.
    UnsupportedDtype {
        /// The `descr` found.
        found: String,
    },
    /// The array is Fortran-ordered, which would need a transpose this reader will not guess at.
    FortranOrder,
    /// The payload length disagrees with the declared shape.
    LengthMismatch {
        /// Elements the shape implies.
        expected_elements: usize,
        /// Elements the payload actually holds.
        found_elements: usize,
    },
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, detail } => write!(f, "cannot read `{path}`: {detail}"),
            Self::TooShort { length } => {
                write!(f, "not a .npy file: {length} bytes is too short for a header")
            }
            Self::BadMagic => f.write_str("not a .npy file: magic prefix is absent"),
            Self::UnsupportedVersion { major, minor } => write!(
                f,
                ".npy format version {major}.{minor} is not supported (expected 1.x or 2.x)"
            ),
            Self::HeaderLength { declared, limit } => {
                write!(f, "header length {declared} exceeds {limit}")
            }
            Self::HeaderMalformed { detail } => write!(f, "malformed .npy header: {detail}"),
            Self::UnsupportedDtype { found } => write!(
                f,
                "dtype `{found}` is not `{ACCEPTED_DESCR}`; this reader accepts only little-endian \
                 float32, because silently reinterpreting another dtype yields a plausible tensor \
                 whose parity failure looks like a kernel bug"
            ),
            Self::FortranOrder => f.write_str(
                "array is fortran_order; refusing rather than guessing a transpose, which would \
                 produce correctly-shaped, wrongly-ordered data",
            ),
            Self::LengthMismatch {
                expected_elements,
                found_elements,
            } => write!(
                f,
                "payload holds {found_elements} elements but the shape implies {expected_elements}"
            ),
        }
    }
}

impl std::error::Error for NpyError {}

/// One decoded array: a C-order `f32` buffer plus its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct NpyArray {
    /// Row-major shape as recorded in the header.
    pub shape: Vec<usize>,
    /// Elements in C order.
    pub data: Vec<f32>,
}

impl NpyArray {
    /// Total element count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Shape rendered for a failure message, e.g. `[1, 28, 1024]`.
    #[must_use]
    pub fn shape_string(&self) -> String {
        format!("{:?}", self.shape)
    }
}

/// Reads a `.npy` file from disk.
///
/// # Errors
///
/// Returns a named [`NpyError`]; see [`parse`] for the structural cases.
pub fn read(path: &Path) -> Result<NpyArray, NpyError> {
    let bytes = fs::read(path).map_err(|error| NpyError::Io {
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    parse(&bytes)
}

/// Parses `.npy` bytes.
///
/// # Errors
///
/// Returns a named [`NpyError`] for any unsupported or malformed input.
pub fn parse(bytes: &[u8]) -> Result<NpyArray, NpyError> {
    // magic(6) + version(2) + at least a u16 header length.
    if bytes.len() < 10 {
        return Err(NpyError::TooShort {
            length: bytes.len(),
        });
    }
    if &bytes[..6] != MAGIC {
        return Err(NpyError::BadMagic);
    }

    let (major, minor) = (bytes[6], bytes[7]);
    // v1 uses a u16 header length, v2 a u32. Both are little-endian and both are otherwise
    // identical for our purposes.
    let (header_len, header_start) = match major {
        1 => (usize::from(u16::from_le_bytes([bytes[8], bytes[9]])), 10),
        2 => {
            if bytes.len() < 12 {
                return Err(NpyError::TooShort {
                    length: bytes.len(),
                });
            }
            let raw = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (raw as usize, 12)
        }
        _ => return Err(NpyError::UnsupportedVersion { major, minor }),
    };

    if header_len > MAX_HEADER_BYTES {
        return Err(NpyError::HeaderLength {
            declared: header_len,
            limit: MAX_HEADER_BYTES,
        });
    }
    let payload_start =
        header_start
            .checked_add(header_len)
            .ok_or_else(|| NpyError::HeaderLength {
                declared: header_len,
                limit: usize::MAX,
            })?;
    if payload_start > bytes.len() {
        return Err(NpyError::HeaderLength {
            declared: header_len,
            limit: bytes.len(),
        });
    }

    let header = std::str::from_utf8(&bytes[header_start..payload_start]).map_err(|error| {
        NpyError::HeaderMalformed {
            detail: format!("header is not UTF-8: {error}"),
        }
    })?;

    let descr = dict_value(header, "descr")?;
    if descr != ACCEPTED_DESCR {
        return Err(NpyError::UnsupportedDtype { found: descr });
    }
    if dict_value(header, "fortran_order")? == "True" {
        return Err(NpyError::FortranOrder);
    }
    let shape = parse_shape(header)?;

    let payload = &bytes[payload_start..];
    let expected_elements: usize = shape.iter().product();
    let found_elements = payload.len() / 4;
    if payload.len() % 4 != 0 || found_elements != expected_elements {
        return Err(NpyError::LengthMismatch {
            expected_elements,
            found_elements,
        });
    }

    let data = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(NpyArray { shape, data })
}

/// Extracts a scalar value for `key` from the Python dict literal header.
///
/// Handles the two forms numpy writes: `'key': 'value'` and `'key': Bareword`.
fn dict_value(header: &str, key: &str) -> Result<String, NpyError> {
    let needle = format!("'{key}':");
    let start = header
        .find(&needle)
        .ok_or_else(|| NpyError::HeaderMalformed {
            detail: format!("no `{key}` field"),
        })?
        + needle.len();
    let rest = header[start..].trim_start();

    if let Some(quoted) = rest.strip_prefix('\'') {
        let end = quoted.find('\'').ok_or_else(|| NpyError::HeaderMalformed {
            detail: format!("unterminated string for `{key}`"),
        })?;
        return Ok(quoted[..end].to_owned());
    }
    let end = rest
        .find([',', '}'])
        .ok_or_else(|| NpyError::HeaderMalformed {
            detail: format!("unterminated value for `{key}`"),
        })?;
    Ok(rest[..end].trim().to_owned())
}

/// Parses the `shape` tuple.
///
/// A 0-d array (`()`) yields an empty shape whose product is 1 — one scalar element, which is what
/// numpy means by it.
fn parse_shape(header: &str) -> Result<Vec<usize>, NpyError> {
    let needle = "'shape':";
    let start = header
        .find(needle)
        .ok_or_else(|| NpyError::HeaderMalformed {
            detail: "no `shape` field".to_owned(),
        })?
        + needle.len();
    let rest = header[start..].trim_start();
    let inner = rest
        .strip_prefix('(')
        .and_then(|body| body.find(')').map(|end| &body[..end]))
        .ok_or_else(|| NpyError::HeaderMalformed {
            detail: "shape is not a parenthesised tuple".to_owned(),
        })?;

    let mut shape = Vec::new();
    for piece in inner.split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        let dim = piece
            .parse::<usize>()
            .map_err(|error| NpyError::HeaderMalformed {
                detail: format!("shape entry `{piece}` is not an integer: {error}"),
            })?;
        shape.push(dim);
    }
    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a valid v1.0 `.npy` in memory, so the happy path is testable without fixtures.
    fn npy_v1(descr: &str, fortran: bool, shape: &[usize], values: &[f32]) -> Vec<u8> {
        let dims = if shape.len() == 1 {
            format!("{},", shape[0])
        } else {
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let header = format!(
            "{{'descr': '{descr}', 'fortran_order': {}, 'shape': ({dims}), }}",
            if fortran { "True" } else { "False" }
        );
        // numpy pads the header so the payload starts 64-byte aligned.
        let mut header = header.into_bytes();
        while (10 + header.len() + 1) % 64 != 0 {
            header.push(b' ');
        }
        header.push(b'\n');

        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&(u16::try_from(header.len()).expect("header fits")).to_le_bytes());
        out.extend_from_slice(&header);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    #[test]
    fn reads_a_c_order_float32_array_with_its_shape() {
        let values = [1.0_f32, -2.5, 3.25, 4.0, 5.0, 6.0];
        let bytes = npy_v1("<f4", false, &[2, 3], &values);
        let array = parse(&bytes).expect("valid npy");
        assert_eq!(array.shape, vec![2, 3]);
        assert_eq!(array.data, values);
        assert_eq!(array.len(), 6);
        assert!(!array.is_empty());
        assert_eq!(array.shape_string(), "[2, 3]");
    }

    #[test]
    fn reads_the_one_dimensional_and_zero_dimensional_forms() {
        let one = parse(&npy_v1("<f4", false, &[3], &[1.0, 2.0, 3.0])).expect("1-d");
        assert_eq!(one.shape, vec![3]);

        // A 0-d array has an empty shape whose product is 1 — exactly one element.
        let zero = parse(&npy_v1("<f4", false, &[], &[7.0])).expect("0-d");
        assert!(zero.shape.is_empty());
        assert_eq!(zero.data, vec![7.0]);
    }

    #[test]
    fn version_two_headers_are_accepted() {
        // Same file, but a u32 header length and version 2.0.
        let v1 = npy_v1("<f4", false, &[2], &[1.0, 2.0]);
        let header_len = u16::from_le_bytes([v1[8], v1[9]]);
        let mut v2 = Vec::new();
        v2.extend_from_slice(MAGIC);
        v2.extend_from_slice(&[2, 0]);
        v2.extend_from_slice(&u32::from(header_len).to_le_bytes());
        v2.extend_from_slice(&v1[10..]);

        let array = parse(&v2).expect("v2 is readable");
        assert_eq!(array.data, vec![1.0, 2.0]);
    }

    /// The refusals that keep a loader bug from masquerading as a kernel bug.
    #[test]
    fn every_unsupported_form_is_a_named_refusal_not_a_silent_reinterpretation() {
        // Big-endian float32 would byte-swap into plausible garbage.
        assert_eq!(
            parse(&npy_v1(">f4", false, &[2], &[1.0, 2.0])),
            Err(NpyError::UnsupportedDtype {
                found: ">f4".to_owned()
            })
        );
        // float64 would halve the element count and shift every value.
        assert!(matches!(
            parse(&npy_v1("<f8", false, &[2], &[1.0, 2.0])),
            Err(NpyError::UnsupportedDtype { .. })
        ));
        // Fortran order is correctly shaped and wrongly ordered — the worst kind of wrong.
        assert_eq!(
            parse(&npy_v1("<f4", true, &[2, 3], &[1.0; 6])),
            Err(NpyError::FortranOrder)
        );
        assert_eq!(parse(b"not an npy file at all"), Err(NpyError::BadMagic));
        assert!(matches!(
            parse(&[]),
            Err(NpyError::TooShort { length: 0 })
        ));
    }

    #[test]
    fn a_payload_that_disagrees_with_the_shape_is_refused() {
        // Header says 6 elements, payload carries 4.
        let mut bytes = npy_v1("<f4", false, &[2, 3], &[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(
            parse(&bytes),
            Err(NpyError::LengthMismatch {
                expected_elements: 6,
                found_elements: 4
            })
        ));
        // A truncated payload mid-element is caught too, rather than dropping a partial float.
        bytes.push(0x00);
        assert!(matches!(parse(&bytes), Err(NpyError::LengthMismatch { .. })));
    }

    #[test]
    fn a_hostile_header_length_cannot_provoke_a_huge_read() {
        let mut bytes = npy_v1("<f4", false, &[1], &[1.0]);
        bytes[8..10].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(matches!(parse(&bytes), Err(NpyError::HeaderLength { .. })));
    }

    #[test]
    fn unsupported_versions_are_named() {
        let mut bytes = npy_v1("<f4", false, &[1], &[1.0]);
        bytes[6] = 9;
        assert_eq!(
            parse(&bytes),
            Err(NpyError::UnsupportedVersion { major: 9, minor: 0 })
        );
    }
}
