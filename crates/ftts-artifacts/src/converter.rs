//! Shared quantization primitives for runtime loading and offline conversion.
//!
//! The offline `.fttsq` converter must not own a second numerical recipe. Both paths call the
//! row primitive in this module, so their Q8 bytes and scales are identical by construction.

use std::fmt;

use crate::safetensors::TensorView;

/// Largest input width accepted by the bounded Q8 matrix-row adapter.
///
/// The pinned tensor inventory's largest trailing shape product is 12,288. This leaves more than
/// five times that headroom while bounding the adapter's `f32` + Q8 scratch space to 320 KiB. A
/// new checkpoint or a malformed external input with a wider row must be given an explicit tiling
/// policy rather than turning one "row" into an unbounded allocation.
pub const MAX_Q8_OUTPUT_CHANNEL_WIDTH: usize = 65_536;

/// Failure while quantizing one output channel.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantizationError {
    /// The caller did not provide one output byte for every input value.
    OutputLength {
        /// Number of source values in the row.
        values: usize,
        /// Number of output slots supplied by the caller.
        output: usize,
    },
    /// A checkpoint value cannot participate in a deterministic finite Q8 recipe.
    NonFiniteValue {
        /// Index within the output channel.
        index: usize,
        /// The rejected value.
        value: f32,
    },
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputLength { values, output } => write!(
                f,
                "Q8 output length {output} does not match input row length {values}"
            ),
            Self::NonFiniteValue { index, value } => {
                write!(
                    f,
                    "Q8 input row has non-finite value {value} at index {index}"
                )
            }
        }
    }
}

impl std::error::Error for QuantizationError {}

/// Destination for one bounded Q8 matrix row.
///
/// The converter owns only a single input row and its Q8 counterpart while invoking this sink.
/// An offline artifact writer can append `values` and record `scale` immediately; a runtime loader
/// can route the exact same bytes directly into its packed-weight allocation. Neither caller needs
/// to materialize a whole widened tensor.
pub trait Q8RowSink {
    /// Failure produced while accepting a quantized row.
    type Error;

    /// Accepts one output channel of a matrix.
    ///
    /// `row` is the outermost matrix index, `scale` is the canonical symmetric Q8 scale, and
    /// `values` has one signed byte for each source element in that row.
    fn write_q8_row(&mut self, row: usize, scale: f32, values: &[i8]) -> Result<(), Self::Error>;
}

/// Failure while streaming a matrix through the canonical Q8 quantizer.
#[derive(Clone, Debug, PartialEq)]
pub enum MatrixQuantizationError<E> {
    /// Q8 weights are defined here only for matrices, never by accidentally flattening vectors.
    ExpectedMatrix {
        /// The source rank presented to the quantizer.
        rank: usize,
    },
    /// An empty trailing dimension cannot represent an output channel for a GEMM weight matrix.
    EmptyOutputChannel {
        /// Source matrix shape.
        shape: Vec<usize>,
    },
    /// One output channel would exceed the adapter's bounded scratch-space contract.
    OutputChannelTooWide {
        /// Number of source values in one output channel.
        width: usize,
        /// Maximum number of values the bounded adapter accepts.
        limit: usize,
    },
    /// A validated view could not provide a complete source row.
    ///
    /// This is defensive: a [`TensorView`] created by [`crate::safetensors::SafetensorsIndex`]
    /// should make it unreachable, but conversion must refuse rather than emit a partial row.
    SourceRowUnavailable {
        /// Source output-channel index.
        row: usize,
    },
    /// A source value cannot be represented by the deterministic Q8 recipe.
    Quantization {
        /// Source output-channel index.
        row: usize,
        /// The underlying numerical refusal.
        source: QuantizationError,
    },
    /// The caller's streaming destination rejected a complete quantized row.
    Sink {
        /// Source output-channel index.
        row: usize,
        /// The destination-specific error.
        source: E,
    },
}

impl<E: fmt::Display> fmt::Display for MatrixQuantizationError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpectedMatrix { rank } => {
                write!(
                    f,
                    "Q8 matrix quantization requires rank 2 or greater, got rank {rank}"
                )
            }
            Self::EmptyOutputChannel { shape } => write!(
                f,
                "Q8 matrix quantization refuses empty output channels for shape {shape:?}"
            ),
            Self::OutputChannelTooWide { width, limit } => write!(
                f,
                "Q8 output-channel width {width} exceeds the bounded adapter limit {limit}"
            ),
            Self::SourceRowUnavailable { row } => {
                write!(f, "Q8 source row {row} is unavailable or incomplete")
            }
            Self::Quantization { row, source } => {
                write!(f, "Q8 source row {row} cannot be quantized: {source}")
            }
            Self::Sink { row, source } => {
                write!(f, "Q8 destination rejected row {row}: {source}")
            }
        }
    }
}

impl<E> std::error::Error for MatrixQuantizationError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Quantization { source, .. } => Some(source),
            Self::Sink { source, .. } => Some(source),
            Self::ExpectedMatrix { .. }
            | Self::EmptyOutputChannel { .. }
            | Self::OutputChannelTooWide { .. }
            | Self::SourceRowUnavailable { .. } => None,
        }
    }
}

/// Quantizes one output channel with the canonical symmetric per-channel Q8 recipe.
///
/// The returned scale is `max(abs(row)) / 127`. All-zero rows use the explicit scale `1.0`,
/// avoiding a NaN-producing divide while preserving zero bytes. Values use ties-to-even rounding
/// after clamping to the symmetric `[-127, 127]` domain; `-128` is never emitted.
///
/// `output` is caller-owned so an offline converter can process a single tile at a time instead
/// of widening or retaining an entire checkpoint tensor. Runtime quantization calls this exact
/// function too.
///
/// # Errors
///
/// Returns an error if the destination length differs from the row length or a source value is
/// NaN or infinite.
pub fn quantize_output_channel_q8(
    row: &[f32],
    output: &mut [i8],
) -> Result<f32, QuantizationError> {
    if output.len() != row.len() {
        return Err(QuantizationError::OutputLength {
            values: row.len(),
            output: output.len(),
        });
    }

    let mut maximum = 0.0_f32;
    for (index, &value) in row.iter().enumerate() {
        if !value.is_finite() {
            return Err(QuantizationError::NonFiniteValue { index, value });
        }
        maximum = maximum.max(value.abs());
    }

    if maximum == 0.0 {
        output.fill(0);
        return Ok(1.0);
    }

    let scale = maximum / 127.0;
    for (&value, slot) in row.iter().zip(output) {
        let rounded = (value / scale).clamp(-127.0, 127.0).round_ties_even();
        // The clamp above proves this conversion is in the i8 range, and the symmetric contract
        // additionally rules out the otherwise-representable -128 value.
        *slot = rounded as i8;
    }
    Ok(scale)
}

/// Quantizes a safetensors matrix one output channel at a time.
///
/// This is the bounded-memory bridge between a zero-copy checkpoint view and a streaming artifact
/// writer. It allocates exactly two row-sized scratch buffers: one widened `f32` row and one Q8
/// row. In particular, it never constructs an `f32` or Q8 copy of the entire matrix. Each row is
/// passed through [`quantize_output_channel_q8`], the primitive runtime quantization also calls,
/// before it reaches `sink`.
///
/// The source must be rank 2 or greater, with its outermost axis representing output channels.
/// Vectors are rejected explicitly so a caller must choose their precision policy rather than
/// silently treating every scalar as an independently scaled output channel.
///
/// # Errors
///
/// Returns a named error for an unsupported shape, malformed source row, non-finite source value,
/// or destination failure. A failure never emits a partial row.
pub fn quantize_matrix_q8_rows<S: Q8RowSink>(
    matrix: &TensorView<'_>,
    sink: &mut S,
) -> Result<(), MatrixQuantizationError<S::Error>> {
    let shape = matrix.shape();
    if shape.len() < 2 {
        return Err(MatrixQuantizationError::ExpectedMatrix { rank: shape.len() });
    }

    let Some(&row_count) = shape.first() else {
        return Err(MatrixQuantizationError::ExpectedMatrix { rank: 0 });
    };
    let row_width = matrix.row_len();
    if row_width == 0 {
        return Err(MatrixQuantizationError::EmptyOutputChannel {
            shape: shape.to_vec(),
        });
    }
    if row_width > MAX_Q8_OUTPUT_CHANNEL_WIDTH {
        return Err(MatrixQuantizationError::OutputChannelTooWide {
            width: row_width,
            limit: MAX_Q8_OUTPUT_CHANNEL_WIDTH,
        });
    }

    let mut source_row = vec![0.0_f32; row_width];
    let mut quantized_row = vec![0_i8; row_width];
    for row in 0..row_count {
        if !matrix.copy_row_f32(row, &mut source_row) {
            return Err(MatrixQuantizationError::SourceRowUnavailable { row });
        }
        let scale = quantize_output_channel_q8(&source_row, &mut quantized_row)
            .map_err(|source| MatrixQuantizationError::Quantization { row, source })?;
        sink.write_q8_row(row, scale, &quantized_row)
            .map_err(|source| MatrixQuantizationError::Sink { row, source })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::SafetensorsIndex;
    use serde_json::json;
    use std::convert::Infallible;

    #[derive(Default)]
    struct RecordingSink {
        rows: Vec<(usize, f32, Vec<i8>)>,
    }

    impl Q8RowSink for RecordingSink {
        type Error = Infallible;

        fn write_q8_row(
            &mut self,
            row: usize,
            scale: f32,
            values: &[i8],
        ) -> Result<(), Self::Error> {
            self.rows.push((row, scale, values.to_vec()));
            Ok(())
        }
    }

    fn f32_matrix(rows: usize, columns: usize, values: &[f32]) -> Vec<u8> {
        assert_eq!(values.len(), rows * columns);
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let header = serde_json::to_vec(&json!({
            "matrix": {
                "dtype": "F32",
                "shape": [rows, columns],
                "data_offsets": [0, payload.len()],
            }
        }))
        .expect("fixture directory serializes");

        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn q8_uses_symmetric_ties_to_even_rounding_and_never_emits_negative_128() {
        let row = [
            -127.0, -126.5, -125.5, -1.5, -0.5, 0.5, 1.5, 125.5, 126.5, 127.0,
        ];
        let mut output = [0_i8; 10];

        let scale = quantize_output_channel_q8(&row, &mut output).expect("finite row");

        assert_eq!(scale, 1.0);
        assert_eq!(output, [-127, -126, -126, -2, 0, 0, 2, 126, 126, 127]);
        assert!(!output.contains(&i8::MIN));
    }

    #[test]
    fn q8_all_zero_row_has_a_finite_unit_scale() {
        let row = [0.0_f32; 4];
        let mut output = [9_i8; 4];

        let scale = quantize_output_channel_q8(&row, &mut output).expect("zero row is valid");

        assert_eq!(scale, 1.0);
        assert_eq!(output, [0; 4]);
    }

    #[test]
    fn q8_refuses_length_mismatch_and_non_finite_input() {
        let error = quantize_output_channel_q8(&[1.0, 2.0], &mut [0]).expect_err("wrong length");
        assert_eq!(
            error,
            QuantizationError::OutputLength {
                values: 2,
                output: 1,
            }
        );

        let error = quantize_output_channel_q8(&[1.0, f32::NAN], &mut [0; 2])
            .expect_err("NaN cannot be quantized deterministically");
        assert!(matches!(
            error,
            QuantizationError::NonFiniteValue { index: 1, value } if value.is_nan()
        ));
    }

    #[test]
    fn runtime_and_offline_callers_receive_byte_identical_q8_rows() {
        let row = [-3.0_f32, -0.75, 0.5, 1.5, 3.0];
        let mut runtime = [0_i8; 5];
        let mut offline = [0_i8; 5];

        let runtime_scale = quantize_output_channel_q8(&row, &mut runtime).expect("runtime Q8");
        let offline_scale = quantize_output_channel_q8(&row, &mut offline).expect("offline Q8");

        assert_eq!(runtime, offline);
        assert_eq!(runtime_scale.to_bits(), offline_scale.to_bits());
    }

    #[test]
    fn matrix_rows_stream_through_the_shared_primitive_in_order() {
        let bytes = f32_matrix(2, 3, &[1.0, -2.0, 0.5, 3.0, 0.0, -3.0]);
        let index = SafetensorsIndex::parse(&bytes).expect("fixture parses");
        let matrix = index.view("matrix", &bytes).expect("matrix view exists");
        let mut sink = RecordingSink::default();

        quantize_matrix_q8_rows(&matrix, &mut sink).expect("finite matrix quantizes");

        assert_eq!(sink.rows.len(), 2);
        assert_eq!(sink.rows[0].0, 0);
        assert_eq!(sink.rows[0].1.to_bits(), (2.0_f32 / 127.0).to_bits());
        assert_eq!(sink.rows[0].2, vec![64, -127, 32]);
        assert_eq!(sink.rows[1].0, 1);
        assert_eq!(sink.rows[1].1.to_bits(), (3.0_f32 / 127.0).to_bits());
        assert_eq!(sink.rows[1].2, vec![127, 0, -127]);
    }

    #[test]
    fn matrix_quantization_refuses_vector_policy_ambiguity() {
        let header = serde_json::to_vec(&json!({
            "vector": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 8],
            }
        }))
        .expect("fixture directory serializes");
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        let index = SafetensorsIndex::parse(&bytes).expect("fixture parses");
        let vector = index.view("vector", &bytes).expect("vector view exists");

        let error = quantize_matrix_q8_rows(&vector, &mut RecordingSink::default())
            .expect_err("vector policy must be explicit");
        assert_eq!(error, MatrixQuantizationError::ExpectedMatrix { rank: 1 });
    }

    #[test]
    fn matrix_quantization_refuses_a_row_that_breaks_its_memory_ceiling() {
        let values = vec![0.0_f32; MAX_Q8_OUTPUT_CHANNEL_WIDTH + 1];
        let bytes = f32_matrix(1, values.len(), &values);
        let index = SafetensorsIndex::parse(&bytes).expect("fixture parses");
        let matrix = index.view("matrix", &bytes).expect("matrix view exists");

        let error = quantize_matrix_q8_rows(&matrix, &mut RecordingSink::default())
            .expect_err("row width must be bounded before scratch allocation");
        assert_eq!(
            error,
            MatrixQuantizationError::OutputChannelTooWide {
                width: MAX_Q8_OUTPUT_CHANNEL_WIDTH + 1,
                limit: MAX_Q8_OUTPUT_CHANNEL_WIDTH,
            }
        );
    }
}
