//! Shared quantization primitives for runtime loading and offline conversion.
//!
//! The offline `.fttsq` converter must not own a second numerical recipe. Both paths call the
//! row primitive in this module, so their Q8 bytes and scales are identical by construction.

use std::fmt;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
