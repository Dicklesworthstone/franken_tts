//! Numeric comparators that localize their own failures.
//!
//! A bare `assert_eq!` on two 4-million-element tensors tells you nothing you can act on. Every
//! comparison here reports the **first** divergent element with its multi-dimensional coordinates,
//! plus summary statistics (max-abs, max-rel, cosine, count-over-tolerance) that distinguish "one
//! bad lane" from "everything is wrong" — the difference between a SIMD bug and a wiring bug — and
//! it names the tolerance and where that tolerance came from.

use serde_json::{Value, json};

/// Result of comparing two numeric slices.
#[derive(Clone, Debug)]
pub struct Comparison {
    /// Element count compared.
    pub len: usize,
    /// Number of elements exceeding the tolerance.
    pub over_tolerance: usize,
    /// Largest absolute difference observed.
    pub max_abs_diff: f64,
    /// Largest relative difference observed, scaled by `max(|expected|, |actual|)`.
    pub max_rel_diff: f64,
    /// Cosine similarity; near 1.0 with a large max-abs suggests a scale error, not a wiring error.
    pub cosine: f64,
    /// Flat index of the first element over tolerance.
    pub first_divergence: Option<usize>,
    /// Expected value at the first divergence.
    pub first_expected: Option<f64>,
    /// Actual value at the first divergence.
    pub first_actual: Option<f64>,
    /// Non-finite values found (NaN or infinity) in `actual`.
    pub non_finite: usize,
}

impl Comparison {
    /// Whether every element was within tolerance and finite.
    #[must_use]
    pub const fn holds(&self) -> bool {
        self.over_tolerance == 0 && self.non_finite == 0
    }

    /// Machine-parseable summary for attaching to a [`crate::report::Receipt`].
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "len": self.len,
            "over_tolerance": self.over_tolerance,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "cosine": self.cosine,
            "first_divergence": self.first_divergence,
            "first_expected": self.first_expected,
            "first_actual": self.first_actual,
            "non_finite": self.non_finite,
        })
    }
}

/// Compares two `f32` slices elementwise.
///
/// # Panics
///
/// Panics when the two slices have different lengths — a shape mismatch is a wiring bug that no
/// tolerance can express, so it must not be reportable as a near-miss.
#[must_use]
pub fn compare_f32(expected: &[f32], actual: &[f32], tolerance: f64) -> Comparison {
    assert_eq!(
        expected.len(),
        actual.len(),
        "shape mismatch: expected {} elements, got {} — this is a wiring bug, not a tolerance question",
        expected.len(),
        actual.len()
    );

    let (mut max_abs, mut max_rel, mut over, mut non_finite) = (0.0_f64, 0.0_f64, 0_usize, 0_usize);
    let (mut dot, mut norm_e, mut norm_a) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut first: Option<(usize, f64, f64)> = None;

    for (index, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
        let (e, a) = (f64::from(e), f64::from(a));
        if !a.is_finite() {
            non_finite += 1;
        }
        dot += e * a;
        norm_e += e * e;
        norm_a += a * a;

        let abs = (e - a).abs();
        let scale = e.abs().max(a.abs());
        let rel = if scale > 0.0 { abs / scale } else { 0.0 };
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > tolerance {
            over += 1;
            if first.is_none() {
                first = Some((index, e, a));
            }
        }
    }

    let denominator = (norm_e.sqrt()) * (norm_a.sqrt());
    let cosine = if denominator > 0.0 {
        dot / denominator
    } else {
        1.0
    };

    Comparison {
        len: expected.len(),
        over_tolerance: over,
        max_abs_diff: max_abs,
        max_rel_diff: max_rel,
        cosine,
        first_divergence: first.map(|(i, _, _)| i),
        first_expected: first.map(|(_, e, _)| e),
        first_actual: first.map(|(_, _, a)| a),
        non_finite,
    }
}

/// Converts a flat index into multi-dimensional coordinates for a row-major `shape`.
///
/// Returns `None` when `shape` does not describe exactly `index`-addressable storage, so a wrong
/// shape annotation cannot silently produce plausible-looking coordinates.
#[must_use]
pub fn coordinates(index: usize, shape: &[usize]) -> Option<Vec<usize>> {
    let total: usize = shape.iter().product();
    if shape.is_empty() || index >= total {
        return None;
    }
    let mut remainder = index;
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = remainder % shape[axis];
        remainder /= shape[axis];
    }
    Some(coords)
}

/// Renders a human-readable, self-localizing failure report.
///
/// Deliberately verbose: this text is what a person or agent reads at 3am, and it must answer
/// "where, how badly, against what, and by whose tolerance" without a rerun.
#[must_use]
pub fn describe_failure(
    seam: &str,
    comparison: &Comparison,
    tolerance: f64,
    tolerance_source: &str,
    shape: Option<&[usize]>,
) -> String {
    let mut out = format!(
        "seam `{seam}` diverged: {}/{} elements over tolerance {tolerance:.3e} (source: {tolerance_source})\n",
        comparison.over_tolerance, comparison.len
    );
    if let (Some(index), Some(e), Some(a)) = (
        comparison.first_divergence,
        comparison.first_expected,
        comparison.first_actual,
    ) {
        let location = shape.and_then(|s| coordinates(index, s)).map_or_else(
            || format!("flat[{index}]"),
            |c| format!("flat[{index}] = index{c:?}"),
        );
        let diff = (e - a).abs();
        out.push_str(&format!(
            "  first divergence at {location}\n    expected {e:+.9e}\n    actual   {a:+.9e}\n    absdiff  {diff:.3e}\n"
        ));
    }
    if let Some(s) = shape {
        out.push_str(&format!("  shape {s:?} (row-major)\n"));
    }
    out.push_str(&format!(
        "  max_abs {:.3e} | max_rel {:.3e} | cosine {:.12} | non_finite {}\n",
        comparison.max_abs_diff, comparison.max_rel_diff, comparison.cosine, comparison.non_finite
    ));
    // The shape of the error is the diagnosis: read these before touching a tolerance.
    if comparison.non_finite > 0 {
        out.push_str("  hint: non-finite values present — suspect uninitialized memory or a divide-by-zero, not precision\n");
    } else if comparison.cosine > 0.999_999 && comparison.max_rel_diff > 1e-3 {
        out.push_str("  hint: cosine ~1 with large relative error — suspect a SCALE/dequant factor, not wiring\n");
    } else if comparison.over_tolerance <= comparison.len / 64 {
        out.push_str("  hint: few divergent elements — suspect a lane/tail bug in a SIMD path, not the whole kernel\n");
    } else if comparison.cosine < 0.9 {
        out.push_str("  hint: low cosine — suspect WIRING (wrong tensor, transposed layout, off-by-one index), not precision\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_slices_hold() {
        let values = [1.0_f32, -2.0, 3.5];
        let comparison = compare_f32(&values, &values, 0.0);
        assert!(comparison.holds());
        assert_eq!(comparison.over_tolerance, 0);
        assert!(comparison.first_divergence.is_none());
        assert!((comparison.cosine - 1.0).abs() < 1e-12);
    }

    #[test]
    fn first_divergence_is_the_first_one_not_the_worst_one() {
        let expected = [0.0_f32, 1.0, 2.0, 3.0];
        // index 1 diverges slightly, index 3 diverges more; the report must name index 1.
        let actual = [0.0_f32, 1.5, 2.0, 9.0];
        let comparison = compare_f32(&expected, &actual, 0.1);
        assert_eq!(comparison.first_divergence, Some(1));
        assert_eq!(comparison.first_expected, Some(1.0));
        assert_eq!(comparison.first_actual, Some(1.5));
        assert_eq!(comparison.over_tolerance, 2);
        assert!((comparison.max_abs_diff - 6.0).abs() < 1e-12);
    }

    #[test]
    fn non_finite_values_are_counted_and_never_pass() {
        let comparison = compare_f32(&[1.0, 2.0], &[1.0, f32::NAN], 1e9);
        assert_eq!(comparison.non_finite, 1);
        assert!(
            !comparison.holds(),
            "a NaN must never be inside tolerance, however wide the tolerance"
        );
    }

    #[test]
    fn coordinates_are_row_major_and_reject_bad_shapes() {
        assert_eq!(coordinates(0, &[2, 3]), Some(vec![0, 0]));
        assert_eq!(coordinates(4, &[2, 3]), Some(vec![1, 1]));
        assert_eq!(coordinates(5, &[2, 3]), Some(vec![1, 2]));
        assert_eq!(
            coordinates(6, &[2, 3]),
            None,
            "out of range must not fabricate coordinates"
        );
        assert_eq!(coordinates(0, &[]), None);
    }

    #[test]
    fn failure_report_localizes_without_a_debugger() {
        let expected = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        let actual = [0.0_f32, 1.0, 2.0, 3.0, 4.25, 5.0];
        let comparison = compare_f32(&expected, &actual, 1e-3);
        assert!(!comparison.holds());

        let report = describe_failure(
            "talker.layer17.attn_out",
            &comparison,
            1e-3,
            "docs/truth-pack/nondeterminism-floor.json",
            Some(&[2, 3]),
        );

        // Everything needed to act, present in one string.
        assert!(
            report.contains("talker.layer17.attn_out"),
            "names the seam:\n{report}"
        );
        assert!(
            report.contains("flat[4]"),
            "gives the flat index:\n{report}"
        );
        assert!(
            report.contains("index[1, 1]"),
            "gives coordinates:\n{report}"
        );
        assert!(report.contains("expected"), "gives expected:\n{report}");
        assert!(report.contains("actual"), "gives actual:\n{report}");
        assert!(report.contains("max_abs"), "gives summary stats:\n{report}");
        assert!(report.contains("cosine"), "gives cosine:\n{report}");
        assert!(
            report.contains("shape [2, 3]"),
            "gives the shape:\n{report}"
        );
        assert!(
            report.contains("nondeterminism-floor.json"),
            "names where the tolerance came from:\n{report}"
        );
        assert!(report.contains("hint:"), "offers a diagnosis:\n{report}");
    }

    #[test]
    fn scale_errors_and_wiring_errors_get_different_hints() {
        // Uniform 2x scaling: cosine stays ~1, relative error is large.
        let expected = [1.0_f32, 2.0, 3.0, 4.0];
        let scaled = [2.0_f32, 4.0, 6.0, 8.0];
        let scale_report = describe_failure(
            "codec.dequant",
            &compare_f32(&expected, &scaled, 1e-6),
            1e-6,
            "test",
            None,
        );
        assert!(scale_report.contains("SCALE"), "{scale_report}");

        // Reversed order: cosine collapses.
        let reversed = [4.0_f32, 3.0, 2.0, 1.0];
        let wiring_report = describe_failure(
            "codec.layout",
            &compare_f32(&expected, &reversed, 1e-6),
            1e-6,
            "test",
            None,
        );
        assert!(wiring_report.contains("WIRING"), "{wiring_report}");
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn length_mismatch_is_a_wiring_bug_not_a_tolerance_question() {
        let _ = compare_f32(&[1.0, 2.0], &[1.0], 1.0);
    }
}
