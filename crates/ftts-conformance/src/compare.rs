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
    //
    // This chain is deliberately TOTAL — every divergence gets a hint. An earlier version tested
    // `over_tolerance <= len / 64`, whose integer division is 0 for any slice under 64 elements, so
    // small tensors fell through the whole chain and got no diagnosis at all. Integer ratios only
    // here; a usize->f64 cast would trip `clippy::cast_precision_loss`.
    let sparse = comparison.over_tolerance == 1
        || comparison.over_tolerance.saturating_mul(64) <= comparison.len;
    if comparison.non_finite > 0 {
        out.push_str("  hint: non-finite values present — suspect uninitialized memory or a divide-by-zero, not precision\n");
    } else if comparison.cosine > 0.999_999 && comparison.max_rel_diff > 1e-3 {
        out.push_str("  hint: cosine ~1 with large relative error — suspect a SCALE/dequant factor, not wiring\n");
    } else if comparison.cosine < 0.9 {
        out.push_str("  hint: low cosine — suspect WIRING (wrong tensor, transposed layout, off-by-one index), not precision\n");
    } else if sparse {
        out.push_str("  hint: few divergent elements — suspect a lane/tail bug in a SIMD path, not the whole kernel\n");
    } else {
        out.push_str("  hint: widespread small divergence — suspect accumulation order or precision; justify against the tolerance source before widening it\n");
    }
    out
}

/// How an exact sequence diverged, beyond "some element differs".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    /// Everything after the divergence matches once `actual` is shifted forward by one: `actual`
    /// is missing exactly one element.
    DroppedFromActual,
    /// Everything after the divergence matches once `expected` is shifted forward by one: `actual`
    /// gained exactly one element.
    InsertedIntoActual,
}

/// Result of comparing two sequences for exact equality.
///
/// Separate from [`Comparison`] because token streams have no tolerance and no cosine: the
/// meaningful diagnostics are *where* the streams parted, whether one is a prefix of the other,
/// and whether the tail realigns under a one-element shift.
#[derive(Clone, Debug)]
pub struct ExactComparison {
    /// Length of the expected sequence.
    pub expected_len: usize,
    /// Length of the actual sequence.
    pub actual_len: usize,
    /// Elements actually compared (the common prefix length).
    pub compared: usize,
    /// Elements differing within the common prefix.
    pub mismatches: usize,
    /// Index of the first differing element, if any.
    pub first_divergence: Option<usize>,
    /// Debug rendering of the expected value at the first divergence.
    pub first_expected: Option<String>,
    /// Debug rendering of the actual value at the first divergence.
    pub first_actual: Option<String>,
    /// Index the context windows start at.
    pub context_start: usize,
    /// Expected values surrounding the first divergence.
    pub expected_context: Vec<String>,
    /// Actual values surrounding the first divergence.
    pub actual_context: Vec<String>,
    /// Whether the tail realigns under a single-element shift.
    pub shift: Option<Shift>,
}

impl ExactComparison {
    /// Whether the sequences are identical in length and content.
    #[must_use]
    pub const fn holds(&self) -> bool {
        self.mismatches == 0 && self.expected_len == self.actual_len
    }

    /// Machine-parseable summary for attaching to a [`crate::report::Receipt`].
    #[must_use]
    pub fn to_json(&self) -> Value {
        json!({
            "expected_len": self.expected_len,
            "actual_len": self.actual_len,
            "compared": self.compared,
            "mismatches": self.mismatches,
            "first_divergence": self.first_divergence,
            "first_expected": self.first_expected,
            "first_actual": self.first_actual,
            "context_start": self.context_start,
            "expected_context": self.expected_context,
            "actual_context": self.actual_context,
            "shift": self.shift.map(|s| match s {
                Shift::DroppedFromActual => "dropped_from_actual",
                Shift::InsertedIntoActual => "inserted_into_actual",
            }),
        })
    }
}

/// Elements of context printed either side of the first divergence.
const CONTEXT_RADIUS: usize = 4;

/// Compares two sequences for exact equality, localizing the first divergence.
///
/// Unlike [`compare_f32`], a length mismatch is *reported* rather than panicked on: a truncated
/// token stream is one of the outcomes this comparator exists to diagnose (an early stop condition),
/// not a caller bug.
#[must_use]
pub fn compare_exact<T>(expected: &[T], actual: &[T]) -> ExactComparison
where
    T: PartialEq + std::fmt::Debug,
{
    let compared = expected.len().min(actual.len());
    let mut mismatches = 0_usize;
    let mut first: Option<usize> = None;
    for index in 0..compared {
        if expected[index] != actual[index] {
            mismatches += 1;
            if first.is_none() {
                first = Some(index);
            }
        }
    }

    // A length difference with a clean common prefix still needs a divergence point to report.
    let divergence = first.or(if expected.len() == actual.len() {
        None
    } else {
        Some(compared)
    });

    let shift = divergence.and_then(|at| {
        if expected.len() == actual.len() + 1 && expected[at + 1..] == actual[at..] {
            Some(Shift::DroppedFromActual)
        } else if actual.len() == expected.len() + 1 && expected[at..] == actual[at + 1..] {
            Some(Shift::InsertedIntoActual)
        } else {
            None
        }
    });

    let context_start = divergence.map_or(0, |at| at.saturating_sub(CONTEXT_RADIUS));
    let context_end = divergence.map_or(0, |at| at + CONTEXT_RADIUS + 1);
    let render = |slice: &[T]| -> Vec<String> {
        slice
            .iter()
            .skip(context_start)
            .take(context_end.saturating_sub(context_start))
            .map(|value| format!("{value:?}"))
            .collect()
    };

    ExactComparison {
        expected_len: expected.len(),
        actual_len: actual.len(),
        compared,
        mismatches,
        first_divergence: divergence,
        first_expected: divergence
            .and_then(|at| expected.get(at))
            .map(|value| format!("{value:?}")),
        first_actual: divergence
            .and_then(|at| actual.get(at))
            .map(|value| format!("{value:?}")),
        context_start,
        expected_context: render(expected),
        actual_context: render(actual),
        shift,
    }
}

/// Renders a self-localizing failure report for an exact-sequence divergence.
#[must_use]
pub fn describe_exact_failure(seam: &str, comparison: &ExactComparison) -> String {
    let mut out = format!(
        "seam `{seam}` diverged (exact): {} mismatches in the {}-element common prefix; \
         expected_len {} vs actual_len {}\n",
        comparison.mismatches, comparison.compared, comparison.expected_len, comparison.actual_len
    );
    if let Some(at) = comparison.first_divergence {
        out.push_str(&format!("  first divergence at index {at}\n"));
        out.push_str(&format!(
            "    expected {}\n    actual   {}\n",
            comparison.first_expected.as_deref().unwrap_or("<past end>"),
            comparison.first_actual.as_deref().unwrap_or("<past end>"),
        ));
        out.push_str(&format!(
            "  context from index {}\n    expected {:?}\n    actual   {:?}\n",
            comparison.context_start, comparison.expected_context, comparison.actual_context,
        ));
    }
    // As in `describe_failure`, the diagnosis chain is total: every divergence gets a hint.
    let clean_prefix = comparison.mismatches == 0;
    match comparison.shift {
        Some(Shift::DroppedFromActual) => out.push_str(
            "  hint: the tail realigns if actual is shifted by one — actual DROPPED an element; \
             suspect a skipped step or an off-by-one in prompt assembly, not a wrong value\n",
        ),
        Some(Shift::InsertedIntoActual) => out.push_str(
            "  hint: the tail realigns if expected is shifted by one — actual INSERTED an element; \
             suspect a duplicated step or an extra special token, not a wrong value\n",
        ),
        None if clean_prefix && comparison.actual_len < comparison.expected_len => out.push_str(
            "  hint: actual is a strict PREFIX of expected — suspect an early stop condition or a \
             truncated decode, not a value bug\n",
        ),
        None if clean_prefix => out.push_str(
            "  hint: expected is a strict PREFIX of actual — suspect a missed stop condition \
             (decode ran long), not a value bug\n",
        ),
        None if comparison.mismatches == 1 => out.push_str(
            "  hint: exactly one element differs — suspect a sampling tie-break or a single bad \
             table entry, not the surrounding pipeline\n",
        ),
        None => out.push_str(
            "  hint: divergence persists from the first mismatch onward — suspect WIRING or state \
             carried into this seam; fix the first index before reading any later one\n",
        ),
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

    /// Every divergence must get a diagnosis, at every size.
    ///
    /// Regression guard: the first version of the hint chain gated the sparse case on
    /// `over_tolerance <= len / 64`, which is `<= 0` for any slice shorter than 64 elements. Small
    /// tensors fell through every branch and the report carried no hint at all.
    #[test]
    fn every_failing_comparison_gets_a_hint_at_every_size() {
        for len in [1_usize, 2, 6, 63, 64, 65, 200] {
            for divergent in [1_usize, len / 2 + 1, len] {
                let divergent = divergent.min(len).max(1);
                let expected = vec![1.0_f32; len];
                let mut actual = expected.clone();
                for slot in actual.iter_mut().take(divergent) {
                    *slot = 5.0;
                }
                let comparison = compare_f32(&expected, &actual, 1e-6);
                assert!(
                    !comparison.holds(),
                    "len={len} divergent={divergent} should fail"
                );
                let report = describe_failure("seam", &comparison, 1e-6, "test", None);
                assert!(
                    report.contains("hint:"),
                    "no diagnosis for len={len} divergent={divergent}:\n{report}"
                );
            }
        }
        // A NaN must be diagnosed too, at the smallest possible size.
        let nan_report = describe_failure(
            "seam",
            &compare_f32(&[1.0], &[f32::NAN], 1e9),
            1e9,
            "test",
            None,
        );
        assert!(nan_report.contains("hint:"), "{nan_report}");
        assert!(nan_report.contains("non-finite"), "{nan_report}");
    }
}
