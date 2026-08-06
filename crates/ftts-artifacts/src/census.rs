//! The load-time weights census: prove the checkpoint is the one we conformed against.
//!
//! A wrong or stale checkpoint that loads "successfully" does not fail loudly — it produces
//! plausible-looking garbage audio hours later, and the bug gets blamed on a kernel. So every load
//! is checked against an expected `(name, shape, dtype)` inventory and any divergence is a
//! **named, itemized refusal** at load time.
//!
//! Three finding classes, all fatal by default:
//!
//! - [`Finding::Missing`] — we need a tensor the file does not have. Certain failure downstream.
//! - [`Finding::ShapeMismatch`] / [`Finding::DtypeMismatch`] — the tensor exists but is not what we
//!   compiled kernels for. This is the class that silently produces garbage rather than crashing.
//! - [`Finding::Extra`] — the file has tensors we do not know about. Not dangerous on its own, but
//!   it is the signature of a *different checkpoint* (a sibling size, a newer revision), which is
//!   exactly what we want to catch before anyone spends a day debugging the audio.
//!
//! The expected inventory itself comes from the OQ-2 tensor census over the pinned weights; this
//! module is the mechanism, not the data.

use std::collections::BTreeMap;
use std::fmt;

use crate::safetensors::{Dtype, SafetensorsIndex};

/// One tensor the manifest requires, as the OQ-2 inventory recorded it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpectedTensor {
    /// Tensor name, exactly as it appears in the checkpoint directory.
    pub name: String,
    /// Required shape.
    pub shape: Vec<usize>,
    /// Required storage dtype.
    pub dtype: Dtype,
}

impl ExpectedTensor {
    /// Build an expectation.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: impl Into<Vec<usize>>, dtype: Dtype) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            dtype,
        }
    }
}

/// How a checkpoint diverged from the manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Finding {
    /// The manifest requires a tensor the checkpoint does not contain.
    Missing {
        /// Tensor name.
        name: String,
        /// Shape we required.
        expected_shape: Vec<usize>,
        /// Dtype we required.
        expected_dtype: Dtype,
    },
    /// The tensor exists with the wrong shape.
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Shape we required.
        expected: Vec<usize>,
        /// Shape the checkpoint declares.
        actual: Vec<usize>,
    },
    /// The tensor exists with the wrong storage dtype.
    DtypeMismatch {
        /// Tensor name.
        name: String,
        /// Dtype we required.
        expected: Dtype,
        /// Dtype the checkpoint declares.
        actual: Dtype,
    },
    /// The checkpoint contains a tensor the manifest does not mention.
    Extra {
        /// Tensor name.
        name: String,
        /// Shape the checkpoint declares.
        shape: Vec<usize>,
        /// Dtype the checkpoint declares.
        dtype: Dtype,
    },
}

impl Finding {
    /// The tensor this finding is about.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Missing { name, .. }
            | Self::ShapeMismatch { name, .. }
            | Self::DtypeMismatch { name, .. }
            | Self::Extra { name, .. } => name,
        }
    }

    /// Short, greppable class label.
    #[must_use]
    pub const fn class(&self) -> &'static str {
        match self {
            Self::Missing { .. } => "MISSING",
            Self::ShapeMismatch { .. } => "SHAPE-MISMATCH",
            Self::DtypeMismatch { .. } => "DTYPE-MISMATCH",
            Self::Extra { .. } => "EXTRA",
        }
    }
}

impl fmt::Display for Finding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing {
                name,
                expected_shape,
                expected_dtype,
            } => write!(
                f,
                "MISSING        {name}: manifest requires {expected_shape:?} {expected_dtype}, \
                 checkpoint has no such tensor"
            ),
            Self::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "SHAPE-MISMATCH {name}: manifest requires {expected:?}, checkpoint has {actual:?}"
            ),
            Self::DtypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "DTYPE-MISMATCH {name}: manifest requires {expected}, checkpoint has {actual}"
            ),
            Self::Extra { name, shape, dtype } => write!(
                f,
                "EXTRA          {name}: checkpoint has {shape:?} {dtype}, manifest does not \
                 mention it"
            ),
        }
    }
}

/// The expected tensor inventory for one checkpoint.
#[derive(Clone, Debug, Default)]
pub struct WeightsManifest {
    label: String,
    expected: BTreeMap<String, ExpectedTensor>,
}

impl WeightsManifest {
    /// Create an empty manifest labelled for diagnostics (e.g. `"talker"`).
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            expected: BTreeMap::new(),
        }
    }

    /// Build a manifest from an iterator of expectations.
    #[must_use]
    pub fn from_expectations(
        label: impl Into<String>,
        expectations: impl IntoIterator<Item = ExpectedTensor>,
    ) -> Self {
        let mut manifest = Self::new(label);
        for expectation in expectations {
            manifest.expect(expectation);
        }
        manifest
    }

    /// Add one expectation, replacing any prior entry for the same name.
    pub fn expect(&mut self, tensor: ExpectedTensor) -> &mut Self {
        self.expected.insert(tensor.name.clone(), tensor);
        self
    }

    /// Diagnostic label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Number of tensors required.
    #[must_use]
    pub fn len(&self) -> usize {
        self.expected.len()
    }

    /// Whether the manifest requires nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.expected.is_empty()
    }

    /// Compare a parsed checkpoint against this manifest.
    ///
    /// Never short-circuits: the report lists *every* divergence, because a first-failure-only
    /// diagnostic turns a wrong-checkpoint diagnosis into a game of whack-a-mole.
    #[must_use]
    pub fn audit(&self, index: &SafetensorsIndex) -> CensusReport {
        let mut findings = Vec::new();

        for (name, expectation) in &self.expected {
            match index.entry(name) {
                None => findings.push(Finding::Missing {
                    name: name.clone(),
                    expected_shape: expectation.shape.clone(),
                    expected_dtype: expectation.dtype,
                }),
                Some(actual) => {
                    if actual.shape != expectation.shape {
                        findings.push(Finding::ShapeMismatch {
                            name: name.clone(),
                            expected: expectation.shape.clone(),
                            actual: actual.shape.clone(),
                        });
                    }
                    // Reported independently of shape: a tensor can be right-shaped and
                    // wrong-typed, and each misleads a different consumer.
                    if actual.dtype != expectation.dtype {
                        findings.push(Finding::DtypeMismatch {
                            name: name.clone(),
                            expected: expectation.dtype,
                            actual: actual.dtype,
                        });
                    }
                }
            }
        }

        for entry in index.entries() {
            if !self.expected.contains_key(&entry.name) {
                findings.push(Finding::Extra {
                    name: entry.name.clone(),
                    shape: entry.shape.clone(),
                    dtype: entry.dtype,
                });
            }
        }

        CensusReport {
            label: self.label.clone(),
            expected_count: self.expected.len(),
            actual_count: index.len(),
            findings,
        }
    }

    /// Audit and refuse if anything diverged.
    ///
    /// # Errors
    ///
    /// Returns the full [`CensusReport`] when any finding is present. Callers should surface
    /// [`CensusReport::render`] verbatim — it is the diagnosis.
    pub fn verify(&self, index: &SafetensorsIndex) -> Result<(), Box<CensusReport>> {
        let report = self.audit(index);
        if report.is_green() {
            Ok(())
        } else {
            Err(Box::new(report))
        }
    }
}

/// The outcome of auditing one checkpoint.
#[derive(Clone, Debug)]
pub struct CensusReport {
    label: String,
    expected_count: usize,
    actual_count: usize,
    findings: Vec<Finding>,
}

impl CensusReport {
    /// Whether the checkpoint matched the manifest exactly.
    #[must_use]
    pub fn is_green(&self) -> bool {
        self.findings.is_empty()
    }

    /// Every divergence found.
    #[must_use]
    pub fn findings(&self) -> &[Finding] {
        &self.findings
    }

    /// Count of findings in one class.
    #[must_use]
    pub fn count_of(&self, class: &str) -> usize {
        self.findings
            .iter()
            .filter(|finding| finding.class() == class)
            .count()
    }

    /// The manifest label this report is for.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// A loud, itemized, greppable diff suitable for printing verbatim on refusal.
    #[must_use]
    pub fn render(&self) -> String {
        use fmt::Write as _;

        if self.is_green() {
            return format!(
                "weights census `{}`: GREEN ({} tensors matched)",
                self.label, self.expected_count
            );
        }

        let mut out = format!(
            "weights census `{}`: REFUSED — {} divergence(s)\n  manifest expects {} tensor(s); \
             checkpoint declares {}\n  MISSING {} · SHAPE-MISMATCH {} · DTYPE-MISMATCH {} · EXTRA \
             {}\n",
            self.label,
            self.findings.len(),
            self.expected_count,
            self.actual_count,
            self.count_of("MISSING"),
            self.count_of("SHAPE-MISMATCH"),
            self.count_of("DTYPE-MISMATCH"),
            self.count_of("EXTRA"),
        );
        for finding in &self.findings {
            // Writing to a String cannot fail; the Result is discarded deliberately.
            let _ = writeln!(out, "  {finding}");
        }
        out.push_str(
            "  this is a wrong or stale checkpoint — refusing to load rather than synthesize \
             garbage",
        );
        out
    }
}

impl fmt::Display for CensusReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

impl std::error::Error for CensusReport {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::SafetensorsIndex;
    use serde_json::{Value, json};

    fn checkpoint(parts: &[(&str, Dtype, &[usize])]) -> Vec<u8> {
        let mut directory = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape) in parts {
            let elements: usize = shape.iter().product();
            let bytes = elements * dtype.size();
            directory.insert(
                (*name).to_owned(),
                json!({
                    "dtype": dtype.as_str(),
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let header = serde_json::to_vec(&Value::Object(directory)).expect("serializes");
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&vec![0u8; offset]);
        out
    }

    fn manifest(parts: &[(&str, &[usize], Dtype)]) -> WeightsManifest {
        WeightsManifest::from_expectations(
            "test",
            parts
                .iter()
                .map(|(name, shape, dtype)| ExpectedTensor::new(*name, shape.to_vec(), *dtype)),
        )
    }

    #[test]
    fn matching_checkpoint_is_green() {
        let buffer = checkpoint(&[("a", Dtype::Bf16, &[2, 2]), ("b", Dtype::F32, &[4])]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[("a", &[2, 2], Dtype::Bf16), ("b", &[4], Dtype::F32)]);

        let report = manifest.audit(&index);
        assert!(report.is_green(), "{}", report.render());
        assert!(manifest.verify(&index).is_ok());
        assert!(report.render().contains("GREEN"));
    }

    #[test]
    fn missing_tensor_is_named_and_refused() {
        let buffer = checkpoint(&[("a", Dtype::Bf16, &[2, 2])]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[("a", &[2, 2], Dtype::Bf16), ("b", &[4], Dtype::F32)]);

        let report = manifest.verify(&index).expect_err("must refuse");
        assert_eq!(report.count_of("MISSING"), 1);
        let rendered = report.render();
        assert!(rendered.contains("MISSING"));
        assert!(rendered.contains('b'));
        assert!(rendered.contains("REFUSED"));
    }

    #[test]
    fn shape_mismatch_is_named_and_refused() {
        let buffer = checkpoint(&[("w", Dtype::Bf16, &[2, 4])]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[("w", &[2, 2], Dtype::Bf16)]);

        let report = manifest.verify(&index).expect_err("must refuse");
        assert_eq!(report.count_of("SHAPE-MISMATCH"), 1);
        let rendered = report.render();
        assert!(rendered.contains("[2, 2]"), "{rendered}");
        assert!(rendered.contains("[2, 4]"), "{rendered}");
    }

    #[test]
    fn dtype_mismatch_is_reported_independently_of_shape() {
        // Right shape, wrong dtype: exactly the case that loads fine and sounds wrong.
        let buffer = checkpoint(&[("w", Dtype::F32, &[2, 2])]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[("w", &[2, 2], Dtype::Bf16)]);

        let report = manifest.verify(&index).expect_err("must refuse");
        assert_eq!(report.count_of("DTYPE-MISMATCH"), 1);
        assert_eq!(report.count_of("SHAPE-MISMATCH"), 0);
    }

    #[test]
    fn wrong_shape_and_dtype_both_report() {
        let buffer = checkpoint(&[("w", Dtype::F32, &[8])]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[("w", &[2, 2], Dtype::Bf16)]);

        let report = manifest.verify(&index).expect_err("must refuse");
        assert_eq!(report.count_of("SHAPE-MISMATCH"), 1);
        assert_eq!(report.count_of("DTYPE-MISMATCH"), 1);
    }

    #[test]
    fn extra_tensor_is_reported() {
        // The signature of a different checkpoint revision.
        let buffer = checkpoint(&[("a", Dtype::Bf16, &[2, 2]), ("surprise", Dtype::F32, &[1])]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[("a", &[2, 2], Dtype::Bf16)]);

        let report = manifest.verify(&index).expect_err("must refuse");
        assert_eq!(report.count_of("EXTRA"), 1);
        assert!(report.render().contains("surprise"));
    }

    #[test]
    fn every_divergence_is_listed_not_just_the_first() {
        let buffer = checkpoint(&[
            ("keep", Dtype::Bf16, &[2, 2]),
            ("wrong_shape", Dtype::Bf16, &[9]),
            ("wrong_dtype", Dtype::F32, &[2]),
            ("unexpected", Dtype::F32, &[1]),
        ]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[
            ("keep", &[2, 2], Dtype::Bf16),
            ("wrong_shape", &[4], Dtype::Bf16),
            ("wrong_dtype", &[2], Dtype::Bf16),
            ("absent", &[7], Dtype::Bf16),
        ]);

        let report = manifest.verify(&index).expect_err("must refuse");
        assert_eq!(report.count_of("MISSING"), 1);
        assert_eq!(report.count_of("SHAPE-MISMATCH"), 1);
        assert_eq!(report.count_of("DTYPE-MISMATCH"), 1);
        assert_eq!(report.count_of("EXTRA"), 1);
        assert_eq!(report.findings().len(), 4);
    }

    #[test]
    fn report_is_greppable_by_class() {
        let buffer = checkpoint(&[("a", Dtype::Bf16, &[1])]);
        let index = SafetensorsIndex::parse(&buffer).expect("parses");
        let manifest = manifest(&[("b", &[1], Dtype::Bf16)]);
        let report = manifest.audit(&index);

        let classes: Vec<_> = report.findings().iter().map(Finding::class).collect();
        assert!(classes.contains(&"MISSING"));
        assert!(classes.contains(&"EXTRA"));
        assert_eq!(report.findings()[0].name(), "b");
    }
}
