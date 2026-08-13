//! Shared quantization primitives for runtime loading and offline conversion.
//!
//! The offline `.fttsq` converter must not own a second numerical recipe. Both paths call the
//! row primitive in this module, so their Q8 bytes and scales are identical by construction.

use std::collections::BTreeSet;
use std::fmt;

use serde_json::Value;

use crate::census::{CensusReport, WeightsManifest};
use crate::fttsq::{
    AccessClass, FttsqError, FttsqStreamPlan, FttsqStreamingWriter, StoredDtype,
    TensorEntry as ArtifactTensorEntry,
};
use crate::safetensors::{Dtype, SafetensorsIndex, TensorView, WeightsError};
use crate::sha256::Sha256;

/// Largest input width accepted by the bounded Q8 matrix-row adapter.
///
/// The pinned tensor inventory's largest trailing shape product is 12,288. This leaves more than
/// five times that headroom while bounding the adapter's `f32` + Q8 scratch space to 320 KiB. A
/// new checkpoint or a malformed external input with a wider row must be given an explicit tiling
/// policy rather than turning one "row" into an unbounded allocation.
pub const MAX_Q8_OUTPUT_CHANNEL_WIDTH: usize = 65_536;

/// Largest number of Q8 output channels whose scales one conversion section may retain.
///
/// The pinned checkpoint's widest matrix is the 151,936-row text embedding, whose scale tail is
/// 607,744 bytes. This cap leaves room for a future similarly sized tensor but fixes the tail at
/// one MiB: a converter cannot quietly turn a malicious outer dimension into an unbounded scale
/// allocation while it waits to append that tail after the Q8 payload.
pub const MAX_Q8_OUTPUT_CHANNELS: usize = 262_144;

/// Group width for [`TensorStoragePolicy::Q8PerGroup64`], fixed rather than configurable.
///
/// 64 is where the cold-embedding SQNR sweep flattened (per-row 23.8 dB worst → 35.0 dB at
/// group 64, +1.2 dB more at group 32 for double the scale bytes, frankentts-6ea1), and one
/// fixed width keeps every grouped artifact readable by every grouped-aware loader — a
/// per-artifact knob would be a compatibility surface with no measured benefit.
pub const Q8_GROUP_WIDTH: usize = 64;

/// Largest number of per-group scales one grouped conversion section may retain.
///
/// The grouped scale tail is necessarily larger than the per-row tail: the 151,936×2048 text
/// embedding at group 64 carries 4,861,952 scales (18.5 MiB), which the sink holds until the
/// payload finishes streaming. This cap admits that tensor with headroom while still refusing
/// to let a malicious shape turn the tail into an unbounded allocation.
pub const MAX_Q8_GROUP_SCALES: usize = 8_388_608;

/// The storage recipe for one source tensor in a portable `.fttsq` artifact.
///
/// The conversion plan must state this policy for every tensor in its source manifest. That makes
/// protected high-precision tensors an explicit, auditable choice rather than an accidental
/// fallback, and it prevents a new checkpoint tensor from being silently omitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorStoragePolicy {
    /// Preserve the source BF16 or F32 bytes exactly.
    Verbatim,
    /// Quantize a rank-two-or-greater weight matrix with canonical per-output-channel Q8 scales.
    Q8PerOutputChannel,
    /// Quantize with one canonical Q8 scale per [`Q8_GROUP_WIDTH`]-element group of each row.
    ///
    /// For rows whose energy is uneven across the row (the cold text embedding's common-token
    /// rows), a single row scale quantizes the quiet stretches at the loud stretch's step size;
    /// per-group scales recover ~11 dB on the worst measured rows for ~3% payload overhead in
    /// scales. The quantization primitive is the same [`quantize_output_channel_q8`], applied
    /// per group, so grouped bytes remain bit-consistent with the canonical recipe.
    Q8PerGroup64,
}

/// One source tensor's explicit artifact location and storage recipe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorConversion {
    source_name: String,
    artifact_name: String,
    access_class: AccessClass,
    storage: TensorStoragePolicy,
}

impl TensorConversion {
    /// Declares a source tensor that remains at its source precision.
    #[must_use]
    pub fn verbatim(
        source_name: impl Into<String>,
        artifact_name: impl Into<String>,
        access_class: AccessClass,
    ) -> Self {
        Self {
            source_name: source_name.into(),
            artifact_name: artifact_name.into(),
            access_class,
            storage: TensorStoragePolicy::Verbatim,
        }
    }

    /// Declares a source matrix that uses the shared canonical Q8 quantization primitive.
    #[must_use]
    pub fn q8_per_output_channel(
        source_name: impl Into<String>,
        artifact_name: impl Into<String>,
        access_class: AccessClass,
    ) -> Self {
        Self {
            source_name: source_name.into(),
            artifact_name: artifact_name.into(),
            access_class,
            storage: TensorStoragePolicy::Q8PerOutputChannel,
        }
    }

    /// Declares a source matrix quantized with one Q8 scale per [`Q8_GROUP_WIDTH`]-element group.
    #[must_use]
    pub fn q8_per_group_64(
        source_name: impl Into<String>,
        artifact_name: impl Into<String>,
        access_class: AccessClass,
    ) -> Self {
        Self {
            source_name: source_name.into(),
            artifact_name: artifact_name.into(),
            access_class,
            storage: TensorStoragePolicy::Q8PerGroup64,
        }
    }

    /// The access-class section this tensor's bytes land in.
    ///
    /// One section per [`AccessClass`], never per tensor: the format caps sections at
    /// [`crate::fttsq::MAX_SECTIONS`] because a section IS an access class (the page-in policy
    /// unit), while tensors locate themselves inside it by offset. Emitting per-tensor sections
    /// overflowed that cap at 478 on the real checkpoint (frankentts-zm5).
    fn section_name(&self) -> &'static str {
        self.access_class.as_str()
    }

    fn scales_name(&self) -> String {
        format!("{}.scales", self.artifact_name)
    }
}

/// Metadata, pinned source digest, and per-tensor policy for one bounded conversion.
///
/// A real model recipe supplies one [`TensorConversion`] for **every** tensor named by its
/// [`WeightsManifest`]. The plan contains no source payload and no machine-specific packing; it
/// can therefore be reviewed before a multi-gigabyte conversion opens an output file.
#[derive(Clone, Debug)]
pub struct StreamingConversionPlan {
    model_family: String,
    source_sha256: String,
    license_notice: String,
    model_config: Value,
    quantization_manifest: Value,
    tensors: Vec<TensorConversion>,
}

impl StreamingConversionPlan {
    /// Starts a portable conversion plan tied to the expected source SHA-256.
    #[must_use]
    pub fn new(model_family: impl Into<String>, source_sha256: impl Into<String>) -> Self {
        Self {
            model_family: model_family.into(),
            source_sha256: source_sha256.into(),
            license_notice: String::new(),
            model_config: Value::Null,
            quantization_manifest: Value::Null,
            tensors: Vec::new(),
        }
    }

    /// Sets the required Apache-2.0 attribution and change notice.
    #[must_use]
    pub fn license_notice(mut self, notice: impl Into<String>) -> Self {
        self.license_notice = notice.into();
        self
    }

    /// Records the frozen source model configuration in the artifact directory.
    #[must_use]
    pub fn model_config(mut self, config: Value) -> Self {
        self.model_config = config;
        self
    }

    /// Records the reviewed per-tensor quantization recipe in the artifact directory.
    #[must_use]
    pub fn quantization_manifest(mut self, manifest: Value) -> Self {
        self.quantization_manifest = manifest;
        self
    }

    /// Adds one explicit source-to-artifact tensor conversion.
    #[must_use]
    pub fn tensor(mut self, tensor: TensorConversion) -> Self {
        self.tensors.push(tensor);
        self
    }
}

/// A plan validation failure detected before a destination stream is opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConversionPlanError {
    /// The plan did not declare a conversion recipe for any source tensor.
    NoTensorPolicies,
    /// One source tensor was declared twice with conflicting or duplicate policies.
    DuplicateSourcePolicy {
        /// Source tensor name.
        name: String,
    },
    /// A policy named a tensor absent from the validated source checkpoint.
    SourceTensorMissing {
        /// Source tensor name.
        name: String,
    },
    /// A source tensor had no policy, so emitting an artifact would silently omit it.
    SourceTensorUnplanned {
        /// Source tensor name.
        name: String,
    },
    /// Two policies would create the same artifact tensor name.
    DuplicateArtifactTensor {
        /// Artifact tensor name.
        name: String,
    },
    /// Artifact names cannot be empty because the container uses them as stable keys.
    EmptyArtifactTensorName {
        /// Source tensor whose artifact name was empty.
        source_name: String,
    },
    /// A Q8 policy was assigned to a non-matrix tensor.
    Q8RequiresMatrix {
        /// Source tensor name.
        name: String,
        /// Source rank.
        rank: usize,
    },
    /// A Q8 matrix had no values in an output channel.
    Q8EmptyOutputChannel {
        /// Source tensor name.
        name: String,
    },
    /// A Q8 matrix would exceed the converter's fixed row scratch bound.
    Q8OutputChannelTooWide {
        /// Source tensor name.
        name: String,
        /// Values per output channel.
        width: usize,
        /// Fixed adapter limit.
        limit: usize,
    },
    /// A Q8 scale tail would exceed the converter's fixed memory bound.
    Q8OutputChannelCountTooLarge {
        /// Source tensor name.
        name: String,
        /// Output-channel count.
        rows: usize,
        /// Fixed scale-tail limit.
        limit: usize,
    },
    /// The source shape cannot be represented by the portable u64 container directory.
    ShapeOutOfRange {
        /// Source tensor name.
        name: String,
    },
    /// The derived section length could not fit in the portable container format.
    SectionLengthOverflow {
        /// Source tensor name.
        name: String,
    },
}

impl fmt::Display for ConversionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTensorPolicies => f.write_str("conversion plan has no tensor policies"),
            Self::DuplicateSourcePolicy { name } => {
                write!(
                    f,
                    "conversion plan names source tensor `{name}` more than once"
                )
            }
            Self::SourceTensorMissing { name } => {
                write!(
                    f,
                    "conversion plan names source tensor `{name}`, which is absent"
                )
            }
            Self::SourceTensorUnplanned { name } => {
                write!(
                    f,
                    "source tensor `{name}` has no explicit conversion policy"
                )
            }
            Self::DuplicateArtifactTensor { name } => {
                write!(
                    f,
                    "conversion plan would emit artifact tensor `{name}` more than once"
                )
            }
            Self::EmptyArtifactTensorName { source_name } => write!(
                f,
                "conversion plan gives source tensor `{source_name}` an empty artifact name"
            ),
            Self::Q8RequiresMatrix { name, rank } => write!(
                f,
                "Q8 conversion for `{name}` requires rank 2 or greater, got rank {rank}"
            ),
            Self::Q8EmptyOutputChannel { name } => {
                write!(f, "Q8 conversion for `{name}` has an empty output channel")
            }
            Self::Q8OutputChannelTooWide { name, width, limit } => write!(
                f,
                "Q8 conversion for `{name}` has output-channel width {width}, exceeding {limit}"
            ),
            Self::Q8OutputChannelCountTooLarge { name, rows, limit } => write!(
                f,
                "Q8 conversion for `{name}` has {rows} output channels, exceeding {limit}"
            ),
            Self::ShapeOutOfRange { name } => {
                write!(
                    f,
                    "source tensor `{name}` has a shape outside the artifact range"
                )
            }
            Self::SectionLengthOverflow { name } => {
                write!(
                    f,
                    "source tensor `{name}` overflows its planned artifact section length"
                )
            }
        }
    }
}

impl std::error::Error for ConversionPlanError {}

/// Failure while converting a manifest-validated safetensors checkpoint.
#[derive(Debug)]
pub enum StreamingConversionError {
    /// The source bytes were not a valid supported safetensors file.
    Source(WeightsError),
    /// The source file did not match its complete pinned manifest.
    SourceCensus(Box<CensusReport>),
    /// The source bytes did not match the plan's pinned SHA-256.
    SourceDigestMismatch {
        /// Digest the plan requires.
        expected: String,
        /// Digest calculated over the exact source bytes.
        actual: String,
    },
    /// The conversion plan was incomplete or structurally inconsistent.
    Plan(ConversionPlanError),
    /// The container refused planned metadata or could not write the destination stream.
    Artifact(FttsqError),
    /// The shared Q8 primitive refused a source matrix or destination section.
    Quantization(MatrixQuantizationError<Q8SectionSinkError>),
}

impl fmt::Display for StreamingConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Source(error) => write!(f, "cannot parse source checkpoint: {error}"),
            Self::SourceCensus(report) => f.write_str(&report.render()),
            Self::SourceDigestMismatch { expected, actual } => write!(
                f,
                "source checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
            ),
            Self::Plan(error) => write!(f, "invalid conversion plan: {error}"),
            Self::Artifact(error) => write!(f, "cannot write .fttsq artifact: {error}"),
            Self::Quantization(error) => write!(f, "cannot quantize artifact matrix: {error}"),
        }
    }
}

impl std::error::Error for StreamingConversionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            Self::SourceCensus(report) => Some(report),
            Self::Plan(error) => Some(error),
            Self::Artifact(error) => Some(error),
            Self::Quantization(error) => Some(error),
            Self::SourceDigestMismatch { .. } => None,
        }
    }
}

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

/// Failure while writing canonical Q8 values and their scale tail into one `.fttsq` section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Q8SectionSinkError {
    /// The conversion plan would require a scale tail larger than the bounded contract permits.
    OutputChannelCountTooLarge {
        /// Number of matrix output channels.
        rows: usize,
        /// Maximum number of rows whose scales this sink can retain.
        limit: usize,
    },
    /// A caller bypassed the matrix adapter and supplied an unbounded Q8 row directly.
    OutputChannelTooWide {
        /// Number of values in the attempted row.
        width: usize,
        /// Maximum row width accepted by the shared adapter.
        limit: usize,
    },
    /// The shared matrix adapter did not present rows in the source's physical order.
    RowOutOfOrder {
        /// Row index the sink expected next.
        expected: usize,
        /// Row index the adapter supplied.
        actual: usize,
    },
    /// The caller tried to finalize before every planned row supplied a scale.
    Incomplete {
        /// Rows the section metadata declared.
        expected: usize,
        /// Rows actually received.
        written: usize,
    },
    /// Writing the values or scale tail into the artifact stream failed.
    Artifact(FttsqError),
}

impl fmt::Display for Q8SectionSinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputChannelCountTooLarge { rows, limit } => write!(
                f,
                "Q8 matrix has {rows} output channels, exceeding the bounded scale-tail limit {limit}"
            ),
            Self::OutputChannelTooWide { width, limit } => write!(
                f,
                "Q8 section row width {width} exceeds the bounded row limit {limit}"
            ),
            Self::RowOutOfOrder { expected, actual } => write!(
                f,
                "Q8 section expected source row {expected}, received row {actual}"
            ),
            Self::Incomplete { expected, written } => write!(
                f,
                "Q8 section needs {expected} scales but received {written}"
            ),
            Self::Artifact(error) => write!(f, "cannot write Q8 section: {error}"),
        }
    }
}

impl std::error::Error for Q8SectionSinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            Self::OutputChannelCountTooLarge { .. }
            | Self::OutputChannelTooWide { .. }
            | Self::RowOutOfOrder { .. }
            | Self::Incomplete { .. } => None,
        }
    }
}

/// A bounded bridge from canonical Q8 matrix rows into one streaming `.fttsq` section.
///
/// `.fttsq` keeps the Q8 tensor contiguous, followed by the contiguous F32 scale tensor that its
/// directory names. The sink therefore streams every Q8 row immediately, retaining only the scale
/// tail (at most one MiB for the pinned inventory) until the values have completed. It also keeps
/// one byte-per-row scratch buffer for the signed-to-wire byte conversion, so the total working
/// set remains bounded by the row adapter's 320 KiB plus at most 64 KiB of value bytes and one MiB
/// of scales — never by the full matrix size.
pub struct Q8SectionSink<'a, W> {
    writer: &'a mut FttsqStreamingWriter<W>,
    section: String,
    expected_rows: usize,
    next_row: usize,
    value_bytes: Vec<u8>,
    scale_bytes: Vec<u8>,
}

impl<'a, W: std::io::Write + std::io::Seek> Q8SectionSink<'a, W> {
    /// Starts writing one Q8 matrix section with a fixed number of output channels.
    ///
    /// # Errors
    ///
    /// Returns [`Q8SectionSinkError::OutputChannelCountTooLarge`] before allocating when the
    /// planned F32 scale tail exceeds this adapter's one-MiB memory ceiling.
    pub fn new(
        writer: &'a mut FttsqStreamingWriter<W>,
        section: impl Into<String>,
        expected_rows: usize,
    ) -> Result<Self, Q8SectionSinkError> {
        if expected_rows > MAX_Q8_OUTPUT_CHANNELS {
            return Err(Q8SectionSinkError::OutputChannelCountTooLarge {
                rows: expected_rows,
                limit: MAX_Q8_OUTPUT_CHANNELS,
            });
        }
        Ok(Self::unbounded(writer, section, expected_rows))
    }

    /// Starts writing one grouped-Q8 matrix section: one scale per group, not per row.
    ///
    /// Split from [`Q8SectionSink::new`] because the two paths have honestly different tail
    /// bounds — see [`MAX_Q8_GROUP_SCALES`] — and sharing the larger bound would quietly weaken
    /// the per-row path's one-MiB refusal.
    ///
    /// # Errors
    ///
    /// Returns [`Q8SectionSinkError::OutputChannelCountTooLarge`] before allocating when the
    /// planned scale tail exceeds [`MAX_Q8_GROUP_SCALES`].
    pub fn new_grouped(
        writer: &'a mut FttsqStreamingWriter<W>,
        section: impl Into<String>,
        expected_groups: usize,
    ) -> Result<Self, Q8SectionSinkError> {
        if expected_groups > MAX_Q8_GROUP_SCALES {
            return Err(Q8SectionSinkError::OutputChannelCountTooLarge {
                rows: expected_groups,
                limit: MAX_Q8_GROUP_SCALES,
            });
        }
        Ok(Self::unbounded(writer, section, expected_groups))
    }

    fn unbounded(
        writer: &'a mut FttsqStreamingWriter<W>,
        section: impl Into<String>,
        expected_rows: usize,
    ) -> Self {
        Self {
            writer,
            section: section.into(),
            expected_rows,
            next_row: 0,
            value_bytes: Vec::new(),
            scale_bytes: Vec::with_capacity(expected_rows * std::mem::size_of::<f32>()),
        }
    }

    /// Appends the scale tail after all Q8 values have been streamed.
    ///
    /// # Errors
    ///
    /// Returns a named refusal when a prior row failed or a stream write cannot complete.
    pub fn finish(self) -> Result<(), Q8SectionSinkError> {
        if self.next_row != self.expected_rows {
            return Err(Q8SectionSinkError::Incomplete {
                expected: self.expected_rows,
                written: self.next_row,
            });
        }
        self.writer
            .write_section(&self.section, &self.scale_bytes)
            .map_err(Q8SectionSinkError::Artifact)
    }
}

impl<W: std::io::Write + std::io::Seek> Q8RowSink for Q8SectionSink<'_, W> {
    type Error = Q8SectionSinkError;

    fn write_q8_row(&mut self, row: usize, scale: f32, values: &[i8]) -> Result<(), Self::Error> {
        if row != self.next_row {
            return Err(Q8SectionSinkError::RowOutOfOrder {
                expected: self.next_row,
                actual: row,
            });
        }
        if values.len() > MAX_Q8_OUTPUT_CHANNEL_WIDTH {
            return Err(Q8SectionSinkError::OutputChannelTooWide {
                width: values.len(),
                limit: MAX_Q8_OUTPUT_CHANNEL_WIDTH,
            });
        }
        self.value_bytes.clear();
        self.value_bytes.extend(
            values
                .iter()
                .map(|&value| u8::from_ne_bytes(value.to_ne_bytes())),
        );
        self.writer
            .write_section(&self.section, &self.value_bytes)
            .map_err(Q8SectionSinkError::Artifact)?;
        self.scale_bytes.extend_from_slice(&scale.to_le_bytes());
        self.next_row += 1;
        Ok(())
    }
}

/// Converts one safetensors matrix directly into its declared Q8 `.fttsq` section.
///
/// The section must have been declared with exactly `matrix.len() + rows * 4` bytes, with a Q8
/// tensor at relative offset zero followed by its F32 scale tensor. This function calls the shared
/// [`quantize_output_channel_q8`] path through [`quantize_matrix_q8_rows`], so offline artifact
/// bytes and runtime quantization are produced by the same numerical primitive.
///
/// # Errors
///
/// Returns a precise source-shape, quantization, bounded-scale-tail, or artifact-write failure.
pub fn stream_matrix_q8_section<W: std::io::Write + std::io::Seek>(
    matrix: &TensorView<'_>,
    writer: &mut FttsqStreamingWriter<W>,
    section: &str,
) -> Result<(), MatrixQuantizationError<Q8SectionSinkError>> {
    let shape = matrix.shape();
    if shape.len() < 2 {
        return Err(MatrixQuantizationError::ExpectedMatrix { rank: shape.len() });
    }
    let Some(&row_count) = shape.first() else {
        return Err(MatrixQuantizationError::ExpectedMatrix { rank: 0 });
    };
    let mut sink = Q8SectionSink::new(writer, section, row_count)
        .map_err(|source| MatrixQuantizationError::Sink { row: 0, source })?;
    quantize_matrix_q8_rows(matrix, &mut sink)?;
    sink.finish()
        .map_err(|source| MatrixQuantizationError::Sink {
            row: row_count,
            source,
        })
}

/// Converts one safetensors matrix into a grouped-Q8 `.fttsq` section (one scale per
/// [`Q8_GROUP_WIDTH`]-element group of every row).
///
/// Payload layout is byte-identical to the per-row form — groups of a row are contiguous, rows
/// follow each other — so a reader walks the same row-major i8 bytes and only the scale lookup
/// changes. Each group runs through the same [`quantize_output_channel_q8`] primitive the
/// per-row path uses (a group is a row of width [`Q8_GROUP_WIDTH`] to the primitive), keeping
/// offline artifact bytes and runtime quantization numerically identical by construction.
///
/// # Errors
///
/// Returns a precise source-shape, quantization, bounded-scale-tail, or artifact-write failure;
/// a row width not divisible by [`Q8_GROUP_WIDTH`] is refused before any byte is written.
pub fn stream_matrix_q8_group64_section<W: std::io::Write + std::io::Seek>(
    matrix: &TensorView<'_>,
    writer: &mut FttsqStreamingWriter<W>,
    section: &str,
) -> Result<(), MatrixQuantizationError<Q8SectionSinkError>> {
    let shape = matrix.shape();
    if shape.len() < 2 {
        return Err(MatrixQuantizationError::ExpectedMatrix { rank: shape.len() });
    }
    let Some(&row_count) = shape.first() else {
        return Err(MatrixQuantizationError::ExpectedMatrix { rank: 0 });
    };
    let row_width = matrix.row_len();
    if row_width == 0 || !row_width.is_multiple_of(Q8_GROUP_WIDTH) {
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
    let groups_per_row = row_width / Q8_GROUP_WIDTH;
    let total_groups = row_count.checked_mul(groups_per_row).ok_or(
        MatrixQuantizationError::OutputChannelTooWide {
            width: row_width,
            limit: MAX_Q8_OUTPUT_CHANNEL_WIDTH,
        },
    )?;
    let mut sink = Q8SectionSink::new_grouped(writer, section, total_groups)
        .map_err(|source| MatrixQuantizationError::Sink { row: 0, source })?;

    let mut source_row = vec![0.0_f32; row_width];
    let mut quantized_group = [0_i8; Q8_GROUP_WIDTH];
    for row in 0..row_count {
        if !matrix.copy_row_f32(row, &mut source_row) {
            return Err(MatrixQuantizationError::SourceRowUnavailable { row });
        }
        for (group_index, group) in source_row
            .as_chunks::<Q8_GROUP_WIDTH>()
            .0
            .iter()
            .enumerate()
        {
            let scale = quantize_output_channel_q8(group, &mut quantized_group)
                .map_err(|source| MatrixQuantizationError::Quantization { row, source })?;
            sink.write_q8_row(row * groups_per_row + group_index, scale, &quantized_group)
                .map_err(|source| MatrixQuantizationError::Sink { row, source })?;
        }
    }
    sink.finish()
        .map_err(|source| MatrixQuantizationError::Sink {
            row: row_count,
            source,
        })
}

/// Converts a manifest-validated safetensors checkpoint into a portable `.fttsq` stream.
///
/// The source is borrowed so a caller may provide a memory map rather than a copied checkpoint.
/// Before the output stream is opened, this function parses the safetensors directory, verifies
/// the complete [`WeightsManifest`], verifies the plan's source SHA-256, and checks that every
/// source tensor has exactly one explicit policy. It then writes one complete section per source
/// tensor in plan order: high-precision payloads are copied verbatim and Q8 matrices use
/// [`quantize_output_channel_q8`] through [`stream_matrix_q8_section`].
///
/// The destination is caller-owned deliberately. Pass a same-filesystem temporary file, sync it,
/// and rename it only after this returns successfully; a failed conversion must never publish a
/// partial artifact. The stream itself never retains a source tensor, Q8 payload, or section
/// payload after it has been written.
///
/// # Errors
///
/// Refuses invalid safetensors bytes, a stale or wrong source manifest, digest mismatches,
/// incomplete/ambiguous policy coverage, non-finite Q8 values, or container I/O/metadata errors.
pub fn convert_safetensors_streaming<W: std::io::Write + std::io::Seek>(
    source: &[u8],
    manifest: &WeightsManifest,
    plan: &StreamingConversionPlan,
    destination: W,
) -> Result<W, StreamingConversionError> {
    let index = SafetensorsIndex::parse(source).map_err(StreamingConversionError::Source)?;
    manifest
        .verify(&index)
        .map_err(StreamingConversionError::SourceCensus)?;

    let actual_digest = sha256_hex(source);
    if actual_digest != plan.source_sha256 {
        return Err(StreamingConversionError::SourceDigestMismatch {
            expected: plan.source_sha256.clone(),
            actual: actual_digest,
        });
    }

    let artifact_plan =
        build_artifact_plan(&index, plan).map_err(StreamingConversionError::Plan)?;
    let mut writer = artifact_plan
        .begin(destination)
        .map_err(StreamingConversionError::Artifact)?;

    for tensor in tensors_in_write_order(plan) {
        let matrix_or_values = index.view(&tensor.source_name, source).ok_or_else(|| {
            StreamingConversionError::Plan(ConversionPlanError::SourceTensorMissing {
                name: tensor.source_name.clone(),
            })
        })?;
        let section = tensor.section_name();
        match tensor.storage {
            TensorStoragePolicy::Verbatim => writer
                .write_section(section, matrix_or_values.as_bytes())
                .map_err(StreamingConversionError::Artifact)?,
            TensorStoragePolicy::Q8PerOutputChannel => {
                stream_matrix_q8_section(&matrix_or_values, &mut writer, section)
                    .map_err(StreamingConversionError::Quantization)?;
            }
            TensorStoragePolicy::Q8PerGroup64 => {
                stream_matrix_q8_group64_section(&matrix_or_values, &mut writer, section)
                    .map_err(StreamingConversionError::Quantization)?;
            }
        }
    }

    writer.finish().map_err(StreamingConversionError::Artifact)
}

fn build_artifact_plan(
    index: &SafetensorsIndex,
    plan: &StreamingConversionPlan,
) -> Result<FttsqStreamPlan, ConversionPlanError> {
    if plan.tensors.is_empty() {
        return Err(ConversionPlanError::NoTensorPolicies);
    }

    let mut seen_sources = BTreeSet::<String>::new();
    let mut seen_artifacts = BTreeSet::<String>::new();
    for tensor in &plan.tensors {
        if !seen_sources.insert(tensor.source_name.clone()) {
            return Err(ConversionPlanError::DuplicateSourcePolicy {
                name: tensor.source_name.clone(),
            });
        }
        if index.entry(&tensor.source_name).is_none() {
            return Err(ConversionPlanError::SourceTensorMissing {
                name: tensor.source_name.clone(),
            });
        }
        if tensor.artifact_name.is_empty() {
            return Err(ConversionPlanError::EmptyArtifactTensorName {
                source_name: tensor.source_name.clone(),
            });
        }
        if !seen_artifacts.insert(tensor.artifact_name.clone()) {
            return Err(ConversionPlanError::DuplicateArtifactTensor {
                name: tensor.artifact_name.clone(),
            });
        }
        if matches!(
            tensor.storage,
            TensorStoragePolicy::Q8PerOutputChannel | TensorStoragePolicy::Q8PerGroup64
        ) {
            let entry = index.entry(&tensor.source_name).ok_or_else(|| {
                ConversionPlanError::SourceTensorMissing {
                    name: tensor.source_name.clone(),
                }
            })?;
            if entry.shape.len() < 2 {
                return Err(ConversionPlanError::Q8RequiresMatrix {
                    name: tensor.source_name.clone(),
                    rank: entry.shape.len(),
                });
            }
            let Some((&rows, trailing_shape)) = entry.shape.split_first() else {
                return Err(ConversionPlanError::Q8RequiresMatrix {
                    name: tensor.source_name.clone(),
                    rank: 0,
                });
            };
            let row_width = trailing_shape
                .iter()
                .try_fold(1_usize, |product, &dimension| {
                    product.checked_mul(dimension)
                })
                .ok_or_else(|| ConversionPlanError::ShapeOutOfRange {
                    name: tensor.source_name.clone(),
                })?;
            if row_width == 0 {
                return Err(ConversionPlanError::Q8EmptyOutputChannel {
                    name: tensor.source_name.clone(),
                });
            }
            if row_width > MAX_Q8_OUTPUT_CHANNEL_WIDTH {
                return Err(ConversionPlanError::Q8OutputChannelTooWide {
                    name: tensor.source_name.clone(),
                    width: row_width,
                    limit: MAX_Q8_OUTPUT_CHANNEL_WIDTH,
                });
            }
            if rows > MAX_Q8_OUTPUT_CHANNELS {
                return Err(ConversionPlanError::Q8OutputChannelCountTooLarge {
                    name: tensor.source_name.clone(),
                    rows,
                    limit: MAX_Q8_OUTPUT_CHANNELS,
                });
            }
            if tensor.storage == TensorStoragePolicy::Q8PerGroup64 {
                // The grouped stream refuses this too, but the plan is reviewed before a
                // multi-gigabyte conversion opens an output file — refuse it here first.
                if !row_width.is_multiple_of(Q8_GROUP_WIDTH) {
                    return Err(ConversionPlanError::Q8EmptyOutputChannel {
                        name: tensor.source_name.clone(),
                    });
                }
                let groups = rows * (row_width / Q8_GROUP_WIDTH);
                if groups > MAX_Q8_GROUP_SCALES {
                    return Err(ConversionPlanError::Q8OutputChannelCountTooLarge {
                        name: tensor.source_name.clone(),
                        rows: groups,
                        limit: MAX_Q8_GROUP_SCALES,
                    });
                }
            }
            let scales_name = tensor.scales_name();
            if !seen_artifacts.insert(scales_name.clone()) {
                return Err(ConversionPlanError::DuplicateArtifactTensor { name: scales_name });
            }
        }
    }

    for entry in index.entries() {
        if !seen_sources.contains(&entry.name) {
            return Err(ConversionPlanError::SourceTensorUnplanned {
                name: entry.name.clone(),
            });
        }
    }

    let mut artifact_plan = FttsqStreamPlan::new(&plan.model_family, &plan.source_sha256)
        .license_notice(&plan.license_notice)
        .model_config(plan.model_config.clone())
        .quantization_manifest(plan.quantization_manifest.clone());

    //  One section per access class, tensors located by running offset inside it. The write loop
    //  must emit payloads in exactly this order, so both sides iterate [`tensors_in_write_order`].
    let mut section_offsets: std::collections::BTreeMap<&'static str, u64> =
        std::collections::BTreeMap::new();
    let mut declared_sections: Vec<&'static str> = Vec::new();
    for tensor in tensors_in_write_order(plan) {
        let entry = index.entry(&tensor.source_name).ok_or_else(|| {
            ConversionPlanError::SourceTensorMissing {
                name: tensor.source_name.clone(),
            }
        })?;
        let shape = artifact_shape(entry, &tensor.source_name)?;
        let section = tensor.section_name();
        if !declared_sections.contains(&section) {
            declared_sections.push(section);
        }
        let running = section_offsets.entry(section).or_insert(0);
        match tensor.storage {
            TensorStoragePolicy::Verbatim => {
                let length = u64::try_from(entry.byte_len()).map_err(|_| {
                    ConversionPlanError::SectionLengthOverflow {
                        name: tensor.source_name.clone(),
                    }
                })?;
                artifact_plan = artifact_plan.tensor(ArtifactTensorEntry {
                    name: tensor.artifact_name.clone(),
                    section: section.to_owned(),
                    dtype: stored_dtype(entry.dtype),
                    shape,
                    offset: *running,
                    length,
                    scales: None,
                });
                *running = running.checked_add(length).ok_or_else(|| {
                    ConversionPlanError::SectionLengthOverflow {
                        name: tensor.source_name.clone(),
                    }
                })?;
            }
            TensorStoragePolicy::Q8PerOutputChannel | TensorStoragePolicy::Q8PerGroup64 => {
                let rows = entry.shape.first().copied().ok_or_else(|| {
                    ConversionPlanError::Q8RequiresMatrix {
                        name: tensor.source_name.clone(),
                        rank: entry.shape.len(),
                    }
                })?;
                // Per-row storage keeps one scale per output channel; grouped storage keeps one
                // per Q8_GROUP_WIDTH-element group. The payload bytes are identical either way;
                // only the scale tensor's element count and declared shape differ.
                let (scale_count, scales_shape) = if tensor.storage
                    == TensorStoragePolicy::Q8PerGroup64
                {
                    let row_width = entry
                        .element_count()
                        .checked_div(rows)
                        .filter(|width| width.is_multiple_of(Q8_GROUP_WIDTH))
                        .ok_or_else(|| ConversionPlanError::Q8EmptyOutputChannel {
                            name: tensor.source_name.clone(),
                        })?;
                    let groups_per_row = row_width / Q8_GROUP_WIDTH;
                    let rows_u64 =
                        u64::try_from(rows).map_err(|_| ConversionPlanError::ShapeOutOfRange {
                            name: tensor.source_name.clone(),
                        })?;
                    let groups_u64 = u64::try_from(groups_per_row).map_err(|_| {
                        ConversionPlanError::ShapeOutOfRange {
                            name: tensor.source_name.clone(),
                        }
                    })?;
                    (rows * groups_per_row, vec![rows_u64, groups_u64])
                } else {
                    (
                        rows,
                        vec![u64::try_from(rows).map_err(|_| {
                            ConversionPlanError::ShapeOutOfRange {
                                name: tensor.source_name.clone(),
                            }
                        })?],
                    )
                };
                let values_len = u64::try_from(entry.element_count()).map_err(|_| {
                    ConversionPlanError::SectionLengthOverflow {
                        name: tensor.source_name.clone(),
                    }
                })?;
                let scales_len = u64::try_from(scale_count)
                    .ok()
                    .and_then(|count| {
                        count.checked_mul(u64::try_from(std::mem::size_of::<f32>()).ok()?)
                    })
                    .ok_or_else(|| ConversionPlanError::SectionLengthOverflow {
                        name: tensor.source_name.clone(),
                    })?;
                let section_len = values_len.checked_add(scales_len).ok_or_else(|| {
                    ConversionPlanError::SectionLengthOverflow {
                        name: tensor.source_name.clone(),
                    }
                })?;
                let scales_name = tensor.scales_name();
                artifact_plan = artifact_plan
                    .tensor(ArtifactTensorEntry {
                        name: tensor.artifact_name.clone(),
                        section: section.to_owned(),
                        dtype: StoredDtype::Q8,
                        shape,
                        offset: *running,
                        length: values_len,
                        scales: Some(scales_name.clone()),
                    })
                    .tensor(ArtifactTensorEntry {
                        name: scales_name,
                        section: section.to_owned(),
                        dtype: StoredDtype::F32,
                        shape: scales_shape,
                        offset: running.checked_add(values_len).ok_or_else(|| {
                            ConversionPlanError::SectionLengthOverflow {
                                name: tensor.source_name.clone(),
                            }
                        })?,
                        length: scales_len,
                        scales: None,
                    });
                *running = running.checked_add(section_len).ok_or_else(|| {
                    ConversionPlanError::SectionLengthOverflow {
                        name: tensor.source_name.clone(),
                    }
                })?;
            }
        }
    }

    //  Declare the access-class sections in first-touch order with their accumulated lengths;
    //  the write loop replays the same order, so every section fills exactly to its declaration.
    for section in declared_sections {
        let class = section_access_class(section);
        let length = section_offsets
            .get(section)
            .copied()
            .expect("declared sections accumulate a length");
        artifact_plan = artifact_plan.section(section, class, length);
    }

    Ok(artifact_plan)
}

/// The stable payload order shared by planning and writing: grouped by access-class section in
/// first-appearance order, original recipe order preserved within each class.
fn tensors_in_write_order(plan: &StreamingConversionPlan) -> Vec<&TensorConversion> {
    let mut order: Vec<&'static str> = Vec::new();
    for tensor in &plan.tensors {
        let section = tensor.section_name();
        if !order.contains(&section) {
            order.push(section);
        }
    }
    let mut grouped = Vec::with_capacity(plan.tensors.len());
    for section in order {
        grouped.extend(
            plan.tensors
                .iter()
                .filter(|tensor| tensor.section_name() == section),
        );
    }
    grouped
}

/// Maps a section wire name back to its access class; sections and classes are one-to-one.
fn section_access_class(name: &str) -> AccessClass {
    for class in [
        AccessClass::HotRecurrentMicrodecoder,
        AccessClass::HotRecurrentTalker,
        AccessClass::HotCodecDecoder,
        AccessClass::ColdTextEmbedding,
        AccessClass::EnrollmentSpeakerEncoder,
        AccessClass::EnrollmentCodecEncoder,
        AccessClass::Metadata,
    ] {
        if class.as_str() == name {
            return class;
        }
    }
    unreachable!("section names are minted from AccessClass::as_str")
}

fn artifact_shape(
    entry: &crate::safetensors::TensorEntry,
    source_name: &str,
) -> Result<Vec<u64>, ConversionPlanError> {
    entry
        .shape
        .iter()
        .copied()
        .map(u64::try_from)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ConversionPlanError::ShapeOutOfRange {
            name: source_name.to_owned(),
        })
}

const fn stored_dtype(source: Dtype) -> StoredDtype {
    match source {
        Dtype::Bf16 => StoredDtype::Bf16,
        Dtype::F32 => StoredDtype::F32,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finish()
    };
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
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
    use crate::census::ExpectedTensor;
    use crate::fttsq::{AccessClass, FttsqReader, FttsqStreamPlan, StoredDtype, TensorEntry};
    use crate::safetensors::SafetensorsIndex;
    use serde_json::json;
    use std::convert::Infallible;
    use std::io::Cursor;

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

    fn safetensors(parts: &[(&str, Dtype, &[usize], &[u8])]) -> Vec<u8> {
        let mut directory = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, bytes) in parts {
            let begin = payload.len();
            payload.extend_from_slice(bytes);
            directory.insert(
                (*name).to_owned(),
                json!({
                    "dtype": dtype.as_str(),
                    "shape": shape,
                    "data_offsets": [begin, payload.len()],
                }),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(directory))
            .expect("fixture directory serializes");
        let mut source = (header.len() as u64).to_le_bytes().to_vec();
        source.extend_from_slice(&header);
        source.extend_from_slice(&payload);
        source
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
    fn matrix_q8_section_streams_values_then_bounded_scale_tail() {
        let source = f32_matrix(2, 3, &[1.0, -2.0, 0.5, 3.0, 0.0, -3.0]);
        let index = SafetensorsIndex::parse(&source).expect("fixture parses");
        let matrix = index.view("matrix", &source).expect("matrix view exists");
        let plan = FttsqStreamPlan::new("test-model", "a".repeat(64))
            .license_notice("Copyright 2026 Alibaba Cloud\nApache-2.0")
            .section("matrix", AccessClass::HotRecurrentTalker, 14)
            .tensor(TensorEntry {
                name: "matrix.weight".to_owned(),
                section: "matrix".to_owned(),
                dtype: StoredDtype::Q8,
                shape: vec![2, 3],
                offset: 0,
                length: 6,
                scales: Some("matrix.weight.scales".to_owned()),
            })
            .tensor(TensorEntry {
                name: "matrix.weight.scales".to_owned(),
                section: "matrix".to_owned(),
                dtype: StoredDtype::F32,
                shape: vec![2],
                offset: 6,
                length: 8,
                scales: None,
            });
        let mut writer = plan
            .begin(Cursor::new(Vec::new()))
            .expect("section metadata is valid");

        stream_matrix_q8_section(&matrix, &mut writer, "matrix")
            .expect("matrix streams through the canonical Q8 primitive");
        let artifact = writer
            .finish()
            .expect("completed section finalizes its digest")
            .into_inner();
        let reader = FttsqReader::open(&artifact).expect("artifact verifies");

        assert_eq!(
            reader
                .tensor_bytes("matrix.weight", &artifact)
                .expect("Q8 bytes resolve"),
            &[64, 129, 32, 127, 0, 129]
        );
        let scales = reader
            .tensor_bytes("matrix.weight.scales", &artifact)
            .expect("scale bytes resolve");
        assert_eq!(
            scales,
            &[
                (2.0_f32 / 127.0).to_le_bytes(),
                (3.0_f32 / 127.0).to_le_bytes(),
            ]
            .concat()
        );
    }

    #[test]
    fn grouped_q8_section_carries_one_scale_per_group_and_dequantizes_per_group() {
        // Two rows of two groups each: within each row, one loud group and one quiet group.
        // A per-row scale would quantize the quiet group at the loud group's step; per-group
        // scales must recover it exactly at this tiny size (each group has <= 127 magnitudes).
        let quiet = [0.00127_f32, -0.0005];
        let source = f32_matrix(
            2,
            2 * Q8_GROUP_WIDTH,
            &[
                std::iter::repeat_n(1.27_f32, Q8_GROUP_WIDTH).collect::<Vec<_>>(),
                quiet.iter().copied().cycle().take(Q8_GROUP_WIDTH).collect(),
                std::iter::repeat_n(-2.54_f32, Q8_GROUP_WIDTH).collect(),
                quiet.iter().copied().cycle().take(Q8_GROUP_WIDTH).collect(),
            ]
            .concat(),
        );
        let index = SafetensorsIndex::parse(&source).expect("fixture parses");
        let matrix = index.view("matrix", &source).expect("matrix view exists");
        let values_len = 2 * 2 * Q8_GROUP_WIDTH as u64;
        let plan = FttsqStreamPlan::new("test-model", "a".repeat(64))
            .license_notice("Copyright 2026 Alibaba Cloud\nApache-2.0")
            .section("matrix", AccessClass::ColdTextEmbedding, values_len + 16)
            .tensor(TensorEntry {
                name: "matrix.weight".to_owned(),
                section: "matrix".to_owned(),
                dtype: StoredDtype::Q8,
                shape: vec![2, 2 * Q8_GROUP_WIDTH as u64],
                offset: 0,
                length: values_len,
                scales: Some("matrix.weight.scales".to_owned()),
            })
            .tensor(TensorEntry {
                name: "matrix.weight.scales".to_owned(),
                section: "matrix".to_owned(),
                dtype: StoredDtype::F32,
                shape: vec![2, 2],
                offset: values_len,
                length: 16,
                scales: None,
            });
        let mut writer = plan
            .begin(Cursor::new(Vec::new()))
            .expect("section metadata is valid");
        stream_matrix_q8_group64_section(&matrix, &mut writer, "matrix")
            .expect("grouped matrix streams through the canonical primitive");
        let artifact = writer
            .finish()
            .expect("completed section finalizes its digest")
            .into_inner();
        let reader = FttsqReader::open(&artifact).expect("artifact verifies");

        let scales: Vec<f32> = reader
            .tensor_bytes("matrix.weight.scales", &artifact)
            .expect("scale bytes resolve")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(
            scales.iter().map(|s| s.to_bits()).collect::<Vec<_>>(),
            [
                1.27_f32 / 127.0,
                0.00127_f32 / 127.0,
                2.54_f32 / 127.0,
                0.00127_f32 / 127.0,
            ]
            .iter()
            .map(|s| s.to_bits())
            .collect::<Vec<_>>(),
            "each group carries its own max-abs scale"
        );
        // The quiet groups dequantize exactly: their values are exact multiples of their own
        // group scale, which the loud rows' scales could never represent.
        let bytes = reader
            .tensor_bytes("matrix.weight", &artifact)
            .expect("Q8 bytes resolve");
        let quiet_group_of_row_0 = &bytes[Q8_GROUP_WIDTH..2 * Q8_GROUP_WIDTH];
        for (index, &byte) in quiet_group_of_row_0.iter().enumerate() {
            let value = f32::from(i8::from_ne_bytes([byte])) * scales[1];
            let expected = quiet[index % 2];
            assert!(
                (value - expected).abs() < 1e-9,
                "quiet element {index}: {value} vs {expected}"
            );
        }
    }

    #[test]
    fn manifest_verified_multi_tensor_stream_is_deterministic_and_verbatim_where_required() {
        let weight = [1.0_f32, -2.0, 0.5, 3.0, 0.0, -3.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let bias = [0x80_u16, 0x3f80]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let source = safetensors(&[
            ("weight", Dtype::F32, &[2, 3], &weight),
            ("bias", Dtype::Bf16, &[2], &bias),
        ]);
        let manifest = WeightsManifest::from_expectations(
            "small pinned fixture",
            [
                ExpectedTensor::new("weight", vec![2, 3], Dtype::F32),
                ExpectedTensor::new("bias", vec![2], Dtype::Bf16),
            ],
        );
        let plan = StreamingConversionPlan::new("qwen3-tts-fixture", sha256_hex(&source))
            .license_notice("Copyright 2026 Alibaba Cloud\nApache-2.0")
            .model_config(json!({ "fixture": true }))
            .quantization_manifest(json!({
                "weight": "q8_per_output_channel",
                "bias": "verbatim_bf16",
            }))
            .tensor(TensorConversion::q8_per_output_channel(
                "weight",
                "weight",
                AccessClass::HotRecurrentTalker,
            ))
            .tensor(TensorConversion::verbatim(
                "bias",
                "bias",
                AccessClass::Metadata,
            ));

        let first =
            convert_safetensors_streaming(&source, &manifest, &plan, Cursor::new(Vec::new()))
                .expect("fixture converts")
                .into_inner();
        let second =
            convert_safetensors_streaming(&source, &manifest, &plan, Cursor::new(Vec::new()))
                .expect("second fixture conversion is deterministic")
                .into_inner();
        assert_eq!(
            first, second,
            "identical source and plan must be byte-identical"
        );

        let reader = FttsqReader::open(&first).expect("artifact verifies its section digests");
        let mut runtime_q8 = [0_i8; 6];
        let runtime_first_scale =
            quantize_output_channel_q8(&[1.0_f32, -2.0, 0.5], &mut runtime_q8[..3])
                .expect("shared runtime primitive quantizes the first row");
        let runtime_second_scale =
            quantize_output_channel_q8(&[3.0_f32, 0.0, -3.0], &mut runtime_q8[3..])
                .expect("shared runtime primitive quantizes the second row");
        assert_eq!(
            reader
                .tensor_bytes("weight", &first)
                .expect("Q8 weights resolve"),
            runtime_q8.map(|value| value as u8)
        );
        assert_eq!(
            reader
                .tensor_bytes("weight.scales", &first)
                .expect("Q8 scales resolve"),
            &[
                runtime_first_scale.to_le_bytes(),
                runtime_second_scale.to_le_bytes(),
            ]
            .concat()
        );
        assert_eq!(
            reader
                .tensor_bytes("bias", &first)
                .expect("protected BF16 values resolve"),
            bias
        );
    }

    #[test]
    fn streaming_conversion_refuses_unpinned_source_before_writing() {
        let source = f32_matrix(1, 2, &[1.0, -1.0]);
        let manifest = WeightsManifest::from_expectations(
            "digest fixture",
            [ExpectedTensor::new("matrix", vec![1, 2], Dtype::F32)],
        );
        let plan = StreamingConversionPlan::new("qwen3-tts-fixture", "0".repeat(64))
            .license_notice("Copyright 2026 Alibaba Cloud\nApache-2.0")
            .tensor(TensorConversion::q8_per_output_channel(
                "matrix",
                "matrix",
                AccessClass::HotRecurrentTalker,
            ));

        let error =
            convert_safetensors_streaming(&source, &manifest, &plan, Cursor::new(Vec::new()))
                .expect_err("a wrong source digest must refuse before artifact construction");
        assert!(matches!(
            error,
            StreamingConversionError::SourceDigestMismatch { .. }
        ));
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
