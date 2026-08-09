//! Hydration of the pinned Qwen3-TTS checkpoint into the borrowed weight structs.
//!
//! Every other module in this crate takes `&[f32]` slices and knows nothing about files. This one
//! is the single place that reads a `.safetensors` checkpoint and produces owned buffers those
//! slices can point into, so `ftts say` and the conformance tests hydrate the *same* way instead
//! of each growing its own copy of two hundred tensor names.
//!
//! # Two checkpoints, not one
//!
//! The talker, microdecoder, and embeddings live in the main `model.safetensors`; the codec
//! decoder lives in `speech_tokenizer/model.safetensors`. They are loaded independently
//! ([`TalkerCheckpoint`], [`CodecCheckpoint`]) because the codec is useful on its own — decoding a
//! captured code stream to PCM needs no talker at all.
//!
//! # The cold text embedding is gathered, never widened whole
//!
//! `talker.model.text_embedding.weight` is `[151936, 2048]`: 622 MB of bf16, and 1.24 GB once
//! widened to `f32`. An utterance touches a few dozen rows of it. [`TextEmbeddingTable::gather`]
//! therefore allocates the full-width table — the generator indexes it by real token id, and
//! remapping ids would break the wrapper-id checks in [`crate::prompt`] — but *fills* only the
//! requested rows. The untouched pages are never faulted in, so the resident cost is the rows we
//! asked for rather than the whole table. This is the `ColdTextEmbedding` access class of the plan
//! showing up as a concrete allocation policy, not an optimization to schedule later.
//!
//! # The prompt header is derived, not invented
//!
//! [`TalkerCheckpoint::xvector_header`] builds the common prompt header from checkpoint tensors
//! and pinned token ids. Every one of its rows was solved against the ft7 oracle capture — see
//! `crates/ftts-conformance/tests/prompt_header_derivation.rs`, which reproduces the reference's
//! own `inputs_embeds` from these tensors and fails if any row drifts. Nothing here is a
//! plausible-looking zero vector.

use crate::codec::{
    CausalConvWeights, CausalTransposeConvWeights, CodecConfig, CodecConvNextWeights,
    CodecDecoderBlockWeights, CodecDecoderWeights, CodecError, CodecPreTransformerWeights,
    CodecResidualUnitWeights, CodecTransformerLayerWeights, CodecUpsampleStageWeights,
    MaterializedCodebook, SplitResidualVectorQuantizer, decode_codec_offline,
};
use crate::generate::{FeedbackTables, TextEmbeddingWeights};
use crate::microdecoder::{LayerWeights, MicrodecoderConfig, MicrodecoderWeights, RESIDUAL_DEPTHS};
use crate::prompt::{HiddenState, PromptHeader, ROLE_PREFIX_IDS};
use crate::talker::{TalkerLayerWeights, TalkerWeights};
use ftts_artifacts::{
    fttsq::{MappedFttsq, StoredDtype},
    safetensors::SafetensorsFile,
};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Talker decoder layers in the pinned Base checkpoint.
pub const TALKER_LAYER_COUNT: usize = 28;

/// Code groups per frame: one primary plus fifteen residuals.
pub const CODE_GROUP_COUNT: usize = 16;

/// Width of the talker hidden state.
pub const TALKER_HIDDEN: usize = 1_024;

/// Width of one cold text-embedding row, before the text projection narrows it to the hidden size.
pub const TEXT_EMBED_WIDTH: usize = 2_048;

/// Rows in the cold text embedding table.
pub const TEXT_VOCAB: usize = 151_936;

/// `<|im_end|>` — closes the assistant wrapper.
pub const IM_END_TOKEN_ID: u32 = 151_645;

/// Newline, the third id of the role prefix and of the target suffix.
pub const NEWLINE_TOKEN_ID: u32 = 198;

/// The text-side TTS pad id, summed into every codec-only header position.
pub const TTS_PAD_TOKEN_ID: u32 = 151_671;

/// The text-side TTS BOS id, which marks the last header position.
pub const TTS_BOS_TOKEN_ID: u32 = 151_672;

/// The text-side TTS EOS id, which trails the target text.
pub const TTS_EOS_TOKEN_ID: u32 = 151_673;

/// Codec-side pad id (`talker_config.codec_pad_id`).
pub const CODEC_PAD_ID: usize = 2_148;

/// Codec-side BOS id (`talker_config.codec_bos_id`).
pub const CODEC_BOS_ID: usize = 2_149;

/// Codec-side think-block open id (`talker_config.codec_think_id`).
pub const CODEC_THINK_ID: usize = 2_154;

/// Codec-side think BOS id (`talker_config.codec_think_bos_id`).
pub const CODEC_THINK_BOS_ID: usize = 2_156;

/// Codec-side think EOS id (`talker_config.codec_think_eos_id`).
pub const CODEC_THINK_EOS_ID: usize = 2_157;

/// `talker_config.codec_language_id.english`.
///
/// The think block carries exactly one language id. The pinned config lists ten; English is the
/// only one this project has an oracle capture for, so it is the only one named here. Adding
/// another means adding its capture, not adding its integer.
pub const CODEC_LANGUAGE_ENGLISH_ID: usize = 2_050;

/// The exact assistant wrapper the reference puts around target text.
///
/// [`crate::prompt::extract_prompt_text_ids`] slices `input_id[:, 3:-5]` back off, so these two
/// must stay in agreement: the wrapper written here is the wrapper stripped there.
const TARGET_SUFFIX_IDS: [u32; 5] = [
    IM_END_TOKEN_ID,
    NEWLINE_TOKEN_ID,
    ROLE_PREFIX_IDS[0],
    ROLE_PREFIX_IDS[1],
    ROLE_PREFIX_IDS[2],
];

/// A checkpoint that cannot be hydrated into a usable model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckpointError {
    /// The file could not be opened or mapped.
    Open { path: PathBuf, detail: String },
    /// A required tensor is absent from the checkpoint.
    MissingTensor { path: PathBuf, tensor: String },
    /// A tensor is present but not the shape this model requires.
    TensorShape {
        tensor: String,
        expected: usize,
        actual: usize,
    },
    /// A digest-verified canonical artifact carries a tensor representation this f32 engine
    /// cannot hydrate safely.
    ArtifactTensor {
        path: PathBuf,
        tensor: String,
        detail: String,
    },
    /// A speaker vector was supplied at the wrong width.
    SpeakerWidth { expected: usize, actual: usize },
    /// The codec refused to decode.
    Codec(CodecError),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open { path, detail } => {
                write!(
                    formatter,
                    "cannot open checkpoint {}: {detail}",
                    path.display()
                )
            }
            Self::MissingTensor { path, tensor } => write!(
                formatter,
                "checkpoint {} has no tensor `{tensor}`; this is not a pinned Qwen3-TTS checkpoint",
                path.display()
            ),
            Self::TensorShape {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor `{tensor}` holds {actual} elements, expected {expected}"
            ),
            Self::ArtifactTensor {
                path,
                tensor,
                detail,
            } => write!(
                formatter,
                "canonical artifact {} cannot hydrate tensor `{tensor}`: {detail}",
                path.display()
            ),
            Self::SpeakerWidth { expected, actual } => write!(
                formatter,
                "speaker embedding is {actual} floats wide, expected {expected}"
            ),
            Self::Codec(error) => write!(formatter, "codec decode failed: {error:?}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Widen a whole tensor to owned `f32`.
pub(crate) fn widen(
    file: &SafetensorsFile,
    path: &Path,
    name: &str,
) -> Result<Vec<f32>, CheckpointError> {
    let view = file
        .view(name)
        .ok_or_else(|| CheckpointError::MissingTensor {
            path: path.to_path_buf(),
            tensor: name.to_owned(),
        })?;
    let mut out = vec![0.0f32; view.len()];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = view.get_f32(index).unwrap_or(0.0);
    }
    Ok(out)
}

pub(crate) fn widen_exact(
    file: &SafetensorsFile,
    path: &Path,
    name: &str,
    expected: usize,
) -> Result<Vec<f32>, CheckpointError> {
    let values = widen(file, path, name)?;
    if values.len() != expected {
        return Err(CheckpointError::TensorShape {
            tensor: name.to_owned(),
            expected,
            actual: values.len(),
        });
    }
    Ok(values)
}

pub(crate) fn open(path: &Path) -> Result<SafetensorsFile, CheckpointError> {
    SafetensorsFile::open(path).map_err(|error| CheckpointError::Open {
        path: path.to_path_buf(),
        detail: format!("{error:?}"),
    })
}

fn artifact_tensor_error(path: &Path, tensor: &str, detail: impl Into<String>) -> CheckpointError {
    CheckpointError::ArtifactTensor {
        path: path.to_path_buf(),
        tensor: tensor.to_owned(),
        detail: detail.into(),
    }
}

fn artifact_element_count(
    artifact: &MappedFttsq,
    path: &Path,
    name: &str,
) -> Result<usize, CheckpointError> {
    let entry = artifact
        .reader()
        .tensor(name)
        .ok_or_else(|| CheckpointError::MissingTensor {
            path: path.to_path_buf(),
            tensor: name.to_owned(),
        })?;
    entry
        .elements()
        .and_then(|elements| usize::try_from(elements).ok())
        .ok_or_else(|| artifact_tensor_error(path, name, "logical element count exceeds usize"))
}

fn artifact_f32_values(
    artifact: &MappedFttsq,
    path: &Path,
    name: &str,
) -> Result<Vec<f32>, CheckpointError> {
    let entry = artifact
        .reader()
        .tensor(name)
        .ok_or_else(|| CheckpointError::MissingTensor {
            path: path.to_path_buf(),
            tensor: name.to_owned(),
        })?;
    if entry.dtype != StoredDtype::F32 {
        return Err(artifact_tensor_error(
            path,
            name,
            format!("expected f32 scales, found {}", entry.dtype),
        ));
    }
    let elements = artifact_element_count(artifact, path, name)?;
    let bytes = artifact
        .tensor_bytes(name)
        .map_err(|error| artifact_tensor_error(path, name, error.to_string()))?;
    if bytes.len() != elements.saturating_mul(std::mem::size_of::<f32>()) {
        return Err(artifact_tensor_error(
            path,
            name,
            "f32 payload length does not match its logical element count",
        ));
    }
    Ok(bytes
        .as_chunks::<{ std::mem::size_of::<f32>() }>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

/// Open a canonical `.fttsq` artifact with its integrity checks, as a [`CheckpointError`].
pub(crate) fn open_fttsq(path: &Path) -> Result<MappedFttsq, CheckpointError> {
    MappedFttsq::open(path).map_err(|error| CheckpointError::Open {
        path: path.to_path_buf(),
        detail: error.to_string(),
    })
}

/// Widen one canonical-artifact tensor for the f32 baseline engine.
///
/// Q8 matrices use the converter's exact row-major layout: signed payload bytes followed by an
/// f32 scale for every output channel. Keeping this reconstruction at the accessor boundary lets
/// the current f32 engine execute canonical artifacts without acquiring a second quantization
/// recipe or materializing the cold text embedding.
pub(crate) fn widen_fttsq(
    artifact: &MappedFttsq,
    path: &Path,
    name: &str,
) -> Result<Vec<f32>, CheckpointError> {
    let entry = artifact
        .reader()
        .tensor(name)
        .ok_or_else(|| CheckpointError::MissingTensor {
            path: path.to_path_buf(),
            tensor: name.to_owned(),
        })?;
    let dtype = entry.dtype;
    let shape = entry.shape.clone();
    let scales_name = entry.scales.clone();
    let elements = artifact_element_count(artifact, path, name)?;
    let bytes = artifact
        .tensor_bytes(name)
        .map_err(|error| artifact_tensor_error(path, name, error.to_string()))?;

    match dtype {
        StoredDtype::Bf16 => {
            if bytes.len() != elements.saturating_mul(2) {
                return Err(artifact_tensor_error(
                    path,
                    name,
                    "bf16 payload length does not match its logical element count",
                ));
            }
            Ok(bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| {
                    let bits = u32::from(u16::from_le_bytes(*chunk)) << 16;
                    f32::from_bits(bits)
                })
                .collect())
        }
        StoredDtype::F32 => artifact_f32_values(artifact, path, name),
        StoredDtype::Q8 => {
            let rows = shape
                .first()
                .copied()
                .and_then(|rows| usize::try_from(rows).ok())
                .filter(|rows| *rows > 0)
                .ok_or_else(|| {
                    artifact_tensor_error(path, name, "q8 tensor must have a non-empty row axis")
                })?;
            let width = elements
                .checked_div(rows)
                .filter(|width| *width > 0)
                .ok_or_else(|| {
                    artifact_tensor_error(
                        path,
                        name,
                        "q8 tensor shape has zero-width output channels",
                    )
                })?;
            if width.checked_mul(rows) != Some(elements) || bytes.len() != elements {
                return Err(artifact_tensor_error(
                    path,
                    name,
                    "q8 payload length does not match its logical matrix shape",
                ));
            }
            let scales_name = scales_name.ok_or_else(|| {
                artifact_tensor_error(path, name, "q8 tensor is missing its per-row scales")
            })?;
            let scales = artifact_f32_values(artifact, path, &scales_name)?;
            if scales.len() != rows {
                return Err(artifact_tensor_error(
                    path,
                    name,
                    format!(
                        "q8 tensor has {} output rows but {} scales",
                        rows,
                        scales.len()
                    ),
                ));
            }
            Ok(bytes
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    f32::from(i8::from_ne_bytes([*value])) * scales[index / width]
                })
                .collect())
        }
        StoredDtype::Q4 => Err(artifact_tensor_error(
            path,
            name,
            "q4 storage is not admitted by the f32 baseline loader",
        )),
    }
}

/// The eleven per-layer tensors shared by the talker and the microdecoder, in field order.
struct OwnedAttentionLayer {
    tensors: [Vec<f32>; 11],
}

/// The eleven per-layer tensor name suffixes, in [`OwnedAttentionLayer`] field order.
const ATTENTION_LAYER_SUFFIXES: [&str; 11] = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// The seven projection suffixes whose f32 widening the armed int8 route never reads.
///
/// `forward_talker_q8` / the microdecoder q8 step consume [`ATTENTION_LAYER_SUFFIXES`] norms plus
/// the artifact's own Q8 tables; the f32 projections exist only for the f32 reference forward and
/// the requantize fallback. When neither can run (stack armed, artifact hydration enabled, and
/// the artifact verifiably carries the tensor as Q8), widening them is pure startup and memory
/// waste — ~2.5 GB of f32 nobody reads.
const HOT_PROJECTION_SUFFIXES: [&str; 7] = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// Which decode stacks may skip widening their hot projections at hydration.
///
/// Both flags must reflect the CURRENT armed route for this process: an elided stack's f32
/// projection slices are empty, and the f32 forward's shape assertions fail loudly (never
/// silently) if a route change routes an elided layer back through f32 arithmetic.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HotElision {
    /// Skip the talker stack's seven per-layer projections.
    pub talker: bool,
    /// Skip the microdecoder stack's seven per-layer projections.
    pub micro: bool,
}

/// Widens `names` concurrently, returning the tensors in the order the names were given.
///
/// Hydration writes gigabytes of freshly widened f32 and was measured wholly serial at ~0.4 GB/s
/// on one core; the tensors are independent, so a scoped worker pool with an atomic cursor turns
/// the load into a page-fault-and-memcpy race instead. Output order is positional and each
/// tensor's bytes are computed exactly as the serial loop computed them, so the result is
/// byte-identical to sequential hydration — the parallelism has no numeric surface. This is
/// startup-only work, deliberately outside the decode path's `KernelTeam` discipline.
///
/// The first error wins and is returned after every worker has stopped.
fn widen_many(
    names: &[(String, bool)],
    widen_tensor: &(dyn Fn(&str) -> Result<Vec<f32>, CheckpointError> + Sync),
) -> Result<Vec<Vec<f32>>, CheckpointError> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let workers = std::thread::available_parallelism()
        .map_or(1, std::num::NonZeroUsize::get)
        .min(names.len())
        .max(1);
    if workers == 1 {
        return names
            .iter()
            .map(|(name, elided)| {
                if *elided {
                    Ok(Vec::new())
                } else {
                    widen_tensor(name)
                }
            })
            .collect();
    }

    let cursor = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<Vec<f32>>>> = Mutex::new((0..names.len()).map(|_| None).collect());
    let first_error: Mutex<Option<CheckpointError>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    if index >= names.len() || first_error.lock().expect("error slot").is_some() {
                        return;
                    }
                    let (name, elided) = &names[index];
                    let outcome = if *elided {
                        Ok(Vec::new())
                    } else {
                        widen_tensor(name)
                    };
                    match outcome {
                        Ok(tensor) => {
                            slots.lock().expect("tensor slots")[index] = Some(tensor);
                        }
                        Err(error) => {
                            first_error.lock().expect("error slot").get_or_insert(error);
                            return;
                        }
                    }
                }
            });
        }
    });
    if let Some(error) = first_error.into_inner().expect("error slot") {
        return Err(error);
    }
    Ok(slots
        .into_inner()
        .expect("tensor slots")
        .into_iter()
        .map(|slot| slot.expect("every tensor hydrated or an error was returned"))
        .collect())
}

impl OwnedAttentionLayer {
    /// Reassembles a layer from eleven already-widened tensors in
    /// [`ATTENTION_LAYER_SUFFIXES`] order.
    fn from_widened(tensors: impl Iterator<Item = Vec<f32>>) -> Self {
        let mut slots: [Vec<f32>; 11] = Default::default();
        let mut count = 0;
        for (slot, tensor) in slots.iter_mut().zip(tensors) {
            *slot = tensor;
            count += 1;
        }
        assert_eq!(count, 11, "a layer is exactly eleven tensors");
        Self { tensors: slots }
    }

    fn talker(&self) -> TalkerLayerWeights<'_> {
        let t = &self.tensors;
        TalkerLayerWeights {
            input_layernorm: &t[0],
            q_proj: &t[1],
            k_proj: &t[2],
            v_proj: &t[3],
            q_norm: &t[4],
            k_norm: &t[5],
            o_proj: &t[6],
            post_attention_layernorm: &t[7],
            gate_proj: &t[8],
            up_proj: &t[9],
            down_proj: &t[10],
        }
    }

    fn microdecoder(&self) -> LayerWeights<'_> {
        let t = &self.tensors;
        LayerWeights {
            input_norm: &t[0],
            q_proj: &t[1],
            k_proj: &t[2],
            v_proj: &t[3],
            q_norm: &t[4],
            k_norm: &t[5],
            o_proj: &t[6],
            post_attention_norm: &t[7],
            gate_proj: &t[8],
            up_proj: &t[9],
            down_proj: &t[10],
        }
    }
}

/// A full-width cold text embedding table with only the requested rows materialized.
///
/// See the module docs: the table must stay indexable by real token id, but an utterance reads a
/// few dozen of its 151,936 rows. Unrequested rows stay zero and their pages are never touched.
/// Reading an ungathered id would therefore yield a zero row — [`TextEmbeddingTable::gather`] is
/// given every id the utterance can reach, and [`TextEmbeddingTable::covers`] lets a caller assert
/// that rather than discover it as silence.
pub struct TextEmbeddingTable {
    rows: Vec<f32>,
    gathered: Vec<u32>,
}

enum TextEmbeddingSource {
    Safetensors,
    Fttsq(Arc<MappedFttsq>),
}

impl TextEmbeddingTable {
    /// Materialize exactly `ids` from the cold table.
    ///
    /// # Errors
    ///
    /// If the embedding tensor is missing or is not `[TEXT_VOCAB, TEXT_EMBED_WIDTH]`.
    pub fn gather(
        file: &SafetensorsFile,
        path: &Path,
        ids: &[u32],
    ) -> Result<Self, CheckpointError> {
        let name = "talker.model.text_embedding.weight";
        let view = file
            .view(name)
            .ok_or_else(|| CheckpointError::MissingTensor {
                path: path.to_path_buf(),
                tensor: name.to_owned(),
            })?;
        if view.len() != TEXT_VOCAB * TEXT_EMBED_WIDTH {
            return Err(CheckpointError::TensorShape {
                tensor: name.to_owned(),
                expected: TEXT_VOCAB * TEXT_EMBED_WIDTH,
                actual: view.len(),
            });
        }

        let mut rows = vec![0.0f32; TEXT_VOCAB * TEXT_EMBED_WIDTH];
        let mut gathered: Vec<u32> = ids.to_vec();
        gathered.sort_unstable();
        gathered.dedup();
        for id in &gathered {
            let row = *id as usize;
            if row >= TEXT_VOCAB {
                return Err(CheckpointError::TensorShape {
                    tensor: name.to_owned(),
                    expected: TEXT_VOCAB,
                    actual: row + 1,
                });
            }
            let start = row * TEXT_EMBED_WIDTH;
            if !view.copy_row_f32(row, &mut rows[start..start + TEXT_EMBED_WIDTH]) {
                // A refused copy would leave the row zero, which reads downstream as a legitimate
                // embedding and produces confident wrong audio rather than an error.
                return Err(CheckpointError::TensorShape {
                    tensor: name.to_owned(),
                    expected: TEXT_EMBED_WIDTH,
                    actual: view.row_len(),
                });
            }
        }
        Ok(Self { rows, gathered })
    }

    fn gather_fttsq(
        artifact: &MappedFttsq,
        path: &Path,
        ids: &[u32],
    ) -> Result<Self, CheckpointError> {
        let name = "talker.model.text_embedding.weight";
        let entry =
            artifact
                .reader()
                .tensor(name)
                .ok_or_else(|| CheckpointError::MissingTensor {
                    path: path.to_path_buf(),
                    tensor: name.to_owned(),
                })?;
        let expected_shape = [TEXT_VOCAB as u64, TEXT_EMBED_WIDTH as u64];
        if entry.shape.as_slice() != expected_shape {
            return Err(artifact_tensor_error(
                path,
                name,
                format!(
                    "expected cold embedding shape {expected_shape:?}, found {:?}",
                    entry.shape
                ),
            ));
        }
        let dtype = entry.dtype;
        let scales_name = entry.scales.clone();
        let bytes = artifact
            .tensor_bytes(name)
            .map_err(|error| artifact_tensor_error(path, name, error.to_string()))?;
        let bytes_per_row = match dtype {
            StoredDtype::Bf16 => TEXT_EMBED_WIDTH * 2,
            StoredDtype::F32 => TEXT_EMBED_WIDTH * 4,
            StoredDtype::Q8 => TEXT_EMBED_WIDTH,
            StoredDtype::Q4 => {
                return Err(artifact_tensor_error(
                    path,
                    name,
                    "q4 cold embeddings are not admitted by the f32 baseline loader",
                ));
            }
        };
        if bytes.len() != TEXT_VOCAB * bytes_per_row {
            return Err(artifact_tensor_error(
                path,
                name,
                "cold embedding payload length does not match its declared shape",
            ));
        }
        let q8_scales = if dtype == StoredDtype::Q8 {
            let scales_name = scales_name.ok_or_else(|| {
                artifact_tensor_error(
                    path,
                    name,
                    "q8 cold embedding is missing its per-row scales",
                )
            })?;
            let scales = artifact_f32_values(artifact, path, &scales_name)?;
            if scales.len() != TEXT_VOCAB {
                return Err(artifact_tensor_error(
                    path,
                    name,
                    format!(
                        "q8 cold embedding has {TEXT_VOCAB} rows but {} scales",
                        scales.len()
                    ),
                ));
            }
            Some(scales)
        } else {
            None
        };

        let mut rows = vec![0.0f32; TEXT_VOCAB * TEXT_EMBED_WIDTH];
        let mut gathered: Vec<u32> = ids.to_vec();
        gathered.sort_unstable();
        gathered.dedup();
        for id in &gathered {
            let row = *id as usize;
            if row >= TEXT_VOCAB {
                return Err(CheckpointError::TensorShape {
                    tensor: name.to_owned(),
                    expected: TEXT_VOCAB,
                    actual: row + 1,
                });
            }
            let source = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let destination = &mut rows[row * TEXT_EMBED_WIDTH..(row + 1) * TEXT_EMBED_WIDTH];
            match dtype {
                StoredDtype::Bf16 => {
                    for (slot, bytes) in destination.iter_mut().zip(source.as_chunks::<2>().0) {
                        let bits = u32::from(u16::from_le_bytes(*bytes)) << 16;
                        *slot = f32::from_bits(bits);
                    }
                }
                StoredDtype::F32 => {
                    for (slot, bytes) in destination.iter_mut().zip(source.as_chunks::<4>().0) {
                        *slot = f32::from_le_bytes(*bytes);
                    }
                }
                StoredDtype::Q8 => {
                    let scale = q8_scales
                        .as_ref()
                        .and_then(|scales| scales.get(row))
                        .copied()
                        .ok_or_else(|| {
                            artifact_tensor_error(
                                path,
                                name,
                                "q8 cold-embedding scale lookup failed",
                            )
                        })?;
                    for (slot, value) in destination.iter_mut().zip(source) {
                        *slot = f32::from(i8::from_ne_bytes([*value])) * scale;
                    }
                }
                StoredDtype::Q4 => unreachable!("q4 cold embeddings returned above"),
            }
        }
        Ok(Self { rows, gathered })
    }

    /// Whether every id in `ids` was materialized.
    #[must_use]
    pub fn covers(&self, ids: &[u32]) -> bool {
        ids.iter().all(|id| self.gathered.binary_search(id).is_ok())
    }

    /// The ids that were materialized, ascending.
    #[must_use]
    pub fn gathered_ids(&self) -> &[u32] {
        &self.gathered
    }

    /// The table, indexable by token id.
    #[must_use]
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }
}

/// The talker, microdecoder, feedback tables, and text projection from `model.safetensors`.
pub struct TalkerCheckpoint {
    path: PathBuf,
    text_embedding_source: TextEmbeddingSource,
    talker_layers: Vec<OwnedAttentionLayer>,
    talker_final_norm: Vec<f32>,
    codec_head: Vec<f32>,
    codec_embedding: Vec<f32>,
    residual_embeddings: Vec<Vec<f32>>,
    micro_layers: Vec<OwnedAttentionLayer>,
    micro_final_norm: Vec<f32>,
    micro_heads: Vec<Vec<f32>>,
    fc1_weight: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<f32>,
    fc2_bias: Vec<f32>,
}

impl TalkerCheckpoint {
    /// Hydrate everything except the cold text embedding, which is gathered separately.
    ///
    /// # Errors
    ///
    /// If the file cannot be opened or a required tensor is missing or misshapen.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = open(path)?;
        Self::load_with(
            path,
            TextEmbeddingSource::Safetensors,
            |name| widen(&file, path, name),
            |_| false,
        )
    }

    /// Hydrate the talker and residual-code microdecoder from a canonical `.fttsq` artifact.
    ///
    /// The digest-verified mapping remains alive for sparse cold-embedding gathers. Every hot
    /// Q8 matrix is reconstructed through the converter's per-output-channel scales for the f32
    /// baseline; the native int8 kernels can replace that reconstruction once their parity gate is
    /// green without changing the artifact contract.
    pub fn load_fttsq(path: &Path) -> Result<Self, CheckpointError> {
        Self::load_fttsq_elided(path, HotElision::default())
    }

    /// [`TalkerCheckpoint::load_fttsq`], skipping the f32 widening of hot projections the caller
    /// certifies the armed int8 route will hydrate artifact-natively instead.
    ///
    /// A projection is elided only when its stack's flag is set AND the artifact verifiably
    /// carries it as Q8 with scales — the exact precondition under which the generator's
    /// artifact-native hydration succeeds and the requantize fallback never runs. Elided
    /// projections hydrate as empty slices; any misrouted f32 use fails that path's shape
    /// assertions immediately rather than computing on garbage.
    ///
    /// # Errors
    ///
    /// As [`TalkerCheckpoint::load_fttsq`].
    pub fn load_fttsq_elided(path: &Path, elision: HotElision) -> Result<Self, CheckpointError> {
        let artifact =
            Arc::new(
                MappedFttsq::open(path).map_err(|error| CheckpointError::Open {
                    path: path.to_path_buf(),
                    detail: error.to_string(),
                })?,
            );
        let source = TextEmbeddingSource::Fttsq(Arc::clone(&artifact));
        let elide = {
            let artifact = Arc::clone(&artifact);
            move |name: &str| {
                let stack_elided = if name.starts_with("talker.code_predictor.model.layers.") {
                    elision.micro
                } else if name.starts_with("talker.model.layers.") {
                    elision.talker
                } else {
                    false
                };
                stack_elided
                    && HOT_PROJECTION_SUFFIXES
                        .iter()
                        .any(|suffix| name.ends_with(suffix))
                    && artifact.reader().tensor(name).is_some_and(|entry| {
                        entry.dtype == StoredDtype::Q8 && entry.scales.is_some()
                    })
            }
        };
        Self::load_with(
            path,
            source,
            |name| widen_fttsq(&artifact, path, name),
            elide,
        )
    }

    fn load_with(
        path: &Path,
        text_embedding_source: TextEmbeddingSource,
        widen_tensor: impl Fn(&str) -> Result<Vec<f32>, CheckpointError> + Sync,
        elide_tensor: impl Fn(&str) -> bool,
    ) -> Result<Self, CheckpointError> {
        let hidden = TALKER_HIDDEN;
        let micro_layer_count = MicrodecoderConfig::default().num_layers;

        // One flat, canonically ordered name list, hydrated in one concurrent pass; the struct is
        // then assembled by walking the results in the same order. Field-order changes here must
        // keep the two walks in lockstep -- the assertions at each seam catch a drift.
        let mut names: Vec<(String, bool)> = Vec::new();
        let mut push = |name: String| {
            let elided = elide_tensor(&name);
            names.push((name, elided));
        };
        for layer in 0..TALKER_LAYER_COUNT {
            for suffix in ATTENTION_LAYER_SUFFIXES {
                push(format!("talker.model.layers.{layer}.{suffix}"));
            }
        }
        for layer in 0..micro_layer_count {
            for suffix in ATTENTION_LAYER_SUFFIXES {
                push(format!(
                    "talker.code_predictor.model.layers.{layer}.{suffix}"
                ));
            }
        }
        for table in 0..CODE_GROUP_COUNT - 1 {
            push(format!(
                "talker.code_predictor.model.codec_embedding.{table}.weight"
            ));
        }
        for head in 0..RESIDUAL_DEPTHS {
            push(format!("talker.code_predictor.lm_head.{head}.weight"));
        }
        for single in [
            "talker.model.norm.weight",
            "talker.codec_head.weight",
            "talker.model.codec_embedding.weight",
            "talker.code_predictor.model.norm.weight",
            "talker.text_projection.linear_fc1.weight",
            "talker.text_projection.linear_fc1.bias",
            "talker.text_projection.linear_fc2.weight",
            "talker.text_projection.linear_fc2.bias",
        ] {
            push(single.to_owned());
        }
        // NLL releases `push`'s borrow of `names` here; the closure needs no explicit drop.

        let mut widened = widen_many(&names, &widen_tensor)?.into_iter();

        let talker_layers: Vec<OwnedAttentionLayer> = (0..TALKER_LAYER_COUNT)
            .map(|_| OwnedAttentionLayer::from_widened(widened.by_ref().take(11)))
            .collect();
        let micro_layers: Vec<OwnedAttentionLayer> = (0..micro_layer_count)
            .map(|_| OwnedAttentionLayer::from_widened(widened.by_ref().take(11)))
            .collect();
        let residual_embeddings: Vec<Vec<f32>> =
            widened.by_ref().take(CODE_GROUP_COUNT - 1).collect();
        let micro_heads: Vec<Vec<f32>> = widened.by_ref().take(RESIDUAL_DEPTHS).collect();
        assert_eq!(
            residual_embeddings.len(),
            CODE_GROUP_COUNT - 1,
            "hydration walk drifted at the residual tables"
        );
        assert_eq!(
            micro_heads.len(),
            RESIDUAL_DEPTHS,
            "hydration walk drifted at the heads"
        );

        let mut single =
            |name: &str, expected: Option<usize>| -> Result<Vec<f32>, CheckpointError> {
                let values = widened
                    .next()
                    .expect("hydration walk covers every singleton tensor");
                if let Some(expected) = expected
                    && values.len() != expected
                {
                    return Err(CheckpointError::TensorShape {
                        tensor: name.to_owned(),
                        expected,
                        actual: values.len(),
                    });
                }
                Ok(values)
            };

        Ok(Self {
            path: path.to_path_buf(),
            text_embedding_source,
            talker_layers,
            talker_final_norm: single("talker.model.norm.weight", Some(hidden))?,
            codec_head: single("talker.codec_head.weight", None)?,
            codec_embedding: single("talker.model.codec_embedding.weight", None)?,
            residual_embeddings,
            micro_layers,
            micro_final_norm: single("talker.code_predictor.model.norm.weight", Some(hidden))?,
            micro_heads,
            fc1_weight: single(
                "talker.text_projection.linear_fc1.weight",
                Some(TEXT_EMBED_WIDTH * TEXT_EMBED_WIDTH),
            )?,
            fc1_bias: single(
                "talker.text_projection.linear_fc1.bias",
                Some(TEXT_EMBED_WIDTH),
            )?,
            fc2_weight: single(
                "talker.text_projection.linear_fc2.weight",
                Some(hidden * TEXT_EMBED_WIDTH),
            )?,
            fc2_bias: single("talker.text_projection.linear_fc2.bias", Some(hidden))?,
        })
    }

    /// The path this checkpoint was read from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Gather the cold text-embedding rows an utterance needs.
    ///
    /// Pass every id the prompt can reach: the wrapped target ids plus the three TTS specials.
    /// [`Self::utterance_text_ids`] assembles that set.
    ///
    /// # Errors
    ///
    /// If the embedding tensor is missing or misshapen.
    pub fn gather_text_rows(&self, ids: &[u32]) -> Result<TextEmbeddingTable, CheckpointError> {
        match &self.text_embedding_source {
            TextEmbeddingSource::Safetensors => {
                let file = open(&self.path)?;
                TextEmbeddingTable::gather(&file, &self.path, ids)
            }
            TextEmbeddingSource::Fttsq(artifact) => {
                TextEmbeddingTable::gather_fttsq(artifact, &self.path, ids)
            }
        }
    }

    /// Every text id one utterance can reach: the wrapped target plus the TTS specials.
    #[must_use]
    pub fn utterance_text_ids(wrapped_target: &[u32]) -> Vec<u32> {
        let mut ids = wrapped_target.to_vec();
        ids.extend([TTS_PAD_TOKEN_ID, TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID]);
        ids.extend(ROLE_PREFIX_IDS);
        ids
    }

    /// Wrap raw tokenizer output in the official assistant wrapper.
    ///
    /// The reference tokenizes `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`;
    /// [`crate::prompt::extract_prompt_text_ids`] slices exactly this wrapper back off, so the two
    /// are one contract expressed twice.
    #[must_use]
    pub fn wrap_target_ids(text_ids: &[u32]) -> Vec<u32> {
        let mut wrapped = Vec::with_capacity(text_ids.len() + 8);
        wrapped.extend(ROLE_PREFIX_IDS);
        wrapped.extend_from_slice(text_ids);
        wrapped.extend(TARGET_SUFFIX_IDS);
        wrapped
    }

    /// Borrowed talker layers, in checkpoint order.
    #[must_use]
    pub fn talker_layer_weights(&self) -> Vec<TalkerLayerWeights<'_>> {
        self.talker_layers
            .iter()
            .map(OwnedAttentionLayer::talker)
            .collect()
    }

    /// Borrowed talker weights over `layers` (from [`Self::talker_layer_weights`]).
    #[must_use]
    pub fn talker_weights<'a>(&'a self, layers: &[TalkerLayerWeights<'a>]) -> TalkerWeights<'a> {
        TalkerWeights {
            layers: layers.to_vec(),
            final_norm: &self.talker_final_norm,
            codec_head: &self.codec_head,
        }
    }

    /// Borrowed microdecoder layers, in checkpoint order.
    #[must_use]
    pub fn microdecoder_layer_weights(&self) -> Vec<LayerWeights<'_>> {
        self.micro_layers
            .iter()
            .map(OwnedAttentionLayer::microdecoder)
            .collect()
    }

    /// The fifteen per-depth residual embedding tables.
    #[must_use]
    pub fn residual_embedding_slices(&self) -> Vec<&[f32]> {
        self.residual_embeddings.iter().map(Vec::as_slice).collect()
    }

    /// The fifteen per-depth 2,048-way heads.
    #[must_use]
    pub fn microdecoder_head_slices(&self) -> Vec<&[f32]> {
        self.micro_heads.iter().map(Vec::as_slice).collect()
    }

    /// Borrowed microdecoder weights.
    ///
    /// `layers` is [`Self::microdecoder_layer_weights`], `residual` is
    /// [`Self::residual_embedding_slices`], and `heads` is [`Self::microdecoder_head_slices`]. The
    /// microdecoder's internal tables for depths 2..=15 are the first fourteen of the same
    /// per-depth set the feedback path uses, so `residual` is passed here already trimmed by the
    /// caller when the microdecoder wants fourteen rather than fifteen.
    #[must_use]
    pub fn microdecoder_weights<'a>(
        &'a self,
        layers: &'a [LayerWeights<'a>],
        residual: &'a [&'a [f32]],
        heads: &'a [&'a [f32]],
    ) -> MicrodecoderWeights<'a> {
        MicrodecoderWeights {
            layers,
            talker_codec_embedding: &self.codec_embedding,
            residual_embeddings: residual,
            heads,
            final_norm: &self.micro_final_norm,
        }
    }

    /// The sixteen talker-input feedback tables.
    #[must_use]
    pub fn feedback_tables<'a>(&'a self, residual: &'a [&'a [f32]]) -> FeedbackTables<'a> {
        FeedbackTables {
            talker_codec: &self.codec_embedding,
            residual: residual.to_vec(),
        }
    }

    /// The text embedding + projection path, over a gathered table.
    #[must_use]
    pub fn text_weights<'a>(&'a self, table: &'a TextEmbeddingTable) -> TextEmbeddingWeights<'a> {
        TextEmbeddingWeights {
            table: table.rows(),
            embed_width: TEXT_EMBED_WIDTH,
            fc1_weight: &self.fc1_weight,
            fc1_bias: &self.fc1_bias,
            fc2_weight: &self.fc2_weight,
            fc2_bias: &self.fc2_bias,
        }
    }

    /// Text ids through the cold embedding and the biased SiLU text projection.
    ///
    /// The generator runs this same path internally for the target text; it is exposed here so a
    /// caller assembling a prompt by hand — the header derivation test, chiefly — projects text the
    /// one way rather than growing a second implementation that can drift.
    #[must_use]
    pub fn project_text_ids(&self, table: &TextEmbeddingTable, ids: &[u32]) -> Vec<HiddenState> {
        ids.iter()
            .map(|id| self.project_text_id(table, *id))
            .collect()
    }

    /// One text id through the cold embedding and the biased SiLU projection.
    fn project_text_id(&self, table: &TextEmbeddingTable, id: u32) -> HiddenState {
        let start = id as usize * TEXT_EMBED_WIDTH;
        let embed = &table.rows()[start..start + TEXT_EMBED_WIDTH];

        let mut inner = vec![0.0f32; TEXT_EMBED_WIDTH];
        for (row, slot) in inner.iter_mut().enumerate() {
            let weights = &self.fc1_weight[row * TEXT_EMBED_WIDTH..(row + 1) * TEXT_EMBED_WIDTH];
            let sum: f32 = weights.iter().zip(embed).map(|(w, x)| w * x).sum();
            let biased = sum + self.fc1_bias[row];
            // SiLU, matching the pinned `hidden_act`.
            *slot = biased / (1.0 + (-biased).exp());
        }

        let mut out = vec![0.0f32; TALKER_HIDDEN];
        for (row, slot) in out.iter_mut().enumerate() {
            let weights = &self.fc2_weight[row * TEXT_EMBED_WIDTH..(row + 1) * TEXT_EMBED_WIDTH];
            let sum: f32 = weights.iter().zip(&inner).map(|(w, x)| w * x).sum();
            *slot = sum + self.fc2_bias[row];
        }
        out
    }

    /// One codec id's talker-input embedding row.
    fn codec_row(&self, id: usize) -> HiddenState {
        self.codec_embedding[id * TALKER_HIDDEN..(id + 1) * TALKER_HIDDEN].to_vec()
    }

    /// Build the x-vector common prompt header.
    ///
    /// The composition — role prefix, then `[think, think_bos, language, think_eos, speaker,
    /// codec_pad, codec_bos]` — was solved against the ft7 oracle's captured `inputs_embeds` and is
    /// re-verified by the prompt-header derivation test. `speaker` is the 1,024-wide x-vector from
    /// `speaker_encoder.fc`; there is no default, because a fabricated speaker vector would
    /// produce confident nonsense.
    ///
    /// # Errors
    ///
    /// If `speaker` is not [`TALKER_HIDDEN`] floats wide.
    pub fn xvector_header(
        &self,
        table: &TextEmbeddingTable,
        speaker: &[f32],
        language_id: usize,
    ) -> Result<PromptHeader, CheckpointError> {
        if speaker.len() != TALKER_HIDDEN {
            return Err(CheckpointError::SpeakerWidth {
                expected: TALKER_HIDDEN,
                actual: speaker.len(),
            });
        }
        Ok(PromptHeader {
            role: ROLE_PREFIX_IDS
                .iter()
                .map(|id| self.project_text_id(table, *id))
                .collect(),
            codec_prefill: vec![
                self.codec_row(CODEC_THINK_ID),
                self.codec_row(CODEC_THINK_BOS_ID),
                self.codec_row(language_id),
                self.codec_row(CODEC_THINK_EOS_ID),
                speaker.to_vec(),
                self.codec_row(CODEC_PAD_ID),
                self.codec_row(CODEC_BOS_ID),
            ],
            tts_bos: self.project_text_id(table, TTS_BOS_TOKEN_ID),
            tts_pad: self.project_text_id(table, TTS_PAD_TOKEN_ID),
        })
    }

    /// The `tts_eos` hidden state that trails the target text.
    #[must_use]
    pub fn tts_eos(&self, table: &TextEmbeddingTable) -> HiddenState {
        self.project_text_id(table, TTS_EOS_TOKEN_ID)
    }
}

/// Owned weight + bias pair for one convolution.
struct OwnedConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl OwnedConv {
    fn load(file: &SafetensorsFile, path: &Path, prefix: &str) -> Result<Self, CheckpointError> {
        Ok(Self {
            weight: widen(file, path, &format!("{prefix}.weight"))?,
            bias: widen(file, path, &format!("{prefix}.bias"))?,
        })
    }
}

struct OwnedCodecLayer {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    self_attn_layer_scale: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    mlp_layer_scale: Vec<f32>,
}

impl OwnedCodecLayer {
    fn load(file: &SafetensorsFile, path: &Path, index: usize) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.pre_transformer.layers.{index}");
        Ok(Self {
            input_layernorm: widen(file, path, &format!("{prefix}.input_layernorm.weight"))?,
            q_proj: widen(file, path, &format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: widen(file, path, &format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: widen(file, path, &format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: widen(file, path, &format!("{prefix}.self_attn.o_proj.weight"))?,
            self_attn_layer_scale: widen(
                file,
                path,
                &format!("{prefix}.self_attn_layer_scale.scale"),
            )?,
            post_attention_layernorm: widen(
                file,
                path,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            gate_proj: widen(file, path, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: widen(file, path, &format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: widen(file, path, &format!("{prefix}.mlp.down_proj.weight"))?,
            mlp_layer_scale: widen(file, path, &format!("{prefix}.mlp_layer_scale.scale"))?,
        })
    }

    fn borrow(&self) -> CodecTransformerLayerWeights<'_> {
        CodecTransformerLayerWeights {
            input_layernorm: &self.input_layernorm,
            q_proj: &self.q_proj,
            k_proj: &self.k_proj,
            v_proj: &self.v_proj,
            o_proj: &self.o_proj,
            self_attn_layer_scale: &self.self_attn_layer_scale,
            post_attention_layernorm: &self.post_attention_layernorm,
            gate_proj: &self.gate_proj,
            up_proj: &self.up_proj,
            down_proj: &self.down_proj,
            mlp_layer_scale: &self.mlp_layer_scale,
        }
    }
}

struct OwnedResidualUnit {
    first_alpha: Vec<f32>,
    first_beta: Vec<f32>,
    first_conv: OwnedConv,
    second_alpha: Vec<f32>,
    second_beta: Vec<f32>,
    second_conv: OwnedConv,
}

impl OwnedResidualUnit {
    fn load(
        file: &SafetensorsFile,
        path: &Path,
        block: usize,
        unit: usize,
    ) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.decoder.{block}.block.{unit}");
        Ok(Self {
            first_alpha: widen(file, path, &format!("{prefix}.act1.alpha"))?,
            first_beta: widen(file, path, &format!("{prefix}.act1.beta"))?,
            first_conv: OwnedConv::load(file, path, &format!("{prefix}.conv1.conv"))?,
            second_alpha: widen(file, path, &format!("{prefix}.act2.alpha"))?,
            second_beta: widen(file, path, &format!("{prefix}.act2.beta"))?,
            second_conv: OwnedConv::load(file, path, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn borrow(&self, channels: usize, dilation: usize) -> CodecResidualUnitWeights<'_> {
        CodecResidualUnitWeights {
            first_alpha_log: &self.first_alpha,
            first_beta_log: &self.first_beta,
            first_conv: CausalConvWeights {
                weight: &self.first_conv.weight,
                bias: &self.first_conv.bias,
                input_channels: channels,
                output_channels: channels,
                kernel: 7,
                dilation,
            },
            second_alpha_log: &self.second_alpha,
            second_beta_log: &self.second_beta,
            second_conv: CausalConvWeights {
                weight: &self.second_conv.weight,
                bias: &self.second_conv.bias,
                input_channels: channels,
                output_channels: channels,
                kernel: 1,
                dilation: 1,
            },
        }
    }
}

struct OwnedBlock {
    alpha: Vec<f32>,
    beta: Vec<f32>,
    transposed: OwnedConv,
    units: [OwnedResidualUnit; 3],
    input_channels: usize,
    output_channels: usize,
    kernel: usize,
    stride: usize,
}

impl OwnedBlock {
    fn load(
        file: &SafetensorsFile,
        path: &Path,
        block: usize,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.decoder.{block}");
        Ok(Self {
            alpha: widen(file, path, &format!("{prefix}.block.0.alpha"))?,
            beta: widen(file, path, &format!("{prefix}.block.0.beta"))?,
            transposed: OwnedConv::load(file, path, &format!("{prefix}.block.1.conv"))?,
            units: [
                OwnedResidualUnit::load(file, path, block, 2)?,
                OwnedResidualUnit::load(file, path, block, 3)?,
                OwnedResidualUnit::load(file, path, block, 4)?,
            ],
            input_channels,
            output_channels,
            kernel,
            stride,
        })
    }

    fn borrow(&self) -> CodecDecoderBlockWeights<'_> {
        CodecDecoderBlockWeights {
            alpha_log: &self.alpha,
            beta_log: &self.beta,
            transposed: CausalTransposeConvWeights {
                weight: &self.transposed.weight,
                bias: &self.transposed.bias,
                input_channels: self.input_channels,
                output_channels: self.output_channels,
                kernel: self.kernel,
                stride: self.stride,
            },
            residual_units: [
                self.units[0].borrow(self.output_channels, 1),
                self.units[1].borrow(self.output_channels, 3),
                self.units[2].borrow(self.output_channels, 9),
            ],
        }
    }
}

struct OwnedUpsampleStage {
    transposed: OwnedConv,
    dwconv: OwnedConv,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    pwconv1: OwnedConv,
    pwconv2: OwnedConv,
    gamma: Vec<f32>,
}

impl OwnedUpsampleStage {
    fn load(file: &SafetensorsFile, path: &Path, stage: usize) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.upsample.{stage}");
        Ok(Self {
            transposed: OwnedConv::load(file, path, &format!("{prefix}.0.conv"))?,
            dwconv: OwnedConv::load(file, path, &format!("{prefix}.1.dwconv.conv"))?,
            norm_weight: widen(file, path, &format!("{prefix}.1.norm.weight"))?,
            norm_bias: widen(file, path, &format!("{prefix}.1.norm.bias"))?,
            pwconv1: OwnedConv::load(file, path, &format!("{prefix}.1.pwconv1"))?,
            pwconv2: OwnedConv::load(file, path, &format!("{prefix}.1.pwconv2"))?,
            gamma: widen(file, path, &format!("{prefix}.1.gamma"))?,
        })
    }

    fn borrow_transposed(&self) -> CausalTransposeConvWeights<'_> {
        CausalTransposeConvWeights {
            weight: &self.transposed.weight,
            bias: &self.transposed.bias,
            input_channels: 1_024,
            output_channels: 1_024,
            kernel: 2,
            stride: 2,
        }
    }

    fn borrow_convnext(&self) -> CodecConvNextWeights<'_> {
        CodecConvNextWeights {
            depthwise_weight: &self.dwconv.weight,
            depthwise_bias: &self.dwconv.bias,
            norm_weight: &self.norm_weight,
            norm_bias: &self.norm_bias,
            pwconv1: &self.pwconv1.weight,
            pwconv1_bias: &self.pwconv1.bias,
            pwconv2: &self.pwconv2.weight,
            pwconv2_bias: &self.pwconv2.bias,
            gamma: &self.gamma,
        }
    }
}

/// The codec decoder from `speech_tokenizer/model.safetensors`.
pub struct CodecCheckpoint {
    config: CodecConfig,
    first_codebook: MaterializedCodebook,
    rest_codebooks: Vec<MaterializedCodebook>,
    first_output_proj: Vec<f32>,
    rest_output_proj: Vec<f32>,
    pre_conv: OwnedConv,
    input_proj: OwnedConv,
    layers: Vec<OwnedCodecLayer>,
    final_norm: Vec<f32>,
    output_proj: OwnedConv,
    upsample: [OwnedUpsampleStage; 2],
    decoder_input: OwnedConv,
    blocks: [OwnedBlock; 4],
    final_alpha: Vec<f32>,
    final_beta: Vec<f32>,
    final_conv: OwnedConv,
}

impl CodecCheckpoint {
    /// Hydrate the codec decoder.
    ///
    /// # Errors
    ///
    /// If the file cannot be opened, a tensor is missing, or a codebook refuses to materialize.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = open(path)?;
        let config = CodecConfig::default();
        let materialize = |prefix: &str| -> Result<MaterializedCodebook, CheckpointError> {
            let sum = widen(&file, path, &format!("{prefix}.embedding_sum"))?;
            let usage = widen(&file, path, &format!("{prefix}.cluster_usage"))?;
            MaterializedCodebook::from_unnormalized(
                &sum,
                &usage,
                config.codebook_size,
                config.codebook_dim,
            )
            .map_err(CheckpointError::Codec)
        };
        let mut rest_codebooks = Vec::with_capacity(CODE_GROUP_COUNT - 1);
        for layer in 0..CODE_GROUP_COUNT - 1 {
            rest_codebooks.push(materialize(&format!(
                "decoder.quantizer.rvq_rest.vq.layers.{layer}._codebook"
            ))?);
        }
        let mut layers = Vec::with_capacity(8);
        for index in 0..8 {
            layers.push(OwnedCodecLayer::load(&file, path, index)?);
        }

        Ok(Self {
            config,
            first_codebook: materialize("decoder.quantizer.rvq_first.vq.layers.0._codebook")?,
            rest_codebooks,
            first_output_proj: widen(
                &file,
                path,
                "decoder.quantizer.rvq_first.output_proj.weight",
            )?,
            rest_output_proj: widen(&file, path, "decoder.quantizer.rvq_rest.output_proj.weight")?,
            pre_conv: OwnedConv::load(&file, path, "decoder.pre_conv.conv")?,
            input_proj: OwnedConv::load(&file, path, "decoder.pre_transformer.input_proj")?,
            layers,
            final_norm: widen(&file, path, "decoder.pre_transformer.norm.weight")?,
            output_proj: OwnedConv::load(&file, path, "decoder.pre_transformer.output_proj")?,
            upsample: [
                OwnedUpsampleStage::load(&file, path, 0)?,
                OwnedUpsampleStage::load(&file, path, 1)?,
            ],
            decoder_input: OwnedConv::load(&file, path, "decoder.decoder.0.conv")?,
            blocks: [
                OwnedBlock::load(&file, path, 1, 1_536, 768, 16, 8)?,
                OwnedBlock::load(&file, path, 2, 768, 384, 10, 5)?,
                OwnedBlock::load(&file, path, 3, 384, 192, 8, 4)?,
                OwnedBlock::load(&file, path, 4, 192, 96, 6, 3)?,
            ],
            final_alpha: widen(&file, path, "decoder.decoder.5.alpha")?,
            final_beta: widen(&file, path, "decoder.decoder.5.beta")?,
            final_conv: OwnedConv::load(&file, path, "decoder.decoder.6.conv")?,
        })
    }

    fn quantizer(&self) -> SplitResidualVectorQuantizer<'_> {
        SplitResidualVectorQuantizer {
            first_codebook: &self.first_codebook,
            rest_codebooks: &self.rest_codebooks,
            first_output_proj: &self.first_output_proj,
            rest_output_proj: &self.rest_output_proj,
        }
    }

    fn conv(conv: &OwnedConv, input: usize, output: usize, kernel: usize) -> CausalConvWeights<'_> {
        CausalConvWeights {
            weight: &conv.weight,
            bias: &conv.bias,
            input_channels: input,
            output_channels: output,
            kernel,
            dilation: 1,
        }
    }

    fn decoder_weights<'a>(
        &'a self,
        layers: &'a [CodecTransformerLayerWeights<'a>],
    ) -> CodecDecoderWeights<'a> {
        CodecDecoderWeights {
            pre_conv: Self::conv(&self.pre_conv, 512, 1_024, 3),
            pre_transformer: CodecPreTransformerWeights {
                input_proj: &self.input_proj.weight,
                input_proj_bias: &self.input_proj.bias,
                layers,
                final_norm: &self.final_norm,
                output_proj: &self.output_proj.weight,
                output_proj_bias: &self.output_proj.bias,
            },
            latent_upsample: [
                CodecUpsampleStageWeights {
                    transposed: self.upsample[0].borrow_transposed(),
                    convnext: self.upsample[0].borrow_convnext(),
                },
                CodecUpsampleStageWeights {
                    transposed: self.upsample[1].borrow_transposed(),
                    convnext: self.upsample[1].borrow_convnext(),
                },
            ],
            decoder_input: Self::conv(&self.decoder_input, 1_024, 1_536, 7),
            decoder_blocks: [
                self.blocks[0].borrow(),
                self.blocks[1].borrow(),
                self.blocks[2].borrow(),
                self.blocks[3].borrow(),
            ],
            final_alpha_log: &self.final_alpha,
            final_beta_log: &self.final_beta,
            final_conv: Self::conv(&self.final_conv, 96, 1, 7),
        }
    }

    /// A fresh streaming decoder state for frame-by-frame decode.
    ///
    /// Streamed output is bit-identical to [`Self::decode`] of the same code stream under every
    /// packet schedule — the standing streaming==offline gate — which is what lets a caller
    /// overlap codec decode with generation.
    #[must_use]
    pub fn stream_state(&self) -> crate::codec::CodecStreamingState {
        let layers: Vec<CodecTransformerLayerWeights<'_>> =
            self.layers.iter().map(OwnedCodecLayer::borrow).collect();
        let weights = self.decoder_weights(&layers);
        crate::codec::CodecStreamingState::new(self.config, &weights)
    }

    /// Decodes one nonempty packet of `frames` code rows, appending finalized PCM to `output`.
    ///
    /// # Errors
    ///
    /// If the packet's code layout is malformed.
    pub fn stream_push(
        &self,
        state: &mut crate::codec::CodecStreamingState,
        codes: &[i32],
        frames: usize,
        output: &mut Vec<f32>,
    ) -> Result<(), CheckpointError> {
        let layers: Vec<CodecTransformerLayerWeights<'_>> =
            self.layers.iter().map(OwnedCodecLayer::borrow).collect();
        let weights = self.decoder_weights(&layers);
        state
            .push(self.quantizer(), &weights, codes, frames, output)
            .map_err(CheckpointError::Codec)
    }

    /// Decode `frames` frame-major 16-code groups to mono 24 kHz `f32` PCM.
    ///
    /// # Errors
    ///
    /// If the codec rejects the codes (wrong length, out-of-range id).
    pub fn decode(&self, codes: &[i32], frames: usize) -> Result<Vec<f32>, CheckpointError> {
        let layers: Vec<CodecTransformerLayerWeights<'_>> =
            self.layers.iter().map(OwnedCodecLayer::borrow).collect();
        let weights = self.decoder_weights(&layers);
        decode_codec_offline(self.config, self.quantizer(), &weights, codes, frames)
            .map_err(CheckpointError::Codec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ftts_artifacts::fttsq::{AccessClass, FttsqWriter, TensorEntry};
    use std::io::Write as _;

    #[test]
    fn runtime_and_converter_q8_quantizers_are_byte_identical_by_construction() {
        // The doctrine's by-construction guarantee: `.fttsq` artifact bytes and runtime W8A8
        // quantization must come from the same arithmetic. The two crates cannot share the
        // function (ftts-artifacts depends on ftts-kernels), so this test pins the restatement.
        let mut state = 0x0dd0_beef_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 29) as f32) - 2.0
        };
        for len in [1_usize, 17, 1024, 3072] {
            let row: Vec<f32> = (0..len).map(|_| next()).collect();
            let mut kernel_bytes = vec![0_i8; len];
            let kernel_scale = ftts_kernels::int8::quantize_row_q8(&row, &mut kernel_bytes);
            let mut converter_bytes = vec![0_i8; len];
            let converter_scale =
                ftts_artifacts::converter::quantize_output_channel_q8(&row, &mut converter_bytes)
                    .expect("finite row");
            assert_eq!(
                kernel_scale.to_bits(),
                converter_scale.to_bits(),
                "len {len}"
            );
            assert_eq!(kernel_bytes, converter_bytes, "len {len}");
        }
        // The all-zero row special case must also agree.
        let zeros = [0.0_f32; 8];
        let mut kernel_bytes = [1_i8; 8];
        let mut converter_bytes = [2_i8; 8];
        let kernel_scale = ftts_kernels::int8::quantize_row_q8(&zeros, &mut kernel_bytes);
        let converter_scale =
            ftts_artifacts::converter::quantize_output_channel_q8(&zeros, &mut converter_bytes)
                .expect("finite row");
        assert_eq!(kernel_scale.to_bits(), converter_scale.to_bits());
        assert_eq!(kernel_bytes, converter_bytes);
    }

    #[test]
    fn the_wrapper_written_here_is_the_wrapper_prompt_strips() {
        // `extract_prompt_text_ids` slices `[3..len-5]`; if the two disagree, the target text is
        // silently truncated or padded with wrapper ids and the model speaks garbage.
        let text = [9707u32, 13];
        let wrapped = TalkerCheckpoint::wrap_target_ids(&text);
        assert_eq!(wrapped.len(), text.len() + 8);
        let extracted = crate::prompt::extract_prompt_text_ids(&wrapped, None)
            .expect("the wrapper we wrote must be the wrapper prompt accepts");
        assert_eq!(extracted.target, text);
    }

    #[test]
    fn the_gathered_id_set_covers_every_position_the_prompt_can_reach() {
        let wrapped = TalkerCheckpoint::wrap_target_ids(&[9707, 13]);
        let ids = TalkerCheckpoint::utterance_text_ids(&wrapped);
        for id in wrapped {
            assert!(ids.contains(&id), "wrapped id {id} must be gathered");
        }
        for special in [TTS_PAD_TOKEN_ID, TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID] {
            assert!(ids.contains(&special), "special {special} must be gathered");
        }
    }

    #[test]
    fn a_speaker_vector_of_the_wrong_width_is_refused() {
        // A short or long x-vector is an enrollment bug; broadcasting or truncating it would make
        // the prompt subtly wrong in a way only listening could catch.
        let error = CheckpointError::SpeakerWidth {
            expected: TALKER_HIDDEN,
            actual: 512,
        };
        assert!(error.to_string().contains("512"));
        assert!(error.to_string().contains("1024"));
    }

    #[test]
    fn q8_artifact_tensor_widens_with_its_per_output_channel_scales() {
        let values = [
            i8::to_ne_bytes(-2)[0],
            i8::to_ne_bytes(1)[0],
            i8::to_ne_bytes(127)[0],
            i8::to_ne_bytes(3)[0],
            i8::to_ne_bytes(-4)[0],
            i8::to_ne_bytes(0)[0],
        ];
        let mut payload = values.to_vec();
        payload.extend_from_slice(&0.25_f32.to_le_bytes());
        payload.extend_from_slice(&0.5_f32.to_le_bytes());
        let tensor = "test.q8.weight";
        let scales = "test.q8.weight.scales";
        let artifact = FttsqWriter::new("test-model", "a".repeat(64))
            .license_notice("Apache-2.0")
            .section(
                "tensor:test.q8.weight",
                AccessClass::HotRecurrentTalker,
                payload,
            )
            .tensor(TensorEntry {
                name: tensor.to_owned(),
                section: "tensor:test.q8.weight".to_owned(),
                dtype: StoredDtype::Q8,
                shape: vec![2, 3],
                offset: 0,
                length: 6,
                scales: Some(scales.to_owned()),
            })
            .tensor(TensorEntry {
                name: scales.to_owned(),
                section: "tensor:test.q8.weight".to_owned(),
                dtype: StoredDtype::F32,
                shape: vec![2],
                offset: 6,
                length: 8,
                scales: None,
            })
            .finish()
            .expect("a valid q8 artifact");

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ftts-checkpoint-q8-{}-{nonce}.fttsq",
            std::process::id()
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .expect("a fresh test artifact path");
        file.write_all(&artifact).expect("write test artifact");
        file.sync_all().expect("sync test artifact");
        drop(file);

        let artifact = MappedFttsq::open(&path).expect("mapped artifact verifies");
        assert_eq!(
            widen_fttsq(&artifact, &path, tensor).expect("q8 tensor widens"),
            vec![-0.5, 0.25, 31.75, 1.5, -2.0, 0.0]
        );
    }
}
