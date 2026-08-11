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
//! therefore materializes ONLY the requested rows, compactly, located by binary search on the
//! sorted gathered-id list. (An earlier revision allocated the full-vocab stride and relied on
//! lazy zero pages; wasm linear memory has no lazy pages, and the 1.24 GB commit was the
//! difference between the browser engine synthesizing and dying at the 4 GB ceiling.) Token ids
//! themselves are never remapped — the wrapper-id checks in [`crate::prompt`] see real ids; only
//! the storage is compact. This is the `ColdTextEmbedding` access class of the plan showing up as
//! a concrete allocation policy, not an optimization to schedule later.
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
    /// A tensor has the right element count but the wrong declared axes.
    ///
    /// This is the silent-transpose defense: a `[in, out]` projection stored where `[out, in]`
    /// is required passes every element-count check and produces confident wrong audio. The
    /// declared shape metadata is the one place the orientation is visible before compute.
    TensorAxes {
        tensor: String,
        expected: String,
        actual: Vec<usize>,
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
            Self::TensorAxes {
                tensor,
                expected,
                actual,
            } => write!(
                formatter,
                "tensor `{tensor}` declares shape {actual:?}, expected {expected}; \
                 a transposed or reshaped export cannot be hydrated"
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
    // One upfront probe instead of a per-element branch: `get_f32` only fails on byte-range
    // overflow, and the range is contiguous, so if the LAST element is readable every element
    // is. Silently hydrating unreadable elements as 0.0 (the old behavior) is exactly the kind
    // of quiet corruption `gather` already refuses loudly.
    if !view.is_empty() && view.get_f32(view.len() - 1).is_none() {
        return Err(CheckpointError::ArtifactTensor {
            path: path.to_path_buf(),
            tensor: name.to_owned(),
            detail: "declared element count exceeds the stored bytes".to_owned(),
        });
    }
    let mut out = vec![0.0f32; view.len()];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = view.get_f32(index).unwrap_or(0.0);
    }
    Ok(out)
}

/// Expected declared axes for one hydrated tensor.
///
/// `Matrix { rows: None, .. }` pins only the input axis — used where the output axis is a
/// vocabulary whose size this crate does not own — which still catches every transpose,
/// because a transposed `[in, out]` store puts the wrong extent in the input position.
#[derive(Clone)]
pub(crate) enum ShapeSpec {
    Vector(usize),
    Matrix { rows: Option<usize>, cols: usize },
}

impl ShapeSpec {
    fn describe(&self) -> String {
        match self {
            Self::Vector(n) => format!("[{n}]"),
            Self::Matrix {
                rows: Some(r),
                cols,
            } => format!("[{r}, {cols}]"),
            Self::Matrix { rows: None, cols } => format!("[_, {cols}]"),
        }
    }

    fn check(&self, tensor: &str, shape: &[usize]) -> Result<(), CheckpointError> {
        let ok = match self {
            Self::Vector(n) => shape == [*n],
            Self::Matrix { rows, cols } => {
                shape.len() == 2
                    && shape[1] == *cols
                    && rows.is_none_or(|expected| shape[0] == expected)
            }
        };
        if ok {
            Ok(())
        } else {
            Err(CheckpointError::TensorAxes {
                tensor: tensor.to_owned(),
                expected: self.describe(),
                actual: shape.to_vec(),
            })
        }
    }
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
    names: &[(String, bool, ShapeSpec)],
    widen_tensor: &(dyn Fn(&str) -> Result<Vec<f32>, CheckpointError> + Sync),
) -> Result<Vec<Vec<f32>>, CheckpointError> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Interleaved A/B under a heavily loaded host measured uncapped hydration workers LOSING ~2x
    // to the serial walk (context-switch thrash against external load); a small fixed team keeps
    // the win on a quiet machine without fighting a busy one. FTTS_LOAD_THREADS overrides; 1 is
    // the serial walk.
    let workers = std::env::var("FTTS_LOAD_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZeroUsize::get)
                .div_euclid(2)
                .min(6)
        })
        .min(names.len())
        .max(1);
    if workers == 1 {
        return names
            .iter()
            .map(|(name, elided, _)| {
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
                    let (name, elided, _) = &names[index];
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
    /// The artifact was deliberately released; cold rows must be supplied by the caller.
    ///
    /// Not an error state. It is the browser's steady state: once the fused int8 tables exist,
    /// nothing reads the artifact's hot bytes again, and holding ~0.69 GB of them for the life of
    /// the page is what put a phone over its limit. Gathering from here fails loudly rather than
    /// returning zeros, because a zero embedding row is a legitimate-looking vector downstream and
    /// would produce fluent wrong audio instead of an error.
    Released,
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

        let mut gathered: Vec<u32> = ids.to_vec();
        gathered.sort_unstable();
        gathered.dedup();
        // Compact storage: one materialized row per gathered id, located by binary search on
        // `gathered`. A full-vocab-stride table would be 1.24 GB of address space per gather —
        // free-ish under lazy native pages, but a hard allocation on wasm linear memory, where
        // it measured as the difference between synthesizing and dying at the 4 GB ceiling.
        let mut rows = vec![0.0f32; gathered.len() * TEXT_EMBED_WIDTH];
        for (slot, id) in gathered.iter().enumerate() {
            let row = *id as usize;
            if row >= TEXT_VOCAB {
                return Err(CheckpointError::TensorShape {
                    tensor: name.to_owned(),
                    expected: TEXT_VOCAB,
                    actual: row + 1,
                });
            }
            let start = slot * TEXT_EMBED_WIDTH;
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

    /// Build the same table from rows the CALLER read, for hosts that do not hold the cold
    /// section at all.
    ///
    /// The browser stages only the artifact's hot prefix and leaves the 622 MB cold table in
    /// OPFS, then reads back the handful of 4 KB rows an utterance needs and hands them here.
    /// The result is indistinguishable from [`TextEmbeddingTable::gather`]: same widening, same
    /// compact layout, same binary-search addressing.
    ///
    /// `rows_bf16` must hold one row per DISTINCT id, ordered by ascending id — the same order
    /// this function derives from `ids` by sorting and deduplicating. That ordering is the whole
    /// contract between the two sides, so it is checked by length here and, more usefully, made
    /// unfalsifiable by having the caller obtain its ids from
    /// [`TalkerCheckpoint::utterance_text_ids`] rather than assembling its own set.
    ///
    /// # Errors
    ///
    /// If any id is outside the vocabulary, or `rows_bf16` is not exactly one bf16 row per
    /// distinct id. A short buffer is refused rather than zero-filled: a zero row is a legitimate
    /// embedding downstream and would yield confident wrong audio instead of an error.
    pub fn from_provided_rows(ids: &[u32], rows_bf16: &[u8]) -> Result<Self, CheckpointError> {
        let name = "talker.model.text_embedding.weight";
        let mut gathered: Vec<u32> = ids.to_vec();
        gathered.sort_unstable();
        gathered.dedup();

        if let Some(over) = gathered.iter().find(|id| **id as usize >= TEXT_VOCAB) {
            return Err(CheckpointError::TensorShape {
                tensor: name.to_owned(),
                expected: TEXT_VOCAB,
                actual: *over as usize + 1,
            });
        }
        let expected_bytes = gathered.len() * TEXT_EMBED_WIDTH * 2;
        if rows_bf16.len() != expected_bytes {
            return Err(CheckpointError::TensorShape {
                tensor: name.to_owned(),
                expected: expected_bytes,
                actual: rows_bf16.len(),
            });
        }

        // bf16 -> f32 is an exact widening: the formats share sign and exponent, so the value is
        // the bf16 bit pattern in the high half of the f32. This is the same conversion the
        // mapped path performs, which is why a provided row is bit-identical to a gathered one.
        let rows = rows_bf16
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
            .collect();
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

        let mut gathered: Vec<u32> = ids.to_vec();
        gathered.sort_unstable();
        gathered.dedup();
        // Compact storage, exactly as the safetensors gather: one row per gathered id.
        let mut rows = vec![0.0f32; gathered.len() * TEXT_EMBED_WIDTH];
        for (slot, id) in gathered.iter().enumerate() {
            let row = *id as usize;
            if row >= TEXT_VOCAB {
                return Err(CheckpointError::TensorShape {
                    tensor: name.to_owned(),
                    expected: TEXT_VOCAB,
                    actual: row + 1,
                });
            }
            let source = &bytes[row * bytes_per_row..(row + 1) * bytes_per_row];
            let destination = &mut rows[slot * TEXT_EMBED_WIDTH..(slot + 1) * TEXT_EMBED_WIDTH];
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

    /// The compact rows, one per gathered id in ascending-id order. Index through
    /// [`TextEmbeddingTable::row`] or the sorted [`TextEmbeddingTable::gathered_ids`], never by
    /// raw token id.
    #[must_use]
    pub fn compact_rows(&self) -> &[f32] {
        &self.rows
    }

    /// One gathered id's embedding row, or `None` when the id was never materialized —
    /// distinguishable, unlike the old full-stride table where an ungathered id read as a
    /// silent zero row.
    #[must_use]
    pub fn row(&self, id: u32) -> Option<&[f32]> {
        let slot = self.gathered.binary_search(&id).ok()?;
        Some(&self.rows[slot * TEXT_EMBED_WIDTH..(slot + 1) * TEXT_EMBED_WIDTH])
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
            |name| file.view(name).map(|view| view.shape().to_vec()),
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
        Self::load_fttsq_mapped(artifact, path, elision)
    }

    /// [`TalkerCheckpoint::load_fttsq_elided`] over an already-verified mapping.
    ///
    /// This is the no-filesystem entry point (wasm hands over a digest-verified in-memory
    /// artifact); `label` stands in for the path in error messages.
    ///
    /// # Errors
    ///
    /// As [`TalkerCheckpoint::load_fttsq`].
    pub fn load_fttsq_mapped(
        artifact: Arc<MappedFttsq>,
        label: &Path,
        elision: HotElision,
    ) -> Result<Self, CheckpointError> {
        let path = label;
        let source = TextEmbeddingSource::Fttsq(Arc::clone(&artifact));
        let elide = {
            let artifact = Arc::clone(&artifact);
            let q8_present = move |name: &str| {
                artifact
                    .reader()
                    .tensor(name)
                    .is_some_and(|entry| entry.dtype == StoredDtype::Q8 && entry.scales.is_some())
            };
            move |name: &str| {
                let stack_elided = if name.starts_with("talker.code_predictor.model.layers.") {
                    elision.micro
                } else if name.starts_with("talker.model.layers.") {
                    elision.talker
                } else {
                    false
                };
                if !stack_elided {
                    return false;
                }
                // The generator's artifact-native hydration is per LAYER: one missing projection
                // sends the whole layer to the requantize fallback, which reads the f32 copies.
                // Elision must therefore be per layer too — a tensor is skipped only when every
                // hot projection of its layer is verifiably Q8 in the artifact, never just itself.
                let Some(suffix) = HOT_PROJECTION_SUFFIXES
                    .iter()
                    .find(|suffix| name.ends_with(*suffix))
                else {
                    return false;
                };
                let layer_prefix = &name[..name.len() - suffix.len()];
                HOT_PROJECTION_SUFFIXES
                    .iter()
                    .all(|suffix| q8_present(&format!("{layer_prefix}{suffix}")))
            }
        };
        let shape_artifact = Arc::clone(&artifact);
        Self::load_with(
            path,
            source,
            |name| widen_fttsq(&artifact, path, name),
            elide,
            move |name| {
                shape_artifact.reader().tensor(name).map(|entry| {
                    entry
                        .shape
                        .iter()
                        .map(|&extent| usize::try_from(extent).unwrap_or(usize::MAX))
                        .collect()
                })
            },
        )
    }

    fn load_with(
        path: &Path,
        text_embedding_source: TextEmbeddingSource,
        widen_tensor: impl Fn(&str) -> Result<Vec<f32>, CheckpointError> + Sync,
        elide_tensor: impl Fn(&str) -> bool,
        tensor_shape: impl Fn(&str) -> Option<Vec<usize>>,
    ) -> Result<Self, CheckpointError> {
        let hidden = TALKER_HIDDEN;
        let micro_config = MicrodecoderConfig::default();
        let micro_layer_count = micro_config.num_layers;
        let talker_config = crate::talker::TalkerConfig::default();

        // Declared-axis expectations for one attention layer, in ATTENTION_LAYER_SUFFIXES
        // order. This is the silent-transpose defense: q/o and gate-up/down are rectangular,
        // so a `[in, out]` export has the same element count and passes every runtime length
        // assert while multiplying garbage.
        let layer_specs = |hidden: usize, q: usize, kv: usize, head: usize, inter: usize| {
            [
                ShapeSpec::Vector(hidden), // input_layernorm
                ShapeSpec::Matrix {
                    rows: Some(q),
                    cols: hidden,
                }, // q_proj
                ShapeSpec::Matrix {
                    rows: Some(kv),
                    cols: hidden,
                }, // k_proj
                ShapeSpec::Matrix {
                    rows: Some(kv),
                    cols: hidden,
                }, // v_proj
                ShapeSpec::Vector(head),   // q_norm
                ShapeSpec::Vector(head),   // k_norm
                ShapeSpec::Matrix {
                    rows: Some(hidden),
                    cols: q,
                }, // o_proj
                ShapeSpec::Vector(hidden), // post_attention_layernorm
                ShapeSpec::Matrix {
                    rows: Some(inter),
                    cols: hidden,
                }, // gate_proj
                ShapeSpec::Matrix {
                    rows: Some(inter),
                    cols: hidden,
                }, // up_proj
                ShapeSpec::Matrix {
                    rows: Some(hidden),
                    cols: inter,
                }, // down_proj
            ]
        };
        let talker_specs = layer_specs(
            talker_config.hidden_size,
            talker_config.num_attention_heads * talker_config.head_dim,
            talker_config.num_key_value_heads * talker_config.head_dim,
            talker_config.head_dim,
            talker_config.intermediate_size,
        );
        let micro_specs = layer_specs(
            micro_config.hidden_size,
            micro_config.num_q_heads * micro_config.head_dim,
            micro_config.num_kv_heads * micro_config.head_dim,
            micro_config.head_dim,
            micro_config.intermediate_size,
        );

        // One flat, canonically ordered name list, hydrated in one concurrent pass; the struct is
        // then assembled by walking the results in the same order. Field-order changes here must
        // keep the two walks in lockstep -- the assertions at each seam catch a drift.
        let mut names: Vec<(String, bool, ShapeSpec)> = Vec::new();
        let mut push = |name: String, spec: ShapeSpec| {
            let elided = elide_tensor(&name);
            names.push((name, elided, spec));
        };
        for layer in 0..TALKER_LAYER_COUNT {
            for (suffix, spec) in ATTENTION_LAYER_SUFFIXES.iter().zip(&talker_specs) {
                push(
                    format!("talker.model.layers.{layer}.{suffix}"),
                    spec.clone(),
                );
            }
        }
        for layer in 0..micro_layer_count {
            for (suffix, spec) in ATTENTION_LAYER_SUFFIXES.iter().zip(&micro_specs) {
                push(
                    format!("talker.code_predictor.model.layers.{layer}.{suffix}"),
                    spec.clone(),
                );
            }
        }
        for table in 0..CODE_GROUP_COUNT - 1 {
            push(
                format!("talker.code_predictor.model.codec_embedding.{table}.weight"),
                ShapeSpec::Matrix {
                    rows: None,
                    cols: hidden,
                },
            );
        }
        for head in 0..RESIDUAL_DEPTHS {
            push(
                format!("talker.code_predictor.lm_head.{head}.weight"),
                ShapeSpec::Matrix {
                    rows: None,
                    cols: hidden,
                },
            );
        }
        for (single, spec) in [
            ("talker.model.norm.weight", ShapeSpec::Vector(hidden)),
            (
                "talker.codec_head.weight",
                ShapeSpec::Matrix {
                    rows: None,
                    cols: hidden,
                },
            ),
            (
                "talker.model.codec_embedding.weight",
                ShapeSpec::Matrix {
                    rows: None,
                    cols: hidden,
                },
            ),
            (
                "talker.code_predictor.model.norm.weight",
                ShapeSpec::Vector(hidden),
            ),
            (
                "talker.text_projection.linear_fc1.weight",
                ShapeSpec::Matrix {
                    rows: Some(TEXT_EMBED_WIDTH),
                    cols: TEXT_EMBED_WIDTH,
                },
            ),
            (
                "talker.text_projection.linear_fc1.bias",
                ShapeSpec::Vector(TEXT_EMBED_WIDTH),
            ),
            (
                "talker.text_projection.linear_fc2.weight",
                ShapeSpec::Matrix {
                    rows: Some(hidden),
                    cols: TEXT_EMBED_WIDTH,
                },
            ),
            (
                "talker.text_projection.linear_fc2.bias",
                ShapeSpec::Vector(hidden),
            ),
        ] {
            push(single.to_owned(), spec);
        }
        // NLL releases `push`'s borrow of `names` here; the closure needs no explicit drop.

        // Axis validation happens BEFORE the (expensive, concurrent) widening pass: a
        // transposed export should refuse in milliseconds, not after gigabytes of hydration.
        // A source that exposes no shape metadata for a tensor skips the check and falls back
        // to the element-count asserts downstream.
        for (name, elided, spec) in &names {
            if *elided {
                continue;
            }
            if let Some(shape) = tensor_shape(name) {
                spec.check(name, &shape)?;
            }
        }

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

    /// The digest-verified canonical artifact this checkpoint hydrated from, when it did.
    ///
    /// Opening a mapping re-verifies the whole artifact's digests (~1.3 GB of hashing), so
    /// consumers that need the mapping — artifact-native int8 hydration, cold-embedding gathers —
    /// must share this one rather than open a second.
    #[must_use]
    pub fn artifact(&self) -> Option<&Arc<MappedFttsq>> {
        match &self.text_embedding_source {
            TextEmbeddingSource::Safetensors | TextEmbeddingSource::Released => None,
            TextEmbeddingSource::Fttsq(artifact) => Some(artifact),
        }
    }

    /// Release the artifact this checkpoint was hydrated from.
    ///
    /// Only safe once every reader of the artifact's bytes has taken what it needs — in practice,
    /// once the fused int8 tables are built ([`crate::generate::prepare_int8_route`]) and the cold
    /// text embedding is being supplied externally ([`TextEmbeddingTable::from_provided_rows`]).
    /// The widened f32 tensors this checkpoint owns were copied at hydration and are unaffected.
    ///
    /// Returns whether the artifact allocation actually went away. It does not when another
    /// `Arc` handle is still outstanding, and that distinction is worth reporting rather than
    /// assuming: a silent failure here looks exactly like the memory saving working, right up
    /// until the device runs out anyway.
    pub fn release_artifact(&mut self) -> bool {
        let previous = std::mem::replace(&mut self.text_embedding_source, TextEmbeddingSource::Released);
        match previous {
            TextEmbeddingSource::Fttsq(artifact) => {
                // One handle in hand means one handle total, so dropping it frees the payload.
                let sole_owner = Arc::strong_count(&artifact) == 1;
                drop(artifact);
                sole_owner
            }
            TextEmbeddingSource::Safetensors => {
                self.text_embedding_source = TextEmbeddingSource::Safetensors;
                false
            }
            TextEmbeddingSource::Released => false,
        }
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
            // Refused, not zero-filled: the caller that released the artifact took on the job of
            // supplying rows, and silently returning an empty table here would surface as fluent
            // wrong audio rather than as the contract violation it is.
            TextEmbeddingSource::Released => Err(CheckpointError::MissingTensor {
                path: self.path.clone(),
                tensor: "talker.model.text_embedding.weight (artifact released; supply rows via \
                         TextEmbeddingTable::from_provided_rows)"
                    .to_owned(),
            }),
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
            table: table.compact_rows(),
            gathered: table.gathered_ids(),
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
        let embed = table
            .row(id)
            .expect("projected ids must be in the gathered set (utterance_text_ids covers them)");

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
/// Where the codec's tensors come from, so they need not all be resident as file bytes first.
///
/// The codec decoder is ~0.457 GB of f32, and building it from a staged safetensors meant holding
/// the source AND the checkpoint at once — 0.92 GB spent to end up with 0.457 GB. On a native host
/// that source is a mapped file and costs nothing. In wasm every staged byte is resident and
/// linear memory only grows, so the doubling was a measured third of a gigabyte of the browser's
/// peak, permanently.
///
/// `take_widened` is a MOVE where the store owns its data: it hands the buffer over and keeps
/// nothing, so the checkpoint holds the only copy that ever existed.
///
/// `&self` rather than `&mut self` because the native loader hydrates pieces on scoped threads;
/// interior mutability keeps that parallel walk intact.
pub trait TensorStore: Sync {
    /// Yield one tensor's f32 values, by move where the store owns them.
    ///
    /// # Errors
    ///
    /// When the tensor is absent — including when it was already taken, which for a moving store
    /// is the same observable failure and equally a caller bug.
    fn take_widened(&self, path: &Path, name: &str) -> Result<Vec<f32>, CheckpointError>;
}

/// A store over a parsed safetensors file: every take widens a fresh copy, exactly as before.
pub struct FileStore<'a>(pub &'a SafetensorsFile);

impl TensorStore for FileStore<'_> {
    fn take_widened(&self, path: &Path, name: &str) -> Result<Vec<f32>, CheckpointError> {
        widen(self.0, path, name)
    }
}

/// Tensors the caller widened already, handed over one at a time.
///
/// The browser fills this as it streams the codec: each tensor is widened on arrival and its
/// incoming bytes released immediately, so the high-water mark is the finished set plus one
/// tensor, rather than the finished set plus the whole file.
pub struct WidenedTensors {
    tensors: std::sync::Mutex<std::collections::HashMap<String, Vec<f32>>>,
}

impl Default for WidenedTensors {
    fn default() -> Self {
        Self::new()
    }
}

impl WidenedTensors {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Widen one tensor's little-endian f32 bytes and keep it under `name`.
    ///
    /// # Errors
    ///
    /// When the byte count is not a whole number of f32 — a truncated push, which must not be
    /// rounded down into a silently short tensor.
    pub fn insert_f32_le(&self, name: &str, bytes: &[u8]) -> Result<(), CheckpointError> {
        if !bytes.len().is_multiple_of(4) {
            return Err(CheckpointError::TensorShape {
                tensor: name.to_owned(),
                expected: bytes.len() / 4 * 4,
                actual: bytes.len(),
            });
        }
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        self.tensors
            .lock()
            .expect("widened tensor map poisoned")
            .insert(name.to_owned(), values);
        Ok(())
    }

    /// How many tensors are held, for the caller's own completeness check.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors
            .lock()
            .expect("widened tensor map poisoned")
            .len()
    }

    /// Whether nothing has been inserted yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl TensorStore for WidenedTensors {
    fn take_widened(&self, path: &Path, name: &str) -> Result<Vec<f32>, CheckpointError> {
        self.tensors
            .lock()
            .expect("widened tensor map poisoned")
            .remove(name)
            .ok_or_else(|| CheckpointError::MissingTensor {
                path: path.to_path_buf(),
                tensor: name.to_owned(),
            })
    }
}

/// Preserves the `widen(file, path, name)` call shape the codec loaders were written against.
fn take(store: &dyn TensorStore, path: &Path, name: &str) -> Result<Vec<f32>, CheckpointError> {
    store.take_widened(path, name)
}

struct OwnedConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl OwnedConv {
    fn load(store: &dyn TensorStore, path: &Path, prefix: &str) -> Result<Self, CheckpointError> {
        Ok(Self {
            weight: take(store, path, &format!("{prefix}.weight"))?,
            bias: take(store, path, &format!("{prefix}.bias"))?,
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
    fn load(store: &dyn TensorStore, path: &Path, index: usize) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.pre_transformer.layers.{index}");
        Ok(Self {
            input_layernorm: take(store, path, &format!("{prefix}.input_layernorm.weight"))?,
            q_proj: take(store, path, &format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: take(store, path, &format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: take(store, path, &format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: take(store, path, &format!("{prefix}.self_attn.o_proj.weight"))?,
            self_attn_layer_scale: take(
                store,
                path,
                &format!("{prefix}.self_attn_layer_scale.scale"),
            )?,
            post_attention_layernorm: take(
                store,
                path,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            gate_proj: take(store, path, &format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: take(store, path, &format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: take(store, path, &format!("{prefix}.mlp.down_proj.weight"))?,
            mlp_layer_scale: take(store, path, &format!("{prefix}.mlp_layer_scale.scale"))?,
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
        store: &dyn TensorStore,
        path: &Path,
        block: usize,
        unit: usize,
    ) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.decoder.{block}.block.{unit}");
        Ok(Self {
            first_alpha: take(store, path, &format!("{prefix}.act1.alpha"))?,
            first_beta: take(store, path, &format!("{prefix}.act1.beta"))?,
            first_conv: OwnedConv::load(store, path, &format!("{prefix}.conv1.conv"))?,
            second_alpha: take(store, path, &format!("{prefix}.act2.alpha"))?,
            second_beta: take(store, path, &format!("{prefix}.act2.beta"))?,
            second_conv: OwnedConv::load(store, path, &format!("{prefix}.conv2.conv"))?,
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
        store: &dyn TensorStore,
        path: &Path,
        block: usize,
        input_channels: usize,
        output_channels: usize,
        kernel: usize,
        stride: usize,
    ) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.decoder.{block}");
        Ok(Self {
            alpha: take(store, path, &format!("{prefix}.block.0.alpha"))?,
            beta: take(store, path, &format!("{prefix}.block.0.beta"))?,
            transposed: OwnedConv::load(store, path, &format!("{prefix}.block.1.conv"))?,
            units: [
                OwnedResidualUnit::load(store, path, block, 2)?,
                OwnedResidualUnit::load(store, path, block, 3)?,
                OwnedResidualUnit::load(store, path, block, 4)?,
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
    fn load(store: &dyn TensorStore, path: &Path, stage: usize) -> Result<Self, CheckpointError> {
        let prefix = format!("decoder.upsample.{stage}");
        Ok(Self {
            transposed: OwnedConv::load(store, path, &format!("{prefix}.0.conv"))?,
            dwconv: OwnedConv::load(store, path, &format!("{prefix}.1.dwconv.conv"))?,
            norm_weight: take(store, path, &format!("{prefix}.1.norm.weight"))?,
            norm_bias: take(store, path, &format!("{prefix}.1.norm.bias"))?,
            pwconv1: OwnedConv::load(store, path, &format!("{prefix}.1.pwconv1"))?,
            pwconv2: OwnedConv::load(store, path, &format!("{prefix}.1.pwconv2"))?,
            gamma: take(store, path, &format!("{prefix}.1.gamma"))?,
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
    /// Hydrate from a parsed safetensors file, as native hosts do.
    ///
    /// # Errors
    ///
    /// As [`CodecCheckpoint::load_from_store`].
    pub fn load_from_file(file: &SafetensorsFile, label: &Path) -> Result<Self, CheckpointError> {
        Self::load_from_store(&FileStore(file), label)
    }

    /// Hydrate from tensors the caller widened and hands over one at a time.
    ///
    /// Hydrate the codec decoder.
    ///
    /// # Errors
    ///
    /// If the store cannot be opened, a tensor is missing, or a codebook refuses to materialize.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let store = open(path)?;
        Self::load_from_file(&store, path)
    }

    /// [`CodecCheckpoint::load`] over an already-parsed checkpoint, for callers with no
    /// filesystem (wasm); `label` stands in for the path in error messages.
    ///
    /// # Errors
    ///
    /// As [`CodecCheckpoint::load`].
    pub fn load_from_store(store: &dyn TensorStore, label: &Path) -> Result<Self, CheckpointError> {
        let path = label;
        let config = CodecConfig::default();
        let materialize = |prefix: &str| -> Result<MaterializedCodebook, CheckpointError> {
            let sum = take(store, path, &format!("{prefix}.embedding_sum"))?;
            let usage = take(store, path, &format!("{prefix}.cluster_usage"))?;
            MaterializedCodebook::from_unnormalized(
                &sum,
                &usage,
                config.codebook_size,
                config.codebook_dim,
            )
            .map_err(CheckpointError::Codec)
        };

        // The pieces are independent, so the heavyweight groups hydrate on scoped threads while
        // the main thread takes the small convs and singles. Each piece's bytes are computed
        // exactly as the serial walk computed them; only wall time changes. wasm32 has no
        // threads, so it takes the same walk serially — same bytes, one lane.
        #[cfg(target_arch = "wasm32")]
        let (codebooks, layers, blocks, upsample) = {
            let codebooks = (|| -> Result<_, CheckpointError> {
                let first = materialize("decoder.quantizer.rvq_first.vq.layers.0._codebook")?;
                let rest = (0..CODE_GROUP_COUNT - 1)
                    .map(|layer| {
                        materialize(&format!(
                            "decoder.quantizer.rvq_rest.vq.layers.{layer}._codebook"
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((first, rest))
            })();
            let layers = (0..8)
                .map(|index| OwnedCodecLayer::load(store, path, index))
                .collect::<Result<Vec<_>, _>>();
            let blocks = [
                OwnedBlock::load(store, path, 1, 1_536, 768, 16, 8),
                OwnedBlock::load(store, path, 2, 768, 384, 10, 5),
                OwnedBlock::load(store, path, 3, 384, 192, 8, 4),
                OwnedBlock::load(store, path, 4, 192, 96, 6, 3),
            ];
            let upsample = (|| -> Result<_, CheckpointError> {
                Ok([
                    OwnedUpsampleStage::load(store, path, 0)?,
                    OwnedUpsampleStage::load(store, path, 1)?,
                ])
            })();
            (codebooks, layers, blocks, upsample)
        };
        #[cfg(not(target_arch = "wasm32"))]
        let (codebooks, layers, blocks, upsample) = std::thread::scope(|scope| {
            let codebooks = scope.spawn(|| -> Result<_, CheckpointError> {
                let first = materialize("decoder.quantizer.rvq_first.vq.layers.0._codebook")?;
                let rest = (0..CODE_GROUP_COUNT - 1)
                    .map(|layer| {
                        materialize(&format!(
                            "decoder.quantizer.rvq_rest.vq.layers.{layer}._codebook"
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((first, rest))
            });
            let layers = scope.spawn(|| {
                (0..8)
                    .map(|index| OwnedCodecLayer::load(store, path, index))
                    .collect::<Result<Vec<_>, _>>()
            });
            let block_handles = [
                scope.spawn(|| OwnedBlock::load(store, path, 1, 1_536, 768, 16, 8)),
                scope.spawn(|| OwnedBlock::load(store, path, 2, 768, 384, 10, 5)),
                scope.spawn(|| OwnedBlock::load(store, path, 3, 384, 192, 8, 4)),
                scope.spawn(|| OwnedBlock::load(store, path, 4, 192, 96, 6, 3)),
            ];
            let upsample = scope.spawn(|| -> Result<_, CheckpointError> {
                Ok([
                    OwnedUpsampleStage::load(store, path, 0)?,
                    OwnedUpsampleStage::load(store, path, 1)?,
                ])
            });
            let mut blocks = block_handles
                .into_iter()
                .map(|handle| handle.join().expect("codec block loader panicked"));
            (
                codebooks.join().expect("codebook loader panicked"),
                layers.join().expect("codec layer loader panicked"),
                [
                    blocks.next().expect("four blocks"),
                    blocks.next().expect("four blocks"),
                    blocks.next().expect("four blocks"),
                    blocks.next().expect("four blocks"),
                ],
                upsample.join().expect("upsample loader panicked"),
            )
        });
        let (first_codebook, rest_codebooks) = codebooks?;
        let [block1, block2, block3, block4] = blocks;

        Ok(Self {
            config,
            first_codebook,
            rest_codebooks,
            first_output_proj: take(store, path, "decoder.quantizer.rvq_first.output_proj.weight")?,
            rest_output_proj: take(store, path, "decoder.quantizer.rvq_rest.output_proj.weight")?,
            pre_conv: OwnedConv::load(store, path, "decoder.pre_conv.conv")?,
            input_proj: OwnedConv::load(store, path, "decoder.pre_transformer.input_proj")?,
            layers: layers?,
            final_norm: take(store, path, "decoder.pre_transformer.norm.weight")?,
            output_proj: OwnedConv::load(store, path, "decoder.pre_transformer.output_proj")?,
            upsample: upsample?,
            decoder_input: OwnedConv::load(store, path, "decoder.decoder.0.conv")?,
            blocks: [block1?, block2?, block3?, block4?],
            final_alpha: take(store, path, "decoder.decoder.5.alpha")?,
            final_beta: take(store, path, "decoder.decoder.5.beta")?,
            final_conv: OwnedConv::load(store, path, "decoder.decoder.6.conv")?,
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

    /// A synthetic artifact shaped like the real one: a hot section, then a cold bf16 table LAST.
    ///
    /// Deliberately narrow in rows and full-width in columns. Row width has to be
    /// `TEXT_EMBED_WIDTH` because that is the stride both readers assume, but the row COUNT is
    /// free — a real-vocabulary table would be 622 MB, which is exactly the thing under test and
    /// not something to allocate to test it.
    fn artifact_with_cold_table(rows: usize) -> (Vec<u8>, Vec<u8>, u64) {
        let cold: Vec<u8> = (0..rows * TEXT_EMBED_WIDTH)
            .flat_map(|i| {
                // Distinct, non-zero, and different in both bytes, so a stride error or a
                // byte-order slip cannot coincidentally still compare equal.
                u16::try_from((i * 7 + 1) % 30_000).unwrap_or(1).to_le_bytes()
            })
            .collect();
        let hot = vec![1_u8, 2, 3, 4, 5, 6, 7, 8];
        let artifact = FttsqWriter::new("cold-elision-test", "b".repeat(64))
            .license_notice("Apache-2.0")
            .section("hot", AccessClass::HotRecurrentTalker, hot)
            .section("cold", AccessClass::ColdTextEmbedding, cold.clone())
            .tensor(TensorEntry {
                name: "talker.model.text_embedding.weight".to_owned(),
                section: "cold".to_owned(),
                dtype: StoredDtype::Bf16,
                shape: vec![rows as u64, TEXT_EMBED_WIDTH as u64],
                offset: 0,
                length: (rows * TEXT_EMBED_WIDTH * 2) as u64,
                scales: None,
            })
            .finish()
            .expect("a valid artifact");
        let cold_offset = {
            let reader =
                ftts_artifacts::fttsq::FttsqReader::parse_directory(&artifact).expect("directory");
            reader
                .prefix_len_omitting(&[AccessClass::ColdTextEmbedding])
                .expect("the cold section is last, so a hot prefix exists")
        };
        (artifact, cold, cold_offset)
    }

    #[test]
    fn a_hot_prefix_verifies_what_it_holds_and_names_what_it_does_not() {
        let (artifact, cold, cold_offset) = artifact_with_cold_table(4);
        assert_eq!(
            cold_offset as usize,
            artifact.len() - cold.len(),
            "the cold section must be the file's tail, or truncating it would drop hot bytes too"
        );

        let reader =
            ftts_artifacts::fttsq::FttsqReader::parse_directory(&artifact).expect("directory");
        assert_eq!(
            reader
                .absent_sections_in_prefix(cold_offset)
                .expect("a clean cut classifies"),
            ["cold"]
        );
        // Every retained section verifies in full against its own digest, exactly as it would in
        // a whole-file load — a prefix weakens coverage, never the strength of what it covers.
        reader
            .verify_digests_of_present(&artifact[..cold_offset as usize])
            .expect("the hot section's digest still holds in the prefix");
        // While the whole-file check must NOT pass, or truncation would go unnoticed.
        assert!(
            reader
                .verify_digests(&artifact[..cold_offset as usize])
                .is_err(),
            "the full digest pass must still notice the missing section"
        );
    }

    #[test]
    fn a_prefix_that_splits_a_section_is_refused_rather_than_half_trusted() {
        let (artifact, _cold, cold_offset) = artifact_with_cold_table(4);
        let reader =
            ftts_artifacts::fttsq::FttsqReader::parse_directory(&artifact).expect("directory");
        // One byte into the cold section: verifiable as neither present nor absent.
        assert!(
            reader.absent_sections_in_prefix(cold_offset + 1).is_err(),
            "a partial section cannot be digest-checked, so it must not load"
        );
        // And a cut inside the HOT section is equally refused.
        assert!(
            reader.absent_sections_in_prefix(cold_offset - 1).is_err(),
            "truncating a retained section must not be mistaken for eliding it"
        );
    }

    #[test]
    fn the_hot_prefix_length_is_derived_from_the_layout_not_assumed() {
        let (artifact, cold, cold_offset) = artifact_with_cold_table(4);
        assert_eq!(cold_offset as usize, artifact.len() - cold.len());

        // A layout with a retained section AFTER the cold one has no hot prefix at all, and must
        // say so rather than silently truncating the retained section away.
        let awkward = FttsqWriter::new("cold-not-last", "c".repeat(64))
            .license_notice("Apache-2.0")
            .section("cold", AccessClass::ColdTextEmbedding, vec![0_u8; 16])
            .section("hot", AccessClass::HotRecurrentTalker, vec![1_u8; 16])
            .finish()
            .expect("a valid artifact");
        let reader =
            ftts_artifacts::fttsq::FttsqReader::parse_directory(&awkward).expect("directory");
        assert!(
            reader
                .prefix_len_omitting(&[AccessClass::ColdTextEmbedding])
                .is_none(),
            "eliding a section that is not last would drop a retained one"
        );
    }

    #[test]
    fn provided_rows_reconstruct_exactly_what_the_mapped_table_would_have_gathered() {
        let (_artifact, cold, _) = artifact_with_cold_table(4);

        // Ids are deliberately unsorted, duplicated, and far apart, because the wire contract is
        // "ascending distinct" and the caller is not asked to know that. Rows are supplied in the
        // order the ids sort into, which is what `text_row_ids` hands the browser.
        let ids = [900_u32, 12, 900, 4, 77];
        let mut distinct = ids.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct, [4, 12, 77, 900]);

        let row_bytes = TEXT_EMBED_WIDTH * 2;
        let provided: Vec<u8> = (0..distinct.len())
            .flat_map(|slot| cold[slot * row_bytes..(slot + 1) * row_bytes].to_vec())
            .collect();
        let table =
            TextEmbeddingTable::from_provided_rows(&ids, &provided).expect("well-formed rows");

        assert_eq!(table.gathered_ids(), distinct.as_slice());
        for (slot, id) in distinct.iter().enumerate() {
            let row = table.row(*id).expect("every supplied id resolves");
            let expected: Vec<f32> = cold[slot * row_bytes..(slot + 1) * row_bytes]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| f32::from_bits(u32::from(u16::from_le_bytes(*p)) << 16))
                .collect();
            assert_eq!(
                row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row for id {id} must widen bit-for-bit"
            );
        }
    }

    #[test]
    fn provided_rows_refuse_a_buffer_that_does_not_match_the_ids() {
        let width = TEXT_EMBED_WIDTH * 2;
        let two_rows = vec![0_u8; width * 2];
        // Three distinct ids, two rows: silently zero-filling the third would be a legitimate-
        // looking embedding and confident wrong audio.
        assert!(TextEmbeddingTable::from_provided_rows(&[1, 2, 3], &two_rows).is_err());
        // Exactly right is accepted, so the check above is about the mismatch and not the shape.
        assert!(TextEmbeddingTable::from_provided_rows(&[1, 2], &two_rows).is_ok());
        // An out-of-vocabulary id is refused before any indexing happens.
        assert!(
            TextEmbeddingTable::from_provided_rows(&[1, TEXT_VOCAB as u32], &two_rows).is_err()
        );
    }

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
    fn a_transposed_projection_is_refused_by_its_declared_axes() {
        // The q_proj geometry: [2048, 1024]. Its transpose has the identical element count,
        // which is exactly why element-count checks alone shipped this hole.
        let q_proj = ShapeSpec::Matrix {
            rows: Some(2048),
            cols: 1024,
        };
        assert!(q_proj.check("q", &[2048, 1024]).is_ok());
        let error = q_proj
            .check("q", &[1024, 2048])
            .expect_err("a transposed store must refuse");
        assert!(error.to_string().contains("transposed"), "{error}");

        // Vocabulary-open matrices (unknown row count) still catch the transpose through the
        // input axis.
        let head = ShapeSpec::Matrix {
            rows: None,
            cols: 1024,
        };
        assert!(head.check("head", &[2048, 1024]).is_ok());
        assert!(head.check("head", &[1024, 2048]).is_err());

        // Norm weights must be genuinely 1-D; a [1, n] reshape is refused.
        let norm = ShapeSpec::Vector(1024);
        assert!(norm.check("norm", &[1024]).is_ok());
        assert!(norm.check("norm", &[1, 1024]).is_err());
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
