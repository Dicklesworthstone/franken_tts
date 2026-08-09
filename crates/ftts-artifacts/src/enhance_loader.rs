//! Hydrates the FastEnhancer denoiser ([`ftts_kernels::enhance::Enhancer`]) from a
//! safetensors artifact holding the inference-form weights.
//!
//! The artifact is `fastenhancer_s_48k_inference.safetensors`: the pinned upstream
//! checkpoint after the reference's own `remove_weight_reparameterizations()` fold,
//! re-serialized as F32 safetensors (see `docs/DENOISER.md` for the pin and recipe).

use std::collections::BTreeMap;

use ftts_kernels::enhance::{EnhanceError, Enhancer};

use crate::safetensors::SafetensorsFile;

/// Why hydration failed.
#[derive(Debug)]
pub enum EnhancerLoadError {
    /// The file could not be opened or its directory parsed.
    Open(crate::safetensors::OpenError),
    /// The tensor set does not describe the expected model geometry.
    Model(EnhanceError),
}

impl std::fmt::Display for EnhancerLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(error) => write!(f, "cannot open denoiser artifact: {error:?}"),
            Self::Model(error) => write!(f, "denoiser artifact is malformed: {error}"),
        }
    }
}

impl std::error::Error for EnhancerLoadError {}

/// Materialize every tensor to `f32` and build the engine.
///
/// The whole artifact is ~830 KB, so whole-tensor materialization is the right shape here —
/// no cold-row machinery.
pub fn enhancer_from_safetensors(file: &SafetensorsFile) -> Result<Enhancer, EnhancerLoadError> {
    let mut tensors = BTreeMap::new();
    for entry in file.index().entries() {
        let view = file
            .view(&entry.name)
            .expect("index entries always resolve against their own file");
        let mut data = vec![0.0f32; view.len()];
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = view
                .get_f32(i)
                .expect("index bounds were validated at parse");
        }
        tensors.insert(entry.name.clone(), (entry.shape.clone(), data));
    }
    Enhancer::load(tensors).map_err(EnhancerLoadError::Model)
}

/// Open a denoiser artifact from disk and build the engine.
pub fn open_enhancer(path: impl AsRef<std::path::Path>) -> Result<Enhancer, EnhancerLoadError> {
    let file = SafetensorsFile::open(path).map_err(EnhancerLoadError::Open)?;
    enhancer_from_safetensors(&file)
}
