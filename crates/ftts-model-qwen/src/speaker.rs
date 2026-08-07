//! The ECAPA-TDNN speaker encoder: 128-mel frames in, a 1024-d x-vector out.
//!
//! This is the whole "quick clone" path's arithmetic. A reference waveform becomes a 128-bin log
//! mel spectrogram (`docs/QWEN3_TTS_SPEAKER_ENCODER_SPEC.md` §1, a separate bead), that becomes an
//! x-vector here, and the x-vector is injected into the talker as **one raw token position with no
//! projection and no normalization** (spec §3). That last fact is what makes this module worth
//! gating block by block: there is no norm anywhere downstream to absorb drift, so whatever error
//! this stack introduces lands in the talker's input sequence at full scale.
//!
//! # Geometry
//!
//! Every hyperparameter is a `Qwen3TTSSpeakerEncoderConfig` class default except `enc_dim`, which
//! `hf/config.json` overrides to 1024. All of them are confirmed against the checkpoint's tensor
//! shapes (spec §2), so a wrong constant here fails to load rather than silently mis-computing.
//!
//! ```text
//! mel (T,128)
//!   blocks[0]  TDNN(128->512, k=5, d=1)         -> h0
//!   blocks[1]  SE-Res2Net(512->512, k=3, d=2)   -> h1
//!   blocks[2]  SE-Res2Net(512->512, k=3, d=3)   -> h2
//!   blocks[3]  SE-Res2Net(512->512, k=3, d=4)   -> h3
//!   MFA        cat(h1,h2,h3) = 1536 -> TDNN(1536->1536, k=1)
//!   ASP        attentive statistics pooling -> 3072 (mean || std)
//!   fc         Conv1d(3072 -> 1024, k=1)  -> 1024-d embedding
//! ```
//!
//! **[TRAP] the MFA concatenation excludes `h0`** — the reference is
//! `torch.cat(hidden_states_list[1:], dim=1)`. Including it would need 2048 input channels and
//! would fail to load; *reordering* the three that are included would load and be wrong, which is
//! why [`Encoder::aggregate`] takes them as three named arguments.
//!
//! # Layout
//!
//! Activations are time-major `[frames, channels]`, matching [`crate::codec`]. The oracle's
//! fixtures are PyTorch-major `(1, channels, frames)`, so the conformance harness transposes at the
//! seam rather than this module carrying two layouts.
//!
//! # Reduction order
//!
//! Every convolution here is an `nn.Conv1d`, so it goes through the same `slow_conv2d` im2col GEMM
//! that [`crate::codec::causal_conv1d`] reproduces bit-exactly: columns ordered `[in_channel,
//! tap]`-major, bias reaching the GEMM as its `beta = 1` accumulator seed. That routing is
//! load-bearing and is used here for the same reason — see the negative evidence recorded above
//! `causal_conv1d`, which measured that adding the bias after a `beta = 0` GEMM is *not* exact.

use crate::checkpoint::{CheckpointError, open, widen_exact};
use ftts_artifacts::safetensors::SafetensorsFile;
use ftts_kernels::f32ref;
use rustfft::{FftPlanner, num_complex::Complex};
use std::fmt;
use std::path::{Path, PathBuf};

/// Mel bins the encoder consumes. `blocks.0.conv.weight` is `[512, 128, 5]`.
pub const MEL_DIM: usize = 128;

/// Width of the emitted x-vector. `fc.weight` is `[1024, 3072, 1]`.
pub const ENC_DIM: usize = 1024;

/// Per-block channel widths, `enc_channels`.
pub const ENC_CHANNELS: [usize; 5] = [512, 512, 512, 512, 1536];

/// Per-block kernel sizes, `enc_kernel_sizes`.
pub const ENC_KERNEL_SIZES: [usize; 5] = [5, 3, 3, 3, 1];

/// Per-block dilations, `enc_dilations`. Not visible in any tensor shape — from `CFGPY` defaults.
pub const ENC_DILATIONS: [usize; 5] = [1, 2, 3, 4, 1];

/// Res2Net cardinality. 512 / 8 = 64 = the width of each `res2net_block.blocks.N` conv.
pub const RES2NET_SCALE: usize = 8;

/// Squeeze-excitation bottleneck, `se_block.conv1` output width.
pub const SE_CHANNELS: usize = 128;

/// Attention bottleneck in the pooling layer, `asp.tdnn` output width.
pub const ATTENTION_CHANNELS: usize = 128;

/// Number of SE-Res2Net blocks, i.e. `enc_channels` minus the initial TDNN and the MFA width.
pub const SE_RES2NET_BLOCKS: usize = 3;

/// The variance clamp inside attentive statistics pooling (`AttentiveStatisticsPooling.eps`).
///
/// Applied to the *variance*, before the square root — `sqrt(sum.clamp(1e-12))`, not
/// `sqrt(sum) + 1e-12`.
pub const ASP_EPS: f32 = 1e-12;

/// The only sample rate accepted by the pinned speaker encoder.
pub const SPEAKER_SAMPLE_RATE_HZ: u32 = 24_000;

const SPEAKER_FFT_SIZE: usize = 1_024;
const SPEAKER_HOP_SAMPLES: usize = 256;
const SPEAKER_REFLECT_PAD_SAMPLES: usize = (SPEAKER_FFT_SIZE - SPEAKER_HOP_SAMPLES) / 2;
const SPEAKER_MIN_AUDIO_SAMPLES: usize = SPEAKER_REFLECT_PAD_SAMPLES + 1;
const SPEAKER_MAGNITUDE_EPSILON: f32 = 1e-9;
const SPEAKER_LOG_FLOOR: f32 = 1e-5;

/// Log-mel features in the encoder's time-major `[frames, 128]` layout.
#[derive(Clone, Debug, PartialEq)]
pub struct LogMel {
    /// Number of complete 1,024-sample analysis windows.
    pub frames: usize,
    /// Time-major log-mel values.
    pub values: Vec<f32>,
}

/// Refusals at the waveform-to-mel boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FeatureError {
    /// Torch reflect padding requires an input longer than either padding side.
    TooShort { samples: usize },
    /// A non-finite PCM value would poison an entire FFT window and speaker embedding.
    NonFinite { index: usize },
}

impl fmt::Display for FeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { samples } => write!(
                formatter,
                "reference has {samples} samples; speaker mel extraction needs more than \
                 {SPEAKER_REFLECT_PAD_SAMPLES} samples for the pinned reflect pad"
            ),
            Self::NonFinite { index } => write!(
                formatter,
                "reference PCM has a non-finite sample at index {index}"
            ),
        }
    }
}

impl std::error::Error for FeatureError {}

/// Compute the pinned ECAPA front end from mono 24 kHz PCM.
///
/// This preserves the upstream geometry: a manual 384-sample reflect pad followed by a
/// centered-off 1,024-point periodic-Hann STFT, magnitude with epsilon inside the square root,
/// Slaney-normalized mel filters, and natural-log compression after the `1e-5` floor. It is an
/// enrollment-only path, so it deliberately favors transparent scalar arithmetic over a hot-loop
/// specialization.
///
/// # Errors
///
/// Returns [`FeatureError::TooShort`] when the pinned reflect pad is invalid, and
/// [`FeatureError::NonFinite`] before any FFT work when the waveform is not usable.
pub fn log_mel_from_24khz_pcm(pcm: &[f32]) -> Result<LogMel, FeatureError> {
    if pcm.len() < SPEAKER_MIN_AUDIO_SAMPLES {
        return Err(FeatureError::TooShort { samples: pcm.len() });
    }
    if let Some(index) = pcm.iter().position(|sample| !sample.is_finite()) {
        return Err(FeatureError::NonFinite { index });
    }

    let mut padded = Vec::with_capacity(pcm.len() + 2 * SPEAKER_REFLECT_PAD_SAMPLES);
    for offset in
        -(SPEAKER_REFLECT_PAD_SAMPLES as isize)..(pcm.len() + SPEAKER_REFLECT_PAD_SAMPLES) as isize
    {
        padded.push(pcm[reflect_index(offset, pcm.len() as isize)]);
    }
    let frames = (padded.len() - SPEAKER_FFT_SIZE) / SPEAKER_HOP_SAMPLES + 1;
    let filters = slaney_mel_filterbank();
    let window = periodic_hann();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(SPEAKER_FFT_SIZE);
    let mut spectrum = vec![Complex::new(0.0f32, 0.0); SPEAKER_FFT_SIZE];
    let mut magnitudes = vec![0.0f32; SPEAKER_FFT_SIZE / 2 + 1];
    let mut values = vec![0.0f32; frames * MEL_DIM];

    for frame in 0..frames {
        let start = frame * SPEAKER_HOP_SAMPLES;
        for (index, slot) in spectrum.iter_mut().enumerate() {
            *slot = Complex::new(padded[start + index] * window[index], 0.0);
        }
        fft.process(&mut spectrum);
        for (magnitude, value) in magnitudes.iter_mut().zip(&spectrum) {
            *magnitude = (value.re.mul_add(value.re, value.im * value.im)
                + SPEAKER_MAGNITUDE_EPSILON)
                .sqrt();
        }
        let output = &mut values[frame * MEL_DIM..][..MEL_DIM];
        for (mel, destination) in output.iter_mut().enumerate() {
            let energy = filters[mel]
                .iter()
                .zip(&magnitudes)
                .map(|(&weight, &magnitude)| weight * magnitude)
                .sum::<f32>();
            *destination = energy.max(SPEAKER_LOG_FLOOR).ln();
        }
    }

    Ok(LogMel { frames, values })
}

fn periodic_hann() -> [f32; SPEAKER_FFT_SIZE] {
    std::array::from_fn(|index| {
        0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / SPEAKER_FFT_SIZE as f32).cos()
    })
}

fn slaney_mel_filterbank() -> Vec<Vec<f32>> {
    let min_mel = hz_to_slaney_mel(0.0);
    let max_mel = hz_to_slaney_mel(SPEAKER_SAMPLE_RATE_HZ as f32 / 2.0);
    let mut mel_points = Vec::with_capacity(MEL_DIM + 2);
    for index in 0..MEL_DIM + 2 {
        let fraction = index as f32 / (MEL_DIM + 1) as f32;
        mel_points.push(slaney_mel_to_hz(min_mel + (max_mel - min_mel) * fraction));
    }
    let frequencies: Vec<f32> = (0..=SPEAKER_FFT_SIZE / 2)
        .map(|bin| SPEAKER_SAMPLE_RATE_HZ as f32 * bin as f32 / SPEAKER_FFT_SIZE as f32)
        .collect();
    (0..MEL_DIM)
        .map(|mel| {
            let lower = mel_points[mel];
            let center = mel_points[mel + 1];
            let upper = mel_points[mel + 2];
            let normalization = 2.0 / (upper - lower);
            frequencies
                .iter()
                .map(|&frequency| {
                    let rising = (frequency - lower) / (center - lower);
                    let falling = (upper - frequency) / (upper - center);
                    rising.min(falling).max(0.0) * normalization
                })
                .collect()
        })
        .collect()
}

fn hz_to_slaney_mel(hz: f32) -> f32 {
    const LINEAR_HZ_PER_MEL: f32 = 200.0 / 3.0;
    const LOG_START_HZ: f32 = 1_000.0;
    const LOG_START_MEL: f32 = LOG_START_HZ / LINEAR_HZ_PER_MEL;
    const LOG_STEP: f32 = 0.068_751_78;
    if hz >= LOG_START_HZ {
        LOG_START_MEL + (hz / LOG_START_HZ).ln() / LOG_STEP
    } else {
        hz / LINEAR_HZ_PER_MEL
    }
}

fn slaney_mel_to_hz(mel: f32) -> f32 {
    const LINEAR_HZ_PER_MEL: f32 = 200.0 / 3.0;
    const LOG_START_HZ: f32 = 1_000.0;
    const LOG_START_MEL: f32 = LOG_START_HZ / LINEAR_HZ_PER_MEL;
    const LOG_STEP: f32 = 0.068_751_78;
    if mel >= LOG_START_MEL {
        LOG_START_HZ * (LOG_STEP * (mel - LOG_START_MEL)).exp()
    } else {
        mel * LINEAR_HZ_PER_MEL
    }
}

/// One `nn.Conv1d`: weight in PyTorch's `[out_channels, in_channels, kernel]` order, plus a bias.
///
/// Every convolution in this encoder carries a bias, and every one of them is `padding="same"` with
/// `padding_mode="reflect"`.
#[derive(Clone, Debug)]
struct Conv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
}

impl Conv {
    fn load(
        file: &SafetensorsFile,
        path: &Path,
        base: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self, CheckpointError> {
        Ok(Self {
            weight: widen_exact(
                file,
                path,
                &format!("{base}.weight"),
                out_channels * in_channels * kernel,
            )?,
            bias: widen_exact(file, path, &format!("{base}.bias"), out_channels)?,
            in_channels,
            out_channels,
            kernel,
            dilation,
        })
    }

    /// `padding="same"`, `padding_mode="reflect"`, no activation.
    fn apply(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; frames * self.out_channels];
        same_conv1d(
            input,
            frames,
            self.in_channels,
            &self.weight,
            Some(&self.bias),
            self.out_channels,
            self.kernel,
            self.dilation,
            &mut out,
        );
        out
    }

    /// `TimeDelayNetBlock::forward` — the convolution followed by `nn.ReLU`.
    fn apply_relu(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let mut out = self.apply(input, frames);
        relu_in_place(&mut out);
        out
    }
}

/// `SqueezeExcitationRes2NetBlock`: TDNN -> Res2Net -> TDNN -> SE, with a residual around all four.
#[derive(Clone, Debug)]
struct SeRes2NetBlock {
    tdnn1: Conv,
    res2net: Vec<Conv>,
    tdnn2: Conv,
    se_conv1: Conv,
    se_conv2: Conv,
    channels: usize,
}

impl SeRes2NetBlock {
    fn load(
        file: &SafetensorsFile,
        path: &Path,
        index: usize,
        channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self, CheckpointError> {
        let base = format!("speaker_encoder.blocks.{index}");
        let split = channels / RES2NET_SCALE;
        let mut res2net = Vec::with_capacity(RES2NET_SCALE - 1);
        for branch in 0..RES2NET_SCALE - 1 {
            res2net.push(Conv::load(
                file,
                path,
                &format!("{base}.res2net_block.blocks.{branch}.conv"),
                split,
                split,
                kernel,
                dilation,
            )?);
        }
        Ok(Self {
            tdnn1: Conv::load(
                file,
                path,
                &format!("{base}.tdnn1.conv"),
                channels,
                channels,
                1,
                1,
            )?,
            res2net,
            tdnn2: Conv::load(
                file,
                path,
                &format!("{base}.tdnn2.conv"),
                channels,
                channels,
                1,
                1,
            )?,
            se_conv1: Conv::load(
                file,
                path,
                &format!("{base}.se_block.conv1"),
                channels,
                SE_CHANNELS,
                1,
                1,
            )?,
            se_conv2: Conv::load(
                file,
                path,
                &format!("{base}.se_block.conv2"),
                SE_CHANNELS,
                channels,
                1,
                1,
            )?,
            channels,
        })
    }

    fn forward(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let hidden = self.tdnn1.apply_relu(input, frames);
        let hidden = self.res2net(&hidden, frames);
        let hidden = self.tdnn2.apply_relu(&hidden, frames);
        let mut hidden = self.squeeze_excite(&hidden, frames);
        // `return hidden_state + residual` — the residual is the block's *input*, not tdnn1's.
        for (slot, &residual) in hidden.iter_mut().zip(input.iter()) {
            *slot += residual;
        }
        hidden
    }

    /// `Res2NetBlock::forward` — a chain across `scale` channel chunks.
    ///
    /// Chunk 0 passes through untouched; chunk 1 is convolved alone; every later chunk is convolved
    /// after adding the *previous chunk's output*. The chain is what makes this Res2Net rather than
    /// grouped convolution, and getting the "previous output" wrong (using the previous input, or
    /// resetting the carry) still produces plausibly-scaled activations.
    fn res2net(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let split = self.channels / RES2NET_SCALE;
        let mut out = vec![0.0f32; frames * self.channels];
        let mut carry = vec![0.0f32; frames * split];
        let mut branch_input = vec![0.0f32; frames * split];

        for chunk in 0..RES2NET_SCALE {
            let offset = chunk * split;
            let part = if chunk == 0 {
                // Identity chunk.
                for frame in 0..frames {
                    carry[frame * split..][..split]
                        .copy_from_slice(&input[frame * self.channels + offset..][..split]);
                }
                carry.clone()
            } else {
                for frame in 0..frames {
                    let source = &input[frame * self.channels + offset..][..split];
                    let target = &mut branch_input[frame * split..][..split];
                    if chunk == 1 {
                        target.copy_from_slice(source);
                    } else {
                        let previous = &carry[frame * split..][..split];
                        for ((slot, &value), &prior) in
                            target.iter_mut().zip(source.iter()).zip(previous.iter())
                        {
                            *slot = value + prior;
                        }
                    }
                }
                self.res2net[chunk - 1].apply_relu(&branch_input, frames)
            };
            for frame in 0..frames {
                out[frame * self.channels + offset..][..split]
                    .copy_from_slice(&part[frame * split..][..split]);
            }
            carry.copy_from_slice(&part);
        }
        out
    }

    /// `SqueezeExcitationBlock::forward` — channel means through a 1x1 bottleneck, then a gate.
    fn squeeze_excite(&self, input: &[f32], frames: usize) -> Vec<f32> {
        let mut pooled = vec![0.0f32; self.channels];
        for frame in 0..frames {
            for (slot, &value) in pooled
                .iter_mut()
                .zip(&input[frame * self.channels..][..self.channels])
            {
                *slot += value;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let inverse = 1.0f32 / frames as f32;
        for slot in &mut pooled {
            *slot *= inverse;
        }
        // The bottleneck runs on a length-1 sequence: `mean(dim=2, keepdim=True)`.
        let gate = self.se_conv1.apply_relu(&pooled, 1);
        let mut gate = self.se_conv2.apply(&gate, 1);
        for slot in &mut gate {
            *slot = sigmoid(*slot);
        }
        let mut out = input.to_vec();
        for frame in 0..frames {
            for (slot, &scale) in out[frame * self.channels..][..self.channels]
                .iter_mut()
                .zip(gate.iter())
            {
                *slot *= scale;
            }
        }
        out
    }
}

/// The loaded ECAPA-TDNN speaker encoder.
#[derive(Clone, Debug)]
pub struct Encoder {
    path: PathBuf,
    initial: Conv,
    blocks: Vec<SeRes2NetBlock>,
    mfa: Conv,
    asp_tdnn: Conv,
    asp_conv: Conv,
    fc: Conv,
}

impl Encoder {
    /// Hydrate the 76 `speaker_encoder.*` tensors from a pinned checkpoint.
    ///
    /// These live in the **top-level** `hf/model.safetensors` alongside the talker, not in
    /// `hf/speech_tokenizer/model.safetensors`.
    ///
    /// # Errors
    ///
    /// If the file cannot be opened, or any tensor is missing or not the shape §2 of the spec
    /// requires. Shapes are checked exactly, so a wrong geometry constant fails here rather than
    /// producing a plausible embedding.
    pub fn load(path: &Path) -> Result<Self, CheckpointError> {
        let file = open(path)?;
        let mut blocks = Vec::with_capacity(SE_RES2NET_BLOCKS);
        for index in 1..=SE_RES2NET_BLOCKS {
            blocks.push(SeRes2NetBlock::load(
                &file,
                path,
                index,
                ENC_CHANNELS[index],
                ENC_KERNEL_SIZES[index],
                ENC_DILATIONS[index],
            )?);
        }
        let mfa_width = ENC_CHANNELS[4];
        Ok(Self {
            path: path.to_path_buf(),
            initial: Conv::load(
                &file,
                path,
                "speaker_encoder.blocks.0.conv",
                MEL_DIM,
                ENC_CHANNELS[0],
                ENC_KERNEL_SIZES[0],
                ENC_DILATIONS[0],
            )?,
            blocks,
            mfa: Conv::load(
                &file,
                path,
                "speaker_encoder.mfa.conv",
                mfa_width,
                mfa_width,
                ENC_KERNEL_SIZES[4],
                ENC_DILATIONS[4],
            )?,
            asp_tdnn: Conv::load(
                &file,
                path,
                "speaker_encoder.asp.tdnn.conv",
                mfa_width * 3,
                ATTENTION_CHANNELS,
                1,
                1,
            )?,
            asp_conv: Conv::load(
                &file,
                path,
                "speaker_encoder.asp.conv",
                ATTENTION_CHANNELS,
                mfa_width,
                1,
                1,
            )?,
            fc: Conv::load(
                &file,
                path,
                "speaker_encoder.fc",
                mfa_width * 2,
                ENC_DIM,
                1,
                1,
            )?,
        })
    }

    /// The checkpoint this encoder was read from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `blocks[0]`: the initial TDNN, mel bins to 512 channels.
    ///
    /// `mel` is time-major `[frames, 128]`.
    ///
    /// # Panics
    ///
    /// If `mel` is not `frames * MEL_DIM` long.
    #[must_use]
    pub fn initial_block(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        assert_eq!(mel.len(), frames * MEL_DIM, "mel must be [frames, 128]");
        self.initial.apply_relu(mel, frames)
    }

    /// `blocks[index]` for `index` in `1..=3`: one SE-Res2Net block.
    ///
    /// Taking the input rather than threading it internally is what lets the conformance harness
    /// feed each block the *oracle's own* input, so a block-2 failure means block 2 and not
    /// accumulated drift from block 0.
    ///
    /// # Panics
    ///
    /// If `index` is outside `1..=3`, or `input` is not `frames * 512` long.
    #[must_use]
    pub fn se_res2net_block(&self, index: usize, input: &[f32], frames: usize) -> Vec<f32> {
        assert!(
            (1..=SE_RES2NET_BLOCKS).contains(&index),
            "SE-Res2Net blocks are 1..=3"
        );
        let block = &self.blocks[index - 1];
        assert_eq!(
            input.len(),
            frames * block.channels,
            "block input must be [frames, channels]"
        );
        block.forward(input, frames)
    }

    /// MFA, attentive statistics pooling and the final projection: three block outputs to 1024-d.
    ///
    /// `h0` is deliberately absent — the reference aggregates `hidden_states_list[1:]`, and the
    /// order of these three arguments *is* the concatenation order.
    ///
    /// # Panics
    ///
    /// If any input is not `frames * 512` long, or `frames` is zero.
    #[must_use]
    pub fn aggregate(&self, h1: &[f32], h2: &[f32], h3: &[f32], frames: usize) -> Vec<f32> {
        assert!(frames > 0, "pooling needs at least one frame");
        let width = ENC_CHANNELS[1];
        for hidden in [h1, h2, h3] {
            assert_eq!(
                hidden.len(),
                frames * width,
                "MFA input must be [frames, 512]"
            );
        }
        let aggregate_width = width * 3;
        let mut aggregated = vec![0.0f32; frames * aggregate_width];
        for frame in 0..frames {
            let row = &mut aggregated[frame * aggregate_width..][..aggregate_width];
            row[..width].copy_from_slice(&h1[frame * width..][..width]);
            row[width..width * 2].copy_from_slice(&h2[frame * width..][..width]);
            row[width * 2..].copy_from_slice(&h3[frame * width..][..width]);
        }
        let hidden = self.mfa.apply_relu(&aggregated, frames);
        let pooled = self.attentive_statistics_pooling(&hidden, frames);
        self.fc.apply(&pooled, 1)
    }

    /// Mel frames to a 1024-d x-vector.
    ///
    /// # Panics
    ///
    /// If `mel` is not `frames * MEL_DIM` long, or `frames` is zero.
    #[must_use]
    pub fn encode(&self, mel: &[f32], frames: usize) -> Vec<f32> {
        let h0 = self.initial_block(mel, frames);
        let h1 = self.se_res2net_block(1, &h0, frames);
        let h2 = self.se_res2net_block(2, &h1, frames);
        let h3 = self.se_res2net_block(3, &h2, frames);
        self.aggregate(&h1, &h2, &h3, frames)
    }

    /// `AttentiveStatisticsPooling::forward` for the unmasked (single-utterance) case.
    ///
    /// The reference builds a mask from `lengths = ones * seq_length`, which is all-ones here, so
    /// the `masked_fill` before the softmax is a no-op and `mask / total` is the uniform `1/T`.
    /// That uniform weight is still applied **elementwise before summing**, matching
    /// `(m * x).sum(dim)` rather than `x.sum(dim) / T`.
    ///
    /// Returns `[3072]`: the attention-weighted mean concatenated with the weighted standard
    /// deviation, as one length-1 sequence for `fc`.
    fn attentive_statistics_pooling(&self, hidden: &[f32], frames: usize) -> Vec<f32> {
        let channels = ENC_CHANNELS[4];
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0f32 / frames as f32;

        let mut mean = vec![0.0f32; channels];
        let mut deviation = vec![0.0f32; channels];
        statistics(hidden, frames, channels, &mut mean, &mut deviation, |_| {
            uniform
        });

        // `cat([x, mean, std], dim=1)` with the two statistics broadcast back across time.
        let context_width = channels * 3;
        let mut context = vec![0.0f32; frames * context_width];
        for frame in 0..frames {
            let row = &mut context[frame * context_width..][..context_width];
            row[..channels].copy_from_slice(&hidden[frame * channels..][..channels]);
            row[channels..channels * 2].copy_from_slice(&mean);
            row[channels * 2..].copy_from_slice(&deviation);
        }

        let mut attention = self.asp_tdnn.apply_relu(&context, frames);
        for slot in &mut attention {
            *slot = slot.tanh();
        }
        let mut attention = self.asp_conv.apply(&attention, frames);

        // Softmax over **time**, per channel. The activations are time-major, so this reduction
        // strides; transposing first would be the same arithmetic in a different order and this
        // seam is tight enough that the order is worth keeping explicit.
        softmax_over_time(&mut attention, frames, channels);

        statistics(
            hidden,
            frames,
            channels,
            &mut mean,
            &mut deviation,
            |slot| attention[slot],
        );

        let mut pooled = Vec::with_capacity(channels * 2);
        pooled.extend_from_slice(&mean);
        pooled.extend_from_slice(&deviation);
        pooled
    }
}

/// `_compute_statistics`: a weighted mean and the square root of the clamped weighted variance.
///
/// `weight` is indexed by the flat time-major slot so the caller can pass either the uniform mask
/// or the per-channel attention without materializing the uniform one.
fn statistics(
    hidden: &[f32],
    frames: usize,
    channels: usize,
    mean: &mut [f32],
    deviation: &mut [f32],
    weight: impl Fn(usize) -> f32,
) {
    mean.fill(0.0);
    for frame in 0..frames {
        let base = frame * channels;
        for (channel, slot) in mean.iter_mut().enumerate() {
            *slot += weight(base + channel) * hidden[base + channel];
        }
    }
    deviation.fill(0.0);
    for frame in 0..frames {
        let base = frame * channels;
        for (channel, slot) in deviation.iter_mut().enumerate() {
            let centered = hidden[base + channel] - mean[channel];
            *slot += weight(base + channel) * centered * centered;
        }
    }
    for slot in deviation.iter_mut() {
        *slot = slot.max(ASP_EPS).sqrt();
    }
}

/// Softmax along the time axis of a time-major `[frames, channels]` buffer.
fn softmax_over_time(values: &mut [f32], frames: usize, channels: usize) {
    for channel in 0..channels {
        let mut peak = f32::NEG_INFINITY;
        for frame in 0..frames {
            peak = peak.max(values[frame * channels + channel]);
        }
        let mut total = 0.0f32;
        for frame in 0..frames {
            let slot = &mut values[frame * channels + channel];
            *slot = (*slot - peak).exp();
            total += *slot;
        }
        let inverse = 1.0f32 / total;
        for frame in 0..frames {
            values[frame * channels + channel] *= inverse;
        }
    }
}

fn relu_in_place(values: &mut [f32]) {
    for slot in values {
        *slot = slot.max(0.0);
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

/// Reflect an index into `0..len`, the way `F.pad(mode="reflect")` does — the edge sample is the
/// mirror axis and is **not** repeated.
fn reflect_index(index: isize, len: isize) -> usize {
    if len == 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let mut wrapped = index.rem_euclid(period);
    if wrapped >= len {
        wrapped = period - wrapped;
    }
    #[allow(clippy::cast_sign_loss)]
    {
        wrapped as usize
    }
}

/// Reference `nn.Conv1d(padding="same", padding_mode="reflect")` over time-major data.
///
/// PyTorch splits `padding="same"` as `left = dilation * (kernel - 1) / 2` rounded **down**, right
/// taking the remainder. Every kernel/dilation pair in this encoder yields an even total, so the
/// split is symmetric — but the rounding is written out rather than assumed, because a future
/// even-kernel block would silently shift every frame.
///
/// The im2col unfolding and the bias-seeded GEMM mirror [`crate::codec::causal_conv1d`]; see that
/// function's notes for the measurement that made the seeding load-bearing.
#[allow(clippy::too_many_arguments)]
fn same_conv1d(
    input: &[f32],
    frames: usize,
    input_channels: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_channels: usize,
    kernel: usize,
    dilation: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), frames * input_channels, "conv input shape");
    assert_eq!(
        weight.len(),
        output_channels * input_channels * kernel,
        "conv weight shape"
    );
    assert_eq!(output.len(), frames * output_channels, "conv output shape");
    assert!(kernel > 0 && dilation > 0, "conv geometry must be positive");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), output_channels, "conv bias shape");
    }

    let left_pad = (dilation * (kernel - 1)) / 2;
    let reduction = input_channels * kernel;
    let mut columns = vec![0.0f32; frames * reduction];
    #[allow(clippy::cast_possible_wrap)]
    let length = frames as isize;
    for frame in 0..frames {
        let target = &mut columns[frame * reduction..][..reduction];
        for tap in 0..kernel {
            #[allow(clippy::cast_possible_wrap)]
            let offset = (frame + tap * dilation) as isize - left_pad as isize;
            let source_frame = reflect_index(offset, length);
            let source = &input[source_frame * input_channels..][..input_channels];
            for (input_channel, &value) in source.iter().enumerate() {
                target[input_channel * kernel + tap] = value;
            }
        }
    }
    f32ref::linear_with_accumulation(
        &columns,
        weight,
        bias,
        frames,
        reduction,
        output_channels,
        f32ref::F32LinearAccumulation::AccelerateBiasSeeded,
        output,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_padding_mirrors_without_repeating_the_edge() {
        // F.pad([a,b,c,d], (2,2), mode="reflect") == [c,b,a,b,c,d,c,b]
        let expected = [2, 1, 0, 1, 2, 3, 2, 1];
        let got: Vec<usize> = (-2..6).map(|index| reflect_index(index, 4)).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn log_mel_uses_the_pinned_frame_geometry_and_stays_finite() {
        let pcm: Vec<f32> = (0..SPEAKER_SAMPLE_RATE_HZ as usize)
            .map(|sample| {
                (std::f32::consts::TAU * 440.0 * sample as f32 / SPEAKER_SAMPLE_RATE_HZ as f32)
                    .sin()
                    * 0.25
            })
            .collect();
        let mel = log_mel_from_24khz_pcm(&pcm).expect("24 kHz reference should yield log-mel");
        assert_eq!(mel.frames, 93);
        assert_eq!(mel.values.len(), mel.frames * MEL_DIM);
        assert!(mel.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn log_mel_refuses_invalid_reflect_padding_and_non_finite_pcm() {
        assert_eq!(
            log_mel_from_24khz_pcm(&[0.0; SPEAKER_REFLECT_PAD_SAMPLES]),
            Err(FeatureError::TooShort {
                samples: SPEAKER_REFLECT_PAD_SAMPLES
            })
        );
        let mut pcm = vec![0.0f32; SPEAKER_MIN_AUDIO_SAMPLES];
        pcm[17] = f32::NAN;
        assert_eq!(
            log_mel_from_24khz_pcm(&pcm),
            Err(FeatureError::NonFinite { index: 17 })
        );
    }

    #[test]
    fn same_padding_preserves_length_for_every_block_geometry() {
        let frames = 16;
        for (kernel, dilation) in ENC_KERNEL_SIZES.iter().zip(ENC_DILATIONS.iter()) {
            let input: Vec<f32> = (0..frames).map(|f| f as f32).collect();
            let weight = vec![1.0f32; *kernel];
            let mut output = vec![0.0f32; frames];
            same_conv1d(
                &input,
                frames,
                1,
                &weight,
                None,
                1,
                *kernel,
                *dilation,
                &mut output,
            );
            assert_eq!(output.len(), frames);
            // A same-padded all-ones kernel over a ramp reproduces `kernel * value` in the
            // interior, which pins the tap centering rather than merely the length.
            let centre = frames / 2;
            #[allow(clippy::cast_precision_loss)]
            let expected = *kernel as f32 * centre as f32;
            assert!(
                (output[centre] - expected).abs() < 1e-3,
                "kernel {kernel} dilation {dilation}: {} vs {expected}",
                output[centre]
            );
        }
    }

    #[test]
    fn statistics_are_weighted_before_summing() {
        // Two frames, one channel: values 1 and 3 under a uniform half weight.
        let hidden = [1.0f32, 3.0];
        let mut mean = [0.0f32];
        let mut deviation = [0.0f32];
        statistics(&hidden, 2, 1, &mut mean, &mut deviation, |_| 0.5);
        assert!((mean[0] - 2.0).abs() < 1e-6);
        assert!((deviation[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn statistics_clamp_the_variance_not_the_deviation() {
        let hidden = [7.0f32, 7.0];
        let mut mean = [0.0f32];
        let mut deviation = [0.0f32];
        statistics(&hidden, 2, 1, &mut mean, &mut deviation, |_| 0.5);
        // sqrt(clamp(0, 1e-12)) == 1e-6, not 1e-12.
        assert!((deviation[0] - 1e-6).abs() < 1e-9, "{}", deviation[0]);
    }

    #[test]
    fn softmax_over_time_normalizes_each_channel_independently() {
        let mut values = vec![0.0f32, 10.0, 1.0, 10.0, 2.0, 10.0];
        softmax_over_time(&mut values, 3, 2);
        for channel in 0..2 {
            let total: f32 = (0..3).map(|frame| values[frame * 2 + channel]).sum();
            assert!((total - 1.0).abs() < 1e-6, "channel {channel}: {total}");
        }
        // The flat channel is uniform; the ramped one is monotone.
        assert!((values[1] - values[3]).abs() < 1e-6);
        assert!(values[0] < values[2] && values[2] < values[4]);
    }
}
