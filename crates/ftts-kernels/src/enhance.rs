//! FastEnhancer-S 48 kHz speech denoiser — a pure-Rust port of the pinned reference.
//!
//! Reference: `aask1357/fastenhancer` @ `f85223bd546b27f39dc0744e0310dcd246f750a4`,
//! checkpoint release `ckpt-v1.0.0-48khz` / `fastenhancer_s.zip` (MIT). The port consumes
//! the *inference-form* weights: the reference's own `remove_weight_reparameterizations()`
//! folds every weight-norm and BatchNorm into plain conv/linear weight+bias before export,
//! so this engine implements only convolutions, GRUs, one tiny frequency attention, and
//! the compressed-STFT front/back ends.
//!
//! Geometry (the `s` config, asserted at load): n_fft 1024, hop 512, 64 encoder channels,
//! stride-4 frequency downsample (512 -> 128 bins), 3 RNNFormer blocks at 48 channels x 48
//! frequency slots with 4 attention heads, complex ratio mask output.
//!
//! Everything is time-causal except the STFT overlap-add; the whole model runs per frame
//! with GRU state carried across frames, so the offline and streaming decompositions are
//! the same arithmetic.

use std::collections::BTreeMap;
use std::fmt;

pub const SAMPLE_RATE_HZ: u32 = 48_000;

const N_FFT: usize = 1024;
const HOP: usize = 512;
/// Model bins: the reference discards the Nyquist bin (`discard_last_freq_bin`).
const FREQ: usize = N_FFT / 2;
const CH: usize = 64;
const STRIDE: usize = 4;
const K0: usize = 8;
const F_ENC: usize = FREQ / STRIDE;
const ENC_CONVS: usize = 3;
const ENC_K: usize = 3;
const RF_CH: usize = 48;
const RF_FREQ: usize = 48;
const HEADS: usize = 4;
const HEAD_DIM: usize = RF_CH / HEADS;
const BLOCKS: usize = 3;
const COMPRESSION: f32 = 0.3;
const MAG_EPS: f32 = 1.0e-5;

#[derive(Debug)]
pub enum EnhanceError {
    MissingTensor(String),
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl fmt::Display for EnhanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor(name) => write!(f, "enhancer tensor {name} is missing"),
            Self::ShapeMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "enhancer tensor {name}: expected shape {expected:?}, got {got:?}"
                )
            }
        }
    }
}

impl std::error::Error for EnhanceError {}

struct Conv1d {
    /// `[out][in][k]` flattened.
    weight: Vec<f32>,
    bias: Vec<f32>,
    out_ch: usize,
    in_ch: usize,
    k: usize,
}

struct Linear {
    /// `[out][in]` flattened.
    weight: Vec<f32>,
}

struct GruWeights {
    /// `[3*H][H]` flattened, gate order r, z, n (PyTorch layout).
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
}

struct RnnFormerBlock {
    rnn: GruWeights,
    rnn_fc: Conv1d,
    qkv: Linear,
    attn_fc: Conv1d,
    /// `[RF_FREQ][RF_CH]`, block 0 only.
    pe: Option<Vec<f32>>,
}

/// The complete inference-form parameter set.
pub struct Enhancer {
    /// enc_pre remapped to a direct strided conv: `[CH][2][K0]`,
    /// kernel index `kk*STRIDE + si` (see `load` for the derivation).
    enc_pre: Conv1d,
    encoder: Vec<Conv1d>,
    rf_pre_lin: Linear,
    rf_pre_conv: Conv1d,
    blocks: Vec<RnnFormerBlock>,
    rf_post_lin: Linear,
    rf_post_conv: Conv1d,
    /// Per decoder stage: 1x1 concat-mix conv then k=3 conv.
    decoder: Vec<(Conv1d, Conv1d)>,
    dec_post_conv: Conv1d,
    /// ConvTranspose1d `[in=CH][out=2][K0]` flattened, plus bias `[2]`.
    dec_post_up: Vec<f32>,
    dec_post_up_bias: Vec<f32>,
    window: Vec<f32>,
    fft: Fft,
}

/// One tensor as handed to [`Enhancer::load`]: shape and row-major data.
pub type TensorEntry = (Vec<usize>, Vec<f32>);

fn take(
    tensors: &mut BTreeMap<String, TensorEntry>,
    name: &str,
    expected: &[usize],
) -> Result<Vec<f32>, EnhanceError> {
    let (shape, data) = tensors
        .remove(name)
        .ok_or_else(|| EnhanceError::MissingTensor(name.to_owned()))?;
    if shape != expected {
        return Err(EnhanceError::ShapeMismatch {
            name: name.to_owned(),
            expected: expected.to_vec(),
            got: shape,
        });
    }
    Ok(data)
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Enhancer {
    /// Builds the engine from named inference-form tensors (reference `named_parameters()`
    /// plus the `buffer.stft.window` buffer). Consumes the map; leftover tensors are ignored
    /// so callers may pass a superset artifact.
    pub fn load(mut tensors: BTreeMap<String, TensorEntry>) -> Result<Self, EnhanceError> {
        // enc_pre.0 is the reference's StridedConv1d: Conv1d(2*S, CH, K0/S) over the
        // stride-S-reshaped input. With reshaped channel index `si*2 + c` and kernel
        // position `kk`, output j reads x_pad[c][(j+kk)*S + si] = x_pad[c][j*S + (kk*S+si)],
        // i.e. a plain stride-S conv whose kernel index is m = kk*S + si.
        let raw = take(
            &mut tensors,
            "enc_pre.0.weight",
            &[CH, 2 * STRIDE, K0 / STRIDE],
        )?;
        let bias = take(&mut tensors, "enc_pre.0.bias", &[CH])?;
        let mut w = vec![0.0f32; CH * 2 * K0];
        for o in 0..CH {
            for si in 0..STRIDE {
                for c in 0..2 {
                    for kk in 0..K0 / STRIDE {
                        let m = kk * STRIDE + si;
                        w[(o * 2 + c) * K0 + m] =
                            raw[(o * (2 * STRIDE) + si * 2 + c) * (K0 / STRIDE) + kk];
                    }
                }
            }
        }
        let enc_pre = Conv1d {
            weight: w,
            bias,
            out_ch: CH,
            in_ch: 2,
            k: K0,
        };

        let mut encoder = Vec::with_capacity(ENC_CONVS);
        for i in 0..ENC_CONVS {
            encoder.push(Conv1d {
                weight: take(
                    &mut tensors,
                    &format!("encoder.{i}.0.weight"),
                    &[CH, CH, ENC_K],
                )?,
                bias: take(&mut tensors, &format!("encoder.{i}.0.bias"), &[CH])?,
                out_ch: CH,
                in_ch: CH,
                k: ENC_K,
            });
        }

        let rf_pre_lin = Linear {
            weight: take(&mut tensors, "rf_pre.0.weight", &[RF_FREQ, F_ENC])?,
        };
        let rf_pre_conv = Conv1d {
            weight: take(&mut tensors, "rf_pre.1.weight", &[RF_CH, CH, 1])?,
            bias: take(&mut tensors, "rf_pre.1.bias", &[RF_CH])?,
            out_ch: RF_CH,
            in_ch: CH,
            k: 1,
        };

        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let pe = if i == 0 {
                Some(take(&mut tensors, "rf_block.0.pe", &[RF_FREQ, RF_CH])?)
            } else {
                None
            };
            blocks.push(RnnFormerBlock {
                rnn: GruWeights {
                    weight_ih: take(
                        &mut tensors,
                        &format!("rf_block.{i}.rnn.weight_ih_l0"),
                        &[3 * RF_CH, RF_CH],
                    )?,
                    weight_hh: take(
                        &mut tensors,
                        &format!("rf_block.{i}.rnn.weight_hh_l0"),
                        &[3 * RF_CH, RF_CH],
                    )?,
                    bias_ih: take(
                        &mut tensors,
                        &format!("rf_block.{i}.rnn.bias_ih_l0"),
                        &[3 * RF_CH],
                    )?,
                    bias_hh: take(
                        &mut tensors,
                        &format!("rf_block.{i}.rnn.bias_hh_l0"),
                        &[3 * RF_CH],
                    )?,
                },
                rnn_fc: Conv1d {
                    weight: take(
                        &mut tensors,
                        &format!("rf_block.{i}.rnn_fc.weight"),
                        &[RF_CH, RF_CH],
                    )?,
                    bias: take(&mut tensors, &format!("rf_block.{i}.rnn_fc.bias"), &[RF_CH])?,
                    out_ch: RF_CH,
                    in_ch: RF_CH,
                    k: 1,
                },
                qkv: Linear {
                    weight: take(
                        &mut tensors,
                        &format!("rf_block.{i}.attn.qkv.weight"),
                        &[3 * RF_CH, RF_CH],
                    )?,
                },
                attn_fc: Conv1d {
                    weight: take(
                        &mut tensors,
                        &format!("rf_block.{i}.attn_fc.weight"),
                        &[RF_CH, RF_CH],
                    )?,
                    bias: take(
                        &mut tensors,
                        &format!("rf_block.{i}.attn_fc.bias"),
                        &[RF_CH],
                    )?,
                    out_ch: RF_CH,
                    in_ch: RF_CH,
                    k: 1,
                },
                pe,
            });
        }

        let rf_post_lin = Linear {
            weight: take(&mut tensors, "rf_post.0.weight", &[F_ENC, RF_FREQ])?,
        };
        let rf_post_conv = Conv1d {
            weight: take(&mut tensors, "rf_post.1.weight", &[CH, RF_CH, 1])?,
            bias: take(&mut tensors, "rf_post.1.bias", &[CH])?,
            out_ch: CH,
            in_ch: RF_CH,
            k: 1,
        };

        let mut decoder = Vec::with_capacity(ENC_CONVS);
        for i in 0..ENC_CONVS {
            decoder.push((
                Conv1d {
                    weight: take(
                        &mut tensors,
                        &format!("decoder.{i}.0.weight"),
                        &[CH, 2 * CH, 1],
                    )?,
                    bias: take(&mut tensors, &format!("decoder.{i}.0.bias"), &[CH])?,
                    out_ch: CH,
                    in_ch: 2 * CH,
                    k: 1,
                },
                Conv1d {
                    weight: take(
                        &mut tensors,
                        &format!("decoder.{i}.2.weight"),
                        &[CH, CH, ENC_K],
                    )?,
                    bias: take(&mut tensors, &format!("decoder.{i}.2.bias"), &[CH])?,
                    out_ch: CH,
                    in_ch: CH,
                    k: ENC_K,
                },
            ));
        }

        let dec_post_conv = Conv1d {
            weight: take(&mut tensors, "dec_post.0.weight", &[CH, 2 * CH, 1])?,
            bias: take(&mut tensors, "dec_post.0.bias", &[CH])?,
            out_ch: CH,
            in_ch: 2 * CH,
            k: 1,
        };
        let dec_post_up = take(&mut tensors, "dec_post.2.weight", &[CH, 2, K0])?;
        let dec_post_up_bias = take(&mut tensors, "dec_post.2.bias", &[2])?;
        let window = take(&mut tensors, "buffer.stft.window", &[N_FFT])?;

        Ok(Self {
            enc_pre,
            encoder,
            rf_pre_lin,
            rf_pre_conv,
            blocks,
            rf_post_lin,
            rf_post_conv,
            decoder,
            dec_post_conv,
            dec_post_up,
            dec_post_up_bias,
            window,
            fft: Fft::new(N_FFT),
        })
    }

    /// Denoises a 48 kHz mono clip. Returns `(frames - 1) * hop` samples where
    /// `frames = wav.len() / hop + 1` (the reference's centered-STFT round trip);
    /// pad the input to a hop multiple to keep the full length.
    pub fn enhance_48k(&self, wav: &[f32]) -> Vec<f32> {
        // Below one hop the length contract is zero samples anyway ((frames-1)*hop == 0),
        // and the reflect-padding walk below does not terminate for 0- or 1-sample input —
        // reflection needs more signal than padding. Return the contracted empty answer.
        if wav.len() < HOP {
            return Vec::new();
        }
        let frames = wav.len() / HOP + 1;
        let mut state = self.new_state();
        // Compressed spectrum per frame, then masked spectrum accumulated into OLA.
        let mut out = vec![0.0f32; (frames - 1) * HOP + N_FFT];
        let mut winsq = vec![0.0f32; (frames - 1) * HOP + N_FFT];
        let mut spec = [0.0f32; 2 * (FREQ + 1)];
        let mut scratch_time = vec![0.0f32; N_FFT];

        for t in 0..frames {
            self.frame_spectrum(wav, t, &mut spec, &mut scratch_time);
            // Compress: x * max(|x|, eps)^(c-1), Nyquist bin discarded.
            let mut comp = [0.0f32; 2 * FREQ];
            for f in 0..FREQ {
                let re = spec[2 * f];
                let im = spec[2 * f + 1];
                let mag = (re * re + im * im).sqrt().max(MAG_EPS);
                let g = mag.powf(COMPRESSION - 1.0);
                comp[2 * f] = re * g;
                comp[2 * f + 1] = im * g;
            }
            let mask = self.frame_forward(&comp, &mut state);
            // spec_hat = comp * mask (complex), then uncompress by |spec_hat|^(1/c - 1).
            let mut frame_spec = [0.0f32; 2 * (FREQ + 1)];
            for f in 0..FREQ {
                let (ar, ai) = (comp[2 * f], comp[2 * f + 1]);
                let (br, bi) = (mask[2 * f], mask[2 * f + 1]);
                let re = ar * br - ai * bi;
                let im = ar * bi + ai * br;
                let mag = (re * re + im * im).sqrt();
                let g = if mag > 0.0 {
                    mag.powf(1.0 / COMPRESSION - 1.0)
                } else {
                    0.0
                };
                frame_spec[2 * f] = re * g;
                frame_spec[2 * f + 1] = im * g;
            }
            self.fft.irfft(&frame_spec, &mut scratch_time);
            let base = t * HOP;
            for i in 0..N_FFT {
                out[base + i] += scratch_time[i] * self.window[i];
                winsq[base + i] += self.window[i] * self.window[i];
            }
        }

        // torch.istft: normalize by the window-square envelope, trim n_fft/2 padding.
        let start = N_FFT / 2;
        let len = (frames - 1) * HOP;
        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            let w = winsq[start + i];
            result.push(if w > 1.0e-11 { out[start + i] / w } else { 0.0 });
        }
        result
    }

    /// Denoises a 24 kHz mono clip: up to the model's native 48 kHz, through the network,
    /// back down to 24 kHz, preserving length exactly.
    ///
    /// This is the shape both product surfaces consume (the engine's pipeline is 24 kHz
    /// end to end); `enhance_48k` stays public for callers already at the native rate.
    pub fn enhance_24k(&self, wav24k: &[f32]) -> Vec<f32> {
        let mut wav48 = resample_lanczos6(wav24k, 24_000, SAMPLE_RATE_HZ);
        let target_len = wav48.len();
        // On the hop grid the STFT round trip returns every sample (see enhance_48k).
        let padded = target_len.div_ceil(HOP) * HOP;
        wav48.resize(padded, 0.0);
        let mut enhanced = self.enhance_48k(&wav48);
        enhanced.truncate(target_len);
        let mut back = resample_lanczos6(&enhanced, SAMPLE_RATE_HZ, 24_000);
        back.truncate(wav24k.len());
        back
    }

    fn new_state(&self) -> Vec<Vec<f32>> {
        vec![vec![0.0f32; RF_FREQ * RF_CH]; BLOCKS]
    }

    /// Centered, reflect-padded, windowed rFFT of frame `t`.
    fn frame_spectrum(&self, wav: &[f32], t: usize, spec: &mut [f32], time: &mut [f32]) {
        let n = wav.len() as isize;
        let start = t as isize * HOP as isize - (N_FFT / 2) as isize;
        for (i, slot) in time.iter_mut().enumerate() {
            let mut idx = start + i as isize;
            // torch reflect padding (no edge repetition). The clip is longer than one
            // reflection order for any real enrollment input; iterate for tiny inputs.
            loop {
                if idx < 0 {
                    idx = -idx;
                } else if idx >= n {
                    idx = 2 * (n - 1) - idx;
                } else {
                    break;
                }
            }
            *slot = wav[idx as usize] * self.window[i];
        }
        self.fft.rfft(time, spec);
    }

    /// One frame through encoder / RNNFormer / decoder; returns the complex mask.
    fn frame_forward(&self, comp: &[f32], state: &mut [Vec<f32>]) -> [f32; 2 * FREQ] {
        // ---- encoder prenet: [2][FREQ] -> [CH][F_ENC], direct stride-4 conv -----------
        // Input layout for the conv: channel 0 = real, channel 1 = imag.
        let pad = (K0 - STRIDE) / 2;
        let mut x = vec![0.0f32; CH * F_ENC];
        for o in 0..CH {
            let w = &self.enc_pre.weight[o * 2 * K0..(o + 1) * 2 * K0];
            let b = self.enc_pre.bias[o];
            for j in 0..F_ENC {
                let mut acc = b;
                for m in 0..K0 {
                    let f = (j * STRIDE + m) as isize - pad as isize;
                    if f >= 0 && (f as usize) < FREQ {
                        let f = f as usize;
                        acc += w[m] * comp[2 * f] + w[K0 + m] * comp[2 * f + 1];
                    }
                }
                x[o * F_ENC + j] = silu(acc);
            }
        }

        // ---- encoder stack, keeping skip outputs ---------------------------------------
        let mut skips: Vec<Vec<f32>> = Vec::with_capacity(1 + ENC_CONVS);
        skips.push(x.clone());
        for conv in &self.encoder {
            x = conv_k_same(conv, &x, F_ENC, true);
            skips.push(x.clone());
        }

        // ---- RNNFormer prenet: freq linear then 1x1 channel mix ------------------------
        // x: [CH][F_ENC] -> lin over freq -> [CH][RF_FREQ] -> conv1x1 -> [RF_CH][RF_FREQ]
        let mut xf = vec![0.0f32; CH * RF_FREQ];
        for c in 0..CH {
            let row = &x[c * F_ENC..(c + 1) * F_ENC];
            for (fr, slot) in xf[c * RF_FREQ..(c + 1) * RF_FREQ].iter_mut().enumerate() {
                let w = &self.rf_pre_lin.weight[fr * F_ENC..(fr + 1) * F_ENC];
                let mut acc = 0.0f32;
                for f in 0..F_ENC {
                    acc += w[f] * row[f];
                }
                *slot = acc;
            }
        }
        // Transpose into token-major [RF_FREQ][RF_CH] while mixing channels.
        let mut tokens = vec![0.0f32; RF_FREQ * RF_CH];
        for oc in 0..RF_CH {
            let w = &self.rf_pre_conv.weight[oc * CH..(oc + 1) * CH];
            let b = self.rf_pre_conv.bias[oc];
            for fr in 0..RF_FREQ {
                let mut acc = b;
                for ic in 0..CH {
                    acc += w[ic] * xf[ic * RF_FREQ + fr];
                }
                tokens[fr * RF_CH + oc] = acc;
            }
        }

        // ---- RNNFormer blocks -----------------------------------------------------------
        for (block, h) in self.blocks.iter().zip(state.iter_mut()) {
            // GRU over time, one independent state per frequency token.
            for fr in 0..RF_FREQ {
                let tok = &mut tokens[fr * RF_CH..(fr + 1) * RF_CH];
                let hcur = &mut h[fr * RF_CH..(fr + 1) * RF_CH];
                let mut gi = [0.0f32; 3 * RF_CH];
                let mut gh = [0.0f32; 3 * RF_CH];
                for g in 0..3 * RF_CH {
                    let wi = &block.rnn.weight_ih[g * RF_CH..(g + 1) * RF_CH];
                    let wh = &block.rnn.weight_hh[g * RF_CH..(g + 1) * RF_CH];
                    let mut ai = block.rnn.bias_ih[g];
                    let mut ah = block.rnn.bias_hh[g];
                    for c in 0..RF_CH {
                        ai += wi[c] * tok[c];
                        ah += wh[c] * hcur[c];
                    }
                    gi[g] = ai;
                    gh[g] = ah;
                }
                let mut rnn_out = [0.0f32; RF_CH];
                for c in 0..RF_CH {
                    let r = sigmoid(gi[c] + gh[c]);
                    let z = sigmoid(gi[RF_CH + c] + gh[RF_CH + c]);
                    let ncand = (gi[2 * RF_CH + c] + r * gh[2 * RF_CH + c]).tanh();
                    let hnew = (1.0 - z) * ncand + z * hcur[c];
                    hcur[c] = hnew;
                    rnn_out[c] = hnew;
                }
                // rnn_fc (post-norm folded into weight+bias) + residual.
                for (c, slot) in tok.iter_mut().enumerate().take(RF_CH) {
                    let w = &block.rnn_fc.weight[c * RF_CH..(c + 1) * RF_CH];
                    let mut acc = block.rnn_fc.bias[c];
                    for i in 0..RF_CH {
                        acc += w[i] * rnn_out[i];
                    }
                    *slot += acc;
                }
            }

            if let Some(pe) = &block.pe {
                for (slot, p) in tokens.iter_mut().zip(pe.iter()) {
                    *slot += p;
                }
            }

            // Frequency attention over RF_FREQ tokens.
            let mut qkv = vec![0.0f32; RF_FREQ * 3 * RF_CH];
            for fr in 0..RF_FREQ {
                let tok = &tokens[fr * RF_CH..(fr + 1) * RF_CH];
                for o in 0..3 * RF_CH {
                    let w = &block.qkv.weight[o * RF_CH..(o + 1) * RF_CH];
                    let mut acc = 0.0f32;
                    for c in 0..RF_CH {
                        acc += w[c] * tok[c];
                    }
                    qkv[fr * 3 * RF_CH + o] = acc;
                }
            }
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            let mut attn_out = vec![0.0f32; RF_FREQ * RF_CH];
            let mut scores = [0.0f32; RF_FREQ];
            for head in 0..HEADS {
                // The reference reshapes qkv to [.., heads, 3*HEAD_DIM]: head h owns
                // contiguous columns [h*3D .. (h+1)*3D] split q/k/v inside.
                let base = head * 3 * HEAD_DIM;
                for i in 0..RF_FREQ {
                    let q = &qkv[i * 3 * RF_CH + base..i * 3 * RF_CH + base + HEAD_DIM];
                    let mut max = f32::NEG_INFINITY;
                    for (j, s) in scores.iter_mut().enumerate() {
                        let k = &qkv
                            [j * 3 * RF_CH + base + HEAD_DIM..j * 3 * RF_CH + base + 2 * HEAD_DIM];
                        let mut acc = 0.0f32;
                        for d in 0..HEAD_DIM {
                            acc += q[d] * k[d];
                        }
                        *s = acc * scale;
                        max = max.max(*s);
                    }
                    let mut denom = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        denom += *s;
                    }
                    let inv = 1.0 / denom;
                    let out = &mut attn_out
                        [i * RF_CH + head * HEAD_DIM..i * RF_CH + (head + 1) * HEAD_DIM];
                    for (j, s) in scores.iter().enumerate() {
                        let v = &qkv[j * 3 * RF_CH + base + 2 * HEAD_DIM
                            ..j * 3 * RF_CH + base + 3 * HEAD_DIM];
                        let p = *s * inv;
                        for d in 0..HEAD_DIM {
                            out[d] += p * v[d];
                        }
                    }
                }
            }
            for fr in 0..RF_FREQ {
                let src = &attn_out[fr * RF_CH..(fr + 1) * RF_CH];
                let tok = &mut tokens[fr * RF_CH..(fr + 1) * RF_CH];
                for (c, slot) in tok.iter_mut().enumerate() {
                    let w = &block.attn_fc.weight[c * RF_CH..(c + 1) * RF_CH];
                    let mut acc = block.attn_fc.bias[c];
                    for i in 0..RF_CH {
                        acc += w[i] * src[i];
                    }
                    *slot += acc;
                }
            }
        }

        // ---- RNNFormer postnet: back to [CH][F_ENC] --------------------------------------
        // tokens [RF_FREQ][RF_CH] -> per channel freq expansion, then 1x1 mix RF_CH -> CH.
        let mut yf = vec![0.0f32; RF_CH * F_ENC];
        for c in 0..RF_CH {
            for f in 0..F_ENC {
                let w = &self.rf_post_lin.weight[f * RF_FREQ..(f + 1) * RF_FREQ];
                let mut acc = 0.0f32;
                for fr in 0..RF_FREQ {
                    acc += w[fr] * tokens[fr * RF_CH + c];
                }
                yf[c * F_ENC + f] = acc;
            }
        }
        let mut y = vec![0.0f32; CH * F_ENC];
        for oc in 0..CH {
            let w = &self.rf_post_conv.weight[oc * RF_CH..(oc + 1) * RF_CH];
            let b = self.rf_post_conv.bias[oc];
            for f in 0..F_ENC {
                let mut acc = b;
                for ic in 0..RF_CH {
                    acc += w[ic] * yf[ic * F_ENC + f];
                }
                y[oc * F_ENC + f] = acc;
            }
        }

        // ---- decoder with encoder skips ---------------------------------------------------
        for (mix, conv) in &self.decoder {
            let skip = skips.pop().expect("one skip per decoder stage");
            y = concat_mix(mix, &y, &skip, F_ENC);
            y = conv_k_same(conv, &y, F_ENC, true);
        }

        // ---- decoder postnet: 1x1 mix, SiLU, transposed conv to the mask ------------------
        let skip = skips.pop().expect("enc_pre skip");
        let z = concat_mix(&self.dec_post_conv, &y, &skip, F_ENC);
        let mut mask = [0.0f32; 2 * FREQ];
        // ConvTranspose1d(CH -> 2, k=K0, stride=STRIDE, padding=pad):
        // out[o][p] = bias[o] + sum_{c, j, m : j*STRIDE + m - pad == p} w[c][o][m] * z[c][j]
        let pad = (K0 - STRIDE) / 2;
        for f in 0..FREQ {
            mask[2 * f] = self.dec_post_up_bias[0];
            mask[2 * f + 1] = self.dec_post_up_bias[1];
        }
        for c in 0..CH {
            let wrow = &self.dec_post_up[c * 2 * K0..(c + 1) * 2 * K0];
            for j in 0..F_ENC {
                let zv = z[c * F_ENC + j];
                if zv == 0.0 {
                    continue;
                }
                let base = j * STRIDE;
                for m in 0..K0 {
                    let p = base + m;
                    if p < pad || p - pad >= FREQ {
                        continue;
                    }
                    let p = p - pad;
                    mask[2 * p] += wrow[m] * zv;
                    mask[2 * p + 1] += wrow[K0 + m] * zv;
                }
            }
        }
        mask
    }
}

/// Windowed-sinc (Lanczos-6) rate conversion, the same kernel enrollment trusts for its
/// any-rate references: cutoff clamped to the lower Nyquist, taps normalized by their own
/// sum so DC gain stays 1 at the clip edges.
#[must_use]
pub fn resample_lanczos6(mono: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return mono.to_vec();
    }
    const LOBES: f64 = 6.0;
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let cutoff = ratio.min(1.0);
    let half = (LOBES / cutoff).ceil() as isize;
    let out_len = ((mono.len() as f64) * ratio).round() as usize;

    let mut out = Vec::with_capacity(out_len);
    for index in 0..out_len {
        let center = index as f64 / ratio;
        let first = center.floor() as isize - half + 1;
        let mut acc = 0.0_f64;
        let mut norm = 0.0_f64;
        for tap in first..first + 2 * half {
            if tap < 0 {
                continue;
            }
            let Some(sample) = mono.get(tap as usize) else {
                break;
            };
            let weight = lanczos6_tap(center - tap as f64, cutoff);
            acc += weight * f64::from(*sample);
            norm += weight;
        }
        out.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.0
        });
    }
    out
}

fn lanczos6_tap(distance: f64, cutoff: f64) -> f64 {
    const LOBES: f64 = 6.0;
    let x = distance * cutoff;
    if x.abs() >= LOBES {
        return 0.0;
    }
    let sinc = |v: f64| {
        if v.abs() < 1e-12 {
            1.0
        } else {
            let p = std::f64::consts::PI * v;
            p.sin() / p
        }
    };
    sinc(x) * sinc(x / LOBES)
}

/// Same-padding k-wide conv over `width` positions, optional SiLU.
fn conv_k_same(conv: &Conv1d, x: &[f32], width: usize, act: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; conv.out_ch * width];
    let pad = (conv.k - 1) / 2;
    for o in 0..conv.out_ch {
        let orow = &mut out[o * width..(o + 1) * width];
        for slot in orow.iter_mut() {
            *slot = conv.bias[o];
        }
        for c in 0..conv.in_ch {
            let w = &conv.weight[(o * conv.in_ch + c) * conv.k..(o * conv.in_ch + c + 1) * conv.k];
            let xrow = &x[c * width..(c + 1) * width];
            for (m, &wv) in w.iter().enumerate() {
                let shift = m as isize - pad as isize;
                let (dst_start, src_start) = if shift < 0 {
                    ((-shift) as usize, 0usize)
                } else {
                    (0usize, shift as usize)
                };
                let count = width - dst_start.max(src_start);
                for i in 0..count {
                    orow[dst_start + i] += wv * xrow[src_start + i];
                }
            }
        }
        if act {
            for slot in orow.iter_mut() {
                *slot = silu(*slot);
            }
        }
    }
    out
}

/// 1x1 conv over the channel concat `[x ; skip]`, then SiLU.
fn concat_mix(conv: &Conv1d, x: &[f32], skip: &[f32], width: usize) -> Vec<f32> {
    let half = conv.in_ch / 2;
    let mut out = vec![0.0f32; conv.out_ch * width];
    for o in 0..conv.out_ch {
        let w = &conv.weight[o * conv.in_ch..(o + 1) * conv.in_ch];
        let orow = &mut out[o * width..(o + 1) * width];
        for slot in orow.iter_mut() {
            *slot = conv.bias[o];
        }
        for c in 0..half {
            let wv = w[c];
            let xrow = &x[c * width..(c + 1) * width];
            for (slot, xv) in orow.iter_mut().zip(xrow.iter()) {
                *slot += wv * xv;
            }
        }
        for c in 0..half {
            let wv = w[half + c];
            let srow = &skip[c * width..(c + 1) * width];
            for (slot, sv) in orow.iter_mut().zip(srow.iter()) {
                *slot += wv * sv;
            }
        }
        for slot in orow.iter_mut() {
            *slot = silu(*slot);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// FFT: iterative radix-2, real transforms via the complex core. n = 1024 only in practice
// but written for any power of two.
// ---------------------------------------------------------------------------------------

struct Fft {
    n: usize,
    /// Twiddles for the forward transform: e^{-2πik/n} for k in 0..n/2.
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
    rev: Vec<u32>,
}

impl Fft {
    fn new(n: usize) -> Self {
        assert!(n.is_power_of_two());
        let mut tw_re = Vec::with_capacity(n / 2);
        let mut tw_im = Vec::with_capacity(n / 2);
        for k in 0..n / 2 {
            let ang = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
            tw_re.push(ang.cos() as f32);
            tw_im.push(ang.sin() as f32);
        }
        let bits = n.trailing_zeros();
        let rev = (0..n as u32)
            .map(|i| i.reverse_bits() >> (32 - bits))
            .collect();
        Self {
            n,
            tw_re,
            tw_im,
            rev,
        }
    }

    /// In-place complex FFT over interleaved (re, im) pairs; `inverse` conjugates the
    /// twiddles (no 1/n scaling — callers scale).
    fn fft_complex(&self, buf: &mut [f32], inverse: bool) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                buf.swap(2 * i, 2 * j);
                buf.swap(2 * i + 1, 2 * j + 1);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            let mut start = 0;
            while start < n {
                for k in 0..half {
                    let wre = self.tw_re[k * step];
                    let wim = if inverse {
                        -self.tw_im[k * step]
                    } else {
                        self.tw_im[k * step]
                    };
                    let a = start + k;
                    let b = a + half;
                    let (bre, bim) = (buf[2 * b], buf[2 * b + 1]);
                    let tre = bre * wre - bim * wim;
                    let tim = bre * wim + bim * wre;
                    let (are, aim) = (buf[2 * a], buf[2 * a + 1]);
                    buf[2 * a] = are + tre;
                    buf[2 * a + 1] = aim + tim;
                    buf[2 * b] = are - tre;
                    buf[2 * b + 1] = aim - tim;
                }
                start += len;
            }
            len *= 2;
        }
    }

    /// Real forward transform: `time[n]` -> `spec[2*(n/2+1)]` interleaved.
    fn rfft(&self, time: &[f32], spec: &mut [f32]) {
        let n = self.n;
        let mut buf = vec![0.0f32; 2 * n];
        for (i, &v) in time.iter().enumerate() {
            buf[2 * i] = v;
        }
        self.fft_complex(&mut buf, false);
        spec[..2 * (n / 2 + 1)].copy_from_slice(&buf[..2 * (n / 2 + 1)]);
    }

    /// Inverse real transform: `spec[2*(n/2+1)]` -> `time[n]` (with 1/n scaling).
    fn irfft(&self, spec: &[f32], time: &mut [f32]) {
        let n = self.n;
        let mut buf = vec![0.0f32; 2 * n];
        buf[..2 * (n / 2 + 1)].copy_from_slice(&spec[..2 * (n / 2 + 1)]);
        for k in 1..n / 2 {
            buf[2 * (n - k)] = spec[2 * k];
            buf[2 * (n - k) + 1] = -spec[2 * k + 1];
        }
        self.fft_complex(&mut buf, true);
        let inv = 1.0 / n as f32;
        for (i, slot) in time.iter_mut().enumerate() {
            *slot = buf[2 * i] * inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_weight_enhancer() -> Enhancer {
        let mut tensors = BTreeMap::new();
        let mut put = |name: &str, shape: &[usize]| {
            let count: usize = shape.iter().product();
            tensors.insert(name.to_owned(), (shape.to_vec(), vec![0.0f32; count]));
        };
        put("enc_pre.0.weight", &[CH, 2 * STRIDE, K0 / STRIDE]);
        put("enc_pre.0.bias", &[CH]);
        for i in 0..ENC_CONVS {
            put(&format!("encoder.{i}.0.weight"), &[CH, CH, ENC_K]);
            put(&format!("encoder.{i}.0.bias"), &[CH]);
            put(&format!("decoder.{i}.0.weight"), &[CH, 2 * CH, 1]);
            put(&format!("decoder.{i}.0.bias"), &[CH]);
            put(&format!("decoder.{i}.2.weight"), &[CH, CH, ENC_K]);
            put(&format!("decoder.{i}.2.bias"), &[CH]);
        }
        put("rf_pre.0.weight", &[RF_FREQ, F_ENC]);
        put("rf_pre.1.weight", &[RF_CH, CH, 1]);
        put("rf_pre.1.bias", &[RF_CH]);
        put("rf_block.0.pe", &[RF_FREQ, RF_CH]);
        for i in 0..BLOCKS {
            put(
                &format!("rf_block.{i}.rnn.weight_ih_l0"),
                &[3 * RF_CH, RF_CH],
            );
            put(
                &format!("rf_block.{i}.rnn.weight_hh_l0"),
                &[3 * RF_CH, RF_CH],
            );
            put(&format!("rf_block.{i}.rnn.bias_ih_l0"), &[3 * RF_CH]);
            put(&format!("rf_block.{i}.rnn.bias_hh_l0"), &[3 * RF_CH]);
            put(&format!("rf_block.{i}.rnn_fc.weight"), &[RF_CH, RF_CH]);
            put(&format!("rf_block.{i}.rnn_fc.bias"), &[RF_CH]);
            put(
                &format!("rf_block.{i}.attn.qkv.weight"),
                &[3 * RF_CH, RF_CH],
            );
            put(&format!("rf_block.{i}.attn_fc.weight"), &[RF_CH, RF_CH]);
            put(&format!("rf_block.{i}.attn_fc.bias"), &[RF_CH]);
        }
        put("rf_post.0.weight", &[F_ENC, RF_FREQ]);
        put("rf_post.1.weight", &[CH, RF_CH, 1]);
        put("rf_post.1.bias", &[CH]);
        put("dec_post.0.weight", &[CH, 2 * CH, 1]);
        put("dec_post.0.bias", &[CH]);
        put("dec_post.2.weight", &[CH, 2, K0]);
        put("dec_post.2.bias", &[2]);
        put("buffer.stft.window", &[N_FFT]);
        Enhancer::load(tensors).expect("all shapes present")
    }

    /// The reflect-padding walk cannot terminate on 0- or 1-sample input; the guard must
    /// return the contracted empty answer instead of spinning, and short-but-real input
    /// must keep its exact length through the 24 kHz round trip.
    #[test]
    fn tiny_inputs_terminate_and_keep_their_length() {
        let enhancer = zero_weight_enhancer();
        assert!(enhancer.enhance_48k(&[]).is_empty());
        assert!(enhancer.enhance_48k(&[0.25]).is_empty());
        assert!(enhancer.enhance_48k(&[0.25; 100]).is_empty());
        assert!(enhancer.enhance_24k(&[]).is_empty());
        assert_eq!(enhancer.enhance_24k(&[0.25; 50]).len(), 50);
        // 2,400 samples = 100 ms: enough to cross several hops without making the
        // debug-profile suite crawl (the full-length case lives in the parity harness).
        assert_eq!(enhancer.enhance_24k(&[0.25; 2_400]).len(), 2_400);
        assert_eq!(enhancer.enhance_48k(&[0.25; 1024]).len(), 1024);
    }

    #[test]
    fn fft_round_trip_recovers_the_signal() {
        let fft = Fft::new(1024);
        let time: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 0.013).sin() + 0.3 * (i as f32 * 0.21).cos())
            .collect();
        let mut spec = vec![0.0f32; 2 * 513];
        let mut back = vec![0.0f32; 1024];
        fft.rfft(&time, &mut spec);
        fft.irfft(&spec, &mut back);
        for (a, b) in time.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn fft_matches_the_dft_definition() {
        let n = 16;
        let fft = Fft::new(n);
        let time: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7).sin()).collect();
        let mut spec = vec![0.0f32; 2 * (n / 2 + 1)];
        fft.rfft(&time, &mut spec);
        for k in 0..=n / 2 {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (i, &v) in time.iter().enumerate() {
                let ang = -2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                re += v as f64 * ang.cos();
                im += v as f64 * ang.sin();
            }
            assert!((spec[2 * k] as f64 - re).abs() < 1.0e-3, "bin {k} re");
            assert!((spec[2 * k + 1] as f64 - im).abs() < 1.0e-3, "bin {k} im");
        }
    }
}
