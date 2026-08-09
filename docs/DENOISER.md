# The neural denoiser: FastEnhancer-S 48 kHz

Enrollment denoises the reference automatically with a pure-Rust port of FastEnhancer-S,
a 207 K-parameter streaming speech-enhancement network (ICASSP 2026). This is the default
whenever the pulled weights are present: `ftts enroll`, the ephemeral enrollment inside
`ftts say --voice <audio-file>`, and browser mic enrollment (weights embedded in the wasm
module) all run it with no flag. `--no-denoise` opts out; an explicit `--denoise` also
engages the classic OM-LSA/IMCRA spectral subtraction when the weights are absent, and
`FTTS_DENOISE_ENGINE=omlsa` forces that engine. `.spk` vectors and presets never enter
the cleanup path.

## Truth pack

| item | pin |
|---|---|
| upstream | `aask1357/fastenhancer` @ `f85223bd546b27f39dc0744e0310dcd246f750a4` (MIT) |
| checkpoint | release `ckpt-v1.0.0-48khz`, asset `fastenhancer_s.zip` (sha256 `20a6269d60304cc55e11fa48192a39d4c716679548226554c38b6228a6fef28d`), `00500.pth` |
| config | `configs/fastenhancer_48khz/s.yaml` (n_fft 1024, hop 512, 64 ch, stride 4, 3 RNNFormer blocks @ 48 ch x 48 freq, 4 heads, SiLU, compression 0.3) |
| shipped artifact | `fastenhancer-s-48k-denoise.safetensors` on release `model-qwen3-tts-v1`, sha256 `28c1807fd9113e4ca09d3aacb2ecb07a742917321bfaced8b92598daffbd098b`, 838 440 bytes |
| artifact form | *inference form*: the reference's own `remove_weight_reparameterizations()` folds every weight-norm and BatchNorm into plain conv/linear weight+bias; re-serialized as F32 safetensors (59 tensors) |
| oracle | pinned PyTorch (torch 2.13.0, CPU, single thread) on the folded model; measured nondeterminism floor between 1 and 4 threads: **0.0** |

The 48 kHz checkpoint is trained with dynamic low-pass augmentation whose target list
includes 24 kHz content, so the engine's 24 kHz pipeline is in-distribution for it.

## Port and parity

- Engine: `ftts_kernels::enhance` — convs along the frequency axis per STFT frame, GRU state
  carried across frames (time-causal), one 48-token frequency attention per block, complex
  ratio mask, compressed STFT front/back ends. No threads, no mmap, no unsafe; compiles for
  `wasm32-unknown-unknown` unchanged.
- Loader: `ftts_artifacts::enhance_loader` (safetensors, whole-tensor materialization —
  the artifact is ~830 KB).
- Parity vs the PyTorch oracle (`enhance_parity` example, three pinned fixtures):
  124.9 / 114.3 / 118.2 dB SNR, max abs diff ≤ 1.0e-7, output length exact. That is float
  round-off, far below the oracle's own quantization of hearing.
- Measured on the enrollment fixture (white static at 15 dB SNR): pause floor
  −51.8 → −117.4 dBFS (OM-LSA reference: −76.4), and the enrolled x-vector moves *closer*
  to the clean-source enrollment (cosine 0.9613 vs 0.9548 OM-LSA vs 0.9459 raw noisy).
- Speed: RTF ≈ 0.24 single-threaded on an M4 Pro *under load average ~100*; enrollment-scale
  clips are effectively instant. No int8/team lever is warranted at this cost.

## Distribution

`ftts pull` fetches the artifact into `<model>/denoise/fastenhancer-s-48k.safetensors`
(manifest-pinned size + sha256, same verify-or-redownload contract as the main model).
When the artifact is present, enrollment cleans with it by default; when absent, the
automatic path skips cleanup (only an explicit `--denoise` engages the classic engine
there). A present-but-corrupt artifact is a hard error naming `ftts pull --force`,
never a silent engine swap. The wasm build carries the same weights via `include_bytes!`,
so the browser needs no fetch.

## Regeneration recipe

1. Clone the pinned upstream commit; download `ckpt-v1.0.0-48khz/fastenhancer_s.zip`.
2. Build `Model(**s.yaml model_kwargs)`, load `00500.pth["model"]`, `eval()`,
   `remove_weight_reparameterizations()`, `flatten_parameters()`.
3. Save `named_parameters()` plus the `stft.window` buffer (as `buffer.stft.window`)
   to F32 safetensors.
4. Verify sha256 against the manifest pin before uploading anywhere.
