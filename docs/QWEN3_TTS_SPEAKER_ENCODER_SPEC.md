# Qwen3-TTS 12Hz — Speaker Encoder: Mel Front End & Conditioning Injection (OQ-9)

Partial resolution of **OQ-9** (`frankentts-oq9-ecapa-features-ac5`). Two of its three exit criteria
are met here — the feature pipeline parameter-by-parameter, and the injection mechanism. The third
(oracle fixtures for audio → mel → embedding → injected state) is **blocked on
`frankentts-t-oracle-fixtures-6w9`**, which is still open; see §5.

Citation keys (paths relative to `docs/truth-pack/snapshots/`):

| Key | File |
|---|---|
| `MODEL` | `gh/qwen_tts/core/models/modeling_qwen3_tts.py` |
| `CFGPY` | `gh/qwen_tts/core/models/configuration_qwen3_tts.py` |
| `CFG` | `hf/config.json` |
| `WT` | `hf/model.safetensors` (header parse; 76 `speaker_encoder.*` tensors) |

---

## 1. The mel front end — every parameter

`MODEL:1941-1954` (call site) and `MODEL:399-464` (implementation). **All values are literals at the
call site, not config-driven** — they cannot be changed without editing code, which makes them safe
to hard-code in our engine but also means nothing in `CFG` records them.

| Parameter | Value | Source |
|---|---|---|
| Input sample rate | **24 000** — asserted, not resampled (`assert sr == 24000`) | `MODEL:1942` |
| `n_fft` | **1024** | `MODEL:1944` |
| `num_mels` | **128** | `MODEL:1946` |
| `hop_size` | **256** (⇒ 93.75 mel frames/s) | `MODEL:1948` |
| `win_size` | **1024** | `MODEL:1949` |
| `fmin` | **0** | `MODEL:1950` |
| `fmax` | **12000** (= Nyquist, passed explicitly) | `MODEL:1951` |
| Window | `torch.hann_window(1024)` — **periodic** (PyTorch default) | `MODEL:440` |
| `center` | **False** — STFT does no centering of its own | `MODEL:408, 453` |
| Manual padding | **reflect, `(n_fft − hop)//2 = 384` on both sides**, applied *before* the STFT | `MODEL:442-445` |
| Magnitude | `sqrt(re² + im² + 1e-9)` — **epsilon inside the sqrt** | `MODEL:459` |
| Mel filterbank | `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` — **Slaney norm, Slaney mel scale** (librosa defaults `norm='slaney'`, `htk=False`) | `MODEL:435-437`, docstring `MODEL:412` |
| Filterbank dtype | built in **float32**, applied via `matmul` to the magnitude spectrum | `MODEL:439, 461` |
| Compression | `log(clamp(x, min=1e-5) × 1)` — **natural log**, clip 1e-5 | `MODEL:462`, `MODEL:395-396` |
| Output layout | `(B, n_mels, T)` transposed to **`(B, T, 128)`** at the call site | `MODEL:1952` |

### 1.1 The five details that decide parity

The bead names mel front ends "the classic silent killer of speaker-embedding parity". These are the
specific traps in this one, all of which our pure-Rust STFT/mel must reproduce exactly:

1. **`center=False` + manual reflect pad of 384.** This is *not* the same as `center=True`: torch's
   centering would pad by `n_fft//2 = 512`. Here the pad is `(1024−256)//2 = 384`, so frame
   alignment differs from the obvious implementation by 128 samples. Getting this wrong shifts every
   frame and silently degrades identity.
2. **Epsilon inside the sqrt** (`sqrt(re²+im²+1e-9)`), not added to the magnitude afterwards. Only
   matters in near-silence — which is exactly where breath and leading silence live in a 3 s
   reference.
3. **Slaney mel, not HTK.** `librosa.filters.mel` defaults to `htk=False` and `norm='slaney'`. An
   HTK-formula filterbank is a different basis and is the single most common cross-implementation
   mel divergence.
4. **Natural log with clamp 1e-5**, not log10, not `log1p`, and the clamp is applied *before* the log.
5. **Hann is periodic** (`torch.hann_window` default `periodic=True`), not symmetric.

`fmax = 12000` equals Nyquist, so the top filter is degenerate at the band edge — worth an explicit
test vector rather than assuming librosa and our implementation agree there.

---

## 2. Encoder architecture (verified against weights)

`MODEL:311-397`. The instance config (`CFG.speaker_encoder_config`) contains **only**
`{enc_dim: 1024, sample_rate: 24000}` — every other hyperparameter comes from the **class defaults**
in `CFGPY`. Unlike the traps recorded in OQ-6/OQ-7, here the defaults *are* authoritative, and all of
them are confirmed against `WT`:

| Field | Value | Confirmed by |
|---|---|---|
| `mel_dim` | 128 | `blocks.0.conv.weight [512, 128, 5]` |
| `enc_channels` | `[512, 512, 512, 512, 1536]` | as below |
| `enc_kernel_sizes` | `[5, 3, 3, 3, 1]` | `blocks.0` k=5; `mfa` k=1 |
| `enc_dilations` | `[1, 2, 3, 4, 1]` | (dilation is not visible in shapes — from `CFGPY` defaults) |
| `enc_dim` | **1024** (class default is 192 — overridden in `CFG`) | `fc.weight [1024, 3072, 1]` |
| `enc_res2net_scale` | 8 | 21 res2net convs of `[64, 64, 3]` = 7 per block × 3 blocks; 512/8 = 64 |
| `enc_se_channels` | 128 | `se_block.conv1 [128, 512, 1]`, `conv2 [512, 128, 1]` |
| `enc_attention_channels` | 128 | `asp.tdnn.conv [128, 4608, 1]`, `asp.conv [1536, 128, 1]` |

Graph (`MODEL:378-397`):

```
mel (B,T,128) → transpose → (B,128,T)
  blocks[0]  TDNN(128→512, k=5, d=1)                    → h0
  blocks[1]  SE-Res2Net(512→512, k=3, d=2)              → h1
  blocks[2]  SE-Res2Net(512→512, k=3, d=3)              → h2
  blocks[3]  SE-Res2Net(512→512, k=3, d=4)              → h3
  MFA:  cat(h1, h2, h3) = 1536ch  →  TDNN(1536→1536, k=1, d=1)
  ASP:  attentive statistics pooling → 3072ch (mean ‖ std)
  fc:   Conv1d(3072 → 1024, k=1, padding='same', padding_mode='reflect')
  squeeze(-1) → 1024-d embedding
```

**[TRAP] the MFA concatenation excludes `h0`.** `MODEL:385` is
`torch.cat(hidden_states_list[1:], dim=1)` — the initial TDNN block's output is appended to the list
but dropped from the aggregation. 3 × 512 = 1536 = `enc_channels[-1]`, which is what makes the shapes
work. An implementation that includes `h0` would need 2048 input channels and would not load these
weights — but one that silently reorders the concatenation *would* load and would be wrong.

`asp.tdnn` taking **4608 = 1536 × 3** input channels confirms the standard ECAPA attentive-pooling
form where the frame features are concatenated with their global mean and standard deviation before
the attention MLP.

---

## 3. Conditioning injection — one raw token position, no projection

`MODEL:2166-2172`. The 1024-d x-vector is inserted as **exactly one extra position in the codec-side
prefix**, between the language/think prefix ids and `[codec_pad_id, codec_bos_id]`:

```python
if speaker_embed is None:
    codec_input_emebdding = cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
else:
    codec_input_emebdding = cat([codec_input_emebdding_0,
                                 speaker_embed.view(1, 1, -1),      # ← the x-vector, raw
                                 codec_input_emebdding_1], dim=1)
```

and the talker input is then assembled (`MODEL:2181-2186`) as

```
_talker_input_embed = cat(tts_pad × (L−2), tts_bos) + codec_input_emebdding[:, :-1]
talker_input_embed  = cat(role_text_embed, _talker_input_embed)
```

| Property | Finding |
|---|---|
| Mechanism | a **token-embedding slot** in the talker's input sequence — not a prefix bias, not cross-attention, not a projection |
| Projection | **none.** ECAPA `enc_dim = 1024` equals the talker `hidden_size = 1024`, so the vector is consumed at native width |
| Normalization | **none.** No L2 norm, no scaling, no layer norm applied to the embedding |
| Dtype | cast to `talker.dtype` only (`MODEL:1963`) |
| Slot sharing | the same variable holds either the x-vector **or** a preset speaker's `get_input_embeddings()(spk_id)` (`MODEL:2095-2101`) — the x-vector literally stands in for a learned speaker-id embedding |
| Modes | injected when `x_vector_only_mode` **or** `icl_mode` (`MODEL:2103-2104`); `None` otherwise |

**The parity consequence is sharp.** Because there is no normalization anywhere between the mel and
the talker's input embedding, **any numeric drift in the mel front end or the ECAPA stack propagates
directly, at full scale, into the talker's input sequence.** There is no normalization step to
absorb a filterbank or windowing mismatch. This is why doctrine #2 keeps the speaker-conditioning
path and the speaker encoder's output layers in high precision, and it is the strongest available
argument for deriving this path's tolerance from the oracle's own floor rather than from any
inherited epsilon.

It also means the x-vector shares an embedding space with preset speaker ids — a useful diagnostic:
the distance between an enrolled x-vector and the preset speaker embeddings is a cheap sanity check
that enrollment produced something in-distribution.

---

## 4. Dispositions

| # | Item | Disposition |
|---|---|---|
| 1 | Exact mel parameters | **[RESOLVED]** §1, all literals at `MODEL:1944-1951`, nothing config-driven |
| 2 | Mel normalization / log scheme | **[RESOLVED]** Slaney-norm librosa filterbank; `log(clamp(x, 1e-5))` natural log |
| 3 | Padding / centering scheme | **[RESOLVED, TRAP]** `center=False` with manual reflect pad of **384**, not torch's 512 |
| 4 | Injection point | **[RESOLVED]** one raw token position in the codec-side prefix, no projection, no normalization |
| 5 | Encoder hyperparameters | **[RESOLVED]** class defaults, all confirmed against weight shapes; `enc_dim` overridden to 1024 |
| 6 | MFA concatenation | **[TRAP RECORDED]** excludes the initial TDNN block's output |
| 7 | Plan §2.9 geometry | **[VERIFIED]** channels/kernels/dilations/1024-d output all confirmed |

---

## 5. What remains before OQ-9 can close

The bead's third exit criterion — *"oracle fixtures for (audio → mel → embedding → injected state)
captured"* — requires a running pinned reference, which does not exist yet:
`frankentts-t-oracle-fixtures-6w9` (reference oracle environment + per-stage fixture generator) is
**still open**, as is `frankentts-t-nondet-floor-7nk`.

Per Doctrine #0 (a bead closes only when its exit criteria are actually met), **OQ-9 is left open**
and a dependency on `frankentts-t-oracle-fixtures-6w9` has been added so the graph stops advertising
it as ready. When the oracle lands, the remaining work is mechanical and fully specified by §1–§3:
dump `(waveform, mel, 1024-d embedding, assembled talker input row)` for the frozen reference set,
with the §1.1 traps as the named diagnostic checkpoints.

### Consumers (partially unblocked)

`frankentts-p1-stft-mel-2xx` (§1 is the complete parameter set — can be built now),
`frankentts-t-executable-spec-1ch` (§1–§3),
`frankentts-p1-audio-1up` (24 kHz assertion, no internal resample on this path).
