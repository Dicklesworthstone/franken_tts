# Qwen3-TTS 12Hz Codec — Executable Spec (OQ-7)

Resolves **OQ-7** (`frankentts-oq7-codec-details-kjz`). Every fact below is derived from the pinned
truth pack, not from the plan or from config metadata alone. Two independent sources were used and
cross-checked for every shape: the **pinned source** and the **pinned weights**.

Citation keys (paths relative to `docs/truth-pack/snapshots/`):

| Key | File |
|---|---|
| `MOD` | `gh/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` |
| `CFGPY` | `gh/qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py` |
| `CFG` | `hf/speech_tokenizer/config.json` |
| `MAINCFG` | `hf/config.json` |
| `INF` | `gh/qwen_tts/inference/qwen3_tts_model.py` |
| `WT` | `hf/speech_tokenizer/model.safetensors` (header parse; 496 tensors) |

**Status of every claim here: [VERIFIED]** against `MOD` + `WT` unless explicitly marked
`[UNVERIFIED]` or `[OPEN]`.

---

## 1. Identity and top-level rates

| Field | Value | Source |
|---|---|---|
| Class | `Qwen3TTSTokenizerV2Model` | `MOD:928` |
| Encoder | `Qwen3TTSTokenizerV2Encoder(MimiModel)` — **HF Mimi**, decoder branches set to `None` | `MOD:899-908` |
| Decoder | `Qwen3TTSTokenizerV2Decoder` — custom, not Mimi | `MOD:824` |
| Sample rate in/out | 24 000 / 24 000 | `CFG`, `MOD:935-936` |
| `decode_upsample_rate` | **1920** samples per code frame | `CFG`, `MOD:938` |
| `encode_downsample_rate` | **1920** | `CFG`, `MOD:939` |
| Quantizers used | **16** (`encoder_valid_num_quantizers`) | `CFG`, `MOD:983` |
| Weight dtype | **F32** for all 496 codec tensors — *not* bf16 | `WT` |

### 1.1 Hop math — CONFIRMED, and the plan's open guess is now closed

`MOD:827`: `total_upsample = np.prod(upsample_rates + upsampling_ratios)`.

```
upsample_rates      = [8, 5, 4, 3]   -> 480      (decoder.decoder.1..4 transposed convs)
upsampling_ratios   = [2, 2]         ->   4      (decoder.upsample.0/1, at latent_dim)
total                                  1920      == decode_upsample_rate   ✓
```

Plan §2.7 hedged: *"product 480; ×4 base hop ⇒ 1,920 — confirm exact hop math at pin"*. **Confirmed.**
The ×4 is `upsampling_ratios [2,2]`, applied at `latent_dim=1024` **before** the decoder conv stack —
it is a distinct pair of stages, not a property of the base hop. Promote §2.7 to [VERIFIED].

---

## 2. Decoder execution graph (per `MOD:869-884`)

All shapes cross-checked against `WT`.

| # | Stage | Op | Shape / params | Src |
|---|---|---|---|---|
| 1 | `quantizer` | SplitRVQ decode, 1 semantic + 15 acoustic | see §3 | `MOD:873, 815-821` |
| 2 | `pre_conv` | causal Conv1d 512→1024, k=3 | `[1024, 512, 3]` +bias | `MOD:839-843` |
| 3 | `pre_transformer` | 8-layer sliding-window transformer | see §4 | `MOD:876, 476-575` |
| 4 | `upsample.{0,1}` | causal ConvTranspose 1024→1024 k=2 s=2, then ConvNeXt block | `[1024,1024,2]`; dw `[1024,1,7]`, pw `[4096,1024]`/`[1024,4096]` | `MOD:845-855, 211-243` |
| 5 | `decoder.0` | causal Conv1d 1024→1536, k=7 | `[1536, 1024, 7]` | `MOD:857` |
| 6 | `decoder.1..4` | 4 upsampling blocks, rates 8/5/4/3 | see below | `MOD:638-658` |
| 7 | `decoder.5` | SnakeBeta(96) | `alpha`,`beta` `[96]` | `MOD:862` |
| 8 | `decoder.6` | causal Conv1d 96→1, k=7 | `[1, 96, 7]` | `MOD:863` |
| 9 | output | `wav.clamp(-1, 1)` | — | `MOD:884` |

**Decoder block *i*** (`MOD:638-658`): `in_dim = 1536 >> i`, `out_dim = 1536 >> (i+1)`,
`upsample_rate = upsample_rates[i]`; body = `SnakeBeta(in_dim)` →
`CausalTransConv(in_dim, out_dim, k=2·rate, s=rate)` → **three** residual units with dilations
**(1, 3, 9)**, each `SnakeBeta → conv k=7 dil=d → SnakeBeta → conv k=1`, residual-added.

| block | in→out | rate | tconv kernel | residual dims |
|---|---|---|---|---|
| `decoder.1` | 1536→768 | 8 | k=16 s=8 | 768, dil 1/3/9 |
| `decoder.2` | 768→384 | 5 | k=10 s=5 | 384, dil 1/3/9 |
| `decoder.3` | 384→192 | 4 | k=8 s=4 | 192, dil 1/3/9 |
| `decoder.4` | 192→96 | 3 | k=6 s=3 | 96, dil 1/3/9 |

**Activations** (closes an OQ-7 sub-question): the conv stack uses **SnakeBeta**
(`x + (1/β)·sin²(αx)`, with α,β **stored as logs** — `alpha=exp(alpha_param)`, `beta=exp(beta_param)`,
plus `no_div_by_zero = 1e-9`) — `MOD:578-616`. The ConvNeXt blocks use **GELU** (`MOD:223`); the
transformer MLP uses **SiLU** (`hidden_act`, `CFG`). Three different activation families in one
decoder — they are not interchangeable.

> **Kernel note (BUILD list, plan §5.3):** SnakeBeta needs `exp`, `sin`, and a reciprocal per element
> at up to 1920×24 000/s output rate. It belongs on the vectorized-transcendental exception list
> (doctrine #3) with a parity-gated switch, not on the "LLVM will autovectorize it" list.

---

## 3. Quantizer, codebooks, and the ID map

### 3.1 Structure

`SplitResidualVectorQuantizer` (`MOD:780-821`), built at `MOD:830-837` with
`dimension = codebook_dim // 2 = 256`, `input_dimension = output_dimension = codebook_dim = 512`,
`bins = codebook_size = 2048`, `n_q = 16`, `n_q_semantic = 1`, `force_projection=True`.

- `rvq_first`: **1** quantizer (the semantic code, group 0)
- `rvq_rest`: **15** quantizers (acoustic residuals, groups 1–15)
- Each: `input_proj` `[256,512,1]`, `output_proj` `[512,256,1]`, Conv1d k=1 **bias-free**
- Each codebook: `embedding_sum [2048, 256]`, `cluster_usage [2048]` — **all 16 are 2048×256** (`WT`)
- Decode sums the per-layer dequantized vectors (`MOD:721-727`), then `output_proj`; `rvq_first` and
  `rvq_rest` results are added (`MOD:818-821`)

### 3.2 [CORRECTION] `semantic_codebook_size: 4096` is dead config

`CFG` declares `"semantic_codebook_size": 4096`, but the decoder is constructed with
`bins=config.codebook_size` (**2048**) for *both* sub-quantizers (`MOD:834`), and `WT` confirms the
semantic codebook is `[2048, 256]`. **`semantic_codebook_size` is never read by this model.** Anyone
sizing the semantic table from config metadata gets it 2× wrong. Ledger this under the plan's
"config metadata lies" hazard.

### 3.3 [PORTING HAZARD] codebooks are stored unnormalized

`MOD:676-679`:

```python
embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]   # epsilon = 1e-5
quantized = F.embedding(codes, embedding)
```

The usable codebook is **not** a stored tensor — it is `embedding_sum / max(cluster_usage, 1e-5)`,
computed row-wise. The converter must materialize this at build time and record the epsilon.
Consistent with doctrine #2 (RVQ codebooks stay high-precision), the materialized table stays
F32/BF16 in `.fttsq`; quantizing it is **not** in the validated set.

### 3.4 Talker/microdecoder → codec token-space map (`MAINCFG`)

| Space | Vocab | Valid code range | Source |
|---|---|---|---|
| Talker primary (group 0) | **3072** | 0–2047 | `MAINCFG talker_config.vocab_size` |
| Microdecoder residual heads (groups 1–15) | **2048** | 0–2047 | `MAINCFG talker_config.code_predictor_config.vocab_size` |
| Codec codebooks (all 16 groups) | 2048 | 0–2047 | `WT` |

**The mapping is the identity — no offset, no shift.** The talker's extra headroom above 2048 holds
control tokens, which are simply out-of-range for the codec:

```
codec_pad_id 2148 · codec_bos_id 2149 · codec_eos_token_id 2150
codec_think_id 2154 · codec_nothink_id 2155 · codec_think_bos_id 2156 · codec_think_eos_id 2157
```

Ids 2048–2147 are unassigned in the instance config (reserved). **This also settles the plan's
"~3 MB primary head oddity" (OQ-2 note): the primary head is 3072×1024, not 2048×1024** — 3.146 M
params ≈ 3.0 MB at Q8. Hand this to `frankentts-oq2-tensor-inventory-ght`.

> **[TRAP] Do not read the class defaults.** `configuration_qwen3_tts.py` defaults are the 25 Hz/v1
> values (`num_code_groups=32`, `codec_eos_token_id=4198`, `codec_pad_id=4196`, `codec_bos_id=4197`).
> The 12 Hz instance overrides to 16 groups and the 21xx ids. Only `MAINCFG` is authoritative.

### 3.5 Padding sentinel

Batched decode pads with **−1** (`INF` wrapper `pad_sequence(..., padding_value=-1)`); `MOD:1012-1014`
computes lengths from `(audio_codes[..., 0] > -1).sum(1) * 1920` and then `clamp(min=0)` before
decoding. Our engine is single-utterance in the hot loop, but any batched path must reproduce both
the sentinel and the clamp.

---

## 4. The pre-transformer

| Field | Value | Source |
|---|---|---|
| Layers | 8, **all `sliding_attention`** | `CFGPY:116-121` |
| Sliding window | **72** | `CFG`, `MOD:309` |
| hidden / intermediate | 512 / 1024 | `WT`, `CFG` |
| Heads | 16 Q / **16 KV** → **no GQA** (`n_rep = 1`) | `CFG`, `WT` (`q,k,v` all `[1024,512]`) |
| head_dim | 64 (attention width 1024 > hidden 512) | `CFG` |
| QK-Norm | **absent** — `q_norm = k_norm = nn.Identity()` | `MOD:307-308` |
| Norm | RMSNorm, eps **1e-5**, float32 upcast | `MOD:373-388`, `CFG` |
| **LayerScale** | per-channel learned scale on **both** residual branches, init 0.01 | `MOD:394-406, 464, 470` |
| MLP | SwiGLU (`gate`/`up`/`down`, SiLU), bias-free | `MOD:357-370` |
| RoPE | plain, **theta 10 000**, `rotate_half` layout | `MOD:82-106`, `CFG` |
| Bias | attention bias-free; `input_proj`/`output_proj` **have bias** | `CFG`, `WT` |
| I/O projections | `input_proj` 1024→512, `output_proj` 512→1024 | `MOD:493-494`, `WT` |

**A third rotary kernel exists.** Plan §2.10/§5.3 names two (talker mRoPE θ=1e6; microdecoder plain
RoPE θ=1e6-family, 16-position table). The codec adds a **plain RoPE at θ=10 000 over up to
`max_position_embeddings=8000` positions** — different base *and* different range from the
microdecoder's. It must be its own conformed kernel/table; reusing the microdecoder's 16-row table
here is a silent-corruption bug.

**LayerScale is not optional decoration** — it multiplies each residual branch by a learned
per-channel vector initialized at 0.01. It has no analogue in the talker and must not be folded away
casually; folding it into `o_proj`/`down_proj` output channels is legal but must be a ledgered,
parity-proven transform.

---

## 5. Causality verdict (feeds the `streaming == batch` gate)

### 5.1 The convolutions are strictly causal

`Qwen3TTSTokenizerV2CausalConvNet` (`MOD:159-192`) left-pads by `padding = kernel_eff − stride` and
right-pads by `extra_padding`. For every stride-1 conv in the decoder, `_get_extra_padding_for_conv1d`
evaluates to **0** (verified algebraically: `n_frames = L`, `ideal_length = L`), so those convs are
pure left-padded causal. `Qwen3TTSTokenizerV2CausalTransConvNet` (`MOD:195-208`) trims
`kernel − stride` from the **right**, which is exactly the causal alignment.

**Verdict: zero lookahead in the conv stack.** Stateful ring-buffer streaming is sound.

### 5.2 [MAJOR FINDING] the official decode path is chunk-approximate beyond 300 frames

`MOD:1015` calls `self.decoder.chunked_decode(...)` with the defaults from `MOD:886`:
**`chunk_size=300`, `left_context_size=25`**. Each chunk is decoded independently with 25 frames of
left context, and `context_size * total_upsample` output samples are discarded (`MOD:894`).

The pre-transformer's effective left receptive field is **568 frames** (8 stacked layers × 71 back
per sliding window of 72) — **23× the 25 frames of context the official chunker supplies.** Even a
single layer needs 71.

Consequences, all load-bearing:

1. For utterances **≤ 300 frames (24 s)** there is exactly one chunk with `context_size = 0`, so
   `chunked_decode ≡ decode`. Our L4/L5 goldens must live here to be unambiguous.
2. For utterances **> 300 frames**, the official output is **not** equal to whole-sequence decode.
   The reference is self-inconsistent past 24 s, and the discrepancy grows with the transformer's
   memory, not with the conv receptive field.
3. Therefore `streaming == batch` must name its reference. **Recommendation for the gate design:**
   our exact target is **whole-sequence decode** (the mathematically defined object), and agreement
   with the official `chunked_decode` beyond 300 frames is a *separate, measured* Contract-B row with
   its own ledger entry — never asserted as bit-exactness. Building our streaming ring buffer to
   imitate a 25-frame-context chunker would bake an upstream artifact into our engine.
4. A correct stateful streaming implementation (full KV retention in the sliding window + conv ring
   buffers) is **strictly better** than the reference here, and that must be stated as a
   *divergence*, not a win, until it is measured — `docs/DISCREPANCIES.md` row required.

RoPE position restart across chunks is *not* a source of divergence (attention scores depend only on
relative offsets), so the truncated context is the sole mechanism.

### 5.3 the ICL reference-prefix decode — [RESOLVED for this decoder; upstream chunked path stays OPEN]

`INF:612-631`: when a voice-clone reference is present, the reference codec codes are **prepended**
to the generated codes, the concatenation is decoded as one sequence, and the leading portion is cut
in the **waveform** domain by the *proportional* rule
`cut = round(ref_len / total_len * wav_len)` rather than by `ref_len * 1920`.

Two things follow, and both need a decision before the codec streaming contract freezes:

- The codec is conditioned on reference audio, so generated-frame output depends on the reference —
  this interacts directly with `.ftvoice-cache` design and with any claim that codec decode is a
  pure function of the generated token stream.
- The proportional cut is only equal to `ref_len * 1920` when `wav_len == total_len * 1920` exactly.
  Whether the chunked path can make it inexact (off-by-a-few-samples boundary offset) is
  **[OPEN]** — filed as a follow-up rather than guessed. Tracked in §8.
- **Derived for THIS implementation (2026-08-22, frankentts-5yl recon):** our decoder is strictly
  causal and emits exactly 1,920 samples per frame with full-context streaming state (§6), so after
  decoding `total_len` frames the waveform length IS `total_len * 1920` — no chunk tail, no
  partial-frame overshoot. Substituting into the official rule:
  `cut = round(ref_len / total_len * total_len * 1920) = ref_len * 1920`, exactly. The [OPEN]
  off-by-a-few-samples concern therefore cannot arise on this decoder: streaming ICL may simply
  drop the first `ref_len * 1920` samples of the prefixed decode and remain identical to the
  reference's proportional rule. This argument covers only OUR causal full-retention path; the
  upstream CHUNKED path keeps its own [OPEN] status above.
- **[LANDED 2026-08-23, frankentts-5yl]** The primitive shipped as
  `CodecStreamingState::prime_reference` (+ `CodecCheckpoint::stream_prime_reference`): decode the
  reference through the streaming state, discard exactly `ref_len * 1920` samples, push generated
  frames as the continuation. Gate `ftts-conformance/tests/icl_prefix_decode.rs` proved both the
  §5.3 identity (primed stream == concatenated decode's tail, BIT-exact over uneven packets) and
  the `.ftvoice-cache` seam (clones of one primed snapshot == freshly primed decodes) against the
  pinned checkpoint. The reference-conditioning consequence stands: ICL codec output is NOT a pure
  function of the generated tokens — the primed state is part of the voice, which is exactly what
  the snapshot captures.


---

## 6. Ring-buffer / left-context table

Left context required per stage to produce output causally, expressed in **code frames** (1 frame =
80 ms = 1920 samples). Computed from the effective kernels above; `x N` is the sample rate at that
stage relative to the code frame rate.

| Stage | Kernel | Rate | Left ctx (frames) | Cumulative |
|---|---|---|---:|---:|
| `pre_conv` | k=3 | ×1 | 2.0 | 2.0 |
| `pre_transformer` (8 × window 72) | — | ×1 | **568.0** | 570.0 |
| `upsample.0.0` tconv | k=2 s=2 | ×2 | 0.0 | 570.0 |
| `upsample.0.1` dwconv | k=7 | ×2 | 3.0 | 573.0 |
| `upsample.1.0` tconv | k=2 s=2 | ×4 | 0.0 | 573.0 |
| `upsample.1.1` dwconv | k=7 | ×4 | 1.5 | 574.5 |
| `decoder.0` | k=7 | ×4 | 1.5 | 576.0 |
| `decoder.1` (tconv k=16 s=8 + dil 1/3/9) | — | ×32 | 2.6875 | 578.6875 |
| `decoder.2` (tconv k=10 s=5 + dil 1/3/9) | — | ×160 | 0.51875 | 579.20625 |
| `decoder.3` (tconv k=8 s=4 + dil 1/3/9) | — | ×640 | 0.128125 | 579.334375 |
| `decoder.4` (tconv k=6 s=3 + dil 1/3/9) | — | ×1920 | 0.0421875 | 579.3765625 |
| `decoder.6` | k=7 | ×1920 | 0.003125 | **579.3797** |

**Two numbers the codec engine is designed around:**

- **Conv-only left context = 11.38 code frames** (= 579.3797 − 568). This is what the causal-conv
  ring buffers must hold. Sized in native units per stage, the buffers are small: worst case is
  `decoder.1`'s dilation-9 unit needing 54 taps at ×32 rate over 768 channels.
- **Transformer left context = 568 code frames = 45.4 s** of KV at full retention (8 layers × 72
  positions × 16 heads × 64 dims × 2 (K,V); at the sliding window the *retained* state is only
  8 × 72 × 1024 × 2 = **1.18 M values**, ≈ 4.7 MB F32 / 2.4 MB BF16). The sliding window bounds the
  live state at 72 positions per layer — the 568 is the *information* horizon, not the storage.

The storage figure is what matters for streaming: **a correct stateful codec needs ≈ 2.4–4.7 MB of
transformer KV plus a few hundred KB of conv ring buffers** — trivially cache-resident, which is why
§5.2's recommendation (do it right, don't imitate the chunker) costs us nothing.

---

## 7. i32-overflow K table (permanent-law rows, plan doctrine #6)

The plan's §6.8 requires the codec's conv reduction lengths (`C_in · kernel`, whether or not im2col
is materialized) to be computed at the census and proven. Here they are.

| Op | C_in × k | **K** | U8S8 worst-case \|acc\| | i32 headroom |
|---|---|---:|---:|---:|
| **`decoder.0` conv** | 1024 × 7 | **7168** | 232 135 680 | **9.25×** |
| `decoder.1.block.{2,3,4}.conv1` | 768 × 7 | 5376 | 174 101 760 | 12.33× |
| `upsample.*.pwconv2` | 4096 × 1 | 4096 | 132 648 960 | 16.19× |
| `decoder.1.block.1` tconv | 1536 × 2 taps | 3072 | 99 486 720 | 21.59× |
| `decoder.2.block.*.conv1` | 384 × 7 | 2688 | 87 050 880 | 24.66× |
| `pre_conv` | 512 × 3 | 1536 | 49 743 360 | 43.18× |
| `decoder.3.block.*.conv1` | 192 × 7 | 1344 | 43 525 440 | 49.33× |
| `upsample.*` tconv / pwconv1 | 1024 × 1 | 1024 | 33 162 240 | 64.76× |
| `decoder.4.block.*.conv1`, `decoder.6` | 96 × 7 | 672 | 21 762 720 | 98.66× |

**Headline: the codec's worst-case K is 7168 — 2.33× the talker's worst case (`down_proj`, K = 3072),
and its i32 headroom is 9.25×, less than half the talker's ≥21×.** Plan §6.8 and doctrine #6 both
state the codec's K is "unknown until the census" and must not inherit the talker's bound; it is now
known, and it is the binding one for the whole project.

Still safe for i32 accumulation with real headroom, but this is the row that must appear in the
shipped selftest, and the AVX2 `vpmaddubsw` path needs its split-accumulate saturation proof here
first (intermediate i16 saturation, not the i32 bound, is the risk at K = 7168).

Note the ×1920 output rate stages (`decoder.4`, `decoder.6`) are the *cheapest* per-element but the
*most numerous* — 96 channels at 24 kHz. Quantizing the final conv to int8 buys little and sits
directly on the waveform; doctrine #2's "codec-boundary-sensitive layers stay high precision"
applies to `decoder.6` by name.

---

## 8. Encoder (Mimi) — enrollment build

The encoder is `transformers.MimiModel` with `upsample`, `decoder_transformer`, and `decoder` set to
`None` (`MOD:899-908`). Geometry from `CFG.encoder_config` + `WT` (225 encoder tensors):

| Field | Value |
|---|---|
| Conv stack | `[64,1,7]` → `[128,64,8]` → `[256,128,10]` → `[512,256,12]` → `[1024,512,16]` → `[512,1024,3]` |
| Ratios | `upsampling_ratios [8,6,5,4]` used as **downsampling**, applied in reverse (kernels = 2·ratio ⇒ 8,10,12,16) |
| Residual units | 1 per stage: `[C/2, C, 3]` + `[C, C/2, 1]` (compress = 2) |
| `encoder.downsample.conv` | `[512, 512, 4]` — the final ×2 |
| Total downsample | 4 · 5 · 6 · 8 · 2 = **1920** ✓ matches `encode_downsample_rate` |
| `encoder_transformer` | 8 layers, hidden 512, **8 heads / 8 KV**, MLP `fc1 [2048,512]` / `fc2 [512,2048]`, **LayerNorm with bias** (not RMSNorm), LayerScale on both branches, sliding window **250** |
| Activation | **GELU** (`hidden_act`) — differs from the decoder's SiLU |
| Causal conv | `use_causal_conv: true` |
| Quantizer | **32** quantizers (1 semantic + 31 acoustic), all `[2048, 256]`; only the **first 16** are read (`MOD:983`) |
| Codebook form | `embed_sum` / `cluster_usage` (+ an `initialized` flag) — same materialization hazard as §3.3, different tensor names |

The encoder transformer is architecturally *different* from the decoder's (LayerNorm-with-bias vs
RMSNorm, 8 vs 16 heads, window 250 vs 72, GELU vs SiLU). They are two separate kernels; the
enrollment build cannot reuse the decode-path transformer.

**[VERIFIED 2026-08-23]** The exact forward was re-asserted against the PINNED runtime source —
`transformers==4.57.3`, `models/mimi/modeling_mimi.py` (the exact version the oracle venv pins).
Operator order, top to bottom:

1. **SEANet conv stack** (`MimiEncoder`): stem Conv1d `[64,1,7]` with **no following activation**;
   then per ratio in reversed(`upsampling_ratios`) → `num_residual_layers` × `MimiResnetBlock`,
   then **ELU**, then the stage downsample Conv1d (k=2·ratio, stride=ratio).
2. **ResnetBlock internals are PRE-ACTIVATION**: `ELU → Conv1d(k=3, d=growthʳ)` then
   `ELU → Conv1d(k=1)`; hidden = dim/compress (=2); shortcut = **Identity** (config
   `use_conv_shortcut` defaults False); output = `shortcut(x) + block(x)` — additive residual,
   **no trailing activation**.
3. After the conv stack: `encoder_transformer` (8 layers — the GELU/LayerNorm-with-bias/window-250
   transformer in the table above), channels transposed for attention and back.
4. **Model-level `downsample` comes AFTER the transformer** (this ordering is the classic port
   trap): Conv1d `[512,512,4]`, stride 2, **bias=False**, `pad_mode="replicate"` — unlike every
   stack conv, which uses the config pad mode with the asymmetric arithmetic below.
5. `quantizer.encode` (first 16 of 32 codebooks) → codes transposed to frames-major.

Padding arithmetic (every `MimiConv1d`, weight-normed): effective kernel `k̂=(k−1)·d+1`;
`padding_total=k̂−stride`; `padding_right=total//2`; `padding_left=total−right`; plus
ceil-to-frame extra input padding trimmed after the conv (`_get_extra_padding_for_conv1d`), with
an optional streaming `padding_cache` carrying per-layer left-context state — the hook an
incremental encoder port must implement.

Source citations: `modeling_mimi.py@v4.57.3` — SEANet construction `L444-470`, resnet block
`L409-441`, conv padding `L227-246`, extra-padding `L262-272`, model assembly + `_encode_frame`
order `L1400-1469`. Shapes above were already solid from `WT`; the operator order is now pinned
too. This resolves the dependency note on `frankentts-oq15-oracle-pins-wjc` and lifts the
enrollment-build-only restriction on these rows: they may graduate to a kernel contract once the
encoder port (snt) lands its roundtrip gate.

**[LANDED 2026-08-23]** The encoder port shipped as `ftts-model-qwen/src/speech_encoder.rs` with a
gate STRONGER than the roundtrip: `ftts-conformance/tests/speech_encoder_oracle.rs` compares our
codes EXACTLY (944/944 ids, two synthetic corpora incl. a ceil-padding case) against a pinned
CPU-fp32 reference capture (`scripts/capture_speech_encoder_oracle.py`). Two source findings the
port added to this section's record: (a) with `use_causal_conv: true` every conv pads LEFT-only
(`padding_total` left, ceil-to-frame extra right) — the asymmetric split described above is the
dead non-causal branch; (b) the encoder transformer's sliding window 250 is INERT on the
eager/SDPA oracle path (`create_causal_mask` never consults it; only flash attention applies it) —
the exact contract is plain causal.

---

## 9. Dispositions and follow-ups

### Corrections to propagate

| # | Item | Disposition |
|---|---|---|
| 1 | §2.7 "confirm exact hop math at pin" | **[VERIFIED]** — 480 × 4 = 1920, ×4 from `upsampling_ratios [2,2]` |
| 2 | §2.7 codec table | **[EXTEND]** — add `latent_dim 1024`, `decoder_dim 1536`, `codebook_dim 512`, LayerScale, SnakeBeta, no-GQA, no-QK-Norm |
| 3 | `semantic_codebook_size: 4096` | **[DEAD CONFIG]** — decoder uses 2048; do not size from it |
| 4 | §2.10 / §5.3 "two rotary kernels" | **[CORRECTED]** — there are **three**; the codec's is plain RoPE θ=10 000 over 8000 positions |
| 5 | §6.8 codec overflow K "unknown until census" | **[RESOLVED]** — K = 7168, headroom 9.25×; now the project's binding worst case |
| 6 | OQ-2's "~3 MB primary head oddity" | **[RESOLVED]** — talker vocab is 3072, not 2048 |
| 7 | Codec weights assumed bf16 | **[CORRECTED]** — shipped F32 |

### Filed as follow-ups (not guessed)
- **[VERIFIED]** §8 — Mimi encoder operator order asserted against pinned `transformers==4.57.3`
  (see §8 verification block); ELU pre-activation in SEANet, GELU only in the transformer,
  transformer-then-downsample ordering confirmed.
- **[OPEN]** §5.3 — whether the ICL proportional waveform cut can be inexact under chunking.
- **[DISC required]** §5.2 — our whole-sequence-exact streaming vs the official 25-frame-context
  chunker beyond 300 frames.

### Consumers unblocked

`frankentts-p1-codec-hu7` (decoder + ring buffers + streaming gate), `frankentts-p1-codec-encoder-snt`
(encoder, with the §8 caveat), `frankentts-t-census-44a` (§6 buffer table + §7 K rows),
`frankentts-t-executable-spec-1ch` (§2–§5 fold into the executable spec),
`frankentts-oq2-tensor-inventory-ght` (§3.4 primary-head resolution).
