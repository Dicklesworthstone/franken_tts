# OQ-13 — The community GGML kept-high-precision set, reconciled against our recipe

**Bead:** `frankentts-oq13-ggml-recipe-bzt` · **Resolved by:** AzureThrush · **Date:** 2026-08-06

**Headline:** the community Q8 conversion is **far more conservative than plan §6.3 assumes in two
places**, and more aggressive in two others.

- ⚠ **"Codec Q8" is wrong as stated.** In their Q8_0 codec, **every actual convolution kernel stays
  F16** and every Snake activation parameter stays F32. Only the codec's *transformer blocks* and its
  *pointwise* (1×1) convs are Q8. Our plan's "codec runs Q8 with high-precision codebooks" would
  quantize the whole conv stack — that is **beyond validated prior art** and needs its own gate.
- ⚠ **They keep the ENTIRE speaker encoder high** (F16 weights / F32 biases, zero Q8). Our plan only
  committed to "the speaker encoder's output layers."
- ✅ They **do** quantize things our plan flagged as risky: the **15 microdecoder heads**, the
  **talker codec head**, and the **622 MB text embedding**. That is headroom we can claim by
  measurement rather than assume.
- ✅ Exact agreement on the load-bearing conservatives: **all norms F32 (including QK-Norm)** and
  **all 16 RVQ codebooks F32**.

Status: **[VERIFIED-MEASURED]** — read from the actual GGUF bytes, not transcribed from prose.

---

## 0. Provenance and method

| | |
|---|---|
| Repo | `Serveurperso/Qwen3-TTS-GGUF` @ `e0f336a048a3de02b29b8ad92969217d9ecffe3e` |
| Talker file | `qwen-talker-0.6b-base-Q8_0.gguf` — **993 MB**, `general.file_type = Q8_0`, `general.name = Qwen3-TTS-12Hz-0.6B-base` |
| Codec file | `qwen-tokenizer-12hz-Q8_0.gguf` — `general.architecture = qwen3-tts-tokenizer` |
| Tool | `scripts/oq13_gguf_tensor_types.py` (stdlib; HTTP Range reads the GGUF header only — ~1–16 MB, not the ~1 GB file) |

**Why this repo.** Plan §2.2 cites "community Q8 ≈ 993 MB talker + 291 MB codec" as [REPORTED].
`qwen-talker-0.6b-base-Q8_0.gguf` is **exactly 993 MB** — this is the conversion the plan was already
referring to, now pinned and read rather than quoted. It targets our exact checkpoint (0.6B Base).

> `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` (the ggml-org org's own conversion) exists but is the
> **1.7B** variant, not ours. Not used as the primary evidence here. Cross-checking it is cheap and
> worth doing if any divergence below is contested — noted, not done.

Reproduce:

```bash
python3 scripts/oq13_gguf_tensor_types.py Serveurperso/Qwen3-TTS-GGUF qwen-talker-0.6b-base-Q8_0.gguf
python3 scripts/oq13_gguf_tensor_types.py Serveurperso/Qwen3-TTS-GGUF qwen-tokenizer-12hz-Q8_0.gguf
```

---

## 1. The talker file (478 tensors: 266 Q8_0 / 175 F32 / 37 F16)

| Component | Q8_0 | F32 | F16 |
|---|---:|---:|---:|
| `talker.*` | 201 | 115 | 0 |
| `code_pred.*` (the microdecoder) | 65 | 21 | 0 |
| `spk_enc.*` (ECAPA speaker encoder) | **0** | 39 | 37 |

### 1.1 Quantized to Q8_0

| Tensor (collapsed over layers) | Count | Dims |
|---|---:|---|
| `talker.blk.N.attn_{q,k,v,output}.weight` | 28 ea | 1024×2048 / 1024×1024 / 1024×1024 / 2048×1024 |
| `talker.blk.N.ffn_{gate,up,down}.weight` | 28 ea | 1024×3072 / 1024×3072 / 3072×1024 |
| **`talker.codec_head.weight`** | 1 | **1024×3072** |
| `talker.codec_embd.weight` | 1 | 1024×3072 |
| **`talker.text_embd.weight`** | 1 | **2048×151936** (the ~622 MB cold embedding) |
| `talker.text_proj.fc1.weight` / `fc2.weight` | 1 ea | 2048×2048 / 2048×1024 |
| `code_pred.blk.N.attn_{q,k,v,output}.weight` | 5 ea | same shapes as talker |
| `code_pred.blk.N.ffn_{gate,up,down}.weight` | 5 ea | same shapes as talker |
| **`code_pred.lm_head.N.weight`** | **15** | **1024×2048** |
| **`code_pred.codec_embd.N.weight`** | **15** | **1024×2048** |

### 1.2 Kept high precision

| Tensor | Count | Type | Note |
|---|---:|---|---|
| `talker.blk.N.attn_norm` / `ffn_norm` | 28 ea | F32 | 1024 |
| **`talker.blk.N.attn_q_norm` / `attn_k_norm`** | 28 ea | **F32** | 128 — **QK-Norm**, independently confirming OQ-5 |
| `talker.output_norm` | 1 | F32 | |
| `talker.text_proj.fc1.bias` / `fc2.bias` | 1 ea | F32 | **biases high, weights Q8** |
| `code_pred.blk.N.{attn,ffn}_norm`, `attn_q_norm`, `attn_k_norm` | 5 ea | F32 | |
| `code_pred.output_norm` | 1 | F32 | |
| **entire `spk_enc.*`** | 76 | **F16 weights / F32 biases** | **no Q8 anywhere** |

### 1.3 Two independent confirmations of OQ-5

Falling out of this dump, from a completely different artifact than the Python source:

- `code_pred.codec_embd.*` is **15** tensors of **1024×2048** and `talker.codec_embd` is a
  **separate** 1024×**3072** tensor — exactly OQ-5's finding that sequence position 1 embeds the
  primary code through the **talker's** 3072-vocab table while positions 2–15 use the 15 predictor
  tables.
- **There is no `small_to_mtp_projection` tensor at all** — confirming OQ-5 §3's conclusion that it
  resolves to `nn.Identity()` for this checkpoint and must be recorded as legitimately absent rather
  than MISSING in the WeightsManifest census.

---

## 2. The codec file (398 tensors: 238 F32 / 50 F16 / 110 Q8_0)

| Component | Q8_0 | F32 | F16 |
|---|---:|---:|---:|
| `tok_dec.*` | 62 | 156 | 35 |
| `tok_enc.*` | 48 | 82 | 15 |

### 2.1 Quantized to Q8_0 — transformer blocks and pointwise convs ONLY

```
tok_dec.pre_tfm.input_proj / output_proj                    1024x512 / 512x1024
tok_dec.pre_tfm.blk.N.attn_{q,k,v,output}, ffn_{gate,up,down}   8 layers, hidden 512, ffn 1024
tok_dec.upsample.N.pwconv1 / pwconv2                        1024x4096 / 4096x1024   (1x1 = matmuls)
tok_enc.blk.N.attn_{q,k,v,output}, ffn_{up,down}            8 layers, hidden 512, ffn 2048
```

### 2.2 Kept high precision — everything else

- **All genuine convolutions F16.** Of 45 conv weight tensors, **41 are F16**; the only 4 "conv"
  tensors that are Q8 are the `upsample.*.pwconv{1,2}` 1×1 pointwise convs, which are matmuls in
  disguise. Includes the whole transposed-conv upsampling stack and the final output conv.
- **All Snake activation parameters F32** — 58 tensors of `snake.alpha` / `snake.beta` /
  `res.N.act{1,2}.{alpha,beta}`.
- **All conv biases F32.**
- **All 32 RVQ codebooks F32**, each **256×2048**:

| Codebook group | Count | Meaning |
|---|---:|---|
| `tok_dec.vq_first.0.codebook` | 1 | decoder, code group 0 |
| `tok_dec.vq_rest.0..14.codebook` | 15 | decoder, code groups 1–15 |
| `tok_enc.vq_semantic.0.codebook` | 1 | encoder, **semantic** |
| `tok_enc.vq_acoustic.0..14.codebook` | 15 | encoder, **acoustic** |

- All VQ input/output projections F32 (`1×512×256` / `1×256×512`).

### 2.3 Codec architecture facts owed to OQ-7 (free, from the same dump)

These were not the question but are directly useful and are handed to `oq7-codec-details-kjz`:

- **The semantic/acoustic split is explicit in the tensor names.** `vq_semantic` (1) + `vq_acoustic`
  (15) on the encoder, mirrored by `vq_first` (1) + `vq_rest` (15) on the decoder. **Code group 0 is
  the semantic code; groups 1–15 are acoustic.** That is the semantic-ID mapping OQ-7 asks for, and
  it lines up exactly with the talker predicting group 0 and the microdecoder the other 15.
- **RVQ operates in a 256-d projected space**, not 512: codebooks are 256×2048 with 512→256 input
  and 256→512 output projections. Plan §2.7 does not state this.
- **The decoder is BigVGAN-style with Snake activations**, not a plain conv stack. Channel ladder
  `1536 → 768 → 384 → 192 → 96 → 1`; transposed-conv kernels **16, 10, 8, 6** — i.e. `kernel = 2 ×
  stride` for strides **8, 5, 4, 3**, confirming plan §2.7's upsample `[8,5,4,3]` from the weights.
- `tok_dec.pre_tfm` is **8 layers, hidden 512, ffn 1024**, matching plan §2.7's decoder geometry;
  `tok_enc.blk` is 8 layers, hidden 512, **ffn 2048** (encoder and decoder MLPs differ).
- **Snake activations are a kernel we have not planned for.** `x + (1/α)·sin²(αx)`-class
  activations are transcendental-heavy and per-channel parameterised — this is exactly the §7.2
  "vectorized polynomial transcendental" exception, and it is on the codec hot path.

---

## 3. Reconciliation: our recipe vs theirs

| Tensor class | Plan §6.3 (ours) | Community GGML | Verdict |
|---|---|---|---|
| Talker attention + MLP GEMMs | int8 | Q8_0 | ✅ agree |
| Microdecoder attention + MLP | int8 | Q8_0 | ✅ agree |
| All norms | high precision | F32 | ✅ agree |
| QK-Norm | high precision (if present) | F32 | ✅ agree — and confirms presence |
| RVQ codebooks | high precision | F32, all 32 | ✅ agree, strongly |
| Microdecoder 15 heads | "beyond validated set — measured kill-switch" | **Q8_0** | 🟢 **we are stricter** — headroom |
| Talker codec head (1024×3072) | same caveat | **Q8_0** | 🟢 **we are stricter** — headroom |
| Text embedding (622 MB) | BF16 default; Q8 a *gated experiment* | **Q8_0** | 🟢 prior art supports the experiment |
| Microdecoder 15 embeddings | (plan protects "acoustic codebooks") | **Q8_0** | 🟡 **not the same thing** — see §3.1 |
| `text_proj` fc1/fc2 | unstated | **weights Q8, biases F32** | 🟡 new detail to encode |
| Speaker encoder | "output layers high precision" | **ENTIRE encoder F16/F32** | 🔴 **they are stricter** |
| Codec convolutions | "codec Q8" | **ALL convs F16** | 🔴 **they are stricter — the big one** |
| Codec Snake params | unstated | **F32** | 🔴 must be added to our kept-high set |
| Codec transformer blocks | "codec Q8" | Q8_0 | ✅ agree |

### 3.1 A distinction the plan blurs

Plan §6.3 says "*acoustic codebooks* stay high-precision" while differentiating the text embedding.
The GGUF shows these are **three** different objects, not two:

1. **RVQ codebooks** (`vq_*.codebook`, 256×2048) — kept **F32** by both of us. ✅
2. **Microdecoder per-depth input embeddings** (`code_pred.codec_embd.N`, 1024×2048) — quantized
   **Q8** by them. These embed *already-sampled code ids* into the microdecoder's hidden space; they
   are not the codec's reconstruction codebooks. Our plan's protective language reads as though it
   covers them; **it should not**.
3. **Text embedding** (2048×151936) — Q8 by them.

Recommend the plan/spec adopt these three names explicitly so the recipe cannot be misread.

---

## 4. Risks flowing to the Phase-2 staged-lever beads

Per the bead's charter — *"any tensor THEY keep high that WE planned to quantize is OUR risk to
retire by measurement, never assumed safe."* Two such risks, both filed:

**R1 — Codec convolutions (`frankentts-p2-codec-q8-3vw`).** HIGH. Our "codec Q8" would quantize the
entire conv/transposed-conv stack; the community conversion quantizes **none** of it. Convolution
kernels here are small (kernel 1–16, channels 96–1536), so Q8 buys little footprint while sitting on
the perceptually most sensitive path (waveform reconstruction). **Recommendation: default the codec
conv stack to F16 and treat conv-Q8 as a separately gated experiment** with its own kill-switch and
listening gate, exactly as int4 is treated. Q8 the codec's transformer blocks and pointwise convs —
that part is validated.

**R2 — Speaker encoder (`frankentts-p2-int8-talker-micro-upg` / enrollment beads).** LOW cost to
retire. They keep 100 % of it high; we committed only to output layers. The encoder is
**enrollment-only and offline**, so quantizing it buys nothing on the hot path. **Recommendation:
keep the whole speaker encoder high precision** — matches prior art, costs nothing that matters, and
removes an identity-risk surface entirely.

**Headroom (the other direction), for `frankentts-p2-int8-talker-micro-upg`:** the 15 microdecoder
heads, the talker codec head, and the text embedding are all Q8 in shipped community artifacts with
real users. That does **not** make them safe by assertion — it makes them *plausible enough to be
worth measuring first* among our staged levers, and it means a negative result from us would be a
divergence from prior art worth ledgering rather than a quiet default.

## 5. Scope limits

- **This is a type census, not a quality result.** That the community ships Q8 on a tensor is
  evidence it is *tolerable to some users*, not that it passes our §9.4 listening protocol. Every
  lever still goes through our own gates; this document only re-ranks what to try and flags where we
  would be exceeding prior art.
- **One conversion, one revision.** Findings are bound to `Serveurperso/Qwen3-TTS-GGUF @ e0f336a`.
  The ggml-org 1.7B conversion was not cross-checked (§0).
- **Q8_0 ≠ our int8.** GGUF `Q8_0` is block-wise (32 elements, one F16 scale, symmetric); our recipe
  is **per-output-channel** int8 with i32 accumulation. Their kept-high *set* transfers as evidence;
  their quantization *format* does not, and error magnitudes are not directly comparable.
- Nothing here was listened to.
