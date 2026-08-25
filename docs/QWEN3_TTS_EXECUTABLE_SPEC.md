# Qwen3-TTS Executable Specification (franken_tts)

Bead: `frankentts-t-executable-spec-1ch` (P0). This document is the single implementation
source for the pinned model graph: an agent implementing a kernel must never need to re-read
the Python. Every load-bearing constant, tensor name, and structural claim carries a citation
to a truth-pack file (`docs/truth-pack/…`) or to a pinned source file. Items that could not be
verified from available material are marked **[UNVERIFIED]** — never guessed.

## §0 Scope, authority, pins

- Weights pin: `Qwen/Qwen3-TTS-12Hz-0.6B-Base@5d83992436eae1d760afd27aff78a71d676296fc`
  (source: docs/truth-pack/PIN_RECORD.md).
- Source pin: `QwenLM/Qwen3-TTS@022e286b98fbec7e1e916cb940cdf532cd9f488e`
  (source: docs/truth-pack/PIN_RECORD.md). The GitHub code (~2026-03-17) is ~7 weeks newer than
  the weights (~2026-01-29); behavioral questions are answered by the *code* at its pin executed
  against the *weights* at theirs (source: docs/truth-pack/PIN_RECORD.md).
- Snapshot paths cited below as `gh/…` and `hf/…` resolve under
  `docs/truth-pack/snapshots/`. Integrity is anchored by `docs/truth-pack/MANIFEST.sha256`;
  weight integrity by LFS oids in `docs/truth-pack/WEIGHTS.lfs.json`.
- Rust cross-check references point at `crates/ftts-model-qwen/src/`. Where doc and code
  disagree, see §16 (Discrepancies), not silent preference.
- Oracle runtime environment: Python 3.13.9, `torch==2.7.1`, `torchaudio==2.7.1`,
  `librosa==0.11.0`, `soundfile==0.13.1`, `qwen-tts==0.1.1`, `transformers==4.57.3`,
  `accelerate==1.12.0`; CPU fixture capture uses eager attention, FP32 parameters,
  `device_map=cpu` (source: docs/truth-pack/PIN_RECORD.md).

## §1 Top-level structure

One checkpoint, two safetensors files, 974 tensors, 2,511,515,780 payload bytes:

| File | Tensors | Dtype | Payload bytes |
|---|---:|---|---:|
| `model.safetensors` | 478 | BF16 | 1,829,286,016 |
| `speech_tokenizer/model.safetensors` | 496 | F32 | 682,229,764 |

(source: docs/truth-pack/OQ2_TENSOR_INVENTORY.md). No shards, no `*.index.json`
(source: docs/truth-pack/FACT_DISPOSITIONS.md, C-6).

Five components in the synthesis path:

1. **Text path** — byte-level BPE tokenizer → padded text embedding `[151936, 2048]` →
   two-layer biased projection to talker width 1024.
2. **Talker** — 28-layer GQA transformer (hidden 1024), autoregressive over 80 ms frames;
   primary head width 3072.
3. **Microdecoder** (`code_predictor`) — 5-layer GQA transformer that expands each talker step
   into 16 code groups via 15 sequential residual steps.
4. **Codec decoder** (speech tokenizer, decode side) — SplitRVQ → causal conv → 8-layer
   sliding-window transformer → upsample cascade → BigVGAN conv stack → 24 kHz PCM.
5. **Speaker encoder** (enrollment only; BF16, in the main checkpoint manifest) — mel-128 ECAPA-style,
   enc_dim 1024 (source: docs/truth-pack/OQ2_TENSOR_INVENTORY.md;
   docs/truth-pack/FACT_DISPOSITIONS.md §2.9).

## §2 Data structures (pinned geometry)

### §2.1 Talker (`config.json → talker_config`)

| Field | Value |
|---|---|
| `num_hidden_layers` | 28 |
| `hidden_size` | 1024 |
| `intermediate_size` | 3072 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 8 / 128 (attention width 2048 > hidden; GQA group 2) |
| `attention_bias` | false (no QKV/O biases) |
| `rms_norm_eps` | 1e-06 |
| `rope_theta` / `rope_scaling` | 1000000 / `{interleaved: true, mrope_section: [24,20,20], rope_type: "default"}` |
| `sliding_window` / `use_sliding_window` | null / false |
| `max_position_embeddings` | 32768 |
| `vocab_size` | **3072** (primary code head) |
| `text_vocab_size` / `text_hidden_size` | 151936 / 2048 |
| `num_code_groups` | 16 |
| `position_id_per_seconds` | 13 (non-operative metadata — see §8.1 and §16 D-6) |

(source: hf/config.json:135-164).

### §2.2 Microdecoder (`talker_config.code_predictor_config`)

| Field | Value |
|---|---|
| `num_hidden_layers` | 5 |
| `hidden_size` | 1024 (= talker hidden) |
| `intermediate_size` | 3072 (SwiGLU) |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 8 / 128 (GQA, width 2048) |
| `num_code_groups` | 16 ⇒ 15 embeddings, 15 heads, 15 sequential steps |
| `vocab_size` | 2048 (residual heads) |
| `layer_types` | `["full_attention"] × 5` — no sliding attention anywhere |
| `sliding_window` / `use_sliding_window` | null / false |
| `rope_theta` / `rope_scaling` | 1000000 / null (plain RoPE) |
| `rms_norm_eps` | 1e-06 |
| `attention_bias` | false |
| stale sampling fields | `temperature 1.0`, `do_sample false`, `repetition_penalty 1.0`, `top_k 50`, `top_p 1.0` — **inert**, overridden at call site (§11) |

(source: hf/config.json:22-111; docs/truth-pack/OQ5_MICRODECODER_WIRING.md §0).

### §2.3 Codec decoder (`speech_tokenizer/config.json → decoder_config`)

| Field | Value |
|---|---|
| `num_hidden_layers` (pre-transformer) | 8 |
| `hidden_size` | 512 |
| `intermediate_size` | 1024 |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 16 / 64 — **MHA, no GQA** |
| `latent_dim` / `codebook_dim` | 1024 / 512 |
| `decoder_dim` | 1536 |
| `layer_scale_initial_scale` | 0.01 |
| `rms_norm_eps` | 1e-05 |
| `rope_theta` | 10000 (plain RoPE) |
| `sliding_window` | 72 |
| `max_position_embeddings` | 8000 |
| `hidden_act` | silu (transformer MLP) |
| `upsample_rates` / `upsampling_ratios` | [8,5,4,3] / [2,2] |
| `codebook_size` / `semantic_codebook_size` | 2048 / 4096 (the 4096 is dead config — §16 D-5) |
| `num_quantizers` / `num_semantic_quantizers` | 16 / 1 |
| top level `decode_upsample_rate` / `encode_downsample_rate` | 1920 / 1920 |

(source: hf/speech_tokenizer/config.json:1-44).

### §2.4 Frame

16 code groups/frame at 12.5 fps ⇒ 80 ms/frame; 15 residual steps/frame; 1920 samples/frame at
24 kHz (source: docs/truth-pack/EXECUTION_CENSUS.json `.frame`).

### §2.5 Component byte census (BF16 unless noted)

| Component | Bytes | Tensors |
|---|---:|---:|
| talker body (28 layers + norms) | 880,934,912 (+ codec_embedding 6,291,456 + primary head 6,291,456 = 893,517,824) | 309 (+2) |
| microdecoder body / per-depth embeddings / per-depth heads | 157,311,488 / 62,914,560 / 62,914,560 | 56 / 15 / 15 |
| text path (cold embedding + projection) | 622,329,856 + 12,589,056 | 1 + 4 |
| codec decoder (F32) | 457,292,548 | 271 |
| codec encoder (F32, enrollment-only) | 224,937,216 | 225 |
| speaker encoder | 17,708,672 | 76 |

(source: docs/truth-pack/EXECUTION_CENSUS.json `.components`).

Microdecoder hot working set: traffic floor **110,100,480 B Q8** (body 78,643,200 reread by the
sequential 15-step loop + 15 depth heads 31,457,280, one selected per depth); the 15 depth
embedding tables add 31,457,280 B of pack *footprint* but only one 1,024-byte row each is
gathered per frame (15,360 B/frame). The shared talker `[3072,1024]` codec embedding supplies
position-1 rows and is an explicit shared dependency (source:
docs/truth-pack/OQ2_TENSOR_INVENTORY.md; docs/truth-pack/EXECUTION_CENSUS.json
`.microdecoder_hot_working_set`).

## §3 Tensor-name intel (canonical names verbatim)

Names are authoritative from the pinned safetensors headers; missing/extra/mismatched entries
must fail loading (source: docs/truth-pack/OQ2_TENSOR_INVENTORY.md). Preserve upstream spelling
exactly, including oddities.

Text path (note: the text embedding lives under the `talker.` prefix):
```
talker.model.text_embedding.weight            [151936, 2048] BF16
talker.text_projection.linear_fc1.weight      [2048, 2048]  BF16   (bias=True)
talker.text_projection.linear_fc1.bias        [2048]
talker.text_projection.linear_fc2.weight      [1024, 2048]  BF16   (bias=True)
talker.text_projection.linear_fc2.bias        [1024]
```
(source: docs/truth-pack/OQ2_TENSOR_INVENTORY.md; docs/truth-pack/TENSOR_INVENTORY.json).

Talker body, per layer `L ∈ [0,27]` (no biases anywhere inside the repeated blocks):
```
talker.model.layers.L.input_layernorm.weight            [1024]
talker.model.layers.L.post_attention_layernorm.weight   [1024]
talker.model.layers.L.self_attn.q_proj.weight           [2048, 1024]
talker.model.layers.L.self_attn.k_proj.weight           [1024, 1024]
talker.model.layers.L.self_attn.v_proj.weight           [1024, 1024]
talker.model.layers.L.self_attn.o_proj.weight           [1024, 2048]
talker.model.layers.L.self_attn.q_norm.weight           [128]    # RMSNorm over head_dim only
talker.model.layers.L.self_attn.k_norm.weight           [128]
talker.model.layers.L.mlp.gate_proj.weight              [3072, 1024]
talker.model.layers.L.mlp.up_proj.weight                [3072, 1024]
talker.model.layers.L.mlp.down_proj.weight              [1024, 3072]
talker.model.norm.weight                                [1024]
talker.model.codec_embedding.weight                     [3072, 1024]   # talker's OWN codec embedding
talker.codec_head.weight                                [3072, 1024]   # bias-free
```
(source: docs/truth-pack/TENSOR_INVENTORY.json; docs/truth-pack/FACT_DISPOSITIONS.md E-1, E-2).
The outer wrapper is NOT bias-free: both `linear_fc1`/`linear_fc2` carry biases by construction
(source: docs/truth-pack/FACT_DISPOSITIONS.md E-1, citing gh/qwen_tts/core/models/modeling_qwen3_tts.py:1575-1577).

Microdecoder, per depth index `j ∈ [0,14]` (index serves code `j+1` — see §10):
```
talker.code_predictor.model.layers.N.{input_layernorm,post_attention_layernorm}.weight   [1024]
talker.code_predictor.model.layers.N.self_attn.{q,k,v,o}_proj.weight                      (same shapes as talker layer 0)
talker.code_predictor.model.layers.N.self_attn.{q,k}_norm.weight                          [128]
talker.code_predictor.model.layers.N.mlp.{gate,up,down}_proj.weight
talker.code_predictor.model.norm.weight                                    [1024]
talker.code_predictor.model.codec_embedding.j.weight                       [2048, 1024]  ×15
talker.code_predictor.lm_head.j.weight                                     [2048, 1024]  ×15, bias=False
```
`small_to_mtp_projection` is **legitimately absent**: it is `nn.Identity()` because
predictor hidden (1024) == talker hidden (1024); a converter expecting the tensor must fail the
manifest census as MISSING otherwise (source: docs/truth-pack/OQ5_MICRODECODER_WIRING.md §3;
docs/truth-pack/OQ2_TENSOR_INVENTORY.md).
Naming traps: upstream uses `codec_embedding` for **two different tables** —
`talker.model.codec_embedding` (vocab 3072, the talker's own) and
`talker.code_predictor.model.codec_embedding.j` (vocab 2048, per-depth). They are never
interchangeable (source: docs/truth-pack/OQ5_MICRODECODER_WIRING.md §1, §3).

Codec decoder (F32), structural names:
```
decoder.quantizer.rvq_first.input_proj.weight            [256, 512, 1]
decoder.quantizer.rvq_first.output_proj.weight           [512, 256, 1]
decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum   [2048, 256]
decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage   [2048]
decoder.quantizer.rvq_rest.vq.layers.K._codebook.{embedding_sum,cluster_usage}   K ∈ [0,14]
decoder.pre_conv.conv.{weight,bias}                       [1024, 512, 3] / [1024]
decoder.pre_transformer.input_proj.{weight,bias}          [512, 1024] / [512]     # biased
decoder.pre_transformer.layers.N.{input_layernorm,post_attention_layernorm}.weight [512]
decoder.pre_transformer.layers.N.self_attn.{q,k,v,o}_proj.weight   [1024,512]/[512,1024]-class, bias-free
decoder.pre_transformer.layers.N.self_attn_layer_scale.scale       [512]   # init 0.01
decoder.pre_transformer.layers.N.mlp_layer_scale.scale             [512]   # init 0.01
decoder.pre_transformer.layers.N.mlp.{gate,up,down}_proj.weight    (512↔1024)
decoder.upsample.S.conv.conv.{weight,bias}                S ∈ {0,1}, tconv 1024→1024 k=2r s=r (r=2)
decoder.upsample.S.convnext.dwconv.conv.{weight,bias}     depthwise k=7 groups=dim
decoder.upsample.S.convnext.norm.{weight,bias}            LayerNorm(dim, eps=1e-6)
decoder.upsample.S.convnext.pwconv1.{weight,bias}         Linear dim→4·dim (biased)
decoder.upsample.S.convnext.pwconv2.{weight,bias}         Linear 4·dim→dim (biased)
decoder.upsample.S.convnext.gamma                         [dim]      # ConvNeXt gate, init 1e-6
decoder.decoder.0.conv.{weight,bias}                      [1536, 1024, 7] / [1536]
decoder.decoder.B.block.0.{alpha,beta}                    SnakeBeta logs, B ∈ [1,4]
decoder.decoder.B.block.1.conv.{weight,bias}              ConvNeXt-in-block depthwise k=7
decoder.decoder.B.block.U.act{1,2}.{alpha,beta}           residual-unit SnakeBeta logs, U ∈ [2,4]
decoder.decoder.B.block.U.conv1.conv.{weight,bias}        k=7 dilation ∈ {1,3,9}
decoder.decoder.B.block.U.conv2.conv.{weight,bias}        k=1
decoder.decoder.5.{alpha,beta}                            final SnakeBeta logs [96]
decoder.decoder.6.conv.{weight,bias}                      [1, 96, 7] / [1]
```
(source: docs/truth-pack/TENSOR_INVENTORY.json; gh/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:211-243, 619-658, 839-865).
Note there are **no** `decoder.pre_transformer.layers.N.self_attn.{q,k}_norm.*` tensors — the
codec transformer has no QK-Norm (absence confirmed against the inventory; consistent with the
bead finding that `q_norm`/`k_norm` are `nn.Identity` there — source: bead
`frankentts-t-executable-spec-1ch` comment 2026-08-06, cc_11). Decoder codebooks store
`embedding_sum` + `cluster_usage` while encoder codebooks store `embed_sum` + `cluster_usage` —
different spellings, do not unify blindly (source: docs/truth-pack/TENSOR_INVENTORY.json).
Codebook dequantization normalizes `embedding_sum / max(cluster_usage, ε)` with ε=1e-5
(source: gh/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:676-679;
crates/ftts-model-qwen/src/codec.rs CODEBOOK_USAGE_EPSILON).

## §4 ID mappings and vocabularies

Four distinct token spaces; never conflate them.

**Text space (byte-level BPE).** Base vocab **151,643**; +33 added specials (ids 151643–151675)
⇒ `len(tokenizer)` **151,676**; text embedding has **151,936** padded rows (rows ≥151,676 are
unreachable padding). There is no `tokenizer.json` (source:
docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §1).

Text-side special ids (source: hf/config.json:5-10; docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §0.2;
docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §5):

| id | token | role |
|---|---|---|
| 151643 | `<\|endoftext\|>` | tokenizer pad_token |
| 151644 | `<\|im_start\|>` | chat header |
| 151645 | `<\|im_end\|>` | tokenizer eos_token |
| 151669 / 151670 | `<\|audio_start\|>` / `<\|audio_end\|>` | not used in prompt assembly above |
| 151671 | `<tts_pad>` | `tts_pad_token_id` |
| 151672 | `<tts_text_bos>` | `tts_bos_token_id` |
| 151673 | `<tts_text_eod>` | `tts_eos_token_id` — content is **eod**, not eos (§16 D-1) |
| 151674 | `<tts_text_bos_single>` | vestigial; zero read sites in the pinned tree — do not build a template around it |
| 151675 | `<\|audio_pad\|>` | |
| 77091 | `assistant` | `assistant_token_id` |

Wrapper slices (measured against the pinned tokenizer): assistant form =
3-token prefix `[151644, 77091, 198]` + text tokens + 5-token suffix
`[151645, 198, 151644, 77091, 198]`; reference form = same prefix + transcript + 2-token suffix
`[151645, 198]`. Hence upstream slice arithmetic `[:, :3]`, `[3:-5]`, `[3:-2]`, `[4:-5]`
(source: docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §0.1; crates/ftts-model-qwen/src/prompt.rs ROLE_PREFIX_IDS).

**Talker primary-code space (width 3072).** Generable alphabet = codes **0–2047 plus EOS 2150**;
ids **2048–3071 are suppressed except 2150** at sampling time. Control tokens (prompt-only,
never generable): `codec_pad_id 2148`, `codec_bos_id 2149`, `codec_eos_token_id 2150`,
`codec_think_id 2154`, `codec_nothink_id 2155`, `codec_think_bos_id 2156`,
`codec_think_eos_id 2157`. Language ids (`codec_language_id`): english 2050, german 2053,
spanish 2054, chinese 2055, japanese 2058, french 2061, korean 2064, russian 2069,
italian 2070, portuguese 2071 (source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.2;
hf/config.json:112-130; docs/truth-pack/FACT_DISPOSITIONS.md E-2). The remaining ids in
2048–3071 (e.g. 2048–2049, 2151–2153) have no assigned meaning in any truth-pack material
[UNVERIFIED whether they are trained or dead rows].

**Residual-code space (width 2048).** Fifteen per-depth embeddings and fifteen per-depth heads,
all `[2048, 1024]`, none shared across depths (source:
docs/truth-pack/OQ2_TENSOR_INVENTORY.md).

**Codec RVQ space.** All live codebooks have 2048 entries (decoder `rvq_first` layer 0 +
`rvq_rest` layers 0–14; encoder side likewise 2048-entry books) (source:
docs/truth-pack/TENSOR_INVENTORY.json). The talker→codec token map is the **identity**: generated
codes are concatenated with reference codes and fed straight to
`speech_tokenizer.decode` with no offset or remap; control ids ≥2048 can never reach the decoder
because they are sampled away (source:
gh/qwen_tts/inference/qwen3_tts_model.py:612-620; crates/ftts-model-qwen/src/codec.rs CODEBOOK_SIZE).
`semantic_codebook_size: 4096` in `decoder_config` is dead config — every materialized book is
2048 rows (§16 D-5).

## §5 Operator DAG — including the nested per-frame loop

Top level (per utterance):

```
tokenize text (verbatim mode, §7)
→ assemble prompt embeddings (§6) → talker prefill (positions 0..P-1)
→ loop frames until stop (§12):
     1. talker decode step (1 position) → hidden state → codec_head logits [3072]
     2. sample primary code c0 (talker sampler stack, §11)
     3. if c0 == EOS 2150 → stop (frame excluded from decode)
     4. RESET microdecoder KV (fresh DynamicCache every frame)
     5. for depth k in 0..14:
          microdecoder step → hidden at position k+1
          → lm_head[k] logits [2048] → sample residual c_{k+1}
          (sampled residual embeds back in for the next depth — autoregressive within frame)
     6. feedback: next talker input =
        Σ_{i=0..15} embed_i(c_i)  +  trailing_text_hidden[step] (else tts_pad_embed)
        where embed_0 = talker's 3072-vocab table, embed_{i≥1} = codec_embedding[i-1]
     7. enqueue 16-code frame for the codec
→ prepend reference codec frames (ICL), decode through codec (§13),
   cut the reference audio prefix off, return PCM
```

Grounding: steps 1–7 mirror
gh/qwen_tts/core/models/modeling_qwen3_tts.py:1670-1692 (microdecoder generate call, 16-way
embedding sum, trailing-text add), :1740 (`past_hidden = hidden_states[:, -1:, :]`), and the
corrected graph in docs/truth-pack/FACT_DISPOSITIONS.md C-1. The frame→talker feedback sum is
load-bearing: text conditioning is consumed one hidden per frame interleaved with audio
generation, not at prefill only (source: docs/truth-pack/FACT_DISPOSITIONS.md C-1). The Rust
implementation realizes this exact order in
crates/ftts-model-qwen/src/generate.rs `next_frame` (sample primary → reset-frame microdecoder
pass with per-step selector → 16-row feedback sum + text row → single-position talker forward).

Reference-priming detail: in ICL mode the reference's own codec frames are decoded **together
with** the generated frames (prepended before `decode`), and the corresponding audio prefix is
cut afterwards (`cut = int(ref_len/total_len * wav.shape[0])`). Decoding only the generated codes
is NOT equivalent — the causal decoder must be warmed by the reference frames or by an equivalent
precomputed ring-buffer state (source: docs/truth-pack/OQ8_WATERMARKING.md §3.1, §3.2;
gh/qwen_tts/inference/qwen3_tts_model.py:612-631).

## §6 Prompt construction: 2 clone formats × 2 streaming modes

Modes: ICL (voice clone with `ref_code`) and x-vector (speaker embedding / named speaker);
each × streaming (`non_streaming_mode=False`) / non-streaming. For the Base checkpoint +
voice cloning the official default is **streaming** (`generate_voice_clone` default
`non_streaming_mode=False`) (source: docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §0.4).

Common header `H` (built identically before every mode branch):
```
role:      text_proj(text_emb([151644, 77091, 198]))                — 3 positions, TEXT ONLY
header:    emb(codec_prefill_list)                                  — |P| = 3 (language auto/None:
             [nothink, think_bos, think_eos]) or 4 (given language: [think, think_bos, lang_id, think_eos])
         ++ [speaker_embed] if present                              — S ∈ {0,1}
         ++ emb([codec_pad, codec_bos])                             — 2
         each summed with tts_pad ×(L_c−2), then tts_bos last; the final codec_bos embedding is held back
H = 3 + (L_c − 1) = L_c + 2     (language+speaker ⇒ H=9; auto+none ⇒ H=7)
```
(source: docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §1, citing gh/qwen_tts/core/models/modeling_qwen3_tts.py:2126-2186).
Traps: role positions get no codec embedding; the held-back `codec_bos` is consumed only in the
x-vector modes — emitting it again in ICL duplicates it and makes the prompt one position too
long (source: docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §1).

Streams: `T1 = |ref_id| + |text_id| + 1` (text stream incl. trailing `tts_eos_embed`);
`T2 = 1 + T_ref` (codec stream = `emb(codec_bos)` + per-frame sums of the 16 reference-code
embeddings, group 0 via the talker's 3072-vocab table, groups 1–15 via
`codec_embedding[i-1]`) (source: docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §2).

| Mode | Prompt positions | Trailing per-frame stream | Cacheable prefix (maximal target-independent) |
|---|---|---|---|
| ICL × streaming | `H + T2` — elementwise SUM of text & codec streams, position p pairs text[p] with codec entry p | leftover text `text[T2:]` then `tts_pad` forever | `H + min(T2, \|ref_id\|)` |
| ICL × non-streaming | `H + T1 + T2` — two sequential blocks: `[ref_id,text_id,eos]+codec_pad`, then `[codec_bos, ref frames]+tts_pad` | none (`tts_pad` forever) | `H + \|ref_id\|` (reference codec frames sit after the text causally — NOT cacheable) |
| x-vector × streaming | `H + 1` — first target token ⊕ held-back `codec_bos` | remaining text + `tts_eos` | `H` |
| x-vector × non-streaming | `H + \|text\| + 2` — full text + `codec_pad`, then `tts_pad ⊕ codec_bos` | none (`tts_pad` forever) | `H` |

(source: docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §2–§5 incl. worked example §7).
Cache key must include `(language_id, speaker_embed presence/value, ref_id token ids, ref_code,
non_streaming_mode)` on top of the engine keys (source:
docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §5.1).

Batch note: mixed-length batches left-pad `talker_input_embeds` with a mask while
`trailing_text_hiddens` are right-padded with `tts_pad` fill — opposite directions; getting one
backwards silently desynchronizes text from frames (source:
docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §6, citing gh/qwen_tts/core/models/modeling_qwen3_tts.py:2240-2269).

## §7 Text front end (verbatim mode)

- Identity: `Qwen2Tokenizer(Fast)`, GPT-2-lineage byte-level BPE over `vocab.json` +
  `merges.txt`; slow and fast agree on all 92 conformance cases, one implementation suffices
  (source: docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §1).
- **verbatim = Unicode NFC, then byte-level BPE with the official (Mistral) pre-tokenizer regex,
  and nothing else.** No case folding, number/abbreviation expansion, punctuation rewriting,
  whitespace collapsing, or NFKC (source: docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §4).
- NFC, not NFKC — proven with compatibility probes (`①≠1`, `Ａ≠A`, `ﬁ≠fi`, `㈱≠(株)`);
  NFC runs inside the tokenizer, not the processor. Pin the Unicode version of the NFC tables
  (source: docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §2).
- `decode(encode(x)) == x` holds **only** for NFC-normalized input; do not spec a global
  round-trip law (source: docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §2).
- Every official entrypoint loads with `fix_mistral_regex=True`, which replaces the fast
  tokenizer's pre-tokenization regex with Mistral's (drops contraction alternation, splits on
  case boundaries, adds `/` to punctuation runs). Conformance default = official regex; expose a
  native-Qwen kill switch later (source: docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §3).
  Reference fixture: docs/truth-pack/tokenizer/tokenizer_conformance.json (92 cases).
- Tokenizer adds no BOS/EOS; `add_bos_token: false`, `add_prefix_space: false`,
  `errors: "replace"`, `model_max_length: 131072` (source:
  docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §5).

## §8 Rotary schemes — there are THREE, never share a kernel

### §8.1 Talker mRoPE (θ=1e6, sections [24,20,20], interleaved, 3-D ids)

Config: `rope_theta 1000000`, `rope_scaling {interleaved: true, mrope_section: [24,20,20]}`
(source: hf/config.json:148-158). Applied by `apply_multimodal_rotary_pos_emb` with
interleaved channel selection over the three axes (source:
gh/qwen_tts/core/models/modeling_qwen3_tts.py:660-724, called at :773-780).

Position schedule: despite the 3-D representation, all three axes receive the **same scalar
causal index** at every element. Prefill:
```text
p[b,j] = cumsum(M[b,:])[j] − 1 ; p[b,j] = 1 where M[b,j]==0
position_ids[axis,b,j] = p[b,j]  ∀ axis ∈ {0,1,2}
```
Decode transition: `delta0 = count(M==0)`; `mrope_delta = max(position_ids)+1 − sum(M)`;
`rope_deltas = mrope_delta − delta0` cached on the talker (≈ `-left_pad_count`);
each subsequent one-position decode uses
`position = cache_position[0] + rope_deltas` replicated across axes. There is **no** 12.5 Hz /
13 Hz rescaling, no axis-specific offset, no reset at the prompt→audio boundary
(source: gh/qwen_tts/core/models/modeling_qwen3_tts.py:1693-1711, 1792-1800;
docs/truth-pack/PIN_RECORD.md-adjacent analysis preserved from the prior spec revision, verified
against those lines). `position_id_per_seconds: 13` (hf/config.json:146) has no read site in the
pinned source tree and is non-operative metadata (§16 D-6).

Because every axis shares one position, section selection leaves cos/sin unchanged relative to a
plain doubled-half representation; the Rust port computes plain doubled-half rows at θ=1e6 and
keeps them type-distinct from the other two rotary tables (source:
crates/ftts-model-qwen/src/talker.rs:149-190, 967-975).

### §8.2 Microdecoder plain RoPE (θ=1e6, positions 0–15)

Plain `apply_rotary_pos_emb` (not multimodal), θ=1e6, no scaling, over absolute positions 0..15
of the 16-position frame sequence (source: docs/truth-pack/OQ5_MICRODECODER_WIRING.md §0, §4;
docs/truth-pack/FACT_DISPOSITIONS.md C-3). "Plain" means *no mRoPE sectioning*, NOT a different
base — defaulting this path to θ=1e4 is silently wrong (source:
docs/truth-pack/FACT_DISPOSITIONS.md C-3). The table covers 16 positions (census
`rope_table_values: 4096` = 16 pos × 128 head_dim × 2 for cos+sin; source:
docs/truth-pack/EXECUTION_CENSUS.json `.microdecoder_hot_working_set`). Distinct type in Rust:
`microdecoder::RopeTable` (source: crates/ftts-model-qwen/src/microdecoder.rs:200-231).

### §8.3 Codec pre-transformer plain RoPE (θ=1e4, window 72, max_pos 8000)

Different base AND different range from the microdecoder's: `rope_theta 10000`,
`sliding_window 72`, `max_position_embeddings 8000` (source:
hf/speech_tokenizer/config.json:22-32). Reusing the microdecoder's 16-row table here is a
silent-corruption bug (source: bead `frankentts-t-executable-spec-1ch` comment cc_11;
crates/ftts-model-qwen/src/codec.rs `codec_rope_rows`).

## §9 Talker block internals

Per repeated layer (pre-norm, two residual additions, no biases):
```text
r0 = x
a  = attention(RMSNorm(x))        # q/k RMSNorm over head_dim AFTER projection, BEFORE RoPE
x1 = r0 + a                        # GQA 16Q/8KV, scaling head_dim^-0.5, eager softmax in f32
m  = down_proj(SiLU(gate_proj(RMSNorm(x1))) * up_proj(RMSNorm(x1)))   # SwiGLU, gate gets SiLU
out = x1 + m
```
(source: docs/truth-pack/FACT_DISPOSITIONS.md E-1, citing
gh/qwen_tts/core/models/modeling_qwen3_tts.py:595-610, 740-780, 842-855, 1348-1417;
`hidden_act: "silu"` at hf/config.json:136). Final norm `talker.model.norm.weight` then unbiased
`codec_head` ([3072,1024]) produce frame logits. Attention softmax runs in fp32 and returns to
input dtype (source: gh/qwen_tts/core/models/modeling_qwen3_tts.py:647-657).

Overflow-K bounds (u8×s8 int8 route): binding K per matmul class — `down_proj` K=3072,
`o_proj` K=2048, text embedding row gather K=2048; worst decode-path accumulator
232,135,680 at K=7168 belongs to the codec's `decoder.decoder.0.conv` (source:
docs/truth-pack/EXECUTION_CENSUS.json `.overflow_k_table`).

Talker KV cost: 57,344 values/token ⇒ 114,688 B/token BF16 (229,376 F32); e.g. 10 s ≈ 125
frames ≈ 14.3 MB BF16 (source: docs/truth-pack/EXECUTION_CENSUS.json `.talker_kv`).

## §10 Microdecoder wiring (the 16-position sequence)

Sequence layout (identical in training and inference — Tier-1 mask/distribution equivalence):

| seq pos p | input vector | scored? | head |
|---|---|---|---|
| 0 | talker hidden state for this frame | no | — |
| 1 | **talker's** codec embedding (vocab 3072) of c0 | yes | `lm_head[0]` → c1 |
| 2..15 | `codec_embedding[p-2]` (vocab 2048) of c_{p-1} | yes | `lm_head[p-1]` → c_p |

Index map stated once: for residual j ∈ [1,15], c_j is produced at position j by
`lm_head[j-1]` and re-enters at position j+1 through `codec_embedding[j-1]`. Embedding-list
index i serves code i+1, not code i — indexing `depth_emb[i]` with code i yields plausible-but-
wrong audio with no crash (source: docs/truth-pack/OQ5_MICRODECODER_WIRING.md §1, §3;
docs/truth-pack/FACT_DISPOSITIONS.md C-2, citing
gh/qwen_tts/core/models/modeling_qwen3_tts.py:1619-1626, 1670-1680).

KV reset semantics: `code_predictor.generate(...)` is invoked fresh per talker step and never
receives `past_key_values`; a new `DynamicCache` allocates each frame, `cache_position` restarts
at 0, maximum extent 16 positions × 5 layers × (8 KV heads × 128) (source:
docs/truth-pack/OQ5_MICRODECODER_WIRING.md §2).

Mask/positions: training pass = plain 16×16 lower-triangular causal mask over absolute positions
0..15 (`layer_types` all `full_attention`, no sliding mask ever built); inference prefill covers
positions 0–1 then appends one position per step — identical structure (source:
docs/truth-pack/OQ5_MICRODECODER_WIRING.md §4).

Equivalence verdict: the training-mode single causal pass is an **exact block verifier**
(FrankenMTP Tier 1) for a drafted residual sequence — strict-mode token identity holds by
construction; sampled-mode distributional exactness is licensed at the logits level; in floating
point, compare argmax/token ids unless the verify kernel reproduces sequential reduction order
(source: docs/truth-pack/OQ5_MICRODECODER_WIRING.md §5, §6).

Body: 5 layers with the same internal schema as a talker layer (QK-Norm over head_dim before
RoPE, SwiGLU, no biases), GQA 16/8×128, final norm, then the per-depth head (source:
docs/truth-pack/OQ5_MICRODECODER_WIRING.md §0, §9).

## §11 Sampling contract

Parameter resolution precedence: user argument → `generation_config.json` → hard-coded default
(source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.1, citing
gh/qwen_tts/inference/qwen3_tts_model.py:318-352). Effective defaults (source:
hf/generation_config.json; docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.1):

| Parameter | Value | Origin |
|---|---|---|
| talker `do_sample` / `temperature` / `top_k` / `top_p` | True / 0.9 / 50 / 1.0 | generation_config.json |
| talker `repetition_penalty` | 1.05 | generation_config.json |
| microdecoder `subtalker_*` dosample/temp/top_k/top_p | True / 0.9 / 50 / 1.0 | generation_config.json |
| microdecoder repetition penalty | **none — not passed at all** | call site |
| talker `min_new_tokens` | 2 | hard-coded (gh/qwen_tts/core/models/modeling_qwen3_tts.py:2046) |
| talker `eos_token_id` | 2150 | config.json |
| talker `suppress_tokens` | 2048..3071 except 2150 | computed (:2059-2063) |
| `max_new_tokens` (talker frames) | **8192 effective** (§16 D-3) | generation_config.json |
| microdecoder `max_new_tokens` | 15 | `num_code_groups - 1` (:1673) |

Processor/warper order (transformers 4.57.3 `_get_logits_processor`; processors always run,
warpers only when `do_sample=True`):
```
TALKER, per frame:
  logits [3072] → repetition_penalty(1.05, applied once per repeated id — duplicates collapse
                  via gather/scatter over input_ids) → min_new_tokens(2: −inf EOS while <2
                  generated) → suppress_tokens(−inf on 2048..3071 except 2150)
                → ÷temperature(0.9) → top_k(50) → top_p(1.0, no-op) → softmax → multinomial
MICRODECODER, per depth (×15):
  logits [2048] → (no penalty, no min-length, no suppression)
                → ÷0.9 → top_k(50) → top_p(1.0, no-op) → softmax → multinomial
```
(source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.3-§1.4; duplicate-collapse semantics per
crates/ftts-model-qwen/src/sampler.rs:178-248 comments — compounding per occurrence is a
silent semantic bug that ends utterances early).

Canonical greedy (Contract-A conformance decoder), forced not chosen:
`argmax(warpers(processors(l))) == argmax(processors(l))` because temperature is monotonic,
top-k keeps the max, top-p(1.0) is a no-op. Canonical greedy is therefore defined as
**processors then argmax** (talker: penalty → min-new-tokens → suppression → argmax;
microdecoder: raw argmax) and is warper-invariant. It is NOT raw argmax of model logits —
dropping processors changes ids (suppressed specials emitted as audio codes; greedy trajectory
diverges after the first repeated code without the penalty) (source:
docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §2).

RNG divergence: the reference draws via `torch.multinomial`; we replicate the distribution
contract with our own seeded RNG (SplitMix64 in
crates/ftts-model-qwen/src/sampler.rs). Determinism scope is {engine build + ISA path + sampler
version + seed + artifact}; comparisons against the reference are distribution-level over many
seeds, never single-seed A/B; `strict` math mode + fixed seed reproduces our own output exactly
(source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §3).

Production-mode note: residuals are SAMPLEd whenever the talker samples (`subtalker_dosample=true`);
greedy residuals under a sampled talker sit in a measured silence attractor (source:
crates/ftts-model-qwen/src/generate.rs:1099-1102, frankentts-p7r).

## §12 Stop semantics and length caps

- **EOS:** talker head id **2150** (`codec_eos_token_id`) ends the utterance; the EOS-bearing
  step produces NO frame (the reference's ft8 capture shows the 8th attempted group-0 token was
  EOS and was excluded; source: docs/truth-pack/PIN_RECORD.md "Machine-local CPU fixture
  pointer"). EOS is never sticky across utterances; a fresh utterance resets state (source:
  crates/ftts-model-qwen/src/generate.rs:1592-1597 test contract).
- **min-new-tokens:** EOS logit forced to −inf until 2 group-0 codes have been generated
  (source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.1, §1.4).
- **Suppression:** ids 2048–3071 except 2150 are never generable, so prompt specials cannot be
  emitted as audio codes (source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.2).
- **Frame cap:** effective `max_new_tokens` = 8192 talker frames ⇒ 8192 × 80 ms ≈ **655 s
  (10.9 min)** hard ceiling on one utterance; admission control must use 8192, not 2048 or 4096
  (§16 D-3) (source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.1 note).
- **Microdecoder never stops early:** exactly 15 residual steps per accepted frame; its generate
  call has no EOS/suppression/min-length (source:
  docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.4).
- **Waveform trim:** decoded audio is cut to `n_frames × 1920` samples
  (`audio_lengths = (audio_codes[...,0] > -1).sum(1) * decode_upsample_rate`, clamp min 0)
  (source: gh/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:1012-1015;
  docs/truth-pack/OQ8_WATERMARKING.md §2).

## §13 Codec decoder structure (decode side; encoder is enrollment-only)

Pipeline (source: gh/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:869-884):

1. **SplitRVQ dequantize.** `rvq_first` over `n_q_semantic=1` code group, `rvq_rest` over the
   other 15; each branch: per-group codebook lookup (row = `embedding_sum /
   max(cluster_usage,1e-5)`) → input_proj [256←512] → output_proj [512←256]; branch outputs
   **summed** (source: modeling_qwen3_tts_tokenizer_v2.py:661-721, 780-821).
2. **pre_conv:** causal Conv1d 512→1024, k=3, left-pad only (constant zero), then transpose to
   time-major (:839-843, :874; causal pad at :189-192).
3. **pre_transformer:** 8 pre-norm layers, hidden 512, MHA 16Q/16KV ×64 (width 1024), biased
   input_proj 1024→512, SiLU SwiGLU MLP (intermediate 1024), RMSNorm eps 1e-5, plain RoPE θ=1e4,
   **sliding-window causal attention window 72**, and **LayerScale (init 0.01) on BOTH residual
   branches** (`self_attn_layer_scale`, `mlp_layer_scale`); no QK-Norm (:282-371, 394-418;
   hf/speech_tokenizer/config.json). Live KV bounded at 72 positions/layer — the 568-frame
   transformer receptive field is an information horizon, not storage (source:
   docs/truth-pack/EXECUTION_CENSUS.json `.codec_receptive_field.ring_buffer_note`).
4. **Latent upsample ×2 stages:** each = causal ConvTranspose1d 1024→1024 (kernel 2r, stride r,
   r=2) followed by a ConvNeXt block (depthwise causal conv k=7 groups=dim → LayerNorm eps 1e-6
   → Linear dim→4·dim (biased) → GELU → Linear 4·dim→dim (biased) → γ·h with γ init 1e-6 →
   residual) (:195-243, 845-855).
5. **BigVGAN conv stack:** `decoder.0` causal Conv1d 1024→1536 k=7; then 4 DecoderBlocks
   (`upsample_rates` [8,5,4,3]): SnakeBeta(in) → causal tconv (kernel 2r, stride r) → 3 residual
   units (dilations 1,3,9) each SnakeBeta → causal conv k=7 dilation d → SnakeBeta → causal conv
   k=1 (+residual). Channel ladder 1536→768→384→192→96 (:619-658, 857-865). SnakeBeta
   `x + (1/b)·sin²(a·x)` with alpha/beta stored as LOGS — exponentiate before the reciprocal
   (:578-616; crates/ftts-model-qwen/src/codec.rs snake_beta_in_place).
6. **Head:** final SnakeBeta(96) → causal Conv1d 96→1 k=7 → `clamp(−1, 1)`
   (:861-884).

Three activation families coexist in one decoder: SnakeBeta (conv stack), GELU (ConvNeXt), SiLU
(transformer MLP) (source: bead comment cc_11; grounding lines above).

Causality: every conv is causal (left padding only); every tconv trims right pad; streaming
decode must retain ring buffers of `(kernel−1)×dilation` input frames per conv and 72-position
KV per transformer layer. Total conv-only left context **11.3796875 latent frames**; total left
context 579.3796875 frames incl. the transformer horizon (source:
docs/truth-pack/EXECUTION_CENSUS.json `.codec_receptive_field`).

Chunking: upstream `chunked_decode(chunk_size=300 frames, left_context_size=25 frames)` is an
approximation; our streaming engine processes packet-at-a-time with persistent causal state and
must prove equality against the whole-sequence offline decode, not against upstream's chunk
approximation (source: modeling_qwen3_tts_tokenizer_v2.py:886-896;
crates/ftts-model-qwen/src/codec.rs:2140-2144 comment).

No post-processing exists between the decoder output and the caller — no watermark, filter,
dither, normalization, or resample stage (length trim only) (source:
docs/truth-pack/OQ8_WATERMARKING.md §1-§2).

## §14 Frame ↔ sample alignment

- Frame rate 12.5 fps (80 ms), `encode_downsample_rate = decode_upsample_rate = 1920`,
  sample rate 24 kHz in/out: 12.5 × 1920 = 24,000 exactly. Hop math: `upsample_rates
  [8,5,4,3] = 480` × `upsampling_ratios [2,2] = 4` ⇒ 1920 ✓ matches
  `decode_upsample_rate` (source: hf/speech_tokenizer/config.json:9-42;
  docs/truth-pack/FACT_DISPOSITIONS.md C-5; docs/truth-pack/EXECUTION_CENSUS.json `.frame`).
- Every accepted frame emits exactly 1,920 PCM samples; streaming pushes concatenate bit-exactly
  when primed with reference frames (`wav_len == total_len × 1920`; the reference prefix cut is
  algebraically `ref_len × 1920`) (source:
  crates/ftts-model-qwen/src/codec.rs:2020-2034; docs/truth-pack/OQ8_WATERMARKING.md §3.2).
- Reference-frame count for a prompt is `T_ref` frames from the codec encoder's framing; the
  working assumption `T_ref = ceil(ref_seconds × 12.5)` scales prompt-length numbers but changes
  no structural conclusion (source: docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §8) [UNVERIFIED
  as an exact law — the encoder's boundary behavior at exact multiples is not pinned in the
  truth pack].
- Talker KV growth: 114,688 B/frame BF16 (source:
  docs/truth-pack/EXECUTION_CENSUS.json `.talker_kv.bytes_per_token_bf16`).

## §15 Determinism and conformance floor (summary)

CPU FP32 oracle repeatability: every recorded seam reproduced exactly (`max_abs = 0.0`,
`differing_elements = 0`) across thread-count variations — L0/L2/L3/L4/L5 tolerances are exact 0
on the CPU tier, observed at 1-frame scale. Native-CUDA repeatability on RTX 4090: 5440/5440
arrays bit-identical ⇒ tolerance 0 at every rung on a pinned device. CPU-vs-CUDA: all seams
bit-exact except `codec.generated_waveform` (cuDNN conv algorithm selection; pointwise max-abs
9.3e-2 near zero crossings) ⇒ waveform comparisons must use spectral/envelope metrics, never raw
sample diff, cross-device. TF32 plays no role on this stack (source:
docs/truth-pack/NONDETERMINISM_FLOOR.md).

## §16 Discrepancies found during extraction

- **D-1 — `tts_eos_token_id` names an *eod*.** Config field `tts_eos_token_id` maps to token
  `<tts_text_eod>`; use the id, never the name (source:
  docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §0.2). Doc-vs-name conflict inside upstream itself.
- **D-2 — "two rotary schemes" is wrong; there are three.** The bead text and plan §2.10 say two
  (talker mRoPE θ=1e6 sections [24,20,20]; microdecoder plain RoPE). The codec pre-transformer's
  plain RoPE at θ=1e4 over max_position_embeddings 8000 is a third configuration needing its own
  kernel/table; reusing the microdecoder's 16-row table there is a silent-corruption bug
  (source: docs/truth-pack/FACT_DISPOSITIONS.md C-3; bead comment cc_11;
  hf/speech_tokenizer/config.json:22-32). Also note the microdecoder's θ is 1e6, same as the
  talker's — "plain" refers to absence of mRoPE sectioning, not base.
- **D-3 — three conflicting `max_new_tokens` sources.** generation_config.json 8192 vs
  `generate` signature default 4096 (gh/qwen_tts/core/models/modeling_qwen3_tts.py:2031) vs
  README eval 2048 (gh/README.md:465). Precedence chain (user → generation_config → hard default)
  makes **8192** effective (source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §1.1;
  docs/truth-pack/FACT_DISPOSITIONS.md E-4).
- **D-4 — stale nested sampling config.** `code_predictor_config` carries `temperature 1.0`,
  `do_sample false`, `repetition_penalty 1.0`, but the call site threads explicit
  `subtalker_*` overrides from generation_config.json (T0.9/k50/p1.0/sample=true, no penalty).
  A port reading the nested config samples greedily at T=1.0 — silently wrong (source:
  docs/truth-pack/OQ5_MICRODECODER_WIRING.md §0, §7; docs/truth-pack/FACT_DISPOSITIONS.md E-4).
- **D-5 — `semantic_codebook_size: 4096` is dead config.** FACT_DISPOSITIONS S-1 listed the
  4096-vs-2048 puzzle as open; the pinned tensor inventory settles it — every materialized RVQ
  codebook is `[2048, 256]` (decoder and encoder alike), and the Rust engine asserts 2048 as the
  only live size (source: docs/truth-pack/FACT_DISPOSITIONS.md S-1;
  docs/truth-pack/TENSOR_INVENTORY.json; crates/ftts-model-qwen/src/codec.rs:15-16).
- **D-6 — `position_id_per_seconds: 13` is non-operative.** No read site in the pinned
  `qwen_tts` tree; treating it as a rescaling schedule breaks parity (source: hf/config.json:146;
  §8.1).
- **D-7 — upstream docstring bug in speaker encoder config.** Docstring says `enc_dim` "defaults
  to 192"; code default is 1024. Port from the signature, never the docstring (source:
  docs/truth-pack/FACT_DISPOSITIONS.md §2.9 note).
- **D-8 — plan §7.5 phrasing risk on microdecoder position 1.** "One conditioning/primary
  position plus 15 residual positions" reads as though c0's embedding were unscored; in fact p=1
  carries `embed(c0)` AND is scored by `lm_head[0]` (only p=0 is unscored conditioning). Same
  arithmetic, easy off-by-one (source: docs/truth-pack/OQ5_MICRODECODER_WIRING.md §5).
- **D-9 — encoder RVQ built with 32 levels, 16 valid.** `encoder_config.num_quantizers: 32` vs
  top-level `encoder_valid_num_quantizers: 16` (source: docs/truth-pack/FACT_DISPOSITIONS.md E-5;
  hf/speech_tokenizer/config.json:6,70). Decode side is unaffected (16 quantizers enforced,
  modeling_qwen3_tts_tokenizer_v2.py:870-871).
- **D-10 — upstream chunked decode vs true streaming.** `chunked_decode(300, 25)` approximates
  streaming with re-decoded context windows; our packet-at-a-time engine keeps persistent causal
  state and proves equality against offline whole-sequence decode instead (source:
  modeling_qwen3_tts_tokenizer_v2.py:886-896; crates/ftts-model-qwen/src/codec.rs:2140-2144).
- **D-11 — repetition-penalty multiplicity.** HF's processor penalizes a repeated code once
  (gather/scatter collapses duplicates); a per-occurrence compounding implementation silently
  truncates utterances (measured: code 1657 held four frames fell 28.425→23.385 compounded and
  lost to EOS at 24.146) (source: crates/ftts-model-qwen/src/sampler.rs:186-196).
- **D-12 — deliverable history.** The previous revision of this file (104 lines) covered only
  the talker mRoPE schedule (SPEC-OQ4); that content is preserved and folded into §8.1. This
  revision supersedes it wholesale (source: bead `frankentts-t-executable-spec-1ch` status-audit
  comment 2026-08-24, GreenCitadel).
- **D-13 — `fix_mistral_regex`: inference ≠ training regex.** The checkpoint was near-certainly
  trained with Qwen's native pre-tokenizer regex while the official inference stack always loads
  Mistral's; "match upstream" and "match training" differ. Conformance follows the official
  stack, with a kill switch reserved (source:
  docs/truth-pack/tokenizer/OQ11_TOKENIZER.md §3).

## §17 Honest gaps carried forward

- Exact reference framing law `T_ref = ceil(ref_seconds × 12.5)` remains an assumption (§14).
- Unassigned talker control-band ids (2048–2049, 2151–2153, others ≤2147) have no documented
  meaning [UNVERIFIED].
- Greedy audio quality viability ((a)) and the L5 fork ((b)) remain oracle-blocked and are out
  of scope here (source: docs/truth-pack/OQ12_SAMPLER_CONTRACT.md §5).
- The numeric oracle trace for the microdecoder Tier-1 verdict lives outside this spec
  (`frankentts-oq5-oracle-trace-*`) (source: docs/truth-pack/OQ5_MICRODECODER_WIRING.md §8).
