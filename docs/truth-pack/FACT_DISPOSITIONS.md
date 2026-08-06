# Plan §2 Fact Dispositions

Every `[SOURCE]`/`[REPORTED]` fact in `COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md` §2, adjudicated against
the pinned bytes in [`PIN_RECORD.md`](PIN_RECORD.md). Citations are `file:line` into `snapshots/`.

**Dispositions:** `VERIFIED` (bytes agree) · `CORRECTED` (bytes disagree — plan must change) ·
`EXTENDED` (plan right, but the bytes add a load-bearing detail a port would otherwise guess) ·
`STILL OPEN` (needs execution or deeper reading, not settled by config).

**Headline: the plan's architecture survived the pin.** All of §2.3–§2.7's geometry is confirmed, and
every number in §2.6 — the planning centerpiece — reproduces *exactly*. Six corrections were found;
one (C-1) changes the per-frame execution graph and is load-bearing for the Phase-1 forward.

---

## Corrections (plan must be edited)

### C-1 — §2.1 execution graph omits the frame→talker feedback path **[load-bearing]**
The plan's per-frame pseudocode ends at "enqueue the 16-code frame for the codec". The source does
something else first: all 16 code embeddings are **summed** into the talker's *next-step* input, and
a text hidden state is added on top.

`gh/qwen_tts/core/models/modeling_qwen3_tts.py:1671-1692`:
```python
predictor_result = self.code_predictor.generate(..., max_new_tokens=num_code_groups - 1)
codec_hiddens = torch.cat(
    [last_id_hidden]                                                   # talker's own embedding of code 0
    + [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1])
       for i in range(self.config.num_code_groups - 1)],               # 15 per-depth embeddings
    dim=1)
inputs_embeds = codec_hiddens.sum(1, keepdim=True)                     # <-- SUM of all 16
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
else:
    inputs_embeds = inputs_embeds + tts_pad_embed                      # <-- text stream or pad
```
Corrected graph:
```
for each 80 ms frame:
    talker step -> hidden ; sample primary code (vocab 3072)
    reset microdecoder KV
    for depth in 0..14: microdecoder step -> per-depth 2048-way head -> sample residual
    talker_next_input = sum(talker_emb(code0), depth_emb[i](code_{i+1}) for i in 0..14)
                        + (trailing_text_hidden[step] if step < len else tts_pad_embed)
    enqueue the 16-code frame for the codec
```
**Why it matters:** the text stream is consumed *one hidden per frame* interleaved with audio
generation — it is not a prefill-only prefix. Any engine that treats text as consumed at prefill will
desync after the first frame. Also fixes the per-frame op set: a 16-way embedding-sum + add is in the
hot loop. Affects `frankentts-p1-e2e-miy`, `frankentts-ep-phase1-forward-s5d`, OQ-10, OQ-14.

### C-2 — §2.5 per-depth embedding→code index mapping is off-by-one from the natural reading
Embedding/head list index `j` (0-based, 15 entries) serves code **`j+1`**, not code `j`. Code 0 is
embedded by the **talker's** `codec_embedding`, not by the microdecoder's.
Generate path `modeling_qwen3_tts.py:1684`; training path `:1620-1626`:
```python
if i == 0: sub_talker_inputs_embeds.append(self.get_input_embeddings()(codec_ids[:, :1]))
else:      sub_talker_inputs_embeds.append(
               self.code_predictor.get_input_embeddings()[i-1](codec_ids[:, i:i+1]))
```
A port that indexes `depth_emb[i]` with code `i` produces plausible-but-wrong audio with no crash.
Affects `frankentts-k-rcd-engine-6e3`, OQ-5.

### C-3 — §2.10 says "two rotary kernels"; there are **three** rotary configurations
| Path | Kernel | θ | Notes |
|---|---|---|---|
| Talker | `apply_multimodal_rotary_pos_emb` (`:660`, used `:778`) | 1e6 | sections `[24,20,20]`, `interleaved: true`, 3-D ids |
| Microdecoder | `apply_rotary_pos_emb` (`:858`, used `:933`) | **1e6** | `rope_scaling: null`; positions 0–15 only |
| Codec decoder | windowed | **1e4** | `speech_tokenizer/config.json` `decoder_config.rope_theta`, `sliding_window: 72` |
Note the microdecoder's θ is **1e6, same as the talker** — "plain RoPE" means *no mRoPE sectioning*,
**not** a different θ. A port defaulting the microdecoder to θ=1e4 is silently wrong. Affects
`frankentts-ep-phase1-forward-s5d`, `frankentts-k-rcd-engine-6e3`.

### C-4 — §2.2 "License Apache-2.0 [REPORTED]" — the *weights repo ships no LICENSE file*
`GET /api/models/.../siblings` lists 13 files; **none is `LICENSE` or `NOTICE`**. The weights'
Apache-2.0 claim rests solely on the `license: apache-2.0` YAML tag in the model card
(`hf/README.md:2`). The *code* repo does ship a verbatim stock Apache-2.0 `LICENSE`
(`gh/LICENSE`, 11,343 B, `Copyright 2026 Alibaba Cloud` at `:189` in the appendix boilerplate); there
is **no separate `NOTICE` file** in the tree. Promote to **[VERIFIED for code / METADATA-ONLY for
weights]**. Direct input to OQ-1: our attribution must cite the model-card tag as the weights'
license evidence, and cannot copy a `NOTICE` that does not exist.

### C-5 — §2.7 "confirm exact hop math at pin (OQ-7)" — **resolved, and the plan's guess was right**
`speech_tokenizer/config.json` states the hop directly: `decode_upsample_rate: 1920`,
`encode_downsample_rate: 1920`. The `×4` the plan inferred is `decoder_config.upsampling_ratios:
[2, 2]`, distinct from `upsample_rates: [8,5,4,3]`: `8·5·4·3 = 480`, `480 · 2 · 2 = 1920`. ✓
This sub-question of OQ-7 is closed by config; the remaining OQ-7 items (see S-1) are not.

### C-6 — §2.2 / bead text says "all weight shards"; there are no shards
Two unsharded safetensors, no `model.safetensors.index.json`. The loader must not expect an index
file. Affects `frankentts-p1-weights-qjl`.

---

## Verified

### §2.1 / §2.5 — the 15-step microdecoder (the thesis)
`max_new_tokens = num_code_groups - 1 = 15` at `modeling_qwen3_tts.py:1673`; exactly 15 per-depth
embeddings (`:1031`) and 15 per-depth `vocab_size`-wide heads (`:1168`), both built with
`range(config.num_code_groups - 1)`. `num_code_groups: 16`, `num_hidden_layers: 5`. Residual codes
are autoregressive within the frame (each sampled code is embedded and fed to the next step,
`:1684`) — **the plan's deletion of any "groups are independent" optimization stands.** **VERIFIED.**

### §2.3 — talker geometry
`config.json` `talker_config`: 28 layers / hidden 1024 / intermediate 3072; 16 Q / 8 KV;
`head_dim 128` ⇒ attention width 2048 > hidden ✓; `rope_theta 1000000`; `rope_scaling.mrope_section
[24,20,20]` with `interleaved: true`; `position_id_per_seconds: 13`; `rms_norm_eps 1e-06`.
Sampling `generation_config.json`: `temperature 0.9`, `top_k 50`, `top_p 1.0`,
`repetition_penalty 1.05`, `do_sample true`. **VERIFIED.**

### §2.4 — the text path
`text_vocab_size 151936`, `text_hidden_size 2048` ⇒ `151936 × 2048 × 2 = 622,329,856 B` = **622.3 MB
BF16**, matching the plan exactly. **VERIFIED.**

### §2.6 — the traffic model reproduces *exactly* from the pinned config
Recomputed from `hidden 1024 / inter 3072 / 16Q·8KV · head_dim 128` ⇒ 15,728,640 params/layer:

| Plan figure | Recomputed | |
|---|---|---|
| talker body ≈440 MB | 440.4 MB | ✓ |
| primary head ≈3 MB | 3.15 MB (1024×**3072**) | ✓ |
| microdecoder body ≈78.6 MB | 78.6 MB | ✓ |
| ×15 ⇒ ≈1.18 GB | 1.180 GB | ✓ |
| 15 per-depth heads ≈31.5 MB | 31.5 MB | ✓ |
| total ≈1.65 GB/frame | 1.655 GB | ✓ |
| ⇒ ≈20.7 GB/s @1× RT | 20.7 GB/s | ✓ |
| one-read floor ≈0.55 GB ⇒ ≈6.9 GB/s | 0.554 GB ⇒ 6.9 GB/s | ✓ |
| hot working set "~110 MB-class" | 110.1 MB (body 78.6 + heads 31.5) | ✓ |

**VERIFIED — no correction.** One clarification for OQ-2: the plan lists "per-depth embeddings" in
the residency sentence but correctly does **not** add their 31.5 MB to the ~110 MB. They are
*gathered* — one 1024-B row per step, 15 KB/frame — so they belong to traffic, not to the resident
footprint. OQ-2's census should record them as a separate access class, not fold them in.

### §2.9 — the speaker encoder
`configuration_qwen3_tts.py:49-57`: `mel_dim=128` ✓, `enc_dim=1024` ✓ (also set in `config.json`
`speaker_encoder_config`), `enc_channels=[512,512,512,512,1536]` ✓,
`enc_kernel_sizes=[5,3,3,3,1]` ✓, `enc_dilations=[1,2,3,4,1]` ✓, `sample_rate=24000` ✓;
`num_mels=128` again at the call site `modeling_qwen3_tts.py:1946`. **VERIFIED.**
⚠️ The docstring at `configuration_qwen3_tts.py:31` says `enc_dim` "defaults to 192" while the code
default is `1024` — an upstream docstring bug. **Port from the signature, never the docstring.**

### §2.7 — codec geometry
`speech_tokenizer/config.json`: 24 kHz in/out ✓; 1,920 samples/frame ✓; 16 quantizers ✓;
`decoder_config` 8 layers ✓, hidden 512 ✓, intermediate 1024 ✓, 16 heads × head_dim 64 ✓,
`sliding_window: 72` ✓, `upsample_rates [8,5,4,3]` ✓. `encoder_config._frame_rate: 12.5` ✓.
**VERIFIED.**

### §2.2 — size / dtype / params
BF16 ✓; 914,643,008 params ("≈0.9B") ✓; 2,511,637,364 B ≈ 2.51 GB ("≈2.52 GB") ✓;
`model.safetensors` 1.829 GB ("≈1.83 GB talker") ✓. **VERIFIED**, promoted from [REPORTED].

---

## Extended (plan right; bytes add a detail a port would guess wrong)

### E-1 — talker QK-Norm, MLP, bias, and residual schema (**resolves OQ-3**)
Plan §2.3 listed QK-Norm presence/eps and MLP details as `[OPEN]`. QK-Norm is present in both
attention modules, as `Qwen3TTSRMSNorm` over **`head_dim` only** (not the full projection width),
applied **after projection and before RoPE**:
- talker: `modeling_qwen3_tts.py:740-780`
- microdecoder: `:898-933` — `# unlike olmo, only on the head dim!`
```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states   = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
```
`Qwen3TTSRMSNorm` is weight-only (no additive bias) and computes
`weight * x / sqrt(mean(x^2) + eps)` in f32 before returning to the input dtype
(`:595-610`). `eps = config.rms_norm_eps` = **1e-06** for both the talker and
`code_predictor_config` (`hf/config.json:90,147`). Per doctrine #2 these norms stay high precision.

The repeated talker layer is pre-norm and has two residual additions:
```text
r0 = x
a  = attention(RMSNorm(x))                 # Q/K head-norm, then mRoPE, then GQA
x1 = r0 + a
r1 = x1
m  = down_proj(SiLU(gate_proj(RMSNorm(x1))) * up_proj(RMSNorm(x1)))
out = r1 + m
```
The source constructs `gate_proj`, `up_proj`, and `down_proj` with `bias=False` and evaluates
exactly `down_proj(act_fn(gate_proj(x)) * up_proj(x))` (`modeling_qwen3_tts.py:842-855`); with
`hidden_act: "silu"` (`hf/config.json:136`), this is SwiGLU with **the gate projection receiving
SiLU and the up projection ungated**. The layer order and both residual additions are explicit at
`:1348-1417`.

There are **no additive biases inside the 28 repeated talker blocks**: `attention_bias: false`
(`hf/config.json:20`) makes Q/K/V/O weight-only (`modeling_qwen3_tts.py:740-750`), and the MLP is
weight-only as above. This is corroborated by the pinned checkpoint header: layers 0 and 27 contain
only `input_layernorm.weight`, `post_attention_layernorm.weight`, `self_attn.{q,k,v,o}_proj.weight`,
`self_attn.{q,k}_norm.weight`, and `mlp.{gate,up,down}_proj.weight`. Do not generalize this to the
whole wrapper: the outer `text_projection.linear_fc1` and `linear_fc2` are intentionally constructed
with `bias=True` (`modeling_qwen3_tts.py:1575-1577`) and the pinned checkpoint contains both biases;
the primary `codec_head` is weight-only (`:1579`).

**Fixture seams for P1 talker L1/L2:** dump and compare (1) each norm input/output, including Q/K
after projection+reshape and after head-RMSNorm but before mRoPE; (2) mRoPE Q/K; (3) attention output
before and after `o_proj`; (4) the first residual sum; (5) post-attention RMSNorm; (6)
`gate_proj`, `SiLU(gate_proj)`, `up_proj`, their elementwise product, and `down_proj`; and (7) the
second residual sum. This isolates an incorrect head-axis norm, gate/up swap, missing bias, or
residual ordering before a fuzzy end-to-end L2 failure.

### E-2 — the talker's primary-code vocab is **3072**, not 2048
`talker_config.vocab_size: 3072` (`codec_embedding = nn.Embedding(3072, 1024)` `:1441`,
`codec_head = nn.Linear(1024, 3072, bias=False)` `:1579`), while the microdecoder's per-depth heads
are `vocab_size: 2048` (`code_predictor_config`). The extra 1024 slots hold control tokens, all
enumerated in `config.json`: `codec_pad_id 2148`, `codec_bos_id 2149`, `codec_eos_token_id 2150`,
`codec_think_id 2154`, `codec_nothink_id 2155`, `codec_think_bos_id 2156`, `codec_think_eos_id 2157`,
and ten `codec_language_id` entries (english 2050, german 2053, spanish 2054, chinese 2055,
japanese 2058, french 2061, korean 2064, russian 2069, italian 2070, portuguese 2071). Text-side
specials: `tts_pad 151671`, `tts_bos 151672`, `tts_eos 151673`, `im_start 151644`, `im_end 151645`,
`assistant 77091`. The plan's "≈3 MB Q8" head figure was right; the *shape* was never stated.
Feeds the sampler bead (`frankentts-p1-sampler-f1e`) — stop detection is `codec_eos_token_id 2150`
on the **talker** head, and the two samplers have **different vocab widths**.

### E-3 — the training forward *is* the FrankenMTP block verifier (direct support for OQ-5 / §7.5)
`forward_sub_talker_finetune` (`:1612-1633`) builds a **single 16-position causal sequence**
`[talker_hidden, talker_emb(code0), depth_emb[0](code1), …, depth_emb[13](code14)]`, runs
`code_predictor.forward_finetune(...)` once, and takes `labels = codec_ids[:, 1:]`. So one causal
pass over the frame yields all 15 residual logits — exactly the verification primitive §7.5 needs,
and it is upstream-exercised code, not an inference we are making. **The claim tier is
"architecturally supported by the pinned source"**; bit-exactness of drafted-vs-sequential logits
still has to be *measured* (the sequential path remains authoritative per §7.5).

### E-4 — microdecoder sampling params come from `generation_config.json`, not the nested config
`code_predictor_config` carries stale HF defaults (`temperature 1.0`, `do_sample false`), but
`:1674-1678` passes `subtalker_dosample / subtalker_top_p / subtalker_top_k / subtalker_temperature`
explicitly into `code_predictor.generate(...)`. `generation_config.json` supplies
`subtalker_temperature 0.9`, `subtalker_top_k 50`, `subtalker_top_p 1.0`, `subtalker_dosample true`.
The plan's T0.9/k50/p1.0 is right; **the precedence is the trap** — a port reading the nested config
would sample greedily at T=1.0. Also: `generation_config.json` sets `max_new_tokens: 8192`
(talker frames) while `qwen3_tts_model.py:2031` defaults `max_new_tokens: 4096` — two different
caps; the plan's long-form chunking (OQ-6) must state which it honors. Note the README's eval used
`max_new_tokens=2048` (`gh/README.md:465`) — a **third** value.

### E-5 — the codec encoder and decoder are structurally different (not mirror images)
`encoder_config`: 8 layers, hidden 512, **intermediate 2048** (decoder: 1024), 8 heads (decoder: 16),
**`hidden_act: "gelu"`** (decoder: `"silu"`), **`sliding_window: 250`** (decoder: 72),
`upsampling_ratios [8,6,5,4]` = 960 (decoder `upsample_rates [8,5,4,3]` = 480),
`codebook_dim 256` (decoder 512), `use_causal_conv: true`, `num_filters 64`, `kernel_size 7`,
`last_kernel_size 3`, `compress 2`, `dilation_growth_rate 2`, `num_residual_layers 1`.
The decoder is **MHA, not GQA**: `num_key_value_heads: 16` = `num_attention_heads: 16`.
Also `encoder_config.num_quantizers: 32` vs top-level `encoder_valid_num_quantizers: 16` — the
encoder RVQ is built with 32 levels but only 16 are valid. The plan's "separate geometry" was right;
these are the numbers. Feeds OQ-7 and the enrollment-only encoder build.

---

## Still open after the pin (config alone cannot settle these)

- **S-1 (OQ-7) — semantic ID mapping is a genuine puzzle, now with evidence.**
  `decoder_config.semantic_codebook_size: 4096` and `num_semantic_quantizers: 1`, but the talker's
  head is 3072-wide with control tokens occupying 2048–2157, and the acoustic `codebook_size` is
  2048. So the codec's semantic codebook has 4096 entries while the model that *produces* semantic
  codes can emit at most ~2048 valid ones. The split RVQ is real
  (`modeling_qwen3_tts_tokenizer_v2.py:797-820`: `rvq_first` over `n_q_semantic=1`, `rvq_rest` over
  the other 15, decoded and **summed**). Whether the extra 2048 semantic rows are unused, or a
  remapping applies, must be settled by reading the quantizer construction and inspecting the
  checkpoint's actual codebook tensor shapes. **This is now OQ-7's sharpest question.**
- **S-2 (OQ-4) — mRoPE position schedule.** `get_rope_index` exists at `:1746` with
  `delta0 = (1 - attention_mask).sum(dim=-1)` and a `self.rope_deltas` cache; the 3-D id assignment
  over a real prompt still needs to be traced or executed. Not settled by config.
- **S-3 (OQ-8) — watermarking: no evidence of any.** A case-insensitive grep for
  `watermark|steganog|perth|audioseal` across the entire `qwen_tts/` package and `examples/` returns
  **zero hits**. Strong negative evidence, but "no watermarking code in the inference path" is not
  the same as "no watermark in the weights"; OQ-8's residual scope is a signal-level check, and the
  doctrine-#10 obligation (never strip one) is unaffected.
- **S-4 (OQ-14) — streaming internals.** `qwen3_tts_model.py` exposes `non_streaming_mode` with
  *different defaults in two entrypoints* (`:478` `False`, `:642` `True`). The paper's 4-frame/320 ms
  packet claim was not located in the code by grep; it needs the PDF (`snapshots/paper/`) plus
  `cli/demo.py`. Unresolved.
- **S-5 (OQ-15) — oracle pins.** No upstream `torch` pin exists to inherit (see PIN_RECORD). We must
  choose and freeze one ourselves; unpinned `librosa` is the sharpest risk because it supplies the
  speaker encoder's mel filterbank (`modeling_qwen3_tts.py:436` uses `librosa`-style
  `sr/n_fft/n_mels/fmin/fmax`), which OQ-9 depends on.
- **S-6 (§2.8, OQ-6) — long-form.** The 25 Hz-vs-12 Hz long-speech result lives in the paper PDF,
  which is snapshotted and hashed but not yet read. `tokenizer_25hz/` is snapshotted for the §2.8
  promotion trigger.
- **S-7 — `embedding_dim` for the per-depth embeddings** is a constructor argument
  (`:1018`, `:1031`), not a config field. It is 1024 by construction (the 16 embeddings are summed
  with the talker's hidden-size-1024 input, C-1), but confirm against the checkpoint tensor shapes
  during the OQ-2 census rather than assuming.
