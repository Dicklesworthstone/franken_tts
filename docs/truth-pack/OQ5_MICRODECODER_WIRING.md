# OQ-5 — Microdecoder wiring + training-mask equivalence

**Bead:** `frankentts-oq5-microdecoder-wiring-hls` · **Resolved by:** AzureThrush · **Date:** 2026-08-06

**Verdict: TIER 1 — the training-mode causal forward is mask- and distribution-equivalent to the
15-step sequential inference loop.** FrankenMTP may be claimed as an **exact** block verifier
(strict-mode token identity + sampled-mode distributional exactness), subject to the single
floating-point caveat in §6.

All line citations are to the pinned truth-pack snapshot
`docs/truth-pack/snapshots/gh/qwen_tts/core/models/modeling_qwen3_tts.py`
(GitHub `QwenLM/Qwen3-TTS`, pinned per `docs/truth-pack/PIN_RECORD.md`), and to
`docs/truth-pack/snapshots/hf/config.json` for `talker_config.code_predictor_config`.

Status of this document's claims: **[VERIFIED-STATIC]** — derived by line-level reading of the
pinned source and its pinned config. The numeric oracle trace required by the bead's third exit
criterion is **NOT** included and is **NOT** claimed; see §8.

---

## 0. Config facts this rests on (pinned)

From `config.json → talker_config.code_predictor_config`:

| Field | Value | Consequence |
|---|---|---|
| `num_hidden_layers` | 5 | the 5-layer body |
| `hidden_size` | 1024 | **equals** `talker_config.hidden_size` (1024) |
| `intermediate_size` | 3072 | SwiGLU MLP |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 8 / 128 | GQA, attention width 2048 > hidden |
| `num_code_groups` | 16 | ⇒ **15** embeddings, **15** heads, **15** sequential steps |
| `vocab_size` | 2048 | the 2,048-way residual heads |
| `layer_types` | `["full_attention"] × 5` | **no sliding attention on any layer** |
| `sliding_window` / `use_sliding_window` | `null` / `false` | idem |
| `rope_theta` / `rope_scaling` | `1000000` / `null` | plain RoPE, no scaling |
| `rms_norm_eps` | `1e-06` | |
| `attention_bias` | `false` | no QKV/O biases |
| `top_k` / `top_p` / `temperature` | 50 / 1.0 / 1.0 | overridden at call site — see §7 |

`talker_config.vocab_size = 3072` (not 2048) — see §5, this settles an OQ-2 open item.

---

## 1. (a) Exact conditioning inputs per depth

The microdecoder consumes a **16-position sequence**. Position 0 is the talker hidden state;
positions 1..15 are code-group embeddings. The answer to "talker hidden? primary-code embedding?"
is **both, at two different sequence positions** — not summed, not concatenated.

| seq pos `p` | input vector (before the projection of §3) | scored? | head | predicts |
|---|---|---|---|---|
| 0 | `talker_hidden_state` (talker's last hidden at this frame) | no | — | — |
| 1 | `talker.get_input_embeddings()(c0)` — **the TALKER's codec embedding**, vocab 3072 | yes | `lm_head[0]` | `c1` |
| 2..15 | `code_predictor.codec_embedding[p-2](c_{p-1})` — vocab 2048 | yes | `lm_head[p-1]` | `c_p` |

`c0` is the primary code sampled by the talker; `c1..c15` are the 15 residual codes.

**Inference construction** (`Qwen3TTSTalkerForConditionalGeneration.forward`, lines 1670–1680):

```python
last_id_hidden = self.get_input_embeddings()(input_ids)          # talker embed of c0
predictor_result = self.code_predictor.generate(
    inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),   # positions 0 and 1
    max_new_tokens=self.config.num_code_groups - 1,                  # 15
    ...
)
```

`past_hidden` is set at line 1740 as `hidden_states[:, -1:, :]` — the talker's own last hidden
state for the frame. So prefill of the microdecoder is exactly `[talker_hidden, embed(c0)]`.

**Training construction** (`forward_sub_talker_finetune`, lines 1619–1626):

```python
sub_talker_inputs_embeds = [talker_hidden_states.unsqueeze(1)]              # position 0
for i in range(self.config.num_code_groups - 1):                           # i = 0..14
    if i == 0:
        sub_talker_inputs_embeds.append(self.get_input_embeddings()(codec_ids[:, :1]))          # pos 1 = talker embed of c0
    else:
        sub_talker_inputs_embeds.append(self.code_predictor.get_input_embeddings()[i-1](codec_ids[:, i:i+1]))  # pos i+1
sub_talker_inputs_embeds = torch.cat(sub_talker_inputs_embeds, dim=1)
```

Substituting `p = i + 1` gives `codec_embedding[p-2](c_{p-1})` for `p = 2..15` — **identical to the
inference table above.**

> **Implementation consequence (load-bearing):** the microdecoder engine needs a row of the
> **talker's** 3072-vocab codec embedding table for position 1. It is *not* one of the 15
> `codec_embedding` tables. An implementation that reaches for `codec_embedding[?]` at position 1
> is wrong. This is the single easiest way to get this module subtly wrong.

## 2. (b) KV reset semantics

**Full reset every frame, by construction.** `code_predictor.generate(...)` (line 1671) is invoked
fresh per talker step and is **never** passed `past_key_values`. `Qwen3TTSTalkerCodePredictorModel.forward`
then allocates a new cache (lines 1082–1083):

```python
if use_cache and past_key_values is None:
    past_key_values = DynamicCache()
```

`cache_position` therefore restarts at 0 each frame (lines 1085–1089) and `position_ids` follow it
(lines 1091–1092). Maximum extent is 16 positions × 5 layers × (8 KV heads × 128) — the plan's
"tiny, cache-resident, reset once per frame" is confirmed exactly.

## 3. (c) Head/embedding layout and the depth→sequence→head index map

- `codec_embedding` (line 1030–1032): `ModuleList[15]` of `nn.Embedding(vocab_size=2048, embedding_dim)`
  where **`embedding_dim = talker_config.hidden_size = 1024`**, passed in at line 1165. Note the
  embedding width is the *talker's* width, not the predictor's — they coincide here (both 1024) but
  the code treats them as distinct.
- `lm_head` (line 1167–1169): `ModuleList[15]` of `nn.Linear(hidden_size=1024, vocab_size=2048, bias=False)`.
- `small_to_mtp_projection` (lines 1171–1174):
  ```python
  if config.hidden_size != talker_config.hidden_size:
      self.small_to_mtp_projection = torch.nn.Linear(talker_config.hidden_size, config.hidden_size, bias=True)
  else:
      self.small_to_mtp_projection = torch.nn.Identity()
  ```
  **For this checkpoint 1024 == 1024, so it is `nn.Identity()` — a no-op with no weights.** It is
  applied to *every* position in both paths (line 1217 finetune, line 1282 inference), so it cannot
  break equivalence. Flagged because the plan does not mention it and a converter that expects the
  tensor to exist will fail the manifest census; it must be recorded as *legitimately absent*, not
  MISSING.

**The index map, stated once:**

```
position p ∈ [0, 15]
  p = 0            → conditioning: talker hidden state, no head
  p = 1            → embed: talker codec embedding (vocab 3072) of c0 ; head lm_head[0]     → c1
  p ∈ [2, 15]      → embed: codec_embedding[p-2] (vocab 2048) of c_{p-1} ; head lm_head[p-1] → c_p
equivalently, for residual index j ∈ [1, 15]: c_j is produced at position j by lm_head[j-1],
and (once sampled) re-enters at position j+1 through codec_embedding[j-1].
```

## 4. (d) The training-mode pass's exact masking and position ids

`forward_finetune` (lines 1197–1247) calls `self.model(...)` with `attention_mask=None`,
`position_ids=None`, `past_key_values=None`, `cache_position=None`. Inside
`Qwen3TTSTalkerCodePredictorModel.forward` this yields:

- `cache_position = torch.arange(0, 16)` (lines 1085–1089, `past_seen_tokens = 0`);
- `position_ids = cache_position.unsqueeze(0)` → `0..15` (lines 1091–1092);
- mask (lines 1095–1110): `causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}`.
  `self.has_sliding_layers` (line 1029) is `"sliding_attention" in config.layer_types`, which is
  **False** for this config (§0), so no sliding mask is ever built and every layer reads
  `causal_mask_mapping["full_attention"]` (line 1127).

⇒ **a plain 16×16 lower-triangular causal mask over absolute positions 0..15.**

In inference, `generate` prefills positions 0–1 (`cache_position = [0,1]`) and then appends one
position per step with `cache_position = 2,3,...,15`, each attending to all previous — the same
lower-triangular structure over the same absolute positions. RoPE is applied from `position_ids` in
both paths through the same `Qwen3TTSRotaryEmbedding` (line 1027) and `apply_rotary_pos_emb`
(line 858, plain — **not** the talker's `apply_multimodal_rotary_pos_emb`).

**Head application, both paths:**

```python
# forward_finetune, lines 1235-1238  (training: all 15 at once)
logits = []
for i in range(1, self.config.num_code_groups):      # i = 1..15
    logits.append(self.lm_head[i-1](hidden_states[:, i]))
logits = torch.stack(logits, dim=1)

# forward, line 1299  (inference: one at a time)
logits = self.lm_head[generation_steps](hidden_states)
```

with `generation_steps` initialised at line 1278 to `inputs_embeds.shape[1] - 2 = 0` on prefill
(the comment reads `# hidden & layer 0`), advanced by `generation_steps + 1` at line 1311 and
propagated at lines 1314–1319. At step `k` the new token lands at position `p = k+1` and uses
`lm_head[k] = lm_head[p-1]` — **identical to the training path's `position i → lm_head[i-1]`.**

## 5. (e) Equivalence verdict — **TIER 1**

Both paths, position by position:

| | training (`forward_finetune`) | inference (15-step `generate`) |
|---|---|---|
| sequence length | 16 | 16 (2 prefill + 14 appended) |
| position ids | `0..15` | `0..15` |
| mask | full causal, no sliding | full causal, no sliding |
| rotary | plain RoPE, θ=1e6, no scaling | identical |
| input at `p=0` | talker hidden | talker hidden |
| input at `p=1` | talker embed(c0) | talker embed(c0) |
| input at `p≥2` | `codec_embedding[p-2](c_{p-1})` | `codec_embedding[p-2](c_{p-1})` |
| projection | `small_to_mtp_projection` (Identity) all positions | idem |
| head at `p` | `lm_head[p-1]` | `lm_head[p-1]` |
| body weights | the same 5 layers | the same 5 layers |

Because the mask is causal, position `p`'s hidden state depends only on inputs at `0..p`; those
inputs are the same function of `(talker_hidden, c0..c_{p-1})` in both paths; therefore the hidden
state — and hence the logits — at every scored position are **mathematically identical**.

**⇒ FrankenMTP Tier 1 is licensed.** The training-mode single causal pass is an exact block
verifier for a drafted residual sequence. Strict-mode token identity holds by construction;
sampled-mode distributional exactness is licensed *at the logits level* (the remaining work is
OQ-19's speculative-sampling rule over those logits, which is unaffected by this verdict).

**The plan's "1 + 15 = 16" framing is correct in count but should be restated in the spec:** there
is **one** unscored conditioning position (p=0, the talker hidden state) and **15** scored positions
(p=1..15). Position 1 is *not* an unscored primary-code slot — it carries `embed(c0)` **and** is
scored by `lm_head[0]`. Plan §7.5's phrase "one conditioning/primary position … plus 15 residual
positions" reads as though c0's embedding were unscored; the arithmetic is the same but an
implementer following the prose could easily place the heads one position off.

## 6. The one caveat (numerical, not semantic)

Equivalence is **exact in exact arithmetic**. In floating point, a seq-16 batched matmul and 15
separate m=1 GEMVs may differ in reduction order and thus in the last ULPs. This is a Contract-A
tolerance question (§9.1/§9.3 of the plan), not a semantic divergence, and in our own Rust
implementation we control both reduction orders. Consequence for the strict-mode claim: the
FrankenMTP acceptance test must compare **argmax/token ids**, not raw logit bits, unless the verify
kernel is built to reproduce the sequential reduction order exactly. Record as a DISCREPANCIES entry
when the verify kernel lands (`frankentts-k-verify-kernel-0i0`).

## 7. Sampling parameters actually used (call site beats config)

`Qwen3TTSForConditionalGeneration.generate` (lines 2022–2046) defaults:
`do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05` for the talker, and
`subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9`
threaded to `code_predictor.generate` at lines 1674–1677. **No repetition penalty on the
microdecoder.** The `code_predictor_config` values (`temperature=1.0`, `repetition_penalty=1.0`) are
inert defaults and must not be used. This confirms plan §2.5.

## 8. What is NOT established here (honest gap)

The bead's exit criteria require **"a worked trace of one frame (inputs/outputs per depth) captured
from the oracle."** That is **not** in this document and is **not** claimed.

- No oracle environment exists yet on this host: `python3 -c "import torch"` →
  `ModuleNotFoundError: No module named 'torch'`, and no `model.safetensors` is present (the truth
  pack stores LFS hashes, not blobs).
- Standing up the oracle is `frankentts-t-oracle-fixtures-6w9`'s job.

What replaces it *in part*: `scripts/oq5_index_equivalence.py`, an executable, dependency-free
replay of **both** code paths' index arithmetic that asserts the position → embedding → head maps
are identical (§3). That is a real guard against the epic-invalidating off-by-one, and it runs in
CI today. It is **not** a numeric oracle capture: it proves the index algebra, not the activations.

The numeric trace is tracked as `frankentts-oq5-oracle-trace-*` (filed as a follow-up, blocked on
the oracle fixtures bead). **OQ-5 stays open until that lands.**

## 9. Findings owed to other beads (propagated)

| Finding | Owed to |
|---|---|
| `talker_config.vocab_size = 3072` (not 2048) — settles the "~3 MB primary head" oddity: the talker codec head is 1024×3072 | `frankentts-oq2-tensor-inventory-ght` |
| Hot-working-set composition: body ≈78.6 MB + 15 heads ≈31.5 MB ≈ **110.1 MB** of *traffic*; the 15 embedding tables are a further ≈31.5 MB of *footprint* but only 15 rows (~15 KB) are touched per frame. The plan's "~110 MB-class" is right for traffic; the ≈141.6 MB footprint figure is the one `.fttspack` must place | `frankentts-oq2-tensor-inventory-ght`, `frankentts-k-rcd-engine-6e3` |
| Position 1 needs a row of the **talker's** 3072-vocab embedding — an extra tensor in the microdecoder hot pack | `frankentts-oq2-tensor-inventory-ght`, `frankentts-k-rcd-engine-6e3` |
| `small_to_mtp_projection` is `Identity` for this checkpoint — census must treat it as legitimately absent, not MISSING | `frankentts-oq2-tensor-inventory-ght`, `frankentts-p1-weights-qjl` |
| **QK-Norm is present**: `q_norm`/`k_norm` are `RMSNorm(head_dim=128, eps=1e-6)` applied per-head *before* RoPE (lines 910–913, 928–933); `attention_bias=false` ⇒ no QKV/O biases; MLP is SwiGLU `gate/up/down`, `hidden_act="silu"` (lines 842–856). Verified for the `Qwen3TTSAttention`/`Qwen3TTSDecoderLayer` class used by the **microdecoder**; the talker uses the separate `Qwen3TTSTalkerAttention` (line 727) and must be confirmed independently | `frankentts-oq3-talker-details-bmg` |
| Microdecoder overflow-K bounds are the same as the talker's: `down_proj` K=3072, `o_proj` K=2048 | `frankentts-p2-overflow-proofs-0o1` |
| Tier-1 verdict ⇒ exact-verifier claims are licensed | `frankentts-k-verify-kernel-0i0`, `frankentts-k-oq19-sampled-rule-w4q`, `frankentts-b-accept-surface-596`, `frankentts-b-metal-spike-zz8`, `frankentts-p1-microdecoder-xst` |
