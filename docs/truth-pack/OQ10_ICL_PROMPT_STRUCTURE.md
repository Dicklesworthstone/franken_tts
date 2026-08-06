# OQ-10 — ICL prompt structure × streaming mode, and the maximal target-independent prefix

**Bead:** `frankentts-oq10-icl-prompt-lkq` · **Resolved by:** AzureThrush · **Date:** 2026-08-06

**Headline result:** the maximal target-text-independent prefix is
**`H + min(T2, |ref_id|)`** positions in the two ICL modes and **exactly `H`** in the two x-vector
modes, where `H` is the fixed 7–9-position header. **The reference codec frames are almost entirely
OUTSIDE the cacheable prefix** — in streaming mode they are summed position-wise with target-text
embeddings, and in non-streaming mode they sit *after* the target text in causal order. Plan §6.7's
demotion of prompt-KV caching from "assumed easy" to "proven-prefix-only" is **confirmed and now
quantified**: a `.ftvoice-cache` may cache roughly `|ref_id|` positions, not the whole voice prompt.

Line citations are to the pinned truth-pack snapshots
`docs/truth-pack/snapshots/gh/qwen_tts/core/models/modeling_qwen3_tts.py` (`M:`) and
`docs/truth-pack/snapshots/gh/qwen_tts/inference/qwen3_tts_model.py` (`I:`), plus
`docs/truth-pack/snapshots/hf/{config,tokenizer_config}.json`.

Status: **[VERIFIED-STATIC]** for the prompt-assembly structure (line-level reading of the pinned
source; no model weights on this host, so nothing was executed through the model) — and
**[VERIFIED-MEASURED]** for the tokenizer-facing claims in §0.1 and §7, which were run against the
pinned tokenizer bytes with the pinned oracle (`transformers==4.57.3` / `tokenizers==0.22.2`,
`fix_mistral_regex=True`).

---

## 0. Building blocks

### 0.1 Text wrappers and what the magic slice indices mean (`I:269-276`)

```python
def _build_assistant_text(self, text):  return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
def _build_ref_text(self, text):        return f"<|im_start|>assistant\n{text}<|im_end|>\n"
def _build_instruct_text(self, instruct):return f"<|im_start|>user\n{instruct}<|im_end|>\n"
```

The wrapper tokenizes to a **3-token prefix** `[151644, 77091, 198]` = (`<|im_start|>`, `assistant`,
`\n`) and, for the assistant form, a **5-token suffix** `[151645, 198, 151644, 77091, 198]` =
(`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`); the ref form has a **2-token suffix**
`[151645, 198]`. That is the whole explanation for the asymmetric slicing in `generate`:

| slice | meaning |
|---|---|
| `input_id[:, :3]` (`M:2178`) | the role header `<|im_start|>assistant\n` |
| `input_id[:, 3:-5]` (`M:2190`, `M:2207`, `M:2212`) | **exactly the target text tokens** |
| `input_id[:, 3:4]` (`M:2201`) | the first target text token |
| `input_id[:, 4:-5]` (`M:2231`) | target text minus its first token |
| `ref_ids[index][:, 3:-2]` (`M:2191`) | **exactly the reference transcript tokens** |

These indices are hard-coded and assume that wrapper tokenization. A tokenizer that splits
`<|im_start|>assistant\n` differently silently corrupts every prompt — a Contract-A L0 concern.

**MEASURED, not assumed.** Run against the pinned tokenizer bytes
(`docs/truth-pack/snapshots/hf/`) with the pinned oracle (`transformers==4.57.3`,
`tokenizers==0.22.2`) using **`fix_mistral_regex=True`** — the flag the official stack loads with
(`I:118`, per cc_1's OQ-11 finding):

```
<|im_start|>assistant\nHello there, friend.<|im_end|>\n<|im_start|>assistant\n
  -> [151644, 77091, 198,  9707, 1052, 11, 4238, 13,  151645, 198, 151644, 77091, 198]
     |<-- [:3] -->|        |<------ [3:-5] ------>|   |<--------- [-5:] ----------->|
     [3:-5] == tokenize("Hello there, friend.")  ✓

<|im_start|>assistant\nThis is the reference recording.<|im_end|>\n
  -> [151644, 77091, 198,  1986, 374, 279, 5785, 14633, 13,  151645, 198]
     [3:-2] == tokenize("This is the reference recording.")  ✓
```

Verified to hold for ASCII, CJK, digits/punctuation, leading whitespace, **empty text**, embedded
newlines, and emoji — 7/7 classes, under both the default and `fix_mistral_regex=True` regexes (the
wrapper boundary is regex-independent because `<|im_start|>`/`<|im_end|>` are *added tokens*, split
before BPE runs; only the inner text ids are regex-sensitive, which is OQ-11's territory).

⇒ **The 3 / 5 / 2 wrapper structure is CONFIRMED**, and the slice arithmetic in §1–§3 rests on
measured ground.

### 0.2 Special token ids (`config.json`, `tokenizer_config.json`)

| config field | id | token content |
|---|---|---|
| `tts_bos_token_id` | 151672 | `<tts_text_bos>` |
| `tts_eos_token_id` | 151673 | **`<tts_text_eod>`** |
| `tts_pad_token_id` | 151671 | `<tts_text_pad>` |
| `im_start_token_id` / `im_end_token_id` | 151644 / 151645 | |
| `assistant_token_id` | 77091 | |

⚠ **Naming discrepancy:** `tts_eos_token_id` maps to a token whose literal content is
`<tts_text_eod>` (**eod**, not eos). Use the id, never the name. Ledger this when the prompt builder
lands.

`tts_bos_embed`, `tts_eos_embed`, `tts_pad_embed` are produced once by embedding
`[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]` through the text path and `.chunk(3, dim=1)`
(`M:2113-2124`).

### 0.3 cc_1's LEAD 1 resolved — `<tts_text_bos_single>` is UNUSED

`<tts_text_bos_single>` (id **151674**) appears **only** as an added-token declaration in
`tokenizer_config.json:253-254,288`. `grep` over the entire pinned GitHub snapshot (modeling,
inference, processing, finetuning, examples) and over `config.json` returns **no reference to the
token or to the id 151674**.

⇒ **There is no separate single-segment prompt template in this checkpoint's inference path.** The
token is vestigial (or reserved for a sibling checkpoint). **Do not invent a template around it.**
Recorded so the question is closed rather than rediscovered.

### 0.4 cc_1's LEAD 2 resolved — which mode is the default

| entrypoint | `non_streaming_mode` default | applies to |
|---|---|---|
| `generate_voice_clone` (`I:470-479`) | **`False`** ⇒ **streaming** | **the Base checkpoint — our target** |
| `generate_voice_design` (`I:637-642`) | `True` | VoiceDesign model (out of scope, plan §2.2) |
| `generate_custom_voice` (`I:733-738`) | `True` | CustomVoice model (out of scope) |

⇒ **For franken_tts's target workload (Base + voice cloning), the official default is STREAMING
mode.** The streaming template in §2 is therefore the primary path, not the exotic one. This should
drive which template `p1-prompt-igr` implements and conforms first.

---

## 1. The common header `H` (identical in all four modes)

Built before any mode branch, at `M:2126-2186`.

```
codec_prefill_list =
    language auto/None : [codec_nothink_id, codec_think_bos_id, codec_think_eos_id]          (M:2136-2140)
    language given     : [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id] (M:2142-2147)

codec_input_embedding = emb(codec_prefill_list)                       # |P| = 3 or 4
                        ++ [speaker_embed]  if speaker_embed is not None   # S = 0 or 1   (M:2166-2172)
                        ++ emb([codec_pad_id, codec_bos_id])               # 2
                                                                       # L_c = |P| + S + 2

_talker_input_embed_role = text_proj(text_emb(input_id[:, :3]))        # 3 positions, TEXT ONLY
_talker_input_embed      = (tts_pad × (L_c-2)  ++  tts_bos)  +  codec_input_embedding[:, :-1]
                                                                       # L_c - 1 positions, SUMMED
talker_input_embed = cat(role, _talker_input_embed)
```

**`H = 3 + (L_c − 1) = L_c + 2`.** Concretely: language given + speaker embed ⇒ `L_c = 7`, **`H = 9`**;
language auto + no speaker embed ⇒ `L_c = 5`, **`H = 7`**.

Two things to note, both easy to get wrong:

1. The role positions are **text-only** — no codec embedding is added to them. Summation starts at
   position 3.
2. The final element of `codec_input_embedding` (the `codec_bos_id` embedding) is **held back**
   (`[:, :-1]`). In the x-vector modes it is consumed at `M:2201`/`M:2220`; in the ICL modes it is
   **never used**, because `generate_icl_prompt` emits its own `codec_bos` (`M:1990-1998`). A builder
   that emits both will produce a duplicate `codec_bos` and a prompt one position too long.

`H` depends only on `(language_id, speaker_embed)` — **never on the target text, and never on the
streaming mode.**

---

## 2. `generate_icl_prompt` — the two ICL templates (`M:1968-2019`)

Called at `M:2188-2197` when `voice_clone_prompt["ref_code"] is not None and icl_mode[index]`.

Two streams are built first, both independent of streaming mode:

```python
# TEXT stream, T1 = |ref_id| + |text_id| + 1                              (M:1978-1981)
text_embed = text_proj(text_emb(cat([ref_id, text_id], dim=-1)))
text_embed = cat([text_embed, tts_eos_embed], dim=1)

# CODEC stream, T2 = 1 + T_ref                                            (M:1983-1998)
codec_embed = sum over the 16 code groups of the reference frames:
    group 0  -> talker.get_input_embeddings()          (vocab 3072)
    group i  -> code_predictor.codec_embedding[i-1]    (vocab 2048)
    summed across groups -> one vector per reference frame
codec_embed = cat([emb(codec_bos_id), codec_embed], dim=1)
```

> The per-frame 16-group **sum** here is the same operation as the decode-time feedback at
> `M:1682-1687`. One implementation serves both. Note group 0 uses the **talker's** 3072-vocab table
> — the same asymmetry OQ-5 found inside the microdecoder.

### 2.1 ICL × **non-streaming** (`non_streaming_mode=True`, `M:2002-2013`)

```python
icl_input_embed = text_embed + emb([codec_pad_id] * text_lens)      # T1 positions
icl_input_embed = cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)   # + T2 positions
return icl_input_embed, tts_pad_embed
```

**Two sequential blocks, no interleaving.** Total prompt = `H + T1 + T2`.
`trailing_text_hidden = tts_pad_embed` — the whole text is consumed in prefill.

```
positions:  [ H header ][ ref_id | text_id | eos   each + codec_pad ][ codec_bos | ref frames  each + tts_pad ]
                        |<--------------- T1 ---------------------->||<--------------- T2 ------------------>|
```

### 2.2 ICL × **streaming** (`non_streaming_mode=False`, `M:2014-2019`)

```python
if text_lens > codec_lens:
    return text_embed[:, :codec_lens] + codec_embed,  text_embed[:, codec_lens:]
else:
    text_embed = cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
    return text_embed + codec_embed,  tts_pad_embed
```

**One block of `T2` positions formed by ELEMENTWISE SUM of the two streams**, position `p` pairing
text token `p` with codec entry `p`. Total prompt = `H + T2`. Whatever text does not fit becomes the
per-frame trailing stream.

```
positions:  [ H header ][ text[0..T2-1]  +  (codec_bos, ref frame 0, ref frame 1, ...) ]
                        |<------------------------ T2 ------------------------------->|
trailing:               text[T2 ..]  (fed one embedding per generated frame — see §4)
```

**This is the structural difference plan §6.7 flagged.** Non-streaming concatenates the streams;
streaming superposes them. The prompt lengths differ (`T1+T2` vs `T2`), the arithmetic differs, and
the trailing stream differs. A cache keyed without the streaming mode is corrupt — now with a
mechanism, not just a warning.

---

## 3. The two x-vector templates (`M:2198-2232`)

Taken when there is no `ref_code` or `icl_mode` is false — i.e. `x_vector_only_mode`, or a named
speaker. The speaker identity has already entered via `speaker_embed` inside `H` (§1).

### 3.1 x-vector × **streaming** (`M:2200-2202`, `2228-2232`)

```python
talker_input_embed = cat([talker_input_embed,
                          text_proj(text_emb(input_id[:, 3:4])) + codec_input_embedding[:, -1:]])
trailing_text_hidden = cat([text_proj(text_emb(input_id[:, 4:-5])), tts_eos_embed])
```

Total prompt = **`H + 1`**. The single added position carries the **first target text token** summed
with `codec_bos`. Everything after the first token streams in per frame.

### 3.2 x-vector × **non-streaming** (`M:2203-2227`)

```python
talker_input_embed = talker_input_embed[:, :-1]   # 去掉原本放进去的text — drop the position just added
talker_input_embed = cat([talker_input_embed,
    cat([text_proj(text_emb(input_id[:, 3:-5])), tts_eos_embed]) + emb([codec_pad_id] * (|text|+1)),
    tts_pad_embed + emb([codec_bos_id])])
trailing_text_hidden = tts_pad_embed
```

Total prompt = **`H + |text| + 2`**. Note the code *builds then removes* the streaming position
(`M:2204`) — a faithful port should simply not build it; the removal is an artifact of the shared
code path, not semantics.

---

## 4. The per-frame trailing-text stream (confirms cc_1's C-1)

`M:1689-1692`, inside the talker's per-frame `forward`:

```python
if generation_step < trailing_text_hidden.shape[1]:
    inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
else:
    inputs_embeds = inputs_embeds + tts_pad_embed
```

where `inputs_embeds` is the sum of the 16 code-group embeddings of the frame just produced
(`M:1682-1687`). So **the talker input at frame `n` is
`sum(16 code embeddings of frame n-1) + trailing_text_hidden[n]`** (or `tts_pad_embed` once the
trailing text is exhausted).

⇒ **The text conditioning is not a prefill-only prefix.** It is a prefix *plus* a per-frame stream,
and which text lands where is decided by the mode:

| mode | text in prefill | text in the per-frame stream |
|---|---|---|
| ICL × non-streaming | all of it | none (`tts_pad` forever) |
| ICL × streaming | `text_embed[:T2]` — the part that fits under the codec stream | `text_embed[T2:]` |
| x-vector × streaming | first token only | the rest + `tts_eos` |
| x-vector × non-streaming | all of it | none (`tts_pad` forever) |

**Consequence for TTFA work** (`frankentts-k-voice-cache-i4t`): in streaming mode the prompt is short
(`H + T2`, independent of target-text length) and the text arrives frame by frame. In non-streaming
mode the whole text is prefilled, so prefill cost grows with text length. These are genuinely
different TTFA profiles and must be benchmarked separately — reinforcing plan §10.1's per-profile
ranking.

---

## 5. The maximal target-text-independent prefix (the proof)

**Claim.** Under the causal mask, a KV entry at position `p` is a function of the inputs at
positions `0..p` only. So a prefix is cacheable across different target texts iff every input vector
at positions `0..p` is independent of `text_id`. Walking the construction:

**Positions `0..2` (role).** `text_proj(text_emb(input_id[:, :3]))` — the fixed wrapper
`<|im_start|>assistant\n`. Independent. ✔

**Positions `3..H-1` (header).** `tts_pad`/`tts_bos` summed with `codec_input_embedding[:-1]`, i.e.
think tags, `language_id`, optional `speaker_embed`, `codec_pad`. Functions of `(language, speaker)`
only. Independent. ✔

**Position `H` onward — mode by mode:**

- **ICL × streaming.** Block position `j` is `text_embed[j] + codec_embed[j]`. `text_embed` is
  `[ref_id tokens ..., text_id tokens ..., eos]`, so `text_embed[j]` is target-independent **iff
  `j < |ref_id|`**. `codec_embed[j]` is always target-independent. Hence block positions
  `0 .. min(T2, |ref_id|) − 1` are independent, and position `|ref_id|` is the **first** contaminated
  one (it carries the first target text token).
  ⇒ **prefix = `H + min(T2, |ref_id|)`.**
- **ICL × non-streaming.** The text block is `[ref_id, text_id, eos] + codec_pad`; positions
  `0..|ref_id|−1` are independent, position `|ref_id|` is the first target token.
  ⇒ **prefix = `H + |ref_id|`.**
  The codec block (`codec_bos` + reference frames) *is* target-independent in content, but it sits
  **after** the target text in sequence order, so under the causal mask its KV depends on the target
  text. **It is not cacheable.**
- **x-vector × streaming.** Position `H` is the first target text token. ⇒ **prefix = `H`.**
- **x-vector × non-streaming.** Position `H` is the first target text token. ⇒ **prefix = `H`.**

### 5.1 What this means for `.ftvoice-cache` (the actionable verdict)

| mode | cacheable prefix | contains |
|---|---|---|
| ICL × streaming | `H + min(T2, |ref_id|)` | header, `codec_bos`, and the first `min(T2,|ref_id|)−1` **reference codec frames** |
| ICL × non-streaming | `H + |ref_id|` | header + reference transcript only — **zero reference codec frames** |
| x-vector × either | `H` (7–9 positions) | header only |

- **The prompt-KV cache is worth far less than "cache the voice prompt" implies.** For a typical 3 s
  reference — `T2 ≈ 39` frames against a `|ref_id|` of roughly 10–15 English tokens — the ICL
  streaming cache covers on the order of a third of the ICL block, and the non-streaming cache
  covers none of the codec block at all. Plan §6.7 was right to demote it; the size of the win is
  now bounded rather than hoped.
- **For x-vector modes the cache is nearly worthless** (7–9 positions). `.ftvoice-cache` should not
  ship a prefix-KV section for x-vector at all; the enrollment-side speaker embedding is the real
  reusable artifact there.
- **Cache key must include** `(language_id, speaker_embed presence/value, ref_id token ids, ref_code,
  non_streaming_mode)` on top of plan §6.7's `{voice_recipe_hash, model_hash, prompt_builder_version,
  quant_recipe, math_mode, engine_abi}`. `language_id` and `speaker_embed` sit **inside `H`**, so
  they invalidate even the header cache — easy to miss because they feel like "runtime options."
- **Two-level cache is available and free:** `H` is built identically in all four modes
  (`M:2126-2186` precedes every branch), so the header KV can be shared across streaming modes for a
  given `(language, speaker)`. Only the ICL block needs a per-mode entry.
- **The exhausted-text tail is a second, unrelated reuse opportunity:** once `generation_step` passes
  the trailing length, every frame adds the *same* `tts_pad_embed` (`M:1691-1692`). That is a
  constant, not a cache, but it means long-form frames after text exhaustion have a strictly simpler
  input path.

### 5.2 An enrollment lever falls out of the streaming template

In ICL × streaming the cacheable fraction is `min(T2, |ref_id|) / T2`. When
**`|ref_id| ≥ T2`** — the reference transcript has at least as many tokens as the reference has
codec frames — the *entire* ICL prefill block becomes target-independent and **100 % cacheable**, with
all target text pushed into the trailing stream. `scripts/oq10_prompt_shapes.py` exercises this case.

How reachable is it? `T2 = 1 + 12.5 × ref_seconds`, so the condition is roughly
`tokens_per_second_of_speech ≥ 12.5`. Ordinary English narration runs far below that, so **for typical
references this bound is not reached** and the realistic cacheable fraction is the ~30–45 % the same
script reports. Token-dense scripts (Chinese/Japanese, dense numerals, fast speech) sit closer to it.

Stated honestly: this is a **bound and a direction, not a measured win**. It says reference *selection*
has a lever on TTFA that is invisible from the kernel side — a longer-transcript reference of equal
audio quality is strictly cheaper to re-prompt. Whether it is worth anything must be measured, and
the measurement belongs to `frankentts-k-voice-cache-i4t`; the enrollment-side ranking hook is
`frankentts-p4-*` / AF-4 (plan §8.6, §10.7). Filed as an observation for those beads, not as an
optimization claim.

---

## 6. Batch left-padding (a note owed to Phase 3D)

`M:2240-2254` left-pads `talker_input_embeds` (reverse → `pad_sequence` → reverse) and builds
`talker_attention_mask` from the original lengths; `trailing_text_hiddens` are **right**-padded and
their pad slots overwritten with `tts_pad_embed` (`M:2256-2269`). Single-stream synthesis never hits
this. **Continuous batching (`frankentts-k-batching-*`, plan §7.11) does** — mixed prompt lengths
mean left-padded talker input with a mask, and right-padded trailing text with `tts_pad` fill. The
two paddings go in *opposite directions*; getting one backwards silently misaligns text against
frames. Flagged rather than solved here.

---

## 7. Worked example (concrete, token-exact)

Reference transcript `"This is the reference recording."` → **6** tokens; target text
`"Hello there, friend."` → **5** tokens (ids in §0.1). A 3 s reference at 12.5 fps → **38** codec
frames. Language given and a speaker embedding present ⇒ `|P| = 4`, `S = 1`, `L_c = 7`, **`H = 9`**.

```
|ref_id| = 6      |text_id| = 5      ref_frames = 38
T1 = 6 + 5 + 1 = 12                  T2 = 1 + 38 = 39
```

| mode | prompt positions | trailing | target-independent prefix |
|---|---:|---:|---:|
| ICL × streaming | `H + T2` = **48** | 1 (`tts_pad`; text fits under the codec stream) | `H + min(39, 6)` = **15** |
| ICL × non-streaming | `H + T1 + T2` = **60** | 1 (`tts_pad`) | `H + 6` = **15** |
| x-vector × streaming | `H + 1` = **10** | 5 (4 remaining text tokens + `eos`) | **9** |
| x-vector × non-streaming | `H + 5 + 2` = **16** | 1 (`tts_pad`) | **9** |

Reading the ICL × streaming prompt position by position:

```
pos 0..2    role: <|im_start|> assistant \n                      (text only, no codec added)
pos 3..8    tts_pad ×4, tts_bos  +  think tags, language_id, speaker_embed, codec_pad
pos 9..14   ref_id[0..5]  +  (codec_bos, ref frame 0..4)         <-- CACHEABLE ends here (pos 14)
pos 15..47  text_id[0..4], eos, tts_pad×27  +  ref frames 5..37  <-- target text enters at pos 15
trailing    tts_pad_embed, fed once per generated frame
```

The cacheable prefix is 15 of 48 positions (**31 %**), and it contains `codec_bos` plus only the
**first 5** of the 38 reference codec frames. That is the concrete shape of the §5.1 verdict.

## 8. What is NOT established here (honest gap)

- **`instruct_ids`** (`M:2094-2098`) prepends an instruct block for the VoiceDesign/CustomVoice
  paths. Out of scope for the Base checkpoint and not analyzed. If a future bead brings those models
  in, the prefix analysis must be redone — the instruct block precedes `H`.
- **No end-to-end prompt was built through the real model**, only through the real tokenizer: the
  embedding/projection arithmetic in §1–§3 is read off the source, not executed (that needs weights,
  which are absent). The *shapes* are asserted executably by `scripts/oq10_prompt_shapes.py`; the
  *values* are owed to the oracle fixtures bead and to `p1-prompt-igr`'s conformance tests.
- **`T_ref` is assumed to be `ceil(ref_seconds × 12.5)`** in the worked example. The exact
  reference-frame count comes from the codec encoder's framing, which is OQ-7's territory; it scales
  the numbers above but changes no structural conclusion.
