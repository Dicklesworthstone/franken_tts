# Qwen3-TTS 12Hz — Decode Limits, Stop Semantics, Long-Form & Admission (OQ-6)

Resolves **OQ-6** (`frankentts-oq6-context-stop-cc3`). Builds on the context/stop numbers cc_1
recorded on the bead from the truth pack, resolves the `max_new_tokens` conflict they flagged, and
adds the parts that were still open: the official long-form path, the exact stop rule, max-duration
behavior, admission formulas, and the long-form corpus boundary design.

Citation keys (paths relative to `docs/truth-pack/snapshots/`):

| Key | File |
|---|---|
| `MODEL` | `gh/qwen_tts/core/models/modeling_qwen3_tts.py` |
| `INF` | `gh/qwen_tts/inference/qwen3_tts_model.py` |
| `CFG` | `hf/config.json` |
| `GENCFG` | `hf/generation_config.json` |
| `RDME` | `gh/README.md` |
| `CODEC` | `docs/QWEN3_TTS_CODEC_SPEC.md` (OQ-7) |

---

## 1. [FINDING] There is no long-form path. The official stack has no chunking at all.

The OQ-6 question "how does the official stack handle text longer than one pass — chunking?
sentence splitting? context carryover?" has a flat answer, verified by exhaustive search of both the
inference wrapper and the modeling code:

**None of the three exist.** There is no sentence splitter, no text chunker, no context carryover,
and no sliding-window re-prompting anywhere in `INF` or `MODEL`. The entire text goes through the
talker in **one pass**; generation is bounded only by EOS and `max_new_tokens`. The only chunking in
the whole stack is on the *decode* side — the codec's `chunked_decode(chunk_size=300,
left_context_size=25)` (`CODEC` §5.2) — and that operates on already-generated codes, not on text.

This is load-bearing in two directions:

- **We must not port a long-form scheme, because there is nothing to port.** The bead's own warning
  ("an invented rolling-window scheme would silently change semantics") is the operative risk here:
  any chunking we add is *our* design, is a **semantic divergence from the reference by
  construction**, and needs a `DISC` entry plus an A/B against single-pass on the long-form drift
  gate. It cannot be presented as "matching upstream."
- **It explains the paper's long-form result.** Plan §2.8 notes the 25 Hz variant beat 12 Hz on
  long-speech evaluation. With no chunking and no carryover, long-form quality rests entirely on the
  talker's own 32 768-position context and on semantic-token stability across a single very long
  autoregressive run. That is exactly the regime where drift accumulates with nothing to arrest it.

---

## 2. [RESOLVED] The `max_new_tokens` three-way conflict — the runtime default is 8192

cc_1 flagged three values and provisionally picked the README's 2048. Tracing the actual precedence
chain resolves it differently, and the difference matters.

`INF:332-337` — precedence is **user value → `self.generate_defaults` → `hard_defaults`**:

```python
def pick(name, user_val):
    if user_val is not None:            return user_val
    if name in self.generate_defaults:  return self.generate_defaults[name]
    return hard_defaults[name]
```

`self.generate_defaults` is `model.generate_config` (`INF:120-121`), which `MODEL:1922-1936` loads
from **`generation_config.json`** — the file that ships in the repo — and `GENCFG` contains
`"max_new_tokens": 8192`.

| Value | Where | Status |
|---|---|---|
| **8192** | `GENCFG` | **the effective runtime default** — `generate_defaults` always wins over `hard_defaults` |
| 2048 | `INF:329` `hard_defaults` | **dead code** — unreachable while the checkpoint ships `generation_config.json` with the key |
| 2048 | `RDME:465` | an **explicit override** upstream passed for published evals, not a default |
| 4096 | `MODEL:2031` signature default | **unreachable via the public wrapper**, which always passes an explicit merged value |

> **Note on the docstring trap:** `INF:302-306` says defaults come from `generate_config.json`; the
> file actually loaded is `generation_config.json` (`MODEL:1924`). The docstring is wrong. Had the
> loader really looked for `generate_config.json`, no such file exists and the effective default
> would have been the 2048 hard default — a 4× difference in max duration decided by one missing
> character. Assert the filename in the oracle harness.

**Decision for this project (both values are needed, for different jobs):**

- Reproducing **published upstream quality numbers** → pass `max_new_tokens=2048` explicitly, as
  `RDME:465` did. Contract-B comparisons against published metrics must use this.
- Reproducing **default library behavior** → 8192. Contract-A oracle fixtures and any "what does the
  reference do out of the box" claim must use this.
- `ftts`'s own default → **8192**, matching runtime behavior, with the cap surfaced as an explicit
  profile parameter. Never silently pick 2048 because it is cheaper to test.

---

## 3. Stop semantics (exact)

### 3.1 The rule

`MODEL:2283-2290`:

```python
first_codebook   = talker_codes[:, :, 0]                       # group 0 only
is_stop_token    = (first_codebook == codec_eos_token_id)      # 2150
stop_indices     = torch.argmax(is_stop_token.int(), dim=1)
has_stop_token   = is_stop_token.any(dim=1)
effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])
talker_codes_list = [talker_codes[i, :length] for i, length in enumerate(effective_lengths)]
```

| Property | Value |
|---|---|
| Stop token | `codec_eos_token_id = 2150`, on the **talker's group-0 head only** (`CFG talker_config`) |
| Which groups can stop | **Only group 0.** The 15 microdecoder heads are 2048-wide with no EOS id — the residual loop is a **fixed 15 steps, never early-stopped** |
| EOS frame fate | **Excluded.** `effective_lengths = stop_indices` (the index *of* the EOS frame), and the slice is `[:length]` — the EOS frame is not decoded to audio |
| No EOS found | `effective_lengths = full generated length` — the utterance is **truncated mid-speech with no error and no signal** (see §4) |
| HF-level stop | `eos_token_id` is passed into `generate` (`MODEL:2055-2057`), so the loop also halts at EOS; the slice above is the post-hoc trim |

### 3.2 [OPTIMIZATION, exactness-preserving] skip the microdecoder on the EOS frame

In the reference, `code_predictor.generate(...)` runs unconditionally for every talker step
(`MODEL:1671-1680`), *including* the frame whose group-0 code is EOS. Those 15 residuals are then
thrown away by the §3.1 trim, and they cannot influence anything else: their only other consumer is
the next step's input embedding (`MODEL:1681-1692`), and there is no next step.

**Therefore checking group 0 for EOS *before* running the 15 microdecoder steps is exactly
equivalent** — token-identical output, one fewer microdecoder invocation per utterance. It saves
~1.18 GB of weight traffic (§`CODEC`/plan §2.6 accounting) once per utterance. Small in aggregate,
but it is free and provably exact, so it belongs in the `ResidualCodeDecoder` from day one rather
than as a later "optimization" needing its own parity argument.

### 3.3 The per-frame feedback path (needed to get stop right)

`MODEL:1681-1692` — the talker's next input embedding is:

```
inputs_embeds = sum over all 16 code embeddings of this frame          # codec_hiddens.sum(1)
              + (trailing_text_hidden[:, generation_step]  if generation_step < text_len
                 else tts_pad_embed)
```

Two consequences for stop/chunking:

- The talker is fed the **sum of all 16 code embeddings**, not just group 0 — so the microdecoder's
  output is on the talker's critical path every frame. There is no "run the talker ahead" freedom.
- The text stream is consumed **one hidden per frame**, falling back to `tts_pad_embed` once
  exhausted. That fallback is the model's only "the text is finished" signal — it is what makes EOS
  likely, but it does **not** force it. This is cc_1's C-1 fact, confirmed at `MODEL:1689-1692`.

---

## 4. Max-duration behavior — the failure mode is silent truncation

If `max_new_tokens` is reached without EOS, `has_stop_token` is `False`, `effective_lengths` becomes
the full generated length, and the codes are returned as-is. **The reference emits truncated audio
with no exception, no warning, and no flag.** The caller cannot distinguish "the model finished" from
"the model was cut off" without re-deriving it.

| Cap | Frames | Audio duration |
|---:|---:|---:|
| 2048 (README evals) | 2048 | **163.84 s** (2 m 44 s) |
| 4096 (unreachable) | 4096 | 327.68 s |
| **8192 (runtime default)** | 8192 | **655.36 s** (10 m 55 s) |
| 32768 (talker context ceiling) | 32768 − prompt | ≈ 43.7 min, **not** the binding limit |

**Requirement this places on `ftts`:** truncation must be a **first-class, reported outcome**, not a
silent one — a distinct robot-mode field and a distinct exit path, since our CLI is agent-facing and
an agent cannot hear that the audio stopped mid-word. Doctrine #0's no-counterfeit-green principle
applies to runtime output too: silently returning a cut-off utterance is a counterfeit success.
This is the concrete reliability requirement for `frankentts-v-reliability-d65`.

---

## 5. Admission formulas

Geometry from `CFG` (re-read, not inherited): talker 28 layers, 16 Q / **8 KV** heads, head_dim 128,
`max_position_embeddings` 32768; code predictor 5 layers, same head geometry; codec decoder 8 layers,
16 heads × 64, sliding window 72.

### 5.1 Talker KV (the only term that grows with duration)

```
values per token per layer = 2 (K,V) × num_key_value_heads 8 × head_dim 128 = 2048
values per token           = 2048 × 28 layers                               = 57 344
bytes per token            = 57 344 × sizeof(dtype)
                           = 112 KiB @ BF16      224 KiB @ F32
```

```
KV_talker(L_prompt, N_frames) = (L_prompt + N_frames) × 57 344 × sizeof(dtype)
```

| Scenario | Talker KV @ BF16 |
|---|---:|
| 2048-frame cap + 512-token prompt | 280 MiB |
| **8192-frame cap + 512-token prompt** | **952 MiB** |
| full 32768 context | 3.50 GiB |

### 5.2 Bounded terms (do not grow with duration)

| Component | Live state | Bytes @ BF16 |
|---|---|---:|
| Microdecoder KV (5 layers × ≤16 positions, reset per frame) | 163 840 values | **320 KiB** |
| Codec decoder KV (8 layers × window 72 × 16 × 64 × 2) | 1 179 648 values | **2.25 MiB** |
| Codec conv ring buffers (`CODEC` §6) | 11.38 code frames of context | a few hundred KiB |

The microdecoder's KV being 320 KiB confirms plan §2.5's "tiny — comfortably cache-resident", and
restates the plan's central point: **the microdecoder's problem is its weights, never its history.**

### 5.3 The admission rule

```
predicted_max_frames = min(max_new_tokens, 32768 − L_prompt)
predicted_peak_bytes = KV_talker(L_prompt, predicted_max_frames)
                     + 320 KiB + 2.25 MiB + ring_buffers + weights_resident
admit  iff  predicted_peak_bytes ≤ budget
```

Computed **before** any allocation, per the plan's reject-before-partial-allocation requirement.
Note `max_new_tokens` is the binding constraint at every realistic prompt length — the 32768 context
ceiling only binds for prompts above ~24 500 tokens, which the 8192 cap can never reach.

---

## 6. Long-form corpus — boundary design

The bead asks for corpus items that **exercise the chunking boundary**. There are four independent
boundaries in this stack, and an item that crosses one tells you nothing about the others. Design one
pair (just-under / just-over) per boundary:

| # | Boundary | Where | Corpus items |
|---|---|---|---|
| 1 | Codec `chunk_size = 300` frames (**24.0 s**) | `CODEC` §5.2 | **290 / 310 frames**, plus **600 / 610** to catch the *second* seam. Below 300 the official path equals whole-sequence decode; above it, it does not. This is the one boundary where our engine is deliberately *not* bit-equal to the reference — items here are the DISC evidence. |
| 2 | Codec `left_context_size = 25` frames | `CODEC` §5.2 | Content with strong long-range prosody spanning a seam (a sentence beginning at frame ~295 and ending at ~320) — the 25-frame context is 23× short of the transformer's 568-frame horizon, so seam artifacts show up as prosody discontinuity, not clicks. |
| 3 | Text-hidden exhaustion (`tts_pad_embed` fallback) | `MODEL:1689-1692` | (a) text much shorter than the natural audio (fallback engages early — probes whether the model runs on past its text); (b) text much longer than the cap allows (probes truncation, §4). |
| 4 | `max_new_tokens` cap | §2 | Items at **~160 s** (straddling the 2048 eval cap) and **~650 s** (straddling the 8192 runtime cap). Both must be checked for the §4 silent-truncation path, which is the failure this corpus most needs to catch. |

Boundary 1 items are the only ones where "our output ≠ official output" is *expected*; that must be
stated in the corpus manifest so a future agent does not read it as a parity regression.

---

## 7. Dispositions and follow-ups

| # | Item | Disposition |
|---|---|---|
| 1 | Official long-form chunking / sentence splitting / carryover | **[RESOLVED — NONE EXISTS]**; any scheme we add is our own divergence and needs a DISC + drift-gate A/B |
| 2 | cc_1's three-way `max_new_tokens` conflict | **[RESOLVED]** runtime default **8192**; 2048 is an eval-time override; 4096 unreachable; the wrapper's 2048 hard default is dead code |
| 3 | Stop rule | **[VERIFIED]** group-0-only, id 2150, **EOS frame excluded**, residual loop never early-stops |
| 4 | Max-duration behavior | **[VERIFIED]** silent truncation — must become a reported outcome in `ftts` |
| 5 | Talker context ceiling vs cap | **[VERIFIED]** 32768 is not binding; `max_new_tokens` always is |
| 6 | Microdecoder-skip on EOS frame | **[EXACT OPTIMIZATION]** provably token-identical; build it into the `ResidualCodeDecoder` |
| 7 | `generate_config.json` docstring vs `generation_config.json` loaded | **[TRAP RECORDED]** assert the filename in the oracle harness |

### Consumers unblocked

`frankentts-t-executable-spec-1ch` (fold §3–§4 into the executable spec),
`frankentts-v-reliability-d65` (§4 truncation-reporting requirement, §5.3 admission rule),
`frankentts-k-kv-layout-j78` (§5.1–§5.2 KV sizing),
`frankentts-t-golden-corpus-881` (§6 boundary items).
