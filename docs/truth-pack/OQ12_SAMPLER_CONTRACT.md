# OQ-12 — Canonical conformance decoder + production sampler contract

**Bead:** `frankentts-oq12-canonical-decoder-yhg` · **Author:** AzureThrush · **Date:** 2026-08-06

**Partial deliverable. Read §5 before treating this as done.** Sub-question **(c) the production
sampler contract is RESOLVED** and is the part `frankentts-p1-sampler-f1e` needs. Sub-questions
**(a) is greedy audio quality-viable** and **(b) the L5 fork** are **NOT resolved** — they require
generated audio and a listening panel, and neither the model weights nor an oracle exist on this
host.

The useful surprise: **§2 shows the canonical greedy decoder is fully specifiable today**, without
answering (a). Greedy token ids turn out to be invariant to the entire sampling-warper stack, so the
canonical decoder's definition does not depend on the fork at all.

Status: **[VERIFIED-STATIC]** — pinned model source + pinned `transformers` source. Nothing was
generated or listened to.

---

## 1. The reference sampling pipeline, exactly

### 1.1 Effective parameters and where they come from

`_merge_generate_kwargs` (`I:318-352`) resolves each parameter with precedence
**user argument → `generation_config.json` → hard-coded default**:

```python
def pick(name, user_val):
    if user_val is not None:            return user_val
    if name in self.generate_defaults:  return self.generate_defaults[name]   # generation_config.json
    return hard_defaults[name]
```

| Parameter | Effective value | Source |
|---|---|---|
| talker `do_sample` / `temperature` / `top_k` / `top_p` | `True` / **0.9** / **50** / 1.0 | generation_config.json |
| talker `repetition_penalty` | **1.05** | generation_config.json |
| microdecoder `subtalker_*` (`do_sample`/`temperature`/`top_k`/`top_p`) | `True` / **0.9** / **50** / 1.0 | generation_config.json |
| microdecoder `repetition_penalty` | **none — not passed at all** (`M:1671-1680`) | — |
| talker `min_new_tokens` | **2** | hard-coded (`M:2046`) |
| talker `eos_token_id` | **2150** (`codec_eos_token_id`) | config.json |
| talker `suppress_tokens` | **2048..3071 except 2150** | computed (`M:2059-2063`) |
| `max_new_tokens` | **8192** | generation_config.json |
| microdecoder `max_new_tokens` | **15** (`num_code_groups - 1`) | computed (`M:1673`) |

> ⚠ **Three sources disagree on `max_new_tokens`**: `generation_config.json` says **8192**,
> `_merge_generate_kwargs`'s hard default says 2048, and
> `Qwen3TTSForConditionalGeneration.generate`'s signature says 4096 (`M:2031`). The precedence chain
> makes **8192** effective, because `generate_defaults` is loaded from `generation_config.json` and
> the inference wrapper always passes an explicit value downstream. 8192 frames × 80 ms ⇒ a
> **≈ 655 s (10.9 min) hard ceiling** on one utterance. Owed to `frankentts-oq6-context-stop-cc3`
> and to admission control (plan §9.6). Do not hard-code 2048 or 4096.

### 1.2 The `suppress_tokens` set settles the 3072-wide head

```python
"suppress_tokens": [i for i in range(vocab_size - 1024, vocab_size)
                    if i not in (codec_eos_token_id,)]        # M:2059-2063
```

With `talker_config.vocab_size = 3072`: **ids 2048–3071 are suppressed except 2150 (`codec_eos`)**.
Every talker special — `codec_pad_id` 2148, `codec_bos_id` 2149, `codec_think_id` 2154,
`codec_nothink_id` 2155, `codec_think_bos_id` 2156, `codec_think_eos_id` 2157 — lives inside that
band and is **prompt-only, never generable**.

⇒ The talker's generable alphabet is **codes 0–2047 plus EOS 2150**. This is *why* the head is
1024×3072 rather than 1024×2048, closing the "~3 MB primary head oddity" from OQ-2 with a mechanism
rather than a guess. **Our sampler must apply this suppression** or it will emit prompt specials as
audio codes.

### 1.3 Processor order (read from the pinned `transformers` source, not from memory)

Both levels use stock `GenerationMixin.generate`, so the pipeline is HF's. From
`transformers/generation/utils.py::_get_logits_processor` (transformers **4.57.3**, the version
cc_1 pinned as the tokenizer oracle), the append order relevant to us is:

```
RepetitionPenaltyLogitsProcessor        <- processor (always applied)
MinNewTokensLengthLogitsProcessor       <- processor (always applied)
SuppressTokensLogitsProcessor           <- processor (always applied)
TemperatureLogitsWarper                 <- warper (do_sample only)
TopKLogitsWarper                        <- warper (do_sample only)
TopPLogitsWarper                        <- warper (do_sample only)
```

**Processors always run; warpers run only when `do_sample=True`** (`if generation_config.do_sample:`
guards the warper block). This distinction is the whole content of §2.

⚠ `transformers` 4.57.3 is the **tokenizer** oracle pin. The authoritative *runtime* pin is
`frankentts-oq15-oracle-pins-wjc`'s deliverable, and must come from the README per doctrine, never
from config metadata. If OQ-15 pins a different version, **re-read this order** — HF has reordered
this list historically, and the source comment `# TODO (joao): find a strategy to specify the order
of the processors` is not reassuring.

### 1.4 The two pipelines, concretely

```
TALKER, per frame:
  logits (1024x3072)
    -> repetition_penalty(1.05) over previously generated talker ids
    -> min_new_tokens(2): force -inf on EOS while fewer than 2 tokens generated
    -> suppress_tokens: -inf on 2048..3071 except 2150
    -> [sampling only] temperature (divide by 0.9)
    -> [sampling only] top_k(50)
    -> [sampling only] top_p(1.0)  == no-op
    -> softmax -> multinomial

MICRODECODER, per depth (x15):
  logits (1024x2048)
    -> (no repetition penalty, no min-length, no suppression)
    -> [sampling only] temperature (divide by 0.9) -> top_k(50) -> top_p(1.0, no-op)
    -> softmax -> multinomial
```

**The microdecoder has no repetition penalty and no suppression.** Its whole 2048 vocabulary is
generable. Applying the talker's penalty to residual depths would be a silent semantic bug — and it
is an easy one to introduce by sharing one sampler implementation across both levels.

---

## 2. The canonical conformance decoder — decidable NOW, independent of the fork

**Definition (recommended, and it is forced rather than chosen):**

> **Canonical greedy = `do_sample=False` through the reference's own logits-*processor* stack.**
> Talker: repetition_penalty 1.05 → min_new_tokens 2 → suppress_tokens → **argmax**.
> Microdecoder: **argmax** on raw logits (it has no processors).

**Why this is forced, not a preference.** `argmax` is invariant to every warper in the stack:

- **Temperature** `l/0.9` is a strictly monotonic transform ⇒ preserves the argmax.
- **Top-k(50)** sets all but the top 50 to −inf ⇒ the max is in the top 50 ⇒ preserves the argmax.
- **Top-p(1.0)** is a no-op.

So `argmax(warpers(processors(l))) == argmax(processors(l))` — **canonical greedy ids are identical
whether or not the warpers are applied.** The definition therefore has no free parameter, and does
not depend on resolving (a).

**What it is NOT.** Canonical greedy is *not* "raw argmax of the model's logits." Dropping the
processors changes the ids:

- Without **suppress_tokens**, greedy can select ids 2048–3071 — prompt specials emitted as audio
  codes. Catastrophic and easy to miss on short clips.
- Without **repetition_penalty**, the talker's greedy trajectory diverges from the reference's after
  the first repeated code (penalty 1.05 is small but greedy is a hard argmax — a 5 % rescale flips
  near-ties).
- Without **min_new_tokens**, a degenerate prompt can emit EOS immediately.

A port that implements "greedy = argmax" and compares against the reference run with
`do_sample=False` **will diverge**, and the divergence will look like a kernel bug. This is the
single highest-value line in this document for `p1-sampler-f1e` and for the Contract-A L3/L4 rungs.

**Consequence for Contract A (plan §9.1):** L4 ("full codec-token stream exact under canonical
greedy") is well-defined today. The rung can be built and run against the oracle the moment weights
exist; it does **not** wait on the (a)/(b) fork, which governs only what **L5** means.

---

## 3. Our RNG divergence — the DISC obligation, and why no entry is filed yet

The reference draws with `torch.multinomial` over the post-warper softmax. We will not replicate
torch's RNG **stream** (plan §9.2 already states this as a standing divergence); we replicate the
**distribution contract** with our own seeded RNG.

**Design contract for `p1-sampler-f1e`:**
- Same *distribution*: identical processor/warper arithmetic and order, identical softmax, then a
  draw from the same categorical distribution.
- Determinism scope is the full tuple **{engine build + ISA path + sampler version + seed +
  artifact}** (plan §9.3) — never cross-build byte identity.
- Free-running comparisons against the reference are **distribution-level over many seeds**, never
  single-seed A/B (plan §9.2).
- `strict` math mode + fixed seed reproduces our own output exactly.

**No `DISC-NNN` entry is filed, deliberately.** `docs/DISCREPANCIES.md`'s own schema requires
`fixture_sha256`, `artifact_sha256`, `cpu_features`, `kill_switch`, and **`measured_impact`**, and
the ledger states plainly: *"Do not create a `DISC-NNN` record from an inherited result: impacts and
restoration switches must be measured on this model."* I have no measured impact — no weights, no
audio, no fixture. Filing a record with `measured_impact: <unmeasured>` would be exactly the
counterfeit-green the ledger exists to prevent.

**The obligation is therefore carried, not discharged:** the DISC entry is filed by
`frankentts-p1-sampler-f1e` when the sampler lands with a fixture and a measured distributional
impact. This is flagged to the orchestrator in §5 because the bead's exit criteria and the ledger's
schema are in direct conflict, and the ledger should win.

---

## 4. Cross-references produced here

| Finding | Owed to |
|---|---|
| Effective `max_new_tokens` is **8192** (≈655 s ceiling), not 2048/4096 — three sources disagree, precedence resolves it | `oq6-context-stop-cc3`, admission control |
| Talker generable alphabet is codes 0–2047 + EOS 2150; all specials 2148–2157 are prompt-only and suppressed — explains the 1024×3072 head | `oq2-tensor-inventory-ght` |
| Microdecoder has **no** repetition penalty / suppression; talker has both | `p1-sampler-f1e` |
| Canonical greedy must run the processors; argmax is warper-invariant | `p1-sampler-f1e`, `v-prod-harness-t96`, Contract-A L3/L4 |
| HF processor order is version-sensitive; re-read it against OQ-15's runtime pin | `oq15-oracle-pins-wjc` |

## 5. What remains — and the decision owed to the orchestrator

**Blocked on the oracle (no torch, no weights on this host):**
- **(a) Is greedy audio quality-viable?** The bead says "Run it and listen." Requires generated audio
  and the §9.4 listening protocol.
- **(b) The L5 fork.** Strictly downstream of (a). Both branches are already pre-committed in plan
  §9.1, so this is an evidence lookup, not a design decision — but it is an evidence lookup that
  needs evidence.

**Deliberately deferred (conflict between exit criteria):**
- The `DISC-NNN` entry — see §3. The bead asks for it filed; the ledger's schema forbids filing it
  unmeasured. Carried by `p1-sampler-f1e`.

**Recommendation:** split (a)+(b) into an oracle-blocked follow-up exactly as OQ-5 → `m1u`, and let
this bead close on the sampler contract + canonical-decoder definition, which are what the
downstream implementation beads (`p1-sampler-f1e`, `v-prod-harness-t96`, `t-executable-spec-1ch`)
actually consume. Not self-approved — flagged for FuchsiaMouse.
