# Canonical Conformance Decoder & Production Sampler Contract (OQ-12)

Resolves the **definition** half of **OQ-12** (`frankentts-oq12-canonical-decoder-yhg`): what the
canonical conformance decoder *is*, what the production sampler contract *is*, and how the L5 fork
gets decided. The one sub-question that cannot be answered without a running oracle — *is greedy
audio quality-viable?* — is split into a carrier bead (§5), not guessed.

Citation keys (paths relative to `docs/truth-pack/snapshots/`):

| Key | File |
|---|---|
| `MODEL` | `gh/qwen_tts/core/models/modeling_qwen3_tts.py` |
| `CFG` | `hf/config.json` |
| `GENCFG` | `hf/generation_config.json` |
| `DECODE` | `docs/QWEN3_TTS_DECODE_AND_ADMISSION.md` (OQ-6) |

---

## 1. [FINDING] The two levels have different processor stacks

Both levels delegate to HuggingFace `generate` with plain kwargs — there are **no custom logits
processors anywhere in the Qwen stack**. The sampler's semantics are therefore the pinned
`transformers` implementation's, not Qwen's, which makes the transformers pin part of the sampler
contract (§4.4).

What each level actually passes:

| | **Talker** (`MODEL:2044-2066`) | **Microdecoder** (`MODEL:1671-1680`) |
|---|---|---|
| `max_new_tokens` | 8192 by default (`DECODE` §2) | **15** (`num_code_groups - 1`) |
| `min_new_tokens` | **2** | — |
| `do_sample` | `do_sample` | `subtalker_dosample` |
| `temperature` | 0.9 | 0.9 |
| `top_k` | 50 | 50 |
| `top_p` | 1.0 (inert) | 1.0 (inert) |
| `repetition_penalty` | **1.05** | **none** |
| `suppress_tokens` | **ids 2048–3071 except 2150** | **none** |
| `eos_token_id` | 2150 | **none** — fixed 15 steps, never early-stops |

**This asymmetry is the mechanical reason the plan says one ladder cannot serve two masters.** A
single "the sampler" abstraction applied to both levels would silently add a repetition penalty and a
suppression mask to the residual heads, which the reference does not do.

### 1.1 `suppress_tokens` proves the ID map

`MODEL:2059-2063`:

```python
"suppress_tokens": [
    i for i in range(vocab_size - 1024, vocab_size)      # vocab_size = 3072 -> ids 2048..3071
    if i not in (codec_eos_token_id,)                    # keep 2150
],
```

The talker's entire control region is masked to `-inf` at every step, **except EOS**. So the talker's
effective support is exactly

```
{0, …, 2047}  ∪  {2150}
```

— every valid codec code, plus the stop token. `codec_pad_id` 2148, `codec_bos_id` 2149 and the four
think ids 2154–2157 exist as *input* embeddings for prompt construction but are **unsamplable**.

This independently confirms the OQ-7 finding that the talker→codec token map is the identity with no
offset: the sampler cannot emit anything the codec cannot consume. It also means a port that forgets
the suppression mask will *occasionally* emit a control id into group 0 and hand the codec an
out-of-range code — a rare, seed-dependent failure that is exactly the kind of bug the conformance
ladder exists to catch early.

### 1.2 `min_new_tokens: 2`

The talker must emit at least two frames before EOS is reachable; transformers enforces this by
setting the EOS logit to `-inf` for the first two steps. A port without it can emit a zero- or
one-frame utterance. This complements the `DECODE` §3 stop rule and belongs in the same test.

---

## 2. DECISION (a): the canonical conformance decoder

**Canonical decode = greedy (argmax) at both levels, with the reference's non-warper logits
processors still applied, and no RNG anywhere.**

Precisely, per step:

| Level | Canonical rule |
|---|---|
| Talker | `argmax` over logits after **repetition penalty (1.05)**, **`min_new_tokens=2`** EOS masking, and the **`suppress_tokens` mask**. Temperature / top-k / top-p are **not** applied. |
| Microdecoder | `argmax` over the raw per-depth head logits. No penalty, no mask, no warpers — the reference passes none. Fixed 15 steps. |

Rationale, and why this is not merely "turn sampling off":

- **Warpers are excluded because they are sampling-only.** Temperature is a monotone positive scaling
  and top-k/top-p are truncations, so none of them can change an argmax — including them would be
  harmless but would misrepresent the contract. Excluding them keeps canonical decode a pure function
  of the logits.
- **Non-warper processors are included because they change which token wins.** Repetition penalty,
  the EOS mask, and the suppression mask all reorder or eliminate candidates. Dropping them would
  make canonical decode diverge from the reference's own greedy behaviour, and every L3/L4 rung would
  be measuring the wrong object.
- **Determinism scope**: canonical decode has no RNG, so its determinism claim is over build + ISA +
  artifact only — no seed, no sampler version. This is what makes it usable as the kernel-development
  ladder (Contract A).

Consequence for the ladder: **L4 ("full codec-token stream exact under canonical greedy") is
well-defined right now** and does not depend on the §5 open question. Kernel work can proceed against
it.

---

## 3. DECISION (b): the L5 fork — decision procedure, not a guess

The plan pre-commits both branches; OQ-12's job is to say exactly what selects one. It cannot be
selected yet, so here is the procedure rather than an invented answer:

| Branch | Taken when | L5 then means |
|---|---|---|
| **L5-greedy** | canonical greedy audio is quality-viable on the frozen corpus | perceptual goldens + metrics computed **over greedy audio** |
| **L5-fixed** | it is not | exactness over **codec-fixed** audio (decode of frozen token streams); the perceptual duty transfers **wholly** to Contract B |

**Selection criterion** (predeclared so it cannot be rationalized after the fact): greedy audio is
"quality-viable" iff, on the frozen conformance corpus, it is free of the failure modes that would
make perceptual goldens meaningless — no degenerate repetition loops, no premature EOS, no
long-silence collapse — judged under the §9.4 listening protocol against sampled audio from the same
token prefix. It does **not** need to be *as good as* sampled audio; it needs to be *stable enough to
be a golden*.

**Agents never invent a tolerance to bridge the branches.** If the evidence is equivocal, the answer
is L5-fixed, because it is the branch that makes no perceptual claim.

Note the two branches differ in *what L5 measures*, not in whether the kernel work is correct — L0–L4
are unaffected either way, which is why this fork does not block Phase 1.

---

## 4. DECISION (c): the production sampler contract

### 4.1 Order of operations

The reference applies transformers' standard `LogitsProcessorList` then its warpers. Our port must
reproduce this order exactly; the order is observable because each stage is non-commutative with the
next.

**Talker, per step:**

```
raw logits
  → repetition penalty 1.05      (over previously generated codes)
  → min_new_tokens EOS mask      (first 2 steps: logit[2150] = -inf)
  → suppress_tokens mask         (ids 2048..3071 except 2150 = -inf)
  → temperature 0.9              (divide)
  → top-k 50                     (keep 50 largest, rest -inf)
  → top-p 1.0                    (inert at 1.0, but implement it)
  → softmax → multinomial sample
```

**Microdecoder, per depth:**

```
raw logits → temperature 0.9 → top-k 50 → top-p 1.0 → softmax → multinomial sample
```

Two traps: applying temperature **before** the repetition penalty changes the penalty's effect
(penalty is applied to raw logits, not scaled ones), and applying top-k **after** top-p gives a
different candidate set than the reference's top-k-then-top-p.

### 4.2 Repetition penalty semantics

transformers' `RepetitionPenaltyLogitsProcessor` is sign-dependent, and getting this wrong is a
classic port bug:

```
score = logit[t] ;  logit[t] = score / penalty   if score > 0
                    logit[t] = score * penalty   if score <= 0
```

Applied to tokens already present in the generated sequence. For the talker that sequence is the
**group-0 code history**, not the full 16-code frame — only group 0 is the talker's own output.
Penalty 1.05 > 1 therefore *reduces* the score of a previously emitted code regardless of sign.

### 4.3 Our RNG is a documented divergence

We replicate the **distribution contract**, never torch's RNG stream. Our sampler draws from the same
post-warper categorical distribution using our own seeded PRNG, so:

- identical seeds do **not** reproduce torch's token sequence, and no claim will say they do;
- our own determinism claim is scoped as **(build, ISA, sampler version, seed, artifact)**;
- token-level agreement with the reference under sampling is a **diagnostic**, never a gate. The gate
  under sampling is distributional (Contract B: logit KL/JS, top-k overlap), per plan §9.2.

This needs a `DISC` entry once the sampler exists and the divergence can be *measured*. Filing one now
with an unmeasured impact would be exactly the counterfeit evidence `DISCREPANCIES.md` forbids at
seed time, so the entry is owed at Phase-1 sampler landing, not here.

### 4.4 The transformers pin is part of this contract

Because no custom processors exist, every semantic above — processor ordering, the penalty formula,
top-k tie handling, whether top-p keeps the first token above threshold — is defined by the pinned
`transformers` version, not by Qwen's repository. `CFG` declares 4.57.3. Any change to that pin can
silently change the sampler contract and must re-run the L3 rung. This is a dependency of
`frankentts-oq15-oracle-pins-wjc`.

---

## 5. What remains, and where it lives

The only unresolved sub-question is the **branch selection** in §3: *is canonical greedy audio
quality-viable?* It requires generating audio from the pinned reference and listening to it, so it is
gated on `frankentts-t-oracle-fixtures-6w9` (open) and the listening protocol.

That is carried by **`frankentts-oq12b`** — [OQ-12b] Greedy-audio viability verdict → L5 branch
selection — which owns the run-and-listen step and the resulting branch commitment. This bead closes
on the definitions above, which are what its dependents actually need:

- `frankentts-p1-sampler-f1e` needs §2 and §4 (both complete) to build the two-level sampler.
- `frankentts-v-prod-harness-t96` needs §4.3's claim scoping (complete) plus the §3 branch, which
  arrives with OQ-12b.

**Auto-reopen condition:** if OQ-12b's listening evidence contradicts §2 — i.e. canonical greedy
turns out not to be a usable ladder at all, rather than merely not a golden — this bead reopens,
because §2 would then be the wrong definition rather than merely a definition with an open consumer.
