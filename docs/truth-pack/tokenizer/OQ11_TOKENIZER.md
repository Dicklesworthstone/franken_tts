# OQ-11 — Tokenizer identity + upstream normalization

Resolved 2026-08-06 against the pinned truth pack (HF `5d839924`, GH `022e286b`) using the pinned
oracle (`transformers==4.57.3`, `tokenizers==0.22.2`). Reference ids:
[`tokenizer_conformance.json`](tokenizer_conformance.json), 92 cases, regenerated only by
`scripts/gen_tokenizer_conformance.py`.

---

## 1. Identity

| | |
|---|---|
| Class | `Qwen2Tokenizer` (slow) / `Qwen2TokenizerFast` — `tokenizer_config.json`, and `Qwen3TTSProcessor.tokenizer_class` accepts both |
| Algorithm | GPT-2-lineage **byte-level BPE** |
| Files | `vocab.json` (2.6 MB) + `merges.txt` (1.6 MB) — **no `tokenizer.json`** |
| Base vocab | **151,643** |
| Added special tokens | **33** (ids 151643–151675) ⇒ `len(tokenizer)` = **151,676** |
| Model text embedding | **151,936** rows — *padded*, larger than the tokenizer |

**Three different "vocab sizes" that must not be conflated.** The bead text expected "vocab
151,936"; that is the *embedding* row count (`config.json talker_config.text_vocab_size`), not the
tokenizer's. The tokenizer tops out at 151,676. Rows 151,676–151,935 are unreachable padding.

**There is no `tokenizer.json`**, so there is no pre-serialized fast-tokenizer model to load or diff
against — HF builds the fast tokenizer by converting `vocab.json` + `merges.txt` at load time. We
implement byte-level BPE ourselves. Measured: **the converted fast tokenizer and the slow tokenizer
agree on all 92 cases (0 divergences)**, so one implementation suffices; we do not have to reproduce
a slow/fast split.

## 2. Normalization — the answer to the OQ's central question

**The official stack applies exactly one normalization: Unicode NFC, inside the tokenizer.**

- **The processor does nothing.** `Qwen3TTSProcessor.__call__` (`processing_qwen3_tts.py`) forwards
  text straight to the tokenizer; its only `text_kwargs` are `padding=False`,
  `padding_side="left"`. No lowercasing, no punctuation rewriting, no number expansion, no
  whitespace collapsing.
- **The tokenizer applies NFC.** Slow: `transformers/models/qwen2/tokenization_qwen2.py:338`,
  `text = unicodedata.normalize("NFC", text)`. Fast: `backend_tokenizer.normalizer == NFC()`.
- **It is NFC, not NFKC** — proven, not assumed. `①`≠`1`, `Ａ`≠`A`, `ﬁ`≠`fi`, `㈱`≠`(株)` all remain
  distinct id sequences. A port that reaches for NFKC (an easy mistake — many toolchains default
  there) silently diverges on fullwidth CJK punctuation, which is common in Japanese and Chinese
  input.

Six invariants are asserted by the generator and must hold (it exits non-zero otherwise):

| Invariant | Meaning |
|---|---|
| `combining-nfc` == `combining-nfd` | NFC applied (Latin + combining acute) |
| `hangul-nfc` == `hangul-nfd` | NFC applied to Hangul jamo — **Korean is a supported language** |
| `vietnamese-nfc` == `vietnamese-nfd` | NFC applied to stacked diacritics |
| `nfkc-circled-digit` != `nfkc-plain-digit` | NFKC **not** applied |
| `nfkc-fullwidth-A` != `nfkc-halfwidth-A` | NFKC **not** applied |
| `nfkc-ligature-fi` != `nfkc-ligature-plain` | NFKC **not** applied |

The NFD entries are built by an `NFD()` helper that asserts decomposition actually happened. A
decomposed case typed as a composed literal would compare equal to its twin and the invariant would
pass vacuously — that mistake was made and caught during this bead; the helper prevents it
recurring.

**Consequence for the port:** we need a real Unicode NFC implementation (canonical decomposition +
canonical ordering + canonical composition), not a lookup shortcut. That is a genuine dependency
decision for `frankentts-p1-tokenizer-2uf`, and NFC tables are versioned by Unicode release — the
Unicode version our NFC targets must be pinned and recorded, because Python's `unicodedata` follows
the Unicode version bundled with the interpreter.

**Consequence for round-trip:** `decode(encode(x)) == x` is **false by design** for non-NFC input
(3 of 92 cases). Round-trip identity holds only for NFC-normalized text. Any test asserting global
round-trip identity is wrong.

## 3. The `fix_mistral_regex` finding — the load-bearing one

**Every official entrypoint tokenizes with a Mistral pre-tokenizer regex, not Qwen's own.**

`qwen_tts/inference/qwen3_tts_model.py:118` loads the processor as:

```python
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True,)
```

All three `examples/*.py` and `qwen_tts/cli/demo.py:608` reach the tokenizer through this one
`Qwen3TTSModel.from_pretrained`, so **there is a single consistent official path** and it always
sets the flag.

Why the flag even engages on a Qwen model — the mechanism, from
`transformers/tokenization_utils_base.py:2451-2499`:

1. The loader reads `config.json` and takes `transformers_version`.
2. It early-returns (no Mistral handling) only if that version is `<= 4.57.2` *and* the load is
   local *and* `model_type` is not a Mistral variant — **or** if the version is `>= 5.0.0`.
3. This checkpoint's `config.json` says exactly **`4.57.3`**, which satisfies neither branch, so it
   falls through and sets `mistral_config_detected = True`.
4. With the flag passed, the loader **replaces the fast tokenizer's pre-tokenizer regex** with
   Mistral's.

So a Qwen checkpoint gets Mistral's pre-tokenization because its recorded `transformers_version`
lands in a two-patch-version gap in an unrelated vendor fix. Without the flag, transformers instead
prints a warning telling you to set it.

The two regexes:

```text
native (Qwen)   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
mistral ("fix") [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

Differences: the Mistral form **drops the contraction alternation**, **splits on case boundaries**,
and **adds `/` to the punctuation-run class**.

### Measured impact — bounded and characterizable

6 of 92 cases diverge, in exactly two classes:

| Class | Example | native ids → pieces | official (Mistral) pieces |
|---|---|---|---|
| **Case-boundary splitting** | `getUserByIdOrThrow` | `get·User`,`ById`,`Or`,`Throw` (4) | `get`,`User`,`By`,`Id`,`Or`,`Throw` (6) |
| | `XMLHttpRequest` | `XMLHttpRequest` (1) | `XML`,`Http`,`Request` (3) |
| | `iPhone` | `ĠiPhone` (1) | `Ġi`,`Phone` (2) |
| **Complex-script segmentation** | Thai `สวัสดีชาวโลก` | 6 tokens, different boundaries | 6 tokens, different boundaries |

**What does NOT diverge — the reassuring part:** ordinary prose in all ten supported languages,
contractions (`don't`, `it's`, `I'll`, `we've`, `they're` — the Mistral form's optional-non-letter
prefix still captures `'t`/`'s`), numbers, currency, dates, URLs, email, punctuation, whitespace,
CJK, emoji, and all NFC/NFKC probes. The divergence is confined to **mixed-case identifiers and
scripts without spaces**.

### The unresolved decision this creates

The model was near-certainly **trained** with Qwen's native regex; the official **inference** stack
uses Mistral's. So "match upstream" and "match training" are not the same target here:

- **Conformance default = the official stack** (`fix_mistral_regex=True`). Contract A is parity with
  the pinned reference implementation, and that is what it does. `cases[].ids` is that.
- `cases[].ids_native_qwen_regex` records the alternative for every case, so the experiment is
  cheap to run later.

**This is not settled by reading — it needs audio.** Which regex produces better speech on
camelCase-bearing text is a listening question, and it should be answered once the engine can
synthesize. Until then we follow upstream, because a port that "fixes" upstream silently is a port
whose outputs cannot be compared to it.

**A `DISC-NNN` entry is owed but deliberately not written yet.** `docs/DISCREPANCIES.md` requires
`our_behavior`, a measured impact, and an `FTTS_*` kill switch — none exist before the tokenizer is
implemented. The obligation is recorded on `frankentts-p1-tokenizer-2uf`: implement the official
regex as default, expose `FTTS_TOKENIZER_REGEX=official|native`, and file the DISC entry with a
measured impact at that point. Writing the record now, with an unmeasured impact and a kill switch
that does not exist, would be a counterfeit ledger entry.

## 4. What "verbatim" mode means

Plan §8.7 makes `verbatim` the conformance path and the default. Now defined concretely:

> **verbatim** = apply Unicode NFC, then byte-level BPE with the official (Mistral) pre-tokenizer
> regex, and do nothing else. No case folding, no number expansion, no abbreviation expansion, no
> punctuation rewriting, no whitespace collapsing, no NFKC.

`conservative` and `locale-aware` are **our** additions layered strictly on top, each emitting a
normalization trace. The model card advertises robustness to noisy text, and the upstream stack does
no rewriting whatsoever — so aggressive silent rewriting on our side is a semantic divergence, as
the bead anticipated.

## 5. Other settings read from `tokenizer_config.json`

`add_bos_token: false`, `add_prefix_space: false`, `clean_up_tokenization_spaces: false`,
`split_special_tokens: false`, `errors: "replace"`, `model_max_length: 131072`,
`bos_token: null`, `eos_token: <|im_end|>`, `pad_token: <|endoftext|>`, `unk_token: null`.

**No BOS/EOS is added by the tokenizer** — prompt assembly is entirely the caller's job (OQ-10).
`errors: "replace"` governs byte→text decoding of invalid UTF-8.

TTS-relevant specials, several of which `config.json` never names:

| id | token | note |
|---|---|---|
| 151643 | `<\|endoftext\|>` | pad_token |
| 151644 / 151645 | `<\|im_start\|>` / `<\|im_end\|>` | eos_token is `<\|im_end\|>` |
| 151669 / 151670 | `<\|audio_start\|>` / `<\|audio_end\|>` | |
| 151671 | `<tts_pad>` | `config.json tts_pad_token_id` |
| 151672 | `<tts_text_bos>` | `config.json tts_bos_token_id` |
| 151673 | `<tts_text_eod>` | `config.json tts_eos_token_id` — note *eod*, not *eos* |
| 151674 | `<tts_text_bos_single>` | **second BOS variant, unnamed in `config.json`** → OQ-10 |
| 151675 | `<\|audio_pad\|>` | |

Special tokens appearing as literal text are matched and emitted as single ids (corpus cases
`special-literal-*`); `<|not_a_real_token|>` is not, and tokenizes as ordinary text.

## 6. Reproducing

```bash
uv venv /tmp/tokvenv
VIRTUAL_ENV=/tmp/tokvenv uv pip install transformers==4.57.3 tokenizers regex
/tmp/tokvenv/bin/python scripts/gen_tokenizer_conformance.py
```

torch is **not** required. The generator refuses to run on any transformers other than the pinned
4.57.3, re-hashes the three pinned tokenizer inputs into its output, and exits non-zero if a
normalization invariant breaks.
