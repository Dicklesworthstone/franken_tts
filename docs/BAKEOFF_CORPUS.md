# The bakeoff corpus

The evaluation corpus for the three-gate bakeoff (plan §11), and the compiler that turns it into
blinded listening panels.

This is **not** the conformance golden corpus. That one exists to prove the implementation is
exact; this one exists to decide which model is better, and it is judged by ears under
[`docs/CONFORMANCE_AND_LISTENING.md`](CONFORMANCE_AND_LISTENING.md) §4.

Bead: `frankentts-bake-corpus-48h`.

---

## Shape

Three tables, deliberately kept apart:

| File | What it holds | Who can produce it |
|---|---|---|
| `corpus/bakeoff/texts.json` | the target texts that get synthesized | authored in-repo |
| `corpus/bakeoff/speakers.json` | pseudonymous voice identities | requires real speakers |
| `corpus/bakeoff/references.json` | reference recordings + **consent record per row** | requires real speakers |

Splitting them is what makes doctrine #10 enforceable. Audio only ever enters through
`references.json`, so there is exactly one place to check, and `validate` refuses **the whole
corpus** if any single row is short a consent field. There is no partial-credit mode — consent is
not a scoring rubric.

`corpus/bakeoff/design.toml` is the coverage contract. "The corpus is ready" is therefore a
computed fact with a specific failure list, not an opinion.

```bash
python3 scripts/bakeoff/build_corpus.py validate     # ready for Gate A?
python3 scripts/bakeoff/build_corpus.py coverage     # machine-readable coverage
python3 scripts/bakeoff/build_corpus.py dryrun       # corpus -> stimuli -> panel -> verdict
```

| Exit | Meaning |
|---|---|
| 0 | ready for Gate A |
| 1 | **BLOCKERS** — consent or integrity failures; the corpus may not be used at all |
| 2 | shortfalls only — consent-clean, coverage incomplete; not ready |
| 3 | usage / load error |

The gap between 1 and 2 is the point. A shortfall means "collect more". A blocker means "this
audio does not belong here". They are never reported as the same condition, and neither is ever
reported as ready.

---

## Current status

| Half | State |
|---|---|
| **Texts** | **assembled** — 100 texts, 11 categories, 5 language tags (`en`, `zh`, `es`, `ja`, `mixed`), every `design.toml` category minimum met |
| **Audio** | **not collected** — 0 speakers, 0 references |

`validate` reports exit 2 (`NOT READY`) and names the single reason: no reference recordings yet.
That is the honest state, and it is the state the tooling was built to describe precisely rather
than paper over.

**Why the audio half is not done here.** Doctrine #10: this project ships no audio-acquisition
features and does not build on voices whose owners did not provide them. Collecting 50–100
consent-clean speakers is a human process — recruitment, recorded attestation, provenance — and it
is a deliberate owner decision, not something an agent can or should do by writing code. The
compiler, the consent schema, and the refusal behaviour are ready for that audio the moment it
exists.

### Text corpus composition

| Category | Texts | | Canary axis | Texts |
|---|---|---|---|---|
| `prose` | 26 | | `numbers` | 20 |
| `dialogue` | 13 | | `sibilants` | 8 |
| `numbers` | 12 | | `code_switching` | 8 |
| `code_switching` | 8 | | `breaths` | 6 |
| `currency` | 7 | | `long_form` | 5 |
| `names` | 7 | | `noisy_reference` | *supplied by the reference recording* |
| `acronyms` | 7 | | | |
| `urls` | 7 | | | |
| `long_sentence` | 5 | | | |
| `long_form` | 5 | | | |
| `quotations` | 3 | | | |

`long_form` is separated from `long_sentence` because they are different quality regimes — the
12 Hz checkpoint is known to be weaker on long speech (plan §2.8), and the long-form drift gate
needs multi-paragraph material, not merely one long sentence.

`noisy_reference` is a property of the **reference recording**, not the target text, so
`design.toml` marks it `reference_supplied` and the validator does not demand text tags for it.

---

## Coverage contract

From `design.toml`. Scale comes from plan §11; per-cell minima come from the power analysis in
`CONFORMANCE_AND_LISTENING.md` §4.2 — the panel needs ≥8 speakers and ≥12 texts per family for its
by-speaker and by-text cluster analyses, and ≥8 texts on a canary axis before that axis can be
tail-tested at all.

| Axis | Requirement |
|---|---|
| Speakers | ≥50 (target 100), ≥8 per language |
| Reference lengths | 3 s / 10 s / 30 s buckets, ≥40 speakers per bucket, ≥2 buckets per speaker |
| Acoustic conditions | `clean_studio`, `ordinary_phone`, `reverberant_room`, `noisy` — ≥12 speakers each |
| Delivery | `neutral`, `emotional` — ≥12 speakers each |
| Languages | `en`, `zh`, `es`, `ja`; ≥6 cross-language (reference, target) pairs |
| Texts | ≥90 total, per-category minima, ≥4 per canary axis |

Condition and delivery vocabularies are **closed sets**. An unrecognised value is a blocker, not a
new category — otherwise free text leaks into a coverage axis and the counts stop meaning anything.

### Consent record

Every reference row carries all of:

`consent_statement`, `consent_obtained_utc`, `consent_scope`, `speaker_pseudonym`, `provenance`,
`sha256`.

`consent_scope` must be one of `explicit_recorded_for_this_project`,
`licensed_dataset_with_speaker_consent`, `public_domain_with_documented_consent`. The scopes
`scraped`, `web_harvested`, `assumed`, `implied`, `unknown` are named explicitly as **forbidden**
so the rejection message can say why rather than just "invalid". An unrecognised scope is treated
as no scope at all.

---

## One panel, one reference length

The listening harness identifies a cell by `(speaker, text, language, regime)`. Reference length
is not in that key, so a panel mixing a speaker's 3 s and 30 s references would put two different
stimuli in one cell and **confound the system contrast with a reference-length effect**.

`emit_stimulus_manifest` therefore takes a `duration_bucket` and raises a specific error if two
references would claim the same cell. Reference length is a real experimental axis — it is varied
*across* panels, one bucket each, never within one. Gate A runs three panels.

Only cells rendered for the reference **and** both systems are emitted. A partially-rendered cell
is dropped and counted in `_provenance.incomplete_cells_dropped`, because silently dropping it
would turn a missing render into invisible coverage loss.

---

## Judging protocol

The bakeoff does not define its own statistics. It uses the `bakeoff_gate_a` instance in
`scripts/listening/margins.toml`, so effect sizes, equivalence margins, the power analysis, the
named owner, and the tail gate are the ones already pre-registered in
`CONFORMANCE_AND_LISTENING.md` §4 — one protocol, several instances.

Gate A reads its result with a deliberate asymmetry, recorded in the margins file: Pocket is
eliminated as primary on `FAIL_DIFFERENT` on the low side of the identity band.
`INSUFFICIENT_POWER` keeps Pocket alive and demands more data — a challenger is never eliminated by
exhaustion.

---

## Dry-run receipt

`build_corpus.py dryrun` drives the full seam — corpus → stimulus manifest → blinded plan →
responses → verdict — against a synthetic speaker/reference set and a synthetic panel.

Run 2026-08-06, seed 20260806, 10 synthetic speakers × 3 reference lengths, the 100 real authored
texts, 10 s bucket: **634 stimulus items, 156 complete cells, 0 dropped**, `pipeline_ok: true`.

Two panels, because a dry run that only shows the happy path proves half of what matters:

| Panel | Recruited | Expected | Observed | `design_valid` |
|---|---|---|---|---|
| `sized` | 32 | a decisive verdict | `PASS` | `pass` |
| `undersized` | 8 | refusal | `INVALID` | `fail` |

The second row is the one worth having. A pipeline that certifies an under-powered panel is worse
than no pipeline; this one refuses, and the dry run fails if it ever stops refusing.

> **The panel was synthetic.** Every verdict carries `is_quality_claim: false` and cannot clear a
> gate without `--allow-synthetic`. This validates the corpus→panel seam and says nothing whatever
> about audio quality.

---

## What remains

1. **Collect consent-clean reference audio** — 50–100 speakers meeting the coverage contract
   above. Owner decision; tracked separately.
2. **Grow the text corpus** toward the full Gate A scale. The current 100 texts clear every
   `design.toml` minimum, so this is headroom rather than a shortfall.
3. **Freeze the listening margins.** They are `PROVISIONAL` until a human calibration panel runs;
   Gate A can proceed on provisional margins (it does not block a release), but the quantization
   gates cannot.
