# Conformance & Listening

Companion document to [`COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md`](../COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md)
(plan §16.2). Scope: Contract A / Contract B details, math modes, and **the listening protocol**.

| Section | Owning bead | Status |
|---|---|---|
| §1 Contract A — `ConformanceExact` | `frankentts-v-ladder-runner-zmk` | not written |
| §2 Contract B — `ProductionQuality` | `frankentts-v-metamorphic-0wq` | not written |
| §3 Math modes & determinism scope | `frankentts-p0-engine-77h` | not written |
| **§4 The listening protocol** | **`frankentts-v-listening-25m`** | **operationalized (this document)** |

Sections 1–3 are deliberately unwritten placeholders, not stubs to be filled with prose — the
beads above own them. Do not merge against §4 to add them; §4 is self-contained.

---

## 4. The listening protocol

Plan §9.4 says what the protocol must be. This section is what it **is**: every number a gate
reads, where that number lives, and what happens when it is not met.

The protocol is not a document a human follows. It is
[`scripts/listening/`](../scripts/listening/): a dependency-free harness that builds blinded
trial plans, screens listeners, runs the equivalence and tail analyses, and emits a
machine-readable verdict that a release gate branches on. This document explains and binds it;
`margins.toml` holds the numbers so the document and the enforcement cannot drift apart.

```
scripts/listening/
  margins.toml     the pre-registered policy: SESOI, margins, design, screening, instances
  equivalence.py   TOST / non-inferiority / ICC / CVaR-with-permutation-null  (stdlib only)
  protocol.py      trial construction, blinding, screening, synthetic panel
  run_panel.py     plan | simulate | analyze | gate | selftest
```

### 4.1 What is being demonstrated

**Equivalence, never failure-to-reject.** Every panel asks "is the candidate within the
declared indifference band of the incumbent?", answered by two one-sided tests (TOST). A
non-significant difference test is *not* evidence of equivalence and is never reported as one.

Every family therefore returns one of **three** states, and the third one is the whole point:

| State | Meaning | Gate effect |
|---|---|---|
| `PASS_EQUIVALENT` | TOST rejects both bounds at α | bit = `pass` |
| `FAIL_DIFFERENT` | the 1−2α interval lies wholly outside the band | bit = `fail` |
| `INSUFFICIENT_POWER` | the interval straddles a bound | bit = `insufficient_power` — **not a pass** |

Collapsing `INSUFFICIENT_POWER` into either neighbour is the counterfeit-green failure mode
this protocol exists to prevent (AGENTS.md Doctrine #0.4). The harness reports achieved power
and the listener count that *would* have been required, so an underpowered panel produces a
concrete remedy rather than an argument.

### 4.2 Smallest effect size of interest, per metric family

The SESOI **is** the equivalence margin: a difference smaller than the margin is declared, in
advance, to be one we do not care about. All values live in
[`scripts/listening/margins.toml`](../scripts/listening/margins.toml); the table below is the
rationale, not the source of truth.

| Family | Unit of measurement | Centre | Margin (SESOI) | Test | Tier |
|---|---|---|---|---|---|
| `identity_abx` | 2AFC selection rate for the candidate, against a natural reference | 0.5 | **±0.075** | equivalence | human |
| `naturalness_mushra` | paired MUSHRA-point difference (candidate − incumbent) | 0.0 | **±5.0** | equivalence | human |
| `wer` | absolute paired WER difference | 0.0 | **+0.005** | non-inferiority | objective |
| `word_errors_structural` | paired repeated/skipped-word rate difference | 0.0 | **+0.003** | non-inferiority | objective |
| `longform_drift` | paired late-minus-early decline difference | 0.0 | **+0.05** | non-inferiority | objective |

Why these values:

- **Identity ±0.075.** A 7.5-point shift in forced-choice identity preference is the smallest
  difference worth a release decision. The observed between-listener SD of that rate at 24
  trials per listener is ≈0.12 (measured by the pilot, §4.9), implying **23 listeners** at 80%
  power — which is where the design's 24 post-screening listeners comes from, not the other way
  round.
- **Naturalness ±5 MUSHRA points.** Below the ~10-point difference usually treated as
  perceptually consequential in codec listening work, and above the noise floor of a screened
  panel. Between-listener SD of the paired difference ≈8 points → ≈22 listeners.
- **WER +0.5pp.** Paired per-utterance differences have SD ≈0.02, implying ≈100 utterances; the
  design requires 200, so intelligibility is the cheapest gate to satisfy and the first to run.
- **Long-form +0.05.** The 12 Hz checkpoint is *already* weaker than 25 Hz on long speech (plan
  §2.8). This gate does not re-litigate that; it bounds the **additional** drift our own levers
  introduce, which is why the metric is a difference of declines rather than a level.

Non-inferiority families are one-sided on purpose: a candidate with *lower* WER has not failed
an intelligibility gate.

### 4.3 Hierarchical structure

Trials are clustered by listener, speaker, text and language. Pooling raw trials would treat
744 correlated judgements as 744 independent ones and manufacture power that does not exist.

Instead every family is tested **twice, on cluster-level summaries**, and both must clear:

- **by-listener** — one mean per listener. Guards against a persuasive minority of listeners.
- **by-speaker** — one mean per speaker. Guards against a single easy voice carrying the panel.

Objective families use **by-speaker** and **by-text**. `FAIL_DIFFERENT` in either analysis fails
the family; `PASS_EQUIVALENT` requires both. ICC(1) and the implied design effect are computed
and reported per analysis as diagnostics — a rising ICC means the panel is becoming less
informative per trial and the design needs more listeners, not more trials.

This is a by-subjects/by-items analysis rather than a mixed model. It is defensible, it needs no
statistical dependency in CI, and it is strictly conservative relative to a model that borrows
strength across clusters.

### 4.4 Tail reporting on the canary axes (AF-2)

The canary axes are `noisy_reference`, `sibilants`, `breaths`, `code_switching`, `numbers`,
`long_form`. Every stimulus and every objective item carries its axes in the manifest; an
unknown axis name is a hard error, so an axis cannot be silently dropped by a typo.

**Subgroups can only fail a gate, never pass one.** Per-axis results are multiplicity-exposed
and are not used to claim equivalence; they are used to block. An axis that is absent or
under-sampled reports `INSUFFICIENT_DATA`, which is not a pass.

The tail statistic is **CVaR over the worst α = 10% of units**, compared against a **sign-flip
permutation null** (999 permutations) rather than an absolute floor.

> **Why not an absolute floor.** The obvious formulation — "the worst 10% of cells must still
> poll ≥ 0.40" — is unusable, and it is worth recording why so nobody re-proposes it. At
> realistic cell sizes (≈6 trials per speaker×text cell) the per-cell rate has SD ≈0.20, so the
> worst decile of a *perfectly equivalent* system lands near 0.14 by sampling noise alone. Such
> a floor fires on noise, and the only way to stop it firing is to widen it until it means
> nothing. Flipping each paired contrast about its centre with probability ½ is the exact
> exchangeability null for paired data; comparing the observed tail against that distribution
> measures **excess** tail risk, which is what AF-2 is actually for. The declared `slack` in
> `margins.toml` is then a real quantity — how far below chance-level tail behaviour we tolerate
> — instead of an arbitrary constant.

The gate is: observed CVaR no worse than the null's 5th (or 95th) percentile, shifted by
`slack`. Units below `min_obs_per_unit` are dropped and counted; dropping more than
`max_dropped_unit_fraction` of them downgrades the result to `INSUFFICIENT_DATA`.

**This machinery bites.** Selftest scenario `canary_axis_only` injects a 0.30 identity penalty
on sibilant items only. Diluted across the corpus the pooled identity test cannot resolve it
(`insufficient_power`), while `tail_cvar_bound` returns `fail` and the verdict is `FAIL`.

### 4.5 The automation boundary

| Tier | What runs | When | What it can decide |
|---|---|---|---|
| **Tier 0 — objective** | `wer`, `word_errors_structural`, `longform_drift`, spectral distances, secondary speaker embeddings from **multiple unrelated encoders** | CI nightly, and on every lever branch | **Screening only.** Can veto. Can never authorize. |
| **Tier 1 — human panel** | `identity_abx`, `naturalness_mushra`, plus all Tier-0 families | quantization gates, bakeoff Gate A/C, enrollment audition, releases | Authorizes a lossy lever |

Two rules make the boundary real rather than aspirational:

1. **A Tier-0 pass never substitutes for a Tier-1 gate on a lossy lever.** The `ci_nightly`
   instance is declared `objective_only = true`, and the harness *refuses* to analyse it if it
   declares a human family — a Tier-0 screen may not borrow a human bit.
2. **A Tier-0 failure blocks the panel from being scheduled.** Human hours are not spent on a
   candidate that already fails cheap proxies.

Speaker-embedding cosine is a Tier-0 diagnostic and is **secondary by construction** (AGENTS.md
doctrine 8). Identity claims are settled by `identity_abx`, not by an embedding distance.

### 4.6 Protocol instances

One protocol, several instances, each naming its consumers. Declared in `[instances.*]`:

| Instance | Purpose | Blocks release | Consumers |
|---|---|---|---|
| `ci_nightly` | Tier-0 objective screen | no | — |
| `quant_gate` | Q8 codec, Q8 text embedding, Q4 MTP waterfall, AF-1 allocation | **yes** | `frankentts-p2-codec-q8-3vw`, `frankentts-k-q4-mtp-lnp`, `frankentts-p2-int8-talker-micro-upg` |
| `bakeoff_gate_a` | upstream quality, official implementations only | no | `frankentts-bake-corpus-48h` |
| `enrollment_audition` | AF-4 segment discovery, multi-reference | no | `frankentts-p4-segment-discovery-qjc`, `frankentts-p4-enrollment-en6` |
| `surgery_canary` | Phase-5 adaptive depth / early exit | **yes** | `frankentts-p5-surgery-00e` |
| `release` | ship gate, all six canary axes required | **yes** | `frankentts-v-af2-tailrisk-0oy` |

Two instances read their result with a deliberate asymmetry, recorded in the file:

- **`bakeoff_gate_a`** is a quality *floor*, not an equivalence claim. Pocket is eliminated as
  primary on `FAIL_DIFFERENT` on the low side. `INSUFFICIENT_POWER` keeps Pocket alive and
  demands more data — it never eliminates a challenger by exhaustion.
- **`enrollment_audition`** must demonstrate *superiority* (selection beats whole-file), so the
  equivalence result serves only as a floor; the one-sided superiority statistic is reported by
  the enrollment bead.

### 4.7 Panel design, recruitment and screening

| Parameter | Value | Source |
|---|---|---|
| listeners surviving screening | **24** | derived from the ±0.075 / ±5.0 margins at 80% power |
| listeners recruited | **32** | 24 + allowance for ≈25% screening loss |
| trials per listener per family | 24 | |
| catch trials per listener | 6 | |
| minimum speakers / texts / languages | 8 / 12 / 2 | per family, checked independently |
| minimum objective utterances | 200 | |

**Recruitment.** Panels are recruited per instance from listeners who have not heard the same
corpus under a prior instance within the same policy version. No panel member may be an author
of the lever under test. Native-language listeners are required for every language in the
corpus; cross-language items are judged by listeners native in at least one side.

**Blinding.** Stimulus files are presented under opaque slot labels (`S1`…`Sk`) in an order
randomized per listener from a recorded seed, with slot assignment counterbalanced. Three
artifacts are produced: `plan.json` (metadata), `trials.blind.json` (the panel-facing export,
which contains no system identities) and `trials.key.json` (**the sealed key**). The Listening
Protocol Owner holds the key until analysis is run.

**Post-screening rules — predeclared, applied by the harness, never adjusted after seeing data:**

| Rule | Threshold |
|---|---|
| catch-trial accuracy (correct speaker vs foil speaker) | ≥ 0.85 |
| hidden reference rated < 90 | on ≤ 15% of MUSHRA trials |
| low anchor rated > 60 | on ≤ 15% of MUSHRA trials |

A listener failing any rule is removed entirely, with the reason recorded in the verdict. If
screening drops the panel below 24 the verdict is `INVALID` — **re-recruit, do not lower the
bar**.

### 4.8 The named owner

| | |
|---|---|
| **Role** | Listening Protocol Owner (LPO) |
| **Holder** | `jemanuel` (declared in `margins.toml [owner]`) |

The LPO approves the power analysis before any panel is recruited, signs off every margin
amendment and bumps `policy_version`, holds the sealed key until analysis, and — the load-bearing
one — **declares a verdict `INVALID` rather than re-running a panel that missed its power
target**. The harness refuses to load a margins file whose `[owner].name` is empty; the role is
bound to a person, never to "the team".

### 4.9 Calibration status

> **The margins are `PROVISIONAL`.** They are derived from the pilot's measured dispersion and
> from prior art, not yet from a human calibration panel on this model's audio.

| | |
|---|---|
| `policy_version` | `2026-08-06.1` |
| `calibration_status` | `PROVISIONAL` |
| Freeze gate | Phase 2 entry (first shipping quantization gate) |

Provisional margins **may** gate exploratory levers. They **may not** gate a release: `run_panel.py
gate` returns `INVALID` (exit 3) for any release-blocking instance while the status is
`PROVISIONAL`, unless `--allow-provisional` is passed explicitly. Freezing means running a human
calibration panel, replacing the assumed between-listener SDs with measured ones, re-deriving
the required listener counts, and bumping `policy_version` — under LPO sign-off.

### 4.10 The release binding

The verdict is a JSON file with **named bits**. AF-2's tail bound is one of them, not advisory
commentary:

```json
"bits": {
  "design_valid":                    "pass",
  "identity_equivalence":            "pass",
  "naturalness_equivalence":         "pass",
  "intelligibility_noninferiority":  "pass",
  "longform_drift":                  "pass",
  "tail_cvar_bound":                 "pass"
}
```

`run_panel.py gate --verdict verdict.json` is the enforcement point. Stable exit codes:

| Code | Meaning |
|---|---|
| 0 | `PASS` — every required bit is `pass` |
| 1 | `FAIL` — a real difference was detected |
| 2 | `INSUFFICIENT_POWER` — the panel could not decide. **Not a pass.** |
| 3 | `INVALID` — design violated, or a synthetic/provisional verdict at a release gate |
| 4 | usage / input error |

The gate refuses by default and must be told what to forgive:

- a verdict from a **synthetic panel** (`synthetic_panel: true`, `is_quality_claim: false`)
  cannot pass without `--allow-synthetic`;
- a **provisional-margin** verdict on a release-blocking instance cannot pass without
  `--allow-provisional`.

Both flags are visible in the command line that CI runs, so forgiveness is auditable rather than
implicit.

### 4.11 Pilot receipt

`run_panel.py selftest` is the standing proof that the harness still detects degradation — a
harness that only ever says `PASS` is worthless. It builds a demo corpus (8 speakers × 16 texts,
all six canary axes, 32 recruited listeners), drives it with a seeded synthetic panel, and
asserts a predeclared verdict for each ground truth.

Run on 2026-08-06, `policy_version 2026-08-06.1`, seed 20260806 — **exit 0, all seven green**:

| Scenario | Ground truth | Expected | Observed |
|---|---|---|---|
| `equivalent` | no difference | `PASS` | `PASS` |
| `equivalent_release_all_axes` | no difference, all 6 axes required | `PASS` | `PASS` |
| `identity_degraded` | identity −0.15 | `FAIL` (`identity_equivalence`) | `FAIL` |
| `naturalness_degraded` | naturalness −9 MUSHRA | `FAIL` (`naturalness_equivalence`) | `FAIL` |
| `wer_regression` | WER +0.02 | `FAIL` (`intelligibility_noninferiority`) | `FAIL` |
| `underpowered` | no difference, heterogeneous pool | `INSUFFICIENT_POWER` | `INSUFFICIENT_POWER` |
| `canary_axis_only` | −0.30 on sibilants only | `FAIL` (`tail_cvar_bound`) | `FAIL` |

Two measurements from the pilot are load-bearing for §4.2, and both confirm the design:

- Observed between-listener SD of the identity rate: **0.121** (assumed ≈0.12).
  `required_n_for_declared_power` came back as **23** against a design of 24. Achieved power
  0.93.
- The `underpowered` scenario returned mean 0.522 with a 90% interval of [0.418, 0.626] — a
  point estimate sitting almost exactly on the centre, correctly reported as *undecided* rather
  than equivalent, with `required_n` of 178 listeners at that dispersion. That is the number a
  panel would need, offered instead of an argument.

**This is a pipeline validation, not an audio claim.** The panel was synthetic; every verdict it
produced carries `is_quality_claim: false` and cannot clear a gate without `--allow-synthetic`.

### 4.12 Running it

```bash
# 1. build a blinded plan from a stimulus manifest
python3 scripts/listening/run_panel.py plan \
    --manifest corpus/quant_q4.json --instance quant_gate --out panels/q4/

# 2. collect responses (panel UI writes responses.jsonl), or simulate for a dry run
python3 scripts/listening/run_panel.py simulate \
    --plan panels/q4/ --manifest corpus/quant_q4.json --out panels/q4/responses.jsonl

# 3. analyse -> the machine-readable verdict
python3 scripts/listening/run_panel.py analyze \
    --plan panels/q4/ --manifest corpus/quant_q4.json --instance quant_gate \
    --responses panels/q4/responses.jsonl --objective panels/q4/objective.json \
    --out panels/q4/verdict.json

# 4. enforce (the release binding)
python3 scripts/listening/run_panel.py gate --verdict panels/q4/verdict.json

# standing CI proof that the harness still bites
python3 scripts/listening/run_panel.py selftest
```

`selftest` belongs in `scripts/check.sh` alongside the Rust gates: it needs no model, no audio
and no network, and it fails loudly if a future edit stops the harness detecting degradation.

### 4.13 What this protocol deliberately does not do

- It does not replicate the reference implementation's RNG stream. Free-running comparisons are
  distribution-level by design (plan §9.2), which is a standing DISCREPANCIES entry, not a bug.
- It does not treat token-level equality as a quality gate. That is a Contract-A concern.
- It does not average away a failing subgroup. Canary axes block; they do not get outvoted.
- It does not report a number it cannot back. `INSUFFICIENT_POWER` and `INSUFFICIENT_DATA` are
  correct outputs.

### 4.14 Amendment log

Every change to `margins.toml` requires an LPO sign-off recorded here and a `policy_version`
bump. Verdicts produced under a superseded `policy_version` are invalid.

| `policy_version` | Date | Change | Signed off |
|---|---|---|---|
| `2026-08-06.1` | 2026-08-06 | Initial operationalization: SESOI + margins for five families, 24/32 panel design, screening rules, permutation-calibrated tail gate, six protocol instances, release binding. Status `PROVISIONAL`. (`frankentts-v-listening-25m`) | pending LPO |

---

## 5. Reliability gates

Plan §9.6. Correctness work concentrates on numerics; production TTS dies on hostile inputs and
stuck consumers. Every gate here exists because the corresponding failure produces output that a
*program* reads as success — a plausible-length WAV, a completed run, a nonzero byte count. An
agent consuming `ftts` cannot listen to the audio, so each failure has to be made inspectable.

Bead: `frankentts-v-reliability-d65`. Implementation: `ftts-core/src/admission.rs`,
`ftts-core/src/health.rs`, `ftts-core/tests/streaming_failures.rs`.

### 5.1 Resource admission — decide before allocating

The failure prevented is dying halfway through a long generation, after the caller has waited a
minute and after hundreds of megabytes are committed. Peak memory is predictable from the prompt
length and the frame cap, so the request is refused whole or admitted whole:

```text
predicted_max_frames = min(max_new_tokens, 32768 - prompt_tokens)
predicted_peak_bytes = KV_talker(prompt_tokens, predicted_max_frames)
                     + 320 KiB + 2.25 MiB + rings + weights_resident
admit iff predicted_peak_bytes <= budget
```

Only the talker KV grows with duration (112 KiB/token at BF16); the microdecoder KV resets per
frame and the codec KV is a fixed 72-frame window. Three worked points are asserted **exactly**, so
a formula drift fails a test instead of quietly changing capacity: 512+2048 → 280 MiB,
512+8192 → 952 MiB, full 32768 context → 3.50 GiB.

Enforced at the engine seam after text preparation (prompt length is not knowable before
tokenization) and before any stage runs. The proof is by *absence*: a rejected request emits no
stage event, because a stage event would mean work was already committed.

Overflow is its own refusal, never a saturating clamp — a wrapped total is a small plausible number
that would admit a request certain to die mid-generation.

### 5.2 Truncation is an outcome, not a silence

The reference implementation reaches `max_new_tokens` without EOS and returns the cut-off audio
with no exception, no warning and no flag. The caller cannot distinguish "the model finished" from
"the model was cut off mid-word", and an agent cannot hear the difference.

`StopReason::EndOfSpeech` is therefore the **only** clean completion. `FrameCapReached` and
`DurationLimitReached` are `is_truncated()` and each carries a remedy. Reporting a truncated
utterance as a plain success is the counterfeit green Doctrine #0.4 forbids.

### 5.3 Runtime health

Seven detectors, each an allocation-free state machine callable per frame. None reads the clock —
the caller supplies the instant, so tests inject time instead of sleeping.

| Detector | Catches | Invalidates output |
|---|---|---|
| `NumericGuard` | NaN/Inf at a named seam, reported at the **first** offending index | yes |
| `ProgressWatchdog` | a decode loop that stopped advancing | yes |
| `check_stop_consistency` | a stop reason that contradicts the frame counters | yes |
| `RunawayDetector` | a stuck token **and** a short repeating cycle | yes |
| `SilenceDetector` | output that produces bytes but no sound | yes |
| `KernelSelector` | an ISA kernel that failed its selftest | no — still correct, just slower |
| `ThermalReporter` | sustained throughput below the opening window | no — a reporting obligation |

Three design points worth stating, because each is a place the obvious implementation is wrong:

- **The seam policy is explicit.** A NaN check over every activation is a real cost in the loop
  that is the whole project. `All` while developing a kernel, `Sampled` in production (a NaN that
  occurs at all recurs within a few frames), `Off` only for a measured benchmark. `Off` reports
  `is_checking() == false`, so a run under it is reported as **unchecked**, never as clean. A
  sampling interval of zero checks every call rather than disabling the guard — a misconfigured
  sampler must not be the reason nothing was inspected.
- **Cycle detection is separate from repeat detection.** A two-token ping-pong never trips a
  consecutive-repeat counter and is just as dead.
- **Kernel demotion is one-way.** A tier that failed its selftest has produced a wrong answer on
  this machine; re-promoting it because a later check passed would trust the evidence that already
  lied. G1 > G2 — the run finishes slower and correct.

Violations travel to the caller through the same `SynthesisObserver` as every other lifecycle event
(`SynthesisEvent::Health { event: HealthEvent::Violation(..) }`), so a problem is visible while the
run is happening. Each carries a remedy, not a label.

### 5.4 Streaming failure semantics

Defined *and* injected — each test forces the failure rather than checking the happy path still
works. `ftts-core/tests/streaming_failures.rs`, 7 tests:

| Injected | Required behaviour |
|---|---|
| consumer stops reading | producer parks at the bound (asserted: no unbounded buffering), cancellation releases it |
| event consumer blocks | PCM keeps flowing — the two bounded queues cannot deadlock each other |
| audio consumer blocks | events keep flowing, or a caller could never learn *why* audio stopped |
| consumer disappears | structured `StreamDisconnected(StreamKind)`, not a panic or a hang |
| cancellation mid-emission | stops on a packet boundary; every delivered packet contains whole frames |
| sink write error / disk-full | partial output finalised with a header declaring exactly the bytes present |

The disk-full promise is specifically that the partial file is *playable*. A WAV whose header
claims more data than the file holds is corrupt, and every player disagrees about how to fail on
it, so both the RIFF size and the data size are patched to what actually landed.

### 5.5 What is not yet wired

Stated because a reliability section that implies more coverage than exists is itself a
reliability problem:

- The detectors in §5.3 are implemented and tested but have **no model to observe yet** — Phase 0
  has no decode loop. They are called from the seams as those seams land.
- `ftts-cli`'s `say --check` still uses its own scaffold `admission_plan()` rather than
  `ftts_core::admission`. The engine-side admission is enforced; the CLI preflight is not yet the
  same computation.
- Ring-buffer and resident-weight terms in the admission prediction are `0` in Phase 0 — the honest
  value. A placeholder would make the prediction look complete while being wrong. Both are bounded
  and land with their components.
