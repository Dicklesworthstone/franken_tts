# PERF_LEDGER

Gate contract: `consumer=performance claim and release gate`; `gate=only a current-tree, pinned-reference, parity-qualified measurement can support a win`; `defect_class=unreproducible or unfair performance claim`; `deletion_condition=never; superseded rows remain evidence`.

No inherited performance result is a ledger row. Inherited priors belong in `NEGATIVE_EVIDENCE.md` until re-confirmed locally; self-speedups are maintenance, not an admissible external ratio.

## Entry schema

```text
PERF-NNN
claim_id: <claim-id>
evidence_id: <artifacts/perf path or id>
status: WIN | PROVISIONAL_LOCAL_WIN | BASELINE | NEGATIVE | NO_EVIDENCE | VOID
model_source_commit: <pinned commit>
fixture_sha256: <sha256>
artifact_sha256: <sha256>
cpu_features: <dispatched feature string>
command_env: <exact command and environment>
kill_switch: <FTTS_* state>
incumbent: <pinned external reference and fairness controls>
before_after: <interleaved paired result>
cv_percent: <value; must be <= 5 for an admissible row>
equivalence: <parity or quality proof tier>
disposition: KEEP | REVERT | DEFER
tally_w_l_n: <wins/losses/neutrals for this lever>
```

`BASELINE` rows are not claims: they are the reference measurement later levers are judged against ("the honest baseline Phase-3B must beat" — z2w addendum). A baseline whose cv exceeds the 5% gate is recorded as INDICATIVE ONLY and must be re-measured in a quiet window before any lever quotes a ratio against it.

## Local entries

```text
PERF-003
claim_id: startup-hydration-campaign
evidence_id: interleaved 3-round load-stage A/B vs installed v0.1.3 binary, 2026-08-09; commits 9478b3c/c311ab1/c883b6c/6c1b80d/4df93f0
status: KEEP (magnitude PROVISIONAL_LOCAL_WIN — host load average 26-54 throughout; direction consistent 3/3 interleaved rounds; quiet-window rerun owed)
model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc
fixture_sha256: not-applicable (load-stage wall time, no fixture)
artifact_sha256: 597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824
cpu_features: Apple M4 Pro aarch64
command_env: ftts say --seed 11 "Hi." (load-stage NDJSON elapsed), new binary interleaved with brew-installed v0.1.3
kill_switch: FTTS_LOAD_THREADS=1 (serial widen), FTTS_ARTIFACT_Q8=0 (requantize + no elision), FTTS_INT8=0 (f32 reference, elision auto-off)
incumbent: installed v0.1.3 (pre-campaign hydration; its own ~1.1 s generator-build requantize additionally falls OUTSIDE its load stage, so the shown gap understates the total)
before_after: load stage 6.3/6.9/8.0 s (v0.1.3) vs 4.8/5.7/5.3 s (campaign) in interleaved pairs under load
cv_percent: not-gated (host load 26-54 — magnitudes are provisional by rule; the 3/3 direction and the eliminated double-digest-verify (~1.3 GB re-hash, structural) support KEEP)
equivalence: BYTE-IDENTICAL same-seed say across armed+elided, FTTS_ARTIFACT_Q8=0, and FTTS_INT8=0 routes at every step of the campaign
disposition: KEEP; levers: (1) concurrent tensor widening (capped avail/2 max 6 after uncapped workers measured LOSING ~2x to serial under external load — that interleaved loss is recorded here as the reason for the cap), (2) hot-projection f32 elision when the armed route hydrates artifact-natively, (3) codec/tokenizer load overlapped with talker, (4) parallel codec piece hydration, (5) shared digest-verified mapping (the double MappedFttsq::open was hashing 1.3 GB twice)
tally_w_l_n: 1/0/0 (campaign; the uncapped-workers sub-lever is the internal L that produced the cap)
```

```text
PERF-002
claim_id: artifact-native-q8-hydration
evidence_id: examples/artifact_q8_hydration.rs run 2026-08-09 + same-seed say byte-compare; commits 90c75d3/9183802/54eb4b8
status: KEEP
model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc
fixture_sha256: not-applicable (whole-checkpoint hydration comparison, no fixture)
artifact_sha256: 597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824 (qwen3-tts-12hz-0.6b-base.fttsq)
cpu_features: Apple M4 Pro aarch64
command_env: cargo run --release -p ftts-model-qwen --example artifact_q8_hydration -- ~/.cache/franken_tts/model/qwen3-tts-12hz-0.6b-base.fttsq   (fleet-loaded host)
kill_switch: FTTS_ARTIFACT_Q8=0 (restores widen-then-requantize hydration)
incumbent: the tree's own runtime requantize hydration (same process, same run)
before_after: int8-table hydration 1.112 s -> 146 ms (7.6x, same-process A/B); interleaved warm e2e say pairs armed 12.2/13.6 s vs off 14.2/15.0 s
cv_percent: not-gated (same-process isolated stage A/B for the headline number; e2e pairs are corroboration under fleet load, not the claim)
equivalence: BYTE-IDENTICAL — fused Q8 payload bytes asserted equal across all 33 layers, worst scale ulp distance 0, and same-seed say output byte-identical armed vs kill-switched
disposition: KEEP; context: the artifact's Q8 payload IS the canonical quantization (shared-quantizer contract), so this also removes a class of drift, not just time. Remaining startup cost is the 9.5 s f32 widen of hot tensors the f32 fallback structs still demand — that lever needs checkpoint-side storage changes (Option-alized hot f32 buffers when the int8 route is armed).
tally_w_l_n: 1/0/0
```

```text
PERF-001
claim_id: talker-f32-reference-baseline
evidence_id: crates/ftts-conformance/tests/talker_perf_baseline.rs receipt, 2026-08-07
status: BASELINE (INDICATIVE ONLY — cv 6.4% exceeds the 5% gate; quiet-window rerun required before any ratio quotes this row)
model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc
fixture_sha256: 5ec2bc3f3217f9e026198c0694b3993d9911f7e954ea726c69ebd95e7d5ba4dd (fixture_manifest.json)
artifact_sha256: not-applicable (raw safetensors, BF16 widened to f32 at hydration)
cpu_features: apple-m-series aarch64, scalar f32 reference, single thread, LLVM autovec only
command_env: CARGO_TARGET_DIR=<private> cargo test --locked -p ftts-conformance --test talker_perf_baseline -- --ignored --nocapture   (2026-08-07, ~12 concurrent agents on the host — the cv overage's likely cause)
kill_switch: none (this IS the unarmed path)
incumbent: NO ADMISSIBLE RATIO (baseline row; no incumbent comparison performed)
before_after: hydrate 29.0 s; prefill(seq 28) 146.3 s; decode mean 4437.8 ms/step over 8 steps (per-step: 4892/4594/4301/4063/4288/4169/4721/4474)
cv_percent: 6.4 (REFUSED as admissible; recorded as indicative)
equivalence: the identical forward is argmax-exact vs the oracle (talker_argmax_l3) and within DISC-002 activation budgets — timing and parity measured on the same code path
disposition: DEFER (quiet-window rerun); context: talker one-read floor 893,517,824 BF16 bytes/step (EXECUTION_CENSUS .components.talker), f32-widened traffic 1.787 GB/step, implied achieved 0.40 GB/s — ~2 orders of magnitude below DRAM bandwidth, i.e. the scalar reference is compute-bound, not bandwidth-bound; talker alone is ~55x over the 80 ms/frame real-time budget before the 15-step microdecoder and codec are counted
tally_w_l_n: 0/0/0 (baseline)
```

## Inherited priors (pre-truth-pack)

None admitted. The inherited graveyard is intentionally indexed only in `NEGATIVE_EVIDENCE.md`; re-confirmed measurements receive new `PERF-NNN` evidence IDs and must satisfy the current-tree gate.
