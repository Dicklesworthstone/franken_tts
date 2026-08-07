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
