# PERF_LEDGER

Gate contract: `consumer=performance claim and release gate`; `gate=only a current-tree, pinned-reference, parity-qualified measurement can support a win`; `defect_class=unreproducible or unfair performance claim`; `deletion_condition=never; superseded rows remain evidence`.

No inherited performance result is a ledger row. Inherited priors belong in `NEGATIVE_EVIDENCE.md` until re-confirmed locally; self-speedups are maintenance, not an admissible external ratio.

## Entry schema

```text
PERF-NNN
claim_id: <claim-id>
evidence_id: <artifacts/perf path or id>
status: WIN | PROVISIONAL_LOCAL_WIN | NEGATIVE | NO_EVIDENCE | VOID
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

## Local entries

None. `truth_pack_status=unavailable`; there is no pinned incumbent, fixture, parity receipt, or admissible timing result.

## Inherited priors (pre-truth-pack)

None admitted. The inherited graveyard is intentionally indexed only in `NEGATIVE_EVIDENCE.md`; re-confirmed measurements receive new `PERF-NNN` evidence IDs and must satisfy the current-tree gate.
