# DISCREPANCIES

Gate contract: `consumer=ConformanceExact and ProductionQuality`; `gate=accepted numeric divergence requires a measured impact and an active restoration path`; `defect_class=numeric or semantic divergence`; `deletion_condition=never; entries are immutable evidence records`.

Local entries are empty at truth-pack seed time. Do not create a `DISC-NNN` record from an inherited result: impacts and restoration switches must be measured on this model.

## Entry schema

```text
DISC-NNN
claim_id: <claim-id>
evidence_id: <artifacts path or id>
status: ACCEPTED | REJECTED | VOID
model_source_commit: <pinned commit>
oracle_source: <file_sha256:lines>
fixture_sha256: <sha256>
artifact_sha256: <sha256 or none>
cpu_features: <dispatched feature string>
command_env: <exact command and environment>
reference_behavior: <quoted source behavior>
our_behavior: <file:fn behavior>
kill_switch: <FTTS_* restoring reference behavior>
measured_impact: <metric, mean, and tail where applicable>
resolution: <decision>
review_date: <YYYY-MM-DD>
test_status: XFAIL | PASS
```

## Local entries

One local tokenizer divergence is recorded below. Other component-level numeric divergences remain
unmeasured.

### DISC-001 tokenizer-regex-official-vs-native

`claim_id: tokenizer-regex-official-vs-native`; `evidence_id: docs/truth-pack/tokenizer/tokenizer_conformance.json`; `status: ACCEPTED`; `model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc`; `oracle_source: docs/truth-pack/tokenizer/OQ11_TOKENIZER.md:72-148`; `fixture_sha256: 78f56d6ab68f2a1927ab33a37497908b722d4d8d77df47ed43149a4fbfeec99a`; `artifact_sha256: vocab.json=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910, merges.txt=599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3, tokenizer_config.json=dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670`; `cpu_features: not-applicable (pure Rust tokenization)`; `command_env: CARGO_TARGET_DIR=/tmp/frankentts-tokenizer-target cargo test -p ftts-model-qwen --no-fail-fast`; `reference_behavior: the official Qwen3-TTS entrypoints load Qwen2TokenizerFast with fix_mistral_regex=True`; `our_behavior: crates/ftts-model-qwen/src/tokenizer.rs:13-14 and :28-39 default to the official Mistral expression, with the native Qwen expression only behind the explicit switch`; `kill_switch: FTTS_TOKENIZER_REGEX=native`; `measured_impact: 6/92 pinned corpus cases change token ids under native mode (five mixed-case identifier cases and Thai); all 92 official ids and their NFC decode values pass`; `resolution: retain official as the conformance default until a separate listening experiment evaluates the native alternative`; `review_date: 2026-08-06`; `test_status: PASS`.

## Inherited priors (pre-truth-pack)

None imported as discrepancy records. Sibling numerical outcomes are hypotheses only; add a `DISC-NNN` only after this project's pinned source, fixture, impact, and `FTTS_*` restoration switch exist.
