# Golden Artifact Inventory

Bead `frankentts-v-metamorphic-0wq` DONE-WHEN item. Every pinned artifact a gate reads,
where it lives, and how it may change. Rule for all of them: **CI never auto-updates**;
regeneration is deliberate (`UPDATE_GOLDENS=1` or an explicit regen script) and lands as a
reviewed diff.

## Golden artifacts

| Artifact | Path | Protocol | Consumer |
|---|---|---|---|
| Reference-route PCM hashes | `crates/ftts-conformance/tests/fixtures/metamorphic/golden_reference_pcm.json` | Schema v2, **per-platform** (`platforms.{os}-{arch}`); `UPDATE_GOLDENS=1 cargo test -p ftts-conformance --test metamorphic_invariants` rewrites the current platform's entry only | `metamorphic_reference_route_golden_pcm`; numerics-regression ratchet for the f32 route |
| Scrubbed robot NDJSON | `crates/ftts-cli/tests/fixtures/robot_run_golden.ndjson` | run-id/timing scrubbed capture; `UPDATE_GOLDENS=1 cargo test --release --test live_stream_cli_e2e robot_run_content_matches_its_scrubbed_golden` | robot event-stream contract |
| Robot schema contract | `crates/ftts-cli/tests/fixtures/robot_schema_v1.json` | versioned schema; changed only by a deliberate schema-version bump | `robot_contract` |
| Oracle fixture packs | `~/.cache/frankentts/oracle-fixtures/*`, staging copy `docs/truth-pack/snapshots/ft7-cpu-fp32-r1` | `fixture_manifest.json` carries per-file SHA-256; regenerated only by the oracle capture flow (`scripts/gen_reference_fixtures.py`, `scripts/measure_oracle_nondeterminism.py`) | every Contract-A L2+ seam test, PQ distributional family |
| Nondeterminism floor | `docs/truth-pack/nondeterminism-floor.json` | hash-bound into [`LadderRunner`](../../crates/ftts-conformance/src/ladder.rs); regenerated only by the floor-measurement flow | Contract-A tolerances |
| Whisper replay fixtures | `crates/ftts-conformance/tests/fixtures/franken_whisper_replay/` | frozen scorer transcripts; regenerate only when the pinned scorer version changes | ASR-path replay tests |

## Invariant → coverage map

| Invariant (bead WHAT) | Coverage |
|---|---|
| batch == singleton (strict) | Satisfied by construction: one engine, one live fan-out (`synthesis_active` CAS refuses concurrent synthesis with `EngineError::Busy`); doctrine 5 forbids batch engines |
| packet-1 == packet-4 (any schedule) | `metamorphic_packet_schedule_content_invariance` (+ sink identity e2e) |
| prompt-cache == full-prefill / warm == cold | resident/warm-start e2e asserts byte-identical output |
| streaming codec == offline codec | codec decode L2 + streaming sink e2e + CLI stream e2e |
| identical voice input → identical `.ftvoice` | `portable_roundtrip_is_byte_identical`, `recipe_hash_is_stable_and_content_sensitive` (`ftts-artifacts::voice`) |
| scalar == SIMD on every tier | Contract-A L1 kernel equivalence (`ftts robot selftest`, ladder runner) |
| thread-count invariance | `worker_partition_count_never_changes_the_audio` (two-process WAV-hash e2e) |
| corrupted/truncated artifacts fail before inference | `bad_magic_and_truncation_are_named_refusals`, `a_truncated_file_never_yields_a_partial_load`, `a_single_flipped_payload_bit_fails_digest_verification` (`ftts-artifacts::fttsq`) |
| greedy FrankenMTP ids == greedy sequential ids | `speculative_greedy_full_accept_matches_the_authoritative_frame`, `block_verifier_accepts_the_complete_sequential_greedy_draft` (unit level, synthetic weights — runs ungated) |
| verify-then-repair respects the causal mask | v1 repair regenerates the whole authoritative frame on first mismatch (`decode_frame_greedy_speculative` doc + verifier tests); suffix reuse must preserve these semantics |
| AF-3 alpha→0 == sequential byte-for-byte | **Phase-gated**: adaptive-depth surgery does not exist yet; register with Phase 5 (`frankentts-p5-surgery-00e`) |

## Deliberately NOT an invariant

"Trailing punctuation must not change the spoken prefix" — deleted in review: punctuation
legitimately alters preceding prosody in a text-conditioned AR model. Behavior study, not a
gate.
