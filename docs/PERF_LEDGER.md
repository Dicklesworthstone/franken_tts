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
PERF-006
claim_id: x86-sha-ni-artifact-verification
evidence_id: three-round interleaved direct-CLI load-stage A/B, 2026-08-11
status: PROVISIONAL_LOCAL_WIN
model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc
fixture_sha256: not-applicable (same mapped production artifact)
artifact_sha256: 597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824
cpu_features: AMD Ryzen AI 9 HX 370, SHA-NI through RustCrypto sha2 0.10
command_env: RUSTFLAGS="-C target-cpu=native" release build; FTTS_NO_RESIDENT=1 FTTS_MAX_FRAMES=4; baseline/candidate ABBA/BABA
kill_switch: not applicable; RustCrypto supplies a portable fallback when SHA-NI is absent
incumbent: current tree's portable scalar section verifier; self-speedup only, NO ADMISSIBLE EXTERNAL RATIO
before_after: scalar load stage 3536/3451/3482 ms (mean 3490 ms) versus RustCrypto 678/716/661 ms (mean 685 ms), 80.4% lower
cv_percent: scalar 1.2%; RustCrypto 4.1%
equivalence: RustCrypto is cross-checked against the portable implementation at boundary sizes; all six production WAVs were byte-identical
disposition: KEEP for one-shot mapped-section verification; streaming writers and file hashing retain the stateful implementation
tally_w_l_n: 1/0/0
```

```text
PERF-005
claim_id: wasm-packed-gemm-plus-codec-team
evidence_id: site/harness/browser.mjs run 2026-08-10 (real Chromium 151 headless, real OPFS, real COOP/COEP, real 1.86 GB model over byte-Range); engine's own stage timing line, captured from the worker console
status: PROVISIONAL_LOCAL_WIN (see cv_percent — single run, not an interleaved same-thermal-window pair)
model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc
fixture_sha256: not-applicable (end-to-end browser synthesis, not a fixture seam)
artifact_sha256: qwen3-tts-12hz-0.6b-base.fttsq=597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824
cpu_features: wasm32 simd128 (Int8Tier::WasmSimd128), 6-partition KernelTeam over SharedArrayBuffer, host Apple M4 Pro
command_env: node site/harness/browser.mjs (site/pkg threaded build; text "The quick brown fox jumps over the lazy dog.")
kill_switch: none single-lever — the packed kernel is the f32 fall-through itself; team dispatch is disarmed by a browser without crossOriginIsolated (which routes to site/pkg-serial), and the codec floor is `TEAM_FLOOR` in f32ref
incumbent: the tree's own pre-lever wasm build, same model and route, measured with the same engine timing instrumentation (NOT a pinned external incumbent — this is a self-speedup and is labelled as maintenance-class per Doctrine #0.5, not a "faster than X" claim)
before_after: codec 89,119 ms -> 6,773 ms (13.2x); talker+micro 7,641 ms -> 3,440 ms (2.2x); prefill 342 ms -> 137 ms; total 97.3 s -> 10.35 s (9.4x). Frame budget reshaped: codec 92% -> 65%, talker+micro 7.9% -> 33%
cv_percent: NOT GATED — one run per side, different utterance lengths between the two measurements (5.12 s vs 3.20 s of audio). The per-stage ratios are like-for-like because both come from the same instrumented timing line, but no repeated interleaved pairs were taken. This row may NOT be quoted as a certified ratio until re-measured under the quiet-window protocol.
equivalence: BIT-IDENTICAL to the scalar reference per element. `packed_matches_scalar_bit_for_bit` pins 11 shapes including k=0, k=1, tile-exact 4x8, NR/MR remainders and a multi-panel k=1024; `every_partition_count_reproduces_the_serial_bits` pins partitions 1/2/3/5/6/8 at real codec geometry (block_00 K=7168; transformer 512x512). Each output element accumulates ascending-k into its own slot, so blocking and partitioning change only WHICH element is computed WHEN. NOTE this TIGHTENS parity: the previous wasm fall-through used eight independent partial chains, a non-reference reduction order, so the browser now tracks the scalar reference more closely than before the lever.
disposition: KEEP; context: ported from franken_numpy/crates/fnp-linalg (packed_gemm_serial_tiled), f64->f32, packing adapted to this project's [n,k] weight layout so no transpose is materialized. NE-004 ("register blocking is neutral, ~3%") does NOT apply — that was measured on int8 GEMV at m=1, where a packed panel has nothing to amortize over; the codec's im2col GEMMs have m>>1, which is the regime blocking exists for. All six codec GEMM sites funnel through f32ref::linear_with_accumulation, so one kernel upgraded every convolution, ConvNeXt pointwise pair and transformer projection at once.
tally_w_l_n: 1/0/0
```

```text
PERF-004
claim_id: codec-dense-blas-form
evidence_id: interleaved 3-round codec_time A/B 2026-08-09 + codec_decode_l2 ratchet re-pin; commit 9528cac
status: KEEP
model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc
fixture_sha256: 5ec2bc3f3217f9e026198c0694b3993d9911f7e954ea726c69ebd95e7d5ba4dd (fixture_manifest.json, via codec_decode_l2)
artifact_sha256: not-applicable (codec loads raw speech_tokenizer safetensors)
cpu_features: Apple M4 Pro aarch64, Accelerate BLAS
command_env: codec_time example, 32 frames, three interleaved arms (FTTS_INT8_CODEC=0 / convnext / all), fleet-loaded host
kill_switch: FTTS_INT8_CODEC=convnext|all|transformer re-arms the int8 arms; off macOS the named BLAS request degrades to the scalar reference
incumbent: the tree's own previous default (convnext int8 + scalar-then-BLAS transformer dense); historical scalar f32 dense was 57-72 ms/frame on this family
before_after: all-BLAS f32 20.5/21.1/21.2 ms/frame vs convnext-int8 default 21.1/22.1/22.1 (3/3 rounds); all-int8 19.4-19.9 remains excluded by its failed 1.24 dB spectral gate
cv_percent: QUIET-WINDOW rerun (load ~5, 4 interleaved super-rounds): all-BLAS 20.5/21.6/20.5/20.0 (mean 20.65, cv 3.3); convnext-int8 20.8/22.7/20.3/20.5 (mean 21.08, cv 5.2 — one flip round); all-int8 18.6-19.1 (mean 18.75, cv 1.3, stays quality-excluded). Honest restatement: vs convnext-int8 the BLAS arm is equal-or-faster WITHIN NOISE on a quiet host (clearly faster under load); the KEEP rests on the fidelity improvement (ratchet tightened at all eight transformer seams, zero quantization) with speed at least at parity
equivalence: same-seed code stream identical (talker untouched, sample counts equal); codec waveform moves at the pinned f32-reorder level — the codec_decode_l2 ratchet TIGHTENED at all eight transformer seams and is re-pinned to the measured values; streaming==offline green; snake/convnext/gemm bisects green
disposition: KEEP; the reference's nn.Linear IS addmm (bias-seeded beta=1 BLAS GEMM), so the named BLAS form is the more oracle-faithful arithmetic, not a tolerance concession — quantizing the codec now costs speed AND fidelity, hence int8 arms demoted to opt-in
tally_w_l_n: 1/0/0
```

```text
PERF-003
claim_id: startup-hydration-campaign
evidence_id: interleaved 3-round load-stage A/B vs installed v0.1.3 binary, 2026-08-09; commits 9478b3c/c311ab1/c883b6c/6c1b80d/4df93f0
status: KEEP (quiet-window certified 2026-08-09: load average ~5, interleaved 4 rounds — see before_after)
model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc
fixture_sha256: not-applicable (load-stage wall time, no fixture)
artifact_sha256: 597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824
cpu_features: Apple M4 Pro aarch64
command_env: ftts say --seed 11 "Hi." (load-stage NDJSON elapsed), new binary interleaved with brew-installed v0.1.3
kill_switch: FTTS_LOAD_THREADS=1 (serial widen), FTTS_ARTIFACT_Q8=0 (requantize + no elision), FTTS_INT8=0 (f32 reference, elision auto-off)
incumbent: installed v0.1.3 (pre-campaign hydration; its own ~1.1 s generator-build requantize additionally falls OUTSIDE its load stage, so the shown gap understates the total)
before_after: QUIET-WINDOW (load ~5, interleaved): v0.1.3 9547/9173/8895/9271 ms (mean 9222, cv 2.9%) vs campaign 5118(cold)/3695/3841/3612 ms (warm mean 3716, cv 3.1%) = 2.48x load stage; v0.1.3 additionally pays ~1.1 s post-load requantize the campaign eliminated. Earlier loaded-host pairs (6.3/6.9/8.0 vs 4.8/5.7/5.3 s) retained as the under-load corroboration
cv_percent: 3.1 (campaign warm rounds) / 2.9 (incumbent) — ADMISSIBLE; first campaign round (5118 ms) excluded as the cold round and shown. FTTS_LOAD_THREADS default validated same window: capped mean 3397 ms cv 0.5% vs serial 3527 ms vs uncapped bimodal (3382/6471/3384 — spikes even on a quiet host)
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
