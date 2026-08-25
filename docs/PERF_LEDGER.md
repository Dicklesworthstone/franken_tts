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
PERF-007
claim_id: warm-interactive-ttfa-certification
evidence_id: cargo test -p ftts-cli --locked --release --test warm_engine_e2e ttfa_certification -- --ignored --nocapture (frankentts-dcfn bench; per-run JSONL receipts ttfa_cert + ttfa_summary); run 2026-08-22 ~21:06-21:16 local
status: PROVISIONAL_LOCAL_WIN — INDICATIVE ONLY until a calm-window rerun (host loadavg 24-36 throughout, cv gate exceeded; see cv_percent)
model_source_commit: pinned Qwen/Qwen3-TTS-12Hz-0.6B-Base@022e286b98fbec7e1e916cb940cdf532cd9f488e; tree HEAD 1633dff at measurement
fixture_sha256: not-applicable (end-to-end synthesis timing, not a fixture seam)
artifact_sha256: qwen3-tts-12hz-0.6b-base.fttsq=597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824
cpu_features: Apple M4 Pro native; int8 W8A8 route with 6-worker KernelTeam; interactive profile (packet_frames = 1)
command_env: seed 0x5EED_0001 pinned; corpus SHORT = "Please call Stella." / LONG = full Stella elicitation paragraph + Rainbow Passage opener (pinned verbatim in the bench source); 1 discarded warmup per class then n = 24 measured runs per class; model dir ~/.cache/franken_tts/model with default.spk
kill_switch: none — this row certifies a number, it does not route a lever
incumbent: NONE QUOTED. This is an absolute certification against the plan's <=200 ms warm target, not a ratio. Upstream's claimed 97 ms first packet remains [NO ADMISSIBLE RATIO] (docs/QWEN3_TTS_STREAMING_CONTRACT.md section 4.2: internal vLLM V0, no hardware named). No pre-RT0 interleaved A/B was run, so no "improved from X to Y" claim is made here either.
results: SHORT ttfa_audible mean 200.069 ms (cv 16.04%) — statistically AT the 200 ms target under load; LONG ttfa_audible mean 225.570 ms (cv 18.74%) — a 25.6 ms miss on the long paragraph. First-byte means equaled audible means to sub-ms precision in both classes (the interactive first packet already carries above-floor samples). Every run produced identical frame counts per class (pinned seed).
cv_percent: 16.04 (short) / 18.74 (long) — both exceed the 5% quiet-window gate because the host carried load average 24-36 with dozens of active agents during measurement. Per the BASELINE rule these rows are INDICATIVE ONLY: they may not be quoted as certified targets or ratios until re-measured in a calm window with the same one-line command.
equivalence: PARITY-QUALIFIED on this tree — streaming_sink_e2e 3/3 and live_stream_cli_e2e 5/5 green at HEAD immediately before the ledger write (streaming==offline bit-identity, packet accounting, live-vs-file byte identity at the same seed). Warm-process repeat identity is separately pinned by warm_engine_e2e's contract test.
disposition: KEEP as the protocol definition and first honest measurement of the RT0/RT1 delivery chain; where remaining time goes is NOT yet instrumented natively (per-stage prefill/first-frame/codec/delivery splits exist only in the wasm timing line) — a native stage-split receipt is the named follow-up if the calm-window rerun misses. Rerun command is the evidence_id line verbatim; admission upgrades automatically once cv <= 5%.
tally_w_l_n: 0/0/1
```

```text
PERF-006
claim_id: wasm-browser-resident-set-reduction
evidence_id: site/harness/browser.mjs (real Chromium and real WebKit, persistent profiles, real 1.86 GB model over byte-Range); the engine's own per-stage `memoryBytes()` readouts and the in-wasm `linear_memory_bytes()` milestones
status: WIN (memory, not throughput — a ceiling claim, gated on the crash it removes rather than on a ratio)
model_source_commit: 2a1ea07 (session head); artifact qwen3-tts-12hz-0.6b-base.fttsq
fixture_sha256: not-applicable (end-to-end browser hydration, not a fixture seam)
artifact_sha256: qwen3-tts-12hz-0.6b-base.fttsq=597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824
cpu_features: wasm32 simd128; Chromium threaded (6-partition team), WebKit serial (the iOS path)
command_env: node site/harness/browser.mjs [--webkit] (text "The quick brown fox jumps over the lazy dog.", voice matt, seed 0)
kill_switch: none — these are structural, not routed levers
incumbent: the tree's own earlier wasm builds, same model, same harness, same instrumentation. Self-improvement, maintenance-class per Doctrine #0.5; NOT a "faster than X" claim.
before_after: peak resident 2.45 GB -> 1.64 GB Chromium / 1.61 GB WebKit, and synthesis growth +0.56 GB per press -> +0.00 GB. Four independent levers: (a) COLD_TEXT_EMBEDDING left in OPFS and read row-wise, -0.62 GB; (b) codec staged decoder-only, dropping the 0.225 GB encoder; (c) fused int8 tables built once at hydration and the artifact released, which removed a 0.46 GB per-utterance rebuild that never came back; (d) codec built from tensors moved in one at a time instead of a 0.46 GB staged file, -0.19 GB. Ordering is itself a lever here: wasm memory only grows, so a freed buffer is a hole and a hole is worth only what lands in it — artifact-first vs codec-first was 1.64 GB vs 2.10 GB for identical work.
cv_percent: NOT GATED as a throughput row. The memory figures are exact byte counts from `WebAssembly.Memory.buffer.byteLength` and `memory_size(0)`, not timings, so run-to-run variance does not apply to the claim being made. Any WALL-TIME number quoted alongside these runs is NOT admissible: another agent held 8 concurrent cargo processes on this machine throughout, violating the same-thermal-window rule.
equivalence: LOSSLESS for (a), (b) and (d) — the cold rows are the same bf16 widened by the same bit pattern (`provided_rows_reconstruct_exactly_what_the_mapped_table_would_have_gathered`), the encoder is never read by CodecCheckpoint (every tensor it loads is `decoder.*`), and the tensor-map path moves the same buffers the file path would have widened. (c) is a lifetime change, not an arithmetic one. Browser-vs-CLI frame 0 stayed 1919/1920 samples identical across the whole sequence, which is the standing check that none of this moved the audio.
disposition: KEEP. Context: the target was an iPhone tab that died during hydration; peak is now below the 1.61 GB that device was observed to survive, though the device itself remains unverified against this build. The remaining resident set is load-bearing — 0.69 GB hot q8 prefix, 0.457 GB codec f32, 0.34 GB talker f32 — and further reduction needs a format or numerics change, both of which are out of scope here.
tally_w_l_n: 1/0/0

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

```text
PERF-008
claim_id: native-route-rtf-and-ttfa-recert
evidence_id: docs/truth-pack/perf/perf-recert-loadavg.ndjson (98-sample machine-state timeline; attempt 2026-08-25 17:26Z–19:07Z+); harness benches/perf_recert_window.sh + analyzer benches/sprt_analyze.py (committed, turnkey for the next window)
status: NO_EVIDENCE — FAILED-TO-CERTIFY. No calm window existed to measure in: across 98 loadavg samples spanning ~1h45m the host ranged 0.0–492.6 (mean 160.8) during a multi-agent build storm that ultimately exhausted the machine's process table (fork EAGAIN killed the sampler). Zero measurement attempts were made — correctly, per the recorded threshold.
model_source_commit: tree at attempt = a7fe166 (v0.1.9 tag 4ff8000 lineage)
fixture_sha256: not-applicable (attempt never reached measurement)
artifact_sha256: qwen3-tts-12hz-0.6b-base.fttsq=597f7eb3314a2fe5be74fa10a6a3a28ace9e10e582c641deccd37348a0ccd824 (staged and verified present pre-attempt)
cpu_features: Apple M4 Pro (14-core); intended route = int8 default (W8A8 talker+microdecoder, f32 BLAS codec), packet_frames=1
command_env: benches/perf_recert_window.sh (admission: loadavg(1m) < 6.0 sustained 3 min, timeline recorded; harness = cargo test --target aarch64-apple-darwin --release -p ftts-cli --test warm_engine_e2e ttfa_certification -- --ignored --nocapture, binary warm_engine_e2e-336520836a6adcda built and verified)
kill_switch: n/a (never measured)
incumbent: this tree's f32 reference per §10.3 (not reached)
before_after: none — no admissible measurement exists
cv_percent: not-applicable
equivalence: parity qualification not reached
disposition: DEFER
tally_w_l_n: 0/0/1

NEXT-WINDOW PLAN (turnkey): when the fleet is quiescent, run `benches/perf_recert_window.sh` (it self-gates on the same recorded threshold, samples machine state throughout, drives the harness, lands NDJSON receipts at docs/truth-pack/perf/), then `python3 benches/sprt_analyze.py <receipts>` — write the CERTIFY/FAIL verdict plus mean ttfa_audible (short+long) and mean RTF into a PERF row here, and update README's "~450 ms ttfa"/"1.4–1.6× RT" claims to the certified figures or drop them. The v0.1.9 release notes deliberately quote no admissible numbers.
```
