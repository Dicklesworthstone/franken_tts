# NEGATIVE_EVIDENCE

Gate contract: `consumer=Phase-4 perf ritual`; `gate=sweep before spending a performance lever`; `defect_class=repeated dead optimization`; `deletion_condition=never; a re-test appends a new local evidence record`.

`status: inherited (pre-truth-pack)` means a sibling result is a hypothesis to re-confirm on Qwen3-TTS shapes and target silicon. It is never local evidence and cannot support a performance claim.

## Entry schema

```text
NE-NNN
claim_id: <claim-id>
evidence_id: <artifacts path or inherited source>
status: inherited (pre-truth-pack) | NEGATIVE | NO_EVIDENCE | VOID
model_source_commit: <pinned commit or inherited/not-local>
fixture_sha256: <sha256 or inherited/not-local>
cpu_features: <dispatched feature string or inherited/not-local>
command_env: <exact command and environment or inherited/not-local>
kill_switch: <FTTS_* or not-applicable>
before_after: <measured ratio or inherited result>
equivalence: <proof tier or inherited/not-local>
killing_metric: <metric that decided the outcome>
disposition: <KEEP | REVERT | DEFER>
do_not_retry: unless <specific changed condition>
tally_local_w_l_n: <wins/losses/neutrals>
```

## Local entries

```text
NE-005
claim_id: int4-microdecoder-arithmetic-landed-routing-not-decided
status: OPEN OBLIGATION (not a rejection — recorded here so the gates are not quietly skipped)
date: 2026-08-10
what_exists: crates/ftts-kernels/src/int4.rs — symmetric per-output-channel W4A8, two biased
  nibbles per byte, exact i32 accumulation with the +8 bias cancelled by ONE correction term
  (`sum(x*nibble) - 8*sum(x)`) so the inner loop is mask/shift/MAC with no per-element sign
  extension. Four tests pass: bias cancellation exact vs a signed reference at k in
  {1,2,3,7,8,15,64,127,1024}; levels confined to [-7,7] with -8 never emitted; negation exact;
  packed storage exactly half of Q8.
gate_a_RESULT: **FAILED, decisively.** Measured 2026-08-10 on M4 Pro (dispatched route
  `neon-sdot`), release build, at the microdecoder's own shapes (hidden 1024, intermediate 3072,
  16 Q / 8 KV heads x 128) at m = 1, 75 rounds = 15 depths x 5 layers, INTERLEAVED with 7 repeats
  reporting the minimum per variant:

    one layer, 75 rounds:  q8-scalar 18.9 ms | q4-scalar 418.8 ms | q8-route (SDOT) 18.8 ms
    int4 vs scalar int8 .... 0.05x  (22x SLOWER)
    int4 vs shipping int8 .. 0.04x  (25x SLOWER)
    weight bytes per layer:  q8 15.7 MB -> q4 7.9 MB

  A FIRST attempt timed the variants in sequential blocks and produced 98.60 ms for k_proj against
  5.14 ms for v_proj — identical 1024x1024 shapes, 19x apart. That run is discarded as noise; only
  the interleaved figures above are admissible, and under them k_proj/v_proj agree.

why_it_lost (first principles): the unpack is not the halved traffic's junior partner, it is the
  whole cost. Per weight the nibble path pays a mask, a shift, two i32 widenings and an
  activation-sum update where Q8 pays one multiply-add — and, decisively, the byte-splitting loop
  DEFEATS LLVM's autovectorizer. This is the same disease NE-001 recorded for the manual 8-lane
  int8 loop (~15x slower than plain `Scalar`): the manual lane structure blocks the vectorizer
  while the plain shape vectorizes to memory bandwidth. Corroborating datum from this same run:
  q8-scalar (18.9 ms) and hand-written SDOT (18.8 ms) are indistinguishable at m = 1, which is
  NE-INH-003 reproduced and says the int8 path is ALREADY at the bandwidth limit there. Halving the
  bytes therefore cannot pay for a 22x compute penalty.

do_not_retry_predicate: do NOT re-run the listening gate, and do NOT route int4, while the unpack
  is scalar. The measurement above is the precondition; re-running it is only meaningful after an
  in-register SIMD unpack exists (NEON: one 16-byte load, mask/shift into two i8 lanes, feed SDOT;
  wasm: the v128 equivalent). That is NE-INH-004's documented escape clause and remains the ONLY
  route by which this lever can come back.

second_strike (2026-08-23, WhiteLynx, frankentts-je4i): the predicate's precondition was BUILT and
  the gate re-run. `crates/ftts-kernels/src/int4.rs` gained a NEON island (`mod neon_q4`,
  FEAT_DotProd-gated, bit-exact vs scalar by new cross-tier tests at k in {1,15,16,17,31,32,33,63,
  1024} x m in {1,2}): one 16-byte load per packed block, `vuzp1q`/`vuzp2q` deinterleave of two
  activation loads (packed byte j holds elements 2j/2j+1, so a contiguous load would mispair),
  shift/mask nibble lanes reinterpreted as signed, straight into `vdotq_s32`; bias cancels via the
  same single correction term. Measured on M4 Pro with the SAME harness, shapes, interleaving, and
  min-of-7 protocol as the first strike:

    one layer, 75 rounds:  q8-scalar 20.0 ms | q4-scalar 42.7 ms | q8-route (SDOT) 22.2 ms
                           q4-neon  (see per-projection rows; totals below)
    int4-neon vs shipping int8 ... 0.52x  (~2x SLOWER)

  why_it_lost_again: the deinterleave is three extra instructions per block pair and SDOT consumes
  both nibble lanes separately, so the q4 path pays ~2.5x the instruction count of the int8 path
  for half the bytes. At m = 1 with each projection's weights warm in cache across rounds, the
  kernel is instruction-bound, not bandwidth-bound — the same regime NE-INH-003 documented for int8
  (q8-scalar == q8-SDOT there). Halved traffic cannot pay doubled instruction count when RAM
  bandwidth is not the binding constraint. The cache-residency thesis would need the FULL ~40 MB
  body streamed from RAM per frame to bind bandwidth; this harness deliberately isolates one layer,
  so it measures the compute side honestly and the residency side not at all.

status_after_second_strike: routing remains OFF; listening gate remains unrun. The unpack exists and
  is exact; the speed gate failed it again under its own pre-committed protocol. Any third attempt
  must change the REGIME (e.g. measure the full-body streaming working set where bandwidth binds),
  not re-tune this kernel — and must say so up front.

what_is_NOT_done: NOTHING ROUTES TO IT. Doctrine #2 requires BOTH gates before the microdecoder
  may select this path — (a) faster end-to-end on each target ISA INCLUDING unpack cost, measured
  not assumed, and (b) blind listeners cannot distinguish it under the equivalence-bound protocol
  (identity, naturalness, sibilance, breath, pitch stability, long-form prosody). Neither has been
  run. A smaller file that runs slower or subtly damages speaker identity is a FAILED artifact.
why_it_is_worth_gating_now: W11 measured the frame at codec 65% / talker+micro 33% after the
  packed GEMM and kernel team took the codec down 13x. The microdecoder body is re-read 15x per
  frame, so Q4 (~79 MB -> ~40 MB) is the one place cache residency is plausibly winnable. Before
  2026-08-10 this lever targeted 7.9% of the frame; it now targets a third of it.
honest_risk: 15 levels against 255 is ~17x the quantization step. `quantization_error_is_bounded_
  by_the_level_step` asserts only that the quantizer is not BROKEN; it makes no claim that the
  error is inaudible, and that distinction is the entire point of gate (b).
do_not_retry_predicate: seq-16 speculative batching remains DEAD (NE-002, drafter ~1% acceptance).
  Do not revive it as a way to amortize int4 unpack.
```


### NE-003 wasm-simd128-is-a-1.8x-lever-not-a-4x-one

`claim_id: wasm-int8-simd128-ceiling`; `evidence_id: scratchpad/bench.mjs — node 25, in-process interleaved ABBA, 7 pairs per shape, 40 rounds per timing, both routes same process/allocator/cache state`; `status: NEGATIVE (against the predicted magnitude; the tier itself is a KEEP)`; `model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc (weights not consumed; synthetic operands at pinned census shapes)`; `fixture_sha256: not-applicable`; `cpu_features: Apple M4 Pro, wasm32 + simd128 under V8`; `command_env: wasm-pack build --target nodejs --release with RUSTFLAGS=-C target-feature=+simd128; node bench.mjs`; `kill_switch: Int8Tier::Scalar remains dispatchable`; `before_after: scalar -> wasm-simd128, per-dot: k=1024/n=1024 299.9 -> 205.4 ns (1.49x); k=3072/n=1024 1026.6 -> 599.8 ns (1.71x); k=1024/n=3072 300.4 -> 204.2 ns (1.47x)`; `equivalence: exact i32 equality with Scalar by construction (every i8xi8 product fits i16, every pairwise sum fits i32, integer addition is associative)`; `killing_metric: per-dot wall time at real census shapes`; `disposition: KEEP the tier, REJECT the 4-8x estimate that motivated it`; `do_not_retry: do not budget more than ~2x for any SIMD128 int8 kernel on this instruction set — wasm has no int8 dot product, so the best available shape is 4 widenings + 2 i32x4.dot_i16x8_s + 2 adds per 16 bytes = 2.0 MACs/op against NEON SDOT's 16 MACs/instruction; the arithmetic ceiling with the activation widening hoisted is 2.67 MACs/op`; `tally_local_w_l_n: 0/1/0`.

### NE-004 register-blocking-does-not-help-the-wasm-int8-gemv

`claim_id: wasm-int8-4column-register-blocking`; `evidence_id: same harness as NE-003, immediately before/after the blocked kernel landed`; `status: NEGATIVE (neutral)`; `cpu_features: Apple M4 Pro, wasm32 + simd128 under V8`; `kill_switch: the blocked path is behind cfg(target_arch = "wasm32") and matches only Int8Tier::WasmSimd128`; `before_after: 1.49 -> 1.53x, 1.71 -> 1.79x, 1.47 -> 1.53x — about 3%, against a predicted 23% from the op-count arithmetic (26 ops per 64 MACs instead of 32)`; `equivalence: exact i32 equality preserved; only the loop nest changed`; `killing_metric: per-dot wall time at real census shapes`; `disposition: KEEP (consistent across all three shapes, no numerics change, no measurable cost) but do NOT treat it as a lever`; `do_not_retry: the doctrine's "the instruction is not the lever; the blocking is" is an *instruction-bound* heuristic and does not apply here. Hoisting the shared activation widening bought nothing, which says the wasm GEMV is not instruction-issue-bound: measured ~0.2 ns per int8 element (~5 GB/s) where SIMD128 issue rates predict several times that. Further kernel micro-optimization is exhausted; the remaining levers are parallelism (threads) and *traffic reduction* (int4 weights, the seq-16 microdecoder batching that removes a 15x reread), not instruction selection`; `tally_local_w_l_n: 0/0/1`.

### NE-006 the-ios-hydration-crash-is-not-reachable-by-memory-tuning — **RETRACTED**

**This entry was wrong, and the way it was wrong is the useful part.** It concluded that 2.45 GB
was irreducible because no single dominant allocation remained. The premise was true and the
conclusion did not follow: 622 MB — 47% of the artifact — was the COLD_TEXT_EMBEDDING section,
held resident to serve a few hundred 4 KB rows per utterance. It was invisible to the census below
because that census only asked *what hydration widens*, and this section is not widened; it is
staged as raw bytes and read in place. Measuring one axis and concluding about the total is how a
0.6 GB allocation hid behind the sentence "there is no large allocation left to remove". Peak is
now 1.86 GB — see the `perf(wasm)` commit that leaves the cold section in OPFS.

The transferable rule: "no big allocation remains" is a claim about a *search*, not about a
system, and it is only as good as the axis the search ran along. Before concluding that a resident
set is irreducible, enumerate it by ACCESS FREQUENCY as well as by size — the thing to remove is
not the biggest allocation, it is the biggest one that is barely read. The decomposition below is
still accurate for what it measured, so it is kept rather than deleted.

Original entry, retained for the record:

`claim_id: ios-playground-hydration-peak`; `evidence_id: site/harness/browser.mjs against the real 1.86 GB assets, both engines, persistent profiles; per-stage wasm memory readouts; fttsq directory census over docs/truth-pack/snapshots/hf/qwen3-tts-12hz-0.6b-base.fttsq`; `status: NEGATIVE (the lever class, not the goal)`; `cpu_features: wasm32+simd128 under WebKit and V8, Apple M4 Pro host`; `kill_switch: not-applicable — this is a measurement, no code path was gated`; `before_after: peak resident 2.45 GB, decomposed as 1.31 GB q8 artifact (read in place through MappedFttsq, zero-copy, cannot be freed while the engine lives) + 0.70 GB widened codec (its safetensors source is already f32, so keeping it narrow saves nothing) + 0.34 GB of talker f32 spread across 477 tensors with no dominant member + ~0.10 GB gathered rows and scratch`; `equivalence: not-applicable`; `killing_metric: peak wasm linear memory at hydrate-talker`; `disposition: REJECT further memory tuning; SHIP an upfront device warning instead (site/app.js isMemoryConstrainedDevice)`; `do_not_retry: do not look for a big single allocation — there is not one. The cold text embedding is already gathered per-row rather than widened, the codec checkpoint is already decoder-only, and the streaming ModelStaging already keeps the codec source out of the artifact's window (2.67 GB against 3.35 GB). Arming FTTS_INT8_CODEC makes this WORSE, not better: that route memoizes the Q8 form ALONGSIDE the f32 weights it quantizes from. The only remaining levers both carry costs already ruled out of scope — a narrower artifact format (rejected: it would break every installed version) or storing the codec quantized and dropping its f32 (a wasm audio numerics change, ~0.5 GB, unevaluated)`; `tally_local_w_l_n: 0/1/0`.

### NE-007 ephemeral-browser-profiles-fabricate-a-webkit-opfs-failure

`claim_id: webkit-opfs-unsupported`; `evidence_id: site/harness/engine_caps.mjs, ephemeral vs launchPersistentContext, both engines on the real served origin`; `status: NEGATIVE (the finding was an artifact of the harness)`; `kill_switch: not-applicable`; `before_after: ephemeral WebKit reported opfs:true but threw "UnknownError: The operation failed for an unknown transient reason (e.g. out of memory)" on the first getFileHandle, and the full harness run stalled at 5 stages with msg:load never arriving. With a persistent profile WebKit matches Chromium exactly (createWritable:true, positionalWrite:16) and the same run reaches 13 stages and synthesizes at 0.25x real time`; `equivalence: not-applicable`; `killing_metric: OPFS capability probe on the served origin`; `disposition: FIXED — both engine_caps.mjs and browser.mjs now launch persistent profiles`; `do_not_retry: never conclude a browser lacks an OPFS capability from an ephemeral context. Chromium tolerates having nowhere to write and WebKit does not, so a Chromium-only harness hides this class of difference entirely — and here it cost real time chasing a site defect that did not exist. A capability probe must run on the real origin AND with real storage`; `tally_local_w_l_n: 0/1/0`.

### NE-008 q8-cold-text-embedding-as-shipping-default

`claim_id: local-embed-q8-default-v1`; `evidence_id: owner listening pairs 2026-08-12 (release binary, --voice james, seeds 7/11, bf16 vs grouped-Q8 artifact from the pinned checkpoint) + per-row error analysis of both artifacts (151,936 rows, all elements)`; `status: NEGATIVE (as the shipping default; the convert-time flag remains)`; `model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc`; `cpu_features: Apple M4 Pro`; `command_env: ftts convert --embed-q8; ftts say --no-resident --model <q8 dir>`; `kill_switch: bf16 IS the default — the lever only arms via ftts convert --embed-q8`; `before_after: artifact 1,312,015,713 -> 1,020,298,885 bytes (-22.2%); per-row SQNR floor 23.8 dB per-row-scaled, 35.0 dB group-64-scaled (zero rows under 30 dB); owner preferred bf16 on BOTH blind test pairs`; `equivalence: FAILED the only arbiter that matters — the owner hears a difference and prefers bf16; note the embedding perturbs token SELECTION only (codec renders tokens identically), so the effect is different sampled trajectories, not a degraded rendering of the same trajectory`; `killing_metric: owner blind preference, 2/2 against Q8`; `disposition: REJECT as default; RETAIN ftts convert --embed-q8 as a documented opt-in for size-constrained deployments`; `do_not_retry: as a default, not without the owner asking. Any future attempt should target token-trajectory preservation (e.g. higher-precision rows for the high-frequency token ids that dominate selection), not further SQNR polishing — group-64 already fixed the measurable floor and still lost`; `tally_local_w_l_n: 0/2/0`.

### NE-001 hand-shaped-8lane-int8-loop

`claim_id: local-int8-autovec-lane-shape`; `evidence_id: crates/ftts-kernels/examples/int8_shape_bench.rs run 2026-08-08 (M4 Pro, shared host under swarm load — cv rows above 5 percent, ratios indicative not certified)`; `status: NEGATIVE`; `model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc (weights not consumed; synthetic operands at pinned census shapes)`; `fixture_sha256: not-applicable (synthetic all-shape operands)`; `cpu_features: Apple M4 Pro, FEAT_DotProd + FEAT_I8MM present`; `command_env: CARGO_TARGET_DIR=<private> cargo run --release --locked -p ftts-kernels --example int8_shape_bench`; `kill_switch: FTTS_INT8_TIER=autovec (route remains selectable for re-measurement)`; `before_after: the hand-shaped 8-lane i8-to-i32 lane loop (Int8Tier::Autovec) ran approximately 15x SLOWER than the plain left-to-right scalar loop (Int8Tier::Scalar) at every m=1 census shape — the manual lane structure defeats LLVM's widening-MAC autovectorization, while the naive scalar shape vectorizes to about 50 GB/s and ties hand-SDOT`; `equivalence: exact i32 equality across all three tiers (selftest S8S8 rows + tier-equality tests)`; `killing_metric: m=1 real-shape wall time`; `disposition: KEEP (route retained as A/B datapoint, never dispatched by default; dispatch order is NeonSdot where detected, else Scalar)`; `do_not_retry: unless LLVM major changes or a quiet-window cv-gated rerun contradicts the 15x gap`; `tally_local_w_l_n: 0/1/0`.

This is also the first local re-confirmation signal for NE-INH-001/NE-INH-003: the winning int8 m=1 shape on Apple Silicon is the one LLVM autovectorizes, and hand SDOT merely ties it at memory bandwidth. Certification to ledger quality still requires a quiet-window rerun that passes the cv gate.

### NE-002 speculative-sampling-with-transition-sketch-drafter

`claim_id: local-frankenmtp-sampled-drafter-v1`; `evidence_id: FTTS_SPEC_PROBE=1 instrumentation in decode_frame_with_selector_inner + probe run 2026-08-09 (960 depth samples, 64 production frames, enrolled voice, seed 0)`; `status: NEGATIVE`; `model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc`; `fixture_sha256: not-applicable (production sampling on the pinned checkpoint)`; `cpu_features: Apple M4 Pro`; `command_env: FTTS_INT8=1 FTTS_SPEC_PROBE=1 int8_greedy_divergence <snapshot> voice "..." 64 production`; `kill_switch: not-applicable (lever never armed; probe is diagnostic-only, default off)`; `before_after: the FrankenMtpDrafter (previous-frame residuals + 64-bucket transition sketch) would be ACCEPTED by exact point-mass rejection sampling with mean probability ~0.01 per depth under the production T0.9/top-50 distribution; expected accepted-prefix length 0.04 of 15; zero near-full-accept frames in 64. Speculative cost ~16.6 body-read units vs 15.0 sequential — strictly worse`; `equivalence: not-applicable (lever rejected before implementation)`; `killing_metric: drafter acceptance probability under the production sampling distribution`; `disposition: REVERT (never implemented; probe instrumentation retained behind FTTS_SPEC_PROBE for future drafters)`; `do_not_retry: unless a drafter with measured mean per-depth acceptance above ~0.6 exists (e.g. a small learned draft head), or the strict-greedy tier is the target (argmax-match acceptance is a different, unmeasured number)`; `tally_local_w_l_n: 0/2/0`.

## Inherited priors (pre-truth-pack)

### NE-INH-001 hand-wide-simd-glue

`claim_id: prior-hand-wide-simd-glue`; `evidence_id: inherited-franken-ocr-kernel-campaign`; `status: inherited (pre-truth-pack)`; `model_source_commit: inherited/not-local`; `fixture_sha256: inherited/not-local`; `cpu_features: inherited/not-local`; `command_env: inherited/not-local`; `kill_switch: not-applicable`; `before_after: hand-wide-SIMD approximately 5x slower than LLVM autovectorized scalar glue`; `equivalence: inherited/not-local`; `killing_metric: real-shape wall time`; `disposition: DEFER`; `do_not_retry: unless current LLVM major and exact Qwen3-TTS shape are A/B measured`; `tally_local_w_l_n: 0/0/0`.

### NE-INH-002 unblocked-smmla

`claim_id: prior-unblocked-smmla`; `evidence_id: inherited-franken-ocr-kernel-campaign`; `status: inherited (pre-truth-pack)`; `model_source_commit: inherited/not-local`; `fixture_sha256: inherited/not-local`; `cpu_features: inherited/not-local`; `command_env: inherited/not-local`; `kill_switch: not-applicable`; `before_after: unblocked SMMLA is load-bound and slower than SDOT`; `equivalence: inherited/not-local`; `killing_metric: real-shape wall time`; `disposition: DEFER`; `do_not_retry: unless register blocking, target silicon, and exact MTP/talker shape are A/B measured`; `tally_local_w_l_n: 0/0/0`.

### NE-INH-003 m1-autovec-vs-sdot

`claim_id: prior-m1-autovec-vs-sdot`; `evidence_id: inherited-franken-ocr-m4-campaign`; `status: inherited (pre-truth-pack)`; `model_source_commit: inherited/not-local`; `fixture_sha256: inherited/not-local`; `cpu_features: Apple-M4`; `command_env: inherited/not-local`; `kill_switch: not-applicable`; `before_after: LLVM autovec beat hand-SDOT by 4.4x at m=1`; `equivalence: inherited/not-local`; `killing_metric: m=1 real-shape wall time`; `disposition: DEFER`; `do_not_retry: unless target CPU, LLVM major, and exact m=1 shape are A/B measured`; `tally_local_w_l_n: 0/0/0`.

### NE-INH-004 int4-unpack-to-memory

`claim_id: prior-int4-unpack-to-memory`; `evidence_id: inherited-franken-ocr-int4-campaign`; `status: inherited (pre-truth-pack)`; `model_source_commit: inherited/not-local`; `fixture_sha256: inherited/not-local`; `cpu_features: inherited/not-local`; `command_env: inherited/not-local`; `kill_switch: not-applicable`; `before_after: int4 unpack-to-memory costs approximately 2.5x int8 traffic and was 5.8x slower`; `equivalence: inherited/not-local`; `killing_metric: bandwidth-bound tensor wall time`; `disposition: DEFER`; `do_not_retry: unless an in-register nibble-to-MAC kernel is used and end-to-end TTS A/B is measured`; `tally_local_w_l_n: 0/0/0`.

### NE-INH-005 amx-f32

`claim_id: prior-amx-f32`; `evidence_id: inherited-franken-ocr-kernel-campaign`; `status: inherited (pre-truth-pack)`; `model_source_commit: inherited/not-local`; `fixture_sha256: inherited/not-local`; `cpu_features: inherited/not-local`; `command_env: inherited/not-local`; `kill_switch: not-applicable`; `before_after: AMX-f32 lost to int8`; `equivalence: inherited/not-local`; `killing_metric: bandwidth-bound real-shape wall time`; `disposition: DEFER`; `do_not_retry: unless the candidate is an int8 implementation on available target hardware`; `tally_local_w_l_n: 0/0/0`.

### NE-INH-006 naive-fused-ops

`claim_id: prior-naive-fused-ops`; `evidence_id: inherited-franken-ocr-forward-campaign`; `status: inherited (pre-truth-pack)`; `model_source_commit: inherited/not-local`; `fixture_sha256: inherited/not-local`; `cpu_features: inherited/not-local`; `command_env: inherited/not-local`; `kill_switch: not-applicable`; `before_after: fused forward retaining naive scalar operations regressed 3x to 10x`; `equivalence: inherited/not-local`; `killing_metric: end-to-end wall time`; `disposition: DEFER`; `do_not_retry: unless every fused operation has a proven peak kernel and parity receipt`; `tally_local_w_l_n: 0/0/0`.

### NE-INH-007 sequential-thermal-ab

`claim_id: prior-sequential-thermal-ab`; `evidence_id: inherited-franken-ocr-perf-ritual`; `status: inherited (pre-truth-pack)`; `model_source_commit: inherited/not-local`; `fixture_sha256: inherited/not-local`; `cpu_features: inherited/not-local`; `command_env: inherited/not-local`; `kill_switch: not-applicable`; `before_after: unchanged code drifted more than 30 percent in sequential thermal measurements`; `equivalence: inherited/not-local`; `killing_metric: timing coefficient of variation`; `disposition: DEFER`; `do_not_retry: unless an interleaved same-thermal-window A/B passes the cv gate`; `tally_local_w_l_n: 0/0/0`.

### NE-009 artifact-embedded-codec-before-dtype-residency

`claim_id: embed-hot-codec-decoder-section-in-fttsq-v1`; `evidence_id: frankentts-kbs4-investigation-2026-08-22`; `status: local (current tree, arithmetic from committed loader choreography + measured OPFS device limits)`; `model_source_commit: n/a (loader/format analysis, no numerics change)`; `fixture_sha256: codec prefix 49fa6cc7af115ea007ee31ab2bde4673658371c76109c73efcad6d0e179c8eef`; `cpu_features: n/a`; `command_env: n/a`; `kill_switch: do-not-implement (bead frankentts-zzqi records the preconditions)`; `before_after: embedding a HOT_CODEC_DECODER section into .fttsq v1 forces the whole 1.77 GB artifact resident during its own staging because prefix elision is trailing-only, peaking ~2.9 GB versus ~1.84 GB for the sidecar flow — past the measured iPhone grow-unshared ~2 GB reclaim line;` `equivalence: n/a (bytes identical either way)`; `killing_metric: wasm linear-memory staging high-water mark`; `disposition: DEFER`; `do_not_retry: unless codec residency is first made dtype-small (frankentts-zzqi Q8-compute all-targets route) AND a mid-artifact section-elision mechanism exists (format v2); also note bf16-residency is not bit-safe here — byte audit of the pinned checkpoint shows ~99.999 percent of f32 values have nonzero low mantissa halves, and the legacy 1.36 GB residency figure is stale (true CodecCheckpoint residency is 457292548 B of F32)`; `tally_local_w_l_n: 0/0/0`.

### NE-010 relaxed-simd-fused-int8-dot-cannot-be-exact

`claim_id: wasm-relaxed-simd-fused-dot-as-wasm-relaxed-simd-tier`; `evidence_id: site/harness/probe_relaxed_dot.cjs + site/harness/fingerprint_relaxed_ops.cjs, executed 2026-08-26`; `status: NEGATIVE (spec-level exactness latitude on the shipping engine)`; `model_source_commit: n/a (kernel-instruction probe; no model numerics involved)`; `fixture_sha256: synthetic vectors only`; `cpu_features: macOS arm64; browsers probed: Chromium 150.0.0.0 headless (playwright), Node v25.4.0 secondary; wat2wasm 1.0.41 for assembly`; `command_env: node site/harness/probe_relaxed_dot.cjs — assembles a probe module with wat2wasm around an unreachable/nop sentinel, splices opcode FD 93 01 (final-spec i32x4.dot_i8x16_i7x16_add_s) into the exact byte slot, validates arity, executes adversarial vectors against scalar reference lanes`; `kill_switch: never route Q8 math through any i8x16.relaxed_dot_* instruction and never build the pkg-relaxed bundle negotiation while this entry stands`; `before_after: candidate lever replaced the SIMD128 island's six widening ops per 16-byte block with one fused dot-add (~3x fewer ops in the hot loop); measured reality: Chromium executes FD 93 01 with signed-x-unsigned latitude on out-of-i7-range bytes — all(-128,-128) returned -65536 per lane (= 128 treated as u8) instead of the exact integer +65536, while all(+127,+127) matched exactly (+64516); conforming engines may pick their own lowering here, so results are engine-dependent by design`; `equivalence: unsatisfiable — the parent gate demands per-element bit equality with Scalar on EVERY input; one Chromium counterexample at saturating Q8 values falsifies it permanently for full-range weights`; `killing_metric: adversarial-vector lane equality vs scalar reference on shipping engines`; `disposition: REJECT the relaxed-dot sub-lane of frankentts-kkuq; island/bundle/selftest scaffolding reverted before landing (crates/ftts-kernels int8.rs + selftest.rs restored to pre-session state); threading half of kkuq unaffected (CyanStork's m=1 codec GEMV stripe receipts stand)`; `do_not_retry: unless the WebAssembly relaxed-simd specification pins exact s8 x s8 semantics across engines for the full [-128,127]^2 operand range AND this same harness passes bit-exactness on stable Chrome/Safari/Firefox`; `tally_local_w_l_n: 0/1/0`.

Naming archaeology for whoever reopens this: the merged spec renamed the proposal-phase `i8x16.relaxed_dot_i8x16_add_i32x4` to `i32x4.dot_i8x16_i7x16_add_s` at FD 93 01 (sibling widen form `i16x8.dot_i8x16_i7x16_s` at FD 92 01); the "i7" refers to the signedness latitude that breaks us. wabt 1.0.41 predates both and mislabels the byte sequence (`i16x8.sub_sat_u` in disassembly) while still assembling surrounding sections correctly. The fingerprint sweep enumerates the whole FD 100..11F range so successor snapshots can be mapped mechanically.
