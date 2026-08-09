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

### NE-003 wasm-simd128-is-a-1.8x-lever-not-a-4x-one

`claim_id: wasm-int8-simd128-ceiling`; `evidence_id: scratchpad/bench.mjs — node 25, in-process interleaved ABBA, 7 pairs per shape, 40 rounds per timing, both routes same process/allocator/cache state`; `status: NEGATIVE (against the predicted magnitude; the tier itself is a KEEP)`; `model_source_commit: 5d83992436eae1d760afd27aff78a71d676296fc (weights not consumed; synthetic operands at pinned census shapes)`; `fixture_sha256: not-applicable`; `cpu_features: Apple M4 Pro, wasm32 + simd128 under V8`; `command_env: wasm-pack build --target nodejs --release with RUSTFLAGS=-C target-feature=+simd128; node bench.mjs`; `kill_switch: Int8Tier::Scalar remains dispatchable`; `before_after: scalar -> wasm-simd128, per-dot: k=1024/n=1024 299.9 -> 205.4 ns (1.49x); k=3072/n=1024 1026.6 -> 599.8 ns (1.71x); k=1024/n=3072 300.4 -> 204.2 ns (1.47x)`; `equivalence: exact i32 equality with Scalar by construction (every i8xi8 product fits i16, every pairwise sum fits i32, integer addition is associative)`; `killing_metric: per-dot wall time at real census shapes`; `disposition: KEEP the tier, REJECT the 4-8x estimate that motivated it`; `do_not_retry: do not budget more than ~2x for any SIMD128 int8 kernel on this instruction set — wasm has no int8 dot product, so the best available shape is 4 widenings + 2 i32x4.dot_i16x8_s + 2 adds per 16 bytes = 2.0 MACs/op against NEON SDOT's 16 MACs/instruction; the arithmetic ceiling with the activation widening hoisted is 2.67 MACs/op`; `tally_local_w_l_n: 0/1/0`.

### NE-004 register-blocking-does-not-help-the-wasm-int8-gemv

`claim_id: wasm-int8-4column-register-blocking`; `evidence_id: same harness as NE-003, immediately before/after the blocked kernel landed`; `status: NEGATIVE (neutral)`; `cpu_features: Apple M4 Pro, wasm32 + simd128 under V8`; `kill_switch: the blocked path is behind cfg(target_arch = "wasm32") and matches only Int8Tier::WasmSimd128`; `before_after: 1.49 -> 1.53x, 1.71 -> 1.79x, 1.47 -> 1.53x — about 3%, against a predicted 23% from the op-count arithmetic (26 ops per 64 MACs instead of 32)`; `equivalence: exact i32 equality preserved; only the loop nest changed`; `killing_metric: per-dot wall time at real census shapes`; `disposition: KEEP (consistent across all three shapes, no numerics change, no measurable cost) but do NOT treat it as a lever`; `do_not_retry: the doctrine's "the instruction is not the lever; the blocking is" is an *instruction-bound* heuristic and does not apply here. Hoisting the shared activation widening bought nothing, which says the wasm GEMV is not instruction-issue-bound: measured ~0.2 ns per int8 element (~5 GB/s) where SIMD128 issue rates predict several times that. Further kernel micro-optimization is exhausted; the remaining levers are parallelism (threads) and *traffic reduction* (int4 weights, the seq-16 microdecoder batching that removes a 15x reread), not instruction selection`; `tally_local_w_l_n: 0/0/1`.

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
