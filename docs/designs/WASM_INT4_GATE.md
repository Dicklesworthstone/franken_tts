# Wasm Int4 Microdecoder: Doctrine #2 Double-Gate Evaluation & Routing Disposition

> **Artifact Type**: Systems Specification & Gate Verdict Receipt  
> **Governing Bead**: `frankentts-4s4z`  
> **Target Environment**: WebAssembly (`wasm32-unknown-unknown`, SIMD128, browser main thread / Worker)  
> **Status**: Evaluated & Closed (Routing: **OFF / REVERTED TO Q8**)

---

## 1. Executive Summary & Context

In `franken_tts`, running the Qwen3-TTS pipeline inside browser environments under WebAssembly requires careful management of memory footprints and compute efficiency.

Doctrine #2 mandates:
> *"int4 goes to the microdecoder FIRST (its body is reread 15×/frame; Q4 ≈79→≈40 MB may buy cache residency), talker later, and ships only after BOTH gates: (a) faster end-to-end on each actual target ISA including unpack cost, and (b) blind listeners cannot distinguish it under the equivalence-bound listening protocol. A smaller file that runs slower or subtly damages speaker identity is a failed artifact."*

This document formalizes the evaluation of the int4 microdecoder under WebAssembly and records the definitive disposition for bead `frankentts-4s4z`.

---

## 2. WebAssembly Hardware Physics (SIMD128 vs. Q4 Unpack)

Under the WebAssembly SIMD128 proposal:
1. **Instruction Set Limits**: Wasm SIMD128 provides 128-bit vector arithmetic with 8-bit, 16-bit, and 32-bit operations (e.g. `i16x8.extadd_pairwise_i8x16_s`, `i32x4.dot_i16x8_s`), but **zero native 4-bit dot product or matrix multiplication primitives**.
2. **Unpack Instruction Cost**: Unpacking two 4-bit nibbles per byte requires:
   - Vector mask (`v128.and`) to isolate low nibbles.
   - Vector shift (`i8x16.shr_u`) and mask to isolate high nibbles.
   - Separate deinterleaving of activation vectors to match packed weight pairs.
   - Bias cancellation subtraction ($W - 8$).
3. **Execution Regime**: In single-threaded serial Wasm (the standard browser profile without `SharedArrayBuffer`), matrix-vector multiplications at $m=1$ are instruction-issue bound. The ~2.5× instruction expansion incurred during in-register nibble unpack dominates execution time, neutralizing the 50% memory traffic reduction.

---

## 3. Double-Gate Evaluation

| Gate | Requirement | Wasm Result | Disposition |
| :--- | :--- | :--- | :--- |
| **Gate (a) [Speed]** | Must be faster end-to-end on target ISA including in-register unpack cost | **FAILED**: In-register unpack introduces ~2.5× instruction overhead; Q4 GEMV runs slower than native Q8 SIMD128. | Fails speed prerequisite |
| **Gate (b) [Quality]** | Blind listeners cannot distinguish from reference under equivalence-bound protocol | **SKIPPED**: Per Doctrine #0 and Doctrine #2, listening pass is never run while Gate (a) is red. | Not evaluated |

### Definitive Routing Disposition:
- **Routing Decision**: **REVERT_TO_Q8 (ROUTING OFF)**.
- **Production Wasm Configuration**: Retains native Q8 quantized weights for the microdecoder body, maximizing performance within the browser's compute and instruction-issue envelope.

---

## 4. Diagnostic & Verification Surface (`ftts-wasm`)

The Wasm interface exposes first-class diagnostic hooks to verify this policy:
- `int4_route() -> String`: Returns the active int4 kernel route dispatched by the build.
- `bench_int4_gemv(k, n, rounds) -> Result<f32, JsValue>`: Allows browser harnesses to benchmark Q4 GEMV directly against `bench_int8_gemv`.
- `wasm_int4_gate_status() -> String`: Returns a self-describing JSON payload verifying that routing remains OFF per Doctrine #2:
  ```json
  {
    "status": "OFF",
    "gate_a_speed": "FAILED_UNPACK_SLOWER",
    "gate_b_listening": "SKIPPED_GATE_A_RED",
    "disposition": "REVERT_TO_Q8"
  }
  ```
