/* tslint:disable */
/* eslint-disable */

/**
 * Model bytes accumulated directly inside wasm linear memory, a slice at a time.
 *
 * # Why this exists
 *
 * Passing the artifact as a `Vec<u8>` makes wasm-bindgen copy it out of the JS `ArrayBuffer`
 * into linear memory, so a 1.3 GB model is **live twice** at the moment of construction. Desktop
 * Chrome absorbs 2.6 GB; an iPhone does not — the tab is reclaimed and the page "crashes while
 * loading". Streaming OPFS slices straight into a buffer that already lives in wasm keeps the JS
 * heap at one slice and the artifact at exactly one copy.
 *
 * Capacity is reserved once, exactly, up front. That is the load-bearing detail: a `Vec` that
 * grows by doubling would transiently hold 1.3 GB *plus* its 2.6 GB successor while copying —
 * worse than the problem being solved. `try_reserve_exact` also turns an allocation failure into
 * a thrown error naming the number of bytes, rather than the opaque `unreachable` a wasm abort
 * shows the user.
 * # Why the two files are staged in sequence rather than together
 *
 * The order is the whole optimization. Staging both raw files first and hydrating afterwards
 * means that at the moment the codec finishes widening, THREE allocations are live at once:
 *
 * ```text
 * artifact raw   1.31 GB   (staged, still untouched)
 * codec raw      0.68 GB   (source, not yet droppable)
 * codec f32     ~1.36 GB   (CodecCheckpoint owns Vec<f32> for every tensor)
 *                =======
 * peak          ~3.35 GB
 * ```
 *
 * Dropping the codec source after the fact does not help: the peak has already happened. iOS
 * Safari reclaims the tab somewhere below that, which is why the page died the instant it said
 * "hydrating" even though the download had just succeeded.
 *
 * Hydrating the codec BEFORE the artifact is staged removes the largest term from the peak
 * instead of from the aftermath:
 *
 * ```text
 * phase 1   codec raw 0.68 + codec f32 1.36  = 2.04 GB, then the source drops -> 1.36 GB
 * phase 2   codec f32 1.36 + artifact 1.31   = 2.67 GB
 * ```
 *
 * 2.67 GB against 3.35 GB, and nothing is ever read twice. The caller therefore drives:
 * `new(codec_bytes)` -> `push_codec` xN -> `finish_codec()` -> `reserve_fttsq(bytes)` ->
 * `push_fttsq` xN -> `WasmEngine::from_staging`. Each step refuses to run out of order.
 */
export class ModelStaging {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Bytes accepted so far, so the caller can drive a progress bar without tracking it twice.
     *
     * Counts the retired codec source rather than the live buffer, so the number keeps rising
     * across the phase boundary instead of collapsing when the source is freed.
     */
    filled(): number;
    /**
     * Hydrate the talker from the staged artifact, then RELEASE the artifact.
     *
     * The ordering step, and the reason this is a separate call rather than part of the final
     * assembly. Building the fused int8 tables is what makes the artifact's 0.69 GB hot prefix
     * dead — and wasm memory only ever grows, so the hole that release leaves is worth exactly
     * whatever gets allocated after it. Running this BEFORE the codec is staged lets the codec's
     * 0.46 GB of source and 0.457 GB of checkpoint land INSIDE that hole rather than on top of
     * it. With the codec first, the same work peaked at 2.10 GB.
     *
     * # Errors
     *
     * Throws when staging is incomplete, the prefix does not match the artifact's own declared
     * layout, or hydration fails — each named.
     */
    finish_artifact(): void;
    /**
     * Parse and widen the codec, then free its source bytes.
     *
     * This is the phase boundary. After it returns, the 0.68 GB of BF16 is gone and only the
     * widened checkpoint remains, so the artifact can be staged against a much lower floor.
     *
     * # Errors
     *
     * Throws when the staged bytes are short of what was reserved, or when the checkpoint does
     * not parse.
     *
     * Gated off unix for the same reason as the constructors below: `SafetensorsFile::from_bytes`
     * is the byte-oriented loader that exists only where there is no filesystem. Native hosts go
     * through the CLI's path-based loader instead, so this arm simply does not exist there — and
     * the workspace's `--all-targets` check compiles this crate natively, which is what caught it.
     */
    finish_codec(): void;
    /**
     * Reserve exact room for the codec checkpoint, the first file staged.
     *
     * The artifact is deliberately NOT reserved here — see the type-level note. Reserving it now
     * would put its 1.31 GB back into the codec-hydration peak and undo the whole point.
     *
     * # Errors
     *
     * Throws when linear memory cannot be reserved, naming the byte count that failed — the
     * honest signal on a device that simply does not have the memory.
     */
    constructor();
    /**
     * Hand over one codec tensor's little-endian f32 bytes, widened on arrival.
     *
     * There is no staged file and no reservation. The codec used to arrive as a 0.46 GB
     * safetensors buffer that a 0.457 GB checkpoint was then built out of — paying for the same
     * weights twice, in a heap that can never shrink. Tensor by tensor, the caller's chunk is
     * widened and released immediately, so the peak is the finished set plus one tensor.
     *
     * # Errors
     *
     * Throws once the codec is hydrated, or when the byte count is not a whole number of f32 —
     * a truncated push, which must not be rounded down into a silently short tensor.
     */
    push_codec_tensor(name: string, bytes: Uint8Array): void;
    /**
     * Append one slice of the artifact, in order.
     *
     * # Errors
     *
     * Throws if the slice would exceed the reserved capacity, or if no reservation was made.
     */
    push_fttsq(chunk: Uint8Array): void;
    /**
     * Reserve exact room for the artifact.
     *
     * # Why this no longer insists the codec goes first
     *
     * It used to, because the codec's source was the whole 0.68 GB file and holding it beside the
     * artifact restored a ~3.35 GB peak. Both halves of that changed. The caller now stages only
     * the codec's DECODER (~0.46 GB, since every tensor `CodecCheckpoint` reads is `decoder.*`),
     * and the artifact is now a hot prefix rather than the whole file — so holding both is
     * ~1.15 GB, below the peak the old ordering reached anyway.
     *
     * Ordering now matters for a different reason, and it points the other way. wasm memory can
     * only grow, so a freed buffer is not returned; it is a hole, and a hole is only worth having
     * where something later can land in it. The codec's source is freed the moment the checkpoint
     * is built, and the checkpoint is ~0.04 GB — so that 0.46 GB hole wants to be the LAST large
     * allocation, where the talker's 0.34 GB of widened tensors can reuse it. Reserving the
     * artifact first puts it there. Codec-first instead committed the hole before the artifact
     * ever grew, and the artifact could not fit in it, so the page paid for both.
     *
     * Measured: 1.64 GB committed codec-first, and the artifact streamed while already carrying
     * ~1.19 GB — which is the phase an iPhone died in.
     *
     * # Errors
     *
     * When the reservation fails.
     */
    reserve_fttsq(fttsq_bytes: number): void;
    /**
     * Reserve room for the artifact's HOT PREFIX only, leaving the cold text embedding on disk.
     *
     * The cold section is 622 MB — 47% of the artifact — and an utterance touches a few hundred
     * of its 4 KB rows. A native host pays for only those rows because it maps the file; wasm has
     * no mapping, and its heap is grow-only, so every staged byte is resident for the life of the
     * page. On a device with a ~2 GB per-tab ceiling that ballast is the difference between
     * running and being killed.
     *
     * `hot_bytes` must be exactly the artifact's cold-section offset. It is not taken on trust:
     * hydration re-derives it from the staged directory and refuses a mismatch, so a caller that
     * truncates at the wrong place gets a named error instead of silently wrong tensors.
     *
     * # Errors
     *
     * As [`ModelStaging::reserve_fttsq`].
     */
    reserve_fttsq_hot_prefix(hot_bytes: number, full_bytes: bigint): void;
}

/**
 * The loaded model: talker+microdecoder from the canonical artifact, codec from the raw
 * speech-tokenizer checkpoint, tokenizer from its three text files.
 */
export class WasmEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Enroll a voice from mono 24 kHz PCM in `[-1, 1]`; returns the 1,024-float x-vector.
     *
     * The reference is denoised first with the embedded FastEnhancer-S port — the same
     * automatic cleanup the CLI applies — so a laptop-mic recording enrolls without its
     * room hiss. Use [`WasmEngine::enroll_raw`] to skip cleanup.
     *
     * The speaker encoder hydrates lazily from the artifact on first use and is cached,
     * as is the denoiser.
     *
     * # Errors
     *
     * Throws when the PCM is too short for the mel front end or hydration fails.
     */
    enroll(pcm: Float32Array): Float32Array;
    /**
     * Enroll exactly the PCM given, with no noise cleanup.
     *
     * # Errors
     *
     * Throws when the PCM is too short for the mel front end or hydration fails.
     */
    enroll_raw(pcm: Float32Array): Float32Array;
    /**
     * Hydrate from bytes already resident in wasm memory, consuming the staging buffer.
     *
     * The streaming counterpart of [`WasmEngine::new`]: identical hydration, but the artifact is
     * moved rather than copied, so the peak is one copy instead of two.
     *
     * # Errors
     *
     * Throws when staging is incomplete, or with the failing hydration stage named.
     */
    static from_staging(staging: ModelStaging, vocab_json: string, merges_txt: string, tokenizer_config_json: string): WasmEngine;
    /**
     * Hydrate the engine from in-memory buffers.
     *
     * `fttsq` is the canonical quantized artifact (digest-verified here before any tensor is
     * read); `codec` is `speech_tokenizer/model.safetensors`; the three strings are the
     * tokenizer files, byte-for-byte as pulled.
     *
     * # Errors
     *
     * Throws with the failing stage named: artifact verification, codec parse, hydration, or
     * tokenizer construction.
     */
    constructor(fttsq: Uint8Array, codec: Uint8Array, vocab_json: string, merges_txt: string, tokenizer_config_json: string);
    /**
     * Synthesize `text` with a 1,024-float speaker vector; returns mono 24 kHz PCM in
     * `[-1, 1]`.
     *
     * The production sampling stack (seeded, deterministic per build+seed) drives both the
     * talker and the subtalker, exactly as the CLI's `say`. `max_frames` bounds runaway
     * generation; 0 selects the CLI's text-proportional backstop.
     *
     * # Errors
     *
     * Throws on an ill-shaped speaker vector, text preparation failure, or a decode error.
     */
    synthesize(text: string, speaker: Float32Array, seed: bigint, max_frames: number): Float32Array;
    /**
     * [`WasmEngine::synthesize`] for an engine whose artifact omits the cold text embedding.
     *
     * `rows_bf16` is one bf16 row per id from [`WasmEngine::text_row_ids`], in that order,
     * concatenated. Roughly 4 KB per distinct token, so a long utterance costs a couple of MB of
     * reads against 622 MB of permanently resident linear memory saved.
     *
     * # Errors
     *
     * As [`WasmEngine::synthesize`], plus a refusal when the rows do not match the ids.
     */
    synthesize_with_text_rows(text: string, speaker: Float32Array, seed: bigint, max_frames: number, rows_bf16: Uint8Array): Float32Array;
    /**
     * The cold-text-embedding ids this exact text will need, in the order the rows must arrive.
     *
     * The caller reads these rows out of the artifact itself (in the browser: out of OPFS) and
     * hands them to [`WasmEngine::synthesize_with_text_rows`]. Ids come from the engine rather
     * than from the caller's own tokenization on purpose — the two must agree exactly, and the
     * only way to guarantee that is to ask the tokenizer that will actually run.
     *
     * # Errors
     *
     * Throws when text preparation fails.
     */
    text_row_ids(text: string): Uint32Array;
}

/**
 * Sizes the team once the Workers that will serve it have confirmed they are parked.
 *
 * `partitions` counts the dispatcher too, so pass `readyWorkers + 1`. Anything <= 1 arms
 * nothing and the engine runs serially — the correct outcome on a browser without
 * `SharedArrayBuffer`, and on one where every Worker failed to start.
 */
export function arm_worker_team(partitions: number): void;

/**
 * Times the per-frame kernel schedule at the model's real shapes; returns a JSON report.
 *
 * This is the Spike-B RTF proxy: one talker step (28 layers x fused qkv/o/gate_up/down at
 * m=1) plus fifteen sequential microdecoder steps (5 layers each), all through the same
 * `linear_q8` the armed route dispatches. Codec cost is excluded (its dense fall-through is
 * BLAS-on-macOS, scalar here) — the report says so rather than pretending.
 */
export function bench_frame_kernels(rounds: number): string;

/**
 * Times one int8 GEMV at a real model reduction length and returns nanoseconds per dot.
 *
 * A kernel benchmark rather than an end-to-end one on purpose: it isolates the thing the SIMD
 * island changes, needs no 2 GB model, and runs in a second. `tier` takes the route names from
 * [`ftts_kernels::int8::Int8Tier::as_str`] so a caller can A/B `scalar` against `wasm-simd128`
 * in the same process — same allocator, same warm caches, same engine — which is the only way
 * the ratio means anything.
 *
 * Timing is the caller's job: wasm has no clock, so this returns after `rounds` passes and the
 * JS side divides by its own `performance.now()` delta.
 *
 * # Errors
 *
 * Throws when `tier` is not a route this build can execute.
 */
export function bench_int8_gemv(tier: string, k: number, n: number, rounds: number): number;

/**
 * Routes Rust panics to `console.error` with the real message and location — without this a
 * release-wasm panic surfaces as an opaque `RuntimeError: unreachable`.
 */
export function install_panic_hook(): void;

/**
 * Which int8 route this build actually dispatches in the browser.
 *
 * Exposed because the browser has no environment variables and no `robot backends`: without a
 * way to ask, a wasm build that silently fell back to the scalar loop would look exactly like
 * one running the SIMD128 island, and that difference is most of the frame time.
 */
export function int8_route(): string;

/**
 * The 1,024-float x-vector of a built-in voice.
 *
 * # Errors
 *
 * Throws when the name is not a built-in.
 */
export function preset_vector(name: string): Float32Array;

/**
 * Names and one-line characters of the built-in voices, as JSON.
 */
export function presets(): string;

/**
 * Publishes the control block Workers park on, before the team has a size.
 *
 * Call from the engine Worker — never the page's main thread, where `atomic.wait` traps. Each
 * Worker then instantiates this same module against the same `WebAssembly.Memory` and calls
 * [`worker_loop_entry`] with its index. Once they report parked, call [`arm_worker_team`] with
 * the count that actually started.
 */
export function publish_team_block(): void;

/**
 * The body a spawned Worker runs; never returns.
 *
 * # Errors
 *
 * Throws if called before [`arm_worker_team`] published the control block, which is a host
 * sequencing bug — parking on a block that does not exist would hang silently instead.
 */
export function worker_loop_entry(worker: number): void;

/**
 * How many partitions the int8 team is running with; 1 means serial.
 */
export function worker_team_width(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_modelstaging_free: (a: number, b: number) => void;
    readonly __wbg_wasmengine_free: (a: number, b: number) => void;
    readonly bench_frame_kernels: (a: number) => [number, number];
    readonly bench_int8_gemv: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly int8_route: () => [number, number];
    readonly modelstaging_filled: (a: number) => number;
    readonly modelstaging_finish_artifact: (a: number) => [number, number];
    readonly modelstaging_finish_codec: (a: number) => [number, number];
    readonly modelstaging_new: () => number;
    readonly modelstaging_push_codec_tensor: (a: number, b: number, c: number, d: number, e: number) => [number, number];
    readonly modelstaging_push_fttsq: (a: number, b: number, c: number) => [number, number];
    readonly modelstaging_reserve_fttsq: (a: number, b: number) => [number, number];
    readonly modelstaging_reserve_fttsq_hot_prefix: (a: number, b: number, c: bigint) => [number, number];
    readonly preset_vector: (a: number, b: number) => [number, number, number, number];
    readonly presets: () => [number, number];
    readonly wasmengine_enroll: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmengine_enroll_raw: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmengine_from_staging: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmengine_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number];
    readonly wasmengine_synthesize: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number) => [number, number, number, number];
    readonly wasmengine_synthesize_with_text_rows: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number, h: number, i: number) => [number, number, number, number];
    readonly wasmengine_text_row_ids: (a: number, b: number, c: number) => [number, number, number, number];
    readonly worker_loop_entry: (a: number) => [number, number];
    readonly worker_team_width: () => number;
    readonly arm_worker_team: (a: number) => void;
    readonly install_panic_hook: () => void;
    readonly publish_team_block: () => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
