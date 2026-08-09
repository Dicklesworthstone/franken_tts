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
 */
export class ModelStaging {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Bytes accepted so far, so the caller can drive a progress bar without tracking it twice.
     */
    filled(): number;
    /**
     * Reserve exact room for both files before any bytes arrive.
     *
     * # Errors
     *
     * Throws when linear memory cannot be reserved, naming the byte count that failed — the
     * honest signal on a device that simply does not have the memory.
     */
    constructor(fttsq_bytes: number, codec_bytes: number);
    /**
     * Append one slice of the codec checkpoint, in order.
     *
     * # Errors
     *
     * As [`ModelStaging::push_fttsq`].
     */
    push_codec(chunk: Uint8Array): void;
    /**
     * Append one slice of the artifact, in order.
     *
     * # Errors
     *
     * Throws if the slice would exceed the reserved capacity, which means the caller's manifest
     * and its download disagree — better caught here than as a corrupt tensor later.
     */
    push_fttsq(chunk: Uint8Array): void;
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
     * The speaker encoder hydrates lazily from the artifact on first use and is cached.
     *
     * # Errors
     *
     * Throws when the PCM is too short for the mel front end or hydration fails.
     */
    enroll(pcm: Float32Array): Float32Array;
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
}

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

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_modelstaging_free: (a: number, b: number) => void;
    readonly __wbg_wasmengine_free: (a: number, b: number) => void;
    readonly bench_frame_kernels: (a: number) => [number, number];
    readonly bench_int8_gemv: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly int8_route: () => [number, number];
    readonly modelstaging_filled: (a: number) => number;
    readonly modelstaging_new: (a: number, b: number) => [number, number, number];
    readonly modelstaging_push_codec: (a: number, b: number, c: number) => [number, number];
    readonly modelstaging_push_fttsq: (a: number, b: number, c: number) => [number, number];
    readonly preset_vector: (a: number, b: number) => [number, number, number, number];
    readonly presets: () => [number, number];
    readonly wasmengine_enroll: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmengine_from_staging: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmengine_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => [number, number, number];
    readonly wasmengine_synthesize: (a: number, b: number, c: number, d: number, e: number, f: bigint, g: number) => [number, number, number, number];
    readonly install_panic_hook: () => void;
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
