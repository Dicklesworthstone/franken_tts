/* @ts-self-types="./ftts_wasm.d.ts" */

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        ModelStagingFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_modelstaging_free(ptr, 0);
    }
    /**
     * Bytes accepted so far, so the caller can drive a progress bar without tracking it twice.
     * @returns {number}
     */
    filled() {
        const ret = wasm.modelstaging_filled(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Reserve exact room for both files before any bytes arrive.
     *
     * # Errors
     *
     * Throws when linear memory cannot be reserved, naming the byte count that failed — the
     * honest signal on a device that simply does not have the memory.
     * @param {number} fttsq_bytes
     * @param {number} codec_bytes
     */
    constructor(fttsq_bytes, codec_bytes) {
        const ret = wasm.modelstaging_new(fttsq_bytes, codec_bytes);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        ModelStagingFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Append one slice of the codec checkpoint, in order.
     *
     * # Errors
     *
     * As [`ModelStaging::push_fttsq`].
     * @param {Uint8Array} chunk
     */
    push_codec(chunk) {
        const ptr0 = passArray8ToWasm0(chunk, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.modelstaging_push_codec(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
    /**
     * Append one slice of the artifact, in order.
     *
     * # Errors
     *
     * Throws if the slice would exceed the reserved capacity, which means the caller's manifest
     * and its download disagree — better caught here than as a corrupt tensor later.
     * @param {Uint8Array} chunk
     */
    push_fttsq(chunk) {
        const ptr0 = passArray8ToWasm0(chunk, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.modelstaging_push_fttsq(this.__wbg_ptr, ptr0, len0);
        if (ret[1]) {
            throw takeFromExternrefTable0(ret[0]);
        }
    }
}
if (Symbol.dispose) ModelStaging.prototype[Symbol.dispose] = ModelStaging.prototype.free;

/**
 * The loaded model: talker+microdecoder from the canonical artifact, codec from the raw
 * speech-tokenizer checkpoint, tokenizer from its three text files.
 */
export class WasmEngine {
    static __wrap(ptr) {
        const obj = Object.create(WasmEngine.prototype);
        obj.__wbg_ptr = ptr;
        WasmEngineFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmEngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmengine_free(ptr, 0);
    }
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
     * @param {Float32Array} pcm
     * @returns {Float32Array}
     */
    enroll(pcm) {
        const ptr0 = passArrayF32ToWasm0(pcm, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmengine_enroll(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Enroll exactly the PCM given, with no noise cleanup.
     *
     * # Errors
     *
     * Throws when the PCM is too short for the mel front end or hydration fails.
     * @param {Float32Array} pcm
     * @returns {Float32Array}
     */
    enroll_raw(pcm) {
        const ptr0 = passArrayF32ToWasm0(pcm, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.wasmengine_enroll_raw(this.__wbg_ptr, ptr0, len0);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v2;
    }
    /**
     * Hydrate from bytes already resident in wasm memory, consuming the staging buffer.
     *
     * The streaming counterpart of [`WasmEngine::new`]: identical hydration, but the artifact is
     * moved rather than copied, so the peak is one copy instead of two.
     *
     * # Errors
     *
     * Throws when staging is incomplete, or with the failing hydration stage named.
     * @param {ModelStaging} staging
     * @param {string} vocab_json
     * @param {string} merges_txt
     * @param {string} tokenizer_config_json
     * @returns {WasmEngine}
     */
    static from_staging(staging, vocab_json, merges_txt, tokenizer_config_json) {
        _assertClass(staging, ModelStaging);
        var ptr0 = staging.__destroy_into_raw();
        const ptr1 = passStringToWasm0(vocab_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(merges_txt, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(tokenizer_config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.wasmengine_from_staging(ptr0, ptr1, len1, ptr2, len2, ptr3, len3);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        return WasmEngine.__wrap(ret[0]);
    }
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
     * @param {Uint8Array} fttsq
     * @param {Uint8Array} codec
     * @param {string} vocab_json
     * @param {string} merges_txt
     * @param {string} tokenizer_config_json
     */
    constructor(fttsq, codec, vocab_json, merges_txt, tokenizer_config_json) {
        const ptr0 = passArray8ToWasm0(fttsq, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArray8ToWasm0(codec, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(vocab_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(merges_txt, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passStringToWasm0(tokenizer_config_json, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len4 = WASM_VECTOR_LEN;
        const ret = wasm.wasmengine_new(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4);
        if (ret[2]) {
            throw takeFromExternrefTable0(ret[1]);
        }
        this.__wbg_ptr = ret[0];
        WasmEngineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
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
     * @param {string} text
     * @param {Float32Array} speaker
     * @param {bigint} seed
     * @param {number} max_frames
     * @returns {Float32Array}
     */
    synthesize(text, speaker, seed, max_frames) {
        const ptr0 = passStringToWasm0(text, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passArrayF32ToWasm0(speaker, wasm.__wbindgen_malloc);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.wasmengine_synthesize(this.__wbg_ptr, ptr0, len0, ptr1, len1, seed, max_frames);
        if (ret[3]) {
            throw takeFromExternrefTable0(ret[2]);
        }
        var v3 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
        return v3;
    }
}
if (Symbol.dispose) WasmEngine.prototype[Symbol.dispose] = WasmEngine.prototype.free;

/**
 * Sizes the team once the Workers that will serve it have confirmed they are parked.
 *
 * `partitions` counts the dispatcher too, so pass `readyWorkers + 1`. Anything <= 1 arms
 * nothing and the engine runs serially — the correct outcome on a browser without
 * `SharedArrayBuffer`, and on one where every Worker failed to start.
 * @param {number} partitions
 */
export function arm_worker_team(partitions) {
    wasm.arm_worker_team(partitions);
}

/**
 * Times the per-frame kernel schedule at the model's real shapes; returns a JSON report.
 *
 * This is the Spike-B RTF proxy: one talker step (28 layers x fused qkv/o/gate_up/down at
 * m=1) plus fifteen sequential microdecoder steps (5 layers each), all through the same
 * `linear_q8` the armed route dispatches. Codec cost is excluded (its dense fall-through is
 * BLAS-on-macOS, scalar here) — the report says so rather than pretending.
 * @param {number} rounds
 * @returns {string}
 */
export function bench_frame_kernels(rounds) {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.bench_frame_kernels(rounds);
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

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
 * @param {string} tier
 * @param {number} k
 * @param {number} n
 * @param {number} rounds
 * @returns {number}
 */
export function bench_int8_gemv(tier, k, n, rounds) {
    const ptr0 = passStringToWasm0(tier, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.bench_int8_gemv(ptr0, len0, k, n, rounds);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return ret[0];
}

/**
 * Routes Rust panics to `console.error` with the real message and location — without this a
 * release-wasm panic surfaces as an opaque `RuntimeError: unreachable`.
 */
export function install_panic_hook() {
    wasm.install_panic_hook();
}

/**
 * Which int8 route this build actually dispatches in the browser.
 *
 * Exposed because the browser has no environment variables and no `robot backends`: without a
 * way to ask, a wasm build that silently fell back to the scalar loop would look exactly like
 * one running the SIMD128 island, and that difference is most of the frame time.
 * @returns {string}
 */
export function int8_route() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.int8_route();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * The 1,024-float x-vector of a built-in voice.
 *
 * # Errors
 *
 * Throws when the name is not a built-in.
 * @param {string} name
 * @returns {Float32Array}
 */
export function preset_vector(name) {
    const ptr0 = passStringToWasm0(name, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.preset_vector(ptr0, len0);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v2;
}

/**
 * Names and one-line characters of the built-in voices, as JSON.
 * @returns {string}
 */
export function presets() {
    let deferred1_0;
    let deferred1_1;
    try {
        const ret = wasm.presets();
        deferred1_0 = ret[0];
        deferred1_1 = ret[1];
        return getStringFromWasm0(ret[0], ret[1]);
    } finally {
        wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Publishes the control block Workers park on, before the team has a size.
 *
 * Call from the engine Worker — never the page's main thread, where `atomic.wait` traps. Each
 * Worker then instantiates this same module against the same `WebAssembly.Memory` and calls
 * [`worker_loop_entry`] with its index. Once they report parked, call [`arm_worker_team`] with
 * the count that actually started.
 */
export function publish_team_block() {
    wasm.publish_team_block();
}

/**
 * The body a spawned Worker runs; never returns.
 *
 * # Errors
 *
 * Throws if called before [`arm_worker_team`] published the control block, which is a host
 * sequencing bug — parking on a block that does not exist would hang silently instead.
 * @param {number} worker
 */
export function worker_loop_entry(worker) {
    const ret = wasm.worker_loop_entry(worker);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

/**
 * How many partitions the int8 team is running with; 1 means serial.
 * @returns {number}
 */
export function worker_team_width() {
    const ret = wasm.worker_team_width();
    return ret >>> 0;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_6d29952fd74a0b78: function(arg0, arg1) {
            console.error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_now_86c0d4ba3fa605b8: function() {
            const ret = Date.now();
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./ftts_wasm_bg.js": import0,
    };
}

const ModelStagingFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_modelstaging_free(ptr, 1));
const WasmEngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmengine_free(ptr, 1));

function _assertClass(instance, klass) {
    if (!(instance instanceof klass)) {
        throw new Error(`expected instance of ${klass.name}`);
    }
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArray8ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 1, 1) >>> 0;
    getUint8ArrayMemory0().set(arg, ptr / 1);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('ftts_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
