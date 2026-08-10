// Download-resume contract test for site/loader.js, in plain Node.
//
// Run:  node site/harness/resume.test.mjs
//
// This exists because the resume path shipped with zero coverage, and the one bug that
// mattered — a `removeEntry` at the wrong nesting depth deleting every partial download —
// was invisible to both the browser harness (which always downloads clean) and the Rust
// suites. The loader's collaborators (OPFS, fetch) are mocked; the loader itself, the
// manifest wiring, the range arithmetic, and the real streaming SHA-256 all run for real.
//
// Scenarios:
//   A. a shorter-than-manifest partial survives and resumes from its own length
//   B. a full-length file with wrong bytes is cleared and redownloaded
//   C. a longer-than-manifest stale file is cleared and redownloaded
//   D. a corrupted download fails the digest gate, is cleared, and the error names it

import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { registerHooks } from "node:module";

// ---------------------------------------------------------------- test manifest fixture
//
// Deterministic content, real digests. CHUNK_BYTES is tiny so one file spans many ranges.
const CONTENT = Uint8Array.from({ length: 1000 }, (_, i) => (i * 31 + 7) & 0xff);
const sha256hex = (bytes) => createHash("sha256").update(bytes).digest("hex");

const FIXTURE = {
  CHUNK_BYTES: 64,
  ENDPOINT_BYTES: 16,
  MODEL_FILES: [
    {
      key: "fttsq",
      asset: "test.bin",
      bytes: CONTENT.length,
      sha256: sha256hex(CONTENT),
      head: sha256hex(CONTENT.subarray(0, 16)),
      tail: sha256hex(CONTENT.subarray(CONTENT.length - 16)),
      text: false,
    },
  ],
};
FIXTURE.TOTAL_BYTES = CONTENT.length;

const fixtureUrl = `data:text/javascript,${encodeURIComponent(
  `const F = ${JSON.stringify(FIXTURE)};` +
    "export const MODEL_FILES = F.MODEL_FILES;" +
    "export const TOTAL_BYTES = F.TOTAL_BYTES;" +
    "export const CHUNK_BYTES = F.CHUNK_BYTES;" +
    "export const ENDPOINT_BYTES = F.ENDPOINT_BYTES;",
)}`;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith("./model-manifest.js")) {
      return { url: fixtureUrl, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

// ------------------------------------------------------------------------- OPFS mock
//
// Just enough of the OPFS surface for loader.js: named byte buffers, positional writable
// streams (`keepExistingData` honored), and removal. No sync access handles, so the loader
// exercises its main-thread `createWritable` path.
function makeBlob(bytes) {
  return {
    size: bytes.length,
    async arrayBuffer() {
      return bytes.slice().buffer;
    },
    async text() {
      return new TextDecoder().decode(bytes);
    },
    slice(start, end) {
      return makeBlob(bytes.subarray(start, end));
    },
  };
}

function makeStore() {
  const files = new Map(); // name -> Uint8Array
  const removed = [];
  const root = {
    async getFileHandle(name, options = {}) {
      if (!files.has(name)) {
        if (!options.create) throw new Error(`NotFound: ${name}`);
        files.set(name, new Uint8Array(0));
      }
      return {
        async getFile() {
          return makeBlob(files.get(name));
        },
        async createWritable({ keepExistingData = false } = {}) {
          let buffer = keepExistingData ? files.get(name).slice() : new Uint8Array(0);
          return {
            async write(payload) {
              // loader.js uses {type:"write", position, data}; writeVerified uses a string.
              if (typeof payload === "string") {
                buffer = new TextEncoder().encode(payload);
                return;
              }
              const { position, data } = payload;
              const incoming = new Uint8Array(
                data instanceof ArrayBuffer ? new Uint8Array(data) : data,
              );
              if (position + incoming.length > buffer.length) {
                const grown = new Uint8Array(position + incoming.length);
                grown.set(buffer);
                buffer = grown;
              }
              buffer.set(incoming, position);
            },
            async close() {
              files.set(name, buffer);
            },
          };
        },
      };
    },
    async removeEntry(name) {
      if (!files.has(name)) throw new Error(`NotFound: ${name}`);
      files.delete(name);
      removed.push(name);
    },
  };
  return { root, files, removed };
}

// ---------------------------------------------------------------------- fetch mock
function installFetch(content, log) {
  globalThis.fetch = async (url, { headers = {} } = {}) => {
    const range = headers.Range ?? headers.range;
    assert.ok(range, `every model fetch must be a range request (got none for ${url})`);
    const match = /bytes=(\d+)-(\d+)/.exec(range);
    assert.ok(match, `unparseable Range: ${range}`);
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), content.length - 1);
    log.push([start, end]);
    const body = content.subarray(start, end + 1);
    return {
      status: 206,
      headers: { get: () => String(body.length) },
      async arrayBuffer() {
        return body.slice().buffer;
      },
    };
  };
}

// -------------------------------------------------------------------------- scenarios
let store;
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: { storage: { getDirectory: async () => store.root } },
});

const { ensureModel } = await import("../loader.js?v=resume-test");
const quiet = () => {};

// A: a partial survives and resumes from its own length.
{
  store = makeStore();
  const ranges = [];
  installFetch(CONTENT, ranges);
  store.files.set("test.bin", CONTENT.slice(0, 300));
  const out = await ensureModel(quiet);
  assert.deepEqual(store.removed, [], "A: the partial must never be deleted");
  assert.equal(ranges[0][0], 300, `A: first range must start at the partial's length, got ${ranges[0][0]}`);
  assert.deepEqual(store.files.get("test.bin"), CONTENT, "A: resumed file must be complete");
  assert.equal(out.fttsq.bytes, CONTENT.length, "A: loader must report the file");
  console.log("PASS  A: partial download resumes from its own length");
}

// B: full-length wrong bytes → cleared and redownloaded.
{
  store = makeStore();
  const ranges = [];
  installFetch(CONTENT, ranges);
  // Corrupt inside the HEAD endpoint window: the cached-file fast path verifies endpoints
  // only (DISC-004's documented tradeoff — a middle flip is invisible to it by design), so
  // this is the corruption class the fast path promises to catch.
  const wrong = CONTENT.slice();
  wrong[5] ^= 0xff;
  store.files.set("test.bin", wrong);
  await ensureModel(quiet);
  assert.deepEqual(store.removed, ["test.bin"], "B: the corrupt full-length file must be cleared");
  assert.equal(ranges[0][0], 0, "B: the redownload must start from zero");
  assert.deepEqual(store.files.get("test.bin"), CONTENT, "B: redownloaded file must be correct");
  console.log("PASS  B: corrupt full-length file is cleared and redownloaded");
}

// C: longer than the manifest (stale) → cleared and redownloaded.
{
  store = makeStore();
  const ranges = [];
  installFetch(CONTENT, ranges);
  const stale = new Uint8Array(1200);
  stale.set(CONTENT);
  store.files.set("test.bin", stale);
  await ensureModel(quiet);
  assert.deepEqual(store.removed, ["test.bin"], "C: the stale longer file must be cleared");
  assert.deepEqual(store.files.get("test.bin"), CONTENT, "C: redownloaded file must be correct");
  console.log("PASS  C: stale longer-than-manifest file is cleared and redownloaded");
}

// D: a corrupting transport fails the digest gate loudly and clears the file.
{
  store = makeStore();
  const ranges = [];
  const corrupted = CONTENT.slice();
  corrupted[123] ^= 0xff;
  installFetch(corrupted, ranges);
  await assert.rejects(
    () => ensureModel(quiet),
    /digest mismatch/,
    "D: a wrong digest must reject",
  );
  assert.ok(store.removed.includes("test.bin"), "D: the failed download must be cleared for retry");
  console.log("PASS  D: digest gate clears a corrupted download and reports it");
}

console.log("\nall 4 resume scenarios passed");
