// Conformance test for the site's SHA-256 core and its range helper.
//
// Run: node site/sha256.test.js   (exit 0 = pass, 1 = fail)
//
// The JS core exists because crypto.subtle needs one contiguous buffer, and a 1.3 GB ArrayBuffer
// is what reclaimed the tab on iOS. That makes WebCrypto the natural oracle: it is the same
// FIPS 180-4 function, implemented independently, and available in Node. Every assertion below is
// JS-core output versus crypto.subtle output over the same bytes.
//
// The lengths are chosen to sit on the structural boundaries of the algorithm, where a padding or
// length-encoding bug hides: empty, sub-block, exactly one block, one-block-plus-one, the 55/56/57
// boundary where the 8-byte length field forces an extra block, and multi-block runs. The split
// points then prove the INCREMENTAL path — update() called across arbitrary chunk boundaries must
// equal one-shot, which is the property the streaming download hash depends on.

import { Sha256, digestBlob, digestRange } from "./sha256.js";

const subtle = globalThis.crypto.subtle;

const hex = (buffer) =>
  [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, "0")).join("");

const oracle = async (bytes) => hex(await subtle.digest("SHA-256", bytes));

/// Deterministic pseudo-random bytes: a fixed LCG, so a failure is reproducible by length alone.
function pattern(length) {
  const out = new Uint8Array(length);
  let state = 0x2545f491;
  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    out[i] = (state >>> 24) & 0xff;
  }
  return out;
}

const LENGTHS = [0, 1, 3, 55, 56, 57, 63, 64, 65, 119, 127, 128, 129, 1000, 4096, 65_536, 100_003];
const SPLITS = [1, 2, 3, 7, 16, 64, 1000];

let failures = 0;
const check = (name, actual, expected) => {
  if (actual !== expected) {
    failures += 1;
    console.error(`FAIL ${name}\n  expected ${expected}\n  actual   ${actual}`);
  }
};

for (const length of LENGTHS) {
  const bytes = pattern(length);
  const expected = await oracle(bytes);

  const oneShot = new Sha256();
  oneShot.update(bytes);
  check(`one-shot len=${length}`, oneShot.hex(), expected);

  // Incremental: the same bytes fed through update() in N pieces must land on the same digest.
  // This is the property the download path relies on — chunks arrive at 32 MiB range boundaries
  // that have nothing to do with SHA-256's 64-byte blocks.
  for (const parts of SPLITS) {
    const incremental = new Sha256();
    const step = Math.ceil(length / parts) || 1;
    for (let offset = 0; offset < length; offset += step) {
      incremental.update(bytes.subarray(offset, Math.min(offset + step, length)));
    }
    check(`incremental len=${length} parts=${parts}`, incremental.hex(), expected);
  }

  // digestBlob slices the Blob itself, so its slice size is a third, independent boundary.
  const blob = new Blob([bytes]);
  check(`digestBlob len=${length}`, await digestBlob(blob, 97), expected);
}

// digestRange is what the warm-cache endpoint check runs (DISC-004). It prefers crypto.subtle, so
// these cases prove the RANGE ARITHMETIC — that head and tail windows select the bytes the
// manifest's `head`/`tail` digests were computed over, and that the JS fallback agrees when
// crypto.subtle is unavailable (insecure origins).
{
  const bytes = pattern(100_003);
  const blob = new Blob([bytes]);
  const window = 4096;

  check("digestRange head", await digestRange(blob, 0, window), await oracle(bytes.subarray(0, window)));
  check(
    "digestRange tail",
    await digestRange(blob, bytes.length - window, bytes.length),
    await oracle(bytes.subarray(bytes.length - window)),
  );
  check("digestRange whole", await digestRange(blob, 0, bytes.length), await oracle(bytes));

  // Force the no-WebCrypto fallback and demand the identical answer. Without this, an insecure
  // origin would silently take an untested path.
  const saved = globalThis.crypto;
  try {
    Object.defineProperty(globalThis, "crypto", { value: undefined, configurable: true });
    check(
      "digestRange head (js fallback)",
      await digestRange(blob, 0, window),
      await oracle(bytes.subarray(0, window)),
    );
  } finally {
    Object.defineProperty(globalThis, "crypto", { value: saved, configurable: true });
  }
}

// A known-answer test, so the whole suite cannot pass by both sides being wrong in the same way.
{
  const abc = new TextEncoder().encode("abc");
  const known = new Sha256();
  known.update(abc);
  check(
    "known-answer abc",
    known.hex(),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
}

if (failures > 0) {
  console.error(`${failures} failure(s)`);
  process.exit(1);
}
console.log(`sha256.test.js: OK (${LENGTHS.length} lengths x ${SPLITS.length} splits + range cases)`);
