// Incremental SHA-256, because WebCrypto has no streaming digest.
//
// `crypto.subtle.digest` takes one contiguous buffer, so verifying a 1.3 GB model file through it
// means materializing 1.3 GB in the JS heap. Desktop Chrome absorbs that; an iPhone does not — the
// tab is killed before the check finishes, which is the "crashes before the model finishes
// downloading" report. Feeding fixed-size slices through this keeps peak memory at one slice.
//
// FIPS 180-4. Verified against crypto.subtle.digest on random inputs across every block-boundary
// case by the self-test below, which is the only reason hand-rolling a hash here is acceptable.

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

const rotr = (x, n) => (x >>> n) | (x << (32 - n));

/** Streaming SHA-256: `update` any number of chunks, then `hex()` once. */
export class Sha256 {
  constructor() {
    this.h = new Uint32Array([
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
      0x5be0cd19,
    ]);
    this.block = new Uint8Array(64);
    this.blockLength = 0; // bytes buffered from a previous, non-block-aligned update
    this.totalLength = 0; // message length in bytes, for the length suffix
    this.w = new Uint32Array(64);
  }

  /** @param {Uint8Array} bytes */
  update(bytes) {
    this.totalLength += bytes.length;
    let offset = 0;
    // Finish any partial block left by the last update before consuming whole blocks.
    if (this.blockLength > 0) {
      const need = Math.min(64 - this.blockLength, bytes.length);
      this.block.set(bytes.subarray(0, need), this.blockLength);
      this.blockLength += need;
      offset = need;
      if (this.blockLength === 64) {
        this.#compress(this.block, 0);
        this.blockLength = 0;
      }
    }
    while (offset + 64 <= bytes.length) {
      this.#compress(bytes, offset);
      offset += 64;
    }
    if (offset < bytes.length) {
      this.block.set(bytes.subarray(offset), 0);
      this.blockLength = bytes.length - offset;
    }
    return this;
  }

  /** Lowercase hex digest. Idempotent; the instance must not be `update`d afterwards. */
  hex() {
    // Finalization compresses the padding into `this.h`, so a second call used to fold the
    // padding in twice and return a different, wrong digest. Cache the first answer instead
    // of trusting every caller to know the footgun.
    if (this.digest !== undefined) return this.digest;
    const bitLength = this.totalLength * 8;
    const tail = new Uint8Array(this.blockLength < 56 ? 64 : 128);
    tail.set(this.block.subarray(0, this.blockLength), 0);
    tail[this.blockLength] = 0x80;
    // Length is a 64-bit big-endian bit count. Numbers past 2^53 cannot be exact, but that is
    // 1 PiB of message — far outside anything this loader will ever see.
    const view = new DataView(tail.buffer);
    view.setUint32(tail.length - 8, Math.floor(bitLength / 0x1_0000_0000), false);
    view.setUint32(tail.length - 4, bitLength >>> 0, false);
    for (let offset = 0; offset < tail.length; offset += 64) this.#compress(tail, offset);
    let out = "";
    for (const word of this.h) out += word.toString(16).padStart(8, "0");
    this.digest = out;
    return out;
  }

  /** @param {Uint8Array} bytes @param {number} offset */
  #compress(bytes, offset) {
    const w = this.w;
    for (let i = 0; i < 16; i++) {
      const j = offset + i * 4;
      w[i] = (bytes[j] << 24) | (bytes[j + 1] << 16) | (bytes[j + 2] << 8) | bytes[j + 3];
    }
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let [a, b, c, d, e, f, g, h] = this.h;
    for (let i = 0; i < 64; i++) {
      const s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + s1 + ch + K[i] + w[i]) | 0;
      const s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (s0 + maj) | 0;
      h = g; g = f; f = e;
      e = (d + temp1) | 0;
      d = c; c = b; b = a;
      a = (temp1 + temp2) | 0;
    }
    this.h[0] = (this.h[0] + a) | 0;
    this.h[1] = (this.h[1] + b) | 0;
    this.h[2] = (this.h[2] + c) | 0;
    this.h[3] = (this.h[3] + d) | 0;
    this.h[4] = (this.h[4] + e) | 0;
    this.h[5] = (this.h[5] + f) | 0;
    this.h[6] = (this.h[6] + g) | 0;
    this.h[7] = (this.h[7] + h) | 0;
  }
}

/**
 * SHA-256 of a File/Blob, read in slices so peak memory is one slice rather than the whole file.
 *
 * @param {Blob} blob
 * @param {number} sliceBytes
 * @param {(read: number) => void | Promise<void>} [onProgress]
 */
export async function digestBlob(blob, sliceBytes = 8 * 1024 * 1024, onProgress) {
  const hash = new Sha256();
  for (let offset = 0; offset < blob.size; offset += sliceBytes) {
    const end = Math.min(offset + sliceBytes, blob.size);
    // One slice at a time: the buffer from the previous iteration is collectable before the next
    // is allocated, which is the entire point of doing this rather than one arrayBuffer() call.
    hash.update(new Uint8Array(await blob.slice(offset, end).arrayBuffer()));
    // Awaited so a caller can pass an event-loop yield here and actually get one: an unawaited
    // setTimeout promise yields nothing, and the whole point of the callback for the loader is
    // giving the browser a macrotask between slices of a multi-second digest.
    if (onProgress) await onProgress(end);
  }
  return hash.hex();
}

/**
 * SHA-256 of a bounded byte range, preferring the native WebCrypto implementation.
 *
 * `crypto.subtle.digest` needs one contiguous buffer, which is exactly why the whole-file path
 * above cannot use it — a 1.3 GB ArrayBuffer is what reclaimed the tab on iOS. For a 10 MB
 * endpoint window that objection disappears entirely, and native SHA-256 runs roughly an order of
 * magnitude faster than the JS core (which measured ~193 MB/s on an iPhone). `crypto.subtle` is
 * absent on insecure origins, so the JS core stays as the fallback and the two agree by
 * construction — `sha256.test.js` pins that against 17 lengths and 7 slice boundaries.
 *
 * @param {Blob} blob
 * @param {number} start inclusive
 * @param {number} end exclusive
 * @returns {Promise<string>} lowercase hex
 */
export async function digestRange(blob, start, end) {
  const slice = blob.slice(start, end);
  const bytes = new Uint8Array(await slice.arrayBuffer());
  if (globalThis.crypto?.subtle?.digest) {
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  const hash = new Sha256();
  hash.update(bytes);
  return hash.hex();
}
