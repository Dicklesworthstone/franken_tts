// Chunked, resumable model loader: /model/* (same-origin proxy) → OPFS → verified bytes.
//
// Each file downloads in CHUNK_BYTES ranges appended to an OPFS staging file, so an
// interrupted 1.3 GB pull resumes from the last complete chunk instead of restarting.
// A file only ever reaches the engine after its full SHA-256 matches the pinned digest;
// a mismatch deletes the staging file and starts that file over.

import { MODEL_FILES, TOTAL_BYTES, CHUNK_BYTES, ENDPOINT_BYTES } from "./model-manifest.js?v=@SITEV@";
import { Sha256, digestBlob, digestRange } from "./sha256.js?v=@SITEV@";

// A verified file records its identity here so it is never re-hashed. Hashing 2 GB in JS costs
// ~10 s on a fast desktop and far more on a phone, all of it on the main thread — long enough for
// iOS to kill the tab as unresponsive. Verification is a property of bytes on disk, not of a page
// load, so it is done ONCE (streamed alongside the download, while the bytes are already in hand)
// and remembered.
const VERIFIED_MARKER = "verified.json";

// Kill switch for the endpoint fast path (DISC-004): `?fullverify` forces the whole-file digest
// on every cached asset, restoring the pre-optimization behavior exactly. It is deliberately a URL
// parameter rather than a build flag so a user who suspects a corrupt cache can prove it in one
// reload, and so the slow path stays continuously exercised rather than rotting.
const FULL_VERIFY = new URLSearchParams(globalThis.location?.search ?? "").has("fullverify");

async function readVerified(root) {
  try {
    const handle = await root.getFileHandle(VERIFIED_MARKER);
    return JSON.parse(await (await handle.getFile()).text());
  } catch {
    return {};
  }
}

/// Hands the event loop back so a long verification cannot look like a hung page.
///
/// Without this the digest is one unbroken JS run — ~10 s of it for a 2 GB model on a fast
/// desktop, considerably more on a phone — and iOS reclaims tabs that stop answering.
function yieldToEventLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Prove a cached file is still the pinned artifact, cheaply.
 *
 * Exact byte length plus SHA-256 over the first and last ENDPOINT_BYTES. This reads 20 MB where a
 * full digest reads 1.3 GB, which is the difference between a page that loads instantly on a warm
 * cache and one that spends ~10 s hashing on every single visit before it can do anything.
 *
 * The length check is what makes the endpoints strong rather than decorative: a truncated or
 * still-being-written file is caught by size alone, and any resumed download that stopped
 * mid-stream has a short length. The endpoints then cover the two regions that actually differ
 * between artifacts — the safetensors/fttsq header at the front, and the last tensor's payload at
 * the back — so a stale manifest or a swapped file fails immediately.
 *
 * Falls back to the full digest for any file too small to have two disjoint endpoint windows, and
 * for any manifest entry that carries no endpoint hashes.
 */
async function endpointsMatch(blob, file) {
  if (blob.size !== file.bytes) return false;
  if (FULL_VERIFY || !file.head || !file.tail || file.bytes < 2 * ENDPOINT_BYTES) {
    return (await digestBlob(blob, 8 * 1024 * 1024, yieldToEventLoop)) === file.sha256;
  }
  if ((await digestRange(blob, 0, ENDPOINT_BYTES)) !== file.head) return false;
  return (await digestRange(blob, file.bytes - ENDPOINT_BYTES, file.bytes)) === file.tail;
}

async function writeVerified(root, ledger) {
  const handle = await root.getFileHandle(VERIFIED_MARKER, { create: true });
  const writable = await handle.createWritable();
  await writable.write(JSON.stringify(ledger));
  await writable.close();
}

async function opfsRoot() {
  return navigator.storage.getDirectory();
}

// Verification reads the file in slices rather than through crypto.subtle, which needs one
// contiguous buffer: hashing a 1.3 GB asset that way allocates 1.3 GB in the JS heap and is what
// killed the tab on iOS before the download could finish. Peak memory here is one slice.

async function fetchRange(asset, start, endInclusive) {
  const response = await fetch(`model/${asset}`, {
    headers: { Range: `bytes=${start}-${endInclusive}` },
  });
  if (!(response.status === 206 || (response.status === 200 && start === 0))) {
    throw new Error(`range fetch for ${asset} failed: HTTP ${response.status}`);
  }
  const window = endInclusive - start + 1;
  if (response.status === 200) {
    // The server ignored the Range header. Buffering the reply would materialize the
    // ENTIRE asset in one JS ArrayBuffer — the exact iOS OOM this chunked loader exists
    // to avoid — so a range-blind server is only acceptable when the window IS the file.
    const total = Number(response.headers.get("Content-Length") ?? NaN);
    if (Number.isFinite(total) && total > window) {
      throw new Error(
        `range fetch for ${asset}: server ignored Range and offered ${total} bytes ` +
          `for a ${window}-byte window`,
      );
    }
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength > window) {
    throw new Error(
      `range fetch for ${asset}: got ${bytes.byteLength} bytes for a ${window}-byte window`,
    );
  }
  return bytes;
}

/**
 * Ensure every model file is present and digest-verified in OPFS.
 * `onProgress({assetsDone, assetsTotal, bytesDone, bytesTotal, phase, asset})`.
 * Returns { fttsq: {asset, bytes}, codec: {asset, bytes}, vocab: string, ... }.
 *
 * The two large files are deliberately NOT returned as buffers. Reading a 1.3 GB artifact into an
 * ArrayBuffer here, then handing it to wasm-bindgen, puts it in memory twice — which is what
 * reclaims the tab on iOS. They stay in OPFS and the worker streams them into wasm linear memory
 * a slice at a time; only the small tokenizer files decode to strings.
 */
export async function ensureModel(onProgress) {
  const root = await opfsRoot();
  const verified = await readVerified(root);
  const out = {};
  let bytesDone = 0;
  let assetsDone = 0;
  const report = (phase, asset, extra = 0) =>
    onProgress?.({
      assetsDone,
      assetsTotal: MODEL_FILES.length,
      bytesDone: bytesDone + extra,
      bytesTotal: TOTAL_BYTES,
      phase,
      asset,
    });

  for (const file of MODEL_FILES) {
    // Fast path: a previously verified copy.
    try {
      const done = await root.getFileHandle(file.asset);
      const blob = await done.getFile();
      if (blob.size === file.bytes) {
        // Two independent reasons to trust a cached file, cheapest first.
        //
        // `verified.json` is the memo from a previous visit and costs one small read. The endpoint
        // check is the fallback, and is what makes a cold cache — a first visit after a download in
        // an earlier session, or a marker that never landed — fast instead of a ~10 s stall. Before
        // endpoints existed, that fallback was a full 1.3 GB rehash on the main thread, which is
        // exactly the "incredibly slow even on desktop" case.
        const noted = verified[file.asset];
        const trusted = noted && noted.bytes === file.bytes && noted.sha256 === file.sha256;
        if (!trusted) report("verifying", file.asset);
        if (trusted || (await endpointsMatch(blob, file))) {
          if (!trusted) {
            verified[file.asset] = { bytes: file.bytes, sha256: file.sha256 };
            await writeVerified(root, verified);
          }
          out[file.key] = file.text
            ? new TextDecoder().decode(await blob.arrayBuffer())
            : { asset: file.asset, bytes: file.bytes };
          bytesDone += file.bytes;
          assetsDone += 1;
          report("cached", file.asset);
          continue;
        }
        // Full-length but the bytes are wrong: corrupt — clear it for a clean redownload.
        await root.removeEntry(file.asset);
      } else if (blob.size > file.bytes) {
        // Longer than the manifest says: a stale file from an older manifest — clear it.
        await root.removeEntry(file.asset);
      }
      // A SHORTER file is an interrupted download and must survive to the resume path
      // below (`offset = size`). Deleting it here was the bug that made "the download
      // resumes if interrupted" false: every partial restarted from zero.
    } catch {
      /* not cached yet */
    }

    // Download straight into the final name, through ONE writable held open for the whole file.
    //
    // Both details are WebKit survival, not tidiness. `createWritable` is implemented with a swap
    // file, and `keepExistingData: true` copies the *entire* existing file into it on open — so
    // opening once per chunk meant ~80 full copies of a 1.3 GB artifact, which is what killed the
    // tab on iOS during download. Staging to `.part` and promoting afterwards then copied the
    // whole file a second time. One open, positional writes, no promotion.
    //
    // Resuming stays safe because completeness is decided by size AND digest, never by the file
    // merely existing: a short file resumes from its own length, and a full-length file with the
    // wrong bytes fails verification below and is deleted.
    const handle = await root.getFileHandle(file.asset, { create: true });
    let freshDigest = null;
    let offset = (await handle.getFile()).size;
    if (offset > file.bytes) offset = 0; // manifest changed under a stale file
    if (offset < file.bytes) {
      // A sync access handle (Workers only) writes without any swap file at all; the writable is
      // the main-thread fallback. Preferring the former keeps this correct if the loader ever
      // moves into the engine Worker.
      const sync = handle.createSyncAccessHandle
        ? await handle.createSyncAccessHandle().catch(() => null)
        : null;
      const writable = sync ? null : await handle.createWritable({ keepExistingData: true });
      // Hash as the bytes arrive. A fresh download therefore needs NO verification pass at all —
      // the second full read of a 1.3 GB file was pure waste, and on a phone it was fatal. A
      // resume cannot do this (the hash state for the existing prefix is gone), so it falls back
      // to reading the finished file once, with yields.
      const live = offset === 0 ? new Sha256() : null;
      try {
        while (offset < file.bytes) {
          const end = Math.min(offset + CHUNK_BYTES, file.bytes) - 1;
          report("downloading", file.asset, offset);
          const chunk = await fetchRange(file.asset, offset, end);
          if (sync) {
            sync.write(chunk, { at: offset });
          } else {
            await writable.write({ type: "write", position: offset, data: chunk });
          }
          live?.update(chunk);
          offset += chunk.byteLength;
        }
      } finally {
        if (sync) {
          sync.flush();
          sync.close();
        } else {
          await writable.close();
        }
      }
      // The live hash is only valid for a download that started at zero; a resume leaves it null
      // and falls through to the one-off read below.
      if (live) freshDigest = live.hex();
    }

    const written = await handle.getFile();
    if (!freshDigest) {
      report("verifying", file.asset, file.bytes);
      freshDigest = await digestBlob(written, 8 * 1024 * 1024, yieldToEventLoop);
    }
    if (written.size !== file.bytes || freshDigest !== file.sha256) {
      await root.removeEntry(file.asset);
      delete verified[file.asset];
      await writeVerified(root, verified);
      throw new Error(`${file.asset}: digest mismatch after download; cleared for retry`);
    }
    verified[file.asset] = { bytes: file.bytes, sha256: file.sha256 };
    await writeVerified(root, verified);

    out[file.key] = file.text
      ? new TextDecoder().decode(await (await (await root.getFileHandle(file.asset)).getFile()).arrayBuffer())
      : { asset: file.asset, bytes: file.bytes };
    bytesDone += file.bytes;
    assetsDone += 1;
    report("done", file.asset);
  }

  return out;
}

/** Bytes currently cached (verified finals only), for the "storage used" line. */
export async function cachedBytes() {
  const root = await opfsRoot();
  let total = 0;
  for (const file of MODEL_FILES) {
    try {
      total += (await (await root.getFileHandle(file.asset)).getFile()).size;
    } catch {
      /* absent */
    }
  }
  return total;
}

/** Remove every cached model file (user-facing "free up storage"). */
export async function clearCache() {
  const root = await opfsRoot();
  // The trust ledger goes too: a cleared cache must not leave behind a memo claiming files
  // were verified, or a future file planted at the same name and size would inherit trust
  // it never earned.
  for (const name of [
    VERIFIED_MARKER,
    ...MODEL_FILES.flatMap((file) => [file.asset, `${file.asset}.part`]),
  ]) {
    try {
      await root.removeEntry(name);
    } catch {
      /* absent */
    }
  }
}
