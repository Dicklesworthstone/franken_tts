// Chunked, resumable model loader: /model/* (same-origin proxy) → OPFS → verified bytes.
//
// Each file downloads in CHUNK_BYTES ranges appended to an OPFS staging file, so an
// interrupted 1.3 GB pull resumes from the last complete chunk instead of restarting.
// A file only ever reaches the engine after its full SHA-256 matches the pinned digest;
// a mismatch deletes the staging file and starts that file over.

import { MODEL_FILES, TOTAL_BYTES, CHUNK_BYTES } from "./model-manifest.js?v=@SITEV@";
import { digestBlob } from "./sha256.js?v=@SITEV@";

async function opfsRoot() {
  return navigator.storage.getDirectory();
}

// Verification reads the file in slices rather than through crypto.subtle, which needs one
// contiguous buffer: hashing a 1.3 GB asset that way allocates 1.3 GB in the JS heap and is what
// killed the tab on iOS before the download could finish. Peak memory here is one slice.

async function stagingHandle(root, asset, create) {
  try {
    return await root.getFileHandle(`${asset}.part`, { create });
  } catch {
    return null;
  }
}

async function fetchRange(asset, start, endInclusive) {
  const response = await fetch(`model/${asset}`, {
    headers: { Range: `bytes=${start}-${endInclusive}` },
  });
  if (!(response.status === 206 || (response.status === 200 && start === 0))) {
    throw new Error(`range fetch for ${asset} failed: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
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
        report("verifying", file.asset);
        if ((await digestBlob(blob)) === file.sha256) {
          out[file.key] = file.text
            ? new TextDecoder().decode(await blob.arrayBuffer())
            : { asset: file.asset, bytes: file.bytes };
          bytesDone += file.bytes;
          assetsDone += 1;
          report("cached", file.asset);
          continue;
        }
      }
      await root.removeEntry(file.asset);
    } catch {
      /* not cached yet */
    }

    // Resume or start the staged download.
    const staging = await stagingHandle(root, file.asset, true);
    let offset = (await staging.getFile()).size;
    if (offset > file.bytes) {
      // Corrupt staging (e.g. manifest changed): restart.
      await root.removeEntry(`${file.asset}.part`);
      offset = 0;
    }
    while (offset < file.bytes) {
      const end = Math.min(offset + CHUNK_BYTES, file.bytes) - 1;
      report("downloading", file.asset, offset);
      const chunk = await fetchRange(file.asset, offset, end);
      const writable = await staging.createWritable({ keepExistingData: true });
      await writable.write({ type: "write", position: offset, data: chunk });
      await writable.close();
      offset += chunk.byteLength;
    }

    report("verifying", file.asset, file.bytes);
    const staged = await staging.getFile();
    if ((await digestBlob(staged)) !== file.sha256) {
      await root.removeEntry(`${file.asset}.part`);
      throw new Error(`${file.asset}: digest mismatch after download; cleared for retry`);
    }
    // Promote by copying slice-wise, for the same reason the digest streams: reading the staged
    // file into one buffer to rewrite it would reintroduce the allocation this just removed.
    const finalHandle = await root.getFileHandle(file.asset, { create: true });
    const writable = await finalHandle.createWritable();
    for (let position = 0; position < staged.size; position += CHUNK_BYTES) {
      const end = Math.min(position + CHUNK_BYTES, staged.size);
      await writable.write({
        type: "write",
        position,
        data: await staged.slice(position, end).arrayBuffer(),
      });
    }
    await writable.close();
    await root.removeEntry(`${file.asset}.part`);

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
  for (const file of MODEL_FILES) {
    for (const name of [file.asset, `${file.asset}.part`]) {
      try {
        await root.removeEntry(name);
      } catch {
        /* absent */
      }
    }
  }
}
