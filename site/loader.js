// Chunked, resumable model loader: /model/* (same-origin proxy) → OPFS → verified bytes.
//
// Each file downloads in CHUNK_BYTES ranges appended to an OPFS staging file, so an
// interrupted 1.3 GB pull resumes from the last complete chunk instead of restarting.
// A file only ever reaches the engine after its full SHA-256 matches the pinned digest;
// a mismatch deletes the staging file and starts that file over.

import { MODEL_FILES, TOTAL_BYTES, CHUNK_BYTES } from "./model-manifest.js";

async function opfsRoot() {
  return navigator.storage.getDirectory();
}

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

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
 * Returns { fttsq: ArrayBuffer, codec: ArrayBuffer, vocab: string, ... } keyed per manifest.
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
        const buffer = await blob.arrayBuffer();
        if ((await sha256Hex(buffer)) === file.sha256) {
          out[file.key] = buffer;
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
    const buffer = await (await staging.getFile()).arrayBuffer();
    if ((await sha256Hex(buffer)) !== file.sha256) {
      await root.removeEntry(`${file.asset}.part`);
      throw new Error(`${file.asset}: digest mismatch after download; cleared for retry`);
    }
    // Promote: verified bytes land under the final name; staging is removed.
    const finalHandle = await root.getFileHandle(file.asset, { create: true });
    const writable = await finalHandle.createWritable();
    await writable.write(buffer);
    await writable.close();
    await root.removeEntry(`${file.asset}.part`);

    out[file.key] = buffer;
    bytesDone += file.bytes;
    assetsDone += 1;
    report("done", file.asset);
  }

  // Text files decode to strings for the tokenizer API.
  const decoder = new TextDecoder();
  for (const key of ["vocab", "merges", "tokenizerConfig"]) {
    out[key] = decoder.decode(out[key]);
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
