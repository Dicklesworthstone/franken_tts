// Playwright's WebKit does not keep OPFS inside the temp profile browser.mjs
// deletes. It writes WebsiteData under ~/Library/WebKit, keyed by origin, and
// an origin includes the port. serve() used to bind port 0, so every --webkit
// run was a new origin and left another copy of the speech tokenizer behind.
// Fourteen of those copies were 9 GB. The profile cleanup never saw them.
//
// WebKit runs therefore bind WEBKIT_HARNESS_PORT and this module drops every
// other 127.0.0.1 origin. One cached copy remains. A random port is refused.

import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

export const WEBKIT_HARNESS_PORT = 41731;

export const WEBKIT_WEBSITE_DATA = path.join(
  os.homedir(),
  "Library/WebKit/org.webkit.Playwright/WebsiteData",
);

/// First record of a WebKit `origin` file: u32le length, 0x01, bytes, repeated
/// for scheme and host, then 0x01 and a big-endian port. The file stores that
/// record twice; the first one is enough.
export function parseWebKitOrigin(buffer) {
  const buf = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
  let offset = 0;
  const readString = () => {
    if (offset + 5 > buf.length) return null;
    const len = buf.readUInt32LE(offset);
    offset += 4;
    const flag = buf[offset];
    offset += 1;
    if (flag !== 1 || len < 0 || offset + len > buf.length) return null;
    const text = buf.toString("utf8", offset, offset + len);
    offset += len;
    return text;
  };
  const scheme = readString();
  const host = readString();
  if (!scheme || !host) return null;
  let port = null;
  if (offset < buf.length && buf[offset] === 1 && offset + 3 <= buf.length) {
    port = buf.readUInt16BE(offset + 1);
  }
  return { scheme, host, port };
}

function isLoopback(host) {
  return host === "127.0.0.1" || host === "localhost" || host === "[::1]" || host === "::1";
}

async function treeBytes(dir) {
  let total = 0;
  const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    const child = path.join(dir, entry.name);
    if (entry.isSymbolicLink()) continue;
    if (entry.isDirectory()) total += await treeBytes(child);
    else if (entry.isFile()) {
      const st = await fs.stat(child).catch(() => null);
      if (st) total += st.blocks * 512;
    }
  }
  return total;
}

/// Remove loopback origins whose port is not `keepPort`. Non-loopback origins
/// are left alone. Returns how many origin directories were removed and the
/// allocated bytes they held.
export async function sweepStaleWebKitOrigins({
  keepPort,
  root = WEBKIT_WEBSITE_DATA,
} = {}) {
  let removed = 0;
  let bytes = 0;
  const profiles = await fs.readdir(root, { withFileTypes: true }).catch(() => []);
  for (const profile of profiles) {
    if (!profile.isDirectory()) continue;
    const profileDir = path.join(root, profile.name);
    const hashes = await fs.readdir(profileDir, { withFileTypes: true }).catch(() => []);
    for (const hash of hashes) {
      if (!hash.isDirectory()) continue;
      const originFile = path.join(profileDir, hash.name, hash.name, "origin");
      const buf = await fs.readFile(originFile).catch(() => null);
      if (!buf) continue;
      const origin = parseWebKitOrigin(buf);
      if (!origin || !isLoopback(origin.host)) continue;
      if (origin.port === keepPort) continue;
      const dir = path.join(profileDir, hash.name);
      bytes += await treeBytes(dir);
      await fs.rm(dir, { recursive: true, force: true });
      removed += 1;
    }
  }
  return { removed, bytes };
}
