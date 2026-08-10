//! Reports the browser capabilities the loader depends on, per engine, on the REAL served origin.
//!
//! Must be run against the harness server rather than `about:blank`: OPFS requires a secure
//! context, so an opaque origin reports "unsupported" for every engine and the answer is worthless.
//!
//! Run: `node site/harness/engine_caps.mjs`

import { chromium, webkit } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import os from "node:os";
import { serve } from "./serve.mjs";

const siteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const { server, port } = await serve({ siteDir, modelFiles: {} });

for (const [name, engine] of [
  ["chromium", chromium],
  ["webkit", webkit],
]) {
  // PERSISTENT context, not the default ephemeral one. WebKit's OPFS needs a real storage
  // directory: in an ephemeral context the first `getFileHandle` fails with "operation failed for
  // an unknown transient reason", which reads like a browser limitation and is actually the
  // harness giving it nowhere to write. Chromium happens to tolerate the ephemeral case, so
  // testing only Chromium hides the difference entirely.
  const profile = await fs.mkdtemp(path.join(os.tmpdir(), `ftts-caps-${name}-`));
  const browser = await engine.launchPersistentContext(profile, {});
  const page = await browser.newPage();
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: "domcontentloaded" });
  const caps = await page.evaluate(async () => {
    const out = {
      secure: self.isSecureContext,
      isolated: self.crossOriginIsolated,
      sab: typeof SharedArrayBuffer !== "undefined",
      opfs: typeof navigator.storage?.getDirectory === "function",
    };
    if (!out.opfs) return out;
    try {
      const root = await navigator.storage.getDirectory();
      const handle = await root.getFileHandle("probe.bin", { create: true });
      out.createWritable = typeof handle.createWritable === "function";
      out.syncAccess = typeof handle.createSyncAccessHandle === "function";
      // The loader writes positionally through ONE writable held open for a whole file; if that
      // shape is unsupported the download cannot work even though `createWritable` exists.
      const writable = await handle.createWritable({ keepExistingData: true });
      await writable.write({ type: "write", position: 0, data: new Uint8Array(16) });
      await writable.close();
      out.positionalWrite = (await handle.getFile()).size;
      await root.removeEntry("probe.bin");
    } catch (error) {
      out.error = String(error);
    }
    return out;
  });
  console.log(`${name.padEnd(9)} ${JSON.stringify(caps)}`);
  await browser.close();
}

server.close();
