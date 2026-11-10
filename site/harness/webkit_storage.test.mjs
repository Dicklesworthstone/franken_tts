import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { parseWebKitOrigin, sweepStaleWebKitOrigins } from "./webkit_storage.mjs";

function originBytes(scheme, host, port) {
  const schemeBuf = Buffer.from(scheme);
  const hostBuf = Buffer.from(host);
  const record = Buffer.alloc(4 + 1 + schemeBuf.length + 4 + 1 + hostBuf.length + 3);
  let offset = 0;
  record.writeUInt32LE(schemeBuf.length, offset);
  offset += 4;
  record[offset] = 1;
  offset += 1;
  schemeBuf.copy(record, offset);
  offset += schemeBuf.length;
  record.writeUInt32LE(hostBuf.length, offset);
  offset += 4;
  record[offset] = 1;
  offset += 1;
  hostBuf.copy(record, offset);
  offset += hostBuf.length;
  record[offset] = 1;
  record.writeUInt16BE(port, offset + 1);
  return Buffer.concat([record, record]);
}

test("parseWebKitOrigin reads the loopback port from a doubled record", () => {
  const parsed = parseWebKitOrigin(originBytes("http", "127.0.0.1", 36056));
  assert.deepEqual(parsed, { scheme: "http", host: "127.0.0.1", port: 36056 });
});

test("sweep removes other loopback ports and keeps the pinned origin", async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "webkit-opfs-"));
  const profile = path.join(root, "Default");
  const stale = path.join(profile, "stale", "stale");
  const kept = path.join(profile, "kept", "kept");
  const other = path.join(profile, "other", "other");
  await fs.mkdir(path.join(stale, "FileSystem"), { recursive: true });
  await fs.mkdir(path.join(kept, "FileSystem"), { recursive: true });
  await fs.mkdir(path.join(other, "FileSystem"), { recursive: true });
  await fs.writeFile(path.join(stale, "origin"), originBytes("http", "127.0.0.1", 52928));
  await fs.writeFile(path.join(kept, "origin"), originBytes("http", "127.0.0.1", 41731));
  await fs.writeFile(path.join(other, "origin"), originBytes("https", "example.com", 443));
  await fs.writeFile(path.join(stale, "FileSystem", "speech_tokenizer_model.safetensors"), "stale");
  await fs.writeFile(path.join(kept, "FileSystem", "speech_tokenizer_model.safetensors"), "kept");
  await fs.writeFile(path.join(other, "FileSystem", "speech_tokenizer_model.safetensors"), "other");

  const result = await sweepStaleWebKitOrigins({ keepPort: 41731, root });
  assert.equal(result.removed, 1);
  await assert.rejects(fs.stat(path.join(profile, "stale")));
  assert.equal(
    await fs.readFile(path.join(kept, "FileSystem", "speech_tokenizer_model.safetensors"), "utf8"),
    "kept",
  );
  assert.equal(
    await fs.readFile(path.join(other, "FileSystem", "speech_tokenizer_model.safetensors"), "utf8"),
    "other",
  );
  await fs.rm(root, { recursive: true, force: true });
});
