// Fingerprint-sweeps relaxed-range SIMD opcodes on THIS engine via the
// proven wat2wasm + byte-splice pipeline (no hand-assembled sections).
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rsd-sweep-"));
const WAT = `(module
  (memory (export "m") 1)
  (func $loadv (param i32) (result v128)
    local.get 0
    v128.load)
  (func (export "go") (local $a v128) (local $b v128) (local $c v128)
    (local.set $a (call $loadv (i32.const 64)))
    (local.set $b (call $loadv (i32.const 80)))
    (local.set $c (call $loadv (i32.const 96)))
    (i32.const 112)
    local.get $a
    local.get $b
    local.get $c
    unreachable nop nop
    v128.store))`;
const watPath = path.join(dir, "sweep.wat");
const wasmPath = path.join(dir, "sweep.wasm");
fs.writeFileSync(watPath, WAT);
execFileSync("wat2wasm", [watPath, "-o", wasmPath]);
const template = [...fs.readFileSync(wasmPath)];

function findSentinel(bytes) {
  const needle = [0x00, 0x01, 0x01];
  outer: for (let i = bytes.length - 3; i >= 16; i--) {
    for (let j = 0; j < 3; j++) if (bytes[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}
// Confirm sentinel in TEMPLATE corresponds to instruction slot (post-splice sanity only).
const SLEB = { 64: [0xc0, 0x00], 80: [0xd0, 0x00], 96: [0xe0, 0x00], 112: [0xf0, 0x00] };
function lebEq(bytes, at, val) {
  const enc = SLEB[val];
  return enc.every((b, k) => bytes[at + k] === b);
}
for (let op = 0x100; op <= 0x11f; op++) {
  // LEB-encode opcode as (op & 0x7f) | continuation pattern
  const ob = op < 0x80 ? [op] : [(op & 0x7f) | 0x80, op >> 7];
  if (template[findSentinel(template)] === undefined) throw new Error("no sentinel");
  const bytes = [...template];
  const idx = findSentinel(bytes);
  bytes.splice(idx, 3, 0xfd, ...ob);
  try {
    const mod = new WebAssembly.Module(new Uint8Array(bytes));
    const ins = new WebAssembly.Instance(mod, {});
    const u8 = new Uint8Array(ins.exports.m.buffer);
    u8.fill(0);
    u8.set(Array(16).fill(127), 64);
    u8.set(Array(16).fill(-128 & 255), 80);
    ins.exports.go();
    const lanes = [...new Int32Array(u8.buffer.slice(112, 128))];
    console.log(op.toString(16).padStart(3), "sss", "lanes=", JSON.stringify(lanes));
  } catch (e) {
    console.log(op.toString(16).padStart(3), ":", String(e.message || e).slice(0, 90));
  }
}
