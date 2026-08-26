// Exactness gate for the fused s8 dot-add relaxed op (spec opcode FD 93 01).
//
// Strategy: wat2wasm assembles the entire module with an `unreachable nop nop`
// sentinel standing in for the relaxed instruction; this script splices
// FD 93 01 into that exact 3-byte slot (same width, so every length prefix
// stays valid) and executes adversarial vectors, comparing each lane against
// scalar integer reference.
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

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
    unreachable nop nop   ;; <- spliced: fd 93 01 leaves one v128 above addr
    v128.store))`;

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rsd-probe-"));
const watPath = path.join(dir, "probe.wat");
const wasmPath = path.join(dir, "probe.wasm");
fs.writeFileSync(watPath, WAT);
execFileSync("wat2wasm", [watPath, "-o", wasmPath]);

const bytes = [...fs.readFileSync(wasmPath)];
// Locate the sentinel byte triple inside the exported func's body.
const needle = [0x00, 0x01, 0x01]; // unreachable, nop, nop
let idx = -1;
outer: for (let i = bytes.length - 3; i >= 16; i--) {
  for (let j = 0; j < 3; j++) if (bytes[i + j] !== needle[j]) continue outer;
  idx = i;
  break;
}
if (idx < 0) throw new Error("sentinel triple not found in assembled module");
const fused = [0xfd, 0x93, 0x01]; // i32x4.dot_i8x16_i7x16_add_s (final spec encoding)
bytes.splice(idx, 3, ...fused);
console.log(`spliced fused dot at byte ${idx}`);
console.log("PROBE_B64=" + Buffer.from(bytes).toString("base64"));

const mod = new WebAssembly.Module(new Uint8Array(bytes));
const ins = new WebAssembly.Instance(mod, {});
const u8 = new Uint8Array(ins.exports.m.buffer);

function run(aBytes, bBytes) {
  u8.set(aBytes, 64);
  u8.set(bBytes, 80);
  ins.exports.go();
  return [...new Int32Array(u8.buffer.slice(112, 128))];
}
function ref(a, b) {
  const out = [];
  for (let lane = 0; lane < 4; lane++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[lane * 4 + k] * b[lane * 4 + k];
    out.push(s);
  }
  return out;
}
const toU8 = a => Uint8Array.from(a.map(v => (v < 0 ? v + 256 : v)));
const cases = [
  ["all+127", Array(16).fill(127), Array(16).fill(127)],
  ["all-128", Array(16).fill(-128), Array(16).fill(-128)],
  ["a=-128,b=-127", Array(16).fill(-128), Array(16).fill(-127)],
  ["alt-signs", Array.from({ length: 16 }, (_, i) => (i % 2 ? -127 : 127)), Array.from({ length: 16 }, (_, i) => (i % 3 ? -128 : 127))],
  ["extremes-mixed", Array.from({ length: 16 }, (_, i) => ((i * i * 31) % 255) - 127), Array.from({ length: 16 }, (_, i) => ((i * 7 * 13) % 255) - 127)],
];
let failures = 0;
for (const [nameCase, a, b] of cases) {
  const got = run(toU8(a), toU8(b));
  const exp = ref(a, b);
  const ok = got.every((v, i) => v === exp[i]);
  if (!ok) failures++;
  console.log(nameCase.padEnd(18), "lanes:", JSON.stringify(got), "expected:", JSON.stringify(exp), ok ? "OK" : "MISMATCH");
}
console.log(failures === 0 ? "EXACTNESS: ALL CASES BIT-CORRECT ON THIS ENGINE" : "EXACTNESS FAILURES PRESENT");
process.exit(failures === 0 ? 0 : 1);
