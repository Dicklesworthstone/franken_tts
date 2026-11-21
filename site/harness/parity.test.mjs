import test from "node:test";
import assert from "node:assert/strict";
import {
  EXPECTED_SAMPLE_RATE,
  wav16,
  readWav16,
  validateBrowserPcm,
  compareBrowserToGolden,
} from "./parity.mjs";

test("wav16 and readWav16 round-trip mono 24kHz PCM16", () => {
  const input = [0, 0.25, -0.5, 1.0, -1.0];
  const wav = wav16(input, EXPECTED_SAMPLE_RATE);
  const parsed = readWav16(wav);
  assert.equal(parsed.sampleRate, EXPECTED_SAMPLE_RATE);
  assert.equal(parsed.channels, 1);
  assert.equal(parsed.bitsPerSample, 16);
  const cmp = compareBrowserToGolden(input, EXPECTED_SAMPLE_RATE, parsed.samples);
  assert.equal(cmp.ok, true);
  assert.equal(cmp.identical, input.length);
});

test("rejects empty, truncated, extended, non-finite, and wrong-rate PCM", () => {
  const golden = readWav16(wav16([0.1, -0.2, 0.3, -0.4])).samples;

  assert.equal(validateBrowserPcm([], EXPECTED_SAMPLE_RATE).ok, false);
  assert.equal(validateBrowserPcm(null, EXPECTED_SAMPLE_RATE).ok, false);
  assert.equal(validateBrowserPcm([0.1, Number.NaN], EXPECTED_SAMPLE_RATE).ok, false);
  assert.equal(validateBrowserPcm([0.1, Number.POSITIVE_INFINITY], EXPECTED_SAMPLE_RATE).ok, false);
  assert.equal(validateBrowserPcm([0.1], 48000).ok, false);

  const emptyCmp = compareBrowserToGolden([], EXPECTED_SAMPLE_RATE, golden);
  assert.equal(emptyCmp.ok, false);

  const truncCmp = compareBrowserToGolden([0.1, -0.2], EXPECTED_SAMPLE_RATE, golden);
  assert.equal(truncCmp.ok, false);
  assert.match(truncCmp.reason, /length mismatch/);

  const extraCmp = compareBrowserToGolden([0.1, -0.2, 0.3, -0.4, 0.0], EXPECTED_SAMPLE_RATE, golden);
  assert.equal(extraCmp.ok, false);
  assert.match(extraCmp.reason, /length mismatch/);
});

test("readWav16 rejects stereo, non-16-bit, wrong sample rate, and non-PCM WAVs", () => {
  const base = wav16([0.1, -0.1], EXPECTED_SAMPLE_RATE);

  const wrongRate = Buffer.from(base);
  wrongRate.writeUInt32LE(44100, 24);
  assert.throws(() => readWav16(wrongRate), /sample rate/);

  const stereo = Buffer.from(base);
  stereo.writeUInt16LE(2, 22);
  assert.throws(() => readWav16(stereo), /channel count/);

  const bit24 = Buffer.from(base);
  bit24.writeUInt16LE(24, 34);
  assert.throws(() => readWav16(bit24), /bit depth/);

  const floatWav = Buffer.from(base);
  floatWav.writeUInt16LE(3, 20);
  assert.throws(() => readWav16(floatWav), /WAV format/);
});
