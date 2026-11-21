export const EXPECTED_SAMPLE_RATE = 24000;

/// Quantize f32 [-1, 1] to signed 16-bit PCM with round-half-away-from-zero
/// matching Rust's `ftts_core::audio::sample_to_i16`.
export function quantizeSampleToI16(v) {
  const scaled = Math.fround(v * 32767);
  const rounded = Math.sign(scaled) * Math.round(Math.abs(scaled));
  return Math.max(-32768, Math.min(32767, rounded));
}

/// Minimal 16-bit mono WAV writer so a parity run needs no audio dependency.
export function wav16(samples, rate = EXPECTED_SAMPLE_RATE) {
  const data = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i += 1) {
    data.writeInt16LE(quantizeSampleToI16(samples[i]), i * 2);
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVEfmt ", 8);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

/// Walk RIFF chunks, validating mono 16-bit PCM at the expected sample rate.
export function readWav16(buffer, expectedRate = EXPECTED_SAMPLE_RATE) {
  if (!Buffer.isBuffer(buffer) || buffer.length < 12) {
    throw new Error("WAV buffer too short");
  }
  if (buffer.toString("ascii", 0, 4) !== "RIFF" || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("not a RIFF/WAVE file");
  }
  let fmt = null;
  let samples = null;
  let offset = 12;
  while (offset + 8 <= buffer.length) {
    const id = buffer.toString("ascii", offset, offset + 4);
    const size = buffer.readUInt32LE(offset + 4);
    const payloadStart = offset + 8;
    const payloadEnd = payloadStart + size;
    if (payloadEnd > buffer.length) {
      throw new Error(`truncated WAV chunk ${id}`);
    }
    if (id === "fmt ") {
      if (size < 16) throw new Error(`WAV fmt chunk too small (${size} bytes)`);
      const audioFormat = buffer.readUInt16LE(payloadStart);
      const channels = buffer.readUInt16LE(payloadStart + 2);
      const sampleRate = buffer.readUInt32LE(payloadStart + 4);
      const bitsPerSample = buffer.readUInt16LE(payloadStart + 14);
      fmt = { audioFormat, channels, sampleRate, bitsPerSample };
    } else if (id === "data") {
      if (size === 0 || size % 2 !== 0) {
        throw new Error(`invalid WAV data chunk size (${size} bytes)`);
      }
      const out = new Array(size / 2);
      for (let i = 0; i < out.length; i += 1) {
        out[i] = buffer.readInt16LE(payloadStart + i * 2);
      }
      samples = out;
    }
    offset = payloadEnd + (size % 2);
  }
  if (!fmt) throw new Error("no fmt chunk in WAV");
  if (fmt.audioFormat !== 1) throw new Error(`unsupported WAV format ${fmt.audioFormat} (expected PCM=1)`);
  if (fmt.channels !== 1) throw new Error(`unsupported WAV channel count ${fmt.channels} (expected mono=1)`);
  if (fmt.bitsPerSample !== 16) throw new Error(`unsupported WAV bit depth ${fmt.bitsPerSample} (expected 16)`);
  if (fmt.sampleRate !== expectedRate) {
    throw new Error(`unexpected WAV sample rate ${fmt.sampleRate} Hz (expected ${expectedRate} Hz)`);
  }
  if (!samples || samples.length === 0) throw new Error("no non-empty data chunk in WAV");
  return { ...fmt, samples };
}

/// Validate fresh browser PCM capture: non-empty, all finite numbers, explicit expected rate.
export function validateBrowserPcm(pcm, rate, expectedRate = EXPECTED_SAMPLE_RATE) {
  if (!Array.isArray(pcm) || pcm.length === 0) {
    return { ok: false, reason: "empty or missing browser PCM" };
  }
  if (rate !== expectedRate) {
    return { ok: false, reason: `unexpected browser sample rate ${rate} (expected ${expectedRate})` };
  }
  for (let i = 0; i < pcm.length; i += 1) {
    if (typeof pcm[i] !== "number" || !Number.isFinite(pcm[i])) {
      return { ok: false, reason: `non-finite browser PCM sample at index ${i}` };
    }
  }
  return { ok: true, reason: "", samples: pcm.length, rate };
}

/// Strict full-length sample-for-sample comparison between browser PCM and a native golden.
export function compareBrowserToGolden(pcm, rate, goldenSamples, expectedRate = EXPECTED_SAMPLE_RATE) {
  const pcmCheck = validateBrowserPcm(pcm, rate, expectedRate);
  if (!pcmCheck.ok) {
    return {
      ok: false,
      reason: pcmCheck.reason,
      identical: 0,
      compared: 0,
      browserLen: Array.isArray(pcm) ? pcm.length : 0,
      goldenLen: Array.isArray(goldenSamples) ? goldenSamples.length : 0,
      maxDiff: 0,
      snr: Number.NEGATIVE_INFINITY,
      rawDiffs: [],
    };
  }
  if (!Array.isArray(goldenSamples) || goldenSamples.length === 0) {
    return {
      ok: false,
      reason: "empty native golden",
      identical: 0,
      compared: 0,
      browserLen: pcm.length,
      goldenLen: 0,
      maxDiff: 0,
      snr: Number.NEGATIVE_INFINITY,
      rawDiffs: [],
    };
  }
  const n = Math.min(goldenSamples.length, pcm.length);
  const ours = pcm.slice(0, n).map(quantizeSampleToI16);
  const theirs = goldenSamples.slice(0, n);
  let identical = 0;
  let maxDiff = 0;
  let sumSq = 0;
  let refSq = 0;
  const rawDiffs = [];
  for (let i = 0; i < n; i += 1) {
    const d = ours[i] - theirs[i];
    if (d === 0) {
      identical += 1;
    } else if (rawDiffs.length < 8) {
      rawDiffs.push({ i, f32: pcm[i], deltaLsb: Math.abs(pcm[i] * 32767 - theirs[i]) });
    }
    maxDiff = Math.max(maxDiff, Math.abs(d));
    sumSq += d * d;
    refSq += theirs[i] * theirs[i];
  }
  const snr = 10 * Math.log10((refSq || 1) / (sumSq || 1e-20));
  if (pcm.length !== goldenSamples.length) {
    return {
      ok: false,
      reason: `length mismatch: browser ${pcm.length} vs cli ${goldenSamples.length}`,
      identical,
      compared: n,
      browserLen: pcm.length,
      goldenLen: goldenSamples.length,
      maxDiff,
      snr,
      rawDiffs,
    };
  }
  const ok = identical === n;
  return {
    ok,
    reason: ok ? "" : `${n - identical} samples differ, max ${maxDiff} LSB, SNR ${snr.toFixed(1)} dB`,
    identical,
    compared: n,
    browserLen: pcm.length,
    goldenLen: goldenSamples.length,
    maxDiff,
    snr,
    rawDiffs,
  };
}
