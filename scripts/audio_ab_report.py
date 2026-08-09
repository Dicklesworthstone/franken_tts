#!/usr/bin/env python3
"""Objective audio A/B report for franken_tts quality gates.

Two comparison modes, because the two questions are different:

  aligned    Sample-aligned comparison. Valid ONLY when both files decode the SAME code
             stream (codec-isolated A/B: f32 codec vs int8 codec on identical codes).
             Reports SNR, max sample error, STFT log-spectral distance, mel-band LSD,
             and spectral convergence.

  stats      Distribution-level comparison for full-pipeline runs, where f32 and int8
             legitimately sample different tokens and the waveforms are different
             utterances of the same text. Reports duration, RMS, envelope correlation,
             and spectral summary statistics. It cannot certify similarity sample-by-
             sample and says so in its output.

Both modes render spectrograms through ffmpeg (showspectrumpic) and write a side-by-side
montage PNG next to the report.

Usage:
  audio_ab_report.py aligned ref.{wav,raw} test.{wav,raw} [--out DIR] [--label NAME]
  audio_ab_report.py stats   ref.{wav,raw} test.{wav,raw} [--out DIR] [--label NAME]

`.raw` inputs are headerless little-endian f32 mono at 24 kHz (the codec_time dump
format); `.wav` inputs are 16-bit PCM mono WAV.
"""

import argparse
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000
N_FFT = 1024
HOP = 256
EPS = 1e-10


def load_audio(path: Path) -> np.ndarray:
    if path.suffix == ".raw":
        data = np.frombuffer(path.read_bytes(), dtype="<f4").astype(np.float64)
        return data
    if path.suffix == ".wav":
        with wave.open(str(path), "rb") as reader:
            assert reader.getnchannels() == 1, "mono only"
            assert reader.getsampwidth() == 2, "16-bit PCM only"
            frames = reader.readframes(reader.getnframes())
        ints = struct.unpack(f"<{len(frames)//2}h", frames)
        return np.asarray(ints, dtype=np.float64) / 32768.0
    raise SystemExit(f"unsupported input {path}; use .wav or .raw")


def stft_mag(x: np.ndarray) -> np.ndarray:
    window = np.hanning(N_FFT)
    count = max(1, 1 + (len(x) - N_FFT) // HOP)
    frames = np.stack(
        [x[i * HOP : i * HOP + N_FFT] * window for i in range(count) if i * HOP + N_FFT <= len(x)]
    )
    return np.abs(np.fft.rfft(frames, axis=1))


def mel_filterbank(bands: int = 40) -> np.ndarray:
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    bins = N_FFT // 2 + 1
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(SAMPLE_RATE / 2), bands + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((N_FFT + 1) * hz_points / SAMPLE_RATE).astype(int).clip(0, bins - 1)
    bank = np.zeros((bands, bins))
    for band in range(bands):
        lo, mid, hi = bin_points[band : band + 3]
        if mid > lo:
            bank[band, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            bank[band, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    return bank


def aligned_report(ref: np.ndarray, test: np.ndarray) -> list[str]:
    n = min(len(ref), len(test))
    if len(ref) != len(test):
        yield_note = f"NOTE: lengths differ ({len(ref)} vs {len(test)}); compared first {n} samples"
    else:
        yield_note = None
    ref, test = ref[:n], test[:n]
    lines = []
    if yield_note:
        lines.append(yield_note)

    err = ref - test
    sig_power = float(np.sum(ref**2))
    err_power = float(np.sum(err**2))
    snr = 10 * math.log10(sig_power / err_power) if err_power > 0 else float("inf")
    peak = float(np.max(np.abs(ref))) or EPS
    lines.append(f"samples compared      : {n}")
    lines.append(f"reference RMS / peak  : {math.sqrt(sig_power/max(n,1)):.5f} / {peak:.4f}")
    lines.append(f"SNR                   : {snr:.1f} dB")
    lines.append(f"max |sample error|    : {float(np.max(np.abs(err))):.5f} ({100*float(np.max(np.abs(err)))/peak:.1f}% of peak)")

    ref_mag = stft_mag(ref)
    test_mag = stft_mag(test)
    frames = min(len(ref_mag), len(test_mag))
    ref_mag, test_mag = ref_mag[:frames], test_mag[:frames]

    # Log-spectral distance per frame, then mean: the standard LSD in dB.
    lsd_frames = np.sqrt(
        np.mean((20 * np.log10(ref_mag + EPS) - 20 * np.log10(test_mag + EPS)) ** 2, axis=1)
    )
    lines.append(f"log-spectral distance : {float(np.mean(lsd_frames)):.2f} dB mean, {float(np.max(lsd_frames)):.2f} dB worst frame")

    bank = mel_filterbank()
    ref_mel = ref_mag @ bank.T
    test_mel = test_mag @ bank.T
    mel_lsd = np.sqrt(
        np.mean((20 * np.log10(ref_mel + EPS) - 20 * np.log10(test_mel + EPS)) ** 2, axis=1)
    )
    lines.append(f"mel-band LSD (40 band): {float(np.mean(mel_lsd)):.2f} dB mean, {float(np.max(mel_lsd)):.2f} dB worst frame")

    convergence = float(np.linalg.norm(ref_mag - test_mag) / (np.linalg.norm(ref_mag) + EPS))
    lines.append(f"spectral convergence  : {convergence:.4f} (0 = identical)")
    return lines


def envelope(x: np.ndarray, hop: int = 480) -> np.ndarray:
    count = max(1, len(x) // hop)
    return np.array([math.sqrt(float(np.mean(x[i * hop : (i + 1) * hop] ** 2))) for i in range(count)])


def spectral_stats(x: np.ndarray) -> dict:
    mag = stft_mag(x)
    freqs = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    power = mag**2
    total = np.sum(power, axis=1) + EPS
    centroid = np.sum(power * freqs, axis=1) / total
    cumulative = np.cumsum(power, axis=1)
    rolloff_idx = np.argmax(cumulative >= 0.85 * total[:, None], axis=1)
    flatness = np.exp(np.mean(np.log(power + EPS), axis=1)) / (np.mean(power, axis=1) + EPS)
    return {
        "duration_s": len(x) / SAMPLE_RATE,
        "rms": math.sqrt(float(np.mean(x**2))),
        "centroid_hz": float(np.median(centroid)),
        "rolloff85_hz": float(np.median(freqs[rolloff_idx])),
        "flatness": float(np.median(flatness)),
    }


def stats_report(ref: np.ndarray, test: np.ndarray) -> list[str]:
    lines = [
        "MODE: distribution-level statistics. The two files are different sampled",
        "utterances of the same text; this cannot certify sample-level similarity.",
        "",
    ]
    a, b = spectral_stats(ref), spectral_stats(test)
    for key in a:
        lines.append(f"{key:14s}: ref {a[key]:10.3f}   test {b[key]:10.3f}")
    env_a, env_b = envelope(ref), envelope(test)
    m = min(len(env_a), len(env_b))
    if m > 3:
        corr = float(np.corrcoef(env_a[:m], env_b[:m])[0, 1])
        lines.append(f"{'envelope corr':14s}: {corr:.3f} over first {m} 20ms hops (alignment-sensitive)")
    return lines


def spectrogram(path: Path, out_png: Path) -> None:
    # ffmpeg needs a real container; wrap .raw on the fly.
    inputs = ["-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", str(path)] if path.suffix == ".raw" else ["-i", str(path)]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs,
         "-lavfi", "showspectrumpic=s=1024x400:legend=1", str(out_png)],
        check=True,
    )


def montage(left: Path, right: Path, out_png: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
         "-filter_complex", "vstack", str(out_png)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["aligned", "stats"])
    parser.add_argument("reference", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--out", type=Path, default=Path("audio_ab_out"))
    parser.add_argument("--label", default="ab")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ref = load_audio(args.reference)
    test = load_audio(args.test)

    lines = aligned_report(ref, test) if args.mode == "aligned" else stats_report(ref, test)

    ref_png = args.out / f"{args.label}_ref.png"
    test_png = args.out / f"{args.label}_test.png"
    both_png = args.out / f"{args.label}_stacked.png"
    try:
        spectrogram(args.reference, ref_png)
        spectrogram(args.test, test_png)
        montage(ref_png, test_png, both_png)
        lines.append(f"spectrograms          : {both_png} (reference on top)")
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        lines.append(f"spectrograms skipped  : {error}")

    report = "\n".join(
        [f"# audio A/B [{args.mode}] {args.reference.name} vs {args.test.name}", *lines]
    )
    (args.out / f"{args.label}_report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
