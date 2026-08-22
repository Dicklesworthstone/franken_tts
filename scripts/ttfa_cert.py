#!/usr/bin/env python3
"""TTFA certification harness (bead frankentts-dcfn): the doctrine-8 measurement, as one command.

Protocol (all mandatory, from the bead):
  - WARM in-process context: one `ftts talk` session, model loaded once; utterances
    N>=2 measured (two warmups per corpus item are run and DISCARDED).
  - Interactive profile (the session's native 1-frame packets).
  - Pinned corpus: a short first clause and a long paragraph, reported SEPARATELY —
    TTFA and RTF are different products.
  - Pinned per-utterance seeds (explicit overrides, so any run is reproducible).
  - TTFA basis: the session's delivery clock — time from the say job to the first PCM
    packet handed to the audio channel (the first `audio` event's `ttfa_ms`). On this
    corpus the first packet is audible (verified in the packet-parity receipts:
    audible==raw at these seeds); the basis is stated in the receipt either way.
  - QUIET WINDOW, fail closed: the harness REFUSES to certify when host load says the
    swarm is building (measurement noise is not a certification). --provisional runs
    anyway and stamps the output PROVISIONAL_LOCAL_WIN — unquotable, by name.
  - n >= 20 per corpus item; mean, stdev, cv reported; every sample retained.

The JSON receipt this prints is the raw material for the docs/PERF_LEDGER.md entry —
the ledger entry is written by a human/agent REVIEWING the receipt, never auto-appended.
No ratio against upstream is computed, ever ([NO ADMISSIBLE RATIO], see
docs/QWEN3_TTS_STREAMING_CONTRACT.md §4.2). A comparison against the pre-RT0 tree is
admissible only if that tree is rebuilt and interleaved under the same controls; this
harness measures ONE binary and says so.
"""

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

SHORT_TEXT = "Right away, then."
LONG_TEXT = (
    "The certification paragraph runs long enough that startup effects wash out and the "
    "decoder settles into its steady state, carrying several clauses of ordinary prose, a "
    "number like three hundred and nineteen, and a final sentence that lets the utterance "
    "end on the model's own terms."
)
QUIET_LOAD_1M = 6.0  # a swarm-idle M4 Pro sits well under this; builds blow far past it


class Talk:
    def __init__(self, ftts, pcm_path):
        self.proc = subprocess.Popen(
            [ftts, "talk", "--pcm-out", str(pcm_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True,
        )
        self.events, self.consumed = [], 0
        self.lock = threading.Condition()
        threading.Thread(target=self._pump, daemon=True).start()
        self.wait(lambda e: e["event"] == "session_start")

    def _pump(self):
        for line in self.proc.stdout:
            with self.lock:
                self.events.append(json.loads(line))
                self.lock.notify_all()

    def send(self, op):
        self.proc.stdin.write(json.dumps(op) + "\n")
        self.proc.stdin.flush()

    def wait(self, want, timeout=300.0):
        deadline = time.monotonic() + timeout
        with self.lock:
            while True:
                while self.consumed < len(self.events):
                    event = self.events[self.consumed]
                    self.consumed += 1
                    if want(event):
                        return event
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("no matching event")
                self.lock.wait(remaining)


def measure(talk, text, seeds, warmups):
    """One corpus item: warmups discarded, then one TTFA sample per seed."""
    samples = []
    for index, seed in enumerate([warmups[0], warmups[1], *seeds]):
        talk.send({"op": "say", "context": "cert", "text": text,
                   "continue": False, "seed": seed})
        first = talk.wait(lambda e: e["event"] == "audio" and "ttfa_ms" in e)
        done = talk.wait(lambda e: e["event"] in ("speak_complete", "speak_cancelled"))
        if done["event"] != "speak_complete":
            raise RuntimeError(f"utterance did not complete: {done}")
        if index >= 2:  # warmups discarded
            samples.append({
                "seed": seed,
                "ttfa_ms": first["ttfa_ms"],
                "frames": done["frames"],
                "rtf": done.get("rtf"),
            })
    return samples


def summarize(samples):
    values = [s["ttfa_ms"] for s in samples]
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean_ms": round(mean, 1),
        "stdev_ms": round(stdev, 1),
        "cv": round(stdev / mean, 4) if mean else None,
        "min_ms": min(values),
        "p50_ms": sorted(values)[len(values) // 2],
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ftts", default=os.environ.get("FTTS_BIN", "ftts"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out-dir", default="/tmp/ftts-ttfa-cert")
    parser.add_argument(
        "--provisional", action="store_true",
        help="run despite a non-quiet host; the receipt is stamped PROVISIONAL_LOCAL_WIN "
             "and is not ledger material",
    )
    args = parser.parse_args()

    load_before = os.getloadavg()
    if load_before[0] > QUIET_LOAD_1M and not args.provisional:
        print(json.dumps({
            "outcome": "refused",
            "reason": f"host is not quiet (load {load_before}); certification under "
                      f"load is noise, not measurement",
            "remediation": "rerun when 1-minute load is under "
                           f"{QUIET_LOAD_1M}, or use --provisional for an unquotable number",
        }))
        raise SystemExit(3)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    version = subprocess.run(
        [args.ftts, "--version"], capture_output=True, text=True
    ).stdout.strip()

    talk = Talk(args.ftts, out / "cert.pcm")
    talk.send({"op": "open", "context": "cert", "voice": "matt", "seed": 1, "id": "o"})
    talk.wait(lambda e: e["event"] == "context_open")

    receipt = {
        "harness": "scripts/ttfa_cert.py",
        "bead": "frankentts-dcfn",
        "claim_tier": "PROVISIONAL_LOCAL_WIN" if args.provisional else "certification-candidate",
        "basis": "delivery clock: say-job start to first PCM packet on the audio channel "
                 "(session audio.ttfa_ms); interactive 1-frame packets; warm in-process "
                 "model; audible==first-packet on this corpus per packet-parity receipts",
        "binary": {"version": version, "path": str(args.ftts)},
        "commit": commit,
        "route": "int8 (session_start.route)",
        "voice": "matt",
        "host_load_before": list(load_before),
        "corpus": {},
        "no_admissible_ratio_note": "no upstream comparison computed; see streaming "
                                    "contract §4.2",
    }
    short = measure(talk, SHORT_TEXT, [1000 + i for i in range(args.n)], [900, 901])
    long_ = measure(talk, LONG_TEXT, [2000 + i for i in range(args.n)], [902, 903])
    receipt["corpus"]["short_clause"] = {"text": SHORT_TEXT, "summary": summarize(short),
                                         "samples": short}
    receipt["corpus"]["long_paragraph"] = {"text": LONG_TEXT, "summary": summarize(long_),
                                           "samples": long_}
    receipt["host_load_after"] = list(os.getloadavg())

    talk.send({"op": "shutdown"})
    talk.wait(lambda e: e["event"] == "session_end")
    talk.proc.stdin.close()
    talk.proc.wait(timeout=30)

    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "corpus"} |
                     {"short_clause": receipt["corpus"]["short_clause"]["summary"],
                      "long_paragraph": receipt["corpus"]["long_paragraph"]["summary"],
                      "receipt": str(out / "receipt.json")}))


if __name__ == "__main__":
    main()
