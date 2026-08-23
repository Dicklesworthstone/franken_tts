#!/usr/bin/env python3
"""The reference conversation loop (bead frankentts-4ie8): documentation that executes.

Wires transcripts -> LLM -> `ftts talk` -> speaker into one spoken exchange and logs a
per-turn latency breakdown. Two input modes:

  --replay <fixture.ndjson>   franken_whisper replay fixture (the four in
                              crates/ftts-conformance/tests/fixtures/franken_whisper_replay/),
                              played on its own timeline. THE default proof mode: it needs
                              no microphone, no AEC, and no live fw build.
  --listen '<command>'        LIVE mode: spawn the command (an `fw robot listen ...`
                              invocation, schema 1.1.0) and consume its NDJSON events from
                              stdout. Default policy is ARM 2 of the AEC spike (bead
                              frankentts-8s0y): the mic is GATED while the agent speaks
                              (plus --gate-hangover-ms), so the loop cannot trigger on its
                              own output — deterministic, but no barge-in. Pass --barge-in
                              only when the capture side runs through a proven echo
                              canceller (VoiceProcessingIO, spike arm 1); it re-enables
                              cancel-on-speech_started exactly as in replay mode.

The LLM is canned by default (--llm canned): deterministic replies, so the latency table
measures OUR loop, not a model's mood. Playback is optional (--play), via the first
available raw-PCM player; without it the audio lands in --session-dir/session.pcm.

This is a REFERENCE, deliberately small: an agent building a real orchestrator should
read it top to bottom. ftts's product surface ends at the talk protocol; nothing here is
product code.

Barge-in rule (from the seam contract): fw's `speech_started` arriving while the agent
is speaking, followed by >=0.5s of continued speech evidence (in replay: the fixture's
next transcript event within the window), triggers `cancel`; the truncation receipt's
spoken_text (a documented UPPER BOUND on what was heard) rewrites the assistant turn.

Latency stamps per turn (NDJSON on stdout, aggregate table on exit):
  t_endpoint      the fixture's utterance_end (user stopped speaking)
  t_llm_first     canned LLM's first sentence ready (simulated TTFT, --llm-ttft-ms)
  t_say_sent      first say op written to ftts
  t_first_audio   first `audio` event for that reply
  voice_to_voice  t_first_audio - t_endpoint
"""

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "crates/ftts-conformance/tests/fixtures/franken_whisper_replay"

CANNED_REPLIES = [
    "Sure — the build finished, and all of the tests passed on the first try.",
    "That part is done as well; the next step would be wiring the demo end to end.",
    "Understood. I will stop there and wait for your go-ahead.",
]


def log(obj):
    print(json.dumps(obj), flush=True)


class Talk:
    """A live `ftts talk` session: ops in, events out, PCM to a file (or player)."""

    def __init__(self, ftts: str, session_dir: Path, play: bool):
        session_dir.mkdir(parents=True, exist_ok=True)
        self.pcm_path = session_dir / "session.pcm"
        self.proc = subprocess.Popen(
            [ftts, "talk", "--pcm-out", str(self.pcm_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.events = []
        self.consumed = 0  # wait() cursor: each event is matched against AT MOST once
        self.lock = threading.Condition()
        threading.Thread(target=self._pump, daemon=True).start()
        self.player = self._spawn_player() if play else None
        self._tail = threading.Thread(target=self._pump_pcm, daemon=True)
        if self.player:
            self._tail.start()
        self.wait(lambda e: e["event"] == "session_start")

    def _spawn_player(self):
        # First available raw-PCM player; short buffer flags matter for barge-in feel.
        candidates = [
            ["play", "-q", "-t", "raw", "-r", "24000", "-e", "signed", "-b", "16", "-c", "1", "-"],
            ["ffplay", "-loglevel", "quiet", "-autoexit", "-nodisp", "-f", "s16le", "-ar", "24000", "-i", "-"],
        ]
        for cmd in candidates:
            if shutil.which(cmd[0]):
                return subprocess.Popen(cmd, stdin=subprocess.PIPE)
        log({"note": "no raw-PCM player found (sox `play` or ffplay); audio stays in the session file"})
        return None

    def _pump_pcm(self):
        # Tail the session PCM file into the player as it grows.
        offset = 0
        while self.proc.poll() is None:
            try:
                with open(self.pcm_path, "rb") as pcm:
                    pcm.seek(offset)
                    chunk = pcm.read()
            except FileNotFoundError:
                chunk = b""
            if chunk and self.player and self.player.stdin:
                try:
                    self.player.stdin.write(chunk)
                    self.player.stdin.flush()
                except BrokenPipeError:
                    return
                offset += len(chunk)
            else:
                time.sleep(0.02)

    def _pump(self):
        for line in self.proc.stdout:
            event = json.loads(line)
            with self.lock:
                self.events.append(event)
                self.lock.notify_all()

    def send(self, op):
        self.proc.stdin.write(json.dumps(op) + "\n")
        self.proc.stdin.flush()

    def wait(self, want, timeout=180.0):
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
                    raise TimeoutError(f"no matching event; last: {self.events[-1] if self.events else None}")
                self.lock.wait(remaining)


def sentence_chunks(text: str):
    """Flush on sentence-final punctuation; the well-known abbreviation hazards do not
    appear in the canned replies, so the reference keeps the split honest and simple."""
    out, current = [], ""
    for word in text.split(" "):
        current = f"{current}{word} "
        if word.endswith((".", "!", "?")):
            out.append(current)
            current = ""
    if current.strip():
        out.append(current)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ftts", default=os.environ.get("FTTS_BIN", "ftts"))
    parser.add_argument("--replay", default=str(FIXTURES / "normal_turn.ndjson"))
    parser.add_argument(
        "--listen",
        default=None,
        help="LIVE mode: shell command producing fw robot-listen NDJSON on stdout "
             "(e.g. \"fw robot listen --source mic\"); overrides --replay",
    )
    parser.add_argument(
        "--barge-in",
        action="store_true",
        help="live mode only: trust the capture side's echo cancellation (AEC spike "
             "arm 1) and allow cancel-on-speech_started while the agent speaks; "
             "WITHOUT this flag the mic is gated during agent speech (arm 2)",
    )
    parser.add_argument(
        "--gate-hangover-ms",
        type=int,
        default=150,
        help="arm-2 gate tail: fw events arriving this soon after the agent stops "
             "speaking are still attributed to self-hearing and dropped",
    )
    parser.add_argument("--voice", default="matt")
    parser.add_argument("--llm-ttft-ms", type=int, default=300, help="simulated canned-LLM first-sentence latency")
    parser.add_argument(
        "--llm",
        default="canned",
        help="'canned' (deterministic, measures OUR loop) or 'exec:<command>' — the "
             "command receives the user text on stdin and prints the reply (a real "
             "agent, ollama wrapper, etc.); its latency then dominates and is reported "
             "as measured, not simulated",
    )
    parser.add_argument("--turns", type=int, default=1,
                        help="replay the fixture this many times for a longer table")
    parser.add_argument("--session-dir", default="/tmp/ftts-talk-demo")
    parser.add_argument("--play", action="store_true", help="play PCM through sox/ffplay if present")
    parser.add_argument(
        "--preamble",
        default=None,
        help="start this agent reply BEFORE replaying — the barge_in fixture presumes "
             "the agent is already mid-speech when the user interrupts (try it with a "
             "long sentence)",
    )
    args = parser.parse_args()

    live_mode = args.listen is not None
    fixture = [] if live_mode else [
        json.loads(line)
        for line in Path(args.replay).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")  # fixtures carry contract headers
    ] * max(args.turns, 1)
    talk = Talk(args.ftts, Path(args.session_dir), args.play)
    talk.send({"op": "open", "context": "demo", "voice": args.voice, "seed": 7, "id": "open"})
    talk.wait(lambda e: e["event"] == "context_open")

    turns = []
    transcript = []
    pending_deltas = []  # transcript.delta fallback when utterance_end carries no text
    reply_index = 0
    utterance = None
    speaking = False
    # Barge-in is a policy, not a capability: replay fixtures presume it; live mode
    # earns it only with proven echo cancellation on the capture side (spike arm 1).
    allow_barge = (not live_mode) or args.barge_in
    # Arm-2 gate state (live mode without --barge-in). One interval suffices: events
    # are processed in arrival order and each agent reply opens exactly one gate.
    gate_open = None
    gate_close = float("-inf")
    swallow_utterance = False  # an utterance that STARTED inside the gate is dropped whole
    gated_drops = 0

    def parse_ts(stamp):
        # fw fixtures may carry RFC3339 strings or float seconds; accept both.
        if isinstance(stamp, (int, float)):
            return float(stamp)
        from datetime import datetime
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()

    def ensure_idle():
        # Wait out the current reply's terminal receipt before the next turn.
        nonlocal speaking
        if speaking:
            talk.wait(lambda e: e["event"] in ("speak_complete", "speak_cancelled"))
            speaking = False

    # In replay mode the fixture runs ON ITS OWN TIMELINE: the inter-event gaps are
    # what create the overlap windows barge-in needs (an agent still speaking when
    # speech_started arrives). Without the sleeps no overlap exists. The preamble
    # serves both modes: it puts the agent mid-speech before the first user event.
    if args.preamble:
        gate_open = time.monotonic()  # consulted only in live arm-2; harmless otherwise
        talk.send({"op": "say", "context": "demo", "text": args.preamble, "continue": False})
        talk.wait(lambda e: e["event"] == "speak_start")
        talk.wait(lambda e: e["event"] == "audio")
        speaking = True
        log({"note": "preamble speaking; the fixture timeline now runs against it"})
        if live_mode and not args.barge_in:
            # Arm 2 cannot let the mic hear the preamble either — hold, then release.
            talk.wait(lambda e: e["event"] in ("speak_complete", "speak_cancelled"))
            speaking = False
            gate_close = time.monotonic() + args.gate_hangover_ms / 1000.0

    def handle_fw_event(event, arrival):
        """One fw event (replay or live), stamped with its LOCAL arrival time."""
        nonlocal speaking, utterance, reply_index
        nonlocal gate_open, gate_close, swallow_utterance, gated_drops
        kind = event.get("event")

        # ARM-2 GATE (live mode, no proven AEC): anything the mic "heard" while the
        # agent was speaking — or within the hangover after — is presumed to be the
        # agent's own output and is dropped, INCLUDING the tail of an utterance that
        # started inside the gate and ended after it (swallow_utterance).
        if live_mode and not args.barge_in:
            in_gate = gate_open is not None and gate_open <= arrival <= gate_close
            if kind == "speech_started":
                if in_gate:
                    swallow_utterance = True
                    gated_drops += 1
                    log({"gated": kind, "reason": "agent was speaking (arm-2 mic gate)"})
                    return
                swallow_utterance = False
            elif kind in ("transcript.delta", "utterance_end") and (in_gate or swallow_utterance):
                if kind == "utterance_end":
                    swallow_utterance = False
                    pending_deltas.clear()
                gated_drops += 1
                log({"gated": kind, "reason": "utterance attributed to self-hearing"})
                return

        if kind == "speech_started" and speaking and allow_barge:
            # Barge-in: the user talks over the agent. The fixtures place a transcript
            # event right behind the trigger, standing in for the >=0.5s evidence.
            talk.send({"op": "cancel", "context": "demo", "id": f"cancel-{utterance}"})
            receipt = talk.wait(lambda e: e["event"] == "speak_cancelled")
            speaking = False
            # THE truncation consumption: the assistant turn in the running transcript
            # becomes only what was (at most) heard — the receipt, used downstream.
            if transcript and transcript[-1][0] == "assistant":
                transcript[-1] = ("assistant (interrupted)", receipt["spoken_text"])
            log({"turn": "barge-in", "spoken_upper_bound": receipt["spoken_text"],
                 "frames_delivered": receipt["frames_delivered"]})
        if kind == "transcript.delta" and event.get("text"):
            # fw contract (both policies): each delta is a FRESH committed span; the
            # utterance text equals the space-joined spans. A delta that repeats the
            # accumulated text is the pre-6589686e leak bug — fw wants it reported,
            # not accommodated, so it is flagged loudly and still joined as-is.
            fresh = event["text"]
            if pending_deltas and fresh.startswith(" ".join(pending_deltas)):
                log({"warning": "cumulative-shaped delta — fw contract violation, "
                                "report to the fw agent", "text": fresh[:120]})
            pending_deltas.append(fresh)
        if kind == "utterance_end":
            user_text = event.get("text") or " ".join(pending_deltas).strip()
            pending_deltas.clear()
            if not user_text:
                return  # fw's empty-utterance pairing: endpoint with nothing committed
            ensure_idle()
            t_endpoint = arrival
            if args.llm.startswith("exec:"):
                # A real model: its own latency, measured not simulated.
                reply = subprocess.run(
                    args.llm[5:], shell=True, input=user_text,
                    capture_output=True, text=True, timeout=120,
                ).stdout.strip() or "I did not catch that."
            else:
                time.sleep(args.llm_ttft_ms / 1000.0)  # canned LLM "thinks"
                reply = CANNED_REPLIES[reply_index % len(CANNED_REPLIES)]
                reply_index += 1
            t_llm_first = time.monotonic()
            chunks = sentence_chunks(reply)
            gate_open = time.monotonic()  # arm-2: the mic is suspect from here on
            gate_close = float("inf")
            talk.send({"op": "say", "context": "demo", "text": chunks[0], "continue": True})
            t_say = time.monotonic()
            for chunk in chunks[1:]:
                talk.send({"op": "say", "context": "demo", "text": chunk, "continue": True})
            talk.send({"op": "flush", "context": "demo"})
            speaking = True
            started = talk.wait(lambda e: e["event"] == "speak_start")
            utterance = started["utterance"]
            first_audio = talk.wait(
                lambda e, u=utterance: e["event"] == "audio" and e.get("utterance") == u
            )
            t_audio = time.monotonic()
            turn = {
                "turn": len(turns),
                "user_text": user_text,
                "reply": reply,
                "llm_ttft_ms": round((t_llm_first - t_endpoint) * 1000),
                "say_to_first_audio_ms": round((t_audio - t_say) * 1000),
                "voice_to_voice_ms": round((t_audio - t_endpoint) * 1000),
                "ttfa_ms_reported": first_audio.get("ttfa_ms"),
            }
            transcript.append(("user", user_text))
            transcript.append(("assistant", reply))
            turns.append(turn)
            log(turn)
            if live_mode and not args.barge_in:
                # Arm 2 is half-duplex BY DESIGN: hold until the reply finishes, then
                # release the gate after the hangover. Events that arrived meanwhile
                # keep their true arrival stamps, so the gate test above stays exact.
                talk.wait(lambda e: e["event"] in ("speak_complete", "speak_cancelled"))
                speaking = False
                gate_close = time.monotonic() + args.gate_hangover_ms / 1000.0
            # Otherwise do NOT block for the receipt: the reply keeps speaking while
            # events keep arriving, which is exactly the window a barge-in needs.

    if live_mode:
        # LIVE: spawn the listen command, stamp each NDJSON line's arrival on the
        # reader thread (so gate decisions survive any processing backlog), and run
        # until the command exits (file-replay sources end themselves) or Ctrl-C.
        fw_events = queue.Queue()
        fw_proc = subprocess.Popen(
            args.listen, shell=True, stdout=subprocess.PIPE, text=True,
        )

        def _fw_pump():
            for line in fw_proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    fw_events.put((time.monotonic(), json.loads(line)))
                except json.JSONDecodeError:
                    log({"note": "non-JSON line from the listen command", "line": line[:200]})

        threading.Thread(target=_fw_pump, daemon=True).start()
        try:
            while True:
                try:
                    arrival, event = fw_events.get(timeout=0.25)
                except queue.Empty:
                    if fw_proc.poll() is not None and fw_events.empty():
                        break
                    continue
                kind = event.get("event")
                if kind in ("session_start", "listen.session_start"):
                    log({"note": "listen session started", "fw": event})
                elif kind == "session_stats":
                    log({"note": "listen session ended", "fw_stats": event})
                else:
                    handle_fw_event(event, arrival)
        except KeyboardInterrupt:
            # fw flushes the open utterance's decoded words on SIGINT (exit 130) — a
            # terminal Ctrl-C reaches the whole process group, so that final text is
            # already in flight. Capture it for the transcript; do NOT speak a reply.
            log({"note": "interrupted; draining fw's final flush, then shutting down"})
            drain_deadline = time.monotonic() + 2.0
            while time.monotonic() < drain_deadline:
                try:
                    _, event = fw_events.get(timeout=0.2)
                except queue.Empty:
                    if fw_proc.poll() is not None:
                        break
                    continue
                if event.get("event") == "utterance_end" and event.get("text"):
                    transcript.append(("user (final, unanswered)", event["text"]))
                    log({"note": "final flushed utterance captured", "text": event["text"]})
        finally:
            if fw_proc.poll() is None:
                fw_proc.terminate()
            try:
                fw_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                fw_proc.kill()
    else:
        previous_ts = None
        for event in fixture:
            stamp = event.get("ts")
            if stamp is not None:
                if previous_ts is not None:
                    gap = min(max(parse_ts(stamp) - previous_ts, 0.0), 3.0)
                    time.sleep(gap)
                previous_ts = parse_ts(stamp)
            handle_fw_event(event, time.monotonic())

    ensure_idle()
    talk.send({"op": "shutdown"})
    talk.wait(lambda e: e["event"] == "session_end")
    talk.proc.stdin.close()
    talk.proc.wait(timeout=30)

    if turns:
        v2v = sorted(t["voice_to_voice_ms"] for t in turns)
        tts = sorted(t["say_to_first_audio_ms"] for t in turns)
        p95 = v2v[min(len(v2v) - 1, int(len(v2v) * 0.95))]
        mode = (
            "live-arm1-barge-in" if live_mode and args.barge_in
            else "live-arm2-gated" if live_mode
            else "replay"
        )
        log({
            "summary": True,
            "mode": mode,
            "turns": len(turns),
            "gated_events_dropped": gated_drops if live_mode else None,
            "voice_to_voice_ms": {"p50": v2v[len(v2v) // 2], "p95": p95, "max": v2v[-1]},
            "say_to_first_audio_ms": {"p50": tts[len(tts) // 2], "max": tts[-1]},
            "host_load_avg": list(os.getloadavg()),
            "note": "one-shot run output; canned llm_ttft is simulated (--llm-ttft-ms); "
                    "fw-side endpointing latency is inside voice_to_voice in live mode",
        })
        log({"final_transcript": [f"{role}: {text}" for role, text in transcript]})


if __name__ == "__main__":
    main()
