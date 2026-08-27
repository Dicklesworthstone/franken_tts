#!/bin/bash
# rc-perf-recert-06us driver (hardened 2026-08-26, PearlRobin): wait for a calm
# machine window, record machine state + provenance, run the ttfa_certification
# harness (release profile), then leave NDJSON receipts for benches/sprt_analyze.py.
#
# Why this exists: PERF_LEDGER's RTF rows are INDICATIVE ONLY until a quiet-window
# run certifies them (cv% <= 5 admission gate). This driver makes that run turnkey
# so the first calm window is spent measuring, not debugging plumbing.
#
# Environment knobs (all optional):
#   PERF_RECERT_THRESHOLD     loadavg(1m) admission threshold   (default 6.0)
#   PERF_RECERT_SUSTAIN_MIN   consecutive calm minutes required (default 3)
#   PERF_RECERT_MAX_WAIT_MIN  cap on pre-window waiting         (default 240)
#   PERF_RECERT_DRYRUN=1      orchestration smoke test: skips the harness,
#                             leaves receipts untouched, exits 0 on plumbing success
#   PERF_RECERT_RECEIPTS      output receipts path override (else repo default)
#   PERF_RECERT_TIMELINE      output timeline path override (else repo default)
#
# Pre-window guidance: build the release test binary BEFORE going idle so the calm
# window compiles nothing:
#   cargo test --release -p ftts-cli --test warm_engine_e2e --no-run
set -u

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
THRESHOLD="${PERF_RECERT_THRESHOLD:-6.0}"
SUSTAIN_MIN="${PERF_RECERT_SUSTAIN_MIN:-3}"
MAX_WAIT_MIN="${PERF_RECERT_MAX_WAIT_MIN:-240}"
DRYRUN="${PERF_RECERT_DRYRUN:-0}"
emit() {
  # Merge "key":value pairs (given wrapped in braces by callers) into one valid
  # single-line JSON object alongside the event name and timestamp.
  local payload="${2#\{}"
  payload="${payload%\}}"
  echo "{\"event\":\"$1\",${payload},\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$TIMELINE"
}
RECEIPTS="${PERF_RECERT_RECEIPTS:-$ROOT/docs/truth-pack/perf/perf-recert-receipts.ndjson}"
TIMELINE="${PERF_RECERT_TIMELINE:-$ROOT/docs/truth-pack/perf/perf-recert-loadavg.ndjson}"
mkdir -p "$(dirname "$RECEIPTS")" "$(dirname "$TIMELINE")"

# Fail fast on malformed knobs: a typo'd threshold used to surface as a silent
# four-hour wait with a misleading no_calm_window reason (python syntax error on
# every sample read as "not calm"), and SUSTAIN_MIN=0 referenced $L1 before any
# sample existed, tripping set -u.
case "$THRESHOLD" in
  ''|*[!0-9.]*) emit param_invalid "{\"param\":\"PERF_RECERT_THRESHOLD\",\"value\":\"$THRESHOLD\"}"; exit 5 ;;
esac
case "$SUSTAIN_MIN" in
  ''|*[!0-9]*) emit param_invalid "{\"param\":\"PERF_RECERT_SUSTAIN_MIN\",\"value\":\"$SUSTAIN_MIN\"}"; exit 5 ;;
esac
case "$MAX_WAIT_MIN" in
  ''|*[!0-9]*) emit param_invalid "{\"param\":\"PERF_RECERT_MAX_WAIT_MIN\",\"value\":\"$MAX_WAIT_MIN\"}"; exit 5 ;;
esac
[ "$SUSTAIN_MIN" -ge 1 ] || { emit param_invalid '{"param":"PERF_RECERT_SUSTAIN_MIN","reason":"must be >= 1"}'; exit 5; }

# Resolve the pinned-revision harness BEFORE waiting: failing here costs seconds;
# discovering a missing binary after a four-hour window wait wastes the window.
# (Commit e417681 ran this resolution after the wait loop.)
BIN=$(ls -t /Volumes/USB_NVME/cargo-target/aarch64-apple-darwin/release/deps/warm_engine_e2e-* 2>/dev/null | grep -v '\.d$' | head -1)
if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
  emit bin_missing "{\"hint\":\"cargo test --release -p ftts-cli --test warm_engine_e2e --no-run\"}"
  exit 4
fi

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
emit window_start "{\"threshold\":$THRESHOLD,\"sustain_min\":$SUSTAIN_MIN,\"max_wait_min\":$MAX_WAIT_MIN,\"dryrun\":$DRYRUN,\"head\":\"$HEAD_SHA\",\"bin\":\"$(basename "$BIN")\"}"

STREAK=0
for _ in $(seq 1 "$MAX_WAIT_MIN"); do
  L1=$(sysctl -n vm.loadavg | awk '{print $2}')
  echo "{\"event\":\"load_sample\",\"load1\":$L1,\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$TIMELINE"
  if python3 -c "exit(0 if $L1 < $THRESHOLD else 1)" 2>/dev/null; then
    STREAK=$((STREAK+1))
  else
    STREAK=0
  fi
  [ "$STREAK" -ge "$SUSTAIN_MIN" ] && break
  sleep 60
done

if [ "$STREAK" -lt "$SUSTAIN_MIN" ]; then
  FINAL_L1=$(sysctl -n vm.loadavg | awk '{print $2}')
  emit no_calm_window "{\"final_load1\":$FINAL_L1}"
  exit 3
fi
emit calm_confirmed "{\"pre_run_load1\":$L1}"

# Background loadavg sampler during the run. The parenthesized group is ONE
# process whose body IS the loop, so killing it stops sampling; any in-flight
# `sleep` child simply expires within 20s and emits nothing further.
( while true; do echo "{\"event\":\"run_load\",\"load1\":$(sysctl -n vm.loadavg | awk '{print $2}'),\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$TIMELINE"; sleep 20; done ) &
SAMPLER=$!

if [ "$DRYRUN" = "1" ]; then
  # Plumbing smoke: prove bin resolution, calm detection, sampler lifecycle and
  # receipt-path handling without burning synthesis minutes or touching files.
  emit dryrun_harness_skip "{\"receipts_untouched\":true}"
  RC=0
else
  "$BIN" ttfa_certification --ignored --nocapture --test-threads=1 > "$RECEIPTS" 2>&1
  RC=$?
fi

kill "$SAMPLER" 2>/dev/null
wait "$SAMPLER" 2>/dev/null || true
emit harness_done "{\"rc\":$RC,\"receipts\":\"$RECEIPTS\"}"
exit $RC