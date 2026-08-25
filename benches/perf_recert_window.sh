#!/bin/bash
# rc-perf-recert-06us driver: wait for calm, record machine state, run the
# ttfa_certification harness (release profile), then SPRT-analyze the receipts.
set -u
THRESHOLD=6.0        # loadavg(1m) admission threshold, recorded pre-run
SUSTAIN_MIN=3        # consecutive minutes below threshold before starting
RECEIPTS="$(git rev-parse --show-toplevel 2>/dev/null || echo .)/docs/truth-pack/perf/perf-recert-receipts.ndjson"
TIMELINE="$(git rev-parse --show-toplevel 2>/dev/null || echo .)/docs/truth-pack/perf/perf-recert-loadavg.ndjson"

echo "{\"event\":\"window_wait_start\",\"threshold\":$THRESHOLD,\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$TIMELINE"
STREAK=0
for i in $(seq 1 240); do
  L1=$(awk '{print $1}' /dev/stdin <<< "$(uptime)")
  # loadavg via sysctl for precision
  L1=$(sysctl -n vm.loadavg | awk '{print $2}')
  echo "{\"event\":\"load_sample\",\"load1\":$L1,\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$TIMELINE"
  OK=$(python3 -c "print(1 if $L1 < $THRESHOLD else 0)")
  if [ "$OK" = "1" ]; then STREAK=$((STREAK+1)); else STREAK=0; fi
  [ $STREAK -ge $SUSTAIN_MIN ] && break
  sleep 60
done
L1=$(sysctl -n vm.loadavg | awk '{print $2}')
if [ "$STREAK" -lt "$SUSTAIN_MIN" ]; then
  echo "{\"event\":\"no_calm_window\",\"final_load1\":$L1}" >> "$TIMELINE"
  exit 3
fi
echo "{\"event\":\"calm_confirmed\",\"pre_run_load1\":$L1,\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$TIMELINE"

# Background loadavg sampler during the run.
( while true; do echo "{\"event\":\"run_load\",\"load1\":$(sysctl -n vm.loadavg | awk '{print $2}'),\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$TIMELINE"; sleep 20; done ) &
SAMPLER=$!

BIN=$(ls -t /Volumes/USB_NVME/cargo-target/aarch64-apple-darwin/release/deps/warm_engine_e2e-* 2>/dev/null | grep -v '\.d$' | head -1)
"$BIN" ttfa_certification --ignored --nocapture --test-threads=1 > "$RECEIPTS" 2>&1
RC=$?
kill $SAMPLER 2>/dev/null
echo "{\"event\":\"harness_done\",\"rc\":$RC,\"receipts\":\"$RECEIPTS\"}" >> "$TIMELINE"
exit $RC
