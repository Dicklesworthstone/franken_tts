#!/usr/bin/env bash
# ProductionQuality (Contract B) nightly objective screen — bead frankentts-v-prod-harness-t96.
#
# Runs the two model-gated harnesses that PR CI skips only when weights are absent:
#   1. pq_distributional_tf      — teacher-forced KL/JS/top-k/rank vs the oracle fixtures
#   2. pq_free_running_battery   — WER / structural word errors / stop / drift /
#                                  duration-per-word / secondary identity cosine, scored
#                                  through the external `fw` ASR scorer when present
#
# This is the Tier-0 objective screen of docs/CONFORMANCE_AND_LISTENING.md §4.5: it may
# veto, never authorize. A red exit blocks lever work; it does not clear a lossy lever.
#
# Outputs land under $FTTS_PQ_OUT (default: target/pq-nightly):
#   receipts.ndjson                    every receipt event from both harnesses
#   pq_scorecard.distributional.json   teacher-forced family scorecard
#   pq_scorecard.free_running.json     free-running family scorecard
#   summary.txt                        skip/fail rollup via scripts/summarize_receipts.py
#
# Usage:
#   scripts/pq_nightly.sh                 # full run
#   FTTS_PQ_FW_BIN=/path/to/fw scripts/pq_nightly.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${FTTS_PQ_OUT:-target/pq-nightly}"
mkdir -p "$OUT"

export FTTS_RECEIPTS="$OUT/receipts.ndjson"
export FTTS_PQ_REPORT="$OUT/pq_scorecard.json"
: > "$FTTS_RECEIPTS"

echo "== pq_nightly: building test binaries =="
cargo test -p ftts-conformance --test pq_distributional_tf --test pq_free_running_battery --no-run

echo "== pq_nightly: distributional (teacher-forced) family =="
cargo test -p ftts-conformance --test pq_distributional_tf -- --nocapture

echo "== pq_nightly: free-running battery =="
cargo test -p ftts-conformance --test pq_free_running_battery -- --nocapture

if [ -f scripts/summarize_receipts.py ]; then
  python3 scripts/summarize_receipts.py "$FTTS_RECEIPTS" | tee "$OUT/summary.txt" || true
fi

SKIPS=$(grep -c '"outcome":"skipped"' "$FTTS_RECEIPTS" || true)
FAILS=$(grep -c '"outcome":"failed"' "$FTTS_RECEIPTS" || true)
echo "== pq_nightly: done — receipts in $FTTS_RECEIPTS (skips: $SKIPS, fails: $FAILS) =="
[ "$FAILS" -eq 0 ]
