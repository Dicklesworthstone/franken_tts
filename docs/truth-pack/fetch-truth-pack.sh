#!/usr/bin/env bash
# franken_tts truth pack — fetch and verify every load-bearing upstream source.
#
#   ./fetch-truth-pack.sh            fetch missing files into snapshots/
#   ./fetch-truth-pack.sh --verify   re-hash snapshots/ against MANIFEST.sha256 (no network)
#   ./fetch-truth-pack.sh --verify --refetch
#                                    re-download everything, then hash (detects upstream drift)
#   ./fetch-truth-pack.sh --with-weights
#                                    also fetch + verify the 2.5 GB safetensors shards
#   ./fetch-truth-pack.sh --write-manifest
#                                    regenerate MANIFEST.sha256 from snapshots/ (pin-time only)
#
# A `core`-class mismatch is a STOP-THE-LINE event: the plan's [SOURCE] facts were
# asserted against these exact bytes (see PIN_RECORD.md, FACT_DISPOSITIONS.md).
# Exit codes: 0 ok · 1 usage/setup error · 2 hash mismatch · 3 fetch failure.

set -uo pipefail

# ---- THE PINS (immutable; changing one invalidates FACT_DISPOSITIONS.md) --------------
HF_REPO="Qwen/Qwen3-TTS-12Hz-0.6B-Base"
HF_REV="5d83992436eae1d760afd27aff78a71d676296fc"
GH_REPO="QwenLM/Qwen3-TTS"
GH_REV="022e286b98fbec7e1e916cb940cdf532cd9f488e"
# --------------------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAP="$HERE/snapshots"
SOURCES="$HERE/SOURCES.tsv"
MANIFEST="$HERE/MANIFEST.sha256"
WEIGHTS="$HERE/WEIGHTS.lfs.json"

VERIFY=0 REFETCH=0 WITH_WEIGHTS=0 WRITE_MANIFEST=0
for arg in "$@"; do
  case "$arg" in
    --verify)         VERIFY=1 ;;
    --refetch)        REFETCH=1 ;;
    --with-weights)   WITH_WEIGHTS=1 ;;
    --write-manifest) WRITE_MANIFEST=1 ;;
    -h|--help)        sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

command -v curl >/dev/null || { echo "need curl" >&2; exit 1; }
if command -v shasum >/dev/null; then SHA="shasum -a 256"
elif command -v sha256sum >/dev/null; then SHA="sha256sum"
else echo "need shasum or sha256sum" >&2; exit 1; fi
[[ -f "$SOURCES" ]] || { echo "missing $SOURCES" >&2; exit 1; }

url_for() { # origin remote_path -> URL
  case "$1" in
    hf)  printf 'https://huggingface.co/%s/resolve/%s/%s' "$HF_REPO" "$HF_REV" "$2" ;;
    gh)  printf 'https://raw.githubusercontent.com/%s/%s/%s' "$GH_REPO" "$GH_REV" "$2" ;;
    url) printf '%s' "$2" ;;
    *)   return 1 ;;
  esac
}

# Emit "origin<TAB>remote<TAB>local<TAB>class" for each real row.
rows() { grep -v '^[[:space:]]*#' "$SOURCES" | grep -v '^[[:space:]]*$'; }

fetch_one() { # url dest -> 0 ok
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  curl -fsSL --retry 3 --retry-delay 2 --max-time 300 -o "$dest.part" "$url" || return 1
  mv "$dest.part" "$dest"
}

fetched=0 skipped=0 failed=0
if [[ $VERIFY -eq 0 || $REFETCH -eq 1 ]]; then
  while IFS=$'\t' read -r origin remote local class; do
    dest="$SNAP/$local"
    if [[ -f "$dest" && $REFETCH -eq 0 ]]; then skipped=$((skipped+1)); continue; fi
    u="$(url_for "$origin" "$remote")" || { echo "bad origin '$origin' for $local" >&2; failed=$((failed+1)); continue; }
    if fetch_one "$u" "$dest"; then fetched=$((fetched+1)); else
      echo "FETCH FAILED [$class] $local <- $u" >&2; failed=$((failed+1))
    fi
  done < <(rows)
  echo "fetch: $fetched new, $skipped cached, $failed failed"
  [[ $failed -gt 0 ]] && exit 3
fi

# ---- weight shards: verified from the LFS oid (SHA-256 of the file content) ----------
if [[ $WITH_WEIGHTS -eq 1 ]]; then
  command -v python3 >/dev/null || { echo "need python3 for --with-weights" >&2; exit 1; }
  [[ -f "$WEIGHTS" ]] || { echo "missing $WEIGHTS" >&2; exit 1; }
  while IFS=$'\t' read -r wpath woid wsize; do
    dest="$SNAP/hf/$wpath"
    if [[ ! -f "$dest" || $REFETCH -eq 1 ]]; then
      echo "downloading $wpath ($wsize bytes) ..."
      fetch_one "$(url_for hf "$wpath")" "$dest" || { echo "FETCH FAILED $wpath" >&2; exit 3; }
    fi
    got="$($SHA "$dest" | awk '{print $1}')"
    if [[ "$got" == "$woid" ]]; then echo "OK   [weights] $wpath"; else
      echo "MISMATCH [weights] $wpath" >&2
      echo "  expected $woid" >&2
      echo "  actual   $got" >&2
      exit 2
    fi
  done < <(python3 -c '
import json,sys
for w in json.load(open(sys.argv[1]))["files"]:
    print(w["path"], w["sha256"], w["size"], sep="\t")
' "$WEIGHTS")
fi

# ---- manifest ------------------------------------------------------------------------
if [[ $WRITE_MANIFEST -eq 1 ]]; then
  {
    echo "# franken_tts truth pack — SHA-256 of every snapshotted source."
    echo "# HF  $HF_REPO @ $HF_REV"
    echo "# GH  $GH_REPO @ $GH_REV"
    echo "# Regenerate: ./fetch-truth-pack.sh --write-manifest   (pin-time only)"
    echo "# Verify:     ./fetch-truth-pack.sh --verify"
    while IFS=$'\t' read -r origin remote local class; do
      [[ -f "$SNAP/$local" ]] || { echo "missing snapshot: $local" >&2; exit 1; }
      printf '%s  %s  %s\n' "$($SHA "$SNAP/$local" | awk '{print $1}')" "$class" "$local"
    done < <(rows | LC_ALL=C sort -t$'\t' -k3,3)
  } > "$MANIFEST" || exit 1
  echo "wrote $MANIFEST"
  exit 0
fi

[[ $VERIFY -eq 1 ]] || exit 0
[[ -f "$MANIFEST" ]] || { echo "missing $MANIFEST — run --write-manifest first" >&2; exit 1; }

ok=0 bad_core=0 bad_drift=0 missing=0
while read -r want class local; do
  [[ "$want" == \#* ]] && continue
  dest="$SNAP/$local"
  if [[ ! -f "$dest" ]]; then echo "MISSING  [$class] $local" >&2; missing=$((missing+1)); continue; fi
  got="$($SHA "$dest" | awk '{print $1}')"
  if [[ "$got" == "$want" ]]; then ok=$((ok+1)); continue; fi
  echo "MISMATCH [$class] $local" >&2
  echo "  expected $want" >&2
  echo "  actual   $got" >&2
  if [[ "$class" == "drift-ok" ]]; then bad_drift=$((bad_drift+1)); else bad_core=$((bad_core+1)); fi
done < "$MANIFEST"

echo "verify: $ok ok, $bad_core core/support mismatched, $bad_drift drift-ok mismatched, $missing missing"
if [[ $bad_core -gt 0 || $missing -gt 0 ]]; then
  echo "STOP THE LINE: a pinned source no longer matches the bytes the plan was asserted against." >&2
  echo "Do not proceed on dependent beads until PIN_RECORD.md and FACT_DISPOSITIONS.md are re-adjudicated." >&2
  exit 2
fi
[[ $bad_drift -gt 0 ]] && echo "note: drift-ok source(s) changed upstream; the authoritative PDF still matches."
exit 0
