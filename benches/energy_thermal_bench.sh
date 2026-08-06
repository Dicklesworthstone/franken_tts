#!/usr/bin/env bash
# OQ-17 energy/thermal harness.  It deliberately retains raw sampler output.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: benches/energy_thermal_bench.sh --label LABEL --generated-seconds SECONDS \
  --output-dir DIR --command 'FTTS_ARGS...'

Run a fixed, already-correct workload once and emit one JSONL observation.  Run
the harness in ABBA order yourself (A B B A) or through the future bakeoff
driver; a sequential A-then-B result is inadmissible.  `--command` is run with
`bash -c`; it must create the declared number of generated audio seconds.

Options:
  --label LABEL                Run label: [A-Za-z0-9._-]+ only.
  --generated-seconds SECONDS  Generated audio duration; must be > 0.
  --output-dir DIR             Artifact directory; it is created if absent.
  --command COMMAND            Exact workload command (not emitted in JSONL).
  --sample-ms N                Sampler cadence, default 1000 ms (100..5000).
EOF
}

label=''
generated_seconds=''
output_dir=''
workload_command=''
sample_ms=1000

while (($#)); do
    case "$1" in
        --label) label=${2:?}; shift 2 ;;
        --generated-seconds) generated_seconds=${2:?}; shift 2 ;;
        --output-dir) output_dir=${2:?}; shift 2 ;;
        --command) workload_command=${2:?}; shift 2 ;;
        --sample-ms) sample_ms=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
    esac
done

[[ $label =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'invalid --label\n' >&2; exit 64; }
[[ $generated_seconds =~ ^[0-9]+([.][0-9]+)?$ ]] || { printf 'invalid --generated-seconds\n' >&2; exit 64; }
if ! [[ $sample_ms =~ ^[0-9]+$ ]] || ((sample_ms < 100 || sample_ms > 5000)); then
    printf 'invalid --sample-ms (expected 100..5000)\n' >&2
    exit 64
fi
[[ -n $output_dir && -n $workload_command ]] || { usage >&2; exit 64; }
awk "BEGIN { exit !($generated_seconds > 0) }" || { printf 'generated seconds must be > 0\n' >&2; exit 64; }

mkdir -p "$output_dir"
run_id="${label}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
raw_dir="$output_dir/$run_id"
mkdir -p "$raw_dir"
jsonl="$output_dir/energy-thermal.jsonl"

now_ns() {
    perl -MTime::HiRes=time -e 'printf "%.0f\n", time() * 1_000_000_000'
}

emit_result() {
    local platform=$1 domain=$2 energy_j=$3 wall_ns=$4 thermal=$5 sampler=$6
    local joules_per_min average_watts valid
    joules_per_min=$(awk "BEGIN { printf \"%.9f\", 60 * $energy_j / $generated_seconds }")
    average_watts=$(awk "BEGIN { printf \"%.9f\", $energy_j / ($wall_ns / 1000000000) }")
    valid=true
    [[ $thermal == Nominal || $thermal == unavailable ]] || valid=false
    printf '{"schema_version":1,"run_id":"%s","label":"%s","platform":"%s","domain":"%s","energy_j":%s,"generated_seconds":%s,"joules_per_generated_minute":%s,"wall_ns":%s,"average_watts":%s,"thermal":"%s","sampler":"%s","valid_for_claim":%s,"raw_artifact":"%s"}\n' \
        "$run_id" "$label" "$platform" "$domain" "$energy_j" "$generated_seconds" \
        "$joules_per_min" "$wall_ns" "$average_watts" "$thermal" "$sampler" "$valid" \
        "$raw_dir" | tee -a "$jsonl"
}

run_macos() {
    local raw_file="$raw_dir/powermetrics.plist" start_ns end_ns pm_pid cpu_mj gpu_mj ane_mj
    local energy_j thermal records=0 fragment
    sudo -n powermetrics --format plist --samplers cpu_power,thermal \
        --sample-rate "$sample_ms" --sample-count -1 --output-file "$raw_file" &
    pm_pid=$!
    # Give powermetrics a chance to initialize, without counting its startup in the workload.
    sleep 1
    start_ns=$(now_ns)
    bash -c "$workload_command"
    end_ns=$(now_ns)
    # SIGTERM asks powermetrics to flush and exit.  Do not use SIGINT here:
    # on this host sudo relays it to the foreground process group.  `sudo`
    # may have a separate powermetrics child, so stop both exact PIDs.
    local pm_child
    local -a pm_children=()
    while IFS= read -r pm_child; do pm_children+=("$pm_child"); done < <(pgrep -P "$pm_pid" 2>/dev/null || true)
    if ((${#pm_children[@]})); then
        kill -TERM "${pm_children[@]}" 2>/dev/null || true
    fi
    kill -TERM "$pm_pid" 2>/dev/null || true
    wait "$pm_pid" || true

    # plist output is NUL-separated.  Preserve every fragment and sum sample energy,
    # whose unit is millijoules on the verified Darwin powermetrics plist surface.
    perl -0 -0777 -e '
        my $i = 0; for my $part (split /\0/, do { local $/; <> }) {
            next unless length $part; open my $fh, ">", "$ARGV.$i.plist" or die $!;
            binmode $fh; print {$fh} $part; close $fh; ++$i;
        }' "$raw_file"
    for fragment in "$raw_file".*.plist; do
        [[ -f $fragment ]] || continue
        cpu_mj=$(plutil -extract processor.cpu_energy raw -o - "$fragment" 2>/dev/null || printf 0)
        gpu_mj=$(plutil -extract processor.gpu_energy raw -o - "$fragment" 2>/dev/null || printf 0)
        ane_mj=$(plutil -extract processor.ane_energy raw -o - "$fragment" 2>/dev/null || printf 0)
        thermal=$(plutil -extract thermal_pressure raw -o - "$fragment" 2>/dev/null || printf unavailable)
        awk -v c="$cpu_mj" -v g="$gpu_mj" -v a="$ane_mj" 'BEGIN { printf "%.9f\n", (c + g + a) / 1000 }' >> "$raw_dir/sample-energy-j.txt"
        printf '%s\n' "$thermal" >> "$raw_dir/thermal-pressure.txt"
        ((records += 1))
    done
    ((records >= 3)) || { printf 'insufficient powermetrics samples (%s; need >= 3)\n' "$records" >&2; exit 65; }
    energy_j=$(awk '{ total += $1 } END { printf "%.9f", total }' "$raw_dir/sample-energy-j.txt")
    thermal=$(sort -u "$raw_dir/thermal-pressure.txt" | paste -sd, -)
    emit_result darwin arm64-soc "$energy_j" "$((end_ns - start_ns))" "$thermal" powermetrics-plist
}

run_linux() {
    local root=/sys/class/powercap start_ns end_ns energy_j=0 thermal=unavailable zone start end maximum delta
    local -a zones=()
    shopt -s nullglob
    for zone in "$root"/*-rapl:[0-9]*; do
        [[ -r $zone/energy_uj && -r $zone/max_energy_range_uj ]] && zones+=("$zone")
    done
    ((${#zones[@]})) || {
        printf 'no top-level RAPL powercap zones with energy_uj/max_energy_range_uj; refusing an unbounded HWMON estimate\n' >&2
        exit 69
    }
    for zone in "${zones[@]}"; do cat "$zone/energy_uj" > "$raw_dir/$(basename "$zone").start_uj"; done
    start_ns=$(now_ns)
    bash -c "$workload_command"
    end_ns=$(now_ns)
    for zone in "${zones[@]}"; do
        start=$(<"$raw_dir/$(basename "$zone").start_uj")
        end=$(cat "$zone/energy_uj")
        maximum=$(cat "$zone/max_energy_range_uj")
        delta=$(awk -v s="$start" -v e="$end" -v m="$maximum" 'BEGIN { d=e-s; if (d < 0) d += m; printf "%.0f", d }')
        printf '%s\t%s\t%s\t%s\n' "$(basename "$zone")" "$start" "$end" "$delta" >> "$raw_dir/rapl-deltas.tsv"
        energy_j=$(awk -v total="$energy_j" -v d="$delta" 'BEGIN { printf "%.9f", total + d / 1000000 }')
    done
    emit_result linux package-rapl "$energy_j" "$((end_ns - start_ns))" "$thermal" powercap-energy_uj
}

case "$(uname -s)" in
    Darwin) run_macos ;;
    Linux) run_linux ;;
    *) printf 'unsupported platform: %s\n' "$(uname -s)" >&2; exit 69 ;;
esac
