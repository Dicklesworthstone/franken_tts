#!/usr/bin/env bash
# Run thermal-window A/B pairs through the OQ-17 per-observation collector.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: benches/energy_thermal_abba.sh --a-label A --a-command CMD \
  --b-label B --b-command CMD --generated-seconds SECONDS --output-dir DIR

Runs alternating ABBA and BAAB pairs (five pairs by default).  It writes a
summary TSV and exits 65 unless both arms have CV <= 5%; that is intentionally
NO VERDICT, never an implicit A/B win. Use `--first-order abba|baab` to apply
the pre-registered randomized order for the first pair.
EOF
}

a_label='' a_command='' b_label='' b_command='' generated_seconds='' output_dir=''
pairs=5 sample_ms=1000 first_order=abba
while (($#)); do
    case "$1" in
        --a-label) a_label=${2:?}; shift 2 ;;
        --a-command) a_command=${2:?}; shift 2 ;;
        --b-label) b_label=${2:?}; shift 2 ;;
        --b-command) b_command=${2:?}; shift 2 ;;
        --generated-seconds) generated_seconds=${2:?}; shift 2 ;;
        --output-dir) output_dir=${2:?}; shift 2 ;;
        --pairs) pairs=${2:?}; shift 2 ;;
        --sample-ms) sample_ms=${2:?}; shift 2 ;;
        --first-order) first_order=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 64 ;;
    esac
done

for value in "$a_label" "$b_label"; do
    [[ $value =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'invalid arm label\n' >&2; exit 64; }
done
if ! [[ $pairs =~ ^[0-9]+$ ]] || ((pairs < 5)); then
    printf '%s\n' '--pairs must be >= 5' >&2
    exit 64
fi
[[ $first_order == abba || $first_order == baab ]] || { printf '%s\n' '--first-order must be abba or baab' >&2; exit 64; }
[[ -n $a_command && -n $b_command && -n $generated_seconds && -n $output_dir ]] || { usage >&2; exit 64; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
collector="$script_dir/energy_thermal_bench.sh"
[[ -x $collector ]] || { printf 'collector is not executable: %s\n' "$collector" >&2; exit 69; }

run_arm() {
    local arm=$1 pair=$2 position=$3 command=$4
    "$collector" --label "${arm}-p${pair}-${position}" --generated-seconds "$generated_seconds" \
        --output-dir "$output_dir" --sample-ms "$sample_ms" --command "$command"
}

for ((pair = 1; pair <= pairs; ++pair)); do
    if [[ $first_order == abba && $((pair % 2)) -eq 1 ]] || [[ $first_order == baab && $((pair % 2)) -eq 0 ]]; then
        run_arm "$a_label" "$pair" 1 "$a_command"
        run_arm "$b_label" "$pair" 2 "$b_command"
        run_arm "$b_label" "$pair" 3 "$b_command"
        run_arm "$a_label" "$pair" 4 "$a_command"
    else
        run_arm "$b_label" "$pair" 1 "$b_command"
        run_arm "$a_label" "$pair" 2 "$a_command"
        run_arm "$a_label" "$pair" 3 "$a_command"
        run_arm "$b_label" "$pair" 4 "$b_command"
    fi
done

summary="$output_dir/abba-summary.tsv"
awk -F'[,:}]' -v a="$a_label" -v b="$b_label" '
    /"valid_for_claim":true/ {
        label=""; value="";
        for (i = 1; i <= NF; ++i) {
            if ($i ~ /"label"/) label=$(i + 1);
            if ($i ~ /"joules_per_generated_minute"/) value=$(i + 1);
        }
        if (index(label, a "-p") == 1) { sum_a += value; sq_a += value * value; n_a += 1 }
        if (index(label, b "-p") == 1) { sum_b += value; sq_b += value * value; n_b += 1 }
    }
    END {
        for (arm = 1; arm <= 2; ++arm) {
            n = arm == 1 ? n_a : n_b; sum = arm == 1 ? sum_a : sum_b; sq = arm == 1 ? sq_a : sq_b;
            name = arm == 1 ? a : b;
            if (n == 0) { printf "%s\t0\tNA\tNA\n", name; bad = 1; continue }
            mean = sum / n; cv = sqrt((sq / n) - (mean * mean)) / mean * 100;
            printf "%s\t%d\t%.9f\t%.9f\n", name, n, mean, cv;
            if (n < 10 || cv > 5) bad = 1;
        }
        exit bad ? 65 : 0;
    }' "$output_dir/energy-thermal.jsonl" > "$summary" || {
        printf 'NO VERDICT: see %s (need >=10 valid observations and CV <=5%% per arm)\n' "$summary" >&2
        exit 65
    }
printf 'ABBA thermal-pair gate passed: %s\n' "$summary"
