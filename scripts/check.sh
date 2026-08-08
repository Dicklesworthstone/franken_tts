#!/usr/bin/env bash
#
# The one command. CI runs exactly this script as its single test step, so the gate lives in
# one place and cannot drift from a duplicated list of workflow commands.
#
#   ./scripts/check.sh
#
# Stages run in cheapest-first order and STOP AT THE FIRST FAILURE, so a structural mistake is
# reported in under two seconds instead of after a ten-minute build.
#
# Every stage prints a receipt: PASS, FAIL, or SKIP with a reason. A SKIP is never folded into
# "green" — the closing banner reads GREEN WITH SKIPS and lists them (AGENTS.md Doctrine #0.4:
# a skipped check is never presented as passing).
#
# Environment:
#   FTTS_CHECK_USE_RCH=1    offload cargo to remote workers — SEE THE WARNING BELOW
#   FTTS_CHECK_UBS_TIMEOUT  seconds to bound `ubs --diff` (default 300)
#
# The test stage sets FTTS_RECEIPTS itself, to target/receipts.ndjson, and the stage after it
# audits that stream: a model-gated test that skipped for want of weights is surfaced in the
# closing banner instead of disappearing into `libtest`'s pass count.
#
# Exit 0 = every required stage passed. Exit 1 = a stage failed.
#
# Bead: frankentts-p0-ci-083.

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
if [[ ! -t 1 ]]; then BOLD=""; RED=""; GREEN=""; YELLOW=""; DIM=""; OFF=""; fi

STAGE_NUM=0
SKIPPED=()
START_ALL=$SECONDS

banner() { printf '%s\n' "${DIM}────────────────────────────────────────────────────────────${OFF}"; }

stage_start() {
    STAGE_NUM=$((STAGE_NUM + 1))
    STAGE_NAME="$1"
    STAGE_START=$SECONDS
    printf '%s\n' "${BOLD}[${STAGE_NUM}] ${STAGE_NAME}${OFF}"
}

stage_pass() { printf '    %sPASS%s  %s (%ss)\n\n' "$GREEN" "$OFF" "$STAGE_NAME" "$((SECONDS - STAGE_START))"; }

stage_skip() {
    SKIPPED+=("$STAGE_NAME: $1")
    printf '    %sSKIP%s  %s — %s\n\n' "$YELLOW" "$OFF" "$STAGE_NAME" "$1"
}

stage_fail() {
    printf '    %sFAIL%s  %s (%ss)\n' "$RED" "$OFF" "$STAGE_NAME" "$((SECONDS - STAGE_START))"
    [[ -n "${1:-}" ]] && printf '    %s\n' "$1"
    banner
    printf '%sGATE FAILED%s at stage %s: %s\n' "$RED$BOLD" "$OFF" "$STAGE_NUM" "$STAGE_NAME"
    # Several agents share one working tree, so most red gates are somebody's half-finished
    # edit rather than anything wrong with committed code. Saying so here is the difference
    # between a one-glance attribution and an investigation that ends with you "fixing"
    # another agent's live crate — which AGENTS.md forbids and which destroys in-flight work.
    if [[ -n "$DIRTY_FILES" ]]; then
        printf '\n%sATTRIBUTION: the working tree is DIRTY (%s uncommitted path(s)).%s\n' \
            "$YELLOW$BOLD" "$DIRTY_COUNT" "$OFF"
        printf 'This failure may belong to an in-flight edit, not to committed code. Check\n'
        printf 'whether the failing file is in this list before changing anything:\n'
        printf '%s\n' "$DIRTY_FILES" | sed 's/^/    /'
        printf '\nTo judge committed code alone, compare against HEAD (%s) — e.g.\n' "$HEAD_SHA"
        printf '    git show HEAD:<failing-file> | grep <the-missing-symbol>\n'
        printf '%sDo NOT edit another agent'"'"'s uncommitted work to make this gate pass.%s\n' \
            "$YELLOW" "$OFF"
    fi
    printf '\nFix the root cause and re-run ./scripts/check.sh — do not skip past this.\n'
    exit 1
}

# The gate runs cargo LOCALLY by default, and that is a deliberate correctness call.
#
# This workspace consumes `../asupersync` and `/dp/frankentorch/crates/*` as PATH dependencies,
# which live outside the repository. rch syncs the repo to a worker but the worker resolves
# those out-of-tree paths against its OWN checkouts — observed 2026-08-06: a remote gate run
# compiled `/data/projects/asupersync` on the worker while the developer's tree pointed at a
# newer local copy, so the remote reported on source that was never in front of anyone.
#
# A gate that validates different source than you have is worse than a slow gate (G1 > G2), so
# remote offload is opt-in. Use it for exploratory builds, not for the gate, until rch can pin
# out-of-tree path deps.
CARGO_RUNNER=(cargo)
CARGO_MODE="local cargo"
if [[ -n "${FTTS_CHECK_USE_RCH:-}" ]] && command -v rch >/dev/null 2>&1; then
    CARGO_RUNNER=(rch exec -- cargo)
    CARGO_MODE="rch exec (OPT-IN: worker resolves out-of-tree path deps against ITS OWN checkouts)"
fi

run_cargo() { "${CARGO_RUNNER[@]}" "$@"; }

# Working-tree provenance, captured ONCE up front so the verdict can be attributed. A gate
# result is only meaningful against a known tree state, and in this repo the tree is shared.
HEAD_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
DIRTY_FILES="$(git status --porcelain 2>/dev/null | grep -v '^?? \.ntm/' || true)"
DIRTY_COUNT="$(printf '%s' "$DIRTY_FILES" | grep -c . || true)"

printf '%sfranken_tts gate%s  %s\n' "$BOLD" "$OFF" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf '%scargo via: %s%s\n' "$DIM" "$CARGO_MODE" "$OFF"
if [[ -z "$DIRTY_FILES" ]]; then
    printf '%stree:      %s (clean)%s\n' "$DIM" "$HEAD_SHA" "$OFF"
else
    printf '%stree:      %s + %s%s uncommitted path(s)%s%s — a failure below may not be yours%s\n' \
        "$DIM" "$HEAD_SHA" "$YELLOW" "$DIRTY_COUNT" "$OFF" "$DIM" "$OFF"
fi
banner

# ─────────────────────────────────────────────────────────────────────────────
# 1. Repo structure — architectural rules cargo cannot state
# ─────────────────────────────────────────────────────────────────────────────
stage_start "repo validators (forbid-unsafe architecture, frankentorch facade, CLI shims)"
if ! python3 scripts/validate_repo.py; then
    stage_fail "structural rule violated; see the file:rule lines above"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 1b. Execution census is regenerated from the pinned inputs, not hand-edited
# ─────────────────────────────────────────────────────────────────────────────
stage_start "execution-census drift guard (census matches its pinned inputs)"
CENSUS_STATUS=0
python3 scripts/generate_execution_census.py --check || CENSUS_STATUS=$?
if [[ "$CENSUS_STATUS" -eq 3 ]]; then
    # Exit 3 = the gitignored config snapshots are not fetched on this machine (fresh checkout,
    # CI). The census cannot be re-derived without them; an honest skip, never a red.
    stage_skip "truth-pack snapshots not fetched; census drift not checkable here"
elif [[ "$CENSUS_STATUS" -ne 0 ]]; then
    stage_fail "docs/truth-pack/EXECUTION_CENSUS.json is stale; run scripts/generate_execution_census.py"
else
    stage_pass
fi

# ─────────────────────────────────────────────────────────────────────────────
# 1c. The conformance corpus is a frozen oracle input, not a mutable text list
# ─────────────────────────────────────────────────────────────────────────────
stage_start "conformance-corpus freeze (source hashes, coverage, capture matrix)"
if ! python3 scripts/conformance_corpus.py; then
    stage_fail "the frozen conformance corpus changed or lost required coverage"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 2. The validators themselves still detect violations
# ─────────────────────────────────────────────────────────────────────────────
stage_start "repo-validator selftest (each rule fires on a mutated fixture)"
if ! python3 scripts/validate_repo.py --selftest target/repo-validate >/dev/null; then
    python3 scripts/validate_repo.py --selftest target/repo-validate || true
    stage_fail "a structural rule stopped detecting its violation"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 3. The listening protocol still detects degradation
# ─────────────────────────────────────────────────────────────────────────────
stage_start "listening-harness selftest (equivalence, tail gate, power controls)"
if ! python3 scripts/listening/run_panel.py selftest --out target/listening-selftest >/dev/null; then
    python3 scripts/listening/run_panel.py selftest --out target/listening-selftest || true
    stage_fail "the listening harness no longer reproduces its predeclared verdicts"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 4. Formatting
# ─────────────────────────────────────────────────────────────────────────────
stage_start "cargo fmt --check"
if ! cargo fmt --check; then
    stage_fail "run \`cargo fmt\`"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 5. Type check — and the multiple-build-targets warning is FATAL
# ─────────────────────────────────────────────────────────────────────────────
#
# Doctrine #9: `ftts` and `franken_tts` are two thin shims over one `cli_main()`, each [[bin]]
# pointing at its OWN file. Two targets sharing a path still compiles, but cargo warns — and
# that warning is the early symptom of the shared-path mistake, so the gate treats it as an
# error rather than letting it scroll past.
stage_start "cargo check --locked --all-targets"
CHECK_LOG="target/check-stage.log"
mkdir -p target
if ! run_cargo check --locked --all-targets 2>&1 | tee "$CHECK_LOG"; then
    stage_fail "see $CHECK_LOG"
fi
if grep -q "present in multiple build targets" "$CHECK_LOG"; then
    grep -n "present in multiple build targets" "$CHECK_LOG"
    stage_fail "a source file is claimed by more than one build target (doctrine #9: each [[bin]] needs its own shim file)"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 5b. The five release targets — the Phase-0 exit criterion, machine-enforced
# ─────────────────────────────────────────────────────────────────────────────
#
# Plan §12 Phase 0 exits on "builds green on all 5 targets". That was previously carried only by
# the workflow's `cross` job, which is `continue-on-error` and skipped on push — so on main the
# criterion neither ran nor blocked, and an exit gate nothing enforces is not a gate
# (frankentts-j5j).
#
# It is enforced HERE, inside the one job that already blocks, rather than by making `cross`
# blocking: the workflow comment records a measured reason not to request six concurrent runner
# slots per push (run 31123043545 sat queued 9m+; the free-plan cap is shared across every
# Dicklesworthstone repo), and starving the gate to prove a cross-check would trade the criterion
# we can enforce for one we cannot even observe.
#
# `cargo check` emits metadata without linking, so every target is checkable from any host without
# a cross-linker. A target whose std is not installed is reported as a SKIP naming that exact
# target — never folded into green — and is a hard FAIL when FTTS_REQUIRE_ALL_TARGETS=1, which CI
# sets after installing all five.
RELEASE_TARGETS=(
    x86_64-unknown-linux-gnu
    aarch64-unknown-linux-gnu
    x86_64-apple-darwin
    aarch64-apple-darwin
    x86_64-pc-windows-msvc
)
stage_start "cross-target check (${#RELEASE_TARGETS[@]} release targets)"
INSTALLED_TARGETS="$(rustup target list --installed 2>/dev/null || true)"
MISSING_TARGETS=()
for target in "${RELEASE_TARGETS[@]}"; do
    if ! printf '%s\n' "$INSTALLED_TARGETS" | grep -qx "$target"; then
        MISSING_TARGETS+=("$target")
        continue
    fi
    if ! run_cargo check --locked --all-targets --target "$target"; then
        stage_fail "$target does not build; the Phase-0 exit criterion covers all ${#RELEASE_TARGETS[@]} targets"
    fi
done
if [[ ${#MISSING_TARGETS[@]} -gt 0 ]]; then
    if [[ "${FTTS_REQUIRE_ALL_TARGETS:-0}" == "1" ]]; then
        stage_fail "std missing for: ${MISSING_TARGETS[*]} — run \`rustup target add\` for each (FTTS_REQUIRE_ALL_TARGETS=1 forbids skipping)"
    fi
    stage_skip "not checked (std not installed): ${MISSING_TARGETS[*]} — \`rustup target add\` them, or set FTTS_REQUIRE_ALL_TARGETS=1 to make this fatal"
else
    stage_pass
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Lints
# ─────────────────────────────────────────────────────────────────────────────
stage_start "cargo clippy --locked --all-targets -- -D warnings"
if ! run_cargo clippy --locked --all-targets -- -D warnings; then
    stage_fail "fix the lint; do not \`allow\` it without a recorded reason"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 7. Tests — the hard gate
# ─────────────────────────────────────────────────────────────────────────────
stage_start "cargo test --locked (HARD GATE — must exit 0 before any bead closes)"
# Capture the receipt stream. `libtest` prints captured stdout only for FAILING tests, so on a
# green run the receipts that distinguish `skipped` from `passed` would be invisible — exactly
# when that distinction is worth auditing. The path must be ABSOLUTE: cargo runs each test
# binary with its own package directory as cwd, so a relative path would scatter one file per
# crate. Bead: frankentts-p0-model-gated-77h.
RECEIPTS="$PWD/target/receipts.ndjson"
mkdir -p target
rm -f "$RECEIPTS"
export FTTS_RECEIPTS="$RECEIPTS"
if ! run_cargo test --locked; then
    stage_fail "no bead closes while this is red"
fi
stage_pass

# ─────────────────────────────────────────────────────────────────────────────
# 8. Receipt honesty — a model-gated test that never ran is not a pass
# ─────────────────────────────────────────────────────────────────────────────
#
# Stage 7 has its own skip-honesty problem one level down: `libtest` has no first-class skip, so
# a test that returns early because the multi-GB weights are absent is reported as `ok`. The
# honest verdict lives in the receipts, and this stage is what READS them — without a reader the
# stream is decoration and "skips stay distinguishable from green" is an unenforced claim.
#
# The selftest runs first for the same reason stages 2 and 3 exist: a check nobody has seen fail
# is a check nobody knows works.
stage_start "receipt honesty (skip-vs-green audit over the test receipts)"
if [[ ${#CARGO_RUNNER[@]} -gt 1 ]]; then
    stage_skip "cargo ran through rch; the receipt file stayed on the remote worker"
else
    if ! python3 scripts/summarize_receipts.py --selftest >/dev/null; then
        python3 scripts/summarize_receipts.py --selftest || true
        stage_fail "a receipt honesty rule stopped detecting its violation"
    fi
    RECEIPT_SKIPS="$PWD/target/receipt-skips.txt"
    if ! python3 scripts/summarize_receipts.py "$RECEIPTS" --skip-summary-file "$RECEIPT_SKIPS"; then
        stage_fail "the receipt stream is dishonest or dead; see the VIOLATION lines above"
    fi
    while IFS= read -r entry; do
        [[ -n "$entry" ]] && SKIPPED+=("model-gated test skipped — $entry")
    done < "$RECEIPT_SKIPS"
    stage_pass
fi

# ─────────────────────────────────────────────────────────────────────────────
# 9. Bug scan over the working-tree diff (bounded; optional tool)
# ─────────────────────────────────────────────────────────────────────────────
stage_start "ubs --diff"
UBS_TIMEOUT="${FTTS_CHECK_UBS_TIMEOUT:-300}"
if ! command -v ubs >/dev/null 2>&1; then
    stage_skip "ubs is not installed on this machine"
else
    # Pick a bounding command if one exists. Note the status is captured from a bare
    # invocation, not from inside `if ! …` — there `$?` is the status of the negation, so a
    # timeout's 124 would be invisible.
    BOUND=()
    if command -v timeout >/dev/null 2>&1; then
        BOUND=(timeout "$UBS_TIMEOUT")
    elif command -v gtimeout >/dev/null 2>&1; then
        BOUND=(gtimeout "$UBS_TIMEOUT")
    fi

    if [[ ${#BOUND[@]} -eq 0 ]]; then
        ubs --diff
        UBS_RC=$?
        [[ $UBS_RC -ne 0 ]] && stage_fail "ubs reported findings (exit $UBS_RC)"
        SKIPPED+=("ubs bound: no timeout(1) on this machine, ran unbounded")
        printf '    %sPASS%s  %s (%ss, %sunbounded — no timeout(1)%s)\n\n' \
            "$GREEN" "$OFF" "$STAGE_NAME" "$((SECONDS - STAGE_START))" "$YELLOW" "$OFF"
    else
        "${BOUND[@]}" ubs --diff
        UBS_RC=$?
        if [[ $UBS_RC -eq 124 ]]; then
            stage_fail "ubs exceeded its ${UBS_TIMEOUT}s bound (raise FTTS_CHECK_UBS_TIMEOUT if that is legitimate)"
        elif [[ $UBS_RC -ne 0 ]]; then
            stage_fail "ubs reported findings (exit $UBS_RC)"
        fi
        stage_pass
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
banner
ELAPSED=$((SECONDS - START_ALL))
if [[ ${#SKIPPED[@]} -eq 0 ]]; then
    printf '%sALL GREEN%s  %s stages, %ss\n' "$GREEN$BOLD" "$OFF" "$STAGE_NUM" "$ELAPSED"
else
    printf '%sGREEN WITH SKIPS%s  %s stages, %ss — %s skipped:\n' \
        "$YELLOW$BOLD" "$OFF" "$STAGE_NUM" "$ELAPSED" "${#SKIPPED[@]}"
    for entry in "${SKIPPED[@]}"; do printf '  %s- %s%s\n' "$YELLOW" "$entry" "$OFF"; done
    printf '%sThis is NOT a full green bar. Do not quote it as one.%s\n' "$DIM" "$OFF"
fi
exit 0
