#!/usr/bin/env python3
"""Aggregate the NDJSON test receipts so a skipped check can never be quoted as green.

`libtest` has no first-class skip: a model-gated test that returns early because the weights
are absent is reported as `ok`. The honest signal lives in the receipts emitted by
`ftts-conformance` (`FTTS_RECEIPTS=<file>`), and this is the thing that reads them — without a
reader, the receipt stream is decoration and the whole "skips stay distinguishable from green"
claim is unenforced.

    python3 scripts/summarize_receipts.py target/receipts.ndjson [--json]
                                          [--skip-summary-file PATH]
    python3 scripts/summarize_receipts.py --selftest

CONSUMER: `scripts/check.sh` stage 7b. It feeds `--skip-summary-file` into the closing banner,
so a run whose model-gated ladders never executed prints `GREEN WITH SKIPS` and lists them.

DEFECT CLASS: a green CI badge over a suite where every weight-dependent assertion silently
returned early — the counterfeit green of AGENTS.md Doctrine #0.4.

DELETION CONDITION: when the ConformanceExact ladder runner
(bead `frankentts-v-ladder-runner-zmk`) emits its own scorecard over this same stream and
check.sh calls that instead, this script goes away rather than being kept "for compatibility".

Exit 0 = the stream is well-formed (skips are fine, and reported). Exit 1 = a receipt is
dishonest or the stream is dead. Exit 2 = the file could not be read.

Bead: frankentts-p0-model-gated-77h.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

#  Wire strings from `ftts_conformance::report::Outcome::as_str`. Kept in sync by
#  `receipt_outcome_wire_strings_match_the_summarizer` in the conformance crate — an outcome
#  added on the Rust side without updating this set fails that test rather than silently
#  becoming an unknown-outcome violation here.
OUTCOMES = {"passed", "failed", "skipped", "xfail"}
EVENTS = {"contract_check", "stage"}

#  Receipts from `tests/model_gate_demo.rs`, which emits skips on purpose to demonstrate the
#  convention. They are counted and printed, never folded into the gate's skip list — a banner
#  that is permanently yellow for staged reasons trains readers to ignore it. Honesty checks
#  below still apply to them in full.
DEMO_CONTRACT_PREFIX = "Demo/"

#  Wire strings from `ftts_conformance::report::OracleTier::as_str`. The tiers exist because the
#  CPU FP32 fallback and the native CUDA path are not interchangeable evidence:
#  `report.rs` states "a CPU green is not native-golden proof", and bead frankentts-rf4 says
#  "CPU-smoke output is explicitly non-golden". Without the checks below, a receipt could claim
#  `native_cuda` while its own capture records `device: cpu`, and this aggregator — the thing
#  standing between the receipt stream and a green gate — would wave it through. That is the
#  counterfeit-green class this script exists to catch, one level deeper than skip-vs-pass.
ORACLE_TIERS = {"cpu_fp32_fallback", "native_cuda"}
NATIVE_TIER = "native_cuda"

#  Devices that cannot substantiate a native-CUDA claim. Matched on the prefix before ':' so
#  `cuda:0` counts as native while `cpu` does not.
NON_NATIVE_DEVICES = {"cpu", "meta", ""}


@dataclass
class Violation:
    rule: str
    location: str
    detail: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "location": self.location, "detail": self.detail}


@dataclass
class Summary:
    events: int = 0
    stages: int = 0
    checks: int = 0
    by_outcome: dict = field(default_factory=dict)
    demo_checks: int = 0
    by_tier: dict = field(default_factory=dict)
    gate_skips: list = field(default_factory=list)
    demo_skips: list = field(default_factory=list)
    xfails: list = field(default_factory=list)
    violations: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "events": self.events,
            "stages": self.stages,
            "checks": self.checks,
            "by_outcome": self.by_outcome,
            "demo_checks": self.demo_checks,
            "by_tier": self.by_tier,
            "gate_skips": self.gate_skips,
            "demo_skips": self.demo_skips,
            "xfails": self.xfails,
            "violations": [v.to_dict() for v in self.violations],
        }


def _check_oracle_tier(event: dict, test: str, where: str, summary: Summary) -> None:
    """Rejects a receipt that claims more oracle authority than it exercised.

    Silence is allowed: most receipts have no oracle at all. What is not allowed is claiming the
    native tier over a capture that says otherwise, because that is precisely how a CPU smoke run
    would get filed as native-golden evidence.
    """
    tier = event.get("oracle_tier")
    capture = event.get("fixture", {}).get("capture") if isinstance(event.get("fixture"), dict) else None

    if tier is not None:
        if tier not in ORACLE_TIERS:
            summary.violations.append(
                Violation("oracle-tier-known", where, f"unknown oracle_tier {tier!r} for {test!r}")
            )
            return
        summary.by_tier[tier] = summary.by_tier.get(tier, 0) + 1

    if capture is None:
        if tier == NATIVE_TIER:
            #  The strongest claim in the stream, with nothing behind it.
            summary.violations.append(
                Violation(
                    "native-claim-needs-capture",
                    where,
                    f"{test!r} claims oracle_tier {NATIVE_TIER!r} but carries no fixture.capture "
                    "recording the device and dtype it actually ran on",
                )
            )
        return

    if not isinstance(capture, dict):
        summary.violations.append(
            Violation("capture-complete", where, f"fixture.capture for {test!r} is not an object")
        )
        return

    missing = [
        name
        for name in ("oracle_class", "device", "dtype")
        if not str(capture.get(name) or "").strip()
    ]
    if missing:
        #  CaptureProvenance::new is all-or-nothing on the Rust side; this catches a receipt
        #  that reached the stream by some other route with a half-filled identity.
        summary.violations.append(
            Violation(
                "capture-complete",
                where,
                f"fixture.capture for {test!r} is missing {', '.join(missing)}",
            )
        )
        return

    oracle_class = str(capture["oracle_class"]).strip()
    device = str(capture["device"]).strip().lower()
    device_kind = device.split(":", 1)[0]

    if tier is not None and oracle_class != tier:
        summary.violations.append(
            Violation(
                "tier-matches-capture",
                where,
                f"{test!r} claims oracle_tier {tier!r} but its capture records "
                f"oracle_class {oracle_class!r}",
            )
        )

    claims_native = tier == NATIVE_TIER or oracle_class == NATIVE_TIER
    if claims_native and device_kind in NON_NATIVE_DEVICES:
        summary.violations.append(
            Violation(
                "native-claim-needs-native-device",
                where,
                f"{test!r} claims the {NATIVE_TIER!r} tier but ran on device {device!r}; "
                "a CPU green is not native-golden proof (frankentts-rf4: CPU-smoke is non-golden)",
            )
        )


def summarize(lines: list[str], origin: str = "<receipts>") -> Summary:
    """Validates and tallies one receipt stream.

    Every rule below has a name, because "the receipts are bad" is not an actionable failure
    and this output is read at 3am by whoever the gate just stopped.
    """
    summary = Summary()

    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        summary.events += 1
        where = f"{origin}:{number}"

        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            summary.violations.append(
                Violation("parse", where, f"not one JSON object per line: {exc}")
            )
            continue
        if not isinstance(event, dict):
            summary.violations.append(
                Violation("parse", where, f"expected a JSON object, got {type(event).__name__}")
            )
            continue

        kind = event.get("event")
        if kind not in EVENTS:
            summary.violations.append(
                Violation("event-kind", where, f"unknown event kind {kind!r}")
            )
            continue
        if kind == "stage":
            summary.stages += 1
            continue

        summary.checks += 1
        test = event.get("test") or ""
        contract = event.get("contract") or ""
        outcome = event.get("outcome")
        is_demo = contract.startswith(DEMO_CONTRACT_PREFIX)
        if is_demo:
            summary.demo_checks += 1

        if not test:
            summary.violations.append(
                Violation("named-test", where, "receipt has no test name; it cannot be traced back")
            )
        if outcome not in OUTCOMES:
            summary.violations.append(
                Violation("outcome-known", where, f"unknown outcome {outcome!r} for {test!r}")
            )
            continue

        summary.by_outcome[outcome] = summary.by_outcome.get(outcome, 0) + 1
        reason = event.get("reason") or ""
        entry = {"test": test, "contract": contract or None, "reason": reason}

        if outcome == "skipped":
            if not reason.strip():
                #  An unexplained skip is indistinguishable from a disabled test. The Rust
                #  `Receipt::emit` refuses to write one; this catches a receipt that reached
                #  the file by any other route.
                summary.violations.append(
                    Violation("skip-has-reason", where, f"skip receipt for {test!r} has no reason")
                )
            (summary.demo_skips if is_demo else summary.gate_skips).append(entry)
        elif outcome == "xfail":
            summary.xfails.append(entry)

        _check_oracle_tier(event, test, where, summary)

    if summary.checks == 0:
        #  The emission path going dead is the silent way this whole mechanism stops working:
        #  no receipts, no skips reported, a permanently reassuring green.
        summary.violations.append(
            Violation(
                "stream-not-empty",
                origin,
                "no contract_check receipts in the stream — either no test emitted one or "
                "FTTS_RECEIPTS never reached the test binaries",
            )
        )
    return summary


def render(summary: Summary) -> str:
    out: list[str] = []
    counts = " | ".join(
        f"{name} {summary.by_outcome.get(name, 0)}"
        for name in ("passed", "failed", "skipped", "xfail")
    )
    out.append(
        f"receipts: {summary.events} events "
        f"({summary.checks} contract_check, {summary.stages} stage)"
    )
    out.append(f"  {counts}")
    if summary.by_tier:
        tiers = " | ".join(f"{name} {count}" for name, count in sorted(summary.by_tier.items()))
        out.append(f"  oracle tiers: {tiers}")
    if summary.demo_checks:
        out.append(
            f"  {summary.demo_checks} from the convention demo "
            f"(Demo/*) — illustrative, not gate signal"
        )
    for entry in summary.xfails:
        out.append(f"  XFAIL {entry['test']}: {entry['reason']}")
    for entry in summary.gate_skips:
        contract = entry["contract"] or "(no contract)"
        out.append(f"  SKIPPED {entry['test']} [{contract}]: {entry['reason']}")
    for violation in summary.violations:
        out.append(f"  VIOLATION [{violation.rule}] {violation.location}: {violation.detail}")
    return "\n".join(out)


#  (description, stream, rule expected to fire). Each case is the receipt a real defect would
#  leave behind, so the checks cannot rot into no-ops unnoticed.
_SELFTEST_CASES: list[tuple[str, list[str], str | None]] = [
    (
        "a well-formed stream with an honest skip passes",
        [
            json.dumps({"event": "contract_check", "test": "t_a", "outcome": "passed"}),
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "t_b",
                    "outcome": "skipped",
                    "contract": "ConformanceExact/L4",
                    "reason": "FTTS_MODEL_DIR is unset or empty",
                }
            ),
            json.dumps({"event": "stage", "stage": "codec", "elapsed_ms": 1.0}),
        ],
        None,
    ),
    (
        "a truncated line is not silently dropped",
        [json.dumps({"event": "contract_check", "test": "t", "outcome": "passed"})[:-3]],
        "parse",
    ),
    (
        "an unknown event kind is rejected",
        [json.dumps({"event": "freeform_note", "text": "hello"})],
        "event-kind",
    ),
    (
        "an unknown outcome cannot sneak past the tally",
        [json.dumps({"event": "contract_check", "test": "t", "outcome": "probably_fine"})],
        "outcome-known",
    ),
    (
        "an unexplained skip is a disabled test",
        [json.dumps({"event": "contract_check", "test": "t", "outcome": "skipped"})],
        "skip-has-reason",
    ),
    (
        "an unexplained skip is rejected even under a Demo/ contract",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "t",
                    "outcome": "skipped",
                    "contract": "Demo/ModelGate",
                }
            )
        ],
        "skip-has-reason",
    ),
    (
        "a receipt with no test name cannot be traced back",
        [json.dumps({"event": "contract_check", "outcome": "passed"})],
        "named-test",
    ),
    (
        "a stream with no contract checks means the emission path is dead",
        [json.dumps({"event": "stage", "stage": "codec", "elapsed_ms": 1.0})],
        "stream-not-empty",
    ),
    (
        "an honest cpu-fallback receipt passes",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "contract_a_l2_talker",
                    "outcome": "passed",
                    "oracle_tier": "cpu_fp32_fallback",
                    "fixture": {
                        "source": "fixtures/l2.npz",
                        "capture": {
                            "oracle_class": "cpu_fp32_fallback",
                            "device": "cpu",
                            "dtype": "float32",
                        },
                    },
                }
            )
        ],
        None,
    ),
    (
        "an honest native receipt on cuda:0 passes",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "contract_a_l2_talker",
                    "outcome": "passed",
                    "oracle_tier": "native_cuda",
                    "fixture": {
                        "source": "fixtures/l2.npz",
                        "capture": {
                            "oracle_class": "native_cuda",
                            "device": "cuda:0",
                            "dtype": "bfloat16",
                        },
                    },
                }
            )
        ],
        None,
    ),
    (
        "a native claim executed on the CPU is not native-golden proof",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "contract_a_l2_talker",
                    "outcome": "passed",
                    "oracle_tier": "native_cuda",
                    "fixture": {
                        "source": "fixtures/l2.npz",
                        "capture": {
                            "oracle_class": "native_cuda",
                            "device": "cpu",
                            "dtype": "float32",
                        },
                    },
                }
            )
        ],
        "native-claim-needs-native-device",
    ),
    (
        "a tier that disagrees with its own capture is rejected",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "contract_a_l2_talker",
                    "outcome": "passed",
                    "oracle_tier": "native_cuda",
                    "fixture": {
                        "source": "fixtures/l2.npz",
                        "capture": {
                            "oracle_class": "cpu_fp32_fallback",
                            "device": "cuda:0",
                            "dtype": "float32",
                        },
                    },
                }
            )
        ],
        "tier-matches-capture",
    ),
    (
        "a native claim with no capture at all is rejected",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "contract_a_l2_talker",
                    "outcome": "passed",
                    "oracle_tier": "native_cuda",
                }
            )
        ],
        "native-claim-needs-capture",
    ),
    (
        "a half-filled capture identity is rejected",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "contract_a_l2_talker",
                    "outcome": "passed",
                    "fixture": {
                        "source": "fixtures/l2.npz",
                        "capture": {"oracle_class": "native_cuda", "device": "", "dtype": ""},
                    },
                }
            )
        ],
        "capture-complete",
    ),
    (
        "an unknown oracle tier cannot sneak past the tally",
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "contract_a_l2_talker",
                    "outcome": "passed",
                    "oracle_tier": "native_metal",
                }
            )
        ],
        "oracle-tier-known",
    ),
    (
        "a receipt with no oracle claim at all is fine",
        [
            json.dumps(
                {"event": "contract_check", "test": "contract_a_l0_tok", "outcome": "passed"}
            )
        ],
        None,
    ),
    (
        "an empty file means the emission path is dead",
        [],
        "stream-not-empty",
    ),
]


def selftest() -> tuple[bool, list[dict]]:
    results: list[dict] = []
    all_ok = True
    for description, stream, expected in _SELFTEST_CASES:
        summary = summarize(stream, origin="selftest")
        fired = [v.rule for v in summary.violations]
        ok = (not fired) if expected is None else (expected in fired)
        all_ok = all_ok and ok
        results.append(
            {"case": description, "expected_rule": expected, "ok": ok, "observed": fired}
        )

    #  A demo skip must be visible but must not become gate signal, or the banner goes
    #  permanently yellow and stops meaning anything.
    demo_only = summarize(
        [
            json.dumps(
                {
                    "event": "contract_check",
                    "test": "demo_t",
                    "outcome": "skipped",
                    "contract": "Demo/ModelGate",
                    "reason": "no model directory supplied",
                }
            )
        ],
        origin="selftest",
    )
    ok = demo_only.ok and not demo_only.gate_skips and len(demo_only.demo_skips) == 1
    all_ok = all_ok and ok
    results.append(
        {
            "case": "a Demo/ skip is counted separately, not as a gate skip",
            "expected_rule": None,
            "ok": ok,
            "observed": [v.rule for v in demo_only.violations] or ["gate_skips=%d" % len(demo_only.gate_skips)],
        }
    )
    return all_ok, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="summarize_receipts.py", description=__doc__)
    parser.add_argument("receipts", nargs="?", help="NDJSON receipt file written by FTTS_RECEIPTS")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--skip-summary-file",
        metavar="PATH",
        help="write one line per gate-relevant skip, for check.sh to fold into its banner",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="prove each honesty rule still fires against fixture streams",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        ok, results = selftest()
        if args.json:
            print(json.dumps({"ok": ok, "cases": results}, indent=2))
        else:
            for entry in results:
                mark = "ok  " if entry["ok"] else "FAIL"
                expected = entry["expected_rule"] or "(none)"
                print(f"{mark} {entry['case']}  -> expected {expected}, got {entry['observed']}")
            passed = sum(1 for e in results if e["ok"])
            print(f"\nsummarize_receipts selftest: {passed}/{len(results)} cases ok")
        return 0 if ok else 1

    if not args.receipts:
        parser.error("a receipts file is required unless --selftest is given")

    path = Path(args.receipts)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        message = f"cannot read {path}: {exc}"
        print(json.dumps({"error": message}) if args.json else message, file=sys.stderr)
        return 2

    summary = summarize(lines, origin=str(path))

    if args.skip_summary_file:
        report = "".join(
            f"{entry['test']} [{entry['contract'] or 'no contract'}]: {entry['reason']}\n"
            for entry in summary.gate_skips
        )
        Path(args.skip_summary_file).write_text(report, encoding="utf-8")

    print(json.dumps(summary.to_dict(), indent=2) if args.json else render(summary))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())
