#!/usr/bin/env python3
"""franken_tts listening-panel harness — plan, run, analyse and gate on a listening panel.

    run_panel.py plan      --manifest M --instance I --out DIR
    run_panel.py simulate  --plan DIR --manifest M --out responses.jsonl [--identity-effect X ...]
    run_panel.py analyze   --plan DIR --manifest M --responses R [--objective O] --out verdict.json
    run_panel.py gate      --verdict verdict.json
    run_panel.py selftest  [--out DIR]

`gate` is the release binding: it exits non-zero unless every named bit for the instance is
`pass`, and it refuses to pass a verdict produced by a synthetic panel or under provisional
margins unless that is explicitly allowed. `selftest` is the standing CI proof that this
harness still detects degradation — a harness that only ever says PASS is worthless.

Exit codes (stable, quoted by docs/CONFORMANCE_AND_LISTENING.md):
    0  PASS
    1  FAIL                 — a real difference was detected
    2  INSUFFICIENT_POWER   — the panel could not decide; NOT a pass
    3  INVALID              — design violated, or a synthetic/provisional verdict at a gate
    4  usage / input error

Bead: frankentts-v-listening-25m.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path
from typing import Any, Literal, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import equivalence as eq  # noqa: E402
import protocol as pr  # noqa: E402

VERDICT_SCHEMA_VERSION = "1.0.0"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INSUFFICIENT = 2
EXIT_INVALID = 3
EXIT_USAGE = 4

BitState = Literal["pass", "fail", "insufficient_power", "not_required"]


# --------------------------------------------------------------------------------------
# Policy loading
# --------------------------------------------------------------------------------------


def load_margins(path: Path) -> dict[str, Any]:
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0.0":
        raise pr.ProtocolError(f"margins schema {raw.get('schema_version')!r} unsupported")
    for required in ("owner", "statistics", "design", "screening", "families", "instances"):
        if required not in raw:
            raise pr.ProtocolError(f"margins file is missing the [{required}] section")
    if not raw["owner"].get("name"):
        raise pr.ProtocolError("margins [owner].name must name a person, not a team")
    return raw


def instance_config(margins: dict[str, Any], instance: str) -> dict[str, Any]:
    try:
        return margins["instances"][instance]
    except KeyError:
        known = ", ".join(sorted(margins["instances"]))
        raise pr.ProtocolError(f"unknown protocol instance {instance!r}; known: {known}") from None


# --------------------------------------------------------------------------------------
# Observation extraction
# --------------------------------------------------------------------------------------


def _panel_observations(
    family: str,
    spec: dict[str, Any],
    key: dict[str, dict[str, Any]],
    responses: Sequence[dict],
    manifest: pr.Manifest,
    kept_listeners: set[str],
) -> list[dict[str, Any]]:
    """One record per usable trial: the paired contrast plus every grouping factor."""
    roles = {i.item_id: i.role for i in manifest.items}
    out: list[dict[str, Any]] = []
    for resp in responses:
        trial = key.get(resp["trial_id"])
        if trial is None or trial["is_catch"] or trial["family"] != family:
            continue
        if trial["listener_id"] not in kept_listeners:
            continue
        slots: dict[str, str] = trial["slots"]
        if spec["kind"] == "identity_abx":
            choice = resp.get("choice")
            if choice not in slots:
                raise pr.ProtocolError(
                    f"trial {trial['trial_id']}: choice {choice!r} is not one of {sorted(slots)}"
                )
            value = 1.0 if roles.get(slots[choice]) == "candidate" else 0.0
        elif spec["kind"] == "mushra":
            ratings = resp.get("ratings") or {}
            cand = next((s for s, i in slots.items() if roles.get(i) == "candidate"), None)
            inc = next((s for s, i in slots.items() if roles.get(i) == "incumbent"), None)
            if cand is None or inc is None:
                raise pr.ProtocolError(
                    f"trial {trial['trial_id']} lacks a candidate/incumbent pair for MUSHRA"
                )
            if cand not in ratings or inc not in ratings:
                continue  # partially rated trial: dropped, and counted in the coverage report
            value = float(ratings[cand]) - float(ratings[inc])
        else:
            raise pr.ProtocolError(f"family {family!r} has unsupported panel kind {spec['kind']!r}")
        out.append(
            {
                "value": value,
                "listener_id": trial["listener_id"],
                "speaker_id": trial["speaker_id"],
                "text_id": trial["text_id"],
                "language": trial["language"],
                "regime": trial["regime"],
                "axes": list(trial["axes"]),
                "cell": "|".join(
                    (trial["speaker_id"], trial["text_id"], trial["language"], trial["regime"])
                ),
                "item": trial["trial_id"],
            }
        )
    return out


def _objective_observations(family_metric: str, objective: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in objective.get("metrics", []):
        if row.get("metric") != family_metric:
            continue
        out.append(
            {
                "value": float(row["candidate"]) - float(row["incumbent"]),
                "listener_id": "objective",
                "speaker_id": row["speaker_id"],
                "text_id": row["text_id"],
                "language": row.get("language", "und"),
                "regime": row.get("regime", "short"),
                "axes": list(row.get("axes", [])),
                "cell": "|".join(
                    (
                        row["speaker_id"],
                        row["text_id"],
                        row.get("language", "und"),
                        row.get("regime", "short"),
                    )
                ),
                "item": row["item_id"],
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Family analysis
# --------------------------------------------------------------------------------------

_CLUSTER_KEY = {
    "by_listener": "listener_id",
    "by_speaker": "speaker_id",
    "by_text": "text_id",
    "by_language": "language",
}


def analyze_family(
    family: str,
    spec: dict[str, Any],
    observations: Sequence[dict[str, Any]],
    *,
    stats_cfg: dict[str, Any],
    required_axes: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    alpha = float(stats_cfg["alpha"])
    power = float(stats_cfg["power"])
    center = float(spec.get("center", 0.0))
    margin = float(spec["margin"])
    cluster_names = spec.get("cluster_analyses") or stats_cfg["cluster_analyses"]

    report: dict[str, Any] = {
        "label": spec["label"],
        "bit": spec["bit"],
        "source": spec["source"],
        "test": spec["test"],
        "center": center,
        "margin": margin,
        "sesoi_rationale": spec.get("sesoi_rationale", ""),
        "n_observations": len(observations),
        "clusters": {},
        "tail": {},
        "notes": [],
    }
    if not observations:
        report["decision"] = "INSUFFICIENT_POWER"
        report["notes"].append("NO_OBSERVATIONS: family has no usable data")
        return report

    decisions: list[str] = []
    for name in cluster_names:
        field = _CLUSTER_KEY.get(name)
        if field is None:
            raise pr.ProtocolError(f"unknown cluster analysis {name!r}")
        grouped = eq.group_by(observations, field)
        means = [math.fsum(o["value"] for o in rows) / len(rows) for rows in grouped.values()]
        if len(means) < 2:
            report["clusters"][name] = {
                "decision": "INSUFFICIENT_POWER",
                "notes": [f"only {len(means)} cluster(s) on {field}"],
            }
            decisions.append("INSUFFICIENT_POWER")
            continue
        if spec["test"] == "equivalence":
            result = eq.tost(means, center=center, margin=margin, alpha=alpha, power=power)
        else:
            result = eq.non_inferiority(
                means,
                center=center,
                margin=margin,
                worse_is=spec["worse_is"],
                alpha=alpha,
                power=power,
            )
        icc = eq.icc_oneway([[o["value"] for o in rows] for rows in grouped.values()])
        entry = result.to_dict()
        entry["cluster_field"] = field
        entry["icc"] = icc.to_dict()
        report["clusters"][name] = entry
        decisions.append(result.decision)

    if "FAIL_DIFFERENT" in decisions:
        report["decision"] = "FAIL_DIFFERENT"
    elif all(d == "PASS_EQUIVALENT" for d in decisions):
        report["decision"] = "PASS_EQUIVALENT"
    else:
        report["decision"] = "INSUFFICIENT_POWER"

    tail_cfg = spec.get("tail")
    if tail_cfg:
        unit_field = {"cell": "cell", "item": "item", "speaker": "speaker_id"}[tail_cfg["unit"]]
        scopes: list[tuple[str, list[dict[str, Any]]]] = [("overall", list(observations))]
        for axis in required_axes:
            scopes.append((axis, [o for o in observations if axis in o["axes"]]))
        for scope, rows in scopes:
            if scope != "overall" and len(rows) < int(tail_cfg["min_items_per_axis"]):
                report["tail"][scope] = {
                    "scope": scope,
                    "n_observations": len(rows),
                    "decision": "INSUFFICIENT_DATA",
                    "notes": [
                        f"REQUIRED_AXIS_UNDER_SAMPLED: {len(rows)} observations < "
                        f"{tail_cfg['min_items_per_axis']}"
                    ],
                }
                continue
            result = eq.tail_gate(
                [(o[unit_field], o["value"]) for o in rows],
                scope=scope,
                center=center,
                alpha=float(tail_cfg["alpha"]),
                tail=tail_cfg["tail"],
                min_obs_per_unit=int(tail_cfg["min_obs_per_unit"]),
                max_dropped_unit_fraction=float(tail_cfg["max_dropped_unit_fraction"]),
                null_permutations=int(tail_cfg["null_permutations"]),
                null_quantile=float(tail_cfg["null_quantile"]),
                slack=float(tail_cfg["slack"]),
                seed=seed,
            )
            report["tail"][scope] = result.to_dict()

    return report


# --------------------------------------------------------------------------------------
# Verdict assembly
# --------------------------------------------------------------------------------------

_DECISION_TO_BIT: dict[str, BitState] = {
    "PASS_EQUIVALENT": "pass",
    "FAIL_DIFFERENT": "fail",
    "INSUFFICIENT_POWER": "insufficient_power",
}
#  Ordering matters: a family that reports `pass` must be able to overwrite the initial
#  `not_required`, while `insufficient_power` and `fail` dominate everything below them.
_BIT_SEVERITY = {"not_required": 0, "pass": 1, "insufficient_power": 2, "fail": 3}


def _worst(a: BitState, b: BitState) -> BitState:
    return a if _BIT_SEVERITY[a] >= _BIT_SEVERITY[b] else b


def build_verdict(
    *,
    margins: dict[str, Any],
    instance: str,
    plan_meta: dict[str, Any],
    family_reports: dict[str, dict[str, Any]],
    screening: dict[str, pr.ScreeningOutcome],
    design: dict[str, Any],
    synthetic: bool,
) -> dict[str, Any]:
    cfg = instance_config(margins, instance)
    bits: dict[str, BitState] = {"design_valid": "pass" if design["valid"] else "fail"}

    tail_bit: BitState = "not_required"
    for report in family_reports.values():
        bit_name = report["bit"]
        state = _DECISION_TO_BIT.get(report.get("decision", ""), "insufficient_power")
        bits[bit_name] = _worst(bits.get(bit_name, "not_required"), state)
        for scope_report in report.get("tail", {}).values():
            decision = scope_report.get("decision")
            state = {
                "PASS": "pass",
                "FAIL": "fail",
                "INSUFFICIENT_DATA": "insufficient_power",
            }.get(decision, "insufficient_power")
            tail_bit = _worst(tail_bit, state)  # type: ignore[arg-type]
    bits["tail_cvar_bound"] = tail_bit

    if design["valid"] and all(b in ("pass", "not_required") for b in bits.values()):
        overall = "PASS"
    elif not design["valid"]:
        overall = "INVALID"
    elif any(b == "fail" for b in bits.values()):
        overall = "FAIL"
    else:
        overall = "INSUFFICIENT_POWER"

    return {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "policy_version": margins["policy_version"],
        "calibration_status": margins["calibration_status"],
        "owner": margins["owner"]["name"],
        "instance": instance,
        "instance_label": cfg["label"],
        "blocks_release": bool(cfg.get("blocks_release", False)),
        "objective_only": bool(cfg.get("objective_only", False)),
        "consumers": list(cfg.get("consumers", [])),
        "corpus_id": plan_meta.get("corpus_id"),
        "incumbent": plan_meta.get("incumbent"),
        "candidate": plan_meta.get("candidate"),
        "seed": plan_meta.get("seed"),
        "synthetic_panel": synthetic,
        "is_quality_claim": not synthetic,
        "bits": bits,
        "overall": overall,
        "design": design,
        "screening": {
            "kept": sorted(s.listener_id for s in screening.values() if s.kept),
            "rejected": [s.to_dict() for s in screening.values() if not s.kept],
        },
        "families": family_reports,
    }


def validate_design(
    margins: dict[str, Any],
    cfg: dict[str, Any],
    *,
    kept_listeners: set[str],
    observations_by_family: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    design_cfg = margins["design"]
    problems: list[str] = []
    panel_families = [
        f for f in cfg["families"] if margins["families"][f]["source"] == "panel"
    ]
    objective_families = [
        f for f in cfg["families"] if margins["families"][f]["source"] == "objective"
    ]

    if panel_families and not cfg.get("objective_only", False):
        if len(kept_listeners) < int(design_cfg["panel_size_post_screen"]):
            problems.append(
                f"panel size after screening {len(kept_listeners)} < required "
                f"{design_cfg['panel_size_post_screen']}"
            )
    if cfg.get("objective_only", False) and panel_families:
        problems.append(
            "objective_only instance declares panel families; a Tier-0 screen may not "
            "borrow a human bit"
        )

    # Coverage is checked per family, not pooled: a large objective corpus must not paper
    # over a panel that only ever heard three speakers.
    speakers: set[str] = set()
    texts: set[str] = set()
    languages: set[str] = set()
    for family in cfg["families"]:
        rows = observations_by_family.get(family, [])
        fam_speakers = {o["speaker_id"] for o in rows}
        fam_texts = {o["text_id"] for o in rows}
        fam_languages = {o["language"] for o in rows}
        speakers |= fam_speakers
        texts |= fam_texts
        languages |= fam_languages
        if len(fam_speakers) < int(design_cfg["min_speakers"]):
            problems.append(
                f"family {family}: {len(fam_speakers)} speakers < required "
                f"{design_cfg['min_speakers']}"
            )
        if len(fam_texts) < int(design_cfg["min_texts"]):
            problems.append(
                f"family {family}: {len(fam_texts)} texts < required {design_cfg['min_texts']}"
            )
        if len(fam_languages) < int(design_cfg["min_languages"]):
            problems.append(
                f"family {family}: {len(fam_languages)} languages < required "
                f"{design_cfg['min_languages']}"
            )

    for family in objective_families:
        n_items = len({o["item"] for o in observations_by_family.get(family, [])})
        if n_items < int(design_cfg["min_objective_utterances"]):
            problems.append(
                f"objective family {family}: {n_items} utterances < required "
                f"{design_cfg['min_objective_utterances']}"
            )

    return {
        "valid": not problems,
        "problems": problems,
        "n_listeners_kept": len(kept_listeners),
        "n_speakers": len(speakers),
        "n_texts": len(texts),
        "languages": sorted(languages),
        "required_axes": list(cfg.get("required_axes", [])),
    }


# --------------------------------------------------------------------------------------
# Demo corpus (selftest + onboarding new consumers)
# --------------------------------------------------------------------------------------

#  Two texts per canary axis so that every axis clears `min_obs_per_unit` and `min_units`
#  in the tail gate at the design's panel size. An axis carried by a single text cannot be
#  tail-tested; that is a corpus-design constraint, not a harness limitation.
_DEMO_AXES = {
    "t03": ("sibilants",),
    "t04": ("sibilants",),
    "t05": ("numbers",),
    "t06": ("numbers",),
    "t07": ("breaths",),
    "t08": ("breaths",),
    "t09": ("code_switching",),
    "t10": ("code_switching",),
    "t11": ("noisy_reference",),
    "t12": ("noisy_reference",),
    "t13": ("long_form",),
    "t14": ("long_form",),
}


def write_demo_manifest(path: Path, *, n_speakers: int = 8, n_texts: int = 16) -> Path:
    items: list[dict[str, Any]] = []
    for s in range(n_speakers):
        speaker = f"spk{s + 1:02d}"
        for t in range(n_texts):
            text = f"t{t + 1:02d}"
            language = "en" if t % 2 == 0 else "zh"
            axes = list(_DEMO_AXES.get(text, ()))
            regime = "long" if "long_form" in axes else "short"
            for role, system in (
                ("reference", "natural"),
                ("incumbent", "q8_baseline"),
                ("candidate", "q4_mtp_depths_8_15"),
                ("anchor_low", "anchor_lp3500"),
            ):
                items.append(
                    {
                        "item_id": f"{speaker}_{text}_{role}",
                        "role": role,
                        "system": system,
                        "speaker_id": speaker,
                        "text_id": text,
                        "language": language,
                        "regime": regime,
                        "axes": axes,
                        "path": f"demo/{speaker}_{text}_{role}.wav",
                    }
                )
        items.append(
            {
                "item_id": f"{speaker}_foil",
                "role": "foil",
                "system": "q8_baseline",
                "speaker_id": speaker,
                "text_id": "t01",
                "language": "en",
                "regime": "short",
                "axes": [],
                "path": f"demo/{speaker}_foil.wav",
            }
        )
    payload = {
        "manifest_version": pr.SCHEMA_VERSION,
        "comparison": {"incumbent": "q8_baseline", "candidate": "q4_mtp_depths_8_15"},
        "items": items,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_demo_objective(
    path: Path,
    *,
    seed: int,
    n_speakers: int = 12,
    n_texts: int = 20,
    wer_effect: float = 0.0,
    drift_effect: float = 0.0,
) -> Path:
    """Objective (Tier-0) metrics over an independent, larger corpus than the panel hears.

    Automated scoring is cheap, so the nightly corpus is deliberately wider than the subset a
    human panel can sit through; the design validator checks each family's coverage separately
    so the wide objective corpus cannot vouch for a narrow panel.
    """
    import random

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for s in range(n_speakers):
        speaker = f"spk{s + 1:02d}"
        for t in range(n_texts):
            text = f"t{t + 1:02d}"
            language = "en" if t % 2 == 0 else "zh"
            axes = list(_DEMO_AXES.get(text, ()))
            regime = "long" if t >= n_texts - 4 else "short"
            if regime == "long":
                axes = sorted(set(axes) | {"long_form"})
            item_id = f"{speaker}_{text}"
            common = {
                "item_id": item_id,
                "speaker_id": speaker,
                "text_id": text,
                "language": language,
                "regime": regime,
                "axes": axes,
            }
            base_wer = abs(rng.gauss(0.030, 0.010))
            rows.append(
                {
                    "metric": "wer",
                    **common,
                    "incumbent": round(base_wer, 5),
                    "candidate": round(
                        max(0.0, base_wer + wer_effect + rng.gauss(0.0, 0.008)), 5
                    ),
                }
            )
            base_rs = abs(rng.gauss(0.004, 0.002))
            rows.append(
                {
                    "metric": "repeat_skip_rate",
                    **common,
                    "incumbent": round(base_rs, 5),
                    "candidate": round(max(0.0, base_rs + rng.gauss(0.0, 0.002)), 5),
                }
            )
            base_drift = abs(rng.gauss(0.05, 0.02))
            rows.append(
                {
                    "metric": "longform_drift",
                    **common,
                    "incumbent": round(base_drift, 5),
                    "candidate": round(
                        max(0.0, base_drift + drift_effect + rng.gauss(0.0, 0.015)), 5
                    ),
                }
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": "1.0.0", "metrics": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------------------
# Sub-commands
# --------------------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    margins = load_margins(args.margins)
    cfg = instance_config(margins, args.instance)
    manifest = pr.load_manifest(args.manifest)
    design = margins["design"]
    families = [f for f in cfg["families"] if margins["families"][f]["source"] == "panel"]
    plan = pr.build_trial_plan(
        manifest,
        instance=args.instance,
        families=families,
        n_listeners=args.listeners or int(design["recruit_target"]),
        trials_per_listener=args.trials or int(design["trials_per_listener_per_family"]),
        catch_trials_per_listener=int(design["catch_trials_per_listener"]),
        seed=args.seed,
    )
    paths = pr.write_plan(plan, args.out)
    print(
        json.dumps(
            {
                "instance": args.instance,
                "n_trials": len(plan.trials),
                "n_listeners": len(plan.listeners),
                **{k: str(v) for k, v in paths.items()},
            },
            indent=2,
        )
    )
    return EXIT_PASS


def cmd_simulate(args: argparse.Namespace) -> int:
    manifest = pr.load_manifest(args.manifest)
    plan_meta = json.loads((Path(args.plan) / "plan.json").read_text(encoding="utf-8"))
    key = pr.load_key(Path(args.plan) / "trials.key.json")
    trials = [
        pr.Trial(
            trial_id=t["trial_id"],
            listener_id=t["listener_id"],
            kind=t["kind"],
            family=t["family"],
            is_catch=t["is_catch"],
            speaker_id=t["speaker_id"],
            text_id=t["text_id"],
            language=t["language"],
            regime=t["regime"],
            axes=tuple(t["axes"]),
            reference_item=t["reference_item"],
            slots=t["slots"],
        )
        for t in key.values()
    ]
    plan = pr.TrialPlan(
        instance=plan_meta["instance"],
        seed=plan_meta["seed"],
        corpus_id=plan_meta["corpus_id"],
        incumbent=plan_meta["incumbent"],
        candidate=plan_meta["candidate"],
        listeners=plan_meta["listeners"],
        trials=trials,
        design=plan_meta["design"],
    )
    model = pr.SyntheticPanelModel(
        identity_effect=args.identity_effect,
        naturalness_effect=args.naturalness_effect,
        bad_listener_rate=args.bad_listener_rate,
        axis_penalty=json.loads(args.axis_penalty) if args.axis_penalty else {},
    )
    responses = pr.simulate_responses(plan, manifest, model, seed=args.seed)
    pr.write_responses(responses, args.out)
    print(json.dumps({"responses": str(args.out), "n": len(responses), "synthetic": True}, indent=2))
    return EXIT_PASS


def cmd_analyze(args: argparse.Namespace) -> int:
    margins = load_margins(args.margins)
    cfg = instance_config(margins, args.instance)
    manifest = pr.load_manifest(args.manifest)
    plan_meta = json.loads((Path(args.plan) / "plan.json").read_text(encoding="utf-8"))
    key = pr.load_key(Path(args.plan) / "trials.key.json")
    responses = pr.read_responses(args.responses)
    if plan_meta["corpus_id"] != manifest.corpus_id:
        raise pr.ProtocolError(
            "manifest does not match the corpus the plan was built from; refusing to analyse"
        )

    roles_by_item = {i.item_id: i.role for i in manifest.items}
    screening = pr.screen_listeners(
        responses,
        key,
        roles_by_item,
        catch_min_correct_rate=float(margins["screening"]["catch_min_correct_rate"]),
        hidden_ref_min_score=float(margins["screening"]["hidden_ref_min_score"]),
        hidden_ref_max_violation_rate=float(margins["screening"]["hidden_ref_max_violation_rate"]),
        anchor_max_score=float(margins["screening"]["anchor_max_score"]),
        anchor_max_violation_rate=float(margins["screening"]["anchor_max_violation_rate"]),
    )
    kept = {s.listener_id for s in screening.values() if s.kept}

    objective: dict[str, Any] = {}
    if args.objective:
        objective = json.loads(Path(args.objective).read_text(encoding="utf-8"))

    observations: dict[str, list[dict[str, Any]]] = {}
    for family in cfg["families"]:
        spec = margins["families"][family]
        if spec["source"] == "panel":
            observations[family] = _panel_observations(
                family, spec, key, responses, manifest, kept
            )
        else:
            if not objective:
                raise pr.ProtocolError(
                    f"instance {args.instance!r} requires objective family {family!r} but no "
                    "--objective metrics file was supplied"
                )
            observations[family] = _objective_observations(spec["metric"], objective)

    required_axes = list(cfg.get("required_axes", []))
    reports = {
        family: analyze_family(
            family,
            margins["families"][family],
            observations[family],
            stats_cfg=margins["statistics"],
            required_axes=required_axes,
            seed=args.seed,
        )
        for family in cfg["families"]
    }
    design = validate_design(
        margins, cfg, kept_listeners=kept, observations_by_family=observations
    )
    synthetic = any(r.get("synthetic") for r in responses)
    verdict = build_verdict(
        margins=margins,
        instance=args.instance,
        plan_meta=plan_meta,
        family_reports=reports,
        screening=screening,
        design=design,
        synthetic=synthetic,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": str(args.out), "overall": verdict["overall"], "bits": verdict["bits"]}, indent=2))
    return EXIT_PASS


def evaluate_gate(
    verdict: dict[str, Any], *, allow_synthetic: bool, allow_provisional: bool
) -> tuple[int, list[str]]:
    reasons: list[str] = []
    if verdict.get("synthetic_panel") and not allow_synthetic:
        reasons.append(
            "verdict came from a SYNTHETIC panel; it validates the pipeline, not the audio"
        )
    if (
        verdict.get("calibration_status") == "PROVISIONAL"
        and verdict.get("blocks_release")
        and not allow_provisional
    ):
        reasons.append(
            "margins are PROVISIONAL and this instance blocks release; freeze the margins first"
        )
    if reasons:
        return EXIT_INVALID, reasons

    overall = verdict.get("overall")
    failed = [name for name, state in verdict["bits"].items() if state == "fail"]
    unpowered = [name for name, state in verdict["bits"].items() if state == "insufficient_power"]
    if overall == "PASS":
        return EXIT_PASS, []
    if overall == "INVALID":
        return EXIT_INVALID, verdict["design"]["problems"]
    if failed:
        return EXIT_FAIL, [f"bit {name} = fail" for name in failed]
    return EXIT_INSUFFICIENT, [f"bit {name} = insufficient_power" for name in unpowered]


def cmd_gate(args: argparse.Namespace) -> int:
    verdict = json.loads(Path(args.verdict).read_text(encoding="utf-8"))
    code, reasons = evaluate_gate(
        verdict, allow_synthetic=args.allow_synthetic, allow_provisional=args.allow_provisional
    )
    print(
        json.dumps(
            {
                "instance": verdict.get("instance"),
                "overall": verdict.get("overall"),
                "bits": verdict.get("bits"),
                "exit_code": code,
                "reasons": reasons,
            },
            indent=2,
        )
    )
    return code


# --------------------------------------------------------------------------------------
# Selftest — the standing proof that this harness still bites
# --------------------------------------------------------------------------------------


def _run_scenario(
    workdir: Path,
    margins_path: Path,
    *,
    name: str,
    instance: str,
    analysis_instances: Sequence[str],
    listeners: int,
    trials: int,
    identity_effect: float,
    naturalness_effect: float,
    axis_penalty: dict[str, float],
    axis_penalty_mushra: dict[str, float],
    wer_effect: float,
    listener_sd_identity: float,
    seed: int,
) -> dict[str, Any]:
    root = workdir / name
    manifest_path = write_demo_manifest(root / "manifest.json")
    manifest = pr.load_manifest(manifest_path)
    margins = load_margins(margins_path)
    cfg = instance_config(margins, instance)
    families = [f for f in cfg["families"] if margins["families"][f]["source"] == "panel"]

    plan = pr.build_trial_plan(
        manifest,
        instance=instance,
        families=families,
        n_listeners=listeners,
        trials_per_listener=trials,
        catch_trials_per_listener=int(margins["design"]["catch_trials_per_listener"]),
        seed=seed,
    )
    pr.write_plan(plan, root / "plan")
    model = pr.SyntheticPanelModel(
        identity_effect=identity_effect,
        naturalness_effect=naturalness_effect,
        axis_penalty=axis_penalty,
        axis_penalty_mushra=axis_penalty_mushra,
        listener_sd_identity=listener_sd_identity,
    )
    responses = pr.simulate_responses(plan, manifest, model, seed=seed + 1)
    pr.write_responses(responses, root / "responses.jsonl")
    objective_path = write_demo_objective(
        root / "objective.json", seed=seed + 2, wer_effect=wer_effect
    )

    verdicts: dict[str, Any] = {}
    for analysis_instance in analysis_instances:
        out = root / f"verdict.{analysis_instance}.json"
        cmd_analyze(
            argparse.Namespace(
                margins=margins_path,
                instance=analysis_instance,
                manifest=manifest_path,
                plan=root / "plan",
                responses=root / "responses.jsonl",
                objective=objective_path,
                out=out,
                seed=seed + 3,
            )
        )
        verdicts[analysis_instance] = json.loads(out.read_text(encoding="utf-8"))
    return verdicts


def cmd_selftest(args: argparse.Namespace) -> int:
    """Four scenarios with predeclared expected verdicts.

    The point is not that the harness produces output; it is that the harness produces the
    RIGHT output when the ground truth is known — including refusing to decide when the panel
    is too small, and failing on a canary axis whose damage is invisible in the pooled mean.
    """
    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)
    margins_path = Path(args.margins)
    design = load_margins(margins_path)["design"]
    # Recruit at the design's recruit_target so screening loss is absorbed exactly as a real
    # panel would absorb it; the design validator still demands panel_size_post_screen.
    full_listeners = int(design["recruit_target"])
    full_trials = int(design["trials_per_listener_per_family"])
    base = dict(
        listeners=full_listeners,
        trials=full_trials,
        identity_effect=0.0,
        naturalness_effect=0.0,
        axis_penalty={},
        axis_penalty_mushra={},
        wer_effect=0.0,
        listener_sd_identity=0.05,
    )

    scenarios: list[dict[str, Any]] = [
        {
            "name": "equivalent",
            "instance": "surgery_canary",
            "expect_overall": "PASS",
            "kwargs": {**base},
        },
        {
            "name": "equivalent_release_all_axes",
            "instance": "release",
            "expect_overall": "PASS",
            "kwargs": {**base},
        },
        {
            "name": "identity_degraded",
            "instance": "surgery_canary",
            "expect_overall": "FAIL",
            "expect_bit": ("identity_equivalence", "fail"),
            "kwargs": {**base, "identity_effect": -0.15},
        },
        {
            "name": "naturalness_degraded",
            "instance": "surgery_canary",
            "expect_overall": "FAIL",
            "expect_bit": ("naturalness_equivalence", "fail"),
            "kwargs": {**base, "naturalness_effect": 9.0},
        },
        {
            "name": "wer_regression",
            "instance": "surgery_canary",
            "expect_overall": "FAIL",
            "expect_bit": ("intelligibility_noninferiority", "fail"),
            "kwargs": {**base, "wer_effect": 0.02},
        },
        # A heterogeneous listener pool with a true effect of zero: the design is fully valid
        # and the point estimate sits on the centre, but the interval is too wide to declare
        # equivalence. The harness must say so rather than round a null result up to a pass.
        {
            "name": "underpowered",
            "instance": "surgery_canary",
            "expect_overall": "INSUFFICIENT_POWER",
            "expect_bit": ("identity_equivalence", "insufficient_power"),
            "kwargs": {**base, "listener_sd_identity": 0.30},
        },
        {
            "name": "canary_axis_only",
            "instance": "surgery_canary",
            "expect_overall": "FAIL",
            "expect_bit": ("tail_cvar_bound", "fail"),
            "kwargs": {**base, "axis_penalty": {"sibilants": 0.30}},
        },
    ]

    results: list[dict[str, Any]] = []
    ok = True
    for scenario in scenarios:
        instance = str(scenario["instance"])
        verdict = _run_scenario(
            workdir,
            margins_path,
            name=str(scenario["name"]),
            instance=instance,
            analysis_instances=[instance],
            seed=20260806,
            **scenario["kwargs"],
        )[instance]
        checks: list[str] = []
        passed = verdict["overall"] == scenario["expect_overall"]
        if not passed:
            checks.append(
                f"overall {verdict['overall']} != expected {scenario['expect_overall']}"
            )
        expect_bit = scenario.get("expect_bit")
        if expect_bit:
            bit, state = expect_bit
            if verdict["bits"].get(bit) != state:
                passed = False
                checks.append(f"bit {bit} = {verdict['bits'].get(bit)} != expected {state}")
        ok = ok and passed
        results.append(
            {
                "scenario": scenario["name"],
                "instance": instance,
                "expected": scenario["expect_overall"],
                "observed": verdict["overall"],
                "bits": verdict["bits"],
                "ok": passed,
                "failures": checks,
            }
        )

    summary = {"selftest": "listening-harness", "all_ok": ok, "scenarios": results}
    (workdir / "selftest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return EXIT_PASS if ok else EXIT_FAIL


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    default_margins = Path(__file__).resolve().parent / "margins.toml"
    parser = argparse.ArgumentParser(prog="run_panel.py", description=__doc__)
    parser.add_argument("--margins", type=Path, default=default_margins)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="build a blinded trial plan")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--listeners", type=int, default=0)
    p.add_argument("--trials", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260806)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("simulate", help="drive the plan with a seeded synthetic panel")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--identity-effect", type=float, default=0.0)
    p.add_argument("--naturalness-effect", type=float, default=0.0)
    p.add_argument("--bad-listener-rate", type=float, default=0.0)
    p.add_argument("--axis-penalty", default="")
    p.add_argument("--seed", type=int, default=20260806)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("analyze", help="produce the machine-readable verdict")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--responses", type=Path, required=True)
    p.add_argument("--objective", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260806)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("gate", help="enforce the verdict (the release binding)")
    p.add_argument("--verdict", type=Path, required=True)
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--allow-provisional", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("selftest", help="prove the harness still detects degradation")
    p.add_argument("--out", type=Path, default=Path("target/listening-selftest"))
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except pr.ProtocolError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return EXIT_USAGE
    except (OSError, KeyError, ValueError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
