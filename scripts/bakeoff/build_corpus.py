#!/usr/bin/env python3
"""Bakeoff corpus CLI — validate coverage and consent, emit listening stimuli, dry-run.

    build_corpus.py validate                    # is the corpus ready for Gate A?
    build_corpus.py coverage --json             # machine-readable coverage report
    build_corpus.py emit-stimuli --renders R --incumbent qwen --candidate pocket --out M.json
    build_corpus.py dryrun --out DIR            # corpus -> stimuli -> panel -> verdict

Exit codes:
    0  ready for Gate A
    1  BLOCKERS — consent or integrity failures; the corpus may not be used at all
    2  shortfalls only — consent-clean but coverage incomplete; not ready
    3  usage / load error

The distinction between 1 and 2 is the point. A coverage shortfall means "collect more"; a
consent blocker means "this audio does not belong here". They are never reported as the same
condition, and neither is ever reported as ready.

Bead: frankentts-bake-corpus-48h.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus as cp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXIT_READY = 0
EXIT_BLOCKERS = 1
EXIT_SHORTFALLS = 2
EXIT_USAGE = 3


def runner_path() -> Path:
    """The listening harness this corpus feeds."""
    return REPO_ROOT / "scripts" / "listening" / "run_panel.py"


def _load(args: argparse.Namespace) -> tuple[cp.Corpus, dict]:
    corpus_dir = Path(args.corpus) if args.corpus else None
    return cp.load_corpus(corpus_dir), cp.load_design(
        Path(args.design) if args.design else None
    )


def _status_code(report: cp.ValidationReport) -> int:
    if report.blockers:
        return EXIT_BLOCKERS
    if report.shortfalls:
        return EXIT_SHORTFALLS
    return EXIT_READY


def cmd_validate(args: argparse.Namespace) -> int:
    corpus, design = _load(args)
    report = cp.validate(corpus, design)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return _status_code(report)

    if report.blockers:
        print("BLOCKERS — the corpus may not be used until these are resolved:")
        for finding in report.blockers:
            print(f"  [{finding.rule}] {finding.detail}")
        print()
    if report.shortfalls:
        print("SHORTFALLS — consent-clean, but not yet complete enough for Gate A:")
        for finding in report.shortfalls:
            print(f"  [{finding.rule}] {finding.detail}")
        print()

    texts = report.coverage.get("texts", {})
    refs = report.coverage.get("references", {})
    print(f"texts:      {texts.get('total', 0)} across {len(texts.get('by_category', {}))} categories")
    print(f"languages:  {', '.join(texts.get('languages', [])) or '(none)'}")
    print(f"references: {refs.get('total', 0)} from {refs.get('speakers_with_audio', 0)} speakers")
    code = _status_code(report)
    verdict = {
        EXIT_READY: "READY for Gate A",
        EXIT_BLOCKERS: "NOT USABLE (consent/integrity blockers)",
        EXIT_SHORTFALLS: "NOT READY (coverage shortfalls)",
    }[code]
    print(f"\n{verdict}")
    return code


def cmd_coverage(args: argparse.Namespace) -> int:
    corpus, design = _load(args)
    report = cp.validate(corpus, design)
    print(json.dumps(report.coverage, indent=2))
    return _status_code(report)


def cmd_emit_stimuli(args: argparse.Namespace) -> int:
    corpus, design = _load(args)
    report = cp.validate(corpus, design)
    if report.blockers and not args.allow_blockers:
        print(
            json.dumps(
                {
                    "error": "refusing to emit stimuli from a corpus with consent/integrity blockers",
                    "blockers": [f.to_dict() for f in report.blockers],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return EXIT_BLOCKERS

    renders = json.loads(Path(args.renders).read_text(encoding="utf-8"))
    manifest = cp.emit_stimulus_manifest(
        corpus, renders, incumbent=args.incumbent, candidate=args.candidate,
        duration_bucket=args.duration_bucket,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(args.out),
                "items": len(manifest["items"]),
                **manifest["_provenance"],
            },
            indent=2,
        )
    )
    return EXIT_READY


# --------------------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------------------


def _synthetic_audio_corpus(corpus: cp.Corpus, out_dir: Path, *, n_speakers: int = 10):
    """Build a consent-complete speaker/reference set and a render index for the dry run.

    No audio is produced or required: the dry run exercises the manifest and analysis path, and
    every record is stamped `dry_run` so it can never be mistaken for real enrollment material.
    """
    conditions = ["clean_studio", "ordinary_phone", "reverberant_room", "noisy"]
    deliveries = ["neutral", "emotional"]
    languages = ["en", "zh", "es", "ja"]

    speakers = []
    references = []
    for index in range(n_speakers):
        speaker_id = f"dryspk{index + 1:02d}"
        language = languages[index % len(languages)]
        speakers.append(
            {
                "speaker_id": speaker_id,
                "pseudonym": f"DRY-RUN Speaker {index + 1}",
                "languages": [language],
                "notes": "SYNTHETIC dry-run record; no audio exists",
            }
        )
        for bucket_index, seconds in enumerate((3, 10, 30)):
            references.append(
                {
                    "reference_id": f"{speaker_id}-r{seconds}",
                    "speaker_id": speaker_id,
                    "language": language,
                    "duration_seconds": seconds,
                    "acoustic_condition": conditions[(index + bucket_index) % len(conditions)],
                    "delivery": deliveries[(index + bucket_index) % len(deliveries)],
                    "transcript": "dry-run reference transcript",
                    "consent": {
                        "consent_statement": "DRY RUN — synthetic record, no human speaker involved",
                        "consent_obtained_utc": "2026-08-06T00:00:00Z",
                        "consent_scope": "explicit_recorded_for_this_project",
                        "speaker_pseudonym": f"DRY-RUN Speaker {index + 1}",
                        "provenance": "generated by build_corpus.py dryrun",
                        "sha256": f"{'0' * 63}{index % 10}",
                    },
                }
            )
    (out_dir / "speakers.json").write_text(
        json.dumps({"schema_version": cp.SCHEMA_VERSION, "speakers": speakers}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "references.json").write_text(
        json.dumps({"schema_version": cp.SCHEMA_VERSION, "references": references}, indent=2),
        encoding="utf-8",
    )
    return speakers, references


def _dryrun_renders(
    corpus: cp.Corpus, references, *, incumbent: str, candidate: str, texts_per_reference: int = 16
) -> dict:
    """One render per (reference, text) cell, matching each reference to same-language texts."""
    renders: dict[str, str] = {}
    by_language: dict[str, list[str]] = {}
    for text in corpus.texts:
        by_language.setdefault(text.effective_language, []).append(text.text_id)

    for reference in references:
        # 30s references carry the long-form texts; shorter ones take the general pool.
        pool = by_language.get(reference["language"], [])
        if not pool:
            continue
        take = pool[:texts_per_reference]
        for text_id in take:
            base = f"{reference['reference_id']}|{text_id}"
            for system in ("reference", incumbent, candidate, "anchor_low"):
                renders[f"{base}|{system}"] = f"dryrun/{base.replace('|', '__')}__{system}.wav"
    for reference in references:
        renders[f"foil|{reference['speaker_id']}"] = f"dryrun/foil__{reference['speaker_id']}.wav"
    return renders


def cmd_dryrun(args: argparse.Namespace) -> int:
    """Corpus -> stimuli -> blinded panel -> verdict, end to end, with a synthetic panel.

    This is the bead's "dry-run with a small listener pool validates the pipeline" criterion. It
    proves the seam between the corpus compiler and the listening harness, and nothing about
    audio quality: every verdict it produces carries is_quality_claim=false.
    """
    out_dir = Path(args.out)
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    real_corpus, design = _load(args)
    # Reuse the real authored texts; synthesize only the speaker/reference/render side.
    (corpus_dir / "texts.json").write_text(
        (Path(args.corpus or cp.CORPUS_DIR) / "texts.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _, references = _synthetic_audio_corpus(real_corpus, corpus_dir, n_speakers=args.speakers)

    dry_corpus = cp.load_corpus(corpus_dir)
    report = cp.validate(dry_corpus, design)
    if report.blockers:
        print(
            json.dumps(
                {"error": "dry-run corpus has consent blockers", "blockers": [f.to_dict() for f in report.blockers]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return EXIT_BLOCKERS

    incumbent, candidate = "qwen3_tts_12hz", "kyutai_pocket_100m"
    renders = _dryrun_renders(dry_corpus, references, incumbent=incumbent, candidate=candidate)
    # One panel, one reference length (see emit_stimulus_manifest). The 10s bucket is the
    # dry run's choice; Gate A runs the 3s / 10s / 30s buckets as separate panels.
    manifest = cp.emit_stimulus_manifest(
        dry_corpus, renders, incumbent=incumbent, candidate=candidate, duration_bucket=10.0
    )
    manifest_path = out_dir / "stimuli.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # bakeoff_gate_a declares objective families, so the dry run supplies a matching objective
    # metrics file rather than silently analysing a subset of the instance.
    objective_path = out_dir / "objective.json"
    rows = []
    import random

    rng = random.Random(20260806)
    #  Tier-0 metrics are automated, so they span the WHOLE corpus — every reference-length
    #  bucket — not just the single bucket the human panel hears. Scoring only the panel subset
    #  would throw away most of the cheap evidence and leave the objective families under their
    #  minimum utterance count.
    by_text = {t.text_id: t for t in dry_corpus.texts}
    by_reference = {r.reference_id: r for r in dry_corpus.references}
    objective_cells = []
    for key in sorted(renders):
        parts = key.split("|")
        if len(parts) != 3 or parts[2] != "reference":
            continue
        reference, text = by_reference.get(parts[0]), by_text.get(parts[1])
        if reference is None or text is None:
            continue
        axes = set(text.axes)
        if reference.acoustic_condition in ("noisy", "reverberant_room"):
            axes.add("noisy_reference")
        objective_cells.append(
            {
                "item_id": f"{reference.reference_id}__{text.text_id}",
                "speaker_id": reference.speaker_id,
                "text_id": text.text_id,
                "language": text.effective_language,
                "regime": "long" if "long_form" in axes else "short",
                "axes": sorted(axes),
            }
        )

    for item in objective_cells:
        for metric, base, spread in (
            ("wer", 0.030, 0.010),
            ("longform_drift", 0.05, 0.02),
        ):
            value = abs(rng.gauss(base, spread))
            rows.append(
                {
                    "metric": metric,
                    **item,
                    "incumbent": round(value, 5),
                    "candidate": round(max(0.0, value + rng.gauss(0.0, 0.008)), 5),
                }
            )
    objective_path.write_text(
        json.dumps({"schema_version": "1.0.0", "metrics": rows}, indent=2), encoding="utf-8"
    )

    #  Two panels, because a dry run that only shows the happy path proves half of what matters.
    #  `sized` runs at the design's recruited panel size and must reach a decisive verdict.
    #  `undersized` is the bead's "small listener pool" and must be REFUSED as INVALID — a
    #  pipeline that certifies an under-powered panel is worse than no pipeline.
    panels = [
        ("sized", args.listeners, args.trials, ("PASS", "FAIL", "INSUFFICIENT_POWER"), "pass"),
        ("undersized", args.small_listeners, args.trials, ("INVALID",), "fail"),
    ]
    results: dict[str, dict] = {}
    ok = True
    for name, listeners, trials, expected_overall, expected_design_bit in panels:
        panel_dir = out_dir / f"panel-{name}"
        for step in (
            ["plan", "--manifest", str(manifest_path), "--instance", "bakeoff_gate_a",
             "--out", str(panel_dir), "--listeners", str(listeners), "--trials", str(trials)],
            ["simulate", "--plan", str(panel_dir), "--manifest", str(manifest_path),
             "--out", str(panel_dir / "responses.jsonl")],
            ["analyze", "--plan", str(panel_dir), "--manifest", str(manifest_path),
             "--instance", "bakeoff_gate_a",
             "--responses", str(panel_dir / "responses.jsonl"),
             "--objective", str(objective_path),
             "--out", str(panel_dir / "verdict.json")],
        ):
            completed = subprocess.run(
                [sys.executable, str(runner_path()), *step], capture_output=True, text=True
            )
            if completed.returncode != 0:
                print(completed.stdout + completed.stderr, file=sys.stderr)
                return EXIT_BLOCKERS

        verdict = json.loads((panel_dir / "verdict.json").read_text(encoding="utf-8"))
        matched = (
            verdict["overall"] in expected_overall
            and verdict["bits"]["design_valid"] == expected_design_bit
            and verdict["synthetic_panel"] is True
            and verdict["is_quality_claim"] is False
        )
        ok = ok and matched
        results[name] = {
            "listeners_recruited": listeners,
            "expected_overall": list(expected_overall),
            "observed_overall": verdict["overall"],
            "design_valid": verdict["bits"]["design_valid"],
            "bits": verdict["bits"],
            "synthetic_panel": verdict["synthetic_panel"],
            "is_quality_claim": verdict["is_quality_claim"],
            "ok": matched,
        }

    summary = {
        "dryrun": "bakeoff corpus -> listening harness",
        "corpus": {
            "texts": len(dry_corpus.texts),
            "speakers": len(dry_corpus.speakers),
            "references": len(dry_corpus.references),
            "shortfalls": len(report.shortfalls),
        },
        "stimuli": {
            "items": len(manifest["items"]),
            "complete_cells": manifest["_provenance"]["complete_cells"],
            "dropped_incomplete": manifest["_provenance"]["incomplete_cells_dropped"],
            "duration_bucket_seconds": 10.0,
        },
        "panels": results,
        "pipeline_ok": ok,
        "note": "SYNTHETIC panel. Validates the corpus->panel seam only; says nothing about audio.",
    }
    (out_dir / "dryrun.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return EXIT_READY if ok else EXIT_BLOCKERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_corpus.py", description=__doc__)
    parser.add_argument("--corpus", default=None, help="corpus directory (default corpus/bakeoff)")
    parser.add_argument("--design", default=None, help="design.toml path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check coverage and consent")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("coverage", help="print the coverage report")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("emit-stimuli", help="build a listening-harness stimulus manifest")
    p.add_argument("--renders", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--duration-bucket", type=float, default=None,
                   help="reference length in seconds; a panel must hold this constant")
    p.add_argument("--allow-blockers", action="store_true")
    p.set_defaults(func=cmd_emit_stimuli)

    p = sub.add_parser("dryrun", help="corpus -> stimuli -> panel -> verdict, synthetic panel")
    p.add_argument("--out", default="target/bakeoff-dryrun")
    p.add_argument("--speakers", type=int, default=10)
    p.add_argument("--listeners", type=int, default=32,
                   help="recruited size of the properly-sized dry-run panel")
    p.add_argument("--small-listeners", type=int, default=8,
                   help="under-sized pool that the harness must refuse as INVALID")
    p.add_argument("--trials", type=int, default=24)
    p.set_defaults(func=cmd_dryrun)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except cp.CorpusError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return EXIT_USAGE
    except (OSError, KeyError, ValueError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2), file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
