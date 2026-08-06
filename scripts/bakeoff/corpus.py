"""Bakeoff corpus model: load, validate coverage and consent, emit listening stimuli.

The bakeoff corpus is three tables that are deliberately kept apart:

    texts.json       what gets synthesized  (authored here; no third-party licensing)
    speakers.json    who the voices are     (pseudonymous identities, no audio)
    references.json  the actual recordings  (consent + provenance REQUIRED on every row)

Splitting them is what makes the consent rule enforceable: audio only ever enters through
`references.json`, so there is exactly one place to check, and `validate` refuses the whole
corpus if any single row is short a field. There is no partial-credit mode — doctrine 10 is not
a scoring rubric.

`corpus/bakeoff/design.toml` is the coverage contract. "The corpus is ready" is therefore a
computed fact, not an opinion, and the failure report says exactly which cells are short.

Standard library only. Bead: frankentts-bake-corpus-48h.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "bakeoff"

#  Zero-width and soft-hyphen characters are invisible in every editor and reviewer's terminal
#  but change what the tokenizer sees, so a corpus carrying them silently tests something other
#  than the sentence on the page. This check has already caught two defects in this corpus's own
#  authoring (a soft hyphen inside a surname, a stray Thai character in a Japanese sentence).
INVISIBLE_CHARS = frozenset("­​‌‍﻿⁠")


class CorpusError(RuntimeError):
    """Raised when the corpus cannot be loaded at all (as opposed to failing validation)."""


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Text:
    text_id: str
    language: str
    category: str
    axes: tuple[str, ...]
    text: str
    primary_language: str | None = None

    @property
    def effective_language(self) -> str:
        return self.primary_language or self.language


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    pseudonym: str
    languages: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class Reference:
    reference_id: str
    speaker_id: str
    language: str
    duration_seconds: float
    acoustic_condition: str
    delivery: str
    transcript: str
    consent: dict[str, Any]
    path: str = ""

    def duration_bucket(self, buckets: Iterable[float], tolerance: float) -> float | None:
        for bucket in buckets:
            if abs(self.duration_seconds - bucket) <= tolerance:
                return bucket
        return None


@dataclass
class Corpus:
    texts: tuple[Text, ...]
    speakers: tuple[Speaker, ...]
    references: tuple[Reference, ...]

    def speakers_by_id(self) -> dict[str, Speaker]:
        return {s.speaker_id: s for s in self.speakers}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read_json(path: Path, expected_key: str) -> list[dict]:
    if not path.is_file():
        raise CorpusError(f"missing corpus file: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path}: invalid JSON: {exc}") from None
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise CorpusError(
            f"{path}: schema {raw.get('schema_version')!r} != supported {SCHEMA_VERSION!r}"
        )
    if expected_key not in raw:
        raise CorpusError(f"{path}: missing top-level {expected_key!r} array")
    return raw[expected_key]


def load_design(path: Path | None = None) -> dict[str, Any]:
    path = path or CORPUS_DIR / "design.toml"
    if not path.is_file():
        raise CorpusError(f"missing design file: {path}")
    with path.open("rb") as handle:
        design = tomllib.load(handle)
    if design.get("schema_version") != SCHEMA_VERSION:
        raise CorpusError(f"{path}: unsupported design schema {design.get('schema_version')!r}")
    return design


def load_corpus(corpus_dir: Path | None = None) -> Corpus:
    """Load all three tables. Missing speakers/references are allowed and reported as empty.

    A corpus with texts but no references is the expected state before any consent-clean audio
    has been collected; `validate` will say so plainly rather than pretending it is ready.
    """
    corpus_dir = corpus_dir or CORPUS_DIR

    texts = tuple(
        Text(
            text_id=row["text_id"],
            language=row["language"],
            category=row["category"],
            axes=tuple(row.get("axes", ())),
            text=row["text"],
            primary_language=row.get("primary_language"),
        )
        for row in _read_json(corpus_dir / "texts.json", "texts")
    )

    speakers_path = corpus_dir / "speakers.json"
    speakers = (
        tuple(
            Speaker(
                speaker_id=row["speaker_id"],
                pseudonym=row["pseudonym"],
                languages=tuple(row.get("languages", ())),
                notes=row.get("notes", ""),
            )
            for row in _read_json(speakers_path, "speakers")
        )
        if speakers_path.is_file()
        else ()
    )

    references_path = corpus_dir / "references.json"
    references = (
        tuple(
            Reference(
                reference_id=row["reference_id"],
                speaker_id=row["speaker_id"],
                language=row["language"],
                duration_seconds=float(row["duration_seconds"]),
                acoustic_condition=row["acoustic_condition"],
                delivery=row["delivery"],
                transcript=row.get("transcript", ""),
                consent=row.get("consent") or {},
                path=row.get("path", ""),
            )
            for row in _read_json(references_path, "references")
        )
        if references_path.is_file()
        else ()
    )

    return Corpus(texts=texts, speakers=speakers, references=references)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


@dataclass
class Finding:
    severity: str  # "blocker" | "shortfall"
    rule: str
    detail: str

    def to_dict(self) -> dict:
        return {"severity": self.severity, "rule": self.rule, "detail": self.detail}


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)

    def blocker(self, rule: str, detail: str) -> None:
        self.findings.append(Finding("blocker", rule, detail))

    def shortfall(self, rule: str, detail: str) -> None:
        self.findings.append(Finding("shortfall", rule, detail))

    @property
    def blockers(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def shortfalls(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "shortfall"]

    @property
    def ready(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        return {
            "ready_for_gate_a": self.ready,
            "consent_clean": not self.blockers,
            "n_blockers": len(self.blockers),
            "n_shortfalls": len(self.shortfalls),
            "findings": [f.to_dict() for f in self.findings],
            "coverage": self.coverage,
        }


def validate_texts(corpus: Corpus, design: dict[str, Any], report: ValidationReport) -> None:
    texts = corpus.texts
    seen: set[str] = set()
    for text in texts:
        if text.text_id in seen:
            report.blocker("text-ids-unique", f"duplicate text_id {text.text_id!r}")
        seen.add(text.text_id)
        if not text.text.strip():
            report.blocker("text-nonempty", f"{text.text_id}: empty text")
        bad = sorted({hex(ord(c)) for c in text.text if c in INVISIBLE_CHARS})
        if bad:
            report.blocker(
                "text-no-invisible-characters",
                f"{text.text_id}: invisible characters {bad} change what the tokenizer sees "
                "while being unreviewable on screen",
            )

    counts: dict[str, int] = {}
    for text in texts:
        counts[text.category] = counts.get(text.category, 0) + 1
    for category, minimum in design.get("text_coverage", {}).items():
        have = counts.get(category, 0)
        if have < int(minimum):
            report.shortfall(
                "text-category-coverage", f"category {category}: {have} texts < required {minimum}"
            )

    scale = design.get("scale", {})
    if len(texts) < int(scale.get("min_texts", 0)):
        report.shortfall(
            "text-scale", f"{len(texts)} texts < required {scale['min_texts']}"
        )

    axes_cfg = design.get("canary_axes", {})
    reference_supplied = set(axes_cfg.get("reference_supplied_axes", []))
    axis_counts: dict[str, int] = {}
    for text in texts:
        for axis in text.axes:
            axis_counts[axis] = axis_counts.get(axis, 0) + 1
    for axis in axes_cfg.get("required", []):
        if axis in reference_supplied:
            continue
        have = axis_counts.get(axis, 0)
        if have < int(axes_cfg.get("min_texts_per_axis", 0)):
            report.shortfall(
                "canary-axis-coverage",
                f"axis {axis}: {have} texts < required {axes_cfg['min_texts_per_axis']}; "
                "the tail gate cannot test an axis the corpus does not populate",
            )

    languages = {t.effective_language for t in texts} | {t.language for t in texts}
    for language in design.get("languages", {}).get("required", []):
        if language not in languages:
            report.shortfall("text-language-coverage", f"no texts in language {language!r}")

    report.coverage["texts"] = {
        "total": len(texts),
        "by_category": counts,
        "by_axis": axis_counts,
        "languages": sorted(languages),
    }


def validate_consent(corpus: Corpus, design: dict[str, Any], report: ValidationReport) -> None:
    """Doctrine 10. Any incomplete record is a blocker for the whole corpus, not for one row."""
    consent_cfg = design.get("consent", {})
    required = list(consent_cfg.get("required_fields", []))
    permitted = set(consent_cfg.get("permitted_scopes", []))
    forbidden = set(consent_cfg.get("forbidden_scopes", []))

    for reference in corpus.references:
        record = reference.consent
        if not record:
            report.blocker(
                "consent-record-present",
                f"{reference.reference_id}: no consent record; reference audio without an "
                "attestation may not enter the corpus",
            )
            continue
        missing = [key for key in required if not str(record.get(key, "")).strip()]
        if missing:
            report.blocker(
                "consent-record-complete",
                f"{reference.reference_id}: consent record missing {missing}",
            )
        scope = str(record.get("consent_scope", "")).strip()
        if scope in forbidden:
            report.blocker(
                "consent-scope-forbidden",
                f"{reference.reference_id}: consent_scope {scope!r} is never acceptable "
                "(doctrine 10: we do not build on voices whose owners did not provide them)",
            )
        elif scope and scope not in permitted:
            report.blocker(
                "consent-scope-unrecognised",
                f"{reference.reference_id}: consent_scope {scope!r} is not in the permitted set "
                f"{sorted(permitted)}; an unrecognised scope is treated as no scope at all",
            )

    report.coverage["consent"] = {
        "references_checked": len(corpus.references),
        "blockers": len([f for f in report.blockers if f.rule.startswith("consent")]),
    }


def validate_references(corpus: Corpus, design: dict[str, Any], report: ValidationReport) -> None:
    refs = corpus.references
    scale = design.get("scale", {})
    speakers_with_audio = {r.speaker_id for r in refs}

    if not refs:
        report.shortfall(
            "reference-scale",
            "no reference recordings yet: the corpus holds texts and its contract only. "
            "Gate A cannot run until consent-clean audio is collected",
        )
        report.coverage["references"] = {"total": 0}
        return

    known = corpus.speakers_by_id()
    for reference in refs:
        if reference.speaker_id not in known:
            report.blocker(
                "reference-speaker-known",
                f"{reference.reference_id}: speaker {reference.speaker_id!r} is not in speakers.json",
            )

    if len(speakers_with_audio) < int(scale.get("min_speakers", 0)):
        report.shortfall(
            "reference-scale",
            f"{len(speakers_with_audio)} speakers with audio < required {scale['min_speakers']}",
        )

    ref_cfg = design.get("references", {})
    buckets = [float(b) for b in ref_cfg.get("duration_buckets_seconds", [])]
    tolerance = float(ref_cfg.get("tolerance_seconds", 1.0))
    per_bucket: dict[float, set[str]] = {b: set() for b in buckets}
    per_speaker_buckets: dict[str, set[float]] = {}
    unbucketed = 0
    for reference in refs:
        bucket = reference.duration_bucket(buckets, tolerance)
        if bucket is None:
            unbucketed += 1
            continue
        per_bucket[bucket].add(reference.speaker_id)
        per_speaker_buckets.setdefault(reference.speaker_id, set()).add(bucket)
    if unbucketed:
        report.shortfall(
            "reference-duration-buckets",
            f"{unbucketed} reference(s) fall outside the {buckets} second buckets "
            f"(+/- {tolerance}s) and cannot be compared across reference length",
        )
    for bucket, speaker_ids in per_bucket.items():
        minimum = int(ref_cfg.get("min_speakers_per_duration_bucket", 0))
        if len(speaker_ids) < minimum:
            report.shortfall(
                "reference-duration-buckets",
                f"{bucket:g}s bucket: {len(speaker_ids)} speakers < required {minimum}",
            )
    min_buckets = int(ref_cfg.get("min_buckets_per_speaker", 0))
    thin = [s for s, b in per_speaker_buckets.items() if len(b) < min_buckets]
    if thin:
        report.shortfall(
            "reference-duration-buckets",
            f"{len(thin)} speaker(s) appear in fewer than {min_buckets} duration buckets, "
            "so reference-length effects cannot be measured within speaker",
        )

    for section, field_name, label in (
        ("acoustic_conditions", "acoustic_condition", "condition"),
        ("delivery", "delivery", "delivery"),
    ):
        cfg = design.get(section, {})
        minimum = int(cfg.get(f"min_speakers_per_{label}", 0) or cfg.get("min_speakers_per_condition", 0))
        by_value: dict[str, set[str]] = {}
        for reference in refs:
            by_value.setdefault(getattr(reference, field_name), set()).add(reference.speaker_id)
        for value in cfg.get("required", []):
            have = len(by_value.get(value, set()))
            if have < minimum:
                report.shortfall(
                    f"{section}-coverage",
                    f"{label} {value!r}: {have} speakers < required {minimum}",
                )
        unknown = sorted(set(by_value) - set(cfg.get("required", [])))
        if unknown:
            report.blocker(
                f"{section}-vocabulary",
                f"unrecognised {label} value(s) {unknown}; extend design.toml deliberately "
                "rather than letting free text into a coverage axis",
            )

    lang_cfg = design.get("languages", {})
    by_language: dict[str, set[str]] = {}
    for reference in refs:
        by_language.setdefault(reference.language, set()).add(reference.speaker_id)
    for language in lang_cfg.get("required", []):
        have = len(by_language.get(language, set()))
        if have < int(lang_cfg.get("min_speakers_per_language", 0)):
            report.shortfall(
                "reference-language-coverage",
                f"language {language!r}: {have} speakers < required "
                f"{lang_cfg['min_speakers_per_language']}",
            )

    text_languages = {t.effective_language for t in corpus.texts}
    cross_pairs = {
        (r.language, tl)
        for r in refs
        for tl in text_languages
        if tl != r.language
    }
    if len(cross_pairs) < int(lang_cfg.get("min_cross_language_pairs", 0)):
        report.shortfall(
            "cross-language-coverage",
            f"{len(cross_pairs)} cross-language (reference, target) pairs available < required "
            f"{lang_cfg['min_cross_language_pairs']}",
        )

    report.coverage["references"] = {
        "total": len(refs),
        "speakers_with_audio": len(speakers_with_audio),
        "by_duration_bucket": {f"{b:g}s": len(s) for b, s in per_bucket.items()},
        "by_condition": {
            c: len({r.speaker_id for r in refs if r.acoustic_condition == c})
            for c in sorted({r.acoustic_condition for r in refs})
        },
        "by_delivery": {
            d: len({r.speaker_id for r in refs if r.delivery == d})
            for d in sorted({r.delivery for r in refs})
        },
        "cross_language_pairs": len(cross_pairs),
    }


def validate(corpus: Corpus, design: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()
    validate_texts(corpus, design, report)
    validate_consent(corpus, design, report)
    validate_references(corpus, design, report)
    return report


# --------------------------------------------------------------------------------------
# Bridge into the listening harness
# --------------------------------------------------------------------------------------


def emit_stimulus_manifest(
    corpus: Corpus,
    renders: dict[str, str],
    *,
    incumbent: str,
    candidate: str,
    duration_bucket: float | None = None,
    tolerance: float = 1.0,
) -> dict[str, Any]:
    """Build the manifest consumed by scripts/listening/run_panel.py.

    `renders` maps "<reference_id>|<text_id>|<system>" to an audio path, where `system` is one
    of the two systems under test, `reference` (the natural recording of the speaker) or
    `anchor_low`. Only cells rendered for BOTH systems plus the reference are emitted: the
    listening harness requires a complete triple per cell and would otherwise silently drop
    them, turning missing renders into quiet coverage loss.

    `duration_bucket` selects one reference-length bucket. A panel must hold reference length
    CONSTANT: the listening harness identifies a cell by (speaker, text, language, regime), so
    mixing 3s and 30s references of the same speaker would put two different stimuli in one
    cell and confound the system contrast with a reference-length effect. Reference length is
    a real axis — it is varied ACROSS panels, one bucket each, not within one.
    """
    by_reference = {r.reference_id: r for r in corpus.references}
    if duration_bucket is not None:
        by_reference = {
            rid: ref
            for rid, ref in by_reference.items()
            if abs(ref.duration_seconds - duration_bucket) <= tolerance
        }
    by_text = {t.text_id: t for t in corpus.texts}
    items: list[dict[str, Any]] = []
    complete_cells = 0
    incomplete: list[str] = []
    claimed_cells: dict[tuple[str, str], str] = {}

    for key in sorted(renders):
        parts = key.split("|")
        if len(parts) != 3:
            continue  # foil entries are keyed "foil|<speaker_id>" and handled below
        reference_id, text_id, system = parts
        if system != "reference":
            continue
        reference = by_reference.get(reference_id)
        text = by_text.get(text_id)
        if reference is None or text is None:
            continue

        roles = {
            "reference": f"{reference_id}|{text_id}|reference",
            "incumbent": f"{reference_id}|{text_id}|{incumbent}",
            "candidate": f"{reference_id}|{text_id}|{candidate}",
        }
        if not all(k in renders for k in roles.values()):
            incomplete.append(f"{reference_id}|{text_id}")
            continue

        cell = (reference.speaker_id, text_id)
        if cell in claimed_cells:
            raise CorpusError(
                f"cell {cell} is claimed by two references ({claimed_cells[cell]} and "
                f"{reference_id}). Pass duration_bucket= to hold reference length constant: a "
                "panel that mixes reference lengths confounds the system contrast with a "
                "reference-length effect"
            )
        claimed_cells[cell] = reference_id
        complete_cells += 1

        axes = set(text.axes)
        if reference.acoustic_condition in ("noisy", "reverberant_room"):
            axes.add("noisy_reference")
        regime = "long" if "long_form" in axes else "short"

        for role, render_key in roles.items():
            system_name = {
                "reference": "natural",
                "incumbent": incumbent,
                "candidate": candidate,
            }[role]
            items.append(
                {
                    "item_id": f"{reference_id}__{text_id}__{role}",
                    "role": role,
                    "system": system_name,
                    "speaker_id": reference.speaker_id,
                    "text_id": text_id,
                    "language": text.effective_language,
                    "regime": regime,
                    "axes": sorted(axes),
                    "path": renders[render_key],
                }
            )
        anchor_key = f"{reference_id}|{text_id}|anchor_low"
        if anchor_key in renders:
            items.append(
                {
                    "item_id": f"{reference_id}__{text_id}__anchor_low",
                    "role": "anchor_low",
                    "system": "anchor_low",
                    "speaker_id": reference.speaker_id,
                    "text_id": text_id,
                    "language": text.effective_language,
                    "regime": regime,
                    "axes": sorted(axes),
                    "path": renders[anchor_key],
                }
            )

    # Foils let the listening harness build catch trials; without them screening is
    # unenforceable and run_panel.py refuses to plan.
    speakers_seen = sorted({i["speaker_id"] for i in items})
    for speaker_id in speakers_seen:
        foil_key = f"foil|{speaker_id}"
        if foil_key in renders:
            items.append(
                {
                    "item_id": f"foil__{speaker_id}",
                    "role": "foil",
                    "system": incumbent,
                    "speaker_id": speaker_id,
                    "text_id": "foil",
                    "language": "und",
                    "regime": "short",
                    "axes": [],
                    "path": renders[foil_key],
                }
            )

    return {
        "manifest_version": "1.0.0",
        "comparison": {"incumbent": incumbent, "candidate": candidate},
        "items": items,
        "_provenance": {
            "complete_cells": complete_cells,
            "incomplete_cells_dropped": len(incomplete),
            "incomplete_examples": incomplete[:10],
        },
    }
