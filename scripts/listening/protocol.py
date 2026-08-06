"""Trial construction, blinding, screening and synthetic-panel simulation.

This is the mechanical half of the franken_tts listening protocol: it turns a stimulus
manifest into a blinded trial plan a listening panel can actually run, screens the returned
responses against predeclared rules, and can drive the whole path with a synthetic panel so
the pipeline is validated before a single human hour is spent.

Everything that consumes randomness takes an explicit seed and records it, so a panel is
reproducible from `plan.json` alone.

Bead: frankentts-v-listening-25m. Policy: docs/CONFORMANCE_AND_LISTENING.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

SCHEMA_VERSION = "1.0.0"

# System roles a manifest item may play. `reference` is the natural recording of the target
# speaker, `anchor_low` is the deliberately degraded MUSHRA anchor, and `foil` is a
# different speaker used to build ABX catch trials.
SystemRole = Literal["reference", "incumbent", "candidate", "anchor_low", "foil"]

CANARY_AXES = (
    "noisy_reference",
    "sibilants",
    "breaths",
    "code_switching",
    "numbers",
    "long_form",
)


class ProtocolError(RuntimeError):
    """Raised for any condition that must stop a panel rather than degrade it silently."""


# --------------------------------------------------------------------------------------
# Stimulus manifest
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stimulus:
    item_id: str
    role: SystemRole
    system: str
    speaker_id: str
    text_id: str
    language: str
    regime: str
    axes: tuple[str, ...]
    path: str
    sha256: str | None = None

    @property
    def cell(self) -> tuple[str, str, str, str]:
        return (self.speaker_id, self.text_id, self.language, self.regime)


@dataclass(frozen=True)
class Manifest:
    incumbent: str
    candidate: str
    items: tuple[Stimulus, ...]
    corpus_id: str

    def by_cell(self) -> dict[tuple[str, str, str, str], dict[SystemRole, Stimulus]]:
        out: dict[tuple[str, str, str, str], dict[SystemRole, Stimulus]] = {}
        for item in self.items:
            slot = out.setdefault(item.cell, {})
            if item.role in slot and item.role != "foil":
                raise ProtocolError(
                    f"duplicate role {item.role!r} for cell {item.cell}: "
                    f"{slot[item.role].item_id} and {item.item_id}"
                )
            slot[item.role] = item
        return out

    def complete_cells(self) -> list[tuple[str, str, str, str]]:
        """Cells carrying the full reference/incumbent/candidate triple."""
        return sorted(
            cell
            for cell, roles in self.by_cell().items()
            if {"reference", "incumbent", "candidate"} <= set(roles)
        )


def load_manifest(path: Path) -> Manifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    version = raw.get("manifest_version")
    if version != SCHEMA_VERSION:
        raise ProtocolError(
            f"manifest schema {version!r} != supported {SCHEMA_VERSION!r}; refusing to guess"
        )
    comparison = raw.get("comparison") or {}
    incumbent = comparison.get("incumbent")
    candidate = comparison.get("candidate")
    if not incumbent or not candidate:
        raise ProtocolError("manifest.comparison must name both `incumbent` and `candidate`")

    items: list[Stimulus] = []
    seen: set[str] = set()
    for entry in raw.get("items", []):
        item_id = entry["item_id"]
        if item_id in seen:
            raise ProtocolError(f"duplicate item_id {item_id!r} in manifest")
        seen.add(item_id)
        axes = tuple(entry.get("axes", ()))
        unknown = [a for a in axes if a not in CANARY_AXES]
        if unknown:
            raise ProtocolError(
                f"item {item_id!r} declares unknown canary axes {unknown}; "
                f"known axes are {list(CANARY_AXES)}"
            )
        items.append(
            Stimulus(
                item_id=item_id,
                role=entry["role"],
                system=entry["system"],
                speaker_id=entry["speaker_id"],
                text_id=entry["text_id"],
                language=entry["language"],
                regime=entry.get("regime", "short"),
                axes=axes,
                path=entry.get("path", ""),
                sha256=entry.get("sha256"),
            )
        )
    if not items:
        raise ProtocolError("manifest contains no items")

    corpus_id = hashlib.sha256(
        "\n".join(sorted(f"{i.item_id}:{i.sha256 or i.path}" for i in items)).encode("utf-8")
    ).hexdigest()[:16]
    return Manifest(incumbent=incumbent, candidate=candidate, items=tuple(items), corpus_id=corpus_id)


# --------------------------------------------------------------------------------------
# Trial plan
# --------------------------------------------------------------------------------------


@dataclass
class Trial:
    trial_id: str
    listener_id: str
    kind: Literal["abx_identity", "mushra_naturalness"]
    family: str
    is_catch: bool
    speaker_id: str
    text_id: str
    language: str
    regime: str
    axes: tuple[str, ...]
    reference_item: str | None
    # slot label -> item_id. The panel-facing export keeps the labels and drops the mapping.
    slots: dict[str, str]

    def blinded(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "listener_id": self.listener_id,
            "kind": self.kind,
            "reference_item": self.reference_item,
            "slots": sorted(self.slots.keys()),
        }


@dataclass
class TrialPlan:
    instance: str
    seed: int
    corpus_id: str
    incumbent: str
    candidate: str
    listeners: list[str]
    trials: list[Trial]
    design: dict[str, Any] = field(default_factory=dict)

    def key_map(self) -> dict[str, dict[str, str]]:
        return {t.trial_id: dict(t.slots) for t in self.trials}


def _slot_labels(n: int) -> list[str]:
    return [f"S{i + 1}" for i in range(n)]


def build_trial_plan(
    manifest: Manifest,
    *,
    instance: str,
    families: Sequence[str],
    n_listeners: int,
    trials_per_listener: int,
    catch_trials_per_listener: int,
    seed: int,
) -> TrialPlan:
    """Generate a counterbalanced, blinded trial plan.

    Each listener sees a rotated slice of the cell list so coverage over speakers/texts is
    balanced across the panel rather than left to independent sampling.
    """
    cells = manifest.complete_cells()
    if not cells:
        raise ProtocolError(
            "no cell carries reference + incumbent + candidate; the panel cannot be built"
        )
    by_cell = manifest.by_cell()
    foils = [i for i in manifest.items if i.role == "foil"]
    if catch_trials_per_listener > 0 and not foils:
        raise ProtocolError(
            "catch trials requested but the manifest declares no `foil` items; "
            "screening would be unenforceable"
        )

    rng = random.Random(seed)
    listeners = [f"L{i + 1:03d}" for i in range(n_listeners)]
    trials: list[Trial] = []
    counter = 0

    for li, listener in enumerate(listeners):
        # Stride the rotation by a full listener block so the panel tiles the cell list
        # evenly; rotating by one cell per listener would pile every listener onto the same
        # opening slice and leave most of the corpus unheard.
        offset = (li * trials_per_listener) % len(cells)
        rotated = cells[offset:] + cells[:offset]
        chosen = [rotated[i % len(rotated)] for i in range(trials_per_listener)]
        rng.shuffle(chosen)

        for family in families:
            kind: Literal["abx_identity", "mushra_naturalness"] = (
                "abx_identity" if family == "identity_abx" else "mushra_naturalness"
            )
            for cell in chosen:
                roles = by_cell[cell]
                counter += 1
                trial_id = f"T{counter:06d}"
                if kind == "abx_identity":
                    members = [roles["incumbent"].item_id, roles["candidate"].item_id]
                else:
                    members = [
                        roles["incumbent"].item_id,
                        roles["candidate"].item_id,
                        roles["reference"].item_id,  # hidden reference
                    ]
                    anchor = roles.get("anchor_low")
                    if anchor is not None:
                        members.append(anchor.item_id)
                rng.shuffle(members)
                labels = _slot_labels(len(members))
                trials.append(
                    Trial(
                        trial_id=trial_id,
                        listener_id=listener,
                        kind=kind,
                        family=family,
                        is_catch=False,
                        speaker_id=cell[0],
                        text_id=cell[1],
                        language=cell[2],
                        regime=cell[3],
                        axes=tuple(sorted(set(roles["candidate"].axes))),
                        reference_item=roles["reference"].item_id,
                        slots=dict(zip(labels, members)),
                    )
                )

        for _ in range(catch_trials_per_listener):
            cell = rng.choice(cells)
            roles = by_cell[cell]
            foil = rng.choice([f for f in foils if f.speaker_id != cell[0]] or foils)
            counter += 1
            members = [roles["incumbent"].item_id, foil.item_id]
            rng.shuffle(members)
            labels = _slot_labels(len(members))
            trials.append(
                Trial(
                    trial_id=f"T{counter:06d}",
                    listener_id=listener,
                    kind="abx_identity",
                    family="screening",
                    is_catch=True,
                    speaker_id=cell[0],
                    text_id=cell[1],
                    language=cell[2],
                    regime=cell[3],
                    axes=(),
                    reference_item=roles["reference"].item_id,
                    slots=dict(zip(labels, members)),
                )
            )

    rng.shuffle(trials)
    return TrialPlan(
        instance=instance,
        seed=seed,
        corpus_id=manifest.corpus_id,
        incumbent=manifest.incumbent,
        candidate=manifest.candidate,
        listeners=listeners,
        trials=trials,
        design={
            "families": list(families),
            "n_listeners": n_listeners,
            "trials_per_listener_per_family": trials_per_listener,
            "catch_trials_per_listener": catch_trials_per_listener,
            "n_cells": len(cells),
            "n_speakers": len({c[0] for c in cells}),
            "n_texts": len({c[1] for c in cells}),
            "languages": sorted({c[2] for c in cells}),
        },
    )


def write_plan(plan: TrialPlan, out_dir: Path) -> dict[str, Path]:
    """Emit the three plan artifacts: metadata, the blinded panel export, and the sealed key."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plan": out_dir / "plan.json",
        "blind": out_dir / "trials.blind.json",
        "key": out_dir / "trials.key.json",
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "instance": plan.instance,
        "seed": plan.seed,
        "corpus_id": plan.corpus_id,
        "incumbent": plan.incumbent,
        "candidate": plan.candidate,
        "listeners": plan.listeners,
        "design": plan.design,
        "n_trials": len(plan.trials),
    }
    paths["plan"].write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["blind"].write_text(
        json.dumps([t.blinded() for t in plan.trials], indent=2) + "\n", encoding="utf-8"
    )
    key = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": plan.corpus_id,
        "trials": [
            {
                "trial_id": t.trial_id,
                "listener_id": t.listener_id,
                "kind": t.kind,
                "family": t.family,
                "is_catch": t.is_catch,
                "speaker_id": t.speaker_id,
                "text_id": t.text_id,
                "language": t.language,
                "regime": t.regime,
                "axes": list(t.axes),
                "reference_item": t.reference_item,
                "slots": t.slots,
            }
            for t in plan.trials
        ],
    }
    paths["key"].write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def load_key(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("trial key schema mismatch")
    return {t["trial_id"]: t for t in raw["trials"]}


# --------------------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------------------


@dataclass
class ScreeningOutcome:
    listener_id: str
    kept: bool
    catch_correct: int
    catch_total: int
    hidden_ref_violations: int
    anchor_violations: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def screen_listeners(
    responses: Sequence[dict],
    key: dict[str, dict[str, Any]],
    roles_by_item: dict[str, SystemRole],
    *,
    catch_min_correct_rate: float,
    hidden_ref_min_score: float,
    hidden_ref_max_violation_rate: float,
    anchor_max_score: float,
    anchor_max_violation_rate: float,
) -> dict[str, ScreeningOutcome]:
    """Apply the predeclared post-screening rules. Rules are fixed before data collection."""
    stats: dict[str, dict[str, int]] = {}

    def bucket(listener: str) -> dict[str, int]:
        return stats.setdefault(
            listener,
            {
                "catch_correct": 0,
                "catch_total": 0,
                "hidden_ref_violations": 0,
                "hidden_ref_total": 0,
                "anchor_violations": 0,
                "anchor_total": 0,
            },
        )

    for resp in responses:
        trial = key.get(resp["trial_id"])
        if trial is None:
            raise ProtocolError(f"response references unknown trial {resp['trial_id']!r}")
        b = bucket(trial["listener_id"])
        if trial["is_catch"]:
            b["catch_total"] += 1
            chosen_item = trial["slots"].get(resp.get("choice", ""))
            if chosen_item is not None and roles_by_item.get(chosen_item) != "foil":
                b["catch_correct"] += 1
            continue
        for slot, score in (resp.get("ratings") or {}).items():
            item = trial["slots"].get(slot)
            role = roles_by_item.get(item or "")
            if role == "reference":
                b["hidden_ref_total"] += 1
                if score < hidden_ref_min_score:
                    b["hidden_ref_violations"] += 1
            elif role == "anchor_low":
                b["anchor_total"] += 1
                if score > anchor_max_score:
                    b["anchor_violations"] += 1

    outcomes: dict[str, ScreeningOutcome] = {}
    for listener, b in sorted(stats.items()):
        reasons: list[str] = []
        if b["catch_total"] > 0:
            rate = b["catch_correct"] / b["catch_total"]
            if rate < catch_min_correct_rate:
                reasons.append(
                    f"catch-trial accuracy {rate:.2f} < {catch_min_correct_rate:.2f}"
                )
        if b["hidden_ref_total"] > 0:
            rate = b["hidden_ref_violations"] / b["hidden_ref_total"]
            if rate > hidden_ref_max_violation_rate:
                reasons.append(
                    f"hidden-reference violations {rate:.2f} > {hidden_ref_max_violation_rate:.2f}"
                )
        if b["anchor_total"] > 0:
            rate = b["anchor_violations"] / b["anchor_total"]
            if rate > anchor_max_violation_rate:
                reasons.append(f"anchor violations {rate:.2f} > {anchor_max_violation_rate:.2f}")
        outcomes[listener] = ScreeningOutcome(
            listener_id=listener,
            kept=not reasons,
            catch_correct=b["catch_correct"],
            catch_total=b["catch_total"],
            hidden_ref_violations=b["hidden_ref_violations"],
            anchor_violations=b["anchor_violations"],
            reasons=reasons,
        )
    return outcomes


# --------------------------------------------------------------------------------------
# Synthetic panel (pipeline validation only — never a quality claim)
# --------------------------------------------------------------------------------------


@dataclass
class SyntheticPanelModel:
    """Response model for the pilot.

    `identity_effect` is the true shift of the candidate's 2AFC identity-preference rate away
    from 0.5 (negative = candidate sounds less like the reference speaker). `naturalness_effect`
    is the true MUSHRA-point deficit of the candidate. `axis_penalty` adds extra degradation on
    the named canary axes, which is how the pilot proves the tail reporting actually bites.
    """

    identity_effect: float = 0.0
    naturalness_effect: float = 0.0
    listener_sd_identity: float = 0.05
    listener_sd_naturalness: float = 6.0
    trial_sd_naturalness: float = 7.0
    incumbent_mushra: float = 78.0
    reference_mushra: float = 98.0
    anchor_mushra: float = 20.0
    #  A competent listener recognises the hidden reference and the low anchor almost
    #  perfectly; only the systems under test attract real rating noise. Modelling them with
    #  the same spread as the test systems would fabricate a ~20% screening loss that has
    #  nothing to do with the protocol.
    reference_sd: float = 1.5
    anchor_sd: float = 6.0
    catch_accuracy: float = 0.99
    axis_penalty: dict[str, float] = field(default_factory=dict)  # identity-rate units
    axis_penalty_mushra: dict[str, float] = field(default_factory=dict)  # MUSHRA points
    bad_listener_rate: float = 0.0


def simulate_responses(
    plan: TrialPlan,
    manifest: Manifest,
    model: SyntheticPanelModel,
    *,
    seed: int,
) -> list[dict]:
    """Drive the full response path with a seeded synthetic panel.

    Emitted responses carry `synthetic: true`; the analyzer propagates that flag into the
    verdict so a pipeline-validation run can never be mistaken for evidence about audio.
    """
    rng = random.Random(seed)
    roles = {i.item_id: i.role for i in manifest.items}
    candidate_ids = {i.item_id for i in manifest.items if i.role == "candidate"}

    listener_identity_bias = {
        listener: rng.gauss(0.0, model.listener_sd_identity) for listener in plan.listeners
    }
    listener_naturalness_bias = {
        listener: rng.gauss(0.0, model.listener_sd_naturalness) for listener in plan.listeners
    }
    inattentive = {
        listener: rng.random() < model.bad_listener_rate for listener in plan.listeners
    }

    out: list[dict] = []
    for trial in plan.trials:
        axis_extra = math.fsum(model.axis_penalty.get(axis, 0.0) for axis in trial.axes)
        axis_extra_mushra = math.fsum(
            model.axis_penalty_mushra.get(axis, 0.0) for axis in trial.axes
        )
        if trial.kind == "abx_identity":
            if trial.is_catch:
                p_correct = 0.5 if inattentive[trial.listener_id] else model.catch_accuracy
                correct_slots = [s for s, item in trial.slots.items() if roles.get(item) != "foil"]
                wrong_slots = [s for s, item in trial.slots.items() if roles.get(item) == "foil"]
                slot = (
                    rng.choice(correct_slots)
                    if rng.random() < p_correct and correct_slots
                    else rng.choice(wrong_slots or correct_slots)
                )
                out.append(
                    {
                        "trial_id": trial.trial_id,
                        "choice": slot,
                        "synthetic": True,
                    }
                )
                continue
            p_candidate = 0.5 + model.identity_effect - axis_extra
            p_candidate += listener_identity_bias[trial.listener_id]
            p_candidate = min(0.99, max(0.01, p_candidate))
            if inattentive[trial.listener_id]:
                p_candidate = 0.5
            picks_candidate = rng.random() < p_candidate
            slot = next(
                s
                for s, item in trial.slots.items()
                if (item in candidate_ids) == picks_candidate
            )
            out.append({"trial_id": trial.trial_id, "choice": slot, "synthetic": True})
        else:
            ratings: dict[str, float] = {}
            for slot, item in trial.slots.items():
                role = roles.get(item)
                if role == "reference":
                    score = model.reference_mushra + rng.gauss(0.0, model.reference_sd)
                elif role == "anchor_low":
                    score = model.anchor_mushra + rng.gauss(0.0, model.anchor_sd)
                else:
                    base = model.incumbent_mushra
                    if role == "candidate":
                        base -= model.naturalness_effect + axis_extra_mushra
                    score = base + listener_naturalness_bias[trial.listener_id]
                    score += rng.gauss(0.0, model.trial_sd_naturalness)
                if inattentive[trial.listener_id]:
                    score = rng.uniform(0.0, 100.0)
                ratings[slot] = round(min(100.0, max(0.0, score)), 2)
            out.append({"trial_id": trial.trial_id, "ratings": ratings, "synthetic": True})
    return out


def write_responses(responses: Iterable[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for resp in responses:
            handle.write(json.dumps(resp, sort_keys=True) + "\n")


def read_responses(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
