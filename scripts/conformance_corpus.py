"""Validate the frozen conformance corpus selection without treating bootstrap audio as speech."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "corpus/conformance/manifest.json"


class CorpusError(RuntimeError):
    """The frozen corpus cannot serve as an oracle input."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusError(f"cannot load {path.relative_to(REPO)}: {error}") from error


def validate(manifest: dict) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    source = manifest.get("text_source", {})
    source_path = REPO / source.get("path", "")
    if not source_path.is_file():
        errors.append(f"text source is missing: {source.get('path')!r}")
        return errors
    if sha256(source_path) != source.get("sha256"):
        errors.append("text source hash changed; deliberately review and re-freeze the corpus")
        return errors

    texts = {item["text_id"]: item for item in load(source_path).get("texts", [])}
    selected = source.get("selection", [])
    if not selected or len(selected) != len(set(selected)):
        errors.append("selection must be a non-empty unique list")
    missing = sorted(set(selected) - texts.keys())
    if missing:
        errors.append(f"selection names missing texts: {missing}")

    composition = manifest.get("composition", {})
    chosen = [texts[item_id] for item_id in selected if item_id in texts]
    categories = {item["category"] for item in chosen}
    required_categories = set(composition.get("categories_required", []))
    if required_categories - categories:
        errors.append(f"missing categories: {sorted(required_categories - categories)}")
    axes = {axis for item in chosen for axis in item.get("axes", [])}
    required_axes = set(composition.get("canary_axes_required", []))
    if required_axes - axes:
        errors.append(f"missing canary axes: {sorted(required_axes - axes)}")

    capture = manifest.get("capture_matrix", {})
    if set(capture.get("clone_modes", [])) != {"xvector", "icl"}:
        errors.append("capture matrix must include exactly xvector and icl")
    if set(capture.get("prompt_modes", [])) != {"non_streaming", "streaming"}:
        errors.append("capture matrix must include exactly non_streaming and streaming")

    admission = manifest.get("reference_admission", {})
    bootstrap = admission.get("bootstrap_fixture", {})
    bootstrap_path = REPO / bootstrap.get("audio_path", "")
    if not bootstrap_path.is_file() or sha256(bootstrap_path) != bootstrap.get("audio_sha256"):
        errors.append("bootstrap fixture is missing or its hash changed")
    if bootstrap.get("classification") != "nonhuman_fixture_only":
        errors.append("bootstrap fixture must remain explicitly nonhuman_fixture_only")

    required_boundaries = {
        "codec-chunk-first-seam",
        "codec-chunk-second-seam",
        "codec-left-context-prosody",
        "published-eval-cap",
        "runtime-default-cap",
    }
    boundary_ids = {item.get("id") for item in manifest.get("long_form_boundaries", [])}
    if boundary_ids != required_boundaries:
        errors.append("long-form boundary set must match the resolved OQ-6 contract exactly")
    return errors


def main() -> int:
    try:
        errors = validate(load(MANIFEST))
    except CorpusError as error:
        print(json.dumps({"ready": False, "errors": [str(error)]}, indent=2))
        return 1
    if errors:
        print(json.dumps({"ready": False, "errors": errors}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "ready": True,
                "corpus": "qwen3-tts-12hz-conformance-r1",
                "claim": "frozen text selection and oracle capture matrix only; consent-clean human reference audio remains externally supplied",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
