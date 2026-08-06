#!/usr/bin/env python3
"""Repo-structure validators — the architectural rules the compiler cannot state.

`cargo` enforces `unsafe_code = "forbid"` where it is applied, but it cannot tell us that a
crate *forgot* to opt into the workspace lints, that someone reached around the `ftts-kernels`
facade to call frankentorch directly, that a second `[[bin]]` started pointing at the same shim
file, or that `rusqlite` crept in beside `fsqlite`. Those are AGENTS.md rules with real
history behind them, so they get a real check that runs before every cargo invocation.

Fast by construction: pure stdlib, no cargo, no network. Runs in well under a second, which is
why it is stage 1 of `scripts/check.sh` rather than a nightly job.

    python3 scripts/validate_repo.py [--json]

Exit 0 = all rules hold. Exit 1 = at least one violation. Exit 2 = the validator could not run.

Bead: frankentts-p0-ci-083.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#  The single crate permitted to contain audited `unsafe`, and the only one allowed to name
#  the frankentorch crates directly (AGENTS.md toolchain + doctrine: one facade).
KERNEL_CRATE = "ftts-kernels"
FRANKENTORCH_CRATES = ("ft-core", "ft-kernel-cpu", "ft-serialize")
FRANKENTORCH_MODULES = ("ft_core", "ft_kernel_cpu", "ft_serialize")

BANNED_DEPENDENCIES = {
    "rusqlite": "durable state uses fsqlite (/dp/frankensqlite), never rusqlite",
}

#  AGENTS.md "No File Proliferation": revise in place, never fork a file.
PROLIFERATION_RE = re.compile(
    r".*(_v\d+|V\d+|_improved|_enhanced|_new|_old|_copy|_backup|_final|_fixed)\.rs$"
)

EXPECTED_BINS = {"ftts": "src/bin/ftts.rs", "franken_tts": "src/main.rs"}
SHIM_BODY = "fn main() -> std::process::ExitCode {\n    ftts_cli::cli_main()\n}\n"

SKIP_DIRS = {"target", ".git", "node_modules", "__pycache__", ".beads"}


@dataclass
class Violation:
    rule: str
    location: str
    detail: str

    def to_dict(self) -> dict:
        return {"rule": self.rule, "location": self.location, "detail": self.detail}


@dataclass
class Report:
    checked: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)

    def fail(self, rule: str, location: str, detail: str) -> None:
        self.violations.append(Violation(rule, location, detail))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def crate_manifests(root: Path) -> dict[str, Path]:
    crates_dir = root / "crates"
    if not crates_dir.is_dir():
        return {}
    out: dict[str, Path] = {}
    for manifest in sorted(crates_dir.glob("*/Cargo.toml")):
        out[manifest.parent.name] = manifest
    return out


def rust_sources(crate_dir: Path):
    """Every .rs file in a crate, skipping build artifacts *inside* it.

    The skip test is applied to the path RELATIVE to the crate: matching against the absolute
    path would silently skip an entire crate whose ancestors happen to include a directory
    named `target` — which is exactly where the selftest fixtures live, and would have made
    every source-scanning rule quietly stop scanning.
    """
    for path in sorted(crate_dir.rglob("*.rs")):
        if SKIP_DIRS & set(path.relative_to(crate_dir).parts):
            continue
        yield path


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------


def check_workspace_lints(report: Report, workspace: dict) -> None:
    """The forbid lint must be declared once, at the workspace, so crates can inherit it."""
    report.checked.append("workspace-forbids-unsafe")
    lints = workspace.get("workspace", {}).get("lints", {}).get("rust", {})
    value = lints.get("unsafe_code")
    if value != "forbid":
        report.fail(
            "workspace-forbids-unsafe",
            "Cargo.toml",
            f"[workspace.lints.rust] unsafe_code must be \"forbid\", found {value!r}",
        )


def check_crate_lint_inheritance(report: Report, manifests: dict[str, Path], root: Path) -> None:
    """Every crate but the kernel crate inherits the workspace lints; the kernel crate must not.

    A `forbid` cannot be lowered by `allow`, which is exactly why the kernel crate is a separate
    crate that stays outside the inheritance. If it ever starts inheriting, the audited-unsafe
    island design silently stops compiling; if any other crate stops inheriting, the memory-safety
    story silently stops being enforced. Both directions are violations.
    """
    report.checked.append("crate-lint-inheritance")
    for name, manifest in manifests.items():
        data = load_toml(manifest)
        inherits = data.get("lints", {}).get("workspace") is True
        if name == KERNEL_CRATE:
            if inherits:
                report.fail(
                    "crate-lint-inheritance",
                    rel(manifest, root),
                    f"{KERNEL_CRATE} must NOT inherit workspace lints: a forbid cannot be "
                    "lowered by allow, so inheriting it makes the audited unsafe islands "
                    "impossible to compile",
                )
        elif not inherits:
            report.fail(
                "crate-lint-inheritance",
                rel(manifest, root),
                f"crate {name} is missing `[lints]\\nworkspace = true`; it is not covered by "
                "unsafe_code = \"forbid\"",
            )


def check_unsafe_islands(report: Report, manifests: dict[str, Path], root: Path) -> None:
    """No `unsafe` outside the kernel crate, and every unsafe block carries a SAFETY note."""
    report.checked.append("unsafe-islands")
    unsafe_re = re.compile(r"(^|[^\w])unsafe\s*[{(]|(^|[^\w])unsafe\s+(fn|impl|trait)\b")
    for name, manifest in manifests.items():
        for source in rust_sources(manifest.parent):
            lines = source.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("//") or stripped.startswith("#!["):
                    continue
                if not unsafe_re.search(line):
                    continue
                if name != KERNEL_CRATE:
                    report.fail(
                        "unsafe-islands",
                        f"{rel(source, root)}:{lineno}",
                        f"`unsafe` outside {KERNEL_CRATE}: {stripped[:80]}",
                    )
                    continue
                window = lines[max(0, lineno - 6) : lineno - 1]
                if not any("SAFETY:" in prior for prior in window):
                    report.fail(
                        "unsafe-islands",
                        f"{rel(source, root)}:{lineno}",
                        "audited unsafe requires a `// SAFETY:` note within the preceding "
                        f"5 lines: {stripped[:80]}",
                    )


def check_facade(report: Report, manifests: dict[str, Path], root: Path) -> None:
    """frankentorch is consumed through one facade; nobody else names it.

    This is the grep-ban the workspace bead called for. Both halves matter: a crate could add a
    manifest dependency without importing it yet, or import it through a re-export path we have
    not thought of, so the manifest and the sources are both checked.
    """
    report.checked.append("frankentorch-facade")
    module_re = re.compile(r"\b(?:use|extern\s+crate)\s+(" + "|".join(FRANKENTORCH_MODULES) + r")\b")
    for name, manifest in manifests.items():
        if name == KERNEL_CRATE:
            continue
        data = load_toml(manifest)
        deps: set[str] = set()
        for table in ("dependencies", "dev-dependencies", "build-dependencies"):
            deps |= set(data.get(table, {}).keys())
        offending = sorted(deps & set(FRANKENTORCH_CRATES))
        if offending:
            report.fail(
                "frankentorch-facade",
                rel(manifest, root),
                f"crate {name} depends on {offending} directly; frankentorch is reached only "
                f"through the {KERNEL_CRATE} facade",
            )
        for source in rust_sources(manifest.parent):
            for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                match = module_re.search(line)
                if match:
                    report.fail(
                        "frankentorch-facade",
                        f"{rel(source, root)}:{lineno}",
                        f"direct `{match.group(1)}` import outside {KERNEL_CRATE}",
                    )


def check_banned_dependencies(report: Report, manifests: dict[str, Path], root: Path) -> None:
    report.checked.append("banned-dependencies")
    for name, manifest in manifests.items():
        data = load_toml(manifest)
        for table in ("dependencies", "dev-dependencies", "build-dependencies"):
            for dep in data.get(table, {}):
                if dep in BANNED_DEPENDENCIES:
                    report.fail(
                        "banned-dependencies",
                        rel(manifest, root),
                        f"crate {name} depends on `{dep}`: {BANNED_DEPENDENCIES[dep]}",
                    )


def check_cli_shims(report: Report, manifests: dict[str, Path], root: Path) -> None:
    """Two binaries, one entrypoint, each `[[bin]]` pointing at its OWN shim file.

    Pointing two targets at one path produces the "present in multiple build targets" warning
    that `cargo check --all-targets` must stay free of (doctrine #9), so the rule is checked
    structurally here and the warning is separately treated as fatal by check.sh.
    """
    report.checked.append("cli-shims")
    manifest = manifests.get("ftts-cli")
    if manifest is None:
        report.fail("cli-shims", "crates/ftts-cli", "the ftts-cli crate is missing")
        return
    data = load_toml(manifest)
    package = data.get("package", {})
    if package.get("autobins") is not False:
        report.fail(
            "cli-shims",
            rel(manifest, root),
            "ftts-cli must set `autobins = false` so only the declared [[bin]] targets exist",
        )
    bins = {entry.get("name"): entry.get("path") for entry in data.get("bin", [])}
    if bins.keys() != EXPECTED_BINS.keys():
        report.fail(
            "cli-shims",
            rel(manifest, root),
            f"expected exactly the binaries {sorted(EXPECTED_BINS)}, found {sorted(bins)}",
        )
        return
    paths = [p for p in bins.values() if p]
    if len(set(paths)) != len(paths):
        report.fail(
            "cli-shims",
            rel(manifest, root),
            f"two [[bin]] targets share a `path` ({paths}); each needs its own shim file or "
            "cargo emits the multiple-build-targets warning",
        )
    for name, expected_path in EXPECTED_BINS.items():
        actual = bins.get(name)
        if actual != expected_path:
            report.fail(
                "cli-shims",
                rel(manifest, root),
                f"binary {name} should build from {expected_path}, found {actual}",
            )
            continue
        shim = manifest.parent / expected_path
        if not shim.is_file():
            report.fail("cli-shims", rel(shim, root), "shim file is missing")
            continue
        body = shim.read_text(encoding="utf-8")
        if body != SHIM_BODY:
            report.fail(
                "cli-shims",
                rel(shim, root),
                "shims must be byte-for-byte identical one-liners over ftts_cli::cli_main(); "
                f"found {len(body.splitlines())} line(s) differing from the canonical body",
            )


def check_file_proliferation(report: Report, manifests: dict[str, Path], root: Path) -> None:
    report.checked.append("no-file-proliferation")
    for manifest in manifests.values():
        for source in rust_sources(manifest.parent):
            if PROLIFERATION_RE.match(source.name):
                report.fail(
                    "no-file-proliferation",
                    rel(source, root),
                    "revise files in place; versioned/duplicated source files are banned",
                )


def check_workspace_membership(report: Report, workspace: dict, manifests: dict[str, Path]) -> None:
    """Every crate directory is a declared member — a virtual root discovers nothing on its own."""
    report.checked.append("workspace-membership")
    members = set(workspace.get("workspace", {}).get("members", []))
    for name in manifests:
        expected = f"crates/{name}"
        if expected not in members:
            report.fail(
                "workspace-membership",
                "Cargo.toml",
                f"crate directory {expected} exists but is not a workspace member; its tests "
                "and benches would never be built",
            )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def run(root: Path = REPO_ROOT) -> Report:
    report = Report()
    workspace_manifest = root / "Cargo.toml"
    if not workspace_manifest.is_file():
        report.fail("workspace-present", "Cargo.toml", "no workspace manifest at the repo root")
        return report
    workspace = load_toml(workspace_manifest)
    manifests = crate_manifests(root)
    if not manifests:
        report.fail("workspace-present", "crates/", "no crates found under crates/")
        return report

    check_workspace_lints(report, workspace)
    check_workspace_membership(report, workspace, manifests)
    check_crate_lint_inheritance(report, manifests, root)
    check_unsafe_islands(report, manifests, root)
    check_facade(report, manifests, root)
    check_banned_dependencies(report, manifests, root)
    check_cli_shims(report, manifests, root)
    check_file_proliferation(report, manifests, root)
    return report


# --------------------------------------------------------------------------------------
# Selftest — proof that each rule actually fires
# --------------------------------------------------------------------------------------
#
#  A validator nobody has seen fail is a validator nobody knows works. Each case below
#  mutates one thing in a minimal fixture workspace and asserts the corresponding rule
#  reports it. If a future refactor silently defangs a rule, this goes red.

_FIXTURE_WORKSPACE = """[workspace]
members = ["crates/ftts-core", "crates/ftts-kernels", "crates/ftts-cli"]
resolver = "3"

[workspace.package]
version = "0.1.0"
edition = "2024"

[workspace.lints.rust]
unsafe_code = "forbid"
"""

_FIXTURE_CLI_MANIFEST = """[package]
name = "ftts-cli"
version.workspace = true
edition.workspace = true
autobins = false

[[bin]]
name = "franken_tts"
path = "src/main.rs"

[[bin]]
name = "ftts"
path = "src/bin/ftts.rs"

[lints]
workspace = true
"""


def _write_fixture(root: Path) -> None:
    """A minimal but structurally faithful workspace that passes every rule."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "Cargo.toml").write_text(_FIXTURE_WORKSPACE, encoding="utf-8")
    for crate, manifest in (
        (
            "ftts-core",
            '[package]\nname = "ftts-core"\nversion.workspace = true\n'
            'edition.workspace = true\n\n[lints]\nworkspace = true\n',
        ),
        (
            "ftts-kernels",
            '[package]\nname = "ftts-kernels"\nversion.workspace = true\n'
            'edition.workspace = true\n\n[dependencies]\nft-core = { path = "/x" }\n',
        ),
        ("ftts-cli", _FIXTURE_CLI_MANIFEST),
    ):
        crate_dir = root / "crates" / crate
        (crate_dir / "src").mkdir(parents=True, exist_ok=True)
        (crate_dir / "Cargo.toml").write_text(manifest, encoding="utf-8")
        (crate_dir / "src" / "lib.rs").write_text("//! fixture\n", encoding="utf-8")
    cli_src = root / "crates" / "ftts-cli" / "src"
    (cli_src / "bin").mkdir(parents=True, exist_ok=True)
    (cli_src / "main.rs").write_text(SHIM_BODY, encoding="utf-8")
    (cli_src / "bin" / "ftts.rs").write_text(SHIM_BODY, encoding="utf-8")


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


_MUTATIONS: list[tuple[str, str, object]] = [
    (
        "workspace drops the forbid lint",
        "workspace-forbids-unsafe",
        lambda r: (r / "Cargo.toml").write_text(
            _FIXTURE_WORKSPACE.replace('unsafe_code = "forbid"', 'unsafe_code = "deny"'),
            encoding="utf-8",
        ),
    ),
    (
        "a crate stops inheriting workspace lints",
        "crate-lint-inheritance",
        lambda r: (r / "crates/ftts-core/Cargo.toml").write_text(
            '[package]\nname = "ftts-core"\nversion.workspace = true\nedition.workspace = true\n',
            encoding="utf-8",
        ),
    ),
    (
        "the kernel crate starts inheriting the forbid it cannot lower",
        "crate-lint-inheritance",
        lambda r: _append(r / "crates/ftts-kernels/Cargo.toml", "\n[lints]\nworkspace = true\n"),
    ),
    (
        "unsafe appears outside the kernel crate",
        "unsafe-islands",
        lambda r: _append(
            r / "crates/ftts-core/src/lib.rs", "pub fn f() { unsafe { core::hint::spin_loop() } }\n"
        ),
    ),
    (
        "audited unsafe lands without a SAFETY note",
        "unsafe-islands",
        lambda r: _append(
            r / "crates/ftts-kernels/src/lib.rs",
            "pub fn f() { unsafe { core::hint::spin_loop() } }\n",
        ),
    ),
    (
        "a crate depends on frankentorch directly",
        "frankentorch-facade",
        lambda r: _append(
            r / "crates/ftts-core/Cargo.toml", '\n[dependencies]\nft-kernel-cpu = { path = "/x" }\n'
        ),
    ),
    (
        "a crate imports a frankentorch module directly",
        "frankentorch-facade",
        lambda r: _append(r / "crates/ftts-core/src/lib.rs", "use ft_core::Tensor;\n"),
    ),
    (
        "rusqlite creeps in beside fsqlite",
        "banned-dependencies",
        lambda r: _append(
            r / "crates/ftts-core/Cargo.toml", '\n[dependencies]\nrusqlite = "0.31"\n'
        ),
    ),
    (
        "both binaries point at one shim file",
        "cli-shims",
        lambda r: (r / "crates/ftts-cli/Cargo.toml").write_text(
            _FIXTURE_CLI_MANIFEST.replace("src/bin/ftts.rs", "src/main.rs"), encoding="utf-8"
        ),
    ),
    (
        "a shim stops being the canonical one-liner",
        "cli-shims",
        lambda r: _append(r / "crates/ftts-cli/src/bin/ftts.rs", "// drifted\n"),
    ),
    (
        "a forked source file appears",
        "no-file-proliferation",
        lambda r: (r / "crates/ftts-core/src/decoder_v2.rs").write_text("//! nope\n", encoding="utf-8"),
    ),
    (
        "a crate directory is not a workspace member",
        "workspace-membership",
        lambda r: (r / "Cargo.toml").write_text(
            _FIXTURE_WORKSPACE.replace('"crates/ftts-core", ', ""), encoding="utf-8"
        ),
    ),
]


def selftest(workdir: Path) -> tuple[bool, list[dict]]:
    """Fixture directories are deterministic and rewritten in place.

    `_write_fixture` overwrites every file a mutation touches, so re-running against the same
    workdir is idempotent — which lets check.sh point this at a fixed path under `target/`
    instead of scattering a new temp tree on every invocation.
    """
    results: list[dict] = []
    workdir.mkdir(parents=True, exist_ok=True)

    clean = workdir / "clean"
    _write_fixture(clean)
    baseline = run(clean)
    baseline_ok = not baseline.violations
    results.append(
        {
            "case": "clean fixture reports no violations",
            "expected_rule": None,
            "ok": baseline_ok,
            "observed": [v.rule for v in baseline.violations],
        }
    )

    all_ok = baseline_ok
    for index, (description, rule, mutate) in enumerate(_MUTATIONS):
        root = workdir / f"case{index:02d}"
        _write_fixture(root)
        mutate(root)  # type: ignore[operator]
        report = run(root)
        fired = [v.rule for v in report.violations]
        ok = rule in fired
        all_ok = all_ok and ok
        results.append(
            {"case": description, "expected_rule": rule, "ok": ok, "observed": fired}
        )
    return all_ok, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validate_repo.py", description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.add_argument(
        "--selftest",
        nargs="?",
        const="",
        default=None,
        metavar="DIR",
        help="prove each rule fires against fixture workspaces instead of checking this repo",
    )
    args = parser.parse_args(argv)

    if args.selftest is not None:
        workdir = Path(args.selftest) if args.selftest else REPO_ROOT / "target" / "repo-validate"
        ok, results = selftest(workdir)
        if args.json:
            print(json.dumps({"ok": ok, "workdir": str(workdir), "cases": results}, indent=2))
        else:
            for entry in results:
                mark = "ok  " if entry["ok"] else "FAIL"
                expected = entry["expected_rule"] or "(none)"
                print(f"{mark} {entry['case']}  -> expected {expected}, got {entry['observed']}")
            print(
                f"\nvalidate_repo selftest: {sum(1 for e in results if e['ok'])}/{len(results)} "
                f"cases ok"
            )
        return 0 if ok else 1

    try:
        report = run()
    except (OSError, tomllib.TOMLDecodeError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        if args.json:
            print(json.dumps({"error": message}, indent=2))
        else:
            print(f"validate_repo: could not run: {message}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "rules_checked": report.checked,
                    "violations": [v.to_dict() for v in report.violations],
                    "ok": not report.violations,
                },
                indent=2,
            )
        )
    else:
        for violation in report.violations:
            print(f"{violation.location}: [{violation.rule}] {violation.detail}")
        if report.violations:
            print(
                f"\nvalidate_repo: {len(report.violations)} violation(s) across "
                f"{len(report.checked)} rule(s)"
            )
        else:
            print(f"validate_repo: {len(report.checked)} rules OK ({', '.join(report.checked)})")
    return 1 if report.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
