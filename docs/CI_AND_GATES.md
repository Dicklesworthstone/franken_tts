# CI & the quality gate

One command gates this repo:

```bash
./scripts/check.sh
```

CI runs exactly that script as its single test step. There is no second list of commands in a
workflow file to drift out of sync — if you want to change what CI enforces, change the script.

Bead: `frankentts-p0-ci-083`.

---

## The stages

They run cheapest-first and **stop at the first failure**, so a structural mistake is reported in
under two seconds instead of after a ten-minute build.

| # | Stage | What it catches |
|---|---|---|
| 1 | `validate_repo.py` | architectural rules cargo cannot state (below) |
| 2 | `validate_repo.py --selftest` | a structural rule that silently stopped detecting its violation |
| 3 | `run_panel.py selftest` | the listening harness no longer detecting audio degradation |
| 4 | `cargo fmt --check` | formatting |
| 5 | `cargo check --locked --all-targets` | type errors, **and the multiple-build-targets warning** |
| 6 | `cargo clippy --locked --all-targets -- -D warnings` | lints |
| 7 | `cargo test --locked` | **the hard gate** — no bead closes while this is red |
| 8 | `ubs --diff` (bounded) | bug scan over working-tree changes |

Stages 2 and 3 exist because a validator nobody has seen fail is a validator nobody knows works.
Both run in about two seconds and guard gates that would otherwise rot silently.

### Skip honesty

Every stage prints `PASS`, `FAIL`, or `SKIP <reason>`. A skip is never folded into green — the
closing banner reads **`GREEN WITH SKIPS`** and lists them, per AGENTS.md Doctrine #0.4. Quote
that banner as it appears; "the gate passed" is not an accurate summary of a run with skips.

`ubs` is the only stage that may skip (it is not installed on hosted runners). Everything else is
required: a missing tool is a failure, not a skip.

### The multiple-build-targets rule

`ftts` and `franken_tts` are two thin shims over one `cli_main()`, and doctrine #9 requires each
`[[bin]]` to point at its **own** file. Two targets sharing a path still compiles — cargo only
warns. That warning is the early symptom, so stage 5 greps for it and treats it as fatal. Stage 1
independently checks the manifest structure, so the mistake is caught before the build even runs.

### Environment knobs

| Variable | Effect |
|---|---|
| `FTTS_CHECK_NO_RCH=1` | bypass the remote compilation helper, run cargo locally (CI sets this) |
| `FTTS_CHECK_UBS_TIMEOUT` | seconds bounding `ubs --diff` (default 300) |

There is deliberately **no knob to skip a stage**. If a stage is in your way, fix it.

---

## The repo validators

`scripts/validate_repo.py` enforces eight rules that the compiler cannot express. Each one has
real history behind it:

| Rule | What it prevents |
|---|---|
| `workspace-forbids-unsafe` | the workspace quietly downgrading `unsafe_code` from `forbid` |
| `workspace-membership` | a crate directory that is not a member — its tests would never build |
| `crate-lint-inheritance` | a crate forgetting `[lints] workspace = true`, **and** `ftts-kernels` accidentally inheriting a `forbid` it cannot lower |
| `unsafe-islands` | `unsafe` outside `ftts-kernels`, or audited unsafe without a `// SAFETY:` note |
| `frankentorch-facade` | reaching around the `ftts-kernels` facade to call `ft-core`/`ft-kernel-cpu`/`ft-serialize` directly, in either the manifest or the source |
| `banned-dependencies` | `rusqlite` creeping in beside `fsqlite` |
| `cli-shims` | the two binaries drifting apart, sharing a path, or losing `autobins = false` |
| `no-file-proliferation` | `decoder_v2.rs`, `nn_improved.rs`, and friends |

The `crate-lint-inheritance` rule is bidirectional on purpose. A `forbid` cannot be lowered by
`allow`, which is exactly why `ftts-kernels` is a separate crate that stays outside the
inheritance. If it ever starts inheriting, the audited-unsafe island design silently stops
compiling; if any other crate stops inheriting, the memory-safety story silently stops being
enforced. Both directions are violations.

```bash
python3 scripts/validate_repo.py              # check this repo
python3 scripts/validate_repo.py --json       # machine-readable
python3 scripts/validate_repo.py --selftest   # prove each rule still fires
```

The selftest builds a minimal fixture workspace, mutates exactly one thing per case, and asserts
the matching rule reports it. Thirteen cases: one clean baseline and one per violation.

---

## The CI workflow

`.github/workflows/ci.yml` has three jobs:

- **`gate`** — blocking. Materializes the sibling dependencies, then runs `./scripts/check.sh`.
- **`cross`** — advisory (`continue-on-error: true`). `cargo check` against all five release
  targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`,
  `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`. `cargo check` emits metadata without linking,
  so every target is checkable from a Linux runner with no cross linker and no SDK. This is what
  catches the scalar kernel fallback when it stops compiling everywhere by construction.
- **`cross-summary`** — reports the advisory result into the job summary and raises a warning
  annotation on failure. An advisory job nobody sees is the same as no advisory job.

### Sibling dependencies (read this before touching the workflow)

The workspace has path dependencies on two sibling repos:

| Dependency | Path in `Cargo.toml` |
|---|---|
| `asupersync` | `../asupersync` (relative) |
| `ft-core`, `ft-kernel-cpu`, `ft-serialize` | `/dp/frankentorch/crates/*` (**absolute**) |

Neither exists on a hosted runner, so CI cannot build without materializing them. The workflow:

1. checks franken_tts out into `frankentts/` rather than the workspace root, so that
   `../asupersync` resolves inside the checkout;
2. checks `asupersync` out beside it;
3. checks `frankentorch` out and symlinks it to `/dp/frankentorch`, because that path is absolute
   and cargo will not look anywhere else.

**Both siblings are pinned by full commit SHA, not by branch.** They are consumed as path deps,
so an upstream force-push would otherwise change what we build against with no diff in this
repository — the same reasoning that produced the truth pack. Bumping a pin is a deliberate,
reviewable edit to `FRANKENTORCH_REF` / `ASUPERSYNC_REF` in the workflow env block.

If either sibling repository is private, CI needs a `SIBLING_REPO_TOKEN` secret with read access.
The workflow falls back to `github.token`, which only works while the repos are public — and it
fails loudly at the mount step rather than proceeding with a broken tree.

### Status: not yet observed running

> The workflow is committed and its YAML parses, and `scripts/check.sh` is verified green
> locally. **The workflow itself has not executed** — that requires a push to GitHub, which this
> bead does not perform. Its first run is the thing that will confirm the sibling-checkout and
> `/dp` symlink steps, and whether `SIBLING_REPO_TOKEN` is needed. Treat the first CI run as part
> of this bead's verification, not as a formality.

`ubs` is not available on hosted runners, so the CI gate closes with `GREEN WITH SKIPS`. `ubs` is
enforced locally before every commit instead (`ubs --diff` during work, `ubs --staged` immediately
before committing).

---

## Troubleshooting

**"Blocking waiting for file lock on build directory."** `CARGO_TARGET_DIR` is shared across
projects and agents on this machine, so concurrent builds serialize. Wait, or set a private
`CARGO_TARGET_DIR` for a one-off — at the cost of a cold rebuild of the siblings.

**`rch` reports "remote execution failed; falling back to local".** Expected — rch fails open.
Check `rch status`; the gate is unaffected, only slower.

**A cross-check target fails but the gate is green.** That is the design: advisory. Fix it before
it compounds, but it does not block the merge.

**`cargo metadata --locked` fails in CI.** `Cargo.lock` is stale relative to a manifest. Run
`cargo check` locally and commit the updated lockfile.
