# License Verification & Attribution Obligations (OQ-1)

Resolves **OQ-1** (`frankentts-oq1-license-mq7`). Primary-source verification against the pinned
bytes, required before any artifact publishing. Plan §2.2 tagged the license `[REPORTED]`; this
promotes it to **[VERIFIED]**, including the `NOTICE`/weights-LICENSE questions, which were probed
at the pinned revisions (§5).

Citation keys (paths relative to `docs/truth-pack/snapshots/`):

| Key | File |
|---|---|
| `LIC` | `gh/LICENSE` (201 lines, sha256 `a44a6081c73ad75f0255bb2bb5cab74ef1829565a895a24e53a4f11290ab7655`) |
| `HFRD` | `hf/README.md` (model card) |
| `SRC` | `docs/truth-pack/SOURCES.tsv` |

---

## 1. Verdict

**Apache License 2.0, verbatim and unmodified, weights included. No additional restrictions found in
the pinned bytes.**

### 1.1 Code repository (`QwenLM/Qwen3-TTS`)

`LIC` is the canonical Apache-2.0 text:

- **201 lines** — matches the canonical file length.
- All nine numbered sections present exactly once, plus `END OF TERMS AND CONDITIONS` and the
  `APPENDIX`.
- **The only deviation from the canonical text is the sanctioned appendix substitution**: line 189
  reads `Copyright 2026 Alibaba Cloud` where the canonical template reads
  `Copyright [yyyy] [name of copyright owner]`. That is precisely what the appendix instructs.
- Scanned for added terms — *acceptable use*, *additional restriction*, *shall not be used*,
  *non-commercial*, *research only*, and vendor names inside the license body: **no hits.** The only
  match for "commercial" is line 162, which is canonical §8 ("other commercial damages or losses").

Copyright holder: **Alibaba Cloud, 2026**.

### 1.2 Weights repository (`Qwen/Qwen3-TTS-12Hz-0.6B-Base`)

`HFRD` frontmatter declares `license: apache-2.0` (line 2). The model card body contains **no
additional terms** — a scan for *license / terms / restrict / prohibit / acceptable use / agreement*
returns only that frontmatter line. `hf/.gitattributes` references no license file.

**There is no separate weights license and no extra weights nuance in the pinned bytes.** See §5.1
for the strength-of-evidence caveat.

---

## 2. Obligations when we redistribute `.fttsq` (Apache-2.0 §4)

Our `.fttsq` is a **derivative work** of Apache-2.0 weights (quantization is a modification), so all
four §4 conditions attach to every artifact we publish:

| §4 | Obligation | How `ftts` satisfies it |
|---|---|---|
| **(a)** | Give recipients a copy of the License | Embed the **full Apache-2.0 text** in the `.fttsq` license section; `ftts license --full` prints it |
| **(b)** | Cause modified files to carry prominent notices **stating that You changed them** | The `.fttsq` header records the transformation explicitly (§3) — quantization is a change and must be *stated*, not merely implied by the format |
| **(c)** | Retain all copyright, patent, trademark, and attribution notices from the source | Carry `Copyright 2026 Alibaba Cloud` verbatim; do not strip or reword |
| **(d)** | If the source includes a `NOTICE` file, include its attribution notices in the derivative | **MOOT** — neither repo carries a `NOTICE` file at its pinned revision (§5). Nothing to propagate |

§4 also permits our own copyright statement for our original contributions (the kernels, the format,
the converter), provided it does not alter the License terms for the upstream material.

---

## 3. The attribution text to embed

Concrete string for the `.fttsq` `license_notice` field and for `ftts --version` / `ftts license`:

```text
This artifact contains model weights derived from
Qwen3-TTS-12Hz-0.6B-Base (https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
and code derived from QwenLM/Qwen3-TTS (https://github.com/QwenLM/Qwen3-TTS).

  Copyright 2026 Alibaba Cloud
  Licensed under the Apache License, Version 2.0.
  http://www.apache.org/licenses/LICENSE-2.0

CHANGES: the original bfloat16 weights were converted to franken_tts's
quantized .fttsq container. Tensors were requantized (see the artifact's
quantization manifest for the exact per-tensor policy); no weight values are
bit-identical to the originals. The model graph is re-implemented in Rust.

franken_tts runtime code: Copyright <year> <holder>, Apache-2.0.
```

Requirements on that text, so it stays honest:

- The `CHANGES:` paragraph is **§4(b) compliance**, not marketing — it must name what changed.
- It must reference the artifact's own quantization manifest rather than hard-coding a recipe, since
  the recipe is per-artifact and is a measured decision (plan §6.3).
- The upstream copyright line is **verbatim**; never merge it with ours.
- `ftts license` must be able to emit the **full** Apache-2.0 text, not just this summary — §4(a)
  requires the license itself, and a summary does not satisfy it.

---

## 4. What this unblocks

Artifact publishing is **legally clear to proceed** on the evidence in the pinned bytes.
Specifically: `frankentts-p2-fttsq-format-wsa` may embed the §3 notice, and the release epic's
checklist may treat license verification as satisfied. The only §4 condition needing engineering
attention is **(b) — state your changes** (§3's `CHANGES:` paragraph); (a) and (c) are mechanical,
and (d) is moot.

---

## 5. Gap closure — probed at the pinned revisions

Two gaps were open on first pass (no HF-side LICENSE file; `NOTICE` absence unobserved rather than
proven, since `SRC` is a curated enumeration and not a mirror). Both are now **closed with positive
evidence**, probed at the **pinned revisions** rather than `main`:

| Probe (at pin) | Result |
|---|---|
| `gh` `LICENSE` @ `022e286b…` | **HTTP 200**, sha256 `a44a6081…` — **byte-identical to our snapshot** |
| `gh` `NOTICE` @ `022e286b…` | **HTTP 404** — no NOTICE file |
| `hf` `LICENSE` @ `5d839924…` | **HTTP 404** — no LICENSE file |
| `hf` `NOTICE` @ `5d839924…` | **HTTP 404** — no NOTICE file |

Consequences:

1. **§4(d) is moot.** Neither repository carries a `NOTICE` file at its pinned revision, so there are
   no NOTICE attribution notices to propagate into `.fttsq`. §2's row (d) is resolved, not
   conditional. (It becomes live again only if a future pin introduces one — the re-fetch script's
   `--verify` mode is where that would surface.)
2. **The weights license is model-card frontmatter, knowingly and solely.** The HF repo carries no
   LICENSE file at the pin, so `HFRD`'s `license: apache-2.0` is the whole of the evidence. That is
   the normal Hub convention and is how the Hub itself renders and filters licensing, so it is
   adequate — but it is recorded here as *sole* evidence rather than assumed to be corroborated by a
   file we never checked.
3. **Bonus: the snapshot is independently re-verified.** The `gh` LICENSE re-fetched at the pin hashes
   to exactly the snapshot's `a44a6081…`, an independent confirmation that the truth pack's copy is
   the pinned byte sequence and has not drifted.

Recommended (housekeeping, not a blocker): add these four paths to `SRC` as `probe`-class entries so
`--verify` re-asserts the two 404s. A NOTICE appearing upstream later is a real obligation change,
and only an enumerated probe will catch it.

### 5.1 Related, tracked elsewhere

Upstream watermarking (`frankentts-oq8-watermark-pqi`, doctrine #10 — preserve if present) is a
separate obligation and is **not** covered by this bead.

---

## 6. Dispositions

| # | Item | Disposition |
|---|---|---|
| 1 | Plan §2.2 `License: Apache-2.0 [REPORTED]` | **[VERIFIED]** for the code repo against `LIC`; weights per §5.1 |
| 2 | LICENSE verbatim? | **[VERIFIED]** canonical 201-line Apache-2.0; sole deviation is the sanctioned appendix copyright substitution |
| 3 | Separate weights-license nuance? | **[NONE FOUND]** in the pinned bytes |
| 4 | Copyright holder | **Alibaba Cloud, 2026** |
| 5 | §4(a)(b)(c) obligations | **[RESOLVED]** §2–§3, with §4(b) "state your changes" called out explicitly |
| 6 | §4(d) NOTICE obligation | **[MOOT — VERIFIED]** no `NOTICE` in either repo at its pinned revision (§5) |
| 7 | HF-side LICENSE file | **[VERIFIED ABSENT]** 404 at the pin; weights license is model-card frontmatter, knowingly sole evidence (§5) |
| 8 | Snapshot integrity | **[RE-VERIFIED]** `gh/LICENSE` re-fetched at the pin hashes to the snapshot's `a44a6081…` |
