# Truth Pack — Pin Record

**Pinned 2026-08-06** by `cc_1` for bead `frankentts-t-pin-sources-b2u`.
These revisions are the bytes every `[SOURCE]` fact in the plan is asserted against.
Changing a pin invalidates [`FACT_DISPOSITIONS.md`](FACT_DISPOSITIONS.md) and requires re-adjudication.

## The pins

| Origin | Identifier | Pinned revision | Upstream date |
|---|---|---|---|
| HF weights repo | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `5d83992436eae1d760afd27aff78a71d676296fc` | 2026-01-29T08:01:19Z (`lastModified`) |
| GitHub source repo | `QwenLM/Qwen3-TTS` | `022e286b98fbec7e1e916cb940cdf532cd9f488e` | 2026-03-17T06:38:41Z (`fix finetuning bug`) |
| Paper | arXiv `2601.15621v1` (PDF) | version-locked URL `arxiv.org/pdf/2601.15621v1` | 2026-01-23T01:25:37Z (`Last-Modified`) |

The two pinned revisions are ~7 weeks apart. **The GitHub code is newer than the weights.** Any
behavioral question is answered by the *code* at its pin executed against the *weights* at theirs —
which is exactly the upstream-supported combination (`pip install qwen-tts` + `from_pretrained`), but
it means "the source" and "the checkpoint" are not one artifact. Record which pin answers which
question when citing.

Repo revision was obtained from `GET https://huggingface.co/api/models/{repo}` (`.sha`) and
`GET https://api.github.com/repos/{repo}/commits/main`. The HF `.sha` is the git commit of the model
repo, so `resolve/{sha}/{path}` is immutable even if `main` moves.

## How to reproduce

```bash
docs/truth-pack/fetch-truth-pack.sh                  # populate snapshots/ (38 files, ~13 MB)
docs/truth-pack/fetch-truth-pack.sh --verify         # offline: re-hash against MANIFEST.sha256
docs/truth-pack/fetch-truth-pack.sh --verify --refetch   # online: detect upstream drift
docs/truth-pack/fetch-truth-pack.sh --with-weights   # + 2.5 GB safetensors, verified by LFS oid
```

`snapshots/` is gitignored; only the hashes are committed. A `core`- or `support`-class mismatch
exits `2` and is a **STOP-THE-LINE** event — upstream force-pushed or re-tagged, and every dependent
bead's premises must be re-checked before more work lands. `drift-ok` covers the arXiv HTML render
only (arXiv rerenders `v1` HTML from unchanged source; the PDF is the authority).

## Weight-shard integrity

There are **no shards** — two unsharded safetensors files, no `*.index.json`
(see CORRECTION C-6). Integrity is anchored on the Git-LFS object id, which for LFS-tracked files
*is* the SHA-256 of the content, so the pack verifies the weights without carrying 2.5 GB. Recorded
in [`WEIGHTS.lfs.json`](WEIGHTS.lfs.json); `--with-weights` downloads and re-hashes to prove it.

| File | Bytes | SHA-256 (LFS oid) |
|---|---:|---|
| `model.safetensors` | 1,829,344,272 | `180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6` |
| `speech_tokenizer/model.safetensors` | 682,293,092 | `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258` |

Total 2,511,637,364 B ≈ 2.51 GB, matching the plan's "repo ≈ 2.52 GB". HF reports
`safetensors.total = 914,643,008` BF16 parameters — the plan's "system ≈ 0.9B params" is
**[VERIFIED]**, and `914,643,008 × 2 = 1,829,286,016` accounts for `model.safetensors` to within
58,256 bytes of header/metadata.

## Runtime pins for the reference oracle (feeds OQ-15)

Per the bead's standing rule, these come from **package metadata, never config metadata**:

| Source | Pin |
|---|---|
| `pyproject.toml` (GH pin) | `qwen-tts` **0.1.1**, `transformers==4.57.3`, `accelerate==1.12.0`, `requires-python>=3.9` |
| `config.json` (HF pin) | `transformers_version: 4.57.3` |
| `config.json` → `speech_tokenizer` `encoder_config` | `transformers_version: 4.57.0.dev0` |

The top-level config agrees with the package pin **this time**, but the *nested* encoder config does
not — `4.57.0.dev0` vs `4.57.3` inside one file. The config-metadata trap is real in this checkpoint;
the rule stands unchanged. **`torch` is not pinned anywhere upstream** — it is not even a direct
dependency (it arrives via `torchaudio`, unpinned). `librosa`, `soundfile`, `sox`, `onnxruntime`,
`einops`, `gradio` are all unpinned. **OQ-15 must therefore choose and freeze our own torch/librosa
versions and assert them at oracle runtime; there is no upstream pin to inherit.** Unpinned `librosa`
is the sharpest risk: it supplies the mel filterbank for the speaker encoder (OQ-9), and mel
implementations drift across releases.

### OQ-15 resolution — frozen local CPU oracle environment

The source-package pins above are authoritative for `qwen-tts`, `transformers`, and `accelerate`.
For the packages the upstream project leaves unpinned, the local oracle smoke environment is
**Python 3.13.9; `torch==2.7.1`; `torchaudio==2.7.1`; `librosa==0.11.0`;
`soundfile==0.13.1`**, with `qwen-tts==0.1.1`, `transformers==4.57.3`, and
`accelerate==1.12.0`. It uses eager attention, BF16 parameters, and `device_map=cpu`. The fixture
generator must assert these versions before producing or consuming oracle fixtures; configuration
metadata is not an acceptable substitute.

At GitHub source pin `022e286b98fbec7e1e916cb940cdf532cd9f488e`, this environment loaded the
model on Apple Silicon CPU and ran canonical-greedy x-vector synthesis twice with the same one-second,
24 kHz, 220 Hz in-memory reference, `text="Hello."`, and `max_new_tokens=2`. Each run produced
1,920 finite 24 kHz samples, with identical waveform SHA-256
`55a955c119292ba1df21d460c1a90eaacbd2384a6b9543cf26e5f546bc24289a`. The source enforces a
minimum of two new tokens; `max_new_tokens=1` is not a valid smoke because it leaves no hidden states.
The curated truth-pack snapshot alone is not importable for this run because it omits
`qwen_tts.core.tokenizer_25hz.vq`; use a full checkout verified at the source pin.

**Performance-baseline decision:** this proves fixed-runtime CPU execution and repeatability only.
It does **not** prove CPU-to-CUDA canonical-greedy token equality (the host had no CUDA device), so
official CPU is **not an admissible G2 performance incumbent** and no CPU/GPU ratio may be reported.
Correctness fixtures remain frozen from the unmodified native-device oracle; a CUDA run with
codec-token capture over the declared corpus must establish the cross-device nondeterminism envelope.

## Non-pinnable reference

Plan §17 also cites `doc.rust-lang.org/.../lints/levels.html` for `forbid` being un-overridable.
That is a living document with no stable revision, so it is **not** in the manifest. The fact it
supports is better established by compilation than by citation: the workspace bead
(`frankentts-p0-workspace-zri`) already requires a deliberate `unsafe` block in `ftts-core` to FAIL
the build. That test, not the URL, is the evidence.
