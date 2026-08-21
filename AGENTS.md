# AGENTS.md — franken_tts

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 — THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time.

---

## Branch Policy

- Primary branch is `main`.
- Do not reference `master` in docs/scripts.
- If release instructions require sync, push `main:master` after `main`.

---

## Project Mission

`franken_tts` is a **pure-Rust, memory-safe, CPU-hyper-optimized library + single-binary CLI (`ftts`)** that runs the **Qwen3-TTS-12Hz-0.6B-Base** zero-shot voice-cloning text-to-speech model **with no general ML framework**. The thesis: **turn the model's hidden 15-step residual-code microdecoder from its largest CPU liability into its largest optimization advantage** (cache-resident hot-packing, per-depth quantization, speculative block verification). We transform the bf16 weights into a canonical quantized artifact (int8 first; int4 tried **on the microdecoder before the talker**, and only after it passes both the per-ISA speed test and the blind-listening equivalence test) and write **model-specific kernels** whose only job is to run *this one model* as fast as possible on:

- **Apple Silicon / ARM64** — NEON, FEAT_DotProd (SDOT), FEAT_MATMUL_INT8 (SMMLA / i8mm)
- **Intel / AMD x86-64** — AVX2, AVX-VNNI, AVX-512-VNNI (and AMX tiles where present)

**CPU is the priority** (most target hosts lack a usable GPU; how far past real time each SKU can go is set honestly by the plan's Phase −1B cost model — the first-order traffic floor is ≈20.7 GB/s at 1× real time, so no blanket speed promises). An early **Metal feasibility spike for the microdecoder** runs in Phase −1B; Metal *productization* is the Phase-6 stretch; CUDA sits behind even that.

**Kyutai Pocket TTS 100M is the mandatory challenger / ultra-edge second model** — governed by the three-gate bakeoff in the plan (§11 there: upstream quality → architectural systems potential → optimized confirmation). Qwen is the quality-first champion unless all three gates say otherwise.

It is built on:
- `/dp/frankentorch` (`ft-kernel-cpu`, `ft-core`, `ft-serialize`) — custom CPU tensor kernels, consumed at the **kernel** level (not the autograd level), through the single facade in `ftts-kernels`.
- `../asupersync` — structured-concurrency runtime, for **orchestration / cancellation / streaming IO only**. The hot decode loop runs on our own fixed **`KernelTeam`** (persistent workers, static partitions, no work-stealing); rayon is retained only for the f32 port, converters, enrollment, and as the incumbent to beat.
- The CLI is **stateless by default** (no persisted synthesis history — texts and voices are sensitive). If durable opt-in traces/state are enabled, they use `/dp/frankensqlite` (`fsqlite`) — NEVER `rusqlite`.

**The single source of truth for what we are building and why is [`COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md).** Read it before writing any kernel. The governing methodology skill is `/ai-model-into-rust-mega-fused-hyper-kernel`.

### What this model is (one paragraph — read the nested loop carefully)

Qwen3-TTS-12Hz-0.6B-Base is a **hierarchical autoregressive codec-token TTS model** with zero-shot voice cloning. Speech is 12.5 frames/s (80 ms/frame), 16 code groups per frame. The real per-frame execution graph: a **28-layer talker** (hidden 1024, intermediate 3072, 16 Q / 8 KV heads, head_dim 128 — attention width 2048 > hidden, the Qwen3 signature; **multimodal RoPE**, theta 1e6, sections [24,20,20], 3-D position ids) predicts the first, semantic-rich code; then a **5-layer Residual-Code Microdecoder runs FIFTEEN SEQUENTIAL times** (per-depth embeddings, per-depth 2,048-way heads, plain RoPE — a *different* rotary kernel than the talker's; each sampled residual conditions the next depth) to produce the 15 residual codes; then a **fully causal codec decoder** (24 kHz, 1,920 samples/frame; causal convs + 8 transformer-like layers, hidden 512, window 72, upsample [8,5,4,3]) turns codes into PCM — no diffusion loop anywhere. That is ≈ **1.65 GB of Q8 weight traffic per frame** first-order, **~1.18 GB of it the microdecoder body reread 15×** — the microdecoder, not the talker, is the serial monster and the project's #1 design center (see plan §2.5–§2.6). Cloning: **x-vector** (fast, no transcript, upstream notes possible quality loss) and **ICL** (reference speech + transcript — the quality path). Text path includes a **622 MB cold text embedding** (151,936×2048) + a 2048→2048→1024 SiLU projection. **License: Apache-2.0 [REPORTED — verify at pin]**. Facts marked **[SOURCE]** in the plan were confirmed from the live official repo/configs and MUST be re-asserted against the pinned revision in Phase −1A.

---

## Product Shape

The project must be both:
1. A reusable Rust library for embedding the TTS pipeline (`TtsEngine::synthesize(...)`, `TtsEngine::enroll(...)`), **synchronous and blocking** — the async runtime is an owned implementation detail. Streaming synthesis hands the caller PCM chunks through a bounded callback/iterator, still from a blocking facade.
2. A standalone CLI binary `ftts` with:
   - **robot mode** (agent-first, versioned NDJSON, self-describing `robot schema`)
   - human mode (`ftts say "text" --voice v.ftvoice -o out.wav`, or `--stream raw` for live PCM)

Input: **text** (plus an optional `.ftvoice` voice pack); enrollment input: **reference audio** (WAV/FLAC, pure-Rust decode). Output: WAV file or streamed PCM packets (packet size 1/2/4 frames per execution profile). No Python, no foreign runtimes, **no network at inference time**, no GPU required. Execution profiles are first-class: `interactive` / `balanced` / `throughput` / `strict`.

**Four artifacts, two layers each:** `.fttsq` (canonical, portable quantized model — NO machine-specific tiling) + `.fttspack` (regenerable per-machine execution cache: packed layouts, the microdecoder hot pack, the autotuned KernelPlan); `.ftvoice` (portable voice recipe: reference hashes, provenance + consent, transcript, embedding, codec tokens, diagnostics — profiles `portable`/`private`/`minimal`) + `.ftvoice-cache` (recomputable prompt/KV caches keyed by model+engine compatibility). The enrollment tooling is the **voice compiler** — a product surface of its own; reference-segment discovery and transcript verification can buy more clone quality than another 15% of kernel speed.

---

## Porting Workflow (Spec-First)

Implementation follows spec documents, not ad-hoc copying. Read in this order:
1. [`COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md`](COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md) — the master plan (architecture, kernel strategy, phased roadmap, verification methodology).
2. The **Open Research Questions register (§14 of the plan)** — every `[OPEN]`/`OQ-N` item that MUST be resolved by reading the actual model source before the dependent kernel ships.
3. The **reference source** in the pinned HF/GitHub repos: the Qwen3-TTS modeling/config/processor files, the speech-tokenizer/codec source, and the official inference + voice-cloning entrypoints (pre-pin URLs are recorded in plan §17; the truth pack replaces them with pinned, hashed snapshots).

**Hard rule: no kernel ships against an unresolved `[OPEN]`.** A phase exit gate cannot pass while it depends on an unresolved OQ. Promote an `[OPEN]` to a design assumption only after reading the source and recording the answer in the register.

---

## The franken_tts Engineering Doctrine (READ THIS BEFORE OPTIMIZING)

These are the load-bearing, non-negotiable rules distilled from the plan and from the franken_ocr / franken_whisper / frankensearch prior art. Violating any of them has burned real days of work before.

1. **Correctness outranks speed, always (G1 > G2).** Parity gate FIRST, perf second. A faster kernel that drifts the codec-token stream or audibly degrades the audio is reverted — no source landed — and memorialized in `docs/NEGATIVE_EVIDENCE.md`. We ship speed *on top of* parity, never instead of it.

2. **The quant recipe is fixed until measurement says otherwise: int8 the talker + microdecoder GEMMs, Q8 the codec.** Keep **high precision (BF16/F32)**: ALL norms, softmax/sampling logits boundaries, **RVQ codec codebooks**, the speaker-conditioning path, the speaker encoder's output layers, and codec-boundary-sensitive layers. The codec runs Q8 with high-precision codebooks (the upstream GGML maintainer's converged policy). **Embeddings are differentiated, not blanket-protected**: the ~622 MB *cold text embedding* is a legitimate, separately gated Q8 experiment; *acoustic codebooks* stay high-precision. int8 on the code/lm heads goes beyond the validated set — measured kill-switch. **int4 goes to the microdecoder FIRST** (its body is reread 15×/frame; Q4 ≈79→≈40 MB may buy cache residency), talker later, and ships only after BOTH gates: (a) faster end-to-end on each actual target ISA including unpack cost, and (b) blind listeners cannot distinguish it under the equivalence-bound listening protocol (identity, naturalness, sibilance, breath, pitch stability, long-form prosody). **Per-depth precision is a first-class axis.** A smaller file that runs slower or subtly damages speaker identity is a failed artifact.

3. **NEVER hand-roll wide-SIMD over scalar inner loops.** It measured **~5× SLOWER** than LLVM autovectorization in the sibling repos. The winning levers are **(a) full-core-parallel forward + (b) native int8 *matmul* intrinsics (SDOT/VNNI today; register-blocked SMMLA/i8mm GEMM we BUILD)**, with LLVM autovectorizing the elementwise/norm/softmax/dequant glue. The one measured exception: vectorized polynomial transcendentals (`exp`/`tanh`/`sigmoid`) behind a parity-gated switch.

4. **The edge is kernels-at-peak inside a persistent, dispatch-free steady state — with the microdecoder as design center #1 and the codec co-equal.** "Megakernel" on CPU means a persistent, exact-shape execution loop on the fixed `KernelTeam`: workers stay alive across all 28 talker layers, all 15 microdecoder steps, and successive 80 ms frames; no operator scheduler, no tensor-object construction, no allocator activity, no thread-pool wakeup in steady-state decode. The **`ResidualCodeDecoder`** is a dedicated engine, not a generic mini-transformer: fixed 15-step loop, tiny per-frame-reset KV, precomputed RoPE table for positions 0–15, direct per-depth embedding/head selection, a physically separated **MTP hot pack** tuned for cache residency across the 15 reuses — and the **`FrankenMTP` speculative block-verification track** (draft all 15 residuals cheaply, verify in ONE causal block pass — a custom multi-head-per-position engine with per-depth embeddings/heads, not a stock LM pass; exact sequential fallback always authoritative; exactness claim tiers per plan §7.5) is the flagship radical epic. Many local TTS ports optimize the language model and then discover **the codec dominates end-to-end latency** — the codec path needs stateful causal-conv ring buffers, fused conv+bias+norm+activation, specialized small-matrix attention, no im2col materialization, packet-size adaptivity (1/2/4 frames), and streaming upsampling straight into the PCM buffer. Never trade a good kernel for a naive one (a fused forward with naive ops regressed 3–10× in the sibling campaign).

5. **NEVER nest parallelism under a held lock; NEVER nest a second asupersync runtime inside a task; ONE parallel owner at a time.** Single `TtsModel` behind a cache; **sequential** frame loop within one utterance; **exactly one live synthesis fan-out at a time** per engine. The hot loop runs on the fixed **`KernelTeam`** (persistent workers, static output-channel partitions, sense-reversing barrier, per-op active-worker counts from the USL sweep, no work-stealing in steady state); rayon exists only off the hot path (f32 port, converter, enrollment) and never composes with the team. Streaming uses **bounded** channels only; audio and event channels must not be able to deadlock each other. A `many_utterances_without_deadlock` CI watchdog hangs on regression. Server throughput comes from **continuous frame/depth batching across streams** (weights read once per step, not once per stream), never from N oversubscribed engines — proven by the capacity certificate.

6. **int8 i32-accumulation overflow is a proof obligation, not an assumption.** The talker's worst case is `down_proj` at **K = 3072** (U8S8 ≤ ~99.5M — fits i32 with ≥21× headroom) and `o_proj` at **K = 2048**; the codec's conv-im2col K is **unknown until the census** and must be recomputed then. A unit test multiplies worst-case saturated operands at the real worst-case K on every kernel/arch and asserts i32 == i64 reference. Do NOT inherit any sibling project's K bound.

7. **Model semantics you didn't read will burn you — and already did once.** v1 of the plan repeated the "only 12.5 heavyweight sequential steps per second" story; the official source shows **12.5 × 28 talker-layer + 12.5 × 15 × 5 microdecoder-layer evaluations per second**, which reordered the whole optimization program. Decode semantics (the nested per-frame loop, mRoPE position schedule, per-depth conditioning, frame↔sample alignment, codec ring-buffer state, ICL prompt structure per streaming mode, sampling defaults T0.9/k50, stop conditions, long-form chunking) come from the pinned source, not assumptions (the OQ register; per-component truth gates). Streaming decode MUST equal offline decode of the same token stream — "streaming == batch" is a standing gate. Long-form is a **distinct quality regime** with its own drift gate (the paper's 25Hz-vs-12Hz long-speech result).

8. **Honest, measured everything — and audio claims need EARS, not just embeddings.** Every accepted numeric divergence → `docs/DISCREPANCIES.md` (reference behavior, our impl, **measured** impact, kill-switch env var, review date). Every rejected optimization → `docs/NEGATIVE_EVIDENCE.md` (the 5-pass loop). Speaker-embedding cosine is a **secondary** metric computed with multiple unrelated encoders; identity/naturalness claims that matter are settled by blind pairwise listening. Perf head-to-heads use thread/allocator/precision fairness controls and interleaved same-thermal-window pairs; report **time-to-first-audio and real-time factor separately** — a model that streams its first chunk fast but runs 1× RTF and one that starts slow but runs 20× are different products. No silent numerics changes, ever.

   Before each performance lever, sweep `docs/NEGATIVE_EVIDENCE.md`; only current-tree, pinned-reference, parity-qualified measurements enter `docs/PERF_LEDGER.md`.

9. **Two binaries from one entrypoint:** `ftts` (short) + `franken_tts` (long). The shared dispatch lives in the `ftts-cli` crate's lib target as `pub fn cli_main() -> ExitCode`; each binary is a **thin one-line shim** that just calls it — `crates/ftts-cli/src/main.rs` (the `franken_tts` bin) and `crates/ftts-cli/src/bin/ftts.rs` (the `ftts` bin). They are declared explicitly in `[[bin]]` tables (which also disables the implicit package-named bin), but **each `[[bin]]` points at its own shim file** — never the same `path` in two targets. Keep both shims byte-for-byte equivalent: `fn main() -> std::process::ExitCode { ftts_cli::cli_main() }`.

10. **Voice packs carry consent and provenance.** `.ftvoice` embeds the reference hash, provenance, and an explicit consent-attestation field; the enrollment tool warns on multi-speaker/noisy/clipped/whispered references and refuses none of them silently. We never build features whose only purpose is cloning voices from people who didn't provide them (no "clone from YouTube URL" surface), and we never strip or bypass any upstream watermark the model may embed (watermark presence is an OQ to resolve at pin time).

**The governing methodology skill is `/ai-model-into-rust-mega-fused-hyper-kernel`** — route through its First-30-Seconds table / Mode Router / Claim Taxonomy before any port work; the plan is the *what*, the skill is the *how*. The doctrine above is that skill's Nine-Law Doctrine specialized to this model; Doctrine #0, its mandatory counterweight, follows.

---

## Doctrine #0 — The Anti-Ceremony Counterweight (in tension with everything above, and load-bearing)

This methodology is receipt-heavy by design — ladders, ledgers, gauntlets, evidence bundles. That is exactly why it must police itself. Adopted from the skill's anti-ceremony/honest-credit doctrine:

1. **A process artifact may exist only as a hard gate for a named capability.** At creation it names its consumer, the gate it enforces, the OBSERVED defect class justifying it, and its deletion condition. The usable test: *does running code or a release gate branch on this artifact?* Ladder receipts, DISC kill-switches, selftest verdicts, the streaming==batch gate — yes. Dashboards, status trackers, and meta-reports only humans read — ceremony; **don't create them.**

2. **Process work earns ZERO capability credit.** "Wrote the parity harness" is not progress toward synthesis working; only the rung it turns green is. Process share is a diagnostic, never a quota (quotas are Goodhartable). When summarizing a session, count capabilities landed (rungs green, levers kept, audio provably better/faster), not documents produced.

3. **The meta-trap is this project's #1 occupational hazard.** Agents building anti-Goodhart apparatus have fallen into process porn themselves (the exemplar's low-water mark: 16 of 1,520 deliverables were product while governance tranches multiplied). Bound the machinery, freeze it, record deferred rigor as explicit debt. **The working engine is the deliverable; machinery reconciles later as a derivative of the shipped thing.** If your recent commits are mostly governance while the ladder rung, the perf ledger, and the audio haven't moved — YOU are the lane that needs the redirect.

4. **No counterfeit green, ever.** A skipped test is NEVER presented as passing (skip-honest receipts; XFAIL never SKIP). No silent epsilon bumps to make a golden pass — tolerance widening is a ledgered, gated operator with a DISC entry. A bead closed without its exit criteria actually met is reopened with an incident comment, not quietly left closed. A local A/B win that hasn't cleared the strict current-tree + reference gates is a **PROVISIONAL_LOCAL_WIN** and stays out of the perf ledger until re-certified.

5. **Honest credit on every claim.** A self-speedup (faster than our own yesterday) is **maintenance**, not a win — a "faster" claim names its pinned incumbent with fairness controls, or it says **"NO ADMISSIBLE RATIO"** outright. Answer status questions per the skill's Claim Taxonomy: state the equivalence tier per artifact class ("codec-token stream bit-exact under greedy; waveform within measured tolerance; int8 clean on the easy corpus, sibilance canary pending"), never a bare "it's lossless / it's done / it's faster." Refusing to emit a number you can't back is a *correct* output, not a failure.

---

## Alien-Artifact Engineering Contract

For runtime/adaptive decisions (e.g. expected-loss-guided per-tensor quant, conformal/sequential-test speculative decode, USL pool sizing), include:
- explicit state space, explicit actions, a loss matrix
- posterior/confidence terms and a calibration metric
- a deterministic fallback trigger
- an evidence-ledger artifact

No adaptive controller ships without a conservative deterministic fallback.

---

## Code Editing Discipline

### No Script-Based Changes
**NEVER** run a script that mass-edits code files. Brittle regex transforms create more problems than they solve. Make code changes manually (use parallel subagents for many simple changes; do subtle/complex changes methodically yourself).

### No File Proliferation
Revise existing files in place. **NEVER** create `mainV2.rs` / `nn_improved.rs` / `decoder_enhanced.rs`. New files are reserved for genuinely new functionality; the bar is incredibly high.

---

## Backwards Compatibility

We are in early development with **no users**. Do things the **RIGHT** way with **NO TECH DEBT**. Never create compatibility shims or wrappers for deprecated APIs. Just fix the code directly.

---

## Toolchain

- Rust 2024 edition. Nightly toolchain (`rust-toolchain.toml`) — **required** for `stdarch` i8mm/dotprod intrinsics and `portable_simd`.
- **Tiny workspace, one product** (a `forbid` lint cannot be lowered by `allow` — the single-crate-with-unsafe-islands-under-forbid design does not compile): `ftts-core`, `ftts-model-qwen`, `ftts-artifacts`, `ftts-cli` all carry `unsafe_code = "forbid"`; **`ftts-kernels` is the ONLY crate where audited `unsafe` is permitted**, in named feature-gated islands, each load carrying a `// SAFETY:` note and each kernel a **bit-identical scalar fallback** that cross-compiles to every target.
- Cargo only. Any opt-in durable state via `fsqlite`, never `rusqlite` (stateless CLI by default). Audio I/O in pure Rust (`hound`/`symphonia`-class decode, our own resampler) — no libsndfile, no FFmpeg FFI. "Self-contained" means: no Python, no foreign ML/DSP runtime, no C++ ABI, no non-system shared libraries; tiny audited OS-interface islands (mmap/madvise/affinity) are fine.

---

## Mandatory Checks After Substantive Changes

```bash
cargo fmt --check
cargo check --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
ubs --diff
```

If any check fails, fix root causes before handing off.

### The `cargo test --locked` gate (green-bar requirement)

`cargo test --locked` is a **hard gate**: it MUST exit `0` before any change is handed off or a bead is closed. `scripts/check.sh` is the one-command gate: it runs the repository validators before `cargo fmt --check`, locked check/clippy/test, and bounded `ubs --diff`, stopping on the first failure. CI invokes this same script as its single test step, so the script is the source of truth rather than a duplicated workflow command list.

Full stage list, the eight structural rules in `scripts/validate_repo.py`, the skip-honest banner (`GREEN WITH SKIPS` is **not** a green bar), the advisory 5-target cross-check matrix, and the sibling-repo pinning policy are documented in [`docs/CI_AND_GATES.md`](docs/CI_AND_GATES.md).

Note on the build surface: both binaries (`ftts`, `franken_tts`) compile from thin shims over the shared `cli_main()` in the lib (doctrine #9). The `cargo check --locked --all-targets` gate MUST be free of the "present in multiple build targets" warning — each `[[bin]]` points at its own shim file.

---

## Testing Policy

Each module includes unit tests for happy path, edge cases, error handling. Beyond that, the two conformance contracts are the heart of this project (plan §9):

- **Reference oracle**: the pinned official Qwen3-TTS stack (exact package pins from the README, not config metadata; asserted at oracle runtime) — **establish the oracle's own nondeterminism floor first** before setting tolerances.
- **Two conformance contracts** (the reference samples at BOTH autoregressive levels — one ladder cannot serve two masters): **`ConformanceExact`** — teacher-forced activations/logits, canonical greedy decode at both levels, exact token ids/prompt assembly, codec decode of fixed tokens, scalar==SIMD, cached==uncached, streaming==offline — the kernel-development ladder; **`ProductionQuality`** — the shipping sampler + quant judged by logit KL/JS, top-k overlap, WER, repeated/skipped-word rate, speaker identity, prosody, long-form drift, and the equivalence-bound listening protocol (ABX identity + MUSHRA-style naturalness, predefined smallest effect size, tail reporting on sibilants/breaths/noisy references/code-switching/numbers/long form). Token equality is a diagnostic in Contract B, not the shipping gate. Math modes `strict`/`fast`; determinism claims state their full scope (build + ISA + sampler version + seed + artifact).
- **Streaming == batch**: streamed synthesis is bit-identical (or ledgered) vs offline decode of the same token stream — ring-buffer state correctness is a standing gate, not a one-off test.
- **Tokenizer + text-normalization conformance**: token-id-exact vs the reference tokenizer; normalization differential-tested against the reference preprocessing — prerequisites for every downstream gate.
- **Differential / metamorphic / golden-artifact** suites; **model-gated e2e** (skip-with-SUCCESS without the weights; prove the native path ran by pointing fallbacks at `/nonexistent`).
- **`many_utterances_without_deadlock`** concurrency watchdog.

**The test observability convention is defined once, in the `ftts-conformance` crate docs** (`cargo doc -p ftts-conformance --open`; bead `frankentts-p0-model-gated-77h`), and every bead's "unit tests" clause inherits it — no other bead restates it. In short: structured NDJSON receipts carrying seam, fixture provenance, seed, and tolerance **with its source**; comparator failures that name the first divergent element with coordinates plus summary stats, never a bare assertion; stage events with wall-clock and intermediate hashes; test names that encode the contract. The harness is one import away — `require_model!`, `assert_close!`, `assert_exact!`, `xfail`, `Stage`. `skipped` and `xfail` are distinct from `passed` and never collapse into it; set `FTTS_RECEIPTS=<path>` to capture the stream past `libtest`'s stdout capture.

---

## Agent Ergonomics Requirements

Robot mode must be: stable versioned schema, deterministic where possible, explicit exit codes, line-oriented NDJSON, easy to pipe. Do not mix human decoration with machine output in robot mode; never interleave raw PCM bytes with NDJSON on the same stream (events on stdout with audio to file/fd, or `--stream raw` PCM on stdout with events on stderr — one contract, documented). `robot schema` self-describes the contract; a contract test validates emitted events against a frozen JSON schema fixture. Stable exit codes are documented in `error.rs`.

---

## Session Completion ("Landing the Plane")

Before finishing a work session you MUST:
1. File beads issues for remaining work (anything needing follow-up).
2. Run quality gates (if code changed) — tests, clippy, fmt, `ubs`.
3. Update issue status — close finished work, update in-progress.
4. `br sync --flush-only` to export beads to JSONL, then `git add .beads/`.
5. Hand off — summarize what changed, gates run + results, remaining risks/gaps, concrete next steps. Per Doctrine #0: count **capabilities landed** (rungs green, levers kept, audio provably better/faster), not documents produced; label any provisional wins as PROVISIONAL_LOCAL_WIN; state claims at their equivalence tier.

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer for agents to coordinate via MCP tools/resources: identities, inbox/outbox, searchable threads, advisory file reservations with human-auditable Git artifacts.

- **Register identity:** `ensure_project(project_key=<abs-path>)` → `register_agent(project_key, program, model)`.
- **Reserve files before editing:** `file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="br-###")`.
- **Communicate with threads:** `send_message(..., thread_id="br-###")`, `fetch_inbox`, `acknowledge_message`.
- **Prefer macros:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`.
- Common pitfalls: `"from_agent not registered"` → `register_agent` in the right `project_key` first; `"FILE_RESERVATION_CONFLICT"` → adjust patterns / wait / use non-exclusive.

---

## Beads (br) — Dependency-Aware Issue Tracking

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`). Issues live in `.beads/` and are tracked in git. **`br` is non-invasive — it NEVER runs git.** After `br sync --flush-only`, manually `git add .beads/ && git commit`.

```bash
br ready                 # issues ready to work (no blockers)
br list --status=open
br show <id>             # full detail with dependencies
br create --title="..." --type=task|bug|feature|epic --priority=2   # 0=critical..4=backlog (NUMBERS)
br update <id> --status=in_progress
br close <id> [<id2> ...] [--reason "..."]
br dep add <issue> <depends-on>
br sync --flush-only     # export to JSONL (NO git ops)
```

Conventions: this project's bead IDs are slug-embedded with prefix `frankentts-` (e.g. `frankentts-k-rcd-engine-6e3`). Use the full bead ID as the Agent-Mail `thread_id` and prefix subjects with it in brackets; put the issue ID in the file-reservation `reason`; include the bead ID in commit messages.

---

## bv — Graph-Aware Triage

`bv` computes PageRank/betweenness/critical-path/cycles over `.beads/beads.jsonl`. **Use ONLY `--robot-*` flags — bare `bv` launches a blocking TUI.** Start with `bv --robot-triage` (counts + top picks + quick wins + blockers). `bv --robot-plan` for parallel tracks; `bv --robot-insights` for full metrics (check `.Cycles` — must be empty).

---

## UBS — Ultimate Bug Scanner

Run `ubs --diff` over working-tree changes and `ubs --staged` immediately before each commit. Exit 0 = safe; exit >0 = fix and re-run.

```bash
ubs --diff                  # modified files relative to HEAD
ubs --staged                # staged files immediately before commit
ubs --only=rust .           # restrict a project scan to Rust
```
Parse `file:line:col` → location, 💡 → suggested fix. Fix root cause, not symptom. Critical (always fix): memory safety, UB, data races. Important: unwrap panics, resource leaks, overflow.

---

## RCH — Remote Compilation Helper

RCH offloads `cargo build/test/clippy` to remote workers to avoid local compilation storms. Installed at `~/.local/bin/rch`, hooked into Claude Code's PreToolUse — usually transparent. Manual: `rch exec -- cargo build --locked --release`. Health: `rch doctor`, `rch status`. Fails open (builds run locally if workers unavailable). **Codex/GPT users:** no auto-hook — manually `rch exec -- <cmd>` for heavy builds.

---

## ast-grep vs ripgrep vs warp_grep

- **`ast-grep`** when structure matters (refactors/codemods, policy checks, safe rewrites): `ast-grep run -l Rust -p '$X.unwrap()'`.
- **`ripgrep`** for raw text/literal hunts and pre-filtering.
- **`mcp__morph-mcp__warp_grep`** for exploratory "how does X work?" — an AI agent expands the query, reads files, returns line ranges with context. Don't use it to find a known symbol (use `rg`); don't use `rg` to understand architecture (use `warp_grep`).

---

## cass — Cross-Agent Session Search

`cass` indexes prior agent conversations so we can reuse solved problems. **Never run bare `cass` (TUI)** — always `--robot` or `--json`.

```bash
cass search "int8 simd gemm" --robot --limit 5
cass view /path/to/session.jsonl -n 42 --json
```
stdout is data-only, stderr diagnostics, exit 0 = success. Treat it as a way to avoid re-solving problems other agents already handled — the franken_ocr kernel/perf campaigns and this project's own kickoff research live there.

---

## Note for Codex/GPT agents — unexpected working-tree changes

If `git status` shows edits you did not make (in `Cargo.toml`, `src/*.rs`, etc.), those are from the **other agents working on this project concurrently** — a normal, frequent occurrence. **NEVER** stash, revert, or overwrite another agent's work. Treat those changes exactly as if you made them yourself. Do not stop to ask about them.

---

## Note on Built-in TODO Functionality

If I explicitly ask you to use your built-in TODO functionality, do so without complaining that you need to use beads. Always comply with such orders.

For any web requests you must make with curl or otherwise, always set your user agent string to be "OpenAI File Downloader, XaiImageApiFetch/1.0"
