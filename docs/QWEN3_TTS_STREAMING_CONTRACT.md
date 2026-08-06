# Qwen3-TTS 12Hz — Streaming Runtime Semantics, Packetization & TTFA Protocol (OQ-14)

Resolves **OQ-14** (`frankentts-oq14-streaming-internals-pqz`). This bead owns the *runtime*
semantics; prompt-content ownership stays with OQ-10.

Citation keys (paths relative to `docs/truth-pack/snapshots/`):

| Key | File |
|---|---|
| `MODEL` | `gh/qwen_tts/core/models/modeling_qwen3_tts.py` |
| `INF` | `gh/qwen_tts/inference/qwen3_tts_model.py` |
| `PAPER` | `paper/2601.15621v1.html` (arXiv 2601.15621) |
| `CODEC` | `docs/QWEN3_TTS_CODEC_SPEC.md` (OQ-7) |
| `DECODE` | `docs/QWEN3_TTS_DECODE_AND_ADMISSION.md` (OQ-6) |

---

## 1. [FINDING] The released code has no streaming generation. The packetization is paper-only.

The bead asks for the official first-packet path, packet assembly, and flush behavior. The released
code contains **none of them**. Upstream says so itself, in the parameter docs repeated verbatim at
`INF:514-515`, `INF:656-657`, and `INF:754-755`:

> "Using non-streaming text input, this option currently only simulates streaming text input when
> set to `false`, rather than enabling true streaming input or streaming generation."

Confirmed by search: there is no generator/iterator/callback path, no packet assembly, no flush or
finalization routine, and no incremental codec invocation anywhere in `INF` or `MODEL`. Generation
runs to completion, and the codec then decodes the whole code sequence in one `chunked_decode` call
(`CODEC` §5.2). `non_streaming_mode` is **a prompt-construction switch only** (§2).

The 4-frame/320 ms packet and the first-packet latency figures come from `PAPER` §4, describing an
**internal serving system that was not open-sourced**:

> "To avoid excessive scheduling overhead caused by very small packets, we define one speech packet
> as 4 tokens, which means 320 ms of speech per packet." — `PAPER`

**Consequence for the bead's own acceptance criterion.** The bead requires that "our packet policies
must be able to reproduce the official behavior exactly in strict mode (conformance
packetization)". There is no released official packetization to reproduce. Strict-mode packetization
must therefore be defined **from the paper's description** (§3) and validated against the *invariant*
that matters — `streaming == batch` (`CODEC` §5.2) — rather than against a reference implementation
that does not exist. This is a genuine narrowing of what OQ-14 can deliver, recorded rather than
papered over.

Plan §2.7's "official system groups 4 frames into a 320 ms packet" is **[VERIFIED as a paper claim]**
and **[CORRECTED as an implementation claim]** — the tag should read `[SOURCE, paper]`, not
`[SOURCE]`.

---

## 2. What `non_streaming_mode` actually does — two different prompt geometries

It selects how the text stream is metered into the talker. The difference is structural and changes
prefill cost, TTFA, and what the model can still "see" of its own text.

### 2.1 ICL path (reference speech + transcript) — `MODEL:1968-2019`

| Mode | Prompt geometry | `trailing_text_hidden` |
|---|---|---|
| `non_streaming_mode=True` (`MODEL:2002-2013`) | text and codec are **concatenated**: `[text_embed + codec_pad_id] ++ [codec_embed + tts_pad]` — all text lands in **prefill** | `tts_pad_embed` (scalar) — text is exhausted before frame 0 |
| `non_streaming_mode=False` (`MODEL:2014-2019`) | text and codec are **summed position-wise** over the overlap; if `text_lens > codec_lens` the prefill takes `text_embed[:, :codec_lens]` and the **remainder becomes the trailing stream**; if shorter, text is right-padded with `tts_pad_embed` | `text_embed[:, codec_lens:]` — the tail streams in during generation |

### 2.2 x-vector / speaker-prompt path (non-ICL) — `MODEL:2198-2232`

| Mode | Prompt geometry | `trailing_text_hidden` |
|---|---|---|
| `non_streaming_mode=True` (`MODEL:2203-2227`) | the single seeded text token is dropped (`[:, :-1]`) and the **whole text + EOS** is appended with `codec_pad_id`, then a final `tts_pad + codec_bos` position | `tts_pad_embed` — pad fallback from frame 0 |
| `non_streaming_mode=False` (`MODEL:2228-2232`) | prefill carries only the **first text token** (`input_id[:, 3:4]`) | `text[4:-5] ++ tts_eos` — essentially the entire text streams in one hidden per frame |

### 2.3 Consequences

- **Simulated streaming meters text at 12.5 tokens/s** — one hidden per 80 ms frame
  (`MODEL:1689-1692`, `DECODE` §3.3). This is the mechanism, and it has a sharp edge: if the text is
  longer than the number of frames generated, **the tail of the text is never seen by the model**.
  The ICL branch's `text_embed[:, :codec_lens]` split exists precisely to manage that.
- **[TRAP] the default differs by entrypoint.** Voice clone defaults to `non_streaming_mode=False`
  (`INF:478`) — i.e. **streaming-simulated is the default on the cloning path we care most about** —
  while voice design and custom voice default to `True` (`INF:642`, `INF:738`). Any A/B that does not
  hold this constant is comparing two different prompt geometries and is invalid.
- **Prefill cost differs by orders of magnitude** between the two modes (whole text vs one token), so
  TTFA numbers are only comparable within a mode (§4).

---

## 3. Strict-mode packetization (defined, since there is nothing to copy)

From `PAPER` §4, specialized to the 12 Hz rate:

```
1 packet = 4 code frames = 4 × 80 ms = 320 ms of audio = 4 × 1920 = 7680 samples @ 24 kHz
```

**Definition for `ftts` strict mode:** emit PCM in 4-frame packets, first packet emitted as soon as
4 frames exist. Packet size stays an execution-policy parameter (1/2/4/auto) per plan §8.2/G8;
`strict` pins it to 4 to match the paper's system.

The correctness obligation is **not** "byte-identical to the official streamer" (which does not
exist) but the standing invariant: **any packet schedule must produce output identical to
whole-sequence decode of the same token stream.** Per `CODEC` §5.2 this is achievable and is
*strictly better* than the reference's `chunked_decode(300, 25)`, which is itself
chunk-approximate beyond 300 frames. Two rows follow:

| Regime | Our streaming vs our offline | Our output vs official `chunked_decode` |
|---|---|---|
| ≤ 300 frames (≤ 24 s) | **bit-identical** (standing gate) | **bit-identical** (single chunk, zero context) |
| > 300 frames | **bit-identical** (standing gate) | **measured divergence**, DISC row — not a regression |

Flush/finalization, having no reference, is our design: at EOS (`DECODE` §3.1) the final partial
packet is emitted with the true remaining frame count, and truncation without EOS must be reported
as a distinct outcome (`DECODE` §4), never silently flushed as if complete.

---

## 4. TTFA measurement protocol

### 4.1 The official numbers (`PAPER` §4, efficiency table) — our model's row

| Model | Concurrency | LM TTFP | Tokenizer decode (TPP) | **First-packet latency** | LM TPP | RTF |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-TTS-12Hz-1.7B | 1 | 97 ms | 4 ms | 101 ms | 21 ms | 0.313 |
| **Qwen3-TTS-12Hz-0.6B** | **1** | **93 ms** | **4 ms** | **97 ms** | **19 ms** | **0.288** |
| Qwen3-TTS-12Hz-0.6B | 3 | 174 ms | 5 ms | 179 ms | 22 ms | 0.338 |
| Qwen3-TTS-12Hz-0.6B | 6 | 294 ms | 5 ms | 299 ms | 30 ms | 0.434 |

Definitions are the paper's own: **First-Packet Latency = LM TTFP + tokenizer decode TPP**; LM TPP is
the steady-state LM time to produce one packet's tokens.

The widely-quoted "97 ms" belongs to **our exact model** (12Hz-0.6B at concurrency 1) — not to the
25 Hz variants, whose first-packet latencies are 138–150 ms at concurrency 1. Worth pinning, because
the 25 Hz rows sit directly above it in the same table.

### 4.2 [NO ADMISSIBLE RATIO] the official numbers are not a comparable baseline

`PAPER`: latency was measured "on our internal vLLM engine (vLLM V0 backend) on a single typical
computational resource with optimizations applied via torch". **No hardware is named.** There is no
device, no thread count, no precision statement.

Per plan §10.6 and Doctrine #0's claim taxonomy, we therefore **cannot** claim a ratio against 97 ms.
Any statement of the form "we beat / approach the official 97 ms" is inadmissible and must be
answered **"NO ADMISSIBLE RATIO"**. The 97 ms row is usable only as (a) a documented upstream
*claim*, and (b) evidence that the codec's contribution to first-packet latency is small (4 ms of
97 ms) — which is consistent with `CODEC` §6's finding that codec streaming state is only a few MB.

### 4.3 Our protocol

1. **Report TTFA and RTF separately, always** (plan doctrine #8) — the two official numbers above are
   already separated in the source table and must stay separated.
2. **Pin the definition to the paper's**: TTFA = time from `synthesize()` entry to the first PCM
   packet leaving the bounded queue = (talker prefill + 4 frames of nested decode) + (codec decode of
   those 4 frames). Report the two addends separately, mirroring the LM-TTFP / tokenizer-TPP split,
   so a regression can be attributed.
3. **Hold `non_streaming_mode` constant** across any comparison (§2.3) and state which geometry was
   used — prefill dominates TTFA and the two modes differ by the whole text length.
4. **Report per packet size** (1/2/4) — packet size trades TTFA against scheduling overhead and is
   the whole point of the `-1B` packet experiments.
5. Our own measurements use the plan's fairness controls (thread/allocator/precision parity,
   interleaved same-thermal-window pairs, cv% ≤ 5) and are compared **only against our own pinned
   incumbent**, never against §4.1.

---

## 5. Dispositions and follow-ups

| # | Item | Disposition |
|---|---|---|
| 1 | Official streaming generation / first-packet path / flush | **[RESOLVED — NOT IMPLEMENTED]** upstream states it only simulates streaming *text input* |
| 2 | Packet assembly (4 frames / 320 ms) | **[VERIFIED as PAPER claim]**, `[SOURCE]` → `[SOURCE, paper]` in plan §2.7; not in the released code |
| 3 | Strict-mode packetization | **[DEFINED]** 4 frames / 7680 samples, validated by `streaming == batch`, not by imitation |
| 4 | Prompt differences vs non-streaming | **[VERIFIED]** two geometries per clone path, §2.1–§2.2 |
| 5 | Entrypoint default divergence | **[TRAP RECORDED]** voice clone defaults to streaming-simulated; the other two do not |
| 6 | TTFA baseline | **[PINNED]** 12Hz-0.6B row = 93 + 4 = 97 ms, and **NO ADMISSIBLE RATIO** against it (hardware unnamed) |
| 7 | Flush/finalization semantics | **[OURS BY NECESSITY]** no reference exists; tie to the EOS rule and the truncation-reporting requirement |

### Consumers unblocked

`frankentts-b-packet-experiments-6rh` (§3 packet definition, §4.3 protocol),
`frankentts-k-packet-tuning-28u` (§3 strict-mode pin),
`frankentts-t-executable-spec-1ch` (§2 prompt geometries),
`frankentts-oq10-icl-prompt-lkq` (§2.1 is the runtime half of the ICL prompt structure).
