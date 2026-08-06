#!/usr/bin/env python3
"""OQ-10 guard: prompt shapes and the target-text-independent prefix, for all four
(clone mode x streaming mode) combinations.

Why this exists: prompt assembly is Contract-A L0 -- a one-position error corrupts every
downstream gate -- and the .ftvoice-cache design is only sound if the cached prefix provably
cannot attend to the target text. This script is the executable statement of both: it computes
each template's position layout from the pinned source's construction, and asserts the prefix
bound. frankentts-p1-prompt-igr must reproduce these numbers exactly.

Dependency-free by design (stdlib only): it models the SHAPE arithmetic transcribed from the
pinned source. It does NOT tokenize -- token-exact verification is owed to OQ-11 / p1-prompt-igr
(see docs/truth-pack/OQ10_ICL_PROMPT_STRUCTURE.md §7).

Source of truth (pinned snapshot):
  docs/truth-pack/snapshots/gh/qwen_tts/core/models/modeling_qwen3_tts.py
Transcribed from:
  Qwen3TTSForConditionalGeneration.generate            lines 2126-2234  (header + mode branches)
  Qwen3TTSForConditionalGeneration.generate_icl_prompt lines 1968-2019  (the two ICL templates)
  Qwen3TTSTalkerForConditionalGeneration.forward       lines 1689-1692  (per-frame trailing text)

Exit 0 = all invariants hold. Exit 1 = a shape or prefix invariant is violated.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

ROLE_TOKENS = 3  # "<|im_start|>assistant\n" -- the wrapper prefix (I:269)


@dataclass(frozen=True)
class Layout:
    mode: str
    prompt_len: int  # total prefill positions
    trailing_len: int  # embeddings fed one-per-frame after prefill
    prefix_len: int  # maximal target-text-INDEPENDENT prefix
    ref_frames_in_prefix: int


def header_len(language_given: bool, speaker_embed: bool) -> int:
    """H = 3 role positions + (L_c - 1) summed positions.  M:2149-2186

    codec_input_embedding = P (3 or 4) ++ [speaker_embed]? ++ [codec_pad, codec_bos]
    and only its first L_c-1 entries are consumed here (the final codec_bos is held back).
    """
    p = 4 if language_given else 3  # M:2136-2147
    l_c = p + (1 if speaker_embed else 0) + 2
    return ROLE_TOKENS + (l_c - 1)


def layouts(
    n_ref_id: int,  # reference transcript tokens  = ref_ids[:, 3:-2]
    n_text_id: int,  # target text tokens          = input_id[:, 3:-5]
    n_ref_frames: int,  # reference codec frames
    language_given: bool = True,
    speaker_embed: bool = True,
) -> list[Layout]:
    h = header_len(language_given, speaker_embed)
    t1 = n_ref_id + n_text_id + 1  # text_embed: ref ++ text ++ eos      M:1978-1981
    t2 = 1 + n_ref_frames  # codec_embed: codec_bos ++ frames    M:1983-1998

    out: list[Layout] = []

    # --- ICL x non-streaming: two sequential blocks.            M:2002-2013
    out.append(
        Layout(
            mode="icl/non-streaming",
            prompt_len=h + t1 + t2,
            trailing_len=1,  # tts_pad_embed
            prefix_len=h + n_ref_id,
            # the codec block sits AFTER the target text -> causally contaminated
            ref_frames_in_prefix=0,
        )
    )

    # --- ICL x streaming: elementwise sum, leftover text trails. M:2014-2019
    covered = min(t2, n_ref_id)
    out.append(
        Layout(
            mode="icl/streaming",
            prompt_len=h + t2,
            trailing_len=max(t1 - t2, 1),
            prefix_len=h + covered,
            # block position 0 is codec_bos; frames start at position 1
            ref_frames_in_prefix=max(covered - 1, 0),
        )
    )

    # --- x-vector x streaming: one position (first text token). M:2200-2202, 2230-2232
    out.append(
        Layout(
            mode="xvector/streaming",
            prompt_len=h + 1,
            trailing_len=(n_text_id - 1) + 1,
            prefix_len=h,
            ref_frames_in_prefix=0,
        )
    )

    # --- x-vector x non-streaming: whole text prefilled.        M:2203-2227
    out.append(
        Layout(
            mode="xvector/non-streaming",
            prompt_len=h + n_text_id + 2,
            trailing_len=1,
            prefix_len=h,
            ref_frames_in_prefix=0,
        )
    )
    return out


def check(n_ref_id: int, n_text_id: int, n_ref_frames: int, **kw) -> list[str]:
    h = header_len(kw.get("language_given", True), kw.get("speaker_embed", True))
    errs: list[str] = []
    for lay in layouts(n_ref_id, n_text_id, n_ref_frames, **kw):
        # The prefix is a prefix.
        if lay.prefix_len > lay.prompt_len:
            errs.append(f"{lay.mode}: prefix {lay.prefix_len} exceeds prompt {lay.prompt_len}")
        # The header is always cacheable; nothing less ever is.
        if lay.prefix_len < h:
            errs.append(f"{lay.mode}: prefix {lay.prefix_len} is shorter than header {h}")
        # x-vector modes cache the header and nothing more.
        if lay.mode.startswith("xvector") and lay.prefix_len != h:
            errs.append(f"{lay.mode}: prefix must equal header {h}, got {lay.prefix_len}")
        # The prefix must never reach the first target-text position.
        if lay.mode.startswith("icl") and lay.prefix_len > h + n_ref_id:
            errs.append(
                f"{lay.mode}: prefix {lay.prefix_len} would include target text "
                f"(first target position is {h + n_ref_id})"
            )
        # Non-streaming ICL can never cache a reference codec frame.
        if lay.mode == "icl/non-streaming" and lay.ref_frames_in_prefix != 0:
            errs.append("icl/non-streaming: reference codec frames follow the target text; none are cacheable")
        if lay.ref_frames_in_prefix > n_ref_frames:
            errs.append(f"{lay.mode}: claims more cached ref frames than exist")
    return errs


def main() -> int:
    # A representative enrollment: 3 s reference (12.5 fps), short transcript, medium target.
    scenarios = [
        ("3s ref, 12-token ref text, 40-token target", 12, 40, 38),
        ("10s ref, 40-token ref text, 40-token target", 40, 40, 125),
        ("long ref text exceeding the codec stream", 200, 40, 38),
        ("minimal: 1-token ref text, 1-token target, 1 frame", 1, 1, 1),
    ]

    failures: list[str] = []
    for title, n_ref_id, n_text_id, n_ref_frames in scenarios:
        h = header_len(True, True)
        print(f"\n{title}   (header H={h}, |ref_id|={n_ref_id}, |text|={n_text_id}, ref_frames={n_ref_frames})")
        print(f"  {'mode':<24} {'prompt':>7} {'trailing':>9} {'prefix':>7} {'ref frames cached':>18}")
        for lay in layouts(n_ref_id, n_text_id, n_ref_frames):
            pct = 100.0 * lay.prefix_len / lay.prompt_len
            print(
                f"  {lay.mode:<24} {lay.prompt_len:>7} {lay.trailing_len:>9} "
                f"{lay.prefix_len:>7} {lay.ref_frames_in_prefix:>18}   ({pct:.0f}% of prompt)"
            )
        failures += [f"[{title}] {e}" for e in check(n_ref_id, n_text_id, n_ref_frames)]

    # Header must not depend on the target text or the streaming mode.
    if len({header_len(True, True) for _ in range(2)}) != 1:
        failures.append("header is not deterministic")
    if header_len(True, True) != 9 or header_len(False, False) != 7:
        failures.append(
            f"header sizes drifted: expected 9 (language+speaker) and 7 (auto, no speaker), "
            f"got {header_len(True, True)} and {header_len(False, False)}"
        )

    if failures:
        print("\nFAIL: prompt-shape / prefix invariants violated")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(
        "\nOK: all four templates' shapes are consistent, and in every mode the "
        "target-text-independent prefix stops before the first target-text position."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
