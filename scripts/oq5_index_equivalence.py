#!/usr/bin/env python3
"""OQ-5 guard: the microdecoder's training-mode causal pass and its 15-step sequential
inference loop must agree, position by position, on which embedding table and which output
head are used.

Why this exists: FrankenMTP (plan §7.5) uses the training-mode single causal pass as an exact
block verifier for a drafted residual sequence. An off-by-one in the
depth -> sequence-position -> head map silently invalidates the entire epic — the verifier
would score the right tokens with the wrong heads and still look plausible. This script
replays both index computations symbolically and asserts they are identical.

It is deliberately dependency-free (stdlib only): it proves the *index algebra* transcribed
from the pinned source, and runs in CI with no torch, no weights, no oracle. It does NOT
prove activation equality — that is the numeric oracle trace, tracked separately (see
docs/truth-pack/OQ5_MICRODECODER_WIRING.md §8).

Source of truth (pinned snapshot):
  docs/truth-pack/snapshots/gh/qwen_tts/core/models/modeling_qwen3_tts.py
Transcribed from:
  Qwen3TTSTalkerForConditionalGeneration.forward_sub_talker_finetune   lines 1612-1633
  Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward_finetune  lines 1197-1247
  Qwen3TTSTalkerForConditionalGeneration.forward (inference call site)  lines 1670-1687
  Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward     lines 1250-1312
  _update_model_kwargs_for_generation                                  lines 1314-1319

Exit 0 = maps agree. Exit 1 = divergence (print shows the first offending position).
"""

from __future__ import annotations

import sys

# From config.json -> talker_config.code_predictor_config (pinned truth pack).
NUM_CODE_GROUPS = 16

# Symbolic markers for the two distinct embedding tables. They are NOT interchangeable:
# position 1 reads the talker's 3072-vocab codec embedding, every later position reads one
# of the 15 code_predictor tables (vocab 2048).
TALKER_EMBED = "talker.embed_tokens[vocab=3072]"
PRED_EMBED = "code_predictor.codec_embedding[vocab=2048]"
TALKER_HIDDEN = "talker_hidden_state"


def training_map(num_code_groups: int = NUM_CODE_GROUPS) -> list[dict]:
    """Replay forward_sub_talker_finetune's sequence assembly + forward_finetune's heads.

    Sequence assembly (lines 1619-1626):
        sub_talker_inputs_embeds = [talker_hidden_states.unsqueeze(1)]
        for i in range(num_code_groups - 1):
            if i == 0:  append(self.get_input_embeddings()(codec_ids[:, :1]))
            else:       append(self.code_predictor.get_input_embeddings()[i-1](codec_ids[:, i:i+1]))

    Heads (lines 1235-1238):
        for i in range(1, num_code_groups):
            logits.append(self.lm_head[i-1](hidden_states[:, i]))
    """
    seq: list[dict] = [
        {"pos": 0, "table": TALKER_HIDDEN, "table_index": None, "token": None}
    ]
    for i in range(num_code_groups - 1):
        if i == 0:
            seq.append(
                {"pos": 1, "table": TALKER_EMBED, "table_index": None, "token": "c0"}
            )
        else:
            seq.append(
                {
                    "pos": i + 1,
                    "table": PRED_EMBED,
                    "table_index": i - 1,
                    "token": f"c{i}",
                }
            )

    heads: dict[int, int] = {}
    for i in range(1, num_code_groups):
        heads[i] = i - 1

    for entry in seq:
        entry["head"] = heads.get(entry["pos"])
        entry["predicts"] = f"c{entry['pos']}" if entry["pos"] in heads else None
    return seq


def inference_map(num_code_groups: int = NUM_CODE_GROUPS) -> list[dict]:
    """Replay the 15-step generate loop.

    Prefill (call site lines 1670-1673): inputs_embeds = cat(past_hidden, embed_talker(c0)),
    so shape[1] == 2 and forward's prefill branch (line 1277-1278) sets
        generation_steps = inputs_embeds.shape[1] - 2  ->  0
    Head (line 1299): logits = self.lm_head[generation_steps], taken at the last position.
    Then generation_steps advances by 1 (line 1311, propagated at lines 1314-1319).

    Generation step k (line 1281):
        inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
    where input_ids is the token just sampled, landing at the next sequence position.
    """
    seq: list[dict] = [
        {"pos": 0, "table": TALKER_HIDDEN, "table_index": None, "token": None},
        {"pos": 1, "table": TALKER_EMBED, "table_index": None, "token": "c0"},
    ]

    # Prefill: shape[1] == 2.
    generation_steps = 2 - 2
    seq[1]["head"] = generation_steps
    seq[1]["predicts"] = f"c{generation_steps + 1}"
    seq[0]["head"] = None
    seq[0]["predicts"] = None
    generation_steps += 1

    # 14 further appended positions; the 15th residual is sampled from the last head and
    # never re-enters the microdecoder.
    for _ in range(num_code_groups - 2):
        sampled = f"c{generation_steps}"  # token produced by the previous head
        pos = generation_steps + 1
        seq.append(
            {
                "pos": pos,
                "table": PRED_EMBED,
                "table_index": generation_steps - 1,
                "token": sampled,
                "head": generation_steps,
                "predicts": f"c{generation_steps + 1}",
            }
        )
        generation_steps += 1

    return seq


def main() -> int:
    train = training_map()
    infer = inference_map()

    failures: list[str] = []

    if len(train) != len(infer):
        failures.append(
            f"sequence length differs: training={len(train)} inference={len(infer)}"
        )
    if len(train) != NUM_CODE_GROUPS:
        failures.append(
            f"expected a {NUM_CODE_GROUPS}-position sequence, training map has {len(train)}"
        )

    for t, i in zip(train, infer):
        for field in ("pos", "table", "table_index", "token", "head", "predicts"):
            if t[field] != i[field]:
                failures.append(
                    f"position {t['pos']}: {field} training={t[field]!r} inference={i[field]!r}"
                )

    # The structural invariants the epic depends on, asserted explicitly rather than implied.
    scored = [e for e in train if e["head"] is not None]
    if len(scored) != NUM_CODE_GROUPS - 1:
        failures.append(f"expected 15 scored positions, got {len(scored)}")
    if [e["head"] for e in scored] != list(range(NUM_CODE_GROUPS - 1)):
        failures.append("heads are not lm_head[0..14] in sequence order")
    if train[0]["head"] is not None:
        failures.append("position 0 (talker hidden) must not be scored")
    if train[1]["table"] != TALKER_EMBED:
        failures.append("position 1 must read the TALKER embedding table, not a predictor table")
    if any(e["table"] == TALKER_EMBED for e in train[2:]):
        failures.append("only position 1 may read the talker embedding table")

    print("pos  table                              idx  token  head        predicts")
    for e in train:
        idx = "-" if e["table_index"] is None else str(e["table_index"])
        head = "-" if e["head"] is None else f"lm_head[{e['head']}]"
        print(
            f"{e['pos']:>3}  {e['table']:<34} {idx:>3}  "
            f"{e['token'] or '-':<5}  {head:<11} {e['predicts'] or '-'}"
        )

    if failures:
        print("\nFAIL: training-mode and inference index maps diverge (OQ-5 Tier 1 broken)")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(
        "\nOK: training-mode causal pass and 15-step sequential loop agree on every "
        "position, embedding table, and head (OQ-5 Tier 1)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
