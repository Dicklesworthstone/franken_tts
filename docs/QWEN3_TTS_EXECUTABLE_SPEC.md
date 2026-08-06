# Qwen3-TTS executable specification

This specification is extracted from the pinned truth pack, not inferred from
configuration names. Each section records the exact source pin it depends on.

## SPEC-OQ4 — Talker mRoPE position schedule

**Authority.** GitHub source pin `022e286b98fbec7e1e916cb940cdf532cd9f488e`
and HF model-config pin `5d83992436eae1d760afd27aff78a71d676296fc`; see
[`docs/truth-pack/PIN_RECORD.md`](truth-pack/PIN_RECORD.md). Source locations
below are relative to the snapshotted source tree.

### Contract

The talker's mRoPE has three axes and is interleaved with sections `[24, 20,
20]`, `rope_theta = 1_000_000`. However, for this TTS path those three axes
receive the **same position index at every sequence element**. This is a 3-D
tensor representation of a single causal position stream, not a modality-aware
text/reference-audio/generated-frame schedule.

For a prefill attention mask `M[b, j]` (`1` = real element, `0` = padding), the
reference calculates:

```text
p[b, j] = cumsum(M[b, :])[j] - 1
p[b, j] = 1                         if M[b, j] == 0
position_ids[axis, b, j] = p[b, j]  for axis in {0, 1, 2}
```

This is verbatim behavior from
`gh/qwen_tts/core/models/modeling_qwen3_tts.py:1794-1800`. The rotary builder
consumes all three rows (`:546-559`) and the talker attention applies the
interleaved multimodal rotary transform (`:660-724`, called at `:773-780`).

The public `generate` path composes role text, speaker/reference material,
target text, and codec-prompt embeddings into **one** `talker_input_embeds`
sequence, then left-pads that sequence and derives `talker_attention_mask`
(`:2236-2254`). Because `get_rope_index` accepts only that mask (`:1746-1800`),
none of those element classes receives a distinct axis value. The Rust port must
therefore pass the same scalar causal index to each of the three axes for all
valid prefill elements.

### Decode transition and `rope_deltas`

On the initial prefill, the reference calculates:

```text
delta0       = count(M == 0)
mrope_delta  = max(position_ids) + 1 - sum(M)
rope_deltas  = mrope_delta - delta0
```

and caches `rope_deltas` on the talker (`modeling_qwen3_tts.py:1693-1705`).
For the normal left-padded prompt with at least two valid elements, this reduces
to `rope_deltas = -left_pad_count`. Each subsequent one-element talker decode
uses:

```text
decode_position = cache_position[0] + rope_deltas
position_ids = [decode_position, decode_position, decode_position]
```

(`:1706-1711`). Thus the first generated talker frame has position equal to the
un-padded prompt length, and later frames increment by one. This remains true
at the prompt-to-audio boundary: there is no 12.5 Hz or 13 Hz rescaling, no
axis-specific offset, and no reset.

`position_id_per_seconds = 13` is configuration metadata
(`hf/config.json:146`) but has no read site in the pinned `qwen_tts` source
tree. It is **not** part of the executable position calculation above. Treating
it as “13 position ids per second” would create a parity-breaking schedule.

### Worked left-padding trace

For a batch whose longest composed prompt is eight embeddings, take one
eight-element prompt and one six-element prompt. The reference mask and all
three mRoPE rows are:

| Item | Mask | `position_ids[0] = position_ids[1] = position_ids[2]` | First decode position |
|---|---|---|---:|
| 8-element prompt | `[1,1,1,1,1,1,1,1]` | `[0,1,2,3,4,5,6,7]` | `8 + 0 = 8` |
| 6-element prompt, left-padded to 8 | `[0,0,1,1,1,1,1,1]` | `[1,1,0,1,2,3,4,5]` | `8 - 2 = 6` |

The two padding positions are masked from attention; their placeholder values
are never a semantic coordinate. The first valid element of each prompt is
position zero, and the first generated frame continues at the corresponding
un-padded prompt length.

### Port obligations and proof boundary

1. Build a dedicated talker mRoPE path: three equal position rows plus the
   interleaved `[24,20,20]` channel selection. Do not reuse the microdecoder's
   plain RoPE path.
2. Preserve left-padding compensation exactly; cache the signed
   `rope_deltas` per utterance/prefill and apply it to every talker decode step.
3. Add oracle-derived mRoPE Q/K test vectors for both rows in the table before
   declaring the L1 talker rotary rung green. This document supplies
   source-derived expected integer positions; it does **not** substitute for
   oracle activation parity.

**OQ-4 verdict:** resolved from the pinned source for the implemented base-model
path. The `position_id_per_seconds` field is non-operative configuration
metadata in that path. The remaining required proof is the downstream oracle
fixture, not another schedule interpretation.
