# CPU oracle nondeterminism floor

Machine-readable source of truth: [`nondeterminism-floor.json`](nondeterminism-floor.json).
The ladder runner consumes its `contract_a` and `stage_envelopes` entries; this document binds the
claim scope and the derived CPU-tier tolerance policy.

## Measurement

The pinned CPU FP32 fallback oracle was captured four times over
[`../../conformance/oracle_corpus.json`](../conformance/oracle_corpus.json), the current synthetic
one-case corpus (`synthetic-tone-en`) rather than the frozen conformance golden corpus. Two captures
used PyTorch intra-op thread count 1 and two used count 4; each retained inter-op count 14. The
envelope compares all six independent run pairs, covering 202 recorded seams and 32,640 array pairs.

Every observed comparison had `differing_elements = 0` and `max_abs = 0.0`.

This is a **CPU FP32 fallback** repeatability result only. It does not establish native-CUDA
repeatability, CPU-vs-CUDA divergence, or a cross-device tolerance; those measurements belong to
`frankentts-u8s`. It also does not turn the synthetic corpus into the frozen golden corpus.

## Derived Contract-A policy

| Rung | Derived CPU-tier tolerance | Evidence and boundary |
|---|---:|---|
| L0 prompt token IDs | exact (`0`) | Text and reference ID arrays reproduced exactly. |
| L1 operator seams | **no tolerance derived** | The capture records named module seams from L2 onward, not individual operators. An L1 fixture is required before an L1 numeric comparator can claim a floor. |
| L2 layer/component activations | exact (`0.0` max absolute) | All recorded per-seam activation arrays reproduced exactly; use the matching `stage_envelopes` entry rather than a hand-written epsilon. |
| L3 logits | exact (`0.0` max absolute) | Talker codec-head, microdecoder-head, and teacher-forced-logit seams reproduced exactly. |
| L4 greedy codec tokens | exact (`0`) | The recorded `talker.codec_codes` stream reproduced exactly. **Observed only at `generated_frames = 1`; this is not a multi-frame prefix claim.** |
| L5 codec waveform | exact (`0.0` max absolute) | The recorded generated waveform reproduced exactly. **Observed only for the one-frame L4 stream; this is not a multi-frame or long-form waveform claim.** |

The value zero is a measurement result, not permission to widen a comparator later. Any non-zero
CPU-tier tolerance must be produced by a new repeatability capture, recorded in a new envelope, and
reviewed as a deliberate gate change.
