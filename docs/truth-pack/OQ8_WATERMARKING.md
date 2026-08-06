# OQ-8 — Does the pinned Qwen3-TTS stack watermark generated audio?

**Bead:** `frankentts-oq8-watermark-pqi` · **Resolved by:** AzureThrush · **Date:** 2026-08-06

## Verdict: **NO.** No watermarking exists anywhere in the pinned stack.

There is no watermark embedder, no perceptual-hash stage, and no audio postprocessing of any kind
between the codec decoder's output and the waveform returned to the caller. Doctrine #10's
"preserve any upstream watermark" obligation is therefore **vacuous for this checkpoint** — there is
nothing to preserve. It is *not* discharged for all time: see §4.

Status: **[VERIFIED-STATIC]** — exhaustive search + line-level trace of the output path over the
pinned truth-pack snapshot. No weights on this host, so no audio was generated and analysed
(see §4 for why that matters and what it would add).

---

## 1. Evidence — exhaustive search

Case-insensitive search for every watermarking term and known audio-watermark library across the
**entire** pinned snapshot (`gh/` — modeling, inference, processing, tokenizers, finetuning,
examples, README; and `hf/` — all configs and tokenizer files):

```
watermark | wm_ | perth | audioseal | silentcipher | steg | fingerprint | mark_audio | embed_mark
```

**Zero matches in any code or config file.**

The single hit anywhere in the truth pack is in the paper's HTML rendering:

```html
docs/truth-pack/snapshots/paper/2601.15621v1.html:246
  <div id="watermark-tr">arXiv:2601.15621v1 [cs.SD] 22 Jan 2026</div>
```

That is **arXiv's own page-header stamp**, not a property of the model. Recorded explicitly so the
next person who greps does not re-open this on a false positive.

Neither `README.md` (GitHub, 59 KB) nor the HF model card mentions watermarking.

## 2. Evidence — the output path, traced end to end

`Qwen3TTSTokenizerV2Model.decode` (`tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:993-1024`) is
the terminal audio stage:

```python
audio_lengths = (audio_codes[..., 0] > -1).sum(1) * self.decode_upsample_rate
audio_codes  = torch.clamp(audio_codes, min=0)
audio_values = self.decoder.chunked_decode(audio_codes.transpose(1, 2)).squeeze(1)
audio_values = [a[:l] for a, l in zip(audio_values, audio_lengths)]
return Qwen3TTSTokenizerV2DecoderOutput(audio_values)
```

RVQ codebook lookup → causal decoder → **length trim only**. Nothing else touches the samples.

Its only caller in the cloning path, `generate_voice_clone`
(`inference/qwen3_tts_model.py:620-632`), likewise applies **only a slice** (§3) before returning
`(wavs_out, fs)`. There is no filter, dither, normalisation, resample, or embed step between the
decoder and the user.

⇒ **A faithful port of the codec decoder reproduces the reference waveform exactly**, with no
hidden stage to account for. Any systematic waveform difference we later observe is our bug, not an
unported watermark — which is precisely the ambiguity this bead existed to remove.

## 3. Two side-findings for the codec beads (the real value here)

Tracing the output path surfaced two behaviours that are **not** in the plan and that materially
affect codec parity and streaming.

### 3.1 The reference codec frames are prepended before decode, then cut off the audio

`inference/qwen3_tts_model.py:613-631`:

```python
codes_for_decode.append(torch.cat([ref_code_list[i], codes], dim=0))   # ref frames PREPENDED
wavs_all, fs = self.model.speech_tokenizer.decode([...])
cut = int(ref_len / max(total_len, 1) * wav.shape[0])
wavs_out.append(wav[cut:])                                             # ref audio CUT OFF
```

In ICL mode the reference's own codec frames are decoded **together with** the generated frames, and
the corresponding audio prefix is discarded afterwards. The reference audio is never returned — but
it is **computed**, and it primes the causal decoder's ring-buffer state so the generated speech
begins with a correctly warmed receptive field rather than from zeros.

**Consequences:**
- **Decoding only the generated codes is NOT equivalent** and will differ at the onset. Our codec
  engine must reproduce this warm-up, either by decoding the reference frames and discarding, or by
  precomputing the equivalent ring-buffer state into the `.ftvoice-cache` (the better design — it is
  exactly the kind of recomputable derived state §6.7 describes, and it removes the reference decode
  from the TTFA path entirely). **Whichever we choose must be proven equivalent, not assumed.**
- **TTFA impact:** naively, first audio costs an extra `ref_frames` worth of codec decode (≈38 frames
  for a 3 s reference). Caching the primed state removes it. This belongs in the −1B TTFA model.
- **This interacts with `streaming == batch` (plan §9.5):** the standing gate must fix the warm-up
  convention on both sides, or it will fail for reasons that have nothing to do with ring buffers.

### 3.2 The cut is computed as a float ratio, not a frame count

`cut = int(ref_len / total_len * wav.shape[0])`. Because `decode` trims to exactly
`n_frames × decode_upsample_rate`, this is **algebraically** `ref_len × decode_upsample_rate` — but
it is *implemented* as a float multiply-and-truncate. A port should compute
`ref_len × upsample_rate` directly (exact integer) and record the equivalence; the two can differ by
a sample only if `wav.shape[0]` is ever not an exact multiple of the frame count, which the trim
above forbids. Cheap to get right, annoying to debug as a one-sample offset in a parity test.

Both findings are owed to `frankentts-oq7-codec-details-kjz` (closed — comment filed), the codec
kernel beads, and `frankentts-k-voice-cache-i4t`.

## 4. Scope of this verdict — what it does and does not license

**Does license:** shipping without watermark-preservation machinery for *this* pinned checkpoint;
stating in the README that the upstream model embeds no watermark.

**Does NOT license:**
- **A claim about the audio itself.** This is a source audit, not a signal analysis. A watermark
  baked into the *decoder weights* — rather than applied as a code stage — would be invisible to
  `grep` and would be ported by us automatically (which satisfies doctrine #10 either way, since we
  reproduce the decoder faithfully). No evidence suggests this; it is simply out of reach without
  weights and a detector, and is not worth pursuing absent a reason.
- **Any other checkpoint or revision.** The verdict is bound to the revision in
  `docs/truth-pack/PIN_RECORD.md`. **Re-run the §1 search at every model-revision bump** — it costs
  one grep. If a future revision adds watermarking, we preserve it (doctrine #10), and the codec
  parity baseline changes.
- **The ethics posture.** Doctrine #10's other obligations — consent attestation in `.ftvoice`, no
  audio-acquisition features — are independent of this finding and unaffected.

**README implication:** the current README says the project "preserves any upstream watermarking."
That remains the correct *policy* statement and needs no change, but it should not be read as
implying a watermark exists. Recorded here rather than editing the README, which another agent may
be holding.
