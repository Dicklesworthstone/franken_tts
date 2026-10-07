# Voice Compiler Design

> Skeleton committed with bead `frankentts-p4-ftvoice-format-x0p`; finalized as the Phase-4 beads
> land (segment discovery, transcript verification, multi-reference policies fold their design
> records in here).

## What the voice compiler is

The pipeline that turns a recording a person had the right to provide into a reusable voice:
decode → diagnose → select mode → extract identity → package. Its outputs are two files with
opposite lifetimes:

| File | Lifetime | Contents | Deletable? |
|---|---|---|---|
| `.ftvoice` | permanent | embedding, consent, transcript + codec tokens, diagnostics, recipe, (optionally) reference audio | **no** — it IS the voice |
| `.ftvoice-cache` | disposable | prompt-header KV, primed codec state, keyed to one engine config | yes — re-derived from the pack |

## Format layer (`crates/ftts-artifacts/src/voice.rs`)

Both containers share the `.fttsq` philosophy at pocket scale: 8-byte magic, versioned header,
sorted-key JSON directory, absolute-offset sections each carrying its own SHA-256. Hardening is
identical: checked arithmetic against real buffer length, capped counts, non-overlapping ranges,
digest verification before any payload is exposed, named refusals everywhere.

Decisions worth remembering:

- **No timestamps anywhere.** Byte-idempotence ("same input → identical pack") is a metamorphic
  gate, not a nicety; wall-clock data would break it. Provenance identifies software versions and
  content hashes instead.
- **Privacy profiles are enforced at READ time** (`FtVoiceError::ProfileViolation`). A file
  claiming `private` that carries embedded audio is refused as a lie about itself; writer-side
  checks mirror this so a lie cannot be serialized either.
- **Consent is inspectable, not hidden**: `attested: false` parses fine so tooling can show what a
  pack claims; synthesis-time behavior around unattested packs belongs to the enrollment bead.
- **Cache keys digest every component OQ-10 §5.1 names** — including `language_id` and
  `speaker_embed`, which sit *inside* the cached header positions and are easy to mistake for
  runtime options. A cache whose stored key digest disagrees with its own components is refused.
- **x-vector profiles get NO prefix-KV section** — the maximal target-independent prefix there is
  the 7–9-position header; the embedding is the reusable artifact (OQ-10 §5.1 verdict).

## ICL synthesis primes the codec with the reference (codec spec §5.3)

Upstream ICL decodes `reference ++ generated` codes as one sequence and cuts the reference's
samples, so generated audio inherits a codec state conditioned on the reference voice (the codec
transformer's information horizon is 568 frames). `ftts say --voice quality.ftvoice` reproduces
that exactly: the codec worker primes its streaming state with the pack's codec codes
(`CodecStreamingState::prime_reference`) before decoding generated frames — on this strictly causal
decoder, priming then pushing equals the concatenated decode's tail bit for bit
(`ftts-conformance/tests/icl_prefix_decode.rs`). Priming runs on the codec worker, overlapping the
talker prefill.

Priming decodes the whole reference — the dominant ICL time-to-first-audio term on hosts where the
codec is slow (≈37 s for a 17 s reference on a 4-vCPU x86 VM). It depends only on the reference
codes, the codec weights, and the codec's numerics route, so the primed state is kept as a
`.ftvoice-cache` blob, `codec_primed_state` (`CodecStreamingState::save`/`restore`: every retained
history, the transformer KV windows, the frame counter, f32 as exact bits), under
`~/.cache/franken_tts/voice-cache/<cache_key>.ftvoice-cache`, written atomically with owner-only
permissions. The key digests the reference codes, a **numerics fingerprint** (the PCM of a fixed
one-frame probe decoded from a fresh state, which moves with the weights and with every route, env,
and platform choice that affects codec arithmetic), the engine version, and a state-format ABI. A
restore is bit-identical to fresh priming (pinned at unit level and on real weights), so the cache
changes time, never audio; any miss, mismatch, or refused blob falls back to priming. Measured on
that VM: TTFA 40.3 s cold → 7.2 s warm, WAV byte-identical. `FTTS_VOICE_CACHE=0` disables it.

## Enrollment modes (bead `frankentts-p4-enrollment-en6`)

QUALITY / QUICK / AUTO are never presented as interchangeable equals:

- **QUALITY** — transcript-backed ICL: verified transcript + codec-encoder tokens + cached prompt
  state. The quality path.
- **QUICK** — x-vector only; upstream documents possible quality reduction and the CLI says so.
- **AUTO** — ICL when the transcript verifies, else x-vector WITH A LOUD WARNING.

Warnings are loud because continuation-style cloners reproduce prompt-recording defects — the user
is told before they hear it. Refusal (exit 8) is reserved for unusable input, with `--force` for
consenting adults. No acquisition features, ever (doctrine 10).

## Cache-key tuple

`{voice_recipe_hash, model_hash, prompt_builder_version, streaming_mode, quant_recipe, math_mode,
engine_abi}` from plan §6.7 plus `(language_id, speaker_embed, ref_transcript_tokens,
ref_codec_codes)` from OQ-10 §5.1. Every component invalidates; tested exhaustively in
`voice.rs`.

## Open design records (fold in as beads land)

- Segment discovery + audition ranking + loss model (`frankentts-p4-segment-discovery-qjc`)
- Transcript verification + ASR plugin contract (`frankentts-p4-transcript-verify-8z7`)
- Multi-reference policies
- Runtime prefix-KV capture and admission integration (`frankentts-k-voice-cache-i4t`)
