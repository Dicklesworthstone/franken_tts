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
