# Frozen conformance corpus r1

This is the small, stable corpus for Qwen3-TTS conformance work. It is not the
Gate-A bakeoff corpus: it binds exact-oracle capture inputs and the tail-risk
strata that future Contract-B work must retain.

`manifest.json` pins the SHA-256 of the authored text source and names 28 text
IDs. The selection spans ordinary prose, dialogue, numbers, currencies, names,
acronyms, URLs, quotations, a long sentence, long form, and code switching.
It explicitly includes the sibilants, breaths, numbers, code-switching, and
long-form canary axes. `python3 scripts/conformance_corpus.py` fails closed if
the source text bytes, selected IDs, coverage, capture matrix, or bootstrap
fixture hash changes.

Every selected target is captured in x-vector and ICL cloning modes, each with
non-streaming and streaming prompts. Short, medium, and long target budgets
are declared in the manifest; the exact generator command remains
`scripts/gen_reference_fixtures.py`, which records the observed frame count.
The target budget never substitutes for observed EOS or cap-truncation.

The five long-form rows are deliberate boundary probes: the first and second
codec chunk seams, the 25-frame left-context prosody seam, the explicit
2,048-frame evaluation cap, and the 8,192-frame runtime-default cap. Above a
300-frame codec chunk boundary, comparison to whole-sequence codec decoding is
an expected, ledgered divergence, never a native parity regression.

The repository currently carries one deterministic 220 Hz sine wave only as a
CPU fixture-plumbing input. It is hash-pinned and classified
`nonhuman_fixture_only`; it is not a voice recording, is not permitted for
voice-cloning or listening claims, and does not satisfy the reference-audio
admission policy. Materializing the consent-clean 3/10/30-second clean, phone,
reverberant, and noisy reference selections requires externally supplied audio
and its per-record consent/provenance attestation. Native-CUDA capture belongs
to `frankentts-rf4` after that materialization.
