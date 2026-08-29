# frankentts-pbrj.8.1 — Physical-iPhone current-tree baseline and ranked hotspot ledger

Consumer: `frankentts-pbrj.8.2` (10+ pass one-lever optimization campaign) and `frankentts-pbrj.8.3`
(waking-the-model latency). Every hotspot row below cites the raw receipt file it is derived from.
Deletion condition: superseded by the 8.2 campaign's refreshed ranking.

## Provenance

- Device: iPhone 17 Pro Max (iPhone18,2), iOS 26.5.2, developer mode enabled.
- App: `FrankenTTS.app`, Release configuration, built from commit `4234b8d5`
  (Swift UI and profile lane); engine static lib `libftts_ffi.a`
  sha256 `80f077713b0096b75f914670039e86b230575bfc1e27d0f6d60f1347831a0795`
  (xcframework build of 2026-08-29 ~15:07 EDT; concurrent-agent Rust edits landed
  after 18:31 EDT and are NOT in the measured binary).
- Model: the downloaded artifact manifest is embedded verbatim in every receipt's
  `run_start.model_manifest` (SHA-256 per file).
- Lane: `FTTS_IOS_PROFILE=1` JSONL receipts (`LabView.runProfilingBenchmarkIfRequested`),
  extended in `4234b8d5` with short/long scenario texts, `enrolled:<name>` cloned-voice
  vectors, and a per-sample streaming pass (`packet_frames=1`) measuring TTFA.
- Campaign receipts (this tree, 2026-08-29): `ftts-ios-profile-2026-08-29T21-27-15Z.jsonl`
  (short×matt), `T22-01-52Z.jsonl` (short×enrolled:Jeff), `T22-06-19Z.jsonl`
  (long×matt), `T22-25-34Z.jsonl` (long×enrolled:Jeff). Historical Aug-28 receipts in
  this directory are from an older engine tree and are cited only as corroboration.

## Refusal rules (applied before any median below)

1. `thermal_state >= 2` (serious): row refused. This dropped 4 long×matt rows and 3
   long×Jeff rows; sequential synthesis heats the phone and inflates walls monotonically
   (e.g. short×matt drifts 4.70 s → 6.62 s while still `nominal`).
2. Rows without `run_complete` in their receipt are not counted (two cancelled receipts
   from screen-lock suspensions are excluded outright).

## Baseline numbers (accepted rows only)

| Cell | cold load_ms | cold-first wall / RTF / TTFA | warm wall med / RTF / TTFA | warm n |
|---|---|---|---|---|
| short×matt | 7242 | 5216 ms / 0.92 / 272 ms | 5146 ms / 0.90 / 286 ms | 11 |
| short×Jeff | 7330 | 6733 ms / 0.94 / 313 ms | 8237 ms / 0.80 / 314 ms | 12 |
| long×matt | 6391 | 24190 ms / 0.86 / 282 ms | 26667 ms / 0.80 / 325 ms | 2 |
| long×Jeff | 4850 | 31344 ms / 0.72 / 322 ms | 31344 ms / 0.72 / 322 ms | 3 |

RTF < 1.0 everywhere: the shipping iPhone route is slower than real time on sustained
runs even in a cool state; TTFA is excellent (272–325 ms at `packet_frames=1`).

Warm stage medians (ms), share of generation (long×Jeff accepted rows, n=3):

| stage | short×matt | short×Jeff | long×matt | long×Jeff | share (long) |
|---|---|---|---|---|---|
| microdecoder | 2272 | 3973 | 10405 | 12611 | ~41% |
| codec_active (overlaps generation) | 2584 | 3840 | 11148 | 13061 | ~42% |
| talker | 1093 | 1944 | 10917 | 12824 | ~41% |
| other_generation | 1510 | 2155 | 5112 | 6060 | ~19% |
| prefill | 52 | 54 | 61 | 68 | <1% |
| feedback | 1 | 1 | 3 | 3 | <0.1% |

`codec_active_ms` overlaps `generation_ms` (concurrent codec worker), so the stage
columns each quote the resource they consumed, not additive wall time.

## Golden audio fixtures (frozen)

Whole-buffer WAV SHA-256 per cell — bit-identical across every sample in the cell, and
the streaming route's joined PCM produced the identical digest in 35/35 measured samples
(`streaming_matches_whole_buffer`):

- short×matt: `ca415d04c4809f7504439985fc42f5783eb05caba4b7733d909db49a0861eed5`
- short×Jeff: `59872f7baff04a4fd05f8eada2d5b5c728f324ccc99d6a36df7043bb18323f71`
- long×matt: `dadc9aba0358139668494aa0853c026bb9edb58ab39fd16b5ba3b0d2f9d7f93d`
- long×Jeff: `01a635cb4fecdc2613d7f6f8ba9386a4ec591f1ee663e949e6ceced9926bf4b7`

Any 8.2 lever that changes these digests has changed the token stream — that is a parity
event, not a timing footnote, and must clear its own conformance gate.

## Ranked top-five costs (falsifiable ledger for 8.2/8.3)

1. **Microdecoder residual loop (~41% of generation).** Hypothesis H1: a cache-resident
   MTP hot pack (packed per-depth weights, hot-loop residency) cuts microdecoder_ms by
   ≥ 25% on long×Jeff accepted rows without changing any fixture digest. Kill switch:
   `FTTS_MTP_HOTPACK=0`. Falsified if the retained lever's ABBA pair shows < 10% or a
   digest mismatch.
2. **Codec decode (~42% concurrent).** Hypothesis H2: overlapping codec work already
   hides most of it behind generation; the residual serial tail after the last frame is
   the real target. Measure the last-packet-to-return gap; if < 5% of wall, H2 is
   falsified and codec optimization is deprioritized on A18.
3. **Talker (~41%).** Hypothesis H3: i8mm-routed attention/projection kernels (A18,2
   supports SMMLA) cut talker_ms ≥ 15% without digest changes. Verify the dispatched
   tier first via `ftts robot backends` on-device equivalent; if i8mm is already
   dispatched, H3 is falsified as stated.
4. **other_generation (~19%).** Hypothesis H4: KV-cache management and sampling glue
   scale with frames; profiling (signposts/xctrace on-device) attributes ≥ 10% of it to
   avoidable copies. Falsified if attribution finds no single site ≥ 5%.
5. **Cold model hydration 4.8–7.3 s (8.3's target).** Hypothesis H5: moving immutable
   preparation (bundle scan, header validation, table hydration) to install/idle time
   saves ≥ 1.5 s of the cold path; the retained engine across UI transitions then makes
   warm-start TTFA independent of load. Falsified if cold load variance across launches
   (n=4 here) shows load_ms already dominated by irreducible weight I/O.

## Explicit non-claims

- No listening/quality claim of any kind; digests prove determinism, not quality.
- Cross-tree RTF comparison vs the Aug-28 receipts (which showed 1.07–1.14 early-run
  RTF) is PROVISIONAL: different engine tree AND different thermal history; it must not
  be cited as a regression without a cool-device pinned A/B.
- TTFA is measured on the streaming instrument route (`packet_frames=1`); the product UI
  still plays whole-buffer audio, where first-audio equals completion.

## Evidence gaps (stated, not hidden)

- No current-tree xctrace capture: the on-device trace tooling proved unstable against
  screen-lock suspensions during this campaign. Stage attribution above comes from the
  native synthesis profile JSON embedded in each receipt (exact per-stage timers, not
  sampling). The historical `ftts-ios-time-profile.xml`/`.trace` (Aug 28, older engine
  tree) corroborates microdecoder/codec dominance and is retained for comparison.
- Cold load sample size is n=4 launches (one per cell); 8.3 should widen it before
  certifying a cold-path claim.
- short×matt cell captured 11 of 12 planned samples (final-row receipt flush raced the
  first pull); all retained rows carry `run_complete` bookkeeping from their receipt.
