# FrankenTTS iPhone, iPad, and Mac Excellence Plan

Status: active

Source fence when opened: `64bf52df7acd11737ce95d068fdcd32630ec9487`

This plan extends `IOS_APP_PLAN.md`; it does not replace the original engine,
memory, privacy, and platform arithmetic. The cross-product requirement ledger
lives in the active Codex workspace's
`FRANKENSUITE_APP_EXCELLENCE_MASTER_TODO.md`. This file keeps the repository's
own execution order and acceptance boundary durable.

## Product outcome

FrankenTTS becomes a premium, private Voice Forge on iPhone and iPad plus an
optimized Mac Catalyst app sold through the same App Store product. The main
screen centers utterance, voice, Synthesize, and result. Model/storage and
diagnostics move into status/settings surfaces. Synthesis gets a spectacular
but truthful Galvanic Voice Forge driven by native engine events.

## Execution checklist

Current source tranche (2026-08-28): the versioned native progress ABI, Swift
bridge/cancellation, event-driven Galvanic Voice Forge, reactive monster,
focused/adaptive shell, Catalyst project and Rust-slice script, focused Mac
commands/import/drop, App Group schema, widget, Live Activity/Dynamic Island,
App Intents, deep links, and selected-text share extension are implemented in
source. Rust ABI tests passed before the system-extension tranche. YAML, plist,
privacy-manifest, and diff hygiene checks are green. Xcode generation, universal
framework completion, extension compilation, Catalyst launch, and installed
iPhone acceptance remain open and must not be inferred from this static state.

- [ ] Define versioned synthesis progress events and callback ownership in `ftts-ffi`.
- [ ] Emit real load, prefill, generation-frame, codec, terminal, and cancellation events.
- [ ] Test event order, callback lifetime, null callback, terminal uniqueness, and panic containment.
- [ ] Bridge native events into the serialized Swift engine actor.
- [ ] Throttle UI publication separately from native event delivery.
- [ ] Replace time-derived fake percentage with real stage/unit state.
- [ ] Build shared semantic theme, panels, controls, telemetry, machine disclosure, and adaptive workspace primitives.
- [ ] Restructure the phone UI around utterance, selected voice, primary action, and result.
- [ ] Preserve native text selection, Select All, Clear, character count, keyboard dismissal, and above-keyboard action reachability.
- [ ] Build the Galvanic Voice Forge hero view with real frame and codec events.
- [ ] Make the cute monster react causally to ready, cold, loading, running, success, cancellation, and error states.
- [ ] Preserve enrollment consent, level meter, normalization, silence refusal, success selection, and haptics.
- [ ] Add private local result history with retention, deletion, privacy redaction, and no-content diagnostics.
- [ ] Preserve automatic model warming and memory-pressure unloading; remove every unnecessary manual wake action.
- [ ] Add responsive iPad sidebar/workspace/inspector layout.
- [ ] Add Mac Catalyst support and Catalyst-compatible Rust library slice.
- [ ] Optimize for the Mac idiom with desktop typography, toolbar, menus, shortcuts, drag/drop, exports, resizable windows, and multiple sessions.
- [ ] Add App Group staging and versioned deep-link routes.
- [ ] Add widgets, Live Activity/Dynamic Island, Speak Text App Intent, control, selected-text share extension, quick actions, and Handoff.
- [ ] Keep the large model out of every extension process.
- [ ] Add accessibility, Reduce Motion, Reduce Transparency, Low Power, thermal throttling, and bounded animation cadence.
- [ ] Add dark/tinted icon variants without losing the cute shared character.
- [ ] Regenerate Xcode, run focused Rust/Swift tests, and build device, simulator, iPad, and Mac Catalyst targets.
- [ ] Install on the connected iPhone and exercise cold/warm synthesis, cancellation, enrollment, playback, share, widget, activity, intent, and share-extension routes.
- [ ] Reconcile App Store privacy/review notes and explicitly state that no user content is sent to any third-party AI service.

## Acceptance boundary

The work is complete only when the native processing view is event-driven, the
primary workflow is usable with VoiceOver and Reduce Motion, the Mac target
launches and behaves like a desktop app, and an installed iPhone build passes
the representative user scenarios. A successful compile alone is not runtime
or App Store proof.
