# franken_tts

**A pure-Rust, memory-safe, CPU-hyper-optimized runtime for one text-to-speech model: [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) — zero-shot voice cloning with no Python, no general ML framework, no GPU required.**

Sibling of [franken_ocr](https://github.com/Dicklesworthstone/franken_ocr): fixed model revision, custom quantized artifacts, model-specific kernels, and no pretense of being a general speech framework.

## The thesis

Qwen3-TTS's public story is "a 12.5 Hz model." Its actual CPU execution graph hides a **15-step autoregressive residual-code microdecoder inside every 80 ms frame**: the 28-layer talker runs once, then a 5-layer code predictor runs *fifteen sequential times* (per-depth embeddings, per-depth 2,048-way heads), then a causal codec decodes. First-order Q8 weight traffic is ≈1.65 GB per frame — and the microdecoder body accounts for ~1.18 GB of it, reread 15×.

`franken_tts` turns that hidden microdecoder from the model's largest CPU liability into its largest optimization advantage: cache-resident hot-packing, per-depth quantization, and **FrankenMTP** — speculative block drafting that the 5-layer predictor itself verifies exactly in a single causal pass.

## Status: planning complete, implementation not started

This repository currently contains the engineering plan and its full work graph:

- **[COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md](COMPREHENSIVE_PLAN_FOR_FRANKEN_TTS.md)** — the master plan (v2.1, two external review rounds integrated)
- **[AGENTS.md](AGENTS.md)** — the engineering doctrine for AI coding agents working here
- **`.beads/`** — the full dependency-wired work graph (15 epics + 106 granular tasks, self-documenting) tracked with [beads_rust](https://github.com/Dicklesworthstone/beads_rust); start with `br ready`

Planned shape: a `ftts` single binary + embeddable Rust library; `.fttsq` portable quantized weights + `.fttspack` per-machine kernel caches; `.ftvoice` voice packs with consent/provenance metadata; two conformance contracts (bit-exact vs. production-quality with blind-listening gates); execution profiles from interactive laptop latency to continuous-batching server throughput.

## Responsible use

Voice cloning is dual-use. This project records consent attestation and provenance in voice packs, ships no audio-acquisition features, preserves any upstream watermarking, and treats identity claims as settled by blind listening — not embedding cosines. Clone voices you have the right to clone.

## License

Runtime code: Apache-2.0 (planned). Model weights: Apache-2.0 by Qwen (attribution preserved in converted artifacts; verification of exact obligations is tracked in the work graph).
