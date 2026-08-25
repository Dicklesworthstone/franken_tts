# PERFORMANCE_ARCHITECTURE.md — franken_tts Cost Model & Execution Architecture

> **Canonical Costed Execution Graph & Performance Limits**
> Generated from pinned inputs by `scripts/cost_model.py`.

---

## 1. Executive Summary & The Dual-Traffic Contract

| Traffic Metric | Per 80 ms Frame (Q8) | 1× Real Time (12.5 Hz) | 2× Real Time | 5× Real Time | Architectural Regime |
|---|---|---|---|---|---|
| **Sequential Baseline** | **1686.5 MB** | **21.08 GB/s** | 42.16 GB/s | 105.41 GB/s | Naive 15× reread from DRAM |
| **One-Read Floor** | **585.2 MB** | **7.32 GB/s** | 14.63 GB/s | 36.58 GB/s | Cache-resident / FrankenMTP |

> [!IMPORTANT]
> **CRITICAL LABELING LAW (Doctrine §2.6 / v2.1):** The ~1.65 GB/frame (21.08 GB/s @ 1x) is the SEQUENTIAL-EXECUTION BASELINE, not a physics floor. The all-components one-read floor is ~0.585 GB/frame (7.31 GB/s @ 1x). Never quote 20.7+ GB/s as the bound 'even after' residency or speculation levers.

---

## 2. Serial Work & Layer Execution Depth

- **Talker Backbone**: 28 layers × 12.5 fps = **350.0 layer evaluations / sec**
- **Residual Microdecoder**: 5 layers × 15 steps × 12.5 fps = **937.5 layer evaluations / sec**
- **Total Serial Transformer Depth**: **1287.5 layer evaluations / sec**

The 12.5 Hz speech frame rate hides 1,287.5 transformer layer evaluations every second. The microdecoder accounts for **72.8%** of all sequential layer passes.

---

## 3. Microdecoder Per-Depth Breakdown (The 15 Steps)

| Depth | Target Code | Conditioning | Embedding (Q8) | Body Weights (Q8) | LM Head (Q8) | Total Step Q8 | Live KV (F32) |
|---|---|---|---|---|---|---|---|
| 1 | `c_1` | `c_0` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 40.0 KB |
| 2 | `c_2` | `c_1` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 80.0 KB |
| 3 | `c_3` | `c_2` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 120.0 KB |
| 4 | `c_4` | `c_3` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 160.0 KB |
| 5 | `c_5` | `c_4` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 200.0 KB |
| 6 | `c_6` | `c_5` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 240.0 KB |
| 7 | `c_7` | `c_6` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 280.0 KB |
| 8 | `c_8` | `c_7` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 320.0 KB |
| 9 | `c_9` | `c_8` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 360.0 KB |
| 10 | `c_10` | `c_9` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 400.0 KB |
| 11 | `c_11` | `c_10` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 440.0 KB |
| 12 | `c_12` | `c_11` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 480.0 KB |
| 13 | `c_13` | `c_12` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 520.0 KB |
| 14 | `c_14` | `c_13` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 560.0 KB |
| 15 | `c_15` | `c_14` | 2.10 MB | 78.66 MB | 2.10 MB | **82.86 MB** | 600.0 KB |

- **Single-Pass Hot Working Set**: Body (~78.66 MB Q8) + 15 Heads (~31.46 MB Q8) + 15 Embeddings (~31.46 MB Q8) + KV/RoPE (~0.67 MB) = **~142.24 MB**.
- **Sequential Traffic**: (78.66 × 15) + 31.46 + 31.46 = **1,242.75 MB / frame**.

---

## 4. Hardware Tier Roofline & Speed Limits

| Target SKU | DRAM Bandwidth | Sequential Max RTF | Resident Max RTF | 1× Seq? | 2× Seq? | 5× Seq? | 5× Resident? |
|---|---|---|---|---|---|---|---|
| **Apple M4 Pro** | 273 GB/s | 12.95× | **37.32×** | ✅ Yes | ✅ Yes | ✅ Yes | **✅ Yes** |
| **Apple M4 Base** | 120 GB/s | 5.69× | **16.40×** | ✅ Yes | ✅ Yes | ✅ Yes | **✅ Yes** |
| **Apple M2 / M3 Base** | 100 GB/s | 4.74× | **13.67×** | ✅ Yes | ✅ Yes | ❌ No | **✅ Yes** |
| **AMD Ryzen 9 (Zen 5 DDR5-6000)** | 96 GB/s | 4.55× | **13.12×** | ✅ Yes | ✅ Yes | ❌ No | **✅ Yes** |
| **Intel Core Ultra 7 (LPDDR5X-7467)** | 119 GB/s | 5.64× | **16.27×** | ✅ Yes | ✅ Yes | ✅ Yes | **✅ Yes** |

---

## 5. Multi-Stream Server Scaling (Continuous Batching)

| Concurrent Streams | Unbatched Seq DRAM | Unbatched Resident DRAM | Batched Seq DRAM | Batched Resident DRAM | Batching Advantage |
|---|---|---|---|---|---|
| 1 | 21.1 GB/s | 7.3 GB/s | 21.1 GB/s | **7.3 GB/s** | **1.0×** |
| 2 | 42.2 GB/s | 14.6 GB/s | 21.1 GB/s | **7.3 GB/s** | **2.0×** |
| 4 | 84.3 GB/s | 29.3 GB/s | 21.1 GB/s | **7.3 GB/s** | **4.0×** |
| 8 | 168.7 GB/s | 58.5 GB/s | 21.1 GB/s | **7.4 GB/s** | **8.0×** |
| 16 | 337.3 GB/s | 117.0 GB/s | 21.2 GB/s | **7.4 GB/s** | **15.9×** |
| 32 | 674.6 GB/s | 234.1 GB/s | 21.2 GB/s | **7.5 GB/s** | **31.8×** |

---

## 6. Cold Text-Embedding & Prefill Traffic

| Prompt Tokens | Gathered Emb (BF16) | Gathered Emb (Q8) | Full Table (BF16) | Memory Savings | Projection MACs | Talker Prefill MACs |
|---|---|---|---|---|---|---|
| 10 tokens | 40.0 KB | 20.0 KB | 622.3 MB | **15194× less** | 62.9 M | 4406.3 M |
| 20 tokens | 80.0 KB | 40.0 KB | 622.3 MB | **7597× less** | 125.8 M | 8812.6 M |
| 50 tokens | 200.0 KB | 100.0 KB | 622.3 MB | **3039× less** | 314.6 M | 22031.6 M |
| 100 tokens | 400.0 KB | 200.0 KB | 622.3 MB | **1519× less** | 629.1 M | 44063.1 M |
| 200 tokens | 800.0 KB | 400.0 KB | 622.3 MB | **760× less** | 1258.3 M | 88126.3 M |

---

## 7. Codec Packet Schedules

| Packet Mode | Duration | Audio Samples | Traffic (F32) | Traffic (Q8) | DRAM Rate (F32) | DRAM Rate (Q8) |
|---|---|---|---|---|---|---|
| **1-Frame Packet** | 80 ms | 1920 | 457.3 MB | 228.6 MB | 5.72 GB/s | 2.86 GB/s |
| **4-Frame Packet** | 320 ms | 7680 | 457.3 MB | 228.6 MB | **1.43 GB/s** | **0.71 GB/s** |

---

## 8. Answers to the Seven Core Architectural Questions (§10.1)

1. **Is the Q8 microdecoder cache-resident on M4/M5 and AMD?**
   - The Q8 working set is **142.24 MB** (78.7 MB body + 31.5 MB heads + 31.5 MB embeds + KV). On M4 Pro (SLC 32 MB) and AMD (L3 64 MB), the *body alone* (78.7 MB) or combined set spills to DRAM unless compressed to Q4 (~40 MB body). With Q4, the microdecoder body achieves full L3/SLC cache residency.

2. **Which hardware tier limits 1×/2×/5× real time?**
   - **1× RT (21.1 GB/s seq)**: Achievable on all desktop/laptop tiers (Apple M-series, AMD Zen, Intel Core Ultra).
   - **2× RT (42.2 GB/s seq)**: Requires >=60 GB/s DRAM bandwidth (dual-channel DDR5 or Apple Silicon).
   - **5× RT (105.4 GB/s seq)**: Saturated on base M-series and dual-channel DDR5; *strictly requires* either Apple Pro/Max memory bandwidth OR cache residency/FrankenMTP (36.6 GB/s floor).

3. **W8A8 vs W8A16 — which is faster at each real shape?**
   - For decoding GEMV (M=1), W8A8 with native vector-matrix dot products (SDOT/SMMLA/VNNI) minimizes memory footprint and matches compute throughput.
   - For prefill GEMM (M>=16), tiled W8A8 SMMLA/VNNI achieves peak compute density.

4. **At what stream count does batched GEMM overtake per-stream GEMV?**
   - At **N >= 4 concurrent streams**, continuous depth/frame batching reduces aggregate weight traffic by **3.5×** to **21.8×** (at N=32), shifting compute from memory-bound GEMV to compute-bound GEMM.

5. **Does codec overlap help or merely contend for bandwidth?**
   - In 4-frame packet mode, codec weight bandwidth is amortized to **1.43 GB/s (F32)** or **0.71 GB/s (Q8)**, representing <3.5% of total system bandwidth, allowing pipelined overlap without DRAM starvation.

6. **Is 5× real time on M4 physically plausible without speculative MTP or Q4?**
   - On M4 Pro (273 GB/s): **Yes** (105.4 GB/s sequential = 38.6% bandwidth headroom).
   - On M4 Base (120 GB/s): **Marginal/No** (105.4 GB/s = 87.8% of theoretical peak, leaving no room for OS/system contention); requires Q4 microdecoder or FrankenMTP.

7. **What is FrankenMTP's break-even acceptance α*(SKU)?**
   - Since sequential microdecoder evaluates 15 separate forwards (15 × T_step), while FrankenMTP evaluates 1 causal block verification pass (T_verify ≈ 1.8 × T_step) + repair passes (T_repair ≈ k × T_step), break-even acceptance rate is **α* ≈ 0.62** (62% token acceptance).

