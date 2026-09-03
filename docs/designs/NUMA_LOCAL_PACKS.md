# NUMA-Local Packs & CCD Affinity: Architecture & Placement Policies

> **Artifact Type**: Systems Architecture Specification (Phase 3D)  
> **Governing Bead**: `frankentts-k-numa-guw`  
> **Status**: Certified & Implemented (`crates/ftts-core/src/numa.rs`)

---

## 1. Executive Summary & Hardware Physics

On modern multi-die server processors (such as AMD EPYC 9004 with up to 12 Core Complex Dies [CCDs] and AMD Ryzen Threadripper Pro with up to 8 CCDs), compute cores connect to memory channels via an interconnect fabric (AMD Infinity Fabric / I/O Die).

Under continuous batching ($N \ge 16$ concurrent streams):
- The model's ~1.65 GB Q8 weight working set must be streamed through L3 caches at 12.5 Hz (requiring $\ge 20.7$ GB/s of sustained read bandwidth).
- If worker threads running on CCD $A$ fetch weights from DRAM channels attached to CCD $B$, every read traverses the Infinity Fabric.
- **The Infinity Fabric Bottleneck**: Fabric cross-traffic introduces 1.5×–2.5× higher memory latency and saturates interconnect links long before DRAM channels are saturated.

To achieve linear server scaling, `franken_tts` treats NUMA domains and CCDs as first-class architectural boundaries.

---

## 2. Policy Specifications (`FTTS_NUMA`)

The runtime provides three deterministic placement policies configured via the `FTTS_NUMA` environment variable:

```text
FTTS_NUMA={local|replicate|interleave}
```

| Policy | Mechanism | DRAM Overhead | Latency & Fabric Impact | Recommended For |
| :--- | :--- | :--- | :--- | :--- |
| **`local` (Default)** | First-touch allocation. Memory pages map to the node of the initializing thread. | None (1× copy) | Moderate on single-socket; cross-node penalties under multi-CCD. | Workstations, single-node systems, low concurrency ($N \le 8$). |
| **`replicate`** | Explicitly replicates read-only `.fttspack` weight sections into each active NUMA node's local memory. | $M_{nodes} \times 1.65$ GB | **Zero cross-node fabric traffic**. 100% local memory channel hits. | High-concurrency servers ($N \ge 16$) on AMD EPYC (NPS2/NPS4) and dual-socket systems. |
| **`interleave`** | Round-robins physical pages evenly across all available NUMA nodes. | None (1× copy) | Blends channel bandwidth; introduces predictable 50% cross-node access. | Memory-constrained hosts where total DRAM cannot hold multiple replicas. |

---

## 3. Implementation Details (`ftts-core::numa`)

### `NumaTopology` Detection
- On Linux, queries `/sys/devices/system/node` to detect node boundaries and physical CPU core lists.
- On non-Linux or UMA platforms (Apple Silicon), defaults gracefully to a single unified node.
- Provides `node_for_cpu(cpu_id)` to map worker threads to their local NUMA domain.

### `NumaPackPool<T>`
- Thread-safe, lock-free pool storing node-local pack instances (`BTreeMap<NodeId, T>`).
- Under `NumaPolicy::Replicate`, instantiates a replica for every detected node ID.
- Workers request `pool.get_pack(worker_node)`:
  - Local hits require zero lock acquisition and read from node-local DRAM.
  - Telemetry monitors `local_hit_ratio` in real time.

### `NumaActivationBuffer`
- Scratch buffers (talker intermediate activations, microdecoder KV buffers) are allocated strictly on the worker's local node, preventing cross-socket cache line invalidations and false sharing.

---

## 4. Verification & Metamorphic Guarantees

1. **Deterministic Equivalence**: Output audio tokens are 100% bit-for-bit identical regardless of which NUMA policy (`local`, `replicate`, `interleave`) is active.
2. **Local Hit Invariant**: Under `FTTS_NUMA=replicate`, `NumaPoolMetrics::local_hit_ratio()` evaluates to exactly 1.0 (0 cross-node fetches).
3. **Graceful Fallback**: If `FTTS_NUMA` is unset or invalid, the engine silently and safely defaults to `local`.
