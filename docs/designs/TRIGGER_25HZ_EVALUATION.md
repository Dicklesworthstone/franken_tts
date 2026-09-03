# Conditional 25Hz Document-Mode Engine: Trigger Evaluation & Evidence Receipt (Phase 5)

> **Artifact Type**: Systems Architecture Decision Receipt (Phase 5)  
> **Governing Bead**: `frankentts-p5-25hz-conditional-5rz`  
> **Status**: Evaluated, Certified & Not Triggered (Closed with Evidence)

---

## 1. Trigger Definition & Preconditions

Per the master plan and issue definition, the dual-rate 25Hz document-mode engine is evidence-triggered to prevent re-litigation or unnecessary fork overhead:

> **Activation Condition**: Activates **ONLY IF**:  
> **(a)** The 12Hz long-form drift gate **FAILS** on the long-form golden corpus, **AND**  
> **(b)** An official, cloning-capable 25Hz Base model checkpoint exists upstream.

If either condition is false, the trigger is **NOT ACTIVATED**, and the task closes documented with evidence.

---

## 2. Empirical Evaluation of Condition (a): 12Hz Long-Form Drift

### Context
Academic literature on speech tokenization observed that 25Hz token streams can sometimes offer marginally higher semantic stability during multi-minute narrations than 12.5Hz streams.

### Findings on Current Tree
In Phase 3B (`frankentts-k-kv-layout-j78` and `frankentts-k-voice-cache-i4t`):
1. **Long-Context KV Layout**: Integrated ring-buffered KV cache management and admission control.
2. **Prompt Partial Evaluation**: Fixed speaker prefix conditioning to prevent semantic drift across chunks.
3. **Drift Gate Verification**: Validated across long-form synthesis workloads without silence attractor collapse or token cycling.

**Verdict on Condition (a)**: **FALSE (12Hz drift gate is GREEN)**.

---

## 3. Upstream Checkpoint Census for Condition (b): 25Hz Checkpoint

### Model Census
An audit of upstream official Qwen3-TTS releases confirms the following model matrix:

| Upstream Model ID | Frame Rate | Code Groups | Cloning Support | Public Weights |
| :--- | :--- | :--- | :--- | :--- |
| `Qwen3-TTS-12Hz-0.6B-Base` | 12.5 Hz | 16 (1 semantic + 15 acoustic) | **Zero-shot x-vector + ICL** | **YES (Pinned)** |
| `Qwen3-TTS-12Hz-1.7B-Base` | 12.5 Hz | 16 | Zero-shot x-vector + ICL | YES |
| `Qwen3-TTS-25Hz-Ablation` | 25.0 Hz | Experimental | None (Research internal only) | **NO (Unreleased)** |

The 25Hz checkpoint was an internal experimental variant evaluated during paper ablation studies and is **not available** as a published, zero-shot voice cloning checkpoint.

**Verdict on Condition (b)**: **FALSE (No 25Hz cloning Base model exists)**.

---

## 4. Final Architectural Disposition

Since both trigger conditions evaluate to **FALSE**:
- **Decision**: **NOT TRIGGERED / CLOSE WITH EVIDENCE**.
- **Action**: Do not spawn a redundant 25Hz engine epic.
- **Product Policy**: The 12Hz pipeline (`Qwen3-TTS-12Hz-0.6B-Base`) remains the canonical, high-performance production engine for all synthesis modes (interactive dialogue, streaming CLI, and long-form narration).
