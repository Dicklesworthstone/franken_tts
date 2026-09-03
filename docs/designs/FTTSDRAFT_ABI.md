# `.fttsdraft` Lab-to-Runtime ABI: Specification, Compatibility Contract & Plugin Safety

> **Artifact Type**: Systems Architecture Specification (Phase 5)  
> **Governing Bead**: `frankentts-p5-fttsdraft-abi-50o`  
> **Status**: Implemented & Certified (`crates/ftts-artifacts/src/fttsdraft.rs`)

---

## 1. Executive Summary & Purpose

Phase 5 introduces structural compression, knowledge distillation, and surgical architecture experiments to `franken_tts`:
- **Distilled Microdecoder Drafters**: 1- to 2-layer student models proposing residual candidate tokens.
- **Parallel Prediction Heads**: Multi-depth heads predicting residual codes concurrently.
- **Adaptive-Depth Controllers**: Early-exit classifiers dynamically terminating microdecoder execution for simple phonemes.

Without a rigorous ABI, laboratory distillation experiments remain confined to notebooks and rot in scratch folders.

The `.fttsdraft` format provides the **production landing path**:
1. Ships draft models as standalone, versioned `.fttsdraft` artifacts alongside canonical `.fttsq` files.
2. Treats drafters as **untrusted plugins**: strict cryptographic binding to the base model hash, hardened parsing, zero-copy safety, and dynamic kill-switch support.
3. Interfaces cleanly with **FrankenMTP** and the **AF-3 reliability monitor**.

---

## 2. Binary Container Layout (`FTTSDRFT`)

The `.fttsdraft` file structure follows the hardened container pattern:

```text
+-------------------+--------------------+--------------------+----------------------+-------------------------+
| Magic (8 bytes)   | Version (4 bytes)  | Length (8 bytes)   | Checksum (64 bytes)  | Payload (JSON bytes)    |
| b"FTTSDRFT"       | u32 LE (= 1)       | u64 LE             | ASCII Hex SHA-256    | UTF-8 JSON content      |
+-------------------+--------------------+--------------------+----------------------+-------------------------+
```

1. **Magic Bytes**: `FTTSDRFT` (ASCII, 8 bytes). Rejects wrong formats at offset 0.
2. **Version**: Little-endian 32-bit integer (`1`). Backward-incompatible schema updates bump this version.
3. **Payload Length**: Little-endian 64-bit integer specifying exact payload byte length. Prevents buffer over-reads.
4. **Integrity Checksum**: 64-character ASCII hex SHA-256 digest computed over the entire payload slice. Any single-bit corruption causes immediate rejection (`DraftError::ChecksumMismatch`).
5. **Payload**: Self-describing UTF-8 JSON encoding the `DraftHeader` and `BTreeMap<String, DraftTensor>`.

---

## 3. The Compatibility Contract

A `.fttsdraft` is strictly bound to its host environment through three validation checks:

### (a) Base Model Binding (`base_model_hash`)
A student drafter is trained against specific latent activations of a base model. Running a drafter against a different `.fttsq` model causes catastrophic divergence.
- Verification: `header.base_model_hash == hex_digest(base_fttsq)`
- Failure: Returns `DraftError::IncompatibleBaseModel`.

### (b) Engine ABI Versioning (`engine_abi_version`)
Tracks tensor layout conventions, kernel conventions, and FrankenMTP execution loops.
- Verification: `header.engine_abi_version <= CURRENT_ENGINE_ABI_VERSION`
- Failure: Returns `DraftError::IncompatibleAbiVersion`.

### (c) Dynamic Kill-Switch (`is_kill_switched`)
If an experimental drafter exhibits runtime instability or regression in the wild:
- Setting `is_kill_switched = true` in the artifact header immediately disables speculative drafting.
- The engine safely falls back to standard sequential execution without binary re-compilation or deployment.
- Failure: Returns `DraftError::KillSwitched(name)`.

---

## 4. Drafter Taxonomy (`DrafterType`)

The ABI classifies models into four operational modes:
- `distilled_mtp`: Lightweight recurrent microdecoder student network.
- `parallel_heads`: Multi-head single-forward residual code predictor.
- `adaptive_depth`: Classifier predicting optimal stopping depth per frame.
- `custom:<name>`: Extensible variant for novel research architectures.

---

## 5. Security & Robustness Verification

1. **`forbid(unsafe_code)`**: The parser contains zero `unsafe` blocks.
2. **Bounds & Overflow Proof**: Tensor shapes `[rows, cols]` are verified against payload data lengths (`data.len() == rows * cols`). Non-matching dimensions trigger `DraftError::CorruptHeader`.
3. **Fuzz Testing**: The `fuzz_decode(&[u8]) -> bool` harness validates that arbitrary corrupted or hostile bytes never trigger panics or out-of-bounds memory accesses.
