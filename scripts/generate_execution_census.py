#!/usr/bin/env python3
"""Derive the integrated execution census from pinned inputs only.

Every number this emits is COMPUTED from one of three pinned sources:

  * ``docs/truth-pack/TENSOR_INVENTORY.json``      (OQ-2, itself generated from safetensors headers)
  * ``docs/truth-pack/snapshots/hf/config.json``   (talker / microdecoder geometry)
  * ``docs/truth-pack/snapshots/hf/speech_tokenizer/config.json``  (codec geometry)

Nothing numeric is transcribed from a prose document. Where structure cannot be read out of a
config -- the residual-unit dilation schedule, which stage of a decoder block is the transposed
conv -- the *structure* is declared here as a cited constant and the *numbers* are still taken from
tensor shapes. That distinction is the whole point: a shape change in the checkpoint must move these
rows, and a re-pin that changes them must fail the drift guard rather than silently disagree with a
markdown table someone updated by hand.

Usage:
    generate_execution_census.py            regenerate docs/truth-pack/EXECUTION_CENSUS.json
    generate_execution_census.py --check    regenerate in memory and fail on any drift
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/truth-pack/TENSOR_INVENTORY.json"
TALKER_CONFIG = ROOT / "docs/truth-pack/snapshots/hf/config.json"
CODEC_CONFIG = ROOT / "docs/truth-pack/snapshots/hf/speech_tokenizer/config.json"
OUTPUT = ROOT / "docs/truth-pack/EXECUTION_CENSUS.json"

# i32 accumulation bound. U8S8 is the worst case our int8 GEMM tiers can present.
I32_MAX = 2**31
U8_MAX = 255
S8_ABS_MAX = 127

# Structural facts that no config records, each cited to the pinned modeling source. Numbers below
# are still read from tensor shapes; only the *arrangement* is asserted here.
# modeling_qwen3_tts_tokenizer_v2.py:650 -- `for dilation in (1, 3, 9)`
RESIDUAL_DILATIONS = (1, 3, 9)
# modeling_qwen3_tts_tokenizer_v2.py:647 -- block.1 is the CausalTransConvNet; block.{2,3,4} are
# residual units whose conv1 carries the dilation and conv2 is a pointwise k=1.
TRANSPOSED_CONV_SUFFIX = ".block.1.conv.weight"

# Components that execute inside the per-frame decode loop. The codec *encoder* and the speaker
# encoder are enrollment-only builds and never run while synthesizing.
DECODE_PATH_COMPONENTS = frozenset({"talker", "microdecoder", "codec_decoder", "text_path"})

# Component partition over the tensor namespace. Prefix -> (component, role). Longest prefix wins.
COMPONENTS = (
    ("talker.code_predictor.lm_head.", "microdecoder", "per_depth_head"),
    ("talker.code_predictor.model.codec_embedding.", "microdecoder", "per_depth_embedding"),
    ("talker.code_predictor.model.", "microdecoder", "body"),
    ("talker.model.text_embedding", "text_path", "cold_embedding"),
    ("talker.model.codec_embedding", "talker", "codec_embedding"),
    ("talker.text_projection.", "text_path", "projection"),
    ("talker.codec_head.", "talker", "primary_head"),
    ("talker.model.layers.", "talker", "body"),
    ("talker.model.norm", "talker", "body"),
    ("decoder.", "codec_decoder", "decoder"),
    ("encoder.", "codec_encoder", "encoder"),
    ("speaker_encoder.", "speaker_encoder", "encoder"),
)


def classify(name: str) -> tuple[str, str]:
    """Assign a tensor to (component, role) by longest matching prefix."""
    best: tuple[str, str] | None = None
    best_len = -1
    for prefix, component, role in COMPONENTS:
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = (component, role), len(prefix)
    if best is None:
        raise SystemExit(f"unclassified tensor (census partition is not exhaustive): {name}")
    return best


def reduction_k(name: str, shape: list[int]) -> int | None:
    """The reduction length of one weight, i.e. the K of its dot product.

    Conv1d stores ``[out, in, k]`` so K = in*k. ConvTranspose1d stores ``[in, out, k]`` and each
    output position accumulates over ``ceil(k/stride)`` taps; the stride is the upsample rate, which
    for every transposed conv in this decoder satisfies k == 2*stride, giving exactly 2 taps.
    Linear stores ``[out, in]`` so K = in. Biases and 1-D parameters have no reduction.
    """
    if len(shape) == 3:
        if name.endswith(TRANSPOSED_CONV_SUFFIX) or ".upsample." in name:
            in_channels, _out, kernel = shape
            taps = 2 if kernel % 2 == 0 else 1
            return in_channels * taps
        out_channels, in_channels, kernel = shape
        del out_channels
        return in_channels * kernel
    if len(shape) == 2:
        return shape[1]
    return None


def overflow_row(name: str, k: int) -> dict[str, object]:
    worst = U8_MAX * S8_ABS_MAX * k
    return {
        "tensor": name,
        "k": k,
        "u8s8_worst_abs_accumulator": worst,
        "i32_headroom": round(I32_MAX / worst, 2),
    }


def codec_receptive_field(tensors: dict[str, list[int]], codec: dict) -> dict[str, object]:
    """Left context the codec decoder needs, in code frames, derived from shapes + config.

    Walks the decode chain in execution order, tracking the sample rate relative to the 12.5 Hz code
    frame. A stride-1 causal conv of effective kernel ``ke`` needs ``ke-1`` samples of history at its
    own rate; a transposed conv with kernel k and stride s reaches back ``ceil(k/s)-1`` input samples
    and multiplies the rate by s.
    """
    decoder = codec["decoder_config"]
    upsample_rates = decoder["upsample_rates"]
    upsampling_ratios = decoder["upsampling_ratios"]
    layers = decoder["num_hidden_layers"]
    window = decoder["sliding_window"]

    rate = Fraction(1)
    context = Fraction(0)
    stages: list[dict[str, object]] = []

    def conv(label: str, kernel: int, dilation: int = 1) -> None:
        nonlocal context
        effective = (kernel - 1) * dilation + 1
        need = Fraction(effective - 1) / rate
        context += need
        stages.append(
            {
                "stage": label,
                "kernel": kernel,
                "dilation": dilation,
                "rate_vs_frame": str(rate),
                "left_context_frames": float(need),
                "cumulative_frames": float(context),
            }
        )

    def transposed(label: str, kernel: int, stride: int) -> None:
        nonlocal context, rate
        taps_back = -(-kernel // stride) - 1
        need = Fraction(taps_back) / rate
        context += need
        rate *= stride
        stages.append(
            {
                "stage": label,
                "kernel": kernel,
                "stride": stride,
                "rate_vs_frame": str(rate),
                "left_context_frames": float(need),
                "cumulative_frames": float(context),
            }
        )

    conv("pre_conv", tensors["decoder.pre_conv.conv.weight"][2])

    # Stacked sliding-window attention: each layer reaches back window-1, and the stack composes.
    transformer_context = layers * (window - 1)
    context += transformer_context
    stages.append(
        {
            "stage": f"pre_transformer ({layers} x sliding window {window})",
            "rate_vs_frame": str(rate),
            "left_context_frames": float(transformer_context),
            "cumulative_frames": float(context),
        }
    )

    for index, ratio in enumerate(upsampling_ratios):
        transposed(
            f"upsample.{index}.0 tconv",
            tensors[f"decoder.upsample.{index}.0.conv.weight"][2],
            ratio,
        )
        conv(
            f"upsample.{index}.1 dwconv",
            tensors[f"decoder.upsample.{index}.1.dwconv.conv.weight"][2],
        )

    conv("decoder.0 conv", tensors["decoder.decoder.0.conv.weight"][2])

    for index, upsample_rate in enumerate(upsample_rates, start=1):
        transposed(
            f"decoder.{index} tconv",
            tensors[f"decoder.decoder.{index}.block.1.conv.weight"][2],
            upsample_rate,
        )
        for unit, dilation in enumerate(RESIDUAL_DILATIONS, start=2):
            conv(
                f"decoder.{index}.block.{unit}.conv1",
                tensors[f"decoder.decoder.{index}.block.{unit}.conv1.conv.weight"][2],
                dilation,
            )

    conv("decoder.6 conv", tensors["decoder.decoder.6.conv.weight"][2])

    total_upsample = 1
    for value in list(upsample_rates) + list(upsampling_ratios):
        total_upsample *= value

    conv_only = context - transformer_context
    # Live streaming state: the sliding window bounds retained KV at `window` positions per layer.
    head_dim = decoder["head_dim"]
    kv_values = layers * window * decoder["num_key_value_heads"] * head_dim * 2
    return {
        "total_upsample": total_upsample,
        "declared_decode_upsample_rate": codec["decode_upsample_rate"],
        "hop_math_agrees": total_upsample == codec["decode_upsample_rate"],
        "stages": stages,
        "total_left_context_frames": float(context),
        "transformer_left_context_frames": float(transformer_context),
        "conv_only_left_context_frames": float(conv_only),
        "ring_buffer_note": (
            "conv_only_left_context_frames is what the causal-conv ring buffers must hold; the "
            "transformer term is an information horizon, not storage -- the sliding window bounds "
            "live KV at `window` positions per layer"
        ),
        "live_transformer_kv_values": kv_values,
        "live_transformer_kv_bytes_bf16": kv_values * 2,
        "live_transformer_kv_bytes_f32": kv_values * 4,
    }


def build(inventory: dict, talker_cfg: dict, codec_cfg: dict) -> dict:
    tensors = inventory["tensors"]
    shapes = {tensor["name"]: tensor["shape"] for tensor in tensors}

    components: dict[str, dict] = {}
    for tensor in tensors:
        component, role = classify(tensor["name"])
        bucket = components.setdefault(
            component, {"tensor_count": 0, "bytes": 0, "by_role": {}, "by_dtype": {}}
        )
        bucket["tensor_count"] += 1
        bucket["bytes"] += tensor["bytes"]
        role_bucket = bucket["by_role"].setdefault(role, {"tensor_count": 0, "bytes": 0})
        role_bucket["tensor_count"] += 1
        role_bucket["bytes"] += tensor["bytes"]
        bucket["by_dtype"][tensor["dtype"]] = (
            bucket["by_dtype"].get(tensor["dtype"], 0) + tensor["bytes"]
        )

    talker = talker_cfg["talker_config"]
    predictor = talker["code_predictor_config"]
    groups = talker["num_code_groups"]
    residual_steps = groups - 1

    micro = components["microdecoder"]
    body_bytes = micro["by_role"]["body"]["bytes"]
    head_bytes = micro["by_role"]["per_depth_head"]["bytes"]
    embed_bytes = micro["by_role"]["per_depth_embedding"]["bytes"]

    # Microdecoder KV: 5 layers x <=16 positions, reset every frame.
    micro_kv_values = (
        predictor["num_hidden_layers"]
        * groups
        * predictor["num_key_value_heads"]
        * predictor["head_dim"]
        * 2
    )
    # Precomputed plain-RoPE table for positions 0..15: cos+sin over head_dim.
    rope_table_values = groups * predictor["head_dim"] * 2

    # Footprint vs traffic is the distinction the hot pack exists to exploit: the body is resident
    # once but read `residual_steps` times per frame.
    hot_footprint = (
        body_bytes + head_bytes + embed_bytes + micro_kv_values * 4 + rope_table_values * 4
    )
    hot_traffic = body_bytes * residual_steps + head_bytes + embed_bytes

    talker_kv_per_token = (
        talker["num_hidden_layers"] * talker["num_key_value_heads"] * talker["head_dim"] * 2
    )
    frame_rate = Fraction(codec_cfg["input_sample_rate"], codec_cfg["decode_upsample_rate"])
    kv_by_duration = []
    for seconds in (10, 60, 163, 300, 655):
        frames = int(frame_rate * seconds)
        kv_by_duration.append(
            {
                "seconds": seconds,
                "frames": frames,
                "kv_values": frames * talker_kv_per_token,
                "kv_bytes_bf16": frames * talker_kv_per_token * 2,
                "kv_bytes_f32": frames * talker_kv_per_token * 4,
            }
        )

    overflow_rows = []
    for tensor in tensors:
        k = reduction_k(tensor["name"], tensor["shape"])
        if k is not None:
            overflow_rows.append((classify(tensor["name"])[0], overflow_row(tensor["name"], k)))
    overflow_rows.sort(key=lambda row: (-row[1]["k"], row[1]["tensor"]))
    binding = overflow_rows[0][1]
    worst_by_component: dict[str, dict] = {}
    for component, row in overflow_rows:
        if component not in worst_by_component:
            worst_by_component[component] = row
    # The enrollment-only encoders never run in the decode loop, so the bound a decode kernel must
    # prove is not the same as the bound the whole checkpoint contains. Report both explicitly
    # rather than letting one shadow the other.
    decode_binding = next(
        row for component, row in overflow_rows if component in DECODE_PATH_COMPONENTS
    )

    return {
        "schema_version": 1,
        "generator": "scripts/generate_execution_census.py",
        "source_pin": inventory["source_pin"],
        "derivation": (
            "every numeric row is computed from TENSOR_INVENTORY.json plus the pinned configs; "
            "no value is transcribed from a prose document"
        ),
        "frame": {
            "code_groups": groups,
            "residual_steps_per_frame": residual_steps,
            "frames_per_second": float(frame_rate),
            "samples_per_frame": codec_cfg["decode_upsample_rate"],
            "sample_rate": codec_cfg["input_sample_rate"],
        },
        "components": {name: components[name] for name in sorted(components)},
        "totals": {
            "tensor_count": len(tensors),
            "payload_bytes": sum(tensor["bytes"] for tensor in tensors),
        },
        "microdecoder_hot_working_set": {
            "stored_dtype": "BF16",
            "body_bytes": body_bytes,
            "per_depth_head_bytes": head_bytes,
            "per_depth_embedding_bytes": embed_bytes,
            "kv_values": micro_kv_values,
            "rope_table_values": rope_table_values,
            "footprint_bytes": hot_footprint,
            "traffic_bytes_per_frame": hot_traffic,
            "traffic_footprint_ratio": round(hot_traffic / hot_footprint, 3),
            "q8_projection": {
                "note": (
                    "the checkpoint stores BF16; the hot pack targets Q8, so weight rows halve "
                    "while the high-precision KV and RoPE table do not. This is the number the "
                    "plan's residency target is stated against"
                ),
                "body_bytes": body_bytes // 2,
                "per_depth_head_bytes": head_bytes // 2,
                "per_depth_embedding_bytes": embed_bytes // 2,
                "footprint_bytes": body_bytes // 2
                + head_bytes // 2
                + embed_bytes // 2
                + micro_kv_values * 4
                + rope_table_values * 4,
                "traffic_bytes_per_frame": (body_bytes // 2) * residual_steps
                + head_bytes // 2
                + embed_bytes // 2,
            },
            "note": (
                "footprint is what must stay resident; traffic is what a naive sequential pass "
                "moves per frame because the body is reread once per residual step. These are "
                "different numbers and conflating them is the error the hot pack exists to avoid"
            ),
        },
        "talker_kv": {
            "values_per_token": talker_kv_per_token,
            "bytes_per_token_bf16": talker_kv_per_token * 2,
            "bytes_per_token_f32": talker_kv_per_token * 4,
            "by_duration": kv_by_duration,
            "max_position_embeddings": talker["max_position_embeddings"],
        },
        "activation_rails": {
            "talker_hidden": talker["hidden_size"],
            "talker_intermediate": talker["intermediate_size"],
            "talker_attention_width": talker["num_attention_heads"] * talker["head_dim"],
            "microdecoder_hidden": predictor["hidden_size"],
            "microdecoder_intermediate": predictor["intermediate_size"],
            "codec_decoder_hidden": codec_cfg["decoder_config"]["hidden_size"],
            "codec_decoder_intermediate": codec_cfg["decoder_config"]["intermediate_size"],
            "codec_latent_dim": codec_cfg["decoder_config"]["latent_dim"],
            "codec_decoder_dim": codec_cfg["decoder_config"]["decoder_dim"],
            "text_embedding_width": shapes["talker.model.text_embedding.weight"][1],
            "text_projection_out": shapes["talker.text_projection.linear_fc2.weight"][0],
        },
        "codec_receptive_field": codec_receptive_field(shapes, codec_cfg),
        "overflow_k_table": {
            "bound": "i32",
            "worst_case_operands": "u8 x s8",
            "global_binding_row": binding,
            "decode_path_binding_row": decode_binding,
            "binding_note": (
                "global_binding_row spans the whole checkpoint including the enrollment-only "
                "encoders; decode_path_binding_row is the bound a per-frame kernel must actually "
                "prove. Both ship selftest rows -- the encoder one only in the enrollment build"
            ),
            "worst_by_component": {
                name: worst_by_component[name] for name in sorted(worst_by_component)
            },
            "rows": [row for _component, row in overflow_rows[:24]],
            "row_count": len(overflow_rows),
        },
    }


def render(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and fail if the committed census has drifted",
    )
    arguments = parser.parse_args()

    for path in (INVENTORY, TALKER_CONFIG, CODEC_CONFIG):
        if not path.is_file():
            print(f"missing pinned input: {path.relative_to(ROOT)}", file=sys.stderr)
            return 1

    document = build(
        json.loads(INVENTORY.read_text()),
        json.loads(TALKER_CONFIG.read_text()),
        json.loads(CODEC_CONFIG.read_text()),
    )
    rendered = render(document)

    if not document["codec_receptive_field"]["hop_math_agrees"]:
        print(
            "derived total upsample disagrees with the codec's declared decode_upsample_rate",
            file=sys.stderr,
        )
        return 1

    if arguments.check:
        if not OUTPUT.is_file():
            print(
                f"{OUTPUT.relative_to(ROOT)} is missing; run scripts/generate_execution_census.py",
                file=sys.stderr,
            )
            return 1
        committed = OUTPUT.read_text()
        if committed != rendered:
            print(
                f"{OUTPUT.relative_to(ROOT)} is stale: a pinned input changed without "
                f"regenerating it.\nRun: python3 scripts/generate_execution_census.py",
                file=sys.stderr,
            )
            return 1
        return 0

    OUTPUT.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
