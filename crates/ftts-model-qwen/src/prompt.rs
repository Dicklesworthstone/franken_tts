//! Exact Base voice-clone prompt assembly.
//!
//! This module owns the four Qwen3-TTS Base prompt geometries.  It deliberately receives vectors
//! that have already passed through their respective embedding/projection paths: the model loader
//! owns weights, while this seam owns the token-position and addition contract.  The latter is L0
//! and must stay independent of weight hydration.

use std::fmt;

/// Width-one hidden state used at the talker input boundary.
pub type HiddenState = Vec<f32>;

/// Whether a voice clone uses reference codec tokens (ICL) or x-vector conditioning only.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CloneMode {
    /// Reference codec frames and transcript condition the talker.
    Icl,
    /// A speaker vector conditions the common header; no reference codec frames are used.
    XVector,
}

/// The four observable Base prompt variants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptMode {
    pub clone_mode: CloneMode,
    /// `false` is the official `generate_voice_clone` default: streaming prompt assembly.
    pub non_streaming_mode: bool,
}

impl PromptMode {
    /// The Base voice-cloning default from `generate_voice_clone` (`non_streaming_mode=False`).
    pub const STREAMING_ICL: Self = Self {
        clone_mode: CloneMode::Icl,
        non_streaming_mode: false,
    };

    pub const fn streaming(self) -> bool {
        !self.non_streaming_mode
    }
}

/// The wrapper delimiters which make the upstream slice indices meaningful.
pub const ROLE_PREFIX_IDS: [u32; 3] = [151_644, 77_091, 198];
const TARGET_SUFFIX_IDS: [u32; 5] = [151_645, 198, 151_644, 77_091, 198];
const REFERENCE_SUFFIX_IDS: [u32; 2] = [151_645, 198];

/// Token ids after removing the exact official assistant wrappers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptTextIds {
    pub target: Vec<u32>,
    pub reference: Option<Vec<u32>>,
}

/// Extracts `input_id[:, 3:-5]` and, when present, `ref_ids[:, 3:-2]`.
///
/// OQ-10 originally marked this asymmetric slicing as an inference.  The oracle-fixture L0 test
/// confirms the wrappers before this function is used; keep that test named after Trap 3 rather
/// than folding the rule into an untraceable convenience slice.
pub fn extract_prompt_text_ids(
    target_wrapped: &[u32],
    reference_wrapped: Option<&[u32]>,
) -> Result<PromptTextIds, PromptError> {
    Ok(PromptTextIds {
        target: extract_wrapped(target_wrapped, &TARGET_SUFFIX_IDS, "target")?,
        reference: reference_wrapped
            .map(|ids| extract_wrapped(ids, &REFERENCE_SUFFIX_IDS, "reference"))
            .transpose()?,
    })
}

/// The common header's already-computed components.
///
/// `codec_prefill` is `[think tags..., optional speaker, codec_pad, codec_bos]`.  Its final
/// codec-BOS vector is held back from the header; ICL builds its own codec stream with it, while
/// x-vector consumes it in the first target-text position.  This avoids Trap 1's duplicate BOS.
#[derive(Clone, Debug)]
pub struct PromptHeader {
    pub role: Vec<HiddenState>,
    pub codec_prefill: Vec<HiddenState>,
    pub tts_bos: HiddenState,
    pub tts_pad: HiddenState,
}

/// Embeddings needed to assemble one unbatched Base prompt.
#[derive(Clone, Debug)]
pub struct PromptAssemblyInput {
    pub mode: PromptMode,
    pub header: PromptHeader,
    /// `input_id[:, 3:-5]`, already embedded and projected through the text path.
    pub target_text: Vec<HiddenState>,
    /// `ref_ids[:, 3:-2]`, required only in ICL mode.
    pub reference_text: Option<Vec<HiddenState>>,
    /// Per-frame sum of the sixteen reference-code embeddings, required only in ICL mode.
    pub reference_codec: Option<Vec<HiddenState>>,
    pub tts_eos: HiddenState,
}

/// A fully assembled talker prefill and its subsequent one-hidden-per-frame text stream.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptAssembly {
    pub prefill: Vec<HiddenState>,
    pub trailing_text_hidden: Vec<HiddenState>,
    /// Maximal causal prefix whose KV is independent of the target text (OQ-10 §5).
    pub target_independent_prefix_len: usize,
}

/// Prompt-building failures are explicit rather than silently truncating or broadcasting vectors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptError {
    InvalidWrapper { kind: &'static str },
    InvalidHeader,
    MissingIclReferenceText,
    MissingIclReferenceCodec,
    WidthMismatch { expected: usize, actual: usize },
}

impl fmt::Display for PromptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWrapper { kind } => write!(formatter, "invalid {kind} assistant wrapper"),
            Self::InvalidHeader => {
                formatter.write_str("prompt header requires three role states and codec pad/BOS")
            }
            Self::MissingIclReferenceText => {
                formatter.write_str("ICL prompt lacks reference transcript states")
            }
            Self::MissingIclReferenceCodec => {
                formatter.write_str("ICL prompt lacks reference codec states")
            }
            Self::WidthMismatch { expected, actual } => write!(
                formatter,
                "hidden-state width {actual} differs from {expected}"
            ),
        }
    }
}

impl std::error::Error for PromptError {}

/// Assembles one Base voice-clone prompt exactly as the pinned `generate` branches do.
///
/// The shape contracts are deliberately encoded in the branches themselves:
/// `H+T1+T2`, `H+T2`, `H+1`, and `H+|text|+2` for nonempty target text.  A zero-token target
/// follows PyTorch's zero-length broadcast behavior in the streaming x-vector branch: no first
/// position is appended and only EOS trails.  It is tested explicitly instead of inventing text.
pub fn assemble_prompt(input: PromptAssemblyInput) -> Result<PromptAssembly, PromptError> {
    validate_input(&input)?;
    let PromptAssemblyInput {
        mode,
        header,
        target_text,
        reference_text,
        reference_codec,
        tts_eos,
    } = input;
    let width = header.tts_pad.len();
    let mut prefill = header.role.clone();
    let Some((codec_bos, codec_header)) = header.codec_prefill.split_last() else {
        return Err(PromptError::InvalidHeader);
    };
    let Some(codec_pad) = codec_header.last() else {
        return Err(PromptError::InvalidHeader);
    };
    let codec_bos = codec_bos.clone();
    let codec_pad = codec_pad.clone();

    // M:2177-2186.  Role positions remain text-only; only header positions are summed.
    for (index, codec_state) in codec_header.iter().enumerate() {
        let text_state = if index + 1 == codec_header.len() {
            &header.tts_bos
        } else {
            &header.tts_pad
        };
        prefill.push(add(text_state, codec_state, width));
    }
    let header_len = prefill.len();

    match mode.clone_mode {
        CloneMode::Icl => assemble_icl(
            &mut prefill,
            header_len,
            target_text,
            reference_text.ok_or(PromptError::MissingIclReferenceText)?,
            reference_codec.ok_or(PromptError::MissingIclReferenceCodec)?,
            codec_pad,
            codec_bos,
            header.tts_pad,
            tts_eos,
            mode.non_streaming_mode,
            width,
        ),
        CloneMode::XVector => Ok(assemble_xvector(
            prefill,
            header_len,
            target_text,
            codec_pad,
            codec_bos,
            header.tts_pad,
            tts_eos,
            mode.non_streaming_mode,
            width,
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble_icl(
    prefill: &mut Vec<HiddenState>,
    header_len: usize,
    target_text: Vec<HiddenState>,
    reference_text: Vec<HiddenState>,
    reference_codec: Vec<HiddenState>,
    codec_pad: HiddenState,
    codec_bos: HiddenState,
    tts_pad: HiddenState,
    tts_eos: HiddenState,
    non_streaming: bool,
    width: usize,
) -> Result<PromptAssembly, PromptError> {
    // M:1978-1998: text is ref ++ target ++ eos; codec is bos ++ reference frames.
    let reference_text_len = reference_text.len();
    let mut text_stream = reference_text;
    text_stream.extend(target_text);
    text_stream.push(tts_eos);
    let mut codec_stream = vec![codec_bos];
    codec_stream.extend(reference_codec);
    let prefix_len = header_len + reference_text_len;

    if non_streaming {
        for state in text_stream {
            prefill.push(add(&state, &codec_pad, width));
        }
        for state in codec_stream {
            prefill.push(add(&state, &tts_pad, width));
        }
        return Ok(PromptAssembly {
            prefill: std::mem::take(prefill),
            trailing_text_hidden: vec![tts_pad],
            target_independent_prefix_len: prefix_len,
        });
    }

    let codec_len = codec_stream.len();
    let text_len = text_stream.len();
    for (index, codec_state) in codec_stream.iter().enumerate() {
        let text = text_stream.get(index).unwrap_or(&tts_pad);
        prefill.push(add(text, codec_state, width));
    }
    let trailing_text_hidden = if text_len > codec_len {
        text_stream.split_off(codec_len)
    } else {
        vec![tts_pad]
    };
    Ok(PromptAssembly {
        prefill: std::mem::take(prefill),
        trailing_text_hidden,
        target_independent_prefix_len: prefix_len.min(header_len + codec_len),
    })
}

#[allow(clippy::too_many_arguments)]
fn assemble_xvector(
    mut prefill: Vec<HiddenState>,
    header_len: usize,
    mut target_text: Vec<HiddenState>,
    codec_pad: HiddenState,
    codec_bos: HiddenState,
    tts_pad: HiddenState,
    tts_eos: HiddenState,
    non_streaming: bool,
    width: usize,
) -> PromptAssembly {
    if non_streaming {
        for state in target_text {
            prefill.push(add(&state, &codec_pad, width));
        }
        prefill.push(add(&tts_eos, &codec_pad, width));
        prefill.push(add(&tts_pad, &codec_bos, width));
        return PromptAssembly {
            prefill,
            trailing_text_hidden: vec![tts_pad],
            target_independent_prefix_len: header_len,
        };
    }

    if target_text.is_empty() {
        return PromptAssembly {
            prefill,
            trailing_text_hidden: vec![tts_eos],
            target_independent_prefix_len: header_len,
        };
    }
    let first = target_text.remove(0);
    prefill.push(add(&first, &codec_bos, width));
    target_text.push(tts_eos);
    PromptAssembly {
        prefill,
        trailing_text_hidden: target_text,
        target_independent_prefix_len: header_len,
    }
}

fn validate_input(input: &PromptAssemblyInput) -> Result<(), PromptError> {
    if input.header.role.len() != ROLE_PREFIX_IDS.len() || input.header.codec_prefill.len() < 2 {
        return Err(PromptError::InvalidHeader);
    }
    let width = input.header.tts_pad.len();
    for state in input
        .header
        .role
        .iter()
        .chain(input.header.codec_prefill.iter())
        .chain([&input.header.tts_bos, &input.header.tts_pad, &input.tts_eos])
        .chain(input.target_text.iter())
        .chain(input.reference_text.iter().flatten())
        .chain(input.reference_codec.iter().flatten())
    {
        if state.len() != width {
            return Err(PromptError::WidthMismatch {
                expected: width,
                actual: state.len(),
            });
        }
    }
    Ok(())
}

fn extract_wrapped(
    ids: &[u32],
    suffix: &[u32],
    kind: &'static str,
) -> Result<Vec<u32>, PromptError> {
    if ids.len() < ROLE_PREFIX_IDS.len() + suffix.len()
        || !ids.starts_with(&ROLE_PREFIX_IDS)
        || !ids.ends_with(suffix)
    {
        return Err(PromptError::InvalidWrapper { kind });
    }
    let body_end = ids.len() - suffix.len();
    ids.get(ROLE_PREFIX_IDS.len()..body_end)
        .map(ToOwned::to_owned)
        .ok_or(PromptError::InvalidWrapper { kind })
}

fn add(left: &[f32], right: &[f32], width: usize) -> HiddenState {
    debug_assert_eq!(left.len(), width);
    debug_assert_eq!(right.len(), width);
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(value: f32) -> HiddenState {
        vec![value, value + 0.25]
    }

    fn input(mode: PromptMode) -> PromptAssemblyInput {
        PromptAssemblyInput {
            mode,
            header: PromptHeader {
                role: vec![v(10.0), v(11.0), v(12.0)],
                // 4 language tags + speaker + codec pad + codec BOS: H = 3 + (7 - 1) = 9.
                codec_prefill: vec![
                    v(20.0),
                    v(21.0),
                    v(22.0),
                    v(23.0),
                    v(24.0),
                    v(25.0),
                    v(26.0),
                ],
                tts_bos: v(100.0),
                tts_pad: v(200.0),
            },
            target_text: vec![v(300.0), v(301.0), v(302.0)],
            reference_text: Some(vec![v(400.0), v(401.0)]),
            reference_codec: Some(vec![v(500.0), v(501.0), v(502.0), v(503.0)]),
            tts_eos: v(600.0),
        }
    }

    #[test]
    fn all_four_template_shape_formulas_and_prefixes_are_exact() {
        let cases = [
            (
                PromptMode {
                    clone_mode: CloneMode::Icl,
                    non_streaming_mode: false,
                },
                14,
                1,
                11,
            ),
            (
                PromptMode {
                    clone_mode: CloneMode::Icl,
                    non_streaming_mode: true,
                },
                20,
                1,
                11,
            ),
            (
                PromptMode {
                    clone_mode: CloneMode::XVector,
                    non_streaming_mode: false,
                },
                10,
                3,
                9,
            ),
            (
                PromptMode {
                    clone_mode: CloneMode::XVector,
                    non_streaming_mode: true,
                },
                14,
                1,
                9,
            ),
        ];
        for (mode, prompt_len, trailing_len, prefix_len) in cases {
            let output = assemble_prompt(input(mode)).expect("valid shape fixture");
            assert_eq!(output.prefill.len(), prompt_len, "{mode:?}");
            assert_eq!(output.trailing_text_hidden.len(), trailing_len, "{mode:?}");
            assert_eq!(output.target_independent_prefix_len, prefix_len, "{mode:?}");
        }
    }

    #[test]
    fn streaming_icl_sums_elementwise_and_only_trails_uncovered_text() {
        let output = assemble_prompt(input(PromptMode::STREAMING_ICL)).expect("valid ICL prompt");
        // Header ends at 9.  First ICL position is ref[0] + codec_bos, retaining source addition order.
        assert_eq!(output.prefill[9], add(&v(400.0), &v(26.0), 2));
        // Text stream has 6 elements; codec stream has 5, so just EOS trails.
        assert_eq!(output.trailing_text_hidden, vec![v(600.0)]);
    }

    #[test]
    fn xvector_streaming_default_consumes_bos_once() {
        let mode = PromptMode {
            clone_mode: CloneMode::XVector,
            non_streaming_mode: false,
        };
        let output = assemble_prompt(input(mode)).expect("valid x-vector prompt");
        assert_eq!(
            output.prefill.len(),
            10,
            "header plus exactly one target/BOS position"
        );
        assert_eq!(output.prefill[9], add(&v(300.0), &v(26.0), 2));
        assert_eq!(
            output.trailing_text_hidden,
            vec![v(301.0), v(302.0), v(600.0)]
        );
    }

    #[test]
    fn empty_target_follows_zero_length_streaming_behavior_without_inventing_a_token() {
        let mode = PromptMode {
            clone_mode: CloneMode::XVector,
            non_streaming_mode: false,
        };
        let mut fixture = input(mode);
        fixture.target_text.clear();
        let output = assemble_prompt(fixture).expect("empty target remains representable");
        assert_eq!(output.prefill.len(), 9);
        assert_eq!(output.trailing_text_hidden, vec![v(600.0)]);
    }

    #[test]
    fn trap_3_asymmetric_wrapper_slices_remain_named_and_checked() {
        let target = [
            151_644, 77_091, 198, 7, 8, 151_645, 198, 151_644, 77_091, 198,
        ];
        let reference = [151_644, 77_091, 198, 9, 10, 151_645, 198];
        let ids = extract_prompt_text_ids(&target, Some(&reference)).expect("valid wrappers");
        assert_eq!(ids.target, [7, 8]);
        assert_eq!(ids.reference, Some(vec![9, 10]));
    }

    #[test]
    fn malformed_wrapper_is_not_silently_sliced() {
        assert!(matches!(
            extract_prompt_text_ids(&[151_644, 77_091, 198], None),
            Err(PromptError::InvalidWrapper { kind: "target" })
        ));
    }
}
