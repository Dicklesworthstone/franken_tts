//! Minimal RIFF/WAV reader for the renderer's audio input.
//!
//! `ftts make-video` feeds this either the WAV that `ftts say` just wrote
//! (mono s16, 24 kHz) or a user-supplied PCM WAV. Only uncompressed PCM
//! (s16/s24/s32) and IEEE f32 are accepted; anything else is a refusal that
//! names the supported forms — the same posture as the rest of the CLI.

/// Decoded audio, downmixed to mono f32 in [-1, 1].
pub struct MonoAudio {
    pub sample_rate: u32,
    pub samples: Vec<f32>,
}

fn le_u32(data: &[u8], at: usize) -> Option<u32> {
    data.get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn le_u16(data: &[u8], at: usize) -> Option<u16> {
    data.get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
}

/// Parse a WAV byte stream and downmix to mono f32.
pub fn decode(data: &[u8]) -> Result<MonoAudio, String> {
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE stream".to_owned());
    }
    let mut at = 12usize;
    let mut format: Option<(u16, u16, u32, u16)> = None; // (tag, channels, rate, bits)
    let mut payload: Option<&[u8]> = None;
    while at + 8 <= data.len() {
        let id = &data[at..at + 4];
        let size = le_u32(data, at + 4).ok_or("chunk header truncated")? as usize;
        let body_start = at.checked_add(8).ok_or("chunk offset overflow")?;
        let body_end = body_start.checked_add(size);
        // Streaming writers leave a placeholder size in the final data chunk;
        // clamp it to what is actually present instead of refusing the file.
        let clamped_final_data = body_end.is_none_or(|end| end > data.len()) && id == b"data";
        let body = match body_end.and_then(|end| data.get(body_start..end)) {
            Some(body) => body,
            None if clamped_final_data => &data[body_start..],
            None => return Err("chunk body truncated".to_owned()),
        };
        match id {
            b"fmt " => {
                let mut tag = le_u16(body, 0).ok_or("fmt chunk truncated")?;
                let channels = le_u16(body, 2).ok_or("fmt chunk truncated")?;
                let rate = le_u32(body, 4).ok_or("fmt chunk truncated")?;
                let bits = le_u16(body, 14).ok_or("fmt chunk truncated")?;
                // WAVE_FORMAT_EXTENSIBLE: the real format code is the first
                // two bytes of the SubFormat GUID at offset 24. The remaining GUID
                // bytes are fixed; checking only the prefix would misclassify an
                // unrelated or truncated extensible format as PCM.
                if tag == 0xFFFE {
                    let extension_size =
                        le_u16(body, 16).ok_or("extensible fmt chunk truncated")?;
                    let valid_bits = le_u16(body, 18).ok_or("extensible fmt chunk truncated")?;
                    let subformat = body.get(24..40).ok_or("extensible fmt chunk truncated")?;
                    let declared_end = 18usize + usize::from(extension_size);
                    const WAVE_SUBFORMAT_TAIL: [u8; 14] = [
                        0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
                        0x9B, 0x71,
                    ];
                    if extension_size < 22
                        || body.len() < declared_end
                        || valid_bits == 0
                        || valid_bits > bits
                        || subformat[2..] != WAVE_SUBFORMAT_TAIL
                    {
                        return Err("unsupported WAVE_FORMAT_EXTENSIBLE descriptor".to_owned());
                    }
                    tag = u16::from_le_bytes([subformat[0], subformat[1]]);
                }
                format = Some((tag, channels, rate, bits));
            }
            b"data" => payload = Some(body),
            _ => {}
        }
        if clamped_final_data {
            break;
        }
        // Chunks are word-aligned; odd sizes carry a pad byte.
        at = body_start
            .checked_add(size)
            .and_then(|next| next.checked_add(size & 1))
            .ok_or("chunk offset overflow")?;
    }
    let (tag, channels, sample_rate, bits) = format.ok_or("missing fmt chunk")?;
    let payload = payload.ok_or("missing data chunk")?;
    if channels == 0 {
        return Err("WAV reports zero channels".to_owned());
    }
    if sample_rate == 0 || sample_rate > 768_000 {
        return Err(format!("implausible WAV sample rate {sample_rate}"));
    }
    let channels = usize::from(channels);
    let bytes_per_sample = match (tag, bits) {
        (1, 16) => 2,
        (1, 24) => 3,
        (1, 32) | (3, 32) => 4,
        _ => {
            return Err(format!(
                "unsupported WAV encoding (format tag {tag}, {bits}-bit); \
                 supply uncompressed PCM s16/s24/s32 or IEEE f32"
            ));
        }
    };
    if !payload.len().is_multiple_of(bytes_per_sample) {
        return Err("WAV data ends with a partial sample".to_owned());
    }

    let frames: Vec<f32> = match (tag, bits) {
        (1, 16) => payload
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from(i16::from_le_bytes(*b)) / 32768.0)
            .collect(),
        (1, 24) => payload
            .as_chunks::<3>()
            .0
            .iter()
            .map(|b| {
                let v = i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8;
                v as f32 / 8_388_608.0
            })
            .collect(),
        (1, 32) => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| i32::from_le_bytes(*b) as f32 / 2_147_483_648.0)
            .collect(),
        (3, 32) => {
            let mut decoded = Vec::with_capacity(payload.len() / 4);
            for bytes in payload.as_chunks::<4>().0 {
                let sample = f32::from_le_bytes(*bytes);
                if !sample.is_finite() {
                    return Err("IEEE-float WAV contains a non-finite sample".to_owned());
                }
                decoded.push(sample.clamp(-1.0, 1.0));
            }
            decoded
        }
        _ => return Err("internal WAV encoding validation drift".to_owned()),
    };

    if !frames.len().is_multiple_of(channels) {
        return Err("WAV data ends with a partial channel frame".to_owned());
    }

    let samples = if channels == 1 {
        frames
    } else {
        frames
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    if samples.is_empty() {
        return Err("WAV contains no audio samples".to_owned());
    }
    Ok(MonoAudio {
        sample_rate,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(format_tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        wav_with_fmt_extra(format_tag, channels, rate, bits, &[], data)
    }

    fn wav_with_fmt_extra(
        format_tag: u16,
        channels: u16,
        rate: u32,
        bits: u16,
        fmt_extra: &[u8],
        data: &[u8],
    ) -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&format_tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * u32::from(channels) * u32::from(bits) / 8).to_le_bytes());
        fmt.extend_from_slice(&(channels * bits / 8).to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        fmt.extend_from_slice(fmt_extra);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&0_u32.to_le_bytes()); // RIFF size: unread
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&fmt);
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn s16_mono_decodes_exactly() {
        let data: Vec<u8> = [0_i16, 16384, -16384, 32767]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let audio = decode(&wav(1, 1, 24000, 16, &data)).expect("valid wav");
        assert_eq!(audio.sample_rate, 24000);
        assert_eq!(audio.samples, [0.0, 0.5, -0.5, 32767.0 / 32768.0]);
    }

    #[test]
    fn stereo_downmixes_to_the_channel_mean() {
        let data: Vec<u8> = [16384_i16, -16384, 8192, 8192]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let audio = decode(&wav(1, 2, 48000, 16, &data)).expect("valid wav");
        assert_eq!(audio.samples, [0.0, 0.25]);
    }

    #[test]
    fn extensible_resolves_the_subformat_and_float_decodes() {
        // WAVE_FORMAT_EXTENSIBLE (0xFFFE): cbSize(22) + validBits + channelMask + SubFormat
        // GUID whose first two bytes are the real format code — 3 = IEEE float here. The old
        // reader pattern-matched 0xFFFE as int PCM, decoding float bytes as garbage ints.
        let mut extra = Vec::new();
        extra.extend_from_slice(&22_u16.to_le_bytes());
        extra.extend_from_slice(&32_u16.to_le_bytes());
        extra.extend_from_slice(&0_u32.to_le_bytes());
        extra.extend_from_slice(&[
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
            0x9B, 0x71,
        ]); // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        let data: Vec<u8> = [0.25_f32, -1.0]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let audio =
            decode(&wav_with_fmt_extra(0xFFFE, 1, 24000, 32, &extra, &data)).expect("valid wav");
        assert_eq!(audio.samples, [0.25, -1.0]);
    }

    #[test]
    fn extensible_rejects_a_nonstandard_subformat_guid() {
        let mut extra = Vec::new();
        extra.extend_from_slice(&22_u16.to_le_bytes());
        extra.extend_from_slice(&32_u16.to_le_bytes());
        extra.extend_from_slice(&0_u32.to_le_bytes());
        extra.extend_from_slice(&3_u16.to_le_bytes());
        extra.extend_from_slice(&[0; 14]);

        assert!(decode(&wav_with_fmt_extra(0xFFFE, 1, 24000, 32, &extra, &[0; 4])).is_err());
    }

    #[test]
    fn extensible_rejects_a_declared_extension_larger_than_the_fmt_chunk() {
        let mut extra = Vec::new();
        extra.extend_from_slice(&23_u16.to_le_bytes());
        extra.extend_from_slice(&32_u16.to_le_bytes());
        extra.extend_from_slice(&0_u32.to_le_bytes());
        extra.extend_from_slice(&[
            0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
            0x9B, 0x71,
        ]);

        assert!(decode(&wav_with_fmt_extra(0xFFFE, 1, 24000, 32, &extra, &[0; 4])).is_err());
    }

    #[test]
    fn ieee_float_refuses_nonfinite_and_clamps_finite_overrange_samples() {
        let finite: Vec<u8> = [-2.0_f32, 0.25, 1.5]
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        let decoded = decode(&wav(3, 1, 24000, 32, &finite)).expect("finite float WAV");
        assert_eq!(decoded.samples, [-1.0, 0.25, 1.0]);

        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                decode(&wav(3, 1, 24000, 32, &invalid.to_le_bytes())).is_err(),
                "non-finite sample {invalid:?} must refuse"
            );
        }
    }

    #[test]
    fn a_placeholder_data_size_clamps_to_the_bytes_present() {
        // Streaming writers leave 0xFFFFFFFF in the final data chunk header.
        let mut bytes = wav(1, 1, 24000, 16, &1234_i16.to_le_bytes());
        let data_size_at = bytes.len() - 2 - 4;
        bytes[data_size_at..data_size_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let audio = decode(&bytes).expect("clamped, not refused");
        assert_eq!(audio.samples.len(), 1);
    }

    #[test]
    fn hostile_headers_are_refused() {
        assert!(decode(b"RIFFxxxxWAVE").is_err(), "no chunks");
        assert!(
            decode(&wav(1, 1, 0, 16, &[0, 0])).is_err(),
            "zero sample rate would divide by zero downstream"
        );
        assert!(
            decode(&wav(1, 0, 24000, 16, &[0, 0])).is_err(),
            "zero channels"
        );
        assert!(
            decode(&wav(85, 1, 24000, 16, &[0, 0])).is_err(),
            "compressed encodings are a typed refusal"
        );
        assert!(decode(&wav(1, 1, 24000, 16, &[])).is_err(), "no samples");
        assert!(
            decode(&wav(1, 1, 24000, 16, &[0, 0, 1])).is_err(),
            "partial samples must not be silently discarded"
        );
        assert!(
            decode(&wav(1, 2, 24000, 16, &[0, 0])).is_err(),
            "partial channel frames must not be silently discarded"
        );
    }
}
