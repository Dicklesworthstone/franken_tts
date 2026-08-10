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
        // Streaming writers leave a placeholder size in the final data chunk;
        // clamp it to what is actually present instead of refusing the file.
        let body = match data.get(at + 8..at + 8 + size) {
            Some(body) => body,
            None if id == b"data" => &data[at + 8..],
            None => return Err("chunk body truncated".to_owned()),
        };
        match id {
            b"fmt " => {
                let mut tag = le_u16(body, 0).ok_or("fmt chunk truncated")?;
                let channels = le_u16(body, 2).ok_or("fmt chunk truncated")?;
                let rate = le_u32(body, 4).ok_or("fmt chunk truncated")?;
                let bits = le_u16(body, 14).ok_or("fmt chunk truncated")?;
                // WAVE_FORMAT_EXTENSIBLE: the real format code is the first
                // two bytes of the SubFormat GUID at offset 24.
                if tag == 0xFFFE
                    && let Some(subformat) = le_u16(body, 24)
                {
                    tag = subformat;
                }
                format = Some((tag, channels, rate, bits));
            }
            b"data" => payload = Some(body),
            _ => {}
        }
        // Chunks are word-aligned; odd sizes carry a pad byte.
        at += 8 + size + (size & 1);
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
        (3, 32) => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        _ => {
            return Err(format!(
                "unsupported WAV encoding (format tag {tag}, {bits}-bit); \
                 supply uncompressed PCM s16/s24/s32 or IEEE f32"
            ));
        }
    };

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
