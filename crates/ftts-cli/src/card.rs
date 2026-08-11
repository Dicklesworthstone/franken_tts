//! `ftts card`: export a voice as a voice card, import a voice card back.
//!
//! A voice card is the interchange picture the iOS app shares: the full 1,024-float
//! x-vector written as `ftts-voicecard`'s self-locating mosaic, plus a lossless
//! private PNG chunk as the byte-exact fast path. Cards made here import on a phone
//! and cards made on a phone import here, through either layer — the mosaic encoder
//! is bit-identical across the two implementations by test.
//!
//! Layout mirrors the app's card (1024×1180: title band, mosaic, name band); the
//! fonts differ (bundled IBM Plex here, the system font there), which is fine — text
//! is for humans, the mosaic and chunk are the data.

use std::path::{Path, PathBuf};

use fmd_font::Font;
use ftts_video::raster::{FontStack, Surface};

use crate::FttsError;
use crate::synth::SPEAKER_VECTOR_BYTES;

/// Card pixel dimensions: the mosaic square plus title and name bands.
const CARD_WIDTH: usize = ftts_voicecard::CARD_SIZE;
const CARD_HEIGHT: usize = ftts_voicecard::CARD_SIZE + 156;
/// The mosaic's y offset: the title band above it.
const MOSAIC_TOP: usize = 72;

/// The lab's palette, matching the app's `Theme.swift`.
const BACKGROUND: [u8; 3] = [2, 10, 6];
const EMERALD: [u8; 3] = [52, 211, 153];
const TEXT_PRIMARY: [u8; 3] = [226, 232, 240];
const TEXT_SECONDARY: [u8; 3] = [148, 163, 184];

/// Render the full card PNG (mosaic, text bands, lossless chunk) for a voice.
///
/// # Errors
///
/// When the vector has the wrong width or the bundled fonts fail to parse.
pub fn render_card_png(name: &str, vector: &[f32]) -> Result<Vec<u8>, FttsError> {
    if vector.len() != ftts_voicecard::VECTOR_WIDTH {
        return Err(FttsError::Input(format!(
            "a voice card carries exactly {} floats, got {}",
            ftts_voicecard::VECTOR_WIDTH,
            vector.len()
        )));
    }
    let mosaic = ftts_voicecard::render_mosaic_pixels(name, vector);

    let mut rgb = vec![0_u8; CARD_WIDTH * CARD_HEIGHT * 3];
    for pixel in rgb.chunks_mut(3) {
        pixel.copy_from_slice(&BACKGROUND);
    }
    for row in 0..ftts_voicecard::CARD_SIZE {
        let source = row * ftts_voicecard::CARD_SIZE * 3;
        let target = (MOSAIC_TOP + row) * CARD_WIDTH * 3;
        rgb[target..target + ftts_voicecard::CARD_SIZE * 3]
            .copy_from_slice(&mosaic[source..source + ftts_voicecard::CARD_SIZE * 3]);
    }

    // Text bands via the video renderer's font stack (drawn onto an RGBA surface,
    // then alpha-blended onto the card).
    let plex_bold = Font::parse(fmd_font::bundled::PLEX_BOLD.to_vec())
        .map_err(|error| FttsError::Generic(format!("bundled font failed to parse: {error:?}")))?;
    let plex_regular = Font::parse(fmd_font::bundled::PLEX_REGULAR.to_vec())
        .map_err(|error| FttsError::Generic(format!("bundled font failed to parse: {error:?}")))?;
    let bold = FontStack {
        faces: vec![&plex_bold],
    };
    let regular = FontStack {
        faces: vec![&plex_regular],
    };
    let mut text_layer = Surface::new(CARD_WIDTH, CARD_HEIGHT);

    let title = "F R A N K E N T T S · V O I C E  C A R D";
    let title_width = bold.measure(title, 26.0);
    bold.draw(
        &mut text_layer,
        title,
        (CARD_WIDTH as f64 - title_width) / 2.0,
        48.0,
        26.0,
        EMERALD,
        1.0,
    );
    let name_width = bold.measure(name, 42.0);
    bold.draw(
        &mut text_layer,
        name,
        (CARD_WIDTH as f64 - name_width) / 2.0,
        (MOSAIC_TOP + ftts_voicecard::CARD_SIZE + 44) as f64,
        42.0,
        TEXT_PRIMARY,
        1.0,
    );
    let tagline = "the green mosaic is the voice · add it from a photo in FrankenTTS";
    let tagline_width = regular.measure(tagline, 20.0);
    regular.draw(
        &mut text_layer,
        tagline,
        (CARD_WIDTH as f64 - tagline_width) / 2.0,
        (MOSAIC_TOP + ftts_voicecard::CARD_SIZE + 76) as f64,
        20.0,
        TEXT_SECONDARY,
        1.0,
    );
    for (pixel, over) in rgb.chunks_mut(3).zip(text_layer.rgba.chunks(4)) {
        if over[3] == 0 {
            continue;
        }
        let alpha = f32::from(over[3]) / 255.0;
        for c in 0..3 {
            let base = f32::from(pixel[c]);
            pixel[c] = (f32::from(over[c]) * alpha + base * (1.0 - alpha))
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }

    let png = encode_png(&rgb, CARD_WIDTH, CARD_HEIGHT)?;
    ftts_voicecard::embed_chunk(name, vector, &png)
        .ok_or_else(|| FttsError::Generic("card PNG lost its structure".to_owned()))
}

/// Decode a voice from card image bytes: lossless chunk first, then the mosaic.
/// Accepts PNG and JPEG, the formats phones share.
///
/// # Errors
///
/// When the file is neither, or carries no intact voice.
pub fn decode_card(bytes: &[u8]) -> Result<(String, Vec<f32>), FttsError> {
    if let Some(found) = ftts_voicecard::decode_chunk(bytes) {
        return Ok(found);
    }
    let (rgb, width, height) = decode_image_rgb(bytes)?;
    ftts_voicecard::decode(&rgb, width, height).ok_or_else(|| {
        FttsError::Input(
            "no voice found in that picture; voice cards must arrive uncropped (screenshots \
             and messaging-app recompression are fine)"
                .to_owned(),
        )
    })
}

/// Read a `.spk` speaker vector file.
///
/// # Errors
///
/// When the file is missing, the wrong size, or carries non-finite values.
pub fn read_spk(path: &Path) -> Result<Vec<f32>, FttsError> {
    let bytes = std::fs::read(path).map_err(|error| {
        FttsError::Input(format!(
            "cannot read voice file {}: {error}",
            path.display()
        ))
    })?;
    if bytes.len() != SPEAKER_VECTOR_BYTES {
        return Err(FttsError::Input(format!(
            "{} is {} bytes; a speaker vector is exactly {SPEAKER_VECTOR_BYTES}",
            path.display(),
            bytes.len()
        )));
    }
    let vector: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(FttsError::Input(format!(
            "{} carries non-finite values; it is not a usable speaker vector",
            path.display()
        )));
    }
    Ok(vector)
}

/// Write an imported vector as a `.spk` file.
///
/// # Errors
///
/// When the file cannot be written.
pub fn write_spk(path: &Path, vector: &[f32]) -> Result<(), FttsError> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for value in vector {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)
        .map_err(|error| FttsError::Generic(format!("cannot write {}: {error}", path.display())))
}

/// A filesystem-safe name for default output paths.
#[must_use]
pub fn safe_file_stem(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { '-' })
        .collect();
    let trimmed = cleaned.trim_matches('-');
    if trimmed.is_empty() {
        "voice".to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// Default output path for an imported voice, next to the card.
#[must_use]
pub fn default_import_path(card: &Path, name: &str) -> PathBuf {
    card.with_file_name(format!("{}.spk", safe_file_stem(name)))
}

// ------------------------------------------------------------------------ image I/O

fn encode_png(rgb: &[u8], width: usize, height: usize) -> Result<Vec<u8>, FttsError> {
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, width as u32, height as u32);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|error| FttsError::Generic(format!("PNG header: {error}")))?;
        writer
            .write_image_data(rgb)
            .map_err(|error| FttsError::Generic(format!("PNG data: {error}")))?;
    }
    Ok(out)
}

fn decode_image_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), FttsError> {
    const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.len() >= 8 && bytes[..8] == PNG_SIGNATURE {
        return decode_png_rgb(bytes);
    }
    if bytes.len() >= 2 && bytes[..2] == [0xFF, 0xD8] {
        return decode_jpeg_rgb(bytes);
    }
    Err(FttsError::Input(
        "that file is neither PNG nor JPEG; share the card picture itself".to_owned(),
    ))
}

fn decode_png_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), FttsError> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|error| FttsError::Input(format!("unreadable PNG: {error}")))?;
    let mut buffer = vec![0_u8; reader.output_buffer_size().unwrap_or_default()];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|error| FttsError::Input(format!("unreadable PNG: {error}")))?;
    buffer.truncate(info.buffer_size());
    let width = info.width as usize;
    let height = info.height as usize;
    // Depth first: the channel unpacking below assumes one byte per sample, and
    // 16-bit input must be refused before it can be misread as two 8-bit samples.
    if info.bit_depth != png::BitDepth::Eight {
        return Err(FttsError::Input(
            "only 8-bit images are supported; re-save the card as a normal screenshot".to_owned(),
        ));
    }
    let rgb = match info.color_type {
        png::ColorType::Rgb => buffer,
        png::ColorType::Rgba => buffer
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect(),
        png::ColorType::Grayscale => buffer.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buffer
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|px| [px[0], px[0], px[0]])
            .collect(),
        png::ColorType::Indexed => {
            return Err(FttsError::Input(
                "indexed-color PNG; re-export the card as a normal screenshot".to_owned(),
            ));
        }
    };
    Ok((rgb, width, height))
}

fn decode_jpeg_rgb(bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), FttsError> {
    use zune_jpeg::JpegDecoder;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::zune_core::options::DecoderOptions;

    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = JpegDecoder::new_with_options(
        zune_jpeg::zune_core::bytestream::ZCursor::new(bytes),
        options,
    );
    let rgb = decoder
        .decode()
        .map_err(|error| FttsError::Input(format!("unreadable JPEG: {error}")))?;
    let (width, height) = decoder
        .dimensions()
        .ok_or_else(|| FttsError::Input("JPEG carries no dimensions".to_owned()))?;
    Ok((rgb, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vector() -> Vec<f32> {
        let mut state: u64 = 0xBEEF_CAFE_1234_5678;
        (0..ftts_voicecard::VECTOR_WIDTH)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f64 / f64::from(1 << 24) - 0.5) as f32 * 2.0
            })
            .collect()
    }

    #[test]
    fn a_rendered_card_imports_through_both_layers() {
        let vector = test_vector();
        let png = render_card_png("Round Trip", &vector).expect("render");

        // Chunk fast path on the raw bytes.
        let (name, decoded) = decode_card(&png).expect("chunk import");
        assert_eq!(name, "Round Trip");
        assert_eq!(decoded, vector);

        // Pixel path: strip the private chunk by re-encoding the decoded image,
        // which is what a screenshot or a messaging app effectively does.
        let (rgb, width, height) = decode_image_rgb(&png).expect("decode image");
        let stripped = encode_png(&rgb, width, height).expect("re-encode");
        let (name, decoded) = decode_card(&stripped).expect("pixel import");
        assert_eq!(name, "Round Trip");
        assert_eq!(decoded, vector);
    }

    #[test]
    fn an_unrelated_image_is_refused_by_name() {
        let flat = vec![128_u8; 300 * 200 * 3];
        let png = encode_png(&flat, 300, 200).expect("encode");
        let error = decode_card(&png).expect_err("no voice in a flat image");
        assert!(error.to_string().contains("no voice found"));
    }
}
