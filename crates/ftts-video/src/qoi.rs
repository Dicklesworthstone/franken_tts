//! Minimal QOI (Quite OK Image) decoder for the embedded background art.
//!
//! The illustration ships inside the binary as QOI because the format is
//! decodable in under a hundred lines of obviously-safe Rust — no DEFLATE,
//! no chunk graph, no color management. The asset is trusted (compiled in),
//! but every read is still bounds-checked so a corrupted build fails loudly.

/// A decoded RGBA8 image.
pub struct Image {
    pub width: u32,
    pub height: u32,
    /// Interleaved RGBA, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
}

/// Decode a QOI byte stream into RGBA8.
pub fn decode(data: &[u8]) -> Result<Image, String> {
    if data.len() < 14 || &data[0..4] != b"qoif" {
        return Err("not a QOI stream".to_owned());
    }
    let width = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
    let height = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
    if !matches!(data[12], 3 | 4) || data[13] > 1 {
        return Err("invalid QOI channels or colorspace".to_owned());
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or("QOI dimensions overflow")?;
    // The embedded asset is ~1.6 Mpx; anything wildly larger is corruption.
    if pixels == 0 || pixels > 64_000_000 {
        return Err(format!("implausible QOI dimensions {width}x{height}"));
    }

    let mut rgba = vec![0u8; pixels * 4];
    let mut index = [[0u8; 4]; 64];
    let (mut r, mut g, mut b, mut a) = (0u8, 0u8, 0u8, 255u8);
    let mut src = 14usize;
    let mut px = 0usize;

    while px < pixels {
        let byte = *data.get(src).ok_or("QOI stream truncated")?;
        src += 1;
        let mut run = 1usize;
        match byte {
            0xFE => {
                // QOI_OP_RGB
                let chunk = data.get(src..src + 3).ok_or("QOI RGB truncated")?;
                (r, g, b) = (chunk[0], chunk[1], chunk[2]);
                src += 3;
            }
            0xFF => {
                // QOI_OP_RGBA
                let chunk = data.get(src..src + 4).ok_or("QOI RGBA truncated")?;
                (r, g, b, a) = (chunk[0], chunk[1], chunk[2], chunk[3]);
                src += 4;
            }
            _ => match byte >> 6 {
                0b00 => [r, g, b, a] = index[usize::from(byte & 0x3F)],
                0b01 => {
                    // QOI_OP_DIFF: 2-bit channel deltas biased by 2.
                    r = r.wrapping_add((byte >> 4) & 0x03).wrapping_sub(2);
                    g = g.wrapping_add((byte >> 2) & 0x03).wrapping_sub(2);
                    b = b.wrapping_add(byte & 0x03).wrapping_sub(2);
                }
                0b10 => {
                    // QOI_OP_LUMA: 6-bit green delta, red/blue relative to it.
                    let dg = (byte & 0x3F).wrapping_sub(32);
                    let rb = *data.get(src).ok_or("QOI LUMA truncated")?;
                    src += 1;
                    g = g.wrapping_add(dg);
                    r = r
                        .wrapping_add(dg)
                        .wrapping_add((rb >> 4) & 0x0F)
                        .wrapping_sub(8);
                    b = b.wrapping_add(dg).wrapping_add(rb & 0x0F).wrapping_sub(8);
                }
                _ => run = usize::from(byte & 0x3F) + 1, // QOI_OP_RUN
            },
        }
        let hash =
            (usize::from(r) * 3 + usize::from(g) * 5 + usize::from(b) * 7 + usize::from(a) * 11)
                % 64;
        index[hash] = [r, g, b, a];
        let end = px.checked_add(run).ok_or("QOI run overflow")?;
        if end > pixels {
            return Err("QOI run exceeds pixel count".to_owned());
        }
        for pixel in rgba[px * 4..end * 4].as_chunks_mut::<4>().0 {
            *pixel = [r, g, b, a];
        }
        px = end;
    }
    Ok(Image {
        width,
        height,
        rgba,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a QOI stream by hand: 14-byte header, the given op bytes, no end marker
    /// (the decoder stops at the pixel count, which the embedded-asset path relies on).
    fn stream(width: u32, height: u32, ops: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"qoif");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[4, 0]); // channels, colorspace: read but unused
        bytes.extend_from_slice(ops);
        bytes
    }

    #[test]
    fn every_op_decodes_to_the_specified_pixels() {
        // 6 pixels: RGB, RUN(1), DIFF(+1,+1,+1), LUMA(dg=8, dr-dg=2, db-dg=-2), INDEX, RGBA.
        let ops = [
            0xFE,
            100,
            150,
            200,         // px0: RGB (100, 150, 200, 255)
            0xC0,        // px1: RUN 1 -> repeat px0
            0b0111_1111, // px2: DIFF +1/+1/+1 -> (101, 151, 201, 255)
            0b10_101000, // px3: LUMA dg=8 (40-32) ...
            0b1010_0110, // ... dr-dg=2, db-dg=-2 -> (111, 159, 207, 255)
            // px4: INDEX of px0's hash slot
            (100_usize * 3 + 150 * 5 + 200 * 7 + 255 * 11) as u8 % 64,
            0xFF,
            1,
            2,
            3,
            4, // px5: RGBA (1, 2, 3, 4)
        ];
        let image = decode(&stream(6, 1, &ops)).expect("valid stream");
        let expected: &[[u8; 4]] = &[
            [100, 150, 200, 255],
            [100, 150, 200, 255],
            [101, 151, 201, 255],
            [111, 159, 207, 255],
            [100, 150, 200, 255],
            [1, 2, 3, 4],
        ];
        for (index, pixel) in expected.iter().enumerate() {
            assert_eq!(
                &image.rgba[index * 4..index * 4 + 4],
                pixel,
                "pixel {index}"
            );
        }
    }

    #[test]
    fn truncated_and_overrunning_streams_are_refused() {
        assert!(decode(b"qoif").is_err(), "short header");
        assert!(decode(&stream(2, 1, &[0xFE, 1, 2])).is_err(), "cut RGB op");
        assert!(
            decode(&stream(2, 1, &[0xFE, 1, 2, 3, 0xC4])).is_err(),
            "a run past the pixel count must refuse, not write out of bounds"
        );
        assert!(decode(&stream(0, 5, &[])).is_err(), "zero dimension");
        let mut invalid_header = stream(1, 1, &[0xC0]);
        invalid_header[12] = 2;
        assert!(decode(&invalid_header).is_err(), "invalid channel count");
        invalid_header[12] = 4;
        invalid_header[13] = 2;
        assert!(decode(&invalid_header).is_err(), "invalid colorspace");
    }

    #[test]
    fn the_embedded_illustration_decodes() {
        let image = decode(crate::ILLUSTRATION_QOI).expect("embedded asset");
        assert_eq!((image.width, image.height), (1672, 941));
        assert_eq!(image.rgba.len(), 1672 * 941 * 4);
        // Not all-black and not all-white: a trivially wrong decode would be one of those.
        let sum: u64 = image.rgba.iter().map(|&byte| u64::from(byte)).sum();
        let max = image.rgba.len() as u64 * 255;
        assert!(
            sum > max / 100 && sum < max * 99 / 100,
            "implausible content"
        );
    }
}
