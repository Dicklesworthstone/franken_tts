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
