//! RGB24 → planar YUV 4:2:0 (BT.709, limited range) and Y4M framing.
//!
//! Y4M is the native, encoder-free output: any player or encoder that speaks
//! YUV4MPEG2 (ffmpeg, mpv, x264) accepts it directly, so the renderer stays
//! useful on a machine with no encoder installed.

use std::io::Write;

/// Convert one RGB24 frame into planar I420 (BT.709 limited range).
///
/// `y`, `u`, `v` must be `w*h`, `w/2*h/2`, `w/2*h/2` bytes; `w` and `h` even.
pub fn rgb_to_i420(
    rgb: &[u8],
    width: usize,
    height: usize,
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
) {
    // Luma per pixel.
    for row in 0..height {
        for col in 0..width {
            let at = (row * width + col) * 3;
            let r = f32::from(rgb[at]);
            let g = f32::from(rgb[at + 1]);
            let b = f32::from(rgb[at + 2]);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            y[row * width + col] = (16.0 + luma * (219.0 / 255.0)).round().clamp(16.0, 235.0) as u8;
        }
    }
    // Chroma per 2×2 block, averaged in R'G'B' before the matrix — adequate
    // for this content and keeps the loop trivially safe.
    let chroma_width = width / 2;
    for row in 0..height / 2 {
        for col in 0..chroma_width {
            let (mut r_sum, mut g_sum, mut b_sum) = (0.0f32, 0.0f32, 0.0f32);
            for dy in 0..2 {
                for dx in 0..2 {
                    let at = ((row * 2 + dy) * width + col * 2 + dx) * 3;
                    r_sum += f32::from(rgb[at]);
                    g_sum += f32::from(rgb[at + 1]);
                    b_sum += f32::from(rgb[at + 2]);
                }
            }
            let (r, g, b) = (r_sum / 4.0, g_sum / 4.0, b_sum / 4.0);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let cb = (b - luma) / 1.8556;
            let cr = (r - luma) / 1.5748;
            u[row * chroma_width + col] =
                (128.0 + cb * (224.0 / 255.0)).round().clamp(16.0, 240.0) as u8;
            v[row * chroma_width + col] =
                (128.0 + cr * (224.0 / 255.0)).round().clamp(16.0, 240.0) as u8;
        }
    }
}

/// Write the YUV4MPEG2 stream header.
pub fn write_y4m_header(
    sink: &mut dyn Write,
    width: usize,
    height: usize,
    fps: u32,
) -> std::io::Result<()> {
    writeln!(
        sink,
        "YUV4MPEG2 W{width} H{height} F{fps}:1 Ip A1:1 C420mpeg2"
    )
}

/// Write one Y4M frame from planar I420 data.
pub fn write_y4m_frame(sink: &mut dyn Write, y: &[u8], u: &[u8], v: &[u8]) -> std::io::Result<()> {
    sink.write_all(b"FRAME\n")?;
    sink.write_all(y)?;
    sink.write_all(u)?;
    sink.write_all(v)
}
