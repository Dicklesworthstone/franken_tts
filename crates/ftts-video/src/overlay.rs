//! The static 1920×1080 branded overlay: gradient scrims plus text.
//!
//! Rendered once per video and composited over every frame. Layout mirrors
//! the launch-announcement videos: title block top-left, voice pill top-right,
//! site URL bottom-left, capability footer bottom-right, with darkening
//! scrims so text and waveform stay legible over the busy illustration.

use crate::raster::{FontStack, Surface};
use fmd_font::Font;

pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;

const GREEN: [u8; 3] = [0x35, 0xE8, 0xA4];
const MINT: [u8; 3] = [0xD8, 0xF4, 0xEA];
const FAINT: [u8; 3] = [0xBF, 0xE9, 0xDD];
const WHITE: [u8; 3] = [0xFF, 0xFF, 0xFF];

/// Signed distance from `point` to a rounded rectangle; negative inside.
fn rounded_rect_distance(point: (f64, f64), min: (f64, f64), max: (f64, f64), radius: f64) -> f64 {
    let center = ((min.0 + max.0) / 2.0, (min.1 + max.1) / 2.0);
    let half = (
        (max.0 - min.0) / 2.0 - radius,
        (max.1 - min.1) / 2.0 - radius,
    );
    let dx = (point.0 - center.0).abs() - half.0;
    let dy = (point.1 - center.1).abs() - half.1;
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    outside + dx.max(dy).min(0.0) - radius
}

/// Fill a rounded rectangle with 1-px analytic edge anti-aliasing.
fn fill_rounded_rect(
    surface: &mut Surface,
    min: (f64, f64),
    max: (f64, f64),
    radius: f64,
    color: [u8; 3],
    alpha: f32,
    border: Option<([u8; 3], f64)>,
) {
    let x0 = (min.0.floor() as i64 - 2).max(0);
    let y0 = (min.1.floor() as i64 - 2).max(0);
    let x1 = (max.0.ceil() as i64 + 2).min(surface.width as i64);
    let y1 = (max.1.ceil() as i64 + 2).min(surface.height as i64);
    for y in y0..y1 {
        for x in x0..x1 {
            let d = rounded_rect_distance((x as f64 + 0.5, y as f64 + 0.5), min, max, radius);
            let coverage = (0.5 - d).clamp(0.0, 1.0) as f32;
            if coverage > 0.0 {
                surface.blend(x, y, color, alpha * coverage);
            }
            if let Some((border_color, border_width)) = border {
                let ring = (border_width / 2.0 + 0.5 - (d + border_width / 2.0).abs())
                    .clamp(0.0, 1.0) as f32;
                if ring > 0.0 {
                    surface.blend(x, y, border_color, ring);
                }
            }
        }
    }
}

/// Build the full overlay for a given voice label.
pub fn build(voice_label: &str) -> Result<Surface, String> {
    let plex_bold = Font::parse(fmd_font::bundled::PLEX_BOLD.to_vec())
        .map_err(|error| format!("bundled IBM Plex Bold failed to parse: {error:?}"))?;
    let plex_regular = Font::parse(fmd_font::bundled::PLEX_REGULAR.to_vec())
        .map_err(|error| format!("bundled IBM Plex Regular failed to parse: {error:?}"))?;
    let noto_math = Font::parse(fmd_font::bundled::NOTO_SANS_MATH_SYMBOLS.to_vec())
        .map_err(|error| format!("bundled Noto Math failed to parse: {error:?}"))?;

    let bold = FontStack {
        faces: vec![&plex_bold, &noto_math],
    };
    let regular = FontStack {
        faces: vec![&plex_regular, &noto_math],
    };

    let mut surface = Surface::new(WIDTH, HEIGHT);

    // Top scrim: 65% black fading out over 270 px.
    for y in 0..270 {
        let alpha = 0.65 * (1.0 - y as f32 / 270.0).powf(1.3);
        for x in 0..WIDTH {
            surface.blend(x as i64, i64::from(y), [0, 0, 0], alpha);
        }
    }
    // Bottom scrim: fades in from y=720.
    for y in 720..HEIGHT {
        let alpha = 0.78 * ((y - 720) as f32 / (HEIGHT - 720) as f32).powf(1.1);
        for x in 0..WIDTH {
            surface.blend(x as i64, y as i64, [0, 0, 0], alpha);
        }
    }
    // Waveform band: extra soft-edged darkening centered on the drawn wave.
    for y in 750..1050 {
        let edge = (((y - 750) as f32) / 40.0)
            .min(((1050 - y) as f32) / 40.0)
            .min(1.0);
        for x in 0..WIDTH {
            surface.blend(x as i64, y as i64, [0, 0, 0], 0.68 * edge);
        }
    }

    // Title block, top-left. Baselines sit at cap-height offsets from the
    // Python/ffmpeg layout's top-left anchors.
    bold.draw_with_shadow(&mut surface, "FrankenTTS", 64.0, 128.0, 96.0, WHITE, 0.8);
    regular.draw_with_shadow(
        &mut surface,
        "Text-to-speech in your browser — pure Rust → WebAssembly · no server · no GPU",
        68.0,
        192.0,
        34.0,
        MINT,
        0.8,
    );

    // Voice pill, top-right.
    let label = format!("Voice: \u{201C}{voice_label}\u{201D}");
    let label_size = 44.0;
    let label_width = bold.measure(&label, label_size);
    let pill_right = WIDTH as f64 - 64.0;
    let pill_left = pill_right - label_width - 80.0;
    fill_rounded_rect(
        &mut surface,
        (pill_left, 52.0),
        (pill_right, 130.0),
        39.0,
        [0, 0, 0],
        0.5,
        Some((GREEN, 3.0)),
    );
    bold.draw_with_shadow(
        &mut surface,
        &label,
        pill_left + 40.0,
        106.0,
        label_size,
        GREEN,
        0.8,
    );

    // Bottom-left URL and bottom-right capability line.
    bold.draw_with_shadow(
        &mut surface,
        "frankentts.com",
        64.0,
        1020.0,
        40.0,
        WHITE,
        0.85,
    );
    let footer =
        "Generated locally by the ftts CLI · clone a voice from a short clip · share it as a URL";
    let footer_width = regular.measure(footer, 26.0);
    regular.draw_with_shadow(
        &mut surface,
        footer,
        WIDTH as f64 - 64.0 - footer_width,
        1022.0,
        26.0,
        FAINT,
        0.85,
    );

    Ok(surface)
}
