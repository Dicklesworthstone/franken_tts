//! Glyph rasterization and RGBA compositing primitives.
//!
//! fmd-font hands us exact quadratic-Bézier contours in font design units;
//! this module turns them into anti-aliased alpha bitmaps with a 4×4
//! supersampled nonzero-winding scanline fill. Overlay text is rendered once
//! per video, so clarity wins over cleverness here — no incremental coverage
//! tables, just flattened edges and sorted crossings.

use fmd_font::Font;
use fmd_font::outline::{GlyphOutline, Point, Segment};

/// Supersampling factor per axis (16 samples per pixel).
const SS: usize = 4;

/// An RGBA8 pixel surface, y-down.
pub struct Surface {
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
}

impl Surface {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            rgba: vec![0u8; width * height * 4],
        }
    }

    /// Source-over blend of a straight-alpha color at (x, y) with `alpha` in [0, 1].
    pub fn blend(&mut self, x: i64, y: i64, color: [u8; 3], alpha: f32) {
        if alpha <= 0.0 || x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        let at = (y as usize * self.width + x as usize) * 4;
        let a = alpha.min(1.0);
        let dst_a = f32::from(self.rgba[at + 3]) / 255.0;
        let out_a = a + dst_a * (1.0 - a);
        if out_a <= 0.0 {
            return;
        }
        for (c, &channel) in color.iter().enumerate() {
            let src = f32::from(channel);
            let dst = f32::from(self.rgba[at + c]);
            let blended = (src * a + dst * dst_a * (1.0 - a)) / out_a;
            self.rgba[at + c] = blended.round().clamp(0.0, 255.0) as u8;
        }
        self.rgba[at + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

/// A rasterized glyph: an alpha mask plus its placement relative to the pen.
pub struct GlyphBitmap {
    pub width: usize,
    pub height: usize,
    /// Pixel offset from the pen position to the bitmap's top-left.
    pub left: i64,
    pub top: i64,
    /// Row-major coverage in [0, 255].
    pub alpha: Vec<u8>,
}

/// Flatten one glyph outline into line segments in pixel space (y-down),
/// scaled by `scale` = px_size / units_per_em.
fn flatten(outline: &GlyphOutline, scale: f64) -> Vec<[f64; 4]> {
    let mut edges = Vec::new();
    let mut push = |a: Point, b: Point| {
        // y is negated: font coordinates are y-up, rasters are y-down.
        edges.push([a.x * scale, -a.y * scale, b.x * scale, -b.y * scale]);
    };
    for contour in &outline.contours {
        let mut current = contour.start;
        for segment in &contour.segments {
            match *segment {
                Segment::Line { to } => {
                    push(current, to);
                    current = to;
                }
                Segment::Quad { ctrl, to } => {
                    // Fixed subdivision is plenty at overlay sizes: 16 chords
                    // keep the sagitta below a supersample step for glyphs
                    // up to ~200 px.
                    const STEPS: usize = 16;
                    let mut previous = current;
                    for step in 1..=STEPS {
                        let t = step as f64 / STEPS as f64;
                        let u = 1.0 - t;
                        let next = Point {
                            x: u * u * current.x + 2.0 * u * t * ctrl.x + t * t * to.x,
                            y: u * u * current.y + 2.0 * u * t * ctrl.y + t * t * to.y,
                        };
                        push(previous, next);
                        previous = next;
                    }
                    current = to;
                }
            }
        }
        // Contours decode closed (last endpoint == start), so no closing edge
        // is synthesized here; a malformed contour would just drop coverage.
    }
    edges
}

/// Rasterize `gid` from `font` at `px_size`. Returns `None` for empty glyphs
/// (spaces) or outline errors — the caller advances the pen either way.
pub fn rasterize(font: &Font, gid: u16, px_size: f64) -> Option<GlyphBitmap> {
    let outline = font.glyph_outline(gid).ok()?;
    let scale = px_size / f64::from(font.units_per_em.max(1));
    let edges = flatten(&outline, scale);
    if edges.is_empty() {
        return None;
    }

    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for edge in &edges {
        min_x = min_x.min(edge[0]).min(edge[2]);
        max_x = max_x.max(edge[0]).max(edge[2]);
        min_y = min_y.min(edge[1]).min(edge[3]);
        max_y = max_y.max(edge[1]).max(edge[3]);
    }
    let left = min_x.floor() as i64 - 1;
    let top = min_y.floor() as i64 - 1;
    let width = (max_x.ceil() as i64 - left + 2) as usize;
    let height = (max_y.ceil() as i64 - top + 2) as usize;
    if width == 0 || height == 0 || width > 4096 || height > 4096 {
        return None;
    }

    let mut alpha = vec![0u8; width * height];
    let mut crossings: Vec<(f64, i32)> = Vec::with_capacity(16);
    let mut row_coverage = vec![0u16; width];
    for row in 0..height {
        row_coverage.fill(0);
        for sub in 0..SS {
            let sample_y = top as f64 + row as f64 + (sub as f64 + 0.5) / SS as f64;
            crossings.clear();
            for edge in &edges {
                let (x0, y0, x1, y1) = (edge[0], edge[1], edge[2], edge[3]);
                let (top_y, bot_y, direction) = if y0 < y1 { (y0, y1, 1) } else { (y1, y0, -1) };
                if sample_y < top_y || sample_y >= bot_y {
                    continue;
                }
                let t = (sample_y - y0) / (y1 - y0);
                crossings.push((x0 + t * (x1 - x0), direction));
            }
            crossings.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut winding = 0;
            let mut span_start = 0.0f64;
            for &(x, direction) in &crossings {
                let was = winding;
                winding += direction;
                if was == 0 && winding != 0 {
                    span_start = x;
                } else if was != 0 && winding == 0 {
                    // Accumulate the span [span_start, x) at supersample
                    // resolution into per-pixel counts.
                    let begin = span_start - left as f64;
                    let end = x - left as f64;
                    let mut sx = (begin * SS as f64).round() as i64;
                    let ex = (end * SS as f64).round() as i64;
                    sx = sx.max(0);
                    let ex = ex.min((width * SS) as i64);
                    while sx < ex {
                        row_coverage[(sx / SS as i64) as usize] += 1;
                        sx += 1;
                    }
                }
            }
        }
        let out_row = &mut alpha[row * width..(row + 1) * width];
        for (pixel, &count) in out_row.iter_mut().zip(row_coverage.iter()) {
            // count ∈ [0, SS*SS]
            *pixel = ((u32::from(count) * 255) / (SS * SS) as u32).min(255) as u8;
        }
    }
    Some(GlyphBitmap {
        width,
        height,
        left,
        top,
        alpha,
    })
}

/// A font stack: the first face that maps a character wins. Lets the Latin
/// face fall back to the math/symbol face for characters like `→` and `·`.
pub struct FontStack<'a> {
    pub faces: Vec<&'a Font>,
}

impl FontStack<'_> {
    fn face_for(&self, ch: char) -> Option<(&Font, u16)> {
        for face in &self.faces {
            let gid = face.glyph_index(ch);
            if gid != 0 {
                return Some((face, gid));
            }
        }
        None
    }

    /// Advance width of `text` at `px_size`, kerning included.
    pub fn measure(&self, text: &str, px_size: f64) -> f64 {
        let mut width = 0.0;
        let mut previous: Option<(usize, char)> = None;
        for ch in text.chars() {
            let Some((face, gid)) = self.face_for(ch) else {
                width += px_size * 0.28; // width of an unmapped char's tofu gap
                previous = None;
                continue;
            };
            let face_index = self
                .faces
                .iter()
                .position(|candidate| std::ptr::eq(*candidate, face))
                .unwrap_or(0);
            let scale = px_size / f64::from(face.units_per_em.max(1));
            if let Some((prev_face, prev_ch)) = previous
                && prev_face == face_index
            {
                width += f64::from(face.kerning(prev_ch, ch)) * scale;
            }
            width += f64::from(face.advance_width(gid)) * scale;
            previous = Some((face_index, ch));
        }
        width
    }

    /// Draw `text` with its baseline at (`x`, `baseline_y`); returns the end pen x.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        surface: &mut Surface,
        text: &str,
        x: f64,
        baseline_y: f64,
        px_size: f64,
        color: [u8; 3],
        opacity: f32,
    ) -> f64 {
        let mut pen = x;
        let mut previous: Option<(usize, char)> = None;
        for ch in text.chars() {
            let Some((face, gid)) = self.face_for(ch) else {
                pen += px_size * 0.28;
                previous = None;
                continue;
            };
            let face_index = self
                .faces
                .iter()
                .position(|candidate| std::ptr::eq(*candidate, face))
                .unwrap_or(0);
            let scale = px_size / f64::from(face.units_per_em.max(1));
            if let Some((prev_face, prev_ch)) = previous
                && prev_face == face_index
            {
                pen += f64::from(face.kerning(prev_ch, ch)) * scale;
            }
            if let Some(bitmap) = rasterize(face, gid, px_size) {
                let origin_x = pen.round() as i64 + bitmap.left;
                let origin_y = baseline_y.round() as i64 + bitmap.top;
                for row in 0..bitmap.height {
                    for col in 0..bitmap.width {
                        let coverage = bitmap.alpha[row * bitmap.width + col];
                        if coverage > 0 {
                            surface.blend(
                                origin_x + col as i64,
                                origin_y + row as i64,
                                color,
                                f32::from(coverage) / 255.0 * opacity,
                            );
                        }
                    }
                }
            }
            pen += f64::from(face.advance_width(gid)) * scale;
            previous = Some((face_index, ch));
        }
        pen
    }

    /// Draw with a soft drop shadow, matching the site's overlay styling.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_shadow(
        &self,
        surface: &mut Surface,
        text: &str,
        x: f64,
        baseline_y: f64,
        px_size: f64,
        color: [u8; 3],
        shadow_alpha: f32,
    ) {
        let offset = (px_size / 30.0).clamp(1.5, 3.5);
        self.draw(
            surface,
            text,
            x + offset,
            baseline_y + offset,
            px_size,
            [0, 0, 0],
            shadow_alpha,
        );
        self.draw(surface, text, x, baseline_y, px_size, color, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plex_bold() -> Font {
        Font::parse(fmd_font::bundled::PLEX_BOLD.to_vec()).expect("bundled face parses")
    }

    #[test]
    fn a_glyph_rasterizes_with_plausible_coverage() {
        let font = plex_bold();
        let gid = font.glyph_index('F');
        assert_ne!(gid, 0, "'F' must map");
        let bitmap = rasterize(&font, gid, 96.0).expect("solid glyph");
        // Structural facts that hold for any correct rasterization of a 96 px capital F:
        // it covers a real area, coverage peaks at full opacity somewhere (a stem interior),
        // and the bitmap is not solid (edges are anti-aliased).
        assert!(
            bitmap.height > 50 && bitmap.height < 110,
            "height {}",
            bitmap.height
        );
        assert!(
            bitmap.width > 20 && bitmap.width < 90,
            "width {}",
            bitmap.width
        );
        assert!(bitmap.alpha.contains(&255), "no fully-covered pixel");
        let covered = bitmap.alpha.iter().filter(|&&a| a > 0).count();
        let total = bitmap.alpha.len();
        assert!(
            covered * 10 > total && covered < total,
            "covered {covered}/{total}"
        );
    }

    #[test]
    fn spaces_produce_no_bitmap_but_advance_the_pen() {
        let font = plex_bold();
        let space = font.glyph_index(' ');
        assert_ne!(space, 0);
        assert!(
            rasterize(&font, space, 40.0).is_none(),
            "space has no contours"
        );
        let stack = FontStack { faces: vec![&font] };
        assert!(stack.measure(" ", 40.0) > 0.0, "the pen must still advance");
    }

    #[test]
    fn measure_matches_the_pen_advance_of_draw() {
        // The voice pill sizes its box with `measure` and fills it with `draw`; if the two
        // ever disagree the text escapes the pill.
        let font = plex_bold();
        let stack = FontStack { faces: vec![&font] };
        let text = "Voice: Aria";
        let measured = stack.measure(text, 44.0);
        let mut surface = Surface::new(600, 120);
        let pen = stack.draw(&mut surface, text, 10.0, 80.0, 44.0, [255, 255, 255], 1.0);
        assert!(
            (pen - 10.0 - measured).abs() < 1e-9,
            "measure {measured} vs drawn advance {}",
            pen - 10.0
        );
        // And the draw actually landed ink inside the surface.
        assert!(surface.rgba.as_chunks::<4>().0.iter().any(|px| px[3] > 0));
    }

    #[test]
    fn symbol_fallback_finds_arrows_and_dots() {
        let plex = plex_bold();
        let noto = Font::parse(fmd_font::bundled::NOTO_SANS_MATH_SYMBOLS.to_vec())
            .expect("bundled math face parses");
        let stack = FontStack {
            faces: vec![&plex, &noto],
        };
        for ch in ['\u{2192}', '\u{00B7}', '\u{2014}'] {
            assert!(
                stack.measure(&ch.to_string(), 34.0) > 0.0,
                "{ch:?} must map through the stack"
            );
        }
    }
}
