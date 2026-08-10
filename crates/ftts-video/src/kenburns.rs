//! Ken Burns camera: a slow zoom-and-drift over the background illustration,
//! resampled bilinearly straight into the output frame.

use crate::qoi::Image;

/// The camera path across the whole video: zoom 1.05 → 1.18 with a gentle
/// left drift and slight downward settle, matching the launch videos.
pub struct Camera<'a> {
    source: &'a Image,
}

impl<'a> Camera<'a> {
    pub fn new(source: &'a Image) -> Self {
        Self { source }
    }

    /// Render frame `index` of `total` into `rgb` (tight RGB24, y-down).
    pub fn render(
        &self,
        index: usize,
        total: usize,
        out_width: usize,
        out_height: usize,
        rgb: &mut [u8],
    ) {
        let t = if total > 1 {
            index as f64 / (total - 1) as f64
        } else {
            0.0
        };
        let src_w = f64::from(self.source.width);
        let src_h = f64::from(self.source.height);
        let zoom = 1.05 + 0.13 * t;

        // Viewport in source pixels, output aspect ratio.
        let view_w = src_w / zoom;
        let view_h = view_w * out_height as f64 / out_width as f64;
        let mut origin_x = (src_w - view_w) / 2.0 - 0.03 * src_w * t;
        let mut origin_y = (src_h - view_h) / 2.0 + 0.02 * src_h * t;
        origin_x = origin_x.clamp(0.0, (src_w - view_w).max(0.0));
        origin_y = origin_y.clamp(0.0, (src_h - view_h).max(0.0));

        let source = &self.source.rgba;
        let stride = self.source.width as usize * 4;
        let max_x = self.source.width as usize - 1;
        let max_y = self.source.height as usize - 1;

        // The horizontal sample positions depend only on the column, so their integer
        // base and 8-bit fraction are computed once per frame; the per-pixel work is
        // then pure u32 arithmetic. This loop is the whole cost of a frame, and the
        // scalar f64 version of it made phone-side export feel broken.
        let columns: Vec<(usize, usize, u32)> = (0..out_width)
            .map(|out_x| {
                let sample_x = origin_x + (out_x as f64 + 0.5) / out_width as f64 * view_w - 0.5;
                let x0f = sample_x.floor().clamp(0.0, max_x as f64);
                let x0 = x0f as usize;
                let fx = ((sample_x - x0f).clamp(0.0, 1.0) * 256.0).round() as u32;
                (x0 * 4, (x0 + 1).min(max_x) * 4, fx)
            })
            .collect();

        for out_y in 0..out_height {
            let sample_y = origin_y + (out_y as f64 + 0.5) / out_height as f64 * view_h - 0.5;
            let y0f = sample_y.floor().clamp(0.0, max_y as f64);
            let y0 = y0f as usize;
            let y1 = (y0 + 1).min(max_y);
            let fy = ((sample_y - y0f).clamp(0.0, 1.0) * 256.0).round() as u32;
            let top_row = &source[y0 * stride..y0 * stride + stride];
            let bottom_row = &source[y1 * stride..y1 * stride + stride];
            let row = &mut rgb[out_y * out_width * 3..(out_y + 1) * out_width * 3];
            for (pixel, &(x0, x1, fx)) in row.as_chunks_mut::<3>().0.iter_mut().zip(&columns) {
                for c in 0..3 {
                    // Weights sum to 256 per axis; the maximum accumulator value is
                    // 255 * 256 * 256 + rounding, comfortably inside u32.
                    let top =
                        u32::from(top_row[x0 + c]) * (256 - fx) + u32::from(top_row[x1 + c]) * fx;
                    let bottom = u32::from(bottom_row[x0 + c]) * (256 - fx)
                        + u32::from(bottom_row[x1 + c]) * fx;
                    let value = top * (256 - fy) + bottom * fy;
                    pixel[c] = ((value + (1 << 15)) >> 16) as u8;
                }
            }
        }
    }
}
