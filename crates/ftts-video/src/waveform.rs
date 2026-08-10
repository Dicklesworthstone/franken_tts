//! Per-frame waveform visualization: a centered filled envelope ("cline")
//! of the audio window each frame covers, gradient-tinted green→cyan.

const TOP_COLOR: [f32; 3] = [0x5A as f32, 0xFF as f32, 0xC4 as f32];
const BOTTOM_COLOR: [f32; 3] = [0x49 as f32, 0xE6 as f32, 0xFF as f32];

/// Draw the waveform for frame `index` directly onto an RGB24 frame.
///
/// The window is the audio the frame spans (1/fps seconds). Amplitudes are
/// cube-root compressed with a 3× drive so quiet speech still registers —
/// the same shaping the launch videos used.
#[allow(clippy::too_many_arguments)]
pub fn draw(
    rgb: &mut [u8],
    width: usize,
    height: usize,
    samples: &[f32],
    sample_rate: u32,
    fps: u32,
    frame_index: usize,
    center_y: usize,
    half_height: f64,
) {
    let window = f64::from(sample_rate) / f64::from(fps);
    let start = frame_index as f64 * window;
    let opacity = 0.6f32;

    for x in 0..width {
        let position = start + (x as f64 / width as f64) * window;
        let base = position.floor() as usize;
        let amplitude = if base + 1 < samples.len() {
            let fraction = (position - base as f64) as f32;
            let sample = samples[base] * (1.0 - fraction) + samples[base + 1] * fraction;
            f64::from((sample * 3.0).abs().min(1.0)).cbrt()
        } else {
            0.0
        };
        // A hairline at zero amplitude keeps the lane alive through silence.
        let extent = (amplitude * half_height).max(1.5);
        let y_top = (center_y as f64 - extent).max(0.0) as usize;
        let y_bottom = ((center_y as f64 + extent) as usize).min(height - 1);
        for y in y_top..=y_bottom {
            let mix = ((y - y_top) as f32 / ((y_bottom - y_top).max(1)) as f32).clamp(0.0, 1.0);
            let at = (y * width + x) * 3;
            for c in 0..3 {
                let wave = TOP_COLOR[c] * (1.0 - mix) + BOTTOM_COLOR[c] * mix;
                let dst = f32::from(rgb[at + c]);
                rgb[at + c] = (dst * (1.0 - opacity) + wave * opacity)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
}
