//! Reports what `trailing_noise_samples` would trim from real WAV files.
//!
//! The unit tests use synthetic shapes, which prove the rule but not that the rule matches the
//! artifact as the model actually produces it. This runs the same detector over real synthesis so
//! the thresholds can be checked against measured audio rather than assumed ones.
//!
//! Usage: `cargo run -p ftts-core --example tail_probe -- <dir-of-wavs>`

fn main() {
    let dir = std::env::args().nth(1).expect("usage: tail_probe <dir>");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("readable directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("wav"))
        .collect();
    paths.sort();

    for path in paths {
        let bytes = std::fs::read(&path).expect("readable wav");
        if bytes.len() < 44 {
            continue;
        }
        // 44-byte canonical header, then interleaved 16-bit little-endian samples.
        let pcm: Vec<f32> = bytes[44..]
            .chunks_exact(2)
            .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_767.0)
            .collect();
        let trim = ftts_core::audio::trailing_noise_samples(&pcm, 24_000);
        println!(
            "{:14} samples={:7} trim={:5} ({:5.1} ms)",
            path.file_name().unwrap_or_default().to_string_lossy(),
            pcm.len(),
            trim,
            trim as f32 / 24.0
        );
    }
}
