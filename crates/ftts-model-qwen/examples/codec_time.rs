//! Times the f32 codec decoder alone, to attribute the post-int8 per-frame budget.
//!
//! With the talker + microdecoder GEMMs on the W8A8 route, the codec (still f32) is the suspected
//! per-frame bottleneck (plan §7.9's warning). This measures hydrate and offline decode wall time
//! for a synthetic `[frames, 16]` code stream — codec compute cost is content-independent, so
//! random in-vocabulary ids exercise the real cost without needing captured tokens.
//!
//! ```sh
//! cargo run --release --locked -p ftts-model-qwen --example codec_time -- \
//!     docs/truth-pack/snapshots/hf/speech_tokenizer/model.safetensors [frames]
//! ```

use ftts_model_qwen::checkpoint::CodecCheckpoint;
use std::path::Path;
use std::time::Instant;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: codec_time <speech_tokenizer/model.safetensors> [frames]");
    let frames: usize = std::env::args()
        .nth(2)
        .map_or(32, |value| value.parse().expect("frames must be a number"));

    let start = Instant::now();
    let codec = CodecCheckpoint::load(Path::new(&path)).expect("codec checkpoint");
    println!("hydrate: {:.2} s", start.elapsed().as_secs_f64());

    // Real codes when a divergence-run dump is supplied (5th arg, lines shaped
    // `frame NNN: c0 c1 ... c15`); otherwise deterministic in-vocabulary ids — decode cost is
    // content-independent, but PCM-divergence measurements should use the real distribution.
    let (codes, frames): (Vec<i32>, usize) = match std::env::args().nth(4) {
        Some(path) => {
            let mut codes = Vec::new();
            for line in std::fs::read_to_string(&path).expect("codes file").lines() {
                let Some((_, ids)) = line.split_once(':') else {
                    continue;
                };
                if ids.trim() == "EOS" {
                    break;
                }
                let row: Vec<i32> = ids
                    .split_whitespace()
                    .map(|id| id.parse().expect("code id"))
                    .collect();
                assert_eq!(row.len(), 16, "each frame line carries 16 codes");
                codes.extend(row);
            }
            let frames = codes.len() / 16;
            println!("using {frames} real frames from {path}");
            (codes, frames)
        }
        None => {
            let mut state = 0x5eed_u64;
            let codes = (0..frames * 16)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1);
                    ((state >> 40) % 2048) as i32
                })
                .collect();
            (codes, frames)
        }
    };

    let mut last_pcm = Vec::new();
    for round in 0..3 {
        let start = Instant::now();
        let pcm = codec.decode(&codes, frames).expect("decode");
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "round {round}: {frames} frames -> {} samples in {:.3} s  ({:.1} ms/frame, {:.2}x real time)",
            pcm.len(),
            elapsed,
            elapsed * 1e3 / frames as f64,
            (frames as f64 * 0.08) / elapsed,
        );
        last_pcm = pcm;
    }

    // Optional raw-f32 PCM dump, for two-process A/B (the int8 kill-switch is read once per
    // process, so f32-vs-q8 comparison runs the binary twice and diffs the dumps).
    if let Some(dump) = std::env::args().nth(3) {
        let bytes: Vec<u8> = last_pcm
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect();
        std::fs::write(&dump, bytes).expect("pcm dump");
        println!("pcm dumped: {dump} ({} samples)", last_pcm.len());
    }
}
