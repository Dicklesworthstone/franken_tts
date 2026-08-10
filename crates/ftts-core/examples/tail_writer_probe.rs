//! Feeds a real WAV through the trimming `WavWriter` exactly as the CLI does, and reports what
//! came out. Isolates "the detector is wrong" from "the writer never asked it".

use std::io::Cursor;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: tail_writer_probe <wav>");
    let bytes = std::fs::read(&path).expect("readable wav");
    let pcm: Vec<f32> = bytes[44..]
        .chunks_exact(2)
        .map(|pair| f32::from(i16::from_le_bytes([pair[0], pair[1]])) / 32_767.0)
        .collect();

    let offline = ftts_core::audio::trailing_noise_samples(&pcm, 24_000);
    let level = ftts_core::audio::speech_level(&pcm, 24_000);
    println!("input samples      : {}", pcm.len());
    println!("offline detector   : {offline} samples");
    println!("utterance level    : {level:.5}");

    // Exactly the CLI's packetization.
    let mut writer =
        ftts_core::audio::WavWriter::new_trimming_tail(Cursor::new(Vec::new()), 24_000)
            .expect("header");
    for packet in pcm.chunks(1_920) {
        writer.write_samples(packet).expect("write");
    }
    let out = writer.finish().expect("finish").into_inner();
    let written = (out.len() - 44) / 2;
    println!(
        "writer wrote       : {written} samples (trimmed {})",
        pcm.len() - written
    );

    // What the detector sees when handed only the tail, which is the writer's situation.
    let hold = 24_000 * 250 / 1000;
    let tail = &pcm[pcm.len().saturating_sub(hold)..];
    println!(
        "tail-only relative : {} samples (tail len {})",
        ftts_core::audio::trailing_noise_samples_relative_to(tail, 24_000, level),
        tail.len()
    );
}
