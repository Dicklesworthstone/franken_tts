//! Streaming C ABI contract (bead frankentts-omad), against the real model.
//!
//! Model-gated: without a complete model directory each test reports the skip and
//! passes. Proven here:
//! 1. packet concatenation from `ftts_synthesize_streaming` is BIT-IDENTICAL to the
//!    whole buffer from `ftts_synthesize` at the same text/speaker/seed;
//! 2. `frame_index` counts delivered frames exactly (cumulative, packet-aligned);
//! 3. a nonzero callback return cancels: delivery stops promptly and the call
//!    returns FTTS_SYNTH_CANCELLED (6) with fewer packets than the full run.
//!
//! The callback-panic case is deliberately NOT tested: the callback is a C function
//! by contract (no unwinding exists on that side), and Rust's extern "C" aborts on
//! unwind by construction — `guarded` covers the Rust side of the boundary.

use std::ffi::{CString, c_void};
use std::path::PathBuf;

use ftts_ffi::{
    FTTS_SYNTH_CANCELLED, FttsEngine, SPEAKER_WIDTH, ftts_engine_close, ftts_engine_open,
    ftts_pcm_free, ftts_preset_vector, ftts_synthesize, ftts_synthesize_streaming,
};

fn model_dir() -> Option<PathBuf> {
    let root = std::env::var("FTTS_MODEL_DIR").map_or_else(
        |_| {
            #[allow(deprecated)]
            std::env::home_dir().map(|home| home.join(".cache/franken_tts/model"))
        },
        |dir| Some(PathBuf::from(dir)),
    )?;
    for required in [
        "vocab.json",
        "merges.txt",
        "speech_tokenizer/model.safetensors",
    ] {
        if !root.join(required).is_file() {
            return None;
        }
    }
    root.join("qwen3-tts-12hz-0.6b-base.fttsq")
        .is_file()
        .then_some(root)
}

const TEXT: &str = "The streaming ABI must deliver the same bytes as the whole buffer.";
const SEED: u64 = 4711;

struct Collector {
    samples: Vec<f32>,
    packets: usize,
    frame_indices: Vec<u64>,
    cancel_after_packets: Option<usize>,
}

/// The C-shaped callback under test. `ctx` is a `*mut Collector` owned by the caller.
#[allow(unsafe_code)] // audited test shim: this IS the C boundary under test
unsafe extern "C" fn collect(
    ctx: *mut c_void,
    samples: *const f32,
    len: usize,
    frame_index: u64,
) -> i32 {
    // SAFETY: `ctx` is the Collector the test passed; `samples` points at `len` floats
    // for the duration of this call, per the header contract.
    let collector = unsafe { &mut *ctx.cast::<Collector>() };
    let packet = unsafe { std::slice::from_raw_parts(samples, len) };
    collector.samples.extend_from_slice(packet);
    collector.frame_indices.push(frame_index);
    collector.packets += 1;
    match collector.cancel_after_packets {
        Some(limit) if collector.packets >= limit => 1,
        _ => 0,
    }
}

fn open_engine(root: &std::path::Path) -> *mut FttsEngine {
    let dir = CString::new(root.to_str().expect("utf-8 model path")).expect("no NUL");
    // SAFETY: dir is NUL-terminated UTF-8 per construction.
    #[allow(unsafe_code)]
    let engine = unsafe { ftts_engine_open(dir.as_ptr()) };
    assert!(!engine.is_null(), "engine opens");
    engine
}

fn preset(engine_dummy: ()) -> Vec<f32> {
    let _ = engine_dummy;
    let mut speaker = vec![0.0_f32; SPEAKER_WIDTH];
    let name = CString::new("matt").expect("no NUL");
    // SAFETY: name is NUL-terminated; the buffer holds SPEAKER_WIDTH floats.
    #[allow(unsafe_code)]
    let status = unsafe { ftts_preset_vector(name.as_ptr(), speaker.as_mut_ptr()) };
    assert_eq!(status, 0, "preset vector resolves");
    speaker
}

#[test]
fn streamed_packets_concatenate_to_the_whole_buffer_bit_for_bit() {
    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"ffi_stream_identity\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let engine = open_engine(&root);
    let speaker = preset(());
    let text = CString::new(TEXT).expect("no NUL");

    // Whole-buffer reference.
    let mut out_pcm: *mut f32 = std::ptr::null_mut();
    let mut out_len: usize = 0;
    // SAFETY: all pointers valid per construction; engine access is serialized (one thread).
    #[allow(unsafe_code)]
    let status = unsafe {
        ftts_synthesize(
            engine,
            text.as_ptr(),
            speaker.as_ptr(),
            speaker.len(),
            SEED,
            &raw mut out_pcm,
            &raw mut out_len,
        )
    };
    assert_eq!(status, 0, "whole-buffer synthesis succeeds");
    // SAFETY: (out_pcm, out_len) is the pair the call just returned.
    #[allow(unsafe_code)]
    let whole: Vec<f32> = unsafe { std::slice::from_raw_parts(out_pcm, out_len) }.to_vec();
    #[allow(unsafe_code)]
    unsafe {
        ftts_pcm_free(out_pcm, out_len)
    };

    // Streamed, 1-frame packets.
    let mut collector = Collector {
        samples: Vec::new(),
        packets: 0,
        frame_indices: Vec::new(),
        cancel_after_packets: None,
    };
    // SAFETY: collect + collector form the contract pair; engine serialized.
    #[allow(unsafe_code)]
    let status = unsafe {
        ftts_synthesize_streaming(
            engine,
            text.as_ptr(),
            speaker.as_ptr(),
            speaker.len(),
            SEED,
            1,
            collect,
            (&raw mut collector).cast(),
        )
    };
    assert_eq!(status, 0, "streaming synthesis succeeds");
    assert_eq!(
        collector.samples.len(),
        whole.len(),
        "streamed total diverges from the whole buffer"
    );
    let divergence = collector
        .samples
        .iter()
        .zip(whole.iter())
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert_eq!(divergence, None, "first divergent sample at {divergence:?}");
    // frame_index is the cumulative count BEFORE each packet: 0,1,2,... at 1-frame packets.
    let expected: Vec<u64> = (0..collector.frame_indices.len() as u64).collect();
    assert_eq!(collector.frame_indices, expected, "frame_index accounting");

    // SAFETY: engine came from ftts_engine_open, closed exactly once.
    #[allow(unsafe_code)]
    unsafe {
        ftts_engine_close(engine)
    };
    eprintln!(
        "receipt: {{\"test\":\"ffi_stream_identity\",\"outcome\":\"passed\",\"packets\":{},\"samples\":{}}}",
        collector.packets,
        collector.samples.len()
    );
}

#[test]
fn a_nonzero_callback_return_cancels_with_the_distinct_status() {
    let Some(root) = model_dir() else {
        eprintln!(
            "receipt: {{\"test\":\"ffi_stream_cancel\",\"outcome\":\"skipped\",\"reason\":\"model directory unavailable\"}}"
        );
        return;
    };
    let engine = open_engine(&root);
    let speaker = preset(());
    let text = CString::new(TEXT).expect("no NUL");

    let mut collector = Collector {
        samples: Vec::new(),
        packets: 0,
        frame_indices: Vec::new(),
        cancel_after_packets: Some(3),
    };
    // SAFETY: as above.
    #[allow(unsafe_code)]
    let status = unsafe {
        ftts_synthesize_streaming(
            engine,
            text.as_ptr(),
            speaker.as_ptr(),
            speaker.len(),
            SEED,
            1,
            collect,
            (&raw mut collector).cast(),
        )
    };
    assert_eq!(
        status, FTTS_SYNTH_CANCELLED,
        "callback cancellation reports the distinct status"
    );
    assert!(
        collector.packets >= 3 && collector.packets <= 8,
        "delivery stops within a few packets of the request, got {}",
        collector.packets
    );
    // SAFETY: engine came from ftts_engine_open, closed exactly once.
    #[allow(unsafe_code)]
    unsafe {
        ftts_engine_close(engine)
    };
    eprintln!(
        "receipt: {{\"test\":\"ffi_stream_cancel\",\"outcome\":\"passed\",\"packets_before_stop\":{}}}",
        collector.packets
    );
}
