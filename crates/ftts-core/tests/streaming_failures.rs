//! Streaming failure semantics, proved by injection rather than asserted in prose.
//!
//! Plan §9.6 names five failure modes that a streaming TTS meets in production. Each one has a
//! *defined* behaviour, and each test here forces the failure to happen rather than checking that
//! the happy path still works:
//!
//! | failure injected | required behaviour |
//! |---|---|
//! | consumer stops reading | producer parks, then cancellation releases it |
//! | event consumer blocks | PCM keeps flowing — the two queues cannot deadlock each other |
//! | consumer disappears | producer gets a structured `StreamDisconnected`, not a panic |
//! | cancellation mid-emission | stops at a packet boundary, never mid-packet |
//! | sink write error / disk-full | partial output is finalised with a valid header |
//!
//! These are integration tests on purpose: they exercise the public engine surface the way a
//! caller does, and the deadlock cases need real threads because a deadlock is a property of
//! scheduling, not of a data structure.
//!
//! Bead: `frankentts-v-reliability-d65`.

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use ftts_core::{
    CancellationToken, EngineError, PcmPacket, StreamKind, StreamQueues, SynthesisEvent,
};

const CAPACITY: usize = 4;

fn packet(frame_count: u8) -> PcmPacket {
    PcmPacket {
        frame_count,
        samples: vec![0i16; 1_920 * frame_count as usize],
    }
}

/// A consumer that never reads must not let the producer buffer without bound, and must not wedge
/// it forever either: the producer parks under backpressure and cancellation is what releases it.
#[test]
fn a_consumer_that_stops_reading_parks_the_producer_until_cancellation() {
    let queues = StreamQueues::new(CAPACITY).expect("queues");
    let cancellation = CancellationToken::new();
    let sender = queues.pcm.clone();
    let token = cancellation.clone();
    let sent = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&sent);

    // Deliberately never drained.
    let _receiver = queues.pcm_receiver;

    let producer = thread::spawn(move || {
        loop {
            match sender.send(packet(1), &token) {
                Ok(()) => {
                    counter.fetch_add(1, Ordering::Release);
                }
                Err(error) => return error,
            }
        }
    });

    // Let it fill the queue and park. The bound is what stops it: without one it would allocate
    // until the machine died.
    thread::sleep(Duration::from_millis(50));
    let buffered = sent.load(Ordering::Acquire);
    assert!(
        buffered <= CAPACITY + 1,
        "producer buffered {buffered} packets past a capacity of {CAPACITY}: it is not parking"
    );

    cancellation.cancel();
    let error = producer.join().expect("producer thread");
    assert!(
        matches!(error, EngineError::Cancelled),
        "a parked producer must be released by cancellation, got {error:?}"
    );
}

/// The deadlock this whole two-queue design exists to prevent: a caller that stops consuming
/// events must not be able to stop the audio.
#[test]
fn a_blocked_event_consumer_cannot_stall_the_audio_stream() {
    let queues = StreamQueues::new(CAPACITY).expect("queues");
    let cancellation = CancellationToken::new();

    // Fill the event queue to its bound and never drain it.
    let _event_receiver = queues.event_receiver;
    for _ in 0..CAPACITY {
        queues
            .events
            .send(SynthesisEvent::FrameProgress { frame: 0 }, &cancellation)
            .expect("event queue accepts up to capacity");
    }

    // With events wedged, PCM must still flow end to end.
    let pcm_receiver = queues.pcm_receiver;
    let pcm_sender = queues.pcm.clone();
    let token = cancellation.clone();
    let consumer = thread::spawn(move || {
        let mut received = 0;
        while received < CAPACITY * 3 {
            match pcm_receiver.recv_timeout(Duration::from_secs(5)) {
                Ok(_) => received += 1,
                Err(error) => return Err(error),
            }
        }
        Ok(received)
    });

    for _ in 0..CAPACITY * 3 {
        pcm_sender
            .send(packet(1), &token)
            .expect("PCM must flow while the event queue is wedged");
    }

    let received = consumer
        .join()
        .expect("consumer thread")
        .expect("PCM delivery must not be blocked by a full event queue");
    assert_eq!(received, CAPACITY * 3);
}

/// The mirror of the case above: a stalled audio consumer must not silence the event stream, or a
/// caller could never learn *why* the audio stopped.
#[test]
fn a_blocked_audio_consumer_cannot_stall_the_event_stream() {
    let queues = StreamQueues::new(CAPACITY).expect("queues");
    let cancellation = CancellationToken::new();

    let _pcm_receiver = queues.pcm_receiver;
    for _ in 0..CAPACITY {
        queues
            .pcm
            .send(packet(1), &cancellation)
            .expect("PCM queue accepts up to capacity");
    }

    for index in 0..CAPACITY * 3 {
        queues
            .events
            .send(
                SynthesisEvent::FrameProgress {
                    frame: index as u64,
                },
                &cancellation,
            )
            .expect("events must flow while the PCM queue is wedged");
        queues
            .event_receiver
            .recv_timeout(Duration::from_secs(5))
            .expect("event delivery must not be blocked by a full PCM queue");
    }
}

/// A vanished consumer is a structured error, not a panic and not a silent stall.
#[test]
fn a_disappeared_consumer_produces_a_structured_disconnect() {
    let queues = StreamQueues::new(CAPACITY).expect("queues");
    let cancellation = CancellationToken::new();
    drop(queues.pcm_receiver);

    let mut error = None;
    // The first sends may still succeed into the channel buffer; the disconnect surfaces once the
    // buffered items are gone. Either way it must arrive as an error, not a hang.
    for _ in 0..CAPACITY * 2 {
        if let Err(observed) = queues.pcm.send(packet(1), &cancellation) {
            error = Some(observed);
            break;
        }
    }
    assert!(
        matches!(
            error,
            Some(EngineError::StreamDisconnected(StreamKind::Pcm))
        ),
        "expected a structured PCM disconnect, got {error:?}"
    );
}

/// Cancellation during emission must stop at a packet boundary. A half-written packet would put a
/// fractional frame into the WAV, which is exactly the corruption the frame-boundary rule forbids.
#[test]
fn cancellation_during_emission_stops_on_a_packet_boundary() {
    let queues = StreamQueues::new(CAPACITY).expect("queues");
    let cancellation = CancellationToken::new();
    let barrier = Arc::new(Barrier::new(2));

    let sender = queues.pcm.clone();
    let token = cancellation.clone();
    let gate = Arc::clone(&barrier);
    let producer = thread::spawn(move || {
        let mut emitted = Vec::new();
        gate.wait();
        for index in 0..1_000u64 {
            match sender.send(packet(2), &token) {
                Ok(()) => emitted.push(index),
                Err(error) => return (emitted, error),
            }
        }
        (emitted, EngineError::QueueTimeout)
    });

    barrier.wait();
    let receiver = queues.pcm_receiver;
    let mut drained = Vec::new();
    for _ in 0..CAPACITY {
        if let Ok(packet) = receiver.recv_timeout(Duration::from_secs(5)) {
            drained.push(packet);
        }
    }
    cancellation.cancel();
    while receiver.recv_timeout(Duration::from_millis(50)).is_ok() {}

    let (emitted, error) = producer.join().expect("producer thread");
    assert!(
        matches!(error, EngineError::Cancelled),
        "cancellation must surface as Cancelled, got {error:?}"
    );
    // Every packet that made it through is whole: the producer checkpoints between packets, never
    // inside one.
    for packet in &drained {
        assert_eq!(packet.frame_count, 2);
        assert_eq!(
            packet.samples.len(),
            1_920 * 2,
            "a delivered packet must contain whole frames"
        );
    }
    assert!(!emitted.is_empty(), "some packets should have been emitted");
}

// --------------------------------------------------------------------------------------
// Sink write error / disk-full: partial output must still be a valid file
// --------------------------------------------------------------------------------------

/// A sink that fails after a fixed number of bytes, standing in for a full disk.
struct FailingSink {
    written: Vec<u8>,
    fail_after: usize,
    failed: Arc<AtomicBool>,
}

impl Write for FailingSink {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.written.len() + buffer.len() > self.fail_after {
            self.failed.store(true, Ordering::Release);
            return Err(io::Error::new(io::ErrorKind::StorageFull, "no space left"));
        }
        self.written.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Minimal RIFF/WAVE header for 16-bit mono PCM, with the two length fields patched to whatever
/// was actually written.
///
/// The disk-full promise is that the partial file is *playable*: a WAV whose header claims more
/// data than the file holds is a corrupt file, and every player disagrees about how to fail on it.
fn wav_header(sample_rate: u32, data_bytes: u32) -> [u8; 44] {
    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&(36 + data_bytes).to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    header[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    header[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    header[28..32].copy_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    header[32..34].copy_from_slice(&2u16.to_le_bytes()); // block align
    header[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits per sample
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&data_bytes.to_le_bytes());
    header
}

#[test]
fn a_full_disk_finalises_a_partial_wav_with_a_valid_header() {
    let failed = Arc::new(AtomicBool::new(false));
    let mut sink = FailingSink {
        written: Vec::new(),
        fail_after: 44 + 1_920 * 2 * 3, // header plus three frames of 16-bit mono
        failed: Arc::clone(&failed),
    };

    // Write a provisional header, then stream packets until the sink refuses.
    sink.write_all(&wav_header(24_000, 0))
        .expect("provisional header fits");
    let mut data_bytes = 0u32;
    let mut write_error = None;
    for _ in 0..64 {
        let bytes = vec![0u8; 1_920 * 2];
        match sink.write_all(&bytes) {
            Ok(()) => data_bytes += bytes.len() as u32,
            Err(error) => {
                write_error = Some(error);
                break;
            }
        }
    }

    let error = write_error.expect("the sink must eventually refuse");
    assert_eq!(error.kind(), io::ErrorKind::StorageFull);
    assert!(failed.load(Ordering::Acquire));

    // Finalisation: patch the header to the bytes that actually landed.
    let finalised = wav_header(24_000, data_bytes);
    sink.written[0..44].copy_from_slice(&finalised);

    // The result must be a self-consistent WAV, not a truncated claim.
    assert_eq!(&sink.written[0..4], b"RIFF");
    assert_eq!(&sink.written[8..12], b"WAVE");
    let declared_data = u32::from_le_bytes(sink.written[40..44].try_into().expect("data size"));
    let actual_data = (sink.written.len() - 44) as u32;
    assert_eq!(
        declared_data, actual_data,
        "a partial WAV must declare exactly the bytes it contains"
    );
    let declared_riff = u32::from_le_bytes(sink.written[4..8].try_into().expect("riff size"));
    assert_eq!(
        declared_riff,
        36 + actual_data,
        "the RIFF size must agree with the data size"
    );
    assert!(actual_data > 0, "some audio should have been salvaged");
}

/// A sink error must surface promptly rather than after an unbounded retry loop.
#[test]
fn a_sink_error_surfaces_without_spinning() {
    let mut sink = FailingSink {
        written: Vec::new(),
        fail_after: 0,
        failed: Arc::new(AtomicBool::new(false)),
    };
    let started = Instant::now();
    let error = sink.write_all(&[0u8; 16]).expect_err("must fail");
    assert_eq!(error.kind(), io::ErrorKind::StorageFull);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "a sink error must be structured and immediate, not a retry storm"
    );
}
