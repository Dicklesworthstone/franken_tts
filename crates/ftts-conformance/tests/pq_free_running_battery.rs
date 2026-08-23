//! ProductionQuality (Contract B), family 2: free-running objective metrics under the
//! production sampler.
//!
//! Bead `frankentts-v-prod-harness-t96`. This battery renders a small eval text list with
//! a real preset voice through the library path `ftts say` uses, then scores the audio:
//!
//! * **WER** against the input text, via an external pinned ASR scorer (`fw` /
//!   franken_whisper) invoked as a subprocess — never linked into `ftts`, exactly as the
//!   plan's automation boundary requires;
//! * **structural word errors** from the same alignment: skipped words (deletions) and
//!   immediate echo repeats;
//! * **stop-token behavior**: whether generation ended in EOS or ran into the frame cap
//!   (a route that never stops anywhere is the babbling signature);
//! * **long-form drift**: late-minus-early RMS decline per utterance, the quantity the
//!   `longform_drift` non-inferiority family pairs on;
//! * **prosody proxy**: duration-per-word;
//! * **speaker-identity diagnostic** (SECONDARY by doctrine): cosine between the
//!   conditioning x-vector and the speaker encoder's embedding of the rendered audio.
//!
//! # What is asserted, and what deliberately is not
//!
//! Assertions here are **screening vetoes** for egregious breakage, not quality gates:
//! every utterance produced finite audible audio; generation stopped somewhere without
//! hitting the cap; and, when the scorer ran, mean WER stayed below 0.7 with at least
//! one non-empty transcript. Quality judgments between two routes belong to the paired,
//! seeded comparisons the listening protocol consumes (`scripts/listening/margins.toml`);
//! inventing an absolute WER threshold here would counterfeit that machinery.
//!
//! Model-gated; scorer absence skips only the transcription families, honestly. When
//! `FTTS_PQ_REPORT` names a path, results land in `<path>.free_running.json`.

use std::path::{Path, PathBuf};

use ftts_cli::preset_voice_path;
use ftts_cli::synth::{
    LoadedModel, ModelBundle, ReferenceCleanup, SynthesizedAudio, read_speaker_vector,
    speaker_from_reference_pcm, synthesize,
};
use ftts_conformance::production::{
    DriftStats, WordEditStats, duration_per_word_ms, immediate_repetitions, longform_drift,
    normalize_words, word_edit_stats,
};
use ftts_conformance::report::{Outcome, Receipt};
use ftts_core::audio::{SAMPLE_RATE_HZ, SAMPLES_PER_FRAME, WavWriter};
use ftts_core::{CancellationToken, SynthesisRequest, TtsEngine};

const CONTRACT: &str = "ProductionQuality/free_running";
const TEST: &str = "production_quality_free_running_objective_battery";
/// Real speech x-vector; the synthetic-tone conformance voice babbles by design and
/// would make every word-level metric meaningless.
const VOICE: &str = "matt";
/// Screening veto: half the words wrong on average is not intelligible speech.
const MEAN_WER_VETO: f64 = 0.7;

struct TextCase {
    name: &'static str,
    axis: &'static str,
    text: &'static str,
}

const TEXTS: [TextCase; 7] = [
    TextCase {
        name: "plain_fox",
        axis: "plain",
        text: "The quick brown fox jumps over the lazy dog.",
    },
    TextCase {
        name: "sibilant_shells",
        axis: "sibilants",
        text: "She sells sea shells by the sea shore.",
    },
    TextCase {
        name: "numbers_1963",
        axis: "numbers",
        text: "In 1963, forty-two volunteers joined the study.",
    },
    TextCase {
        name: "plosive_piper",
        axis: "plosives",
        text: "Peter Piper picked a peck of pickled peppers.",
    },
    TextCase {
        name: "repeat_risk_woodchuck",
        axis: "repeats",
        text: "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    },
    TextCase {
        name: "plain_rain_spain",
        axis: "plain",
        text: "The rain in Spain falls mainly on the plain.",
    },
    TextCase {
        name: "longform_rainbow",
        axis: "long_form",
        text: "When the sunlight strikes raindrops in the air, they act as a prism and form a \
               rainbow. The rainbow is a division of white light into many beautiful colors.",
    },
];

fn bundle_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/truth-pack/snapshots/hf")
}

/// Text-proportional admission cap: room for the words plus prompt/settling overhead,
/// clamped so one runaway text cannot dominate the battery.
fn frame_cap_for(text: &str) -> u64 {
    let words = normalize_words(text).len().max(1) as u64;
    (words * 6 + 20).clamp(40, 150)
}

fn skip(reason: &str) {
    Receipt::new(TEST, Outcome::Skipped)
        .contract(CONTRACT)
        .seam("production.free_running")
        .reason(reason)
        .emit();
}

/// Locates the external ASR scorer: `$FTTS_PQ_FW_BIN` first, then `$PATH`.
///
/// `None` means the transcription families skip honestly; synthesis-side families run
/// regardless.
fn locate_fw() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FTTS_PQ_FW_BIN") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var)
        .map(|dir| dir.join("fw"))
        .find(|candidate| candidate.is_file())
}

/// Transcribes one WAV through the external scorer, returning its text.
///
/// The scorer's exact JSON layout may evolve, so the parser looks for a `text`-like
/// string field anywhere in the document rather than pinning one address.
fn transcribe(fw: &Path, wav: &Path) -> Result<String, String> {
    let output = std::process::Command::new(fw)
        .args(["transcribe", "--input"])
        .arg(wav)
        .arg("--json")
        .output()
        .map_err(|error| format!("cannot spawn {}: {error}", fw.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} exited {}: {}",
            fw.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(300)
                .collect::<String>()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).map_err(|error| {
        format!(
            "scorer output is not JSON ({error}): {:#?}",
            stdout.chars().take(300).collect::<String>()
        )
    })?;
    find_transcript(&value)
        .ok_or_else(|| format!("no transcript text field in scorer output: {value:#?}"))
}

/// Depth-first search for the first string under a `text`/`transcript`-style key.
fn find_transcript(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Object(map) => map
            .iter()
            .filter(|(key, _)| matches!(key.as_str(), "text" | "transcript" | "transcription"))
            .find_map(|(_, value)| value.as_str().map(str::to_owned))
            .or_else(|| map.values().find_map(find_transcript)),
        serde_json::Value::Array(items) => items.iter().find_map(find_transcript),
        _ => None,
    }
}

fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }
    Some(dot / (norm_a * norm_b))
}

/// One rendered utterance and everything measured about it before transcription.
struct Rendered {
    name: &'static str,
    axis: &'static str,
    audio: SynthesizedAudio,
    hit_cap: bool,
    drift: Option<DriftStats>,
    ms_per_word: Option<f64>,
    identity_cosine: Option<f64>,
    wav_path: PathBuf,
}

/// One text's transcription-side measurements, when a transcript exists.
struct ScoredWords {
    case_index: usize,
    wer: Option<f64>,
    stats: Option<WordEditStats>,
    echo_repeats: usize,
    transcript_words: usize,
}

#[test]
fn production_quality_free_running_objective_battery() {
    let bundle = match ModelBundle::resolve(&bundle_root()) {
        Ok(bundle) => bundle,
        Err(error) => {
            skip(&format!("checkpoint bundle incomplete: {error}"));
            return;
        }
    };
    let model = match LoadedModel::load(&bundle) {
        Ok(model) => model,
        Err(error) => {
            skip(&format!("checkpoint unusable: {error}"));
            return;
        }
    };
    let voice_path = match preset_voice_path(VOICE) {
        Ok(path) => path,
        Err(error) => {
            skip(&format!("preset voice unusable: {error}"));
            return;
        }
    };
    let speaker = match read_speaker_vector(&voice_path) {
        Ok(speaker) => speaker,
        Err(error) => {
            skip(&format!("cannot read preset voice vector: {error}"));
            return;
        }
    };

    // Per-run directory: unique by pid+millis and never deleted by us (deletion needs
    // explicit operator approval), so reruns never collide and old runs stay inspectable.
    let unique = format!(
        "ftts-pq-free-running-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before the epoch")
            .as_millis()
    );
    let work_dir = std::env::temp_dir().join(unique);
    if let Err(error) = std::fs::create_dir_all(&work_dir) {
        skip(&format!(
            "cannot create work dir {}: {error}",
            work_dir.display()
        ));
        return;
    }

    // --- render every text ----------------------------------------------------------------------
    let mut rendered: Vec<Rendered> = Vec::new();
    for (index, case) in TEXTS.iter().enumerate() {
        let cap = frame_cap_for(case.text);
        let engine = match TtsEngine::new(ftts_core::EngineConfig {
            synthesis_stage_budget: std::time::Duration::from_secs(3_600),
            admission: ftts_core::admission::AdmissionPolicy {
                max_new_tokens: cap,
                ..ftts_core::admission::AdmissionPolicy::default()
            },
            ..ftts_core::EngineConfig::default()
        }) {
            Ok(engine) => engine,
            Err(error) => {
                skip(&format!("engine construction failed: {error}"));
                return;
            }
        };
        let cancellation = CancellationToken::new();
        let observer = |_event: ftts_core::SynthesisEvent| {};
        let request = SynthesisRequest::new(case.text);

        let audio = match synthesize(
            &model,
            &engine,
            &request,
            &speaker,
            1_000 + index as u64,
            &cancellation,
            &observer,
            4,
            None,
            None,
        ) {
            Ok(audio) => audio,
            Err(error) => {
                skip(&format!("synthesis refused for {}: {error}", case.name));
                return;
            }
        };

        let drift = longform_drift(&audio.pcm, SAMPLES_PER_FRAME);
        let ms_per_word = duration_per_word_ms(
            audio.pcm.len(),
            SAMPLE_RATE_HZ,
            normalize_words(case.text).len(),
        );
        // Secondary identity diagnostic: does the rendered audio embed near the voice we
        // conditioned on? Cleanup stays off so the measurement sees OUR audio verbatim.
        let identity_cosine =
            speaker_from_reference_pcm(&bundle, audio.pcm.clone(), ReferenceCleanup::default())
                .ok()
                .and_then(|embedding| cosine(&embedding, &speaker));

        let wav_path = work_dir.join(format!("{}.wav", case.name));
        let write_result = (|| -> std::io::Result<()> {
            let file = std::fs::File::create(&wav_path)?;
            let mut writer = WavWriter::new(file, SAMPLE_RATE_HZ)?;
            for packet in audio.pcm.chunks(SAMPLES_PER_FRAME * 4) {
                writer.write_samples(packet)?;
            }
            writer.finish()?;
            Ok(())
        })();
        if let Err(error) = write_result {
            skip(&format!("cannot write {}: {error}", wav_path.display()));
            return;
        }

        rendered.push(Rendered {
            name: case.name,
            axis: case.axis,
            hit_cap: audio.frames >= cap,
            drift,
            ms_per_word,
            identity_cosine,
            audio,
            wav_path,
        });
    }

    // --- synthesis-side assertions --------------------------------------------------------------
    for item in &rendered {
        assert!(item.audio.frames > 0, "{} produced no frames", item.name);
        assert_eq!(
            item.audio.pcm.len(),
            item.audio.frames as usize * SAMPLES_PER_FRAME,
            "{} frame/sample disagreement",
            item.name
        );
        assert!(
            item.audio.pcm.iter().all(|sample| sample.is_finite()),
            "{} produced non-finite samples",
            item.name
        );
    }

    // --- transcription ---------------------------------------------------------------------------
    let fw = locate_fw();
    let mut transcripts: Vec<(usize, String)> = Vec::new();
    let mut asr_failures: Vec<String> = Vec::new();
    if let Some(fw) = &fw {
        for (index, item) in rendered.iter().enumerate() {
            match transcribe(fw, &item.wav_path) {
                Ok(text) => transcripts.push((index, text)),
                Err(reason) => asr_failures.push(format!("{}: {reason}", item.name)),
            }
        }
    }

    // --- word-level scoring ----------------------------------------------------------------------
    let mut scored: Vec<ScoredWords> = Vec::new();
    for (index, case) in TEXTS.iter().enumerate() {
        let scored_case = match transcripts.iter().find(|(found, _)| *found == index) {
            Some((_, transcript)) => {
                let reference = normalize_words(case.text);
                let hypothesis = normalize_words(transcript);
                let stats = word_edit_stats(&reference, &hypothesis);
                ScoredWords {
                    case_index: index,
                    wer: stats.wer(reference.len()),
                    echo_repeats: immediate_repetitions(&hypothesis),
                    transcript_words: hypothesis.len(),
                    stats: Some(stats),
                }
            }
            None => ScoredWords {
                case_index: index,
                wer: None,
                stats: None,
                echo_repeats: 0,
                transcript_words: 0,
            },
        };
        scored.push(scored_case);
    }

    // --- screening vetoes (egregious breakage only — see the module docs) ------------------------
    let wers: Vec<f64> = scored.iter().filter_map(|item| item.wer).collect();
    if !wers.is_empty() {
        let mean_wer = wers.iter().sum::<f64>() / wers.len() as f64;
        assert!(
            mean_wer <= MEAN_WER_VETO,
            "mean WER {mean_wer:.3} exceeds the {MEAN_WER_VETO} screening veto: the route is not \
             producing intelligible speech — investigate before any lever work"
        );
        assert!(
            scored.iter().any(|item| item.transcript_words > 0),
            "every transcript came back empty while every input had words"
        );
    }
    assert!(
        !rendered.iter().all(|item| item.hit_cap),
        "generation hit the frame cap on every text: nothing stops — the babbling signature"
    );

    // --- per-text receipts -----------------------------------------------------------------------
    for (index, item) in rendered.iter().enumerate() {
        let words = scored.iter().find(|scored| scored.case_index == index);
        Receipt::new(TEST, Outcome::Passed)
            .contract(CONTRACT)
            .seam(format!("production.free_running.{}", item.name))
            .reason(format!(
                "{} [{}]: {} frames (cap-hit: {}), {:.1} ms/word, drift {:?}, identity cosine {}, wer {:?}",
                item.name,
                item.axis,
                item.audio.frames,
                item.hit_cap,
                item.ms_per_word.unwrap_or(f64::NAN),
                item.drift.map(|drift| drift.decline_db),
                item.identity_cosine.map_or("-".to_owned(), |value| format!("{value:.4}")),
                words.and_then(|words| words.wer),
            ))
            .detail(serde_json::json!({
                "axis": item.axis,
                "frames": item.audio.frames,
                "hit_cap": item.hit_cap,
                "drift": item.drift.map(|drift| serde_json::json!({
                    "early_rms_db": drift.early_rms_db,
                    "late_rms_db": drift.late_rms_db,
                    "decline_db": drift.decline_db,
                })),
                "ms_per_word": item.ms_per_word,
                "identity_cosine": item.identity_cosine,
                "word_stats": words.and_then(|words| words.stats).map(|stats| serde_json::json!({
                    "distance": stats.distance,
                    "substitutions": stats.substitutions,
                    "deletions_skipped_words": stats.deletions,
                    "insertions": stats.insertions,
                })),
                "echo_repeats": words.map(|words| words.echo_repeats),
            }))
            .emit();
    }

    // --- aggregate receipt + scorecard -----------------------------------------------------------
    let mean_identity = {
        let values: Vec<f64> = rendered
            .iter()
            .filter_map(|item| item.identity_cosine)
            .collect();
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    let fw_display = fw.as_ref().map_or_else(
        || "absent (word families skipped)".to_owned(),
        |fw| format!("{}", fw.display()),
    );
    let mean_wer_text = if wers.is_empty() {
        "n/a".to_owned()
    } else {
        format!("{:.3}", wers.iter().sum::<f64>() / wers.len() as f64)
    };
    Receipt::new(TEST, Outcome::Passed)
        .contract(CONTRACT)
        .seam("production.free_running")
        .reason(format!(
            "{} texts, voice {VOICE}; scorer {fw_display}; mean WER {mean_wer_text}; mean identity \
             cosine {}; cap-hits {}; asr failures {}",
            rendered.len(),
            mean_identity.map_or("n/a".to_owned(), |value| format!("{value:.4}")),
            rendered.iter().filter(|item| item.hit_cap).count(),
            asr_failures.len(),
        ))
        .detail(serde_json::json!({
            "texts": rendered.iter().map(|item| item.name).collect::<Vec<_>>(),
            "asr_failures": asr_failures,
            "mean_identity_cosine": mean_identity,
        }))
        .emit();

    write_scorecard(&rendered, &scored, mean_identity);
}

fn write_scorecard(rendered: &[Rendered], scored: &[ScoredWords], mean_identity: Option<f64>) {
    let Some(base) = std::env::var_os("FTTS_PQ_REPORT") else {
        return;
    };
    let mut path = PathBuf::from(base);
    path.set_file_name(format!(
        "{}.free_running.json",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("pq_scorecard")
    ));
    let wers: Vec<f64> = scored.iter().filter_map(|item| item.wer).collect();
    let scorecard = serde_json::json!({
        "schema_version": 1,
        "generator": "pq_free_running_battery",
        "bead": "frankentts-v-prod-harness-t96",
        "voice": VOICE,
        "free_running_objective": {
            "texts": rendered.iter().enumerate().map(|(index, item)| serde_json::json!({
                "name": item.name,
                "axis": item.axis,
                "frames": item.audio.frames,
                "hit_cap": item.hit_cap,
                "drift_decline_db": item.drift.map(|drift| drift.decline_db),
                "ms_per_word": item.ms_per_word,
                "identity_cosine": item.identity_cosine,
                "wer": scored.iter().find(|scored| scored.case_index == index).and_then(|scored| scored.wer),
                "skipped_words": scored.iter().find(|scored| scored.case_index == index).and_then(|scored| scored.stats).map(|stats| stats.deletions),
                "inserted_words": scored.iter().find(|scored| scored.case_index == index).and_then(|scored| scored.stats).map(|stats| stats.insertions),
                "echo_repeats": scored.iter().find(|scored| scored.case_index == index).map(|scored| scored.echo_repeats),
            })).collect::<Vec<_>>(),
            "mean_wer": if wers.is_empty() { serde_json::Value::Null } else { serde_json::json!(wers.iter().sum::<f64>() / wers.len() as f64) },
            "mean_identity_cosine": mean_identity,
        },
    });
    if let Ok(json) = serde_json::to_string_pretty(&scorecard) {
        let _ = std::fs::write(&path, json + "\n");
    }
}
