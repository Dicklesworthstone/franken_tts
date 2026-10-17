//! Opt-in `say` generation receipts. Separate v1 schema; robot NDJSON stays unchanged.
//! No text, reference transcript, absolute paths, or environment dump is persisted.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use ftts_artifacts::sha256::{digest, hex_digest, hex_digest_file};
use ftts_core::UtteranceStart;
use ftts_model_qwen::sampler::{self, SamplingMode};
use serde_json::{Value, json};

use crate::FttsError;
use crate::synth::{ModelBundle, SynthesizedAudio, VoiceConditioning};

pub(crate) struct GenerationTrace {
    path: PathBuf,
    receipt: Value,
}

fn hash_file(path: &Path) -> Result<String, FttsError> {
    hex_digest_file(path)
        .map_err(|error| FttsError::Input(format!("cannot hash generation artifact: {error}")))
}

fn voice_identity(voice: &VoiceConditioning) -> Value {
    let embedding: Vec<u8> = voice
        .embedding()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut identity = json!({"embedding_sha256": hex_digest(&digest(&embedding))});
    match voice {
        VoiceConditioning::XVector(_) => identity["route"] = json!("xvector"),
        VoiceConditioning::Icl {
            transcript,
            codec_codes,
            ..
        } => {
            let codes: Vec<u8> = codec_codes.iter().flat_map(|v| v.to_le_bytes()).collect();
            identity["route"] = json!("icl");
            identity["reference_transcript_sha256"] =
                json!(hex_digest(&digest(transcript.as_bytes())));
            identity["reference_codec_sha256"] = json!(hex_digest(&digest(&codes)));
            identity["reference_codec_frames"] = json!(codec_codes.len() / 16);
        }
    }
    identity
}

impl GenerationTrace {
    pub(crate) fn prepare(
        directory: &Path,
        run_id: &str,
        requested_seed: Option<u64>,
        bundle: &ModelBundle,
        voice_path: &Path,
        voice: &VoiceConditioning,
    ) -> Result<Self, FttsError> {
        fs::create_dir_all(directory)
            .map_err(|error| FttsError::Input(format!("cannot create trace directory: {error}")))?;
        let main = bundle.canonical_main.as_ref().unwrap_or(&bundle.main);
        let mut assets = json!({
            "talker_sha256": hash_file(main)?,
            "talker_format": if bundle.canonical_main.is_some() { "fttsq" } else { "safetensors" },
            "codec_sha256": hash_file(&bundle.codec)?,
            "voice_source_sha256": hash_file(voice_path)?,
        });
        for name in ["vocab.json", "merges.txt", "tokenizer_config.json"] {
            assets[name] = json!({"sha256": hash_file(&bundle.root.join(name))?});
        }
        let executable = std::env::current_exe()
            .map_err(|error| FttsError::Input(format!("cannot resolve executable: {error}")))?;
        Ok(Self {
            path: directory.join(format!("{run_id}.generation.json")),
            receipt: json!({
                "schema": "ftts.generation", "schema_version": 1, "run_id": run_id,
                "producer": {"version": env!("CARGO_PKG_VERSION"),
                    "executable_sha256": hash_file(&executable)?,
                    "arch": std::env::consts::ARCH, "os": std::env::consts::OS},
                "requested": {"seed": requested_seed},
                "assets": assets, "voice": voice_identity(voice),
                "execution": "inline", "state": "prepared",
                "claim": "generation provenance only; EOS does not prove text accuracy or voice acceptance"
            }),
        })
    }

    pub(crate) fn finish(
        mut self,
        audio: &SynthesizedAudio,
        samples: u64,
    ) -> Result<(), FttsError> {
        self.receipt["effective"] = effective(audio);
        self.receipt["state"] = json!(if audio.stop_reason.is_clean_completion() {
            "completed"
        } else {
            "truncated"
        });
        self.receipt["output"] = json!({"frames": audio.frames, "samples": samples,
            "sample_rate": ftts_core::audio::SAMPLE_RATE_HZ,
            "stop_reason": audio.stop_reason.as_str(), "frame_limit": audio.frame_limit});
        // A unique run-id file is never overwritten, including on an accidental replay.
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)
            .map_err(|error| {
                FttsError::Input(format!("cannot create generation receipt: {error}"))
            })?;
        serde_json::to_writer_pretty(&mut file, &self.receipt).map_err(|error| {
            FttsError::Generic(format!("cannot serialize generation receipt: {error}"))
        })?;
        file.write_all(b"\n")
            .and_then(|()| file.sync_all())
            .map_err(|error| FttsError::Input(format!("cannot finish generation receipt: {error}")))
    }
}

fn effective(audio: &SynthesizedAudio) -> Value {
    let normalization = audio.normalization.as_ref().map(|summary| {
        json!({
            "mode": match summary.mode {
                ftts_core::NormalizationMode::Verbatim => "verbatim",
                ftts_core::NormalizationMode::Conservative => "conservative",
                ftts_core::NormalizationMode::LocaleAware => "locale-aware",
            },
            "unicode_version": summary.unicode_version,
            "rules": summary.rules, "change_count": summary.change_count,
        })
    });
    json!({
        "seed": audio.seed,
        "sampling": {"mode": match audio.sampling_mode {
            SamplingMode::Production => "production", SamplingMode::CanonicalGreedy => "canonical_greedy" },
            "temperature": sampler::TEMPERATURE, "top_k": sampler::TOP_K, "top_p": 1.0,
            "talker_repetition_penalty": sampler::REPETITION_PENALTY,
            "subtalker_sampled": audio.sampling_mode == SamplingMode::Production,
            "determinism_scope": "build + ISA + sampler version + seed"},
        "utterance_start": match audio.utterance_start {
            UtteranceStart::Fresh => "fresh", UtteranceStart::Continuation => "continuation" },
        "normalization": normalization,
        "prepared_token_count": audio.prepared_token_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditioning_identity_changes_with_reference_without_exposing_transcript() {
        let quick = VoiceConditioning::XVector(vec![0.25]);
        let quality = VoiceConditioning::Icl {
            embedding: vec![0.25],
            transcript: "private reference sentence".into(),
            codec_codes: vec![7; 16],
        };
        let x = voice_identity(&quick);
        let icl = voice_identity(&quality);
        assert_eq!(x["embedding_sha256"], icl["embedding_sha256"]);
        assert_ne!(x["route"], icl["route"]);
        assert!(icl["reference_transcript_sha256"].as_str().unwrap().len() == 64);
        assert!(!icl.to_string().contains("private reference sentence"));
        assert!(!icl.to_string().contains("/Users/"));
    }

    #[test]
    fn capped_receipt_is_versioned_private_and_never_overwrites_evidence() {
        let path = std::env::temp_dir().join(format!(
            "ftts-trace-test-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let audio = SynthesizedAudio {
            stop_reason: ftts_core::admission::StopReason::FrameCapReached,
            frame_limit: 8,
            utterance_start: UtteranceStart::Fresh,
            normalization: Some(ftts_core::NormalizationTraceSummary {
                mode: ftts_core::NormalizationMode::Verbatim,
                unicode_version: "test".into(),
                rules: vec!["unicode_nfc".into()],
                change_count: 1,
            }),
            seed: 0,
            sampling_mode: SamplingMode::Production,
            frames: 8,
            prepared_token_count: 10,
            pcm: vec![],
            ttfa: None,
            ttfa_audible: None,
        };
        let receipt = json!({"schema": "ftts.generation", "schema_version": 1,
            "requested": {"seed": null}});
        GenerationTrace {
            path: path.clone(),
            receipt: receipt.clone(),
        }
        .finish(&audio, 100)
        .unwrap();
        let first = fs::read(&path).unwrap();
        let parsed: Value = serde_json::from_slice(&first).unwrap();
        assert_eq!(parsed["state"], "truncated");
        assert_eq!(parsed["requested"]["seed"], Value::Null);
        assert_eq!(parsed["effective"]["seed"], 0);
        assert_eq!(parsed["output"]["stop_reason"], "frame_cap_reached");
        assert_eq!(
            parsed["effective"]["normalization"]["rules"],
            json!(["unicode_nfc"])
        );
        assert!(
            GenerationTrace {
                path: path.clone(),
                receipt
            }
            .finish(&audio, 100)
            .is_err()
        );
        assert_eq!(fs::read(&path).unwrap(), first);
    }
}
