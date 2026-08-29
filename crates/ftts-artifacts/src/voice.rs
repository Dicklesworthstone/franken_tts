//! `.ftvoice` and `.ftvoice-cache` — the two-layer voice artifact (bead
//! `frankentts-p4-ftvoice-format-x0p`).
//!
//! # Two layers, opposite lifetimes
//!
//! A **`.ftvoice`** is *portable identity*: everything needed to re-derive a voice on any future
//! model revision — the speaker embedding, consent attestation, transcript, reference codec
//! tokens, signal diagnostics, and the exact preprocessing recipe that produced them. It is
//! written once per enrollment and read forever.
//!
//! A **`.ftvoice-cache`** is *derived state* keyed to one engine configuration: prompt-header KV,
//! primed codec streaming state, whatever a later bead stores. It is an optimization, never the
//! only copy: delete it and the pack re-derives everything from its primitives. Its validity is a
//! function of the full key tuple — see [`FtVoiceCacheKey`] for why every component is load-bearing.
//!
//! # Privacy profiles
//!
//! | Profile | Embedding | Codec tokens / transcript | Embedded audio |
//! |---|---|---|---|
//! | [`VoiceProfile::Portable`] | yes | yes | yes |
//! | [`VoiceProfile::Private`] | yes | yes | **no** |
//! | [`VoiceProfile::Minimal`] | yes | **no** | **no** |
//!
//! Privacy is enforced at READ time, not merely by writer discipline: a file claiming `private`
//! while carrying a `reference_audio` section is refused as a lie about itself
//! ([`FtVoiceError::ProfileViolation`]). A reader that accepts mislabeled packs makes every
//! downstream promise about privacy conditional on the writer having been honest.
//!
//! # Determinism (the idempotence gate)
//!
//! The container contains **no timestamps and no host-dependent data**. Directory JSON serializes
//! through `serde_json`'s ordered map, sections are laid out in a fixed canonical order, and the
//! recipe hash digests content only. Enrolling the *same* recording with the *same* engine
//! therefore produces byte-identical files — a metamorphic invariant tested here, not an accident
//! to be fixed later.
//!
//! # Hardening
//!
//! Same contract as `.fttsq`: checked arithmetic against the real buffer length, non-overlapping
//! digest-verified sections, capped counts and directory sizes, unknown values refused rather than
//! defaulted, and every refusal names the offending field or offset. A malformed pack is a named
//! error, never a partial load that resurfaces as a wrong-sounding clone.

use std::collections::BTreeMap;
use std::fmt;

use crate::sha256::{digest, hex_digest};

/// File magic for a portable `.ftvoice` pack.
pub const VOICE_MAGIC: &[u8; 8] = b"FTVCE\0\0\0";

/// File magic for a derived `.ftvoice-cache`.
pub const CACHE_MAGIC: &[u8; 8] = b"FTVCACH\0";

/// Format version this binary writes and the newest it can read.
///
/// A reader refuses anything newer: a future version may relocate bytes this binary would
/// otherwise misinterpret, and reading anyway is how a container acquires silent corruption.
pub const VOICE_FORMAT_VERSION: u32 = 1;

/// Format version for `.ftvoice-cache`; independent of [`VOICE_FORMAT_VERSION`] so the two
/// surfaces can evolve without coupling.
pub const CACHE_FORMAT_VERSION: u32 = 1;

/// Fixed prefix length: magic + version + directory length.
pub const HEADER_PREFIX_BYTES: u64 = 20;

/// Largest directory we will parse, guarding against a hostile length prefix.
///
/// Matches `fttsq::MAX_DIRECTORY_BYTES` deliberately — two artifact readers with different
/// limits is a bug waiting to be found.
pub const MAX_DIRECTORY_BYTES: u64 = 64 * 1024 * 1024;

/// Most sections one pack may declare. Three exist today; headroom is for alignment,
/// multi-reference, and future derived features.
pub const MAX_SECTIONS: usize = 16;

/// Speaker-embedding width. The ECAPA x-vector is 1,024 `f32`s — the same vector the legacy raw
/// `.spk` container stores (`TALKER_HIDDEN` in `ftts-model-qwen`, mirrored here because this crate
/// must not depend on the model crate for one integer).
pub const SPEAKER_EMBEDDING_LEN: usize = 1_024;

/// Byte length of the embedding section payload.
pub const EMBEDDING_BYTES: usize = SPEAKER_EMBEDDING_LEN * 4;

/// Canonical section order. The writer emits exactly this sequence; the reader accepts any
/// declaration order but the writer's determinism depends on this one.
pub const CANONICAL_SECTION_ORDER: [&str; 3] = [
    "speaker_embedding",
    "reference_codec_codes",
    "reference_audio",
];

/// Consent statement recorded at enrollment. Versioned text: bumping the wording bumps the id so
/// old packs stay distinguishable from ones enrolled under new language.
pub const CONSENT_STATEMENT_ID: &str = "frankentts-consent-v1";

/// How consent was attested. Doctrine 10: enrollment records it; there is no pack without an
/// answer, and `false` is inspectable rather than forbidden — inspection of what someone handed
/// you must remain possible even when synthesis should ask questions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsentMethod {
    /// Confirmed interactively at the terminal.
    Interactive,
    /// Passed explicitly via a flag by a script or agent.
    Flag,
    /// Inherited from an imported voice card or pack whose origin recorded consent.
    Imported,
}

impl ConsentMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Flag => "flag",
            Self::Imported => "imported",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "interactive" => Some(Self::Interactive),
            "flag" => Some(Self::Flag),
            "imported" => Some(Self::Imported),
            _ => None,
        }
    }
}

/// The consent record embedded in every pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsentAttestation {
    /// Whether the enroller affirmed they have the right to clone this voice.
    pub attested: bool,
    /// How the affirmation was captured.
    pub method: ConsentMethod,
}

/// Which privacy profile a pack was written under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoiceProfile {
    /// Everything, including optionally the normalized reference audio.
    Portable,
    /// Derived features only — no embedded recording.
    Private,
    /// The embedding and nothing else.
    Minimal,
}

impl VoiceProfile {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Private => "private",
            Self::Minimal => "minimal",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "portable" => Some(Self::Portable),
            "private" => Some(Self::Private),
            "minimal" => Some(Self::Minimal),
            _ => None,
        }
    }

    /// May this profile carry an embedded `reference_audio` payload?
    const fn allows_reference_audio(self) -> bool {
        matches!(self, Self::Portable)
    }

    /// May this profile carry the transcript / codec-token identity block?
    const fn allows_transcript(self) -> bool {
        !matches!(self, Self::Minimal)
    }
}

/// Signal-quality measurements taken over the reference at enrollment time.
///
/// Mirrors the enrollment-side detector output field-for-field; `None` where no speech was found
/// or the estimator had nothing to say. Stored so a later re-enrollment or diagnostic re-run can
/// compare against what was originally measured.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnrollmentDiagnostics {
    /// Fraction of samples railed at or over the clip level.
    pub clipping_fraction: f64,
    /// Longest consecutive at-rail run, samples.
    pub longest_clip_run: usize,
    /// Estimated true-peak overshoot over the digital peak, dB.
    pub intersample_overshoot_db: f64,
    /// Active-speech vs pause-floor power, dB; absent when no speech detected.
    pub snr_estimate_db: Option<f64>,
    /// 10th-percentile frame RMS, dBFS.
    pub pause_floor_dbfs: f64,
    /// Reverb-time equivalent, seconds; absent when the estimator could not measure.
    pub reverb_time_s: Option<f64>,
    /// Music-bed likelihood in `[0, 1]`.
    pub music_bed_likelihood: f64,
    /// Quarter-to-quarter energy spread, relative.
    pub stationarity_drift: f64,
    /// Whole-file RMS, dBFS (a loudness approximation, explicitly NOT LUFS).
    pub loudness_rms_dbfs: f64,
    /// Fraction of the timeline inside a voice-activity region.
    pub voice_activity_ratio: f64,
}

impl EnrollmentDiagnostics {
    fn to_json(&self) -> serde_json::Value {
        let finite_or_null = |value: f64| {
            if value.is_finite() {
                serde_json::Value::from(value)
            } else {
                serde_json::Value::Null
            }
        };
        serde_json::json!({
            "clipping_fraction": self.clipping_fraction,
            "longest_clip_run": self.longest_clip_run,
            "intersample_overshoot_db": self.intersample_overshoot_db,
            "snr_estimate_db": self.snr_estimate_db,
            "pause_floor_dbfs": finite_or_null(self.pause_floor_dbfs),
            "reverb_time_s": self.reverb_time_s,
            "music_bed_likelihood": self.music_bed_likelihood,
            "stationarity_drift": self.stationarity_drift,
            "loudness_rms_dbfs": finite_or_null(self.loudness_rms_dbfs),
            "voice_activity_ratio": self.voice_activity_ratio,
        })
    }

    fn parse(value: &serde_json::Value) -> Result<Self, FtVoiceError> {
        let field = |name: &str| -> Result<&serde_json::Value, FtVoiceError> {
            value.get(name).ok_or_else(|| FtVoiceError::Field {
                path: format!("diagnostics.{name}"),
                expected: "present".to_owned(),
            })
        };
        let f64_field = |name: &str| -> Result<f64, FtVoiceError> {
            field(name)?.as_f64().ok_or_else(|| FtVoiceError::Field {
                path: format!("diagnostics.{name}"),
                expected: "a number".to_owned(),
            })
        };
        let f64_or_null = |name: &str| -> Result<f64, FtVoiceError> {
            match field(name)? {
                serde_json::Value::Null => Ok(f64::NEG_INFINITY),
                other => other.as_f64().ok_or_else(|| FtVoiceError::Field {
                    path: format!("diagnostics.{name}"),
                    expected: "a number or null".to_owned(),
                }),
            }
        };
        let usize_field = |name: &str| -> Result<usize, FtVoiceError> {
            field(name)?
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| FtVoiceError::Field {
                    path: format!("diagnostics.{name}"),
                    expected: "a non-negative integer".to_owned(),
                })
        };
        let option_f64 =
            |name: &str| -> Result<Option<f64>, FtVoiceError> { Ok(field(name)?.as_f64()) };
        Ok(Self {
            clipping_fraction: f64_field("clipping_fraction")?,
            longest_clip_run: usize_field("longest_clip_run")?,
            intersample_overshoot_db: f64_field("intersample_overshoot_db")?,
            snr_estimate_db: option_f64("snr_estimate_db")?,
            pause_floor_dbfs: f64_or_null("pause_floor_dbfs")?,
            reverb_time_s: option_f64("reverb_time_s")?,
            music_bed_likelihood: f64_field("music_bed_likelihood")?,
            stationarity_drift: f64_field("stationarity_drift")?,
            loudness_rms_dbfs: f64_or_null("loudness_rms_dbfs")?,
            voice_activity_ratio: f64_field("voice_activity_ratio")?,
        })
    }
}

/// The reproducible preprocessing recipe: what happened to the raw recording before the embedding
/// was computed. Re-enrollment applies the same recipe deterministically; the cache key digests
/// this so a recipe change invalidates derived state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreprocessingRecipe {
    /// Sample rate everything was resampled to, Hz.
    pub target_sample_rate_hz: u32,
    /// Resampler kernel identifier, e.g. `"lanczos6"`.
    pub resampler: String,
    /// Denoise pass record, when one ran.
    pub denoise: Option<CleanupRecord>,
    /// Dereverb pass record, when one ran.
    pub dereverb: Option<CleanupRecord>,
}

/// One cleanup pass before/after measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct CleanupRecord {
    /// Engine that performed the pass, e.g. `"fastenhancer-s"` or `"omlsa"`.
    pub engine: String,
    /// Metric before the pass (pause-floor dBFS for denoise, RT60-equivalent seconds for dereverb).
    pub before: f64,
    /// The same metric after the pass.
    pub after: f64,
}

impl PreprocessingRecipe {
    fn to_json(&self) -> serde_json::Value {
        let cleanup = |record: &Option<CleanupRecord>| match record {
            None => serde_json::Value::Null,
            Some(record) => serde_json::json!({
                "engine": record.engine,
                "before": record.before,
                "after": record.after,
            }),
        };
        serde_json::json!({
            "target_sample_rate_hz": self.target_sample_rate_hz,
            "resampler": self.resampler,
            "denoise": cleanup(&self.denoise),
            "dereverb": cleanup(&self.dereverb),
        })
    }

    fn parse_cleanup(
        value: Option<&serde_json::Value>,
    ) -> Result<Option<CleanupRecord>, FtVoiceError> {
        match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(value) => {
                let string_field = |name: &str| -> Result<String, FtVoiceError> {
                    value
                        .get(name)
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                        .ok_or_else(|| FtVoiceError::Field {
                            path: format!("preprocessing.cleanup.{name}"),
                            expected: "a string".to_owned(),
                        })
                };
                let number_field = |name: &str| -> Result<f64, FtVoiceError> {
                    value
                        .get(name)
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| FtVoiceError::Field {
                            path: format!("preprocessing.cleanup.{name}"),
                            expected: "a number".to_owned(),
                        })
                };
                Ok(Some(CleanupRecord {
                    engine: string_field("engine")?,
                    before: number_field("before")?,
                    after: number_field("after")?,
                }))
            }
        }
    }

    fn parse(value: &serde_json::Value) -> Result<Self, FtVoiceError> {
        let rate = value
            .get("target_sample_rate_hz")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| FtVoiceError::Field {
                path: "preprocessing.target_sample_rate_hz".to_owned(),
                expected: "a non-negative integer".to_owned(),
            })?;
        let resampler = value
            .get("resampler")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| FtVoiceError::Field {
                path: "preprocessing.resampler".to_owned(),
                expected: "a string".to_owned(),
            })?;
        Ok(Self {
            target_sample_rate_hz: rate as u32,
            resampler: resampler.to_owned(),
            denoise: Self::parse_cleanup(value.get("denoise"))?,
            dereverb: Self::parse_cleanup(value.get("dereverb"))?,
        })
    }
}

/// Where the pack came from. Deliberately **no wall-clock timestamps**: provenance identifies
/// inputs and software, and a timestamp would break the byte-idempotence gate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Provenance {
    /// Engine name and version that performed the enrollment, e.g. `ftts 0.1.8`.
    pub engine: String,
    /// SHA-256 of the source recording bytes, when a recording was the input.
    pub source_audio_sha256: Option<String>,
    /// Length of the source recording in samples at its native rate, when known.
    pub source_frames: Option<u64>,
}

impl Provenance {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "engine": self.engine,
            "source_audio_sha256": self.source_audio_sha256,
            "source_frames": self.source_frames,
        })
    }

    fn parse(value: &serde_json::Value) -> Result<Self, FtVoiceError> {
        let engine = value
            .get("engine")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| FtVoiceError::Field {
                path: "provenance.engine".to_owned(),
                expected: "a string".to_owned(),
            })?;
        Ok(Self {
            engine: engine.to_owned(),
            source_audio_sha256: value
                .get("source_audio_sha256")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            source_frames: value
                .get("source_frames")
                .and_then(serde_json::Value::as_u64),
        })
    }
}

/// A parsed, fully verified `.ftvoice` pack.
///
/// Payloads are owned: a voice pack is kilobytes to a few megabytes, so unlike `.fttsq` there is
/// no mapping layer to justify.
#[derive(Clone, Debug)]
pub struct VoicePack {
    /// Privacy profile the pack declares (and has been validated against).
    pub profile: VoiceProfile,
    /// The consent record.
    pub consent: ConsentAttestation,
    /// BCP-47-ish language tag of the transcript, when present.
    pub language: Option<String>,
    /// The reference transcript, when the profile carries one.
    pub transcript: Option<String>,
    /// Speech regions kept for enrollment, as `(start_sample, end_sample)` pairs at the recipe's
    /// target sample rate.
    pub speech_regions: Vec<(u64, u64)>,
    /// Signal diagnostics measured at enrollment, when recorded.
    pub diagnostics: Option<EnrollmentDiagnostics>,
    /// The preprocessing recipe, when recorded.
    pub preprocessing: Option<PreprocessingRecipe>,
    /// Provenance of the enrollment run.
    pub provenance: Provenance,
    /// The speaker embedding, validated finite and [`SPEAKER_EMBEDDING_LEN`] wide.
    pub embedding: Vec<f32>,
    /// Reference codec tokens (ICL identity), when the profile carries them.
    pub codec_codes: Option<Vec<u32>>,
    /// Embedded normalized reference audio (portable profile only), verbatim section bytes.
    pub reference_audio: Option<Vec<u8>>,
    /// Digests by section name, as recorded in the directory (verified during parse).
    pub section_digests: BTreeMap<String, String>,
}

impl VoicePack {
    /// Stable content hash of the whole voice recipe: the `voice_recipe_hash` component of every
    /// cache key. Digests the embedding plus every identity-affecting primitive; excludes nothing
    /// user-visible on purpose — two packs with the same hash ARE the same voice recipe.
    ///
    /// # Errors
    ///
    /// Returns [`FtVoiceError::Field`] if a profile-required component is somehow absent — only
    /// reachable on hand-constructed structs, never on parsed packs.
    pub fn recipe_hash(&self) -> Result<String, FtVoiceError> {
        let codes_digest = match &self.codec_codes {
            Some(codes) => {
                let mut bytes = Vec::with_capacity(codes.len() * 4);
                for code in codes {
                    bytes.extend_from_slice(&code.to_le_bytes());
                }
                hex_digest(&digest(&bytes))
            }
            None => String::new(),
        };
        // serde_json's map is order-canonical (BTreeMap), so identical recipes hash identically.
        let value = serde_json::json!({
            "profile": self.profile.as_str(),
            "consent_attested": self.consent.attested,
            "language": self.language,
            "transcript": self.transcript,
            "speech_regions": self.speech_regions.iter().map(|(start, end)| [start, end])
                .collect::<Vec<_>>(),
            "diagnostics": self.diagnostics.as_ref().map(EnrollmentDiagnostics::to_json),
            "preprocessing": self.preprocessing.as_ref().map(PreprocessingRecipe::to_json),
            "embedding_sha256": hex_digest(&digest(f32_le_bytes(&self.embedding).as_slice())),
            "codec_codes_sha256": codes_digest,
        });
        Ok(hex_digest(&digest(value.to_string().as_bytes())))
    }
}

fn f32_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// What went wrong reading a voice artifact. Every variant names the offending field, section, or
/// offset: a refusal that does not say what it refused costs an hour of bisecting.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FtVoiceError {
    /// Buffer shorter than the fixed header.
    TooShort {
        /// Actual length.
        length: usize,
    },
    /// Leading bytes are not this format's magic.
    BadMagic {
        /// The eight bytes found.
        found: [u8; 8],
    },
    /// Written by a newer binary.
    UnsupportedVersion {
        /// Version found.
        found: u32,
        /// Newest supported.
        supported: u32,
    },
    /// Declared directory length exceeds the cap or the buffer.
    DirectoryLength {
        /// Declared length.
        declared: u64,
        /// The limit or bound exceeded.
        limit: u64,
    },
    /// Directory is not valid UTF-8 JSON.
    DirectoryMalformed {
        /// Serializer/parser detail.
        detail: String,
    },
    /// A required field is missing or has the wrong JSON type.
    Field {
        /// Dotted path to the field.
        path: String,
        /// What type/presence was expected.
        expected: String,
    },
    /// A field carries a value outside the known enumeration.
    UnknownValue {
        /// Dotted path to the field.
        path: String,
        /// The unrecognized string.
        found: String,
    },
    /// A count exceeds its cap.
    LimitExceeded {
        /// What was capped.
        what: &'static str,
        /// Count found.
        found: usize,
        /// The cap.
        limit: usize,
    },
    /// A range escapes the buffer or its declared bound.
    RangeOutOfBounds {
        /// What the range belongs to.
        what: String,
        /// Start offset.
        offset: u64,
        /// Length.
        length: u64,
        /// The bound crossed.
        bound: u64,
    },
    /// Two sections claim overlapping bytes.
    SectionOverlap {
        /// First section.
        first: String,
        /// Second section.
        second: String,
    },
    /// A section name appears twice.
    DuplicateSection {
        /// The repeated name.
        name: String,
    },
    /// Section bytes do not match their recorded digest.
    DigestMismatch {
        /// Section name.
        section: String,
        /// Recorded digest.
        expected: String,
        /// Computed digest.
        actual: String,
    },
    /// The mandatory embedding section is missing.
    EmbeddingMissing,
    /// The embedding is not exactly [`SPEAKER_EMBEDDING_LEN`] floats.
    EmbeddingLength {
        /// Byte length found.
        found: usize,
    },
    /// The embedding holds a NaN or infinity.
    NonFiniteEmbedding {
        /// Index of the first offending value.
        index: usize,
    },
    /// The pack's contents contradict its claimed privacy profile.
    ProfileViolation {
        /// Claimed profile.
        profile: VoiceProfile,
        /// What contradicts it.
        detail: String,
    },
    /// Reference codec tokens fall outside the 2,048-way code space.
    CodecCodeOutOfRange {
        /// Index of the offending token.
        index: usize,
        /// The token.
        code: u32,
    },
    /// A speech region is malformed (end before start).
    SpeechRegionInvalid {
        /// Region index.
        index: usize,
        /// Start sample.
        start: u64,
        /// End sample.
        end: u64,
    },
}

impl fmt::Display for FtVoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort { length } => write!(
                f,
                "not a .ftvoice artifact: {length} bytes is shorter than the \
                 {HEADER_PREFIX_BYTES}-byte header"
            ),
            Self::BadMagic { found } => {
                write!(f, "not a .ftvoice/.ftvoice-cache artifact: magic {found:?}")
            }
            Self::UnsupportedVersion { found, supported } => write!(
                f,
                "artifact format version {found} is newer than this binary supports ({supported}); \
                 upgrade ftts rather than reading it with a stale layout"
            ),
            Self::DirectoryLength { declared, limit } => {
                write!(f, "directory length {declared} exceeds {limit}")
            }
            Self::DirectoryMalformed { detail } => write!(f, "directory is malformed: {detail}"),
            Self::Field { path, expected } => {
                write!(f, "directory field `{path}` is missing or not {expected}")
            }
            Self::UnknownValue { path, found } => write!(
                f,
                "directory field `{path}` has unknown value `{found}`; this artifact needs a \
                 newer ftts"
            ),
            Self::LimitExceeded { what, found, limit } => {
                write!(f, "{what} count {found} exceeds the cap of {limit}")
            }
            Self::RangeOutOfBounds {
                what,
                offset,
                length,
                bound,
            } => write!(
                f,
                "{what} range [{offset}, {offset}+{length}) runs past its bound {bound}"
            ),
            Self::SectionOverlap { first, second } => {
                write!(
                    f,
                    "sections `{first}` and `{second}` claim overlapping bytes"
                )
            }
            Self::DuplicateSection { name } => write!(f, "section `{name}` is declared twice"),
            Self::DigestMismatch {
                section,
                expected,
                actual,
            } => write!(
                f,
                "section `{section}` is corrupt: recorded sha256 {expected}, computed {actual}"
            ),
            Self::EmbeddingMissing => f.write_str(
                "pack carries no `speaker_embedding` section; a voice pack without a voice is a \
                 refusal, not an empty default",
            ),
            Self::EmbeddingLength { found } => write!(
                f,
                "speaker_embedding section is {found} bytes; expected {EMBEDDING_BYTES} \
                 ({SPEAKER_EMBEDDING_LEN} f32)"
            ),
            Self::NonFiniteEmbedding { index } => write!(
                f,
                "speaker_embedding holds a non-finite value at index {index}; refusing instead of \
                 cloning garbage"
            ),
            Self::ProfileViolation { profile, detail } => write!(
                f,
                "pack claims profile `{}` but {} — refusing a pack that lies about its privacy",
                profile.as_str(),
                detail
            ),
            Self::CodecCodeOutOfRange { index, code } => write!(
                f,
                "reference_codec_codes[{index}] = {code} is outside the 2,048-way code space"
            ),
            Self::SpeechRegionInvalid { index, start, end } => {
                write!(
                    f,
                    "speech_regions[{index}] = ({start}, {end}) ends before it starts"
                )
            }
        }
    }
}

impl std::error::Error for FtVoiceError {}

/// Parses and fully verifies a `.ftvoice` pack from its complete bytes.
///
/// Verification order: header, directory bounds, JSON structure, profile/privacy consistency,
/// then per-section digests, then semantic checks on payloads. Nothing is exposed before every
/// check has passed.
///
/// # Errors
///
/// See [`FtVoiceError`] — every failure mode is a named variant.
pub fn parse_voice_pack(bytes: &[u8]) -> Result<VoicePack, FtVoiceError> {
    if bytes.len() < HEADER_PREFIX_BYTES as usize {
        return Err(FtVoiceError::TooShort {
            length: bytes.len(),
        });
    }
    if &bytes[0..8] != VOICE_MAGIC {
        let mut found = [0u8; 8];
        found.copy_from_slice(&bytes[0..8]);
        return Err(FtVoiceError::BadMagic { found });
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version > VOICE_FORMAT_VERSION {
        return Err(FtVoiceError::UnsupportedVersion {
            found: version,
            supported: VOICE_FORMAT_VERSION,
        });
    }
    let directory_len = u64::from_le_bytes(bytes[12..20].try_into().expect("8 bytes"));
    if directory_len > MAX_DIRECTORY_BYTES {
        return Err(FtVoiceError::DirectoryLength {
            declared: directory_len,
            limit: MAX_DIRECTORY_BYTES,
        });
    }
    let directory_end =
        HEADER_PREFIX_BYTES
            .checked_add(directory_len)
            .ok_or(FtVoiceError::DirectoryLength {
                declared: directory_len,
                limit: MAX_DIRECTORY_BYTES,
            })?;
    if directory_end > bytes.len() as u64 {
        return Err(FtVoiceError::RangeOutOfBounds {
            what: "directory".to_owned(),
            offset: HEADER_PREFIX_BYTES,
            length: directory_len,
            bound: bytes.len() as u64,
        });
    }
    let directory_bytes = &bytes[HEADER_PREFIX_BYTES as usize..directory_end as usize];
    let directory: serde_json::Value =
        serde_json::from_slice(directory_bytes).map_err(|error| {
            FtVoiceError::DirectoryMalformed {
                detail: error.to_string(),
            }
        })?;
    let object = directory
        .as_object()
        .ok_or_else(|| FtVoiceError::DirectoryMalformed {
            detail: "directory is not a JSON object".to_owned(),
        })?;

    // -- scalar identity fields -------------------------------------------------
    let schema_version = object
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    match schema_version {
        Some(1) => {}
        other => {
            return Err(FtVoiceError::Field {
                path: "schema_version".to_owned(),
                expected: format!("1, found {other:?}"),
            });
        }
    }
    let profile = object
        .get("profile")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| FtVoiceError::Field {
            path: "profile".to_owned(),
            expected: "a string".to_owned(),
        })?;
    let profile = VoiceProfile::parse(profile).ok_with(|| FtVoiceError::UnknownValue {
        path: "profile".to_owned(),
        found: profile.to_owned(),
    })?;

    // -- consent ---------------------------------------------------------------
    let consent_value = object.get("consent").ok_with(|| FtVoiceError::Field {
        path: "consent".to_owned(),
        expected: "an object".to_owned(),
    })?;
    let attested = consent_value
        .get("attested")
        .and_then(serde_json::Value::as_bool)
        .ok_with(|| FtVoiceError::Field {
            path: "consent.attested".to_owned(),
            expected: "a boolean".to_owned(),
        })?;
    let method = consent_value
        .get("method")
        .and_then(serde_json::Value::as_str)
        .ok_with(|| FtVoiceError::Field {
            path: "consent.method".to_owned(),
            expected: "a string".to_owned(),
        })?;
    let method = ConsentMethod::parse(method).ok_with(|| FtVoiceError::UnknownValue {
        path: "consent.method".to_owned(),
        found: method.to_owned(),
    })?;
    let statement = consent_value
        .get("statement_id")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    if !statement.is_empty() && statement != CONSENT_STATEMENT_ID {
        return Err(FtVoiceError::UnknownValue {
            path: "consent.statement_id".to_owned(),
            found: statement.to_owned(),
        });
    }
    let consent = ConsentAttestation { attested, method };

    // -- optional identity blocks ----------------------------------------------
    let language = object
        .get("language")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let transcript = object
        .get("transcript")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let mut speech_regions = Vec::new();
    if let Some(regions) = object
        .get("speech_regions")
        .and_then(serde_json::Value::as_array)
    {
        if regions.len() > MAX_SECTIONS * 1024 {
            return Err(FtVoiceError::LimitExceeded {
                what: "speech_regions",
                found: regions.len(),
                limit: MAX_SECTIONS * 1024,
            });
        }
        for (index, region) in regions.iter().enumerate() {
            let pair = region.as_array().ok_with(|| FtVoiceError::Field {
                path: format!("speech_regions[{index}]"),
                expected: "a [start, end] pair".to_owned(),
            })?;
            let start = pair
                .first()
                .and_then(serde_json::Value::as_u64)
                .ok_with(|| FtVoiceError::Field {
                    path: format!("speech_regions[{index}].start"),
                    expected: "a non-negative integer".to_owned(),
                })?;
            let end = pair
                .get(1)
                .and_then(serde_json::Value::as_u64)
                .ok_with(|| FtVoiceError::Field {
                    path: format!("speech_regions[{index}].end"),
                    expected: "a non-negative integer".to_owned(),
                })?;
            if end < start {
                return Err(FtVoiceError::SpeechRegionInvalid { index, start, end });
            }
            speech_regions.push((start, end));
        }
    }
    let diagnostics = match object.get("diagnostics") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => Some(EnrollmentDiagnostics::parse(value)?),
    };
    let preprocessing = match object.get("preprocessing") {
        None | Some(serde_json::Value::Null) => None,
        Some(value) => Some(PreprocessingRecipe::parse(value)?),
    };
    let provenance = match object.get("provenance") {
        None => Provenance::default(),
        Some(value) => Provenance::parse(value)?,
    };

    // -- sections ---------------------------------------------------------------
    let sections_value = object.get("sections").and_then(serde_json::Value::as_array);
    let mut sections: BTreeMap<String, (u64, u64, String)> = BTreeMap::new();
    if let Some(entries) = sections_value {
        if entries.len() > MAX_SECTIONS {
            return Err(FtVoiceError::LimitExceeded {
                what: "sections",
                found: entries.len(),
                limit: MAX_SECTIONS,
            });
        }
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].name".to_owned(),
                    expected: "a string".to_owned(),
                })?;
            if !CANONICAL_SECTION_ORDER.contains(&name) {
                return Err(FtVoiceError::UnknownValue {
                    path: "sections[].name".to_owned(),
                    found: name.to_owned(),
                });
            }
            let offset = entry
                .get("offset")
                .and_then(serde_json::Value::as_u64)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].offset".to_owned(),
                    expected: "a non-negative integer".to_owned(),
                })?;
            let length = entry
                .get("length")
                .and_then(serde_json::Value::as_u64)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].length".to_owned(),
                    expected: "a non-negative integer".to_owned(),
                })?;
            let sha256 = entry
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].sha256".to_owned(),
                    expected: "a string".to_owned(),
                })?
                .to_owned();
            if sections
                .insert(name.to_owned(), (offset, length, sha256))
                .is_some()
            {
                return Err(FtVoiceError::DuplicateSection {
                    name: name.to_owned(),
                });
            }
        }

        // -- profile/privacy consistency BEFORE geometry and digests -----------------
        // A pack that lies about its profile is refused on its declared CONTENTS alone,
        // even when its offsets are inconsistent — the lie is the earliest failure.
        if !profile.allows_transcript()
            && (transcript.is_some() || sections.contains_key("reference_codec_codes"))
        {
            return Err(FtVoiceError::ProfileViolation {
                profile,
                detail: "it carries a transcript or reference codec tokens".to_owned(),
            });
        }
        if !profile.allows_reference_audio() && sections.contains_key("reference_audio") {
            return Err(FtVoiceError::ProfileViolation {
                profile,
                detail: "it carries embedded reference audio".to_owned(),
            });
        }

        // -- range and overlap validation -------------------------------------------
        let file_len = bytes.len() as u64;
        for (name, (offset, length, _)) in &sections {
            let end = offset
                .checked_add(*length)
                .ok_with(|| FtVoiceError::RangeOutOfBounds {
                    what: format!("section `{name}`"),
                    offset: *offset,
                    length: *length,
                    bound: file_len,
                })?;
            if end > file_len {
                return Err(FtVoiceError::RangeOutOfBounds {
                    what: format!("section `{name}`"),
                    offset: *offset,
                    length: *length,
                    bound: file_len,
                });
            }
        }
        let mut ranges: Vec<(String, u64, u64)> = sections
            .iter()
            .map(|(name, (offset, length, _))| (name.clone(), *offset, *length))
            .collect();
        ranges.sort_by_key(|(_, offset, _)| *offset);
        for window in ranges.windows(2) {
            let (first_name, first_offset, first_length) = &window[0];
            let (second_name, second_offset, _) = &window[1];
            if first_offset + first_length > *second_offset {
                return Err(FtVoiceError::SectionOverlap {
                    first: first_name.clone(),
                    second: second_name.clone(),
                });
            }
        }
    }

    // -- digests, then payloads --------------------------------------------------
    let mut section_digests = BTreeMap::new();
    for (name, (offset, length, sha256)) in &sections {
        let payload = &bytes[*offset as usize..(offset + length) as usize];
        let actual = hex_digest(&digest(payload));
        if &actual != sha256 {
            return Err(FtVoiceError::DigestMismatch {
                section: name.clone(),
                expected: sha256.clone(),
                actual,
            });
        }
        section_digests.insert(name.clone(), sha256.clone());
    }

    let Some(&(embedding_offset, embedding_len, _)) = sections.get("speaker_embedding") else {
        return Err(FtVoiceError::EmbeddingMissing);
    };
    if embedding_len != EMBEDDING_BYTES as u64 {
        return Err(FtVoiceError::EmbeddingLength {
            found: embedding_len as usize,
        });
    }
    let embedding_bytes =
        &bytes[embedding_offset as usize..(embedding_offset + embedding_len) as usize];
    let embedding: Vec<f32> = embedding_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    if let Some(index) = embedding.iter().position(|value| !value.is_finite()) {
        return Err(FtVoiceError::NonFiniteEmbedding { index });
    }

    let codec_codes = match sections.get("reference_codec_codes") {
        None => None,
        Some(&(offset, length, _)) => {
            let payload = &bytes[offset as usize..(offset + length) as usize];
            if !payload.len().is_multiple_of(std::mem::size_of::<u32>()) {
                return Err(FtVoiceError::Field {
                    path: "sections.reference_codec_codes.length".to_owned(),
                    expected: "a multiple of 4".to_owned(),
                });
            }
            let codes: Vec<u32> = payload
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| u32::from_le_bytes(*chunk))
                .collect();
            if let Some((index, &code)) = codes.iter().enumerate().find(|&(_, &code)| code >= 2_048)
            {
                return Err(FtVoiceError::CodecCodeOutOfRange { index, code });
            }
            Some(codes)
        }
    };
    let reference_audio = sections
        .get("reference_audio")
        .map(|&(offset, length, _)| bytes[offset as usize..(offset + length) as usize].to_vec());
    Ok(VoicePack {
        profile,
        consent,
        language,
        transcript,
        speech_regions,
        diagnostics,
        preprocessing,
        provenance,
        embedding,
        codec_codes,
        reference_audio,
        section_digests,
    })
}

/// Serializes a pack to canonical bytes: fixed header, sorted-key JSON directory, sections in
/// [`CANONICAL_SECTION_ORDER`], digest per section.
///
/// The output is a pure function of the input — same pack in, same bytes out, always. That is the
/// idempotence metamorphic gate ("identical voice input -> identical portable .ftvoice"), which
/// works because nothing host-dependent (time, paths, iteration order) enters the encoding.
///
/// # Errors
///
/// Returns [`FtVoiceError::EmbeddingLength`] / [`FtVoiceError::NonFiniteEmbedding`] /
/// [`FtVoiceError::ProfileViolation`] for structurally invalid packs, so a bad pack cannot be
/// serialized into existence either.
pub fn serialize_voice_pack(pack: &VoicePack) -> Result<Vec<u8>, FtVoiceError> {
    if pack.embedding.len() != SPEAKER_EMBEDDING_LEN {
        return Err(FtVoiceError::EmbeddingLength {
            found: pack.embedding.len() * 4,
        });
    }
    if let Some(index) = pack.embedding.iter().position(|value| !value.is_finite()) {
        return Err(FtVoiceError::NonFiniteEmbedding { index });
    }
    // Writer-side privacy enforcement mirrors the reader: a lie must not be writable.
    if !pack.profile.allows_transcript()
        && (pack.transcript.is_some() || pack.codec_codes.is_some())
    {
        return Err(FtVoiceError::ProfileViolation {
            profile: pack.profile,
            detail: "transcript or codec tokens were supplied".to_owned(),
        });
    }
    if !pack.profile.allows_reference_audio() && pack.reference_audio.is_some() {
        return Err(FtVoiceError::ProfileViolation {
            profile: pack.profile,
            detail: "embedded reference audio was supplied".to_owned(),
        });
    }
    if let Some((index, &code)) = pack
        .codec_codes
        .as_ref()
        .and_then(|codes| codes.iter().enumerate().find(|&(_, &code)| code >= 2_048))
    {
        return Err(FtVoiceError::CodecCodeOutOfRange { index, code });
    }
    for (index, (start, end)) in pack.speech_regions.iter().enumerate() {
        if end < start {
            return Err(FtVoiceError::SpeechRegionInvalid {
                index,
                start: *start,
                end: *end,
            });
        }
    }

    // Payload bytes and per-section digests are content-only; offsets depend on the directory
    // length, and the directory length depends on the offset digits — so iterate to a fixed
    // point (converges in a couple of passes; digit-count changes only shrink).
    let mut section_payloads: Vec<(&str, Vec<u8>)> = Vec::new();
    for name in CANONICAL_SECTION_ORDER {
        let bytes: Vec<u8> = match name {
            "speaker_embedding" => f32_le_bytes(&pack.embedding),
            "reference_codec_codes" => match &pack.codec_codes {
                None => continue,
                Some(codes) => {
                    let mut bytes = Vec::with_capacity(codes.len() * 4);
                    for code in codes {
                        bytes.extend_from_slice(&code.to_le_bytes());
                    }
                    bytes
                }
            },
            "reference_audio" => match &pack.reference_audio {
                None => continue,
                Some(audio) => audio.clone(),
            },
            other => unreachable!("canonical section set gained `{other}` without a writer arm"),
        };
        section_payloads.push((name, bytes));
    }

    let build_directory = |directory_len: u64| {
        let base = HEADER_PREFIX_BYTES + directory_len;
        let mut cursor = 0u64;
        let sections_json: Vec<serde_json::Value> = section_payloads
            .iter()
            .map(|(name, bytes)| {
                let offset = base + cursor;
                cursor += bytes.len() as u64;
                serde_json::json!({
                    "length": bytes.len() as u64,
                    "name": name,
                    "offset": offset,
                    "sha256": hex_digest(&digest(bytes)),
                })
            })
            .collect();
        serde_json::json!({
            "consent": {
                "attested": pack.consent.attested,
                "method": pack.consent.method.as_str(),
                "statement_id": CONSENT_STATEMENT_ID,
            },
            "diagnostics": pack.diagnostics.as_ref().map(EnrollmentDiagnostics::to_json),
            "language": pack.language,
            "preprocessing": pack.preprocessing.as_ref().map(PreprocessingRecipe::to_json),
            "profile": pack.profile.as_str(),
            "provenance": pack.provenance.to_json(),
            "schema_version": VOICE_FORMAT_VERSION,
            "sections": sections_json,
            "speech_regions": pack.speech_regions.iter().map(|(start, end)| [start, end]).collect::<Vec<_>>(),
            "transcript": pack.transcript,
        })
    };

    // serde_json's default map is a BTreeMap, so key order — and therefore bytes — is stable.
    let mut directory_len = 0u64;
    let directory_bytes = loop {
        let candidate = build_directory(directory_len).to_string().into_bytes();
        let actual_len = candidate.len() as u64;
        if actual_len == directory_len {
            break candidate;
        }
        directory_len = actual_len;
    };
    let payload: Vec<u8> = section_payloads
        .iter()
        .flat_map(|(_, bytes)| bytes.iter().copied())
        .collect();

    let mut out =
        Vec::with_capacity(HEADER_PREFIX_BYTES as usize + directory_bytes.len() + payload.len());
    out.extend_from_slice(VOICE_MAGIC);
    out.extend_from_slice(&VOICE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(directory_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&directory_bytes);
    out.extend_from_slice(&payload);
    Ok(out)
}

// ---------------------------------------------------------------------------
// .ftvoice-cache
// ---------------------------------------------------------------------------

/// One component of a `.ftvoice-cache` key. Every field is load-bearing: OQ-10 §5.1 shows that
/// `language_id` and `speaker_embed` sit inside the cached header positions, so leaving any of
/// these out of the key produces a cache that serves the wrong voice's state silently.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FtVoiceCacheKey {
    /// [`VoicePack::recipe_hash`] of the pack this cache derives from.
    pub voice_recipe_hash: String,
    /// Hash of the loaded model artifact.
    pub model_hash: String,
    /// Prompt-builder implementation version.
    pub prompt_builder_version: u32,
    /// `"streaming"` or `"non_streaming"` — the modes build different prompts.
    pub streaming_mode: String,
    /// Quantization recipe identifier of the runtime route.
    pub quant_recipe: String,
    /// Math mode (`strict`/`fast`) in force.
    pub math_mode: String,
    /// Engine ABI revision that produced the payloads.
    pub engine_abi: u32,
    /// Language id fed to the prompt builder.
    pub language_id: String,
    /// Digest of the speaker embedding actually used (redundant with the recipe hash on purpose:
    /// OQ-10 calls out that it feels like a runtime option but invalidates even the header).
    pub speaker_embed_sha256: String,
    /// Digest of the reference-transcript token ids, when ICL.
    pub ref_transcript_tokens_sha256: Option<String>,
    /// Digest of the reference codec codes, when ICL.
    pub ref_codec_codes_sha256: Option<String>,
}

impl FtVoiceCacheKey {
    /// The single derived digest identifying this configuration.
    ///
    /// Changing ANY component changes the digest — that totality is the invalidation contract and
    /// is tested exhaustively in this module.
    #[must_use]
    pub fn cache_key(&self) -> String {
        let value = serde_json::json!({
            "engine_abi": self.engine_abi,
            "language_id": self.language_id,
            "math_mode": self.math_mode,
            "model_hash": self.model_hash,
            "prompt_builder_version": self.prompt_builder_version,
            "quant_recipe": self.quant_recipe,
            "ref_codec_codes_sha256": self.ref_codec_codes_sha256,
            "ref_transcript_tokens_sha256": self.ref_transcript_tokens_sha256,
            "speaker_embed_sha256": self.speaker_embed_sha256,
            "streaming_mode": self.streaming_mode,
            "voice_recipe_hash": self.voice_recipe_hash,
        });
        hex_digest(&digest(value.to_string().as_bytes()))
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "cache_key": self.cache_key(),
            "components": {
                "engine_abi": self.engine_abi,
                "language_id": self.language_id,
                "math_mode": self.math_mode,
                "model_hash": self.model_hash,
                "prompt_builder_version": self.prompt_builder_version,
                "quant_recipe": self.quant_recipe,
                "ref_codec_codes_sha256": self.ref_codec_codes_sha256,
                "ref_transcript_tokens_sha256": self.ref_transcript_tokens_sha256,
                "speaker_embed_sha256": self.speaker_embed_sha256,
                "streaming_mode": self.streaming_mode,
                "voice_recipe_hash": self.voice_recipe_hash,
            }
        })
    }
}

/// A parsed `.ftvoice-cache`: the key it was built under plus opaque, digest-verified payload
/// blobs. This crate defines the CONTAINER; later beads define what goes inside each blob and own
/// their schemas.
#[derive(Clone, Debug)]
pub struct FtVoiceCache {
    /// The key the cache was computed under.
    pub key: FtVoiceCacheKey,
    /// Payload blobs by name, verified against their digests.
    pub blobs: BTreeMap<String, Vec<u8>>,
}

/// Serializes a `.ftvoice-cache` with the same layout guarantees as the pack.
///
/// Blob names must be lowercase identifiers; blob order in the file is sorted-name order, keeping
/// output deterministic.
///
/// # Errors
///
/// Returns [`FtVoiceError::LimitExceeded`] above [`MAX_SECTIONS`] blobs and
/// [`FtVoiceError::UnknownValue`] for malformed blob names.
pub fn serialize_ftvoice_cache(
    key: &FtVoiceCacheKey,
    blobs: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<u8>, FtVoiceError> {
    if blobs.len() > MAX_SECTIONS {
        return Err(FtVoiceError::LimitExceeded {
            what: "blobs",
            found: blobs.len(),
            limit: MAX_SECTIONS,
        });
    }
    for name in blobs.keys() {
        let valid = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit());
        if !valid {
            return Err(FtVoiceError::UnknownValue {
                path: "blobs[].name".to_owned(),
                found: name.clone(),
            });
        }
    }
    // Same fixed-point layout as the pack writer: offsets need the directory length, the
    // directory needs the offset digits; iterate until stable (converges in a couple passes).
    let build_directory = |directory_len: u64| {
        let base = HEADER_PREFIX_BYTES + directory_len;
        let mut cursor = 0u64;
        let sections_json: Vec<serde_json::Value> = blobs
            .iter()
            .map(|(name, bytes)| {
                let offset = base + cursor;
                cursor += bytes.len() as u64;
                serde_json::json!({
                    "length": bytes.len() as u64,
                    "name": name,
                    "offset": offset,
                    "sha256": hex_digest(&digest(bytes)),
                })
            })
            .collect();
        serde_json::json!({
            "key": key.to_json(),
            "schema_version": CACHE_FORMAT_VERSION,
            "sections": sections_json,
        })
    };
    let mut directory_len = 0u64;
    let directory_bytes = loop {
        let candidate = build_directory(directory_len).to_string().into_bytes();
        let actual_len = candidate.len() as u64;
        if actual_len == directory_len {
            break candidate;
        }
        directory_len = actual_len;
    };
    let payload: Vec<u8> = blobs
        .values()
        .flat_map(|bytes| bytes.iter().copied())
        .collect();

    let mut out =
        Vec::with_capacity(HEADER_PREFIX_BYTES as usize + directory_bytes.len() + payload.len());
    out.extend_from_slice(CACHE_MAGIC);
    out.extend_from_slice(&CACHE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(directory_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&directory_bytes);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Parses and verifies a `.ftvoice-cache`. The stored `cache_key` digest is recomputed from the
/// stored components and must match — a cache whose key disagrees with its own components was
/// tampered with or written by a buggy writer, and serving it would defeat the entire invalidation
/// scheme.
///
/// # Errors
///
/// See [`FtVoiceError`]; additionally [`FtVoiceError::DigestMismatch`] when the recomputed key
/// digest disagrees with the stored one.
pub fn parse_ftvoice_cache(bytes: &[u8]) -> Result<FtVoiceCache, FtVoiceError> {
    if bytes.len() < HEADER_PREFIX_BYTES as usize {
        return Err(FtVoiceError::TooShort {
            length: bytes.len(),
        });
    }
    if &bytes[0..8] != CACHE_MAGIC {
        let mut found = [0u8; 8];
        found.copy_from_slice(&bytes[0..8]);
        return Err(FtVoiceError::BadMagic { found });
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
    if version > CACHE_FORMAT_VERSION {
        return Err(FtVoiceError::UnsupportedVersion {
            found: version,
            supported: CACHE_FORMAT_VERSION,
        });
    }
    let directory_len = u64::from_le_bytes(bytes[12..20].try_into().expect("8 bytes"));
    if directory_len > MAX_DIRECTORY_BYTES {
        return Err(FtVoiceError::DirectoryLength {
            declared: directory_len,
            limit: MAX_DIRECTORY_BYTES,
        });
    }
    let directory_end = HEADER_PREFIX_BYTES.checked_add(directory_len).ok_with(|| {
        FtVoiceError::DirectoryLength {
            declared: directory_len,
            limit: MAX_DIRECTORY_BYTES,
        }
    })?;
    if directory_end > bytes.len() as u64 {
        return Err(FtVoiceError::RangeOutOfBounds {
            what: "directory".to_owned(),
            offset: HEADER_PREFIX_BYTES,
            length: directory_len,
            bound: bytes.len() as u64,
        });
    }
    let directory: serde_json::Value =
        serde_json::from_slice(&bytes[HEADER_PREFIX_BYTES as usize..directory_end as usize])
            .map_err(|error| FtVoiceError::DirectoryMalformed {
                detail: error.to_string(),
            })?;
    let object = directory
        .as_object()
        .ok_with(|| FtVoiceError::DirectoryMalformed {
            detail: "directory is not a JSON object".to_owned(),
        })?;
    let schema_version = object
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    match schema_version {
        Some(1) => {}
        other => {
            return Err(FtVoiceError::Field {
                path: "schema_version".to_owned(),
                expected: format!("1, found {other:?}"),
            });
        }
    }
    let key_value = object.get("key").ok_with(|| FtVoiceError::Field {
        path: "key".to_owned(),
        expected: "an object".to_owned(),
    })?;
    let components = key_value
        .get("components")
        .ok_with(|| FtVoiceError::Field {
            path: "key.components".to_owned(),
            expected: "an object".to_owned(),
        })?;
    let string_field = |name: &str| -> Result<String, FtVoiceError> {
        components
            .get(name)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_with(|| FtVoiceError::Field {
                path: format!("key.components.{name}"),
                expected: "a string".to_owned(),
            })
    };
    let option_string_field = |name: &str| -> Result<Option<String>, FtVoiceError> {
        match components.get(name) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(value) => {
                value
                    .as_str()
                    .map(str::to_owned)
                    .map(Some)
                    .ok_with(|| FtVoiceError::Field {
                        path: format!("key.components.{name}"),
                        expected: "a string or null".to_owned(),
                    })
            }
        }
    };
    let key = FtVoiceCacheKey {
        voice_recipe_hash: string_field("voice_recipe_hash")?,
        model_hash: string_field("model_hash")?,
        prompt_builder_version: components
            .get("prompt_builder_version")
            .and_then(serde_json::Value::as_u64)
            .ok_with(|| FtVoiceError::Field {
                path: "key.components.prompt_builder_version".to_owned(),
                expected: "a non-negative integer".to_owned(),
            })? as u32,
        streaming_mode: string_field("streaming_mode")?,
        quant_recipe: string_field("quant_recipe")?,
        math_mode: string_field("math_mode")?,
        engine_abi: components
            .get("engine_abi")
            .and_then(serde_json::Value::as_u64)
            .ok_with(|| FtVoiceError::Field {
                path: "key.components.engine_abi".to_owned(),
                expected: "a non-negative integer".to_owned(),
            })? as u32,
        language_id: string_field("language_id")?,
        speaker_embed_sha256: string_field("speaker_embed_sha256")?,
        ref_transcript_tokens_sha256: option_string_field("ref_transcript_tokens_sha256")?,
        ref_codec_codes_sha256: option_string_field("ref_codec_codes_sha256")?,
    };
    let stored_key = key_value
        .get("cache_key")
        .and_then(serde_json::Value::as_str)
        .ok_with(|| FtVoiceError::Field {
            path: "key.cache_key".to_owned(),
            expected: "a string".to_owned(),
        })?;
    let computed_key = key.cache_key();
    if stored_key != computed_key {
        return Err(FtVoiceError::DigestMismatch {
            section: "key".to_owned(),
            expected: stored_key.to_owned(),
            actual: computed_key,
        });
    }

    let mut blobs = BTreeMap::new();
    if let Some(entries) = object.get("sections").and_then(serde_json::Value::as_array) {
        if entries.len() > MAX_SECTIONS {
            return Err(FtVoiceError::LimitExceeded {
                what: "blobs",
                found: entries.len(),
                limit: MAX_SECTIONS,
            });
        }
        let file_len = bytes.len() as u64;
        for entry in entries {
            let name = entry
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].name".to_owned(),
                    expected: "a string".to_owned(),
                })?;
            let offset = entry
                .get("offset")
                .and_then(serde_json::Value::as_u64)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].offset".to_owned(),
                    expected: "a non-negative integer".to_owned(),
                })?;
            let length = entry
                .get("length")
                .and_then(serde_json::Value::as_u64)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].length".to_owned(),
                    expected: "a non-negative integer".to_owned(),
                })?;
            let sha256 = entry
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .ok_with(|| FtVoiceError::Field {
                    path: "sections[].sha256".to_owned(),
                    expected: "a string".to_owned(),
                })?;
            let end = offset
                .checked_add(length)
                .ok_with(|| FtVoiceError::RangeOutOfBounds {
                    what: format!("blob `{name}`"),
                    offset,
                    length,
                    bound: file_len,
                })?;
            if end > file_len {
                return Err(FtVoiceError::RangeOutOfBounds {
                    what: format!("blob `{name}`"),
                    offset,
                    length,
                    bound: file_len,
                });
            }
            let payload_bytes = &bytes[offset as usize..end as usize];
            let actual = hex_digest(&digest(payload_bytes));
            if actual != sha256 {
                return Err(FtVoiceError::DigestMismatch {
                    section: name.to_owned(),
                    expected: sha256.to_owned(),
                    actual,
                });
            }
            if blobs
                .insert(name.to_owned(), payload_bytes.to_vec())
                .is_some()
            {
                return Err(FtVoiceError::DuplicateSection {
                    name: name.to_owned(),
                });
            }
        }
    }
    Ok(FtVoiceCache { key, blobs })
}

// A tiny local combinator so `ok_or_else(...)?` chains stay readable at this density.
trait OkWith<T, E> {
    fn ok_with<F: FnOnce() -> E>(self, error: F) -> Result<T, E>;
}

impl<T, E> OkWith<T, E> for Option<T> {
    fn ok_with<F: FnOnce() -> E>(self, error: F) -> Result<T, E> {
        self.ok_or_else(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_embedding() -> Vec<f32> {
        (0..SPEAKER_EMBEDDING_LEN)
            .map(|i| (i as f32 * 0.001 - 0.5).sin())
            .collect()
    }

    fn sample_pack(profile: VoiceProfile) -> VoicePack {
        VoicePack {
            profile,
            consent: ConsentAttestation {
                attested: true,
                method: ConsentMethod::Interactive,
            },
            language: Some("en".to_owned()),
            transcript: (profile != VoiceProfile::Minimal)
                .then(|| "Please call Stella.".to_owned()),
            speech_regions: vec![(0, 24_000), (26_000, 48_000)],
            diagnostics: Some(EnrollmentDiagnostics {
                clipping_fraction: 0.0,
                longest_clip_run: 0,
                intersample_overshoot_db: 0.1,
                snr_estimate_db: Some(28.5),
                pause_floor_dbfs: -51.0,
                reverb_time_s: Some(0.31),
                music_bed_likelihood: 0.02,
                stationarity_drift: 0.11,
                loudness_rms_dbfs: -19.7,
                voice_activity_ratio: 0.83,
            }),
            preprocessing: Some(PreprocessingRecipe {
                target_sample_rate_hz: 24_000,
                resampler: "lanczos6".to_owned(),
                denoise: Some(CleanupRecord {
                    engine: "fastenhancer-s".to_owned(),
                    before: -51.8,
                    after: -117.4,
                }),
                dereverb: None,
            }),
            provenance: Provenance {
                engine: "ftts 0.1.8".to_owned(),
                source_audio_sha256: Some("ab".repeat(32)),
                source_frames: Some(48_000),
            },
            embedding: sample_embedding(),
            codec_codes: (profile != VoiceProfile::Minimal).then(|| vec![0, 2_047, 12]),
            reference_audio: (profile == VoiceProfile::Portable).then(|| vec![1u8, 2, 3, 4]),
            section_digests: BTreeMap::new(),
        }
    }

    #[test]
    fn portable_roundtrip_is_byte_identical() {
        let pack = sample_pack(VoiceProfile::Portable);
        let first = serialize_voice_pack(&pack).expect("serialize");
        let parsed = parse_voice_pack(&first).expect("parse");
        let second = serialize_voice_pack(&parsed).expect("reserialize");
        assert_eq!(
            first, second,
            "write-read-write must reproduce identical bytes"
        );
        assert_eq!(parsed.embedding.len(), SPEAKER_EMBEDDING_LEN);
        assert_eq!(parsed.transcript.as_deref(), Some("Please call Stella."));
        assert_eq!(parsed.reference_audio.as_deref(), Some(&[1u8, 2, 3, 4][..]));
        assert!(parsed.consent.attested);
    }

    #[test]
    fn recipe_hash_is_stable_and_content_sensitive() {
        let pack = sample_pack(VoiceProfile::Private);
        let hash_a = pack.recipe_hash().expect("hash");
        let reparsed =
            parse_voice_pack(&serialize_voice_pack(&pack).expect("bytes")).expect("parse");
        assert_eq!(hash_a, reparsed.recipe_hash().expect("hash"));
        let mut different = pack.clone();
        different.embedding[512] += 0.125;
        assert_ne!(hash_a, different.recipe_hash().expect("hash"));
    }

    #[test]
    fn minimal_profile_refuses_identity_blocks() {
        // Writer side refuses construction.
        let mut minimal = sample_pack(VoiceProfile::Minimal);
        minimal.transcript = Some("sneaked".to_owned());
        assert!(matches!(
            serialize_voice_pack(&minimal),
            Err(FtVoiceError::ProfileViolation { .. })
        ));
        let private_bytes =
            serialize_voice_pack(&sample_pack(VoiceProfile::Private)).expect("bytes");
        let (version, mut directory, payload) = container_parts(&private_bytes);
        directory["profile"] = "minimal".into();
        let flipped_profile = assemble(VOICE_MAGIC, version, &directory, &payload);
        assert!(matches!(
            parse_voice_pack(&flipped_profile),
            Err(FtVoiceError::ProfileViolation { .. })
        ));
    }

    #[test]
    fn private_profile_refuses_embedded_audio() {
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Portable)).expect("bytes");
        let (version, mut directory, payload) = container_parts(&bytes);
        // Relabel honestly-formatted portable content as private: same sections, new claim.
        directory["profile"] = "private".into();
        let relabeled = assemble(VOICE_MAGIC, version, &directory, &payload);
        assert!(matches!(
            parse_voice_pack(&relabeled),
            Err(FtVoiceError::ProfileViolation {
                profile: VoiceProfile::Private,
                ..
            })
        ));
    }

    /// Splits a serialized container into `(version, directory-json, payload)`.
    fn container_parts(bytes: &[u8]) -> (u32, serde_json::Value, Vec<u8>) {
        let version = u32::from_le_bytes(bytes[8..12].try_into().expect("4 bytes"));
        let directory_len = u64::from_le_bytes(bytes[12..20].try_into().expect("8 bytes")) as usize;
        let directory: serde_json::Value =
            serde_json::from_slice(&bytes[20..20 + directory_len]).expect("directory");
        (version, directory, bytes[20 + directory_len..].to_vec())
    }

    /// Reassembles a container from parts, recomputing the length prefix.
    fn assemble(
        magic: &[u8; 8],
        version: u32,
        directory: &serde_json::Value,
        payload: &[u8],
    ) -> Vec<u8> {
        let directory_bytes = directory.to_string().into_bytes();
        let mut out = Vec::with_capacity(
            HEADER_PREFIX_BYTES as usize + directory_bytes.len() + payload.len(),
        );
        out.extend_from_slice(magic);
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&(directory_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&directory_bytes);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn truncation_at_every_boundary_is_refused() {
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Portable)).expect("bytes");
        // Header prefix, then every boundary inside directory and payload.
        for cut in 0..bytes.len() {
            let result = parse_voice_pack(&bytes[..cut]);
            assert!(result.is_err(), "truncation to {cut} bytes was accepted");
        }
    }

    #[test]
    fn single_bit_flips_are_named_refusals() {
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Portable)).expect("bytes");
        // Magic.
        let mut corrupt = bytes.clone();
        corrupt[3] ^= 0x01;
        assert!(matches!(
            parse_voice_pack(&corrupt),
            Err(FtVoiceError::BadMagic { .. })
        ));
        // Version.
        let mut corrupt = bytes.clone();
        corrupt[8] ^= 0x80;
        assert!(matches!(
            parse_voice_pack(&corrupt),
            Err(FtVoiceError::UnsupportedVersion { .. })
        ));
        // Directory length (huge).
        let mut corrupt = bytes.clone();
        corrupt[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            parse_voice_pack(&corrupt),
            Err(FtVoiceError::DirectoryLength { .. })
        ));
        // A byte inside the directory: JSON becomes malformed OR a digest/profile mismatch —
        // anything EXCEPT silent success.
        let mut corrupt = bytes.clone();
        let mid_directory = HEADER_PREFIX_BYTES as usize + 4;
        corrupt[mid_directory] = b'!';
        assert!(parse_voice_pack(&corrupt).is_err());
        // A byte inside the embedding payload: digest mismatch.
        let mut corrupt = bytes.clone();
        *corrupt.last_mut().expect("payload") ^= 0xFF;
        assert!(matches!(
            parse_voice_pack(&corrupt),
            Err(FtVoiceError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn nan_embedding_refused_on_both_paths() {
        let mut pack = sample_pack(VoiceProfile::Minimal);
        pack.embedding[0] = f32::NAN;
        assert!(matches!(
            serialize_voice_pack(&pack),
            Err(FtVoiceError::NonFiniteEmbedding { .. })
        ));
        // Reader-side: patch the last embedding float to a NaN bit pattern, then refresh the
        // section digest in the directory so ONLY the finiteness check can fire.
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Minimal)).expect("bytes");
        let (version, mut directory, mut payload) = container_parts(&bytes);
        let last = payload.len() - 4;
        payload[last..].copy_from_slice(&0x7F80_0001u32.to_le_bytes()); // signaling NaN
        let sections = directory["sections"]
            .as_array_mut()
            .expect("sections array");
        sections[0]["sha256"] = crate::sha256::hex_digest(&crate::sha256::digest(&payload)).into();
        let rebuilt = assemble(VOICE_MAGIC, version, &directory, &payload);
        let last_index = SPEAKER_EMBEDDING_LEN - 1;
        match parse_voice_pack(&rebuilt) {
            Err(FtVoiceError::NonFiniteEmbedding { index }) if index == last_index => {}
            other => panic!("expected NonFiniteEmbedding at {last_index}, got {other:?}"),
        }
    }

    #[test]
    fn wrong_embedding_length_refused() {
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Minimal)).expect("bytes");
        let truncated = &bytes[..bytes.len() - 8]; // two floats short
        assert!(parse_voice_pack(truncated).is_err());
    }

    #[test]
    fn future_version_refused() {
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Minimal)).expect("bytes");
        let mut bumped = bytes.clone();
        bumped[8..12].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            parse_voice_pack(&bumped),
            Err(FtVoiceError::UnsupportedVersion { found: 2, .. })
        ));
    }

    #[test]
    fn unknown_profile_refused() {
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Minimal)).expect("bytes");
        let (version, mut directory, payload) = container_parts(&bytes);
        directory["profile"] = "mystical".into();
        let tampered = assemble(VOICE_MAGIC, version, &directory, &payload);
        assert!(matches!(
            parse_voice_pack(&tampered),
            Err(FtVoiceError::UnknownValue { path, .. }) if path == "profile"
        ));
    }

    #[test]
    fn unattested_consent_roundtrips_inspectably() {
        let mut pack = sample_pack(VoiceProfile::Private);
        pack.consent = ConsentAttestation {
            attested: false,
            method: ConsentMethod::Flag,
        };
        let parsed = parse_voice_pack(&serialize_voice_pack(&pack).expect("bytes")).expect("parse");
        assert!(
            !parsed.consent.attested,
            "inspection of consent state must survive"
        );
        assert_eq!(parsed.consent.method, ConsentMethod::Flag);
    }

    #[test]
    fn codec_code_out_of_space_refused() {
        let mut pack = sample_pack(VoiceProfile::Private);
        pack.codec_codes = Some(vec![2_048]);
        assert!(matches!(
            serialize_voice_pack(&pack),
            Err(FtVoiceError::CodecCodeOutOfRange {
                index: 0,
                code: 2_048
            })
        ));
    }

    #[test]
    fn reversed_speech_region_refused() {
        let mut pack = sample_pack(VoiceProfile::Minimal);
        pack.speech_regions = vec![(100, 50)];
        assert!(matches!(
            serialize_voice_pack(&pack),
            Err(FtVoiceError::SpeechRegionInvalid { .. })
        ));
    }

    #[test]
    fn cache_container_roundtrip_and_key_disagreement_detected() {
        let mut blobs = BTreeMap::new();
        blobs.insert("header_kv".to_owned(), vec![9u8; 37]);
        blobs.insert("codec_state".to_owned(), vec![7u8; 11]);
        let key = FtVoiceCacheKey {
            voice_recipe_hash: "recipe".to_owned(),
            model_hash: "model".to_owned(),
            prompt_builder_version: 3,
            streaming_mode: "streaming".to_owned(),
            quant_recipe: "int8-default".to_owned(),
            math_mode: "strict".to_owned(),
            engine_abi: 1,
            language_id: "en".to_owned(),
            speaker_embed_sha256: "embed".to_owned(),
            ref_transcript_tokens_sha256: Some("tokens".to_owned()),
            ref_codec_codes_sha256: None,
        };
        let bytes = serialize_ftvoice_cache(&key, &blobs).expect("serialize");
        let parsed = parse_ftvoice_cache(&bytes).expect("parse");
        assert_eq!(parsed.key, key);
        assert_eq!(parsed.blobs, blobs);

        // Tamper with a COMPONENT (not the stored digest): recomputation must catch it.
        let mut corrupt = bytes.clone();
        let needle = b"\"model_hash\":\"model\"";
        let start = corrupt
            .windows(needle.len())
            .position(|window| window == needle)
            .expect("needle");
        corrupt[start..start + needle.len()].copy_from_slice(b"\"model_hash\":\"modeL\"");
        assert!(matches!(
            parse_ftvoice_cache(&corrupt),
            Err(FtVoiceError::DigestMismatch { section, .. }) if section == "key"
        ));
    }

    #[test]
    fn every_cache_key_component_invalidates_the_digest() {
        let base = FtVoiceCacheKey {
            voice_recipe_hash: "r".into(),
            model_hash: "m".into(),
            prompt_builder_version: 1,
            streaming_mode: "streaming".into(),
            quant_recipe: "q".into(),
            math_mode: "strict".into(),
            engine_abi: 1,
            language_id: "en".into(),
            speaker_embed_sha256: "e".into(),
            ref_transcript_tokens_sha256: Some("t".into()),
            ref_codec_codes_sha256: Some("c".into()),
        };
        let base_digest = base.cache_key();
        let variations: Vec<(String, FtVoiceCacheKey)> = vec![
            (
                "voice_recipe_hash".into(),
                FtVoiceCacheKey {
                    voice_recipe_hash: "r2".into(),
                    ..base.clone()
                },
            ),
            (
                "model_hash".into(),
                FtVoiceCacheKey {
                    model_hash: "m2".into(),
                    ..base.clone()
                },
            ),
            (
                "prompt_builder_version".into(),
                FtVoiceCacheKey {
                    prompt_builder_version: 2,
                    ..base.clone()
                },
            ),
            (
                "streaming_mode".into(),
                FtVoiceCacheKey {
                    streaming_mode: "non_streaming".into(),
                    ..base.clone()
                },
            ),
            (
                "quant_recipe".into(),
                FtVoiceCacheKey {
                    quant_recipe: "q2".into(),
                    ..base.clone()
                },
            ),
            (
                "math_mode".into(),
                FtVoiceCacheKey {
                    math_mode: "fast".into(),
                    ..base.clone()
                },
            ),
            (
                "engine_abi".into(),
                FtVoiceCacheKey {
                    engine_abi: 2,
                    ..base.clone()
                },
            ),
            (
                "language_id".into(),
                FtVoiceCacheKey {
                    language_id: "zh".into(),
                    ..base.clone()
                },
            ),
            (
                "speaker_embed_sha256".into(),
                FtVoiceCacheKey {
                    speaker_embed_sha256: "e2".into(),
                    ..base.clone()
                },
            ),
            (
                "ref_transcript_tokens".into(),
                FtVoiceCacheKey {
                    ref_transcript_tokens_sha256: Some("t2".into()),
                    ..base.clone()
                },
            ),
            (
                "ref_transcript_absent".into(),
                FtVoiceCacheKey {
                    ref_transcript_tokens_sha256: None,
                    ..base.clone()
                },
            ),
            (
                "ref_codec_codes".into(),
                FtVoiceCacheKey {
                    ref_codec_codes_sha256: Some("c2".into()),
                    ..base.clone()
                },
            ),
            (
                "ref_codec_absent".into(),
                FtVoiceCacheKey {
                    ref_codec_codes_sha256: None,
                    ..base.clone()
                },
            ),
        ];
        for (component, variation) in variations {
            assert_ne!(
                variation.cache_key(),
                base_digest,
                "changing {component} did not invalidate the cache key"
            );
        }
    }

    #[test]
    fn cache_truncation_refused() {
        let mut blobs = BTreeMap::new();
        blobs.insert("state".to_owned(), vec![1u8; 64]);
        let key = FtVoiceCacheKey::default();
        let bytes = serialize_ftvoice_cache(&key, &blobs).expect("serialize");
        for cut in 0..bytes.len() {
            assert!(parse_ftvoice_cache(&bytes[..cut]).is_err(), "cut={cut}");
        }
    }
    /// Manual-inspection hook: with `FTTS_SMOKE_DIR` set, writes a valid portable pack there so
    /// `ftts voice inspect` can be exercised against a real file. Without the variable this is a
    /// no-op dev aid, not a gate.
    #[test]
    fn smoke_writes_pack_for_manual_cli_inspection() {
        let Some(dir) = std::env::var("FTTS_SMOKE_DIR").ok() else {
            eprintln!(
                "skipped: set FTTS_SMOKE_DIR to write a sample .ftvoice for manual inspection"
            );
            return;
        };
        let bytes = serialize_voice_pack(&sample_pack(VoiceProfile::Portable)).expect("serialize");
        std::fs::create_dir_all(&dir).expect("create smoke dir");
        let path = std::path::Path::new(&dir).join("smoke.ftvoice");
        std::fs::write(&path, bytes).expect("write pack");
        println!("wrote {}", path.display());
    }
}
