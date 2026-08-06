//! The robot NDJSON contract — defined once, consumed three ways.
//!
//! Agent users are the primary consumers of `ftts` (goal G5), so the event stream is a
//! promised interface, not incidental logging. The catalogue below is the *only* place that
//! definition lives:
//!
//! 1. `ftts robot schema` serialises it via [`schema_document`], so the machine-readable self-description cannot drift
//!    from what the binary actually emits;
//! 2. [`validate_event`] checks emitted objects against it, so an event that grew a field or
//!    lost one fails a test rather than silently changing an agent's parse;
//! 3. the frozen fixture in `ftts-conformance` pins it, so a schema change is a deliberate
//!    fixture update and never a silent one.
//!
//! Validation is **strict in both directions**: a missing required field and an *unknown* field
//! are both violations. A contract that only checks for absence lets the surface grow silently,
//! which is the failure mode that breaks downstream parsers.
//!
//! Bead: frankentts-p0-robot-82c.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::error::FttsExitCode;

/// Version carried by every emitted object. Bump deliberately; the frozen fixture will fail.
pub const SCHEMA_VERSION: u8 = 1;

/// Environment recognised by the CLI, as reported by `robot schema`.
///
/// It lives beside the catalogue rather than in `lib.rs` because it is part of the promised
/// agent-facing contract, and the frozen fixture pins it: adding a variable without documenting
/// it here fails the contract test.
pub const DOCUMENTED_ENVIRONMENT: &[&str] = &[
    "FTTS_MODEL_DIR",
    "FTTS_THREADS",
    "FTTS_PROFILE",
    "FTTS_PACKET_FRAMES",
    "FTTS_MATH_MODE",
    "FTTS_QUANT",
    "FTTS_FORCE_ARCH",
    "FTTS_NUMA",
    "FTTS_STAGE_BUDGET_SYNTHESIS_MS",
    "FTTS_STAGE_BUDGET_ENROLL_MS",
];

/// Which stream an object is written to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stream {
    /// Written to stdout, unless `say --stream raw` has given stdout to PCM.
    Events,
    /// Always stderr: the human-facing error channel mirrors the machine one.
    Stderr,
}

impl Stream {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Events => "events",
            Self::Stderr => "stderr",
        }
    }
}

/// Stream events describe a run in progress; replies answer a one-shot query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Stream,
    Reply,
}

impl Kind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Stream => "stream",
            Self::Reply => "reply",
        }
    }
}

/// One field of one event.
pub struct FieldSpec {
    pub name: &'static str,
    /// A tiny type language: `u64`, `i64`, `u8`, `bool`, `string`, `object`, `array`, and
    /// `<ty>|null` for a nullable field. Deliberately small — this describes a wire contract,
    /// not a type system.
    pub ty: &'static str,
    pub required: bool,
    pub summary: &'static str,
}

pub struct EventSpec {
    pub name: &'static str,
    pub kind: Kind,
    pub stream: Stream,
    pub summary: &'static str,
    pub fields: &'static [FieldSpec],
}

/// Present on every object, so the per-event lists below do not repeat them.
pub const COMMON_FIELDS: &[FieldSpec] = &[
    FieldSpec {
        name: "schema_version",
        ty: "u8",
        required: true,
        summary: "contract version; pinned by the frozen fixture",
    },
    FieldSpec {
        name: "event",
        ty: "string",
        required: true,
        summary: "discriminator naming this object's type",
    },
];

pub const EVENTS: &[EventSpec] = &[
    EventSpec {
        name: "run_start",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "opens every run; every later event in the run repeats its run_id",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "correlates the events of one run",
            },
            FieldSpec {
                name: "command",
                ty: "string",
                required: true,
                summary: "subcommand being run",
            },
            FieldSpec {
                name: "profile",
                ty: "string",
                required: true,
                summary: "execution profile in force",
            },
            FieldSpec {
                name: "packet_frames",
                ty: "string",
                required: true,
                summary: "streaming packet size",
            },
            FieldSpec {
                name: "math_mode",
                ty: "string",
                required: true,
                summary: "strict or fast",
            },
            FieldSpec {
                name: "stateless",
                ty: "bool",
                required: true,
                summary: "true unless durable tracing was opted into",
            },
            FieldSpec {
                name: "seed",
                ty: "u64|null",
                required: true,
                summary: "sampler seed, null when unset",
            },
            FieldSpec {
                name: "model",
                ty: "string|null",
                required: true,
                summary: "resolved model artifact, null before resolution",
            },
            FieldSpec {
                name: "voice",
                ty: "string|null",
                required: true,
                summary: "resolved voice pack, null when none",
            },
        ],
    },
    EventSpec {
        name: "stage",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "one pipeline stage began or ended",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "name",
                ty: "string",
                required: true,
                summary: "stage name",
            },
            FieldSpec {
                name: "seq",
                ty: "u64",
                required: true,
                summary: "monotonic index within the run",
            },
            FieldSpec {
                name: "state",
                ty: "string",
                required: true,
                summary: "started or finished",
            },
            FieldSpec {
                name: "elapsed_ms",
                ty: "u64",
                required: true,
                summary: "milliseconds since run_start",
            },
            FieldSpec {
                name: "budget_ms",
                ty: "u64|null",
                required: true,
                summary: "stage budget, null when unbounded",
            },
        ],
    },
    EventSpec {
        name: "frame",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "coarse decode progress; throttled, never one per frame",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "index",
                ty: "u64",
                required: true,
                summary: "frames emitted so far",
            },
            FieldSpec {
                name: "total_estimate",
                ty: "u64|null",
                required: true,
                summary: "predicted total, null when unknown",
            },
            FieldSpec {
                name: "elapsed_ms",
                ty: "u64",
                required: true,
                summary: "milliseconds since run_start",
            },
        ],
    },
    EventSpec {
        name: "audio_chunk",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "PCM was written to the sink; the bytes themselves never appear here",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "byte_offset",
                ty: "u64",
                required: true,
                summary: "offset of this chunk within the stream",
            },
            FieldSpec {
                name: "bytes",
                ty: "u64",
                required: true,
                summary: "chunk length in bytes",
            },
            FieldSpec {
                name: "duration_ms",
                ty: "u64",
                required: true,
                summary: "audio duration of this chunk",
            },
            FieldSpec {
                name: "packet_frames",
                ty: "string",
                required: true,
                summary: "packet size that produced it",
            },
            FieldSpec {
                name: "sink",
                ty: "string",
                required: true,
                summary: "where the PCM went: file, fd, or stdout",
            },
        ],
    },
    EventSpec {
        name: "health",
        kind: Kind::Reply,
        stream: Stream::Events,
        summary: "readiness snapshot; also answers `robot health`",
        fields: &[
            FieldSpec {
                name: "status",
                ty: "string",
                required: true,
                summary: "coarse readiness state",
            },
            FieldSpec {
                name: "model_loaded",
                ty: "bool",
                required: true,
                summary: "whether a model artifact is resident",
            },
            FieldSpec {
                name: "model_present",
                ty: "bool",
                required: true,
                summary: "artifact found by magic-bytes sniff, never a tensor load",
            },
            FieldSpec {
                name: "model_path",
                ty: "string|null",
                required: true,
                summary: "resolved artifact path when present",
            },
            FieldSpec {
                name: "model_dir",
                ty: "string|null",
                required: true,
                summary: "FTTS_MODEL_DIR as configured, null when unset",
            },
            FieldSpec {
                name: "searched",
                ty: "array",
                required: true,
                summary: "every directory consulted; makes resolution failures actionable",
            },
            FieldSpec {
                name: "stateless_default",
                ty: "bool",
                required: true,
                summary: "no synthesis history is persisted by default",
            },
            FieldSpec {
                name: "threads",
                ty: "u64|null",
                required: true,
                summary: "configured worker count, null when unset",
            },
            FieldSpec {
                name: "recommended_command",
                ty: "string",
                required: true,
                summary: "next actionable command",
            },
        ],
    },
    EventSpec {
        name: "run_complete",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "closes a successful run",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "exit_code",
                ty: "u8",
                required: true,
                summary: "process exit code this run will produce",
            },
            FieldSpec {
                name: "elapsed_ms",
                ty: "u64",
                required: true,
                summary: "total run duration",
            },
            FieldSpec {
                name: "frames",
                ty: "u64",
                required: true,
                summary: "frames produced",
            },
            FieldSpec {
                name: "audio_bytes",
                ty: "u64",
                required: true,
                summary: "PCM bytes written",
            },
        ],
    },
    EventSpec {
        name: "health_violation",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "a runtime-health detector fired mid-run (ftts_core::health)",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "violation",
                ty: "string",
                required: true,
                summary: "stable violation class, e.g. non_finite or output_silent",
            },
            FieldSpec {
                name: "detail",
                ty: "string",
                required: true,
                summary: "the specific occurrence, including the locating scalars",
            },
            FieldSpec {
                name: "remedy",
                ty: "string",
                required: true,
                summary: "the concrete next action, not a restatement of the problem",
            },
            FieldSpec {
                name: "invalidates_output",
                ty: "bool",
                required: true,
                summary: "false for a kernel demotion or thermal report — those runs are still correct",
            },
            FieldSpec {
                name: "elapsed_ms",
                ty: "u64",
                required: true,
                summary: "milliseconds since run_start",
            },
        ],
    },
    EventSpec {
        name: "run_error",
        kind: Kind::Stream,
        stream: Stream::Stderr,
        summary: "closes a failed run and carries the exit code the process will return",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "exit_code",
                ty: "u8",
                required: true,
                summary: "process exit code; matches the stable table in `robot schema`",
            },
            FieldSpec {
                name: "kind",
                ty: "string",
                required: true,
                summary: "stable error class",
            },
            FieldSpec {
                name: "message",
                ty: "string",
                required: true,
                summary: "what went wrong",
            },
            FieldSpec {
                name: "remediation",
                ty: "string",
                required: true,
                summary: "the concrete next command to try",
            },
            FieldSpec {
                name: "elapsed_ms",
                ty: "u64",
                required: true,
                summary: "run duration before the failure",
            },
        ],
    },
    EventSpec {
        name: "text_prepared",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "text normalization finished; reports SHAPE and PROVENANCE only, never the text",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "normalize",
                ty: "string",
                required: true,
                summary: "normalization mode in force",
            },
            FieldSpec {
                name: "unicode_version",
                ty: "string",
                required: true,
                summary: "Unicode version this build normalizes against",
            },
            FieldSpec {
                name: "char_count",
                ty: "u64",
                required: true,
                summary: "input length in characters; a size, never the content",
            },
            FieldSpec {
                name: "trace_requested",
                ty: "bool",
                required: true,
                summary: "whether a normalization trace was requested",
            },
        ],
    },
    EventSpec {
        name: "check_complete",
        kind: Kind::Reply,
        stream: Stream::Events,
        summary: "answers `say --check`: resolution and admission without synthesis",
        fields: &[
            FieldSpec {
                name: "run_id",
                ty: "string",
                required: true,
                summary: "owning run",
            },
            FieldSpec {
                name: "model",
                ty: "string",
                required: true,
                summary: "resolved model artifact",
            },
            FieldSpec {
                name: "voice",
                ty: "string|null",
                required: true,
                summary: "resolved voice pack",
            },
            FieldSpec {
                name: "profile",
                ty: "string",
                required: true,
                summary: "execution profile",
            },
            FieldSpec {
                name: "packet_frames",
                ty: "string",
                required: true,
                summary: "streaming packet size",
            },
            FieldSpec {
                name: "math_mode",
                ty: "string",
                required: true,
                summary: "strict or fast",
            },
            FieldSpec {
                name: "voice_pack",
                ty: "string",
                required: true,
                summary: "voice-pack privacy profile",
            },
            FieldSpec {
                name: "normalize",
                ty: "string",
                required: true,
                summary: "text-transformation mode",
            },
            FieldSpec {
                name: "normalization_trace_requested",
                ty: "bool",
                required: true,
                summary: "whether an emitted normalization trace was asked for",
            },
            FieldSpec {
                name: "seed",
                ty: "u64|null",
                required: true,
                summary: "sampler seed",
            },
            FieldSpec {
                name: "trace",
                ty: "string|null",
                required: true,
                summary: "opt-in trace path",
            },
            FieldSpec {
                name: "output",
                ty: "string|null",
                required: true,
                summary: "audio sink path",
            },
            FieldSpec {
                name: "admission",
                ty: "object",
                required: true,
                summary: "resource-admission decision",
            },
        ],
    },
    EventSpec {
        name: "robot_schema",
        kind: Kind::Reply,
        stream: Stream::Events,
        summary: "answers `robot schema`: this catalogue, machine-readably",
        fields: &[
            FieldSpec {
                name: "events",
                ty: "array",
                required: true,
                summary: "every object type this binary can emit",
            },
            FieldSpec {
                name: "stdout_contract",
                ty: "string",
                required: true,
                summary: "what owns stdout",
            },
            FieldSpec {
                name: "raw_stream_contract",
                ty: "string",
                required: true,
                summary: "stream split under --stream raw",
            },
            FieldSpec {
                name: "environment_variables",
                ty: "array",
                required: true,
                summary: "recognised environment",
            },
            FieldSpec {
                name: "exit_codes",
                ty: "object",
                required: true,
                summary: "stable exit-code table",
            },
        ],
    },
    EventSpec {
        name: "backends",
        kind: Kind::Reply,
        stream: Stream::Events,
        summary: "answers `robot backends`: detected ISA tier, kernel plan, pool sizing",
        fields: &[
            FieldSpec {
                name: "available",
                ty: "array",
                required: true,
                summary: "kernel tiers this build can dispatch",
            },
            FieldSpec {
                name: "dispatched",
                ty: "string|null",
                required: true,
                summary: "tier actually selected, null before selection",
            },
            FieldSpec {
                name: "isa_features",
                ty: "array",
                required: true,
                summary: "CPU features detected at runtime",
            },
            FieldSpec {
                name: "kernel_plan",
                ty: "string|null",
                required: true,
                summary: "autotuned plan identity, null when unpacked",
            },
            FieldSpec {
                name: "pool_sizing",
                ty: "object|null",
                required: true,
                summary: "USL-derived worker counts, null when unmeasured",
            },
            FieldSpec {
                name: "force_arch",
                ty: "string|null",
                required: true,
                summary: "FTTS_FORCE_ARCH override",
            },
        ],
    },
    EventSpec {
        name: "selftest",
        kind: Kind::Reply,
        stream: Stream::Events,
        summary: "answers `robot selftest`: shipped kernel proofs, including the i32-overflow rows",
        fields: &[
            FieldSpec {
                name: "status",
                ty: "string",
                required: true,
                summary: "passed, failed, or skipped",
            },
            FieldSpec {
                name: "reason",
                ty: "string|null",
                required: true,
                summary: "why it skipped; null when it ran",
            },
            FieldSpec {
                name: "checks",
                ty: "array",
                required: true,
                summary: "per-check outcomes",
            },
        ],
    },
    EventSpec {
        name: "voice_inspect",
        kind: Kind::Reply,
        stream: Stream::Events,
        summary: "answers `voice inspect`",
        fields: &[
            FieldSpec {
                name: "path",
                ty: "string",
                required: true,
                summary: "voice pack inspected",
            },
            FieldSpec {
                name: "status",
                ty: "string",
                required: true,
                summary: "inspection outcome",
            },
        ],
    },
];

/// Look up one event's specification by name.
pub fn event_spec(name: &str) -> Option<&'static EventSpec> {
    EVENTS.iter().find(|spec| spec.name == name)
}

/// A named event, as a constructor for a partially-filled object.
///
/// Emitters build an object with the common fields already correct and then insert their own,
/// so `schema_version` and the `event` discriminator can never be forgotten or mistyped at a
/// call site. (This ergonomic shape is preserved from a concurrent implementation of this
/// contract that I destroyed — see the incident note on frankentts-p0-robot-82c.)
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventType {
    RunStart,
    Stage,
    Frame,
    AudioChunk,
    Health,
    RunComplete,
    RunError,
    HealthViolation,
    CheckComplete,
    TextPrepared,
    Backends,
    Selftest,
    VoiceInspect,
}

impl EventType {
    pub const fn name(self) -> &'static str {
        match self {
            Self::RunStart => "run_start",
            Self::Stage => "stage",
            Self::Frame => "frame",
            Self::AudioChunk => "audio_chunk",
            Self::Health => "health",
            Self::RunComplete => "run_complete",
            Self::RunError => "run_error",
            Self::HealthViolation => "health_violation",
            Self::CheckComplete => "check_complete",
            Self::TextPrepared => "text_prepared",
            Self::Backends => "backends",
            Self::Selftest => "selftest",
            Self::VoiceInspect => "voice_inspect",
        }
    }

    /// An object carrying the common fields, ready for this event's own fields.
    pub fn event(self) -> serde_json::Map<String, Value> {
        let mut object = serde_json::Map::new();
        object.insert("schema_version".to_owned(), json!(SCHEMA_VERSION));
        object.insert("event".to_owned(), json!(self.name()));
        object
    }

    /// The catalogue entry for this event.
    pub fn spec(self) -> &'static EventSpec {
        event_spec(self.name()).expect("every EventType variant is catalogued")
    }
}

/// Run-scoped emitter state: the shared `run_id` and the run's own clock.
///
/// Every event in a run repeats its `run_id`, and the lifecycle events carry `elapsed_ms` measured
/// from the same origin, so an agent can stitch a run together from either stream without guessing.
/// The id is derived from the process id and a monotonic start stamp: unique per run, cheap, and
/// carrying nothing about the user's text or filesystem — the CLI is stateless by default and the
/// run id must not become a back door for run history.
#[derive(Debug)]
pub struct RunContext {
    run_id: String,
    started: std::time::Instant,
}

impl RunContext {
    /// Adopt an explicit id — used by tests so assertions are not clock-dependent.
    #[must_use]
    pub fn with_id(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            started: std::time::Instant::now(),
        }
    }

    /// Derive a fresh id for this process invocation.
    #[must_use]
    pub fn generate() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.subsec_nanos());
        Self::with_id(format!("r{:x}{:08x}", std::process::id(), nanos))
    }

    /// The id every event in this run repeats.
    #[must_use]
    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    /// Milliseconds since the run began.
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// An object carrying the common fields plus this run's id.
    #[must_use]
    pub fn event(&self, event_type: EventType) -> serde_json::Map<String, Value> {
        let mut object = event_type.event();
        object.insert("run_id".to_owned(), json!(self.run_id));
        object
    }
}

fn matches_type(value: &Value, ty: &str) -> bool {
    if let Some(inner) = ty.strip_suffix("|null") {
        return value.is_null() || matches_type(value, inner);
    }
    match ty {
        "string" => value.is_string(),
        "bool" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "u8" => value.as_u64().is_some_and(|n| n <= u64::from(u8::MAX)),
        "u64" => value.as_u64().is_some(),
        "i64" => value.as_i64().is_some(),
        _ => false,
    }
}

/// Check one emitted object against the catalogue. An empty result means it conforms.
///
/// Unknown fields are violations, not warnings: a contract that only checks for absence lets
/// the surface grow silently and breaks downstream parsers at their leisure.
pub fn validate_event(value: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let Some(object) = value.as_object() else {
        problems.push("emitted object is not a JSON object".to_owned());
        return problems;
    };

    match object.get("schema_version").and_then(Value::as_u64) {
        Some(version) if version == u64::from(SCHEMA_VERSION) => {}
        Some(version) => problems.push(format!(
            "schema_version {version} != contract version {SCHEMA_VERSION}"
        )),
        None => problems.push("missing schema_version".to_owned()),
    }

    let Some(name) = object.get("event").and_then(Value::as_str) else {
        problems.push("missing or non-string `event` discriminator".to_owned());
        return problems;
    };
    let Some(spec) = event_spec(name) else {
        problems.push(format!(
            "unknown event {name:?}; the catalogue defines: {}",
            EVENTS
                .iter()
                .map(|spec| spec.name)
                .collect::<Vec<_>>()
                .join(", ")
        ));
        return problems;
    };

    for field in COMMON_FIELDS.iter().chain(spec.fields.iter()) {
        match object.get(field.name) {
            Some(value) => {
                if !matches_type(value, field.ty) {
                    problems.push(format!(
                        "{name}.{}: expected {} but found {value}",
                        field.name, field.ty
                    ));
                }
            }
            None if field.required => {
                problems.push(format!("{name}: missing required field `{}`", field.name));
            }
            None => {}
        }
    }

    let known: Vec<&str> = COMMON_FIELDS
        .iter()
        .chain(spec.fields.iter())
        .map(|field| field.name)
        .collect();
    for key in object.keys() {
        if !known.contains(&key.as_str()) {
            problems.push(format!(
                "{name}: unknown field `{key}`; extending the contract requires a catalogue \
                 entry and a frozen-fixture update"
            ));
        }
    }

    problems
}

/// Validate a whole NDJSON stream, reporting the line number of each violation.
pub fn validate_ndjson(stream: &str) -> Vec<String> {
    let mut problems = Vec::new();
    for (index, line) in stream.lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() {
            problems.push(format!(
                "line {line_number}: blank line in an NDJSON stream"
            ));
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(value) => problems.extend(
                validate_event(&value)
                    .into_iter()
                    .map(|problem| format!("line {line_number}: {problem}")),
            ),
            Err(error) => problems.push(format!("line {line_number}: not valid JSON: {error}")),
        }
    }
    problems
}

fn exit_codes_json() -> BTreeMap<String, String> {
    [
        FttsExitCode::Success,
        FttsExitCode::Generic,
        FttsExitCode::Usage,
        FttsExitCode::ModelNotFound,
        FttsExitCode::Input,
        FttsExitCode::BudgetTimeout,
        FttsExitCode::Cancelled,
        FttsExitCode::ArtifactFormat,
        FttsExitCode::EnrollmentQualityRefusal,
    ]
    .into_iter()
    .map(|code| (code.as_u8().to_string(), code.description().to_owned()))
    .collect()
}

/// The machine-readable self-description emitted by `ftts robot schema`.
/// Render an engine health signal as its robot event.
///
/// The single conversion point between `ftts_core::health` and the wire, so the two crates cannot
/// disagree about what a violation is called or whether it invalidates the run. The wire strings
/// and the `invalidates_output` predicate come from the engine types rather than being restated
/// here — restating them is how a detector and its report drift apart.
///
/// `detail` is the violation's `Display`, which carries the locating scalars (which seam, which
/// index, how many milliseconds), and `remedy` is the action. An agent needs both: the class tells
/// it what happened, the remedy tells it what to do, and neither substitutes for the other.
#[must_use]
pub fn health_violation_event(
    run_id: &str,
    event: ftts_core::HealthEvent,
    elapsed_ms: u64,
) -> Value {
    let mut object = EventType::HealthViolation.event();
    object.insert("run_id".to_owned(), json!(run_id));
    object.insert("violation".to_owned(), json!(event.as_str()));
    object.insert(
        "invalidates_output".to_owned(),
        json!(event.invalidates_output()),
    );
    object.insert("elapsed_ms".to_owned(), json!(elapsed_ms));
    let (detail, remedy) = match event {
        ftts_core::HealthEvent::Violation(violation) => (violation.to_string(), violation.remedy()),
        ftts_core::HealthEvent::BudgetExceeded => (
            "a stage exceeded its configured budget".to_owned(),
            "raise FTTS_STAGE_BUDGET_SYNTHESIS_MS or shorten the request; the partial result is \
             not a completed utterance",
        ),
        ftts_core::HealthEvent::Cancelled => (
            "the run observed cooperative cancellation".to_owned(),
            "this is the caller's own cancellation taking effect; the audio stops at a frame \
             boundary and is truncated, not finished",
        ),
    };
    object.insert("detail".to_owned(), json!(detail));
    object.insert("remedy".to_owned(), json!(remedy));
    Value::Object(object)
}

pub fn schema_document(environment_variables: &[&str]) -> Value {
    let events: Vec<Value> = EVENTS
        .iter()
        .map(|spec| {
            let fields: Vec<Value> = COMMON_FIELDS
                .iter()
                .chain(spec.fields.iter())
                .map(|field| {
                    json!({
                        "name": field.name,
                        "type": field.ty,
                        "required": field.required,
                        "summary": field.summary,
                    })
                })
                .collect();
            json!({
                "name": spec.name,
                "kind": spec.kind.as_str(),
                "stream": spec.stream.as_str(),
                "summary": spec.summary,
                "fields": fields,
            })
        })
        .collect();

    json!({
        "schema_version": SCHEMA_VERSION,
        "event": "robot_schema",
        "events": events,
        "stdout_contract": "one JSON object per line; stdout carries events unless `say --stream raw` gives stdout to PCM",
        "raw_stream_contract": "under `say --stream raw`, PCM owns stdout and every event goes to stderr; the two are never interleaved on one stream",
        "environment_variables": environment_variables,
        "exit_codes": exit_codes_json(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal(name: &str) -> Value {
        let spec = event_spec(name).expect("known event");
        let mut object = serde_json::Map::new();
        object.insert("schema_version".to_owned(), json!(SCHEMA_VERSION));
        object.insert("event".to_owned(), json!(name));
        for field in spec.fields.iter().filter(|field| field.required) {
            let value = match field.ty.strip_suffix("|null") {
                Some(_) => Value::Null,
                None => match field.ty {
                    "string" => json!("x"),
                    "bool" => json!(true),
                    "object" => json!({}),
                    "array" => json!([]),
                    _ => json!(1),
                },
            };
            object.insert(field.name.to_owned(), value);
        }
        Value::Object(object)
    }

    #[test]
    fn every_catalogued_event_validates_when_minimally_populated() {
        for spec in EVENTS {
            let problems = validate_event(&minimal(spec.name));
            assert!(problems.is_empty(), "{}: {problems:?}", spec.name);
        }
    }

    #[test]
    fn a_missing_required_field_is_a_violation() {
        let mut event = minimal("run_complete");
        event.as_object_mut().expect("object").remove("exit_code");
        let problems = validate_event(&event);
        assert!(
            problems.iter().any(|problem| problem.contains("exit_code")),
            "{problems:?}"
        );
    }

    #[test]
    fn an_unknown_field_is_a_violation() {
        // The direction that matters: a surface may not grow without a catalogue entry.
        let mut event = minimal("health");
        event
            .as_object_mut()
            .expect("object")
            .insert("undeclared".to_owned(), json!(1));
        let problems = validate_event(&event);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("undeclared")),
            "{problems:?}"
        );
    }

    #[test]
    fn a_wrong_type_is_a_violation() {
        let mut event = minimal("frame");
        event
            .as_object_mut()
            .expect("object")
            .insert("index".to_owned(), json!("not a number"));
        let problems = validate_event(&event);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("frame.index")),
            "{problems:?}"
        );
    }

    #[test]
    fn a_stale_schema_version_is_a_violation() {
        let mut event = minimal("run_start");
        event
            .as_object_mut()
            .expect("object")
            .insert("schema_version".to_owned(), json!(SCHEMA_VERSION + 1));
        assert!(!validate_event(&event).is_empty());
    }

    #[test]
    fn nullable_fields_accept_both_null_and_their_type() {
        let mut event = minimal("run_start");
        assert!(validate_event(&event).is_empty());
        event
            .as_object_mut()
            .expect("object")
            .insert("seed".to_owned(), json!(7));
        assert!(validate_event(&event).is_empty());
    }

    #[test]
    fn schema_output_describes_itself_and_conforms() {
        let schema = schema_document(&["FTTS_MODEL_DIR"]);
        assert!(
            validate_event(&schema).is_empty(),
            "robot schema must satisfy its own contract"
        );
        let names: Vec<&str> = schema["events"]
            .as_array()
            .expect("events array")
            .iter()
            .map(|event| event["name"].as_str().expect("name"))
            .collect();
        for required in [
            "run_start",
            "stage",
            "frame",
            "audio_chunk",
            "health",
            "run_complete",
            "run_error",
        ] {
            assert!(names.contains(&required), "catalogue is missing {required}");
        }
    }

    #[test]
    fn every_event_type_variant_is_catalogued_and_prefilled() {
        // Guards the EventType <-> EVENTS correspondence in both directions: a new catalogue
        // entry without a variant is fine, but a variant whose name is not catalogued would
        // panic in spec() at runtime, which is exactly when nobody is looking.
        for variant in [
            EventType::RunStart,
            EventType::Stage,
            EventType::Frame,
            EventType::AudioChunk,
            EventType::Health,
            EventType::RunComplete,
            EventType::RunError,
            EventType::CheckComplete,
            EventType::TextPrepared,
            EventType::Backends,
            EventType::Selftest,
            EventType::VoiceInspect,
        ] {
            let spec = variant.spec();
            assert_eq!(spec.name, variant.name());
            let object = variant.event();
            assert_eq!(object["schema_version"], json!(SCHEMA_VERSION));
            assert_eq!(object["event"], json!(variant.name()));
        }
    }

    #[test]
    fn ndjson_validation_reports_line_numbers() {
        let stream = format!(
            "{}\n{{\"schema_version\":1,\"event\":\"nope\"}}\n",
            serde_json::to_string(&minimal("health")).expect("json")
        );
        let problems = validate_ndjson(&stream);
        assert!(
            problems
                .iter()
                .any(|problem| problem.starts_with("line 2:")),
            "{problems:?}"
        );
    }
}
