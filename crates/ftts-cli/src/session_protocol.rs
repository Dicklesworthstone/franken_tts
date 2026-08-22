//! The `ftts talk` session wire contract: schema v2.
//!
//! A long-lived session speaks a sibling vocabulary to the one-shot run contract
//! ([`crate::robot`], schema v1): ops arrive on stdin, events leave on stdout under an
//! envelope that adds `session_id` and a monotonic `seq`, and raw PCM flows on a separate
//! channel sequenced by the `audio` events' `byte_offset`/`bytes`. The v1 contract is frozen
//! byte-for-byte by its conformance fixture, so the session vocabulary is versioned
//! separately (v2) instead of growing v1 — the two catalogues are disjoint by design.
//!
//! Everything here is contract, not runtime: the session process itself is bead
//! frankentts-edz0. This module pins the shapes that runtime must speak, validates them
//! through the same strict-closed walk as v1 ([`robot::validate_object`] — parameterized,
//! never forked), freezes the self-description in a conformance fixture, and pins the
//! per-utterance seed derivation with fixed vectors.
//!
//! Bead: frankentts-e3zz.

use serde_json::{Value, json};

use crate::robot::{EventSpec, FieldSpec, Kind, Stream, WireContract, validate_object};

/// The session contract's version, carried by every emitted object.
pub const SESSION_SCHEMA_VERSION: u8 = 2;

/// Envelope fields on every session event, beyond v1's `schema_version` + `event`.
pub const SESSION_COMMON_FIELDS: &[FieldSpec] = &[
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
    FieldSpec {
        name: "session_id",
        ty: "string",
        required: true,
        summary: "correlates the events of one session process",
    },
    FieldSpec {
        name: "seq",
        ty: "u64",
        required: true,
        summary: "monotonically increasing per stdout line within the session",
    },
];

/// A named session event, as a constructor for a partially-filled object (the same ergonomic
/// shape as the run contract's [`crate::robot::EventType`], so `schema_version`, `event`,
/// `session_id`, and `seq` cannot be forgotten at a call site).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionEvent {
    SessionStart,
    Ack,
    ContextOpen,
    SpeakStart,
    Audio,
    Progress,
    Buffer,
    TextUnderrun,
    SpeakCancelled,
    SpeakComplete,
    ContextClosed,
    SessionEnd,
    SessionError,
    SessionSchema,
}

impl SessionEvent {
    pub const fn name(self) -> &'static str {
        match self {
            Self::SessionStart => "session_start",
            Self::Ack => "ack",
            Self::ContextOpen => "context_open",
            Self::SpeakStart => "speak_start",
            Self::Audio => "audio",
            Self::Progress => "progress",
            Self::Buffer => "buffer",
            Self::TextUnderrun => "text_underrun",
            Self::SpeakCancelled => "speak_cancelled",
            Self::SpeakComplete => "speak_complete",
            Self::ContextClosed => "context_closed",
            Self::SessionEnd => "session_end",
            Self::SessionError => "session_error",
            Self::SessionSchema => "session_schema",
        }
    }

    /// An object carrying the envelope, ready for this event's own fields.
    pub fn object(self, session_id: &str, seq: u64) -> serde_json::Map<String, Value> {
        let mut object = serde_json::Map::new();
        object.insert(
            "schema_version".to_owned(),
            json!(SESSION_SCHEMA_VERSION),
        );
        object.insert("event".to_owned(), json!(self.name()));
        object.insert("session_id".to_owned(), json!(session_id));
        object.insert("seq".to_owned(), json!(seq));
        object
    }

    /// The catalogue entry for this event.
    pub fn spec(self) -> &'static EventSpec {
        SESSION_EVENTS
            .iter()
            .find(|spec| spec.name == self.name())
            .expect("every SessionEvent variant is catalogued")
    }
}

/// The client-to-server vocabulary: one JSON object per stdin line.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionOp {
    Open,
    Say,
    Flush,
    Cancel,
    Close,
    Shutdown,
}

impl SessionOp {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Say => "say",
            Self::Flush => "flush",
            Self::Cancel => "cancel",
            Self::Close => "close",
            Self::Shutdown => "shutdown",
        }
    }
}

const CONTEXT: FieldSpec = FieldSpec {
    name: "context",
    ty: "string",
    required: true,
    summary: "names one conversation context within the session",
};

const OP_ID: FieldSpec = FieldSpec {
    name: "id",
    ty: "string",
    required: false,
    summary: "optional client correlation id, echoed in the matching ack or session_error",
};

pub const SESSION_EVENTS: &[EventSpec] = &[
    EventSpec {
        name: "session_start",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "emitted once when the model is loaded and the session accepts ops",
        fields: &[
            FieldSpec {
                name: "version",
                ty: "string",
                required: true,
                summary: "binary version of the serving process",
            },
            FieldSpec {
                name: "model",
                ty: "string",
                required: true,
                summary: "model digest or artifact identifier in force",
            },
            FieldSpec {
                name: "route",
                ty: "string",
                required: true,
                summary: "kernel route identifier (e.g. the int8 default)",
            },
            FieldSpec {
                name: "pcm",
                ty: "object",
                required: true,
                summary: "audio shape: {format: \"s16le\", rate: 24000, channels: 1}",
            },
            FieldSpec {
                name: "pid",
                ty: "u64",
                required: true,
                summary: "process id of the session, for lifecycle management",
            },
        ],
    },
    EventSpec {
        name: "ack",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "acceptance receipt for one well-formed op",
        fields: &[
            FieldSpec {
                name: "id",
                ty: "string|null",
                required: true,
                summary: "the op's client id, or null when the op carried none",
            },
            FieldSpec {
                name: "op",
                ty: "string",
                required: true,
                summary: "the op name being acknowledged",
            },
            FieldSpec {
                name: "context",
                ty: "string|null",
                required: true,
                summary: "the op's context, or null for context-less ops (shutdown)",
            },
        ],
    },
    EventSpec {
        name: "context_open",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "a context exists and its voice is resolved",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "voice",
                ty: "string",
                required: true,
                summary: "resolved voice descriptor: preset name or file path",
            },
            FieldSpec {
                name: "seed",
                ty: "u64",
                required: true,
                summary: "the context seed in force; per-utterance seeds derive from it",
            },
        ],
    },
    EventSpec {
        name: "speak_start",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "synthesis began for one utterance",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "utterance",
                ty: "u64",
                required: true,
                summary: "per-context utterance counter, starting at 0",
            },
            FieldSpec {
                name: "seed",
                ty: "u64",
                required: true,
                summary: "the effective seed for this utterance (derived or overridden)",
            },
        ],
    },
    EventSpec {
        name: "audio",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "one packet of PCM was handed to the audio channel",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "utterance",
                ty: "u64",
                required: true,
                summary: "which utterance the packet belongs to",
            },
            FieldSpec {
                name: "byte_offset",
                ty: "u64",
                required: true,
                summary: "cumulative offset of this packet's first byte in the session-global \
                          PCM stream",
            },
            FieldSpec {
                name: "bytes",
                ty: "u64",
                required: true,
                summary: "packet size in bytes on the audio channel",
            },
            FieldSpec {
                name: "frames",
                ty: "u64",
                required: true,
                summary: "80 ms codec frames the packet carries",
            },
            FieldSpec {
                name: "frame_index",
                ty: "u64",
                required: true,
                summary: "index of the packet's first frame within its utterance",
            },
            FieldSpec {
                name: "ttfa_ms",
                ty: "u64",
                required: false,
                summary: "time to first audible sample, present only on an utterance's first \
                          packet",
            },
        ],
    },
    EventSpec {
        name: "progress",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "rate-limited (~1/s) generation progress for a speaking utterance",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "utterance",
                ty: "u64",
                required: true,
                summary: "which utterance is progressing",
            },
            FieldSpec {
                name: "frames_emitted",
                ty: "u64",
                required: true,
                summary: "codec frames delivered so far",
            },
            FieldSpec {
                name: "text_consumed_tokens",
                ty: "u64",
                required: true,
                summary: "tokens the generator has already consumed",
            },
            FieldSpec {
                name: "text_buffered_tokens",
                ty: "u64",
                required: true,
                summary: "tokens appended but not yet consumed",
            },
        ],
    },
    EventSpec {
        name: "buffer",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "audio-channel depth: the orchestrator's pacing signal",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "queued_ms",
                ty: "u64",
                required: true,
                summary: "delivered-but-unconsumed audio, in milliseconds",
            },
        ],
    },
    EventSpec {
        name: "text_underrun",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "the generator ran out of text and waited for the client",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "waited_ms",
                ty: "u64",
                required: true,
                summary: "how long synthesis stalled awaiting text",
            },
        ],
    },
    EventSpec {
        name: "speak_cancelled",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "frame-boundary stop with a truncation receipt",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "utterance",
                ty: "u64",
                required: true,
                summary: "which utterance was cut short",
            },
            FieldSpec {
                name: "frames_delivered",
                ty: "u64",
                required: true,
                summary: "codec frames delivered before the stop",
            },
            FieldSpec {
                name: "audio_ms",
                ty: "u64",
                required: true,
                summary: "audio delivered before the stop",
            },
            FieldSpec {
                name: "text_spoken_tokens",
                ty: "u64",
                required: true,
                summary: "tokens whose text is inside the delivered audio",
            },
            FieldSpec {
                name: "spoken_text",
                ty: "string",
                required: true,
                summary: "tokenizer-decoded prefix actually delivered, so the orchestrator can \
                          rewrite its turn to what the user heard",
            },
        ],
    },
    EventSpec {
        name: "speak_complete",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "an utterance finished on its own (EOS)",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "utterance",
                ty: "u64",
                required: true,
                summary: "which utterance completed",
            },
            FieldSpec {
                name: "frames",
                ty: "u64",
                required: true,
                summary: "total codec frames",
            },
            FieldSpec {
                name: "audio_ms",
                ty: "u64",
                required: true,
                summary: "total audio duration",
            },
            FieldSpec {
                name: "ttfa_ms",
                ty: "u64",
                required: true,
                summary: "time to first audible sample for this utterance",
            },
            FieldSpec {
                name: "rtf",
                ty: "number",
                required: true,
                summary: "real-time factor: synthesis wall time over audio duration",
            },
        ],
    },
    EventSpec {
        name: "context_closed",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "a context is finished; its id may be reused by a later open",
        fields: &[CONTEXT],
    },
    EventSpec {
        name: "session_end",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "the session drained and is exiting cleanly",
        fields: &[],
    },
    EventSpec {
        name: "session_error",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "a rejected op or a session-level problem; the session survives op errors",
        fields: &[
            FieldSpec {
                name: "kind",
                ty: "string",
                required: true,
                summary: "stable error class the client can branch on",
            },
            FieldSpec {
                name: "message",
                ty: "string",
                required: true,
                summary: "what happened, naming the offending value",
            },
            FieldSpec {
                name: "remediation",
                ty: "string",
                required: true,
                summary: "the action that fixes it",
            },
        ],
    },
    EventSpec {
        name: "session_schema",
        kind: Kind::Reply,
        stream: Stream::Events,
        summary: "self-description of this contract, printed by `ftts robot schema session`",
        fields: &[
            FieldSpec {
                name: "contract",
                ty: "string",
                required: true,
                summary: "always \"session\"; distinguishes the reply from the v1 run schema",
            },
            FieldSpec {
                name: "events",
                ty: "array",
                required: true,
                summary: "the event catalogue",
            },
            FieldSpec {
                name: "ops",
                ty: "array",
                required: true,
                summary: "the op catalogue",
            },
            FieldSpec {
                name: "stdin_contract",
                ty: "string",
                required: true,
                summary: "how a client drives the session",
            },
            FieldSpec {
                name: "audio_contract",
                ty: "string",
                required: true,
                summary: "how PCM reaches the client and how it is sequenced",
            },
            FieldSpec {
                name: "seed_derivation",
                ty: "string",
                required: true,
                summary: "the pinned per-utterance seed function",
            },
            FieldSpec {
                name: "environment_variables",
                ty: "array",
                required: true,
                summary: "environment knobs the session reads",
            },
            FieldSpec {
                name: "exit_codes",
                ty: "object",
                required: true,
                summary: "every documented exit code, including the session-transport code",
            },
        ],
    },
];

pub const SESSION_OPS: &[EventSpec] = &[
    EventSpec {
        name: "open",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "create a context with a resolved voice and a context seed",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "voice",
                ty: "string",
                required: false,
                summary: "preset name, .spk path, or voice-card image path; default voice when \
                          absent",
            },
            FieldSpec {
                name: "seed",
                ty: "u64",
                required: false,
                summary: "context seed; derived from the session default when absent",
            },
            OP_ID,
        ],
    },
    EventSpec {
        name: "say",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "speak text in a context; continue:true holds the utterance open for appends",
        fields: &[
            CONTEXT,
            FieldSpec {
                name: "text",
                ty: "string",
                required: true,
                summary: "text to speak; chunks concatenate verbatim across says",
            },
            FieldSpec {
                name: "continue",
                ty: "bool",
                required: false,
                summary: "keep the utterance open for further say ops (default false)",
            },
            FieldSpec {
                name: "seed",
                ty: "u64",
                required: false,
                summary: "per-utterance seed override",
            },
            OP_ID,
        ],
    },
    EventSpec {
        name: "flush",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "release any withheld partial word and finish the open utterance",
        fields: &[CONTEXT, OP_ID],
    },
    EventSpec {
        name: "cancel",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "stop the speaking utterance at the next frame boundary",
        fields: &[CONTEXT, OP_ID],
    },
    EventSpec {
        name: "close",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "finish and remove a context",
        fields: &[CONTEXT, OP_ID],
    },
    EventSpec {
        name: "shutdown",
        kind: Kind::Stream,
        stream: Stream::Events,
        summary: "drain and exit the session cleanly",
        fields: &[OP_ID],
    },
];

/// The v2 session event contract.
pub fn session_contract() -> WireContract {
    WireContract {
        version: Some(SESSION_SCHEMA_VERSION),
        discriminator: "event",
        common: SESSION_COMMON_FIELDS,
        catalogue: SESSION_EVENTS,
    }
}

/// The session op contract: unversioned client-to-server objects, strict-closed all the same.
pub fn session_op_contract() -> WireContract {
    WireContract {
        version: None,
        discriminator: "op",
        common: &[],
        catalogue: SESSION_OPS,
    }
}

/// Check one session event against the v2 catalogue. An empty result means it conforms.
pub fn validate_session_event(value: &Value) -> Vec<String> {
    validate_object(value, &session_contract())
}

/// Check one client op against the v2 catalogue. An empty result means it conforms.
pub fn validate_session_op(value: &Value) -> Vec<String> {
    validate_object(value, &session_op_contract())
}

/// The pinned per-utterance seed derivation, part of the frozen contract.
///
/// `effective = splitmix64(context_seed ^ utterance_index)`, with splitmix64 exactly as
/// published (Steele, Lea, Flood 2014): add the golden gamma, then two xor-multiply rounds.
/// Deriving per utterance (instead of reusing the context seed verbatim) makes every
/// utterance independently reproducible given `(context_seed, N)` or its override; a fixed
/// derivation (instead of a std hasher) makes a replay on a future build derive the same
/// seeds. Fixed input/output vectors are asserted in this module's tests.
#[must_use]
pub fn utterance_seed(context_seed: u64, utterance_index: u64) -> u64 {
    splitmix64(context_seed ^ utterance_index)
}

fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The full exit-code map, including the session-transport code that postdates the frozen
/// v1 schema document. The v1 map stays byte-frozen; this one is the talk process's truth.
fn session_exit_codes() -> std::collections::BTreeMap<String, String> {
    use crate::error::FttsExitCode;
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
        FttsExitCode::SessionTransport,
    ]
    .into_iter()
    .map(|code| (code.as_u8().to_string(), code.description().to_owned()))
    .collect()
}

/// The machine-readable self-description printed by `ftts robot schema session`.
pub fn session_schema_document(environment_variables: &[&str]) -> Value {
    let render = |contract: &WireContract| {
        contract
            .catalogue
            .iter()
            .map(|spec| {
                let fields: Vec<Value> = contract
                    .common
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
            .collect::<Vec<Value>>()
    };
    let contract = session_contract();
    let ops = session_op_contract();

    let mut object = SessionEvent::SessionSchema.object("schema", 0);
    object.insert("contract".to_owned(), json!("session"));
    object.insert("events".to_owned(), json!(render(&contract)));
    object.insert("ops".to_owned(), json!(render(&ops)));
    object.insert(
        "stdin_contract".to_owned(),
        json!("one JSON object per line; unknown fields are rejected; a malformed line yields \
               session_error and the session survives"),
    );
    object.insert(
        "audio_contract".to_owned(),
        json!("raw PCM s16le mono 24 kHz on a separate channel; nothing but PCM ever appears \
               there; audio events' byte_offset/bytes are the sequencing authority"),
    );
    object.insert(
        "seed_derivation".to_owned(),
        json!("effective_seed(context_seed, utterance_index) = splitmix64(context_seed ^ \
               utterance_index), pinned with fixed vectors"),
    );
    object.insert(
        "environment_variables".to_owned(),
        json!(environment_variables),
    );
    object.insert("exit_codes".to_owned(), json!(session_exit_codes()));
    Value::Object(object)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The derivation is a frozen contract: these vectors are load-bearing for replay.
    #[test]
    fn utterance_seed_matches_its_pinned_vectors() {
        let cases = [
            (0_u64, 0_u64, 0xe220_a839_7b1d_cdaf_u64),
            (7, 1, 0xbd64_a5d9_adef_e000),
            (0xDEAD_BEEF, 42, 0xdf4d_f98c_dad6_1b87),
            (u64::MAX, 999, 0xb256_9106_6218_4be2),
        ];
        for (context_seed, index, expected) in cases {
            assert_eq!(
                utterance_seed(context_seed, index),
                expected,
                "context_seed {context_seed:#x}, utterance {index}"
            );
        }
    }

    fn minimal_event(name: &str) -> Value {
        let spec = SESSION_EVENTS
            .iter()
            .find(|spec| spec.name == name)
            .expect("catalogued event");
        let mut object = SessionEvent::SessionSchema.object("s", 0);
        object.insert("event".to_owned(), json!(name));
        for field in spec.fields.iter().filter(|field| field.required) {
            let value = match field.ty.strip_suffix("|null") {
                Some(_) => Value::Null,
                None => match field.ty {
                    "string" => json!("x"),
                    "bool" => json!(true),
                    "object" => json!({}),
                    "array" => json!([]),
                    "number" => json!(1.5),
                    _ => json!(1),
                },
            };
            object.insert(field.name.to_owned(), value);
        }
        Value::Object(object)
    }

    fn minimal_op(name: &str) -> Value {
        let spec = SESSION_OPS
            .iter()
            .find(|spec| spec.name == name)
            .expect("catalogued op");
        let mut object = serde_json::Map::new();
        object.insert("op".to_owned(), json!(name));
        for field in spec.fields.iter().filter(|field| field.required) {
            object.insert(
                field.name.to_owned(),
                match field.ty {
                    "string" => json!("x"),
                    _ => json!(1),
                },
            );
        }
        Value::Object(object)
    }

    #[test]
    fn every_catalogued_session_event_validates_minimally() {
        for spec in SESSION_EVENTS {
            let problems = validate_session_event(&minimal_event(spec.name));
            assert!(problems.is_empty(), "{}: {problems:?}", spec.name);
        }
    }

    #[test]
    fn every_catalogued_session_op_validates_minimally() {
        for spec in SESSION_OPS {
            let problems = validate_session_op(&minimal_op(spec.name));
            assert!(problems.is_empty(), "{}: {problems:?}", spec.name);
        }
    }

    #[test]
    fn session_events_reject_unknown_fields_and_wrong_envelopes() {
        let mut event = minimal_event("buffer");
        event
            .as_object_mut()
            .expect("object")
            .insert("extra".to_owned(), json!(1));
        let problems = validate_session_event(&event);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("unknown field `extra`")),
            "{problems:?}"
        );

        // A v1-stamped object is not a session event, and vice versa.
        let mut wrong_version = minimal_event("ack");
        wrong_version["schema_version"] = json!(1);
        assert!(
            validate_session_event(&wrong_version)
                .iter()
                .any(|problem| problem.contains("schema_version")),
            "a v1 stamp must not validate as v2"
        );
    }

    #[test]
    fn session_ops_reject_unknown_fields_and_require_their_discriminator() {
        let mut op = minimal_op("open");
        op.as_object_mut()
            .expect("object")
            .insert("voice".to_owned(), json!("matt"));
        assert!(validate_session_op(&op).is_empty());

        op.as_object_mut()
            .expect("object")
            .insert("bogus".to_owned(), json!(true));
        let problems = validate_session_op(&op);
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("unknown field `bogus`")),
            "{problems:?}"
        );

        let problems = validate_session_op(&json!({"context": "c1"}));
        assert!(
            problems
                .iter()
                .any(|problem| problem.contains("`op` discriminator")),
            "{problems:?}"
        );
    }

    #[test]
    fn the_schema_document_satisfies_its_own_contract() {
        let document = session_schema_document(&[]);
        let problems = validate_session_event(&document);
        assert!(problems.is_empty(), "{problems:?}");
    }

    #[test]
    fn the_session_exit_code_map_is_total() {
        let map = session_exit_codes();
        let all = [
            crate::error::FttsExitCode::Success,
            crate::error::FttsExitCode::Generic,
            crate::error::FttsExitCode::Usage,
            crate::error::FttsExitCode::ModelNotFound,
            crate::error::FttsExitCode::Input,
            crate::error::FttsExitCode::BudgetTimeout,
            crate::error::FttsExitCode::Cancelled,
            crate::error::FttsExitCode::ArtifactFormat,
            crate::error::FttsExitCode::EnrollmentQualityRefusal,
            crate::error::FttsExitCode::SessionTransport,
        ];
        for code in all {
            assert!(
                map.contains_key(&code.as_u8().to_string()),
                "exit code {} ({}) is missing from the session document",
                code.as_u8(),
                code.description()
            );
        }
        assert!(map.contains_key("9"), "the transport code must be published");
    }
}
