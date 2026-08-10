#![forbid(unsafe_code)]

//! Shared, stateless command-line dispatch for both FrankenTTS binaries.

mod error;
pub mod resident;
pub mod robot;
pub mod style;
pub mod synth;

pub use error::{FttsError, FttsExitCode};
pub use robot::{EventType, validate_event, validate_ndjson};

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::OnceLock;

#[cfg(test)]
use clap::CommandFactory;
use clap::{Parser, Subcommand, ValueEnum};
use ftts_artifacts::census::{ExpectedTensor, WeightsManifest};
use ftts_artifacts::converter::{
    StreamingConversionPlan, TensorConversion, TensorStoragePolicy, convert_safetensors_streaming,
};
use ftts_artifacts::fttsq::{AccessClass, MappedFttsq};
use ftts_artifacts::safetensors::Dtype;
use ftts_core::{NormalizationMode, NormalizationOptions, SynthesisRequest};
use ftts_kernels::mmap::MappedFile;
use serde_json::{Value, json};

const ROBOT_SCHEMA_VERSION: u8 = 1;
const SCAFFOLD_ADMISSION_TEXT_LIMIT_BYTES: usize = 1_048_576;
const MODEL_BASENAME: &str = "qwen3-tts-12hz-0.6b-base.fttsq";

/// Built-in voices: name, one-line character, and the enrolled 1,024-float x-vector.
///
/// "matt" is the out-of-box default when no enrollment exists. Enrolling a real reference
/// always takes precedence over any of them.
const PRESET_VOICES: &[(&str, &str, &[u8])] = &[
    (
        "aria",
        "clear, warm, feminine",
        include_bytes!("../presets/aria.spk"),
    ),
    (
        "ember",
        "the same character a few semitones deeper",
        include_bytes!("../presets/ember.spk"),
    ),
    (
        "james",
        "natural, conversational, masculine",
        include_bytes!("../presets/james.spk"),
    ),
    (
        "matt",
        "warm, easy, masculine — the out-of-box default",
        include_bytes!("../presets/matt.spk"),
    ),
    (
        "leo",
        "relaxed, resonant, masculine",
        include_bytes!("../presets/leo.spk"),
    ),
    (
        "robert",
        "steady, measured, masculine",
        include_bytes!("../presets/robert.spk"),
    ),
    (
        "judy",
        "bright, articulate, feminine",
        include_bytes!("../presets/judy.spk"),
    ),
];

/// The preset used when `--voice`, `FTTS_DEFAULT_VOICE`, and MODEL_DIR/default.spk are all
/// absent, so a fresh install speaks out of the box.
const DEFAULT_PRESET_VOICE: &str = "matt";

/// Names a preset resolves to a temp-materialized `.spk` path the existing voice loaders read.
///
/// Only fires when the value is NOT an existing file, so a file named like a preset still wins.
/// The file is rewritten unconditionally: 4 KB per run is cheaper than trusting stale content.
fn materialize_preset_voice(name: &str) -> Option<Result<PathBuf, FttsError>> {
    let (_, _, bytes) = PRESET_VOICES
        .iter()
        .find(|(preset, _, _)| *preset == name)?;
    let staging_dir = match synth::private_staging_dir() {
        Ok(dir) => dir,
        Err(error) => {
            return Some(Err(FttsError::Generic(format!(
                "cannot create staging directory for preset voice {name}: {error}"
            ))));
        }
    };
    let path = staging_dir.join(format!("ftts-preset-{name}-{}.spk", std::process::id()));
    Some(
        fs::write(&path, bytes)
            .map(|()| path.clone())
            .map_err(|error| {
                FttsError::Generic(format!(
                    "cannot materialize preset voice {name} at {}: {error}",
                    path.display()
                ))
            }),
    )
}

fn preset_names() -> String {
    PRESET_VOICES
        .iter()
        .map(|(name, _, _)| *name)
        .collect::<Vec<_>>()
        .join(", ")
}
const PINNED_MAIN_WEIGHTS_FILENAME: &str = "model.safetensors";
const PINNED_MAIN_WEIGHTS_SHA256: &str =
    "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6";
const PINNED_MODEL_REVISION: &str = "5d83992436eae1d760afd27aff78a71d676296fc";
const PINNED_MAIN_TENSOR_COUNT: usize = 478;
//  Crate-local pinned copies (`pinned/`): `cargo package` cannot ship files outside the crate
//  root, and these three are compile-time product surfaces (the artifact census, the model-dir
//  pin assertion, and the Apache attribution the binary prints). A unit test asserts each copy is
//  byte-identical to the truth-pack canonical whenever the truth pack is present.
const PINNED_TENSOR_INVENTORY: &str = include_str!("../pinned/TENSOR_INVENTORY.json");
const PINNED_MODEL_CONFIG: &str = include_str!("../pinned/model_config.json");
const APACHE_LICENSE: &str = include_str!("../pinned/QWEN_APACHE_LICENSE");
//  The `ftts pull` download contract: which release assets make a complete model directory, and
//  the exact digest each must carry. Embedded so a shipped binary can fetch and verify the model
//  with no network-served manifest to trust.
const PINNED_MODEL_MANIFEST: &str = include_str!("../pinned/model_manifest.json");
/// Subdirectory of `$HOME/.cache` that `ftts pull` fills and model resolution falls back to.
const DEFAULT_MODEL_CACHE_SUBDIR: &str = ".cache/franken_tts/model";
const ENVIRONMENT_VARIABLES: [&str; 11] = [
    "FTTS_MODEL_DIR",
    "FTTS_DEFAULT_VOICE",
    "FTTS_THREADS",
    "FTTS_PROFILE",
    "FTTS_PACKET_FRAMES",
    "FTTS_MATH_MODE",
    "FTTS_QUANT",
    "FTTS_FORCE_ARCH",
    "FTTS_NUMA",
    "FTTS_MAX_FRAMES",
    "FTTS_MEMORY_BUDGET_MB",
];

/// Runs the shared `ftts` / `franken_tts` command-line interface.
pub fn cli_main() -> ExitCode {
    // The optimized route is the DEFAULT everywhere (library-level: see
    // `ftts_kernels::route`). `FTTS_INT8=0` selects the f32 reference route end to end;
    // DISC-003 records the decision and the evidence.

    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let exit_code = match error.kind() {
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => {
                    FttsExitCode::Success
                }
                _ => FttsExitCode::Usage,
            };
            let _ = error.print();
            return exit_code.as_exit_code();
        }
    };

    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    match dispatch(cli, environment(), &mut stdin, &mut stdout, &mut stderr) {
        Ok(()) => FttsExitCode::Success.as_exit_code(),
        Err(error) => {
            let _ = writeln!(stderr, "error: {error}");
            error.exit_code().as_exit_code()
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "ftts",
    version,
    about = "Pure-Rust Qwen3-TTS command-line interface",
    long_about = "FrankenTTS is stateless by default: synthesis history is never persisted. \
                  Use `ftts robot schema` for the versioned NDJSON contract.",
    arg_required_else_help = true
)]
struct Cli {
    /// Execution profile. The default is balanced unless FTTS_PROFILE overrides it.
    #[arg(long, global = true, value_enum)]
    profile: Option<ExecutionProfile>,

    /// Codec packet size in frames. The default is profile-dependent.
    #[arg(long, global = true, value_enum)]
    packet_frames: Option<PacketFrames>,

    /// Math contract used by this invocation.
    #[arg(long, global = true, value_enum)]
    math_mode: Option<MathMode>,

    /// Voice-pack serialization profile used by enrollment.
    #[arg(long, global = true, value_enum)]
    voice_pack: Option<VoicePackProfile>,

    /// Text-normalization policy.
    #[arg(long, global = true, value_enum)]
    normalize: Option<NormalizeMode>,

    /// Request a structured trace from synthesis without default-persisting sensitive text.
    #[arg(long, global = true, value_name = "DIR")]
    trace: Option<PathBuf>,

    /// Reproducibility seed for a future sampler.
    #[arg(long, global = true)]
    seed: Option<u64>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate or synthesize text with an optional voice pack.
    Say(SayArgs),
    /// Synthesize text and render a share-ready branded video of it.
    #[command(name = "make-video")]
    MakeVideo(MakeVideoArgs),
    /// Build a consent-bearing voice pack from reference audio.
    Enroll(EnrollArgs),
    /// Inspect a portable voice pack.
    Voice(VoiceArgs),
    /// Convert pinned source weights into a portable .fttsq artifact.
    Convert(ConvertArgs),
    /// Download and verify the pinned model files into the model directory.
    Pull(PullArgs),
    /// Emit versioned, line-oriented robot contract data.
    Robot(RobotArgs),
    /// Report local configuration and readiness without inference.
    Doctor(DoctorArgs),
    /// Internal: run the resident engine daemon. Spawned by `ftts say`; not for direct use.
    #[command(hide = true, name = "resident-daemon")]
    ResidentDaemon(ResidentDaemonArgs),
}

#[derive(Debug, clap::Args)]
struct ResidentDaemonArgs {
    /// Model directory this daemon serves.
    #[arg(long, value_name = "PATH")]
    bundle_root: PathBuf,
}

#[derive(Debug, clap::Args)]
struct SayArgs {
    /// Text to synthesize. Use `-` to read UTF-8 text from stdin.
    #[arg(value_name = "TEXT")]
    text: Option<String>,

    /// Output file (same as -o). Format follows the extension: .wav is written natively;
    /// .m4a, .mp3, and .flac are encoded from the native WAV by the first available system
    /// encoder (afconvert, ffmpeg, lame, flac).
    #[arg(value_name = "OUTPUT", conflicts_with_all = ["stream", "output"])]
    output_positional: Option<PathBuf>,

    /// Read UTF-8 text from PATH. Use `-` for stdin.
    #[arg(long, value_name = "PATH", conflicts_with = "text")]
    file: Option<PathBuf>,

    /// Explicit .fttsq model artifact. No network lookup is performed.
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Voice source: a .spk vector, reference audio, or a built-in voice name
    /// (matt, james, leo, robert, judy, aria, ember).
    /// Default: MODEL_DIR/default.spk when enrolled, else the built-in "matt".
    #[arg(long, value_name = "PATH|NAME")]
    voice: Option<PathBuf>,

    /// Write WAV output here. Mutually exclusive with raw stdout streaming.
    #[arg(short = 'o', long, value_name = "PATH", conflicts_with = "stream")]
    output: Option<PathBuf>,

    /// Stream raw PCM on stdout; robot events then use stderr.
    #[arg(long, value_enum)]
    stream: Option<StreamMode>,

    /// Parse inputs and run the conservative admission preflight without synthesis.
    #[arg(long)]
    check: bool,

    /// Emit the NDJSON event stream even when stdout is a terminal.
    ///
    /// Piped, redirected and CI runs already get NDJSON — nothing but a terminal gets the human
    /// view — so this exists for a person who wants to watch the machine contract directly.
    #[arg(long)]
    robot: bool,

    /// Load the model in this process instead of using the resident engine.
    ///
    /// By default `ftts say` keeps the loaded model in a background process so the next
    /// invocation starts without the multi-second load, unloading itself after ten idle
    /// minutes (FTTS_RESIDENT_IDLE_SECS overrides). This flag, or FTTS_NO_RESIDENT=1,
    /// opts a run out; results are identical either way.
    #[arg(long)]
    no_resident: bool,
}

#[derive(Debug, clap::Args)]
struct MakeVideoArgs {
    /// Text to synthesize. Use `-` to read UTF-8 text from stdin.
    #[arg(value_name = "TEXT")]
    text: Option<String>,

    /// Output video. `.mp4` uses the first available system encoder (ffmpeg);
    /// `.y4m` renders natively with a `.wav` sibling and needs no encoder.
    #[arg(value_name = "OUTPUT", conflicts_with = "output")]
    output_positional: Option<PathBuf>,

    /// Read UTF-8 text from PATH. Use `-` for stdin.
    #[arg(long, value_name = "PATH", conflicts_with = "text")]
    file: Option<PathBuf>,

    /// Explicit .fttsq model artifact. No network lookup is performed.
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Voice source: a .spk vector, reference audio, or a built-in voice name
    /// (matt, james, leo, robert, judy, aria, ember).
    #[arg(long, value_name = "PATH|NAME")]
    voice: Option<PathBuf>,

    /// Write the video here. Same as the positional OUTPUT.
    #[arg(short = 'o', long, value_name = "PATH")]
    output: Option<PathBuf>,

    /// Skip synthesis and render this existing PCM WAV instead.
    #[arg(long, value_name = "PATH", conflicts_with_all = ["text", "file"])]
    audio: Option<PathBuf>,

    /// Voice name shown on the video. Defaults to the voice's name.
    #[arg(long, value_name = "NAME")]
    label: Option<String>,

    /// Load the model in this process instead of using the resident engine.
    #[arg(long)]
    no_resident: bool,
}

#[derive(Debug, clap::Args)]
struct EnrollArgs {
    /// Reference audio: WAV/FLAC decode natively; m4a/mp3/aac/ogg/opus route through the first
    /// system decoder found (afconvert on macOS, ffmpeg).
    #[arg(value_name = "REFERENCE_AUDIO")]
    reference_audio: PathBuf,

    /// Explicit model directory or .fttsq model artifact. No network lookup is performed.
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Write the enrolled raw 1,024-wide x-vector here. Refuses to overwrite an existing file.
    #[arg(short = 'o', long, value_name = "PATH", conflicts_with = "default")]
    output: Option<PathBuf>,

    /// Write MODEL_DIR/default.spk, which `ftts say` uses when `--voice` is absent.
    #[arg(long, conflicts_with = "output")]
    default: bool,

    /// Explicitly proceed after an enrollment-quality warning where safe.
    #[arg(long)]
    force: bool,

    /// Replace an existing voice at the destination without asking.
    ///
    /// Interactive runs are asked to confirm instead; this is how a script or an agent gives that
    /// consent up front. The displaced voice is copied to `<name>.spk.bak` either way.
    #[arg(long)]
    overwrite: bool,

    /// Remove late reverberation from the reference before enrolling.
    ///
    /// A room is convolutive, so `--denoise` cannot touch it; this is the lever for a reference
    /// that sounds "wet". It matters because the speaker encoder cannot separate voice from room,
    /// so a reverberant reference enrolls the room as part of the speaker and every utterance the
    /// clone speaks is rendered in it. Off by default: it changes the enrolled identity.
    #[arg(long)]
    dereverb: bool,

    /// Clean stationary noise from the reference before enrolling.
    ///
    /// This is the default whenever the neural denoiser's weights are present (`ftts pull`
    /// fetches them; measured on a static-hiss reference, the cleaned enrollment lands
    /// closer to a clean-source enrollment than the raw recording does). Passing the flag
    /// explicitly additionally engages the classic no-weights spectral subtraction when the
    /// weights are absent, where the automatic path would skip cleanup rather than swap in
    /// a different engine unannounced.
    #[arg(long, overrides_with = "no_denoise")]
    denoise: bool,

    /// Enroll the recording exactly as given, with no noise cleanup.
    #[arg(long, overrides_with = "denoise")]
    no_denoise: bool,
}

#[derive(Debug, clap::Args)]
struct VoiceArgs {
    #[command(subcommand)]
    command: VoiceCommand,
}

#[derive(Debug, Subcommand)]
enum VoiceCommand {
    /// Inspect a .ftvoice header without synthesizing.
    Inspect { path: PathBuf },
}

#[derive(Debug, clap::Args)]
struct ConvertArgs {
    /// Pinned source-weight directory or file.
    #[arg(value_name = "SOURCE")]
    source: PathBuf,

    /// Destination .fttsq path. Refuses to overwrite an existing artifact.
    #[arg(short = 'o', long, value_name = "PATH")]
    output: PathBuf,
}

#[derive(Debug, clap::Args)]
struct PullArgs {
    /// Destination model directory. Defaults to FTTS_MODEL_DIR, then ~/.cache/franken_tts/model.
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Re-download every file even when it is already present and verified.
    #[arg(long)]
    force: bool,
}

#[derive(Debug, clap::Args)]
struct RobotArgs {
    #[command(subcommand)]
    command: RobotCommand,
}

#[derive(Clone, Debug, Subcommand)]
enum RobotCommand {
    /// Print the versioned NDJSON event schema.
    Schema,
    /// Print a versioned machine-readable readiness event.
    Health,
    /// Print available backend routes without probing model weights.
    Backends,
    /// Print the self-test state; no unavailable kernel is reported as passing.
    Selftest,
}

#[derive(Debug, clap::Args)]
struct DoctorArgs {
    /// Emit one JSON object on stdout instead of a human-readable report.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ExecutionProfile {
    Interactive,
    Balanced,
    Throughput,
    Strict,
}

impl ExecutionProfile {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Balanced => "balanced",
            Self::Throughput => "throughput",
            Self::Strict => "strict",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PacketFrames {
    #[value(name = "1")]
    One,
    #[value(name = "2")]
    Two,
    #[value(name = "4")]
    Four,
    Auto,
}

impl PacketFrames {
    const fn as_str(self) -> &'static str {
        match self {
            Self::One => "1",
            Self::Two => "2",
            Self::Four => "4",
            Self::Auto => "auto",
        }
    }

    /// Codec frames carried by one PCM packet.
    ///
    /// `auto` resolves to 4 here. The autotuner that will choose it per machine is
    /// `frankentts-k-packet-tuning-28u`; until it exists, `auto` means "the balanced default"
    /// rather than a number this call site invented on the spot.
    const fn frames_per_packet(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
            Self::Four | Self::Auto => 4,
        }
    }

    /// Samples in one packet: frames times the codec's 1,920 samples per 80 ms frame.
    const fn samples_per_packet(self) -> usize {
        self.frames_per_packet() as usize * ftts_core::audio::SAMPLES_PER_FRAME
    }

    const fn default_for(profile: ExecutionProfile) -> Self {
        match profile {
            ExecutionProfile::Interactive => Self::One,
            ExecutionProfile::Balanced => Self::Four,
            ExecutionProfile::Throughput => Self::Auto,
            ExecutionProfile::Strict => Self::Four,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MathMode {
    Strict,
    Fast,
}

impl MathMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Strict => "strict",
            Self::Fast => "fast",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum VoicePackProfile {
    Portable,
    Private,
    Minimal,
}

impl VoicePackProfile {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Portable => "portable",
            Self::Private => "private",
            Self::Minimal => "minimal",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum NormalizeMode {
    Verbatim,
    Conservative,
    LocaleAware,
}

impl NormalizeMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Verbatim => "verbatim",
            Self::Conservative => "conservative",
            Self::LocaleAware => "locale-aware",
        }
    }
}

impl From<NormalizeMode> for NormalizationMode {
    fn from(mode: NormalizeMode) -> Self {
        match mode {
            NormalizeMode::Verbatim => Self::Verbatim,
            NormalizeMode::Conservative => Self::Conservative,
            NormalizeMode::LocaleAware => Self::LocaleAware,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum StreamMode {
    Raw,
}

#[derive(Debug, Default)]
struct Environment {
    values: BTreeMap<&'static str, Option<OsString>>,
    stage_budget_values: BTreeMap<OsString, OsString>,
}

impl Environment {
    fn from_process() -> Self {
        let values = ENVIRONMENT_VARIABLES
            .into_iter()
            .map(|name| (name, std::env::var_os(name)))
            .collect();
        let stage_budget_values = std::env::vars_os()
            .filter(|(name, _)| {
                name.to_str().is_some_and(|name| {
                    name.starts_with("FTTS_STAGE_BUDGET_") && name.ends_with("_MS")
                })
            })
            .collect();
        Self {
            values,
            stage_budget_values,
        }
    }

    fn value(&self, name: &'static str) -> Option<&str> {
        self.values.get(name)?.as_deref()?.to_str()
    }

    fn documented_values(&self) -> BTreeMap<String, Option<String>> {
        let mut values = self
            .values
            .iter()
            .map(|(name, value)| {
                (
                    (*name).to_owned(),
                    value
                        .as_ref()
                        .map(|value| value.to_string_lossy().into_owned()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        values.insert("FTTS_STAGE_BUDGET_*_MS".to_owned(), None);
        values.extend(self.stage_budget_values.iter().map(|(name, value)| {
            (
                name.to_string_lossy().into_owned(),
                Some(value.to_string_lossy().into_owned()),
            )
        }));
        values
    }
}

fn environment() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(Environment::from_process)
}

#[derive(Debug)]
struct EffectiveSettings {
    profile: ExecutionProfile,
    packet_frames: PacketFrames,
    math_mode: MathMode,
    voice_pack: VoicePackProfile,
    normalize: NormalizeMode,
}

impl EffectiveSettings {
    fn resolve(cli: &Cli, environment: &Environment) -> Result<Self, FttsError> {
        let profile = cli
            .profile
            .or(parse_env_value(
                environment.value("FTTS_PROFILE"),
                "FTTS_PROFILE",
                ExecutionProfile::value_variants(),
            )?)
            .unwrap_or(ExecutionProfile::Balanced);
        let packet_frames = cli
            .packet_frames
            .or(parse_env_value(
                environment.value("FTTS_PACKET_FRAMES"),
                "FTTS_PACKET_FRAMES",
                PacketFrames::value_variants(),
            )?)
            .unwrap_or_else(|| PacketFrames::default_for(profile));
        let math_mode = cli
            .math_mode
            .or(parse_env_value(
                environment.value("FTTS_MATH_MODE"),
                "FTTS_MATH_MODE",
                MathMode::value_variants(),
            )?)
            .unwrap_or(MathMode::Fast);
        let voice_pack = cli.voice_pack.unwrap_or(VoicePackProfile::Portable);
        let normalize = cli.normalize.unwrap_or(NormalizeMode::Verbatim);
        Ok(Self {
            profile,
            packet_frames,
            math_mode,
            voice_pack,
            normalize,
        })
    }

    fn normalization_options(&self) -> NormalizationOptions {
        NormalizationOptions {
            mode: self.normalize.into(),
            ..NormalizationOptions::default()
        }
    }
}

fn parse_env_value<T>(
    value: Option<&str>,
    name: &str,
    variants: &'static [T],
) -> Result<Option<T>, FttsError>
where
    T: ValueEnum + Copy,
{
    match value {
        None => Ok(None),
        Some(value) => T::from_str(value, true).map(Some).map_err(|_| {
            let choices = variants
                .iter()
                .filter_map(|variant| variant.to_possible_value())
                .map(|variant| variant.get_name().to_owned())
                .collect::<Vec<_>>()
                .join(", ");
            FttsError::Usage(format!("invalid {name}={value:?}; use one of: {choices}"))
        }),
    }
}

fn dispatch(
    cli: Cli,
    environment: &Environment,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), FttsError> {
    match &cli.command {
        Command::Say(args) => run_say(&cli, args, environment, stdin, stdout, stderr),
        Command::MakeVideo(args) => run_make_video(&cli, args, environment, stdin, stdout, stderr),
        Command::Enroll(args) => run_enroll(args, environment, stdout),
        Command::Voice(VoiceArgs {
            command: VoiceCommand::Inspect { path },
        }) => run_voice_inspect(path, stdout),
        Command::Convert(args) => run_convert(&cli, args, environment, stdout, stderr),
        Command::Pull(args) => run_pull(args, environment, stdout),
        Command::Robot(args) => run_robot(args.command.clone(), environment, stdout),
        Command::Doctor(args) => run_doctor(args, environment, stdout),
        Command::ResidentDaemon(args) => resident::run_daemon(&args.bundle_root),
    }
}

/// A source tensor pinned by the truth-pack inventory and its reviewed storage policy.
#[derive(Clone, Debug)]
struct PinnedMainTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    access_class: AccessClass,
    storage: TensorStoragePolicy,
}

fn run_convert(
    cli: &Cli,
    args: &ConvertArgs,
    environment: &Environment,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), FttsError> {
    let run = robot::RunContext::generate();
    let outcome = run_convert_events(cli, args, environment, &run, &mut |event| {
        write_json_line(stdout, event)
    });

    if let Err(error) = &outcome {
        let mut event = run.event(robot::EventType::RunError);
        event.insert("exit_code".to_owned(), json!(error.exit_code().as_u8()));
        event.insert("kind".to_owned(), json!(error.exit_code().description()));
        event.insert("message".to_owned(), json!(error.to_string()));
        event.insert("remediation".to_owned(), json!(error.remediation()));
        event.insert("elapsed_ms".to_owned(), json!(run.elapsed_ms()));
        write_json_line(stderr, &Value::Object(event))?;
    }

    outcome
}

/// Converts the pinned main checkpoint and emits the normal run lifecycle receipt.
///
/// The source mapping owns no writable state. The destination is first created under a unique
/// sibling name with `create_new`, then atomically renamed only after the streaming writer, an
/// `fsync`, and a digest-validating mapped re-read all succeed. We deliberately leave a failed
/// staging file in place for diagnosis rather than deleting data behind the caller's back.
fn run_convert_events(
    cli: &Cli,
    args: &ConvertArgs,
    environment: &Environment,
    run: &robot::RunContext,
    emit: &mut dyn FnMut(&Value) -> Result<(), FttsError>,
) -> Result<(), FttsError> {
    let settings = EffectiveSettings::resolve(cli, environment)?;
    let mut start = run.event(robot::EventType::RunStart);
    start.insert("command".to_owned(), json!("convert"));
    start.insert("profile".to_owned(), json!(settings.profile.as_str()));
    start.insert(
        "packet_frames".to_owned(),
        json!(settings.packet_frames.as_str()),
    );
    start.insert("math_mode".to_owned(), json!(settings.math_mode.as_str()));
    start.insert("stateless".to_owned(), json!(true));
    start.insert("seed".to_owned(), json!(cli.seed));
    start.insert("model".to_owned(), Value::Null);
    start.insert("voice".to_owned(), Value::Null);
    emit(&Value::Object(start))?;

    let mut seq = 0_u64;
    emit_stage(run, emit, "source_preflight", "begin", &mut seq)?;
    let source = resolve_pinned_main_source(&args.source)?;
    let mapping = MappedFile::open(&source).map_err(|error| {
        FttsError::Input(format!(
            "cannot memory-map pinned source checkpoint {}: {error}",
            source.display()
        ))
    })?;
    let (manifest, plan) = pinned_main_conversion_plan()?;
    let staging = conversion_staging_path(&args.output)?;
    emit_stage(run, emit, "source_preflight", "end", &mut seq)?;

    emit_stage(run, emit, "convert", "begin", &mut seq)?;
    let destination = std::fs::File::options()
        .write(true)
        .create_new(true)
        .open(&staging)
        .map_err(|error| {
            FttsError::Input(format!(
                "cannot create conversion staging artifact {}: {error}; the output path is never overwritten",
                staging.display()
            ))
        })?;
    let destination = convert_safetensors_streaming(
        mapping.as_slice(),
        &manifest,
        &plan,
        destination,
    )
    .map_err(|error| {
        FttsError::ArtifactFormat(format!(
            "conversion failed before publication: {error}; staging artifact retained at {}",
            staging.display()
        ))
    })?;
    destination.sync_all().map_err(|error| {
        FttsError::ArtifactFormat(format!(
            "cannot sync converted artifact at {}: {error}; staging artifact retained",
            staging.display()
        ))
    })?;
    drop(destination);
    emit_stage(run, emit, "convert", "end", &mut seq)?;

    emit_stage(run, emit, "verify", "begin", &mut seq)?;
    let verified = MappedFttsq::open(&staging).map_err(|error| {
        FttsError::ArtifactFormat(format!(
            "converted staging artifact did not pass digest re-read: {error}; retained at {}",
            staging.display()
        ))
    })?;
    if verified.reader().source_sha256() != PINNED_MAIN_WEIGHTS_SHA256 {
        return Err(FttsError::ArtifactFormat(format!(
            "converted staging artifact recorded an unexpected source digest {}; retained at {}",
            verified.reader().source_sha256(),
            staging.display()
        )));
    }
    drop(verified);
    std::fs::rename(&staging, &args.output).map_err(|error| {
        FttsError::ArtifactFormat(format!(
            "converted artifact verified but could not be published from {} to {}: {error}; staging artifact retained",
            staging.display(),
            args.output.display()
        ))
    })?;
    emit_stage(run, emit, "verify", "end", &mut seq)?;

    let mut complete = run.event(robot::EventType::RunComplete);
    complete.insert("exit_code".to_owned(), json!(FttsExitCode::Success.as_u8()));
    complete.insert("elapsed_ms".to_owned(), json!(run.elapsed_ms()));
    complete.insert("frames".to_owned(), json!(0));
    complete.insert("audio_bytes".to_owned(), json!(0));
    emit(&Value::Object(complete))
}

fn resolve_pinned_main_source(source: &Path) -> Result<PathBuf, FttsError> {
    let source = if source.is_dir() {
        source.join(PINNED_MAIN_WEIGHTS_FILENAME)
    } else {
        source.to_owned()
    };
    if !source.is_file() {
        return Err(FttsError::Input(format!(
            "pinned main checkpoint {} does not exist or is not a file; pass model.safetensors or its containing directory",
            source.display()
        )));
    }
    if source.file_name().and_then(|name| name.to_str()) != Some(PINNED_MAIN_WEIGHTS_FILENAME) {
        return Err(FttsError::Input(format!(
            "this converter accepts the pinned main checkpoint named {PINNED_MAIN_WEIGHTS_FILENAME}, not {}",
            source.display()
        )));
    }
    Ok(source)
}

fn conversion_staging_path(output: &Path) -> Result<PathBuf, FttsError> {
    if output.exists() {
        return Err(FttsError::Input(format!(
            "refusing to overwrite existing output {}; choose a new -o path",
            output.display()
        )));
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            FttsError::Usage("conversion output must name a file, not a directory".to_owned())
        })?;
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| FttsError::Generic(format!("system clock is before UNIX_EPOCH: {error}")))?
        .as_nanos();
    let staging = parent.join(format!(
        ".{file_name}.fttsq-converting-{}-{nonce}",
        std::process::id()
    ));
    if staging.exists() {
        return Err(FttsError::Input(format!(
            "conversion staging path already exists {}; inspect or move it before retrying",
            staging.display()
        )));
    }
    Ok(staging)
}

fn pinned_main_conversion_plan() -> Result<(WeightsManifest, StreamingConversionPlan), FttsError> {
    let specs = pinned_main_tensor_specs()?;
    let manifest = WeightsManifest::from_expectations(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base main checkpoint",
        specs
            .iter()
            .map(|spec| ExpectedTensor::new(&spec.name, spec.shape.clone(), spec.dtype)),
    );
    let model_config = serde_json::from_str(PINNED_MODEL_CONFIG).map_err(|error| {
        FttsError::Generic(format!(
            "checked-in pinned model config is invalid JSON: {error}"
        ))
    })?;
    let q8_count = specs
        .iter()
        .filter(|spec| spec.storage == TensorStoragePolicy::Q8PerOutputChannel)
        .count();
    let mut plan = StreamingConversionPlan::new(
        "qwen3-tts-12hz-0.6b-base",
        PINNED_MAIN_WEIGHTS_SHA256,
    )
    .license_notice(pinned_license_notice())
    .model_config(model_config)
    .quantization_manifest(json!({
        "source": {
            "repository": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            "revision": PINNED_MODEL_REVISION,
            "file": PINNED_MAIN_WEIGHTS_FILENAME,
            "sha256": PINNED_MAIN_WEIGHTS_SHA256,
        },
        "q8_recipe": "symmetric per-output-channel int8; zero_point=0; scale=max_abs(row)/127",
        "q8_tensor_count": q8_count,
        "verbatim_tensor_count": specs.len() - q8_count,
        "q8_scope": "talker and residual-code-microdecoder attention/MLP projection matrices only",
        "verbatim_scope": "norms, heads, embeddings, speaker path, and every tensor outside the reviewed Q8 projection set",
    }));
    for spec in specs {
        let conversion = match spec.storage {
            TensorStoragePolicy::Verbatim => {
                TensorConversion::verbatim(&spec.name, &spec.name, spec.access_class)
            }
            TensorStoragePolicy::Q8PerOutputChannel => {
                TensorConversion::q8_per_output_channel(&spec.name, &spec.name, spec.access_class)
            }
        };
        plan = plan.tensor(conversion);
    }
    Ok((manifest, plan))
}

fn pinned_main_tensor_specs() -> Result<Vec<PinnedMainTensor>, FttsError> {
    let inventory: Value = serde_json::from_str(PINNED_TENSOR_INVENTORY).map_err(|error| {
        FttsError::Generic(format!(
            "checked-in tensor inventory is invalid JSON: {error}"
        ))
    })?;
    if inventory.get("source_pin").and_then(Value::as_str)
        != Some(&format!(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base@{PINNED_MODEL_REVISION}"
        ))
    {
        return Err(FttsError::Generic(
            "checked-in tensor inventory does not name the pinned Qwen3-TTS revision".to_owned(),
        ));
    }
    let records = inventory
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            FttsError::Generic("checked-in tensor inventory lacks tensors[]".to_owned())
        })?;
    let mut specs = Vec::new();
    for record in records {
        if record.get("source").and_then(Value::as_str) != Some(PINNED_MAIN_WEIGHTS_FILENAME) {
            continue;
        }
        let name = required_inventory_string(record, "name")?.to_owned();
        let dtype = match required_inventory_string(record, "dtype")? {
            "BF16" => Dtype::Bf16,
            "F32" => Dtype::F32,
            other => {
                return Err(FttsError::Generic(format!(
                    "pinned main inventory has unsupported dtype {other:?} for {name}"
                )));
            }
        };
        let shape = record
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                FttsError::Generic(format!("pinned inventory tensor {name} lacks shape[]"))
            })?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .and_then(|dimension| usize::try_from(dimension).ok())
                    .ok_or_else(|| {
                        FttsError::Generic(format!(
                            "pinned inventory tensor {name} has a non-usize shape dimension"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let storage = if is_q8_projection(&name) {
            TensorStoragePolicy::Q8PerOutputChannel
        } else {
            TensorStoragePolicy::Verbatim
        };
        specs.push(PinnedMainTensor {
            access_class: main_access_class(&name)?,
            name,
            dtype,
            shape,
            storage,
        });
    }
    if specs.len() != PINNED_MAIN_TENSOR_COUNT {
        return Err(FttsError::Generic(format!(
            "pinned main inventory contains {} tensors, expected {PINNED_MAIN_TENSOR_COUNT}",
            specs.len()
        )));
    }
    Ok(specs)
}

fn required_inventory_string<'a>(record: &'a Value, field: &str) -> Result<&'a str, FttsError> {
    record.get(field).and_then(Value::as_str).ok_or_else(|| {
        FttsError::Generic(format!(
            "checked-in tensor inventory record lacks string {field:?}"
        ))
    })
}

fn is_q8_projection(name: &str) -> bool {
    (name.starts_with("talker.model.layers.")
        || name.starts_with("talker.code_predictor.model.layers."))
        && [
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

fn main_access_class(name: &str) -> Result<AccessClass, FttsError> {
    if name == "talker.model.text_embedding.weight" {
        Ok(AccessClass::ColdTextEmbedding)
    } else if name.starts_with("speaker_encoder.") {
        Ok(AccessClass::EnrollmentSpeakerEncoder)
    } else if name.starts_with("talker.code_predictor.")
        || name == "talker.model.codec_embedding.weight"
    {
        Ok(AccessClass::HotRecurrentMicrodecoder)
    } else if name.starts_with("talker.model.")
        || name.starts_with("talker.codec_head.")
        || name.starts_with("talker.text_projection.")
    {
        Ok(AccessClass::HotRecurrentTalker)
    } else {
        Err(FttsError::Generic(format!(
            "pinned main tensor {name} has no reviewed access-class assignment"
        )))
    }
}

fn pinned_license_notice() -> String {
    format!(
        "This artifact contains model weights derived from\n\
         Qwen3-TTS-12Hz-0.6B-Base (https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)\n\
         and code derived from QwenLM/Qwen3-TTS (https://github.com/QwenLM/Qwen3-TTS).\n\n\
         Copyright 2026 Alibaba Cloud\n\n\
         Licensed under the Apache License, Version 2.0.\n\
         http://www.apache.org/licenses/LICENSE-2.0\n\n\
         CHANGES: the original bfloat16 weights were converted to franken_tts's\n\
         quantized .fttsq container. Tensors were requantized according to the\n\
         artifact's quantization manifest; protected tensors remain verbatim.\n\
         The model graph is re-implemented in Rust.\n\n\
         Apache License, Version 2.0:\n\n{APACHE_LICENSE}"
    )
}

fn run_say(
    cli: &Cli,
    args: &SayArgs,
    environment: &Environment,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), FttsError> {
    let run = robot::RunContext::generate();
    // `--stream raw` puts PCM on stdout, so events move to stderr. One contract, chosen once here
    // so no later emission can pick the other stream and interleave NDJSON with audio bytes. The
    // two branches also settle the borrows: in raw mode audio owns `stdout` and events own
    // `stderr`; otherwise events own `stdout` and the raw sink is a discard that never runs.
    let outcome = if args.stream == Some(StreamMode::Raw) {
        run_say_events(cli, args, environment, stdin, &run, stdout, &mut |event| {
            write_json_line(stderr, event)
        })
    } else if args.robot || !style::is_interactive() {
        let mut discard = io::sink();
        run_say_events(
            cli,
            args,
            environment,
            stdin,
            &run,
            &mut discard,
            &mut |event| write_json_line(stdout, event),
        )
    } else {
        // A terminal gets the human view of the same lifecycle. The NDJSON contract is untouched:
        // it is what every pipe, file, CI job and agent still receives, because none of them is a
        // terminal. `--robot` forces it back on for a human debugging the stream itself.
        let mut discard = io::sink();
        let destination = args
            .output
            .as_deref()
            .or(args.output_positional.as_deref())
            .map(|path| path.display().to_string());
        let mut presenter = style::SayPresenter::writing_to(destination);
        run_say_events(
            cli,
            args,
            environment,
            stdin,
            &run,
            &mut discard,
            &mut |event| {
                presenter
                    .event(event, stdout)
                    .map_err(|error| FttsError::Generic(format!("cannot write progress: {error}")))
            },
        )
    };

    if let Err(error) = &outcome {
        // The error is reported on the machine contract, not only as a human line: run_error is a
        // stderr event by definition (see the catalogue), so it goes to stderr on both stream
        // shapes.
        let mut event = run.event(robot::EventType::RunError);
        event.insert("exit_code".to_owned(), json!(error.exit_code().as_u8()));
        event.insert("kind".to_owned(), json!(error.exit_code().description()));
        event.insert("message".to_owned(), json!(error.to_string()));
        event.insert("remediation".to_owned(), json!(error.remediation()));
        event.insert("elapsed_ms".to_owned(), json!(run.elapsed_ms()));
        write_json_line(stderr, &Value::Object(event))?;
    }

    outcome
}

/// `ftts make-video`: synthesize (or take a WAV) and render the branded
/// share video. Frames, waveform, and text are pure Rust (`ftts-video`);
/// `.mp4` goes through the same first-available-system-encoder contract as
/// `ftts say`'s `.m4a` path, and `.y4m` + `.wav` is the native no-encoder
/// output.
fn run_make_video(
    cli: &Cli,
    args: &MakeVideoArgs,
    environment: &Environment,
    stdin: &mut dyn Read,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), FttsError> {
    let output = args
        .output
        .clone()
        .or_else(|| args.output_positional.clone())
        .ok_or_else(|| {
            FttsError::Usage(
                "`ftts make-video` needs an output path (`ftts make-video \"text\" out.mp4`)"
                    .to_owned(),
            )
        })?;
    let extension = output
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    let is_mp4 = match extension.as_deref() {
        Some("mp4") => true,
        Some("y4m") => false,
        other => {
            return Err(FttsError::Usage(format!(
                "unsupported video extension `.{}`; use .mp4 (system encoder) or .y4m (native)",
                other.unwrap_or("<none>")
            )));
        }
    };

    // The voice pill needs a human name: an explicit --label wins, a preset
    // keeps its capitalized name, a custom voice shows its file stem. With
    // no voice given, the label follows the same default chain `say` uses
    // (FTTS_DEFAULT_VOICE, then an enrolled default.spk, then built-in matt)
    // rather than claiming "Matt" over someone's enrolled voice.
    let capitalize = |raw: &str| {
        let mut chars = raw.chars();
        chars
            .next()
            .map(|first| first.to_uppercase().collect::<String>() + chars.as_str())
            .unwrap_or_else(|| raw.to_owned())
    };
    let stem_of = |path: &Path| {
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .map(str::to_owned)
    };
    let label = args.label.clone().unwrap_or_else(|| {
        if let Some(voice) = &args.voice {
            return stem_of(voice)
                .map(|stem| capitalize(&stem))
                .unwrap_or_else(|| "Voice".to_owned());
        }
        if let Some(audio) = &args.audio {
            // Rendering someone's own recording: name it after the file.
            return stem_of(audio)
                .map(|stem| capitalize(&stem))
                .unwrap_or_else(|| "Voice".to_owned());
        }
        if let Some(default_voice) = environment.value("FTTS_DEFAULT_VOICE")
            && let Some(stem) = stem_of(Path::new(default_voice))
        {
            return capitalize(&stem);
        }
        let enrolled_default = resolve_model(args.model.as_deref(), environment)
            .ok()
            .and_then(|model| synth::ModelBundle::resolve(Path::new(&model)).ok())
            .is_some_and(|bundle| bundle.root.join("default.spk").is_file());
        if enrolled_default {
            "My voice".to_owned()
        } else {
            "Matt".to_owned()
        }
    });

    // Audio: either supplied, or synthesized through the full `say` pipeline
    // (same events, presenter, and resident engine). An mp4 gets a staging
    // WAV that is consumed by the encoder; a y4m synthesizes straight into
    // its `.wav` sibling, which stays as the video's audio track.
    let staging: Option<PathBuf> = if args.audio.is_none() {
        if is_mp4 {
            let mut path = output.as_os_str().to_owned();
            path.push(".ftts-staging.wav");
            Some(PathBuf::from(path))
        } else {
            Some(output.with_extension("wav"))
        }
    } else {
        None
    };
    if let Some(staging_path) = &staging {
        let say_args = SayArgs {
            text: args.text.clone(),
            output_positional: None,
            file: args.file.clone(),
            model: args.model.clone(),
            voice: args.voice.clone(),
            output: Some(staging_path.clone()),
            stream: None,
            check: false,
            robot: false,
            no_resident: args.no_resident,
        };
        run_say(cli, &say_args, environment, stdin, stdout, stderr)?;
    }
    let audio_path = args
        .audio
        .clone()
        .or_else(|| staging.clone())
        .unwrap_or_default();

    let interactive = style::is_interactive();
    if interactive {
        writeln!(stdout, "rendering video: {}", output.display())
            .map_err(|error| FttsError::Generic(format!("cannot write progress: {error}")))?;
    }
    let request = ftts_video::VideoRequest {
        audio: &audio_path,
        output: &output,
        voice_label: &label,
    };
    let mut last_percent = 0usize;
    let render_result = ftts_video::render(&request, &mut |progress| {
        if !interactive {
            return;
        }
        let percent = progress.frame * 100 / progress.total_frames;
        if percent >= last_percent + 10 || progress.frame == progress.total_frames {
            last_percent = percent;
            let _ = write!(
                stdout,
                "\r  frame {}/{} ({percent}%)",
                progress.frame, progress.total_frames
            );
            let _ = stdout.flush();
        }
    });
    if interactive {
        let _ = writeln!(stdout);
    }
    render_result.map_err(FttsError::Generic)?;

    // The staging WAV is consumed into the mp4; remove it exactly as the
    // `.m4a` OutputPlan removes its staging file. The `.y4m` path keeps its
    // audio, renamed to the output's `.wav` sibling by the renderer.
    if is_mp4 && let Some(staging_path) = &staging {
        let _ = fs::remove_file(staging_path);
    }
    if interactive {
        writeln!(stdout, "wrote {}", output.display())
            .map_err(|error| FttsError::Generic(format!("cannot write result: {error}")))?;
    }
    Ok(())
}

/// Emit one `stage` event and advance the run's stage counter.
fn emit_stage(
    run: &robot::RunContext,
    emit: &mut dyn FnMut(&Value) -> Result<(), FttsError>,
    name: &str,
    state: &str,
    seq: &mut u64,
) -> Result<(), FttsError> {
    let mut event = run.event(robot::EventType::Stage);
    event.insert("name".to_owned(), json!(name));
    event.insert("seq".to_owned(), json!(*seq));
    event.insert("state".to_owned(), json!(state));
    event.insert("elapsed_ms".to_owned(), json!(run.elapsed_ms()));
    event.insert("budget_ms".to_owned(), Value::Null);
    *seq += 1;
    emit(&Value::Object(event))
}

/// The `say` pipeline proper, emitting its lifecycle through `emit`.
///
/// Split out so the caller owns stream selection and the single `run_error` emission point: a
/// pipeline that emitted its own errors would have to know which stream it was on at every `?`.
fn run_say_events(
    cli: &Cli,
    args: &SayArgs,
    environment: &Environment,
    stdin: &mut dyn Read,
    run: &robot::RunContext,
    raw_audio: &mut dyn Write,
    emit: &mut dyn FnMut(&Value) -> Result<(), FttsError>,
) -> Result<(), FttsError> {
    let settings = EffectiveSettings::resolve(cli, environment)?;

    let mut start = run.event(robot::EventType::RunStart);
    start.insert("command".to_owned(), json!("say"));
    start.insert("profile".to_owned(), json!(settings.profile.as_str()));
    start.insert(
        "packet_frames".to_owned(),
        json!(settings.packet_frames.as_str()),
    );
    start.insert("math_mode".to_owned(), json!(settings.math_mode.as_str()));
    start.insert("stateless".to_owned(), json!(true));
    start.insert("seed".to_owned(), json!(cli.seed));
    start.insert("model".to_owned(), json!(args.model.as_deref()));
    start.insert(
        "voice".to_owned(),
        json!(args.voice.as_ref().map(|path| path.display().to_string())),
    );
    emit(&Value::Object(start))?;

    let mut seq = 0u64;

    emit_stage(run, emit, "resolve", "begin", &mut seq)?;
    let text = read_text(args, stdin)?;
    let model = resolve_model(args.model.as_deref(), environment)?;
    let voice = resolve_requested_voice(args.voice.as_deref(), environment)?;
    emit_stage(run, emit, "resolve", "end", &mut seq)?;

    let request = SynthesisRequest::new(text)
        .with_normalization_options(settings.normalization_options())
        .with_normalization_trace(cli.trace.is_some());

    // Privacy-safe by construction: shape and rule names only, never the text itself. The CLI
    // promises no persisted synthesis history, and an event stream an agent may log is exactly
    // where that promise would leak if this carried the input.
    let mut prepared = run.event(robot::EventType::TextPrepared);
    prepared.insert("normalize".to_owned(), json!(settings.normalize.as_str()));
    prepared.insert(
        "unicode_version".to_owned(),
        json!(ftts_model_qwen::tokenizer::unicode_version()),
    );
    prepared.insert("char_count".to_owned(), json!(request.text.chars().count()));
    prepared.insert(
        "trace_requested".to_owned(),
        json!(request.trace_normalization),
    );
    emit(&Value::Object(prepared))?;

    // `-o PATH` and the positional OUTPUT are the same request; clap rejects supplying both.
    let requested_output: Option<PathBuf> = args
        .output
        .clone()
        .or_else(|| args.output_positional.clone());
    let output_plan = requested_output
        .as_deref()
        .map(OutputPlan::for_path)
        .transpose()?;

    emit_stage(run, emit, "admission", "begin", &mut seq)?;
    let admission = admission_plan(&request.text, &settings)?;
    emit_stage(run, emit, "admission", "end", &mut seq)?;

    if args.check {
        let event = json!({
            "schema_version": ROBOT_SCHEMA_VERSION,
            "event": "check_complete",
            "run_id": run.run_id(),
            "model": model,
            "voice": voice,
            "profile": settings.profile.as_str(),
            "packet_frames": settings.packet_frames.as_str(),
            "math_mode": settings.math_mode.as_str(),
            "voice_pack": settings.voice_pack.as_str(),
            "normalize": settings.normalize.as_str(),
            "normalization_trace_requested": request.trace_normalization,
            "seed": cli.seed,
            "trace": cli.trace.as_ref().map(|path| path.display().to_string()),
            "output": requested_output.as_ref().map(|path| path.display().to_string()),
            "admission": admission,
        });
        emit(&event)?;
        let mut complete = run.event(robot::EventType::RunComplete);
        complete.insert("exit_code".to_owned(), json!(FttsExitCode::Success.as_u8()));
        complete.insert("elapsed_ms".to_owned(), json!(run.elapsed_ms()));
        complete.insert("frames".to_owned(), json!(0));
        complete.insert("audio_bytes".to_owned(), json!(0));
        emit(&Value::Object(complete))?;
        return Ok(());
    }

    // --- audio destination, decided before any model work ----------------------------------
    // A run that synthesizes for thirty seconds and then discovers it has nowhere to put the
    // result has wasted the user's time; the refusal belongs here, before the weights load.
    let raw_stream = args.stream == Some(StreamMode::Raw);
    let mut audio = match (&output_plan, raw_stream) {
        (Some(plan), false) => AudioOutput::wav(&plan.wav_path)?,
        (None, true) => AudioOutput::raw(),
        (None, false) => {
            return Err(FttsError::Usage(
                "`ftts say` has nowhere to put the audio; add an output path (`ftts say \"text\" \
                 out.wav`, or `-o PATH`) or `--stream raw` for PCM on stdout"
                    .to_owned(),
            ));
        }
        // clap declares every output form and `--stream` mutually exclusive.
        (Some(_), true) => unreachable!("clap enforces the conflict"),
    };

    // --- model load ------------------------------------------------------------------------
    emit_stage(run, emit, "load", "begin", &mut seq)?;
    let bundle = synth::ModelBundle::resolve(Path::new(&model))?;
    let voice_path = match voice.as_deref().map(PathBuf::from).or_else(|| {
        let candidate = bundle.root.join("default.spk");
        candidate.is_file().then_some(candidate)
    }) {
        Some(path) => path,
        // Out-of-box: no --voice, no FTTS_DEFAULT_VOICE, no enrollment — speak with the built-in
        // default preset rather than refusing. The presets are real enrolled x-vectors taken from
        // speech, so they sit on the speaker encoder's manifold; any user enrollment or explicit
        // voice always outranks them.
        None => materialize_preset_voice(DEFAULT_PRESET_VOICE)
            .expect("the default preset name is a member of PRESET_VOICES")?,
    };
    // A `--voice` that names an audio file (not a .spk vector) computes an ephemeral
    // enrollment, and gets the same automatic denoise `ftts enroll` applies — otherwise the
    // one-off form of the exact same operation would sound worse than the saved form. The
    // report goes unread here: `say` has no enrollment console, and the .spk/preset paths
    // never enter the cleanup code.
    let mut say_denoise_report = None;
    let denoise_ephemeral = bundle.root.join(synth::DENOISE_ARTIFACT_RELPATH).is_file();
    let speaker = synth::speaker_from_voice(
        &bundle,
        &voice_path,
        synth::ReferenceCleanup {
            denoise: denoise_ephemeral.then_some(&mut say_denoise_report),
            dereverb: None,
        },
    )?;
    // With the resident engine (the default), the model stays loaded in a background
    // process and this invocation skips its own hydration; the daemon's load happens
    // inside the synthesis stage on its first request. Any resident-path unavailability
    // falls back to the classic in-process load below, never to a failure.
    let use_resident = resident::enabled(args.no_resident);
    let loaded = if use_resident {
        None
    } else {
        Some(synth::LoadedModel::load(&bundle)?)
    };
    emit_stage(run, emit, "load", "end", &mut seq)?;

    // --- synthesis -------------------------------------------------------------------------
    // The engine's observer is not forwarded to the event stream here. Its events are lifecycle
    // facts the robot contract already carries as `stage` events, and per-frame progress cannot be
    // emitted from inside this call anyway: `emit` is borrowed for the duration. Progress becomes
    // observable per packet once synthesis returns, and genuinely incremental frame events belong
    // with the streaming decode path rather than being faked from a completed run.
    let observer = |_event: ftts_core::SynthesisEvent| {};

    // Canonical greedy consumes no RNG state, so an absent `--seed` changes nothing today;
    // 0 is the documented default rather than a value picked per run, which would make a
    // future switch to the production sampler silently irreproducible.
    let seed = cli.seed.unwrap_or(0);

    emit_stage(run, emit, "synthesis", "begin", &mut seq)?;
    let resident_audio = if use_resident {
        resident::try_synthesize(
            &bundle,
            &resident::WireRequest {
                text: &request.text,
                normalize: settings.normalize.as_str(),
                trace: request.trace_normalization,
                speaker: &speaker,
                seed,
            },
        )?
    } else {
        None
    };
    let audio_result = match resident_audio {
        Some(audio) => audio,
        None => {
            let loaded = match loaded {
                Some(loaded) => loaded,
                // The resident path was requested but no daemon could serve it.
                None => synth::LoadedModel::load(&bundle)?,
            };
            let engine = ftts_core::TtsEngine::from_process_environment()
                .map_err(|error| FttsError::Generic(format!("cannot start the engine: {error}")))?;
            let cancellation = ftts_core::CancellationToken::new();
            synth::synthesize(
                &loaded,
                &engine,
                &request,
                &speaker,
                seed,
                &cancellation,
                &observer,
            )?
        }
    };
    emit_stage(run, emit, "synthesis", "end", &mut seq)?;

    // --- the output tail ---------------------------------------------------------------------
    let packet_samples = settings.packet_frames.samples_per_packet();
    let packet_frame_count = settings.packet_frames.frames_per_packet();
    emit_stage(run, emit, "output", "begin", &mut seq)?;
    for packet in audio_result.pcm.chunks(packet_samples) {
        let event = audio.write_packet(packet, raw_audio, run.run_id(), packet_frame_count)?;
        emit(&event)?;
    }
    let audio_bytes = audio.byte_offset();
    let samples = audio.finish()?;
    if let Some(plan) = &output_plan {
        plan.finalize()?;
    }
    emit_stage(run, emit, "output", "end", &mut seq)?;

    let mut complete = run.event(robot::EventType::RunComplete);
    complete.insert("exit_code".to_owned(), json!(FttsExitCode::Success.as_u8()));
    complete.insert("elapsed_ms".to_owned(), json!(run.elapsed_ms()));
    complete.insert("frames".to_owned(), json!(audio_result.frames));
    complete.insert("audio_bytes".to_owned(), json!(audio_bytes));
    complete.insert("samples".to_owned(), json!(samples));
    complete.insert(
        "duration_ms".to_owned(),
        json!(samples * 1000 / u64::from(ftts_core::audio::SAMPLE_RATE_HZ)),
    );
    complete.insert(
        "prepared_token_count".to_owned(),
        json!(audio_result.prepared_token_count),
    );
    if let Some(ttfa) = audio_result.ttfa {
        complete.insert(
            "ttfa_ms".to_owned(),
            json!(u64::try_from(ttfa.as_millis()).unwrap_or(u64::MAX)),
        );
    }
    emit(&Value::Object(complete))?;
    Ok(())
}

/// Where synthesised PCM goes, and the `audio_chunk` events that describe it.
///
/// The two destinations are mutually exclusive by contract (AGENTS.md agent ergonomics): either
/// events own stdout and audio goes to `-o PATH`, or `--stream raw` gives stdout to PCM and every
/// event goes to stderr. Raw bytes and NDJSON are never interleaved on one stream, so this type
/// owns the decision once instead of leaving it to each call site.
///
/// `audio_chunk` reports the bytes written, never the bytes themselves.
pub enum AudioSink {
    /// A WAV file. The header is finalised on [`AudioSink::finish`], so a run cut short still
    /// leaves a playable file describing the samples that landed.
    Wav(Box<ftts_core::audio::WavWriter<fs::File>>),
    /// Raw little-endian 16-bit PCM on a caller-supplied stream (`--stream raw`).
    RawPcm,
    /// `--check` and other non-synthesising paths.
    None,
}

/// Accumulating state for the `audio_chunk` event stream.
pub struct AudioOutput {
    sink: AudioSink,
    byte_offset: u64,
    samples_written: u64,
}

/// Audio container selected by the output path's extension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputFormat {
    /// Written natively by the pure-Rust WAV writer.
    Wav,
    /// AAC in an MPEG-4 container, encoded by `afconvert` (macOS) or `ffmpeg`.
    M4a,
    /// MP3, encoded by `lame` or `ffmpeg`.
    Mp3,
    /// FLAC, encoded by `flac` or `ffmpeg`.
    Flac,
}

/// Where the WAV bytes land and what happens to them after synthesis.
///
/// Synthesis always writes the pure-Rust WAV stream ("self-contained" covers everything up to
/// and including that file). For a compressed extension the WAV goes to a sibling staging file
/// and is then handed to the first available *system* encoder — an optional post-step, never a
/// runtime dependency of synthesis itself. No encoder found is a refusal with the tool list,
/// not a silent format switch.
#[derive(Clone, Debug)]
struct OutputPlan {
    /// The path the user asked for.
    final_path: PathBuf,
    /// Where the WAV sink writes; equals `final_path` for `.wav`.
    wav_path: PathBuf,
    format: OutputFormat,
}

impl OutputPlan {
    fn for_path(path: &Path) -> Result<Self, FttsError> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);
        let format = match extension.as_deref() {
            Some("wav") | None => OutputFormat::Wav,
            Some("m4a" | "aac") => OutputFormat::M4a,
            Some("mp3") => OutputFormat::Mp3,
            Some("flac") => OutputFormat::Flac,
            Some(other) => {
                return Err(FttsError::Usage(format!(
                    "unsupported output extension `.{other}`; use .wav (native), .m4a, .mp3, or \
                     .flac (system encoder)"
                )));
            }
        };
        let wav_path = if format == OutputFormat::Wav {
            path.to_path_buf()
        } else {
            let mut staging = path.as_os_str().to_owned();
            staging.push(".ftts-staging.wav");
            PathBuf::from(staging)
        };
        Ok(Self {
            final_path: path.to_path_buf(),
            wav_path,
            format,
        })
    }

    /// Encodes the staged WAV into the requested container and removes the staging file.
    fn finalize(&self) -> Result<(), FttsError> {
        if self.format == OutputFormat::Wav {
            return Ok(());
        }
        let wav = self.wav_path.as_os_str();
        let target = self.final_path.as_os_str();
        // (encoder, arguments) attempts in preference order; the first tool present decides.
        let attempts: &[(&str, Vec<&std::ffi::OsStr>)] = &match self.format {
            OutputFormat::M4a => [
                (
                    "afconvert",
                    vec![
                        "-f".as_ref(),
                        "m4af".as_ref(),
                        "-d".as_ref(),
                        "aac".as_ref(),
                        wav,
                        target,
                    ],
                ),
                (
                    "ffmpeg",
                    vec![
                        "-y".as_ref(),
                        "-loglevel".as_ref(),
                        "error".as_ref(),
                        "-i".as_ref(),
                        wav,
                        "-c:a".as_ref(),
                        "aac".as_ref(),
                        target,
                    ],
                ),
            ],
            OutputFormat::Mp3 => [
                (
                    "lame",
                    vec!["--quiet".as_ref(), "-V2".as_ref(), wav, target],
                ),
                (
                    "ffmpeg",
                    vec![
                        "-y".as_ref(),
                        "-loglevel".as_ref(),
                        "error".as_ref(),
                        "-i".as_ref(),
                        wav,
                        "-codec:a".as_ref(),
                        "libmp3lame".as_ref(),
                        "-q:a".as_ref(),
                        "2".as_ref(),
                        target,
                    ],
                ),
            ],
            OutputFormat::Flac => [
                (
                    "flac",
                    vec![
                        "--totally-silent".as_ref(),
                        "-f".as_ref(),
                        "-o".as_ref(),
                        target,
                        wav,
                    ],
                ),
                (
                    "ffmpeg",
                    vec![
                        "-y".as_ref(),
                        "-loglevel".as_ref(),
                        "error".as_ref(),
                        "-i".as_ref(),
                        wav,
                        "-c:a".as_ref(),
                        "flac".as_ref(),
                        target,
                    ],
                ),
            ],
            OutputFormat::Wav => unreachable!("handled above"),
        };

        let mut tried = Vec::new();
        for (tool, arguments) in attempts {
            // `tool` is always one of the fixed string literals in `attempts` above — a
            // compile-time allowlist. User-controlled data (the two paths) enters only as argv.
            match std::process::Command::new(tool).args(arguments).status() {
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    tried.push(*tool);
                }
                Err(error) => {
                    return Err(FttsError::Generic(format!(
                        "audio encoder `{tool}` could not run: {error}; the synthesized WAV is \
                         preserved at {}",
                        self.wav_path.display()
                    )));
                }
                Ok(status) if status.success() => {
                    // The staging WAV is an intermediate this run created; the requested artifact
                    // now exists, so the intermediate is removed.
                    let _ = fs::remove_file(&self.wav_path);
                    return Ok(());
                }
                Ok(status) => {
                    return Err(FttsError::Generic(format!(
                        "audio encoder `{tool}` exited with {status}; the synthesized WAV is \
                         preserved at {}",
                        self.wav_path.display()
                    )));
                }
            }
        }
        Err(FttsError::Generic(format!(
            "no system audio encoder found for {} (tried: {}); install one or use a .wav output. \
             The synthesized WAV is preserved at {}",
            self.final_path.display(),
            tried.join(", "),
            self.wav_path.display()
        )))
    }
}

impl AudioOutput {
    /// Open a WAV file sink.
    ///
    /// # Errors
    ///
    /// If the file cannot be created or the provisional header cannot be written.
    pub fn wav(path: &Path) -> Result<Self, FttsError> {
        let file = fs::File::create(path).map_err(|error| {
            FttsError::Generic(format!(
                "cannot create audio output {}: {error}",
                path.display()
            ))
        })?;
        let writer = ftts_core::audio::WavWriter::new(file, ftts_core::audio::SAMPLE_RATE_HZ)
            .map_err(|error| {
                FttsError::Generic(format!(
                    "cannot write WAV header to {}: {error}",
                    path.display()
                ))
            })?;
        Ok(Self {
            sink: AudioSink::Wav(Box::new(writer)),
            byte_offset: 0,
            samples_written: 0,
        })
    }

    /// A raw-PCM sink; the caller supplies the stream on each write.
    #[must_use]
    pub const fn raw() -> Self {
        Self {
            sink: AudioSink::RawPcm,
            byte_offset: 0,
            samples_written: 0,
        }
    }

    /// A sink that discards audio, for paths that synthesise nothing.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            sink: AudioSink::None,
            byte_offset: 0,
            samples_written: 0,
        }
    }

    /// The stable `sink` string reported in `audio_chunk`.
    #[must_use]
    pub const fn sink_name(&self) -> &'static str {
        match self.sink {
            AudioSink::Wav(_) => "file",
            AudioSink::RawPcm => "stdout",
            AudioSink::None => "none",
        }
    }

    /// Bytes of audio emitted so far.
    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Write one packet and return its `audio_chunk` event.
    ///
    /// `raw` is where PCM goes under `--stream raw`; it is ignored by the other sinks. The event's
    /// `byte_offset` is the offset *before* this packet, so a consumer can seek with it.
    ///
    /// `duration_ms` is derived from the sample count rather than taken from a caller-supplied
    /// clock: it describes how much *audio* this packet holds, which is a property of the samples,
    /// not of how long the run took to produce them.
    ///
    /// # Errors
    ///
    /// If the sink rejects the write.
    pub fn write_packet(
        &mut self,
        pcm: &[f32],
        raw: &mut dyn Write,
        run_id: &str,
        frame_count: u8,
    ) -> Result<Value, FttsError> {
        let offset_before = self.byte_offset;
        let bytes = (pcm.len() * 2) as u64;

        match &mut self.sink {
            AudioSink::Wav(writer) => writer.write_samples(pcm).map_err(|error| {
                FttsError::Generic(format!("cannot write audio samples: {error}"))
            })?,
            AudioSink::RawPcm => {
                let mut buffer = Vec::with_capacity(pcm.len() * 2);
                for sample in pcm {
                    buffer
                        .extend_from_slice(&ftts_core::audio::sample_to_i16(*sample).to_le_bytes());
                }
                raw.write_all(&buffer).map_err(|error| {
                    FttsError::Generic(format!("cannot write raw PCM: {error}"))
                })?;
            }
            AudioSink::None => {}
        }

        self.byte_offset += bytes;
        self.samples_written += pcm.len() as u64;

        let mut event = robot::EventType::AudioChunk.event();
        event.insert("run_id".to_owned(), json!(run_id));
        event.insert("byte_offset".to_owned(), json!(offset_before));
        event.insert("bytes".to_owned(), json!(bytes));
        event.insert(
            "duration_ms".to_owned(),
            json!((pcm.len() as u64) * 1000 / u64::from(ftts_core::audio::SAMPLE_RATE_HZ.max(1))),
        );
        event.insert("packet_frames".to_owned(), json!(frame_count.to_string()));
        event.insert("sink".to_owned(), json!(self.sink_name()));
        Ok(Value::Object(event))
    }

    /// Finalise the sink, patching the WAV header to the real length.
    ///
    /// # Errors
    ///
    /// If the header cannot be rewritten.
    pub fn finish(self) -> Result<u64, FttsError> {
        let samples = self.samples_written;
        if let AudioSink::Wav(writer) = self.sink {
            writer.finish().map_err(|error| {
                FttsError::Generic(format!("cannot finalize the WAV header: {error}"))
            })?;
        }
        Ok(samples)
    }
}

fn read_text(args: &SayArgs, stdin: &mut dyn Read) -> Result<String, FttsError> {
    let text = match (&args.text, &args.file) {
        (Some(text), None) if text == "-" => read_utf8(stdin, "stdin")?,
        (Some(text), None) => text.clone(),
        (None, Some(path)) if path == Path::new("-") => read_utf8(stdin, "stdin")?,
        (None, Some(path)) => fs::read_to_string(path).map_err(|error| {
            FttsError::Input(format!(
                "cannot read text file {}: {error}; use `ftts say --file PATH --check --model PATH`",
                path.display()
            ))
        })?,
        (None, None) => {
            return Err(FttsError::Usage(
                "missing text; use `ftts say TEXT`, `ftts say --file PATH`, or `ftts say -`".to_owned(),
            ));
        }
        (Some(_), Some(_)) => unreachable!("clap enforces the conflict"),
    };

    if text.trim().is_empty() {
        return Err(FttsError::Input(
            "text is empty; provide non-whitespace UTF-8 text to `ftts say`".to_owned(),
        ));
    }
    Ok(text)
}

fn read_utf8(reader: &mut dyn Read, source: &str) -> Result<String, FttsError> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).map_err(|error| {
        FttsError::Input(format!(
            "cannot read {source}: {error}; retry with readable UTF-8 input"
        ))
    })?;
    String::from_utf8(bytes).map_err(|error| {
        FttsError::Input(format!(
            "{source} is not valid UTF-8: {error}; transcode it before `ftts say`"
        ))
    })
}

fn resolve_model(explicit: Option<&Path>, environment: &Environment) -> Result<String, FttsError> {
    resolve_model_from(
        explicit,
        &model_search_paths(environment),
        default_pull_model_dir(environment).as_deref(),
    )
}

/// The full resolution order — `--model`, then the searched artifact paths (`FTTS_MODEL_DIR`,
/// then the home cache), then the `ftts pull` destination directory, accepted only when it holds
/// a complete bundle. Takes its inputs as data so tests can exercise the order against temp
/// directories without mutating process environment.
fn resolve_model_from(
    explicit: Option<&Path>,
    searched: &[PathBuf],
    pull_dir: Option<&Path>,
) -> Result<String, FttsError> {
    if let Some(path) = explicit {
        // A pinned checkpoint is a *directory* of five files, so `--model DIR` is the natural
        // thing to type and is accepted as such. `.fttsq` is a single file, and both forms reach
        // the same resolver rather than one being a special case documented somewhere else.
        if path.is_dir() {
            return Ok(path.display().to_string());
        }
        return resolve_existing_file(path, "model artifact")
            .map(|path| path.display().to_string());
    }

    if let Some(path) = searched.iter().find(|path| path.is_file()) {
        return Ok(path.display().to_string());
    }
    // A directory named by FTTS_MODEL_DIR may itself BE the model: a pinned checkpoint snapshot
    // (`model.safetensors` + configs) or a directory holding the canonical artifact. `--model DIR`
    // already accepts that shape; the search path accepting it too is what lets a bare
    // `ftts say "text" out.wav` work after one exported variable.
    if let Some(directory) = searched
        .iter()
        .filter_map(|path| path.parent())
        .find(|directory| directory.join("model.safetensors").is_file())
    {
        return Ok(directory.display().to_string());
    }

    // The `ftts pull` destination is a *bundle* directory (checkpoints + tokenizer files), not a
    // single artifact, so it counts only when the whole bundle is present — a half-finished pull
    // resolving here would fail later with a less actionable error than the one below.
    if let Some(directory) = pull_dir
        && directory.is_dir()
        && synth::ModelBundle::resolve(directory).is_ok()
    {
        return Ok(directory.display().to_string());
    }

    let searched = searched
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(FttsError::ModelNotFound(format!(
        "no model artifact was found; searched: [{searched}]; run `ftts pull` to fetch the model \
         (~2.0 GB), or pass --model PATH or set FTTS_MODEL_DIR"
    )))
}

fn resolve_optional_file(path: Option<&Path>, label: &str) -> Result<Option<String>, FttsError> {
    path.map(|path| resolve_existing_file(path, label).map(|path| path.display().to_string()))
        .transpose()
}

fn resolve_requested_voice(
    explicit: Option<&Path>,
    environment: &Environment,
) -> Result<Option<String>, FttsError> {
    if let Some(path) = explicit {
        // A bare preset name selects a built-in voice — but only when no such file exists, so
        // `--voice aria` in a directory containing a file named `aria` still means the file.
        if !path.exists()
            && let Some(name) = path.to_str()
            && let Some(materialized) = materialize_preset_voice(name)
        {
            return materialized.map(|path| Some(path.display().to_string()));
        }
        // A failed bare word was probably a preset-name attempt: name the built-ins in the
        // refusal so the user does not have to hunt the docs for the list.
        let looks_like_name =
            path.extension().is_none() && path.components().count() == 1 && !path.exists();
        return resolve_optional_file(Some(path), "voice source").map_err(|error| {
            if looks_like_name {
                FttsError::Input(format!(
                    "{error}; built-in voice names are: {}",
                    preset_names()
                ))
            } else {
                error
            }
        });
    }
    environment
        .value("FTTS_DEFAULT_VOICE")
        .map(Path::new)
        .map(|path| resolve_existing_file(path, "FTTS_DEFAULT_VOICE"))
        .transpose()
        .map(|path| path.map(|path| path.display().to_string()))
}

fn run_enroll(
    args: &EnrollArgs,
    environment: &Environment,
    stdout: &mut dyn Write,
) -> Result<(), FttsError> {
    let _ = args.force;
    let model = resolve_model(args.model.as_deref(), environment)?;
    let bundle = synth::ModelBundle::resolve(Path::new(&model))?;
    let output = match (&args.output, args.default) {
        (Some(path), false) => path.clone(),
        (None, true) => bundle.root.join("default.spk"),
        (None, false) => {
            return Err(FttsError::Usage(
                "`ftts enroll` needs -o PATH or --default; enrollment never overwrites a voice source"
                    .to_owned(),
            ));
        }
        (Some(_), true) => unreachable!("clap enforces the conflict"),
    };
    let mut denoise_report = None;
    let mut dereverb_report = None;
    // Denoise resolution: --no-denoise wins, --denoise forces (including the classic engine
    // when the weights are absent), and the default is neural-when-pulled — never a silent
    // fallback to a different engine than the one the default advertises.
    let denoise = if args.no_denoise {
        false
    } else {
        args.denoise || bundle.root.join(synth::DENOISE_ARTIFACT_RELPATH).is_file()
    };
    let speaker = synth::speaker_from_voice(
        &bundle,
        &args.reference_audio,
        synth::ReferenceCleanup {
            denoise: denoise.then_some(&mut denoise_report),
            dereverb: args.dereverb.then_some(&mut dereverb_report),
        },
    )?;
    // Same reporting discipline as the denoise below: state what was measured. A reference whose
    // reverb time barely moves was not the problem, and a better recording beats more filtering.
    if let Some(report) = dereverb_report {
        style::ok(
            stdout,
            &format!(
                "dereverberated reference {}",
                style::detail(&format!(
                    "RT60-equivalent {:.2} → {:.2} s",
                    report.before_rt60_s, report.after_rt60_s
                )),
            ),
        )
        .map_err(|error| FttsError::Generic(format!("cannot write dereverb report: {error}")))?;
    }
    // Report what the denoise measured rather than asserting it helped: a reference whose floor
    // barely moves was not noisy, and the user should reach for a better recording instead.
    if let Some(report) = denoise_report {
        let moved = report.before_dbfs - report.after_dbfs;
        style::ok(
            stdout,
            &format!(
                "denoised reference {}",
                style::detail(&format!(
                    "pause floor {:.1} → {:.1} dBFS ({moved:.1} dB quieter)",
                    report.before_dbfs, report.after_dbfs
                )),
            ),
        )
        .map_err(|error| FttsError::Generic(format!("cannot write denoise report: {error}")))?;
    }

    // Enrollment is cheap to redo; the recording behind an existing voice may not still exist. So
    // an occupied destination asks rather than refuses — but only when somebody is there to ask.
    // A pipe, a CI job, or an agent gets the explicit error it can act on instead of a prompt that
    // would hang forever, and says `--overwrite` when it means it.
    let backup = if output.exists() {
        let consented = if args.overwrite {
            true
        } else {
            style::warn(
                stdout,
                &format!(
                    "{} already holds an enrolled voice",
                    style::emphasis(&output.display().to_string())
                ),
            )
            .map_err(|error| {
                FttsError::Generic(format!("cannot write overwrite notice: {error}"))
            })?;
            match style::confirm(stdout, "Replace it?")
                .map_err(|error| FttsError::Generic(format!("cannot read a reply: {error}")))?
            {
                Some(reply) => reply,
                None => {
                    return Err(FttsError::Input(format!(
                        "{} already exists; pass --overwrite to replace it (the displaced voice is \
                         kept as {}.bak)",
                        output.display(),
                        output.display()
                    )));
                }
            }
        };
        if !consented {
            style::info(stdout, "left the existing voice in place")
                .map_err(|error| FttsError::Generic(format!("cannot write result: {error}")))?;
            return Ok(());
        }
        Some(synth::replace_speaker_vector(&output, &speaker)?)
    } else {
        synth::write_speaker_vector_new(&output, &speaker)?;
        None
    };

    style::ok(
        stdout,
        &format!(
            "enrolled {} → {}",
            style::emphasis(&args.reference_audio.display().to_string()),
            style::emphasis(&output.display().to_string()),
        ),
    )
    .map_err(|error| FttsError::Generic(format!("cannot write enrollment result: {error}")))?;
    if let Some(backup) = backup {
        style::info(
            stdout,
            &format!(
                "previous voice kept at {}",
                style::emphasis(&backup.display().to_string())
            ),
        )
        .map_err(|error| FttsError::Generic(format!("cannot write backup notice: {error}")))?;
    }
    if args.default {
        style::info(
            stdout,
            &format!(
                "{} will use it when --voice is absent",
                style::emphasis("ftts say")
            ),
        )
        .map_err(|error| FttsError::Generic(format!("cannot write result: {error}")))?;
    }
    Ok(())
}

/// One downloadable model file from the embedded manifest.
#[derive(Clone, Debug)]
struct ModelManifestFile {
    /// The bare release-asset name on the GitHub release.
    asset: String,
    /// Relative path under the model directory the asset lands at.
    dest: String,
    /// Pinned lowercase-hex SHA-256 the downloaded bytes must carry.
    sha256: String,
    /// Pinned exact size, checked before the (much more expensive) digest.
    bytes: u64,
}

/// The embedded `ftts pull` download contract: release coordinates plus per-file pins.
#[derive(Clone, Debug)]
struct ModelManifest {
    model_id: String,
    release_tag: String,
    repo: String,
    files: Vec<ModelManifestFile>,
}

impl ModelManifest {
    /// The compiled-in manifest. Parsing it can only fail if the checked-in copy is malformed,
    /// which the unit tests catch before a binary ships.
    fn embedded() -> Result<Self, FttsError> {
        Self::parse(PINNED_MODEL_MANIFEST)
    }

    /// Parses and validates manifest text; every refusal names the offending field, because a
    /// manifest bug otherwise surfaces as a mystery mid-download.
    fn parse(text: &str) -> Result<Self, FttsError> {
        let value: Value = serde_json::from_str(text).map_err(|error| {
            FttsError::ArtifactFormat(format!("model manifest is not valid JSON: {error}"))
        })?;
        if value["schema_version"].as_u64() != Some(1) {
            return Err(FttsError::ArtifactFormat(format!(
                "model manifest schema_version {} is not the supported 1",
                value["schema_version"]
            )));
        }
        let model_id = manifest_string(&value, "model_id")?;
        let release_tag = manifest_string(&value, "release_tag")?;
        let repo = manifest_string(&value, "repo")?;
        let files = value["files"]
            .as_array()
            .filter(|files| !files.is_empty())
            .ok_or_else(|| {
                FttsError::ArtifactFormat("model manifest needs a non-empty files array".to_owned())
            })?
            .iter()
            .map(parse_manifest_file)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            model_id,
            release_tag,
            repo,
            files,
        })
    }

    /// The release-asset URL for one file; the only endpoint `ftts pull` ever contacts.
    fn download_url(&self, file: &ModelManifestFile) -> String {
        format!(
            "https://github.com/{}/releases/download/{}/{}",
            self.repo, self.release_tag, file.asset
        )
    }

    fn total_bytes(&self) -> u64 {
        self.files
            .iter()
            .fold(0, |sum, file| sum.saturating_add(file.bytes))
    }
}

fn manifest_string(value: &Value, field: &str) -> Result<String, FttsError> {
    value[field]
        .as_str()
        .filter(|text| !text.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            FttsError::ArtifactFormat(format!(
                "model manifest field {field} must be a non-empty string"
            ))
        })
}

fn parse_manifest_file(value: &Value) -> Result<ModelManifestFile, FttsError> {
    let asset = manifest_string(value, "asset")?;
    if asset.contains('/') || asset.contains('\\') {
        return Err(FttsError::ArtifactFormat(format!(
            "manifest asset {asset:?} must be a bare release-asset name"
        )));
    }
    let dest = manifest_string(value, "dest")?;
    validate_manifest_dest(&dest)?;
    let sha256 = manifest_string(value, "sha256")?;
    if !is_sha256_hex(&sha256) {
        return Err(FttsError::ArtifactFormat(format!(
            "manifest sha256 for {asset} must be 64 lowercase hex characters"
        )));
    }
    let bytes = value["bytes"]
        .as_u64()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            FttsError::ArtifactFormat(format!(
                "manifest bytes for {asset} must be a positive integer"
            ))
        })?;
    Ok(ModelManifestFile {
        asset,
        dest,
        sha256,
        bytes,
    })
}

/// A manifest `dest` is joined under the model directory, so it must not be able to escape it:
/// no absolute paths, no `..`, no `.`, no backslash separators.
fn validate_manifest_dest(dest: &str) -> Result<(), FttsError> {
    let path = Path::new(dest);
    let traversal_free = path
        .components()
        .all(|component| matches!(component, std::path::Component::Normal(_)));
    if path.is_absolute() || dest.contains('\\') || !traversal_free {
        return Err(FttsError::ArtifactFormat(format!(
            "manifest dest {dest:?} must be a relative path with no traversal; it is joined under the model directory"
        )));
    }
    Ok(())
}

fn is_sha256_hex(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// The directory `ftts pull` fills when `--model` is absent, and the last resort model resolution
/// falls back to: the first `FTTS_MODEL_DIR` entry when set, else `$HOME/.cache/franken_tts/model`.
fn default_pull_model_dir(environment: &Environment) -> Option<PathBuf> {
    if let Some(first) = environment
        .value("FTTS_MODEL_DIR")
        .and_then(|dirs| std::env::split_paths(dirs).next())
        .filter(|path| !path.as_os_str().is_empty())
    {
        return Some(first);
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(DEFAULT_MODEL_CACHE_SUBDIR))
}

/// Skip-versus-download for one manifest file, factored out so it is testable without a network.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PullDecision {
    /// The destination already carries the pinned size AND the pinned digest.
    Skip,
    /// Absent, wrong-sized, wrong-hashed, or `--force`.
    Download,
}

/// `Skip` requires both pins to hold: size alone accepts a same-length corruption, and hashing
/// alone would digest gigabytes a length check could reject for free (which is why the cheap check
/// runs first). Any mismatch re-downloads rather than erroring — repairing a bad file is exactly
/// what `pull` is for.
fn pull_decision(dest: &Path, file: &ModelManifestFile, force: bool) -> PullDecision {
    if force {
        return PullDecision::Download;
    }
    let Ok(metadata) = fs::metadata(dest) else {
        return PullDecision::Download;
    };
    if !metadata.is_file() || metadata.len() != file.bytes {
        return PullDecision::Download;
    }
    match ftts_artifacts::sha256::hex_digest_file(dest) {
        Ok(digest) if digest == file.sha256 => PullDecision::Skip,
        _ => PullDecision::Download,
    }
}

/// Downloads `url` to `staging` with the system `curl`.
///
/// Shelling out is deliberate: inference never touches the network, so the binary links no HTTP
/// stack. Only `pull` needs one, and `curl` is the same system-tool seam the audio encoders and
/// decoders already use.
fn download_with_curl(url: &str, staging: &Path, pinned_bytes: u64) -> Result<(), FttsError> {
    // Hardening beyond the happy path: HTTPS only through every redirect (a release URL should
    // never bounce through http), a connect timeout and a stall detector instead of hanging a
    // silent `-sS` transfer forever, and the pinned size as a hard transfer cap so a
    // misbehaving endpoint cannot fill the disk before the post-download size check runs.
    let outcome = std::process::Command::new("curl")
        .args([
            "-L",
            "--fail",
            "--retry",
            "3",
            "-sS",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
            "--max-filesize",
        ])
        .arg(pinned_bytes.to_string())
        .arg("-o")
        .arg(staging)
        .arg(url)
        .status();
    match outcome {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Err(FttsError::Generic(
            "`ftts pull` downloads with the system `curl`, which was not found on PATH; \
             install curl and retry, or download the release assets by hand"
                .to_owned(),
        )),
        Err(error) => Err(FttsError::Generic(format!("cannot run curl: {error}"))),
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(FttsError::Generic(format!(
            "curl failed downloading {url} ({status}); check network access and retry `ftts pull`"
        ))),
    }
}

/// Size first, digest second, both against the embedded pins.
fn verify_pulled_file(path: &Path, file: &ModelManifestFile) -> Result<(), FttsError> {
    let metadata = fs::metadata(path).map_err(|error| {
        FttsError::Generic(format!(
            "cannot stat downloaded {}: {error}",
            path.display()
        ))
    })?;
    if metadata.len() != file.bytes {
        return Err(FttsError::ArtifactFormat(format!(
            "downloaded {} is {} bytes, expected {}; the incomplete download was discarded, retry `ftts pull`",
            file.asset,
            metadata.len(),
            file.bytes
        )));
    }
    let digest = ftts_artifacts::sha256::hex_digest_file(path).map_err(|error| {
        FttsError::Generic(format!(
            "cannot hash downloaded {}: {error}",
            path.display()
        ))
    })?;
    if digest != file.sha256 {
        return Err(FttsError::ArtifactFormat(format!(
            "downloaded {} carries sha256 {digest}, expected {}; the corrupt download was discarded, retry `ftts pull`",
            file.asset, file.sha256
        )));
    }
    Ok(())
}

/// `<dest>.part`, in the destination directory so the final rename never crosses a filesystem.
fn pull_staging_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(OsString::from).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// Downloads one asset to `<dest>.part`, verifies it against the pins, and atomically publishes.
///
/// A staging file that fails verification is removed. This is the opposite of `convert`'s
/// retained staging, on purpose: a failed local conversion is diagnosable evidence, while a
/// corrupt download says nothing beyond "the network truncated it", and a multi-gigabyte corpse
/// in the cache directory helps no one.
fn pull_one_file(
    manifest: &ModelManifest,
    file: &ModelManifestFile,
    dest: &Path,
) -> Result<(), FttsError> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            FttsError::Generic(format!(
                "cannot create model directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let staging = pull_staging_path(dest);
    // The staging name is predictable, and `curl -o` opens it with a plain create-or-truncate
    // that follows symlinks. A pre-planted entry (stale crash debris, or a symlink in a shared
    // model directory) must be cleared first, checked via symlink_metadata so a link is seen as
    // itself rather than its target.
    match fs::symlink_metadata(&staging) {
        Ok(_) => fs::remove_file(&staging).map_err(|error| {
            FttsError::Generic(format!(
                "cannot clear stale staging file {}: {error}",
                staging.display()
            ))
        })?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(FttsError::Generic(format!(
                "cannot stat staging path {}: {error}",
                staging.display()
            )));
        }
    }
    let url = manifest.download_url(file);
    let outcome = download_with_curl(&url, &staging, file.bytes)
        .and_then(|()| verify_pulled_file(&staging, file));
    if let Err(error) = outcome {
        let _ = fs::remove_file(&staging);
        return Err(error);
    }
    // Durability before publish: rename orders the directory entry, not the data. Without the
    // fsync a crash can leave a truncated file at the verified name — and the tokenizer/codec
    // sidecars, unlike the .fttsq, carry no load-time digest to catch that. Same contract as
    // FttsqWriter::write_to_path. The handle must be writable: Windows refuses to flush a
    // read-only handle (ERROR_ACCESS_DENIED), while Unix fsync accepts one.
    fs::OpenOptions::new()
        .write(true)
        .open(&staging)
        .and_then(|file| file.sync_all())
        .map_err(|error| {
            FttsError::Generic(format!(
                "cannot fsync downloaded {}: {error}",
                staging.display()
            ))
        })?;
    fs::rename(&staging, dest).map_err(|error| {
        FttsError::Generic(format!(
            "downloaded {} verified but could not be published to {}: {error}",
            file.asset,
            dest.display()
        ))
    })
}

fn run_pull(
    args: &PullArgs,
    environment: &Environment,
    stdout: &mut dyn Write,
) -> Result<(), FttsError> {
    let manifest = ModelManifest::embedded()?;
    let destination = match &args.model {
        Some(path) => path.clone(),
        None => default_pull_model_dir(environment).ok_or_else(|| {
            FttsError::Usage(
                "cannot choose a model directory: pass --model PATH, or set FTTS_MODEL_DIR or HOME"
                    .to_owned(),
            )
        })?,
    };
    writeln!(
        stdout,
        "pulling {} ({} files, {} bytes) into {}",
        manifest.model_id,
        manifest.files.len(),
        manifest.total_bytes(),
        destination.display()
    )
    .map_err(output_error)?;
    for file in &manifest.files {
        let dest = destination.join(&file.dest);
        match pull_decision(&dest, file, args.force) {
            PullDecision::Skip => writeln!(
                stdout,
                "{} ({} bytes): already present, verified",
                file.dest, file.bytes
            )
            .map_err(output_error)?,
            PullDecision::Download => {
                writeln!(stdout, "{} ({} bytes): downloading", file.dest, file.bytes)
                    .map_err(output_error)?;
                pull_one_file(&manifest, file, &dest)?;
                writeln!(stdout, "{} ({} bytes): verified", file.dest, file.bytes)
                    .map_err(output_error)?;
            }
        }
    }
    writeln!(stdout, "model ready at {}", destination.display()).map_err(output_error)
}

fn resolve_existing_file<'a>(path: &'a Path, label: &str) -> Result<&'a Path, FttsError> {
    if path.is_file() {
        Ok(path)
    } else {
        Err(FttsError::ModelNotFound(format!(
            "{label} {} does not exist or is not a file; use an existing PATH",
            path.display()
        )))
    }
}

/// Preflight admission for `say --check`, computed by the **engine**, not by the CLI.
///
/// This used to be a CLI-local heuristic whose own text said "model-specific KV and memory
/// admission is pending the V_REL engine". That engine now exists, so the preflight calls
/// [`ftts_core::admission`] directly. The point is not code reuse: it is that `--check` and the
/// synthesis that follows it must reach the *same* verdict for the same request. A preflight that
/// says yes and an engine that then says no is worse than no preflight, because the caller
/// budgeted on the first answer.
///
/// Prompt length is not knowable before tokenization, so `--check` reports the admission decision
/// for an *estimated* prompt length and labels it as such. The binding decision remains the
/// engine's, taken after real tokenization.
fn admission_plan(text: &str, settings: &EffectiveSettings) -> Result<Value, FttsError> {
    if text.len() > SCAFFOLD_ADMISSION_TEXT_LIMIT_BYTES {
        return Err(FttsError::BudgetTimeout(format!(
            "text is {} bytes, above the Phase-0 admission bound of {} bytes; split the document before retrying",
            text.len(),
            SCAFFOLD_ADMISSION_TEXT_LIMIT_BYTES
        )));
    }

    let characters = text.chars().count();
    // A deliberately conservative stand-in until the tokenizer is on this path: over-estimating
    // the prompt can only make the preflight refuse something the engine would admit, which is the
    // safe direction. Under-estimating would promise capacity that is not there.
    let estimated_prompt_tokens = u64::try_from(characters).unwrap_or(u64::MAX);
    // The engine's own env-resolved policy (FTTS_MEMORY_BUDGET_MB / FTTS_MAX_FRAMES), not a copy
    // of it — a second parse of the same variables is a second thing to drift.
    let policy = ftts_core::process_engine_config().admission;

    match policy.admit(estimated_prompt_tokens) {
        Ok(plan) => Ok(json!({
            "status": "accepted",
            "scope": "preflight on an ESTIMATED prompt length; the binding decision is the \
                      engine's, taken after tokenization",
            "text_bytes": text.len(),
            "text_characters": characters,
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "predicted_max_frames": plan.predicted_max_frames,
            "predicted_peak_bytes": plan.predicted_peak_bytes,
            "budget_bytes": plan.budget_bytes,
            "binding_constraint": plan.binding_constraint.as_str(),
            "packet_frames": settings.packet_frames.as_str(),
            "profile": settings.profile.as_str(),
        })),
        // AdmissionRejection's Display already carries the shortfall, the binding constraint and
        // what to do about it, so it is passed through rather than re-summarised into something
        // less specific.
        Err(rejection) => Err(FttsError::BudgetTimeout(rejection.to_string())),
    }
}

fn run_voice_inspect(path: &Path, stdout: &mut dyn Write) -> Result<(), FttsError> {
    let path = resolve_existing_file(path, "voice pack")?;
    write_json_line(
        stdout,
        &json!({
            "schema_version": ROBOT_SCHEMA_VERSION,
            "event": "voice_inspect",
            "path": path.display().to_string(),
            "status": "header_inspection_pending_artifact_reader",
        }),
    )
}

fn run_robot(
    command: RobotCommand,
    environment: &Environment,
    stdout: &mut dyn Write,
) -> Result<(), FttsError> {
    // Every object below is built from `robot::EventType`, so the discriminator and
    // schema_version cannot be forgotten, and the frozen contract test in ftts-conformance
    // fails if any of these stops matching the catalogue.
    let event = match command {
        RobotCommand::Schema => robot::schema_document(robot::DOCUMENTED_ENVIRONMENT),
        RobotCommand::Health => {
            let searched = model_search_paths(environment);
            let found = searched.iter().find(|path| looks_like_model_artifact(path));
            let mut object = robot::EventType::Health.event();
            object.insert("status".to_owned(), json!("phase0_skeleton"));
            object.insert("model_loaded".to_owned(), json!(false));
            // Presence is a magic-bytes header sniff, never a tensor load: `robot health` must
            // stay cheap enough for an agent to call it on every invocation.
            object.insert("model_present".to_owned(), json!(found.is_some()));
            object.insert(
                "model_path".to_owned(),
                json!(found.map(|path| path.display().to_string())),
            );
            object.insert(
                "model_dir".to_owned(),
                json!(environment.value("FTTS_MODEL_DIR")),
            );
            // Every directory consulted, so a resolution failure is actionable rather than a
            // bare "not found".
            object.insert(
                "searched".to_owned(),
                json!(
                    searched
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                ),
            );
            object.insert("stateless_default".to_owned(), json!(true));
            object.insert(
                "threads".to_owned(),
                json!(
                    environment
                        .value("FTTS_THREADS")
                        .and_then(|value| value.parse::<u64>().ok())
                ),
            );
            object.insert(
                "recommended_command".to_owned(),
                json!("ftts say --check --model PATH TEXT"),
            );
            Value::Object(object)
        }
        RobotCommand::Backends => {
            let mut object = robot::EventType::Backends.event();
            // Capability vs executed-route split: `available` is every tier this build can
            // certify on this CPU; `dispatched` is the one the int8 route would actually run.
            object.insert(
                "available".to_owned(),
                json!(
                    ftts_kernels::int8::Int8Tier::available()
                        .iter()
                        .map(|tier| tier.as_str())
                        .collect::<Vec<_>>()
                ),
            );
            object.insert(
                "dispatched".to_owned(),
                json!(ftts_kernels::int8::Int8Tier::dispatch().as_str()),
            );
            object.insert("isa_features".to_owned(), json!(detected_isa_features()));
            let plan = ftts_kernels::int8::autotuned_plan();
            object.insert(
                "kernel_plan".to_owned(),
                json!({
                    "version": 0,
                    "decode_gemv": plan.decode_gemv.as_str(),
                    "batch_gemm": plan.batch_gemm.as_str(),
                    "persisted": false,
                }),
            );
            object.insert("pool_sizing".to_owned(), Value::Null);
            object.insert(
                "force_arch".to_owned(),
                json!(environment.value("FTTS_FORCE_ARCH")),
            );
            Value::Object(object)
        }
        RobotCommand::Selftest => {
            // The permanent integer-kernel law, executed on the end user's silicon: every census
            // binding row through the real dot kernels on every dispatchable tier. The event's
            // top-level fields are pinned by the frozen v1 schema fixture (status/reason/checks);
            // per-row detail lives inside `checks`.
            let report = ftts_kernels::selftest::run_selftest();
            let checks: Vec<Value> = report
                .checks
                .iter()
                .map(|check| {
                    json!({
                        "row": check.row.id,
                        "scope": check.row.scope.as_str(),
                        "census_tensor": check.row.census_tensor,
                        "reduction_k": check.row.reduction_k,
                        "tier": check.tier.as_str(),
                        "contract": check.contract.as_str(),
                        "dispatched": check.tier == report.dispatched,
                        "accumulator_i32": check.accumulator_i32,
                        "reference_i64": check.reference_i64,
                        "passed": check.passed,
                    })
                })
                .collect();
            let mut object = robot::EventType::Selftest.event();
            object.insert(
                "status".to_owned(),
                json!(if report.passed() { "passed" } else { "failed" }),
            );
            object.insert("reason".to_owned(), Value::Null);
            object.insert("checks".to_owned(), json!(checks));
            Value::Object(object)
        }
    };
    write_json_line(stdout, &event)
}

/// Every path the model resolver consults, in order.
///
/// Shared by `robot health` and the resolution error so the two can never disagree about what
/// was searched — a "not found" that lists different directories than `health` reports is worse
/// than no list at all.
fn model_search_paths(environment: &Environment) -> Vec<PathBuf> {
    let mut searched = environment
        .value("FTTS_MODEL_DIR")
        .map(std::env::split_paths)
        .map(|paths| {
            paths
                .map(|path| path.join(MODEL_BASENAME))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        searched.push(home.join(".cache/franken_tts/models").join(MODEL_BASENAME));
        // The `ftts pull` destination, listed after the legacy plural directory so existing
        // installs keep winning; it is also the bundle-directory fallback in `resolve_model_from`,
        // which is what lets a bare `ftts pull` then `ftts say` work with no env var at all.
        searched.push(home.join(DEFAULT_MODEL_CACHE_SUBDIR).join(MODEL_BASENAME));
    }
    searched
}

/// A cheap header sniff: is there a plausible `.fttsq` artifact at this path?
///
/// Reads the magic bytes only. Deliberately never opens the tensor data — `robot health` is
/// meant to be callable on every agent invocation, and a multi-gigabyte read would make it a
/// thing agents avoid calling, which defeats the point.
fn looks_like_model_artifact(path: &Path) -> bool {
    use std::io::Read as _;

    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut magic = [0u8; 5];
    file.read_exact(&mut magic).is_ok() && &magic == b"FTTSQ"
}

/// ISA features detected at runtime, for `robot backends`.
///
/// Reported as a plain list so an agent can see what the dispatcher had available; the kernel
/// tiers themselves land with the Phase-3 engines.
fn detected_isa_features() -> Vec<&'static str> {
    let mut features = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            features.push("dotprod");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            features.push("i8mm");
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("avxvnni") {
            features.push("avx-vnni");
        }
        if std::arch::is_x86_feature_detected!("avx512vnni") {
            features.push("avx512-vnni");
        }
    }
    features
}

fn run_doctor(
    args: &DoctorArgs,
    environment: &Environment,
    stdout: &mut dyn Write,
) -> Result<(), FttsError> {
    let report = json!({
        "schema_version": ROBOT_SCHEMA_VERSION,
        "status": "phase0_skeleton",
        "stateless_default": true,
        "persistent_history": false,
        "environment": environment.documented_values(),
        "recommended_command": "ftts robot schema",
    });
    if args.json {
        write_json_line(stdout, &report)
    } else {
        writeln!(stdout, "FrankenTTS Phase-0 CLI skeleton")
            .and_then(|_| writeln!(stdout, "stateless default: yes"))
            .and_then(|_| writeln!(stdout, "model loaded: no"))
            .and_then(|_| writeln!(stdout, "next: ftts robot schema"))
            .map_err(output_error)
    }
}

fn write_json_line(writer: &mut dyn Write, value: &Value) -> Result<(), FttsError> {
    serde_json::to_writer(&mut *writer, value)
        .map_err(|error| FttsError::Generic(format!("cannot serialize CLI JSON: {error}")))?;
    writer.write_all(b"\n").map_err(output_error)
}

fn output_error(error: io::Error) -> FttsError {
    FttsError::Generic(format!("cannot write CLI output: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn preset_voices_are_valid_speaker_vectors() {
        assert!(
            PRESET_VOICES
                .iter()
                .any(|(name, _, _)| *name == DEFAULT_PRESET_VOICE),
            "the default preset must exist in the table"
        );
        for (name, character, bytes) in PRESET_VOICES {
            assert_eq!(
                bytes.len(),
                synth::SPEAKER_VECTOR_BYTES,
                "preset {name} must be exactly one 1,024-float x-vector"
            );
            assert!(
                !character.is_empty(),
                "preset {name} needs a character line"
            );
            for chunk in bytes.as_chunks::<4>().0 {
                let value = f32::from_le_bytes(*chunk);
                assert!(
                    value.is_finite(),
                    "preset {name} carries a non-finite value"
                );
            }
        }
    }

    #[test]
    fn preset_names_resolve_and_unknown_names_do_not() {
        let environment = Environment {
            values: BTreeMap::new(),
            stage_budget_values: BTreeMap::new(),
        };
        let resolved = resolve_requested_voice(Some(Path::new("aria")), &environment)
            .expect("preset name resolves")
            .expect("preset yields a path");
        let bytes = fs::read(&resolved).expect("materialized preset readable");
        assert_eq!(bytes.len(), synth::SPEAKER_VECTOR_BYTES);

        let error = resolve_requested_voice(Some(Path::new("no-such-voice")), &environment)
            .expect_err("unknown names are refused");
        assert!(
            error.to_string().contains("aria"),
            "the refusal must list the built-in names, got: {error}"
        );
    }

    // Updated 2026-08-10 for the `make-video` subcommand, which landed without re-baselining this
    // snapshot. The snapshot exists to make CLI-surface changes deliberate rather than accidental,
    // so it is re-baselined only alongside a real, intended command — never widened to stop failing.
    const CLAP_SURFACE_SNAPSHOT: &str = "commands=say,make-video,enroll,voice,convert,pull,robot,doctor,resident-daemon\nrobot=schema,health,backends,selftest\nsay=file,model,voice,output,stream,check,robot,no-resident\npull=model,force\nglobal=profile,packet-frames,math-mode,voice-pack,normalize,trace,seed\n";

    #[test]
    fn clap_surface_matches_snapshot() {
        let command = Cli::command();
        let commands = command
            .get_subcommands()
            .map(|command| command.get_name())
            .collect::<Vec<_>>()
            .join(",");
        let robot = command
            .get_subcommands()
            .find(|command| command.get_name() == "robot")
            .expect("robot subcommand")
            .get_subcommands()
            .map(|command| command.get_name())
            .collect::<Vec<_>>()
            .join(",");
        let say = command
            .get_subcommands()
            .find(|command| command.get_name() == "say")
            .expect("say subcommand")
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .collect::<Vec<_>>()
            .join(",");
        let pull = command
            .get_subcommands()
            .find(|command| command.get_name() == "pull")
            .expect("pull subcommand")
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .collect::<Vec<_>>()
            .join(",");
        let global = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .filter(|argument| *argument != "help")
            .collect::<Vec<_>>()
            .join(",");
        let actual = format!(
            "commands={commands}\nrobot={robot}\nsay={say}\npull={pull}\nglobal={global}\n"
        );
        assert_eq!(actual, CLAP_SURFACE_SNAPSHOT);
    }

    #[test]
    fn argument_file_and_stdin_text_are_identical() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../README.md");
        let expected = fs::read_to_string(&root).expect("checked-in README");
        let from_argument = read_text(
            &SayArgs {
                output_positional: None,
                text: Some(expected.clone()),
                file: None,
                model: None,
                voice: None,
                output: None,
                stream: None,
                check: true,
                robot: false,
                no_resident: true,
            },
            &mut Cursor::new(Vec::<u8>::new()),
        )
        .expect("argument text");
        let from_file = read_text(
            &SayArgs {
                output_positional: None,
                text: None,
                file: Some(root),
                model: None,
                voice: None,
                output: None,
                stream: None,
                check: true,
                robot: false,
                no_resident: true,
            },
            &mut Cursor::new(Vec::<u8>::new()),
        )
        .expect("file text");
        let from_stdin = read_text(
            &SayArgs {
                output_positional: None,
                text: Some("-".to_owned()),
                file: None,
                model: None,
                voice: None,
                output: None,
                stream: None,
                check: true,
                robot: false,
                no_resident: true,
            },
            &mut Cursor::new(expected.as_bytes()),
        )
        .expect("stdin text");

        assert_eq!(from_argument, from_file);
        assert_eq!(from_argument, from_stdin);
    }

    #[test]
    fn check_plan_is_deterministic_and_marks_its_scope() {
        let settings = EffectiveSettings {
            profile: ExecutionProfile::Strict,
            packet_frames: PacketFrames::Four,
            math_mode: MathMode::Strict,
            voice_pack: VoicePackProfile::Portable,
            normalize: NormalizeMode::Conservative,
        };
        let first = admission_plan("hello", &settings).expect("admission plan");
        let second = admission_plan("hello", &settings).expect("admission plan");
        assert_eq!(first, second);
        assert_eq!(first["status"], "accepted");
        // The preflight is explicit that it estimates the prompt and that the engine decides.
        assert!(
            first["scope"]
                .as_str()
                .unwrap_or_default()
                .contains("ESTIMATED")
        );
    }

    #[test]
    fn the_cli_preflight_and_the_engine_agree_on_the_same_request() {
        // The property that matters: `--check` saying yes and the engine then saying no is worse
        // than no preflight, because the caller budgeted on the first answer. Both must be the
        // same computation, not two implementations of the same rule.
        let settings = EffectiveSettings {
            profile: ExecutionProfile::Balanced,
            packet_frames: PacketFrames::Four,
            math_mode: MathMode::Strict,
            voice_pack: VoicePackProfile::Portable,
            normalize: NormalizeMode::Verbatim,
        };
        let text = "a moderately sized utterance for admission";
        let plan = admission_plan(text, &settings).expect("preflight admits");

        let policy = ftts_core::process_engine_config().admission;
        let engine = policy
            .admit(text.chars().count() as u64)
            .expect("engine admits the same request");

        assert_eq!(plan["predicted_peak_bytes"], engine.predicted_peak_bytes);
        assert_eq!(plan["predicted_max_frames"], engine.predicted_max_frames);
        assert_eq!(plan["budget_bytes"], engine.budget_bytes);
        assert_eq!(
            plan["binding_constraint"],
            engine.binding_constraint.as_str()
        );
    }

    #[test]
    fn a_wav_sink_writes_a_playable_file_and_conforming_audio_chunk_events() {
        let dir = std::env::temp_dir().join(format!("ftts-wav-sink-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("out.wav");

        let frame: Vec<f32> = (0..1_920)
            .map(|i| (i as f32 / 1_920.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        let mut sink = AudioOutput::wav(&path).expect("wav sink");
        let mut discard = Vec::new();

        let first = sink
            .write_packet(&frame, &mut discard, "run-1", 1)
            .expect("packet 1");
        let second = sink
            .write_packet(&frame, &mut discard, "run-1", 1)
            .expect("packet 2");

        // Every emitted object must satisfy the frozen robot contract, not merely look plausible.
        assert!(robot::validate_event(&first).is_empty(), "{first:?}");
        assert!(robot::validate_event(&second).is_empty(), "{second:?}");
        assert_eq!(first["sink"], "file");
        assert_eq!(first["byte_offset"], 0);
        assert_eq!(first["bytes"], 1_920 * 2);
        assert_eq!(first["duration_ms"], 80, "1,920 samples at 24 kHz is 80 ms");
        // The offset is cumulative, so a consumer can seek with it.
        assert_eq!(second["byte_offset"], 1_920 * 2);

        let samples = sink.finish().expect("finish");
        assert_eq!(samples, 1_920 * 2);

        // The file on disk must describe exactly what it holds.
        let bytes = std::fs::read(&path).expect("read wav");
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let declared = u32::from_le_bytes(bytes[40..44].try_into().expect("data size"));
        assert_eq!(declared as usize, 1_920 * 2 * 2);
        assert_eq!(bytes.len(), 44 + 1_920 * 2 * 2);
        assert!(
            discard.is_empty(),
            "a file sink must not also emit raw PCM to the stream"
        );
    }

    #[test]
    fn a_raw_sink_writes_pcm_to_the_stream_and_never_mixes_it_with_events() {
        // The stream contract: under --stream raw, stdout carries PCM only. An event object landing
        // in the same buffer would corrupt both — the audio and the NDJSON.
        let mut sink = AudioOutput::raw();
        let mut raw = Vec::new();
        let pcm = vec![0.5f32; 4];
        let event = sink
            .write_packet(&pcm, &mut raw, "run-1", 1)
            .expect("packet");

        assert!(robot::validate_event(&event).is_empty(), "{event:?}");
        assert_eq!(event["sink"], "stdout");
        assert_eq!(raw.len(), 8, "four 16-bit samples");
        let first = i16::from_le_bytes([raw[0], raw[1]]);
        assert_eq!(first, ftts_core::audio::sample_to_i16(0.5));
        // The PCM buffer must contain no JSON.
        assert!(
            !raw.windows(2).any(|w| w == b"{\""),
            "raw PCM stream must never contain an event object"
        );
    }

    #[test]
    fn a_none_sink_still_reports_conforming_events() {
        let mut sink = AudioOutput::none();
        let mut discard = Vec::new();
        let event = sink
            .write_packet(&[0.0f32; 960], &mut discard, "run-1", 2)
            .expect("packet");
        assert!(robot::validate_event(&event).is_empty(), "{event:?}");
        assert_eq!(event["sink"], "none");
        assert_eq!(event["packet_frames"], "2");
        assert!(discard.is_empty());
        assert_eq!(sink.finish().expect("finish"), 960);
    }

    #[test]
    fn a_health_violation_renders_as_a_contract_conforming_robot_event() {
        // The engine-to-wire seam: the violation's class, remedy and invalidates_output must
        // survive the crossing, and the result must satisfy the frozen robot contract.
        let silent =
            ftts_core::HealthEvent::Violation(ftts_core::health::HealthViolation::OutputSilent {
                silent_millis: 1_500,
            });
        let event = robot::health_violation_event("run-1", silent, 42);
        assert!(
            robot::validate_event(&event).is_empty(),
            "{:?}",
            robot::validate_event(&event)
        );
        assert_eq!(event["event"], "health_violation");
        assert_eq!(event["violation"], "output_silent");
        assert_eq!(event["invalidates_output"], true);
        assert!(event["detail"].as_str().expect("detail").contains("1500"));
        assert!(event["remedy"].as_str().expect("remedy").len() > 40);

        // A kernel demotion is informational: the run stayed correct, just slower. If this were
        // reported as invalidating, an agent would discard good audio.
        let demoted =
            ftts_core::HealthEvent::Violation(ftts_core::health::HealthViolation::KernelDemoted {
                from: ftts_core::health::KernelTier::Optimized("i8mm"),
                to: ftts_core::health::KernelTier::Scalar,
            });
        let event = robot::health_violation_event("run-1", demoted, 43);
        assert!(robot::validate_event(&event).is_empty());
        assert_eq!(event["invalidates_output"], false);

        // Budget and cancellation are health signals too, and both truncate the audio.
        for event in [
            ftts_core::HealthEvent::BudgetExceeded,
            ftts_core::HealthEvent::Cancelled,
        ] {
            let rendered = robot::health_violation_event("run-1", event, 44);
            assert!(robot::validate_event(&rendered).is_empty());
            assert_eq!(rendered["invalidates_output"], true);
        }
    }

    #[test]
    fn normalization_defaults_to_verbatim_conformance_mode() {
        let cli = Cli {
            profile: None,
            packet_frames: None,
            math_mode: None,
            voice_pack: None,
            normalize: None,
            trace: None,
            seed: None,
            command: Command::Robot(RobotArgs {
                command: RobotCommand::Health,
            }),
        };
        assert_eq!(
            EffectiveSettings::resolve(&cli, &Environment::default())
                .expect("default settings")
                .normalize,
            NormalizeMode::Verbatim
        );
        assert_eq!(
            EffectiveSettings::resolve(&cli, &Environment::default())
                .expect("default settings")
                .normalization_options(),
            NormalizationOptions::default(),
            "CLI defaults must use the same verbatim options as the library"
        );
    }

    #[test]
    fn cli_normalization_modes_map_to_shared_engine_options() {
        for (cli_mode, engine_mode) in [
            (NormalizeMode::Verbatim, NormalizationMode::Verbatim),
            (NormalizeMode::Conservative, NormalizationMode::Conservative),
            (NormalizeMode::LocaleAware, NormalizationMode::LocaleAware),
        ] {
            let settings = EffectiveSettings {
                profile: ExecutionProfile::Balanced,
                packet_frames: PacketFrames::Four,
                math_mode: MathMode::Strict,
                voice_pack: VoicePackProfile::Portable,
                normalize: cli_mode,
            };
            assert_eq!(settings.normalization_options().mode, engine_mode);
        }
    }

    #[test]
    fn pinned_main_conversion_plan_preserves_the_reviewed_q8_boundary() {
        let specs = pinned_main_tensor_specs().expect("checked-in main inventory parses");
        let (_manifest, _plan) =
            pinned_main_conversion_plan().expect("checked-in main conversion plan builds");
        assert_eq!(specs.len(), PINNED_MAIN_TENSOR_COUNT);
        assert_eq!(
            specs
                .iter()
                .filter(|spec| spec.storage == TensorStoragePolicy::Q8PerOutputChannel)
                .count(),
            231,
            "28 talker + 5 microdecoder layers times seven attention/MLP projections"
        );

        let text_embedding = specs
            .iter()
            .find(|spec| spec.name == "talker.model.text_embedding.weight");
        assert!(
            text_embedding.is_some(),
            "pinned inventory must contain the text embedding"
        );
        if let Some(text_embedding) = text_embedding {
            assert_eq!(text_embedding.storage, TensorStoragePolicy::Verbatim);
            assert_eq!(text_embedding.access_class, AccessClass::ColdTextEmbedding);
        }

        let talker_projection = specs
            .iter()
            .find(|spec| spec.name == "talker.model.layers.0.mlp.down_proj.weight");
        assert!(
            talker_projection.is_some(),
            "pinned inventory must contain the talker projection"
        );
        if let Some(talker_projection) = talker_projection {
            assert_eq!(
                talker_projection.storage,
                TensorStoragePolicy::Q8PerOutputChannel
            );
            assert_eq!(
                talker_projection.access_class,
                AccessClass::HotRecurrentTalker
            );
        }

        let micro_projection = specs
            .iter()
            .find(|spec| spec.name == "talker.code_predictor.model.layers.0.mlp.down_proj.weight");
        assert!(
            micro_projection.is_some(),
            "pinned inventory must contain the microdecoder projection"
        );
        if let Some(micro_projection) = micro_projection {
            assert_eq!(
                micro_projection.storage,
                TensorStoragePolicy::Q8PerOutputChannel
            );
            assert_eq!(
                micro_projection.access_class,
                AccessClass::HotRecurrentMicrodecoder
            );
        }

        let primary_embedding = specs
            .iter()
            .find(|spec| spec.name == "talker.model.codec_embedding.weight");
        assert!(
            primary_embedding.is_some(),
            "pinned inventory must contain the primary-code embedding"
        );
        if let Some(primary_embedding) = primary_embedding {
            assert_eq!(
                primary_embedding.access_class,
                AccessClass::HotRecurrentMicrodecoder,
                "the primary-code embedding feeds residual depth one every frame"
            );
        }

        let primary_head = specs
            .iter()
            .find(|spec| spec.name == "talker.codec_head.weight");
        assert!(
            primary_head.is_some(),
            "pinned inventory must contain the primary-code head"
        );
        if let Some(primary_head) = primary_head {
            assert_eq!(primary_head.storage, TensorStoragePolicy::Verbatim);
            assert_eq!(primary_head.access_class, AccessClass::HotRecurrentTalker);
        }

        let text_projection = specs
            .iter()
            .find(|spec| spec.name == "talker.text_projection.linear_fc1.weight");
        assert!(
            text_projection.is_some(),
            "pinned inventory must contain the text-projection MLP"
        );
        if let Some(text_projection) = text_projection {
            assert_eq!(text_projection.storage, TensorStoragePolicy::Verbatim);
            assert_eq!(
                text_projection.access_class,
                AccessClass::HotRecurrentTalker
            );
        }

        let head = specs
            .iter()
            .find(|spec| spec.name == "talker.code_predictor.lm_head.0.weight");
        assert!(
            head.is_some(),
            "pinned inventory must contain the residual-code head"
        );
        if let Some(head) = head {
            assert_eq!(head.storage, TensorStoragePolicy::Verbatim);
            assert_eq!(head.access_class, AccessClass::HotRecurrentMicrodecoder);
        }

        let speaker = specs
            .iter()
            .find(|spec| spec.name == "speaker_encoder.fc.weight");
        assert!(
            speaker.is_some(),
            "pinned inventory must contain the speaker encoder"
        );
        if let Some(speaker) = speaker {
            assert_eq!(speaker.storage, TensorStoragePolicy::Verbatim);
            assert_eq!(speaker.access_class, AccessClass::EnrollmentSpeakerEncoder);
        }
    }

    #[test]
    fn conversion_notice_carries_changes_and_the_full_license() {
        let notice = pinned_license_notice();
        assert!(notice.contains("Copyright 2026 Alibaba Cloud"));
        assert!(notice.contains("CHANGES: the original bfloat16 weights were converted"));
        assert!(notice.contains("Apache License"));
        assert!(notice.contains("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION"));
    }

    #[test]
    fn convert_refusal_still_emits_a_versioned_robot_lifecycle() {
        let cli = Cli {
            profile: None,
            packet_frames: None,
            math_mode: None,
            voice_pack: None,
            normalize: None,
            trace: None,
            seed: None,
            command: Command::Robot(RobotArgs {
                command: RobotCommand::Health,
            }),
        };
        let args = ConvertArgs {
            // A readable file with the wrong name reaches the explicit pinned-source refusal
            // before any destination is created.
            source: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
            output: PathBuf::from("never-created.fttsq"),
        };
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let error = run_convert(
            &cli,
            &args,
            &Environment::default(),
            &mut stdout,
            &mut stderr,
        )
        .expect_err("the non-pinned source must be refused");
        assert_eq!(error.exit_code(), FttsExitCode::Input);

        let stdout = String::from_utf8(stdout).expect("NDJSON stdout");
        let stderr = String::from_utf8(stderr).expect("NDJSON stderr");
        assert!(robot::validate_ndjson(&stdout).is_empty());
        assert!(robot::validate_ndjson(&stderr).is_empty());
        let stdout_events = stdout
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).expect("JSON event"))
            .collect::<Vec<_>>();
        assert_eq!(stdout_events[0]["event"], "run_start");
        assert_eq!(stdout_events[1]["event"], "stage");
        assert_eq!(
            serde_json::from_str::<Value>(stderr.trim()).expect("run error")["event"],
            "run_error"
        );
    }

    #[test]
    fn say_check_emits_a_versioned_admission_outcome() {
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let cli = Cli {
            profile: Some(ExecutionProfile::Balanced),
            packet_frames: Some(PacketFrames::Four),
            math_mode: Some(MathMode::Strict),
            voice_pack: Some(VoicePackProfile::Portable),
            normalize: Some(NormalizeMode::Conservative),
            trace: None,
            seed: Some(7),
            command: Command::Robot(RobotArgs {
                command: RobotCommand::Health,
            }),
        };
        let args = SayArgs {
            text: Some("checked text".to_owned()),
            output_positional: None,
            file: None,
            model: Some(model),
            voice: None,
            output: None,
            stream: None,
            check: true,
            robot: false,
            no_resident: true,
        };
        let mut stdin = Cursor::new(Vec::<u8>::new());
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();

        run_say(
            &cli,
            &args,
            &Environment::default(),
            &mut stdin,
            &mut stdout,
            &mut stderr,
        )
        .expect("check path");

        assert!(stderr.is_empty());
        let text = String::from_utf8(stdout).expect("utf-8 events");

        // The whole emitted stream must conform, not just the event this test cares about.
        assert!(
            robot::validate_ndjson(&text).is_empty(),
            "emitted stream violates the contract: {:?}",
            robot::validate_ndjson(&text)
        );

        let events: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).expect("one JSON object per line"))
            .collect();
        let names: Vec<&str> = events
            .iter()
            .map(|event| event["event"].as_str().expect("event name"))
            .collect();
        assert_eq!(
            names,
            vec![
                "run_start",
                "stage",
                "stage",
                "text_prepared",
                "stage",
                "stage",
                "check_complete",
                "run_complete",
            ],
            "the skeleton lifecycle must flow end-to-end on the empty pipeline"
        );

        // Every event in a run repeats the same run_id, which is what lets an agent stitch a run
        // together across the two streams.
        let run_id = events[0]["run_id"]
            .as_str()
            .expect("run_start carries run_id");
        assert!(!run_id.is_empty());
        assert!(events.iter().all(|event| event["run_id"] == run_id));
        assert!(
            events
                .iter()
                .all(|event| event["schema_version"] == ROBOT_SCHEMA_VERSION)
        );

        // Stage sequence numbers are dense and ordered, so a consumer can detect a dropped event.
        let seqs: Vec<u64> = events
            .iter()
            .filter(|event| event["event"] == "stage")
            .map(|event| event["seq"].as_u64().expect("seq"))
            .collect();
        assert_eq!(seqs, vec![0, 1, 2, 3]);

        let check = &events[6];
        // "accepted", not "scaffold_accepted": the preflight is now the engine's own
        // ftts_core::admission computation rather than a CLI-local heuristic, so `--check` and the
        // synthesis that follows it cannot reach different verdicts for the same request.
        assert_eq!(check["admission"]["status"], "accepted");
        assert!(
            check["admission"]["predicted_peak_bytes"].is_u64(),
            "the engine-backed plan reports a real predicted peak"
        );
        assert_eq!(check["normalization_trace_requested"], false);

        // text_prepared reports shape and provenance only; the input text must never appear.
        let prepared = &events[3];
        assert_eq!(prepared["char_count"], "checked text".chars().count());
        assert!(prepared["unicode_version"].is_string());
        assert!(
            !text.contains("checked text"),
            "the event stream must not carry the user's text"
        );

        assert_eq!(events[7]["exit_code"], 0);
    }

    #[test]
    fn a_newline_inside_a_field_cannot_break_ndjson_framing() {
        // The entire contract rests on one JSON object per line. serde_json escapes control
        // characters, so a message containing a newline stays one line and the newline survives as
        // data -- but nothing pinned that until now, and a hand-rolled serializer or a raw write
        // path would silently break every downstream parser.
        let run = robot::RunContext::with_id("r-test");
        let error = FttsError::Generic("first\nsecond".to_owned());
        let mut event = run.event(robot::EventType::RunError);
        event.insert("exit_code".to_owned(), json!(error.exit_code().as_u8()));
        event.insert("kind".to_owned(), json!(error.exit_code().description()));
        event.insert("message".to_owned(), json!(error.to_string()));
        event.insert("remediation".to_owned(), json!(error.remediation()));
        event.insert("elapsed_ms".to_owned(), json!(0));
        let value = Value::Object(event);

        let mut buffer = Vec::new();
        write_json_line(&mut buffer, &value).expect("serializes");
        let text = String::from_utf8(buffer).expect("utf-8");

        assert_eq!(
            text.lines().count(),
            1,
            "framing broken by an embedded newline"
        );
        assert!(robot::validate_ndjson(&text).is_empty());
        let parsed: Value = serde_json::from_str(text.trim_end()).expect("still one object");
        assert!(
            parsed["message"].as_str().expect("message").contains('\n'),
            "the newline must survive as data, not be stripped"
        );
    }

    #[test]
    fn pinned_copies_match_the_truth_pack_canonicals() {
        //  `pinned/` exists because `cargo package` cannot ship the truth pack. The truth pack
        //  stays canonical; a drifted copy would embed a stale pin assertion or attribution in
        //  the shipped binary, so byte-identity is asserted whenever the repo checkout is present.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for (canonical, embedded, name) in [
            (
                "docs/truth-pack/TENSOR_INVENTORY.json",
                PINNED_TENSOR_INVENTORY,
                "TENSOR_INVENTORY.json",
            ),
            (
                "docs/truth-pack/snapshots/hf/config.json",
                PINNED_MODEL_CONFIG,
                "model_config.json",
            ),
            (
                "docs/truth-pack/snapshots/gh/LICENSE",
                APACHE_LICENSE,
                "QWEN_APACHE_LICENSE",
            ),
        ] {
            match std::fs::read_to_string(root.join(canonical)) {
                Ok(bytes) => assert_eq!(
                    bytes, embedded,
                    "pinned/{name} drifted from {canonical}; re-copy it"
                ),
                Err(_) => eprintln!(
                    "SKIP pinned-copy check for {name}: {canonical} absent (no repo checkout)"
                ),
            }
        }
    }

    #[test]
    fn embedded_model_manifest_is_wellformed_and_agrees_with_the_converter_pin() {
        let manifest = ModelManifest::embedded().expect("embedded manifest parses");
        assert_eq!(manifest.model_id, "qwen3-tts-12hz-0.6b-base");
        assert_eq!(manifest.release_tag, "model-qwen3-tts-v1");
        assert_eq!(manifest.repo, "Dicklesworthstone/franken_tts");
        assert_eq!(manifest.files.len(), 8);

        for file in &manifest.files {
            assert!(
                is_sha256_hex(&file.sha256),
                "{} carries a malformed digest",
                file.asset
            );
            assert!(file.bytes > 0, "{} has no pinned size", file.asset);
            let dest = Path::new(&file.dest);
            assert!(!dest.is_absolute(), "{} dest is absolute", file.asset);
            assert!(
                dest.components()
                    .all(|component| matches!(component, std::path::Component::Normal(_))),
                "{} dest can traverse out of the model directory",
                file.asset
            );
        }

        // Since frankentts-zm5 the pull ships the canonical quantized artifact, not the raw main
        // checkpoint: enrollment and synthesis both hydrate from the .fttsq, so pulling the raw
        // 1.7 GB main would be pure waste. The artifact lands at the exact basename every model
        // search path probes for.
        let main = manifest
            .files
            .iter()
            .find(|file| file.dest == MODEL_BASENAME)
            .expect("manifest carries the canonical artifact");
        assert_eq!(
            manifest.download_url(main),
            "https://github.com/Dicklesworthstone/franken_tts/releases/download/model-qwen3-tts-v1/qwen3-tts-12hz-0.6b-base.fttsq"
        );
        assert!(
            !manifest
                .files
                .iter()
                .any(|file| file.dest == PINNED_MAIN_WEIGHTS_FILENAME),
            "pull must not fetch the raw main checkpoint alongside the canonical artifact"
        );

        // Together the files are exactly what ModelBundle::resolve requires plus the two config
        // sidecars, so a completed pull always resolves.
        let dests: Vec<&str> = manifest
            .files
            .iter()
            .map(|file| file.dest.as_str())
            .collect();
        for required in [
            MODEL_BASENAME,
            "speech_tokenizer/model.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
        ] {
            assert!(dests.contains(&required), "manifest is missing {required}");
        }
    }

    #[test]
    fn malformed_model_manifests_are_refused_with_the_field_named() {
        fn manifest_with(
            schema_version: u64,
            asset: &str,
            dest: &str,
            sha256: &str,
            bytes: u64,
        ) -> String {
            json!({
                "schema_version": schema_version,
                "model_id": "m",
                "release_tag": "t",
                "repo": "owner/repo",
                "files": [{"asset": asset, "dest": dest, "sha256": sha256, "bytes": bytes}],
            })
            .to_string()
        }
        let good_sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        // The happy shape parses, so every refusal below is attributable to its one bad field.
        ModelManifest::parse(&manifest_with(1, "a.bin", "a.bin", good_sha, 1))
            .expect("well-formed manifest parses");

        for (label, text) in [
            (
                "unsupported schema_version",
                manifest_with(2, "a.bin", "a.bin", good_sha, 1),
            ),
            (
                "short sha256",
                manifest_with(1, "a.bin", "a.bin", "abc123", 1),
            ),
            (
                "uppercase sha256",
                manifest_with(1, "a.bin", "a.bin", &good_sha.to_uppercase(), 1),
            ),
            (
                "zero bytes",
                manifest_with(1, "a.bin", "a.bin", good_sha, 0),
            ),
            (
                "absolute dest",
                manifest_with(1, "a.bin", "/etc/passwd", good_sha, 1),
            ),
            (
                "traversal dest",
                manifest_with(1, "a.bin", "../escape.bin", good_sha, 1),
            ),
            (
                "asset with a path separator",
                manifest_with(1, "dir/a.bin", "a.bin", good_sha, 1),
            ),
            (
                "empty files array",
                json!({
                    "schema_version": 1,
                    "model_id": "m",
                    "release_tag": "t",
                    "repo": "owner/repo",
                    "files": [],
                })
                .to_string(),
            ),
        ] {
            let error = ModelManifest::parse(&text)
                .expect_err(&format!("a manifest with {label} must be refused"));
            assert_eq!(error.exit_code(), FttsExitCode::ArtifactFormat, "{label}");
        }
    }

    #[test]
    fn pull_skips_only_a_file_matching_both_pinned_size_and_digest() {
        let dir = std::env::temp_dir().join(format!("ftts-pull-decision-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("temp dir");
        let payload = b"pinned payload";
        let file = ModelManifestFile {
            asset: "a.bin".to_owned(),
            dest: "a.bin".to_owned(),
            sha256: ftts_artifacts::sha256::hex_digest(payload),
            bytes: payload.len() as u64,
        };
        let dest = dir.join("a.bin");

        let _ = fs::remove_file(&dest);
        assert_eq!(
            pull_decision(&dest, &file, false),
            PullDecision::Download,
            "absent file must download"
        );

        fs::write(&dest, payload).expect("write verified payload");
        assert_eq!(
            pull_decision(&dest, &file, false),
            PullDecision::Skip,
            "matching size and digest must skip"
        );
        assert_eq!(
            pull_decision(&dest, &file, true),
            PullDecision::Download,
            "--force must re-download even a verified file"
        );

        fs::write(&dest, b"pinned_payload").expect("write same-length corruption");
        assert_eq!(
            pull_decision(&dest, &file, false),
            PullDecision::Download,
            "a same-length corruption must be caught by the digest"
        );

        fs::write(&dest, b"short").expect("write truncation");
        assert_eq!(
            pull_decision(&dest, &file, false),
            PullDecision::Download,
            "a truncated file must be caught by the size check"
        );
    }

    #[test]
    fn model_resolution_prefers_explicit_then_searched_then_the_pull_directory() {
        let root = std::env::temp_dir().join(format!("ftts-resolve-order-{}", std::process::id()));

        // A complete bundle directory: ModelBundle::resolve only asks `is_file`, so empty files
        // are a sufficient fake (and would fail loudly if the resolver ever started reading).
        let bundle = root.join("bundle");
        for relative in [
            "model.safetensors",
            "speech_tokenizer/model.safetensors",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
        ] {
            let path = bundle.join(relative);
            fs::create_dir_all(path.parent().expect("bundle parent")).expect("bundle dirs");
            fs::write(&path, b"").expect("bundle file");
        }
        assert!(
            synth::ModelBundle::resolve(&bundle).is_ok(),
            "five empty files must satisfy the resolver's is_file checks"
        );

        let searched_artifact = root.join("searched").join(MODEL_BASENAME);
        fs::create_dir_all(searched_artifact.parent().expect("searched parent"))
            .expect("searched dir");
        fs::write(&searched_artifact, b"").expect("searched artifact");
        let searched = vec![searched_artifact.clone()];
        let absent = vec![root.join("absent").join(MODEL_BASENAME)];

        // 1. `--model` outranks everything.
        assert_eq!(
            resolve_model_from(Some(&bundle), &searched, Some(&bundle)).expect("explicit"),
            bundle.display().to_string()
        );

        // 2. A searched artifact outranks the pull directory.
        assert_eq!(
            resolve_model_from(None, &searched, Some(&bundle)).expect("searched"),
            searched_artifact.display().to_string()
        );

        // 3. The pull directory resolves when nothing searched exists.
        assert_eq!(
            resolve_model_from(None, &absent, Some(&bundle)).expect("pull fallback"),
            bundle.display().to_string()
        );

        // 4. An incomplete pull directory does not resolve, and the error teaches `ftts pull`.
        let incomplete = root.join("incomplete");
        fs::create_dir_all(&incomplete).expect("incomplete dir");
        let error = resolve_model_from(None, &absent, Some(&incomplete))
            .expect_err("an empty pull directory must not resolve");
        assert_eq!(error.exit_code(), FttsExitCode::ModelNotFound);
        assert!(error.to_string().contains("ftts pull"), "{error}");
        assert!(error.to_string().contains("2.0 GB"), "{error}");
        assert!(error.to_string().contains("FTTS_MODEL_DIR"), "{error}");
    }

    #[test]
    fn the_pull_directory_default_prefers_the_env_override() {
        let mut environment = Environment::default();
        environment
            .values
            .insert("FTTS_MODEL_DIR", Some(OsString::from("/tmp/env-model-dir")));
        assert_eq!(
            default_pull_model_dir(&environment),
            Some(PathBuf::from("/tmp/env-model-dir"))
        );

        // Without the override the default lives under $HOME; the exact suffix is the contract
        // `ftts pull` and model resolution share.
        if std::env::var_os("HOME").is_some() {
            let fallback = default_pull_model_dir(&Environment::default())
                .expect("HOME is set, so a default exists");
            assert!(
                fallback.ends_with(DEFAULT_MODEL_CACHE_SUBDIR),
                "{fallback:?}"
            );
        }
    }
}
