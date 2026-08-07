#![forbid(unsafe_code)]

//! Shared, stateless command-line dispatch for both FrankenTTS binaries.

mod error;
pub mod robot;

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
const PINNED_MAIN_WEIGHTS_FILENAME: &str = "model.safetensors";
const PINNED_MAIN_WEIGHTS_SHA256: &str =
    "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6";
const PINNED_MODEL_REVISION: &str = "5d83992436eae1d760afd27aff78a71d676296fc";
const PINNED_MAIN_TENSOR_COUNT: usize = 478;
const PINNED_TENSOR_INVENTORY: &str =
    include_str!("../../../docs/truth-pack/TENSOR_INVENTORY.json");
const PINNED_MODEL_CONFIG: &str = include_str!("../../../docs/truth-pack/snapshots/hf/config.json");
const APACHE_LICENSE: &str = include_str!("../../../docs/truth-pack/snapshots/gh/LICENSE");
const ENVIRONMENT_VARIABLES: [&str; 8] = [
    "FTTS_MODEL_DIR",
    "FTTS_THREADS",
    "FTTS_PROFILE",
    "FTTS_PACKET_FRAMES",
    "FTTS_MATH_MODE",
    "FTTS_QUANT",
    "FTTS_FORCE_ARCH",
    "FTTS_NUMA",
];

/// Runs the shared `ftts` / `franken_tts` command-line interface.
pub fn cli_main() -> ExitCode {
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
    /// Build a consent-bearing voice pack from reference audio.
    Enroll(EnrollArgs),
    /// Inspect a portable voice pack.
    Voice(VoiceArgs),
    /// Convert pinned source weights into a portable .fttsq artifact.
    Convert(ConvertArgs),
    /// Emit versioned, line-oriented robot contract data.
    Robot(RobotArgs),
    /// Report local configuration and readiness without inference.
    Doctor(DoctorArgs),
}

#[derive(Debug, clap::Args)]
struct SayArgs {
    /// Text to synthesize. Use `-` to read UTF-8 text from stdin.
    #[arg(value_name = "TEXT")]
    text: Option<String>,

    /// Read UTF-8 text from PATH. Use `-` for stdin.
    #[arg(long, value_name = "PATH", conflicts_with = "text")]
    file: Option<PathBuf>,

    /// Explicit .fttsq model artifact. No network lookup is performed.
    #[arg(long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Optional .ftvoice voice pack.
    #[arg(long, value_name = "PATH")]
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
}

#[derive(Debug, clap::Args)]
struct EnrollArgs {
    /// Reference WAV or FLAC audio.
    #[arg(value_name = "REFERENCE_AUDIO")]
    reference_audio: PathBuf,

    /// Explicitly proceed after an enrollment-quality warning where safe.
    #[arg(long)]
    force: bool,
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
        Command::Enroll(args) => Err(FttsError::Generic(format!(
            "enrollment is not implemented in the Phase-0 skeleton (reference: {}, force: {}); use `ftts robot health` to inspect readiness",
            args.reference_audio.display(),
            args.force,
        ))),
        Command::Voice(VoiceArgs {
            command: VoiceCommand::Inspect { path },
        }) => run_voice_inspect(path, stdout),
        Command::Convert(args) => run_convert(&cli, args, environment, stdout, stderr),
        Command::Robot(args) => run_robot(args.command.clone(), environment, stdout),
        Command::Doctor(args) => run_doctor(args, environment, stdout),
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
    // so no later emission can pick the other stream and interleave NDJSON with audio bytes.
    let raw_stream = args.stream == Some(StreamMode::Raw);

    let outcome = run_say_events(cli, args, environment, stdin, &run, &mut |event| {
        if raw_stream {
            write_json_line(stderr, event)
        } else {
            write_json_line(stdout, event)
        }
    });

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
    let voice = resolve_optional_file(args.voice.as_deref(), "voice pack")?;
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
            "output": args.output.as_ref().map(|path| path.display().to_string()),
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

    Err(FttsError::Generic(
        "synthesis is not implemented in the Phase-0 skeleton; use `ftts say --check --model PATH TEXT` for stateless input and admission validation".to_owned(),
    ))
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
    if let Some(path) = explicit {
        return resolve_existing_file(path, "model artifact")
            .map(|path| path.display().to_string());
    }

    let searched = model_search_paths(environment);
    if let Some(path) = searched.iter().find(|path| path.is_file()) {
        return Ok(path.display().to_string());
    }

    let searched = searched
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Err(FttsError::ModelNotFound(format!(
        "no model artifact was found; searched: [{}]; use `ftts say --model PATH --check TEXT`",
        searched
    )))
}

fn resolve_optional_file(path: Option<&Path>, label: &str) -> Result<Option<String>, FttsError> {
    path.map(|path| resolve_existing_file(path, label).map(|path| path.display().to_string()))
        .transpose()
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
            object.insert("available".to_owned(), json!(["scalar-placeholder"]));
            object.insert("dispatched".to_owned(), Value::Null);
            object.insert("isa_features".to_owned(), json!(detected_isa_features()));
            object.insert("kernel_plan".to_owned(), Value::Null);
            object.insert("pool_sizing".to_owned(), Value::Null);
            object.insert(
                "force_arch".to_owned(),
                json!(environment.value("FTTS_FORCE_ARCH")),
            );
            Value::Object(object)
        }
        RobotCommand::Selftest => json!({
            "schema_version": ROBOT_SCHEMA_VERSION,
            "event": "selftest",
            "status": "skipped",
            "reason": "no model-specific kernel is implemented in the Phase-0 skeleton",
        }),
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
        searched.push(
            PathBuf::from(home)
                .join(".cache/franken_tts/models")
                .join(MODEL_BASENAME),
        );
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

    const CLAP_SURFACE_SNAPSHOT: &str = "commands=say,enroll,voice,convert,robot,doctor\nrobot=schema,health,backends,selftest\nsay=file,model,voice,output,stream,check\nglobal=profile,packet-frames,math-mode,voice-pack,normalize,trace,seed\n";

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
        let global = command
            .get_arguments()
            .filter_map(|argument| argument.get_long())
            .filter(|argument| *argument != "help")
            .collect::<Vec<_>>()
            .join(",");
        let actual = format!("commands={commands}\nrobot={robot}\nsay={say}\nglobal={global}\n");
        assert_eq!(actual, CLAP_SURFACE_SNAPSHOT);
    }

    #[test]
    fn argument_file_and_stdin_text_are_identical() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../README.md");
        let expected = fs::read_to_string(&root).expect("checked-in README");
        let from_argument = read_text(
            &SayArgs {
                text: Some(expected.clone()),
                file: None,
                model: None,
                voice: None,
                output: None,
                stream: None,
                check: true,
            },
            &mut Cursor::new(Vec::<u8>::new()),
        )
        .expect("argument text");
        let from_file = read_text(
            &SayArgs {
                text: None,
                file: Some(root),
                model: None,
                voice: None,
                output: None,
                stream: None,
                check: true,
            },
            &mut Cursor::new(Vec::<u8>::new()),
        )
        .expect("file text");
        let from_stdin = read_text(
            &SayArgs {
                text: Some("-".to_owned()),
                file: None,
                model: None,
                voice: None,
                output: None,
                stream: None,
                check: true,
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
            file: None,
            model: Some(model),
            voice: None,
            output: None,
            stream: None,
            check: true,
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
}
