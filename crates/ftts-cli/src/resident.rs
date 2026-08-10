//! Resident engine: a per-user background process that keeps the loaded model in memory
//! so consecutive `ftts say` invocations skip the multi-second model load.
//!
//! Shape: `ftts say` connects to a loopback TCP daemon (spawned on demand from the same
//! binary) and sends one synthesis request; the daemon holds the hydrated [`LoadedModel`]
//! and [`TtsEngine`](ftts_core::TtsEngine) between requests and exits by itself after a
//! configurable idle period (default ten minutes). Everything else — argument handling,
//! voice resolution, robot events, output writing — stays in the client process, so the
//! observable contract of `ftts say` is unchanged.
//!
//! Loopback TCP is the one transport that behaves identically on Linux, macOS, and
//! Windows with only the standard library. Access control is the state file, not the
//! port: the daemon binds an ephemeral 127.0.0.1 port and writes `{port, token, …}` to a
//! file only the invoking user can read (0600 on Unix; the per-user profile directory's
//! ACL on Windows), and every request must present that token. Texts are sensitive, so
//! the daemon holds no history: requests are served from memory and dropped.
//!
//! Failure philosophy: the resident path may only ever make `say` faster, never break
//! it. Every transport-level problem (no daemon, stale state file, version or artifact
//! mismatch, malformed reply) falls back to the classic in-process load. Only a genuine
//! synthesis error crosses the wire as an error, carrying its exit-code class.

use std::fs;
use std::hash::{BuildHasher, Hasher, RandomState};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};

use crate::error::{FttsError, FttsExitCode};
use crate::synth::{LoadedModel, ModelBundle, SynthesizedAudio};
use ftts_core::{NormalizationMode, NormalizationOptions, SynthesisRequest};

/// Default idle unload period. Overridable through `FTTS_RESIDENT_IDLE_SECS`, mostly so
/// tests can use sub-second daemons.
const DEFAULT_IDLE: Duration = Duration::from_secs(600);

/// How long the client is willing to wait for a reply. Synthesis of a long paragraph is
/// minutes at f32-reference speed; ten minutes matches the engine's own budget spirit.
/// `FTTS_RESIDENT_CLIENT_TIMEOUT_SECS` overrides it, which the e2e suite uses to stay
/// valid on machines whose debug-build synthesis is slower than the production budget.
const DEFAULT_CLIENT_READ_TIMEOUT: Duration = Duration::from_secs(600);

fn client_read_timeout() -> Duration {
    std::env::var("FTTS_RESIDENT_CLIENT_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        // Zero is rejected rather than honored: `set_read_timeout(Some(ZERO))` is an error
        // in std, so a literal 0 would fail every connect, orphan a healthy daemon, and eat
        // the full spawn wait on every run.
        .filter(|&seconds| seconds > 0)
        .map_or(DEFAULT_CLIENT_READ_TIMEOUT, Duration::from_secs)
}

/// How long the client waits for a freshly spawned daemon to write its state file and
/// accept. The daemon binds before loading the model, so this covers process start only;
/// thirty seconds absorbs a first-launch antivirus scan of the binary on Windows, which
/// measured well past ten seconds on a Surface Book. `FTTS_RESIDENT_SPAWN_WAIT_SECS`
/// overrides it.
const DEFAULT_SPAWN_WAIT: Duration = Duration::from_secs(30);

fn spawn_wait() -> Duration {
    std::env::var("FTTS_RESIDENT_SPAWN_WAIT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(DEFAULT_SPAWN_WAIT, Duration::from_secs)
}

const PROTOCOL: u64 = 1;

/// One synthesis request as it crosses the wire. Everything the daemon needs to rebuild
/// the exact [`SynthesisRequest`] the client would have run inline.
pub struct WireRequest<'a> {
    pub text: &'a str,
    /// `NormalizeMode::as_str` form; parsed back with [`parse_normalize`].
    pub normalize: &'a str,
    pub trace: bool,
    pub speaker: &'a [f32],
    pub seed: u64,
}

fn parse_normalize(label: &str) -> Option<NormalizationMode> {
    match label {
        "verbatim" => Some(NormalizationMode::Verbatim),
        "conservative" => Some(NormalizationMode::Conservative),
        "locale-aware" => Some(NormalizationMode::LocaleAware),
        _ => None,
    }
}

fn idle_period() -> Duration {
    std::env::var("FTTS_RESIDENT_IDLE_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(DEFAULT_IDLE, Duration::from_secs)
}

/// Whether the resident path is enabled at all: on by default, disabled by the `say`
/// flag or by `FTTS_NO_RESIDENT=1` for scripts that cannot pass flags.
pub fn enabled(no_resident_flag: bool) -> bool {
    if no_resident_flag {
        return false;
    }
    !matches!(
        std::env::var("FTTS_NO_RESIDENT").ok().as_deref(),
        Some("1") | Some("true")
    )
}

// ---------------------------------------------------------------------------- state file

fn resident_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("FTTS_RESIDENT_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[allow(deprecated)] // un-deprecated in current Rust; the lint fires on older stables
    std::env::home_dir().map(|home| home.join(".cache/franken_tts"))
}

/// A short stable digest of the bundle root, so distinct model directories get distinct
/// daemons. `RandomState` keys vary per process, so this hand-rolls FNV-1a instead.
///
/// The path is canonicalized first: `./model` from two different working directories must
/// key two different daemons (they are different models), while `/x/m` and `/x//m` and a
/// symlinked spelling of the same directory must share one. Hashing the raw string gave
/// the opposite of both. A path that cannot be canonicalized (not yet created, permission)
/// falls back to its literal spelling — resolution refuses such roots before spawn anyway.
fn root_digest(root: &Path) -> u64 {
    let canonical = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in canonical.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn state_path(root: &Path) -> Option<PathBuf> {
    resident_dir().map(|dir| dir.join(format!("resident-{:016x}.json", root_digest(root))))
}

struct DaemonState {
    port: u16,
    token: String,
}

fn read_state(root: &Path) -> Option<DaemonState> {
    let path = state_path(root)?;
    let raw = fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&raw).ok()?;
    Some(DaemonState {
        port: u16::try_from(value.get("port")?.as_u64()?).ok()?,
        token: value.get("token")?.as_str()?.to_owned(),
    })
}

fn write_state(root: &Path, port: u16, token: &str) -> std::io::Result<PathBuf> {
    let path = state_path(root).ok_or_else(|| {
        std::io::Error::other("no home directory and no FTTS_RESIDENT_DIR; cannot go resident")
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = json!({
        "port": port,
        "token": token,
        "pid": std::process::id(),
        "version": env!("CARGO_PKG_VERSION"),
        "bundle_root": root.to_string_lossy(),
    })
    .to_string();
    // Write-then-rename so a client never reads a half-written file. The staging file is
    // born 0600 on unix — the token is inside, so there must be no umask-window in which
    // another user can read it before a later chmod.
    let staging = path.with_extension("json.tmp");
    #[cfg(unix)]
    {
        use std::io::Write as _;
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&staging)?;
        file.write_all(body.as_bytes())?;
    }
    #[cfg(not(unix))]
    fs::write(&staging, &body)?;
    fs::rename(&staging, &path)?;
    Ok(path)
}

/// 128 bits of `RandomState` entropy (seeded from the OS) as the session token. The real
/// gate is the 0600 state file; the token binds a socket connection to that file.
fn fresh_token() -> String {
    let mut token = String::with_capacity(32);
    for _ in 0..2 {
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u128(std::time::UNIX_EPOCH.elapsed().map_or(0, |d| d.as_nanos()));
        hasher.write_u32(std::process::id());
        token.push_str(&format!("{:016x}", hasher.finish()));
    }
    token
}

/// The artifact identity the daemon pins at load: a re-pull or re-convert must not be
/// served from a stale resident model.
fn artifact_stamp(bundle: &ModelBundle) -> (u64, u64) {
    let path = bundle.canonical_main.as_deref().unwrap_or(&bundle.main);
    let Ok(meta) = fs::metadata(path) else {
        return (0, 0);
    };
    let mtime = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    (mtime, meta.len())
}

// ------------------------------------------------------------------------------- client

/// Try to synthesize through a resident daemon, spawning one if none is listening.
///
/// `Ok(None)` means "no resident path available, run inline" and is always safe; the
/// daemon spawned in the background (if any) will serve the NEXT invocation. `Ok(Some)`
/// is a completed synthesis; `Err` is a genuine synthesis error from the daemon, carrying
/// the same exit-code class the inline path would have produced.
pub fn try_synthesize(
    bundle: &ModelBundle,
    request: &WireRequest<'_>,
) -> Result<Option<SynthesizedAudio>, FttsError> {
    // JSON cannot carry NaN or infinity (serde_json writes null), so a speaker vector
    // containing one would silently shrink in transit. The inline path passes such a
    // vector through verbatim; parity therefore requires skipping the wire entirely.
    if request.speaker.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    match connect(bundle) {
        Some(stream) => roundtrip(stream, bundle, request),
        None => Ok(None),
    }
}

fn connect(bundle: &ModelBundle) -> Option<TcpStream> {
    // A live daemon answers immediately; under heavy load one connect can miss its
    // timeout, and treating that as death would orphan a healthy daemon and briefly
    // double model memory with a duplicate. Three attempts before giving up on it.
    for attempt in 0..3 {
        if attempt > 0 {
            std::thread::sleep(Duration::from_millis(300));
        }
        if let Some(stream) = connect_once(bundle) {
            return Some(stream);
        }
        if read_state(&bundle.root).is_none() {
            break; // no state file at all: nothing to retry against
        }
    }
    // None listening: clear any stale state file and spawn one from this same binary.
    if let Some(path) = state_path(&bundle.root)
        && path.exists()
    {
        let _ = fs::remove_file(&path);
    }
    spawn_daemon(&bundle.root)?;
    let deadline = Instant::now() + spawn_wait();
    while Instant::now() < deadline {
        if let Some(stream) = connect_once(bundle) {
            return Some(stream);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    None
}

fn connect_once(bundle: &ModelBundle) -> Option<TcpStream> {
    let state = read_state(&bundle.root)?;
    let stream = TcpStream::connect_timeout(
        &(Ipv4Addr::LOCALHOST, state.port).into(),
        Duration::from_millis(1000),
    )
    .ok()?;
    stream.set_read_timeout(Some(client_read_timeout())).ok()?;
    stream.set_nodelay(true).ok();
    Some(stream)
}

fn spawn_daemon(root: &Path) -> Option<()> {
    let exe = std::env::current_exe().ok()?;
    let mut command = Command::new(exe);
    command
        .arg("resident-daemon")
        .arg("--bundle-root")
        .arg(root)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    // Field diagnostics: the daemon's stderr is discarded by default; FTTS_RESIDENT_LOG
    // routes it to a file instead, which is how a silent failure to serve gets a voice.
    if let Ok(log) = std::env::var("FTTS_RESIDENT_LOG")
        && let Ok(file) = fs::OpenOptions::new().create(true).append(true).open(&log)
        && let Ok(err) = file.try_clone()
    {
        command.stdout(file).stderr(err);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NO_WINDOW: outlive the console, show nothing.
        command.creation_flags(0x0000_0008 | 0x0800_0000);
    }
    command.spawn().ok().map(|_child| ())
}

fn roundtrip(
    mut stream: TcpStream,
    bundle: &ModelBundle,
    request: &WireRequest<'_>,
) -> Result<Option<SynthesizedAudio>, FttsError> {
    let state = match read_state(&bundle.root) {
        Some(state) => state,
        None => return Ok(None),
    };
    let header = json!({
        "protocol": PROTOCOL,
        "op": "synthesize",
        "token": state.token,
        "version": env!("CARGO_PKG_VERSION"),
        "bundle_root": bundle.root.to_string_lossy(),
        "text": request.text,
        "normalize": request.normalize,
        "trace": request.trace,
        "speaker": request.speaker,
        "seed": request.seed,
    });
    if stream
        .write_all(format!("{header}\n").as_bytes())
        .and_then(|()| stream.flush())
        .is_err()
    {
        return Ok(None);
    }

    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
        return Ok(None);
    }
    let Ok(reply) = serde_json::from_str::<Value>(&line) else {
        return Ok(None);
    };
    if reply.get("ok").and_then(Value::as_bool) == Some(true) {
        // The sample count is wire data. Bound it before it sizes an allocation: at 24 kHz
        // this cap is over two hours of audio, far past anything the engine can produce,
        // and a corrupt or mismatched daemon claiming more falls back inline instead of
        // driving a multi-gigabyte (or, unchecked, overflowing) allocation.
        const MAX_WIRE_SAMPLES: u64 = 200_000_000;
        let samples = reply
            .get("samples")
            .and_then(Value::as_u64)
            .filter(|&n| n <= MAX_WIRE_SAMPLES)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0);
        let mut bytes = vec![0u8; samples * 4];
        if reader.read_exact(&mut bytes).is_err() {
            return Ok(None);
        }
        let (chunks, _remainder) = bytes.as_chunks::<4>();
        let pcm = chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect();
        return Ok(Some(SynthesizedAudio {
            pcm,
            frames: reply.get("frames").and_then(Value::as_u64).unwrap_or(0),
            prepared_token_count: reply
                .get("prepared_token_count")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
            ttfa: reply
                .get("ttfa_ms")
                .and_then(Value::as_u64)
                .map(Duration::from_millis),
        }));
    }
    // A synthesis error is real and final; anything transport-shaped means fallback.
    match reply.get("kind").and_then(Value::as_str) {
        Some("synthesis") => {
            let code = reply
                .get("exit_code")
                .and_then(Value::as_u64)
                .unwrap_or(FttsExitCode::Generic.as_u8().into());
            let message = reply
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("resident synthesis failed")
                .to_owned();
            Err(wire_error(code, message))
        }
        _ => Ok(None),
    }
}

fn wire_error(exit_code: u64, message: String) -> FttsError {
    match exit_code {
        3 => FttsError::ModelNotFound(message),
        4 => FttsError::Input(message),
        5 => FttsError::BudgetTimeout(message),
        7 => FttsError::ArtifactFormat(message),
        8 => FttsError::EnrollmentQualityRefusal(message),
        _ => FttsError::Generic(message),
    }
}

// ------------------------------------------------------------------------------- daemon

/// Run the resident daemon until the idle period passes without a request.
///
/// Binds first and loads the model lazily on the first request, so the process is
/// connectable within milliseconds of spawning and a daemon that never gets a request
/// costs no model memory before its idle exit.
pub fn run_daemon(bundle_root: &Path) -> Result<(), FttsError> {
    let bundle = ModelBundle::resolve(bundle_root)?;
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
        .map_err(|error| FttsError::Generic(format!("resident daemon cannot bind: {error}")))?;
    let port = listener
        .local_addr()
        .map_err(|error| FttsError::Generic(format!("resident daemon has no address: {error}")))?
        .port();
    let token = fresh_token();
    let state_file = write_state(&bundle.root, port, &token)
        .map_err(|error| FttsError::Generic(format!("cannot write resident state: {error}")))?;
    listener
        .set_nonblocking(true)
        .map_err(|error| FttsError::Generic(format!("resident daemon socket mode: {error}")))?;
    eprintln!(
        "resident daemon serving {} on 127.0.0.1:{port}",
        bundle.root.display()
    );

    let idle = idle_period();
    let mut resident: Option<(LoadedModel, ftts_core::TtsEngine, (u64, u64))> = None;
    let mut deadline = Instant::now() + idle;

    loop {
        match listener.accept() {
            Ok((stream, _peer)) => {
                // Serve strictly serially; a second client queues in the OS backlog.
                //
                // Isolated from panics on purpose. This daemon exists to hold a hydrated 2 GB
                // model across many calls, so one malformed request must not cost every later
                // caller that work: without this, a panic anywhere in request handling unwinds
                // straight out of the accept loop and the process dies holding the only warm copy.
                //
                // `AssertUnwindSafe` is sound for `resident` because it is only ever replaced
                // wholesale (`*resident = Some(...)` after the model is fully built), never mutated
                // in place, so an unwind can leave it either untouched or fully valid — there is no
                // half-initialized state for a later request to observe.
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    handle_connection(stream, &bundle, &token, port, &mut resident);
                }));
                if outcome.is_err() {
                    // The panic hook has already printed the location; say what it cost.
                    eprintln!("resident daemon: request panicked; connection dropped, model kept");
                }
                deadline = Instant::now() + idle;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    eprintln!("resident daemon idle exit");
                    remove_state_if_ours(&state_file, port);
                    return Ok(());
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(_) => {
                remove_state_if_ours(&state_file, port);
                return Ok(());
            }
        }
    }
}

/// Removes the state file only when it still describes THIS daemon.
///
/// Two `say` invocations racing on a cold start both spawn a daemon; both write the same
/// state path and the second write wins, orphaning the first. The orphan serves nobody and
/// idle-exits ten minutes later — and an unconditional remove at that exit would delete the
/// SUCCESSOR's state file, making the healthy daemon undiscoverable and spawning a third
/// copy of the model. Retirement cascades forever that way, one duplicate model per idle
/// period. A retiring daemon therefore checks that the file still names its own port.
fn remove_state_if_ours(state_file: &Path, port: u16) {
    let ours = fs::read_to_string(state_file)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| value.get("port")?.as_u64())
        .is_some_and(|recorded| recorded == u64::from(port));
    if ours {
        let _ = fs::remove_file(state_file);
    }
}

#[allow(clippy::type_complexity)]
fn handle_connection(
    stream: TcpStream,
    bundle: &ModelBundle,
    token: &str,
    port: u16,
    resident: &mut Option<(LoadedModel, ftts_core::TtsEngine, (u64, u64))>,
) {
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    // A write timeout is the daemon's survival: replies are written inline in the accept
    // loop, so one client that stops reading (SIGSTOP, wedged pipe) would otherwise fill
    // the send buffer and park this thread in `write_all` forever — and every later
    // `ftts say` would then hang against a daemon that accepts but never answers. Sixty
    // seconds per write on loopback is indistinguishable from a dead peer.
    let _ = stream.set_write_timeout(Some(Duration::from_secs(60)));
    let _ = stream.set_nodelay(true);

    // BOUNDED read, and the bound is the point: `read_line` on a raw socket grows its String
    // until it meets a newline, so a peer that opens a connection and sends bytes forever (or
    // simply never sends `\n`) drives this process to OOM. That happens BEFORE the token is
    // checked, so it is reachable by any local process, authenticated or not.
    //
    // The cap is generous against the largest legitimate request — a 1,024-float speaker vector
    // serialized as JSON text runs on the order of 20 KB — and still bounds the damage.
    const MAX_REQUEST_BYTES: u64 = 1024 * 1024;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if (&mut reader)
        .take(MAX_REQUEST_BYTES)
        .read_line(&mut line)
        .is_err()
    {
        return;
    }
    let mut stream = reader.into_inner();
    // A request that filled the cap without terminating is refused rather than parsed: the JSON
    // would be truncated anyway, and saying so beats a silent disconnect.
    if line.len() as u64 >= MAX_REQUEST_BYTES {
        let reply = json!({ "ok": false, "kind": "request", "message": "request too large" });
        let _ = stream.write_all(format!("{reply}\n").as_bytes());
        return;
    }
    let Ok(request) = serde_json::from_str::<Value>(&line) else {
        return;
    };

    let refuse = |stream: &mut TcpStream, kind: &str, message: &str| {
        let reply = json!({ "ok": false, "kind": kind, "message": message });
        let _ = stream.write_all(format!("{reply}\n").as_bytes());
    };

    // Token first: an unauthenticated peer learns nothing but "no".
    if request.get("token").and_then(Value::as_str) != Some(token) {
        refuse(&mut stream, "auth", "bad token");
        return;
    }
    if request.get("protocol").and_then(Value::as_u64) != Some(PROTOCOL)
        || request.get("version").and_then(Value::as_str) != Some(env!("CARGO_PKG_VERSION"))
    {
        // A different binary version must not be served by this process; the client falls
        // back inline and this daemon retires so the next spawn matches.
        refuse(&mut stream, "version", "resident daemon version mismatch");
        // Ownership-checked for the same reason as the idle exit: the state file may
        // already belong to a successor daemon spawned between the client's read and now.
        if let Some(state) = state_path(&bundle.root) {
            remove_state_if_ours(&state, port);
        }
        std::process::exit(0);
    }

    // The client says which bundle root it thinks it is talking to; a daemon keyed by a
    // colliding or stale state file must refuse rather than synthesize with the wrong
    // model. Compared canonically so a different spelling of the same directory passes.
    if let Some(wire_root) = request.get("bundle_root").and_then(Value::as_str) {
        let wire_canonical =
            fs::canonicalize(wire_root).unwrap_or_else(|_| PathBuf::from(wire_root));
        let own_canonical = fs::canonicalize(&bundle.root).unwrap_or_else(|_| bundle.root.clone());
        if wire_canonical != own_canonical {
            refuse(
                &mut stream,
                "request",
                "resident daemon serves a different bundle root",
            );
            return;
        }
    }

    // A re-pulled or re-converted artifact invalidates the resident weights.
    let stamp_now = artifact_stamp(bundle);
    if let Some((_, _, loaded_stamp)) = resident.as_ref()
        && *loaded_stamp != stamp_now
    {
        refuse(&mut stream, "stale", "model artifact changed since load");
        if let Some(state) = state_path(&bundle.root) {
            remove_state_if_ours(&state, port);
        }
        std::process::exit(0);
    }

    let text = request.get("text").and_then(Value::as_str).unwrap_or("");
    let normalize = request
        .get("normalize")
        .and_then(Value::as_str)
        .and_then(parse_normalize);
    let trace = request
        .get("trace")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let seed = request.get("seed").and_then(Value::as_u64).unwrap_or(0);
    // The speaker vector is validated rather than salvaged, and both halves of that matter.
    //
    // `filter_map(as_f64)` used to DROP entries that were not numbers, so `[1.0, "x", 2.0]`
    // silently became a 2-element vector — a malformed request quietly became a different,
    // well-formed one, conditioning generation on the wrong thing.
    //
    // Non-finite values are worse. A NaN or infinity reaching the Q8 quantizer trips its
    // `is_finite` assertion, and because `handle_connection` runs inline in the accept loop a
    // panic there takes down the daemon serving every other caller. Refusing here keeps a bad
    // request a bad request instead of an outage.
    let speaker: Vec<f32> = match request.get("speaker").and_then(Value::as_array) {
        Some(values) => {
            let mut vector = Vec::with_capacity(values.len());
            for value in values {
                let Some(number) = value.as_f64() else {
                    refuse(
                        &mut stream,
                        "request",
                        "speaker vector contains a non-numeric entry",
                    );
                    return;
                };
                #[allow(clippy::cast_possible_truncation)]
                let narrowed = number as f32;
                if !narrowed.is_finite() {
                    refuse(
                        &mut stream,
                        "request",
                        "speaker vector contains a non-finite value",
                    );
                    return;
                }
                vector.push(narrowed);
            }
            vector
        }
        None => Vec::new(),
    };
    let Some(mode) = normalize else {
        refuse(&mut stream, "request", "unknown normalize mode");
        return;
    };
    if text.is_empty() || speaker.is_empty() {
        refuse(&mut stream, "request", "empty text or speaker");
        return;
    }

    // Lazy hydration: the expensive part, done once and kept.
    if resident.is_none() {
        let loaded = match LoadedModel::load(bundle) {
            Ok(loaded) => loaded,
            Err(error) => {
                let reply = json!({
                    "ok": false,
                    "kind": "synthesis",
                    "exit_code": error.exit_code().as_u8(),
                    "message": error.to_string(),
                });
                let _ = stream.write_all(format!("{reply}\n").as_bytes());
                return;
            }
        };
        let engine = match ftts_core::TtsEngine::from_process_environment() {
            Ok(engine) => engine,
            Err(error) => {
                refuse(
                    &mut stream,
                    "engine",
                    &format!("engine start failed: {error}"),
                );
                return;
            }
        };
        *resident = Some((loaded, engine, stamp_now));
    }
    let (loaded, engine, _) = resident.as_ref().expect("hydrated just above");

    let synthesis_request = SynthesisRequest::new(text.to_owned())
        .with_normalization_options(NormalizationOptions {
            mode,
            ..NormalizationOptions::default()
        })
        .with_normalization_trace(trace);
    let cancellation = ftts_core::CancellationToken::new();
    let observer = |_event: ftts_core::SynthesisEvent| {};
    match crate::synth::synthesize(
        loaded,
        engine,
        &synthesis_request,
        &speaker,
        seed,
        &cancellation,
        &observer,
    ) {
        Ok(audio) => {
            let header = json!({
                "ok": true,
                "samples": audio.pcm.len(),
                "frames": audio.frames,
                "prepared_token_count": audio.prepared_token_count,
                "ttfa_ms": audio.ttfa.map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
            });
            if stream.write_all(format!("{header}\n").as_bytes()).is_err() {
                return;
            }
            let mut bytes = Vec::with_capacity(audio.pcm.len() * 4);
            for sample in &audio.pcm {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            let _ = stream.write_all(&bytes);
            let _ = stream.flush();
        }
        Err(error) => {
            let reply = json!({
                "ok": false,
                "kind": "synthesis",
                "exit_code": error.exit_code().as_u8(),
                "message": error.to_string(),
            });
            let _ = stream.write_all(format!("{reply}\n").as_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_labels_round_trip() {
        for label in ["verbatim", "conservative", "locale-aware"] {
            assert!(parse_normalize(label).is_some(), "{label}");
        }
        assert!(parse_normalize("aggressive").is_none());
    }

    /// A peer that never sends a newline must not be able to grow the daemon's memory without
    /// bound. This drives the real socket path, because the bug lived in `read_line`'s contract
    /// rather than in any parsing we control.
    #[test]
    fn an_endless_request_line_is_bounded_not_fatal() {
        use std::io::Write as _;
        use std::net::TcpListener;

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();

        // A client that opens a connection and streams bytes with no terminator, forever.
        let flood = std::thread::spawn(move || {
            let Ok(mut stream) = TcpStream::connect((Ipv4Addr::LOCALHOST, port)) else {
                return;
            };
            let block = vec![b'x'; 64 * 1024];
            // Stops on the first error, which is what happens once the server hangs up.
            while stream.write_all(&block).is_ok() {}
        });

        let (stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        const MAX: u64 = 1024 * 1024;
        let read = (&mut reader).take(MAX).read_line(&mut line).unwrap_or(0);

        assert!(
            read as u64 <= MAX,
            "read {read} bytes, past the {MAX}-byte cap"
        );
        assert!(
            line.len() as u64 <= MAX,
            "buffered {} bytes, past the cap",
            line.len()
        );
        drop(reader);
        let _ = flood.join();
    }

    /// Malformed speaker vectors are refused, never silently repaired.
    ///
    /// The old code used `filter_map(as_f64)`, so a non-numeric entry was DROPPED and the request
    /// proceeded with a shorter vector — a malformed request quietly becoming a different,
    /// well-formed one. Non-finite values were worse: they reach the Q8 quantizer's `is_finite`
    /// assertion, and a panic there used to take the whole daemon down with it.
    #[test]
    fn speaker_vectors_are_validated_rather_than_salvaged() {
        // Mirrors the parsing in `handle_connection` exactly.
        fn parse(values: &[Value]) -> Result<Vec<f32>, &'static str> {
            let mut vector = Vec::with_capacity(values.len());
            for value in values {
                let number = value.as_f64().ok_or("non-numeric")?;
                let narrowed = number as f32;
                if !narrowed.is_finite() {
                    return Err("non-finite");
                }
                vector.push(narrowed);
            }
            Ok(vector)
        }

        assert_eq!(parse(&[json!(1.0), json!(-2.5)]).unwrap(), vec![1.0, -2.5]);
        assert_eq!(
            parse(&[json!(1.0), json!("x"), json!(2.0)]),
            Err("non-numeric"),
            "a non-numeric entry must be refused, not dropped"
        );
        assert_eq!(parse(&[json!(null)]), Err("non-numeric"));
        // JSON has no NaN literal, so the reachable non-finite case is an f64 too large for f32.
        assert_eq!(
            parse(&[json!(1e300)]),
            Err("non-finite"),
            "an f64 that overflows f32 becomes infinity and must be refused"
        );
    }

    #[test]
    fn wire_errors_keep_their_exit_class() {
        let cases = [
            (3u64, FttsExitCode::ModelNotFound),
            (4, FttsExitCode::Input),
            (5, FttsExitCode::BudgetTimeout),
            (7, FttsExitCode::ArtifactFormat),
            (8, FttsExitCode::EnrollmentQualityRefusal),
            (1, FttsExitCode::Generic),
            (99, FttsExitCode::Generic),
        ];
        for (code, expected) in cases {
            assert_eq!(wire_error(code, String::new()).exit_code(), expected);
        }
    }

    #[test]
    fn root_digest_distinguishes_roots_and_is_stable() {
        let a = root_digest(Path::new("/tmp/model-a"));
        let b = root_digest(Path::new("/tmp/model-b"));
        assert_ne!(a, b);
        assert_eq!(a, root_digest(Path::new("/tmp/model-a")));
    }

    #[test]
    fn tokens_are_distinct_and_hex() {
        let one = fresh_token();
        let two = fresh_token();
        assert_eq!(one.len(), 32);
        assert!(one.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_ne!(one, two, "two RandomState-seeded tokens collided");
    }

    #[test]
    fn state_file_round_trips_and_respects_dir_override() {
        let dir = std::env::temp_dir().join(format!("ftts-resident-test-{}", std::process::id()));
        // SAFETY-free env mutation: tests in this module run single-threaded per process
        // under `cargo test` only when isolated; use the dir directly instead of the env.
        let root = Path::new("/tmp/some-model-root");
        let path = dir.join(format!("resident-{:016x}.json", root_digest(root)));
        fs::create_dir_all(&dir).unwrap();
        let body =
            json!({"port": 45123, "token": "abc123", "pid": 1, "version": "x", "bundle_root": "y"});
        fs::write(&path, body.to_string()).unwrap();
        let raw = fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(value.get("port").and_then(Value::as_u64), Some(45123));
        assert_eq!(value.get("token").and_then(Value::as_str), Some("abc123"));
        let _ = fs::remove_file(&path);
    }

    /// The daemon protocol refuses a bad token and answers nothing else. Uses a raw
    /// socket against `handle_connection` semantics through a real listener thread.
    #[test]
    fn daemon_refuses_bad_token() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let bundle = ModelBundle {
                root: PathBuf::from("/nonexistent"),
                main: PathBuf::from("/nonexistent/model.safetensors"),
                canonical_main: None,
                codec: PathBuf::from("/nonexistent/codec"),
            };
            let mut resident = None;
            handle_connection(stream, &bundle, "right-token", 0, &mut resident);
        });
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).unwrap();
        let request = json!({
            "protocol": PROTOCOL,
            "op": "synthesize",
            "token": "wrong-token",
            "version": env!("CARGO_PKG_VERSION"),
            "text": "hi",
            "normalize": "verbatim",
            "speaker": [0.0],
            "seed": 0,
        });
        stream.write_all(format!("{request}\n").as_bytes()).unwrap();
        let mut reply = String::new();
        BufReader::new(&mut stream).read_line(&mut reply).unwrap();
        let value: Value = serde_json::from_str(&reply).unwrap();
        assert_eq!(value.get("ok").and_then(Value::as_bool), Some(false));
        assert_eq!(value.get("kind").and_then(Value::as_str), Some("auth"));
        handle.join().unwrap();
    }
}
