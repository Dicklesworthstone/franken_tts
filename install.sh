#!/usr/bin/env bash
#
# franken_tts installer — cross-platform binary installer
#
# Pure-Rust neural text-to-speech engine (FrankenTTS). Installs the `ftts`
# and `franken_tts` binaries from GitHub release archives.
#
# One-liner install (with cache buster):
#   curl -fsSL "https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.sh?$(date +%s)" | bash
#
# Or without cache buster:
#   curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.sh | bash
#
# Options:
#   --version vX.Y.Z   Install specific version (default: latest)
#   --dest DIR         Install to DIR (default: ~/.local/bin)
#   --system           Install to /usr/local/bin (requires sudo)
#   --easy-mode        Auto-update PATH in shell rc files
#   --verify           Run self-test after install
#   --force            Reinstall even if the same version is already present
#   --artifact-url URL Use a custom release artifact URL
#   --checksum SHA     Provide expected SHA256 checksum
#   --offline TARBALL  Install from a local archive (airgap); verifies a
#                      sibling .sha256 if present
#   --from-source      Build from crates.io instead (cargo +nightly install
#                      ftts-cli — nightly Rust required)
#   --no-verify        Skip checksum verification (testing only)
#   --quiet            Suppress non-error output
#   --no-gum           Disable gum formatting even if available
#   --uninstall        Remove franken_tts and clean up
#   --help             Show this help
#
# Environment:
#   HTTP_PROXY / HTTPS_PROXY   Honored on every download
#   FTTS_INSTALL_DIR           Override default install directory
#   VERSION                    Override version to install
#
# Platforms (prebuilt binaries — 5 release targets):
#   Linux x86_64             franken_tts-X.Y.Z-linux_amd64.tar.gz
#   Linux aarch64            franken_tts-X.Y.Z-linux_arm64.tar.gz
#   macOS x86_64 (Intel)     franken_tts-X.Y.Z-darwin_amd64.tar.gz
#   macOS aarch64 (M-series) franken_tts-X.Y.Z-darwin_arm64.tar.gz
#   Windows x64              franken_tts-X.Y.Z-windows_amd64.zip
#
# A combined SHA256SUMS (or SHA256SUMS.txt) manifest plus per-asset
# <archive>.sha256 sidecars ship alongside every release and are used to verify
# each download. Some releases publish SHA256SUMS.txt instead of SHA256SUMS.
#
# Windows users: this bash installer covers linux + darwin (and WSL). On
# native Windows, download the windows_amd64.zip from the releases page and
# unzip ftts.exe / franken_tts.exe onto your PATH manually.
#
# Model weights are NOT bundled — see the README's "Getting the model"
# section after installing.
#
set -euo pipefail
umask 022
shopt -s lastpipe 2>/dev/null || true

# ============================================================================
# Configuration
# ============================================================================
VERSION="${VERSION:-}"
OWNER="${OWNER:-Dicklesworthstone}"
REPO="${REPO:-franken_tts}"
BINARY_NAME="franken_tts"
ALIAS_NAME="ftts"
CHECKSUMS_ASSET="SHA256SUMS"
# Fallback basename used when release tooling publishes SHA256SUMS.txt
# (observed on v0.1.5) instead of the canonical SHA256SUMS name.
CHECKSUMS_ASSET_FALLBACK="SHA256SUMS.txt"
DEST_DEFAULT="$HOME/.local/bin"
DEST="${DEST:-$DEST_DEFAULT}"
DEST_EXPLICIT=0
EASY=0
QUIET=0
VERIFY=0
FROM_SOURCE=0
UNINSTALL=0
FORCE_INSTALL=0
NO_CHECKSUM=0
CHECKSUM="${CHECKSUM:-}"
ARTIFACT_URL="${ARTIFACT_URL:-}"
OFFLINE_TARBALL=""
LOCK_FILE="/tmp/franken-tts-install.lock"
INSTALLER_TEMP_ROOT="${TMPDIR:-/tmp}"
while [ "$INSTALLER_TEMP_ROOT" != "/" ] && [[ "$INSTALLER_TEMP_ROOT" == */ ]]; do
    INSTALLER_TEMP_ROOT="${INSTALLER_TEMP_ROOT%/}"
done
if [ "$INSTALLER_TEMP_ROOT" = "/" ]; then
    INSTALLER_TEMP_TEMPLATE="/franken-tts-install.XXXXXX"
else
    INSTALLER_TEMP_TEMPLATE="$INSTALLER_TEMP_ROOT/franken-tts-install.XXXXXX"
fi
SYSTEM=0
NO_GUM=0
MAX_RETRIES=3
DOWNLOAD_TIMEOUT=180
# shellcheck disable=SC2034  # informational metadata, not read by the script
INSTALLER_VERSION="1.0.0"

# The running binary is authoritative for its version. The marker remains a
# fallback for older/manual installations that cannot self-report.
VERSION_MARKER_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/franken_tts"
VERSION_MARKER="$VERSION_MARKER_DIR/.installed-version"

# Proxy args applied to EVERY curl invocation. Empty array expands to nothing.
PROXY_ARGS=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

GUM_AVAILABLE=false

# ============================================================================
# Gum detection (no auto-install — keep installer lean)
# ============================================================================
check_gum() {
    [[ "$NO_GUM" -eq 1 ]] && return 1
    if command -v gum &>/dev/null && [ -t 1 ]; then
        GUM_AVAILABLE=true
        return 0
    fi
    return 1
}

# ============================================================================
# Proxy
# ============================================================================
setup_proxy() {
    PROXY_ARGS=()
    if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
        PROXY_ARGS=(--proxy "${HTTPS_PROXY:-$https_proxy}")
    elif [[ -n "${HTTP_PROXY:-${http_proxy:-}}" ]]; then
        PROXY_ARGS=(--proxy "${HTTP_PROXY:-$http_proxy}")
    fi
}

# ============================================================================
# Styled output
# ============================================================================
print_banner() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style \
            --border double \
            --border-foreground 135 \
            --padding "0 2" \
            --margin "1 0" \
            --bold \
            "$(gum style --foreground 135 'franken_tts installer')" \
            "$(gum style --foreground 245 'Pure-Rust neural text-to-speech')"
    else
        echo ""
        echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  ${BOLD}${GREEN}franken_tts installer${NC}                               ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  ${DIM}Pure-Rust neural text-to-speech${NC}                     ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
        echo ""
    fi
}

# Draw a box around text with automatic width calculation.
# Usage: draw_box "ansi_color" "line1" "line2" ...
draw_box() {
    local color="$1"; shift
    local lines=("$@")
    local max_width=0 esc stripped len
    esc=$(printf '\033')
    local strip_ansi_sed="s/${esc}\\[[0-9;]*m//g"

    for line in "${lines[@]}"; do
        stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
        len=${#stripped}
        [ "$len" -gt "$max_width" ] && max_width=$len
    done

    local inner_width=$((max_width + 4))
    local border=""
    for ((i=0; i<inner_width; i++)); do border+="═"; done

    printf "\033[%sm╔%s╗\033[0m\n" "$color" "$border"
    for line in "${lines[@]}"; do
        stripped=$(printf '%b' "$line" | LC_ALL=C sed "$strip_ansi_sed")
        len=${#stripped}
        local padding=$((max_width - len)) pad_str=""
        for ((i=0; i<padding; i++)); do pad_str+=" "; done
        printf "\033[%sm║\033[0m  %b%s  \033[%sm║\033[0m\n" "$color" "$line" "$pad_str" "$color"
    done
    printf "\033[%sm╚%s╝\033[0m\n" "$color" "$border"
}

log_info() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 39 "→ $1" >&2
    else
        echo -e "${BLUE}→${NC} $1" >&2
    fi
}

log_warn() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 214 "⚠ $1" >&2
    else
        echo -e "${YELLOW}⚠${NC} $1" >&2
    fi
}

log_error() {
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 196 "✗ $1" >&2
    else
        echo -e "${RED}✗${NC} $1" >&2
    fi
}

log_step() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 135 "→ $1" >&2
    else
        echo -e "${BLUE}→${NC} $1" >&2
    fi
}

log_success() {
    [ "$QUIET" -eq 1 ] && return 0
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style --foreground 82 "✓ $1" >&2
    else
        echo -e "${GREEN}✓${NC} $1" >&2
    fi
}

log_debug() {
    [[ "${DEBUG:-0}" -eq 1 ]] || return 0
    echo -e "${CYAN}[ftts:debug]${NC} $1" >&2
}

# Run a command behind a gum spinner (or a plain step line as fallback).
run_with_spinner() {
    local title="$1"; shift
    if [[ "$GUM_AVAILABLE" == "true" ]] && [ "$QUIET" -eq 0 ]; then
        gum spin --spinner dot --title "$title" -- "$@"
    else
        log_step "$title"
        "$@"
    fi
}

die() {
    log_error "$@"
    exit 1
}

# ============================================================================
# Usage
# ============================================================================
usage() {
    cat <<'EOF'
franken_tts installer — install the pure-Rust neural TTS CLI

Usage:
  curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.sh | bash
  curl -fsSL .../install.sh | bash -s -- [OPTIONS]

Options:
  --version vX.Y.Z   Install specific version (default: latest)
  --dest DIR         Install to DIR (default: ~/.local/bin)
  --system           Install to /usr/local/bin (requires sudo)
  --easy-mode        Auto-update PATH in shell rc files
  --verify           Run self-test after install
  --force            Reinstall even if the same version is already present
  --artifact-url URL Use a custom release artifact URL
  --checksum SHA     Provide expected SHA256 checksum
  --offline TARBALL  Install from a local archive (airgap); verifies a
                     sibling <TARBALL>.sha256 if present
  --from-source      Build from crates.io instead (cargo +nightly install
                     ftts-cli — nightly Rust required)
  --no-verify        Skip checksum verification (testing only)
  --quiet            Suppress non-error output
  --no-gum           Disable gum formatting even if available
  --uninstall        Remove franken_tts and clean up
  --help             Show this help

Environment Variables:
  HTTP_PROXY / HTTPS_PROXY   Honored on every download
  FTTS_INSTALL_DIR           Override default install directory
  VERSION                    Override version to install

Platforms (prebuilt binaries):
  Linux x86_64             franken_tts-X.Y.Z-linux_amd64.tar.gz
  Linux aarch64            franken_tts-X.Y.Z-linux_arm64.tar.gz
  macOS x86_64 (Intel)     franken_tts-X.Y.Z-darwin_amd64.tar.gz
  macOS aarch64 (M-series) franken_tts-X.Y.Z-darwin_arm64.tar.gz
  Windows x64              franken_tts-X.Y.Z-windows_amd64.zip

Windows note:
  This bash installer covers linux + darwin (and WSL). On native Windows,
  download the windows_amd64.zip from the releases page and unzip
  ftts.exe / franken_tts.exe onto your PATH manually.

Model weights:
  Weights are NOT bundled or downloaded by this installer. See the README's
  "Getting the model" section after installing.

Examples:
  # Default install (latest release)
  curl -fsSL .../install.sh | bash

  # System install with PATH auto-update
  curl -fsSL .../install.sh | sudo bash -s -- --system --easy-mode

  # Specific version
  curl -fsSL .../install.sh | bash -s -- --version v0.1.0

  # Airgapped install from a local archive
  bash install.sh --offline ./franken_tts-0.1.0-linux_amd64.tar.gz

  # Uninstall
  curl -fsSL .../install.sh | bash -s -- --uninstall
EOF
    exit 0
}

# ============================================================================
# Argument Parsing
# ============================================================================
require_option_value() {
    local option="$1"
    if [ "$#" -lt 2 ] || [ -z "${2:-}" ] || [[ "${2:-}" == --* ]]; then
        die "$option requires a non-empty value"
    fi
}

require_assignment_value() {
    local option="$1" value="$2"
    [ -n "$value" ] || die "$option requires a non-empty value"
}

# shellcheck disable=SC2034  # SYSTEM records --system intent for clarity; DEST is what's actually used
while [ $# -gt 0 ]; do
    case "$1" in
        --version) require_option_value "$1" "${2:-}"; VERSION="$2"; shift 2;;
        --version=*) VERSION="${1#*=}"; require_assignment_value "--version" "$VERSION"; shift;;
        --dest) require_option_value "$1" "${2:-}"; DEST="$2"; DEST_EXPLICIT=1; shift 2;;
        --dest=*) DEST="${1#*=}"; require_assignment_value "--dest" "$DEST"; DEST_EXPLICIT=1; shift;;
        --system)
            SYSTEM=1; DEST="/usr/local/bin"
            # Keep the version marker beside the system binary (not under the
            # invoking user's HOME) so a later `sudo ... --uninstall` finds and
            # removes it regardless of which user's HOME sudo preserves.
            VERSION_MARKER_DIR="/usr/local/share/franken_tts"
            VERSION_MARKER="$VERSION_MARKER_DIR/.installed-version"
            shift;;
        --easy-mode) EASY=1; shift;;
        --verify) VERIFY=1; shift;;
        --force) FORCE_INSTALL=1; shift;;
        --artifact-url) require_option_value "$1" "${2:-}"; ARTIFACT_URL="$2"; shift 2;;
        --artifact-url=*) ARTIFACT_URL="${1#*=}"; require_assignment_value "--artifact-url" "$ARTIFACT_URL"; shift;;
        --checksum) require_option_value "$1" "${2:-}"; CHECKSUM="$2"; shift 2;;
        --checksum=*) CHECKSUM="${1#*=}"; require_assignment_value "--checksum" "$CHECKSUM"; shift;;
        --offline) require_option_value "$1" "${2:-}"; OFFLINE_TARBALL="$2"; shift 2;;
        --offline=*) OFFLINE_TARBALL="${1#*=}"; require_assignment_value "--offline" "$OFFLINE_TARBALL"; shift;;
        --from-source) FROM_SOURCE=1; shift;;
        --no-verify) NO_CHECKSUM=1; shift;;
        --quiet|-q) QUIET=1; shift;;
        --no-gum) NO_GUM=1; shift;;
        --uninstall) UNINSTALL=1; shift;;
        -h|--help) usage;;
        --*) die "Unknown option: $1 (run with --help for supported options)";;
        *) die "Unexpected positional argument: $1 (run with --help for usage)";;
    esac
done

# Environment variable overrides
if [ -n "${FTTS_INSTALL_DIR:-}" ]; then
    [ "$SYSTEM" -eq 0 ] || die "FTTS_INSTALL_DIR cannot be combined with --system"
    DEST="$FTTS_INSTALL_DIR"
    DEST_EXPLICIT=1
fi

[ "$SYSTEM" -eq 0 ] || [ "$DEST_EXPLICIT" -eq 0 ] || die "--system cannot be combined with --dest"
[ -z "$OFFLINE_TARBALL" ] || [ "$FROM_SOURCE" -eq 0 ] || die "--offline cannot be combined with --from-source"
[ -z "$OFFLINE_TARBALL" ] || [ -z "$ARTIFACT_URL" ] || die "--offline cannot be combined with --artifact-url"
[ "$FROM_SOURCE" -eq 0 ] || [ -z "$ARTIFACT_URL" ] || die "--from-source cannot be combined with --artifact-url"
if [ -n "$VERSION" ] && [[ ! "$VERSION" =~ ^v?[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]]; then
    die "Invalid --version '$VERSION' (expected vX.Y.Z or X.Y.Z)"
fi
if [ -n "$VERSION" ] && [[ "$VERSION" != v* ]]; then
    VERSION="v$VERSION"
fi
if [ -n "$CHECKSUM" ] && [[ ! "${CHECKSUM%% *}" =~ ^[0-9A-Fa-f]{64}$ ]]; then
    die "Invalid --checksum (expected exactly 64 hexadecimal characters)"
fi
if [ -n "$ARTIFACT_URL" ] && [[ ! "$ARTIFACT_URL" =~ ^https:// ]]; then
    die "--artifact-url must use HTTPS; use --offline for local archives"
fi
[ -z "$ARTIFACT_URL" ] || [ -n "$VERSION" ] || die "--artifact-url requires --version so the staged binary can be authenticated"
if [ -n "$ARTIFACT_URL" ] && [ -z "$CHECKSUM" ] && [ "$NO_CHECKSUM" -eq 0 ]; then
    die "--artifact-url requires --checksum (or explicit testing-only --no-verify)"
fi
if [ "$UNINSTALL" -eq 1 ] && { [ -n "$VERSION" ] || [ -n "$OFFLINE_TARBALL" ] || [ "$FROM_SOURCE" -eq 1 ] || [ -n "$ARTIFACT_URL" ] || [ -n "$CHECKSUM" ] || [ "$VERIFY" -eq 1 ]; }; then
    die "--uninstall cannot be combined with install-source, version, checksum, or verification options"
fi

check_gum || true
setup_proxy

# ============================================================================
# Uninstall
# ============================================================================
do_uninstall() {
    print_banner
    log_step "Uninstalling franken_tts..."

    if [ -f "$DEST/$BINARY_NAME" ]; then
        rm -f "$DEST/$BINARY_NAME"
        log_success "Removed $DEST/$BINARY_NAME"
    else
        log_warn "Binary not found at $DEST/$BINARY_NAME"
    fi

    if [ -f "$DEST/$ALIAS_NAME" ]; then
        rm -f "$DEST/$ALIAS_NAME"
        log_success "Removed $DEST/$ALIAS_NAME"
    fi

    if [ -f "$VERSION_MARKER" ]; then
        rm -f "$VERSION_MARKER"
        rmdir "$VERSION_MARKER_DIR" 2>/dev/null || true
        log_step "Removed version marker"
    fi

    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$rc" ] && grep -q "# franken_tts installer" "$rc" 2>/dev/null; then
            if [[ "$OSTYPE" == "darwin"* ]]; then
                sed -i '' '/# franken_tts installer/d' "$rc" 2>/dev/null || true
            else
                sed -i '/# franken_tts installer/d' "$rc" 2>/dev/null || true
            fi
            log_step "Cleaned $rc"
        fi
    done

    log_success "franken_tts uninstalled (model weights, if any, were left untouched)"
    exit 0
}

[ "$UNINSTALL" -eq 1 ] && do_uninstall

# ============================================================================
# Platform Detection
# ============================================================================
# Sets PLATFORM (release asset suffix) and IS_WSL. Unsupported targets fail
# with an explicit message; source builds require an explicit --from-source.
PLATFORM=""
IS_WSL=0
detect_platform() {
    local os arch
    case "$(uname -s)" in
        Linux*)  os="linux" ;;
        Darwin*) os="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) os="windows" ;;
        *) die "Unsupported OS: $(uname -s). Prebuilt binaries cover linux, darwin, and windows (zip) only." ;;
    esac

    case "$(uname -m)" in
        x86_64|amd64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) die "Unsupported architecture: $(uname -m). No prebuilt binary exists; try --from-source on a Rust-supported target." ;;
    esac

    # WSL: behaves like linux for our purposes, but warn the user.
    if [ "$os" = "linux" ] && grep -qi microsoft /proc/version 2>/dev/null; then
        # shellcheck disable=SC2034  # IS_WSL records detection state for future use/debugging
        IS_WSL=1
        log_warn "WSL detected — installing the linux_${arch} binary (audio playback may need extra setup)"
    fi

    # All four POSIX targets ship a prebuilt binary:
    #   linux_amd64, linux_arm64, darwin_amd64, darwin_arm64
    # (windows_amd64 ships a .zip but native Windows isn't covered by this
    # bash installer — see the --help Windows note.)
    PLATFORM="${os}_${arch}"
    if [ "$os" = "windows" ]; then
        die "Native Windows is not supported by this bash installer. Download franken_tts-<ver>-windows_amd64.zip from the releases page instead."
    fi
}

# ============================================================================
# Version Resolution
# ============================================================================
resolve_version() {
    if [ -n "$VERSION" ]; then return 0; fi

    log_step "Resolving latest version..."
    local latest_url="https://api.github.com/repos/${OWNER}/${REPO}/releases/latest"
    local tag="" attempts=0

    while [ $attempts -lt $MAX_RETRIES ] && [ -z "$tag" ]; do
        attempts=$((attempts + 1))
        if command -v curl &>/dev/null; then
            tag=$(curl -fsSL "${PROXY_ARGS[@]}" \
                --connect-timeout 10 --max-time 30 \
                -H "Accept: application/vnd.github.v3+json" \
                "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")
        elif command -v wget &>/dev/null; then
            tag=$(wget -qO- --timeout=30 "$latest_url" 2>/dev/null | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")
        fi
        [ -z "$tag" ] && [ $attempts -lt $MAX_RETRIES ] && sleep 2
    done

    if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]]; then
        VERSION="$tag"
        log_success "Latest version: $VERSION"
        return 0
    fi

    # Fallback: parse the latest-release redirect.
    log_step "Trying redirect-based version resolution..."
    local redirect_url="https://github.com/${OWNER}/${REPO}/releases/latest"
    if command -v curl &>/dev/null; then
        tag=$(curl -fsSL "${PROXY_ARGS[@]}" -o /dev/null -w '%{url_effective}' "$redirect_url" 2>/dev/null | sed -E 's|.*/tag/||' || echo "")
    fi

    if [ -n "$tag" ] && [[ "$tag" =~ ^v[0-9] ]] && [[ "$tag" != *"/"* ]]; then
        VERSION="$tag"
        log_success "Latest version (via redirect): $VERSION"
        return 0
    fi

    die "Could not resolve the latest release. Retry with --version vX.Y.Z (e.g. v0.1.0), --offline ARCHIVE, or explicitly choose --from-source."
}

# ============================================================================
# Locking
# ============================================================================
LOCK_DIR="${LOCK_FILE}.d"
LOCKED=0

acquire_lock() {
    if mkdir "$LOCK_DIR" 2>/dev/null; then
        LOCKED=1
        echo $$ > "$LOCK_DIR/pid"
        return 0
    fi

    if [ -f "$LOCK_DIR/pid" ]; then
        local old_pid
        old_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || echo "")
        if [[ ! "$old_pid" =~ ^[0-9]+$ ]]; then
            log_warn "Removing invalid installer lock metadata"
            rm -f "$LOCK_DIR/pid"
            rmdir "$LOCK_DIR" 2>/dev/null || true
            if mkdir "$LOCK_DIR" 2>/dev/null; then
                LOCKED=1; echo $$ > "$LOCK_DIR/pid"; return 0
            fi
        elif ! kill -0 "$old_pid" 2>/dev/null; then
            log_warn "Removing stale lock metadata (PID $old_pid not running)"
            rm -f "$LOCK_DIR/pid"
            rmdir "$LOCK_DIR" 2>/dev/null || true
            if mkdir "$LOCK_DIR" 2>/dev/null; then
                LOCKED=1; echo $$ > "$LOCK_DIR/pid"; return 0
            fi
        fi
    fi

    if [ "$LOCKED" -eq 0 ]; then
        die "Another installation is running. Inspect PID metadata at $LOCK_DIR/pid before retrying."
    fi
}

# ============================================================================
# Cleanup
# ============================================================================
TMP=""
cleanup() {
    if [ -n "$TMP" ]; then
        local temp_parent temp_name
        temp_parent=$(dirname "$TMP")
        temp_name=$(basename "$TMP")
        if [ "$temp_parent" = "$INSTALLER_TEMP_ROOT" ] && [[ "$temp_name" == franken-tts-install.* ]]; then
            # `TMP` is a freshly-created installer-owned directory. Delete its
            # children bottom-up, then remove the now-empty root. Never apply a
            # recursive removal command to an unvalidated path.
            find "$TMP" -depth -mindepth 1 -delete 2>/dev/null || \
                log_warn "Temporary installer files remain at $TMP"
            rmdir "$TMP" 2>/dev/null || true
        else
            log_warn "Refusing to clean unexpected temporary path: $TMP"
        fi
    fi
    if [ "$LOCKED" -eq 1 ]; then
        rm -f "$LOCK_DIR/pid" 2>/dev/null || true
        rmdir "$LOCK_DIR" 2>/dev/null || \
            log_warn "Installer lock directory was not empty and was retained: $LOCK_DIR"
    fi
}
trap cleanup EXIT

# ============================================================================
# Preflight checks
# ============================================================================
check_disk_space() {
    # ~100MB headroom for the archive + two extracted binaries. Source builds
    # need far more (crate cache + release target dir) — build_from_source
    # emits its own explicit disk warning before compiling.
    local min_kb=102400
    # Walk up to the nearest existing ancestor so df has a real path to stat.
    local path="$DEST"
    while [ -n "$path" ] && [ ! -d "$path" ]; do
        local parent; parent=$(dirname "$path")
        [ "$parent" = "$path" ] && break
        path="$parent"
    done
    [ -d "$path" ] || path="/"
    if command -v df >/dev/null 2>&1; then
        local avail_kb
        avail_kb=$(df -Pk "$path" 2>/dev/null | awk 'NR==2 {print $4}' || true)
        if [ -n "$avail_kb" ] && [ "$avail_kb" -lt "$min_kb" ]; then
            die "Insufficient disk space in $path (need at least 100MB)"
        fi
    fi
}

check_write_permissions() {
    if [ ! -d "$DEST" ]; then
        if ! mkdir -p "$DEST" 2>/dev/null; then
            log_error "Cannot create $DEST (insufficient permissions)"
            die "Try running with sudo or choose a writable --dest"
        fi
    fi
    if [ ! -w "$DEST" ]; then
        log_error "No write permission to $DEST"
        die "Try running with sudo or choose a writable --dest"
    fi
}

check_existing_install() {
    if [ -x "$DEST/$ALIAS_NAME" ] || [ -x "$DEST/$BINARY_NAME" ]; then
        local current
        current=$(read_installed_version)
        log_info "Existing franken_tts detected (version: ${current})"
    fi
}

check_network() {
    # Offline / source / custom-artifact paths don't need github reachable.
    [ -n "$OFFLINE_TARBALL" ] && return 0
    [ "$FROM_SOURCE" -eq 1 ] && return 0
    command -v curl >/dev/null 2>&1 || { log_warn "curl not found; skipping network check"; return 0; }
    if ! curl -fsSL "${PROXY_ARGS[@]}" --connect-timeout 3 --max-time 5 -o /dev/null "https://github.com" 2>/dev/null; then
        log_warn "Could not reach github.com; download may fail"
    fi
}

preflight_checks() {
    log_info "Running preflight checks"
    check_disk_space
    check_write_permissions
    check_existing_install
    check_network
}

# ============================================================================
# Installed-version detection
# ============================================================================
binary_reported_version() {
    local binary="$1" reported
    [ -x "$binary" ] || return 1
    reported=$("$binary" --version 2>/dev/null | head -1 || true)
    if [[ "$reported" =~ ([0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?) ]]; then
        printf 'v%s\n' "${BASH_REMATCH[1]}"
        return 0
    fi
    return 1
}

read_installed_version() {
    local reported
    if reported=$(binary_reported_version "$DEST/$ALIAS_NAME"); then
        printf '%s\n' "$reported"
        return 0
    fi
    if reported=$(binary_reported_version "$DEST/$BINARY_NAME"); then
        printf '%s\n' "$reported"
        return 0
    fi
    if [ -f "$VERSION_MARKER" ]; then
        cat "$VERSION_MARKER" 2>/dev/null || echo "unknown"
    else
        echo "unknown"
    fi
}

write_installed_version() {
    mkdir -p "$VERSION_MARKER_DIR" 2>/dev/null || true
    printf '%s\n' "${1:-unknown}" > "$VERSION_MARKER" 2>/dev/null || true
}

# Already-installed short-circuit (idempotent re-runs). The binary self-report
# wins, with the marker retained only for legacy/manual installs. Honors
# --force.
already_installed() {
    [ "$FORCE_INSTALL" -eq 1 ] && return 1
    [ -n "$OFFLINE_TARBALL" ] && return 1   # offline: caller knows what they want
    [ -z "$VERSION" ] && return 1
    [ -x "$DEST/$BINARY_NAME" ] || return 1
    [ -x "$DEST/$ALIAS_NAME" ] || return 1
    local installed
    installed=$(read_installed_version)
    [ "$installed" = "$VERSION" ]
}

# ============================================================================
# PATH modification
# ============================================================================
maybe_add_path() {
    case ":$PATH:" in
        *:"$DEST":*) return 0;;
        *)
            if [ "$EASY" -eq 1 ]; then
                local updated=0
                for rc in "$HOME/.zshrc" "$HOME/.bashrc"; do
                    if [ -f "$rc" ] && [ -w "$rc" ]; then
                        if ! grep -qF "$DEST" "$rc" 2>/dev/null; then
                            echo "" >> "$rc"
                            echo "export PATH=\"$DEST:\$PATH\"  # franken_tts installer" >> "$rc"
                        fi
                        updated=1
                    fi
                done
                if [ "$updated" -eq 1 ]; then
                    log_warn "PATH updated; restart shell or run: export PATH=\"$DEST:\$PATH\""
                else
                    log_warn "Add $DEST to PATH to use ftts"
                fi
            else
                log_warn "Add $DEST to PATH to use ftts"
            fi
        ;;
    esac
}

# ============================================================================
# Download with retry (proxy-aware)
# ============================================================================
download_file() {
    local url="$1" dest="$2" attempt=0
    local partial="${dest}.part"

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        log_debug "Download attempt $attempt for $url"
        if command -v curl &>/dev/null; then
            if curl -fsSL "${PROXY_ARGS[@]}" --connect-timeout 30 --max-time "$DOWNLOAD_TIMEOUT" \
                --retry 2 -o "$partial" "$url"; then
                mv -f "$partial" "$dest"; return 0
            fi
        elif command -v wget &>/dev/null; then
            if wget --quiet --timeout="$DOWNLOAD_TIMEOUT" -O "$partial" "$url"; then
                mv -f "$partial" "$dest"; return 0
            fi
        else
            die "Neither curl nor wget found"
        fi
        [ $attempt -lt $MAX_RETRIES ] && { log_warn "Download failed, retrying in 3s..."; sleep 3; }
    done
    rm -f "$partial" 2>/dev/null || true
    return 1
}

# ============================================================================
# Checksum verification (dual tool: sha256sum / shasum)
# ============================================================================
sha256_of() {
    if command -v sha256sum &>/dev/null; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo ""
    fi
}

# ============================================================================
# Validated binary-pair install
# ============================================================================
validate_binary_for_install() {
    local src="$1" label="$2"
    local reported expected
    if ! reported=$(binary_reported_version "$src"); then
        log_error "$label failed --version validation"
        return 1
    fi
    if [ -n "$VERSION" ]; then
        expected="$VERSION"
        [[ "$expected" == v* ]] || expected="v$expected"
        if [ "$reported" != "$expected" ]; then
            log_error "$label reports $reported but the requested release is $expected"
            return 1
        fi
    fi
}

install_binary_atomic() {
    local src="$1" dest="$2"
    local tmp_dest="${dest}.tmp.$$"
    install -m 0755 "$src" "$tmp_dest"
    if ! mv -f "$tmp_dest" "$dest"; then
        rm -f "$tmp_dest" 2>/dev/null || true
        die "Failed to move binary into place"
    fi
}

# Validate both command names before replacing either one. Install the long
# name first and the primary `ftts` command last, so an unlikely second
# rename failure never leaves the primary command on an unvalidated release.
install_binary_pair() {
    local long_src="$1" alias_src="$2"
    local long_report alias_report
    validate_binary_for_install "$long_src" "$BINARY_NAME binary" || \
        die "Release archive failed binary validation; existing installation was not replaced"
    validate_binary_for_install "$alias_src" "$ALIAS_NAME binary" || \
        die "Release archive failed $ALIAS_NAME validation; existing installation was not replaced"
    long_report=$(binary_reported_version "$long_src")
    alias_report=$(binary_reported_version "$alias_src")
    [ "$long_report" = "$alias_report" ] || \
        die "Release archive contains mismatched binary versions; existing installation was not replaced"
    install_binary_atomic "$long_src" "$DEST/$BINARY_NAME"
    install_binary_atomic "$alias_src" "$DEST/$ALIAS_NAME"
}

# Admit only flat, regular-file release archives. This rejects absolute paths,
# traversal, nested files, and symlink/hardlink entries before extraction.
# Release archives contain: franken_tts (required), plus optionally ftts,
# README.md, LICENSE, CHANGELOG.md (the release packager may ship one binary
# per archive with include files; `ftts` is reconstructed when absent).
validate_archive_members() {
    local archive="$1" archive_ext="$2" members member normalized
    local binary_count=0 alias_count=0 readme_count=0 license_count=0 changelog_count=0
    if [ "$archive_ext" = "zip" ]; then
        members=$(unzip -Z1 "$archive") || return 1
    else
        members=$(tar -tzf "$archive") || return 1
        # Ignore AppleDouble / __MACOSX paths when classifying entry types.
        if tar -tvzf "$archive" | awk '{
            name=$NF
            sub(/^\.\//, "", name)
            if (name ~ /^\._/ || name == "__MACOSX" || name ~ /^__MACOSX\//) next
            if (substr($1, 1, 1) != "-") { bad=1; exit }
        } END { exit bad ? 0 : 1 }'; then
            log_error "Archive contains a non-regular entry"
            return 1
        fi
    fi
    while IFS= read -r member; do
        [ -n "$member" ] || continue
        normalized="${member#./}"
        # macOS packagers often embed AppleDouble / resource-fork junk
        # (._* and __MACOSX/**). Ignore those so linux installs still work
        # when a release tarball was built on Darwin without COPYFILE_DISABLE.
        case "$normalized" in
            ._*|__MACOSX|__MACOSX/*) continue ;;
        esac
        case "$normalized" in
            "$BINARY_NAME") binary_count=$((binary_count + 1)) ;;
            "$ALIAS_NAME") alias_count=$((alias_count + 1)) ;;
            "README.md") readme_count=$((readme_count + 1)) ;;
            "LICENSE") license_count=$((license_count + 1)) ;;
            "CHANGELOG.md") changelog_count=$((changelog_count + 1)) ;;
            *)
                log_error "Archive contains an unexpected member: $normalized"
                return 1
                ;;
        esac
    done <<< "$members"
    [ "$binary_count" -eq 1 ] && [ "$alias_count" -le 1 ] && \
        [ "$readme_count" -le 1 ] && [ "$license_count" -le 1 ] && \
        [ "$changelog_count" -le 1 ] || {
        log_error "Archive must contain exactly one $BINARY_NAME and no duplicate allowlisted members"
        return 1
    }
}

# Extract an archive into TMP and install both command names. `ftts` and
# `franken_tts` are byte-equivalent shims by design, so an archive that ships
# only `franken_tts` still yields a valid `ftts`: we install a byte-copy.
extract_and_install() {
    local archive="$1" archive_ext="$2"

    log_step "Extracting..."
    if [[ "$archive_ext" == "zip" ]]; then
        command -v unzip &>/dev/null || die "unzip required for .zip archives"
    else
        command -v tar &>/dev/null || die "tar required for .tar.gz archives"
    fi
    validate_archive_members "$archive" "$archive_ext" || return 1
    if [[ "$archive_ext" == "zip" ]]; then
        unzip -o "$archive" -d "$TMP/extract" >/dev/null 2>&1 || return 1
    else
        mkdir -p "$TMP/extract"
        # Exclude AppleDouble / __MACOSX junk if present in the archive.
        tar --exclude='._*' --exclude='__MACOSX' --exclude='__MACOSX/*' \
            -xzf "$archive" -C "$TMP/extract" 2>/dev/null || return 1
    fi

    local bin="$TMP/extract/$BINARY_NAME"
    if [ ! -f "$bin" ] || [ -L "$bin" ]; then
        log_error "Binary not found after extraction"
        return 1
    fi

    local alias_bin="$TMP/extract/$ALIAS_NAME"
    if [ -e "$alias_bin" ] && { [ ! -f "$alias_bin" ] || [ -L "$alias_bin" ]; }; then
        log_error "$ALIAS_NAME entry is not a regular file"
        return 1
    fi
    # ftts absent: install it as a byte-copy of franken_tts (byte-equivalent
    # shims by design).
    [ -f "$alias_bin" ] || alias_bin="$bin"
    chmod +x "$bin" "$alias_bin"
    install_binary_pair "$bin" "$alias_bin"
    return 0
}

# ============================================================================
# Offline install (airgap) — install from a local archive
# ============================================================================
install_offline() {
    [ -f "$OFFLINE_TARBALL" ] || die "Offline archive not found: $OFFLINE_TARBALL"
    log_step "Installing from local archive: $OFFLINE_TARBALL"

    local archive_ext="tar.gz"
    [[ "$OFFLINE_TARBALL" == *.zip ]] && archive_ext="zip"

    # Verify against a sibling .sha256 if one exists (or an explicit --checksum).
    local expected=""
    if [ "$NO_CHECKSUM" -eq 0 ]; then
        if [ -n "$CHECKSUM" ]; then
            expected="${CHECKSUM%% *}"
        elif [ -f "${OFFLINE_TARBALL}.sha256" ]; then
            expected=$(awk '{print $1}' "${OFFLINE_TARBALL}.sha256" 2>/dev/null | head -1)
        fi
        if [ -n "$expected" ]; then
            log_step "Verifying checksum..."
            local actual; actual=$(sha256_of "$OFFLINE_TARBALL")
            expected=$(printf '%s' "$expected" | tr 'A-F' 'a-f')
            actual=$(printf '%s' "$actual" | tr 'A-F' 'a-f')
            if [ -z "$actual" ]; then
                die "No SHA256 tool found; use --no-verify only in a controlled test environment"
            elif [ "$expected" != "$actual" ]; then
                die "Checksum mismatch! expected=$expected got=$actual"
            else
                log_success "Checksum verified"
            fi
        else
            die "No checksum available for offline archive; provide --checksum, a sibling .sha256, or explicit testing-only --no-verify"
        fi
    else
        log_warn "Checksum verification disabled (--no-verify)"
    fi

    extract_and_install "$OFFLINE_TARBALL" "$archive_ext" || die "Failed to install from offline archive"
    write_installed_version "${VERSION:-offline}"
    log_success "Installed to $DEST/$ALIAS_NAME and $DEST/$BINARY_NAME (offline)"
}

# ============================================================================
# Build from source (cargo +nightly install ftts-cli, from crates.io)
# ============================================================================
# HONESTY NOTE on source builds:
#   The git workspace has a Cargo PATH dependency on ../asupersync, which does
#   not exist inside a `cargo install --git` checkout — a git-based install
#   can never work. The PACKAGED crates on crates.io instead resolve
#   asupersync 0.4.0 from the registry, so source builds go through crates.io:
#     cargo +nightly install ftts-cli --locked
#   NIGHTLY IS REQUIRED: the crate uses #![feature(float_erf)], so a plain
#   stable cargo fails to compile it.
ensure_rust_nightly() {
    if ! command -v rustup >/dev/null 2>&1; then
        if command -v cargo >/dev/null 2>&1; then
            log_error "cargo was found but rustup was not. franken_tts source builds"
            log_error "require the NIGHTLY toolchain (#![feature(float_erf)]), which"
            log_error "this installer only provisions via rustup."
            die "Install rustup (https://rustup.rs), run 'rustup toolchain install nightly', then retry --from-source"
        fi
        log_step "Installing Rust (nightly) via rustup..."
        log_warn "cargo not found: about to install Rust via the official rustup.rs"
        log_warn "bootstrap (a second remote script, auto-accepted). Ctrl-C now to"
        log_warn "abort and install Rust yourself if you prefer."
        # The grace period only makes sense when the warning was visible.
        [ "$QUIET" -eq 1 ] || sleep 3
        curl -fsSL "${PROXY_ARGS[@]}" https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly --profile minimal
        export PATH="$HOME/.cargo/bin:$PATH"
        # shellcheck disable=SC1091  # rustup-generated env file not present at lint time
        [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
        command -v rustup >/dev/null 2>&1 || return 1
    fi
    if ! rustup toolchain list 2>/dev/null | grep -q '^nightly'; then
        log_step "Installing the nightly toolchain (required: #![feature(float_erf)])..."
        rustup toolchain install nightly --profile minimal || return 1
    fi
    command -v cargo >/dev/null 2>&1
}

build_from_source() {
    log_step "Building from source (cargo +nightly install ftts-cli)..."
    log_warn "Source builds compile a release binary from crates.io: expect"
    log_warn "several GB of disk use (crate cache + target dir) and multiple"
    log_warn "minutes of compile time."
    ensure_rust_nightly || die "Rust nightly (rustup + cargo) is required for source builds and could not be installed"

    local cargo_root="$TMP/cargo-root"
    mkdir -p "$cargo_root"

    local version_args=()
    if [ -n "$VERSION" ]; then
        version_args=(--version "${VERSION#v}")
    fi

    log_step "Running cargo +nightly install (this may take several minutes)..."
    if ! cargo +nightly install ftts-cli --locked \
        "${version_args[@]}" \
        --root "$cargo_root" \
        --bins; then
        log_error "Source build failed (cargo's own error above is authoritative —"
        log_error "e.g. the crate may not be published on crates.io yet)."
        log_error "Reproduce manually with:"
        log_error "  cargo +nightly install ftts-cli --locked${VERSION:+ --version ${VERSION#v}}"
        die "Build from source failed"
    fi

    local bin="$cargo_root/bin/$BINARY_NAME"
    local alias_bin="$cargo_root/bin/$ALIAS_NAME"
    [ -x "$bin" ] || die "$BINARY_NAME binary not found after build"
    # ftts is a byte-equivalent shim; fall back to a byte-copy if a future
    # packaging change ever drops the second [[bin]].
    [ -x "$alias_bin" ] || alias_bin="$bin"

    install_binary_pair "$bin" "$alias_bin"
    write_installed_version "${VERSION:-source}"
    log_success "Installed to $DEST/$ALIAS_NAME and $DEST/$BINARY_NAME (source build)"
}

# ============================================================================
# Download release binary
# ============================================================================
download_release() {
    local platform="$1"
    local archive_ext="tar.gz"
    [[ "$platform" == windows_* ]] && archive_ext="zip"

    local archive_name url
    if [ -n "$ARTIFACT_URL" ]; then
        url="$ARTIFACT_URL"
        printf -v archive_name '%s' "${ARTIFACT_URL##*/}"
        [[ "$archive_name" == *.zip ]] && archive_ext="zip"
    else
        # Assets use the version WITHOUT the leading 'v' (e.g. 0.1.0):
        #   franken_tts-0.1.0-darwin_arm64.tar.gz
        local ver_no_v="${VERSION#v}"
        archive_name="${BINARY_NAME}-${ver_no_v}-${platform}.${archive_ext}"
        url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${archive_name}"
    fi

    log_step "Downloading $archive_name..."
    download_file "$url" "$TMP/$archive_name" || return 1
    [ -f "$TMP/$archive_name" ] || return 1

    # Checksum verification. Primary source: the release's combined SHA256SUMS
    # manifest. Fallback: the per-asset <archive>.sha256 sidecar that ships
    # next to every artifact.
    if [ "$NO_CHECKSUM" -eq 1 ]; then
        log_warn "Checksum verification disabled (--no-verify)"
    else
        local expected=""
        if [ -n "$CHECKSUM" ]; then
            expected="${CHECKSUM%% *}"
        else
            local manifest_file=""
            local checksums_url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${CHECKSUMS_ASSET}"
            if download_file "$checksums_url" "$TMP/$CHECKSUMS_ASSET"; then
                manifest_file="$TMP/$CHECKSUMS_ASSET"
            else
                # Some releases publish SHA256SUMS.txt instead of SHA256SUMS.
                local fallback_url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${CHECKSUMS_ASSET_FALLBACK}"
                log_step "Combined manifest ${CHECKSUMS_ASSET} unavailable; trying ${CHECKSUMS_ASSET_FALLBACK}..."
                if download_file "$fallback_url" "$TMP/$CHECKSUMS_ASSET_FALLBACK"; then
                    manifest_file="$TMP/$CHECKSUMS_ASSET_FALLBACK"
                fi
            fi
            if [ -n "$manifest_file" ]; then
                # Match a line whose filename field equals the exact asset name
                # (tolerating the `*name` binary-mode marker) so an unrelated
                # entry can never shadow the real checksum. `|| true` keeps a
                # no-match from killing the script under `set -euo pipefail`;
                # we fall through to the sidecar, then the honest error.
                expected=$(awk -v name="$archive_name" \
                    '$2 == name || $2 == "*" name { print $1; exit }' \
                    "$manifest_file" 2>/dev/null || true)
            fi
            if [ -z "$expected" ]; then
                local sidecar_url="https://github.com/${OWNER}/${REPO}/releases/download/${VERSION}/${archive_name}.sha256"
                log_step "Combined manifest unavailable; trying ${archive_name}.sha256 sidecar..."
                if download_file "$sidecar_url" "$TMP/${archive_name}.sha256"; then
                    expected=$(awk '{print $1; exit}' "$TMP/${archive_name}.sha256" 2>/dev/null || true)
                    if [ -n "$expected" ] && [[ ! "$expected" =~ ^[0-9A-Fa-f]{64}$ ]]; then
                        expected=""
                    fi
                fi
            fi
        fi

        if [ -n "$expected" ]; then
            log_step "Verifying checksum..."
            local actual; actual=$(sha256_of "$TMP/$archive_name")
            expected=$(printf '%s' "$expected" | tr 'A-F' 'a-f')
            actual=$(printf '%s' "$actual" | tr 'A-F' 'a-f')
            if [ -z "$actual" ]; then
                log_error "No SHA256 tool found (sha256sum/shasum)"
                return 1
            elif [ "$expected" != "$actual" ]; then
                log_error "Checksum mismatch!"
                log_error "  Expected: $expected"
                log_error "  Got:      $actual"
                rm -f "$TMP/$archive_name"
                return 1
            else
                log_success "Checksum verified: ${actual:0:16}..."
            fi
        else
            log_error "Checksum not available for $archive_name (not in $CHECKSUMS_ASSET or $CHECKSUMS_ASSET_FALLBACK and no ${archive_name}.sha256 sidecar)"
            return 1
        fi
    fi

    extract_and_install "$TMP/$archive_name" "$archive_ext" || return 1
    write_installed_version "$VERSION"
    log_success "Installed to $DEST/$ALIAS_NAME and $DEST/$BINARY_NAME"
    return 0
}

# ============================================================================
# Self-test
# ============================================================================
# Validate the installed binaries via their stable --version surface. Model
# readiness is intentionally NOT checked: weights are operator-provisioned
# and never bundled by this installer.
run_self_test() {
    log_step "Verifying installed binaries..."
    local long_version alias_version
    long_version=$(binary_reported_version "$DEST/$BINARY_NAME") || die "Installed franken_tts cannot report its version"
    alias_version=$(binary_reported_version "$DEST/$ALIAS_NAME") || die "Installed ftts cannot report its version"
    [ "$long_version" = "$alias_version" ] || die "Installed binary names report different versions"
    log_success "Installation verified: ftts $alias_version (model readiness remains operator-configured)"
}

# ============================================================================
# Shell completions
# ============================================================================
# NOTE: franken_tts v0.1.0 has no `completions` subcommand. There is nothing
# to generate, so completion installation is deliberately skipped.

# ============================================================================
# AI agent hooks / skills
# ============================================================================
# NOTE: franken_tts is a plain TTS CLI, not a guardrail/hook tool. It has no
# PreToolUse/BeforeTool semantics and ships no agent skill, so agent
# auto-configuration is deliberately omitted (unlike dcg/rch which gate or
# offload agent tool calls).

# ============================================================================
# Summary
# ============================================================================
print_summary() {
    [ "$QUIET" -eq 1 ] && return 0
    local installed_version path_status
    installed_version=$(read_installed_version)

    if [[ ":$PATH:" == *":$DEST:"* ]]; then
        path_status="on PATH"
    else
        path_status="NOT on PATH"
    fi

    echo ""
    if [[ "$GUM_AVAILABLE" == "true" ]]; then
        gum style \
            --border rounded --border-foreground 82 --padding "1 2" --margin "1 0" \
            "$(gum style --foreground 82 --bold '✓ franken_tts installed!')" \
            "" \
            "$(gum style --foreground 245 "Version:  $installed_version")" \
            "$(gum style --foreground 245 "Commands: $DEST/$ALIAS_NAME, $DEST/$BINARY_NAME")" \
            "$(gum style --foreground 245 "PATH:     $path_status")"
    else
        draw_box "0;32" \
            "${GREEN}✓ franken_tts installed!${NC}" \
            "" \
            "Version:  $installed_version" \
            "Commands: $DEST/$ALIAS_NAME, $DEST/$BINARY_NAME" \
            "PATH:     $path_status"
    fi
    echo ""

    if [[ ":$PATH:" != *":$DEST:"* ]]; then
        log_warn "Add to PATH: export PATH=\"$DEST:\$PATH\"   (or re-run with --easy-mode)"
        echo ""
    fi

    echo "  Quick start:"
    echo "    ftts --version        # verify the install"
    echo "    ftts --help           # explore commands"
    echo ""
    echo "  Model weights:"
    echo "    NOT bundled with this install. Before synthesizing anything, fetch"
    echo "    the model as described in the README's \"Getting the model\" section:"
    echo "    https://github.com/${OWNER}/${REPO}#getting-the-model"
    echo ""
    echo "  Uninstall:"
    echo "    curl -fsSL https://raw.githubusercontent.com/${OWNER}/${REPO}/main/install.sh | bash -s -- --uninstall"
    echo ""
}

# ============================================================================
# Main
# ============================================================================
main() {
    acquire_lock
    print_banner

    TMP=$(mktemp -d "$INSTALLER_TEMP_TEMPLATE")

    # Offline / airgap path: no platform/version resolution needed.
    if [ -n "$OFFLINE_TARBALL" ]; then
        log_step "Install directory: $DEST"
        check_write_permissions
        install_offline
        maybe_add_path
        [ "$VERIFY" -eq 1 ] && run_self_test
        print_summary
        return 0
    fi

    detect_platform
    log_step "Platform: $PLATFORM"
    log_step "Install directory: $DEST"

    if [ "$FROM_SOURCE" -eq 0 ]; then
        resolve_version
    fi

    preflight_checks

    # Already-installed short-circuit (binary self-report, marker fallback).
    if already_installed; then
        log_success "franken_tts $VERSION is already installed at $DEST/$ALIAS_NAME"
        log_info "Use --force to reinstall"
        maybe_add_path
        print_summary
        return 0
    fi

    if [ "$FROM_SOURCE" -eq 0 ] && [ -n "$VERSION" ]; then
        if download_release "$PLATFORM"; then
            :
        else
            die "Binary download or verification failed. Retry, use --offline ARCHIVE, or explicitly choose --from-source."
        fi
    else
        build_from_source
    fi

    maybe_add_path
    [ "$VERIFY" -eq 1 ] && run_self_test
    print_summary
}

if [[ "${BASH_SOURCE[0]:-}" == "${0:-}" ]] || [[ -z "${BASH_SOURCE[0]:-}" ]]; then
    main "$@"
fi
