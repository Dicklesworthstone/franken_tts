<#
.SYNOPSIS
    Installs franken_tts (ftts) on Windows.

.DESCRIPTION
    Downloads the prebuilt Windows binaries from the GitHub release, verifies their SHA-256
    against the release's own SHA256SUMS manifest, and installs `ftts.exe` and `franken_tts.exe`.

    install.sh covers Linux and macOS and says outright that native Windows is not its job —
    this is that job. The two stay deliberately alike: same artifact names, same checksum
    manifest, same install-then-verify order, so a release that is good on one is good on both.

    Model weights are NOT downloaded here. They are ~2 GB, they are Apache-2.0 material with its
    own provenance, and a shell one-liner is the wrong place to pull them silently. Run
    `ftts pull` once after installing.

.PARAMETER Version
    Release tag to install, with or without the leading "v". Defaults to the latest release.

.PARAMETER InstallDir
    Where the binaries go. Defaults to %LOCALAPPDATA%\Programs\franken_tts\bin, which needs no
    administrator rights.

.PARAMETER EasyMode
    Also add InstallDir to your user PATH, so `ftts` works in a new terminal without further setup.

.PARAMETER Force
    Reinstall even when the requested version is already present.

.PARAMETER NoVerify
    Skip SHA-256 verification. Provided for an air-gapped mirror that carries no manifest; using
    it on a network download means trusting whatever arrives.

.PARAMETER Quiet
    Print only warnings and errors.

.EXAMPLE
    & ([scriptblock]::Create((irm "https://raw.githubusercontent.com/Dicklesworthstone/franken_tts/main/install.ps1"))) -EasyMode

.EXAMPLE
    .\install.ps1 -Version 0.1.4 -InstallDir C:\tools\ftts -EasyMode
#>
[CmdletBinding()]
param(
    [string] $Version,
    [string] $InstallDir,
    [switch] $EasyMode,
    [switch] $Force,
    [switch] $NoVerify,
    [switch] $Quiet
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = 'Dicklesworthstone/franken_tts'

# Windows PowerShell 5.1 still negotiates TLS 1.0 by default, which GitHub refuses. Tls13 is not
# defined on older .NET Framework, and naming it unconditionally throws on exactly the stock 5.1
# hosts this one-liner most needs to work on — so ask for it, and settle for 1.2 when absent.
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13
} catch {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
}

function Write-Info { param([string] $Message) if (-not $Quiet) { Write-Host "-> $Message" -ForegroundColor Cyan } }
function Write-Ok   { param([string] $Message) if (-not $Quiet) { Write-Host "OK $Message" -ForegroundColor Green } }
function Write-Warn { param([string] $Message) Write-Host "!  $Message" -ForegroundColor Yellow }

function Get-ProxyArgs {
    # Honour the same environment corporate networks already set for curl.
    $proxy = if ($env:HTTPS_PROXY) { $env:HTTPS_PROXY } elseif ($env:HTTP_PROXY) { $env:HTTP_PROXY } else { $null }
    if ($proxy) { return @{ Proxy = $proxy; ProxyUseDefaultCredentials = $true } }
    return @{}
}

function Get-RemoteText {
    <#
        Fetches a URL as text.

        `Invoke-WebRequest -UseBasicParsing` hands back `.Content` as a **byte[]** whenever the
        response is not a recognized text type — and GitHub serves release assets, SHA256SUMS
        included, as application/octet-stream. Splitting that byte array on newlines silently
        yields one "line" per byte (516 lines of integers for a 516-byte manifest), so every
        checksum lookup misses and the installer reports "no checksum published" for a release
        that published one. Decoding explicitly is the fix; the string branch keeps PowerShell 7,
        where the same call already returns text, on the same path.
    #>
    param([string] $Uri, [hashtable] $ProxyArgs)
    $raw = (Invoke-WebRequest -Uri $Uri -UseBasicParsing @ProxyArgs).Content
    if ($raw -is [byte[]]) { return [Text.Encoding]::UTF8.GetString($raw) }
    return [string] $raw
}

function Resolve-Architecture {
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($env:PROCESSOR_ARCHITEW6432) { $arch = $env:PROCESSOR_ARCHITEW6432 }
    switch ($arch) {
        'AMD64' { return 'windows_amd64' }
        'ARM64' {
            # x64 emulation runs the amd64 build correctly on ARM64 Windows; say so rather than
            # failing, but name it, because the user is not getting native performance.
            Write-Warn 'ARM64 Windows detected. There is no native arm64 build yet; installing the x64 build, which runs under emulation.'
            return 'windows_amd64'
        }
        default { throw "Unsupported processor architecture '$arch'. Prebuilt Windows binaries cover x64 only; build from source with cargo instead." }
    }
}

function Resolve-Version {
    param([string] $Requested)
    if ($Requested) { return $Requested.TrimStart('v') }
    Write-Info 'Resolving the latest release'
    $proxyArgs = Get-ProxyArgs
    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" `
            -Headers @{ 'User-Agent' = 'franken_tts-installer' } @proxyArgs
        return ([string] $release.tag_name).TrimStart('v')
    } catch {
        throw "Could not resolve the latest release from the GitHub API: $($_.Exception.Message). Pass -Version explicitly."
    }
}

function Get-InstalledVersion {
    param([string] $Exe)
    if (-not (Test-Path -LiteralPath $Exe)) { return $null }
    try {
        $output = & $Exe --version 2>$null
        if ($output -match '(\d+\.\d+\.\d+)') { return $Matches[1] }
    } catch { }
    return $null
}

function Assert-Checksum {
    param([string] $File, [string] $Expected)
    $actual = (Get-FileHash -LiteralPath $File -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "Checksum mismatch for $(Split-Path -Leaf $File): expected $Expected, got $actual. The download was corrupted or tampered with; nothing was installed."
    }
    Write-Ok "Checksum verified ($($Expected.Substring(0, 16))...)"
}

# --- main ---------------------------------------------------------------------------------------

$platform = Resolve-Architecture
$resolved = Resolve-Version -Requested $Version
if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\franken_tts\bin' }
$ftts = Join-Path $InstallDir 'ftts.exe'

Write-Host ''
Write-Host 'franken_tts installer' -ForegroundColor Green
Write-Host 'Pure-Rust Qwen3-TTS voice synthesis - no Python, no GPU' -ForegroundColor DarkGray
Write-Host ''

$installed = Get-InstalledVersion -Exe $ftts
if ($installed -eq $resolved -and -not $Force) {
    Write-Ok "franken_tts $resolved is already installed at $InstallDir"
    Write-Info 'Use -Force to reinstall.'
    exit 0
}

$archive = "franken_tts-$resolved-$platform.zip"
$base = "https://github.com/$Repo/releases/download/v$resolved"
$temp = Join-Path ([IO.Path]::GetTempPath()) ("ftts-install-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temp -Force | Out-Null

try {
    $proxyArgs = Get-ProxyArgs
    $archivePath = Join-Path $temp $archive

    Write-Info "Downloading $archive"
    try {
        # -UseBasicParsing everywhere: on Windows PowerShell 5.1 the default engine is Internet
        # Explorer's, which throws outright when IE's first-launch configuration has never run —
        # a stock, freshly-imaged machine, i.e. the common case for a one-liner install.
        Invoke-WebRequest -Uri "$base/$archive" -OutFile $archivePath -UseBasicParsing @proxyArgs
    } catch {
        throw "Download failed for $base/$archive : $($_.Exception.Message). Check the release page for the assets that exist."
    }

    if ($NoVerify) {
        Write-Warn 'Skipping checksum verification (-NoVerify).'
    } else {
        # The combined manifest is authoritative; the per-asset sidecar is the fallback, matching
        # what install.sh accepts so both installers verify against the same published bytes.
        #
        # Why the failures are recorded rather than ignored: swallowing them makes a network hiccup
        # or a parsing quirk indistinguishable from a release that genuinely published no
        # checksums, and sends the user to -NoVerify — the one place they should never be sent by
        # a bug. Verified on Windows PowerShell 5.1, where the first version of this did exactly
        # that.
        $expected = $null
        $why = @()
        try {
            $sums = Get-RemoteText -Uri "$base/SHA256SUMS" -ProxyArgs $proxyArgs
            foreach ($line in ($sums -split "`n")) {
                if ($line -match '^\s*([0-9a-fA-F]{64})\s+[\* ]?(.+?)\s*$') {
                    # sha256sum writes "hash  name", "hash *name" for binary mode, and manifests
                    # generated from a directory walk often carry a "./" prefix. Compare on the
                    # base name so a cosmetic difference in how the manifest was produced cannot
                    # send us down the "no checksum published" path and refuse a good download.
                    $listed = ($Matches[2] -replace '^\.[\\/]', '')
                    if ((Split-Path -Leaf $listed) -eq $archive) {
                        $expected = $Matches[1]
                        break
                    }
                }
            }
        } catch {
            $why += "SHA256SUMS: $($_.Exception.Message)"
        }
        if (-not $expected) {
            try {
                $sidecar = Get-RemoteText -Uri "$base/$archive.sha256" -ProxyArgs $proxyArgs
                if ($sidecar -match '([0-9a-fA-F]{64})') { $expected = $Matches[1] }
            } catch {
                $why += "$archive.sha256: $($_.Exception.Message)"
            }
        }
        if (-not $expected) {
            $detail = if ($why) { " Reason: " + ($why -join '; ') } else { ' Both were fetched but neither listed this file.' }
            throw "No checksum published for $archive (neither SHA256SUMS nor $archive.sha256).$detail Refusing to install unverified binaries; pass -NoVerify to override."
        }
        Assert-Checksum -File $archivePath -Expected $expected
    }

    Write-Info 'Extracting'
    $extract = Join-Path $temp 'extract'
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extract -Force

    $binaries = Get-ChildItem -Path $extract -Recurse -Filter '*.exe' |
        Where-Object { $_.BaseName -in @('ftts', 'franken_tts') }
    if (-not $binaries) { throw "The archive contained no ftts.exe or franken_tts.exe. Asset $archive may be malformed." }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    foreach ($binary in $binaries) {
        Copy-Item -LiteralPath $binary.FullName -Destination (Join-Path $InstallDir $binary.Name) -Force
        Write-Ok "Installed $($binary.Name)"
    }

    $onPath = ($env:PATH -split ';') -contains $InstallDir
    if (-not $onPath) {
        if ($EasyMode) {
            # User PATH only: a machine-wide change would need elevation and is not this script's
            # business.
            $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
            if (-not $userPath) { $userPath = '' }
            if (($userPath -split ';') -notcontains $InstallDir) {
                [Environment]::SetEnvironmentVariable('PATH', ($userPath.TrimEnd(';') + ';' + $InstallDir).TrimStart(';'), 'User')
            }
            $env:PATH = $env:PATH + ';' + $InstallDir
            Write-Ok "Added $InstallDir to your user PATH (new terminals pick it up)"
        } else {
            Write-Warn "$InstallDir is not on your PATH. Re-run with -EasyMode to add it, or add it yourself."
        }
    }

    $verified = Get-InstalledVersion -Exe $ftts
    if ($verified) { Write-Ok "ftts $verified responds" } else { Write-Warn 'Installed, but ftts --version did not report a version.' }

    Write-Host ''
    Write-Host '  Next: fetch the model once (about 2 GB, SHA-256 verified)' -ForegroundColor DarkGray
    Write-Host '    ftts pull' -ForegroundColor White
    Write-Host '    ftts say "Now is the time for all good men to come to the aid of the agents" out.wav' -ForegroundColor White
    Write-Host ''
    Write-Host "  Uninstall: Remove-Item -Recurse -Force '$InstallDir'" -ForegroundColor DarkGray
    Write-Host '             then drop it from PATH in System Properties > Environment Variables' -ForegroundColor DarkGray
    Write-Host ''
} finally {
    Remove-Item -Recurse -Force -LiteralPath $temp -ErrorAction SilentlyContinue
}
