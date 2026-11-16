# Installs reims-ctl, and the QEMU it runs, on Windows: what scripts/install-ctl.sh needs that
# Windows lacks, then that script, in MSYS2's CLANG64 shell.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Yes] [-DryRun] [-Checkout DIR] [-Channel stable|development]
#   irm https://raw.githubusercontent.com/steelbrain/reims-vgpu/master/scripts/install.ps1 | iex
#
# On Windows 10 or 11 on x86-64, it:
#   1. installs MSYS2 (into C:\msys64) and Git for Windows with winget, where they are missing:
#      MSYS2's CLANG64 builds the library and QEMU, and reims-ctl runs Git's git;
#   2. turns on the Windows Hypervisor Platform, the optional feature VMs run on, where it is
#      off: that step alone runs as an administrator (Windows asks), and Windows needs a restart
#      before a VM starts;
#   3. clones the repository into DIR (default %USERPROFILE%\reims-vgpu, the stable channel's
#      master branch) where it is not run from a checkout of it;
#   4. runs scripts/install-ctl.sh there in MSYS2's CLANG64 shell, which installs CLANG64's
#      packages and rustup where they are missing, builds the library and QEMU, and installs
#      QEMU, reims-ctl and reims-util in your profile (install-ctl.sh --help).
# Each step is skipped where its work is done, so a second run changes nothing it need not.
#
# -Yes      install what is missing without asking (install-ctl.sh --yes too)
# -DryRun   say what each step finds and would do, and change nothing
#
# REIMS_INSTALL_PRETEND_MISSING (msys2, git, whp, comma-separated) has the checks find those
# missing, for a dry run of the steps that install them on a machine that has them.
param(
    [switch]$Yes,
    [switch]$DryRun,
    [string]$Checkout = "",
    [ValidateSet("stable", "development")][string]$Channel = "stable",
    [string]$Msys2 = "C:\msys64"
)
$ErrorActionPreference = "Stop"
$Repo = "https://github.com/steelbrain/reims-vgpu"

function Say($m) { Write-Host "==> $m" -ForegroundColor White }
function Warn($m) { Write-Host "warning: $m" -ForegroundColor Yellow }
function Die($m) { Write-Host "error: $m" -ForegroundColor Red; exit 1 }
function Ask($q) {
    if ($Yes) { return $true }
    $a = Read-Host "$q [Y/n]"
    return ($a -eq "" -or $a -match "^[yY]")
}
$pretend = @()
if ($env:REIMS_INSTALL_PRETEND_MISSING) { $pretend = $env:REIMS_INSTALL_PRETEND_MISSING.Split(",") | ForEach-Object { $_.Trim() } }
# Runs a step that changes the machine, or says it would.
function Step($what, [scriptblock]$do) {
    if ($DryRun) { Say "would $what"; return }
    Say $what
    & $do
}

if (-not [Environment]::Is64BitOperatingSystem -or $env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
    Die "reims-ctl runs VMs on x86-64 Windows; this is $env:PROCESSOR_ARCHITECTURE"
}

# 1. MSYS2 and Git, with winget.
$bash = Join-Path $Msys2 "usr\bin\bash.exe"
$haveMsys2 = (Test-Path $bash) -and -not ($pretend -contains "msys2")
$gitCmd = Get-Command git.exe -ErrorAction SilentlyContinue
$haveGit = ($null -ne $gitCmd) -and ($gitCmd.Source -notlike "$Msys2*") -and -not ($pretend -contains "git")
$freshMsys2 = $false
foreach ($p in @(@{ name = "MSYS2"; id = "MSYS2.MSYS2"; have = $haveMsys2 }, @{ name = "Git for Windows"; id = "Git.Git"; have = $haveGit })) {
    if ($p.have) { Say "$($p.name) is installed"; continue }
    if (-not $DryRun -and -not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        Die "$($p.name) is missing, and winget, which installs it, is too: install App Installer from the Microsoft Store, or $($p.name) yourself, then run this again"
    }
    if (-not $DryRun -and -not (Ask "Install $($p.name) with winget?")) { Die "install $($p.name) ($($p.id)), then run this again" }
    Step "install $($p.name): winget install --id $($p.id) -e" {
        winget install --id $p.id -e --accept-source-agreements --accept-package-agreements
        if ($LASTEXITCODE -ne 0) { Die "winget could not install $($p.name) (exit $LASTEXITCODE)" }
    }
    if ($p.id -eq "MSYS2.MSYS2") { $freshMsys2 = $true }
}
if (-not $DryRun) {
    # The PATH winget's installers set, for git in this window.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Test-Path $bash)) { Die "MSYS2 is not in $Msys2 (no $bash): give its folder with -Msys2" }
}

# 2. The Windows Hypervisor Platform. Win32_OptionalFeature answers without an administrator's
# rights: InstallState 1 is on.
$whp = (Get-CimInstance Win32_OptionalFeature -Filter "Name='HypervisorPlatform'").InstallState
if ($pretend -contains "whp") { $whp = 2 }
$restart = $false
if ($whp -eq 1) {
    Say "The Windows Hypervisor Platform is on"
} else {
    if (-not $DryRun -and -not (Ask "Turn on the Windows Hypervisor Platform, which VMs run on? Windows asks for an administrator, and needs a restart after")) {
        Warn "VMs don't start until the Windows Hypervisor Platform is on: turn it on in Windows Features (optionalfeatures), then restart"
    } else {
        Step "turn on the Windows Hypervisor Platform, as an administrator: Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -All -NoRestart" {
            $p = Start-Process powershell.exe -Verb RunAs -Wait -PassThru -ArgumentList @(
                "-NoProfile", "-Command",
                "Enable-WindowsOptionalFeature -Online -FeatureName HypervisorPlatform -All -NoRestart | Out-Null; exit 0")
            if ($p.ExitCode -ne 0) { Die "turning on the Windows Hypervisor Platform failed (exit $($p.ExitCode))" }
        }
        $restart = $true
    }
}

# 3. The checkout: this script's own, else a clone.
$here = if ($PSScriptRoot) { Split-Path -Parent $PSScriptRoot } else { "" }
if ($Checkout -eq "") {
    if ($here -and (Test-Path (Join-Path $here "scripts\install-ctl.sh"))) { $Checkout = $here }
    else { $Checkout = Join-Path $env:USERPROFILE "reims-vgpu" }
}
$branch = if ($Channel -eq "development") { "development" } else { "master" }
if (Test-Path (Join-Path $Checkout "scripts\install-ctl.sh")) {
    Say "The checkout is $Checkout"
} else {
    if ((Test-Path $Checkout) -and (Get-ChildItem $Checkout -Force | Select-Object -First 1)) { Die "$Checkout is not empty and holds no checkout: give another folder with -Checkout" }
    Step "clone $Repo ($branch) into $Checkout" {
        git clone -b $branch $Repo $Checkout
        if ($LASTEXITCODE -ne 0) { Die "git clone failed (exit $LASTEXITCODE)" }
    }
}

# 4. install-ctl.sh in CLANG64: a login shell, which puts CLANG64's tools first on PATH, in the
# checkout (CHERE_INVOKING). An MSYS2 that has never fetched CLANG64's package list, as a new one
# hasn't, brings its packages up to date first, twice: the first update can replace pacman and
# MSYS2's runtime themselves, and then ends every MSYS2 program, as MSYS2 does.
$env:MSYSTEM = "CLANG64"
$env:CHERE_INVOKING = "1"
if ($freshMsys2 -or -not (Test-Path (Join-Path $Msys2 "var\lib\pacman\sync\clang64.db"))) {
    Step "bring the new MSYS2 up to date: pacman -Syuu, twice" {
        foreach ($i in 1, 2) { & $bash -lc "pacman -Syuu --noconfirm" }
    }
}
$unix = if ($DryRun) { $Checkout } else { (& $bash -lc "cygpath -u '$Checkout'").Trim() }
$flags = if ($Yes) { " --yes" } else { "" }
Step "run scripts/install-ctl.sh$flags in MSYS2's CLANG64 shell, in $Checkout" {
    Push-Location $Checkout
    try { & $bash -lc "cd '$unix' && scripts/install-ctl.sh$flags" } finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { Die "scripts/install-ctl.sh failed (exit $LASTEXITCODE)" }
}
if ($DryRun) { exit 0 }
Write-Host ""
Write-Host "reims-ctl is installed. Open a new PowerShell or Windows Terminal window and run reims-ctl." -ForegroundColor Green
if ($restart) {
    Warn "restart Windows before you start a VM: the Windows Hypervisor Platform, which VMs run on, turns on at the restart"
}
