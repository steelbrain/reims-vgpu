#!/usr/bin/env bash
#
# Builds and installs reims-ctl, and the QEMU it runs, with reims-vgpu's own
# GPU device on this checkout's reims-vgpu.
#
#   scripts/install-ctl.sh [--yes] [--update] [--skip-qemu]
#
# Linux on x86-64 (Ubuntu or Debian, Arch or CachyOS; other distributions get
# the list of what to install), where the library runs on Vulkan and VMs run;
# Windows on x86-64, in MSYS2's CLANG64 shell, whose clang builds the library
# and QEMU (QEMU builds on Windows only with MinGW), where VMs run on the
# Windows Hypervisor Platform; and arm64 Macs and arm64 Linux (Ubuntu or Debian, Arch, Fedora), where
# reims-ctl makes arm64 VMs by restoring macOS from an IPSW, which runs no
# QEMU: there it installs reims-ctl alone (step 4). It:
#   1. checks the build's dependencies, and offers to install those missing
#      (apt, pacman or dnf, through sudo; on a Mac, Xcode's command line tools
#      are needed), and rustup through rustup.rs, beside any other cargo, whose
#      installer asks its own questions;
#   2. fetches vendor/qemu, QEMU's source, without QEMU's own submodules (its
#      build fetches what it needs); cargo fetches metal2vulkan itself, at the
#      commit the library's Cargo.toml names;
#   3. builds the library (cargo xtask dist) on Vulkan and QEMU on it, in
#      ~/.cache/reims-ctl, and installs QEMU into
#      ~/.local/share/reims-ctl/qemu/<build>, linking qemu/current to it. A
#      build never replaces another: a running VM keeps the binary it started
#      with;
#   4. installs reims-ctl with cargo install (into ~/.cargo/bin, or
#      CARGO_INSTALL_ROOT's bin);
#   5. installs reims-util beside it, which makes a new VM's OpenCore image and
#      downloads its macOS installer from Apple. reims-ctl does
#      (__record-install), at the commit it names
#      (crates/reims-ctl/src/reims_util.rs), replacing one from another commit
#      and leaving one at that commit as it is.
#
# --yes       install what is missing without asking (rustup with -y too)
# --update    pull this checkout first (fast-forward only), putting back the
#             committed Cargo.lock the build rewrote; where the remote branch
#             was force-pushed, the checkout's commit is kept on a branch
#             BRANCH-before-update-DATE and the checkout moves to the remote's;
#             reims-ctl update runs this
# --skip-qemu install reims-ctl alone, keeping the installed QEMU, which it
#             needs; it warns where the QEMU or library this checkout builds
#             changed since that QEMU's build
#
# Environment: CARGO_TARGET_DIR, the library's build (default the checkout's
# target/); XDG_DATA_HOME and XDG_CACHE_HOME move the install and the build
# from ~/.local/share and ~/.cache.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The PATH this ran with, which the user's shell has; the build's own puts rustup's first.
START_PATH="$PATH"
YES=0 UPDATE=0 SKIP_QEMU=0
for arg in "$@"; do
  case "$arg" in
    --yes | -y) YES=1 ;;
    --update) UPDATE=1 ;;
    --skip-qemu) SKIP_QEMU=1 ;;
    -h | --help)
      sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown option: $arg (see --help)" >&2
      exit 2
      ;;
  esac
done

bold=$'\033[1m' red=$'\033[31m' yellow=$'\033[33m' green=$'\033[32m' off=$'\033[0m'
[[ -t 1 ]] || bold='' red='' yellow='' green='' off=''
say() { printf '%s==> %s%s\n' "$bold" "$*" "$off"; }
warn() { printf '%swarning:%s %s\n' "$yellow" "$off" "$*" >&2; }
die() {
  printf '%serror:%s %s\n' "$red" "$off" "$*" >&2
  exit 1
}
# ask QUESTION: yes with --yes; no without a terminal to ask on.
ask() {
  [[ $YES == 1 ]] && return 0
  [[ -t 0 ]] || return 1
  local a
  read -r -p "$1 [Y/n] " a
  [[ -z $a || $a == [yY]* ]]
}

OS="$(uname -s)" ARCH="$(uname -m)"
# BUILDS_QEMU: whether this host runs VMs, on QEMU built on the library: Linux
# x86-64. An arm64 host makes arm64 VMs by a restore, which runs no QEMU, and
# reims-ctl doesn't start them: it gets reims-ctl alone.
BUILDS_QEMU=1
case "$OS/$ARCH" in
  Linux/x86_64)
    PLATFORM=linux-x86_64
    # libusb (usb-host, USB passthrough) and SPICE (-spice, a display a SPICE
    # client connects to) are enabled by name: configure leaves out an
    # optional feature whose library it does not find, saying nothing.
    QEMU_CONFIGURE=(--target-list=x86_64-softmmu --enable-kvm --disable-sdl --enable-gtk --enable-slirp
      --enable-libusb --enable-spice)
    QEMU_BINARY=qemu-system-x86_64
    ;;
  MINGW*/x86_64 | MSYS*/x86_64)
    [[ ${MSYSTEM:-} == CLANG64 ]] || die "on Windows, run this in MSYS2's CLANG64 shell (\"MSYS2 CLANG64\" in the Start menu), whose clang builds the library and QEMU; this is its ${MSYSTEM:-MSYS} shell"
    PLATFORM=windows-x86_64
    # WHPX, the Windows Hypervisor Platform, in place of KVM; the binaries in
    # the prefix's bin, as reims-ctl runs them (qemu/current/bin); and no
    # guest agent, which reims-ctl does not use and whose Windows build needs
    # windmc, which none of the packages below provides.
    QEMU_CONFIGURE=(--target-list=x86_64-softmmu --enable-whpx --disable-sdl --enable-gtk --enable-slirp
      --enable-libusb --disable-guest-agent --bindir=bin)
    QEMU_BINARY=qemu-system-x86_64.exe
    ;;
  Darwin/arm64 | Linux/aarch64) BUILDS_QEMU=0 ;;
  *) die "reims-ctl runs VMs on Linux and Windows x86-64 hosts, and makes arm64 VMs from an IPSW on arm64 Macs and arm64 Linux; this is $OS on $ARCH" ;;
esac
# WINDOWS: in MSYS2 on Windows, whose own HOME is not the user's profile, where
# reims-ctl keeps its files (%USERPROFILE%, as rustup and cargo do).
WINDOWS=0 USER_HOME=$HOME
if [[ ${PLATFORM:-} == windows-* ]]; then
  WINDOWS=1 USER_HOME="$(cygpath -u "$USERPROFILE")"
fi
# shown PATH: PATH as the user names it, for a message: on Windows in Windows'
# own form (C:\Users\me\...), which PowerShell and cmd take, where MSYS2's
# (/c/Users/me/...) is this bash's alone; elsewhere as it is.
shown() {
  if [[ $WINDOWS == 1 ]]; then cygpath -w "$1"; else printf '%s' "$1"; fi
}
# This script as the messages that send the user back to it name it: on Windows
# scripts\install.ps1, which runs it in MSYS2's CLANG64 shell (as reims-ctl's
# messages name it there).
SELF=$0
[[ $WINDOWS == 1 ]] && SELF='scripts\install.ps1'

# The XDG directory VAR names, else ~/FALLBACK: ignored unless absolute (as
# reims-ctl does).
xdg() { # VAR FALLBACK
  local v="${!1:-}"
  [[ $WINDOWS == 1 && $v == [A-Za-z]:* ]] && v="$(cygpath -u "$v")"
  if [[ $v == /* ]]; then printf '%s' "$v"; else printf '%s' "$USER_HOME/$2"; fi
}
DATA="$(xdg XDG_DATA_HOME .local/share)/reims-ctl"
CACHE="$(xdg XDG_CACHE_HOME .cache)/reims-ctl"
mkdir -p "$DATA" "$CACHE"

# Every VM runs on QEMU: --skip-qemu keeps the installed one, and needs one.
if [[ $SKIP_QEMU == 1 && $BUILDS_QEMU == 0 ]]; then
  die "--skip-qemu keeps the installed QEMU, and this host has none: reims-ctl makes arm64 VMs here by a restore, which runs no QEMU; run this without --skip-qemu"
fi
if [[ $SKIP_QEMU == 1 && ! -x "$DATA/qemu/current/bin/$QEMU_BINARY" ]]; then
  die "--skip-qemu keeps the installed QEMU, and none is installed ($(shown "$DATA/qemu/current/bin/$QEMU_BINARY") is missing): run this without --skip-qemu, which builds it"
fi

# ------------------------------------------------------------------ update

# published BEFORE: whether HEAD is a commit the remote branch once held, as
# BEFORE (its remote-tracking ref before the fetch) or an earlier value of that
# ref contains it. The ref's reflog holds each value, an update's old one as
# well as its new: a clone logs none, so the first fetch's old one is its only
# record of what the clone took. A HEAD the new remote branch lacks was then
# taken away by a force push; local commits never were the remote's.
published() {
  local c log
  log=$(git -C "$ROOT" rev-parse --git-path "logs/$upref")
  [[ $log == /* ]] || log=$ROOT/$log
  while read -r c; do
    git -C "$ROOT" merge-base --is-ancestor HEAD "$c" 2>/dev/null && return 0
  done < <(echo "$1"; [[ -f $log ]] && awk '{ print $1; print $2 }' "$log")
  return 1
}

if [[ $UPDATE == 1 ]]; then
  say "Pulling $(shown "$ROOT")"
  if upref=$(git -C "$ROOT" rev-parse --symbolic-full-name '@{u}' 2>/dev/null); then
    # The build writes Cargo.lock again (cargo adds the git sources the
    # committed lock leaves out), and a changed lock stops a pull that changes
    # it too: the committed one goes back first.
    git -C "$ROOT" checkout -q HEAD -- Cargo.lock
    start=$(git -C "$ROOT" rev-parse HEAD)
    branch=$(git -C "$ROOT" symbolic-ref --short HEAD)
    remote=$(git -C "$ROOT" config "branch.$branch.remote")
    before=$(git -C "$ROOT" rev-parse "$upref")
    if [[ $remote != . ]]; then
      # Forced (+), so a remote branch that was force-pushed still moves its
      # remote-tracking ref, where a plain fetch refuses it.
      merge=$(git -C "$ROOT" config "branch.$branch.merge")
      git -C "$ROOT" fetch -q "$remote" "+$merge:$upref" || die "git fetch from $remote failed"
    fi
    if git -C "$ROOT" merge-base --is-ancestor HEAD "$upref"; then
      git -C "$ROOT" merge -q --ff-only "$upref" || die "the checkout's local changes stop the update: commit or set them aside, then run $SELF"
    elif published "$before"; then
      # The checkout's commit is kept on a branch of its own, made once the
      # move has worked; --keep refuses to move over a local change the move
      # would overwrite.
      old=$(git -C "$ROOT" rev-parse HEAD)
      git -C "$ROOT" reset -q --keep "$upref" || die "the checkout's local changes stop the update: commit or set them aside, then run $SELF"
      backup="$branch-before-update-$(date +%Y%m%d-%H%M%S)"
      git -C "$ROOT" branch "$backup" "$old"
      warn "${upref#refs/remotes/} was force-pushed: the checkout's earlier commit stays on the branch $backup"
    else
      die "the checkout has commits of its own that ${upref#refs/remotes/} lacks: update it yourself, then run $SELF"
    fi
    # bash reads on from the file it opened, which the move replaced: where the
    # move changed this script, the new one runs the update instead, its pull
    # then a no-op, so its dependencies and QEMU options reach this update.
    if ! git -C "$ROOT" diff --quiet "$start" HEAD -- scripts/install-ctl.sh; then
      say "The update changed scripts/install-ctl.sh: running the new one"
      exec bash "$0" "$@"
    fi
  else
    warn "the checkout's branch tracks no remote branch: building it as it is"
  fi
fi

# ------------------------------------------------------------------ line ends

# rewrite_crlf: writes each text file of the checkout whose working copy differs
# from its commit only in CRLF line ends out again from the commit. Git for
# Windows' installer sets core.autocrlf=true, and a checkout it made before the
# repository had .gitattributes holds every text file so, while the build reads
# some of them byte for byte (.gitattributes). git keeps such a file as it is:
# .gitattributes has it compare the file as LF, so it finds nothing to write,
# and a pull leaves each file the pull does not change as it was (measured: a
# clone at core.autocrlf=true moved to the commit that added .gitattributes
# kept all 1730 text files CRLF). A file with changes of its own is kept.
rewrite_crlf() {
  local crlf own redo
  crlf="$(git -C "$ROOT" -c core.quotePath=false ls-files --eol |
    awk -F '\t' '$1 ~ /^i\/lf +w\/(crlf|mixed) / { print $2 }')"
  [[ -n $crlf ]] || return 0
  own="$(git -C "$ROOT" -c core.quotePath=false diff --name-only)"
  redo="$(grep -vxF -f <(printf '%s\n' "$own") <<<"$crlf" || true)"
  [[ -n $redo ]] || return 0
  say "Writing $(wc -l <<<"$redo" | tr -d ' ') text files out again with the LF line ends they were committed with"
  (cd "$ROOT" && tr '\n' '\0' <<<"$redo" | xargs -0 rm -f && git checkout-index -u --stdin <<<"$redo")
}
if [[ $WINDOWS == 1 ]]; then
  rewrite_crlf || die "could not write the checkout's CRLF text files out again from its commit"
fi

# ------------------------------------------------------------------ paths

# QEMU's configure refuses a source or build directory whose path holds a space
# ("main directory cannot contain spaces nor colons"), and sdl3-sys, which
# builds SDL3 under cargo's target directory and then reads SDL3's pkg-config
# file, finds no library where a space splits that file's -L path ("could not
# find native static library SDL3"). A Windows user's folder can hold a space
# (C:\Users\First Last): the checkout and the target directory must be outside
# one, and QEMU builds beside the library in the target directory where the
# cache directory is in one (QEMU_BUILDS below). Measured on the mini with a
# profile named "Test User", 2026-11-16: QEMU's configure stopped on the cache,
# and SDL3 on a checkout in the profile.
if [[ $WINDOWS == 1 && $BUILDS_QEMU == 1 && $SKIP_QEMU == 0 ]]; then
  [[ $ROOT != *[[:space:]]* ]] ||
    die "the checkout's path, $(shown "$ROOT"), has a space, which QEMU's build and SDL3's refuse: clone the repository into a folder without spaces (install.ps1 -Checkout C:\\reims-vgpu), and run it there"
  [[ ${CARGO_TARGET_DIR:-} != *[[:space:]]* ]] ||
    die "CARGO_TARGET_DIR, $(shown "$CARGO_TARGET_DIR"), has a space, which SDL3's build refuses: give it a folder without spaces, or unset it"
fi

# ------------------------------------------------------------------ dependencies

missing=()

# A Mac's C compiler and linker, which Rust's builds use.
if [[ $OS == Darwin ]]; then
  xcode-select -p >/dev/null 2>&1 || die "Xcode's command line tools are missing: run xcode-select --install, then this again"
fi

distro_id="" distro_like=""
if [[ $OS == Linux && -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  distro_id="$(. /etc/os-release && echo "${ID:-}")"
  # shellcheck disable=SC1091
  distro_like="$(. /etc/os-release && echo "${ID_LIKE:-}")"
fi
# The library (cargo xtask dist: binutils's ld, nm, objcopy and ar for the
# archive it installs; CMake and the X11, Wayland, xkbcommon and EGL headers
# for SDL3's video, which it builds from source), and QEMU (glib, pixman, libslirp; GTK 3 for
# its own window, which a VM on VMware SVGA shows its display in; pipewire,
# pulse and ALSA for its audio backends; libusb for USB passthrough; the
# SPICE server for -spice; bzip2 unpacks its firmware). A new x86-64 VM's
# OVMF comes from reims-util, with its OpenCore image.
apt_pkgs=(build-essential git curl ca-certificates pkg-config python3 python3-venv ninja-build
  flex bison bzip2 cmake binutils libglib2.0-dev libpixman-1-dev libslirp-dev
  zlib1g-dev libfdt-dev libpipewire-0.3-dev libpulse-dev libasound2-dev libusb-1.0-0-dev
  libspice-server-dev libvulkan1
  libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxfixes-dev libxss-dev
  libxtst-dev libxkbcommon-dev libwayland-dev libwayland-bin libdecor-0-dev
  libegl-dev libgl-dev libdrm-dev libgbm-dev libdbus-1-dev libudev-dev libgtk-3-dev)
pacman_pkgs=(base-devel git curl python ninja pkgconf flex bison bzip2 cmake
  glib2 pixman libslirp zlib dtc pipewire libpulse alsa-lib libusb spice vulkan-icd-loader
  libx11 libxext libxrandr libxcursor libxi libxfixes libxss libxtst libxkbcommon
  wayland libdecor mesa libdrm dbus systemd-libs gtk3)
dnf_pkgs=()
# Windows: MSYS2's own git and curl, make (the cmake crate, building SDL3, looks
# for it to choose CMake's generator, and stops without it), and CLANG64's clang
# and lld, which build the library (x86_64-pc-windows-gnullvm) and QEMU, QEMU's
# build tools and the libraries it links; and libva and Mesa, whose VA-API driver
# on Direct3D 12 (vaon12) decodes the reims-graphics device's video.
msys2_pkgs=(git curl diffutils make
  mingw-w64-clang-x86_64-{clang,lld,pkgconf,meson,ninja,cmake,python,glib2,pixman,libslirp,gtk3,libusb,zstd,libva,mesa})
if [[ $BUILDS_QEMU == 0 ]]; then
  # reims-ctl alone, for arm64 Linux: a C compiler and linker for Rust's
  # builds, git, and curl for rustup. Fedora is Asahi Linux's own distribution.
  apt_pkgs=(build-essential git curl ca-certificates)
  pacman_pkgs=(base-devel git curl)
  dnf_pkgs=(gcc make git curl)
fi
case "$OS $distro_id $distro_like " in
  Darwin*) ;;
  MINGW* | MSYS*)
    for p in "${msys2_pkgs[@]}"; do
      pacman -Qq "$p" >/dev/null 2>&1 || missing+=("$p")
    done
    if ((${#missing[@]})); then
      say "Missing packages: ${missing[*]}"
      ask "Install them with pacman?" || die "install them (pacman -S --needed ${missing[*]}), then run this again"
      pacman -S --needed --noconfirm "${missing[@]}"
    fi
    ;;
  *" debian "* | *" ubuntu "*)
    for p in "${apt_pkgs[@]}"; do
      dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || missing+=("$p")
    done
    if ((${#missing[@]})); then
      say "Missing packages: ${missing[*]}"
      ask "Install them with sudo apt-get?" || die "install them (sudo apt-get install ${missing[*]}), then run this again"
      sudo apt-get update
      sudo apt-get install -y "${missing[@]}"
    fi
    ;;
  *" fedora "*)
    if ((${#dnf_pkgs[@]} == 0)); then
      warn "$distro_id is neither Debian/Ubuntu nor Arch/CachyOS: install the equivalents of these Ubuntu packages yourself if the build stops: ${apt_pkgs[*]}"
    else
      for p in "${dnf_pkgs[@]}"; do
        rpm -q "$p" >/dev/null 2>&1 || missing+=("$p")
      done
      if ((${#missing[@]})); then
        say "Missing packages: ${missing[*]}"
        ask "Install them with sudo dnf?" || die "install them (sudo dnf install ${missing[*]}), then run this again"
        sudo dnf install -y "${missing[@]}"
      fi
    fi
    ;;
  *" arch "* | *" cachyos "*)
    for p in "${pacman_pkgs[@]}"; do
      pacman -Qq "$p" >/dev/null 2>&1 || pacman -Qqg "$p" >/dev/null 2>&1 || missing+=("$p")
    done
    if ((${#missing[@]})); then
      say "Missing packages: ${missing[*]}"
      ask "Install them with sudo pacman?" || die "install them (sudo pacman -S --needed ${missing[*]}), then run this again"
      sudo pacman -S --needed --noconfirm "${missing[@]}"
    fi
    ;;
  *)
    warn "$distro_id is neither Debian/Ubuntu nor Arch/CachyOS: install the equivalents of these Ubuntu packages yourself if the build stops: ${apt_pkgs[*]}"
    ;;
esac
# On Windows, VMs run on the Windows Hypervisor Platform, an optional feature of
# Windows: Win32_OptionalFeature's InstallState is 1 where it is on.
if [[ $WINDOWS == 1 ]]; then
  whp="$(powershell -NoProfile -Command "(Get-CimInstance Win32_OptionalFeature -Filter \"Name='HypervisorPlatform'\").InstallState" 2>/dev/null | tr -d '\r')"
  [[ $whp == 1 ]] || warn "the Windows Hypervisor Platform is off, and VMs run on it: turn it on in Windows Features (run optionalfeatures, tick Windows Hypervisor Platform), then restart Windows"
fi
# The rest of the host's checks are for the VMs Linux runs: KVM, the library's
# GPU access to the guest's memory, and USB passthrough.
if [[ $BUILDS_QEMU == 1 && $WINDOWS == 0 ]]; then
  if [[ ! -r /dev/kvm || ! -w /dev/kvm ]]; then
    warn "/dev/kvm is not usable by you: enable virtualization in the firmware and add yourself to the kvm group (sudo usermod -aG kvm \$USER; log in again), or VMs will not start"
  fi
  # The guest's memory reaches the GPU through /dev/udmabuf, on every GPU: without
  # it the library's views of that memory take a kernel mapping per 4 KiB page,
  # past vm.max_map_count at 4K even at the guest's desktop (reims-ctl's
  # runtime::udmabuf_problem, docs/qemu.md "Linux hosts: vm.max_map_count").
  why="the guest's memory reaches the GPU through it, and without it the guest can crash at higher resolutions, at 4K even at its desktop"
  udmabuf=0
  if [[ ! -e /dev/udmabuf ]]; then
    warn "there's no /dev/udmabuf: $why. Load it with sudo modprobe udmabuf, and put udmabuf in a file in /etc/modules-load.d to keep it"
  else
    group="$(stat -c %G /dev/udmabuf)"
    if [[ -r /dev/udmabuf && -w /dev/udmabuf ]]; then
      udmabuf=1
      if [[ $group != root ]] && ! id -nG | tr ' ' '\n' | grep -qx "$group"; then
        say "You can open /dev/udmabuf in this session, but you're not in its $group group: over ssh, where a desktop session's access doesn't apply, add yourself to it (sudo usermod -aG $group \$USER) to start VMs there"
      fi
    elif [[ $group == root ]]; then
      warn "only root can open /dev/udmabuf: $why. Give it to the kvm group with a udev rule, a file in /etc/udev/rules.d holding KERNEL==\"udmabuf\", GROUP=\"kvm\", MODE=\"0660\", add yourself to kvm (sudo usermod -aG kvm \$USER), and restart"
    else
      say "You can't open /dev/udmabuf: $why. You need its $group group"
      if ask "Add you to the $group group with sudo usermod?"; then
        sudo usermod -aG "$group" "$USER"
        say "Added you to $group: log in again for it to take effect"
      else
        warn "add yourself to the $group group (sudo usermod -aG $group \$USER) and log in again, or the guest can crash at higher resolutions"
      fi
    fi
  fi
  # With udmabuf, vm.max_map_count is only a speed-up, on NVIDIA: a view of an
  # x86-64 guest's memory takes a mapping per 4 KiB page there, and past the
  # library's budget (the limit less 16,384 and 8,192) its buffers are copied,
  # which works and is slower (reims-ctl's runtime::max_map_count). A new VM
  # gets half the host's memory, 4 to 16 GiB.
  maps="$(cat /proc/sys/vm/max_map_count 2>/dev/null || echo 0)"
  vm_mib=$(($(awk '/^MemTotal:/ { print $2 }' /proc/meminfo) / 2 / 1048576 * 1024))
  vm_mib=$((vm_mib < 4096 ? 4096 : vm_mib > 16384 ? 16384 : vm_mib))
  want=$((vm_mib * 256 + 24576))
  if ((udmabuf && maps < want)) && grep -qsx 0x10de /sys/class/drm/*/device/vendor; then
    raise=1048576
    while ((raise < want)); do raise=$((raise * 2)); done
    uncopied=$((maps > 24576 ? (maps - 24576) / 256 : 0))
    say "vm.max_map_count is $maps: on an NVIDIA GPU, past about $uncopied MiB of a new VM's $((vm_mib / 1024)) GiB the GPU gets copies of its memory, which is slower; sudo sysctl -w vm.max_map_count=$raise, and a file in /etc/sysctl.d to keep it, avoids them"
  fi
  # MGLRU's thrashing guard (reims-ctl's runtime::MIN_TTL_FILE): above 0, the
  # kernel kills a process where it can't keep the pages used in the last that
  # many ms, even with RAM free, and a VM's QEMU can be the one. Measured on
  # CachyOS hosts at 100, its kernel's default there. Offered off on a terminal
  # only, and never required; the tmpfiles.d line keeps it off.
  min_ttl_file=/sys/kernel/mm/lru_gen/min_ttl_ms
  min_ttl_line='w /sys/kernel/mm/lru_gen/min_ttl_ms - - - - 0'
  min_ttl="$(cat "$min_ttl_file" 2>/dev/null || echo 0)"
  if [[ $min_ttl =~ ^[0-9]+$ ]] && ((min_ttl > 0)); then
    warn "$min_ttl_file is $min_ttl: where the kernel can't keep the pages used in the last $min_ttl ms in memory, it kills a process, which can be a VM's QEMU, even while RAM is free (measured on hosts at 100, killing builds with most RAM free); 0 turns this off"
    if [[ -t 0 ]] && ask "Write 0 to $min_ttl_file now with sudo?"; then
      echo 0 | sudo tee "$min_ttl_file" >/dev/null || warn "sudo didn't write it"
    fi
    keep="the line $min_ttl_line in a file in /etc/tmpfiles.d, such as /etc/tmpfiles.d/reims-ctl-min-ttl.conf"
    if [[ "$(cat "$min_ttl_file" 2>/dev/null)" == 0 ]]; then
      say "It's 0 now; $keep, keeps it at 0 after a restart"
    else
      say "echo 0 | sudo tee $min_ttl_file turns it off now, and $keep, keeps it at 0 after a restart"
    fi
  fi
  # USB passthrough (reims-ctl usb): QEMU opens a USB device's file in
  # /dev/bus/usb, which only root may write unless a udev rule says otherwise.
  # This one gives USB devices other than hubs to the kvm group, over SSH too,
  # and to the user at the host's screen (uaccess, which a file sorting before
  # systemd's 73-seat-late.rules sets). reims-ctl's usb::UDEV_RULE is the same
  # line, which its tests check.
  usb_rule_file=/etc/udev/rules.d/70-reims-ctl-usb.rules
  usb_rule='SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{bDeviceClass}!="09", GROUP="kvm", MODE="0664", TAG+="uaccess"'
  if ! grep -qsxF "$usb_rule" "$usb_rule_file"; then
    say "A VM can be given this host's USB devices once QEMU, running as you, may open them"
    if ask "Add the udev rule $usb_rule_file, giving USB devices other than hubs to the kvm group and to you at this host's screen, with sudo?"; then
      printf '# reims-ctl: USB devices for VMs (reims-ctl usb --help)\n%s\n' "$usb_rule" |
        sudo tee "$usb_rule_file" >/dev/null
      # Applied to the devices plugged in now, as to those plugged in later.
      sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=usb ||
        warn "udev did not reload its rules: unplug a USB device and plug it in again, or restart, before giving it to a VM"
    else
      warn "without the rule, a VM can be given only the USB devices you may open; reims-ctl usb --help shows it"
    fi
  fi
  # Unless /dev/udmabuf's group, kvm, was the one named above.
  if [[ ${group:-} != kvm ]] && getent group kvm >/dev/null && ! id -nG | tr ' ' '\n' | grep -qx kvm; then
    say "You're not in the kvm group: over SSH, where a desktop session's access doesn't apply, add yourself to it (sudo usermod -aG kvm \$USER) to give VMs USB devices there"
  fi
fi

# The build runs the toolchain rust-toolchain.toml pins, through rustup's proxies in Cargo's
# home, so they come first on PATH here: a shell that sourced no Cargo env has none on it, and
# another cargo found first, a distribution's (Arch's rust, which its cargo-nextest pulls in),
# builds with its own release.
CARGO_BIN="${CARGO_HOME:-$USER_HOME/.cargo}/bin"
[[ $WINDOWS == 1 ]] && CARGO_BIN="$(cygpath -u "$CARGO_BIN")"
# On Windows, rustup's host is x86_64-pc-windows-gnullvm, whose builds link with
# CLANG64's clang, so the build needs no Visual Studio.
RUSTUP_ARGS=()
[[ $WINDOWS == 1 ]] && RUSTUP_ARGS=(--default-host x86_64-pc-windows-gnullvm)
if [[ -x "$CARGO_BIN/rustup" ]]; then
  export PATH="$CARGO_BIN:$PATH"
fi
if ! command -v rustup >/dev/null 2>&1; then
  if command -v cargo >/dev/null 2>&1; then
    say "$(shown "$(command -v cargo)") is a cargo without rustup: the build needs the toolchain rust-toolchain.toml pins, which rustup installs beside it"
  else
    say "Rust is not installed"
  fi
  ask "Install rustup from https://rustup.rs?" || die "install rustup (https://rustup.rs), then run this again"
  if [[ $YES == 1 ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y ${RUSTUP_ARGS[@]+"${RUSTUP_ARGS[@]}"}
  else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- ${RUSTUP_ARGS[@]+"${RUSTUP_ARGS[@]}"}
  fi
  export PATH="$CARGO_BIN:$PATH"
fi
# On Windows the build runs rust-toolchain.toml's channel for the x86_64-pc-windows-gnullvm host
# whatever rustup's default host is, as a rustup this script installs has it: its builds link
# with CLANG64's clang and lld. One whose default host is x86_64-pc-windows-msvc would build
# xtask, the build scripts, reims-ctl and reims-util for it, which link only with Visual Studio's
# link.exe. RUSTUP_TOOLCHAIN reaches every cargo run below, reims-util's install among them.
if [[ $WINDOWS == 1 ]]; then
  CHANNEL="$(sed -n 's/^channel *= *"\([^"]*\)".*/\1/p' "$ROOT/rust-toolchain.toml")"
  [[ -n $CHANNEL ]] || die "$(shown "$ROOT/rust-toolchain.toml") names no channel"
  export RUSTUP_TOOLCHAIN="$CHANNEL-x86_64-pc-windows-gnullvm"
  rustup toolchain install --profile minimal "$RUSTUP_TOOLCHAIN" ||
    die "rustup could not install $RUSTUP_TOOLCHAIN, the toolchain the build runs on Windows"
  # Every C and C++ build below runs CLANG64's clang and llvm-ar, whatever
  # compiler the environment names. A user's CC named Visual Studio's cl.exe:
  # cc-rs gave SDL3's CMake build (sdl3-sys, through the cmake crate) MSVC's C
  # flags, and CMake, which reads CC itself, took cl.exe, whose test compile
  # failed. cc-rs takes CC_<target> before HOST_CC and CC; CMake, QEMU's
  # configure and meson read CC, CXX and AR; the cmake crate takes
  # CMAKE_GENERATOR_<target> before CMAKE_GENERATOR, and chooses MSYS Makefiles
  # itself where neither is set. CLANG64's ar is llvm-ar.
  export CC=clang CXX=clang++ AR=llvm-ar CMAKE_GENERATOR="MSYS Makefiles"
  export CC_x86_64_pc_windows_gnullvm=clang CXX_x86_64_pc_windows_gnullvm=clang++ \
    AR_x86_64_pc_windows_gnullvm=llvm-ar CMAKE_GENERATOR_x86_64_pc_windows_gnullvm="MSYS Makefiles"
fi

# ------------------------------------------------------------------ sources

# vendor/qemu at the commit this checkout names: shallow, else whole (a server
# that refuses a shallow fetch of a commit by its hash).
fetch_qemu() {
  git -C "$ROOT" submodule update --init --depth 1 vendor/qemu ||
    git -C "$ROOT" submodule update --init vendor/qemu ||
    die "could not fetch vendor/qemu's commit $(git -C "$ROOT" ls-tree HEAD vendor/qemu | awk '{print $3}') from $(git -C "$ROOT" config -f .gitmodules --get submodule.vendor/qemu.url): check that you can read that repository; a commit it does not have was not pushed to it"
}

if [[ $BUILDS_QEMU == 1 && $SKIP_QEMU == 0 ]]; then
  # A linked worktree shares the main checkout's module repository: initialising
  # vendor/qemu here would rewrite that repository's core.worktree for every
  # worktree (docs/lessons.md).
  gitdir="$(git -C "$ROOT" rev-parse --absolute-git-dir)"
  commondir="$(cd "$ROOT" && cd "$(git rev-parse --git-common-dir)" && pwd)"
  if [[ ! -f "$ROOT/vendor/qemu/configure" ]]; then
    if [[ "$gitdir" != "$commondir" ]]; then
      die "this is a git worktree, and vendor/qemu is empty: run $SELF in the main checkout, or clone vendor/qemu into this worktree yourself"
    fi
    say "Fetching QEMU's source (vendor/qemu, without its own submodules)"
    fetch_qemu
  elif [[ $UPDATE == 1 && "$gitdir" == "$commondir" ]]; then
    fetch_qemu
  fi
fi

# ------------------------------------------------------------------ the library and QEMU

# The files running processes execute, their links resolved, a line each:
# every process's /proc/PID/exe. A QEMU's executable names the build it started
# from, where its command line, and ps, show only qemu/current, through which
# reims-ctl runs QEMU, and which a new build moves.
qemu_executables() {
  if [[ ${WINDOWS:-0} == 1 ]]; then
    # Windows' own list, as MSYS2 paths: there is no /proc/PID/exe for a Windows program.
    powershell -NoProfile -Command '(Get-CimInstance Win32_Process).ExecutablePath' 2>/dev/null |
      tr -d '\r' | grep . | cygpath -u -f - 2>/dev/null || true
    return 0
  fi
  readlink /proc/[0-9]*/exe 2>/dev/null || true
}

# prune_qemu DIR: removes the QEMU builds in DIR but the three newest and those
# a running process executes a file of.
prune_qemu() {
  local real in_use old
  real="$(cd "$1" && pwd -P)"
  in_use="$(qemu_executables)"
  # Windows names a running QEMU's executable by the path it started from,
  # qemu/current's, not the build behind it: while one runs from DIR, none goes.
  if [[ ${WINDOWS:-0} == 1 ]] && grep -qiF "$real/" <<<"$in_use"; then
    return 0
  fi
  ls -1d "$1/"[0-9]*/ 2>/dev/null | sort -r | tail -n +4 | while read -r old; do
    old="${old%/}"
    grep -qF "$real/${old##*/}/" <<<"$in_use" || rm -rf "$old"
  done
}

# kept_qemu DIR: why the QEMU build DIR, which --skip-qemu keeps, is not the
# one this checkout builds, a line each. DIR's name ends in the commit it was
# built at. The build is QEMU's source and the library and option ROM it links
# (cargo xtask dist): all of the tree's crates but reims-ctl, which is its own
# workspace.
kept_qemu() {
  local built="${1##*/}"
  built="${built#*-*-}"
  if ! git -C "$ROOT" cat-file -e "$built^{commit}" 2>/dev/null; then
    echo "the installed QEMU was built at $built, which this checkout does not have: run this without --skip-qemu to build this checkout's"
  elif ! git -C "$ROOT" diff --quiet "$built" HEAD -- vendor/qemu crates firmware xtask \
    Cargo.toml Cargo.lock rust-toolchain.toml .cargo ':!crates/reims-ctl'; then
    echo "the installed QEMU was built at $built, and QEMU or the library it links changed since: run this without --skip-qemu to build this checkout's"
  fi
}

COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
DIST="$TARGET/dist-vulkan"
# Where QEMU builds and is staged: the cache directory, else, where its path
# holds a space (a Windows user's folder: "paths" above), the target directory.
QEMU_BUILDS="$CACHE"
[[ $WINDOWS == 1 && $CACHE == *[[:space:]]* ]] && QEMU_BUILDS="$TARGET"

if [[ $BUILDS_QEMU == 1 && $SKIP_QEMU == 0 ]]; then
  say "Building the library (cargo xtask dist --platform vulkan)"
  (cd "$ROOT" && CARGO_TARGET_DIR="$TARGET" cargo xtask dist --platform vulkan)
  [[ -f "$DIST/lib/pkgconfig/reims-vgpu.pc" ]] || die "cargo xtask dist wrote no $(shown "$DIST/lib/pkgconfig/reims-vgpu.pc")"

  BUILD="$QEMU_BUILDS/qemu-build-$PLATFORM"
  PC="$DIST/lib/pkgconfig"
  # QEMU's install prefix: never used where it is, since QEMU finds its files
  # relative to its binary; the build is staged with DESTDIR and moved.
  PREFIX=/reims-ctl-qemu
  # QEMU of another commit builds in an empty directory. meson, configuring
  # or regenerating in place, deletes each file the directory's last build made
  # that the new build.ninja does not name (ninja -t cleandead), even one it
  # has just written at configure: QEMU of 2026-11-13 writes qemu-version.h
  # there where its source has no version tag, as this shallow fetch's has
  # none, where the QEMU before made it in the build, and the build then
  # stopped on the missing header (docs/lessons.md).
  QEMU_SOURCE=$(git -C "$ROOT/vendor/qemu" rev-parse HEAD) || die "could not read vendor/qemu's commit"
  [[ "$(cat "$BUILD/qemu-source" 2>/dev/null)" == "$QEMU_SOURCE" ]] || rm -rf "$BUILD"
  mkdir -p "$BUILD"
  # No docs and no test suite: users run QEMU, not its tests.
  CONFIGURE=("$ROOT/vendor/qemu/configure" --prefix="$PREFIX" "${QEMU_CONFIGURE[@]}"
    --enable-reims-vgpu --disable-docs --disable-tests --disable-werror)
  CONFIGURED="PKG_CONFIG_PATH=$PC ${CONFIGURE[*]}"
  # Configure again when the source, the options, the package's directory, or
  # the package itself changed: meson keeps the package's flags from configure
  # (docs/lessons.md); and when ninja's dry run fails, as it does once a system
  # update removed a library file build.ninja names (/usr/lib/libfdt.so.1.8.1).
  if [[ ! -f "$BUILD/build.ninja" ]] ||
    [[ "$(cat "$BUILD/configured-with" 2>/dev/null)" != "$CONFIGURED" ]] ||
    ! cmp -s "$PC/reims-vgpu.pc" "$BUILD/reims-vgpu.pc.configured" ||
    ! ninja -C "$BUILD" -n >/dev/null 2>&1; then
    say "Configuring QEMU"
    (cd "$BUILD" && PKG_CONFIG_PATH="$PC" "${CONFIGURE[@]}")
    cp "$PC/reims-vgpu.pc" "$BUILD/reims-vgpu.pc.configured"
    printf '%s\n' "$CONFIGURED" >"$BUILD/configured-with"
    printf '%s\n' "$QEMU_SOURCE" >"$BUILD/qemu-source"
  fi
  say "Building QEMU"
  # A build that stops configures again on the next run: when ninja has meson
  # regenerate build.ninja, meson reuses the dependencies it found before, a
  # removed one too, which the dry run above does not catch.
  ninja -C "$BUILD" || { rm -f "$BUILD/configured-with"; die "QEMU's build stopped; the next run configures it again"; }
  STAGE="$QEMU_BUILDS/qemu-stage"
  rm -rf "$STAGE"
  if [[ $WINDOWS == 1 ]]; then
    # QEMU's meson, a Windows program, gets the prefix as MSYS2 hands it a path,
    # under MSYS2's own folder (C:/msys64/reims-ctl-qemu), which the stage holds
    # without its drive.
    DESTDIR="$(cygpath -w "$STAGE")" ninja -C "$BUILD" install >/dev/null
    staged="$(find "$STAGE" -path "*/bin/$QEMU_BINARY" -print -quit)"
    [[ -n $staged ]] || die "QEMU's install has no bin/$QEMU_BINARY"
    PREFIX="${staged%/bin/*}" PREFIX="${PREFIX#"$STAGE"}"
    # The CLANG64 DLLs QEMU and qemu-img load, copied beside them: reims-ctl
    # starts QEMU outside MSYS2, and an update of MSYS2's packages then changes
    # no build a VM runs. libva's and its vaon12 driver too, with what they
    # load, which the library opens by name when the device's video starts
    # (so ldd lists none of them): libva looks for the driver beside itself.
    {
      for exe in "$QEMU_BINARY" qemu-img.exe; do
        ldd "$STAGE$PREFIX/bin/$exe" | awk '$3 ~ "^/clang64/" { print $3 }'
      done
      for dll in libva libva_win32 vaon12_drv_video; do
        echo "/clang64/bin/$dll.dll"
        ldd "/clang64/bin/$dll.dll" | awk '$3 ~ "^/clang64/" { print $3 }'
      done
    } | sort -u | while read -r dll; do cp "$dll" "$STAGE$PREFIX/bin/"; done
  else
    DESTDIR="$STAGE" ninja -C "$BUILD" install >/dev/null
  fi
  [[ -x "$STAGE$PREFIX/bin/$QEMU_BINARY" ]] || die "QEMU's install has no bin/$QEMU_BINARY"
  "$STAGE$PREFIX/bin/$QEMU_BINARY" -device help 2>/dev/null | grep -q '"reims-graphics-pci"' || die "QEMU's install has no reims-graphics-pci: was it configured with reims-vgpu?"
  ID="$(date +%Y%m%d-%H%M%S)-$COMMIT"
  mkdir -p "$DATA/qemu"
  if [[ $WINDOWS == 1 ]]; then
    # Windows refuses to move a folder while a file in it is open, and just
    # after the stage was written something held one: the move failed with
    # "Permission denied", and the same move succeeded minutes later (measured
    # on the mini, m3-max-36159 and m3-max-36188). Tried for a minute.
    for _ in $(seq 60); do mv "$STAGE$PREFIX" "$DATA/qemu/$ID" 2>/dev/null && break; sleep 1; done
    [[ -d $DATA/qemu/$ID ]] || mv "$STAGE$PREFIX" "$DATA/qemu/$ID"
  else
    mv "$STAGE$PREFIX" "$DATA/qemu/$ID"
  fi
  rm -rf "$STAGE"
  if [[ $WINDOWS == 1 ]]; then
    # A junction, which needs no administrator, where MSYS2's ln -s copies.
    current="$(cygpath -w "$DATA/qemu/current")"
    [[ -e "$DATA/qemu/current" || -L "$DATA/qemu/current" ]] && cmd //c rmdir "$current"
    cmd //c mklink //J "$current" "$(cygpath -w "$DATA/qemu/$ID")" >/dev/null
  else
    ln -sfn "$ID" "$DATA/qemu/current"
  fi
  prune_qemu "$DATA/qemu"
  QEMU_DIR="$DATA/qemu/$ID"
elif [[ $BUILDS_QEMU == 1 ]]; then
  QEMU_DIR="$(cd "$DATA/qemu/current" && pwd -P)"
  kept_qemu "$QEMU_DIR" | while read -r why; do warn "$why"; done
else
  QEMU_DIR=""
fi

# ------------------------------------------------------------------ reims-ctl

say "Installing reims-ctl (cargo install)"
(cd "$ROOT" && CARGO_TARGET_DIR="$CACHE/ctl-target" cargo install --locked --force --quiet --path crates/reims-ctl)
BIN="${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin/reims-ctl"
if [[ $WINDOWS == 1 ]]; then
  BIN="$(cygpath -u "${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$USER_HOME/.cargo}}")/bin/reims-ctl.exe"
fi
[[ -x "$BIN" ]] || die "cargo install left no $(shown "$BIN")"
if [[ $WINDOWS == 1 ]]; then
  # reims-ctl, a Windows program, takes Windows paths; its update runs this
  # script in this bash.
  "$BIN" __record-install --repo "$(cygpath -w "$ROOT")" ${QEMU_DIR:+--qemu "$(cygpath -w "$QEMU_DIR")"} \
    --commit "$COMMIT" --shell "$(cygpath -w "$BASH")"
else
  "$BIN" __record-install --repo "$ROOT" ${QEMU_DIR:+--qemu "$QEMU_DIR"} --commit "$COMMIT"
fi

echo
printf '%sreims-ctl is installed.%s\n' "$green$bold" "$off"
if [[ $BUILDS_QEMU == 1 ]]; then
  echo "  reims-ctl            the interactive interface: create, start and stop VMs"
else
  echo "  reims-ctl            the interactive interface: make arm64 VMs from IPSWs"
fi
echo "  reims-ctl help       the same from a shell"
[[ -z $QEMU_DIR ]] || echo "  QEMU                 $(shown "$QEMU_DIR")"
echo
if [[ $WINDOWS == 1 ]]; then
  # rustup put cargo's bin on the user's PATH, which a new window has.
  echo "Check this host, in a new PowerShell or Windows Terminal window:"
  echo "  reims-ctl doctor"
  exit 0
fi
# By its full path, which works before PATH finds reims-ctl.
echo "Check this host:"
printf '  %q doctor\n' "$BIN"
if [[ "$(PATH="$START_PATH" command -v reims-ctl 2>/dev/null)" != "$BIN" ]]; then
  echo
  warn "$(dirname "$BIN") is not first on your PATH: open a new terminal, or run: . \"\$HOME/.cargo/env\""
fi
