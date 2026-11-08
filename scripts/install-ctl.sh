#!/usr/bin/env bash
#
# Builds and installs reims-ctl, and the QEMU it runs, with apple-gfx on this
# checkout's reims-vgpu.
#
#   scripts/install-ctl.sh [--yes] [--update] [--skip-qemu]
#
# On macOS, GPU_ARM picks what VMs' graphics run on:
#
#   GPU_ARM=metal scripts/install-ctl.sh    Metal (the default)
#   GPU_ARM=vulkan scripts/install-ctl.sh   Vulkan, on Homebrew's MoltenVK
#
# Run it again with the other value to switch; switching back to one built
# before rebuilds only what changed since. Without GPU_ARM it keeps the arm
# installed now (a Vulkan install's QEMU directory ends in -vulkan), so
# reims-ctl update never switches it. Linux has Vulkan alone.
#
# macOS on arm64 and Linux on x86-64 (Ubuntu or Debian, Arch or CachyOS; other
# distributions get the list of what to install). It:
#   1. checks the build's dependencies, and offers to install those missing
#      (Homebrew on macOS; apt or pacman on Linux, through sudo), and rustup
#      through rustup.rs, beside any other cargo, whose installer asks its own
#      questions;
#   2. fetches vendor/qemu, QEMU's source, without QEMU's own submodules (its
#      build fetches what it needs); cargo fetches metal2vulkan itself, at the
#      commit the library's Cargo.toml names;
#   3. builds the library (cargo xtask dist) on GPU_ARM's platform and QEMU on
#      it, in ~/.cache/reims-ctl, and installs QEMU into
#      ~/.local/share/reims-ctl/qemu/<build>, linking qemu/current to it. A
#      build never replaces another: a running VM keeps the binary it started
#      with;
#   4. installs reims-ctl with cargo install (into ~/.cargo/bin, or
#      CARGO_INSTALL_ROOT's bin), signed on macOS with the virtualization
#      entitlement, which installing macOS from an IPSW needs;
#   5. installs reims-util beside it, which makes a new x86-64 VM's OpenCore
#      image and downloads its macOS installer from Apple: on Linux, and on any
#      host where one is there already. reims-ctl does (__record-install), at
#      the commit it names (crates/reims-ctl/src/reims_util.rs), replacing one
#      from another commit and leaving one at that commit as it is.
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
# Environment: GPU_ARM (above); CARGO_TARGET_DIR, the library's build (default
# the checkout's target/); XDG_DATA_HOME and XDG_CACHE_HOME move the install
# and the build from ~/.local/share and ~/.cache, where they are on macOS too.
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
case "$OS/$ARCH" in
  Darwin/arm64)
    PLATFORM=macos-arm64
    # gnutls gives QEMU its ciphers. Without a crypto library QEMU has no AES,
    # which vmapple's AES engine needs (a macOS guest uses it as it mounts its
    # root volume), and refuses to start a VM. It is enabled by name, so that
    # configure stops where it is missing.
    QEMU_CONFIGURE=(--target-list=aarch64-softmmu --enable-hvf --enable-gnutls --disable-sdl
      --disable-gtk)
    QEMU_BINARY=qemu-system-aarch64
    ;;
  Linux/x86_64)
    PLATFORM=linux-x86_64
    # libusb (usb-host, USB passthrough) and SPICE (-spice, a display a SPICE
    # client connects to) are enabled by name: configure leaves out an
    # optional feature whose library it does not find, saying nothing.
    QEMU_CONFIGURE=(--target-list=x86_64-softmmu --enable-kvm --disable-sdl --enable-gtk --enable-slirp
      --enable-libusb --enable-spice)
    QEMU_BINARY=qemu-system-x86_64
    ;;
  *) die "reims-ctl supports macOS on arm64 and Linux on x86-64; this is $OS on $ARCH" ;;
esac

# The XDG directory VAR names, else ~/FALLBACK: unset on macOS, and ignored
# unless absolute (as reims-ctl does).
xdg() { # VAR FALLBACK
  local v="${!1:-}"
  if [[ $v == /* ]]; then printf '%s' "$v"; else printf '%s' "$HOME/$2"; fi
}
DATA="$(xdg XDG_DATA_HOME .local/share)/reims-ctl"
CACHE="$(xdg XDG_CACHE_HOME .cache)/reims-ctl"
mkdir -p "$DATA" "$CACHE"

# The GPU arm (header). ARM_SUFFIX names a macOS build on Vulkan, its QEMU
# build directory and its install, apart from the Metal one.
case "$OS/${GPU_ARM:-}" in
  Darwin/)
    if [[ "$(readlink "$DATA/qemu/current" 2>/dev/null)" == *-vulkan ]]; then GPU_ARM=vulkan; else GPU_ARM=metal; fi
    ;;
  Linux/) GPU_ARM=vulkan ;;
  Darwin/metal | Darwin/vulkan | Linux/vulkan) ;;
  Linux/metal) die "GPU_ARM=metal is macOS's: on Linux the library runs on Vulkan alone" ;;
  *) die "GPU_ARM is metal or vulkan, not $GPU_ARM" ;;
esac
ARM_SUFFIX=""
[[ $OS == Darwin && $GPU_ARM == vulkan ]] && ARM_SUFFIX=-vulkan

# Every VM runs on QEMU: --skip-qemu keeps the installed one, and needs one.
if [[ $SKIP_QEMU == 1 && ! -x "$DATA/qemu/current/bin/$QEMU_BINARY" ]]; then
  die "--skip-qemu keeps the installed QEMU, and none is installed ($DATA/qemu/current/bin/$QEMU_BINARY is missing): run this without --skip-qemu, which builds it"
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
  say "Pulling $ROOT"
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
      git -C "$ROOT" merge -q --ff-only "$upref" || die "the checkout's local changes stop the update: commit or set them aside, then run $0"
    elif published "$before"; then
      # The checkout's commit is kept on a branch of its own, made once the
      # move has worked; --keep refuses to move over a local change the move
      # would overwrite.
      old=$(git -C "$ROOT" rev-parse HEAD)
      git -C "$ROOT" reset -q --keep "$upref" || die "the checkout's local changes stop the update: commit or set them aside, then run $0"
      backup="$branch-before-update-$(date +%Y%m%d-%H%M%S)"
      git -C "$ROOT" branch "$backup" "$old"
      warn "${upref#refs/remotes/} was force-pushed: the checkout's earlier commit stays on the branch $backup"
    else
      die "the checkout has commits of its own that ${upref#refs/remotes/} lacks: update it yourself, then run $0"
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

# ------------------------------------------------------------------ dependencies

missing=()

if [[ $OS == Darwin ]]; then
  xcode-select -p >/dev/null 2>&1 || die "Xcode's command line tools are missing: run xcode-select --install, then this again"
  command -v brew >/dev/null 2>&1 || die "Homebrew is missing (https://brew.sh): install it, then run this again"
  formulas=(ninja pkgconf glib pixman libslirp lld gnutls)
  # The Vulkan arm's loader and driver, which metal2vulkan opens from
  # Homebrew's lib.
  [[ $GPU_ARM == vulkan ]] && formulas+=(vulkan-loader molten-vk)
  for f in "${formulas[@]}"; do
    brew list --formula "$f" >/dev/null 2>&1 || missing+=("$f")
  done
  if ((${#missing[@]})); then
    say "Missing Homebrew packages: ${missing[*]}"
    ask "Install them with brew?" || die "install them (brew install ${missing[*]}), then run this again"
    brew install "${missing[@]}"
  fi
else
  distro_id="" distro_like=""
  if [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    distro_id="$(. /etc/os-release && echo "${ID:-}")"
    # shellcheck disable=SC1091
    distro_like="$(. /etc/os-release && echo "${ID_LIKE:-}")"
  fi
  # The library (cargo xtask dist: clang, lld and binutils for the option ROM;
  # CMake and the X11, Wayland, xkbcommon and EGL headers for SDL3's video,
  # which it builds from source), and QEMU (glib, pixman, libslirp; GTK 3 for
  # its own window, which a VM on VMware SVGA shows its display in; pipewire,
  # pulse and ALSA for its audio backends; libusb for USB passthrough; the
  # SPICE server for -spice; bzip2 unpacks its firmware). A new x86-64 VM's
  # OVMF comes from reims-util, with its OpenCore image.
  apt_pkgs=(build-essential git curl ca-certificates pkg-config python3 python3-venv ninja-build
    flex bison bzip2 cmake clang lld binutils libglib2.0-dev libpixman-1-dev libslirp-dev
    zlib1g-dev libfdt-dev libpipewire-0.3-dev libpulse-dev libasound2-dev libusb-1.0-0-dev
    libspice-server-dev libvulkan1
    libx11-dev libxext-dev libxrandr-dev libxcursor-dev libxi-dev libxfixes-dev libxss-dev
    libxtst-dev libxkbcommon-dev libwayland-dev libwayland-bin libdecor-0-dev
    libegl-dev libgl-dev libdrm-dev libgbm-dev libdbus-1-dev libudev-dev libgtk-3-dev)
  pacman_pkgs=(base-devel git curl python ninja pkgconf flex bison bzip2 cmake clang lld
    glib2 pixman libslirp zlib dtc pipewire libpulse alsa-lib libusb spice vulkan-icd-loader
    libx11 libxext libxrandr libxcursor libxi libxfixes libxss libxtst libxkbcommon
    wayland libdecor mesa libdrm dbus systemd-libs gtk3)
  case " $distro_id $distro_like " in
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
CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
if [[ -x "$CARGO_BIN/rustup" ]]; then
  export PATH="$CARGO_BIN:$PATH"
fi
if ! command -v rustup >/dev/null 2>&1; then
  if command -v cargo >/dev/null 2>&1; then
    say "$(command -v cargo) is a cargo without rustup: the build needs the toolchain rust-toolchain.toml pins, which rustup installs beside it"
  else
    say "Rust is not installed"
  fi
  ask "Install rustup from https://rustup.rs?" || die "install rustup (https://rustup.rs), then run this again"
  if [[ $YES == 1 ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  fi
  export PATH="$CARGO_BIN:$PATH"
fi

# ------------------------------------------------------------------ sources

# vendor/qemu at the commit this checkout names: shallow, else whole (a server
# that refuses a shallow fetch of a commit by its hash).
fetch_qemu() {
  git -C "$ROOT" submodule update --init --depth 1 vendor/qemu ||
    git -C "$ROOT" submodule update --init vendor/qemu ||
    die "could not fetch vendor/qemu's commit $(git -C "$ROOT" ls-tree HEAD vendor/qemu | awk '{print $3}') from $(git -C "$ROOT" config -f .gitmodules --get submodule.vendor/qemu.url): check that you can read that repository; a commit it does not have was not pushed to it"
}

if [[ $SKIP_QEMU == 0 ]]; then
  # A linked worktree shares the main checkout's module repository: initialising
  # vendor/qemu here would rewrite that repository's core.worktree for every
  # worktree (docs/lessons.md).
  gitdir="$(git -C "$ROOT" rev-parse --absolute-git-dir)"
  commondir="$(cd "$ROOT" && cd "$(git rev-parse --git-common-dir)" && pwd)"
  if [[ ! -f "$ROOT/vendor/qemu/configure" ]]; then
    if [[ "$gitdir" != "$commondir" ]]; then
      die "this is a git worktree, and vendor/qemu is empty: run $0 in the main checkout, or clone vendor/qemu into this worktree yourself"
    fi
    say "Fetching QEMU's source (vendor/qemu, without its own submodules)"
    fetch_qemu
  elif [[ $UPDATE == 1 && "$gitdir" == "$commondir" ]]; then
    fetch_qemu
  fi
fi

# ------------------------------------------------------------------ the library and QEMU

# The files running processes execute, their links resolved, a line each: on
# Linux every process's /proc/PID/exe; on macOS the txt files lsof lists of
# those whose name starts with qemu (QEMU, qemu-img). A QEMU's executable names
# the build it started from, where its command line, and ps, show only
# qemu/current, through which reims-ctl runs QEMU, and which a new build moves.
qemu_executables() {
  if [[ -d /proc/self ]]; then
    readlink /proc/[0-9]*/exe 2>/dev/null || true
  else
    lsof -nP -a -c qemu -d txt -Fn 2>/dev/null | sed -n 's/^n//p' || true
  fi
}

# prune_qemu DIR: removes the QEMU builds in DIR but the three newest and those
# a running process executes a file of.
prune_qemu() {
  local real in_use old
  real="$(cd "$1" && pwd -P)"
  in_use="$(qemu_executables)"
  ls -1d "$1/"[0-9]*/ 2>/dev/null | sort -r | tail -n +4 | while read -r old; do
    old="${old%/}"
    grep -qF "$real/${old##*/}/" <<<"$in_use" || rm -rf "$old"
  done
}

# kept_qemu DIR: why the QEMU build DIR, which --skip-qemu keeps, is not the
# one this checkout builds, a line each. DIR's name ends in the commit it was
# built at, then -vulkan for a macOS build on Vulkan. The build is QEMU's
# source and the library and option ROM it links (cargo xtask dist): all of
# the tree's crates but reims-ctl, which is its own workspace.
kept_qemu() {
  local built="${1##*/}" suffix="" arm=Metal
  built="${built#*-*-}"
  [[ $built == *-vulkan ]] && suffix=-vulkan arm=Vulkan
  built="${built%-vulkan}"
  if [[ $suffix != "$ARM_SUFFIX" ]]; then
    echo "the installed QEMU runs on $arm, not GPU_ARM=$GPU_ARM: run this without --skip-qemu to build it on $GPU_ARM"
  fi
  if ! git -C "$ROOT" cat-file -e "$built^{commit}" 2>/dev/null; then
    echo "the installed QEMU was built at $built, which this checkout does not have: run this without --skip-qemu to build this checkout's"
  elif ! git -C "$ROOT" diff --quiet "$built" HEAD -- vendor/qemu crates firmware xtask \
    Cargo.toml Cargo.lock rust-toolchain.toml .cargo ':!crates/reims-ctl'; then
    echo "the installed QEMU was built at $built, and QEMU or the library it links changed since: run this without --skip-qemu to build this checkout's"
  fi
}

COMMIT="$(git -C "$ROOT" rev-parse --short HEAD)"
TARGET="${CARGO_TARGET_DIR:-$ROOT/target}"
if [[ $GPU_ARM == metal ]]; then DIST="$TARGET/dist"; else DIST="$TARGET/dist-vulkan"; fi

if [[ $SKIP_QEMU == 0 ]]; then
  say "Building the library on $GPU_ARM (cargo xtask dist --platform $GPU_ARM)"
  (cd "$ROOT" && CARGO_TARGET_DIR="$TARGET" cargo xtask dist --platform "$GPU_ARM")
  [[ -f "$DIST/lib/pkgconfig/reims-vgpu.pc" ]] || die "cargo xtask dist wrote no $DIST/lib/pkgconfig/reims-vgpu.pc"

  BUILD="$CACHE/qemu-build-$PLATFORM$ARM_SUFFIX"
  PC="$DIST/lib/pkgconfig"
  # QEMU's install prefix: never used where it is, since QEMU finds its files
  # relative to its binary; the build is staged with DESTDIR and moved.
  PREFIX=/reims-ctl-qemu
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
  fi
  say "Building QEMU"
  # A build that stops configures again on the next run: when ninja has meson
  # regenerate build.ninja, meson reuses the dependencies it found before, a
  # removed one too, which the dry run above does not catch.
  ninja -C "$BUILD" || { rm -f "$BUILD/configured-with"; die "QEMU's build stopped; the next run configures it again"; }
  STAGE="$CACHE/qemu-stage"
  rm -rf "$STAGE"
  DESTDIR="$STAGE" ninja -C "$BUILD" install >/dev/null
  [[ -x "$STAGE$PREFIX/bin/$QEMU_BINARY" ]] || die "QEMU's install has no bin/$QEMU_BINARY"
  "$STAGE$PREFIX/bin/$QEMU_BINARY" -device help 2>/dev/null | grep -q '"reims-graphics-pci"' || die "QEMU's install has no reims-graphics-pci: was it configured with reims-vgpu?"
  if [[ $OS == Darwin ]] && ! codesign -d --entitlements - "$STAGE$PREFIX/bin/$QEMU_BINARY" 2>/dev/null | grep -q com.apple.security.hypervisor; then
    codesign -s - -f --entitlements "$ROOT/vendor/qemu/accel/hvf/entitlements.plist" "$STAGE$PREFIX/bin/$QEMU_BINARY"
  fi
  ID="$(date +%Y%m%d-%H%M%S)-$COMMIT$ARM_SUFFIX"
  mkdir -p "$DATA/qemu"
  mv "$STAGE$PREFIX" "$DATA/qemu/$ID"
  rm -rf "$STAGE"
  ln -sfn "$ID" "$DATA/qemu/current"
  prune_qemu "$DATA/qemu"
  QEMU_DIR="$DATA/qemu/$ID"
else
  QEMU_DIR="$(cd "$DATA/qemu/current" && pwd -P)"
  kept_qemu "$QEMU_DIR" | while read -r why; do warn "$why"; done
fi

# ------------------------------------------------------------------ reims-ctl

say "Installing reims-ctl (cargo install)"
(cd "$ROOT" && CARGO_TARGET_DIR="$CACHE/ctl-target" cargo install --locked --force --quiet --path crates/reims-ctl)
BIN="${CARGO_INSTALL_ROOT:-${CARGO_HOME:-$HOME/.cargo}}/bin/reims-ctl"
[[ -x "$BIN" ]] || die "cargo install left no $BIN"
if [[ $OS == Darwin ]]; then
  ent="$CACHE/reims-ctl.entitlements"
  cat >"$ent" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>com.apple.security.virtualization</key>
	<true/>
</dict>
</plist>
EOF
  codesign -s - -f --entitlements "$ent" "$BIN" 2>/dev/null
fi
"$BIN" __record-install --repo "$ROOT" --qemu "$QEMU_DIR" --commit "$COMMIT"

echo
printf '%sreims-ctl is installed.%s\n' "$green$bold" "$off"
echo "  reims-ctl            the interactive interface: create, start and stop VMs"
echo "  reims-ctl help       the same from a shell"
echo "  QEMU                 $QEMU_DIR"
echo
# By its full path, which works before PATH finds reims-ctl.
echo "Check this host:"
printf '  %q doctor\n' "$BIN"
if [[ "$(PATH="$START_PATH" command -v reims-ctl 2>/dev/null)" != "$BIN" ]]; then
  echo
  warn "$(dirname "$BIN") is not first on your PATH: open a new terminal, or run: . \"\$HOME/.cargo/env\""
fi
