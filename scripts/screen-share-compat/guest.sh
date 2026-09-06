#!/bin/bash
# Run inside the guest as the desktop user, after SSH authorization.
set -euo pipefail

mode=${1:---apply}
case "$mode" in --apply|--undo) ;; *) echo "screen-share-compat: use --apply or --undo" >&2; exit 2 ;; esac
state_dir="$HOME/Library/Application Support/reims-vgpu/screen-share-compat"
state="$state_dir/previous-values"
fail() { echo "screen-share-compat: $*" >&2; exit 1; }

read_bool() {
  local domain=$1 key=$2 value kind
  if ! value=$(defaults read "$domain" "$key" 2>/dev/null); then
    echo absent
  else
    kind=$(defaults read-type "$domain" "$key")
    [ "$kind" = 'Type is boolean' ] \
      || fail "non-boolean preference $domain/$key; refusing to replace it"
    case "$value" in
      0|1) echo "$value" ;;
      *) fail "unexpected boolean value for $domain/$key" ;;
    esac
  fi
}

restore_bool() {
  local domain=$1 key=$2 value=$3
  case "$value" in
    absent) if defaults read "$domain" "$key" >/dev/null 2>&1; then defaults delete "$domain" "$key"; fi ;;
    0|1) defaults write "$domain" "$key" -bool "$value" ;;
    *) fail "invalid saved preference state" ;;
  esac
}

restart_agent() {
  local service="gui/$(id -u)/com.apple.screensharing.menuextra"
  if launchctl print "$service" >/dev/null 2>&1; then
    launchctl kickstart -k "$service" || fail "preferences saved, but could not restart SSMenuAgent"
  fi
}

if [ "$mode" = --undo ]; then
  [ -f "$state" ] || { echo "screen-share-compat: nothing to undo"; exit 0; }
  read -r old_gate old_archive < "$state"
  case "$old_gate" in absent|0|1) ;; *) fail "invalid saved global preference" ;; esac
  case "$old_archive" in absent|0|1) ;; *) fail "invalid saved archive preference" ;; esac
  restore_bool com.apple.SSMenuAgent RBUseBinaryArchive "$old_archive"
  restore_bool -g RBAllowDebuggingDefaults "$old_gate"
  restart_agent
  rm "$state"
  echo "screen-share-compat: restored previous preferences"
  exit 0
fi

# Only this guest pathway's RenderBox preference contract has been recovered.
[ "$(uname -m)" = x86_64 ] || { echo "screen-share-compat: other architecture; unchanged"; exit 0; }
case "$(sw_vers -productVersion)" in 15.*) ;; *) echo "screen-share-compat: other OS rail; unchanged"; exit 0 ;; esac
scratch=$(mktemp -d /tmp/reims-screen-share-compat.XXXXXX)
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/probe.m" <<'OBJC'
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/message.h>
int main(void) { @autoreleasepool {
    id device = MTLCreateSystemDefaultDevice();
    SEL supports = NSSelectorFromString(@"supportsBinaryArchives");
    SEL architecture = NSSelectorFromString(@"targetDeviceArchitecture");
    if (!device || ![device respondsToSelector:supports] ||
        ![device respondsToSelector:architecture]) {
        fprintf(stderr, "screen-share-compat: cannot query archive capability\n");
        return 2;
    }
    BOOL supported = ((BOOL (*)(id, SEL))objc_msgSend)(device, supports);
    id target = ((id (*)(id, SEL))objc_msgSend)(device, architecture);
    puts(!supported && !target ? "archive-unavailable" : "unchanged");
    return 0;
}}
OBJC
clang -framework Foundation -framework Metal "$scratch/probe.m" -o "$scratch/probe" \
  || fail "archive capability probe needs the guest's Command Line Tools"
capability=$("$scratch/probe") || fail "archive capability probe failed"
[ "$capability" = archive-unavailable ] || { echo "screen-share-compat: device does not need this bypass"; exit 0; }

# An environment override wins over defaults. Do not report success while the
# launch service supplies a different source for the same decision.
service="gui/$(id -u)/com.apple.screensharing.menuextra"
service_info=$(launchctl print "$service" 2>/dev/null || true)
if printf '%s\n' "$service_info" | grep -q 'RB_USE_BINARY_ARCHIVE =>'; then
  fail "SSMenuAgent has an RB_USE_BINARY_ARCHIVE environment override; remove it before applying preferences"
fi

# Save only the two keys we own, once. Reapplying must not replace the user's
# original values with the bypass values. Do not export/restore entire domains.
mkdir -p "$state_dir"
if [ ! -f "$state" ]; then
  old_gate=$(read_bool -g RBAllowDebuggingDefaults)
  old_archive=$(read_bool com.apple.SSMenuAgent RBUseBinaryArchive)
  printf '%s %s\n' "$old_gate" "$old_archive" > "$state.tmp"
  mv "$state.tmp" "$state"
fi

# RenderBox checks this user-global opt-in before consulting per-app defaults.
# RBUseBinaryArchive is scoped to SSMenuAgent; other apps keep their own choice.
defaults write -g RBAllowDebuggingDefaults -bool true
defaults write com.apple.SSMenuAgent RBUseBinaryArchive -bool false
restart_agent
echo "screen-share-compat: SSMenuAgent will render without binary archives"
