#!/bin/bash
# Keeps Setup Assistant's MiniBuddy panes ("Update Mac Automatically", "Sign in to Your Apple
# Account", "Software Update Complete") from coming up at the next login of a macOS guest. Run
# it in the guest, over SSH, as the user that logs in automatically, on a copy of the guest's
# disk (an overlay or a clone), before a boot's workload:
#
#   ssh -p PORT USER@127.0.0.1 'bash -s' < scripts/internal/guest-minibuddy-off.sh
#
# loginwindow launches MiniBuddy (Setup Assistant -MiniBuddyYes) at a login while the user's
# com.apple.loginwindow preferences for this host hold MiniBuddyLaunch = true; its log says so
# ("MiniBuddyLaunch pref is set, setting miniBuddyOption to kMinibuddyOptionMBPrefSet"). Setup
# Assistant sets the key, in the user's domain for any host and in its ByHost file for this
# host, while its panes are unseen, so a disk saved so brings them up at every boot of a copy.
# Deleted from the domain for any host alone, the key still launched MiniBuddy after a reboot.
#
# Where MiniBuddy runs, this ends it first, which ends the user's session (loginwindow shows its
# login window). Then it deletes the key for any host and for this host, and waits until neither
# preferences file holds it. Where MiniBuddy ran (exit 3), reboot the guest
# (sudo shutdown -r now) and wait for its SSH again: the automatic login comes up on the desktop.
#
# Prints what it did. Exits 0 where MiniBuddy was not running, 3 where it was, 1 where it could
# not be ended or the key deleted.
set -u
domain=com.apple.loginwindow
uuid=$(ioreg -rd1 -c IOPlatformExpertDevice | sed -n 's/.*"IOPlatformUUID" = "\(.*\)".*/\1/p')
files=("$HOME/Library/Preferences/$domain.plist" "$HOME/Library/Preferences/ByHost/$domain.$uuid.plist")
running() { pgrep -q -f -- '[S]etup Assistant.app/Contents/MacOS/Setup Assistant -MiniBuddyYes'; }
in_files() {
	local f
	for f in "${files[@]}"; do
		[ "$(plutil -extract MiniBuddyLaunch raw "$f" 2>/dev/null)" = true ] && return 0
	done
	return 1
}
ran=0
if running; then
	ran=1
	killall "Setup Assistant"
	for _ in $(seq 1 40); do running || break; sleep 0.25; done
	if running; then
		echo "minibuddy: Setup Assistant still runs after killall"
		exit 1
	fi
	echo "minibuddy: ended Setup Assistant, and with it the session"
fi
if in_files || [ "$(defaults read "$domain" MiniBuddyLaunch 2>/dev/null)" = 1 ] ||
	[ "$(defaults -currentHost read "$domain" MiniBuddyLaunch 2>/dev/null)" = 1 ]; then
	defaults delete "$domain" MiniBuddyLaunch 2>/dev/null
	defaults -currentHost delete "$domain" MiniBuddyLaunch 2>/dev/null
	for _ in $(seq 1 40); do in_files || break; sleep 0.25; done
	if in_files; then
		echo "minibuddy: MiniBuddyLaunch is still true in ${files[*]}"
		exit 1
	fi
	echo "minibuddy: deleted $domain MiniBuddyLaunch for any host and for this host ($uuid)"
else
	echo "minibuddy: $domain MiniBuddyLaunch is not set"
fi
if [ "$ran" = 1 ]; then
	echo "minibuddy: reboot the guest to log in without it"
	exit 3
fi
exit 0
