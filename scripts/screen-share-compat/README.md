# Screen Sharing archive compatibility

On the macOS 15 x86/PCI pathway, SSMenuAgent's observed-screen indicator can
abort in Metal with `Target device architecture is nil`. The guest device
reports binary archives unsupported and supplies no target architecture;
changing the reims device-info AIR version cannot supply that object.

`guest.sh` configures RenderBox to render this agent without binary archives.
It runs inside the guest as the desktop user and is called by
`vm/guest-authorize.sh --screen-share`. Command Line Tools must already be
installed for its Metal capability probe, and macOS must have provisioned
SSMenuAgent's container by launching the agent at least once. Other architectures
and OS rails are left unchanged; on the supported rail, the bypass is selected only when
the device reports archives unsupported and its target architecture is nil.

```sh
# Apply to an already authorized, running guest.
ssh -o BatchMode=yes macos-vm 'bash -s -- --apply' < scripts/screen-share-compat/guest.sh

# Restore the previous values of both preferences.
ssh -o BatchMode=yes macos-vm 'bash -s -- --undo' < scripts/screen-share-compat/guest.sh
```

The recovered RenderBox configuration contract is:

- `RB_USE_BINARY_ARCHIVE` controls the archive path. Its preference spelling
  is `RBUseBinaryArchive`; environment settings take precedence.
- Preference lookup requires the user-global boolean `RBAllowDebuggingDefaults`.
- Once enabled, lookup checks the current application's domain, the global
  domain, then `com.apple.RenderBox`.
- SSMenuAgent has the App Sandbox entitlement. Its current-application
  preferences are in
  `~/Library/Containers/com.apple.SSMenuAgent/Data/Library/Preferences/com.apple.SSMenuAgent.plist`.
  An unqualified `defaults` domain from an SSH shell does not reliably select
  that container. A matching preference under `~/Library/Preferences` is
  insufficient, even when `defaults read com.apple.SSMenuAgent` returns false.
- False bypasses archive loading and attachment to render pipelines. Metal
  shader compilation and rendering remain enabled.

The helper enables the user-global preference opt-in and writes
`RBUseBinaryArchive=false` in SSMenuAgent's container, using the absolute
preference-domain path. The opt-in also allows other RenderBox consumers to
read any debugging preferences already
present for them; their archive preference is not changed. An inherited
`RB_USE_BINARY_ARCHIVE` environment override takes precedence over this fix.

Original values are saved once under
`~/Library/Application Support/reims-vgpu/screen-share-compat/previous-values`.
Reapplying preserves that backup. Version 2 migrates the original two-value
backup, restores the previous non-container app preference, and saves the
original container preference alongside the original global opt-in value.
Undo accepts either backup version.

Both apply and undo send normal SIGTERM to the current user's SSMenuAgent
process. The system launch service starts it again on the next Screen Sharing
request. This can interrupt an active VNC connection. SIP may reject
`launchctl kickstart` or `launchctl kill` for this Apple service; no launchd
override is needed to terminate the user's process normally.
No system binary, launch plist, SIP setting, or GPU identity is changed.

The preferences live in the guest disk. A snapshot-reverting boot discards them;
run `--screen-share` again after such a boot, or include them in an explicitly
captured snapshot. This is a guest configuration workaround, not implementation
of binary archives in the paravirtualized driver.
