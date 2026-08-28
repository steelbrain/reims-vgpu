# vm/lib/usb-passthrough.sh — resolve host USB passthrough specs into QEMU
# `-device usb-host,...` properties. Sourced by vm/boot-x86.sh and
# vm/boot-arm64.sh; not executable on its own.
#
# WHY THIS IS A LIBRARY AND NOT A BLOCK IN EACH BOOT SCRIPT. The two boot
# scripts already carry hand-copied versions of several things, and a spec
# parser is the worst candidate for a fourth: three accepted spellings, each
# mapping to a different pair of QEMU properties, and a wrong mapping does not
# fail — it hands the guest a device the caller did not name, or nothing at all.
# One owner means the arm64 host cannot drift from the x86 one on a rule neither
# of them can test for the other.
#
# WHAT `usb-host` ACTUALLY NEEDS, which is the whole reason this file resolves
# anything instead of passing the spec straight through:
#
#   1. The device has to be there. QEMU's usb-host will realize against a
#      vendorid/productid that matches nothing and simply present no device to
#      the guest, which is indistinguishable from a driverless guest. A typo in
#      four hex digits therefore costs a whole boot and reads as a device bug,
#      so a spec matching nothing is refused here by name.
#   2. This process has to be able to open /dev/bus/usb/BBB/DDD read-write.
#      libusb needs write access to claim an interface; read access alone gets
#      far enough to enumerate and then fails at claim time, deep inside the
#      guest's own driver probe. The node is checked before QEMU starts.
#   3. The host driver is detached for the life of the boot. That is the point
#      of passthrough and it is also how somebody loses the keyboard they are
#      typing on, so every resolved device is announced with its strings.
#
# The sysfs walk is the contract here: /sys/bus/usb/devices holds one directory
# per device named BUS-PORT[.PORT...] (and `usbN` for a root hub), each carrying
# idVendor, idProduct, busnum and devnum. Those four files are what `lsusb`
# reads too, so the three spec forms below are all derived from one source
# rather than from parsing another tool's output.

USB_SYSFS_ROOT="${USB_SYSFS_ROOT:-/sys/bus/usb/devices}"

# Read a sysfs attribute, empty if absent. Keeps the callers below free of
# `2>/dev/null || true` on every field.
_usb_attr() {
  [ -r "$1/$2" ] && tr -d '\n' < "$1/$2" || true
}

# Every passthrough-eligible device directory, one per line. A root hub (`usbN`)
# is excluded: it is the controller, has no host driver to detach, and passing
# it is not a thing QEMU can do.
_usb_device_dirs() {
  local dir
  for dir in "$USB_SYSFS_ROOT"/*; do
    [ -d "$dir" ] || continue
    case "${dir##*/}" in
      usb*) continue ;;      # root hub
      *:*)  continue ;;      # an interface, e.g. 1-1:1.0
    esac
    [ -r "$dir/idVendor" ] && [ -r "$dir/busnum" ] && [ -r "$dir/devnum" ] || continue
    printf '%s\n' "$dir"
  done
}

# Human description: "Manufacturer Product" with either half allowed to be
# missing, falling back to the VID:PID that is always present.
_usb_describe() {
  local dir="$1" man prod
  man="$(_usb_attr "$dir" manufacturer)"
  prod="$(_usb_attr "$dir" product)"
  if [ -n "$man$prod" ]; then
    printf '%s' "$(printf '%s %s' "$man" "$prod" | sed 's/^ *//; s/ *$//')"
  else
    printf '%s:%s' "$(_usb_attr "$dir" idVendor)" "$(_usb_attr "$dir" idProduct)"
  fi
}

# `--list-usb`: print all three spec forms for every device, so a caller can
# copy one rather than translate `lsusb` output by hand.
usb_list_devices() {
  local dir name bus dev vid pid port
  printf '%-12s %-10s %-10s %s\n' 'VID:PID' 'BUS.ADDR' 'BUS-PORT' 'DEVICE'
  while IFS= read -r dir; do
    name="${dir##*/}"
    bus="$(_usb_attr "$dir" busnum)"
    dev="$(_usb_attr "$dir" devnum)"
    vid="$(_usb_attr "$dir" idVendor)"
    pid="$(_usb_attr "$dir" idProduct)"
    port="${name#*-}"
    printf '%-12s %-10s %-10s %s\n' \
      "$vid:$pid" "$((10#$bus)).$((10#$dev))" "$name" "$(_usb_describe "$dir")"
  done <<EOF
$(_usb_device_dirs)
EOF
}

# usb_resolve_spec SPEC
#
# On success sets USB_R_PROPS (the QEMU property fragment), USB_R_DESC (what to
# print), USB_R_BUSNUM and USB_R_DEVNUM (for the /dev node check), and returns 0.
# On failure sets USB_R_ERROR and returns 1. Never exits — the caller owns how a
# refusal is reported, because the two boot scripts spell `die` differently.
usb_resolve_spec() {
  local spec="$1"
  USB_R_PROPS=""; USB_R_DESC=""; USB_R_BUSNUM=""; USB_R_DEVNUM=""; USB_R_ERROR=""

  local kind vid pid want_bus want_addr want_port
  if printf '%s' "$spec" | grep -Eq '^[0-9a-fA-F]{4}:[0-9a-fA-F]{4}$'; then
    kind=id
    # sysfs writes idVendor/idProduct in lowercase hex; normalise so `046D:C099`
    # and `046d:c099` are the same spec.
    vid="$(printf '%s' "${spec%%:*}" | tr 'A-F' 'a-f')"
    pid="$(printf '%s' "${spec##*:}" | tr 'A-F' 'a-f')"
  elif printf '%s' "$spec" | grep -Eq '^[0-9]+\.[0-9]+$'; then
    kind=addr
    want_bus="$((10#${spec%%.*}))"
    want_addr="$((10#${spec##*.}))"
  elif printf '%s' "$spec" | grep -Eq '^[0-9]+-[0-9]+(\.[0-9]+)*$'; then
    kind=port
    want_port="$spec"
  else
    USB_R_ERROR="unrecognised USB spec '$spec' (want VID:PID like 046d:c099, BUS.ADDR like 5.3, or BUS-PORT like 5-1.2)"
    return 1
  fi

  local dir name bus dev match=""
  while IFS= read -r dir; do
    [ -n "$dir" ] || continue
    name="${dir##*/}"
    bus="$((10#$(_usb_attr "$dir" busnum)))"
    dev="$((10#$(_usb_attr "$dir" devnum)))"
    case "$kind" in
      id)
        [ "$(_usb_attr "$dir" idVendor)" = "$vid" ] || continue
        [ "$(_usb_attr "$dir" idProduct)" = "$pid" ] || continue
        ;;
      addr)
        [ "$bus" = "$want_bus" ] && [ "$dev" = "$want_addr" ] || continue
        ;;
      port)
        [ "$name" = "$want_port" ] || continue
        ;;
    esac
    match="$dir"
    break
  done <<EOF
$(_usb_device_dirs)
EOF

  if [ -z "$match" ]; then
    USB_R_ERROR="no USB device matches '$spec' — plug it in before booting, or run --list-usb to see what is present"
    return 1
  fi

  name="${match##*/}"
  USB_R_BUSNUM="$((10#$(_usb_attr "$match" busnum)))"
  USB_R_DEVNUM="$((10#$(_usb_attr "$match" devnum)))"
  USB_R_DESC="$(_usb_describe "$match") [$(_usb_attr "$match" idVendor):$(_usb_attr "$match" idProduct) bus $USB_R_BUSNUM addr $USB_R_DEVNUM port $name]"

  # Each spec form keeps the identity the caller asked for. Resolving BUS-PORT or
  # VID:PID down to hostbus/hostaddr here would silently convert a stable spec
  # into a one-boot one, so the property pair mirrors the form that was given.
  case "$kind" in
    id)   USB_R_PROPS="vendorid=0x$vid,productid=0x$pid" ;;
    addr) USB_R_PROPS="hostbus=$USB_R_BUSNUM,hostaddr=$USB_R_DEVNUM" ;;
    port) USB_R_PROPS="hostbus=$USB_R_BUSNUM,hostport=${name#*-}" ;;
  esac

  # libusb claims an interface, which needs write access and not just read.
  local node
  node="$(printf '/dev/bus/usb/%03d/%03d' "$USB_R_BUSNUM" "$USB_R_DEVNUM")"
  if [ ! -e "$node" ]; then
    USB_R_ERROR="$spec resolved to $USB_R_DESC but $node does not exist (is usbfs mounted?)"
    return 1
  fi
  if [ ! -r "$node" ] || [ ! -w "$node" ]; then
    USB_R_ERROR="$spec resolved to $USB_R_DESC but $node is not readable+writable by $(id -un).
libusb must claim the interface, so read access alone is not enough — it fails later, inside the guest's driver probe.
Grant it with a udev rule (survives replug), then replug the device:
  echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"$(_usb_attr "$match" idVendor)\", ATTR{idProduct}==\"$(_usb_attr "$match" idProduct)\", MODE=\"0660\", TAG+=\"uaccess\"' | sudo tee /etc/udev/rules.d/70-reims-vgpu-usb.rules
  sudo udevadm control --reload && sudo udevadm trigger
Or, for this boot only:  sudo chmod 0666 $node"
    return 1
  fi
  return 0
}
