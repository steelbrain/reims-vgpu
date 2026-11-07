//! USB passthrough: the host's USB devices a VM's guest is given, through QEMU's `usb-host` on
//! the VM's xHCI controller, on Linux hosts.
//!
//! - A chosen device is kept in the VM's settings ([`UsbDevice`], a `[[usb]]` table) by its
//!   vendor and product IDs, which follow it from port to port. One chosen while the host has
//!   another device with its IDs is kept by its port too (Linux's name for it, `3-2.1`), which
//!   tells the two apart while each stays in its port. QEMU's `usb-host` matches the same way
//!   (`vendorid` and `productid`, with `hostbus` and `hostport`).
//! - QEMU takes the device from the host's drivers when it opens it, and gives it back when it
//!   closes it: when the VM stops, or the device is taken back. The guest sees it on its xHCI
//!   controller as a Mac sees one plugged into it, and drives it with macOS's own drivers.
//! - A chosen device that is not plugged in when the VM starts joins the guest when it is: QEMU
//!   looks for the match of each `usb-host` device without one every 2 s
//!   (hw/usb/host-libusb.c, `usb_host_auto_check`). One unplugged leaves the guest, and comes
//!   back when it is plugged in again.
//! - While the VM runs, a device chosen or taken back is attached or detached at once, with
//!   QMP's `device_add` and `device_del`, and kept for the next start ([`apply`]).
//! - QEMU opens the device's file, `/dev/bus/usb/BBB/DDD`, which only root may write unless a
//!   udev rule says otherwise: scripts/install-ctl.sh adds one ([`UDEV_RULE`]), which gives USB
//!   devices other than hubs to the kvm group and to the user at the host's screen.
//! - A macOS host has none ([`MAC_HOST`]).

use crate::vm::{UsbDevice, Vm, parse_usb_port};
use crate::{qmp, runtime};
use anyhow::{Context, Result, bail};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Why a macOS host passes no USB device through: macOS's IOKit gives a USB device that one of
/// its drivers holds, which is almost every device, to another process only when that process
/// runs as root or holds `com.apple.vm.device-access`, an entitlement Apple grants VM apps with
/// a provisioning profile (libusb's `darwin_detach_kernel_driver`), which no QEMU built here
/// can hold.
pub const MAC_HOST: &str = "USB passthrough needs a Linux host: macOS gives a USB device that \
    one of its drivers holds, almost any device, only to a program running as root or holding an \
    entitlement Apple grants VM apps, and QEMU here can't hold it";

/// The udev rule scripts/install-ctl.sh writes, and where: USB devices other than hubs to the
/// kvm group (over SSH too) and to the user at the host's screen (`uaccess`, which needs a file
/// that sorts before systemd's 73-seat-late.rules). Group and others keep the read access
/// udev's default gives, 0664.
pub const UDEV_RULE_FILE: &str = "/etc/udev/rules.d/70-reims-ctl-usb.rules";
pub const UDEV_RULE: &str = r#"SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{bDeviceClass}!="09", GROUP="kvm", MODE="0664", TAG+="uaccess""#;

/// Where Linux lists the host's USB devices, and lsusb's names for them.
const SYSFS: &str = "/sys/bus/usb/devices";
const USB_IDS: &str = "/usr/share/hwdata/usb.ids";

/// A USB device plugged into the host, other than a hub.
#[derive(Clone, Debug, PartialEq)]
pub struct HostDevice {
    pub vendor: u16,
    pub product: u16,
    /// The port it is plugged into, as Linux names it: `3-2.1` is port 1 of the hub on port 2
    /// of bus 3.
    pub port: String,
    pub bus: u32,
    /// Its address on the bus, which names its file in `/dev/bus/usb`.
    pub address: u32,
    pub name: String,
    /// What its interfaces say it is, where they say: a keyboard, storage, Bluetooth.
    pub kinds: Vec<&'static str>,
}

impl HostDevice {
    /// Its IDs as lsusb shows them, `vvvv:pppp`.
    pub fn id(&self) -> String {
        format!("{:04x}:{:04x}", self.vendor, self.product)
    }

    /// The file QEMU opens it by.
    pub fn file(&self) -> PathBuf {
        PathBuf::from(format!("/dev/bus/usb/{:03}/{:03}", self.bus, self.address))
    }

    /// Whether this process may open its file to drive it, as QEMU, which runs as this user,
    /// opens it.
    pub fn usable(&self) -> bool {
        use std::os::unix::ffi::OsStrExt;
        let Ok(f) = std::ffi::CString::new(self.file().as_os_str().as_bytes()) else {
            return false;
        };
        // SAFETY: f is a NUL-terminated path; access only reads it.
        unsafe { libc::access(f.as_ptr(), libc::R_OK | libc::W_OK) == 0 }
    }

    /// Whether `u` names it.
    pub fn is(&self, u: &UsbDevice) -> bool {
        u.ids().is_ok_and(|ids| ids == (self.vendor, self.product))
            && u.port.as_ref().is_none_or(|p| *p == self.port)
    }
}

/// The USB devices plugged into the host, other than hubs, by bus and port; none on a host
/// that has no Linux sysfs.
pub fn host_devices() -> Vec<HostDevice> {
    // Read once, where a device first needs it: the interface's view rescans every second.
    static IDS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    read_devices(Path::new(SYSFS), || {
        IDS.get_or_init(|| {
            String::from_utf8_lossy(&std::fs::read(USB_IDS).unwrap_or_default()).into_owned()
        })
    })
}

/// The devices under `sysfs` (Linux's `/sys/bus/usb/devices`, where every device and interface
/// has an entry), named from lsusb's `usb.ids`, which `usb_ids` gives, where a device gives no
/// name of its own.
fn read_devices<'a>(sysfs: &Path, usb_ids: impl Fn() -> &'a str) -> Vec<HostDevice> {
    let Ok(dir) = std::fs::read_dir(sysfs) else {
        return Vec::new();
    };
    let names: Vec<String> = dir
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    let mut out = Vec::new();
    for port in &names {
        // A device's entry is its port; a root hub's is usbN, an interface's PORT:C.I.
        if parse_usb_port(port).is_none() {
            continue;
        }
        let d = sysfs.join(port);
        let read = |f: &str| {
            std::fs::read_to_string(d.join(f))
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        let hex = |f: &str| read(f).and_then(|s| u16::from_str_radix(&s, 16).ok());
        let num = |f: &str| read(f).and_then(|s| s.parse::<u32>().ok());
        let (Some(vendor), Some(product), Some(bus), Some(address)) = (
            hex("idVendor"),
            hex("idProduct"),
            num("busnum"),
            num("devnum"),
        ) else {
            continue;
        };
        // QEMU passes no hub through (usb_host_auto_check).
        if hex("bDeviceClass") == Some(0x09) {
            continue;
        }
        let name = name_of(read("manufacturer"), read("product"))
            .or_else(|| usb_ids_name(usb_ids(), vendor, product))
            .unwrap_or_else(|| "USB device".into());
        let prefix = format!("{port}:");
        let mut interfaces: Vec<&String> =
            names.iter().filter(|n| n.starts_with(&prefix)).collect();
        interfaces.sort();
        let mut kinds = Vec::new();
        for i in interfaces {
            for k in interface_kinds(&sysfs.join(i)) {
                if !kinds.contains(&k) {
                    kinds.push(k);
                }
            }
        }
        out.push(HostDevice {
            vendor,
            product,
            port: port.clone(),
            bus,
            address,
            name,
            kinds,
        });
    }
    out.sort_by_key(|d| port_order(&d.port));
    out
}

/// A port's place among its bus's: by bus, then port by port.
fn port_order(port: &str) -> Vec<u32> {
    port.split(['-', '.'])
        .map(|n| n.parse().unwrap_or(u32::MAX))
        .collect()
}

/// A device's name from its own strings: its product, after its maker where the product does
/// not name it.
fn name_of(manufacturer: Option<String>, product: Option<String>) -> Option<String> {
    let product = product?;
    let named = |m: &str| {
        let first = m.split([' ', ',']).next().unwrap_or(m).to_ascii_lowercase();
        product.to_ascii_lowercase().contains(&first)
    };
    Some(match manufacturer {
        Some(m) if !named(&m) => format!("{m} {product}"),
        _ => product,
    })
}

/// lsusb's name for a device in `usb.ids`: its vendor's line, `vvvv  Name`, then the product's
/// line under it, a tab before `pppp  Name`.
fn usb_ids_name(ids: &str, vendor: u16, product: u16) -> Option<String> {
    let v = format!("{vendor:04x}  ");
    let p = format!("\t{product:04x}  ");
    let mut lines = ids.lines().skip_while(|l| !l.starts_with(&v));
    let vendor_name = lines.next()?[v.len()..].trim().to_string();
    let product_name = lines
        .take_while(|l| l.starts_with('\t') || l.starts_with('#'))
        .find_map(|l| l.strip_prefix(&p))
        .map(str::trim);
    Some(match product_name {
        Some(n) => format!("{vendor_name} {n}"),
        None => vendor_name,
    })
}

/// What an interface says its device is, by its class, subclass and protocol (USB-IF's class
/// codes), or by the network or Bluetooth device Linux made of it.
fn interface_kinds(dir: &Path) -> Vec<&'static str> {
    let hex = |f: &str| {
        std::fs::read_to_string(dir.join(f))
            .ok()
            .and_then(|s| u8::from_str_radix(s.trim(), 16).ok())
    };
    let mut kinds = Vec::new();
    let class = (
        hex("bInterfaceClass"),
        hex("bInterfaceSubClass"),
        hex("bInterfaceProtocol"),
    );
    kinds.extend(match class {
        (Some(0x03), Some(0x01), Some(0x01)) => Some("keyboard"),
        (Some(0x03), Some(0x01), Some(0x02)) => Some("mouse"),
        (Some(0x01), ..) => Some("audio"),
        (Some(0x02), Some(0x02), _) => Some("serial"),
        (Some(0x02), Some(0x06 | 0x0d), _) | (Some(0xe0), Some(0x01), Some(0x03)) => {
            Some("network")
        }
        (Some(0x07), ..) => Some("printer"),
        (Some(0x08), ..) => Some("storage"),
        (Some(0x0b), ..) => Some("smart card"),
        (Some(0x0e), ..) => Some("camera"),
        (Some(0xe0), Some(0x01), Some(0x01)) => Some("Bluetooth"),
        _ => None,
    });
    if dir.join("net").is_dir() {
        kinds.push("network");
    }
    if dir.join("bluetooth").is_dir() {
        kinds.push("Bluetooth");
    }
    kinds
}

/// One line of a VM's choice: a device plugged in, with the setting that gives it to the VM if
/// one does, or a device the VM is given that is not plugged in.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub device: Option<HostDevice>,
    pub chosen: Option<UsbDevice>,
}

impl Row {
    pub fn name(&self) -> &str {
        match (&self.device, &self.chosen) {
            (Some(d), _) => &d.name,
            (None, Some(u)) if !u.name.is_empty() => &u.name,
            _ => "USB device",
        }
    }

    pub fn id(&self) -> String {
        match (&self.device, &self.chosen) {
            (Some(d), _) => d.id(),
            (None, Some(u)) => u.id.clone(),
            _ => String::new(),
        }
    }

    /// What to say of it beside its name and ID: what it is, its port where another device has
    /// its IDs, and what keeps it from the guest. `devices` are the host's.
    pub fn notes(&self, devices: &[HostDevice]) -> Vec<String> {
        let mut n = Vec::new();
        match &self.device {
            None => {
                n.push(match self.chosen.as_ref().and_then(|u| u.port.as_ref()) {
                    Some(p) => format!("not plugged in at {p}"),
                    None => "not plugged in".into(),
                });
            }
            Some(d) => {
                n.extend(d.kinds.iter().map(|k| k.to_string()));
                let alike = devices.iter().filter(|o| o.id() == d.id()).count();
                if alike > 1 {
                    n.push(format!("port {}", d.port));
                    if self.chosen.as_ref().is_some_and(|u| u.port.is_none()) {
                        n.push(format!("the VM takes one of these {alike}"));
                    }
                }
                if !d.usable() {
                    n.push("you can't open it".into());
                }
            }
        }
        n
    }
}

/// The host's devices, each with the choice in `chosen` that gives it, then the chosen devices
/// that are not plugged in.
pub fn rows(chosen: &[UsbDevice], devices: &[HostDevice]) -> Vec<Row> {
    let mut rows: Vec<Row> = devices
        .iter()
        .map(|d| Row {
            device: Some(d.clone()),
            chosen: chosen.iter().find(|u| d.is(u)).cloned(),
        })
        .collect();
    rows.extend(
        chosen
            .iter()
            .filter(|u| !devices.iter().any(|d| d.is(u)))
            .map(|u| Row {
                device: None,
                chosen: Some(u.clone()),
            }),
    );
    rows
}

/// The choice that gives the VM `d`: by its IDs, and by its port too where another of
/// `devices` has its IDs.
pub fn choice_of(d: &HostDevice, devices: &[HostDevice]) -> UsbDevice {
    let twin = devices
        .iter()
        .any(|o| o.port != d.port && o.vendor == d.vendor && o.product == d.product);
    UsbDevice {
        id: d.id(),
        port: twin.then(|| d.port.clone()),
        name: d.name.clone(),
    }
}

/// `chosen` with `row` turned over: a device given to the VM is taken back, and one not given
/// is given. Taking back one of two devices alike that one choice by their IDs gave keeps the
/// other, by its port.
pub fn toggle(chosen: &[UsbDevice], devices: &[HostDevice], row: &Row) -> Vec<UsbDevice> {
    let Some(d) = &row.device else {
        return chosen
            .iter()
            .filter(|u| Some(*u) != row.chosen.as_ref())
            .cloned()
            .collect();
    };
    let (gone, mut kept): (Vec<UsbDevice>, Vec<UsbDevice>) =
        chosen.iter().cloned().partition(|u| d.is(u));
    if gone.is_empty() {
        kept.push(choice_of(d, devices));
        return kept;
    }
    for u in gone.iter().filter(|u| u.port.is_none()) {
        for o in devices.iter().filter(|o| o.port != d.port && o.is(u)) {
            if !kept.iter().any(|k| o.is(k)) {
                kept.push(UsbDevice {
                    id: o.id(),
                    port: Some(o.port.clone()),
                    name: o.name.clone(),
                });
            }
        }
    }
    kept
}

/// The id of the `usb-host` device that passes `u` through, which `device_del` names: from its
/// IDs and port, in the letters, digits, `-` and `.` QEMU's ids take.
pub fn qemu_id(u: &UsbDevice) -> String {
    let mut id = match u.ids() {
        Ok((v, p)) => format!("usb-{v:04x}-{p:04x}"),
        Err(_) => format!("usb-{}", u.id),
    };
    if let Some(p) = &u.port {
        id.push('-');
        id.push_str(p);
    }
    id
}

/// The `-device` that passes `u` through, on the q35 machine's xHCI controller.
pub fn device_arg(u: &UsbDevice) -> Result<String> {
    let (v, p) = u.ids()?;
    let mut a = format!(
        "usb-host,id={},bus=xhci.0,vendorid={v:#06x},productid={p:#06x}",
        qemu_id(u)
    );
    if let Some((bus, port)) = u.bus_port()? {
        a.push_str(&format!(",hostbus={bus},hostport={port}"));
    }
    Ok(a)
}

/// QMP's `device_add` arguments for the same device, typed as QMP takes them.
fn device_add(u: &UsbDevice) -> Result<serde_json::Value> {
    let (v, p) = u.ids()?;
    let mut a = json!({
        "driver": "usb-host",
        "id": qemu_id(u),
        "bus": "xhci.0",
        "vendorid": v,
        "productid": p,
    });
    if let Some((bus, port)) = u.bus_port()? {
        a["hostbus"] = json!(bus);
        a["hostport"] = json!(port);
    }
    Ok(a)
}

/// Gives the VM the USB devices `change` makes of those it has, in its settings as they are
/// now, and saves them; while it runs, detaches the devices it no longer has and attaches the
/// new ones at once. Returns the VM as saved, and what happened, a sentence a device.
pub fn apply(
    vm: &Vm,
    change: impl FnOnce(&[UsbDevice]) -> Vec<UsbDevice>,
) -> Result<(Vm, Vec<String>)> {
    let mut now = Vm::load(&vm.dir)?;
    let old = now.config.usb.clone();
    now.config.usb = change(&old);
    let name = now.name().to_string();
    if now.config.usb == old {
        return Ok((now, vec![format!("{name}: nothing changed")]));
    }
    now.save()?;
    let gone: Vec<&UsbDevice> = old.iter().filter(|u| !now.config.usb.contains(u)).collect();
    let new: Vec<&UsbDevice> = now.config.usb.iter().filter(|u| !old.contains(u)).collect();
    let label = |u: &UsbDevice| match u.name.as_str() {
        "" => u.id.clone(),
        n => n.to_string(),
    };
    if !runtime::running(&now) {
        let mut said: Vec<String> = gone
            .iter()
            .map(|u| format!("{name}: {} is no longer chosen", label(u)))
            .collect();
        said.extend(
            new.iter()
                .map(|u| format!("{name}: {} is the guest's from its next start", label(u))),
        );
        return Ok((now, said));
    }
    let devices = host_devices();
    let mut q =
        qmp::Qmp::connect(&now.qmp_socket(), Duration::from_secs(5)).with_context(|| {
            format!("{name}: saved for its next start, but its QEMU did not answer")
        })?;
    let mut said = Vec::new();
    // Detached first, to free their ports.
    for u in gone {
        let plugged = devices.iter().any(|d| d.is(u));
        match q.execute("device_del", Some(json!({ "id": qemu_id(u) }))) {
            Ok(_) if plugged => said.push(format!("{name}: {} is back with this host", label(u))),
            Ok(_) => said.push(format!("{name}: {} is no longer chosen", label(u))),
            // Never attached, as when the VM started before it was chosen and QMP failed.
            Err(e) if format!("{e}").contains("not found") => {
                said.push(format!("{name}: {} is no longer chosen", label(u)))
            }
            Err(e) => said.push(format!(
                "{name}: {} stays the guest's until the VM stops: {e:#}",
                label(u)
            )),
        }
    }
    for u in new {
        let plugged = devices.iter().find(|d| d.is(u));
        match q.execute("device_add", Some(device_add(u)?)) {
            Ok(_) => said.push(match plugged {
                Some(d) if !d.usable() => {
                    format!("{name}: {} can't be the guest's: {}", label(u), unusable(d))
                }
                Some(_) => format!("{name}: {} is the guest's", label(u)),
                None => format!(
                    "{name}: {} isn't plugged in; the guest gets it when it is",
                    label(u)
                ),
            }),
            Err(e) => said.push(format!(
                "{name}: {} is the guest's from its next start; attaching it now failed: {e:#}",
                label(u)
            )),
        }
    }
    Ok((now, said))
}

/// Why `d` can't be opened, and what lets QEMU open it.
pub fn unusable(d: &HostDevice) -> String {
    format!("you can't open {}: {}", d.file().display(), fix())
}

/// What lets QEMU, as this user, open USB devices: the udev rule the install adds, or, where it
/// is there, the kvm group it gives them to.
pub fn fix() -> String {
    if Path::new(UDEV_RULE_FILE).exists() {
        "the udev rule reims-ctl's install added gives USB devices to the kvm group and to \
         whoever is at this host's screen: add yourself to kvm (sudo usermod -aG kvm $USER) and \
         log in again"
            .into()
    } else {
        format!(
            "run reims-ctl update, whose install adds the udev rule {UDEV_RULE_FILE} for USB \
             devices (reims-ctl usb --help shows it)"
        )
    }
}

/// What keeps the VM's chosen USB devices from its guest, a sentence each: one not plugged in,
/// which joins when it is, or one this user can't open.
pub fn problems(vm: &Vm) -> Vec<String> {
    if vm.config.usb.is_empty() {
        return Vec::new();
    }
    let devices = host_devices();
    let mut p = Vec::new();
    for u in &vm.config.usb {
        let name = match u.name.as_str() {
            "" => u.id.clone(),
            n => format!("{n} ({})", u.id),
        };
        match devices.iter().find(|d| d.is(u)) {
            None => p.push(format!(
                "USB device {name} isn't plugged in{}: the guest gets it when it is.",
                u.port
                    .as_ref()
                    .map(|p| format!(" at port {p}"))
                    .unwrap_or_default()
            )),
            Some(d) if !d.usable() => p.push(format!(
                "USB device {name} can't be the guest's: {}.",
                unusable(d)
            )),
            Some(_) => {}
        }
    }
    p
}

/// The row a command line's `DEVICE` names: its number in the list, its IDs, its port, or words
/// of its name.
pub fn find(rows: &[Row], arg: &str) -> Result<usize> {
    let a = arg.trim();
    if let Ok(n) = a.parse::<usize>()
        && (1..=rows.len()).contains(&n)
    {
        return Ok(n - 1);
    }
    let low = a.to_ascii_lowercase();
    let by = |f: &dyn Fn(&Row) -> bool| -> Vec<usize> {
        rows.iter()
            .enumerate()
            .filter(|(_, r)| f(r))
            .map(|(i, _)| i)
            .collect()
    };
    for hits in [
        by(&|r| r.device.as_ref().is_some_and(|d| d.port == a)),
        by(&|r| r.id() == low),
        by(&|r| r.name().to_ascii_lowercase().contains(&low)),
    ] {
        match hits.len() {
            0 => continue,
            1 => return Ok(hits[0]),
            _ => bail!("{a:?} names several USB devices: give its number or its port"),
        }
    }
    bail!("no USB device matches {a:?}: reims-ctl usb lists them")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sysfs of USB devices in a temporary directory, as Linux lists them: each a directory
    /// of its attributes, its interfaces beside it.
    struct Sysfs(PathBuf);

    impl Sysfs {
        fn new(tag: &str) -> Sysfs {
            let d =
                std::env::temp_dir().join(format!("reims-ctl-usb-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Sysfs(d)
        }

        fn entry(&self, name: &str, attrs: &[(&str, &str)]) -> &Sysfs {
            let d = self.0.join(name);
            std::fs::create_dir_all(&d).unwrap();
            for (k, v) in attrs {
                std::fs::write(d.join(k), format!("{v}\n")).unwrap();
            }
            self
        }

        fn device(&self, port: &str, id: &str, devnum: u32, strings: &[(&str, &str)]) -> &Sysfs {
            let (v, p) = id.split_once(':').unwrap();
            let bus = port.split('-').next().unwrap();
            let num = devnum.to_string();
            let mut attrs = vec![
                ("idVendor", v),
                ("idProduct", p),
                ("busnum", bus),
                ("devnum", num.as_str()),
                ("bDeviceClass", "00"),
            ];
            attrs.extend_from_slice(strings);
            self.entry(port, &attrs)
        }

        fn interface(&self, name: &str, class: &str) -> &Sysfs {
            let c: Vec<&str> = class.split('/').collect();
            self.entry(
                name,
                &[
                    ("bInterfaceClass", c[0]),
                    ("bInterfaceSubClass", c[1]),
                    ("bInterfaceProtocol", c[2]),
                ],
            )
        }
    }

    impl Drop for Sysfs {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const USB_IDS_TEXT: &str = "# usb.ids\n\
        0bda  Realtek Semiconductor Corp.\n\
        \t5458  BillBoard Device\n\
        8087  Intel Corp.\n\
        \t0029  AX200 Bluetooth\n\
        \t\t00  an interface\n\
        C 00  (Defined at Interface level)\n";

    /// The host's devices: hubs and root hubs left out, each named from its own strings or
    /// else lsusb's, and what its interfaces say it is; in port order.
    #[test]
    fn host_devices_are_read_from_sysfs() {
        let s = Sysfs::new("read");
        s.entry(
            "usb1",
            &[
                ("idVendor", "1d6b"),
                ("idProduct", "0002"),
                ("bDeviceClass", "09"),
            ],
        )
        .entry("1-0:1.0", &[("bInterfaceClass", "09")])
        .device("1-5", "1a40:0101", 3, &[("product", "USB 2.0 Hub")])
        .entry("1-5", &[("bDeviceClass", "09")])
        .device("1-5.1", "8087:0029", 4, &[])
        .interface("1-5.1:1.0", "e0/01/01")
        .interface("1-5.1:1.1", "e0/01/01")
        .device(
            "1-10",
            "046d:c52b",
            7,
            &[("manufacturer", "Logitech"), ("product", "USB Receiver")],
        )
        .interface("1-10:1.0", "03/01/01")
        .interface("1-10:1.1", "03/01/02")
        .interface("1-10:1.2", "03/00/00")
        .device(
            "1-3",
            "1050:0407",
            2,
            &[
                ("manufacturer", "Yubico"),
                ("product", "YubiKey OTP+FIDO+CCID"),
            ],
        )
        .device(
            "2-1",
            "0781:5567",
            2,
            &[
                ("manufacturer", "SanDisk"),
                ("product", "SanDisk Cruzer Blade"),
            ],
        )
        .interface("2-1:1.0", "08/06/50");
        std::fs::create_dir_all(s.0.join("2-1:1.0/host0")).unwrap();
        s.device("1-4", "0bda:5458", 5, &[]);
        let d = read_devices(&s.0, || USB_IDS_TEXT);
        let got: Vec<(&str, String, &str, &[&str])> = d
            .iter()
            .map(|d| (d.port.as_str(), d.id(), d.name.as_str(), d.kinds.as_slice()))
            .collect();
        assert_eq!(
            got,
            [
                (
                    "1-3",
                    "1050:0407".into(),
                    "Yubico YubiKey OTP+FIDO+CCID",
                    &[][..]
                ),
                (
                    "1-4",
                    "0bda:5458".into(),
                    "Realtek Semiconductor Corp. BillBoard Device",
                    &[]
                ),
                (
                    "1-5.1",
                    "8087:0029".into(),
                    "Intel Corp. AX200 Bluetooth",
                    &["Bluetooth"]
                ),
                (
                    "1-10",
                    "046d:c52b".into(),
                    "Logitech USB Receiver",
                    &["keyboard", "mouse"]
                ),
                (
                    "2-1",
                    "0781:5567".into(),
                    "SanDisk Cruzer Blade",
                    &["storage"]
                ),
            ]
        );
        assert_eq!(d[2].file(), Path::new("/dev/bus/usb/001/004"));
        assert_eq!((d[2].bus, d[2].address), (1, 4));
        // No sysfs, as on macOS: no devices.
        assert!(read_devices(&s.0.join("none"), || USB_IDS_TEXT).is_empty());
    }

    #[test]
    fn names_from_usb_ids() {
        assert_eq!(
            usb_ids_name(USB_IDS_TEXT, 0x8087, 0x0029).as_deref(),
            Some("Intel Corp. AX200 Bluetooth")
        );
        // A product it does not list: the vendor's name; a vendor it does not list: none.
        assert_eq!(
            usb_ids_name(USB_IDS_TEXT, 0x8087, 0x1234).as_deref(),
            Some("Intel Corp.")
        );
        assert_eq!(usb_ids_name(USB_IDS_TEXT, 0x1234, 0x0029), None);
        assert_eq!(
            name_of(
                Some("Logitech, Inc.".into()),
                Some("Logitech Unifying".into())
            )
            .as_deref(),
            Some("Logitech Unifying")
        );
        assert_eq!(name_of(Some("FTDI".into()), None), None);
    }

    fn dev(port: &str, id: &str, name: &str) -> HostDevice {
        let (v, p) = id.split_once(':').unwrap();
        HostDevice {
            vendor: u16::from_str_radix(v, 16).unwrap(),
            product: u16::from_str_radix(p, 16).unwrap(),
            port: port.into(),
            bus: port.split('-').next().unwrap().parse().unwrap(),
            address: 9,
            name: name.into(),
            kinds: vec![],
        }
    }

    fn choice(id: &str, port: Option<&str>, name: &str) -> UsbDevice {
        UsbDevice {
            id: id.into(),
            port: port.map(String::from),
            name: name.into(),
        }
    }

    /// A device is chosen by its IDs, which follow it to any port; where the host has two
    /// alike, by its port too, which tells them apart.
    #[test]
    fn a_device_is_chosen_by_its_ids_and_by_its_port_beside_its_twin() {
        let key = dev("1-3", "1050:0407", "YubiKey");
        let a = dev("1-4", "0403:6001", "FT232R");
        let b = dev("2-1.2", "0403:6001", "FT232R");
        let devices = [key.clone(), a.clone(), b.clone()];
        assert_eq!(
            choice_of(&key, &devices),
            choice("1050:0407", None, "YubiKey")
        );
        assert_eq!(
            choice_of(&b, &devices),
            choice("0403:6001", Some("2-1.2"), "FT232R")
        );
        // In another port, the key is still the one chosen; the adapter at 2-1.2 is only there.
        let moved = dev("3-1", "1050:0407", "YubiKey");
        assert!(moved.is(&choice_of(&key, &devices)));
        assert!(!a.is(&choice_of(&b, &devices)));
        assert!(b.is(&choice_of(&b, &devices)));
    }

    /// Space on a row gives the device or takes it back; a device chosen that is not plugged in
    /// has a row of its own, after those plugged in, which takes it back.
    #[test]
    fn toggling_gives_a_device_and_takes_it_back() {
        let key = dev("1-3", "1050:0407", "YubiKey");
        let stick = dev("2-1", "0781:5567", "Cruzer");
        let devices = [key.clone(), stick.clone()];
        let away = choice("0403:6001", None, "FT232R");
        let mut chosen = vec![away.clone()];
        let r = rows(&chosen, &devices);
        assert_eq!(r.len(), 3);
        assert_eq!((r[0].chosen.as_ref(), r[2].device.as_ref()), (None, None));
        assert_eq!(r[2].chosen.as_ref(), Some(&away));
        assert_eq!(r[2].notes(&devices), ["not plugged in"]);
        chosen = toggle(&chosen, &devices, &r[1]);
        assert_eq!(chosen, [away.clone(), choice("0781:5567", None, "Cruzer")]);
        let r = rows(&chosen, &devices);
        assert!(r[1].chosen.is_some());
        chosen = toggle(&chosen, &devices, &r[1]);
        assert_eq!(chosen, [away]);
        chosen = toggle(&chosen, &devices, &r[2]);
        assert!(chosen.is_empty());
    }

    /// Two devices alike that one choice by their IDs gives (a twin plugged in after it was
    /// chosen) both show chosen, and say the VM takes one; taking one back keeps the other, by
    /// its port.
    #[test]
    fn taking_back_one_of_two_alike_keeps_the_other_by_its_port() {
        let a = dev("1-4", "0403:6001", "FT232R");
        let b = dev("2-1.2", "0403:6001", "FT232R");
        let devices = [a.clone(), b.clone()];
        let chosen = vec![choice("0403:6001", None, "FT232R")];
        let r = rows(&chosen, &devices);
        assert!(r.iter().all(|r| r.chosen.is_some()), "{r:?}");
        let notes = r[0].notes(&devices);
        assert!(notes.contains(&"port 1-4".to_string()), "{notes:?}");
        assert!(
            notes.contains(&"the VM takes one of these 2".to_string()),
            "{notes:?}"
        );
        let kept = toggle(&chosen, &devices, &r[0]);
        assert_eq!(kept, [choice("0403:6001", Some("2-1.2"), "FT232R")]);
        let r = rows(&kept, &devices);
        assert_eq!(
            (r[0].chosen.is_some(), r[1].chosen.is_some()),
            (false, true)
        );
        // Given again, each is its own, by its port.
        let both = toggle(&kept, &devices, &r[0]);
        assert_eq!(
            both,
            [
                choice("0403:6001", Some("2-1.2"), "FT232R"),
                choice("0403:6001", Some("1-4"), "FT232R"),
            ]
        );
    }

    #[test]
    fn qemu_passes_the_device_through_on_the_xhci_controller() {
        let u = choice("1050:0407", None, "YubiKey");
        assert_eq!(qemu_id(&u), "usb-1050-0407");
        assert_eq!(
            device_arg(&u).unwrap(),
            "usb-host,id=usb-1050-0407,bus=xhci.0,vendorid=0x1050,productid=0x0407"
        );
        let u = choice("0403:6001", Some("3-2.1"), "FT232R");
        assert_eq!(qemu_id(&u), "usb-0403-6001-3-2.1");
        assert_eq!(
            device_arg(&u).unwrap(),
            "usb-host,id=usb-0403-6001-3-2.1,bus=xhci.0,vendorid=0x0403,productid=0x6001,\
             hostbus=3,hostport=2.1"
        );
        // QMP takes the IDs and the bus as numbers, the port as a string.
        assert_eq!(
            device_add(&u).unwrap(),
            json!({
                "driver": "usb-host",
                "id": "usb-0403-6001-3-2.1",
                "bus": "xhci.0",
                "vendorid": 0x0403,
                "productid": 0x6001,
                "hostbus": 3,
                "hostport": "2.1",
            })
        );
        assert!(device_arg(&choice("1050:04070", None, "")).is_err());
        assert!(device_arg(&choice("1050:0407", Some("3-2.1,x=y"), "")).is_err());
    }

    /// The rule `usb --help` shows and the fix names is the one scripts/install-ctl.sh writes,
    /// where it writes it.
    #[test]
    fn the_install_writes_the_udev_rule_reims_ctl_names() {
        let script = include_str!("../../../scripts/install-ctl.sh");
        assert!(script.contains(&format!("usb_rule='{UDEV_RULE}'\n")));
        assert!(script.contains(&format!("usb_rule_file={UDEV_RULE_FILE}\n")));
    }

    #[test]
    fn a_command_line_names_a_device_by_number_id_port_or_name() {
        let devices = [
            dev("1-3", "1050:0407", "Yubico YubiKey"),
            dev("1-4", "0403:6001", "FTDI FT232R"),
            dev("2-1.2", "0403:6001", "FTDI FT232R"),
        ];
        let r = rows(&[choice("0781:5567", None, "SanDisk Cruzer")], &devices);
        assert_eq!(find(&r, "1").unwrap(), 0);
        assert_eq!(find(&r, "4").unwrap(), 3);
        assert_eq!(find(&r, "1050:0407").unwrap(), 0);
        assert_eq!(find(&r, "2-1.2").unwrap(), 2);
        assert_eq!(find(&r, "yubikey").unwrap(), 0);
        assert_eq!(find(&r, "cruzer").unwrap(), 3);
        let e = format!("{:#}", find(&r, "0403:6001").unwrap_err());
        assert!(e.contains("several"), "{e}");
        assert!(find(&r, "5").is_err());
        assert!(find(&r, "webcam").is_err());
    }
}
