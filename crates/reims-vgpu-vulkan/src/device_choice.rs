//! Which GPU the library opens where the host has more than one Vulkan device: the runtime's
//! default device is made of the physical device this module picks, once a process ([`chosen`]),
//! and every default open reaches it (`crate::platform`'s `default_device`).
//!
//! **Candidates.** The physical devices are those the window's probe lists, in the loader's order
//! (`crate::os`'s `seen_devices`, read through the probe's one instance). A candidate is one the
//! runtime makes a device of ([`Seen::usable`]): at Vulkan 1.2 or later, with a queue family of
//! graphics and compute work, and a `deviceUUID`, which names it to the runtime: the registry ID
//! the runtime answers for the physical device of that UUID names it ([`listed_registry_id`]).
//! And it is one a display's window presents on: it lists every device extension the window's
//! swapchain is made with ([`Seen::window_lacks`]): on the Vulkan platform `VK_KHR_swapchain` and
//! the OS's own (`crate::os`'s `WINDOW_DEVICE_EXTENSIONS`: `VK_EXT_metal_objects` on macOS, none
//! on Linux and Windows), on the rt platform `VK_KHR_swapchain`, which rt's floor takes. The rule
//! is the same on every OS, and no vendor, device or driver takes part in it.
//!
//! **Why the window decides.** The platform names the window's device extensions in the options
//! of every device it makes where a candidate lists them (`crate::os`'s `add_window_extensions`),
//! and the runtime refuses to make a device that lacks one, so a device that lacks one is never
//! made: not picked, not timed, and not the compositor's GPU ([`compositor_gpu`]). Before, the
//! extensions were named only where every device listed them, and a device that lacked one was a
//! candidate like any other. On the M3 Max with Mesa's KosmicKrisp beside MoltenVK (Mesa 26.3.1,
//! its manifest where Homebrew's `mesa` puts it), the loader listed KosmicKrisp first, which lists
//! no `VK_EXT_metal_objects`; the choice timed it at 1.71 ms to MoltenVK's 3.01 ms and took it,
//! no device was given the window's extensions, and `firmware/ambrosia/check-boot.sh`'s
//! full-screen Metal app presented 294 frames at 30 fps and dropped 106, where on MoltenVK alone
//! it presented 601 at 60 fps (**measured**, 2026-11-19, a macOS 27.0.1 guest under HVF,
//! m3-max-57095 against m3-max-57094). A device passed over for the window is logged with what it
//! lacks, on its own line and on the pick's.
//!
//! **The pick.**
//! - Where the host's user picked a device, through `MESA_VK_DEVICE_SELECT` (with or without its
//!   `!`), `DRI_PRIME` or `VK_LOADER_DEVICE_SELECT` ([`SELECTION_VARIABLES`]), the candidate the
//!   variable names ([`named`]): by its vendor and device IDs (`vendor:device` in hex, in any of
//!   the three), or by its PCI address (`DRI_PRIME=pci-0000_03_00_0`); of two candidates of the
//!   same IDs, the first in the loader's order. The variables are read in that order, and one
//!   that names no candidate (no device of the host's, one below Vulkan 1.2, or one the window
//!   cannot present on) is passed over for the next. The loader's order is not the pick: Mesa's
//!   device-select layer sorts the devices after the loader and lists its own default first
//!   unless its own variables name a device, so with `VK_LOADER_DEVICE_SELECT` alone naming
//!   um870's RTX 3090, the loader's first device was the layer's default, its Radeon 780M
//!   (**measured**, 2026-11-14, m3-max-30538, Mesa 26.2.6 and the loader 1.4.363).
//!   `DRI_PRIME`'s index (`DRI_PRIME=1`, with or without its `!`) selects the Nth GPU other than
//!   the layer's default, which the library cannot tell, and the layer lists the GPU it selects
//!   first, so there the pick is the loader's first candidate (read from Mesa's documentation of
//!   `DRI_PRIME`). Where no variable set names a candidate, the pick is the loader's first
//!   candidate, and the log line says that none names one. The runtime alone ranks the devices
//!   by kind first, so left to itself it opens a discrete GPU even where the user's variable
//!   named an integrated one.
//! - A device the window cannot present on is passed over even where the user's variable names
//!   it, as one below Vulkan 1.2 is: the window's extensions are named before the pick, from the
//!   candidates, and the runtime refuses to make a device that lacks one, so the device would not
//!   open. A host's user who wants such a device hides the other drivers from the loader (its
//!   `VK_DRIVER_FILES`): with no candidate, no device is given the window's extensions, and the
//!   runtime's default device opens as before.
//! - Otherwise, where one candidate is a GPU (any kind but a CPU implementation such as
//!   llvmpipe), that GPU, never timed.
//! - Where several are, the one that ran a fixed workload fastest ([`measured`]): the display's
//!   composite at a fixed size and one pass over textures in host memory, timed on a device made
//!   for each GPU and torn down after, apart from the process's own device, which is never one of
//!   them (`timing.rs`). The verdict is kept in the library's cache
//!   directory, under a key of every timed GPU's UUID and driver version (`verdict.rs`), so only
//!   the first open after a GPU or driver change times them; a verdict that does not read times
//!   them again. Where the runtime refuses any GPU's workload, nothing is kept and the runtime's
//!   own order of the candidates stands.
//! - A CPU implementation is picked only where no GPU is a candidate, and never timed. The
//!   runtime's own order: discrete GPUs, then integrated, virtual and CPU ones, each kind in the
//!   loader's order (`Device::system_default_with`, read from code). The runtime tries the
//!   devices it makes in that order and takes the first it does not refuse, so where the first
//!   lacks one of the window's extensions it makes the next; the choice opens its pick by registry
//!   ID wherever the runtime's first try is another device ([`runtime_first`]).
//!
//! No vendor or device ID takes part: what a device reports picks the GPUs to time, and their
//! times pick among them. The kind, the memory heaps and the PCIe link do not tell a GPU's speed
//! for the guest's work (an integrated GPU runs from 2 compute units to 40, and a discrete one of
//! 4 GiB may be a GTX 1650 or an RX 550), so where they cannot decide alone, a measurement does.
//!
//! **Measured** on 2026-11-08 (Mesa 26.2.6, loader 1.4.363), the workload's two timed frames:
//! - um790: the Radeon 780M took 30 ms and the RX 550 40 ms, so the 780M is picked. The runtime's
//!   order opened the RX 550, which composites a 3840x2400 frame in 40.4 ms to the 780M's 24.4 ms
//!   (reims-vgpu's `composite_cost`).
//! - gem10: the RX 7900 XTX took 9 ms and the Radeon 780M 29 ms, so the XTX stays the pick.
//! - The measurement took 197 ms in all on um790 and about 180 ms on gem10, once; a first open
//!   that reads the kept verdict took 39 to 45 ms on either.
//!
//! **Unchanged hosts.** Where the pick is the device the runtime tries first for its default
//! device, it is made as before, by `Device::system_default_with` ([`chosen`] answers `None`). The
//! choice is logged once, at info: every device seen, each GPU's time and what its measurement
//! cost in all, then the one picked and why, with each device passed over for the window and why
//! ([`choice_line`]).

mod timing;
mod verdict;

use std::ffi::{CStr, OsString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use ash::vk;
use metal2vulkan_runtime as rt;

use crate::core::platform::os::Clock as _;

/// The log target of the choice's lines.
const TARGET: &str = "reims_vgpu_vulkan::device_choice";

/// The variables through which the host's user picks a Vulkan device, in the order the pick
/// reads them: Mesa's device-select layer's `MESA_VK_DEVICE_SELECT` (`vendor:device`, with a `!`
/// it lists that device alone) and `DRI_PRIME` (`vendor:device`, `pci-DOMAIN_BUS_DEVICE_FUNCTION`
/// or an index, each with or without a `!`), and the loader's `VK_LOADER_DEVICE_SELECT`
/// (`vendor:device`), whose order the layer's sort comes after ([`named`]). Measured: Mesa
/// 26.2.6's layer and the loader 1.4.363 read these names.
pub(crate) const SELECTION_VARIABLES: [&str; 3] = [
    "MESA_VK_DEVICE_SELECT",
    "DRI_PRIME",
    "VK_LOADER_DEVICE_SELECT",
];

/// What the probe read of one physical device.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Seen {
    /// `VkPhysicalDeviceProperties::deviceName`.
    pub(crate) name: String,
    /// `VkPhysicalDeviceProperties::deviceType`.
    pub(crate) kind: vk::PhysicalDeviceType,
    /// `VkPhysicalDeviceProperties::vendorID`, logged only.
    pub(crate) vendor_id: u32,
    /// `VkPhysicalDeviceProperties::deviceID`, logged only.
    pub(crate) device_id: u32,
    /// `VkPhysicalDeviceProperties::apiVersion`.
    pub(crate) api_version: u32,
    /// `VkPhysicalDeviceProperties::driverVersion`, in the driver's own encoding.
    pub(crate) driver_version: u32,
    /// Whether a queue family does graphics and compute work.
    pub(crate) graphics_compute: bool,
    /// The size of its largest device-local memory heap, in bytes, logged only.
    pub(crate) device_local: u64,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`; `None` where there is none to read (below
    /// Vulkan 1.1).
    pub(crate) uuid: Option<[u8; 16]>,
    /// Its DRM nodes (`VK_EXT_physical_device_drm`); `None` where it does not list the extension.
    pub(crate) drm: Option<DrmNodes>,
    /// Its PCI domain, bus, device and function (`VK_EXT_pci_bus_info`); `None` where it does not
    /// list the extension.
    pub(crate) pci: Option<(u32, u32, u32, u32)>,
    /// The device extensions a display's window's swapchain is made with that it does not list,
    /// in their order: on the Vulkan platform `VK_KHR_swapchain` and the OS's own (`crate::os`'s
    /// `window_device_extensions`), on the rt platform `VK_KHR_swapchain`. Empty where it lists
    /// them all, as a device the window presents on does (the module doc's candidates).
    pub(crate) window_lacks: Vec<&'static CStr>,
}

/// The device a selection variable's value names ([`named`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Named {
    /// The device of these PCI vendor and device IDs.
    Ids(u32, u32),
    /// The device at this PCI domain, bus, device and function.
    Pci(u32, u32, u32, u32),
    /// No device by itself: `DRI_PRIME`'s index, the Nth GPU other than Mesa's layer's default,
    /// which the layer lists first.
    Index,
}

impl fmt::Display for Named {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Named::Ids(vendor, device) => write!(f, "{vendor:04x}:{device:04x}"),
            Named::Pci(domain, bus, device, function) => {
                write!(f, "pci-{domain:04x}_{bus:02x}_{device:02x}_{function:x}")
            }
            Named::Index => f.write_str("an index"),
        }
    }
}

/// What `value`, the value of the selection variable `var`, names, in the forms Mesa's
/// documentation of its layer's variables and the loader's code give: `vendor:device` in hex,
/// each part with or without `0x`, in any of them, with `MESA_VK_DEVICE_SELECT`'s and
/// `DRI_PRIME`'s `!` after it; and in `DRI_PRIME` also `pci-DOMAIN_BUS_DEVICE_FUNCTION` in hex,
/// or a decimal index. `None` where it is none of these, which names no device.
fn named(var: &str, value: &str) -> Option<Named> {
    let hex = |x: &str| {
        let x = (x.strip_prefix("0x").or_else(|| x.strip_prefix("0X"))).unwrap_or(x);
        (!x.is_empty() && x.len() <= 8 && x.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| u32::from_str_radix(x, 16).ok())
            .flatten()
    };
    let ids = |s: &str| {
        let (vendor, device) = s.split_once(':')?;
        Some(Named::Ids(hex(vendor)?, hex(device)?))
    };
    let value = value.trim();
    match var {
        "VK_LOADER_DEVICE_SELECT" => ids(value),
        "DRI_PRIME" => {
            let value = value.strip_suffix('!').unwrap_or(value);
            if let Some(tag) = value.strip_prefix("pci-") {
                let parts: Vec<u32> = tag.split('_').map(hex).collect::<Option<_>>()?;
                match parts.as_slice() {
                    &[domain, bus, device, function] => {
                        Some(Named::Pci(domain, bus, device, function))
                    }
                    _ => None,
                }
            } else if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) {
                Some(Named::Index)
            } else {
                ids(value)
            }
        }
        _ => ids(value.strip_suffix('!').unwrap_or(value)),
    }
}

/// A physical device's DRM nodes, by major and minor number (`VkPhysicalDeviceDrmPropertiesEXT`):
/// what a window system names a GPU by ([`compositor_gpu`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DrmNodes {
    /// Its primary node, where it has one.
    pub(crate) primary: Option<(u32, u32)>,
    /// Its render node, where it has one.
    pub(crate) render: Option<(u32, u32)>,
}

impl DrmNodes {
    /// Whether `node` is one of these.
    pub(crate) fn names(&self, node: (u32, u32)) -> bool {
        self.primary == Some(node) || self.render == Some(node)
    }
}

/// The GPU a window's compositor composites on, against the runtime's own
/// ([`compositor_gpu`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositorGpu {
    /// The runtime's own GPU, or none the probe can tell: the window presents from the
    /// runtime's device, as on a host of one GPU.
    Own,
    /// Another GPU the runtime makes devices of, by its registry ID: the window may present
    /// from a device of its own there.
    Other(u64),
}

/// The GPU the compositor that names DRM node `node` composites on, by what the probe read of
/// each physical device (`crate::os`'s `seen_devices`), against `own`, the registry
/// ID of the runtime's default device: [`CompositorGpu::Other`] only where `node` is a node of a
/// candidate (the module doc's) with a registry ID that is not `own`. Every other answer is
/// [`CompositorGpu::Own`]: no device lists the node (none lists `VK_EXT_physical_device_drm`, or
/// the node is a GPU the runtime cannot use), or it is the runtime's own.
pub fn compositor_gpu(node: (u32, u32), own: Option<u64>) -> CompositorGpu {
    compositor_gpu_of(crate::os::seen_devices(), node, own, Seen::registry_id)
}

/// [`compositor_gpu`] of the devices `seen`, their registry IDs as `registry_id` answers them.
fn compositor_gpu_of(
    seen: &[Seen],
    node: (u32, u32),
    own: Option<u64>,
    registry_id: impl Fn(&Seen) -> Option<u64>,
) -> CompositorGpu {
    let listed = (seen.iter())
        .filter(|s| s.candidate() && s.drm.is_some_and(|d| d.names(node)))
        .find_map(&registry_id);
    match listed {
        Some(id) if Some(id) != own => CompositorGpu::Other(id),
        _ => CompositorGpu::Own,
    }
}

impl Seen {
    /// Whether it is the device `named` names.
    fn is(&self, named: Named) -> bool {
        match named {
            Named::Ids(vendor, device) => (self.vendor_id, self.device_id) == (vendor, device),
            Named::Pci(domain, bus, device, function) => {
                self.pci == Some((domain, bus, device, function))
            }
            Named::Index => false,
        }
    }

    /// Its registry ID, the runtime's for the physical device of its UUID
    /// ([`listed_registry_id`]), where it has a UUID the runtime lists.
    pub(crate) fn registry_id(&self) -> Option<u64> {
        self.uuid.as_ref().and_then(listed_registry_id)
    }

    /// Whether the runtime makes a device of it: at Vulkan 1.2 or later, with a graphics and
    /// compute queue family and a UUID.
    pub(crate) fn usable(&self) -> bool {
        self.api_version >= vk::API_VERSION_1_2 && self.graphics_compute && self.uuid.is_some()
    }

    /// Whether it is a candidate (the module doc's): the runtime makes a device of it, and a
    /// display's window presents on it.
    pub(crate) fn candidate(&self) -> bool {
        self.usable() && self.window_lacks.is_empty()
    }

    /// Why a display's window cannot present on it, where it cannot: the window's device
    /// extensions it does not list.
    fn window_refusal(&self) -> Option<String> {
        let lacks: Vec<_> = (self.window_lacks.iter())
            .map(|name| name.to_string_lossy())
            .collect();
        (!lacks.is_empty()).then(|| {
            format!(
                "a display's window cannot present on it: it lists no {}",
                lacks.join(" and no ")
            )
        })
    }

    /// Whether it is a GPU: any kind but a CPU implementation.
    fn gpu(&self) -> bool {
        self.kind != vk::PhysicalDeviceType::CPU
    }

    /// The device as its log line names it.
    fn describe(&self) -> String {
        let kind = match self.kind {
            vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
            vk::PhysicalDeviceType::CPU => "CPU",
            _ => "other",
        };
        let not = if self.api_version < vk::API_VERSION_1_2 {
            ", not usable: below Vulkan 1.2".to_owned()
        } else if !self.graphics_compute {
            ", not usable: no graphics and compute queue".to_owned()
        } else if self.uuid.is_none() {
            ", not usable: no device UUID".to_owned()
        } else if let Some(why) = self.window_refusal() {
            format!(", not a candidate: {why}")
        } else {
            String::new()
        };
        format!(
            "{} ({kind}, {:04x}:{:04x}, Vulkan {}.{}, driver {:#x}, {:.2} GiB device-local){not}",
            self.name,
            self.vendor_id,
            self.device_id,
            vk::api_version_major(self.api_version),
            vk::api_version_minor(self.api_version),
            self.driver_version,
            gib(self.device_local),
        )
    }
}

/// Its log line's description and its registry ID: ash's types answer no `Debug` in this build.
impl fmt::Debug for Seen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{:x?}]", self.describe(), self.registry_id())
    }
}

/// `bytes` in GiB, for a log line.
#[allow(clippy::cast_precision_loss)] // two decimals of a heap's size
fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

/// Why [`choose`] picked its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// It is the only candidate.
    Only,
    /// It is the only GPU among the candidates, the others CPU implementations.
    OnlyGpu,
    /// The host's user picked it through this variable, which names it so.
    UserChose(&'static str, Named),
    /// The host's user set this variable to an index, which names no device by itself: the
    /// loader's first candidate, which Mesa's device-select layer lists first for it.
    UserIndex(&'static str),
    /// The host's user set selection variables, the first of them this one, and none names a
    /// candidate: the loader's first candidate.
    UserNamedNone(&'static str),
    /// It ran the workload fastest, timed now.
    Measured,
    /// It ran the workload fastest when the same GPUs on the same drivers were timed before.
    Cached,
    /// It is first in the runtime's own order.
    RuntimeOrder,
}

impl Why {
    /// The reason as the log line gives it.
    fn describe(self) -> String {
        match self {
            Why::Only => "the only candidate".to_owned(),
            Why::OnlyGpu => "the only GPU of the candidates, beside CPU devices".to_owned(),
            Why::Measured => "the fastest of the GPUs at the choice's workload".to_owned(),
            Why::Cached => "the fastest of the GPUs at the choice's workload, as measured \
                            before on the same GPUs and drivers"
                .to_owned(),
            Why::UserChose(var, named) => {
                format!("{var} names it ({named}), as the host's user picked")
            }
            Why::UserIndex(var) => format!(
                "{var} is an index, which names no device by itself, so the loader's first \
                 device, which Mesa's device-select layer lists first for it"
            ),
            Why::UserNamedNone(var) => format!(
                "{var} is set, but no selection variable set names a candidate here, so the \
                 loader's first candidate"
            ),
            Why::RuntimeOrder => "first of the candidates in the runtime's own order (discrete, \
                                  integrated, virtual, CPU)"
                .to_owned(),
        }
    }
}

/// The device of `seen` (the loader's order) the library opens, and why: where `user` lists the
/// selection variables the host's user set, with what each names ([`user_selection`]), the
/// candidate the first of them to name one names, or the loader's first candidate where an index
/// comes before it or none names one (the module doc); else the one candidate, or the one GPU
/// among the candidates; else, of several GPUs, the one `measure` answers for them (their
/// indices in `seen`), or the first candidate of the runtime's own order where it answers none
/// (the module doc). `measure` is asked only of two GPUs or more. `None` where no device is a
/// candidate.
pub(crate) fn choose(
    seen: &[Seen],
    user: &[(&'static str, Option<Named>)],
    measure: impl FnOnce(&[usize]) -> Option<(usize, Why)>,
) -> Option<(usize, Why)> {
    let candidates: Vec<usize> = (seen.iter().enumerate())
        .filter(|(_, s)| s.candidate())
        .map(|(at, _)| at)
        .collect();
    if let Some(&(first, _)) = user.first() {
        let loaders_first = |why: Why| candidates.first().map(|&at| (at, why));
        for &(var, named) in user {
            match named {
                Some(Named::Index) => return loaders_first(Why::UserIndex(var)),
                Some(named) => {
                    let listed = candidates.iter().find(|&&at| seen[at].is(named));
                    if let Some(&at) = listed {
                        return Some((at, Why::UserChose(var, named)));
                    }
                }
                None => {}
            }
        }
        return loaders_first(Why::UserNamedNone(first));
    }
    if let &[only] = candidates.as_slice() {
        return Some((only, Why::Only));
    }
    let gpus: Vec<usize> = (candidates.iter().copied())
        .filter(|&at| seen.get(at).is_some_and(Seen::gpu))
        .collect();
    match gpus.as_slice() {
        [only] => Some((*only, Why::OnlyGpu)),
        [_, _, ..] => measure(&gpus),
        [] => None,
    }
    .or_else(|| first_of_kinds(seen, Seen::candidate).map(|at| (at, Why::RuntimeOrder)))
}

/// Of the GPUs `gpus` (indices in `seen`), the one that ran the workload fastest: the verdict kept
/// at `cache` for them, where it reads (`verdict.rs`), else the fastest of the times `time` answers
/// for each, kept at `cache` for the next process; `None` where `time` answers none for any of
/// them, and then nothing is kept (the module doc). The times and what they cost in all are
/// logged.
pub(crate) fn measured(
    seen: &[Seen],
    gpus: &[usize],
    cache: Option<&Path>,
    mut time: impl FnMut(usize, &Seen) -> Option<Duration>,
) -> Option<(usize, Why)> {
    let timed: Vec<(usize, &Seen)> = (gpus.iter())
        .map(|&at| Some((at, seen.get(at)?)))
        .collect::<Option<_>>()?;
    let key = verdict::key(&timed.iter().map(|(_, s)| *s).collect::<Vec<_>>())?;
    let kept = cache.and_then(|path| verdict::read(path, &key));
    if let Some(&(at, _)) = timed.iter().find(|(_, s)| kept.is_some() && s.uuid == kept) {
        return Some((at, Why::Cached));
    }
    let clock = crate::os::OsClock::new();
    let start = clock.now_mono();
    let mut times = Vec::with_capacity(timed.len());
    for &(at, device) in &timed {
        let Some(took) = time(at, device) else {
            log::warn!(
                target: TARGET,
                "the choice's workload failed on Vulkan device {at}: the runtime's own order \
                 stands, and no verdict is kept"
            );
            return None;
        };
        times.push((at, device, took));
    }
    let &(fastest, device, _) = times.iter().min_by_key(|(_, _, took)| *took)?;
    log::info!(
        target: TARGET,
        "timed {} GPUs for the choice in {:.1} ms in all",
        times.len(),
        timing::millis(clock.now_mono().saturating_duration_since(start))
    );
    if let (Some(path), Some(pick)) = (cache, device.uuid) {
        let rows: Vec<([u8; 16], Duration)> = (times.iter())
            .filter_map(|(_, s, took)| Some((s.uuid?, *took)))
            .collect();
        if let Err(e) = verdict::write(path, &key, &pick, &rows) {
            log::warn!(
                target: TARGET,
                "the choice's verdict was not kept at {} ({e}): the next process times the GPUs \
                 again",
                path.display()
            );
        }
    }
    Some((fastest, Why::Measured))
}

/// The file the measured pick's verdict is kept in: `device-checks/gpu-choice` under the library's
/// cache directory (`crate::platform::cache_directory`); none where there is no such directory.
fn verdict_file() -> Option<PathBuf> {
    Some(
        crate::platform::cache_directory()?
            .join("device-checks")
            .join("gpu-choice"),
    )
}

/// The device of `seen` the runtime tries first for its default device: of those it makes a
/// device of ([`Seen::usable`]), the first of its order of kinds (`Device::system_default_with`,
/// read from code). Where it refuses that one, as a device that lacks an extension its options
/// name, it makes the next, so a pick that is not this device is opened by its registry ID
/// ([`to_open`]).
fn runtime_first(seen: &[Seen]) -> Option<usize> {
    first_of_kinds(seen, Seen::usable)
}

/// Of the devices of `seen` that `keep` keeps, the first of the runtime's order of kinds
/// (discrete, integrated, virtual, CPU, any other), then the loader's.
fn first_of_kinds(seen: &[Seen], keep: impl Fn(&Seen) -> bool) -> Option<usize> {
    let preference = |kind: vk::PhysicalDeviceType| match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 3,
        _ => 4,
    };
    seen.iter()
        .enumerate()
        .filter(|&(_, s)| keep(s))
        .min_by_key(|&(at, s)| (preference(s.kind), at))
        .map(|(at, _)| at)
}

/// The device `picked` of `seen` to open by its registry ID ([`Seen::registry_id`]), where the
/// runtime tries another first for its default device ([`runtime_first`]): `None` where nothing
/// was picked, or the pick is the device the runtime tries first.
fn to_open(seen: &[Seen], picked: Option<usize>) -> Option<usize> {
    let at = picked?;
    (Some(at) != runtime_first(seen) && at < seen.len()).then_some(at)
}

/// The [`SELECTION_VARIABLES`] `var` answers a value that is not empty for, in their order, each
/// with what its value names ([`named`]).
fn user_selection(var: impl Fn(&str) -> Option<OsString>) -> Vec<(&'static str, Option<Named>)> {
    (SELECTION_VARIABLES.into_iter())
        .filter_map(|name| {
            let value = var(name).filter(|value| !value.is_empty())?;
            Some((name, named(name, &value.to_string_lossy())))
        })
        .collect()
}

/// The registry ID of the device the process opens as its default device, where it is not the one
/// the runtime makes its default device of anyway: read once from the probe's devices and the
/// environment, and logged (the module doc). `None` where the runtime's default device is the
/// pick, as on a host of one GPU, or where nothing was probed.
pub(crate) fn chosen() -> Option<u64> {
    static CHOSEN: LazyLock<Option<u64>> = LazyLock::new(|| {
        let seen = crate::os::seen_devices();
        for (at, device) in seen.iter().enumerate() {
            log::info!(target: TARGET, "Vulkan device {at}: {}", device.describe());
        }
        let file = verdict_file();
        let picked = choose(
            seen,
            &user_selection(|name| std::env::var_os(name)),
            |gpus| measured(seen, gpus, file.as_deref(), timing::time_on),
        );
        log_choice(seen, picked);
        let at = to_open(seen, picked.map(|(at, _)| at))?;
        seen.get(at)?.registry_id()
    });
    *CHOSEN
}

/// The device of `seen` (an instance's physical devices that meet metal2vulkan-rt's floor, in the
/// loader's order) the rt platform opens (`crate::rt_platform`), logged as [`chosen`] logs its
/// pick: the device the host's user's selection variables name, else the one GPU, else the first
/// of the runtime's own order (discrete, integrated, virtual, CPU). No GPU is timed, as the
/// workload runs on metal2vulkan-runtime (`timing.rs`), which the rt platform does not use, and
/// no verdict is read or kept. `None` where no device is a candidate.
#[cfg(feature = "rt")]
pub(crate) fn chosen_untimed(seen: &[Seen]) -> Option<usize> {
    for (at, device) in seen.iter().enumerate() {
        log::info!(target: TARGET, "Vulkan device {at}: {}", device.describe());
    }
    let picked = choose(seen, &user_selection(|name| std::env::var_os(name)), |_| {
        None
    });
    log_choice(seen, picked);
    picked.map(|(at, _)| at)
}

/// Logs the device of `seen` `picked`, with why, at info ([`choice_line`]).
fn log_choice(seen: &[Seen], picked: Option<(usize, Why)>) {
    log::info!(target: TARGET, "{}", choice_line(seen, picked));
}

/// The log line of the device of `seen` `picked`, with why, and each device the runtime makes
/// that was passed over as no candidate, with why (the module doc).
fn choice_line(seen: &[Seen], picked: Option<(usize, Why)>) -> String {
    let mut line = match picked.and_then(|(at, why)| Some((at, seen.get(at)?, why))) {
        Some((at, device, why)) => format!(
            "the library's GPU is Vulkan device {at}, {}: {}",
            device.name,
            why.describe()
        ),
        None => "no candidate was listed: the runtime's default device".to_owned(),
    };
    for (at, device) in seen.iter().enumerate() {
        if let Some(why) = device.window_refusal().filter(|_| device.usable()) {
            line.push_str(&format!(
                "; passed over Vulkan device {at}, {}: {why}",
                device.name
            ));
        }
    }
    line
}

/// The registry ID the runtime answers for the physical device whose `deviceUUID` is `uuid`,
/// which `Device::with_registry_id` opens it by: that of the first of the runtime's identities of
/// its physical devices (`Device::identities_with`, with [`crate::platform::device_options`], in
/// its order) of that UUID. The identities are read once a process, at the first ask, which only
/// a pick the runtime would not make its default device of, or a GPU's timing, makes: a host of
/// one GPU never reads them. `None` where the runtime lists no physical device of the UUID, or
/// refuses to list them, which is logged: such a device is not opened by ID.
fn listed_registry_id(uuid: &[u8; 16]) -> Option<u64> {
    static LISTED: LazyLock<Vec<rt::PhysicalDeviceIdentity>> = LazyLock::new(|| {
        rt::Device::identities_with(crate::platform::device_options()).unwrap_or_else(|e| {
            log::warn!(
                target: TARGET,
                "the runtime listed no identities of its physical devices ({e}): no device is \
                 opened by its registry ID"
            );
            Vec::new()
        })
    });
    (LISTED.iter())
        .find(|identity| identity.device_uuid == *uuid)
        .map(|identity| identity.registry_id)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// A candidate of `kind` with `device_local` bytes of device-local memory, its UUID's first
    /// byte `id`.
    pub(crate) fn seen(
        name: &str,
        kind: vk::PhysicalDeviceType,
        device_local: u64,
        id: u8,
    ) -> Seen {
        Seen {
            name: name.to_owned(),
            kind,
            vendor_id: 0,
            device_id: 0,
            api_version: vk::API_VERSION_1_3,
            driver_version: 1,
            graphics_compute: true,
            device_local,
            uuid: Some(std::array::from_fn(|i| if i == 0 { id } else { 0 })),
            drm: None,
            pci: None,
            window_lacks: Vec::new(),
        }
    }

    /// `seen` with DRM nodes `card` (primary) and `render`.
    fn with_nodes(seen: Seen, card: u32, render: u32) -> Seen {
        Seen {
            drm: Some(DrmNodes {
                primary: Some((226, card)),
                render: Some((226, render)),
            }),
            ..seen
        }
    }

    /// The compositor's GPU is another only where a candidate other than the runtime's own lists
    /// the node it names, primary or render; a node no device lists, the runtime's own node, and a
    /// device the runtime cannot make all answer the runtime's own GPU
    #[test]
    fn the_compositors_gpu_is_another_only_where_a_candidate_lists_its_node() {
        let id = |s: &Seen| s.uuid.map(|u| u64::from(u[0]));
        let list = [
            with_nodes(discrete(24, 1), 0, 128),
            with_nodes(integrated(2, 2), 1, 129),
        ];
        // The runtime on the discrete GPU, the compositor on the integrated one.
        assert_eq!(
            compositor_gpu_of(&list, (226, 129), Some(1), id),
            CompositorGpu::Other(2)
        );
        assert_eq!(
            compositor_gpu_of(&list, (226, 1), Some(1), id),
            CompositorGpu::Other(2)
        );
        // The compositor on the runtime's own GPU.
        assert_eq!(
            compositor_gpu_of(&list, (226, 128), Some(1), id),
            CompositorGpu::Own
        );
        assert_eq!(
            compositor_gpu_of(&list, (226, 0), Some(1), id),
            CompositorGpu::Own
        );
        // A node no device lists.
        assert_eq!(
            compositor_gpu_of(&list, (226, 130), Some(1), id),
            CompositorGpu::Own
        );
        // No device lists its nodes (no `VK_EXT_physical_device_drm`).
        let bare = [discrete(24, 1), integrated(2, 2)];
        assert_eq!(
            compositor_gpu_of(&bare, (226, 129), Some(1), id),
            CompositorGpu::Own
        );
        // A device the runtime cannot make.
        let old = Seen {
            api_version: vk::API_VERSION_1_1,
            ..with_nodes(integrated(2, 2), 1, 129)
        };
        let list = [with_nodes(discrete(24, 1), 0, 128), old];
        assert_eq!(
            compositor_gpu_of(&list, (226, 129), Some(1), id),
            CompositorGpu::Own
        );
    }

    pub(crate) fn integrated(gib: u64, id: u8) -> Seen {
        seen(
            "integrated",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            gib * GIB,
            id,
        )
    }

    pub(crate) fn discrete(gib: u64, id: u8) -> Seen {
        seen(
            "discrete",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            gib * GIB,
            id,
        )
    }

    pub(crate) fn cpu(id: u8) -> Seen {
        seen("llvmpipe", vk::PhysicalDeviceType::CPU, 29 * GIB, id)
    }

    /// A `measure` that must not be asked.
    fn unasked(gpus: &[usize]) -> Option<(usize, Why)> {
        panic!("measured {gpus:?}")
    }

    /// Of several GPUs, the one the measurement answers, asked of the GPUs alone and never of
    /// llvmpipe; by ID where the runtime alone would make another
    #[test]
    fn several_gpus_are_measured_and_the_cpu_never() {
        let list = [integrated(11, 1), discrete(4, 2), cpu(3)];
        let mut asked = Vec::new();
        let picked = choose(&list, &[], |gpus| {
            asked = gpus.to_vec();
            Some((0, Why::Measured))
        });
        assert_eq!(picked, Some((0, Why::Measured)));
        assert_eq!(asked, [0, 1]);
        assert_eq!(to_open(&list, Some(0)), Some(0));
        assert_eq!(to_open(&list, Some(1)), None);
    }

    /// Where the measurement answers nothing, the runtime's own order stands: a discrete GPU
    /// before an integrated one wherever the loader lists it, opened as before
    #[test]
    fn an_unmeasured_choice_keeps_the_runtimes_order() {
        let list = [integrated(11, 1), discrete(4, 2), cpu(3)];
        assert_eq!(choose(&list, &[], |_| None), Some((1, Why::RuntimeOrder)));
        assert_eq!(to_open(&list, Some(1)), None);
    }

    /// One GPU beside llvmpipe is picked unmeasured, as the runtime picks it, and two CPU devices
    /// keep the runtime's order unmeasured
    #[test]
    fn one_gpu_is_never_measured() {
        let list = [cpu(1), discrete(16, 2)];
        assert_eq!(choose(&list, &[], unasked), Some((1, Why::OnlyGpu)));
        assert_eq!(to_open(&list, Some(1)), None);
        let list = [cpu(1), cpu(2)];
        assert_eq!(choose(&list, &[], unasked), Some((0, Why::RuntimeOrder)));
    }

    /// A host of one candidate opens it as the runtime does, with nothing to open by ID
    #[test]
    fn one_candidate_is_opened_as_before() {
        for list in [vec![discrete(4, 1)], vec![integrated(2, 1)], vec![cpu(1)]] {
            assert_eq!(choose(&list, &[], unasked), Some((0, Why::Only)));
            assert_eq!(to_open(&list, Some(0)), None);
        }
        assert_eq!(choose(&[], &[], unasked), None);
        assert_eq!(to_open(&[], None), None);
    }

    /// `seen` with PCI vendor and device IDs `ids` at PCI bus `bus` (domain 0, device 0,
    /// function 0).
    fn with_ids(seen: Seen, ids: (u32, u32), bus: u32) -> Seen {
        Seen {
            vendor_id: ids.0,
            device_id: ids.1,
            pci: Some((0, bus, 0, 0)),
            ..seen
        }
    }

    /// um870's devices in the order Mesa's device-select layer lists them with no variable of its
    /// own set (m3-max-30538): the Radeon 780M, the RTX 3090, lavapipe.
    fn um870() -> [Seen; 3] {
        [
            with_ids(integrated(11, 1), (0x1002, 0x1900), 0xc5),
            with_ids(discrete(24, 2), (0x10de, 0x2204), 0x67),
            Seen {
                vendor_id: 0x10005,
                ..cpu(3)
            },
        ]
    }

    /// The user's variable picks the candidate it names, wherever the loader lists it,
    /// unmeasured: `VK_LOADER_DEVICE_SELECT`'s too, which Mesa's layer leaves behind its own
    /// default; by ID where the runtime alone would make another
    #[test]
    fn the_users_pick_is_the_device_its_variable_names() {
        let list = um870();
        let ids = |v, d| Some(Named::Ids(v, d));
        let loader = [("VK_LOADER_DEVICE_SELECT", ids(0x10de, 0x2204))];
        let picked = choose(&list, &loader, unasked);
        let named = Named::Ids(0x10de, 0x2204);
        assert_eq!(
            picked,
            Some((1, Why::UserChose("VK_LOADER_DEVICE_SELECT", named)))
        );
        assert_eq!(to_open(&list, Some(1)), None);
        let mesa = [("MESA_VK_DEVICE_SELECT", ids(0x1002, 0x1900))];
        assert_eq!(choose(&list, &mesa, unasked).map(|(at, _)| at), Some(0));
        assert_eq!(to_open(&list, Some(0)), Some(0));
        let lavapipe = [("MESA_VK_DEVICE_SELECT", ids(0x10005, 0))];
        assert_eq!(choose(&list, &lavapipe, unasked).map(|(at, _)| at), Some(2));
        let pci = Named::Pci(0, 0x67, 0, 0);
        let prime = [("DRI_PRIME", Some(pci))];
        assert_eq!(
            choose(&list, &prime, unasked),
            Some((1, Why::UserChose("DRI_PRIME", pci)))
        );
        // Of two devices of the IDs, the first; a device the runtime cannot make is no pick.
        let not_usable = Seen {
            api_version: vk::API_VERSION_1_1,
            ..with_ids(discrete(24, 9), (0x10de, 0x2204), 0x01)
        };
        let list = [
            not_usable,
            with_ids(integrated(11, 1), (0x1002, 0x1900), 0xc5),
            with_ids(discrete(24, 2), (0x10de, 0x2204), 0x67),
            with_ids(discrete(24, 4), (0x10de, 0x2204), 0x68),
        ];
        assert_eq!(choose(&list, &loader, unasked).map(|(at, _)| at), Some(2));
    }

    /// A variable that names no candidate is passed over for the next one set; where none names
    /// one, the loader's first candidate, said so. An index names no device: the loader's first
    /// candidate, which Mesa's layer lists first for it, even before a later variable that names
    /// one
    #[test]
    fn a_variable_that_names_no_candidate_is_passed_over() {
        let list = um870();
        let nothing = Some(Named::Ids(0xdead, 0xbeef));
        let pci = Named::Pci(0, 0x67, 0, 0);
        let user = [("MESA_VK_DEVICE_SELECT", nothing), ("DRI_PRIME", Some(pci))];
        assert_eq!(
            choose(&list, &user, unasked),
            Some((1, Why::UserChose("DRI_PRIME", pci)))
        );
        let user = [
            ("MESA_VK_DEVICE_SELECT", nothing),
            ("VK_LOADER_DEVICE_SELECT", None),
        ];
        assert_eq!(
            choose(&list, &user, unasked),
            Some((0, Why::UserNamedNone("MESA_VK_DEVICE_SELECT")))
        );
        let user = [
            ("DRI_PRIME", Some(Named::Index)),
            ("VK_LOADER_DEVICE_SELECT", Some(Named::Ids(0x10de, 0x2204))),
        ];
        assert_eq!(
            choose(&list, &user, unasked),
            Some((0, Why::UserIndex("DRI_PRIME")))
        );
        assert_eq!(choose(&[], &user, unasked), None);
    }

    /// A variable's value names a device by its IDs in hex, with or without `0x` and in either
    /// case, `MESA_VK_DEVICE_SELECT`'s and `DRI_PRIME`'s with a `!` after them; `DRI_PRIME` also
    /// by its PCI address, or an index; anything else names nothing
    #[test]
    fn a_variables_value_names_a_device_by_its_ids_its_address_or_an_index() {
        let ids = |v, d| Some(Named::Ids(v, d));
        for var in SELECTION_VARIABLES {
            assert_eq!(named(var, "10de:2204"), ids(0x10de, 0x2204), "{var}");
            assert_eq!(named(var, "0x10DE:0x2204"), ids(0x10de, 0x2204), "{var}");
            assert_eq!(named(var, "10005:0"), ids(0x10005, 0), "{var}");
            for nothing in ["", "10de", "10de:", ":2204", "nvidia", "10de:22g4"] {
                assert_eq!(named(var, nothing), None, "{var} {nothing:?}");
            }
        }
        assert_eq!(
            named("MESA_VK_DEVICE_SELECT", "1002:15bf!"),
            ids(0x1002, 0x15bf)
        );
        assert_eq!(named("DRI_PRIME", "1002:15bf!"), ids(0x1002, 0x15bf));
        assert_eq!(named("VK_LOADER_DEVICE_SELECT", "1002:15bf!"), None);
        assert_eq!(
            named("DRI_PRIME", "pci-0000_c5_00_0"),
            Some(Named::Pci(0, 0xc5, 0, 0))
        );
        assert_eq!(named("DRI_PRIME", "pci-0000_c5_00"), None);
        assert_eq!(named("DRI_PRIME", "1"), Some(Named::Index));
        assert_eq!(named("DRI_PRIME", "0"), Some(Named::Index));
        assert_eq!(named("MESA_VK_DEVICE_SELECT", "1"), None);
        assert_eq!(Named::Pci(0, 0xc5, 0, 0).to_string(), "pci-0000_c5_00_0");
        assert_eq!(Named::Ids(0x8086, 0x46a6).to_string(), "8086:46a6");
    }

    /// The user's pick is read from every selection variable set to a value, in Mesa's layer's
    /// order, then the loader's, each with what it names; an empty value is no pick
    #[test]
    fn the_users_pick_is_read_from_the_environment() {
        let env = |set: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                set.iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| OsString::from(v))
            }
        };
        assert!(user_selection(env(&[])).is_empty());
        assert!(user_selection(env(&[("MESA_VK_DEVICE_SELECT", "")])).is_empty());
        assert_eq!(
            user_selection(env(&[("MESA_VK_DEVICE_SELECT", "1002:15bf!")])),
            [("MESA_VK_DEVICE_SELECT", Some(Named::Ids(0x1002, 0x15bf)))]
        );
        assert_eq!(
            user_selection(env(&[
                ("VK_LOADER_DEVICE_SELECT", "0x1002:0x699f"),
                ("DRI_PRIME", "1")
            ])),
            [
                ("DRI_PRIME", Some(Named::Index)),
                ("VK_LOADER_DEVICE_SELECT", Some(Named::Ids(0x1002, 0x699f)))
            ]
        );
        assert!(user_selection(env(&[("VK_LOADER_DEBUG", "all")])).is_empty());
    }

    /// A device below Vulkan 1.2, without a graphics and compute queue, or without a UUID is no
    /// candidate, whatever its kind
    #[test]
    fn a_device_the_runtime_cannot_make_is_no_candidate() {
        let old = Seen {
            api_version: vk::API_VERSION_1_1,
            ..discrete(24, 1)
        };
        let no_queue = Seen {
            graphics_compute: false,
            ..discrete(24, 2)
        };
        let no_id = Seen {
            uuid: None,
            ..discrete(24, 3)
        };
        let list = [old, no_queue, no_id, integrated(2, 4)];
        assert_eq!(choose(&list, &[], unasked), Some((3, Why::Only)));
        assert_eq!(runtime_first(&list), Some(3));
        assert_eq!(choose(&list[..3], &[], unasked), None);
    }

    /// `seen` named `name`, of PCI vendor and device IDs `ids`, lacking the window's device
    /// extensions `lacks`.
    fn lacking(seen: Seen, name: &str, ids: (u32, u32), lacks: &[&'static CStr]) -> Seen {
        Seen {
            name: name.to_owned(),
            vendor_id: ids.0,
            device_id: ids.1,
            window_lacks: lacks.to_vec(),
            ..seen
        }
    }

    /// The M3 Max's devices with Mesa's KosmicKrisp beside MoltenVK, in the loader's order
    /// (m3-max-57095): KosmicKrisp, which lists no `VK_EXT_metal_objects`, then MoltenVK, both
    /// integrated.
    fn kosmickrisp_then_moltenvk() -> [Seen; 2] {
        let metal_objects = ash::ext::metal_objects::NAME;
        [
            lacking(
                integrated(51, 1),
                "kosmickrisp",
                (0x106b, 0x64),
                &[metal_objects],
            ),
            lacking(integrated(64, 2), "moltenvk", (0x106b, 0x1b00_0209), &[]),
        ]
    }

    /// A device a display's window cannot present on is no candidate, whatever its kind and
    /// wherever the loader lists it: never picked, timed or the compositor's GPU, on every OS by
    /// the same rule; the pick is opened by ID where the runtime would try such a device first,
    /// and alone it leaves no pick
    #[test]
    fn a_device_the_window_cannot_present_on_is_no_candidate() {
        let list = kosmickrisp_then_moltenvk();
        assert!(list[0].usable() && !list[0].candidate() && list[1].candidate());
        assert_eq!(choose(&list, &[], unasked), Some((1, Why::Only)));
        assert_eq!(runtime_first(&list), Some(0));
        assert_eq!(to_open(&list, Some(1)), Some(1));
        assert_eq!(choose(&list[..1], &[], unasked), None);
        // Of three GPUs, the two the window presents on are timed; where the timing answers
        // nothing, the first candidate of the runtime's order.
        let three = [list[0].clone(), list[1].clone(), integrated(8, 3)];
        let mut asked = Vec::new();
        let picked = choose(&three, &[], |gpus| {
            asked = gpus.to_vec();
            None
        });
        assert_eq!(
            (picked, asked.as_slice()),
            (Some((1, Why::RuntimeOrder)), &[1, 2][..])
        );
        // A GPU without `VK_KHR_swapchain`, which every OS's window takes, beside llvmpipe.
        let swapchain = ash::khr::swapchain::NAME;
        let headless = lacking(discrete(24, 4), "headless", (0x10de, 0x2204), &[swapchain]);
        let list = [headless.clone(), cpu(5)];
        assert_eq!(choose(&list, &[], unasked), Some((1, Why::Only)));
        assert_eq!(to_open(&list, Some(1)), Some(1));
        // The compositor on such a GPU composites on none the runtime opens for it.
        let id = |s: &Seen| s.uuid.map(|u| u64::from(u[0]));
        let list = [
            with_nodes(discrete(24, 1), 0, 128),
            with_nodes(headless, 1, 129),
        ];
        assert_eq!(
            compositor_gpu_of(&list, (226, 129), Some(1), id),
            CompositorGpu::Own
        );
    }

    /// A selection variable that names a device the window cannot present on is passed over,
    /// as one that names a device below Vulkan 1.2 is: the next variable's device, else the
    /// loader's first candidate
    #[test]
    fn a_variable_naming_a_device_the_window_cannot_present_on_is_passed_over() {
        let list = kosmickrisp_then_moltenvk();
        let kosmickrisp = ("MESA_VK_DEVICE_SELECT", Some(Named::Ids(0x106b, 0x64)));
        assert_eq!(
            choose(&list, &[kosmickrisp], unasked),
            Some((1, Why::UserNamedNone("MESA_VK_DEVICE_SELECT")))
        );
        let moltenvk = Named::Ids(0x106b, 0x1b00_0209);
        let user = [kosmickrisp, ("VK_LOADER_DEVICE_SELECT", Some(moltenvk))];
        assert_eq!(
            choose(&list, &user, unasked),
            Some((1, Why::UserChose("VK_LOADER_DEVICE_SELECT", moltenvk)))
        );
        assert_eq!(choose(&list[..1], &[kosmickrisp], unasked), None);
    }

    /// The choice's line names the pick and why, then each device the runtime makes that was
    /// passed over for the window, with the extensions it lacks; a device the runtime cannot
    /// make is not named there, and a device's own line says why it is no candidate
    #[test]
    fn the_choices_line_names_each_device_passed_over_for_the_window() {
        let list = kosmickrisp_then_moltenvk();
        let lacks = "a display's window cannot present on it: it lists no VK_EXT_metal_objects";
        assert_eq!(
            choice_line(&list, Some((1, Why::Only))),
            format!(
                "the library's GPU is Vulkan device 1, moltenvk: the only candidate; passed over \
                 Vulkan device 0, kosmickrisp: {lacks}"
            )
        );
        assert_eq!(
            choice_line(&list[..1], None),
            format!(
                "no candidate was listed: the runtime's default device; passed over Vulkan \
                 device 0, kosmickrisp: {lacks}"
            )
        );
        assert!(
            list[0]
                .describe()
                .ends_with(&format!(", not a candidate: {lacks}"))
        );
        assert!(!list[1].describe().contains("not"));
        let old = Seen {
            api_version: vk::API_VERSION_1_1,
            ..list[0].clone()
        };
        assert_eq!(
            choice_line(&[old.clone(), list[1].clone()], Some((1, Why::Only))),
            "the library's GPU is Vulkan device 1, moltenvk: the only candidate"
        );
        assert!(old.describe().ends_with(", not usable: below Vulkan 1.2"));
        let both = [ash::khr::swapchain::NAME, ash::ext::metal_objects::NAME];
        let bare = lacking(integrated(2, 3), "bare", (0, 0), &both);
        assert_eq!(
            bare.window_refusal().as_deref(),
            Some(
                "a display's window cannot present on it: it lists no VK_KHR_swapchain and no \
                 VK_EXT_metal_objects"
            )
        );
    }

    /// A timer of fixed times, in milliseconds by index in `seen`, counting its calls; a time of
    /// 0 fails.
    fn timer(
        ms: &[u64],
        calls: &std::cell::Cell<usize>,
    ) -> impl FnMut(usize, &Seen) -> Option<Duration> {
        move |at, _| {
            calls.set(calls.get() + 1);
            let ms = *ms.get(at)?;
            (ms > 0).then(|| Duration::from_millis(ms))
        }
    }

    /// The fastest GPU is picked and its verdict kept; the next choice of the same GPUs on the
    /// same drivers reads it and times nothing
    #[test]
    fn the_fastest_gpu_is_picked_and_kept() {
        let scratch = verdict::tests::Scratch::new("kept");
        let file = scratch.file();
        let list = [integrated(11, 1), discrete(4, 2)];
        let calls = std::cell::Cell::new(0);
        let picked = measured(&list, &[0, 1], Some(&file), timer(&[24, 40], &calls));
        assert_eq!((picked, calls.get()), (Some((0, Why::Measured)), 2));
        let again = measured(&list, &[0, 1], Some(&file), timer(&[99, 1], &calls));
        assert_eq!((again, calls.get()), (Some((0, Why::Cached)), 2));
        let fast_discrete = measured(
            &[discrete(24, 3), integrated(11, 1)],
            &[0, 1],
            None,
            timer(&[5, 25], &calls),
        );
        assert_eq!(fast_discrete, Some((0, Why::Measured)));
    }

    /// A driver change, another GPU or a corrupt verdict times the GPUs again, and the new verdict
    /// is kept in place of the old
    #[test]
    fn a_changed_host_or_a_corrupt_verdict_is_measured_again() {
        let scratch = verdict::tests::Scratch::new("changed");
        let file = scratch.file();
        let list = [integrated(11, 1), discrete(4, 2)];
        let calls = std::cell::Cell::new(0);
        measured(&list, &[0, 1], Some(&file), timer(&[24, 40], &calls));
        let updated = [
            integrated(11, 1),
            Seen {
                driver_version: 2,
                ..discrete(4, 2)
            },
        ];
        let picked = measured(&updated, &[0, 1], Some(&file), timer(&[24, 12], &calls));
        assert_eq!((picked, calls.get()), (Some((1, Why::Measured)), 4));
        let read = measured(&updated, &[0, 1], Some(&file), timer(&[1, 99], &calls));
        assert_eq!((read, calls.get()), (Some((1, Why::Cached)), 4));
        let another = [integrated(11, 1), discrete(24, 3)];
        let picked = measured(&another, &[0, 1], Some(&file), timer(&[24, 5], &calls));
        assert_eq!((picked, calls.get()), (Some((1, Why::Measured)), 6));
        std::fs::write(&file, "reims-vgpu gpu choice 1\nkey \u{0}").unwrap();
        let picked = measured(&another, &[0, 1], Some(&file), timer(&[3, 5], &calls));
        assert_eq!((picked, calls.get()), (Some((0, Why::Measured)), 8));
        let read = measured(&another, &[0, 1], Some(&file), timer(&[9, 1], &calls));
        assert_eq!((read, calls.get()), (Some((0, Why::Cached)), 8));
    }

    /// Where any GPU's workload fails, nothing is picked and nothing kept, so the runtime's own
    /// order stands and the next process times them again
    #[test]
    fn a_failed_measurement_picks_and_keeps_nothing() {
        let scratch = verdict::tests::Scratch::new("failed");
        let file = scratch.file();
        let list = [integrated(11, 1), discrete(4, 2)];
        let calls = std::cell::Cell::new(0);
        assert_eq!(
            measured(&list, &[0, 1], Some(&file), timer(&[24, 0], &calls)),
            None
        );
        assert!(!file.exists());
        let picked = choose(&list, &[], |gpus| {
            measured(&list, gpus, Some(&file), timer(&[0, 40], &calls))
        });
        assert_eq!(picked, Some((1, Why::RuntimeOrder)));
        assert!(!file.exists());
    }

    /// The registry ID the choice answered before it read the runtime's identities: the UUID's
    /// two halves, little-endian, exclusive-ored, and 1 where that is 0.
    fn folded(uuid: &[u8; 16]) -> u64 {
        let (low, high) = uuid.split_at(8);
        let half = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().unwrap_or_default());
        match half(low) ^ half(high) {
            0 => 1,
            id => id,
        }
    }

    /// The fold of the UUID the choice answered before: its halves exclusive-ored, little-endian;
    /// 1 where they cancel
    #[test]
    fn the_old_registry_id_folds_the_uuid() {
        let mut uuid = [0u8; 16];
        uuid[4] = 0xc5;
        assert_eq!(folded(&uuid), 0xc5_0000_0000);
        uuid[12] = 0xc5;
        assert_eq!(folded(&uuid), 1);
        let uuid: [u8; 16] = std::array::from_fn(|i| i as u8);
        assert_eq!(folded(&uuid), 0x0706_0504_0302_0100 ^ 0x0f0e_0d0c_0b0a_0908);
    }

    /// Every device the probe lists that the choice can pick is named by the registry ID it was
    /// named by before the choice read the runtime's identities (the fold of its UUID), which is
    /// the ID of the runtime's identity of that UUID, and the device the runtime makes of that ID
    /// answers it: the choice opens the device it opened before. The runtime lists every
    /// candidate. The devices and their IDs are printed for the run's record
    #[test]
    fn each_candidate_keeps_its_registry_id() {
        let seen = crate::os::seen_devices();
        let options = crate::platform::device_options();
        let identities = rt::Device::identities_with(options).unwrap_or_default();
        for (n, s) in seen.iter().enumerate() {
            eprintln!("device {n}: {s:?}");
        }
        for identity in &identities {
            assert_eq!(
                identity.registry_id,
                folded(&identity.device_uuid),
                "{identity:?}"
            );
        }
        for s in seen.iter().filter(|s| s.candidate()) {
            let uuid = s.uuid.expect("a candidate has a UUID");
            assert_eq!(s.registry_id(), Some(folded(&uuid)), "{s:?}");
            let id = s.registry_id().unwrap_or_default();
            let device = crate::answer::unheard(|| rt::Device::with_registry_id(id, options));
            assert_eq!(device.map(|d| d.registry_id()).ok(), Some(id), "{s:?}");
        }
    }

    /// The process's default device is the one picked of the devices the probe lists, named by
    /// the registry ID the runtime answers for it: on a host of several GPUs, the verdict the
    /// process's choice kept, read back here. The list and the pick are printed for the run's
    /// record, with how long the process's first open took, the choice's included, and the
    /// verdict kept
    #[test]
    fn the_default_device_is_the_pick() {
        let clock = crate::os::OsClock::new();
        let start = clock.now_mono();
        let Ok(device) = crate::platform::default_device() else {
            return;
        };
        let first_open = clock.now_mono().saturating_duration_since(start);
        eprintln!("first open: {:.1} ms", timing::millis(first_open));
        let seen = crate::os::seen_devices();
        let file = verdict_file();
        let picked = choose(
            seen,
            &user_selection(|name| std::env::var_os(name)),
            |gpus| measured(seen, gpus, file.as_deref(), |_, _| None),
        );
        for (n, s) in seen.iter().enumerate() {
            eprintln!("device {n}: {s:?}");
        }
        let (at, why) = picked.expect("a candidate, where the runtime made a device");
        eprintln!("picked device {at}: {}", why.describe());
        if let Some(text) = file.and_then(|path| std::fs::read_to_string(path).ok()) {
            eprintln!("verdict kept:\n{text}");
        }
        assert_eq!(
            seen.get(at).and_then(Seen::registry_id),
            Some(device.registry_id()),
            "{seen:?}"
        );
    }
}
