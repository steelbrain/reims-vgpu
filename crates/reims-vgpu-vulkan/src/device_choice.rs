//! Which GPU the library opens where the host has more than one Vulkan device: the rt
//! platform's one device is made of the physical device this module picks ([`chosen`]), of the
//! physical devices of the platform's instance that meet metal2vulkan-runtime's floor
//! (`crate::rt_platform::device`).
//!
//! **Candidates.** The physical devices are those the platform's instance lists, in the loader's
//! order, as `crate::os`'s `seen` reads each. A candidate is one rt makes a device of
//! ([`Seen::usable`]): at Vulkan 1.2 or later, with a queue family of graphics and compute work,
//! and a `deviceUUID`. And it is one a display's window presents on: it lists every device
//! extension the window's swapchain is made with ([`Seen::window_lacks`]), `VK_KHR_swapchain`,
//! which rt's floor takes. The rule is the same on every OS, and no vendor, device or driver
//! takes part in it. A device passed over for the window is logged with what it lacks, on its own
//! line and on the pick's.
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
//!   candidate, and the log line says that none names one.
//! - A device the window cannot present on is passed over even where the user's variable names
//!   it, as one below Vulkan 1.2 is.
//! - Otherwise, where one candidate is a GPU (any kind but a CPU implementation such as
//!   llvmpipe), that GPU.
//! - Where several are, the first by kind: discrete GPUs, then integrated, virtual and CPU ones,
//!   each kind in the loader's order. No GPU is timed. A CPU implementation is picked only where
//!   no GPU is a candidate.
//!
//! No vendor or device ID takes part: what a device reports picks it. The choice is logged once,
//! at info: every device seen, then the one picked and why, with each device passed over for the
//! window and why ([`choice_line`]).

use std::ffi::{CStr, OsString};
use std::fmt;

use ash::vk;

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
    /// in their order: `VK_KHR_swapchain`. Empty where it lists them all, as a device the window
    /// presents on does (the module doc's candidates).
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
/// what a window system names a GPU by.
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

    /// Whether rt makes a device of it: at Vulkan 1.2 or later, with a graphics and compute queue
    /// family and a UUID.
    pub(crate) fn usable(&self) -> bool {
        self.api_version >= vk::API_VERSION_1_2 && self.graphics_compute && self.uuid.is_some()
    }

    /// Whether it is a candidate (the module doc's): rt makes a device of it, and a display's
    /// window presents on it.
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

/// Its log line's description and its UUID: ash's types answer no `Debug` in this build.
impl fmt::Debug for Seen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{:x?}]", self.describe(), self.uuid)
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
    /// It is the first of several GPUs by kind (discrete, integrated, virtual, CPU), then in
    /// the loader's order.
    KindOrder,
}

impl Why {
    /// The reason as the log line gives it.
    fn describe(self) -> String {
        match self {
            Why::Only => "the only candidate".to_owned(),
            Why::OnlyGpu => "the only GPU of the candidates, beside CPU devices".to_owned(),
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
            Why::KindOrder => "first of the candidates by kind (discrete, integrated, virtual, \
                               CPU), then in the loader's order"
                .to_owned(),
        }
    }
}

/// The device of `seen` (the loader's order) the library opens, and why: where `user` lists the
/// selection variables the host's user set, with what each names ([`user_selection`]), the
/// candidate the first of them to name one names, or the loader's first candidate where an index
/// comes before it or none names one (the module doc); else the one candidate, or the one GPU
/// among the candidates; else the first candidate by kind (the module doc). `None` where no
/// device is a candidate.
pub(crate) fn choose(
    seen: &[Seen],
    user: &[(&'static str, Option<Named>)],
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
        _ => first_of_kinds(seen, Seen::candidate).map(|at| (at, Why::KindOrder)),
    }
}

/// Of the devices of `seen` that `keep` keeps, the first by kind (discrete, integrated, virtual,
/// CPU, any other), then in the loader's order.
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

/// The device of `seen` (an instance's physical devices that meet metal2vulkan-runtime's floor, in
/// the loader's order) the rt platform opens (`crate::rt_platform`), and logged: the device the
/// host's user's selection variables name, else the one GPU, else the first by kind (the module
/// doc). `None` where no device is a candidate.
pub(crate) fn chosen(seen: &[Seen]) -> Option<usize> {
    for (at, device) in seen.iter().enumerate() {
        log::info!(target: TARGET, "Vulkan device {at}: {}", device.describe());
    }
    let picked = choose(seen, &user_selection(|name| std::env::var_os(name)));
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
        None => "no candidate was listed".to_owned(),
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

    /// Of several GPUs, the first by kind, never llvmpipe: a discrete GPU before an integrated
    /// one wherever the loader lists it
    #[test]
    fn several_gpus_are_picked_by_kind_and_the_cpu_never() {
        let list = [integrated(11, 1), discrete(4, 2), cpu(3)];
        assert_eq!(choose(&list, &[]), Some((1, Why::KindOrder)));
        let list = [integrated(11, 1), integrated(2, 2), cpu(3)];
        assert_eq!(choose(&list, &[]), Some((0, Why::KindOrder)));
    }

    /// One GPU beside llvmpipe is picked, and of two CPU devices the loader's first
    #[test]
    fn one_gpu_is_picked_beside_the_cpu() {
        let list = [cpu(1), discrete(16, 2)];
        assert_eq!(choose(&list, &[]), Some((1, Why::OnlyGpu)));
        let list = [cpu(1), cpu(2)];
        assert_eq!(choose(&list, &[]), Some((0, Why::KindOrder)));
    }

    /// A host of one candidate opens it
    #[test]
    fn one_candidate_is_opened() {
        for list in [vec![discrete(4, 1)], vec![integrated(2, 1)], vec![cpu(1)]] {
            assert_eq!(choose(&list, &[]), Some((0, Why::Only)));
        }
        assert_eq!(choose(&[], &[]), None);
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

    /// The user's variable picks the candidate it names, wherever the loader lists it:
    /// `VK_LOADER_DEVICE_SELECT`'s too, which Mesa's layer leaves behind its own default
    #[test]
    fn the_users_pick_is_the_device_its_variable_names() {
        let list = um870();
        let ids = |v, d| Some(Named::Ids(v, d));
        let loader = [("VK_LOADER_DEVICE_SELECT", ids(0x10de, 0x2204))];
        let picked = choose(&list, &loader);
        let named = Named::Ids(0x10de, 0x2204);
        assert_eq!(
            picked,
            Some((1, Why::UserChose("VK_LOADER_DEVICE_SELECT", named)))
        );
        let mesa = [("MESA_VK_DEVICE_SELECT", ids(0x1002, 0x1900))];
        assert_eq!(choose(&list, &mesa).map(|(at, _)| at), Some(0));
        let lavapipe = [("MESA_VK_DEVICE_SELECT", ids(0x10005, 0))];
        assert_eq!(choose(&list, &lavapipe).map(|(at, _)| at), Some(2));
        let pci = Named::Pci(0, 0x67, 0, 0);
        let prime = [("DRI_PRIME", Some(pci))];
        assert_eq!(
            choose(&list, &prime),
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
        assert_eq!(choose(&list, &loader).map(|(at, _)| at), Some(2));
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
            choose(&list, &user),
            Some((1, Why::UserChose("DRI_PRIME", pci)))
        );
        let user = [
            ("MESA_VK_DEVICE_SELECT", nothing),
            ("VK_LOADER_DEVICE_SELECT", None),
        ];
        assert_eq!(
            choose(&list, &user),
            Some((0, Why::UserNamedNone("MESA_VK_DEVICE_SELECT")))
        );
        let user = [
            ("DRI_PRIME", Some(Named::Index)),
            ("VK_LOADER_DEVICE_SELECT", Some(Named::Ids(0x10de, 0x2204))),
        ];
        assert_eq!(choose(&list, &user), Some((0, Why::UserIndex("DRI_PRIME"))));
        assert_eq!(choose(&[], &user), None);
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
        assert_eq!(choose(&list, &[]), Some((3, Why::Only)));
        assert_eq!(choose(&list[..3], &[]), None);
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
    /// wherever the loader lists it: never picked, on every OS by the same rule, and alone it
    /// leaves no pick
    #[test]
    fn a_device_the_window_cannot_present_on_is_no_candidate() {
        let list = kosmickrisp_then_moltenvk();
        assert!(list[0].usable() && !list[0].candidate() && list[1].candidate());
        assert_eq!(choose(&list, &[]), Some((1, Why::Only)));
        assert_eq!(choose(&list[..1], &[]), None);
        // Of three GPUs, the first candidate by kind of the two the window presents on.
        let three = [list[0].clone(), list[1].clone(), integrated(8, 3)];
        assert_eq!(choose(&three, &[]), Some((1, Why::KindOrder)));
        // A GPU without `VK_KHR_swapchain`, which every OS's window takes, beside llvmpipe.
        let swapchain = ash::khr::swapchain::NAME;
        let headless = lacking(discrete(24, 4), "headless", (0x10de, 0x2204), &[swapchain]);
        let list = [headless, cpu(5)];
        assert_eq!(choose(&list, &[]), Some((1, Why::Only)));
    }

    /// A selection variable that names a device the window cannot present on is passed over,
    /// as one that names a device below Vulkan 1.2 is: the next variable's device, else the
    /// loader's first candidate
    #[test]
    fn a_variable_naming_a_device_the_window_cannot_present_on_is_passed_over() {
        let list = kosmickrisp_then_moltenvk();
        let kosmickrisp = ("MESA_VK_DEVICE_SELECT", Some(Named::Ids(0x106b, 0x64)));
        assert_eq!(
            choose(&list, &[kosmickrisp]),
            Some((1, Why::UserNamedNone("MESA_VK_DEVICE_SELECT")))
        );
        let moltenvk = Named::Ids(0x106b, 0x1b00_0209);
        let user = [kosmickrisp, ("VK_LOADER_DEVICE_SELECT", Some(moltenvk))];
        assert_eq!(
            choose(&list, &user),
            Some((1, Why::UserChose("VK_LOADER_DEVICE_SELECT", moltenvk)))
        );
        assert_eq!(choose(&list[..1], &[kosmickrisp]), None);
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
            format!("no candidate was listed; passed over Vulkan device 0, kosmickrisp: {lacks}")
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
}
