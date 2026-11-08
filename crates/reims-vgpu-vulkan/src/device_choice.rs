//! Which GPU the library opens where the host has more than one Vulkan device: the runtime's
//! default device is made of the physical device this module picks, once a process ([`chosen`]),
//! and every default open reaches it (`crate::platform`'s `default_device`).
//!
//! **Candidates.** The physical devices are those the window's probe lists, in the loader's order
//! (`crate::os`'s `seen_devices`, read through the probe's one instance). A candidate is one the
//! runtime makes a device of: at Vulkan 1.2 or later, with a queue family of graphics and compute
//! work, and a `deviceUUID`, which names it to the runtime (its registry ID, [`registry_id`]).
//!
//! **The pick.**
//! - Where the host's user picked a device, through `MESA_VK_DEVICE_SELECT` (with or without its
//!   `!`), `DRI_PRIME` or `VK_LOADER_DEVICE_SELECT` ([`SELECTION_VARIABLES`]), the loader's first
//!   candidate: Mesa's device-select layer and the loader list the device those name first. The
//!   runtime alone ranks the candidates by kind first, so it opened a discrete GPU even where the
//!   user's variable listed an integrated one first without its `!`.
//! - Otherwise the runtime's own order: discrete GPUs, then integrated, virtual and CPU ones, each
//!   kind in the loader's order (`Device::system_default_with`, read from code).
//!
//! **Unchanged hosts.** Where the pick is the device the runtime would make its default device of
//! anyway, it is made as before, by `Device::system_default_with` ([`chosen`] answers `None`). The
//! choice is logged once, at info: every device seen, then the one picked and why.

use std::ffi::OsString;
use std::fmt;
use std::sync::LazyLock;

use ash::vk;

/// The log target of the choice's lines.
const TARGET: &str = "reims_vgpu_vulkan::device_choice";

/// The variables through which the host's user picks a Vulkan device, each of which reorders the
/// loader's list so the device it names comes first: Mesa's device-select layer's
/// `MESA_VK_DEVICE_SELECT` (`vendor:device`, with a `!` it lists that device alone) and
/// `DRI_PRIME`, and the loader's `VK_LOADER_DEVICE_SELECT`. Measured: Mesa 26.2.6's layer and
/// the loader 1.4.363 read these names.
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
}

impl Seen {
    /// Its registry ID ([`registry_id`] of its UUID), where it has a UUID.
    pub(crate) fn registry_id(&self) -> Option<u64> {
        self.uuid.as_ref().map(registry_id)
    }

    /// Whether the runtime makes a device of it (the module doc's candidates).
    pub(crate) fn candidate(&self) -> bool {
        self.api_version >= vk::API_VERSION_1_2 && self.graphics_compute && self.uuid.is_some()
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
            ", not usable: below Vulkan 1.2"
        } else if !self.graphics_compute {
            ", not usable: no graphics and compute queue"
        } else if self.uuid.is_none() {
            ", not usable: no device UUID"
        } else {
            ""
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
    /// The host's user picked it through this variable: the loader's first candidate.
    UserChose(&'static str),
    /// It is first in the runtime's own order.
    RuntimeOrder,
}

impl Why {
    /// The reason as the log line gives it.
    fn describe(self) -> String {
        match self {
            Why::Only => "the only device the runtime can use".to_owned(),
            Why::UserChose(var) => {
                format!("{var} is set, so the loader's first device, as the host's user picked")
            }
            Why::RuntimeOrder => {
                "first in the runtime's own order (discrete, integrated, virtual, CPU)".to_owned()
            }
        }
    }
}

/// The device of `seen` (the loader's order) the library opens, and why: the first candidate
/// where `user` names the variable the host's user picked a device through, else the first of the
/// runtime's own order (the module doc). `None` where no device is a candidate.
pub(crate) fn choose(seen: &[Seen], user: Option<&'static str>) -> Option<(usize, Why)> {
    let mut candidates = seen.iter().enumerate().filter(|(_, s)| s.candidate());
    if let Some(var) = user {
        return candidates.next().map(|(at, _)| (at, Why::UserChose(var)));
    }
    if candidates.count() == 1 {
        return runtime_first(seen).map(|at| (at, Why::Only));
    }
    runtime_first(seen).map(|at| (at, Why::RuntimeOrder))
}

/// The candidate of `seen` the runtime alone makes its default device of: the first of its order
/// of kinds (discrete, integrated, virtual, CPU, any other), then the loader's
/// (`Device::system_default_with`, read from code).
fn runtime_first(seen: &[Seen]) -> Option<usize> {
    let preference = |kind: vk::PhysicalDeviceType| match kind {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 3,
        _ => 4,
    };
    seen.iter()
        .enumerate()
        .filter(|(_, s)| s.candidate())
        .min_by_key(|&(at, s)| (preference(s.kind), at))
        .map(|(at, _)| at)
}

/// The registry ID to open the device `picked` of `seen` by, where the runtime's default device
/// is another: `None` where nothing was picked, or the pick is the device the runtime makes its
/// default device of anyway.
fn to_open(seen: &[Seen], picked: Option<usize>) -> Option<u64> {
    let at = picked?;
    if Some(at) == runtime_first(seen) {
        return None;
    }
    seen.get(at)?.registry_id()
}

/// The first of [`SELECTION_VARIABLES`] `var` answers a value that is not empty for.
fn user_selection(var: impl Fn(&str) -> Option<OsString>) -> Option<&'static str> {
    SELECTION_VARIABLES
        .into_iter()
        .find(|name| var(name).is_some_and(|value| !value.is_empty()))
}

/// The registry ID of the device the process opens as its default device, where it is not the one
/// the runtime makes its default device of anyway: read once from the probe's devices and the
/// environment, and logged (the module doc). `None` where the runtime's default device is the
/// pick, as on a host of one GPU, or where nothing was probed.
pub(crate) fn chosen() -> Option<u64> {
    static CHOSEN: LazyLock<Option<u64>> = LazyLock::new(|| {
        let seen = crate::os::seen_devices();
        let picked = choose(seen, user_selection(|name| std::env::var_os(name)));
        log_choice(seen, picked);
        to_open(seen, picked.map(|(at, _)| at))
    });
    *CHOSEN
}

/// Logs every device of `seen` and the one `picked`, with why, at info.
fn log_choice(seen: &[Seen], picked: Option<(usize, Why)>) {
    for (at, device) in seen.iter().enumerate() {
        log::info!(target: TARGET, "Vulkan device {at}: {}", device.describe());
    }
    match picked.and_then(|(at, why)| Some((at, seen.get(at)?, why))) {
        Some((at, device, why)) => log::info!(
            target: TARGET,
            "the library's GPU is Vulkan device {at}, {}: {}",
            device.name,
            why.describe()
        ),
        None => log::info!(
            target: TARGET,
            "no Vulkan device the runtime can use was listed: the runtime's default device"
        ),
    }
}

/// The registry ID the runtime answers for a device whose `deviceUUID` is `uuid`, which
/// `Device::with_registry_id` opens it by (`PhysicalDeviceIdentity::registry_id`, read from
/// code): the UUID's two halves, little-endian, exclusive-ored, and 1 where that is 0. A test
/// holds it to the runtime's answer for the device it makes.
pub(crate) fn registry_id(uuid: &[u8; 16]) -> u64 {
    let (low, high) = uuid.split_at(8);
    let half = |bytes: &[u8]| u64::from_le_bytes(bytes.try_into().unwrap_or_default());
    match half(low) ^ half(high) {
        0 => 1,
        id => id,
    }
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

    /// Without a selection variable the pick is the runtime's own: a discrete GPU before an
    /// integrated one wherever the loader lists it, then llvmpipe; so nothing is opened by ID
    #[test]
    fn without_a_selection_the_runtimes_order_stands() {
        let list = [integrated(11, 1), discrete(4, 2), cpu(3)];
        assert_eq!(choose(&list, None), Some((1, Why::RuntimeOrder)));
        assert_eq!(to_open(&list, Some(1)), None);
        let list = [cpu(1), integrated(2, 2)];
        assert_eq!(choose(&list, None), Some((1, Why::RuntimeOrder)));
        assert_eq!(to_open(&list, Some(1)), None);
    }

    /// A host of one candidate opens it as the runtime does, with nothing to open by ID
    #[test]
    fn one_candidate_is_opened_as_before() {
        for list in [vec![discrete(4, 1)], vec![integrated(2, 1)], vec![cpu(1)]] {
            assert_eq!(choose(&list, None), Some((0, Why::Only)));
            assert_eq!(to_open(&list, Some(0)), None);
        }
        assert_eq!(choose(&[], None), None);
        assert_eq!(to_open(&[], None), None);
    }

    /// Where the host's user picked a device, the loader's first candidate is opened: by ID where
    /// the runtime alone would make another
    #[test]
    fn the_users_pick_is_the_loaders_first() {
        let list = [integrated(11, 1), discrete(24, 2), cpu(3)];
        let picked = choose(&list, Some("MESA_VK_DEVICE_SELECT"));
        assert_eq!(picked, Some((0, Why::UserChose("MESA_VK_DEVICE_SELECT"))));
        assert_eq!(to_open(&list, Some(0)), list[0].registry_id());
        let not_usable = Seen {
            api_version: vk::API_VERSION_1_1,
            ..discrete(24, 9)
        };
        let list = [not_usable, discrete(4, 2), integrated(11, 1)];
        assert_eq!(choose(&list, Some("DRI_PRIME")).map(|(at, _)| at), Some(1));
        assert_eq!(to_open(&list, Some(1)), None);
    }

    /// The user's pick is read from the first selection variable set to a value, in Mesa's
    /// layer's, then the loader's; an empty value picks nothing
    #[test]
    fn the_users_pick_is_read_from_the_environment() {
        let env = |set: &'static [(&'static str, &'static str)]| {
            move |name: &str| {
                set.iter()
                    .find(|(n, _)| *n == name)
                    .map(|(_, v)| OsString::from(v))
            }
        };
        assert_eq!(user_selection(env(&[])), None);
        assert_eq!(user_selection(env(&[("MESA_VK_DEVICE_SELECT", "")])), None);
        assert_eq!(
            user_selection(env(&[("MESA_VK_DEVICE_SELECT", "1002:15bf!")])),
            Some("MESA_VK_DEVICE_SELECT")
        );
        assert_eq!(
            user_selection(env(&[
                ("VK_LOADER_DEVICE_SELECT", "0x1002:0x699f"),
                ("DRI_PRIME", "1")
            ])),
            Some("DRI_PRIME")
        );
        assert_eq!(
            user_selection(env(&[("VK_LOADER_DEVICE_SELECT", "0x1002:0x699f")])),
            Some("VK_LOADER_DEVICE_SELECT")
        );
        assert_eq!(user_selection(env(&[("VK_LOADER_DEBUG", "all")])), None);
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
        assert_eq!(choose(&list, None), Some((3, Why::Only)));
        assert_eq!(runtime_first(&list), Some(3));
        assert_eq!(choose(&list[..3], None), None);
    }

    /// The registry ID is the UUID's halves exclusive-ored, little-endian; 1 where they cancel
    #[test]
    fn the_registry_id_folds_the_uuid() {
        let mut uuid = [0u8; 16];
        uuid[4] = 0xc5;
        assert_eq!(registry_id(&uuid), 0xc5_0000_0000);
        uuid[12] = 0xc5;
        assert_eq!(registry_id(&uuid), 1);
        let uuid: [u8; 16] = std::array::from_fn(|i| i as u8);
        assert_eq!(
            registry_id(&uuid),
            0x0706_0504_0302_0100 ^ 0x0f0e_0d0c_0b0a_0908
        );
    }

    /// The process's default device is the one picked of the devices the probe lists, named by
    /// the registry ID the runtime answers for it. The list and the pick are printed for the
    /// run's record
    #[test]
    fn the_default_device_is_the_pick() {
        let Ok(device) = crate::platform::default_device() else {
            return;
        };
        let seen = crate::os::seen_devices();
        let picked = choose(seen, user_selection(|name| std::env::var_os(name)));
        for (n, s) in seen.iter().enumerate() {
            eprintln!("device {n}: {s:?}");
        }
        let (at, why) = picked.expect("a candidate, where the runtime made a device");
        eprintln!("picked device {at}: {}", why.describe());
        assert_eq!(
            seen.get(at).and_then(Seen::registry_id),
            Some(device.registry_id()),
            "{seen:?}"
        );
    }
}
