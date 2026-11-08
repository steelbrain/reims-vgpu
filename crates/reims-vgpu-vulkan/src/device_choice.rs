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
//! - Otherwise, where one candidate is a GPU (any kind but a CPU implementation such as
//!   llvmpipe), that GPU, never timed.
//! - Where several are, the one that ran a fixed workload fastest ([`measured`]): the display's
//!   composite at a fixed size and one pass over textures in host memory, timed on a device made
//!   for each GPU and torn down after (`timing.rs`). The verdict is kept in the library's cache
//!   directory, under a key of every timed GPU's UUID and driver version (`verdict.rs`), so only
//!   the first open after a GPU or driver change times them; a verdict that does not read times
//!   them again. Where the runtime refuses any GPU's workload, nothing is kept and the runtime's
//!   own order stands.
//! - A CPU implementation is picked only where no GPU is a candidate, and never timed. The
//!   runtime's own order: discrete GPUs, then integrated, virtual and CPU ones, each kind in the
//!   loader's order (`Device::system_default_with`, read from code).
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
//! **Unchanged hosts.** Where the pick is the device the runtime would make its default device of
//! anyway, it is made as before, by `Device::system_default_with` ([`chosen`] answers `None`). The
//! choice is logged once, at info: every device seen, each GPU's time and what its measurement
//! cost in all, then the one picked and why.

mod timing;
mod verdict;

use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Duration;

use ash::vk;

use crate::core::platform::os::Clock as _;

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
    /// It is the only GPU among the candidates, the others CPU implementations.
    OnlyGpu,
    /// The host's user picked it through this variable: the loader's first candidate.
    UserChose(&'static str),
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
            Why::Only => "the only device the runtime can use".to_owned(),
            Why::OnlyGpu => "the only GPU the runtime can use, beside CPU devices".to_owned(),
            Why::Measured => "the fastest of the GPUs at the choice's workload".to_owned(),
            Why::Cached => "the fastest of the GPUs at the choice's workload, as measured \
                            before on the same GPUs and drivers"
                .to_owned(),
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
/// where `user` names the variable the host's user picked a device through; else the one GPU
/// among the candidates; else, of several GPUs, the one `measure` answers for them (their
/// indices in `seen`), or the first of the runtime's own order where it answers none (the module
/// doc). `measure` is asked only of two GPUs or more. `None` where no device is a candidate.
pub(crate) fn choose(
    seen: &[Seen],
    user: Option<&'static str>,
    measure: impl FnOnce(&[usize]) -> Option<(usize, Why)>,
) -> Option<(usize, Why)> {
    let candidates: Vec<usize> = (seen.iter().enumerate())
        .filter(|(_, s)| s.candidate())
        .map(|(at, _)| at)
        .collect();
    if let Some(var) = user {
        return candidates.first().map(|&at| (at, Why::UserChose(var)));
    }
    if candidates.len() == 1 {
        return runtime_first(seen).map(|at| (at, Why::Only));
    }
    let gpus: Vec<usize> = (candidates.iter().copied())
        .filter(|&at| seen.get(at).is_some_and(Seen::gpu))
        .collect();
    match gpus.as_slice() {
        [only] => Some((*only, Why::OnlyGpu)),
        [_, _, ..] => measure(&gpus),
        [] => None,
    }
    .or_else(|| runtime_first(seen).map(|at| (at, Why::RuntimeOrder)))
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
        for (at, device) in seen.iter().enumerate() {
            log::info!(target: TARGET, "Vulkan device {at}: {}", device.describe());
        }
        let file = verdict_file();
        let picked = choose(
            seen,
            user_selection(|name| std::env::var_os(name)),
            |gpus| measured(seen, gpus, file.as_deref(), timing::time_on),
        );
        log_choice(seen, picked);
        to_open(seen, picked.map(|(at, _)| at))
    });
    *CHOSEN
}

/// Logs the device of `seen` `picked`, with why, at info.
fn log_choice(seen: &[Seen], picked: Option<(usize, Why)>) {
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
        let picked = choose(&list, None, |gpus| {
            asked = gpus.to_vec();
            Some((0, Why::Measured))
        });
        assert_eq!(picked, Some((0, Why::Measured)));
        assert_eq!(asked, [0, 1]);
        assert_eq!(to_open(&list, Some(0)), list[0].registry_id());
        assert_eq!(to_open(&list, Some(1)), None);
    }

    /// Where the measurement answers nothing, the runtime's own order stands: a discrete GPU
    /// before an integrated one wherever the loader lists it, opened as before
    #[test]
    fn an_unmeasured_choice_keeps_the_runtimes_order() {
        let list = [integrated(11, 1), discrete(4, 2), cpu(3)];
        assert_eq!(choose(&list, None, |_| None), Some((1, Why::RuntimeOrder)));
        assert_eq!(to_open(&list, Some(1)), None);
    }

    /// One GPU beside llvmpipe is picked unmeasured, as the runtime picks it, and two CPU devices
    /// keep the runtime's order unmeasured
    #[test]
    fn one_gpu_is_never_measured() {
        let list = [cpu(1), discrete(16, 2)];
        assert_eq!(choose(&list, None, unasked), Some((1, Why::OnlyGpu)));
        assert_eq!(to_open(&list, Some(1)), None);
        let list = [cpu(1), cpu(2)];
        assert_eq!(choose(&list, None, unasked), Some((0, Why::RuntimeOrder)));
    }

    /// A host of one candidate opens it as the runtime does, with nothing to open by ID
    #[test]
    fn one_candidate_is_opened_as_before() {
        for list in [vec![discrete(4, 1)], vec![integrated(2, 1)], vec![cpu(1)]] {
            assert_eq!(choose(&list, None, unasked), Some((0, Why::Only)));
            assert_eq!(to_open(&list, Some(0)), None);
        }
        assert_eq!(choose(&[], None, unasked), None);
        assert_eq!(to_open(&[], None), None);
    }

    /// Where the host's user picked a device, the loader's first candidate is opened, unmeasured:
    /// by ID where the runtime alone would make another
    #[test]
    fn the_users_pick_is_the_loaders_first() {
        let list = [integrated(11, 1), discrete(24, 2), cpu(3)];
        let picked = choose(&list, Some("MESA_VK_DEVICE_SELECT"), unasked);
        assert_eq!(picked, Some((0, Why::UserChose("MESA_VK_DEVICE_SELECT"))));
        assert_eq!(to_open(&list, Some(0)), list[0].registry_id());
        let not_usable = Seen {
            api_version: vk::API_VERSION_1_1,
            ..discrete(24, 9)
        };
        let list = [not_usable, discrete(4, 2), integrated(11, 1)];
        let picked = choose(&list, Some("DRI_PRIME"), unasked);
        assert_eq!(picked.map(|(at, _)| at), Some(1));
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
        assert_eq!(choose(&list, None, unasked), Some((3, Why::Only)));
        assert_eq!(runtime_first(&list), Some(3));
        assert_eq!(choose(&list[..3], None, unasked), None);
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
        let picked = choose(&list, None, |gpus| {
            measured(&list, gpus, Some(&file), timer(&[0, 40], &calls))
        });
        assert_eq!(picked, Some((1, Why::RuntimeOrder)));
        assert!(!file.exists());
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
            user_selection(|name| std::env::var_os(name)),
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
