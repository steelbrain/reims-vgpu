//! Which backend the video device runs on, and on which GPU. Nothing configures it: [`choose`]
//! decides from the host, and logs what it chose and why at info. (The scaler device runs on the
//! GPU the library draws with, `crate::scaler`, and chooses nothing.)
//!
//! - **macOS**: VideoToolbox, the host's own; there is nothing to choose.
//! - **Linux**: the GPU the library's GPU device runs on, which is the Vulkan runtime's system
//!   default device's (reims-vgpu-vulkan's `host_gpus`), so a VM pinned to a GPU through the
//!   loader's environment (`VK_DRIVER_FILES`, and `VK_LOADER_DEVICE_ID_FILTER` where one driver
//!   drives several GPUs, as reims-ctl pins one) decodes and encodes on that GPU too.
//!   The answer is the one the GPU device's would be, whether a GPU device was made first or not.
//!   On that GPU:
//!   - NVIDIA's NVDEC and NVENC, where it is an NVIDIA GPU and CUDA opens the GPU of its PCI
//!     address;
//!   - VA-API on its DRM render node otherwise.
//!
//!   Where that GPU is a CPU implementation (lavapipe, llvmpipe), or its backend lacks what the
//!   device needs ([`Need`]: any part), the device picks a hardware GPU itself: the first of the
//!   host's hardware GPUs (`host_gpus::hardware_gpus`) whose backend has it, discrete ones first,
//!   then integrated, virtual and other ones, each kind in the loader's order. Where none has,
//!   the device has no backend, and offers the guest no codec.
//!
//! Each backend is reims-vgpu-video-core's `Backend`, the same to the device whichever it is: the
//! video device takes its codecs from it.
//!
//! **Shared.** A backend is open once per place, VA-API once per render node, NVIDIA's once per
//! GPU and VideoToolbox once per host, and every device that picks it there shares it
//! ([`SharedBackend`]): the video devices of a process on one GPU open it once. It is released
//! when the last device that holds it goes, in whichever order the devices go, and a device made
//! after that opens it anew. A backend that does not open, or opens with no part, is not kept, so
//! the next device asks again.

// On macOS the devices take VideoToolbox alone: the GPU choice is Linux's, and its tests run on
// every OS.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::fmt;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_core::backend::Backend;
use reims_vgpu_video_core::runtime::Runtime;

/// A backend's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kind {
    /// VA-API, on a GPU's render node (Linux).
    Vaapi,
    /// NVIDIA's NVDEC and NVENC, on an NVIDIA GPU by its PCI address (Linux).
    Nvidia,
    /// VideoToolbox (macOS).
    #[cfg_attr(
        not(target_os = "macos"),
        allow(dead_code, reason = "macOS's alone, named by the tests elsewhere")
    )]
    VideoToolbox,
}

/// What a device needs of a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Need {
    /// Any part: what the video device offers the guest.
    Video,
}

impl Need {
    /// Whether `backend` has it.
    fn met(self, backend: &Backend) -> bool {
        match self {
            Need::Video => {
                backend.decoders.is_some()
                    || backend.encoders.is_some()
                    || backend.motion.is_some()
                    || backend.jpeg.is_some()
            }
        }
    }

    /// What the log says of a backend or GPU that lacks it.
    fn lacking(self) -> &'static str {
        match self {
            Need::Video => "offers no video",
        }
    }

    /// What the log says of the GPU sought for it.
    fn sought(self) -> &'static str {
        match self {
            Need::Video => "that offers video",
        }
    }
}

impl Kind {
    /// What the log calls it.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Kind::Vaapi => "VA-API",
            Kind::Nvidia => "NVIDIA",
            Kind::VideoToolbox => "VideoToolbox",
        }
    }
}

/// The PCI vendor ID of NVIDIA's GPUs.
const NVIDIA_VENDOR: u32 = 0x10de;

/// The kind of a GPU, from `VkPhysicalDeviceType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuKind {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl GpuKind {
    /// The order the hardware GPUs are tried in: discrete, integrated, virtual, other, as the
    /// runtime ranks its devices.
    fn rank(self) -> u8 {
        match self {
            GpuKind::Discrete => 0,
            GpuKind::Integrated => 1,
            GpuKind::Virtual => 2,
            GpuKind::Other => 3,
            GpuKind::Cpu => 4,
        }
    }

    /// What the log calls it.
    fn word(self) -> &'static str {
        match self {
            GpuKind::Discrete => "discrete",
            GpuKind::Integrated => "integrated",
            GpuKind::Virtual => "virtual",
            GpuKind::Cpu => "CPU",
            GpuKind::Other => "other",
        }
    }
}

/// One of the host's GPUs, as the choice sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Gpu {
    /// Its name, as its Vulkan driver gives it.
    pub(crate) name: String,
    pub(crate) kind: GpuKind,
    /// Its PCI vendor ID.
    pub(crate) vendor_id: u32,
    /// Its PCI address, `0000:03:00.0`, which NVIDIA's backend opens it by.
    pub(crate) pci_address: Option<String>,
    /// Its DRM render node, `/dev/dri/renderD128`, which VA-API opens it by.
    pub(crate) render_node: Option<PathBuf>,
}

impl Gpu {
    /// Whether `other` is this GPU: of the same PCI address where both have one.
    fn is(&self, other: &Gpu) -> bool {
        match (&self.pci_address, &other.pci_address) {
            (Some(a), Some(b)) => a == b,
            _ => self == other,
        }
    }
}

impl fmt::Display for Gpu {
    /// `AMD Radeon RX 7900 XTX (RADV NAVI31) (discrete, 0000:03:00.0, /dev/dri/renderD128)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}", self.name, self.kind.word())?;
        if let Some(address) = &self.pci_address {
            write!(f, ", {address}")?;
        }
        if let Some(node) = &self.render_node {
            write!(f, ", {}", node.display())?;
        }
        f.write_str(")")
    }
}

/// What the choice asks of the host: its GPUs, and a backend's open on one. Linux's is
/// [`LinuxHost`]; the tests bring stand-ins.
pub(crate) trait Host {
    /// The GPU the library's GPU device runs on; `None` where the Vulkan runtime makes no device.
    fn gpu_device(&self) -> Option<Gpu>;
    /// The host's hardware GPUs, in the loader's order.
    fn hardware(&self) -> Vec<Gpu>;
    /// `kind`'s backend on `gpu`: VA-API's at its render node, NVIDIA's at its PCI address.
    fn open(&self, kind: Kind, gpu: &Gpu) -> Result<Backend, String>;
}

/// A backend picked for a device, and what its log calls it. The device keeps it for its life:
/// it holds the device's share of the backend.
pub(crate) struct Chosen {
    /// The backend, shared with the other devices that picked it there; `None` where none
    /// opened.
    opened: Option<Arc<SharedBackend>>,
    /// `VA-API`, `NVIDIA`, `VideoToolbox`, or `none`.
    pub(crate) label: &'static str,
}

impl Chosen {
    /// No backend, for `device`, which opens none (`VideoDevice::without_backend`), `why` in
    /// words, as the log says.
    #[cfg(any(test, feature = "testkit"))]
    pub(crate) fn none(device: &str, why: &str) -> Chosen {
        log::info!("{device} opens no backend: {why}");
        Chosen {
            opened: None,
            label: "none",
        }
    }

    /// The backend's parts, each shared with the other devices that hold it; with no part where
    /// none opened.
    pub(crate) fn backend(&self) -> Backend {
        self.opened.as_ref().map_or_else(
            || Backend::none(self.label),
            |shared| shared.backend.clone(),
        )
    }

    /// Whether a backend opened for it.
    pub(crate) fn has_backend(&self) -> bool {
        self.opened.is_some()
    }
}

/// A backend open in one place, which every device that picks it there shares (module doc).
pub(crate) struct SharedBackend {
    backend: Backend,
    /// Where it is open, as the log says it: `VA-API at /dev/dri/renderD128`.
    place: String,
    /// The device it was opened for, as the log names it: `video device`.
    opened_for: String,
}

impl Drop for SharedBackend {
    fn drop(&mut self) {
        log::info!("{} released: no device holds it", self.place);
    }
}

/// Where a backend is open: its kind, and the GPU it runs on, by its render node (resolved) for
/// VA-API and by its PCI address for NVIDIA; none for VideoToolbox.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Place {
    kind: Kind,
    at: Option<String>,
}

impl Place {
    /// `kind`'s place on `gpu`; `None` where the GPU has no render node (VA-API) or no PCI
    /// address (NVIDIA).
    fn on(kind: Kind, gpu: &Gpu) -> Option<Place> {
        let at = match kind {
            Kind::Vaapi => {
                let node = gpu.render_node.as_ref()?;
                let node = std::fs::canonicalize(node).unwrap_or_else(|_| node.clone());
                Some(node.display().to_string())
            }
            Kind::Nvidia => Some(gpu.pci_address.clone()?),
            Kind::VideoToolbox => None,
        };
        Some(Place { kind, at })
    }
}

impl fmt::Display for Place {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.at {
            Some(at) => write!(f, "{} at {at}", self.kind.label()),
            None => f.write_str(self.kind.label()),
        }
    }
}

/// Backends open, each held weakly: a device holds its share, this only finds it for the next
/// device. The devices pick from the process's, [`OPEN`]; a test that counts what its own devices
/// hold gives them one of its own.
pub(crate) struct OpenBackends {
    open: Mutex<Vec<(Place, Weak<SharedBackend>)>>,
}

impl OpenBackends {
    pub(crate) const fn new() -> OpenBackends {
        OpenBackends {
            open: Mutex::new(Vec::new()),
        }
    }

    /// The backend open at `place`, and `true`; else the one `open` opens there now for
    /// `device`, and `false`. A backend that does not open is not kept, so the next device asks
    /// again. The lock is held while it opens, so that two devices made at once open it once.
    fn get_or_open(
        &self,
        place: Place,
        device: &str,
        open: impl FnOnce() -> Result<Backend, String>,
    ) -> Result<(Arc<SharedBackend>, bool), String> {
        let mut open_now = self.open.lock();
        open_now.retain(|(_, held)| held.strong_count() > 0);
        let found = (open_now.iter())
            .filter(|(at, _)| *at == place)
            .find_map(|(_, held)| held.upgrade());
        if let Some(shared) = found {
            return Ok((shared, true));
        }
        let backend = open()?;
        let shared = Arc::new(SharedBackend {
            backend,
            place: place.to_string(),
            opened_for: device.to_owned(),
        });
        open_now.push((place, Arc::downgrade(&shared)));
        Ok((shared, false))
    }
}

/// The process's open backends.
pub(crate) static OPEN: OpenBackends = OpenBackends::new();

/// How many devices hold the backend open in `open` at VideoToolbox's place: 0 where none is.
/// For the devices' test on macOS, the one host whose backend a test can open.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn videotoolbox_holders(open: &OpenBackends) -> usize {
    let place = Place {
        kind: Kind::VideoToolbox,
        at: None,
    };
    (open.open.lock().iter())
        .filter(|(at, _)| *at == place)
        .map(|(_, held)| held.strong_count())
        .sum()
}

/// A backend, shared, and whether it was open already.
type Opened = (Arc<SharedBackend>, bool);

/// What [`pick`] picked: the backend's label, why, and the backend; or why none opened.
struct Picked {
    label: &'static str,
    why: String,
    opened: Result<Opened, String>,
}

/// The host's backend with what `device` (`video device`) needs, opened, or `device`'s share of
/// it where it is open already in `open` (module doc), with what it chose and why logged; where
/// none has it, none, which the device says what it does without. A backend it opens runs on
/// `runtime`, the device's platform's, for its life (`super::runtime`).
pub(crate) fn choose(
    open: &OpenBackends,
    device: &str,
    need: Need,
    runtime: &Arc<dyn Runtime>,
) -> Chosen {
    #[cfg(target_os = "macos")]
    let picked = Picked {
        label: Kind::VideoToolbox.label(),
        why: "this host is macOS".to_owned(),
        opened: open.get_or_open(
            Place {
                kind: Kind::VideoToolbox,
                at: None,
            },
            device,
            || open_videotoolbox(runtime),
        ),
    };
    #[cfg(target_os = "linux")]
    let picked = pick(device, need, &LinuxHost { runtime }, open);
    #[cfg(target_os = "macos")]
    let _ = need;
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let _ = (open, runtime, need);
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    let picked = Picked {
        label: "none",
        why: "this host has no video backend".to_owned(),
        opened: Err("none is built for this OS".to_owned()),
    };
    let Picked { label, why, opened } = picked;
    match opened {
        Ok((shared, was_open)) => {
            let summary = shared.backend.summary();
            if was_open {
                log::info!(
                    "{device} backend {label} ({why}): the one open for the {}, {}: {summary}",
                    shared.opened_for,
                    shared.place
                );
            } else {
                log::info!("{device} backend {label} ({why}): {summary}");
            }
            Chosen {
                opened: Some(shared),
                label,
            }
        }
        Err(error) => {
            log::warn!("{device} has no backend ({why}; {error})");
            Chosen {
                opened: None,
                label,
            }
        }
    }
}

/// The backend `host` offers `device`, with what it needs (module doc): on the GPU device's GPU,
/// else on the first hardware GPU that has it, discrete ones first, each opened at `open`'s
/// place or taken from there.
fn pick(device: &str, need: Need, host: &dyn Host, open: &OpenBackends) -> Picked {
    let gpu = host.gpu_device();
    let reason = match &gpu {
        Some(g) if g.kind != GpuKind::Cpu => match open_on(device, need, host, open, g) {
            Ok((kind, opened, note)) => {
                return Picked {
                    label: kind.label(),
                    why: format!("the GPU device's GPU, {g}{note}"),
                    opened: Ok(opened),
                };
            }
            Err(why) => format!("the GPU device's GPU, {g}, {} ({why})", need.lacking()),
        },
        Some(g) => format!("the GPU device's GPU, {g}, is a software rasterizer"),
        None => "the Vulkan runtime makes no device, so there is no GPU device's GPU".to_owned(),
    };
    let mut others: Vec<Gpu> = (host.hardware().into_iter())
        .filter(|other| other.kind != GpuKind::Cpu)
        .filter(|other| !gpu.as_ref().is_some_and(|g| g.is(other)))
        .collect();
    others.sort_by_key(|other| other.kind.rank());
    let mut refused = Vec::new();
    for other in others {
        match open_on(device, need, host, open, &other) {
            Ok((kind, opened, note)) => {
                return Picked {
                    label: kind.label(),
                    why: format!(
                        "{reason}: the first hardware GPU {}, discrete ones first, {other}{note}",
                        need.sought()
                    ),
                    opened: Ok(opened),
                };
            }
            Err(why) => refused.push(format!("{other}: {why}")),
        }
    }
    let error = if refused.is_empty() {
        "the host has no other hardware GPU".to_owned()
    } else {
        format!(
            "no other hardware GPU {}: {}",
            need.sought(),
            refused.join("; ")
        )
    };
    Picked {
        label: "none",
        why: reason,
        opened: Err(error),
    }
}

/// The backend on `gpu` with what `device` needs, at `open`'s place or opened there now: NVIDIA's
/// where it is an NVIDIA GPU and it has it, else VA-API's; with what the log adds to why (what
/// NVIDIA's lacked, on an NVIDIA GPU). `Err` with why where neither has it. A backend opened here
/// that lacks it goes again at once.
fn open_on(
    device: &str,
    need: Need,
    host: &dyn Host,
    open: &OpenBackends,
    gpu: &Gpu,
) -> Result<(Kind, Opened, String), String> {
    let mut why_not = Vec::new();
    let kinds: &[Kind] = if gpu.vendor_id == NVIDIA_VENDOR {
        &[Kind::Nvidia, Kind::Vaapi]
    } else {
        &[Kind::Vaapi]
    };
    for &kind in kinds {
        let Some(place) = Place::on(kind, gpu) else {
            why_not.push(match kind {
                Kind::Nvidia => "NVIDIA's: it has no PCI address".to_owned(),
                _ => format!("{}'s: it has no DRM render node", kind.label()),
            });
            continue;
        };
        match open.get_or_open(place, device, || offering(host.open(kind, gpu))) {
            Ok(opened) if need.met(&opened.0.backend) => {
                let note = if why_not.is_empty() {
                    String::new()
                } else {
                    format!(", where {}", why_not.join("; "))
                };
                return Ok((kind, opened, note));
            }
            Ok(_) => why_not.push(format!("{}'s {}", kind.label(), need.lacking())),
            Err(why) => why_not.push(format!("{}'s: {why}", kind.label())),
        }
    }
    Err(why_not.join("; "))
}

/// `opened`, where it has a part: a backend with none offers no video, and is not kept.
fn offering(opened: Result<Backend, String>) -> Result<Backend, String> {
    let backend = opened?;
    if Need::Video.met(&backend) {
        Ok(backend)
    } else {
        Err(format!("it opened with no part: {}", backend.summary()))
    }
}

/// The host's GPUs as reims-vgpu-vulkan's `host_gpus` reads them, and its backends, which run
/// on `runtime`.
#[cfg(target_os = "linux")]
struct LinuxHost<'a> {
    runtime: &'a Arc<dyn Runtime>,
}

#[cfg(target_os = "linux")]
impl From<reims_vgpu_vulkan::host_gpus::HostGpu> for Gpu {
    fn from(gpu: reims_vgpu_vulkan::host_gpus::HostGpu) -> Gpu {
        use reims_vgpu_vulkan::host_gpus::GpuKind as K;
        Gpu {
            name: gpu.name,
            kind: match gpu.kind {
                K::Discrete => GpuKind::Discrete,
                K::Integrated => GpuKind::Integrated,
                K::Virtual => GpuKind::Virtual,
                K::Cpu => GpuKind::Cpu,
                K::Other => GpuKind::Other,
            },
            vendor_id: gpu.vendor_id,
            pci_address: gpu.pci_address,
            render_node: gpu.render_node,
        }
    }
}

#[cfg(target_os = "linux")]
impl Host for LinuxHost<'_> {
    fn gpu_device(&self) -> Option<Gpu> {
        reims_vgpu_vulkan::host_gpus::gpu_device_gpu().map(Gpu::from)
    }

    fn hardware(&self) -> Vec<Gpu> {
        (reims_vgpu_vulkan::host_gpus::hardware_gpus().into_iter())
            .map(Gpu::from)
            .collect()
    }

    /// VA-API's decoders, encoders and JPEG decoder on the GPU's render node; NVIDIA's NVDEC and
    /// NVENC on the GPU of its PCI address, with no JPEG decoder, its libraries loaded now, so a
    /// host without them does not open it.
    fn open(&self, kind: Kind, gpu: &Gpu) -> Result<Backend, String> {
        match (kind, &gpu.render_node, &gpu.pci_address) {
            (Kind::Vaapi, Some(node), _) => {
                reims_vgpu_video_vaapi::open_backend(node, Arc::clone(self.runtime))
                    .map_err(|e| e.to_string())
            }
            (Kind::Nvidia, _, Some(address)) => {
                reims_vgpu_video_nvidia::open_backend(address, Arc::clone(self.runtime))
                    .map_err(|e| e.to_string())
            }
            (Kind::VideoToolbox, ..) => Err("VideoToolbox is macOS's".to_owned()),
            _ => Err("the GPU has no place it opens at".to_owned()),
        }
    }
}

/// VideoToolbox's hardware decoders and encoders, with no JPEG decoder; it always opens, with no
/// part where this Mac lists none. Its encoders wait on `runtime`'s clock.
#[cfg(target_os = "macos")]
fn open_videotoolbox(runtime: &Arc<dyn Runtime>) -> Result<Backend, String> {
    Ok(reims_vgpu_video_videotoolbox::backend(Arc::clone(runtime)))
}

#[cfg(test)]
mod tests;
