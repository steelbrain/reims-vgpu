//! [`VideoDevice`]: the host side of the paravirtualized VideoToolbox device that a macOS guest's
//! AppleVideoToolboxParavirtualization drives (virtio-PCI `106b:1a03`), made from a
//! [`VideoDeviceConfig`]: reims-vgpu-video-core's host session, over this library's IOSurface
//! mapper and the host's callback.
//!
//! The device has no transport of its own. Its host (QEMU's `apple-videotoolbox-pci`) owns the
//! two virtqueues: it hands each guest-to-host message of queue 0 to [`VideoDevice::deliver`] and
//! writes the status that answers into the element, and it takes each host-to-guest message
//! through [`VideoDeviceConfig::send_to_guest`] into the next buffer the guest posts on queue 1.
//!
//! - **Lifecycle**, as Apple's host runs it: [`VideoDevice::start`], at the guest's DRIVER_OK,
//!   makes the host session (core's `HostSession`). A reset or a stop ends it in two phases:
//!   [`VideoDevice::invalidate`], after which it sends the guest nothing more, then the
//!   completion, once the work it had started has drained ([`VideoDevice::reset`] runs both). A
//!   message delivered while there is no session answers [`INVALID_SESSION`].
//! - **Messages.** The session reads each message as Apple's device does: a leading `subs` atom
//!   of 0x30 bytes names up to five IOSurfaces by mapping ID, then the `vtpv` message. It answers
//!   the status, and its replies and the messages of its decoders go to `send_to_guest`, from the
//!   delivering thread or from a session's own.
//! - **Surfaces** are reached through the IOSurface device the config names, held unretained, as
//!   the graphics device's mapper service holds it: a lookup after its teardown finds no surface.
//!   With none, each lookup goes to the process's latest IOSurface device, as a device made with
//!   `use_iosurface_mapper` finds it, so the host need not make the two in any order. A surface
//!   the session holds is one reference to it ([`MappedSurface`]), a hold the guest sees: the
//!   mapper answers an unmap of it Busy while the hold lives. Its bytes are lent to the session's
//!   decoders through the platform ([`ApiPlatform::lend_surface_bytes`]), and to its encoders to
//!   read ([`ApiPlatform::lend_surface_bytes_for_reading`]). Its memory is lent to the decoders'
//!   device too, where the platform lends it ([`ApiPlatform::surface_device_memory`]), so that
//!   the device writes frames into it with no copy by the CPU, while the session holds it. Each
//!   write of it, in a lend or by a device, is reported with its guest pages
//!   ([`crate::device_writes`]), so a graphics device's cache of the surface does not hide it.
//! - **Codecs.** The session decodes and encodes on the host's backend, which the device picks
//!   itself ([`backend_choice`]): VideoToolbox on macOS (reims-vgpu-video-videotoolbox); on
//!   Linux, on the GPU the library's GPU device runs on, NVIDIA's NVDEC and NVENC where it is an
//!   NVIDIA GPU that CUDA opens (reims-vgpu-video-nvidia), else VA-API on its render node
//!   (reims-vgpu-video-vaapi), and a hardware GPU of the host's where that GPU is a software
//!   rasterizer or offers no video. It offers what the driver decodes and encodes. The backend is
//!   opened once, at the device's creation, and the choice and its reason are logged; where none
//!   opens the session lists no codec, and a guest's VideoToolbox decodes and encodes in
//!   software. Its JPEG engine (`jcap`, `jdec`) decodes on the backend's JPEG decoder, VA-API's
//!   `VaapiJpegDecoder` on the decoders' display; where the backend has none (NVIDIA's and
//!   VideoToolbox's have none), the engine answers as Apple's host without JPEG hardware. Core
//!   copies a `jdec`'s source out of its read lend, decodes on a thread of its own with no surface
//!   lent, waiting 10 s at most for it, and copies the picture into the destination in a write
//!   lend after. The `jdec` is answered inside [`VideoDevice::deliver`], so QEMU's device delivers
//!   on a worker thread of its own, off the main loop.
//!
//! Every call runs as an entry ([`crate::entry`]); the device's last drop is a reset.

use std::fmt;
use std::ops::Range;
use std::sync::{Arc, OnceLock, Weak};

use reims_vgpu_core::iosurface::{HostDeviceCore, IoSurfaceHostDevice};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::os::surface::Surface as _;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_core::backend::{Backend, DecoderBackend};
use reims_vgpu_video_core::device_memory::DeviceMemory;
use reims_vgpu_video_core::encoder_backend::EncoderBackend;
use reims_vgpu_video_core::host::{GuestSink, HostSession};
use reims_vgpu_video_core::runtime::Runtime;
use reims_vgpu_video_core::status::Status;
use reims_vgpu_video_core::surfaces::{
    GuestSurface, MappingId, PlaneLayout, SurfaceError, SurfaceLayout, Surfaces,
};
use reims_vgpu_video_core::wire::message::read_header;

use crate::DefaultPlatform;
use crate::entry;
use crate::iosurface::IoSurfaceDevice;
use crate::platform::ApiPlatform;
use crate::retained::Ours;
use backend_choice::OpenBackends;

pub(crate) mod backend_choice;
mod logged;
pub(crate) mod runtime;
#[cfg(test)]
mod tests;

/// What [`VideoDevice::deliver`] answers while the device has no host session: VideoToolbox's
/// `kVTInvalidSessionErr`. Apple's host has no such answer: it hands the message to a session it
/// does not have.
pub const INVALID_SESSION: i32 = -12903;

/// The longest guest-to-host message [`VideoDevice::deliver`] takes: 16 MiB, ours. A guest's
/// descriptors may add up to many times the host's memory, which a host that gathers an element
/// before it delivers it would allocate and copy. The guest's VideoToolbox cuts a value across
/// messages at 0x200000 bytes ("Fragments", docs/video/devices.md), and the session gathers no
/// reply past 16 MiB, so no message of a guest it drives comes near this. A host checks it
/// before it copies an element out of the guest's memory, as QEMU's `apple-videotoolbox-pci`
/// does, and answers a longer one [`ALLOCATION_FAILED`] itself.
pub const MAX_MESSAGE_SIZE: usize = 16 << 20;

/// What [`VideoDevice::deliver`] answers a message longer than [`MAX_MESSAGE_SIZE`], unread:
/// VideoToolbox's `kVTAllocationFailedErr`, as the session answers a delivery past what it
/// holds of the guest's messages.
pub const ALLOCATION_FAILED: i32 = Status::AllocationFailed.code();

/// `send_to_guest(message)`: one host-to-guest message, the whole `vtpv` container (or `rels`
/// box), borrowed for the call. It arrives on a session's thread, or on the host's own inside
/// [`VideoDevice::deliver`] when the session answers at once, so it must not wait for a thread
/// that may be inside a call on the same device; it copies the message and queues it for the
/// guest.
pub type SendToGuest = Arc<dyn Fn(&[u8]) + Send + Sync>;

/// A video device's config.
pub struct VideoDeviceConfig<P: Platform = DefaultPlatform> {
    /// The IOSurface device whose mapping IDs the guest's messages name, held unretained; `None`
    /// is the process's latest at each lookup.
    pub iosurface_device: Option<IoSurfaceDevice<P>>,
    /// Where host-to-guest messages go. Without it they are dropped, as Apple's host drops them
    /// with no reply queue.
    pub send_to_guest: Option<SendToGuest>,
}

impl<P: Platform> Default for VideoDeviceConfig<P> {
    fn default() -> Self {
        VideoDeviceConfig {
            iosurface_device: None,
            send_to_guest: None,
        }
    }
}

impl<P: Platform> fmt::Debug for VideoDeviceConfig<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoDeviceConfig")
            .field("iosurface_device", &self.iosurface_device)
            .field("send_to_guest", &self.send_to_guest.is_some())
            .finish()
    }
}

/// One plane of a surface: its bytes' offset from the surface's base address, its size in
/// elements, and its bytes per element and per row ([`ApiPlatform::surface_planes`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SurfacePlane {
    /// Its bytes' offset from the surface's base address.
    pub offset: u64,
    /// Its width, in elements.
    pub width: u64,
    /// Its height, in elements.
    pub height: u64,
    /// Its bytes per element.
    pub bytes_per_element: u64,
    /// Its bytes per row.
    pub bytes_per_row: u64,
}

/// One guest IOSurface, retained through the mapper by its mapping ID: one reference to the
/// surface, so a hold the guest sees (an unmap of it answers Busy while this lives). Its release
/// runs as an entry.
///
/// Its bytes are the guest's own pages, as the host maps them: on Linux, a surface of several
/// pages is one view of them, contiguous from its base address, and a surface of one run of pages
/// is that run in the guest's RAM; on macOS, IOSurface's mapping of them at its base address,
/// which a lend reaches under `IOSurfaceLock` (`reims_vgpu_metal::os::surface_bytes`); a
/// host-only surface is memory of the host's own.
pub(crate) struct MappedSurface<P: ApiPlatform = DefaultPlatform> {
    mapping_id: u64,
    surface: Ours<P::Surface>,
    /// Its guest pages, host addresses in the guest's RAM, as runs: what a write of it writes
    /// ([`crate::device_writes`]); none for a host-only surface.
    pages: Vec<Range<u64>>,
    platform: Arc<P>,
    /// Its memory as the platform lends it to a device, asked for once
    /// ([`GuestSurface::device_memory`]).
    device: OnceLock<Option<Box<dyn DeviceMemory>>>,
}

impl<P: ApiPlatform> MappedSurface<P> {
    /// Its planes, in order; empty for a surface without planes, whose layout is its own.
    fn planes(&self) -> Vec<SurfacePlane> {
        self.platform.surface_planes(&self.surface)
    }
}

impl<P: ApiPlatform> fmt::Debug for MappedSurface<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format = self.surface.pixel_format().to_be_bytes();
        f.debug_struct("MappedSurface")
            .field("mapping_id", &self.mapping_id)
            .field("pixel_format", &FourCc(format))
            .field("width", &self.surface.width())
            .field("height", &self.surface.height())
            .field("bytes_per_row", &self.surface.bytes_per_row())
            .field("alloc_size", &self.surface.alloc_size())
            .field("planes", &self.planes())
            .finish_non_exhaustive()
    }
}

impl<P: ApiPlatform> GuestSurface for MappedSurface<P> {
    /// Its pixel format, size and planes, as the guest made it.
    fn layout(&self) -> SurfaceLayout {
        let size = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        let bytes = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
        SurfaceLayout {
            pixel_format: reims_vgpu_video_core::fourcc::FourCc(self.surface.pixel_format()),
            width: size(self.surface.width()),
            height: size(self.surface.height()),
            planes: (self.planes().iter())
                .map(|p| PlaneLayout {
                    offset: bytes(p.offset),
                    bytes_per_row: bytes(p.bytes_per_row),
                    width: size(p.width),
                    height: size(p.height),
                })
                .collect(),
            bytes_per_row: bytes(self.surface.bytes_per_row()),
        }
    }

    /// Lends its bytes, from its first byte to the end of its allocation, through the platform
    /// ([`ApiPlatform::lend_surface_bytes`]); a surface they cannot be lent of is unreachable.
    /// Once the lend returns the host's devices' writes have its pages
    /// ([`crate::device_writes`]).
    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
        self.platform
            .lend_surface_bytes(&self.surface, write)
            .map_err(|why| {
                log::warn!(
                    "surface {:#x}: its bytes cannot be lent: {why}",
                    self.mapping_id
                );
                SurfaceError::Unreachable(why.to_owned())
            })?;
        crate::device_writes::wrote(&self.pages);
        Ok(())
    }

    /// Lends its bytes for reading, laid out as [`Self::write`] lends them, through the platform
    /// ([`ApiPlatform::lend_surface_bytes_for_reading`]): a surface the guest made read-only is
    /// lent too. One they cannot be lent of is unreachable.
    fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
        self.platform
            .lend_surface_bytes_for_reading(&self.surface, read)
            .map_err(|why| {
                log::warn!(
                    "surface {:#x}: its bytes cannot be lent for reading: {why}",
                    self.mapping_id
                );
                SurfaceError::Unreachable(why.to_owned())
            })
    }

    /// Its memory as the platform lends it to a device ([`ApiPlatform::surface_device_memory`]),
    /// asked for once for this hold: the bytes [`Self::write`] lends, for the decoders' device to
    /// write a frame into while the session holds the surface. `None` where the platform lends
    /// none, said once a hold at debug.
    fn device_memory(&self) -> Option<&dyn DeviceMemory> {
        let lent = self.device.get_or_init(|| {
            entry::run(|_| self.platform.surface_device_memory(&self.surface))
                .inspect_err(|why| {
                    log::debug!(
                        "surface {:#x}: its memory is lent to no device: {why}",
                        self.mapping_id
                    );
                })
                .ok()
        });
        lent.as_deref()
    }

    /// A device wrote its memory ([`Self::device_memory`]): the host's devices' writes have its
    /// pages ([`crate::device_writes`]).
    fn device_wrote(&self) {
        crate::device_writes::wrote(&self.pages);
    }
}

/// A four-character code as it prints: its characters where all are printable, else its hex.
pub(crate) struct FourCc(pub(crate) [u8; 4]);

impl fmt::Debug for FourCc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
            let s: String = self.0.iter().map(|&b| char::from(b)).collect();
            write!(f, "'{s}'")
        } else {
            write!(f, "{:#010x}", u32::from_be_bytes(self.0))
        }
    }
}

/// The guest's surfaces, through the IOSurface device's mapper (module doc): the video
/// device's, and the scaler device's (`crate::scaler`).
pub(crate) struct MapperSurfaces<P: ApiPlatform> {
    platform: Arc<P>,
    /// The IOSurface device the config named, unretained; `None`: the process's latest.
    mapper: Option<Weak<HostDeviceCore<P>>>,
}

impl<P: ApiPlatform> MapperSurfaces<P> {
    /// `device`'s surfaces (`video device`, `scaler device`), mapped through `mapper`, held
    /// unretained, or through the process's latest IOSurface device at each lookup; logged.
    pub(crate) fn new(device: &str, platform: Arc<P>, mapper: Option<&IoSurfaceDevice<P>>) -> Self {
        log::info!(
            "{device} created, mapping IOSurfaces through {}",
            if mapper.is_some() {
                "its IOSurface device"
            } else {
                "the process's latest IOSurface device"
            }
        );
        MapperSurfaces {
            platform,
            mapper: mapper.map(IoSurfaceDevice::core_weak),
        }
    }

    /// Whether they are mapped through an IOSurface device the config named.
    pub(crate) fn names_its_mapper(&self) -> bool {
        self.mapper.is_some()
    }

    /// The surface the guest maps at `mapping_id`, retained through the mapper.
    fn retain(&self, mapping_id: u64) -> Result<MappedSurface<P>, SurfaceError> {
        let core = match &self.mapper {
            Some(weak) => weak.upgrade(),
            None => IoSurfaceHostDevice::latest_core(&*self.platform).and_then(|w| w.upgrade()),
        };
        let Some(core) = core else {
            return Err(SurfaceError::Unreachable(
                "no IOSurface device to map it through".to_owned(),
            ));
        };
        let surface = core.retain_surface_and_pages_with_mapping_id(mapping_id);
        drop(core);
        let (surface, pages) = surface.ok_or(SurfaceError::Unknown(mapping_id))?;
        let pages = crate::device_writes::page_runs(&pages, self.platform.vm().page_size());
        let surface = MappedSurface {
            mapping_id,
            surface: Ours::new(surface),
            pages,
            platform: Arc::clone(&self.platform),
            device: OnceLock::new(),
        };
        log::debug!("surface {mapping_id:#x} held: {surface:?}");
        Ok(surface)
    }
}

impl<P: ApiPlatform> Surfaces for MapperSurfaces<P> {
    fn retain(&self, mapping_id: MappingId) -> Result<Box<dyn GuestSurface>, SurfaceError> {
        entry::run(|_| MapperSurfaces::retain(self, mapping_id.0))
            .map(|s| Box::new(s) as Box<dyn GuestSurface>)
    }
}

/// The host's `send_to_guest`, as the session's sink: each message goes to it as it comes.
struct CallbackSink(Option<SendToGuest>);

impl GuestSink for CallbackSink {
    fn send(&self, message: Vec<u8>) {
        match read_header(&message, 0) {
            Ok(h) => log::debug!(
                "to the guest: {:?} flags {:#x} reply {:#x}, {} bytes",
                FourCc(h.message_type.to_be_bytes()),
                h.flags.0,
                h.reply_id,
                message.len()
            ),
            Err(_) => log::debug!("to the guest: {} bytes", message.len()),
        }
        if let Some(send) = &self.0 {
            send(&message);
        }
    }
}

/// The session's backend: the parts of `chosen`, the device's share of the host's backend
/// ([`backend_choice::choose`]), its decoders and encoders logged, and its JPEG decoder too. One
/// that did not open has no part, and lists no codec.
fn device_backend<P: Platform>(chosen: &backend_choice::Chosen, platform: &Arc<P>) -> Backend {
    if !chosen.has_backend() {
        log::info!(
            "video device: the guest is offered no codec, and its VideoToolbox decodes and \
             encodes in software"
        );
    }
    let mut backend = chosen.backend();
    backend.decoders = backend.decoders.map(|inner| {
        Arc::new(logged::LoggedBackend {
            inner,
            name: chosen.label,
            platform: Arc::clone(platform),
        }) as Arc<dyn DecoderBackend>
    });
    backend.encoders = backend.encoders.map(|inner| {
        Arc::new(logged::LoggedEncoders {
            inner,
            name: chosen.label,
            platform: Arc::clone(platform),
        }) as Arc<dyn EncoderBackend>
    });
    match &backend.jpeg {
        Some(jpeg) => log::info!(
            "video device decodes JPEG on {}: {:?}",
            chosen.label,
            jpeg.capabilities()
        ),
        None => log::info!(
            "video device: no JPEG decoder on {}: its JPEG engine answers as a host without one",
            chosen.label
        ),
    }
    backend
}

/// The host side of one VideoToolbox device (module doc). Its last drop is a reset.
pub struct VideoDevice<P: ApiPlatform = DefaultPlatform> {
    sink: Arc<CallbackSink>,
    surfaces: Arc<MapperSurfaces<P>>,
    /// The backend the host sessions run on, opened with the device.
    backend: Backend,
    /// The platform as the host sessions' runtime: their threads and their clock.
    runtime: Arc<dyn Runtime>,
    /// The host session, from `start` to the reset that completes its invalidation.
    session: Mutex<Option<Arc<HostSession>>>,
    /// The device's share of its backend, which another video device on the same GPU shares
    /// (`backend_choice`). Last, so that it goes after the session and the parts.
    _chosen: backend_choice::Chosen,
}

impl<P: ApiPlatform> fmt::Debug for VideoDevice<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VideoDevice")
            .field("mapper", &self.surfaces.names_its_mapper())
            .field("send_to_guest", &self.sink.0.is_some())
            .field("backend", &self.backend)
            .field("session", &self.session.lock().is_some())
            .finish()
    }
}

impl VideoDevice {
    /// A new video device on the process's platform ([`VideoDevice::with_platform`]).
    pub fn new(config: VideoDeviceConfig) -> VideoDevice {
        VideoDevice::with_platform(DefaultPlatform::process(), config)
    }
}

impl<P: ApiPlatform> VideoDevice<P> {
    /// A new video device on `platform`, with no host session until [`Self::start`]; its backend
    /// opens now (module doc). Nothing refuses one: a host with no codecs has a session that lists
    /// none.
    pub fn with_platform(platform: Arc<P>, config: VideoDeviceConfig<P>) -> VideoDevice<P> {
        let runtime = runtime::PlatformRuntime::of(&platform);
        VideoDevice::picking_from(&backend_choice::OPEN, platform, runtime, config)
    }

    /// [`Self::with_platform`], its backend picked from `open`, the process's or a test's own,
    /// its sessions and the backend it opens running on `runtime`: the platform's, or, in a test
    /// that opens the host's backend, std's, since a backend's waits for its GPU end only on a
    /// clock that runs.
    pub(crate) fn picking_from(
        open: &OpenBackends,
        platform: Arc<P>,
        runtime: Arc<dyn Runtime>,
        config: VideoDeviceConfig<P>,
    ) -> VideoDevice<P> {
        entry::run(|_| {
            VideoDevice::make(platform, runtime, config, |runtime| {
                backend_choice::choose(open, "video device", backend_choice::Need::Video, runtime)
            })
        })
    }

    /// A new video device on `platform` that opens none of the host's backends, `why` in words:
    /// its sessions list no codec, as on a host where none opens. For the tests of the device and
    /// of the C API, which run on core's testkit platform: its clock is virtual, and no test
    /// advances it while a backend waits for its GPU, so a backend opened there would wait at
    /// its first such wait for good.
    #[cfg(any(test, feature = "testkit"))]
    pub fn without_backend(
        platform: Arc<P>,
        config: VideoDeviceConfig<P>,
        why: &str,
    ) -> VideoDevice<P> {
        entry::run(|_| {
            let runtime = runtime::PlatformRuntime::of(&platform);
            VideoDevice::make(platform, runtime, config, |_| {
                backend_choice::Chosen::none("video device", why)
            })
        })
    }

    /// The device, its backend the one `choose` gives it on `runtime`, after its surfaces.
    fn make(
        platform: Arc<P>,
        runtime: Arc<dyn Runtime>,
        config: VideoDeviceConfig<P>,
        choose: impl FnOnce(&Arc<dyn Runtime>) -> backend_choice::Chosen,
    ) -> VideoDevice<P> {
        let surfaces = MapperSurfaces::new(
            "video device",
            Arc::clone(&platform),
            config.iosurface_device.as_ref(),
        );
        let chosen = choose(&runtime);
        let backend = device_backend(&chosen, &platform);
        VideoDevice {
            sink: Arc::new(CallbackSink(config.send_to_guest)),
            surfaces: Arc::new(surfaces),
            backend,
            runtime,
            session: Mutex::new(None),
            _chosen: chosen,
        }
    }

    /// The guest's DRIVER_OK: a new host session. A session already there is ended for it (both
    /// phases), as Apple's host replaces its own, which may wait for its threads as
    /// [`Self::reset`] does.
    pub fn start(&self) {
        entry::run(|_| {
            let session = Arc::new(HostSession::with_backend(
                Arc::clone(&self.runtime),
                Arc::clone(&self.sink) as Arc<dyn GuestSink>,
                Arc::clone(&self.surfaces) as Arc<dyn Surfaces>,
                &self.backend,
            ));
            let old = self.session.lock().replace(session);
            log::info!("video device started: a new host session");
            if let Some(old) = old {
                old.invalidate();
                old.complete_invalidate();
            }
        });
    }

    /// One guest-to-host message: the session holds the surfaces its `subs` atom names while it
    /// handles it, and answers the status the host writes back (module doc). Without a session,
    /// [`INVALID_SESSION`]; for a message longer than [`MAX_MESSAGE_SIZE`], [`ALLOCATION_FAILED`],
    /// and the session sees nothing of it. May call back (`send_to_guest`, on this thread). May
    /// wait for the IOSurface device's thread, and may take a while: a `jdec` waits for its
    /// decode, and a session's end (`-dec`, `-enc`, `-mep`) for its thread, 10 s at most each.
    pub fn deliver(&self, message: &[u8]) -> i32 {
        entry::run(|_| {
            if message.len() > MAX_MESSAGE_SIZE {
                log::warn!(
                    "a guest message of {} bytes, over the {MAX_MESSAGE_SIZE} a message may hold: \
                     answered {ALLOCATION_FAILED}, unread",
                    message.len()
                );
                return ALLOCATION_FAILED;
            }
            let header = read_header(message, 0).ok();
            let session = self.session.lock().clone();
            let status = match session {
                Some(session) => session.deliver(message),
                None => {
                    log::warn!(
                        "a guest message of {} bytes with no host session: answered \
                         {INVALID_SESSION}",
                        message.len()
                    );
                    INVALID_SESSION
                }
            };
            match header {
                Some(h) => log::debug!(
                    "from the guest: {:?} flags {:#x} reply {:#x}, {} bytes: answered {status}",
                    FourCc(h.message_type.to_be_bytes()),
                    h.flags.0,
                    h.reply_id,
                    message.len()
                ),
                None => log::debug!(
                    "from the guest: {} bytes with no message header: answered {status}",
                    message.len()
                ),
            }
            status
        })
    }

    /// The first phase of a reset or a stop: the session sends the guest nothing more, and
    /// answers no new message. Each of its decoder, encoder and motion-estimation sessions is
    /// told on its thread, after the message in hand there, and this waits for them, 10 s at
    /// most in all, as core's `HostSession::invalidate` does.
    pub fn invalidate(&self) {
        entry::run(|_| {
            let session = self.session.lock().clone();
            if let Some(session) = session {
                session.invalidate();
            }
        });
    }

    /// A reset or a stop, both phases: the session is invalidated, if it was not, then its work
    /// drains, and it is dropped. May wait for the session's threads: 10 s at most for them to
    /// take the invalidation, as [`Self::invalidate`] waits, then for those that took it to
    /// end. One left running past the bound is left detached.
    pub fn reset(&self) {
        entry::run(|_| self.end_session());
    }

    /// Both phases of the session's end, if there is one.
    fn end_session(&self) {
        let session = self.session.lock().take();
        if let Some(session) = session {
            session.invalidate();
            session.complete_invalidate();
            log::info!("video device reset: the host session ended");
        }
    }
}

impl<P: ApiPlatform> Drop for VideoDevice<P> {
    /// A reset, as an entry.
    fn drop(&mut self) {
        entry::run(|_| self.end_session());
    }
}
