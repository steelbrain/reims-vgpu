//! [`Device`]: one paravirtualized GPU, made from a [`DeviceConfig`].
//!
//! A [`Device`] is one counted reference: [`Device::clone_ref`] takes another, and the last
//! drop, or the last drop of what keeps it alive (a [`Display`]), is the device's teardown, run
//! on the releasing thread, where it may block. Every method may be called from any thread;
//! each runs as an entry ([`crate::entry`]), and so does the last drop: marked as the library's,
//! so a panic in the library aborts at its op instead of unwinding into the caller, and with
//! what the call held or Metal autoreleased on the thread released before it returns.
//!
//! The host's callbacks are closures ([`HostCallbacks`]), called synchronously on the thread
//! core calls them on, with the arguments core passes, in core's order: the order of VM-memory
//! operations is core's (docs/api.md section 5). A refused create drops the closures it was given
//! before it returns.
//!
//! A device made over a memory map holds what its platform holds of the guest's RAM for it
//! ([`ApiPlatform::guest_memory`], on Linux the files behind the map's ranges, which the Vulkan
//! platform's views map), from before core's device is made to after core's teardown: every
//! reference to the device and each of its displays keep the lease, and each drops it after its
//! core handle.
//!
//! A device made over a memory map with a host's dirty source ([`DeviceConfig::dirty_source`])
//! keeps core's dirty log of the guest's RAM in that lease, harvested from the source on a thread
//! of its own ([`crate::dirty`]), and hands the log to the platform's lease. The log is closed
//! while each call that stops or rebuilds the device's work runs (reset, pause, stop, suspend and
//! resume), so no wait inside one is left for a harvest, and retired with the lease.

use std::any::Any;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use reims_vgpu_core::device::Device as CoreDevice;
use reims_vgpu_core::device::config::DeviceConfig as CoreDeviceConfig;
use reims_vgpu_core::host::{
    AddTraceRange, CreateTask, DestroyTask, HostBlocks, HostPtr, MapMemory, RaiseInterrupt,
    ReadMemory, RemoveTraceRange, UnmapMemory,
};
use reims_vgpu_core::memory::descriptor::{
    GuestPhysicalRange, MemoryMapDescriptor as CoreMemoryMap,
};
use reims_vgpu_core::memory::dirty::DirtyLog;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::rt::Hold;

use crate::DefaultPlatform;
use crate::dirty::{DirtySource, Harvester};
use crate::display::{Display, DisplayConfig};
use crate::entry;
use crate::error::{Error, ResumeError};
use crate::frame::{FrameTarget, PixelFormat};
use crate::platform::{ApiPlatform, api_state};
use crate::retained::Ours;

/// Guest RAM and the host memory behind it, in order, with the file behind each range where the
/// host names one. The ranges are copied into the device at its creation; nothing is checked
/// until then.
#[derive(Clone, Default, Debug)]
pub struct MemoryMap {
    ranges: Vec<GuestPhysicalRange>,
    /// One entry per range, in the ranges' order.
    files: Vec<Option<MemoryFile>>,
}

/// The file behind one range of a [`MemoryMap`]: the range's host memory is a shared mapping
/// (`MAP_SHARED`) of `fd` from byte `offset`.
#[derive(Clone, Debug)]
pub struct MemoryFile {
    /// The file, held open while any map or device holds it.
    pub fd: Arc<OwnedFd>,
    /// The file offset of the range's first byte, a multiple of the host's page size.
    pub offset: u64,
}

impl MemoryMap {
    /// An empty map.
    pub fn new() -> MemoryMap {
        MemoryMap::default()
    }

    /// Appends one range with no file, as given (no check, no merge).
    ///
    /// # Safety
    /// As [`MemoryMap::push_with_file`].
    pub unsafe fn push(&mut self, guest_physical_address: u64, length: u64, host: *mut u8) {
        // SAFETY: the caller's contract.
        unsafe { self.push_with_file(guest_physical_address, length, host, None) }
    }

    /// Appends one range and the file behind it, if any, as given (no check, no merge).
    ///
    /// # Safety
    /// `host .. host + length` is host memory (`mach_vm_allocate`d or equivalent) that stays
    /// mapped while any device or IOSurface device made over the map lives: the device remaps
    /// it and reads and writes the guest's memory through it. With `file`, that memory is a
    /// shared mapping of `file.fd` from `file.offset`, and the file holds at least `length`
    /// bytes from there.
    pub unsafe fn push_with_file(
        &mut self,
        guest_physical_address: u64,
        length: u64,
        host: *mut u8,
        file: Option<MemoryFile>,
    ) {
        self.ranges.push(GuestPhysicalRange {
            physical_address: guest_physical_address,
            physical_length: length,
            host_address: HostPtr::from_ptr(host),
        });
        self.files.push(file);
    }

    /// The ranges, in order.
    pub fn ranges(&self) -> &[GuestPhysicalRange] {
        &self.ranges
    }

    /// The file behind each range, in the ranges' order: `None` for a range with no file.
    pub fn files(&self) -> &[Option<MemoryFile>] {
        &self.files
    }

    /// Core's descriptor of the map.
    pub(crate) fn to_core(&self) -> CoreMemoryMap {
        let mut m = CoreMemoryMap::new();
        for r in &self.ranges {
            m.add_range(*r);
        }
        m
    }
}

/// The device's callbacks. `None` is a callback the host does not give: the device behaves as
/// for a NULL C function pointer (docs/api.md section 5, and each C typedef's comment).
#[derive(Clone, Default)]
pub struct HostCallbacks {
    /// `(vm_size, &mut base) -> task`: reserve `vm_size` bytes of host address space.
    pub create_task: Option<CreateTask>,
    /// `(task)`: the task's last call.
    pub destroy_task: Option<DestroyTask>,
    /// `(task, range_count, virtual_offset, read_only, ranges) -> ok`.
    pub map_memory: Option<MapMemory>,
    /// `(task, virtual_offset, length) -> ok`.
    pub unmap_memory: Option<UnmapMemory>,
    /// `(physical_address, length, dst) -> ok`: copy guest memory into `dst`.
    pub read_memory: Option<ReadMemory>,
    /// `(vector)`, under the device's signal lock. Without it, the first interrupt ends the
    /// process.
    pub raise_interrupt: Option<RaiseInterrupt>,
    /// `(range, handler) -> trace range`: watch guest writes to the EFI framebuffer.
    pub add_trace_range: Option<AddTraceRange>,
    /// `(trace range)`: stop watching.
    pub remove_trace_range: Option<RemoveTraceRange>,
}

impl HostCallbacks {
    /// Core's blocks: the same closures.
    fn into_core(self) -> HostBlocks {
        HostBlocks {
            raise_interrupt: self.raise_interrupt,
            create_task: self.create_task,
            destroy_task: self.destroy_task,
            map_memory: self.map_memory,
            unmap_memory: self.unmap_memory,
            read_memory: self.read_memory,
            add_trace_range: self.add_trace_range,
            remove_trace_range: self.remove_trace_range,
        }
    }
}

impl std::fmt::Debug for HostCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostCallbacks")
            .field("create_task", &self.create_task.is_some())
            .field("destroy_task", &self.destroy_task.is_some())
            .field("map_memory", &self.map_memory.is_some())
            .field("unmap_memory", &self.unmap_memory.is_some())
            .field("read_memory", &self.read_memory.is_some())
            .field("raise_interrupt", &self.raise_interrupt.is_some())
            .field("add_trace_range", &self.add_trace_range.is_some())
            .field("remove_trace_range", &self.remove_trace_range.is_some())
            .finish()
    }
}

/// A device's config.
#[derive(Clone, Debug)]
pub struct DeviceConfig {
    /// The Metal device's registry ID; `None` is the system's default Metal device.
    pub metal_registry_id: Option<u64>,
    /// 1 to [`crate::MAX_DISPLAY_PORTS`]; 0 is refused. Default 1.
    pub display_port_count: u32,
    /// The feature level: 0 and 4 offer every feature; 1, 2 and 3 cap the binary version the
    /// guest may choose at 62, 113 and 209, and trim the device info it reads. A level above 4
    /// is accepted here; the guest's first binary version write then ends the process, as do
    /// [`Device::reset`] and [`Device::will_resume`] once it applies the state's version.
    pub feature_level: u64,
    /// The display port connection types the guest reads (0: none set); below binary version 245
    /// it reads their low 2 bits. A resume of a state from binary version 238 or later refuses
    /// one taken with other types.
    pub display_port_connection_types: u32,
    /// The guest's RAM, or `None` to reach it through the task and memory callbacks.
    pub memory_map: Option<MemoryMap>,
    /// Map guest IOSurfaces through the process's latest [`crate::IoSurfaceDevice`].
    pub use_iosurface_mapper: bool,
    /// Run each isolated guest task with the task service's semantics (needs a memory map).
    pub process_isolation: bool,
    /// With `process_isolation`: isolate a task other than task 0 only when, as the guest
    /// defines it, the [`crate::IoSurfaceDevice`] the device maps through has a descriptor table
    /// base other than 0; otherwise the task runs as without isolation. The guest's mapper
    /// driver writes the base at its start, before Apple's arm64 GPU driver makes a task. Apple's
    /// x86-64 driver never touches the IOSurface device, and references IOSurfaces through task
    /// 0's backings, which an isolated task does not serve. Device info's argument-buffer answer
    /// follows the same rule, read when the guest asks, since only isolated tasks serve argument
    /// buffers. Default `false`.
    pub process_isolation_follows_guest: bool,
    /// Offer argument buffers: they reach the guest only with process isolation, on a Metal
    /// device that can detach backing and is not itself paravirtual, and, with
    /// `process_isolation_follows_guest`, only while the IOSurface device has a descriptor
    /// table. Default `true`.
    pub argument_buffers: bool,
    /// The callbacks.
    pub callbacks: HostCallbacks,
    /// With `memory_map`: the host's dirty source, which tells the device which of the guest's
    /// pages the guest wrote (QEMU under KVM), so that the Vulkan runtime copies again only
    /// those; `None` for none, and without a memory map. Default `None`.
    pub dirty_source: Option<DirtySource>,
}

impl Default for DeviceConfig {
    fn default() -> DeviceConfig {
        DeviceConfig {
            metal_registry_id: None,
            display_port_count: 1,
            feature_level: 0,
            display_port_connection_types: 0,
            memory_map: None,
            use_iosurface_mapper: false,
            process_isolation: false,
            process_isolation_follows_guest: false,
            argument_buffers: true,
            callbacks: HostCallbacks::default(),
            dirty_source: None,
        }
    }
}

impl DeviceConfig {
    /// Core's config over `device`: the same closures, the map's ranges copied.
    fn into_core<P: Platform>(self, device: Option<P::Device>) -> CoreDeviceConfig<P> {
        CoreDeviceConfig {
            device,
            display_port_count: self.display_port_count,
            blocks: self.callbacks.into_core(),
            memory_map_descriptor: self.memory_map.as_ref().map(|m| Arc::new(m.to_core())),
            using_iosurface_mapper: self.use_iosurface_mapper,
            enable_argument_buffers: self.argument_buffers,
            enable_process_isolation: self.process_isolation,
            process_isolation_follows_guest: self.process_isolation_follows_guest,
            device_feature_level: self.feature_level,
            display_port_connection_types: self.display_port_connection_types,
            ..CoreDeviceConfig::default()
        }
    }
}

/// The platform's lease on the guest's RAM for one device ([`ApiPlatform::guest_memory`]),
/// shared by the device's references and its displays.
pub(crate) type GuestMemoryLease = Arc<dyn Any + Send + Sync>;

/// Whether a device takes its host's dirty source: unless `REIMS_VGPU_DIRTY_PAGE_TRACKING` is `0`,
/// read once per process.
fn dirty_page_tracking_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("REIMS_VGPU_DIRTY_PAGE_TRACKING").is_none_or(|v| v != "0"))
}

/// What a device over a memory map holds of the guest's RAM: the platform's lease, then the
/// harvest of the host's dirty source, whose drop retires the log after the lease has gone.
struct Lease {
    _platform: Box<dyn Any + Send + Sync>,
    _dirty: Option<Harvester>,
}

/// One reference to a device. `P` is the platform: [`DefaultPlatform`] in the product, core's
/// fake platform in tests.
pub struct Device<P: Platform = DefaultPlatform> {
    pub(crate) handle: Ours<CoreDevice<P>>,
    /// The platform the device was made on.
    platform: Arc<P>,
    /// The platform's lease on the guest's RAM, for a device made over a memory map. Declared
    /// after `handle`, so a reference drops it after its core handle: the last drop, core's
    /// teardown among them, runs before the lease goes.
    guest_memory: Option<GuestMemoryLease>,
    /// The dirty log of a device made with a dirty source (the module docs).
    dirty_log: Option<Arc<DirtyLog>>,
}

impl<P: Platform> std::fmt::Debug for Device<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Device")
            .field("core", &Arc::as_ptr(&**self.handle))
            .finish()
    }
}

impl Device {
    /// A new device on the process's platform ([`Device::with_platform`]).
    pub fn new(config: DeviceConfig) -> Result<Device, Error> {
        Device::with_platform(DefaultPlatform::process(), config)
    }
}

impl<P: ApiPlatform> Device<P> {
    /// A new device on `platform`. Refusals, in core's order: [`Error::DisplayPortCount`],
    /// [`Error::NoMetalDevice`], [`Error::MemoryConflict`], [`Error::MemoryMap`],
    /// [`Error::InvalidArgument`] (`add_trace_range` without `remove_trace_range`),
    /// [`Error::Metal`], [`Error::IsolationWithoutMap`]. The first device given a Metal device
    /// makes the process's frame queue for it, before the device is built, even when the device
    /// is then refused. A refusal has dropped every closure of the config when it returns.
    ///
    /// With a memory map, the platform's lease on the guest's RAM
    /// ([`ApiPlatform::guest_memory`]) is taken before core's device is made, and a refusal
    /// drops it before it returns.
    pub fn with_platform(platform: Arc<P>, config: DeviceConfig) -> Result<Device<P>, Error> {
        entry::run(|_| {
            let metal = platform.metal_device(config.metal_registry_id);
            if let Some(metal) = &metal {
                drop(api_state(&platform).frame_queue(metal));
            }
            let mut guest_memory: Option<GuestMemoryLease> = None;
            let mut dirty_log = None;
            if let Some(map) = &config.memory_map {
                let dirty = (config.dirty_source.clone())
                    .filter(|_| dirty_page_tracking_on())
                    .map(|source| {
                        let ram: Vec<_> = (map.ranges().iter())
                            .map(|r| r.physical_address..r.physical_address + r.physical_length)
                            .collect();
                        Harvester::new(&ram, source)
                    });
                dirty_log = dirty.as_ref().map(|h| h.log().clone());
                guest_memory = Some(Arc::new(Lease {
                    _platform: platform.guest_memory(map, dirty_log.clone()),
                    _dirty: dirty,
                }));
            }
            let core = config.into_core::<P>(metal);
            match CoreDevice::new(platform.clone(), core) {
                Ok(handle) => Ok(Device {
                    handle: Ours::new(handle),
                    platform: platform.clone(),
                    guest_memory,
                    dirty_log,
                }),
                Err(refusal) => Err(refusal.into()),
            }
        })
    }

    /// A new display. Refusals, in core's order: [`Error::PortOutOfRange`], [`Error::Metal`],
    /// [`Error::SerialInUse`], [`Error::PortInUse`]. The display keeps the device alive. May
    /// call back. With a window ([`DisplayConfig::window`]), the display has every callback as
    /// the guest reads them, and a refusal closes the window before it returns.
    pub fn new_display(&self, config: DisplayConfig) -> Result<Display<P>, Error> {
        Display::new(self, config)
    }
}

impl<P: Platform> Device<P> {
    /// Another reference to this device. The device ends when the last one drops.
    #[must_use]
    pub fn clone_ref(&self) -> Device<P> {
        Device {
            handle: Ours::new(self.handle.clone_ref()),
            platform: self.platform.clone(),
            guest_memory: self.guest_memory.clone(),
            dirty_log: self.dirty_log.clone(),
        }
    }

    /// The platform's lease on the guest's RAM, for a device made over a memory map: what a
    /// display keeps beside its core display, which keeps core's device alive.
    pub(crate) fn guest_memory_lease(&self) -> Option<GuestMemoryLease> {
        self.guest_memory.clone()
    }

    /// Whether both are references to one device.
    pub fn ptr_eq(&self, other: &Device<P>) -> bool {
        self.handle.ptr_eq(&other.handle)
    }

    /// Core's device (tests).
    #[cfg(any(test, feature = "testkit"))]
    pub fn core(&self) -> &CoreDevice<P> {
        &self.handle
    }

    /// The platform the device was made on.
    pub(crate) fn platform(&self) -> &Arc<P> {
        &self.platform
    }

    /// Runs `body` as an entry ([`entry::run`]), with a hold of the entry's holder.
    fn entry<T>(&self, body: impl for<'u> FnOnce(Hold<'u>, &CoreDevice<P>) -> T) -> T {
        entry::run(|hold| body(hold, &self.handle))
    }

    /// As [`Device::entry`], with the dirty log closed while `body` runs (the module docs).
    fn closed_entry<T>(&self, body: impl for<'u> FnOnce(Hold<'u>, &CoreDevice<P>) -> T) -> T {
        let Some(log) = &self.dirty_log else {
            return self.entry(body);
        };
        log.close();
        let r = self.entry(body);
        log.reopen();
        r
    }

    /// A 32-bit read at `offset` in the device's MMIO region. May block and may call back.
    pub fn mmio_read(&self, offset: u64) -> u32 {
        self.entry(|_, d| d.mmio_read(offset))
    }

    /// A 32-bit write at `offset`. May block: a root FIFO write waits for the FIFO's thread,
    /// and `raise_interrupt` may run inside it.
    pub fn mmio_write(&self, offset: u64, value: u32) {
        self.entry(|hold, d| d.mmio_write(hold, offset, value));
    }

    /// Resets the device as the guest's bus reset does: stops it as [`Device::stop`] does, sets
    /// the guest's binary version back to 0 and turns EFI off, and returns once it is quiesced.
    /// Blocks as [`Device::stop`] does, and for the EFI display's present thread. At a
    /// `feature_level` above 4 it ends the process.
    pub fn reset(&self) {
        self.closed_entry(|hold, d| d.reset(hold));
    }

    /// Pauses the device: each FIFO finishes the commands it started and stops, then the guest's
    /// displays pause. Blocks until each FIFO's thread has stopped and those commands have
    /// completed, and hangs while one never does. Where the guest has given a port a display
    /// page, a second pause before [`Device::unpause`] ends the process.
    pub fn pause(&self) {
        self.closed_entry(|_, d| d.pause());
    }

    /// Resumes the device after [`Device::pause`]: unpauses the guest's displays, starts every
    /// FIFO and raises one interrupt. Blocks while the FIFOs start on the library's FIFO control
    /// thread. Where the guest has given a port a display page, an unpause while the guest's
    /// displays are not paused ends the process: [`Device::pause`] and
    /// [`Device::finish_suspend`] pause them, and [`Device::will_suspend`] does not.
    pub fn unpause(&self) {
        self.entry(|_, d| d.unpause());
    }

    /// Stops the device as a root page of 0 does: its FIFOs, the guest's tasks and the guest's
    /// display pages are torn down. Blocks until each FIFO's thread has stopped and the commands
    /// it started have completed.
    pub fn stop(&self) {
        self.closed_entry(|hold, d| d.stop(hold));
    }

    /// Starts a suspend: from here the device raises no interrupt, and drops those not yet
    /// raised. It quiesces the FIFOs without waiting for them, and pauses none of the guest's
    /// displays.
    pub fn will_suspend(&self) {
        self.closed_entry(|_, d| d.will_suspend());
    }

    /// Ends a suspend: the device's state. The device always has one to give, so
    /// [`Error::Suspend`] never comes. Without [`Device::will_suspend`] before it, it ends the
    /// process. Blocks until each FIFO's thread has stopped and the commands it started have
    /// completed, then flushes and ends the guest's tasks and pauses the guest's displays.
    pub fn finish_suspend(&self) -> Result<Vec<u8>, Error> {
        self.closed_entry(|_, d| d.finish_suspend())
            .ok_or(Error::Suspend)
    }

    /// Offers the device a state [`Device::finish_suspend`] gave, before the guest runs again.
    /// `None` is no state, which the device refuses as invalid content. A refusal is
    /// [`Error::Resume`] with the device's reason, or `None` where it stored none. From this call
    /// until [`Device::did_resume`], whether or not it accepted the state, a new mapping of guest
    /// memory (the guest's root page write, for one) ends the process. A state whose binary
    /// version has no EFI display turns EFI off, which waits for the EFI display's present thread.
    pub fn will_resume(&self, state: Option<&[u8]>) -> Result<(), Error> {
        self.closed_entry(|_, d| {
            let mut code = None;
            if d.will_resume(state, Some(&mut code)) {
                Ok(())
            } else {
                Err(Error::Resume(code.and_then(ResumeError::from_code)))
            }
        })
    }

    /// Ends a resume [`Device::will_resume`] accepted: rebuilds the device from the state, starts
    /// its FIFOs and raises one interrupt. Rebuilding the EFI display waits for its present
    /// thread, and a root the device still has is torn down as [`Device::stop`] does.
    pub fn did_resume(&self) {
        self.closed_entry(|hold, d| d.did_resume(hold));
    }

    /// A frame target on this device's Metal device: `width` x `height` pixels of `format`,
    /// each 1 to 16384 ([`Error::InvalidArgument`] otherwise). It holds no reference to the
    /// device.
    pub fn new_frame_target(
        &self,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<FrameTarget<P>, Error> {
        FrameTarget::new(
            self.platform(),
            self.handle.mtl_device_ref(),
            width,
            height,
            format,
        )
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::rt::is_our_thread;
    use reims_vgpu_core::testkit::host::FakeHost;
    use reims_vgpu_core::testkit::{FakePlatform, GuestRam};

    use super::*;

    /// A fake host's callbacks as the API's.
    pub(crate) fn callbacks(host: &Arc<FakeHost>) -> HostCallbacks {
        let b = host.blocks();
        HostCallbacks {
            create_task: b.create_task,
            destroy_task: b.destroy_task,
            map_memory: b.map_memory,
            unmap_memory: b.unmap_memory,
            read_memory: b.read_memory,
            raise_interrupt: b.raise_interrupt,
            add_trace_range: b.add_trace_range,
            remove_trace_range: b.remove_trace_range,
        }
    }

    fn fake(
        ports: u32,
    ) -> (
        Arc<FakePlatform>,
        Arc<FakeHost>,
        Result<Device<FakePlatform>, Error>,
    ) {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).unwrap());
        let c = DeviceConfig {
            display_port_count: ports,
            callbacks: callbacks(&host),
            ..DeviceConfig::default()
        };
        let d = Device::with_platform(p.clone(), c);
        (p, host, d)
    }

    /// The defaults: one port, argument buffers offered, everything else off
    #[test]
    fn config_defaults() {
        let c = DeviceConfig::default();
        assert_eq!(c.display_port_count, 1);
        assert!(c.argument_buffers && !c.process_isolation && !c.use_iosurface_mapper);
        assert!(!c.process_isolation_follows_guest);
        assert_eq!((c.feature_level, c.display_port_connection_types), (0, 0));
        assert!(c.metal_registry_id.is_none() && c.memory_map.is_none());
        let core = c.into_core::<FakePlatform>(None);
        let want = CoreDeviceConfig::<FakePlatform>::default();
        assert_eq!(core.display_port_count, want.display_port_count);
        assert_eq!(core.enable_argument_buffers, want.enable_argument_buffers);
        assert_eq!(core.mmio_length, want.mmio_length);
        assert!(!core.process_isolation_follows_guest);
    }

    /// Isolation and its following the guest reach core's config as set
    #[test]
    fn isolation_fields_reach_core() {
        for (isolate, follows) in [(true, true), (true, false), (false, true)] {
            let c = DeviceConfig {
                process_isolation: isolate,
                process_isolation_follows_guest: follows,
                ..DeviceConfig::default()
            };
            let core = c.into_core::<FakePlatform>(None);
            assert_eq!(
                (
                    core.enable_process_isolation,
                    core.process_isolation_follows_guest
                ),
                (isolate, follows)
            );
        }
    }

    /// Each refusal names its check; the closures of a refused config are dropped
    #[test]
    fn refusals() {
        assert_eq!(fake(0).2.err(), Some(Error::DisplayPortCount));
        assert_eq!(fake(9).2.err(), Some(Error::DisplayPortCount));
        assert!(fake(8).2.is_ok());
        let p = Arc::new(FakePlatform::new());
        let raise: RaiseInterrupt = Arc::new(|_| {});
        let c = DeviceConfig {
            metal_registry_id: Some(1),
            callbacks: HostCallbacks {
                raise_interrupt: Some(raise.clone()),
                ..HostCallbacks::default()
            },
            ..DeviceConfig::default()
        };
        assert_eq!(
            Device::with_platform(p.clone(), c).err(),
            Some(Error::NoMetalDevice)
        );
        assert_eq!(Arc::strong_count(&raise), 1);
        // add_trace_range alone is an invalid argument.
        let host = FakeHost::new(&p, GuestRam::new());
        let mut cb = callbacks(&host);
        cb.remove_trace_range = None;
        let c = DeviceConfig {
            callbacks: cb,
            ..DeviceConfig::default()
        };
        assert_eq!(
            Device::with_platform(p.clone(), c).err(),
            Some(Error::InvalidArgument)
        );
        // A memory map beside the task callbacks.
        let c = DeviceConfig {
            memory_map: Some(MemoryMap::new()),
            callbacks: callbacks(&host),
            ..DeviceConfig::default()
        };
        assert_eq!(
            Device::with_platform(p.clone(), c).err(),
            Some(Error::MemoryConflict)
        );
        let c = DeviceConfig {
            process_isolation: true,
            callbacks: callbacks(&host),
            ..DeviceConfig::default()
        };
        assert_eq!(
            Device::with_platform(p, c).err(),
            Some(Error::IsolationWithoutMap)
        );
    }

    /// The API answers what core answers, and a write reaches the device; clone_ref is another
    /// reference and the last drop ends the device
    #[test]
    fn mmio_and_references() {
        let (_p, _h, d) = fake(1);
        let d = d.unwrap();
        for off in (0..0x100).step_by(4) {
            assert_eq!(d.mmio_read(off), d.core().mmio_read(off), "{off:#x}");
        }
        d.mmio_write(0x2000, 0);
        let d2 = d.clone_ref();
        assert!(d.ptr_eq(&d2));
        let weak = Arc::downgrade(&**d.core());
        drop(d);
        assert!(weak.upgrade().is_some());
        drop(d2);
        assert!(weak.upgrade().is_none());
        assert!(!is_our_thread());
    }

    /// A device over a memory map holds the platform's lease on the guest's RAM, which its
    /// references share; a device without a map holds none
    #[test]
    fn a_memory_map_takes_the_platform_s_lease() {
        let (_p, _h, d) = fake(1);
        assert!(d.unwrap().guest_memory_lease().is_none());
        let p = Arc::new(FakePlatform::new());
        let c = DeviceConfig {
            memory_map: Some(MemoryMap::new()),
            callbacks: HostCallbacks {
                raise_interrupt: Some(Arc::new(|_| {})),
                ..HostCallbacks::default()
            },
            ..DeviceConfig::default()
        };
        let d = Device::with_platform(p, c).unwrap();
        let lease = d.guest_memory_lease().unwrap();
        let d2 = d.clone_ref();
        assert!(Arc::ptr_eq(&d2.guest_memory_lease().unwrap(), &lease));
        assert_eq!(Arc::strong_count(&lease), 3);
        drop((d, d2));
        assert_eq!(Arc::strong_count(&lease), 1);
    }

    /// No state is refused as invalid content; a suspend gives a state the device accepts
    #[test]
    fn suspend_and_resume() {
        let (_p, _h, d) = fake(1);
        let d = d.unwrap();
        assert_eq!(
            d.will_resume(None),
            Err(Error::Resume(Some(ResumeError::InvalidContent)))
        );
        d.will_suspend();
        let state = d.finish_suspend().unwrap();
        assert_eq!(d.will_resume(Some(&state)), Ok(()));
        d.did_resume();
        d.pause();
        d.unpause();
        d.reset();
        d.stop();
    }

    /// The memory map keeps its ranges in order, and the file behind each beside it
    #[test]
    fn memory_map_ranges() {
        let mut m = MemoryMap::new();
        let mut page = [0u8; 8];
        let fd = Arc::new(OwnedFd::from(std::fs::File::open("/dev/null").unwrap()));
        let file = MemoryFile {
            fd: fd.clone(),
            offset: 0x8000,
        };
        // SAFETY: nothing is built over this map.
        unsafe {
            m.push(0x4000, 0x4000, page.as_mut_ptr());
            m.push_with_file(0, 0x4000, page.as_mut_ptr(), Some(file));
        }
        let got: Vec<u64> = m.ranges().iter().map(|r| r.physical_address).collect();
        assert_eq!(got, [0x4000, 0]);
        assert_eq!(m.to_core().ranges(), m.ranges());
        let files: Vec<Option<u64>> = m
            .files()
            .iter()
            .map(|f| f.as_ref().map(|f| f.offset))
            .collect();
        assert_eq!(files, [None, Some(0x8000)]);
        assert!(Arc::ptr_eq(&m.files()[1].as_ref().unwrap().fd, &fd));
        // A clone of the map holds the same file.
        let copy = m.clone();
        assert_eq!(Arc::strong_count(&fd), 3);
        drop((m, copy));
        assert_eq!(Arc::strong_count(&fd), 1);
    }
}
