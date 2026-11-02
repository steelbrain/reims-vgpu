//! The host's object: its creation, the process's last device, the surface lookup the graphics
//! device makes, and the teardown.
//!
//! Two tiers. [`IoSurfaceHostDevice`] is the host's reference: its last drop is the
//! teardown, at once, on the releasing thread. The state is [`HostDeviceCore`], which the
//! doorbell's handler and every queue block also hold, so neither keeps the host's device alive,
//! and a handler queued ahead of the teardown's wait for the queue still runs, on what the
//! teardown left.
//!
//! The six registers are seq_cst atomics ([`regs`](super::regs)). What the queue works on is
//! [`QueueState`], behind a mutex that each command, register write and surface lookup holds for
//! its own span, never across the host's interrupt block or a queue `sync`. What the teardown
//! releases is [`Mappings`], apart from it: the teardown unmaps on the releasing thread whatever
//! command the queue is running, while a command holds [`QueueState`] from its ring read to its
//! status store. [`Mappings`] is held only for one call into the memory map or one guest access
//! through the ring, the table or a descriptor page, and is taken after [`QueueState`], never
//! before. An access through a pointer into a mapping the teardown unmapped is the device's fault
//! at that access ([`Mappings::unmapped`]).
//!
//! What the device never releases stays leaked: the ring mapping a new ring size replaces, and,
//! when the state goes, every IOSurface still mapped.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

use std::num::NonZeroU64;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::guest::GuestPtr;
use crate::host::{HostPtr, RaiseInterrupt};
use crate::iosurface::DeathKind;
use crate::iosurface::descriptor::{Form, IoSurfaceHostDeviceDescriptor};
use crate::iosurface::regs::Registers;
use crate::iosurface::table::{MAPPING_ID_COUNT, PAGE_SLOTS, SurfaceSlot, TABLE_ENTRY_SIZE};
use crate::memory::map::MemoryMap;
use crate::memory::mapping::Mapping;
use crate::num::{ToU64, ToUsize};
use crate::platform::{Handle, Platform};
use crate::raise::{Kept, keep_forever};
use crate::rt::queue::{Executor, SerialQueue};
use crate::rt::signal::CoalescingSignal;

/// What the queue works on: the ring, the descriptor table and its cached pages, and the 8192
/// surface slots. The mappings behind the ring, the table and the pages are [`Mappings`].
pub(super) struct QueueState<P: Platform> {
    /// The descriptor table's entries: one page of 4-byte entries.
    pub(super) table_len: u64,
    /// Descriptors in one page, in the current form.
    pub(super) descriptors_per_page: u64,
    /// The mapped ring. Its length is never recorded, so the read counter is writable exactly
    /// while no ring is mapped.
    pub(super) ring: Option<GuestPtr>,
    /// Bytes of host-only surfaces made.
    pub(super) host_only_bytes: u64,
    /// One slot per mapping id.
    pub(super) slots: Vec<SurfaceSlot<P>>,
    /// The mapped descriptor table.
    pub(super) table: Option<GuestPtr>,
    /// Each cached descriptor page's host address; its mapping is in [`Mappings::pages`].
    pub(super) pages: Vec<Option<GuestPtr>>,
}

/// What the teardown releases, apart from [`QueueState`] so that the teardown never waits for a
/// command (see the module doc).
pub(super) struct Mappings<P: Platform> {
    /// The guest's memory; `None` after the teardown.
    pub(super) memory_map: Option<Arc<MemoryMap<P>>>,
    /// The ring's mapping.
    pub(super) ring: Option<Arc<Mapping<P>>>,
    /// The descriptor table's mapping.
    pub(super) table: Option<Arc<Mapping<P>>>,
    /// Each cached descriptor page's mapping.
    pub(super) pages: Vec<Option<Arc<Mapping<P>>>>,
    /// The host ranges the teardown unmapped while the ring, table and page pointers kept
    /// pointing into them. A load or store there faults: the fake keeps deallocated memory
    /// readable and the real platform faults at no access of the device's, so each access checks
    /// this first and dies as its own. Nothing is mapped after the teardown: the memory map is
    /// gone then, and every mapping comes from it.
    pub(super) unmapped: Vec<(usize, u64)>,
}

impl<P: Platform> Mappings<P> {
    /// A read-write mapping of `length` guest bytes at `address`; `None` when the map refuses it
    /// or is gone.
    pub(super) fn new_contiguous_mapping(
        &self,
        address: u64,
        length: u64,
    ) -> Option<Arc<Mapping<P>>> {
        self.memory_map
            .as_ref()
            .and_then(|m| m.new_contiguous_mapping(address, length, 0))
    }

    /// The teardown's release of one mapping. Its last release unmaps it: the range is recorded
    /// in [`Self::unmapped`] before it goes.
    pub(super) fn release(&mut self, mapping: Option<Arc<Mapping<P>>>) {
        let Some(mapping) = mapping.and_then(Arc::into_inner) else {
            return;
        };
        self.unmapped
            .push((mapping.virtual_address().addr(), mapping.length()));
        drop(mapping);
    }

    /// Whether `[at, at + len)` overlaps a range the teardown unmapped.
    pub(super) fn is_unmapped(&self, at: GuestPtr, len: u64) -> bool {
        let start = at.addr().addr().to_u64();
        let end = start.saturating_add(len);
        self.unmapped.iter().any(|&(base, length)| {
            let base = base.to_u64();
            base < end && start < base.saturating_add(length)
        })
    }
}

/// The host's IOSurface host device. Its last drop is the teardown, on the dropping thread at
/// once. The state is [`HostDeviceCore`], which the doorbell's handler and every queue block
/// hold too; neither keeps this object alive.
pub struct IoSurfaceHostDevice<P: Platform> {
    core: Arc<HostDeviceCore<P>>,
}

/// Why an IOSurface host device was refused, in the order the checks are made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IoSurfaceRefusal {
    /// The descriptor has no memory map.
    NoMemoryMap,
    /// The memory map refused its descriptor.
    MemoryMap,
}

/// The host's handle to an IOSurface host device: one counted reference to its
/// [`IoSurfaceHostDevice`]. There is no `Clone`: [`HostDeviceHandle::clone_ref`] takes another
/// reference, and the last one's drop is the device's teardown, on the releasing thread at
/// once.
pub struct HostDeviceHandle<P: Platform>(pub(super) Arc<IoSurfaceHostDevice<P>>);

impl<P: Platform> HostDeviceHandle<P> {
    /// One more reference.
    pub fn clone_ref(&self) -> HostDeviceHandle<P> {
        HostDeviceHandle(Arc::clone(&self.0))
    }

    /// Whether `self` and `other` are references to one device.
    pub fn ptr_eq(&self, other: &HostDeviceHandle<P>) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// The device's address, as its creation and teardown lines print it.
    pub fn as_ptr(&self) -> *const IoSurfaceHostDevice<P> {
        Arc::as_ptr(&self.0)
    }
}

impl<P: Platform> std::ops::Deref for HostDeviceHandle<P> {
    type Target = IoSurfaceHostDevice<P>;

    fn deref(&self) -> &IoSurfaceHostDevice<P> {
        &self.0
    }
}

impl<P: Platform> std::fmt::Debug for HostDeviceHandle<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.0, f)
    }
}

impl<P: Platform> IoSurfaceHostDevice<P> {
    /// The state, unretained: what the graphics device's mapper service holds, so its lookups
    /// never keep the host's device alive or run its teardown.
    pub fn core_weak(&self) -> Weak<HostDeviceCore<P>> {
        Arc::downgrade(&self.core)
    }
}

impl<P: Platform> std::ops::Deref for IoSurfaceHostDevice<P> {
    type Target = HostDeviceCore<P>;
    fn deref(&self) -> &HostDeviceCore<P> {
        &self.core
    }
}

impl<P: Platform> std::fmt::Debug for IoSurfaceHostDevice<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&*self.core, f)
    }
}

/// The interrupt block: held from creation, released by the teardown but still called after it.
enum Interrupt {
    /// No block was given.
    Nil,
    /// The device's reference.
    Held(RaiseInterrupt),
    /// After the teardown's release: the block is live only while the host keeps its own
    /// reference.
    Released(Weak<dyn Fn(u32) + Send + Sync>),
}

/// The device's state. The registers are seq_cst atomics; the rest is worked on by the device's
/// queue, and [`HostDeviceCore::retain_surface_with_mapping_id`] syncs onto it.
pub struct HostDeviceCore<P: Platform> {
    /// The MMIO window's length (unused but for `Debug`).
    mmio_length: u64,
    /// The interrupt block.
    interrupt: Mutex<Interrupt>,
    /// The device's serial queue (`reims-vgpu.iosurface`).
    pub(crate) hw_queue: SerialQueue,
    /// The doorbell: a coalescing signal whose handler runs the ring on the queue. Taken by the
    /// teardown.
    pub(super) doorbell: Mutex<Option<CoalescingSignal>>,
    /// The MMIO registers.
    pub(super) regs: Registers,
    /// What the queue works on.
    pub(super) state: Mutex<QueueState<P>>,
    /// What the teardown releases. Taken after `state`, never before.
    pub(super) maps: Mutex<Mappings<P>>,
    /// The platform (the VM page size, IOSurfaces, logs).
    pub(super) platform: Arc<P>,
    /// This state's own `Weak`, for the blocks the device queues: a block owns what it uses and
    /// holds the state, never the host's device.
    weak_core: Weak<HostDeviceCore<P>>,
}

impl<P: Platform> std::fmt::Debug for HostDeviceCore<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("IoSurfaceHostDevice");
        d.field("mmio_length", &self.mmio_length);
        self.regs.debug_fields(&mut d);
        d.finish_non_exhaustive()
    }
}

impl<P: Platform> IoSurfaceHostDevice<P> {
    /// The process's last device created, while it is alive.
    pub fn latest_handle(platform: &P) -> Option<HostDeviceHandle<P>> {
        Self::latest(platform).map(HostDeviceHandle)
    }

    /// The state of the process's last device, unretained, while the device is alive: for the
    /// graphics device's mapper service and the video devices' surfaces, on their threads. The
    /// lookup never retains the device, so the host's last release is always its teardown, on
    /// the host's thread, whatever lookups run beside it.
    pub fn latest_core(platform: &P) -> Option<Weak<HostDeviceCore<P>>> {
        platform.globals().latest_iosurface_core()
    }

    /// The process's last device created; `None` once its teardown has begun.
    fn latest(platform: &P) -> Option<Arc<Self>> {
        platform.globals().latest_iosurface_device()
    }

    /// A device over the descriptor's memory map; the reason for a refusal without one, or when
    /// the memory map refuses it. The descriptor's interrupt block is read once the memory map is
    /// made, before the queue and the doorbell; a refused device never reads it.
    ///
    /// The doorbell's handler holds the state, not the device, so the host's last release runs
    /// the teardown at once on the releasing thread, even from inside the handler's interrupt
    /// block, where the teardown's wait for the queue the thread is running is libdispatch's
    /// crash. A handler still pending or running when the teardown releases the mappings runs on
    /// what it left: its next access through them faults ([`Mappings::unmapped`]), except
    /// through a ring a new ring size leaked, which stays mapped and whose commands it reads and
    /// runs.
    #[expect(
        clippy::new_ret_no_self,
        reason = "the host's first reference is the handle"
    )]
    pub fn new(
        platform: &Arc<P>,
        descriptor: &IoSurfaceHostDeviceDescriptor,
    ) -> Result<HostDeviceHandle<P>, IoSurfaceRefusal> {
        let page = platform.vm().page_size();
        let table_len = page / const { NonZeroU64::new(TABLE_ENTRY_SIZE).unwrap() };
        let descriptors_per_page = page / Form::First.size();
        let mmio_length = descriptor.mmio_length;
        let Some(map_descriptor) = descriptor.memory_map_descriptor.as_ref() else {
            log::warn!("host device refused: no memory map descriptor");
            return Err(IoSurfaceRefusal::NoMemoryMap);
        };
        let Some(memory_map) = MemoryMap::new(platform.clone(), map_descriptor) else {
            log::warn!("host device refused: the memory map refused its descriptor");
            return Err(IoSurfaceRefusal::MemoryMap);
        };
        // The block is retained, not copied.
        let interrupt = match descriptor.raise_interrupt.clone() {
            Some(raise) => Interrupt::Held(raise),
            None => Interrupt::Nil,
        };
        let queue = SerialQueue::new(platform.threads(), "reims-vgpu.iosurface");
        let core = Arc::new_cyclic(|weak_core: &Weak<HostDeviceCore<P>>| HostDeviceCore {
            mmio_length,
            interrupt: Mutex::new(interrupt),
            hw_queue: queue.clone(),
            doorbell: Mutex::new(None),
            regs: Registers::new(),
            state: Mutex::new(QueueState {
                table_len,
                descriptors_per_page,
                ring: None,
                host_only_bytes: 0,
                slots: (0..MAPPING_ID_COUNT)
                    .map(|_| SurfaceSlot::empty())
                    .collect(),
                table: None,
                pages: (0..PAGE_SLOTS).map(|_| None).collect(),
            }),
            maps: Mutex::new(Mappings {
                memory_map: Some(memory_map),
                ring: None,
                table: None,
                pages: (0..PAGE_SLOTS).map(|_| None).collect(),
                unmapped: Vec::new(),
            }),
            platform: platform.clone(),
            weak_core: weak_core.clone(),
        });
        let target: Arc<dyn Executor> = Arc::new(queue);
        // The state, not the device: the handler never runs the teardown. It keeps the state
        // until the doorbell frees it after the teardown's release, which breaks the cycle.
        let handler_core = core.clone();
        let doorbell = CoalescingSignal::new_or(target, move |_| {
            handler_core.run_ring();
        });
        *core.doorbell.lock() = Some(doorbell);
        let device = Arc::new(IoSurfaceHostDevice { core });
        // The last device created wins, process-wide.
        platform
            .globals()
            .register_iosurface_device(Arc::downgrade(&device), device.core_weak());
        log::info!("host device {:#x} created", Arc::as_ptr(&device).addr());
        Ok(HostDeviceHandle(device))
    }
}

impl<P: Platform> HostDeviceCore<P> {
    /// This state, for a block the device queues: the state, never the host's device, so the
    /// host's last release during a block is the last one; the teardown runs on the releasing
    /// thread and waits for the block, and the block's caller returns normally. `None` never
    /// happens: the state is always in its `Arc`.
    pub(super) fn block_self(&self) -> Option<Arc<HostDeviceCore<P>>> {
        self.weak_core.upgrade()
    }

    /// One guest access through the ring, the table or a descriptor page, of `len` bytes at
    /// `at`: the fault `kind` when the teardown unmapped what it points into. Under `maps`, so the
    /// access is wholly before or after the teardown's releases.
    pub(super) fn mapped_access<T>(
        &self,
        at: GuestPtr,
        len: u64,
        kind: DeathKind,
        access: impl FnOnce() -> T,
    ) -> T {
        let maps = self.maps.lock();
        if maps.is_unmapped(at, len) {
            crate::fatal!(kind);
        }
        let v = access();
        drop(maps);
        v
    }

    /// The host address of `length` guest bytes at `address`; NULL when unmapped, or after the
    /// teardown.
    pub(super) fn virtual_address_for_physical(&self, address: u64, length: u64) -> HostPtr {
        self.maps
            .lock()
            .memory_map
            .as_ref()
            .map_or(HostPtr::NULL, |m| {
                m.virtual_address_for_physical(address, length)
            })
    }

    /// The VM page size.
    pub(super) fn page_size(&self) -> u64 {
        self.platform.vm().page_size()
    }

    /// The VM page shift.
    pub(super) fn page_shift(&self) -> u32 {
        self.platform.vm().page_shift()
    }

    /// Another reference to the IOSurface mapped at `mapping_id`, made on the device's queue;
    /// `None` for an id out of range or a slot with no surface.
    pub fn retain_surface_with_mapping_id(&self, mapping_id: u64) -> Option<P::Surface> {
        self.retain_mapped(mapping_id, false)
            .map(|(surface, _)| surface)
    }

    /// [`Self::retain_surface_with_mapping_id`], with the host address of each guest page the
    /// surface was made over, in order, each a VM page: none for a host-only surface. They stay
    /// its pages while the reference lives, as the guest's unmap of it is Busy.
    pub fn retain_surface_and_pages_with_mapping_id(
        &self,
        mapping_id: u64,
    ) -> Option<(P::Surface, Vec<u64>)> {
        self.retain_mapped(mapping_id, true)
    }

    /// The surface at `mapping_id`, retained on the device's queue, and its pages where `pages`.
    fn retain_mapped(&self, mapping_id: u64, pages: bool) -> Option<(P::Surface, Vec<u64>)> {
        log::debug!("retain surface {mapping_id}");
        let this = self.block_self()?;
        self.hw_queue.sync(move || {
            if mapping_id == 0 || mapping_id >= MAPPING_ID_COUNT {
                log::warn!(
                    "retain surface: mapping id {mapping_id} is not in 1..{MAPPING_ID_COUNT}"
                );
                return None;
            }
            let st = this.state.lock();
            let slot = st.slots.get(mapping_id.to_usize())?;
            let surface = slot.surface.as_ref().map(Handle::clone_ref)?;
            Some((
                surface,
                if pages {
                    slot.pages.clone()
                } else {
                    Vec::new()
                },
            ))
        })
    }

    /// Raises interrupt 0 through the block, which is called whether or not the device holds
    /// one: with none, or one the teardown released and the host freed, the call faults.
    pub(super) fn raise_interrupt(&self) {
        let raise = match &*self.interrupt.lock() {
            Interrupt::Held(raise) => Ok(raise.clone()),
            Interrupt::Released(weak) => weak.upgrade().ok_or(DeathKind::FreedInterrupt),
            Interrupt::Nil => Err(DeathKind::NoInterrupt),
        };
        match raise {
            Ok(raise) => raise(0),
            Err(kind) => crate::fatal!(kind),
        }
    }

    /// The teardown, for the host's last release. The mapped IOSurfaces are not released; the
    /// doorbell is released on the queue, not cancelled. Its only wait is the queue `sync`: a
    /// doorbell handler or a queue block may be running, even inside a command, and the mappings
    /// are released under it without taking [`QueueState`] (its next access through them
    /// faults). Run on the queue itself (the host released the device from inside the interrupt
    /// block), that `sync` is libdispatch's crash.
    fn tear_down(&self, this: usize) {
        log::info!("host device {this:#x} tearing down");
        {
            let mut maps = self.maps.lock();
            // The map is there until this release (the teardown runs once).
            if maps.memory_map.is_some() {
                log::debug!(
                    "host device {this:#x}: unmapping the ring, the descriptor table and its pages"
                );
                // The ring, table and page pointers stay set.
                let ring = maps.ring.take();
                maps.release(ring);
                let table = maps.table.take();
                maps.release(table);
                let pages: Vec<_> = maps.pages.iter_mut().map(Option::take).collect();
                for page in pages {
                    maps.release(page);
                }
            }
            maps.memory_map = None;
        }
        // Released, not cleared.
        let raise = {
            let mut slot = self.interrupt.lock();
            match std::mem::replace(&mut *slot, Interrupt::Nil) {
                Interrupt::Held(raise) => {
                    *slot = Interrupt::Released(Arc::downgrade(&raise));
                    Some(raise)
                }
                other => {
                    *slot = other;
                    None
                }
            }
        };
        // The release itself, outside the lock: the host's block may be freed with it.
        drop(raise);
        let doorbell = self.doorbell.lock().take();
        if let Some(doorbell) = doorbell {
            log::debug!("host device {this:#x}: releasing the doorbell");
            // On the queue itself (the host's last release inside a handler) this runs at once;
            // elsewhere it waits for a running handler or block.
            self.hw_queue.sync(move || {
                drop(doorbell);
            });
        }
        // The queue goes with the state.
        log::debug!("host device {this:#x} torn down");
    }
}

impl<P: Platform> Drop for IoSurfaceHostDevice<P> {
    /// The teardown, on the thread that drops the last reference, at once: a doorbell handler
    /// and a queue block hold only the state, never the device.
    fn drop(&mut self) {
        // The address the creation line printed.
        let this = std::ptr::from_ref(self).addr();
        self.core.tear_down(this);
    }
}

impl<P: Platform> Drop for HostDeviceCore<P> {
    /// The state going, after the teardown and after the last handler or block that held it. The
    /// IOSurfaces still mapped are kept for good: their use counts stay up.
    fn drop(&mut self) {
        for slot in &mut self.state.get_mut().slots {
            if let Some(surface) = slot.surface.take() {
                keep_forever(surface, Kept::SurfaceUseCount);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    use crate::iosurface::regs::{REG_RING_BASE, REG_RING_SIZE, REG_RING_WR_COUNTER};
    use crate::iosurface::ring::IoReturn;
    use crate::iosurface::testing::*;
    use crate::memory::descriptor::GuestPhysicalRange;
    use crate::platform::os::surface::Surface;
    use crate::testkit::capture::expect_death;
    use crate::testkit::logs;
    use crate::testkit::platform::FakePlatform;

    /// The `Debug` form shows the MMIO registers (diagnostics only).
    #[test]
    fn debug_shows_the_registers() {
        let r = rig();
        let s = format!("{:?}", r.dev);
        assert!(s.starts_with("IoSurfaceHostDevice { mmio_length: "), "{s}");
        for f in [
            "reg_ring_base",
            "reg_ring_size",
            "reg_descriptor_base",
            "reg_ring_wr_counter",
            "reg_ring_rd_counter",
            "reg_desc_enable",
        ] {
            assert!(s.contains(f), "{s}");
        }
        assert!(s.ends_with(", .. }"), "{s}");
    }

    /// A memory map descriptor the map refuses: two ranges over the same guest page.
    fn intersecting(r: &Rig) -> IoSurfaceHostDeviceDescriptor {
        let mut bad = map_descriptor(&r.ram);
        let range = r.ram.ranges()[0].clone();
        bad.add_range(GuestPhysicalRange {
            physical_address: B + PAGE,
            physical_length: PAGE,
            host_address: range.host(),
        });
        let mut refused = IoSurfaceHostDeviceDescriptor::new();
        refused.memory_map_descriptor = Some(bad);
        refused
    }

    mod creation {
        use super::*;

        /// No memory map descriptor: nil, and no queue.
        #[test]
        fn refuses_without_memory_map_descriptor() {
            let (p, _ram) = platform_and_ram();
            assert_eq!(
                Dev::new(&p, &IoSurfaceHostDeviceDescriptor::new()).err(),
                Some(IoSurfaceRefusal::NoMemoryMap)
            );
            assert!(p.fake_threads().spawned().is_empty());
            assert!(Dev::latest_handle(&*p).is_none());
        }

        /// A refused device writes one warning, which says why, and no teardown: nothing was
        /// made. Both refusals.
        #[test]
        fn a_refused_device_logs_why() {
            let r = rig();
            let logs = logs::capture();
            for (d, why) in [
                (
                    IoSurfaceHostDeviceDescriptor::new(),
                    "host device refused: no memory map descriptor",
                ),
                (
                    intersecting(&r),
                    "host device refused: the memory map refused its descriptor",
                ),
            ] {
                assert!(Dev::new(&r.p, &d).is_err());
                assert_eq!(
                    lines_from(&logs, module_path!()),
                    [(log::Level::Warn, why.to_owned())]
                );
            }
            // The live device is untouched: still the latest.
            assert!(Dev::latest_handle(&*r.p).unwrap().ptr_eq(&r.dev));
        }

        /// A memory map the map refuses (intersecting ranges): nil, no queue.
        #[test]
        fn refuses_a_bad_memory_map() {
            let (p, ram) = platform_and_ram();
            let mut d = IoSurfaceHostDeviceDescriptor::new();
            let mut m = map_descriptor(&ram);
            let r = ram.ranges()[0].clone();
            m.add_range(GuestPhysicalRange {
                physical_address: B + PAGE,
                physical_length: PAGE,
                host_address: r.host(),
            });
            d.memory_map_descriptor = Some(m);
            assert_eq!(Dev::new(&p, &d).err(), Some(IoSurfaceRefusal::MemoryMap));
            assert!(p.fake_threads().spawned().is_empty());
        }

        /// The fields, the queue and the latest device; the last device created wins, and the
        /// process does not keep it alive.
        #[test]
        fn defaults_and_the_last_device_wins() {
            let r = rig();
            assert_eq!(r.dev.mmio_length, 16384);
            {
                let st = r.dev.state.lock();
                assert_eq!(st.table_len, PAGE / 4);
                assert_eq!(st.descriptors_per_page, PAGE / 0x200);
                assert!(r.dev.maps.lock().memory_map.is_some() && st.ring.is_none());
                assert_eq!((st.slots.len(), st.pages.len()), (8192, 4096));
            }
            for off in [0x1000, 0x1008, 0x1010, 0x1018, 0x1020, 0x1030] {
                assert_eq!(r.dev.mmio_read(off), 0);
            }
            // A dispatch queue, which starts no thread of its own.
            assert!(
                r.p.fake_threads()
                    .queues()
                    .contains(&"reims-vgpu.iosurface")
            );
            assert!(
                !r.p.fake_threads()
                    .spawned()
                    .iter()
                    .any(|s| s.0 == "reims-vgpu.iosurface")
            );
            assert!(Dev::latest_handle(&*r.p).unwrap().ptr_eq(&r.dev));
            let mut d = IoSurfaceHostDeviceDescriptor::new();
            d.memory_map_descriptor = Some(map_descriptor(&r.ram));
            let second = Dev::new(&r.p, &d).unwrap();
            assert!(Dev::latest_handle(&*r.p).unwrap().ptr_eq(&second));
            drop(second);
            assert!(Dev::latest_handle(&*r.p).is_none());
        }

        /// A lookup of the latest device's state never retains the device: the host's last
        /// release is its teardown, on the releasing thread, while another thread looks it up
        /// over and over, as the video devices' session threads and the mapper service may.
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: the looking thread
        fn a_lookup_never_takes_the_teardown_from_the_release() {
            for _ in 0..50 {
                let r = rig();
                r.mapped_v2(1, 1);
                let Rig { p, dev, .. } = r;
                let looking = Arc::new(AtomicBool::new(true));
                let started = Arc::new(AtomicBool::new(false));
                let looker = {
                    let (p, looking, started) = (p.clone(), looking.clone(), started.clone());
                    std::thread::Builder::new()
                        .name("lookup".to_owned())
                        .spawn(move || {
                            while looking.load(Ordering::Acquire) {
                                let _ = Dev::latest_core(&*p);
                                started.store(true, Ordering::Release);
                            }
                        })
                        .unwrap()
                };
                while !started.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                let mark = p.call_log().mark();
                drop(dev);
                looking.store(false, Ordering::Release);
                looker.join().unwrap();
                let unmapped: Vec<Option<String>> = (p.call_log().since(mark).into_iter())
                    .filter(|call| call.sel == "mach_vm_deallocate")
                    .map(|call| call.thread)
                    .collect();
                assert_eq!(unmapped.len(), 3, "{unmapped:?}");
                assert!(
                    unmapped
                        .iter()
                        .all(|thread| thread.as_deref() != Some("lookup")),
                    "the lookup ran the teardown: {unmapped:?}"
                );
                assert!(Dev::latest_core(&*p).is_none());
            }
        }
    }

    /// Out of range and empty slots answer nil; a mapped surface is one more reference, which
    /// makes its unmap Busy.
    #[test]
    fn retain_surface_adds_a_reference() {
        let r = rig();
        assert!(r.dev.retain_surface_with_mapping_id(0).is_none());
        assert!(r.dev.retain_surface_with_mapping_id(8192).is_none());
        assert!(r.dev.retain_surface_with_mapping_id(1).is_none());
        r.mapped_v2(1, 1);
        let s = r.dev.retain_surface_with_mapping_id(1).unwrap();
        assert_eq!(Some(s.id()), r.surface(1));
        assert_eq!(Surface::retain_count(&s), 2);
        assert_eq!(r.unmap(1), IoReturn::Busy);
    }

    /// A surface retained with its pages comes with the host address of each guest page it was
    /// made over, in order, as its slot holds them; a host-only one with none
    #[test]
    fn a_surface_is_retained_with_its_pages() {
        let r = rig();
        assert!(r.dev.retain_surface_and_pages_with_mapping_id(1).is_none());
        r.mapped_v2(1, 2);
        let want = r.dev.state.lock().slots[1].pages.clone();
        assert_eq!(want.len(), 2);
        let (s, pages) = r.dev.retain_surface_and_pages_with_mapping_id(1).unwrap();
        assert_eq!(Some(s.id()), r.surface(1));
        assert_eq!(pages, want);
        assert_eq!(Surface::retain_count(&s), 2);
    }

    /// A surface a host holds keeps its pages while it holds it: the guest's unmap of it is
    /// Busy, and a map of its mapping ID changes nothing, its descriptor not read again though the
    /// guest wrote another, so its memory, which a device may be writing a frame into, is the
    /// guest's pages it was made over until the hold goes.
    #[test]
    fn a_held_surface_keeps_its_pages() {
        let r = rig();
        r.mapped_v2(1, 2);
        let pages = r.dev.state.lock().slots[1].pages.clone();
        let held = r.dev.retain_surface_with_mapping_id(1).unwrap();
        assert_eq!(r.unmap(1), IoReturn::Busy);
        // Another descriptor at the same mapping ID, of one page: not read while it is mapped.
        r.put_v2(1, &v2(1, L2A));
        assert_eq!(r.map(1), IoReturn::Success);
        assert_eq!(r.created().len(), 1);
        assert_eq!(r.surface(1), Some(held.id()));
        assert_eq!(r.dev.state.lock().slots[1].pages, pages);
        assert_eq!(r.unmap(1), IoReturn::Busy);
        drop(held);
        assert_eq!(r.unmap(1), IoReturn::Success);
        assert!(r.surface(1).is_none());
    }

    /// The teardown unmaps the ring, the table and the descriptor page, and leaks the surface.
    #[test]
    fn teardown_unmaps_and_leaks_the_surfaces() {
        let r = rig();
        r.mapped_v2(1, 1);
        let held = r.held(1);
        assert_eq!(Surface::retain_count(&held), 2);
        let mark = r.p.call_log().mark();
        let Rig { p, dev, .. } = r;
        drop(dev);
        let sels = p.call_log().sels_since(mark);
        assert_eq!(
            sels.iter().filter(|s| **s == "mach_vm_deallocate").count(),
            3
        );
        assert!(!sels.contains(&"CFRelease"));
        assert_eq!(Surface::retain_count(&held), 2);
        assert!(Dev::latest_handle(&*p).is_none());
    }

    /// The creation and every teardown line name the device by the address its handle has.
    #[test]
    fn lifecycle_lines_name_the_device() {
        use log::Level::{Debug, Info};
        let logs = logs::capture();
        let r = rig();
        let ptr = format!("{:p}", r.dev.as_ptr());
        let Rig { dev, .. } = r;
        drop(dev);
        assert_eq!(
            lines_from(&logs, module_path!()),
            [
                (Info, format!("host device {ptr} created")),
                (Info, format!("host device {ptr} tearing down")),
                (
                    Debug,
                    format!(
                        "host device {ptr}: unmapping the ring, the descriptor table and its pages"
                    )
                ),
                (Debug, format!("host device {ptr}: releasing the doorbell")),
                (Debug, format!("host device {ptr} torn down")),
            ]
        );
    }

    /// A doorbell still owed when the last reference goes: the teardown releases the ring, then
    /// its wait for the queue sits behind the handler, which runs the ring over the unmapped
    /// ring.
    #[test]
    fn teardown_with_a_doorbell_owed_dies() {
        expect_death(DeathKind::RingRead.site(), || {
            let r = rig();
            r.setup(true);
            r.command(0, 9, 1);
            r.sync();
            let mark = r.p.call_log().mark();
            let log = r.p.clone();
            // Hold the queue until the teardown has released the ring mapping.
            r.dev.hw_queue.exec_async(Box::new(move || {
                while !log
                    .call_log()
                    .sels_since(mark)
                    .contains(&"mach_vm_deallocate")
                {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }));
            r.dev.mmio_write(REG_RING_WR_COUNTER, 1);
            let Rig { dev, .. } = r;
            drop(dev);
        });
    }

    /// The same drop with nothing owed: the pending handler finds the counters level, and the
    /// teardown completes.
    #[test]
    fn teardown_with_a_doorbell_pending_and_nothing_owed() {
        let r = rig();
        r.setup(true);
        r.command(0, 9, 1);
        r.sync();
        let mark = r.p.call_log().mark();
        let log = r.p.clone();
        // Hold the queue until the teardown has released the ring mapping.
        r.dev.hw_queue.exec_async(Box::new(move || {
            while !log
                .call_log()
                .sels_since(mark)
                .contains(&"mach_vm_deallocate")
            {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }));
        // A doorbell with rd == wr: merged and pending, nothing owed.
        r.dev.mmio_write(REG_RING_WR_COUNTER, 0);
        let Rig { p, ram, dev, .. } = r;
        drop(dev);
        p.fake_vm().sync();
        assert_eq!(ram.read_u32(RING + 8).unwrap(), 0x5555);
        assert!(Dev::latest_handle(&*p).is_none());
    }

    /// The host's interrupt block releases its last reference to the device. The handler holds
    /// only the state, so the teardown runs there, on the queue, inside the block: it logs,
    /// releases the mappings, and its sync onto the queue the thread already holds runs at once,
    /// so the teardown completes there. The handler then reads the ring's next command through
    /// the released mapping, and dies there; the second command never runs.
    #[test]
    fn teardown_inside_the_interrupt_block_completes_then_the_next_ring_read_dies() {
        let held: Arc<Mutex<Option<HostDeviceHandle<FakePlatform>>>> = Arc::new(Mutex::new(None));
        let h2 = held.clone();
        let r = rig_raising(move |_| {
            let last = h2.lock().take();
            drop(last);
        });
        r.setup(true);
        r.command(0, 9, 1);
        r.command(1, 9, 2);
        r.sync();
        let Rig { p, ram, dev, .. } = r;
        *held.lock() = Some(dev.clone_ref());
        let core = dev.core.clone();
        let queue = dev.hw_queue.clone();
        expect_death(DeathKind::RingRead.site(), move || {
            // Hold the queue until the host's is the only reference left.
            let go = Arc::new(AtomicBool::new(false));
            let g2 = go.clone();
            queue.exec_async(move || {
                while !g2.load(Ordering::SeqCst) {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            });
            dev.mmio_write(REG_RING_WR_COUNTER, 2);
            drop(dev);
            go.store(true, Ordering::SeqCst);
            // The dead worker keeps the queue: this never returns.
            queue.sync(|| ());
        });
        // The teardown released the mappings and took the doorbell before its wait.
        assert!(core.maps.lock().memory_map.is_none());
        assert!(core.doorbell.lock().is_none());
        p.fake_vm().sync();
        // The first command's status was written; the second never ran.
        assert_eq!(
            ram.read_u32(RING + 8).unwrap() as i32,
            IoReturn::Unsupported.code()
        );
        assert_eq!(ram.read_u32(RING + 16 + 8).unwrap(), 0x5555);
    }

    /// The last reference goes on another thread while the handler runs: the teardown runs on
    /// that thread at once (the handler holds no reference), releasing the mappings under the
    /// handler, then its wait for the queue waits the handler out, and only then does the release
    /// return.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: the host's releasing thread
    fn teardown_runs_on_the_releasing_thread_during_a_doorbell() {
        let gate = Arc::new(Gate::default());
        let g2 = gate.clone();
        let r = rig_raising(move |_| g2.enter_and_wait());
        r.setup(true);
        r.command(0, 9, 1);
        r.sync();
        let Rig { p, ram, dev, .. } = r;
        let core = Arc::downgrade(&dev.core);
        dev.mmio_write(REG_RING_WR_COUNTER, 1);
        gate.wait_entered();
        let mark = p.call_log().mark();
        let t = std::thread::spawn(move || drop(dev));
        wait_doorbell_taken(&core);
        // The ring and the table are unmapped while the handler is still in the host's block.
        assert!(
            p.call_log()
                .sels_since(mark)
                .contains(&"mach_vm_deallocate")
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!t.is_finished(), "the queue sync waits for the handler");
        gate.open();
        t.join().unwrap();
        p.fake_vm().sync();
        assert_eq!(
            ram.read_u32(RING + 8).unwrap() as i32,
            IoReturn::Unsupported.code()
        );
        assert!(Dev::latest_handle(&*p).is_none());
    }

    /// The same release with a second command owed: after the host's block returns, the
    /// handler's next ring read is through the ring the teardown unmapped: a fault.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: the host's releasing thread
    fn teardown_during_a_doorbell_faults_the_next_ring_read() {
        let gate = Arc::new(Gate::default());
        let g2 = gate.clone();
        let r = rig_raising(move |_| g2.enter_and_wait());
        r.setup(true);
        r.command(0, 9, 1);
        r.command(1, 9, 2);
        r.sync();
        let Rig { dev, .. } = r;
        let core = Arc::downgrade(&dev.core);
        expect_death(DeathKind::RingRead.site(), move || {
            dev.mmio_write(REG_RING_WR_COUNTER, 2);
            gate.wait_entered();
            let t = std::thread::spawn(move || drop(dev));
            wait_doorbell_taken(&core);
            gate.open();
            // The dead handler keeps the queue: the release never returns.
            let _ = t.join();
        });
    }

    /// The teardown's release unmaps a mapping only as its last reference, and what it unmapped
    /// is what a later access through a stale pointer faults on: the whole mapping, nothing past
    /// it. A missing mapping is nothing.
    #[test]
    fn release_records_what_it_unmapped() {
        let r = rig();
        r.setup(true);
        let mut maps = r.dev.maps.lock();
        let ring = maps.ring.take().unwrap();
        let base = GuestPtr::new(ring.virtual_address());
        let other = ring.clone();
        let mark = r.p.call_log().mark();
        maps.release(Some(ring));
        maps.release(None);
        assert!(maps.unmapped.is_empty());
        assert_eq!(deallocations(&r.p, mark), 0);
        maps.release(Some(other));
        assert_eq!(deallocations(&r.p, mark), 1);
        assert_eq!(maps.unmapped, vec![(base.addr().addr(), 1024 * 16)]);
        assert!(maps.is_unmapped(base, 4));
        assert!(maps.is_unmapped(base.offset(1024 * 16 - 4), 8));
        assert!(!maps.is_unmapped(base.offset(1024 * 16), 4));
        assert!(!maps.is_unmapped(GuestPtr::new(HostPtr::NULL), 4));
    }

    /// The last reference goes on another thread while the handler is inside a command (a test,
    /// waiting in `IOSurfaceLock`): the teardown does not wait for the command. It logs and
    /// unmaps the ring, the table and the descriptor page under it and reaches its queue `sync`
    /// at once. The command then finishes, and its status store is through the unmapped ring: a
    /// fault.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: the host's releasing thread
    fn teardown_during_a_command_faults_its_status_store() {
        let r = rig();
        r.mapped_v2(1, 1);
        r.command(0, 3, 1);
        r.sync();
        let surface = r.held(1);
        let Rig { p, dev, .. } = r;
        let core = Arc::downgrade(&dev.core);
        expect_death(DeathKind::StatusStore.site(), move || {
            // The IOSurface's own lock: the handler waits in `IOSurfaceLock`.
            let held = surface.state();
            let mark = p.call_log().mark();
            dev.mmio_write(REG_RING_WR_COUNTER, 1);
            wait_called(&p, mark, "IOSurfaceLock");
            let t = std::thread::spawn(move || drop(dev));
            wait_doorbell_taken(&core);
            assert_eq!(deallocations(&p, mark), 3);
            drop(held);
            // The dead handler keeps the queue: the release never returns.
            let _ = t.join();
        });
    }

    /// A rig whose ring a new size leaked: surface 1 mapped (ring of 1024 at `RING`, table,
    /// descriptor page 0), then a resize to 0x100000 commands whose 16 MiB lie past guest RAM, so
    /// the map refuses it: the ring mapping is gone and the ring pointer the old ring, which is
    /// never released. The interrupt block waits on `gate`; the returned block is the host's own
    /// reference to it.
    fn rig_with_leaked_ring(gate: &Arc<Gate>) -> (Rig, RaiseInterrupt) {
        let g2 = gate.clone();
        let raise: RaiseInterrupt = Arc::new(move |_| g2.enter_and_wait());
        let (p, ram) = platform_and_ram();
        let mut d = IoSurfaceHostDeviceDescriptor::new();
        d.raise_interrupt = Some(raise.clone());
        d.memory_map_descriptor = Some(map_descriptor(&ram));
        let dev = Dev::new(&p, &d).unwrap();
        let r = Rig {
            p,
            ram,
            dev,
            irqs: Arc::new(Mutex::new(Vec::new())),
        };
        r.mapped_v2(1, 1);
        let old = r.dev.state.lock().ring;
        r.dev.mmio_write(REG_RING_SIZE, 0x10_0000);
        assert_eq!(r.dev.mmio_read(REG_RING_SIZE), 0x10_0000);
        assert!(old.is_some());
        assert_eq!(r.dev.state.lock().ring, old);
        assert!(r.dev.maps.lock().ring.is_none());
        (r, raise)
    }

    /// After a refused resize, the teardown has no ring mapping to release: the ring the pointer
    /// uses stays mapped, and only the table and the descriptor page are unmapped. A handler still
    /// running (here in the first command's interrupt when the last reference goes) reads its next
    /// commands from that ring and runs them: an unsupported command and a map of a mapped surface
    /// complete, each status is written back and each raises the host's block, which the host
    /// still holds. Then the teardown completes.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: the host's releasing thread
    fn teardown_leaves_a_leaked_ring_mapped() {
        let gate = Arc::new(Gate::default());
        let (r, raise) = rig_with_leaked_ring(&gate);
        r.command(0, 9, 1);
        r.command(1, 9, 2);
        r.command(2, 1, 1);
        r.sync();
        let Rig { p, ram, dev, .. } = r;
        let core = Arc::downgrade(&dev.core);
        let mark = p.call_log().mark();
        dev.mmio_write(REG_RING_WR_COUNTER, 3);
        gate.wait_entered();
        let t = std::thread::spawn(move || drop(dev));
        wait_doorbell_taken(&core);
        // The table and the descriptor page; not the ring.
        assert_eq!(deallocations(&p, mark), 2);
        gate.open();
        t.join().unwrap();
        assert_eq!(*gate.entered.lock(), 3);
        p.fake_vm().sync();
        assert_eq!(
            ram.read_u32(RING + 8).unwrap() as i32,
            IoReturn::Unsupported.code()
        );
        assert_eq!(
            ram.read_u32(RING + 16 + 8).unwrap() as i32,
            IoReturn::Unsupported.code()
        );
        assert_eq!(
            ram.read_u32(RING + 32 + 8).unwrap() as i32,
            IoReturn::Success.code()
        );
        drop(raise);
    }

    /// The same leaked ring, with an unmap of the mapped surface next: the handler reads it from
    /// the ring, and the unmap's mapped-bit store is through the descriptor page the teardown
    /// unmapped (the page pointer stays set): a fault.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: the host's releasing thread
    fn teardown_leaves_a_leaked_ring_but_not_the_descriptor_page() {
        let gate = Arc::new(Gate::default());
        let (r, raise) = rig_with_leaked_ring(&gate);
        r.command(0, 9, 1);
        r.command(1, 2, 1);
        r.sync();
        let Rig { dev, .. } = r;
        let core = Arc::downgrade(&dev.core);
        expect_death(DeathKind::MappedBitClear.site(), move || {
            dev.mmio_write(REG_RING_WR_COUNTER, 2);
            gate.wait_entered();
            let t = std::thread::spawn(move || drop(dev));
            wait_doorbell_taken(&core);
            gate.open();
            // The dead handler keeps the queue: the release never returns.
            let _ = t.join();
            drop(raise);
        });
    }

    /// Surface 1 mapped (ring, table and descriptor page 0), then the host's last release: the
    /// state as a handler still running finds it.
    fn state_after_teardown() -> (
        Arc<HostDeviceCore<FakePlatform>>,
        Arc<FakePlatform>,
        crate::testkit::guest_ram::GuestRam,
    ) {
        let r = rig();
        r.mapped_v2(1, 1);
        let core = r.dev.core.clone();
        let Rig { p, ram, dev, .. } = r;
        drop(dev);
        (core, p, ram)
    }

    /// What a command running after the teardown touches: a descriptor copy through the cached
    /// page, a table read for a page not cached and an unmap's mapped-bit store are through what
    /// the teardown unmapped, and fault; a test touches only the IOSurface, which the teardown
    /// leaks, and succeeds.
    #[test]
    fn commands_after_teardown_fault_where_they_reach_an_unmapped_page() {
        expect_death(DeathKind::DescriptorRead.site(), || {
            let (core, _p, _ram) = state_after_teardown();
            let _ = core.map(&mut core.state.lock(), 2);
        });
        // v2 has 16 descriptors a page: id 16 is on page 1, never cached.
        expect_death(DeathKind::TableRead.site(), || {
            let (core, _p, _ram) = state_after_teardown();
            let _ = core.map(&mut core.state.lock(), 16);
        });
        expect_death(DeathKind::MappedBitClear.site(), || {
            let (core, _p, _ram) = state_after_teardown();
            let _ = core.unmap(&mut core.state.lock(), 1);
        });
        let (core, _p, _ram) = state_after_teardown();
        assert!(core.maps.lock().memory_map.is_none());
        assert_eq!(core.test_pattern(&core.state.lock(), 1), IoReturn::Success);
        // A mapped surface is still Success without a read.
        assert_eq!(core.map(&mut core.state.lock(), 1), IoReturn::Success);
    }

    /// A ring base write with no ring mapped, racing the host's last release on another thread:
    /// the block keeps no reference, so the teardown runs on the releasing thread and completes,
    /// and the write returns normally, having stored the base.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: the host's threads
    fn teardown_during_an_mmio_write_block() {
        let r = rig();
        let core = r.dev.core.clone();
        let Rig { dev, .. } = r;
        let held = core.state.lock();
        let write = core.clone();
        let a = std::thread::spawn(move || write.mmio_write(REG_RING_BASE, RING));
        let b = std::thread::spawn(move || drop(dev));
        wait_doorbell_taken(&Arc::downgrade(&core));
        drop(held);
        a.join().unwrap();
        b.join().unwrap();
        assert!(core.maps.lock().memory_map.is_none());
        assert_eq!(core.mmio_read(REG_RING_BASE), RING);
        // Whatever the doorbell's release queued has run.
        core.hw_queue.sync(|| ());
        assert_eq!(Arc::strong_count(&core), 1);
    }
}
