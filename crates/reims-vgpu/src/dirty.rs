//! A host's dirty source ([`DirtySource`]), and the device's harvest of it.
//!
//! On a discrete GPU, and on an iGPU whose driver cannot reference the guest's memory, the
//! Vulkan runtime keeps copies of the guest's pages, and copies again only the pages core's
//! [`DirtyLog`] shows written since its last copy. The guest's own writes reach the log here:
//! QEMU under KVM gives the device its VM's file descriptor and a [`DirtyHost`], and the device
//! reads KVM's dirty log itself, on a thread of its own (`reims-vgpu.dirty`), each time the log
//! wants a harvest. A harvest, in this order:
//!
//! 1. [`DirtyLog::begin`] numbers it and gives the watched runs;
//! 2. where the CPU buffers a vCPU's writes until the vCPU exits (Intel's page-modification
//!    log, `pml` of `kvm_intel` or `kvm_amd`), [`DirtyHost::flush_vcpus`] makes each vCPU in
//!    `KVM_RUN` leave it once: `KVM_GET_DIRTY_LOG` kicks them, and does not wait;
//! 3. for each memory slot that logs ([`DirtyHost::for_each_slot`], under the host's lock on its
//!    slots) that holds a watched page: one `KVM_GET_DIRTY_LOG`, which under KVM's manual
//!    dirty-log protection clears nothing, then one `KVM_CLEAR_DIRTY_LOG` of exactly the watched
//!    pages it shows written, which KVM then protects again; and the host's own writes into those
//!    pages ([`DirtyHost::take_host_writes`], QEMU's DMA);
//! 4. [`DirtyLog::end`] publishes the pages written.
//!
//! A write between the read and the clear of a page the clear names is in memory before the
//! publish, so before any copy the publish allows; one to a page it does not name keeps its
//! bit for the next harvest. What the harvest cannot answer reads as written: a page no logged
//! slot of address space 0 holds (the log before its start among them), a slot whose read KVM
//! refused, a host that reads the log itself ([`DirtyHost::for_each_slot`] false), a flush past
//! its time, and the host's writes where a granule's place in the host's record does not keep
//! its guest address's offset in a [`GRANULE`](reims_vgpu_core::memory::dirty::GRANULE). Each costs a copy, never a missed write.
//!
//! The log is started ([`DirtyHost::start`]) at the first harvest, not with the device: KVM maps
//! a logged slot's pages 4 KiB at a time, so a device whose runtime never asks costs the guest
//! no huge pages.

#[cfg(any(unix, test))]
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(unix, test))]
use std::thread;
use std::thread::{JoinHandle, Thread};
use std::time::Duration;

use reims_vgpu_core::memory::dirty::DirtyLog;
#[cfg(any(unix, test))]
use reims_vgpu_core::memory::dirty::{DirtyStamp, GRANULE, PAGE_SIZE};

/// How long a flush of the vCPUs may take: the log's own wait, past which a harvest no longer
/// spares the copy it was wanted for.
#[cfg(any(unix, test))]
const FLUSH_TIMEOUT: Duration = Duration::from_millis(4);

/// One memory slot of the host's KVM VM that logs dirty pages ([`DirtyHost::for_each_slot`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvmSlot {
    /// As `KVM_GET_DIRTY_LOG` and `KVM_CLEAR_DIRTY_LOG` take it: the address space's ID << 16 |
    /// the slot's ID.
    pub slot: u32,
    /// The guest physical address of its first page.
    pub guest_physical_address: u64,
    /// Its length in bytes, a multiple of 4 KiB.
    pub length: u64,
    /// Where its first page lies in the host's own record of its writes
    /// ([`DirtyHost::take_host_writes`]): QEMU's `ram_addr_t`.
    pub host_offset: u64,
}

/// What the host does for a harvest. Each method is called on the device's harvest thread, one
/// at a time, and must not wait for the library or take a lock the host holds while it calls the
/// library (QEMU's BQL).
pub trait DirtyHost: Send + Sync {
    /// Starts logging the guest's RAM's dirty pages, with KVM's manual dirty-log protection and
    /// every page logged as written at its start (`KVM_DIRTY_LOG_INITIALLY_SET`). Called once, at
    /// the first harvest. It must not block: the log may start later, and until it does
    /// [`DirtyHost::for_each_slot`] lists none of the RAM's slots.
    fn start(&self);

    /// Waits until each vCPU that was inside `KVM_RUN` has returned from it once, so that KVM has
    /// drained the CPU's buffer of its writes into the slots' logs, `timeout` at most. False at
    /// the timeout. Called only where the CPU buffers them.
    fn flush_vcpus(&self, timeout: Duration) -> bool;

    /// Calls `slot` for each memory slot that logs dirty pages, holding the host's lock on its
    /// slots, so none changes before it returns. False, calling none, where the host reads and
    /// clears the log itself (QEMU's global dirty tracking).
    fn for_each_slot(&self, slot: &mut dyn FnMut(KvmSlot)) -> bool;

    /// ORs into `words`, one bit a 4 KiB page from `host_offset`, the pages of `length` bytes
    /// the host itself wrote since its last take, and clears them in its record. Called only
    /// inside [`DirtyHost::for_each_slot`]'s `slot`, with `host_offset` and `length` multiples
    /// of [`GRANULE`](reims_vgpu_core::memory::dirty::GRANULE).
    fn take_host_writes(&self, host_offset: u64, length: u64, words: &mut [u64]);
}

/// A host's dirty source ([`crate::DeviceConfig::dirty_source`]).
#[derive(Clone)]
pub struct DirtySource {
    /// The KVM VM's file descriptor, open while the device lives.
    pub kvm_vm_fd: i32,
    /// What the host does for a harvest.
    pub host: Arc<dyn DirtyHost>,
}

impl std::fmt::Debug for DirtySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirtySource")
            .field("kvm_vm_fd", &self.kvm_vm_fd)
            .finish_non_exhaustive()
    }
}

/// KVM's two dirty-log ioctls on one slot (the VM's in the product, a fake in tests).
#[cfg(any(unix, test))]
pub(crate) trait KvmLog: Send {
    /// `KVM_GET_DIRTY_LOG` into `bitmap`, which holds a bit for each of the slot's pages.
    fn get(&self, slot: u32, bitmap: &mut [u64]) -> bool;
    /// `KVM_CLEAR_DIRTY_LOG` of the pages `bitmap` sets, from `first_page` (a multiple of 64)
    /// for `num_pages`.
    fn clear(&self, slot: u32, first_page: u64, num_pages: u32, bitmap: &[u64]) -> bool;
}

/// The VM's file descriptor, borrowed from the host.
#[cfg(target_os = "linux")]
struct VmFd(i32);

/// `struct kvm_dirty_log` (linux/kvm.h).
#[repr(C)]
#[cfg(target_os = "linux")]
struct KvmDirtyLog {
    slot: u32,
    padding1: u32,
    dirty_bitmap: *mut u64,
}

/// `struct kvm_clear_dirty_log` (linux/kvm.h).
#[repr(C)]
#[cfg(target_os = "linux")]
struct KvmClearDirtyLog {
    slot: u32,
    num_pages: u32,
    first_page: u64,
    dirty_bitmap: *const u64,
}

/// `_IOW(KVMIO, 0x42, struct kvm_dirty_log)`.
#[cfg(target_os = "linux")]
const KVM_GET_DIRTY_LOG: u64 = 0x4010_ae42;
/// `_IOWR(KVMIO, 0xc0, struct kvm_clear_dirty_log)`.
#[cfg(target_os = "linux")]
const KVM_CLEAR_DIRTY_LOG: u64 = 0xc018_aec0;

#[cfg(target_os = "linux")]
const _: () = assert!(size_of::<KvmDirtyLog>() == 16 && size_of::<KvmClearDirtyLog>() == 24);

#[cfg(target_os = "linux")]
impl KvmLog for VmFd {
    fn get(&self, slot: u32, bitmap: &mut [u64]) -> bool {
        let mut log = KvmDirtyLog {
            slot,
            padding1: 0,
            dirty_bitmap: bitmap.as_mut_ptr(),
        };
        // SAFETY: KVM writes the slot's bitmap, a bit a page rounded up to a 64-bit word, which
        // `bitmap` holds: the host's lock on its slots keeps the slot as listed.
        unsafe { libc::ioctl(self.0, KVM_GET_DIRTY_LOG as _, &raw mut log) == 0 }
    }

    fn clear(&self, slot: u32, first_page: u64, num_pages: u32, bitmap: &[u64]) -> bool {
        let mut log = KvmClearDirtyLog {
            slot,
            num_pages,
            first_page,
            dirty_bitmap: bitmap.as_ptr(),
        };
        // SAFETY: KVM reads `num_pages` bits of `bitmap`, which holds them.
        unsafe { libc::ioctl(self.0, KVM_CLEAR_DIRTY_LOG as _, &raw mut log) == 0 }
    }
}

/// Whether the CPU buffers a vCPU's dirty pages until the vCPU exits: `pml` of `kvm_intel`, or
/// of `kvm_amd` where KVM has it. Without it KVM write-protects a logged page, and marks it in the
/// slot's log at the fault, before the write completes.
#[cfg(target_os = "linux")]
fn cpu_buffers_writes() -> bool {
    [
        "/sys/module/kvm_intel/parameters/pml",
        "/sys/module/kvm_amd/parameters/pml",
    ]
    .iter()
    .any(|p| std::fs::read(p).is_ok_and(|v| matches!(v.first(), Some(b'Y' | b'1'))))
}

/// The scratch one harvest uses, kept for the next.
#[cfg(any(unix, test))]
#[derive(Default)]
pub(crate) struct Scratch {
    watched: Vec<Range<u64>>,
    /// The pages written, a word a granule of `watched`.
    written: Vec<u64>,
    /// The pages a logged slot of address space 0 answered for, a word a granule.
    known: Vec<u64>,
    /// The first granule of each run of `watched`, in their order.
    firsts: Vec<usize>,
    log: Vec<u64>,
    clear: Vec<u64>,
}

/// The `n` bits of `bitmap` from bit `at`, `n` at most 64, in the low bits of the result.
#[cfg(any(unix, test))]
fn bits(bitmap: &[u64], at: u64, n: u64) -> u64 {
    let (w, s) = ((at / 64) as usize, at % 64);
    let lo = bitmap.get(w).copied().unwrap_or(0) >> s;
    let hi = if s == 0 {
        0
    } else {
        bitmap.get(w + 1).copied().unwrap_or(0) << (64 - s)
    };
    let all = lo | hi;
    if n >= 64 {
        all
    } else {
        all & ((1u64 << n) - 1)
    }
}

/// Sets in `bitmap` the low `n` bits of `value` from bit `at`.
#[cfg(any(unix, test))]
fn set_bits(bitmap: &mut [u64], at: u64, value: u64) {
    let (w, s) = ((at / 64) as usize, at % 64);
    if let Some(x) = bitmap.get_mut(w) {
        *x |= value << s;
    }
    if s != 0
        && let Some(x) = bitmap.get_mut(w + 1)
    {
        *x |= value >> (64 - s);
    }
}

/// One slot's part of a harvest (step 3 of the module docs).
#[cfg(any(unix, test))]
fn harvest_slot(host: &dyn DirtyHost, kvm: &dyn KvmLog, s: &mut Scratch, slot: KvmSlot) {
    let base = slot.guest_physical_address;
    let end = base.saturating_add(slot.length);
    let pages = slot.length / PAGE_SIZE;
    let primary = slot.slot >> 16 == 0;
    // The granules of the watched runs this slot holds a page of, by run.
    let mut parts: Vec<(usize, Range<u64>)> = Vec::new();
    for (i, run) in s.watched.iter().enumerate() {
        let (a, b) = (run.start.max(base), run.end.min(end));
        if a < b {
            parts.push((i, a..b));
        }
    }
    let (Some(first), Some(last)) = (parts.first(), parts.last()) else {
        return;
    };
    let lo = (first.1.start - base) / PAGE_SIZE;
    let hi = (last.1.end - base).div_ceil(PAGE_SIZE);
    s.log.clear();
    s.log.resize(pages.div_ceil(64) as usize, 0);
    let read = kvm.get(slot.slot, &mut s.log);
    let first_page = lo / 64 * 64;
    let num_pages = (hi.div_ceil(64) * 64).min(pages) - first_page;
    s.clear.clear();
    s.clear.resize(num_pages.div_ceil(64) as usize, 0);
    let mut cleared = false;
    for (i, part) in &parts {
        let run_start = s.watched[*i].start;
        let mut g = (part.start - run_start) / GRANULE * GRANULE + run_start;
        while g < part.end {
            let (a, b) = (g.max(part.start), (g + GRANULE).min(part.end));
            let n = (b - a) / PAGE_SIZE;
            let at = (a - base) / PAGE_SIZE;
            let shift = (a - g) / PAGE_SIZE;
            let mask = if n >= 64 {
                !0
            } else {
                ((1u64 << n) - 1) << shift
            };
            let word = s.firsts[*i] + ((g - run_start) / GRANULE) as usize;
            if read {
                let dirty = bits(&s.log, at, n);
                s.written[word] |= dirty << shift;
                if primary {
                    s.known[word] |= mask;
                }
                if dirty != 0 {
                    set_bits(&mut s.clear, at - first_page, dirty);
                    cleared = true;
                }
            } else {
                s.written[word] |= mask;
            }
            g += GRANULE;
        }
    }
    if cleared && !kvm.clear(slot.slot, first_page, num_pages as u32, &s.clear) {
        log::warn!(
            "reims-vgpu: KVM refused a clear of its dirty log for slot {:#x}",
            slot.slot
        );
    }
    if !primary || !read {
        return;
    }
    // The host's own writes, from its record a word a granule: where a granule lies whole in
    // the slot and keeps its offset in the record; elsewhere its pages read as written.
    for (i, part) in &parts {
        let run_start = s.watched[*i].start;
        let mut g = (part.start - run_start) / GRANULE * GRANULE + run_start;
        while g < part.end {
            let word = s.firsts[*i] + ((g - run_start) / GRANULE) as usize;
            let offset = slot.host_offset.wrapping_add(g.wrapping_sub(base));
            let whole = g >= base && g + GRANULE <= end && offset.is_multiple_of(GRANULE);
            if whole {
                // Every whole granule from here on in this part, in one take.
                let mut e = g;
                while e + GRANULE <= part.end.min(end) {
                    e += GRANULE;
                }
                let words = ((e - g) / GRANULE) as usize;
                if let Some(out) = s.written.get_mut(word..word + words) {
                    host.take_host_writes(offset, e - g, out);
                }
                g = e;
            } else {
                s.known[word] = 0;
                g += GRANULE;
            }
        }
    }
}

/// The log's side of a harvest: [`DirtyLog`]'s, or a test's.
#[cfg(any(unix, test))]
pub(crate) trait HarvestLog {
    /// [`DirtyLog::begin`].
    fn begin(&self, watched: &mut Vec<Range<u64>>) -> Option<DirtyStamp>;
    /// [`DirtyLog::end`].
    fn end(&self, stamp: DirtyStamp, written: &[u64]) -> bool;
}

#[cfg(any(unix, test))]
impl HarvestLog for DirtyLog {
    fn begin(&self, watched: &mut Vec<Range<u64>>) -> Option<DirtyStamp> {
        DirtyLog::begin(self, watched)
    }

    fn end(&self, stamp: DirtyStamp, written: &[u64]) -> bool {
        DirtyLog::end(self, stamp, written)
    }
}

/// One harvest of `log` (the module docs).
#[cfg(any(unix, test))]
pub(crate) fn harvest(
    log: &dyn HarvestLog,
    host: &dyn DirtyHost,
    kvm: &dyn KvmLog,
    flush: bool,
    s: &mut Scratch,
) -> Option<DirtyStamp> {
    let stamp = log.begin(&mut s.watched)?;
    s.firsts.clear();
    let mut granules = 0usize;
    for run in &s.watched {
        s.firsts.push(granules);
        granules += ((run.end - run.start) / GRANULE) as usize;
    }
    s.written.clear();
    s.written.resize(granules, 0);
    s.known.clear();
    s.known.resize(granules, 0);
    if !flush || host.flush_vcpus(FLUSH_TIMEOUT) {
        let mut scratch = std::mem::take(s);
        if !host.for_each_slot(&mut |slot| harvest_slot(host, kvm, &mut scratch, slot)) {
            scratch.known.fill(0);
        }
        *s = scratch;
    }
    for (w, k) in s.written.iter_mut().zip(&s.known) {
        *w |= !k;
    }
    log.end(stamp, &s.written);
    Some(stamp)
}

/// What the harvest thread and the log's hook share.
struct Wake {
    #[cfg(any(unix, test))]
    wanted: AtomicBool,
    stop: AtomicBool,
    thread: std::sync::OnceLock<Thread>,
}

impl Wake {
    fn wake(&self) {
        if let Some(t) = self.thread.get() {
            t.unpark();
        }
    }
}

/// The device's harvest of a host's dirty source: its log and the thread that harvests it. Its
/// drop retires the log and ends the thread.
pub(crate) struct Harvester {
    log: Arc<DirtyLog>,
    wake: Arc<Wake>,
    thread: Option<JoinHandle<()>>,
}

impl Harvester {
    /// The log of the guest's RAM `ram` (guest physical ranges), harvested from `source`.
    #[cfg(target_os = "linux")]
    pub(crate) fn new(ram: &[Range<u64>], source: DirtySource) -> Harvester {
        let kvm = VmFd(source.kvm_vm_fd);
        Harvester::with_kvm(ram, source.host, Box::new(kvm), cpu_buffers_writes())
    }

    /// As [`Harvester::new`], over `kvm`, flushing the vCPUs where `flush` (tests).
    #[cfg(any(unix, test))]
    pub(crate) fn with_kvm(
        ram: &[Range<u64>],
        host: Arc<dyn DirtyHost>,
        kvm: Box<dyn KvmLog>,
        flush: bool,
    ) -> Harvester {
        let wake = Arc::new(Wake {
            wanted: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            thread: std::sync::OnceLock::new(),
        });
        let hook = wake.clone();
        let log = Arc::new(DirtyLog::new(
            ram,
            Box::new(move || {
                hook.wanted.store(true, Ordering::SeqCst);
                hook.wake();
            }),
        ));
        let (l, w) = (log.clone(), wake.clone());
        let thread = thread::Builder::new()
            .name("reims-vgpu.dirty".into())
            .spawn(move || {
                let mut scratch = Scratch::default();
                let mut started = false;
                loop {
                    if w.stop.load(Ordering::SeqCst) {
                        return;
                    }
                    if !w.wanted.swap(false, Ordering::SeqCst) {
                        thread::park();
                        continue;
                    }
                    if !started {
                        host.start();
                        started = true;
                    }
                    harvest(&*l, &*host, &*kvm, flush, &mut scratch);
                }
            });
        let thread = match thread {
            Ok(t) => {
                let _ = wake.thread.set(t.thread().clone());
                wake.wake();
                Some(t)
            }
            Err(e) => {
                log::warn!(
                    "reims-vgpu: no dirty-log harvest thread ({e}); every page reads as written"
                );
                None
            }
        };
        Harvester { log, wake, thread }
    }

    /// The log.
    pub(crate) fn log(&self) -> &Arc<DirtyLog> {
        &self.log
    }
}

impl Drop for Harvester {
    fn drop(&mut self) {
        self.log.retire();
        self.wake.stop.store(true, Ordering::SeqCst);
        self.wake.wake();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests;
