//! The EFI display: the firmware's framebuffer in guest RAM, which the guest programs through the
//! device's EFI registers and the host sees as frames presented to the port-0 display.
//!
//! **Registers.** Each write stores its value, then rebuilds:
//! - the start page and the length rebuild the framebuffer's task, then its mode;
//! - the depth, the mode (width in the low 16 bits, height in the high 16) and the stride
//!   rebuild the mode only.
//!
//! The task rebuild destroys a ready task, then asks the host for a task of the framebuffer's
//! length (a page multiple, at most 64 MiB). The mode rebuild stops presentation, checks the
//! stride (4 bytes a pixel at least, aligned as Metal asks for BGRA8), the mode, the depth (32)
//! and the task, maps the framebuffer read-only through the task, makes a BGRA8 texture over a
//! no-copy buffer of the mapping, and schedules presents.
//!
//! Two of the device's limits reach the guest. The stride-alignment register (0x1224) reads the
//! device's `minimumLinearTextureAlignmentForPixelFormat:` of BGRA8Unorm, cut to 32 bits, and a
//! mode register wider than `maxTextureWidth2D` or taller than `maxTextureHeight2D` makes a mode
//! of 0 by 0 (equal is accepted).
//!
//! **Presents.** With a host trace range over the framebuffer: one present at once, then one per
//! traced write, coalesced by a DATA_ADD source on the present queue. Without one: a 100 ms timer
//! (1 ms leeway) on the present queue, starting at the wall clock's now. A present sends the
//! texture to the port-0 display as a 'BGRA' frame, signalled at once (EFI ignores sleep); with
//! no display bound, presentation stops until the next rebuild.
//!
//! **Threads and locks.** Register writes run on the MMIO thread, under the device's display
//! mutex; presents run on the present queue. The state is under one leaf mutex, never held
//! across a queue sync, a device call, a Metal send or a present. Stopping presentation runs on
//! the present queue: the mode rebuild and the teardown `sync` onto it, so the block, with the
//! host's `removeTraceRange` and `destroyTask`, runs on the MMIO thread (the suite's
//! `efi_teardown` and `efi_rebuild_conditions`; the device's own teardown is
//! `tests::the_device_teardown_destroys_the_task_through_the_host`).
//!
//! **A display's last release here hangs, as in Apple's framework.** A present, and the
//! mode-count and mode-value reads, load the bound display and hold it to their end, so a host
//! release of the display meanwhile leaves its last reference there, and the display's end runs
//! there: its unplug detaches it under the device's display mutex. A register write holds that
//! mutex while its mode rebuild waits for the present queue, so a present that ends the display
//! then waits for the mutex, and the register write for the present, for good; a mode-count or
//! mode-value read holds the mutex around its own load, so its end waits for a lock its own
//! thread holds. Apple's framework is built the same way, and hangs the same (testing.md 7.3;
//! the tests are `device::displays::tests::efi_display_holding_the_last_reference`).
//!
//! **Raises.** No handler is live on any path into the EFI display: the register writes (and
//! the device's binding, teardown and resume replay) run in the host's call, which has none, and
//! a present in its block, which starts with none and holds under [`Hold::NONE`]. So every raise
//! site in it is labelled as having none: a Metal or display raise dies at its send, and
//! [`unraised`] only restates that.
//!
//! **Edge cases.**
//! - A task rebuild destroys the ready task and keeps it ready; a refusal after the destroy
//!   leaves a stale task, which the next mode rebuild maps with (testing.md 7.1; the suite's
//!   `efi_stale_task` cases).
//! - The timer starts at the wall time of the call and holds the EFI display: one whose timer is
//!   never cancelled is leaked with it (an `rt::timer::Timer` dropped uncancelled keeps firing,
//!   `rt::timer::tests::dropped_uncancelled_it_keeps_firing`; the timer's presents are the
//!   suite's `efi_timer` and `efi_gap_timer_collapsed`).
//! - The source holds the EFI display weakly; its handler lets its load go off the present
//!   queue (`rt::held::release_off_thread`), so a last release there runs after the block,
//!   elsewhere.
//! - The tiled path (Metal's `newTiledTextureWithBytesNoCopy:…`) is dead: the device is asked
//!   whether it responds, and never does.

use std::sync::{Arc, Weak};
use std::time::Duration;

use parking_lot::Mutex;

use crate::device::DeviceCore;
use crate::display::{DeathKind, Display, DisplayHandle, GammaTableArgs};
use crate::guest::records::APV_PAGE_SHIFT;
use crate::host::{HostPtr, HostTask, HostTraceRange, PhysRange, TraceRangeHandler};
use crate::memory::mapping::Mapping;
use crate::num::{self, ToUsize, Truncate};
use crate::platform::descriptor::TextureDescriptor as _;
use crate::platform::device::GpuDevice as _;
use crate::platform::os::{Clock, Deadline};
use crate::platform::resource::{Buffer as _, GpuResource as _};
use crate::platform::values::TextureDesc;
use crate::platform::{Handle as _, Platform};
use crate::raise::{OnFault, Raise};
use crate::rt::held::release_off_thread;
use crate::rt::queue::{Executor, SerialQueue};
use crate::rt::signal::CoalescingSignal;
use crate::rt::timer::Timer;
use crate::rt::{Hold, Pool};
use crate::tasks::TaskInfo;

/// The present queue's label.
const PRESENT_QUEUE_LABEL: &str = "reims-vgpu.efi-present";
/// `MTLPixelFormatBGRA8Unorm`.
const MTL_PIXEL_FORMAT_BGRA8_UNORM: u64 = 80;
/// `MTLTextureUsageShaderRead`.
const MTL_TEXTURE_USAGE_SHADER_READ: u64 = 1;
/// `'BGRA'`, the IOSurface pixel format of every EFI present.
const IOSURFACE_PIXEL_FORMAT_BGRA: u32 = 0x4247_5241;
/// The framebuffer page shift: the start register counts guest pages (16 KiB, 4 KiB on x86-64,
/// as the option ROM writes it).
const PAGE_SHIFT: u32 = APV_PAGE_SHIFT;
/// The page mask a length must clear.
pub(crate) const PAGE_MASK: u32 = num::bit32(PAGE_SHIFT).wrapping_sub(1);
/// The largest framebuffer length, 64 MiB.
const MAX_FRAMEBUFFER_LENGTH: u32 = 0x400_0000;
/// The only depth that presents.
const FRAMEBUFFER_DEPTH: u32 = 32;
/// The timer's interval.
const PRESENT_INTERVAL: Duration = Duration::from_millis(100);
/// The timer's leeway.
const PRESENT_LEEWAY: Duration = Duration::from_millis(1);

/// The framebuffer registers, as the guest wrote them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Registers {
    /// The first page of the framebuffer.
    pub(crate) start_page: u32,
    /// Its length in bytes.
    pub(crate) length: u32,
    /// Bits per pixel.
    pub(crate) depth: u32,
    /// The mode: the width in the low 16 bits, the height in the high 16. The raw value stays
    /// when the size is refused.
    pub(crate) mode: u32,
    /// Bytes per row; 0 for 4 bytes a pixel.
    pub(crate) stride: u32,
}

/// The framebuffer's task, which the host makes at the length's size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct FramebufferTask {
    /// The host's task and its base. The host fills it as it goes, even when it refuses; a
    /// destroy nulls the task and keeps the base.
    pub(crate) info: TaskInfo,
    /// Whether the mode rebuild may map through it. A task rebuild's destroy leaves it set.
    pub(crate) ready: bool,
}

/// What the register writes and the present queue share, under [`EfiDisplay::state`]'s leaf
/// mutex.
pub(crate) struct EfiState<P: Platform> {
    /// The port-0 display the presents go to, held weakly: the device binds it when it makes the
    /// EFI display, and clears it when that display detaches.
    pub(crate) display: Option<Weak<Display<P>>>,
    /// The mode index the guest selected, for the mode-value register; a new binding resets it.
    pub(crate) mode_select: u32,
    /// The registers.
    pub(crate) regs: Registers,
    /// The mode's width: its low 16 bits, or 0 when Metal cannot make a texture that large.
    pub(crate) width: u32,
    /// Its height: its high 16 bits, or 0 likewise.
    pub(crate) height: u32,
    /// The framebuffer's task.
    pub(crate) task: FramebufferTask,
    /// The framebuffer, mapped through the task; its last release unmaps it.
    pub(crate) mapping: Option<Arc<Mapping<P>>>,
    /// The BGRA8 texture over a no-copy buffer of the mapping (the buffer's one owner).
    pub(crate) texture: Option<P::Texture>,
    /// The length the framebuffer was last mapped with.
    pub(crate) mapped_length: u32,
    /// The framebuffer's size: the height times the stride, rounded up to a page, or the product
    /// unrounded (wrapped on overflow) when it did not fit the mapping.
    pub(crate) total_size: u32,
    /// The timer of timed presents.
    pub(crate) timer: Option<Timer>,
    /// The host's trace range of traced presents.
    pub(crate) trace_range: Option<HostTraceRange>,
}

impl<P: Platform> EfiState<P> {
    fn new() -> Self {
        EfiState {
            display: None,
            mode_select: 0,
            regs: Registers::default(),
            width: 0,
            height: 0,
            task: FramebufferTask::default(),
            mapping: None,
            texture: None,
            mapped_length: 0,
            total_size: 0,
            timer: None,
            trace_range: None,
        }
    }
}

/// The EFI display (module doc). The device owns it; the timer holds it while presents are
/// timed.
pub struct EfiDisplay<P: Platform> {
    /// The device, unretained: it upgrades until the device's teardown has ended, so
    /// the teardown's `destroyTask` and `removeTraceRange` reach the host
    /// (`teardown::the_device_teardown_destroys_the_task_through_the_host`). A device that is
    /// gone is sent nothing and answers zero.
    device: Weak<DeviceCore<P>>,
    /// `reims-vgpu.efi-present`: presents, and every stop of presentation.
    pub(crate) present_queue: SerialQueue,
    /// The shared state.
    pub(crate) state: Mutex<EfiState<P>>,
    /// The DATA_ADD source that presents on the present queue, holding the EFI display weakly.
    /// Its last release cancels it and drops what is pending (never cancelled otherwise).
    pub(crate) source: CoalescingSignal,
    platform: Arc<P>,
    /// Runs once at the next load of the bound display, after the load and before anything the
    /// loader does with it (tests).
    #[cfg(test)]
    pub(crate) loaded_probe: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

/// A send's or a present's value. No handler is live on any path into the EFI display (module
/// doc), so a raise has already died at its site; were one live, it could not leave these
/// methods, which answer no raise, so it dies here.
fn unraised<T, E: Into<Raise>>(r: Result<T, E>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => e.into().fatal(),
    }
}

impl<P: Platform> EfiDisplay<P> {
    /// The EFI display of `device`: its present queue and source made, its state zero.
    pub fn new(platform: &Arc<P>, device: Weak<DeviceCore<P>>) -> Arc<Self> {
        Arc::new_cyclic(|this: &Weak<EfiDisplay<P>>| {
            let present_queue = SerialQueue::new(platform.threads(), PRESENT_QUEUE_LABEL);
            let target: Arc<dyn Executor> = Arc::new(present_queue.clone());
            let weak_self = this.clone();
            let source = CoalescingSignal::new_add(target, move |_| Self::source_fired(&weak_self));
            EfiDisplay {
                device,
                present_queue,
                state: Mutex::new(EfiState::new()),
                source,
                platform: platform.clone(),
                #[cfg(test)]
                loaded_probe: Mutex::new(None),
            }
        })
    }

    /// The probe of a load, once (tests).
    #[cfg(test)]
    fn after_load(&self) {
        let probe = self.loaded_probe.lock().take();
        if let Some(probe) = probe {
            probe();
        }
    }

    /// The source's handler: presents when the EFI display is alive, nothing once its last
    /// release has begun. A teardown's release meanwhile leaves the last reference to the
    /// handler, which lets it go after the block, off the present queue (the teardown waits in
    /// its `sync` there).
    fn source_fired(weak_self: &Weak<Self>) {
        if let Some(s) = weak_self.upgrade() {
            // A block on the present queue: no handler.
            s.present(Hold::NONE);
            let platform = Arc::clone(&s.platform);
            release_off_thread(platform.threads(), s);
        }
    }

    /// The device, while it lives.
    fn device(&self) -> Option<Arc<DeviceCore<P>>> {
        self.device.upgrade()
    }

    /// Dies unless on the present queue (always true: every caller is on it).
    fn assert_on_present_queue(&self) {
        crate::fatal_unless!(
            self.present_queue.is_current(),
            DeathKind::EfiOffPresentQueue
        );
    }

    /// The host destroys the task; the device nulls the task in the info it was given, which
    /// [`DeviceCore::destroy_task`] (over a `&TaskInfo`) cannot, so it is done here. Sent whether
    /// or not a task is ready.
    fn destroy_task(&self) {
        let info = self.state.lock().task.info;
        if let Some(d) = self.device() {
            d.destroy_task(&info);
            self.state.lock().task.info.task = HostTask::NULL;
        }
    }

    /// The device's EFI register turns the display off: presentation stops and the task is
    /// destroyed, on the present queue (run on this thread). The caller's release follows.
    pub fn teardown(self: &Arc<Self>) {
        let this = Arc::clone(self);
        self.present_queue.sync(move || {
            this.stop_presenting();
            this.destroy_task();
        });
    }

    /// Cancels the presents, then releases the texture (it keeps the no-copy buffer) and then the
    /// mapping (its last release unmaps the framebuffer). The mapped length stays.
    fn stop_presenting(&self) {
        self.assert_on_present_queue();
        self.cancel_presents();
        let texture = self.state.lock().texture.take();
        drop(texture);
        let mapping = self.state.lock().mapping.take();
        drop(mapping);
    }

    /// The task rebuild (module doc).
    fn rebuild_task(&self) {
        if self.state.lock().task.ready {
            // The task is destroyed and stays ready; a refusal below leaves it so, and the next
            // mode rebuild maps with the NULL task (testing.md 7.1).
            self.destroy_task();
        }
        let Registers {
            start_page, length, ..
        } = self.state.lock().regs;
        if start_page == 0 || length == 0 {
            return;
        }
        if length & PAGE_MASK != 0 {
            log::warn!("EFI framebuffer length {length:#x} is not page aligned");
            return;
        }
        if length > MAX_FRAMEBUFFER_LENGTH {
            log::warn!(
                "EFI framebuffer length {length:#x} is past the limit of {MAX_FRAMEBUFFER_LENGTH:#x}"
            );
            return;
        }
        let mut info = self.state.lock().task.info;
        let created = match self.device() {
            Some(d) => d.create_task(u64::from(length), &mut info),
            None => false,
        };
        let mut s = self.state.lock();
        s.task.info = info;
        if !created {
            drop(s);
            log::error!("no task made for the EFI framebuffer of {length:#x} bytes");
            self.state.lock().task.ready = false;
            return;
        }
        s.task.ready = true;
    }

    /// The mode rebuild (module doc).
    fn rebuild_mode(self: &Arc<Self>) {
        let this = Arc::clone(self);
        self.present_queue.sync(move || this.stop_presenting());
        let (width, stride) = {
            let s = self.state.lock();
            (s.width, s.regs.stride)
        };
        // Never overflows: the width is at most 0xffff.
        let (minimum_stride, overflow) = width.overflowing_mul(4);
        if overflow {
            log::warn!("EFI framebuffer width {width} overflows its stride");
            return;
        }
        let stride = if stride != 0 { stride } else { minimum_stride };
        // Before the checks below; the alignment is asked only of a stride long enough. An
        // alignment of 0 leaves the stride as the remainder (`udiv`'s), which Metal never
        // answers.
        let stride_valid =
            stride >= minimum_stride && num::rem0(stride, self.get_stride_alignment()) == 0;
        let (task, regs) = {
            let s = self.state.lock();
            (s.task, s.regs)
        };
        // Silently.
        if !task.ready || regs.mode == 0 || regs.depth != FRAMEBUFFER_DEPTH || !stride_valid {
            return;
        }
        let range = PhysRange {
            physical_address: u64::from(regs.start_page).wrapping_shl(PAGE_SHIFT),
            physical_length: u64::from(regs.length),
        };
        // With a stale task, through the NULL task.
        let mapping = match self.device() {
            Some(d) => d.new_task_mapping(&task.info, &[range], u64::from(regs.length), 0, true),
            None => None,
        };
        self.state.lock().mapping = mapping.clone();
        let Some(mapping) = mapping else {
            log::warn!(
                "EFI framebuffer of {:#x} bytes at guest address {:#x} could not be mapped",
                range.physical_length,
                range.physical_address
            );
            self.destroy_task();
            self.state.lock().task.ready = false;
            return;
        };
        let (height, total) = {
            let mut s = self.state.lock();
            s.mapped_length = regs.length;
            // The product is stored even when it overflows.
            let (total, overflow) = s.height.overflowing_mul(stride);
            s.total_size = total;
            if overflow || total > s.mapped_length {
                let (height, mapped_length) = (s.height, s.mapped_length);
                drop(s);
                // The mapping is kept; nothing is scheduled.
                log::warn!(
                    "EFI framebuffer of {height} rows of {stride:#x} bytes does not fit the \
                     {mapped_length:#x} bytes mapped"
                );
                return;
            }
            let total = total.wrapping_add(PAGE_MASK) & !PAGE_MASK;
            s.total_size = total;
            (s.height, total)
        };
        // The descriptor lives to the block's end.
        {
            let descriptor = self.platform.texture_descriptor(&TextureDesc {
                width: u64::from(width),
                height: u64::from(height),
                usage: MTL_TEXTURE_USAGE_SHADER_READ,
                pixel_format: MTL_PIXEL_FORMAT_BGRA8_UNORM,
                ..TextureDesc::default()
            });
            let device = self.device();
            // No deallocator; nil for a size of 0.
            let buffer = match &device {
                Some(d) => unraised(new_framebuffer_buffer::<P>(
                    d.mtl_device_ref(),
                    mapping.virtual_address(),
                    u64::from(total),
                )),
                None => None,
            };
            // A send to a nil buffer answers a nil texture. The texture takes the buffer's
            // resource options, as Metal's texture over a buffer must: the descriptor's default
            // storage is Managed, and with Metal's validation layer on, a texture of other options
            // than its buffer's ends the process ("resourceOptions (0x10) must match backing
            // buffer resource options (0x0)", measured on the M3 Max under macOS 27.0.1).
            let texture = match &buffer {
                Some(b) => {
                    descriptor.set_resource_options(b.resource_options());
                    unraised(b.new_texture(OnFault::Die, &descriptor, 0, u64::from(stride)))
                }
                None => None,
            };
            self.state.lock().texture = texture;
            // The texture keeps the buffer.
            drop(buffer);
        }
        self.schedule_presents();
    }

    /// Removes the trace range from the host (when there is one), then cancels the timer.
    fn cancel_presents(&self) {
        self.assert_on_present_queue();
        let trace_range = self.state.lock().trace_range;
        if let Some(r) = trace_range {
            if let Some(d) = self.device() {
                d.remove_trace_range(r);
            }
            self.state.lock().trace_range = None;
        }
        let timer = self.state.lock().timer.take();
        if let Some(t) = timer {
            t.cancel();
            drop(t);
        }
    }

    /// Asks the host to trace the framebuffer: traced presents when it does, timed ones when it
    /// answers none (module doc).
    fn schedule_presents(self: &Arc<Self>) {
        let range = {
            let s = self.state.lock();
            PhysRange {
                physical_address: u64::from(s.regs.start_page).wrapping_shl(PAGE_SHIFT),
                physical_length: u64::from(s.total_size),
            }
        };
        // The handler only merges into the source; the host may call it on any thread.
        let source = self.source.clone();
        let handler: TraceRangeHandler = Arc::new(move |_dirty| source.merge(1));
        let trace_range = match self.device() {
            Some(d) => d.add_trace_range(range, handler),
            None => HostTraceRange::NULL,
        };
        let trace_range = (!trace_range.is_null()).then_some(trace_range);
        self.state.lock().trace_range = trace_range;
        if trace_range.is_some() {
            // One present now, then one per traced write, coalesced.
            self.source.merge(1);
            return;
        }
        // Resumed at once; the handler holds the EFI display (module doc).
        let start = Deadline::Wall(self.platform.clock().now_wall());
        let this = Arc::clone(self);
        let timer = Timer::start(
            &self.present_queue,
            start,
            PRESENT_INTERVAL,
            PRESENT_LEEWAY,
            // A timer's handler: no handler.
            move |_| this.present(Hold::NONE),
        );
        self.state.lock().timer = Some(timer);
    }

    /// One present (module doc), in a pool (the present's Metal `+0` results) popped at its end,
    /// under the present queue's block's `hold`.
    fn present(&self, hold: Hold<'_>) {
        self.assert_on_present_queue();
        let _pool = Pool::push(&*self.platform);
        let (display, texture) = {
            let s = self.state.lock();
            (
                s.display.as_ref().and_then(Weak::upgrade),
                s.texture.as_ref().map(|t| t.clone_ref()),
            )
        };
        // The display, loaded: bound before the present, it is released after it, and after
        // the rest the present held, before the pool pops.
        let Some((_display, display)) = display.and_then(Self::loaded) else {
            // Presentation stops until the next rebuild.
            drop(texture);
            self.stop_presenting();
            return;
        };
        #[cfg(test)]
        self.after_load();
        let Some(texture) = texture else { return };
        // The completion runs inside the present, and always signals.
        let completion_display = Arc::clone(&display);
        unraised(display.present_frame(
            hold,
            OnFault::Die,
            Some(texture),
            IOSURFACE_PIXEL_FORMAT_BGRA,
            0,
            None,
            GammaTableArgs::default(),
            None,
            Box::new(move || completion_display.signal_current_frame()),
        ));
    }

    /// A weak load of the display: the host's display, with a retain of it that the caller binds
    /// first, so a host release meanwhile leaves the last one to the caller's end, after its
    /// other references. Nil once the display's last release has begun.
    fn loaded(display: Arc<Display<P>>) -> Option<(DisplayHandle<P>, Arc<Display<P>>)> {
        Some((display.retain_self()?, display))
    }

    /// The bound display, loaded (see [`EfiDisplay::loaded`]).
    fn load_display(&self) -> Option<(DisplayHandle<P>, Arc<Display<P>>)> {
        let display = self.state.lock().display.as_ref().and_then(Weak::upgrade)?;
        let loaded = Self::loaded(display);
        #[cfg(test)]
        if loaded.is_some() {
            self.after_load();
        }
        loaded
    }

    /// Binds `display` (weakly; `None` unbinds) and resets the mode select.
    pub fn set_display(&self, display: Option<Weak<Display<P>>>) {
        let mut s = self.state.lock();
        s.display = display;
        s.mode_select = 0;
    }

    /// The mode-select register: any index is stored.
    pub fn set_mode_select(&self, mode: u32) {
        self.state.lock().mode_select = mode;
    }

    /// The bound display's mode count, truncated; 0 without one.
    pub fn get_mode_count(&self) -> u32 {
        // The display's retain, bound first, goes last.
        let Some((_display, display)) = self.load_display() else {
            return 0;
        };
        display.mode_list().len().low_u32()
    }

    /// The selected mode's size, packed as the mode register packs it; all ones past the end of
    /// the list, or without a display.
    pub fn get_mode_value(&self) -> u32 {
        // The display's retain, bound first, goes last.
        let (_display, mode_list) = match self.load_display() {
            Some((retain, display)) => (Some(retain), display.mode_list()),
            None => (None, Vec::new()),
        };
        let index = self.state.lock().mode_select;
        let Some(mode) = mode_list.get(index.to_usize()) else {
            return u32::MAX;
        };
        let size = mode.size_in_pixels;
        u32::from(size.x) | u32::from(size.y).wrapping_shl(16)
    }

    /// The row alignment Metal asks for BGRA8, truncated; 0 without a device.
    pub fn get_stride_alignment(&self) -> u32 {
        match self.device() {
            Some(d) => unraised(d.mtl_device_ref().caps().min_linear_alignment_bgra8()).low_u32(),
            None => 0,
        }
    }

    /// The start-page register.
    pub fn set_framebuffer_start(self: &Arc<Self>, start: u32) {
        self.state.lock().regs.start_page = start;
        self.rebuild_task();
        self.rebuild_mode();
    }

    /// The length register.
    pub fn set_framebuffer_length(self: &Arc<Self>, length: u32) {
        self.state.lock().regs.length = length;
        self.rebuild_task();
        self.rebuild_mode();
    }

    /// The depth register.
    pub fn set_framebuffer_depth(self: &Arc<Self>, depth: u32) {
        self.state.lock().regs.depth = depth;
        self.rebuild_mode();
    }

    /// The mode register: the size is split from it, and zeroed when Metal cannot make a
    /// texture that large (equal is accepted); the raw value stays.
    pub fn set_framebuffer_mode(self: &Arc<Self>, mode: u32) {
        let (width, height) = {
            let mut s = self.state.lock();
            s.regs.mode = mode;
            s.width = mode & 0xffff;
            s.height = mode.wrapping_shr(16);
            (s.width, s.height)
        };
        // Without a device, both limits are 0.
        let (max_width, max_height) = self.device().map_or((0, 0), |d| {
            let limits = d.mtl_device_ref().caps().limits;
            (limits.max_texture_width_2d, limits.max_texture_height_2d)
        });
        let too_large = u64::from(width) > max_width || u64::from(height) > max_height;
        if too_large {
            log::warn!("EFI framebuffer mode {width}x{height} is larger than Metal's textures");
            let mut s = self.state.lock();
            s.width = 0;
            s.height = 0;
        }
        self.rebuild_mode();
    }

    /// The stride register.
    pub fn set_framebuffer_stride(self: &Arc<Self>, stride: u32) {
        self.state.lock().regs.stride = stride;
        self.rebuild_mode();
    }
}

impl<P: Platform> Drop for EfiDisplay<P> {
    /// What presentation still holds, in order: the mapping (its last release unmaps the
    /// framebuffer), then the texture, then the timer (none: a timer holds the EFI display). Then
    /// the fields: the queue, the rest of the state, the source (its last release cancels it,
    /// dropping what is pending; the handler holds the EFI display weakly, so a later merge would
    /// present nothing anyway).
    fn drop(&mut self) {
        let s = self.state.get_mut();
        drop(s.mapping.take());
        drop(s.texture.take());
        drop(s.timer.take());
    }
}

/// `newBufferWithBytesNoCopy:length:options:0 deallocator:nil` over the framebuffer mapping, for
/// a register write (an API entry with no handler), its bytes never watched by the device's
/// dirty log ([`HostRange::unlogged`](crate::platform::HostRange::unlogged)): the firmware's
/// presents would start the host's log within seconds of the boot, at the cost of the guest's
/// huge pages for the whole run, to skip copies of a framebuffer that lives only until the
/// guest's own driver takes the display.
#[allow(unsafe_code)] // architecture 17.3: the call of an `unsafe fn` over host memory
fn new_framebuffer_buffer<P: Platform>(
    device: &P::Device,
    pointer: HostPtr,
    length: u64,
) -> crate::raise::RaiseResult<Option<P::Buffer>> {
    // SAFETY: the mapping behind `pointer` (`HostRange::new`'s contract) is the EFI display's,
    // which it releases only when presentation stops, after releasing the texture, the buffer's
    // only owner here. A display that retained the texture as its current frame can outlive the
    // mapping; that lifetime is kept.
    let bytes = unsafe { crate::platform::HostRange::new(pointer, length) }.unlogged();
    device.new_buffer_over_host_memory(OnFault::Die, bytes, 0, None)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    /// The device's teardown tears the EFI display down while the device still upgrades, so
    /// the task's destroy reaches the host's block.
    #[test]
    fn the_device_teardown_destroys_the_task_through_the_host() {
        use crate::device::test_support::{Opts, TestDevice, device};
        let t = device(Opts::default());
        t.device.set_efi_display(1);
        t.device.set_efi_framebuffer_start(1);
        t.device.set_efi_framebuffer_length(0x4000);
        assert_eq!(t.platform.call_log().count("createTask"), 1);
        assert!(t.efi_display().unwrap().state.lock().task.ready);
        t.platform.call_log().clear();
        let TestDevice {
            platform,
            host,
            device,
            ..
        } = t;
        assert_eq!(Arc::strong_count(&device), 1);
        drop(device);
        let sels = platform.call_log().sels();
        drop(host);
        assert!(
            sels.contains(&"destroyTask"),
            "the device's teardown destroys the EFI task: {sels:?}"
        );
    }
}
