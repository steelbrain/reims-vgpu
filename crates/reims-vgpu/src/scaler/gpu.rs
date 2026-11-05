//! The scaler device's backend: core's transform (`reims_vgpu_video_core::scaler::reference`)
//! as kernels of our own, `scaler.metal`, on the GPU the library draws with: Metal on macOS
//! (`gpu/metal.rs`), the Vulkan platform's runtime elsewhere (`gpu/vulkan.rs`), whichever GPU that
//! is, a software rasterizer's included. Nothing chooses it: every host has it where it has that
//! GPU.
//!
//! - **What the kernels compute.** The CPU works out from a transform's plan where each
//!   destination pixel samples and with what weights, the reference's own taps, and every number
//!   a format, a matrix or a window gives ([`params`]); the kernels apply them, a dispatch a plane
//!   of the destination, each thread one 32-bit word of it, after a dispatch that sums a reduction
//!   down into lines where the transform is separable (`scaler.metal`'s header). So the output is
//!   the reference's but for float rounding: the kernels compute in 32-bit floats, the reference
//!   in 64-bit doubles, each sum in the reference's order.
//! - **Guest memory in place.** A surface whose memory the host lends a device
//!   ([`DeviceMemory`]) is bound where it is: its own IOSurface on macOS, the host's view of its
//!   pages on Linux, imported once a mapping and kept until the mapping goes. A destination is
//!   bound so only where its planes start on 4-byte boundaries and their rows are a multiple of 4
//!   bytes, the host lends it for writing (never a surface the guest made read-only for its
//!   clients), and it is not the source's own memory. Anything else is **staged**: the source
//!   copied in its lend into a buffer of the GPU's own, the destination written there and copied
//!   back in its lend, the bytes the reference writes alone ([`covered`]). A destination written
//!   in place is told so once the GPU is done with it
//!   ([`reims_vgpu_video_core::surfaces::GuestSurface::device_wrote`]), as a lend is a write of
//!   its own.
//! - **Statuses**, in [`TransformJob::with_bytes`]'s order: a source that cannot be read, a
//!   destination that cannot be written, a surface too small for its layout
//!   (kIOReturnBadArgument), then the transform's own: kIOReturnUnsupported for a side outside
//!   the capabilities' or an offset past the kernel's 4 GiB, kIOReturnError where the GPU fails or
//!   is not done by its deadline.
//! - **A bounded wait.** The GPU is waited for until core's deadline ([`device_wait`]); work not
//!   done by then is kept with everything it may still write and the device's hold on both
//!   surfaces ([`Lingering`]), and the accelerator answers kIOReturnError to every transform after
//!   it, until the guest resets the device. A destination written in place is told written at the
//!   deadline, as far as the GPU ran, and again once the kept work is done and lets the hold go
//!   (core's `GuestHold::report_late_write`).
//! - **Made ahead.** A Vulkan driver makes a kernel's pipeline at its first dispatch of each
//!   threadgroup size, not where the runtime makes its pipeline state, and with the shader caches
//!   cold the making of the costliest took 2.5 s on MoltenVK on the M3 Max and 1.5 s on an RTX
//!   5080 Laptop GPU (2026-10-26), within the 5 s a transform is given. So once a process,
//!   when the first scaler device opens, a thread of the runtime's dispatches each kernel at each
//!   threadgroup size the transforms dispatch it at ([`make_ahead`], [`params::DISPATCHED`]), one
//!   after another, with parameters of zeros, so that every thread returns at its bounds check, on
//!   a queue apart from the transforms' ([`Gpu::submit_apart`]), binding only a buffer of its own.
//!   Nothing waits for it: neither the device's opening nor a transform, which takes a pipeline
//!   made or in the making as it stands, and makes its own where the making ahead has not reached
//!   it. And it gives way to the transforms: it starts no making while one runs, so that the
//!   driver's making of what a transform waits for shares the CPU with no more than the one it
//!   has in hand. The guest sees nothing of it (docs/video/scaler.md, "Made ahead").
//! - **Speed.** A source pixel is decoded once a line of a reduction, not once a destination pixel
//!   ([`params`]). Two facts the kernels' shapes rest on, of an RTX 5080 Laptop GPU (driver
//!   615.71.09) through the Vulkan platform, 2026-10-25:
//!   - it writes guest memory where the host lends it, in host-cached system memory, across the
//!     bus, and stores with gaps between them crawl there (**measured**): a 4K 420v copy into
//!     BGRA in place took 320 ms with each thread writing its own two by two pixels, so that each
//!     store's words lay 8 bytes apart, about 0.1 GB/s, and 2.8 ms with a threadgroup's words
//!     written side by side, as `scaler_quads` writes them;
//!   - a buffer of the GPU's own in the default CPU cache mode lies in that memory too, and a
//!     write-combined one in the GPU's (**read** in the runtime's notice of it,
//!     `metal2vulkan-degrade-shared-in-system-memory`), so the taps, which every thread reads,
//!     are in a write-combined one ([`Gpu::table`]), and the lines in a private one
//!     ([`Gpu::scratch`]).
//!
//!   Each host's GPU against its native scaler is docs/video/scaler.md, "Speed".
//!
//! [`device_wait`]: reims_vgpu_video_core::device_wait

mod params;

#[cfg(target_os = "macos")]
pub(crate) mod metal;
#[cfg(feature = "vulkan")]
#[cfg_attr(
    target_os = "macos",
    allow(
        dead_code,
        reason = "Linux's; on macOS only the tests run it, on MoltenVK"
    )
)]
pub(crate) mod vulkan;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use reims_vgpu_core::num::ToUsize as _;
use reims_vgpu_video_core::device_memory::DeviceMemory;
use reims_vgpu_video_core::device_wait::{self, GuestHold, Lingering, Unfinished};
use reims_vgpu_video_core::runtime::{self, Answer, Bell, Runtime};
use reims_vgpu_video_core::scaler::backend::{Accelerator, ScalerBackend, TransformJob};
use reims_vgpu_video_core::scaler::capabilities::{Capabilities, FormatClass};
use reims_vgpu_video_core::scaler::coverage::{Window, covered};
use reims_vgpu_video_core::scaler::format::PlaneSpan;
use reims_vgpu_video_core::scaler::io_return::{self, IoReturn};
use reims_vgpu_video_core::scaler::plan::{FormatOf, Plan};
use reims_vgpu_video_core::surfaces::SurfaceError;

pub(crate) use params::Kernel;
use params::{BoundPlane, TooLarge};

/// The smallest side of a surface the scaler takes, either way: Apple's scaler's on the M3 Max
/// (docs/video/devices.md, "The capabilities").
pub(crate) const SMALLEST_SIDE: u32 = 4;

/// The largest: Apple's scaler's.
pub(crate) const LARGEST_SIDE: u32 = 65536;

/// The largest ratios, `MaxHUpscale`, `MaxHDownscale`, `MaxVUpscale` and `MaxVDownscale`:
/// Apple's scaler's.
pub(crate) const MAX_SCALE: [u32; 4] = [8, 8, 8, 4];

/// What the backend asks of a GPU: buffers, the kernels' dispatches, and when they are done.
pub(crate) trait Gpu: Send + Sync + 'static {
    /// A buffer the kernels bind.
    type Buffer: Send + Sync + 'static;
    /// A submission of dispatches.
    type Run: Send + 'static;

    /// The GPU's name, for the log.
    fn name(&self) -> &str;

    /// A buffer of `length` bytes of the GPU's own memory, zeroed, which the CPU reaches.
    fn shared(&self, length: usize) -> Result<Self::Buffer, String>;

    /// A buffer of the GPU's own memory holding `bytes`, which the CPU writes once and never
    /// reads: the taps.
    fn table(&self, bytes: &[u8]) -> Result<Self::Buffer, String>;

    /// A buffer of `length` bytes of the GPU's own memory, which only the GPU reaches: the lines.
    fn scratch(&self, length: usize) -> Result<Self::Buffer, String>;

    /// `buffer`'s bytes as the CPU reaches them: a buffer [`Gpu::shared`] made's.
    fn contents(buffer: &mut Self::Buffer) -> Option<&mut [u8]>;

    /// A buffer over `memory`, and where the surface's first byte lies in it; `Ok(None)` where
    /// this GPU reaches none of it, or none it may write where `writes` says it will.
    fn memory(
        &self,
        memory: &dyn DeviceMemory,
        writes: bool,
    ) -> Result<Option<(Self::Buffer, usize)>, String>;

    /// Encodes `dispatches` in one submission, in order, and submits it; `done` runs once the GPU
    /// is done with it, on a thread of the GPU's.
    fn submit(
        &self,
        dispatches: &[Dispatch<'_, Self::Buffer>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Self::Run, String>;

    /// [`Gpu::submit`], on a queue apart from the transforms', after whose work nothing of
    /// theirs waits: the making ahead's ([`make_ahead`]).
    fn submit_apart(
        &self,
        dispatches: &[Dispatch<'_, Self::Buffer>],
        done: Box<dyn FnOnce() + Send>,
    ) -> Result<Self::Run, String>;

    /// `None` while the GPU runs `run`; then `Ok`, or why it failed.
    fn finished(run: &Self::Run) -> Option<Result<(), String>>;

    /// How long the GPU took for `run`, once it is done, where the GPU says.
    fn gpu_time(run: &Self::Run) -> Option<Duration>;
}

/// How long the GPU took for each transform the process ran, where it said, which the tests'
/// benches read.
#[cfg(test)]
static GPU_TIMES: reims_vgpu_core::sync::Mutex<Vec<Duration>> =
    reims_vgpu_core::sync::Mutex::new(Vec::new());

/// One dispatch of the kernel: its kernel, buffers, parameters and threads.
pub(crate) struct Dispatch<'a, B> {
    pub(crate) kernel: Kernel,
    pub(crate) source: &'a B,
    pub(crate) destination: &'a B,
    pub(crate) taps: &'a B,
    pub(crate) params: &'a [u8],
    pub(crate) lines: &'a B,
    pub(crate) threads: (u32, u32),
    pub(crate) group: (u32, u32),
}

/// The backend: an accelerator a DRIVER_OK, each on the one GPU.
pub(crate) struct GpuScalerBackend<G: Gpu> {
    gpu: Arc<G>,
    runtime: Arc<dyn Runtime>,
    smallest: u32,
}

impl<G: Gpu> GpuScalerBackend<G> {
    /// The backend on `gpu`, waiting on `runtime`'s clock.
    pub(crate) fn new(gpu: Arc<G>, runtime: Arc<dyn Runtime>) -> Self {
        Self {
            gpu,
            runtime,
            smallest: SMALLEST_SIDE,
        }
    }

    /// The same, taking sides from `smallest`: for the tests, which run pictures under Apple's
    /// smallest against the reference too.
    #[cfg(test)]
    pub(crate) fn with_smallest_side(mut self, smallest: u32) -> Self {
        self.smallest = smallest;
        self
    }
}

impl<G: Gpu> ScalerBackend for GpuScalerBackend<G> {
    fn create_accelerator(&self) -> Result<Box<dyn Accelerator>, String> {
        Ok(Box::new(GpuAccelerator {
            gpu: Arc::clone(&self.gpu),
            runtime: Arc::clone(&self.runtime),
            smallest: self.smallest,
            stuck: false,
            lines: None,
            prepared: None,
        }))
    }
}

/// The capabilities: the four classes of [`ScalerFormat`], each both ways, Apple's sides from
/// `smallest`, and Apple's ratios.
///
/// [`ScalerFormat`]: reims_vgpu_video_core::scaler::format::ScalerFormat
fn capabilities(smallest: u32) -> Capabilities {
    Capabilities {
        // No hardware revision of Apple's describes this scaler, and nothing reads it: 0, ours.
        scaler_version: 0,
        formats: vec![
            FormatClass::TwoPlanes420,
            FormatClass::Argb8888,
            FormatClass::Interleaved422,
            FormatClass::TwoPlanes42010,
        ],
        min_source: smallest,
        max_source: (LARGEST_SIDE, LARGEST_SIDE),
        min_destination: smallest,
        max_destination: (LARGEST_SIDE, LARGEST_SIDE),
        max_scale: MAX_SCALE,
        // Windows go anywhere in a surface.
        xy_offsets_with_tiling: (true, true),
    }
}

/// One accelerator: the device's from DRIVER_OK to its reset.
struct GpuAccelerator<G: Gpu> {
    gpu: Arc<G>,
    runtime: Arc<dyn Runtime>,
    smallest: u32,
    /// Whether a transform's work was not done by its deadline: every later one fails.
    stuck: bool,
    /// The lines buffer of the separable transforms, and its bytes: made at the first that needs
    /// it, and again larger where one needs more.
    lines: Option<(G::Buffer, usize)>,
    /// The last transform's work, which the next takes where its plan and planes are the same.
    prepared: Option<Prepared<G>>,
}

/// One dispatch of a transform: its kernel, its parameters' bytes, its threads and its
/// threadgroup's, across and down.
type Pass = (Kernel, Vec<u8>, (u32, u32), (u32, u32));

/// A transform's work, ready to dispatch: what it is for, each pass's kernel, parameters' bytes
/// and threads, and the taps in a buffer of the GPU's.
struct Prepared<G: Gpu> {
    plan: Plan,
    source: Vec<BoundPlane>,
    destination: Vec<BoundPlane>,
    passes: Vec<Pass>,
    taps: G::Buffer,
}

/// How long a wait for the GPU goes before it asks whether the GPU is done, where nothing rang:
/// a backstop to the GPU's ring.
const ASK_AGAIN: Duration = Duration::from_millis(2);

/// How long the making ahead waits for one kernel's dispatch, its pipeline's making included,
/// before it leaves the rest to their first transforms: many times the 2.5 s the costliest took
/// (module docs).
const AHEAD_WAIT: Duration = Duration::from_secs(60);

/// The making ahead's thread.
const AHEAD_THREAD_NAME: &str = "reims-vgpu-video.scaler-ahead";

/// The transforms running in the process, which the making ahead gives way to, and the bell their
/// ends ring.
static RUNNING: AtomicUsize = AtomicUsize::new(0);
static RUNNING_ENDED: LazyLock<Bell> = LazyLock::new(Bell::new);

/// One transform running, counted in [`RUNNING`] while it lives.
struct Running;

impl Running {
    fn start() -> Running {
        RUNNING.fetch_add(1, Ordering::AcqRel);
        Running
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if RUNNING.fetch_sub(1, Ordering::AcqRel) == 1 {
            RUNNING_ENDED.ring();
        }
    }
}

/// Makes each kernel's pipeline ahead of a guest's first transform, on a thread of `runtime`'s
/// (module docs, "Made ahead"): a dispatch of each of [`params::DISPATCHED`] in turn, each of one
/// threadgroup, on the queue apart, its buffers all one buffer of the making's own, made here, and
/// its parameters zeros, so that no thread reads or writes a buffer. Each waits until no transform
/// runs ([`RUNNING`]), or for [`AHEAD_WAIT`] at most. The answer is given once the
/// last is done, or the making stopped: where the GPU refused or failed a dispatch, or did not
/// finish one within [`AHEAD_WAIT`], the kernels after it are made at their first transforms, as
/// they would be without it. `Err` with why where it could not start.
pub(crate) fn make_ahead<G: Gpu>(
    gpu: &Arc<G>,
    runtime: &Arc<dyn Runtime>,
) -> Result<Answer<()>, String> {
    let buffer = gpu.scratch(16)?;
    let (answerer, answer) = runtime::answer();
    let (gpu, clock) = (Arc::clone(gpu), Arc::clone(runtime));
    let body = move || {
        let params = params::Params::default().bytes();
        let started = clock.now();
        for (kernel, group) in params::DISPATCHED {
            let dispatch = Dispatch {
                kernel,
                source: &buffer,
                destination: &buffer,
                taps: &buffer,
                params: &params,
                lines: &buffer,
                threads: group,
                group,
            };
            // Way given to the transforms running, for as long as a making may take.
            let way = clock.deadline_in(AHEAD_WAIT);
            let _ = RUNNING_ENDED.wait_until(&*clock, way, || {
                (RUNNING.load(Ordering::Acquire) == 0).then_some(())
            });
            let bell = Arc::new(Bell::new());
            let ring = Arc::clone(&bell);
            let one = clock.now();
            let run = match gpu.submit_apart(&[dispatch], Box::new(move || ring.ring())) {
                Ok(run) => run,
                Err(why) => {
                    log::warn!(
                        "scaler: {} refused a kernel's making ahead: {why}",
                        gpu.name()
                    );
                    break;
                }
            };
            let deadline = clock.deadline_in(AHEAD_WAIT);
            match bell.wait_until(&*clock, deadline, || G::finished(&run)) {
                Some(Ok(())) => log::debug!(
                    "scaler: {kernel:?} at {group:?} made ahead in {:.1} ms, {:.0} ms after the \
                     making ahead started",
                    clock.elapsed_since(one).as_secs_f64() * 1000.0,
                    clock.elapsed_since(started).as_secs_f64() * 1000.0
                ),
                Some(Err(why)) => {
                    log::warn!("scaler: the making ahead of {kernel:?} failed: {why}");
                    break;
                }
                None => {
                    // The run's command buffer holds the buffer until the GPU is done with it.
                    log::warn!(
                        "scaler: the making ahead of {kernel:?} did not end within {} s: the \
                         kernels left are made at their first transforms",
                        AHEAD_WAIT.as_secs()
                    );
                    break;
                }
            }
        }
        log::info!(
            "scaler device: kernels made ahead in {:.0} ms",
            clock.elapsed_since(started).as_secs_f64() * 1000.0
        );
        answerer.give(());
    };
    runtime
        .spawn(AHEAD_THREAD_NAME, Box::new(body))
        .map_err(|e| format!("no thread makes the kernels ahead: {e}"))?;
    Ok(answer)
}

/// A surface as the kernel reaches it: a buffer, its planes in it, and whether it is staged.
struct Bound<B> {
    buffer: B,
    planes: Vec<BoundPlane>,
    staged: bool,
}

/// Work the GPU did not finish by its deadline, with everything it may still read or write,
/// kept until it is done: the buffers go before the hold on the surfaces, as a buffer over a
/// surface's IOSurface retains it.
struct Kept<G: Gpu> {
    run: G::Run,
    _buffers: Vec<G::Buffer>,
    _hold: GuestHold,
}

impl<G: Gpu> Unfinished for Kept<G> {
    fn finished(&mut self) -> bool {
        G::finished(&self.run).is_some()
    }
}

/// `planes` at `base`, as the kernel reaches them; `None` past the address space.
fn at_base(planes: &[PlaneSpan], base: usize) -> Option<Vec<BoundPlane>> {
    planes.iter().map(|p| BoundPlane::at(p, base)).collect()
}

/// The bytes the planes of `of` span from its first byte, as a lend must hold them.
fn extent(of: &FormatOf) -> Option<usize> {
    of.planes
        .iter()
        .map(|p| p.extent()?.checked_add(p.offset))
        .try_fold(0usize, |most, end| Some(most.max(end?)))
}

/// The planes a staged destination of `of` is written in: each from a 4-byte boundary, its rows
/// a multiple of 4 bytes, one after another; and the bytes they take.
fn staged_planes(of: &FormatOf) -> Option<(Vec<BoundPlane>, usize)> {
    let mut at = 0usize;
    let mut planes = Vec::with_capacity(of.planes.len());
    for span in &of.planes {
        let pitch = span.row_bytes.checked_next_multiple_of(4)?;
        planes.push(BoundPlane {
            offset: at,
            pitch,
            row_bytes: span.row_bytes,
            rows: span.rows,
        });
        at = at.checked_add(pitch.checked_mul(span.rows)?)?;
    }
    Some((planes, at))
}

impl<G: Gpu> GpuAccelerator<G> {
    /// The source, bound where it is, or staged: copied in its lend.
    fn source(&self, job: &TransformJob<'_>) -> Result<Bound<G::Buffer>, IoReturn> {
        let src = &job.plan.source_format;
        if let Some(memory) = job.source.device_memory() {
            match self.gpu.memory(memory, false) {
                Ok(Some((buffer, base))) => {
                    if !src.fits(memory.length()) {
                        return Err(io_return::BAD_ARGUMENT);
                    }
                    let planes = at_base(&src.planes, base).ok_or(io_return::UNSUPPORTED)?;
                    return Ok(Bound {
                        buffer,
                        planes,
                        staged: false,
                    });
                }
                Ok(None) => {}
                Err(why) => log::debug!("scaler: the source is staged: {why}"),
            }
        }
        let length = extent(src).ok_or(io_return::UNSUPPORTED)?;
        let mut buffer = self.gpu.shared(length).map_err(|why| {
            log::warn!("scaler: no staging buffer of {length} bytes: {why}");
            io_return::NO_MEMORY
        })?;
        let staging = G::contents(&mut buffer).ok_or(io_return::ERROR)?;
        let mut fits = false;
        let read = job.source.read(&mut |bytes| {
            fits = src.fits(bytes.len());
            if let (true, Some(to), Some(from)) =
                (fits, staging.get_mut(..length), bytes.get(..length))
            {
                to.copy_from_slice(from);
            }
        });
        if let Err(error) = read {
            log::warn!("scaler: the source surface cannot be read: {error}");
            return Err(io_return::NOT_READABLE);
        }
        if !fits {
            return Err(io_return::BAD_ARGUMENT);
        }
        Ok(Bound {
            buffer,
            planes: at_base(&src.planes, 0).ok_or(io_return::UNSUPPORTED)?,
            staged: true,
        })
    }

    /// The destination, bound where it is, or staged: written in a buffer of the GPU's own and
    /// copied back in its lend.
    fn destination(
        &self,
        job: &TransformJob<'_>,
        source_in_place: bool,
    ) -> Result<Bound<G::Buffer>, IoReturn> {
        let dst = &job.plan.destination_format;
        let same = |memory: &dyn DeviceMemory| {
            source_in_place
                && job
                    .source
                    .device_memory()
                    .is_some_and(|source| source.identity().key() == memory.identity().key())
        };
        if let Some(memory) = job.destination.device_memory().filter(|m| !same(*m)) {
            match self.gpu.memory(memory, true) {
                Ok(Some((buffer, base))) => {
                    if !dst.fits(memory.length()) {
                        return Err(io_return::BAD_ARGUMENT);
                    }
                    let planes = at_base(&dst.planes, base).ok_or(io_return::UNSUPPORTED)?;
                    if planes.iter().all(BoundPlane::word_aligned) {
                        return Ok(Bound {
                            buffer,
                            planes,
                            staged: false,
                        });
                    }
                    log::debug!("scaler: the destination's planes are not word-aligned: staged");
                }
                Ok(None) => {}
                Err(why) => log::debug!("scaler: the destination is staged: {why}"),
            }
        }
        let (planes, length) = staged_planes(dst).ok_or(io_return::UNSUPPORTED)?;
        let buffer = self.gpu.shared(length).map_err(|why| {
            log::warn!("scaler: no staging buffer of {length} bytes: {why}");
            io_return::NO_MEMORY
        })?;
        Ok(Bound {
            buffer,
            planes,
            staged: true,
        })
    }

    /// The transform's work for `plan` with its surfaces' planes bound as they are, from the last
    /// transform's where they are the same, as a guest's frames of one stream are: its passes'
    /// parameters, and its taps in a buffer of the GPU's.
    fn prepare(
        &mut self,
        plan: &Plan,
        source: &[BoundPlane],
        destination: &[BoundPlane],
    ) -> Result<(), IoReturn> {
        let same = self
            .prepared
            .as_ref()
            .is_some_and(|p| p.plan == *plan && p.source == source && p.destination == destination);
        if same {
            return Ok(());
        }
        self.prepared = None;
        let work = match params::work(plan, source, destination) {
            Ok(work) => work,
            Err(TooLarge) => {
                log::warn!("scaler: refused, as a surface lies past the kernel's 4 GiB");
                return Err(io_return::UNSUPPORTED);
            }
        };
        let table: Vec<u8> = params::taps_bytes(&work.taps).collect();
        let taps = self.gpu.table(&table).map_err(|why| {
            log::warn!("scaler: no buffer of {} bytes of taps: {why}", table.len());
            io_return::NO_MEMORY
        })?;
        if work.lines > self.lines.as_ref().map_or(0, |(_, bytes)| *bytes) {
            self.lines = None;
            let buffer = self.gpu.scratch(work.lines).map_err(|why| {
                log::warn!("scaler: no buffer of {} bytes of lines: {why}", work.lines);
                io_return::NO_MEMORY
            })?;
            self.lines = Some((buffer, work.lines));
        }
        self.prepared = Some(Prepared {
            plan: plan.clone(),
            source: source.to_vec(),
            destination: destination.to_vec(),
            passes: (work.passes.iter())
                .map(|(kernel, p, threads, group)| (*kernel, p.bytes(), *threads, *group))
                .collect(),
            taps,
        });
        Ok(())
    }

    /// Runs the kernel from `source` into `destination` and waits for it; `Err` with the
    /// transform's status where it fails. Past the deadline, what the GPU may still use is kept.
    fn run(
        &mut self,
        job: &TransformJob<'_>,
        source: Bound<G::Buffer>,
        destination: Bound<G::Buffer>,
    ) -> Result<Bound<G::Buffer>, IoReturn> {
        let _running = Running::start();
        self.prepare(&job.plan, &source.planes, &destination.planes)?;
        let prepared = self.prepared.as_ref().ok_or(io_return::ERROR)?;
        let lines = self
            .lines
            .as_ref()
            .map_or(&prepared.taps, |(buffer, _)| buffer);
        let dispatches: Vec<Dispatch<'_, G::Buffer>> = (prepared.passes.iter())
            .map(|(kernel, params, threads, group)| Dispatch {
                kernel: *kernel,
                source: &source.buffer,
                destination: &destination.buffer,
                taps: &prepared.taps,
                params,
                lines,
                threads: *threads,
                group: *group,
            })
            .collect();
        // The GPU rings the bell once it is done, which ends the wait at once; the wait asks
        // again at least every `ASK_AGAIN` all the same.
        let bell = Arc::new(Bell::new());
        let ring = Arc::clone(&bell);
        let run = self
            .gpu
            .submit(&dispatches, Box::new(move || ring.ring()))
            .map_err(|why| {
                log::warn!("scaler: {} refused the transform: {why}", self.gpu.name());
                io_return::ERROR
            })?;
        drop(dispatches);
        let deadline = device_wait::deadline(&*self.runtime);
        let finished = loop {
            let ask = self.runtime.now().saturating_add(ASK_AGAIN).min(deadline);
            if let Some(done) = bell.wait_until(&*self.runtime, ask, || G::finished(&run)) {
                break Some(done);
            }
            if self.runtime.now() >= deadline {
                break G::finished(&run);
            }
        };
        match finished {
            Some(Ok(())) => {
                if let Some(time) = G::gpu_time(&run) {
                    #[cfg(test)]
                    GPU_TIMES.lock().push(time);
                    log::debug!(
                        "scaler: {} {}x{} to {} {}x{}: {:.3} ms on the GPU",
                        job.plan.source_format.format.fourcc(),
                        job.plan.source_format.width,
                        job.plan.source_format.height,
                        job.plan.destination_format.format.fourcc(),
                        job.plan.destination_format.width,
                        job.plan.destination_format.height,
                        time.as_secs_f64() * 1000.0
                    );
                }
                Ok(destination)
            }
            Some(Err(why)) => {
                log::warn!("scaler: the transform failed on {}: {why}", self.gpu.name());
                Err(io_return::ERROR)
            }
            None => {
                self.stuck = true;
                let mut buffers = vec![source.buffer, destination.buffer];
                buffers.extend(self.prepared.take().map(|p| p.taps));
                buffers.extend(self.lines.take().map(|(buffer, _)| buffer));
                let kept = Kept::<G> {
                    run,
                    _buffers: buffers,
                    _hold: job.hold.clone(),
                };
                Lingering::process(&self.runtime).keep(
                    format!("a scaler transform on {}", self.gpu.name()),
                    Box::new(kept),
                );
                Err(io_return::ERROR)
            }
        }
    }
}

/// Copies what the reference writes of a destination of `of` into `window` from `staged`, laid
/// out as `planes`, into the destination's `bytes`. Each run lies within both, as the staged
/// planes hold every row and byte of the format's and the caller checked that `bytes` holds the
/// destination's planes; one that would not is skipped.
fn copy_back(
    of: &FormatOf,
    window: Window,
    planes: &[BoundPlane],
    staged: &[u8],
    bytes: &mut [u8],
) {
    let at = |offset: usize, pitch: usize, row: usize, start: usize, length: usize| {
        let first = row
            .checked_mul(pitch)?
            .checked_add(offset)?
            .checked_add(start)?;
        Some(first..first.checked_add(length)?)
    };
    for run in covered(of.format, window) {
        let (Some(from), Some(to)) = (planes.get(run.plane), of.planes.get(run.plane)) else {
            continue;
        };
        let source = at(from.offset, from.pitch, run.row, run.start, run.length);
        let target = at(to.offset, to.bytes_per_row, run.row, run.start, run.length);
        if let (Some(from), Some(to)) = (
            source.and_then(|r| staged.get(r)),
            target.and_then(|r| bytes.get_mut(r)),
        ) {
            to.copy_from_slice(from);
        }
    }
}

/// The status of a transform whose source does not hold its layout: kIOReturnNotWritable where
/// the destination cannot be lent for writing, kIOReturnBadArgument where it can, as
/// [`TransformJob::with_bytes`] answers.
fn too_small(job: &TransformJob<'_>) -> IoReturn {
    match job.destination.write(&mut |_| {}) {
        Ok(()) | Err(SurfaceError::Layout) => io_return::BAD_ARGUMENT,
        Err(error) => {
            log::warn!("scaler: the destination surface cannot be written: {error}");
            io_return::NOT_WRITABLE
        }
    }
}

impl<G: Gpu> Accelerator for GpuAccelerator<G> {
    fn capabilities(&self) -> Capabilities {
        capabilities(self.smallest)
    }

    fn transform(&mut self, job: &mut TransformJob<'_>) -> IoReturn {
        if self.stuck {
            log::warn!(
                "scaler: refused, as {} did not finish an earlier transform",
                self.gpu.name()
            );
            return io_return::ERROR;
        }
        let plan = &job.plan;
        let sides = |of: &FormatOf| {
            let within = |side: u32| (self.smallest..=LARGEST_SIDE).contains(&side);
            within(of.width) && within(of.height)
        };
        if !sides(&plan.source_format) || !sides(&plan.destination_format) {
            return io_return::UNSUPPORTED;
        }
        let source = match self.source(job) {
            Ok(source) => source,
            Err(io_return::BAD_ARGUMENT) => return too_small(job),
            Err(status) => return status,
        };
        let destination = match self.destination(job, !source.staged) {
            Ok(destination) => destination,
            Err(status) => return status,
        };
        let staged = destination.staged;
        let ran = self.run(job, source, destination);
        if !staged {
            // Written in place, by the GPU, as far as it ran.
            job.destination.device_wrote();
            return ran.err().unwrap_or(io_return::SUCCESS);
        }
        // Staged: the destination's lend says whether it can be written, then what the kernel
        // wrote is copied into it.
        let mut output = ran;
        let dst = &job.plan.destination_format;
        let d = job.plan.destination_rect;
        let window = Window {
            x: d.x.to_usize(),
            y: d.y.to_usize(),
            width: d.width.to_usize(),
            height: d.height.to_usize(),
        };
        let mut status = io_return::SUCCESS;
        let written = job.destination.write(&mut |bytes| {
            status = if !dst.fits(bytes.len()) {
                io_return::BAD_ARGUMENT
            } else {
                match &mut output {
                    Ok(bound) => match G::contents(&mut bound.buffer) {
                        Some(staged) => {
                            copy_back(dst, window, &bound.planes, staged, bytes);
                            io_return::SUCCESS
                        }
                        None => io_return::ERROR,
                    },
                    Err(status) => *status,
                }
            };
        });
        drop(output);
        match written {
            Ok(()) => status,
            Err(SurfaceError::Layout) => io_return::BAD_ARGUMENT,
            Err(error) => {
                log::warn!("scaler: the destination surface cannot be written: {error}");
                io_return::NOT_WRITABLE
            }
        }
    }
}

#[cfg(test)]
mod tests;
