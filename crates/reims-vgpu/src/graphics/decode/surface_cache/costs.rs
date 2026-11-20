//! What a `SURFACE_DOWNLOAD` of an IOSurface's device cache costs on a real platform, and where
//! its time goes (ignored: a measurement, not a check). Run it alone, holding the host's GPU:
//!
//! ```sh
//! cargo nextest run -p reims-vgpu -E 'test(/download_costs/)' \
//!   --run-ignored only --no-capture
//! ```
//!
//! The guest's RAM is a memfd's shared mapping, as QEMU gives it (`surface_pages`'s costs), and
//! each surface is pages of it picked at random, a guest page a run, as the kext names a
//! surface's. Each block is named cached (`RGFX_SURFACE_PAGES_CACHED`), with its surface's layout
//! where the case gives one, as on a GPU of its own memory, and a `BUFFER_OVER_SURFACE_PAGES`
//! over it is its cache. Each round a command buffer of queue 1 names the surface's use as a
//! write (`RGFX_OP_SURFACE_USE`, `RGFX_SURFACE_USE_WRITES`) and does nothing else; after a pause
//! of `RGK_BENCH_GAP_US` microseconds (default 3000), in which the GPU runs it, the cache is
//! downloaded, timed on the host's clock, which the platform's GPU times are on
//! (`handler_cost::host_ns`):
//!
//! - `write`: the writing command buffer's decode, which, once a download has copied the surface
//!   since its last write, encodes and commits the write's early copy behind it, on the
//!   decoder's thread (`SurfaceCaches::early_copies`); a case's first write takes none;
//! - `record+commit`: the download's call, which records the copy and commits it;
//! - `to GPU start`: from the call's return to the copy's `GPUStartTime`, when the GPU began it
//!   (the rt platform's GPU times: `rt_platform/command.rs`);
//! - `GPU start to end`: from `GPUStartTime` to `GPUEndTime`: the GPU's run of the copy;
//! - `answer`: from `GPUEndTime` to the answer;
//! - `lock`: from the call to the answer, which a CPU lock of the surface waits for, past its
//!   ring's messages.
//!
//! Each line names how the cache reaches the pages (`view`: a buffer over the pages' view;
//! `runs`: the runtime's copy by the pages' runs) and the copies one download encodes (`parts`).
//! Then `RGK_BENCH_BURST` (default 8) surfaces of the case are written by one command buffer and
//! downloaded back to back, timed from the first call to the last answer, as a burst of locks.
//! Last, copies through the platform alone, with no cache: a `Private` buffer into a `Shared` one
//! (`down`) and back (`up`), one copy a command buffer, of each of the cases' sizes and of
//! `RGK_BENCH_RAW_MIB` MiB (default 256), the runtime's own rate between the GPU's memory and the
//! host's.
//!
//! `RGK_BENCH_ITERS` rounds a case (default 30), `RGK_BENCH_RAM_MIB` MiB of guest RAM (default
//! 768), `RGK_BENCH_SPREAD_MIB` the MiB from its start that a surface's pages are picked in
//! (default all of it: a booted guest's surfaces are pages from anywhere in its RAM), and
//! `RGK_BENCH_CASES` a comma list of the cases' names to run; `RGK_BENCH_TOUCH=0` leaves the RAM's
//! pages untouched until the platform maps them. Each case's line names the [`CHUNK`]s of the RAM
//! its pages lie in (`chunks`).

use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use reims_vgpu_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_SURFACE_USE, RGFX_SURFACE_USE_WRITES,
    RgfxCommandBuffer, RgfxOpHeader, RgfxOpSurfaceUse,
};
use reims_vgpu_graphics::wire::{
    RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_OBJECT_DESTROY,
    RGFX_CMD_QUEUE_CREATE, RGFX_CMD_SURFACE_PAGES, RGFX_CONTEXT_ALLOCATIONS_OFFSET,
    RGFX_CURSOR_SPAN, RGFX_STATUS_OK, RGFX_SURFACE_PAGES_CACHED, RgfxBufferOverSurfacePages,
    RgfxMsgHeader, RgfxObjectDestroy, RgfxQueueCreate, RgfxSurfacePages, RgfxSurfacePagesLayout,
    RgfxSurfacePlaneLayout,
};
use reims_vgpu_vulkan::core::memory::map::MemoryMap as CoreMemoryMap;
use reims_vgpu_vulkan::core::num::ToUsize as _;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{
    CommandBuffer as _, CommandQueue as _, CompletedInfo,
};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use zerocopy::IntoBytes;

use super::super::handler_cost::host_ns;
use super::super::surface_pages::costs::{RAM, XorShift, env_u64, guest_ram, surface_runs};
use super::super::surface_pages::{GUEST_PAGE, Lease};
use super::super::{GraphicsDecoder, no_raise};
use super::{CachePages, CopyTrace, ON, STORAGE_PRIVATE, STORAGE_SHARED, runs_at};
use crate::platform::ApiPlatform;

/// The first block a case's surfaces are named at, each next [`BLOCK_STEP`] on.
const BLOCK: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET + 0x4000;
const BLOCK_STEP: u64 = 0x4000;

const MIB: u64 = 1 << 20;

/// The pieces of the RAM a case's line counts its pages in (module docs).
const CHUNK: u64 = 64 * MIB;

/// The command queue every write is committed on.
const QUEUE: u64 = 1;

/// One surface of one plane: `rows` rows of `row_bytes` bytes of texels, `bytes_per_row` apart,
/// named with that layout where `laid`, else with none.
struct Case {
    name: &'static str,
    row_bytes: u64,
    bytes_per_row: u64,
    rows: u64,
    laid: bool,
}

/// The surfaces measured, `BGRA8` all but the first: a page; a 800x512 tile (1.6 MB, about the
/// walks' mean download), its rows on and off the 64 bytes CoreAnimation lays them at; and a
/// 3840x2160 layer (33 MB), with no layout, laid out, and 3830 texels wide in rows of 3840.
fn cases() -> Vec<Case> {
    let case = |name, row_bytes, bytes_per_row, rows, laid| Case {
        name,
        row_bytes,
        bytes_per_row,
        rows,
        laid,
    };
    vec![
        case("page", 4096, 4096, 1, false),
        case("tile", 3200, 3200, 512, true),
        case("tile-rows", 3188, 3200, 512, true),
        case("layer-whole", 15360, 15360, 2160, false),
        case("layer", 15360, 15360, 2160, true),
        case("layer-rows", 15320, 15360, 2160, true),
    ]
}

impl Case {
    /// The surface's bytes, whole pages.
    fn length(&self) -> u64 {
        (self.bytes_per_row * self.rows).next_multiple_of(GUEST_PAGE)
    }

    /// Its texels' bytes, which a download moves.
    fn texels(&self) -> u64 {
        if self.laid {
            self.row_bytes * self.rows
        } else {
            self.length()
        }
    }

    /// The layout its `SURFACE_PAGES` carries.
    fn layout(&self) -> Option<RgfxSurfacePagesLayout> {
        self.laid.then(|| {
            let mut layout = RgfxSurfacePagesLayout {
                plane_count: 1,
                ..RgfxSurfacePagesLayout::default()
            };
            layout.planes[0] = RgfxSurfacePlaneLayout {
                offset: 0,
                bytes_per_row: self.bytes_per_row,
                row_bytes: self.row_bytes,
                rows: self.rows,
            };
            layout
        })
    }
}

/// One download's times on the host's clock, in nanoseconds: its call, the call's return, the
/// answer, and its copy's GPU times where it made one.
struct Timed {
    called: u64,
    returned: u64,
    answered: u64,
    gpu: Option<(u64, u64)>,
}

/// Microseconds from `from` to `to`, signed.
#[allow(clippy::cast_precision_loss)] // microseconds of a measurement
fn us(from: u64, to: u64) -> f64 {
    (to as f64 - from as f64) / 1e3
}

/// The p50, p90 and most of `us`, and their mean.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)] // percentiles of a measurement
fn spread(mut us: Vec<f64>) -> String {
    if us.is_empty() {
        return "none".to_owned();
    }
    us.sort_by(f64::total_cmp);
    let at = |q: f64| us[((us.len() - 1) as f64 * q).round() as usize];
    let mean = us.iter().sum::<f64>() / us.len() as f64;
    format!(
        "p50 {:>9.1} p90 {:>9.1} max {:>9.1} mean {:>9.1} us",
        at(0.5),
        at(0.9),
        at(1.0),
        mean
    )
}

/// The median of `us`.
fn median(mut us: Vec<f64>) -> f64 {
    if us.is_empty() {
        return f64::NAN;
    }
    us.sort_by(f64::total_cmp);
    us[us.len() / 2]
}

/// MB a second of `bytes` in `us` microseconds.
#[allow(clippy::cast_precision_loss)] // a rate of a measurement
fn rate(bytes: u64, us: f64) -> f64 {
    bytes as f64 / us
}

fn header(r#type: u32, size: usize) -> RgfxMsgHeader {
    RgfxMsgHeader {
        r#type,
        size: u32::try_from(size).unwrap_or(0),
        seqno: 0,
    }
}

/// A decoder over guest RAM on the platform's device, with command queue [`QUEUE`].
struct Bench<P: ApiPlatform> {
    dec: GraphicsDecoder<P>,
    device: P::Device,
    /// The bytes of the RAM, from its start, that a surface's pages are picked in.
    spread: u64,
    page: u64,
    rng: XorShift,
    /// The [`CHUNK`]s of the RAM the last surface named lies in.
    chunks: usize,
    /// The next object id for a buffer over a block, and for a command buffer.
    obj: u64,
    cb: u64,
}

impl<P: ApiPlatform> Bench<P> {
    /// The block at `block` named cached as `case`'s surface, over fresh pages, with a buffer over
    /// it (its cache); the buffer's object id.
    fn name(&mut self, block: u64, case: &Case) -> u64 {
        let length = case.length();
        let runs = surface_runs(&mut self.rng, self.spread, length, self.page);
        let chunks: std::collections::BTreeSet<u64> =
            runs.iter().map(|r| (r.address - RAM) / CHUNK).collect();
        self.chunks = chunks.len();
        let pages = RgfxSurfacePages {
            header: header(RGFX_CMD_SURFACE_PAGES, 0),
            context: 0,
            run_count: u32::try_from(runs.len()).unwrap_or(0),
            block_offset: block,
            block_size: 0x2000,
            runs_offset: block + 0x1000,
            length,
        };
        let status = (self.dec).surface_pages_laid_out(
            &self.device,
            &pages,
            &runs,
            RGFX_SURFACE_PAGES_CACHED,
            case.layout(),
        );
        assert_eq!(status, RGFX_STATUS_OK, "{}: its SURFACE_PAGES", case.name);
        self.obj += 1;
        let over = RgfxBufferOverSurfacePages {
            header: header(RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, 0),
            obj_id: self.obj,
            block_offset: block,
            block_size: 0x1000,
            length,
            options: 0,
            reserved_34: 0,
        };
        let made = self.dec.decode(&self.device, over.as_bytes());
        assert_eq!(made.ok(), Some(Ok(())), "{}: the buffer over it", case.name);
        self.obj
    }

    /// The block at `block` and buffer `obj` let go of, nothing written since its last copy.
    fn release(&mut self, block: u64, obj: u64) {
        let destroy = RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY, 0),
            obj_id: obj,
        };
        let gone = self.dec.decode(&self.device, destroy.as_bytes());
        assert_eq!(gone.ok(), Some(Ok(())));
        (self.dec).block_released(&self.device, block, 0, Box::new(|| ()), no_raise());
    }

    /// A command buffer of [`QUEUE`] that names a write of each block of `blocks`, at the CPU's
    /// generation 0, and does nothing else: the microseconds its decode took.
    fn write(&mut self, blocks: &[u64]) -> f64 {
        let mut ops = Vec::new();
        for &block_offset in blocks {
            let body = RgfxOpSurfaceUse {
                block_offset,
                cpu_gen: 0,
                flags: RGFX_SURFACE_USE_WRITES,
                reserved_14: 0,
            };
            let op = RgfxOpHeader {
                tag: RGFX_OP_SURFACE_USE,
                flags: 0,
                len: u32::try_from(8 + size_of::<RgfxOpSurfaceUse>()).unwrap_or(0),
            };
            ops.extend_from_slice(op.as_bytes());
            ops.extend_from_slice(body.as_bytes());
        }
        self.cb += 1;
        let m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: u32::try_from(RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len())
                    .unwrap_or(0),
                seqno: self.cb,
            },
            queue_obj_id: QUEUE,
            cb_obj_id: self.cb,
            op_bytes: ops.len() as u64,
        };
        let mut message = m.as_bytes().to_vec();
        message.extend_from_slice(&ops);
        let called = host_ns();
        let committed = self.dec.decode(&self.device, &message);
        let took = us(called, host_ns());
        assert_eq!(committed.ok(), Some(Ok(())), "the writing command buffer");
        took
    }

    /// Asks the download of the block at `block`: what the answer sends, and the trace of its
    /// copy. Timed from just before the call.
    fn ask(&self, block: u64, tx: mpsc::Sender<u64>) -> (u64, u64, Arc<CopyTrace>) {
        self.dec.settle_surface(block);
        let trace = Arc::new(CopyTrace::default());
        let called = host_ns();
        self.dec.surface_caches.download(
            &self.device,
            block,
            Arc::clone(&self.dec.region),
            Box::new(|| {}),
            Arc::clone(&trace),
            Box::new(move || {
                let _ = tx.send(host_ns());
            }),
            Box::new(|raise| panic!("the platform raised at a download: {raise:?}")),
        );
        (called, host_ns(), trace)
    }

    /// The download of the block at `block`, waited for.
    fn download(&self, block: u64) -> Timed {
        let (tx, rx) = mpsc::channel();
        let (called, returned, trace) = self.ask(block, tx);
        let answered = (rx.recv_timeout(Duration::from_secs(20))).expect("a download answered");
        Timed {
            called,
            returned,
            answered,
            gpu: gpu_of(&trace),
        }
    }

    /// How the block at `block`'s cache reaches its pages, and the copies a download encodes.
    fn route(&self, block: u64) -> (&'static str, usize) {
        let caches = self.dec.caches();
        let Some(cache) = caches.by_block.get(&block) else {
            return ("none", 0);
        };
        let whole = 0..cache.length;
        let spans = (cache.texels.as_deref()).unwrap_or(std::slice::from_ref(&whole));
        match &cache.pages {
            Some(CachePages::View(_)) => {
                let parts = cache.moves(spans.iter().map(|span| (span.clone(), span.start)));
                ("view", parts.len())
            }
            Some(CachePages::Runs { runs, .. }) => {
                let parts = match &cache.repitched {
                    Some(laid) => laid.runs(runs).len(),
                    None => runs_at(runs, spans).len(),
                };
                ("runs", parts)
            }
            Some(CachePages::Copied(_)) => ("staging", 1),
            None => ("no pages", 0),
        }
    }
}

/// The GPU times of the copy a download's trace waited for.
fn gpu_of(trace: &CopyTrace) -> Option<(u64, u64)> {
    let traced = trace.traced.lock();
    traced
        .landings
        .first()
        .and_then(|landing| landing.times.lock().gpu)
}

/// The lines of a series of downloads of `bytes` each.
fn lines(label: &str, what: &str, bytes: u64, timed: &[Timed]) {
    let pick = |f: &dyn Fn(&Timed) -> Option<f64>| timed.iter().filter_map(f).collect::<Vec<_>>();
    let commit = pick(&|t| Some(us(t.called, t.returned)));
    let submit = pick(&|t| t.gpu.map(|(start, _)| us(t.returned, start)));
    let seen = pick(&|t| t.gpu.map(|(start, end)| us(start, end)));
    let answer = pick(&|t| t.gpu.map(|(_, end)| us(end, t.answered)));
    let lock = pick(&|t| Some(us(t.called, t.answered)));
    let (seen_p50, lock_p50) = (median(seen.clone()), median(lock.clone()));
    eprintln!(
        "{label}: {what}: {bytes} bytes, n {}: {:.0} MB/s GPU start to end, {:.0} MB/s lock",
        timed.len(),
        rate(bytes, seen_p50),
        rate(bytes, lock_p50)
    );
    for (step, values) in [
        ("record+commit", commit),
        ("to GPU start", submit),
        ("GPU start to end", seen),
        ("answer", answer),
        ("lock", lock),
    ] {
        eprintln!("{label}: {what}:   {step:<16} {}", spread(values));
    }
}

/// Measures every case on `P`'s device, printing its lines under `label`.
fn measure<P: ApiPlatform>(label: &str) {
    let platform = P::process();
    let device = platform.metal_device(None).expect("a GPU device");
    let page = platform.vm().page_size().max(GUEST_PAGE);
    let ram_bytes = env_u64("RGK_BENCH_RAM_MIB", 768) * MIB;
    let iters = env_u64("RGK_BENCH_ITERS", 30);
    let burst = env_u64("RGK_BENCH_BURST", 8);
    let gap = Duration::from_micros(env_u64("RGK_BENCH_GAP_US", 3000));
    let only = std::env::var("RGK_BENCH_CASES").ok();
    let picked_in = env_u64("RGK_BENCH_SPREAD_MIB", ram_bytes / MIB).min(ram_bytes / MIB) * MIB;
    let ram = guest_ram(ram_bytes, env_u64("RGK_BENCH_TOUCH", 1) == 1);
    let lease: Lease = Arc::from(platform.guest_memory(&ram.map));
    let core = CoreMemoryMap::new(Arc::clone(&platform), &ram.map.to_core()).expect("a memory map");
    let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
    let region = Arc::new(crate::platform::test_region(&*platform, size.to_usize()));
    let mut dec = GraphicsDecoder::<P>::new(Arc::clone(&platform), region);
    dec.set_guest_memory(Arc::clone(&core), Arc::clone(&lease));
    let queue = RgfxQueueCreate {
        header: header(RGFX_CMD_QUEUE_CREATE, size_of::<RgfxQueueCreate>()),
        obj_id: QUEUE,
        ..RgfxQueueCreate::default()
    };
    assert_eq!(dec.decode(&device, queue.as_bytes()).ok(), Some(Ok(())));
    let mut b = Bench {
        dec,
        device,
        spread: picked_in,
        page,
        rng: XorShift(0x9e37_79b9_7f4a_7c15),
        chunks: 0,
        obj: 100,
        cb: 0x1_0000,
    };
    eprintln!(
        "{label}: {} MiB of guest RAM, pages picked in its first {} MiB, {iters} downloads a \
         case, {} us after each write",
        ram_bytes >> 20,
        picked_in >> 20,
        gap.as_micros()
    );
    for case in cases() {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|n| n == case.name))
        {
            continue;
        }
        // One surface, written and downloaded `iters` times, after a first write that fills it.
        let obj = b.name(BLOCK, &case);
        let wrote = b.write(&[BLOCK]);
        let first = b.download(BLOCK);
        let (route, parts) = b.route(BLOCK);
        let what = format!("{} {route} parts {parts} chunks {}", case.name, b.chunks);
        eprintln!(
            "{label}: {what} first:   {:<15} {}",
            "write",
            spread(vec![wrote])
        );
        lines(label, &format!("{what} first"), case.texels(), &[first]);
        let (mut timed, mut writes) = (Vec::new(), Vec::new());
        for _ in 0..iters {
            writes.push(b.write(&[BLOCK]));
            std::thread::sleep(gap);
            timed.push(b.download(BLOCK));
        }
        eprintln!("{label}: {what}:   {:<15} {}", "write", spread(writes));
        lines(label, &what, case.texels(), &timed);
        b.release(BLOCK, obj);
        // `burst` surfaces written by one command buffer, then downloaded back to back.
        if burst > 1 && case.length() * burst <= picked_in / 2 {
            let blocks: Vec<u64> = (0..burst).map(|i| BLOCK + i * BLOCK_STEP).collect();
            let objs: Vec<u64> = blocks.iter().map(|&block| b.name(block, &case)).collect();
            b.write(&blocks);
            for &block in &blocks {
                b.download(block);
            }
            let (mut spans, mut each) = (Vec::new(), Vec::new());
            for _ in 0..iters.div_ceil(4) {
                b.write(&blocks);
                std::thread::sleep(gap);
                let (tx, rx) = mpsc::channel();
                let asked: Vec<_> = blocks
                    .iter()
                    .map(|&block| b.ask(block, tx.clone()))
                    .collect();
                drop(tx);
                let answers: Vec<u64> = (0..blocks.len())
                    .map(|_| (rx.recv_timeout(Duration::from_secs(20))).expect("an answer"))
                    .collect();
                let last = answers.iter().copied().max().unwrap_or(0);
                spans.push(us(asked[0].0, last));
                for (_, _, trace) in &asked {
                    if let Some((start, end)) = gpu_of(trace) {
                        each.push(us(start, end));
                    }
                }
            }
            let bytes = case.texels() * burst;
            eprintln!(
                "{label}: {} burst of {burst}: {:.0} MB/s first call to last answer; \
                 first call to last answer {}; each copy's GPU start to end {}",
                case.name,
                rate(bytes, median(spans.clone())),
                spread(spans),
                spread(each)
            );
            for (&block, &obj) in blocks.iter().zip(&objs) {
                b.release(block, obj);
            }
        }
    }
    let raw_mib = env_u64("RGK_BENCH_RAW_MIB", 256);
    let mut sizes: Vec<u64> = cases().iter().map(Case::length).collect();
    sizes.push(raw_mib * MIB);
    sizes.dedup();
    raw::<P>(label, &b.device, &sizes, iters.min(10));
    b.dec.views_of_surfaces().reset();
    drop(b);
    drop(core);
    drop(lease);
    drop(ram);
}

/// Copies through the platform alone (module docs): of each of `sizes`, `rounds` times each way.
fn raw<P: ApiPlatform>(label: &str, device: &P::Device, sizes: &[u64], rounds: u64) {
    let queue = device.new_command_queue().expect("a queue");
    for &bytes in sizes {
        let private = (device.new_buffer_of_length(ON, bytes, STORAGE_PRIVATE))
            .expect("no raise")
            .expect("a Private buffer");
        let shared = (device.new_buffer_of_length(ON, bytes, STORAGE_SHARED))
            .expect("no raise")
            .expect("a Shared buffer");
        for down in [true, false] {
            let (source, destination) = if down {
                (&private, &shared)
            } else {
                (&shared, &private)
            };
            let mut timed = Vec::new();
            for round in 0..=rounds {
                let cb = queue
                    .command_buffer(ON)
                    .expect("no raise")
                    .expect("a command buffer");
                let blit = cb
                    .blit_encoder(ON)
                    .expect("no raise")
                    .expect("a blit encoder");
                (blit.encode(
                    ON,
                    BlitCmd::CopyBufferToBuffer {
                        source_buffer: Some(source),
                        source_offset: 0,
                        destination_buffer: Some(destination),
                        destination_offset: 0,
                        size: bytes,
                    },
                ))
                .expect("no raise");
                blit.encode(ON, BlitCmd::EndEncoding).expect("no raise");
                let (tx, rx) = mpsc::channel::<(u64, Option<(u64, u64)>)>();
                (cb.add_completion_seen_handler(
                    ON,
                    Box::new(move |info: CompletedInfo| {
                        let _ = tx.send((host_ns(), info.gpu_times_ns));
                    }),
                ))
                .expect("no raise");
                let called = host_ns();
                cb.commit(ON).expect("no raise");
                let returned = host_ns();
                let (answered, gpu) =
                    (rx.recv_timeout(Duration::from_secs(20))).expect("a copy done");
                // The first round warms the buffers' memory and the queue up.
                if round > 0 {
                    timed.push(Timed {
                        called,
                        returned,
                        answered,
                        gpu,
                    });
                }
            }
            let way = if down { "down" } else { "up" };
            lines(label, &format!("raw {way} {}", bytes), bytes, &timed);
        }
    }
}

/// The rt platform's downloads (module docs).
#[test]
#[ignore = "a measurement: run it alone, holding the host's GPU"]
fn download_costs_vulkan() {
    measure::<reims_vgpu_vulkan::RtPlatform>("rt");
}
