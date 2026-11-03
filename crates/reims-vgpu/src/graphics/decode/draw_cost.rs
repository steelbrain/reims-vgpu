//! What a draw of the bench's `draws.10k` shape costs the thread that decodes it, through the
//! decoder as a queue's ring hands it a `COMMAND_BUFFER`, against the same commands sent through
//! the platform's core traits and, on the Vulkan platform, through the runtime alone: a
//! measurement, run by hand in a release build, holding the host's GPU
//! (`scripts/internal/gpu-locks.sh -- cargo nextest run -p reims-vgpu --release
//! --no-default-features --features reims-vgpu/vulkan --run-ignored only -E 'test(draw_cost)'
//! --no-capture`).
//!
//! A command buffer is one render pass of draws of 3 vertices ([`Shape`]), each after a
//! `setVertexBuffer:` at index 0 of an upload chunk at its next 256-byte offset, as the bundle
//! sends `setVertexBytes:` (a 64 KiB `BUFFER_OVER_HOST_MEMORY` chunk of the region, 256 draws a
//! chunk). Three command buffers go back to back and are then waited for, as `draws.10k` commits
//! them. Each path's figures are the thread's CPU time, instructions and cycles a draw, from the
//! first command buffer's begin to the last one's commit: the decoder's include the message's
//! take, its walk and its commit with the decoder's handlers; the gaps between the paths are the
//! decoder's share and the platform's. Instructions are the figure a busy host leaves alone
//! (measured on the M3 Max at a load of 50: within 0.2% run to run, where CPU time moved 15%).
//!
//! The environment picks what runs: `DRAW_COST_SHAPE` the shape (`10k`, the default, or `many`,
//! 64 command buffers of 16 draws, whose figures show a command buffer's own cost),
//! `DRAW_COST_RUNS` the runs (9), `DRAW_COST_ONLY` one path, for a profile of it (`decoder`,
//! `platform` or `runtime`), and `DRAW_COST_SECONDS` a time to keep running that path for, so a
//! sampling profiler can attach.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use reims_graphics::ops::render::{
    RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_SET_FRAGMENT_BUFFER,
    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RGFX_OP_RENDER_SET_VERTEX_BUFFER,
    RgfxOpDrawPrimitives, RgfxOpSetFragmentBuffer, RgfxOpSetRenderPipelineState,
    RgfxOpSetVertexBuffer,
};
use reims_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_END,
    RGFX_PASS_FLAG_RT_SIZE, RgfxCommandBuffer, RgfxOpHeader, RgfxRenderPassColorAttachment,
    RgfxRenderPassDesc,
};
use reims_graphics::wire::{
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_QUEUE_CREATE,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RgfxBufferOverHostMemory, RgfxMsgHeader,
    RgfxQueueCreate,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_core::platform::resource::Library as _;
use reims_vgpu_core::platform::values::{ColorAttachmentSet, RenderPipelineDesc, TextureDesc};
use reims_vgpu_core::raise::OnFault;
use zerocopy::IntoBytes;

use super::{GraphicsDecoder, ObjHandle};
use crate::platform::ApiPlatform;

/// How a repetition's work is cut: command buffers, committed back to back and then waited for,
/// each of one render pass of `draws` draws.
#[derive(Debug, Clone, Copy)]
struct Shape {
    cbs: usize,
    draws: usize,
}

impl Shape {
    /// `draws.10k`'s: 3 command buffers of 10,000 draws.
    const TEN_K: Shape = Shape {
        cbs: 3,
        draws: 10_000,
    };
    /// Many short command buffers, as a guest's frame of 60 to 100 is cut: 64 of 16 draws.
    const MANY: Shape = Shape { cbs: 64, draws: 16 };

    /// The upload chunks a command buffer's draws take.
    fn chunks_a_cb(self) -> u64 {
        (self.draws as u64 * ALIGN).div_ceil(CHUNK)
    }

    /// The chunks the region holds: a repetition's command buffers each take their own.
    fn chunks(self) -> u64 {
        self.chunks_a_cb() * self.cbs as u64
    }

    /// The shape `DRAW_COST_SHAPE` names: `many` or, by default, `10k`.
    fn chosen() -> Shape {
        match std::env::var("DRAW_COST_SHAPE").as_deref() {
            Ok("many") => Shape::MANY,
            _ => Shape::TEN_K,
        }
    }
}
/// Repetitions a run times, after [`WARM`] repetitions.
const REPS: usize = 6;
/// Repetitions run before the timed ones.
const WARM: usize = 2;
/// An upload chunk's bytes, and the alignment of each draw's bytes in it (the bundle's
/// `kUploadChunkSize` and `kUploadAlign`).
const CHUNK: u64 = 0x1_0000;
const ALIGN: u64 = 256;

/// The object ids: the queue, the target, the pipeline, the per-draw constant buffers (vertex 1
/// and fragment 0), then the chunks from [`CHUNK_ID`].
const QUEUE_ID: u64 = 1;
const TARGET_ID: u64 = 2;
const PIPELINE_ID: u64 = 3;
const CONSTANTS_ID: u64 = 4;
const CHUNK_ID: u64 = 0x100;

/// `MTLPixelFormatBGRA8Unorm`; `MTLTextureUsageShaderRead | RenderTarget`; `Clear`, `Store`;
/// `MTLPrimitiveTypeTriangle`.
const BGRA8: u64 = 80;
const TARGET_USAGE: u64 = 1 | 4;
const CLEAR: u64 = 2;
const STORE: u64 = 1;
const TRIANGLE: u64 = 3;
/// The target's side.
const SIDE: u64 = 512;

/// This thread's CPU time in nanoseconds, the instructions it retired and its cycles, each as a
/// count that only grows; the counters are 0 where the host gives none.
fn counts() -> Counts {
    let mut at = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `at` is a timespec, which `clock_gettime` writes.
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut at) },
        0
    );
    let ns = u64::try_from(at.tv_sec).unwrap() * 1_000_000_000 + u64::try_from(at.tv_nsec).unwrap();
    let [instructions, cycles] = counters::now();
    Counts([ns, instructions, cycles])
}

/// What [`counts`] answers, and the differences of two answers.
#[derive(Debug, Clone, Copy, Default)]
struct Counts([u64; 3]);

impl std::ops::Sub for Counts {
    type Output = Counts;
    fn sub(self, before: Counts) -> Counts {
        Counts([0, 1, 2].map(|i| self.0[i] - before.0[i]))
    }
}

impl std::ops::AddAssign for Counts {
    fn add_assign(&mut self, more: Counts) {
        for (s, m) in self.0.iter_mut().zip(more.0) {
            *s += m;
        }
    }
}

/// The names of [`Counts`]' figures.
const FIGURES: [&str; 3] = ["CPU ns", "instructions", "cycles"];

/// This thread's instructions and cycles: on macOS from the kernel's own per-thread counters
/// (`thread_selfcounts`, which a process reads of its threads unprivileged).
#[cfg(target_os = "macos")]
mod counters {
    unsafe extern "C" {
        fn thread_selfcounts(kind: i32, buf: *mut u64, nbytes: usize) -> i32;
    }

    pub(super) fn now() -> [u64; 2] {
        let mut out = [0u64; 2];
        // SAFETY: kind 1 writes two `u64`s, the thread's instructions and cycles, into `out`.
        let r = unsafe { thread_selfcounts(1, out.as_mut_ptr(), size_of_val(&out)) };
        if r != 0 { [0, 0] } else { out }
    }
}

/// This thread's instructions and cycles: on Linux from two perf events of the thread's own,
/// counting user space, opened at its first read.
#[cfg(target_os = "linux")]
mod counters {
    use std::cell::Cell;

    thread_local! {
        static FDS: Cell<Option<[i32; 2]>> = const { Cell::new(None) };
    }

    /// `struct perf_event_attr` as far as the fields set here, at the size of its first version.
    #[repr(C)]
    #[derive(Default)]
    struct Attr {
        kind: u32,
        size: u32,
        config: u64,
        sample: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup: u32,
        bp_type: u32,
        config1: u64,
    }

    fn open(config: u64) -> i32 {
        let attr = Attr {
            kind: 0, // PERF_TYPE_HARDWARE
            size: size_of::<Attr>() as u32,
            config,
            // exclude_kernel (bit 5) and exclude_hv (bit 6).
            flags: (1 << 5) | (1 << 6),
            ..Attr::default()
        };
        // SAFETY: `attr` is a perf_event_attr of the size it names; pid 0 and cpu -1 count this
        // thread on any CPU.
        unsafe { libc::syscall(libc::SYS_perf_event_open, &attr, 0, -1, -1, 0) as i32 }
    }

    pub(super) fn now() -> [u64; 2] {
        let fds = FDS.with(|f| match f.get() {
            Some(fds) => fds,
            None => {
                let fds = [open(1), open(0)]; // PERF_COUNT_HW_INSTRUCTIONS, _CPU_CYCLES
                f.set(Some(fds));
                fds
            }
        });
        fds.map(|fd| {
            let mut v = 0u64;
            // SAFETY: a counting event's read writes one `u64`.
            let n = unsafe { libc::read(fd, (&raw mut v).cast(), 8) };
            if fd < 0 || n != 8 { 0 } else { v }
        })
    }
}

/// One op: its header, then its body.
fn op(tag: u16, body: &[u8]) -> Vec<u8> {
    let mut v = RgfxOpHeader {
        tag,
        flags: 0,
        len: u32::try_from(8 + body.len()).unwrap(),
    }
    .as_bytes()
    .to_vec();
    v.extend_from_slice(body);
    v
}

/// The chunk and offset of draw `draw` of command buffer `cb` of a repetition of `shape`.
fn upload_at(shape: Shape, cb: usize, draw: usize) -> (u64, u64) {
    let per_cb = shape.chunks_a_cb();
    let at = draw as u64 * ALIGN;
    (CHUNK_ID + cb as u64 * per_cb + at / CHUNK, at % CHUNK)
}

/// The `COMMAND_BUFFER` message of command buffer `cb` of a repetition of `shape`, as the bundle
/// sends `draws.10k`'s.
fn command_buffer(shape: Shape, cb: usize, cb_obj_id: u64) -> Vec<u8> {
    let pass = RgfxRenderPassDesc {
        color_count: 1,
        flags: RGFX_PASS_FLAG_RT_SIZE,
        render_target_width: SIDE as u32,
        render_target_height: SIDE as u32,
        ..RgfxRenderPassDesc::default()
    };
    let color = RgfxRenderPassColorAttachment {
        texture_obj_id: TARGET_ID,
        load_action: CLEAR as u32,
        store_action: STORE as u32,
        clear_color_bits: [0, 0, 0, 1.0f64.to_bits()],
        ..RgfxRenderPassColorAttachment::default()
    };
    let mut begin = pass.as_bytes().to_vec();
    begin.extend_from_slice(color.as_bytes());
    let mut ops = op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin);
    let state = RgfxOpSetRenderPipelineState {
        pipeline_obj_id: PIPELINE_ID,
    };
    ops.extend(op(
        RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
        state.as_bytes(),
    ));
    let constants = RgfxOpSetVertexBuffer {
        buffer_obj_id: CONSTANTS_ID,
        offset: 0,
        index: 1,
    };
    ops.extend(op(RGFX_OP_RENDER_SET_VERTEX_BUFFER, constants.as_bytes()));
    let fragment = RgfxOpSetFragmentBuffer {
        buffer_obj_id: CONSTANTS_ID,
        offset: 0,
        index: 0,
    };
    ops.extend(op(RGFX_OP_RENDER_SET_FRAGMENT_BUFFER, fragment.as_bytes()));
    for draw in 0..shape.draws {
        let (chunk, offset) = upload_at(shape, cb, draw);
        let bind = RgfxOpSetVertexBuffer {
            buffer_obj_id: chunk,
            offset,
            index: 0,
        };
        ops.extend(op(RGFX_OP_RENDER_SET_VERTEX_BUFFER, bind.as_bytes()));
        let drawn = RgfxOpDrawPrimitives {
            primitive_type: TRIANGLE,
            vertex_start: 0,
            vertex_count: 3,
        };
        ops.extend(op(RGFX_OP_RENDER_DRAW_PRIMITIVES, drawn.as_bytes()));
    }
    ops.extend(op(RGFX_OP_ENCODER_END, &[]));
    let m = RgfxCommandBuffer {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_COMMAND_BUFFER,
            size: u32::try_from(RGFX_COMMAND_BUFFER_OPS_OFFSET as usize + ops.len()).unwrap(),
            seqno: cb_obj_id,
        },
        queue_obj_id: QUEUE_ID,
        cb_obj_id,
        op_bytes: ops.len() as u64,
    };
    let mut message = m.as_bytes().to_vec();
    message.extend(ops);
    message
}

/// A function of the suite's fixture `name`.
fn fixture<P: Platform>(d: &P::Device, name: &str) -> P::Function {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/suite/fixtures/shaders")
        .join(format!("{name}.metallib"));
    let library = d
        .new_library(&std::fs::read(path).unwrap())
        .unwrap()
        .unwrap()
        .unwrap();
    library.new_function(OnFault::Die, name).unwrap().unwrap()
}

/// A decoder on `P`'s system device, holding the objects [`command_buffer`] names, and the
/// channel its completions come on.
struct Bench<P: ApiPlatform> {
    shape: Shape,
    platform: Arc<P>,
    device: P::Device,
    decoder: GraphicsDecoder<P>,
    completed: mpsc::Receiver<u64>,
}

impl<P: ApiPlatform> Bench<P> {
    fn new(shape: Shape) -> Bench<P> {
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let region_size =
            RGFX_CONTEXT_ALLOCATIONS_OFFSET + shape.chunks() * CHUNK + RGFX_CURSOR_SPAN;
        let region = reims_graphics::region::Region::new(region_size as usize).unwrap();
        let mut decoder = GraphicsDecoder::<P>::new(Arc::clone(&platform), Arc::new(region));
        let (tx, completed) = mpsc::channel();
        let tx = reims_vgpu_core::sync::Mutex::new(tx);
        decoder.set_completion_sink(Arc::new(move |m| {
            let _ = tx.lock().send(m.cb_obj_id);
        }));
        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: QUEUE_ID,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(decoder.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
        for n in 0..shape.chunks() {
            let chunk = RgfxBufferOverHostMemory {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
                    size: size_of::<RgfxBufferOverHostMemory>() as u32,
                    seqno: 0,
                },
                obj_id: CHUNK_ID + n,
                region_offset: RGFX_CONTEXT_ALLOCATIONS_OFFSET + n * CHUNK,
                length: CHUNK,
                ..RgfxBufferOverHostMemory::default()
            };
            assert_eq!(decoder.decode(&device, chunk.as_bytes()).unwrap(), Ok(()));
        }
        let target = platform.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: SIDE,
            height: SIDE,
            usage: TARGET_USAGE,
            ..TextureDesc::default()
        });
        let target = device.new_texture(OnFault::Die, &target).unwrap().unwrap();
        let vertex = fixture::<P>(&device, "render_vbuf");
        let fragment = fixture::<P>(&device, "render_fbuf");
        let pipeline = device
            .new_render_pipeline_state(&RenderPipelineDesc {
                label: None,
                vertex: &vertex,
                fragment: &fragment,
                color0_pixel_format: BGRA8,
                blend_over: false,
                resource_index: None,
            })
            .unwrap()
            .unwrap();
        let constants = device
            .new_buffer_of_length(OnFault::Die, 4096, 0)
            .unwrap()
            .unwrap();
        for (id, handle) in [
            (TARGET_ID, ObjHandle::Texture(target)),
            (PIPELINE_ID, ObjHandle::RenderPipeline(pipeline)),
            (CONSTANTS_ID, ObjHandle::Buffer(constants)),
        ] {
            assert_eq!(decoder.insert(id, handle).unwrap(), Ok(()));
        }
        Bench {
            shape,
            platform,
            device,
            decoder,
            completed,
        }
    }

    /// A buffer the decoder holds.
    fn buffer(&self, id: u64) -> &P::Buffer {
        match self.decoder.object(id) {
            Some(ObjHandle::Buffer(b)) => b,
            _ => panic!("no buffer {id}"),
        }
    }

    /// One repetition through the decoder: the thread's CPU nanoseconds from the first message's
    /// take to the last one's commit.
    fn through_decoder(&mut self, messages: &[Vec<u8>]) -> Counts {
        let start = counts();
        for m in messages {
            assert_eq!(self.decoder.decode(&self.device, m).unwrap(), Ok(()));
        }
        let spent = counts() - start;
        for _ in messages {
            self.completed
                .recv_timeout(Duration::from_secs(30))
                .expect("a command buffer never completed");
        }
        spent
    }

    /// One repetition through the platform's core traits, on the decoder's objects.
    fn through_platform(&self) -> Counts {
        let o = OnFault::Die;
        let Some(ObjHandle::CommandQueue(queue)) = self.decoder.object(QUEUE_ID) else {
            panic!("no queue");
        };
        let Some(ObjHandle::Texture(target)) = self.decoder.object(TARGET_ID) else {
            panic!("no target");
        };
        let Some(ObjHandle::RenderPipeline(pipeline)) = self.decoder.object(PIPELINE_ID) else {
            panic!("no pipeline");
        };
        let constants = self.buffer(CONSTANTS_ID);
        let chunks: Vec<&P::Buffer> = (0..self.shape.chunks())
            .map(|n| self.buffer(CHUNK_ID + n))
            .collect();
        let (tx, rx) = mpsc::channel();
        let start = counts();
        for cb_at in 0..self.shape.cbs {
            let cb = queue.command_buffer(o).unwrap().unwrap();
            let pass = self.platform.render_pass_descriptor();
            let attachment = ColorAttachmentSet {
                texture: Some(Some(target)),
                load_action: Some(CLEAR),
                store_action: Some(STORE),
                ..ColorAttachmentSet::default()
            };
            pass.set_color_attachment(o, 0, &attachment).unwrap();
            let e = cb.render_encoder(o, &pass).unwrap().unwrap();
            e.encoding(|| {
                let cmds = [
                    RenderCmd::SetRenderPipelineState {
                        pipeline_state: Some(pipeline),
                    },
                    RenderCmd::SetVertexBuffer {
                        buffer: Some(constants),
                        offset: 0,
                        index: 1,
                    },
                    RenderCmd::SetFragmentBuffer {
                        buffer: Some(constants),
                        offset: 0,
                        index: 0,
                    },
                ];
                for cmd in cmds {
                    e.encode(o, cmd).unwrap();
                }
                for draw in 0..self.shape.draws {
                    let (chunk, offset) = upload_at(self.shape, cb_at, draw);
                    let bind = RenderCmd::SetVertexBuffer {
                        buffer: Some(chunks[(chunk - CHUNK_ID) as usize]),
                        offset,
                        index: 0,
                    };
                    e.encode(o, bind).unwrap();
                    let drawn = RenderCmd::DrawPrimitives {
                        primitive_type: TRIANGLE,
                        vertex_start: 0,
                        vertex_count: 3,
                    };
                    e.encode(o, drawn).unwrap();
                }
            });
            e.encode(o, RenderCmd::EndEncoding).unwrap();
            let tx = tx.clone();
            cb.add_completed_handler(o, Box::new(move |_| tx.send(()).unwrap()))
                .unwrap();
            cb.commit(o).unwrap();
        }
        let spent = counts() - start;
        for _ in 0..self.shape.cbs {
            rx.recv_timeout(Duration::from_secs(30))
                .expect("a command buffer never completed");
        }
        spent
    }
}

/// The runtime alone, on the Vulkan platform's objects of `bench`.
#[cfg(feature = "vulkan")]
fn through_runtime(bench: &Bench<reims_vgpu_vulkan::VulkanPlatform>) -> Counts {
    use metal2vulkan_runtime as rt;
    let triangle = rt::DrawPrimitiveType::from_raw(TRIANGLE).unwrap();
    let Some(ObjHandle::CommandQueue(queue)) = bench.decoder.object(QUEUE_ID) else {
        panic!("no queue");
    };
    let Some(ObjHandle::Texture(target)) = bench.decoder.object(TARGET_ID) else {
        panic!("no target");
    };
    let Some(ObjHandle::RenderPipeline(pipeline)) = bench.decoder.object(PIPELINE_ID) else {
        panic!("no pipeline");
    };
    let constants = &bench.buffer(CONSTANTS_ID).raw;
    let shape = bench.shape;
    let chunks: Vec<&rt::Buffer> = (0..shape.chunks())
        .map(|n| &bench.buffer(CHUNK_ID + n).raw)
        .collect();
    let pass = bench.platform.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(target)),
        load_action: Some(CLEAR),
        store_action: Some(STORE),
        ..ColorAttachmentSet::default()
    };
    pass.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    let (pass, _) = pass.pass();
    let (tx, rx) = mpsc::channel();
    let start = counts();
    for cb_at in 0..shape.cbs {
        let cb = queue.raw.command_buffer().unwrap();
        let mut e = cb.render_command_encoder(&pass).unwrap();
        e.set_render_pipeline_state(&pipeline.raw).unwrap();
        e.set_vertex_buffer(Some(constants), 0, 1).unwrap();
        e.set_fragment_buffer(Some(constants), 0, 0).unwrap();
        for draw in 0..shape.draws {
            let (chunk, offset) = upload_at(shape, cb_at, draw);
            e.set_vertex_buffer(Some(chunks[(chunk - CHUNK_ID) as usize]), offset, 0)
                .unwrap();
            e.draw_primitives(triangle, 0, 3, 1, 0).unwrap();
        }
        e.end_encoding().unwrap();
        let tx = tx.clone();
        cb.add_completed_handler(Box::new(move |_: &rt::CommandBuffer| tx.send(()).unwrap()))
            .unwrap();
        cb.commit().unwrap();
    }
    let spent = counts() - start;
    for _ in 0..shape.cbs {
        rx.recv_timeout(Duration::from_secs(30))
            .expect("a command buffer never completed");
    }
    spent
}

/// The median, lowest and highest of `values`.
fn spread(values: &mut [f64]) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    };
    [median, values[0], values[n - 1]]
}

/// Runs the paths of `bench`, `runtime` the runtime-alone one where the platform has it, and
/// prints each one's figures a draw.
#[allow(clippy::disallowed_methods)] // a measurement: real time decides how long a profile runs
fn measure<P: ApiPlatform>(
    name: &str,
    mut bench: Bench<P>,
    runtime: Option<fn(&Bench<P>) -> Counts>,
) {
    let runs: usize = std::env::var("DRAW_COST_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9);
    let only = std::env::var("DRAW_COST_ONLY").ok();
    let only = only.as_deref();
    let seconds: Option<u64> = std::env::var("DRAW_COST_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok());
    let shape = bench.shape;
    let mut seqno = 0x1000u64;
    let mut messages = || {
        (0..shape.cbs)
            .map(|cb| {
                seqno += 1;
                command_buffer(shape, cb, seqno)
            })
            .collect::<Vec<_>>()
    };
    let _pool = reims_vgpu_core::rt::Pool::here();
    // On a Mac, the thread asks for the performance cores, which an efficiency core's slower
    // draws would otherwise mix into the figures.
    #[cfg(target_os = "macos")]
    // SAFETY: sets this thread's own class, which takes no pointer.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let a_draw = |c: Counts, reps: u64| {
        c.0.map(|v| v as f64 / (reps * (shape.cbs * shape.draws) as u64) as f64)
    };
    if let Some(seconds) = seconds {
        let until = Instant::now() + Duration::from_secs(seconds);
        let mut reps = 0u64;
        let mut spent = Counts::default();
        while Instant::now() < until {
            let m = messages();
            spent += match only {
                Some("platform") => bench.through_platform(),
                Some("runtime") => runtime.map_or(Counts::default(), |r| r(&bench)),
                _ => bench.through_decoder(&m),
            };
            reps += 1;
        }
        let [ns, instructions, cycles] = a_draw(spent, reps);
        println!(
            "{name}: {} for {seconds} s: {reps} repetitions, a draw {ns:.1} ns, {instructions:.0} \
             instructions, {cycles:.0} cycles",
            only.unwrap_or("decoder"),
        );
        return;
    }
    let paths = ["decoder", "platform", "runtime"];
    // Each path's figures a draw, by run.
    let mut seen: [Vec<[f64; 3]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..runs {
        let mut totals = [Counts::default(); 3];
        for rep in 0..WARM + REPS {
            let m = messages();
            let mut spent = [Counts::default(); 3];
            if only.is_none_or(|o| o == "decoder") {
                spent[0] = bench.through_decoder(&m);
            }
            if only.is_none_or(|o| o == "platform") {
                spent[1] = bench.through_platform();
            }
            if let Some(runtime) = runtime
                && only.is_none_or(|o| o == "runtime")
            {
                spent[2] = runtime(&bench);
            }
            if rep >= WARM {
                for (t, s) in totals.iter_mut().zip(spent) {
                    *t += s;
                }
            }
        }
        for (seen, total) in seen.iter_mut().zip(totals) {
            seen.push(a_draw(total, REPS as u64));
        }
    }
    println!(
        "{name}: a draw on the decoding thread, median [lowest, highest] of {runs} runs of {REPS} \
         repetitions of {} command buffers of {} draws (a command buffer's: {} times as much)",
        shape.cbs, shape.draws, shape.draws
    );
    for (f, figure) in FIGURES.iter().enumerate() {
        let mut medians = [0.0f64; 3];
        for (p, path) in paths.iter().enumerate() {
            let mut values: Vec<f64> = seen[p].iter().map(|v| v[f]).collect();
            if values.iter().all(|&v| v == 0.0) {
                continue;
            }
            let [median, lowest, highest] = spread(&mut values);
            medians[p] = median;
            println!("  {figure:>12} {path:>9}: {median:9.1} [{lowest:.1}, {highest:.1}]");
        }
        if medians[0] > 0.0 && medians[1] > 0.0 {
            println!(
                "  {figure:>12} decoder above platform: {:.1}",
                medians[0] - medians[1]
            );
        }
        if medians[1] > 0.0 && medians[2] > 0.0 {
            println!(
                "  {figure:>12} platform above runtime: {:.1}",
                medians[1] - medians[2]
            );
        }
    }
}

/// On the Vulkan platform: the decoder, the platform and the runtime alone.
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU: prints a draw's cost"]
fn on_vulkan_draw_cost() {
    let bench = Bench::<reims_vgpu_vulkan::VulkanPlatform>::new(Shape::chosen());
    measure("vulkan", bench, Some(through_runtime));
}

/// On the Metal platform: the decoder and the platform.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU: prints a draw's cost"]
fn on_metal_draw_cost() {
    let bench = Bench::<reims_vgpu_metal::MetalPlatform>::new(Shape::chosen());
    measure("metal", bench, None);
}
