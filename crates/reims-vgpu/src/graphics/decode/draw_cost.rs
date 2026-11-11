//! What a draw costs the thread that decodes it, through the decoder as a queue's ring hands it a
//! `COMMAND_BUFFER`, against the same commands sent through the platform's core traits and
//! through what the platform drives alone, metal2vulkan's runtime. A measurement, run by hand in a release build, in a potluck job
//! holding the host's GPU (`--gpu`: `cargo nextest run -p reims-vgpu --release
//! --no-default-features --features reims-vgpu/vulkan --run-ignored only -E 'test(draw_cost)'
//! --no-capture`).
//!
//! A command buffer is one render pass of draws, or, in a compositor's frame, many ([`Shape`],
//! [`Stream`]). In `draws.10k`'s stream
//! each draw of 3 vertices comes after a `setVertexBuffer:` at index 0 of an upload chunk at its
//! next 256-byte offset, as the bundle sends `setVertexBytes:` (a 64 KiB
//! `BUFFER_OVER_HOST_MEMORY` chunk of the region, 256 draws a chunk). In a compositor's, each
//! draw of a 4-vertex strip binds a texture of 16 and its own vertex and fragment bytes, under one
//! of 4 pipelines that change every 4 draws with the sampler set again. The command buffers go
//! back to back and are then waited for, as `draws.10k` commits them. Each path's figures are
//! the thread's CPU time, instructions and cycles a draw, from the first command buffer's begin
//! to the last one's commit: the decoder's include the message's take, its walk and its commit
//! with the decoder's handlers; the gaps between the paths are the decoder's share and the
//! platform's. Instructions are the figure a busy host leaves alone (measured on the M3 Max at a
//! load of 50: within 0.2% run to run, where CPU time moved 15%).
//!
//! The environment picks what runs: `DRAW_COST_SHAPE` the shape (`10k`, the default; `many`, 64
//! command buffers of 16 draws, whose figures show a command buffer's own cost; `ws`, 8 of 500
//! draws of a compositor's stream; `mc`, 8 of WindowServer's Mission Control composites, each of
//! 92 render passes of 3 draws; or `mc1`, the same ops in one pass a command buffer, whose gap to
//! `mc` is what 91 more passes cost), `DRAW_COST_RUNS` the runs (9), `DRAW_COST_ONLY` one path, for
//! a profile of it (`decoder`, `platform` or `runtime`, the runtime alone), and
//! `DRAW_COST_SECONDS` a time to keep running that path for, so a sampling profiler can attach.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use reims_vgpu_graphics::ops::render::{
    RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_SET_FRAGMENT_BUFFER,
    RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES, RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RGFX_OP_RENDER_SET_VERTEX_BUFFER,
    RgfxOpDrawPrimitives, RgfxOpSetFragmentBuffer, RgfxOpSetFragmentSamplerStates,
    RgfxOpSetFragmentTexture, RgfxOpSetRenderPipelineState, RgfxOpSetVertexBuffer,
    RgfxSamplerBinding,
};
use reims_vgpu_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_ENCODER_BEGIN_RENDER, RGFX_OP_ENCODER_END,
    RGFX_PASS_FLAG_RT_SIZE, RgfxCommandBuffer, RgfxOpHeader, RgfxRenderPassColorAttachment,
    RgfxRenderPassDesc,
};
use reims_vgpu_graphics::wire::{
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_QUEUE_CREATE,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RgfxBufferOverHostMemory, RgfxMsgHeader,
    RgfxQueueCreate,
};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::render::{RenderCmd, RenderEncoder as _, VertexDraw};
use reims_vgpu_vulkan::core::platform::resource::Library as _;
use reims_vgpu_vulkan::core::platform::types::NsRange;
use reims_vgpu_vulkan::core::platform::values::{
    ColorAttachmentDesc, ColorAttachmentSet, FullRenderPipelineDesc, SamplerDesc, TextureDesc,
};
use reims_vgpu_vulkan::core::raise::OnFault;
use zerocopy::IntoBytes;

use super::{GraphicsDecoder, ObjHandle};
use crate::platform::ApiPlatform;

/// How a repetition's work is cut: command buffers, committed back to back and then waited for,
/// each of `draws` draws of `stream`'s in `groups` equal runs, each run after the pipeline and
/// the constant buffers are set, the runs in `passes` render passes.
#[derive(Debug, Clone, Copy)]
struct Shape {
    cbs: usize,
    draws: usize,
    stream: Stream,
    groups: usize,
    passes: usize,
}

/// What a command buffer's draws send ([`steps`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stream {
    /// `draws.10k`'s: one pipeline, and each draw of 3 vertices after its `setVertexBytes:`.
    Bytes,
    /// A compositor's, as a frame of layers draws: each draw of a 4-vertex strip after its
    /// texture, its fragment bytes and its vertex bytes, under one of [`PIPELINES`] that change
    /// every [`RUN`] draws, the sampler set again with each. The textures are plain ones, not
    /// over surfaces.
    Layers,
}

impl Shape {
    /// `draws.10k`'s: 3 command buffers of 10,000 draws.
    const TEN_K: Shape = Shape {
        cbs: 3,
        draws: 10_000,
        stream: Stream::Bytes,
        groups: 1,
        passes: 1,
    };
    /// Many short command buffers, as a guest's frame of 60 to 100 is cut: 64 of 16 draws.
    const MANY: Shape = Shape {
        cbs: 64,
        draws: 16,
        stream: Stream::Bytes,
        groups: 1,
        passes: 1,
    };
    /// A compositor's frames: 8 command buffers of 500 draws.
    const WS: Shape = Shape {
        cbs: 8,
        draws: 500,
        stream: Stream::Layers,
        groups: 1,
        passes: 1,
    };
    /// WindowServer's Mission Control composites as the msi 5080's car 13 boot counted them
    /// (c13-mission's P2, a median of 3,954: 92 encoders, 270 draws, 2,110 render ops, no use
    /// ops): 8 command buffers of 92 render passes of 3 draws of a compositor's stream, each pass
    /// after its pipeline and constant buffers, the first pass clearing the target and the rest
    /// loading it. The draws' binds are the compositor stream's, which the counts do not name:
    /// 17 render ops a pass here against the 23 counted, the attachments one target.
    const MC: Shape = Shape {
        cbs: 8,
        draws: 276,
        stream: Stream::Layers,
        groups: 92,
        passes: 92,
    };
    /// [`MC`](Shape::MC)'s ops in one render pass a command buffer.
    const MC1: Shape = Shape {
        passes: 1,
        ..Shape::MC
    };

    /// The 256-byte upload slots a draw takes: its vertex bytes, and its fragment bytes in a
    /// compositor's stream.
    fn slots(self) -> u64 {
        match self.stream {
            Stream::Bytes => 1,
            Stream::Layers => 2,
        }
    }

    /// The upload chunks a command buffer's draws take.
    fn chunks_a_cb(self) -> u64 {
        (self.draws as u64 * self.slots() * ALIGN).div_ceil(CHUNK)
    }

    /// The chunks the region holds: a repetition's command buffers each take their own.
    fn chunks(self) -> u64 {
        self.chunks_a_cb() * self.cbs as u64
    }

    /// The shape `DRAW_COST_SHAPE` names: `many`, `ws`, `mc`, `mc1` or, by default, `10k`.
    fn chosen() -> Shape {
        match std::env::var("DRAW_COST_SHAPE").as_deref() {
            Ok("many") => Shape::MANY,
            Ok("ws") => Shape::WS,
            Ok("mc") => Shape::MC,
            Ok("mc1") => Shape::MC1,
            _ => Shape::TEN_K,
        }
    }

    /// The load action of pass `pass` of a command buffer: the first clears the target, the
    /// rest load it.
    fn load(pass: usize) -> u64 {
        if pass == 0 { CLEAR } else { LOAD }
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

/// The object ids: the queue, the target, the per-draw constant buffers (vertex 1 and fragment
/// 0), the pipelines from [`PIPELINE_ID`], the textures from [`TEXTURE_ID`], the sampler, then
/// the chunks from [`CHUNK_ID`].
const QUEUE_ID: u64 = 1;
const TARGET_ID: u64 = 2;
const CONSTANTS_ID: u64 = 4;
const PIPELINE_ID: u64 = 0x10;
const TEXTURE_ID: u64 = 0x20;
const SAMPLER_ID: u64 = 0x40;
const CHUNK_ID: u64 = 0x100;
/// A compositor's pipelines: a fragment buffer's colour and a texture's, each opaque and blended
/// over (`draws.10k`'s is the first); its textures; and the draws a pipeline runs for.
const PIPELINES: u64 = 4;
const TEXTURES: u64 = 16;
const RUN: usize = 4;

/// `MTLPixelFormatBGRA8Unorm`; `MTLTextureUsageShaderRead`, and `| RenderTarget`; `Clear`,
/// `Load`, `Store`; `MTLPrimitiveTypeTriangle` and `TriangleStrip`.
const BGRA8: u64 = 80;
const SAMPLED_USAGE: u64 = 1;
const TARGET_USAGE: u64 = 1 | 4;
const CLEAR: u64 = 2;
const LOAD: u64 = 1;
const STORE: u64 = 1;
const TRIANGLE: u64 = 3;
const TRIANGLE_STRIP: u64 = 4;
/// The target's side, and a texture's.
const SIDE: u64 = 512;
const TEXTURE_SIDE: u64 = 64;

/// This thread's CPU time in nanoseconds, the instructions it retired and its cycles, each as a
/// count that only grows; the counters are 0 where the host gives none.
pub(in crate::graphics) fn counts() -> Counts {
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
pub(in crate::graphics) struct Counts(pub(in crate::graphics) [u64; 3]);

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
pub(in crate::graphics) const FIGURES: [&str; 3] = ["CPU ns", "instructions", "cycles"];

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

/// The chunk and offset of upload slot `slot` of command buffer `cb` of a repetition of `shape`.
fn upload_at(shape: Shape, cb: usize, slot: u64) -> (u64, u64) {
    let per_cb = shape.chunks_a_cb();
    let at = slot * ALIGN;
    (CHUNK_ID + cb as u64 * per_cb + at / CHUNK, at % CHUNK)
}

/// One render op of a command buffer's, by the ids of the objects it names.
#[derive(Debug, Clone, Copy)]
enum Step {
    /// `setRenderPipelineState:`.
    Pipeline(u64),
    /// `setVertexBuffer:offset:atIndex:` of a buffer, at an offset and an index.
    VertexBuffer(u64, u64, u64),
    /// `setFragmentBuffer:offset:atIndex:`.
    FragmentBuffer(u64, u64, u64),
    /// `setFragmentTexture:atIndex:`.
    FragmentTexture(u64, u64),
    /// `setFragmentSamplerState:atIndex:` at index 0, which the bundle sends as a range of one.
    FragmentSampler(u64),
    /// `drawPrimitives:vertexStart:vertexCount:` of a primitive type and a count, from vertex 0.
    Draw(u64, u64),
}

/// Command buffer `cb`'s render ops, of a repetition of `shape`, a list a render pass: each of
/// the shape's groups the pipeline and the constant buffers, then its draws of the stream.
fn steps(shape: Shape, cb: usize) -> Vec<Vec<Step>> {
    assert!(shape.draws.is_multiple_of(shape.groups) && shape.groups.is_multiple_of(shape.passes));
    let a_group = shape.draws / shape.groups;
    let mut passes = vec![Vec::new(); shape.passes];
    let mut steps = Vec::new();
    for draw in 0..shape.draws {
        if draw.is_multiple_of(a_group) {
            if draw > 0 && (draw / a_group).is_multiple_of(shape.groups / shape.passes) {
                passes[draw / a_group / (shape.groups / shape.passes) - 1] =
                    std::mem::take(&mut steps);
            }
            steps.extend([
                Step::Pipeline(PIPELINE_ID),
                Step::VertexBuffer(CONSTANTS_ID, 0, 1),
                Step::FragmentBuffer(CONSTANTS_ID, 0, 0),
            ]);
        }
        let slot = draw as u64 * shape.slots();
        match shape.stream {
            Stream::Bytes => {
                let (chunk, offset) = upload_at(shape, cb, slot);
                steps.push(Step::VertexBuffer(chunk, offset, 0));
                steps.push(Step::Draw(TRIANGLE, 3));
            }
            Stream::Layers => {
                if draw % RUN == 0 {
                    let pipeline = (draw / RUN) as u64 % PIPELINES;
                    steps.push(Step::Pipeline(PIPELINE_ID + pipeline));
                    steps.push(Step::FragmentSampler(SAMPLER_ID));
                }
                steps.push(Step::FragmentTexture(
                    TEXTURE_ID + draw as u64 % TEXTURES,
                    0,
                ));
                let (chunk, offset) = upload_at(shape, cb, slot + 1);
                steps.push(Step::FragmentBuffer(chunk, offset, 0));
                let (chunk, offset) = upload_at(shape, cb, slot);
                steps.push(Step::VertexBuffer(chunk, offset, 0));
                steps.push(Step::Draw(TRIANGLE_STRIP, 4));
            }
        }
    }
    passes[shape.passes - 1] = steps;
    passes
}

/// The op the bundle sends for `step`.
fn step_op(step: Step) -> Vec<u8> {
    match step {
        Step::Pipeline(id) => op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState {
                pipeline_obj_id: id,
            }
            .as_bytes(),
        ),
        Step::VertexBuffer(id, offset, index) => op(
            RGFX_OP_RENDER_SET_VERTEX_BUFFER,
            RgfxOpSetVertexBuffer {
                buffer_obj_id: id,
                offset,
                index,
            }
            .as_bytes(),
        ),
        Step::FragmentBuffer(id, offset, index) => op(
            RGFX_OP_RENDER_SET_FRAGMENT_BUFFER,
            RgfxOpSetFragmentBuffer {
                buffer_obj_id: id,
                offset,
                index,
            }
            .as_bytes(),
        ),
        Step::FragmentTexture(id, index) => op(
            RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
            RgfxOpSetFragmentTexture {
                texture_obj_id: id,
                index,
            }
            .as_bytes(),
        ),
        Step::FragmentSampler(id) => {
            let mut body = RgfxOpSetFragmentSamplerStates {
                range_location: 0,
                range_length: 1,
            }
            .as_bytes()
            .to_vec();
            body.extend_from_slice(RgfxSamplerBinding { sampler_obj_id: id }.as_bytes());
            op(RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES, &body)
        }
        Step::Draw(primitive_type, vertex_count) => op(
            RGFX_OP_RENDER_DRAW_PRIMITIVES,
            RgfxOpDrawPrimitives {
                primitive_type,
                vertex_start: 0,
                vertex_count,
            }
            .as_bytes(),
        ),
    }
}

/// The `COMMAND_BUFFER` message of command buffer `cb` of a repetition of `shape`, as the bundle
/// sends its ops ([`steps`]).
fn command_buffer(shape: Shape, cb: usize, cb_obj_id: u64) -> Vec<u8> {
    let desc = RgfxRenderPassDesc {
        color_count: 1,
        flags: RGFX_PASS_FLAG_RT_SIZE,
        render_target_width: SIDE as u32,
        render_target_height: SIDE as u32,
        ..RgfxRenderPassDesc::default()
    };
    let mut ops = Vec::new();
    for (pass, steps) in steps(shape, cb).into_iter().enumerate() {
        let color = RgfxRenderPassColorAttachment {
            texture_obj_id: TARGET_ID,
            load_action: Shape::load(pass) as u32,
            store_action: STORE as u32,
            clear_color_bits: [0, 0, 0, 1.0f64.to_bits()],
            ..RgfxRenderPassColorAttachment::default()
        };
        let mut begin = desc.as_bytes().to_vec();
        begin.extend_from_slice(color.as_bytes());
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin));
        for step in steps {
            ops.extend(step_op(step));
        }
        ops.extend(op(RGFX_OP_ENCODER_END, &[]));
    }
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

/// The committed test shaders, `tests/shaders`, under the manifest directory
/// cargo or nextest gives the test as it runs, which a run of a nextest archive on another host
/// remaps to that host's tree.
pub(super) fn test_shaders() -> std::path::PathBuf {
    std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo or nextest"),
    )
    .join("../../tests/shaders")
}

/// A colour attachment of `pixel_format` that blends each fragment, its colour premultiplied,
/// over the target by its alpha: both source factors `one` (1), both destination factors
/// `oneMinusSourceAlpha` (5).
pub(super) fn over(pixel_format: u64) -> ColorAttachmentDesc {
    ColorAttachmentDesc {
        pixel_format,
        blending_enabled: true,
        source_rgb_blend_factor: 1,
        destination_rgb_blend_factor: 5,
        source_alpha_blend_factor: 1,
        destination_alpha_blend_factor: 5,
        ..ColorAttachmentDesc::default()
    }
}

/// A function of the suite's fixture `name`.
pub(super) fn fixture<P: Platform>(d: &P::Device, name: &str) -> P::Function {
    let path = test_shaders().join(format!("{name}.metallib"));
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
        let region = reims_vgpu_graphics::region::Region::new(region_size as usize).unwrap();
        let mut decoder = GraphicsDecoder::<P>::new(Arc::clone(&platform), Arc::new(region));
        let (tx, completed) = mpsc::channel();
        let tx = reims_vgpu_vulkan::core::sync::Mutex::new(tx);
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
        let colours = [
            fixture::<P>(&device, "render_fbuf"),
            fixture::<P>(&device, "render_ftex"),
        ];
        for n in 0..PIPELINES {
            let color = if n >= 2 {
                over(BGRA8)
            } else {
                ColorAttachmentDesc {
                    pixel_format: BGRA8,
                    ..ColorAttachmentDesc::default()
                }
            };
            let pipeline = device
                .new_render_pipeline_state_full(&FullRenderPipelineDesc {
                    colors: vec![(0, color)],
                    ..FullRenderPipelineDesc::new(&vertex, Some(&colours[(n % 2) as usize]))
                })
                .unwrap()
                .unwrap();
            let handle = ObjHandle::RenderPipeline(pipeline);
            assert_eq!(decoder.insert(PIPELINE_ID + n, handle).unwrap(), Ok(()));
        }
        let sampled = platform.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: TEXTURE_SIDE,
            height: TEXTURE_SIDE,
            usage: SAMPLED_USAGE,
            ..TextureDesc::default()
        });
        for n in 0..TEXTURES {
            let texture = device.new_texture(OnFault::Die, &sampled).unwrap().unwrap();
            let handle = ObjHandle::Texture(texture);
            assert_eq!(decoder.insert(TEXTURE_ID + n, handle).unwrap(), Ok(()));
        }
        let sampler = device
            .new_sampler_state(&SamplerDesc::default())
            .unwrap()
            .unwrap();
        let constants = device
            .new_buffer_of_length(OnFault::Die, 4096, 0)
            .unwrap()
            .unwrap();
        for (id, handle) in [
            (TARGET_ID, ObjHandle::Texture(target)),
            (SAMPLER_ID, ObjHandle::Sampler(sampler)),
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

    /// The queue, and the pass's target.
    fn queue_and_target(&self) -> (&P::CommandQueue, &P::Texture) {
        let Some(ObjHandle::CommandQueue(queue)) = self.decoder.object(QUEUE_ID) else {
            panic!("no queue");
        };
        let Some(ObjHandle::Texture(target)) = self.decoder.object(TARGET_ID) else {
            panic!("no target");
        };
        (queue, target)
    }

    /// The sampler.
    fn sampler(&self) -> &P::SamplerState {
        let Some(ObjHandle::Sampler(sampler)) = self.decoder.object(SAMPLER_ID) else {
            panic!("no sampler");
        };
        sampler
    }

    /// Each command buffer's render ops of a repetition, a list a pass, their objects looked
    /// up: made before a count starts.
    fn bound(&self) -> Vec<Vec<Vec<Bound<'_, P>>>> {
        let named = |step: Step| match step {
            Step::Pipeline(id) => match self.decoder.object(id) {
                Some(ObjHandle::RenderPipeline(p)) => Bound::Pipeline(p),
                _ => panic!("no pipeline {id}"),
            },
            Step::VertexBuffer(id, offset, index) => {
                Bound::VertexBuffer(self.buffer(id), offset, index)
            }
            Step::FragmentBuffer(id, offset, index) => {
                Bound::FragmentBuffer(self.buffer(id), offset, index)
            }
            Step::FragmentTexture(id, index) => match self.decoder.object(id) {
                Some(ObjHandle::Texture(t)) => Bound::FragmentTexture(t, index),
                _ => panic!("no texture {id}"),
            },
            Step::FragmentSampler(_) => Bound::FragmentSampler(self.sampler()),
            Step::Draw(primitive_type, count) => Bound::Draw(primitive_type, count),
        };
        (0..self.shape.cbs)
            .map(|cb| {
                steps(self.shape, cb)
                    .into_iter()
                    .map(|pass| pass.into_iter().map(named).collect())
                    .collect()
            })
            .collect()
    }

    /// What the decoder's walk sends the platform for `bound`: each op's command, but for a
    /// `setVertexBuffer:` and the draw after it, which go with the draws beside them, as many
    /// at once as the walk holds. `samplers` is the list of the bench's one sampler.
    fn sent<'a>(
        bound: &[Bound<'a, P>],
        samplers: &'a [Option<&'a P::SamplerState>],
    ) -> Vec<Sent<'a, P>> {
        let mut sent = Vec::new();
        let mut held = Vec::new();
        let mut at = 0;
        while at < bound.len() {
            if let (Bound::VertexBuffer(buffer, offset, index), Some(&Bound::Draw(primitive, n))) =
                (bound[at], bound.get(at + 1))
            {
                if held.len() == super::cmdbuf::DRAWS_HELD {
                    sent.push(Sent::Draws(std::mem::take(&mut held)));
                }
                held.push(VertexDraw {
                    buffer: Some(buffer),
                    offset,
                    index,
                    primitive_type: primitive,
                    vertex_start: 0,
                    vertex_count: n,
                });
                at += 2;
                continue;
            }
            if !held.is_empty() {
                sent.push(Sent::Draws(std::mem::take(&mut held)));
            }
            sent.push(Sent::Cmd(match bound[at] {
                Bound::Pipeline(p) => RenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(p),
                },
                Bound::VertexBuffer(buffer, offset, index) => RenderCmd::SetVertexBuffer {
                    buffer: Some(buffer),
                    offset,
                    index,
                },
                Bound::FragmentBuffer(buffer, offset, index) => RenderCmd::SetFragmentBuffer {
                    buffer: Some(buffer),
                    offset,
                    index,
                },
                Bound::FragmentTexture(texture, index) => RenderCmd::SetFragmentTexture {
                    texture: Some(texture),
                    index,
                },
                Bound::FragmentSampler(_) => RenderCmd::SetFragmentSamplerStates {
                    samplers,
                    range: NsRange {
                        location: 0,
                        length: 1,
                    },
                },
                Bound::Draw(primitive_type, vertex_count) => RenderCmd::DrawPrimitives {
                    primitive_type,
                    vertex_start: 0,
                    vertex_count,
                },
            }));
            at += 1;
        }
        if !held.is_empty() {
            sent.push(Sent::Draws(held));
        }
        sent
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
        let (queue, target) = self.queue_and_target();
        let samplers = [Some(self.sampler())];
        // Each command buffer's commands, made before the count starts; the draws' lists are
        // kept until it ends, so their frees fall outside it.
        let bound = self.bound();
        let mut sent: Vec<Vec<Vec<Sent<'_, P>>>> = bound
            .iter()
            .map(|cb| cb.iter().map(|b| Self::sent(b, &samplers)).collect())
            .collect();
        let mut kept = Vec::with_capacity(sent.iter().flatten().map(Vec::len).sum());
        let (tx, rx) = mpsc::channel();
        let start = counts();
        for cb_sent in &mut sent {
            let cb = queue.command_buffer(o).unwrap().unwrap();
            for (p, pass_sent) in cb_sent.iter_mut().enumerate() {
                let pass = self.platform.render_pass_descriptor();
                let attachment = ColorAttachmentSet {
                    texture: Some(Some(target)),
                    load_action: Some(Shape::load(p)),
                    store_action: Some(STORE),
                    ..ColorAttachmentSet::default()
                };
                pass.set_color_attachment(o, 0, &attachment).unwrap();
                let e = cb.render_encoder(o, &pass).unwrap().unwrap();
                e.encoding(|| {
                    for s in pass_sent.drain(..) {
                        match s {
                            Sent::Cmd(cmd) => e.encode(o, cmd).unwrap(),
                            Sent::Draws(draws) => {
                                e.encode_draws(o, &draws).unwrap();
                                kept.push(draws);
                            }
                        }
                    }
                });
                e.encode(o, RenderCmd::EndEncoding).unwrap();
            }
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

/// A render op of [`Step`]'s with its objects looked up.
enum Bound<'a, P: Platform> {
    Pipeline(&'a P::RenderPipeline),
    VertexBuffer(&'a P::Buffer, u64, u64),
    FragmentBuffer(&'a P::Buffer, u64, u64),
    FragmentTexture(&'a P::Texture, u64),
    FragmentSampler(&'a P::SamplerState),
    Draw(u64, u64),
}

impl<P: Platform> Clone for Bound<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Platform> Copy for Bound<'_, P> {}

/// What the decoder's walk sends the platform ([`Bench::sent`]): a command, or draws together.
enum Sent<'a, P: Platform> {
    Cmd(RenderCmd<'a, P>),
    Draws(Vec<VertexDraw<'a, P>>),
}

/// The pass descriptor of the bench's target for pass `pass` of a command buffer, as the
/// platform's own form of it.
fn target_pass<P: ApiPlatform>(bench: &Bench<P>, pass: usize) -> P::RenderPassDescriptor {
    let (_, target) = bench.queue_and_target();
    let desc = bench.platform.render_pass_descriptor();
    let attachment = ColorAttachmentSet {
        texture: Some(Some(target)),
        load_action: Some(Shape::load(pass)),
        store_action: Some(STORE),
        ..ColorAttachmentSet::default()
    };
    desc.set_color_attachment(OnFault::Die, 0, &attachment)
        .unwrap();
    desc
}

/// The runtime alone, on the Vulkan platform's objects of `bench`: the calls the platform makes
/// of it for each command.
fn through_runtime(bench: &Bench<reims_vgpu_vulkan::VulkanPlatform>) -> Counts {
    use metal2vulkan_runtime as rt;
    let (queue, _) = bench.queue_and_target();
    let shape = bench.shape;
    let bound = bench.bound();
    let passes = [
        target_pass(bench, 0).pass().0,
        target_pass(bench, 1).pass().0,
    ];
    let (tx, rx) = mpsc::channel();
    let start = counts();
    for cb_bound in &bound {
        let cb = queue.raw.command_buffer().unwrap();
        for (p, pass_bound) in cb_bound.iter().enumerate() {
            let mut e = cb.render_command_encoder(&passes[p.min(1)]).unwrap();
            for &b in pass_bound {
                match b {
                    Bound::Pipeline(p) => e.set_render_pipeline_state(&p.raw),
                    Bound::VertexBuffer(buffer, offset, index) => {
                        e.set_vertex_buffer(Some(&buffer.raw), offset, index as u32)
                    }
                    Bound::FragmentBuffer(buffer, offset, index) => {
                        e.set_fragment_buffer(Some(&buffer.raw), offset, index as u32)
                    }
                    Bound::FragmentTexture(texture, index) => {
                        let texels = texture.texels("setFragmentTexture:atIndex:");
                        e.set_fragment_texture(Some(texels), index as u32)
                    }
                    Bound::FragmentSampler(sampler) => {
                        e.set_fragment_sampler_states(&[Some(&**sampler)], None, 0)
                    }
                    Bound::Draw(primitive_type, count) => {
                        let primitive = rt::DrawPrimitiveType::from_raw(primitive_type).unwrap();
                        e.draw_primitives(primitive, 0, count as u32, 1, 0)
                    }
                }
                .unwrap();
            }
            e.end_encoding().unwrap();
        }
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
pub(super) fn spread(values: &mut [f64]) -> [f64; 3] {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let median = if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    };
    [median, values[0], values[n - 1]]
}

/// Runs the paths of `bench`, `alone` what the platform drives alone (the runtime) and
/// its name, and prints each one's figures a draw.
#[allow(clippy::disallowed_methods)] // a measurement: real time decides how long a profile runs
fn measure<P: ApiPlatform>(
    name: &str,
    mut bench: Bench<P>,
    (alone, runtime): (&str, fn(&Bench<P>) -> Counts),
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
    let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
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
                Some("runtime") => runtime(&bench),
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
    let paths = ["decoder", "platform", alone];
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
            if only.is_none_or(|o| o == "runtime") {
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
    let ops: usize = steps(shape, 0).iter().map(|p| p.len() + 2).sum();
    println!(
        "{name}: a draw on the decoding thread, median [lowest, highest] of {runs} runs of {REPS} \
         repetitions of {} command buffers of {} draws of {:?} in {} passes, {ops} ops (a command \
         buffer's: {} times as much)",
        shape.cbs, shape.draws, shape.stream, shape.passes, shape.draws
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
                "  {figure:>12} platform above {alone}: {:.1}",
                medians[1] - medians[2]
            );
        }
    }
}

/// The mc shapes' command buffers: in `mc`, 92 passes of 3 draws, each pass's ops beginning with
/// its pipeline and constant buffers; in `mc1`, the same ops in one pass. The other shapes keep
/// one pass.
#[test]
fn the_mc_shapes_cut_their_draws_into_passes() {
    let draws = |pass: &Vec<Step>| pass.iter().filter(|s| matches!(s, Step::Draw(..))).count();
    let mc = steps(Shape::MC, 0);
    assert_eq!(mc.len(), 92);
    for pass in &mc {
        assert_eq!(draws(pass), 3);
        assert!(matches!(pass[0], Step::Pipeline(PIPELINE_ID)));
    }
    let mc1 = steps(Shape::MC1, 0);
    assert_eq!(mc1.len(), 1);
    assert_eq!(mc1[0].len(), mc.iter().map(Vec::len).sum::<usize>());
    for shape in [Shape::TEN_K, Shape::MANY, Shape::WS] {
        let one = steps(shape, 0);
        assert_eq!(one.len(), 1);
        assert_eq!(draws(&one[0]), shape.draws);
    }
}

/// On the Vulkan platform: the decoder, the platform and the runtime alone.
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU: prints a draw's cost"]
fn on_vulkan_draw_cost() {
    let bench = Bench::<reims_vgpu_vulkan::VulkanPlatform>::new(Shape::chosen());
    measure("vulkan", bench, ("runtime", through_runtime));
}
