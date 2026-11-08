//! How long a compositor's frame takes, through the decoder as a queue's ring hands it a
//! `COMMAND_BUFFER`: one command buffer of many render passes, as WindowServer's composite of a
//! Liquid Glass desktop is cut (a lane's bench of a window drag at 1920 x 1200 points counted 63
//! render encoders and about 170 draws a frame of context 1 on car 9), its walk on the decoding
//! thread, and when its GPU work ends. A measurement, run by hand in a release build, holding the
//! host's GPU (`cargo nextest run -p reims-vgpu --release --no-default-features --features
//! reims-vgpu/vulkan --run-ignored only -E 'test(composite_cost)' --no-capture`).
//!
//! A frame is one command buffer of [`Shape::passes`] render passes over a target of
//! [`Shape::width`] x [`Shape::height`] BGRA8 texels, each pass loading what the one before it
//! stored (the first clears it), and drawing [`Shape::draws`] textured quads blended over it, each
//! a [`Shape::cover`] of the target, after its own texture, vertex bytes and the pipeline. Every
//! [`Shape::reads`]th pass then reads its own target, as the glass material reads the backdrop it
//! draws over: a memory barrier of the render targets from fragment to fragment, and a quad
//! sampling the target. Frames go one at a time, each waited for, as a compositor's do.
//!
//! Each frame's figures: its walk (the decode call, from the message's take to its commit, wall
//! time and the thread's instructions), its latency (from the take to the completion record's
//! arrival in the decoder's sink), and its GPU span (the record's GPU start to end). The frame's
//! latency is what a compositor's pacing sees: where the GPU starts only at the commit, it is the
//! walk and the GPU in series.
//!
//! The environment picks the shape: `COMPOSITE_PASSES` (60), `COMPOSITE_DRAWS` (3),
//! `COMPOSITE_COVER` (a draw's share of the target, 0.25), `COMPOSITE_READS` (every 4th pass; 0
//! for none), `COMPOSITE_READ_FIRST` (1: a reading pass clears its target and reads it first,
//! the barrier its first command, as the glass material's passes that start by reading what
//! they clear do; 0, the default: it loads it and reads it after its draws),
//! `COMPOSITE_WIDTH` and `COMPOSITE_HEIGHT` (3840 x 2400), `COMPOSITE_FRAMES` (200, after 20
//! not counted) and `COMPOSITE_SPLIT_US`, how long a part of the command buffer is walked before
//! the walk commits it (the decoder's own `SPLIT_AFTER` where unset; one longer than the walk
//! keeps it whole, [`super::split`]).

use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use reims_vgpu_graphics::ops::render::{
    RGFX_OP_RENDER_DRAW_PRIMITIVES, RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE,
    RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES, RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE, RGFX_OP_RENDER_SET_VERTEX_BUFFER,
    RgfxOpDrawPrimitives, RgfxOpMemoryBarrierWithScope, RgfxOpSetFragmentSamplerStates,
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
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RgfxBufferOverHostMemory, RgfxCbComplete,
    RgfxMsgHeader, RgfxQueueCreate,
};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::values::{FullRenderPipelineDesc, SamplerDesc, TextureDesc};
use reims_vgpu_vulkan::core::raise::OnFault;
use zerocopy::IntoBytes;

use super::draw_cost::{counts, fixture, over, spread};
use super::{GraphicsDecoder, ObjHandle};
use crate::platform::ApiPlatform;

/// A frame's cut ([`Shape::chosen`]).
#[derive(Debug, Clone, Copy)]
struct Shape {
    passes: usize,
    draws: usize,
    cover: f32,
    reads: usize,
    read_first: bool,
    width: u32,
    height: u32,
    frames: usize,
    split_us: Option<u64>,
}

impl Shape {
    /// The shape the environment names, each figure by default as the module docs say.
    fn chosen() -> Shape {
        fn var<T: std::str::FromStr>(name: &str, default: T) -> T {
            (std::env::var(name).ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        }
        Shape {
            passes: var("COMPOSITE_PASSES", 60),
            draws: var("COMPOSITE_DRAWS", 3),
            cover: var("COMPOSITE_COVER", 0.25),
            reads: var("COMPOSITE_READS", 4),
            read_first: var("COMPOSITE_READ_FIRST", 0u8) != 0,
            width: var("COMPOSITE_WIDTH", 3840),
            height: var("COMPOSITE_HEIGHT", 2400),
            frames: var("COMPOSITE_FRAMES", 200),
            split_us: (std::env::var("COMPOSITE_SPLIT_US").ok()).and_then(|v| v.parse().ok()),
        }
    }

    /// Whether pass `pass` reads its own target.
    fn reads(self, pass: usize) -> bool {
        self.reads != 0 && pass % self.reads == self.reads - 1
    }

    /// The quads a frame draws, each with its 4 vertices' bytes in the upload chunk.
    fn quads(self) -> u64 {
        (0..self.passes)
            .map(|p| self.draws + usize::from(self.reads(p)))
            .sum::<usize>() as u64
    }
}

/// Frames counted after the first ones.
const WARM: usize = 20;
/// A quad's vertex bytes: 4 `float4` positions.
const QUAD: u64 = 64;
/// The object ids: the queue, the target, the colour buffer at vertex index 1, the pipeline, the
/// textures from [`TEXTURE_ID`], the sampler and the upload chunk.
const QUEUE_ID: u64 = 1;
const TARGET_ID: u64 = 2;
const CONSTANTS_ID: u64 = 4;
const PIPELINE_ID: u64 = 0x10;
const TEXTURE_ID: u64 = 0x20;
const TEXTURES: u64 = 16;
const SAMPLER_ID: u64 = 0x40;
const CHUNK_ID: u64 = 0x100;
/// `MTLPixelFormatBGRA8Unorm`; `MTLTextureUsageShaderRead`, and `| RenderTarget`; `Load`,
/// `Clear`, `Store`; `MTLPrimitiveTypeTriangleStrip`; `MTLBarrierScopeRenderTargets` and
/// `MTLRenderStageFragment`.
const BGRA8: u64 = 80;
const SAMPLED_USAGE: u64 = 1;
const TARGET_USAGE: u64 = 1 | 4;
const LOAD: u32 = 1;
const CLEAR: u32 = 2;
const STORE: u32 = 1;
const TRIANGLE_STRIP: u64 = 4;
const SCOPE_RENDER_TARGETS: u64 = 4;
const STAGE_FRAGMENT: u64 = 2;

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

/// The ops of a quad drawn at upload slot `quad`, sampling texture `texture`.
fn quad_ops(texture: u64, quad: u64) -> Vec<u8> {
    let mut ops = op(
        RGFX_OP_RENDER_SET_FRAGMENT_TEXTURE,
        RgfxOpSetFragmentTexture {
            texture_obj_id: texture,
            index: 0,
        }
        .as_bytes(),
    );
    ops.extend(op(
        RGFX_OP_RENDER_SET_VERTEX_BUFFER,
        RgfxOpSetVertexBuffer {
            buffer_obj_id: CHUNK_ID,
            offset: quad * QUAD,
            index: 0,
        }
        .as_bytes(),
    ));
    ops.extend(op(
        RGFX_OP_RENDER_DRAW_PRIMITIVES,
        RgfxOpDrawPrimitives {
            primitive_type: TRIANGLE_STRIP,
            vertex_start: 0,
            vertex_count: 4,
        }
        .as_bytes(),
    ));
    ops
}

/// A frame's `COMMAND_BUFFER` message, command buffer `cb_obj_id`.
fn frame(shape: Shape, cb_obj_id: u64) -> Vec<u8> {
    let mut ops = Vec::new();
    let mut quad = 0;
    for pass in 0..shape.passes {
        let desc = RgfxRenderPassDesc {
            color_count: 1,
            flags: RGFX_PASS_FLAG_RT_SIZE,
            render_target_width: shape.width,
            render_target_height: shape.height,
            ..RgfxRenderPassDesc::default()
        };
        let reads = shape.reads(pass);
        let color = RgfxRenderPassColorAttachment {
            texture_obj_id: TARGET_ID,
            load_action: if pass == 0 || (reads && shape.read_first) {
                CLEAR
            } else {
                LOAD
            },
            store_action: STORE,
            clear_color_bits: [0, 0, 0, 1.0f64.to_bits()],
            ..RgfxRenderPassColorAttachment::default()
        };
        let mut begin = desc.as_bytes().to_vec();
        begin.extend_from_slice(color.as_bytes());
        ops.extend(op(RGFX_OP_ENCODER_BEGIN_RENDER, &begin));
        ops.extend(op(
            RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
            RgfxOpSetRenderPipelineState {
                pipeline_obj_id: PIPELINE_ID,
            }
            .as_bytes(),
        ));
        ops.extend(op(
            RGFX_OP_RENDER_SET_VERTEX_BUFFER,
            RgfxOpSetVertexBuffer {
                buffer_obj_id: CONSTANTS_ID,
                offset: 0,
                index: 1,
            }
            .as_bytes(),
        ));
        let mut samplers = RgfxOpSetFragmentSamplerStates {
            range_location: 0,
            range_length: 1,
        }
        .as_bytes()
        .to_vec();
        samplers.extend_from_slice(
            RgfxSamplerBinding {
                sampler_obj_id: SAMPLER_ID,
            }
            .as_bytes(),
        );
        ops.extend(op(RGFX_OP_RENDER_SET_FRAGMENT_SAMPLER_STATES, &samplers));
        let read = |ops: &mut Vec<u8>, quad: &mut u64| {
            ops.extend(op(
                RGFX_OP_RENDER_MEMORY_BARRIER_WITH_SCOPE,
                RgfxOpMemoryBarrierWithScope {
                    scope: SCOPE_RENDER_TARGETS,
                    after: STAGE_FRAGMENT,
                    before: STAGE_FRAGMENT,
                }
                .as_bytes(),
            ));
            ops.extend(quad_ops(TARGET_ID, *quad));
            *quad += 1;
        };
        if reads && shape.read_first {
            read(&mut ops, &mut quad);
        }
        for draw in 0..shape.draws {
            let texture = TEXTURE_ID + (pass * shape.draws + draw) as u64 % TEXTURES;
            ops.extend(quad_ops(texture, quad));
            quad += 1;
        }
        if reads && !shape.read_first {
            read(&mut ops, &mut quad);
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

/// The 4 positions of quad `quad` of a frame of `shape`: a [`Shape::cover`] of the target, its
/// place moving across it from quad to quad.
fn quad_positions(shape: Shape, quad: u64) -> [f32; 16] {
    let side = 2.0 * shape.cover.sqrt();
    let room = 2.0 - side;
    let x0 = -1.0 + room * ((quad * 7 % 11) as f32 / 10.0);
    let y0 = -1.0 + room * ((quad * 5 % 13) as f32 / 12.0);
    let (x1, y1) = (x0 + side, y0 + side);
    [
        x0, y0, 0.0, 1.0, x1, y0, 0.0, 1.0, x0, y1, 0.0, 1.0, x1, y1, 0.0, 1.0,
    ]
}

/// A decoder on `P`'s system device holding a frame's objects, and the channel each completion
/// record comes on with its arrival.
struct Bench<P: ApiPlatform> {
    shape: Shape,
    device: P::Device,
    decoder: GraphicsDecoder<P>,
    completed: mpsc::Receiver<(RgfxCbComplete, Instant)>,
}

impl<P: ApiPlatform> Bench<P> {
    #[allow(clippy::disallowed_methods)] // a measurement: real time is what it measures
    fn new(shape: Shape) -> Bench<P> {
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let chunk = (shape.quads() * QUAD).next_multiple_of(0x4000);
        let region_size = RGFX_CONTEXT_ALLOCATIONS_OFFSET + chunk + RGFX_CURSOR_SPAN;
        let region = reims_vgpu_graphics::region::Region::new(region_size as usize).unwrap();
        for quad in 0..shape.quads() {
            let at = RGFX_CONTEXT_ALLOCATIONS_OFFSET + quad * QUAD;
            assert!(region.write(at, quad_positions(shape, quad).as_bytes()));
        }
        let mut decoder = GraphicsDecoder::<P>::new(Arc::clone(&platform), Arc::new(region));
        let (tx, completed) = mpsc::channel();
        let tx = reims_vgpu_vulkan::core::sync::Mutex::new(tx);
        decoder.set_completion_sink(Arc::new(move |m| {
            let _ = tx.lock().send((m, Instant::now()));
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
        let upload = RgfxBufferOverHostMemory {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
                size: size_of::<RgfxBufferOverHostMemory>() as u32,
                seqno: 0,
            },
            obj_id: CHUNK_ID,
            region_offset: RGFX_CONTEXT_ALLOCATIONS_OFFSET,
            length: chunk,
            ..RgfxBufferOverHostMemory::default()
        };
        assert_eq!(decoder.decode(&device, upload.as_bytes()).unwrap(), Ok(()));
        let target = platform.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: u64::from(shape.width),
            height: u64::from(shape.height),
            usage: TARGET_USAGE,
            ..TextureDesc::default()
        });
        let target = device.new_texture(OnFault::Die, &target).unwrap().unwrap();
        let (vertex, fragment) = (
            fixture::<P>(&device, "render_vbuf"),
            fixture::<P>(&device, "render_ftex"),
        );
        let pipeline = device
            .new_render_pipeline_state_full(&FullRenderPipelineDesc {
                colors: vec![(0, over(BGRA8))],
                ..FullRenderPipelineDesc::new(&vertex, Some(&fragment))
            })
            .unwrap()
            .unwrap();
        let sampled = platform.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: 64,
            height: 64,
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
            (PIPELINE_ID, ObjHandle::RenderPipeline(pipeline)),
            (SAMPLER_ID, ObjHandle::Sampler(sampler)),
            (CONSTANTS_ID, ObjHandle::Buffer(constants)),
        ] {
            assert_eq!(decoder.insert(id, handle).unwrap(), Ok(()));
        }
        // Its frames are a process's, as a queue's ring hands them, in context 0, whose objects
        // the ids above are.
        decoder.set_context(Some(0));
        if let Some(us) = shape.split_us {
            decoder.set_split_after(Some(Duration::from_micros(us)));
        }
        Bench {
            shape,
            device,
            decoder,
            completed,
        }
    }
}

/// Runs the frames of `bench` and prints the figures of those after the first [`WARM`]: each
/// one's median, its 90th percentile, and its lowest and highest.
#[allow(clippy::disallowed_methods)] // a measurement: real time is what it measures
fn measure<P: ApiPlatform>(name: &str, mut bench: Bench<P>) {
    let shape = bench.shape;
    let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
    // On a Mac, the thread asks for the performance cores, as a lane's thread is mostly on one.
    #[cfg(target_os = "macos")]
    // SAFETY: sets this thread's own class, which takes no pointer.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let mut walk_us = Vec::new();
    let mut walk_instructions = Vec::new();
    let mut latency_us = Vec::new();
    let mut gpu_us = Vec::new();
    let mut after_walk_us = Vec::new();
    for n in 0..WARM + shape.frames {
        let cb_obj_id = 0x1000 + n as u64;
        let message = frame(shape, cb_obj_id);
        let start = Instant::now();
        let before = counts();
        assert_eq!(
            bench.decoder.decode(&bench.device, &message).unwrap(),
            Ok(())
        );
        let spent = counts() - before;
        let walked = start.elapsed();
        let (record, at) = bench
            .completed
            .recv_timeout(Duration::from_secs(30))
            .expect("a frame never completed");
        assert_eq!(record.cb_obj_id, cb_obj_id);
        assert_eq!(record.status, 0, "a frame failed: {record:?}");
        if n < WARM {
            continue;
        }
        let latency = at - start;
        walk_us.push(walked.as_secs_f64() * 1e6);
        walk_instructions.push(spent.0[1] as f64);
        latency_us.push(latency.as_secs_f64() * 1e6);
        after_walk_us.push((latency - walked).as_secs_f64() * 1e6);
        gpu_us.push(record.gpu_end_ns.saturating_sub(record.gpu_start_ns) as f64 / 1e3);
    }
    println!(
        "{name}: {} frames of 1 command buffer of {} passes of {} draws over {} x {}, a draw {} \
         of the target, every {}th pass reading its target {}, parts walked {} us at least",
        shape.frames,
        shape.passes,
        shape.draws,
        shape.width,
        shape.height,
        shape.cover,
        shape.reads,
        if shape.read_first {
            "first, cleared"
        } else {
            "after its draws"
        },
        (shape.split_us).map_or(super::split::SPLIT_AFTER.as_micros(), u128::from)
    );
    for (figure, values) in [
        ("walk us", &mut walk_us),
        ("walk instructions", &mut walk_instructions),
        ("GPU span us", &mut gpu_us),
        ("after the walk us", &mut after_walk_us),
        ("latency us", &mut latency_us),
    ] {
        if values.iter().all(|&v| v == 0.0) {
            continue;
        }
        let p90 = {
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)]
        };
        let [median, lowest, highest] = spread(values);
        println!("  {figure:>18}: {median:10.1} p90 {p90:10.1} [{lowest:.1}, {highest:.1}]");
    }
}

/// On the Vulkan platform.
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU: prints a frame's times"]
fn on_vulkan_composite_cost() {
    let bench = Bench::<reims_vgpu_vulkan::VulkanPlatform>::new(Shape::chosen());
    measure("vulkan", bench);
}
