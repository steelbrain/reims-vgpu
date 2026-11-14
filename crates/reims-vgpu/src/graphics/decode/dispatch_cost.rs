//! What a compute dispatch costs the thread that decodes it, through the decoder as a queue's ring
//! hands it a `COMMAND_BUFFER`, against the same commands sent through the platform's core traits
//! and through what the platform drives alone, metal2vulkan's runtime. A measurement, run by hand
//! in a release build, in a potluck job holding the host's GPU (`--gpu`: `cargo nextest run -p
//! reims-vgpu --release --no-default-features --features reims-vgpu/vulkan --run-ignored only -E
//! 'test(dispatch_cost)' --no-capture`).
//!
//! Its default shape is Geekbench 7's Fluid Simulation as the trace of a timed pass on a Radeon
//! 780M counted it (perfscout4's gbr-780m, its 20 big command buffers): a command buffer of 2,600
//! compute encoders of one dispatch each. The trace names neither the dispatches' bindings nor
//! their sizes, so each encoder here binds what a kernel over a grid of textures with its
//! parameters as bytes would: its pipeline state, two textures, a sampler, an output buffer and
//! 16 bytes of `setBytes:`, which the bundle sends as a `setBuffers:` of one upload chunk at its
//! next 256-byte offset (a 64 KiB `BUFFER_OVER_HOST_MEMORY` chunk of the region, 256 dispatches a
//! chunk), then dispatches one threadgroup of 32 threads ([`Shape`]). The command buffers of a
//! repetition go back to back and are then waited for. Each path's figures are the thread's CPU
//! time, instructions and cycles a dispatch, from the first command buffer's begin to the last
//! one's commit: the decoder's include the message's take, its walk and its commit with the
//! decoder's handlers; the gaps between the paths are the decoder's share and the platform's. The
//! runtime's path commits as the platform does (`commit_and_submit`), submitting on the thread.
//!
//! The environment picks what runs: `DISPATCH_COST_SHAPE` the shape (`fluid`, the default; `one`,
//! the same dispatches in one encoder, each after its bindings, whose gap to `fluid` is what an
//! encoder a dispatch costs; or `small`, 64 command buffers of 5 encoders, the median of the pass's
//! other command buffers), `DISPATCH_COST_RUNS` the runs (5), `DISPATCH_COST_ONLY` one path, for a
//! profile of it (`decoder`, `platform` or `runtime`), and `DISPATCH_COST_SECONDS` a time to keep
//! running that path for, so a sampling profiler can attach.

use std::sync::Arc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use reims_vgpu_graphics::ops::compute::{
    RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, RGFX_OP_COMPUTE_SET_BUFFERS,
    RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RGFX_OP_COMPUTE_SET_SAMPLER_STATES,
    RGFX_OP_COMPUTE_SET_TEXTURE, RgfxComputeBufferBinding, RgfxComputeOpDispatchThreadgroups,
    RgfxComputeOpSetBuffers, RgfxComputeOpSetComputePipelineState, RgfxComputeOpSetSamplerStates,
    RgfxComputeOpSetTexture, RgfxComputeSamplerBinding,
};
use reims_vgpu_graphics::ops::{
    RGFX_COMMAND_BUFFER_OPS_OFFSET, RGFX_OP_ENCODER_BEGIN_COMPUTE, RGFX_OP_ENCODER_END,
    RgfxCommandBuffer, RgfxOpHeader,
};
use reims_vgpu_graphics::wire::{
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_QUEUE_CREATE,
    RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN, RgfxBufferOverHostMemory, RgfxMsgHeader,
    RgfxQueueCreate,
};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::Library as _;
use reims_vgpu_vulkan::core::platform::types::{MtlSize, NsRange};
use reims_vgpu_vulkan::core::platform::values::{ComputePassDesc, SamplerDesc, TextureDesc};
use reims_vgpu_vulkan::core::raise::OnFault;
use zerocopy::IntoBytes;

use super::draw_cost::{Counts, FIGURES, counts, spread, test_shaders};
use super::{GraphicsDecoder, ObjHandle};
use crate::platform::ApiPlatform;

/// How a repetition's work is cut: `cbs` command buffers, committed back to back and then waited
/// for, each of `dispatches` dispatches in `encoders` compute encoders of as many each.
#[derive(Debug, Clone, Copy)]
struct Shape {
    cbs: usize,
    dispatches: usize,
    encoders: usize,
}

impl Shape {
    /// Fluid Simulation's big command buffers: 2,600 encoders of one dispatch.
    const FLUID: Shape = Shape {
        cbs: 1,
        dispatches: 2600,
        encoders: 2600,
    };
    /// The same dispatches in one encoder.
    const ONE: Shape = Shape {
        encoders: 1,
        ..Shape::FLUID
    };
    /// The pass's other command buffers: 64 of 5 encoders of one dispatch.
    const SMALL: Shape = Shape {
        cbs: 64,
        dispatches: 5,
        encoders: 5,
    };

    /// The shape `DISPATCH_COST_SHAPE` names: `one`, `small` or, by default, `fluid`.
    fn chosen() -> Shape {
        match std::env::var("DISPATCH_COST_SHAPE").as_deref() {
            Ok("one") => Shape::ONE,
            Ok("small") => Shape::SMALL,
            _ => Shape::FLUID,
        }
    }

    /// The upload chunks a command buffer's bytes take.
    fn chunks_a_cb(self) -> u64 {
        (self.dispatches as u64 * ALIGN).div_ceil(CHUNK)
    }

    /// The chunks the region holds: a repetition's command buffers each take their own.
    fn chunks(self) -> u64 {
        self.chunks_a_cb() * self.cbs as u64
    }
}

/// Repetitions a run times, after [`WARM`] repetitions.
const REPS: usize = 6;
const WARM: usize = 2;
/// An upload chunk's bytes, and the alignment of each dispatch's bytes in it (the bundle's
/// `kUploadChunkSize` and `kUploadAlign`).
const CHUNK: u64 = 0x1_0000;
const ALIGN: u64 = 256;
/// The object ids: the queue, the pipeline, the two textures, the sampler, the output buffer,
/// then the chunks from [`CHUNK_ID`].
const QUEUE_ID: u64 = 1;
const PIPELINE_ID: u64 = 2;
const TEXTURE_ID: u64 = 3;
const SAMPLER_ID: u64 = 5;
const OUTPUT_ID: u64 = 6;
const CHUNK_ID: u64 = 0x100;
/// `MTLPixelFormatRGBA8Unorm`, `MTLTextureUsageShaderRead`, and a texture's side.
const RGBA8: u64 = 70;
const SAMPLED_USAGE: u64 = 1;
const SIDE: u64 = 64;
/// The threads of the threadgroup each dispatch runs.
const THREADS: u64 = 32;

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

/// The chunk and offset of the bytes of dispatch `dispatch` of command buffer `cb`.
fn upload_at(shape: Shape, cb: usize, dispatch: usize) -> (u64, u64) {
    let at = dispatch as u64 * ALIGN;
    (
        CHUNK_ID + cb as u64 * shape.chunks_a_cb() + at / CHUNK,
        at % CHUNK,
    )
}

/// One compute op of a command buffer's, by the ids of the objects it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// The encoder's begin.
    Begin,
    /// `setComputePipelineState:`.
    Pipeline,
    /// `setTexture:atIndex:` of a texture at an index.
    Texture(u64, u64),
    /// `setSamplerState:atIndex:` at index 0, which the bundle sends as a range of one.
    Sampler,
    /// `setBuffer:offset:atIndex:` (or `setBytes:length:atIndex:`, an upload chunk at its offset),
    /// which the bundle sends as a `setBuffers:` of one.
    Buffer(u64, u64, u64),
    /// `dispatchThreadgroups:threadsPerThreadgroup:` of one threadgroup.
    Dispatch,
    /// The encoder's end.
    End,
}

/// Command buffer `cb`'s ops, of a repetition of `shape`: each encoder's begin, then each of its
/// dispatches after its bindings, then its end.
fn steps(shape: Shape, cb: usize) -> Vec<Step> {
    assert!(shape.dispatches.is_multiple_of(shape.encoders));
    let a_encoder = shape.dispatches / shape.encoders;
    let mut steps = Vec::new();
    for dispatch in 0..shape.dispatches {
        if dispatch.is_multiple_of(a_encoder) {
            steps.push(Step::Begin);
        }
        let (chunk, offset) = upload_at(shape, cb, dispatch);
        steps.extend([
            Step::Pipeline,
            Step::Texture(TEXTURE_ID, 0),
            Step::Texture(TEXTURE_ID + 1, 1),
            Step::Sampler,
            Step::Buffer(OUTPUT_ID, 0, 0),
            Step::Buffer(chunk, offset, 1),
            Step::Dispatch,
        ]);
        if (dispatch + 1).is_multiple_of(a_encoder) {
            steps.push(Step::End);
        }
    }
    steps
}

/// The op the bundle sends for `step`.
fn step_op(step: Step) -> Vec<u8> {
    match step {
        Step::Begin => op(RGFX_OP_ENCODER_BEGIN_COMPUTE, &[]),
        Step::End => op(RGFX_OP_ENCODER_END, &[]),
        Step::Pipeline => op(
            RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
            RgfxComputeOpSetComputePipelineState {
                pipeline_obj_id: PIPELINE_ID,
            }
            .as_bytes(),
        ),
        Step::Texture(id, index) => op(
            RGFX_OP_COMPUTE_SET_TEXTURE,
            RgfxComputeOpSetTexture {
                texture_obj_id: id,
                index,
            }
            .as_bytes(),
        ),
        Step::Sampler => {
            let mut body = RgfxComputeOpSetSamplerStates {
                location: 0,
                length: 1,
            }
            .as_bytes()
            .to_vec();
            body.extend_from_slice(
                RgfxComputeSamplerBinding {
                    sampler_obj_id: SAMPLER_ID,
                }
                .as_bytes(),
            );
            op(RGFX_OP_COMPUTE_SET_SAMPLER_STATES, &body)
        }
        Step::Buffer(id, offset, index) => {
            let mut body = RgfxComputeOpSetBuffers {
                location: index,
                length: 1,
            }
            .as_bytes()
            .to_vec();
            body.extend_from_slice(
                RgfxComputeBufferBinding {
                    buffer_obj_id: id,
                    offset,
                }
                .as_bytes(),
            );
            op(RGFX_OP_COMPUTE_SET_BUFFERS, &body)
        }
        Step::Dispatch => op(
            RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS,
            RgfxComputeOpDispatchThreadgroups {
                threadgroups_width: 1,
                threadgroups_height: 1,
                threadgroups_depth: 1,
                threads_width: THREADS,
                threads_height: 1,
                threads_depth: 1,
            }
            .as_bytes(),
        ),
    }
}

/// The `COMMAND_BUFFER` message of command buffer `cb` of a repetition of `shape`, as the bundle
/// sends its ops ([`steps`]).
fn command_buffer(shape: Shape, cb: usize, cb_obj_id: u64) -> Vec<u8> {
    let ops: Vec<u8> = steps(shape, cb).into_iter().flat_map(step_op).collect();
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

/// A decoder on `P`'s system device, holding the objects [`command_buffer`] names, and the
/// channel its completions come on.
struct Bench<P: ApiPlatform> {
    shape: Shape,
    device: P::Device,
    decoder: GraphicsDecoder<P>,
    completed: mpsc::Receiver<u64>,
}

impl<P: ApiPlatform> Bench<P> {
    fn new(shape: Shape) -> Bench<P> {
        let o = OnFault::Die;
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let region_size =
            RGFX_CONTEXT_ALLOCATIONS_OFFSET + shape.chunks() * CHUNK + RGFX_CURSOR_SPAN;
        let region = crate::platform::test_region(&*platform, region_size as usize);
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
        let library = device
            .new_library(&std::fs::read(test_shaders().join("compute_sample.metallib")).unwrap())
            .unwrap()
            .unwrap()
            .unwrap();
        let kernel = library.new_function(o, "sample_k").unwrap().unwrap();
        let pipeline = device
            .new_compute_pipeline_state_for_function(Some(&kernel))
            .unwrap()
            .unwrap();
        let sampled = platform.texture_descriptor(&TextureDesc {
            pixel_format: RGBA8,
            width: SIDE,
            height: SIDE,
            usage: SAMPLED_USAGE,
            ..TextureDesc::default()
        });
        for n in 0..2 {
            let texture = device.new_texture(o, &sampled).unwrap().unwrap();
            let handle = ObjHandle::Texture(texture);
            assert_eq!(decoder.insert(TEXTURE_ID + n, handle).unwrap(), Ok(()));
        }
        let sampler = device
            .new_sampler_state(&SamplerDesc::default())
            .unwrap()
            .unwrap();
        let output = device
            .new_buffer_of_length(o, THREADS * 4, 0)
            .unwrap()
            .unwrap();
        for (id, handle) in [
            (PIPELINE_ID, ObjHandle::ComputePipeline(pipeline)),
            (SAMPLER_ID, ObjHandle::Sampler(sampler)),
            (OUTPUT_ID, ObjHandle::Buffer(output)),
        ] {
            assert_eq!(decoder.insert(id, handle).unwrap(), Ok(()));
        }
        Bench {
            shape,
            device,
            decoder,
            completed,
        }
    }

    /// The queue.
    fn queue(&self) -> &P::CommandQueue {
        let Some(ObjHandle::CommandQueue(queue)) = self.decoder.object(QUEUE_ID) else {
            panic!("no queue");
        };
        queue
    }

    /// Each command buffer's ops of a repetition, their objects looked up: made before a count
    /// starts.
    fn bound(&self) -> Vec<Vec<Bound<'_, P>>> {
        let named = |step: Step| match (step, self.decoder.object(match step {
            Step::Pipeline => PIPELINE_ID,
            Step::Texture(id, _) | Step::Buffer(id, _, _) => id,
            Step::Sampler => SAMPLER_ID,
            Step::Begin | Step::Dispatch | Step::End => 0,
        })) {
            (Step::Begin, _) => Bound::Begin,
            (Step::End, _) => Bound::End,
            (Step::Dispatch, _) => Bound::Dispatch,
            (Step::Pipeline, Some(ObjHandle::ComputePipeline(p))) => Bound::Pipeline(p),
            (Step::Texture(_, index), Some(ObjHandle::Texture(t))) => Bound::Texture(t, index),
            (Step::Sampler, Some(ObjHandle::Sampler(s))) => Bound::Sampler(s),
            (Step::Buffer(_, offset, index), Some(ObjHandle::Buffer(b))) => {
                Bound::Buffer(b, offset, index)
            }
            (step, _) => panic!("no object for {step:?}"),
        };
        (0..self.shape.cbs)
            .map(|cb| steps(self.shape, cb).into_iter().map(named).collect())
            .collect()
    }

    /// One repetition through the decoder: the thread's CPU time from the first message's take
    /// to the last one's commit.
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

    /// One repetition through the platform's core traits, on the decoder's objects, each op's
    /// command as the decoder sends it.
    fn through_platform(&self) -> Counts {
        let o = OnFault::Die;
        let queue = self.queue();
        let bound = self.bound();
        let (tx, rx) = mpsc::channel();
        let start = counts();
        for cb_bound in &bound {
            let cb = queue.command_buffer(o).unwrap().unwrap();
            let mut enc = None;
            for &b in cb_bound {
                if let Bound::Begin = b {
                    enc = cb
                        .compute_encoder_for_pass(&ComputePassDesc::default())
                        .unwrap();
                    continue;
                }
                let e = enc.as_ref().unwrap();
                match b {
                    Bound::Begin => unreachable!(),
                    Bound::End => {
                        e.encode(o, ComputeCmd::EndEncoding).unwrap();
                        enc = None;
                    }
                    Bound::Pipeline(p) => e
                        .encode(o, ComputeCmd::SetComputePipelineState { state: Some(p) })
                        .unwrap(),
                    Bound::Texture(t, index) => e
                        .encode(
                            o,
                            ComputeCmd::SetTexture {
                                texture: Some(t),
                                index,
                            },
                        )
                        .unwrap(),
                    Bound::Sampler(s) => e
                        .encode(
                            o,
                            ComputeCmd::SetSamplerStates {
                                samplers: &[Some(s)],
                                range: NsRange::new(0, 1),
                            },
                        )
                        .unwrap(),
                    Bound::Buffer(buffer, offset, index) => e
                        .encode(
                            o,
                            ComputeCmd::SetBuffers {
                                buffers: &[Some(buffer)],
                                offsets: &[offset],
                                range: NsRange::new(index, 1),
                            },
                        )
                        .unwrap(),
                    Bound::Dispatch => e
                        .encode(
                            o,
                            ComputeCmd::DispatchThreadgroups {
                                threadgroups_per_grid: MtlSize::new(1, 1, 1),
                                threads_per_threadgroup: MtlSize::new(THREADS, 1, 1),
                            },
                        )
                        .unwrap(),
                }
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

/// A compute op of [`Step`]'s with its objects looked up.
enum Bound<'a, P: Platform> {
    Begin,
    Pipeline(&'a P::ComputePipeline),
    Texture(&'a P::Texture, u64),
    Sampler(&'a P::SamplerState),
    Buffer(&'a P::Buffer, u64, u64),
    Dispatch,
    End,
}

impl<P: Platform> Clone for Bound<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: Platform> Copy for Bound<'_, P> {}

/// The runtime alone, on the Vulkan platform's objects of `bench`: the calls the platform makes
/// of it for each command.
fn through_runtime(bench: &Bench<reims_vgpu_vulkan::VulkanPlatform>) -> Counts {
    use metal2vulkan_runtime as rt;
    let queue = bench.queue();
    let bound = bench.bound();
    let (tx, rx) = mpsc::channel();
    let start = counts();
    for cb_bound in &bound {
        let cb = queue.raw.command_buffer().unwrap();
        let mut enc = None;
        for &b in cb_bound {
            match b {
                Bound::Begin => {
                    enc = Some(cb.compute_command_encoder().unwrap());
                    continue;
                }
                Bound::End => {
                    enc.take().unwrap().end_encoding().unwrap();
                    continue;
                }
                _ => {}
            }
            let e = enc.as_mut().unwrap();
            match b {
                Bound::Begin | Bound::End => unreachable!(),
                Bound::Pipeline(p) => e.set_compute_pipeline_state(&p.raw),
                Bound::Texture(t, index) => {
                    e.set_texture(Some(t.texels("setTexture:atIndex:")), index as u32)
                }
                Bound::Sampler(s) => e.set_sampler_states(&[Some(&**s)], 0),
                Bound::Buffer(buffer, offset, index) => {
                    e.set_buffers(&[Some(&buffer.raw)], &[offset], index as u32)
                }
                Bound::Dispatch => e.dispatch_threadgroups(
                    rt::Size::new(1, 1, 1),
                    rt::Size::new(THREADS, 1, 1),
                ),
            }
            .unwrap();
        }
        let tx = tx.clone();
        cb.add_completed_handler(Box::new(move |_: &rt::CommandBuffer| tx.send(()).unwrap()))
            .unwrap();
        // As the platform commits: submitting on this thread.
        cb.commit_and_submit().unwrap();
    }
    let spent = counts() - start;
    for _ in 0..bench.shape.cbs {
        rx.recv_timeout(Duration::from_secs(30))
            .expect("a command buffer never completed");
    }
    spent
}

/// Runs the paths of `bench`, `runtime` what the platform drives alone, and prints each one's
/// figures a dispatch.
#[allow(clippy::disallowed_methods)] // a measurement: real time decides how long a profile runs
fn measure<P: ApiPlatform>(name: &str, mut bench: Bench<P>, runtime: fn(&Bench<P>) -> Counts) {
    let runs: usize = std::env::var("DISPATCH_COST_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let only = std::env::var("DISPATCH_COST_ONLY").ok();
    let only = only.as_deref();
    let seconds: Option<u64> = std::env::var("DISPATCH_COST_SECONDS")
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
    // dispatches would otherwise mix into the figures.
    #[cfg(target_os = "macos")]
    // SAFETY: sets this thread's own class, which takes no pointer.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let a_dispatch = |c: Counts, reps: u64| {
        c.0.map(|v| v as f64 / (reps * (shape.cbs * shape.dispatches) as u64) as f64)
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
        let [ns, instructions, cycles] = a_dispatch(spent, reps);
        println!(
            "{name}: {} for {seconds} s: {reps} repetitions, a dispatch {ns:.1} ns, \
             {instructions:.0} instructions, {cycles:.0} cycles",
            only.unwrap_or("decoder"),
        );
        return;
    }
    let paths = ["decoder", "platform", "runtime"];
    // Each path's figures a dispatch, by run.
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
            seen.push(a_dispatch(total, REPS as u64));
        }
    }
    println!(
        "{name}: a dispatch on the decoding thread, median [lowest, highest] of {runs} runs of \
         {REPS} repetitions of {} command buffers of {} dispatches in {} encoders, {} ops a \
         command buffer",
        shape.cbs,
        shape.dispatches,
        shape.encoders,
        steps(shape, 0).len()
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

/// The fluid shape holds an encoder a dispatch, each dispatch after its pipeline and its five
/// bindings, its bytes at its own 256-byte offset; the one shape holds the same ops in one encoder.
#[test]
fn the_fluid_shape_holds_an_encoder_a_dispatch() {
    let fluid = steps(Shape::FLUID, 0);
    let count = |steps: &[Step], want: Step| steps.iter().filter(|s| **s == want).count();
    assert_eq!(count(&fluid, Step::Begin), 2600);
    assert_eq!(count(&fluid, Step::End), 2600);
    assert_eq!(count(&fluid, Step::Dispatch), 2600);
    assert_eq!(fluid.len(), 2600 * 9);
    assert_eq!(fluid[..9][0], Step::Begin);
    assert_eq!(fluid[..9][7], Step::Dispatch);
    assert_eq!(fluid[..9][6], Step::Buffer(CHUNK_ID, 0, 1));
    assert_eq!(fluid[9 * 257 + 6], Step::Buffer(CHUNK_ID + 1, ALIGN, 1));
    let one = steps(Shape::ONE, 0);
    assert_eq!(count(&one, Step::Begin), 1);
    assert_eq!(one.len(), 2600 * 7 + 2);
    assert_eq!(Shape::SMALL.chunks(), 64);
}

/// On the Vulkan platform: the decoder, the platform and the runtime alone.
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU: prints a dispatch's cost"]
fn on_vulkan_dispatch_cost() {
    println!(
        "the runtime's compute encoder: {} bytes; its render encoder: {} bytes",
        size_of::<metal2vulkan_runtime::ComputeCommandEncoder>(),
        size_of::<metal2vulkan_runtime::RenderCommandEncoder>()
    );
    let bench = Bench::<reims_vgpu_vulkan::VulkanPlatform>::new(Shape::chosen());
    measure("vulkan", bench, through_runtime);
}
