//! What one frame's work costs on the encoding thread, call by call, through this platform as
//! core drives it, against the runtime alone on the same device and objects, and (on macOS)
//! against the Metal platform through the same core traits: a measurement, run by hand in a
//! release build (`cargo nextest run -p reims-vgpu-vulkan --release --run-ignored only -E
//! 'test(frame_cost)' --no-capture`).
//!
//! A frame is command buffers, each of a render pass of draws, two compute encoders of
//! dispatches and a blit encoder of copies, with a completed handler, committed, then waited for:
//! one long command buffer ([`WIDE`]) or many short ones ([`MANY`]). Each draw binds what a
//! guest's draws bind: two vertex buffers at offsets that move, then a fragment texture and
//! sampler or a fragment buffer by the state, which changes every eighth draw. A pass's commands
//! go as one run of its encoder's (`RenderEncoder::encoding`), as the graphics lane's decoder
//! sends a pass's ops. Each dispatch
//! binds two buffers and runs 64 threads, in an encoder that made nothing resident and in one
//! that made [`RESIDENT`] buffers resident. The platform's device announces what the product's
//! does ([`product_device`]), and its calls run outside strict mode, as the product's.
//!
//! Beside each step's cost a call, each path prints what a frame costs as a whole: the CPU time
//! of the encoding thread and of the whole process (the runtime's threads and the driver's
//! included), and wall time, each run's mean a frame, as their median, lowest and highest over the
//! runs. A frame's work is waited for inside it, so the process's CPU is that frame's.
//!
//! The environment picks what runs: `FRAME_COST_RUNS` the runs (5), `FRAME_COST_SHAPE` one shape
//! (`wide` or `many`), `FRAME_COST_ONLY` one path, for a profile of it (`platform`, `runtime` or
//! `metal`), `FRAME_COST_OPTIONS` other device options ([`product_device`]), and
//! `FRAME_COST_RUNTIME_CHECK=1` a submission check on each of the runtime's command buffers
//! ([`runtime_checks`]).

use metal2vulkan::metal::{PrimitiveType, ResourceUsage};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::os::Clock as _;
use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_core::platform::resource::{Buffer as _, Library as _};
use reims_vgpu_core::platform::types::{MtlSize, NsRange};
use reims_vgpu_core::platform::values::{
    ColorAttachmentSet, RenderPipelineDesc, SamplerDesc, TextureDesc,
};
use reims_vgpu_core::raise::OnFault;

use crate::gpu::device::VulkanDevice;
use crate::platform::VulkanPlatform;

/// How a frame's work is cut: command buffers, each of a render pass of `draws` draws, two
/// compute encoders of `dispatches` dispatches and a blit encoder of `blits` copies.
#[derive(Debug, Clone, Copy)]
struct Shape {
    command_buffers: usize,
    draws: usize,
    dispatches: usize,
    blits: usize,
}

/// One command buffer of long encoders.
const WIDE: Shape = Shape {
    command_buffers: 1,
    draws: 512,
    dispatches: 64,
    blits: 32,
};

/// Many command buffers of short encoders, as a guest's frame of 60 to 100 command buffers is cut.
const MANY: Shape = Shape {
    command_buffers: 64,
    draws: 16,
    dispatches: 1,
    blits: 1,
};
/// Buffers the second compute encoder makes resident.
const RESIDENT: usize = 16;
/// Frames a run times, after [`WARM`] frames.
const FRAMES: usize = 40;
/// Frames run before the timed ones.
const WARM: usize = 5;

/// `MTLPixelFormatBGRA8Unorm`; `MTLTextureUsageShaderRead`, `RenderTarget`; `MTLLoadActionClear`,
/// `MTLStoreActionStore`; `MTLPrimitiveTypeTriangleStrip`.
const BGRA8: u64 = 80;
const READ: u64 = 1;
const TARGET: u64 = 4;
const CLEAR: u64 = 2;
const STORE: u64 = 1;
const STRIP: u64 = 4;

/// The steps a frame's calls are timed by, in the order printed.
const STEPS: [&str; 16] = [
    "command buffer",
    "render encoder",
    "set render state",
    "set vertex buffers",
    "set fragment textures",
    "set fragment samplers",
    "set fragment buffer",
    "draw",
    "end render",
    "compute encoder",
    "dispatch, nothing resident",
    "dispatch, 16 resident",
    "end compute",
    "blit copy",
    "end blit",
    "handler and commit",
];

/// Each step's total and count over a run.
#[derive(Debug, Clone, Copy, Default)]
struct Tally([(u128, u64); STEPS.len()]);

impl Tally {
    /// Runs `call`, timed as one of `step`.
    fn time<T>(&mut self, step: usize, call: impl FnOnce() -> T) -> T {
        let clock = crate::os::OsClock::new();
        let start = clock.now_mono();
        let out = call();
        let ns = clock.now_mono().saturating_duration_since(start).as_nanos();
        let (total, count) = &mut self.0[step];
        *total += ns;
        *count += 1;
        out
    }

    /// Each step's mean in nanoseconds a call.
    #[expect(clippy::as_conversions, reason = "a mean, printed")]
    fn means(&self) -> [f64; STEPS.len()] {
        self.0
            .map(|(total, count)| total as f64 / (count.max(1) as f64))
    }
}

/// CPU time in nanoseconds: this thread's, the process's, and the monotonic clock's (wall).
fn cpu_now() -> [u64; 3] {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, at: *mut Timespec) -> i32;
    }
    // `CLOCK_THREAD_CPUTIME_ID`, `CLOCK_PROCESS_CPUTIME_ID`, `CLOCK_MONOTONIC`.
    #[cfg(target_os = "macos")]
    const CLOCKS: [i32; 3] = [16, 12, 6];
    #[cfg(target_os = "linux")]
    const CLOCKS: [i32; 3] = [3, 2, 1];
    CLOCKS.map(|clock| {
        let mut at = Timespec { sec: 0, nsec: 0 };
        // SAFETY: `at` is a timespec, which `clock_gettime` writes.
        assert_eq!(unsafe { clock_gettime(clock, &mut at) }, 0);
        u64::try_from(at.sec).unwrap() * 1_000_000_000 + u64::try_from(at.nsec).unwrap()
    })
}

/// Runs `call`, adding the CPU it took ([`cpu_now`]) to `into`.
fn cpu_of<T>(into: &mut [u64; 3], call: impl FnOnce() -> T) -> T {
    let before = cpu_now();
    let out = call();
    for ((total, after), before) in into.iter_mut().zip(cpu_now()).zip(before) {
        *total += after - before;
    }
    out
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

/// Per-step means of `frames`: the lowest of each over runs of [`FRAMES`] frames, so a run a
/// neighbour slowed counts least.
fn best(runs: &[[f64; STEPS.len()]]) -> [f64; STEPS.len()] {
    let mut best = [f64::INFINITY; STEPS.len()];
    for run in runs {
        for (b, v) in best.iter_mut().zip(run) {
            *b = b.min(*v);
        }
    }
    best
}

/// The objects a frame binds, of one platform.
struct Objects<P: Platform> {
    queue: P::CommandQueue,
    /// The pass's colour attachment, held while the pass draws into it.
    _target: P::Texture,
    pass: P::RenderPassDescriptor,
    positions: P::Buffer,
    colours: P::Buffer,
    uniforms: P::Buffer,
    texture: P::Texture,
    sampler: P::SamplerState,
    states: [P::RenderPipeline; 2],
    kernel: P::ComputePipeline,
    input: P::Buffer,
    output: P::Buffer,
    resident: Vec<P::Buffer>,
    from: P::Buffer,
    to: P::Buffer,
}

/// A function of the suite's fixture `file`.
fn fixture<P: Platform>(d: &P::Device, file: &str, name: &str) -> P::Function {
    let path = crate::gpu::device::tests::fixture_shaders().join(format!("{file}.metallib"));
    let library = d
        .new_library(&std::fs::read(path).unwrap())
        .unwrap()
        .unwrap()
        .unwrap();
    library.new_function(OnFault::Die, name).unwrap().unwrap()
}

/// A Shared buffer of `length` bytes.
fn buffer<P: Platform>(d: &P::Device, length: u64) -> P::Buffer {
    let b = (d.new_buffer_of_length(OnFault::Die, length, 0).unwrap()).unwrap();
    // SAFETY: a Shared buffer's `length` bytes, which no command uses yet.
    unsafe {
        std::ptr::write_bytes(
            b.contents().as_ptr::<u8>(),
            0x3c,
            length.try_into().unwrap(),
        )
    };
    b
}

/// A 2D BGRA8 texture of `usage`, `side` texels a side.
fn texture<P: Platform>(p: &P, d: &P::Device, usage: u64, side: u64) -> P::Texture {
    let desc = p.texture_descriptor(&TextureDesc {
        pixel_format: BGRA8,
        width: side,
        height: side,
        usage,
        ..TextureDesc::default()
    });
    d.new_texture(OnFault::Die, &desc).unwrap().unwrap()
}

impl<P: Platform> Objects<P> {
    fn new(p: &P, d: &P::Device) -> Objects<P> {
        let target = texture(p, d, TARGET | READ, 512);
        let pass = p.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&target)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(OnFault::Die, 0, &attachment)
            .unwrap();
        let vertex = fixture::<P>(d, "render_vbuf", "render_vbuf");
        let state = |file: &str| {
            let fragment = fixture::<P>(d, file, file);
            d.new_render_pipeline_state(&RenderPipelineDesc {
                label: None,
                vertex: &vertex,
                fragment: &fragment,
                color0_pixel_format: BGRA8,
                blend_over: false,
                resource_index: None,
            })
            .unwrap()
            .unwrap()
        };
        let add = fixture::<P>(d, "compute_add", "add_one");
        Objects {
            queue: d.new_command_queue().unwrap(),
            _target: target,
            pass,
            positions: buffer::<P>(d, 4096),
            colours: buffer::<P>(d, 4096),
            uniforms: buffer::<P>(d, 65536),
            texture: texture(p, d, READ, 64),
            sampler: d
                .new_sampler_state(&SamplerDesc::default())
                .unwrap()
                .unwrap(),
            states: [state("render_ftex"), state("render_fbuf")],
            kernel: (d
                .new_compute_pipeline_state_for_function(Some(&add))
                .unwrap())
            .unwrap(),
            input: buffer::<P>(d, 65536),
            output: buffer::<P>(d, 65536),
            resident: (0..RESIDENT).map(|_| buffer::<P>(d, 4096)).collect(),
            from: buffer::<P>(d, 65536),
            to: buffer::<P>(d, 65536),
        }
    }

    /// One frame of `shape` through the platform's core traits, timed into `t`: its command
    /// buffers committed, then waited for.
    fn frame(&self, shape: Shape, t: &mut Tally) {
        let committed: Vec<_> = (0..shape.command_buffers)
            .map(|_| self.command_buffer(shape, t))
            .collect();
        for (cb, rx) in committed {
            rx.recv().unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
        }
    }

    /// One command buffer of `shape`, committed, and its completed handler's channel.
    fn command_buffer(
        &self,
        shape: Shape,
        t: &mut Tally,
    ) -> (P::CommandBuffer, std::sync::mpsc::Receiver<()>) {
        let o = OnFault::Die;
        let cb = t.time(0, || self.queue.command_buffer(o).unwrap().unwrap());
        let e = t.time(1, || cb.render_encoder(o, &self.pass).unwrap().unwrap());
        let texture = [Some(&self.texture)];
        let sampler = [Some(&self.sampler)];
        let vertex = [Some(&self.positions), Some(&self.colours)];
        // The pass's commands, one run of the encoder's, as the decoder sends a pass's ops
        // (`RenderEncoder::encoding`).
        e.encoding(|| {
            for i in 0..shape.draws {
                let at = u64::try_from(i).unwrap();
                let textured = (i / 8) % 2 == 0;
                if i % 8 == 0 {
                    let state = Some(&self.states[usize::from(!textured)]);
                    let cmd = RenderCmd::SetRenderPipelineState {
                        pipeline_state: state,
                    };
                    t.time(2, || e.encode(o, cmd).unwrap());
                }
                let cmd = RenderCmd::SetVertexBuffers {
                    buffers: &vertex,
                    offsets: &[0, (at % 64) * 16],
                    range: NsRange::new(0, 2),
                };
                t.time(3, || e.encode(o, cmd).unwrap());
                if textured {
                    let cmd = RenderCmd::SetFragmentTextures {
                        textures: &texture,
                        range: NsRange::new(0, 1),
                    };
                    t.time(4, || e.encode(o, cmd).unwrap());
                    let cmd = RenderCmd::SetFragmentSamplerStates {
                        samplers: &sampler,
                        range: NsRange::new(0, 1),
                    };
                    t.time(5, || e.encode(o, cmd).unwrap());
                } else {
                    let cmd = RenderCmd::SetFragmentBuffer {
                        buffer: Some(&self.uniforms),
                        offset: (at % 256) * 256,
                        index: 0,
                    };
                    t.time(6, || e.encode(o, cmd).unwrap());
                }
                let cmd = RenderCmd::DrawPrimitives {
                    primitive_type: STRIP,
                    vertex_start: 0,
                    vertex_count: 4,
                };
                t.time(7, || e.encode(o, cmd).unwrap());
            }
        });
        t.time(8, || e.encode(o, RenderCmd::EndEncoding).unwrap());
        let resident: Vec<_> = (self.resident.iter())
            .map(|b| Some(ResourceRef::Buffer(b)))
            .collect();
        for (step, used) in [(10, &resident[..0]), (11, &resident[..])] {
            let e = t.time(9, || cb.compute_encoder(o).unwrap().unwrap());
            let cmds = [
                ComputeCmd::SetComputePipelineState {
                    state: Some(&self.kernel),
                },
                ComputeCmd::UseResources {
                    resources: used,
                    usage: 1,
                },
            ];
            for cmd in cmds {
                e.encode(o, cmd).unwrap();
            }
            for i in 0..shape.dispatches {
                let at = u64::try_from(i).unwrap() * 256;
                t.time(step, || {
                    let set = ComputeCmd::SetBuffers {
                        buffers: &[Some(&self.output), Some(&self.input)],
                        offsets: &[at, at],
                        range: NsRange::new(0, 2),
                    };
                    e.encode(o, set).unwrap();
                    let dispatch = ComputeCmd::DispatchThreads {
                        threads_per_grid: MtlSize::new(64, 1, 1),
                        threads_per_threadgroup: MtlSize::new(64, 1, 1),
                    };
                    e.encode(o, dispatch).unwrap();
                });
            }
            t.time(12, || e.encode(o, ComputeCmd::EndEncoding).unwrap());
        }
        let blit = cb.blit_encoder(o).unwrap().unwrap();
        for i in 0..shape.blits {
            let at = u64::try_from(i).unwrap() * 1024;
            let cmd = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&self.from),
                source_offset: at,
                destination_buffer: Some(&self.to),
                destination_offset: at,
                size: 1024,
            };
            t.time(13, || blit.encode(o, cmd).unwrap());
        }
        t.time(14, || blit.encode(o, BlitCmd::EndEncoding).unwrap());
        let (tx, rx) = std::sync::mpsc::channel();
        t.time(15, || {
            let handler = Box::new(move |_| tx.send(()).unwrap());
            cb.add_completed_handler(o, handler).unwrap();
            cb.commit(o).unwrap();
        });
        (cb, rx)
    }
}

/// The same frame through the runtime alone, on the objects of `o`, timed into `t`.
fn runtime_frame(
    o: &Objects<VulkanPlatform>,
    queue: &rt::CommandQueue,
    shape: Shape,
    t: &mut Tally,
) {
    let (pass, _) = o.pass.pass();
    let committed: Vec<_> = (0..shape.command_buffers)
        .map(|_| runtime_command_buffer(o, queue, &pass, shape, t))
        .collect();
    for (cb, rx) in committed {
        rx.recv().unwrap();
        cb.wait_until_completed().unwrap();
    }
}

/// One command buffer of `shape` through the runtime alone, committed, and its completed
/// handler's channel.
fn runtime_command_buffer(
    o: &Objects<VulkanPlatform>,
    queue: &rt::CommandQueue,
    pass: &rt::RenderPassDescriptor,
    shape: Shape,
    t: &mut Tally,
) -> (rt::CommandBuffer, std::sync::mpsc::Receiver<()>) {
    let cb = t.time(0, || queue.command_buffer().unwrap());
    let mut e = t.time(1, || cb.render_command_encoder(pass).unwrap());
    let texture = [Some(&o.texture.raw)];
    let sampler: [Option<&rt::SamplerState>; 1] = [Some(&o.sampler)];
    let vertex = [Some(&o.positions.raw), Some(&o.colours.raw)];
    for i in 0..shape.draws {
        let at = u64::try_from(i).unwrap();
        let textured = (i / 8) % 2 == 0;
        if i % 8 == 0 {
            let state = &o.states[usize::from(!textured)].raw;
            t.time(2, || e.set_render_pipeline_state(state).unwrap());
        }
        t.time(3, || {
            e.set_vertex_buffers(&vertex, &[0, (at % 64) * 16], 0)
                .unwrap();
        });
        if textured {
            t.time(4, || e.set_fragment_textures(&texture, 0).unwrap());
            t.time(5, || {
                e.set_fragment_sampler_states(&sampler, None, 0).unwrap()
            });
        } else {
            t.time(6, || {
                e.set_fragment_buffer(Some(&o.uniforms.raw), (at % 256) * 256, 0)
                    .unwrap();
            });
        }
        t.time(7, || {
            e.draw_primitives(PrimitiveType::TriangleStrip, 0, 4, 1, 0)
                .unwrap();
        });
    }
    t.time(8, || e.end_encoding().unwrap());
    let resident: Vec<_> = (o.resident.iter())
        .map(|b| rt::ResourceRef::Buffer(&b.raw))
        .collect();
    for (step, used) in [(10, &resident[..0]), (11, &resident[..])] {
        let mut e = t.time(9, || cb.compute_command_encoder().unwrap());
        e.set_compute_pipeline_state(&o.kernel.raw).unwrap();
        if !used.is_empty() {
            e.use_resources(used, ResourceUsage::READ).unwrap();
        }
        for i in 0..shape.dispatches {
            let at = u64::try_from(i).unwrap() * 256;
            t.time(step, || {
                let buffers = [Some(&o.output.raw), Some(&o.input.raw)];
                e.set_buffers(&buffers, &[at, at], 0).unwrap();
                let (grid, group) = (rt::Size::new(64, 1, 1), rt::Size::new(64, 1, 1));
                e.dispatch_threads(grid, group).unwrap();
            });
        }
        t.time(12, || e.end_encoding().unwrap());
    }
    let mut blit = cb.blit_command_encoder().unwrap();
    for i in 0..shape.blits {
        let at = u64::try_from(i).unwrap() * 1024;
        t.time(13, || {
            blit.copy_from_buffer_to_buffer(&o.from.raw, at, &o.to.raw, at, 1024)
                .unwrap();
        });
    }
    t.time(14, || blit.end_encoding().unwrap());
    let (tx, rx) = std::sync::mpsc::channel();
    t.time(15, || {
        let handler = Box::new(move |_: &rt::CommandBuffer| tx.send(()).unwrap());
        cb.add_completed_handler(handler).unwrap();
        if runtime_checks() {
            cb.set_submission_check(std::sync::Arc::new(|_| None))
                .unwrap();
        }
        cb.commit().unwrap();
    });
    (cb, rx)
}

/// Whether the runtime alone commits each command buffer with a submission check that passes
/// every submission (`FRAME_COST_RUNTIME_CHECK=1`), as the platform commits each with its own
/// ([`crate::gpu::backing::submission_check`]): what the check costs the runtime, apart from what
/// the platform's check does.
fn runtime_checks() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FRAME_COST_RUNTIME_CHECK").is_some_and(|v| v == "1")
    });
    *ON
}

/// A device of the system default's physical device made with the product's options: the
/// platform's, but announcing at every occurrence only what the product's device does (the kinds
/// a call hears and those that die by default), where a test process's device announces every
/// kind that dies, as in strict mode ([`crate::platform`]'s `announced_each`).
fn product_device() -> VulkanDevice {
    use crate::answer::{DIES_BY_DEFAULT, HEARD};
    let mut options = crate::platform::device_options().clone();
    options.announce_each = DIES_BY_DEFAULT.iter().chain(&HEARD).copied().collect();
    // What the runtime costs under other options, for a measurement of the runtime alone:
    // `each` announcing every occurrence of nothing, `sink` without the message sink, `default`
    // the runtime's default options.
    match std::env::var("FRAME_COST_OPTIONS").as_deref() {
        Ok("each") => options.announce_each.clear(),
        Ok("sink") => options.messages = None,
        Ok("default") => options = rt::DeviceOptions::default(),
        _ => {}
    }
    crate::gpu::device::tests::device_of_options(&options)
}

/// Prints, in nanoseconds a call, each step's best mean over runs of [`FRAMES`] frames: through
/// this platform as core drives it, through the runtime alone on the same objects, and through
/// the Metal platform on macOS, frames of the three alternating.
#[test]
#[ignore = "a measurement, run by hand in a release build: prints each step's cost"]
fn frame_cost() {
    let runs: usize = std::env::var("FRAME_COST_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    // One path alone, for a profile of it: `platform`, `runtime` or `metal`.
    let only = std::env::var("FRAME_COST_ONLY").ok();
    let only = only.as_deref();
    let p = crate::platform::tests::platform();
    let d = product_device();
    let ours = crate::answer::relaxed(|| Objects::new(&p, &d));
    let raw_queue = d.raw.new_command_queue().unwrap();
    #[cfg(target_os = "macos")]
    let metal = reims_vgpu_metal::MetalPlatform::system_default_device().map(|md| {
        use crate::gpu::device::tests::metal_tests::NoLauncher;
        let mp = reims_vgpu_metal::MetalPlatform::new(Box::new(NoLauncher));
        let objects = Objects::new(&mp, &md);
        (mp, md, objects)
    });
    // One shape alone: `wide` or `many`.
    let shape = std::env::var("FRAME_COST_SHAPE").ok();
    let shapes = [("wide", WIDE), ("many", MANY)];
    for (name, shape) in shapes
        .into_iter()
        .filter(|(name, _)| shape.as_deref().is_none_or(|s| s == *name))
    {
        let mut seen: [Vec<[f64; STEPS.len()]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        // Each run's CPU a frame, by path: the encoding thread's, the process's, and wall time.
        let mut cpu_runs: [Vec<[f64; 3]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut counts = Tally::default();
        for _ in 0..runs {
            let mut tallies = [Tally::default(); 3];
            let mut cpu = [[0u64; 3]; 3];
            for frame in 0..WARM + FRAMES {
                let mut t = [Tally::default(); 3];
                let mut c = [[0u64; 3]; 3];
                if only.is_none_or(|o| o == "platform") {
                    cpu_of(&mut c[0], || {
                        crate::answer::relaxed(|| ours.frame(shape, &mut t[0]));
                    });
                }
                if only.is_none_or(|o| o == "runtime") {
                    cpu_of(&mut c[1], || {
                        runtime_frame(&ours, &raw_queue, shape, &mut t[1]);
                    });
                }
                #[cfg(target_os = "macos")]
                if let Some((_, _, objects)) = &metal
                    && only.is_none_or(|o| o == "metal")
                {
                    cpu_of(&mut c[2], || objects.frame(shape, &mut t[2]));
                }
                if frame >= WARM {
                    for (cpu, c) in cpu.iter_mut().zip(c) {
                        for (s, v) in cpu.iter_mut().zip(c) {
                            *s += v;
                        }
                    }
                    for (tally, t) in tallies.iter_mut().zip(t) {
                        for (s, (total, count)) in tally.0.iter_mut().zip(t.0) {
                            s.0 += total;
                            s.1 += count;
                        }
                    }
                }
                if frame == 0 {
                    let calls = |t: &Tally| t.0.iter().map(|&(_, n)| n).sum::<u64>();
                    counts = t.into_iter().max_by_key(calls).unwrap_or_default();
                }
            }
            for (seen, tally) in seen.iter_mut().zip(tallies) {
                seen.push(tally.means());
            }
            #[expect(clippy::as_conversions, reason = "a mean, printed")]
            let us_a_frame = |ns: u64| ns as f64 / (FRAMES as f64) / 1000.0;
            for (runs, cpu) in cpu_runs.iter_mut().zip(cpu) {
                runs.push(cpu.map(us_a_frame));
            }
        }
        let [platform, runtime, metal_platform] = seen.map(|runs| best(&runs));
        println!(
            "{name} frames of {shape:?}, ns a call, best of {runs} runs of {FRAMES} frames: \
             platform, runtime alone, ours above it, Metal platform"
        );
        for (i, step) in STEPS.iter().enumerate() {
            println!(
                "  {step:>28}: {:>9.0} {:>9.0} {:>9.0} {:>9.0}",
                platform[i],
                runtime[i],
                platform[i] - runtime[i],
                metal_platform[i]
            );
        }
        #[expect(clippy::as_conversions, reason = "a sum, printed")]
        let frame = |m: &[f64; STEPS.len()]| -> f64 {
            m.iter().zip(counts.0).map(|(v, (_, n))| v * n as f64).sum()
        };
        println!(
            "  a frame's calls: platform {:.0} us, runtime alone {:.0} us, Metal platform {:.0} us",
            frame(&platform) / 1000.0,
            frame(&runtime) / 1000.0,
            frame(&metal_platform) / 1000.0
        );
        let paths = ["platform", "runtime alone", "Metal platform"];
        for (path, runs) in paths.iter().zip(&cpu_runs) {
            if runs.is_empty() || runs.iter().all(|r| r[2] == 0.0) {
                continue;
            }
            let [thread, process, wall] = [0, 1, 2].map(|i| {
                let mut values: Vec<f64> = runs.iter().map(|r| r[i]).collect();
                spread(&mut values)
            });
            let show = |[median, lowest, highest]: [f64; 3]| {
                format!("{median:.0} [{lowest:.0}, {highest:.0}]")
            };
            println!(
                "  {path:>14} us a frame, median [lowest, highest] of {n}: encoding thread CPU \
                 {}, process CPU {}, wall {}",
                show(thread),
                show(process),
                show(wall),
                n = runs.len(),
            );
        }
    }
}
