//! What a context's close costs the thread that carries it out
//! ([`GraphicsDecoder::drop_context`]), by the kind of object it lets go of. A run makes
//! `CLOSE_COST_COUNT` objects of one kind (1000 by default) and puts them in a context's table, as
//! their creates do, then closes the context, timing the thread's CPU time and the wall time
//! between. On the fake platform, whose objects cost next to nothing to let go of, the figures are
//! the decoder's own an object; on the Vulkan platform they add the platform's release
//! and its driver's. Each kind runs once to warm up, then [`REPS`] times, and the median of each
//! figure is printed. A measurement, run by hand in a release build, in a potluck job holding the
//! host's GPU (`--gpu`: `cargo nextest run -p reims-vgpu --release
//! --no-default-features --features reims-vgpu/vulkan --run-ignored only -E 'test(close_cost)'
//! --no-capture`); `CLOSE_COST_KINDS` names the kinds to run, a comma-separated list of
//! [`KINDS`]' names, every one by default.
//!
//! The kinds are those a closing process holds most of: buffers of the GPU's memory and over the
//! region's (the bundle's `BUFFER_OVER_HOST_MEMORY`, which imports the guest's pages on the Vulkan
//! platform), textures, samplers, render and compute pipeline states, events and command queues.

use std::sync::Arc;
use std::time::Instant;

use reims_vgpu_graphics::wire::{
    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_QUEUE_CREATE, RGFX_CONTEXT_ALLOCATIONS_OFFSET,
    RGFX_CURSOR_SPAN, RgfxBufferOverHostMemory, RgfxMsgHeader, RgfxQueueCreate,
};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::values::{RenderPipelineDesc, SamplerDesc, TextureDesc};
use reims_vgpu_vulkan::core::raise::OnFault;
use zerocopy::IntoBytes;

use super::draw_cost::{counts, fixture, spread, test_shaders};
use super::{GraphicsDecoder, ObjHandle};
use crate::platform::ApiPlatform;

/// The kinds, by name.
const KINDS: [(&str, Kind); 11] = [
    (
        "buffer.shared.4k",
        Kind::Buffer {
            length: 0x1000,
            options: 0,
        },
    ),
    (
        "buffer.private.256k",
        Kind::Buffer {
            length: 0x4_0000,
            options: PRIVATE,
        },
    ),
    ("buffer.host.64k", Kind::HostBuffer),
    ("texture.16", Kind::Texture { side: 16 }),
    ("texture.256", Kind::Texture { side: 256 }),
    ("sampler", Kind::Sampler),
    ("render.pipeline", Kind::RenderPipeline),
    ("compute.pipeline", Kind::ComputePipeline),
    ("event", Kind::Event),
    ("shared.event", Kind::SharedEvent),
    ("queue", Kind::Queue),
];

/// What a run's objects are.
#[derive(Debug, Clone, Copy)]
enum Kind {
    /// `newBufferWithLength:options:` of `length` bytes.
    Buffer { length: u64, options: u64 },
    /// A `BUFFER_OVER_HOST_MEMORY` of [`HOST_LENGTH`] bytes of the region, each its own.
    HostBuffer,
    /// A `side` by `side` BGRA8 texture a render pass draws and a shader samples.
    Texture { side: u64 },
    /// A sampler of the default descriptor.
    Sampler,
    /// A render pipeline state of the suite's `render_vbuf` and `render_fbuf`.
    RenderPipeline,
    /// A compute pipeline state of the suite's `add_one`.
    ComputePipeline,
    /// `newEvent`.
    Event,
    /// `newSharedEvent`.
    SharedEvent,
    /// A `QUEUE_CREATE`.
    Queue,
}

/// `MTLResourceStorageModePrivate`.
const PRIVATE: u64 = 0x20;
/// `MTLPixelFormatBGRA8Unorm`; `MTLTextureUsageShaderRead | RenderTarget`.
const BGRA8: u64 = 80;
const USAGE: u64 = 1 | 4;
/// A host buffer's bytes.
const HOST_LENGTH: u64 = 0x1_0000;
/// Timed runs of a kind, after one warm run.
const REPS: usize = 5;

/// The objects a run makes: `CLOSE_COST_COUNT`, or 1000.
fn count() -> u64 {
    std::env::var("CLOSE_COST_COUNT")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(1000)
}

/// Whether `CLOSE_COST_KINDS` names `name`, or is not set.
fn chosen(name: &str) -> bool {
    std::env::var("CLOSE_COST_KINDS").map_or(true, |kinds| kinds.split(',').any(|k| k == name))
}

/// A decoder on `P`'s system device whose context 0 holds `n` objects of `kind`, with its
/// device.
fn made<P: ApiPlatform>(kind: Kind, n: u64) -> (GraphicsDecoder<P>, P::Device) {
    let platform = P::process();
    let device = platform.metal_device(None).unwrap();
    let region_size = RGFX_CONTEXT_ALLOCATIONS_OFFSET + n * HOST_LENGTH + RGFX_CURSOR_SPAN;
    let region = reims_vgpu_graphics::region::Region::new(region_size as usize).unwrap();
    let mut decoder = GraphicsDecoder::<P>::new(Arc::clone(&platform), Arc::new(region));
    let o = OnFault::Die;
    let pipeline_functions = match kind {
        Kind::RenderPipeline => Some((
            fixture::<P>(&device, "render_vbuf"),
            fixture::<P>(&device, "render_fbuf"),
        )),
        _ => None,
    };
    let kernel = matches!(kind, Kind::ComputePipeline).then(|| {
        let path = test_shaders().join("compute_add.metallib");
        let library = device
            .new_library(&std::fs::read(path).unwrap())
            .unwrap()
            .unwrap()
            .unwrap();
        reims_vgpu_vulkan::core::platform::resource::Library::new_function(&library, o, "add_one")
            .unwrap()
            .unwrap()
    });
    for id in 1..=n {
        let handle = match kind {
            Kind::Buffer { length, options } => ObjHandle::Buffer(
                device
                    .new_buffer_of_length(o, length, options)
                    .unwrap()
                    .unwrap(),
            ),
            Kind::HostBuffer => {
                let buffer = RgfxBufferOverHostMemory {
                    header: header(
                        RGFX_CMD_BUFFER_OVER_HOST_MEMORY,
                        size_of::<RgfxBufferOverHostMemory>(),
                    ),
                    obj_id: id,
                    region_offset: RGFX_CONTEXT_ALLOCATIONS_OFFSET + (id - 1) * HOST_LENGTH,
                    length: HOST_LENGTH,
                    ..RgfxBufferOverHostMemory::default()
                };
                assert_eq!(decoder.decode(&device, buffer.as_bytes()).unwrap(), Ok(()));
                continue;
            }
            Kind::Queue => {
                let queue = RgfxQueueCreate {
                    header: header(RGFX_CMD_QUEUE_CREATE, size_of::<RgfxQueueCreate>()),
                    obj_id: id,
                    ..RgfxQueueCreate::default()
                };
                assert_eq!(decoder.decode(&device, queue.as_bytes()).unwrap(), Ok(()));
                continue;
            }
            Kind::Texture { side } => {
                let desc = platform.texture_descriptor(&TextureDesc {
                    pixel_format: BGRA8,
                    width: side,
                    height: side,
                    usage: USAGE,
                    ..TextureDesc::default()
                });
                ObjHandle::Texture(device.new_texture(o, &desc).unwrap().unwrap())
            }
            Kind::Sampler => ObjHandle::Sampler(
                device
                    .new_sampler_state(&SamplerDesc::default())
                    .unwrap()
                    .unwrap(),
            ),
            Kind::RenderPipeline => {
                let (vertex, fragment) = pipeline_functions.as_ref().unwrap();
                ObjHandle::RenderPipeline(
                    device
                        .new_render_pipeline_state(&RenderPipelineDesc {
                            label: None,
                            vertex,
                            fragment,
                            color0_pixel_format: BGRA8,
                            resource_index: None,
                        })
                        .unwrap()
                        .unwrap(),
                )
            }
            Kind::ComputePipeline => ObjHandle::ComputePipeline(
                device
                    .new_compute_pipeline_state_for_function(kernel.as_ref())
                    .unwrap()
                    .unwrap(),
            ),
            Kind::Event => ObjHandle::Event(device.new_event(o).unwrap().unwrap()),
            Kind::SharedEvent => ObjHandle::SharedEvent(device.new_shared_event().unwrap()),
        };
        assert_eq!(decoder.insert(id, handle).unwrap(), Ok(()));
    }
    (decoder, device)
}

/// A ring-0 message's header of `type`, `size` bytes long.
fn header(r#type: u32, size: usize) -> RgfxMsgHeader {
    RgfxMsgHeader {
        r#type,
        size: size as u32,
        seqno: 0,
    }
}

/// One close of `n` objects of `kind`: its CPU nanoseconds, instructions and wall nanoseconds
/// an object.
#[allow(clippy::disallowed_methods)] // a measurement: the wall time is a figure it prints
fn closed<P: ApiPlatform>(kind: Kind, n: u64) -> [f64; 3] {
    let (decoder, device) = made::<P>(kind, n);
    let wall = Instant::now();
    let start = counts();
    let dropped = decoder.drop_context(&device, 0);
    let spent = counts() - start;
    let wall = wall.elapsed();
    assert_eq!(dropped as u64, n, "{kind:?}");
    let each = |v: f64| v / n as f64;
    [
        each(spent.0[0] as f64),
        each(spent.0[1] as f64),
        each(wall.as_nanos() as f64),
    ]
}

/// Runs every chosen kind on `P`, printing a line of each under `platform`'s name.
fn measure<P: ApiPlatform>(platform: &str) {
    let n = count();
    println!(
        "close_cost on {platform}, {n} objects a close; each figure the median of {REPS}, an object:"
    );
    for (name, kind) in KINDS {
        if !chosen(name) {
            continue;
        }
        closed::<P>(kind, n);
        let runs: Vec<[f64; 3]> = (0..REPS).map(|_| closed::<P>(kind, n)).collect();
        let median = |i: usize| {
            let mut v: Vec<f64> = runs.iter().map(|r| r[i]).collect();
            spread(&mut v)[0]
        };
        println!(
            "  {name:<20} cpu {:>8.2} us  wall {:>8.2} us  instructions {:>9.0}",
            median(0) / 1000.0,
            median(2) / 1000.0,
            median(1),
        );
    }
}

/// On the fake platform: the decoder's own cost.
#[test]
#[ignore = "a measurement, run by hand in a release build: prints a close's cost an object"]
fn on_fake_close_cost() {
    measure::<reims_vgpu_vulkan::core::testkit::FakePlatform>("the fake platform");
}

/// On the Vulkan platform: the decoder's, the platform's and the runtime's.
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU: prints a close's cost an object"]
fn on_vulkan_close_cost() {
    measure::<reims_vgpu_vulkan::VulkanPlatform>("vulkan");
}
