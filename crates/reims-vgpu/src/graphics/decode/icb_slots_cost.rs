//! What an `ICB_SLOTS` costs the thread that carries it out, a command at a time: 16384 render
//! commands, each a full slot (its pipeline state, two vertex buffers, a fragment buffer and a
//! draw), through the decoder as ring 0 hands it the message, against the same setters sent
//! through the platform's core traits and, on the Vulkan platform, through the runtime alone: a
//! measurement, run by hand in a release build, holding the host's GPU
//! (`scripts/internal/gpu-locks.sh -- cargo nextest run -p reims-vgpu --release
//! --no-default-features --features reims-vgpu/vulkan --run-ignored only -E
//! 'test(icb_slots_cost)' --no-capture`).
//!
//! Each path's figures are the thread's CPU time, instructions and cycles a command, over the
//! message's carrying out alone: the slots are written into the region before each repetition,
//! as the bundle writes them before its commit. The gaps between the paths are the decoder's
//! share (the slots' take and decode, the objects' lookups, the refusal scopes) and the
//! platform's (its command object and its sends). By default nothing executes the buffer.
//!
//! `ICB_SLOTS_COST_EXECUTED=1` has a render pass execute every command, and waits for it, before
//! each repetition of each path, as a frame executes the buffer and the next one encodes it
//! again. `ICB_SLOTS_COST_RUNS` sets the runs (5), `ICB_SLOTS_COST_ONLY` runs one path (`decoder`,
//! `platform` or `runtime`), and `ICB_SLOTS_COST_SECONDS` keeps running that path for a time, so
//! a sampling profiler can attach.

use std::sync::Arc;

use reims_vgpu_graphics::icb::{
    RGFX_ICB_OP_DRAW, RGFX_ICB_SLOT_ACTION, RGFX_ICB_SLOT_BUFFER_0, RGFX_ICB_SLOT_PIPELINE,
    RgfxIcbDraw, RgfxIcbRenderSlot, RgfxIcbSlotBuffer,
};
use reims_vgpu_graphics::region::Region;
use reims_vgpu_graphics::wire::{
    RGFX_CMD_ICB_CREATE, RGFX_CMD_ICB_SLOTS, RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN,
    RGFX_ICB_SLOTS_RENDER, RgfxIcbCreate, RgfxIcbSlotRun, RgfxIcbSlots, RgfxMsgHeader,
};
use reims_vgpu_vulkan::core::platform::blit::ResourceRef;
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::descriptor::RenderPassDescriptor as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::icb::{
    IcbRenderCmd, IndirectCommandBuffer as _, IndirectRenderCommand as _,
};
use reims_vgpu_vulkan::core::platform::render::{RenderCmd, RenderEncoder as _};
use reims_vgpu_vulkan::core::platform::types::NsRange;
use reims_vgpu_vulkan::core::platform::values::{
    ColorAttachmentDesc, ColorAttachmentSet, FullRenderPipelineDesc, LinkedFunctionsDesc,
    TextureDesc,
};
use reims_vgpu_vulkan::core::raise::OnFault;
use zerocopy::IntoBytes;

use super::draw_cost::{Counts, FIGURES, counts, fixture, spread};
use super::{GraphicsDecoder, ObjHandle};
use crate::platform::ApiPlatform;

/// The commands an `ICB_SLOTS` carries out, as oracle_icb's timing case encodes them.
const COMMANDS: u64 = 16384;
/// Repetitions a run times, after [`WARM`] repetitions.
const REPS: usize = 6;
const WARM: usize = 2;
/// The object ids: the buffer, its pipeline state, and the two buffers its commands bind.
const ICB_ID: u64 = 1;
const PIPELINE_ID: u64 = 2;
const BUFFER_A: u64 = 3;
const BUFFER_B: u64 = 4;
/// Where the slots lie, and a slot's bytes: its fixed part and its three bindings.
const SLOTS_AT: u64 = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
const STRIDE: u64 = (size_of::<RgfxIcbRenderSlot>() + 3 * size_of::<RgfxIcbSlotBuffer>()) as u64;
/// `MTLPixelFormatBGRA8Unorm`; `MTLPrimitiveTypeTriangle`; `MTLIndirectCommandTypeDraw`;
/// `MTLTextureUsageShaderRead | RenderTarget`; `Clear`, `Store`; the target's side.
const BGRA8: u64 = 80;
const TARGET_USAGE: u64 = 1 | 4;
const CLEAR: u64 = 2;
const STORE: u64 = 1;
const SIDE: u64 = 64;
const TRIANGLE: u64 = 3;
const DRAW: u32 = 1;

/// The draw every command holds.
const THE_DRAW: RgfxIcbDraw = RgfxIcbDraw {
    primitive_type: TRIANGLE as u32,
    vertex_start: 0,
    vertex_count: 3,
    instance_count: 1,
    base_instance: 0,
    reserved_14: 0,
};

/// The slots of every command, as the bundle writes them: each field set.
fn slots() -> Vec<u8> {
    let mut action = [0u64; 9];
    action.as_mut_bytes()[..size_of::<RgfxIcbDraw>()].copy_from_slice(THE_DRAW.as_bytes());
    let fixed = RgfxIcbRenderSlot {
        mask: RGFX_ICB_SLOT_PIPELINE
            | RGFX_ICB_SLOT_ACTION
            | RGFX_ICB_SLOT_BUFFER_0
            | (RGFX_ICB_SLOT_BUFFER_0 << 1),
        mask_b: 1,
        pipeline: PIPELINE_ID,
        op: RGFX_ICB_OP_DRAW,
        barrier: 0,
        strided: 0,
        action,
    };
    let bound = |obj_id, offset| RgfxIcbSlotBuffer {
        obj_id,
        offset,
        stride: 0,
    };
    let mut all = Vec::with_capacity((COMMANDS * STRIDE) as usize);
    for i in 0..COMMANDS {
        all.extend_from_slice(fixed.as_bytes());
        for b in [
            bound(BUFFER_A, (i % 16) * 16),
            bound(BUFFER_B, 0),
            bound(BUFFER_A, 256),
        ] {
            all.extend_from_slice(b.as_bytes());
        }
    }
    all
}

/// The `ICB_SLOTS` of one run of every command.
fn message() -> Vec<u8> {
    let run = RgfxIcbSlotRun {
        icb_obj_id: ICB_ID,
        region_offset: SLOTS_AT,
        first_command: 0,
        command_count: COMMANDS as u32,
        slot_bytes: STRIDE as u32,
        kind: RGFX_ICB_SLOTS_RENDER,
        bindings_a: 2,
        bindings_b: 1,
    };
    let len = size_of::<RgfxIcbSlots>() + size_of::<RgfxIcbSlotRun>();
    let mut m = RgfxIcbSlots {
        header: RgfxMsgHeader {
            r#type: RGFX_CMD_ICB_SLOTS,
            size: len as u32,
            seqno: 0,
        },
        obj_id: ICB_ID,
        run_count: 1,
        reserved_1c: 0,
    }
    .as_bytes()
    .to_vec();
    m.extend_from_slice(run.as_bytes());
    m
}

/// A decoder on `P`'s system device holding the buffer, its pipeline state and the two buffers.
struct Bench<P: ApiPlatform> {
    platform: Arc<P>,
    device: P::Device,
    decoder: GraphicsDecoder<P>,
    region: Arc<Region>,
    slots: Vec<u8>,
    queue: P::CommandQueue,
    target: P::Texture,
}

impl<P: ApiPlatform> Bench<P> {
    fn new() -> Bench<P> {
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let size = SLOTS_AT + (COMMANDS * STRIDE).next_multiple_of(0x1_0000) + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let mut decoder = GraphicsDecoder::<P>::new(Arc::clone(&platform), Arc::clone(&region));
        let create = RgfxIcbCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ICB_CREATE,
                size: size_of::<RgfxIcbCreate>() as u32,
                seqno: 0,
            },
            obj_id: ICB_ID,
            max_command_count: COMMANDS,
            command_types: DRAW,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 1,
            ..RgfxIcbCreate::default()
        };
        assert_eq!(decoder.decode(&device, create.as_bytes()).unwrap(), Ok(()));
        let vertex = fixture::<P>(&device, "render_vbuf");
        let fragment = fixture::<P>(&device, "render_fbuf");
        let pipeline = device
            .new_render_pipeline_state_full(&FullRenderPipelineDesc {
                label: None,
                vertex: &vertex,
                fragment: Some(&fragment),
                vertex_descriptor: None,
                colors: vec![(
                    0,
                    ColorAttachmentDesc {
                        pixel_format: BGRA8,
                        ..ColorAttachmentDesc::default()
                    },
                )],
                depth_pixel_format: 0,
                stencil_pixel_format: 0,
                raster_sample_count: 1,
                alpha_to_coverage: false,
                alpha_to_one: false,
                rasterization_enabled: true,
                input_primitive_topology: 0,
                max_vertex_amplification_count: 1,
                support_indirect_command_buffers: true,
                vertex_linked: LinkedFunctionsDesc::default(),
                fragment_linked: LinkedFunctionsDesc::default(),
                private_state: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(
            decoder
                .insert(PIPELINE_ID, ObjHandle::RenderPipeline(pipeline))
                .unwrap(),
            Ok(())
        );
        for id in [BUFFER_A, BUFFER_B] {
            let b = device
                .new_buffer_of_length(OnFault::Die, 4096, 0)
                .unwrap()
                .unwrap();
            assert_eq!(decoder.insert(id, ObjHandle::Buffer(b)).unwrap(), Ok(()));
        }
        decoder.end_message();
        let queue = device.new_command_queue().unwrap();
        let target = platform.texture_descriptor(&TextureDesc {
            pixel_format: BGRA8,
            width: SIDE,
            height: SIDE,
            usage: TARGET_USAGE,
            ..TextureDesc::default()
        });
        let target = device.new_texture(OnFault::Die, &target).unwrap().unwrap();
        Bench {
            platform,
            device,
            decoder,
            region,
            slots: slots(),
            queue,
            target,
        }
    }

    /// A render pass executing every command, committed and waited for: a frame's.
    fn execute(&self) {
        let o = OnFault::Die;
        let cb = self.queue.command_buffer(o).unwrap().unwrap();
        let pass = self.platform.render_pass_descriptor();
        let attachment = ColorAttachmentSet {
            texture: Some(Some(&self.target)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            ..ColorAttachmentSet::default()
        };
        pass.set_color_attachment(o, 0, &attachment).unwrap();
        let e = cb.render_encoder(o, &pass).unwrap().unwrap();
        let used = [
            Some(ResourceRef::Buffer(self.buffer(BUFFER_A))),
            Some(ResourceRef::Buffer(self.buffer(BUFFER_B))),
        ];
        let cmds = [
            RenderCmd::UseResources {
                resources: &used,
                usage: 1,
            },
            RenderCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: self.icb(),
                execution_range: NsRange::new(0, COMMANDS),
            },
            RenderCmd::EndEncoding,
        ];
        for cmd in cmds {
            e.encode(o, cmd).unwrap();
        }
        cb.commit(o).unwrap();
        cb.wait_until_completed(o).unwrap();
        assert_eq!(cb.error(), None, "the execution failed");
    }

    fn icb(&self) -> &P::Icb {
        match self.decoder.object(ICB_ID) {
            Some(ObjHandle::Icb(icb)) => icb,
            _ => panic!("no indirect command buffer"),
        }
    }

    fn buffer(&self, id: u64) -> &P::Buffer {
        match self.decoder.object(id) {
            Some(ObjHandle::Buffer(b)) => b,
            _ => panic!("no buffer {id}"),
        }
    }

    fn pipeline(&self) -> &P::RenderPipeline {
        match self.decoder.object(PIPELINE_ID) {
            Some(ObjHandle::RenderPipeline(p)) => p,
            _ => panic!("no pipeline"),
        }
    }

    /// One repetition through the decoder: the slots written, then the message carried out.
    fn through_decoder(&mut self, message: &[u8]) -> Counts {
        assert!(self.region.write(SLOTS_AT, &self.slots));
        let start = counts();
        assert_eq!(self.decoder.decode(&self.device, message).unwrap(), Ok(()));
        let spent = counts() - start;
        self.decoder.end_message();
        spent
    }

    /// One repetition through the platform's core traits, on the decoder's objects.
    fn through_platform(&self) -> Counts {
        let icb = self.icb();
        let (a, b, pipeline) = (
            self.buffer(BUFFER_A),
            self.buffer(BUFFER_B),
            self.pipeline(),
        );
        let start = counts();
        for i in 0..COMMANDS {
            let c = icb.render_command(i).unwrap().unwrap();
            let cmds = [
                IcbRenderCmd::SetRenderPipelineState {
                    pipeline_state: Some(pipeline),
                },
                IcbRenderCmd::SetVertexBuffer {
                    buffer: Some(a),
                    offset: (i % 16) * 16,
                    index: 0,
                },
                IcbRenderCmd::SetVertexBuffer {
                    buffer: Some(b),
                    offset: 0,
                    index: 1,
                },
                IcbRenderCmd::SetFragmentBuffer {
                    buffer: Some(a),
                    offset: 256,
                    index: 0,
                },
                IcbRenderCmd::DrawPrimitives {
                    primitive_type: TRIANGLE,
                    vertex_start: 0,
                    vertex_count: 3,
                    instance_count: 1,
                    base_instance: 0,
                },
            ];
            for cmd in cmds {
                c.encode(cmd).unwrap();
            }
        }
        counts() - start
    }
}

/// The runtime alone, on the Vulkan platform's objects of `bench`.
fn through_runtime(bench: &Bench<reims_vgpu_vulkan::VulkanPlatform>) -> Counts {
    let triangle = metal2vulkan_runtime::DrawPrimitiveType::from_raw(TRIANGLE).unwrap();
    let icb = &bench.icb().raw;
    let (a, b) = (&bench.buffer(BUFFER_A).raw, &bench.buffer(BUFFER_B).raw);
    let pipeline = &bench.pipeline().raw;
    let start = counts();
    for i in 0..COMMANDS {
        let c = icb.indirect_render_command_at(i).unwrap();
        c.set_render_pipeline_state(pipeline).unwrap();
        c.set_vertex_buffer(Some(a), (i % 16) * 16, 0).unwrap();
        c.set_vertex_buffer(Some(b), 0, 1).unwrap();
        c.set_fragment_buffer(Some(a), 256, 0).unwrap();
        c.draw_primitives(triangle, 0, 3, 1, 0).unwrap();
    }
    counts() - start
}

/// Runs the paths of `bench` and prints each one's figures a command.
#[allow(clippy::disallowed_methods)] // a measurement: real time decides how long a profile runs
fn measure<P: ApiPlatform>(
    name: &str,
    mut bench: Bench<P>,
    runtime: Option<fn(&Bench<P>) -> Counts>,
) {
    let runs: usize = std::env::var("ICB_SLOTS_COST_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let only = std::env::var("ICB_SLOTS_COST_ONLY").ok();
    let only = only.as_deref();
    let message = message();
    let executed = std::env::var("ICB_SLOTS_COST_EXECUTED").is_ok_and(|v| v == "1");
    let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
    #[cfg(target_os = "macos")]
    // SAFETY: sets this thread's own class, which takes no pointer.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0);
    }
    let a_command = |c: Counts| c.0.map(|v| v as f64 / (REPS as u64 * COMMANDS) as f64);
    if let Some(seconds) = std::env::var("ICB_SLOTS_COST_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        let until = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
        let mut reps = 0u64;
        let mut spent = Counts::default();
        while std::time::Instant::now() < until {
            spent += match only {
                Some("platform") => bench.through_platform(),
                Some("runtime") => runtime.map_or(Counts::default(), |r| r(&bench)),
                _ => bench.through_decoder(&message),
            };
            reps += 1;
        }
        let [ns, instructions, cycles] = spent.0.map(|v| v as f64 / (reps * COMMANDS) as f64);
        println!(
            "{name}: {} for {seconds} s: {reps} repetitions, a command {ns:.1} ns, \
             {instructions:.0} instructions, {cycles:.0} cycles",
            only.unwrap_or("decoder"),
        );
        return;
    }
    let paths = ["decoder", "platform", "runtime"];
    let mut seen: [Vec<[f64; 3]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for _ in 0..runs {
        let mut totals = [Counts::default(); 3];
        for rep in 0..WARM + REPS {
            let mut spent = [Counts::default(); 3];
            if only.is_none_or(|o| o == "decoder") {
                if executed {
                    bench.execute();
                }
                spent[0] = bench.through_decoder(&message);
            }
            if only.is_none_or(|o| o == "platform") {
                if executed {
                    bench.execute();
                }
                spent[1] = bench.through_platform();
            }
            if let Some(runtime) = runtime
                && only.is_none_or(|o| o == "runtime")
            {
                if executed {
                    bench.execute();
                }
                spent[2] = runtime(&bench);
            }
            if rep >= WARM {
                for (t, s) in totals.iter_mut().zip(spent) {
                    *t += s;
                }
            }
        }
        for (seen, total) in seen.iter_mut().zip(totals) {
            seen.push(a_command(total));
        }
    }
    println!(
        "{name}: a full render command's slot carried out{}, median [lowest, highest] of {runs} \
         runs of {REPS} repetitions of {COMMANDS} commands",
        if executed {
            " after an execution of every command"
        } else {
            ""
        }
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
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU: prints a command's cost"]
fn on_vulkan_icb_slots_cost() {
    measure(
        "vulkan",
        Bench::<reims_vgpu_vulkan::VulkanPlatform>::new(),
        Some(through_runtime),
    );
}
