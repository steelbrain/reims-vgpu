//! Mesh render pipelines on this platform
//! (`crate::gpu::states::new_mesh_render_pipeline_state_full` and the encoder's mesh draws): what
//! a state answers, what its creation refuses, and what its draws draw, against what the graphics
//! lane's mesh oracle measured of Metal on the M3 Max
//! (`guest/bundle/tools/oracle/oracle_mesh.m`, `draw` and `make`, macOS 27.0.1), whose functions
//! `mesh.metal` holds. MoltenVK has no `VK_EXT_mesh_shader`, so on a Mac the runtime draws them
//! emulated (the module docs).

use reims_vgpu_core::platform::resource::Library as _;
use reims_vgpu_core::platform::types::NsErrorText;
use reims_vgpu_core::platform::values::{
    ColorAttachmentDesc, FullMeshRenderPipelineDesc, LinkedFunctionsDesc, SamplerDesc,
};

use super::*;

/// The tests' functions (`mesh.metal`).
const MESH_LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/mesh.metallib"));

/// `MTLPixelFormatRGBA8Unorm`.
const RGBA8: u64 = 70;

/// The target's size: the oracle's.
const W: u64 = 16;
const H: u64 = 4;

/// Opaque red, as a BGRA8 texel read as a little-endian word.
const RED: u32 = 0xffff_0000;

/// No size: a descriptor's required threads left unset.
const UNSET: MtlSize = MtlSize {
    width: 0,
    height: 0,
    depth: 0,
};

/// The library's functions named `names`, in their order.
fn functions<P: Platform>(d: &P::Device, names: &[&str]) -> Vec<P::Function> {
    let library = d.new_library(MESH_LIBRARY).unwrap().unwrap().unwrap();
    (names.iter())
        .map(|name| library.new_function(OnFault::Die, name).unwrap().unwrap())
        .collect()
}

/// The oracle's base descriptor: `mesh` and, where given, `object` and `fragment`, a BGRA8
/// colour attachment 0, and everything else Metal's default.
fn descriptor<'a, P: Platform>(
    object: Option<&'a P::Function>,
    mesh: &'a P::Function,
    fragment: Option<&'a P::Function>,
) -> FullMeshRenderPipelineDesc<'a, P> {
    FullMeshRenderPipelineDesc {
        label: None,
        object,
        mesh,
        fragment,
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
        max_total_threads_per_object_threadgroup: 0,
        max_total_threads_per_mesh_threadgroup: 0,
        max_total_threadgroups_per_mesh_grid: 0,
        object_threadgroup_size_is_multiple_of_execution_width: false,
        mesh_threadgroup_size_is_multiple_of_execution_width: false,
        payload_memory_length: 0,
        max_vertex_amplification_count: 1,
        required_threads_per_object_threadgroup: UNSET,
        required_threads_per_mesh_threadgroup: UNSET,
        support_indirect_command_buffers: false,
        object_linked: LinkedFunctionsDesc::default(),
        mesh_linked: LinkedFunctionsDesc::default(),
        fragment_linked: LinkedFunctionsDesc::default(),
    }
}

/// A state of `desc`, which must be made.
fn state<P: Platform>(
    d: &P::Device,
    desc: &FullMeshRenderPipelineDesc<'_, P>,
) -> P::RenderPipeline {
    d.new_mesh_render_pipeline_state_full(desc)
        .unwrap()
        .unwrap()
}

/// A state's five mesh answers: `maxTotalThreadsPerObjectThreadgroup`,
/// `maxTotalThreadsPerMeshThreadgroup`, `objectThreadExecutionWidth`, `meshThreadExecutionWidth`
/// and `maxTotalThreadgroupsPerMeshGrid`.
fn answers<P: Platform>(s: &P::RenderPipeline) -> [u64; 5] {
    [
        s.max_total_threads_per_object_threadgroup(),
        s.max_total_threads_per_mesh_threadgroup(),
        s.object_thread_execution_width(),
        s.mesh_thread_execution_width(),
        s.max_total_threadgroups_per_mesh_grid(),
    ]
}

/// The oracle's draws (`oracle_mesh draw`), each into a 16 by 4 BGRA8 target cleared to zeros.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Draw {
    /// `m_plain`, red at mesh buffer 0, four threadgroups of 4 threads.
    Threadgroups,
    /// The same, three threadgroups from indirect arguments at offset 16.
    Indirect,
    /// `m_grid`, green at mesh buffer 1 through `setMeshBufferOffset:atIndex:` 32,
    /// `drawMeshThreads:` of 10 threads in threadgroups of 4: two threadgroups, of an 8-thread
    /// grid, the alpha.
    Threads,
    /// `o_fn` and `m_payload`, green then blue at object buffer 0, 16 bytes of object
    /// threadgroup memory at 0, two object threadgroups of 3 threads, each launching two mesh
    /// threadgroups: the second's blue over the whole target, its 3 threads the alpha.
    Object,
    /// `m_tex`, a 1x1 RGBA8 blue texture sampled through a default sampler, two threadgroups.
    Texture,
    /// `m_plain`, a grid of no threadgroup along x: nothing drawn.
    Zero,
    /// `m_writes`, as [`Draw::Threadgroups`], each threadgroup also writing its position plus 1
    /// into mesh buffer 2.
    Writes,
}

/// What the M3 Max's Metal draws of each of [`Draw`] at the oracle's eight samples (row 2,
/// columns 1, 3 to 15), as little-endian words of BGRA8 texels (`draw.*` of the oracle's
/// results); [`Draw::Writes`] draws as [`Draw::Threadgroups`].
fn metals(draw: Draw) -> [u32; 8] {
    [match draw {
        Draw::Threadgroups | Draw::Indirect | Draw::Writes => RED,
        Draw::Threads => 0x0800_ff00,
        Draw::Object => 0x0300_00ff,
        Draw::Texture => 0xff00_00ff,
        Draw::Zero => 0,
    }; 8]
}

/// The oracle's samples of `draw` on `p`, and the words mesh buffer 2 holds after it (four
/// zeros but for [`Draw::Writes`]).
fn drawn<P: Platform>(p: &P, d: &P::Device, draw: Draw) -> ([u32; 8], [u32; 4]) {
    let names: &[&str] = match draw {
        Draw::Threadgroups | Draw::Indirect | Draw::Zero => &["m_plain", "f_col"],
        Draw::Threads => &["m_grid", "f_col"],
        Draw::Object => &["m_payload", "f_col", "o_fn"],
        Draw::Texture => &["m_tex", "f_col"],
        Draw::Writes => &["m_writes", "f_col"],
    };
    let fs = functions::<P>(d, names);
    let s = state(d, &descriptor::<P>(fs.get(2), &fs[0], Some(&fs[1])));
    let target = texture(p, d, BGRA8, TARGET | READ, (W, H), None);
    let pass = p.render_pass_descriptor();
    pass.set_color_attachment(
        OnFault::Die,
        0,
        &ColorAttachmentSet {
            texture: Some(Some(&target)),
            load_action: Some(CLEAR),
            store_action: Some(STORE),
            clear_color: Some(ClearColor {
                red: 0.0,
                green: 0.0,
                blue: 0.0,
                alpha: 0.0,
            }),
            ..ColorAttachmentSet::default()
        },
    )
    .unwrap();
    let red = bytes(&[1.0, 0.0, 0.0, 1.0]);
    let mut green_at_32 = vec![0; 32];
    green_at_32.extend(bytes(&[0.0, 1.0, 0.0, 1.0]));
    green_at_32.resize(64, 0);
    let mut args = vec![0; 16];
    args.extend([3u32, 1, 1].iter().flat_map(|v| v.to_ne_bytes()));
    let colours = bytes(&[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    let red_buffer = buffer::<P>(d, 16, 0, &red);
    let green_buffer = buffer::<P>(d, 64, 0, &green_at_32);
    let args_buffer = buffer::<P>(d, 64, 0, &args);
    let colours_buffer = buffer::<P>(d, 32, 0, &colours);
    let out = buffer::<P>(d, 16, 0, &[0; 16]);
    let blue: &[u8] = &[0, 0, 255, 255];
    let source = texture(p, d, RGBA8, READ, (1, 1), Some((blue, 4)));
    let sampler = d
        .new_sampler_state(&SamplerDesc::default())
        .unwrap()
        .unwrap();
    let one = MtlSize::new(1, 1, 1);
    let four = MtlSize::new(4, 1, 1);
    let range = |location| NsRange {
        location,
        length: 1,
    };
    let red_list = [Some(&red_buffer)];
    let green_list = [Some(&green_buffer)];
    let colours_list = [Some(&colours_buffer)];
    let out_list = [Some(&out)];
    let textures = [Some(&source)];
    let samplers = [Some(&sampler)];
    let mut cmds = vec![RenderCmd::SetRenderPipelineState {
        pipeline_state: Some(&s),
    }];
    let threadgroups = |n| RenderCmd::DrawMeshThreadgroups {
        threadgroups_per_grid: MtlSize::new(n, 1, 1),
        threads_per_object_threadgroup: one,
        threads_per_mesh_threadgroup: four,
    };
    let red_at_0 = RenderCmd::SetMeshBuffers {
        buffers: &red_list,
        offsets: &[0],
        range: range(0),
    };
    match draw {
        Draw::Threadgroups => cmds.extend([red_at_0, threadgroups(4)]),
        Draw::Zero => cmds.extend([
            red_at_0,
            RenderCmd::DrawMeshThreadgroups {
                threadgroups_per_grid: MtlSize::new(0, 1, 1),
                threads_per_object_threadgroup: one,
                threads_per_mesh_threadgroup: four,
            },
        ]),
        Draw::Indirect => cmds.extend([
            red_at_0,
            RenderCmd::DrawMeshThreadgroupsIndirect {
                indirect_buffer: &args_buffer,
                indirect_buffer_offset: 16,
                threads_per_object_threadgroup: one,
                threads_per_mesh_threadgroup: four,
            },
        ]),
        Draw::Threads => cmds.extend([
            RenderCmd::SetMeshBuffers {
                buffers: &green_list,
                offsets: &[0],
                range: range(1),
            },
            RenderCmd::SetMeshBufferOffset {
                offset: 32,
                index: 1,
            },
            RenderCmd::DrawMeshThreads {
                threads_per_grid: MtlSize::new(10, 1, 1),
                threads_per_object_threadgroup: one,
                threads_per_mesh_threadgroup: four,
            },
        ]),
        Draw::Object => cmds.extend([
            RenderCmd::SetObjectBuffers {
                buffers: &colours_list,
                offsets: &[0],
                range: range(0),
            },
            RenderCmd::SetObjectThreadgroupMemoryLength {
                length: 16,
                index: 0,
            },
            RenderCmd::DrawMeshThreadgroups {
                threadgroups_per_grid: MtlSize::new(2, 1, 1),
                threads_per_object_threadgroup: MtlSize::new(3, 1, 1),
                threads_per_mesh_threadgroup: four,
            },
        ]),
        Draw::Texture => cmds.extend([
            RenderCmd::SetMeshTextures {
                textures: &textures,
                range: range(0),
            },
            RenderCmd::SetMeshSamplerStates {
                samplers: &samplers,
                range: range(0),
            },
            threadgroups(2),
        ]),
        Draw::Writes => cmds.extend([
            red_at_0,
            RenderCmd::SetMeshBuffers {
                buffers: &out_list,
                offsets: &[0],
                range: range(2),
            },
            threadgroups(4),
        ]),
    }
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.render_encoder(OnFault::Die, &pass).unwrap().unwrap();
    for cmd in cmds.into_iter().chain([RenderCmd::EndEncoding]) {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    let pixels = buffer::<P>(d, W * H * 4, 0, &[]);
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    blit.encode(
        OnFault::Die,
        BlitCmd::CopyTextureToBuffer {
            source_texture: &target,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(W, H, 1),
            destination_buffer: &pixels,
            destination_offset: 0,
            destination_bytes_per_row: W * 4,
            destination_bytes_per_image: W * H * 4,
        },
    )
    .unwrap();
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{draw:?}: {:?}", cb.error());
    let words = |b: &P::Buffer, n: usize| {
        // SAFETY: a Shared buffer of at least `n` words, read after its command buffer completed.
        let bytes = unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), n * 4) };
        let (words, _) = bytes.as_chunks::<4>();
        words
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect::<Vec<_>>()
    };
    let texels = words(&pixels, usize::try_from(W * H).unwrap());
    let w = usize::try_from(W).unwrap();
    let samples = std::array::from_fn(|c| texels[w * 2 + c * 2 + 1]);
    let written = words(&out, 4).try_into().unwrap();
    (samples, written)
}

/// A mesh render pipeline of a mesh and a fragment function, and one with an object function
/// too, draw what Metal draws on the M3 Max, directly, from indirect arguments and by
/// `drawMeshThreads:`, from buffers, an offset set after the bind, a texture and a sampler,
/// object threadgroup memory and the payload; a grid of no threadgroup draws nothing; and a mesh
/// function's writes into a buffer bound to its stage land
#[test]
fn mesh_draws_draw_what_metal_draws() {
    let (p, d) = (platform(), dev());
    let draws = [
        Draw::Threadgroups,
        Draw::Indirect,
        Draw::Threads,
        Draw::Object,
        Draw::Texture,
        Draw::Zero,
        Draw::Writes,
    ];
    let mut differ = Vec::new();
    for draw in draws {
        let ours = drawn(&p, &d, draw);
        let written = if draw == Draw::Writes {
            [1, 2, 3, 4]
        } else {
            [0; 4]
        };
        if ours != (metals(draw), written) {
            differ.push(format!("{draw:?}: drew {ours:08x?}"));
        }
    }
    assert!(differ.is_empty(), "{differ:#?}");
}

/// The most threads this device runs an object (`true`) or mesh threadgroup of, within Metal's
/// 1024 ([`crate::gpu::states::stage_threads`]).
fn limit(object: bool) -> u64 {
    crate::gpu::states::stage_threads(&dev(), object)
}

/// A mesh render pipeline state answers its descriptor's maxima within the device's limits, and
/// the device's own where none is set: 0 and an object width of 0 without an object function,
/// the simdgroups' 32 as its execution widths, Metal's grid of 2^20 threadgroups or the
/// descriptor's. Its required sizes change no answer, and a state of a payload of 16 KiB is made,
/// but of one byte more is refused with the M3 Max's code and words
#[test]
fn mesh_states_answer_their_maxima_and_refuse_big_payloads() {
    let d = dev();
    let fs = functions::<VulkanPlatform>(&d, &["m_plain", "m_payload", "o_fn", "f_col"]);
    let [plain_fn, payload_fn, object_fn, fragment] = [&fs[0], &fs[1], &fs[2], &fs[3]];
    let width = u64::from(plain(|| {
        d.raw
            .compile_device()
            .metal_properties()
            .thread_execution_width
    }));
    assert_eq!(width, 32, "the runtime's simdgroups");
    let plain_desc = descriptor::<VulkanPlatform>(None, plain_fn, Some(fragment));
    let object = descriptor::<VulkanPlatform>(Some(object_fn), payload_fn, Some(fragment));
    let maxima = FullMeshRenderPipelineDesc {
        max_total_threads_per_object_threadgroup: 64,
        max_total_threads_per_mesh_threadgroup: 32,
        max_total_threadgroups_per_mesh_grid: 8,
        payload_memory_length: 256,
        ..descriptor::<VulkanPlatform>(Some(object_fn), payload_fn, Some(fragment))
    };
    let required = FullMeshRenderPipelineDesc {
        required_threads_per_object_threadgroup: MtlSize::new(2, 1, 1),
        required_threads_per_mesh_threadgroup: MtlSize::new(4, 1, 1),
        ..descriptor::<VulkanPlatform>(Some(object_fn), payload_fn, Some(fragment))
    };
    let (object_limit, mesh_limit) = (limit(true), limit(false));
    assert_eq!(
        answers::<VulkanPlatform>(&state(&d, &plain_desc)),
        [0, mesh_limit, 0, 32, 1 << 20]
    );
    let of_object = [object_limit, mesh_limit, 32, 32, 1 << 20];
    assert_eq!(answers::<VulkanPlatform>(&state(&d, &object)), of_object);
    assert_eq!(answers::<VulkanPlatform>(&state(&d, &required)), of_object);
    assert_eq!(
        answers::<VulkanPlatform>(&state(&d, &maxima)),
        [64.min(object_limit), 32.min(mesh_limit), 32, 32, 8]
    );
    let payload = |length| FullMeshRenderPipelineDesc {
        payload_memory_length: length,
        ..descriptor::<VulkanPlatform>(Some(object_fn), payload_fn, Some(fragment))
    };
    assert!(matches!(
        d.new_mesh_render_pipeline_state_full(&payload(16384)),
        Ok(Some(_))
    ));
    let refused = d
        .new_mesh_render_pipeline_state_full(&payload(16385))
        .unwrap_err();
    let words =
        "Object shader payload size (16385) exceeds the maximum payload size allowed (16384)";
    assert_eq!((refused.code, refused.description.as_str()), (3, words));
}

/// A mesh render pipeline whose mesh, object or fragment function is of another stage's type is
/// refused in the compiler's domain, before the runtime sees it: code
/// 2 for a vertex function as the mesh function, 1 for an object function as the mesh function,
/// a mesh function as the object function, a fragment function as the object function, and a
/// mesh function as the fragment function. The mesh function is asked first
#[test]
fn mesh_states_of_wrong_stages_are_refused() {
    let d = dev();
    let fs = functions::<VulkanPlatform>(&d, &["m_plain", "o_fn", "f_col", "v_fn"]);
    let [mesh, object, fragment, vertex] = [&fs[0], &fs[1], &fs[2], &fs[3]];
    let refusal = |code, description: &str| NsErrorText {
        domain: "CompilerError".to_owned(),
        code,
        description: description.to_owned(),
    };
    let vertex_words =
        "RasterizationEnabled is false but the vertex shader's return type is not void";
    let mesh_words = "meshFunction functionType is not MTLFunctionTypeMesh.";
    let object_words = "objectFunction functionType is not MTLFunctionTypeObject.";
    let fragment_words = "fragmentFunction functionType is not MTLFunctionTypeFragment.";
    let cases = [
        (None, vertex, Some(fragment), refusal(2, vertex_words)),
        (None, object, Some(fragment), refusal(1, mesh_words)),
        (Some(object), fragment, Some(mesh), refusal(1, mesh_words)),
        (Some(mesh), mesh, Some(fragment), refusal(1, object_words)),
        (
            Some(fragment),
            mesh,
            Some(fragment),
            refusal(1, object_words),
        ),
        (None, mesh, Some(mesh), refusal(1, fragment_words)),
        (Some(object), mesh, Some(vertex), refusal(1, fragment_words)),
    ];
    for (i, (object, mesh, fragment, wanted)) in cases.into_iter().enumerate() {
        let desc = descriptor::<VulkanPlatform>(object, mesh, fragment);
        let made = d.new_mesh_render_pipeline_state_full(&desc);
        assert_eq!(made.err(), Some(wanted), "case {i}");
    }
}

/// A mesh draw of a threadgroup past the maximum its state's descriptor sets for the stage, or
/// of another size than the one it requires, dies at the draw, directly, by threads and from
/// indirect arguments; one within them goes on, and so does a threadgroup of no thread or past
/// Metal's 1024, which draws nothing. The object stage's are asked only of a state with an
/// object function
#[test]
fn mesh_draws_outside_their_states_threadgroups_die() {
    let d = dev();
    let fs = functions::<VulkanPlatform>(&d, &["m_plain", "m_payload", "o_fn", "f_col"]);
    let [plain_fn, payload_fn, object_fn, fragment] = [&fs[0], &fs[1], &fs[2], &fs[3]];
    let held = state(
        &d,
        &FullMeshRenderPipelineDesc {
            max_total_threads_per_mesh_threadgroup: 32,
            required_threads_per_object_threadgroup: MtlSize::new(4, 1, 1),
            ..descriptor::<VulkanPlatform>(None, plain_fn, Some(fragment))
        },
    );
    let required = state(
        &d,
        &FullMeshRenderPipelineDesc {
            required_threads_per_object_threadgroup: MtlSize::new(2, 1, 1),
            required_threads_per_mesh_threadgroup: MtlSize::new(4, 1, 1),
            ..descriptor::<VulkanPlatform>(Some(object_fn), payload_fn, Some(fragment))
        },
    );
    let args = buffer::<VulkanPlatform>(&d, 16, 0, &[0; 16]);
    let colours = buffer::<VulkanPlatform>(&d, 32, 0, &bytes(&[1.0; 8]));
    let colours_list = [Some(&colours)];
    let threadgroups = |object, mesh| RenderCmd::DrawMeshThreadgroups {
        threadgroups_per_grid: MtlSize::new(1, 1, 1),
        threads_per_object_threadgroup: object,
        threads_per_mesh_threadgroup: mesh,
    };
    let threads = |object, mesh| RenderCmd::DrawMeshThreads {
        threads_per_grid: MtlSize::new(8, 1, 1),
        threads_per_object_threadgroup: object,
        threads_per_mesh_threadgroup: mesh,
    };
    let indirect = |object, mesh| RenderCmd::DrawMeshThreadgroupsIndirect {
        indirect_buffer: &args,
        indirect_buffer_offset: 0,
        threads_per_object_threadgroup: object,
        threads_per_mesh_threadgroup: mesh,
    };
    let size = MtlSize::new;
    let cases = [
        // Past the mesh stage's 32, by each draw.
        (&held, threadgroups(size(1, 1, 1), size(64, 1, 1)), true),
        (&held, threads(size(1, 1, 1), size(8, 8, 1)), true),
        (&held, indirect(size(1, 1, 1), size(33, 1, 1)), true),
        // Within it, of no thread and past 1024: drawn, as nothing.
        (&held, threadgroups(size(1, 1, 1), size(32, 1, 1)), false),
        (&held, threadgroups(size(1, 1, 1), size(0, 1, 1)), false),
        (&held, threadgroups(size(1, 1, 1), size(1025, 1, 1)), false),
        // No object function: its required size is not asked.
        (&held, threadgroups(size(3, 1, 1), size(4, 1, 1)), false),
        // Required object and mesh sizes.
        (&required, threadgroups(size(2, 1, 1), size(4, 1, 1)), false),
        (&required, threadgroups(size(4, 1, 1), size(4, 1, 1)), true),
        (&required, threadgroups(size(2, 1, 1), size(2, 2, 1)), true),
        (&required, threads(size(2, 1, 1), size(8, 1, 1)), true),
        (&required, indirect(size(1, 2, 1), size(4, 1, 1)), true),
    ];
    for (i, (s, draw, dies)) in cases.into_iter().enumerate() {
        let sel = draw.selector();
        let (e, _target) = open_encoder();
        let range = NsRange {
            location: 0,
            length: 1,
        };
        for cmd in [
            RenderCmd::SetRenderPipelineState {
                pipeline_state: Some(s),
            },
            RenderCmd::SetMeshBuffers {
                buffers: &colours_list,
                offsets: &[0],
                range,
            },
            RenderCmd::SetObjectBuffers {
                buffers: &colours_list,
                offsets: &[0],
                range,
            },
            RenderCmd::SetObjectThreadgroupMemoryLength {
                length: 16,
                index: 0,
            },
        ] {
            e.encode(OnFault::Die, cmd).unwrap();
        }
        let ended = catch_death(|| e.encode(OnFault::Return, draw));
        match dies {
            true => assert_eq!(ended.unwrap_err().file, sel, "case {i}"),
            false => {
                assert!(ended.is_ok(), "case {i}: {ended:?}");
                e.encode(OnFault::Die, RenderCmd::EndEncoding).unwrap();
            }
        }
    }
}

/// A mesh render pipeline state's reflection asks the object, mesh and fragment stages' slot
/// writes, in that order: the mesh function may store through its `device` buffer 2 and not
/// through its `constant` buffer 0, and the object function not through its `constant` buffer 0
#[test]
fn mesh_states_reflect_their_stages_slot_writes() {
    let d = dev();
    let fs = functions::<VulkanPlatform>(&d, &["m_writes", "o_fn", "f_col", "m_payload"]);
    let writes = state(
        &d,
        &descriptor::<VulkanPlatform>(None, &fs[0], Some(&fs[2])),
    );
    let reflection = &writes.reflection;
    assert_eq!(reflection.stages(), &[OBJECT, MESH, FRAGMENT]);
    let slots = reflection.slots().unwrap();
    use metal2vulkan::reflection::SlotWritability as S;
    assert_eq!(slots[MESH].buffer(2), S::Maybe);
    assert_eq!(slots[MESH].buffer(0), S::Never);
    let object = state(
        &d,
        &descriptor::<VulkanPlatform>(Some(&fs[1]), &fs[3], Some(&fs[2])),
    );
    assert_eq!(
        object.reflection.slots().unwrap()[OBJECT].buffer(0),
        S::Never
    );
}
