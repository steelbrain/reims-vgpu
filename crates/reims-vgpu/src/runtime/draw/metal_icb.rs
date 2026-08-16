//! Execute a host ICB on a render encoder over the request's color RTs:
//! parent-encoder inheritance, the Metal execute itself, and color writeback.
//!
//! The whole module is gated on `all(backend-metal, target_os = "macos")` at
//! its declaration in [`super`], which also re-exports these items flat so
//! callers keep addressing them as `crate::runtime::draw::<name>`. The
//! `backend-vulkan` arm of `encode_icb_execute_and_writeback` is a stub that
//! lives in [`super`], since it is the other half of the same entry point.
//! `use super::*` pulls in the parent's imports, which this module shares.

use super::*;

/// Retained Metal objects for the duration of one ICB execute command buffer.
#[derive(Default)]
struct IcbEncoderKeepAlive {
    buffers: Vec<metal::Buffer>,
    textures: Vec<metal::Texture>,
    samplers: Vec<metal::SamplerState>,
    pso: Option<metal::RenderPipelineState>,
}

/// A parent-render-encoder state bind that cannot be inherited by a Metal ICB.
///
/// This is separate from [`crate::runtime::icb::IcbStatus`]: that type owns ICB
/// descriptor, command-memory, and command-fill failures. These checks happen
/// later, while applying the decoded draw stream to the parent encoder, and
/// carry the bind/pipeline values needed to diagnose the rejected execute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum MetalIcbInheritanceDecline {
    CullModeUnsupported {
        value: u32,
    },
    FrontFacingUnsupported {
        value: u32,
    },
    /// A live bind names a slot past its class's argument table. One variant
    /// for all three classes and both stages, because
    /// [`crate::runtime::draw::first_bind_past_table`] is one check: six
    /// variants here meant six copies of one bound at six call sites, and the
    /// two sibling draw arms had each drifted to a different answer for the
    /// identical input.
    BindSlotPastTable {
        bind: crate::runtime::draw::PastTableBind,
    },
    VertexBufferMissing {
        buffer_ref: u32,
        index: u32,
        offset: u64,
    },
    FragmentBufferMissing {
        buffer_ref: u32,
        index: u32,
        offset: u64,
    },
    VertexTextureMissing {
        texture_ref: u32,
        index: u32,
        detail: String,
    },
    FragmentTextureMissing {
        texture_ref: u32,
        index: u32,
        detail: String,
    },
    PipelineRefZero,
    PipelineMissing {
        pipeline_ref: u32,
    },
    VertexMtlbMissing {
        function_ref: u32,
    },
    FragmentMtlbMissing {
        function_ref: u32,
    },
    VertexLibraryLoad {
        function_ref: u32,
        detail: String,
    },
    FragmentLibraryLoad {
        function_ref: u32,
        detail: String,
    },
    VertexFunctionCount {
        function_ref: u32,
        count: usize,
    },
    FragmentFunctionCount {
        function_ref: u32,
        count: usize,
    },
    VertexFunctionGet {
        function_ref: u32,
        detail: String,
    },
    FragmentFunctionGet {
        function_ref: u32,
        detail: String,
    },
    VertexDescriptorMissing {
        pipeline_ref: u32,
        attribute_count: usize,
    },
    /// One declared vertex attribute names an `MTLVertexFormat` or
    /// `MTLVertexStepFunction` ordinal Metal does not declare. Distinct from
    /// [`Self::VertexDescriptorMissing`], which is the whole block coming back
    /// empty: this one is a block that would have encoded *around* the bad
    /// attribute and left the shader's `[[stage_in]]` a field short.
    VertexAttributeUnencodable {
        pipeline_ref: u32,
        location: u32,
        value: u32,
    },
    RenderPipelineCreate {
        pipeline_ref: u32,
        detail: String,
    },
    /// Metal answered nil for an object this pass had to create, so the
    /// inheritance cannot be applied.
    ///
    /// `what` names which one. A nil from a texture or a buffer is the device
    /// out of memory; the whole encoder is declined rather than inherited
    /// partially, because a pass missing one of its inherited binds draws with
    /// whatever the encoder held before.
    AllocationFailed {
        what: &'static str,
    },
}

/// How many checks [`MetalIcbInheritanceDecline`] declares.
///
/// The fixture behind
/// [`every_metal_icb_inheritance_check_is_unique_namespaced_and_log_safe`](super::tests)
/// must carry one value per variant, or a check ships with nothing asserting
/// its slug is namespaced, distinct, and free of whitespace. `rustc` cannot
/// count an enum's variants on stable, so the number is written by hand — but
/// it is written *here*, beside the variants, because the same number spelled
/// in the test file went stale by five the day six per-class bind variants
/// became [`MetalIcbInheritanceDecline::BindSlotPastTable`], and nothing was
/// red until the next Apple host ran that arm.
#[cfg(test)]
pub(super) const METAL_ICB_INHERITANCE_CHECKS: usize = 21;

impl crate::observe::Decline for MetalIcbInheritanceDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CullModeUnsupported { .. } => "metal_icb_inherit_cull_mode_unsupported",
            Self::FrontFacingUnsupported { .. } => "metal_icb_inherit_front_facing_unsupported",
            Self::BindSlotPastTable { .. } => "metal_icb_inherit_bind_slot_past_table",
            Self::VertexBufferMissing { .. } => "metal_icb_inherit_vertex_buffer_missing",
            Self::FragmentBufferMissing { .. } => "metal_icb_inherit_fragment_buffer_missing",
            Self::VertexTextureMissing { .. } => "metal_icb_inherit_vertex_texture_missing",
            Self::FragmentTextureMissing { .. } => "metal_icb_inherit_fragment_texture_missing",
            Self::PipelineRefZero => "metal_icb_inherit_pipeline_ref_zero",
            Self::PipelineMissing { .. } => "metal_icb_inherit_pipeline_missing",
            Self::VertexMtlbMissing { .. } => "metal_icb_inherit_vertex_mtlb_missing",
            Self::FragmentMtlbMissing { .. } => "metal_icb_inherit_fragment_mtlb_missing",
            Self::VertexLibraryLoad { .. } => "metal_icb_inherit_vertex_library_load",
            Self::FragmentLibraryLoad { .. } => "metal_icb_inherit_fragment_library_load",
            Self::VertexFunctionCount { .. } => "metal_icb_inherit_vertex_function_count",
            Self::FragmentFunctionCount { .. } => "metal_icb_inherit_fragment_function_count",
            Self::VertexFunctionGet { .. } => "metal_icb_inherit_vertex_function_get",
            Self::FragmentFunctionGet { .. } => "metal_icb_inherit_fragment_function_get",
            Self::VertexDescriptorMissing { .. } => "metal_icb_inherit_vertex_descriptor_missing",
            Self::VertexAttributeUnencodable { .. } => {
                "metal_icb_inherit_vertex_attribute_unencodable"
            }
            Self::RenderPipelineCreate { .. } => "metal_icb_inherit_render_pipeline_create",
            Self::AllocationFailed { .. } => "metal_icb_inherit_allocation_failed",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        fn token(detail: &str) -> String {
            detail.replace(char::is_whitespace, "_")
        }

        match self {
            Self::CullModeUnsupported { value } | Self::FrontFacingUnsupported { value } => {
                vec![("value", value.to_string())]
            }
            Self::BindSlotPastTable { bind } => vec![
                ("class", bind.class.name().to_string()),
                ("stage", bind.stage_name().to_string()),
                ("index", bind.index.to_string()),
                ("table", bind.class.table().to_string()),
                ("ref", bind.resource_ref.to_string()),
            ],
            Self::VertexBufferMissing {
                buffer_ref,
                index,
                offset,
            }
            | Self::FragmentBufferMissing {
                buffer_ref,
                index,
                offset,
            } => vec![
                ("buffer_ref", buffer_ref.to_string()),
                ("index", index.to_string()),
                ("offset", offset.to_string()),
            ],
            Self::VertexTextureMissing {
                texture_ref,
                index,
                detail,
            }
            | Self::FragmentTextureMissing {
                texture_ref,
                index,
                detail,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("index", index.to_string()),
                ("detail", token(detail)),
            ],
            Self::PipelineRefZero => Vec::new(),
            Self::AllocationFailed { what } => vec![("what", (*what).to_string())],
            Self::VertexAttributeUnencodable {
                pipeline_ref,
                location,
                value,
            } => vec![
                ("pipeline_ref", pipeline_ref.to_string()),
                ("location", location.to_string()),
                ("value", value.to_string()),
            ],
            Self::PipelineMissing { pipeline_ref }
            | Self::VertexDescriptorMissing { pipeline_ref, .. }
            | Self::RenderPipelineCreate { pipeline_ref, .. } => {
                let mut fields = vec![("pipeline_ref", pipeline_ref.to_string())];
                match self {
                    Self::VertexDescriptorMissing {
                        attribute_count, ..
                    } => fields.push(("attribute_count", attribute_count.to_string())),
                    Self::RenderPipelineCreate { detail, .. } => {
                        fields.push(("detail", token(detail)))
                    }
                    _ => {}
                }
                fields
            }
            Self::VertexMtlbMissing { function_ref }
            | Self::FragmentMtlbMissing { function_ref } => {
                vec![("function_ref", function_ref.to_string())]
            }
            Self::VertexLibraryLoad {
                function_ref,
                detail,
            }
            | Self::FragmentLibraryLoad {
                function_ref,
                detail,
            }
            | Self::VertexFunctionGet {
                function_ref,
                detail,
            }
            | Self::FragmentFunctionGet {
                function_ref,
                detail,
            } => vec![
                ("function_ref", function_ref.to_string()),
                ("detail", token(detail)),
            ],
            Self::VertexFunctionCount {
                function_ref,
                count,
            }
            | Self::FragmentFunctionCount {
                function_ref,
                count,
            } => vec![
                ("function_ref", function_ref.to_string()),
                ("count", count.to_string()),
            ],
        }
    }
}

/// Apply stream-accumulated state to the parent render encoder before
/// `executeCommandsInBuffer`.
///
/// Metal contract:
/// - **Viewport / scissor / textures / samplers / raster / blend color** always
///   come from the parent encoder (not recordable into `MTLIndirectRenderCommand`
///   for classic ICB).
/// - **Vertex/fragment buffers** are used when the ICB was created with
///   `inheritBuffers=true`.
/// - **Pipeline** is used when created with `inheritPipelineState=true`.
// As the compute twin in `compute_session`: the parent encoder, the device
// state and the ICB request are all needed to decide what the ICB inherits.
#[allow(clippy::too_many_arguments)]
fn apply_icb_encoder_inheritance<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    device: &metal::Device,
    enc: &metal::RenderCommandEncoderRef,
    req: &DrawEncodeRequest,
    icb_desc: &crate::runtime::decode::resource::IndirectCommandBufferDescriptor,
    pass_w: u32,
    pass_h: u32,
    keep: &mut IcbEncoderKeepAlive,
) -> Result<(), MetalIcbInheritanceDecline> {
    use crate::backend::metal::samplers::{make_default_sampler, make_explicit_sampler};
    use metal::*;
    use std::os::raw::c_char;

    // One bound for all three classes and both stages, asked before any resource
    // is resolved. Metal answers an out-of-range argument-table index with a
    // process-aborting exception, so this is where the inheritance stops.
    if let Some(bind) = first_bind_past_table(req) {
        return Err(MetalIcbInheritanceDecline::BindSlotPastTable { bind });
    }

    // Viewport: stream absolute, or full pass when absent (Metal default is not
    // a full drawable — product sets an explicit full RT viewport when the
    // guest stream never issued setViewport).
    // The whole array, through the plural setters, so an ICB executed under a
    // parent encoder inherits every viewport the stream bound rather than its
    // first. `setViewports:count:` with one entry is `setViewport:`, so the
    // single-viewport stream is unchanged.
    if req.viewports.is_empty() {
        enc.set_viewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: pass_w as f64,
            height: pass_h as f64,
            znear: 0.0,
            zfar: 1.0,
        });
    } else {
        let vps: Vec<MTLViewport> = req
            .viewports
            .iter()
            .map(|vp| MTLViewport {
                originX: vp[0],
                originY: vp[1],
                width: vp[2],
                height: vp[3],
                znear: vp[4],
                zfar: vp[5],
            })
            .collect();
        enc.set_viewports(&vps);
    }
    if !req.scissors.is_empty() {
        let rects: Vec<MTLScissorRect> = req
            .scissors
            .iter()
            .map(|r| MTLScissorRect {
                x: r.x as u64,
                y: r.y as u64,
                width: r.width as u64,
                height: r.height as u64,
            })
            .collect();
        enc.set_scissor_rects(&rects);
    }
    if let Some(c) = req.blend_color {
        enc.set_blend_color(c[0], c[1], c[2], c[3]);
    }
    if let Some(c) = req.cull_mode {
        // SDK MTLCullMode: 0=None, 1=Front, 2=Back.
        let mode = match c {
            0 => MTLCullMode::None,
            1 => MTLCullMode::Front,
            2 => MTLCullMode::Back,
            _ => return Err(MetalIcbInheritanceDecline::CullModeUnsupported { value: c }),
        };
        enc.set_cull_mode(mode);
    }
    if let Some(f) = req.front_facing {
        // SDK MTLWinding: 0=Clockwise, 1=CounterClockwise.
        let wind = match f {
            0 => MTLWinding::Clockwise,
            1 => MTLWinding::CounterClockwise,
            _ => {
                return Err(MetalIcbInheritanceDecline::FrontFacingUnsupported { value: f });
            }
        };
        enc.set_front_facing_winding(wind);
    }
    if let Some(d) = req.depth_bias {
        enc.set_depth_bias(d[0], d[1], d[2]);
    }

    // Shared buffer staging (copy into Metal so guest Vec can drop). A
    // successful `load_buffer_bytes` is nonempty by construction: its
    // `host_alloc_len(avail).filter(|n| n > 0)` rejects the zero-span case.
    let stage_mtl_buf = |bytes: &[u8]| -> Result<metal::Buffer, MetalIcbInheritanceDecline> {
        unsafe {
            crate::backend::metal::raw_metal::new_buffer_with_data(
                device,
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalIcbInheritanceDecline::AllocationFailed {
            what: "staged_buffer",
        })
    };

    // Buffers: applied when inheritBuffers or when the request carries binds
    // (inherit path). Metal ignores encoder buffers for ICB draws when
    // inheritBuffers is false; setting them is still safe for textures-only ICBs.
    if icb_desc.inherit_buffers()
        || !req.vertex_buffers.is_empty()
        || !req.fragment_buffers.is_empty()
    {
        for b in req.vertex_buffers.iter() {
            if b.buffer_ref == 0 {
                continue;
            }
            let Some(bytes) = load_buffer_bytes(state, host, req.task_id, b.buffer_ref, b.offset)
            else {
                return Err(MetalIcbInheritanceDecline::VertexBufferMissing {
                    buffer_ref: b.buffer_ref,
                    index: b.index,
                    offset: b.offset,
                });
            };
            let mtl = stage_mtl_buf(&bytes)?;
            enc.set_vertex_buffer(b.index as u64, Some(mtl.as_ref()), 0);
            keep.buffers.push(mtl);
        }
        for b in req.fragment_buffers.iter() {
            if b.buffer_ref == 0 {
                continue;
            }
            let Some(bytes) = load_buffer_bytes(state, host, req.task_id, b.buffer_ref, b.offset)
            else {
                return Err(MetalIcbInheritanceDecline::FragmentBufferMissing {
                    buffer_ref: b.buffer_ref,
                    index: b.index,
                    offset: b.offset,
                });
            };
            let mtl = stage_mtl_buf(&bytes)?;
            enc.set_fragment_buffer(b.index as u64, Some(mtl.as_ref()), 0);
            keep.buffers.push(mtl);
        }
    }

    // Sampled textures — always encoder-side (not in IndirectRenderCommand).
    // Same gate as direct draws: unbound/missing textures must not sample garbage.
    for t in req.vertex_textures.iter() {
        if t.texture_ref == 0 {
            continue;
        }
        let Some((w, h, rgba)) = load_sampled_rgba(state, host, req.task_id, t.texture_ref) else {
            return Err(MetalIcbInheritanceDecline::VertexTextureMissing {
                texture_ref: t.texture_ref,
                index: t.index,
                detail: sample_miss_detail(state, host, req.task_id, t.texture_ref),
            });
        };
        let td = TextureDescriptor::new();
        td.set_texture_type(MTLTextureType::D2);
        td.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        td.set_width(w as u64);
        td.set_height(h as u64);
        td.set_storage_mode(MTLStorageMode::Shared);
        td.set_usage(MTLTextureUsage::ShaderRead);
        let tex = crate::backend::metal::raw_metal::new_texture(device, &td).ok_or(
            MetalIcbInheritanceDecline::AllocationFailed {
                what: "inherited_texture",
            },
        )?;
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w as u64,
                height: h as u64,
                depth: 1,
            },
        };
        tex.replace_region(region, 0, rgba.as_ptr() as *const _, (w as u64) * 4);
        enc.set_vertex_texture(t.index as u64, Some(tex.as_ref()));
        keep.textures.push(tex);
    }
    for t in req.fragment_textures.iter() {
        if t.texture_ref == 0 {
            continue;
        }
        let Some((w, h, rgba)) = load_sampled_rgba(state, host, req.task_id, t.texture_ref) else {
            return Err(MetalIcbInheritanceDecline::FragmentTextureMissing {
                texture_ref: t.texture_ref,
                index: t.index,
                detail: sample_miss_detail(state, host, req.task_id, t.texture_ref),
            });
        };
        let td = TextureDescriptor::new();
        td.set_texture_type(MTLTextureType::D2);
        td.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        td.set_width(w as u64);
        td.set_height(h as u64);
        td.set_storage_mode(MTLStorageMode::Shared);
        td.set_usage(MTLTextureUsage::ShaderRead);
        let tex = crate::backend::metal::raw_metal::new_texture(device, &td).ok_or(
            MetalIcbInheritanceDecline::AllocationFailed {
                what: "inherited_texture",
            },
        )?;
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: w as u64,
                height: h as u64,
                depth: 1,
            },
        };
        tex.replace_region(region, 0, rgba.as_ptr() as *const _, (w as u64) * 4);
        enc.set_fragment_texture(t.index as u64, Some(tex.as_ref()));
        keep.textures.push(tex);
    }

    for s in req.vertex_samplers.iter() {
        if s.sampler_ref == 0 {
            continue;
        }
        let mut err_buf = [0i8; 128];
        let err = (err_buf.as_mut_ptr() as *mut c_char, err_buf.len());
        let reims_vgpu = load_sampler(state, host, req.task_id, s.sampler_ref, s.index)
            .unwrap_or_else(|error| {
                crate::observe::Emit::decline("metal_icb_sampler_fallback", &error)
                    .field("task", req.task_id)
                    .field("pipe", req.pipeline_ref)
                    .field("stage", "vertex")
                    .fail_once(
                        (u64::from(s.sampler_ref) << 32) | (1_u64 << 28) | u64::from(s.index),
                    );
                default_sampler(s.index)
            });
        let mtl = match make_explicit_sampler(device, &reims_vgpu, err) {
            Ok(st) => st,
            Err(status) => {
                crate::observe::Emit::refusal("metal_icb_sampler_fallback", &status)
                    .expect("explicit sampler construction returned a refusal")
                    .field("task", req.task_id)
                    .field("pipe", req.pipeline_ref)
                    .field("stage", "vertex")
                    .field("sampler", s.sampler_ref)
                    .field("index", s.index)
                    .fail_once(s.sampler_ref as u64);
                make_default_sampler(device)
            }
        };
        enc.set_vertex_sampler_state(s.index as u64, Some(mtl.as_ref()));
        keep.samplers.push(mtl);
    }
    for s in req.fragment_samplers.iter() {
        if s.sampler_ref == 0 {
            continue;
        }
        let mut err_buf = [0i8; 128];
        let err = (err_buf.as_mut_ptr() as *mut c_char, err_buf.len());
        let reims_vgpu = load_sampler(state, host, req.task_id, s.sampler_ref, s.index)
            .unwrap_or_else(|error| {
                crate::observe::Emit::decline("metal_icb_sampler_fallback", &error)
                    .field("task", req.task_id)
                    .field("pipe", req.pipeline_ref)
                    .field("stage", "fragment")
                    .fail_once(
                        (u64::from(s.sampler_ref) << 32) | (1_u64 << 27) | u64::from(s.index),
                    );
                default_sampler(s.index)
            });
        let mtl = match make_explicit_sampler(device, &reims_vgpu, err) {
            Ok(st) => st,
            Err(status) => {
                crate::observe::Emit::refusal("metal_icb_sampler_fallback", &status)
                    .expect("explicit sampler construction returned a refusal")
                    .field("task", req.task_id)
                    .field("pipe", req.pipeline_ref)
                    .field("stage", "fragment")
                    .field("sampler", s.sampler_ref)
                    .field("index", s.index)
                    .fail_once(s.sampler_ref as u64);
                make_default_sampler(device)
            }
        };
        enc.set_fragment_sampler_state(s.index as u64, Some(mtl.as_ref()));
        keep.samplers.push(mtl);
    }

    // Pipeline when inheritPipelineState — PSO is not recorded into the slot,
    // so a parent pipeline is required rather than optional.
    if icb_desc.inherit_pipeline_state() {
        if req.pipeline_ref == 0 {
            return Err(MetalIcbInheritanceDecline::PipelineRefZero);
        }
        let pipeline = load_render_pipeline(state, host, req.task_id, req.pipeline_ref).ok_or(
            MetalIcbInheritanceDecline::PipelineMissing {
                pipeline_ref: req.pipeline_ref,
            },
        )?;
        let Some(vert) = load_mtlb(
            state,
            host,
            req.task_id,
            pipeline.vertex_func_ref,
            AirLoadRail::Draw,
        ) else {
            return Err(MetalIcbInheritanceDecline::VertexMtlbMissing {
                function_ref: pipeline.vertex_func_ref,
            });
        };
        let Some(frag) = load_mtlb(
            state,
            host,
            req.task_id,
            pipeline.fragment_func_ref,
            AirLoadRail::Draw,
        ) else {
            return Err(MetalIcbInheritanceDecline::FragmentMtlbMissing {
                function_ref: pipeline.fragment_func_ref,
            });
        };
        let vlib = device.new_library_with_data(&vert).map_err(|error| {
            MetalIcbInheritanceDecline::VertexLibraryLoad {
                function_ref: pipeline.vertex_func_ref,
                detail: format!("{error:?}"),
            }
        })?;
        let flib = device.new_library_with_data(&frag).map_err(|error| {
            MetalIcbInheritanceDecline::FragmentLibraryLoad {
                function_ref: pipeline.fragment_func_ref,
                detail: format!("{error:?}"),
            }
        })?;
        let vnames = vlib.function_names();
        let fnames = flib.function_names();
        if vnames.len() != 1 {
            return Err(MetalIcbInheritanceDecline::VertexFunctionCount {
                function_ref: pipeline.vertex_func_ref,
                count: vnames.len(),
            });
        }
        if fnames.len() != 1 {
            return Err(MetalIcbInheritanceDecline::FragmentFunctionCount {
                function_ref: pipeline.fragment_func_ref,
                count: fnames.len(),
            });
        }
        let vf = vlib.get_function(&vnames[0], None).map_err(|error| {
            MetalIcbInheritanceDecline::VertexFunctionGet {
                function_ref: pipeline.vertex_func_ref,
                detail: format!("{error:?}"),
            }
        })?;
        let ff = flib.get_function(&fnames[0], None).map_err(|error| {
            MetalIcbInheritanceDecline::FragmentFunctionGet {
                function_ref: pipeline.fragment_func_ref,
                detail: format!("{error:?}"),
            }
        })?;
        let pdesc = RenderPipelineDescriptor::new();
        pdesc.set_vertex_function(Some(&vf));
        pdesc.set_fragment_function(Some(&ff));
        pdesc.set_support_indirect_command_buffers(true);
        if let Some(ca) = pdesc.color_attachments().object_at(0) {
            ca.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        }
        match crate::runtime::icb::metal_vertex_descriptor_from_attrs(&pipeline.vertex_attributes) {
            Ok(Some(vd)) => pdesc.set_vertex_descriptor(Some(vd.as_ref())),
            Ok(None) if pipeline.vertex_attributes.is_empty() => {}
            Ok(None) => {
                return Err(MetalIcbInheritanceDecline::VertexDescriptorMissing {
                    pipeline_ref: req.pipeline_ref,
                    attribute_count: pipeline.vertex_attributes.len(),
                });
            }
            // This arm is why the builder returns a `Result`. It already
            // refused a descriptor that came back empty; what it could not see
            // was one that came back *partial*, because a surviving attribute
            // made the whole set look encodable.
            Err(refusal) => {
                return Err(MetalIcbInheritanceDecline::VertexAttributeUnencodable {
                    pipeline_ref: req.pipeline_ref,
                    location: refusal.location,
                    value: refusal.value,
                });
            }
        }
        let pso = device.new_render_pipeline_state(&pdesc).map_err(|error| {
            MetalIcbInheritanceDecline::RenderPipelineCreate {
                pipeline_ref: req.pipeline_ref,
                detail: format!("{error:?}"),
            }
        })?;
        enc.set_render_pipeline_state(&pso);
        keep.pso = Some(pso);
    }

    Ok(())
}

/// Name an ICB refusal on the render rail, then collapse it to `EncodeStatus`.
///
/// The ICB's own check — one of 153 — used to arrive at `exec` as a bare
/// `bad_args` or `metal_failed`, and the log could not tell a missing pipeline
/// object from a mesh library that failed to build. Emitting here is what makes
/// the ICB vocabulary reach the sink on the render path; the collapse now
/// **carries** that slug onward, so the boundary counter in `exec` names the
/// ICB's check rather than re-stating the class it was flattened into.
///
/// Latched per `(reason, icb_ref)`: the guest re-submits the same ICB every
/// frame, so an unlatched line would be one per frame per refusal.
fn render_icb_declined(
    e: crate::runtime::icb::IcbStatus,
    task_id: u32,
    icb_ref: u32,
) -> EncodeStatus {
    use crate::observe::Decline as _;
    use crate::runtime::icb::IcbStatus;
    crate::observe::Emit::decline("render_icb", &e)
        .field("task", task_id)
        .field("icb", icb_ref)
        .fail_once(icb_ref as u64);
    // The class each ICB refusal collapses to is unchanged; what is new is that
    // the reason travels with it. Forwarded, not re-named — `IcbStatus` owns
    // these slugs and its own registry row counts them.
    let slug = e.slug();
    match e {
        IcbStatus::Missing(_) => EncodeStatus::BadArgs(slug),
        IcbStatus::NoMetal(_) => EncodeStatus::NoMetal(slug),
        IcbStatus::Args(_) | IcbStatus::BadDescriptor(_) | IcbStatus::MetalFailed(_) => {
            EncodeStatus::MetalFailed(slug)
        }
        IcbStatus::Unsupported(_) => EncodeStatus::Unsupported(slug),
    }
}

/// Materializes type-7 ICB `0x36`, optionally re-fills from bound command
/// memory (`0x1d1` / associate), then `executeCommandsInBuffer:withRange:`.
/// Empty ranges are valid Metal (no-op). Color writeback mirrors draw path.
///
/// Before execute, applies parent-encoder inheritance from [`DrawEncodeRequest`]
/// (viewport, scissor, buffers when `inheritBuffers`, textures/samplers, and
/// pipeline when `inheritPipelineState`).
pub fn encode_icb_execute_and_writeback<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    req: &DrawEncodeRequest,
    icb_ref: u32,
    range_location: u64,
    range_length: u64,
) -> EncodeStatus {
    use crate::backend::metal::runtime::{system_device, thread_queue};
    use crate::contract::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use crate::runtime::icb::{fill_icb_from_command_memory, resolve_metal_icb};
    use metal::*;

    if icb_ref == 0 {
        return EncodeStatus::BadArgs("icb_exec_ref_zero");
    }
    let color_list: Vec<ColorRtRequest> = req.colors.clone();
    if color_list.is_empty() {
        return EncodeStatus::BadArgs("icb_exec_no_color_target");
    }
    let width = color_list[0].width;
    let height = color_list[0].height;
    if width == 0
        || height == 0
        || color_list.iter().any(|c| {
            c.width != width || c.height != height || (c.mapping_id == 0 && c.target_gva == 0)
        })
    {
        return EncodeStatus::BadArgs("icb_exec_geom_mismatch");
    }

    if let Some(error) = icb_depth_stencil_decline(req) {
        crate::observe::Emit::decline("metal_icb_depth_stencil_refused", &error)
            .field("task", req.task_id)
            .field("pipe", req.pipeline_ref)
            .field("icb", icb_ref)
            .fail_once(u64::from(icb_ref));
        return EncodeStatus::MetalFailed(error.slug());
    }

    let Some(device) = system_device() else {
        return EncodeStatus::NoMetal("icb_exec_no_metal_device");
    };
    let queue = thread_queue(device);

    let (icb_desc, icb) = match resolve_metal_icb(state, host, req.task_id, icb_ref) {
        Ok(v) => v,
        Err(e) => return render_icb_declined(e, req.task_id, icb_ref),
    };
    let size = icb.size();
    if range_location.saturating_add(range_length) > size {
        return EncodeStatus::BadArgs("icb_exec_range_past_size");
    }
    // Fill the host ICB's slots from the guest's command memory. What an
    // unfilled ICB costs, and which outcomes an execute may carry on from, is
    // `icb_fill_outcome`'s to say — the compute arm asks the same function, so
    // the two cannot answer it differently.
    match crate::runtime::icb::icb_fill_outcome(
        fill_icb_from_command_memory(
            state,
            host,
            req.task_id,
            icb_ref,
            range_location,
            range_length,
        ),
        req.task_id,
        icb_ref,
    ) {
        Ok(()) => {}
        Err(e) => return render_icb_declined(e, req.task_id, icb_ref),
    }

    // Build color RT textures (seeded from mapping / clear).
    let mut color_tex: Vec<(u32, Texture, Vec<u8>)> = Vec::new();
    for c in &color_list {
        // This path renders every colour attachment as BGRA8Unorm whatever the
        // guest declared, and writes it back unconverted, so a narrower or
        // wider attachment is reinterpreted rather than translated. Say so:
        // the alternative is a well-formed frame in the wrong format, which is
        // indistinguishable from a correct one until something samples it.
        if c.format != 0 && c.format != MTL_FORMAT_BGRA8_UNORM {
            crate::runtime::drain::census::note_store_route("icb_color_format_reinterpreted");
            if crate::observe::first_sight("icb_color_not_bgra8", u64::from(c.format)) {
                crate::observe::fail(format!(
                    "icb_color_format_reinterpreted reason=icb_color_not_bgra8 \
                     mapping={} format={:#x} rendered_as=bgra8_unorm",
                    c.mapping_id, c.format
                ));
            }
        }
        // The texture below is BGRA8Unorm, so its staging length and the row
        // stride `replace_region` is given must come from one place — see
        // `contract::extent::tight_image_layout`, which carries what happened
        // when they did not.
        let Some((row_bytes, nbytes)) =
            crate::contract::extent::tight_image_layout(width, height, RGBA8_BPP)
        else {
            return EncodeStatus::BadArgs("icb_color_target_degenerate_geometry");
        };
        let mut seed = c
            .target_seed_rgba
            .clone()
            .unwrap_or_else(|| vec![0u8; nbytes]);
        if seed.len() < nbytes {
            seed.resize(nbytes, 0);
        }
        let td = TextureDescriptor::new();
        td.set_texture_type(MTLTextureType::D2);
        td.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        td.set_width(width as u64);
        td.set_height(height as u64);
        td.set_storage_mode(MTLStorageMode::Shared);
        td.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        let Some(tex) = crate::backend::metal::raw_metal::new_texture(device, &td) else {
            return EncodeStatus::MetalFailed("icb_color_texture_alloc_failed");
        };
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };
        tex.replace_region(region, 0, seed.as_ptr() as *const _, u64::from(row_bytes));
        color_tex.push((c.mapping_id, tex, seed));
    }

    let pass = RenderPassDescriptor::new();
    for (i, (_, tex, _)) in color_tex.iter().enumerate() {
        let att = pass.color_attachments().object_at(i as u64).unwrap();
        att.set_texture(Some(tex));
        att.set_load_action(MTLLoadAction::Load);
        att.set_store_action(MTLStoreAction::Store);
    }

    let Some(cb) = crate::backend::metal::raw_metal::new_command_buffer(&queue) else {
        return EncodeStatus::MetalFailed("icb_command_buffer_unavailable");
    };
    let cb = cb.to_owned();
    let Some(enc) = crate::backend::metal::raw_metal::new_render_command_encoder(&cb, pass) else {
        return EncodeStatus::MetalFailed("icb_render_encoder_unavailable");
    };
    // Parent-encoder inheritance: stream viewport/scissor/buffers/textures/samplers
    // and (when ICB create flags say so) pipeline. Textures/samplers are never
    // recordable into IndirectRenderCommand — they always come from the encoder.
    let mut inherit_keep = IcbEncoderKeepAlive::default();
    if let Err(error) = apply_icb_encoder_inheritance(
        state,
        host,
        device,
        enc,
        req,
        &icb_desc,
        width,
        height,
        &mut inherit_keep,
    ) {
        crate::observe::Emit::decline("metal_icb_inheritance", &error)
            .field("task", req.task_id)
            .field("pipe", req.pipeline_ref)
            .field("icb", icb_ref)
            .fail_once(u64::from(icb_ref));
        enc.end_encoding();
        return EncodeStatus::MetalFailed(error.slug());
    }
    enc.execute_commands_in_buffer(
        icb.as_ref(),
        NSRange {
            location: range_location,
            length: range_length,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    drop(inherit_keep);
    if cb.status() == MTLCommandBufferStatus::Error {
        return EncodeStatus::MetalFailed("icb_exec_command_buffer_error");
    }

    // Writeback each color RT (type-11 mapping or type-2/3 GVA).
    // Same one derivation as the seed side above, so the two halves of this
    // function cannot disagree about the layout of the buffer they share.
    let Some((stride, need)) = crate::contract::extent::tight_image_layout(width, height, RGBA8_BPP)
    else {
        return EncodeStatus::BadArgs("icb_color_target_degenerate_geometry");
    };
    let mut any_write = false;
    for (i, (mapping_id, tex, seed)) in color_tex.iter().enumerate() {
        let c = &color_list[i];
        if c.store_action == MTL_STORE_ACTION_DONT_CARE {
            continue;
        }
        let mut pixels = seed.clone();
        if pixels.len() < need {
            pixels.resize(need, 0);
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width: width as u64,
                height: height as u64,
                depth: 1,
            },
        };
        tex.get_bytes(pixels.as_mut_ptr() as *mut _, (width as u64) * 4, region, 0);
        let wrote = if c.target_gva != 0 {
            // Shared texture is BGRA8; convert to RGBA for write_gva_rgba8.
            //
            // Walked over `need` — the length `tight_image_layout` just returned
            // — rather than over a freshly computed `(width * height)`. That
            // product is in `u32` and overflows at 65536×65536, which the check
            // above does not prevent: it refuses a zero axis and asks
            // `tight_image_layout` for the byte count, and that one widens to
            // `u64` and checks, so it answers happily for geometries this
            // multiplication cannot express. The same quantity twice, and only
            // one of the two derivations right — which is the defect
            // `tight_image_layout` itself was written to retire, one level up.
            for texel in pixels[..need].chunks_exact_mut(RGBA8_BPP as usize) {
                texel.swap(0, 2);
            }
            write_gva_rgba8(
                state,
                host,
                req.task_id,
                c.target_gva,
                width,
                height,
                c.row_stride,
                c.format,
                &pixels,
            )
            .is_ok()
        } else if *mapping_id != 0 {
            let _ = mapper::ensure_resolved_for_scanout(state, host, *mapping_id);
            mapping_write::write_bgra8(state, host, *mapping_id, &pixels, stride, width, height)
        } else {
            false
        };
        if wrote {
            any_write = true;
            if *mapping_id != 0 {
                crate::runtime::scanout::note_front_buffer_writeback(
                    state,
                    host,
                    *mapping_id,
                    width,
                    height,
                    c.format,
                );
            }
        }
    }
    if !any_write {
        return EncodeStatus::WritebackFailed("icb_exec_writeback_none");
    }
    EncodeStatus::Ok
}

#[cfg(test)]
mod metal_icb_split_tests {
    use super::*;

    #[test]
    fn metal_icb_depth_stencil_preflight_refuses_state_the_encoder_does_not_apply() {
        use crate::observe::Emit;

        let req = DrawEncodeRequest {
            depth_stencil_ref: 61,
            // A bound attachment is one naming a texture. These used to say
            // `present: true` over a zero `texture_ref`, which is a pair the
            // decoder cannot produce — so the preflight was being proved
            // against a state no guest can reach.
            depth_attach: Some(DepthAttachment {
                texture_ref: 41,
                ..DepthAttachment::default()
            }),
            stencil_attach: Some(StencilAttachment {
                texture_ref: 42,
                ..StencilAttachment::default()
            }),
            ..DrawEncodeRequest::default()
        };
        let decline = icb_depth_stencil_decline(&req)
            .expect("bound depth/stencil state cannot silently reach a color-only ICB pass");
        assert_eq!(
            decline,
            MetalStateDecline::IcbDepthStencilUnsupported {
                depth_stencil_ref: 61,
                depth_attachment: true,
                stencil_attachment: true,
            }
        );
        assert_eq!(
            Emit::decline("metal_icb_depth_stencil_refused", &decline)
                .field("task", 7)
                .field("pipe", 9)
                .field("icb", 11)
                .render(),
            "metal_icb_depth_stencil_refused reason=metal_icb_depth_stencil_unsupported \
             depth_stencil_ref=61 depth_attachment=1 stencil_attachment=1 \
             task=7 pipe=9 icb=11"
        );

        assert!(
            icb_depth_stencil_decline(&DrawEncodeRequest::default()).is_none(),
            "genuinely unbound depth/stencil state is expected control flow"
        );
    }
}
