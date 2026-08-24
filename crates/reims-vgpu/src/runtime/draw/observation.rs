//! Observation of completed semantic draws.
//!
//! This module may inspect completion output, but it cannot choose execution or
//! Store behavior. The prepared request and completion route are already fixed
//! before any function here runs.

use super::*;

fn sampled_source_note(image: &reims_vgpu_core::SampledImageResource) -> String {
    use reims_vgpu_core::SampledSource;
    let source = match &image.source {
        SampledSource::Null => "null".to_string(),
        SampledSource::Bytes(bytes) => format!("bytes:{}", bytes.len()),
        SampledSource::Target(identity) => format!("target:{identity:?}"),
        SampledSource::Attachment { identity, initial } => {
            format!("attachment:{identity:?}:{initial:?}")
        }
        SampledSource::GuestImage(source, _) => format!(
            "guest_image:direct={}:mips={}:view={:?}",
            u8::from(source.direct.is_some()),
            source.allocation.mips.len(),
            source.view
        ),
        SampledSource::GuestRuns(source, _) => format!(
            "guest_runs:off={}:len={}:row={}",
            source.source_offset, source.total_len, source.row_length_texels
        ),
    };
    format!(
        "b{}[{}]:{}x{}:{:?}:swizzle={:?}:{source}",
        image.binding, image.array_element, image.width, image.height, image.format, image.swizzle
    )
}

fn screen_resource_join_line(
    pipeline_ref: u32,
    resources: &reims_vgpu_core::DrawRequest,
) -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scissors = resources
        .scissors
        .iter()
        .map(|s| format!("{}x{}+{}+{}", s.width, s.height, s.x, s.y))
        .collect::<Vec<_>>()
        .join(",");
    let viewports = resources
        .viewports
        .iter()
        .map(|v| {
            format!(
                "{:.1}x{:.1}+{:.1}+{:.1}@{:.3}..{:.3}",
                v.width, v.height, v.x, v.y, v.min_depth, v.max_depth
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let sampled = resources
        .sampled_images
        .iter()
        .map(sampled_source_note)
        .collect::<Vec<_>>()
        .join(";");
    format!(
        "screen_resource_join seq={seq} pipe={pipeline_ref} target={:?} extent={}x{} \
         viewport=[{viewports}] scissor=[{scissors}] load={:?} clear={:?} continues={}/{} \
         from_target={} blend={:?} mask={:?} sampled=[{sampled}]",
        resources.target_identity,
        resources.width,
        resources.height,
        resources.color_load_action,
        resources.target_clear,
        u8::from(resources.continues_render_pass),
        u8::from(resources.render_pass_continues),
        u8::from(resources.load_from_target),
        resources.blend,
        resources.color_write_mask,
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PixelSummary {
    rgb_nonzero: usize,
    max_rgb: u8,
    first: [u8; 4],
}

fn pixel_summary(pixels: &[u8]) -> PixelSummary {
    let mut rgb_nonzero = 0;
    let mut max_rgb = 0;
    for pixel in pixels.chunks_exact(4) {
        let value = pixel[0].max(pixel[1]).max(pixel[2]);
        rgb_nonzero += usize::from(value != 0);
        max_rgb = max_rgb.max(value);
    }
    let mut first = [0; 4];
    let prefix = pixels.len().min(first.len());
    first[..prefix].copy_from_slice(&pixels[..prefix]);
    PixelSummary {
        rgb_nonzero,
        max_rgb,
        first,
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the observer records one already-planned draw without owning a second aggregate"
)]
pub(super) fn observe_prepared_resources(
    state: &mut Device,
    req: &DrawEncodeRequest,
    resources: &reims_vgpu_core::DrawRequest,
    v_variant: &reims_vgpu_core::PreparedShaderVariant,
    f_variant: &reims_vgpu_core::PreparedShaderVariant,
    sampler_origin: &std::collections::BTreeMap<u32, u8>,
    w: u32,
    h: u32,
    vertex_count: u32,
) -> bool {
    let census_verbose = crate::observe::draw_log_enabled();
    // The per-draw resource census describes the *decoded* request — vertex
    // attribute declarations, storage-buffer bindings, sampler state, colour
    // targets. It is verbose-gated (REIMS_VGPU_DRAW_LOG →
    // /tmp/reims-vgpu-draw.log) because it costs a `format!` per binding.
    if census_verbose {
        crate::observe::line(screen_resource_join_line(req.pipeline_ref, resources));
        let attr_meta: String = resources
            .vertex_attributes
            .iter()
            .map(|a| {
                format!(
                    "L{}:fmt={:?}:off={}:str={}:sf={:?}:sr={}:n={}",
                    a.location,
                    a.format,
                    a.offset,
                    a.stride,
                    a.step_function,
                    a.step_rate,
                    a.content.len()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let ssbo_meta: String = resources
            .storage_buffers
            .iter()
            .map(|b| format!("b{}:n={}", b.binding, b.content.len()))
            .collect::<Vec<_>>()
            .join(";");
        let sampler_meta: String = resources
            .samplers
            .iter()
            .map(|s| {
                format!(
                    "b{}:un={}:min={:?}:mag={:?}:mip={:?}:lod={}/{}:uvw={:?}/{:?}/{:?}",
                    s.binding,
                    s.unnormalized_coordinates as u8,
                    s.min_filter,
                    s.mag_filter,
                    s.mip_filter,
                    s.lod_min_f32(),
                    s.lod_max_f32(),
                    s.address_mode_u,
                    s.address_mode_v,
                    s.address_mode_w
                )
            })
            .collect::<Vec<_>>()
            .join(";");
        crate::observe::line(format!(
            "linux_m2v_resources pipe={} {}x{} vtx={} attrs={} ssbo={} img={} smp={} rt_n={} rt=[{}] seed={} idx={} idx_n={} meta=[{}] ssbo=[{}] sampler=[{}]",
        req.pipeline_ref,
        w,
        h,
        vertex_count,
        resources.vertex_attributes.len(),
        resources.storage_buffers.len(),
        resources.sampled_images.len(),
        resources.samplers.len(),
        req.colors.len(),
        color_target_diag(&req.colors),
            (resources.target_rgba8.is_some()
            || resources
                .target_guest
                .as_ref()
                .is_some_and(|target| target.seed().is_some())) as u8,
        resources.indexed.is_some() as u8,
        resources.indexed.as_ref().map(|i| i.index_count).unwrap_or(0),
        attr_meta,
        ssbo_meta,
        sampler_meta
    ));
    }

    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::AssembleTrail);
    // Asked of the module rather than of m2v's reflection, which is the
    // whole point: the render path's existing unbound guard walks
    // `f_shader.interface.bindings`, so a binding the translated SPIR-V
    // carries and the reflection omits is checked by nothing.
    // `descriptor_static_use` cannot close it either — it answers
    // `NotDeclared` for anything that is not a `UniformConstant`, which by
    // construction excludes every storage buffer.
    // Memoized on the `Arc`, because this is a per-draw walk of the whole
    // module and the words behind an `Arc` cannot change.
    let frag_declared_bindings = f_variant.declared_bindings.clone();
    let frag_layout_bindings: Vec<u32> = resources
        .storage_buffers
        .iter()
        .map(|s| s.binding)
        .chain(resources.sampled_images.iter().map(|i| i.binding))
        .chain(resources.samplers.iter().map(|s| s.binding))
        .collect();
    // The hang trail, recorded here because this is the last point at which
    // the guest's pipeline ref and both translated module sizes are in scope
    // together: past it the engine keys on digests and the ref is gone. See
    // the executor's hang-trail service for what reads it and why a counter
    // could not answer the question.
    // What the draw is about to sample, lowest binding first. The trail's
    // whole subject is a fragment module that walks a pointer chain through
    // a sampled image, and until this it recorded the module's *size* and
    // nothing about its inputs — so a wedged boot could not say which rail
    // supplied the walked texture, what format the shader would read it as,
    // or whether the extent was the one the guest meant.
    //
    // Sorted here rather than relied upon: `sampled_images` is in the order
    // the two texture loops pushed it, vertex stage first, so the fragment
    // bindings are neither first nor contiguous.
    let mut sampled_notes = Vec::with_capacity(resources.sampled_images.len());
    let mut by_binding: Vec<&reims_vgpu_core::SampledImageResource> =
        resources.sampled_images.iter().collect();
    by_binding.sort_unstable_by_key(|i| i.binding);
    for image in by_binding {
        sampled_notes.push(crate::runtime::executor::DrawHangSampledNote {
            binding: image.binding,
            kind: match &image.source {
                reims_vgpu_core::SampledSource::Null => 0,
                reims_vgpu_core::SampledSource::Bytes(_) => 1,
                reims_vgpu_core::SampledSource::Target(_)
                | reims_vgpu_core::SampledSource::Attachment { .. } => 2,
                reims_vgpu_core::SampledSource::GuestRuns(..) => 3,
                reims_vgpu_core::SampledSource::GuestImage(..) => 6,
            },
            format: image.format,
            width: image.width,
            height: image.height,
            // Only the CPU-bytes rail has bytes here to read. The gather
            // rail's texels are in guest RAM and the target rail's are on
            // the GPU; reading either one would be a device-memory access
            // taken to write a log line, which is not a trade this makes.
            texel0: match &image.source {
                reims_vgpu_core::SampledSource::Null => 0,
                reims_vgpu_core::SampledSource::Bytes(b) => b
                    .get(..4)
                    .map(|t| u32::from_le_bytes([t[0], t[1], t[2], t[3]]))
                    .unwrap_or(0),
                _ => 0,
            },
        });
    }
    // And what it will sample them *through*. All four of the uber shader's
    // unbounded loops share one sampler, and a `LINEAR` filter on a texture
    // whose texels are the next UV walks a blend of two cells rather than
    // either — so the third of the wedge's three hypotheses is a property of
    // this list and of nothing the trail recorded before.
    let mut sampler_notes = Vec::with_capacity(resources.samplers.len());
    let mut smp_by_binding: Vec<&reims_vgpu_core::SamplerResource> =
        resources.samplers.iter().collect();
    smp_by_binding.sort_unstable_by_key(|s| s.binding);
    for smp in smp_by_binding {
        use reims_vgpu_core::{SamplerAddressMode as A, SamplerFilter as F, SamplerMipFilter as M};
        let filter = |f: F| match f {
            F::Nearest => b'N',
            F::Linear => b'L',
        };
        let address = |a: A| match a {
            A::ClampToEdge => b'e',
            A::MirrorClampToEdge => b'E',
            A::Repeat => b'r',
            A::MirrorRepeat => b'R',
            A::ClampToZero => b'z',
            A::ClampToBorderColor => b'b',
        };
        sampler_notes.push(crate::runtime::executor::DrawHangSamplerNote {
            binding: smp.binding,
            min_filter: filter(smp.min_filter),
            mag_filter: filter(smp.mag_filter),
            mip_filter: match smp.mip_filter {
                M::NotMipmapped => b'n',
                M::Nearest => b'N',
                M::Linear => b'L',
            },
            address_u: address(smp.address_mode_u),
            address_v: address(smp.address_mode_v),
            // `?` is a sampler that reached the list by a route that did not
            // record where its state came from, which is the reading that
            // would send the next session looking for a fourth path rather
            // than concluding anything about the three.
            provenance: sampler_origin.get(&smp.binding).copied().unwrap_or(b'?'),
            unnormalized: smp.unnormalized_coordinates,
            lod_min: smp.lod_min,
            lod_max: smp.lod_max,
        });
    }
    state
        .executor
        .note_draw_hang_candidate(crate::runtime::executor::DrawHangCandidate {
            sampled: sampled_notes,
            samplers: sampler_notes,
            pipeline_ref: req.pipeline_ref,
            vert_words: v_variant.word_count,
            frag_words: f_variant.word_count,
            width: w,
            height: h,
            vertex_count,
            instance_count: req.instance_count,
            indexed: resources.indexed.as_ref().map(|indexed| {
                use crate::runtime::executor::{DrawHangIndexSource, DrawHangIndexedNote};
                use reims_vgpu_core::BufferContent;
                DrawHangIndexedNote {
                    index_count: indexed.index_count,
                    index_width: indexed.index_type.byte_size() as u8,
                    vertex_offset: indexed.vertex_offset,
                    base_instance: resources.base_instance,
                    byte_len: indexed.content.len() as u64,
                    source: match &indexed.content {
                        BufferContent::Bytes(_) => DrawHangIndexSource::CpuBytes,
                        BufferContent::GuestRuns(_) => DrawHangIndexSource::GuestRuns,
                    },
                }
            }),
            // Asked of the module rather than of m2v's reflection, which is the
            // whole point: the render path's existing unbound guard walks
            // `f_shader.interface.bindings`, so a binding the translated
            // SPIR-V carries and the reflection omits is checked by nothing.
            // `descriptor_static_use` cannot close it either — it answers
            // `NotDeclared` for anything that is not a `UniformConstant`, which
            // by construction excludes every storage buffer.
            fragment_declared_bindings: frag_declared_bindings,
            // What the engine will build the layout from: the storage binds this
            // draw resolved, at the numbers they will carry, plus the textures
            // and samplers it provided at theirs.
            fragment_provided_bindings: frag_layout_bindings,
        });
    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Assemble);

    census_verbose
}
pub(super) fn report_completed_pixels(
    enabled: bool,
    pipeline_ref: u32,
    width: u32,
    height: u32,
    output: &reims_vgpu_core::DrawOutput,
) {
    if !enabled {
        return;
    }
    if output.pixels.is_empty() {
        crate::observe::line(format!(
            "linux_m2v_pixels pipe={pipeline_ref} {width}x{height} skip_readback=1 \
             (no CPU pixels; see import_content)"
        ));
        return;
    }
    let summary = pixel_summary(&output.pixels);
    crate::observe::line(format!(
        "linux_m2v_pixels pipe={pipeline_ref} {width}x{height} \
         rgb_nz={} max_rgb={} px0=[{},{},{},{}]",
        summary.rgb_nonzero,
        summary.max_rgb,
        summary.first[0],
        summary.first[1],
        summary.first[2],
        summary.first[3],
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_observation_ignores_alpha_and_never_changes_completion_bytes() {
        let pixels = [0, 0, 0, 255, 1, 2, 3, 0, 9, 4, 8, 7];
        let before = pixels;
        assert_eq!(
            pixel_summary(&pixels),
            PixelSummary {
                rgb_nonzero: 2,
                max_rgb: 9,
                first: [0, 0, 0, 255],
            }
        );
        assert_eq!(pixels, before, "observation cannot rewrite Store output");
    }

    #[test]
    fn screen_join_names_target_scissor_load_and_sample_source() {
        let request = reims_vgpu_core::DrawRequest {
            width: 1920,
            height: 1080,
            target_identity: Some(crate::model::TargetIdentity::Gva {
                gva: 0x1000,
                width: 1920,
                height: 1080,
                generation: 7,
                format: reims_vgpu_protocol::TexelLayout::Bgra8,
            }),
            scissors: vec![reims_vgpu_core::ScissorResource {
                x: 900,
                y: 400,
                width: 160,
                height: 48,
            }],
            viewports: vec![reims_vgpu_core::ViewportResource {
                x: 640.0,
                y: 180.0,
                width: 768.0,
                height: 64.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
            color_load_action: reims_vgpu_core::ColorLoadAction::Load,
            load_from_target: true,
            sampled_images: vec![reims_vgpu_core::SampledImageResource {
                binding: 3,
                array_element: 0,
                descriptor_count: 1,
                width: 16,
                height: 16,
                layers: 1,
                arrayed: false,
                volume: false,
                cube: false,
                one_dim: false,
                multisampled: false,
                source: reims_vgpu_core::SampledSource::Bytes(std::sync::Arc::new(vec![1; 16])),
                content: None,
                byte_origin: Default::default(),
                format: reims_vgpu_protocol::ImageFormat::linear(
                    reims_vgpu_protocol::TexelLayout::Rgba8,
                ),
                identity: None,
                resource_lifetime: None,
                swizzle: reims_vgpu_protocol::SwizzlePlan::default(),
            }],
            ..Default::default()
        };
        let line = screen_resource_join_line(42, &request);
        assert!(line.contains("pipe=42"));
        assert!(line.contains("160x48+900+400"));
        assert!(line.contains("768.0x64.0+640.0+180.0@0.000..1.000"));
        assert!(line.contains("load=Load"));
        assert!(line.contains("clear=[0.0, 0.0, 0.0, 0.0]"));
        assert!(line.contains("b3[0]:16x16"));
        assert!(line.contains("bytes:16"));
    }
}
