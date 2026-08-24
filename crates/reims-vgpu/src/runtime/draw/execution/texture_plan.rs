//! Complete sampled-texture planning for one resolved draw.
//!
//! Texture resolution, semantic shape/access checks, binding projection, byte
//! provenance, view swizzles, and required null descriptors are owned here.
//! The executor receives a complete image list or one typed preparation refusal.

use super::*;

fn reflected_sampled_binding(
    interface: &reims_vgpu_core::ShaderInterface,
    metal_index: u32,
) -> Result<
    Option<(
        reims_vgpu_core::ReflectedTextureDescriptor,
        reims_vgpu_core::SampledImageKind,
    )>,
    reims_vgpu_core::ReflectedSampledKind,
> {
    let Some(descriptor) = interface.texture_descriptor(metal_index) else {
        return Ok(None);
    };
    match interface.sampled_kind(descriptor.binding) {
        reims_vgpu_core::ReflectedSampledKind::Kind(kind) => Ok(Some((descriptor, kind))),
        other => Err(other),
    }
}

fn guest_image_plane_count(layout: reims_vgpu_memory::GuestImageLayout) -> u32 {
    if layout.is_volume() {
        layout.depth()
    } else {
        layout.array_layers()
    }
}

fn sampled_extent(one_dimensional: bool, width: u32, height: u32) -> Option<(u32, u32)> {
    (!one_dimensional || (width != 0 && height == 1)).then_some((width, height))
}

#[allow(
    clippy::too_many_arguments,
    reason = "the planner consumes both resolved shader interfaces and their selected numbering variants"
)]
pub(super) fn plan_sampled_textures<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    vertex_shader: &reims_vgpu_core::PreparedShaderFamily,
    fragment_shader: &reims_vgpu_core::PreparedShaderFamily,
    vertex_variant: &reims_vgpu_core::PreparedShaderVariant,
    fragment_variant: &reims_vgpu_core::PreparedShaderVariant,
    fragment_unbound_used: &[u32],
    gva_alloc_generation: u64,
) -> Result<Vec<reims_vgpu_core::SampledImageResource>, DrawPreparationDecline> {
    let v_shader = vertex_shader;
    let f_shader = fragment_shader;
    let v_variant = vertex_variant;
    let f_variant = fragment_variant;
    let frag_unbound_used_textures = fragment_unbound_used;

    crate::runtime::chain_phase::enter(crate::runtime::chain_phase::Phase::Sampled);
    // The four `sampled_phase` spans below divide this phase's `sampled_us`,
    // the same way `bind_phase` divides `binds_us`. Counted here rather than
    // where a span opens, so a draw that samples nothing is still in the
    // denominator.
    crate::runtime::sampled_phase::note_sampled();
    // Sampled textures and samplers occupy independent reflected ranges.
    // Texture and sampler **indices are independent**. Their exact executable
    // bindings come from disjoint reflected descriptor classes; pairing a
    // sampler to the texture index can therefore leave the sampler descriptor
    // the shader actually names empty.
    let mut images: Vec<reims_vgpu_core::SampledImageResource> = Vec::new();
    {
        let mut push_tex = |index: u32,
                            texture_ref: u32,
                            retained: Option<&std::sync::Arc<crate::model::TaskResource>>,
                            frag_stage: bool|
         -> Result<(), DrawPreparationDecline> {
            if texture_ref == 0 {
                return Ok(());
            }
            let interface = if frag_stage {
                &f_shader.interface
            } else {
                &v_shader.interface
            };
            let variant = if frag_stage { f_variant } else { v_variant };
            // A guest encoder may bind more texture slots than this shader
            // statically uses. Such a slot has no descriptor in the translated
            // interface and therefore contributes no Vulkan resource. Treating
            // absence as a 2D texture manufactures both a shader type and GPU
            // work that the contract never requested.
            let Some((reflected_descriptor, image_kind)) =
                reflected_sampled_binding(interface, index).map_err(|reason| {
                    DrawPreparationDecline::TextureDimensionUnsupported {
                        stage: if frag_stage { "fragment" } else { "vertex" },
                        index,
                        texture_ref,
                        binding: variant.texture_binding(index, None),
                        kind: match reason {
                            reims_vgpu_core::ReflectedSampledKind::Absent => "reflected_absent",
                            reims_vgpu_core::ReflectedSampledKind::Unsupported => {
                                "reflected_unsupported"
                            }
                            reims_vgpu_core::ReflectedSampledKind::Kind(_) => unreachable!(),
                        }
                        .into(),
                    }
                })?
            else {
                return Ok(());
            };
            let img_bind = variant.texture_binding(index, Some(reflected_descriptor.binding));
            use reims_vgpu_core::ReflectedTextureAccess;
            let unsupported = match reflected_descriptor.access {
                ReflectedTextureAccess::Sampled => None,
                ReflectedTextureAccess::Storage => Some("storage"),
                ReflectedTextureAccess::Unknown => Some("unknown"),
            };
            if let Some(access) = unsupported {
                return Err(DrawPreparationDecline::TextureAccessUnsupported {
                    stage: if frag_stage { "fragment" } else { "vertex" },
                    index,
                    texture_ref,
                    binding: img_bind,
                    access,
                });
            }
            // Texture dimensionality comes solely from the translator's
            // reflection. Resolve it before the content source so a backing is
            // offered to the direct-image rail only with the descriptor's real
            // shader-visible shape.
            let Some(shape) = sampled_image_shape(image_kind) else {
                return Err(DrawPreparationDecline::TextureDimensionUnsupported {
                    stage: if frag_stage { "fragment" } else { "vertex" },
                    index,
                    texture_ref,
                    binding: img_bind,
                    kind: format!("{image_kind:?}"),
                });
            };
            // The two guest reads this bind needs before anything can decide
            // where its texels come from. `sampled_phase::Part::Lookup` is
            // this pair and nothing else, so the object-list walk is priced
            // against the resolve below rather than summed into it.
            let (texture_resource, view_swizzle) = {
                let _s = crate::runtime::sampled_phase::Span::open(
                    crate::runtime::sampled_phase::Part::Lookup,
                );
                let texture_resource = retained.cloned().or_else(|| {
                    objects::resolve_resource(state, host, req.task_id, texture_ref).ok()
                });
                // A type-8 view's channel remap. Resolved here rather than in
                // the loaders because it describes how the bind READS the
                // texture, not what the texture contains: the engine hands it
                // to the image view as a component mapping and the hardware
                // applies it at sample time, so the texels stay untouched and
                // the bind keeps whatever content rail it was already on.
                let view_swizzle = texture_resource
                    .as_ref()
                    .filter(|resource| resource.entry().kind == ObjectKind::TextureView)
                    .and_then(|_| resolve_texture_view(state, host, req.task_id, texture_ref))
                    .and_then(|view| view.swizzle)
                    .filter(|plan| !pixel_format::swizzle_is_identity(plan));
                (texture_resource, view_swizzle)
            };
            // Where the texels come from, which is the part with the cache
            // behind it. Scoped to a block so it closes at the resolve and
            // not at the end of the bind: everything after it — the
            // reflection read, the shape fold, the pushes — is deliberately
            // unbracketed, and a span held to the closure's end would
            // swallow it and make the four parts look like they summed.
            // A bind that declines from inside here charges its remainder
            // to `Resolve`, because the span commits on `Drop`.
            let (tw, th, loaded) = {
                // The probe is charged to the alias part it belongs to, and
                // the span is handed off to `ResolveSource` on the branch
                // where the probe found nothing — so the two parts partition
                // this scope rather than overlapping it.
                let alias_span = crate::runtime::sampled_phase::Span::open(
                    crate::runtime::sampled_phase::Part::ResolveAlias,
                );
                let attachment_alias = frag_stage
                    .then(|| fragment_attachment_alias_initial(req, index, texture_ref))
                    .flatten();
                if let Some((aw, ah, alias)) = attachment_alias {
                    let color = req
                        .colors
                        .iter()
                        .find(|color| color.slot == index && color.texture_ref == texture_ref)
                        .expect("attachment alias resolver returned its matching colour");
                    let format = pixel_format::color_attachment_format_checked(color.format)
                        .map_err(|reason| DrawPreparationDecline::ColorAttachmentFormat {
                            reason,
                        })?;
                    let identity = if req
                        .colors
                        .first()
                        .is_some_and(|primary| primary.slot == color.slot)
                    {
                        render_chain_identity(state, req, gva_alloc_generation)
                    } else {
                        color_target_identity(
                            state,
                            host,
                            req.task_id,
                            color,
                            format.layout(),
                            None,
                        )
                    }
                    .ok_or(
                        DrawPreparationDecline::AttachmentAliasIdentityMissing {
                            index,
                            texture_ref,
                        },
                    )?;
                    (aw, ah, attachment_alias_source(identity, format, alias))
                } else {
                    drop(alias_span);
                    let _s = crate::runtime::sampled_phase::Span::open(
                        crate::runtime::sampled_phase::Part::ResolveSource,
                    );
                    let Some(loaded) = resolve_sampled_source(
                        state,
                        host,
                        req.task_id,
                        texture_ref,
                        texture_resource.clone(),
                        view_swizzle.is_none(),
                        shape,
                    ) else {
                        let detail = texture_resource
                            .as_deref()
                            .and_then(retained_linear_sample_miss_detail)
                            .unwrap_or_else(|| {
                                sample_miss_detail(state, host, req.task_id, texture_ref)
                            });
                        return Err(DrawPreparationDecline::TextureResolveMissing {
                            stage: if frag_stage { "fragment" } else { "vertex" },
                            index,
                            texture_ref,
                            detail,
                        });
                    };
                    let (rw, rh, _mid, src) = loaded;
                    (rw, rh, src)
                }
            };
            let mut bytes_identity = None;
            let mut byte_origin = reims_vgpu_core::SampledByteOrigin::Synthetic;
            // Byte layout of a CPU-origin bind. Default RGBA8; a source that
            // already holds its bytes in an uploadable order keeps them —
            // BGRA8 from the surface backing scanout cache, a native single/dual-channel
            // video plane — and the host spelling is applied once, where the
            // engine resource is built (`vk_texel_layout` below).
            let sampled_format;
            // How the bound texels' channels sit on the host format, from
            // the rail that produced them. Identity for every CPU-origin
            // bind, because those loaders have already put the channels
            // where Metal presents them; non-identity only where a rail
            // handed the guest's own bytes over untouched.
            let mut sampled_components = pixel_format::swizzle_identity();
            let mut source_planes = 1;
            let source_is_target = matches!(
                &loaded,
                SampledSourceRequest::Target(_, _) | SampledSourceRequest::Attachment(_, _, _)
            );
            let source = match loaded {
                SampledSourceRequest::Bytes(rgba, identity, byte_format, origin) => {
                    bytes_identity = identity;
                    let (format, downgrade) = byte_format.image_format();
                    if let Some(source) = downgrade {
                        srgb_census::note_downgrade(srgb_census::site::SAMPLED_BYTE_UPLOAD, source);
                    }
                    sampled_format = format;
                    byte_origin = origin;
                    reims_vgpu_core::SampledSource::Bytes(rgba)
                }
                SampledSourceRequest::Target(identity, format) => {
                    sampled_format = format;
                    // The source resolver carries the sampled texture's
                    // exact view format beside the allocation identity. A
                    // resident attachment view is not necessarily the view
                    // this bind names; collapsing the two loses both sRGB
                    // interpretation and physical channel order.
                    //
                    // The resource's `swizzle` below remains independent and
                    // is composed once with the format's component plan.
                    //
                    // This arm used to `return Ok(())` — no resource pushed
                    // at all. That was not a decline: the unbound scan had
                    // already counted `texture_ref != 0` as provided, so no
                    // null image descriptor was inserted either, and the binding
                    // went missing from a layout the fragment module
                    // statically uses. The engine's
                    // `used_binding_absent_from_layout` then refused the
                    // whole draw, which cost the guest every pixel of it.
                    reims_vgpu_core::SampledSource::Target(identity)
                }
                SampledSourceRequest::Attachment(identity, initial, format) => {
                    sampled_format = format;
                    reims_vgpu_core::SampledSource::Attachment { identity, initial }
                }
                SampledSourceRequest::GuestRuns(
                    src,
                    _native,
                    format,
                    planes,
                    identity,
                    vouch,
                    components,
                ) => {
                    sampled_format = format;
                    sampled_components = components;
                    source_planes = planes;
                    bytes_identity = identity;
                    reims_vgpu_core::SampledSource::GuestRuns(src, vouch)
                }
                SampledSourceRequest::GuestImage(source, format, identity, vouch, components) => {
                    sampled_format = format;
                    sampled_components = components;
                    source_planes = source
                        .viewed_base_layout()
                        .map(guest_image_plane_count)
                        .unwrap_or(0);
                    bytes_identity = identity;
                    reims_vgpu_core::SampledSource::GuestImage(source, vouch)
                }
            };
            let array_element = reflected_descriptor.array_element;
            let descriptor_count = reflected_descriptor.descriptor_count;
            let SampledImageShape {
                arrayed,
                volume,
                cube,
                one_dim,
                multisampled,
                mut layers,
            } = shape;
            if multisampled && !source_is_target {
                return Err(DrawPreparationDecline::TextureDimensionUnsupported {
                    stage: if frag_stage { "fragment" } else { "vertex" },
                    index,
                    texture_ref,
                    binding: img_bind,
                    kind: format!("{image_kind:?}"),
                });
            }
            if volume {
                layers = texture_resource
                    .as_ref()
                    .and_then(|resource| {
                        match crate::runtime::objects::decoded_resource(resource) {
                            Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => {
                                tex.levels.first().map(|level| level.planes())
                            }
                            _ => None,
                        }
                    })
                    .unwrap_or(source_planes);
            } else if arrayed {
                layers = texture_resource
                    .as_ref()
                    .and_then(|resource| {
                        match crate::runtime::objects::decoded_resource(resource) {
                            Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => tex
                                .declaration
                                .map(|declaration| u32::from(declaration.array_length))
                                .filter(|layers| *layers != 0),
                            _ => None,
                        }
                    })
                    .unwrap_or(source_planes);
            }
            // A one-dimensional image has one row. Width and height are
            // independent decoded descriptor fields; multiplying them because
            // a particular payload happened to fit would manufacture geometry
            // the API never declared.
            let Some((tw, th)) = sampled_extent(one_dim, tw, th) else {
                return Err(DrawPreparationDecline::TextureDimensionUnsupported {
                    stage: if frag_stage { "fragment" } else { "vertex" },
                    index,
                    texture_ref,
                    binding: img_bind,
                    kind: format!("one_dim_extent_{tw}x{th}"),
                });
            };
            images.push(reims_vgpu_core::SampledImageResource {
                binding: img_bind,
                array_element,
                descriptor_count,
                width: tw,
                height: th,
                layers,
                arrayed,
                volume,
                cube,
                one_dim,
                multisampled,
                source,
                content: texture_resource.as_ref().and_then(|resource| {
                    state
                        .task_objects
                        .resources
                        .content_stamp_for(resource.as_ref())
                }),
                byte_origin,
                format: sampled_format,
                identity: bytes_identity.map(|i| reims_vgpu_core::SampledContentIdentity {
                    key: i.key,
                    generation: i.generation,
                }),
                resource_lifetime: texture_resource
                    .as_ref()
                    .map(|resource| resource.lifetime_ref()),
                // The guest's view swizzle applied *after* the format's own
                // channel plan, folded into the one mapping the image view
                // can carry. Composed unconditionally rather than behind a
                // "does this need it" branch: identity is the unit on both
                // sides, so the fold is a no-op for every bind that does not
                // need it, and there is no case left to forget.
                swizzle: view_swizzle.unwrap_or_default().after(&sampled_components),
            });
            Ok(())
        };
        for t in req.vertex_textures.iter() {
            push_tex(t.index, t.texture_ref, t.resource.as_ref(), false)?;
        }
        for t in req.fragment_textures.iter() {
            push_tex(t.index, t.texture_ref, t.resource.as_ref(), true)?;
        }
    }
    // Provision the null bindings the guard found. A fragment texture the module
    // statically uses and this draw did not bind is absent from the
    // descriptor set layout *entirely* — `engine/exec.rs` builds the layout
    // from provided resources alone, so it is not an unwritten slot in a
    // layout that has the binding. Vulkan requires a descriptor for every
    // statically-used resource, and on Mesa's Intel driver the omission is
    // fatal rather than undefined: it sizes its binding array to
    // `max_binding + 1`, zero-fills every number nothing declared, and
    // scores each *used* binding as `(use_count << 7) / array_size`, so the
    // hole divides by zero and the host process dies of `SIGFPE` inside
    // pipeline creation with nothing returned for this device to decline on.
    //
    // Cold path: the vector is empty on every draw that binds what it
    // samples, so this costs one `is_empty` on the hot path.
    for &index in frag_unbound_used_textures {
        use reims_vgpu_core::ReflectedSampledKind;
        let img_bind = f_variant.texture_binding(index, None);
        if images.iter().any(|img| img.binding == img_bind) {
            continue;
        }
        // The shape has to be the one the module declared: a plain 2D view
        // bound where the shader samples an array is a different violation,
        // not a repair. The semantic interface and executable variant carry the
        // same translator-selected descriptor layout.
        let kind = match f_shader.interface.sampled_kind(
            f_variant.texture_declared_binding(
                index,
                f_shader
                    .interface
                    .texture_descriptor(index)
                    .map(|descriptor| descriptor.binding),
            ),
        ) {
            ReflectedSampledKind::Kind(k) => k,
            ReflectedSampledKind::Absent => {
                return Err(DrawPreparationDecline::TextureDimensionUnsupported {
                    stage: "fragment",
                    index,
                    texture_ref: 0,
                    binding: img_bind,
                    kind: "reflected_absent".into(),
                });
            }
            ReflectedSampledKind::Unsupported => {
                return Err(DrawPreparationDecline::TextureDimensionUnsupported {
                    stage: "fragment",
                    index,
                    texture_ref: 0,
                    binding: img_bind,
                    kind: "reflected_unsupported".into(),
                });
            }
        };
        let Some(shape) = sampled_image_shape(kind) else {
            // Cube and cube-array need six faces, and this engine declines
            // them where they are bound too. The hole stays and is named,
            // rather than papered over with a shape the shader did not
            // declare — which would be a second violation wearing the
            // repair's clothes.
            return Err(DrawPreparationDecline::TextureDimensionUnsupported {
                stage: "fragment",
                index,
                texture_ref: 0,
                binding: img_bind,
                kind: format!("{kind:?}"),
            });
        };
        if shape.multisampled {
            return Err(DrawPreparationDecline::TextureDimensionUnsupported {
                stage: "fragment",
                index,
                texture_ref: 0,
                binding: img_bind,
                kind: format!("{kind:?}"),
            });
        }
        // The serialized argument table permits a null texture reference. Keep
        // that semantic state through planning; the backend either binds a
        // native null descriptor or refuses by capability.
        crate::runtime::drain::note_store_route("frag_null_texture");
        images.push(reims_vgpu_core::SampledImageResource {
            binding: img_bind,
            array_element: 0,
            descriptor_count: 1,
            width: 0,
            height: 0,
            layers: shape.layers,
            arrayed: shape.arrayed,
            volume: shape.volume,
            cube: shape.cube,
            one_dim: shape.one_dim,
            multisampled: false,
            source: reims_vgpu_core::SampledSource::Null,
            content: None,
            byte_origin: reims_vgpu_core::SampledByteOrigin::Synthetic,
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Rgba8,
            ),
            identity: None,
            resource_lifetime: None,
            swizzle: Default::default(),
        });
    }

    Ok(images)
}

#[cfg(test)]
mod tests {
    use super::{guest_image_plane_count, reflected_sampled_binding, sampled_extent};
    use reims_vgpu_core::{
        ReflectedSampledKind, ReflectedShaderStage, ReflectedTextureAccess, SampledImageKind,
        ShaderDescriptorLocation, ShaderInterface, ShaderResourceAccess, ShaderResourceBinding,
        ShaderResourceKind, ShaderTextureComponent, ShaderTextureDimension, ShaderTextureShape,
    };

    fn interface(binding: Option<ShaderResourceBinding>) -> ShaderInterface {
        ShaderInterface {
            stage: ReflectedShaderStage::Fragment,
            bindings: binding.into_iter().collect(),
            local_size: None,
            unsupported: None,
        }
    }

    fn texture(shape: Option<ShaderTextureShape>) -> ShaderResourceBinding {
        ShaderResourceBinding {
            kind: ShaderResourceKind::Texture,
            metal_index: 7,
            descriptor: Some(ShaderDescriptorLocation {
                set: 0,
                binding: 39,
                count: 1,
            }),
            extent: None,
            footprint: None,
            texture_shape: shape,
            access: Some(ShaderResourceAccess::Sampled),
        }
    }

    #[test]
    fn an_encoder_binding_absent_from_the_shader_is_not_invented_as_2d() {
        assert_eq!(reflected_sampled_binding(&interface(None), 7), Ok(None));
    }

    #[test]
    fn a_reflected_binding_keeps_its_real_dimension() {
        let shape = ShaderTextureShape {
            dimension: ShaderTextureDimension::D3,
            arrayed: false,
            multisampled: false,
            component: ShaderTextureComponent::Float,
            writable: false,
            array_ref: false,
            array_length: None,
            storage_format: None,
        };
        assert_eq!(
            reflected_sampled_binding(&interface(Some(texture(Some(shape)))), 7),
            Ok(Some((
                reims_vgpu_core::ReflectedTextureDescriptor {
                    binding: 39,
                    array_element: 0,
                    descriptor_count: 1,
                    access: ReflectedTextureAccess::Sampled,
                },
                SampledImageKind::D3,
            )))
        );
    }

    #[test]
    fn a_declared_descriptor_without_shape_is_malformed_not_2d() {
        assert_eq!(
            reflected_sampled_binding(&interface(Some(texture(None))), 7),
            Err(ReflectedSampledKind::Absent)
        );
    }

    #[test]
    fn direct_images_keep_array_layers_and_volume_depth() {
        assert_eq!(
            guest_image_plane_count(reims_vgpu_memory::GuestImageLayout::D1Array {
                width: 8,
                layers: 5,
                array_pitch: 64,
            }),
            5
        );
        assert_eq!(
            guest_image_plane_count(reims_vgpu_memory::GuestImageLayout::D3 {
                width: 8,
                height: 4,
                depth: 7,
                depth_pitch: 128,
            }),
            7
        );
    }

    #[test]
    fn one_dimensional_extent_is_not_reconstructed_from_a_second_axis() {
        assert_eq!(sampled_extent(true, 17, 1), Some((17, 1)));
        assert_eq!(sampled_extent(true, 17, 3), None);
        assert_eq!(sampled_extent(true, 0, 1), None);
        assert_eq!(sampled_extent(false, 17, 3), Some((17, 3)));
    }
}
