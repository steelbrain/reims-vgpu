//! Complete sampler provisioning for one resolved draw.
//!
//! Stream binds and reflected sampler declarations share one binding namespace.
//! This planner owns that occupancy relation, applies per-bind LOD overrides,
//! records diagnostic provenance, and refuses collisions before execution.

use super::*;

pub(super) struct SamplerPlan {
    pub(super) resources: Vec<reims_vgpu_core::SamplerResource>,
    pub(super) provenance: std::collections::BTreeMap<u32, u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SamplerSlotOwner {
    Stream {
        stage: reims_vgpu_core::ShaderStage,
        metal_index: u32,
    },
    Reflected {
        stage: reims_vgpu_core::ShaderStage,
        metal_index: u32,
    },
}

pub(super) fn plan_samplers<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    req: &DrawEncodeRequest,
    vertex: &reims_vgpu_core::PreparedShaderVariant,
    fragment: &reims_vgpu_core::PreparedShaderVariant,
) -> Result<SamplerPlan, DrawPreparationDecline> {
    let mut resources = Vec::new();
    let mut occupied = std::collections::BTreeMap::new();
    // Guest, constexpr, and null descriptors are distinct contract states, so
    // provenance remains separate observation metadata.
    let mut provenance = std::collections::BTreeMap::new();

    {
        let mut push_stream = |index: u32,
                               sampler_ref: u32,
                               lod_clamp: Option<(u32, u32)>,
                               stage: reims_vgpu_core::ShaderStage|
         -> Result<(), DrawPreparationDecline> {
            let variant = match stage {
                reims_vgpu_core::ShaderStage::Vertex => vertex,
                reims_vgpu_core::ShaderStage::Fragment => fragment,
                reims_vgpu_core::ShaderStage::Unknown => unreachable!("draw stage is known"),
            };
            let binding = variant.sampler_binding(index);
            // A constexpr sampler is part of the executable shader, not an
            // argument at this Metal index. Encoder argument tables are sticky,
            // so a sampler left bound by an earlier draw can still be present
            // when the next shader owns this descriptor location immutably.
            // The executable descriptor is authoritative: do not turn
            // irrelevant encoder state into a collision or let it replace the
            // shader's sampler state.
            if variant
                .samplers
                .iter()
                .any(|reflected| reflected.binding == binding && reflected.static_state.is_some())
            {
                return Ok(());
            }
            if occupied
                .insert(
                    binding,
                    SamplerSlotOwner::Stream {
                        stage,
                        metal_index: index,
                    },
                )
                .is_some()
            {
                crate::runtime::drain::note_store_route("sampler_bind_collided");
                return Err(DrawPreparationDecline::SamplerBindingCollision {
                    stage,
                    index,
                    binding,
                    source: reims_vgpu_core::SamplerBindingSource::Stream,
                });
            }
            provenance.insert(binding, b'g');
            let mut sampler = load_vulkan_sampler(state, host, req.task_id, sampler_ref, binding)?;
            if let Some((min_bits, max_bits)) = lod_clamp {
                sampler.lod_min = min_bits;
                sampler.lod_max = max_bits;
            }
            resources.push(sampler);
            Ok(())
        };

        let _span = crate::runtime::sampled_phase::Span::open(
            crate::runtime::sampled_phase::Part::Samplers,
        );
        for sampler in req
            .vertex_samplers
            .iter()
            .filter(|sampler| sampler.sampler_ref != 0)
        {
            push_stream(
                sampler.index,
                sampler.sampler_ref,
                sampler.lod_clamp,
                reims_vgpu_core::ShaderStage::Vertex,
            )?;
        }
        for sampler in req
            .fragment_samplers
            .iter()
            .filter(|sampler| sampler.sampler_ref != 0)
        {
            push_stream(
                sampler.index,
                sampler.sampler_ref,
                sampler.lod_clamp,
                reims_vgpu_core::ShaderStage::Fragment,
            )?;
        }
    }

    let _span =
        crate::runtime::sampled_phase::Span::open(crate::runtime::sampled_phase::Part::Reflect);
    for (variant, stage) in [
        (vertex, reims_vgpu_core::ShaderStage::Vertex),
        (fragment, reims_vgpu_core::ShaderStage::Fragment),
    ] {
        for reflected in variant.samplers.iter() {
            let stream_supplies_dynamic_slot = reflected.static_state.is_none()
                && occupied.get(&reflected.binding)
                    == Some(&SamplerSlotOwner::Stream {
                        stage,
                        metal_index: reflected.metal_index,
                    });
            if stream_supplies_dynamic_slot {
                continue;
            }
            if occupied
                .insert(
                    reflected.binding,
                    SamplerSlotOwner::Reflected {
                        stage,
                        metal_index: reflected.metal_index,
                    },
                )
                .is_some()
            {
                crate::runtime::drain::note_store_route("sampler_bind_collided");
                return Err(DrawPreparationDecline::SamplerBindingCollision {
                    stage,
                    index: reflected.metal_index,
                    binding: reflected.binding,
                    source: reims_vgpu_core::SamplerBindingSource::Reflected,
                });
            }
            let binding = reflected.binding;
            if let Some(static_state) = reflected.static_state {
                provenance.insert(binding, b'c');
                resources.push(reflected_static_sampler_resource(
                    stage.name(),
                    binding,
                    static_state,
                )?);
            } else {
                provenance.insert(binding, b'n');
                resources.push(reims_vgpu_core::SamplerResource::null(binding));
            }
        }
    }

    Ok(SamplerPlan {
        resources,
        provenance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn variant(index: u32, binding: u32) -> reims_vgpu_core::PreparedShaderVariant {
        reims_vgpu_core::PreparedShaderVariant {
            program: Default::default(),
            samplers: Arc::from([reims_vgpu_core::ReflectedSamplerDescriptor {
                metal_index: index,
                binding,
                static_state: None,
            }]),
            declared_bindings: Arc::from([]),
            descriptor_uses: Arc::from([]),
            texture_uses: Arc::from([]),
            buffer_binding_base: 0,
            texture_binding_base: 32,
            sampler_binding_base: 64,
            word_count: 0,
        }
    }

    #[test]
    fn reflected_sampler_collision_refuses_the_complete_plan() {
        let mut state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        let mut host = crate::runtime::host::FakeHost::new();
        let request = DrawEncodeRequest::default();

        let result = plan_samplers(
            &mut state,
            &mut host,
            &request,
            &variant(1, 64),
            &variant(2, 64),
        );
        assert!(matches!(
            result,
            Err(DrawPreparationDecline::SamplerBindingCollision {
                stage: reims_vgpu_core::ShaderStage::Fragment,
                index: 2,
                binding: 64,
                source: reims_vgpu_core::SamplerBindingSource::Reflected,
            })
        ));
    }

    #[test]
    fn an_unbound_dynamic_sampler_remains_null() {
        let mut state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        let mut host = crate::runtime::host::FakeHost::new();
        let empty = reims_vgpu_core::PreparedShaderVariant {
            samplers: Arc::default(),
            ..variant(0, 64)
        };

        let plan = plan_samplers(
            &mut state,
            &mut host,
            &DrawEncodeRequest::default(),
            &empty,
            &variant(3, 67),
        )
        .expect("a null serialized sampler is representable semantically");

        assert_eq!(plan.resources.len(), 1);
        assert_eq!(plan.resources[0].binding, 67);
        assert_eq!(
            plan.resources[0].source,
            reims_vgpu_core::SamplerSource::Null
        );
        assert_eq!(plan.provenance.get(&67), Some(&b'n'));
    }

    #[test]
    fn stream_sampler_supplies_its_reflected_dynamic_slot() {
        let mut state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        state.task_objects.samplers.register(
            1,
            reims_vgpu_protocol::SerializerRef::new(7),
            Arc::new(reims_vgpu_protocol::SamplerDescriptor::default()),
        );
        let mut host = crate::runtime::host::FakeHost::new();
        let request = DrawEncodeRequest {
            task_id: 1,
            fragment_samplers: Arc::new(vec![SamplerBind {
                index: 0,
                sampler_ref: 7,
                lod_clamp: None,
            }]),
            ..Default::default()
        };
        let empty = reims_vgpu_core::PreparedShaderVariant {
            samplers: Arc::default(),
            ..variant(0, 64)
        };

        let plan = plan_samplers(&mut state, &mut host, &request, &empty, &variant(0, 64))
            .expect("the encoder bind supplies the reflected dynamic sampler");

        assert_eq!(plan.resources.len(), 1);
        assert_eq!(plan.resources[0].binding, 64);
        assert_eq!(plan.provenance.get(&64), Some(&b'g'));
    }

    #[test]
    fn another_stage_stream_cannot_supply_a_reflected_sampler_slot() {
        let mut state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        state.task_objects.samplers.register(
            1,
            reims_vgpu_protocol::SerializerRef::new(7),
            Arc::new(reims_vgpu_protocol::SamplerDescriptor::default()),
        );
        let mut host = crate::runtime::host::FakeHost::new();
        let request = DrawEncodeRequest {
            task_id: 1,
            vertex_samplers: Arc::new(vec![SamplerBind {
                index: 0,
                sampler_ref: 7,
                lod_clamp: None,
            }]),
            ..Default::default()
        };
        let empty = reims_vgpu_core::PreparedShaderVariant {
            samplers: Arc::default(),
            ..variant(0, 64)
        };

        let result = plan_samplers(&mut state, &mut host, &request, &empty, &variant(0, 64));

        assert!(matches!(
            result,
            Err(DrawPreparationDecline::SamplerBindingCollision {
                stage: reims_vgpu_core::ShaderStage::Fragment,
                index: 0,
                binding: 64,
                source: reims_vgpu_core::SamplerBindingSource::Reflected,
            })
        ));
    }

    #[test]
    fn reflected_static_sampler_owns_its_slot_over_stale_stream_state() {
        let mut state = Device::new(
            crate::model::DeviceId(1),
            reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E,
        );
        state.task_objects.samplers.register(
            1,
            reims_vgpu_protocol::SerializerRef::new(7),
            Arc::new(reims_vgpu_protocol::SamplerDescriptor::default()),
        );
        let mut host = crate::runtime::host::FakeHost::new();
        let request = DrawEncodeRequest {
            task_id: 1,
            fragment_samplers: Arc::new(vec![SamplerBind {
                index: 0,
                sampler_ref: 7,
                lod_clamp: None,
            }]),
            ..Default::default()
        };
        let empty = reims_vgpu_core::PreparedShaderVariant {
            samplers: Arc::default(),
            ..variant(0, 64)
        };
        let mut static_variant = variant(0, 64);
        static_variant.samplers = Arc::from([reims_vgpu_core::ReflectedSamplerDescriptor {
            metal_index: 0,
            binding: 64,
            static_state: Some(reims_vgpu_core::ReflectedStaticSamplerState {
                min_filter: reims_vgpu_core::ReflectedSamplerFilter::Linear,
                mag_filter: reims_vgpu_core::ReflectedSamplerFilter::Linear,
                mip_filter: reims_vgpu_core::ReflectedSamplerMipFilter::None,
                address_mode_s: reims_vgpu_core::ReflectedSamplerAddressMode::ClampToZero,
                address_mode_t: reims_vgpu_core::ReflectedSamplerAddressMode::ClampToZero,
                address_mode_r: reims_vgpu_core::ReflectedSamplerAddressMode::ClampToZero,
                coordinates: reims_vgpu_core::ReflectedSamplerCoordinates::Pixel,
                compare_function: reims_vgpu_core::ReflectedSamplerCompareFunction::Never,
                max_anisotropy: 1,
                lod_min_clamp: 0.0,
                lod_max_clamp: 65504.0,
                border_color: reims_vgpu_core::ReflectedSamplerBorderColor::TransparentBlack,
                reduction: reims_vgpu_core::ReflectedSamplerReduction::WeightedAverage,
                lod_bias: 0.0,
                raw_words: [0, 0],
            }),
        }]);

        let plan = plan_samplers(&mut state, &mut host, &request, &empty, &static_variant)
            .expect("the executable constexpr sampler owns the descriptor slot");

        assert_eq!(plan.resources.len(), 1);
        assert_eq!(plan.resources[0].binding, 64);
        assert_eq!(plan.provenance.get(&64), Some(&b'c'));
    }
}
