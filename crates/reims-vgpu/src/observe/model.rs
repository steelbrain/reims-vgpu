//! Observation adapters for semantic model events.
//!
//! The dependency points inward: this module knows the model vocabulary and
//! renders it, while the model returns typed events without knowing the sink.

use crate::model::{FailEvent, PresentBacking};

pub(crate) const UNKNOWN_OPCODE_ECHO_WORDS_MAX: usize = 4;

fn packet_echo_fields(
    channel: u32,
    opcode: u16,
    total_size: u32,
    stamp_count: u16,
    payload: &[u8],
) -> Vec<(&'static str, String)> {
    let mut fields = vec![
        ("ch", channel.to_string()),
        ("opcode", format!("{opcode:#x}")),
        ("total_size", total_size.to_string()),
        ("stamps", stamp_count.to_string()),
        ("plen", payload.len().to_string()),
    ];
    let words = payload
        .chunks_exact(4)
        .take(UNKNOWN_OPCODE_ECHO_WORDS_MAX)
        .map(|word| format!("{:#010x}", reims_vgpu_core::endian::ld32(word)))
        .collect::<Vec<_>>()
        .join(":");
    if !words.is_empty() {
        fields.push(("payload", words));
    }
    fields
}

impl crate::observe::Decline for FailEvent {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnknownRootOpcode { .. } => "unknown_root_opcode",
            Self::UnknownChildOpcode { .. } => "unknown_child_opcode",
            Self::UnimplementedChildCommand { command, .. } => command.slug(),
            Self::MalformedRootPacket { fault, .. } | Self::MalformedChildPacket { fault, .. } => {
                fault.slug()
            }
            Self::UnsupportedExec { fault, .. } => fault.slug(),
            Self::BadMmioAccess { .. } => "bad_mmio_access",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnknownRootOpcode { opcode, total_size } => vec![
                ("opcode", format!("{opcode:#x}")),
                ("total_size", total_size.to_string()),
            ],
            Self::UnknownChildOpcode {
                channel,
                opcode,
                total_size,
                stamp_count,
                payload,
            } => packet_echo_fields(*channel, *opcode, *total_size, *stamp_count, payload),
            Self::UnimplementedChildCommand {
                channel,
                command,
                opcode,
                total_size,
                stamp_count,
                payload,
            } => {
                let mut fields =
                    packet_echo_fields(*channel, *opcode, *total_size, *stamp_count, payload);
                fields.insert(0, ("cmd", command.command().to_string()));
                fields
            }
            Self::MalformedRootPacket { head, .. } => vec![("head", head.to_string())],
            Self::MalformedChildPacket { channel, head, .. } => {
                vec![("ch", channel.to_string()), ("head", head.to_string())]
            }
            Self::UnsupportedExec { channel, .. } => vec![("ch", channel.to_string())],
            Self::BadMmioAccess { offset, size } => vec![
                ("offset", format!("{offset:#x}")),
                ("size", size.to_string()),
            ],
        }
    }
}

impl crate::observe::Decline for PresentBacking {
    fn slug(&self) -> &'static str {
        match self {
            Self::Restaled { .. } => "present_backing_restaled",
            Self::NeverStored => "present_backing_never_stored",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Restaled { seq } => vec![("since_seq", seq.to_string())],
            Self::NeverStored => Vec::new(),
        }
    }
}

impl crate::observe::Decline for crate::model::StateMutationDecline {
    fn slug(&self) -> &'static str {
        use crate::model::StateMutationDecline as D;
        match self {
            D::SetObjectListTaskInactive { .. } => "model_set_object_list_task_inactive",
            #[cfg(test)]
            D::InsertObjectTaskInactive { .. } => "model_insert_object_task_inactive",
            D::MapSurfaceIdSentinel { .. } => "model_map_surface_id_sentinel",
            D::UnmapSurfaceIdSentinel { .. } => "model_unmap_surface_id_sentinel",
            D::AttachMappingIdSentinel { .. } => "model_attach_mapping_id_sentinel",
            D::AttachMappingInternalZero { .. } => "model_attach_mapping_internal_zero",
            D::MappingDeviceDescIdSentinel { .. } => "model_mapping_device_desc_id_sentinel",
            D::MappingDeviceDescEmpty { .. } => "model_mapping_device_desc_empty",
            D::MappingGeomIdSentinel { .. } => "model_mapping_geom_id_sentinel",
            D::MappingGeomWidthZero { .. } => "model_mapping_geom_width_zero",
            D::MappingGeomHeightZero { .. } => "model_mapping_geom_height_zero",
            D::MappingGeomWidthRange { .. } => "model_mapping_geom_width_range",
            D::MappingGeomHeightRange { .. } => "model_mapping_geom_height_range",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        use crate::model::StateMutationDecline as D;
        let mut fields = match self {
            D::SetObjectListTaskInactive { task_id } => vec![("task", task_id.to_string())],
            #[cfg(test)]
            D::InsertObjectTaskInactive {
                task_id,
                object_ref,
            } => vec![
                ("task", task_id.to_string()),
                ("ref", object_ref.to_string()),
            ],
            D::MapSurfaceIdSentinel { mapping_id }
            | D::UnmapSurfaceIdSentinel { mapping_id }
            | D::AttachMappingIdSentinel { mapping_id }
            | D::AttachMappingInternalZero { mapping_id }
            | D::MappingDeviceDescIdSentinel { mapping_id }
            | D::MappingDeviceDescEmpty { mapping_id }
            | D::MappingGeomIdSentinel { mapping_id }
            | D::MappingGeomWidthZero { mapping_id }
            | D::MappingGeomHeightZero { mapping_id }
            | D::MappingGeomWidthRange { mapping_id, .. }
            | D::MappingGeomHeightRange { mapping_id, .. } => {
                vec![("mapping", mapping_id.to_string())]
            }
        };
        match self {
            D::MappingGeomWidthRange { width, .. } => {
                fields.push(("width", width.to_string()));
            }
            D::MappingGeomHeightRange { height, .. } => {
                fields.push(("height", height.to_string()));
            }
            _ => {}
        }
        fields
    }
}
