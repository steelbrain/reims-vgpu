//! Backend-independent facts and refusals produced while resolving commands.

use reims_vgpu_observe::Decline;
use reims_vgpu_protocol::RenderPipelineDescriptor;
use std::sync::Arc;

/// Backend-independent image shape required by one reflected sampled texture.
///
/// Array layers are one because the current execution contract materializes a
/// single layer. Cube images are deliberately not representable by
/// [`sampled_image_shape`] until the executor request can carry their faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampledImageShape {
    pub arrayed: bool,
    pub volume: bool,
    pub cube: bool,
    pub one_dim: bool,
    pub multisampled: bool,
    pub layers: u32,
}

/// Project reflected texture dimensionality into the semantic shape consumed
/// by draw execution. `None` is a typed inability to express a cube-array
/// image, not a backend-capability result.
pub fn sampled_image_shape(kind: crate::SampledImageKind) -> Option<SampledImageShape> {
    use crate::SampledImageKind;
    let d2 = SampledImageShape {
        arrayed: false,
        volume: false,
        cube: false,
        one_dim: false,
        multisampled: false,
        layers: 1,
    };
    Some(match kind {
        SampledImageKind::D1 => SampledImageShape {
            one_dim: true,
            ..d2
        },
        SampledImageKind::D1Array => SampledImageShape {
            one_dim: true,
            arrayed: true,
            ..d2
        },
        SampledImageKind::D2 => d2,
        SampledImageKind::D2Multisample => SampledImageShape {
            multisampled: true,
            ..d2
        },
        SampledImageKind::D2Array => SampledImageShape {
            arrayed: true,
            ..d2
        },
        SampledImageKind::D3 => SampledImageShape { volume: true, ..d2 },
        // A cube is six faces, always, by the definition of the type the shader
        // declared -- not a count read back from a resource. The six are also
        // what the guest's own dimension record means when it marks its slices
        // as cube slices, which is why `LinearTextureDescriptor` expands
        // `slice_count` by exactly this factor.
        SampledImageKind::Cube => SampledImageShape {
            cube: true,
            layers: reims_vgpu_protocol::CUBE_FACES,
            ..d2
        },
        // Still refused, and by name rather than by approximation. The executor
        // treats `arrayed`, `volume` and `cube` as mutually exclusive, and its
        // view-type selection tests `cube` before `arrayed`, so a cube-array
        // admitted here would be bound as a plain cube -- the first six faces
        // of a longer array, silently. Widening this needs a `CUBE_ARRAY` view
        // type in the backend first.
        SampledImageKind::CubeArray => return None,
    })
}

/// Buffer-index facts derived once from an immutable render-pipeline descriptor.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexBindPlan {
    constant_step: Box<[u32]>,
    attribute: Box<[u32]>,
}

impl VertexBindPlan {
    pub fn build(desc: &RenderPipelineDescriptor) -> Self {
        let mut constant_step: Vec<u32> = desc
            .vertex_attributes
            .iter()
            .filter(|attribute| {
                attribute.format != 0
                    && attribute.declared_step_function
                        == Some(reims_vgpu_protocol::vertex_step::MTL_VERTEX_STEP_FUNCTION_CONSTANT)
            })
            .map(|attribute| attribute.buffer_index)
            .collect();
        constant_step.sort_unstable();
        constant_step.dedup();

        let mut attribute: Vec<u32> = desc
            .vertex_attributes
            .iter()
            .map(|attribute| attribute.buffer_index)
            .collect();
        attribute.sort_unstable();
        attribute.dedup();

        Self {
            constant_step: constant_step.into_boxed_slice(),
            attribute: attribute.into_boxed_slice(),
        }
    }

    /// Whether this index feeds a Constant-step attribute and must remain on
    /// the CPU staging read which prepends the base-instance prefix.
    pub fn is_constant_step(&self, buffer_index: u32) -> bool {
        self.constant_step.binary_search(&buffer_index).is_ok()
    }

    /// Whether the pipeline's attribute list names this buffer index at all.
    pub fn feeds_stage_in(&self, buffer_index: u32) -> bool {
        self.attribute.binary_search(&buffer_index).is_ok()
    }
}

/// Backend-neutral immutable state retained by one render-pipeline object.
#[derive(Clone)]
pub struct ResolvedRenderPipeline {
    /// Present only when the guest pipeline object owns this retained state.
    pub pipeline_lifetime: Option<crate::ResourceLifetime>,
    pub desc: Arc<RenderPipelineDescriptor>,
    pub vertex: crate::PreparedShaderFamily,
    pub fragment: crate::PreparedShaderFamily,
    /// Derived once from `desc` and retained with the pipeline lifetime.
    pub bind_plan: Arc<VertexBindPlan>,
}

pub const MAX_BUFFER_BIND_SLOTS: u32 = 31;
pub const MAX_TEXTURE_BIND_SLOTS: u32 = 128;
pub const MAX_SAMPLER_BIND_SLOTS: u32 = 32;
pub const MAX_ANY_BIND_SLOTS: u32 = {
    let widest = if MAX_TEXTURE_BIND_SLOTS > MAX_SAMPLER_BIND_SLOTS {
        MAX_TEXTURE_BIND_SLOTS
    } else {
        MAX_SAMPLER_BIND_SLOTS
    };
    if widest > MAX_BUFFER_BIND_SLOTS {
        widest
    } else {
        MAX_BUFFER_BIND_SLOTS
    }
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Unknown,
    Vertex,
    Fragment,
}

impl ShaderStage {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindTableClass {
    Buffer,
    Texture,
    Sampler,
}

impl BindTableClass {
    pub const fn table(self) -> u32 {
        match self {
            Self::Buffer => MAX_BUFFER_BIND_SLOTS,
            Self::Texture => MAX_TEXTURE_BIND_SLOTS,
            Self::Sampler => MAX_SAMPLER_BIND_SLOTS,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::Sampler => "sampler",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PastTableBind {
    pub class: BindTableClass,
    pub stage: ShaderStage,
    pub index: u32,
    pub resource_ref: u32,
}

impl PastTableBind {
    pub const fn stage_name(&self) -> &'static str {
        self.stage.name()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexLoadReason {
    TypeUnsupported,
    CountOverflow,
    CountZero,
    EntryMissing,
    ObjectType,
    DescRead,
    DescDecode,
    BackingMissing,
    OffsetOverflow,
    OutOfBounds,
    ReadFail,
    BaseVertexOutOfRange,
}

impl Decline for IndexLoadReason {
    fn slug(&self) -> &'static str {
        match self {
            Self::TypeUnsupported => "draw_index_type_unsupported",
            Self::CountOverflow => "draw_index_count_overflow",
            Self::CountZero => "draw_index_count_zero",
            Self::EntryMissing => "draw_index_no_list_entry",
            Self::ObjectType => "draw_index_wrong_type",
            Self::DescRead => "draw_index_desc_read",
            Self::DescDecode => "draw_index_desc_decode",
            Self::BackingMissing => "draw_index_backing_missing",
            Self::OffsetOverflow => "draw_index_offset_overflow",
            Self::OutOfBounds => "draw_index_out_of_bounds",
            Self::ReadFail => "draw_index_read_fail",
            Self::BaseVertexOutOfRange => "draw_index_base_vertex_out_of_range",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MrtDrop {
    NonContiguousSlot,
    UnknownFormat,
    NoIdentity,
    AliasesPrimary,
}

impl Decline for MrtDrop {
    fn slug(&self) -> &'static str {
        match self {
            Self::NonContiguousSlot => "mrt_drop_non_contiguous_slot",
            Self::UnknownFormat => "mrt_drop_unknown_format",
            Self::NoIdentity => "mrt_drop_no_identity",
            Self::AliasesPrimary => "mrt_drop_aliases_primary",
        }
    }
}

impl MrtDrop {
    pub const fn code(self) -> u8 {
        match self {
            Self::NonContiguousSlot => 1,
            Self::UnknownFormat => 3,
            Self::NoIdentity => 4,
            Self::AliasesPrimary => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecondaryMrtRefusal {
    pub slot: u32,
    pub reason: MrtDrop,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MtlbDecline {
    WrappedAirMissing {
        data_len: usize,
    },
    WrapperHeaderTruncated {
        offset: usize,
        data_len: usize,
    },
    BlobOutOfBounds {
        offset: usize,
        blob_len: u64,
        data_len: usize,
    },
}

impl Decline for MtlbDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::WrappedAirMissing { .. } => "mtlb_wrapped_air_missing",
            Self::WrapperHeaderTruncated { .. } => "mtlb_wrapper_header_truncated",
            Self::BlobOutOfBounds { .. } => "mtlb_blob_out_of_bounds",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::WrappedAirMissing { data_len } => vec![("data_len", data_len.to_string())],
            Self::WrapperHeaderTruncated { offset, data_len } => vec![
                ("offset", offset.to_string()),
                ("data_len", data_len.to_string()),
            ],
            Self::BlobOutOfBounds {
                offset,
                blob_len,
                data_len,
            } => vec![
                ("offset", offset.to_string()),
                ("blob_len", blob_len.to_string()),
                ("data_len", data_len.to_string()),
            ],
        }
    }
}

reims_vgpu_observe::decline_display!(MtlbDecline);
impl std::error::Error for MtlbDecline {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_bind_class_owns_a_distinct_request_table_bound() {
        assert_eq!(BindTableClass::Buffer.table(), MAX_BUFFER_BIND_SLOTS);
        assert_eq!(BindTableClass::Texture.table(), MAX_TEXTURE_BIND_SLOTS);
        assert_eq!(BindTableClass::Sampler.table(), MAX_SAMPLER_BIND_SLOTS);
        assert_eq!(MAX_ANY_BIND_SLOTS, MAX_TEXTURE_BIND_SLOTS);
    }

    #[test]
    fn sampled_shape_preserves_every_expressible_dimension_flag() {
        use crate::SampledImageKind;

        for (kind, arrayed, volume, one_dim, multisampled) in [
            (SampledImageKind::D2, false, false, false, false),
            (SampledImageKind::D2Array, true, false, false, false),
            (SampledImageKind::D2Multisample, false, false, false, true),
            (SampledImageKind::D3, false, true, false, false),
            (SampledImageKind::D1, false, false, true, false),
            (SampledImageKind::D1Array, true, false, true, false),
        ] {
            let shape = sampled_image_shape(kind).expect("expressible shape");
            assert_eq!(
                (
                    shape.arrayed,
                    shape.volume,
                    shape.cube,
                    shape.one_dim,
                    shape.multisampled,
                    shape.layers,
                ),
                (arrayed, volume, false, one_dim, multisampled, 1),
                "{kind:?} did not retain its complete semantic shape"
            );
        }
    }

    /// A cube's six faces come from the declared type, so the shape states them
    /// without consulting any resource. The executor validates
    /// `layers == 6 && width == height` against this, and would decline a cube
    /// carrying any other layer count — so a shape that left `layers` at 1
    /// would trade one refusal for another rather than binding anything.
    #[test]
    fn sampled_shape_gives_a_cube_its_six_faces() {
        use crate::SampledImageKind;

        let shape = sampled_image_shape(SampledImageKind::Cube).expect("cube is expressible");
        assert_eq!(
            (
                shape.cube,
                shape.layers,
                shape.arrayed,
                shape.volume,
                shape.one_dim,
                shape.multisampled,
            ),
            (true, 6, false, false, false, false),
        );
    }

    /// The one shape still refused, and deliberately: the executor treats
    /// `arrayed`/`volume`/`cube` as mutually exclusive and picks its view type
    /// by testing `cube` first, so admitting a cube-array would bind the first
    /// six faces of a longer array as though they were the whole texture.
    /// A typed refusal costs the guest that draw and says so; the alternative
    /// costs it silently.
    #[test]
    fn sampled_shape_refuses_cube_arrays_by_name() {
        use crate::SampledImageKind;

        assert!(sampled_image_shape(SampledImageKind::CubeArray).is_none());
    }
}
