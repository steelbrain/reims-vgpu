//! Backend-neutral interface of one prepared shader.
//!
//! Translation owns how this information is discovered. Preparation owns how
//! guest resources satisfy it. This projection prevents either side from
//! passing a translator-native reflection graph across the executor boundary.

use reims_vgpu_protocol::StorageImageFormat;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedShaderStage {
    Vertex,
    TessellationEvaluation,
    Fragment,
    Kernel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderDescriptorLocation {
    pub set: u32,
    pub binding: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderResourceKind {
    Buffer,
    ThreadgroupBuffer,
    KernelStageInput,
    Texture,
    TextureArray,
    StorageImage,
    Sampler,
    StaticSampler,
    ColorInput,
    AccelerationStructureShadow,
    PrimitiveAccelerationStructure,
    VisibleFunctionTable,
    IntersectionFunctionTable,
    EmbeddedArgBufferTexture,
    EmbeddedArgBufferBuffer,
    BufferAddressTable,
}

impl ShaderResourceKind {
    pub fn unsupported_vulkan_name(self) -> Option<&'static str> {
        match self {
            Self::KernelStageInput => Some("kernel_stage_input"),
            Self::AccelerationStructureShadow => Some("acceleration_structure_shadow"),
            Self::PrimitiveAccelerationStructure => Some("primitive_acceleration_structure"),
            Self::EmbeddedArgBufferTexture => Some("embedded_texture"),
            Self::EmbeddedArgBufferBuffer => Some("embedded_buffer"),
            Self::BufferAddressTable => Some("buffer_address_table"),
            Self::Buffer
            | Self::ThreadgroupBuffer
            | Self::Texture
            | Self::TextureArray
            | Self::StorageImage
            | Self::Sampler
            | Self::StaticSampler
            | Self::ColorInput
            | Self::VisibleFunctionTable
            | Self::IntersectionFunctionTable => None,
        }
    }

    pub fn is_texture(self) -> bool {
        matches!(
            self,
            Self::Texture
                | Self::TextureArray
                | Self::StorageImage
                | Self::EmbeddedArgBufferTexture
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderResourceAccess {
    Unused,
    ReadOnly,
    WriteOnly,
    ReadWrite,
    Sampled,
    Storage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderBufferExtent {
    Object { bytes: u32 },
    Unbounded,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ShaderBufferIndexSource {
    VertexIndex,
    InstanceIndex,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    LocalInvocationIndex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShaderBufferByteRange {
    pub offset: u64,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShaderBufferStrideTerm {
    pub source: ShaderBufferIndexSource,
    pub stride: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ShaderBufferStridedAccess {
    pub base_offset: u64,
    pub access_size: u64,
    pub terms: Vec<ShaderBufferStrideTerm>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShaderBufferFootprint {
    pub static_ranges: Vec<ShaderBufferByteRange>,
    pub strided_accesses: Vec<ShaderBufferStridedAccess>,
    pub has_unbounded_access: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderTextureDimension {
    D1,
    D2,
    D3,
    Cube,
    Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderTextureComponent {
    Float,
    Sint,
    Uint,
}

/// Image dimensionality a semantic sampled-image binding requires.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampledImageKind {
    D1,
    D1Array,
    D2,
    D2Multisample,
    D2Array,
    D3,
    Cube,
    CubeArray,
}

/// Exact sampled-versus-storage class of a reflected texture binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageAccess {
    Sampled,
    Storage,
}

/// Content access proven for one storage-image binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageImageAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
    Unknown,
    AmbiguousBinding,
}

/// One Metal texture-table slot's semantic descriptor location.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReflectedTextureDescriptor {
    pub binding: u32,
    pub array_element: u32,
    pub descriptor_count: u32,
    pub access: ReflectedTextureAccess,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedTextureAccess {
    Sampled,
    Storage,
    Unknown,
}

/// Total semantic answer for one reflected buffer argument.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedBufferAccess {
    Unused,
    ReadOnly,
    Writable,
    Absent,
    Unknown,
}

/// Sampled-image classification for one semantic descriptor binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSampledKind {
    Kind(SampledImageKind),
    Unsupported,
    Absent,
}

/// Texture shape and access supported by the current flat compute staging rail.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedComputeTexture {
    Absent,
    Plain2d(ImageAccess),
    UnstageableShape { axis: &'static str },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShaderTextureShape {
    pub dimension: ShaderTextureDimension,
    pub arrayed: bool,
    pub multisampled: bool,
    pub component: ShaderTextureComponent,
    pub writable: bool,
    pub array_ref: bool,
    pub array_length: Option<u32>,
    pub storage_format: Option<StorageImageFormat>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShaderResourceBinding {
    pub kind: ShaderResourceKind,
    pub metal_index: u32,
    pub descriptor: Option<ShaderDescriptorLocation>,
    pub extent: Option<ShaderBufferExtent>,
    pub footprint: Option<ShaderBufferFootprint>,
    pub texture_shape: Option<ShaderTextureShape>,
    pub access: Option<ShaderResourceAccess>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedShaderInterface {
    pub feature: &'static str,
    pub count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSamplerFilter {
    Nearest,
    Linear,
    Bicubic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSamplerMipFilter {
    None,
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSamplerAddressMode {
    ClampToZero,
    ClampToEdge,
    Repeat,
    MirroredRepeat,
    ClampToBorder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSamplerCoordinates {
    Normalized,
    Pixel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSamplerCompareFunction {
    None,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Always,
    Never,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSamplerBorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReflectedSamplerReduction {
    WeightedAverage,
    Minimum,
    Maximum,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectedStaticSamplerState {
    pub min_filter: ReflectedSamplerFilter,
    pub mag_filter: ReflectedSamplerFilter,
    pub mip_filter: ReflectedSamplerMipFilter,
    pub address_mode_s: ReflectedSamplerAddressMode,
    pub address_mode_t: ReflectedSamplerAddressMode,
    pub address_mode_r: ReflectedSamplerAddressMode,
    pub coordinates: ReflectedSamplerCoordinates,
    pub compare_function: ReflectedSamplerCompareFunction,
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub border_color: ReflectedSamplerBorderColor,
    pub reduction: ReflectedSamplerReduction,
    pub lod_bias: f32,
    pub raw_words: [u64; 2],
}

/// One sampler in an executable semantic descriptor interface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReflectedSamplerDescriptor {
    pub metal_index: u32,
    pub binding: u32,
    pub static_state: Option<ReflectedStaticSamplerState>,
}

/// Static use of one executable descriptor binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DescriptorUse {
    NotDeclared,
    DeclaredUnused,
    Used,
    Ambiguous,
}

impl DescriptorUse {
    pub fn slug(self) -> &'static str {
        match self {
            Self::NotDeclared => "frag_unbound_not_declared",
            Self::DeclaredUnused => "frag_unbound_declared_unused",
            Self::Used => "frag_declared_descriptor_unbound",
            Self::Ambiguous => "frag_unbound_ambiguous_binding",
        }
    }

    pub fn is_violation(self) -> bool {
        matches!(self, Self::Used)
    }
}

/// Backend-neutral facts attached to one executable shader numbering.
#[derive(Clone, Debug)]
pub struct PreparedShaderVariant {
    pub program: crate::PreparedShaderStage,
    pub samplers: Arc<[ReflectedSamplerDescriptor]>,
    pub declared_bindings: Arc<[u32]>,
    pub descriptor_uses: Arc<[(u32, DescriptorUse)]>,
    /// Texture descriptor use keyed by the guest's Metal argument index.
    ///
    /// The backend projects this after applying its executable binding
    /// numbering, so command preparation never needs that private numbering.
    pub texture_uses: Arc<[(u32, DescriptorUse)]>,
    /// Effective descriptor-band bases reported by the translator for this
    /// exact module. Runtime supplies Metal indices and never reconstructs the
    /// selected descriptor layout.
    pub buffer_binding_base: u32,
    pub texture_binding_base: u32,
    pub sampler_binding_base: u32,
    pub word_count: u32,
}

impl PreparedShaderVariant {
    pub fn declares_descriptor(&self, binding: u32) -> bool {
        self.descriptor_uses
            .binary_search_by_key(&binding, |(binding, _)| *binding)
            .is_ok()
    }

    pub fn descriptor_use(&self, binding: u32) -> DescriptorUse {
        self.descriptor_uses
            .binary_search_by_key(&binding, |(binding, _)| *binding)
            .ok()
            .map(|index| self.descriptor_uses[index].1)
            .unwrap_or(DescriptorUse::NotDeclared)
    }

    pub fn buffer_binding(&self, metal_index: u32) -> u32 {
        self.buffer_binding_base + metal_index
    }

    pub fn texture_binding(&self, metal_index: u32, declared_binding: Option<u32>) -> u32 {
        self.texture_declared_binding(metal_index, declared_binding)
    }

    pub fn texture_declared_binding(&self, metal_index: u32, declared_binding: Option<u32>) -> u32 {
        declared_binding.unwrap_or(self.texture_binding_base + metal_index)
    }

    pub fn sampler_binding(&self, metal_index: u32) -> u32 {
        self.sampler_binding_base + metal_index
    }

    pub fn texture_use(&self, metal_index: u32) -> DescriptorUse {
        self.texture_uses
            .binary_search_by_key(&metal_index, |(index, _)| *index)
            .ok()
            .map(|index| self.texture_uses[index].1)
            .unwrap_or(DescriptorUse::NotDeclared)
    }
}

/// One translated render stage and the effective descriptor layout baked into
/// its executable module.
#[derive(Clone, Debug)]
pub struct PreparedShaderFamily {
    pub interface: Arc<ShaderInterface>,
    variant: PreparedShaderVariant,
}

impl PreparedShaderFamily {
    pub fn new(interface: Arc<ShaderInterface>, variant: PreparedShaderVariant) -> Self {
        Self { interface, variant }
    }

    pub fn variant(&self) -> &PreparedShaderVariant {
        &self.variant
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShaderInterface {
    pub stage: ReflectedShaderStage,
    pub bindings: Vec<ShaderResourceBinding>,
    pub local_size: Option<[u32; 3]>,
    pub unsupported: Option<UnsupportedShaderInterface>,
}

impl ShaderInterface {
    pub fn first_unsupported_resource(&self) -> Option<&ShaderResourceBinding> {
        self.bindings
            .iter()
            .find(|resource| resource.kind.unsupported_vulkan_name().is_some())
    }

    pub fn first_unsupported_interface(
        &self,
        expected: ReflectedShaderStage,
    ) -> Option<UnsupportedShaderInterface> {
        if self.stage != expected {
            return Some(UnsupportedShaderInterface {
                feature: match self.stage {
                    ReflectedShaderStage::Vertex => "shader_stage_vertex",
                    ReflectedShaderStage::TessellationEvaluation => {
                        "shader_stage_tessellation_evaluation"
                    }
                    ReflectedShaderStage::Fragment => "shader_stage_fragment",
                    ReflectedShaderStage::Kernel => "shader_stage_kernel",
                },
                count: 1,
            });
        }
        self.unsupported
    }

    fn texture_shape_for_binding(&self, binding: u32) -> Option<&ShaderTextureShape> {
        self.bindings.iter().find_map(|resource| {
            (resource.kind.is_texture()
                && resource.descriptor.map(|descriptor| descriptor.binding) == Some(binding))
            .then_some(resource.texture_shape.as_ref())
            .flatten()
        })
    }

    /// Resolve one Metal texture-table index through the semantic descriptor
    /// interface. Exact scalar declarations win over an enclosing array.
    pub fn texture_descriptor(&self, metal_index: u32) -> Option<ReflectedTextureDescriptor> {
        let exact = self
            .bindings
            .iter()
            .find(|resource| resource.kind.is_texture() && resource.metal_index == metal_index);
        let reflected = exact.or_else(|| {
            self.bindings.iter().find(|resource| {
                if resource.kind != ShaderResourceKind::TextureArray {
                    return false;
                }
                let Some(descriptor) = resource.descriptor else {
                    return false;
                };
                metal_index
                    .checked_sub(resource.metal_index)
                    .is_some_and(|element| element < descriptor.count)
            })
        })?;
        let descriptor = reflected.descriptor?;
        let array_element = if reflected.kind == ShaderResourceKind::TextureArray {
            metal_index.checked_sub(reflected.metal_index)?
        } else {
            0
        };
        (descriptor.count > 0 && array_element < descriptor.count).then_some(
            ReflectedTextureDescriptor {
                binding: descriptor.binding,
                array_element,
                descriptor_count: descriptor.count,
                access: match reflected.access {
                    Some(ShaderResourceAccess::Sampled) => ReflectedTextureAccess::Sampled,
                    Some(ShaderResourceAccess::Storage) => ReflectedTextureAccess::Storage,
                    _ => ReflectedTextureAccess::Unknown,
                },
            },
        )
    }

    pub fn first_non_sampled_texture_descriptor(
        &self,
    ) -> Option<(u32, ReflectedTextureDescriptor)> {
        self.bindings.iter().find_map(|resource| {
            if !resource.kind.is_texture() || resource.access == Some(ShaderResourceAccess::Sampled)
            {
                return None;
            }
            self.texture_descriptor(resource.metal_index)
                .map(|descriptor| (resource.metal_index, descriptor))
        })
    }

    pub fn buffer_access(&self, metal_index: u32) -> ReflectedBufferAccess {
        let Some(access) = self.bindings.iter().find_map(|resource| {
            (resource.kind == ShaderResourceKind::Buffer && resource.metal_index == metal_index)
                .then_some(resource.access)
        }) else {
            return ReflectedBufferAccess::Absent;
        };
        match access {
            Some(ShaderResourceAccess::Unused) => ReflectedBufferAccess::Unused,
            Some(ShaderResourceAccess::ReadOnly) => ReflectedBufferAccess::ReadOnly,
            Some(ShaderResourceAccess::WriteOnly | ShaderResourceAccess::ReadWrite) => {
                ReflectedBufferAccess::Writable
            }
            Some(ShaderResourceAccess::Sampled | ShaderResourceAccess::Storage) | None => {
                ReflectedBufferAccess::Unknown
            }
        }
    }

    pub fn sampled_kind(&self, binding: u32) -> ReflectedSampledKind {
        let Some(shape) = self.texture_shape_for_binding(binding) else {
            return ReflectedSampledKind::Absent;
        };
        let kind = match (shape.dimension, shape.arrayed, shape.multisampled) {
            (ShaderTextureDimension::D1, false, false) => Some(SampledImageKind::D1),
            (ShaderTextureDimension::D1, true, false) => Some(SampledImageKind::D1Array),
            (ShaderTextureDimension::D2, false, false) => Some(SampledImageKind::D2),
            (ShaderTextureDimension::D2, false, true) => Some(SampledImageKind::D2Multisample),
            (ShaderTextureDimension::D2, true, false) => Some(SampledImageKind::D2Array),
            (ShaderTextureDimension::D3, false, false) => Some(SampledImageKind::D3),
            (ShaderTextureDimension::Cube, false, false) => Some(SampledImageKind::Cube),
            (ShaderTextureDimension::Cube, true, false) => Some(SampledImageKind::CubeArray),
            _ => None,
        };
        kind.map_or(
            ReflectedSampledKind::Unsupported,
            ReflectedSampledKind::Kind,
        )
    }

    pub fn compute_texture(&self, binding: u32) -> ReflectedComputeTexture {
        let Some(shape) = self.texture_shape_for_binding(binding) else {
            return ReflectedComputeTexture::Absent;
        };
        let axis = match shape.dimension {
            ShaderTextureDimension::D1 => Some("dim_1d"),
            ShaderTextureDimension::D3 => Some("dim_3d"),
            ShaderTextureDimension::Cube => Some("dim_cube"),
            ShaderTextureDimension::Buffer => Some("dim_buffer"),
            ShaderTextureDimension::D2 if shape.arrayed => Some("arrayed"),
            ShaderTextureDimension::D2 if shape.multisampled => Some("multisampled"),
            ShaderTextureDimension::D2 => None,
        };
        match axis {
            Some(axis) => ReflectedComputeTexture::UnstageableShape { axis },
            None => ReflectedComputeTexture::Plain2d(if shape.writable {
                ImageAccess::Storage
            } else {
                ImageAccess::Sampled
            }),
        }
    }

    pub fn storage_image_format(&self, binding: u32) -> Option<StorageImageFormat> {
        self.texture_shape_for_binding(binding)?.storage_format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(
        kind: ShaderResourceKind,
        metal_index: u32,
        binding: u32,
        count: u32,
        shape: ShaderTextureShape,
        access: ShaderResourceAccess,
    ) -> ShaderResourceBinding {
        ShaderResourceBinding {
            kind,
            metal_index,
            descriptor: Some(ShaderDescriptorLocation {
                set: 0,
                binding,
                count,
            }),
            extent: None,
            footprint: None,
            texture_shape: Some(shape),
            access: Some(access),
        }
    }

    fn shape(dimension: ShaderTextureDimension) -> ShaderTextureShape {
        ShaderTextureShape {
            dimension,
            arrayed: false,
            multisampled: false,
            component: ShaderTextureComponent::Float,
            writable: false,
            array_ref: false,
            array_length: None,
            storage_format: None,
        }
    }

    #[test]
    fn semantic_texture_arrays_resolve_slots_without_backend_numbering() {
        let interface = ShaderInterface {
            stage: ReflectedShaderStage::Fragment,
            bindings: vec![texture(
                ShaderResourceKind::TextureArray,
                4,
                33,
                3,
                ShaderTextureShape {
                    array_ref: true,
                    array_length: Some(3),
                    ..shape(ShaderTextureDimension::D2)
                },
                ShaderResourceAccess::Sampled,
            )],
            local_size: None,
            unsupported: None,
        };

        assert_eq!(
            interface.texture_descriptor(6),
            Some(ReflectedTextureDescriptor {
                binding: 33,
                array_element: 2,
                descriptor_count: 3,
                access: ReflectedTextureAccess::Sampled,
            })
        );
        assert_eq!(interface.texture_descriptor(7), None);
        assert_eq!(
            interface.sampled_kind(33),
            ReflectedSampledKind::Kind(SampledImageKind::D2)
        );
    }

    #[test]
    fn semantic_access_and_shape_classification_fail_closed() {
        let mut volume = shape(ShaderTextureDimension::D3);
        volume.writable = true;
        volume.storage_format = Some(StorageImageFormat::Rgba16Float);
        let interface = ShaderInterface {
            stage: ReflectedShaderStage::Kernel,
            bindings: vec![
                ShaderResourceBinding {
                    kind: ShaderResourceKind::Buffer,
                    metal_index: 2,
                    descriptor: Some(ShaderDescriptorLocation {
                        set: 0,
                        binding: 2,
                        count: 1,
                    }),
                    extent: None,
                    footprint: None,
                    texture_shape: None,
                    access: None,
                },
                texture(
                    ShaderResourceKind::StorageImage,
                    5,
                    37,
                    1,
                    volume,
                    ShaderResourceAccess::Storage,
                ),
            ],
            local_size: Some([8, 4, 1]),
            unsupported: None,
        };

        assert_eq!(interface.buffer_access(2), ReflectedBufferAccess::Unknown);
        assert_eq!(interface.buffer_access(3), ReflectedBufferAccess::Absent);
        assert_eq!(
            interface.compute_texture(37),
            ReflectedComputeTexture::UnstageableShape { axis: "dim_3d" }
        );
        assert_eq!(
            interface.storage_image_format(37),
            Some(StorageImageFormat::Rgba16Float)
        );
        assert_eq!(
            interface
                .first_non_sampled_texture_descriptor()
                .map(|(index, descriptor)| (index, descriptor.access)),
            Some((5, ReflectedTextureAccess::Storage))
        );
    }
}
