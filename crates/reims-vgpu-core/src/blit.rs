//! Immutable, resource-resolved blit operations.

use crate::{pixel_format::BlitAspect, ContentStamp};
use reims_vgpu_protocol::{ByteLength, GuestVirtualAddress, MappingId, ResourceId, ResourceObject};

/// One resolved mip level in a task-address texture allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedLinearTextureLevel {
    /// Allocation base guest virtual address.
    pub base_gva: u64,
    pub alloc_size: u64,
    pub level_offset: u64,
    pub row_stride: u64,
    /// Byte stride between array slices or cube faces; zero for one slice.
    pub slice_stride: u64,
    /// Absolute array slice or cube face selected by view resolution.
    pub slice_index: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub bpp: u32,
    pub pixel_format: u16,
}

impl ResolvedLinearTextureLevel {
    pub fn bytes_per_image(&self) -> Option<u64> {
        self.row_stride.checked_mul(u64::from(self.height))
    }

    /// Byte offset of texel origin `(x, y, z)` within the allocation.
    pub fn texel_offset(&self, x: u64, y: u64, z: u64) -> Option<u64> {
        let plane = z.checked_mul(self.bytes_per_image()?)?;
        let slice = if self.slice_index == 0 || self.slice_stride == 0 {
            0
        } else {
            u64::from(self.slice_index).checked_mul(self.slice_stride)?
        };
        self.level_offset
            .checked_add(slice)?
            .checked_add(plane)?
            .checked_add(y.checked_mul(self.row_stride)?)?
            .checked_add(x.checked_mul(u64::from(self.bpp))?)
    }
}

/// One resolved plane in a registered surface mapping.
///
/// Registered-surface textures are single-level and two-dimensional. Plane
/// selection has already happened before this value exists; `surface_offset`,
/// `row_stride`, and `span_end` describe the selected window in the mapping.
/// The mapping remains a typed relation and is not interchangeable with an
/// object-table or serializer reference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedSurfaceTextureBacking {
    pub mapping_id: MappingId,
    pub width: u32,
    pub height: u32,
    pub surface_offset: u64,
    /// The surface-window contract carries its row stride as `u32` end to end.
    pub row_stride: u32,
    /// Exclusive end of the selected plane window in the mapping.
    pub span_end: u64,
    pub bpp: u32,
    pub pixel_format: u16,
}

/// Backend-independent guest storage behind one resolved texture endpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedTextureBacking {
    Linear(ResolvedLinearTextureLevel),
    Surface(ResolvedSurfaceTextureBacking),
}

/// One generational texture resource paired with its resolved guest storage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedTextureEndpoint {
    pub content: ContentStamp,
    pub backing: ResolvedTextureBacking,
}

/// Texel origin within a resolved texture level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextureOrigin {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

/// Three-dimensional extent of a texture transfer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextureExtent {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// A buffer-to-texture transfer after both serializer references have resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedBufferToTextureBlit {
    /// Source bytes beginning at the command's source offset and extending to
    /// the end of the resolved buffer allocation.
    pub source: ResolvedBufferRange,
    pub source_bytes_per_row: u64,
    pub source_bytes_per_image: u64,
    pub destination: ResolvedTextureEndpoint,
    pub destination_origin: TextureOrigin,
    pub extent: TextureExtent,
    pub aspect: BlitAspect,
}

/// A texture-to-buffer transfer after both serializer references have resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedTextureToBufferBlit {
    pub source: ResolvedTextureEndpoint,
    pub source_origin: TextureOrigin,
    pub extent: TextureExtent,
    /// Destination bytes beginning at the command's destination offset and
    /// extending to the end of the resolved buffer allocation.
    pub destination: ResolvedBufferRange,
    pub destination_bytes_per_row: u64,
    pub destination_bytes_per_image: u64,
    pub aspect: BlitAspect,
}

/// A texture-region copy after both serializer references have resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedTextureToTextureBlit {
    pub source: ResolvedTextureEndpoint,
    pub source_origin: TextureOrigin,
    pub destination: ResolvedTextureEndpoint,
    pub destination_origin: TextureOrigin,
    pub extent: TextureExtent,
    pub aspect: BlitAspect,
}

/// All slice pairs at one mip level of a whole-texture copy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedTextureLevelCopy {
    pub first_slice: (ResolvedTextureEndpoint, ResolvedTextureEndpoint),
    pub remaining_slices: Box<[(ResolvedTextureEndpoint, ResolvedTextureEndpoint)]>,
}

/// One multi-slice, multi-level texture copy resolved without forcing guest
/// bytes current. Execution may consume an authoritative resident directly;
/// its guest-byte fallback explicitly settles the named resources first.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedTextureCopyBatch {
    pub source_base_slice: u16,
    pub destination_base_slice: u16,
    pub first_level: ResolvedTextureLevelCopy,
    pub remaining_levels: Box<[ResolvedTextureLevelCopy]>,
}

impl ResolvedTextureBacking {
    pub const fn width(&self) -> u32 {
        match self {
            Self::Linear(texture) => texture.width,
            Self::Surface(texture) => texture.width,
        }
    }

    pub const fn height(&self) -> u32 {
        match self {
            Self::Linear(texture) => texture.height,
            Self::Surface(texture) => texture.height,
        }
    }

    pub const fn depth(&self) -> u32 {
        match self {
            Self::Linear(texture) => texture.depth,
            Self::Surface(_) => 1,
        }
    }

    pub const fn bpp(&self) -> u32 {
        match self {
            Self::Linear(texture) => texture.bpp,
            Self::Surface(texture) => texture.bpp,
        }
    }

    pub const fn pixel_format(&self) -> u16 {
        match self {
            Self::Linear(texture) => texture.pixel_format,
            Self::Surface(texture) => texture.pixel_format,
        }
    }

    pub const fn is_surface(&self) -> bool {
        matches!(self, Self::Surface(_))
    }
}

/// One checked byte range over a resolved buffer resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResolvedBufferRange {
    pub content: ContentStamp,
    pub address: GuestVirtualAddress,
    pub length: ByteLength,
}

impl ResolvedBufferRange {
    pub const fn resource(self) -> ResourceId<ResourceObject> {
        self.content.resource
    }
}

/// The contract-defined repeating unit of a buffer fill.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferFillPattern {
    Byte(u8),
    Word([u8; 4]),
}

impl BufferFillPattern {
    pub const fn bytes(&self) -> &[u8] {
        match self {
            Self::Byte(value) => core::slice::from_ref(value),
            Self::Word(value) => value,
        }
    }
}

/// A blit whose serializer references and backing identities are resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResolvedBlit {
    Fill {
        destination: ResolvedBufferRange,
        pattern: BufferFillPattern,
    },
    Copy {
        source: ResolvedBufferRange,
        destination: ResolvedBufferRange,
    },
    BufferToTexture(ResolvedBufferToTextureBlit),
    TextureToBuffer(ResolvedTextureToBufferBlit),
    TextureToTexture(ResolvedTextureToTextureBlit),
    TextureCopyBatch(ResolvedTextureCopyBatch),
}

impl ResolvedBlit {
    pub const fn destination_content(&self) -> ContentStamp {
        match self {
            Self::Fill { destination, .. } | Self::Copy { destination, .. } => destination.content,
            Self::BufferToTexture(operation) => operation.destination.content,
            Self::TextureToBuffer(operation) => operation.destination.content,
            Self::TextureToTexture(operation) => operation.destination.content,
            Self::TextureCopyBatch(operation) => operation.first_level.first_slice.1.content,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_protocol::{ContentVersion, ResourceId};

    fn range(index: u32, generation: u32, address: u64) -> ResolvedBufferRange {
        ResolvedBufferRange {
            content: ContentStamp {
                resource: ResourceId::new(index, generation),
                version: ContentVersion::new(4),
            },
            address: GuestVirtualAddress::new(address),
            length: ByteLength::new(16),
        }
    }

    fn linear_endpoint(index: u32) -> ResolvedTextureEndpoint {
        ResolvedTextureEndpoint {
            content: range(index, 1, u64::from(index) << 12).content,
            backing: ResolvedTextureBacking::Linear(ResolvedLinearTextureLevel {
                base_gva: u64::from(index) << 12,
                alloc_size: 0x1000,
                level_offset: 0,
                row_stride: 64,
                slice_stride: 0,
                slice_index: 0,
                width: 8,
                height: 4,
                depth: 1,
                bpp: 4,
                pixel_format: 80,
            }),
        }
    }

    #[test]
    fn resolved_blits_carry_generational_resources_not_serializer_ordinals() {
        let operation = ResolvedBlit::Copy {
            source: range(7, 2, 0x1000),
            destination: range(7, 3, 0x2000),
        };

        assert_eq!(
            operation.destination_content().resource,
            ResourceId::new(7, 3)
        );
    }

    #[test]
    fn resolved_texture_storage_keeps_mapping_identity_and_level_geometry() {
        let linear = ResolvedLinearTextureLevel {
            base_gva: 0x1000,
            alloc_size: 0x8000,
            level_offset: 0x200,
            row_stride: 64,
            slice_stride: 0x1000,
            slice_index: 2,
            width: 8,
            height: 4,
            depth: 1,
            bpp: 4,
            pixel_format: 80,
        };
        assert_eq!(linear.bytes_per_image(), Some(256));
        assert_eq!(
            linear.texel_offset(3, 2, 0),
            Some(0x200 + 0x2000 + 128 + 12)
        );

        let surface = ResolvedTextureBacking::Surface(ResolvedSurfaceTextureBacking {
            mapping_id: MappingId::new(9),
            width: 1920,
            height: 1080,
            surface_offset: 0,
            row_stride: 7680,
            span_end: 8_294_400,
            bpp: 4,
            pixel_format: 80,
        });
        assert!(surface.is_surface());
        assert_eq!(surface.depth(), 1);
        let ResolvedTextureBacking::Surface(surface) = surface else {
            unreachable!()
        };
        assert_eq!(surface.mapping_id, MappingId::new(9));
    }

    #[test]
    fn resolved_buffer_to_texture_carries_only_generational_endpoints() {
        let destination = ContentStamp {
            resource: ResourceId::new(11, 4),
            version: ContentVersion::new(6),
        };
        let operation = ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
            source: range(7, 3, 0x2000),
            source_bytes_per_row: 64,
            source_bytes_per_image: 256,
            destination: ResolvedTextureEndpoint {
                content: destination,
                backing: ResolvedTextureBacking::Surface(ResolvedSurfaceTextureBacking {
                    mapping_id: MappingId::new(9),
                    width: 8,
                    height: 4,
                    surface_offset: 0,
                    row_stride: 64,
                    span_end: 256,
                    bpp: 4,
                    pixel_format: 80,
                }),
            },
            destination_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 8,
                height: 4,
                depth: 1,
            },
            aspect: BlitAspect::Full,
        });

        assert_eq!(operation.destination_content(), destination);
    }

    #[test]
    fn resolved_texture_to_buffer_names_the_destination_lifetime() {
        let source = ContentStamp {
            resource: ResourceId::new(11, 4),
            version: ContentVersion::new(6),
        };
        let destination = range(12, 8, 0x4000);
        let operation = ResolvedBlit::TextureToBuffer(ResolvedTextureToBufferBlit {
            source: ResolvedTextureEndpoint {
                content: source,
                backing: ResolvedTextureBacking::Linear(ResolvedLinearTextureLevel {
                    base_gva: 0x1000,
                    alloc_size: 0x1000,
                    level_offset: 0,
                    row_stride: 64,
                    slice_stride: 0,
                    slice_index: 0,
                    width: 8,
                    height: 4,
                    depth: 1,
                    bpp: 4,
                    pixel_format: 80,
                }),
            },
            source_origin: TextureOrigin { x: 0, y: 0, z: 0 },
            extent: TextureExtent {
                width: 4,
                height: 1,
                depth: 1,
            },
            destination,
            destination_bytes_per_row: 16,
            destination_bytes_per_image: 16,
            aspect: BlitAspect::Full,
        });

        assert_eq!(operation.destination_content(), destination.content);
    }

    #[test]
    fn a_texture_copy_batch_is_non_empty_by_construction() {
        let destination = linear_endpoint(22);
        let operation = ResolvedBlit::TextureCopyBatch(ResolvedTextureCopyBatch {
            source_base_slice: 0,
            destination_base_slice: 0,
            first_level: ResolvedTextureLevelCopy {
                first_slice: (linear_endpoint(21), destination.clone()),
                remaining_slices: Box::new([]),
            },
            remaining_levels: Box::new([]),
        });

        assert_eq!(operation.destination_content(), destination.content);
    }
}
