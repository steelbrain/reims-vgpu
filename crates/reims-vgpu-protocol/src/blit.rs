//! Decoded blit command semantics.

/// Semantic class of one decoded blit command.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlitKind {
    #[default]
    Unknown,
    Copy,
    FillBuffer,
    Resource,
    Image,
    Fence,
    IcbRange,
    IcbCopy,
    FillBufferPattern4,
    FillTexture,
    InvalidateCompressedTexture,
}

/// Where a texture fill takes the value it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlitFillSource {
    #[default]
    None,
    Color,
    Bytes,
}

/// Direction and endpoint classes of a decoded blit copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlitCopyKind {
    #[default]
    None,
    BufferToTexture,
    BufferToBuffer,
    TextureToBuffer,
    TextureToTexture,
    TextureToTextureSliceLevel,
}

/// Semantic object class named by a blit reference field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlitRefKind {
    #[default]
    None,
    Buffer,
    Texture,
    Resource,
    IndirectCommandBuffer,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlitPoint {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlitSize {
    pub width: u64,
    pub height: u64,
    pub depth: u64,
}

/// One fully decoded blit record.
///
/// References are still serializer references at this boundary. Resolution
/// replaces them with core resource identities before immutable execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlitCommand {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: BlitKind,
    pub copy_kind: BlitCopyKind,
    pub source_kind: BlitRefKind,
    pub destination_kind: BlitRefKind,
    pub source: u32,
    pub destination: u32,
    pub source_offset: u64,
    pub source_bytes_per_row: u64,
    pub source_bytes_per_image: u64,
    pub source_origin: BlitPoint,
    pub source_size: BlitSize,
    pub destination_offset: u64,
    pub destination_bytes_per_row: u64,
    pub destination_bytes_per_image: u64,
    pub destination_origin: BlitPoint,
    pub size: u64,
    pub source_slice: u16,
    pub source_level: u16,
    pub destination_slice: u16,
    pub destination_level: u16,
    pub slice_count: u16,
    pub level_count: u16,
    pub has_options: bool,
    pub options: u32,
    pub resource: u32,
    pub resource_kind: BlitRefKind,
    pub buffer: u32,
    pub range_location: u64,
    pub range_length: u64,
    pub destination_index: u64,
    pub fill_value: u8,
    pub fill_pattern: u32,
    pub texture: u32,
    pub slice: u16,
    pub level: u16,
    pub fence: u32,
    pub fill_source: BlitFillSource,
    pub fill_origin: BlitPoint,
    pub fill_size: BlitSize,
    pub fill_color_raw: [u64; 4],
    pub fill_pixel_format: u16,
    pub fill_bytes_ref: u32,
    pub fill_bytes_offset: u64,
    pub fill_bytes_length: u64,
}
