//! Plain Metal value types: `MtlSize`/`Origin`/`Region`, `NsRange`, `SizeAndAlign`, `ClearColor`,
//! `Swizzle`, `ResourceAddressRange`, `PixelFormatInfo`, `TextureLevelInfo`,
//! `CommandBufferStatus`, and the other structs core passes by value.
//!
//! Widths follow the Objective-C runtime's encodings (D41): `NSUInteger` is `u64`, `float` is
//! `f32`, `double` is `f64`, `MTLTextureSwizzle` is `u8`. Every struct is `#[repr(C)]` with its
//! C size asserted, so the metal crate can pass it by value with a matching `Encode`.

/// `MTLSize` (`{?=QQQ}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct MtlSize {
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
    /// `depth`.
    pub depth: u64,
}

impl MtlSize {
    /// `MTLSizeMake(width, height, depth)`.
    pub const fn new(width: u64, height: u64, depth: u64) -> Self {
        MtlSize {
            width,
            height,
            depth,
        }
    }
}

/// `MTLOrigin` (`{?=QQQ}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct MtlOrigin {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `z`.
    pub z: u64,
}

impl MtlOrigin {
    /// `MTLOriginMake(x, y, z)`.
    pub const fn new(x: u64, y: u64, z: u64) -> Self {
        MtlOrigin { x, y, z }
    }
}

/// `MTLRegion` (`{?={?=QQQ}{?=QQQ}}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct MtlRegion {
    /// `origin`.
    pub origin: MtlOrigin,
    /// `size`.
    pub size: MtlSize,
}

impl MtlRegion {
    /// `MTLRegionMake3D(x, y, z, width, height, depth)`.
    pub const fn new(origin: MtlOrigin, size: MtlSize) -> Self {
        MtlRegion { origin, size }
    }
}

/// `NSRange` (`{_NSRange=QQ}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct NsRange {
    /// `location`.
    pub location: u64,
    /// `length`.
    pub length: u64,
}

impl NsRange {
    /// `NSMakeRange(location, length)`.
    pub const fn new(location: u64, length: u64) -> Self {
        NsRange { location, length }
    }
}

/// `MTLSizeAndAlign` (`{?=QQ}`): `heap*SizeAndAlignWith*`'s result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct SizeAndAlign {
    /// `size`.
    pub size: u64,
    /// `align`.
    pub align: u64,
}

/// What one argument of an argument buffer holds ([`ArgumentMember::kind`]): every kind but
/// [`Constant`](Self::Constant) and [`Struct`](Self::Struct) is an 8-byte word, little endian, a
/// buffer's GPU address plus its offset or another object's `gpuResourceID`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArgumentKind {
    /// A buffer's address (`setBuffer:offset:atIndex:`); a pointer to a nested argument buffer
    /// where the member has a layout.
    Buffer,
    /// A texture's resource ID.
    Texture,
    /// A sampler state's resource ID.
    Sampler,
    /// Plain bytes (`constantDataAtIndex:`).
    Constant,
    /// An indirect command buffer's resource ID.
    IndirectCommandBuffer,
    /// A compute pipeline state's resource ID.
    ComputePipelineState,
    /// A render pipeline state's resource ID.
    RenderPipelineState,
    /// A visible function table's resource ID.
    VisibleFunctionTable,
    /// An intersection function table's resource ID.
    IntersectionFunctionTable,
    /// An acceleration structure's resource ID.
    AccelerationStructure,
    /// A depth/stencil state.
    DepthStencilState,
    /// A struct of arguments, laid out as its member's layout says.
    Struct,
}

/// One member of an [`ArgumentLayout`]: an argument, or an array of them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArgumentMember {
    /// The argument index (`[[id(n)]]`) of element 0.
    pub id: u32,
    /// Elements: 1 for a single argument.
    pub count: u32,
    /// Argument indices each element takes: 1, or a struct's span.
    pub ids_per_element: u32,
    /// Byte offset of element 0 in the argument buffer.
    pub offset: u64,
    /// Bytes from one element to the next.
    pub stride: u64,
    /// Bytes of one element.
    pub size: u64,
    /// What the member holds.
    pub kind: ArgumentKind,
    /// For a [`ArgumentKind::Buffer`] that points to an argument buffer, that buffer's layout
    /// (`newArgumentEncoderForBufferAtIndex:`); for a [`ArgumentKind::Struct`], one element's,
    /// at indices and offsets from the element's. `None` otherwise.
    pub layout: Option<Box<ArgumentLayout>>,
}

/// The layout of one argument buffer, as an `MTLArgumentEncoder` writes it: its length, its
/// alignment, and where each argument index lands.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ArgumentLayout {
    /// `encodedLength`.
    pub encoded_length: u64,
    /// `alignment`.
    pub alignment: u64,
    /// The members, in layout order. An index two members hold lands in the later one.
    pub members: Vec<ArgumentMember>,
}

/// `MTLClearColor` (`{?=dddd}`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct ClearColor {
    /// `red`.
    pub red: f64,
    /// `green`.
    pub green: f64,
    /// `blue`.
    pub blue: f64,
    /// `alpha`.
    pub alpha: f64,
}

/// `MTLTextureSwizzleChannels` (`{?=CCCC}`): each channel a raw `MTLTextureSwizzle` (`uint8_t`),
/// unvalidated as core forwards it (4.2 item 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct Swizzle {
    /// `red`.
    pub red: u8,
    /// `green`.
    pub green: u8,
    /// `blue`.
    pub blue: u8,
    /// `alpha`.
    pub alpha: u8,
}

/// `MTLViewport` (`{?=dddddd}`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct MtlViewport {
    /// `originX`.
    pub origin_x: f64,
    /// `originY`.
    pub origin_y: f64,
    /// `width`.
    pub width: f64,
    /// `height`.
    pub height: f64,
    /// `znear`.
    pub znear: f64,
    /// `zfar`.
    pub zfar: f64,
}

/// `MTLScissorRect` (`{?=QQQQ}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct MtlScissorRect {
    /// `x`.
    pub x: u64,
    /// `y`.
    pub y: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
}

/// `MTLSamplePosition` (`{?=ff}`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct MtlSamplePosition {
    /// `x`.
    pub x: f32,
    /// `y`.
    pub y: f32,
}

/// `MTLCoordinate2D`: the SDK's typedef of `MTLSamplePosition`.
pub type MtlCoordinate2D = MtlSamplePosition;

/// `MTLVertexAmplificationViewMapping` (`{?=II}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct VertexAmplificationViewMapping {
    /// `viewportArrayIndexOffset`.
    pub viewport_array_index_offset: u32,
    /// `renderTargetArrayIndexOffset`.
    pub render_target_array_index_offset: u32,
}

/// `MTLResourceAddressRange`: one range of an `MTLResourceAddressRangeArray`. The type keeps the
/// SPI's name, but the runtime tags the struct
/// `MTLAddressRange` (`^{MTLAddressRange=QQ}` in `-initWithRanges:count:`), so the metal crate
/// encodes it as `Struct("MTLAddressRange", [u64, u64])` (D41). The array copies its ranges, so a
/// slice on the stack may be passed (the metal crate's test `new_address_range_array`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct ResourceAddressRange {
    /// `address` (name is ours: the host VA the range starts at).
    pub address: u64,
    /// `length` (name is ours: the range's length in bytes).
    pub length: u64,
}

/// `MTLPixelFormatInfo`: what `MTLPixelFormatGetInfoForDevice` returns (0x38 bytes; names are
/// Metal's metadata's). From +0x18 Metal has a union, and these are its compressed form's names;
/// copies the four words at +0x18 whole as a plane, so all four are kept raw. They are a block's
/// only for a format of blocks (flags bit `0x400`: the compressed formats, `GBGR422` and
/// `BGRG422`); any other format's are its texel's bytes, a packed word and two more (R8Unorm's
/// `[1, 257, 1, 1]`), so `block_width` is no width there.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PixelFormatInfo {
    /// `name` (+0x00): Metal's C string, `None` for NULL. Diagnostics only (the blit decoder's
    /// throw reason).
    pub name: Option<String>,
    /// `flags` (+0x08).
    pub flags: u64,
    /// `castClass` (+0x10).
    pub cast_class: u64,
    /// `blockBytes` (+0x18).
    pub block_bytes: u64,
    /// `blockWidth` (+0x20).
    pub block_width: u64,
    /// `blockHeight` (+0x28).
    pub block_height: u64,
    /// `blockDepth` (+0x30).
    pub block_depth: u64,
}

/// `MTLGetTextureLevelInfoForDeviceWithOptions`'s arguments after the device (passes
/// `format, w, h, 1, 1, 0, 0`). The parameter names are ours: the function has C linkage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TextureLevelQuery {
    /// `pixelFormat`.
    pub pixel_format: u64,
    /// `width`.
    pub width: u64,
    /// `height`.
    pub height: u64,
    /// `depth`.
    pub depth: u64,
    /// `arrayLength`.
    pub array_length: u64,
    /// `options`.
    pub options: u64,
    /// `flags`.
    pub flags: u64,
}

/// `MTLTextureLevelInfo`: what `MTLGetTextureLevelInfoForDeviceWithOptions` returns, 0x68 bytes.
/// The function has C linkage, so nothing states the fields' names; the three core reads carry
/// names that are ours, the rest stay `field_xx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(C)]
pub struct TextureLevelInfo {
    /// +0x00.
    pub field_0: u64,
    /// +0x08.
    pub field_8: u64,
    /// +0x10.
    pub field_10: u64,
    /// +0x18.
    pub field_18: u64,
    /// +0x20.
    pub field_20: u64,
    /// +0x28.
    pub field_28: u64,
    /// +0x30.
    pub field_30: u64,
    /// +0x38: the height rounded up to whole blocks (name is ours: core multiplies it by
    /// the array length).
    pub aligned_height: u64,
    /// +0x40.
    pub field_40: u64,
    /// +0x48 (name is ours: compared with the surface's bytes per element).
    pub bytes_per_pixel: u64,
    /// +0x50 (name is ours: compared with the surface's row bytes).
    pub bytes_per_row: u64,
    /// +0x58.
    pub field_58: u64,
    /// +0x60.
    pub field_60: u64,
}

/// `MTLCommandBufferStatus` (`NSUInteger`), raw as Metal returns it (4.2 item 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CommandBufferStatus(pub u64);

impl CommandBufferStatus {
    /// `MTLCommandBufferStatusNotEnqueued`.
    pub const NOT_ENQUEUED: CommandBufferStatus = CommandBufferStatus(0);
    /// `MTLCommandBufferStatusEnqueued`.
    pub const ENQUEUED: CommandBufferStatus = CommandBufferStatus(1);
    /// `MTLCommandBufferStatusCommitted`.
    pub const COMMITTED: CommandBufferStatus = CommandBufferStatus(2);
    /// `MTLCommandBufferStatusScheduled`.
    pub const SCHEDULED: CommandBufferStatus = CommandBufferStatus(3);
    /// `MTLCommandBufferStatusCompleted`.
    pub const COMPLETED: CommandBufferStatus = CommandBufferStatus(4);
    /// `MTLCommandBufferStatusError`.
    pub const ERROR: CommandBufferStatus = CommandBufferStatus(5);

    /// The raw value.
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// An `NSError` core logs or forwards (the service's `completeAsyncTokenID:success:error:`).
/// Name from architecture 4.4. `None` is nil.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct NsErrorText {
    /// `-[NSError domain]`.
    pub domain: String,
    /// `-[NSError code]`.
    pub code: i64,
    /// `-[NSError description]`.
    pub description: String,
}

impl std::fmt::Display for NsErrorText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.description)
    }
}

/// The zero an Objective-C message to nil returns (architecture 4.1): what `nil_send!` yields for
/// a nil receiver. `RaiseResult<T>` gives `Ok(T::nil_default())`: a send to nil never raises.
///
/// Only types a platform method returns implement it; the other platform files add theirs.
pub trait NilDefault {
    /// The value a message to nil returns.
    fn nil_default() -> Self;
}

impl<T> NilDefault for Option<T> {
    fn nil_default() -> Self {
        None
    }
}

/// A creator with an `NSError **` out-parameter sent to nil: nil, and the error left nil
/// (`Ok(None)` for the `Option` results these methods return).
impl<T: NilDefault> NilDefault for Result<T, NsErrorText> {
    fn nil_default() -> Self {
        Ok(T::nil_default())
    }
}

impl<T: NilDefault> NilDefault for crate::raise::RaiseResult<T> {
    fn nil_default() -> Self {
        Ok(T::nil_default())
    }
}

/// `NilDefault` as `Default`, for the types whose default is all zero.
macro_rules! nil_default_is_default {
    ($($t:ty),* $(,)?) => {
        $(impl NilDefault for $t {
            fn nil_default() -> Self {
                <$t as Default>::default()
            }
        })*
    };
}

nil_default_is_default!(
    (),
    bool,
    u8,
    u16,
    u32,
    u64,
    i32,
    i64,
    f32,
    f64,
    Vec<String>,
    MtlSize,
    MtlOrigin,
    MtlRegion,
    NsRange,
    SizeAndAlign,
    ClearColor,
    Swizzle,
    MtlViewport,
    MtlScissorRect,
    MtlSamplePosition,
    VertexAmplificationViewMapping,
    ResourceAddressRange,
    PixelFormatInfo,
    TextureLevelInfo,
    CommandBufferStatus,
    crate::host::HostPtr,
);

const _: () = assert!(size_of::<MtlSize>() == 24);
const _: () = assert!(size_of::<MtlOrigin>() == 24);
const _: () = assert!(size_of::<MtlRegion>() == 48);
const _: () = assert!(size_of::<NsRange>() == 16);
const _: () = assert!(size_of::<SizeAndAlign>() == 16);
const _: () = assert!(size_of::<ClearColor>() == 32);
const _: () = assert!(size_of::<Swizzle>() == 4);
const _: () = assert!(size_of::<MtlViewport>() == 48);
const _: () = assert!(size_of::<MtlScissorRect>() == 32);
const _: () = assert!(size_of::<MtlSamplePosition>() == 8);
const _: () = assert!(size_of::<VertexAmplificationViewMapping>() == 8);
const _: () = assert!(size_of::<ResourceAddressRange>() == 16);
const _: () = assert!(size_of::<TextureLevelInfo>() == 0x68);
const _: () = assert!(std::mem::offset_of!(TextureLevelInfo, aligned_height) == 0x38);
const _: () = assert!(std::mem::offset_of!(TextureLevelInfo, bytes_per_pixel) == 0x48);
const _: () = assert!(std::mem::offset_of!(TextureLevelInfo, bytes_per_row) == 0x50);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::RaiseResult;

    /// The `Make` constructors put each argument in its field
    #[test]
    fn makers() {
        let s = MtlSize::new(1, 2, 3);
        assert_eq!((s.width, s.height, s.depth), (1, 2, 3));
        let o = MtlOrigin::new(4, 5, 6);
        assert_eq!((o.x, o.y, o.z), (4, 5, 6));
        let r = MtlRegion::new(o, s);
        assert_eq!((r.origin, r.size), (o, s));
        let n = NsRange::new(7, 8);
        assert_eq!((n.location, n.length), (7, 8));
    }

    /// The status constants are Metal's raw values
    #[test]
    fn command_buffer_status_values() {
        assert_eq!(CommandBufferStatus::NOT_ENQUEUED.raw(), 0);
        assert_eq!(CommandBufferStatus::ENQUEUED.raw(), 1);
        assert_eq!(CommandBufferStatus::COMMITTED.raw(), 2);
        assert_eq!(CommandBufferStatus::SCHEDULED.raw(), 3);
        assert_eq!(CommandBufferStatus::COMPLETED.raw(), 4);
        assert_eq!(CommandBufferStatus::ERROR.raw(), 5);
        // An unknown value stays representable (4.2 item 4).
        assert_eq!(CommandBufferStatus(9).raw(), 9);
        assert_eq!(
            CommandBufferStatus::default(),
            CommandBufferStatus::NOT_ENQUEUED
        );
    }

    /// An NSError prints as its description
    #[test]
    fn ns_error_text_displays_description() {
        let e = NsErrorText {
            domain: "MTLCommandBufferErrorDomain".into(),
            code: 2,
            description: "Execution of the command buffer was aborted".into(),
        };
        assert_eq!(e.to_string(), "Execution of the command buffer was aborted");
    }

    /// Nil returns zero, nil, and `Ok(zero)` for a raising method
    #[test]
    fn nil_defaults_are_zero() {
        assert_eq!(<u64 as NilDefault>::nil_default(), 0);
        assert!(!<bool as NilDefault>::nil_default());
        assert_eq!(<Option<u32> as NilDefault>::nil_default(), None);
        assert_eq!(<RaiseResult<u64> as NilDefault>::nil_default(), Ok(0));
        assert_eq!(
            <RaiseResult<Option<u8>> as NilDefault>::nil_default(),
            Ok(None)
        );
        assert_eq!(<RaiseResult<()> as NilDefault>::nil_default(), Ok(()));
        assert_eq!(
            <SizeAndAlign as NilDefault>::nil_default(),
            SizeAndAlign { size: 0, align: 0 }
        );
        assert_eq!(
            <MtlRegion as NilDefault>::nil_default(),
            MtlRegion::default()
        );
        assert_eq!(<PixelFormatInfo as NilDefault>::nil_default().name, None);
        assert_eq!(
            <CommandBufferStatus as NilDefault>::nil_default(),
            CommandBufferStatus(0)
        );
    }
}
