//! Platform trait: `Surface` and `IoSurfaces`: IOSurface creation, getters and retain counting.
//!
//! The retain count is guest-visible: a mapper unmap is `kIOReturnBusy` unless it is 1, and every
//! live Metal texture or buffer over the surface holds one. So a `Surface` is exactly one CF
//! reference: not `Clone`, `.clone_ref()` is a `CFRetain` core makes, and `Drop` is `CFRelease`
//! (D6).

use crate::core::host::{HostPtr, PhysicalMemoryRange};
use crate::core::platform::{Handle, Platform};

use super::vm::KernReturn;

/// One CF reference to an IOSurface (`IOSurfaceRef`, toll-free bridged with `IOSurface *`).
pub trait Surface: Handle {
    /// `CFGetRetainCount` — guest-visible: a mapper unmap is Busy unless 1.
    fn retain_count(&self) -> i64;
    /// `IOSurfaceGetPixelFormat` (`OSType`).
    fn pixel_format(&self) -> u32;
    /// `IOSurfaceGetWidth`.
    fn width(&self) -> u64;
    /// `IOSurfaceGetHeight`.
    fn height(&self) -> u64;
    /// `IOSurfaceGetBytesPerElement`.
    fn bytes_per_element(&self) -> u64;
    /// `IOSurfaceGetBytesPerRow`.
    fn bytes_per_row(&self) -> u64;
    /// `IOSurfaceGetPlaneCount`.
    fn plane_count(&self) -> u64;
    /// `IOSurfaceAllowsPixelSizeCasting`.
    fn allows_pixel_size_casting(&self) -> bool;
    /// SPI `IOSurfaceGetProtectionOptions`.
    fn protection_options(&self) -> u64;
    /// `IOSurfaceLock(surface, options, NULL)` (the result is ignored).
    fn lock(&self, options: u32) -> KernReturn;
    /// `IOSurfaceUnlock(surface, options, NULL)`.
    fn unlock(&self, options: u32) -> KernReturn;
    /// `IOSurfaceGetBaseAddress`.
    fn base_address(&self) -> HostPtr;
    /// `IOSurfaceGetAllocSize`.
    fn alloc_size(&self) -> u64;
    /// `IOSurfaceLookupFromMachPort(IOSurfaceCreateMachPort(surface))`: another object over the
    /// same IOSurface, as a process the surface was sent to gets. Its retain count is its own:
    /// what holds it never holds `self`. `None` when the lookup finds nothing.
    fn lookup_by_port(&self) -> Option<Self>
    where
        Self: Sized;
}

/// Creates IOSurfaces. The private keys a spec can carry (client address ranges, direction,
/// address format, protection options, plane compression and extended pixels) are constants
/// IOSurface exports but no SDK header declares (the Vulkan crate's `os/macos/iosurface.rs`).
pub trait IoSurfaces<P: Platform>: Send + Sync {
    /// `IOSurfaceCreate(dict)` with the dictionary `spec` describes, private keys included. `None`
    /// (NULL) is the caller's `kIOReturnBadArgument`.
    fn create(&self, spec: &SurfaceSpec) -> Option<P::Surface>;
    /// `[[IOSurface alloc] initWithProperties:]`. Core does not check for nil.
    fn create_backing(&self, spec: &BackingSurfaceSpec) -> Option<P::Surface>;
    /// The `-description` of the dictionary `spec` describes, for the debug line that names the
    /// dictionary `IOSurfaceCreate` refused.
    fn describe(&self, spec: &SurfaceSpec) -> String;
}

/// The dictionary `-createSurfaceDictionaryWithDescriptor:hostData:` builds, key by key. Integer
/// widths are the `NSNumber` constructors core uses: `u64` for `numberWithUnsignedLongLong:`, `u32`
/// for `numberWithUnsignedInt:`. The element dimensions are already decoded
/// (`PGIOSurfaceElementDimension`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SurfaceSpec {
    /// `kIOSurfacePlaneInfo`: set only when the descriptor has planes (1..7); `None` leaves the
    /// key out (a plane count of 0).
    pub planes: Option<Vec<SurfacePlaneSpec>>,
    /// `kIOSurfacePixelFormat`.
    pub pixel_format: u32,
    /// `kIOSurfaceOffset`.
    pub offset: u64,
    /// `kIOSurfaceAllocSize`.
    pub alloc_size: u64,
    /// `kIOSurfaceWidth`.
    pub width: u64,
    /// `kIOSurfaceHeight`.
    pub height: u64,
    /// `kIOSurfaceBytesPerElement`.
    pub bytes_per_element: u64,
    /// `kIOSurfaceBytesPerRow`.
    pub bytes_per_row: u64,
    /// `kIOSurfaceElementWidth`.
    pub element_width: u32,
    /// `kIOSurfaceElementHeight`.
    pub element_height: u32,
    /// `kIOSurfaceAddressFormat` (private key).
    pub address_format: u32,
    /// `kIOSurfacePlaneCompressionType` (private key, at the surface level).
    pub compression_type: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnLeft` (private key).
    pub extended_pixels_on_left: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnTop` (private key).
    pub extended_pixels_on_top: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnRight` (private key).
    pub extended_pixels_on_right: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnBottom` (private key).
    pub extended_pixels_on_bottom: u32,
    /// `kIOSurfaceProtectionOptions` (private key).
    pub protection_options: u64,
    /// `kIOSurfaceClientDirection` (private key): `Some(2)` for a read-only mapping, else absent.
    pub client_direction: Option<u32>,
    /// `kIOSurfaceClientAddressRanges` (private key): an `NSData` of `PGPhysicalMemoryRange_t`, one
    /// per mapped page, each `{host VA, vm_page_size}`; absent for a host-only surface.
    pub client_address_ranges: Option<Vec<PhysicalMemoryRange>>,
}

/// One `kIOSurfacePlaneInfo` dictionary.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SurfacePlaneSpec {
    /// `kIOSurfacePlaneBase`.
    pub base: u64,
    /// `kIOSurfacePlaneOffset`.
    pub offset: u64,
    /// `kIOSurfacePlaneWidth`.
    pub width: u64,
    /// `kIOSurfacePlaneHeight`.
    pub height: u64,
    /// `kIOSurfacePlaneSize`.
    pub size: u64,
    /// `kIOSurfacePlaneBytesPerRow`.
    pub bytes_per_row: u64,
    /// `kIOSurfacePlaneBytesPerElement`.
    pub bytes_per_element: u64,
    /// `kIOSurfacePlaneElementWidth`.
    pub element_width: u32,
    /// `kIOSurfacePlaneElementHeight`.
    pub element_height: u32,
    /// `kIOSurfacePlaneCompressionFootprint` (private key).
    pub compression_footprint: u32,
    /// `kIOSurfaceAddressFormat` (private key, at the plane level).
    pub address_format: u32,
    /// `kIOSurfacePlaneCompressionType` (private key).
    pub compression_type: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnLeft` (private key).
    pub extended_pixels_on_left: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnTop` (private key).
    pub extended_pixels_on_top: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnRight` (private key).
    pub extended_pixels_on_right: u32,
    /// `kIOSurfacePlaneExtendedPixelsOnBottom` (private key).
    pub extended_pixels_on_bottom: u32,
    /// The five component arrays
    /// (`kIOSurfacePlaneComponent{BitDepths,BitOffsets,Names,Types, Ranges}`), one entry per
    /// component; empty leaves the five keys out (a component count of 0).
    pub components: Vec<SurfaceComponentSpec>,
}

/// One component of a plane: the `numberWithUnsignedChar:` entries at index `i` of the five
/// component arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SurfaceComponentSpec {
    /// `kIOSurfacePlaneComponentBitDepths[i]`.
    pub bit_depth: u8,
    /// `kIOSurfacePlaneComponentBitOffsets[i]`.
    pub bit_offset: u8,
    /// `kIOSurfacePlaneComponentNames[i]`.
    pub name: u8,
    /// `kIOSurfacePlaneComponentTypes[i]`.
    pub type_: u8,
    /// `kIOSurfacePlaneComponentRanges[i]`.
    pub range: u8,
}

/// The properties `-[PGBackingResource initWithTask:descriptor:planes:planeCount:]` builds for
/// `initWithProperties:`. The `@(…)` boxings: `u32` fields are `unsigned int` numbers; the two
/// bit-field planes' values promote to `int` (`i32`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BackingSurfaceSpec {
    /// `IOSurfacePropertyKeyPixelFormat` (`descriptor->pixelFormat`).
    pub pixel_format: u32,
    /// `IOSurfacePropertyKeyPlaneInfo`, one per plane, in order.
    pub planes: Vec<BackingPlaneSpec>,
    /// `IOSurfacePropertyKeyWidth` (`_planes[0].width`).
    pub width: u32,
    /// `IOSurfacePropertyKeyHeight` (`_planes[0].height`).
    pub height: u32,
}

/// One plane dictionary of [`BackingSurfaceSpec`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BackingPlaneSpec {
    /// `IOSurfacePropertyKeyPlaneWidth`.
    pub width: u32,
    /// `IOSurfacePropertyKeyPlaneHeight`.
    pub height: u32,
    /// `IOSurfacePropertyKeyPlaneBytesPerElement` (an 8-bit bit-field, boxed as `int`).
    pub bytes_per_element: i32,
    /// `IOSurfacePropertyKeyPlaneBytesPerRow` (a 24-bit bit-field, boxed as `int`).
    pub bytes_per_row: i32,
    /// `IOSurfacePropertyKeyPlaneBase`: core sets it to the plane's offset too.
    pub base: u32,
    /// `IOSurfacePropertyKeyPlaneOffset`.
    pub offset: u32,
}
