//! `IOSurfaceCreate`, the private key externs, CF retain counting.
//!
//! Which keys a dictionary holds, and every check made before `IOSurfaceCreate` (the plane and
//! component counts), are core's (`iosurface::host_device`, the `SurfaceSpec` it builds). Here the
//! spec becomes an `NSMutableDictionary`, key by key in the spec's order, with these `NSNumber`
//! constructors (`numberWithUnsignedLongLong:` for `u64`, `numberWithUnsignedInt:` for `u32`,
//! `numberWithUnsignedChar:` for the component arrays). Each Foundation send is a
//! [`value`] send, which dies at the send if it raises.

use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::AnyObject;
use objc2_foundation::{
    NSArray, NSDictionary, NSMutableArray, NSMutableData, NSMutableDictionary, NSNumber, NSString,
};
use objc2_io_surface::{
    IOSurface, IOSurfacePropertyKeyHeight, IOSurfacePropertyKeyPixelFormat,
    IOSurfacePropertyKeyPlaneBase, IOSurfacePropertyKeyPlaneBytesPerElement,
    IOSurfacePropertyKeyPlaneBytesPerRow, IOSurfacePropertyKeyPlaneHeight,
    IOSurfacePropertyKeyPlaneInfo, IOSurfacePropertyKeyPlaneOffset, IOSurfacePropertyKeyPlaneWidth,
    IOSurfacePropertyKeyWidth, IOSurfaceRef,
};
use reims_vgpu_core::host::{HostPtr, PhysicalMemoryRange};
use reims_vgpu_core::num::ToU64;
use reims_vgpu_core::platform::os::surface::{
    BackingPlaneSpec, BackingSurfaceSpec, SurfacePlaneSpec, SurfaceSpec,
};
use reims_vgpu_core::platform::os::vm::KernReturn;
use reims_vgpu_core::platform::{Handle, ObjId, Platform};

use crate::msg::{fatal_send, value};
use crate::platform::MetalPlatform;

/// One CF reference to an `IOSurfaceRef` (core's `Surface`: `retained()` is `CFRetain`, `Drop`
/// is `CFRelease`).
#[derive(Debug)]
pub struct MetalSurface {
    ptr: NonNull<IOSurfaceRef>,
}

// SAFETY: an IOSurfaceRef is a thread-safe CF object, shared across threads.
unsafe impl Send for MetalSurface {}
// SAFETY: as above.
unsafe impl Sync for MetalSurface {}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRetain(cf: *const std::ffi::c_void) -> *const std::ffi::c_void;
    fn CFRelease(cf: *const std::ffi::c_void);
    /// `CFIndex CFGetRetainCount(CFTypeRef)`.
    fn CFGetRetainCount(cf: *const std::ffi::c_void) -> isize;
}

// IOSurface's C API. Declared here rather than taken from objc2-io-surface, whose bindings wrap a
// NULL base address in `NonNull` and take core-foundation types this crate does not link.
#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    /// `IOSurfaceRef IOSurfaceCreate(CFDictionaryRef)`: +1 or NULL.
    fn IOSurfaceCreate(properties: &NSDictionary<NSString, AnyObject>) -> *mut IOSurfaceRef;
    fn IOSurfaceGetPixelFormat(buffer: &IOSurfaceRef) -> u32;
    fn IOSurfaceGetWidth(buffer: &IOSurfaceRef) -> usize;
    fn IOSurfaceGetHeight(buffer: &IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerElement(buffer: &IOSurfaceRef) -> usize;
    fn IOSurfaceGetBytesPerRow(buffer: &IOSurfaceRef) -> usize;
    fn IOSurfaceGetPlaneCount(buffer: &IOSurfaceRef) -> usize;
    fn IOSurfaceAllowsPixelSizeCasting(buffer: &IOSurfaceRef) -> u8;
    /// SPI, declared here: `extern "C" uint64_t IOSurfaceGetProtectionOptions(IOSurfaceRef)`.
    fn IOSurfaceGetProtectionOptions(buffer: &IOSurfaceRef) -> u64;
    fn IOSurfaceLock(buffer: &IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(buffer: &IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(buffer: &IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(buffer: &IOSurfaceRef) -> usize;
    /// `mach_port_t IOSurfaceCreateMachPort(IOSurfaceRef)`: a send right the caller owns.
    fn IOSurfaceCreateMachPort(buffer: &IOSurfaceRef) -> u32;
    /// `IOSurfaceRef IOSurfaceLookupFromMachPort(mach_port_t)`: +1 or NULL.
    fn IOSurfaceLookupFromMachPort(port: u32) -> *mut IOSurfaceRef;

    // The public dictionary keys (`<IOSurface/IOSurfaceRef.h>`), CFStrings toll-free bridged.
    static kIOSurfacePlaneBase: &'static NSString;
    static kIOSurfacePlaneOffset: &'static NSString;
    static kIOSurfacePlaneWidth: &'static NSString;
    static kIOSurfacePlaneHeight: &'static NSString;
    static kIOSurfacePlaneSize: &'static NSString;
    static kIOSurfacePlaneBytesPerRow: &'static NSString;
    static kIOSurfacePlaneBytesPerElement: &'static NSString;
    static kIOSurfacePlaneElementWidth: &'static NSString;
    static kIOSurfacePlaneElementHeight: &'static NSString;
    static kIOSurfacePlaneComponentBitDepths: &'static NSString;
    static kIOSurfacePlaneComponentBitOffsets: &'static NSString;
    static kIOSurfacePlaneComponentNames: &'static NSString;
    static kIOSurfacePlaneComponentTypes: &'static NSString;
    static kIOSurfacePlaneComponentRanges: &'static NSString;
    static kIOSurfacePlaneInfo: &'static NSString;
    static kIOSurfacePixelFormat: &'static NSString;
    static kIOSurfaceOffset: &'static NSString;
    static kIOSurfaceAllocSize: &'static NSString;
    static kIOSurfaceWidth: &'static NSString;
    static kIOSurfaceHeight: &'static NSString;
    static kIOSurfaceBytesPerElement: &'static NSString;
    static kIOSurfaceBytesPerRow: &'static NSString;
    static kIOSurfaceElementWidth: &'static NSString;
    static kIOSurfaceElementHeight: &'static NSString;

    // The keys IOSurface exports but the SDK does not declare, linked by name. Their values are
    // IOSurface's own: `kIOSurfaceClientAddressRanges` is "IOSurfaceAddressRanges",
    // `kIOSurfaceClientDirection` "IOSurfaceDirection" (`private_keys_link`).
    static kIOSurfaceAddressFormat: &'static NSString;
    static kIOSurfaceClientAddressRanges: &'static NSString;
    static kIOSurfaceClientDirection: &'static NSString;
    static kIOSurfacePlaneCompressionFootprint: &'static NSString;
    static kIOSurfacePlaneCompressionType: &'static NSString;
    static kIOSurfacePlaneExtendedPixelsOnBottom: &'static NSString;
    static kIOSurfacePlaneExtendedPixelsOnLeft: &'static NSString;
    static kIOSurfacePlaneExtendedPixelsOnRight: &'static NSString;
    static kIOSurfacePlaneExtendedPixelsOnTop: &'static NSString;
    static kIOSurfaceProtectionOptions: &'static NSString;
}

impl MetalSurface {
    /// Takes over one CF reference to `ptr` that the caller owns (a `Create`/`Copy` result, or
    /// a `CFRetain` the caller made). Dropping the value releases it.
    ///
    /// # Safety
    ///
    /// `ptr` is a live `IOSurfaceRef`, and the caller owns the one reference it hands over (the
    /// retain made after a `+0` getter).
    pub(crate) unsafe fn from_retained(ptr: NonNull<IOSurfaceRef>) -> MetalSurface {
        MetalSurface { ptr }
    }

    /// `CFRetain(ptr)` then [`Self::from_retained`]: a new reference to a `+0` surface.
    ///
    /// # Safety
    ///
    /// `ptr` is a live `IOSurfaceRef` for the duration of the call.
    pub(crate) unsafe fn retain(ptr: NonNull<IOSurfaceRef>) -> MetalSurface {
        // SAFETY: `ptr` is live (the caller's precondition); `CFRetain` returns its argument.
        unsafe { CFRetain(ptr.as_ptr().cast_const().cast()) };
        // SAFETY: the reference just taken is handed over.
        unsafe { MetalSurface::from_retained(ptr) }
    }

    /// The surface, for a call or send that takes an `IOSurfaceRef`.
    pub(crate) fn as_iosurface_ref(&self) -> &IOSurfaceRef {
        // SAFETY: the pointer is a live IOSurfaceRef this value holds one reference to.
        unsafe { self.ptr.as_ref() }
    }
}

impl Drop for MetalSurface {
    /// The one `CFRelease` for the reference this value holds.
    fn drop(&mut self) {
        // SAFETY: this value owns one reference to a live IOSurfaceRef (`from_retained`).
        unsafe { CFRelease(self.ptr.as_ptr().cast_const().cast()) }
    }
}

impl Handle for MetalSurface {
    /// `CFRetain`.
    fn clone_ref(&self) -> Self {
        // SAFETY: the pointer is live while `self` holds its reference.
        unsafe { MetalSurface::retain(self.ptr) }
    }
    /// The `IOSurfaceRef`'s address.
    fn id(&self) -> ObjId {
        ObjId(self.ptr.as_ptr().cast::<u8>().addr().to_u64())
    }
}

impl reims_vgpu_core::platform::os::surface::Surface for MetalSurface {
    fn retain_count(&self) -> i64 {
        // SAFETY: a live CF object this value holds a reference to.
        let n = unsafe { CFGetRetainCount(self.ptr.as_ptr().cast_const().cast()) };
        // CFIndex is 64-bit on every target this crate builds (D32).
        i64::try_from(n).unwrap_or(i64::MAX)
    }

    fn pixel_format(&self) -> u32 {
        // SAFETY: a live IOSurfaceRef.
        unsafe { IOSurfaceGetPixelFormat(self.as_iosurface_ref()) }
    }

    fn width(&self) -> u64 {
        // SAFETY: a live IOSurfaceRef.
        (unsafe { IOSurfaceGetWidth(self.as_iosurface_ref()) }).to_u64()
    }

    fn height(&self) -> u64 {
        // SAFETY: a live IOSurfaceRef.
        (unsafe { IOSurfaceGetHeight(self.as_iosurface_ref()) }).to_u64()
    }

    fn bytes_per_element(&self) -> u64 {
        // SAFETY: a live IOSurfaceRef.
        (unsafe { IOSurfaceGetBytesPerElement(self.as_iosurface_ref()) }).to_u64()
    }

    fn bytes_per_row(&self) -> u64 {
        // SAFETY: a live IOSurfaceRef.
        (unsafe { IOSurfaceGetBytesPerRow(self.as_iosurface_ref()) }).to_u64()
    }

    fn plane_count(&self) -> u64 {
        // SAFETY: a live IOSurfaceRef.
        (unsafe { IOSurfaceGetPlaneCount(self.as_iosurface_ref()) }).to_u64()
    }

    fn allows_pixel_size_casting(&self) -> bool {
        // SAFETY: a live IOSurfaceRef.
        (unsafe { IOSurfaceAllowsPixelSizeCasting(self.as_iosurface_ref()) }) != 0
    }

    /// SPI `IOSurfaceGetProtectionOptions`.
    fn protection_options(&self) -> u64 {
        // SAFETY: a live IOSurfaceRef.
        unsafe { IOSurfaceGetProtectionOptions(self.as_iosurface_ref()) }
    }

    fn lock(&self, options: u32) -> KernReturn {
        // SAFETY: a live IOSurfaceRef; a NULL seed is allowed.
        KernReturn(unsafe { IOSurfaceLock(self.as_iosurface_ref(), options, std::ptr::null_mut()) })
    }

    fn unlock(&self, options: u32) -> KernReturn {
        // SAFETY: a live IOSurfaceRef; a NULL seed is allowed.
        KernReturn(unsafe {
            IOSurfaceUnlock(self.as_iosurface_ref(), options, std::ptr::null_mut())
        })
    }

    fn base_address(&self) -> HostPtr {
        // SAFETY: a live IOSurfaceRef.
        HostPtr::from_ptr(unsafe { IOSurfaceGetBaseAddress(self.as_iosurface_ref()) })
    }

    fn alloc_size(&self) -> u64 {
        // SAFETY: a live IOSurfaceRef.
        (unsafe { IOSurfaceGetAllocSize(self.as_iosurface_ref()) }).to_u64()
    }

    /// The port's send right goes once the lookup has returned.
    fn lookup_by_port(&self) -> Option<Self> {
        // SAFETY: a live IOSurfaceRef.
        let port = unsafe { IOSurfaceCreateMachPort(self.as_iosurface_ref()) };
        // SAFETY: any port name is accepted; the result is +1 or NULL.
        let ptr = unsafe { IOSurfaceLookupFromMachPort(port) };
        // SAFETY: the send right IOSurfaceCreateMachPort handed over, in this task's space.
        let _ =
            unsafe { mach2::mach_port::mach_port_deallocate(mach2::traps::mach_task_self(), port) };
        // SAFETY: the lookup's +1 reference, handed over.
        NonNull::new(ptr).map(|p| unsafe { MetalSurface::from_retained(p) })
    }
}

/// A key: one of IOSurface's exported CFString constants.
type Key = &'static NSString;

/// An `NSNumber` as the dictionary's value type.
fn any(n: &NSNumber) -> &AnyObject {
    n.as_ref()
}

/// `[NSMutableDictionary new]`.
fn new_dictionary() -> Retained<NSMutableDictionary<NSString, AnyObject>> {
    value("new", NSMutableDictionary::<NSString, AnyObject>::new)
}

/// `[NSMutableArray new]`.
fn new_array() -> Retained<NSMutableArray<AnyObject>> {
    value("new", NSMutableArray::<AnyObject>::new)
}

/// `[dictionary setObject:object forKey:key]`.
fn set(d: &NSMutableDictionary<NSString, AnyObject>, key: Key, object: &AnyObject) {
    value("setObject:forKey:", || d.insert(key, object));
}

/// `[array addObject:object]`.
fn add(a: &NSMutableArray<AnyObject>, object: &AnyObject) {
    value("addObject:", || a.addObject(object));
}

/// `[dictionary setObject:[NSNumber numberWithUnsignedLongLong:v] forKey:key]`.
fn set_u64(d: &NSMutableDictionary<NSString, AnyObject>, key: Key, v: u64) {
    let n = value("numberWithUnsignedLongLong:", || NSNumber::new_u64(v));
    set(d, key, any(&n));
}

/// `[dictionary setObject:[NSNumber numberWithUnsignedInt:v] forKey:key]`.
fn set_u32(d: &NSMutableDictionary<NSString, AnyObject>, key: Key, v: u32) {
    let n = value("numberWithUnsignedInt:", || NSNumber::new_u32(v));
    set(d, key, any(&n));
}

/// One component array: `[NSNumber numberWithUnsignedChar:…]` per component, in order.
fn component_array(values: impl Iterator<Item = u8>) -> Retained<NSMutableArray<AnyObject>> {
    let a = new_array();
    for v in values {
        let n = value("numberWithUnsignedChar:", || NSNumber::new_u8(v));
        add(&a, any(&n));
    }
    a
}

/// One plane's dictionary under `kIOSurfacePlaneInfo`, for a plane core has checked (component
/// count 0..=4). The element dimensions are already decoded.
fn plane_dictionary(
    plane: &SurfacePlaneSpec,
) -> Retained<NSMutableDictionary<NSString, AnyObject>> {
    let d = new_dictionary();
    // SAFETY (every key read below): IOSurface's exported CFString constants, live for the
    // process.
    unsafe {
        set_u64(&d, kIOSurfacePlaneBase, plane.base);
        set_u64(&d, kIOSurfacePlaneOffset, plane.offset);
        set_u64(&d, kIOSurfacePlaneWidth, plane.width);
        set_u64(&d, kIOSurfacePlaneHeight, plane.height);
        set_u64(&d, kIOSurfacePlaneSize, plane.size);
        set_u64(&d, kIOSurfacePlaneBytesPerRow, plane.bytes_per_row);
        set_u64(&d, kIOSurfacePlaneBytesPerElement, plane.bytes_per_element);
        set_u32(&d, kIOSurfacePlaneElementWidth, plane.element_width);
        set_u32(&d, kIOSurfacePlaneElementHeight, plane.element_height);
        set_u32(
            &d,
            kIOSurfacePlaneCompressionFootprint,
            plane.compression_footprint,
        );
        set_u32(&d, kIOSurfaceAddressFormat, plane.address_format);
        set_u32(&d, kIOSurfacePlaneCompressionType, plane.compression_type);
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnLeft,
            plane.extended_pixels_on_left,
        );
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnTop,
            plane.extended_pixels_on_top,
        );
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnRight,
            plane.extended_pixels_on_right,
        );
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnBottom,
            plane.extended_pixels_on_bottom,
        );
        // A plane with no component has none of the five component keys.
        if !plane.components.is_empty() {
            let c = &plane.components;
            // The five arrays filled index by index.
            let depths = component_array(c.iter().map(|x| x.bit_depth));
            let offsets = component_array(c.iter().map(|x| x.bit_offset));
            let names = component_array(c.iter().map(|x| x.name));
            let types = component_array(c.iter().map(|x| x.type_));
            let ranges = component_array(c.iter().map(|x| x.range));
            set(&d, kIOSurfacePlaneComponentBitDepths, depths.as_ref());
            set(&d, kIOSurfacePlaneComponentBitOffsets, offsets.as_ref());
            set(&d, kIOSurfacePlaneComponentNames, names.as_ref());
            set(&d, kIOSurfacePlaneComponentTypes, types.as_ref());
            set(&d, kIOSurfacePlaneComponentRanges, ranges.as_ref());
        }
    }
    d
}

/// `PGPhysicalMemoryRange_t[]` as the bytes of an `NSMutableData` (host-endian `u64` pairs).
fn range_bytes(ranges: &[PhysicalMemoryRange]) -> Vec<u8> {
    let mut v = Vec::with_capacity(ranges.len().saturating_mul(16));
    for r in ranges {
        v.extend_from_slice(&r.physical_address.to_ne_bytes());
        v.extend_from_slice(&r.physical_length.to_ne_bytes());
    }
    v
}

/// The dictionary `IOSurfaceCreate` takes for `spec`. Core's checks (a plane count of 8 or more,
/// a plane whose dictionary fails) and their logs come before, in core.
pub(crate) fn surface_dictionary(
    spec: &SurfaceSpec,
) -> Retained<NSMutableDictionary<NSString, AnyObject>> {
    let d = new_dictionary();
    // SAFETY (every key read below): IOSurface's exported CFString constants.
    unsafe {
        // The plane array first, when there are planes.
        if let Some(planes) = &spec.planes {
            let a = new_array();
            // One dictionary per plane, in order.
            for p in planes {
                add(&a, plane_dictionary(p).as_ref());
            }
            set(&d, kIOSurfacePlaneInfo, a.as_ref());
        }
        set_u32(&d, kIOSurfacePixelFormat, spec.pixel_format);
        set_u64(&d, kIOSurfaceOffset, spec.offset);
        set_u64(&d, kIOSurfaceAllocSize, spec.alloc_size);
        set_u64(&d, kIOSurfaceWidth, spec.width);
        set_u64(&d, kIOSurfaceHeight, spec.height);
        set_u64(&d, kIOSurfaceBytesPerElement, spec.bytes_per_element);
        set_u64(&d, kIOSurfaceBytesPerRow, spec.bytes_per_row);
        set_u32(&d, kIOSurfaceElementWidth, spec.element_width);
        set_u32(&d, kIOSurfaceElementHeight, spec.element_height);
        set_u32(&d, kIOSurfaceAddressFormat, spec.address_format);
        set_u32(&d, kIOSurfacePlaneCompressionType, spec.compression_type);
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnLeft,
            spec.extended_pixels_on_left,
        );
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnTop,
            spec.extended_pixels_on_top,
        );
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnRight,
            spec.extended_pixels_on_right,
        );
        set_u32(
            &d,
            kIOSurfacePlaneExtendedPixelsOnBottom,
            spec.extended_pixels_on_bottom,
        );
        set_u64(&d, kIOSurfaceProtectionOptions, spec.protection_options);
        // Only for a read-only mapping.
        if let Some(direction) = spec.client_direction {
            set_u32(&d, kIOSurfaceClientDirection, direction);
        }
        // Absent for a host-only surface.
        if let Some(ranges) = &spec.client_address_ranges {
            let bytes = range_bytes(ranges);
            let data = value("initWithBytes:length:", || {
                NSMutableData::with_bytes(&bytes)
            });
            set(&d, kIOSurfaceClientAddressRanges, data.as_ref());
        }
    }
    d
}

/// `@(v)` for the `unsigned int` fields: `numberWithUnsignedInt:`.
fn boxed_u32(v: u32) -> Retained<NSNumber> {
    value("numberWithUnsignedInt:", || NSNumber::new_u32(v))
}

/// `@(v)` for the bit-fields that promote to `int`: `numberWithInt:`.
fn boxed_i32(v: i32) -> Retained<NSNumber> {
    value("numberWithInt:", || NSNumber::new_i32(v))
}

/// One `@{…}` of `planeInfo`.
fn backing_plane_dictionary(p: &BackingPlaneSpec) -> Retained<NSDictionary<NSString, AnyObject>> {
    let values = [
        boxed_u32(p.width),
        boxed_u32(p.height),
        boxed_i32(p.bytes_per_element),
        boxed_i32(p.bytes_per_row),
        boxed_u32(p.base),
        boxed_u32(p.offset),
    ];
    // SAFETY: IOSurface's exported NSString constants.
    let keys: [&NSString; 6] = unsafe {
        [
            IOSurfacePropertyKeyPlaneWidth,
            IOSurfacePropertyKeyPlaneHeight,
            IOSurfacePropertyKeyPlaneBytesPerElement,
            IOSurfacePropertyKeyPlaneBytesPerRow,
            IOSurfacePropertyKeyPlaneBase,
            IOSurfacePropertyKeyPlaneOffset,
        ]
    };
    let objects: Vec<&AnyObject> = values.iter().map(|v| any(v)).collect();
    value("initWithObjects:forKeys:count:", || {
        NSDictionary::from_slices(&keys, &objects)
    })
}

/// The properties `initWithProperties:` takes.
pub(crate) fn backing_properties(
    spec: &BackingSurfaceSpec,
) -> Retained<NSDictionary<NSString, AnyObject>> {
    // `planeInfo`, one dictionary per plane, in order
    let plane_dicts: Vec<Retained<NSDictionary<NSString, AnyObject>>> =
        spec.planes.iter().map(backing_plane_dictionary).collect();
    let plane_refs: Vec<&AnyObject> = plane_dicts.iter().map(|d| d.as_ref()).collect();
    let plane_info: Retained<NSArray<AnyObject>> = value("initWithObjects:count:", || {
        NSArray::from_slice(&plane_refs)
    });
    let pixel_format = boxed_u32(spec.pixel_format);
    let width = boxed_u32(spec.width);
    let height = boxed_u32(spec.height);
    // SAFETY: IOSurface's exported NSString constants.
    let keys: [&NSString; 4] = unsafe {
        [
            IOSurfacePropertyKeyPixelFormat,
            IOSurfacePropertyKeyPlaneInfo,
            IOSurfacePropertyKeyWidth,
            IOSurfacePropertyKeyHeight,
        ]
    };
    let objects: [&AnyObject; 4] = [
        any(&pixel_format),
        plane_info.as_ref(),
        any(&width),
        any(&height),
    ];
    value("initWithObjects:forKeys:count:", || {
        NSDictionary::from_slices(&keys, &objects)
    })
}

/// `IOSurfaceCreate` and `[[IOSurface alloc] initWithProperties:]`.
#[derive(Debug, Default)]
pub struct MetalSurfaces;

impl MetalSurfaces {
    /// The process's IoSurfaces.
    pub fn new() -> MetalSurfaces {
        MetalSurfaces
    }

    /// The `-description` of the dictionary `IOSurfaceCreate` refused, for the debug line that
    /// names it: rebuilt from the same spec key by key, as `create` builds it.
    pub fn describe(&self, spec: &SurfaceSpec) -> String {
        use objc2::runtime::NSObjectProtocol;
        let dictionary = surface_dictionary(spec);
        let description: Retained<NSString> = value("description", || {
            // SAFETY: `-description` always returns an NSString (typed NSObject by objc2's
            // runtime trait).
            unsafe { Retained::cast_unchecked(dictionary.description()) }
        });
        crate::msg::string_text(&description)
    }
}

impl reims_vgpu_core::platform::os::surface::IoSurfaces<MetalPlatform> for MetalSurfaces {
    /// The dictionary's description (the inherent [`MetalSurfaces::describe`]).
    fn describe(&self, spec: &SurfaceSpec) -> String {
        MetalSurfaces::describe(self, spec)
    }

    /// `IOSurfaceCreate((CFDictionaryRef)dictionary)`: over the dictionary
    /// [`surface_dictionary`] builds. NULL is `None`, the caller's `kIOReturnBadArgument`.
    fn create(&self, spec: &SurfaceSpec) -> Option<<MetalPlatform as Platform>::Surface> {
        let dictionary = surface_dictionary(spec);
        // SAFETY: a live dictionary (toll-free bridged with CFDictionaryRef); the result is +1
        // or NULL.
        let ptr = unsafe { IOSurfaceCreate(&dictionary) };
        // SAFETY: IOSurfaceCreate's +1 reference, handed over.
        NonNull::new(ptr).map(|p| unsafe { MetalSurface::from_retained(p) })
    }

    /// `[[IOSurface alloc] initWithProperties:properties]`, not checked for nil. A raise there has
    /// no handler to reach in the contract (`Option`), so it is a death at the send.
    fn create_backing(
        &self,
        spec: &BackingSurfaceSpec,
    ) -> Option<<MetalPlatform as Platform>::Surface> {
        let properties = backing_properties(spec);
        let allocated = ManuallyDrop::new(value("alloc", IOSurface::alloc));
        // SAFETY (D34): the closure owns only a `ManuallyDrop` (no drop glue) and borrows the
        // properties; `initWithProperties:` consumes the allocation.
        let surface: Option<Retained<IOSurface>> = fatal_send("initWithProperties:", || {
            IOSurface::initWithProperties(ManuallyDrop::into_inner(allocated), &properties)
        });
        let raw = Retained::into_raw(surface?);
        // SAFETY: an `IOSurface *` is toll-free bridged with `IOSurfaceRef`; `into_raw` hands
        // over the init's +1 reference.
        NonNull::new(raw.cast::<IOSurfaceRef>()).map(|p| unsafe { MetalSurface::from_retained(p) })
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use objc2::runtime::AnyObject;
    use reims_vgpu_core::platform::os::surface::{IoSurfaces, Surface, SurfaceComponentSpec};
    use reims_vgpu_core::platform::os::vm::HostVm;

    const BGRA: u32 = u32::from_be_bytes(*b"BGRA");

    /// A 16x16 BGRA host-only surface spec (no planes, no client memory).
    fn host_only() -> SurfaceSpec {
        SurfaceSpec {
            pixel_format: BGRA,
            alloc_size: 16 * 64,
            width: 16,
            height: 16,
            bytes_per_element: 4,
            bytes_per_row: 64,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        }
    }

    /// The number at `key`. (Its `objCType` does not show the constructor: Foundation stores small
    /// values of every unsigned constructor as `q`.)
    fn number(d: &NSDictionary<NSString, AnyObject>, key: &NSString) -> u64 {
        let v = d.objectForKey(key).unwrap();
        v.downcast_ref::<NSNumber>()
            .unwrap()
            .unsignedLongLongValue()
    }

    /// An object known to be a dictionary of NSString keys (every one these tests build).
    fn as_dict(o: &AnyObject) -> &NSDictionary<NSString, AnyObject> {
        let d = o.downcast_ref::<NSDictionary>().unwrap();
        // SAFETY: the dictionaries under test are built with NSString keys.
        unsafe { &*std::ptr::from_ref(d).cast::<NSDictionary<NSString, AnyObject>>() }
    }

    fn has(d: &NSDictionary<NSString, AnyObject>, key: &NSString) -> bool {
        d.objectForKey(key).is_some()
    }

    mod surface_dictionary {
        use super::*;

        /// The surface-mapping-failure debug line's description is the dictionary's
        /// `-description`, keys by their CFString names.
        #[test]
        fn describe_is_the_dictionary_description() {
            let spec = host_only();
            let text = MetalSurfaces::new().describe(&spec);
            use objc2::runtime::NSObjectProtocol;
            let d = surface_dictionary(&spec);
            // SAFETY: `-description` returns an NSString.
            let want: Retained<NSString> = unsafe { Retained::cast_unchecked(d.description()) };
            assert_eq!(text, want.to_string());
            assert!(text.starts_with('{'), "{text}");
            assert!(
                text.contains("IOSurfacePixelFormat = 1111970369;"),
                "{text}"
            );
            assert!(text.contains("IOSurfaceWidth = 16;"), "{text}");
        }

        /// The surface keys and their values
        #[test]
        fn surface_keys_and_number_types() {
            let spec = SurfaceSpec {
                offset: 0x40,
                address_format: 3,
                compression_type: 4,
                extended_pixels_on_left: 5,
                extended_pixels_on_top: 6,
                extended_pixels_on_right: 7,
                extended_pixels_on_bottom: 8,
                protection_options: 0x1_0000_0001,
                element_width: 2,
                element_height: 1 << 3,
                ..host_only()
            };
            let d = surface_dictionary(&spec);
            // SAFETY: IOSurface's exported constants.
            unsafe {
                assert_eq!(number(&d, kIOSurfacePixelFormat), u64::from(BGRA));
                assert_eq!(number(&d, kIOSurfaceOffset), 0x40);
                assert_eq!(number(&d, kIOSurfaceAllocSize), 1024);
                assert_eq!(number(&d, kIOSurfaceWidth), 16);
                assert_eq!(number(&d, kIOSurfaceHeight), 16);
                assert_eq!(number(&d, kIOSurfaceBytesPerElement), 4);
                assert_eq!(number(&d, kIOSurfaceBytesPerRow), 64);
                assert_eq!(number(&d, kIOSurfaceElementWidth), 2);
                assert_eq!(number(&d, kIOSurfaceElementHeight), 8);
                assert_eq!(number(&d, kIOSurfaceAddressFormat), 3);
                assert_eq!(number(&d, kIOSurfacePlaneCompressionType), 4);
                assert_eq!(number(&d, kIOSurfacePlaneExtendedPixelsOnLeft), 5);
                assert_eq!(number(&d, kIOSurfacePlaneExtendedPixelsOnTop), 6);
                assert_eq!(number(&d, kIOSurfacePlaneExtendedPixelsOnRight), 7);
                assert_eq!(number(&d, kIOSurfacePlaneExtendedPixelsOnBottom), 8);
                assert_eq!(number(&d, kIOSurfaceProtectionOptions), 0x1_0000_0001);
                // A host-only, read-write surface with no planes: none of the optional keys.
                assert!(!has(&d, kIOSurfacePlaneInfo));
                assert!(!has(&d, kIOSurfaceClientDirection));
                assert!(!has(&d, kIOSurfaceClientAddressRanges));
                assert_eq!(d.count(), 16);
            }
        }

        /// The private keys link with IOSurface's values
        #[test]
        fn private_keys_link() {
            // SAFETY: IOSurface's exported constants.
            unsafe {
                assert_eq!(
                    kIOSurfaceClientAddressRanges.to_string(),
                    "IOSurfaceAddressRanges"
                );
                assert_eq!(kIOSurfaceClientDirection.to_string(), "IOSurfaceDirection");
                assert_eq!(
                    kIOSurfaceAddressFormat.to_string(),
                    "IOSurfaceAddressFormat"
                );
                assert_eq!(
                    kIOSurfaceProtectionOptions.to_string(),
                    "IOSurfaceProtectionOptions"
                );
                for k in [
                    kIOSurfacePlaneCompressionFootprint,
                    kIOSurfacePlaneCompressionType,
                    kIOSurfacePlaneExtendedPixelsOnBottom,
                    kIOSurfacePlaneExtendedPixelsOnLeft,
                    kIOSurfacePlaneExtendedPixelsOnRight,
                    kIOSurfacePlaneExtendedPixelsOnTop,
                ] {
                    assert!(!k.to_string().is_empty());
                }
            }
        }

        /// readOnly sets kIOSurfaceClientDirection = 2
        #[test]
        fn read_only_direction() {
            let spec = SurfaceSpec {
                client_direction: Some(2),
                ..host_only()
            };
            let d = surface_dictionary(&spec);
            // SAFETY: IOSurface's exported constant.
            assert_eq!(unsafe { number(&d, kIOSurfaceClientDirection) }, 2);
        }

        /// The client ranges are NSData of {address, length} pairs, empty included
        #[test]
        fn client_address_ranges_data() {
            let ranges = vec![
                PhysicalMemoryRange {
                    physical_address: 0x1_2345_c000,
                    physical_length: 0x4000,
                },
                PhysicalMemoryRange {
                    physical_address: 0x2_0000_0000,
                    physical_length: 0x4000,
                },
            ];
            for rs in [ranges, Vec::new()] {
                let spec = SurfaceSpec {
                    client_address_ranges: Some(rs.clone()),
                    ..host_only()
                };
                let d = surface_dictionary(&spec);
                // SAFETY: IOSurface's exported constant.
                let v = d
                    .objectForKey(unsafe { kIOSurfaceClientAddressRanges })
                    .unwrap();
                let data = v.downcast_ref::<NSMutableData>().unwrap();
                assert_eq!(data.to_vec(), range_bytes(&rs));
                assert_eq!(data.length(), rs.len() * 16);
            }
            assert_eq!(
                range_bytes(&[PhysicalMemoryRange {
                    physical_address: 1,
                    physical_length: 2
                }]),
                [1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
            );
        }
    }

    mod plane_dictionary {
        use super::*;

        fn plane(components: usize) -> SurfacePlaneSpec {
            SurfacePlaneSpec {
                base: 1,
                offset: 2,
                width: 3,
                height: 4,
                size: 5,
                bytes_per_row: 6,
                bytes_per_element: 7,
                element_width: 8,
                element_height: 9,
                compression_footprint: 10,
                address_format: 11,
                compression_type: 12,
                extended_pixels_on_left: 13,
                extended_pixels_on_top: 14,
                extended_pixels_on_right: 15,
                extended_pixels_on_bottom: 16,
                components: (0..components)
                    .map(|i| {
                        let i = u8::try_from(i).unwrap();
                        SurfaceComponentSpec {
                            bit_depth: 8 + i,
                            bit_offset: 8 * i,
                            name: 1 + i,
                            type_: 2 + i,
                            range: 3 + i,
                        }
                    })
                    .collect(),
            }
        }

        /// A plane's keys and number types, in order, under kIOSurfacePlaneInfo
        #[test]
        fn plane_keys() {
            let spec = SurfaceSpec {
                planes: Some(vec![plane(0), plane(4)]),
                ..host_only()
            };
            let d = surface_dictionary(&spec);
            // SAFETY: IOSurface's exported constants.
            unsafe {
                let planes = d.objectForKey(kIOSurfacePlaneInfo).unwrap();
                let planes = planes.downcast_ref::<NSMutableArray<AnyObject>>().unwrap();
                assert_eq!(planes.count(), 2);
                let p0 = planes.objectAtIndex(0);
                let p0 = as_dict(&p0);
                let expect: [(Key, u64); 16] = [
                    (kIOSurfacePlaneBase, 1),
                    (kIOSurfacePlaneOffset, 2),
                    (kIOSurfacePlaneWidth, 3),
                    (kIOSurfacePlaneHeight, 4),
                    (kIOSurfacePlaneSize, 5),
                    (kIOSurfacePlaneBytesPerRow, 6),
                    (kIOSurfacePlaneBytesPerElement, 7),
                    (kIOSurfacePlaneElementWidth, 8),
                    (kIOSurfacePlaneElementHeight, 9),
                    (kIOSurfacePlaneCompressionFootprint, 10),
                    (kIOSurfaceAddressFormat, 11),
                    (kIOSurfacePlaneCompressionType, 12),
                    (kIOSurfacePlaneExtendedPixelsOnLeft, 13),
                    (kIOSurfacePlaneExtendedPixelsOnTop, 14),
                    (kIOSurfacePlaneExtendedPixelsOnRight, 15),
                    (kIOSurfacePlaneExtendedPixelsOnBottom, 16),
                ];
                for (k, v) in expect {
                    assert_eq!(number(p0, k), v, "{k}");
                }
                // No components: none of the five arrays.
                assert_eq!(p0.count(), 16);
                assert!(!has(p0, kIOSurfacePlaneComponentBitDepths));

                let p1 = planes.objectAtIndex(1);
                let p1 = as_dict(&p1);
                assert_eq!(p1.count(), 21);
                let arrays: [(Key, [u8; 4]); 5] = [
                    (kIOSurfacePlaneComponentBitDepths, [8, 9, 10, 11]),
                    (kIOSurfacePlaneComponentBitOffsets, [0, 8, 16, 24]),
                    (kIOSurfacePlaneComponentNames, [1, 2, 3, 4]),
                    (kIOSurfacePlaneComponentTypes, [2, 3, 4, 5]),
                    (kIOSurfacePlaneComponentRanges, [3, 4, 5, 6]),
                ];
                for (k, want) in arrays {
                    let a = p1.objectForKey(k).unwrap();
                    let a = a.downcast_ref::<NSMutableArray<AnyObject>>().unwrap();
                    assert_eq!(a.count(), 4);
                    for (i, w) in want.iter().enumerate() {
                        let n = a.objectAtIndex(i);
                        let n = n.downcast_ref::<NSNumber>().unwrap();
                        assert_eq!(n.unsignedCharValue(), *w);
                    }
                }
            }
        }
    }

    mod create {
        use super::*;

        /// IOSurfaceCreate gives one reference and the getters read the surface
        #[test]
        fn host_only_surface_and_getters() {
            let s = MetalSurfaces::new().create(&host_only()).unwrap();
            assert_eq!(s.retain_count(), 1);
            assert_eq!(s.pixel_format(), BGRA);
            assert_eq!(s.width(), 16);
            assert_eq!(s.height(), 16);
            assert_eq!(s.bytes_per_element(), 4);
            assert_eq!(s.bytes_per_row(), 64);
            assert_eq!(s.plane_count(), 0);
            assert!(s.alloc_size() >= 1024);
            assert_eq!(s.protection_options(), 0);
            let _ = s.allows_pixel_size_casting();
            assert_eq!(s.id().0, s.ptr.as_ptr().addr().to_u64());
        }

        /// retained() is one CFRetain, drop one CFRelease (the count is guest-visible)
        #[test]
        fn retain_and_release_count() {
            let s = MetalSurfaces::new().create(&host_only()).unwrap();
            let t = s.clone_ref();
            assert_eq!(s.id(), t.id());
            assert_eq!(s.retain_count(), 2);
            drop(t);
            assert_eq!(s.retain_count(), 1);
        }

        /// lookup_by_port is another object over the same surface, with a retain count of its
        /// own: what holds it never counts on the original
        #[test]
        fn a_port_lookup_is_another_object() {
            let s = MetalSurfaces::new().create(&host_only()).unwrap();
            let t = s.lookup_by_port().unwrap();
            assert_ne!(s.id(), t.id());
            let held = t.clone_ref();
            assert_eq!((s.retain_count(), t.retain_count()), (1, 2));
            assert_eq!(s.base_address(), t.base_address());
            assert_eq!(t.alloc_size(), s.alloc_size());
            drop((t, held));
            assert_eq!(s.retain_count(), 1);
        }

        /// IOSurfaceLock/Unlock and the test pattern writes through the base address
        #[test]
        fn lock_base_address_unlock() {
            let s = MetalSurfaces::new().create(&host_only()).unwrap();
            assert_eq!(s.lock(0), KernReturn::SUCCESS);
            let base = s.base_address();
            assert!(!base.is_null());
            // The test pattern on the first word. SAFETY: the locked surface's own memory, at least
            // alloc_size bytes.
            unsafe { base.as_ptr::<u64>().write_volatile(0x1122) };
            assert_eq!(unsafe { base.as_ptr::<u64>().read_volatile() }, 0x1122);
            assert_eq!(s.unlock(0), KernReturn::SUCCESS);
        }

        /// A dictionary IOSurface refuses is NULL, None (the caller's BadArgument)
        #[test]
        fn refused_dictionary_is_none() {
            let spec = SurfaceSpec {
                width: u64::MAX,
                height: u64::MAX,
                ..host_only()
            };
            assert!(MetalSurfaces::new().create(&spec).is_none());
        }

        /// A surface over client memory (kIOSurfaceClientAddressRanges) aliases it
        #[test]
        fn client_memory_surface_aliases_the_pages() {
            let vm = crate::os::vm::MetalVm::new();
            let page = vm.page_size();
            let mem = vm.allocate_anywhere(page).unwrap();
            let spec = SurfaceSpec {
                alloc_size: page,
                width: 16,
                height: 16,
                client_address_ranges: Some(vec![PhysicalMemoryRange {
                    physical_address: mem.addr().to_u64(),
                    physical_length: page,
                }]),
                ..host_only()
            };
            let s = MetalSurfaces::new().create(&spec).unwrap();
            assert_eq!(s.retain_count(), 1);
            assert_eq!(s.lock(0), KernReturn::SUCCESS);
            let base = s.base_address();
            // SAFETY: the locked surface's memory, one page.
            unsafe { base.as_ptr::<u8>().write_volatile(0x5a) };
            assert_eq!(s.unlock(0), KernReturn::SUCCESS);
            // SAFETY: the test's own page.
            assert_eq!(unsafe { mem.as_ptr::<u8>().read_volatile() }, 0x5a);
            drop(s);
            vm.deallocate(mem, page);
        }
    }

    mod create_backing {
        use super::*;

        fn backing() -> BackingSurfaceSpec {
            BackingSurfaceSpec {
                pixel_format: BGRA,
                planes: vec![BackingPlaneSpec {
                    width: 16,
                    height: 8,
                    bytes_per_element: 4,
                    bytes_per_row: 64,
                    base: 0,
                    offset: 0,
                }],
                width: 16,
                height: 8,
            }
        }

        /// The properties literal's keys and values
        #[test]
        fn properties_literal() {
            let d = backing_properties(&backing());
            // SAFETY: IOSurface's exported constants.
            unsafe {
                assert_eq!(d.count(), 4);
                assert_eq!(number(&d, IOSurfacePropertyKeyPixelFormat), u64::from(BGRA));
                assert_eq!(number(&d, IOSurfacePropertyKeyWidth), 16);
                assert_eq!(number(&d, IOSurfacePropertyKeyHeight), 8);
                let planes = d.objectForKey(IOSurfacePropertyKeyPlaneInfo).unwrap();
                let planes = planes.downcast_ref::<NSArray<AnyObject>>().unwrap();
                assert_eq!(planes.count(), 1);
                let p = planes.objectAtIndex(0);
                let p = as_dict(&p);
                assert_eq!(p.count(), 6);
                assert_eq!(number(p, IOSurfacePropertyKeyPlaneWidth), 16);
                assert_eq!(number(p, IOSurfacePropertyKeyPlaneHeight), 8);
                assert_eq!(number(p, IOSurfacePropertyKeyPlaneBytesPerElement), 4);
                assert_eq!(number(p, IOSurfacePropertyKeyPlaneBytesPerRow), 64);
                assert_eq!(number(p, IOSurfacePropertyKeyPlaneBase), 0);
                assert_eq!(number(p, IOSurfacePropertyKeyPlaneOffset), 0);
            }
        }

        /// [[IOSurface alloc] initWithProperties:] gives one reference to a planar surface
        #[test]
        fn backing_surface() {
            let s = MetalSurfaces::new().create_backing(&backing()).unwrap();
            assert_eq!(s.retain_count(), 1);
            assert_eq!(s.pixel_format(), BGRA);
            assert_eq!(s.width(), 16);
            assert_eq!(s.height(), 8);
            assert_eq!(s.plane_count(), 1);
        }

        /// Properties IOSurface refuses are nil (not checked)
        #[test]
        fn refused_properties_are_nil() {
            let spec = BackingSurfaceSpec {
                width: u32::MAX,
                height: u32::MAX,
                planes: vec![BackingPlaneSpec {
                    width: u32::MAX,
                    height: u32::MAX,
                    bytes_per_element: 4,
                    bytes_per_row: 64,
                    base: 0,
                    offset: 0,
                }],
                ..backing()
            };
            assert!(MetalSurfaces::new().create_backing(&spec).is_none());
        }
    }
}
