//! Metal's resources and state objects: core's `GpuResource`, `Buffer`, `Texture`, `Heap`,
//! `SamplerState`, `DepthStencilState`, `ComputePipeline`, `RenderPipeline` and `Library`.
//!
//! Every method is one typed call through the message layer. A getter's raise ends the process at the call;
//! a raise inside a method with arguments, an action or a creator is the caller's to catch.
//! Creators return Metal's `+1` object (nil is `None`); a `+0` object result is retained
//! ([`Obj::from_retained`]).

use std::ptr::NonNull;

use objc2::runtime::ProtocolObject;
use objc2_metal::*;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::types::*;
use reims_vgpu_core::platform::values::SurfacePlane;
use reims_vgpu_core::platform::{HostRange, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use crate::msg::encode::ToMtl;
use crate::msg::{self, Policy};
use crate::os::iosurface::MetalSurface;
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::{
    BufferSpi, HeapSpi, LibrarySpi, ResourceIndexSpi, TextureSpi, UniqueIdentifierSpi,
};

/// A creator's `+1` result as a handle; nil is `None`.
pub(crate) fn created<T: objc2::Message>(made: Option<objc2::rc::Retained<T>>) -> Option<Obj<T>> {
    made.map(Obj::from_retained)
}

/// An `iosurface` getter's `+0` surface, with one retain, as core's one-reference `Surface`;
/// `None` for NULL. The retain is the only one the getter makes (the surface's retain count is
/// guest-visible).
fn surface_plus_one(surface: *mut objc2_io_surface::IOSurfaceRef) -> Option<MetalSurface> {
    let surface = NonNull::new(surface)?;
    // SAFETY: the getter's `+0` surface is live while its owner is.
    Some(unsafe { MetalSurface::retain(surface) })
}

// ---------------------------------------------------------------------------------------------
// Resources.
// ---------------------------------------------------------------------------------------------

/// `id<MTLBuffer>`.
pub type MetalBuffer = Obj<ProtocolObject<dyn MTLBuffer>>;

/// `id<MTLTexture>`.
pub type MetalTexture = Obj<ProtocolObject<dyn MTLTexture>>;

/// Core's `GpuResource` for a buffer and a texture alike: both classes answer its selectors.
macro_rules! gpu_resource {
    ($t:ty) => {
        impl reims_vgpu_core::platform::resource::GpuResource<MetalPlatform> for $t {
            /// A `+0` device, retained.
            fn device(&self) -> Option<<MetalPlatform as Platform>::Device> {
                let r = self.get();
                Some(Obj::from_retained(msg::get("device", || {
                    MTLResource::device(r)
                })))
            }

            fn resource_options(&self) -> u64 {
                let r = self.get();
                msg::get("resourceOptions", || MTLResource::resourceOptions(r))
                    .0
                    .to_u64()
            }

            fn set_label(&self, label: &str) -> RaiseResult<()> {
                let (r, label) = (self.get(), msg::new_string(label));
                msg::send("setLabel:", Policy::Caught, || {
                    MTLResource::setLabel(r, Some(&label))
                })
            }

            /// The state before the change.
            fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
                let (r, state) = (self.get(), MTLPurgeableState(state.to_usize()));
                msg::send("setPurgeableState:", Policy::Caught, || {
                    MTLResource::setPurgeableState(r, state)
                })
                .map(|s| s.0.to_u64())
            }

            fn storage_mode(&self) -> u64 {
                let r = self.get();
                msg::get("storageMode", || MTLResource::storageMode(r))
                    .0
                    .to_u64()
            }
        }
    };
}

gpu_resource!(MetalBuffer);
gpu_resource!(MetalTexture);

impl reims_vgpu_core::platform::resource::Buffer<MetalPlatform> for MetalBuffer {
    /// NULL for a private buffer.
    fn contents(&self) -> HostPtr {
        let b = self.get();
        HostPtr::from_ptr(msg::get("contents", || BufferSpi::contents(b)))
    }

    /// The `BOOL` Metal answers is not read.
    fn detach_backing(&self) -> RaiseResult<()> {
        let b = self.get();
        msg::send("detachBacking", Policy::Caught, || {
            BufferSpi::detachBacking(b)
        })
        .map(drop)
    }

    fn did_modify_range(&self, on: OnFault, range: NsRange) -> RaiseResult<()> {
        let (b, range) = (self.get(), range.mtl());
        msg::send("didModifyRange:", Policy::On(on), || {
            b.didModifyRange(range)
        })
    }

    fn gpu_address(&self) -> u64 {
        let b = self.get();
        msg::get("gpuAddress", || b.gpuAddress())
    }

    fn iosurface(&self) -> Option<<MetalPlatform as Platform>::Surface> {
        let b = self.get();
        surface_plus_one(msg::get("iosurface", || BufferSpi::iosurface(b)))
    }

    fn length(&self) -> u64 {
        let b = self.get();
        msg::get("length", || b.length()).to_u64()
    }

    fn new_linear_texture(
        &self,
        descriptor: &<MetalPlatform as Platform>::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let b = self.get();
        msg::send(
            "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
            Policy::Caught,
            || {
                b.newLinearTextureWithDescriptor_offset_bytesPerRow_bytesPerImage(
                    descriptor,
                    offset,
                    bytes_per_row,
                    bytes_per_image,
                )
            },
        )
        .map(created)
    }

    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &<MetalPlatform as Platform>::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let b = self.get();
        let (offset, bytes_per_row) = (offset.to_usize(), bytes_per_row.to_usize());
        msg::send(
            "newTextureWithDescriptor:offset:bytesPerRow:",
            Policy::On(on),
            || b.newTextureWithDescriptor_offset_bytesPerRow(descriptor, offset, bytes_per_row),
        )
        .map(created)
    }

    /// Metal's `newTextureWithDescriptor:offset:bytesPerRow:` at the plane's offset and rows:
    /// Metal makes a texture over a surface's plane only over an `IOSurfaceRef`, and no public
    /// call makes one over a buffer's bytes. A format Metal makes no texture of over a buffer (a
    /// two-plane YCbCr or XR10 one among them) is that creator's nil, whatever the facts; any
    /// other is made as over the buffer, the facts unread.
    fn new_texture_over_surface_plane(
        &self,
        on: OnFault,
        descriptor: &<MetalPlatform as Platform>::TextureDescriptor,
        plane: SurfacePlane,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        self.new_texture(on, descriptor, plane.offset, plane.bytes_per_row)
    }

    fn replace_backing(
        &self,
        ranges: &<MetalPlatform as Platform>::AddressRangeArray,
        read_only: bool,
    ) -> RaiseResult<bool> {
        let b = self.get();
        msg::send("replaceBackingWithRanges:readOnly:", Policy::Caught, || {
            BufferSpi::replaceBackingWithRanges_readOnly(b, ranges, read_only)
        })
    }
}

impl reims_vgpu_core::platform::resource::Texture<MetalPlatform> for MetalTexture {
    fn array_length(&self) -> u64 {
        let t = self.get();
        msg::get("arrayLength", || t.arrayLength()).to_u64()
    }

    fn depth(&self) -> u64 {
        let t = self.get();
        msg::get("depth", || t.depth()).to_u64()
    }

    /// `MTLResourceID`'s one word, `_impl`.
    fn gpu_resource_id(&self) -> u64 {
        let t = self.get();
        msg::get("gpuResourceID", || t.gpuResourceID()).to_raw()
    }

    fn height(&self) -> u64 {
        let t = self.get();
        msg::get("height", || t.height()).to_u64()
    }

    fn iosurface(&self) -> Option<<MetalPlatform as Platform>::Surface> {
        let t = self.get();
        surface_plus_one(msg::get("iosurface", || TextureSpi::iosurface(t)))
    }

    fn is_shareable(&self) -> bool {
        let t = self.get();
        msg::get("isShareable", || t.isShareable())
    }

    fn mipmap_level_count(&self) -> u64 {
        let t = self.get();
        msg::get("mipmapLevelCount", || t.mipmapLevelCount()).to_u64()
    }

    fn new_shared_texture_handle(
        &self,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::SharedTextureHandle>> {
        let t = self.get();
        msg::send("newSharedTextureHandle", Policy::Caught, || {
            t.newSharedTextureHandle()
        })
        .map(created)
    }

    fn new_texture_view(
        &self,
        pixel_format: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let (t, format) = (self.get(), MTLPixelFormat(pixel_format.to_usize()));
        msg::send("newTextureViewWithPixelFormat:", Policy::Caught, || {
            t.newTextureViewWithPixelFormat(format)
        })
        .map(created)
    }

    fn new_texture_view_at_index(
        &self,
        pixel_format: u64,
        resource_index: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let t = self.get();
        msg::send(
            "newTextureViewWithPixelFormat:resourceIndex:",
            Policy::Caught,
            || t.newTextureViewWithPixelFormat_resourceIndex(pixel_format, resource_index),
        )
        .map(created)
    }

    fn new_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let t = self.get();
        let format = MTLPixelFormat(pixel_format.to_usize());
        let ty = MTLTextureType(texture_type.to_usize());
        let (levels, slices) = (level_range.mtl(), slice_range.mtl());
        // SAFETY: raw guest values go to Metal unvalidated; Metal checks the ranges itself.
        msg::send(
            "newTextureViewWithPixelFormat:textureType:levels:slices:",
            Policy::Caught,
            || unsafe {
                t.newTextureViewWithPixelFormat_textureType_levels_slices(
                    format, ty, levels, slices,
                )
            },
        )
        .map(created)
    }

    fn new_swizzled_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
        swizzle: Swizzle,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let t = self.get();
        let format = MTLPixelFormat(pixel_format.to_usize());
        let ty = MTLTextureType(texture_type.to_usize());
        let (levels, slices, swizzle) = (level_range.mtl(), slice_range.mtl(), swizzle.mtl());
        // SAFETY: as above.
        msg::send(
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
            Policy::Caught,
            || unsafe {
                t.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle(
                    format, ty, levels, slices, swizzle,
                )
            },
        )
        .map(created)
    }

    fn new_swizzled_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        swizzle: Swizzle,
        resource_index: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let t = self.get();
        let (levels, slices, swizzle) = (levels.mtl(), slices.mtl(), swizzle.mtl());
        msg::send(
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
            Policy::Caught,
            || {
                t.newTextureViewWithPixelFormat_textureType_levels_slices_swizzle_resourceIndex(
                    pixel_format,
                    texture_type,
                    levels,
                    slices,
                    swizzle,
                    resource_index,
                )
            },
        )
        .map(created)
    }

    fn new_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        resource_index: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let t = self.get();
        let (levels, slices) = (levels.mtl(), slices.mtl());
        msg::send(
            "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:",
            Policy::Caught,
            || {
                t.newTextureViewWithPixelFormat_textureType_levels_slices_resourceIndex(
                    pixel_format,
                    texture_type,
                    levels,
                    slices,
                    resource_index,
                )
            },
        )
        .map(created)
    }

    fn pixel_format(&self) -> u64 {
        let t = self.get();
        msg::get("pixelFormat", || t.pixelFormat()).0.to_u64()
    }

    fn protection_options(&self) -> u64 {
        let t = self.get();
        msg::get("protectionOptions", || t.protectionOptions())
    }

    /// Metal copies `pixel_bytes` during the call.
    fn replace_region(
        &self,
        region: MtlRegion,
        level: u64,
        slice: u64,
        pixel_bytes: HostRange<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<()> {
        let t = self.get();
        let bytes = pixel_bytes.ptr().as_ptr::<std::ffi::c_void>().cast_const();
        let region = region.mtl();
        // SAFETY: `HostRange::new`'s contract covers the region's bytes for the call.
        msg::send(
            "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:",
            Policy::Caught,
            || unsafe {
                TextureSpi::replaceRegion_mipmapLevel_slice_withBytes_bytesPerRow_bytesPerImage(
                    t,
                    region,
                    level,
                    slice,
                    bytes,
                    bytes_per_row,
                    bytes_per_image,
                )
            },
        )
    }

    fn resource_index(&self) -> u64 {
        let t = self.get();
        msg::get("resourceIndex", || TextureSpi::resourceIndex(t))
    }

    fn sample_count(&self) -> u64 {
        let t = self.get();
        msg::get("sampleCount", || t.sampleCount()).to_u64()
    }

    fn texture_type(&self) -> u64 {
        let t = self.get();
        msg::get("textureType", || t.textureType()).0.to_u64()
    }

    fn unique_identifier(&self) -> u64 {
        let t = self.get();
        msg::get("uniqueIdentifier", || TextureSpi::uniqueIdentifier(t))
    }

    fn usage(&self) -> u64 {
        let t = self.get();
        msg::get("usage", || t.usage()).0.to_u64()
    }

    fn width(&self) -> u64 {
        let t = self.get();
        msg::get("width", || t.width()).to_u64()
    }
}

/// `id<MTLHeap>`.
pub type MetalHeap = Obj<ProtocolObject<dyn MTLHeap>>;

impl reims_vgpu_core::platform::resource::Heap<MetalPlatform> for MetalHeap {
    /// The `BOOL` Metal answers is not read.
    fn detach_backing(&self) -> RaiseResult<()> {
        let h = self.get();
        msg::send("detachBacking", Policy::Caught, || {
            HeapSpi::detachBacking(h)
        })
        .map(drop)
    }

    fn gpu_address(&self) -> u64 {
        let h = self.get();
        msg::get("gpuAddress", || HeapSpi::gpuAddress(h))
    }

    fn new_buffer_of_length(
        &self,
        length: u64,
        options: u64,
        offset: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Buffer>> {
        let h = self.get();
        let options = MTLResourceOptions(options.to_usize());
        let (length, offset) = (length.to_usize(), offset.to_usize());
        // SAFETY: raw guest values go to Metal unvalidated; Metal checks the offset itself.
        msg::send(
            "newBufferWithLength:options:offset:",
            Policy::Caught,
            || unsafe { h.newBufferWithLength_options_offset(length, options, offset) },
        )
        .map(created)
    }

    fn new_texture(
        &self,
        descriptor: &<MetalPlatform as Platform>::TextureDescriptor,
        offset: u64,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Texture>> {
        let (h, offset) = (self.get(), offset.to_usize());
        // SAFETY: as above.
        msg::send(
            "newTextureWithDescriptor:offset:",
            Policy::Caught,
            || unsafe { h.newTextureWithDescriptor_offset(descriptor, offset) },
        )
        .map(created)
    }

    fn replace_backing(
        &self,
        ranges: &<MetalPlatform as Platform>::AddressRangeArray,
        read_only: bool,
    ) -> RaiseResult<bool> {
        let h = self.get();
        msg::send("replaceBackingWithRanges:readOnly:", Policy::Caught, || {
            HeapSpi::replaceBackingWithRanges_readOnly(h, ranges, read_only)
        })
    }

    fn resource_options(&self) -> u64 {
        let h = self.get();
        msg::get("resourceOptions", || h.resourceOptions())
            .0
            .to_u64()
    }

    /// The state before the change.
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        let (h, state) = (self.get(), MTLPurgeableState(state.to_usize()));
        msg::send("setPurgeableState:", Policy::Caught, || {
            h.setPurgeableState(state)
        })
        .map(|s| s.0.to_u64())
    }

    fn storage_mode(&self) -> u64 {
        let h = self.get();
        msg::get("storageMode", || h.storageMode()).0.to_u64()
    }
}

// ---------------------------------------------------------------------------------------------
// State objects, functions and libraries.
// ---------------------------------------------------------------------------------------------

/// `id<MTLSamplerState>`.
pub type MetalSamplerState = Obj<ProtocolObject<dyn MTLSamplerState>>;

impl reims_vgpu_core::platform::resource::SamplerState for MetalSamplerState {
    fn gpu_resource_id(&self) -> u64 {
        let s = self.get();
        msg::get("gpuResourceID", || s.gpuResourceID()).to_raw()
    }

    fn resource_index(&self) -> u64 {
        let s = self.get();
        msg::get("resourceIndex", || s.resourceIndex())
    }
}

/// `id<MTLDepthStencilState>`.
pub type MetalDepthStencilState = Obj<ProtocolObject<dyn MTLDepthStencilState>>;

/// The host's depth-stencil state implements the getters as stubs, which the caller catches
/// where they raise. Measured on the M3 Max: at macOS 26.7 (25G229) both raise
/// `NSInvalidArgumentException` (unrecognized selector); at macOS 27.0.1 (26A434) `gpuResourceID`
/// answers 0 and `resourceIndex` still raises. Core's info reply reads both, so on either release
/// the guest's request raises.
impl reims_vgpu_core::platform::resource::DepthStencilState for MetalDepthStencilState {
    fn gpu_resource_id(&self) -> RaiseResult<u64> {
        let s = self.get();
        msg::send("gpuResourceID", Policy::Caught, || s.gpuResourceID()).map(|id| id.to_raw())
    }

    fn resource_index(&self) -> RaiseResult<u64> {
        let s = self.get();
        msg::send("resourceIndex", Policy::Caught, || s.resourceIndex())
    }
}

/// `id<MTLFence>`.
pub type MetalFence = Obj<ProtocolObject<dyn MTLFence>>;

/// `id<MTLLibrary>`.
pub type MetalLibrary = Obj<ProtocolObject<dyn MTLLibrary>>;

impl reims_vgpu_core::platform::resource::Library<MetalPlatform> for MetalLibrary {
    /// The names copied out of the `NSArray<NSString *>` in its order; nil is empty.
    fn function_names(&self) -> Vec<String> {
        let l = self.get();
        let names = msg::get("functionNames", || l.functionNames_or_nil());
        let Some(names) = names else {
            return Vec::new();
        };
        let count = msg::value("count", || names.count());
        (0..count)
            .map(|i| msg::string_text(&msg::value("objectAtIndex:", || names.objectAtIndex(i))))
            .collect()
    }

    fn new_function(
        &self,
        on: OnFault,
        function_name: &str,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Function>> {
        let (l, name) = (self.get(), msg::new_string(function_name));
        msg::send("newFunctionWithName:", Policy::On(on), || {
            l.newFunctionWithName(&name)
        })
        .map(created)
    }

    /// Metal's `newFunctionWithName:constantValues:error:`, each constant set by its index. A
    /// constant of a type no function constant takes, or an `NSError`, answers nil.
    fn new_function_with_constants(
        &self,
        _on: OnFault,
        function_name: &str,
        constants: &[reims_vgpu_core::platform::resource::FunctionConstant],
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Function>> {
        let (l, name) = (self.get(), msg::new_string(function_name));
        let Some(values) = constant_values(constants) else {
            return Ok(None);
        };
        msg::send(
            "newFunctionWithName:constantValues:error:",
            Policy::Caught,
            || {
                l.newFunctionWithName_constantValues_error(&name, &values)
                    .ok()
            },
        )
        .map(created)
    }

    /// Metal's `newFunctionWithDescriptor:error:` over a descriptor of the name, the options, the
    /// specialized name where given and, where there are any, the constants, each set by its
    /// index. A constant of a type no function constant takes, or an `NSError`, answers nil.
    fn new_function_with_options(
        &self,
        _on: OnFault,
        function_name: &str,
        constants: &[reims_vgpu_core::platform::resource::FunctionConstant],
        options: u64,
        specialized_name: Option<&str>,
    ) -> RaiseResult<Option<<MetalPlatform as Platform>::Function>> {
        let (l, name) = (self.get(), msg::new_string(function_name));
        let d = msg::get(
            "functionDescriptor",
            MTLFunctionDescriptor::functionDescriptor,
        );
        msg::get("setName:", || d.setName(Some(&name)));
        if !constants.is_empty() {
            let Some(values) = constant_values(constants) else {
                return Ok(None);
            };
            msg::get("setConstantValues:", || d.setConstantValues(Some(&values)));
        }
        msg::get("setOptions:", || {
            d.setOptions(MTLFunctionOptions(options.to_usize()))
        });
        if let Some(specialized) = specialized_name {
            let specialized = msg::new_string(specialized);
            msg::get("setSpecializedName:", || {
                d.setSpecializedName(Some(&specialized))
            });
        }
        msg::send("newFunctionWithDescriptor:error:", Policy::Caught, || {
            l.newFunctionWithDescriptor_error(&d).ok()
        })
        .map(created)
    }
}

/// `MTLFunctionConstantValues` of `constants`, each set by its index: `None` where one is of a
/// type no function constant takes.
fn constant_values(
    constants: &[reims_vgpu_core::platform::resource::FunctionConstant],
) -> Option<objc2::rc::Retained<MTLFunctionConstantValues>> {
    let values = msg::get("new", MTLFunctionConstantValues::new);
    for c in constants {
        let bytes = c.bytes()?;
        // SAFETY: `bytes` holds as many bytes as `data_type` names; Metal copies them.
        msg::get("setConstantValue:type:atIndex:", || unsafe {
            values.setConstantValue_type_atIndex(
                NonNull::new(bytes.as_ptr().cast_mut().cast()).unwrap_or(NonNull::dangling()),
                MTLDataType(c.data_type.to_usize()),
                c.index.to_usize(),
            );
        });
    }
    Some(values)
}

/// `id<MTLFunction>`.
pub type MetalFunction = Obj<ProtocolObject<dyn MTLFunction>>;

/// `id<MTLComputePipelineState>`.
pub type MetalComputePipeline = Obj<ProtocolObject<dyn MTLComputePipelineState>>;

impl reims_vgpu_core::platform::resource::ComputePipeline for MetalComputePipeline {
    fn gpu_resource_id(&self) -> u64 {
        let p = self.get();
        msg::get("gpuResourceID", || p.gpuResourceID()).to_raw()
    }

    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        let (p, dimensions) = (self.get(), imageblock_dimensions.mtl());
        // SAFETY: raw guest values go to Metal unvalidated.
        msg::send(
            "imageblockMemoryLengthForDimensions:",
            Policy::Caught,
            || unsafe { p.imageblockMemoryLengthForDimensions(dimensions) },
        )
        .map(ToU64::to_u64)
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        let p = self.get();
        msg::get("maxTotalThreadsPerThreadgroup", || {
            p.maxTotalThreadsPerThreadgroup()
        })
        .to_u64()
    }

    fn resource_index(&self) -> u64 {
        let p = self.get();
        msg::get("resourceIndex", || ResourceIndexSpi::resourceIndex(p))
    }

    fn static_threadgroup_memory_length(&self) -> u64 {
        let p = self.get();
        msg::get("staticThreadgroupMemoryLength", || {
            p.staticThreadgroupMemoryLength()
        })
        .to_u64()
    }

    fn support_indirect_command_buffers(&self) -> bool {
        let p = self.get();
        msg::get("supportIndirectCommandBuffers", || {
            p.supportIndirectCommandBuffers()
        })
    }

    fn thread_execution_width(&self) -> u64 {
        let p = self.get();
        msg::get("threadExecutionWidth", || p.threadExecutionWidth()).to_u64()
    }

    fn unique_identifier(&self) -> u64 {
        let p = self.get();
        msg::get("uniqueIdentifier", || {
            UniqueIdentifierSpi::uniqueIdentifier(p)
        })
    }
}

/// `id<MTLRenderPipelineState>`.
pub type MetalRenderPipeline = Obj<ProtocolObject<dyn MTLRenderPipelineState>>;

impl reims_vgpu_core::platform::resource::RenderPipeline for MetalRenderPipeline {
    fn gpu_resource_id(&self) -> u64 {
        let p = self.get();
        msg::get("gpuResourceID", || p.gpuResourceID()).to_raw()
    }

    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        let (p, dimensions) = (self.get(), imageblock_dimensions.mtl());
        msg::send(
            "imageblockMemoryLengthForDimensions:",
            Policy::Caught,
            || p.imageblockMemoryLengthForDimensions(dimensions),
        )
        .map(ToU64::to_u64)
    }

    fn imageblock_sample_length(&self) -> u64 {
        let p = self.get();
        msg::get("imageblockSampleLength", || p.imageblockSampleLength()).to_u64()
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        let p = self.get();
        msg::get("maxTotalThreadsPerThreadgroup", || {
            p.maxTotalThreadsPerThreadgroup()
        })
        .to_u64()
    }

    fn max_total_threads_per_object_threadgroup(&self) -> u64 {
        let p = self.get();
        msg::get("maxTotalThreadsPerObjectThreadgroup", || {
            p.maxTotalThreadsPerObjectThreadgroup()
        })
        .to_u64()
    }

    fn max_total_threads_per_mesh_threadgroup(&self) -> u64 {
        let p = self.get();
        msg::get("maxTotalThreadsPerMeshThreadgroup", || {
            p.maxTotalThreadsPerMeshThreadgroup()
        })
        .to_u64()
    }

    fn object_thread_execution_width(&self) -> u64 {
        let p = self.get();
        msg::get("objectThreadExecutionWidth", || {
            p.objectThreadExecutionWidth()
        })
        .to_u64()
    }

    fn mesh_thread_execution_width(&self) -> u64 {
        let p = self.get();
        msg::get("meshThreadExecutionWidth", || p.meshThreadExecutionWidth()).to_u64()
    }

    fn max_total_threadgroups_per_mesh_grid(&self) -> u64 {
        let p = self.get();
        msg::get("maxTotalThreadgroupsPerMeshGrid", || {
            p.maxTotalThreadgroupsPerMeshGrid()
        })
        .to_u64()
    }

    fn resource_index(&self) -> u64 {
        let p = self.get();
        msg::get("resourceIndex", || ResourceIndexSpi::resourceIndex(p))
    }

    fn support_indirect_command_buffers(&self) -> bool {
        let p = self.get();
        msg::get("supportIndirectCommandBuffers", || {
            p.supportIndirectCommandBuffers()
        })
    }

    fn threadgroup_size_matches_tile_size(&self) -> bool {
        let p = self.get();
        msg::get("threadgroupSizeMatchesTileSize", || {
            p.threadgroupSizeMatchesTileSize()
        })
    }

    fn unique_identifier(&self) -> u64 {
        let p = self.get();
        msg::get("uniqueIdentifier", || {
            UniqueIdentifierSpi::uniqueIdentifier(p)
        })
    }
}

/// `MTLSharedTextureHandle *`.
pub type MetalSharedTextureHandle = Obj<MTLSharedTextureHandle>;

/// Live Metal objects of every kind the families send to, for the unit tests of their files.
#[cfg(test)]
pub(crate) mod testkit {
    #![allow(missing_docs)]

    use std::ffi::CString;

    use objc2::AnyThread as _;
    use objc2::rc::Retained;
    use objc2::runtime::{AnyObject, Sel};
    use objc2_foundation::NSString;

    use reims_vgpu_core::platform::device::GpuDevice;

    use crate::gpu::command::{MetalCommandBuffer, MetalCommandQueue};
    use crate::gpu::desc::{MetalRenderPassDescriptor, MetalTextureDescriptor};
    use crate::gpu::device::MetalDevice;
    use crate::gpu::event::MetalSharedEvent;
    use crate::gpu::icb::MetalIcb;
    use crate::gpu::rate_map::MetalRateMap;
    use crate::gpu::rate_map::{MetalRateMapDescriptor, MetalRateMapLayerDescriptor};
    use crate::platform::MetalAddressRangeArray;
    use crate::platform::Obj;

    use super::*;

    /// `MTLResourceID` as the raw getters read it: its one word, `_impl`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    #[repr(C)]
    pub(crate) struct ResourceId(pub u64);

    // SAFETY: one `u64`, and the runtime's tag (`{MTLResourceID=Q}16@0:8`).
    unsafe impl objc2::encode::Encode for ResourceId {
        const ENCODING: objc2::encode::Encoding =
            objc2::encode::Encoding::Struct("MTLResourceID", &[u64::ENCODING]);
    }

    /// Any handle as the `id` the raw test sends take.
    pub(crate) fn obj<T: objc2::Message>(o: &Obj<T>) -> &AnyObject {
        // SAFETY: every `T: Message` is an Objective-C object; the reference lives as long as `o`.
        unsafe { &*Retained::as_ptr(o.as_retained()).cast::<AnyObject>() }
    }

    /// [`obj`] under the name the encoder tests use.
    pub(crate) fn id<T: objc2::Message>(o: &Obj<T>) -> &AnyObject {
        obj(o)
    }

    /// Metal Shading Language for the pipelines the tests need.
    const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct V { float4 p [[position]]; };
kernel void k(device uint *b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] = i; }
kernel void tk(ushort2 lid [[thread_position_in_threadgroup]]) {}
struct CP { float4 p [[attribute(0)]]; };
struct PI { patch_control_point<CP> cp; };
[[patch(triangle, 3)]] vertex V pv(PI pi [[stage_in]], float3 bc [[position_in_patch]]) {
    V o; o.p = float4(0, 0, 0, 1); return o;
}
vertex V v(uint i [[vertex_id]]) { V o; o.p = float4(0, 0, 0, 1); return o; }
fragment float4 f() { return float4(1); }
"#;

    pub(crate) fn dev() -> MetalDevice {
        crate::gpu::device::tests::dev()
    }

    pub(crate) fn queue(d: &MetalDevice) -> MetalCommandQueue {
        d.new_command_queue().unwrap()
    }

    pub(crate) fn command_buffer(q: &MetalCommandQueue) -> MetalCommandBuffer {
        Obj::from_retained(q.get().commandBuffer().unwrap())
    }

    pub(crate) fn library(d: &MetalDevice) -> MetalLibrary {
        let src = NSString::from_str(MSL);
        Obj::from_retained(
            d.get()
                .newLibraryWithSource_options_error(&src, None)
                .unwrap(),
        )
    }

    pub(crate) fn function(lib: &MetalLibrary, name: &str) -> MetalFunction {
        Obj::from_retained(
            lib.get()
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap(),
        )
    }

    pub(crate) fn buffer(d: &MetalDevice, len: u64) -> MetalBuffer {
        d.new_buffer_of_length(OnFault::Die, len, 0)
            .unwrap()
            .unwrap()
    }

    /// A 16x16 BGRA8 texture of `mode` that can be rendered to, sampled and written.
    pub(crate) fn texture(d: &MetalDevice, mode: MTLStorageMode) -> MetalTexture {
        let td = MTLTextureDescriptor::new();
        // SAFETY: plain property sets on a fresh descriptor.
        unsafe {
            td.setWidth(16);
            td.setHeight(16);
        }
        td.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        td.setStorageMode(mode);
        td.setUsage(
            MTLTextureUsage::RenderTarget
                | MTLTextureUsage::ShaderRead
                | MTLTextureUsage::ShaderWrite,
        );
        Obj::from_retained(d.get().newTextureWithDescriptor(&td).unwrap())
    }

    pub(crate) fn heap(d: &MetalDevice) -> MetalHeap {
        let hd = MTLHeapDescriptor::new();
        hd.setSize(1 << 20);
        hd.setStorageMode(MTLStorageMode::Private);
        Obj::from_retained(d.get().newHeapWithDescriptor(&hd).unwrap())
    }

    pub(crate) fn sampler(d: &MetalDevice) -> MetalSamplerState {
        let sd = MTLSamplerDescriptor::new();
        sd.setSupportArgumentBuffers(true);
        Obj::from_retained(d.get().newSamplerStateWithDescriptor(&sd).unwrap())
    }

    pub(crate) fn depth_stencil(d: &MetalDevice) -> MetalDepthStencilState {
        let dd = MTLDepthStencilDescriptor::new();
        Obj::from_retained(d.get().newDepthStencilStateWithDescriptor(&dd).unwrap())
    }

    pub(crate) fn compute_pipeline(d: &MetalDevice) -> MetalComputePipeline {
        let lib = library(d);
        let cd = MTLComputePipelineDescriptor::new();
        cd.setComputeFunction(Some(&function(&lib, "k")));
        cd.setSupportIndirectCommandBuffers(true);
        Obj::from_retained(
            d.get()
                .newComputePipelineStateWithDescriptor_options_reflection_error(
                    &cd,
                    MTLPipelineOption::None,
                    None,
                )
                .unwrap(),
        )
    }

    pub(crate) fn render_pipeline_descriptor(
        d: &MetalDevice,
    ) -> Retained<MTLRenderPipelineDescriptor> {
        let lib = library(d);
        let rd = MTLRenderPipelineDescriptor::new();
        rd.setVertexFunction(Some(&function(&lib, "v")));
        rd.setFragmentFunction(Some(&function(&lib, "f")));
        // SAFETY: index 0 of the colour attachments.
        let ca = unsafe { rd.colorAttachments().objectAtIndexedSubscript(0) };
        ca.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        rd.setSupportIndirectCommandBuffers(true);
        rd
    }

    pub(crate) fn render_pipeline(d: &MetalDevice) -> MetalRenderPipeline {
        let rd = render_pipeline_descriptor(d);
        Obj::from_retained(
            d.get()
                .newRenderPipelineStateWithDescriptor_error(&rd)
                .unwrap(),
        )
    }

    /// A tile pipeline for the 16x16 BGRA8 pass (`tk`).
    pub(crate) fn tile_pipeline(d: &MetalDevice) -> MetalRenderPipeline {
        let lib = library(d);
        let td = MTLTileRenderPipelineDescriptor::new();
        // SAFETY: a kernel function of this device.
        unsafe { td.setTileFunction(&function(&lib, "tk")) };
        // SAFETY: index 0 of the colour attachments.
        let ca = unsafe { td.colorAttachments().objectAtIndexedSubscript(0) };
        ca.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        td.setThreadgroupSizeMatchesTileSize(true);
        Obj::from_retained(
            d.get()
                .newRenderPipelineStateWithTileDescriptor_options_reflection_error(
                    &td,
                    MTLPipelineOption::None,
                    None,
                )
                .unwrap(),
        )
    }

    /// A tessellated pipeline of triangle patches of 3 control points (`pv`, `f`).
    pub(crate) fn patch_pipeline(d: &MetalDevice) -> MetalRenderPipeline {
        let lib = library(d);
        let rd = MTLRenderPipelineDescriptor::new();
        rd.setVertexFunction(Some(&function(&lib, "pv")));
        rd.setFragmentFunction(Some(&function(&lib, "f")));
        // SAFETY: index 0 of the colour attachments, the vertex attributes and layouts.
        unsafe {
            rd.colorAttachments()
                .objectAtIndexedSubscript(0)
                .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            let vd = MTLVertexDescriptor::vertexDescriptor();
            vd.attributes()
                .objectAtIndexedSubscript(0)
                .setFormat(MTLVertexFormat::Float4);
            let l = vd.layouts().objectAtIndexedSubscript(0);
            l.setStride(16);
            l.setStepFunction(MTLVertexStepFunction::PerPatchControlPoint);
            rd.setVertexDescriptor(Some(&vd));
            rd.setMaxTessellationFactor(16);
            rd.setTessellationPartitionMode(MTLTessellationPartitionMode::Pow2);
        }
        Obj::from_retained(
            d.get()
                .newRenderPipelineStateWithDescriptor_error(&rd)
                .unwrap(),
        )
    }

    pub(crate) fn rate_layer(w: usize, h: usize) -> Retained<MTLRasterizationRateLayerDescriptor> {
        let size = MTLSize {
            width: w,
            height: h,
            depth: 0,
        };
        // SAFETY: `+alloc` then `-initWithSampleCount:` with a small count.
        unsafe {
            MTLRasterizationRateLayerDescriptor::initWithSampleCount(
                MTLRasterizationRateLayerDescriptor::alloc(),
                size,
            )
        }
    }

    pub(crate) fn rate_map(d: &MetalDevice) -> MetalRateMap {
        let rd = MTLRasterizationRateMapDescriptor::new();
        let layer = rate_layer(2, 2);
        // SAFETY: the screen size and layer 0 of a fresh descriptor.
        unsafe {
            rd.setScreenSize(MTLSize {
                width: 64,
                height: 64,
                depth: 0,
            });
            rd.setLayer_atIndex(Some(&layer), 0);
        }
        Obj::from_retained(d.get().newRasterizationRateMapWithDescriptor(&rd).unwrap())
    }

    /// An indirect command buffer of draws (`render`) or dispatches, 4 commands, with one bind
    /// slot of each kind, inheriting nothing.
    pub(crate) fn icb(d: &MetalDevice, render: bool) -> MetalIcb {
        let id = MTLIndirectCommandBufferDescriptor::new();
        if render {
            id.setCommandTypes(
                MTLIndirectCommandType::Draw
                    | MTLIndirectCommandType::DrawIndexed
                    | MTLIndirectCommandType::DrawPatches
                    | MTLIndirectCommandType::DrawIndexedPatches,
            );
            id.setMaxVertexBufferBindCount(2);
            id.setMaxFragmentBufferBindCount(2);
        } else {
            id.setCommandTypes(
                MTLIndirectCommandType::ConcurrentDispatch
                    | MTLIndirectCommandType::ConcurrentDispatchThreads,
            );
            id.setMaxKernelBufferBindCount(2);
        }
        id.setInheritPipelineState(false);
        id.setInheritBuffers(false);
        Obj::from_retained(
            // SAFETY: a valid descriptor; 4 commands.
            unsafe {
                d.get()
                    .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                        &id,
                        4,
                        MTLResourceOptions::empty(),
                    )
            }
            .unwrap(),
        )
    }

    /// A render pass onto `t` that loads nothing and stores its colour.
    pub(crate) fn render_pass(t: &MetalTexture) -> Retained<MTLRenderPassDescriptor> {
        let rp = MTLRenderPassDescriptor::new();
        // SAFETY: index 0 of the colour attachments.
        let ca = unsafe { rp.colorAttachments().objectAtIndexedSubscript(0) };
        ca.setTexture(Some(t.get()));
        ca.setLoadAction(MTLLoadAction::Clear);
        ca.setStoreAction(MTLStoreAction::Store);
        rp
    }

    /// One of every object the traits take as an argument.
    pub(crate) struct Kit {
        pub(crate) device: MetalDevice,
        pub(crate) queue: MetalCommandQueue,
        pub(crate) buffer: MetalBuffer,
        pub(crate) texture: MetalTexture,
        pub(crate) sampler: MetalSamplerState,
        pub(crate) depth_stencil: MetalDepthStencilState,
        pub(crate) render_pipeline: MetalRenderPipeline,
        pub(crate) compute_pipeline: MetalComputePipeline,
        pub(crate) fence: MetalFence,
        pub(crate) heap: MetalHeap,
        pub(crate) rate_map: MetalRateMap,
        pub(crate) function: MetalFunction,
        pub(crate) texture_descriptor: MetalTextureDescriptor,
        pub(crate) ranges: MetalAddressRangeArray,
        pub(crate) render_pass: MetalRenderPassDescriptor,
        pub(crate) rate_map_descriptor: MetalRateMapDescriptor,
        pub(crate) rate_layer: MetalRateMapLayerDescriptor,
        pub(crate) shared_event: MetalSharedEvent,
        pub(crate) icb_render: MetalIcb,
        pub(crate) icb_compute: MetalIcb,
    }

    pub(crate) fn kit() -> Kit {
        let device = dev();
        let texture = texture(&device, MTLStorageMode::Private);
        let render_pass = Obj::from_retained(render_pass(&texture));
        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: plain property sets on a fresh descriptor.
        unsafe {
            texture_descriptor.setWidth(4);
            texture_descriptor.setHeight(4);
        }
        texture_descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.setStorageMode(MTLStorageMode::Private);
        let rate_map_descriptor = MTLRasterizationRateMapDescriptor::new();
        let lib = library(&device);
        Kit {
            queue: queue(&device),
            buffer: buffer(&device, 4096),
            sampler: sampler(&device),
            depth_stencil: depth_stencil(&device),
            render_pipeline: render_pipeline(&device),
            compute_pipeline: compute_pipeline(&device),
            fence: device.new_fence().unwrap().unwrap(),
            heap: heap(&device),
            rate_map: rate_map(&device),
            function: function(&lib, "k"),
            texture_descriptor: Obj::from_retained(texture_descriptor),
            ranges: crate::platform::tests::platform()
                .new_address_range_array(&[])
                .unwrap(),
            render_pass,
            rate_map_descriptor: Obj::from_retained(rate_map_descriptor),
            rate_layer: Obj::from_retained(rate_layer(2, 2)),
            shared_event: device.new_shared_event().unwrap(),
            icb_render: icb(&device, true),
            icb_compute: icb(&device, false),
            texture,
            device,
        }
    }

    /// Sends a zero-argument getter returning `R`, outside the code under test.
    pub(crate) fn raw<R: objc2::encode::EncodeReturn>(o: &AnyObject, sel: &str) -> R {
        let sel = Sel::register(&CString::new(sel).unwrap());
        // SAFETY: test-only; the caller names a getter of `o` that returns `R`.
        unsafe { objc2::runtime::MessageReceiver::send_message(o, sel, ()) }
    }

    pub(crate) fn raw_u64(o: &AnyObject, sel: &str) -> u64 {
        raw(o, sel)
    }

    pub(crate) fn raw_bool(o: &AnyObject, sel: &str) -> bool {
        raw::<objc2::runtime::Bool>(o, sel).as_bool()
    }

    /// A value a setter's getter can read back.
    pub(crate) trait Readback: Copy + PartialEq + std::fmt::Debug {
        fn read(o: &AnyObject, sel: &str) -> Self;
    }
    impl Readback for u64 {
        fn read(o: &AnyObject, sel: &str) -> u64 {
            raw(o, sel)
        }
    }
    impl Readback for u32 {
        fn read(o: &AnyObject, sel: &str) -> u32 {
            raw(o, sel)
        }
    }
    impl Readback for f32 {
        fn read(o: &AnyObject, sel: &str) -> f32 {
            raw(o, sel)
        }
    }
    impl Readback for f64 {
        fn read(o: &AnyObject, sel: &str) -> f64 {
            raw(o, sel)
        }
    }
    impl Readback for bool {
        fn read(o: &AnyObject, sel: &str) -> bool {
            raw_bool(o, sel)
        }
    }

    /// After `setter` (`setFooBar:`), reads `fooBar` (or `isFooBar`) back when the object has
    /// such a getter, and checks that it is `v`. Returns whether a getter was found.
    pub(crate) fn readback<T: Readback>(o: &AnyObject, setter: &str, v: T) -> bool {
        let base = setter
            .strip_prefix("set")
            .unwrap()
            .strip_suffix(':')
            .unwrap();
        let mut chars = base.chars();
        let first = chars.next().unwrap();
        let rest: String = chars.collect();
        let lower = format!("{}{rest}", first.to_ascii_lowercase());
        // Leading acronyms keep their case (`GPU…`, `LOD…`).
        let candidates = [lower, base.to_string(), format!("is{base}")];
        for g in candidates {
            let sel = Sel::register(&CString::new(g.clone()).unwrap());
            if o.class().responds_to(sel) {
                assert_eq!(T::read(o, &g), v, "{setter} read back through {g}");
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {

    use objc2_metal::{MTLOrigin, MTLRegion, MTLSize, MTLStorageMode};
    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::resource::*;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::resource::testkit::*;

    // BEGIN GENERATED TESTS

    /// Every GpuResource method on a buffer
    #[test]
    fn buffer_resource_sends() {
        let k = kit();
        let x = buffer(&k.device, 4096);
        let _ = x.device();
        assert_eq!(x.resource_options(), raw_u64(obj(&x), "resourceOptions"));
        let r = x.set_label("label");
        assert!(r.is_ok(), "set_label: {r:?}");
        let r = x.set_purgeable_state(1);
        assert!(r.is_ok(), "set_purgeable_state: {r:?}");
        assert_eq!(x.storage_mode(), raw_u64(obj(&x), "storageMode"));
    }

    /// Every GpuResource method on a texture
    #[test]
    fn texture_resource_sends() {
        let k = kit();
        let x = texture(&k.device, MTLStorageMode::Shared);
        let _ = x.device();
        assert_eq!(x.resource_options(), raw_u64(obj(&x), "resourceOptions"));
        let r = x.set_label("label");
        assert!(r.is_ok(), "set_label: {r:?}");
        let r = x.set_purgeable_state(1);
        assert!(r.is_ok(), "set_purgeable_state: {r:?}");
        assert_eq!(x.storage_mode(), raw_u64(obj(&x), "storageMode"));
    }

    /// Every Buffer method but iosurface (a shared buffer; linear textures from it)
    #[test]
    fn buffer_sends() {
        let k = kit();
        let x = buffer(&k.device, 4096);
        let _ = x.contents();
        let r = x.detach_backing();
        assert!(r.is_ok(), "detach_backing: {r:?}");
        let r = x.did_modify_range(OnFault::Return, NsRange::new(0, 1));
        assert!(r.is_ok(), "did_modify_range: {r:?}");
        assert_eq!(x.gpu_address(), raw_u64(obj(&x), "gpuAddress"));
        assert_eq!(x.length(), raw_u64(obj(&x), "length"));
        let r = x.new_linear_texture(&k.texture_descriptor, 0, 64, 1024);
        assert!(r.is_ok(), "new_linear_texture: {r:?}");
        let r = x.new_texture(OnFault::Return, &k.texture_descriptor, 0, 64);
        assert!(r.is_ok(), "new_texture: {r:?}");
        let r = x.replace_backing(&k.ranges, false);
        assert!(r.is_ok(), "replace_backing: {r:?}");
    }

    /// Every Texture method but iosurface and replace_region (a shared texture; its views)
    #[test]
    fn texture_sends() {
        let k = kit();
        let x = texture(&k.device, MTLStorageMode::Shared);
        assert_eq!(x.array_length(), raw_u64(obj(&x), "arrayLength"));
        assert_eq!(x.depth(), raw_u64(obj(&x), "depth"));
        assert_eq!(
            x.gpu_resource_id(),
            raw::<ResourceId>(obj(&x), "gpuResourceID").0
        );
        assert_eq!(x.height(), raw_u64(obj(&x), "height"));
        assert_eq!(x.is_shareable(), raw_bool(obj(&x), "isShareable"));
        assert_eq!(x.mipmap_level_count(), raw_u64(obj(&x), "mipmapLevelCount"));
        let r = x.new_shared_texture_handle();
        assert!(r.is_ok(), "new_shared_texture_handle: {r:?}");
        let r = x.new_texture_view(80);
        assert!(r.is_ok(), "new_texture_view: {r:?}");
        let r = x.new_texture_view_at_index(80, 0);
        assert!(r.is_ok(), "new_texture_view_at_index: {r:?}");
        let r = x.new_texture_subview(80, 2, NsRange::new(0, 1), NsRange::new(0, 1));
        assert!(r.is_ok(), "new_texture_subview: {r:?}");
        let r = x.new_swizzled_texture_subview(
            80,
            2,
            NsRange::new(0, 1),
            NsRange::new(0, 1),
            Swizzle {
                red: 2,
                green: 3,
                blue: 4,
                alpha: 5,
            },
        );
        assert!(r.is_ok(), "new_swizzled_texture_subview: {r:?}");
        let r = x.new_swizzled_texture_subview_at_index(
            80,
            2,
            NsRange::new(0, 1),
            NsRange::new(0, 1),
            Swizzle {
                red: 2,
                green: 3,
                blue: 4,
                alpha: 5,
            },
            0,
        );
        assert!(r.is_ok(), "new_swizzled_texture_subview_at_index: {r:?}");
        let r = x.new_texture_subview_at_index(80, 2, NsRange::new(0, 1), NsRange::new(0, 1), 0);
        assert!(r.is_ok(), "new_texture_subview_at_index: {r:?}");
        assert_eq!(x.pixel_format(), raw_u64(obj(&x), "pixelFormat"));
        assert_eq!(
            x.protection_options(),
            raw_u64(obj(&x), "protectionOptions")
        );
        assert_eq!(x.resource_index(), raw_u64(obj(&x), "resourceIndex"));
        assert_eq!(x.sample_count(), raw_u64(obj(&x), "sampleCount"));
        assert_eq!(x.texture_type(), raw_u64(obj(&x), "textureType"));
        assert_eq!(x.unique_identifier(), raw_u64(obj(&x), "uniqueIdentifier"));
        assert_eq!(x.usage(), raw_u64(obj(&x), "usage"));
        assert_eq!(x.width(), raw_u64(obj(&x), "width"));
    }

    /// Every Heap method
    #[test]
    fn heap_sends() {
        let k = kit();
        let x = heap(&k.device);
        let r = x.detach_backing();
        assert!(r.is_ok(), "detach_backing: {r:?}");
        assert_eq!(x.gpu_address(), raw_u64(obj(&x), "gpuAddress"));
        let r = x.new_buffer_of_length(16, 0, 0);
        assert!(r.is_ok(), "new_buffer_of_length: {r:?}");
        let r = x.new_texture(&k.texture_descriptor, 0);
        assert!(r.is_ok(), "new_texture: {r:?}");
        let r = x.replace_backing(&k.ranges, false);
        assert!(r.is_ok(), "replace_backing: {r:?}");
        assert_eq!(x.resource_options(), raw_u64(obj(&x), "resourceOptions"));
        let r = x.set_purgeable_state(1);
        assert!(r.is_ok(), "set_purgeable_state: {r:?}");
        assert_eq!(x.storage_mode(), raw_u64(obj(&x), "storageMode"));
    }

    /// Every SamplerState getter
    #[test]
    fn sampler_state_sends() {
        let k = kit();
        let x = k.sampler;
        assert_eq!(
            x.gpu_resource_id(),
            raw::<ResourceId>(obj(&x), "gpuResourceID").0
        );
        assert_eq!(x.resource_index(), raw_u64(obj(&x), "resourceIndex"));
    }

    /// Every ComputePipeline method
    #[test]
    fn compute_pipeline_sends() {
        let k = kit();
        let x = k.compute_pipeline;
        assert_eq!(
            x.gpu_resource_id(),
            raw::<ResourceId>(obj(&x), "gpuResourceID").0
        );
        let r = x.imageblock_memory_length(MtlSize::new(1, 1, 1));
        assert!(r.is_ok(), "imageblock_memory_length: {r:?}");
        assert_eq!(
            x.max_total_threads_per_threadgroup(),
            raw_u64(obj(&x), "maxTotalThreadsPerThreadgroup")
        );
        assert_eq!(x.resource_index(), raw_u64(obj(&x), "resourceIndex"));
        assert_eq!(
            x.static_threadgroup_memory_length(),
            raw_u64(obj(&x), "staticThreadgroupMemoryLength")
        );
        assert_eq!(
            x.support_indirect_command_buffers(),
            raw_bool(obj(&x), "supportIndirectCommandBuffers")
        );
        assert_eq!(
            x.thread_execution_width(),
            raw_u64(obj(&x), "threadExecutionWidth")
        );
        assert_eq!(x.unique_identifier(), raw_u64(obj(&x), "uniqueIdentifier"));
    }

    /// Every RenderPipeline method
    #[test]
    fn render_pipeline_sends() {
        let k = kit();
        let x = k.render_pipeline;
        assert_eq!(
            x.gpu_resource_id(),
            raw::<ResourceId>(obj(&x), "gpuResourceID").0
        );
        let r = x.imageblock_memory_length(MtlSize::new(1, 1, 1));
        assert!(r.is_ok(), "imageblock_memory_length: {r:?}");
        assert_eq!(
            x.imageblock_sample_length(),
            raw_u64(obj(&x), "imageblockSampleLength")
        );
        assert_eq!(
            x.max_total_threads_per_threadgroup(),
            raw_u64(obj(&x), "maxTotalThreadsPerThreadgroup")
        );
        assert_eq!(x.resource_index(), raw_u64(obj(&x), "resourceIndex"));
        assert_eq!(
            x.support_indirect_command_buffers(),
            raw_bool(obj(&x), "supportIndirectCommandBuffers")
        );
        assert_eq!(
            x.threadgroup_size_matches_tile_size(),
            raw_bool(obj(&x), "threadgroupSizeMatchesTileSize")
        );
        assert_eq!(x.unique_identifier(), raw_u64(obj(&x), "uniqueIdentifier"));
    }

    /// Library::function_names and new_function
    #[test]
    fn library_sends() {
        let k = kit();
        let x = library(&k.device);
        let _ = x.function_names();
        let r = x.new_function(OnFault::Return, "label");
        assert!(r.is_ok(), "new_function: {r:?}");
    }
    // END GENERATED TESTS

    /// The buffer getters agree with objc2-metal's (contents, length, gpuAddress, storageMode,
    /// resourceOptions), and didModifyRange: sends
    #[test]
    fn buffer_values() {
        let k = kit();
        let b = buffer(&k.device, 4096);
        assert_eq!(
            b.contents().as_ptr::<std::ffi::c_void>(),
            MTLBuffer::contents(b.get()).as_ptr()
        );
        assert_eq!(b.length(), 4096);
        assert_eq!(b.gpu_address(), b.get().gpuAddress());
        assert_eq!(b.storage_mode(), 0);
        // Default options: shared storage, default cache mode, tracked hazards (0x200).
        assert_eq!(b.resource_options(), 0x200);
        b.did_modify_range(OnFault::Return, NsRange::new(0, 16))
            .unwrap();
        b.set_label("reims-vgpu.buffer").unwrap();
        assert_eq!(b.get().label().unwrap().to_string(), "reims-vgpu.buffer");
        assert_eq!(GpuResource::device(&b).unwrap().id(), k.device.id());
    }

    /// setPurgeableState: returns the previous state (KeepCurrent reads it, then Volatile and
    /// NonVolatile), on a buffer and a heap
    #[test]
    fn purgeable_state_returns_the_previous_state() {
        let k = kit();
        // A private buffer (a shared one always reports NonVolatile, measured).
        let b = k
            .device
            .new_buffer_of_length(OnFault::Return, 65536, 0x20)
            .unwrap()
            .unwrap();
        assert_eq!(b.set_purgeable_state(1).unwrap(), 2);
        assert_eq!(b.set_purgeable_state(3).unwrap(), 2);
        assert_eq!(b.set_purgeable_state(2).unwrap(), 3);
        // A private heap reports NonVolatile whatever it was asked (measured).
        let h = heap(&k.device);
        assert_eq!(h.set_purgeable_state(1).unwrap(), 2);
    }

    /// Iosurface is None for a buffer and a texture that have no IOSurface (the NULL branch; the
    /// surface-backed one is retain_audit's)
    #[test]
    fn iosurface_is_none_without_a_surface() {
        let k = kit();
        assert!(buffer(&k.device, 64).iosurface().is_none());
        assert!(k.texture.iosurface().is_none());
    }

    /// replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage: copies the bytes into
    /// the region, which getBytes reads back
    #[test]
    fn replace_region_writes_the_texture() {
        let k = kit();
        let t = texture(&k.device, MTLStorageMode::Shared);
        let mut px = [7u8; 16 * 4];
        let region = MtlRegion::new(MtlOrigin::new(0, 2, 0), MtlSize::new(16, 1, 1));
        // SAFETY: `px` holds one row of 16 BGRA8 pixels (64 bytes) for the duration of the call.
        let pixels = unsafe { HostRange::new(HostPtr::from_ptr(px.as_mut_ptr()), 64) };
        t.replace_region(region, 0, 0, pixels, 64, 64).unwrap();
        let mut out = [0u8; 64];
        // SAFETY: 64 bytes of row 2, as written.
        unsafe {
            t.get().getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(out.as_mut_ptr().cast()).unwrap(),
                64,
                MTLRegion {
                    origin: MTLOrigin { x: 0, y: 2, z: 0 },
                    size: MTLSize {
                        width: 16,
                        height: 1,
                        depth: 1,
                    },
                },
                0,
            )
        };
        assert_eq!(out, px);
    }

    /// The texture getters agree with objc2-metal's; a view has the format asked for
    #[test]
    fn texture_values_and_views() {
        let k = kit();
        let t = &k.texture;
        assert_eq!((t.width(), t.height(), t.depth()), (16, 16, 1));
        assert_eq!(
            (t.array_length(), t.mipmap_level_count(), t.sample_count()),
            (1, 1, 1)
        );
        assert_eq!((t.pixel_format(), t.texture_type()), (80, 2));
        assert_eq!(t.usage(), 7);
        assert_eq!(t.storage_mode(), 2);
        assert!(!t.is_shareable());
        assert_eq!(
            t.gpu_resource_id(),
            raw::<ResourceId>(obj(t), "gpuResourceID").0
        );
        assert_ne!(t.gpu_resource_id(), 0);
        let v = t.new_texture_view(81).unwrap().unwrap();
        assert_eq!(v.pixel_format(), 81);
        let v = t
            .new_swizzled_texture_subview(
                80,
                2,
                NsRange::new(0, 1),
                NsRange::new(0, 1),
                Swizzle {
                    red: 5,
                    green: 4,
                    blue: 3,
                    alpha: 2,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(v.width(), 16);
    }

    /// functionNames lists the library's functions; newFunctionWithName: is nil for a name it lacks
    #[test]
    fn library_functions() {
        let k = kit();
        let lib = library(&k.device);
        let names = lib.function_names();
        assert_eq!(names.len(), 5, "{names:?}");
        for n in ["f", "k", "pv", "tk", "v"] {
            assert!(names.iter().any(|m| m == n), "{n} in {names:?}");
        }
        assert!(lib.new_function(OnFault::Return, "nope").unwrap().is_none());
        assert!(lib.new_function(OnFault::Return, "k").unwrap().is_some());
    }

    /// A function made with options is Metal's function of the descriptor's name and options,
    /// and a name the library does not hold is nil
    #[test]
    fn a_function_made_with_options_answers_them() {
        let k = kit();
        let lib = library(&k.device);
        let f = lib
            .new_function_with_options(OnFault::Return, "k", &[], 1, None)
            .unwrap()
            .unwrap();
        assert_eq!(f.get().options(), MTLFunctionOptions::CompileToBinary);
        assert_eq!(f.get().name().to_string(), "k");
        assert!(
            lib.new_function_with_options(OnFault::Return, "nope", &[], 1, None)
                .unwrap()
                .is_none()
        );
    }

    /// A function made with a specialized name answers that name, and a kernel that calls a
    /// visible function of that name links it as a private function; the same function made
    /// under its own name leaves the call unresolved, and the pipeline refused
    #[test]
    fn a_specialized_name_resolves_a_kernel_s_call() {
        use objc2_foundation::NSString;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::values::{FullComputePipelineDesc, LinkedFunctionsDesc};
        let k = kit();
        let source = |src: &str| -> MetalLibrary {
            Obj::from_retained(
                k.device
                    .get()
                    .newLibraryWithSource_options_error(&NSString::from_str(src), None)
                    .unwrap(),
            )
        };
        let caller = source(
            "[[visible]] int custom_fn(int x);\n\
             kernel void caller(device int *o [[buffer(0)]]) { o[0] = custom_fn(o[0]); }",
        );
        let callee = source("[[visible]] int impl_fn(int x) { return x + 1; }");
        let kernel = function(&caller, "caller");
        let pipeline = |linked: MetalFunction| {
            k.device
                .new_compute_pipeline_state_full(&FullComputePipelineDesc {
                    label: None,
                    function: &kernel,
                    threadgroup_size_is_multiple_of_execution_width: false,
                    max_total_threads_per_threadgroup: 0,
                    max_call_stack_depth: 1,
                    support_indirect_command_buffers: false,
                    stage_input: None,
                    linked: LinkedFunctionsDesc {
                        private_functions: vec![linked],
                        ..LinkedFunctionsDesc::default()
                    },
                })
        };
        let specialized = callee
            .new_function_with_options(OnFault::Return, "impl_fn", &[], 0, Some("custom_fn"))
            .unwrap()
            .unwrap();
        assert_eq!(specialized.get().name().to_string(), "custom_fn");
        assert!(matches!(pipeline(specialized), Ok(Some(_))));
        let plain = callee
            .new_function_with_options(OnFault::Return, "impl_fn", &[], 0, None)
            .unwrap()
            .unwrap();
        assert_eq!(plain.get().name().to_string(), "impl_fn");
        let refused = pipeline(plain);
        assert!(refused.is_err(), "{refused:?}");
    }

    /// The pipeline getters agree with objc2-metal's
    #[test]
    fn pipeline_values() {
        let k = kit();
        let c = &k.compute_pipeline;
        assert_eq!(
            c.max_total_threads_per_threadgroup(),
            u64::try_from(c.get().maxTotalThreadsPerThreadgroup()).unwrap()
        );
        assert_eq!(
            c.thread_execution_width(),
            u64::try_from(c.get().threadExecutionWidth()).unwrap()
        );
        assert!(c.support_indirect_command_buffers());
        assert_eq!(
            c.gpu_resource_id(),
            raw::<ResourceId>(obj(c), "gpuResourceID").0
        );
        assert_ne!(c.gpu_resource_id(), 0);
        let r = &k.render_pipeline;
        assert!(r.support_indirect_command_buffers());
        assert_eq!(
            r.gpu_resource_id(),
            raw::<ResourceId>(obj(r), "gpuResourceID").0
        );
        assert_ne!(r.gpu_resource_id(), 0);
        let s = &k.sampler;
        assert_eq!(
            s.gpu_resource_id(),
            raw::<ResourceId>(obj(s), "gpuResourceID").0
        );
        assert_ne!(s.gpu_resource_id(), 0);
    }

    /// Which of the host's depth-stencil getters raise, `gpuResourceID`'s then `resourceIndex`'s:
    /// on macOS 27.0.1, the newest release measured, `resourceIndex` alone, `gpuResourceID`
    /// answering 0.
    const RAISING_GETTERS: (bool, bool) = (false, true);

    /// The host's AGX depth-stencil state implements gpuResourceID and resourceIndex as stubs. One
    /// that raises NSInvalidArgumentException (unrecognized selector) would die at its send were
    /// nothing to catch it; only the info decode sends them, under a handler (`Caught`;
    /// architecture 6.3). One that does not raise answers 0 ([`RAISING_GETTERS`])
    #[test]
    fn depth_stencil_state_getters_raise() {
        let k = kit();
        let (id_raises, index_raises) = RAISING_GETTERS;
        for (sel, f, raises) in [
            (
                "gpuResourceID",
                DepthStencilState::gpu_resource_id
                    as fn(&MetalDepthStencilState) -> RaiseResult<u64>,
                id_raises,
            ),
            (
                "resourceIndex",
                DepthStencilState::resource_index,
                index_raises,
            ),
        ] {
            match f(&k.depth_stencil) {
                Err(t) => {
                    assert!(raises, "{sel} raised");
                    let site = t.site();
                    assert_eq!(site.kind, DeathKind::UncaughtObjc);
                    assert_eq!(site.file, sel);
                }
                Ok(v) => assert_eq!((raises, v), (false, 0), "{sel}"),
            }
        }
    }

    /// The raising depth-stencil getters' raise comes back as `Err` (they are caught sends), so the
    /// process lives; the others answer 0
    #[test]
    fn depth_stencil_info_getters_raise_comes_back_as_err() {
        let k = kit();
        let (id_raises, index_raises) = RAISING_GETTERS;
        for (sel, f, raises) in [
            (
                "gpuResourceID",
                DepthStencilState::gpu_resource_id
                    as fn(&MetalDepthStencilState) -> RaiseResult<u64>,
                id_raises,
            ),
            (
                "resourceIndex",
                DepthStencilState::resource_index,
                index_raises,
            ),
        ] {
            let r = catch_death(|| f(&k.depth_stencil).map_err(|t| t.site().file));
            let expected = if raises { Err(sel) } else { Ok(0) };
            assert_eq!(r, Ok(expected), "{sel} comes back as Err where it raises");
        }
    }
}
