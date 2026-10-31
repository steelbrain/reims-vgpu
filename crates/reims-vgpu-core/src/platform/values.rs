//! Descriptors as values: plain Rust data that the platform turns into a Metal descriptor inside
//! the creator that takes it, and the `*Set` values that set a made descriptor's properties in
//! one call.
//!
//! A value holds no Metal object of its own and takes no retain: an object it names is a
//! borrowed `&P::X`. The platform sends a value's setters in the order its type states, which is
//! the order core set the properties in before values; an implementation with no Metal
//! descriptor of its own reads the value directly instead.

use std::fmt;

use super::host_range::HostRange;
use super::types::{ClearColor, MtlSize, Swizzle};
use super::{Handle, ObjId, Platform};
use crate::host::HostPtr;

/// One storage property of a buffer or heap descriptor. They alias one another in Metal
/// (`resourceOptions` holds the storage and hazard-tracking modes), so they apply in list order
/// and the last one wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptionSet {
    /// `setResourceOptions:`.
    ResourceOptions(u64),
    /// `setResourceOptions:` of the descriptor's `resourceOptions`, read at this point, with
    /// these bits or'ed in.
    ResourceOptionsOr(u64),
    /// `setStorageMode:`.
    StorageMode(u64),
    /// `setHazardTrackingMode:` (a heap descriptor's; a buffer descriptor has none, and a buffer
    /// value's is not sent).
    HazardTrackingMode(u64),
}

/// One `MTLArgumentDescriptor` of `newArgumentEncoderWithArguments:`, as a value: each field is
/// its Metal raw value, sent with its setter in field order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ArgumentDesc {
    /// `setDataType:` (`MTLDataType`).
    pub data_type: u64,
    /// `setIndex:`.
    pub index: u64,
    /// `setArrayLength:`.
    pub array_length: u64,
    /// `setAccess:` (`MTLBindingAccess`).
    pub access: u64,
    /// `setTextureType:` (`MTLTextureType`).
    pub texture_type: u64,
    /// `setConstantBlockAlignment:`.
    pub constant_block_alignment: u64,
}

/// The block Metal calls once, with the contents and the length, when a buffer made from a
/// [`BufferDesc`] is released.
pub type BufferDeallocator = Box<dyn FnOnce(HostPtr, u64) + Send>;

/// The private `MTLBufferDescriptor`, as a value. Order: `setLength:`; the options in list order
/// (`ResourceOptionsOr` reads `resourceOptions` first; a hazard-tracking mode is not sent);
/// `setPinnedGPUAddress:`; `setContents:`; `setNoCopy:`; `setDeallocator:`.
pub struct BufferDesc<'a> {
    /// `setLength:`.
    pub length: u64,
    /// The storage properties, in order.
    pub options: Vec<OptionSet>,
    /// `setPinnedGPUAddress:`, when set.
    pub pinned_gpu_address: Option<u64>,
    /// `setContents:`, when set: only the pointer is sent, and the range's constructor keeps it
    /// mapped until the deallocator runs.
    pub contents: Option<HostRange<'a>>,
    /// `setNoCopy:YES` when true (nothing is sent when false).
    pub no_copy: bool,
    /// `setDeallocator:`, when set.
    pub deallocator: Option<BufferDeallocator>,
}

impl fmt::Debug for BufferDesc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferDesc")
            .field("length", &self.length)
            .field("options", &self.options)
            .field("pinned_gpu_address", &self.pinned_gpu_address)
            .field("contents", &self.contents)
            .field("no_copy", &self.no_copy)
            .field("deallocator", &self.deallocator.is_some())
            .finish()
    }
}

/// An object argument in a value's `Debug`: its identity.
fn id_of<H: Handle>(h: Option<&H>) -> Option<ObjId> {
    h.map(Handle::id)
}

/// `MTLHeapDescriptor`, as a value. Order: `setSize:`; `setType:`; the options up to the first
/// hazard-tracking mode; `setAddressRanges:`; the rest of the options; `setPinnedGPUAddress:`.
pub struct HeapDesc<'a, P: Platform> {
    /// `setSize:`.
    pub size: u64,
    /// `setType:` (`MTLHeapType` is `NSInteger`).
    pub heap_type: i64,
    /// The storage properties, in order.
    pub options: Vec<OptionSet>,
    /// `setAddressRanges:`, when set.
    pub address_ranges: Option<&'a P::AddressRangeArray>,
    /// `setPinnedGPUAddress:`, when set.
    pub pinned_gpu_address: Option<u64>,
}

impl<P: Platform> fmt::Debug for HeapDesc<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeapDesc")
            .field("size", &self.size)
            .field("heap_type", &self.heap_type)
            .field("options", &self.options)
            .field("address_ranges", &id_of(self.address_ranges))
            .field("pinned_gpu_address", &self.pinned_gpu_address)
            .finish()
    }
}

/// What `GpuDevice::new_heap` answers: the heap, and its descriptor's options.
pub struct HeapMade<P: Platform> {
    /// The heap; `None` for nil.
    pub heap: Option<P::Heap>,
    /// The descriptor's `resourceOptions` as Metal reports them after the value was applied.
    pub descriptor_resource_options: u64,
}

/// `MTLSamplerDescriptor`, as a value: each field is sent when set. Order: min filter, mag filter,
/// LOD average, compare function, S, T and R address modes, mip filter, max anisotropy,
/// normalized coordinates, argument buffer support, reduction mode, LOD min and max clamps,
/// border colour, resource index.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct SamplerDesc {
    pub min_filter: Option<u64>,
    pub mag_filter: Option<u64>,
    pub lod_average: Option<bool>,
    pub compare_function: Option<u64>,
    pub s_address_mode: Option<u64>,
    pub t_address_mode: Option<u64>,
    pub r_address_mode: Option<u64>,
    pub mip_filter: Option<u64>,
    pub max_anisotropy: Option<u64>,
    pub normalized_coordinates: Option<bool>,
    pub support_argument_buffers: Option<bool>,
    pub reduction_mode: Option<u64>,
    pub lod_min_clamp: Option<f32>,
    pub lod_max_clamp: Option<f32>,
    pub border_color: Option<u64>,
    /// `setResourceIndex:` (SPI).
    pub resource_index: Option<u64>,
}

/// `MTLStencilDescriptor`, as a value: every field is sent, in field order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct StencilDesc {
    pub stencil_compare_function: u64,
    pub stencil_failure_operation: u64,
    pub depth_failure_operation: u64,
    pub depth_stencil_pass_operation: u64,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// `MTLDepthStencilDescriptor`, as a value. Order: depth compare function, depth write, then the
/// front face's six properties, then the back face's (a nil face takes none).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DepthStencilDesc {
    /// `setDepthCompareFunction:`, when set.
    pub depth_compare_function: Option<u64>,
    /// `setDepthWriteEnabled:`, when set.
    pub depth_write_enabled: Option<bool>,
    /// The front face's stencil (`frontFaceStencil`), when set.
    pub front: Option<StencilDesc>,
    /// The back face's stencil (`backFaceStencil`), when set.
    pub back: Option<StencilDesc>,
}

/// `MTLComputePassDescriptor`, as a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ComputePassDesc {
    /// `setDispatchType:`, when set.
    pub dispatch_type: Option<u64>,
}

/// `MTLIndirectCommandBufferDescriptor`, as a value: every field is sent but the resource index,
/// which is sent when set. Order: field order, the resource index last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct IcbDesc {
    pub command_types: u64,
    pub max_vertex_buffer_bind_count: u64,
    pub max_fragment_buffer_bind_count: u64,
    pub max_kernel_buffer_bind_count: u64,
    pub max_object_buffer_bind_count: u64,
    pub max_mesh_buffer_bind_count: u64,
    pub max_kernel_threadgroup_memory_bind_count: u64,
    pub max_object_threadgroup_memory_bind_count: u64,
    pub inherit_pipeline_state: bool,
    pub inherit_buffers: bool,
    pub support_ray_tracing: bool,
    pub support_dynamic_attribute_stride: bool,
    /// `setMaxScissorRectCount:` (SPI).
    pub max_scissor_rect_count: u64,
    /// `setMaxViewportCount:` (SPI).
    pub max_viewport_count: u64,
    /// `setInheritStencilReferenceValues:` (SPI).
    pub inherit_stencil_reference_values: bool,
    /// `setInheritDepthTestBounds:` (SPI).
    pub inherit_depth_test_bounds: bool,
    /// `setInheritScissorRects:` (SPI).
    pub inherit_scissor_rects: bool,
    /// `setInheritViewports:` (SPI).
    pub inherit_viewports: bool,
    /// `setInheritBlendColor:` (SPI).
    pub inherit_blend_color: bool,
    /// `setAllowOverrideRenderStates:` (SPI; `NSInteger` in the runtime).
    pub allow_override_render_states: i64,
    pub inherit_depth_stencil_state: bool,
    pub inherit_depth_bias: bool,
    pub inherit_depth_clip_mode: bool,
    pub inherit_cull_mode: bool,
    pub inherit_front_facing_winding: bool,
    pub inherit_triangle_fill_mode: bool,
    /// `setResourceIndex:` (SPI), when set.
    pub resource_index: Option<u64>,
}

/// `MTLRenderPipelineDescriptor` for the display's pipeline, or a window's cursor's, as a value.
/// Order: label, vertex function, fragment function, then colour attachment 0's pixel format
/// (through `colorAttachments` and `objectAtIndexedSubscript:`, which may raise) and, where set,
/// its blending, then the resource index and its force flag.
pub struct RenderPipelineDesc<'a, P: Platform> {
    /// `setLabel:`, when set.
    pub label: Option<&'a str>,
    /// `setVertexFunction:`.
    pub vertex: &'a P::Function,
    /// `setFragmentFunction:`.
    pub fragment: &'a P::Function,
    /// `colorAttachments[0].pixelFormat`.
    pub color0_pixel_format: u64,
    /// Colour attachment 0 blends each fragment over the target by its alpha, its colour taken
    /// as premultiplied: `setBlendingEnabled:`, then both source factors `one` and both
    /// destination factors `oneMinusSourceAlpha`. Off (Metal's default, `one` over `zero`) for
    /// the display's own pipelines.
    pub blend_over: bool,
    /// `setResourceIndex:` then `setForceResourceIndex:` (SPI), when set.
    pub resource_index: Option<(u64, bool)>,
}

impl<P: Platform> fmt::Debug for RenderPipelineDesc<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RenderPipelineDesc")
            .field("label", &self.label)
            .field("vertex", &self.vertex.id())
            .field("fragment", &self.fragment.id())
            .field("color0_pixel_format", &self.color0_pixel_format)
            .field("blend_over", &self.blend_over)
            .field("resource_index", &self.resource_index)
            .finish()
    }
}

/// `MTLTextureDescriptor`, as a value, from which the platform makes the descriptor.
///
/// The first nine fields are always sent; [`Default`] gives Metal's own defaults for them. The
/// others are sent when set. Order: texture type, framebuffer-only, drawable, GPU-optimized
/// contents, width, height, depth, mipmap level count, sample count, array length, usage, pixel
/// format, resource options, protection options, write swizzle, swizzle, resource index and its
/// force flag, then rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct TextureDesc {
    pub texture_type: u64,
    pub pixel_format: u64,
    pub width: u64,
    pub height: u64,
    pub depth: u64,
    pub mipmap_level_count: u64,
    pub sample_count: u64,
    pub array_length: u64,
    pub usage: u64,
    pub resource_options: Option<u64>,
    pub swizzle: Option<Swizzle>,
    /// `setResourceIndex:` then `setForceResourceIndex:` (SPI).
    pub resource_index: Option<(u64, bool)>,
    /// `setProtectionOptions:` (SPI).
    pub protection_options: Option<u64>,
    /// `setFramebufferOnly:` (SPI).
    pub framebuffer_only: Option<bool>,
    /// `setIsDrawable:` (SPI).
    pub is_drawable: Option<bool>,
    /// `setWriteSwizzleEnabled:` (SPI).
    pub write_swizzle_enabled: Option<bool>,
    pub allow_gpu_optimized_contents: Option<bool>,
    /// `setRotation:` (SPI).
    pub rotation: Option<u64>,
}

/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;
/// `MTLPixelFormatRGBA8Unorm`.
const PIXEL_FORMAT_RGBA8_UNORM: u64 = 70;
/// `MTLTextureUsageShaderRead`.
const TEXTURE_USAGE_SHADER_READ: u64 = 1;

impl Default for TextureDesc {
    /// A new `MTLTextureDescriptor`'s values: a 1×1×1 2D RGBA8Unorm texture of one level, one
    /// sample and one slice, read by shaders.
    fn default() -> TextureDesc {
        TextureDesc {
            texture_type: TEXTURE_TYPE_2D,
            pixel_format: PIXEL_FORMAT_RGBA8_UNORM,
            width: 1,
            height: 1,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            usage: TEXTURE_USAGE_SHADER_READ,
            resource_options: None,
            swizzle: None,
            resource_index: None,
            protection_options: None,
            framebuffer_only: None,
            is_drawable: None,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: None,
            rotation: None,
        }
    }
}

/// `MTLRasterizationRateMapDescriptor`, as a value, from which the platform makes the descriptor;
/// its layers are set on the made descriptor. Order: screen size, mutability, min factor.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RateMapDesc {
    /// `setScreenSize:`.
    pub screen_size: MtlSize,
    /// `setMutability:`, when set.
    pub mutability: Option<u64>,
    /// `setMinFactor:` (SPI), when set.
    pub min_factor: Option<f32>,
}

/// What core reads back from a render pass attachment: Metal's own reading of what was set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AttachmentState {
    /// `loadAction`.
    pub load_action: u64,
    /// `storeAction`.
    pub store_action: u64,
}

/// Implements `Default` and a `Debug` that shows objects by identity, for a `*Set` of `Option`
/// fields.
macro_rules! set_struct_impls {
    ($name:ident { $($obj:ident),* } { $($val:ident),* }) => {
        impl<P: Platform> Default for $name<'_, P> {
            fn default() -> Self {
                $name { $($obj: None,)* $($val: None,)* }
            }
        }

        impl<P: Platform> fmt::Debug for $name<'_, P> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct(stringify!($name))
                    $(.field(stringify!($obj), &self.$obj.map(id_of)))*
                    $(.field(stringify!($val), &self.$val))*
                    .finish()
            }
        }
    };
}

/// The render pass descriptor's own properties, each sent when set (`Some(None)` sends nil).
/// Order: visibility result buffer, render target array length, width and height, default raster
/// and colour sample counts, rasterization rate map, imageblock sample length, threadgroup memory
/// length, tile width and height.
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct PassSet<'a, P: Platform> {
    pub visibility_result_buffer: Option<Option<&'a P::Buffer>>,
    pub render_target_array_length: Option<u64>,
    pub render_target_width: Option<u64>,
    pub render_target_height: Option<u64>,
    pub default_raster_sample_count: Option<u64>,
    /// `setDefaultColorSampleCount:` (SPI).
    pub default_color_sample_count: Option<u64>,
    pub rasterization_rate_map: Option<Option<&'a P::RateMap>>,
    pub imageblock_sample_length: Option<u64>,
    pub threadgroup_memory_length: Option<u64>,
    pub tile_width: Option<u64>,
    pub tile_height: Option<u64>,
}

set_struct_impls!(PassSet { visibility_result_buffer, rasterization_rate_map } {
    render_target_array_length, render_target_width, render_target_height,
    default_raster_sample_count, default_color_sample_count, imageblock_sample_length,
    threadgroup_memory_length, tile_width, tile_height
});

/// A colour attachment's properties, each sent when set (`Some(None)` sends nil). Order:
/// texture, resolve texture, level, slice, depth plane, resolve level, slice and depth
/// plane, load and store actions, store action options, then the clear colour.
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct ColorAttachmentSet<'a, P: Platform> {
    pub texture: Option<Option<&'a P::Texture>>,
    pub resolve_texture: Option<Option<&'a P::Texture>>,
    pub level: Option<u64>,
    pub slice: Option<u64>,
    pub depth_plane: Option<u64>,
    pub resolve_level: Option<u64>,
    pub resolve_slice: Option<u64>,
    pub resolve_depth_plane: Option<u64>,
    pub load_action: Option<u64>,
    pub store_action: Option<u64>,
    pub store_action_options: Option<u64>,
    pub clear_color: Option<ClearColor>,
}

set_struct_impls!(ColorAttachmentSet { texture, resolve_texture } {
    level, slice, depth_plane, resolve_level, resolve_slice, resolve_depth_plane, load_action,
    store_action, store_action_options, clear_color
});

/// The depth attachment's properties, each sent when set (`Some(None)` sends nil). Order:
/// texture, resolve texture, level, slice, depth plane, resolve level, slice and depth plane,
/// load and store actions, store action options, then the clear depth, then the resolve
/// filter.
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct DepthAttachmentSet<'a, P: Platform> {
    pub texture: Option<Option<&'a P::Texture>>,
    pub resolve_texture: Option<Option<&'a P::Texture>>,
    pub level: Option<u64>,
    pub slice: Option<u64>,
    pub depth_plane: Option<u64>,
    pub resolve_level: Option<u64>,
    pub resolve_slice: Option<u64>,
    pub resolve_depth_plane: Option<u64>,
    pub load_action: Option<u64>,
    pub store_action: Option<u64>,
    pub store_action_options: Option<u64>,
    pub clear_depth: Option<f64>,
    pub depth_resolve_filter: Option<u64>,
}

set_struct_impls!(DepthAttachmentSet { texture, resolve_texture } {
    level, slice, depth_plane, resolve_level, resolve_slice, resolve_depth_plane, load_action,
    store_action, store_action_options, clear_depth, depth_resolve_filter
});

/// The stencil attachment's properties, each sent when set (`Some(None)` sends nil). Order:
/// texture, resolve texture, level, slice, depth plane, resolve level, slice and depth plane,
/// load and store actions, store action options, then the clear stencil, then the resolve
/// filter.
#[allow(missing_docs)] // each field is its setter's argument, named after it
pub struct StencilAttachmentSet<'a, P: Platform> {
    pub texture: Option<Option<&'a P::Texture>>,
    pub resolve_texture: Option<Option<&'a P::Texture>>,
    pub level: Option<u64>,
    pub slice: Option<u64>,
    pub depth_plane: Option<u64>,
    pub resolve_level: Option<u64>,
    pub resolve_slice: Option<u64>,
    pub resolve_depth_plane: Option<u64>,
    pub load_action: Option<u64>,
    pub store_action: Option<u64>,
    pub store_action_options: Option<u64>,
    pub clear_stencil: Option<u32>,
    pub stencil_resolve_filter: Option<u64>,
}

set_struct_impls!(StencilAttachmentSet { texture, resolve_texture } {
    level, slice, depth_plane, resolve_level, resolve_slice, resolve_depth_plane, load_action,
    store_action, store_action_options, clear_stencil, stencil_resolve_filter
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::command::{CommandBuffer, CommandQueue};
    use crate::platform::descriptor::{RenderPassDescriptor, TextureDescriptor};
    use crate::platform::device::GpuDevice;
    use crate::platform::rate_map::RateMapDescriptor;
    use crate::platform::resource::{
        Buffer, GpuResource, Heap, Library, RenderPipeline, SamplerState,
    };
    use crate::raise::OnFault;

    use crate::testkit::calllog::Arg;
    use crate::testkit::{FakePlatform, Step};

    /// The selectors sent since `m`.
    fn sels(p: &FakePlatform, m: u64) -> Vec<&'static str> {
        p.call_log().sels_since(m)
    }

    /// A texture value's ordered setters (the pixel format, which Metal checks, and the resource
    /// options, which alias the storage mode) are logged; the descriptor holds every field
    #[test]
    fn texture_descriptor_from_a_value() {
        let p = FakePlatform::new();
        let m = p.call_log().mark();
        let d = p.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 8,
            resource_options: Some(0x20),
            resource_index: Some((5, true)),
            rotation: Some(1),
            is_drawable: Some(false),
            ..TextureDesc::default()
        });
        assert_eq!(sels(&p, m), ["setPixelFormat:", "setResourceOptions:"]);
        assert_eq!(d.pixel_format(), 80);
        assert_eq!(d.texture_type(), 2);
        assert_eq!(d.storage_mode(), 2);
        assert_eq!(d.rotation(), 1);
        let m = p.call_log().mark();
        d.set_resource_options(0);
        d.set_rotation(2);
        d.set_usage(3);
        assert_eq!(sels(&p, m), ["setResourceOptions:"]);
        assert_eq!((d.rotation(), d.storage_mode()), (2, 0));
        let t = p.device().new_texture(OnFault::Die, &d).unwrap().unwrap();
        assert_eq!(crate::platform::resource::Texture::resource_index(&t), 5);
        assert_eq!(crate::platform::resource::Texture::width(&t), 8);
    }

    /// The rate-map value, then the typed layer reads
    #[test]
    fn rate_map_descriptor_from_a_value() {
        let p = FakePlatform::new();
        let m = p.call_log().mark();
        let d = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(4, 4, 0),
            mutability: Some(1),
            min_factor: Some(0.5),
        });
        assert!(sels(&p, m).is_empty(), "no ordered call");
        assert_eq!(d.layer_count(), 0);
        let layer = p
            .new_rate_map_layer(OnFault::Die, MtlSize::new(1, 1, 0))
            .unwrap()
            .unwrap();
        d.set_layer(OnFault::Die, &layer, 0).unwrap();
        assert_eq!(d.layer_count(), 1);
        assert_eq!(
            d.layer(OnFault::Die, 0).unwrap().map(|l| l.id()),
            Some(layer.id())
        );
        assert!(d.layer(OnFault::Die, 1).unwrap().is_none());
    }

    /// The render pass's typed sets reach the pass and its attachments, with no ordered call
    #[test]
    fn render_pass_sets() {
        let p = FakePlatform::new();
        let pass = p.new_render_pass_descriptor();
        let t = p
            .device()
            .new_texture(OnFault::Die, &p.texture_descriptor(&TextureDesc::default()))
            .unwrap()
            .unwrap();
        let m = p.call_log().mark();
        pass.set_pass(&PassSet {
            render_target_width: Some(4),
            tile_height: Some(8),
            visibility_result_buffer: Some(None),
            ..PassSet::default()
        });
        assert_eq!(
            pass.pass(|p| (p.render_target_width, p.tile_height)),
            (Some(4), Some(8))
        );
        pass.set_color_attachment(
            OnFault::Die,
            2,
            &ColorAttachmentSet {
                texture: Some(Some(&t)),
                load_action: Some(2),
                store_action: Some(1),
                clear_color: Some(ClearColor::default()),
                ..ColorAttachmentSet::default()
            },
        )
        .unwrap();
        assert_eq!(
            pass.color(
                2,
                crate::testkit::impls::descriptor::AttachmentModel::texture_id
            ),
            Some(t.id())
        );
        assert_eq!(
            pass.color_attachment_state(2).unwrap(),
            AttachmentState {
                load_action: 2,
                store_action: 1
            }
        );
        pass.set_depth_attachment(&DepthAttachmentSet {
            clear_depth: Some(1.0),
            store_action: Some(3),
            ..DepthAttachmentSet::default()
        });
        pass.set_stencil_attachment(&StencilAttachmentSet {
            load_action: Some(1),
            clear_stencil: Some(7),
            ..StencilAttachmentSet::default()
        });
        assert_eq!(pass.depth_attachment_state().store_action, 3);
        assert_eq!(pass.stencil_attachment_state().load_action, 1);
        assert!(sels(&p, m).is_empty(), "no ordered call");
        assert!(format!("{:?}", PassSet::<FakePlatform>::default()).starts_with("PassSet {"));
    }

    /// The fake's creators from values log the creator with the value, and model what Metal's
    /// descriptor would hold
    #[test]
    fn fake_creators_from_values() {
        let p = FakePlatform::new();
        let d = p.device();
        let bytes = [0u8; 64];
        let m = p.call_log().mark();
        let b = d
            .new_buffer(BufferDesc {
                length: 64,
                options: vec![
                    OptionSet::ResourceOptions(0x1),
                    OptionSet::StorageMode(0),
                    OptionSet::ResourceOptionsOr(0x100),
                ],
                pinned_gpu_address: Some(0x5_0000_0000),
                contents: Some(HostRange::of_slice(&bytes)),
                no_copy: true,
                deallocator: None,
            })
            .unwrap()
            .unwrap();
        assert_eq!(sels(&p, m), ["newBufferWithDescriptor:"]);
        assert!(matches!(
            &p.call_log().calls_to("newBufferWithDescriptor:")[0].args[0],
            Arg::Val(t) if t.starts_with("BufferDesc { length: 64")
        ));
        assert_eq!(
            (b.length(), b.resource_options(), b.gpu_address()),
            (64, 0x101, 0x5_0000_0000)
        );
        assert_eq!(b.contents(), HostRange::of_slice(&bytes).ptr());

        let made = d
            .new_heap(&HeapDesc::<FakePlatform> {
                size: 0x4000,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(2), OptionSet::HazardTrackingMode(2)],
                address_ranges: None,
                pinned_gpu_address: None,
            })
            .unwrap();
        assert_eq!(made.descriptor_resource_options, 0x20);
        assert_eq!(made.heap.map(|h| h.resource_options()), Some(0x20));

        let s = d
            .new_sampler_state(&SamplerDesc {
                resource_index: Some(6),
                ..SamplerDesc::default()
            })
            .unwrap()
            .unwrap();
        assert_eq!(s.resource_index(), 6);
        assert!(
            d.new_depth_stencil_state(&DepthStencilDesc::default())
                .unwrap()
                .is_some()
        );
        let icb = d
            .new_indirect_command_buffer(
                &IcbDesc {
                    resource_index: Some(3),
                    ..IcbDesc::default()
                },
                16,
                0,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            crate::platform::icb::IndirectCommandBuffer::resource_index(&icb),
            3
        );

        p.world()
            .set_library_functions(vec!["v".to_owned(), "f".to_owned()]);
        let lib = d.new_shader_library().unwrap().unwrap();
        let v = lib.new_function(OnFault::Die, "v").unwrap().unwrap();
        let f = lib.new_function(OnFault::Die, "f").unwrap().unwrap();
        let rp = d
            .new_render_pipeline_state(&RenderPipelineDesc::<FakePlatform> {
                label: Some("x"),
                vertex: &v,
                fragment: &f,
                color0_pixel_format: 80,
                blend_over: false,
                resource_index: Some((12, true)),
            })
            .unwrap()
            .unwrap();
        assert_eq!(rp.resource_index(), 12);

        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
        let m = p.call_log().mark();
        assert!(
            cb.compute_encoder_for_pass(&ComputePassDesc {
                dispatch_type: Some(1)
            })
            .unwrap()
            .is_some()
        );
        assert_eq!(
            p.call_log().since(m)[0].args,
            [Arg::Val(
                "ComputePassDesc { dispatch_type: Some(1) }".to_owned()
            )]
        );
        // Scripts keyed by the creator's selector apply.
        p.script().push("newSamplerStateWithDescriptor:", Step::Nil);
        assert!(
            d.new_sampler_state(&SamplerDesc::default())
                .unwrap()
                .is_none()
        );
    }
}
