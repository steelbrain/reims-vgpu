//! Fake: the `platform/resource.rs` traits over the fake models (architecture 13.2).
//!
//! The ordered calls (creators, labels, purgeable states, backing changes, the depth-stencil
//! state's raising getters) log their selector and arguments and consult the script; the free
//! getters read the models.

use super::super::calllog::Arg;
use super::super::objects::{
    FakeVariant, Fk, Model, Prepared, PreparedGrid, ResourceCommon, TexDesc, TextureModel, kind,
};
use super::super::platform::FakePlatform;
use super::{Backing, fatal, new_buffer, new_texture};
use crate::core::host::HostPtr;
use crate::core::num::ToU64 as _;
use crate::core::platform::compute::Dispatch;
use crate::core::platform::resource::*;
use crate::core::platform::types::{MtlRegion, MtlSize, NsRange, Swizzle};
use crate::core::platform::values::SurfacePlane;
use crate::core::platform::{Handle as _, HostRange, Platform};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::testkit::script::Policy;

/// The name the fake logs and scripts [`Buffer::make_memory`] of a deferred buffer whose memory
/// is not made under: no selector of Metal's names it.
pub const MAKE_MEMORY: &str = "makeMemory (deferred buffer)";

impl Fk<kind::Buffer> {
    /// `f` of the buffer's model; the default of `T` for a buffer a test made with no model.
    fn buffer<T: Default>(&self, f: impl FnOnce(&super::super::objects::BufferModel) -> T) -> T {
        match &self.state().model {
            Model::Buffer(m) => f(m),
            _ => T::default(),
        }
    }
}

impl Fk<kind::Texture> {
    /// `f` of the texture's model; the default of `T` for a texture a test made with no model.
    fn texture<T: Default>(&self, f: impl FnOnce(&TextureModel) -> T) -> T {
        match &self.state().model {
            Model::Texture(m) => f(m),
            _ => T::default(),
        }
    }

    /// The descriptor a view of this texture starts from.
    fn view_desc(&self) -> TexDesc {
        match &self.state().model {
            Model::Texture(m) => m.desc,
            _ => TexDesc::default(),
        }
    }
}

impl GpuResource<FakePlatform> for Fk<kind::Buffer> {
    fn device(&self) -> Option<<FakePlatform as Platform>::Device> {
        self.device_handle()
    }

    fn resource_options(&self) -> u64 {
        self.buffer(|m| m.options)
    }

    fn set_label(&self, label: &str) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "setLabel:",
            vec![Arg::Str(label.to_owned())],
        )?;
        let _ = self.with_common(|c| c.label = Some(label.to_owned()));
        Ok(())
    }

    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "setPurgeableState:",
            vec![Arg::U64(state)],
            || self.purgeable_swap(state),
        )
    }

    fn storage_mode(&self) -> u64 {
        self.buffer(|m| m.options.wrapping_shr(4) & 0xf)
    }
}

impl GpuResource<FakePlatform> for Fk<kind::Texture> {
    fn device(&self) -> Option<<FakePlatform as Platform>::Device> {
        self.device_handle()
    }

    fn resource_options(&self) -> u64 {
        self.texture(|m| m.desc.resource_options)
    }

    fn set_label(&self, label: &str) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "setLabel:",
            vec![Arg::Str(label.to_owned())],
        )?;
        let _ = self.with_common(|c| c.label = Some(label.to_owned()));
        Ok(())
    }

    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "setPurgeableState:",
            vec![Arg::U64(state)],
            || self.purgeable_swap(state),
        )
    }

    fn storage_mode(&self) -> u64 {
        self.texture(|m| m.desc.storage_mode())
    }
}

impl Buffer<FakePlatform> for Fk<kind::Buffer> {
    fn contents(&self) -> HostPtr {
        self.buffer(|m| m.contents)
    }

    fn did_modify_range(&self, on: OnFault, range: NsRange) -> RaiseResult<()> {
        self.act(Policy::On(on), "didModifyRange:", vec![Arg::val(&range)])
    }

    fn gpu_address(&self) -> u64 {
        self.buffer(|m| m.gpu_address)
    }

    fn length(&self) -> u64 {
        self.buffer(|m| m.length)
    }

    /// A deferred buffer whose memory is not made logs [`MAKE_MEMORY`] and answers its scripted
    /// value, else `true`, its memory then made; any other answers `true`, unlogged.
    fn make_memory(&self, _made_for: MemoryFor) -> bool {
        if self.buffer(|m| m.deferred) != Some(false) {
            return true;
        }
        let made = self.get_logged(MAKE_MEMORY, vec![], || true);
        if made && let Model::Buffer(m) = &mut self.state().model {
            m.deferred = Some(true);
        }
        made
    }

    /// `false` for a deferred buffer whose memory is not made ([`Self::make_memory`]).
    fn memory_made(&self) -> bool {
        self.buffer(|m| m.deferred) != Some(false)
    }

    fn new_linear_texture(
        &self,
        descriptor: &<FakePlatform as Platform>::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let desc = descriptor.desc();
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
            vec![
                Arg::obj(descriptor),
                Arg::U64(offset),
                Arg::U64(bytes_per_row),
                Arg::U64(bytes_per_image),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &<FakePlatform as Platform>::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let desc = descriptor.desc();
        let parent = self.erase();
        self.make(
            Policy::On(on),
            "newTextureWithDescriptor:offset:bytesPerRow:",
            vec![
                Arg::obj(descriptor),
                Arg::U64(offset),
                Arg::U64(bytes_per_row),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn new_tiled_texture(
        &self,
        descriptor: &<FakePlatform as Platform>::TextureDescriptor,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let desc = descriptor.desc();
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTiledTextureWithDescriptor:offset:bytesPerRow:",
            vec![
                Arg::obj(descriptor),
                Arg::U64(offset),
                Arg::U64(bytes_per_row),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    /// Logged with the plane's index, offset and rows, whether the surface has planes, and its
    /// facts (`Arg::val` of the `Option`); a texture over the buffer, as `new_texture`'s.
    fn new_texture_over_surface_plane(
        &self,
        on: OnFault,
        descriptor: &<FakePlatform as Platform>::TextureDescriptor,
        plane: SurfacePlane,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let desc = descriptor.desc();
        let parent = self.erase();
        self.make(
            Policy::On(on),
            "newTextureWithDescriptor:iosurface:plane:",
            vec![
                Arg::obj(descriptor),
                Arg::U64(plane.plane),
                Arg::U64(plane.offset),
                Arg::U64(plane.bytes_per_row),
                Arg::Bool(plane.planar),
                Arg::val(&plane.facts),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    /// Logged with the new bytes' pointer and length (`rebindNoCopy`); a scripted raise refuses
    /// it, as does a buffer of no contents or of another length, and a refusal drops `deallocator`
    /// unrun as it returns, with no lock held. Otherwise `contents` answers the new pointer, the
    /// buffer's deallocator runs at once, as the fake holds no import of the old bytes, and
    /// `deallocator` takes its place.
    fn rebind_no_copy(
        &self,
        bytes: HostRange<'_>,
        deallocator: Option<crate::core::platform::device::NoCopyDeallocator>,
    ) -> Result<(), String> {
        let (ptr, len) = (bytes.ptr(), bytes.len());
        self.act(
            Policy::Caught,
            "rebindNoCopy",
            vec![Arg::U64(ptr.addr().to_u64()), Arg::U64(len)],
        )
        .map_err(|_| "a scripted refusal".to_owned())?;
        let (old, old_desc) = {
            let mut state = self.state();
            let Model::Buffer(m) = &mut state.model else {
                return Err("no buffer model".to_owned());
            };
            if m.contents.is_null() || m.length != len {
                return Err("no contents, or another length".to_owned());
            }
            let old_ptr = m.contents;
            m.contents = ptr;
            let old = std::mem::replace(&mut m.deallocator, deallocator);
            let old_desc = m.desc_deallocator.take().map(|d| (d, old_ptr, m.length));
            (old, old_desc)
        };
        if let Some(d) = old {
            d();
        }
        if let Some((d, at, length)) = old_desc {
            (d.0)(at, length);
        }
        Ok(())
    }
}

impl Texture<FakePlatform> for Fk<kind::Texture> {
    fn array_length(&self) -> u64 {
        self.texture(|m| m.desc.array_length)
    }

    fn depth(&self) -> u64 {
        self.texture(|m| m.desc.depth)
    }

    fn did_modify_data(&self) {
        fatal(self.act(Policy::Fatal, "didModifyData", vec![]));
    }

    fn gpu_resource_id(&self) -> u64 {
        self.gpu_resource_id_value()
    }

    fn height(&self) -> u64 {
        self.texture(|m| m.desc.height)
    }

    fn is_shareable(&self) -> bool {
        self.texture(|m| m.shareable)
    }

    fn mipmap_level_count(&self) -> u64 {
        self.texture(|m| m.desc.mipmap_level_count)
    }

    fn new_shared_texture_handle(
        &self,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::SharedTextureHandle>> {
        let target = self.erase();
        self.make(Policy::Caught, "newSharedTextureHandle", vec![], |w| {
            Some(w.new_obj::<kind::SharedTextureHandle>(Model::Target(target)))
        })
    }

    fn new_texture_view(
        &self,
        pixel_format: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let mut desc = self.view_desc();
        desc.pixel_format = pixel_format;
        desc.resource_index = None;
        desc.force_resource_index = false;
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTextureViewWithPixelFormat:",
            vec![Arg::U64(pixel_format)],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn new_texture_view_at_index(
        &self,
        pixel_format: u64,
        resource_index: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let mut desc = self.view_desc();
        desc.pixel_format = pixel_format;
        desc.resource_index = Some(resource_index);
        desc.force_resource_index = true;
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTextureViewWithPixelFormat:resourceIndex:",
            vec![Arg::U64(pixel_format), Arg::U64(resource_index)],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn new_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let mut desc = self.view_desc();
        desc.pixel_format = pixel_format;
        desc.texture_type = texture_type;
        desc.mipmap_level_count = level_range.length;
        desc.array_length = slice_range.length;
        desc.resource_index = None;
        desc.force_resource_index = false;
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTextureViewWithPixelFormat:textureType:levels:slices:",
            vec![
                Arg::U64(pixel_format),
                Arg::U64(texture_type),
                Arg::val(&level_range),
                Arg::val(&slice_range),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn new_swizzled_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
        swizzle: Swizzle,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let mut desc = self.view_desc();
        desc.pixel_format = pixel_format;
        desc.texture_type = texture_type;
        desc.mipmap_level_count = level_range.length;
        desc.array_length = slice_range.length;
        desc.swizzle = swizzle;
        desc.resource_index = None;
        desc.force_resource_index = false;
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
            vec![
                Arg::U64(pixel_format),
                Arg::U64(texture_type),
                Arg::val(&level_range),
                Arg::val(&slice_range),
                Arg::val(&swizzle),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn new_swizzled_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        swizzle: Swizzle,
        resource_index: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let mut desc = self.view_desc();
        desc.pixel_format = pixel_format;
        desc.texture_type = texture_type;
        desc.mipmap_level_count = levels.length;
        desc.array_length = slices.length;
        desc.swizzle = swizzle;
        desc.resource_index = Some(resource_index);
        desc.force_resource_index = true;
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
            vec![
                Arg::U64(pixel_format),
                Arg::U64(texture_type),
                Arg::val(&levels),
                Arg::val(&slices),
                Arg::val(&swizzle),
                Arg::U64(resource_index),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn new_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        resource_index: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let mut desc = self.view_desc();
        desc.pixel_format = pixel_format;
        desc.texture_type = texture_type;
        desc.mipmap_level_count = levels.length;
        desc.array_length = slices.length;
        desc.resource_index = Some(resource_index);
        desc.force_resource_index = true;
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:",
            vec![
                Arg::U64(pixel_format),
                Arg::U64(texture_type),
                Arg::val(&levels),
                Arg::val(&slices),
                Arg::U64(resource_index),
            ],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn pixel_format(&self) -> u64 {
        self.texture(|m| m.desc.pixel_format)
    }

    fn protection_options(&self) -> u64 {
        self.texture(|m| m.desc.protection_options)
    }

    fn replace_region(
        &self,
        region: MtlRegion,
        level: u64,
        slice: u64,
        pixel_bytes: HostRange<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:",
            vec![
                Arg::val(&region),
                Arg::U64(level),
                Arg::U64(slice),
                Arg::Ptr(pixel_bytes.ptr().addr()),
                Arg::U64(bytes_per_row),
                Arg::U64(bytes_per_image),
            ],
        )
    }

    fn resource_index(&self) -> u64 {
        self.resource_index_value()
    }

    fn sample_count(&self) -> u64 {
        self.texture(|m| m.desc.sample_count)
    }

    fn texture_type(&self) -> u64 {
        self.texture(|m| m.desc.texture_type)
    }

    fn unique_identifier(&self) -> u64 {
        self.unique_id_value()
    }

    fn usage(&self) -> u64 {
        self.texture(|m| m.desc.usage)
    }

    fn width(&self) -> u64 {
        self.texture(|m| m.desc.width)
    }
}

impl Heap<FakePlatform> for Fk<kind::Heap> {
    fn gpu_address(&self) -> u64 {
        self.heap(|m| m.gpu_address)
    }

    fn new_buffer_of_length(
        &self,
        length: u64,
        options: u64,
        offset: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        let (base, gpu, size) = match &self.state().model {
            Model::Heap(m) => (m.mem.as_ref().map(|r| r.addr()), m.gpu_address, m.size),
            _ => (None, 0, 0),
        };
        let heap = self.erase();
        self.make(
            Policy::Caught,
            "newBufferWithLength:options:offset:",
            vec![Arg::U64(length), Arg::U64(options), Arg::U64(offset)],
            |w| {
                let fits = offset.checked_add(length).is_some_and(|e| e <= size);
                let contents = match base {
                    Some(b) if fits => b.wrapping_add(offset),
                    _ => HostPtr::NULL,
                };
                let b = new_buffer(
                    w,
                    length,
                    options,
                    Backing::At(contents),
                    Some(gpu.wrapping_add(offset)),
                )?;
                if let Model::Buffer(m) = &mut b.state().model {
                    m.heap = Some(heap);
                }
                Some(b)
            },
        )
    }

    fn new_texture(
        &self,
        descriptor: &<FakePlatform as Platform>::TextureDescriptor,
        offset: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let desc = descriptor.desc();
        let parent = self.erase();
        self.make(
            Policy::Caught,
            "newTextureWithDescriptor:offset:",
            vec![Arg::obj(descriptor), Arg::U64(offset)],
            |w| Some(new_texture(w, desc, Some(parent), false)),
        )
    }

    fn resource_options(&self) -> u64 {
        self.heap(|m| m.options)
    }

    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "setPurgeableState:",
            vec![Arg::U64(state)],
            || self.purgeable_swap(state),
        )
    }

    fn storage_mode(&self) -> u64 {
        self.heap(|m| m.options.wrapping_shr(4) & 0xf)
    }
}

impl SamplerState for Fk<kind::SamplerState> {
    fn gpu_resource_id(&self) -> u64 {
        self.gpu_resource_id_value()
    }

    fn resource_index(&self) -> u64 {
        self.resource_index_value()
    }
}

impl DepthStencilState for Fk<kind::DepthStencilState> {
    fn gpu_resource_id(&self) -> RaiseResult<u64> {
        self.thr(Policy::Caught, "gpuResourceID", vec![], || {
            self.gpu_resource_id_value()
        })
    }

    fn resource_index(&self) -> RaiseResult<u64> {
        self.thr(Policy::Caught, "resourceIndex", vec![], || {
            self.resource_index_value()
        })
    }
}

impl ComputePipeline<FakePlatform> for Fk<kind::ComputePipeline> {
    fn gpu_resource_id(&self) -> u64 {
        self.gpu_resource_id_value()
    }

    /// A new handle each ask; a script answers nil for a function the state does not link.
    fn function_handle(
        &self,
        function: &<FakePlatform as Platform>::Function,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::FunctionHandle>> {
        self.make(
            Policy::Caught,
            "functionHandleWithFunction:",
            vec![Arg::obj(function)],
            |w| Some(w.new_obj(Model::None)),
        )
    }

    /// A table of `function_count` null entries, with a fresh `gpuResourceID`.
    fn new_visible_function_table(
        &self,
        function_count: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::VisibleFunctionTable>> {
        self.make(
            Policy::Caught,
            "newVisibleFunctionTableWithDescriptor:",
            vec![Arg::U64(function_count)],
            |w| {
                Some(w.new_obj(Model::FunctionTable(
                    super::function_table::FunctionTableModel {
                        common: super::common(w, None),
                        entries: vec![None; usize::try_from(function_count).unwrap_or(0)],
                    },
                )))
            },
        )
    }

    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "imageblockMemoryLengthForDimensions:",
            vec![Arg::val(&imageblock_dimensions)],
            Default::default,
        )
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        1024
    }

    fn resource_index(&self) -> u64 {
        self.resource_index_value()
    }

    fn static_threadgroup_memory_length(&self) -> u64 {
        self.with_model(|m: &PipelineModel| m.static_threadgroup_memory_length)
    }

    fn support_indirect_command_buffers(&self) -> bool {
        true
    }

    fn thread_execution_width(&self) -> u64 {
        32
    }

    fn unique_identifier(&self) -> u64 {
        self.unique_id_value()
    }

    /// Logged as [`TEXTURE_SLOT_WRITES`]; the script's answer, else none known.
    fn texture_slot_writes(&self, index: u64) -> Option<bool> {
        self.get_logged(TEXTURE_SLOT_WRITES, vec![Arg::U64(index)], || None)
    }

    /// Noted beside the call log
    /// ([`World::prepared`](super::super::objects::World::prepared)), and held where the test
    /// asked; it makes nothing.
    fn prepare_dispatch(&self, dispatch: Dispatch, threads: MtlSize, lengths: &[(u64, u64)]) {
        self.world()
            .note_prepared(prepared(self, dispatch, threads, lengths));
    }

    /// The world's key of the dispatch
    /// ([`World::key_dispatches`](super::super::objects::World::key_dispatches)): by default one
    /// of the state and the dispatch's own, made where the test said so
    /// ([`World::variants_made`](super::super::objects::World::variants_made)). Not logged, as
    /// its make ahead is not.
    fn dispatch_variant(
        &self,
        dispatch: Dispatch,
        threads: MtlSize,
        lengths: &[(u64, u64)],
    ) -> Option<(FakeVariant, bool)> {
        self.world()
            .dispatch_variant(prepared(self, dispatch, threads, lengths))
    }
}

/// A dispatch of `state`, of `threads` threads a threadgroup with the threadgroup memory
/// `lengths`, as the world notes its make ahead.
fn prepared(
    state: &Fk<kind::ComputePipeline>,
    dispatch: Dispatch,
    threads: MtlSize,
    lengths: &[(u64, u64)],
) -> Prepared {
    let size = |s: MtlSize| (s.width, s.height, s.depth);
    let dispatch = match dispatch {
        Dispatch::Threads(s) => PreparedGrid::Threads(s.width, s.height, s.depth),
        Dispatch::Threadgroups(s) => PreparedGrid::Threadgroups(s.width, s.height, s.depth),
        Dispatch::ThreadgroupsIndirect => PreparedGrid::ThreadgroupsIndirect,
    };
    Prepared {
        state: state.id(),
        dispatch,
        threads: size(threads),
        lengths: lengths.to_vec(),
    }
}

/// The fake's name for [`ComputePipeline::texture_slot_writes`] and
/// [`RenderPipeline::texture_slot_writes`] in the call log and the script, which no selector of
/// Metal's is: a render pipeline's call takes its function's [`RenderFunction`] as its index in
/// the enum's order, then the slot.
pub const TEXTURE_SLOT_WRITES: &str = "textureSlotWrites";

impl RenderPipeline for Fk<kind::RenderPipeline> {
    fn gpu_resource_id(&self) -> u64 {
        self.gpu_resource_id_value()
    }

    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        self.thr(
            Policy::Caught,
            "imageblockMemoryLengthForDimensions:",
            vec![Arg::val(&imageblock_dimensions)],
            Default::default,
        )
    }

    fn imageblock_sample_length(&self) -> u64 {
        0
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        1024
    }

    fn max_total_threads_per_object_threadgroup(&self) -> u64 {
        self.with_model(|m: &PipelineModel| m.mesh.object_threads)
    }

    fn max_total_threads_per_mesh_threadgroup(&self) -> u64 {
        self.with_model(|m: &PipelineModel| m.mesh.mesh_threads)
    }

    fn object_thread_execution_width(&self) -> u64 {
        self.with_model(|m: &PipelineModel| m.mesh.object_width)
    }

    fn mesh_thread_execution_width(&self) -> u64 {
        self.with_model(|m: &PipelineModel| m.mesh.mesh_width)
    }

    fn max_total_threadgroups_per_mesh_grid(&self) -> u64 {
        self.with_model(|m: &PipelineModel| m.mesh.grid)
    }

    fn resource_index(&self) -> u64 {
        self.resource_index_value()
    }

    fn support_indirect_command_buffers(&self) -> bool {
        true
    }

    fn threadgroup_size_matches_tile_size(&self) -> bool {
        true
    }

    fn unique_identifier(&self) -> u64 {
        self.unique_id_value()
    }

    /// Logged as [`TEXTURE_SLOT_WRITES`]; the script's answer, else none known.
    fn texture_slot_writes(&self, function: RenderFunction, index: u64) -> Option<bool> {
        let function = match function {
            RenderFunction::Vertex => 0,
            RenderFunction::Fragment => 1,
            RenderFunction::Tile => 2,
            RenderFunction::Object => 3,
            RenderFunction::Mesh => 4,
        };
        let args = vec![Arg::U64(function), Arg::U64(index)];
        self.get_logged(TEXTURE_SLOT_WRITES, args, || None)
    }
}

impl Library<FakePlatform> for Fk<kind::Library> {
    fn function_names(&self) -> Vec<String> {
        match &self.state().model {
            Model::Names(n) => n.clone(),
            _ => Vec::new(),
        }
    }

    fn new_function(
        &self,
        on: OnFault,
        function_name: &str,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Function>> {
        let name = function_name.to_owned();
        self.make(
            Policy::On(on),
            "newFunctionWithName:",
            vec![Arg::Str(function_name.to_owned())],
            |w| Some(w.new_obj::<kind::Function>(Model::Names(vec![name]))),
        )
    }

    fn new_function_with_constants(
        &self,
        on: OnFault,
        function_name: &str,
        constants: &[crate::core::platform::resource::FunctionConstant],
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Function>> {
        let name = function_name.to_owned();
        let mut args = vec![Arg::Str(function_name.to_owned())];
        for c in constants {
            args.push(Arg::U64(u64::from(c.index)));
            args.push(Arg::U64(u64::from(c.data_type)));
            args.push(Arg::Bytes(c.value.to_vec()));
        }
        self.make(
            Policy::On(on),
            "newFunctionWithName:constantValues:error:",
            args,
            |w| Some(w.new_obj::<kind::Function>(Model::Names(vec![name]))),
        )
    }

    fn new_function_with_options(
        &self,
        on: OnFault,
        function_name: &str,
        constants: &[crate::core::platform::resource::FunctionConstant],
        options: u64,
        specialized_name: Option<&str>,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Function>> {
        let name = specialized_name.unwrap_or(function_name).to_owned();
        let mut args = vec![
            Arg::Str(function_name.to_owned()),
            Arg::U64(options),
            Arg::Str(specialized_name.unwrap_or_default().to_owned()),
        ];
        for c in constants {
            args.push(Arg::U64(u64::from(c.index)));
            args.push(Arg::U64(u64::from(c.data_type)));
            args.push(Arg::Bytes(c.value.to_vec()));
        }
        self.make(
            Policy::On(on),
            "newFunctionWithDescriptor:error:",
            args,
            |w| Some(w.new_obj::<kind::Function>(Model::Names(vec![name]))),
        )
    }
}

impl Fk<kind::Texture> {
    /// Makes `protectionOptions` answer `options`.
    pub fn set_protection_options(&self, options: u64) {
        self.with_model_mut(|m: &mut TextureModel| m.desc.protection_options = options);
    }

    /// Makes `storageMode` answer `mode`: bits 4-7 of the resource options.
    pub fn set_storage_mode(&self, mode: u64) {
        self.with_model_mut(|m: &mut TextureModel| {
            let options = m.desc.resource_options;
            m.desc.resource_options = (options & !0xf0) | (mode.wrapping_shl(4) & 0xf0);
        });
    }
}

impl Fk<kind::Heap> {
    /// `f` of the heap's model; the default of `T` for a heap a test made with no model.
    fn heap<T: Default>(&self, f: impl FnOnce(&super::super::objects::HeapModel) -> T) -> T {
        match &self.state().model {
            Model::Heap(m) => f(m),
            _ => T::default(),
        }
    }
}

/// A compute or render pipeline state's model.
#[derive(Debug, Default)]
pub struct PipelineModel {
    /// Common resource state.
    pub common: ResourceCommon,
    /// A compute pipeline's `staticThreadgroupMemoryLength`.
    pub static_threadgroup_memory_length: u64,
    /// A mesh render pipeline's answers, in the order of [`MeshLimits`]'s fields (all 0 for any
    /// other pipeline).
    pub mesh: MeshLimits,
}

/// What a fake mesh render pipeline answers for its stages' limits.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MeshLimits {
    /// `maxTotalThreadsPerObjectThreadgroup`.
    pub object_threads: u64,
    /// `maxTotalThreadsPerMeshThreadgroup`.
    pub mesh_threads: u64,
    /// `objectThreadExecutionWidth`.
    pub object_width: u64,
    /// `meshThreadExecutionWidth`.
    pub mesh_width: u64,
    /// `maxTotalThreadgroupsPerMeshGrid`.
    pub grid: u64,
}

impl Fk<kind::ComputePipeline> {
    /// Makes `staticThreadgroupMemoryLength` answer `length`.
    pub fn set_static_threadgroup_memory_length(&self, length: u64) {
        self.with_model_mut(|m: &mut PipelineModel| m.static_threadgroup_memory_length = length);
    }
}
