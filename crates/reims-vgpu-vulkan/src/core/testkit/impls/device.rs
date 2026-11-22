//! Fake: the `platform/device.rs` traits over the fake models (architecture 13.2).
//!
//! Each creator logs its selector and arguments (`fake::calllog`) and consults the script
//! (`fake::script`) before its model; a value's creator logs the value. The device's
//! capabilities come from its [`DeviceModel`], neither logged nor scripted.

use crate::core::testkit::script::{Policy, Step};
use std::sync::Arc;

use super::super::calllog::Arg;
use super::super::guest_ram::{PAGE_SIZE, RawRegion};
use super::super::objects::{DescDeallocator, FAKE_MAX_BYTES, Fk, HeapModel, Model, Space, kind};
use super::super::platform::FakePlatform;
use super::fatal;
use super::icb::IcbModel;
use super::rate_map::RateMapModel;
use super::resource::{MeshLimits, PipelineModel};
use super::{Backing, common, new_buffer, new_state, new_texture};
use crate::core::num;
use crate::core::platform::caps::{
    self, DeviceCaps, FormatAnswers, IdentityCaps, IdentityFlags, IdentitySizes,
};
use crate::core::platform::device::*;
use crate::core::platform::stage::{self, Site};
use crate::core::platform::types::{
    ArgumentKind, ArgumentLayout, ArgumentMember, MtlSize, NsErrorText, PixelFormatInfo,
    SizeAndAlign, TextureLevelInfo, TextureLevelQuery,
};
use crate::core::platform::values::{
    ArgumentDesc, BufferDesc, DepthStencilDesc, FullComputePipelineDesc,
    FullMeshRenderPipelineDesc, FullRenderPipelineDesc, HeapDesc, HeapMade, IcbDesc, OptionSet,
    PagesInPlace, PagesPlane, RenderPipelineDesc, SamplerDesc,
};
use crate::core::platform::{HostRange, Platform};
use crate::core::raise::{OnFault, RaiseResult};

/// The name the fake logs and scripts [`GpuDevice::new_command_queue_unshared`] under: no selector
/// of Metal's names it.
pub const UNSHARED_QUEUE: &str = "newCommandQueue (unshared)";

/// The name the fake logs and scripts [`GpuDevice::new_buffer_of_length_padding_unobserved`]
/// under: no selector of Metal's names it.
pub const PADDING_UNOBSERVED_BUFFER: &str = "newBufferWithLength:options: (padding unobserved)";

/// The name the fake logs and scripts `GpuDevice::new_dma_buf_buffer_of_length_padding_unobserved`
/// under: no selector of Metal's names it.
pub const DMA_BUF_BUFFER: &str = "newBufferWithLength:options: (dma-buf, padding unobserved)";

/// The name the fake logs and scripts `GpuDevice::new_exportable_buffer_of_length` under: no
/// selector of Metal's names it.
pub const EXPORTABLE_BUFFER: &str = "newBufferWithLength:options: (exportable)";

/// The name the fake logs and scripts `GpuDevice::new_deferred_exportable_buffer_of_length` under
/// for `SurfaceHandle::OpaqueFd`: no selector of Metal's names it.
pub const DEFERRED_EXPORTABLE_BUFFER: &str = "newBufferWithLength:options: (deferred, exportable)";

/// The name the fake logs and scripts `GpuDevice::new_deferred_exportable_buffer_of_length` under
/// for `SurfaceHandle::DmaBuf`: no selector of Metal's names it.
pub const DEFERRED_DMA_BUF_BUFFER: &str =
    "newBufferWithLength:options: (deferred, dma-buf, padding unobserved)";

/// The name the fake logs and scripts each plane's texture of `GpuDevice::new_exported_surface`
/// under, where its world makes such surfaces ([`World::give_exported_surfaces`]), with the
/// plane's pixel format, width, height and usage: no selector of Metal's names it.
///
/// [`World::give_exported_surfaces`]: crate::core::testkit::objects::World::give_exported_surfaces
#[cfg(unix)]
pub const EXPORTED_SURFACE_PLANE: &str = "newExportedSurface (a plane)";

/// What the fake's exported surface lies in ([`EXPORTED_SURFACE_PLANE`]): memory that exports
/// nothing, its planes in place unless its world makes the copy form
/// ([`World::copy_exported_surfaces`]), laid at the description's pitch and starts.
///
/// [`World::copy_exported_surfaces`]: crate::core::testkit::objects::World::copy_exported_surfaces
#[cfg(unix)]
struct FakeSurfaceMemory {
    handle: crate::core::platform::resource::SurfaceHandle,
    in_place: bool,
    layout: Vec<crate::core::platform::resource::SurfacePlaneLayout>,
}

#[cfg(unix)]
impl FakeSurfaceMemory {
    /// The memory of `desc`'s planes: one pitch, the widest plane's row rounded up to its
    /// multiple, the first plane at 0 and each after it on its start multiple past the one
    /// before and its rows of room; in place as `in_place` says.
    fn of(desc: &crate::core::platform::resource::SurfaceDesc<'_>, in_place: bool) -> Self {
        use crate::core::platform::resource::SurfacePlaneLayout;
        // `MTLPixelFormatR8Unorm`, `RG8Unorm`, `R16Unorm` and `RG16Unorm`'s texel bytes; 4 for
        // any other.
        let texel = |format: u64| match format {
            10 => 1,
            30 | 20 => 2,
            _ => 4,
        };
        let row =
            |p: &crate::core::platform::resource::SurfacePlaneDesc| p.width * texel(p.pixel_format);
        let widest = desc.planes.iter().map(row).max().unwrap_or(0);
        let pitch = widest.next_multiple_of(desc.pitch_multiple.max(1));
        let mut offset: u64 = 0;
        let layout = (desc.planes.iter())
            .map(|p| {
                let at = offset.next_multiple_of(desc.start_multiple.max(1));
                offset = at + pitch * (p.height + u64::from(p.rows_below));
                SurfacePlaneLayout {
                    offset: at,
                    row_pitch: pitch,
                }
            })
            .collect();
        FakeSurfaceMemory {
            handle: desc.handle,
            in_place,
            layout,
        }
    }
}

#[cfg(unix)]
impl crate::core::platform::resource::SurfaceMemory for FakeSurfaceMemory {
    fn export(&self) -> Result<crate::core::platform::resource::ExportedMemory, String> {
        Err("the fake's surface exports no memory".to_owned())
    }

    fn external_writes_done(&self) -> Result<(), String> {
        Ok(())
    }

    fn handle(&self) -> crate::core::platform::resource::SurfaceHandle {
        self.handle
    }

    fn in_place(&self) -> bool {
        self.in_place
    }

    fn layout(&self) -> &[crate::core::platform::resource::SurfacePlaneLayout] {
        &self.layout
    }
}

/// The name the fake logs and scripts [`GpuDevice::new_deferred_buffer_of_length`] under: no
/// selector of Metal's names it.
pub const DEFERRED_BUFFER: &str = "newBufferWithLength:options: (deferred)";

/// The name the fake logs and scripts [`GpuDevice::new_scratch_buffer_of_length`] under: no
/// selector of Metal's names it.
pub const SCRATCH_BUFFER: &str = "newBufferWithLength:options: (scratch)";

/// Each byte of a scratch buffer the fake makes ([`SCRATCH_BUFFER`]) before its maker writes it,
/// as a platform that zeroes none leaves them: a byte the maker did not write reads as this, not
/// as zero.
pub const SCRATCH_MARKER: u8 = 0xa5;

/// A device's model: its capabilities.
#[derive(Debug)]
pub struct DeviceModel {
    /// What `caps` answers.
    pub caps: Arc<DeviceCaps>,
    /// What `identity_caps` answers for `hasUnifiedMemory`
    /// ([`set_unified_memory`](Fk::set_unified_memory)).
    pub unified_memory: bool,
    /// What [`is_integrated`](GpuDevice::is_integrated) answers
    /// ([`set_integrated`](Fk::set_integrated)).
    pub integrated: bool,
}

impl Default for DeviceModel {
    /// The fake's default capabilities ([`fake_caps`]), of an integrated GPU, which shares the
    /// CPU's memory.
    fn default() -> DeviceModel {
        DeviceModel {
            caps: Arc::new(fake_caps()),
            unified_memory: true,
            integrated: true,
        }
    }
}

impl GpuDevice<FakePlatform> for Fk<kind::Device> {
    fn allow_foreign_libraries(&self) {
        fatal(self.act(Policy::Fatal, "allowLibrariesFromOtherPlatforms", vec![]));
    }

    /// The fake's layout: each argument's elements in turn, 8 bytes each, in the order given;
    /// `None` for no arguments.
    fn argument_layout(&self, arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>> {
        let shown: Vec<(u64, u64, u64)> = arguments
            .iter()
            .map(|a| (a.data_type, a.index, a.array_length))
            .collect();
        self.thr(
            Policy::Caught,
            "newArgumentEncoderWithArguments:",
            vec![Arg::val(&shown)],
            || fake_argument_layout(arguments),
        )
    }

    /// The fake's layout of buffer index 0 of any function: a texture at index 0, then a
    /// pointer at index 1 to an argument buffer of one sampler; `None` at every other index.
    fn function_argument_layout(
        &self,
        function: &<FakePlatform as Platform>::Function,
        buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>> {
        self.thr(
            Policy::Caught,
            "newArgumentEncoderWithBufferIndex:",
            vec![Arg::obj(function), Arg::val(&buffer_index)],
            || (buffer_index == 0).then(fake_function_layout),
        )
    }

    fn heap_texture_size_and_align(
        &self,
        desc: &<FakePlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        let t = desc.desc();
        let bytes = [t.height, t.depth, t.array_length, t.sample_count, 4]
            .iter()
            .fold(t.width, |a, b| a.wrapping_mul(*b));
        self.thr(
            Policy::Caught,
            "heapTextureSizeAndAlignWithDescriptor:",
            vec![Arg::obj(desc)],
            || SizeAndAlign {
                size: num::align_up_wrapping(bytes, 0x4000),
                align: 0x4000,
            },
        )
    }

    fn new_buffer_over_host_memory(
        &self,
        on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        let (pointer, length) = (bytes.ptr(), bytes.len());
        let mut deallocator = deallocator;
        let made = self.make(
            Policy::On(on),
            "newBufferWithBytesNoCopy:length:options:deallocator:",
            vec![
                Arg::Ptr(pointer.addr()),
                Arg::U64(length),
                Arg::U64(options),
                Arg::Block(deallocator.is_some()),
            ],
            |w| {
                let b = new_buffer(w, length, options, Backing::At(pointer), None)?;
                if let Model::Buffer(m) = &mut b.state().model {
                    m.deallocator = deallocator.take();
                }
                Some(b)
            },
        );
        // Refused: dropped unrun here, with no lock of the world's held, as the Vulkan platform
        // drops it (core's `NoCopyDeallocator`).
        drop(deallocator);
        made
    }

    fn new_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        self.make(
            Policy::On(on),
            "newBufferWithLength:options:",
            vec![Arg::U64(length), Arg::U64(options)],
            |w| new_buffer(w, length, options, Backing::Fresh, None),
        )
    }

    /// Logged and scripted under [`PADDING_UNOBSERVED_BUFFER`], as no selector of Metal's names
    /// it, with the aligned byte after the length and options (0 for none); its buffer is
    /// `newBufferWithLength:options:`'s, its memory starting where the fake's starts.
    fn new_buffer_of_length_padding_unobserved(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        self.make(
            Policy::On(on),
            PADDING_UNOBSERVED_BUFFER,
            vec![
                Arg::U64(length),
                Arg::U64(options),
                Arg::U64(aligned_byte.unwrap_or(0)),
            ],
            |w| new_buffer(w, length, options, Backing::Fresh, None),
        )
    }

    /// Logged and scripted under [`DEFERRED_BUFFER`], as no selector of Metal's names it; its
    /// buffer is `newBufferWithLength:options:`'s, its memory not made until
    /// [`Buffer::make_memory`](crate::core::platform::resource::Buffer::make_memory) makes it
    /// ([`BufferModel::deferred`](crate::core::testkit::objects::BufferModel::deferred)).
    fn new_deferred_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        self.make(
            Policy::On(on),
            DEFERRED_BUFFER,
            vec![Arg::U64(length), Arg::U64(options)],
            |w| {
                let b = new_buffer(w, length, options, Backing::Fresh, None)?;
                if let Model::Buffer(m) = &mut b.state().model {
                    m.deferred = Some(false);
                }
                Some(b)
            },
        )
    }

    /// Logged and scripted under [`SCRATCH_BUFFER`], as no selector of Metal's names it; its
    /// buffer is `newBufferWithLength:options:`'s, each of its bytes [`SCRATCH_MARKER`] where it
    /// has any, not zero.
    fn new_scratch_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        self.make(
            Policy::On(on),
            SCRATCH_BUFFER,
            vec![Arg::U64(length), Arg::U64(options)],
            |w| {
                let b = new_buffer(w, length, options, Backing::Fresh, None)?;
                if let Model::Buffer(m) = &b.state().model
                    && let Some(mem) = &m.mem
                {
                    mem.write(0, &vec![SCRATCH_MARKER; num::ToUsize::to_usize(mem.len())]);
                }
                Some(b)
            },
        )
    }

    /// Logged and scripted under [`EXPORTABLE_BUFFER`], as no selector of Metal's names it, with
    /// the aligned byte after the length and options (0 for none); its buffer is
    /// `newBufferWithLength:options:`'s, which exports nothing.
    #[cfg(unix)]
    fn new_exportable_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        self.make(
            Policy::On(on),
            EXPORTABLE_BUFFER,
            vec![
                Arg::U64(length),
                Arg::U64(options),
                Arg::U64(aligned_byte.unwrap_or(0)),
            ],
            |w| new_buffer(w, length, options, Backing::Fresh, None),
        )
    }

    /// Logged and scripted under [`DMA_BUF_BUFFER`], as no selector of Metal's names it, with the
    /// aligned byte after the length and options (0 for none); its buffer is
    /// `newBufferWithLength:options:`'s, which exports nothing.
    #[cfg(unix)]
    fn new_dma_buf_buffer_of_length_padding_unobserved(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        self.make(
            Policy::On(on),
            DMA_BUF_BUFFER,
            vec![
                Arg::U64(length),
                Arg::U64(options),
                Arg::U64(aligned_byte.unwrap_or(0)),
            ],
            |w| new_buffer(w, length, options, Backing::Fresh, None),
        )
    }

    /// Logged and scripted under [`DEFERRED_EXPORTABLE_BUFFER`] (`SurfaceHandle::OpaqueFd`) or
    /// [`DEFERRED_DMA_BUF_BUFFER`] (`SurfaceHandle::DmaBuf`), as no selector of Metal's names it,
    /// with the aligned byte after the length and options (0 for none); its buffer is
    /// [`DEFERRED_BUFFER`]'s, its memory not made until
    /// [`Buffer::make_memory`](crate::core::platform::resource::Buffer::make_memory) makes it,
    /// which exports nothing.
    #[cfg(unix)]
    fn new_deferred_exportable_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
        handle: crate::core::platform::resource::SurfaceHandle,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        let sel = match handle {
            crate::core::platform::resource::SurfaceHandle::OpaqueFd => DEFERRED_EXPORTABLE_BUFFER,
            crate::core::platform::resource::SurfaceHandle::DmaBuf => DEFERRED_DMA_BUF_BUFFER,
        };
        self.make(
            Policy::On(on),
            sel,
            vec![
                Arg::U64(length),
                Arg::U64(options),
                Arg::U64(aligned_byte.unwrap_or(0)),
            ],
            |w| {
                let b = new_buffer(w, length, options, Backing::Fresh, None)?;
                if let Model::Buffer(m) = &mut b.state().model {
                    m.deferred = Some(false);
                }
                Some(b)
            },
        )
    }

    /// `None`, unlogged, as the platform's default, unless the world makes such surfaces
    /// ([`World::give_exported_surfaces`]): then each plane a Private texture of the plane's
    /// format, size and usage, logged and scripted under [`EXPORTED_SURFACE_PLANE`], in memory
    /// that exports nothing, in place unless the world makes the copy form
    /// ([`FakeSurfaceMemory`]); `None` where a plane's make answers none.
    ///
    /// [`World::give_exported_surfaces`]: crate::core::testkit::objects::World::give_exported_surfaces
    #[cfg(unix)]
    fn new_exported_surface(
        &self,
        on: OnFault,
        desc: &crate::core::platform::resource::SurfaceDesc<'_>,
    ) -> RaiseResult<Option<crate::core::platform::resource::ExportedSurface<FakePlatform>>> {
        if !self.world().gives_exported_surfaces() {
            return Ok(None);
        }
        let mut planes = Vec::with_capacity(desc.planes.len());
        for plane in desc.planes {
            let texture = super::super::objects::TexDesc {
                pixel_format: plane.pixel_format,
                width: plane.width,
                height: plane.height,
                usage: plane.usage,
                resource_options: 2 << 4,
                ..super::super::objects::TexDesc::default()
            };
            let args = [plane.pixel_format, plane.width, plane.height, plane.usage];
            let made = self.make(
                Policy::On(on),
                EXPORTED_SURFACE_PLANE,
                args.iter().copied().map(Arg::U64).collect(),
                |w| Some(new_texture(w, texture, None, false)),
            )?;
            let Some(made) = made else {
                return Ok(None);
            };
            planes.push(made);
        }
        let in_place = !self.world().copies_exported_surfaces();
        let memory = Arc::new(FakeSurfaceMemory::of(desc, in_place));
        Ok(Some(crate::core::platform::resource::ExportedSurface {
            planes,
            memory,
        }))
    }

    fn new_command_queue(&self) -> Option<<FakePlatform as Platform>::CommandQueue> {
        fatal(self.make(Policy::Fatal, "newCommandQueue", vec![], |w| {
            Some(w.new_obj::<kind::CommandQueue>(Model::None))
        }))
    }

    /// Logged and scripted under [`UNSHARED_QUEUE`], as no selector of Metal's names it; its
    /// queue is `newCommandQueue`'s.
    fn new_command_queue_unshared(&self) -> Option<<FakePlatform as Platform>::CommandQueue> {
        fatal(self.make(Policy::Fatal, UNSHARED_QUEUE, vec![], |w| {
            Some(w.new_obj::<kind::CommandQueue>(Model::None))
        }))
    }

    /// Unlogged, as the property the platform answers it from: [`World::copy_guest_ram`].
    ///
    /// [`World::copy_guest_ram`]: crate::core::testkit::objects::World::copy_guest_ram
    fn copies_guest_ram(&self) -> bool {
        self.world().copies_guest_ram()
    }

    /// Unlogged, as the world answers it: [`World::recover`].
    ///
    /// [`World::recover`]: crate::core::testkit::objects::World::recover
    fn recoveries(&self) -> u64 {
        self.world().recoveries()
    }

    /// Unlogged: suspends the world's stage fills, as the rt platform suspends its device's.
    fn reset_begins(&self) {
        self.world().stages().fills().suspend();
    }

    /// Unlogged: resumes the world's stage fills.
    fn reset_ends(&self) {
        self.world().stages().fills().resume();
    }

    /// Unlogged, as the world answers it: [`World::answer_surface_pages`].
    ///
    /// [`World::answer_surface_pages`]: crate::core::testkit::objects::World::answer_surface_pages
    fn surface_pages_in_place(
        &self,
        _pages: &<FakePlatform as Platform>::Buffer,
        planes: &[PagesPlane],
    ) -> PagesInPlace {
        self.world().surface_pages_in_place(planes)
    }

    /// Unlogged, as the world answers it: [`World::answer_surface_pages_unmade`].
    ///
    /// [`World::answer_surface_pages_unmade`]: crate::core::testkit::objects::World::answer_surface_pages_unmade
    fn surface_pages_never_in_place(
        &self,
        planes: &[PagesPlane],
        pages: &[std::ops::Range<u64>],
    ) -> Option<String> {
        self.world().surface_pages_never_in_place(planes, pages)
    }

    /// Unlogged, as the world answers it: [`World::answer_fresh_surfaces`].
    ///
    /// [`World::answer_fresh_surfaces`]: crate::core::testkit::objects::World::answer_fresh_surfaces
    fn fresh_surface_never_in_place(&self, planes: &[PagesPlane], length: u64) -> Option<String> {
        self.world().fresh_surface_never_in_place(planes, length)
    }

    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&<FakePlatform as Platform>::Function>,
    ) -> Result<Option<<FakePlatform as Platform>::ComputePipeline>, NsErrorText> {
        fatal(self.make_or_error(
            Policy::Fatal,
            "newComputePipelineStateWithFunction:error:",
            vec![Arg::opt(compute_function)],
            |w| {
                Some(new_pipeline::<kind::ComputePipeline>(
                    w,
                    Space::COMPUTE_PIPELINE,
                    None,
                ))
            },
        ))
    }

    fn new_event(&self, on: OnFault) -> RaiseResult<Option<<FakePlatform as Platform>::Event>> {
        self.make(Policy::On(on), "newEvent", vec![], |w| {
            Some(w.new_obj::<kind::Event>(Model::Event { value: 0 }))
        })
    }

    fn new_fence(&self) -> RaiseResult<Option<<FakePlatform as Platform>::Fence>> {
        self.make(Policy::Caught, "newFence", vec![], |w| {
            Some(w.new_obj::<kind::Fence>(Model::None))
        })
    }

    fn new_library(
        &self,
        data: &[u8],
    ) -> RaiseResult<Result<Option<<FakePlatform as Platform>::Library>, NsErrorText>> {
        self.make_or_error(
            Policy::Caught,
            "newLibraryWithData:error:",
            vec![Arg::Bytes(data.to_vec())],
            |w| Some(w.new_obj::<kind::Library>(Model::Names(w.library_functions()))),
        )
    }

    /// A new dynamic library each call, of any library: the fake checks nothing of Metal's.
    fn new_dynamic_library(
        &self,
        library: &<FakePlatform as Platform>::Library,
    ) -> RaiseResult<Result<Option<<FakePlatform as Platform>::DynamicLibrary>, NsErrorText>> {
        self.make_or_error(
            Policy::Caught,
            "newDynamicLibrary:error:",
            vec![Arg::obj(library)],
            |w| Some(w.new_obj::<kind::DynamicLibrary>(Model::None)),
        )
    }

    /// A new library each call, holding the world's function names; the Vulkan platform's loads
    /// once per device, which no core behaviour depends on.
    fn new_shader_library(
        &self,
    ) -> Result<Option<<FakePlatform as Platform>::Library>, NsErrorText> {
        fatal(self.make_or_error(
            Policy::Fatal,
            "newLibraryWithSource:options:error:",
            vec![],
            |w| Some(w.new_obj::<kind::Library>(Model::Names(w.library_functions()))),
        ))
    }

    fn new_rate_map(
        &self,
        descriptor: &<FakePlatform as Platform>::RateMapDescriptor,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::RateMap>> {
        let (layers, desc) = (descriptor.layers_set(), descriptor.desc());
        self.make(
            Policy::Caught,
            "newRasterizationRateMapWithDescriptor:",
            vec![Arg::obj(descriptor)],
            |w| {
                Some(w.new_obj::<kind::RateMap>(Model::RateMap(RateMapModel {
                    common: common(w, None),
                    layer_count: layers,
                    screen_size: desc.screen_size,
                    mutability: desc.mutability.unwrap_or(0),
                    ..RateMapModel::default()
                })))
            },
        )
    }

    fn new_shared_event(&self) -> Option<<FakePlatform as Platform>::SharedEvent> {
        fatal(self.make(Policy::Fatal, "newSharedEvent", vec![], |w| {
            Some(w.new_obj::<kind::SharedEvent>(Model::Event { value: 0 }))
        }))
    }

    fn new_shared_texture(
        &self,
        descriptor: &<FakePlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let desc = descriptor.desc();
        self.make(
            Policy::Caught,
            "newSharedTextureWithDescriptor:",
            vec![Arg::obj(descriptor)],
            |w| Some(new_texture(w, desc, None, true)),
        )
    }

    fn open_shared_texture(
        &self,
        handle: Option<&<FakePlatform as Platform>::SharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let target = handle.and_then(|h| match &h.state().model {
            Model::Target(t) => Some(t.clone_ref()),
            _ => None,
        });
        self.make(
            Policy::Caught,
            "newSharedTextureWithHandle:withResourceIndex:",
            vec![Arg::opt(handle), Arg::U64(resource_index)],
            |w| {
                let t = target?;
                let mut desc = t
                    .cast::<kind::Texture>()
                    .and_then(|tx| match &tx.state().model {
                        Model::Texture(m) => Some(m.desc),
                        _ => None,
                    })
                    .unwrap_or_default();
                desc.resource_index = Some(resource_index);
                desc.force_resource_index = true;
                Some(new_texture(w, desc, Some(t), true))
            },
        )
    }

    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &<FakePlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Texture>> {
        let desc = descriptor.desc();
        self.make(
            Policy::On(on),
            "newTextureWithDescriptor:",
            vec![Arg::obj(descriptor)],
            |w| Some(new_texture(w, desc, None, false)),
        )
    }

    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "reserveResourceIndicesForResourceType:indices:indexCount:",
            vec![Arg::U64(resource_type), Arg::U64s(indices.to_vec())],
        )?;
        self.world().reserve_indices(Space(resource_type), indices);
        Ok(())
    }

    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo {
        self.get_logged(
            "MTLPixelFormatGetInfoForDevice()",
            vec![Arg::U64(format)],
            || PixelFormatInfo {
                name: Some(format!("MTLPixelFormat{format}")),
                flags: 0,
                cast_class: 0,
                block_bytes: 4,
                block_width: 1,
                block_height: 1,
                block_depth: 1,
            },
        )
    }

    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo {
        self.get_logged(
            "MTLGetTextureLevelInfoForDeviceWithOptions()",
            vec![Arg::val(q)],
            || TextureLevelInfo {
                aligned_height: q.height,
                bytes_per_pixel: 4,
                bytes_per_row: q.width.wrapping_mul(4),
                ..TextureLevelInfo::default()
            },
        )
    }

    /// The caps [`set_caps`](Fk::set_caps) set, else [`fake_caps`].
    fn caps(&self) -> Arc<DeviceCaps> {
        self.with_model(|m: &DeviceModel| m.caps.clone())
    }

    /// [`fake_identity_caps`], with `hasUnifiedMemory` as
    /// [`set_unified_memory`](Fk::set_unified_memory) set it (yes by default), neither logged
    /// nor scripted.
    fn identity_caps(&self) -> IdentityCaps {
        let mut caps = fake_identity_caps();
        caps.flags.has_unified_memory = Some(self.with_model(|m: &DeviceModel| m.unified_memory));
        caps
    }

    /// No: the fake device takes no private render state, unlogged.
    fn takes_private_render_state(&self) -> bool {
        false
    }

    /// The identity's working set ([`FAKE_OWN_MEMORY`]), unlogged.
    fn own_memory_budget(&self) -> u64 {
        FAKE_OWN_MEMORY
    }

    /// [`FAKE_MEMORY_HEAPS`], unlogged.
    fn memory_heaps_total(&self) -> u64 {
        FAKE_MEMORY_HEAPS
    }

    /// [`FAKE_LINEAR_ROW_ALIGNMENT`], unlogged.
    fn linear_row_alignment(&self) -> u64 {
        FAKE_LINEAR_ROW_ALIGNMENT
    }

    /// As [`set_integrated`](Fk::set_integrated) set it (yes by default), unlogged.
    fn is_integrated(&self) -> bool {
        self.with_model(|m: &DeviceModel| m.integrated)
    }

    /// `row_bytes` rounded up to the world's linear rows' alignment where it has one
    /// ([`World::lay_linear_rows`](super::super::objects::World::lay_linear_rows)), else `None`;
    /// unlogged.
    fn linear_rows_pitch(&self, row_bytes: u64, rows: u64) -> Option<u64> {
        let align = self.world().linear_rows()?;
        (row_bytes != 0 && rows != 0).then(|| row_bytes.next_multiple_of(align))
    }

    /// No: the fake device makes no pipeline of OpenCL mode, unlogged.
    fn makes_opencl_mode_pipelines(&self) -> bool {
        false
    }

    /// The value's options folded as Metal's buffer descriptor folds them ([`fold_options`]), its
    /// contents, pin and deallocator.
    fn new_buffer(
        &self,
        desc: BufferDesc<'_>,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Buffer>> {
        let arg = Arg::val(&desc);
        let opts = fold_options(&desc.options);
        let BufferDesc {
            length,
            pinned_gpu_address,
            contents,
            mut deallocator,
            ..
        } = desc;
        let made = self.make(Policy::Caught, "newBufferWithDescriptor:", vec![arg], |w| {
            let backing = contents.map_or(Backing::Fresh, |c| Backing::At(c.ptr()));
            let b = new_buffer(w, length, opts, backing, w.kept_pin(pinned_gpu_address))?;
            if let Model::Buffer(m) = &mut b.state().model {
                m.desc_deallocator = deallocator.take().map(DescDeallocator);
            }
            Some(b)
        });
        // Refused: dropped unrun here, with no lock of the world's held, as the Vulkan platform
        // drops it (core's `NoCopyDeallocator`).
        drop(deallocator);
        made
    }

    /// A heap of the value's size, with its options folded as Metal's heap descriptor folds
    /// them ([`fold_options`]): memory of its own for a shared or managed heap.
    fn new_heap(&self, desc: &HeapDesc) -> RaiseResult<HeapMade<FakePlatform>> {
        let opts = fold_options(&desc.options);
        let (size, pinned) = (desc.size, desc.pinned_gpu_address);
        let heap = self.make(
            Policy::Caught,
            "newHeapWithDescriptor:",
            vec![Arg::val(desc)],
            |w| {
                let storage = opts.wrapping_shr(4) & 0xf;
                let mem = if storage < 2 && size <= FAKE_MAX_BYTES {
                    RawRegion::new(size)
                } else {
                    None
                };
                Some(w.new_obj::<kind::Heap>(Model::Heap(HeapModel {
                    size,
                    options: opts,
                    gpu_address: (w.kept_pin(pinned)).unwrap_or_else(|| w.alloc_gpu_va(size)),
                    mem,
                    deallocator: None,
                    common: common(w, None),
                })))
            },
        )?;
        Ok(HeapMade {
            heap,
            descriptor_resource_options: opts,
        })
    }

    /// A heap of the value over `bytes`, logged with the bytes' pointer and length first; it
    /// owns no memory of its own, and runs `deallocator` as it dies (or later, where the world
    /// runs deallocators late).
    fn new_heap_over_host_memory(
        &self,
        bytes: HostRange<'_>,
        desc: &HeapDesc,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<HeapMade<FakePlatform>> {
        let opts = fold_options(&desc.options);
        let (size, pinned) = (desc.size, desc.pinned_gpu_address);
        let heap = self.make(
            Policy::Caught,
            "newHeapWithDescriptor:",
            vec![
                Arg::Ptr(bytes.ptr().addr()),
                Arg::U64(bytes.len()),
                Arg::val(desc),
            ],
            |w| {
                Some(w.new_obj::<kind::Heap>(Model::Heap(HeapModel {
                    size,
                    options: opts,
                    gpu_address: (w.kept_pin(pinned)).unwrap_or_else(|| w.alloc_gpu_va(size)),
                    mem: None,
                    deallocator,
                    common: common(w, None),
                })))
            },
        )?;
        Ok(HeapMade {
            heap,
            descriptor_resource_options: opts,
        })
    }

    /// The value's resource index is the state's forced index.
    fn new_sampler_state(
        &self,
        desc: &SamplerDesc,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::SamplerState>> {
        self.make(
            Policy::Caught,
            "newSamplerStateWithDescriptor:",
            vec![Arg::val(desc)],
            |w| {
                Some(new_state::<kind::SamplerState>(
                    w,
                    Some((Space::SAMPLER, desc.resource_index)),
                ))
            },
        )
    }

    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::DepthStencilState>> {
        self.make(
            Policy::Caught,
            "newDepthStencilStateWithDescriptor:",
            vec![Arg::val(desc)],
            |w| {
                Some(new_state::<kind::DepthStencilState>(
                    w,
                    Some((Space::DEPTH_STENCIL, None)),
                ))
            },
        )
    }

    /// The value's resource index is the ICB's forced index.
    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::Icb>> {
        self.make(
            Policy::Caught,
            "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
            vec![Arg::val(desc), Arg::U64(max_count), Arg::U64(options)],
            |w| {
                Some(w.new_obj::<kind::Icb>(Model::Icb(IcbModel {
                    common: common(w, Some((Space::ICB, desc.resource_index))),
                    size: max_count,
                    options,
                })))
            },
        )
    }

    /// The value's resource index is forced when its flag is set.
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, FakePlatform>,
    ) -> Result<Option<<FakePlatform as Platform>::RenderPipeline>, NsErrorText> {
        let forced = desc
            .resource_index
            .and_then(|(index, force)| force.then_some(index));
        fatal(self.make_or_error(
            Policy::Fatal,
            "newRenderPipelineStateWithDescriptor:error:",
            vec![Arg::val(desc)],
            |w| {
                Some(new_pipeline::<kind::RenderPipeline>(
                    w,
                    Space::RENDER_PIPELINE,
                    forced,
                ))
            },
        ))
    }

    /// The whole value is the call's one argument.
    fn new_render_pipeline_state_full(
        &self,
        desc: &FullRenderPipelineDesc<'_, FakePlatform>,
    ) -> Result<Option<<FakePlatform as Platform>::RenderPipeline>, NsErrorText> {
        fatal(self.make_or_error(
            Policy::Fatal,
            "newRenderPipelineStateWithDescriptor:error:",
            vec![Arg::val(desc)],
            |w| {
                Some(new_pipeline::<kind::RenderPipeline>(
                    w,
                    Space::RENDER_PIPELINE,
                    None,
                ))
            },
        ))
    }

    /// The whole value is the call's first argument, then the options (0). The pipeline answers
    /// the descriptor's thread maxima, else 1024, with an execution width of 32 a stage, the
    /// object stage's answers 0 without an object function, and the grid's maximum
    /// `NSUIntegerMax`, as the x86 Macs answer (docs/graphics-kext/dgpu-oracle.md).
    fn new_mesh_render_pipeline_state_full(
        &self,
        desc: &FullMeshRenderPipelineDesc<'_, FakePlatform>,
    ) -> Result<Option<<FakePlatform as Platform>::RenderPipeline>, NsErrorText> {
        let or_1024 = |v: u64| if v == 0 { 1024 } else { v };
        let has_object = desc.object.is_some();
        let limits = MeshLimits {
            object_threads: if has_object {
                or_1024(desc.max_total_threads_per_object_threadgroup)
            } else {
                0
            },
            mesh_threads: or_1024(desc.max_total_threads_per_mesh_threadgroup),
            object_width: if has_object { 32 } else { 0 },
            mesh_width: 32,
            grid: u64::MAX,
        };
        fatal(self.make_or_error(
            Policy::Fatal,
            "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:",
            vec![Arg::val(desc), Arg::U64(0)],
            |w| {
                let p = new_pipeline::<kind::RenderPipeline>(w, Space::RENDER_PIPELINE, None);
                p.with_model_mut(|m: &mut PipelineModel| m.mesh = limits);
                Some(p)
            },
        ))
    }

    /// The whole value is the call's first argument, then the options (0). One refused for a
    /// stage ([`Step::Stage`]) asks for the stage's fill at the make and notes it for the make's
    /// caller ([`stage::refused_for`]), as the rt platform's make does, and answers
    /// [`stage_refusal`], a gap of the platform's own.
    fn new_compute_pipeline_state_full(
        &self,
        desc: &FullComputePipelineDesc<'_, FakePlatform>,
    ) -> Result<Option<<FakePlatform as Platform>::ComputePipeline>, NsErrorText> {
        let sel = "newComputePipelineStateWithDescriptor:options:reflection:error:";
        let step = self.call(sel, vec![Arg::val(desc), Arg::U64(0)]);
        if let Step::Stage(refused) = step {
            let fills = self.world().stages().fills();
            stage::refused_for(&fills.fill(refused, Site::Make));
            note_relayed_gap();
            return Err(stage_refusal(refused));
        }
        let w = self.world().clone();
        fatal(step.create_or_error(Policy::Fatal, sel, || {
            let made = new_pipeline::<kind::ComputePipeline>(&w, Space::COMPUTE_PIPELINE, None);
            Some(self.adopt(made))
        }))
    }
}

/// The `NSError` of a compute state's make refused for `stage`, whose compile is not in the
/// store ([`Step::Stage`]).
pub fn stage_refusal(stage: u64) -> NsErrorText {
    NsErrorText {
        domain: "FakeStageDomain".to_owned(),
        code: 0,
        description: format!("the fake refused it: stage {stage}'s compile is not in its store"),
    }
}

/// The resource options a descriptor reports after `options`, as the fake's buffer and heap
/// descriptors fold them from 0: a storage mode replaces bits 4-7, and a hazard-tracking mode
/// is kept apart.
pub fn fold_options(options: &[OptionSet]) -> u64 {
    options.iter().fold(0u64, |opts, o| match *o {
        OptionSet::ResourceOptions(v) => v,
        OptionSet::ResourceOptionsOr(bits) => opts | bits,
        OptionSet::StorageMode(m) => (opts & !0xf0) | (m.wrapping_shl(4) & 0xf0),
        OptionSet::HazardTrackingMode(_) => opts,
    })
}

impl Fk<kind::Device> {
    /// Sets the device's capabilities, which [`caps`](GpuDevice::caps) answers from then on.
    pub fn set_caps(&self, caps: DeviceCaps) {
        self.with_model_mut(|m: &mut DeviceModel| m.caps = Arc::new(caps));
    }

    /// Sets whether the device's GPU shares the CPU's memory, which
    /// [`identity_caps`](GpuDevice::identity_caps) answers for `hasUnifiedMemory` from then on:
    /// `false` makes the fake a GPU of its own memory.
    pub fn set_unified_memory(&self, unified: bool) {
        self.with_model_mut(|m: &mut DeviceModel| m.unified_memory = unified);
    }

    /// Sets whether the device's GPU is an integrated one, which
    /// [`is_integrated`](GpuDevice::is_integrated) answers from then on.
    pub fn set_integrated(&self, integrated: bool) {
        self.with_model_mut(|m: &mut DeviceModel| m.integrated = integrated);
    }
}

/// A new pipeline state with an index in `space` (`forced` when given).
fn new_pipeline<C: super::super::objects::Kind>(
    w: &Arc<super::super::objects::World>,
    space: Space,
    forced: Option<u64>,
) -> Fk<C> {
    w.new_obj(Model::Pipeline(PipelineModel {
        common: common(w, Some((space, forced))),
        ..PipelineModel::default()
    }))
}

/// The fake device's capabilities when no test sets them: an Apple-silicon device that answers
/// every query and supports every feature core asks about, but the 24-bit depth format and an
/// 8-sample texture.
pub fn fake_caps() -> DeviceCaps {
    DeviceCaps {
        name: Some(String::from("FakeMTLDevice")),
        registry_id: super::super::platform::FAKE_REGISTRY_ID,
        gpu_core_count: Some(10),
        max_threads_per_threadgroup: MtlSize::new(1024, 1024, 1024),
        max_vertex_amplification_count: 2,
        limits: caps::Limits {
            device_linear_texture_alignment_bytes: 16,
            iosurface_read_only_texture_alignment_bytes: 16,
            iosurface_texture_alignment_bytes: 16,
            linear_texture_alignment_bytes: 16,
            max_compute_buffers: 31,
            max_compute_local_memory_sizes: 32768,
            max_compute_samplers: 16,
            max_compute_textures: 128,
            max_compute_threadgroup_memory: 32768,
            max_compute_threadgroup_memory_alignment_bytes: 16,
            max_fragment_buffers: 31,
            max_fragment_samplers: 16,
            max_fragment_textures: 128,
            max_predicated_nesting_depth: 8,
            max_rasterization_rate_layer_count: 2,
            max_texture_height_2d: 16384,
            max_texture_layers: 2048,
            max_texture_width_2d: 16384,
            max_threadgroup_memory_length: 32768,
            max_tile_buffers: 31,
            max_tile_samplers: 16,
            max_tile_textures: 128,
            max_total_compute_threads_per_threadgroup: 1024,
            max_vertex_buffers: 31,
            max_vertex_samplers: 16,
            max_vertex_textures: 128,
            max_viewport_count: 16,
        },
        flags: caps::Flags {
            are_raster_order_groups_supported: true,
            is_depth24_stencil8_pixel_format_supported: false,
            is_framebuffer_read_supported: true,
            is_large_mrt_supported: true,
            is_rgb10a2_gamma_supported: true,
            supports_buffer_with_address_ranges: true,
            supports_buffer_with_iosurface: true,
            supports_command_buffer_jump: true,
            supports_compute_compressed_texture_write: true,
            supports_conditional_load_store: true,
            supports_dynamic_attribute_stride: true,
            supports_extended_xr10_formats: true,
            supports_float16_bcubic_filtering: true,
            supports_heap_with_address_ranges: true,
            supports_image_blocks: true,
            supports_memory_order_atomics: true,
            supports_native_hardware_fp16: true,
            supports_non_square_tile_shaders: true,
            supports_programmable_sample_positions: true,
            supports_resource_detach_backing: true,
            supports_shared_texture_handles: true,
            supports_simd_reduction: true,
            supports_simd_shuffle_and_fill: true,
            supports_texture_2d_multisample_array: true,
            supports_tile_shaders: true,
        },
        sample_counts: caps::SAMPLE_COUNTS.map(|n| Ok(matches!(n, 1 | 2 | 4))),
        primitive_types: [const { Ok(true) }; 10],
        families: [const { Ok(true) }; 7],
        write_rounding_modes: [const { Ok(true) }; 2],
        heap_buffer_page_private: Ok(SizeAndAlign {
            size: num::align_up_wrapping(PAGE_SIZE, 0x4000),
            align: 0x4000,
        }),
        min_linear_alignment_bgra8: Ok(16),
    }
}

/// The fake device's `recommendedMaxWorkingSetSize`, which it answers as its own memory's budget
/// too ([`GpuDevice::own_memory_budget`]).
pub const FAKE_OWN_MEMORY: u64 = 0xcf5c_2c000;

/// The fake device's memory heaps, summed ([`GpuDevice::memory_heaps_total`]): 24 GiB less
/// 1 MiB, off the pool's 2 MiB granule, so the pool's range is seen rounded up.
pub const FAKE_MEMORY_HEAPS: u64 = (24 << 30) - (1 << 20);

/// The fake device's alignment of a linear BGRA8 image's rows ([`GpuDevice::linear_row_alignment`]):
/// 256 bytes, as the Vulkan platform's runtime answers on Mesa's RADV, apart from Metal's 16.
pub const FAKE_LINEAR_ROW_ALIGNMENT: u64 = 256;

/// The fake device's identity answers: those of an Apple M3 Max under macOS 27 (measured with
/// `guest/bundle/tools/metal_enum.m`), whose families are Apple1 to Apple9, Mac1 and Mac2, the
/// Common families and Metal3 and Metal4, and whose feature sets are every iOS one and the macOS
/// ones but `macOS_ReadWriteTextureTier2` (10002). The architecture's name is the fake's own.
pub fn fake_identity_caps() -> IdentityCaps {
    let families = caps::SDK_FAMILIES.map(|f| {
        Some(matches!(
            f,
            1001..=1009 | 2001 | 2002 | 3001..=3003 | 5001 | 5002
        ))
    });
    let feature_sets =
        caps::FEATURE_SETS.map(|fs| Some(fs <= 16 || matches!(fs, 10000 | 10001 | 10003..=10005)));
    IdentityCaps {
        families,
        feature_sets,
        counter_sampling: caps::COUNTER_SAMPLING_POINTS.map(|p| Some(p == 0)),
        architecture_name: Some(String::from("fake_gpu")),
        // 16 for each texture-buffer format's texture buffers and linear textures, as the
        // fake's device-wide linear alignments; a 4-byte texel's tiles for every format.
        formats: caps::SDK_PIXEL_FORMATS
            .iter()
            .map(|&format| FormatAnswers {
                format,
                texture_buffer_alignment: caps::TEXTURE_BUFFER_FORMATS
                    .contains(&format)
                    .then_some(16),
                linear_alignment: caps::TEXTURE_BUFFER_FORMATS.contains(&format).then_some(16),
                sparse_tiles: Some([
                    [64, 64, 1],
                    [128, 128, 1],
                    [256, 256, 1],
                    [64, 32, 1],
                    [128, 64, 1],
                    [256, 128, 1],
                    [32, 32, 1],
                    [64, 64, 1],
                    [128, 128, 1],
                ]),
            })
            .collect(),
        sizes: IdentitySizes {
            recommended_max_working_set_size: Some(FAKE_OWN_MEMORY),
            location: Some(0),
            location_number: Some(0),
            max_transfer_rate: Some(0),
            peer_group_id: Some(0),
            peer_index: Some(0),
            peer_count: Some(0),
            max_buffer_length: Some(0x9_b852_0000),
            max_argument_buffer_sampler_count: Some(500_000),
            argument_buffers_support: Some(1),
            read_write_texture_support: Some(2),
            sparse_tile_size_in_bytes: Some(0x4000),
            maximum_concurrent_compilation_task_count: Some(16),
            largest_vertex_amplification_count: Some(8),
            largest_rasterization_rate_map_layer_count: Some(2),
            feature_profile: Some(10),
            half_fp_config: Some(0xe),
            single_fp_config: Some(0x54),
            double_fp_config: Some(0),
        },
        flags: IdentityFlags {
            is_low_power: Some(false),
            is_headless: Some(false),
            is_removable: Some(false),
            has_unified_memory: Some(true),
            supports_32bit_float_filtering: Some(true),
            supports_32bit_msaa: Some(true),
            supports_query_texture_lod: Some(true),
            supports_bc_texture_compression: Some(true),
            supports_pull_model_interpolation: Some(true),
            are_barycentric_coords_supported: Some(true),
            supports_shader_barycentric_coordinates: Some(true),
            supports_dynamic_libraries: Some(true),
            supports_render_dynamic_libraries: Some(true),
            supports_raytracing: Some(true),
            supports_function_pointers: Some(true),
            supports_function_pointers_from_render: Some(true),
            supports_raytracing_from_render: Some(true),
            supports_primitive_motion_blur: Some(true),
            supports_placement_sparse: Some(true),
        },
    }
}

/// [`GpuDevice::argument_layout`]'s fake answer (its doc).
fn fake_argument_layout(arguments: &[ArgumentDesc]) -> Option<ArgumentLayout> {
    if arguments.is_empty() {
        return None;
    }
    let mut members = Vec::with_capacity(arguments.len());
    let mut offset = 0u64;
    for a in arguments {
        let count = a.array_length.max(1);
        members.push(ArgumentMember {
            id: u32::try_from(a.index).ok()?,
            count: u32::try_from(count).ok()?,
            ids_per_element: 1,
            offset,
            stride: 8,
            size: 8,
            kind: fake_kind(a.data_type),
            layout: None,
        });
        offset = offset.checked_add(count.checked_mul(8)?)?;
    }
    Some(ArgumentLayout {
        encoded_length: offset,
        alignment: 8,
        members,
    })
}

/// The kind of `MTLDataType` `data_type`: Pointer, Texture, Sampler, the two pipeline states and
/// the indirect command buffer by their raw values, the rest constants.
fn fake_kind(data_type: u64) -> ArgumentKind {
    match data_type {
        60 => ArgumentKind::Buffer,
        58 => ArgumentKind::Texture,
        59 => ArgumentKind::Sampler,
        78 => ArgumentKind::RenderPipelineState,
        79 => ArgumentKind::ComputePipelineState,
        80 => ArgumentKind::IndirectCommandBuffer,
        _ => ArgumentKind::Constant,
    }
}

/// [`GpuDevice::function_argument_layout`]'s fake answer at buffer index 0.
fn fake_function_layout() -> ArgumentLayout {
    let nested = ArgumentLayout {
        encoded_length: 8,
        alignment: 8,
        members: vec![ArgumentMember {
            id: 0,
            count: 1,
            ids_per_element: 1,
            offset: 0,
            stride: 8,
            size: 8,
            kind: ArgumentKind::Sampler,
            layout: None,
        }],
    };
    ArgumentLayout {
        encoded_length: 16,
        alignment: 8,
        members: vec![
            ArgumentMember {
                id: 0,
                count: 1,
                ids_per_element: 1,
                offset: 0,
                stride: 8,
                size: 8,
                kind: ArgumentKind::Texture,
                layout: None,
            },
            ArgumentMember {
                id: 1,
                count: 1,
                ids_per_element: 1,
                offset: 8,
                stride: 8,
                size: 8,
                kind: ArgumentKind::Buffer,
                layout: Some(Box::new(nested)),
            },
        ],
    }
}
