//! The rt platform's device ([`RtDevice`]): one metal2vulkan-rt device for the process, made of
//! the physical device the library's choice of device picks ([`open`]), with its answers
//! (`super::answers`), its command queues (`super::queue`), events and fences, and the
//! reservation of resource indices as rt's index grants. Every other creator is refused while
//! its piece is unbuilt (`super::unbuilt`), as Metal's creator answers where it fails.
//!
//! **One device.** The process makes one rt device, the first time a device is asked for, and
//! every ask answers it (one `VkDevice` per VM, the host process). A failure to make it is
//! counted once as rt's refusal and again at each later ask, which answers nil.
//!
//! **The choice.** The candidates are the physical devices that meet rt's floor; the pick among
//! them is the library's (`crate::device_choice::chosen_untimed`): the device the host's user's
//! selection variables name, else the one GPU, else the first of the runtime's order. Unlike the
//! Vulkan platform's, it times no GPU (the module's docs there).
//!
//! **Its registry ID** folds its `deviceUUID` (`super::answers::registry_id`), so it differs from
//! the Vulkan platform's for the same GPU, which metal2vulkan-runtime gives.

use std::sync::Arc;

use ash::vk;
use metal2vulkan_rt as rt;

use crate::core::platform::Platform;
use crate::core::platform::caps::{DeviceCaps, IdentityCaps, SAMPLE_COUNTS};
use crate::core::platform::device::{GpuDevice, NoCopyDeallocator};
use crate::core::platform::host_range::HostRange;
use crate::core::platform::types::{
    ArgumentLayout, NsErrorText, PixelFormatInfo, SizeAndAlign, TextureLevelInfo, TextureLevelQuery,
};
use crate::core::platform::values::{
    ArgumentDesc, BufferDesc, DepthStencilDesc, FullComputePipelineDesc,
    FullMeshRenderPipelineDesc, FullRenderPipelineDesc, HeapDesc, HeapMade, IcbDesc,
    RenderPipelineDesc, SamplerDesc,
};
use crate::core::raise::{OnFault, Raise, RaiseResult};
use crate::gpu::device::{HEAP_DESCRIPTOR_OPTIONS, descriptor_options};
use crate::gpu::identity;

use super::answers::{self, Facts};
use super::queue::RtQueue;
use super::unbuilt::{
    RtRefusal, Unbuilt, refuse, refuse_create, refused_as, refused_error, rt_refused,
    rt_refused_create,
};
use super::{RtObj, RtPlatform};

/// The log target of the device's lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// Metal's `maxCommandBufferCount` of a queue made by `newCommandQueue`.
pub const METAL_MAX_COMMAND_BUFFER_COUNT: u32 = 64;

/// The rt device's cache directory in the user's cache directory, beside the Vulkan platform's
/// (`crate::platform::cache_directory`), which it shares nothing with: rt keeps its lowered
/// functions and its pipeline caches there.
#[cfg(not(any(test, feature = "testkit")))]
const CACHE: &str = "reims-vgpu-rt";

/// A test build's rt cache directory, the tests' own, as the Vulkan platform's tests keep theirs.
#[cfg(any(test, feature = "testkit"))]
const CACHE: &str = "reims-vgpu-rt-tests";

/// `id<MTLDevice>` on metal2vulkan-rt: rt's device and what the platform read of it.
pub struct RtDevice {
    /// rt's device.
    rt: rt::Device,
    /// What its answers depend on.
    facts: Facts,
    /// Its capabilities, read as it was made.
    caps: Arc<DeviceCaps>,
}

impl RtDevice {
    /// rt's device.
    pub fn rt(&self) -> &rt::Device {
        &self.rt
    }

    /// What the platform read of it.
    pub fn facts(&self) -> &Facts {
        &self.facts
    }
}

/// Why the platform has no device.
#[derive(Debug)]
pub enum Unopened {
    /// rt refused a call making it.
    Rt(rt::Error),
    /// No physical device meets rt's floor: why each listed one does not, or that none is listed.
    NoDevice(String),
}

impl Unopened {
    /// Counts the refusal at `sel` ([`super::unbuilt::rt_refused_create`]); its kind.
    pub fn refuse(&self, sel: &'static str) -> RtRefusal {
        match self {
            Unopened::Rt(error) => rt_refused_create(sel, error),
            Unopened::NoDevice(why) => {
                crate::core::platform::device::note_relayed_gap();
                refused_as(sel, RtRefusal::NoDevice, why);
                RtRefusal::NoDevice
            }
        }
    }
}

impl From<rt::Error> for Unopened {
    fn from(error: rt::Error) -> Unopened {
        Unopened::Rt(error)
    }
}

/// Makes the process's rt device (the module's docs), on a host of pages of `page_size` bytes,
/// and logs it.
pub fn open(page_size: u64) -> Result<RtObj<RtDevice>, Unopened> {
    let instance = rt::Instance::new(&[])?;
    let listed = instance.physical_devices()?;
    let (_, raw) = instance.raw();
    let mut below = Vec::new();
    let mut candidates = Vec::new();
    for physical in listed {
        // SAFETY: the instance listed the device and is alive.
        let properties = unsafe { raw.get_physical_device_properties(physical.raw()) };
        let name = (properties.device_name_as_c_str())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        match physical.check() {
            Ok(floor) => candidates.push((physical, floor)),
            Err(why) => below.push(format!("{name}: {why}")),
        }
    }
    for line in &below {
        log::info!(target: TARGET, "not a candidate for the rt platform's device: {line}");
    }
    let seen: Vec<crate::device_choice::Seen> = (candidates.iter())
        .map(|(physical, _)| {
            // SAFETY: the instance listed the device and is alive.
            let names = unsafe { raw.enumerate_device_extension_properties(physical.raw()) }
                .unwrap_or_default();
            let lists = |name: &std::ffi::CStr| {
                (names.iter()).any(|p| p.extension_name_as_c_str() == Ok(name))
            };
            let drm = lists(ash::ext::physical_device_drm::NAME);
            let pci = lists(ash::ext::pci_bus_info::NAME);
            // SAFETY: as above; rt's instance is at Vulkan 1.3, where `properties2` is core.
            unsafe { crate::os::seen(raw, physical.raw(), true, drm, pci) }
        })
        .collect();
    let Some(at) = crate::device_choice::chosen_untimed(&seen) else {
        return Err(Unopened::NoDevice(if below.is_empty() {
            "the loader lists no physical device".to_owned()
        } else {
            below.join("; ")
        }));
    };
    let (Some((physical, floor)), Some(uuid)) =
        (candidates.get(at), seen.get(at).and_then(|s| s.uuid))
    else {
        return Err(Unopened::NoDevice(
            "the chosen device has no device UUID".to_owned(),
        ));
    };
    let mut options = rt::DeviceOptions::default();
    options.pipeline_cache_dir = crate::os::user_cache_directory().map(|dir| dir.join(CACHE));
    options.pipeline_cache_cap = Some(crate::platform::SHADER_CACHE_CAPACITY);
    let device = rt::Device::new(physical, options)?;
    let facts = facts(&device, floor, &uuid, page_size);
    log::info!(
        target: TARGET,
        "the rt platform's device: {} ({:04x}:{:04x}), registry ID {:#x}",
        facts.name,
        facts.vendor_id,
        facts.device_id,
        facts.registry_id
    );
    let caps = answers::caps(&facts);
    Ok(RtObj::new(RtDevice {
        rt: device,
        facts,
        caps,
    }))
}

/// What the answers of `device`, made with rt's floor answers `floor`, of `deviceUUID` `uuid`, on
/// a host of `page_size`-byte pages, depend on ([`Facts`]).
fn facts(
    device: &rt::Device,
    floor: &rt::floor::Capabilities,
    uuid: &[u8; vk::UUID_SIZE],
    page_size: u64,
) -> Facts {
    let raw = device.raw();
    let (instance, physical) = (raw.instance, raw.physical_device);
    // SAFETY: the instance and physical device are rt's, read only, while `device` lives.
    let (properties, features) = unsafe {
        (
            instance.get_physical_device_properties(physical),
            instance.get_physical_device_features(physical),
        )
    };
    let float = vk::Format::R32G32B32A32_SFLOAT;
    // SAFETY: as above.
    let filtering = unsafe { instance.get_physical_device_format_properties(physical, float) }
        .optimal_tiling_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
    // SAFETY: as above.
    let msaa = unsafe {
        instance.get_physical_device_image_format_properties(
            physical,
            float,
            vk::ImageType::TYPE_2D,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::ImageCreateFlags::empty(),
        )
    }
    .is_ok_and(|p| p.sample_counts.contains(vk::SampleCountFlags::TYPE_4));
    let sampled = |count: u64| {
        u32::try_from(count).is_ok_and(|count| device.supports_texture_sample_count(count))
    };
    let barycentrics = (device.extensions().iter())
        .any(|name| name.as_c_str() == ash::khr::fragment_shader_barycentric::NAME);
    Facts {
        name: (properties.device_name_as_c_str())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        registry_id: answers::registry_id(uuid),
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        pipeline_cache_uuid: properties.pipeline_cache_uuid,
        limits: properties.limits,
        float16: floor.float16,
        unified_memory: device.has_unified_memory(),
        sample_counts: SAMPLE_COUNTS.map(sampled),
        bc: features.texture_compression_bc == vk::TRUE,
        barycentrics,
        float32_filtering: filtering,
        float32_msaa: msaa,
        // SAFETY: as above.
        device_local: unsafe { identity::device_local_bytes_of(instance, physical) },
        page_size,
    }
}

/// The raise of a query rt refused at `sel`, counted already under `kind`: Metal's exception at
/// the send, its detail the kind's code and why.
fn refused_raise(sel: &'static str, kind: RtRefusal, why: &dyn std::fmt::Display) -> Raise {
    Raise::platform(sel).with_detail(format!("{}: {why}", kind.code()))
}

/// Grants `indices` of `class` as runs of consecutive indices, each one grant; an index granted
/// already stays granted, as a second reservation of Metal's of an ID reserved changes nothing.
/// One rt refuses otherwise (past the class's range) is counted at `sel`.
fn grant(device: &rt::Device, sel: &'static str, class: rt::IndexClass, indices: &[u64]) {
    let mut sorted: Vec<u64> = indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for index in sorted {
        match runs.last_mut() {
            Some((first, count)) if *first + *count == index => *count += 1,
            _ => runs.push((index, 1)),
        }
    }
    for (first, count) in runs {
        let (Ok(first), Ok(count)) = (u32::try_from(first), u32::try_from(count)) else {
            let why = format!("index {first} is past 32 bits");
            refused_as(sel, RtRefusal::Limit, &why);
            continue;
        };
        let Err(error) = device.grant_indices(class, first, count) else {
            continue;
        };
        if !granted_already(&error) {
            rt_refused(sel, &error);
            continue;
        }
        for index in first..first.saturating_add(count) {
            match device.grant_indices(class, index, 1) {
                Err(error) if !granted_already(&error) => {
                    rt_refused(sel, &error);
                }
                _ => {}
            }
        }
    }
}

/// Whether rt refused a grant because an index of it is granted already.
fn granted_already(error: &rt::Error) -> bool {
    matches!(
        error,
        rt::Error::Refused(rt::Refusal::IndicesGranted { .. })
    )
}

/// The selector of the resource indices' reservation.
const RESERVE: &str = "reserveResourceIndicesForResourceType:indices:indexCount:";

impl GpuDevice<RtPlatform> for RtObj<RtDevice> {
    /// Nothing: rt takes a library of any platform's.
    fn allow_foreign_libraries(&self) {}

    /// Unbuilt: no encoder ([`Unbuilt::ArgumentLayout`]).
    fn argument_layout(&self, arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>> {
        let _ = arguments;
        refuse_create(Unbuilt::ArgumentLayout, "newArgumentEncoderWithArguments:");
        Ok(None)
    }

    fn function_argument_layout(
        &self,
        function: &<RtPlatform as Platform>::Function,
        buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>> {
        let _ = buffer_index;
        function.unreachable()
    }

    /// rt's answer: the device's image's own memory requirements, rounded to its buffer-image
    /// granularity, which the guest lays its heaps out by, not Metal's. Where rt refuses (a
    /// texture it makes none of, or `Gap::HeapTexture`, an image that takes an allocation of its
    /// own), the refusal is counted under its kind and raised ([`refused_raise`]), which the
    /// decoder answers the guest as refused, and the guest then lays the texture out by a bound
    /// of its own: never 0 bytes at 0, which places the texture at its neighbours' offset. A
    /// value rt's descriptor has no value of ([`super::descriptor::rt_texture_descriptor`]) is
    /// counted as Metal's refusal (`rt-refused-metal`) and raised alike.
    fn heap_texture_size_and_align(
        &self,
        desc: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        const SEL: &str = "heapTextureSizeAndAlignWithDescriptor:";
        let d = match super::descriptor::rt_texture_descriptor(&desc.desc()) {
            Ok(d) => d,
            Err(why) => {
                refused_as(SEL, RtRefusal::Refused, &why);
                return Err(refused_raise(SEL, RtRefusal::Refused, &why));
            }
        };
        match self.rt.heap_texture_size_and_align(&d) {
            Ok(rt::SizeAndAlign { size, align }) => Ok(SizeAndAlign { size, align }),
            Err(error) => {
                let kind = rt_refused(SEL, &error);
                Err(refused_raise(SEL, kind, &error))
            }
        }
    }

    /// Unbuilt: nil ([`Unbuilt::HostImport`]); the deallocator is not run, as Metal runs none for
    /// a buffer it does not make.
    fn new_buffer_over_host_memory(
        &self,
        on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let _ = (on, bytes, options);
        std::mem::forget(deallocator);
        refuse_create(
            Unbuilt::HostImport,
            "newBufferWithBytesNoCopy:length:options:deallocator:",
        );
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::Buffer`]).
    fn new_buffer_of_length(
        &self,
        on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        let _ = (on, length, options);
        refuse_create(Unbuilt::Buffer, "newBufferWithLength:options:");
        Ok(None)
    }

    /// rt's queue, of Metal's default `maxCommandBufferCount`; nil where rt refuses it.
    fn new_command_queue(&self) -> Option<RtObj<RtQueue>> {
        match self.rt.new_command_queue(METAL_MAX_COMMAND_BUFFER_COUNT) {
            Ok(queue) => Some(RtObj::new(RtQueue::new(queue))),
            Err(error) => {
                rt_refused_create("newCommandQueue", &error);
                None
            }
        }
    }

    /// [`new_command_queue`](GpuDevice::new_command_queue)'s: rt submits every command queue's
    /// work to its one GPU queue.
    fn new_command_queue_unshared(&self) -> Option<RtObj<RtQueue>> {
        self.new_command_queue()
    }

    /// No: rt copies no guest RAM by its host addresses.
    fn copies_guest_ram(&self) -> bool {
        false
    }

    /// rt's `minimumLinearTextureAlignmentForPixelFormat:` of BGRA8, the device's linear row
    /// rounding; 0 where rt refuses it.
    fn linear_row_alignment(&self) -> u64 {
        (self.rt)
            .minimum_linear_texture_alignment(rt::PixelFormat::BGRA8Unorm)
            .unwrap_or(0)
    }

    /// Unbuilt ([`Unbuilt::Pipeline`]): the `NSError`.
    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&<RtPlatform as Platform>::Function>,
    ) -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
        if let Some(function) = compute_function {
            function.unreachable()
        }
        Err(refused_error(
            Unbuilt::Pipeline,
            "newComputePipelineStateWithFunction:error:",
        ))
    }

    /// Unbuilt ([`Unbuilt::Library`]): the `NSError`.
    fn new_shader_library(&self) -> Result<Option<<RtPlatform as Platform>::Library>, NsErrorText> {
        Err(refused_error(
            Unbuilt::Library,
            "newLibraryWithSource:options:error:",
        ))
    }

    /// rt's event; nil where rt refuses it.
    fn new_event(&self, on: OnFault) -> RaiseResult<Option<RtObj<rt::Event>>> {
        let _ = on;
        Ok(match self.rt.new_event() {
            Ok(event) => Some(RtObj::new(event)),
            Err(error) => {
                rt_refused_create("newEvent", &error);
                None
            }
        })
    }

    /// rt's fence.
    fn new_fence(&self) -> RaiseResult<Option<RtObj<rt::Fence>>> {
        Ok(Some(RtObj::new(self.rt.new_fence())))
    }

    /// Unbuilt ([`Unbuilt::Library`]): the `NSError`.
    fn new_library(
        &self,
        data: &[u8],
    ) -> RaiseResult<Result<Option<<RtPlatform as Platform>::Library>, NsErrorText>> {
        let _ = data;
        Ok(Err(refused_error(
            Unbuilt::Library,
            "newLibraryWithData:error:",
        )))
    }

    fn new_dynamic_library(
        &self,
        library: &<RtPlatform as Platform>::Library,
    ) -> RaiseResult<Result<Option<<RtPlatform as Platform>::DynamicLibrary>, NsErrorText>> {
        library.unreachable()
    }

    /// Unbuilt: nil ([`Unbuilt::RateMap`]).
    fn new_rate_map(
        &self,
        descriptor: &<RtPlatform as Platform>::RateMapDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::RateMap>> {
        let _ = descriptor;
        refuse_create(Unbuilt::RateMap, "newRasterizationRateMapWithDescriptor:");
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::SharedEvent`]).
    fn new_shared_event(&self) -> Option<<RtPlatform as Platform>::SharedEvent> {
        refuse_create(Unbuilt::SharedEvent, "newSharedEvent");
        None
    }

    /// Unbuilt: nil ([`Unbuilt::SharedTexture`]).
    fn new_shared_texture(
        &self,
        descriptor: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let _ = descriptor;
        refuse_create(Unbuilt::SharedTexture, "newSharedTextureWithDescriptor:");
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::SharedTexture`]).
    fn open_shared_texture(
        &self,
        handle: Option<&<RtPlatform as Platform>::SharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        if let Some(handle) = handle {
            handle.unreachable()
        }
        let _ = resource_index;
        refuse_create(
            Unbuilt::SharedTexture,
            "newSharedTextureWithHandle:withResourceIndex:",
        );
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::Texture`]).
    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &<RtPlatform as Platform>::TextureDescriptor,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Texture>> {
        let _ = (on, descriptor);
        refuse_create(Unbuilt::Texture, "newTextureWithDescriptor:");
        Ok(None)
    }

    /// Textures' (type 0) and sampler states' (type 1) indices as rt's grants of their class
    /// ([`grant`]); indirect command buffers' (type 5) are unbuilt ([`Unbuilt::Icb`]), and those of
    /// another type ([`Unbuilt::ResourceType`]): each reserves nothing. Never a raise.
    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()> {
        let class = match resource_type {
            0 => rt::IndexClass::Texture,
            1 => rt::IndexClass::Sampler,
            5 => {
                refuse(Unbuilt::Icb, RESERVE);
                return Ok(());
            }
            _ => {
                refuse(Unbuilt::ResourceType, RESERVE);
                return Ok(());
            }
        };
        grant(&self.rt, RESERVE, class, indices);
        Ok(())
    }

    /// Unbuilt ([`Unbuilt::FormatInfo`]): Metal's answer for a value it describes no format for,
    /// `MTLPixelFormatInvalid` with every word 0.
    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo {
        let _ = format;
        refuse(Unbuilt::FormatInfo, "MTLPixelFormatGetInfoForDevice");
        PixelFormatInfo {
            name: Some("MTLPixelFormatInvalid".to_owned()),
            ..PixelFormatInfo::default()
        }
    }

    /// Unbuilt ([`Unbuilt::FormatInfo`]): every word 0.
    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo {
        let _ = q;
        refuse(
            Unbuilt::FormatInfo,
            "MTLGetTextureLevelInfoForDeviceWithOptions",
        );
        TextureLevelInfo::default()
    }

    /// Read as the device was made ([`answers::caps`]).
    fn caps(&self) -> Arc<DeviceCaps> {
        Arc::clone(&self.caps)
    }

    /// [`answers::identity`], with rt's linear alignment of each format.
    fn identity_caps(&self) -> IdentityCaps {
        identity_of(self)
    }

    /// Yes, as on the Vulkan platform: the graphics lane offers its pipelines' private state, and
    /// rt's pipelines, once built, take it.
    fn takes_private_render_state(&self) -> bool {
        true
    }

    /// The device-local heaps' budgets, as the Vulkan platform answers
    /// (`crate::gpu::identity::device_local_budget`).
    fn own_memory_budget(&self) -> u64 {
        let raw = self.rt.raw();
        // SAFETY: the instance and physical device are rt's, read only, while the device lives.
        unsafe { identity::device_local_budget_of(raw.instance, raw.physical_device) }
    }

    /// Every heap's size, as the Vulkan platform answers.
    fn memory_heaps_total(&self) -> u64 {
        let raw = self.rt.raw();
        // SAFETY: as above.
        let memory =
            unsafe { (raw.instance).get_physical_device_memory_properties(raw.physical_device) };
        identity::memory_heaps_total_of(&memory)
    }

    /// Yes, as on the Vulkan platform: rt's translator links an OpenCL-mode kernel as Metal does.
    fn makes_opencl_mode_pipelines(&self) -> bool {
        true
    }

    /// Unbuilt: nil ([`Unbuilt::Buffer`]).
    fn new_buffer(
        &self,
        desc: BufferDesc<'_>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Buffer>> {
        std::mem::forget(desc.deallocator);
        refuse_create(Unbuilt::Buffer, "newBufferWithDescriptor:");
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::Heap`]), with the descriptor's options as Metal's descriptor
    /// holds them.
    fn new_heap(&self, desc: &HeapDesc) -> RaiseResult<HeapMade<RtPlatform>> {
        refuse_create(Unbuilt::Heap, "newHeapWithDescriptor:");
        Ok(HeapMade {
            heap: None,
            descriptor_resource_options: descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options),
        })
    }

    /// Unbuilt: nil ([`Unbuilt::HostImport`]), with the descriptor's options.
    fn new_heap_over_host_memory(
        &self,
        bytes: HostRange<'_>,
        desc: &HeapDesc,
    ) -> RaiseResult<HeapMade<RtPlatform>> {
        let _ = bytes;
        refuse_create(Unbuilt::HostImport, "newHeapWithDescriptor:");
        Ok(HeapMade {
            heap: None,
            descriptor_resource_options: descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options),
        })
    }

    /// Unbuilt: nil ([`Unbuilt::Sampler`]).
    fn new_sampler_state(
        &self,
        desc: &SamplerDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::SamplerState>> {
        let _ = desc;
        refuse_create(Unbuilt::Sampler, "newSamplerStateWithDescriptor:");
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::DepthStencil`]).
    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::DepthStencilState>> {
        let _ = desc;
        refuse_create(Unbuilt::DepthStencil, "newDepthStencilStateWithDescriptor:");
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::Icb`]).
    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::Icb>> {
        let _ = (desc, max_count, options);
        refuse_create(
            Unbuilt::Icb,
            "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:",
        );
        Ok(None)
    }

    /// Unbuilt ([`Unbuilt::Pipeline`]): the `NSError`.
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let _ = desc;
        Err(refused_error(
            Unbuilt::Pipeline,
            "newRenderPipelineStateWithDescriptor:error:",
        ))
    }

    /// Unbuilt ([`Unbuilt::Pipeline`]): the `NSError`.
    fn new_render_pipeline_state_full(
        &self,
        desc: &FullRenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let _ = desc;
        Err(refused_error(
            Unbuilt::Pipeline,
            "newRenderPipelineStateWithDescriptor:error:",
        ))
    }

    /// Unbuilt ([`Unbuilt::Pipeline`]): the `NSError`.
    fn new_mesh_render_pipeline_state_full(
        &self,
        desc: &FullMeshRenderPipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::RenderPipeline>, NsErrorText> {
        let _ = desc;
        Err(refused_error(
            Unbuilt::Pipeline,
            "newRenderPipelineStateWithMeshDescriptor:options:reflection:error:",
        ))
    }

    /// Unbuilt ([`Unbuilt::Pipeline`]): the `NSError`.
    fn new_compute_pipeline_state_full(
        &self,
        desc: &FullComputePipelineDesc<'_, RtPlatform>,
    ) -> Result<Option<<RtPlatform as Platform>::ComputePipeline>, NsErrorText> {
        let _ = desc;
        Err(refused_error(
            Unbuilt::Pipeline,
            "newComputePipelineStateWithDescriptor:options:reflection:error:",
        ))
    }
}

/// The identity of `device` ([`answers::identity`]): rt's linear alignment of each SDK format rt
/// has a format of, none where it refuses one.
fn identity_of(device: &RtObj<RtDevice>) -> IdentityCaps {
    answers::identity(&device.facts, &device.caps, |format| {
        let pf = crate::gpu::runtime_pixel_format(format)?;
        device.rt.minimum_linear_texture_alignment(pf).ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::Handle as _;
    use crate::core::platform::device::relaying_gaps;
    use crate::core::platform::values::{OptionSet, TextureDesc};

    /// The flat `path = value` pairs of a value's `Debug` text, one per field of each struct in
    /// it, for the list of the rt device's answers that differ from the Vulkan platform's.
    fn fields(path: &str, debug: &str, out: &mut Vec<(String, String)>) {
        let debug = debug.trim();
        let open = debug.find(" { ").filter(|&at| {
            debug.ends_with(" }") && debug[..at].chars().all(|c| c.is_alphanumeric() || c == '_')
        });
        let Some(at) = open else {
            out.push((path.to_owned(), debug.to_owned()));
            return;
        };
        let inner = &debug[at + 3..debug.len() - 2];
        let (mut depth, mut start, mut quoted, mut escaped) = (0i32, 0usize, false, false);
        let mut items = Vec::new();
        for (i, c) in inner.char_indices() {
            match c {
                _ if escaped => escaped = false,
                '\\' if quoted => escaped = true,
                '"' => quoted = !quoted,
                _ if quoted => {}
                '{' | '[' | '(' => depth += 1,
                '}' | ']' | ')' => depth -= 1,
                ',' if depth == 0 => {
                    items.push(&inner[start..i]);
                    start = i + 1;
                }
                _ => {}
            }
        }
        items.push(&inner[start..]);
        for item in items {
            let item = item.trim();
            match item.split_once(": ") {
                Some((key, value)) => fields(&format!("{path}.{key}"), value, out),
                None if !item.is_empty() => out.push((path.to_owned(), item.to_owned())),
                None => {}
            }
        }
    }

    /// Each field whose value differs between `rt` and `old`, as a line.
    fn differences(name: &str, rt: &str, old: &str) -> Vec<String> {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        fields(name, rt, &mut a);
        fields(name, old, &mut b);
        let mut lines = Vec::new();
        for (path, value) in &a {
            match b.iter().find(|(p, _)| p == path) {
                Some((_, other)) if other == value => {}
                Some((_, other)) => lines.push(format!("{path}: rt {value}, vulkan {other}")),
                None => lines.push(format!("{path}: rt {value}, vulkan has none")),
            }
        }
        lines
    }

    /// The flat fields of a struct's `Debug` text name each field once.
    #[test]
    fn a_debug_text_flattens_to_its_fields() {
        let mut out = Vec::new();
        fields(
            "x",
            "A { b: 1, c: C { d: [1, 2], e: Some(3) }, f: \"g, h\" }",
            &mut out,
        );
        let paths: Vec<&str> = out.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, ["x.b", "x.c.d", "x.c.e", "x.f"]);
        assert_eq!(
            differences("x", "A { b: 1, c: 2 }", "A { b: 1, c: 3 }"),
            ["x.c: rt 2, vulkan 3"]
        );
    }

    /// The rt platform's device on the GPU the test holds: made once and answered at every ask,
    /// its registry ID opening it again; its answers bounded by its GPU; its queues, events and
    /// fences made; the guest's texture indices granted, twice without a refusal; a heap texture's
    /// size rt's, and a shape rt or Metal refuses a counted raise, never 0 bytes at 0; every
    /// creator of an unbuilt piece nil and counted. The answers that differ from the Vulkan
    /// platform's on the same GPU are printed, for RESULT.md: run it with `--no-capture`.
    #[test]
    fn the_device_and_its_answers_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        let platform = RtPlatform::new();
        let device = platform
            .system_default_device()
            .expect("an rt device on the GPU the test holds");
        let again = platform.system_default_device().expect("the same device");
        assert_eq!(again.id(), device.id());
        let caps = device.caps();
        assert_eq!(
            platform.open_device(caps.registry_id).map(|d| d.id()),
            Some(device.id())
        );
        assert!(platform.open_device(caps.registry_id ^ 1).is_none());
        let limits = device.facts().limits;
        assert!(caps.limits.max_texture_width_2d <= u64::from(limits.max_image_dimension2_d));
        assert!(caps.sample_counts[0] == Ok(true));
        let identity = device.identity_caps();
        assert_eq!(identity.formats.len(), crate::core::platform::caps::SDK_PIXEL_FORMATS.len());
        assert!(device.own_memory_budget() > 0);
        assert!(device.memory_heaps_total() >= device.own_memory_budget());
        assert!(device.linear_row_alignment() > 0);
        assert!(device.new_command_queue().is_some());
        assert!(device.new_command_queue_unshared().is_some());
        assert!(matches!(device.new_event(OnFault::Return), Ok(Some(_))));
        assert!(matches!(device.new_fence(), Ok(Some(_))));
        let guests: Vec<u64> = (1..=16320).collect();
        let before = crate::answer::degradation_counts();
        assert_eq!(device.reserve_resource_indices(0, &guests), Ok(()));
        assert_eq!(device.reserve_resource_indices(0, &guests[..64]), Ok(()));
        assert_eq!(crate::answer::degradation_counts(), before);
        let (made, relayed) = relaying_gaps(|| device.new_buffer_of_length(OnFault::Return, 64, 0));
        assert!(matches!(made, Ok(None)) && relayed);
        let heap = device.new_heap(&HeapDesc {
            size: 0x4000,
            heap_type: 0,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
        });
        assert!(heap.as_ref().is_ok_and(|h| h.heap.is_none()));
        assert!(device.new_library(&[]).is_ok_and(|made| made.is_err()));
        let texture = platform.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 64,
            height: 64,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        let sized = device.heap_texture_size_and_align(&texture);
        let Ok(SizeAndAlign { size, align }) = sized else {
            panic!("no size: {sized:?}");
        };
        assert!(size >= 64 * 64 * 4 && align.is_power_of_two(), "{size} at {align}");
        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let metal = RtRefusal::Refused.code();
        let before = counted(metal);
        let refused = [
            TextureDesc {
                texture_type: 99,
                ..TextureDesc::default()
            },
            TextureDesc {
                width: 0,
                ..TextureDesc::default()
            },
        ];
        for shape in refused {
            let descriptor = platform.texture_descriptor(&shape);
            let asked = device.heap_texture_size_and_align(&descriptor);
            assert!(asked.is_err(), "{shape:?}: {asked:?}");
        }
        assert_eq!(counted(metal), before + 2);

        let old = crate::gpu::device::tests::dev();
        let mut lines = differences(
            "caps",
            &format!("{:?}", *caps),
            &format!("{:?}", *old.caps()),
        );
        let old_identity = old.identity_caps();
        let (formats, old_formats) = (identity.formats.clone(), old_identity.formats.clone());
        let strip = |mut id: IdentityCaps| {
            id.formats.clear();
            format!("{id:?}")
        };
        lines.extend(differences(
            "identity",
            &strip(identity),
            &strip(old_identity),
        ));
        let differing: Vec<String> = (formats.iter().zip(&old_formats))
            .filter(|(a, b)| a != b)
            .map(|(a, b)| format!("format {}: rt {a:?}, vulkan {b:?}", a.format))
            .collect();
        lines.push(format!(
            "formats: {} of {} differ",
            differing.len(),
            formats.len()
        ));
        lines.extend(differing.into_iter().take(6));
        lines.push(format!(
            "own_memory_budget: rt {}, vulkan {}",
            device.own_memory_budget(),
            old.own_memory_budget()
        ));
        lines.push(format!(
            "memory_heaps_total: rt {}, vulkan {}",
            device.memory_heaps_total(),
            old.memory_heaps_total()
        ));
        lines.push(format!(
            "heapTextureSizeAndAlign of a 64x64 BGRA8Unorm Private texture: rt {size} at {align}"
        ));
        lines.push(format!(
            "linear_row_alignment: rt {}, vulkan {}",
            device.linear_row_alignment(),
            old.linear_row_alignment()
        ));
        println!(
            "rt platform's answers on {} that differ from the Vulkan platform's:",
            device.facts().name
        );
        for line in lines {
            println!("  {line}");
        }
    }
}
