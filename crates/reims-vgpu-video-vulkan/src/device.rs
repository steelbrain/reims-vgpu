//! The Vulkan device a decoder runs on, which it need not own: the
//! instance, the device, a queue of a video-decode family, and the
//! graphics family whose work samples the decoded images.
//!
//! [`VideoDevice::from_raw`] takes another's device as it is, as the
//! product's will be metal2vulkan's, and
//! [`VideoDevice::from_raw_copying_on`] copies pictures out on a queue of
//! another submitter's that it lends ([`LentQueue`]).
//! [`VideoDevice::create`] makes one of its own, with a graphics queue and a
//! video-decode queue, for the tests and the tools.

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, OnceLock};

use ash::vk;
use reims_vgpu_video_core::sync::Mutex;

use crate::encode::device::{EncodeChoice, EncodeQueue, RawEncodeQueue};
use crate::error::{Error, Named};
use crate::lent::{LentCopies, LentQueue, Submitter};

/// `VK_KHR_unified_image_layouts`, which ash 0.38's headers predate.
const UNIFIED_IMAGE_LAYOUTS: &CStr = c"VK_KHR_unified_image_layouts";

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR`
/// (vulkan_core.h).
const UNIFIED_IMAGE_LAYOUTS_FEATURES: vk::StructureType =
    vk::StructureType::from_raw(1_000_527_000);

/// `VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR`, as vulkan_core.h
/// declares it.
#[repr(C)]
struct UnifiedImageLayoutsFeatures {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    unified_image_layouts: vk::Bool32,
    unified_image_layouts_video: vk::Bool32,
}

impl UnifiedImageLayoutsFeatures {
    fn new(enabled: bool) -> Self {
        Self {
            s_type: UNIFIED_IMAGE_LAYOUTS_FEATURES,
            p_next: std::ptr::null_mut(),
            unified_image_layouts: vk::Bool32::from(enabled),
            unified_image_layouts_video: vk::Bool32::from(enabled),
        }
    }
}

/// The device extensions an H.264 decode needs, which a device
/// [`VideoDevice::create`] makes has.
const REQUIRED_EXTENSIONS: [&CStr; 3] = [
    ash::khr::video_queue::NAME,
    ash::khr::video_decode_queue::NAME,
    ash::khr::video_decode_h264::NAME,
];

/// The device extension an H.265 decode needs besides the first two of
/// [`REQUIRED_EXTENSIONS`], which [`VideoDevice::create`] enables where the
/// decode family decodes H.265.
const H265_EXTENSION: &CStr = ash::khr::video_decode_h265::NAME;

/// Another's device, as [`VideoDevice::from_raw`] takes it.
pub struct RawDevice {
    /// The loader the instance was made through.
    pub entry: ash::Entry,
    /// The instance.
    pub instance: ash::Instance,
    /// The `VkApplicationInfo::apiVersion` the instance was made with, at
    /// least Vulkan 1.2: below 1.3, the crate records and submits through
    /// `VK_KHR_synchronization2`'s commands, which the device then enables.
    pub instance_api_version: u32,
    /// The device's physical device.
    pub physical_device: vk::PhysicalDevice,
    /// The device, made with `VK_KHR_video_queue`,
    /// `VK_KHR_video_decode_queue` and the decode extension of each codec
    /// the decoders on it decode (`VK_KHR_video_decode_h264` for
    /// [`VideoDevice::from_raw`]; [`VideoDevice::from_raw_with_codecs`]
    /// names them), and `VK_KHR_synchronization2` on an instance below
    /// Vulkan 1.3, and the `timelineSemaphore` and `synchronization2`
    /// features.
    pub device: ash::Device,
    /// A queue family with `VK_QUEUE_VIDEO_DECODE_BIT_KHR` whose video
    /// codec operations include the decode of each of those codecs.
    pub decode_family: u32,
    /// A queue of `decode_family`, which only the decoders on this device
    /// submit to.
    pub decode_queue: vk::Queue,
    /// The family of `copy_queue`, which copies (`TRANSFER`, `GRAPHICS` or
    /// `COMPUTE`).
    pub copy_family: u32,
    /// The queue a picture decoded into a DPB picture is copied out to its
    /// output on, which only the decoders submit to: `decode_queue` itself
    /// where the decode family copies.
    pub copy_queue: vk::Queue,
    /// The family of the graphics queue whose work samples the outputs,
    /// which share them with the decode and copy families.
    pub graphics_family: u32,
    /// Whether the device was made with `VK_KHR_unified_image_layouts`'s
    /// `unifiedImageLayoutsVideo` feature enabled.
    pub unified_image_layouts_video: bool,
}

/// The queue a device's decoders copy pictures out on.
enum CopyQueue {
    /// The decode queue, in the decode's own submission.
    Decode,
    /// A queue of the decoders' own.
    Own(Mutex<vk::Queue>),
    /// A queue another submitter lends, each copy submitted once the
    /// host sees its decode done ([`LentCopies`]).
    Lent(LentCopies),
}

/// The Vulkan device decoders run on.
///
/// Its decode queue and copy queue are the decoders' alone, but for a lent
/// copy queue ([`LentQueue`]): they serialize their submissions to them.
/// Made by [`VideoDevice::create`], it owns its device and instance and
/// destroys them as it goes, after every decoder, image and picture made on
/// it, which each hold it.
pub struct VideoDevice {
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    decode_family: u32,
    decode_queue: Mutex<vk::Queue>,
    copy_family: u32,
    copy_queue: CopyQueue,
    graphics_family: u32,
    /// The graphics queue of a device of our own.
    graphics_queue: Option<vk::Queue>,
    unified_layouts: bool,
    /// The codecs decoders on it decode.
    codecs: vk::VideoCodecOperationFlagsKHR,
    memory: vk::PhysicalDeviceMemoryProperties,
    video_instance: ash::khr::video_queue::Instance,
    video: ash::khr::video_queue::Device,
    decode: ash::khr::video_decode_queue::Device,
    /// The device's submit commands: `VK_KHR_synchronization2`'s, on an
    /// instance below Vulkan 1.3, where the core ones are not there.
    submitter: Submitter,
    /// Whether the device and instance are ours to destroy.
    owned: bool,
    /// The encoders' queue, once attached ([`VideoDevice::attach_encode`]).
    pub(crate) encode: OnceLock<EncodeQueue>,
}

/// What [`VideoDevice::create`] found on one physical device.
struct Choice {
    physical_device: vk::PhysicalDevice,
    decode_family: u32,
    graphics_family: u32,
    /// The queues the graphics family has, for a device whose decode and
    /// graphics family are one.
    family_queues: u32,
    /// Whether the decode family copies.
    decode_family_copies: bool,
    /// Whether the decode family decodes H.265 and the device has its
    /// extension.
    decodes_h265: bool,
    maintenance1: bool,
    unified_layouts: bool,
    /// The kind of device, best first: discrete, integrated, virtual,
    /// then any other.
    rank: u8,
    /// A queue of a video-encode family the device has one free for, and
    /// the codecs it encodes.
    encode: Option<EncodeChoice>,
}

/// The queue flags of a family that copies: every graphics or compute
/// family copies, whether or not it lists `TRANSFER`.
fn copies(flags: vk::QueueFlags) -> bool {
    flags.intersects(vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
}

impl VideoDevice {
    /// H.264 decoders on another's device, which it keeps:
    /// [`Self::from_raw_with_codecs`] of H.264 decode alone.
    ///
    /// # Safety
    ///
    /// Every handle of `raw` is valid and made as its fields say, and
    /// stays so for as long as the returned device and anything made on it
    /// lives; nothing but the decoders made on it submits to
    /// `raw.decode_queue` and `raw.copy_queue` meanwhile.
    pub unsafe fn from_raw(raw: RawDevice) -> Arc<Self> {
        // SAFETY: as the caller promises, of H.264 decode.
        unsafe { Self::from_raw_with_codecs(raw, vk::VideoCodecOperationFlagsKHR::DECODE_H264) }
    }

    /// Decoders of `codecs` (`DECODE_H264`, `DECODE_H265` or both) on
    /// another's device, which it keeps. A codec left out has no
    /// capabilities on it ([`crate::h265_capabilities`] answers none), so
    /// no decoder of it starts.
    ///
    /// # Safety
    ///
    /// As for [`Self::from_raw`], `raw.device` having been made with the
    /// decode extension of each codec of `codecs`
    /// (`VK_KHR_video_decode_h264`, `VK_KHR_video_decode_h265`), and
    /// `raw.decode_family` decoding each.
    pub unsafe fn from_raw_with_codecs(
        raw: RawDevice,
        codecs: vk::VideoCodecOperationFlagsKHR,
    ) -> Arc<Self> {
        let copy_queue = if raw.copy_queue == raw.decode_queue {
            CopyQueue::Decode
        } else {
            CopyQueue::Own(Mutex::new(raw.copy_queue))
        };
        Arc::new(Self::assemble(raw, copy_queue, None, false, codecs))
    }

    /// Decoders of `codecs` on another's device, as
    /// [`Self::from_raw_with_codecs`], that copy each picture out on
    /// `copies`, a queue of the device's that another submitter lends, in
    /// place of `raw.copy_queue` and `raw.copy_family`, which it ignores. A
    /// thread of the device's submits each copy there once the host sees
    /// its decode done, so the copy never waits on the GPU in the lender's
    /// queue. Refused where that thread cannot start.
    ///
    /// # Safety
    ///
    /// As for [`Self::from_raw_with_codecs`], `raw.copy_queue` aside;
    /// `copies` lends a queue of `raw.device`, of the family it names.
    pub unsafe fn from_raw_copying_on(
        mut raw: RawDevice,
        codecs: vk::VideoCodecOperationFlagsKHR,
        copies: Box<dyn LentQueue>,
    ) -> Result<Arc<Self>, Error> {
        raw.copy_family = copies.family();
        let submitter = Self::submitter(&raw);
        let copies = LentCopies::new(copies, submitter)?;
        Ok(Arc::new(Self::assemble(
            raw,
            CopyQueue::Lent(copies),
            None,
            false,
            codecs,
        )))
    }

    /// The submit commands of `raw`'s device.
    fn submitter(raw: &RawDevice) -> Submitter {
        Submitter {
            device: raw.device.clone(),
            synchronization2: (raw.instance_api_version < vk::API_VERSION_1_3)
                .then(|| ash::khr::synchronization2::Device::new(&raw.instance, &raw.device)),
        }
    }

    /// A device of its own, on the first physical device, discrete GPUs
    /// first, that decodes H.264 through Vulkan Video: one graphics queue
    /// and one video-decode queue, with `VK_KHR_video_decode_h265` where
    /// the decode family decodes H.265 too, and
    /// `VK_KHR_video_maintenance1` and `VK_KHR_unified_image_layouts`'s
    /// `unifiedImageLayoutsVideo` where the device has them. Its copies run on the decode queue where the
    /// decode family copies, else on the graphics queue. Where a family
    /// encodes H.264 or H.265 through Vulkan Video, it has one encode queue
    /// too, of a family of its own, else a second queue of the decode
    /// family's, with the encode extensions the family's codecs need
    /// ([`Self::attach_encode`]'s queue).
    /// [`Error::NoVideoDecode`] says why each device was passed over where
    /// none decodes H.264.
    pub fn create() -> Result<Arc<Self>, Error> {
        // SAFETY: opening the loader runs its initialization; the entry
        // keeps it open while it lives.
        let entry = unsafe { ash::Entry::load() }.map_err(|e| Error::Loader(e.to_string()))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"reims-vgpu-video-vulkan")
            .api_version(vk::API_VERSION_1_3);
        let info = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `info` and what it points at live through the call.
        let instance = unsafe { entry.create_instance(&info, None) }.named("vkCreateInstance")?;
        match make_device(&instance) {
            Ok((choice, device)) => {
                let encode = choice.encode;
                let maintenance1 = choice.maintenance1;
                let made = Self::owned(entry, instance, choice, device);
                if let Some(encode) = encode {
                    // SAFETY: the device was made with this queue and the
                    // encode extensions of its codecs; nothing else submits
                    // to it.
                    let queue =
                        unsafe { made.device.get_device_queue(encode.family, encode.index) };
                    let raw = RawEncodeQueue {
                        family: encode.family,
                        queue,
                        codecs: encode.codecs,
                        video_maintenance1: maintenance1,
                    };
                    // SAFETY: as above.
                    if let Err(error) = unsafe { made.attach_encode(raw) } {
                        log::debug!("no encode queue: {error}");
                    }
                }
                Ok(Arc::new(made))
            }
            Err(error) => {
                // SAFETY: nothing made from the instance is left.
                unsafe { instance.destroy_instance(None) };
                Err(error)
            }
        }
    }

    fn owned(
        entry: ash::Entry,
        instance: ash::Instance,
        choice: Choice,
        device: ash::Device,
    ) -> Self {
        // A family of both gives the graphics queue index 0 and the decode
        // queue index 1, or one queue to both where it has one.
        let decode_index =
            u32::from(choice.decode_family == choice.graphics_family && choice.family_queues >= 2);
        // SAFETY: the device was made with these queues.
        let graphics = unsafe { device.get_device_queue(choice.graphics_family, 0) };
        // SAFETY: as above.
        let decode = unsafe { device.get_device_queue(choice.decode_family, decode_index) };
        let (copy_family, copy_queue) = if choice.decode_family_copies {
            (choice.decode_family, decode)
        } else {
            (choice.graphics_family, graphics)
        };
        let mut codecs = vk::VideoCodecOperationFlagsKHR::DECODE_H264;
        if choice.decodes_h265 {
            codecs |= vk::VideoCodecOperationFlagsKHR::DECODE_H265;
        }
        let raw = RawDevice {
            entry,
            instance,
            instance_api_version: vk::API_VERSION_1_3,
            physical_device: choice.physical_device,
            device,
            decode_family: choice.decode_family,
            decode_queue: decode,
            copy_family,
            copy_queue,
            graphics_family: choice.graphics_family,
            unified_image_layouts_video: choice.unified_layouts,
        };
        let copy_queue = if copy_queue == decode {
            CopyQueue::Decode
        } else {
            CopyQueue::Own(Mutex::new(copy_queue))
        };
        Self::assemble(raw, copy_queue, Some(graphics), true, codecs)
    }

    fn assemble(
        raw: RawDevice,
        copy_queue: CopyQueue,
        graphics_queue: Option<vk::Queue>,
        owned: bool,
        codecs: vk::VideoCodecOperationFlagsKHR,
    ) -> Self {
        // SAFETY: the physical device is the instance's.
        let memory = unsafe {
            raw.instance
                .get_physical_device_memory_properties(raw.physical_device)
        };
        let video_instance = ash::khr::video_queue::Instance::new(&raw.entry, &raw.instance);
        let video = ash::khr::video_queue::Device::new(&raw.instance, &raw.device);
        let decode = ash::khr::video_decode_queue::Device::new(&raw.instance, &raw.device);
        let submitter = Self::submitter(&raw);
        Self {
            entry: raw.entry,
            instance: raw.instance,
            physical_device: raw.physical_device,
            device: raw.device,
            decode_family: raw.decode_family,
            decode_queue: Mutex::new(raw.decode_queue),
            copy_family: raw.copy_family,
            copy_queue,
            graphics_family: raw.graphics_family,
            graphics_queue,
            unified_layouts: raw.unified_image_layouts_video,
            codecs,
            memory,
            video_instance,
            video,
            decode,
            submitter,
            owned,
            encode: OnceLock::new(),
        }
    }

    /// The loader.
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    /// The instance.
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    /// The device.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// The physical device.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// The video-decode queue family.
    pub fn decode_family(&self) -> u32 {
        self.decode_family
    }

    /// The video-decode queue, which the decoders made on this device
    /// alone submit to: for a tool that hands the device to
    /// [`Self::from_raw`].
    pub fn decode_queue(&self) -> vk::Queue {
        *self.decode_queue.lock()
    }

    /// The family of the queue pictures are copied out on.
    pub fn copy_family(&self) -> u32 {
        self.copy_family
    }

    /// Whether pictures are copied out on the decode queue, in the decode's
    /// own submission.
    pub fn copies_on_decode_queue(&self) -> bool {
        matches!(self.copy_queue, CopyQueue::Decode)
    }

    /// The graphics queue family the outputs are shared with.
    pub fn graphics_family(&self) -> u32 {
        self.graphics_family
    }

    /// The graphics queue of a device [`Self::create`] made, which no
    /// decoder submits to: for a tool that hands the device to
    /// [`Self::from_raw`] with its copies on it. `None` on another's.
    pub fn graphics_queue(&self) -> Option<vk::Queue> {
        self.graphics_queue
    }

    /// Whether `unifiedImageLayoutsVideo` is enabled: decoded images are
    /// then kept in `VK_IMAGE_LAYOUT_GENERAL`.
    pub fn unified_image_layouts(&self) -> bool {
        self.unified_layouts
    }

    /// Whether `physical_device` lists `VK_KHR_unified_image_layouts` and
    /// reports both its features, `unifiedImageLayouts` and
    /// `unifiedImageLayoutsVideo`: for an owner that enables them where
    /// the device has them, what [`RawDevice::unified_image_layouts_video`]
    /// then is.
    pub fn reports_unified_image_layouts_video(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> bool {
        // SAFETY: the physical device is the instance's.
        let listed = unsafe { instance.enumerate_device_extension_properties(physical_device) }
            .is_ok_and(|extensions| {
                extensions.iter().any(|extension| {
                    extension.extension_name_as_c_str() == Ok(UNIFIED_IMAGE_LAYOUTS)
                })
            });
        if !listed {
            return false;
        }
        let mut unified = UnifiedImageLayoutsFeatures::new(false);
        let mut features = vk::PhysicalDeviceFeatures2 {
            p_next: std::ptr::from_mut(&mut unified).cast(),
            ..vk::PhysicalDeviceFeatures2::default()
        };
        // SAFETY: as above; the device lists the extension whose structure
        // is chained, which lives and is writable through the call.
        unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
        unified.unified_image_layouts == vk::TRUE && unified.unified_image_layouts_video == vk::TRUE
    }

    /// Whether a session's decodes run beside the copies out of earlier
    /// pictures: where decoded images stay in `GENERAL`
    /// ([`Self::unified_image_layouts`]), so that a copy changes no layout
    /// a decode reads, and the copies run on a queue of the decoders' own.
    /// A decode then waits only for the copies out of the images it
    /// writes, and the batch that ends a decode's work for its owner
    /// ([`Self::signal_after_decode`]) goes on the copy queue. Elsewhere
    /// each decode waits for the copy before it.
    pub fn overlaps_copies(&self) -> bool {
        self.unified_layouts && matches!(self.copy_queue, CopyQueue::Own(_))
    }

    /// The codec operations decoders on it decode: `DECODE_H264`, with
    /// `DECODE_H265` where the device was made for it.
    pub fn codecs(&self) -> vk::VideoCodecOperationFlagsKHR {
        self.codecs
    }

    /// The device's name, as the driver gives it.
    pub fn name(&self) -> String {
        // SAFETY: the physical device is the instance's.
        let properties = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        properties
            .device_name_as_c_str()
            .map_or_else(|_| "?".into(), |n| n.to_string_lossy().into_owned())
    }

    pub(crate) fn video(&self) -> &ash::khr::video_queue::Device {
        &self.video
    }

    pub(crate) fn video_instance(&self) -> &ash::khr::video_queue::Instance {
        &self.video_instance
    }

    pub(crate) fn decode(&self) -> &ash::khr::video_decode_queue::Device {
        &self.decode
    }

    /// The families in `families`, each once, in their order.
    fn unique(families: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(families.len());
        for &family in families {
            if !out.contains(&family) {
                out.push(family);
            }
        }
        out
    }

    /// The families an output is shared between: graphics work samples
    /// it, the decode queue writes it, the copy queue copies into it and
    /// reads it back.
    pub(crate) fn output_families(&self) -> Vec<u32> {
        Self::unique(&[self.graphics_family, self.decode_family, self.copy_family])
    }

    /// The families a DPB picture is shared between: the decode queue, and
    /// the copy queue that copies it out.
    pub(crate) fn dpb_families(&self) -> Vec<u32> {
        Self::unique(&[self.decode_family, self.copy_family])
    }

    /// Records `dependency` into `command`: `vkCmdPipelineBarrier2`, or
    /// its `VK_KHR_synchronization2` alias on an instance below 1.3.
    ///
    /// # Safety
    ///
    /// `command` is the device's and recording, and what `dependency`
    /// points at lives through the call.
    pub(crate) unsafe fn cmd_pipeline_barrier2(
        &self,
        command: vk::CommandBuffer,
        dependency: &vk::DependencyInfo<'_>,
    ) {
        match &self.submitter.synchronization2 {
            // SAFETY: as the caller promises.
            Some(khr) => unsafe { khr.cmd_pipeline_barrier2(command, dependency) },
            // SAFETY: as the caller promises.
            None => unsafe { self.device.cmd_pipeline_barrier2(command, dependency) },
        }
    }

    /// Submits `submits` to `queue` with `fence`: `vkQueueSubmit2`, or its
    /// `VK_KHR_synchronization2` alias on an instance below 1.3.
    ///
    /// # Safety
    ///
    /// `queue` is the device's and held by the caller for the call; the
    /// submissions' command buffers are recorded and their semaphores, and
    /// `fence` where given, the device's.
    pub(crate) unsafe fn queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), Error> {
        // SAFETY: as the caller promises.
        unsafe { self.submitter.queue_submit2(queue, submits, fence) }
    }

    /// Submits to the decode queue.
    pub(crate) fn submit_decode(&self, submits: &[vk::SubmitInfo2<'_>]) -> Result<(), Error> {
        let queue = self.decode_queue.lock();
        // SAFETY: the queue is the device's and is held for the call; the
        // submissions' command buffers are recorded and their semaphores
        // the device's.
        unsafe { self.queue_submit2(*queue, submits, vk::Fence::null()) }
    }

    /// Submits to the decode queue a batch of no work that waits for
    /// `waits` and then signals `signals`: the caller's own semaphores,
    /// ordered after a decoder's work by a wait for its timeline
    /// semaphore's value ([`crate::H264Decoder::submitted`]).
    pub fn signal_on_decode_queue(
        &self,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<(), Error> {
        let submits = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(waits)
            .signal_semaphore_infos(signals)];
        self.submit_decode(&submits)
    }

    /// Submits a batch of no work that waits for `waits` and then signals
    /// `signals`, as [`Self::signal_on_decode_queue`] does, on the queue
    /// whose work ends a decoder's: the copy queue where decodes run beside
    /// the copies ([`Self::overlaps_copies`]), so that a wait for a copy's
    /// value holds no later decode back, else the decode queue.
    pub fn signal_after_decode(
        &self,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<(), Error> {
        if !self.overlaps_copies() {
            return self.signal_on_decode_queue(waits, signals);
        }
        let submits = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(waits)
            .signal_semaphore_infos(signals)];
        self.submit_copy(&submits, vk::Fence::null())
    }

    /// Submits to the copy queue, with `fence`.
    pub(crate) fn submit_copy(
        &self,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> Result<(), Error> {
        let queue = match &self.copy_queue {
            CopyQueue::Decode => self.decode_queue.lock(),
            CopyQueue::Own(queue) => queue.lock(),
            // SAFETY: as in `submit_decode`, with the fence unsignalled;
            // each wait is of a decoder's timeline semaphore, to a value it
            // submitted, and a decoder's command buffers, semaphores and
            // fences live until its timeline's last value is reached.
            CopyQueue::Lent(lent) => return unsafe { lent.submit(submits, fence) },
        };
        // SAFETY: as in `submit_decode`, with the fence unsignalled.
        unsafe { self.queue_submit2(*queue, submits, fence) }
    }

    /// The first memory type of `bits` with every flag of one of `wanted`,
    /// tried in their order.
    pub(crate) fn memory_type(&self, bits: u32, wanted: &[vk::MemoryPropertyFlags]) -> Option<u32> {
        let count = usize::try_from(self.memory.memory_type_count).unwrap_or(0);
        let types = self.memory.memory_types.get(..count)?;
        wanted.iter().find_map(|&flags| {
            (0u32..)
                .zip(types)
                .find(|&(i, t)| bits & (1 << i) != 0 && t.property_flags.contains(flags))
                .map(|(i, _)| i)
        })
    }

    /// Memory for `requirements` of the first kind of `wanted` that holds
    /// it, dedicated to `image` where one is given.
    pub(crate) fn allocate(
        &self,
        requirements: &vk::MemoryRequirements,
        wanted: &[vk::MemoryPropertyFlags],
        image: Option<vk::Image>,
    ) -> Result<vk::DeviceMemory, Error> {
        let index = self
            .memory_type(requirements.memory_type_bits, wanted)
            .ok_or_else(|| {
                let wanted: Vec<String> = wanted
                    .iter()
                    .map(|f| format!("{:#x}", f.as_raw()))
                    .collect();
                Error::Unsupported(format!(
                    "no memory type of {:#x} has the flags of one of {wanted:?}",
                    requirements.memory_type_bits
                ))
            })?;
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default();
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index);
        if let Some(image) = image {
            dedicated = dedicated.image(image);
            info = info.push_next(&mut dedicated);
        }
        // SAFETY: `info` and its chain live through the call.
        unsafe { self.device.allocate_memory(&info, None) }.named("vkAllocateMemory")
    }
}

impl Drop for VideoDevice {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        // SAFETY: every object made on the device held this one and has
        // gone; the queues are idle once the wait returns.
        if let Err(error) = unsafe { self.device.device_wait_idle() } {
            log::error!("vkDeviceWaitIdle: {error}");
        }
        // SAFETY: as above: nothing made on the device is left.
        unsafe { self.device.destroy_device(None) };
        // SAFETY: the device made from it is destroyed.
        unsafe { self.instance.destroy_instance(None) };
        let _ = &self.entry;
    }
}

/// The device [`VideoDevice::create`] makes, on the best physical device
/// that decodes H.264.
fn make_device(instance: &ash::Instance) -> Result<(Choice, ash::Device), Error> {
    // SAFETY: the instance is valid.
    let physical_devices =
        unsafe { instance.enumerate_physical_devices() }.named("vkEnumeratePhysicalDevices")?;
    let mut passed_over = Vec::new();
    let mut best: Option<Choice> = None;
    for physical_device in physical_devices {
        match examine(instance, physical_device) {
            Ok(choice) => {
                if best.as_ref().is_none_or(|b| choice.rank < b.rank) {
                    best = Some(choice);
                }
            }
            Err(why) => passed_over.push(why),
        }
    }
    let Some(choice) = best else {
        return Err(Error::NoVideoDecode(if passed_over.is_empty() {
            "the loader lists no device".into()
        } else {
            passed_over.join("; ")
        }));
    };

    let priorities = [1.0f32; 2];
    let same = choice.decode_family == choice.graphics_family;
    let queue_infos: Vec<vk::DeviceQueueCreateInfo<'_>> = if same {
        let count = choice.family_queues.clamp(1, 2);
        let count = usize::try_from(count).unwrap_or(1);
        vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(choice.decode_family)
                .queue_priorities(priorities.get(..count).unwrap_or(&priorities)),
        ]
    } else {
        vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(choice.graphics_family)
                .queue_priorities(&priorities[..1]),
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(choice.decode_family)
                .queue_priorities(&priorities[..1]),
        ]
    };
    let mut queue_infos = queue_infos;
    if let Some(encode) = choice.encode.filter(|e| e.index == 0) {
        queue_infos.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(encode.family)
                .queue_priorities(&priorities[..1]),
        );
    } else if let Some(encode) = choice.encode {
        // The encode queue is the decode family's second.
        for info in &mut queue_infos {
            if info.queue_family_index == encode.family {
                *info = info.queue_priorities(&priorities[..2]);
            }
        }
    }
    let mut extensions: Vec<*const c_char> =
        REQUIRED_EXTENSIONS.iter().map(|e| e.as_ptr()).collect();
    if let Some(encode) = choice.encode {
        extensions.extend(encode.extensions().iter().map(|e| e.as_ptr()));
    }
    if choice.maintenance1 {
        extensions.push(ash::khr::video_maintenance1::NAME.as_ptr());
    }
    if choice.unified_layouts {
        extensions.push(UNIFIED_IMAGE_LAYOUTS.as_ptr());
    }
    if choice.decodes_h265 {
        extensions.push(H265_EXTENSION.as_ptr());
    }
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
    let mut v13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
    let mut maintenance1 =
        vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default().video_maintenance1(true);
    let mut unified = UnifiedImageLayoutsFeatures::new(true);
    let mut info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions)
        .push_next(&mut v12)
        .push_next(&mut v13);
    if choice.maintenance1 {
        info = info.push_next(&mut maintenance1);
    }
    if choice.unified_layouts {
        unified.p_next = info.p_next.cast_mut();
        info.p_next = std::ptr::from_mut(&mut unified).cast_const().cast();
    }
    // SAFETY: `info` and everything its chain points at live through the
    // call; each extension and feature asked for is the device's.
    let device = unsafe { instance.create_device(choice.physical_device, &info, None) }
        .named("vkCreateDevice")?;
    Ok((choice, device))
}

/// Whether `physical_device` decodes H.264 through Vulkan Video, and what
/// a device of it would be made with; why not where it does not.
fn examine(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<Choice, String> {
    // SAFETY: the physical device is the instance's; so for each query.
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let name = properties
        .device_name_as_c_str()
        .map_or_else(|_| "?".into(), |n| n.to_string_lossy().into_owned());
    if properties.api_version < vk::API_VERSION_1_3 {
        return Err(format!("{name}: Vulkan 1.3 is needed"));
    }
    // SAFETY: as above.
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .map_err(|e| format!("{name}: vkEnumerateDeviceExtensionProperties: {e}"))?;
    let has = |wanted: &CStr| {
        extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(wanted))
    };
    if let Some(missing) = REQUIRED_EXTENSIONS.iter().find(|e| !has(e)) {
        return Err(format!("{name}: no {}", missing.to_string_lossy()));
    }

    // SAFETY: as above.
    let count =
        unsafe { instance.get_physical_device_queue_family_properties2_len(physical_device) };
    let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
    let mut families: Vec<vk::QueueFamilyProperties2<'_>> = video
        .iter_mut()
        .map(|v| vk::QueueFamilyProperties2::default().push_next(v))
        .collect();
    // SAFETY: as above, with an output of the length the call gave.
    unsafe {
        instance.get_physical_device_queue_family_properties2(physical_device, &mut families)
    };
    let flags: Vec<(vk::QueueFlags, u32)> = families
        .iter()
        .map(|f| {
            (
                f.queue_family_properties.queue_flags,
                f.queue_family_properties.queue_count,
            )
        })
        .collect();
    drop(families);
    let decode_family = (0u32..)
        .zip(flags.iter().zip(&video))
        .find(|(_, ((f, _), v))| {
            f.contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                && v.video_codec_operations
                    .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
        })
        .map(|(i, _)| i)
        .ok_or_else(|| format!("{name}: no queue family decodes H.264"))?;
    let (graphics_family, family_queues) = (0u32..)
        .zip(&flags)
        .find(|(_, (f, _))| f.contains(vk::QueueFlags::GRAPHICS))
        .map(|(i, &(_, count))| (i, count))
        .ok_or_else(|| format!("{name}: no graphics queue family"))?;
    let decode_family_copies = flags
        .get(usize::try_from(decode_family).unwrap_or(usize::MAX))
        .is_some_and(|&(f, _)| copies(f));
    let decodes_h265 = has(H265_EXTENSION)
        && video
            .get(usize::try_from(decode_family).unwrap_or(usize::MAX))
            .is_some_and(|v| {
                v.video_codec_operations
                    .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
            });

    let maintenance1_listed = has(ash::khr::video_maintenance1::NAME);
    let unified_listed = has(UNIFIED_IMAGE_LAYOUTS);
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut maintenance1 = vk::PhysicalDeviceVideoMaintenance1FeaturesKHR::default();
    let mut unified = UnifiedImageLayoutsFeatures::new(false);
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut v12)
        .push_next(&mut v13);
    if maintenance1_listed {
        features = features.push_next(&mut maintenance1);
    }
    if unified_listed {
        unified.p_next = features.p_next;
        features.p_next = std::ptr::from_mut(&mut unified).cast();
    }
    // SAFETY: as above, with every structure of the chain alive and
    // writable through the call.
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
    if v12.timeline_semaphore == vk::FALSE || v13.synchronization2 == vk::FALSE {
        return Err(format!(
            "{name}: timeline semaphores and synchronization2 are needed"
        ));
    }
    let encode = EncodeChoice::find(
        &flags,
        &video,
        &has,
        decode_family,
        graphics_family,
        flags
            .get(usize::try_from(decode_family).unwrap_or(usize::MAX))
            .map_or(0, |&(_, count)| count),
    );
    let rank = match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        _ => 3,
    };
    Ok(Choice {
        physical_device,
        decode_family,
        graphics_family,
        family_queues,
        decode_family_copies,
        decodes_h265,
        maintenance1: maintenance1_listed && maintenance1.video_maintenance1 == vk::TRUE,
        unified_layouts: unified_listed
            && unified.unified_image_layouts == vk::TRUE
            && unified.unified_image_layouts_video == vk::TRUE,
        rank,
        encode,
    })
}
