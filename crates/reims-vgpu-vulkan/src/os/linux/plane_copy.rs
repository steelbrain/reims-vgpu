//! Copies by the GPU of planes another driver wrote into dma-bufs, a VA-API decode surface's
//! planes exported with `vaExportSurfaceHandle`, into a runtime buffer, at the rows and offset
//! the buffer's planes have ([`PlaneCopier`]): the reims-graphics device's VA-API pictures land so
//! in their output surfaces, whose planes lie 32 bytes into their pages, as CoreVideo lays them
//! out, where no driver's own video processing writes.
//!
//! - **The source.** Each plane is imported as a single-plane image of its own format (R8 and
//!   RG8 for NV12, R16 and RG16 for P010) over its dma-buf, at its offset and pitch, tiled as the
//!   dma-buf's DRM format modifier says (`VK_EXT_image_drm_format_modifier`,
//!   `VK_EXT_external_memory_dma_buf`), which the runtime enables wherever the device lists them.
//!   An import is kept by its key for later copies of the same plane, until its owner says it is
//!   gone ([`DmaBufPlane::alive`]) or the copier goes; the driver's surfaces are lent again
//!   picture after picture, so each is imported once. The image is taken from the driver's queue
//!   family (`VK_QUEUE_FAMILY_FOREIGN_EXT`, in `GENERAL`) for the copy and handed back after it.
//! - **The destination** is a runtime buffer whose bytes are its memory: one in the device's own
//!   memory, or over host pages the device imported in place. A buffer whose pages are a copy the
//!   runtime keeps (`copied-host-pages`), or that is detached, or whose GPU writes never reach its
//!   memory, is refused: a write into its `VkBuffer` would never reach the pages.
//! - **The copy** is one command buffer of the copier's own on a runtime queue of its own, whose
//!   `VkQueue` it borrows for the submission, as `crate::awake` does: `vkCmdCopyImageToBuffer` of
//!   each plane's texels into the buffer at the plane's first byte, its rows `bytes_per_row` apart,
//!   then a barrier that makes the writes visible to every later command and the host. The copier
//!   waits for it on its fence before it returns, so the runtime's later work on the buffer,
//!   which the caller orders after the return, reads the planes. Its GPU time is measured with two
//!   timestamps where the queue's family writes them.
//! - **A copy past its wait** is kept, with the caller's hold of the source (so the driver writes
//!   nothing more into it), until its fence signals; the copier refuses copies meanwhile.
//! - **Refusals** ([`CopyError::Refused`]) are made before anything is submitted, so the caller
//!   lands the planes another way; a failure once submitted ([`CopyError::Failed`]) is not, since
//!   the GPU may still write the buffer.
//! - **Across the device's recoveries.** The copier's objects and imports are of the generation
//!   of the device they were made on (`crate::generation`). Where the runtime has made its
//!   `VkDevice` again since, a copy first destroys them on the old one and makes them on the new
//!   one. A buffer made before the recovery is refused, and so is a copy whose queue a recovery
//!   moved as it was submitted, with nothing submitted.
//!
//! **Measured** with a libva and Vulkan program written from scratch (2026-11-08), surfaces filled
//! by `vaPutImage`, each plane copied into a buffer at 0x20 with rows of the frame's bytes, every
//! byte compared, the bytes before 0x20 untouched: RADV (Mesa 26.2.6) on a Radeon 780M imported
//! radeonsi's planes (modifier `0x200000010401b04`) and copied a 3840x2160 NV12 picture in 0.65 ms
//! (luma) and 0.16 ms (chroma) of GPU time, P010 in 0.65 and 0.21 ms; ANV on an Iris Xe imported
//! iHD's (Y tiling, `0x100000000000002`) and copied NV12 in 0.24 and 0.12 ms, P010 in 0.47 and
//! 0.24 ms. Every byte equalled the picture put in, at 3840x2160 and 1920x1080.

use std::any::Any;
use std::os::fd::{AsRawFd as _, BorrowedFd, IntoRawFd as _};
use std::sync::Weak;
use std::time::Duration;

use ash::vk;
use metal2vulkan_runtime as rt;

use crate::core::platform::Handle as _;
use crate::generation::Generation;
use crate::gpu::device::VulkanDevice;
use crate::gpu::resource::VulkanBuffer;

/// The most imports a copier keeps: the planes of the surfaces of a few pools of a decoder's.
const MOST_IMPORTS: usize = 128;

/// How long the copier's end waits for a copy kept past its wait.
const LAST_WAIT: Duration = Duration::from_secs(1);

/// A plane's format: its texels' components and bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneFormat {
    /// One byte a texel: NV12's luma.
    R8,
    /// Two bytes a texel: NV12's interleaved chroma.
    Rg8,
    /// Two bytes a texel: P010's luma.
    R16,
    /// Four bytes a texel: P010's interleaved chroma.
    Rg16,
}

impl PlaneFormat {
    fn vk(self) -> vk::Format {
        match self {
            PlaneFormat::R8 => vk::Format::R8_UNORM,
            PlaneFormat::Rg8 => vk::Format::R8G8_UNORM,
            PlaneFormat::R16 => vk::Format::R16_UNORM,
            PlaneFormat::Rg16 => vk::Format::R16G16_UNORM,
        }
    }

    /// The bytes of one texel.
    pub fn texel_bytes(self) -> u64 {
        match self {
            PlaneFormat::R8 => 1,
            PlaneFormat::Rg8 | PlaneFormat::R16 => 2,
            PlaneFormat::Rg16 => 4,
        }
    }
}

/// One plane in a dma-buf, as its driver exported it.
#[derive(Clone, Debug)]
pub struct DmaBufPlane<'a> {
    /// What names the plane's import while it is kept: a key names one dma-buf, plane and layout.
    pub key: u64,
    /// Lives while the dma-buf's owner keeps it: an import whose owner is gone is let go.
    pub alive: Weak<()>,
    /// The dma-buf, which the copier duplicates for its import.
    pub fd: BorrowedFd<'a>,
    /// Its DRM format modifier.
    pub modifier: u64,
    /// The plane's first byte in it.
    pub offset: u64,
    /// The bytes from one of its rows to the next, as the modifier counts them.
    pub pitch: u64,
    /// Its texels' format.
    pub format: PlaneFormat,
    /// Its width in texels.
    pub width: u32,
    /// Its height in texels.
    pub height: u32,
}

/// One plane's copy: `extent` texels from `origin` of `source`, into `to` from its byte `offset`,
/// rows `bytes_per_row` apart.
#[derive(Clone, Debug)]
pub struct PlaneCopy<'a> {
    /// The plane copied from.
    pub source: DmaBufPlane<'a>,
    /// The first texel copied, its column and row.
    pub origin: [u32; 2],
    /// The texels copied: columns and rows.
    pub extent: [u32; 2],
    /// The buffer written.
    pub to: &'a VulkanBuffer,
    /// The byte of `to` the first row starts at.
    pub offset: u64,
    /// The bytes from one row to the next in `to`.
    pub bytes_per_row: u64,
}

/// Why a copy did not land ([`PlaneCopier::copy`]).
#[derive(Debug)]
pub enum CopyError {
    /// Nothing was submitted: the planes are to land another way.
    Refused(String),
    /// The copy was submitted and failed or ran past its wait: the GPU may write the buffers
    /// still.
    Failed(String),
}

impl std::fmt::Display for CopyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopyError::Refused(why) => write!(f, "refused: {why}"),
            CopyError::Failed(why) => write!(f, "failed: {why}"),
        }
    }
}

/// A kept import of one plane.
struct Import {
    key: u64,
    alive: Weak<()>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// The copy that used it last, by the copier's count.
    used: u64,
}

/// A copy past its wait, kept until its fence signals, with the caller's hold of its sources and
/// the buffers it writes.
struct Pending {
    _hold: Box<dyn Any + Send>,
    _targets: Vec<VulkanBuffer>,
}

/// The copies of one caller, a decode session's (module docs).
pub struct PlaneCopier {
    queue: rt::CommandQueue,
    /// The objects, on the generation of the device they were made on; `None` where their making
    /// after a recovery was refused, until a later copy makes them.
    objects: Option<Objects>,
    /// The generations the copier made its objects on: one more at each making after the first.
    generations: u64,
}

/// A copier's objects on one generation of the runtime's device (`crate::generation`), made with
/// its handles and destroyed with them, at the copier's end or once a recovery moved the device
/// to a new `VkDevice`.
struct Objects {
    generation: Generation,
    memory_fd: ash::khr::external_memory_fd::Device,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Two timestamps around the copies, where the queue's family writes them, and the
    /// nanoseconds a tick.
    timestamps: Option<(vk::QueryPool, f64)>,
    imports: Vec<Import>,
    /// The modifiers and formats the device refused to import, with why.
    refused: Vec<(u64, PlaneFormat, String)>,
    copies: u64,
    pending: Option<Pending>,
}

impl PlaneCopier {
    /// A copier on `device`'s runtime device, with a queue of its own; `Err` with why where the
    /// device imports no dma-buf as an image or refuses the copier's objects.
    pub fn new(device: &VulkanDevice) -> Result<PlaneCopier, String> {
        let queue = (device.raw.new_command_queue()).map_err(|e| format!("no queue ({e:?})"))?;
        let objects = Objects::new(&queue)?;
        Ok(PlaneCopier {
            queue,
            objects: Some(objects),
            generations: 1,
        })
    }

    /// Copies each of `copies` (module docs) and waits up to `wait` for the GPU: `Ok` with the
    /// copies' GPU time where the queue measures it, once every plane is in its buffer. Past the
    /// wait the copy is kept with `hold`, which keeps the sources unwritten, until its fence
    /// signals ([`CopyError::Failed`]). Where the device recovered since the copier's objects
    /// were made, they are destroyed and made again on its new `VkDevice` first.
    pub fn copy(
        &mut self,
        copies: &[PlaneCopy<'_>],
        wait: Duration,
        hold: Box<dyn Any + Send>,
    ) -> Result<Option<Duration>, CopyError> {
        if rt::process_exiting() {
            return Err(CopyError::Refused("the process is exiting".to_owned()));
        }
        let serves = (self.objects.as_ref()).is_some_and(|o| o.generation.serves(&self.queue));
        if !serves {
            // The old objects go first, on the `VkDevice` they were made on.
            self.objects = None;
            let objects = Objects::new(&self.queue).map_err(|why| {
                CopyError::Refused(format!("its objects after the device's recovery: {why}"))
            })?;
            self.generations += 1;
            log::info!(
                "a plane copier's objects were made again on the device's new VkDevice, after \
                 its recovery ({} generations)",
                self.generations
            );
            self.objects = Some(objects);
        }
        let Some(objects) = self.objects.as_mut() else {
            return Err(CopyError::Refused("the copier has no objects".to_owned()));
        };
        objects.copy(&self.queue, copies, wait, hold)
    }

    /// The generations of the device the copier made its objects on: 1, and one more each time
    /// a recovery moved the device to a new `VkDevice` before a copy.
    pub fn generations(&self) -> u64 {
        self.generations
    }
}

impl Objects {
    /// The objects on the generation of `queue`'s device that serves; `Err` with why where the
    /// device imports no dma-buf as an image or refuses them.
    fn new(queue: &rt::CommandQueue) -> Result<Objects, String> {
        let generation = Generation::serving(queue)?;
        let family = generation.family();
        let (vk_device, timestamped, period, memory_fd) = {
            // SAFETY: the handles are used while `generation`, which the objects hold, lives; the
            // objects made of them are their own, destroyed at their end.
            let handles = unsafe { generation.raw() };
            let instance = handles.instance;
            // SAFETY: the physical device is the device's own.
            let (families, extensions) = unsafe {
                (
                    instance.get_physical_device_queue_family_properties(handles.physical_device),
                    instance.enumerate_device_extension_properties(handles.physical_device),
                )
            };
            let listed = |name: &std::ffi::CStr| {
                (extensions.iter().flatten()).any(|e| e.extension_name_as_c_str() == Ok(name))
            };
            let needed = [
                ash::khr::external_memory_fd::NAME,
                ash::ext::external_memory_dma_buf::NAME,
                ash::ext::image_drm_format_modifier::NAME,
                ash::ext::queue_family_foreign::NAME,
            ];
            if let Some(missing) = needed.iter().find(|name| !listed(name)) {
                return Err(format!("the device lists no {}", missing.to_string_lossy()));
            }
            let family_props = usize::try_from(family).ok().and_then(|at| families.get(at));
            let timestamped = family_props.is_some_and(|f| f.timestamp_valid_bits > 0);
            // SAFETY: the physical device is the device's own.
            let period =
                unsafe { instance.get_physical_device_properties(handles.physical_device) }
                    .limits
                    .timestamp_period;
            let memory_fd = ash::khr::external_memory_fd::Device::new(instance, handles.device);
            (handles.device.clone(), timestamped, period, memory_fd)
        };
        let device = generation.device().clone();
        let mut objects = Objects {
            generation,
            memory_fd,
            pool: vk::CommandPool::null(),
            buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            timestamps: None,
            imports: Vec::new(),
            refused: Vec::new(),
            copies: 0,
            pending: None,
        };
        // An object made before a refusal is destroyed by the objects' end.
        device
            .with_loader_word(|| objects.make(&vk_device, timestamped, f64::from(period)))
            .map_err(|e| format!("its objects were refused ({e})"))?;
        Ok(objects)
    }

    /// Makes the copier's command pool, buffer, fence and, where `timestamped`, its query pool of
    /// two timestamps of `period` nanoseconds a tick.
    fn make(
        &mut self,
        vk_device: &ash::Device,
        timestamped: bool,
        period: f64,
    ) -> Result<(), vk::Result> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.generation.family());
        // SAFETY: the infos are valid, and every object is this copier's.
        unsafe {
            self.pool = vk_device.create_command_pool(&pool_info, None)?;
            let buffer_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.buffer = vk_device.allocate_command_buffers(&buffer_info)?[0];
            self.fence = vk_device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            if timestamped {
                let query_info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2);
                self.timestamps = Some((vk_device.create_query_pool(&query_info, None)?, period));
            }
        }
        Ok(())
    }

    /// [`PlaneCopier::copy`] with these objects, on `queue`, the copier's.
    fn copy(
        &mut self,
        queue: &rt::CommandQueue,
        copies: &[PlaneCopy<'_>],
        wait: Duration,
        hold: Box<dyn Any + Send>,
    ) -> Result<Option<Duration>, CopyError> {
        let device = self.generation.device().clone();
        // SAFETY: the handles are used while the generation, which `device` holds, lives.
        let vk_device = unsafe { device.raw() }.device;
        if self.pending.is_some() {
            let fence = self.fence;
            // SAFETY: the fence is this copier's.
            let done = device.with_loader_word(|| unsafe { vk_device.get_fence_status(fence) });
            match done {
                // SAFETY: the fence is this copier's, signalled, and in no submission.
                Ok(true) => {
                    match device.with_loader_word(|| unsafe { vk_device.reset_fences(&[fence]) }) {
                        Ok(()) => self.pending = None,
                        Err(e) => {
                            return Err(CopyError::Refused(format!(
                                "its fence's reset answered {e}"
                            )));
                        }
                    }
                }
                Ok(false) => {
                    return Err(CopyError::Refused(
                        "an earlier copy has not finished on the GPU".to_owned(),
                    ));
                }
                Err(e) => return Err(CopyError::Refused(format!("its fence answered {e}"))),
            }
        }
        let mut targets = Vec::with_capacity(copies.len());
        for copy in copies {
            targets.push(destination(copy, &self.generation)?);
        }
        self.let_go_of_gone(vk_device);
        self.copies += 1;
        let mut images = Vec::with_capacity(copies.len());
        for copy in copies {
            images.push(self.import(vk_device, &copy.source)?);
        }
        device
            .with_loader_word(|| self.record(vk_device, copies, &images, &targets))
            .map_err(|e| CopyError::Refused(format!("its command buffer was refused ({e})")))?;
        let buffers = [self.buffer];
        let submit = [vk::SubmitInfo::default().command_buffers(&buffers)];
        let fence = self.fence;
        // SAFETY: the submission is the closure's only call, keeps no handle and waits for
        // nothing; the buffer is executable and not pending (no copy is pending), and the fence
        // is unsignalled. The queue is the one the generation was taken with.
        let submitted = unsafe {
            self.generation
                .with_queue(queue, |queue| vk_device.queue_submit(queue, &submit, fence))
        };
        let Some(submitted) = submitted else {
            return Err(CopyError::Refused(
                "the device recovered as the copy was submitted: nothing was submitted".to_owned(),
            ));
        };
        submitted.map_err(|e| CopyError::Refused(format!("its submission was refused ({e})")))?;
        let pending = |hold: Box<dyn Any + Send>| Pending {
            _hold: hold,
            _targets: copies.iter().map(|copy| copy.to.clone_ref()).collect(),
        };
        let nanos = u64::try_from(wait.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the fence is this copier's.
        let waited =
            device.with_loader_word(|| unsafe { vk_device.wait_for_fences(&[fence], true, nanos) });
        match waited {
            Ok(()) => {}
            Err(vk::Result::TIMEOUT) => {
                self.pending = Some(pending(hold));
                return Err(CopyError::Failed(format!(
                    "the GPU had not copied the planes {:.0} ms on",
                    wait.as_secs_f64() * 1e3
                )));
            }
            Err(e) => {
                self.pending = Some(pending(hold));
                return Err(CopyError::Failed(format!("its fence answered {e}")));
            }
        }
        // SAFETY: the fence is this copier's, signalled.
        let reset = device.with_loader_word(|| unsafe { vk_device.reset_fences(&[fence]) });
        if let Err(e) = reset {
            self.pending = Some(pending(hold));
            return Err(CopyError::Failed(format!("its fence's reset answered {e}")));
        }
        drop(hold);
        Ok(self.gpu_time(vk_device))
    }

    /// The last copy's GPU time, where the queue measured it.
    fn gpu_time(&self, vk_device: &ash::Device) -> Option<Duration> {
        let (pool, period) = self.timestamps?;
        let mut ticks = [0u64; 2];
        // SAFETY: the pool is this copier's, its two queries written by the copy waited for.
        let read = self.generation.device().with_loader_word(|| unsafe {
            vk_device.get_query_pool_results(pool, 0, &mut ticks, vk::QueryResultFlags::TYPE_64)
        });
        read.ok()?;
        let nanos = (ticks[1].saturating_sub(ticks[0]) as f64) * period;
        Some(Duration::from_nanos(nanos as u64))
    }

    /// Lets go of the imports whose owners are gone.
    fn let_go_of_gone(&mut self, vk_device: &ash::Device) {
        let (gone, kept): (Vec<Import>, Vec<Import>) = std::mem::take(&mut self.imports)
            .into_iter()
            .partition(|import| import.alive.strong_count() == 0);
        self.imports = kept;
        self.destroy(vk_device, gone);
    }

    /// Destroys `imports`, which no pending copy uses.
    fn destroy(&self, vk_device: &ash::Device, imports: Vec<Import>) {
        if imports.is_empty() {
            return;
        }
        self.generation.device().with_loader_word(|| {
            for import in imports {
                // SAFETY: the import's objects are this copier's, and no copy in flight uses
                // them: every copy that did was waited for.
                unsafe {
                    vk_device.destroy_image(import.image, None);
                    vk_device.free_memory(import.memory, None);
                }
            }
        });
    }

    /// The image of `plane`, imported now or kept from an earlier copy.
    fn import(
        &mut self,
        vk_device: &ash::Device,
        plane: &DmaBufPlane<'_>,
    ) -> Result<vk::Image, CopyError> {
        let copies = self.copies;
        if let Some(kept) = self.imports.iter_mut().find(|i| i.key == plane.key) {
            kept.used = copies;
            return Ok(kept.image);
        }
        if let Some((.., why)) = (self.refused.iter())
            .find(|(modifier, format, _)| *modifier == plane.modifier && *format == plane.format)
        {
            return Err(CopyError::Refused(why.clone()));
        }
        if let Err(why) = self.importable(plane) {
            self.refused
                .push((plane.modifier, plane.format, why.clone()));
            return Err(CopyError::Refused(why));
        }
        if self.imports.len() >= MOST_IMPORTS
            && let Some(oldest) = (self.imports.iter().enumerate())
                .min_by_key(|(_, import)| import.used)
                .map(|(at, _)| at)
        {
            let import = self.imports.swap_remove(oldest);
            self.destroy(vk_device, vec![import]);
        }
        let device = self.generation.device().clone();
        let (image, memory) = device
            .with_loader_word(|| self.make_import(vk_device, plane))
            .map_err(|why| CopyError::Refused(format!("the plane's import: {why}")))?;
        self.imports.push(Import {
            key: plane.key,
            alive: plane.alive.clone(),
            image,
            memory,
            used: copies,
        });
        Ok(image)
    }

    /// `Ok` where the device imports a dma-buf of `plane`'s modifier as an image of its format
    /// that copies out; `Err` with why otherwise.
    fn importable(&self, plane: &DmaBufPlane<'_>) -> Result<(), String> {
        // SAFETY: the handles are used while the generation lives.
        let handles = unsafe { self.generation.raw() };
        let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(plane.modifier)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(plane.format.vk())
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .push_next(&mut modifier)
            .push_next(&mut external);
        let mut external_props = vk::ExternalImageFormatProperties::default();
        let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
        // SAFETY: the physical device is the device's own, and the structures are valid.
        let answered = unsafe {
            handles
                .instance
                .get_physical_device_image_format_properties2(
                    handles.physical_device,
                    &info,
                    &mut props,
                )
        };
        let format = plane.format;
        let modifier = plane.modifier;
        answered.map_err(|e| {
            format!("the device makes no {format:?} image of modifier {modifier:#x} from a dma-buf ({e})")
        })?;
        let max = props.image_format_properties.max_extent;
        if !(external_props
            .external_memory_properties
            .external_memory_features)
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        {
            return Err(format!(
                "the device imports no dma-buf as a {format:?} image of modifier {modifier:#x}"
            ));
        }
        if plane.width > max.width || plane.height > max.height {
            return Err(format!(
                "a {format:?} image of modifier {modifier:#x} is at most {}x{}, the plane {}x{}",
                max.width, max.height, plane.width, plane.height
            ));
        }
        Ok(())
    }

    /// An image over `plane`'s dma-buf, and its import (module docs).
    fn make_import(
        &self,
        vk_device: &ash::Device,
        plane: &DmaBufPlane<'_>,
    ) -> Result<(vk::Image, vk::DeviceMemory), String> {
        let layouts = [vk::SubresourceLayout {
            offset: plane.offset,
            size: 0,
            row_pitch: plane.pitch,
            array_pitch: 0,
            depth_pitch: 0,
        }];
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(plane.modifier)
            .plane_layouts(&layouts);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(plane.format.vk())
            .extent(vk::Extent3D {
                width: plane.width,
                height: plane.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut explicit)
            .push_next(&mut external);
        // SAFETY: the info is valid; the image is this copier's, destroyed below on a refusal.
        let image = unsafe { vk_device.create_image(&info, None) }
            .map_err(|e| format!("vkCreateImage answered {e}"))?;
        let imported = self.import_memory(vk_device, image, plane);
        match imported {
            Ok(memory) => Ok((image, memory)),
            Err(why) => {
                // SAFETY: the image is this copier's, bound to nothing and used by nothing.
                unsafe { vk_device.destroy_image(image, None) };
                Err(why)
            }
        }
    }

    /// `plane`'s dma-buf imported for `image`, and bound to it.
    fn import_memory(
        &self,
        vk_device: &ash::Device,
        image: vk::Image,
        plane: &DmaBufPlane<'_>,
    ) -> Result<vk::DeviceMemory, String> {
        let fd =
            (plane.fd.try_clone_to_owned()).map_err(|e| format!("the dma-buf's duplicate: {e}"))?;
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: a dma-buf, and a structure to fill.
        unsafe {
            self.memory_fd.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut fd_props,
            )
        }
        .map_err(|e| format!("vkGetMemoryFdPropertiesKHR answered {e}"))?;
        // SAFETY: the image is this copier's.
        let needs = unsafe { vk_device.get_image_memory_requirements(image) };
        let types = needs.memory_type_bits & fd_props.memory_type_bits;
        if types == 0 {
            return Err(format!(
                "no memory type both the image ({:#x}) and the dma-buf ({:#x}) take",
                needs.memory_type_bits, fd_props.memory_type_bits
            ));
        }
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let raw_fd = fd.as_raw_fd();
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(raw_fd);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(types.trailing_zeros())
            .push_next(&mut import)
            .push_next(&mut dedicated);
        // SAFETY: the info is valid; on success the import owns the descriptor, which is then
        // released from `fd` so it is closed once, by the driver.
        let memory = unsafe { vk_device.allocate_memory(&info, None) }
            .map_err(|e| format!("the dma-buf's import answered {e}"))?;
        let _owned_by_the_import = fd.into_raw_fd();
        // SAFETY: the memory was made for the image, which is bound to nothing yet.
        if let Err(e) = unsafe { vk_device.bind_image_memory(image, memory, 0) } {
            // SAFETY: the memory is this copier's, bound to nothing.
            unsafe { vk_device.free_memory(memory, None) };
            return Err(format!("vkBindImageMemory answered {e}"));
        }
        Ok(memory)
    }

    /// Records the copies of `copies` from `images` into `targets` (module docs).
    fn record(
        &self,
        vk_device: &ash::Device,
        copies: &[PlaneCopy<'_>],
        images: &[vk::Image],
        targets: &[(vk::Buffer, u64)],
    ) -> Result<(), vk::Result> {
        let color = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = |image, acquire: bool| {
            let (old, new) = (
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            let (foreign, ours) = (vk::QUEUE_FAMILY_FOREIGN_EXT, self.generation.family());
            let b = vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(color);
            if acquire {
                b.old_layout(old)
                    .new_layout(new)
                    .src_queue_family_index(foreign)
                    .dst_queue_family_index(ours)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            } else {
                b.old_layout(new)
                    .new_layout(old)
                    .src_queue_family_index(ours)
                    .dst_queue_family_index(foreign)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            }
        };
        let acquires: Vec<_> = images.iter().map(|&image| barrier(image, true)).collect();
        let releases: Vec<_> = images.iter().map(|&image| barrier(image, false)).collect();
        let cb = self.buffer;
        // SAFETY: the buffer is this copier's and not pending; every handle recorded is live
        // through the copy (the images are kept, the buffers held by the caller's copies).
        unsafe {
            vk_device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            vk_device.begin_command_buffer(cb, &begin)?;
            if let Some((pool, _)) = self.timestamps {
                vk_device.cmd_reset_query_pool(cb, pool, 0, 2);
            }
            vk_device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &acquires,
            );
            if let Some((pool, _)) = self.timestamps {
                vk_device.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0);
            }
            for ((copy, &image), &(buffer, offset)) in copies.iter().zip(images).zip(targets) {
                let texel = copy.source.format.texel_bytes();
                let region = vk::BufferImageCopy {
                    buffer_offset: offset,
                    buffer_row_length: u32::try_from(copy.bytes_per_row / texel)
                        .unwrap_or(u32::MAX),
                    buffer_image_height: copy.extent[1],
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D {
                        x: i32::try_from(copy.origin[0]).unwrap_or(i32::MAX),
                        y: i32::try_from(copy.origin[1]).unwrap_or(i32::MAX),
                        z: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: copy.extent[0],
                        height: copy.extent[1],
                        depth: 1,
                    },
                };
                vk_device.cmd_copy_image_to_buffer(
                    cb,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    buffer,
                    &[region],
                );
            }
            if let Some((pool, _)) = self.timestamps {
                vk_device.cmd_write_timestamp(cb, vk::PipelineStageFlags::TRANSFER, pool, 1);
            }
            vk_device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &releases,
            );
            let written = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::MEMORY_READ
                        | vk::AccessFlags::MEMORY_WRITE
                        | vk::AccessFlags::HOST_READ,
                )];
            vk_device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &written,
                &[],
                &[],
            );
            vk_device.end_command_buffer(cb)
        }
    }
}

/// The `VkBuffer` `copy` writes and its byte there: `Err` where its buffer is of another
/// generation of the device than the copier's `generation` (made before the device's recovery),
/// its bytes are not its memory, or the copy lies past the buffer or off its texels' alignment
/// (module docs).
fn destination(
    copy: &PlaneCopy<'_>,
    generation: &Generation,
) -> Result<(vk::Buffer, u64), CopyError> {
    let refused = |why: String| CopyError::Refused(why);
    let buffer = &copy.to.raw;
    if !generation.is_of(buffer.device()) {
        return Err(refused(
            "the buffer was made on the device's VkDevice before its recovery".to_owned(),
        ));
    }
    if copy.to.gpu_read_only {
        return Err(refused("the buffer drops the GPU's writes".to_owned()));
    }
    if copy.to.backing.is_detached() {
        return Err(refused("the buffer is detached from its memory".to_owned()));
    }
    match buffer.host_backing() {
        Some(rt::HostBacking::Copied) => {
            return Err(refused(
                "the buffer's pages are a copy the runtime keeps, which the GPU's own writes \
                 would not reach"
                    .to_owned(),
            ));
        }
        Some(rt::HostBacking::Detached) => {
            return Err(refused("the buffer is detached from its pages".to_owned()));
        }
        _ => {}
    }
    let texel = copy.source.format.texel_bytes();
    let [width, height] = copy.extent;
    if width == 0 || height == 0 {
        return Err(refused("an empty plane".to_owned()));
    }
    if !copy.bytes_per_row.is_multiple_of(texel) || !copy.offset.is_multiple_of(texel) {
        return Err(refused(format!(
            "a plane at {} with rows of {} bytes, not of whole {texel}-byte texels",
            copy.offset, copy.bytes_per_row
        )));
    }
    let row = u64::from(width) * texel;
    let end = (copy.bytes_per_row.checked_mul(u64::from(height - 1)))
        .and_then(|rows| rows.checked_add(row))
        .and_then(|bytes| bytes.checked_add(copy.offset));
    if row > copy.bytes_per_row || !end.is_some_and(|end| end <= buffer.length()) {
        return Err(refused(format!(
            "a plane of {height} rows of {row} bytes, {} apart, at {}, past its buffer's {} bytes",
            copy.bytes_per_row,
            copy.offset,
            buffer.length()
        )));
    }
    let (origin_x, origin_y) = (u64::from(copy.origin[0]), u64::from(copy.origin[1]));
    if origin_x + u64::from(width) > u64::from(copy.source.width)
        || origin_y + u64::from(height) > u64::from(copy.source.height)
    {
        return Err(refused(format!(
            "{width}x{height} texels from ({origin_x}, {origin_y}) of a {}x{} plane",
            copy.source.width, copy.source.height
        )));
    }
    // SAFETY: the copier uses the `VkBuffer` only while the caller's copy holds the buffer, and
    // orders its write against the runtime's work as the module docs say.
    let (vk_buffer, base) = unsafe { buffer.vk_buffer() };
    let at = base + copy.offset;
    if !at.is_multiple_of(texel) {
        return Err(refused(format!(
            "a plane at byte {at} of its VkBuffer, not on a {texel}-byte texel"
        )));
    }
    Ok((vk_buffer, at))
}

impl Drop for Objects {
    fn drop(&mut self) {
        let device = self.generation.device().clone();
        // SAFETY: the handles are used while the generation, which `device` holds, lives: its
        // `VkDevice`, on which every object here was made, whether or not it still serves.
        let vk_device = unsafe { device.raw() }.device;
        let fence = self.fence;
        if self.pending.is_some() {
            let nanos = u64::try_from(LAST_WAIT.as_nanos()).unwrap_or(u64::MAX);
            // SAFETY: the fence is this copier's.
            let done = device
                .with_loader_word(|| unsafe { vk_device.wait_for_fences(&[fence], true, nanos) });
            if done.is_err() {
                // A copy the GPU has not finished uses the objects: they are left.
                log::warn!(
                    "a plane copier's last copy had not finished on the GPU at its end: its \
                     objects are left"
                );
                return;
            }
        }
        let imports = std::mem::take(&mut self.imports);
        self.destroy(vk_device, imports);
        let (pool, timestamps) = (self.pool, self.timestamps);
        device.with_loader_word(|| {
            // SAFETY: the objects are this copier's, and no copy in flight uses them.
            unsafe {
                if let Some((query, _)) = timestamps {
                    vk_device.destroy_query_pool(query, None);
                }
                vk_device.destroy_fence(fence, None);
                vk_device.destroy_command_pool(pool, None);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `VkDevice` of a copier's objects.
    fn made_on(copier: &PlaneCopier) -> vk::Device {
        let objects = copier.objects.as_ref().expect("the copier's objects");
        // SAFETY: only the handle is read, while the generation lives.
        unsafe { objects.generation.raw() }.device.handle()
    }

    /// A copy after the device's recovery is made with objects of its new `VkDevice`: the
    /// copier destroys its objects on the old one and makes them again on the new one, and the
    /// copy (of no plane, so no dma-buf is needed) runs on the GPU there. Under the test hook's
    /// recovery the old `VkDevice` is not lost, so a copy with the old objects would run too: the
    /// objects' device is what tells the two apart. A device that imports no dma-buf as an
    /// image makes no copier: the test says so and checks nothing.
    #[test]
    fn a_copy_after_the_devices_recovery_is_made_on_its_new_device() {
        // A device of its own, as the recovery moves every user of the device.
        let device =
            crate::gpu::device::adopt(rt::Device::system_default().expect("a runtime device"));
        let mut copier = match PlaneCopier::new(&device) {
            Ok(copier) => copier,
            Err(why) if why.starts_with("the device lists no ") => {
                eprintln!("skipped: {why}");
                return;
            }
            Err(why) => panic!("no copier: {why}"),
        };
        let wait = Duration::from_secs(5);
        assert!(copier.copy(&[], wait, Box::new(())).is_ok());
        let first = made_on(&copier);
        assert_eq!(copier.generations(), 1);

        device.raw.recover_for_test().expect("the recovery");
        assert!(copier.copy(&[], wait, Box::new(())).is_ok());
        assert_eq!(copier.generations(), 2);
        assert_ne!(
            made_on(&copier),
            first,
            "the objects are the new VkDevice's"
        );
        let objects = copier.objects.as_ref().expect("the copier's objects");
        assert!(objects.generation.serves(&copier.queue));
        assert!(objects.generation.is_of(device.raw.new_fence().device()));
        assert!(copier.copy(&[], wait, Box::new(())).is_ok());
        assert_eq!(copier.generations(), 2, "made again once");
    }
}
