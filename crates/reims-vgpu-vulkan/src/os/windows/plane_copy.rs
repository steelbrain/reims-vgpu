//! Copies by the GPU of the planes another driver wrote into a shared Direct3D 12 texture, a
//! VA-API decode surface's on Windows (the VA crate's pools there are such textures, which Mesa's
//! `vaon12` decodes into), into a runtime buffer, at the rows and offset the buffer's planes have
//! ([`PlaneCopier::copy`]): the reims-graphics device's VA-API pictures land so in their output
//! surfaces on Windows, whose planes lie 32 bytes into their pages, as CoreVideo lays them out.
//!
//! - **The source.** The texture is imported by its NT handle as one two-plane image of its
//!   format (`G8_B8R8_2PLANE_420_UNORM` for NV12, `G16_B16R16_2PLANE_420_UNORM` for P010), of the
//!   first handle type the device imports such an image from: `D3D12_RESOURCE`, the type of a
//!   Direct3D 12 resource's handle, else `OPAQUE_WIN32`, which the UHD 630's driver imports it as
//!   where it lists no `D3D12_RESOURCE` import (measured, below). The image is in optimal tiling,
//!   and the memory is the image's own (`VkMemoryDedicatedAllocateInfo`). An import is kept by
//!   its key for later copies of the same texture, until its owner says it is gone
//!   ([`SharedPlanes::alive`]) or the copier goes: a pool's textures are lent again picture after
//!   picture, so each is imported once. The import holds a reference of its own to the texture:
//!   Vulkan takes no ownership of the handle, which stays the caller's. The image is taken from
//!   outside the device (`VK_QUEUE_FAMILY_EXTERNAL`, in `GENERAL`) for the copy and handed back
//!   after it.
//! - **The destination** is a runtime buffer whose bytes are its memory: one in the device's own
//!   memory, or over host pages the device imported in place. A buffer whose pages are a copy the
//!   runtime keeps (`copied-host-pages`) is refused: a write into its `VkBuffer` would never
//!   reach the pages.
//! - **The copy** is one command buffer of the copier's own on a runtime queue of its own, whose
//!   `VkQueue` it borrows for the submission: `vkCmdCopyImageToBuffer` of each plane's aspect
//!   into the buffer at the plane's first byte, its rows `bytes_per_row` apart, then a barrier
//!   that makes the writes visible to every later command and the host. The caller waits for the
//!   driver's write of the texture before the copy (a decode's sync), and the copier waits for
//!   the copy on its fence before it returns, as on Linux; its GPU time is measured with two
//!   timestamps where the queue's family writes them.
//! - **The wait, a failed copy, refusals and the device's recoveries** are as on Linux
//!   (`os/linux/plane_copy.rs`): no bound of its own but a look every [`PARK_LOOK`] whether the
//!   copier's generation was parked; a copy whose fence answered an error is kept, with the
//!   caller's hold of the source, until its fence signals, and the copier refuses copies
//!   meanwhile; [`CopyError::Refused`] is answered before anything is submitted; objects of a
//!   `VkDevice` a recovery replaced are destroyed and made again on the new one.
//!
//! **Measured** with a libva, Direct3D 12 and Vulkan program written from scratch (2026-11-16,
//! Intel UHD 630, driver 31.0.101.2145, Mesa 26.2.7's `vaon12`): the driver lists
//! `VK_KHR_external_memory_win32`, and imports an NV12 and a P016 two-plane image in optimal
//! tiling from `OPAQUE_WIN32` and `D3D11_TEXTURE` handles, none from `D3D12_RESOURCE`. Shared
//! NV12 and P010 textures that `vaon12` decoded H.264 and HEVC streams into, imported as
//! `OPAQUE_WIN32` and copied by plane into a buffer over a pagefile-backed section's view, the
//! planes 32 bytes into their pages, gave every picture of nine streams MD5-equal to ffmpeg's
//! software decode, from 1280x720 to 3840x2160, and changed no byte past the planes; a 4K NV12
//! picture's copy and wait took 4.3 ms, a 4K P010 picture's 10.6 ms.

use std::any::Any;
use std::os::windows::io::{AsRawHandle as _, BorrowedHandle};
use std::sync::Weak;
use std::time::Duration;

use ash::vk;
use metal2vulkan_runtime as rt;

use crate::core::platform::Handle as _;
use crate::generation::Generation;
use crate::gpu::device::VulkanDevice;
use crate::gpu::resource::VulkanBuffer;

/// The most imports a copier keeps: the textures of a few pools of a decoder's.
const MOST_IMPORTS: usize = 64;

/// How long the copier's end waits for a copy kept after its fence answered an error.
const LAST_WAIT: Duration = Duration::from_secs(1);

/// How long each wait of a copy's for its fence runs before it looks whether the generation was
/// parked (the module docs): its wait has no bound but that one.
const PARK_LOOK: Duration = Duration::from_secs(1);

/// The handle types a texture is imported as, in the order tried (the module docs).
const HANDLE_TYPES: [vk::ExternalMemoryHandleTypeFlags; 2] = [
    vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE,
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32,
];

/// A handle type's name, as the Vulkan specification spells it.
fn type_name(handle_type: vk::ExternalMemoryHandleTypeFlags) -> String {
    match handle_type {
        vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE => "D3D12_RESOURCE".to_owned(),
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32 => "OPAQUE_WIN32".to_owned(),
        other => format!("{:#x}", other.as_raw()),
    }
}

/// A shared texture's format: two planes, 4:2:0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureFormat {
    /// 8-bit: a byte of luma a texel, and two of chroma.
    Nv12,
    /// 10-bit in 16: two bytes of luma a texel, and four of chroma.
    P010,
}

impl TextureFormat {
    fn vk(self) -> vk::Format {
        match self {
            TextureFormat::Nv12 => vk::Format::G8_B8R8_2PLANE_420_UNORM,
            TextureFormat::P010 => vk::Format::G16_B16R16_2PLANE_420_UNORM,
        }
    }

    /// The bytes of one texel of plane `plane`: 0 the luma, 1 the chroma.
    pub fn texel_bytes(self, plane: usize) -> u64 {
        let luma = match self {
            TextureFormat::Nv12 => 1,
            TextureFormat::P010 => 2,
        };
        if plane == 0 { luma } else { luma * 2 }
    }
}

/// A shared texture of two planes, as its NT handle opens it.
#[derive(Clone, Debug)]
pub struct SharedPlanes<'a> {
    /// What names the texture's import while it is kept: a key names one texture.
    pub key: u64,
    /// Lives while the texture's owner keeps it: an import whose owner is gone is let go.
    pub alive: Weak<()>,
    /// The texture's NT handle, which stays its owner's.
    pub handle: BorrowedHandle<'a>,
    /// Its format.
    pub format: TextureFormat,
    /// Its width in luma texels.
    pub width: u32,
    /// Its height in luma texels.
    pub height: u32,
}

/// One plane's copy: `extent` texels from `origin` of plane `plane` of the source, into `to` from
/// its byte `offset`, rows `bytes_per_row` apart.
#[derive(Clone, Debug)]
pub struct PlaneCopy<'a> {
    /// The plane copied from: 0 the luma, 1 the chroma, whose texels are a pair of samples.
    pub plane: usize,
    /// The first texel copied, its column and row in the plane.
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
    /// The copy was submitted and its fence answered an error: the GPU may write the buffers
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

/// A kept import of one texture.
struct Import {
    key: u64,
    alive: Weak<()>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// The copy that used it last, by the copier's count.
    used: u64,
}

/// A copy whose fence answered an error, kept until its fence signals, with the caller's hold of
/// its texture and the buffers it writes.
struct Pending {
    _hold: Box<dyn Any + Send>,
    _buffers: Vec<VulkanBuffer>,
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
    memory_win32: ash::khr::external_memory_win32::Device,
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Two timestamps around the copies, where the queue's family writes them, and the
    /// nanoseconds a tick.
    timestamps: Option<(vk::QueryPool, f64)>,
    imports: Vec<Import>,
    /// The handle type each format is imported as, or why none is, as the device answered.
    handle_types: Vec<(
        TextureFormat,
        Result<vk::ExternalMemoryHandleTypeFlags, String>,
    )>,
    copies: u64,
    pending: Option<Pending>,
}

impl PlaneCopier {
    /// A copier on `device`'s runtime device, with a queue of its own; `Err` with why where the
    /// device enables no `VK_KHR_external_memory_win32` or refuses the copier's objects.
    pub fn new(device: &VulkanDevice) -> Result<PlaneCopier, String> {
        let queue = (device.raw.new_command_queue()).map_err(|e| format!("no queue ({e:?})"))?;
        let objects = Objects::new(&queue)?;
        Ok(PlaneCopier {
            queue,
            objects: Some(objects),
            generations: 1,
        })
    }

    /// Copies each of `copies` from `source` (module docs) and waits for the GPU as long as it
    /// takes: `Ok` with the copies' GPU time where the queue measures it, once every plane is in
    /// its buffer. Where the fence answers an error, the device lost, or the device stopped and
    /// its `VkDevice` was parked while the copy waited, the copy is kept with `hold`, which keeps
    /// the source unwritten, until its fence signals ([`CopyError::Failed`]). Where the device
    /// recovered since the copier's objects were made, they are destroyed and made again on its
    /// new `VkDevice` first.
    pub fn copy(
        &mut self,
        source: &SharedPlanes<'_>,
        copies: &[PlaneCopy<'_>],
        hold: Box<dyn Any + Send>,
    ) -> Result<Option<Duration>, CopyError> {
        self.serve()?;
        let objects = (self.objects.as_mut())
            .ok_or_else(|| CopyError::Refused("the copier has no objects".to_owned()))?;
        objects.copy(&self.queue, source, copies, hold)
    }

    /// Makes the objects again on the generation of the device that serves where the device
    /// recovered since they were made; refused where the process is exiting.
    fn serve(&mut self) -> Result<(), CopyError> {
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
        Ok(())
    }

    /// The generations of the device the copier made its objects on: 1, and one more each time
    /// a recovery moved the device to a new `VkDevice` before a copy.
    pub fn generations(&self) -> u64 {
        self.generations
    }
}

impl Objects {
    /// The objects on the generation of `queue`'s device that serves; `Err` with why where the
    /// device enables no `VK_KHR_external_memory_win32` or refuses them.
    fn new(queue: &rt::CommandQueue) -> Result<Objects, String> {
        let generation = Generation::serving(queue)?;
        let name = ash::khr::external_memory_win32::NAME;
        if !generation.device().enables_extension(name) {
            return Err(format!("the device enables no {}", name.to_string_lossy()));
        }
        let family = generation.family();
        let (vk_device, timestamped, period, memory_win32) = {
            // SAFETY: the handles are used while `generation`, which the objects hold, lives; the
            // objects made of them are their own, destroyed at their end.
            let handles = unsafe { generation.raw() };
            let instance = handles.instance;
            // SAFETY: the physical device is the device's own.
            let families = unsafe {
                instance.get_physical_device_queue_family_properties(handles.physical_device)
            };
            let family_props = usize::try_from(family).ok().and_then(|at| families.get(at));
            let timestamped = family_props.is_some_and(|f| f.timestamp_valid_bits > 0);
            // SAFETY: the physical device is the device's own.
            let period =
                unsafe { instance.get_physical_device_properties(handles.physical_device) }
                    .limits
                    .timestamp_period;
            let memory_win32 =
                ash::khr::external_memory_win32::Device::new(instance, handles.device);
            (handles.device.clone(), timestamped, period, memory_win32)
        };
        let device = generation.device().clone();
        let mut objects = Objects {
            generation,
            memory_win32,
            pool: vk::CommandPool::null(),
            buffer: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            timestamps: None,
            imports: Vec::new(),
            handle_types: Vec::new(),
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
        source: &SharedPlanes<'_>,
        copies: &[PlaneCopy<'_>],
        hold: Box<dyn Any + Send>,
    ) -> Result<Option<Duration>, CopyError> {
        let device = self.generation.device().clone();
        // SAFETY: the handles are used while the generation, which `device` holds, lives.
        let vk_device = unsafe { device.raw() }.device;
        self.ready(vk_device)?;
        let mut targets = Vec::with_capacity(copies.len());
        for copy in copies {
            targets.push(destination(source, copy, &self.generation)?);
        }
        self.let_go_of_gone(vk_device);
        self.copies += 1;
        let image = self.import(vk_device, source)?;
        device
            .with_loader_word(|| self.record(vk_device, source, copies, image, &targets))
            .map_err(|e| CopyError::Refused(format!("its command buffer was refused ({e})")))?;
        let buffers = copies.iter().map(|copy| copy.to.clone_ref()).collect();
        self.run(queue, vk_device, hold, buffers)
    }

    /// `Ok` once no copy is pending: the last one kept after its fence answered an error is let
    /// go where its fence has signalled since; refused where it has not.
    fn ready(&mut self, vk_device: &ash::Device) -> Result<(), CopyError> {
        if self.pending.is_none() {
            return Ok(());
        }
        let device = self.generation.device().clone();
        let fence = self.fence;
        // SAFETY: the fence is this copier's.
        let done = device.with_loader_word(|| unsafe { vk_device.get_fence_status(fence) });
        match done {
            // SAFETY: the fence is this copier's, signalled, and in no submission.
            Ok(true) => {
                match device.with_loader_word(|| unsafe { vk_device.reset_fences(&[fence]) }) {
                    Ok(()) => {
                        self.pending = None;
                        Ok(())
                    }
                    Err(e) => Err(CopyError::Refused(format!(
                        "its fence's reset answered {e}"
                    ))),
                }
            }
            Ok(false) => Err(CopyError::Refused(
                "an earlier copy has not finished on the GPU".to_owned(),
            )),
            Err(e) => Err(CopyError::Refused(format!("its fence answered {e}"))),
        }
    }

    /// Submits the recorded command buffer on `queue` and waits for it (module docs), keeping
    /// `hold` and `buffers` with it where its fence answers an error; the copy's GPU time once
    /// done.
    fn run(
        &mut self,
        queue: &rt::CommandQueue,
        vk_device: &ash::Device,
        hold: Box<dyn Any + Send>,
        buffers: Vec<VulkanBuffer>,
    ) -> Result<Option<Duration>, CopyError> {
        let device = self.generation.device().clone();
        let command_buffers = [self.buffer];
        let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
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
        // No bound of its own but the generation's park (module docs).
        let look = u64::try_from(PARK_LOOK.as_nanos()).unwrap_or(u64::MAX);
        let waited = loop {
            // SAFETY: the fence is this copier's.
            let waited = device
                .with_loader_word(|| unsafe { vk_device.wait_for_fences(&[fence], true, look) });
            match waited {
                Err(vk::Result::TIMEOUT) if self.generation.is_parked() => {
                    break Err("the device stopped and its VkDevice was parked".to_owned());
                }
                Err(vk::Result::TIMEOUT) => {}
                Ok(()) => break Ok(()),
                Err(e) => break Err(format!("its fence answered {e}")),
            }
        };
        if let Err(why) = waited {
            self.pending = Some(Pending {
                _hold: hold,
                _buffers: buffers,
            });
            return Err(CopyError::Failed(why));
        }
        // SAFETY: the fence is this copier's, signalled.
        let reset = device.with_loader_word(|| unsafe { vk_device.reset_fences(&[fence]) });
        if let Err(e) = reset {
            self.pending = Some(Pending {
                _hold: hold,
                _buffers: buffers,
            });
            return Err(CopyError::Failed(format!("its fence's reset answered {e}")));
        }
        drop((hold, buffers));
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

    /// The image of `source`, imported now or kept from an earlier copy.
    fn import(
        &mut self,
        vk_device: &ash::Device,
        source: &SharedPlanes<'_>,
    ) -> Result<vk::Image, CopyError> {
        let copies = self.copies;
        if let Some(kept) = self.imports.iter_mut().find(|i| i.key == source.key) {
            kept.used = copies;
            return Ok(kept.image);
        }
        let handle_type = self
            .handle_type(source.format)
            .map_err(CopyError::Refused)?;
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
            .with_loader_word(|| self.make_import(vk_device, source, handle_type))
            .map_err(|why| CopyError::Refused(format!("the texture's import: {why}")))?;
        self.imports.push(Import {
            key: source.key,
            alive: source.alive.clone(),
            image,
            memory,
            used: copies,
        });
        Ok(image)
    }

    /// The handle type a texture of `format` is imported as (module docs), as the device first
    /// answered for it; `Err` with why where it imports none.
    fn handle_type(
        &mut self,
        format: TextureFormat,
    ) -> Result<vk::ExternalMemoryHandleTypeFlags, String> {
        if let Some((_, answer)) = self.handle_types.iter().find(|(f, _)| *f == format) {
            return answer.clone();
        }
        let mut answers = Vec::new();
        let mut chosen = None;
        for handle_type in HANDLE_TYPES {
            match self.importable(format, handle_type) {
                Ok(()) => {
                    chosen = Some(handle_type);
                    break;
                }
                Err(why) => answers.push(why),
            }
        }
        let answer = chosen.ok_or_else(|| answers.join("; "));
        self.handle_types.push((format, answer.clone()));
        answer
    }

    /// `Ok` where the device imports a handle of `handle_type` as a two-plane image of
    /// `format` that copies out; `Err` with why otherwise.
    fn importable(
        &self,
        format: TextureFormat,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<(), String> {
        // SAFETY: the handles are used while the generation lives.
        let handles = unsafe { self.generation.raw() };
        let mut external =
            vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle_type);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format.vk())
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
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
        answered.map_err(|e| {
            format!(
                "the device makes no {format:?} image from a {} handle ({e})",
                type_name(handle_type)
            )
        })?;
        if !(external_props
            .external_memory_properties
            .external_memory_features)
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        {
            return Err(format!(
                "the device imports no {} handle as a {format:?} image",
                type_name(handle_type)
            ));
        }
        Ok(())
    }

    /// An image over `source`'s texture, and its import as `handle_type` (module docs).
    fn make_import(
        &self,
        vk_device: &ash::Device,
        source: &SharedPlanes<'_>,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<(vk::Image, vk::DeviceMemory), String> {
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle_type);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(source.format.vk())
            .extent(vk::Extent3D {
                width: source.width,
                height: source.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external);
        // SAFETY: the info is valid; the image is this copier's, destroyed below on a refusal.
        let image = unsafe { vk_device.create_image(&info, None) }
            .map_err(|e| format!("vkCreateImage answered {e}"))?;
        match self.import_memory(vk_device, image, source, handle_type) {
            Ok(memory) => Ok((image, memory)),
            Err(why) => {
                // SAFETY: the image is this copier's, bound to nothing and used by nothing.
                unsafe { vk_device.destroy_image(image, None) };
                Err(why)
            }
        }
    }

    /// `source`'s texture imported for `image` by its handle, and bound to it.
    fn import_memory(
        &self,
        vk_device: &ash::Device,
        image: vk::Image,
        source: &SharedPlanes<'_>,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
    ) -> Result<vk::DeviceMemory, String> {
        // A Win32 `HANDLE`, as ash types it.
        let handle = source.handle.as_raw_handle().addr().cast_signed();
        // SAFETY: the image is this copier's.
        let needs = unsafe { vk_device.get_image_memory_requirements(image) };
        let mut types = needs.memory_type_bits;
        // An opaque handle's properties are not asked: its memory types are the image's.
        if handle_type == vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE {
            let mut props = vk::MemoryWin32HandlePropertiesKHR::default();
            // SAFETY: an NT handle of a Direct3D 12 resource, and a structure to fill.
            unsafe {
                self.memory_win32.get_memory_win32_handle_properties(
                    handle_type,
                    handle,
                    &mut props,
                )
            }
            .map_err(|e| format!("vkGetMemoryWin32HandlePropertiesKHR answered {e}"))?;
            types &= props.memory_type_bits;
        }
        if types == 0 {
            return Err(format!(
                "no memory type the image ({:#x}) and the handle take",
                needs.memory_type_bits
            ));
        }
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut import = vk::ImportMemoryWin32HandleInfoKHR::default()
            .handle_type(handle_type)
            .handle(handle);
        let info = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(types.trailing_zeros())
            .push_next(&mut import)
            .push_next(&mut dedicated);
        // SAFETY: the info is valid; the import takes a reference of its own to the texture, and
        // the handle stays its owner's (module docs).
        let memory = unsafe { vk_device.allocate_memory(&info, None) }
            .map_err(|e| format!("the texture's import answered {e}"))?;
        // SAFETY: the memory was made for the image, which is bound to nothing yet.
        if let Err(e) = unsafe { vk_device.bind_image_memory(image, memory, 0) } {
            // SAFETY: the memory is this copier's, bound to nothing.
            unsafe { vk_device.free_memory(memory, None) };
            return Err(format!("vkBindImageMemory answered {e}"));
        }
        Ok(memory)
    }

    /// Records the copies of `copies` from `image`, `source`'s, into `targets` (module docs).
    fn record(
        &self,
        vk_device: &ash::Device,
        source: &SharedPlanes<'_>,
        copies: &[PlaneCopy<'_>],
        image: vk::Image,
        targets: &[(vk::Buffer, u64)],
    ) -> Result<(), vk::Result> {
        let color = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let (external, ours) = (vk::QUEUE_FAMILY_EXTERNAL, self.generation.family());
        let acquire = [vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(color)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(external)
            .dst_queue_family_index(ours)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        let release = [vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(color)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(ours)
            .dst_queue_family_index(external)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)];
        let cb = self.buffer;
        // SAFETY: the buffer is this copier's and not pending; every handle recorded is live
        // through the copy (the image is kept, the buffers held by the caller's copies).
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
                &acquire,
            );
            if let Some((pool, _)) = self.timestamps {
                vk_device.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0);
            }
            for (copy, &(buffer, offset)) in copies.iter().zip(targets) {
                let texel = source.format.texel_bytes(copy.plane);
                let aspect = if copy.plane == 0 {
                    vk::ImageAspectFlags::PLANE_0
                } else {
                    vk::ImageAspectFlags::PLANE_1
                };
                let region = vk::BufferImageCopy {
                    buffer_offset: offset,
                    buffer_row_length: u32::try_from(copy.bytes_per_row / texel)
                        .unwrap_or(u32::MAX),
                    buffer_image_height: copy.extent[1],
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: aspect,
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
                &release,
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
/// its bytes are not its memory, the copy names no plane of `source`, or lies past the buffer,
/// the plane, or off its texels' alignment (module docs).
fn destination(
    source: &SharedPlanes<'_>,
    copy: &PlaneCopy<'_>,
    generation: &Generation,
) -> Result<(vk::Buffer, u64), CopyError> {
    let refused = |why: String| CopyError::Refused(why);
    let buffer = &copy.to.raw;
    if !generation.is_of(&buffer.device()) {
        return Err(refused(
            "the buffer was made on the device's VkDevice before its recovery".to_owned(),
        ));
    }
    if let Some(rt::HostBacking::Copied) = buffer.host_backing() {
        return Err(refused(
            "the buffer's pages are a copy the runtime keeps, which the GPU's own writes would \
             not reach"
                .to_owned(),
        ));
    }
    if copy.plane > 1 {
        return Err(refused(format!(
            "plane {} of a two-plane texture",
            copy.plane
        )));
    }
    let texel = source.format.texel_bytes(copy.plane);
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
    // The chroma plane is half the texture's size each way.
    let shift = u32::from(copy.plane == 1);
    let (plane_width, plane_height) = (source.width >> shift, source.height >> shift);
    let (origin_x, origin_y) = (u64::from(copy.origin[0]), u64::from(copy.origin[1]));
    if origin_x + u64::from(width) > u64::from(plane_width)
        || origin_y + u64::from(height) > u64::from(plane_height)
    {
        return Err(refused(format!(
            "{width}x{height} texels from ({origin_x}, {origin_y}) of a {plane_width}x\
             {plane_height} plane"
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
    /// Waits up to [`LAST_WAIT`] for a copy kept, then destroys the objects and imports on the
    /// generation they were made on; on a parked one, leaves them as they are (the module docs).
    fn drop(&mut self) {
        if self.generation.is_parked() {
            log::debug!("a plane copier's objects are left on the device's parked VkDevice");
            return;
        }
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
