//! One stream's video session, of either codec: the `VkVideoSessionKHR`
//! and its memory, its DPB pictures, its parameters object, and the command
//! buffers and bitstream buffers its decodes are submitted with, each
//! reused once the timeline has passed the work that used it. A bitstream
//! buffer is made only for a picture whose slices are copied: one read in
//! place makes none, so a stream read in place maps no host memory for its
//! slices. On an RTX 5080, about 7 of a session's new submissions, each
//! with its 2 MiB buffer, took over 1 ms, up to 16 ms (measured). What is the
//! codec's goes in with each decode ([`CodecInfo`]) and in the parameters
//! object its accelerator makes ([`SessionParameters`]).
//!
//! **DPB pictures.** The session owns one image a DPB slot, in the
//! profile's format (NV12, or P010 at 10 bits), so that a
//! reference is never an output: a guest recycles an output once it has
//! shown it, whether or not the picture is still a reference. Each decode
//! sets up a slot no reference holds ([`CopyOrder::free_slot`]), and takes
//! its picture from `UNDEFINED`. With unified layouts every DPB picture
//! stays in `GENERAL`; without, in `VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR`.
//!
//! **Outputs.** Where the device decodes into an output beside the DPB
//! picture (`DPB_AND_OUTPUT_DISTINCT`), the decode writes an output image.
//! Where the DPB picture is the decode's output (`DPB_AND_OUTPUT_COINCIDE`,
//! all the 5080 offers), one GPU copy then takes the picture from it into
//! the output. A picture that lands in a buffer is always copied: out of
//! its DPB picture where the two coincide, else out of the session's
//! scratch image, which the decode writes. A copy runs in the decode's own
//! command buffer where the decode family copies, else on the copy queue
//! behind the decode's value, signalling the next. Every output image is
//! left in `GENERAL`.
//!
//! **Order.** A later decode may write the image a copy on the copy queue
//! reads ([`CopyOrder`]). Where decodes overlap copies
//! ([`VideoDevice::overlaps_copies`]: unified layouts, so that no copy
//! changes the layout of a picture a decode reads, and copies on a queue of
//! the decoders' own), a decode waits only for the last copy out of the
//! image it writes, sets up the free slot copied out of longest ago, in a
//! session of one more slot where the device has it, and so runs while the
//! copy before it does. As the timeline's values must be signalled in
//! their order, a decode's value is then signalled by a batch of no work
//! after it on the decode queue, which waits for the copy before it too.
//! Elsewhere each decode waits for the last copy, and sets up the least
//! free slot.

use std::collections::VecDeque;
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;
use ash::vk::native::{
    StdVideoDecodeH264PictureInfo, StdVideoDecodeH264ReferenceInfo, StdVideoDecodeH265PictureInfo,
    StdVideoDecodeH265ReferenceInfo,
};

use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::import::{AnyProfile, BufferUse, ImportBuffer, SampleMemory, buffer_over};
use crate::picture::{ImageView, color_range, plane, planes};
use crate::profile::{Capabilities, VideoProfile};
use crate::surface::{BufferPlanes, bind_image_memory};
use crate::timeline::Timeline;

/// The least a bitstream buffer holds, so that most streams make one
/// buffer a submission and never grow it.
const MIN_BITSTREAM: u64 = 2 << 20;

/// The start code put before each slice: the Vulkan specification has
/// `pSliceOffsets` (H.264) and `pSliceSegmentOffsets` (H.265) locate each
/// slice in the buffer and says nothing of start codes; ffmpeg's Vulkan
/// decoders (`vulkan_h264.c`, `vulkan_hevc.c`) put a three-byte start code
/// at each offset, and the 5080 decodes every stream of the H.264 and
/// H.265 checks so.
const START_CODE: [u8; 3] = [0, 0, 1];

/// Where each slice goes in a picture's bitstream, after its start code,
/// and the bytes the decode reads: the slices' total, padded to
/// `size_alignment`.
pub(crate) fn slice_layout(
    slices: &[&[u8]],
    size_alignment: u64,
) -> Result<(Vec<u32>, u64), Error> {
    let mut offsets = Vec::with_capacity(slices.len());
    let mut at = 0u64;
    for slice in slices {
        offsets.push(
            u32::try_from(at).map_err(|_| Error::Unsupported("a picture over 4 GiB".into()))?,
        );
        at += u64::try_from(START_CODE.len() + slice.len()).unwrap_or(u64::MAX);
    }
    Ok((offsets, at.next_multiple_of(size_alignment.max(1))))
}

/// A sample whose bytes lie in device memory too
/// ([`crate::H264Decoder::decode_in_place`],
/// [`crate::H265Decoder::decode_in_place`]): where its bytes are on the
/// host, which its slices lie in.
pub(crate) struct InMemory {
    /// Its first byte.
    pub(crate) start: NonNull<u8>,
    /// Its bytes.
    pub(crate) len: usize,
    /// Where they lie in device memory.
    pub(crate) memory: SampleMemory,
}

// SAFETY: `start` is read only while the call that set it borrows the
// sample, on that call's thread.
unsafe impl Send for InMemory {}

/// A picture's slices as its decode reads them in place: a buffer of the
/// session's bound over the sample's memory, where the first slice's start
/// code lies in it, the bytes the decode reads from there, each slice's
/// offset from there, and what keeps the memory. It goes once the decode
/// is done.
struct InPlace {
    buffer: ImportBuffer,
    offset: u64,
    range: u64,
    offsets: Vec<u32>,
}

/// A decode as [`Session::decode`] records it.
pub(crate) struct Decode<'a> {
    /// The picture's slices (H.265's slice segments), each its NAL unit
    /// from the header bytes on.
    pub(crate) slices: &'a [&'a [u8]],
    /// The sample they lie in, where its bytes lie in device memory too.
    pub(crate) in_memory: Option<&'a InMemory>,
    /// The parameters object holding the picture's parameter sets.
    pub(crate) parameters: vk::VideoSessionParametersKHR,
    /// Where the picture lands.
    pub(crate) output: Target,
    /// The DPB slot the picture is set up in.
    pub(crate) setup_slot: i32,
    /// Each reference's slot, in the order of the codec's references.
    pub(crate) reference_slots: &'a [i32],
    /// The codec's std information of the picture, of its setup slot and
    /// of each reference.
    pub(crate) codec: CodecInfo<'a>,
}

/// What a decode names of its codec: the std structures Vulkan Video's
/// codec extension chains to the decode and to its slots, the references'
/// in the order of [`Decode::reference_slots`].
#[derive(Clone, Copy)]
pub(crate) enum CodecInfo<'a> {
    H264 {
        picture: &'a StdVideoDecodeH264PictureInfo,
        setup: &'a StdVideoDecodeH264ReferenceInfo,
        references: &'a [StdVideoDecodeH264ReferenceInfo],
    },
    H265 {
        picture: &'a StdVideoDecodeH265PictureInfo,
        setup: &'a StdVideoDecodeH265ReferenceInfo,
        references: &'a [StdVideoDecodeH265ReferenceInfo],
    },
}

/// A DPB slot's codec structure, chained to its slot.
enum SlotInfo<'a> {
    H264(vk::VideoDecodeH264DpbSlotInfoKHR<'a>),
    H265(vk::VideoDecodeH265DpbSlotInfoKHR<'a>),
}

impl<'a> SlotInfo<'a> {
    /// `slot` with this chained.
    fn chain<'b>(
        &'b mut self,
        slot: vk::VideoReferenceSlotInfoKHR<'b>,
    ) -> vk::VideoReferenceSlotInfoKHR<'b>
    where
        'a: 'b,
    {
        match self {
            Self::H264(info) => slot.push_next(info),
            Self::H265(info) => slot.push_next(info),
        }
    }
}

/// A decode's codec structure, chained to its decode info.
enum PictureChain<'a> {
    H264(vk::VideoDecodeH264PictureInfoKHR<'a>),
    H265(vk::VideoDecodeH265PictureInfoKHR<'a>),
}

impl<'a> PictureChain<'a> {
    /// `info` with this chained.
    fn chain<'b>(&'b mut self, info: vk::VideoDecodeInfoKHR<'b>) -> vk::VideoDecodeInfoKHR<'b>
    where
        'a: 'b,
    {
        match self {
            Self::H264(picture) => info.push_next(picture),
            Self::H265(picture) => info.push_next(picture),
        }
    }
}

impl<'a> CodecInfo<'a> {
    /// The setup slot's codec structure.
    fn setup(&self) -> SlotInfo<'a> {
        match *self {
            Self::H264 { setup, .. } => SlotInfo::H264(
                vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(setup),
            ),
            Self::H265 { setup, .. } => SlotInfo::H265(
                vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(setup),
            ),
        }
    }

    /// Each reference's codec structure.
    fn references(&self) -> Vec<SlotInfo<'a>> {
        match *self {
            Self::H264 { references, .. } => references
                .iter()
                .map(|info| {
                    SlotInfo::H264(
                        vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(info),
                    )
                })
                .collect(),
            Self::H265 { references, .. } => references
                .iter()
                .map(|info| {
                    SlotInfo::H265(
                        vk::VideoDecodeH265DpbSlotInfoKHR::default().std_reference_info(info),
                    )
                })
                .collect(),
        }
    }

    /// The references it names.
    fn reference_count(&self) -> usize {
        match self {
            Self::H264 { references, .. } => references.len(),
            Self::H265 { references, .. } => references.len(),
        }
    }

    /// The picture's codec structure, its slices at `offsets`.
    fn picture(&self, offsets: &'a [u32]) -> PictureChain<'a> {
        match *self {
            Self::H264 { picture, .. } => PictureChain::H264(
                vk::VideoDecodeH264PictureInfoKHR::default()
                    .std_picture_info(picture)
                    .slice_offsets(offsets),
            ),
            Self::H265 { picture, .. } => PictureChain::H265(
                vk::VideoDecodeH265PictureInfoKHR::default()
                    .std_picture_info(picture)
                    .slice_segment_offsets(offsets),
            ),
        }
    }
}

/// A codec's session parameters object, with what it holds, which a
/// [`Session`] keeps and destroys before itself.
pub(crate) trait SessionParameters: Send + 'static {
    /// The object.
    fn handle(&self) -> vk::VideoSessionParametersKHR;
}

/// How the slices a decoder's decodes read reached them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BitstreamCounts {
    /// Pictures whose slices were read in place, where the caller's sample
    /// lies in device memory ([`crate::H264Decoder::decode_in_place`]).
    pub in_place: u64,
    /// Pictures whose slices were copied into a bitstream buffer of the
    /// decoder's.
    pub copied: u64,
}

impl BitstreamCounts {
    /// Counts one more decode, in place or copied.
    pub(crate) fn count(&mut self, in_place: bool) {
        let counter = if in_place {
            &mut self.in_place
        } else {
            &mut self.copied
        };
        *counter = counter.saturating_add(1);
    }
}

/// What [`Session::decode`] submitted: the value after which the output
/// holds the picture, and whether its slices were read in place.
#[derive(Clone, Copy)]
pub(crate) struct Submitted {
    pub(crate) landed: u64,
    pub(crate) in_place: bool,
}

/// Where a decode's picture lands.
#[derive(Clone, Copy)]
pub(crate) enum Target {
    /// An output image, and its decode view where the decode writes it.
    Image(vk::Image, Option<vk::ImageView>),
    /// Planes in a buffer, cropped to the rectangle: x, y, width and
    /// height in luma samples.
    Buffer(BufferPlanes, (u32, u32, u32, u32)),
}

/// The copy regions of the crop `(x, y, width, height)` into `planes`:
/// the luma plane's, then the chroma plane's at half the size, of samples
/// of `sample_bytes` bytes (1 in NV12, 2 in P010).
pub(crate) fn buffer_regions(
    planes: &BufferPlanes,
    crop: (u32, u32, u32, u32),
    sample_bytes: u32,
) -> [vk::BufferImageCopy; 2] {
    let sample_bytes = sample_bytes.max(1);
    let (x, y, width, height) = crop;
    let offset = |x: u32, y: u32| vk::Offset3D {
        x: i32::try_from(x).unwrap_or(i32::MAX),
        y: i32::try_from(y).unwrap_or(i32::MAX),
        z: 0,
    };
    let [luma, chroma] = [vk::ImageAspectFlags::PLANE_0, vk::ImageAspectFlags::PLANE_1];
    [
        // The row lengths count texels: a luma sample, a chroma pair.
        vk::BufferImageCopy::default()
            .buffer_offset(planes.luma.0)
            .buffer_row_length(planes.luma.1 / sample_bytes)
            .image_subresource(plane(luma))
            .image_offset(offset(x, y))
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            }),
        vk::BufferImageCopy::default()
            .buffer_offset(planes.chroma.0)
            .buffer_row_length(planes.chroma.1 / (2 * sample_bytes))
            .image_subresource(plane(chroma))
            .image_offset(offset(x / 2, y / 2))
            .image_extent(vk::Extent3D {
                width: width.div_ceil(2),
                height: height.div_ceil(2),
                depth: 1,
            }),
    ]
}

/// What a copy reads: an image of the session's, `read`, in `layout`,
/// which it returns to where `restore` says.
#[derive(Clone, Copy)]
struct CopySource {
    image: vk::Image,
    read: CopyRead,
    layout: vk::ImageLayout,
    restore: bool,
}

/// What a copy out on the copy queue reads: a DPB slot's picture, or the
/// scratch image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CopyRead {
    Slot(i32),
    Scratch,
}

/// How a session's decodes are ordered against the copies out on the copy
/// queue, by the values of the session's timeline those copies signal
/// (the module doc). Where decodes overlap copies
/// ([`VideoDevice::overlaps_copies`]), a decode waits only for the last
/// copy out of the image it writes, sets up the free slot whose last copy
/// is the oldest, and has its value signalled after the copy before it,
/// so that the timeline's values are signalled in their order though a
/// decode may end before the copy before it. Elsewhere each decode waits
/// for the last copy, and sets up the least free slot.
#[derive(Debug)]
pub(crate) struct CopyOrder {
    overlap: bool,
    /// The value the last copy signals; 0 before one.
    last: u64,
    /// By slot: the value the last copy out of its picture signals; 0
    /// before one.
    slots: Vec<u64>,
    /// The scratch image's.
    scratch: u64,
}

impl CopyOrder {
    /// The order of a session of `slots` DPB slots, its decodes overlapping
    /// its copies where `overlap`.
    pub(crate) fn new(overlap: bool, slots: u32) -> Self {
        Self {
            overlap,
            last: 0,
            slots: vec![0; usize::try_from(slots).unwrap_or(0)],
            scratch: 0,
        }
    }

    /// The value a decode waits for before it starts, 0 for none: one that
    /// sets up `slot`, and writes the scratch image where `scratch`.
    pub(crate) fn before_decode(&self, slot: i32, scratch: bool) -> u64 {
        if !self.overlap {
            return self.last;
        }
        let slot = usize::try_from(slot)
            .ok()
            .and_then(|slot| self.slots.get(slot))
            .copied()
            // A slot past the session's is refused before it is submitted.
            .unwrap_or(self.last);
        if scratch {
            slot.max(self.scratch)
        } else {
            slot
        }
    }

    /// The value a decode's own value is signalled after, in a batch of its
    /// own: the last copy's, where decodes overlap copies and one was
    /// submitted; `None` where the decode's submission signals it.
    pub(crate) fn before_signal(&self) -> Option<u64> {
        (self.overlap && self.last > 0).then_some(self.last)
    }

    /// A copy out of `read` was submitted, signalling `landed`.
    pub(crate) fn copied(&mut self, read: CopyRead, landed: u64) {
        self.last = landed;
        match read {
            CopyRead::Scratch => self.scratch = landed,
            CopyRead::Slot(slot) => {
                if let Some(at) = usize::try_from(slot)
                    .ok()
                    .and_then(|slot| self.slots.get_mut(slot))
                {
                    *at = landed;
                }
            }
        }
    }

    /// The slot a decode sets up whose references hold `used`: the least
    /// free one, or, where decodes overlap copies, the free one whose last
    /// copy is the oldest (the least of those). `None` where every slot is
    /// held.
    pub(crate) fn free_slot(&self, used: &[i32]) -> Option<i32> {
        let free = (0i32..)
            .zip(&self.slots)
            .filter(|(slot, _)| !used.contains(slot));
        if self.overlap {
            free.min_by_key(|&(_, &copied)| copied)
                .map(|(slot, _)| slot)
        } else {
            free.map(|(slot, _)| slot).next()
        }
    }
}

/// One stream's session, its parameters object a `P`.
pub(crate) struct Session<P: SessionParameters> {
    device: Arc<VideoDevice>,
    timeline: Arc<Timeline>,
    /// What the device answers for the session's profile.
    pub(crate) caps: Capabilities<VideoProfile>,
    /// The coded extent each picture is decoded at.
    pub(crate) extent: vk::Extent2D,
    /// The extent of each image a decode writes: the coded extent, rounded
    /// up to the profile's picture access granularity.
    pub(crate) image_extent: vk::Extent2D,
    handle: vk::VideoSessionKHR,
    memory: Vec<vk::DeviceMemory>,
    /// One a slot, by its index.
    dpb: Vec<SessionImage>,
    /// What a decode writes where the device decodes into outputs and the
    /// picture lands in a buffer, made by the first such decode.
    scratch: Option<SessionImage>,
    parameters: Option<P>,
    /// Whether a decode has reset the session, which its first does.
    reset: bool,
    pool: vk::CommandPool,
    /// The copy queue's, where copies run on it rather than the decode
    /// queue.
    copy_pool: vk::CommandPool,
    /// The copies out on the copy queue that each decode waits for.
    order: CopyOrder,
    /// Submissions whose work may be in flight, oldest first.
    busy: VecDeque<Submission>,
    /// Submissions whose work is done.
    free: Vec<Submission>,
}

/// An image of the session's, a DPB picture or the scratch image,
/// destroyed as it goes.
struct SessionImage {
    device: Arc<VideoDevice>,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: Option<ImageView>,
}

impl Drop for SessionImage {
    fn drop(&mut self) {
        // The view goes before its image.
        drop(self.view.take());
        // SAFETY: the image is the device's, and its session's work is
        // done; a null handle is nothing.
        unsafe { self.device.device().destroy_image(self.image, None) };
        // SAFETY: its memory, now bound to nothing.
        unsafe { self.device.device().free_memory(self.memory, None) };
    }
}

impl SessionImage {
    /// An image of `extent` for `caps`'s profile, in its format, of
    /// `usage`, shared between the decode and copy families, with a view
    /// of the decode usages `view_usage`.
    fn new(
        device: &Arc<VideoDevice>,
        caps: &Capabilities<VideoProfile>,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        view_usage: vk::ImageUsageFlags,
    ) -> Result<Self, Error> {
        let families = device.dpb_families();
        let image = caps.profile.with_list(|list| {
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(caps.profile.format())
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(list);
            let info = if families.len() > 1 {
                info.sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&families)
            } else {
                info.sharing_mode(vk::SharingMode::EXCLUSIVE)
            };
            // SAFETY: `info` and its chain live through the call.
            unsafe { device.device().create_image(&info, None) }.named("vkCreateImage")
        })?;
        let mut made = Self {
            device: Arc::clone(device),
            image,
            memory: vk::DeviceMemory::null(),
            view: None,
        };
        made.memory = bind_image_memory(device, image)?;
        made.view = Some(ImageView::for_decode(
            device,
            image,
            caps.profile.format(),
            view_usage,
        )?);
        Ok(made)
    }

    /// A DPB picture of `extent` for `caps`'s profile: also the decode's
    /// output, copied out of, where the two coincide.
    fn dpb(
        device: &Arc<VideoDevice>,
        caps: &Capabilities<VideoProfile>,
        extent: vk::Extent2D,
    ) -> Result<Self, Error> {
        let dpb = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
        if caps.decodes_into_output {
            Self::new(device, caps, extent, dpb, dpb)
        } else {
            let view_usage = dpb | vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR;
            let usage = view_usage | vk::ImageUsageFlags::TRANSFER_SRC;
            Self::new(device, caps, extent, usage, view_usage)
        }
    }

    fn view(&self) -> vk::ImageView {
        self.view
            .as_ref()
            .map_or(vk::ImageView::null(), ImageView::handle)
    }
}

/// The command buffers and the bitstream buffer one decode at a time uses.
struct Submission {
    command: vk::CommandBuffer,
    /// The copy queue's, where copies run on it.
    copy_command: vk::CommandBuffer,
    /// Made by the first decode with it whose slices are not read in place,
    /// so a stream read in place makes none.
    bitstream: Option<Bitstream>,
    /// The value the last work submitted with it signals.
    value: u64,
}

impl Submission {
    /// Whether its bitstream buffer holds `size` bytes: any submission
    /// holds none.
    fn holds(&self, size: u64) -> bool {
        size == 0 || self.bitstream.as_ref().is_some_and(|b| b.size >= size)
    }
}

/// The size of the bitstream buffer a submission makes for a picture of
/// `size` bytes: none where `size` is 0, a picture read in place, and
/// otherwise a power of two of [`MIN_BITSTREAM`] at least, a multiple of
/// `alignment`, so that the buffer serves larger pictures after it.
fn bitstream_size(size: u64, alignment: u64) -> Option<u64> {
    (size != 0).then(|| {
        size.max(MIN_BITSTREAM)
            .next_power_of_two()
            .next_multiple_of(alignment.max(1))
    })
}

/// A host-visible buffer a picture's slices are written into, mapped for
/// as long as it lives.
struct Bitstream {
    device: Arc<VideoDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: NonNull<u8>,
    size: u64,
}

// SAFETY: the mapping is the buffer's memory alone, written by the one
// session that holds it, through `&mut`.
unsafe impl Send for Bitstream {}

/// Where a pipeline barrier's first scope starts: after a decode on this
/// queue, or after a semaphore wait that already holds it.
#[derive(Clone, Copy)]
enum After {
    DecodeOnThisQueue,
    SemaphoreWait,
}

impl<P: SessionParameters> Session<P> {
    /// A session for frames of `extent` whose stream holds `references`
    /// references at most, of `caps`'s profile: a DPB slot for each
    /// reference and one for the picture being decoded, and, where its
    /// decodes overlap its copies ([`VideoDevice::overlaps_copies`]) and the
    /// device has the slot, one more, so that a picture no later one
    /// references leaves a slot free beside its own while it is copied out.
    pub(crate) fn new(
        device: &Arc<VideoDevice>,
        timeline: &Arc<Timeline>,
        caps: Capabilities<VideoProfile>,
        extent: vk::Extent2D,
        references: u32,
    ) -> Result<Self, Error> {
        let mut slots = references.saturating_add(1);
        if slots > caps.max_dpb_slots || references > caps.max_active_reference_pictures {
            return Err(Error::Unsupported(format!(
                "{references} references: the device has {} DPB slots and {} active references",
                caps.max_dpb_slots, caps.max_active_reference_pictures
            )));
        }
        let overlap = device.overlaps_copies();
        if overlap && slots < caps.max_dpb_slots {
            slots += 1;
        }
        let mut session = Self {
            device: Arc::clone(device),
            timeline: Arc::clone(timeline),
            caps,
            extent,
            image_extent: granular(extent, caps.picture_access_granularity),
            handle: vk::VideoSessionKHR::null(),
            memory: Vec::new(),
            dpb: Vec::new(),
            scratch: None,
            parameters: None,
            reset: false,
            pool: vk::CommandPool::null(),
            copy_pool: vk::CommandPool::null(),
            order: CopyOrder::new(overlap, slots),
            busy: VecDeque::new(),
            free: Vec::new(),
        };
        let fp = device.video().fp();
        let vk_device = device.device().handle();
        caps.profile.with_info(|profile| {
            let info = vk::VideoSessionCreateInfoKHR::default()
                .queue_family_index(device.decode_family())
                .video_profile(profile)
                .picture_format(caps.profile.format())
                .max_coded_extent(extent)
                .reference_picture_format(caps.profile.format())
                .max_dpb_slots(slots)
                .max_active_reference_pictures(references)
                .std_header_version(&caps.std_header_version);
            // SAFETY: `info` and what it points at live through the call.
            unsafe {
                (fp.create_video_session_khr)(
                    vk_device,
                    &info,
                    std::ptr::null(),
                    &mut session.handle,
                )
            }
            .named("vkCreateVideoSessionKHR")
        })?;
        session.bind_memory()?;
        for _ in 0..slots {
            let image = SessionImage::dpb(device, &caps, session.image_extent)?;
            session.dpb.push(image);
        }
        session.pool = command_pool(device, device.decode_family())?;
        if !device.copies_on_decode_queue() {
            session.copy_pool = command_pool(device, device.copy_family())?;
        }
        Ok(session)
    }

    /// Binds memory of its own to each of the session's bindings.
    fn bind_memory(&mut self) -> Result<(), Error> {
        bind_session_memory(&self.device, self.handle, &mut self.memory)
    }

    /// The DPB slots the session has.
    pub(crate) fn slots(&self) -> u32 {
        u32::try_from(self.dpb.len()).unwrap_or(u32::MAX)
    }

    /// The slot a picture whose references hold `used` is set up in
    /// ([`CopyOrder::free_slot`]); `None` where every slot is held.
    pub(crate) fn free_slot(&self, used: &[i32]) -> Option<i32> {
        self.order.free_slot(used)
    }

    /// The parameters object a picture needs: the one the session has,
    /// where `holds` says it holds the picture's sets, else the one `make`
    /// makes for the session from the old one, in its place, the old
    /// retired once the decodes submitted with it are done.
    pub(crate) fn parameters(
        &mut self,
        holds: impl FnOnce(&P) -> bool,
        make: impl FnOnce(&Arc<VideoDevice>, vk::VideoSessionKHR, Option<&P>) -> Result<P, Error>,
    ) -> Result<vk::VideoSessionParametersKHR, Error> {
        if let Some(sets) = &self.parameters
            && holds(sets)
        {
            return Ok(sets.handle());
        }
        let new = make(&self.device, self.handle, self.parameters.as_ref())?;
        let handle = new.handle();
        if let Some(old) = self.parameters.replace(new) {
            self.timeline
                .retire(self.timeline.last_submitted(), Box::new(old));
        }
        Ok(handle)
    }

    /// Records and submits `decode`, and the copy of its picture into its
    /// output where the device decodes into DPB pictures alone. Returns
    /// the value after which the output holds the picture, and whether its
    /// slices were read in place.
    pub(crate) fn decode(&mut self, decode: &Decode<'_>) -> Result<Submitted, Error> {
        if decode.reference_slots.len() != decode.codec.reference_count() {
            return Err(Error::Unsupported(
                "references with no std information of their own".into(),
            ));
        }
        let in_place = match decode.in_memory {
            Some(sample) => self.in_place(sample, decode.slices)?,
            None => None,
        };
        let (offsets, range) = match &in_place {
            Some(place) => (place.offsets.clone(), place.range),
            None => slice_layout(decode.slices, self.caps.min_bitstream_buffer_size_alignment)?,
        };
        let mut submission = self.submission(if in_place.is_some() { 0 } else { range.max(1) })?;
        let source = in_place
            .as_ref()
            .map(|place| (place.buffer.buffer, place.offset));
        let decoded = self.record_and_submit(&mut submission, decode, &offsets, range, source);
        // Where it failed, its work was not submitted, or it signals no
        // later than the timeline's last value: done once that is.
        submission.value = match decoded {
            Ok(value) => value,
            Err(_) => self.timeline.last_submitted(),
        };
        let read_in_place = in_place.is_some();
        if let Some(place) = in_place {
            self.timeline.retire(submission.value, Box::new(place));
        }
        self.busy.push_back(submission);
        Ok(Submitted {
            landed: decoded?,
            in_place: read_in_place,
        })
    }

    /// Where `slices`, which lie in `sample`, are read in place: a buffer
    /// bound over the sample's memory from the first slice's start code,
    /// where the device takes them there (see
    /// [`crate::H264Decoder::decode_in_place`]); `None` where it does not,
    /// or a slice lies elsewhere.
    fn in_place(&self, sample: &InMemory, slices: &[&[u8]]) -> Result<Option<InPlace>, Error> {
        let import = &sample.memory.import;
        // Each slice's start code, from the sample's first byte.
        let mut starts = Vec::with_capacity(slices.len());
        let mut end = 0;
        for slice in slices {
            let at = slice
                .as_ptr()
                .addr()
                .wrapping_sub(sample.start.as_ptr().addr());
            let Some(code) = at.checked_sub(START_CODE.len()) else {
                return Ok(None);
            };
            let slice_end = at.saturating_add(slice.len());
            if at > sample.len || slice_end > sample.len {
                return Ok(None);
            }
            // SAFETY: the sample's bytes, `len` of them from `start`, are
            // the caller's slice, borrowed for the call that decodes it,
            // and `code` lies in them.
            let code_at = unsafe { sample.start.as_ptr().add(code) };
            // SAFETY: as above: `code..at` lies in them.
            let before = unsafe { std::slice::from_raw_parts(code_at, START_CODE.len()) };
            if before != START_CODE {
                return Ok(None);
            }
            starts.push(code);
            end = end.max(slice_end);
        }
        let Some(&first) = starts.iter().min() else {
            return Ok(None);
        };
        let to_u64 = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        let offset_alignment = self.caps.min_bitstream_buffer_offset_alignment.max(1);
        let range = to_u64(end - first)
            .next_multiple_of(self.caps.min_bitstream_buffer_size_alignment.max(1));
        let at = import.offset.saturating_add(to_u64(first));
        if at % offset_alignment != 0
            || to_u64(first).saturating_add(range) > to_u64(sample.len)
            || to_u64(sample.len) > import.size
        {
            return Ok(None);
        }
        let Ok(offsets) = starts
            .iter()
            .map(|&start| u32::try_from(start - first))
            .collect::<Result<Vec<u32>, _>>()
        else {
            return Ok(None);
        };
        let families = [self.device.decode_family()];
        let made_for = BufferUse {
            usage: vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR,
            flags: vk::BufferCreateFlags::empty(),
            families: &families,
            profile: Some(AnyProfile::Decode(self.caps.profile)),
        };
        let Some((buffer, base)) = buffer_over(&self.device, import, at, at + range, made_for)?
        else {
            return Ok(None);
        };
        Ok(Some(InPlace {
            buffer: ImportBuffer::new(&self.device, buffer, Arc::clone(&sample.memory.keep)),
            offset: at - base,
            range,
            offsets,
        }))
    }

    /// The layout DPB pictures are kept in.
    fn dpb_layout(&self) -> vk::ImageLayout {
        if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::VIDEO_DECODE_DPB_KHR
        }
    }

    fn dpb_picture(&self, slot: i32) -> Result<&SessionImage, Error> {
        usize::try_from(slot)
            .ok()
            .and_then(|slot| self.dpb.get(slot))
            .ok_or_else(|| Error::Unsupported(format!("DPB slot {slot} past the session's")))
    }

    /// Records and submits `decode`, its slices read from `source` (a
    /// buffer and the offset of the first slice's start code in it), or,
    /// with none, from the submission's own buffer, written here.
    fn record_and_submit(
        &mut self,
        submission: &mut Submission,
        decode: &Decode<'_>,
        offsets: &[u32],
        range: u64,
        source: Option<(vk::Buffer, u64)>,
    ) -> Result<u64, Error> {
        if let Target::Buffer(planes, (_, _, width, height)) = decode.output {
            planes.check(width, height, self.caps.profile.sample_bytes())?;
        }
        // Where the device decodes into outputs, a picture that lands in a
        // buffer, or in an image that cannot be the decode's target, is
        // decoded into the session's scratch image and copied out of it.
        if self.caps.decodes_into_output
            && (matches!(decode.output, Target::Buffer(..)) || !self.caps.outputs_are_targets)
            && self.scratch.is_none()
        {
            let usage = vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR;
            self.scratch = Some(SessionImage::new(
                &self.device,
                &self.caps,
                self.image_extent,
                usage | vk::ImageUsageFlags::TRANSFER_SRC,
                usage,
            )?);
        }
        let (src_buffer, src_offset) = match source {
            Some(source) => source,
            None => {
                let bitstream = (submission.bitstream.as_mut())
                    .ok_or_else(|| Error::Unsupported("a submission with no bitstream".into()))?;
                bitstream.write(decode.slices, offsets, range);
                (bitstream.buffer, 0)
            }
        };
        let device = self.device.device();
        let command = submission.command;
        begin(device, command)?;

        let (setup_image, setup_view) = {
            let setup = self.dpb_picture(decode.setup_slot)?;
            (setup.image, setup.view())
        };
        let dpb_layout = self.dpb_layout();
        let decode_output_layout = if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::VIDEO_DECODE_DST_KHR
        };
        // What the decode writes beside the DPB picture, if anything, and
        // what the picture is then copied out of, if anything.
        let (written, copy) = match (self.caps.decodes_into_output, decode.output) {
            (true, Target::Image(image, view)) if self.caps.outputs_are_targets => {
                let view = view.ok_or_else(|| {
                    Error::Unsupported("an output a decode writes, with no view".into())
                })?;
                (Some((image, view)), None)
            }
            (true, _) => {
                let scratch = self
                    .scratch
                    .as_ref()
                    .ok_or_else(|| Error::Unsupported("no scratch image".into()))?;
                let source = CopySource {
                    image: scratch.image,
                    read: CopyRead::Scratch,
                    layout: decode_output_layout,
                    restore: false,
                };
                (Some((scratch.image, scratch.view())), Some(source))
            }
            (false, _) => {
                let source = CopySource {
                    image: setup_image,
                    read: CopyRead::Slot(decode.setup_slot),
                    layout: dpb_layout,
                    restore: true,
                };
                (None, Some(source))
            }
        };

        let decode_access =
            vk::AccessFlags2::VIDEO_DECODE_READ_KHR | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR;
        let fresh = |image, layout| {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(decode_access)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())
        };
        // The DPB picture set up drops what it held; an image the decode
        // writes, likewise. The references were written by decodes before
        // this one, on this queue.
        let mut images = vec![fresh(setup_image, dpb_layout)];
        if let Some((image, _)) = written {
            images.push(fresh(image, decode_output_layout));
        }
        let references_barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
            .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_READ_KHR)];
        let dependency = vk::DependencyInfo::default()
            .memory_barriers(&references_barrier)
            .image_memory_barriers(&images);
        // SAFETY: recording into the session's command buffer, whose last
        // work is done, with the arrays alive through the call; so for each
        // command recorded below.
        unsafe { self.device.cmd_pipeline_barrier2(command, &dependency) };

        let resource = |view| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D { x: 0, y: 0 })
                .coded_extent(self.extent)
                .base_array_layer(0)
                .image_view_binding(view)
        };
        let mut reference_resources = Vec::with_capacity(decode.reference_slots.len());
        for &slot in decode.reference_slots {
            reference_resources.push(resource(self.dpb_picture(slot)?.view()));
        }
        let mut reference_codec = decode.codec.references();
        let setup_resource = resource(setup_view);
        let output_resource = written.map_or(setup_resource, |(_, view)| resource(view));
        let mut setup_codec = decode.codec.setup();

        // The slots bound for the scope: every reference's, and the
        // picture's DPB picture, not yet a slot's (-1), as the decode sets
        // it up.
        let mut bound: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = decode
            .reference_slots
            .iter()
            .zip(&reference_resources)
            .map(|(&slot, resource)| {
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(slot)
                    .picture_resource(resource)
            })
            .collect();
        bound.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_resource),
        );
        let begin_coding = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.handle)
            .video_session_parameters(decode.parameters)
            .reference_slots(&bound);
        let fp = self.device.video().fp();
        // SAFETY: as above.
        unsafe { (fp.cmd_begin_video_coding_khr)(command, &begin_coding) };
        if !self.reset {
            let control = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::RESET);
            // SAFETY: as above, inside the coding scope.
            unsafe { (fp.cmd_control_video_coding_khr)(command, &control) };
        }
        let references: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = decode
            .reference_slots
            .iter()
            .zip(&reference_resources)
            .zip(reference_codec.iter_mut())
            .map(|((&slot, resource), codec)| {
                codec.chain(
                    vk::VideoReferenceSlotInfoKHR::default()
                        .slot_index(slot)
                        .picture_resource(resource),
                )
            })
            .collect();
        let setup_slot = setup_codec.chain(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(decode.setup_slot)
                .picture_resource(&setup_resource),
        );
        let mut picture_codec = decode.codec.picture(offsets);
        let info = picture_codec.chain(
            vk::VideoDecodeInfoKHR::default()
                .src_buffer(src_buffer)
                .src_buffer_offset(src_offset)
                .src_buffer_range(range)
                .dst_picture_resource(output_resource)
                .setup_reference_slot(&setup_slot)
                .reference_slots(&references),
        );
        // SAFETY: as above, inside the coding scope, with every structure
        // the decode reads alive through the call.
        unsafe { (self.device.decode().fp().cmd_decode_video_khr)(command, &info) };
        let end = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: as above, closing the scope.
        unsafe { (fp.cmd_end_video_coding_khr)(command, &end) };

        let copy_here = copy.is_some() && self.device.copies_on_decode_queue();
        match (copy, decode.output) {
            (None, Target::Image(image, _)) if decode_output_layout != vk::ImageLayout::GENERAL => {
                let to_general = [vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                    .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(decode_output_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color_range())];
                // SAFETY: as above.
                unsafe {
                    self.device.cmd_pipeline_barrier2(
                        command,
                        &vk::DependencyInfo::default().image_memory_barriers(&to_general),
                    );
                }
            }
            (Some(source), output) if copy_here => {
                self.record_copy(command, &source, &output, After::DecodeOnThisQueue);
            }
            _ => {}
        }
        // SAFETY: as above.
        unsafe { device.end_command_buffer(command) }.named("vkEndCommandBuffer")?;

        // A copy on the copy queue may still read an image this decode
        // writes again ([`CopyOrder`]).
        let value = self.timeline.next();
        let writes_scratch = copy.is_some_and(|source| source.read == CopyRead::Scratch);
        let before = self.order.before_decode(decode.setup_slot, writes_scratch);
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore())
            .value(before)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let waits = if before > 0 { &waits[..] } else { &[][..] };
        submit(
            &self.device,
            &self.timeline,
            (submission.command, waits),
            (value, self.order.before_signal()),
            Queue::Decode,
        )?;
        self.reset = true;
        let Some(source) = copy.filter(|_| !copy_here) else {
            return Ok(value);
        };

        // The copy out, on the copy queue once the decode is done.
        let copy = submission.copy_command;
        begin(device, copy)?;
        self.record_copy(copy, &source, &decode.output, After::SemaphoreWait);
        // SAFETY: as above, for the copy queue's command buffer.
        unsafe { device.end_command_buffer(copy) }.named("vkEndCommandBuffer")?;
        let landed = self.timeline.next();
        let waits = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline.semaphore())
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        submit(
            &self.device,
            &self.timeline,
            (copy, &waits),
            (landed, None),
            Queue::Copy,
        )?;
        self.order.copied(source.read, landed);
        Ok(landed)
    }

    /// Records the copy of `source`'s two planes into `target`: the whole
    /// coded frame into an output image, which it takes from `UNDEFINED`
    /// and leaves in `GENERAL`, or the crop into a buffer's planes, made
    /// available to the host. The source is left as it was where it is to
    /// be restored.
    fn record_copy(
        &self,
        command: vk::CommandBuffer,
        source: &CopySource,
        target: &Target,
        after: After,
    ) {
        let device = self.device.device();
        let copy_layout = if self.device.unified_image_layouts() {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        };
        // After the decode on this queue, or after the semaphore wait,
        // which makes its writes visible to the copy queue.
        let (src_stage, src_access) = match after {
            After::DecodeOnThisQueue => (
                vk::PipelineStageFlags2::VIDEO_DECODE_KHR,
                vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR,
            ),
            After::SemaphoreWait => (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
        };
        let barrier = |image, old, new| {
            vk::ImageMemoryBarrier2::default()
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .old_layout(old)
                .new_layout(new)
                .image(image)
                .subresource_range(color_range())
        };
        let mut before = vec![
            barrier(source.image, source.layout, copy_layout)
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ),
        ];
        if let &Target::Image(output, _) = target {
            before.push(
                barrier(output, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL)
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE),
            );
        }
        // SAFETY: recording into a command buffer of the session's, with
        // the array alive through the call; so for each command below.
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command,
                &vk::DependencyInfo::default().image_memory_barriers(&before),
            );
        }
        match *target {
            Target::Image(output, _) => {
                let regions = planes(self.extent).map(|(aspect, extent)| {
                    vk::ImageCopy::default()
                        .src_subresource(plane(aspect))
                        .dst_subresource(plane(aspect))
                        .extent(extent)
                });
                // SAFETY: as above; both images hold the extent, in the
                // layouts the barrier left them in.
                unsafe {
                    device.cmd_copy_image(
                        command,
                        source.image,
                        copy_layout,
                        output,
                        vk::ImageLayout::GENERAL,
                        &regions,
                    );
                }
            }
            Target::Buffer(planes, crop) => {
                let regions = buffer_regions(&planes, crop, self.caps.profile.sample_bytes());
                // SAFETY: as above; the source holds the crop, in the layout
                // the barrier left it in, and the planes fit the buffer
                // (its holder's word) and the copy's rules
                // (`BufferPlanes::check`).
                unsafe {
                    device.cmd_copy_image_to_buffer(
                        command,
                        source.image,
                        copy_layout,
                        planes.buffer,
                        &regions,
                    );
                }
                let to_host = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::HOST | vk::PipelineStageFlags2::ALL_COMMANDS,
                    )
                    .dst_access_mask(vk::AccessFlags2::HOST_READ | vk::AccessFlags2::MEMORY_READ)];
                // SAFETY: as above.
                unsafe {
                    self.device.cmd_pipeline_barrier2(
                        command,
                        &vk::DependencyInfo::default().memory_barriers(&to_host),
                    );
                }
            }
        }
        if source.restore && copy_layout != source.layout {
            let back = [barrier(source.image, copy_layout, source.layout)
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::NONE)];
            // SAFETY: as above.
            unsafe {
                self.device.cmd_pipeline_barrier2(
                    command,
                    &vk::DependencyInfo::default().image_memory_barriers(&back),
                );
            }
        }
    }

    /// A submission whose last work is done, with a bitstream buffer of
    /// `size` bytes at least, or with none where `size` is 0 and it has
    /// none: a picture read in place needs none, and the buffer, host
    /// memory the device maps, is the costly part of a new one.
    fn submission(&mut self, size: u64) -> Result<Submission, Error> {
        let completed = self.timeline.completed();
        while self.busy.front().is_some_and(|s| s.value <= completed) {
            if let Some(done) = self.busy.pop_front() {
                self.free.push(done);
            }
        }
        if let Some(at) = self.free.iter().position(|s| s.holds(size)) {
            return Ok(self.free.swap_remove(at));
        }
        let (command, copy_command) = match self.free.pop() {
            // Its buffer is let go: too small, and its work done.
            Some(small) => (small.command, small.copy_command),
            None => {
                let command = allocate_command(&self.device, self.pool)?;
                let copy_command = if self.copy_pool == vk::CommandPool::null() {
                    vk::CommandBuffer::null()
                } else {
                    allocate_command(&self.device, self.copy_pool)?
                };
                (command, copy_command)
            }
        };
        let alignment = self.caps.min_bitstream_buffer_size_alignment;
        let bitstream = match bitstream_size(size, alignment) {
            // The command buffers go with their pools where this fails.
            Some(size) => Some(Bitstream::new(&self.device, &self.caps, size)?),
            None => None,
        };
        Ok(Submission {
            command,
            copy_command,
            bitstream,
            value: 0,
        })
    }
}

impl<P: SessionParameters> Drop for Session<P> {
    fn drop(&mut self) {
        // Every work of the session was submitted before the timeline's
        // last value.
        if let Err(error) = self.timeline.wait(
            self.timeline.last_submitted(),
            std::time::Duration::from_secs(10),
        ) {
            log::error!("a video session's work did not finish ({error}): leaking it");
            return;
        }
        self.busy.clear();
        self.free.clear();
        drop(self.parameters.take());
        self.dpb.clear();
        drop(self.scratch.take());
        let device = self.device.device();
        // SAFETY: the pools are the session's, their command buffers idle;
        // they go with them. A null pool is nothing.
        unsafe { device.destroy_command_pool(self.pool, None) };
        // SAFETY: as above.
        unsafe { device.destroy_command_pool(self.copy_pool, None) };
        if self.handle != vk::VideoSessionKHR::null() {
            // SAFETY: the session is the device's and no work uses it.
            unsafe {
                (self.device.video().fp().destroy_video_session_khr)(
                    device.handle(),
                    self.handle,
                    std::ptr::null(),
                );
            }
        }
        for &memory in &self.memory {
            // SAFETY: the session bound to it is destroyed.
            unsafe { device.free_memory(memory, None) };
        }
    }
}

/// Binds memory of its own to each of `session`'s bindings, pushing each
/// allocation to `memory`, which the caller frees once the session is
/// destroyed.
pub(crate) fn bind_session_memory(
    device: &VideoDevice,
    session: vk::VideoSessionKHR,
    memory: &mut Vec<vk::DeviceMemory>,
) -> Result<(), Error> {
    let fp = device.video().fp();
    let vk_device = device.device().handle();
    let mut count = 0;
    // SAFETY: the session is the device's; a null output asks for the
    // count alone.
    unsafe {
        (fp.get_video_session_memory_requirements_khr)(
            vk_device,
            session,
            &mut count,
            std::ptr::null_mut(),
        )
    }
    .named("vkGetVideoSessionMemoryRequirementsKHR")?;
    let mut requirements =
        vec![vk::VideoSessionMemoryRequirementsKHR::default(); usize::try_from(count).unwrap_or(0)];
    // SAFETY: as above, with an output of `count` elements.
    unsafe {
        (fp.get_video_session_memory_requirements_khr)(
            vk_device,
            session,
            &mut count,
            requirements.as_mut_ptr(),
        )
    }
    .named("vkGetVideoSessionMemoryRequirementsKHR")?;
    requirements.truncate(usize::try_from(count).unwrap_or(0));
    let mut binds = Vec::with_capacity(requirements.len());
    for requirement in &requirements {
        let allocated = device.allocate(
            &requirement.memory_requirements,
            &[
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            ],
            None,
        )?;
        memory.push(allocated);
        binds.push(
            vk::BindVideoSessionMemoryInfoKHR::default()
                .memory_bind_index(requirement.memory_bind_index)
                .memory(allocated)
                .memory_offset(0)
                .memory_size(requirement.memory_requirements.size),
        );
    }
    if binds.is_empty() {
        return Ok(());
    }
    // SAFETY: each memory was allocated for its binding, which is bound
    // to none yet; the array lives through the call.
    unsafe {
        (fp.bind_video_session_memory_khr)(
            vk_device,
            session,
            u32::try_from(binds.len()).unwrap_or(u32::MAX),
            binds.as_ptr(),
        )
    }
    .named("vkBindVideoSessionMemoryKHR")
}

/// `extent` rounded up to whole multiples of `granularity` each way.
pub(crate) fn granular(extent: vk::Extent2D, granularity: vk::Extent2D) -> vk::Extent2D {
    let up = |size: u32, step: u32| size.next_multiple_of(step.max(1));
    vk::Extent2D {
        width: up(extent.width, granularity.width),
        height: up(extent.height, granularity.height),
    }
}

/// A command pool of `family` whose command buffers are reset one by one.
pub(crate) fn command_pool(device: &VideoDevice, family: u32) -> Result<vk::CommandPool, Error> {
    let info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(family);
    // SAFETY: `info` lives through the call.
    unsafe { device.device().create_command_pool(&info, None) }.named("vkCreateCommandPool")
}

/// A primary command buffer of `pool`, which the session alone uses.
pub(crate) fn allocate_command(
    device: &VideoDevice,
    pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, Error> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the pool is the session's, used by it alone.
    let commands = unsafe { device.device().allocate_command_buffers(&info) }
        .named("vkAllocateCommandBuffers")?;
    commands.first().copied().ok_or(Error::Vulkan(
        "vkAllocateCommandBuffers",
        vk::Result::ERROR_UNKNOWN,
    ))
}

/// Begins `command` anew, its last work done.
pub(crate) fn begin(device: &ash::Device, command: vk::CommandBuffer) -> Result<(), Error> {
    // SAFETY: the command buffer is a session's, from a pool that resets
    // them one by one, and its last work is done.
    unsafe { device.reset_command_buffer(command, vk::CommandBufferResetFlags::empty()) }
        .named("vkResetCommandBuffer")?;
    let info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    // SAFETY: as above.
    unsafe { device.begin_command_buffer(command, &info) }.named("vkBeginCommandBuffer")
}

/// The queue a submission goes to.
#[derive(Clone, Copy)]
enum Queue {
    Decode,
    Copy,
}

/// Submits `command` to `queue` after `waits`, signalling the timeline's
/// `value`, which it records as submitted: in the command's own batch, or,
/// where `after` names a value of the timeline, in a batch of no work
/// after it that waits for that value too, so that `value` is signalled
/// after `after` ([`CopyOrder::before_signal`]).
fn submit(
    device: &VideoDevice,
    timeline: &Timeline,
    (command, waits): (vk::CommandBuffer, &[vk::SemaphoreSubmitInfo<'_>]),
    (value, after): (u64, Option<u64>),
    queue: Queue,
) -> Result<(), Error> {
    let commands = [vk::CommandBufferSubmitInfo::default().command_buffer(command)];
    let signal = |value| {
        [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline.semaphore())
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)]
    };
    let signals = signal(value);
    let after = after.map(signal);
    let work = vk::SubmitInfo2::default()
        .command_buffer_infos(&commands)
        .wait_semaphore_infos(waits);
    let (one, two);
    let submits: &[vk::SubmitInfo2<'_>] = match &after {
        None => {
            one = [work.signal_semaphore_infos(&signals)];
            &one
        }
        Some(after) => {
            two = [
                work,
                vk::SubmitInfo2::default()
                    .wait_semaphore_infos(after)
                    .signal_semaphore_infos(&signals),
            ];
            &two
        }
    };
    match queue {
        Queue::Decode => device.submit_decode(submits)?,
        Queue::Copy => device.submit_copy(submits, vk::Fence::null())?,
    }
    timeline.submitted(value);
    Ok(())
}

impl Bitstream {
    /// A buffer of `size` bytes for `caps`'s profile, mapped.
    fn new(
        device: &Arc<VideoDevice>,
        caps: &Capabilities<VideoProfile>,
        size: u64,
    ) -> Result<Self, Error> {
        let vk_device = device.device();
        let buffer = caps.profile.with_list(|list| {
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(list);
            // SAFETY: `info` and its chain live through the call.
            unsafe { vk_device.create_buffer(&info, None) }.named("vkCreateBuffer")
        })?;
        let mut made = Self {
            device: Arc::clone(device),
            buffer,
            memory: vk::DeviceMemory::null(),
            mapped: NonNull::dangling(),
            size,
        };
        // SAFETY: the buffer is the device's.
        let requirements = unsafe { vk_device.get_buffer_memory_requirements(buffer) };
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        made.memory = device.allocate(&requirements, &[host], None)?;
        // SAFETY: the memory is unbound and of the buffer's requirements.
        unsafe { vk_device.bind_buffer_memory(buffer, made.memory, 0) }
            .named("vkBindBufferMemory")?;
        // SAFETY: the memory is host-visible and unmapped.
        let mapped = unsafe {
            vk_device.map_memory(made.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .named("vkMapMemory")?;
        made.mapped = NonNull::new(mapped.cast::<u8>()).ok_or(Error::Vulkan(
            "vkMapMemory",
            vk::Result::ERROR_MEMORY_MAP_FAILED,
        ))?;
        Ok(made)
    }

    /// Writes each slice after its start code at its offset, and zeros to
    /// `range`.
    fn write(&mut self, slices: &[&[u8]], offsets: &[u32], range: u64) {
        let len = usize::try_from(range.min(self.size)).unwrap_or(0);
        // SAFETY: the mapping is `size` bytes of host-coherent memory that
        // no GPU work reads now (its last decode is done), and `len` is no
        // more than that.
        let out = unsafe { std::slice::from_raw_parts_mut(self.mapped.as_ptr(), len) };
        out.fill(0);
        for (slice, &offset) in slices.iter().zip(offsets) {
            let at = usize::try_from(offset).unwrap_or(usize::MAX);
            let end = at.saturating_add(START_CODE.len() + slice.len());
            if let Some(to) = out.get_mut(at..end) {
                let (code, data) = to.split_at_mut(START_CODE.len());
                code.copy_from_slice(&START_CODE);
                data.copy_from_slice(slice);
            }
        }
    }
}

impl Drop for Bitstream {
    fn drop(&mut self) {
        let vk_device = self.device.device();
        // SAFETY: the buffer is the device's, and no work uses it: its
        // holder lets it go once its last decode is done. A null handle is
        // nothing.
        unsafe { vk_device.destroy_buffer(self.buffer, None) };
        // SAFETY: its memory, unmapped as it is freed.
        unsafe { vk_device.free_memory(self.memory, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes(luma: (u64, u32), chroma: (u64, u32)) -> BufferPlanes {
        BufferPlanes {
            buffer: vk::Buffer::null(),
            luma,
            chroma,
            size: 2_000_000,
        }
    }

    #[test]
    fn buffer_planes_take_what_a_transfer_queue_copies() {
        let check = |luma, chroma, width| planes(luma, chroma).check(width, 720, 1);
        assert!(check((256, 1344), (1_000_064, 1344), 1280).is_ok());
        // Offsets a multiple of 4, pitches at least the width, the chroma
        // pitch even.
        assert!(check((2, 1344), (1_000_064, 1344), 1280).is_err());
        assert!(check((256, 1344), (1_000_066, 1344), 1280).is_err());
        assert!(check((256, 1279), (1_000_064, 1344), 1280).is_err());
        assert!(check((256, 1281), (1_000_064, 1281), 1281).is_err());
        assert!(check((256, 1281), (1_000_064, 1282), 1281).is_ok());
    }

    #[test]
    fn p010_planes_take_two_bytes_a_sample() {
        let check = |luma, chroma, width| planes(luma, chroma).check(width, 360, 2);
        // 640 samples are 1280 bytes a row in either plane.
        assert!(check((0, 1280), (500_000, 1280), 640).is_ok());
        assert!(check((0, 1278), (500_000, 1280), 640).is_err());
        assert!(check((0, 1280), (500_000, 1278), 640).is_err());
        // A pitch of whole samples, and of whole chroma pairs.
        assert!(check((0, 1282), (500_000, 1284), 640).is_ok());
        assert!(check((0, 1281), (500_000, 1284), 640).is_err());
        assert!(check((0, 1282), (500_000, 1282), 640).is_err());
        // An odd width rounds its chroma up to a whole pair.
        assert!(check((0, 1282), (500_000, 1284), 641).is_ok());
        assert!(check((0, 1282), (500_000, 1282), 641).is_err());
    }

    #[test]
    fn buffer_planes_hold_the_crop_within_the_buffer() {
        // The chroma plane's 360 rows of 1280 bytes end 1_000_064 + 1344 *
        // 359 + 1280 = 1_483_840 bytes in; the luma plane's 720 end before
        // its start.
        let fits = planes((0, 1344), (1_000_064, 1344));
        assert!(fits.check(1280, 720, 1).is_ok());
        let short = BufferPlanes {
            size: 1_483_839,
            ..fits
        };
        assert!(short.check(1280, 720, 1).is_err());
        let exact = BufferPlanes {
            size: 1_483_840,
            ..fits
        };
        assert!(exact.check(1280, 720, 1).is_ok());
        // The luma plane's rows run into the end of a buffer as small.
        assert!(planes((0, 1344), (0, 1344)).check(1280, 1500, 1).is_err());
        // At two bytes a sample, 640 samples fill those 1280 bytes a row.
        assert!(exact.check(640, 720, 2).is_ok());
        assert!(exact.check(641, 720, 2).is_err());
    }

    #[test]
    fn a_crop_lands_at_each_planes_origin() {
        let [luma, chroma] =
            buffer_regions(&planes((256, 1344), (968_064, 1344)), (8, 4, 1280, 720), 1);
        assert_eq!(
            (
                luma.buffer_offset,
                luma.buffer_row_length,
                luma.buffer_image_height
            ),
            (256, 1344, 0)
        );
        assert_eq!((luma.image_offset.x, luma.image_offset.y), (8, 4));
        assert_eq!(
            (luma.image_extent.width, luma.image_extent.height),
            (1280, 720)
        );
        assert_eq!(
            luma.image_subresource.aspect_mask.as_raw(),
            vk::ImageAspectFlags::PLANE_0.as_raw()
        );
        // The chroma plane counts sample pairs.
        assert_eq!(
            (chroma.buffer_offset, chroma.buffer_row_length),
            (968_064, 672)
        );
        assert_eq!((chroma.image_offset.x, chroma.image_offset.y), (4, 2));
        assert_eq!(
            (chroma.image_extent.width, chroma.image_extent.height),
            (640, 360)
        );
        assert_eq!(
            chroma.image_subresource.aspect_mask.as_raw(),
            vk::ImageAspectFlags::PLANE_1.as_raw()
        );
    }

    #[test]
    fn p010_row_lengths_count_samples_and_pairs() {
        let [luma, chroma] =
            buffer_regions(&planes((0, 2688), (1_935_360, 2688)), (0, 0, 1280, 720), 2);
        // 2688 bytes a row: 1344 luma samples, 672 chroma pairs of 4 bytes.
        assert_eq!(luma.buffer_row_length, 1344);
        assert_eq!(chroma.buffer_row_length, 672);
        assert_eq!(
            (chroma.image_extent.width, chroma.image_extent.height),
            (640, 360)
        );
    }

    #[test]
    fn images_round_up_to_the_access_granularity() {
        let rounded = |(width, height), (x, y)| {
            let out = granular(
                vk::Extent2D { width, height },
                vk::Extent2D {
                    width: x,
                    height: y,
                },
            );
            (out.width, out.height)
        };
        assert_eq!(rounded((1368, 768), (16, 16)), (1376, 768));
        assert_eq!(rounded((1920, 1080), (16, 16)), (1920, 1088));
        assert_eq!(rounded((1368, 766), (1, 1)), (1368, 766));
        // A granularity of 0 is taken as 1.
        assert_eq!(rounded((8, 8), (0, 0)), (8, 8));
    }

    #[test]
    fn each_slice_offset_is_its_start_code() {
        let slices: [&[u8]; 3] = [&[0x65, 1, 2], &[0x41; 10], &[0x01]];
        let (offsets, range) = slice_layout(&slices, 256).unwrap();
        assert_eq!(offsets, [0, 6, 19]);
        assert_eq!(range, 256);
        let (_, range) = slice_layout(&slices, 1).unwrap();
        assert_eq!(range, 23);
        // An alignment of 0 is taken as 1.
        assert_eq!(slice_layout(&slices, 0).unwrap().1, 23);
    }

    #[test]
    fn a_picture_read_in_place_makes_no_bitstream_buffer() {
        assert_eq!(bitstream_size(0, 256), None);
        let read_in_place = Submission {
            command: vk::CommandBuffer::null(),
            copy_command: vk::CommandBuffer::null(),
            bitstream: None,
            value: 0,
        };
        // It serves every picture read in place, and no copied one.
        assert!(read_in_place.holds(0));
        assert!(!read_in_place.holds(1));
        // A copied picture's buffer is a power of two of 2 MiB at least.
        assert_eq!(bitstream_size(1, 256), Some(MIN_BITSTREAM));
        assert_eq!(bitstream_size(MIN_BITSTREAM, 256), Some(MIN_BITSTREAM));
        assert_eq!(
            bitstream_size(MIN_BITSTREAM + 1, 256),
            Some(2 * MIN_BITSTREAM)
        );
        assert_eq!(bitstream_size(1, 3 << 20), Some(3 << 20));
    }

    #[test]
    fn a_picture_takes_the_least_slot_no_reference_holds_where_decodes_wait_for_copies() {
        assert_eq!(CopyOrder::new(false, 17).free_slot(&[]), Some(0));
        assert_eq!(CopyOrder::new(false, 17).free_slot(&[0, 1, 3]), Some(2));
        assert_eq!(CopyOrder::new(false, 2).free_slot(&[0, 1]), None);
        // Whatever the copies: each decode waits for the last.
        let mut order = CopyOrder::new(false, 4);
        order.copied(CopyRead::Slot(0), 2);
        order.copied(CopyRead::Slot(1), 4);
        assert_eq!(order.free_slot(&[1]), Some(0));
        assert_eq!(
            (order.before_decode(2, false), order.before_signal()),
            (4, None)
        );
    }

    #[test]
    fn where_decodes_overlap_copies_a_decode_waits_for_what_reads_its_images() {
        let mut order = CopyOrder::new(true, 4);
        assert_eq!(
            (order.before_decode(0, true), order.before_signal()),
            (0, None)
        );
        order.copied(CopyRead::Slot(0), 2);
        order.copied(CopyRead::Slot(1), 4);
        order.copied(CopyRead::Scratch, 6);
        assert_eq!(order.before_decode(0, false), 2);
        assert_eq!(order.before_decode(2, false), 0);
        assert_eq!(order.before_decode(2, true), 6);
        assert_eq!(order.before_signal(), Some(6));
        // The free slot copied out of longest ago, the least of those.
        assert_eq!(order.free_slot(&[]), Some(2));
        assert_eq!(order.free_slot(&[2, 3]), Some(0));
        assert_eq!(order.free_slot(&[0, 2, 3]), Some(1));
        assert_eq!(order.free_slot(&[0, 1, 2, 3]), None);
    }

    /// Work on a fake GPU's queue.
    #[derive(Clone, Copy)]
    enum Work {
        Decode(i32),
        Copy(i32),
        Nothing,
    }

    /// A batch on a fake GPU's queue: its work and how many ticks it
    /// takes, the timeline value it waits for (0: none) and the one it
    /// signals.
    #[derive(Clone, Copy)]
    struct Batch {
        work: Work,
        ticks: u64,
        wait: u64,
        signal: Option<u64>,
    }

    /// What a stream's decodes did on a fake GPU.
    #[derive(Debug, Default, PartialEq)]
    struct Run {
        /// Ticks the decode queue sat between one decode's end and the
        /// next one's start.
        idle: u64,
        /// Signals of a value the timeline was already past, which Vulkan
        /// forbids.
        out_of_order: usize,
        /// Decodes that started on a slot a copy still read.
        overwritten: usize,
    }

    /// Twelve pictures, every third a reference that the next two
    /// references keep, each set up as `CopyOrder` chooses on a session
    /// of two references' slots and one more (and the overlapping
    /// session's extra slot), submitted as `record_and_submit` submits
    /// them, a decode's value signalled after the copy before it only
    /// where `in_order`, run on a fake GPU: a decode queue and a copy
    /// queue that each start their next batch once the batch before it
    /// ended and the timeline is at its wait. A decode takes 10 ticks and
    /// picture `i`'s copy `copy_ticks(i)`.
    fn run(overlap: bool, in_order: bool, copy_ticks: impl Fn(usize) -> u64) -> Run {
        let mut order = CopyOrder::new(overlap, 3 + u32::from(overlap));
        let mut queues = [VecDeque::new(), VecDeque::new()];
        let mut next = 1;
        let mut held: Vec<i32> = Vec::new();
        for i in 0..12 {
            let slot = order.free_slot(&held).unwrap();
            let value = next;
            let after = order.before_signal().filter(|_| in_order);
            queues[0].push_back(Batch {
                work: Work::Decode(slot),
                ticks: 10,
                wait: order.before_decode(slot, false),
                signal: if after.is_some() { None } else { Some(value) },
            });
            if let Some(after) = after {
                queues[0].push_back(Batch {
                    work: Work::Nothing,
                    ticks: 0,
                    wait: after,
                    signal: Some(value),
                });
            }
            let landed = value + 1;
            next = landed + 1;
            queues[1].push_back(Batch {
                work: Work::Copy(slot),
                ticks: copy_ticks(i),
                wait: value,
                signal: Some(landed),
            });
            order.copied(CopyRead::Slot(slot), landed);
            if i % 3 == 0 {
                held.push(slot);
                if held.len() > 2 {
                    held.remove(0);
                }
            }
        }

        let mut run = Run::default();
        let mut timeline = 0;
        let signal = |timeline: &mut u64, run: &mut Run, value: Option<u64>| {
            if let Some(value) = value {
                if value <= *timeline {
                    run.out_of_order += 1;
                } else {
                    *timeline = value;
                }
            }
        };
        let mut busy: [Option<(Batch, u64)>; 2] = [None, None];
        let mut reading: Vec<i32> = Vec::new();
        let mut decoded_at: Option<u64> = None;
        for tick in 0..10_000 {
            for running in &mut busy {
                if let Some((batch, end)) = *running
                    && end == tick
                {
                    signal(&mut timeline, &mut run, batch.signal);
                    match batch.work {
                        Work::Decode(_) => decoded_at = Some(tick),
                        Work::Copy(slot) => reading.retain(|&s| s != slot),
                        Work::Nothing => {}
                    }
                    *running = None;
                }
            }
            // A batch of no work ends as it starts.
            let mut started = true;
            while started {
                started = false;
                for (queue, running) in busy.iter_mut().enumerate() {
                    let Some(&batch) = queues[queue].front() else {
                        continue;
                    };
                    if running.is_some() || batch.wait > timeline {
                        continue;
                    }
                    queues[queue].pop_front();
                    started = true;
                    match batch.work {
                        Work::Decode(slot) => {
                            run.overwritten += usize::from(reading.contains(&slot));
                            run.idle += decoded_at.map_or(0, |at| tick - at);
                        }
                        Work::Copy(slot) => reading.push(slot),
                        Work::Nothing => {}
                    }
                    if batch.ticks == 0 {
                        signal(&mut timeline, &mut run, batch.signal);
                    } else {
                        *running = Some((batch, tick + batch.ticks));
                    }
                }
            }
            if busy.iter().all(Option::is_none) && queues.iter().all(VecDeque::is_empty) {
                return run;
            }
        }
        panic!("the fake GPU's queues hung");
    }

    #[test]
    fn where_decodes_overlap_copies_the_decode_queue_never_waits_for_a_copy() {
        // Each decode waits for the copy before it: the copy's 3 ticks
        // between each two decodes.
        let waiting = Run {
            idle: 3 * 11,
            ..Run::default()
        };
        assert_eq!(run(false, true, |_| 3), waiting);
        // Overlapping, none: no decode sets up a slot a copy reads.
        assert_eq!(run(true, true, |_| 3), Run::default());
    }

    #[test]
    fn where_decodes_overlap_copies_the_timeline_is_signalled_in_order() {
        // A copy slower than the next decode (picture 4's): the decode
        // after it waits for its value to be signalled after the copy's,
        // and none sets up a slot the copy still reads.
        let slow = |i| if i == 4 { 25 } else { 3 };
        let overlapped = run(true, true, slow);
        assert_eq!((overlapped.out_of_order, overlapped.overwritten), (0, 0));
        assert!(overlapped.idle > 0);
        // Had the decode signalled its own value, the timeline would have
        // been signalled out of order.
        assert!(run(true, false, slow).out_of_order > 0);
    }
}
