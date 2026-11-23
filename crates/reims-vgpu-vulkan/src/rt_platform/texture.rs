//! The rt platform's textures ([`RtTexture`]): rt's textures, made by the device's
//! `newTextureWithDescriptor:` ([`make`]) and by a texture's views, each an optimal image in
//! device-local memory (or a view of one) that the guest's CPU reaches only through
//! `replaceRegion:`; and textures over a buffer's bytes, made by the buffers' piece
//! (`super::buffer`) through [`make_over`], of the buffer's storage mode, a tiled one of more
//! levels among them, an image of the platform's own that takes the bytes at its first use
//! (`super::tiled`). A texture buffer the device makes is rt's over a buffer of the platform's
//! own, which it holds ([`make_texture_buffer`]): rt makes texture buffers only over buffers.
//! Textures in heaps and over IOSurfaces are their pieces'.
//!
//! **Images for views.** A shape's image of an IOSurface plane (`reims_vgpu`'s
//! `ApiPlatform::shape_image`) is rt's texture for views (the device's `new_texture_for_views`):
//! made as a texture of its descriptor, of any usage, with no index of its own, so it takes none
//! of the platform's own indices. The guest's textures over the plane are its views, each at the
//! guest's index (`newTextureViewWithPixelFormat:resourceIndex:`). Copies, uploads and readbacks
//! reach the image as any texture. Its `resourceIndex`, `uniqueIdentifier` and the index in its
//! `gpuResourceID` are 0, which a shader reads as Metal's nil texture, and where its usage
//! reaches shaders, a bind of it to a function's texture slot is rt's refusal
//! (`Refusal::TextureWithoutIndex`), which binds nothing, counted (`rt-refused-ours`): no slot
//! reaches it but through a view. A recording holds it, placed or not, as the device's
//! `newTextureWithDescriptor:` of a descriptor whose `m2v.textureDescriptor` record marks it
//! outside the guest's texture indices (`crate::recorder::outside_descriptor`), so the views,
//! copies and commands that name it name its stream id. A surface's plane (`super::surface`, `super::d3d11_surface`), which the guest's
//! textures over the plane are views of in the same way, is recorded so too (`surface_plane`).
//!
//! **Placed images for views.** A shape's image may be placed instead
//! (`ApiPlatform::placed_shape_image`, [`RtObj::new_placed_texture_for_views`]): rt's texture for
//! views bound at the start of a Private block of its own (rt's `take_block` and
//! `new_heap_texture_for_views`), the block `max(size, align)` bytes of the image's
//! `heap_texture_size_and_align`, so its base, which rt aligns to the block's length rounded up
//! to a power of two, is aligned for the image. rt clears no placed image: its bytes are the
//! block's as they are, which the caller fills before anything reads them. The texture holds the
//! block, and the block goes back to rt as the image does. Counted with the blocks' bytes
//! ([`PLACED_IMAGES_CODE`], [`PLACED_IMAGE_BYTES_CODE`]); one rt places none of is counted by its
//! cause ([`UNPLACED_OWN_CODE`], [`UNPLACED_REFUSED_CODE`], logged once) and made as before.
//!
//! **Indices.** rt places a texture that shaders reach (shader-read or shader-write usage,
//! `Unknown`, which Metal takes as both, or render-target usage of a pixel format with no depth or
//! stencil aspect, a colour target, which a fragment function reads through its attachment;
//! [`reaches`]) at an index of a live texture grant, and no other. A texture the guest places (the
//! descriptor's `resourceIndex`, a view's `resourceIndex:`) takes the guest's index, from the
//! guest's reservation (`reserveResourceIndicesForResourceType:`). One it does not place takes one
//! of the platform's own ([`OwnIndices`]): the highest index of the class that no reservation holds
//! and no texture of the platform's took yet, granted as it is taken, and once those are gone the
//! one given back longest ago whose wait is over. So an index a released texture held is taken
//! again as late as can be, and never while work reads it: rt leaves to the program that no work
//! in flight reads a released texture through its index, and Metal frees a texture's ID only once
//! its command buffers completed. An index given back waits for every command buffer its device
//! had committed then, on any of its queues, to complete ([`OwnIndices::give_back`]), by commit
//! order alone: work reads a texture through its index where its encoders bind it to a shader's
//! slot or render it as a colour attachment, and through an argument buffer that holds its ID with
//! or without a `useResource:` naming it, so what the work names does not bound what it reads. A
//! command buffer not yet committed at the release holds nothing: whatever it names, its caller
//! keeps until its commit (the decoder's views close only after their command buffer's commit).
//! Where only waiting indices are left, a take gets none, and the texture is refused as at the
//! class's last. A texture no shader reaches has no index. Its `gpuResourceID` is rt's,
//! the index and the translator's word of its format and swizzle, and its `resourceIndex` and
//! `uniqueIdentifier` the index: 0 for one with no index, where Metal numbers every texture from
//! its lowest free ID. Nothing of core or the decoder reads them but the check of the guest's
//! reservation.
//!
//! **Answers.** A texture's descriptor values are rt's descriptor of it. Its usage is Metal's
//! answer (measured on the M3 Max, `crate::gpu::desc`'s docs): the usage set without 0x8, which
//! Metal's texture drops (a texture of 0x8 alone is rt's of `ShaderRead`, which answers 0, and
//! Metal asserts at a swizzled view of it, refused as Metal's refusal), `Unknown` as
//! `ShaderRead | ShaderWrite | RenderTarget | PixelFormatView`, and a view of a swizzle other than
//! the default keeps `ShaderRead`, `PixelFormatView`, 0x4000 and 0x10000 of its texture's. Its
//! resource options are the descriptor's (of a texture over a buffer, with the buffer's storage
//! mode) with Metal's default hazard tracking resolved to tracked, as the Vulkan platform's
//! runtime answers them.
//!
//! **Refusals.** A texture rt refuses is nil, counted under rt's kind
//! ([`super::unbuilt::rt_refused_texture_create`]), a texture of a format rt makes none of under
//! the format's own code; a value rt's descriptor has no value of is Metal's refusal
//! (`rt-refused-metal`), where Metal aborts. A `replaceRegion:` rt refuses changes nothing, counted
//! likewise.
//!
//! **A copy.** `replaceRegion:`'s bytes are guest memory, which the guest writes while they are
//! read, so no shared slice may cover them, and rt takes a slice: the platform copies them into
//! memory of its own first (`crate::gpu::resource::first_bytes`), a copy Metal does not make,
//! counted per call ([`COPY_CODE`]) and by bytes ([`COPY_BYTES_CODE`]), until rt reads them through
//! their pointer.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use metal2vulkan_runtime as rt;

use crate::core::num::{ToU64 as _, ToUsize as _};
use crate::core::platform::Handle as _;
use crate::core::platform::Platform;
use crate::core::platform::host_range::HostRange;
use crate::core::platform::resource::{GpuResource, Texture};
use crate::core::platform::types::{MtlRegion, NsRange, Swizzle};
use crate::core::platform::values::TextureDesc;
use crate::core::raise::RaiseResult;
use crate::core::sync::Mutex;

use super::buffer::RtBuffer;
use super::copied::Copied;
use super::descriptor::rt_texture_descriptor;
use super::device::RtDevice;
use super::diag::{Call, Made, Shape};
use super::fault::Flights;
use super::tiled::Tiled;
use super::unbuilt::{
    RtRefusal, Unbuilt, refuse_create, refused_as, rt_refused, rt_refused_texture_create,
};
use super::{RtObj, RtPlatform};

/// The code of a `replaceRegion:` whose bytes the platform copied first (the module's docs),
/// counted per call.
pub const COPY_CODE: &str = "rt-copy-replace-region";

/// The code of the bytes [`COPY_CODE`]'s calls copied, counted by the byte.
pub const COPY_BYTES_CODE: &str = "rt-copy-replace-region-bytes";

/// The code of the placed images for views made (the module's docs, "Placed images for views").
pub const PLACED_IMAGES_CODE: &str = "rt-placed-shape-images";

/// The code of the bytes of those images' blocks, counted by the byte.
pub const PLACED_IMAGE_BYTES_CODE: &str = "rt-placed-shape-image-bytes";

/// The code of the images for views rt places none of as it takes them in an allocation of their
/// own ([`rt::Gap::HeapTexture`]).
pub const UNPLACED_OWN_CODE: &str = "rt-placed-shape-images-own";

/// The code of the images for views rt places none of for any other refusal: of the block, or of
/// the place.
pub const UNPLACED_REFUSED_CODE: &str = "rt-placed-shape-images-refused";

/// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:`.
const REPLACE: &str = "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:";

/// `MTLTextureUsageShaderRead | ShaderWrite | RenderTarget | PixelFormatView`: the usage a texture
/// made of `Unknown` usage answers (measured, `crate::gpu::desc`'s docs).
const UNKNOWN_ANSWERS: u64 = 0x17;

/// The usage bit Metal's descriptor takes and its texture drops.
const DROPPED: u64 = 0x8;

/// `MTLTextureUsageShaderWrite`.
const USAGE_SHADER_WRITE: u64 = 0x2;

/// `MTLTextureUsageShaderAtomic`.
const USAGE_SHADER_ATOMIC: u64 = 0x20;

/// The usage bits a view of a swizzle other than the default keeps of its texture's:
/// `ShaderRead`, `PixelFormatView`, 0x4000 and 0x10000 (measured, `crate::gpu::desc`'s docs).
const SWIZZLED_VIEW_KEEPS: u64 = 0x1 | 0x10 | 0x4000 | 0x10000;

/// `MTLTextureSwizzleChannels`' default: each channel from itself (red 2, green 3, blue 4,
/// alpha 5).
const DEFAULT_SWIZZLE: Swizzle = Swizzle {
    red: 2,
    green: 3,
    blue: 4,
    alpha: 5,
};

/// `MTLResourceHazardTrackingModeTracked`, in the resource options' bits 8 and 9.
const HAZARD_TRACKED: u64 = 0x200;

/// The resource options' storage mode, bits 4 to 7.
pub(super) const STORAGE_BITS: u64 = 0xf0;

/// `MTLTextureTypeTextureBuffer`, which the device makes over a buffer of its own
/// ([`make_texture_buffer`]).
pub(super) const TYPE_TEXTURE_BUFFER: u64 = 9;

/// What a texture's make asked of rt, for the process's first line of each kind of rt's
/// refusals of it (`super::unbuilt::refusal_detail`): the guest's descriptor `desc` (its pixel
/// format, size, levels, array length, sample count, texture type and usage, Metal's numbers),
/// and the storage mode of the resource options `options` the texture is made of.
pub(super) struct TextureDetail<'a> {
    /// The guest's descriptor.
    pub desc: &'a TextureDesc,
    /// The resource options the texture is made of.
    pub options: u64,
}

impl std::fmt::Display for TextureDetail<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.desc;
        write!(
            f,
            "a texture of format {} {}x{}x{}, {} levels, array length {}, {} samples, type {}, \
             usage {:#x}, storage {}",
            d.pixel_format,
            d.width,
            d.height,
            d.depth,
            d.mipmap_level_count,
            d.array_length,
            d.sample_count,
            d.texture_type,
            d.usage,
            (self.options & STORAGE_BITS) >> 4
        )
    }
}

/// What a texture view's make asked of rt, for the process's first line of each kind of rt's
/// refusals of it (`super::unbuilt::refusal_detail`), as structure alone: the view's pixel format,
/// texture type, levels and slices, Metal's numbers, and whether it swizzles; and the texture it
/// views, of rt's descriptor of it: its pixel format, type, levels, array length, sample count,
/// usage and storage mode, and how it was made (`super::diag::Made::made_as`).
struct ViewDetail<'a> {
    format: u32,
    view_type: u32,
    levels: Range<u32>,
    slices: Range<u32>,
    swizzled: bool,
    /// The texture it views.
    of: &'a RtTexture,
}

impl std::fmt::Display for ViewDetail<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let d = self.of.rt.descriptor();
        let swizzled = if self.swizzled { ", swizzled" } else { "" };
        write!(
            f,
            "a view of format {}, type {}, levels {:?}, slices {:?}{swizzled}, of a texture of \
             format {} {}x{}x{}, {} levels, array length {}, {} samples, type {}, usage {:#x}, \
             storage {}, {}",
            self.format,
            self.view_type,
            self.levels,
            self.slices,
            d.pixel_format.as_raw(),
            d.width,
            d.height,
            d.depth,
            d.mipmap_level_count,
            d.array_length,
            d.sample_count,
            d.texture_type.as_raw(),
            d.usage.as_raw(),
            super::descriptor::storage_number(d.storage_mode),
            self.of.made.made_as()
        )
    }
}

/// `MTLPurgeableStateKeepCurrent`.
const KEEP_CURRENT: u64 = 1;

/// `MTLPurgeableStateNonVolatile`, a new texture's state.
pub(super) const NON_VOLATILE: u64 = 2;

/// `MTLPurgeableStateEmpty`, the last state.
const EMPTY: u64 = 4;

/// `id<MTLTexture>` on metal2vulkan-runtime: rt's texture and what the platform answers of it
/// beyond rt's descriptor.
pub struct RtTexture {
    /// rt's texture.
    rt: rt::Texture,
    /// The device that made it.
    device: RtObj<RtDevice>,
    /// The resource options it answers: the descriptor's, hazard tracking resolved.
    options: u64,
    /// The usage it answers (the module's docs).
    usage: u64,
    /// The descriptor's `protectionOptions`, 0 where none was set.
    protection_options: u64,
    /// Its index where it is one of the platform's own ([`OwnIndices`]), which its drop gives
    /// back.
    own: Option<u32>,
    /// Its purgeable state, as Metal keeps it ([`GpuResource::set_purgeable_state`]).
    purgeable: AtomicU64,
    /// What it is, for the boot diagnostics (`super::diag`).
    made: Arc<Made>,
    /// What a tiled texture takes its bytes from, and a view's of one (`super::tiled`); unset
    /// for any other.
    tiled: OnceLock<Arc<Tiled>>,
    /// The copy of host pages it lies over, a buffer's or a heap's, and a view's of one
    /// (`super::copied`); unset for any other.
    copied: OnceLock<Arc<Copied>>,
    /// The buffer it lies over and which bytes of it, and a view's of one: what the recorder
    /// writes for it ([`crate::recorder`]); unset for any other.
    over: OnceLock<(Weak<RtBuffer>, Range<u64>)>,
    /// The buffer of the platform's own it lies over, a device-placed texture buffer's
    /// ([`make_texture_buffer`]), and a view's of one: held, so it goes with the last of them;
    /// unset for any other.
    held: OnceLock<RtObj<RtBuffer>>,
    /// The guest's object id for it, which its release's entry in the device's history names
    /// (`super::fault`).
    named: super::fault::Named,
}

impl Drop for RtTexture {
    /// Counts its release where a completed handler's run is timed (`super::handler_times`),
    /// records its range's release in its device's history where a table named it and it lies
    /// over a buffer that is still there (`super::fault`), then gives back its own index.
    fn drop(&mut self) {
        let kind = super::handler_times::texture_kind(self.made.made_as());
        let device = self.device.rt();
        super::handler_times::released(kind, || {
            super::handler_times::texture_bytes(device, self.rt.descriptor())
        });
        let id = self.named.id();
        if id != 0
            && let Some((buffer, bytes)) = self.over.get()
            && let Some(buffer) = buffer.upgrade()
            && let Some(at) = buffer.fault_address()
            && let Some(gpu_address) = at.checked_add(bytes.start)
        {
            let kind = self.named.kind(super::fault::ObjectKind::Texture);
            let storage = (self.options >> 4) & 0xf;
            let len = bytes.end.saturating_sub(bytes.start);
            (self.device.releases()).push(kind, id, storage, (gpu_address, len));
        }
        if let Some(index) = self.own {
            self.device.indices().give_back(index);
        }
    }
}

impl RtTexture {
    /// rt's texture.
    pub fn rt(&self) -> &rt::Texture {
        &self.rt
    }

    /// What it is, for the boot diagnostics (`super::diag`).
    pub(super) fn made(&self) -> &Arc<Made> {
        &self.made
    }

    /// What it takes its bytes from, where it is a tiled texture or a view of one
    /// (`super::tiled`).
    pub(super) fn tiled(&self) -> Option<&Arc<Tiled>> {
        self.tiled.get()
    }

    /// The copy of host pages it lies over, where it lies over one (`super::copied`), which a
    /// command buffer that uses it syncs.
    pub(super) fn copied(&self) -> Option<&Arc<Copied>> {
        self.copied.get()
    }

    /// Whether a shader can write it, bound to a slot: of shader-write or shader-atomic usage,
    /// which `Unknown` answers too (the module's docs, "Answers"). rt makes a texture of no
    /// such usage with no storage usage, so no shader writes it (`super::copied`'s docs).
    pub(super) fn shader_writes(&self) -> bool {
        self.usage & (USAGE_SHADER_WRITE | USAGE_SHADER_ATOMIC) != 0
    }

    /// Makes it a tiled texture, taking its bytes as `tiled` says (`super::tiled`): once, as
    /// the buffers' piece makes it.
    pub(super) fn set_tiled(&self, tiled: Arc<Tiled>) {
        let _ = self.tiled.set(tiled);
    }

    /// The buffer it lies over and which bytes of it, where it lies over one, a view's of one
    /// among them: what the recorder writes for it ([`crate::recorder`]).
    pub(crate) fn over(&self) -> Option<&(Weak<RtBuffer>, Range<u64>)> {
        self.over.get()
    }

    /// The guest's object id for it (`super::fault`).
    pub(super) fn named(&self) -> &super::fault::Named {
        &self.named
    }

    /// Notes that it lies over `bytes` of `buffer`: once, as the buffers' piece makes it.
    pub(super) fn set_over(&self, buffer: Weak<RtBuffer>, bytes: Range<u64>) {
        let _ = self.over.set((buffer, bytes));
    }

    /// The host's read-back (`reims_vgpu`'s `ApiPlatform::read_pixels`): the whole of level 0 and
    /// slice 0 into `dst`, rows `bytes_per_row` apart, through rt's `getBytes:`. Where rt refuses
    /// it (a Private texture among them), `dst` is left as it was, counted under rt's kind.
    pub fn read_pixels(&self, dst: &mut [u8], bytes_per_row: usize) {
        const SEL: &str = "getBytes:bytesPerRow:fromRegion:mipmapLevel:";
        let d = self.rt.descriptor();
        let region = rt::Region {
            origin: [0; 3],
            size: [d.width, d.height, 1],
        };
        let row = bytes_per_row.to_u64();
        if let Err(error) = self.rt.get_bytes(dst, row, 0, region, 0, 0) {
            rt_refused(SEL, &error);
        }
    }

    /// The view `sel` makes of this texture: in `pixel_format`, of `texture_type` (this texture's
    /// where `None`), over `levels` and `slices` (every one where `None`), read through `swizzle`
    /// where given, at the guest's `resource_index` (0: none placed). Nil, counted, where rt
    /// refuses it or a value is one rt has none of.
    #[expect(
        clippy::too_many_arguments,
        reason = "the widest view selector's arguments and its name"
    )]
    fn view(
        &self,
        sel: &'static str,
        pixel_format: u64,
        texture_type: Option<u64>,
        levels: Option<NsRange>,
        slices: Option<NsRange>,
        swizzle: Option<Swizzle>,
        resource_index: u64,
    ) -> Option<RtObj<RtTexture>> {
        let d = self.rt.descriptor();
        let Some(format) = crate::gpu::runtime_pixel_format(pixel_format) else {
            return refused_metal(sel, &format!("pixel format {pixel_format}"));
        };
        let raw_type = texture_type.unwrap_or(u64::from(d.texture_type.as_raw()));
        let Some(view_type) =
            (u32::try_from(raw_type).ok()).and_then(|raw| rt::TextureType::try_from(raw).ok())
        else {
            return refused_metal(sel, &format!("texture type {raw_type}"));
        };
        let levels = match levels {
            None => 0..d.mipmap_level_count,
            Some(r) => match words(r) {
                Some(levels) => levels,
                None => return refused_metal(sel, &format!("the levels {r:?}")),
            },
        };
        let slices = match slices {
            None => 0..slice_count(d),
            Some(r) => match words(r) {
                Some(slices) => slices,
                None => return refused_metal(sel, &format!("the slices {r:?}")),
            },
        };
        let channels = match swizzle {
            None => None,
            Some(s) => match crate::gpu::runtime_swizzle(s) {
                Some(channels) => Some(channels),
                None => return refused_metal(sel, &format!("the swizzle {s:?}")),
            },
        };
        if swizzle.is_some_and(|s| s != DEFAULT_SWIZZLE) && self.usage == 0 {
            // Metal asserts `_usage != MTLTextureUsageUnknown` here: a texture of 0x8 alone, which
            // rt makes as one of `ShaderRead`, answers no usage (`crate::gpu::desc`'s docs).
            return refused_metal(sel, &"a swizzled view of a texture of no usage");
        }
        let (index, own) = self
            .device
            .index_for(sel, reaches(d.usage, format), resource_index)?;
        let first = (levels.start, slices.start);
        let format_raw = u32::try_from(pixel_format).unwrap_or(u32::MAX);
        let asked = ViewDetail {
            format: format_raw,
            view_type: view_type.as_raw(),
            levels: levels.clone(),
            slices: slices.clone(),
            swizzled: channels.is_some(),
            of: self,
        };
        let made = match channels {
            None => self
                .rt
                .new_texture_view(format, view_type, levels, slices, index),
            Some(channels) => (self.rt)
                .new_texture_view_with_swizzle(format, view_type, levels, slices, channels, index),
        };
        if let Err(error) = &made {
            super::unbuilt::refusal_detail(sel, error, Some(&asked));
        }
        super::diag::plane_view(&self.made, format_raw, made.is_ok());
        let usage = match swizzle {
            Some(s) if s != DEFAULT_SWIZZLE => self.usage & SWIZZLED_VIEW_KEEPS,
            _ => self.usage,
        };
        self.device.made(sel, made, own, |texture| RtTexture {
            named: super::fault::Named::default(),
            rt: texture,
            device: self.device.clone_ref(),
            options: self.options,
            usage,
            protection_options: self.protection_options,
            own,
            purgeable: purgeable_state(),
            made: Made::view(&self.made, format_raw, first),
            tiled: self.tiled.clone(),
            copied: self.copied.clone(),
            over: self.over.clone(),
            held: match self.held.get() {
                Some(buffer) => OnceLock::from(buffer.clone_ref()),
                None => OnceLock::new(),
            },
        })
    }
}

/// `newTextureWithDescriptor:` of `desc` on `device`, for `sel`: rt's texture of the descriptor,
/// at the guest's index where it places one and shaders reach the texture, else at one of the
/// platform's own where shaders reach it (the module's docs). Nil where rt refuses it, counted.
pub fn make(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    desc: &TextureDesc,
) -> Option<RtObj<RtTexture>> {
    let options = desc.resource_options.unwrap_or(0);
    let make = |d: &rt::TextureDescriptor, index: Option<u32>| device.rt().new_texture(d, index);
    let diag = |shape: &Shape| Made::new(Call::Plain, shape);
    made_of(device, sel, desc, options, make, diag)
}

/// `newTextureWithDescriptor:` of a texture buffer (`MTLTextureTypeTextureBuffer`) of
/// `descriptor` on `device`, for `sel`: Metal's call composed of two of rt's, as rt makes a
/// texture buffer only over a buffer its consumer places (`Buffer::new_texture`) and refuses one
/// of its device (`Refusal::TextureBufferNotOverBuffer`). First a buffer of the texture's bytes
/// ([`super::buffer::of_length`]), of the descriptor's resource options, so of its storage mode:
/// its row of `width` texels, `bytesPerRow` the row rounded up to rt's texture-buffer alignment
/// of the format (`Device::minimum_texture_buffer_alignment`, the device's
/// `minimumTextureBufferAlignmentForPixelFormat:`); then the texture over it from offset 0, rows
/// `bytesPerRow` apart, as the buffer's `newTextureWithDescriptor:offset:bytesPerRow:` makes it,
/// at the index [`make`] would give it. The texture holds the buffer ([`RtTexture::held`]), which
/// goes with it. The buffer is the platform's: the guest names it by no object id, GPU address or
/// resource index, and the call stays the device's in a recording. Nil where rt refuses the
/// format as a texture buffer, the buffer or the texture, counted under `sel`.
pub(super) fn make_texture_buffer(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    descriptor: &RtObj<super::TextureFields>,
) -> Option<RtObj<RtTexture>> {
    let desc = descriptor.desc();
    let d = match rt_texture_descriptor(&desc) {
        Ok(d) => d,
        Err(why) => return refused_metal(sel, &why),
    };
    let options = desc.resource_options.unwrap_or(0);
    let alignment = match device.rt().minimum_texture_buffer_alignment(d.pixel_format) {
        Ok(alignment) => alignment.max(1),
        Err(error) => {
            let diag = |shape: &Shape| Made::new(Call::Plain, shape);
            return answered(device, sel, &desc, options, None, Err(error), diag);
        }
    };
    // rt answers an alignment only of a format not of blocks, whose first word is a texel's bytes.
    let texel = device.rt().pixel_format_info(desc.pixel_format).words[0];
    let row = u64::from(d.width).saturating_mul(texel);
    let bytes_per_row = row.next_multiple_of(alignment);
    let length = bytes_per_row.max(alignment);
    let raw = options & super::buffer::FIELDS;
    let buffer = super::buffer::of_length(device, sel, length, raw)?;
    let texture = buffer.texture(sel, descriptor, 0, bytes_per_row)?;
    let _ = texture.held.set(buffer);
    Some(texture)
}

/// A texture of `desc` over a buffer's or a heap's memory of the resource options `held`, for
/// `sel`: the texture `over` makes of rt's descriptor at the index [`make`] would give it (rt's
/// `Buffer::new_texture`, `Device::new_heap_texture`), answering the descriptor's options with
/// `held`'s bits `taken` (a buffer's storage mode, as rt makes the texture of the buffer's whatever
/// the descriptor's; a heap's modes, which its resources take), what it is for the boot
/// diagnostics `diag`'s of its shape, lying over `copied` where the memory is a copy of host
/// pages (`super::copied`). Nil where rt refuses it, counted.
pub(super) fn make_over(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    desc: &TextureDesc,
    (held, taken): (u64, u64),
    over: impl FnOnce(&rt::TextureDescriptor, Option<u32>) -> Result<rt::Texture, rt::Error>,
    diag: impl FnOnce(&Shape) -> Arc<Made>,
    copied: Option<&Arc<Copied>>,
) -> Option<RtObj<RtTexture>> {
    let options = desc.resource_options.unwrap_or(0);
    let options = (options & !taken) | (held & taken);
    let made = made_of(device, sel, desc, options, over, diag);
    if let (Some(texture), Some(copied)) = (&made, copied) {
        let _ = texture.copied.set(Arc::clone(copied));
    }
    made
}

/// The texture `make` of `desc` makes for `sel` (the module's docs), of the resource options
/// `options`, hazard tracking yet to resolve, what it is for the boot diagnostics `diag`'s.
fn made_of(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    desc: &TextureDesc,
    options: u64,
    make: impl FnOnce(&rt::TextureDescriptor, Option<u32>) -> Result<rt::Texture, rt::Error>,
    diag: impl FnOnce(&Shape) -> Arc<Made>,
) -> Option<RtObj<RtTexture>> {
    let d = match rt_texture_descriptor(desc) {
        Ok(d) => d,
        Err(why) => return refused_metal(sel, &why),
    };
    let placed = desc.resource_index.map_or(0, |(index, _forced)| index);
    let (index, own) = device.index_for(sel, reaches(d.usage, d.pixel_format), placed)?;
    answered(device, sel, desc, options, own, make(&d, index), diag)
}

/// The texture of rt's answer `made` to `sel`, of `desc`, the resource options `options` and the
/// platform's own index `own`: nil where rt refused it, counted, `own` given back. What it is for
/// the boot diagnostics is `diag`'s of its shape.
fn answered(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    desc: &TextureDesc,
    options: u64,
    own: Option<u32>,
    made: Result<rt::Texture, rt::Error>,
    diag: impl FnOnce(&Shape) -> Arc<Made>,
) -> Option<RtObj<RtTexture>> {
    let shape = Shape {
        format: u32::try_from(desc.pixel_format).unwrap_or(u32::MAX),
        width: u32::try_from(desc.width).unwrap_or(u32::MAX),
        height: u32::try_from(desc.height).unwrap_or(u32::MAX),
        storage: (options & STORAGE_BITS) >> 4,
        usage: desc.usage,
    };
    let usage = match desc.usage {
        0 => UNKNOWN_ANSWERS,
        usage => usage & !DROPPED,
    };
    if let Err(error) = &made {
        let what = TextureDetail { desc, options };
        super::unbuilt::refusal_detail(sel, error, Some(&what));
    }
    device.made(sel, made, own, |texture| RtTexture {
        named: super::fault::Named::default(),
        rt: texture,
        device: device.clone_ref(),
        options: tracked(options),
        usage,
        protection_options: desc.protection_options.unwrap_or(0),
        own,
        purgeable: purgeable_state(),
        made: diag(&shape),
        tiled: OnceLock::new(),
        copied: OnceLock::new(),
        over: OnceLock::new(),
        held: OnceLock::new(),
    })
}

/// `MTLResourceStorageModePrivate` in a resource's options.
const STORAGE_PRIVATE: u64 = 0x20;

/// `MTLTextureUsageRenderTarget`.
const USAGE_RENDER_TARGET: u64 = 0x4;

/// The texture over a window swapchain's image (`super::swapchain`): rt's drawable `texture`, made
/// of [`drawable_descriptor`]'s descriptor at `own`, one of the platform's own indices, which its
/// drop gives back.
pub(super) fn drawable(
    device: &RtObj<RtDevice>,
    texture: rt::Texture,
    own: Option<u32>,
) -> RtObj<RtTexture> {
    let d = texture.descriptor();
    let shape = Shape {
        format: d.pixel_format.as_raw(),
        width: d.width,
        height: d.height,
        storage: STORAGE_PRIVATE >> 4,
        usage: USAGE_RENDER_TARGET,
    };
    let made = Made::new(Call::Drawable, &shape);
    RtObj::new(RtTexture {
        named: super::fault::Named::default(),
        rt: texture,
        device: device.clone_ref(),
        options: tracked(STORAGE_PRIVATE),
        usage: USAGE_RENDER_TARGET,
        protection_options: 0,
        own,
        purgeable: purgeable_state(),
        made,
        tiled: OnceLock::new(),
        copied: OnceLock::new(),
        over: OnceLock::new(),
        held: OnceLock::new(),
    })
}

/// The descriptor of a drawable's texture over a swapchain's image of `width` x `height` pixels:
/// a `Private` `BGRA8Unorm` render target of one level, which rt holds in `B8G8R8A8_UNORM`, the
/// swapchain's format. A colour target, shaders reach it ([`reaches`]), so it takes one of the
/// platform's own indices, which no texture the guest places at its own index takes: such a
/// texture's `gpuResourceID` is the same whether or not a window is open.
pub(super) fn drawable_descriptor(width: u32, height: u32) -> rt::TextureDescriptor {
    let mut d = rt::TextureDescriptor::new(
        rt::TextureType::Type2D,
        rt::PixelFormat::BGRA8Unorm,
        width,
        height,
    );
    d.usage = rt::TextureUsage::RENDER_TARGET;
    d.storage_mode = rt::StorageMode::Private;
    d
}

/// `options` with Metal's default hazard tracking resolved to tracked, as Metal's resources
/// answer it.
pub(super) fn tracked(options: u64) -> u64 {
    match (options >> 8) & 0x3 {
        0 => options | HAZARD_TRACKED,
        _ => options,
    }
}

/// A new resource's purgeable state: `NonVolatile`.
pub(super) fn purgeable_state() -> AtomicU64 {
    AtomicU64::new(NON_VOLATILE)
}

/// `setPurgeableState:` of a resource whose state `held` holds: the state before the change, kept
/// as Metal keeps it. `KeepCurrent` changes nothing, and `NonVolatile`, `Volatile` and `Empty`
/// take its place. Nothing is discarded at `Empty`, whose contents Metal leaves undefined. A raw
/// state no `MTLPurgeableState` is is refused, as Metal's validation refuses it
/// (`rt-refused-metal`), changing nothing.
pub(super) fn set_purgeable(held: &AtomicU64, state: u64) -> u64 {
    match state {
        KEEP_CURRENT => held.load(Ordering::Relaxed),
        NON_VOLATILE..=EMPTY => held.swap(state, Ordering::Relaxed),
        _ => {
            let why = format!("purgeable state {state}");
            refused_as("setPurgeableState:", RtRefusal::Refused, &why);
            held.load(Ordering::Relaxed)
        }
    }
}

/// The plane texture `plane` of an exported surface rt made (`super::surface`), of `desc`, for
/// `sel`: a texture with no index of its own, the image under the textures made as its views, as
/// [`RtObj::new_texture_for_views`]'s is. On Windows, a plane of a surface over Direct3D 11
/// textures (`super::d3d11_surface`). A recording holds it as it holds an image for views
/// (`recorded_image`): the device's `newTextureWithDescriptor:` of a descriptor of `desc` marked
/// outside the guest's texture indices, so the views, copies and commands that name the plane
/// name its stream id. What the other API writes into the plane is not in the stream.
pub(super) fn surface_plane(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    desc: &TextureDesc,
    plane: rt::Texture,
) -> Option<RtObj<RtTexture>> {
    let options = desc.resource_options.unwrap_or(0);
    let diag = |shape: &Shape| Made::new(Call::SurfaceImage, shape);
    recorded_image(device, desc, || {
        answered(device, sel, desc, options, None, Ok(plane), diag)
    })
}

/// An image for views, or a surface's plane (`surface_plane`), as a recording holds it: the
/// device's `newTextureWithDescriptor:` of a descriptor of `desc` whose `m2v.textureDescriptor`
/// record marks it [`crate::recorder::OUTSIDE`] ([`crate::recorder::outside_descriptor`]), so a
/// player places it outside the guest's texture indices, as rt gives it none, and the views, copies and commands that name the image name its stream
/// id. `make` makes the image; where the process does not record, nothing else is made.
fn recorded_image(
    device: &RtObj<RtDevice>,
    desc: &TextureDesc,
    make: impl FnOnce() -> Option<RtObj<RtTexture>>,
) -> Option<RtObj<RtTexture>> {
    if !crate::recorder::on() {
        return make();
    }
    let descriptor = RtObj::new(super::TextureFields::new(*desc));
    crate::recorder::outside_descriptor(&descriptor, desc);
    let recorded_call =
        crate::recorder::rec_call!("newTextureWithDescriptor:", device, &descriptor);
    let recorded_answer = crate::recorder::run(make);
    crate::recorder::rec_done!(recorded_call, recorded_answer);
    recorded_answer
}

impl RtObj<RtDevice> {
    /// A texture of `desc` with no index of its own, the image under the textures made as its
    /// views (the module's docs, "Images for views"), as `newTextureWithDescriptor:` would make
    /// it otherwise. Nil where rt refuses it, counted. A recording holds it as the device's
    /// `newTextureWithDescriptor:` of a descriptor marked outside the guest's texture indices
    /// (`recorded_image`).
    pub fn new_texture_for_views(&self, desc: &TextureDesc) -> Option<RtObj<RtTexture>> {
        recorded_image(self, desc, || self.image_for_views(desc))
    }

    /// A texture of `desc` for views alone, as [`Self::new_texture_for_views`] makes one, with its
    /// image placed in a Private block of its own (the module's docs, "Placed images for views"):
    /// its bytes are the block's as they are. `None` where rt places none, counted by its cause,
    /// and where rt takes no descriptor of `desc`, as [`Self::new_texture_for_views`] then
    /// refuses it. A recording holds it as it holds [`Self::new_texture_for_views`]'s image: a
    /// player makes the image as one that is not placed, its bytes those the caller fills.
    pub fn new_placed_texture_for_views(&self, desc: &TextureDesc) -> Option<RtObj<RtTexture>> {
        recorded_image(self, desc, || self.placed_image_for_views(desc))
    }

    /// [`Self::new_texture_for_views`]'s image, unrecorded.
    fn image_for_views(&self, desc: &TextureDesc) -> Option<RtObj<RtTexture>> {
        let sel = "newTextureWithDescriptor:";
        let d = match rt_texture_descriptor(desc) {
            Ok(d) => d,
            Err(why) => return refused_metal(sel, &why),
        };
        let options = desc.resource_options.unwrap_or(0);
        let made = self.rt().new_texture_for_views(&d);
        let diag = |shape: &Shape| Made::new(Call::SurfaceImage, shape);
        answered(self, sel, desc, options, None, made, diag)
    }

    /// [`Self::new_placed_texture_for_views`]'s image, unrecorded.
    fn placed_image_for_views(&self, desc: &TextureDesc) -> Option<RtObj<RtTexture>> {
        let sel = "newTextureWithDescriptor:offset:";
        let d = rt_texture_descriptor(desc).ok()?;
        let placed = (|| {
            let rt::SizeAndAlign { size, align } = self.rt().heap_texture_size_and_align(&d)?;
            let block = self
                .rt()
                .take_block(rt::MemoryClass::Private, size.max(align))?;
            let heap = rt::HeapRange {
                block: block.id,
                offset: 0,
                len: block.len,
            };
            // The texture holds the block from here: this handle goes at once.
            let made = self.rt().new_heap_texture_for_views(&d, heap, 0);
            drop(block);
            made.map(|texture| (texture, heap.len))
        })();
        let (texture, len) = match placed {
            Ok(placed) => placed,
            Err(error) => {
                let code = match error {
                    rt::Error::Unsupported(rt::Gap::HeapTexture { .. }) => UNPLACED_OWN_CODE,
                    _ => UNPLACED_REFUSED_CODE,
                };
                crate::answer::count_kind(code);
                if !UNPLACED_LOGGED.swap(true, Ordering::Relaxed) {
                    log::info!(
                        kind:% = code;
                        "a shape's image of format {} and {}x{}: rt places none in a block of its \
                         own ({error:?}); it takes rt's image as before ({code}; the rest are \
                         counted)",
                        desc.pixel_format, desc.width, desc.height
                    );
                }
                return None;
            }
        };
        crate::answer::count_kind(PLACED_IMAGES_CODE);
        crate::answer::count_kind_by(PLACED_IMAGE_BYTES_CODE, len);
        let options = desc.resource_options.unwrap_or(0);
        let diag = |shape: &Shape| Made::new(Call::SurfaceImage, shape);
        answered(self, sel, desc, options, None, Ok(texture), diag)
    }

    /// The index a texture `sel` makes takes, as rt takes it, and whether it is one of the
    /// platform's own: none where shaders do not reach it (`reached`), the guest's `placed` where
    /// it places one (0: none), else one of the platform's own. `None` where none can be had,
    /// counted.
    pub(super) fn index_for(
        &self,
        sel: &'static str,
        reached: bool,
        placed: u64,
    ) -> Option<(Option<u32>, Option<u32>)> {
        if !reached {
            return Some((None, None));
        }
        if placed != 0 {
            let Ok(index) = u32::try_from(placed) else {
                return refused_metal(sel, &format!("resource index {placed}"));
            };
            return Some((Some(index), None));
        }
        match self.indices().take(self.rt()) {
            Ok(Some(index)) => Some((Some(index), Some(index))),
            Ok(None) => {
                crate::core::platform::device::note_relayed_gap();
                let why = "no texture index of the platform's own is left";
                refused_as(sel, RtRefusal::Limit, &why);
                None
            }
            Err(error) => {
                rt_refused_texture_create(sel, &error);
                None
            }
        }
    }

    /// The texture `sel` made, of rt's answer `made`, or nil where rt refused it, counted, its
    /// own index given back.
    fn made(
        &self,
        sel: &'static str,
        made: Result<rt::Texture, rt::Error>,
        own: Option<u32>,
        texture: impl FnOnce(rt::Texture) -> RtTexture,
    ) -> Option<RtObj<RtTexture>> {
        match made {
            Ok(made) => Some(RtObj::new(texture(made))),
            Err(error) => {
                if let Some(index) = own {
                    self.indices().give_back(index);
                }
                rt_refused_texture_create(sel, &error);
                None
            }
        }
    }
}

/// Whether shaders reach a texture of `usage` and `format`, as rt decides where it takes an
/// index: shader-read or shader-write usage, `Unknown`, which Metal takes as both, or
/// render-target usage of a format with no depth or stencil aspect, a colour target, which a
/// fragment function reads through its attachment. A depth or stencil target shaders neither
/// read nor write is not reached.
pub(super) fn reaches(usage: rt::TextureUsage, format: rt::PixelFormat) -> bool {
    let shaders = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
    let colour_target = usage.contains(rt::TextureUsage::RENDER_TARGET)
        && !metal2vulkan::metal::format::is_depth_stencil(format);
    usage.is_empty() || usage.intersects(shaders) || colour_target
}

/// The slices of a texture of `d`: a cube's six faces each a slice, a 3D texture's one.
fn slice_count(d: &rt::TextureDescriptor) -> u32 {
    match d.texture_type {
        rt::TextureType::Cube | rt::TextureType::CubeArray => d.array_length.saturating_mul(6),
        rt::TextureType::Type3D => 1,
        _ => d.array_length,
    }
}

/// The levels or slices `r` names, in rt's words; `None` where one passes `u32::MAX`.
fn words(r: NsRange) -> Option<Range<u32>> {
    let end = r.location.checked_add(r.length)?;
    Some(u32::try_from(r.location).ok()?..u32::try_from(end).ok()?)
}

/// rt's region of `r`; `None` where a value passes `u32::MAX`.
fn rt_region(r: MtlRegion) -> Option<rt::Region> {
    let word = |v: u64| u32::try_from(v).ok();
    Some(rt::Region {
        origin: [word(r.origin.x)?, word(r.origin.y)?, word(r.origin.z)?],
        size: [
            word(r.size.width)?,
            word(r.size.height)?,
            word(r.size.depth)?,
        ],
    })
}

/// A creator's refusal of a value rt has none of, as Metal's validation refuses it: noted as a gap
/// of the platform's own and counted (`rt-refused-metal`). Nil.
pub(super) fn refused_metal<T>(sel: &'static str, why: &dyn std::fmt::Display) -> Option<T> {
    crate::core::platform::device::note_relayed_gap();
    refused_as(sel, RtRefusal::Refused, why);
    None
}

/// The platform's own indices of a class, textures' (the module's docs) or sampler states'
/// (`super::state`): those the guest's reservations hold are never taken. An index given back
/// waits for every command buffer its device had committed then to complete, by its device's
/// commit order (`super::fault`'s `Flights`), before it is taken again: rt writes the new owner
/// into the index's array element at once, which work committed before the release could read,
/// through a slot it bound or an argument buffer holding the old owner's ID, whether or not it
/// named the old owner. A command buffer committed later waits for nothing of it.
pub struct OwnIndices {
    /// Their class.
    class: rt::IndexClass,
    /// What was taken and reserved.
    own: Mutex<Own>,
    /// Its device's command buffers committed and not complete, by their commit order, which an
    /// index given back waits on.
    order: Arc<Flights>,
    /// How many taken were given back before. A take of an index not taken before counts in
    /// neither.
    taken_again: AtomicU64,
    /// How many were given back.
    given_back: AtomicU64,
}

/// [`OwnIndices`]' state.
struct Own {
    /// The indices a reservation holds, a bit each.
    reserved: Vec<u64>,
    /// The class's first index.
    first: u32,
    /// The next index not taken yet, counting down; none once every one was taken.
    fresh: Option<u32>,
    /// The indices given back, the longest ago first, each with the number of the last command
    /// buffer its device had committed at the give-back ([`Flights::last`]), which it waits for
    /// with every one before it; one reserved since goes as a take reaches it.
    free: VecDeque<(u32, u64)>,
}

impl OwnIndices {
    /// None of `class` taken, none reserved, its give-backs waiting on the commit order `order`,
    /// its device's.
    pub fn new(class: rt::IndexClass, order: Arc<Flights>) -> OwnIndices {
        OwnIndices {
            class,
            own: Mutex::new(Own::new(class.range())),
            order,
            taken_again: AtomicU64::new(0),
            given_back: AtomicU64::new(0),
        }
    }

    /// How many taken were given back before.
    pub fn taken_again(&self) -> u64 {
        self.taken_again.load(Ordering::Relaxed)
    }

    /// How many were given back ([`OwnIndices::give_back`]).
    pub fn given_back(&self) -> u64 {
        self.given_back.load(Ordering::Relaxed)
    }

    /// What the boot diagnostics name the class's objects (`super::diag`).
    fn what(&self) -> &'static str {
        match self.class {
            rt::IndexClass::Sampler => "sampler",
            _ => "texture",
        }
    }

    /// Notes `indices` as the guest's, never to be taken: those past the class's range are none of
    /// its.
    pub fn reserve(&self, indices: &[u64]) {
        let mut own = self.own.lock();
        for &index in indices {
            if let Ok(index) = u32::try_from(index) {
                own.reserve(index);
            }
        }
    }

    /// An index for an object of the platform's own, granted on `device` where it is new; `None`
    /// where every index is reserved, held, or given back and waiting for a command buffer
    /// committed before its give-back ([`Own::next`]). One given back before is counted
    /// ([`OwnIndices::taken_again`]).
    pub fn take(&self, device: &rt::Device) -> Result<Option<u32>, rt::Error> {
        let done = self.order.done_through();
        let mut own = self.own.lock();
        let Some((index, fresh)) = own.next(done) else {
            return Ok(None);
        };
        if fresh {
            match device.grant_indices(self.class, index, 1) {
                Ok(_) => {}
                Err(rt::Error::Refused(rt::Refusal::IndicesGranted { .. })) => {}
                Err(error) => {
                    own.fresh = Some(index);
                    return Err(error);
                }
            }
        }
        drop(own);
        if !fresh {
            self.taken_again.fetch_add(1, Ordering::Relaxed);
            super::diag::own_index_taken_again(self.what());
        }
        Ok(Some(index))
    }

    /// Gives `index` back, counted ([`OwnIndices::given_back`]): it is taken again after those
    /// given back before it whose waits are over, once every command buffer its device has
    /// committed by now, on any of its queues, has completed, its first completed handler run, or
    /// rt refused its commit ([`Flights::done_through`]), in whatever order they complete. One
    /// that never completes (a wait for an event never signalled, a GPU hang rt does not end)
    /// keeps the index from every take: a take looks at each waiting index once and passes it over,
    /// so the class runs short and its creations are refused as at its last, and nothing waits or
    /// spins. A lost device ends each of its command buffers with an error, which completes it.
    pub fn give_back(&self, index: u32) {
        let after = self.order.last();
        self.own.lock().give_back(index, after);
        self.given_back.fetch_add(1, Ordering::Relaxed);
        super::diag::own_index_given_back(self.what());
    }

    /// Leaves at most `n` of the class's indices not taken before, the next ones down: a test's
    /// small range, which its takes use up before they take any given back.
    #[cfg(any(test, feature = "testkit"))]
    pub fn leave_fresh(&self, n: u32) {
        let mut own = self.own.lock();
        match (own.fresh, n.checked_sub(1)) {
            (Some(next), Some(more)) => own.first = own.first.max(next.saturating_sub(more)),
            _ => own.fresh = None,
        }
    }
}

impl Own {
    /// None taken from `range`, none reserved.
    fn new(range: Range<u32>) -> Own {
        Own {
            reserved: vec![0; range.end.div_ceil(64).to_usize()],
            first: range.start,
            fresh: (range.end.checked_sub(1)).filter(|&last| last >= range.start),
            free: VecDeque::new(),
        }
    }

    /// Notes `index` as reserved: one past the class's last is none of its.
    fn reserve(&mut self, index: u32) {
        if let Some(word) = self.reserved.get_mut((index / 64).to_usize()) {
            *word |= 1 << (index % 64);
        }
    }

    /// Whether a reservation holds `index`.
    fn is_reserved(&self, index: u32) -> bool {
        let word = self.reserved.get((index / 64).to_usize());
        word.is_some_and(|word| word & (1 << (index % 64)) != 0)
    }

    /// The next index to take, and whether it is new (not given back before): the highest not
    /// taken yet and not reserved, else the one given back longest ago that is not reserved since
    /// and whose wait is over, every command buffer up to the one it waits for done (`done`, its
    /// device's [`Flights::done_through`]); one still waiting stays in its place. `None` where
    /// neither is left.
    fn next(&mut self, done: u64) -> Option<(u32, bool)> {
        while let Some(index) = self.fresh {
            self.fresh = (index.checked_sub(1)).filter(|&next| next >= self.first);
            if !self.is_reserved(index) {
                return Some((index, true));
            }
        }
        let mut at = 0;
        while let Some(&(index, after)) = self.free.get(at) {
            if self.is_reserved(index) {
                self.free.remove(at);
            } else if after > done {
                at += 1;
            } else {
                return (self.free.remove(at)).map(|(index, _)| (index, false));
            }
        }
        None
    }

    /// Gives `index` back, to wait for the command buffers up to number `after`, but one reserved
    /// since it was taken.
    fn give_back(&mut self, index: u32, after: u64) {
        if !self.is_reserved(index) {
            self.free.push_back((index, after));
        }
    }
}

impl GpuResource<RtPlatform> for RtObj<RtTexture> {
    fn device(&self) -> Option<RtObj<RtDevice>> {
        Some(self.device.clone_ref())
    }

    /// The descriptor's, Metal's default hazard tracking resolved to tracked (the module's docs).
    fn resource_options(&self) -> u64 {
        self.options
    }

    /// Labels are diagnostics: none is kept.
    fn set_label(&self, label: &str) -> RaiseResult<()> {
        let _ = label;
        Ok(())
    }

    /// [`set_purgeable`]'s.
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        let recorded_call = crate::recorder::rec_call!("setPurgeableState:", self, state);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<u64> {
            Ok(set_purgeable(&self.purgeable, state))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Bits 4 to 7 of its resource options.
    fn storage_mode(&self) -> u64 {
        (self.options >> 4) & 0xf
    }
}

impl Texture<RtPlatform> for RtObj<RtTexture> {
    fn array_length(&self) -> u64 {
        u64::from(self.rt.descriptor().array_length)
    }

    fn depth(&self) -> u64 {
        u64::from(self.rt.descriptor().depth)
    }

    /// A tiled texture's, or a view's of one, next use takes the buffer's bytes again
    /// (`super::tiled`). Nothing for any other: a texture over a buffer reads the buffer's bytes,
    /// whose changes a Managed buffer's `didModifyRange:` names to rt.
    fn did_modify_data(&self) {
        let recorded_call = crate::recorder::rec_call!("didModifyData", self);
        crate::recorder::run(|| {
            if let Some(tiled) = self.tiled() {
                tiled.modified();
            }
        });
        crate::recorder::rec_done!(recorded_call, ());
    }

    /// rt's: its index and the translator's word of its format and swizzle.
    fn gpu_resource_id(&self) -> u64 {
        crate::recorder::plain_resource_id(self, || self.rt.gpu_resource_id().to_bits())
    }

    fn height(&self) -> u64 {
        u64::from(self.rt.descriptor().height)
    }

    /// No: the platform makes no shareable texture yet ([`Unbuilt::SharedTexture`]).
    fn is_shareable(&self) -> bool {
        false
    }

    fn mipmap_level_count(&self) -> u64 {
        u64::from(self.rt.descriptor().mipmap_level_count)
    }

    /// Unbuilt: nil ([`Unbuilt::SharedTexture`]), as Metal answers for a texture not shareable.
    fn new_shared_texture_handle(
        &self,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::SharedTextureHandle>> {
        let recorded_call = crate::recorder::rec_call!("newSharedTextureHandle", self);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::SharedTextureHandle>> {
                refuse_create(Unbuilt::SharedTexture, "newSharedTextureHandle");
                Ok(None)
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn new_texture_view(&self, pixel_format: u64) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call =
            crate::recorder::rec_call!("newTextureViewWithPixelFormat:", self, pixel_format);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let sel = "newTextureViewWithPixelFormat:";
            Ok(self.view(sel, pixel_format, None, None, None, None, 0))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn new_texture_view_at_index(
        &self,
        pixel_format: u64,
        resource_index: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newTextureViewWithPixelFormat:resourceIndex:",
            self,
            pixel_format,
            resource_index
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let sel = "newTextureViewWithPixelFormat:resourceIndex:";
            Ok(self.view(sel, pixel_format, None, None, None, None, resource_index))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn new_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newTextureViewWithPixelFormat:textureType:levels:slices:",
            self,
            pixel_format,
            texture_type,
            level_range,
            slice_range
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:";
            let (levels, slices) = (Some(level_range), Some(slice_range));
            Ok(self.view(
                sel,
                pixel_format,
                Some(texture_type),
                levels,
                slices,
                None,
                0,
            ))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn new_swizzled_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
        swizzle: Swizzle,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:",
            self,
            pixel_format,
            texture_type,
            level_range,
            slice_range,
            swizzle
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:";
            let (levels, slices) = (Some(level_range), Some(slice_range));
            let texture_type = Some(texture_type);
            Ok(self.view(
                sel,
                pixel_format,
                texture_type,
                levels,
                slices,
                Some(swizzle),
                0,
            ))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn new_swizzled_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        swizzle: Swizzle,
        resource_index: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:",
            self,
            pixel_format,
            texture_type,
            levels,
            slices,
            swizzle,
            resource_index
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let sel =
                "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:";
            let (levels, slices) = (Some(levels), Some(slices));
            let (texture_type, swizzle) = (Some(texture_type), Some(swizzle));
            Ok(self.view(
                sel,
                pixel_format,
                texture_type,
                levels,
                slices,
                swizzle,
                resource_index,
            ))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn new_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        resource_index: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:",
            self,
            pixel_format,
            texture_type,
            levels,
            slices,
            resource_index
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex:";
            let (levels, slices) = (Some(levels), Some(slices));
            let texture_type = Some(texture_type);
            Ok(self.view(
                sel,
                pixel_format,
                texture_type,
                levels,
                slices,
                None,
                resource_index,
            ))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn pixel_format(&self) -> u64 {
        u64::from(self.rt.descriptor().pixel_format.as_raw())
    }

    /// The descriptor's, 0 where none was set.
    fn protection_options(&self) -> u64 {
        self.protection_options
    }

    /// rt's, of a copy of the bytes (the module's docs, "A copy"): a range at or near NULL dies, as
    /// Metal's read of it faults (`crate::gpu::resource::first_bytes`). What rt refuses (a Private
    /// texture, a region outside the level or off its format's block grid, rows that do not cover
    /// it) changes nothing, counted, where Metal's validation aborts. Of a tiled texture's level 0
    /// the bytes go into its buffer's bytes, which its next use takes (`super::tiled`).
    fn replace_region(
        &self,
        region: MtlRegion,
        level: u64,
        slice: u64,
        pixel_bytes: HostRange<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!(
            "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:",
            self,
            region,
            level,
            slice,
            pixel_bytes,
            bytes_per_row,
            bytes_per_image
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let (Some(r), Ok(level), Ok(slice)) = (
                rt_region(region),
                u32::try_from(level),
                u32::try_from(slice),
            ) else {
                let why = format!("the region {region:?} of level {level} and slice {slice}");
                refused_as(REPLACE, RtRefusal::Refused, &why);
                return Ok(());
            };
            let bytes = crate::gpu::resource::first_bytes(REPLACE, pixel_bytes, usize::MAX);
            counted_copy(bytes.len());
            if level == 0
                && let Some(tiled) = self.tiled()
            {
                if let Err((kind, why)) =
                    tiled.replace(region, u64::from(slice), &bytes, bytes_per_row)
                {
                    refused_as(REPLACE, kind, &why);
                }
                return Ok(());
            }
            if let Err(error) = (self.rt).replace_region(
                r,
                level,
                slice,
                rt::HostBytes::from(bytes.as_slice()),
                bytes_per_row,
                bytes_per_image,
            ) {
                rt_refused(REPLACE, &error);
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Its index (the module's docs): 0 for a texture no shader reaches.
    fn resource_index(&self) -> u64 {
        u64::from(self.rt.gpu_resource_id().index)
    }

    fn sample_count(&self) -> u64 {
        u64::from(self.rt.descriptor().sample_count)
    }

    fn texture_type(&self) -> u64 {
        u64::from(self.rt.descriptor().texture_type.as_raw())
    }

    /// Its index, as [`Texture::resource_index`].
    fn unique_identifier(&self) -> u64 {
        u64::from(self.rt.gpu_resource_id().index)
    }

    /// Metal's answer (the module's docs).
    fn usage(&self) -> u64 {
        self.usage
    }

    fn width(&self) -> u64 {
        u64::from(self.rt.descriptor().width)
    }
}

/// Whether the process copied a `replaceRegion:`'s bytes yet.
static COPIED: AtomicBool = AtomicBool::new(false);

/// Whether an image for views rt placed none of was logged
/// ([`RtObj::new_placed_texture_for_views`]).
static UNPLACED_LOGGED: AtomicBool = AtomicBool::new(false);

/// Counts a `replaceRegion:` copy of `bytes` bytes ([`COPY_CODE`], [`COPY_BYTES_CODE`]); the
/// process's first writes its line.
fn counted_copy(bytes: usize) {
    crate::answer::count_kind(COPY_CODE);
    crate::answer::count_kind_by(COPY_BYTES_CODE, u64::try_from(bytes).unwrap_or(u64::MAX));
    if !COPIED.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: "reims_vgpu_vulkan::rt_platform",
            "{REPLACE}: the guest's bytes are copied before metal2vulkan-runtime takes them, a \
             copy Metal does not make ({COPY_CODE}), counted from here"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A refused texture's detail names its descriptor's pixel format, size, levels, array
    /// length, sample count, texture type and usage, as Metal numbers them, and the storage mode
    /// of its resource options.
    #[test]
    fn a_refused_texture_s_detail_names_its_descriptor() {
        let desc = TextureDesc {
            texture_type: 4,
            pixel_format: 70,
            width: 826,
            height: 826,
            sample_count: 4,
            usage: 0x5,
            ..TextureDesc::default()
        };
        let what = TextureDetail {
            desc: &desc,
            options: STORAGE_PRIVATE,
        };
        assert_eq!(
            what.to_string(),
            "a texture of format 70 826x826x1, 1 levels, array length 1, 4 samples, type 4, \
             usage 0x5, storage 2"
        );
    }

    /// The platform's own indices: the highest first, none reserved, each fresh one once, then
    /// those given back in the order they were, never one reserved since.
    #[test]
    fn own_indices_go_down_and_come_back_in_order() {
        fn next(own: &mut Own) -> Option<(u32, bool)> {
            own.next(0)
        }
        let mut own = Own::new(1..8);
        own.reserve(7);
        own.reserve(5);
        own.reserve(64);
        assert_eq!(next(&mut own), Some((6, true)));
        assert_eq!(next(&mut own), Some((4, true)));
        own.give_back(6, 0);
        own.give_back(4, 0);
        own.give_back(5, 0);
        assert_eq!(
            [next(&mut own), next(&mut own), next(&mut own)],
            [Some((3, true)), Some((2, true)), Some((1, true))]
        );
        own.reserve(6);
        assert_eq!(next(&mut own), Some((4, false)));
        assert_eq!(next(&mut own), None);
        own.give_back(2, 0);
        assert_eq!(next(&mut own), Some((2, false)));
    }

    /// A texture and a sampler state of the platform's own indices on `device`, made with
    /// `platform`'s descriptor of a small texture, each `None` where no index is left for it.
    fn own_pair(
        platform: &RtPlatform,
        device: &RtObj<RtDevice>,
    ) -> (
        Option<RtObj<RtTexture>>,
        Option<RtObj<super::super::state::RtSampler>>,
    ) {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::values::SamplerDesc;
        use crate::core::raise::OnFault;
        let small = TextureDesc {
            width: 4,
            height: 4,
            ..TextureDesc::default()
        };
        let descriptor = platform.texture_descriptor(&small);
        let texture = device.new_texture(OnFault::Return, &descriptor);
        let sampler = device.new_sampler_state(&SamplerDesc::default());
        (
            texture.expect("never a raise"),
            sampler.expect("never a raise"),
        )
    }

    /// Each class's (textures', then sampler states') takes of indices given back and give-backs
    /// on `device`.
    fn own_counts(device: &RtObj<RtDevice>) -> [[u64; 2]; 2] {
        let counts = |own: &OwnIndices| [own.taken_again(), own.given_back()];
        [counts(device.indices()), counts(device.sampler_indices())]
    }

    /// Waits up to 10 s for `device`'s command buffers committed up to `number` to be done
    /// ([`Flights::done_through`]); whether they were.
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the GPU's work
    fn done_through(device: &RtObj<RtDevice>, number: u64) -> bool {
        for _ in 0..1000 {
            if device.loss().flights().done_through() >= number {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        false
    }

    /// On the GPU the test holds, each class's fresh indices used up: an own index goes by its
    /// device's commit order, not by what reads its last owner. One given back while a command
    /// buffer whose compute encoder bound its last owner, a texture or a sampler state, is still
    /// uncommitted is taken again at once, as one whose last owner no command buffer bound: no
    /// command buffer committed before the give-back is in flight. One given back while such a
    /// command buffer is committed and in flight, held by a wait for a shared event, is not taken
    /// again, and none is left; once it completes and its completed handlers ran, a take gets it.
    /// Each give-back counts once, and each take of an index given back once.
    #[test]
    fn own_indices_wait_for_a_command_buffer_committed_before_their_release_on_vulkan() {
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::event::{EventRef, SharedEvent as _};
        use crate::core::platform::resource::SamplerState as _;
        use crate::core::raise::OnFault;
        let on = OnFault::Return;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let made = || {
            let made = own_pair(&platform, &device);
            (made.0.expect("a texture"), made.1.expect("a sampler state"))
        };
        // The first of each class is bound by a command buffer left uncommitted, the second by
        // one committed, and the third by none.
        let [(t0, s0), (t1, s1), (t2, s2)] = [made(), made(), made()];
        let textures = [&t0, &t1, &t2].map(|t| t.resource_index());
        let samplers = [&s0, &s1, &s2].map(|s| s.resource_index());
        device.indices().leave_fresh(0);
        device.sampler_indices().leave_fresh(0);
        let base = own_counts(&device);
        let since = || {
            let now = own_counts(&device);
            [0usize, 1].map(|c| [0usize, 1].map(|k| now[c][k] - base[c][k]))
        };

        // A command buffer whose compute encoder binds `t` and `s`, behind a wait for the shared
        // event where `wait`.
        let shared = device.new_shared_event().expect("a shared event");
        let queue = device.new_command_queue().expect("a command queue");
        let bound = |t: &RtObj<RtTexture>, s: &RtObj<super::super::state::RtSampler>, wait| {
            let commands = queue.command_buffer(on).expect("never a raise");
            let commands = commands.expect("a command buffer");
            if wait {
                let wait = commands.encode_wait(Some(EventRef::Shared(&shared)), 1);
                assert_eq!(wait, Ok(()));
            }
            let encoder = commands.compute_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a compute encoder");
            let binds = [
                ComputeCmd::SetTextures {
                    textures: &[Some(t)],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::SetSamplerStates {
                    samplers: &[Some(s)],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::EndEncoding,
            ];
            for bind in binds {
                assert_eq!(encoder.encode(on, bind), Ok(()));
            }
            drop(encoder);
            commands
        };
        let none_left = || {
            let (t, s) = own_pair(&platform, &device);
            assert!(t.is_none() && s.is_none(), "an index taken that work reads");
        };

        // Uncommitted: the first of each, given back, is taken again at once, and then the third,
        // given back after it; then none is left.
        let first = bound(&t0, &s0, false);
        assert_eq!(since(), [[0, 0]; 2]);
        drop((t0, s0));
        drop((t2, s2));
        let (u0, v0) = made();
        assert_eq!(
            (u0.resource_index(), v0.resource_index()),
            (textures[0], samplers[0])
        );
        let (u1, v1) = made();
        assert_eq!(
            (u1.resource_index(), v1.resource_index()),
            (textures[2], samplers[2])
        );
        none_left();
        assert_eq!(since(), [[2, 2]; 2]);
        drop(first);

        // Committed and in flight, held by the wait: the second of each, given back, is passed
        // over, and none is left.
        let second = bound(&t1, &s1, true);
        assert_eq!(second.commit(on), Ok(()));
        let committed = device.loss().flights().last();
        drop((t1, s1));
        none_left();
        assert_eq!(since(), [[2, 3]; 2]);
        // Completed, its completed handlers run: the second of each is taken again.
        assert_eq!(shared.set_signaled_value(1), Ok(()));
        assert_eq!(second.wait_until_completed(on), Ok(()));
        assert!(
            done_through(&device, committed),
            "the command buffer never completed"
        );
        let (u2, v2) = made();
        assert_eq!(
            (u2.resource_index(), v2.resource_index()),
            (textures[1], samplers[1])
        );
        assert_eq!(since(), [[3, 3]; 2]);
        drop((u0, u1, u2, v0, v1, v2));
    }

    /// On the GPU the test holds, each class's fresh indices used up: a command buffer committed
    /// on one queue, held by a wait for a shared event, whose compute encoder binds only a buffer
    /// that holds a texture's and a sampler state's `gpuResourceID`s, an argument buffer that
    /// reads them through their indices with no use of either named, keeps their indices from
    /// every take once the two go, while a command buffer committed after it on another queue,
    /// before the two went, has completed, and while one committed on that queue after they went
    /// waits for the event; once the first completes, a take gets each, the last still waiting.
    #[test]
    fn own_indices_wait_for_every_command_buffer_committed_before_their_release_on_vulkan() {
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::event::{EventRef, SharedEvent as _};
        use crate::core::platform::resource::{Buffer as _, SamplerState as _, Texture as _};
        use crate::core::raise::OnFault;
        let on = OnFault::Return;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let (texture, sampler) = own_pair(&platform, &device);
        let (texture, sampler) = (
            texture.expect("a texture"),
            sampler.expect("a sampler state"),
        );
        let at = (texture.resource_index(), sampler.resource_index());
        device.indices().leave_fresh(0);
        device.sampler_indices().leave_fresh(0);
        let base = own_counts(&device);
        let since = || {
            let now = own_counts(&device);
            [0usize, 1].map(|c| [0usize, 1].map(|k| now[c][k] - base[c][k]))
        };

        // The argument buffer: the two's IDs, as a guest's argument buffer holds them.
        let ids = [texture.gpu_resource_id(), sampler.gpu_resource_id()];
        let arguments = device.new_buffer_of_length(on, 16, 0);
        let arguments = arguments.expect("never a raise").expect("a buffer");
        // SAFETY: a Shared buffer's contents are mapped, 16 bytes long, and no work reads them yet.
        unsafe {
            let bytes = ids.map(u64::to_le_bytes).concat();
            let to = arguments.contents().as_ptr::<u8>();
            to.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        }
        // A command buffer of `queue` whose compute encoder binds the argument buffer alone,
        // behind a wait for the shared event's `wait` where one is given, committed.
        let shared = device.new_shared_event().expect("a shared event");
        let committed = |queue: &RtObj<super::super::RtQueue>, wait: Option<u64>| {
            let commands = queue.command_buffer(on).expect("never a raise");
            let commands = commands.expect("a command buffer");
            if let Some(value) = wait {
                let wait = commands.encode_wait(Some(EventRef::Shared(&shared)), value);
                assert_eq!(wait, Ok(()));
            }
            let encoder = commands.compute_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a compute encoder");
            let bind = ComputeCmd::SetBuffers {
                buffers: &[Some(&arguments)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            };
            assert_eq!(encoder.encode(on, bind), Ok(()));
            assert_eq!(encoder.encode(on, ComputeCmd::EndEncoding), Ok(()));
            drop(encoder);
            assert_eq!(commands.commit(on), Ok(()));
            commands
        };
        let none_left = || {
            let (t, s) = own_pair(&platform, &device);
            assert!(t.is_none() && s.is_none(), "an index taken that work reads");
        };
        // Two queues that share no GPU queue, so the second's work runs past the first's wait.
        let first_queue = device
            .new_command_queue_unshared()
            .expect("a command queue");
        let other_queue = device
            .new_command_queue_unshared()
            .expect("a command queue");

        // Committed before the two go: the first, held by the wait, and a later one on the other
        // queue, which completes.
        let reading = committed(&first_queue, Some(1));
        let earlier = device.loss().flights().last();
        let later = committed(&other_queue, None);
        assert_eq!(later.wait_until_completed(on), Ok(()));
        let mut ended = device.loss().flights().last();
        assert!(ended > earlier);
        assert!(device.loss().flights().done_through() < earlier);
        drop((texture, sampler));
        assert_eq!(since(), [[0, 1]; 2]);
        none_left();
        // Committed after the two went, on the other queue, held by a wait for 2.
        let after = committed(&other_queue, Some(2));
        none_left();
        assert!(device.loss().flights().done_through() < earlier);
        // The first completes: each index is taken again, while the last still waits.
        assert_eq!(shared.set_signaled_value(1), Ok(()));
        assert_eq!(reading.wait_until_completed(on), Ok(()));
        assert!(
            done_through(&device, ended),
            "the command buffers never completed"
        );
        let (again_texture, again_sampler) = own_pair(&platform, &device);
        let again_texture = again_texture.expect("the texture's index, taken again");
        let again_sampler = again_sampler.expect("the sampler state's index, taken again");
        assert_eq!(
            (
                again_texture.resource_index(),
                again_sampler.resource_index()
            ),
            at
        );
        assert_eq!(since(), [[1, 1]; 2]);
        ended = device.loss().flights().last();
        assert!(device.loss().flights().done_through() < ended);
        assert_eq!(shared.set_signaled_value(2), Ok(()));
        assert_eq!(after.wait_until_completed(on), Ok(()));
        assert!(
            done_through(&device, ended),
            "the last command buffer never completed"
        );
        drop((again_texture, again_sampler));
    }

    /// A test's small range ([`OwnIndices::leave_fresh`]): the next ones down, then those given
    /// back.
    #[test]
    fn own_indices_left_fresh_are_the_next_ones_down() {
        let own = OwnIndices::new(rt::IndexClass::Sampler, Arc::default());
        own.leave_fresh(2);
        let mut state = own.own.lock();
        let mut next = || state.next(0);
        assert_eq!(
            [next(), next(), next()],
            [Some((3999, true)), Some((3998, true)), None]
        );
        state.give_back(3999, 0);
        assert_eq!(state.next(0), Some((3999, false)));
    }

    /// The texture class's own indices start at its last, 65,535.
    #[test]
    fn own_indices_start_at_the_class_s_last() {
        let own = OwnIndices::new(rt::IndexClass::Texture, Arc::default());
        own.reserve(&[65535, 1 << 40]);
        assert_eq!(own.own.lock().next(0), Some((65534, true)));
    }

    /// The commit order an own index given back waits on ([`Flights`]): a give-back waits for the
    /// last command buffer committed then and every one before it, whatever order they complete
    /// in, on whichever queues. A take passes over an index whose wait is not over for one given
    /// back after it whose wait is, takes none where only such indices are left, and takes each
    /// once the command buffers it waits for are done, a refused commit's included; a give-back
    /// with none committed waits for nothing, and one reserved since goes.
    #[test]
    fn own_indices_given_back_wait_for_every_command_buffer_committed_before() {
        use super::super::fault::Flight;
        let order = Arc::new(Flights::default());
        let commit = || {
            order.add(Flight {
                cb: None,
                committed: 0,
                encoders: Vec::new(),
            })
        };
        let own = OwnIndices::new(rt::IndexClass::Sampler, Arc::clone(&order));
        own.leave_fresh(4);
        let take = || -> Option<u32> {
            let mut own = own.own.lock();
            own.next(order.done_through()).map(|(index, _)| index)
        };
        let taken = [take(), take(), take(), take()];
        assert_eq!(taken, [3999, 3998, 3997, 3996].map(Some));
        assert_eq!((order.last(), order.done_through()), (0, 0));
        // None committed: 3999 waits for nothing.
        own.give_back(3999);
        assert_eq!(take(), Some(3999));
        // Committed in the order 1, 2, 3 (on any queues); 3998 waits for all three.
        let [first, second, third] = [commit(), commit(), commit()];
        assert_eq!([first, second, third], [1, 2, 3]);
        own.give_back(3998);
        // 4 is committed after the give-back; 3997, given back after it, waits for it too.
        let fourth = commit();
        own.give_back(3997);
        own.give_back(3996);
        own.own.lock().reserve(3996);
        // The later ones complete first: the first, still in flight, holds every give-back.
        order.remove(third);
        order.remove(second);
        assert_eq!(order.done_through(), 0);
        assert_eq!(take(), None);
        // The first completes: 3998 is taken; 3997 waits for the fourth still.
        order.remove(first);
        assert_eq!(order.done_through(), 3);
        assert_eq!(take(), Some(3998));
        assert_eq!(take(), None);
        // The fourth's commit is refused, which takes it off: 3997 is taken, and 3996, reserved
        // since, never.
        order.remove(fourth);
        assert_eq!(order.done_through(), 4);
        assert_eq!(take(), Some(3997));
        assert_eq!(take(), None);
        assert_eq!(own.given_back(), 4);
    }

    /// A texture reaches shaders by shader-read or shader-write usage, `Unknown`, or render-target
    /// usage of a colour format; a depth or stencil target of neither read nor write usage does
    /// not.
    #[test]
    fn shaders_reach_read_write_unknown_and_colour_target_usage() {
        use rt::PixelFormat as F;
        let usage = |raw: u32| rt::TextureUsage::try_from(raw).unwrap();
        for format in [F::BGRA8Unorm, F::Depth32Float] {
            assert!(reaches(usage(0), format));
            assert!(reaches(usage(1), format) && reaches(usage(2), format));
            assert!(reaches(usage(0x15), format));
            assert!(!reaches(usage(0x10), format) && !reaches(usage(0x8), format));
        }
        assert!(reaches(usage(4), F::BGRA8Unorm) && reaches(usage(4), F::RGBA16Float));
        assert!(reaches(usage(0x14), F::R32Uint));
        for depth in [F::Depth32Float, F::Stencil8, F::Depth32FloatStencil8] {
            assert!(!reaches(usage(4), depth) && !reaches(usage(0x14), depth));
        }
    }

    /// A texture of 0x8 alone goes to rt as one of `ShaderRead` alone, and so reaches shaders,
    /// takes an index and samples as that one does; 0x8 beside other bits goes as it is.
    #[test]
    fn a_texture_of_0x8_alone_is_rt_s_of_shader_read() {
        let rt_usage = |usage| {
            let desc = TextureDesc {
                usage,
                ..TextureDesc::default()
            };
            rt_texture_descriptor(&desc).map(|d| d.usage)
        };
        assert_eq!(rt_usage(0x8), rt_usage(0x1));
        assert_eq!(rt_usage(0x8), Ok(rt::TextureUsage::SHADER_READ));
        assert!(rt_usage(0x8).is_ok_and(|usage| reaches(usage, rt::PixelFormat::Depth32Float)));
        assert_eq!(rt_usage(0), Ok(rt::TextureUsage::UNKNOWN));
        assert_eq!(rt_usage(0xc).map(|u| u.as_raw()), Ok(0xc));
    }

    /// A view's default slices: a cube's six faces each, a 3D texture's one, else the array's.
    #[test]
    fn a_view_takes_every_slice_of_its_texture() {
        let d = |texture_type, array_length| {
            let mut d = rt::TextureDescriptor::new(texture_type, rt::PixelFormat::RGBA8Unorm, 4, 4);
            d.array_length = array_length;
            d
        };
        assert_eq!(slice_count(&d(rt::TextureType::Cube, 1)), 6);
        assert_eq!(slice_count(&d(rt::TextureType::CubeArray, 3)), 18);
        assert_eq!(slice_count(&d(rt::TextureType::Type3D, 1)), 1);
        assert_eq!(slice_count(&d(rt::TextureType::Type2DArray, 5)), 5);
        assert_eq!(words(NsRange::new(2, 3)), Some(2..5));
        assert_eq!(words(NsRange::new(u64::MAX, 1)), None);
        assert_eq!(words(NsRange::new(1 << 32, 0)), None);
    }

    /// Textures on the GPU the test holds: those the guest does not place take the platform's own
    /// indices, distinct and from the top down past the guest's reservation, a released one's index
    /// not taken again before the fresh ones; one the guest places answers its index, a colour
    /// render target's too, which a fragment function reads through its attachment; a depth target
    /// no shader reads or writes has none, the guest's placement dropped; `Unknown` usage answers
    /// Metal's 0x17 and 0x8 is dropped; a texture of 0x8 alone answers no usage and is rt's of
    /// `ShaderRead`, with an index, a format view of it answering no usage and a swizzled one
    /// refused and counted; views of another format, at a placed index, of one slice of a 2D array
    /// as a 2D texture and through a swizzle; `replaceRegion:` of a Shared texture copied and
    /// counted, of a Private one refused and counted; a texture type rt has none of refused and
    /// counted, never a raise.
    #[test]
    fn textures_and_their_indices_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::types::{MtlOrigin, MtlSize};
        use crate::core::raise::OnFault;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let make = |desc: &TextureDesc| {
            let made = device.new_texture(OnFault::Return, &platform.texture_descriptor(desc));
            made.expect("never a raise")
        };
        let index = |t: &RtObj<RtTexture>| t.resource_index();
        let small = TextureDesc {
            width: 4,
            height: 4,
            ..TextureDesc::default()
        };
        let guests: Vec<u64> = (1..=16320).collect();
        assert_eq!(device.reserve_resource_indices(0, &guests), Ok(()));

        let own: Vec<_> = (0..16).map(|_| make(&small).expect("a texture")).collect();
        let top: Vec<u64> = (0..16).map(|i| 65535 - i).collect();
        assert_eq!(own.iter().map(index).collect::<Vec<u64>>(), top);
        for t in &own {
            assert_eq!(t.gpu_resource_id() & 0xffff_ffff, index(t));
            assert_eq!(t.unique_identifier(), index(t));
            assert_eq!((t.width(), t.height(), t.pixel_format()), (4, 4, 70));
            assert_eq!((t.usage(), t.resource_options()), (1, HAZARD_TRACKED));
        }
        drop(own);
        assert_eq!(make(&small).as_ref().map(index), Some(65519));

        let placed = TextureDesc {
            resource_index: Some((7, false)),
            ..small
        };
        assert_eq!(make(&placed).as_ref().map(index), Some(7));
        let target = TextureDesc {
            usage: 0x4,
            resource_index: Some((9, false)),
            ..small
        };
        let target = make(&target).expect("a render target");
        assert_eq!(
            (index(&target), target.gpu_resource_id() & 0xffff_ffff),
            (9, 9)
        );
        let depth = TextureDesc {
            pixel_format: 252,
            usage: 0x4,
            resource_options: Some(0x20),
            resource_index: Some((11, false)),
            ..small
        };
        let depth = make(&depth).expect("a depth target");
        assert_eq!(
            (index(&depth), depth.gpu_resource_id() & 0xffff_ffff),
            (0, 0)
        );
        let unknown = make(&TextureDesc { usage: 0, ..small }).expect("a texture");
        assert_eq!(unknown.usage(), UNKNOWN_ANSWERS);
        assert_ne!(index(&unknown), 0);
        let dropped = make(&TextureDesc {
            usage: 0x9,
            ..small
        })
        .expect("a texture");
        assert_eq!(dropped.usage(), 0x1);
        let alone = make(&TextureDesc {
            usage: 0x8,
            ..small
        })
        .expect("a texture of 0x8 alone");
        assert_eq!(alone.usage(), 0);
        assert_ne!(index(&alone), 0);
        assert_eq!(alone.rt().descriptor().usage, rt::TextureUsage::SHADER_READ);
        let view = alone.new_texture_view(71).expect("never a raise");
        assert_eq!(view.expect("a format view").usage(), 0);

        let array = TextureDesc {
            texture_type: 3,
            array_length: 4,
            usage: 0x15,
            ..small
        };
        let array = make(&array).expect("a 2D array");
        let srgb = array.new_texture_view(71).expect("never a raise");
        let srgb = srgb.expect("an sRGB view");
        assert_eq!(
            (srgb.pixel_format(), srgb.array_length(), srgb.usage()),
            (71, 4, 0x15)
        );
        let at = array
            .new_texture_view_at_index(70, 8)
            .expect("never a raise");
        assert_eq!(at.as_ref().map(index), Some(8));
        let (level, slice) = (NsRange::new(0, 1), NsRange::new(2, 1));
        let one = array
            .new_texture_subview(70, 2, level, slice)
            .expect("never a raise");
        let one = one.expect("a slice's 2D view");
        assert_eq!((one.texture_type(), one.array_length()), (2, 1));
        let swizzle = Swizzle {
            red: 5,
            ..DEFAULT_SWIZZLE
        };
        let every = NsRange::new(0, 4);
        let swizzled = array.new_swizzled_texture_subview(70, 3, level, every, swizzle);
        let swizzled = swizzled.expect("never a raise").expect("a swizzled view");
        assert_eq!(swizzled.usage(), 0x11);

        let region = MtlRegion {
            origin: MtlOrigin { x: 0, y: 0, z: 0 },
            size: MtlSize {
                width: 4,
                height: 4,
                depth: 1,
            },
        };
        let bytes = [0x5a_u8; 64];
        let (copies, metal) = (counted(COPY_CODE), counted(RtRefusal::Refused.code()));
        let shared = make(&small).expect("a Shared texture");
        let pixels = HostRange::of_slice(&bytes);
        assert_eq!(shared.replace_region(region, 0, 0, pixels, 16, 0), Ok(()));
        assert_eq!(counted(COPY_CODE), copies + 1);
        assert_eq!(counted(RtRefusal::Refused.code()), metal);

        // An image for views takes no index; a view of it at the guest's index reads its texels.
        let image = TextureDesc { usage: 0, ..small };
        let image = device
            .new_texture_for_views(&image)
            .expect("an image for views");
        assert_eq!(
            (index(&image), image.gpu_resource_id() & 0xffff_ffff),
            (0, 0)
        );
        assert_eq!(image.replace_region(region, 0, 0, pixels, 16, 0), Ok(()));
        let view = image.new_texture_view_at_index(70, 10);
        let view = view
            .expect("never a raise")
            .expect("a view at the guest's index");
        assert_eq!(index(&view), 10);
        let mut texels = [0_u8; 64];
        view.read_pixels(&mut texels, 16);
        assert_eq!(texels, bytes);
        assert_eq!(counted(RtRefusal::Refused.code()), metal);
        let private = TextureDesc {
            resource_options: Some(0x20),
            ..small
        };
        let private = make(&private).expect("a Private texture");
        assert_eq!(private.storage_mode(), 2);
        assert_eq!(private.replace_region(region, 0, 0, pixels, 16, 0), Ok(()));
        assert_eq!(counted(RtRefusal::Refused.code()), metal + 1);
        assert!(
            make(&TextureDesc {
                texture_type: 99,
                ..small
            })
            .is_none()
        );
        assert_eq!(counted(RtRefusal::Refused.code()), metal + 2);
        let refused = alone.new_swizzled_texture_subview(70, 2, level, level, swizzle);
        assert!(refused.expect("never a raise").is_none());
        assert_eq!(counted(RtRefusal::Refused.code()), metal + 3);
    }

    /// On the GPU the test holds: a view of a guest surface texture, and a view of that view, is
    /// counted by its pixel format, and a view of a texture no surface's is not; the surface
    /// texture, declared with `PixelFormatView` usage, is counted for it.
    #[test]
    fn plane_views_count_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::raise::OnFault;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let make = |pixel_format| {
            let desc = TextureDesc {
                pixel_format,
                width: 4,
                height: 4,
                usage: 0x11,
                ..TextureDesc::default()
            };
            let made = device.new_texture(OnFault::Return, &platform.texture_descriptor(&desc));
            made.expect("never a raise").expect("a texture")
        };
        let view = |of: &RtObj<RtTexture>, pixel_format| {
            let made = of.new_texture_view(pixel_format).expect("never a raise");
            made.expect("a view")
        };
        let (srgb, plain_srgb) = ("rt-plane-views-format-81", "rt-plane-views-format-71");
        let pfv = super::super::diag::PLANE_PIXEL_FORMAT_VIEW_CODE;
        let before = (counted(srgb), counted("rt-plane-views-format-80"));
        let (pfv_before, plain_before) = (counted(pfv), counted(plain_srgb));

        let surface = make(80);
        let form = "over-buffer: no cache";
        super::super::diag::surface_texture(Some(&surface), 7, (0x5d5d_0000, 0), (0x11, 64), form);
        assert_eq!(counted(pfv), pfv_before + 1);
        let of_surface = view(&surface, 81);
        let _of_view = view(&of_surface, 80);
        assert_eq!(
            (counted(srgb), counted("rt-plane-views-format-80")),
            (before.0 + 1, before.1 + 1)
        );

        let plain = make(70);
        let _of_plain = view(&plain, 71);
        assert_eq!(counted(plain_srgb), plain_before);
    }

    /// On the GPU the test holds: the device's texture buffer (`newTextureWithDescriptor:` of
    /// `MTLTextureTypeTextureBuffer`), which rt makes only over a buffer, lies over a buffer of the
    /// platform's own ([`make_texture_buffer`]) of the descriptor's storage mode, Shared and
    /// Private, its bytes the row rounded up to rt's texture-buffer alignment, which the texture
    /// holds and lies over whole (one row, `bytesPerRow` long: on KosmicKrisp 1001 RGBA8 texels
    /// lie over 4016 bytes, m3-max-64400; where rt answers 4, as on RADV, ANV and NVIDIA, 4004);
    /// none is rt's own refusal (`rt-refused-ours`). A format rt makes no texture buffer of is
    /// nil, counted, and not under that code.
    #[test]
    fn the_device_s_texture_buffer_lies_over_a_buffer_of_its_own_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::raise::OnFault;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let ours = counted(RtRefusal::Ours.code());
        let rgba8 = rt::PixelFormat::RGBA8Unorm;
        let alignment = (device.rt().minimum_texture_buffer_alignment(rgba8))
            .expect("rt makes RGBA8 texture buffers");
        let make = |pixel_format, resource_options| {
            let desc = TextureDesc {
                texture_type: TYPE_TEXTURE_BUFFER,
                pixel_format,
                width: 1001,
                usage: 0x1,
                resource_options: Some(resource_options),
                ..TextureDesc::default()
            };
            let made = device.new_texture(OnFault::Return, &platform.texture_descriptor(&desc));
            made.expect("never a raise")
        };
        let row = 4004_u64.next_multiple_of(alignment);
        for storage in [0, STORAGE_PRIVATE] {
            let texture = make(70, storage).expect("a texture buffer");
            let answers = (
                texture.texture_type(),
                texture.width(),
                texture.storage_mode(),
            );
            assert_eq!(answers, (TYPE_TEXTURE_BUFFER, 1001, storage >> 4));
            let held = texture.held.get().expect("the buffer it lies over");
            assert_eq!(held.byte_length(), row);
            assert_eq!(held.resource_options() & STORAGE_BITS, storage);
            let (over, bytes) = texture.over().expect("the bytes it lies over");
            assert!(std::ptr::eq(over.as_ptr(), &**held));
            assert_eq!(*bytes, 0..row);
        }
        // BC1, compressed: rt makes no texture buffer of it, counted under its format's code.
        let bc1 = "rt-refused-texture-format-130";
        let refused = counted(bc1);
        assert!(make(130, 0).is_none());
        assert_eq!(counted(bc1), refused + 1);
        assert_eq!(counted(RtRefusal::Ours.code()), ours);
    }

    /// On the GPU the test holds: a refused view's detail names the view asked (its pixel format,
    /// type, levels and slices, as Metal numbers them, and whether it swizzles) and the texture it
    /// views, of rt's descriptor of it, its usage, storage mode and how it was made among the rest.
    #[test]
    fn a_refused_view_s_detail_names_the_texture_s_usage_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::raise::OnFault;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let desc = TextureDesc {
            pixel_format: 80,
            width: 8,
            height: 4,
            usage: 0x1,
            resource_options: Some(STORAGE_PRIVATE),
            ..TextureDesc::default()
        };
        let made = device.new_texture(OnFault::Return, &platform.texture_descriptor(&desc));
        let texture = made.expect("never a raise").expect("a texture");
        let what = ViewDetail {
            format: 81,
            view_type: 2,
            levels: 0..1,
            slices: 0..1,
            swizzled: true,
            of: &texture,
        };
        assert_eq!(
            what.to_string(),
            "a view of format 81, type 2, levels 0..1, slices 0..1, swizzled, of a texture of \
             format 80 8x4x1, 1 levels, array length 1, 1 samples, type 2, usage 0x1, storage 2, \
             ordinary"
        );
    }

    /// On the GPU the test holds: an image for views (a shape's image of the platform's own,
    /// `reims_vgpu`'s `ApiPlatform::shape_image`) reads zero before any write, through a blit and
    /// through a view at a guest's index that a kernel samples, though an image of its descriptor
    /// held other bytes just before it, read back so and let go of: the per-shape form takes such
    /// an image's fill from a buffer with no memory as made (`shapes`, "Zeros"), which a runtime
    /// that left a new image's bytes as its memory held them would break here, not in a guest
    #[test]
    fn a_texture_for_views_reads_zero_before_any_write_on_vulkan() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::resource::{Buffer as _, Library as _};
        use crate::core::platform::types::{MtlOrigin, MtlSize};
        use crate::core::platform::values::SamplerDesc;
        use crate::core::raise::OnFault;
        const SAMPLE: &[u8] = include_bytes!("../../../../tests/shaders/compute_sample.metallib");
        const SIDE: u64 = 64;
        const BYTES: u64 = SIDE * SIDE * 4;
        const INDEX: u64 = 4000;
        let on = OnFault::Return;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let queue = device.new_command_queue().expect("a command queue");
        let desc = TextureDesc {
            pixel_format: 80, // BGRA8Unorm
            width: SIDE,
            height: SIDE,
            usage: 0x5,                   // shader read, render target
            resource_options: Some(0x20), // Private
            ..TextureDesc::default()
        };
        let shared = |length: u64, byte: u8| {
            let made = device.new_buffer_of_length(on, length, 0);
            let made = made.expect("never a raise").expect("a buffer");
            // SAFETY: a Shared buffer of `length` bytes, which no command uses yet.
            unsafe {
                std::ptr::write_bytes(made.contents().as_ptr::<u8>(), byte, length.to_usize())
            };
            made
        };
        let blit = |cmd: BlitCmd<'_, RtPlatform>| {
            let cb = queue.command_buffer(on).expect("never a raise");
            let cb = cb.expect("a command buffer");
            let encoder = cb.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            assert_eq!(encoder.encode(on, cmd), Ok(()));
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            assert_eq!(cb.commit(on), Ok(()));
            assert_eq!(cb.wait_until_completed(on), Ok(()));
        };
        let read = |texture: &RtObj<RtTexture>| -> Vec<u8> {
            let out = shared(BYTES, 0x5a);
            blit(BlitCmd::CopyTextureToBuffer {
                source_texture: texture,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::default(),
                source_size: MtlSize::new(SIDE, SIDE, 1),
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: SIDE * 4,
                destination_bytes_per_image: BYTES,
            });
            // SAFETY: the Shared buffer's bytes, which no command writes any more.
            let bytes = unsafe {
                std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), BYTES.to_usize())
            };
            bytes.to_vec()
        };

        let dirty = device
            .new_texture_for_views(&desc)
            .expect("an image for views");
        let bytes = shared(BYTES, 0xa5);
        blit(BlitCmd::CopyBufferToTexture {
            source_buffer: &bytes,
            source_offset: 0,
            source_bytes_per_row: SIDE * 4,
            source_bytes_per_image: BYTES,
            source_size: MtlSize::new(SIDE, SIDE, 1),
            destination_texture: &dirty,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::default(),
        });
        assert!(
            read(&dirty).iter().all(|&b| b == 0xa5),
            "the write read back"
        );
        drop(dirty);

        let fresh = device
            .new_texture_for_views(&desc)
            .expect("an image for views");
        assert!(read(&fresh).iter().all(|&b| b == 0), "zero through a blit");

        assert_eq!(device.reserve_resource_indices(0, &[INDEX]), Ok(()));
        let view = fresh.new_texture_view_at_index(80, INDEX);
        let view = view
            .expect("never a raise")
            .expect("a view at the guest's index");
        let library = device.new_library(SAMPLE).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let function = library.new_function(on, "sample_k").expect("never a raise");
        let function = function.expect("a function of the table");
        let state = device.new_compute_pipeline_state_for_function(Some(&function));
        let state = state.expect("a compute state").expect("a compute state");
        let sampler = device.new_sampler_state(&SamplerDesc::default());
        let sampler = sampler.expect("never a raise").expect("a sampler state");
        let sampled = shared(16, 0xa5);
        let cb = queue.command_buffer(on).expect("never a raise");
        let cb = cb.expect("a command buffer");
        let encoder = cb.compute_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a compute encoder");
        let cmds = [
            ComputeCmd::SetComputePipelineState {
                state: Some(&state),
            },
            ComputeCmd::SetTextures {
                textures: &[Some(&view)],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::SetSamplerStates {
                samplers: &[Some(&sampler)],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&sampled)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ];
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(cb.commit(on), Ok(()));
        assert_eq!(cb.wait_until_completed(on), Ok(()));
        // SAFETY: the Shared buffer's 4 words, which no command writes any more.
        let words = unsafe { std::slice::from_raw_parts(sampled.contents().as_ptr::<u32>(), 4) };
        assert_eq!(words.to_vec(), vec![0u32; 4], "zero through a sample");
    }

    /// On the GPU the test holds, kept busy: a texture released while command buffers are in
    /// flight goes once the work committed before its release completes, while later work still
    /// runs. The test keeps `DEPTH` command buffers of buffer fills in flight, each completing,
    /// committing the next once the oldest completes, and releases `PER` textures no work reads
    /// after each commit, so the GPU never idles. At the load's end, rt keeps at most the
    /// textures released for the command buffers in flight and those of 12 command buffers more,
    /// for its threads' lag, far fewer than the load released; and once the work is done, its
    /// textures and releases waiting come back to where they started.
    #[test]
    fn released_textures_go_while_the_gpu_stays_busy_on_vulkan() {
        use std::time::Duration;

        use crate::core::num::ToU64 as _;
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::types::CommandBufferStatus;
        use crate::core::raise::OnFault;
        use crate::rt_platform::RtCommandBuffer;
        // Command buffers in flight, textures released after each commit, and commits.
        const DEPTH: usize = 3;
        const PER: usize = 50;
        const ROUNDS: usize = 60;
        // Each command buffer fills a Private buffer of `FILL` bytes `FILLS` times: a GiB.
        const FILL: u64 = 32 << 20;
        const FILLS: usize = 32;
        let on = OnFault::Return;
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let queue = device.new_command_queue().expect("a command queue");
        // rt's textures and releases waiting now.
        let gauges = || {
            let gauges = device.rt().counters().gauges();
            (gauges.textures, gauges.releases_waiting)
        };
        let filled = device.new_buffer_of_length(on, FILL, STORAGE_PRIVATE);
        let filled = filled.expect("never a raise").expect("a buffer");
        let desc = platform.texture_descriptor(&TextureDesc {
            width: 64,
            height: 64,
            resource_options: Some(STORAGE_PRIVATE),
            ..TextureDesc::default()
        });
        let start = gauges();
        let made = |_| {
            let made = device.new_texture(on, &desc).expect("never a raise");
            made.expect("a texture")
        };
        let mut textures: Vec<RtObj<RtTexture>> = (0..ROUNDS * PER).map(made).collect();
        // rt's textures a texture of ours makes.
        let each = (gauges().0 - start.0) / (ROUNDS * PER).to_u64();
        assert!(each >= 1, "rt counts the textures made");
        let waited = |commands: &RtObj<RtCommandBuffer>| {
            assert_eq!(commands.wait_until_completed(on), Ok(()));
            assert_eq!(commands.status(), CommandBufferStatus::COMPLETED);
        };
        let mut flight = VecDeque::new();
        // The most textures rt kept after the test released them, at a commit.
        let mut most = 0;
        for _ in 0..ROUNDS {
            let commands = queue.command_buffer(on).expect("never a raise");
            let commands = commands.expect("a command buffer");
            let encoder = commands.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            for _ in 0..FILLS {
                let fill = BlitCmd::FillBuffer {
                    buffer: &filled,
                    range: NsRange::new(0, FILL),
                    value: 0x5a,
                };
                assert_eq!(encoder.encode(on, fill), Ok(()));
            }
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            assert_eq!(commands.commit(on), Ok(()));
            flight.push_back(commands);
            textures.truncate(textures.len() - PER);
            if flight.len() == DEPTH {
                waited(&flight.pop_front().expect("the oldest in flight"));
            }
            let held = start.0 + each * textures.len().to_u64();
            let kept = gauges().0.saturating_sub(held);
            most = most.max(kept);
        }
        let end = gauges();
        let kept = end.0.saturating_sub(start.0);
        for commands in &flight {
            waited(commands);
        }
        // Every 10 ms, up to 1000 times (about 10 s).
        for _ in 0..1000 {
            if gauges() == start {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let after = gauges();
        let bound = each * ((DEPTH + 12) * PER).to_u64();
        assert!(
            kept <= bound,
            "rt kept {kept} of the {} textures released under the load, at most {bound} for the \
             work in flight: {} releases waiting, at most {most} kept at a commit; from {start:?} \
             (textures, releases waiting), {after:?} once the work was done",
            each * (ROUNDS * PER).to_u64(),
            end.1,
        );
        assert_eq!(after, start, "rt's gauges once the work was done");
    }
}
