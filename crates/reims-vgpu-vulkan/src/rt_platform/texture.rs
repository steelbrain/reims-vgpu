//! The rt platform's textures ([`RtTexture`]): rt's textures, made by the device's
//! `newTextureWithDescriptor:` ([`make`]) and by a texture's views, each an optimal image in
//! device-local memory (or a view of one) that the guest's CPU reaches only through
//! `replaceRegion:`; and textures over a buffer's bytes, made by the buffers' piece
//! (`super::buffer`) through [`make_over`], of the buffer's storage mode. Textures in heaps and
//! over IOSurfaces are their pieces'.
//!
//! **Indices.** rt places a texture that shaders reach (shader-read or shader-write usage, or
//! `Unknown`, which Metal takes as both) at an index of a live texture grant, and no other. A
//! texture the guest places (the descriptor's `resourceIndex`, a view's `resourceIndex:`) takes the
//! guest's index, from the guest's reservation (`reserveResourceIndicesForResourceType:`). One it
//! does not place takes one of the platform's own ([`OwnIndices`]): the highest index of the class
//! that no reservation holds and no texture of the platform's took yet, granted as it is taken,
//! and once those are gone the one given back longest ago. So an index a released texture held is
//! taken again as late as can be: rt leaves to the program that no work in flight reads a released
//! texture through its index, and Metal frees a texture's ID only once its command buffers
//! completed. A texture no shader reaches has no index. Its `gpuResourceID` is rt's, the index and
//! the translator's word of its format and swizzle, and its `resourceIndex` and `uniqueIdentifier`
//! the index: 0 for one with no index, where Metal numbers every texture from its lowest free ID.
//! Nothing of core or the decoder reads them but the check of the guest's reservation.
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

use metal2vulkan_rt as rt;

use crate::core::num::{ToU64 as _, ToUsize as _};
use crate::core::platform::Handle as _;
use crate::core::platform::Platform;
use crate::core::platform::host_range::HostRange;
use crate::core::platform::resource::{GpuResource, Texture};
use crate::core::platform::types::{MtlRegion, NsRange, Swizzle};
use crate::core::platform::values::TextureDesc;
use crate::core::raise::RaiseResult;
use crate::core::sync::Mutex;

use super::descriptor::rt_texture_descriptor;
use super::device::RtDevice;
use super::unbuilt::{
    RtRefusal, Unbuilt, refuse_create, refused_as, rt_refused, rt_refused_texture_create,
};
use super::{RtObj, RtPlatform};

/// The code of a `replaceRegion:` whose bytes the platform copied first (the module's docs),
/// counted per call.
pub const COPY_CODE: &str = "rt-copy-replace-region";

/// The code of the bytes [`COPY_CODE`]'s calls copied, counted by the byte.
pub const COPY_BYTES_CODE: &str = "rt-copy-replace-region-bytes";

/// `replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:`.
const REPLACE: &str = "replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:";

/// `MTLTextureUsageShaderRead | ShaderWrite | RenderTarget | PixelFormatView`: the usage a texture
/// made of `Unknown` usage answers (measured, `crate::gpu::desc`'s docs).
const UNKNOWN_ANSWERS: u64 = 0x17;

/// The usage bit Metal's descriptor takes and its texture drops.
const DROPPED: u64 = 0x8;

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

/// `MTLPurgeableStateKeepCurrent`.
const KEEP_CURRENT: u64 = 1;

/// `MTLPurgeableStateNonVolatile`, a new texture's state.
pub(super) const NON_VOLATILE: u64 = 2;

/// `MTLPurgeableStateEmpty`, the last state.
const EMPTY: u64 = 4;

/// `id<MTLTexture>` on metal2vulkan-rt: rt's texture and what the platform answers of it beyond
/// rt's descriptor.
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
}

impl Drop for RtTexture {
    fn drop(&mut self) {
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
            .index_for(sel, reaches(d.usage), resource_index)?;
        let made = match channels {
            None => self
                .rt
                .new_texture_view(format, view_type, levels, slices, index),
            Some(channels) => (self.rt)
                .new_texture_view_with_swizzle(format, view_type, levels, slices, channels, index),
        };
        let usage = match swizzle {
            Some(s) if s != DEFAULT_SWIZZLE => self.usage & SWIZZLED_VIEW_KEEPS,
            _ => self.usage,
        };
        self.device.made(sel, made, own, |texture| RtTexture {
            rt: texture,
            device: self.device.clone_ref(),
            options: self.options,
            usage,
            protection_options: self.protection_options,
            own,
            purgeable: purgeable_state(),
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
    made_of(device, sel, desc, options, make)
}

/// A texture of `desc` over a buffer's or a heap's memory of the resource options `held`, for
/// `sel`: the texture `over` makes of rt's descriptor at the index [`make`] would give it (rt's
/// `Buffer::new_texture`, `Device::new_heap_texture`), answering the descriptor's options with
/// `held`'s bits `taken` (a buffer's storage mode, as rt makes the texture of the buffer's whatever
/// the descriptor's; a heap's modes, which its resources take). Nil where rt refuses it, counted.
pub(super) fn make_over(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    desc: &TextureDesc,
    (held, taken): (u64, u64),
    over: impl FnOnce(&rt::TextureDescriptor, Option<u32>) -> Result<rt::Texture, rt::Error>,
) -> Option<RtObj<RtTexture>> {
    let options = desc.resource_options.unwrap_or(0);
    let options = (options & !taken) | (held & taken);
    made_of(device, sel, desc, options, over)
}

/// The texture `make` of `desc` makes for `sel` (the module's docs), of the resource options
/// `options`, hazard tracking yet to resolve.
fn made_of(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    desc: &TextureDesc,
    options: u64,
    make: impl FnOnce(&rt::TextureDescriptor, Option<u32>) -> Result<rt::Texture, rt::Error>,
) -> Option<RtObj<RtTexture>> {
    let d = match rt_texture_descriptor(desc) {
        Ok(d) => d,
        Err(why) => return refused_metal(sel, &why),
    };
    let placed = desc.resource_index.map_or(0, |(index, _forced)| index);
    let (index, own) = device.index_for(sel, reaches(d.usage), placed)?;
    let made = make(&d, index);
    let usage = match desc.usage {
        0 => UNKNOWN_ANSWERS,
        usage => usage & !DROPPED,
    };
    device.made(sel, made, own, |texture| RtTexture {
        rt: texture,
        device: device.clone_ref(),
        options: tracked(options),
        usage,
        protection_options: desc.protection_options.unwrap_or(0),
        own,
        purgeable: purgeable_state(),
    })
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

impl RtObj<RtDevice> {
    /// The index a texture `sel` makes takes, as rt takes it, and whether it is one of the
    /// platform's own: none where shaders do not reach it (`reached`), the guest's `placed` where
    /// it places one (0: none), else one of the platform's own. `None` where none can be had,
    /// counted.
    fn index_for(
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

/// Whether shaders reach a texture of `usage`, as rt decides where it takes an index: shader-read
/// or shader-write usage, or `Unknown`, which Metal takes as both.
fn reaches(usage: rt::TextureUsage) -> bool {
    let shaders = rt::TextureUsage::SHADER_READ | rt::TextureUsage::SHADER_WRITE;
    usage.is_empty() || usage.intersects(shaders)
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
/// (`super::state`): those the guest's reservations hold are never taken.
pub struct OwnIndices {
    /// Their class.
    class: rt::IndexClass,
    /// What was taken and reserved.
    own: Mutex<Own>,
}

/// [`OwnIndices`]' state.
struct Own {
    /// The indices a reservation holds, a bit each.
    reserved: Vec<u64>,
    /// The class's first index.
    first: u32,
    /// The next index not taken yet, counting down; none once every one was taken.
    fresh: Option<u32>,
    /// The indices given back, the longest ago first.
    free: VecDeque<u32>,
}

impl OwnIndices {
    /// None of `class` taken, none reserved.
    pub fn new(class: rt::IndexClass) -> OwnIndices {
        OwnIndices {
            class,
            own: Mutex::new(Own::new(class.range())),
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
    /// where every index is reserved or held.
    pub fn take(&self, device: &rt::Device) -> Result<Option<u32>, rt::Error> {
        let mut own = self.own.lock();
        let Some((index, fresh)) = own.next() else {
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
        Ok(Some(index))
    }

    /// Gives `index` back, to be taken again once those given back before it are.
    pub fn give_back(&self, index: u32) {
        self.own.lock().give_back(index);
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
    /// taken yet and not reserved, else the one given back longest ago that is not reserved since.
    fn next(&mut self) -> Option<(u32, bool)> {
        while let Some(index) = self.fresh {
            self.fresh = (index.checked_sub(1)).filter(|&next| next >= self.first);
            if !self.is_reserved(index) {
                return Some((index, true));
            }
        }
        while let Some(index) = self.free.pop_front() {
            if !self.is_reserved(index) {
                return Some((index, false));
            }
        }
        None
    }

    /// Gives `index` back, but one reserved since it was taken.
    fn give_back(&mut self, index: u32) {
        if !self.is_reserved(index) {
            self.free.push_back(index);
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
        Ok(set_purgeable(&self.purgeable, state))
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

    /// Nothing: a texture over a buffer reads the buffer's bytes, whose changes a Managed buffer's
    /// `didModifyRange:` names to rt; this SPI tells of a tiled texture's, which the platform does
    /// not make ([`Unbuilt::Buffer`]).
    fn did_modify_data(&self) {}

    /// rt's: its index and the translator's word of its format and swizzle.
    fn gpu_resource_id(&self) -> u64 {
        self.rt.gpu_resource_id().to_bits()
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
        refuse_create(Unbuilt::SharedTexture, "newSharedTextureHandle");
        Ok(None)
    }

    fn new_texture_view(&self, pixel_format: u64) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let sel = "newTextureViewWithPixelFormat:";
        Ok(self.view(sel, pixel_format, None, None, None, None, 0))
    }

    fn new_texture_view_at_index(
        &self,
        pixel_format: u64,
        resource_index: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let sel = "newTextureViewWithPixelFormat:resourceIndex:";
        Ok(self.view(sel, pixel_format, None, None, None, None, resource_index))
    }

    fn new_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
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
    }

    fn new_swizzled_texture_subview(
        &self,
        pixel_format: u64,
        texture_type: u64,
        level_range: NsRange,
        slice_range: NsRange,
        swizzle: Swizzle,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
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
        let sel = "newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex:";
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
    }

    fn new_texture_subview_at_index(
        &self,
        pixel_format: u64,
        texture_type: u64,
        levels: NsRange,
        slices: NsRange,
        resource_index: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
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
    /// it) changes nothing, counted, where Metal's validation aborts.
    fn replace_region(
        &self,
        region: MtlRegion,
        level: u64,
        slice: u64,
        pixel_bytes: HostRange<'_>,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<()> {
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
        if let Err(error) =
            (self.rt).replace_region(r, level, slice, &bytes, bytes_per_row, bytes_per_image)
        {
            rt_refused(REPLACE, &error);
        }
        Ok(())
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

/// Counts a `replaceRegion:` copy of `bytes` bytes ([`COPY_CODE`], [`COPY_BYTES_CODE`]); the
/// process's first writes its line.
fn counted_copy(bytes: usize) {
    crate::answer::count_kind(COPY_CODE);
    crate::answer::count_kind_by(COPY_BYTES_CODE, u64::try_from(bytes).unwrap_or(u64::MAX));
    if !COPIED.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: "reims_vgpu_vulkan::rt_platform",
            "{REPLACE}: the guest's bytes are copied before metal2vulkan-rt takes them, a copy \
             Metal does not make ({COPY_CODE}), counted from here"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The platform's own indices: the highest first, none reserved, each fresh one once, then
    /// those given back in the order they were, never one reserved since.
    #[test]
    fn own_indices_go_down_and_come_back_in_order() {
        let mut own = Own::new(1..8);
        own.reserve(7);
        own.reserve(5);
        own.reserve(64);
        assert_eq!(own.next(), Some((6, true)));
        assert_eq!(own.next(), Some((4, true)));
        own.give_back(6);
        own.give_back(4);
        own.give_back(5);
        assert_eq!(
            [own.next(), own.next(), own.next()],
            [Some((3, true)), Some((2, true)), Some((1, true))]
        );
        own.reserve(6);
        assert_eq!(own.next(), Some((4, false)));
        assert_eq!(own.next(), None);
        own.give_back(2);
        assert_eq!(own.next(), Some((2, false)));
    }

    /// The texture class's own indices start at its last, 65,535.
    #[test]
    fn own_indices_start_at_the_class_s_last() {
        let own = OwnIndices::new(rt::IndexClass::Texture);
        own.reserve(&[65535, 1 << 40]);
        assert_eq!(own.own.lock().next(), Some((65534, true)));
    }

    /// A texture reaches shaders by shader-read or shader-write usage, or `Unknown`.
    #[test]
    fn shaders_reach_read_write_and_unknown_usage() {
        let usage = |raw: u32| rt::TextureUsage::try_from(raw).unwrap();
        assert!(reaches(usage(0)));
        assert!(reaches(usage(1)) && reaches(usage(2)) && reaches(usage(0x15)));
        assert!(!reaches(usage(4)) && !reaches(usage(0x10)) && !reaches(usage(0x8)));
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
        assert!(rt_usage(0x8).is_ok_and(reaches));
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
    /// indices, distinct and from the top down past the guest's reservation, a released one's
    /// index not taken again before the fresh ones; one the guest places answers its index; one
    /// no shader reaches has none, the guest's placement dropped; `Unknown` usage answers Metal's
    /// 0x17 and 0x8 is dropped; a texture of 0x8 alone answers no usage and is rt's of
    /// `ShaderRead`, with an index, a format view of it answering no usage and a swizzled one
    /// refused and counted; views of another format, at a placed index, of one slice of a 2D
    /// array as a 2D texture and through a swizzle; `replaceRegion:` of a Shared texture copied and
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
}
