//! `id<MTLBlitCommandEncoder>`: core's `BlitEncoder` over the runtime's, one runtime call per
//! `BlitCmd`.
//!
//! - **A runtime refusal** of any command, `endEncoding` too, dies at its selector naming the
//!   runtime's error (`Policy::Dies`): Metal raises at none of them. A copy between a buffer and
//!   a multisampled texture, which Metal copies per sample, the pixel's samples consecutive in
//!   the buffer, and no Vulkan copy does, the runtime runs by Metal's layout (a depth or stencil
//!   texture's through the aspect's bits) and refuses only where the
//!   device cannot run it (`RuntimeGap::MultisampledBufferCopy`): without sample-rate shading,
//!   an integer multisample attachment at the count or storage for the format, and for a
//!   `Depth16Unorm` texture where the device's maintenance8 copies are missing or wrong
//!   (MoltenVK 1.4.1 among them). Such a refusal is one, its line naming the copy's shape
//!   (`buffer_copy_shape`).
//! - **`fillBuffer:range:pattern4:`** is the runtime's `fill_buffer_pattern4`, which writes the
//!   pattern as Metal does ([`crate::gpu::fill`]'s docs; `pattern_fills_are_the_metal_platforms`,
//!   over more than the runtime stages at once).
//! - **The private texture fills** (the two `fillTexture:...`) are the runtime's: a bytes fill
//!   its `fill_texture` of the pattern, a colour fill its colour fill, which fills the texel it
//!   packs as `fill_texture` fills a pattern ([`crate::gpu::fill`]), each copies from a staging
//!   buffer outside the Metal address space, at the level and slice and in the region
//!   [`crate::gpu::fill::place`] gives (a texture buffer's as the call gives them, the bytes
//!   of its buffer written from the region's `x` whatever the level, slice and row, the pattern
//!   running across them past the width, `texture_buffer_fills_are_the_metal_platforms` and
//!   `texture_buffer_fills_metal_streams_are_metals`); a fill that writes nothing encodes
//!   nothing. They write
//!   what Metal's fill writes on this host (`texture_fills_are_the_metal_platforms`, of colour
//!   and depth formats, levels, slices, cube faces, 3D depths, cut regions and a fill larger than
//!   the runtime stages at once) and:
//!   - a compressed format's blocks, each the pattern cut at its bytes or repeated to them, the
//!     region counting blocks, which the runtime cuts at the level's blocks (the cut `place`
//!     makes, at the level's texels, is no tighter): BC, 4 by 4 ASTC, ETC2 and EAC
//!     (`fills_the_runtime_writes_are_the_metal_platforms`), PVRTC at the suite's shape, of a
//!     shareable texture (`pvrtc_fills_are_the_metal_platforms`), and ASTC of HDR blocks and
//!     of other blocks (`astc_hdr_fills_are_the_metal_platforms`,
//!     `astc_fills_of_other_blocks_are_the_metal_platforms`);
//!   - a 4:2:2 format's texels, 2 bytes each (`GBGR422`,
//!     `fills_the_runtime_writes_are_the_metal_platforms`, and `BGRG422`,
//!     `bgrg422_fills_are_the_metal_platforms`);
//!   - a stencil format's texels, 1 byte each (`Stencil8`,
//!     `fills_the_runtime_writes_are_the_metal_platforms`, and `X32_Stencil8`'s stencil,
//!     `x32_stencil8_fills_are_the_metal_platforms`); `Depth32Float_Stencil8`'s depth, its
//!     stencil kept (`depth32_stencil8_fills_keep_the_stencil_as_metals`);
//!   - a two-plane XR10 texture's (550, 551) colour words in plane 0, the alpha bytes kept: over
//!     a surface's planes at Metal's blit rows
//!     (`a_filled_two_plane_xr10_texture_over_a_surface_writes_as_metals`); in a placement heap's
//!     bytes, of one level, at Metal's blit rows, which overlap where the width is no multiple of
//!     4 (`two_plane_fills_in_a_heap_are_the_metal_platforms`); and of its own memory, of one
//!     level a multiple of 4 wide, at its rows
//!     (`two_plane_xr10_fills_off_a_surface_are_the_metal_platforms`, the alpha kept
//!     `two_plane_fills_off_a_surface_keep_the_alpha_as_metals`).
//!
//!   It fills every texture type [`crate::gpu::fill::place`] measured, a compressed cube and
//!   cube array too (`compressed_cube_fills_are_the_metal_platforms`). The runtime fills the
//!   rest of the formats Metal fills too: a two-plane XR10 texture of its own memory at any width
//!   and level count, and a heap's of more than one level or slice, at level 0 of the slice's
//!   words, rows `max(16, ⌊4 w / 16⌋ · 16)` bytes apart; the YCbCr and padded formats of a
//!   texture over a surface or memory (`fills_of_ycbcr_and_padded_textures_are_metals`); and a
//!   multisample texture's samples (`multisample_fills_are_the_metal_platforms`). It refuses a
//!   multisample texture's fill only where the device cannot run its sample copies (sample-rate
//!   shading, an integer attachment at the sample count), which dies naming the refusal.
//! - **A colour fill** is the runtime's colour fill, which packs the colour in the call's pixel
//!   format as Metal does ([`crate::gpu::fill`]) and fills that texel as its `fill_texture` fills
//!   a pattern (above), at the level and slice and in the region `place` gives. A colour in a
//!   format of [`crate::gpu::fill::OWN_FORMAT_PACKED`] (a bfloat format, an extended-range
//!   `_sRGB` twin, an RGBX format) into a texture of another format is filled, relayed, and dies
//!   in strict mode (`colours_the_runtime_packs_are_the_metal_platforms`): a twin's bytes there
//!   follow no rule measured on Metal, and a bfloat or RGBX colour's were not compared, where the
//!   runtime repeats the packed texel. A colour in a format Metal packs none in fills the bytes
//!   of the command buffer's last packed texel, as Metal's do ([`crate::gpu::fill`]).
//! - **The indirect command buffer blits** copy and reset the commands' records as the command
//!   buffer runs, and what Metal answers of the commands beside their records
//!   ([`crate::gpu::icb`]) once it completed; a range past a buffer's size, which Metal runs as
//!   a corruption without validation, dies. A copy past the destination's end, which core lets
//!   through (it checks the source's range and the destination index alone), is such a
//!   corruption, measured on this host: Metal writes the commands that fit, and the rest over
//!   the records of the destination's other commands, in its own layout, so their types and
//!   kernel buffer addresses change (2 commands copied to index 1 of a buffer of 2 turn its
//!   command 0 from a threads dispatch to a threadgroups one). What the overflow reaches past
//!   the destination is not measured, so it dies at the copy.
//! - **What the runtime has no form for dies**: a nil resource (but `synchronizeResource:`'s), a
//!   blit option other than none, depth or stencil, a texture copy's options other than none, a
//!   command after `endEncoding`. A blit of a rotated texture is the runtime's, which addresses
//!   its plane as Metal's does, or refuses it ([`crate::gpu::resource::TextureObject::texels`]).
//! - **A copy between two textures of the two-plane XR10 formats** (550, 551) moves both planes'
//!   bytes as Metal's blit does on this host, at Metal's rows, from either format into either:
//!   the whole level by a region and by `copyFromTexture:sourceSlice:sourceLevel:toTexture:...`,
//!   and regions of any origin and width, between textures of one width and of two, whose widths
//!   are a multiple of 16, of 4 or of neither, of the default storage, `Private`, and a placement
//!   heap's (measured on the M3 Max: `two_plane_xr10_blits_are_the_metal_platforms`,
//!   `two_plane_xr10_copies_of_any_region_are_the_metal_platforms`). Their copies to and from a
//!   buffer move plane 0's colour words, as Metal's do on this host
//!   (`two_plane_xr10_copies_are_the_metal_platforms`). Where the runtime's answer is not Metal's
//!   measured one, it announces so, which is relayed, a death in strict mode (`TwoPlaneTextureCopy`
//!   and `TwoPlaneBufferBlit`, [`crate::answer::DIES`]): a copy within one texture, of a texture of
//!   more than one level or slice, of rows that overlap in the destination, of a texture narrower
//!   than 4 texels, or reading bytes outside the source's texels that an earlier blit wrote or a
//!   heap's other resources hold, and a copy from a buffer into a level 32 texels wide or wider
//!   whose width is no multiple of 4 whose rows meet at Metal's addresses, or out of such a level
//!   after a copy from another texture wrote the bytes between them
//!   (`two_plane_xr10_blits_metal_was_not_measured_at_die`); where a narrower level's rows meet
//!   between Metal's groups of 32 rows, the earlier row's bytes stand, one of Metal's outcomes
//!   (`two-plane-buffer-blit-order`, [`crate::answer`]). A copy into a
//!   heap's texture writes the heap's bytes at Metal's blit rows, as Metal's blit does, the bytes
//!   outside its texels and past its alpha plane over the heap's other resources included
//!   (`two_plane_copies_into_a_heap_are_the_metal_platforms`).
//! - **Bytes past a buffer's end** in a copy between it and a texture: the runtime reads a
//!   no-copy buffer's to the end of its pages, as Metal reads them, and refuses another buffer's,
//!   which dies, where Metal copies bytes no one wrote.
//! - **`synchronizeResource:` of nil** does nothing, which Metal takes
//!   (`blit_texture_maintenance`) and the runtime has no form for.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder, ResourceRef};
use reims_vgpu_core::platform::icb::IndirectCommandBuffer as _;
use reims_vgpu_core::platform::types::{ClearColor, MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::{Mutex, ReentrantMutex};

use crate::answer::{
    Policy, decide, metal_aborts, plain, send, send_describing, unrepresentable, unstopped_at,
};
use crate::gpu::command::{CommandBufferObject, Failing, Progress, VulkanCommandBuffer};
use crate::gpu::fill;
use crate::gpu::metal_space::{self, FirstUse};
use crate::gpu::resource::{VulkanBuffer, VulkanTexture};
use crate::gpu::{origin, range, size};
use crate::platform::{Obj, VulkanPlatform};

/// An open runtime encoder: `None` once `endEncoding` consumed it.
#[derive(Debug)]
pub struct EncoderObject<E> {
    /// The runtime's encoder, until it ends. Only the thread that encodes into it takes it,
    /// once a command ([`Self::with`]), or once for a run of commands ([`Self::running`]),
    /// inside which each command's take is the same thread's again, which no atomic operation
    /// takes. A command's call holds it borrowed, and makes no command of its own encoder.
    pub raw: ReentrantMutex<RefCell<Option<E>>>,
    /// Its command buffer's count of open encoders, which counts it until it ends or goes.
    open: Arc<AtomicUsize>,
    /// Its command buffer, held weakly: once it goes, nothing encoded here runs.
    command_buffer: Weak<CommandBufferObject>,
    /// Whether the pipeline state a compute or render encoder set last can encode indirect
    /// commands (the runtime's `can_encode_indirect_commands`).
    pub(crate) encodes_icbs: AtomicBool,
    /// The indirect command buffers a compute or render encoder made resident for writing, whose
    /// records work of a pipeline state that can encode indirect commands may write
    /// ([`crate::gpu::command::executions::gpu_writes`]).
    pub(crate) writable_icbs: Mutex<Vec<crate::gpu::icb::VulkanIcb>>,
    /// The `uniqueIdentifier` of the pipeline state a compute or render encoder set last, 0 for
    /// none, which a slow command's debug line names ([`crate::gpu::timing::encoder_call`]).
    pub(crate) pipeline: AtomicU64,
    /// Its place among the encoders its command buffer's runtime ends
    /// ([`CommandBufferObject::made_encoder`]).
    at: usize,
    /// The protection options it was made under where its first command ends the process: a
    /// blit encoder's under 1, 3 or 7 ([`CommandBufferObject::made_encoder`]).
    aborts: Option<u64>,
}

impl<E> EncoderObject<E> {
    /// Its command buffer, while it lives.
    pub(crate) fn command_buffer(&self) -> Option<VulkanCommandBuffer> {
        Obj::upgrade(&self.command_buffer)
    }

    /// Whether its command buffer lives, as [`Self::command_buffer`] answers, without taking a
    /// reference to it.
    pub(crate) fn has_command_buffer(&self) -> bool {
        self.command_buffer.strong_count() != 0
    }

    /// An open encoder of `command_buffer`, which counts it among its open encoders and the
    /// encoders it made, under its protection options.
    pub fn new(raw: E, command_buffer: &VulkanCommandBuffer) -> EncoderObject<E> {
        let open = &command_buffer.open_encoders;
        open.fetch_add(1, Ordering::SeqCst);
        let (at, aborts) = command_buffer.made_encoder(true);
        EncoderObject {
            raw: ReentrantMutex::new(RefCell::new(Some(raw))),
            open: Arc::clone(open),
            command_buffer: command_buffer.downgrade(),
            encodes_icbs: AtomicBool::new(false),
            writable_icbs: Mutex::new(Vec::new()),
            pipeline: AtomicU64::new(0),
            at,
            aborts,
        }
    }

    /// Tells its command buffer that a draw of primitive type 9 in it fails it with `code`, as
    /// the runtime does, its event signals firing
    /// ([`CommandBufferObject::fails_at_type_9`]); nothing once core let go of the command
    /// buffer, which it can then commit no more.
    pub(crate) fn fails_at_type_9(&self, code: u64) {
        if let Some(cb) = Obj::upgrade(&self.command_buffer) {
            cb.fails_at_type_9(Failing {
                code,
                from: self.at,
                signals: true,
            });
        }
    }

    /// Tells its command buffer that this encoder, a render pass, stores an attachment into
    /// read-only memory the runtime copies, at which Metal's GPU faults it
    /// ([`CommandBufferObject::faults_at_read_only_store`]); nothing once core let go of the
    /// command buffer, which it can then commit no more.
    pub(crate) fn faults_at_read_only_store(&self) {
        if let Some(cb) = Obj::upgrade(&self.command_buffer) {
            cb.faults_at_read_only_store(self.at);
        }
    }

    /// Its command buffer's progress, for `sel` ([`CommandBufferObject::progress`]); none once
    /// core let go of the command buffer, which it can then commit no more.
    pub(crate) fn progress(&self, sel: &'static str) -> Option<Arc<Progress>> {
        Obj::upgrade(&self.command_buffer).map(|cb| cb.progress(sel))
    }

    /// `object`, which a command `sel` just encoded uses, is held until the command buffer has
    /// run ([`Progress::hold`]), as Metal's command buffer retains what its commands use.
    pub(crate) fn hold(&self, sel: &'static str, object: impl std::any::Any + Send + Sync) {
        if let Some(progress) = self.progress(sel) {
            progress.hold(object);
        }
    }

    /// Runs `call` on the open encoder, for `sel`: a command after `endEncoding` dies. A slow
    /// call is a debug line ([`crate::gpu::timing::encoder_call`]).
    pub fn with<T>(&self, sel: &'static str, call: impl FnOnce(&mut E) -> T) -> T {
        let held = self.raw.lock();
        let mut raw = held.borrow_mut();
        let e = raw
            .as_mut()
            .unwrap_or_else(|| unrepresentable(sel, "a command after endEncoding"));
        crate::gpu::timing::encoder_call(sel, &self.pipeline, || call(e))
    }

    /// Runs `run`, a run of commands this thread encodes one after another, holding the encoder
    /// for it: each command's take inside it is this thread's again ([`Self::raw`]). It encodes
    /// nothing itself, and its commands run as each would alone, in their order.
    pub fn running<T>(&self, run: impl FnOnce() -> T) -> T {
        let _held = self.raw.lock();
        run()
    }

    /// Encodes nothing for `sel`, a command Metal encodes nothing for; after `endEncoding` it
    /// dies, as every command does.
    pub fn nothing(&self, sel: &'static str) -> RaiseResult<()> {
        plain(|| self.with(sel, |_| ()));
        Ok(())
    }

    /// The open encoder, taken for `endEncoding`: a second one dies.
    pub fn take(&self, sel: &'static str) -> E {
        let e = self
            .raw
            .lock()
            .borrow_mut()
            .take()
            .unwrap_or_else(|| unrepresentable(sel, "a second endEncoding"));
        self.open.fetch_sub(1, Ordering::SeqCst);
        e
    }

    /// `endEncoding`, `end` the runtime's, whose refusal dies at `sel`, as Metal raises at none,
    /// as does a degradation of [`crate::answer::DIES`] it announces. The task's end is checked
    /// once, before the encoder is taken: a thread of an ended task stops with the encoder in
    /// place, untouched, as Metal's is left (`stop_if_task_ended`), and none stops while it holds
    /// the encoder taken, whose drop would abandon its command buffer.
    pub fn end(
        &self,
        sel: &'static str,
        end: impl FnOnce(E) -> Result<(), rt::Error>,
    ) -> RaiseResult<()> {
        let e = plain(|| self.take(sel));
        unstopped_at(sel, || end(e)).or_else(|err| decide(sel, Policy::Dies, err))
    }
}

impl<E> Drop for EncoderObject<E> {
    /// An encoder that goes without `endEncoding` is open no more: the runtime abandons it.
    fn drop(&mut self) {
        if self.raw.get_mut().get_mut().is_some() {
            self.open.fetch_sub(1, Ordering::SeqCst);
        }
    }
}

/// `id<MTLBlitCommandEncoder>`.
///
/// The runtime's encoder is boxed: its handle is about 1.2 KiB, which each move between its making
/// and its end would copy.
pub type VulkanBlitEncoder = Obj<EncoderObject<Box<rt::BlitCommandEncoder>>>;

/// What the encoder's lines name [`BlitEncoder::copy_from_guest_ram`] by: no selector of Metal's
/// does.
#[cfg(target_os = "linux")]
const GUEST_RAM_TO_BUFFER: &str = "copyFromGuestRam:toBuffer:destinationOffset: (no Metal)";

/// What they name [`BlitEncoder::copy_to_guest_ram`] by.
#[cfg(target_os = "linux")]
const BUFFER_TO_GUEST_RAM: &str = "copyFromBuffer:sourceOffset:toGuestRam: (no Metal)";

/// A raw `MTLBlitOption` of a copy between a buffer and a texture, sent with `sel`: none, depth
/// or stencil of a combined texture, or a PVRTC texture's blocks row by row. The runtime copies
/// the depth of a combined depth/stencil texture for none, as Metal does (measured: a
/// Depth32Float_Stencil8 texture's depth, 4 bytes a texel, both ways), and a PVRTC texture's
/// blocks row by row as Metal does with `MTLBlitOptionRowLinearPVRTC`; another combination of
/// options, which nothing measured, dies.
fn blit_option(sel: &'static str, raw: u64) -> rt::BlitOption {
    match raw {
        0 => rt::BlitOption::None,
        1 => rt::BlitOption::DepthFromDepthStencil,
        2 => rt::BlitOption::StencilFromDepthStencil,
        4 => rt::BlitOption::RowLinearPVRTC,
        _ => unrepresentable(sel, &format!("blit option {raw:#x}")),
    }
}

/// `object`, which core may pass as nil, for `sel`: the runtime takes no nil one.
fn some<'a, T>(sel: &'static str, object: Option<&'a T>, what: &str) -> &'a T {
    object.unwrap_or_else(|| unrepresentable(sel, &format!("a nil {what}")))
}

/// The indices `r` names, for `sel`; a range whose end passes `u64::MAX` dies.
fn indices(
    sel: &'static str,
    r: reims_vgpu_core::platform::types::NsRange,
) -> std::ops::Range<u64> {
    range(r).unwrap_or_else(|| unrepresentable(sel, &format!("the range {r:?}")))
}

impl VulkanBlitEncoder {
    /// A fill of `region` of `texture` at `level` and `slice` with the pattern `bytes`, for
    /// `sel`: the runtime's fill, where [`fill_place`] says, which cuts the pattern at its unit
    /// or repeats it to it, or runs it across a texture buffer's bytes (the module's docs), of the
    /// pattern's bytes it reads ([`crate::gpu::resource::fill_bytes`]); a texture buffer's of a
    /// pattern past the bytes Metal fills dies, as Metal's process ends there
    /// ([`fill::streamed_pattern`]). An empty pattern (a length of 0 at a readable address, which
    /// nothing measured on Metal) fills zeroes.
    fn fill_texture(
        &self,
        sel: &'static str,
        texture: &VulkanTexture,
        (level, slice, region): (u64, u64, MtlRegion),
        bytes: reims_vgpu_core::platform::HostRange<'_>,
    ) -> RaiseResult<()> {
        let raw = texture.copied(sel);
        let shape = plain(|| fill::Shape {
            texture_type: raw.texture_type(),
            extent: [raw.width(), raw.height(), raw.depth()],
            levels: raw.mipmap_level_count(),
            array_length: raw.array_length(),
        });
        let streamed = shape.texture_type == metal2vulkan::metal::TextureType::TextureBuffer;
        let most = match streamed {
            true => fill::MOST_STREAMED_PATTERN_BYTES,
            false => fill::MOST_PATTERN_BYTES,
        };
        let pattern = crate::gpu::resource::fill_bytes(sel, region, bytes, most);
        if streamed {
            fill::streamed_pattern(sel, bytes.len());
        }
        let pattern = if pattern.is_empty() {
            &[0][..]
        } else {
            &pattern[..]
        };
        let Some((level, slice, region)) = fill_place(sel, &shape, (level, slice, region)) else {
            return self.nothing(sel);
        };
        send(sel, Policy::Dies, || {
            self.with(sel, |e| e.fill_texture(raw, level, slice, region, pattern))
        })
    }
}

impl VulkanBlitEncoder {
    /// A colour fill of `region` of `texture` at `level` and `slice` with `color` packed in
    /// `format`, one of [`fill::colour_format`]'s, for `sel`: the runtime's colour fill, which
    /// packs it, where [`fill_place`] says. A colour in a format of [`fill::OWN_FORMAT_PACKED`]
    /// into a texture of another format is filled, the runtime repeating the packed texel,
    /// relayed, and dies in strict mode ([`crate::answer::Inexact::FillFormat`]): an
    /// extended-range `_sRGB` twin's bytes there follow no rule measured on Metal (into an
    /// `RGBA32Uint` texel, a colour filled alone repeats; colours filled one after another in one
    /// encoder each left their packed bytes, then `01` and zeroes), and a bfloat or RGBX colour's
    /// were not compared with Metal's here, until a comparison across formats holds them.
    fn fill_texture_color(
        &self,
        sel: &'static str,
        texture: &VulkanTexture,
        (level, slice, region): (u64, u64, MtlRegion),
        color: ClearColor,
        format: PixelFormat,
    ) -> RaiseResult<()> {
        let raw = texture.copied(sel);
        let (own, shape) = plain(|| {
            let shape = fill::Shape {
                texture_type: raw.texture_type(),
                extent: [raw.width(), raw.height(), raw.depth()],
                levels: raw.mipmap_level_count(),
                array_length: raw.array_length(),
            };
            (raw.pixel_format(), shape)
        });
        if own != format && fill::OWN_FORMAT_PACKED.contains(&format) {
            let twin = matches!(format, PixelFormat::BGRA10XRSRGB | PixelFormat::BGR10XRSRGB);
            let why = match twin {
                true => "follow no rule measured",
                false => "were not compared with the runtime's",
            };
            crate::answer::inexact(sel, crate::answer::Inexact::FillFormat, || {
                format!(
                    "a colour in pixel format {format:?} into a texture of {own:?}, whose bytes \
                     on Metal {why}"
                )
            });
        }
        let Some((level, slice, region)) = fill_place(sel, &shape, (level, slice, region)) else {
            return self.nothing(sel);
        };
        let color = rt::ClearColor {
            red: color.red,
            green: color.green,
            blue: color.blue,
            alpha: color.alpha,
        };
        send(sel, Policy::Dies, || {
            self.with(sel, |e| {
                e.fill_texture_color(raw, level, slice, region, color, Some(format))
            })
        })
    }
}

/// The level, slice and region a fill of `region` at `level` and `slice` of a texture of `shape`
/// gives the runtime's fill, for `sel`: [`fill::place`]'s; a texture buffer's as they are, which
/// the runtime fills as Metal does, whatever the level, slice, row and depth
/// ([`fill::texture_buffer_writes`]). `None` where the fill writes nothing.
fn fill_place(
    sel: &'static str,
    shape: &fill::Shape,
    (level, slice, region): (u64, u64, MtlRegion),
) -> Option<(u64, u64, rt::Region)> {
    if shape.texture_type == metal2vulkan::metal::TextureType::TextureBuffer {
        let (o, s) = (region.origin, region.size);
        return fill::texture_buffer_writes(sel, region).then(|| {
            let region = rt::Region::new_3d(o.x, o.y, o.z, s.width, s.height, s.depth);
            (level, slice, region)
        });
    }
    let place = fill::place(sel, shape, level, slice, region)?;
    let ([x, y, z], [w, h, d]) = (place.origin, place.size);
    Some((
        place.level,
        place.slice,
        rt::Region::new_3d(x, y, z, w, h, d),
    ))
}

/// What a blit `cmd` writes: its destination's memory, as a bind keeps it
/// ([`crate::gpu::writes::Bound`]).
fn written_memory(cmd: &BlitCmd<'_, VulkanPlatform>) -> crate::gpu::writes::Bound {
    use crate::gpu::writes::{buffer_bound, texture_bound};
    let texture = |t: Option<&VulkanTexture>| t.map(|t| texture_bound(t)).unwrap_or_default();
    let buffer = |b: Option<&crate::gpu::resource::VulkanBuffer>| {
        b.map(|b| buffer_bound(b)).unwrap_or_default()
    };
    match cmd {
        BlitCmd::CopyBufferToTexture {
            destination_texture,
            ..
        }
        | BlitCmd::CopyTextureRegion {
            destination_texture,
            ..
        }
        | BlitCmd::CopyTextureRegionOptions {
            destination_texture,
            ..
        }
        | BlitCmd::CopyTextureSlices {
            destination_texture,
            ..
        } => texture(Some(destination_texture)),
        BlitCmd::CopyBufferToTextureOptions {
            destination_texture,
            ..
        } => texture(*destination_texture),
        BlitCmd::CopyBufferToBuffer {
            destination_buffer, ..
        }
        | BlitCmd::CopyTextureToBufferOptions {
            destination_buffer, ..
        } => buffer(*destination_buffer),
        BlitCmd::CopyTextureToBuffer {
            destination_buffer, ..
        } => buffer(Some(destination_buffer)),
        BlitCmd::FillBuffer { buffer: b, .. } | BlitCmd::FillBufferPattern4 { buffer: b, .. } => {
            buffer(Some(b))
        }
        BlitCmd::FillTexture { texture: t, .. }
        | BlitCmd::FillTextureColor { texture: t, .. }
        | BlitCmd::InvalidateCompressedTexture { texture: t }
        | BlitCmd::InvalidateCompressedTextureSlice { texture: t, .. } => texture(Some(t)),
        BlitCmd::GenerateMipmaps { texture: t } => texture(*t),
        _ => Default::default(),
    }
}

/// Whether Metal's GPU drops `cmd`'s write into a surface whose GPU writes never reach its
/// memory at every extent measured, as this platform's does, which sends it: a fill of a buffer
/// on the surface, and a copy into one from another buffer (`crate::os::dropped_write`).
fn dropped_at_every_extent(cmd: &BlitCmd<'_, VulkanPlatform>) -> bool {
    matches!(
        cmd,
        BlitCmd::FillBuffer { .. } | BlitCmd::CopyBufferToBuffer { .. }
    )
}

/// The most commands past an indirect command buffer's end a copy was measured to write
/// ([`overrun_measured`]).
const OVERRUN_MEASURED: u64 = 63;

/// Whether a copy of `count` commands to index `to` of an indirect command buffer of `size`
/// commands, `past` of them past its end, from a buffer of shape `from` and its count of
/// commands into one of shape `into` (`crate::gpu::icb::OverrunShape`), is of a shape Metal was
/// measured at (on this host, two runs a shape, compute and render buffers of 2 to 64 commands
/// of two buffers a stage, 1 to 63 commands past the end): both buffers of one measured shape and
/// one count of commands, as every measurement's were, no more commands than the destination
/// holds, an index inside it, and no more than [`OVERRUN_MEASURED`] past it. Metal copies the
/// commands that fit, writes those past the end where no other buffer changes, and may write
/// over the records of the commands before the index
/// (`crate::gpu::icb::IcbObject::overrun`). Larger records, which a descriptor of more buffers
/// gives, or more commands, may reach past the buffer's slot, which nothing measured.
fn overrun_measured(
    (count, size, to, past): (u64, u64, u64, u64),
    from: (Option<crate::gpu::icb::OverrunShape>, u64),
    into: Option<crate::gpu::icb::OverrunShape>,
) -> bool {
    into.is_some() && from == (into, size) && count <= size && to < size && past <= OVERRUN_MEASURED
}

/// The widest and tallest region of a blit into a texture made on a surface whose GPU writes
/// never reach its memory that Metal's GPU was measured to drop at every origin
/// ([`dropped_where_measured`]).
const DROPPED_EXTENT: u64 = 8;

/// The fewest bytes a row of such a surface held where Metal's GPU was measured to drop those
/// blits ([`dropped_where_measured`]).
const DROPPED_ROW_BYTES: u64 = 256;

/// Whether Metal's GPU drops `cmd`, a blit's copy from a buffer or colour fill into a texture
/// made on a surface whose GPU writes never reach its memory, where it was measured to, as this
/// platform's does, which sends it: into a `BGRA8Unorm` 2D texture made on the surface (not a
/// view), at level 0 and slice 0, a region inside the level of 1 to 8 texels each way, on a
/// surface whose rows hold 256 bytes or more. Measured on this host (macOS 26.7, M3 Max, one case
/// a process, two runs: `tests/client_direction.rs`'s `sweep`), on surfaces of 64 by 16 and 256
/// by 256 texels at origins 0,0, 1,1 and 40,10 or 150,200: every region of 1 to 8 texels each way
/// was dropped, the command buffer completing and the guest's pages unchanged. A row of 16 texels
/// or more faults the command buffer at some origins (16 at 0,0, 24 at 40,10, 32 at 1,1) and is
/// dropped at others; a region of more than 8 rows, a smaller surface, another format and every
/// other blit into such a texture were not measured, and die ([`crate::os::dropped_write`]).
fn dropped_where_measured(cmd: &BlitCmd<'_, VulkanPlatform>) -> bool {
    let (texture, level, slice, region) = match cmd {
        BlitCmd::CopyBufferToTexture {
            destination_texture,
            destination_slice,
            destination_level,
            destination_origin,
            source_size,
            ..
        } => (
            *destination_texture,
            *destination_level,
            *destination_slice,
            MtlRegion {
                origin: *destination_origin,
                size: *source_size,
            },
        ),
        BlitCmd::FillTextureColor {
            texture,
            level,
            slice,
            region,
            ..
        } => (*texture, *level, *slice, *region),
        _ => return false,
    };
    let Some(plane) = texture.plane.filter(|_| texture.surface.is_some()) else {
        return false;
    };
    let (format, ty, width, height, view) = plain(|| {
        let raw = &texture.raw;
        (
            raw.pixel_format(),
            raw.texture_type(),
            raw.width(),
            raw.height(),
            raw.parent_texture().is_some(),
        )
    });
    let (origin, size) = (region.origin, region.size);
    let inside = |at: u64, extent: u64, end: u64| {
        (1..=DROPPED_EXTENT).contains(&extent) && at.checked_add(extent).is_some_and(|e| e <= end)
    };
    format == metal2vulkan::metal::PixelFormat::BGRA8Unorm
        && ty == metal2vulkan::metal::TextureType::Type2D
        && !view
        && (level, slice) == (0, 0)
        && (origin.z, size.depth) == (0, 1)
        && inside(origin.x, size.width, width)
        && inside(origin.y, size.height, height)
        && plane.bytes_per_row >= DROPPED_ROW_BYTES
}

/// What the line of a refused copy between a buffer and `texture` names of it: the texture's
/// format, sample count and size, the copy's level and slice, origin and size, and the buffer's
/// bytes a row and an image. A diagnostic: the shapes a copy the runtime does not build takes
/// (`RuntimeGap::MultisampledBufferCopy`).
fn buffer_copy_shape(
    texture: &rt::Texture,
    (level, slice): (u64, u64),
    (origin, size): (MtlOrigin, MtlSize),
    (bytes_per_row, bytes_per_image): (u64, u64),
) -> String {
    format!(
        "the copy of a {:?} texture of {} samples, {}x{}x{}, at level {level}, slice {slice}: \
         {}x{}x{} texels at ({}, {}, {}), {bytes_per_row} bytes a row and {bytes_per_image} an \
         image of the buffer",
        texture.pixel_format(),
        texture.sample_count(),
        texture.width(),
        texture.height(),
        texture.depth(),
        size.width,
        size.height,
        size.depth,
        origin.x,
        origin.y,
        origin.z,
    )
}

impl BlitEncoder<VulkanPlatform> for VulkanBlitEncoder {
    /// On Linux, through the RAM the destination's device uses that holds every run
    /// (`os/linux/host_ram.rs`), where the runtime copies by runs: a first use of Metal's device,
    /// as a blit's copy is. False, with nothing encoded, where no such RAM holds them or the
    /// runtime refuses the copy; never on macOS.
    fn copy_from_guest_ram(
        &self,
        _on: OnFault,
        runs: &[std::ops::Range<u64>],
        destination: &VulkanBuffer,
        destination_offset: u64,
    ) -> RaiseResult<bool> {
        #[cfg(target_os = "linux")]
        {
            let device = &destination.device.raw;
            let Some(ram) = (crate::os::copies_guest_ram(device))
                .then(|| crate::os::ram_over(device, runs))
                .flatten()
            else {
                return Ok(false);
            };
            metal_space::first_use(FirstUse::BlitCommand);
            let copied = self.with(GUEST_RAM_TO_BUFFER, |e| {
                crate::os::copy_from_host_ram(e, &ram, runs, &destination.raw, destination_offset)
            });
            Ok(copied.is_ok())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (runs, destination, destination_offset);
            Ok(false)
        }
    }

    /// As [`Self::copy_from_guest_ram`], the other way, through the RAM the source's device uses.
    fn copy_to_guest_ram(
        &self,
        _on: OnFault,
        source: &VulkanBuffer,
        source_offset: u64,
        runs: &[std::ops::Range<u64>],
    ) -> RaiseResult<bool> {
        #[cfg(target_os = "linux")]
        {
            let device = &source.device.raw;
            let Some(ram) = (crate::os::copies_guest_ram(device))
                .then(|| crate::os::ram_over(device, runs))
                .flatten()
            else {
                return Ok(false);
            };
            metal_space::first_use(FirstUse::BlitCommand);
            let copied = self.with(BUFFER_TO_GUEST_RAM, |e| {
                crate::os::copy_to_host_ram(e, &source.raw, source_offset, &ram, runs)
            });
            Ok(copied.is_ok())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (source, source_offset, runs);
            Ok(false)
        }
    }

    /// Each command but `endEncoding` is a first use of Metal's device
    /// ([`FirstUse::BlitCommand`]: measured for fills and copies, taken alike for the rest), and
    /// a reset or copy of an indirect command buffer is work for Metal's tiler
    /// ([`FirstUse::TilerBlit`]).
    fn encode(&self, _on: OnFault, cmd: BlitCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        let written = written_memory(&cmd);
        if let Some(options) = self.aborts {
            metal_aborts(
                sel,
                &format!("a blit encoder's first command under protection options {options}"),
            );
        }
        if written.kept.holds() {
            crate::gpu::writes::kept_write(sel, "a blit's write into");
        }
        let dropped = written.dropped;
        if dropped && !dropped_at_every_extent(&cmd) && !dropped_where_measured(&cmd) {
            crate::os::dropped_write(sel, "a blit's write into");
        }
        match &cmd {
            BlitCmd::EndEncoding => {}
            BlitCmd::ResetCommandsInBuffer { .. } | BlitCmd::CopyIndirectCommandBuffer { .. } => {
                metal_space::first_use(FirstUse::BlitCommand);
                if let Some(cb) = self.command_buffer() {
                    cb.tiler_work(sel, FirstUse::TilerBlit);
                }
            }
            _ => metal_space::first_use(FirstUse::BlitCommand),
        }
        let caught =
            |call: &mut dyn FnMut(&mut rt::BlitCommandEncoder) -> Result<(), rt::Error>| {
                send(sel, Policy::Dies, || self.with(sel, |e| call(e)))
            };
        // A copy between a buffer and a texture, whose refusal names the copy's shape
        // ([`buffer_copy_shape`]).
        let shaped = |shape: &dyn Fn() -> String,
                      call: &mut dyn FnMut(
            &mut rt::BlitCommandEncoder,
        ) -> Result<(), rt::Error>| {
            send_describing(sel, Policy::Dies, shape, || self.with(sel, |e| call(e)))
        };
        match cmd {
            BlitCmd::CopyBufferToTexture {
                source_buffer,
                source_offset,
                source_bytes_per_row,
                source_bytes_per_image,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
            } => {
                let destination = destination_texture.copied(sel);
                let shape = || {
                    buffer_copy_shape(
                        destination,
                        (destination_level, destination_slice),
                        (destination_origin, source_size),
                        (source_bytes_per_row, source_bytes_per_image),
                    )
                };
                shaped(&shape, &mut |e| {
                    e.copy_from_buffer_to_texture(
                        &source_buffer.raw,
                        source_offset,
                        source_bytes_per_row,
                        source_bytes_per_image,
                        size(source_size),
                        destination,
                        destination_slice,
                        destination_level,
                        origin(destination_origin),
                        rt::BlitOption::None,
                    )
                })
            }
            BlitCmd::CopyBufferToTextureOptions {
                source_buffer,
                source_offset,
                source_bytes_per_row,
                source_bytes_per_image,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
                options,
            } => {
                let source = some(sel, source_buffer, "buffer");
                let destination = some(sel, destination_texture, "texture").copied(sel);
                let options = blit_option(sel, options);
                let shape = || {
                    buffer_copy_shape(
                        destination,
                        (destination_level, destination_slice),
                        (destination_origin, source_size),
                        (source_bytes_per_row, source_bytes_per_image),
                    )
                };
                shaped(&shape, &mut |e| {
                    e.copy_from_buffer_to_texture(
                        &source.raw,
                        source_offset,
                        source_bytes_per_row,
                        source_bytes_per_image,
                        size(source_size),
                        destination,
                        destination_slice,
                        destination_level,
                        origin(destination_origin),
                        options,
                    )
                })
            }
            BlitCmd::CopyBufferToBuffer {
                source_buffer,
                source_offset,
                destination_buffer,
                destination_offset,
                size,
            } => {
                let source = some(sel, source_buffer, "buffer");
                let destination = some(sel, destination_buffer, "buffer");
                caught(&mut |e| {
                    e.copy_from_buffer_to_buffer(
                        &source.raw,
                        source_offset,
                        &destination.raw,
                        destination_offset,
                        size,
                    )
                })
            }
            BlitCmd::CopyTextureToBuffer {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_buffer,
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
            } => {
                let source = source_texture.copied(sel);
                let shape = || {
                    buffer_copy_shape(
                        source,
                        (source_level, source_slice),
                        (source_origin, source_size),
                        (destination_bytes_per_row, destination_bytes_per_image),
                    )
                };
                shaped(&shape, &mut |e| {
                    e.copy_from_texture_to_buffer(
                        source,
                        source_slice,
                        source_level,
                        origin(source_origin),
                        size(source_size),
                        &destination_buffer.raw,
                        destination_offset,
                        destination_bytes_per_row,
                        destination_bytes_per_image,
                        rt::BlitOption::None,
                    )
                })
            }
            BlitCmd::CopyTextureToBufferOptions {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_buffer,
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
                options,
            } => {
                let source = some(sel, source_texture, "texture").copied(sel);
                let destination = some(sel, destination_buffer, "buffer");
                let options = blit_option(sel, options);
                let shape = || {
                    buffer_copy_shape(
                        source,
                        (source_level, source_slice),
                        (source_origin, source_size),
                        (destination_bytes_per_row, destination_bytes_per_image),
                    )
                };
                shaped(&shape, &mut |e| {
                    e.copy_from_texture_to_buffer(
                        source,
                        source_slice,
                        source_level,
                        origin(source_origin),
                        size(source_size),
                        &destination.raw,
                        destination_offset,
                        destination_bytes_per_row,
                        destination_bytes_per_image,
                        options,
                    )
                })
            }
            BlitCmd::CopyTextureRegion {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
            }
            | BlitCmd::CopyTextureRegionOptions {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
                options: 0,
            } => {
                let (source, destination) =
                    (source_texture.copied(sel), destination_texture.copied(sel));
                caught(&mut |e| {
                    e.copy_from_texture_to_texture(
                        source,
                        source_slice,
                        source_level,
                        origin(source_origin),
                        size(source_size),
                        destination,
                        destination_slice,
                        destination_level,
                        origin(destination_origin),
                    )
                })
            }
            BlitCmd::CopyTextureRegionOptions { options, .. } => {
                unrepresentable(sel, &format!("texture copy options {options:#x}"))
            }
            BlitCmd::CopyTextureSlices {
                source_texture,
                source_slice,
                source_level,
                destination_texture,
                destination_slice,
                destination_level,
                slice_count,
                level_count,
            } => {
                let (source, destination) =
                    (source_texture.copied(sel), destination_texture.copied(sel));
                caught(&mut |e| {
                    e.copy_texture_slices(
                        source,
                        source_slice,
                        source_level,
                        destination,
                        destination_slice,
                        destination_level,
                        slice_count,
                        level_count,
                    )
                })
            }
            BlitCmd::CopyIndirectCommandBuffer {
                source,
                source_range,
                destination,
                destination_index,
            } => {
                let from = indices(sel, source_range);
                source.copied_intact(sel, &from);
                let (count, size) = (from.end - from.start, destination.size());
                let past = destination_index
                    .checked_add(count)
                    .map(|end| end.saturating_sub(size));
                let fitting = match past {
                    Some(0) => count,
                    Some(past)
                        if overrun_measured(
                            (count, size, destination_index, past),
                            (source.overrun_shape, source.size()),
                            destination.overrun_shape,
                        ) && !std::ptr::eq(&**source, &**destination) =>
                    {
                        size - destination_index
                    }
                    _ => unrepresentable(
                        sel,
                        &format!(
                            "a copy of {count} commands to index {destination_index} of an \
                             indirect command buffer of {size}, past its end, in a shape nothing \
                             measured (more commands than the destination holds, an index at or \
                             past its end, more than {OVERRUN_MEASURED} past it, a buffer into \
                             itself, or buffers other than two of one measured shape and one \
                             count: 2 to 64 Shared commands of 2 kernel buffers or of 2 vertex and \
                             2 fragment buffers)"
                        ),
                    ),
                };
                let landing = from.start..from.start + fitting;
                caught(&mut |e| {
                    e.copy_indirect_command_buffer(
                        &source.raw,
                        landing.clone(),
                        &destination.raw,
                        destination_index,
                    )
                })?;
                let progress = self.progress(sel);
                if fitting < count {
                    destination.overrun(sel, progress.clone(), 0..destination_index);
                }
                if !landing.is_empty() {
                    crate::gpu::command::executions::gpu_writes(sel, destination);
                }
                source.copied(progress, landing, destination, destination_index);
                Ok(())
            }
            BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer,
                range,
            } => {
                let range = indices(sel, range);
                caught(&mut |e| {
                    e.optimize_indirect_command_buffer(&indirect_command_buffer.raw, range.clone())
                })
            }
            BlitCmd::ResetCommandsInBuffer { buffer, range } => {
                let range = indices(sel, range);
                caught(&mut |e| e.reset_commands_in_buffer(&buffer.raw, range.clone()))?;
                if !range.is_empty() {
                    crate::gpu::command::executions::gpu_writes(sel, buffer);
                }
                buffer.reset(self.progress(sel), range);
                Ok(())
            }
            BlitCmd::FillBufferPattern4 {
                buffer,
                range,
                pattern,
            } => {
                let range = indices(sel, range);
                caught(&mut |e| e.fill_buffer_pattern4(&buffer.raw, range.clone(), pattern))
            }
            BlitCmd::FillTexture {
                texture,
                level,
                slice,
                region,
                bytes,
            } => self.fill_texture(sel, texture, (level, slice, region), bytes),
            BlitCmd::FillTextureColor {
                texture,
                level,
                slice,
                region,
                color,
                pixel_format,
            } => {
                let format = fill::colour_format(sel, pixel_format);
                let place = (level, slice, region);
                self.fill_texture_color(sel, texture, place, color, format)
            }
            BlitCmd::EndEncoding => self.end(sel, |e| e.end_encoding()),
            BlitCmd::FillBuffer {
                buffer,
                range,
                value,
            } => {
                let range = indices(sel, range);
                caught(&mut |e| e.fill_buffer(&buffer.raw, range.clone(), value))
            }
            BlitCmd::GenerateMipmaps { texture } => {
                let texture = some(sel, texture, "texture").copied(sel);
                caught(&mut |e| e.generate_mipmaps(texture))
            }
            BlitCmd::InvalidateCompressedTexture { texture } => {
                let texture = texture.copied(sel);
                caught(&mut |e| e.invalidate_compressed_texture(texture))
            }
            BlitCmd::InvalidateCompressedTextureSlice {
                texture,
                slice,
                level,
            } => {
                let texture = texture.copied(sel);
                caught(&mut |e| e.invalidate_compressed_texture_slice(texture, slice, level))
            }
            BlitCmd::OptimizeForCpu { texture } => {
                caught(&mut |e| e.optimize_contents_for_cpu_access(texture.kept(sel)))
            }
            BlitCmd::OptimizeForCpuSlice {
                texture,
                slice,
                level,
            } => caught(&mut |e| {
                e.optimize_contents_for_cpu_access_slice_level(texture.kept(sel), slice, level)
            }),
            BlitCmd::OptimizeForGpu { texture } => {
                caught(&mut |e| e.optimize_contents_for_gpu_access(texture.kept(sel)))
            }
            BlitCmd::OptimizeForGpuSlice {
                texture,
                slice,
                level,
            } => caught(&mut |e| {
                e.optimize_contents_for_gpu_access_slice_level(texture.kept(sel), slice, level)
            }),
            BlitCmd::SynchronizeResource { resource } => {
                let resource = match resource {
                    // Metal encodes nothing for nil (`blit_texture_maintenance`).
                    None => return self.nothing(sel),
                    Some(ResourceRef::Buffer(b)) => rt::ResourceRef::Buffer(&b.raw),
                    Some(ResourceRef::Texture(t)) => rt::ResourceRef::Texture(t.kept(sel)),
                    Some(ResourceRef::Icb(icb)) => rt::ResourceRef::IndirectCommandBuffer(&icb.raw),
                };
                caught(&mut |e| e.synchronize_resource(resource))
            }
            BlitCmd::SynchronizeTexture {
                texture,
                slice,
                level,
            } => caught(&mut |e| e.synchronize_texture(texture.kept(sel), slice, level)),
            BlitCmd::UpdateFence { fence } => caught(&mut |e| e.update_fence(fence)),
            BlitCmd::WaitForFence { fence } => caught(&mut |e| e.wait_for_fence(fence)),
        }
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::Buffer as _;
    use reims_vgpu_core::platform::types::{MtlOrigin, MtlSize, NsRange};
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::command::VulkanCommandBuffer;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::tests::{bgra, pattern, read_back, shared};

    /// A command buffer of a fresh queue.
    fn buffer() -> VulkanCommandBuffer {
        let q = dev().new_command_queue().unwrap();
        q.command_buffer(OnFault::Return).unwrap().unwrap()
    }

    /// The bytes of a Shared buffer.
    fn bytes(b: &crate::gpu::resource::VulkanBuffer) -> Vec<u8> {
        let length = usize::try_from(b.length()).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), length) }.to_vec()
    }

    /// endEncoding in a thread of an isolated task that has ended stops before it takes the
    /// encoder, which stays in place, untouched, for the blit and the compute encoder alike, as
    /// Metal's is left
    #[test]
    fn an_ended_tasks_end_encoding_leaves_the_encoder() {
        use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use reims_vgpu_core::tasks::isolation::TaskLife;

        let cb = buffer();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let life = TaskLife::new();
        let stopped = life.run(|| {
            life.exit();
            blit.encode(OnFault::Return, BlitCmd::EndEncoding)
        });
        assert!(stopped.is_none());
        assert!(
            blit.raw.lock().borrow().is_some(),
            "the blit encoder was taken"
        );
        let cb = buffer();
        let compute = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let life = TaskLife::new();
        let stopped = life.run(|| {
            life.exit();
            compute.encode(OnFault::Return, ComputeCmd::EndEncoding)
        });
        assert!(stopped.is_none());
        assert!(
            compute.raw.lock().borrow().is_some(),
            "the compute encoder was taken"
        );
    }

    /// Fills and copies between buffers and a texture land on the GPU: a fill, an empty fill and
    /// buffer copy that encode nothing, a buffer copy, a buffer-to-texture copy and back, a
    /// texture-to-texture copy; then endEncoding, commit and the read-back
    #[test]
    fn blits_copy_and_fill() {
        let d = dev();
        let (a, b, c) = (shared(1024), shared(1024), shared(16 * 4 * 4));
        let source = pattern(16, 4);
        // SAFETY: the buffer's Shared bytes, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(source.as_ptr(), c.contents().as_ptr(), source.len())
        };
        let t = d
            .new_texture(OnFault::Return, &bgra(16, 4, 0))
            .unwrap()
            .unwrap();
        let u = d
            .new_texture(OnFault::Return, &bgra(16, 4, 0))
            .unwrap()
            .unwrap();
        let cb = buffer();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer: &a,
            range: NsRange::new(3, 500),
            value: 0x5a,
        };
        e.encode(OnFault::Return, fill).unwrap();
        // An empty fill and an empty copy encode nothing, and the commands after them run.
        let empty_fill = BlitCmd::FillBuffer {
            buffer: &b,
            range: NsRange::new(0x50, 0),
            value: 0x77,
        };
        e.encode(OnFault::Return, empty_fill).unwrap();
        let empty_copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&a),
            source_offset: 0,
            destination_buffer: Some(&b),
            destination_offset: 0,
            size: 0,
        };
        e.encode(OnFault::Return, empty_copy).unwrap();
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&a),
            source_offset: 0,
            destination_buffer: Some(&b),
            destination_offset: 512,
            size: 512,
        };
        e.encode(OnFault::Return, copy).unwrap();
        let up = BlitCmd::CopyBufferToTexture {
            source_buffer: &c,
            source_offset: 0,
            source_bytes_per_row: 64,
            source_bytes_per_image: 256,
            source_size: MtlSize::new(16, 4, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
        };
        e.encode(OnFault::Return, up).unwrap();
        let across = BlitCmd::CopyTextureRegionOptions {
            source_texture: &t,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(16, 4, 1),
            destination_texture: &u,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
            options: 0,
        };
        e.encode(OnFault::Return, across).unwrap();
        let down = BlitCmd::CopyTextureToBufferOptions {
            source_texture: Some(&u),
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(16, 4, 1),
            destination_buffer: Some(&a),
            destination_offset: 768,
            destination_bytes_per_row: 64,
            destination_bytes_per_image: 256,
            options: 0,
        };
        e.encode(OnFault::Return, down).unwrap();
        let fence = d.new_fence().unwrap().unwrap();
        e.encode(OnFault::Return, BlitCmd::UpdateFence { fence: &fence })
            .unwrap();
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let (a, b) = (bytes(&a), bytes(&b));
        assert_eq!(a[..3], [0; 3]);
        assert!(a[3..503].iter().all(|&x| x == 0x5a));
        assert_eq!(a[503..768], [0; 265][..]);
        assert_eq!(&a[768..], &source[..]);
        assert_eq!(b[..512], [0; 512][..]);
        assert_eq!(b[512..515], [0; 3]);
        assert!(b[515..1015].iter().all(|&x| x == 0x5a));
        assert_eq!(read_back(&u), source);
    }

    /// A copy with no blit option between a buffer and a Depth32Float_Stencil8 texture copies
    /// its depth, 4 bytes a texel, both ways, as Metal's does on this host (the same bytes, with
    /// the stencil left as it was). The stencil is written first, by a stencil-only upload: a new
    /// texture's never-written stencil reads what its memory held before, which on an RX 550
    /// (RADV) was an earlier process's bytes. A device that may clamp a depth outside `[0, 1]`
    /// (neither `VK_EXT_depth_range_unrestricted` nor the runtime's
    /// `DepthImageKeepsValuesOutsideUnitRange`: ANV) dies at the upload in strict mode, naming
    /// `restricted-depth-upload`, and copies depths in `[0, 1]` both ways, by default, where any
    /// other device copies any bytes
    #[test]
    fn no_option_copies_a_combined_textures_depth() {
        use metal2vulkan::device::VkExtension;
        const UPLOAD: &str = "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:\
                              sourceSize:toTexture:destinationSlice:destinationLevel:\
                              destinationOrigin:";
        let d = dev();
        let keeps = (d.raw.enabled().extensions).contains(&VkExtension::ExtDepthRangeUnrestricted)
            || (d.raw.properties().driver_facts)
                .contains(&rt::DriverFact::DepthImageKeepsValuesOutsideUnitRange);
        let desc = crate::platform::tests::platform().texture_descriptor(
            &reims_vgpu_core::platform::values::TextureDesc {
                pixel_format: 260,
                width: 4,
                height: 4,
                usage: 3,
                resource_options: Some(0x20),
                ..Default::default()
            },
        );
        let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let seed: Vec<u8> = (0..16u8)
            .map(|i| i.wrapping_mul(13).wrapping_add(0x40))
            .collect();
        let seeds = shared(16);
        // SAFETY: the buffer's Shared bytes, written before any command uses them.
        unsafe { std::ptr::copy_nonoverlapping(seed.as_ptr(), seeds.contents().as_ptr(), 16) };
        let source = shared(128);
        let p: Vec<u8> = if keeps {
            (0..128u8)
                .map(|i| i.wrapping_mul(7).wrapping_add(1))
                .collect()
        } else {
            // Normal depths in [0, 1], which a device that clamps stores as they are.
            (0..32u8)
                .flat_map(|i| (0.25 + f32::from(i) / 64.0).to_le_bytes())
                .collect()
        };
        // SAFETY: the buffer's Shared bytes, written before any command uses them.
        unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), source.contents().as_ptr(), 128) };
        let (depth, stencil, none) = (shared(64), shared(16), shared(128));
        let cb = buffer();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let stencil_up = BlitCmd::CopyBufferToTextureOptions {
            source_buffer: Some(&seeds),
            source_offset: 0,
            source_bytes_per_row: 4,
            source_bytes_per_image: 16,
            source_size: MtlSize::new(4, 4, 1),
            destination_texture: Some(&t),
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
            options: 2,
        };
        e.encode(OnFault::Return, stencil_up).unwrap();
        let up = || BlitCmd::CopyBufferToTexture {
            source_buffer: &source,
            source_offset: 0,
            source_bytes_per_row: 32,
            source_bytes_per_image: 128,
            source_size: MtlSize::new(4, 4, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(0, 0, 0),
        };
        if keeps {
            e.encode(OnFault::Return, up()).unwrap();
        } else {
            let restricted = rt::Degradation::RestrictedDepthUpload;
            let site = crate::answer::dies_of(restricted, || e.encode(OnFault::Return, up()));
            assert_eq!(site.file, UPLOAD);
            crate::answer::relaxed(|| e.encode(OnFault::Return, up())).unwrap();
        }
        for (options, to, row) in [(1, &depth, 16), (2, &stencil, 4), (0, &none, 32)] {
            let down = BlitCmd::CopyTextureToBufferOptions {
                source_texture: Some(&t),
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(4, 4, 1),
                destination_buffer: Some(to),
                destination_offset: 0,
                destination_bytes_per_row: row,
                destination_bytes_per_image: row * 4,
                options,
            };
            e.encode(OnFault::Return, down).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let rows: Vec<u8> = (0..4)
            .flat_map(|r| p[32 * r..32 * r + 16].to_vec())
            .collect();
        assert_eq!(bytes(&depth), rows);
        assert_eq!(bytes(&stencil), seed);
        let padded: Vec<u8> = rows
            .chunks(16)
            .flat_map(|row| row.iter().copied().chain([0; 16]))
            .collect();
        assert_eq!(bytes(&none), padded);
    }

    /// A blit of the two-plane XR10 formats whose answer the runtime does not give as Metal's
    /// dies at its selector, as it announces it (`TwoPlaneTextureCopy`, `TwoPlaneBufferBlit`):
    /// a copy within one texture; one of a texture narrower than 4 texels; one of a texture of
    /// two levels, and of a 2D array of two slices; one whose rows overlap in the destination
    /// (5 texels into rows of 6, which Metal's blit addresses 16 bytes apart); one reading bytes
    /// outside the source's texels where they may hold other bytes than 0: written there by an
    /// earlier blit, and a placement heap's; and a copy from a buffer into a level 34 texels wide,
    /// whose rows meet at Metal's addresses (128 bytes apart, 136 long), where Metal's GPU writes
    /// them in an order that varies from run to run. Each case's last blit is the one that dies
    #[test]
    fn two_plane_xr10_blits_metal_was_not_measured_at_die() {
        use reims_vgpu_core::platform::resource::Heap as _;
        use reims_vgpu_core::platform::values::{HeapDesc, OptionSet};
        let d = dev();
        let p = crate::platform::tests::platform();
        let made = |desc: TextureDesc| {
            let desc = p.texture_descriptor(&TextureDesc {
                pixel_format: 550,
                usage: 0x3,
                ..desc
            });
            d.new_texture(OnFault::Return, &desc).unwrap().unwrap()
        };
        let texture = |width, height| {
            made(TextureDesc {
                width,
                height,
                ..TextureDesc::default()
            })
        };
        let at = MtlOrigin::new(0, 0, 0);
        let (wide, narrow, odd) = (texture(16, 4), texture(2, 4), texture(34, 4));
        let (narrow_into, six) = (texture(2, 4), texture(6, 4));
        let (first, second, third) = (texture(16, 4), texture(16, 4), texture(16, 4));
        let levels = || {
            made(TextureDesc {
                width: 16,
                height: 4,
                mipmap_level_count: 2,
                ..TextureDesc::default()
            })
        };
        let (levels_from, levels_into) = (levels(), levels());
        let slices = || {
            made(TextureDesc {
                texture_type: 3,
                width: 16,
                height: 4,
                array_length: 2,
                ..TextureDesc::default()
            })
        };
        let (slices_from, slices_into) = (slices(), slices());
        let heap = d
            .new_heap(&HeapDesc {
                size: 1 << 20,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(2)],
                address_ranges: None,
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        let placed = heap
            .new_texture(
                &p.texture_descriptor(&TextureDesc {
                    pixel_format: 550,
                    width: 16,
                    height: 8,
                    usage: 0x3,
                    resource_options: Some(0x20),
                    ..TextureDesc::default()
                }),
                0,
            )
            .unwrap()
            .unwrap();
        let wide_into = texture(16, 8);
        let b = d
            .new_buffer_of_length(OnFault::Return, 4096, 0)
            .unwrap()
            .unwrap();
        let region = |from, (sx, sy), (w, h), to, (dx, dy)| BlitCmd::CopyTextureRegion {
            source_texture: from,
            source_slice: 0,
            source_level: 0,
            source_origin: MtlOrigin::new(sx, sy, 0),
            source_size: MtlSize::new(w, h, 1),
            destination_texture: to,
            destination_slice: 0,
            destination_level: 0,
            destination_origin: MtlOrigin::new(dx, dy, 0),
        };
        let all = |from, to, (slice_count, level_count)| BlitCmd::CopyTextureSlices {
            source_texture: from,
            source_slice: 0,
            source_level: 0,
            destination_texture: to,
            destination_slice: 0,
            destination_level: 0,
            slice_count,
            level_count,
        };
        let cases: Vec<(&str, Vec<BlitCmd<'_, VulkanPlatform>>)> = vec![
            (
                "within one texture",
                vec![region(&wide, (0, 0), (4, 2), &wide, (8, 2))],
            ),
            (
                "2 texels wide",
                vec![region(&narrow, (0, 0), (2, 4), &narrow_into, (0, 0))],
            ),
            (
                "of two levels",
                vec![all(&levels_from, &levels_into, (1, 1))],
            ),
            (
                "of two slices",
                vec![all(&slices_from, &slices_into, (2, 1))],
            ),
            (
                "rows overlapping",
                vec![region(&wide, (0, 0), (5, 2), &six, (1, 1))],
            ),
            (
                "outside the texels an earlier blit wrote",
                vec![
                    region(&first, (0, 0), (16, 4), &second, (0, 0)),
                    region(&second, (0, 0), (16, 4), &third, (0, 0)),
                ],
            ),
            (
                "outside a heap's texels",
                vec![region(&placed, (0, 0), (16, 8), &wide_into, (0, 0))],
            ),
            (
                "34 texels wide, from a buffer",
                vec![BlitCmd::CopyBufferToTexture {
                    source_buffer: &b,
                    source_offset: 0,
                    source_bytes_per_row: 136,
                    source_bytes_per_image: 544,
                    source_size: MtlSize::new(34, 4, 1),
                    destination_texture: &odd,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: at,
                }],
            ),
        ];
        for (case, cmds) in cases {
            let sel = cmds.last().unwrap().selector();
            let last = cmds.len() - 1;
            let site = catch_death(|| {
                let cb = buffer();
                let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
                for (i, cmd) in cmds.into_iter().enumerate() {
                    e.encode(OnFault::Return, cmd).unwrap();
                    assert!(i < last, "{case}: blit {i} went on");
                }
                e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
                cb.commit(OnFault::Return).unwrap();
                cb.wait_until_completed(OnFault::Return).unwrap();
            })
            .unwrap_err();
            let abort = reims_vgpu_core::raise::death::DeathKind::Abort;
            assert_eq!((site.kind, site.file), (abort, sel), "{case}");
        }
    }

    /// A refused blit, a nil buffer and a command after endEncoding (an empty fill too) die at
    /// their selectors
    #[test]
    fn blit_refusals() {
        let a = shared(64);
        let cb = buffer();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let past = BlitCmd::FillBuffer {
            buffer: &a,
            range: NsRange::new(0, 65),
            value: 1,
        };
        let site = catch_death(|| e.encode(OnFault::Return, past)).unwrap_err();
        assert_eq!(site.file, "fillBuffer:range:value:");
        let site = catch_death(|| {
            let nil = BlitCmd::CopyBufferToBuffer {
                source_buffer: None,
                source_offset: 0,
                destination_buffer: Some(&a),
                destination_offset: 0,
                size: 4,
            };
            e.encode(OnFault::Return, nil)
        })
        .unwrap_err();
        assert_eq!(
            site.file,
            "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"
        );
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        let site = catch_death(|| {
            let fill = BlitCmd::FillBuffer {
                buffer: &a,
                range: NsRange::new(0, 4),
                value: 1,
            };
            e.encode(OnFault::Return, fill)
        })
        .unwrap_err();
        assert_eq!(site.file, "fillBuffer:range:value:");
        let site = catch_death(|| {
            let empty = BlitCmd::FillBuffer {
                buffer: &a,
                range: NsRange::new(0, 0),
                value: 1,
            };
            e.encode(OnFault::Return, empty)
        })
        .unwrap_err();
        assert_eq!(site.file, "fillBuffer:range:value:");
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
