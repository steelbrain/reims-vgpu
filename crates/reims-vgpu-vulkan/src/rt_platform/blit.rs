//! The rt platform's blit encoders ([`RtBlitEncoder`]): rt's, which records each copy into its
//! command buffer at once, ordered against every command before and after it.
//!
//! - **Copies** between buffers, between a buffer and a texture and between textures, buffer
//!   fills, mipmap generation, synchronizes of Managed resources and fences are rt's. A copy of
//!   whole slices and levels (`copyFromTexture:sourceSlice:sourceLevel:toTexture:...sliceCount:
//!   levelCount:`) is one rt copy of each level of each slice, the level's whole extent.
//! - **`synchronizeTexture:slice:level:`** synchronizes the whole texture, rt's one form: the
//!   other slices' and levels' bytes the host reads take the GPU's too, which for a Managed
//!   texture whose host writes reached the GPU (`replaceRegion`, `didModifyRange`) are the
//!   bytes they hold.
//! - **Nothing to record**: `optimizeContentsFor{CPU,GPU}Access:` (rt lays a texture out one way,
//!   and Metal's optimize changes no texel), and `invalidateCompressedTexture:` (rt keeps no
//!   lossless compression whose contents an invalidate would drop).
//! - **Unbuilt**: a fill of a texture with bytes ([`Unbuilt::FillTexture`]); a fill of a
//!   texture with a colour, a fill of a buffer with a 4-byte pattern of more than one byte value,
//!   and a copy with blit options ([`Unbuilt::Blit`]). Each encodes nothing, counted.
//! - **The guest's RAM** is copied by no blit: the device copies none
//!   (`GpuDevice::copies_guest_ram`), so each copy of it answers false.
//!
//! A command rt refuses encodes nothing, counted under the refusal's kind
//! ([`rt_refused_texture`]); so does one Metal's validation stops before rt (a nil argument, a
//! value past rt's 32 bits, a command after `endEncoding`), as Metal's refusal.

use std::ops::Range;

use metal2vulkan_rt as rt;

use crate::core::platform::Platform;
use crate::core::platform::blit::{BlitCmd, BlitEncoder, ResourceRef};
use crate::core::platform::types::{MtlOrigin, MtlSize, NsRange};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::Mutex;

use super::command::OpenMark;
use super::unbuilt::{RtRefusal, Unbuilt, refuse, refused_as, rt_refused_texture};
use super::{RtObj, RtPlatform};

/// `id<MTLBlitCommandEncoder>` on metal2vulkan-rt.
pub struct RtBlitEncoder {
    /// rt's encoder, until `endEncoding` ends it.
    rt: Mutex<Option<rt::BlitCommandEncoder>>,
    /// Its command buffer's mark of it open, until it ends.
    open: OpenMark,
}

impl RtBlitEncoder {
    /// The encoder over rt's `encoder`, open as `open` marks it.
    pub fn new(encoder: rt::BlitCommandEncoder, open: OpenMark) -> RtBlitEncoder {
        RtBlitEncoder {
            rt: Mutex::new(Some(encoder)),
            open,
        }
    }
}

/// Why a command encodes nothing before rt sees it.
enum Stop {
    /// Metal's validation stops it: counted as Metal's refusal, with what it was.
    Metal(&'static str),
    /// An unbuilt piece.
    Unbuilt(Unbuilt),
    /// rt refused it.
    Rt(rt::Error),
}

impl From<rt::Error> for Stop {
    fn from(error: rt::Error) -> Stop {
        Stop::Rt(error)
    }
}

/// A nil argument Metal's validation stops.
const NIL: Stop = Stop::Metal("a nil argument");

/// A value past rt's 32 bits, which no texture holds.
const WIDE: Stop = Stop::Metal("a slice, level, origin or size past 32 bits");

/// A byte range past the end of 64 bits.
const PAST: Stop = Stop::Metal("a range past 2^64");

/// `value` as rt's 32 bits.
fn narrow(value: u64) -> Result<u32, Stop> {
    u32::try_from(value).map_err(|_| WIDE)
}

/// A texture's slice, level and origin as rt's place.
fn place(slice: u64, level: u64, origin: MtlOrigin) -> Result<rt::TexturePlace, Stop> {
    Ok(rt::TexturePlace {
        slice: narrow(slice)?,
        level: narrow(level)?,
        origin: [narrow(origin.x)?, narrow(origin.y)?, narrow(origin.z)?],
    })
}

/// A region's size as rt's.
fn size(size: MtlSize) -> Result<[u32; 3], Stop> {
    Ok([
        narrow(size.width)?,
        narrow(size.height)?,
        narrow(size.depth)?,
    ])
}

/// A byte range as rt's; one past the end of 64 bits is Metal's refusal.
fn bytes(range: NsRange) -> Result<Range<u64>, Stop> {
    let end = (range.location.checked_add(range.length)).ok_or(PAST)?;
    Ok(range.location..end)
}

/// The buffer's layout of a copy between a buffer and a texture.
fn layout(offset: u64, bytes_per_row: u64, bytes_per_image: u64) -> rt::BufferLayout {
    rt::BufferLayout {
        offset,
        bytes_per_row,
        bytes_per_image,
    }
}

/// `Ok` for no blit options; [`Unbuilt::Blit`] for any.
fn no_options(options: u64) -> Result<(), Stop> {
    match options {
        0 => Ok(()),
        _ => Err(Stop::Unbuilt(Unbuilt::Blit)),
    }
}

/// Copies `slices` slices and `levels` levels of `source` from `from` (slice, level) into
/// `destination` from `to`, each level's whole extent of `source`, one rt copy each.
fn copy_slices(
    e: &mut rt::BlitCommandEncoder,
    (source, from): (&rt::Texture, (u32, u32)),
    (destination, to): (&rt::Texture, (u32, u32)),
    (slices, levels): (u32, u32),
) -> Result<(), Stop> {
    let d = source.descriptor();
    let three_d = d.texture_type == rt::TextureType::Type3D;
    for level in 0..levels {
        let (at, into) = (from.1.saturating_add(level), to.1.saturating_add(level));
        let extent = |of: u32| (of.checked_shr(at).unwrap_or(0)).max(1);
        let depth = if three_d { extent(d.depth) } else { 1 };
        let size = [extent(d.width), extent(d.height), depth];
        for slice in 0..slices {
            let (s, t) = (from.0.saturating_add(slice), to.0.saturating_add(slice));
            let from = rt::TexturePlace {
                slice: s,
                level: at,
                origin: [0; 3],
            };
            let to = rt::TexturePlace {
                slice: t,
                level: into,
                origin: [0; 3],
            };
            e.copy_from_texture(source, from, size, destination, to)?;
        }
    }
    Ok(())
}

/// Encodes `cmd` into `e`, rt's open encoder (the module's docs); `EndEncoding` is the caller's.
fn encode(e: &mut rt::BlitCommandEncoder, cmd: BlitCmd<'_, RtPlatform>) -> Result<(), Stop> {
    match cmd {
        BlitCmd::EndEncoding => {}
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
            let to = place(destination_slice, destination_level, destination_origin)?;
            let lies = layout(source_offset, source_bytes_per_row, source_bytes_per_image);
            let (source, destination) = (source_buffer.rt(), destination_texture.rt());
            e.copy_from_buffer(source, lies, size(source_size)?, destination, to)?;
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
            let (Some(source), Some(destination)) = (source_buffer, destination_texture) else {
                return Err(NIL);
            };
            no_options(options)?;
            let to = place(destination_slice, destination_level, destination_origin)?;
            let lies = layout(source_offset, source_bytes_per_row, source_bytes_per_image);
            let (source, destination) = (source.rt(), destination.rt());
            e.copy_from_buffer(source, lies, size(source_size)?, destination, to)?;
        }
        BlitCmd::CopyBufferToBuffer {
            source_buffer,
            source_offset,
            destination_buffer,
            destination_offset,
            size,
        } => {
            let (Some(source), Some(destination)) = (source_buffer, destination_buffer) else {
                return Err(NIL);
            };
            let (from, to) = (
                (source.rt(), source_offset),
                (destination.rt(), destination_offset),
            );
            e.copy_from_buffer_to_buffer(from.0, from.1, to.0, to.1, size)?;
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
            let from = place(source_slice, source_level, source_origin)?;
            let lies = layout(
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
            );
            let (source, destination) = (source_texture.rt(), destination_buffer.rt());
            e.copy_from_texture_to_buffer(source, from, size(source_size)?, destination, lies)?;
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
            let (Some(source), Some(destination)) = (source_texture, destination_buffer) else {
                return Err(NIL);
            };
            no_options(options)?;
            let from = place(source_slice, source_level, source_origin)?;
            let lies = layout(
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
            );
            let (source, destination) = (source.rt(), destination.rt());
            e.copy_from_texture_to_buffer(source, from, size(source_size)?, destination, lies)?;
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
            let from = place(source_slice, source_level, source_origin)?;
            let to = place(destination_slice, destination_level, destination_origin)?;
            let (source, destination) = (source_texture.rt(), destination_texture.rt());
            e.copy_from_texture(source, from, size(source_size)?, destination, to)?;
        }
        BlitCmd::CopyTextureRegionOptions { .. } => return Err(Stop::Unbuilt(Unbuilt::Blit)),
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
            let from = (narrow(source_slice)?, narrow(source_level)?);
            let to = (narrow(destination_slice)?, narrow(destination_level)?);
            let counts = (narrow(slice_count)?, narrow(level_count)?);
            let source = (source_texture.rt(), from);
            copy_slices(e, source, (destination_texture.rt(), to), counts)?;
        }
        BlitCmd::CopyIndirectCommandBuffer { source, .. } => source.unreachable(),
        BlitCmd::OptimizeIndirectCommandBuffer {
            indirect_command_buffer,
            ..
        } => indirect_command_buffer.unreachable(),
        BlitCmd::ResetCommandsInBuffer { buffer, .. } => buffer.unreachable(),
        BlitCmd::FillBufferPattern4 {
            buffer,
            range,
            pattern,
        } => {
            let [value, rest @ ..] = pattern.to_le_bytes();
            if rest.iter().any(|&byte| byte != value) {
                return Err(Stop::Unbuilt(Unbuilt::Blit));
            }
            e.fill_buffer(buffer.rt(), bytes(range)?, value)?;
        }
        BlitCmd::FillBuffer {
            buffer,
            range,
            value,
        } => e.fill_buffer(buffer.rt(), bytes(range)?, value)?,
        BlitCmd::FillTexture { .. } => return Err(Stop::Unbuilt(Unbuilt::FillTexture)),
        BlitCmd::FillTextureColor { .. } => return Err(Stop::Unbuilt(Unbuilt::Blit)),
        BlitCmd::GenerateMipmaps { texture } => {
            e.generate_mipmaps(texture.ok_or(NIL)?.rt())?;
        }
        BlitCmd::InvalidateCompressedTexture { .. }
        | BlitCmd::InvalidateCompressedTextureSlice { .. }
        | BlitCmd::OptimizeForCpu { .. }
        | BlitCmd::OptimizeForCpuSlice { .. }
        | BlitCmd::OptimizeForGpu { .. }
        | BlitCmd::OptimizeForGpuSlice { .. } => {}
        BlitCmd::SynchronizeResource { resource } => match resource {
            // Metal encodes nothing for nil.
            None => {}
            Some(ResourceRef::Buffer(buffer)) => {
                e.synchronize_resource(rt::Resource::Buffer(buffer.rt()))?;
            }
            Some(ResourceRef::Texture(texture)) => {
                e.synchronize_resource(rt::Resource::Texture(texture.rt()))?;
            }
            Some(ResourceRef::Icb(icb)) => icb.unreachable(),
        },
        BlitCmd::SynchronizeTexture { texture, .. } => {
            e.synchronize_resource(rt::Resource::Texture(texture.rt()))?;
        }
        BlitCmd::UpdateFence { fence } => e.update_fence(fence)?,
        BlitCmd::WaitForFence { fence } => e.wait_for_fence(fence)?,
    }
    Ok(())
}

impl BlitEncoder<RtPlatform> for RtObj<RtBlitEncoder> {
    /// rt's command (the module's docs); `endEncoding` ends rt's encoder. A command after it is
    /// Metal's refusal, counted.
    fn encode(&self, on: OnFault, cmd: BlitCmd<'_, RtPlatform>) -> RaiseResult<()> {
        let _ = on;
        let sel = cmd.selector();
        let mut open = self.rt.lock();
        let ending = matches!(cmd, BlitCmd::EndEncoding);
        let stopped = match open.as_mut() {
            Some(e) => encode(e, cmd).err(),
            None => Some(Stop::Metal("a command after endEncoding")),
        };
        if ending && let Some(e) = open.take() {
            e.end_encoding();
            self.open.end();
        }
        drop(open);
        match stopped {
            None => {}
            Some(Stop::Metal(what)) => refused_as(sel, RtRefusal::Refused, &what),
            Some(Stop::Unbuilt(piece)) => refuse(piece, sel),
            Some(Stop::Rt(error)) => {
                rt_refused_texture(sel, &error);
            }
        }
        Ok(())
    }

    /// False: the device copies none of the guest's RAM.
    fn copy_from_guest_ram(
        &self,
        on: OnFault,
        runs: &[Range<u64>],
        destination: &<RtPlatform as Platform>::Buffer,
        destination_offset: u64,
    ) -> RaiseResult<bool> {
        let _ = (on, runs, destination, destination_offset);
        Ok(false)
    }

    /// False: the device copies none of the guest's RAM.
    fn copy_from_guest_ram_at(
        &self,
        on: OnFault,
        runs: &[(u64, Range<u64>)],
        destination: &<RtPlatform as Platform>::Buffer,
    ) -> RaiseResult<bool> {
        let _ = (on, runs, destination);
        Ok(false)
    }

    /// False: the device copies none of the guest's RAM.
    fn copy_to_guest_ram(
        &self,
        on: OnFault,
        source: &<RtPlatform as Platform>::Buffer,
        runs: &[(u64, Range<u64>)],
    ) -> RaiseResult<bool> {
        let _ = (on, source, runs);
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::num::{ToU64 as _, ToUsize as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::host_range::HostRange;
    use crate::core::platform::resource::Buffer as _;
    use crate::core::platform::types::{ClearColor, CommandBufferStatus, MtlRegion};
    use crate::core::platform::values::TextureDesc;
    use crate::rt_platform::{RtBuffer, RtQueue, RtTexture};

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// Writes `bytes` at the start of the Shared `buffer`.
    fn write(buffer: &RtObj<RtBuffer>, bytes: &[u8]) {
        assert!(bytes.len().to_u64() <= buffer.length());
        // SAFETY: a Shared buffer's contents are mapped, at least `bytes` long.
        unsafe {
            let at = buffer.contents().as_ptr::<u8>();
            at.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        }
    }

    /// The first `length` bytes of the Shared or Managed `buffer`.
    fn read(buffer: &RtObj<RtBuffer>, length: usize) -> Vec<u8> {
        assert!(length.to_u64() <= buffer.length());
        // SAFETY: as `write`'s.
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr::<u8>(), length) }.to_vec()
    }

    /// Encodes `cmds` into one blit encoder of a command buffer of `queue`, ends it, commits the
    /// command buffer and waits for it: its status then.
    fn blit(queue: &RtObj<RtQueue>, cmds: Vec<BlitCmd<'_, RtPlatform>>) -> CommandBufferStatus {
        let on = OnFault::Return;
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let encoder = made.blit_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a blit encoder");
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        made.status()
    }

    /// A copy of `extent` × `extent` texels between `buffer` at `offset`, rows tight, and the
    /// texture's `slice` and `level`: into the texture where `up`, else out of it.
    fn copy<'a>(
        up: bool,
        (buffer, offset): (&'a RtObj<RtBuffer>, u64),
        (texture, slice, level): (&'a RtObj<RtTexture>, u64, u64),
        extent: u64,
    ) -> BlitCmd<'a, RtPlatform> {
        let (row, size) = (4 * extent, MtlSize::new(extent, extent, 1));
        let origin = MtlOrigin::default();
        if up {
            BlitCmd::CopyBufferToTexture {
                source_buffer: buffer,
                source_offset: offset,
                source_bytes_per_row: row,
                source_bytes_per_image: row * extent,
                source_size: size,
                destination_texture: texture,
                destination_slice: slice,
                destination_level: level,
                destination_origin: origin,
            }
        } else {
            BlitCmd::CopyTextureToBuffer {
                source_texture: texture,
                source_slice: slice,
                source_level: level,
                source_origin: origin,
                source_size: size,
                destination_buffer: buffer,
                destination_offset: offset,
                destination_bytes_per_row: row,
                destination_bytes_per_image: row * extent,
            }
        }
    }

    /// Blits on the GPU the test holds: buffer fills, a pattern of one byte value among them,
    /// and copies; a buffer to a texture and back, a region of it to another texture, through
    /// the plain and options forms; whole slices and levels of a 2D array, each level's whole
    /// extent; mipmaps; a Managed buffer filled on the GPU and synchronized, and a Managed
    /// texture's synchronizes; optimizes and invalidates; fences between two encoders. Each
    /// encodes nothing, counted: an unequal pattern, blit options, a texture fill with bytes or
    /// a colour, a nil argument, a copy rt refuses, a second encoder while one is open and a
    /// command after `endEncoding`. No blit copies the guest's RAM.
    #[test]
    fn blits_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let buffer = |length: u64, options: u64| {
            let made = device.new_buffer_of_length(on, length, options);
            made.expect("never a raise").expect("a buffer")
        };
        let texture = |desc: &TextureDesc| -> RtObj<RtTexture> {
            let made = device.new_texture(on, &platform.texture_descriptor(desc));
            made.expect("never a raise").expect("a texture")
        };
        let done = CommandBufferStatus::COMPLETED;
        let metal = RtRefusal::Refused.code();
        let (fill_texture, unbuilt) = (Unbuilt::FillTexture.code(), Unbuilt::Blit.code());

        // Buffers.
        let (a, b) = (buffer(256, 0), buffer(256, 0));
        let cmds = vec![
            BlitCmd::FillBuffer {
                buffer: &a,
                range: NsRange::new(0, 256),
                value: 0x5a,
            },
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&a),
                source_offset: 0,
                destination_buffer: Some(&b),
                destination_offset: 16,
                size: 128,
            },
            BlitCmd::FillBufferPattern4 {
                buffer: &a,
                range: NsRange::new(0, 64),
                pattern: 0x1111_1111,
            },
        ];
        assert_eq!(blit(&queue, cmds), done);
        let mut copied = vec![0; 256];
        copied[16..144].fill(0x5a);
        assert_eq!(read(&b, 256), copied);
        let mut filled = vec![0x5a; 256];
        filled[..64].fill(0x11);
        assert_eq!(read(&a, 256), filled);

        // A buffer to a texture and back; a region of it to another texture.
        let small = TextureDesc {
            width: 4,
            height: 4,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        };
        let (t, u) = (texture(&small), texture(&small));
        let texels: Vec<u8> = (0..64).collect();
        let (up, down, corner) = (buffer(64, 0), buffer(64, 0), buffer(16, 0));
        write(&up, &texels);
        let cmds = vec![
            copy(true, (&up, 0), (&t, 0, 0), 4),
            copy(false, (&down, 0), (&t, 0, 0), 4),
            BlitCmd::CopyTextureRegion {
                source_texture: &t,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(1, 1, 0),
                source_size: MtlSize::new(2, 2, 1),
                destination_texture: &u,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::default(),
            },
            BlitCmd::CopyTextureToBufferOptions {
                source_texture: Some(&u),
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::default(),
                source_size: MtlSize::new(2, 2, 1),
                destination_buffer: Some(&corner),
                destination_offset: 0,
                destination_bytes_per_row: 8,
                destination_bytes_per_image: 16,
                options: 0,
            },
        ];
        assert_eq!(blit(&queue, cmds), done);
        assert_eq!(read(&down, 64), texels);
        let texel = |x: usize, y: usize| texels[(y * 4 + x) * 4..][..4].to_vec();
        let region = [texel(1, 1), texel(2, 1), texel(1, 2), texel(2, 2)].concat();
        assert_eq!(read(&corner, 16), region);

        // Whole slices and levels of a 2D array.
        let array = TextureDesc {
            texture_type: 3,
            array_length: 2,
            mipmap_level_count: 2,
            ..small
        };
        let (from, into) = (texture(&array), texture(&array));
        let places = [(0, 0, 4), (0, 1, 2), (1, 0, 4), (1, 1, 2)];
        let (staged, back) = (buffer(256, 0), buffer(256, 0));
        let mut bytes = vec![0; 256];
        for (at, &(slice, level, extent)) in places.iter().enumerate() {
            let value = u8::try_from(0x10 * slice + level + 1).expect("a byte");
            bytes[at * 64..][..(4 * extent * extent).to_usize()].fill(value);
        }
        write(&staged, &bytes);
        let mut cmds = Vec::new();
        for (at, &(slice, level, extent)) in places.iter().enumerate() {
            let offset = 64 * at.to_u64();
            cmds.push(copy(true, (&staged, offset), (&from, slice, level), extent));
        }
        cmds.push(BlitCmd::CopyTextureSlices {
            source_texture: &from,
            source_slice: 0,
            source_level: 0,
            destination_texture: &into,
            destination_slice: 0,
            destination_level: 0,
            slice_count: 2,
            level_count: 2,
        });
        for (at, &(slice, level, extent)) in places.iter().enumerate() {
            let offset = 64 * at.to_u64();
            cmds.push(copy(false, (&back, offset), (&into, slice, level), extent));
        }
        assert_eq!(blit(&queue, cmds), done);
        assert_eq!(read(&back, 256), bytes);

        // Mipmaps of a level of one colour: every level that colour.
        let mipped = texture(&TextureDesc {
            mipmap_level_count: 3,
            ..small
        });
        let (grey, last) = (buffer(64, 0), buffer(4, 0));
        write(&grey, &[0x80; 64]);
        let cmds = vec![
            copy(true, (&grey, 0), (&mipped, 0, 0), 4),
            BlitCmd::GenerateMipmaps {
                texture: Some(&mipped),
            },
            copy(false, (&last, 0), (&mipped, 0, 2), 1),
        ];
        assert_eq!(blit(&queue, cmds), done);
        assert_eq!(read(&last, 4), [0x80; 4]);

        // Synchronizes, optimizes and invalidates.
        let managed = buffer(64, 0x10);
        let kept = texture(&TextureDesc {
            resource_options: Some(0x10),
            ..small
        });
        let before = counted(metal);
        let cmds = vec![
            BlitCmd::FillBuffer {
                buffer: &managed,
                range: NsRange::new(0, 64),
                value: 0x77,
            },
            BlitCmd::SynchronizeResource {
                resource: Some(ResourceRef::Buffer(&managed)),
            },
            BlitCmd::SynchronizeResource {
                resource: Some(ResourceRef::Texture(&kept)),
            },
            BlitCmd::SynchronizeTexture {
                texture: &kept,
                slice: 0,
                level: 0,
            },
            BlitCmd::SynchronizeResource { resource: None },
            BlitCmd::OptimizeForCpu { texture: &t },
            BlitCmd::OptimizeForGpuSlice {
                texture: &t,
                slice: 0,
                level: 0,
            },
            BlitCmd::InvalidateCompressedTexture { texture: &t },
        ];
        assert_eq!(blit(&queue, cmds), done);
        assert_eq!(read(&managed, 64), [0x77; 64]);
        assert_eq!(counted(metal), before);

        // What encodes nothing: the bytes stay as they were.
        let before = [counted(metal), counted(fill_texture), counted(unbuilt)];
        let whole = MtlRegion {
            origin: MtlOrigin::default(),
            size: MtlSize::new(4, 4, 1),
        };
        let zeros = [0; 64];
        let red = ClearColor {
            red: 1.0,
            green: 0.0,
            blue: 0.0,
            alpha: 1.0,
        };
        let cmds = vec![
            BlitCmd::FillBufferPattern4 {
                buffer: &a,
                range: NsRange::new(0, 256),
                pattern: 0x0102_0304,
            },
            BlitCmd::CopyTextureRegionOptions {
                source_texture: &t,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::default(),
                source_size: MtlSize::new(4, 4, 1),
                destination_texture: &u,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::default(),
                options: 1,
            },
            BlitCmd::FillTexture {
                texture: &t,
                level: 0,
                slice: 0,
                region: whole,
                bytes: HostRange::of_slice(&zeros),
            },
            BlitCmd::FillTextureColor {
                texture: &t,
                level: 0,
                slice: 0,
                region: whole,
                color: red,
                pixel_format: 70,
            },
            BlitCmd::CopyBufferToTextureOptions {
                source_buffer: None,
                source_offset: 0,
                source_bytes_per_row: 16,
                source_bytes_per_image: 64,
                source_size: MtlSize::new(4, 4, 1),
                destination_texture: Some(&t),
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::default(),
                options: 0,
            },
            BlitCmd::GenerateMipmaps { texture: None },
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&a),
                source_offset: 0,
                destination_buffer: Some(&b),
                destination_offset: 0,
                size: 3,
            },
            copy(false, (&down, 0), (&t, 0, 0), 4),
        ];
        assert_eq!(blit(&queue, cmds), done);
        assert_eq!((read(&a, 256), read(&b, 256)), (filled, copied));
        assert_eq!(read(&down, 64), texels);
        let after = [counted(metal), counted(fill_texture), counted(unbuilt)];
        assert_eq!(after, [before[0] + 3, before[1] + 1, before[2] + 3]);

        // Fences between two encoders; a second encoder while one is open, a command after
        // `endEncoding` and the guest's RAM.
        let fence = device.new_fence().expect("never a raise");
        let fence = fence.expect("a fence");
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let before = counted(metal);
        for cmd in [
            BlitCmd::UpdateFence { fence: &fence },
            BlitCmd::WaitForFence { fence: &fence },
        ] {
            let encoder = made.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            assert!(matches!(made.blit_encoder(on), Ok(None)));
            assert_eq!(encoder.encode(on, cmd), Ok(()));
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            let run = 0..64;
            assert_eq!(
                encoder.copy_from_guest_ram(on, std::slice::from_ref(&run), &a, 0),
                Ok(false)
            );
            assert_eq!(
                encoder.copy_from_guest_ram_at(on, &[(0, 0..64)], &a),
                Ok(false)
            );
            assert_eq!(encoder.copy_to_guest_ram(on, &a, &[(0, 0..64)]), Ok(false));
        }
        assert_eq!(counted(metal), before + 4);
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        assert_eq!(made.status(), done);
    }
}
