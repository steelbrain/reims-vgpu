//! The rt platform's tiled textures ([`Tiled`]): `newTiledTextureWithDescriptor:offset:
//! bytesPerRow:` of a 2D texture of more than one level, which rt makes no form of
//! (`super::buffer`). Of one level, or of another type, the call is
//! `newTextureWithDescriptor:offset:bytesPerRow:`'s, rt's texture over the buffer's bytes.
//!
//! A tiled texture of more levels is a copy of the buffer's bytes, as an x86 Mac's is
//! (docs/graphics-kext/dgpu-oracle.md, Q11): rt's texture of the descriptor, an image of every
//! level of the platform's own (`super::texture::make_over`, of the buffer's storage mode), whose
//! bytes are taken at its first use, and again at its first use after a `didModifyData`:
//!
//! - **A use** is a command of an encoder that names the texture or a view of it: a texture a
//!   render or compute command binds, a resource `useResources:` names, an attachment of a render
//!   pass, a blit's source or destination, `generateMipmapsForTexture:` and a synchronize. The
//!   command buffer notes the texture where its bytes are to take ([`Takes`]).
//! - **The take.** The commit of a command buffer that used the texture while its bytes were to
//!   take, the first such commit, takes them before it commits: one command buffer, of a queue of
//!   its own that holds nothing, committed ahead of it, copies the bytes from the offset, rows
//!   `bytes_per_row` apart, into level 0 (rt's `copy_from_buffer`), then makes every later level
//!   from it as `generateMipmapsForTexture:` makes them (rt's `generate_mipmaps`, the device's
//!   linear filter). rt submits in the device's commit order where nothing is held, and each of
//!   its copies' barriers reaches every command submitted after it, so the guest's commands read
//!   the levels taken. The copy reads the bytes as they are when it runs: the CPU's writes made
//!   before the commit are in every level, of a Managed buffer those its `didModifyRange:` named.
//! - **Between takes nothing moves**: a CPU write of the bytes reaches no level, and what the GPU
//!   writes into a level stays in the texture, never in the bytes.
//! - **`replaceRegion:` of level 0** writes the buffer's bytes at the texture's rows, as the Macs'
//!   does, and the next use takes them, as after a `didModifyData` ([`Tiled::replace`]); of a
//!   Managed buffer it names the rows it wrote to rt, as `didModifyRange:` does. Of a later level
//!   it writes the image, whose texels the GPU reads until the next take makes that level again.
//! - **Made only of rows Metal takes** ([`bad_layout`]): rows a whole number of the format's
//!   blocks apart and no closer than a row, an offset a whole number of blocks in, and every row
//!   within the buffer. Any other is Metal's validation's stop as the texture is made, here nil,
//!   counted (`rt-refused-metal`).
//!
//! What differs from the Macs, and from the Vulkan platform's runtime, which takes the bytes as the
//! using command runs: a CPU write after the commit and before the command, and a GPU write of the
//! bytes by an earlier command buffer the device still holds (at an event wait, or in a queue's
//! group), are in no level. A format rt makes no mipmaps of, or whose blits do not filter, takes
//! level 0 alone, the later levels keeping what they hold, its refusal counted under rt's kind.
//! `replaceRegion:` of level 0 over a buffer whose bytes the host has no mapping of writes nothing,
//! counted (`rt-refused-other`). Each texture's take is counted ([`TAKES_CODE`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_rt as rt;

use crate::core::num::{ToU64 as _, ToUsize as _};
use crate::core::platform::Handle as _;
use crate::core::platform::resource::Buffer as _;
use crate::core::platform::types::{MtlRegion, NsRange};
use crate::core::platform::values::TextureDesc;
use crate::core::raise::OnFault;
use crate::core::sync::Mutex;

use super::RtObj;
use super::buffer::RtBuffer;
use super::device::RtDevice;
use super::texture::RtTexture;
use super::unbuilt::{RtRefusal, rt_refused, rt_refused_texture};

/// The selector whose texture's take a refusal names.
const SEL: &str = "newTiledTextureWithDescriptor:offset:bytesPerRow:";

/// The code of a tiled texture's take, counted per texture taken.
pub const TAKES_CODE: &str = "rt-tiled-takes";

/// Metal's flag of a format of blocks (`MTLPixelFormatGetInfoForDevice`'s flags bit 10).
const BLOCKS: u64 = 0x400;

/// What a tiled texture takes its bytes from, and whether they are to take (the module's docs).
pub struct Tiled {
    /// The device of its texture.
    device: rt::Device,
    /// rt's texture, whose levels the take fills.
    texture: rt::Texture,
    /// The buffer whose bytes it takes, held as Metal's texture holds its buffer.
    buffer: RtObj<RtBuffer>,
    /// Where level 0's first row lies in the buffer.
    offset: u64,
    /// From one row's first byte to the next's.
    bytes_per_row: u64,
    /// Its next use takes the bytes: from its making, and again from a `didModifyData`.
    stale: AtomicBool,
}

impl Tiled {
    /// rt's `texture` of `device`, taking the bytes of `buffer` from `offset`, rows
    /// `bytes_per_row` apart, at its first use.
    pub(super) fn new(
        device: &rt::Device,
        texture: &rt::Texture,
        buffer: &RtObj<RtBuffer>,
        offset: u64,
        bytes_per_row: u64,
    ) -> Arc<Tiled> {
        Arc::new(Tiled {
            device: device.clone(),
            texture: texture.clone(),
            buffer: buffer.clone_ref(),
            offset,
            bytes_per_row,
            stale: AtomicBool::new(true),
        })
    }

    /// `didModifyData`: its next use takes the bytes again.
    pub(super) fn modified(&self) {
        self.stale.store(true, Ordering::Release);
    }

    /// `replaceRegion:` of level 0 and slice `slice` (the module's docs): writes `region`'s rows of
    /// `bytes`, `bytes_per_row` apart, into the buffer's bytes at the texture's rows, names them to
    /// rt where the buffer is Managed, and makes the next use take the bytes. `Err` with the kind
    /// of refusal and why, which writes nothing: Metal's validation's stops (a Private texture, a
    /// slice past 0, a region outside the level or off the format's blocks, source rows that do not
    /// cover the region), and a buffer whose bytes the host has no mapping of.
    pub(super) fn replace(
        &self,
        region: MtlRegion,
        slice: u64,
        bytes: &[u8],
        bytes_per_row: u64,
    ) -> Result<(), (RtRefusal, String)> {
        let refused = |why: String| Err((RtRefusal::Refused, why));
        let d = self.texture.descriptor();
        if d.storage_mode == rt::StorageMode::Private {
            return refused("a Private texture's bytes".to_owned());
        }
        let (block_bytes, [bw, bh]) = block(&self.device, u64::from(d.pixel_format.as_raw()));
        let (width, height) = (u64::from(d.width), u64::from(d.height));
        let MtlRegion { origin, size } = region;
        let fits = |at: u64, of: u64, side: u64, unit: u64| {
            at.checked_add(of).is_some_and(|end| {
                end <= side && at.is_multiple_of(unit) && (of.is_multiple_of(unit) || end == side)
            })
        };
        if slice != 0
            || block_bytes == 0
            || origin.z != 0
            || size.depth != 1
            || !fits(origin.x, size.width, width, bw)
            || !fits(origin.y, size.height, height, bh)
        {
            return refused(format!(
                "the region {region:?} of slice {slice} of a {width}x{height} texture"
            ));
        }
        let row = size.width.div_ceil(bw) * block_bytes;
        let rows = size.height.div_ceil(bh);
        // An empty region writes nothing, and the next use has nothing new to take.
        if row == 0 || rows == 0 {
            return Ok(());
        }
        let source_end = ((rows - 1).checked_mul(bytes_per_row)).and_then(|b| b.checked_add(row));
        if bytes_per_row < row || source_end.is_none_or(|end| end > bytes.len().to_u64()) {
            return refused(format!(
                "{} bytes in rows {bytes_per_row} apart for {rows} rows of {row} bytes",
                bytes.len()
            ));
        }
        let first = (origin.y / bh)
            .checked_mul(self.bytes_per_row)
            .and_then(|b| b.checked_add((origin.x / bw) * block_bytes))
            .and_then(|b| b.checked_add(self.offset));
        let end = first.and_then(|f| {
            ((rows - 1).checked_mul(self.bytes_per_row))
                .and_then(|b| b.checked_add(row))
                .and_then(|b| b.checked_add(f))
        });
        let (Some(first), Some(end)) = (first, end) else {
            return refused(format!("the region {region:?} past the buffer"));
        };
        if end > self.buffer.length() {
            return refused(format!("the region {region:?} past the buffer"));
        }
        let contents = self.buffer.contents();
        if contents.is_null() {
            let why = "a buffer whose bytes the host has no mapping of".to_owned();
            return Err((RtRefusal::Other, why));
        }
        let to = contents.as_ptr::<u8>();
        for r in 0..rows {
            let from = (r * bytes_per_row).to_usize();
            let at = (first + r * self.bytes_per_row).to_usize();
            // SAFETY: `from..from + row` lies within `bytes` (`source_end` above) and
            // `at..at + row` within the buffer's `length` bytes (`end` above), which its contents
            // keep mapped while the buffer lives, and the texture holds the buffer. `bytes` is the
            // platform's own copy of the guest's, so the two do not overlap. The write goes
            // through raw pointers alone: no reference covers the buffer's bytes, which the guest
            // may write meanwhile.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(from), to.add(at), row.to_usize());
            }
        }
        let written = NsRange {
            location: first,
            length: end - first,
        };
        let _ = self.buffer.did_modify_range(OnFault::Return, written);
        self.modified();
        Ok(())
    }
}

/// The bytes of one of `raw`'s blocks and its width and height in texels, from Metal's description
/// (rt's `pixel_format_info`): of a format not of blocks, a texel's bytes and one texel; 0 bytes for
/// a value Metal describes no format for.
fn block(device: &rt::Device, raw: u64) -> (u64, [u64; 2]) {
    let info = device.pixel_format_info(raw);
    let [bytes, width, height, _] = info.words;
    match info.flags & BLOCKS != 0 {
        true => (bytes, [width.max(1), height.max(1)]),
        false => (bytes, [1, 1]),
    }
}

/// Why Metal's validation stops a tiled texture of `desc` over a buffer of `length` bytes, from
/// `offset`, rows `bytes_per_row` apart, as it is made (the module's docs): rows not a whole number
/// of the format's blocks apart or closer than a row, an offset not a whole number of blocks in, or
/// rows past the buffer. `None` where it stops none of them, and for a value Metal describes no
/// format for, which the make refuses.
pub(super) fn bad_layout(
    device: &rt::Device,
    desc: &TextureDesc,
    length: u64,
    offset: u64,
    bytes_per_row: u64,
) -> Option<String> {
    let (bytes, [bw, bh]) = block(device, desc.pixel_format);
    if bytes == 0 {
        return None;
    }
    let row = desc.width.div_ceil(bw).saturating_mul(bytes);
    if bytes_per_row < row || !bytes_per_row.is_multiple_of(bytes) {
        return Some(format!(
            "rows {bytes_per_row} bytes apart, for rows of {row} bytes in blocks of {bytes}"
        ));
    }
    if !offset.is_multiple_of(bytes) {
        return Some(format!("an offset of {offset} bytes, in blocks of {bytes}"));
    }
    let rows = desc.height.div_ceil(bh).max(1);
    let end = ((rows - 1).checked_mul(bytes_per_row))
        .and_then(|b| b.checked_add(row))
        .and_then(|b| b.checked_add(offset));
    match end {
        Some(end) if end <= length => None,
        _ => Some(format!(
            "{rows} rows {bytes_per_row} bytes apart from an offset of {offset}, past the \
             buffer's {length} bytes"
        )),
    }
}

/// A command buffer's tiled textures used while their bytes were to take, each once, until its
/// commit takes those whose bytes still are ([`Takes::take`]).
#[derive(Clone, Default)]
pub struct Takes(Arc<Mutex<Vec<Arc<Tiled>>>>);

impl Takes {
    /// The command buffer used `texture`: noted where it is tiled and its bytes are to take.
    pub fn used(&self, texture: &RtTexture) {
        if let Some(tiled) = texture.tiled() {
            self.used_tiled(tiled);
        }
    }

    /// The command buffer used the tiled texture `tiled`: noted where its bytes are to take.
    pub fn used_tiled(&self, tiled: &Arc<Tiled>) {
        if !tiled.stale.load(Ordering::Acquire) {
            return;
        }
        let mut noted = self.0.lock();
        if !noted.iter().any(|t| Arc::ptr_eq(t, tiled)) {
            noted.push(Arc::clone(tiled));
        }
    }

    /// The commit, on `device`: takes the bytes of each texture noted whose bytes are still to
    /// take, ahead of the command buffer (the module's docs). None is noted since.
    pub fn take(&self, device: &RtDevice) {
        let noted = std::mem::take(&mut *self.0.lock());
        let stale: Vec<Arc<Tiled>> = (noted.into_iter())
            .filter(|t| t.stale.swap(false, Ordering::AcqRel))
            .collect();
        if !stale.is_empty() {
            take(device.rt(), &stale);
        }
    }
}

/// Takes the bytes of each of `tiled` on `device`: one command buffer of a new queue, which holds
/// nothing, its copies and mipmaps encoded in order, committed. What rt refuses is counted under
/// its kind; the levels of a texture whose copy it refused are left as they are.
fn take(device: &rt::Device, tiled: &[Arc<Tiled>]) {
    let made = (device.new_command_queue(1)).and_then(|queue| queue.command_buffer());
    let command_buffer = match made {
        Ok(command_buffer) => command_buffer,
        Err(error) => {
            rt_refused(SEL, &error);
            return;
        }
    };
    let mut encoder = match command_buffer.blit_command_encoder() {
        Ok(encoder) => encoder,
        Err(error) => {
            rt_refused(SEL, &error);
            return;
        }
    };
    for t in tiled {
        let d = t.texture.descriptor();
        let layout = rt::BufferLayout {
            offset: t.offset,
            bytes_per_row: t.bytes_per_row,
            bytes_per_image: 0,
        };
        let level_0 = rt::TexturePlace::default();
        let size = [d.width, d.height, 1];
        match encoder.copy_from_buffer(t.buffer.rt(), layout, size, &t.texture, level_0) {
            Ok(()) => {
                if let Err(error) = encoder.generate_mipmaps(&t.texture) {
                    rt_refused_texture(SEL, &error);
                }
            }
            Err(error) => {
                rt_refused_texture(SEL, &error);
            }
        }
    }
    encoder.end_encoding();
    if let Err(error) = command_buffer.commit() {
        rt_refused(SEL, &error);
        return;
    }
    crate::answer::count_kind_by(TAKES_CODE, tiled.len().to_u64());
}
