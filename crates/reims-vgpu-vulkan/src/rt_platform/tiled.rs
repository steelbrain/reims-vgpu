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
//!
//! What differs from the Macs, and from the Vulkan platform's runtime, which takes the bytes as the
//! using command runs: a CPU write after the commit and before the command, and a GPU write of the
//! bytes by an earlier command buffer the device still holds (at an event wait, or in a queue's
//! group), are in no level. `replaceRegion:` of level 0 writes the image, not the buffer's bytes,
//! and leaves the next use to take nothing: counted ([`REPLACED_FIRST_LEVEL_CODE`]), the process's
//! first with a line. A format rt makes no mipmaps of, or whose blits do not filter, takes level 0
//! alone, the later levels keeping what they hold. The arguments Metal's validation stops as the
//! texture is made (rows that do not cover a row, an offset off the format's texels, rows past the
//! buffer) are refused at the take: the copy encodes nothing. Each refusal is counted under rt's
//! kind, and each texture's take ([`TAKES_CODE`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_rt as rt;

use crate::core::num::ToU64 as _;
use crate::core::platform::Handle as _;
use crate::core::sync::Mutex;

use super::RtObj;
use super::buffer::RtBuffer;
use super::device::RtDevice;
use super::texture::RtTexture;
use super::unbuilt::{rt_refused, rt_refused_texture};

/// The selector whose texture's take a refusal names.
const SEL: &str = "newTiledTextureWithDescriptor:offset:bytesPerRow:";

/// The code of a tiled texture's take, counted per texture taken.
pub const TAKES_CODE: &str = "rt-tiled-takes";

/// The code of a `replaceRegion:` of a tiled texture's level 0, which writes the image, not the
/// buffer's bytes (the module's docs), counted per call.
pub const REPLACED_FIRST_LEVEL_CODE: &str = "rt-tiled-replace-first-level";

/// What a tiled texture takes its bytes from, and whether they are to take (the module's docs).
pub struct Tiled {
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
    /// rt's `texture`, taking the bytes of `buffer` from `offset`, rows `bytes_per_row` apart, at
    /// its first use.
    pub(super) fn new(
        texture: &rt::Texture,
        buffer: &RtObj<RtBuffer>,
        offset: u64,
        bytes_per_row: u64,
    ) -> Arc<Tiled> {
        Arc::new(Tiled {
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

/// Whether the process wrote [`REPLACED_FIRST_LEVEL_CODE`]'s line yet.
static REPLACED_SAID: AtomicBool = AtomicBool::new(false);

/// Counts a `replaceRegion:` of a tiled texture's level 0 ([`REPLACED_FIRST_LEVEL_CODE`]); the
/// process's first writes its line.
pub(super) fn replaced_first_level(sel: &'static str) {
    crate::answer::count_kind(REPLACED_FIRST_LEVEL_CODE);
    if !REPLACED_SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: "reims_vgpu_vulkan::rt_platform",
            "{sel}: of a tiled texture's level 0, the bytes go into its image, not into its \
             buffer's bytes, and its next use takes none, where Metal writes the buffer's bytes \
             and its next use takes them ({REPLACED_FIRST_LEVEL_CODE}), counted from here"
        );
    }
}
