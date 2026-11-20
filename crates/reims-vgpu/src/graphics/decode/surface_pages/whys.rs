//! Why the guest found a read-only pages block's pages not unchanging (`RGFX_SURFACE_PAGES_WHY_SHIFT`,
//! the module doc, "Unchanging pages"): a diagnostic, which the host counts by and does nothing
//! else with, the guest seeing nothing of it. The views count the read-only blocks that are not
//! unchanging by it ([`WhyCounts`], logged with the views' summary), and the decoder each texture
//! made over such a block ([`note_texture`]): by why ([`TEXTURE_CODES`]), the first of each pixel
//! format, size and why logged with the bytes and page runs of the texture's range.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use reims_vgpu_graphics::wire::{
    RGFX_PAGES_WHY_NOT_A_FILE, RGFX_PAGES_WHY_OTHER, RGFX_PAGES_WHY_REFUSED,
    RGFX_PAGES_WHY_SHADOWED, RGFX_PAGES_WHY_SPANS_REGIONS, RGFX_PAGES_WHY_UNSAID,
    RGFX_PAGES_WHY_WRITABLE_MAPPING, RGFX_PAGES_WHY_WRITABLE_MAX_PROTECTION,
    RGFX_PAGES_WHY_WRITABLE_VOLUME, RGFX_SURFACE_PAGES_WHY_MASK, RGFX_SURFACE_PAGES_WHY_SHIFT,
    RgfxTextureDescriptor,
};
use reims_vgpu_vulkan::core::sync::Mutex;

/// How many whys the flags' bits hold.
pub(crate) const WHYS: usize = 16;

/// The why `flags`, a `SURFACE_PAGES`' [`reims_vgpu_graphics::wire::RgfxSurfacePagesCache::flags`],
/// carry: a value below [`WHYS`].
pub(crate) fn of_flags(flags: u32) -> u8 {
    // At most 15, which the mask's four bits hold.
    ((flags & RGFX_SURFACE_PAGES_WHY_MASK) >> RGFX_SURFACE_PAGES_WHY_SHIFT) as u8
}

/// `why` in words, as the guest's `RGFX_PAGES_WHY_*` values name it.
pub(crate) fn name(why: u8) -> &'static str {
    match u32::from(why) {
        RGFX_PAGES_WHY_UNSAID => "unsaid",
        RGFX_PAGES_WHY_NOT_A_FILE => "not a file",
        RGFX_PAGES_WHY_SPANS_REGIONS => "spans regions",
        RGFX_PAGES_WHY_WRITABLE_MAPPING => "writable mapping",
        RGFX_PAGES_WHY_WRITABLE_MAX_PROTECTION => "writable max protection",
        RGFX_PAGES_WHY_SHADOWED => "shadowed",
        RGFX_PAGES_WHY_WRITABLE_VOLUME => "writable volume",
        RGFX_PAGES_WHY_OTHER => "other",
        RGFX_PAGES_WHY_REFUSED => "refused unchanging",
        _ => "of no name",
    }
}

/// The read-only pages blocks that are not unchanging that a guest made with one why, as their
/// `SURFACE_PAGES` gave them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct WhyCounts {
    /// The blocks.
    pub(crate) blocks: u64,
    /// Their bytes.
    pub(crate) bytes: u64,
    /// Their page runs.
    pub(crate) runs: u64,
    /// The most page runs of one.
    pub(crate) most_runs: u64,
}

impl WhyCounts {
    /// Counts a block of `bytes` in `runs` page runs.
    pub(crate) fn note(&mut self, bytes: u64, runs: u64) {
        self.blocks += 1;
        self.bytes = self.bytes.saturating_add(bytes);
        self.runs = self.runs.saturating_add(runs);
        self.most_runs = self.most_runs.max(runs);
    }
}

/// The line the views' summary logs of `whys`, by why; `None` where no block was counted.
pub(crate) fn line(whys: &[WhyCounts; WHYS]) -> Option<String> {
    let mut line = String::from("read-only pages blocks not unchanging, by the guest's why:");
    let mut any = false;
    for (why, c) in (0_u8..).zip(whys).filter(|(_, c)| c.blocks != 0) {
        let _ = write!(
            line,
            "{} {why} ({}): {} blocks of {} KiB in {} page runs, at most {} runs a block",
            if any { ";" } else { "" },
            name(why),
            c.blocks,
            c.bytes.div_ceil(1024),
            c.runs,
            c.most_runs
        );
        any = true;
    }
    any.then_some(line)
}

/// The kinds each texture made over a read-only pages block that is not unchanging is counted
/// as, once each, in the degradation counts' line, by its block's why (the index).
pub(crate) const TEXTURE_CODES: [&str; WHYS] = [
    "read-only-pages-textures-why-0",
    "read-only-pages-textures-why-1",
    "read-only-pages-textures-why-2",
    "read-only-pages-textures-why-3",
    "read-only-pages-textures-why-4",
    "read-only-pages-textures-why-5",
    "read-only-pages-textures-why-6",
    "read-only-pages-textures-why-7",
    "read-only-pages-textures-why-8",
    "read-only-pages-textures-why-9",
    "read-only-pages-textures-why-10",
    "read-only-pages-textures-why-11",
    "read-only-pages-textures-why-12",
    "read-only-pages-textures-why-13",
    "read-only-pages-textures-why-14",
    "read-only-pages-textures-why-15",
];

/// The bytes of a texture's range over a read-only pages block that is not unchanging, and the
/// block's ([`super::SurfaceViews::read_only_range`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TextureRange {
    /// The block's why.
    pub(crate) why: u8,
    /// The bytes of the 4 KiB pages the range touches.
    pub(crate) bytes: u64,
    /// The block's page runs the range touches.
    pub(crate) runs: u64,
    /// The block's bytes.
    pub(crate) block_bytes: u64,
    /// The block's page runs.
    pub(crate) block_runs: u64,
}

/// How many pixel formats, sizes and whys of textures [`note_texture`] logs at most.
const TEXTURES_LOGGED_MOST: usize = 64;

/// The pixel formats, sizes and whys of the textures [`note_texture`] logged.
static TEXTURES_LOGGED: Mutex<BTreeSet<(u32, u32, u32, u8)>> = Mutex::new(BTreeSet::new());

/// Counts a texture of `desc` made over `range` of a read-only pages block that is not unchanging
/// ([`TEXTURE_CODES`]); the first of each pixel format, size and why is logged, of at most
/// [`TEXTURES_LOGGED_MOST`].
pub(crate) fn note_texture(desc: &RgfxTextureDescriptor, range: &TextureRange) {
    let code = TEXTURE_CODES.get(usize::from(range.why)).copied();
    if let Some(code) = code {
        reims_vgpu_vulkan::answer::count_kind(code);
    }
    let key = (desc.pixel_format, desc.width, desc.height, range.why);
    let first = {
        let mut logged = TEXTURES_LOGGED.lock();
        logged.len() < TEXTURES_LOGGED_MOST && logged.insert(key)
    };
    if first {
        log::info!(
            target: "reims_vgpu::graphics",
            "a {}x{} texture of pixel format {} over a read-only pages block not unchanging (why \
             {}, {}): {} KiB of its pages in {} page runs, of the block's {} KiB in {} runs; \
             counted as {}",
            desc.width,
            desc.height,
            desc.pixel_format,
            range.why,
            name(range.why),
            range.bytes.div_ceil(1024),
            range.runs,
            range.block_bytes.div_ceil(1024),
            range.block_runs,
            code.unwrap_or("none")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `SURFACE_PAGES`' flags carry their why in bits 8 to 11, apart from every flag; each of
    /// the guest's values has a name.
    #[test]
    fn the_why_is_the_flags_bits_8_to_11() {
        assert_eq!(of_flags(0), 0);
        assert_eq!(
            of_flags((RGFX_PAGES_WHY_SHADOWED << RGFX_SURFACE_PAGES_WHY_SHIFT) | 0xff),
            5
        );
        assert_eq!(of_flags(u32::MAX), 15);
        assert_eq!(usize::from(of_flags(u32::MAX)) + 1, WHYS);
        for why in 0..=RGFX_PAGES_WHY_REFUSED {
            assert_ne!(name(why as u8), "of no name", "why {why}");
        }
        assert_eq!(name(9), "of no name");
    }

    /// The summary's line counts each why's blocks, bytes and runs and the most runs of one, and
    /// is none where no block was counted.
    #[test]
    fn the_line_counts_each_why() {
        let mut whys = [WhyCounts::default(); WHYS];
        assert_eq!(line(&whys), None);
        whys[5].note(8192, 2);
        whys[5].note(4096, 5);
        whys[4].note(1000, 1);
        assert_eq!(
            whys[5],
            WhyCounts {
                blocks: 2,
                bytes: 12288,
                runs: 7,
                most_runs: 5
            }
        );
        assert_eq!(
            line(&whys).as_deref(),
            Some(
                "read-only pages blocks not unchanging, by the guest's why: 4 (writable max \
                 protection): 1 blocks of 1 KiB in 1 page runs, at most 1 runs a block; 5 \
                 (shadowed): 2 blocks of 12 KiB in 7 page runs, at most 5 runs a block"
            )
        );
    }
}
