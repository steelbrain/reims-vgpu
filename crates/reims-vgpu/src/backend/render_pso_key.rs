//! What decides `MTLRenderPipelineState` identity, in the two halves every
//! cache in this crate now uses: what a lookup borrows and what an entry
//! retains.
//!
//! # Why it is out here
//!
//! Nothing in this file names the `metal` crate — every field is a scalar, an
//! array of them, or a shader blob — and everything under `backend/metal/` is
//! `cfg`-ed out of the arm a Linux host builds. While this lived there, its two
//! `#[test]`s had never executed on any machine: the Vulkan arm does not compile
//! them, and `cargo test --target aarch64-apple-darwin --no-run` fails at the
//! link step for want of an Apple linker. The identity of a pipeline state is
//! the last thing that should be tested nowhere, and it is pure arithmetic. Same
//! argument, and the same fix, as [`crate::backend::hash`] and
//! [`crate::backend::blob`].
//!
//! The two array widths are the reason this could not move earlier: they were
//! spelled `REIMS_VGPU_METAL_MAX_ATTRS` and `REIMS_VGPU_METAL_MAX_COLOR_RTS`,
//! which live in a `constants` module that *does* name `metal`. They are taken
//! here from the decoder bounds each of those is `const`-asserted equal to at
//! its own declaration, so the widths cannot drift without a build failure and
//! nothing had to be dragged out of the gated tree.
//!
//! # Why the shader is not a number
//!
//! The key used to carry `vert_hash`/`frag_hash` beside `vert_len`/`frag_len`
//! and no copy of either blob, so `equal` decided two pipelines were one
//! pipeline on 128 bits of non-keyed FNV-1a over two lengths. A collision handed
//! a draw an `MTLRenderPipelineState` compiled from a shader it never submitted.
//! Nothing refused, nothing logged; the frame was simply wrong.
//!
//! The blobs now travel as [`crate::backend::blob::BlobKey`] and are retained as
//! `BlobIdentity`, so the digest picks the bucket and the bytes decide the hit.
//! That module carries the full argument for why this is a fix rather than a
//! wider hash. What is worth adding here is the size of the trade: a
//! `RenderPsoIdentity` sits beside an `MTLRenderPipelineState` that Metal
//! compiled from those same two blobs, so retaining them is a fraction of the
//! entry rather than a doubling of it.

use crate::backend::blob::{BlobIdentity, BlobKey};
use crate::backend::hash::hash_u64;
use crate::contract::fnv::FNV_OFFSET_BASIS;
use crate::runtime::decode::render::PASS_MAX_COLOR_ATTACHMENTS;
use crate::runtime::decode::resource::{MAX_VERTEX_ATTRS, MTL_COLOR_WRITE_MASK_ALL};

/// Every word of pipeline state, other than the two shaders, that Metal bakes
/// into an `MTLRenderPipelineState`.
///
/// `Clone` is derived. It used to be a hand-written `RenderPsoKeyClone` trait
/// carrying the comment "not Clone by default with arrays"; that was true before
/// const generics and has not been for years — the widest array here is 31.
#[derive(Clone)]
pub struct RenderPsoKey {
    /// The fold over every other field, assigned by [`RenderPsoKey::rehash`].
    ///
    /// It picks the bucket and decides nothing on its own:
    /// [`RenderPsoLookup::bucket`] folds the two shader digests in beside it,
    /// and [`RenderPsoKey::equal`] is what answers.
    pub key_hash: u64,
    pub attr_count: u32,
    pub attr_location: [u32; MAX_VERTEX_ATTRS],
    pub attr_format: [u32; MAX_VERTEX_ATTRS],
    pub attr_offset: [u32; MAX_VERTEX_ATTRS],
    pub attr_buffer_index: [u32; MAX_VERTEX_ATTRS],
    pub attr_stride: [u32; MAX_VERTEX_ATTRS],
    /// Resolved step state, not the record's optionals: the caller has already
    /// applied the absent-field defaults, so the presence bits that used to sit
    /// beside these could only ever repeat what they had been folded into.
    pub attr_step_function: [u32; MAX_VERTEX_ATTRS],
    pub attr_step_rate: [u32; MAX_VERTEX_ATTRS],
    pub blend_enable: u8,
    pub blend_src_rgb: u32,
    pub blend_dst_rgb: u32,
    pub blend_op_rgb: u32,
    pub blend_src_alpha: u32,
    pub blend_dst_alpha: u32,
    pub blend_op_alpha: u32,
    /// Number of active color RTs (`0..=PASS_MAX_COLOR_ATTACHMENTS`). Slot
    /// `i` uses `color_formats[i]` — backticked because a bare `[i]` is link
    /// syntax, and rustdoc was reporting an unresolved link to `i`. The sibling
    /// field below already spelled it this way.
    pub color_count: u32,
    pub color_formats: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    pub color_slot: [u8; PASS_MAX_COLOR_ATTACHMENTS],
    /// Per-RT blend enable + factors (aligned with color_count entries).
    pub color_blend_enable: [u8; PASS_MAX_COLOR_ATTACHMENTS],
    pub color_blend_src_rgb: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    pub color_blend_dst_rgb: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    pub color_blend_op_rgb: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    pub color_blend_src_alpha: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    pub color_blend_dst_alpha: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    pub color_blend_op_alpha: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    /// Per-RT `MTLColorWriteMask`, in Metal's own bit order.
    ///
    /// Outside the `color_blend_*` group on purpose: the mask applies whether
    /// or not the slot blends, so it is keyed and applied unconditionally
    /// while the blend fields are only meaningful under
    /// `color_blend_enable[i]`.
    pub color_write_mask: [u32; PASS_MAX_COLOR_ATTACHMENTS],
    pub depth_pixel_format: u32,
    pub stencil_pixel_format: u32,
}

impl Default for RenderPsoKey {
    fn default() -> Self {
        Self {
            key_hash: 0,
            attr_count: 0,
            attr_location: [0; MAX_VERTEX_ATTRS],
            attr_format: [0; MAX_VERTEX_ATTRS],
            attr_offset: [0; MAX_VERTEX_ATTRS],
            attr_buffer_index: [0; MAX_VERTEX_ATTRS],
            attr_stride: [0; MAX_VERTEX_ATTRS],
            attr_step_function: [0; MAX_VERTEX_ATTRS],
            attr_step_rate: [0; MAX_VERTEX_ATTRS],
            blend_enable: 0,
            blend_src_rgb: 0,
            blend_dst_rgb: 0,
            blend_op_rgb: 0,
            blend_src_alpha: 0,
            blend_dst_alpha: 0,
            blend_op_alpha: 0,
            color_count: 0,
            color_formats: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_slot: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_blend_enable: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_blend_src_rgb: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_blend_dst_rgb: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_blend_op_rgb: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_blend_src_alpha: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_blend_dst_alpha: [0; PASS_MAX_COLOR_ATTACHMENTS],
            color_blend_op_alpha: [0; PASS_MAX_COLOR_ATTACHMENTS],
            // `MTLColorWriteMaskAll`. Zero here would mean a default-built key
            // describes a pipeline that writes no channel at all.
            color_write_mask: [MTL_COLOR_WRITE_MASK_ALL; PASS_MAX_COLOR_ATTACHMENTS],
            depth_pixel_format: 0,
            stencil_pixel_format: 0,
        }
    }
}

impl RenderPsoKey {
    /// Attribute slots this key holds, which is not always the number it names.
    ///
    /// `attr_count` is the guest's **untruncated** count, deliberately — see
    /// `backend::metal::constants::REIMS_VGPU_METAL_MAX_ATTRS`, which explains
    /// that clamping it would let two descriptors differing only past the table
    /// compare equal, and a wrong pipeline is worse than a refused one. So the
    /// count can name more attributes than the arrays hold, and the walks below
    /// stop at the arrays instead.
    ///
    /// Nothing is lost by that and no attribute is skipped: `attr_count` is
    /// itself compared and folded, so a key naming 32 never matches or buckets
    /// with one naming 31. What it removes is an index panic reachable only
    /// because the guard that makes it unreachable — `make_vertex_descriptor`
    /// refusing above the table — lives in another module and another crate
    /// feature. This walk should not depend on it.
    fn active_attrs(&self) -> usize {
        (self.attr_count as usize).min(MAX_VERTEX_ATTRS)
    }

    /// Attachment slots this key holds. `color_count` is clamped by its builder
    /// rather than carried untruncated, so this agrees with it today; it is here
    /// for the same reason as [`Self::active_attrs`].
    fn active_colors(&self) -> usize {
        (self.color_count as usize).min(PASS_MAX_COLOR_ATTACHMENTS)
    }

    /// Fold every field into [`Self::key_hash`].
    ///
    /// Folded off the key's own fields rather than off parallel locals, so the
    /// hash cannot describe a key different from the one it is stored with.
    /// Every declared field must appear here *and* in [`Self::equal`]: a field
    /// in one and not the other is a latent bug, not half a fix.
    pub fn rehash(&mut self) {
        let mut h = FNV_OFFSET_BASIS;
        h = hash_u64(h, self.attr_count as u64);
        for i in 0..self.active_attrs() {
            h = hash_u64(h, self.attr_location[i] as u64);
            h = hash_u64(h, self.attr_format[i] as u64);
            h = hash_u64(h, self.attr_offset[i] as u64);
            h = hash_u64(h, self.attr_buffer_index[i] as u64);
            h = hash_u64(h, self.attr_stride[i] as u64);
            h = hash_u64(h, self.attr_step_function[i] as u64);
            h = hash_u64(h, self.attr_step_rate[i] as u64);
        }
        h = hash_u64(h, self.blend_enable as u64);
        h = hash_u64(h, self.blend_src_rgb as u64);
        h = hash_u64(h, self.blend_dst_rgb as u64);
        h = hash_u64(h, self.blend_op_rgb as u64);
        h = hash_u64(h, self.blend_src_alpha as u64);
        h = hash_u64(h, self.blend_dst_alpha as u64);
        h = hash_u64(h, self.blend_op_alpha as u64);
        h = hash_u64(h, self.color_count as u64);
        for i in 0..self.active_colors() {
            h = hash_u64(h, self.color_slot[i] as u64);
            h = hash_u64(h, self.color_formats[i] as u64);
            h = hash_u64(h, self.color_blend_enable[i] as u64);
            h = hash_u64(h, self.color_blend_src_rgb[i] as u64);
            h = hash_u64(h, self.color_blend_dst_rgb[i] as u64);
            h = hash_u64(h, self.color_blend_op_rgb[i] as u64);
            h = hash_u64(h, self.color_blend_src_alpha[i] as u64);
            h = hash_u64(h, self.color_blend_dst_alpha[i] as u64);
            h = hash_u64(h, self.color_blend_op_alpha[i] as u64);
            h = hash_u64(h, self.color_write_mask[i] as u64);
        }
        h = hash_u64(h, self.depth_pixel_format as u64);
        h = hash_u64(h, self.stencil_pixel_format as u64);
        self.key_hash = h;
    }

    /// The full comparison of the descriptor half. The shaders are compared by
    /// [`RenderPsoIdentity::is`], which calls this.
    pub fn equal(&self, other: &Self) -> bool {
        if self.key_hash != other.key_hash
            || self.attr_count != other.attr_count
            || self.blend_enable != other.blend_enable
            || self.blend_src_rgb != other.blend_src_rgb
            || self.blend_dst_rgb != other.blend_dst_rgb
            || self.blend_op_rgb != other.blend_op_rgb
            || self.blend_src_alpha != other.blend_src_alpha
            || self.blend_dst_alpha != other.blend_dst_alpha
            || self.blend_op_alpha != other.blend_op_alpha
            || self.color_count != other.color_count
            || self.depth_pixel_format != other.depth_pixel_format
            || self.stencil_pixel_format != other.stencil_pixel_format
        {
            return false;
        }
        for i in 0..self.active_colors() {
            if self.color_formats[i] != other.color_formats[i]
                || self.color_slot[i] != other.color_slot[i]
                || self.color_blend_enable[i] != other.color_blend_enable[i]
                || self.color_blend_src_rgb[i] != other.color_blend_src_rgb[i]
                || self.color_blend_dst_rgb[i] != other.color_blend_dst_rgb[i]
                || self.color_blend_op_rgb[i] != other.color_blend_op_rgb[i]
                || self.color_blend_src_alpha[i] != other.color_blend_src_alpha[i]
                || self.color_blend_dst_alpha[i] != other.color_blend_dst_alpha[i]
                || self.color_blend_op_alpha[i] != other.color_blend_op_alpha[i]
                || self.color_write_mask[i] != other.color_write_mask[i]
            {
                return false;
            }
        }
        for i in 0..self.active_attrs() {
            if self.attr_location[i] != other.attr_location[i]
                || self.attr_format[i] != other.attr_format[i]
                || self.attr_offset[i] != other.attr_offset[i]
                || self.attr_buffer_index[i] != other.attr_buffer_index[i]
                || self.attr_stride[i] != other.attr_stride[i]
                || self.attr_step_function[i] != other.attr_step_function[i]
                || self.attr_step_rate[i] != other.attr_step_rate[i]
            {
                return false;
            }
        }
        true
    }
}

/// A pipeline a caller is asking about: the descriptor state and both shaders,
/// all borrowed.
#[derive(Clone, Copy)]
pub struct RenderPsoLookup<'a> {
    pub desc: &'a RenderPsoKey,
    pub vert: BlobKey<'a>,
    pub frag: BlobKey<'a>,
}

impl RenderPsoLookup<'_> {
    /// The cache bucket: the descriptor fold with both shader digests folded in.
    ///
    /// The shader digests belong here rather than inside
    /// [`RenderPsoKey::key_hash`] because that is the only thing a digest is
    /// entitled to do once the bytes are retained — narrow the walk. Two
    /// pipelines sharing a vertex shader and a descriptor but differing in the
    /// fragment shader still land in different buckets, which is what keeps a
    /// bucket a handful of entries rather than a per-shader list.
    pub fn bucket(&self) -> u64 {
        hash_u64(hash_u64(self.desc.key_hash, self.vert.hash), self.frag.hash)
    }
}

/// The retained half: what a cache entry files a pipeline state under.
pub struct RenderPsoIdentity {
    pub key: RenderPsoKey,
    vert: BlobIdentity,
    frag: BlobIdentity,
}

impl RenderPsoIdentity {
    /// Retain everything `lookup` borrows. One copy of each shader, taken once
    /// per distinct pipeline.
    pub fn of(lookup: &RenderPsoLookup<'_>) -> Self {
        Self {
            key: lookup.desc.clone(),
            vert: BlobIdentity::of(&lookup.vert),
            frag: BlobIdentity::of(&lookup.frag),
        }
    }

    /// Lend this identity back as a lookup key, so an insert re-scans with
    /// exactly what it files rather than with a second key from the caller.
    pub fn as_lookup(&self) -> RenderPsoLookup<'_> {
        RenderPsoLookup {
            desc: &self.key,
            vert: self.vert.as_key(),
            frag: self.frag.as_key(),
        }
    }

    /// The full identity compare. This alone decides a hit.
    ///
    /// Shaders first: they reject on one word each before touching the
    /// descriptor's thirty fields, and they are the half a bucket collision is
    /// most likely to have already agreed on.
    pub fn is(&self, lookup: &RenderPsoLookup<'_>) -> bool {
        self.vert.is(&lookup.vert) && self.frag.is(&lookup.frag) && self.key.equal(lookup.desc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(desc: &RenderPsoKey, vert: &[u8], frag: &[u8]) -> RenderPsoIdentity {
        RenderPsoIdentity::of(&RenderPsoLookup {
            desc,
            vert: BlobKey::new(vert),
            frag: BlobKey::new(frag),
        })
    }

    fn counted(attrs: u32, colors: u32) -> RenderPsoKey {
        RenderPsoKey {
            attr_count: attrs,
            color_count: colors,
            ..Default::default()
        }
    }

    fn lookup<'a>(desc: &'a RenderPsoKey, vert: &'a [u8], frag: &'a [u8]) -> RenderPsoLookup<'a> {
        RenderPsoLookup {
            desc,
            vert: BlobKey::new(vert),
            frag: BlobKey::new(frag),
        }
    }

    /// The tail past `attr_count`/`color_count` is scratch, and comparing it
    /// would make two identical pipelines miss.
    #[test]
    fn render_key_compares_only_active_attachment_and_attribute_prefixes() {
        let mut left = RenderPsoKey::default();
        let mut right = RenderPsoKey::default();

        left.color_formats[7] = 70;
        right.color_formats[7] = 71;
        left.attr_location[30] = 30;
        right.attr_location[30] = 31;
        assert!(left.equal(&right), "inactive cache-key tails are ignored");

        left.color_count = 8;
        right.color_count = 8;
        assert!(
            !left.equal(&right),
            "an active attachment must affect equality"
        );
        right.color_formats[7] = left.color_formats[7];
        left.attr_count = 31;
        right.attr_count = 31;
        assert!(
            !left.equal(&right),
            "an active attribute must affect equality"
        );
        right.attr_location[30] = left.attr_location[30];
        assert!(left.equal(&right));
    }

    /// The shaders are the identity, and they are compared as bytes.
    ///
    /// This is what the key used to get wrong. It carried `vert_hash`,
    /// `frag_hash` and the two lengths and retained neither blob, so two
    /// distinct shaders of equal length whose digests collided were one
    /// pipeline. A natural collision is a 2^32 meet-in-the-middle, so drive the
    /// state one produces — same digest, different bytes — through the real
    /// compare.
    #[test]
    fn a_collided_shader_digest_is_not_the_same_pipeline() {
        let desc = RenderPsoKey::default();
        let vert: Vec<u8> = (0..64u8).collect();
        let frag: Vec<u8> = (0..64u8).map(|b| b ^ 0x33).collect();
        let id = identity(&desc, &vert, &frag);

        assert!(id.is(&lookup(&desc, &vert, &frag)), "the same draw hits");

        let other_vert: Vec<u8> = (0..64u8).map(|b| b ^ 0x5a).collect();
        let collided = RenderPsoLookup {
            desc: &desc,
            vert: BlobKey {
                hash: BlobKey::new(&vert).hash,
                bytes: &other_vert,
            },
            frag: BlobKey::new(&frag),
        };
        assert!(
            !id.is(&collided),
            "a different vertex shader under a collided digest is a miss, not \
             somebody else's pipeline state"
        );
    }

    /// A different fragment shader is a different pipeline even when everything
    /// else agrees — the half a vertex-only compare would have missed.
    #[test]
    fn a_different_fragment_shader_is_a_different_pipeline() {
        let desc = RenderPsoKey::default();
        let vert: Vec<u8> = (0..32u8).collect();
        let frag: Vec<u8> = (0..32u8).map(|b| b ^ 1).collect();
        let id = identity(&desc, &vert, &frag);
        assert!(!id.is(&lookup(&desc, &vert, &vert)));
        assert!(id.is(&id.as_lookup()));
    }

    /// Descriptor state that differs puts two pipelines in different buckets as
    /// well as making them compare unequal. Both halves are required: a field
    /// only in the hash splits buckets that should share one, and a field only
    /// in the compare lets two different pipelines collide in a bucket and then
    /// be told apart too late.
    #[test]
    fn a_differing_descriptor_changes_the_bucket_and_the_compare() {
        let shader: Vec<u8> = (0..16u8).collect();
        let mut base = RenderPsoKey::default();
        base.rehash();

        let mutations: [fn(&mut RenderPsoKey); 4] = [
            |key| key.depth_pixel_format = 1,
            |key| key.stencil_pixel_format = 1,
            |key| key.blend_enable = 1,
            |key| {
                key.color_count = 1;
                key.color_formats[0] = 80;
            },
        ];
        for mutate in mutations {
            let mut changed = RenderPsoKey::default();
            mutate(&mut changed);
            changed.rehash();
            assert!(!base.equal(&changed), "the compare must separate them");
            assert_ne!(
                lookup(&base, &shader, &shader).bucket(),
                lookup(&changed, &shader, &shader).bucket(),
                "and so must the bucket"
            );
        }
    }

    /// Two pipelines sharing a descriptor and a vertex shader bucket apart on
    /// the fragment shader, which is what keeps a bucket short.
    #[test]
    fn the_bucket_separates_pipelines_that_differ_only_in_a_shader() {
        let desc = RenderPsoKey::default();
        let vert: Vec<u8> = (0..16u8).collect();
        let frag_a: Vec<u8> = (0..16u8).map(|b| b ^ 2).collect();
        let frag_b: Vec<u8> = (0..16u8).map(|b| b ^ 3).collect();
        assert_ne!(
            lookup(&desc, &vert, &frag_a).bucket(),
            lookup(&desc, &vert, &frag_b).bucket()
        );
        assert_ne!(
            lookup(&desc, &frag_a, &vert).bucket(),
            lookup(&desc, &vert, &frag_a).bucket(),
            "and the two stages are not interchangeable in the fold"
        );
    }

    /// An `attr_count` past the table walks the table, not past it.
    ///
    /// The guest's count is stored untruncated on purpose, so nothing in this
    /// type stops it naming 32 attributes into 31-wide arrays. Before this walk
    /// clamped, that was an index panic held off only by
    /// `make_vertex_descriptor` refusing first — in a different module, behind a
    /// different crate feature, with nothing tying the two together.
    #[test]
    fn an_attribute_count_past_the_table_does_not_index_past_it() {
        let mut over = counted(u32::MAX, 0);
        let mut also_over = counted(u32::MAX, 0);
        over.rehash();
        also_over.rehash();
        assert!(over.equal(&also_over));

        // And the count still separates, so clamping the walk merges nothing.
        let mut one_less = counted(MAX_VERTEX_ATTRS as u32, 0);
        one_less.rehash();
        assert!(!over.equal(&one_less));
        assert_ne!(over.key_hash, one_less.key_hash);
    }

    /// The same for `color_count`, whose builder clamps it — so this is the
    /// guard holding rather than the guard being needed.
    #[test]
    fn an_attachment_count_past_the_table_does_not_index_past_it() {
        let mut over = counted(0, u32::MAX);
        over.rehash();
        let mut same = counted(0, u32::MAX);
        same.rehash();
        assert!(over.equal(&same));
    }
}
