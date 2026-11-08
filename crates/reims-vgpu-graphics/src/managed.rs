//! The linear layout of a `Managed` texture's CPU copy ([`RgfxManagedTextureCreate`]), which the
//! guest's bundle (`guest/bundle/src/RgfxManagedLayout.hpp`) and the host each compute from the
//! create's fields alone, so neither tells the other where a level or a slice lies.
//!
//! Every level in order; each level its slices in order (the array's elements, six faces a cube);
//! each slice its images (the level's depth, for a 3D texture; else one); each image its rows of
//! texel blocks, each row's bytes rounded up to [`RGFX_MANAGED_ROW_ALIGN`]. Both tests hold the
//! same cases.

use crate::wire::{RGFX_MANAGED_ROW_ALIGN, RgfxManagedTextureCreate, RgfxTextureDescriptor};

/// `MTLTextureTypeCube`.
const TEXTURE_TYPE_CUBE: u32 = 5;
/// `MTLTextureTypeCubeArray`.
const TEXTURE_TYPE_CUBE_ARRAY: u32 = 6;
/// `MTLTextureType3D`.
const TEXTURE_TYPE_3D: u32 = 7;
/// The most levels a layout takes: a texture of 2^31 texels a side has 32.
const MAX_LEVELS: u32 = 32;

/// Where one level of a CPU copy lies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LevelLayout {
    /// Where its first slice starts.
    pub offset: u64,
    /// Its width, height and depth in texels.
    pub size: [u64; 3],
    /// The bytes from one row of blocks to the next.
    pub bytes_per_row: u64,
    /// The bytes from one image to the next: its rows of blocks.
    pub bytes_per_image: u64,
    /// The bytes from one slice to the next: its images.
    pub bytes_per_slice: u64,
}

/// A CPU copy's layout ([`LinearLayout::of`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearLayout {
    levels: Vec<LevelLayout>,
    slices: u64,
    bytes_per_block: u64,
    block: [u64; 2],
    size: u64,
}

impl LinearLayout {
    /// The layout of texture `desc`'s CPU copy, of blocks of `bytes_per_block` bytes and
    /// `block_width` by `block_height` texels. `None` for a shape no CPU copy has: a dimension or
    /// a count of 0, more than one sample, more levels than a side halves to, or a size past 2^64.
    pub fn new(
        desc: &RgfxTextureDescriptor,
        bytes_per_block: u32,
        block_width: u32,
        block_height: u32,
    ) -> Option<LinearLayout> {
        let (bpb, bw, bh) = (
            u64::from(bytes_per_block),
            u64::from(block_width),
            u64::from(block_height),
        );
        let three_d = desc.texture_type == TEXTURE_TYPE_3D;
        let (width, height) = (u64::from(desc.width), u64::from(desc.height));
        let depth = if three_d { u64::from(desc.depth) } else { 1 };
        let largest = width.max(height).max(depth);
        if bpb == 0
            || bw == 0
            || bh == 0
            || width == 0
            || height == 0
            || depth == 0
            || desc.array_length == 0
            || desc.sample_count > 1
            || desc.mipmap_level_count == 0
            || desc.mipmap_level_count > MAX_LEVELS
            || largest >> (desc.mipmap_level_count - 1) == 0
        {
            return None;
        }
        let faces = match desc.texture_type {
            TEXTURE_TYPE_CUBE | TEXTURE_TYPE_CUBE_ARRAY => 6,
            _ => 1,
        };
        let slices = u64::from(desc.array_length).checked_mul(faces)?;
        let mut levels = Vec::with_capacity(desc.mipmap_level_count as usize);
        let mut offset = 0u64;
        for level in 0..desc.mipmap_level_count {
            let size = [
                (width >> level).max(1),
                (height >> level).max(1),
                (depth >> level).max(1),
            ];
            let bytes_per_row = size[0]
                .div_ceil(bw)
                .checked_mul(bpb)?
                .checked_next_multiple_of(RGFX_MANAGED_ROW_ALIGN)?;
            let bytes_per_image = bytes_per_row.checked_mul(size[1].div_ceil(bh))?;
            let bytes_per_slice = bytes_per_image.checked_mul(size[2])?;
            levels.push(LevelLayout {
                offset,
                size,
                bytes_per_row,
                bytes_per_image,
                bytes_per_slice,
            });
            offset = offset.checked_add(bytes_per_slice.checked_mul(slices)?)?;
        }
        Some(LinearLayout {
            levels,
            slices,
            bytes_per_block: bpb,
            block: [bw, bh],
            size: offset,
        })
    }

    /// The layout a `MANAGED_TEXTURE_CREATE` names.
    pub fn of(create: &RgfxManagedTextureCreate) -> Option<LinearLayout> {
        LinearLayout::new(
            &create.desc,
            create.bytes_per_block,
            create.block_width,
            create.block_height,
        )
    }

    /// Level `level`, where the texture has it.
    pub fn level(&self, level: u64) -> Option<&LevelLayout> {
        self.levels.get(usize::try_from(level).ok()?)
    }

    /// The levels.
    pub fn levels(&self) -> &[LevelLayout] {
        &self.levels
    }

    /// The slices of every level.
    pub fn slices(&self) -> u64 {
        self.slices
    }

    /// The CPU copy's bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Where the texels of `size` at `origin` of `level` and `slice` lie: the offset of the
    /// first, and the bytes from it to the end of the last. `None` for an empty region, one past
    /// the level or the slices, or one whose origin is not at a block's start.
    pub fn region(
        &self,
        level: u64,
        slice: u64,
        origin: [u64; 3],
        size: [u64; 3],
    ) -> Option<(u64, u64)> {
        let l = self.level(level)?;
        let [bw, bh] = self.block;
        let fits = (0..3).all(|i| {
            size[i] > 0
                && origin[i]
                    .checked_add(size[i])
                    .is_some_and(|end| end <= l.size[i])
        });
        if !fits
            || slice >= self.slices
            || !origin[0].is_multiple_of(bw)
            || !origin[1].is_multiple_of(bh)
        {
            return None;
        }
        let offset = l.offset
            + slice * l.bytes_per_slice
            + origin[2] * l.bytes_per_image
            + origin[1] / bh * l.bytes_per_row
            + origin[0] / bw * self.bytes_per_block;
        let span = (size[2] - 1) * l.bytes_per_image
            + (size[1].div_ceil(bh) - 1) * l.bytes_per_row
            + size[0].div_ceil(bw) * self.bytes_per_block;
        Some((offset, span))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(
        texture_type: u32,
        size: [u32; 3],
        levels: u32,
        array_length: u32,
    ) -> RgfxTextureDescriptor {
        RgfxTextureDescriptor {
            pixel_format: 70,
            texture_type,
            width: size[0],
            height: size[1],
            depth: size[2],
            mipmap_level_count: levels,
            array_length,
            sample_count: 1,
            ..RgfxTextureDescriptor::default()
        }
    }

    /// (offset, bytes per row, bytes per image, bytes per slice) of each level.
    fn levels(layout: &LinearLayout) -> Vec<(u64, u64, u64, u64)> {
        (layout.levels().iter())
            .map(|l| {
                (
                    l.offset,
                    l.bytes_per_row,
                    l.bytes_per_image,
                    l.bytes_per_slice,
                )
            })
            .collect()
    }

    /// The cases `guest/bundle/tests/shell_tests.mm` holds too: a 2D RGBA8 texture with three
    /// levels, a BC1 one whose sides are no multiple of its blocks, a cube array, and a 3D one.
    #[test]
    fn lays_out_levels_slices_and_images_as_the_bundle_does() {
        let rgba = LinearLayout::new(&desc(2, [100, 60, 1], 3, 1), 4, 1, 1).unwrap();
        assert_eq!(
            levels(&rgba),
            [
                (0, 512, 30720, 30720),
                (30720, 256, 7680, 7680),
                (38400, 256, 3840, 3840)
            ]
        );
        assert_eq!((rgba.size(), rgba.slices()), (42240, 1));

        let bc1 = LinearLayout::new(&desc(2, [30, 30, 1], 2, 1), 8, 4, 4).unwrap();
        assert_eq!(
            levels(&bc1),
            [(0, 256, 2048, 2048), (2048, 256, 1024, 1024)]
        );

        let cubes = LinearLayout::new(&desc(6, [16, 16, 1], 2, 2), 4, 1, 1).unwrap();
        assert_eq!(cubes.slices(), 12);
        assert_eq!(
            levels(&cubes),
            [(0, 256, 4096, 4096), (49152, 256, 2048, 2048)]
        );
        assert_eq!(cubes.size(), 49152 + 12 * 2048);

        let volume = LinearLayout::new(&desc(7, [8, 8, 4], 2, 1), 4, 1, 1).unwrap();
        assert_eq!(
            levels(&volume),
            [(0, 256, 2048, 8192), (8192, 256, 1024, 2048)]
        );
        // A 2D texture's depth is 1, whatever the descriptor says.
        let flat = LinearLayout::new(&desc(2, [8, 8, 4], 1, 1), 4, 1, 1).unwrap();
        assert_eq!(flat.size(), 2048);
    }

    /// A region's first texel and the bytes it spans; none past the level, the slices, or off a
    /// block's start.
    #[test]
    fn finds_a_region_s_bytes() {
        let cubes = LinearLayout::new(&desc(6, [16, 16, 1], 2, 2), 4, 1, 1).unwrap();
        assert_eq!(
            cubes.region(1, 3, [2, 1, 0], [4, 3, 1]),
            Some((49152 + 3 * 2048 + 256 + 8, 2 * 256 + 16))
        );
        assert_eq!(cubes.region(1, 12, [0, 0, 0], [1, 1, 1]), None);
        assert_eq!(cubes.region(1, 0, [7, 0, 0], [2, 1, 1]), None);
        assert_eq!(cubes.region(2, 0, [0, 0, 0], [1, 1, 1]), None);
        assert_eq!(cubes.region(0, 0, [0, 0, 0], [0, 1, 1]), None);

        let bc1 = LinearLayout::new(&desc(2, [30, 30, 1], 1, 1), 8, 4, 4).unwrap();
        assert_eq!(
            bc1.region(0, 0, [4, 8, 0], [26, 22, 1]),
            Some((512 + 8, 5 * 256 + 56))
        );
        assert_eq!(bc1.region(0, 0, [2, 0, 0], [4, 4, 1]), None);

        let volume = LinearLayout::new(&desc(7, [8, 8, 4], 1, 1), 4, 1, 1).unwrap();
        assert_eq!(
            volume.region(0, 0, [0, 0, 1], [8, 8, 2]),
            Some((2048, 2048 + 7 * 256 + 32))
        );
    }

    /// No layout for a shape no CPU copy has.
    #[test]
    fn has_no_layout_of_a_shape_with_no_cpu_copy() {
        assert!(LinearLayout::new(&desc(2, [0, 8, 1], 1, 1), 4, 1, 1).is_none());
        assert!(LinearLayout::new(&desc(2, [8, 8, 1], 0, 1), 4, 1, 1).is_none());
        assert!(LinearLayout::new(&desc(2, [8, 8, 1], 5, 1), 4, 1, 1).is_none());
        assert!(LinearLayout::new(&desc(2, [8, 8, 1], 1, 0), 4, 1, 1).is_none());
        assert!(LinearLayout::new(&desc(2, [8, 8, 1], 1, 1), 0, 1, 1).is_none());
        let mut msaa = desc(4, [8, 8, 1], 1, 1);
        msaa.sample_count = 4;
        assert!(LinearLayout::new(&msaa, 4, 1, 1).is_none());
        assert!(LinearLayout::new(&desc(3, [65535, 65535, 1], 1, 65535), 16, 1, 1).is_some());
        assert!(
            LinearLayout::new(&desc(6, [u32::MAX, u32::MAX, 1], 1, u32::MAX), 16, 1, 1).is_none()
        );
    }
}
