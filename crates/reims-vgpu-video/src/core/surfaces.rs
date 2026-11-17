//! A guest surface's layout as a backend reaches it ([`SurfaceLayout`]).

use crate::core::fourcc::FourCc;

/// A surface's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceLayout {
    /// The CoreVideo pixel format, such as `420v`.
    pub pixel_format: FourCc,
    /// The width in pixels.
    pub width: u32,
    /// The height in pixels.
    pub height: u32,
    /// The planes, in plane order; empty for a surface with one plane
    /// that starts at offset 0 (then [`SurfaceLayout::bytes_per_row`] and
    /// the height describe it).
    pub planes: Vec<PlaneLayout>,
    /// The bytes per row of a surface without planes.
    pub bytes_per_row: usize,
}

/// One plane of a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    /// Where the plane starts in the surface's memory.
    pub offset: usize,
    /// The bytes from one row to the next.
    pub bytes_per_row: usize,
    /// The width in samples (half the surface's for a 4:2:0 chroma plane).
    pub width: u32,
    /// The height in rows.
    pub height: u32,
}

/// Why a surface could not be held or reached.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceError {
    /// The surface exists but its memory cannot be reached.
    #[error("surface memory unreachable: {0}")]
    Unreachable(String),
}

/// A region of a picture's crop, `[x, y, width, height]` in luma samples from the crop's corner,
/// as a crop of the picture: where `x` and `y` are even, as 4:2:0's crops start on a pair of
/// samples, and the region lies inside `crop` (`[x, y, width, height]` of the picture) and holds a
/// sample; `None` otherwise. Its width and height may be odd, as a crop's may.
pub fn crop_within(crop: [u32; 4], region: [u32; 4]) -> Option<[u32; 4]> {
    let [crop_x, crop_y, crop_width, crop_height] = crop;
    let [x, y, width, height] = region;
    let even = (x | y) & 1 == 0;
    let inside = (x.checked_add(width)).is_some_and(|end| end <= crop_width)
        && (y.checked_add(height)).is_some_and(|end| end <= crop_height);
    if !even || !inside || width == 0 || height == 0 {
        return None;
    }
    Some([
        crop_x.checked_add(x)?,
        crop_y.checked_add(y)?,
        width,
        height,
    ])
}

#[cfg(test)]
mod crop_tests {
    use super::crop_within;

    #[test]
    fn a_region_of_a_crop_starts_from_the_crops_corner() {
        let crop = [0, 8, 1024, 1016];
        assert_eq!(crop_within(crop, [0, 0, 1024, 1016]), Some(crop));
        assert_eq!(
            crop_within(crop, [128, 64, 896, 952]),
            Some([128, 72, 896, 952])
        );
        assert_eq!(
            crop_within([0, 0, 1024, 1024], [512, 0, 511, 1023]),
            Some([512, 0, 511, 1023]),
            "an odd width and height"
        );
    }

    #[test]
    fn a_region_past_the_crop_from_an_odd_sample_or_empty_is_refused() {
        let crop = [0, 0, 1024, 1024];
        for region in [
            [0, 0, 1026, 1024],
            [2, 0, 1024, 1024],
            [1, 0, 512, 512],
            [0, 3, 512, 512],
            [0, 0, 0, 512],
            [u32::MAX - 1, 0, 2, 2],
        ] {
            assert_eq!(crop_within(crop, region), None, "{region:?}");
        }
    }
}
