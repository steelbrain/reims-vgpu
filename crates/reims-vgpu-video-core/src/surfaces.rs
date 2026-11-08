//! A guest surface's layout as a backend reaches it ([`SurfaceLayout`]).

use crate::fourcc::FourCc;

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
