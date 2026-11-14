//! The measurement harness of the YCbCr family: surfaces of the shapes the family's formats are
//! made over, filled with codes, and what a texture of any of the family's formats over one
//! reads on this platform, through the probe kernels (`ycbcr_probe.metal`), into
//! halves and into floats. The same probes run for every member: a surface's shape is its
//! OSType's ([`Shape`]), and the texture's format is a parameter of each probe ([`probes`]).
//! Floats show a texel's value at full precision; a kernel that reads into halves rounds it.
//! A surface is the buffer that stands for it ([`Surface`]), as the reims-graphics device's
//! surfaces reach this platform.
#[cfg(target_os = "macos")]
use crate::core::platform::values::SurfaceFacts;
#[cfg(target_os = "macos")]
use crate::gpu::device::VulkanDevice;
#[cfg(target_os = "macos")]
use crate::gpu::resource::tests::{StandIn, facts_of};
#[cfg(target_os = "macos")]
use crate::platform::VulkanPlatform;

/// A surface of the harness: the buffer that stands for it, with its facts and planes.
#[cfg(target_os = "macos")]
pub(crate) type Surface = StandIn<VulkanPlatform>;

/// How a surface of two planes lays out its samples: plane 0 the luma, plane 1 Cb and Cr
/// interleaved, each sample a code in the top bits of `container` bytes, or, `packed`, three
/// codes to a little-endian 32-bit word, at its bits 0, 10 and 20 (bits 30 and 31 unread): luma
/// code `x` of a row in word `x / 3`, the chroma pair `x` as codes `2 x` (Cb) and `2 x + 1`
/// (Cr).
#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Shape {
    /// The surface's OSType.
    pub ostype: [u8; 4],
    /// The chroma plane's subsampling: the luma texels per chroma texel across and down, (2, 2)
    /// for 4:2:0, (2, 1) for 4:2:2 and (1, 1) for 4:4:4.
    pub subsampling: (u64, u64),
    /// The bits of a code.
    pub bits: u32,
    /// The bytes a sample is stored in: of a packed shape, the 4 bytes of a word of three.
    pub container: u64,
    /// Whether three codes share each 32-bit word.
    pub packed: bool,
}

#[cfg(target_os = "macos")]
impl Shape {
    /// The shape CoreVideo gives `ostype`, one of the two-plane OSTypes measured.
    pub(crate) fn of(ostype: &[u8; 4]) -> Shape {
        let (subsampling, bits, container) = match ostype {
            b"420v" | b"420f" => ((2, 2), 8, 1),
            b"422v" | b"422f" => ((2, 1), 8, 1),
            b"444v" | b"444f" => ((1, 1), 8, 1),
            b"x420" | b"xf20" => ((2, 2), 10, 2),
            b"x422" | b"xf22" => ((2, 1), 10, 2),
            b"x444" | b"xf44" => ((1, 1), 10, 2),
            b"p420" | b"pf20" => ((2, 2), 10, 4),
            b"p422" | b"pf22" => ((2, 1), 10, 4),
            b"p444" | b"pf44" => ((1, 1), 10, 4),
            other => panic!("no shape for {}", String::from_utf8_lossy(other)),
        };
        Shape {
            ostype: *ostype,
            subsampling,
            bits,
            container,
            packed: container == 4,
        }
    }

    /// The chroma plane's size for a luma plane of `size`.
    pub(crate) fn chroma_size(&self, (width, height): (u64, u64)) -> (u64, u64) {
        (
            width.div_ceil(self.subsampling.0),
            height.div_ceil(self.subsampling.1),
        )
    }

    /// The bytes of a row of `codes` codes: a sample each, or a word of each three, packed.
    pub(crate) fn row_bytes(&self, codes: u64) -> u64 {
        match self.packed {
            true => codes.div_ceil(3) * 4,
            false => codes * self.container,
        }
    }

    /// The layout of a surface of this shape whose luma plane is `size`: its facts, its planes
    /// (each a first byte and bytes per row) and its length. Rows of a multiple of 64 bytes and
    /// plane 1 at a multiple of 4096, as CoreVideo lays them out; a packed row holds a word of
    /// three luma codes (3 texels in 4 bytes), or two words of three chroma pairs (3 texels in 8
    /// bytes).
    pub(crate) fn layout(&self, size: (u64, u64)) -> (SurfaceFacts, Vec<(u64, u64)>, u64) {
        let (width, height) = size;
        let (chroma_width, chroma_height) = self.chroma_size(size);
        let (luma_row, chroma_row) = (
            self.row_bytes(width).next_multiple_of(64),
            self.row_bytes(chroma_width * 2).next_multiple_of(64),
        );
        let chroma_offset = (luma_row * height).next_multiple_of(4096);
        let planes = vec![(0, luma_row), (chroma_offset, chroma_row)];
        let facts = facts_of(self.ostype, size, &planes);
        let length = (chroma_offset + chroma_row * chroma_height).next_multiple_of(16384);
        (facts, planes, length)
    }

    /// A surface of this shape whose luma plane is `size` ([`Shape::layout`]), zeros, the buffer
    /// that stands for it `d`'s.
    pub(crate) fn surface(&self, d: &VulkanDevice, size: (u64, u64)) -> Surface {
        let (facts, planes, length) = self.layout(size);
        Surface::shared(d, facts, planes, length)
    }
}

#[cfg(target_os = "macos")]
pub(crate) mod host;
#[cfg(target_os = "macos")]
pub(crate) use host::*;
