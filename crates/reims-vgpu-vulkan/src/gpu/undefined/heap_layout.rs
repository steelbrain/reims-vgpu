//! `heap-texture-lossless-layout`: the bytes a single-sample texture placed in a heap leaves in
//! the heap's memory, as a buffer over the same bytes reads them.
//!
//! **Metal's rule** (this host, every probe below the same in each of 5 runs, each run a
//! new heap over new pages filled with `0xa5` first): a 64 by 64 `RGBA8Unorm` render target placed
//! at offset 0 of a `Shared` placement heap over host pages, read from the pages:
//! - **Deterministic.** The same texels written the same way leave the same bytes.
//! - **Uncompressed** (`allowGPUOptimizedContents` off, `ShaderWrite` usage, or a shorter side
//!   under 16): the texels' own bytes, in Metal's tiles (a permutation of the texels:
//!   the runtime's `texture/heap_tiled.rs` layout), and nothing else.
//! - **Compressed** (Metal's lossless compression: `allowGPUOptimizedContents`, the shorter side
//!   16 or more, neither `ShaderWrite` nor `PixelFormatView` usage): the texture takes the
//!   uncompressed size plus 8 bytes of metadata per 16 by 16 block (16512 bytes here). The bytes
//!   depend on how the texels were written, not only on the texels:
//!   - a GPU blit, or a pass that stores, of texels that compress (an index pattern, one colour)
//!     leaves bytes that are no permutation of the texels: each 16 by 16 block holds a short
//!     encoding at the start of its 1024 bytes, and the rest of the block keeps what the heap's
//!     memory held before (`0xa5` here);
//!   - a GPU blit of texels that do not compress (random bytes) leaves the texels in Metal's
//!     tiles;
//!   - a CPU `replaceRegion` leaves the texels in Metal's tiles, the uncompressed texture's
//!     bytes, whatever they are, its metadata marking each block so.
//!
//!   The encoding is Apple's and undocumented. An `ObjC` probe of this host saw: each block's 8
//!   metadata bytes one value repeated (`0x7f` for a block held as texels, `0x03` for one colour,
//!   others for an encoding); a one-colour block's payload the colour 8 times then 32 zero
//!   bytes; an index pattern's a header of 8 bytes then each channel's values at fewer bits.
//!   Storage mode decides nothing: a Private placement heap read through an aliasing buffer held
//!   the same bytes as a Shared one.
//!
//! **Not probed**: other formats (an `ObjC` probe of `RGBA16Float` and `Depth32Float` found them
//! compressed alike, not recorded here), mipmapped, array and multisample textures, levels past
//! the first; offsets other than 0; other Apple GPU generations.
//!
//! **This platform today**: a texture in a heap over host pages lies in the device's linear image
//! bound to the pages wherever one serves, in the driver's rows with no copy, announcing
//! `heap-texture-untiled-layout`: its bytes are its texels row by row at the driver's pitch, not
//! Metal's tiles, compressed or not. Where no such image serves, an uncompressed texture is a
//! shadow of its texels in Metal's tiles, Metal's bytes, and a compressed one's bytes are in a
//! layout of the runtime's own, the metadata's bytes left as the heap held them, announcing
//! `heap-texture-lossless-layout`. Both kinds are relayed for a single-sample texture
//! ([`crate::answer::HEARD`]). The tests hold each shape's bytes to Metal's, or require the
//! difference to come with a relayed kind: `heap-texture-untiled-layout`, with the texels in
//! rows where the probe knows them, or, for the compressed shape, `heap-texture-lossless-layout`.
//!
//! **To answer as Metal**: in part, a compressed texture's texels laid out in Metal's tiles with
//! each block's metadata marked as held as texels (`0x7f`, measured for `RGBA8Unorm` alone) is
//! Metal's answer after a CPU write and after a blit of texels that do not compress. Every other
//! write needs Apple's encoder bit for bit: each block's choice of encoding and its bytes, for
//! every format and sample count, by every path that writes (blits, a pass's store, a clear),
//! with the block's unused bytes kept. No public specification of the encoding is known.

use crate::core::platform::Platform;
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::resource::Heap as _;
use crate::core::platform::types::ResourceAddressRange;
use crate::core::platform::values::{HeapDesc, OptionSet};
use metal2vulkan_runtime as rt;

use super::{
    Aspect, CLEAR, Pass, READ, RGBA8, STORE, Shape, Step, TARGET, WRITE, heard, random_bytes,
    replace, run,
};
use crate::gpu::device::tests::dev;
use crate::platform::tests::platform;

/// The heap's size: 64 KiB.
const HEAP: u64 = 0x10000;

/// How a probe writes the texels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Write {
    /// A GPU blit of each texel's index, a little-endian `u32`.
    BlitIndex,
    /// A GPU blit of one colour.
    BlitColour,
    /// A GPU blit of random bytes.
    BlitRandom,
    /// A CPU `replaceRegion` of each texel's index.
    CpuIndex,
    /// A pass's `Clear`, stored.
    Cleared,
}

/// The texels `write` writes, `length` bytes.
fn texels(write: Write, length: usize) -> Vec<u8> {
    match write {
        Write::BlitIndex | Write::CpuIndex => (0..u32::try_from(length / 4).unwrap())
            .flat_map(u32::to_le_bytes)
            .collect(),
        Write::BlitColour => [0x40, 0x50, 0x60, 0x70].repeat(length / 4),
        Write::BlitRandom => random_bytes(length, 53),
        Write::Cleared => Vec::new(),
    }
}

/// The heap's bytes of a texture of `shape` placed at offset 0 of a `Shared` placement heap over
/// new host pages, filled with `0xa5` first, after `write`, and the texture's size in the heap.
fn heap_bytes<P: Platform>((p, d): (&P, &P::Device), shape: Shape, write: Write) -> (Vec<u8>, u64) {
    let pages = p.vm().allocate_anywhere(HEAP).unwrap();
    let length = usize::try_from(HEAP).unwrap();
    // SAFETY: the new pages, `HEAP` bytes, which nothing else uses yet; left mapped until the
    // process ends, as the heap's alias shares them.
    unsafe { std::ptr::write_bytes(pages.as_ptr::<u8>(), 0xa5, length) };
    let ranges = p
        .new_address_range_array(&[ResourceAddressRange {
            address: u64::try_from(pages.addr()).unwrap(),
            length: HEAP,
        }])
        .unwrap();
    let heap = d
        .new_heap(&HeapDesc {
            size: HEAP,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
            address_ranges: Some(&ranges),
            pinned_gpu_address: None,
        })
        .unwrap()
        .heap
        .unwrap();
    let shape = Shape {
        options: heap.resource_options(),
        ..shape
    };
    let desc = p.texture_descriptor(&shape.desc());
    let size = d.heap_texture_size_and_align(&desc).unwrap().size;
    let t = heap.new_texture(&desc, 0).unwrap().unwrap();
    let bytes = texels(write, shape.length());
    match write {
        Write::CpuIndex => replace::<P>(&t, shape, &bytes),
        Write::Cleared => {
            run(
                p,
                d,
                &[Step::Pass(Pass::of(
                    &t,
                    Aspect::Colour,
                    (CLEAR, STORE),
                    Vec::new(),
                ))],
            );
        }
        _ => {
            run(p, d, &[Step::Upload(&t, shape, &bytes)]);
        }
    }
    // SAFETY: the heap's pages, read after every command that writes them completed.
    let held = unsafe { std::slice::from_raw_parts(pages.as_ptr::<u8>(), length) }.to_vec();
    drop(t);
    (held, size)
}

/// The shapes probed: compressed (Metal's default), with `allowGPUOptimizedContents` off, and of
/// `ShaderWrite` usage, `RGBA8Unorm` 64 by 64; and 16 by 8, too narrow to compress.
fn shapes() -> [(&'static str, Shape, bool); 4] {
    let base = Shape::of(RGBA8, (64, 64));
    [
        ("compressed", base, true),
        (
            "optimized off",
            Shape {
                optimized: Some(false),
                ..base
            },
            false,
        ),
        (
            "ShaderWrite",
            Shape {
                usage: READ | WRITE | TARGET,
                ..base
            },
            false,
        ),
        ("16 by 8", Shape::of(RGBA8, (16, 8)), false),
    ]
}

/// The pitch at which `held` holds `texels`, `rows` rows of `row` bytes each, row by row from its
/// start: the least multiple of 16 bytes, from `row` to 16 times it, at which every row's bytes
/// are the texels' own; `None` where none is.
fn pitch_of_rows(held: &[u8], texels: &[u8], row: usize, rows: usize) -> Option<usize> {
    (row..=16 * row).step_by(16).find(|&pitch| {
        (0..rows).all(|y| {
            let at = y * pitch;
            held.get(at..at + row) == Some(&texels[y * row..(y + 1) * row])
        })
    })
}

/// A texture placed in a `Shared` heap over host pages leaves bytes there that are one value for
/// the same writes, on this host's Metal in every run: an uncompressed texture's texels in
/// Metal's tiles; a compressed one's, after a GPU blit or a stored clear of texels that compress,
/// no permutation of them, each 16 by 16 block's bytes past its encoding keeping the heap's
/// earlier bytes, and after a blit of texels that do not compress, or a CPU `replaceRegion`, the
/// texels in Metal's tiles, the uncompressed texture's bytes. This platform's bytes are Metal's,
/// or they differ and the runtime announced why: `heap-texture-untiled-layout`, the texture in the
/// device's image over the pages, where the probe knows the texels, every row of them at one
/// pitch; or, for the compressed shape, `heap-texture-lossless-layout`
#[test]
fn heap_texture_bytes_are_metals_tiles_or_its_compressed_form() {
    let (p, d) = (platform(), dev());
    let writes = [
        Write::BlitIndex,
        Write::BlitColour,
        Write::BlitRandom,
        Write::CpuIndex,
        Write::Cleared,
    ];
    let mut differ = Vec::new();
    for (name, shape, compresses) in shapes() {
        for write in writes {
            let case = format!("{name} {write:?}");
            let (o, ours_size) = heap_bytes((&p, &d), shape, write);
            // The texture's size in the heap, Metal's.
            crate::metal_check!(format!("{case}: size"), ours_size);
            if !crate::metal_matches_bytes!(&case, &o[..usize::try_from(ours_size).unwrap()]) {
                let untiled = heard(rt::Degradation::HeapTextureUntiledLayout);
                let lossless = compresses && heard(rt::Degradation::HeapTextureLosslessLayout);
                assert!(untiled || lossless, "{case}: off Metal's, no kind heard");
                // The texels the probe wrote, where it knows them: a blit's or a CPU write's.
                let known = !matches!(write, Write::Cleared);
                if untiled && !lossless && known {
                    let texels = texels(write, shape.length());
                    let row = shape.length() / usize::try_from(shape.size.1).unwrap();
                    let rows = usize::try_from(shape.size.1).unwrap();
                    let pitch = pitch_of_rows(&o, &texels, row, rows);
                    assert!(
                        pitch.is_some(),
                        "{case}: off Metal's, not its texels in rows"
                    );
                    differ.push(format!(
                        "{case}: untiled, rows {} bytes apart",
                        pitch.unwrap()
                    ));
                } else {
                    differ.push(format!(
                        "{case}: {}",
                        if lossless { "lossless" } else { "untiled" }
                    ));
                }
            }
        }
    }
    eprintln!("off Metal's, with the kind heard: {differ:#?}");
}
