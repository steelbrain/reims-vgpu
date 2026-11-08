//! A kernel's write of a slice of its explicit imageblock into a texture
//! (`air.write_imageblock_slice_to_texture_2d`), on Metal and on this platform, compared level by
//! level. macOS 26's glass blur writes its mip texture so, of usage ShaderRead | RenderTarget |
//! BlockWritesOnly (0x10005), at a level from its uniforms, and reads zeros where the write is
//! dropped. The kernels are `slice_write_repro.metal`'s, hand-written.

use crate::core::platform::Platform;
use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use crate::core::platform::device::GpuDevice as _;
use crate::core::platform::resource::{Buffer as _, Library as _, Texture as _};
use crate::core::platform::types::{MtlOrigin, MtlSize, NsRange};
use crate::core::platform::values::TextureDesc;
use crate::core::raise::OnFault;

use crate::gpu::device::tests::dev;
use crate::platform::tests::platform;

/// The library of `slice_write_repro.metal`.
const LIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/slice_write_repro.metallib"));

/// `MTLPixelFormatRGBA16Float`, 8 bytes a texel.
const RGBA16F: u64 = 115;

/// `MTLResourceStorageModePrivate`.
const PRIVATE: u64 = 0x20;

/// The written texture's side at level 0, and its levels.
const SIDE: u64 = 64;
const LEVELS: u64 = 3;

/// A Shared buffer of `length` bytes, `contents` at its start.
fn buffer<P: Platform>(d: &P::Device, length: u64, contents: &[u8]) -> P::Buffer {
    let b = d
        .new_buffer_of_length(OnFault::Die, length, 0)
        .unwrap()
        .unwrap();
    // SAFETY: a Shared buffer of at least `contents.len()` bytes, which no command uses yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            contents.as_ptr(),
            b.contents().as_ptr::<u8>(),
            contents.len(),
        );
    }
    b
}

/// Copies of each level of `t` into Shared buffers, by a blit encoder on `cb`.
fn read_levels<P: Platform>(
    d: &P::Device,
    cb: &P::CommandBuffer,
    t: &P::Texture,
) -> Vec<(P::Buffer, u64)> {
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let mut out = Vec::new();
    for level in 0..LEVELS {
        let (w, h) = ((t.width() >> level).max(1), (t.height() >> level).max(1));
        let n = w * h * 8;
        let b = buffer::<P>(d, n, &[]);
        let copy = BlitCmd::CopyTextureToBuffer {
            source_texture: t,
            source_slice: 0,
            source_level: level,
            source_origin: MtlOrigin::new(0, 0, 0),
            source_size: MtlSize::new(w, h, 1),
            destination_buffer: &b,
            destination_offset: 0,
            destination_bytes_per_row: w * 8,
            destination_bytes_per_image: n,
        };
        blit.encode(OnFault::Die, copy).unwrap();
        out.push((b, n));
    }
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    out
}

/// The bytes of each of `levels`, read after their command buffer completed.
fn bytes_of<P: Platform>(levels: &[(P::Buffer, u64)]) -> Vec<Vec<u8>> {
    let bytes = |(b, n): &(P::Buffer, u64)| {
        let length = usize::try_from(*n).unwrap();
        // SAFETY: a Shared buffer of `n` bytes, read after its command buffer completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), length) }.to_vec()
    };
    levels.iter().map(bytes).collect()
}

/// A dispatch of 16 by 16 threadgroups of `slice_write` (`lod` read from a buffer) or, for no
/// `lod`, `slice_write_l0`, over level `lod` of a [`SIDE`]-texel RGBA16Float texture of
/// [`LEVELS`] levels, Private, of usage `usage`, every level zeroed first by a blit; each level
/// then read back.
fn slice_write_levels<P: Platform>(
    p: &P,
    d: &P::Device,
    usage: u64,
    lod: Option<u64>,
) -> Vec<Vec<u8>> {
    let name = match lod {
        Some(_) => "slice_write",
        None => "slice_write_l0",
    };
    let library = d.new_library(LIB).unwrap().unwrap().unwrap();
    let f = library.new_function(OnFault::Die, name).unwrap().unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&f))
        .unwrap()
        .unwrap();
    let desc = TextureDesc {
        pixel_format: RGBA16F,
        width: SIDE,
        height: SIDE,
        mipmap_level_count: LEVELS,
        usage,
        resource_options: Some(PRIVATE),
        ..TextureDesc::default()
    };
    let t = d
        .new_texture(OnFault::Die, &p.texture_descriptor(&desc))
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Die).unwrap().unwrap();
    let blit = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
    let mut zeros = Vec::new();
    for level in 0..LEVELS {
        let side = SIDE >> level;
        let length = side * side * 8;
        let b = buffer::<P>(d, length, &vec![0; usize::try_from(length).unwrap()]);
        let copy = BlitCmd::CopyBufferToTexture {
            source_buffer: &b,
            source_offset: 0,
            source_bytes_per_row: side * 8,
            source_bytes_per_image: length,
            source_size: MtlSize::new(side, side, 1),
            destination_texture: &t,
            destination_slice: 0,
            destination_level: level,
            destination_origin: MtlOrigin::new(0, 0, 0),
        };
        blit.encode(OnFault::Die, copy).unwrap();
        zeros.push(b);
    }
    blit.encode(OnFault::Die, BlitCmd::EndEncoding).unwrap();
    let level = lod.unwrap_or(0);
    let lb = buffer::<P>(d, 4, &u32::try_from(level).unwrap().to_ne_bytes());
    let groups = (SIDE >> level) / 16;
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    let lbs = [Some(&lb)];
    let mut cmds = vec![
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetTexture {
            texture: Some(&t),
            index: 0,
        },
        ComputeCmd::SetImageblockWidth {
            width: 16,
            height: 16,
        },
    ];
    if lod.is_some() {
        cmds.push(ComputeCmd::SetBuffers {
            buffers: &lbs,
            offsets: &[0],
            range: NsRange {
                location: 0,
                length: 1,
            },
        });
    }
    cmds.push(ComputeCmd::DispatchThreadgroups {
        threadgroups_per_grid: MtlSize::new(groups, groups, 1),
        threads_per_threadgroup: MtlSize::new(16, 16, 1),
    });
    cmds.push(ComputeCmd::EndEncoding);
    for c in cmds {
        e.encode(OnFault::Die, c).unwrap();
    }
    let levels = read_levels::<P>(d, &cb, &t);
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_completed(OnFault::Die).unwrap();
    assert!(cb.error().is_none(), "{:?}", cb.error());
    bytes_of::<P>(&levels)
}

/// A kernel's imageblock slice write into a texture of the glass blur's usage, ShaderRead |
/// RenderTarget | BlockWritesOnly (0x10005), of ShaderRead | BlockWritesOnly (0x10001), and of
/// either with ShaderWrite (0x10007, 0x7), at levels 0 and 1 read from a buffer and at level 0 a
/// constant, writes every level of the texture as Metal's does, byte for byte
#[test]
#[cfg_attr(
    target_os = "macos",
    ignore = "MoltenVK: the runtime records device-nan-encoding, a death in strict mode; Mac-only test, off everywhere; ledger: MoltenVK strict deaths"
)]
fn imageblock_slice_writes_at_every_usage_are_metals() {
    let (vp, vd) = (platform(), dev());
    for usage in [0x10005, 0x10001, 0x10007, 0x7] {
        for lod in [Some(0), Some(1), None] {
            let ours = slice_write_levels(&vp, &vd, usage, lod);
            let case = format!("usage {usage:#x} lod {lod:?} (None: a constant 0)");
            crate::metal_check_raw!(&case, ours);
        }
    }
}
