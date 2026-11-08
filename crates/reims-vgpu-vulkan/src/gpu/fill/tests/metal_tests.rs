//! The tests of `fill` that compare with Metal on this host: macOS only.

use std::ffi::{c_char, c_void};

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder};
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::resource::Buffer;
use reims_vgpu_core::platform::types::ClearColor;
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;

use crate::platform::VulkanPlatform;

use super::*;

/// Component values that reach every code's edges: zeroes, ones, NaNs, infinities, the
/// integer types' bounds and past them, the half float's, denormals, halves between every
/// step of the narrow normalized codes, a ramp past [-1, 1] and scattered values.
fn values() -> Vec<f64> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        f64::NAN,
        -f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e-40,
        -1e-40,
        1e-6,
        6.1e-5,
        3e-8,
        65504.0,
        65519.0,
        65520.0,
        70000.0,
        127.0,
        128.0,
        -128.0,
        -129.0,
        255.0,
        256.0,
        1023.0,
        1024.0,
        65535.0,
        65536.0,
        32767.0,
        32768.0,
        -32768.0,
        -32769.0,
        2_147_483_647.0,
        2_147_483_648.0,
        -2_147_483_649.0,
        4_294_967_295.0,
        4_294_967_296.0,
        1e20,
        0.1,
        0.2,
        0.3,
        0.7,
        0.998_039_2,
        1.5,
        2.5,
        -1.5,
        -2.5,
    ];
    for den in [1.0, 3.0, 15.0, 31.0, 63.0, 127.0, 255.0, 1023.0] {
        for k in 0..64 {
            v.push((f64::from(k) + 0.5) / den);
            v.push(-(f64::from(k) + 0.5) / den);
        }
    }
    for den in [32767.0, 65535.0] {
        for k in (0..32767).step_by(1021) {
            v.push((f64::from(k) + 0.5) / den);
            v.push(-(f64::from(k) + 0.5) / den);
        }
    }
    for i in -250..1250 {
        v.push(f64::from(i) / 1000.0);
    }
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for _ in 0..600 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let unit = (seed >> 11) as f64 / (1u64 << 53) as f64;
        v.push(unit * 3.0 - 1.0);
        v.push((unit - 0.5) * 600.0);
        v.push((unit - 0.5) * 2f64.powi(34));
    }
    v
}

/// Every pixel format of [`PACKED`].
fn packed_formats() -> Vec<u64> {
    PACKED.iter().map(|f| u64::from(f.as_raw())).collect()
}

/// The least value of each of the 8-bit sRGB codes 1 to 255 that Metal stores it for, measured
/// (the runtime's `fill_value.rs`): halfway from the code below's decoded value to the code's,
/// each an `f32` of Metal's table of them, for codes 0 to 10 `code` times `1 / 3294.6`, both
/// `f32`s, in `f32` arithmetic, for the rest the `f32` nearest the curve's
/// `((code / 255 + 0.055) / 1.055)^2.4`.
fn srgb_edges() -> [f64; 255] {
    let decoded = |code: u16| -> f64 {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the step is an f32 operation of Metal's, from the f64 constant"
        )]
        let value = if code <= 10 {
            f32::from(code) * (1.0 / 3294.6f64) as f32
        } else {
            ((f64::from(code) / 255.0 + 0.055) / 1.055).powf(2.4) as f32
        };
        f64::from(value)
    };
    std::array::from_fn(|k| {
        let k = u16::try_from(k).unwrap_or(u16::MAX);
        f64::midpoint(decoded(k), decoded(k + 1))
    })
}

/// Colours of `v`, each value in each component.
fn colors_of(v: &[f64]) -> Vec<ClearColor> {
    (0..v.len())
        .map(|i| ClearColor {
            red: v[i],
            green: v[(i + 1) % v.len()],
            blue: v[(i + 2) % v.len()],
            alpha: v[(i + 3) % v.len()],
        })
        .collect()
}

/// A colour of each pixel format this platform packs, for many colours, is the bytes Metal
/// writes here: an RGBA32Uint texel of each, its 16 bytes the colour's repeated
#[test]
fn colors_are_the_metal_platforms_here() {
    assert_metals(&packed_formats(), &colors_of(&values()));
}

/// `edge`, the `f64`s next to it, the `f32` nearest it and those next to that
fn beside(edge: f64) -> [f64; 6] {
    let next = |x: f64, by: i64| f64::from_bits(x.to_bits().wrapping_add_signed(by));
    #[expect(clippy::cast_possible_truncation, reason = "the f32 nearest the edge")]
    let nearest = edge as f32;
    let next32 = |x: f32, by: i32| f64::from(f32::from_bits(x.to_bits().wrapping_add_signed(by)));
    [
        edge,
        next(edge, -1),
        next(edge, 1),
        next32(nearest, -1),
        f64::from(nearest),
        next32(nearest, 1),
    ]
}

/// Values on and beside every sRGB and extended-range code's least value, and past each end
#[test]
fn srgb_and_extended_range_edges_are_the_metal_platforms_here() {
    let mut v = vec![-1e-30, 1e-30, 0.999_999_9, 1.000_000_1, 1.5, -0.8];
    for edge in srgb_edges() {
        v.extend(beside(edge));
    }
    for code in 0..=1024 {
        v.extend(beside((f64::from(code) - 384.5) / 510.0));
    }
    let formats = [11, 31, 71, 81, 552, 554];
    assert_metals(&formats, &colors_of(&v));
}

/// Values on and beside the halfway points of RG11B10Float's and RGB9E5Float's mantissas, at
/// every exponent and below the least normal, and past their largest values
#[test]
fn small_and_shared_exponent_float_edges_are_the_metal_platforms_here() {
    let mut v = vec![
        65024.0, 65279.0, 65280.0, 64512.0, 64767.0, 64768.0, 65408.0, 65472.0,
    ];
    v.extend([4e38, 3e38, 1e-40, -1e-40]);
    for mantissa in [5, 6] {
        for exponent in -21..=15 {
            let step = 2f64.powi(exponent.max(-14) - mantissa);
            let base = if exponent < -14 {
                0.0
            } else {
                2f64.powi(exponent)
            };
            for q in [0, 1, 2, 3, (1 << mantissa) - 2, (1 << mantissa) - 1] {
                v.extend(beside(base + (f64::from(q) + 0.5) * step));
            }
        }
    }
    let mut colors = colors_of(&v);
    for exponent in -16..=16 {
        // The largest component sets the exponent, whose step is 2^(exponent - 8).
        let largest = 1.5 * 2f64.powi(exponent);
        let step = 2f64.powi(exponent - 8);
        for m in [0, 1, 2, 255] {
            for c in beside((f64::from(m) + 0.5) * step) {
                colors.extend([
                    ClearColor {
                        red: largest,
                        green: c,
                        blue: 0.0,
                        alpha: 0.0,
                    },
                    ClearColor {
                        red: c,
                        green: 0.0,
                        blue: largest,
                        alpha: 0.0,
                    },
                ]);
            }
        }
        for m in [256, 300, 511] {
            for c in beside((f64::from(m) + 0.5) * step) {
                colors.push(ClearColor {
                    red: 0.0,
                    green: c,
                    blue: 0.0,
                    alpha: 0.0,
                });
            }
        }
    }
    assert_metals(&[92, 93], &colors);
}

/// NaNs of either sign, quiet and signalling, whose payloads set the bits a half, a float
/// and a bfloat keep and the bits below them, in every pixel format this platform packs; and
/// doubles on and beside the 8-, 16- and 32-bit integer types' edges (2^31 and 2^32 and
/// their negatives among them, and those strictly between 2^31 - 1 and 2^31 and between
/// 2^32 - 1 and 2^32), and doubles strictly between two integers inside each type's range,
/// far from its edges (2^24 + 0.5, 3e9 + 0.5 and their kin, of either sign), in the integer
/// formats
#[test]
fn nan_payloads_and_integer_edges_are_the_metal_platforms_here() {
    let sign = 1u64 << 63;
    let nans = [
        0x7ff8_0000_0000_0000u64,
        0x7ff8_0000_0000_0001,
        0x7ff8_0000_1000_0000,
        0x7ff8_0000_2000_0000,
        0x7ff8_0200_0000_0000,
        0x7ff8_0400_0000_0000,
        0x7ff8_1000_0000_0000,
        0x7ff8_2000_0000_0000,
        0x7ffa_5a5a_5a5a_5a5a,
        0x7ffc_0000_0000_0000,
        0x7fff_ffff_ffff_ffff,
        0x7ff0_0000_0000_0001,
        0x7ff0_0000_2000_0000,
        0x7ff0_0400_0000_0000,
        0x7ff0_2000_0000_0000,
        0x7ff4_0000_0000_0000,
        0x7ff5_a5a5_a5a5_a5a5,
        0x7ff7_ffff_ffff_ffff,
    ];
    let mut v: Vec<f64> = nans
        .iter()
        .flat_map(|&bits| [f64::from_bits(bits), f64::from_bits(bits | sign)])
        .collect();
    v.extend([0.0, 1.0, -1.0, 0.5]);
    assert_metals(&packed_formats(), &colors_of(&v));
    let mut v = Vec::new();
    for edge in [
        2f64.powi(31) - 1.0,
        2f64.powi(31),
        2f64.powi(32) - 1.0,
        2f64.powi(32),
        -(2f64.powi(31)),
        -(2f64.powi(31)) - 1.0,
        -(2f64.powi(32)),
        127.0,
        128.0,
        255.0,
        256.0,
        32767.0,
        32768.0,
        65535.0,
        65536.0,
        -128.0,
        -129.0,
        -32768.0,
        -32769.0,
        1023.0,
        1024.0,
    ] {
        v.extend(beside(edge));
        v.extend([-0.75, -0.5, -0.25, 0.25, 0.5].map(|by| edge + by));
    }
    for inside in [
        100.5,
        200.25,
        1000.75,
        30_000.5,
        65_534.5,
        123_456.5,
        16_777_216.5,
        33_554_433.5,
        1_000_000_000.25,
        1_073_741_824.5,
        2_147_483_648.5,
        3_000_000_000.5,
        4_000_000_000.75,
        4_294_967_294.5,
    ] {
        v.extend([inside, -inside]);
    }
    let integers = [13, 14, 23, 24, 53, 54, 73, 74, 91, 113, 114, 123, 124];
    assert_metals(&integers, &colors_of(&v));
}

/// Each of `colors` packed in the pixel format `format`, filled into an RGBA32Uint texel each of
/// a `Private` texture 4096 wide on platform `P` (`desc` its descriptor), in one blit encoder,
/// read back: 16 bytes a colour.
fn fills_of<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    format: u64,
    colors: &[ClearColor],
) -> Vec<u8> {
    let width = 4096u64;
    let height = (colors.len() as u64).div_ceil(width);
    let desc = desc(&TextureDesc {
        pixel_format: 123,
        width,
        height,
        usage: 3,
        resource_options: Some(0x20),
        ..TextureDesc::default()
    });
    let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
    let length = width * height * 16;
    let out = d
        .new_buffer_of_length(OnFault::Return, length, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    for (i, color) in (0u64..).zip(colors) {
        let fill = BlitCmd::<P>::FillTextureColor {
            texture: &t,
            level: 0,
            slice: 0,
            region: MtlRegion::new(
                MtlOrigin::new(i % width, i / width, 0),
                MtlSize::new(1, 1, 1),
            ),
            color: *color,
            pixel_format: format,
        };
        e.encode(OnFault::Return, fill).unwrap();
    }
    let down = BlitCmd::<P>::CopyTextureToBufferOptions {
        source_texture: Some(&t),
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(width, height, 1),
        destination_buffer: Some(&out),
        destination_offset: 0,
        destination_bytes_per_row: width * 16,
        destination_bytes_per_image: length,
        options: 0,
    };
    e.encode(OnFault::Return, down).unwrap();
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    let length = usize::try_from(length).unwrap();
    // SAFETY: a live Shared buffer of `length` bytes, read after the GPU's writes.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), length) }.to_vec()
}

/// Asserts that each of `colors`, in each pixel format of `formats`, fills the bytes Metal
/// fills here, where there is a Metal device: an RGBA32Uint texel each ([`fills_of`]).
fn assert_metals(formats: &[u64], colors: &[ClearColor]) {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let d = crate::gpu::device::tests::dev();
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let metal_desc = |t: &TextureDesc| reims_vgpu_metal::gpu::desc::texture_descriptor(t);
    let mut wrong = Vec::new();
    for &format in formats {
        let theirs = fills_of::<MetalPlatform>(&metal, &metal_desc, format, colors);
        let ours = fills_of::<VulkanPlatform>(&d, &vulkan_desc, format, colors);
        for ((color, metal), ours) in colors.iter().zip(theirs.chunks(16)).zip(ours.chunks(16)) {
            if ours != metal {
                wrong.push(format!(
                    "{format} {color:?}: ours {ours:02x?} Metal {metal:02x?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} differ, the first: {:#?}",
        wrong.len(),
        colors.len() * formats.len(),
        &wrong[..wrong.len().min(40)]
    );
}

/// `MTLRegion` as `objc_msgSend` takes it: the origin, then the size.
#[repr(C)]
#[derive(Clone, Copy)]
struct Region {
    x: u64,
    y: u64,
    z: u64,
    width: u64,
    height: u64,
    depth: u64,
}

/// `MTLClearColor` as `objc_msgSend` takes it.
#[repr(C)]
#[derive(Clone, Copy)]
struct Colour {
    red: f64,
    green: f64,
    blue: f64,
    alpha: f64,
}

/// `fillTexture:level:slice:region:color:pixelFormat:` as `objc_msgSend` takes it: the encoder,
/// the selector, the texture, the level, the slice, the region, the colour and the pixel format.
type FillColour = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    u64,
    u64,
    Region,
    Colour,
    u64,
);

// libobjc's, which the Metal platform links.
unsafe extern "C" {
    fn objc_msgSend();
    fn sel_registerName(name: *const c_char) -> *const c_void;
}

/// The Objective-C object `o` is, as `objc_msgSend` takes it.
fn object<T>(o: &T) -> *const c_void {
    std::ptr::from_ref(o).cast()
}

/// Sets the 256 KiB of the stack below the caller's frame to `byte`.
#[inline(never)]
fn set_stack(byte: u8) {
    let mut bytes = [byte; 256 * 1024];
    std::hint::black_box(&mut bytes);
}

/// What Metal fills for a colour (0.25, 0.5, 0.75, 1) in `leftover`, a format it packs no colour
/// in, into each texel of a `Private` texture of `format` `width` texels wide, sent right after a
/// fill of (`red`, 9, 3, 1) packed in RGBA32Uint into another texture, in the same encoder, with no
/// call between the two but, with `stack`, one that sets the stack below to that byte
/// ([`set_stack`]): the texture's bytes, read back by a blit. Both fills are sent by
/// `objc_msgSend` itself, not by the Metal platform, whose sends in a dev build first run objc2's
/// check of the method, which leaves its own values in the bytes Metal fills (the module's docs).
fn metals_leftover(
    d: &<MetalPlatform as Platform>::Device,
    leftover: u64,
    (format, bytes, width): (u64, u64, u64),
    red: u8,
    stack: Option<u8>,
) -> Vec<u8> {
    let texture = |format, width| {
        let desc = reims_vgpu_metal::gpu::desc::texture_descriptor(&TextureDesc {
            pixel_format: format,
            width,
            height: 1,
            usage: 3,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Return, &desc).unwrap().unwrap()
    };
    let (earlier, t) = (texture(123, 1), texture(format, width));
    let row = bytes * width;
    let out = d
        .new_buffer_of_length(OnFault::Return, row, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
    // SAFETY: a NUL-terminated selector name.
    let sel =
        unsafe { sel_registerName(c"fillTexture:level:slice:region:color:pixelFormat:".as_ptr()) };
    // SAFETY: `objc_msgSend` takes the arguments of the method it sends, which `FillColour` gives.
    let send = unsafe { std::mem::transmute::<unsafe extern "C" fn(), FillColour>(objc_msgSend) };
    let (encoder, earlier_texture, texture) =
        (object(e.get()), object(earlier.get()), object(t.get()));
    let region = |width| Region {
        x: 0,
        y: 0,
        z: 0,
        width,
        height: 1,
        depth: 1,
    };
    let (one, row_region) = (region(1), region(width));
    let first = Colour {
        red: f64::from(red),
        green: 9.0,
        blue: 3.0,
        alpha: 1.0,
    };
    let asked = Colour {
        red: 0.25,
        green: 0.5,
        blue: 0.75,
        alpha: 1.0,
    };
    // SAFETY: a live encoder and texture, and the method's arguments.
    unsafe { send(encoder, sel, earlier_texture, 0, 0, one, first, 123) };
    if let Some(byte) = stack {
        set_stack(byte);
    }
    // SAFETY: as above.
    unsafe { send(encoder, sel, texture, 0, 0, row_region, asked, leftover) };
    let down = BlitCmd::<MetalPlatform>::CopyTextureToBufferOptions {
        source_texture: Some(&t),
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(width, 1, 1),
        destination_buffer: Some(&out),
        destination_offset: 0,
        destination_bytes_per_row: row,
        destination_bytes_per_image: row,
        options: 0,
    };
    e.encode(OnFault::Return, down).unwrap();
    e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    let n = usize::try_from(row).unwrap();
    // SAFETY: a live Shared buffer of `n` bytes, read after the GPU's writes.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// Formats outside [`LEFTOVER_FORMATS`] Metal packs no fill's colour in either: compressed (BC1,
/// BC4, BC6H, BC7, PVRTC, EAC, ETC2, ASTC LDR, sRGB and HDR), 4:2:2, YCbCr (8-bit, 10-bit and
/// packed 10-bit) and padded 10-bit.
const UNPACKED_FORMATS: [u64; 19] = [
    130, 140, 150, 152, 160, 170, 180, 186, 204, 222, 240, 241, 500, 505, 508, 576, 577, 578, 579,
];

/// What Metal writes for a colour in a format it packs none in, on this host: never the colour's
/// bytes, but those of the calling thread's stack where its packing writes a colour in a format
/// it packs ([`metals_leftover`]): sent right after a fill packed in RGBA32Uint, that fill's bytes,
/// whatever the colour; sent right after the stack below was set to one byte, that byte. Each of
/// [`LEFTOVER_FORMATS`] and of [`UNPACKED_FORMATS`], after a first pair that is not compared: the
/// process's first such pair reads other bytes, the selector's address and a heap address, which
/// the first send's own work left there (measured on the M3 Max under macOS 27.0.1, 2026-11-07:
/// the first of four pairs in one process, where the other three read the earlier fill's bytes,
/// and a pair that set the stack read its byte even first; which work leaves them is not
/// established). This platform's colour fill writes the earlier fill's bytes
/// (`leftover_colours_fill_an_earlier_colours_bytes_as_metals`)
#[test]
fn metal_fills_a_colour_it_packs_none_in_with_its_stacks_bytes() {
    let Some(d) = MetalPlatform::system_default_device() else {
        return;
    };
    let leftover = LEFTOVER_FORMATS.iter().map(|f| u64::from(f.as_raw()));
    // The process's first pair, whose bytes are the first send's (the test's docs).
    let first = u64::from(LEFTOVER_FORMATS[0].as_raw());
    metals_leftover(&d, first, (123, 16, 1), 6, None);
    for (red, format) in (7u8..).zip(leftover.chain(UNPACKED_FORMATS)) {
        assert_eq!(
            metals_leftover(&d, format, (123, 16, 1), red, None),
            [red, 0, 0, 0, 9, 0, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0],
            "{format}"
        );
        let byte = 0xa0 | (red & 0x0f);
        assert_eq!(
            metals_leftover(&d, format, (123, 16, 1), red, Some(byte)),
            [byte; 16],
            "{format}"
        );
    }
}

/// What a colour fill in `leftover`, a format Metal packs no colour in, of each texel of a
/// `Private` texture of `format` `width` texels wide writes on platform `P`, after a fill of
/// (7, 9, 3, 1) packed in RGBA32Uint into another texture, earlier in the same encoder or, with
/// `earlier_encoder`, in an earlier encoder of the command buffer: the texture's bytes, read
/// back by a blit.
fn leftover_on<P: Platform>(
    d: &P::Device,
    desc: &dyn Fn(&TextureDesc) -> P::TextureDescriptor,
    leftover: u64,
    (format, bytes, width): (u64, u64, u64),
    earlier_encoder: bool,
) -> Vec<u8> {
    let texture = |format, width| {
        let desc = desc(&TextureDesc {
            pixel_format: format,
            width,
            height: 1,
            usage: 3,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        });
        d.new_texture(OnFault::Return, &desc).unwrap().unwrap()
    };
    let (earlier, t) = (texture(123, 1), texture(format, width));
    let row = bytes * width;
    let out = d
        .new_buffer_of_length(OnFault::Return, row, 0)
        .unwrap()
        .unwrap();
    let q = d.new_command_queue().unwrap();
    let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
    let region = |w| MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(w, 1, 1));
    let first = BlitCmd::<P>::FillTextureColor {
        texture: &earlier,
        level: 0,
        slice: 0,
        region: region(1),
        color: ClearColor {
            red: 7.0,
            green: 9.0,
            blue: 3.0,
            alpha: 1.0,
        },
        pixel_format: 123,
    };
    let fill = BlitCmd::<P>::FillTextureColor {
        texture: &t,
        level: 0,
        slice: 0,
        region: region(width),
        color: ClearColor {
            red: 0.25,
            green: 0.5,
            blue: 0.75,
            alpha: 1.0,
        },
        pixel_format: leftover,
    };
    let down = BlitCmd::<P>::CopyTextureToBufferOptions {
        source_texture: Some(&t),
        source_slice: 0,
        source_level: 0,
        source_origin: MtlOrigin::new(0, 0, 0),
        source_size: MtlSize::new(width, 1, 1),
        destination_buffer: Some(&out),
        destination_offset: 0,
        destination_bytes_per_row: row,
        destination_bytes_per_image: row,
        options: 0,
    };
    let encode = |cmds: Vec<BlitCmd<'_, P>>| {
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for c in cmds {
            e.encode(OnFault::Return, c).unwrap();
        }
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
    };
    if earlier_encoder {
        encode(vec![first]);
        encode(vec![fill, down]);
    } else {
        encode(vec![first, fill, down]);
    }
    cb.commit(OnFault::Return).unwrap();
    cb.wait_until_completed(OnFault::Return).unwrap();
    let n = usize::try_from(row).unwrap();
    // SAFETY: a live Shared buffer of `n` bytes, read after the GPU's writes.
    unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), n) }.to_vec()
}

/// A colour fill in a format Metal packs no colour in writes what Metal's does when sent right
/// after a colour fill packed in RGBA32Uint ([`metals_leftover`]): that fill's 16 bytes, repeated
/// or cut at the texel, into an RGBA32Uint texel, 4 RGBA8Unorm ones and 4 R8Unorm ones, whatever
/// the colour, after that fill in the same encoder or in an earlier encoder of the command buffer
/// ([`leftover_on`]). The runtime holds the command buffer's last packed texel; Metal's bytes
/// after an encoder's end are those its own calls left on the stack (the module's docs). The
/// formats: each of [`LEFTOVER_FORMATS`] and of [`UNPACKED_FORMATS`]. Measured on the M3 Max
/// (macOS 27.0.1), the same for every one of them: `07 00 00 00 09 00 00 00 03 00 00 00 01 00 00
/// 00`, `07 00 00 00` each and `07` each
#[test]
fn leftover_colours_fill_an_earlier_colours_bytes_as_metals() {
    let Some(metal) = MetalPlatform::system_default_device() else {
        return;
    };
    let d = crate::gpu::device::tests::dev();
    let vulkan_desc = |t: &TextureDesc| crate::platform::tests::platform().texture_descriptor(t);
    let mut differ = Vec::new();
    let leftover = LEFTOVER_FORMATS.iter().map(|f| u64::from(f.as_raw()));
    for leftover in leftover.chain(UNPACKED_FORMATS) {
        for texture in [(123, 16, 1), (70, 4, 4), (10, 1, 4)] {
            let theirs = metals_leftover(&metal, leftover, texture, 7, None);
            for earlier_encoder in [false, true] {
                let what = format!(
                    "a colour in {leftover} into {texture:?}, earlier encoder {earlier_encoder}"
                );
                let ours = catch_death(|| {
                    leftover_on::<VulkanPlatform>(
                        &d,
                        &vulkan_desc,
                        leftover,
                        texture,
                        earlier_encoder,
                    )
                });
                match ours {
                    Ok(ours) if ours == theirs => {}
                    Ok(ours) => {
                        differ.push(format!("{what}: ours {ours:02x?}, Metal's {theirs:02x?}"))
                    }
                    Err(site) => {
                        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                        differ.push(format!(
                            "{what}: Metal's {theirs:02x?}, ours died at {}: {line}",
                            site.file
                        ));
                    }
                }
            }
        }
    }
    assert!(
        differ.is_empty(),
        "{} differ:\n{}",
        differ.len(),
        differ.join("\n")
    );
}
