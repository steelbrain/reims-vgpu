//! What a frame of the window's draw costs, on Metal, at 3840 × 2400, with the display's colour
//! off and with it on at the identity matrix, which the colour pass draws through: the scanout
//! (a blit, against the scanout's colour pass), a BGRA8 surface (a copy) and a 10-bit one (the
//! present shaders), each against the surface's colour pass. A measurement, run by hand in a
//! release build, holding the host's GPU (`cargo nextest run -p reims-vgpu --release
//! --run-ignored only -E 'test(display_frame_cost)' --no-capture`).
//!
//! Each frame is the window's draw into a drawable-sized render target ([`ScanoutDraw::encode_shown`]),
//! committed and waited for. Its figures are the thread's CPU time, instructions and cycles from
//! the draw's encode to its commit, and the GPU's run of it (`GPUEndTime - GPUStartTime`), over
//! [`FRAMES`] frames after [`WARM`] to warm up, as medians, the GPU's with its 90th percentile.

#![cfg(target_os = "macos")]

use std::sync::Arc;
use std::sync::mpsc;

use reims_graphics::Scanout;
use reims_graphics::region::Region;
use reims_graphics::wire::RGFX_ALLOCATIONS_OFFSET;
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::{CommandBuffer as _, CompletedInfo};
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_window::PixelRect;

use super::color::{DisplayColor, Table};
use super::decode::draw_cost::{Counts, FIGURES, counts};
use super::sink::ScanoutDraw;
use super::{DisplayHold, Presented, Shown, ShownTexture};
use crate::platform::ApiPlatform as _;

/// The frames measured, and those before them.
const FRAMES: usize = 300;
const WARM: usize = 30;
/// The display's size.
const WIDTH: u32 = 3840;
const HEIGHT: u32 = 2400;
/// `MTLPixelFormatBGRA8Unorm` and `MTLPixelFormatBGR10A2Unorm`.
const BGRA8: u64 = 80;
const BGR10A2: u64 = 94;
/// `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`, and storage mode private.
const USAGE: u64 = 0x1 | 0x4;
const PRIVATE: u64 = 0x20;

/// A texture of the display's size and `format`, on the platform's default device, the draw's.
fn texture(platform: &MetalPlatform, format: u64) -> <MetalPlatform as Platform>::Texture {
    let descriptor = platform.texture_descriptor(&TextureDesc {
        pixel_format: format,
        width: u64::from(WIDTH),
        height: u64::from(HEIGHT),
        usage: USAGE,
        resource_options: Some(PRIVATE),
        ..TextureDesc::default()
    });
    (platform.metal_device(None).unwrap())
        .new_texture(OnFault::Die, &descriptor)
        .unwrap()
        .unwrap()
}

/// `frames` frames of `draw` into `target`, each committed and waited for: each one's CPU counts
/// and GPU nanoseconds.
fn frames(
    draw: &ScanoutDraw<MetalPlatform>,
    target: &<MetalPlatform as Platform>::Texture,
    frames: usize,
) -> Vec<(Counts, u64)> {
    let whole = PixelRect {
        x: 0,
        y: 0,
        width: u64::from(WIDTH),
        height: u64::from(HEIGHT),
    };
    (0..frames)
        .map(|_| {
            let before = counts();
            let (command_buffer, hold) = draw.encode_shown(target, whole).expect("a frame");
            let (gpu, ran) = mpsc::channel();
            command_buffer
                .add_completed_handler(
                    OnFault::Die,
                    Box::new(move |info: CompletedInfo| {
                        let _ = gpu.send(info.gpu_times_ns.map_or(0, |(s, e)| e - s));
                    }),
                )
                .unwrap();
            command_buffer.commit(OnFault::Die).unwrap();
            let spent = counts() - before;
            command_buffer.wait_until_completed(OnFault::Die).unwrap();
            drop(hold);
            (spent, ran.recv().unwrap())
        })
        .collect()
}

/// The median of `values`, and its 90th percentile.
fn p50_p90(mut values: Vec<u64>) -> (u64, u64) {
    values.sort_unstable();
    let n = values.len();
    (values[n / 2], values[n * 9 / 10])
}

#[test]
#[ignore = "a measurement, run by hand"]
fn display_frame_cost() {
    let platform = MetalPlatform::process();
    let region = Arc::new(Region::new(reims_graphics::REGION_SIZE as usize).unwrap());
    let stride = WIDTH * 4;
    // The guest's pixels, written so the pages are the guest's.
    // SAFETY: the scanout lies inside the region, mapped while it lives.
    unsafe {
        let at = region.base().add(RGFX_ALLOCATIONS_OFFSET as usize);
        for i in 0..(stride as usize * HEIGHT as usize) {
            *at.add(i) = (i * 31 % 251) as u8;
        }
    }
    let shown = Arc::new(Shown::default());
    *shown.scanout.lock() = Some(Scanout {
        width: WIDTH,
        height: HEIGHT,
        stride,
        offset: RGFX_ALLOCATIONS_OFFSET,
    });
    let presented = Arc::new(Presented::<MetalPlatform>::default());
    let draw = ScanoutDraw::new(Arc::clone(&platform), &region, &shown, &presented).unwrap();
    let target = texture(&platform, BGRA8);
    let identity: Table = Arc::from(vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]);
    let surfaces = [
        ("scanout", None),
        ("BGRA8 surface", Some(BGRA8)),
        ("BGR10A2 surface", Some(BGR10A2)),
    ];
    println!("display_frame_cost: {WIDTH}x{HEIGHT}, {FRAMES} frames each");
    for (name, format) in surfaces {
        *presented.texture.lock() = format.map(|f| ShownTexture {
            texture: texture(&platform, f),
            hold: Arc::new(DisplayHold {
                span: None,
                answers: Mutex::default(),
            }),
        });
        for (colour, stages) in [
            ("no colour", Default::default()),
            (
                "identity matrix",
                [None, None, Some(Arc::clone(&identity)), None],
            ),
        ] {
            {
                let mut color = presented.color.lock();
                *color = (color.0 + 1, Arc::new(DisplayColor { stages }));
            }
            let _ = frames(&draw, &target, WARM);
            let measured = frames(&draw, &target, FRAMES);
            let cpu: Vec<String> = (0..FIGURES.len())
                .map(|f| {
                    let (p50, _) = p50_p90(measured.iter().map(|(c, _)| c.0[f]).collect());
                    format!("{} {p50}", FIGURES[f])
                })
                .collect();
            let (gpu50, gpu90) = p50_p90(measured.iter().map(|&(_, g)| g).collect());
            println!(
                "display_frame_cost: {name}, {colour}: CPU p50 {}; GPU p50 {:.1} us, p90 {:.1} us",
                cpu.join(", "),
                gpu50 as f64 / 1e3,
                gpu90 as f64 / 1e3,
            );
        }
    }
}
