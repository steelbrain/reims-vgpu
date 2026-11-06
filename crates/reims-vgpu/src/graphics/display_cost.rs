//! What a frame of the window's draw costs, on Metal, at 3840 × 2400, with the display's colour
//! off; on at the identity matrix, which the colour's pass draws through; and on at a display
//! profile's plane gamma of 256 entries, through the colour's pass and through its table (the
//! sink's module doc): the scanout (a blit with no colour), a BGRA8 surface (a copy) and a 10-bit
//! one (the present shaders). A measurement, run by hand in a release build, holding the host's
//! GPU (`cargo nextest run -p reims-vgpu --release --run-ignored only -E
//! 'test(display_frame_cost)' --no-capture`).
//!
//! Each frame is the window's draw into a drawable-sized render target
//! ([`ScanoutDraw::encode_shown`]), committed and waited for. Its figures are the thread's CPU
//! time, instructions and cycles from the draw's encode to its commit, and the GPU's run of it
//! (`GPUEndTime - GPUStartTime`), as medians (and the GPU's 10th and 90th percentiles).
//! Each case draws through a sink of its own, so its colour and its table stay made. For each
//! surface the cases take turns frame by frame, every other turn in the reverse order, so the
//! GPU's clock, which moves between frames, and a load on the host fall on each case alike:
//! [`WARM`] turns to warm up, then [`ROUNDS`] rounds of [`FRAMES`] turns. Each round's figures
//! are printed, then each case's median of its rounds and their spread (the lowest and highest).

#![cfg(target_os = "macos")]

use std::sync::Arc;
use std::sync::mpsc;

use reims_graphics::Scanout;
use reims_graphics::region::Region;
use reims_graphics::wire::RGFX_ALLOCATIONS_OFFSET;
use reims_vgpu_core::platform::{Handle as _, Platform};
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

/// The frames measured in each round, those before them, and the rounds.
const FRAMES: usize = 200;
const WARM: usize = 30;
const ROUNDS: usize = 5;
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

/// A frame of `draw` into `target`, committed and waited for: its CPU counts and GPU nanoseconds.
fn frame(
    draw: &ScanoutDraw<MetalPlatform>,
    target: &<MetalPlatform as Platform>::Texture,
) -> (Counts, u64) {
    let whole = PixelRect {
        x: 0,
        y: 0,
        width: u64::from(WIDTH),
        height: u64::from(HEIGHT),
    };
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
}

/// `turns` turns of `draws`' frames into `target`, one frame of each a turn, every other turn in
/// the reverse order: each draw's frames' CPU counts and GPU nanoseconds.
fn turns(
    draws: &[&ScanoutDraw<MetalPlatform>],
    target: &<MetalPlatform as Platform>::Texture,
    turns: usize,
) -> Vec<Vec<(Counts, u64)>> {
    let mut measured = vec![Vec::with_capacity(turns); draws.len()];
    for turn in 0..turns {
        let mut order: Vec<usize> = (0..draws.len()).collect();
        if turn % 2 == 1 {
            order.reverse();
        }
        for case in order {
            measured[case].push(frame(draws[case], target));
        }
    }
    measured
}

/// The median of `values`, and its 10th and 90th percentiles.
fn p50_p10_p90(mut values: Vec<u64>) -> (u64, u64, u64) {
    values.sort_unstable();
    let n = values.len();
    (values[n / 2], values[n / 10], values[n * 9 / 10])
}

/// The median of `values`, its lowest and its highest.
fn median_spread(mut values: Vec<u64>) -> (u64, u64, u64) {
    values.sort_unstable();
    (
        values[values.len() / 2],
        values[0],
        values[values.len() - 1],
    )
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
    let target = texture(&platform, BGRA8);
    let identity: Table = Arc::from(vec![
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]);
    // A display profile's: a power curve in each channel.
    #[expect(clippy::as_conversions, reason = "256 entries")]
    let plane: Table = Arc::from(
        (0..256)
            .map(|k| {
                let x = k as f32 / 255.0;
                [x.powf(1.0 / 2.2), x.powf(1.0 / 1.8), x.powf(1.2), 0.0]
            })
            .collect::<Vec<_>>(),
    );
    let surfaces = [
        ("scanout", None),
        ("BGRA8 surface", Some(BGRA8)),
        ("BGR10A2 surface", Some(BGR10A2)),
    ];
    println!("display_frame_cost: {WIDTH}x{HEIGHT}, {ROUNDS} rounds of {FRAMES} turns");
    let cases: [(&str, [Option<Table>; 4], bool); 4] = [
        ("no colour", Default::default(), true),
        (
            "identity matrix, the pass",
            [None, None, Some(Arc::clone(&identity)), None],
            true,
        ),
        (
            "plane gamma of 256, the pass",
            [Some(Arc::clone(&plane)), None, None, None],
            false,
        ),
        (
            "plane gamma of 256, its table",
            [Some(Arc::clone(&plane)), None, None, None],
            true,
        ),
    ];
    for (name, format) in surfaces {
        let surface = format.map(|f| texture(&platform, f));
        // A sink for each case, its colour set once.
        let sinks: Vec<_> = (cases.iter())
            .map(|(_, stages, tables)| {
                let presented = Arc::new(Presented::<MetalPlatform>::default());
                *presented.texture.lock() = surface.as_ref().map(|s| ShownTexture {
                    texture: s.clone_ref(),
                    hold: Arc::new(DisplayHold {
                        span: None,
                        answers: Mutex::default(),
                    }),
                });
                *presented.color.lock() = (
                    1,
                    Arc::new(DisplayColor {
                        stages: stages.clone(),
                    }),
                );
                let draw =
                    ScanoutDraw::new(Arc::clone(&platform), &region, &shown, &presented).unwrap();
                if !tables {
                    draw.without_tables();
                }
                (presented, draw)
            })
            .collect();
        let draws: Vec<_> = sinks.iter().map(|(_, draw)| draw).collect();
        let _ = turns(&draws, &target, WARM);
        // By case: each round's GPU p50 and p10 in nanoseconds, and its CPU figures' p50s.
        let mut gpu: Vec<Vec<(u64, u64)>> = vec![Vec::new(); cases.len()];
        let mut cpu: Vec<Vec<Vec<u64>>> = vec![vec![Vec::new(); FIGURES.len()]; cases.len()];
        for round in 0..ROUNDS {
            for (case, measured) in turns(&draws, &target, FRAMES).into_iter().enumerate() {
                let figures: Vec<u64> = (0..FIGURES.len())
                    .map(|f| p50_p10_p90(measured.iter().map(|(c, _)| c.0[f]).collect()).0)
                    .collect();
                let (gpu50, gpu10, gpu90) = p50_p10_p90(measured.iter().map(|&(_, g)| g).collect());
                println!(
                    "display_frame_cost: {name}, {}, round {round}: CPU p50 {}; GPU p50 {:.1} \
                     us, p10 {:.1} us, p90 {:.1} us",
                    cases[case].0,
                    (FIGURES.iter().zip(&figures))
                        .map(|(f, v)| format!("{f} {v}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    gpu50 as f64 / 1e3,
                    gpu10 as f64 / 1e3,
                    gpu90 as f64 / 1e3,
                );
                gpu[case].push((gpu50, gpu10));
                for (f, v) in figures.into_iter().enumerate() {
                    cpu[case][f].push(v);
                }
            }
        }
        for (case, (colour, _, _)) in cases.iter().enumerate() {
            let (g, g_low, g_high) = median_spread(gpu[case].iter().map(|g| g.0).collect());
            let (t, t_low, t_high) = median_spread(gpu[case].iter().map(|g| g.1).collect());
            let figures: Vec<String> = (0..FIGURES.len())
                .map(|f| {
                    let (m, low, high) = median_spread(cpu[case][f].clone());
                    format!("{} {m} ({low} to {high})", FIGURES[f])
                })
                .collect();
            println!(
                "display_frame_cost: {name}, {colour}: median of {ROUNDS} rounds: GPU p50 {:.1} us \
                 ({:.1} to {:.1}), p10 {:.1} us ({:.1} to {:.1}); CPU p50 {}",
                g as f64 / 1e3,
                g_low as f64 / 1e3,
                g_high as f64 / 1e3,
                t as f64 / 1e3,
                t_low as f64 / 1e3,
                t_high as f64 / 1e3,
                figures.join(", "),
            );
        }
    }
}
