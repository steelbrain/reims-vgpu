//! The kernel against core's reference, on every GPU the host's tests reach: Metal on macOS, the
//! Vulkan platform's runtime where the crate has it (MoltenVK on macOS). Each GPU test says which
//! GPU it ran on, and skips a GPU the host has none of, saying so.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_core::platform::Platform as _;
use reims_vgpu_core::platform::os::surface::{SurfacePlaneSpec, SurfaceSpec};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_core::device_memory::DeviceMemory;
use reims_vgpu_video_core::runtime::StdRuntime;
use reims_vgpu_video_core::scaler::format::{Packing, ScalerFormat};
use reims_vgpu_video_core::scaler::plan::{FormatOf, Matrix, Plan, Siting};
use reims_vgpu_video_core::scaler::reference::{self, Space};
use reims_vgpu_video_core::scaler::request::{
    FIXED_ONE, FixedRect, Rect, Transform, TransformFlags,
};
use reims_vgpu_video_core::surfaces::{GuestSurface, PlaneLayout, SurfaceLayout};

use super::*;
use crate::platform::ApiPlatform as _;

/// The kernel on each GPU this host's tests reach, taking sides from 1, by name.
fn gpus() -> Vec<(String, Arc<dyn ScalerBackend>)> {
    let runtime = StdRuntime::shared();
    let mut out: Vec<(String, Arc<dyn ScalerBackend>)> = Vec::new();
    #[cfg(target_os = "macos")]
    {
        let gpu = metal::MetalGpu::process().expect("Metal opens the kernel");
        let name = format!("Metal, {}", gpu.name());
        let backend = GpuScalerBackend::new(gpu, Arc::clone(&runtime)).with_smallest_side(1);
        out.push((name, Arc::new(backend)));
    }
    #[cfg(feature = "vulkan")]
    match vulkan::VulkanGpu::process(&runtime) {
        Ok(gpu) => {
            let name = format!("Vulkan, {}", gpu.name());
            let backend = GpuScalerBackend::new(gpu, Arc::clone(&runtime)).with_smallest_side(1);
            out.push((name, Arc::new(backend)));
        }
        Err(why) => eprintln!("no Vulkan GPU: {why}; skipped"),
    }
    out
}

/// A surface of the test's own memory, which counts its lends.
struct Memory {
    layout: SurfaceLayout,
    bytes: Mutex<Vec<u8>>,
    lends: AtomicUsize,
}

impl GuestSurface for Memory {
    fn layout(&self) -> SurfaceLayout {
        self.layout.clone()
    }

    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
        self.lends.fetch_add(1, Ordering::Relaxed);
        write(&mut self.bytes.lock());
        Ok(())
    }

    fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
        self.lends.fetch_add(1, Ordering::Relaxed);
        read(&self.bytes.lock());
        Ok(())
    }
}

/// A layout of `format`, `width` by `height`: rows rounded up to `align` bytes, a biplanar
/// format's chroma after its luma, at a multiple of `align` plus `skew`.
fn layout(
    format: ScalerFormat,
    width: u32,
    height: u32,
    align: usize,
    skew: usize,
) -> SurfaceLayout {
    let row = |bytes: usize| bytes.div_ceil(align) * align;
    let (w, h) = (width.to_usize(), height.to_usize());
    let s = format.bytes_per_sample();
    match format.packing() {
        Packing::Biplanar420 => {
            let luma = row(w * s);
            let chroma = row(w.div_ceil(2) * 2 * s);
            SurfaceLayout {
                pixel_format: format.fourcc(),
                width,
                height,
                planes: vec![
                    PlaneLayout {
                        offset: 0,
                        bytes_per_row: luma,
                        width,
                        height,
                    },
                    PlaneLayout {
                        offset: row(luma * h) + skew,
                        bytes_per_row: chroma,
                        width: width.div_ceil(2),
                        height: height.div_ceil(2),
                    },
                ],
                bytes_per_row: luma,
            }
        }
        Packing::Packed422(_) => SurfaceLayout {
            pixel_format: format.fourcc(),
            width,
            height,
            planes: Vec::new(),
            bytes_per_row: row(w.div_ceil(2) * 4),
        },
        Packing::Rgb32(_) => SurfaceLayout {
            pixel_format: format.fourcc(),
            width,
            height,
            planes: Vec::new(),
            bytes_per_row: row(w * 4),
        },
    }
}

/// The bytes a layout's planes span.
fn length(layout: &SurfaceLayout) -> usize {
    let of = FormatOf::of(layout).unwrap();
    extent(&of).unwrap().next_multiple_of(64)
}

/// `layout`'s bytes holding the test picture, or `rgb` flat.
fn picture_bytes(layout: &SurfaceLayout, rgb: Option<[f64; 4]>) -> Vec<u8> {
    let of = FormatOf::of(layout).unwrap();
    let mut bytes = vec![0u8; length(layout)];
    let (w, h) = (layout.width.to_usize(), layout.height.to_usize());
    let picture = match rgb {
        Some(rgb) => reference::flat_picture(w, h, rgb),
        None => reference::test_picture(w, h),
    };
    reference::encode(
        &of,
        &mut bytes,
        &picture,
        0,
        0,
        Space::Rgb,
        Matrix::Bt601,
        Siting::CENTRED,
    );
    bytes
}

fn memory(layout: SurfaceLayout, rgb: Option<[f64; 4]>) -> Memory {
    let bytes = picture_bytes(&layout, rgb);
    Memory {
        layout,
        bytes: Mutex::new(bytes),
        lends: AtomicUsize::new(0),
    }
}

/// The largest difference between two surfaces of `of`, a sample at a time over each plane's
/// samples, and how many samples differ.
fn difference(of: &FormatOf, a: &[u8], b: &[u8]) -> (u32, usize) {
    let (mut largest, mut count) = (0u32, 0usize);
    let s = of.format.bytes_per_sample();
    for span in &of.planes {
        for row in 0..span.rows {
            let at = span.offset + row * span.bytes_per_row;
            for k in (0..span.row_bytes).step_by(s) {
                let sample = |bytes: &[u8]| {
                    if s == 2 {
                        u32::from(u16::from_le_bytes([bytes[at + k], bytes[at + k + 1]]) >> 6)
                    } else {
                        u32::from(bytes[at + k])
                    }
                };
                let d = sample(a).abs_diff(sample(b));
                largest = largest.max(d);
                count += usize::from(d != 0);
            }
        }
    }
    (largest, count)
}

/// One transform of a sweep: its name, the surfaces, and its flags and windows.
struct Case {
    name: String,
    source: SurfaceLayout,
    destination: SurfaceLayout,
    flags: u64,
    crop: FixedRect,
    rect: Rect,
}

/// The plan of `transform`, as the device makes it, but for a pair of formats Apple's scaler
/// refuses, which the device refuses (`format::apple_refuses`) and the kernel does all the same.
fn plan(
    transform: &Transform,
    source: &SurfaceLayout,
    destination: &SurfaceLayout,
) -> Result<Plan, IoReturn> {
    let planned = Plan::new(transform, source, destination);
    match planned {
        Err(io_return::UNSUPPORTED) => Plan::of_any_pair(transform, source, destination),
        planned => planned,
    }
}

/// Runs `case` on `backend`, from the test picture into a destination of mid grey; the largest
/// difference from the reference and how many samples differ, or the status.
fn run(accelerator: &mut dyn Accelerator, case: &Case) -> Result<(u32, usize, u64), IoReturn> {
    let source = memory(case.source.clone(), None);
    let destination = memory(case.destination.clone(), Some([0.5, 0.5, 0.5, 1.0]));
    let transform = Transform {
        flags: TransformFlags(case.flags),
        source_rect: case.crop,
        destination_rect: case.rect,
        ..Transform::default()
    };
    let plan = plan(&transform, &source.layout, &destination.layout)?;
    let mut expected = destination.bytes.lock().clone();
    reference::run(&plan, &source.bytes.lock(), &mut expected)?;
    let mut job = TransformJob {
        transform: &transform,
        plan: plan.clone(),
        source: &source,
        destination: &destination,
        hold: GuestHold::nothing(),
        inputs: Default::default(),
        outputs: Default::default(),
    };
    match accelerator.transform(&mut job) {
        io_return::SUCCESS => {}
        status => return Err(status),
    }
    let out = destination.bytes.lock().clone();
    let (largest, count) = difference(&plan.destination_format, &out, &expected);
    Ok((largest, count, digest(&out)))
}

/// A digest of `bytes`, FNV-1a's of 64 bits: two builds whose kernels write the same bytes print
/// the same digests.
fn digest(bytes: &[u8]) -> u64 {
    (bytes.iter()).fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Every pair of formats, through each kind of transform: copies at an odd size, scales down and
/// up, a quarter turn, a flip, a crop at a fraction, an odd window, and the nearest filter (which
/// filters as the smooth one, as Apple's scaler does), each within a code of the reference a
/// sample, and exact where the reference only moves samples: copies and flips of a format into
/// itself, its quarter turns but for 4:2:2's, whose chroma the turn resamples, and copies, flips
/// and turns between RGB formats. The pairs the device refuses as Apple's scaler does run too, on
/// plans made by hand. Prints each case's largest difference, how many samples differ and a
/// digest of the destination's bytes, for the record, and to tell two builds' outputs apart.
#[test]
fn every_pair_of_formats_is_within_a_code_of_the_reference() {
    let fixed = |x: f64, y: f64, w: f64, h: f64| {
        let f = |v: f64| {
            #[expect(
                clippy::as_conversions,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let v = (v * FIXED_ONE as f64).round() as u64;
            v
        };
        FixedRect {
            x: f(x),
            y: f(y),
            width: f(w),
            height: f(h),
        }
    };
    let whole = FixedRect::default();
    let all = Rect::default();
    let mut failures = Vec::new();
    for (gpu, backend) in gpus() {
        let mut accelerator = backend.create_accelerator().unwrap();
        let mut inexact = 0;
        for from in ScalerFormat::ALL {
            for to in ScalerFormat::ALL {
                let n = name;
                let l = |f, w, h| layout(f, w, h, 64, 0);
                let kinds: [(&str, SurfaceLayout, SurfaceLayout, u64, FixedRect, Rect); 9] = [
                    ("copy", l(from, 37, 23), l(to, 37, 23), 0, whole, all),
                    ("down", l(from, 96, 54), l(to, 40, 22), 0x1000, whole, all),
                    ("up", l(from, 24, 14), l(to, 60, 34), 0x1000, whole, all),
                    ("transfer-up", l(from, 24, 14), l(to, 61, 35), 0, whole, all),
                    ("turn", l(from, 38, 22), l(to, 22, 38), 4, whole, all),
                    ("flip", l(from, 37, 23), l(to, 37, 23), 3, whole, all),
                    (
                        "crop",
                        l(from, 64, 48),
                        l(to, 30, 20),
                        0x1000,
                        fixed(5.5, 3.25, 40.0, 30.0),
                        all,
                    ),
                    (
                        "window",
                        l(from, 32, 24),
                        l(to, 48, 40),
                        0x1000,
                        whole,
                        Rect {
                            x: 7,
                            y: 5,
                            width: 31,
                            height: 25,
                        },
                    ),
                    ("nearest", l(from, 64, 40), l(to, 40, 24), 0x80, whole, all),
                ];
                for (kind, source, destination, flags, crop, rect) in kinds {
                    let case = Case {
                        name: format!("{kind} {} to {}", n(from), n(to)),
                        source,
                        destination,
                        flags,
                        crop,
                        rect,
                    };
                    match run(accelerator.as_mut(), &case) {
                        Ok((largest, count, digest)) => {
                            println!(
                                "{gpu}: {}: largest {largest}, {count} differ, digest {digest:016x}",
                                case.name
                            );
                            let rgb = !from.is_ycbcr() && !to.is_ycbcr();
                            let moves = match kind {
                                "copy" | "flip" => true,
                                "turn" => !matches!(from.packing(), Packing::Packed422(_)),
                                _ => false,
                            };
                            let exact = moves && (from == to || rgb);
                            if largest > 1 || (exact && largest > 0) {
                                failures.push(format!("{gpu}: {}: largest {largest}", case.name));
                            }
                            inexact += usize::from(largest > 0);
                        }
                        Err(status) => {
                            failures.push(format!("{gpu}: {}: status {status:#x}", case.name));
                        }
                    }
                }
            }
        }
        println!("{gpu}: {inexact} of {} cases not exact", 64 * 9);
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A format's name as the suite prints it.
fn name(format: ScalerFormat) -> String {
    reims_vgpu_video_conformance::scaler::format_name(format)
}

/// The conformance suite's cases of small surfaces pass, the ones under Apple's smallest side
/// among them, with the siting and map lines it prints
#[test]
fn the_suites_small_cases_pass() {
    let only: Vec<String> = [
        "siting", "map", "tiny-", "odd-", "rect-odd", "rotate-", "nearest-",
    ]
    .map(str::to_owned)
    .to_vec();
    for (gpu, backend) in gpus() {
        assert!(
            reims_vgpu_video_conformance::scaler::check(backend.as_ref(), &gpu, &only, false),
            "{gpu}"
        );
    }
}

/// The whole conformance suite, with its benches: run alone, in release, on each GPU
/// (`cargo nextest run --release -p reims-vgpu -E 'test(/whole_suite/)' --run-ignored only
/// --no-capture`), for the record in docs/video/scaler.md.
#[test]
#[ignore = "the whole suite, for the record: run alone, in release"]
fn the_whole_suite() {
    for (gpu, backend) in gpus() {
        assert!(
            reims_vgpu_video_conformance::scaler::check(backend.as_ref(), &gpu, &[], true),
            "{gpu}"
        );
        bench_in_place(&gpu, backend.as_ref());
    }
}

/// The suite's benches on surfaces of the platform's, as a guest's reach the device: each
/// transform's wall time through the accelerator and its GPU time, the median of 20 after a
/// first, and whether the surfaces were reached in place.
fn bench_in_place(gpu: &str, backend: &dyn ScalerBackend) {
    use ScalerFormat::{Bgra, Nv12VideoRange};

    let mut accelerator = backend.create_accelerator().unwrap();
    let runtime = StdRuntime::shared();
    let benches = [
        (
            "1080p 420v to 720p BGRA",
            Nv12VideoRange,
            (1920, 1080),
            Bgra,
            (1280, 720),
            0x1000,
        ),
        (
            "2160p 420v to 1080p 420v",
            Nv12VideoRange,
            (3840, 2160),
            Nv12VideoRange,
            (1920, 1080),
            0x1000,
        ),
        (
            "720p BGRA to 720p 420v",
            Bgra,
            (1280, 720),
            Nv12VideoRange,
            (1280, 720),
            0x1000,
        ),
        (
            "1080p BGRA turned 90",
            Bgra,
            (1920, 1080),
            Bgra,
            (1080, 1920),
            0x1004,
        ),
        (
            "2160p 420v to 2160p BGRA",
            Nv12VideoRange,
            (3840, 2160),
            Bgra,
            (3840, 2160),
            0x1000,
        ),
    ];
    for (label, from, (sw, sh), to, (dw, dh), flags) in benches {
        let (sl, dl) = (layout(from, sw, sh, 64, 0), layout(to, dw, dh, 64, 0));
        let source = platform_surface(sl.clone(), &picture_bytes(&sl, None), None);
        let grey = picture_bytes(&dl, Some([0.5, 0.5, 0.5, 1.0]));
        let destination = platform_surface(dl.clone(), &grey, None);
        let transform = Transform {
            flags: TransformFlags(flags),
            ..Transform::default()
        };
        let plan = Plan::new(&transform, &sl, &dl).unwrap();
        let (mut wall, mut on_gpu) = (Vec::new(), Vec::new());
        for _ in 0..21 {
            let mut job = TransformJob {
                transform: &transform,
                plan: plan.clone(),
                source: &source,
                destination: &destination,
                hold: GuestHold::nothing(),
                inputs: Default::default(),
                outputs: Default::default(),
            };
            GPU_TIMES.lock().clear();
            let started = runtime.now();
            assert_eq!(
                accelerator.transform(&mut job),
                io_return::SUCCESS,
                "{gpu}: {label}"
            );
            wall.push(runtime.elapsed_since(started).as_secs_f64() * 1000.0);
            on_gpu.extend(GPU_TIMES.lock().iter().map(|t| t.as_secs_f64() * 1000.0));
        }
        let median = |times: &[f64]| {
            let mut rest = times.get(1..).unwrap_or_default().to_vec();
            rest.sort_by(f64::total_cmp);
            rest.get(rest.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let lends =
            source.lends.load(Ordering::Relaxed) + destination.lends.load(Ordering::Relaxed);
        println!(
            "bench in place {label}: {gpu}: first {:.1} ms, median of 20 {:.1} ms wall, \
             {:.2} ms on the GPU, {}",
            wall[0],
            median(&wall),
            median(&on_gpu),
            if lends == 0 {
                "in place".to_owned()
            } else {
                format!("{lends} lends")
            }
        );
    }
}

/// Apple's sides, from 4 to 65536 each way, and ratios; a surface outside the sides is refused
#[test]
fn a_side_outside_apples_is_refused() {
    let caps = capabilities(SMALLEST_SIDE);
    assert_eq!(caps.min_source, 4);
    assert_eq!(caps.max_source, (65536, 65536));
    assert_eq!(caps.max_scale, [8, 8, 8, 4]);
    let runtime = StdRuntime::shared();
    #[cfg(target_os = "macos")]
    let gpu = Some(metal::MetalGpu::process().unwrap());
    #[cfg(not(target_os = "macos"))]
    let gpu = vulkan::VulkanGpu::process(&runtime).ok();
    let Some(gpu) = gpu else {
        eprintln!("no GPU: skipped");
        return;
    };
    let mut accelerator = GpuScalerBackend::new(gpu, runtime)
        .create_accelerator()
        .unwrap();
    let b = ScalerFormat::Bgra;
    let case = |source: SurfaceLayout, destination: SurfaceLayout| Case {
        name: "small".to_owned(),
        source,
        destination,
        flags: 0,
        crop: FixedRect::default(),
        rect: Rect::default(),
    };
    let refused = case(layout(b, 3, 8, 64, 0), layout(b, 8, 8, 64, 0));
    assert_eq!(
        run(accelerator.as_mut(), &refused),
        Err(io_return::UNSUPPORTED)
    );
    let refused = case(layout(b, 8, 8, 64, 0), layout(b, 8, 3, 64, 0));
    assert_eq!(
        run(accelerator.as_mut(), &refused),
        Err(io_return::UNSUPPORTED)
    );
    let taken = case(layout(b, 4, 4, 64, 0), layout(b, 4, 4, 64, 0));
    assert_eq!(
        run(accelerator.as_mut(), &taken).map(|(largest, count, _)| (largest, count)),
        Ok((0, 0))
    );
}

/// A destination whose planes do not start on 4-byte boundaries is staged, and the same: the
/// bytes the reference writes, and nothing else
#[test]
fn a_misaligned_destination_is_staged_and_the_same() {
    for (gpu, backend) in gpus() {
        let mut accelerator = backend.create_accelerator().unwrap();
        for format in [ScalerFormat::Nv12VideoRange, ScalerFormat::P010FullRange] {
            let case = Case {
                name: format!("{} skewed", name(format)),
                source: layout(ScalerFormat::Bgra, 40, 30, 64, 0),
                destination: layout(format, 50, 34, 64, 2),
                flags: 0x1000,
                crop: FixedRect::default(),
                rect: Rect {
                    x: 3,
                    y: 1,
                    width: 41,
                    height: 31,
                },
            };
            let (largest, _, _) = run(accelerator.as_mut(), &case).unwrap();
            assert!(largest <= 1, "{gpu}: {}: {largest}", case.name);
        }
    }
}

/// The platform the library runs on.
type P = crate::DefaultPlatform;

/// A surface of the platform's, as the mapper gives the guest's, its lends, and the writes of
/// its memory by a device it was told of.
struct Platform {
    platform: Arc<P>,
    surface: <P as reims_vgpu_core::platform::Platform>::Surface,
    layout: SurfaceLayout,
    memory: Option<Box<dyn DeviceMemory>>,
    lends: AtomicUsize,
    device_writes: AtomicUsize,
}

impl GuestSurface for Platform {
    fn layout(&self) -> SurfaceLayout {
        self.layout.clone()
    }

    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
        self.lends.fetch_add(1, Ordering::Relaxed);
        self.platform
            .lend_surface_bytes(&self.surface, write)
            .map_err(|e| SurfaceError::Unreachable(e.to_owned()))
    }

    fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
        self.lends.fetch_add(1, Ordering::Relaxed);
        self.platform
            .lend_surface_bytes_for_reading(&self.surface, read)
            .map_err(|e| SurfaceError::Unreachable(e.to_owned()))
    }

    fn device_memory(&self) -> Option<&dyn DeviceMemory> {
        self.memory.as_deref()
    }

    fn device_wrote(&self) {
        self.device_writes.fetch_add(1, Ordering::Relaxed);
    }
}

/// A surface of the platform's of `layout`, holding `bytes`, read-only for its clients where
/// `client_direction` says.
fn platform_surface(
    layout: SurfaceLayout,
    bytes: &[u8],
    client_direction: Option<u32>,
) -> Platform {
    let platform = P::process();
    let of = FormatOf::of(&layout).unwrap();
    let alloc = bytes.len().next_multiple_of(16384);
    let planes = (of.format.packing() == Packing::Biplanar420).then(|| {
        (of.planes.iter().zip(&layout.planes))
            .zip([1, 2])
            .map(|((span, plane), elements)| SurfacePlaneSpec {
                base: span.offset.to_u64(),
                offset: span.offset.to_u64(),
                width: u64::from(plane.width),
                height: u64::from(plane.height),
                size: (span.bytes_per_row * span.rows).to_u64(),
                bytes_per_row: span.bytes_per_row.to_u64(),
                bytes_per_element: (elements * of.format.bytes_per_sample()).to_u64(),
                element_width: 1,
                element_height: 1,
                ..SurfacePlaneSpec::default()
            })
            .collect()
    });
    let spec = SurfaceSpec {
        planes,
        pixel_format: layout.pixel_format.0,
        alloc_size: alloc.to_u64(),
        width: u64::from(layout.width),
        height: u64::from(layout.height),
        bytes_per_element: match of.format.packing() {
            Packing::Rgb32(_) | Packing::Packed422(_) => 4,
            Packing::Biplanar420 => of.format.bytes_per_sample().to_u64(),
        },
        bytes_per_row: of.planes[0].bytes_per_row.to_u64(),
        element_width: match of.format.packing() {
            Packing::Packed422(_) => 2,
            _ => 1,
        },
        element_height: 1,
        client_direction,
        ..SurfaceSpec::default()
    };
    let surface = platform.surfaces().create(&spec).unwrap();
    // Refused for a surface read-only for its clients where the platform refuses its CPU's
    // writes too: its bytes stay zeros.
    let _ =
        platform.lend_surface_bytes(&surface, &mut |to| to[..bytes.len()].copy_from_slice(bytes));
    let memory = platform.surface_device_memory(&surface).ok();
    Platform {
        platform,
        surface,
        layout,
        memory,
        lends: AtomicUsize::new(0),
        device_writes: AtomicUsize::new(0),
    }
}

/// A platform surface's bytes, through a lend.
fn snapshot(s: &Platform) -> Vec<u8> {
    let mut out = Vec::new();
    s.platform
        .lend_surface_bytes_for_reading(&s.surface, &mut |b| out.extend_from_slice(b))
        .unwrap();
    out
}

/// Surfaces the process's platform makes, whose memory it lends to devices, are reached in place:
/// no lend of either, the reference's bytes, and the destination told its memory was written; one
/// the guest made read-only for its clients is never written: staged, its write lend refused,
/// kIOReturnNotWritable, its bytes as they were
#[test]
fn platform_surfaces_are_reached_in_place() {
    for (gpu, backend) in gpus() {
        let mut accelerator = backend.create_accelerator().unwrap();
        for (from, to) in [
            (ScalerFormat::Nv12VideoRange, ScalerFormat::Bgra),
            (ScalerFormat::Bgra, ScalerFormat::P010VideoRange),
            (ScalerFormat::Yuyv, ScalerFormat::Nv12FullRange),
        ] {
            let (sl, dl) = (layout(from, 64, 48, 64, 0), layout(to, 96, 72, 64, 0));
            let source = platform_surface(sl.clone(), &picture_bytes(&sl, None), None);
            let grey = picture_bytes(&dl, Some([0.5, 0.5, 0.5, 1.0]));
            let destination = platform_surface(dl.clone(), &grey, None);
            let in_place = source.memory.is_some() && destination.memory.is_some();
            let transform = Transform {
                flags: TransformFlags(0x1000),
                ..Transform::default()
            };
            let plan = Plan::new(&transform, &sl, &dl).unwrap();
            let mut expected = snapshot(&destination);
            reference::run(&plan, &snapshot(&source), &mut expected).unwrap();
            let mut job = TransformJob {
                transform: &transform,
                plan: plan.clone(),
                source: &source,
                destination: &destination,
                hold: GuestHold::nothing(),
                inputs: Default::default(),
                outputs: Default::default(),
            };
            assert_eq!(accelerator.transform(&mut job), io_return::SUCCESS, "{gpu}");
            let (largest, _) =
                difference(&plan.destination_format, &snapshot(&destination), &expected);
            let lends =
                source.lends.load(Ordering::Relaxed) + destination.lends.load(Ordering::Relaxed);
            println!(
                "{gpu}: {} to {} on surfaces of the platform's: largest {largest}, {lends} lends{}",
                name(from),
                name(to),
                if in_place { ", in place" } else { ", staged" }
            );
            assert!(largest <= 1, "{gpu}: {largest}");
            if in_place
                && gpu.starts_with(if cfg!(target_os = "macos") {
                    "Metal"
                } else {
                    "Vulkan"
                })
            {
                assert_eq!(lends, 0, "{gpu}: in place, no lend");
                assert_eq!(
                    destination.device_writes.load(Ordering::Relaxed),
                    1,
                    "{gpu}: told of the write"
                );
            }
        }
        // A destination the guest made read-only for its clients: lent to no device, so staged,
        // and written only where the platform's own lend takes the CPU's writes (a surface of its
        // own memory on macOS does; none over the guest's pages does). The Vulkan platform on
        // Linux takes the direction only for a surface over a guest's pages, whose lends to
        // devices and for writing it refuses (its own tests), so a surface of its own memory is
        // no such surface there.
        let b = ScalerFormat::Bgra;
        let (sl, dl) = (layout(b, 32, 32, 64, 0), layout(b, 32, 32, 64, 0));
        let source = platform_surface(sl.clone(), &picture_bytes(&sl, None), None);
        let grey = picture_bytes(&dl, Some([0.5, 0.5, 0.5, 1.0]));
        let destination = platform_surface(dl.clone(), &grey, Some(2));
        if cfg!(target_os = "linux") && destination.memory.is_some() {
            println!("{gpu}: a read-only destination: no such surface of the platform's own");
            continue;
        }
        assert!(destination.memory.is_none(), "{gpu}: lent to no device");
        let writable = (destination.platform)
            .lend_surface_bytes(&destination.surface, &mut |_| {})
            .is_ok();
        let before = snapshot(&destination);
        let transform = Transform::default();
        let mut job = TransformJob {
            transform: &transform,
            plan: Plan::new(&transform, &sl, &dl).unwrap(),
            source: &source,
            destination: &destination,
            hold: GuestHold::nothing(),
            inputs: Default::default(),
            outputs: Default::default(),
        };
        let status = accelerator.transform(&mut job);
        if writable {
            assert_eq!(status, io_return::SUCCESS, "{gpu}");
        } else {
            assert_eq!(status, io_return::NOT_WRITABLE, "{gpu}");
            assert_eq!(snapshot(&destination), before, "{gpu}: as it was");
        }
        assert_eq!(
            destination.device_writes.load(Ordering::Relaxed),
            0,
            "{gpu}: staged, written in its lend alone"
        );
    }
}

#[cfg(target_os = "macos")]
mod native;

mod firsts;
