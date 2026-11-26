//! The plane draws on the default adapter, against `vaGetImage`'s read back of each picture,
//! and the zeros their targets owe.

use std::ffi::c_void;
use std::path::PathBuf;
use std::ptr::null_mut;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::core::runtime::{Runtime as _, StdRuntime};

use super::super::d3d12::{
    Com, Device, Guid, Hresult, SharedTexture, check, default_adapter_device_references,
    default_adapter_device_references_once_let_go, guid, process_gpu_memory_settled,
};
use super::super::display::Display;
use super::super::frame::{Frame, PlaneRows};
use super::super::h264::H264Decoder;
use super::super::hevc::HevcDecoder;
use super::super::readback::{ReadbackWay, read_back_by};
use super::super::surface::SurfaceFormat;
use super::super::teardown::Stream;
use super::{PlaneDraw, PlaneTargets, covers};

/// The streams of the directories `RVG_VA_STREAMS` lists (as `PATH` lists them): each H.264
/// (`.264`) and HEVC (`.265`) file, in name order.
fn streams() -> Vec<PathBuf> {
    let dirs =
        std::env::var_os("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    let mut streams: Vec<PathBuf> = std::env::split_paths(&dirs)
        .flat_map(|dir| std::fs::read_dir(dir).expect("a streams directory"))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path.to_string_lossy();
            name.ends_with(".264") || name.ends_with(".265")
        })
        .collect();
    streams.sort();
    assert!(!streams.is_empty(), "no streams in {dirs:?}");
    streams
}

/// `frame`'s crop as `vaGetImage` reads it back: the luma's rows and the chroma's, each row's
/// bytes with nothing between them.
fn got_image(frame: &Frame) -> [Vec<u8>; 2] {
    let sample = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (frame.width() as usize, frame.height() as usize);
    let rows = [
        (width * sample, height),
        (width.div_ceil(2) * 2 * sample, height.div_ceil(2)),
    ];
    let mut planes = rows.map(|(row, rows)| vec![0u8; row * rows]);
    let [luma, chroma] = &mut planes;
    let back = frame.read_back().expect("a read back");
    back.copy_into(plane(luma, rows[0]), plane(chroma, rows[1]))
        .expect("its copy");
    planes
}

/// `bytes` as a plane of `rows.1` rows of `rows.0` bytes, with nothing between them.
fn plane(bytes: &mut [u8], (row, rows): (usize, usize)) -> PlaneRows<'_> {
    PlaneRows {
        bytes,
        bytes_per_row: row,
        rows,
        row_bytes: row,
    }
}

/// The times of a stream's draws, in milliseconds: the first, and the rest's mean and most.
fn times(draws: &[Duration]) -> String {
    let ms = |time: Duration| time.as_secs_f64() * 1e3;
    let Some((first, rest)) = draws.split_first() else {
        return "none".to_owned();
    };
    let rest_mean = match rest.len() {
        0 => 0.0,
        n => ms(rest.iter().sum::<Duration>()) / n as f64,
    };
    let all_mean = ms(draws.iter().sum::<Duration>()) / draws.len() as f64;
    let most = rest.iter().copied().max().map_or(0.0, ms);
    format!(
        "{} draws: mean {all_mean:.3} ms, the first {:.3} ms, the rest's mean {rest_mean:.3} ms \
         and most {most:.3} ms",
        draws.len(),
        ms(*first)
    )
}

/// Every picture of each H.264 and HEVC stream of `RVG_VA_STREAMS`, 8 and 10 bits, drawn into
/// targets of its crop's size, made before its first picture, holds exactly what `vaGetImage`
/// reads back of its crop, the luma's samples and the chroma's pairs, as the CPU's copy writes
/// them; the draws, made before the first stream, say how long they took to make, and each
/// stream how long its targets took and its draws did, the first apart from the rest. Run
/// holding the GPU, on vaon12 on the default adapter.
#[test]
#[ignore = "needs vaon12 and test streams: run on the Windows host"]
fn each_picture_drawn_into_its_targets_is_what_get_image_reads_back() {
    // The tests' clock: std's, which the library does not read.
    let clock = StdRuntime;
    read_back_by(Some(ReadbackWay::GetImage));
    let display = Display::open_adapter(None, StdRuntime::shared()).expect("a VA display");
    let made = clock.now();
    let draw = PlaneDraw::new(None).expect("the plane draws");
    eprintln!(
        "d3d11: the draws made in {:.3} ms",
        clock.elapsed_since(made).as_secs_f64() * 1e3
    );
    let (mut pictures, mut differ) = (0usize, Vec::new());
    for stream in streams() {
        let name = stream
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let file = std::fs::read(&stream).expect("a stream");
        let hevc = name.ends_with(".265");
        let units = if hevc {
            crate::codec::hevc::nal::access_units(&file)
        } else {
            crate::codec::h264::nal::access_units(&file)
        };
        let (mut h264, mut h265) = if hevc {
            (None, Some(HevcDecoder::new_annex_b(&display)))
        } else {
            (Some(H264Decoder::new_annex_b(&display)), None)
        };
        let mut targets: Option<PlaneTargets> = None;
        let mut draws = Vec::new();
        let mut land = |frame: &Frame| {
            frame.sync().expect("the decode");
            let texture = frame.texture().expect("a picture in a shared texture");
            let size = (frame.width(), frame.height());
            let to = targets.get_or_insert_with(|| {
                let start = clock.now();
                let made = draw
                    .targets(size, texture.is_10_bit())
                    .expect("the targets");
                eprintln!(
                    "d3d11: {name}: {}x{} {}-bit targets made in {:.3} ms",
                    size.0,
                    size.1,
                    frame.bit_depth(),
                    clock.elapsed_since(start).as_secs_f64() * 1e3
                );
                made
            });
            assert_eq!(to.size(), size, "one size a stream");
            let start = clock.now();
            draw.draw(&texture, frame.crop_origin(), to, size)
                .expect("the draw");
            draws.push(clock.elapsed_since(start));
            let drawn = draw.read_back(to).expect("the targets' read back");
            let want = got_image(frame);
            for (plane, (drawn, want)) in ["luma", "chroma"].iter().zip(drawn.iter().zip(&want)) {
                assert_eq!(drawn.len(), want.len(), "{name}: the {plane}'s bytes");
                let wrong = drawn.iter().zip(want).filter(|(a, b)| a != b).count();
                if wrong > 0 {
                    differ.push(format!("{name}: picture {pictures}: {wrong} {plane} bytes"));
                }
            }
            pictures += 1;
        };
        for (id, unit) in units.iter().enumerate() {
            let id = u64::try_from(id).expect("an id");
            let frames = match (&mut h264, &mut h265) {
                (Some(decoder), _) => decoder.decode(unit, id),
                (_, Some(decoder)) => decoder.decode(unit, id),
                _ => unreachable!("a decoder"),
            }
            .expect("a decode");
            frames.iter().for_each(&mut land);
        }
        let flushed = match (&mut h264, &mut h265) {
            (Some(decoder), _) => decoder.flush(),
            (_, Some(decoder)) => decoder.flush(),
            _ => unreachable!("a decoder"),
        }
        .expect("the flush");
        flushed.iter().for_each(&mut land);
        drop(land);
        eprintln!("d3d11: {name}: {}", times(&draws));
    }
    eprintln!(
        "d3d11: {pictures} pictures drawn, {} differ: {differ:?}",
        differ.len()
    );
    assert!(pictures > 0);
    assert!(differ.is_empty(), "{differ:?}");
}

/// A shared texture of the default adapter's, of `ten_bit` samples or 8-bit ones, 64 by 48: a
/// picture's pool texture, its texels as they were made.
fn source(ten_bit: bool) -> Arc<SharedTexture> {
    let format = if ten_bit {
        SurfaceFormat::P010
    } else {
        SurfaceFormat::Nv12
    };
    let device = Device::new(None).expect("a Direct3D 12 device");
    let texture = device
        .shared_texture(format, 64, 48)
        .expect("a shared texture");
    Arc::new(texture)
}

/// The bytes of `targets`' planes past a picture of `extent` luma samples drawn from their first
/// texel, of `sample` bytes a luma sample: the luma's and the chroma's, each a plane of `size`
/// luma samples read back row after row with nothing between them.
fn past_picture(
    planes: &[Vec<u8>; 2],
    size: (usize, usize),
    extent: (usize, usize),
    sample: usize,
) -> [Vec<u8>; 2] {
    let half = |(w, h): (usize, usize)| (w.div_ceil(2), h.div_ceil(2));
    let shapes = [
        (size.0 * sample, extent.0 * sample, extent.1),
        (
            half(size).0 * 2 * sample,
            half(extent).0 * 2 * sample,
            half(extent).1,
        ),
    ];
    std::array::from_fn(|n| {
        let (row, drawn_row, drawn_rows) = shapes[n];
        (planes[n].chunks(row).enumerate())
            .flat_map(|(y, bytes)| {
                if y < drawn_rows {
                    &bytes[drawn_row..]
                } else {
                    bytes
                }
            })
            .copied()
            .collect()
    })
}

/// Targets of 64 by 48 luma samples, of 8-bit samples and of 10-bit ones, cleared to bytes of
/// their own, then drawn a picture of 32 by 16 from a pool texture: where they owe zeros, the
/// draw clears them first, so every byte past the picture reads zero, and they owe none after;
/// where they owe none, every byte past the picture is as it was. Run holding the GPU, on the
/// default adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn a_draw_short_of_its_targets_leaves_zeros_past_the_picture_only_where_they_owe_them() {
    let draw = PlaneDraw::new(None).expect("the plane draws");
    for ten_bit in [false, true] {
        let texture = source(ten_bit);
        let sample = if ten_bit { 2 } else { 1 };
        for owed in [true, false] {
            let to = draw.targets((64, 48), ten_bit).expect("the targets");
            draw.clear_to(&to, [255.0; 4]).expect("the targets' clear");
            let before = draw.read_back(&to).expect("the targets' read back");
            let kept = past_picture(&before, (64, 48), (32, 16), sample);
            assert!(
                kept.iter().all(|plane| plane.iter().any(|&b| b != 0)),
                "{ten_bit}: the clear wrote bytes of its own"
            );
            if owed {
                to.owe_zeros();
            }
            draw.draw(&texture, (0, 0), &to, (32, 16))
                .expect("the draw");
            assert!(
                !to.owes_zeros(),
                "{ten_bit}, {owed}: none owed after the draw"
            );
            let after = draw.read_back(&to).expect("the targets' read back");
            let past = past_picture(&after, (64, 48), (32, 16), sample);
            for (plane, (past, kept)) in ["luma", "chroma"].iter().zip(past.iter().zip(&kept)) {
                if owed {
                    assert!(
                        past.iter().all(|&b| b == 0),
                        "{ten_bit}: the {plane}'s bytes past the picture read zero"
                    );
                } else {
                    assert_eq!(
                        past, kept,
                        "{ten_bit}: the {plane}'s bytes past the picture"
                    );
                }
            }
        }
    }
}

const IID_IDXGI_KEYED_MUTEX: Guid = guid(
    0x9d8e_1289,
    0xd7b3,
    0x465f,
    [0x81, 0x26, 0x25, 0x0e, 0x34, 0x9a, 0xf8, 0x5d],
);

/// The misc flags of `texture`, an `ID3D11Texture2D` (`GetDesc`, its method 10).
fn misc_flags(texture: &Com) -> u32 {
    let mut desc = super::Texture2dDesc {
        width: 0,
        height: 0,
        mip_levels: 0,
        array_size: 0,
        format: 0,
        sample_count: 0,
        sample_quality: 0,
        usage: 0,
        bind_flags: 0,
        cpu_access_flags: 0,
        misc_flags: 0,
    };
    // SAFETY: the method is ID3D11Texture2D's `GetDesc`, which writes the description.
    unsafe {
        let get: unsafe extern "system" fn(*mut c_void, *mut super::Texture2dDesc) =
            texture.method(10);
        get(texture.raw(), &mut desc);
    }
    desc.misc_flags
}

/// A picture's targets, of 8-bit samples and of 10-bit ones, are textures of the native device
/// shared by NT handle with no keyed mutex (`D3D11_RESOURCE_MISC_SHARED` and
/// `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`, 0x802), which the draws' device opens
/// (`OpenSharedResource1`, as [`PlaneDraw::targets`] makes them): each answers those flags alone
/// and no `IDXGIKeyedMutex`. Run holding the GPU, on the default adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn the_targets_are_shared_by_nt_handle_with_no_keyed_mutex() {
    let draw = PlaneDraw::new(None).expect("the plane draws");
    for ten_bit in [false, true] {
        let to = draw
            .targets((64, 48), ten_bit)
            .expect("the targets, opened");
        for (plane, target) in ["luma", "chroma"].iter().zip(&to.planes) {
            assert_eq!(
                misc_flags(&target.native),
                0x802,
                "{ten_bit}: the {plane}'s flags"
            );
            assert!(
                (target.native)
                    .query(&IID_IDXGI_KEYED_MUTEX, "IDXGIKeyedMutex")
                    .is_err(),
                "{ten_bit}: the {plane}'s texture answers a keyed mutex"
            );
        }
    }
}

/// [`PlaneDraw::zero_owed`] clears targets that owe zeros, and they owe none after; it keeps
/// the bytes of targets that owe none, as of targets another device has since said it wrote
/// whole ([`PlaneTargets::written`]). Run holding the GPU, on the default adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn zero_owed_clears_only_targets_that_owe_zeros() {
    let draw = PlaneDraw::new(None).expect("the plane draws");
    for ten_bit in [false, true] {
        let to = draw.targets((64, 48), ten_bit).expect("the targets");
        draw.clear_to(&to, [255.0; 4]).expect("the targets' clear");
        let before = draw.read_back(&to).expect("the targets' read back");
        assert!(before.iter().all(|plane| plane.iter().any(|&b| b != 0)));
        draw.zero_owed(&to).expect("no clear");
        assert_eq!(
            draw.read_back(&to).expect("a read back"),
            before,
            "{ten_bit}: none owed"
        );
        to.owe_zeros();
        to.written();
        draw.zero_owed(&to).expect("no clear");
        assert_eq!(
            draw.read_back(&to).expect("a read back"),
            before,
            "{ten_bit}: written"
        );
        to.owe_zeros();
        draw.zero_owed(&to).expect("the clear");
        assert!(!to.owes_zeros(), "{ten_bit}: none owed after");
        let after = draw.read_back(&to).expect("a read back");
        assert!(
            after.iter().all(|plane| plane.iter().all(|&b| b == 0)),
            "{ten_bit}: every byte reads zero"
        );
    }
}

const IID_IDXGI_DEVICE: Guid = guid(
    0x54ec_77fa,
    0x1377,
    0x44e6,
    [0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c],
);
const IID_IDXGI_ADAPTER3: Guid = guid(
    0x6459_67a4,
    0x1392,
    0x4310,
    [0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd],
);
// The vtable indices of the methods called (the SDK's `<interface>Vtbl` structures).
const DXGI_DEVICE_GET_ADAPTER: usize = 7;
const ADAPTER3_QUERY_VIDEO_MEMORY_INFO: usize = 14;
const DXGI_MEMORY_SEGMENT_GROUP_LOCAL: u32 = 0;
const DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL: u32 = 1;

/// `DXGI_QUERY_VIDEO_MEMORY_INFO`.
#[repr(C)]
#[derive(Default)]
struct VideoMemoryInfo {
    budget: u64,
    current_usage: u64,
    available_for_reservation: u64,
    current_reservation: u64,
}

/// The bytes of the native device's adapter the process holds: its current usage of the
/// adapter's local segment group and of its non-local one, as
/// `IDXGIAdapter3::QueryVideoMemoryInfo` answers for node 0.
fn memory_used(draw: &PlaneDraw) -> u64 {
    let state = draw.state.lock();
    let device = (state.native)
        .query(&IID_IDXGI_DEVICE, "IDXGIDevice")
        .expect("the native device's IDXGIDevice");
    let mut adapter = null_mut();
    // SAFETY: the method is IDXGIDevice's `GetAdapter`, which writes its output once.
    let result = unsafe {
        let get: unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> Hresult =
            device.method(DXGI_DEVICE_GET_ADAPTER);
        get(device.raw(), &mut adapter)
    };
    let adapter = Com::made("IDXGIDevice::GetAdapter", result, adapter)
        .expect("the device's adapter")
        .query(&IID_IDXGI_ADAPTER3, "IDXGIAdapter3")
        .expect("its IDXGIAdapter3");
    [
        DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
        DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,
    ]
    .into_iter()
    .map(|group| {
        let mut info = VideoMemoryInfo::default();
        // SAFETY: the method is IDXGIAdapter3's `QueryVideoMemoryInfo`, of node 0 and a segment
        // group, which writes a `DXGI_QUERY_VIDEO_MEMORY_INFO`.
        let result = unsafe {
            let query: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *mut VideoMemoryInfo,
            ) -> Hresult = adapter.method(ADAPTER3_QUERY_VIDEO_MEMORY_INFO);
            query(adapter.raw(), 0, group, &mut info)
        };
        check("IDXGIAdapter3::QueryVideoMemoryInfo", result).expect("the adapter's memory");
        info.current_usage
    })
    .sum()
}

/// Pairs of targets let go give back their memory as they go, with no draw or read back after
/// (the module doc's deferred destruction): 16 pairs of targets of 3840 by 2160 luma samples, of
/// 8-bit samples, hold at least their 16 times 12.4 MB of the adapter's memory more while they
/// live, and less than a tenth of what they held once let go. Run holding the GPU, on the default
/// adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn targets_let_go_give_back_their_memory() {
    const SIZE: (u32, u32) = (3840, 2160);
    let pair = u64::from(SIZE.0 * SIZE.1 + SIZE.0.div_ceil(2) * SIZE.1.div_ceil(2) * 2);
    let draw = PlaneDraw::new(None).expect("the plane draws");
    let before = memory_used(&draw);
    let pairs: Vec<PlaneTargets> = (0..16)
        .map(|_| draw.targets(SIZE, false).expect("the targets"))
        .collect();
    let held = memory_used(&draw).saturating_sub(before);
    assert!(
        held >= 16 * pair,
        "16 pairs hold {held} bytes more, of their {} bytes",
        16 * pair
    );
    drop(pairs);
    let left = memory_used(&draw).saturating_sub(before);
    assert!(
        left < held / 10,
        "{left} bytes of the {held} that 16 pairs held are left once they are let go"
    );
}

/// Pool textures let go give back their memory as they go, with no draw after (the module doc's
/// opens). A cycle makes 16 shared textures of 3840 by 2160 luma samples, of 8-bit samples, draws
/// from each once, as a session's pool is drawn from, and lets them go, as a pool is at its
/// session's end: while they live the draws keep an open of each, and once they are let go none
/// (the fix's own claim, by count). By memory, three cycles: the level after the last, against the
/// level after the first, is up by less than a tenth of what a cycle's 16 textures held, so that
/// what a pool left behind would add up past the bound, as DXGI counts the process's memory and
/// as Windows charges it (`super::super::d3d12::process_gpu_memory`): an open of each texture's NT
/// handle, which DXGI did not count, stayed charged for good, 16 times a texture's 12.4 MB a cycle
/// (`super::super::d3d12`'s module doc), where the draws' wrap of its resource leaves nothing. The
/// level after the first cycle is not asserted: on the UHD Graphics 630 that cycle left
/// 63,930,368 bytes as DXGI counts them, with one flush of the draws' contexts and with a second
/// after a wait for the GPU alike, of the 466,493,440 that its textures held in a shared heap with
/// an open of each, and of the 265,150,464 that they hold in no shared heap, wrapped; and
/// 64,036,864 bytes charged (**measured** 2026-11-26, driver 31.0.101.2145, potluck jobs
/// m3-max-75548, m3-max-75554 and m3-max-75658); the test prints both. Run holding the GPU, on the
/// default adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn pool_textures_let_go_give_back_their_memory() {
    const SIZE: (u32, u32) = (3840, 2160);
    let texture = u64::from(SIZE.0 * SIZE.1 + SIZE.0.div_ceil(2) * SIZE.1.div_ceil(2) * 2);
    let draw = PlaneDraw::new(None).expect("the plane draws");
    let to = draw.targets((64, 48), false).expect("the targets");
    let device = Device::new(None).expect("a Direct3D 12 device");
    // A cycle (above): what its textures held above the level it began at, and the level after,
    // as DXGI counts it and as Windows charges it.
    let cycle = |n: usize| -> (u64, u64, u64) {
        let start = memory_used(&draw);
        let textures: Vec<Arc<SharedTexture>> = (0..16)
            .map(|_| {
                let texture = (device.shared_texture(SurfaceFormat::Nv12, SIZE.0, SIZE.1))
                    .expect("a shared texture");
                Arc::new(texture)
            })
            .collect();
        for texture in &textures {
            draw.draw(texture, (0, 0), &to, (64, 48)).expect("the draw");
        }
        let held = memory_used(&draw).saturating_sub(start);
        assert!(
            held >= 16 * texture,
            "cycle {n}: 16 pool textures hold {held} bytes more, of their {} bytes",
            16 * texture
        );
        assert_eq!(
            draw.state.lock().opened.len(),
            16,
            "cycle {n}: an open of each texture"
        );
        drop(textures);
        assert_eq!(
            draw.state.lock().opened.len(),
            0,
            "cycle {n}: no open of a texture let go"
        );
        (held, memory_used(&draw), process_gpu_memory_settled())
    };
    let before = (memory_used(&draw), process_gpu_memory_settled());
    let (held, first, first_charged) = cycle(1);
    println!(
        "the first cycle left {} bytes of the {held} that its 16 pool textures held, and {} bytes \
         charged",
        first.saturating_sub(before.0),
        first_charged.saturating_sub(before.1)
    );
    cycle(2);
    let (held, last, last_charged) = cycle(3);
    let grown = last.saturating_sub(first);
    assert!(
        grown < held / 10,
        "{grown} bytes more after the third cycle than after the first, of the {held} that a \
         cycle's 16 pool textures held"
    );
    let grown = last_charged.saturating_sub(first_charged);
    assert!(
        grown < held / 10,
        "{grown} bytes more charged after the third cycle than after the first, of the {held} \
         that a cycle's 16 pool textures held"
    );
}

/// Pairs of targets let go of together share their flushes (the module doc's): four pairs of
/// 3840 by 2160 luma samples let go of on four threads while the draws' lock is held, as a draw
/// or another pair's flush holds it, wait for a flush begun after their release, and the first of
/// them makes one, once the lock is let go, that destroys all four: no drop returns before it,
/// and their memory goes as they do. Run holding the GPU, on the default adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn pairs_let_go_together_share_one_flush() {
    const SIZE: (u32, u32) = (3840, 2160);
    let draw = PlaneDraw::new(None).expect("the plane draws");
    let before = memory_used(&draw);
    let pairs: Vec<PlaneTargets> = (0..4)
        .map(|_| draw.targets(SIZE, false).expect("the targets"))
        .collect();
    let held = memory_used(&draw).saturating_sub(before);
    let lock = draw.state.lock();
    let returned = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (pairs.into_iter())
        .map(|pair| {
            let returned = Arc::clone(&returned);
            StdRuntime::thread("pair-drop", move || {
                drop(pair);
                returned.fetch_add(1, Ordering::SeqCst);
            })
            .expect("a thread")
        })
        .collect();
    // The tests' clock: std's, which the library does not read.
    let clock = StdRuntime;
    let asked = clock.now();
    while draw.flushes.owed.lock().asked < 4 {
        assert!(
            clock.elapsed_since(asked) < Duration::from_secs(10),
            "the four pairs ask for a flush"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        returned.load(Ordering::SeqCst),
        0,
        "no pair's drop returns before a flush begun after it"
    );
    drop(lock);
    for thread in threads {
        thread.join().expect("a pair's drop");
    }
    let owed = draw.flushes.owed.lock();
    assert_eq!(
        (owed.asked, owed.covered, owed.made),
        (4, 4, 1),
        "one flush for the four pairs"
    );
    drop(owed);
    let left = memory_used(&draw).saturating_sub(before);
    assert!(
        left < held / 10,
        "{left} bytes of the {held} that 4 pairs held are left once they are let go"
    );
}

/// A plane draw let go of lets go of the adapter's Direct3D 12 device, and of everything its
/// devices made on it: once a draw from a pool texture, its targets, the texture and the draws are
/// let go of, as a session's and its decoder's are at the video's teardown, the Direct3D 12
/// device made on the adapter (one a process, which `D3D12CreateDevice` answers while it lives)
/// has no reference but ours. Direct3D 11 destroys a released object only at a later flush of its
/// device's immediate context (the module doc): the draws' open of the texture, and its views,
/// would otherwise keep the Direct3D 11 device over Direct3D 12, and through it the Direct3D 12
/// device and the texture's memory, until the process exits. Run holding the GPU, on the default
/// adapter.
#[test]
#[ignore = "needs Direct3D 11 on the default adapter: run on the Windows host"]
fn a_plane_draw_let_go_leaves_no_reference_on_the_adapters_device() {
    assert_eq!(
        default_adapter_device_references(),
        0,
        "the adapter's Direct3D 12 device, made alone, has no reference but ours"
    );
    let draw = PlaneDraw::new(None).expect("the plane draws");
    for ten_bit in [false, true] {
        let texture = source(ten_bit);
        let to = draw.targets((64, 48), ten_bit).expect("the targets");
        draw.draw(&texture, (0, 0), &to, (64, 48))
            .expect("the draw");
        drop(to);
        drop(texture);
    }
    drop(draw);
    assert_eq!(
        default_adapter_device_references(),
        0,
        "the adapter's Direct3D 12 device, made once the plane draws are let go of, has no \
         reference but ours"
    );
}

/// The pictures drawn from at once in [`a_video_let_go_leaves_no_reference_on_the_adapters_device`].
const VIDEO_PICTURES: usize = 16;

/// A video let go of lets go of the adapter's Direct3D 12 device, and of everything vaon12 and the
/// draws' devices made on it: once a decoder decoded the first pictures of `RVG_VA_H264` into its
/// pool, each drawn from its shared texture into targets as the host's video draws it, and the
/// pictures, the targets, the decoder, the draws and the display are let go of, as they are at
/// the video's teardown, the Direct3D 12 device made on the adapter (one a process, which
/// `D3D12CreateDevice` answers while it lives) has no reference but ours, within 5 s of the
/// display's drop (its `vaTerminate` may be made on its call thread). Run holding the GPU, on
/// vaon12 on the default adapter.
///
/// **Measured** to fail on vaon12 of Mesa 26.2.7 over an Intel UHD Graphics 630, 2026-11-20: one
/// reference beside ours, as the driver leaves one of every decode session, where
/// [`a_plane_draw_let_go_leaves_no_reference_on_the_adapters_device`] passes (docs/video/bugs.md,
/// U31).
#[test]
#[ignore = "vaon12 of Mesa 26.2.7 fails it (docs/video/bugs.md, U31); needs RVG_VA_H264: run on \
            the Windows host"]
fn a_video_let_go_leaves_no_reference_on_the_adapters_device() {
    assert_eq!(
        default_adapter_device_references(),
        0,
        "the adapter's Direct3D 12 device, made alone, has no reference but ours"
    );
    let display = Display::open_adapter(None, StdRuntime::shared()).expect("a VA display");
    let stream = Stream::new(&display);
    let draw = PlaneDraw::new(None).expect("the plane draws");
    let mut decoder = stream.decoder(&display);
    let mut frames = Vec::new();
    for (id, sample) in (0u64..).zip(stream.samples.iter().take(VIDEO_PICTURES)) {
        frames.extend(decoder.decode(sample, id).expect("a decode"));
    }
    frames.extend(decoder.flush().expect("the flush"));
    assert!(!frames.is_empty(), "pictures decoded");
    let mut targets: Option<PlaneTargets> = None;
    for frame in &frames {
        frame.sync().expect("the decode");
        let texture = frame.texture().expect("a picture in a shared texture");
        let size = (frame.width(), frame.height());
        let to = targets.get_or_insert_with(|| {
            draw.targets(size, texture.is_10_bit())
                .expect("the targets")
        });
        draw.draw(&texture, frame.crop_origin(), to, size)
            .expect("the draw");
    }
    drop(targets);
    drop(frames);
    drop(decoder);
    drop(draw);
    drop(display);
    assert_eq!(
        default_adapter_device_references_once_let_go(),
        0,
        "the adapter's Direct3D 12 device, made once the video is let go of, has no reference \
         but ours"
    );
}

/// A picture's extent covers its targets where it is their size, or more, each way, and only
/// then.
#[test]
fn a_picture_covers_its_targets_only_at_their_size_each_way() {
    assert!(covers((64, 48), (64, 48)));
    assert!(covers((63, 47), (64, 48)));
    assert!(!covers((64, 48), (64, 47)));
    assert!(!covers((64, 48), (63, 48)));
    assert!(!covers((3840, 2160), (1920, 1080)));
}
