use reims_vgpu_video::vaapi::ImportWrite;
use reims_vgpu_vulkan::core::testkit::impls::device::DMA_BUF_BUFFER;
use reims_vgpu_vulkan::core::testkit::{FakePlatform, Step};

use super::*;

/// iHD's answer of NV12 (docs/video/bugs.md U26): rows of whole alignments of 64 bytes alone.
const WHOLE_ROWS: ImportWrite = ImportWrite {
    alignment: 64,
    whole_rows: true,
};

/// The display's import check's answers: NV12 and P010 as iHD's, and `xyuv` of XYUV.
fn answers(xyuv: Result<ImportWrite, String>) -> ImportWrites {
    ImportWrites::of_answers(Ok(WHOLE_ROWS), Ok(WHOLE_ROWS)).with_xyuv(xyuv)
}

/// A route on a stand-in device, where the check answered `xyuv` of XYUV.
fn route_on(
    platform: &Arc<FakePlatform>,
    xyuv: Result<ImportWrite, String>,
) -> Result<XyuvGathers<FakePlatform>, String> {
    let pipeline = Arc::new(GatherPipeline::<FakePlatform>::new(&platform.device()));
    route(
        &answers(xyuv),
        &pipeline,
        &crate::video::runtime::PlatformRuntime::of(platform),
    )
}

/// A decoder's 4:4:4 pictures go by the GPU where the display's import check passed XYUV, at
/// any alignment and either way of rows, and by the CPU where it did not, saying why: the
/// check's own answer, not the driver's name, decides.
#[test]
fn the_route_is_the_import_check_s_xyuv_answer() {
    let platform = Arc::new(FakePlatform::new());
    for passed in [WHOLE_ROWS, ImportWrite::at(256), ImportWrite::at(1)] {
        let gathers = route_on(&platform, Ok(passed)).expect("a route where XYUV passed");
        assert!(gathers.takes_pictures(), "{passed:?}");
    }
    let refused = route_on(&platform, Err("a wrong byte".to_owned()))
        .err()
        .expect("no route where XYUV failed");
    assert_eq!(
        refused,
        "the driver's video processing writes no XYUV import right: a wrong byte"
    );
    let unanswered = ImportWrites::of_answers(Ok(WHOLE_ROWS), Ok(WHOLE_ROWS));
    let pipeline = Arc::new(GatherPipeline::<FakePlatform>::new(&platform.device()));
    let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
    assert!(route(&unanswered, &pipeline, &runtime).is_err());
}

/// A route makes nothing before its first picture.
#[test]
fn a_route_makes_nothing_before_its_first_picture() {
    let platform = Arc::new(FakePlatform::new());
    let pipeline = Arc::new(GatherPipeline::<FakePlatform>::new(&platform.device()));
    let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
    let mark = platform.call_log().mark();
    let gathers = route(&answers(Ok(WHOLE_ROWS)), &pipeline, &runtime).expect("a route");
    assert!(gathers.scratch.is_none());
    assert_eq!(platform.call_log().sels_since(mark), Vec::<&str>::new());
}

/// A scratch's rows hold the even width's pixels, four bytes each, rounded up to the alignment
/// a driver that writes whole rows of it alone takes (module doc), and its rows are the even
/// height's.
#[test]
fn a_scratch_s_rows_are_its_pixels_on_the_target_alignment() {
    assert_eq!(TARGET_ALIGNMENT, 256);
    assert_eq!(scratch_shape((640, 360)), (2560, 360));
    assert_eq!(scratch_shape((1280, 720)), (5120, 720));
    assert_eq!(scratch_shape((1366, 766)), (5632, 766));
    assert_eq!(scratch_shape((1920, 1080)), (7680, 1080));
    assert_eq!(scratch_shape((2, 2)), (256, 2));
}

/// A picture written by the GPU goes no further; one declined or ended is split by the CPU, its
/// reason logged, an ended route's saying so.
#[test]
fn a_picture_the_gpu_did_not_write_is_split_by_the_cpu_with_why() {
    assert_eq!(Gathered::Written.written(), Ok(()));
    assert_eq!(
        Gathered::Declined("an odd crop".to_owned()).written(),
        Err("an odd crop".to_owned())
    );
    assert_eq!(
        Gathered::Ended("no scratch".to_owned()).written(),
        Err("the GPU's writes ended: no scratch".to_owned())
    );
}

/// Where the device makes no dma-buf scratch, the route ends: the decoder takes no later 4:4:4
/// picture by the GPU (`VaGuestDecoder`'s `lands_4_4_4_by_gpu`), and its pictures are split by
/// the CPU, as where the import check refused XYUV.
#[test]
fn a_scratch_the_device_refuses_ends_the_route() {
    let platform = Arc::new(FakePlatform::new());
    platform.script().push(DMA_BUF_BUFFER, Step::Nil);
    let mut gathers = route_on(&platform, Ok(WHOLE_ROWS)).expect("a route");
    let why = gathers.scratch_for((640, 360)).expect_err("no scratch");
    assert_eq!(why, "the device makes no scratch of a dma-buf");
    assert!(gathers.scratch.is_none());
    assert_eq!(
        gathers.end(why),
        Gathered::Ended("the device makes no scratch of a dma-buf".to_owned())
    );
    assert!(!gathers.takes_pictures());
}

/// A scratch the device makes but exports as no dma-buf ends the route too, and the buffer goes
/// with the refusal.
#[test]
fn a_scratch_the_device_exports_no_dma_buf_of_ends_the_route() {
    let platform = Arc::new(FakePlatform::new());
    let mut gathers = route_on(&platform, Ok(WHOLE_ROWS)).expect("a route");
    let mark = platform.call_log().mark();
    assert!(gathers.scratch_for((640, 360)).is_err());
    assert!(
        platform
            .call_log()
            .sels_since(mark)
            .contains(&DMA_BUF_BUFFER)
    );
    assert!(gathers.scratch.is_none());
}

/// This thread's CPU time (`CLOCK_THREAD_CPUTIME_ID`).
fn thread_cpu() -> Duration {
    let mut at = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `at` is a timespec, which `clock_gettime` writes.
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut at) },
        0
    );
    Duration::new(
        u64::try_from(at.tv_sec).expect("seconds"),
        u32::try_from(at.tv_nsec).expect("nanoseconds"),
    )
}

/// A 4:4:4 picture the driver decoded into an XYUV surface of its own, written by the GPU into a
/// '444v' surface's two planes (the route: the driver's video processing into the decoder's
/// scratch, then `videoGather444`), is the CPU's split of the same picture's read back byte for
/// byte: each plane's crop rows, and every other byte of the surface as its fill left it. The
/// surface is a Shared buffer of the device's, the luma plane 32 bytes into it and the chroma
/// plane past it on 64 bytes, rows of the width's bytes rounded up to 16, as CoreVideo lays a
/// surface out. The HEVC 4:4:4 streams in `RVG_VA_STREAMS`'s directories, their first
/// `RVG_XYUV_PICTURES` pictures (30 by default); a picture the route declines is counted and
/// named. Each picture's CPU time on this thread and its wall time are printed for both ways:
/// the GPU's, from the driver's write to the pass's end, and the CPU's, its read back and split.
/// Where the import check refused XYUV, the route is none and the test says so.
#[test]
#[ignore = "needs a VA-API GPU and HEVC 4:4:4 streams: run on the VA-API host"]
#[allow(clippy::disallowed_methods)] // a measurement: the wall clock's time
fn a_4_4_4_picture_written_by_the_gpu_is_the_cpus_split() {
    use reims_vgpu_video::vaapi::Display;
    use reims_vgpu_vulkan::RtPlatform;

    use super::super::tests::{UNTOUCHED, each_picture, plane_rows, usize_of, va_streams};

    let most = std::env::var("RVG_XYUV_PICTURES").map_or(30, |n| n.parse().expect("a count"));
    let counted = || {
        (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
            .find(|&(code, _)| code == super::super::super::gather::GATHERED_CODE)
            .map_or(0, |(_, n)| n)
    };
    let streams = va_streams();
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(
            &device,
            crate::graphics::video_choice::tested_gpu::Arm::VaApi,
        );
        let gpu = device.host_gpu();
        let node = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci);
        let node = node.expect("a render node");
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(&node, Arc::clone(&runtime)).expect("a VA display");
        let checked = display.import_writes();
        eprintln!(
            "the import check: NV12 {:?}, P010 {:?}, XYUV {:?}",
            checked.nv12(),
            checked.p010(),
            checked.xyuv()
        );
        let pipeline = Arc::new(GatherPipeline::<RtPlatform>::new(&device));
        let mut gathers = match route(&checked, &pipeline, &runtime) {
            Ok(gathers) => gathers,
            Err(why) => {
                eprintln!("no route by the GPU: {why}");
                return;
            }
        };
        let writes = GuestWrites::new(&display);
        let before = counted();
        let (mut pictures, mut declined) = (0usize, Vec::new());
        // Each written picture's CPU and wall times, the GPU's way then the CPU's.
        let mut times: Vec<[Duration; 4]> = Vec::new();
        let mut each = |stream: &std::path::Path, id: u64, frame: &Frame| -> bool {
            if !frame.is_4_4_4() {
                return false;
            }
            let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
            let (luma_pitch, chroma_pitch) =
                (width.next_multiple_of(16), (2 * width).next_multiple_of(16));
            let luma_at = 32;
            let chroma_at = (luma_at + luma_pitch * height).next_multiple_of(64);
            let len = chroma_at + chroma_pitch * height;
            let surface = (device.new_buffer_of_length(OnFault::Return, len, 0))
                .expect("no raise")
                .expect("a Shared buffer for the surface");
            // SAFETY: the surface's `len` bytes, Shared, which nothing else writes before the
            // route's pass, and which the pass has finished writing when they are read.
            let bytes = || unsafe {
                std::slice::from_raw_parts_mut(surface.contents().as_ptr::<u8>(), usize_of(len))
            };
            bytes().fill(UNTOUCHED);
            let plane = |offset: u64, pitch: u64, row_bytes: u64| TargetPlane {
                offset: usize_of(offset),
                bytes_per_row: usize_of(pitch),
                rows: u32::try_from(height).expect("rows"),
                row_bytes: usize_of(row_bytes),
            };
            let planes = [
                plane(luma_at, luma_pitch, width),
                plane(chroma_at, chroma_pitch, 2 * width),
            ];
            let (cpu_at, wall_at) = (thread_cpu(), std::time::Instant::now());
            let gathered =
                (gathers.land(frame, &writes, &planes, Some(&surface))).expect("the frame's write");
            let (gpu_cpu, gpu_wall) = (thread_cpu() - cpu_at, wall_at.elapsed());
            match gathered {
                Gathered::Written => {}
                Gathered::Declined(why) => {
                    declined.push(format!("{} picture {id}: {why}", stream.display()));
                    return false;
                }
                Gathered::Ended(why) => panic!("the route ended: {why}"),
            }
            let mut cpu = vec![UNTOUCHED; usize_of(len)];
            let (y, uv) = cpu.split_at_mut(usize_of(chroma_at));
            let (cpu_at, wall_at) = (thread_cpu(), std::time::Instant::now());
            (frame.read_back().expect("the read back"))
                .copy_4_4_4_into(
                    plane_rows(&mut y[usize_of(luma_at)..], luma_pitch, height, width),
                    plane_rows(uv, chroma_pitch, height, 2 * width),
                )
                .expect("the CPU's split");
            times.push([gpu_cpu, gpu_wall, thread_cpu() - cpu_at, wall_at.elapsed()]);
            let gpu = bytes();
            if let Some(first) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
                panic!(
                    "{} picture {id}, {width}x{height}, luma at {luma_at} in rows of \
                     {luma_pitch}, chroma at {chroma_at} in rows of {chroma_pitch}: byte \
                     {first} is {:#x} by the GPU and {:#x} by the CPU's split",
                    stream.display(),
                    gpu[first],
                    cpu[first]
                );
            }
            pictures += 1;
            true
        };
        let refused = each_picture(&display, &streams, most, true, &mut each);
        eprintln!(
            "{pictures} 4:4:4 pictures of {} streams written by the GPU as split by the CPU; \
             declined: {declined:?}; refused: {refused:?}",
            streams.len()
        );
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let what = [
            "the GPU's way, CPU time",
            "the GPU's way, wall time",
            "the CPU's way, CPU time",
            "the CPU's way, wall time",
        ];
        for (at, what) in what.iter().enumerate() {
            let mut each: Vec<Duration> = times.iter().map(|t| t[at]).collect();
            each.sort();
            if let (Some(least), Some(most)) = (each.first(), each.last()) {
                let sum: Duration = each.iter().sum();
                eprintln!(
                    "{what}: {} pictures, {:.3} ms least, {:.3} ms median, {:.3} ms mean, \
                     {:.3} ms at most",
                    each.len(),
                    ms(*least),
                    ms(each[each.len() / 2]),
                    ms(sum) / each.len() as f64,
                    ms(*most)
                );
            }
        }
        assert!(pictures > 0, "no picture written: {declined:?}");
        assert_eq!(counted(), before + pictures as u64);
    });
}
