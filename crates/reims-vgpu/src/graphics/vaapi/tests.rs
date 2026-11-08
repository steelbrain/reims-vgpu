use reims_vgpu_video::vaapi::{ImportWrite, SurfaceCapabilities};
use reims_vgpu_vulkan::gpu::resource::{VulkanBuffer, VulkanTexture};

use super::*;

/// A VLD config of `profile`, decoding `rt_formats`, up to `max` with surfaces from `min`.
fn config(
    profile: Profile,
    rt_formats: u32,
    max: (u32, u32),
    min: Option<(u32, u32)>,
) -> ConfigCapabilities {
    ConfigCapabilities {
        profile,
        entrypoint: Entrypoint::VLD,
        rt_formats: Some(rt_formats),
        max_width: Some(max.0),
        max_height: Some(max.1),
        dec_slice_mode: None,
        rate_control: None,
        packed_headers: None,
        max_ref_frames: None,
        max_slices: None,
        surfaces: Ok(SurfaceCapabilities {
            min_width: min.map(|m| m.0),
            min_height: min.map(|m| m.1),
            ..SurfaceCapabilities::default()
        }),
        proc_filters: None,
    }
}

fn driver(configs: Vec<ConfigCapabilities>) -> Capabilities {
    Capabilities {
        vendor: "a stand-in".to_owned(),
        api_version: (1, 22),
        configs,
        image_formats: Vec::new(),
    }
}

/// The caps are the driver's answers: each profile it lists, at the least of their sizes, the
/// codecs' top levels and VA's picture-structure bounds, any bitstream place
#[test]
fn the_caps_are_the_driver_s_answers() {
    let caps = caps_of(&driver(vec![
        config(Profile::H264_MAIN, RT_YUV420, (4096, 4096), Some((64, 64))),
        config(Profile::H264_HIGH, RT_YUV420, (4096, 2304), None),
        config(
            Profile::HEVC_MAIN,
            RT_YUV420,
            (8192, 8192),
            Some((128, 128)),
        ),
        config(
            Profile::HEVC_MAIN10,
            RT_YUV420 | RT_YUV420_10,
            (8192, 4352),
            None,
        ),
    ]));
    assert_eq!(caps.len(), 2);
    let h264 = caps[0];
    assert_eq!(
        (h264.codec, h264.direction),
        (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE)
    );
    assert_eq!(
        h264.profiles,
        RGFX_VIDEO_H264_BASELINE | RGFX_VIDEO_H264_MAIN | RGFX_VIDEO_H264_HIGH
    );
    assert_eq!((h264.min_size, h264.max_size), ([64, 64], [4096, 2304]));
    assert_eq!(
        (h264.max_level, h264.max_dpb_pictures, h264.max_references),
        (62, 17, 16)
    );
    assert_eq!(
        (h264.formats, h264.bitstream_alignment),
        (RGFX_VIDEO_FORMAT_NV12, 1)
    );
    let hevc = caps[1];
    assert_eq!(hevc.codec, RGFX_VIDEO_CODEC_HEVC);
    assert_eq!(
        hevc.profiles,
        RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10
    );
    assert_eq!(
        hevc.formats,
        RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010
    );
    assert_eq!((hevc.min_size, hevc.max_size), ([128, 128], [8192, 4352]));
    assert_eq!(
        (hevc.max_level, hevc.max_dpb_pictures, hevc.max_references),
        (186, 16, 8)
    );
}

/// H.264's largest size keeps its pictures to `H264_MAX_MACROBLOCKS`: radeonsi's 4096x4096, which
/// it decodes nothing at (bugs.md, U28), and iHD's 8192x8192, whose pictures past 4096 wide it
/// refuses, are both answered 4096x4080, which holds 3840x2160 and 4096x2160; sizes within the
/// bound are the driver's
#[test]
fn h264_s_largest_size_stays_under_the_macroblocks_the_drivers_decode() {
    let largest = |max| {
        caps_of(&driver(vec![config(
            Profile::H264_HIGH,
            RT_YUV420,
            max,
            None,
        )]))[0]
            .max_size
    };
    assert_eq!(largest((4096, 4096)), [4096, 4080]);
    assert_eq!(largest((8192, 8192)), [4096, 4080]);
    assert_eq!(largest((4096, 2304)), [4096, 2304]);
    assert_eq!(largest((4080, 4096)), [4080, 4096]);
    for max in [(4096, 4096), (8192, 8192)] {
        let [width, height] = largest(max);
        assert!((width / 16) * (height / 16) <= H264_MAX_MACROBLOCKS);
        for (w, h) in [(3840, 2160), (4096, 2160)] {
            assert!(w <= width && h <= height, "{w}x{h} within {width}x{height}");
        }
    }
}

/// A profile whose config decodes no format of its own is left out, and a codec of none goes
/// unanswered: Main 10 that decodes only 8-bit, Baseline on ConstrainedBaseline alone
#[test]
fn a_profile_whose_config_decodes_none_of_its_formats_is_left_out() {
    let caps = caps_of(&driver(vec![
        config(
            Profile::H264_CONSTRAINED_BASELINE,
            RT_YUV420,
            (1920, 1088),
            None,
        ),
        config(Profile::HEVC_MAIN10, RT_YUV420, (8192, 8192), None),
    ]));
    assert_eq!(caps.len(), 1);
    assert_eq!(caps[0].profiles, RGFX_VIDEO_H264_BASELINE);
    assert_eq!(caps[0].min_size, [MIN_SIZE; 2]);
    assert!(caps_of(&driver(Vec::new())).is_empty());
}

/// The layout is granularity [2, 2] at the largest alignment the import check found for the
/// codec's formats, and zero where it found none
#[test]
fn the_layout_is_the_import_check_s_alignment() {
    let hevc = RgfxVideoCodecCaps {
        codec: RGFX_VIDEO_CODEC_HEVC,
        direction: RGFX_VIDEO_DECODE,
        formats: RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010,
        ..RgfxVideoCodecCaps::default()
    };
    let h264 = RgfxVideoCodecCaps {
        formats: RGFX_VIDEO_FORMAT_NV12,
        ..hevc
    };
    let writes = ImportWrites::of_answers(Ok(ImportWrite::at(64)), Ok(ImportWrite::at(256)));
    assert_eq!(
        (
            layout(&hevc, &writes).granularity,
            layout(&hevc, &writes).alignment
        ),
        ([2, 2], 256)
    );
    assert_eq!(layout(&h264, &writes).alignment, 64);
    let p010_only = ImportWrites::of_answers(Err("refused".to_owned()), Ok(ImportWrite::at(128)));
    assert_eq!(layout(&hevc, &p010_only).alignment, 128);
    assert_eq!(layout(&h264, &p010_only), RgfxVideoOutputLayout::default());
    // An encode's sources are read by the CPU: no layout is asked of them.
    let encode = RgfxVideoCodecCaps {
        direction: RGFX_VIDEO_ENCODE,
        ..hevc
    };
    assert_eq!(layout(&encode, &writes), RgfxVideoOutputLayout::default());
}

fn place(
    offset: u64,
    bytes_per_row: u64,
    row_bytes: u64,
    width: u64,
    height: u64,
    rows: u64,
) -> Place {
    Place {
        offset,
        bytes_per_row,
        row_bytes,
        width,
        height,
        rows_held: rows,
    }
}

/// A frame lands in place at the planes' own rows and starts, its crop's width and height each
/// rounded up to even as far as the planes hold them, `420v` or `x420` by its depth
#[test]
fn an_in_place_layout_holds_the_crop_s_even_rows_and_columns() {
    // 1080p NV12 in a surface holding the coded 1088 rows (544 chroma), rows of 2048.
    let luma = place(4096, 2048, 1920, 1920, 1080, 1088);
    let chroma = place(4096 + 2048 * 1088, 2048, 1920, 960, 540, 544);
    let layout = in_place_layout(luma, chroma, 8).unwrap();
    assert_eq!(layout.pixel_format, FourCc::new(*b"420v"));
    assert_eq!((layout.width, layout.height), (1920, 1080));
    assert_eq!(
        (layout.planes[0].offset, layout.planes[0].bytes_per_row),
        (4096, 2048)
    );
    assert_eq!(layout.planes[1].offset, 4096 + 2048 * 1088);
    assert_eq!(
        (layout.planes[1].width, layout.planes[1].height),
        (960, 540)
    );
    // 641x361 P010 in planes holding its even 642 columns and 362 rows.
    let luma = place(0, 1536, 1284, 641, 361, 362);
    let chroma = place(1536 * 362, 1536, 1284, 321, 181, 181);
    let layout = in_place_layout(luma, chroma, 10).unwrap();
    assert_eq!(layout.pixel_format, FourCc::new(*b"x420"));
    assert_eq!((layout.width, layout.height), (642, 362));
    // The same crop in planes of its odd size alone: as much as they hold.
    let luma = place(0, 1536, 1282, 641, 361, 361);
    let chroma = place(1536 * 361, 1536, 1284, 321, 181, 181);
    let layout = in_place_layout(luma, chroma, 10).unwrap();
    assert_eq!((layout.width, layout.height), (640, 361));
    assert!(in_place_layout(luma, chroma, 12).is_err());
}

fn room(row_bytes: u64, bytes_per_row: u64, rows: u64) -> Room {
    Room {
        row_bytes,
        bytes_per_row,
        rows,
    }
}

/// The GPU's copy takes the columns and rows the CPU's copy writes: the crop where the planes
/// hold it, else as much as each plane's rows and their bytes hold, the chroma plane's columns in
/// whole pairs
#[test]
fn the_gpu_copy_takes_the_crop_the_cpu_copy_writes() {
    // CoreVideo's 4K NV12 frame: rows of 3840 bytes, 2160 luma rows and 1080 chroma.
    let (luma, chroma) = (room(3840, 3840, 2160), room(3840, 3840, 1080));
    assert_eq!(
        crop_extent((3840, 2160), 1, luma, chroma),
        Some((3840, 2160))
    );
    // P010: two bytes a sample.
    let (luma, chroma) = (room(7680, 7680, 2160), room(7680, 7680, 1080));
    assert_eq!(
        crop_extent((3840, 2160), 2, luma, chroma),
        Some((3840, 2160))
    );
    // An odd crop, 641x361, in planes of its size: its chroma's last pair and row taken whole.
    let (luma, chroma) = (room(641, 704, 361), room(642, 704, 181));
    assert_eq!(crop_extent((641, 361), 1, luma, chroma), Some((641, 361)));
    // Planes smaller than the picture take its top left, as much as fits.
    let (luma, chroma) = (room(1280, 1280, 720), room(1280, 1280, 360));
    assert_eq!(
        crop_extent((1920, 1080), 1, luma, chroma),
        Some((1280, 720))
    );
    // A chroma plane narrower than the luma's bounds the width in whole pairs.
    let (luma, chroma) = (room(1920, 1920, 1080), room(1001, 1024, 540));
    assert_eq!(
        crop_extent((1920, 1080), 1, luma, chroma),
        Some((1000, 1080))
    );
    assert_eq!(
        crop_extent((1920, 1080), 1, room(0, 64, 1080), chroma),
        None
    );
    assert_eq!(
        crop_extent((1920, 1080), 1, luma, room(1920, 1920, 0)),
        None
    );
}

/// The conformance suite's streams, through the arm's decoders, each picture landed by the GPU's
/// copy into planes laid out as CoreVideo lays a frame's (32 bytes into their pages), and by the
/// CPU's copy into the same layout: every byte of the two must be equal, the bytes before, between
/// and past the planes' rows included. The planes' rows are the frame's bytes for one picture and
/// 64 bytes past a multiple of 64 for the next, so rows with bytes past them are copied too. The
/// pages are host memory a no-copy buffer lies over, as a surface's in the guest's RAM do.
///
/// The runtime's default device must be of the GPU `RVG_VAAPI_NODE` names, where it is set.
/// `RVG_VA_STREAMS` names the directories, `:`-separated, whose Annex B streams (`*.264`,
/// `*.265`, but `*.part.*`) are decoded: the conformance scripts' `build/h264-check` and
/// `build/hevc-check`. A stream the decoder refuses (as the suite has it refuse some) is decoded up
/// to its refusal; a leading picture random access skips is skipped.
#[test]
#[ignore = "needs a VA-API GPU and the conformance suite's streams: run on the VA-API host"]
fn a_picture_copied_by_the_gpu_is_the_cpus_copy_byte_for_byte() {
    use reims_vgpu_video::codec::{h264, hevc};

    let dirs = std::env::var("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    let mut streams: Vec<std::path::PathBuf> = (dirs.split(':'))
        .flat_map(|dir| std::fs::read_dir(dir).expect("a streams directory"))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path.to_string_lossy();
            (name.ends_with(".264") || name.ends_with(".265")) && !name.contains(".part.")
        })
        .collect();
    streams.sort();
    assert!(!streams.is_empty(), "no streams in {dirs}");
    let platform = Arc::new(VulkanPlatform::new());
    crate::entry::run(|_| {
        let device = VulkanPlatform::system_default_device().expect("a Vulkan device");
        let node = reims_vgpu_vulkan::host_gpus::render_node_of(&device.raw).expect("a node");
        if let Ok(asked) = std::env::var("RVG_VAAPI_NODE") {
            let canonical = |p: &std::path::Path| std::fs::canonicalize(p).expect("a node");
            assert_eq!(
                canonical(&node),
                canonical(std::path::Path::new(&asked)),
                "the runtime device is not of the GPU RVG_VAAPI_NODE names"
            );
        }
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(&node, runtime).expect("a VA display on the device's GPU");
        let copies = gpu_copy::RuntimeCopies::new(&device);
        let mut copier = copies.copier().expect("a GPU copier on the device");
        let (mut pictures, mut refused) = (0usize, Vec::new());
        for stream in &streams {
            let file = std::fs::read(stream).expect("a stream");
            let (units, mut decoder) = if stream.to_string_lossy().ends_with(".265") {
                let decoder = HevcDecoder::new_annex_b(&display);
                (hevc::nal::access_units(&file), Codec::Hevc(decoder))
            } else {
                let decoder = H264Decoder::new_annex_b(&display);
                (h264::nal::access_units(&file), Codec::H264(decoder))
            };
            for (id, unit) in units.iter().enumerate() {
                let id = u64::try_from(id).expect("an id");
                let decoded = match &mut decoder {
                    Codec::H264(d) => d.decode(unit, id),
                    Codec::Hevc(d) => d.decode(unit, id),
                };
                match decoded {
                    Ok(frames) => {
                        for frame in &frames {
                            land_both_ways(&mut *copier, &device, frame, pictures, stream);
                            pictures += 1;
                        }
                    }
                    // A leading picture random access skips: no picture, and the stream goes on.
                    Err(reims_vgpu_video::vaapi::Error::HevcBitstream(
                        hevc::Error::SkippedLeadingPicture,
                    )) => {}
                    Err(error) => {
                        refused.push(format!("{}: {error}", stream.display()));
                        break;
                    }
                }
            }
            let flushed = match &mut decoder {
                Codec::H264(d) => d.flush(),
                Codec::Hevc(d) => d.flush(),
            };
            for frame in &flushed.unwrap_or_default() {
                land_both_ways(&mut *copier, &device, frame, pictures, stream);
                pictures += 1;
            }
        }
        eprintln!(
            "{pictures} pictures of {} streams copied by the GPU as by the CPU; refused: {refused:?}",
            streams.len()
        );
        assert!(pictures > 0);
    });
}

/// A copy after the runtime made its device's `VkDevice` again lands as the CPU's copy, byte for
/// byte: the copier made before the recovery makes its objects again on the new `VkDevice`
/// (`reims_vgpu_vulkan::os::PlaneCopier`), and copies into buffers made there. Six pictures of
/// the first H.264 stream in `RVG_VA_STREAMS`'s first directory, three each side of the recovery,
/// each landed as [`land_both_ways`] lands it.
#[test]
#[ignore = "needs a VA-API GPU and the conformance suite's streams: run on the VA-API host"]
fn a_copy_after_the_devices_recovery_is_the_cpus_copy_byte_for_byte() {
    use reims_vgpu_video::codec::h264;

    const EACH_SIDE: usize = 3;
    let dirs = std::env::var("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    let dir = dirs.split(':').next().expect("a streams directory");
    let mut streams: Vec<std::path::PathBuf> = (std::fs::read_dir(dir).expect("a directory"))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path.to_string_lossy();
            name.ends_with(".264") && !name.contains(".part.")
        })
        .collect();
    streams.sort();
    let platform = Arc::new(VulkanPlatform::new());
    crate::entry::run(|_| {
        let device = VulkanPlatform::system_default_device().expect("a Vulkan device");
        let node = reims_vgpu_vulkan::host_gpus::render_node_of(&device.raw).expect("a node");
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(&node, runtime).expect("a VA display on the device's GPU");
        let copies = gpu_copy::RuntimeCopies::new(&device);
        let mut copier = copies.copier().expect("a GPU copier on the device");
        let mut n = 0;
        'streams: for stream in &streams {
            let file = std::fs::read(stream).expect("a stream");
            let mut decoder = H264Decoder::new_annex_b(&display);
            for (id, unit) in h264::nal::access_units(&file).iter().enumerate() {
                let Ok(frames) = decoder.decode(unit, u64::try_from(id).expect("an id")) else {
                    continue 'streams;
                };
                for frame in &frames {
                    if n == EACH_SIDE {
                        device
                            .raw
                            .recover_for_test()
                            .expect("the device's recovery");
                    }
                    land_both_ways(&mut *copier, &device, frame, n, stream);
                    n += 1;
                    if n == 2 * EACH_SIDE {
                        break 'streams;
                    }
                }
            }
        }
        assert_eq!(n, 2 * EACH_SIDE, "pictures in {dir}");
        eprintln!(
            "{} pictures copied by the GPU as by the CPU, {EACH_SIDE} after the device's recovery",
            2 * EACH_SIDE
        );
    });
}

/// A picture the driver decodes straight into a buffer the platform made as a dma-buf device
/// cache (`new_dma_buf_buffer_of_length_padding_unobserved`), its memory exported and imported as
/// the decode's target as the arm imports a cache's ([`CacheTargets`]), reads through textures
/// over the cache's planes in a kernel, once the platform was told what the driver wrote
/// (`Buffer::external_write_done`), as the CPU's copy of the same picture decoded into a surface
/// of the decoder's own, texel for texel of the crop: the first [`PICTURES`] pictures of each
/// H.264 stream in `RVG_VA_STREAMS`'s first directory, their references in such caches too. Each
/// texture is read once before the decode writes its cache, as a guest's texture over a pool's
/// surface is used before a decode lands in it, so a shadow the runtime keeps of a texture over
/// the cache's bytes was copied in from the cache's zeros, and only a copy in again after the
/// driver's write reads the picture. The planes lie as CoreVideo lays a frame's in the guest's
/// caches, 32 bytes into their pages, the first on 256 bytes of the cache's memory, so the chroma
/// plane lies past a gap (`docs/video/bugs.md`, U28). A stream the decoder refuses is left at its
/// refusal. Where the driver's check refuses H.264 into NV12 targets the arm decodes into no
/// cache, and the test says so and ends.
#[test]
#[ignore = "needs a VA-API GPU and the conformance suite's streams: run on the VA-API host"]
fn a_picture_decoded_into_a_dma_buf_cache_reads_through_its_textures_as_the_cpus_copy() {
    use reims_vgpu_video::codec::h264;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;

    /// The pictures of each stream decoded into caches.
    const PICTURES: usize = 6;
    let dirs = std::env::var("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    let dir = dirs.split(':').next().expect("a streams directory");
    let mut streams: Vec<std::path::PathBuf> = (std::fs::read_dir(dir).expect("a directory"))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path.to_string_lossy();
            name.ends_with(".264") && !name.contains(".part.")
        })
        .collect();
    streams.sort();
    let platform = Arc::new(VulkanPlatform::new());
    crate::entry::run(|_| {
        let device = VulkanPlatform::system_default_device().expect("a Vulkan device");
        let node = reims_vgpu_vulkan::host_gpus::render_node_of(&device.raw).expect("a node");
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(&node, runtime).expect("a VA display on the device's GPU");
        let checks = TargetChecks::of(&display, None);
        if let Err(why) = checks.passed(TargetCodec::H264, TargetFormat::Nv12) {
            eprintln!("the driver's check refuses H.264 into NV12 targets: {why}");
            return;
        }
        // The checks the device starts make textures of their own: they end before the runtime's
        // counts of shadows are read.
        let wait = std::time::Duration::from_secs(60);
        let _ = device.raw.settle_in_place_checks(wait);
        let _ = device.raw.settle_queue_switch_check(wait);
        let _ = device.raw.settle_submit_checks(wait);
        let reader = Arc::new(PlaneReader::new(&device));
        let (mut pictures, mut refused) = (0usize, Vec::new());
        // The planes' textures over the caches' memory, and those that are shadows.
        let (mut in_place, mut shadows) = (0usize, 0usize);
        'streams: for stream in &streams {
            let file = std::fs::read(stream).expect("a stream");
            let mut ours = H264Decoder::new_annex_b(&display);
            let mut theirs = H264Decoder::new_annex_b(&display);
            let mut landed = 0;
            for (id, unit) in h264::nal::access_units(&file).iter().enumerate() {
                let id = u64::try_from(id).expect("an id");
                let made = Arc::new(Mutex::new(None));
                let target_for = cache_target(&platform, &reader, &display, Arc::clone(&made));
                let (decoded, reference) = match (
                    ours.decode_into(unit, id, Some(target_for)),
                    theirs.decode(unit, id),
                ) {
                    (Ok(decoded), Ok(reference)) => (decoded, reference),
                    (decoded, reference) => {
                        let error = decoded.err().or(reference.err()).expect("a refusal");
                        refused.push(format!("{}: {error}", stream.display()));
                        continue 'streams;
                    }
                };
                assert_eq!(decoded.len(), reference.len(), "{}", stream.display());
                for (frame, reference) in decoded.iter().zip(&reference) {
                    let cache = made.lock().take();
                    let Some(cache) = cache.filter(|_| frame.is_in_target()) else {
                        panic!(
                            "{} picture {id}: not decoded into a cache",
                            stream.display()
                        );
                    };
                    frame.sync().expect("the decode");
                    cache.buffer.external_write_done().expect("the acquire");
                    compare_cache(&reader, &cache, reference, stream, id);
                    let shadowed = cache.planes.iter().filter(|(_, _, shadowed)| *shadowed);
                    let shadowed = shadowed.count();
                    (shadows, in_place) = (shadows + shadowed, in_place + 2 - shadowed);
                    pictures += 1;
                    landed += 1;
                }
                if landed == PICTURES {
                    break;
                }
            }
        }
        eprintln!(
            "{pictures} pictures of {} streams decoded into dma-buf caches read through textures \
             as by the CPU, the planes' textures {in_place} over the caches' memory and {shadows} \
             shadows, as over caches made as any other; refused: {refused:?}",
            streams.len()
        );
        assert!(pictures > 0);
    });
}

/// A cache [`cache_target`] made: its buffer, its planes' rows' pitch, and a texture over each
/// plane, luma's then chroma's, of the coded picture's texels, with its width and height and
/// whether the runtime made it a shadow of the cache's bytes.
struct MadeCache {
    buffer: VulkanBuffer,
    pitch: u64,
    planes: [(VulkanTexture, (u64, u64), bool); 2],
}

/// What gives a decode its picture's target in a new dma-buf cache of `reader`'s device laid out
/// for the picture's shape, as
/// [`a_picture_decoded_into_a_dma_buf_cache_reads_through_its_textures_as_the_cpus_copy`] lays
/// it, with a texture over each plane, read once by `reader` before the decode, the cache kept in
/// `made`; no target where the platform makes or exports none.
fn cache_target(
    platform: &Arc<VulkanPlatform>,
    reader: &Arc<PlaneReader>,
    display: &Display,
    made: Arc<Mutex<Option<MadeCache>>>,
) -> TargetFor {
    use std::os::fd::AsFd as _;

    use reims_vgpu_vulkan::core::platform::Handle as _;
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::platform::values::TextureDesc;
    use reims_vgpu_vulkan::core::raise::OnFault;

    const PAGE: u64 = 4096;
    /// `MTLPixelFormatR8Unorm`, `RG8Unorm`, `R16Unorm` and `RG16Unorm`.
    const R8: u64 = 10;
    const RG8: u64 = 30;
    const R16: u64 = 20;
    const RG16: u64 = 60;
    let (platform, reader, display) = (
        Arc::clone(platform),
        Arc::clone(reader),
        display.clone_ref(),
    );
    Box::new(move |shape: TargetShape| {
        let (width, height) = shape.coded;
        let sample = shape.format.bytes_per_sample();
        let row = u64::from(width) * sample;
        let pitch = row.next_multiple_of(TARGET_ALIGNMENT);
        let luma_at = 0x20;
        let chroma_at = (luma_at + pitch * u64::from(height)).next_multiple_of(PAGE) + 0x20;
        let len = (chroma_at + pitch * u64::from(height.div_ceil(2))).next_multiple_of(PAGE);
        let buffer = (reader.device)
            .new_dma_buf_buffer_of_length_padding_unobserved(
                OnFault::Return,
                len,
                STORAGE_PRIVATE,
                Some(luma_at),
            )
            .expect("no raise")
            .expect("a dma-buf cache");
        let (formats, extents) = (
            if sample == 2 { [R16, RG16] } else { [R8, RG8] },
            [
                (u64::from(width), u64::from(height)),
                (u64::from(width.div_ceil(2)), u64::from(height.div_ceil(2))),
            ],
        );
        // A twin of the cache made as any other laid-out cache, its memory exported nowhere: a
        // plane's texture over it takes the route the runtime is to take for the cache's, a shadow
        // of the bytes or an image over the memory, which the runtime announces in its count of
        // shadows, as an export changes no texture's route.
        let twin = (reader.device)
            .new_buffer_of_length_padding_unobserved(
                OnFault::Return,
                len,
                STORAGE_PRIVATE,
                Some(luma_at),
            )
            .expect("no raise")
            .expect("a cache made as any other");
        let shadows = || reader.device.raw.work_counts().shadowed_textures;
        let route = |shadowed: bool| ["over its memory", "a shadow"][usize::from(shadowed)];
        let planes = [0, 1].map(|n| {
            let descriptor = platform.texture_descriptor(&TextureDesc {
                pixel_format: formats[n],
                width: extents[n].0,
                height: extents[n].1,
                resource_options: Some(STORAGE_PRIVATE),
                ..TextureDesc::default()
            });
            let at = [luma_at, chroma_at][n];
            let over = |cache: &VulkanBuffer| {
                let before = shadows();
                let texture = (cache.new_texture(OnFault::Return, &descriptor, at, pitch))
                    .expect("no raise")
                    .expect("a texture over the cache's plane");
                (texture, shadows() > before)
            };
            let (_, twin_shadowed) = over(&twin);
            let (texture, shadowed) = over(&buffer);
            assert_eq!(
                shadowed,
                twin_shadowed,
                "plane {n} of a {width}x{height} picture: its texture is {} over the dma-buf \
                 cache, {} over a cache made as any other",
                route(shadowed),
                route(twin_shadowed)
            );
            // Read before the decode writes the cache: what a shadow of it holds is copied in.
            let _ = reader.read(&texture, extents[n]);
            (texture, extents[n], shadowed)
        });
        let exported = buffer.export_dma_buf().expect("the cache's dma-buf");
        let laid = [luma_at, chroma_at].map(|at| TargetPlane {
            offset: exported.offset + at,
            bytes_per_row: pitch,
        });
        let target = Target::import(
            &display,
            exported.fd.as_fd(),
            exported.allocation_size,
            shape,
            laid,
            Box::new(buffer.clone_ref()),
        )
        .expect("the import of the cache's memory");
        *made.lock() = Some(MadeCache {
            buffer,
            pitch,
            planes,
        });
        Some(target)
    })
}

/// What reads a texture's texels in a kernel, as a guest's shader reads a texture over a cache's
/// plane: the Vulkan crate's `read_texels_float` (its `ycbcr_probe.metal`), on a queue of the
/// device's own.
struct PlaneReader {
    device: VulkanDevice,
    state: <VulkanPlatform as Platform>::ComputePipeline,
    queue: <VulkanPlatform as Platform>::CommandQueue,
}

/// The Vulkan crate's metallib of the readers of a texture (its `ycbcr_probe.metal`), which a
/// host without Metal takes as a Mac does.
const READERS: &[u8] =
    include_bytes!("../../../../reims-vgpu-vulkan/metallibs/ycbcr_probe.metallib");

impl PlaneReader {
    /// The reader on `device`.
    fn new(device: &VulkanDevice) -> Self {
        use reims_vgpu_vulkan::core::platform::Handle as _;
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::resource::Library as _;
        use reims_vgpu_vulkan::core::raise::OnFault;

        let library = (device.new_library(READERS))
            .expect("no raise")
            .expect("the readers' library")
            .expect("a library");
        let function = (library.new_function(OnFault::Return, "read_texels_float"))
            .expect("no raise")
            .expect("the reader");
        let state = (device.new_compute_pipeline_state_for_function(Some(&function)))
            .expect("the reader's pipeline")
            .expect("a pipeline");
        Self {
            device: device.clone_ref(),
            state,
            queue: device.new_command_queue().expect("a queue"),
        }
    }

    /// What `texture` reads of its `width` by `height` texels, each its four components as
    /// floats, in rows: one dispatch in a command buffer of its own, waited for.
    fn read(&self, texture: &VulkanTexture, (width, height): (u64, u64)) -> Vec<[f32; 4]> {
        use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, MtlSize, NsRange};
        use reims_vgpu_vulkan::core::raise::OnFault;

        let length = width * height * 16;
        let out = (self.device.new_buffer_of_length(OnFault::Return, length, 0))
            .expect("no raise")
            .expect("a buffer for the texels");
        let cb = (self.queue.command_buffer(OnFault::Return))
            .expect("no raise")
            .expect("a command buffer");
        let encoder = (cb.compute_encoder(OnFault::Return))
            .expect("no raise")
            .expect("a compute encoder");
        let buffers = [Some(&out)];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&self.state),
            },
            ComputeCmd::SetTexture {
                texture: Some(texture),
                index: 0,
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(width, height, 1),
                threads_per_threadgroup: MtlSize::new(8, 8, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            encoder.encode(OnFault::Return, cmd).expect("the read");
        }
        cb.commit(OnFault::Return).expect("the commit");
        cb.wait_until_completed(OnFault::Return).expect("the wait");
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED, "the read");
        // SAFETY: the Shared buffer's `length` bytes, which the kernel finished writing before
        // its command buffer completed, and which nothing writes meanwhile.
        let bytes =
            unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), usize_of(length)) };
        let (texels, _) = bytes.as_chunks::<16>();
        (texels.iter())
            .map(|t| {
                std::array::from_fn(|i| f32::from_le_bytes(t[i * 4..][..4].try_into().unwrap()))
            })
            .collect()
    }
}

/// Panics where a byte of `reference`'s crop, copied by the CPU, differs from what the cache's
/// textures read in `reader`'s kernel: the cache holds the coded picture, the crop at its origin,
/// each texel's components its samples as unorms of their bits.
fn compare_cache(
    reader: &PlaneReader,
    cache: &MadeCache,
    reference: &Frame,
    stream: &std::path::Path,
    id: u64,
) {
    let MadeCache { pitch, planes, .. } = cache;
    let sample: u64 = if reference.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (u64::from(reference.width()), u64::from(reference.height()));
    let (luma_bytes, chroma_bytes) = (width * sample, width.div_ceil(2) * 2 * sample);
    let chroma_rows = height.div_ceil(2);
    let (luma_len, chroma_len) = (pitch * height, pitch * chroma_rows);
    let mut cpu = vec![0; usize_of(luma_len + chroma_len)];
    let (y, uv) = cpu.split_at_mut(usize_of(luma_len));
    (reference.read_back().expect("the read back"))
        .copy_into(
            plane_rows(y, *pitch, height, luma_bytes),
            plane_rows(uv, *pitch, chroma_rows, chroma_bytes),
        )
        .expect("the CPU's copy");
    let (x, top) = reference.crop_origin();
    let (x, top) = (u64::from(x), u64::from(top));
    // Each plane: its name, its components a texel, its crop's first texel, the CPU's copy's
    // first byte, rows and bytes a row.
    let compared = [
        ("luma", 1, (x, top), 0, height, luma_bytes),
        (
            "chroma",
            2,
            (x / 2, top / 2),
            luma_len,
            chroma_rows,
            chroma_bytes,
        ),
    ];
    let most = if sample == 2 { 65535.0 } else { 255.0 };
    for ((plane, components, (left, first_row), from, rows, row_bytes), (texture, extent, _)) in
        compared.into_iter().zip(planes)
    {
        let texels = reader.read(texture, *extent);
        for r in 0..rows {
            let row = &texels[usize_of((first_row + r) * extent.0 + left)..];
            let ours: Vec<u8> = (row.iter())
                .flat_map(|texel| &texel[..components])
                .flat_map(|&v| {
                    let bits = (v * most).round() as u16;
                    let bytes = bits.to_le_bytes();
                    bytes[..usize_of(sample)].to_vec()
                })
                .take(usize_of(row_bytes))
                .collect();
            let theirs = &cpu[usize_of(from + r * pitch)..][..usize_of(row_bytes)];
            if let Some(first) = (0..theirs.len()).find(|&i| ours.get(i) != Some(&theirs[i])) {
                panic!(
                    "{} picture {id} decoded into a cache, {width}x{height} at {sample}-byte \
                     samples, rows of {pitch}: {plane} row {r} byte {first} reads {:#x?} \
                     through its texture and is {:#x} by the CPU's copy",
                    stream.display(),
                    ours.get(first),
                    theirs[first]
                );
            }
        }
    }
}

/// Lands `frame`, the `n`th picture of the run, by `copier` into planes laid out as CoreVideo
/// lays a frame's, in host pages a buffer of `device` lies over, or, every third picture, in a
/// buffer made as a device cache is ([`land_in_a_cache`]), and by the CPU's copy into the same
/// layout in memory of its own; panics where a byte of the two differs.
fn land_both_ways(
    copier: &mut dyn GpuCopier<VulkanPlatform>,
    device: &VulkanDevice,
    frame: &Frame,
    n: usize,
    stream: &std::path::Path,
) {
    use reims_vgpu_vulkan::core::platform::HostRange;
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::os::vm::HostVm as _;
    use reims_vgpu_vulkan::core::raise::OnFault;
    use reims_vgpu_vulkan::remap::VulkanVm;

    let vm = VulkanVm::new();
    let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
    let luma_bytes = width * sample;
    let chroma_bytes = width.div_ceil(2) * 2 * sample;
    let pitch = if n % 2 == 0 {
        luma_bytes.max(chroma_bytes)
    } else {
        luma_bytes.max(chroma_bytes).next_multiple_of(64) + 64
    };
    let chroma_rows = height.div_ceil(2);
    let (luma_at, chroma_at) = (0x20, 0x20 + pitch * height);
    let len = (chroma_at + pitch * chroma_rows).next_multiple_of(vm.page_size());
    let cached = n % 3 == 2;
    if cached {
        land_in_a_cache(
            copier,
            device,
            frame,
            n,
            stream,
            (luma_at, chroma_at, pitch, len),
        );
        return;
    }
    let at = vm.allocate_anywhere(len).expect("host pages");
    // SAFETY: the pages stay mapped until the runtime lets go of them, when its deallocator
    // unmaps them, and nothing else reads or writes them meanwhile but the test's reads below.
    let bytes = unsafe { HostRange::new(at, len) };
    // The runtime runs the deallocator at the buffer's release, Metal's moment, and frees its
    // import of the pages after it, on a handler thread where a command buffer held the release.
    // The platform's VM leaves pages an import holds mapped until that free
    // (`reims_vgpu_vulkan::remap`, "Unmaps an import waits for"); the OS's unmap would take them
    // from under the import, and RADV, which hands every pointer import to each submission,
    // would reject every submission of the device until the free (`-EFAULT`).
    let unmap: Box<dyn FnOnce() + Send> = Box::new(move || {
        let _ = VulkanVm::new().deallocate(at, len);
    });
    let buffer = (device.new_buffer_over_host_memory(OnFault::Return, bytes, 0, Some(unmap)))
        .expect("no raise")
        .expect("a buffer over the pages");
    // SAFETY: the pages' `len` bytes, which no GPU work reaches yet.
    unsafe { std::ptr::write_bytes(at.as_ptr::<u8>(), UNTOUCHED, usize_of(len)) };
    let plane = |offset: u64, row_bytes: u64, w: u64, h: u64| OutputPlane {
        buffer: buffer.clone_ref(),
        offset,
        bytes_per_row: pitch,
        row_bytes,
        width: w,
        height: h,
        rows_held: h,
        cached: false,
        host: Some(at.wrapping_add(offset)),
        pages: None,
        cache: None,
    };
    let luma = plane(luma_at, luma_bytes, width, height);
    let chroma = plane(chroma_at, chroma_bytes, width.div_ceil(2), chroma_rows);
    match copier.copy(frame, &luma, &chroma).expect("the GPU's copy") {
        DeviceWrite::Written => {}
        DeviceWrite::Declined(why) => panic!("{}: declined: {why}", stream.display()),
    }
    let mut cpu = vec![UNTOUCHED; usize_of(len)];
    let (y, uv) = cpu.split_at_mut(usize_of(chroma_at));
    (frame.read_back().expect("the read back"))
        .copy_into(
            plane_rows(&mut y[usize_of(luma_at)..], pitch, height, luma_bytes),
            plane_rows(uv, pitch, chroma_rows, chroma_bytes),
        )
        .expect("the CPU's copy");
    // SAFETY: the pages' `len` bytes, which the GPU's copy finished writing before it returned,
    // and which nothing writes meanwhile.
    let gpu = unsafe { std::slice::from_raw_parts(at.as_ptr::<u8>(), usize_of(len)) };
    if let Some(first) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
        panic!(
            "{} picture {n}, {width}x{height} at {sample}-byte samples, rows of {pitch}: byte \
             {first} is {:#x} by the GPU's copy and {:#x} by the CPU's (luma at {luma_at}, chroma \
             at {chroma_at})",
            stream.display(),
            gpu[first],
            cpu[first]
        );
    }
    drop((luma, chroma, buffer));
}

/// [`land_both_ways`]'s landing of `frame` in a buffer of `device`'s own memory, made as a device
/// cache is: Private, the bytes between its rows declared unobserved and its first plane's byte
/// on 256 bytes of its memory (`surface_cache::aligned_byte`), its planes laid out by `(luma_at, chroma_at, pitch)` in its
/// `len` bytes, read back by a blit into a Shared buffer. As a device cache is filled from its
/// surface's pages as it is made, before a picture lands in it, the runtime fills the buffer with
/// [`UNTOUCHED`] first; the CPU's copy starts from those bytes.
fn land_in_a_cache(
    copier: &mut dyn GpuCopier<VulkanPlatform>,
    device: &VulkanDevice,
    frame: &Frame,
    n: usize,
    stream: &std::path::Path,
    (luma_at, chroma_at, pitch, len): (u64, u64, u64, u64),
) {
    use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsRange};
    use reims_vgpu_vulkan::core::raise::OnFault;

    let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
    let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
    let (luma_bytes, chroma_bytes) = (width * sample, width.div_ceil(2) * 2 * sample);
    let chroma_rows = height.div_ceil(2);
    // `MTLResourceStorageModePrivate`, a device cache's storage.
    let private = 2 << 4;
    let laid_out = device.new_buffer_of_length_padding_unobserved(
        OnFault::Return,
        len,
        private,
        Some(luma_at),
    );
    let cache = laid_out.expect("no raise").expect("a cache");
    let queue = device.new_command_queue().expect("a queue");
    let blit_and_wait = |cmd: BlitCmd<'_, VulkanPlatform>, what: &str| {
        let cb = (queue.command_buffer(OnFault::Return))
            .expect("no raise")
            .expect("a command buffer");
        let blit = (cb.blit_encoder(OnFault::Return))
            .expect("no raise")
            .expect("a blit encoder");
        blit.encode(OnFault::Return, cmd).expect(what);
        blit.encode(OnFault::Return, BlitCmd::EndEncoding)
            .expect("its end");
        cb.commit(OnFault::Return).expect("the commit");
        cb.wait_until_completed(OnFault::Return).expect("the wait");
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED, "{what}");
    };
    let range = NsRange {
        location: 0,
        length: len,
    };
    blit_and_wait(
        BlitCmd::FillBuffer {
            buffer: &cache,
            range,
            value: UNTOUCHED,
        },
        "the cache's fill",
    );
    let plane = |offset: u64, row_bytes: u64, w: u64, h: u64| OutputPlane {
        buffer: cache.clone_ref(),
        offset,
        bytes_per_row: pitch,
        row_bytes,
        width: w,
        height: h,
        rows_held: h,
        cached: true,
        host: None,
        pages: None,
        cache: None,
    };
    let luma = plane(luma_at, luma_bytes, width, height);
    let chroma = plane(chroma_at, chroma_bytes, width.div_ceil(2), chroma_rows);
    match copier.copy(frame, &luma, &chroma).expect("the GPU's copy") {
        DeviceWrite::Written => {}
        DeviceWrite::Declined(why) => panic!("{}: declined into a cache: {why}", stream.display()),
    }
    let back = (device.new_buffer_of_length(OnFault::Return, len, 0))
        .expect("no raise")
        .expect("a buffer to read the cache back into");
    blit_and_wait(
        BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&cache),
            source_offset: 0,
            destination_buffer: Some(&back),
            destination_offset: 0,
            size: len,
        },
        "the copy back",
    );
    let mut cpu = vec![UNTOUCHED; usize_of(len)];
    let (y, uv) = cpu.split_at_mut(usize_of(chroma_at));
    (frame.read_back().expect("the read back"))
        .copy_into(
            plane_rows(&mut y[usize_of(luma_at)..], pitch, height, luma_bytes),
            plane_rows(uv, pitch, chroma_rows, chroma_bytes),
        )
        .expect("the CPU's copy");
    // SAFETY: the read-back buffer's `len` bytes, which the blit finished writing before its
    // command buffer completed, and which nothing writes meanwhile.
    let gpu = unsafe { std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), usize_of(len)) };
    if let Some(first) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
        panic!(
            "{} picture {n} into a cache, {width}x{height} at {sample}-byte samples, rows of \
             {pitch}: byte {first} is {:#x} by the GPU's copy and {:#x} by the CPU's (luma at \
             {luma_at}, chroma at {chroma_at})",
            stream.display(),
            gpu[first],
            cpu[first]
        );
    }
    drop((luma, chroma, cache, back));
}

/// What the GPU's copies leave as it was: every byte past the planes' rows.
const UNTOUCHED: u8 = 0xee;

fn usize_of(n: u64) -> usize {
    usize::try_from(n).expect("a size")
}

/// `rows` rows of `row_bytes` in `bytes`, `pitch` apart, for the CPU's copy.
fn plane_rows(bytes: &mut [u8], pitch: u64, rows: u64, row_bytes: u64) -> PlaneRows<'_> {
    PlaneRows {
        bytes,
        bytes_per_row: usize_of(pitch),
        rows: usize_of(rows),
        row_bytes: usize_of(row_bytes),
    }
}
