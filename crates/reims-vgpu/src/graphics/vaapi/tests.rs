use super::*;
use crate::graphics::video_choice::tested_gpu::Arm;
use reims_vgpu_video::vaapi::{ImportWrite, SurfaceCapabilities, TargetFormat};

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

/// References are kept apart (the rt platform's rules): a reference picture is given no cache to
/// be decoded into, and a picture no later one predicts from is. A session counts each picture by
/// its role, and the references it kept apart, in the line it logs at its end, and logs none where
/// it asked a cache for no picture.
#[test]
fn references_are_kept_apart_from_their_caches_where_the_rules_say_and_counted() {
    use PictureRole::{NonReference, Reference};

    let apart = RoleCounts::default();
    assert_eq!(apart.line(), None);
    assert!(!apart.count(Reference, CachedReferences::Apart));
    assert!(apart.count(NonReference, CachedReferences::Apart));
    assert!(!apart.count(Reference, CachedReferences::Apart));
    let line = apart.line().expect("a line");
    assert!(
        line.ends_with(
            "2 references (2 kept apart, decoded in the decoder's own surfaces), 1 non-references"
        ),
        "{line}"
    );
}

/// The device's caches are made dma-bufs for a driver whose check of decodes into targets passed a
/// codec and format, whose pictures are then decoded into them, and for one whose decodes passed
/// none where its import check passed a format, whose video processing then writes each picture
/// into its cache (iHD's, U30); for neither, none is, the decline naming both checks' answers
#[test]
fn caches_are_dma_bufs_where_either_of_the_drivers_checks_passed() {
    let refused = |why: &str| Err::<(), String>(why.to_owned());
    let checks = |hevc: Result<(), String>| {
        TargetChecks::of_answers(vec![
            (TargetCodec::H264, TargetFormat::Nv12, refused("Y-tiled")),
            (TargetCodec::Hevc, TargetFormat::Nv12, refused("Y-tiled")),
            (TargetCodec::Hevc, TargetFormat::P010, hevc),
        ])
    };
    let whole_rows = ImportWrite {
        alignment: 64,
        whole_rows: true,
    };
    let none = ImportWrites::of_answers(Err("no NV12".to_owned()), Err("no P010".to_owned()));
    let p010 = ImportWrites::of_answers(Err("no NV12".to_owned()), Ok(whole_rows));
    let nv12 = ImportWrites::of_answers(Ok(whole_rows), Err("no P010".to_owned()));
    assert_eq!(
        cache_writes(&checks(Ok(())), &none),
        Ok(CacheWrites::Decodes)
    );
    assert_eq!(
        cache_writes(&checks(Ok(())), &p010),
        Ok(CacheWrites::Decodes)
    );
    for writes in [&p010, &nv12] {
        assert_eq!(
            cache_writes(&checks(refused("Y-tiled")), writes),
            Ok(CacheWrites::VideoProcessing)
        );
    }
    let why = cache_writes(&checks(refused("Y-tiled")), &none).expect_err("no cache made so");
    assert!(
        why.starts_with("the driver's check: h264-nv12 not taken: Y-tiled;")
            && why.ends_with(
                "; and its video processing writes no import: NV12: no NV12; P010: no P010"
            ),
        "{why}"
    );
}

/// On the rt platform the VA-API arm answers what the driver decodes and no encoder (the platform
/// releases no encoder's sources to the driver yet), and writes into the device's caches where the
/// driver's check of decodes into targets passed a codec, or its import check a format: rt's
/// exported blocks, as dma-bufs, take the pictures no later one predicts from, decoded there, or
/// the pictures the driver's video processing writes there.
/// The rt device must be of a GPU the arm runs on (`video_choice::tested_gpu::check_rt`).
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn the_rt_platforms_va_api_arm_decodes_alone() {
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::VaApi);
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let video = open_on_rt(&device, runtime).expect("VA-API on the rt device's GPU");
        let caps = video.caps();
        assert!(!caps.is_empty());
        assert!(
            caps.iter().all(|c| c.direction == RGFX_VIDEO_DECODE),
            "{caps:?}"
        );
        assert!(video.guest_encoders().is_empty());
        assert!(video.guest_decodes());
        assert!(video.lands_in_device_caches());
        let gpu = device.host_gpu();
        let node = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci);
        let display = Display::open(
            node.expect("a render node"),
            crate::video::runtime::PlatformRuntime::of(&platform),
        )
        .expect("a VA display");
        let checks = TargetChecks::of(&display, None);
        let passed = [
            (TargetCodec::H264, TargetFormat::Nv12),
            (TargetCodec::Hevc, TargetFormat::Nv12),
            (TargetCodec::Hevc, TargetFormat::P010),
        ]
        .into_iter()
        .any(|(codec, format)| checks.passed(codec, format).is_ok());
        let writes = display.import_writes();
        let written = writes.nv12().is_ok() || writes.p010().is_ok();
        assert_eq!(
            video.writes_into_device_caches(),
            passed || written,
            "{}; NV12 {:?}, P010 {:?}",
            checks.summary(),
            writes.nv12(),
            writes.p010()
        );
    });
    crate::rt_test_end::drop_last(platform);
}

/// A picture the driver decoded into a surface of its own, written by its video processing into a
/// buffer the platform made as a dma-buf device cache
/// (`new_dma_buf_buffer_of_length_padding_unobserved`), through the cache's memory as the arm
/// hands it ([`CacheBytes`], [`in_place_layout`]), as a picture lands in its output's cache where
/// the driver's check refused to decode it there, is the CPU's copy of the same picture, read back by a blit once the platform was told what the
/// driver wrote (`Buffer::external_write_done`): each plane's rows of the crop byte for byte, and
/// every byte before the planes, past each row's even width and past the planes' even rows as the
/// runtime's fill left it. The first [`PICTURES`] pictures of each H.264 and HEVC stream in
/// `RVG_VA_STREAMS`'s directories, 8 and 10 bits; the planes' rows a multiple of 256 bytes, each
/// plane on 256 bytes of the cache's memory and the chroma plane past a gap, as the device lays a
/// cache out at the GPU's pitch. A stream the decoder refuses is left at its refusal; a picture
/// the driver's video processing declines is counted and named.
#[test]
#[ignore = "needs a VA-API GPU and the conformance suite's streams: run on the VA-API host"]
fn a_picture_written_into_a_dma_buf_cache_by_video_processing_is_the_cpus_copy() {
    use reims_vgpu_video::core::device_memory::MemoryLife;
    use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, NsRange};
    use reims_vgpu_vulkan::core::raise::OnFault;

    const PAGE: u64 = 4096;
    let streams = va_streams();
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::VaApi);
        let gpu = device.host_gpu();
        let node = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci);
        let node = node.expect("a render node");
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(&node, runtime).expect("a VA display on the device's GPU");
        let writes = GuestWrites::new(&display);
        let queue = device.new_command_queue().expect("a queue");
        let blit_and_wait = |cmd: BlitCmd<'_, reims_vgpu_vulkan::RtPlatform>, what: &str| {
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
        let (mut pictures, mut declined) = (0usize, Vec::new());
        let mut each = |stream: &std::path::Path, id: u64, frame: &Frame| -> bool {
            let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
            let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
            let (even_w, even_h) = (width.next_multiple_of(2), height.next_multiple_of(2));
            let (luma_bytes, chroma_bytes) = (width * sample, width.div_ceil(2) * 2 * sample);
            let pitch = (even_w * sample).next_multiple_of(TARGET_ALIGNMENT);
            let chroma_rows = height.div_ceil(2);
            let (luma_at, chroma_at) = (0, (pitch * even_h).next_multiple_of(PAGE));
            let len = (chroma_at + pitch * even_h / 2).next_multiple_of(PAGE);
            let cache = (device.new_dma_buf_buffer_of_length_padding_unobserved(
                OnFault::Return,
                len,
                STORAGE_PRIVATE,
                Some(luma_at),
            ))
            .expect("no raise")
            .expect("a dma-buf cache");
            blit_and_wait(
                BlitCmd::FillBuffer {
                    buffer: &cache,
                    range: NsRange {
                        location: 0,
                        length: len,
                    },
                    value: UNTOUCHED,
                },
                "the cache's fill",
            );
            let exported = cache.export_dma_buf().expect("the cache's dma-buf");
            let life = MemoryLife::new();
            let bytes = CacheBytes::<reims_vgpu_vulkan::RtPlatform> {
                memory: Arc::new(CacheMemory {
                    dma_buf: exported.fd,
                    size: exported.allocation_size,
                    offset: exported.offset,
                    cache: CacheOf::Buffer(cache.clone_ref()),
                    last: Mutex::new(None),
                }),
                identity: life.identity(),
            };
            let place = |offset: u64, row_bytes: u64, w: u64, h: u64| Place {
                offset,
                bytes_per_row: pitch,
                row_bytes,
                width: w,
                height: h,
                rows_held: h.next_multiple_of(2),
            };
            let layout = in_place_layout(
                place(luma_at, even_w * sample, width, height),
                place(chroma_at, even_w * sample, width.div_ceil(2), even_h / 2),
                frame.bit_depth(),
            )
            .expect("a layout");
            let target = GuestTarget::new(&layout, &bytes).expect("a target");
            match frame.write_to_guest(&target, &writes).expect("the write") {
                DeviceWrite::Written => {}
                DeviceWrite::Declined(why) => {
                    declined.push(format!("{} picture {id}: {why}", stream.display()));
                    return false;
                }
            }
            cache.external_write_done().expect("the acquire");
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
            // SAFETY: the read-back buffer's `len` bytes, which the blit finished writing
            // before its command buffer completed, and which nothing writes meanwhile.
            let gpu = unsafe {
                std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), usize_of(len))
            };
            // The bytes the run may write past the crop: each row's samples to the even
            // width, and the even height's last row.
            let (luma_even, chroma_even) = (even_w * sample, even_w * sample);
            let past_crop = |i: u64| {
                let in_plane = |at: u64, rows: u64, even_rows: u64, crop: u64, even: u64| {
                    let Some(from) = i.checked_sub(at) else {
                        return false;
                    };
                    let (row, column) = (from / pitch, from % pitch);
                    row < even_rows
                        && ((row < rows && column >= crop && column < even)
                            || (row >= rows && column < even))
                };
                in_plane(luma_at, height, even_h, luma_bytes, luma_even)
                    || in_plane(
                        chroma_at,
                        chroma_rows,
                        even_h / 2,
                        chroma_bytes,
                        chroma_even,
                    )
            };
            let first = (0..gpu.len()).find(|&i| gpu[i] != cpu[i] && !past_crop(i.to_u64()));
            if let Some(first) = first {
                panic!(
                    "{} picture {id} written into a cache, {width}x{height} at \
                     {sample}-byte samples, rows of {pitch}: byte {first} is {:#x} by the \
                     driver's video processing and {:#x} by the CPU's copy (luma at \
                     {luma_at}, chroma at {chroma_at})",
                    stream.display(),
                    gpu[first],
                    cpu[first]
                );
            }
            drop((bytes, life, cache, back));
            pictures += 1;
            true
        };
        let refused = each_picture(&display, &streams, PICTURES, &mut each);
        eprintln!(
            "{pictures} pictures of {} streams written into dma-buf caches as by the CPU; \
             declined: {declined:?}; refused: {refused:?}",
            streams.len()
        );
        assert!(pictures > 0, "no picture written: {declined:?}");
    });
    crate::rt_test_end::drop_last(platform);
}

/// A picture the driver decoded into a surface of its own, written by its video processing into a
/// surface the platform exported as a dma-buf, laid out as the device asks a dma-buf cache's
/// surface of the platform (`surface_cache/exported.rs`: each plane an image of `R8Unorm` and
/// `RG8Unorm`, or `R16Unorm` and `RG16Unorm`, the pitch and the chroma plane's start on 256 bytes,
/// each plane holding its coded rows below its own), through the surface's memory as the arm hands
/// it, as a picture lands in its output's cache's surface where the driver's check refused to
/// decode it there, is the CPU's copy of the same picture in each plane image's crop, read back by
/// a blit once the platform was told what the driver wrote (`SurfaceMemory::external_writes_done`).
/// The first [`PICTURES`] pictures of each H.264 and HEVC stream in `RVG_VA_STREAMS`'s
/// directories, 8 and 10 bits. The surface's luma plane lies at its memory's first byte and its
/// chroma plane whole rows after it, as a driver whose video processing writes whole rows alone
/// (iHD's, U26) takes them; a picture the driver's video processing declines is counted and
/// named, and a stream the decoder refuses is left at its refusal.
#[test]
#[ignore = "needs a VA-API GPU and the conformance suite's streams: run on the VA-API host"]
fn a_picture_written_into_a_dma_buf_caches_surface_by_video_processing_is_the_cpus_copy() {
    use reims_vgpu_video::core::device_memory::MemoryLife;
    use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
    use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
    use reims_vgpu_vulkan::core::platform::resource::{
        SurfaceDesc, SurfaceHandle, SurfacePlaneDesc,
    };
    use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, MtlOrigin, MtlSize};
    use reims_vgpu_vulkan::core::raise::OnFault;

    /// `MTLPixelFormatR8Unorm` and `RG8Unorm`, then `R16Unorm` and `RG16Unorm`: the plane images'
    /// formats, by the bytes of a sample.
    const FORMATS: [[u64; 2]; 2] = [[10, 30], [20, 60]];
    /// `ShaderRead | RenderTarget | PixelFormatView`.
    const USAGE: u64 = 0x1 | 0x4 | 0x10;
    /// The coded rows each plane holds: its rows rounded up to these, luma's and chroma's.
    const CODED_ROWS: [u64; 2] = [64, 32];
    let streams = va_streams();
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        crate::graphics::video_choice::tested_gpu::check_rt(&device, Arm::VaApi);
        let gpu = device.host_gpu();
        let node = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci);
        let node = node.expect("a render node");
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display = Display::open(&node, runtime).expect("a VA display on the device's GPU");
        let writes = GuestWrites::new(&display);
        let queue = device.new_command_queue().expect("a queue");
        let blit_and_wait = |cmd: BlitCmd<'_, reims_vgpu_vulkan::RtPlatform>, what: &str| {
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
        let (mut pictures, mut declined) = (0usize, Vec::new());
        let mut each = |stream: &std::path::Path, id: u64, frame: &Frame| -> bool {
            let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
            let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
            let (even_w, even_h) = (width.next_multiple_of(2), height.next_multiple_of(2));
            let sizes = [(even_w, even_h), (even_w / 2, even_h / 2)];
            let planes: [SurfacePlaneDesc; 2] = std::array::from_fn(|n| {
                let rows = sizes[n].1;
                SurfacePlaneDesc {
                    pixel_format: FORMATS[usize_of(sample - 1)][n],
                    width: sizes[n].0,
                    height: rows,
                    usage: USAGE,
                    rows_below: u32::try_from(rows.next_multiple_of(CODED_ROWS[n]) - rows)
                        .expect("rows"),
                }
            });
            let desc = SurfaceDesc {
                handle: SurfaceHandle::DmaBuf,
                pitch_multiple: TARGET_ALIGNMENT,
                start_multiple: TARGET_ALIGNMENT,
                planes: &planes,
            };
            let made = (device.new_exported_surface(OnFault::Return, &desc))
                .expect("no raise")
                .expect("a dma-buf surface");
            let laid = made.memory.layout().to_vec();
            let exported = made.memory.export().expect("the surface's dma-buf");
            let life = MemoryLife::new();
            let cache = CacheBytes::<reims_vgpu_vulkan::RtPlatform> {
                memory: Arc::new(CacheMemory {
                    dma_buf: exported.fd,
                    size: exported.allocation_size,
                    offset: exported.offset,
                    cache: CacheOf::Surface(Arc::clone(&made.memory)),
                    last: Mutex::new(None),
                }),
                identity: life.identity(),
            };
            // Each plane's rows of the image, the luma's samples and the chroma's pairs of them
            // to the even width.
            let row = even_w * sample;
            let place = |n: usize, w: u64, h: u64| Place {
                offset: laid[n].offset,
                bytes_per_row: laid[n].row_pitch,
                row_bytes: row,
                width: w,
                height: h,
                rows_held: sizes[n].1,
            };
            let layout = in_place_layout(
                place(0, width, height),
                place(1, width.div_ceil(2), height.div_ceil(2)),
                frame.bit_depth(),
            )
            .expect("a layout");
            let target = GuestTarget::new(&layout, &cache).expect("a target");
            match frame.write_to_guest(&target, &writes).expect("the write") {
                DeviceWrite::Written => {}
                DeviceWrite::Declined(why) => {
                    declined.push(format!("{} picture {id}: {why}", stream.display()));
                    return false;
                }
            }
            made.memory.external_writes_done().expect("the acquire");
            let rows = [height, height.div_ceil(2)];
            let crop = [width * sample, width.div_ceil(2) * 2 * sample];
            let mut cpu = [0, 1].map(|n| vec![0u8; usize_of(row * sizes[n].1)]);
            let [luma, chroma] = &mut cpu;
            (frame.read_back().expect("the read back"))
                .copy_into(
                    plane_rows(luma, row, rows[0], crop[0]),
                    plane_rows(chroma, row, rows[1], crop[1]),
                )
                .expect("the CPU's copy");
            for n in 0..2 {
                let len = row * sizes[n].1;
                let back = (device.new_buffer_of_length(OnFault::Return, len, 0))
                    .expect("no raise")
                    .expect("a buffer to read the plane back into");
                blit_and_wait(
                    BlitCmd::CopyTextureToBuffer {
                        source_texture: &made.planes[n],
                        source_slice: 0,
                        source_level: 0,
                        source_origin: MtlOrigin::new(0, 0, 0),
                        source_size: MtlSize::new(sizes[n].0, sizes[n].1, 1),
                        destination_buffer: &back,
                        destination_offset: 0,
                        destination_bytes_per_row: row,
                        destination_bytes_per_image: len,
                    },
                    "the plane's read back",
                );
                // SAFETY: the read-back buffer's `len` bytes, which the blit finished writing
                // before its command buffer completed, and which nothing writes meanwhile.
                let gpu = unsafe {
                    std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), usize_of(len))
                };
                for y in 0..rows[n] {
                    let at = usize_of(y * row);
                    let (by_gpu, by_cpu) = (
                        &gpu[at..at + usize_of(crop[n])],
                        &cpu[n][at..at + usize_of(crop[n])],
                    );
                    if let Some(x) = (0..by_gpu.len()).find(|&x| by_gpu[x] != by_cpu[x]) {
                        panic!(
                            "{} picture {id} written into a dma-buf cache's surface, \
                             {width}x{height} at {sample}-byte samples, planes at {laid:?}: \
                             plane {n} row {y} byte {x} is {:#x} by the driver's video \
                             processing and {:#x} by the CPU's copy",
                            stream.display(),
                            by_gpu[x],
                            by_cpu[x]
                        );
                    }
                }
            }
            drop((cache, life, made));
            pictures += 1;
            true
        };
        let refused = each_picture(&display, &streams, PICTURES, &mut each);
        eprintln!(
            "{pictures} pictures of {} streams written into dma-buf caches' surfaces as by the \
             CPU; declined: {declined:?}; refused: {refused:?}",
            streams.len()
        );
        assert!(pictures > 0, "no picture written: {declined:?}");
    });
    crate::rt_test_end::drop_last(platform);
}

/// The pictures of each stream the tests of writes into caches write.
const PICTURES: usize = 4;

/// The H.264 and HEVC streams in `RVG_VA_STREAMS`'s directories, the conformance suite's: their
/// Annex B forms but the parts a run left.
fn va_streams() -> Vec<std::path::PathBuf> {
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
    streams
}

/// Decodes each of `streams` on `display`, in decode order, and hands each frame to `each` with
/// its stream and its access unit's id, until `each` answered `true` (written) for `most` of the
/// stream's frames or the stream ends. A stream the decoder refuses is left at its refusal, which
/// is answered.
fn each_picture(
    display: &Display,
    streams: &[std::path::PathBuf],
    most: usize,
    each: &mut dyn FnMut(&std::path::Path, u64, &Frame) -> bool,
) -> Vec<String> {
    use reims_vgpu_video::codec::{h264, hevc};

    let mut refused = Vec::new();
    for stream in streams {
        let file = std::fs::read(stream).expect("a stream");
        let (units, mut decoder) = if stream.to_string_lossy().ends_with(".265") {
            let decoder = HevcDecoder::new_annex_b(display);
            (hevc::nal::access_units(&file), Codec::Hevc(decoder))
        } else {
            let decoder = H264Decoder::new_annex_b(display);
            (h264::nal::access_units(&file), Codec::H264(decoder))
        };
        let mut written = 0;
        'units: for (id, unit) in units.iter().enumerate() {
            let id = u64::try_from(id).expect("an id");
            let frames = match &mut decoder {
                Codec::H264(d) => d.decode(unit, id),
                Codec::Hevc(d) => d.decode(unit, id),
            };
            let frames = match frames {
                Ok(frames) => frames,
                Err(reims_vgpu_video::vaapi::Error::HevcBitstream(
                    hevc::Error::SkippedLeadingPicture,
                )) => continue,
                Err(error) => {
                    refused.push(format!("{}: {error}", stream.display()));
                    break;
                }
            };
            for frame in &frames {
                if each(stream, id, frame) {
                    written += 1;
                    if written == most {
                        break 'units;
                    }
                }
            }
        }
    }
    refused
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
