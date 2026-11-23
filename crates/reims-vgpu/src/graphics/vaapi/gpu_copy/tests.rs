use reims_vgpu_video::core::fourcc::FourCc;

use super::*;

/// A plane of `height` rows of `row_bytes`, `bytes_per_row` apart from `offset`, in a buffer of
/// `length` bytes.
fn plane(length: u64, offset: u64, bytes_per_row: u64, row_bytes: u64, height: u64) -> Source {
    Source {
        length,
        offset,
        bytes_per_row,
        row_bytes,
        height,
    }
}

/// CoreVideo's 1080p NV12 output, as the guest's video driver hands it (`docs/video/devices.md`,
/// "Output layouts"): rows of 1,920 bytes, the luma plane 32 bytes into its surface and the
/// chroma plane 1,088 rows after it.
fn core_video_1080p() -> [Source; 2] {
    let chroma_at = 32 + 1920 * 1088;
    let length = chroma_at + 1920 * 544;
    [
        plane(length, 32, 1920, 1920, 1080),
        plane(length, chroma_at, 1920, 1920, 540),
    ]
}

/// A picture's scratch holds its even width's bytes at a pitch on 256 bytes, its chroma plane
/// right after its even rows of luma; a 10-bit picture's samples are two bytes, and a picture of
/// another depth, or of no size, has none. The driver writes it as `in_place_layout` lays a
/// picture out.
#[test]
fn a_scratch_holds_the_pictures_even_size_at_a_pitch_on_256() {
    let nv12 = ScratchLayout::of(1920, 1080, 8).expect("a scratch");
    assert_eq!(
        nv12,
        ScratchLayout {
            width: 1920,
            height: 1080,
            sample: 1,
            pitch: 2048,
            chroma_at: 2048 * 1080,
            length: 2048 * 1080 + 2048 * 540,
        }
    );
    let odd = ScratchLayout::of(1279, 719, 8).expect("a scratch");
    assert_eq!((odd.width, odd.height, odd.pitch), (1280, 720, 1280));
    assert_eq!(odd.chroma_at, 1280 * 720);
    let p010 = ScratchLayout::of(1920, 1080, 10).expect("a scratch");
    assert_eq!((p010.sample, p010.pitch), (2, 3840));
    assert_eq!(p010.length, 3840 * 1080 + 3840 * 540);
    assert!(ScratchLayout::of(1920, 1080, 12).is_err());
    assert!(ScratchLayout::of(0, 1080, 8).is_err());
    let surface = nv12.surface(8).expect("a layout");
    assert_eq!(surface.pixel_format, FourCc::new(*b"420v"));
    assert_eq!((surface.width, surface.height), (1920, 1080));
    let planes: Vec<_> = (surface.planes.iter())
        .map(|p| (p.offset, p.bytes_per_row, p.width, p.height))
        .collect();
    assert_eq!(
        planes,
        [(0, 2048, 1920, 1080), (2048 * 1080, 2048, 960, 540)]
    );
    let surface = p010.surface(10).expect("a layout");
    assert_eq!(surface.pixel_format, FourCc::new(*b"x420"));
}

/// A 1080p picture is copied from its scratch into CoreVideo's planes: each plane's crop rows
/// whole, bound from the scratch's plane and the output's first byte, with no margin, at the
/// output's own pitch. So is a picture whose rows end inside a word, 1366 wide: each row's last
/// word holds its last 2 bytes.
#[test]
fn a_pictures_crop_is_copied_from_its_scratch_into_core_videos_planes() {
    let scratch = ScratchLayout::of(1920, 1080, 8).expect("a scratch");
    let expected = |bound, at, rows| PlaneFill {
        bound,
        at,
        plane: FillPlane {
            first: 0,
            source_pitch: 2048,
            row_bytes: 1920,
            rows,
            unit: 0,
            coded_words: 480,
            coded_rows: rows,
            pitch_words: 480,
            tiling: TILING_LINEAR,
        },
    };
    let fills = plane_copies(scratch, (1920, 1080), core_video_1080p()).expect("the copies");
    assert_eq!(
        fills,
        [
            expected(0, 32, 1080),
            expected(2048 * 1080, 32 + 1920 * 1088, 540),
        ]
    );
    assert_eq!(copied_kib(&fills), (1920 * 1080 + 1920 * 540) >> 10);
    // CoreVideo's 1366x768 NV12: rows of 1,376 bytes, the chroma plane 768 rows after the luma.
    let wxga = ScratchLayout::of(1366, 768, 8).expect("a scratch");
    let chroma_at = 32 + 1376 * 768;
    let length = chroma_at + 1376 * 384;
    let outputs = [
        plane(length, 32, 1376, 1366, 768),
        plane(length, chroma_at, 1376, 1366, 384),
    ];
    let expected = |bound, at, rows| PlaneFill {
        bound,
        at,
        plane: FillPlane {
            first: 0,
            source_pitch: 1536,
            row_bytes: 1366,
            rows,
            unit: 0,
            coded_words: 342,
            coded_rows: rows,
            pitch_words: 344,
            tiling: TILING_LINEAR,
        },
    };
    assert_eq!(
        plane_copies(wxga, (1366, 768), outputs).expect("the copies"),
        [expected(0, 32, 768), expected(1536 * 768, chroma_at, 384)]
    );
    let p010 = ScratchLayout::of(1280, 720, 10).expect("a scratch");
    let chroma_at = 32 + 2560 * 736;
    let length = chroma_at + 2560 * 368;
    let outputs = [
        plane(length, 32, 2560, 2560, 720),
        plane(length, chroma_at, 2560, 2560, 360),
    ];
    let fills = plane_copies(p010, (1280, 720), outputs).expect("the copies");
    assert_eq!(
        fills.map(|f| (
            f.bound,
            f.at,
            f.plane.source_pitch,
            f.plane.pitch_words,
            f.plane.unit
        )),
        [(0, 32, 2560, 640, 0), (2560 * 720, chroma_at, 2560, 640, 0)]
    );
}

/// A picture is copied only where the CPU's copy's crop fills its planes' rows, from a first
/// byte and at a pitch on words, each row's last word within the planes' buffer: otherwise the
/// copy declines, and the CPU's copy lands it. Rows that end inside a word are copied.
#[test]
fn a_crop_that_does_not_fill_its_planes_rows_on_words_is_declined() {
    let scratch = ScratchLayout::of(1920, 1080, 8).expect("a scratch");
    let [luma, chroma] = core_video_1080p();
    let declined = |outputs: [Source; 2], size: (u64, u64)| {
        plane_copies(scratch, size, outputs).expect_err("a decline")
    };
    // Planes that hold more rows than the crop, or wider rows.
    let taller = Source {
        height: 1088,
        ..luma
    };
    declined([taller, chroma], (1920, 1080));
    let wider = Source {
        row_bytes: 1984,
        bytes_per_row: 1984,
        ..luma
    };
    declined([wider, chroma], (1920, 1080));
    // A smaller picture than its planes.
    declined([luma, chroma], (1280, 720));
    // A first byte or a pitch off a word.
    declined([Source { offset: 34, ..luma }, chroma], (1920, 1080));
    let off_word = Source {
        bytes_per_row: 1922,
        ..chroma
    };
    declined([luma, off_word], (1920, 1080));
    // Rows that end inside a word are copied, each row's last word taking its last 2 bytes; but
    // not where the last row's last word ends past the planes' buffer, though the row does not.
    let odd = ScratchLayout::of(1918, 1080, 8).expect("a scratch");
    let narrow = [
        Source {
            row_bytes: 1918,
            ..luma
        },
        Source {
            row_bytes: 1918,
            ..chroma
        },
    ];
    let fills = plane_copies(odd, (1918, 1080), narrow).expect("the copies");
    assert_eq!(
        fills.map(|f| (f.plane.row_bytes, f.plane.coded_words, f.plane.unit)),
        [(1918, 480, 0), (1918, 480, 0)]
    );
    let [narrow_luma, narrow_chroma] = narrow;
    let ends_in_its_last_word = Source {
        length: narrow_chroma.offset + 1920 * 539 + 1918,
        ..narrow_chroma
    };
    let outputs = [narrow_luma, ends_in_its_last_word];
    plane_copies(odd, (1918, 1080), outputs).expect_err("a decline");
    // Rows past the planes' buffer.
    let short = [
        luma,
        Source {
            length: chroma.offset + 1920 * 539,
            ..chroma
        },
    ];
    declined(short, (1920, 1080));
}

/// `length` bytes, each a pattern of its index that no two nearby bytes share.
fn pattern(length: usize) -> Vec<u8> {
    (0..length)
        .map(|i| (i.wrapping_mul(131) ^ (i >> 8).wrapping_mul(29)) as u8)
        .collect()
}

/// On the host's GPU, the kernel copies each plane's crop rows from a picture's scratch into its
/// output's planes byte for byte, at the planes' own first bytes and pitches, CoreVideo's among
/// them (each plane 32 bytes off a multiple of 256, rows of 1,920 bytes), and writes nothing else
/// of the output's buffer: NV12 and P010, planes on words off 256 bytes, and NV12 rows that end
/// inside a word, 2 bytes past it (1366, 854 and 426 wide), whose last word keeps the output's
/// other 2 bytes as they were, each byte of the output before the copy its own.
#[test]
fn on_rt_the_kernel_copies_a_pictures_crop_from_its_scratch_and_nothing_else() {
    use reims_vgpu_vulkan::RtPlatform;

    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        let pipeline = Pipeline::<RtPlatform>::new(&device);
        let pipeline = pipeline.get().expect("the kernel's pipeline");
        let queue = device.new_command_queue().expect("a queue");
        let shared = |length: u64| {
            (device.new_buffer_of_length(OnFault::Return, length, 0))
                .expect("no raise")
                .expect("a buffer")
        };
        let arguments = shared(2 * ARGUMENTS_APART);
        // (size, bit depth, the luma plane's first byte, the output's pitch, the chroma plane's
        // first byte).
        let cases = [
            ((1920, 1080), 8, 32, 1920, 32 + 1920 * 1088),
            ((1280, 720), 10, 32, 2560, 32 + 2560 * 736),
            ((64, 48), 8, 4, 68, 4 + 68 * 48 + 8),
            ((640, 360), 8, 0, 640, 640 * 368),
            ((1366, 768), 8, 32, 1376, 32 + 1376 * 768),
            ((854, 480), 8, 4, 856, 4 + 856 * 480 + 8),
            ((426, 240), 8, 32, 432, 32 + 432 * 240),
            ((1366, 768), 10, 32, 2736, 32 + 2736 * 768),
        ];
        for ((width, height), depth, luma_at, pitch, chroma_at) in cases {
            let what = format!("{width}x{height} at {depth} bits into {luma_at}, {chroma_at}");
            let scratch = ScratchLayout::of(width, height, depth).expect("a scratch");
            let row_bytes = u64::from(width) * scratch.sample;
            let chroma_bytes = u64::from(width).div_ceil(2) * 2 * scratch.sample;
            let rows = u64::from(height);
            let length = chroma_at + pitch * rows.div_ceil(2) + 256;
            let outputs = [
                plane(length, luma_at, pitch, row_bytes, rows),
                plane(length, chroma_at, pitch, chroma_bytes, rows.div_ceil(2)),
            ];
            let size = (u64::from(width), rows);
            let fills = plane_copies(scratch, size, outputs).expect(&what);
            let bytes = pattern(scratch.length as usize);
            let before: Vec<u8> = pattern(length as usize + 7)[7..]
                .iter()
                .map(|b| !b)
                .collect();
            let source = shared(scratch.length);
            let output = shared(length);
            // SAFETY: the buffers' host bytes, which no GPU work reaches yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    source.contents().as_ptr::<u8>(),
                    bytes.len(),
                );
                std::ptr::copy_nonoverlapping(
                    before.as_ptr(),
                    output.contents().as_ptr::<u8>(),
                    before.len(),
                );
            }
            let cb = (queue.command_buffer(OnFault::Return))
                .expect("no raise")
                .expect("a command buffer");
            let planes = Planes::<RtPlatform> {
                pipeline: &pipeline,
                arguments: &arguments,
                destinations: [&output, &output],
                sources: [&source, &source],
            };
            assert_eq!(
                encode_planes::<RtPlatform>(&cb, &planes, &fills),
                None,
                "{what}"
            );
            cb.commit(OnFault::Return).expect("the commit");
            cb.wait_until_completed(OnFault::Return).expect("the wait");
            assert_eq!(cb.status(), CommandBufferStatus::COMPLETED, "{what}");
            // SAFETY: the output's host bytes, which the GPU wrote and the wait saw done.
            let written = unsafe {
                std::slice::from_raw_parts(output.contents().as_ptr::<u8>(), length as usize)
            };
            let mut want = before;
            for fill in &fills {
                let p = &fill.plane;
                for row in 0..u64::from(p.rows) {
                    let from = (fill.bound + row * u64::from(p.source_pitch)) as usize;
                    let to = (fill.at + row * u64::from(p.pitch_words) * 4) as usize;
                    let n = p.row_bytes as usize;
                    want[to..to + n].copy_from_slice(&bytes[from..from + n]);
                }
            }
            if let Some(at) = (0..want.len()).find(|&at| written[at] != want[at]) {
                panic!(
                    "{what}: byte {at} is {:#04x}, not {:#04x}",
                    written[at], want[at]
                );
            }
        }
    });
}

/// A picture the driver decoded into a surface of its own, copied by the session's copier
/// ([`ScratchCopier`]: written by the driver's video processing into its scratch, then by the
/// kernel into the output's planes) into a dma-buf device cache laid out as CoreVideo lays a
/// surface out (each plane 32 bytes into its memory's multiple of 256, rows of the width's bytes
/// rounded up to 16, as no driver writes them in place), is the CPU's copy of the same picture
/// byte for byte, the cache read back by a blit: each plane's crop rows, and every other byte of
/// the cache as its fill left it. The first [`PICTURES`] pictures of each H.264 and HEVC stream in
/// `RVG_VA_STREAMS`'s directories, 8 and 10 bits; a picture the copier declines is counted and
/// named, and a stream the decoder refuses is left at its refusal. Each copied picture is counted
/// under [`COPIED_CODE`], and the copier's time for it, from the driver's write to the copy's end,
/// is printed by size and sample.
#[test]
#[ignore = "needs a VA-API GPU and the conformance suite's streams: run on the VA-API host"]
fn a_picture_copied_by_the_gpu_from_its_scratch_is_the_cpus_copy() {
    use reims_vgpu_video::vaapi::Display;
    use reims_vgpu_vulkan::RtPlatform;

    use super::super::tests::{
        PICTURES, UNTOUCHED, each_picture, plane_rows, usize_of, va_streams,
    };

    const PAGE: u64 = 4096;
    let counted = || {
        (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
            .find(|&(code, _)| code == COPIED_CODE)
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
        let writes = GuestWrites::new(&display);
        let mut copier =
            ScratchCopier::<RtPlatform>::new(Arc::new(Pipeline::new(&device)), runtime);
        let queue = device.new_command_queue().expect("a queue");
        let blit_and_wait = |cmd: BlitCmd<'_, RtPlatform>, what: &str| {
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
        let before = counted();
        let (mut pictures, mut declined) = (0usize, Vec::new());
        // By (width, height, sample bytes): pictures, and their least, total and most times.
        let mut times = std::collections::BTreeMap::<(u64, u64, u64), (u32, [Duration; 3])>::new();
        let mut each = |stream: &std::path::Path, id: u64, frame: &Frame| -> bool {
            let sample: u64 = if frame.bit_depth() > 8 { 2 } else { 1 };
            let (width, height) = (u64::from(frame.width()), u64::from(frame.height()));
            let (luma_bytes, chroma_bytes) = (width * sample, width.div_ceil(2) * 2 * sample);
            let chroma_rows = height.div_ceil(2);
            let pitch = (width.next_multiple_of(2) * sample).next_multiple_of(16);
            let luma_at = 32;
            let chroma_at = luma_at + pitch * height.next_multiple_of(16);
            let len = (chroma_at + pitch * chroma_rows).next_multiple_of(PAGE);
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
            let output = |offset: u64, row_bytes: u64, w: u64, rows: u64| OutputPlane {
                buffer: cache.clone_ref(),
                offset,
                bytes_per_row: pitch,
                row_bytes,
                width: w,
                height: rows,
                rows_held: rows,
                cached: true,
                host: None,
                pages: None,
                cache: None,
                surface: None,
            };
            let luma = output(luma_at, luma_bytes, width, height);
            let chroma = output(chroma_at, chroma_bytes, width.div_ceil(2), chroma_rows);
            let total = copier.total;
            match copier
                .copy(frame, &writes, &luma, &chroma)
                .expect("the copy")
            {
                DeviceWrite::Written => {
                    let took = copier.total - total;
                    let (n, [least, sum, most]) = (times.entry((width, height, sample)))
                        .or_insert((0, [Duration::MAX, Duration::ZERO, Duration::ZERO]));
                    *n += 1;
                    (*least, *sum, *most) = ((*least).min(took), *sum + took, (*most).max(took));
                }
                DeviceWrite::Declined(why) => {
                    declined.push(format!("{} picture {id}: {why}", stream.display()));
                    return false;
                }
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
            // SAFETY: the read-back buffer's `len` bytes, which the blit finished writing
            // before its command buffer completed, and which nothing writes meanwhile.
            let gpu = unsafe {
                std::slice::from_raw_parts(back.contents().as_ptr::<u8>(), usize_of(len))
            };
            if let Some(first) = (0..gpu.len()).find(|&i| gpu[i] != cpu[i]) {
                panic!(
                    "{} picture {id}, {width}x{height} at {sample}-byte samples, rows of \
                     {pitch}, luma at {luma_at}, chroma at {chroma_at}: byte {first} is {:#x} \
                     by the GPU's copy and {:#x} by the CPU's",
                    stream.display(),
                    gpu[first],
                    cpu[first]
                );
            }
            drop((luma, chroma, cache, back));
            pictures += 1;
            true
        };
        let refused = each_picture(&display, &streams, PICTURES, &mut each);
        eprintln!(
            "{pictures} pictures of {} streams copied by the GPU as by the CPU; declined: \
             {declined:?}; refused: {refused:?}",
            streams.len()
        );
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        for ((width, height, sample), (n, [least, sum, most])) in &times {
            eprintln!(
                "copy times: {width}x{height} at {sample}-byte samples: {n} pictures, {:.3} ms \
                 least, {:.3} ms each, {:.3} ms at most",
                ms(*least),
                ms(*sum / *n),
                ms(*most)
            );
        }
        assert!(pictures > 0, "no picture copied: {declined:?}");
        assert_eq!(counted(), before + pictures as u64);
    });
}
