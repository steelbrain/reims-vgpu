use reims_vgpu_video::vaapi::TargetFormat;
use reims_vgpu_vulkan::core::testkit::FakePlatform;

use super::*;

/// A frame's shape: NV12 or P010, coded at `coded`.
fn shape(format: TargetFormat, coded: (u32, u32)) -> TargetShape {
    TargetShape { format, coded }
}

/// A plane of `height` rows of `row_bytes`, `bytes_per_row` apart from `offset`, in a buffer of
/// `length` bytes.
fn source(length: u64, offset: u64, bytes_per_row: u64, row_bytes: u64, height: u64) -> Source {
    Source {
        length,
        offset,
        bytes_per_row,
        row_bytes,
        height,
    }
}

/// The bytes the VA crate's upload writes into a plane of the input from `bytes` (its
/// `copy_planes`): `coded_rows` rows of `coded_bytes`, each the source's row `fill.first` and on,
/// the last row repeated past the frame's rows, each row's bytes past `row_bytes` its last `unit`
/// bytes repeated.
fn uploaded(bytes: &[u8], fill: &PlaneFill, coded_bytes: usize) -> Vec<u8> {
    let p = &fill.plane;
    let (row_bytes, unit) = (p.row_bytes as usize, p.unit as usize);
    let mut plane = Vec::new();
    for row in 0..p.coded_rows {
        let from = fill.bound as usize
            + p.first as usize
            + row.min(p.rows - 1) as usize * p.source_pitch as usize;
        let source = &bytes[from..from + row_bytes];
        plane.extend_from_slice(source);
        let last = &source[row_bytes - unit..];
        plane.extend((0..coded_bytes - row_bytes).map(|m| last[m % unit]));
    }
    plane
}

/// An input's rows lie a pitch apart, the coded row's bytes rounded up to 256, and its chroma
/// plane's coded rows, half as many, right after the luma plane's.
#[test]
fn an_inputs_planes_lie_one_pitch_on_256_apart() {
    for (format, coded, pitch) in [
        (TargetFormat::Nv12, (336, 208), 512),
        (TargetFormat::P010, (336, 208), 768),
        (TargetFormat::Nv12, (1920, 1088), 2048),
        (TargetFormat::P010, (3840, 2160), 7680),
    ] {
        let shape = shape(format, coded);
        assert_eq!(pitch_of(shape), pitch, "{shape:?}");
        let rows = u64::from(coded.1);
        assert_eq!(
            input_bytes(shape, pitch),
            (pitch * rows * 3 / 2, pitch * rows),
            "{shape:?}"
        );
    }
}

/// A 330x202 frame into an input coded 336x208: each plane bound from its first byte rounded
/// down to 256, the rows and units the upload reads, the coded rows' words, and the chroma
/// plane's from the luma plane's coded rows on.
#[test]
fn a_frames_planes_fill_the_input_as_the_upload_does() {
    let luma_at = 800;
    let chroma_at = luma_at + 512 * 202;
    let length = chroma_at + 512 * 101;
    let sources = [
        source(length, luma_at, 512, 330, 202),
        source(length, chroma_at, 512, 330, 101),
    ];
    let nv12 = plane_fills(sources, (330, 202), shape(TargetFormat::Nv12, (336, 208)));
    let expected = |bound, first, rows, unit, coded_rows, at| PlaneFill {
        bound,
        at,
        plane: FillPlane {
            first,
            source_pitch: 512,
            row_bytes: 330,
            rows,
            unit,
            coded_words: 84,
            coded_rows,
            pitch_words: 128,
            tiling: TILING_LINEAR,
        },
    };
    assert_eq!(
        nv12,
        Ok([
            expected(768, 32, 202, 1, 208, 0),
            expected(chroma_at - 32, 32, 101, 2, 104, 512 * 208),
        ])
    );
    // P010's samples are two bytes: rows of 660 of 672 coded, 768 apart in the input.
    let sources = [
        source(1 << 20, 0, 1024, 660, 202),
        source(1 << 20, 1024 * 202, 1024, 660, 101),
    ];
    let p010 =
        plane_fills(sources, (330, 202), shape(TargetFormat::P010, (336, 208))).expect("the fills");
    assert_eq!(
        p010.map(|f| (
            f.plane.row_bytes,
            f.plane.unit,
            f.plane.coded_words,
            f.plane.pitch_words
        )),
        [(660, 2, 168, 192), (660, 4, 168, 192)]
    );
    assert_eq!(p010.map(|f| f.at), [0, 768 * 208]);
}

/// A frame is declined, with why, where the input does not hold it or its coded rows are not
/// whole words, and where a plane holds fewer rows, or narrower ones, than the frame, or its last
/// row's last word lies past its buffer; a plane whose last word ends at its buffer's end is
/// taken.
#[test]
fn a_frame_the_input_or_its_planes_cannot_hold_is_declined() {
    let nv12 = shape(TargetFormat::Nv12, (336, 208));
    let fits = |luma: Source, chroma: Source, size: (u32, u32), shape: TargetShape| {
        plane_fills([luma, chroma], size, shape)
    };
    // Luma: 202 rows of 330 from byte 0, 512 apart; its last word ends at byte 202 * 512 - 180.
    let luma_end = 201 * 512 + 332;
    let luma = source(luma_end, 0, 512, 330, 202);
    let chroma = source(1 << 20, 0, 512, 330, 101);
    assert!(fits(luma, chroma, (330, 202), nv12).is_ok());
    let declined = |r: Result<[PlaneFill; 2], String>, what: &str| match r {
        Ok(fills) => panic!("{what}: taken: {fills:?}"),
        Err(why) => why,
    };
    let why = declined(fits(luma, chroma, (340, 202), nv12), "a frame wider");
    assert!(why.starts_with("a frame of 340x202"), "{why}");
    let why = declined(
        fits(
            luma,
            chroma,
            (330, 202),
            shape(TargetFormat::Nv12, (330, 208)),
        ),
        "coded rows of 330 bytes",
    );
    assert!(why.contains("not whole words"), "{why}");
    let short = source(1 << 20, 0, 512, 330, 201);
    let why = declined(fits(short, chroma, (330, 202), nv12), "a luma plane short");
    assert!(why.starts_with("a luma plane of 201 rows"), "{why}");
    let narrow = source(1 << 20, 0, 512, 328, 101);
    let why = declined(
        fits(luma, narrow, (330, 202), nv12),
        "a chroma plane narrow",
    );
    assert!(
        why.starts_with("a chroma plane of 101 rows of 328 bytes"),
        "{why}"
    );
    let close = source(luma_end - 1, 0, 512, 330, 202);
    let why = declined(fits(close, chroma, (330, 202), nv12), "a last word past");
    assert!(why.contains(&format!("ends at byte {luma_end}")), "{why}");
}

/// A plane's arguments are `shaders.metal`'s `VideoFillPlane`, nine 32-bit words in its order,
/// and its dispatch a threadgroup of 64 threads a 64 words of a row, a row of threadgroups each
/// coded row.
#[test]
fn a_planes_arguments_are_the_kernels_and_its_grid_covers_its_words() {
    let plane = FillPlane {
        first: 1,
        source_pitch: 2,
        row_bytes: 3,
        rows: 4,
        unit: 5,
        coded_words: 130,
        coded_rows: 7,
        pitch_words: 8,
        tiling: 9,
    };
    let bytes = plane.bytes();
    let words: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    assert_eq!(words, [1, 2, 3, 4, 5, 130, 7, 8, 9]);
    assert_eq!(plane.threadgroups(), MtlSize::new(3, 7, 1));
}

/// The device's GPU copies make no copier, so a decode's pictures land as with none, its reason
/// the same; and a filler each encode session.
#[test]
fn the_devices_fills_make_no_copier_and_a_filler_each_session() {
    let platform = FakePlatform::new();
    let runtime: Arc<dyn Runtime> = Arc::new(reims_vgpu_video::core::runtime::StdRuntime);
    let fills = DeviceFills::<FakePlatform>::new(&platform.device(), &runtime);
    match fills.copier() {
        Ok(_) => panic!("a copier"),
        Err(why) => assert_eq!(why, "the platform makes no GPU copies"),
    }
    assert!(fills.filler().is_ok());
    assert!(fills.filler().is_ok());
}

/// `length` bytes, each a pattern of its index that no two nearby bytes share.
fn pattern(length: usize) -> Vec<u8> {
    (0..length)
        .map(|i| (i.wrapping_mul(131) ^ (i >> 8).wrapping_mul(29)) as u8)
        .collect()
}

/// On the host's GPU, the kernel writes each plane of a frame into an input as the VA crate's
/// upload writes it, byte for byte: every coded row's bytes, the frame's rows and the margin past
/// them, and nothing past each coded row's bytes. NV12 and P010, frames with a margin of columns
/// and rows and with none, from planes at offsets and pitches on 256 bytes and off them, odd
/// among them.
#[test]
fn on_rt_the_kernel_writes_a_frame_as_the_upload_writes_it() {
    use reims_vgpu_vulkan::RtPlatform;

    const UNTOUCHED: u8 = 0xa5;
    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        let pipeline = Pipeline::<RtPlatform> {
            device: device.clone_ref(),
            made: OnceLock::new(),
        };
        let pipeline = pipeline.get().expect("the kernel's pipeline");
        let queue = device.new_command_queue().expect("a queue");
        let shared = |length: u64| {
            (device.new_buffer_of_length(OnFault::Return, length, 0))
                .expect("no raise")
                .expect("a buffer")
        };
        let arguments = shared(2 * ARGUMENTS_APART);
        let cases = [
            (TargetFormat::Nv12, (330, 202), (336, 208), 32, 512),
            (TargetFormat::Nv12, (320, 192), (320, 192), 0, 320),
            (TargetFormat::Nv12, (330, 202), (336, 208), 35, 333),
            (TargetFormat::Nv12, (64, 48), (64, 64), 258, 700),
            (TargetFormat::P010, (330, 202), (336, 208), 32, 768),
            (TargetFormat::P010, (330, 202), (352, 224), 2, 662),
        ];
        for (format, (width, height), coded, first, pitch) in cases {
            let what = format!("{format:?} {width}x{height} in {coded:?} from {first} at {pitch}");
            let shape = shape(format, coded);
            let sample = format.bytes_per_sample();
            let row_bytes = u64::from(width) * sample;
            let chroma_first = first + pitch * u64::from(height) + 3;
            let length = (chroma_first + pitch * u64::from(height / 2) + 64).next_multiple_of(4);
            let bytes = pattern(length as usize);
            let source_buffer = shared(length);
            // SAFETY: the buffer's `length` host bytes, which no GPU work reaches yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    source_buffer.contents().as_ptr::<u8>(),
                    bytes.len(),
                );
            }
            let sources = [
                source(length, first, pitch, row_bytes, u64::from(height)),
                source(
                    length,
                    chroma_first,
                    pitch,
                    row_bytes,
                    u64::from(height / 2),
                ),
            ];
            let fills = plane_fills(sources, (width, height), shape).expect(&what);
            let input_pitch = pitch_of(shape);
            let (input_length, _) = input_bytes(shape, input_pitch);
            let input = shared(input_length);
            // SAFETY: the input's host bytes, which no GPU work reaches yet.
            unsafe {
                std::ptr::write_bytes(
                    input.contents().as_ptr::<u8>(),
                    UNTOUCHED,
                    input_length as usize,
                );
            }
            let cb = (queue.command_buffer(OnFault::Return))
                .expect("no raise")
                .expect("a command buffer");
            let planes = Planes::<RtPlatform> {
                pipeline: &pipeline,
                arguments: &arguments,
                destination: &input,
                sources: [&source_buffer, &source_buffer],
            };
            assert_eq!(
                encode_planes::<RtPlatform>(&cb, &planes, &fills),
                None,
                "{what}"
            );
            cb.commit(OnFault::Return).expect("the commit");
            cb.wait_until_completed(OnFault::Return).expect("the wait");
            assert_eq!(cb.status(), CommandBufferStatus::COMPLETED, "{what}");
            // SAFETY: the input's host bytes, which the GPU wrote and the wait saw done.
            let written = unsafe {
                std::slice::from_raw_parts(input.contents().as_ptr::<u8>(), input_length as usize)
            };
            let coded_bytes = (u64::from(coded.0) * sample) as usize;
            for (fill, which) in fills.iter().zip(["luma", "chroma"]) {
                let expected = uploaded(&bytes, fill, coded_bytes);
                for row in 0..fill.plane.coded_rows as usize {
                    let at = fill.at as usize + row * input_pitch as usize;
                    let got = &written[at..at + coded_bytes];
                    let want = &expected[row * coded_bytes..(row + 1) * coded_bytes];
                    assert!(got == want, "{what}: {which} row {row}");
                    let past = &written[at + coded_bytes..at + input_pitch as usize];
                    assert!(
                        past.iter().all(|&b| b == UNTOUCHED),
                        "{what}: {which} row {row} past its coded bytes"
                    );
                }
            }
        }
    });
}
