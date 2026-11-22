use reims_vgpu_video::vaapi::TargetFormat;
use reims_vgpu_vulkan::core::testkit::FakePlatform;

use super::*;

/// A frame's shape: NV12 or P010, coded at `coded`.
fn shape(format: TargetFormat, coded: (u32, u32)) -> TargetShape {
    TargetShape { format, coded }
}

/// The layout iHD gives its own input surfaces of `shape` (the VA crate's encoder docs,
/// "Imported inputs"), in Intel's Y tiles: the pitch the coded row's bytes rounded up to 128, the
/// chroma plane at the pitch times the coded height rounded up to 32 rows, its rows, half as many,
/// rounded up to 32 too.
fn y_tiled(shape: TargetShape) -> InputLayout {
    let pitch = (u64::from(shape.coded.0) * shape.format.bytes_per_sample()).next_multiple_of(128);
    let rows = u64::from(shape.coded.1).next_multiple_of(32);
    InputLayout {
        tiling: TargetTiling::IntelY,
        pitch,
        chroma_at: pitch * rows,
        length: pitch * rows + pitch * (rows / 2).next_multiple_of(32),
    }
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

/// A 330x202 frame into an input coded 336x208: each plane bound from its first byte rounded
/// down to 256, the rows and units the upload reads, the coded rows' words, and the chroma
/// plane's at the layout's chroma byte, linear at rows of 512 right after the luma plane's coded
/// rows, or in iHD's Y tiles at rows of 384, 224 rows on.
#[test]
fn a_frames_planes_fill_the_input_as_the_upload_does() {
    let luma_at = 800;
    let chroma_at = luma_at + 512 * 202;
    let length = chroma_at + 512 * 101;
    let sources = [
        source(length, luma_at, 512, 330, 202),
        source(length, chroma_at, 512, 330, 101),
    ];
    let nv12 = shape(TargetFormat::Nv12, (336, 208));
    let expected = |bound, first, rows, unit, coded_rows, at, (pitch_words, tiling)| PlaneFill {
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
            pitch_words,
            tiling,
        },
    };
    let linear = (128, TILING_LINEAR);
    assert_eq!(
        plane_fills(sources, (330, 202), nv12, InputLayout::linear(nv12)),
        Ok([
            expected(768, 32, 202, 1, 208, 0, linear),
            expected(chroma_at - 32, 32, 101, 2, 104, 512 * 208, linear),
        ])
    );
    let tiled = (96, TILING_INTEL_Y);
    assert_eq!(
        plane_fills(sources, (330, 202), nv12, y_tiled(nv12)),
        Ok([
            expected(768, 32, 202, 1, 208, 0, tiled),
            expected(chroma_at - 32, 32, 101, 2, 104, 384 * 224, tiled),
        ])
    );
    // P010's samples are two bytes: rows of 660 of 672 coded, 768 apart in the input.
    let sources = [
        source(1 << 20, 0, 1024, 660, 202),
        source(1 << 20, 1024 * 202, 1024, 660, 101),
    ];
    let p010 = shape(TargetFormat::P010, (336, 208));
    let fills = plane_fills(sources, (330, 202), p010, InputLayout::linear(p010));
    let fills = fills.expect("the fills");
    assert_eq!(
        fills.map(|f| (
            f.plane.row_bytes,
            f.plane.unit,
            f.plane.coded_words,
            f.plane.pitch_words
        )),
        [(660, 2, 168, 192), (660, 4, 168, 192)]
    );
    assert_eq!(fills.map(|f| f.at), [0, 768 * 208]);
}

/// A frame is declined, with why, where the input does not hold it or its coded rows are not
/// whole words, and where a plane holds fewer rows, or narrower ones, than the frame, or its last
/// row's last word lies past its buffer; a plane whose last word ends at its buffer's end is
/// taken. An input's layout is declined where its pitch is narrower than a coded row or, in Y
/// tiles, not a whole number of tiles, where its chroma plane's binding is not on 256, and where
/// a plane's rows, of whole rows of tiles in Y tiles, do not fit before the chroma plane or the
/// input's end.
#[test]
fn a_frame_the_input_or_its_planes_cannot_hold_is_declined() {
    let nv12 = shape(TargetFormat::Nv12, (336, 208));
    let fits = |luma: Source, chroma: Source, size: (u32, u32), shape: TargetShape| {
        plane_fills([luma, chroma], size, shape, InputLayout::linear(shape))
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
    let laid = |layout: InputLayout| plane_fills([luma, chroma], (330, 202), nv12, layout);
    assert!(laid(y_tiled(nv12)).is_ok());
    for (layout, what) in [
        (
            InputLayout {
                pitch: 320,
                ..InputLayout::linear(nv12)
            },
            "a pitch narrower than a coded row",
        ),
        (
            InputLayout {
                pitch: 512,
                chroma_at: 512 * 224,
                length: 512 * (224 + 128),
                ..y_tiled(nv12)
            },
            "a pitch of 4 Y tiles",
        ),
        (
            InputLayout {
                pitch: 448,
                chroma_at: 448 * 224,
                length: 448 * (224 + 128),
                ..y_tiled(nv12)
            },
            "a pitch not of whole Y tiles",
        ),
        (
            InputLayout {
                chroma_at: 512 * 208 + 128,
                length: 512 * 400,
                ..InputLayout::linear(nv12)
            },
            "a chroma plane off 256",
        ),
    ] {
        let is_ok = laid(layout).is_ok();
        assert_eq!(is_ok, what == "a pitch of 4 Y tiles", "{what}: {layout:?}");
    }
    for (layout, why_starts) in [
        (
            InputLayout {
                chroma_at: 384 * 208,
                ..y_tiled(nv12)
            },
            "a luma plane of 86016 bytes in 79872",
        ),
        (
            InputLayout {
                length: 384 * (224 + 104),
                ..y_tiled(nv12)
            },
            "a chroma plane of 49152 bytes in 39936",
        ),
        (
            InputLayout {
                length: 512 * (208 + 103),
                ..InputLayout::linear(nv12)
            },
            "a chroma plane of 53248 bytes in 52736",
        ),
    ] {
        let why = declined(laid(layout), why_starts);
        assert!(why.starts_with(why_starts), "{why}");
    }
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

/// The device's GPU copies make a copier each decode session (`gpu_copy`), and a filler each
/// encode session.
#[test]
fn the_devices_fills_make_a_copier_and_a_filler_each_session() {
    let platform = FakePlatform::new();
    let runtime: Arc<dyn Runtime> = Arc::new(reims_vgpu_video::core::runtime::StdRuntime);
    let fills = DeviceFills::<FakePlatform>::new(&platform.device(), &runtime);
    assert!(fills.copier().is_ok());
    assert!(fills.copier().is_ok());
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
/// upload writes it, byte for byte, each byte where the input's layout puts it: every coded row's
/// bytes, the frame's rows and the margin past them, and nothing else of the input. NV12 and P010,
/// frames with a margin of columns and rows and with none, from planes at offsets and pitches on
/// 256 bytes and off them, odd among them, into linear inputs and inputs in iHD's Y tiles.
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
            (TargetFormat::Nv12, (640, 360), (640, 360), 0, 640),
            (TargetFormat::Nv12, (1280, 720), (1280, 720), 0, 1280),
        ];
        let laid_out = |shape| [InputLayout::linear(shape), y_tiled(shape)];
        let cases = cases.into_iter().flat_map(|case| {
            let (format, _, coded, _, _) = case;
            laid_out(shape(format, coded)).map(|layout| (case, layout))
        });
        for ((format, (width, height), coded, first, pitch), layout) in cases {
            let what = format!(
                "{format:?} {width}x{height} in {coded:?} from {first} at {pitch}, into {layout:?}"
            );
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
            let fills = plane_fills(sources, (width, height), shape, layout).expect(&what);
            let input_length = layout.length;
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
                destinations: [&input, &input],
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
            let mut want = vec![UNTOUCHED; input_length as usize];
            let mut plane_of = vec![""; input_length as usize];
            for (fill, which) in fills.iter().zip(["luma", "chroma"]) {
                let expected = uploaded(&bytes, fill, coded_bytes);
                for (row, bytes) in expected.chunks_exact(coded_bytes).enumerate() {
                    for (x, &byte) in bytes.iter().enumerate() {
                        let at = layout.tiling.byte_at(layout.pitch, x as u64, row as u64);
                        let at = (fill.at + at) as usize;
                        assert_eq!(plane_of[at], "", "{what}: {which} {x}, {row} at {at} twice");
                        (want[at], plane_of[at]) = (byte, which);
                    }
                }
            }
            if let Some(at) = (0..want.len()).find(|&at| written[at] != want[at]) {
                panic!(
                    "{what}: byte {at} (of the {}) is {:#04x}, not {:#04x}",
                    if plane_of[at].is_empty() {
                        "unwritten"
                    } else {
                        plane_of[at]
                    },
                    written[at],
                    want[at]
                );
            }
        }
    });
}
