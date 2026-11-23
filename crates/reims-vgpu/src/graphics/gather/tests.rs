use reims_vgpu_vulkan::core::testkit::FakePlatform;

use super::*;

/// A surface's plane at `offset`, `rows` rows of `row_bytes` (a row a frame as wide as the
/// surface fills), `bytes_per_row` apart.
fn target(offset: usize, bytes_per_row: usize, row_bytes: usize, rows: u32) -> TargetPlane {
    TargetPlane {
        offset,
        bytes_per_row,
        rows,
        row_bytes,
    }
}

/// A picture's three planes of `rows` rows at `pitch`, one after another from the source's
/// first byte, and nothing past them.
fn planar(pitch: u64, rows: u64) -> Planar {
    Planar {
        planes: [0, pitch * rows, 2 * pitch * rows],
        pitch,
        length: 3 * pitch * rows,
    }
}

/// A 513 by 301 picture's planes gather into a '444v' surface's: the luma's rows from the
/// source's first plane, a word a thread; the chroma's from its Cb and Cr planes, two bytes a
/// pixel; each from the plane's first word at its pitch's words.
#[test]
fn a_pictures_planes_gather_into_the_surfaces_at_their_words() {
    let planes = [
        target(0x400, 576, 513, 301),
        target(0x2_c000, 1088, 1026, 301),
    ];
    let gathered = gathers(planar(768, 301), (513, 301), &planes, 0x8_0000).expect("gathers");
    assert_eq!(
        gathered,
        [
            GatherPlane {
                first: 0,
                second: 0,
                source_pitch: 768,
                step: 1,
                components: 1,
                row_bytes: 513,
                rows: 301,
                row_words: 129,
                out_first: 0x100,
                pitch_words: 144,
            },
            GatherPlane {
                first: 768 * 301,
                second: 2 * 768 * 301,
                source_pitch: 768,
                step: 1,
                components: 2,
                row_bytes: 1026,
                rows: 301,
                row_words: 257,
                out_first: 0xb000,
                pitch_words: 272,
            },
        ]
    );
}

/// The rows and bytes gathered are the CPU's interleave's: as many rows as both the crop and the
/// plane hold, each as far as the plane's rows hold it, the chroma's in whole Cb/Cr pairs.
#[test]
fn a_gather_takes_the_rows_the_crop_and_the_planes_both_hold() {
    // A crop narrower and shorter than the planes.
    let planes = [target(0, 64, 64, 48), target(0x1000, 128, 128, 48)];
    let [luma, chroma] = gathers(planar(256, 40), (50, 40), &planes, 0x3000).expect("gathers");
    assert_eq!((luma.row_bytes, luma.rows, luma.row_words), (50, 40, 13));
    assert_eq!(
        (chroma.row_bytes, chroma.rows, chroma.row_words),
        (100, 40, 25)
    );
    // A crop wider and taller than the planes, whose chroma rows end inside a pair.
    let planes = [target(0, 64, 60, 30), target(0x1000, 128, 121, 30)];
    let [luma, chroma] = gathers(planar(256, 40), (64, 40), &planes, 0x3000).expect("gathers");
    assert_eq!((luma.row_bytes, luma.rows), (60, 30));
    assert_eq!(
        (chroma.row_bytes, chroma.rows, chroma.row_words),
        (120, 30, 30)
    );
}

/// A gather is refused, with why, where a plane lies off a word, past the surface or past the
/// source, where it would write nothing, and where a value passes 32 bits.
#[test]
fn a_gather_the_kernel_cannot_make_is_refused() {
    let surface = 0x3000;
    let good = [target(0, 64, 64, 48), target(0x1000, 128, 128, 48)];
    let source = planar(256, 48);
    assert!(gathers(source, (64, 48), &good, surface).is_ok());
    let refused = |planes: [TargetPlane; 2], source: Planar, size: (u32, u32), surface: u64| {
        gathers(source, size, &planes, surface).expect_err("a refusal")
    };
    let off_word = [target(2, 64, 64, 48), good[1]];
    assert!(refused(off_word, source, (64, 48), surface).contains("not on words"));
    let off_pitch = [good[0], target(0x1000, 130, 128, 48)];
    assert!(refused(off_pitch, source, (64, 48), surface).contains("not on words"));
    // The chroma's last row's last word ends 4 bytes past the surface.
    let short = 0x1000 + 128 * 47 + 124;
    assert!(refused(good, source, (64, 48), short).contains("past the surface"));
    // A source whose Cr plane's last row's last word ends a byte past it.
    let few = Planar {
        length: 2 * 256 * 48 + 47 * 256 + 63,
        ..source
    };
    assert!(refused(good, few, (64, 48), surface).contains("past its source"));
    let narrow = Planar {
        pitch: 32,
        ..source
    };
    assert!(refused(good, narrow, (64, 48), surface).contains("past its source"));
    assert!(refused(good, source, (64, 0), surface).contains("an empty luma plane"));
    assert!(refused(good, source, (0, 48), surface).contains("an empty luma plane"));
    let far = Planar {
        planes: [0, 1 << 32, 2 << 32],
        pitch: 256,
        length: 3 << 32,
    };
    assert!(refused(good, far, (64, 48), surface).contains("a source's reach of"));
}

/// A plane's arguments are the kernel's, in its order, and its grid covers its words.
#[test]
fn a_planes_arguments_are_the_kernels_and_its_grid_covers_its_words() {
    let plane = GatherPlane {
        first: 1,
        second: 2,
        source_pitch: 3,
        step: 4,
        components: 5,
        row_bytes: 6,
        rows: 7,
        row_words: 130,
        out_first: 9,
        pitch_words: 10,
    };
    let bytes = plane.bytes();
    let words: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    assert_eq!(words, [1, 2, 3, 4, 5, 6, 7, 130, 9, 10]);
    assert_eq!(plane.threadgroups(), MtlSize::new(3, 7, 1));
}

/// A gatherer is made with no device work: its queue and arguments come at its first gather.
#[test]
fn a_gatherer_makes_nothing_before_its_first_gather() {
    let platform = FakePlatform::new();
    let pipeline = Arc::new(GatherPipeline::<FakePlatform>::new(&platform.device()));
    let gatherer = Gatherer::new(pipeline);
    assert!(gatherer.queue.is_none() && gatherer.arguments.is_none());
}

/// `length` bytes, each a pattern of its index and `seed` that no two nearby bytes share.
fn pattern(length: usize, seed: usize) -> Vec<u8> {
    (0..length)
        .map(|i| ((i + seed).wrapping_mul(131) ^ ((i + seed) >> 8).wrapping_mul(29)) as u8)
        .collect()
}

/// A case of the kernel's test on the GPU: the crop's width and height, the surface's width and
/// height, the luma's first byte, and the luma's and the chroma's pitch.
type Case = ((u32, u32), (u32, u32), usize, usize, usize);

/// On the host's GPU, the kernel writes a 4:4:4 picture's three planes into a surface's two as
/// the CPU's interleave writes them, byte for byte: each luma row, each chroma row's Cb then Cr
/// for each pixel, at the planes' offsets and pitches, and nothing else of the surface, the bytes
/// that share a row's last word with it among them. Rows of 1 to 3 bytes past a word and none,
/// odd widths among them, CoreVideo's 1366-wide pitch, a crop narrower and shorter than the
/// planes and one wider than them.
#[test]
fn on_vulkan_the_kernel_gathers_a_4_4_4_picture_as_the_cpu_interleaves_it() {
    use reims_vgpu_vulkan::RtPlatform;

    let platform = crate::rt_test_end::platform();
    crate::entry::run(|_| {
        let device = platform.system_default_device().expect("an rt device");
        let pipeline = Arc::new(GatherPipeline::<RtPlatform>::new(&device));
        let mut gatherer = Gatherer::new(pipeline);
        let device = gatherer.device().clone_ref();
        let shared = |length: u64| {
            (device.new_buffer_of_length(OnFault::Return, length, 0))
                .expect("no raise")
                .expect("a buffer")
        };
        // A new buffer's zeros, which the gatherer fills and waits for.
        let zeroed = shared(4096);
        // SAFETY: the buffer's host bytes, which no GPU work reaches yet.
        unsafe { std::ptr::write_bytes(zeroed.contents().as_ptr::<u8>(), 0xff, 4096) };
        gatherer.zero(&zeroed, 4096).expect("the zeros");
        // SAFETY: the buffer's host bytes, which the GPU wrote and the wait saw done.
        let zeros = unsafe { std::slice::from_raw_parts(zeroed.contents().as_ptr::<u8>(), 4096) };
        assert!(zeros.iter().all(|&b| b == 0), "the zeros");
        let cases: [Case; 8] = [
            ((64, 48), (64, 48), 0x400, 64, 128),
            ((513, 37), (513, 37), 0x400, 576, 1088),
            ((1366, 9), (1366, 9), 0x20, 1376, 2752),
            ((1366, 7), (1366, 7), 0x1000, 1368, 2732),
            ((7, 5), (7, 5), 0x40, 8, 16),
            ((1, 3), (1, 3), 0, 4, 4),
            ((500, 30), (513, 37), 0x400, 576, 1088),
            ((70, 50), (66, 45), 0x80, 68, 132),
        ];
        for ((width, height), (columns, rows), luma_at, luma_pitch, chroma_pitch) in cases {
            let what = format!(
                "{width}x{height} into {columns}x{rows} from {luma_at} at {luma_pitch} and \
                 {chroma_pitch}"
            );
            let chroma_at = (luma_at + luma_pitch * rows as usize).next_multiple_of(0x1000);
            let length = chroma_at + chroma_pitch * rows as usize + 64;
            let planes = [
                target(luma_at, luma_pitch, columns as usize, rows),
                target(chroma_at, chroma_pitch, 2 * columns as usize, rows),
            ];
            let pitch = u64::from(width).next_multiple_of(256);
            let source = planar(pitch, u64::from(height));
            let gathered = gathers(source, (width, height), &planes, length as u64).expect(&what);
            let bytes = pattern(source.length as usize, 0);
            let before = pattern(length, 7)
                .into_iter()
                .map(|b| !b)
                .collect::<Vec<_>>();
            let source_buffer = shared(source.length);
            let surface = shared(length as u64);
            // SAFETY: the buffers' host bytes, which no GPU work reaches yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    source_buffer.contents().as_ptr::<u8>(),
                    bytes.len(),
                );
                std::ptr::copy_nonoverlapping(
                    before.as_ptr(),
                    surface.contents().as_ptr::<u8>(),
                    before.len(),
                );
            }
            gatherer
                .gather(&source_buffer, &surface, &gathered)
                .expect(&what);
            // SAFETY: the surface's host bytes, which the GPU wrote and the wait saw done.
            let written =
                unsafe { std::slice::from_raw_parts(surface.contents().as_ptr::<u8>(), length) };
            // The CPU's interleave: as many rows as both hold, each as far as both hold it.
            let mut want = before.clone();
            let (pitch, plane_rows) = (pitch as usize, height.min(rows) as usize);
            let [y, cb, cr] = source.planes.map(|at| at as usize);
            let luma_bytes = (width as usize).min(columns as usize);
            for row in 0..plane_rows {
                let to = luma_at + row * luma_pitch;
                want[to..to + luma_bytes].copy_from_slice(&bytes[y + row * pitch..][..luma_bytes]);
                for x in 0..luma_bytes {
                    let to = chroma_at + row * chroma_pitch + 2 * x;
                    want[to] = bytes[cb + row * pitch + x];
                    want[to + 1] = bytes[cr + row * pitch + x];
                }
            }
            if let Some(at) = (0..length).find(|&at| written[at] != want[at]) {
                panic!(
                    "{what}: byte {at} is {:#04x}, not {:#04x} (it was {:#04x})",
                    written[at], want[at], before[at]
                );
            }
        }
    });
}
