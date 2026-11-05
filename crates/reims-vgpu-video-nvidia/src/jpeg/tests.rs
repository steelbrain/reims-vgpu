use super::plan::{ByteRect, Route};
use super::*;
use reims_vgpu_video_codec::jpeg::{DEFAULT_AC_LUMINANCE, DEFAULT_DC_LUMINANCE, parse};

use crate::cuda::GpuSelector;

/// A baseline image of `width` by `height` whose components are sampled
/// `factors`, written here from T.81's syntax: one table of quantization
/// values of 1, Annex K's luminance Huffman tables for every component, and
/// scan data that nothing here decodes.
fn image(width: u16, height: u16, factors: &[(u8, u8)]) -> Vec<u8> {
    let mut out = vec![0xff, 0xd8];
    let mut dqt = vec![0];
    dqt.extend_from_slice(&[1; 64]);
    stream_segment(&mut out, 0xdb, &dqt);
    let mut sof = vec![8];
    sof.extend_from_slice(&height.to_be_bytes());
    sof.extend_from_slice(&width.to_be_bytes());
    sof.push(u8::try_from(factors.len()).unwrap_or(0));
    for (id, &(h, v)) in (1u8..).zip(factors) {
        sof.extend_from_slice(&[id, (h << 4) | v, 0]);
    }
    stream_segment(&mut out, 0xc0, &sof);
    for (class, table) in [(0x00, DEFAULT_DC_LUMINANCE), (0x10, DEFAULT_AC_LUMINANCE)] {
        let mut dht = vec![class];
        dht.extend_from_slice(&table.counts);
        dht.extend_from_slice(table.values);
        stream_segment(&mut out, 0xc4, &dht);
    }
    let mut sos = vec![u8::try_from(factors.len()).unwrap_or(0)];
    for id in (1u8..).take(factors.len()) {
        sos.extend_from_slice(&[id, 0x00]);
    }
    sos.extend_from_slice(&[0, 63, 0]);
    stream_segment(&mut out, 0xda, &sos);
    out.extend_from_slice(&[0x12, 0x34]);
    out.extend_from_slice(&[0xff, 0xd9]);
    out
}

fn stream_segment(out: &mut Vec<u8>, marker: u8, payload: &[u8]) {
    let length = u16::try_from(payload.len() + 2).unwrap_or(u16::MAX);
    out.extend_from_slice(&[0xff, marker]);
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(payload);
}

const YUV420: &[(u8, u8)] = &[(2, 2), (1, 1), (1, 1)];
const YUV444: &[(u8, u8)] = &[(1, 1), (1, 1), (1, 1)];

fn request(
    data: &[u8],
    (width, height): (u32, u32),
    subsampling: JpegSubsampling,
    region: Option<PixelRect>,
    scale: u32,
    format: JpegOutputFormat,
) -> JpegDecode<'_> {
    JpegDecode {
        data,
        width,
        height,
        subsampling,
        region: region.unwrap_or(PixelRect {
            x: 0,
            y: 0,
            width,
            height,
        }),
        scale,
        format,
    }
}

fn plan(request: &JpegDecode<'_>) -> Result<Plan, JpegError> {
    let image = parse(request.data).map_err(stream_error)?;
    Plan::of(&image, request)
}

#[test]
fn a_whole_image_is_decoded_at_its_own_sampling() {
    let data = image(640, 480, YUV420);
    let request = request(
        &data,
        (640, 480),
        JpegSubsampling::Yuv420,
        None,
        1,
        JpegOutputFormat::Biplanar420,
    );
    let Ok(plan) = plan(&request) else {
        unreachable!("a plan");
    };
    assert_eq!(plan.output, nvjpeg::NVJPEG_OUTPUT_YUV);
    let layout = JpegSubsampling::Yuv420;
    assert_eq!(
        (plan.scale_factor, plan.roi, plan.route),
        (0, None, Route::Planes { layout })
    );
    assert!(!plan.resampled());
    let decoded: Vec<_> = plan.planes.iter().map(|p| p.decoded).collect();
    assert_eq!(decoded, [(640, 480), (320, 240), (320, 240)]);
    // Rows at a pitch of 768, the planes one after another.
    let offsets: Vec<_> = plan.planes.iter().map(|p| p.device_offset).collect();
    assert_eq!(offsets, [0, 768 * 480, 768 * 480 + 512 * 240]);
    assert_eq!(plan.staged_bytes, 640 * 480 + 2 * 320 * 240);
}

#[test]
fn a_crop_at_scale_1_is_the_region_of_interest() {
    let data = image(333, 211, YUV420);
    let region = PixelRect {
        x: 320,
        y: 208,
        width: 13,
        height: 3,
    };
    let request = request(
        &data,
        (333, 211),
        JpegSubsampling::Yuv420,
        Some(region),
        1,
        JpegOutputFormat::Biplanar420,
    );
    let Ok(plan) = plan(&request) else {
        unreachable!("a plan");
    };
    assert_eq!(plan.roi, Some([320, 208, 13, 3]));
    assert!(!plan.resampled());
    let shapes: Vec<_> = plan.planes.iter().map(|p| (p.decoded, p.copied)).collect();
    let rect = |width, height| ByteRect {
        x: 0,
        y: 0,
        width,
        height,
    };
    assert_eq!(
        shapes,
        [
            ((13, 3), rect(13, 3)),
            ((7, 2), rect(7, 2)),
            ((7, 2), rect(7, 2))
        ]
    );
}

#[test]
fn a_crop_at_a_scale_is_cut_from_the_whole_scaled_image() {
    let data = image(640, 480, YUV444);
    let region = PixelRect {
        x: 240,
        y: 160,
        width: 160,
        height: 160,
    };
    let request = request(
        &data,
        (640, 480),
        JpegSubsampling::Yuv444,
        Some(region),
        4,
        JpegOutputFormat::Biplanar420,
    );
    let Ok(plan) = plan(&request) else {
        unreachable!("a plan");
    };
    assert_eq!((plan.scale_factor, plan.roi), (2, None));
    assert_eq!(
        plan.route,
        Route::Planes {
            layout: JpegSubsampling::Yuv420
        }
    );
    // nvJPEG's samples from 60, 40 of them; the luma is them, and the
    // chroma, at 4:2:0's sampling, is made of them.
    let crop = ByteRect {
        x: 60,
        y: 40,
        width: 40,
        height: 40,
    };
    let [luma, cb, cr] = &plan.planes[..] else {
        unreachable!("three planes");
    };
    assert_eq!(
        (luma.decoded, luma.copied, luma.resample),
        ((160, 120), crop, None)
    );
    let halved = |start: f64, length: f64, end: f64| convert::Axis {
        start,
        length,
        end,
        base: start,
        unit: 4.0,
        source: 40,
        output: 20,
    };
    for chroma in [cb, cr] {
        assert_eq!((chroma.decoded, chroma.copied), ((160, 120), crop));
        assert_eq!(
            chroma.resample,
            Some([halved(240.0, 160.0, 400.0), halved(160.0, 160.0, 320.0)])
        );
    }
}

#[test]
fn an_odd_size_at_a_scale_is_resampled_from_the_samples_it_covers() {
    // 333x211 at 1/2 is 167x106 over the whole image, where nvJPEG's 167
    // samples cover 334: each plane is resampled, from all of nvJPEG's.
    let data = image(333, 211, YUV420);
    let request = request(
        &data,
        (333, 211),
        JpegSubsampling::Yuv420,
        None,
        2,
        JpegOutputFormat::Biplanar420,
    );
    let Ok(plan) = plan(&request) else {
        unreachable!("a plan");
    };
    let shapes: Vec<_> = plan
        .planes
        .iter()
        .map(|p| (p.copied.width, p.copied.height, p.resample.is_some()))
        .collect();
    assert_eq!(shapes, [(167, 106, true), (84, 53, true), (84, 53, true)]);
    let planes = written(&plan, &request, [50, 60, 70]);
    assert!(planes[0][..167 * 106].iter().all(|&y| y == 50));
    assert!(
        planes[1][..168 * 53]
            .as_chunks::<2>()
            .0
            .iter()
            .all(|pair| *pair == [60, 70])
    );
}

#[test]
fn chroma_finer_than_the_scaled_decode_s_is_decoded_at_a_finer_scale() {
    // 4:2:0 into 4:2:2 at 1/2: the layout's chroma rows are the image's, so
    // nvJPEG decodes at 1/1, and the luma is resampled to 1/2.
    let data = image(640, 480, YUV420);
    let at = |scale| {
        let request = request(
            &data,
            (640, 480),
            JpegSubsampling::Yuv420,
            Some(PixelRect {
                x: 64,
                y: 32,
                width: 320,
                height: 160,
            }),
            scale,
            JpegOutputFormat::Yuyv,
        );
        plan(&request).ok()
    };
    let Some(plan) = at(2) else {
        unreachable!("a plan");
    };
    assert_eq!(
        (plan.decode_scale(), plan.roi),
        (1, Some([64, 32, 320, 160]))
    );
    let [luma, cb, _] = &plan.planes[..] else {
        unreachable!("three planes");
    };
    assert_eq!(
        (luma.decoded, luma.copied.width, luma.copied.height),
        ((320, 160), 320, 160)
    );
    assert!(luma.resample.is_some());
    // Its chroma: 160 columns halved into 80, and 80 rows into 80, copied.
    assert_eq!(
        (cb.decoded, cb.copied.width, cb.copied.height),
        ((160, 80), 160, 80)
    );
    let Some([across, down]) = cb.resample else {
        unreachable!("resampled across");
    };
    assert!(!across.is_copy() && down.is_copy());
    // At 1/4, nvJPEG's 1/2 has the rows.
    let Some(plan) = at(4) else {
        unreachable!("a plan");
    };
    assert_eq!((plan.decode_scale(), plan.roi), (2, None));
}

#[test]
fn bgra_at_scale_1_is_decoded_as_interleaved_bgr() {
    let data = image(333, 211, YUV420);
    let request = request(
        &data,
        (333, 211),
        JpegSubsampling::Yuv420,
        None,
        1,
        JpegOutputFormat::Bgra,
    );
    let Ok(plan) = plan(&request) else {
        unreachable!("a plan");
    };
    assert_eq!(plan.output, nvjpeg::NVJPEG_OUTPUT_BGRI);
    assert_eq!(plan.route, Route::Bgr);
    let shapes: Vec<_> = plan.planes.iter().map(|p| (p.decoded, p.copied)).collect();
    let copied = ByteRect {
        x: 0,
        y: 0,
        width: 999,
        height: 211,
    };
    assert_eq!(shapes, [((999, 211), copied)]);
}

#[test]
fn bgra_at_a_scale_is_made_of_the_planes() {
    let data = image(640, 480, YUV420);
    let request = request(
        &data,
        (640, 480),
        JpegSubsampling::Yuv420,
        None,
        2,
        JpegOutputFormat::Bgra,
    );
    let Ok(plan) = plan(&request) else {
        unreachable!("a plan");
    };
    assert_eq!(
        (plan.output, plan.route),
        (nvjpeg::NVJPEG_OUTPUT_YUV, Route::Bgra)
    );
    assert!(!plan.resampled());
    let decoded: Vec<_> = plan.planes.iter().map(|p| p.decoded).collect();
    assert_eq!(decoded, [(320, 240), (160, 120), (160, 120)]);
    let [pixels] = &written(&plan, &request, [100, 128, 128])[..] else {
        unreachable!("one plane");
    };
    assert!(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == [100, 100, 100, 255])
    );
}

#[test]
fn what_is_refused_is_refused_before_the_gpu() {
    let data = image(640, 480, YUV420);
    let refused =
        |subsampling, format, size| plan(&request(&data, size, subsampling, None, 1, format)).err();
    use JpegOutputFormat as F;
    use JpegSubsampling as S;
    assert!(matches!(
        refused(S::Yuv420, F::Biplanar444, (640, 480)),
        Some(JpegError::UnsupportedFormat(_))
    ));
    assert!(matches!(
        refused(S::Yuv422, F::Biplanar422, (640, 480)),
        Some(JpegError::SubsamplingMismatch)
    ));
    assert!(matches!(
        refused(S::Yuv420, F::Biplanar420, (320, 480)),
        Some(JpegError::SizeMismatch { .. })
    ));
    let yuv411 = image(640, 480, &[(4, 1), (1, 1), (1, 1)]);
    let refused = plan(&request(
        &yuv411,
        (640, 480),
        S::Yuv411,
        None,
        1,
        F::Biplanar422,
    ));
    assert!(matches!(refused, Err(JpegError::UnsupportedStream(_))));
}

/// Writes a plan's output from staged planes of `values`, each sample its
/// plane's value, into a destination of `format`, and gives the
/// destination's planes.
fn written(plan: &Plan, request: &JpegDecode<'_>, values: [u8; 3]) -> Vec<Vec<u8>> {
    let mut staged = vec![0; plan.staged_bytes];
    for (plane, value) in plan.planes.iter().zip(values) {
        let len = plane.copied.width * plane.copied.height;
        staged[plane.staged_offset..plane.staged_offset + len].fill(value);
    }
    let (width, height) = plan.output_size;
    let shapes: Vec<(usize, usize)> = match request.format {
        JpegOutputFormat::Biplanar420 => {
            vec![(width, height), (width.div_ceil(2) * 2, height.div_ceil(2))]
        }
        JpegOutputFormat::Biplanar422 => vec![(width, height), (width.div_ceil(2) * 2, height)],
        JpegOutputFormat::Biplanar444 => vec![(width, height), (width * 2, height)],
        JpegOutputFormat::Yuyv | JpegOutputFormat::Uyvy => vec![(width.div_ceil(2) * 4, height)],
        JpegOutputFormat::Bgra => vec![(width * 4, height)],
    };
    let mut buffers: Vec<Vec<u8>> = shapes
        .iter()
        .map(|&(row, rows)| vec![7; row * rows])
        .collect();
    let mut planes: Vec<PlaneDestination<'_>> = buffers
        .iter_mut()
        .zip(&shapes)
        .map(|(bytes, &(row, rows))| PlaneDestination {
            bytes,
            bytes_per_row: row,
            rows: u32::try_from(rows).unwrap_or(0),
        })
        .collect();
    let result = plan.write(&staged, request, &mut planes);
    assert_eq!(result, Ok(()));
    drop(planes);
    buffers
}

#[test]
fn chroma_of_another_sampling_is_resampled_and_written() {
    // 4:4:4 at 1/2 into 4:2:2's packed layout: each sample written.
    let data = image(64, 32, YUV444);
    let request = request(
        &data,
        (64, 32),
        JpegSubsampling::Yuv444,
        None,
        2,
        JpegOutputFormat::Yuyv,
    );
    let Ok(plan) = plan(&request) else {
        unreachable!("a plan");
    };
    let [packed] = &written(&plan, &request, [16, 100, 200])[..] else {
        unreachable!("one plane");
    };
    assert_eq!(packed.len(), 32 * 2 * 16);
    assert!(
        packed
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pair| *pair == [16, 100, 16, 200])
    );
}

#[test]
fn grey_writes_neutral_chroma_and_bgra_is_opaque() {
    let data = image(17, 9, &[(1, 1)]);
    let grey = request(
        &data,
        (17, 9),
        JpegSubsampling::Grey,
        None,
        1,
        JpegOutputFormat::Biplanar420,
    );
    let Ok(grey_plan) = plan(&grey) else {
        unreachable!("a plan");
    };
    let planes = written(&grey_plan, &grey, [50, 0, 0]);
    assert!(planes[0].iter().all(|&y| y == 50));
    assert!(planes[1].iter().all(|&c| c == 0x80));

    let bgra = request(
        &data,
        (17, 9),
        JpegSubsampling::Grey,
        None,
        1,
        JpegOutputFormat::Bgra,
    );
    let Ok(plan) = plan(&bgra) else {
        unreachable!("a plan");
    };
    let planes = written(&plan, &bgra, [60, 0, 0]);
    assert!(
        planes[0]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| *pixel == [60, 60, 60, 255])
    );
}

/// nvJPEG's hardware decoder opens on the first NVIDIA GPU and decodes its trial image. libnvjpeg
/// is a CUDA toolkit library, which the driver does not install: where it does not load, the
/// backend has no JPEG engine (`open_backend`), and the test says so and checks nothing.
#[test]
#[ignore = "needs an NVIDIA GPU and libnvjpeg: run on the NVIDIA host"]
fn the_hardware_decoder_opens_and_decodes_its_trial_image() {
    if let Err(error) = Nvjpeg::get()
        && error.symbol.is_none()
    {
        eprintln!("libnvjpeg does not load ({error}): skipped");
        return;
    }
    let gpu = match Gpu::open(&GpuSelector::First) {
        Ok(gpu) => Arc::new(gpu),
        Err(error) => unreachable!("an NVIDIA GPU: {error}"),
    };
    let decoder = NvidiaJpegDecoder::new(gpu);
    assert!(decoder.is_ok(), "{:?}", decoder.err());
}
