//! What the card tests share: the streams they decode, a frame read back, and the judgement of
//! a stream's frames against its references, film grain's by its residual's statistics where
//! the driver's grain is its own. The VA-API and NVIDIA modules' ignored tests use it, on a
//! Linux host.

use super::*;

/// The streams a card test decodes: those the decoder takes, or of them those `AV1_STREAMS`
/// names (comma-separated) where it is set, for a run on one stream.
pub(crate) fn card_streams() -> Vec<&'static Stream> {
    let named = std::env::var("AV1_STREAMS").ok();
    (STREAMS.iter())
        .filter(|s| s.decodes)
        .filter(|s| {
            named
                .as_deref()
                .is_none_or(|n| n.split(',').any(|n| n == s.name))
        })
        .collect()
}

/// A plane of a frame a card test read back, in code values (P010's shifted down to 10 bits),
/// its rows cropped.
pub(crate) struct Plane {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) samples: Vec<i32>,
}

/// A frame a card test read back: its MD5 as `framemd5` hashes it, and its planes, Y, U and V.
pub(crate) struct ReadFrame {
    pub(crate) md5: String,
    pub(crate) planes: [Plane; 3],
}

impl ReadFrame {
    /// The frame of `size` whose luma rows are `luma` and whose interleaved chroma rows are
    /// `chroma`, packed, at `bit_depth` (past 8, two bytes a sample, as P010 holds them).
    pub(crate) fn of(luma: &[u8], chroma: &[u8], size: (usize, usize), bit_depth: u8) -> Self {
        let mut md5 = reims_vgpu_video_conformance::md5::Md5::new();
        md5.update(luma);
        md5.update(chroma);
        let values = |bytes: &[u8]| -> Vec<i32> {
            if bit_depth > 8 {
                let sample = |c: &[u8; 2]| i32::from(u16::from_le_bytes(*c) >> 6);
                bytes.as_chunks::<2>().0.iter().map(sample).collect()
            } else {
                bytes.iter().map(|&b| i32::from(b)).collect()
            }
        };
        let (width, height) = size;
        let uv = values(chroma);
        let chroma_plane = |first: usize| Plane {
            width: width.div_ceil(2),
            height: height.div_ceil(2),
            samples: uv.iter().skip(first).step_by(2).copied().collect(),
        };
        let luma = Plane {
            width,
            height,
            samples: values(luma),
        };
        assert_eq!(
            luma.samples.len(),
            width * height,
            "a luma plane of its size"
        );
        Self {
            md5: md5.hex(),
            planes: [luma, chroma_plane(0), chroma_plane(1)],
        }
    }
}

/// A film grain residual's statistics, of one plane: the frame with grain less the frame
/// without, its mean and standard deviation, and its variance in each whole 32x32 block of the
/// plane, in raster order.
#[derive(Debug)]
struct Residual {
    mean: f64,
    std: f64,
    blocks: Vec<f64>,
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, variance.sqrt())
}

impl Residual {
    fn of(on: &Plane, off: &Plane) -> Self {
        assert_eq!((on.width, on.height), (off.width, off.height));
        let r: Vec<f64> = (on.samples.iter().zip(&off.samples))
            .map(|(&a, &b)| f64::from(a - b))
            .collect();
        let (mean, std) = mean_std(&r);
        let mut blocks = Vec::new();
        for y in (0..on.height / 32).map(|by| by * 32) {
            for x in (0..on.width / 32).map(|bx| bx * 32) {
                let block: Vec<f64> = (y..y + 32)
                    .flat_map(|row| {
                        r[row * on.width + x..row * on.width + x + 32]
                            .iter()
                            .copied()
                    })
                    .collect();
                blocks.push(mean_std(&block).1.powi(2));
            }
        }
        Self { mean, std, blocks }
    }

    /// A stream's `.grain` file: each frame's Y, U and V.
    fn parse(text: &str) -> Vec<[Residual; 3]> {
        let mut frames: Vec<[Residual; 3]> = Vec::new();
        for line in text
            .lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
        {
            let fields: Vec<&str> = line.split(' ').collect();
            let number = |s: &str| s.parse::<f64>().expect("a number");
            let residual = Residual {
                mean: number(fields[2]),
                std: number(fields[3]),
                blocks: fields[4..].iter().map(|s| number(s)).collect(),
            };
            let frame: usize = fields[0].parse().expect("a frame");
            let plane = ["Y", "U", "V"]
                .iter()
                .position(|p| *p == fields[1])
                .expect("a plane");
            if plane == 0 {
                assert_eq!(frame, frames.len());
                frames.push([residual, Residual::empty(), Residual::empty()]);
            } else {
                frames[frame][plane] = residual;
            }
        }
        frames
    }

    fn empty() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            blocks: Vec::new(),
        }
    }
}

/// The Pearson correlation of `a` and `b`.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let ((ma, sa), (mb, sb)) = (mean_std(a), mean_std(b));
    let covariance = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / a.len() as f64;
    covariance / (sa * sb)
}

/// How far a driver's grain residual may lie from the specification's, plane by plane, for
/// the frame to count as the grain the stream asks for, of the driver's own synthesis: its
/// mean, in code values; its standard deviation, relative; and the correlation of its 32x32
/// blocks' variances with the specification's, which follow the scaling function, so the
/// grain's strength by the picture's intensity.
///
/// The mean's bound is one code value. The others are iHD's worst over `aom-grain`'s 9 frames
/// and 3 planes, a deviation 4.86% off the specification's and a correlation of 0.8695
/// (**measured** on the Iris Xe, intel-media-driver 26.3.5, 2026-11-27, m3-max-77438);
/// radeonsi's and NVDEC's grain is the specification's (m3-max-77429, m3-max-77421). Grain of
/// other parameters lies past them on every card, each frame (`WRONG_GRAIN`, the same jobs):
/// half as strong, 50 to 52% off in deviation; on luma alone, 100%; strong where it is weak,
/// 23% off and a correlation of -0.87. iHD's grain is the same for the same stream each run
/// (the same numbers in m3-max-77435 and m3-max-77438), so a driver whose grain moves past these
/// bounds fails the test, and is measured again: the bounds are never widened without a
/// measurement.
pub(crate) const GRAIN_MEAN: f64 = 1.0;
/// iHD's 0.0486 (m3-max-77438).
pub(crate) const GRAIN_STD: f64 = 0.049;
/// iHD's 0.8695 (m3-max-77438).
pub(crate) const GRAIN_CORRELATION: f64 = 0.869;

/// The worst of a driver's grain residuals against the specification's, over the frames
/// judged: the largest difference of means, the largest relative difference of standard
/// deviations, and the lowest correlation of block variances.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GrainFit {
    pub(crate) mean: f64,
    pub(crate) std: f64,
    pub(crate) correlation: f64,
}

impl Default for GrainFit {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std: 0.0,
            correlation: 1.0,
        }
    }
}

impl GrainFit {
    fn worst(self, other: Self) -> Self {
        Self {
            mean: self.mean.max(other.mean),
            std: self.std.max(other.std),
            correlation: self.correlation.min(other.correlation),
        }
    }
}

/// Frame `on`'s grain, less frame `off`, against `reference`'s, plane by plane: the planes past
/// the bounds, in words, and the fit.
fn grain_fit(
    reference: &[Residual; 3],
    on: &ReadFrame,
    off: &ReadFrame,
) -> (Vec<String>, GrainFit) {
    let mut past = Vec::new();
    let mut fit = GrainFit::default();
    for ((want, (a, b)), name) in reference
        .iter()
        .zip(on.planes.iter().zip(&off.planes))
        .zip(["Y", "U", "V"])
    {
        let got = Residual::of(a, b);
        let plane = GrainFit {
            mean: (got.mean - want.mean).abs(),
            std: if want.std > 0.0 {
                (got.std / want.std - 1.0).abs()
            } else {
                got.std
            },
            correlation: correlation(&got.blocks, &want.blocks),
        };
        if plane.mean > GRAIN_MEAN
            || plane.std > GRAIN_STD
            || plane.correlation.is_nan()
            || plane.correlation < GRAIN_CORRELATION
        {
            past.push(format!(
                "{name}: mean {:.3} (the specification's {:.3}), deviation {:.3} ({:.3}), block \
                 variances' correlation {:.3}",
                got.mean, want.mean, got.std, want.std, plane.correlation
            ));
        }
        fit = fit.worst(plane);
    }
    (past, fit)
}

/// A frame of a stream a card decoded: read back, or what the hardware reported of a picture it
/// says it failed, which the decoder delivers as no frame (NVDEC's `cuvidDecodeStatus_Error`
/// and `_Error_Concealed`, `nvidia::nvdec::decoder`'s `OnFailed::Refuse`).
pub(crate) type CardFrame = Result<ReadFrame, String>;

/// Judges `stream`'s frames a card decoded, `grained` as the decoder outputs them and, for a
/// stream with film grain, `plain` without it ([`NO_GRAIN`]): a line that says what matched and
/// what the hardware reported failed, or the frames that are wrong, in words.
///
/// Each frame is its reference or one the hardware reported failed; a frame delivered as
/// decoded that is not its reference is wrong, whatever the card. A stream without grain
/// matches its reference frame by frame. A stream with grain matches its reference without
/// grain frame by frame in `plain`, the decode itself, on every card. Each frame in `grained` is
/// then the specification's grain, its reference with grain, or grain of the driver's own
/// synthesis whose residual (`grained` less `plain`, each plane) has the statistics of the
/// specification's ([`grain_refused`]'s bounds): a driver's grain is its own, and iHD's differs
/// from the specification's, as ffmpeg's VA-API decode on iHD gives the same frames
/// (**measured** on the Iris Xe, 2026-11-27, m3-max-77364).
pub(crate) fn judge(
    stream: &Stream,
    grained: &[CardFrame],
    plain: Option<&[CardFrame]>,
) -> Result<String, Vec<String>> {
    let name = stream.name;
    let mut wrong = Vec::new();
    // Each frame against its reference: the frames the hardware reported failed, with the
    // first one's report.
    let mut exact = |frames: &[CardFrame], references: &str, what: &str| {
        let expected = md5s(references);
        if frames.len() != expected.len() {
            wrong.push(format!(
                "{name}: {} frames{what}, not {}",
                frames.len(),
                expected.len()
            ));
        }
        let mut failed = Failed::default();
        for (i, (frame, want)) in frames.iter().zip(&expected).enumerate() {
            match frame {
                Ok(frame) if frame.md5 != *want => {
                    wrong.push(format!("{name} frame {i}{what}: {}, not {want}", frame.md5));
                }
                Ok(_) => {}
                Err(why) => failed.count(why),
            }
        }
        failed
    };
    let Some(without) = stream.md5_without_grain else {
        let failed = exact(grained, stream.md5, "");
        return if wrong.is_empty() {
            Ok(format!("{name}: every frame matches{}", failed.said()))
        } else {
            Err(wrong)
        };
    };
    let plain = plain.expect("the stream's frames decoded without grain");
    let failed_before = exact(plain, without, " before grain");
    let with = md5s(stream.md5);
    if grained.len() != with.len() {
        wrong.push(format!(
            "{name}: {} frames with grain, not {}",
            grained.len(),
            with.len()
        ));
    }
    let reference = Residual::parse(stream.grain.expect("the stream's grain statistics"));
    let (mut specification, mut own, mut judged, mut fit) = (0, 0, 0, GrainFit::default());
    let mut failed = Failed::default();
    for (i, (((frame, before), want), residual)) in grained
        .iter()
        .zip(plain)
        .zip(&with)
        .zip(&reference)
        .enumerate()
    {
        let frame = match frame {
            Ok(frame) if frame.md5 == *want => {
                specification += 1;
                continue;
            }
            Ok(frame) => frame,
            Err(why) => {
                failed.count(why);
                continue;
            }
        };
        let Ok(before) = before else {
            wrong.push(format!(
                "{name} frame {i}'s grain is not the specification's, and the frame before \
                 grain, to judge it by, failed"
            ));
            continue;
        };
        let (past, frame_fit) = grain_fit(residual, frame, before);
        fit = fit.worst(frame_fit);
        judged += 1;
        if past.is_empty() {
            own += 1;
        } else {
            wrong.extend(
                past.into_iter()
                    .map(|why| format!("{name} frame {i}'s grain, {why}")),
            );
        }
    }
    if judged > 0 {
        // Each judgement's worst fit, passed or not: what the bounds are set from.
        eprintln!(
            "{name}: of {judged} frames judged by their grain's statistics, the worst: mean {:.4}, \
             deviation {:.4}, block variance correlation {:.4}",
            fit.mean, fit.std, fit.correlation
        );
    }
    if !wrong.is_empty() {
        return Err(wrong);
    }
    let mut line = format!(
        "{name}: every frame before grain matches{}; with grain, {specification} as the \
         specification's",
        failed_before.said()
    );
    if own > 0 {
        line.push_str(&format!(
            ", {own} of the driver's own synthesis, its residual within {:.3} of the \
             specification's mean, {:.1}% of its deviation, and a block variance correlation \
             of {:.3} at least",
            fit.mean,
            fit.std * 100.0,
            fit.correlation
        ));
    }
    line.push_str(&failed.said());
    Ok(line)
}

/// The frames of a stream the hardware reported failed, and the first one's report.
#[derive(Default)]
struct Failed {
    frames: usize,
    first: Option<String>,
}

impl Failed {
    fn count(&mut self, why: &str) {
        self.frames += 1;
        self.first.get_or_insert_with(|| why.to_owned());
    }

    /// What a judgement's line says of them: nothing where there are none.
    fn said(&self) -> String {
        match &self.first {
            Some(first) => format!(
                ", but for {} the hardware reported failed (the first: {first})",
                self.frames
            ),
            None => String::new(),
        }
    }
}

/// The frames of `grained`, `stream` decoded with grain of altered parameters, that the grain
/// check refuses, of those judged, the frames decoded both with that grain and without
/// (`plain`): an alteration's grain is not the stream's, and each of its frames is to be
/// refused, as neither the specification's nor within [`GRAIN_MEAN`], [`GRAIN_STD`] and
/// [`GRAIN_CORRELATION`] of its statistics. With the worst fit, to say how far off it lies.
pub(crate) fn grain_refused(
    stream: &Stream,
    grained: &[CardFrame],
    plain: &[CardFrame],
) -> (usize, usize, GrainFit) {
    let with = md5s(stream.md5);
    let reference = Residual::parse(stream.grain.expect("the stream's grain statistics"));
    let mut fit = GrainFit::default();
    let (mut refused, mut judged) = (0, 0);
    for (((frame, before), want), residual) in grained.iter().zip(plain).zip(&with).zip(&reference)
    {
        let (Ok(frame), Ok(before)) = (frame, before) else {
            continue;
        };
        judged += 1;
        let (past, frame_fit) = grain_fit(residual, frame, before);
        fit = fit.worst(frame_fit);
        if frame.md5 != *want && !past.is_empty() {
            refused += 1;
        }
    }
    (refused, judged, fit)
}

/// A change of a frame's film grain parameters, which a decoder can be handed in place of them.
pub(crate) type Alter = fn(&mut crate::codec::av1::frame::FilmGrainParams);

/// The alteration that takes a frame's grain away: the decode of a stream's frames before grain.
pub(crate) const NO_GRAIN: Alter = |g| *g = crate::codec::av1::frame::FilmGrainParams::default();

/// Alterations of a stream's film grain that a driver applies as asked and that are not the
/// stream's grain: each the grain check is to refuse ([`grain_refused`]).
pub(crate) const WRONG_GRAIN: [(&str, Alter); 3] = [
    ("half as strong", |g| {
        for s in g
            .point_y_scaling
            .iter_mut()
            .chain(&mut g.point_cb_scaling)
            .chain(&mut g.point_cr_scaling)
        {
            *s /= 2;
        }
    }),
    ("on luma alone", |g| {
        g.num_cb_points = 0;
        g.num_cr_points = 0;
        g.chroma_scaling_from_luma = false;
    }),
    ("strong where it is weak", |g| {
        g.point_y_scaling[..usize::from(g.num_y_points)].reverse();
        g.point_cb_scaling[..usize::from(g.num_cb_points)].reverse();
        g.point_cr_scaling[..usize::from(g.num_cr_points)].reverse();
    }),
];
