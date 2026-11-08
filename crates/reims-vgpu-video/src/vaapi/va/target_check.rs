//! Whether the driver decodes right into targets over the caller's memory (`super::target`), and
//! keeps each picture's references apart from those targets, learned once a GPU and driver by a
//! check that decodes streams of our own into them ([`TargetChecks::of`]), so that what a caller
//! hands a driver follows what the driver does, not its name.
//!
//! A decode into a guest's output can only be exact on a driver that reads no reference from its
//! render target: the output holds the picture alone, and the guest may write it or have it
//! decoded into again while the stream still predicts from the picture (the wire's
//! `DECODE_H264`). So for each codec and format the driver decodes (H.264 into NV12, HEVC Main
//! into NV12, HEVC Main 10 into P010), the check decodes one of its streams
//! (`target_check::streams`: an IDR picture of PCM, then pictures that copy the one before) into
//! targets over memory of its own, laid out as the caller's must be (below), and checks every
//! byte of that memory after each picture:
//! 1. The IDR picture into target A: each plane's samples, and every byte around them left as it
//!    was, but the bytes past each row's samples to its pitch, which a decode may write.
//! 2. A's samples overwritten, the next picture into target B: still exact, so the driver read
//!    its reference from a buffer of its own, not from A.
//! 3. The next into a new import of B's memory while B's picture is that one's reference: exact,
//!    so a picture decoded over its own reference's memory, as a guest's reuse of an output
//!    has it, reads its reference apart from it too.
//! 4. For H.264, the last picture after a gap in `frame_num`, which the decoder stands in for with
//!    a copy of the picture before, from its target (`Surface::copy_to`, by the driver's video
//!    processing), into target A, which the decoder let go of: exact.
//!
//! The memory is laid out with each plane's rows a multiple of 256 bytes apart (the least pitch
//! of 256 bytes at or past the rows' samples), the luma plane 256 bytes into the dma-buf, and the
//! chroma plane 256 bytes past the next multiple of 256 after the luma's rows, at the coded size
//! a picture is written in, the last row of the HEVC stream's coding tree blocks half in the
//! picture. The gap before the chroma plane has each target imported taller than the picture, its
//! luma plane declared over the gap and into the chroma plane (`super::target`, "The import"), as
//! a caller's outputs with CoreVideo's gap between their planes are. A caller imports only memory
//! of that kind: rows and planes on multiples of 256 bytes of the dma-buf ([`ALIGNMENT`]), each
//! plane's rows holding the coded picture's.
//!
//! **The answer is kept** in one file the caller names, under a key of the driver's vendor string,
//! the render node's PCI IDs, libva's version and the kernel's release, so only the first display
//! a driver opens after one of those changes runs the check; where the file is missing,
//! unreadable or of another key, the check runs and the file is written anew, through a file
//! beside it renamed over it. Until a check passes, nothing is imported.

mod streams;

use std::fmt::Write as _;
use std::path::Path;

use super::display::Display;
use super::frame::Frame;
use super::host_memory::HostMemory;
use super::target::{Target, TargetFormat, TargetPlane, TargetShape};
use super::{Entrypoint, H264Decoder, HevcDecoder, Profile};

/// What a caller's target memory is aligned to (module docs): the bytes from one row of a plane to
/// the next, and each plane's first byte from the dma-buf's, are multiples of it.
pub const ALIGNMENT: u64 = 256;

/// The byte the check's memory holds where no sample is.
const UNWRITTEN: u8 = 0xa5;

/// The byte the check writes over a picture's samples once they are checked.
const SCRIBBLE: u8 = 0x5a;

/// The check's pictures' sizes: the H.264 stream's 15 by 9 macroblocks, and the HEVC stream's
/// 15 coding tree blocks by 8.5.
const H264_SIZE: (u32, u32) = (240, 144);
const HEVC_SIZE: (u32, u32) = (240, 136);

/// The file's first line: its format.
const HEADER: &str = "reims-vgpu va targets 1";

/// A codec the check decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetCodec {
    /// H.264, into NV12.
    H264,
    /// HEVC, Main into NV12 and Main 10 into P010.
    Hevc,
}

/// The check's answers for a driver ([`TargetChecks::of`]): for each codec and format, whether its
/// decodes may be handed targets, or why not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetChecks {
    answers: Vec<(TargetCodec, TargetFormat, Result<(), String>)>,
}

/// The codecs and formats the check runs, with their names in the file.
const KINDS: [(TargetCodec, TargetFormat, &str); 3] = [
    (TargetCodec::H264, TargetFormat::Nv12, "h264-nv12"),
    (TargetCodec::Hevc, TargetFormat::Nv12, "hevc-nv12"),
    (TargetCodec::Hevc, TargetFormat::P010, "hevc-p010"),
];

impl TargetChecks {
    /// The answers for `display`'s driver: kept in `file` under its key where they are, else
    /// checked now and kept there (module docs); with no file, checked now.
    pub fn of(display: &Display, file: Option<&Path>) -> TargetChecks {
        let key = key(display);
        if let Some(kept) = file.and_then(|file| read(file, &key)) {
            return kept;
        }
        let started = display.runtime().now();
        let checks = TargetChecks::check(display);
        log::info!(
            "hardware video: VA-API targets checked in {:.1} ms: {}",
            display.runtime().elapsed_since(started).as_secs_f64() * 1e3,
            checks.summary()
        );
        if let Some(file) = file
            && let Err(e) = write(file, &key, &checks)
        {
            log::info!(
                "hardware video: the VA-API targets' check was not kept at {} ({e}): the next \
                 display checks again",
                file.display()
            );
        }
        checks
    }

    /// The answers of a check run now on `display`.
    pub(crate) fn check(display: &Display) -> TargetChecks {
        let answers = KINDS
            .iter()
            .map(|&(codec, format, _)| (codec, format, check_kind(display, codec, format)))
            .collect();
        TargetChecks { answers }
    }

    /// Whether decodes of `codec` into `format` may be handed targets: `Err` with why not.
    pub fn passed(&self, codec: TargetCodec, format: TargetFormat) -> Result<(), String> {
        (self.answers.iter())
            .find(|(c, f, _)| *c == codec && *f == format)
            .map_or_else(|| Err("not checked".to_owned()), |(_, _, a)| a.clone())
    }

    /// The answers in one line, for the log.
    pub fn summary(&self) -> String {
        let mut line = String::new();
        for (codec, format, answer) in &self.answers {
            let name = name_of(*codec, *format);
            let _ = match answer {
                Ok(()) => write!(line, "{name} passed; "),
                Err(why) => write!(line, "{name} not taken: {why}; "),
            };
        }
        line.trim_end_matches("; ").to_owned()
    }
}

fn name_of(codec: TargetCodec, format: TargetFormat) -> &'static str {
    (KINDS.iter())
        .find(|(c, f, _)| *c == codec && *f == format)
        .map_or("?", |(_, _, name)| name)
}

/// The key the answers are kept under (module docs).
fn key(display: &Display) -> String {
    let node = Path::new(display.path());
    let sysfs = |field: &str| {
        let name = node.file_name()?.to_str()?;
        let path = format!("/sys/class/drm/{name}/device/{field}");
        Some(std::fs::read_to_string(path).ok()?.trim().to_owned())
    };
    let pci = ["vendor", "device", "revision"]
        .map(|field| sysfs(field).unwrap_or_else(|| "?".to_owned()))
        .join(":");
    let kernel = std::fs::read_to_string("/proc/sys/kernel/osrelease")
        .map_or_else(|_| "?".to_owned(), |r| r.trim().to_owned());
    let (major, minor) = display.api_version();
    // One line: the vendor string's own breaks are spaces here.
    let vendor = display.vendor().replace(['\n', '\r'], " ");
    format!("{vendor} | pci {pci} | libva {major}.{minor} | kernel {kernel}")
}

/// The answers `file` keeps under `key`; `None` where it is missing, unreadable, of another format
/// or key, or does not answer every kind.
fn read(file: &Path, key: &str) -> Option<TargetChecks> {
    let text = std::fs::read_to_string(file).ok()?;
    let mut lines = text.lines();
    if lines.next()? != HEADER || lines.next()?.strip_prefix("key ")? != key {
        return None;
    }
    let mut answers = Vec::new();
    for line in lines {
        let (name, answer) = line.split_once(' ')?;
        let &(codec, format, _) = KINDS.iter().find(|(_, _, n)| *n == name)?;
        let answer = match answer.strip_prefix("failed ") {
            Some(why) => Err(why.to_owned()),
            None if answer == "passed" => Ok(()),
            None => return None,
        };
        answers.push((codec, format, answer));
    }
    (answers.len() == KINDS.len()).then_some(TargetChecks { answers })
}

/// Keeps `checks` in `file` under `key`, through a file beside it renamed over it.
fn write(file: &Path, key: &str, checks: &TargetChecks) -> std::io::Result<()> {
    let mut text = format!("{HEADER}\nkey {key}\n");
    for (codec, format, answer) in &checks.answers {
        let name = name_of(*codec, *format);
        let _ = match answer {
            Ok(()) => writeln!(text, "{name} passed"),
            Err(why) => writeln!(text, "{name} failed {}", why.replace(['\n', '\r'], " ")),
        };
    }
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut beside = file.as_os_str().to_owned();
    beside.push(format!(".{}", std::process::id()));
    std::fs::write(&beside, text)?;
    std::fs::rename(&beside, file)
}

/// Memory of the check's own for a target of `shape`, laid out as the module docs say, every byte
/// [`UNWRITTEN`].
struct Laid {
    memory: HostMemory,
    planes: [TargetPlane; 2],
    shape: TargetShape,
}

impl Laid {
    fn new(shape: TargetShape) -> Result<Laid, String> {
        let (width, height) = shape.coded;
        let row = u64::from(width) * shape.format.bytes_per_sample();
        let pitch = row.next_multiple_of(ALIGNMENT);
        let luma = TargetPlane {
            offset: ALIGNMENT,
            bytes_per_row: pitch,
        };
        let luma_end = luma.offset + pitch * u64::from(height);
        let chroma = TargetPlane {
            offset: luma_end.next_multiple_of(ALIGNMENT) + ALIGNMENT,
            bytes_per_row: pitch,
        };
        let end = chroma.offset + pitch * u64::from(height.div_ceil(2)) + 2 * 4096;
        let len = usize::try_from(end).map_err(|_| "a check's memory past the address space")?;
        let mut memory = HostMemory::new(len).map_err(|e| format!("its memory: {e}"))?;
        memory.bytes_mut().fill(UNWRITTEN);
        Ok(Laid {
            memory,
            planes: [luma, chroma],
            shape,
        })
    }

    /// A new import of the memory as a target on `display`.
    fn import(&self, display: &Display) -> Result<Target, String> {
        let size = u64::try_from(self.memory.len()).unwrap_or(u64::MAX);
        Target::import(
            display,
            self.memory.dma_buf_fd(),
            size,
            self.shape,
            self.planes,
            Box::new(()),
        )
        .map_err(|e| format!("the import: {e}"))
    }

    /// Each plane's rows of samples: its first byte, its rows, and the bytes of each.
    fn rows(&self) -> [(usize, usize, usize, usize); 2] {
        let (width, height) = self.shape.coded;
        let row = usize::try_from(u64::from(width) * self.shape.format.bytes_per_sample())
            .unwrap_or(usize::MAX);
        let heights = [height, height.div_ceil(2)];
        [0, 1].map(|plane| {
            let laid = self.planes[plane];
            (
                usize::try_from(laid.offset).unwrap_or(usize::MAX),
                usize::try_from(heights[plane]).unwrap_or(usize::MAX),
                row,
                usize::try_from(laid.bytes_per_row).unwrap_or(usize::MAX),
            )
        })
    }

    /// `Err` with the first byte that is wrong: a sample that is not the stream's, or a byte
    /// outside the planes' rows of samples and their pitches' padding that is not [`UNWRITTEN`].
    fn verify(&self, what: &str, bit_depth: u32) -> Result<(), String> {
        let bytes = self.memory.bytes();
        let mut expected = vec![UNWRITTEN; bytes.len()];
        let mut padding = vec![false; bytes.len()];
        for (plane, (first, rows, row, pitch)) in self.rows().into_iter().enumerate() {
            for y in 0..rows {
                let start = first + y * pitch;
                padding[start + row..start + pitch].fill(true);
                let line = &mut expected[start..start + row];
                write_samples(line, plane, y, bit_depth, self.shape.format);
            }
        }
        let wrong = (0..bytes.len()).find(|&i| !padding[i] && bytes[i] != expected[i]);
        let Some(at) = wrong else {
            return Ok(());
        };
        // Where each plane's first row's samples are in the memory, if anywhere: the layout the
        // driver wrote, for the log.
        let found = self.rows().map(|(first, _, row, _)| {
            let want = &expected[first..first + row.min(32)];
            (bytes.windows(want.len()))
                .position(|w| w == want)
                .map_or_else(|| "nowhere".to_owned(), |p| format!("at byte {p}"))
        });
        Err(format!(
            "{what}: byte {at} of the memory is {:#04x}, not {:#04x} ({}); the luma's first row \
             is {}, the chroma's {}",
            bytes[at],
            expected[at],
            self.place(at),
            found[0],
            found[1]
        ))
    }

    /// Where byte `at` of the memory lies, for the log.
    fn place(&self, at: usize) -> String {
        for (plane, (first, rows, row, pitch)) in self.rows().into_iter().enumerate() {
            let name = ["luma", "chroma"][plane];
            if at >= first && at < first + rows * pitch {
                let (y, x) = ((at - first) / pitch, (at - first) % pitch);
                return match x < row {
                    true => format!("{name} row {y}, byte {x}"),
                    false => format!("{name} row {y}, padding"),
                };
            }
        }
        "outside the planes".to_owned()
    }

    /// Every sample of the planes' rows overwritten with [`SCRIBBLE`].
    fn scribble(&mut self) {
        let rows = self.rows();
        let bytes = self.memory.bytes_mut();
        for (first, count, row, pitch) in rows {
            for y in 0..count {
                let start = first + y * pitch;
                bytes[start..start + row].fill(SCRIBBLE);
            }
        }
    }
}

/// Row `y` of `plane` (0 luma, 1 the interleaved chroma) of the check's pictures, as `format` lays
/// it in memory.
fn write_samples(line: &mut [u8], plane: usize, y: usize, bit_depth: u32, format: TargetFormat) {
    let y = u32::try_from(y).unwrap_or(u32::MAX);
    let bytes = usize::try_from(format.bytes_per_sample()).unwrap_or(1);
    for (i, out) in line.chunks_exact_mut(bytes).enumerate() {
        let i = u32::try_from(i).unwrap_or(u32::MAX);
        let value = match plane {
            0 => streams::sample(0, i, y, bit_depth),
            _ => streams::sample(1 + usize::from(i % 2 == 1), i / 2, y, bit_depth),
        };
        match format {
            TargetFormat::Nv12 => out[0] = u8::try_from(value).unwrap_or(u8::MAX),
            // P010: the sample in the high ten bits of a little-endian word.
            TargetFormat::P010 => out.copy_from_slice(&(value << 6).to_le_bytes()),
        }
    }
}

/// A decoder of the check's codec.
enum Decoder {
    H264(H264Decoder),
    Hevc(HevcDecoder),
}

impl Decoder {
    /// Decodes `unit` into `target`, and answers its one frame, decoded, which must be in it.
    fn decode(&mut self, unit: &[u8], id: u64, target: Target) -> Result<Frame, String> {
        let shape = target.shape();
        let target_for: super::target::TargetFor =
            Box::new(move |asked| (asked == shape).then_some(target));
        let decoded = match self {
            Decoder::H264(decoder) => decoder.decode_into(unit, id, Some(target_for)),
            Decoder::Hevc(decoder) => decoder.decode_into(unit, id, Some(target_for)),
        };
        let mut frames = decoded.map_err(|e| format!("picture {id}: {e}"))?;
        let frame = match (frames.pop(), frames.is_empty()) {
            (Some(frame), true) => frame,
            _ => return Err(format!("picture {id}: not one frame answered")),
        };
        if !frame.is_in_target() {
            return Err(format!("picture {id}: not decoded into its target"));
        }
        frame.sync().map_err(|e| format!("picture {id}: {e}"))?;
        Ok(frame)
    }
}

/// The check of `codec` into `format` (module docs) on `display`: `Err` with why it failed, or
/// why it was not run.
fn check_kind(display: &Display, codec: TargetCodec, format: TargetFormat) -> Result<(), String> {
    let profile = match (codec, format) {
        (TargetCodec::H264, TargetFormat::Nv12) => Profile::H264_CONSTRAINED_BASELINE,
        (TargetCodec::Hevc, TargetFormat::Nv12) => Profile::HEVC_MAIN,
        (TargetCodec::Hevc, TargetFormat::P010) => Profile::HEVC_MAIN10,
        (TargetCodec::H264, TargetFormat::P010) => return Err("no such decode".to_owned()),
    };
    let decodes = [profile, Profile::H264_MAIN, Profile::H264_HIGH]
        .iter()
        .take(if codec == TargetCodec::H264 { 3 } else { 1 })
        .any(|&p| display.supports(p, Entrypoint::VLD));
    if !decodes {
        return Err("the driver does not decode it".to_owned());
    }
    let (stream, mut decoder) = match codec {
        TargetCodec::H264 => (
            streams::h264(H264_SIZE.0, H264_SIZE.1),
            Decoder::H264(H264Decoder::new_annex_b(display)),
        ),
        TargetCodec::Hevc => (
            streams::hevc(
                HEVC_SIZE.0,
                HEVC_SIZE.1,
                match format {
                    TargetFormat::Nv12 => 8,
                    TargetFormat::P010 => 10,
                },
            ),
            Decoder::Hevc(HevcDecoder::new_annex_b(display)),
        ),
    };
    let shape = TargetShape {
        format,
        coded: (stream.width, stream.height),
    };
    let depth = stream.bit_depth;
    let mut a = Laid::new(shape)?;
    let b = Laid::new(shape)?;
    let [first, second, third, rest @ ..] = stream.units.as_slice() else {
        return Err("a stream of fewer than three pictures".to_owned());
    };
    // 1. The IDR picture into A.
    let target_a = a.import(display)?;
    let frame = decoder.decode(first, 0, target_a.clone())?;
    a.verify("the first picture", depth)?;
    drop(frame);
    // 2. The next into B, A overwritten.
    a.scribble();
    let frame = decoder.decode(second, 1, b.import(display)?)?;
    b.verify("a picture whose reference's target was overwritten", depth)?;
    drop(frame);
    // 3. The next into a new import of B's memory, B's picture its reference.
    let frame = decoder.decode(third, 2, b.import(display)?)?;
    b.verify("a picture over its own reference's memory", depth)?;
    drop(frame);
    // 4. H.264's picture after its gap, into A again, its stand-in copied from B's memory.
    if let Some(last) = rest.first() {
        if target_a.is_held() {
            return Err("the decoder still holds the first picture's target".to_owned());
        }
        let frame = decoder.decode(last, 3, target_a)?;
        a.verify("a picture after a gap, into a target let go of", depth)?;
        drop(frame);
    }
    // The decoder goes before the memory under its targets.
    drop(decoder);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode::{EncoderSettings, InputFrame, Plane, Profile as Encoded};
    use crate::core::runtime::StdRuntime;
    use crate::vaapi::va::encoder::VideoEncoder;
    use crate::vaapi::va::frame::PlaneRows;

    /// The encoded streams' size: rows of a multiple of 256 bytes.
    const SIZE: (u32, u32) = (1280, 720);
    const PICTURES: u64 = 12;

    fn display() -> Display {
        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        Display::open(node, StdRuntime::shared()).expect("a VA display")
    }

    /// `PICTURES` pictures of a gradient moving a few samples a picture, by the driver's encoder
    /// of `profile` with no B pictures: its configuration record and samples; `None` where the
    /// driver encodes no such stream.
    fn encoded(display: &Display, profile: Encoded) -> Option<(Vec<u8>, Vec<Vec<u8>>)> {
        let mut settings = EncoderSettings::new(profile, SIZE.0, SIZE.1);
        settings.allow_frame_reordering = false;
        let mut encoder = VideoEncoder::new(display, settings).ok()?;
        let (w, h) = (SIZE.0 as usize, SIZE.1 as usize);
        let mut samples = Vec::new();
        for n in 0..PICTURES {
            let shift = n as usize * 3;
            let mut bytes = vec![0u8; w * h * 3 / 2];
            for (i, b) in bytes.iter_mut().enumerate() {
                let (x, y) = (i % w, i / w);
                *b = ((x + shift) * 3 + y * 5 + (x * y) % 7) as u8;
            }
            let (luma, chroma) = bytes.split_at(w * h);
            let frame = InputFrame {
                planes: [
                    Plane {
                        data: luma,
                        bytes_per_row: w,
                    },
                    Plane {
                        data: chroma,
                        bytes_per_row: w,
                    },
                ],
                pts: n as i64 * 3000,
                duration: 3000,
                force_key_frame: false,
                id: n,
            };
            let coded = encoder.encode(&frame).expect("a frame taken");
            samples.extend(coded.into_iter().map(|c| c.expect("a sample").data));
        }
        samples.extend(
            encoder
                .flush()
                .into_iter()
                .map(|c| c.expect("a sample").data),
        );
        Some((encoder.configuration().to_vec(), samples))
    }

    fn decoder(display: &Display, codec: TargetCodec, record: &[u8]) -> Decoder {
        match codec {
            TargetCodec::H264 => Decoder::H264(H264Decoder::new(display, record).expect("one")),
            TargetCodec::Hevc => Decoder::Hevc(HevcDecoder::new(display, record).expect("one")),
        }
    }

    /// Each picture's crop, its luma rows then its chroma rows, as the CPU's copy reads it back
    /// from the decoder's own surfaces.
    fn read_back(
        display: &Display,
        codec: TargetCodec,
        record: &[u8],
        samples: &[Vec<u8>],
    ) -> Vec<Vec<u8>> {
        let mut decoder = decoder(display, codec, record);
        let (w, h) = (SIZE.0 as usize, SIZE.1 as usize);
        let mut out = Vec::new();
        for (n, sample) in samples.iter().enumerate() {
            let frames = match &mut decoder {
                Decoder::H264(d) => d.decode(sample, n as u64),
                Decoder::Hevc(d) => d.decode(sample, n as u64),
            }
            .expect("a decode");
            for frame in frames {
                let mut bytes = vec![0u8; w * h * 3 / 2];
                let (luma, chroma) = bytes.split_at_mut(w * h);
                let rows = |bytes, rows| PlaneRows {
                    bytes,
                    bytes_per_row: w,
                    rows,
                    row_bytes: w,
                };
                (frame.read_back().expect("a read back"))
                    .copy_into(rows(luma, h), rows(chroma, h / 2))
                    .expect("a copy");
                out.push(bytes);
            }
        }
        out
    }

    /// Each picture's crop as decoded into targets over three memories in turn, each a new import,
    /// each scribbled over once read, while the decoder may still reference its picture.
    fn into_targets(
        display: &Display,
        codec: TargetCodec,
        record: &[u8],
        samples: &[Vec<u8>],
    ) -> Vec<Vec<u8>> {
        let mut decoder = decoder(display, codec, record);
        let shape = TargetShape {
            format: TargetFormat::Nv12,
            coded: SIZE,
        };
        let mut memories: Vec<Laid> = (0..3).map(|_| Laid::new(shape).expect("memory")).collect();
        let mut out = Vec::new();
        for (n, sample) in samples.iter().enumerate() {
            let laid = &mut memories[n % 3];
            let frame = decoder
                .decode(sample, n as u64, laid.import(display).expect("an import"))
                .expect("a decode into a target");
            let mut bytes = Vec::new();
            for (first, rows, row, pitch) in laid.rows() {
                for y in 0..rows {
                    let start = first + y * pitch;
                    bytes.extend_from_slice(&laid.memory.bytes()[start..start + row]);
                }
            }
            out.push(bytes);
            drop(frame);
            laid.scribble();
        }
        drop(decoder);
        out
    }

    /// Where the check passes a codec, a stream of the driver's own encoder (with motion and
    /// residuals, which the check's streams have none of) decodes into targets, each overwritten
    /// once read, exactly as into the decoder's own surfaces. Prints the check's answers.
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn where_the_check_passes_an_encoded_stream_decodes_into_targets_exactly() {
        let display = display();
        let checks = TargetChecks::check(&display);
        eprintln!("the check: {}", checks.summary());
        for (codec, profile) in [
            (TargetCodec::H264, Encoded::H264High),
            (TargetCodec::Hevc, Encoded::HevcMain),
        ] {
            if let Err(why) = checks.passed(codec, TargetFormat::Nv12) {
                eprintln!("{codec:?}: not compared: the check answered {why}");
                continue;
            }
            let Some((record, samples)) = encoded(&display, profile) else {
                eprintln!("{codec:?}: not compared: the driver encodes no {profile:?}");
                continue;
            };
            let reference = read_back(&display, codec, &record, &samples);
            let targets = into_targets(&display, codec, &record, &samples);
            assert_eq!(reference.len(), targets.len(), "{codec:?}: the pictures");
            for (n, (r, t)) in reference.iter().zip(&targets).enumerate() {
                assert!(
                    r == t,
                    "{codec:?}: picture {n} differs from the decoder's own surface's"
                );
            }
            eprintln!(
                "{codec:?}: {} pictures into targets as into the decoder's own",
                targets.len()
            );
        }
    }

    /// The check's streams decode to their samples on the driver's own surfaces, read back by the
    /// CPU's copy: where they do not, a check's failure is the stream's, not the target's. Prints
    /// each picture's first wrong sample.
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn the_checks_streams_decode_to_their_samples_on_the_drivers_own_surfaces() {
        let display = display();
        let mut wrong = Vec::new();
        for (codec, format, name) in KINDS {
            let (stream, mut decoder) = match codec {
                TargetCodec::H264 => (
                    streams::h264(H264_SIZE.0, H264_SIZE.1),
                    Decoder::H264(H264Decoder::new_annex_b(&display)),
                ),
                TargetCodec::Hevc => (
                    streams::hevc(
                        HEVC_SIZE.0,
                        HEVC_SIZE.1,
                        if format == TargetFormat::P010 { 10 } else { 8 },
                    ),
                    Decoder::Hevc(HevcDecoder::new_annex_b(&display)),
                ),
            };
            let (w, h) = (stream.width as usize, stream.height as usize);
            let bytes = format.bytes_per_sample() as usize;
            for (n, unit) in stream.units.iter().enumerate() {
                let frames = match &mut decoder {
                    Decoder::H264(d) => d.decode(unit, n as u64),
                    Decoder::Hevc(d) => d.decode(unit, n as u64),
                };
                let frames = match frames {
                    Ok(frames) => frames,
                    Err(e) => {
                        wrong.push(format!("{name} picture {n}: {e}"));
                        continue;
                    }
                };
                for frame in frames {
                    let row = w * bytes;
                    let mut luma = vec![0u8; row * h];
                    let mut chroma = vec![0u8; row * h.div_ceil(2)];
                    let rows = |bytes, rows| PlaneRows {
                        bytes,
                        bytes_per_row: row,
                        rows,
                        row_bytes: row,
                    };
                    (frame.read_back().expect("a read back"))
                        .copy_into(rows(&mut luma, h), rows(&mut chroma, h.div_ceil(2)))
                        .expect("a copy");
                    let mut first = None;
                    for (plane, data, count) in [(0, &luma, h), (1, &chroma, h.div_ceil(2))] {
                        for y in 0..count {
                            let mut expected = vec![0u8; row];
                            write_samples(&mut expected, plane, y, stream.bit_depth, format);
                            let got = &data[y * row..(y + 1) * row];
                            if let Some(x) = (0..row).find(|&x| got[x] != expected[x]) {
                                first.get_or_insert(format!(
                                    "{name} picture {n}: plane {plane} row {y} byte {x} is {:#04x}, not {:#04x}",
                                    got[x], expected[x]
                                ));
                            }
                        }
                    }
                    match first {
                        Some(why) => wrong.push(why),
                        None => eprintln!("{name} picture {n}: every sample right"),
                    }
                }
            }
        }
        for why in &wrong {
            eprintln!("{why}");
        }
        assert!(wrong.is_empty(), "{} pictures wrong", wrong.len());
    }

    /// The check runs to an answer for each codec and format on the GPU: the driver may refuse
    /// any, but each of the check's streams decodes one picture an access unit. Prints the
    /// answers.
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn the_check_runs_on_the_gpu() {
        let display = display();
        let checks = TargetChecks::check(&display);
        eprintln!("the check: {}", checks.summary());
        for (codec, format, _) in KINDS {
            if let Err(why) = checks.passed(codec, format) {
                // A refusal by the driver is an answer; a failure of the check's own is not.
                assert!(
                    !why.contains("not one frame") && !why.contains("fewer than"),
                    "{why}"
                );
            }
        }
    }

    #[test]
    fn answers_are_kept_and_read_back_under_their_key() {
        let dir = std::env::temp_dir().join(format!("rvg-target-check-{}", std::process::id()));
        let file = dir.join("va-targets");
        let checks = TargetChecks {
            answers: vec![
                (TargetCodec::H264, TargetFormat::Nv12, Ok(())),
                (
                    TargetCodec::Hevc,
                    TargetFormat::Nv12,
                    Err("a byte\nwrong".into()),
                ),
                (TargetCodec::Hevc, TargetFormat::P010, Ok(())),
            ],
        };
        write(&file, "a key", &checks).expect("written");
        let read_back = read(&file, "a key").expect("read back");
        assert_eq!(
            read_back.passed(TargetCodec::H264, TargetFormat::Nv12),
            Ok(())
        );
        assert_eq!(
            read_back.passed(TargetCodec::Hevc, TargetFormat::Nv12),
            Err("a byte wrong".into())
        );
        assert!(read(&file, "another key").is_none());
        std::fs::remove_dir_all(&dir).expect("removed");
    }
}
