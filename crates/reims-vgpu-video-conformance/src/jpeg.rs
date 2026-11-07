//! The JPEG engine's decoder against planes a reference decoder wrote, on
//! any backend's `jpeg` part.
//!
//! ```sh
//! conformance NODE jpeg-check [OPTION]... FILE.jpg
//! ```
//!
//! It opens the whole backend on `NODE`, as the glue does, and decodes
//! `FILE.jpg` through its `jpeg` part, core's
//! `JpegDecoder`, with the request core would make of CMPhoto's options
//! for it: the stream's size and subsampling, the whole image or a crop,
//! at a scale. The destination is the layout `--format` names, rows 64-byte
//! aligned as an IOSurface's are. Then it reads the layout back into
//! planes and compares them with each reference.
//!
//! A decode that copies the stream's samples, at scale 1 into a layout of
//! its sampling, is compared sample for sample, within the tolerance. One
//! that scales, resamples the chroma or converts to BGRA is compared with
//! core's scaler reference (`reims_vgpu_video_core::scaler::reference`)
//! run on the reference's planes, by PSNR per plane, as the scaler's check
//! is: it passes at 34 dB for a conversion, 30 dB for a scale, and 28 dB
//! where the output's size is odd.
//!
//! A stream outside what the backend answers it decodes
//! (`JpegDecoder::decodes`: its samplings, and its largest picture) is to
//! be refused as `UnsupportedStream`: that refusal passes, as answered,
//! where a decode was asked for, and a decode of it fails. Inside the
//! answer, every refusal fails.
//!
//! - `--format F`: `420f`, `422f`, `444f`, `yuvs`, `2vuy` or `BGRA`; the
//!   stream's own (and `420f` for grey) by default.
//! - `--crop X,Y,W,H`: a rectangle of MCUs, as `CropRect` gives it.
//! - `--scale N`: 1, 2, 4 or 8: the region at 1/N, rounded up.
//! - `--dump PREFIX`: for a decode compared by PSNR, writes our planes to
//!   `PREFIX.ours` and the reference's to `PREFIX.NAME`, one after another
//!   in the order the compare line names them.
//! - `--reference NAME=PATH`: planar Y, Cb and Cr at the stream's
//!   sampling, as libjpeg's raw output (`tools/jpeg-planes.c`) and
//!   ffmpeg's `yuvj4xxp` rawvideo have them. Repeatable; a decode that
//!   should succeed fails with none, having been compared with nothing.
//! - `--tolerance N`: the largest difference from a reference that passes.
//! - `--strip-dht`: removes the stream's DHT segments first, so that it
//!   decodes with Annex K's tables, as a Motion JPEG frame does.
//! - `--annex-k`: checks that the stream's Huffman tables are Annex K's,
//!   as cjpeg writes them without `-optimize`.
//! - `--expect-error KIND`: passes only if the decode fails with core's
//!   `JpegError` of that kind (`BadData`, `UnsupportedStream`,
//!   `UnsupportedFormat`, `UnsupportedScale`, ...).
//! - `--repeat N`: decodes N times and reports the mean time.
//!
//! Lines start with a word to grep: `backend`, `stream`, `decode`,
//! `digest`, `compare`, `tables`, `engines`, `result`, and `log` for the
//! library's log lines at info and above. The digest line is the MD5 of
//! every byte of the destination's planes, their padding included, as the
//! last decode left them, so that two runs can be shown to have written the
//! same bytes. The engines line is the
//! kernel's busy time for this process's render-node descriptors
//! (`/proc/self/fdinfo`, amdgpu's `drm-engine-*`, where the VCN's JPEG
//! decoder is `jpeg`), empty off Linux.

pub use check::run_args;

mod check {
    use std::process::ExitCode;

    use reims_vgpu_video_codec::jpeg::{self, HuffmanTable};
    use reims_vgpu_video_core::fourcc::FourCc;
    use reims_vgpu_video_core::jpeg::{
        JpegDecode, JpegError, JpegOutputFormat, JpegSubsampling, PixelRect,
    };
    use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
    use reims_vgpu_video_core::scaler::plan::{Filter, Matrix, Orientation};
    use reims_vgpu_video_core::scaler::reference::{Picture, resample, to_rgb};
    use reims_vgpu_video_core::surfaces::PlaneDestination;

    use crate::cli::Open;

    struct Options {
        format: Option<String>,
        crop: Option<[u32; 4]>,
        scale: u32,
        dump: Option<String>,
        references: Vec<(String, String)>,
        tolerance: u8,
        strip_dht: bool,
        annex_k: bool,
        expect_error: Option<String>,
        repeat: u32,
        node: String,
        file: String,
    }

    fn options(args: &[String]) -> Result<Options, String> {
        let mut args = args.iter().cloned();
        let mut options = Options {
            format: None,
            crop: None,
            scale: 1,
            dump: None,
            references: Vec::new(),
            tolerance: 0,
            strip_dht: false,
            annex_k: false,
            expect_error: None,
            repeat: 1,
            node: String::new(),
            file: String::new(),
        };
        let mut positional = Vec::new();
        while let Some(arg) = args.next() {
            let mut value = || args.next().ok_or(format!("{arg} needs a value"));
            match arg.as_str() {
                "--format" => options.format = Some(value()?),
                "--crop" => {
                    let parts: Vec<u32> = value()?
                        .split(',')
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .map_err(|e| format!("--crop: {e}"))?;
                    let [x, y, w, h] = parts[..] else {
                        return Err("--crop takes X,Y,W,H".into());
                    };
                    options.crop = Some([x, y, w, h]);
                }
                "--reference" => {
                    let v = value()?;
                    let (name, path) = v.split_once('=').ok_or("--reference takes NAME=PATH")?;
                    options.references.push((name.into(), path.into()));
                }
                "--scale" => {
                    options.scale = value()?.parse().map_err(|e| format!("--scale: {e}"))?;
                    if ![1, 2, 4, 8].contains(&options.scale) {
                        return Err("--scale takes 1, 2, 4 or 8".into());
                    }
                }
                "--dump" => options.dump = Some(value()?),
                "--tolerance" => {
                    options.tolerance =
                        value()?.parse().map_err(|e| format!("--tolerance: {e}"))?;
                }
                "--strip-dht" => options.strip_dht = true,
                "--annex-k" => options.annex_k = true,
                "--expect-error" => options.expect_error = Some(value()?),
                "--repeat" => {
                    options.repeat = value()?.parse().map_err(|e| format!("--repeat: {e}"))?;
                }
                _ => positional.push(arg),
            }
        }
        let [node, file] = <[String; 2]>::try_from(positional)
            .map_err(|_| "usage: conformance NODE jpeg-check [OPTION]... FILE.jpg".to_owned())?;
        options.node = node;
        options.file = file;
        Ok(options)
    }

    /// `jpeg-check`'s arguments (module doc), its NODE before FILE.jpg, on
    /// the backend `open` makes of NODE.
    pub fn run_args(args: &[String], open: Open<'_>) -> ExitCode {
        match run(args, open) {
            Ok(true) => {
                println!("result pass");
                ExitCode::SUCCESS
            }
            Ok(false) => {
                println!("result fail");
                ExitCode::FAILURE
            }
            Err(error) => {
                println!("result error: {error}");
                ExitCode::FAILURE
            }
        }
    }

    /// The library's log lines, on stdout.
    struct Logger;

    impl log::Log for Logger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Info
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                println!("log {} {}", record.level(), record.args());
            }
        }

        fn flush(&self) {}
    }

    fn run(args: &[String], open: Open<'_>) -> Result<bool, String> {
        let options = options(args)?;
        if log::set_logger(&Logger).is_ok() {
            log::set_max_level(log::LevelFilter::Info);
        }
        let mut data =
            std::fs::read(&options.file).map_err(|e| format!("{}: {e}", options.file))?;
        if options.strip_dht {
            data = strip_dht(&data)?;
        }
        let backend = open(&options.node).map_err(|e| format!("{}: {e}", options.node))?;
        println!("backend {} {}", options.node, backend.summary());
        let decoder = backend
            .jpeg
            .clone()
            .ok_or("the backend has no JPEG decoder")?;
        let parsed = jpeg::parse(&data);
        let (width, height, subsampling) = match &parsed {
            Ok(image) => {
                let frame = &image.frame;
                let subsampling = frame.subsampling().and_then(core_subsampling);
                println!(
                    "stream {} {}x{} {:?} restart {} tables {}",
                    options.file,
                    frame.width,
                    frame.height,
                    frame.subsampling(),
                    image.restart_interval,
                    if image.default_huffman_tables {
                        "Annex K"
                    } else {
                        "its own"
                    }
                );
                (
                    u32::from(frame.width),
                    u32::from(frame.height),
                    subsampling.unwrap_or(JpegSubsampling::Yuv420),
                )
            }
            Err(error) => {
                println!("stream {} does not parse: {error}", options.file);
                (0, 0, JpegSubsampling::Yuv420)
            }
        };
        let mut pass = true;
        if options.annex_k {
            pass &= check_annex_k(parsed.as_ref().map_err(ToString::to_string)?);
        }
        let format_name = options.format.clone().unwrap_or_else(|| {
            match subsampling {
                JpegSubsampling::Yuv422 => "422f",
                JpegSubsampling::Yuv444 => "444f",
                _ => "420f",
            }
            .to_owned()
        });
        let fourcc: [u8; 4] = format_name
            .as_bytes()
            .try_into()
            .map_err(|_| format!("{format_name}: not a four-character code"))?;
        let format = JpegOutputFormat::from_fourcc(FourCc::new(fourcc))
            .ok_or(format!("{format_name}: not a layout the engine has"))?;
        let (mcu_width, mcu_height) = subsampling.mcu_size();
        let region = match options.crop {
            None => PixelRect {
                x: 0,
                y: 0,
                width,
                height,
            },
            Some([x, y, w, h]) => {
                let (x, y) = (x * mcu_width, y * mcu_height);
                PixelRect {
                    x,
                    y,
                    width: (w * mcu_width).min(width.saturating_sub(x)),
                    height: (h * mcu_height).min(height.saturating_sub(y)),
                }
            }
        };
        let request = JpegDecode {
            data: &data,
            width,
            height,
            subsampling,
            region,
            scale: options.scale,
            format,
        };
        let (output_width, output_height) = request.output_size();
        let mut planes = Planes::for_layout(format, output_width, output_height);
        let before = crate::engines::busy();
        let mut elapsed = Vec::new();
        let mut outcome = Ok(());
        for _ in 0..options.repeat.max(1) {
            let started = StdRuntime.now();
            outcome = decoder.decode(&request, &mut planes.destinations());
            elapsed.push(StdRuntime.elapsed_since(started).as_secs_f64() * 1000.0);
            if outcome.is_err() {
                break;
            }
        }
        let after = crate::engines::busy();
        let mean =
            elapsed.iter().sum::<f64>() / f64::from(u32::try_from(elapsed.len()).unwrap_or(1));
        let busy: Vec<String> = crate::engines::spent(&before, &after)
            .into_iter()
            .filter(|&(_, spent)| spent > 0)
            .map(|(engine, spent)| format!("{engine}={spent}ns"))
            .collect();
        println!("engines {}", busy.join(" "));
        // What the backend says it decodes: a stream outside it is to be
        // refused, as unsupported, and one inside it to decode.
        let decodes = decoder.decodes();
        let answered = parsed.is_ok() && !decodes.takes(subsampling, (width, height));
        let answer = || {
            format!(
                "the backend decodes {:?} up to {}x{}",
                decodes.subsamplings, decodes.max_size.0, decodes.max_size.1
            )
        };
        match (outcome, &options.expect_error) {
            (Err(error), Some(expected)) => {
                let kind = kind(&error);
                println!("decode refused, {kind}: {error}");
                return Ok(pass && kind == expected);
            }
            (Err(error @ JpegError::UnsupportedStream(_)), None) if answered => {
                println!(
                    "decode refused as answered, UnsupportedStream: {error}; {}",
                    answer()
                );
                return Ok(pass);
            }
            (Err(error), None) => {
                println!("decode failed, {}: {error}", kind(&error));
                return Ok(false);
            }
            (Ok(()), None) if answered => {
                println!("decode ok, where {} refuses it", answer());
                return Ok(false);
            }
            (Ok(()), Some(expected)) => {
                println!("decode ok, where {expected} was expected");
                return Ok(false);
            }
            (Ok(()), None) => println!(
                "decode ok {}x{} at ({}, {}) at 1/{} into {format_name}, {:.2} ms (mean of {})",
                region.width,
                region.height,
                region.x,
                region.y,
                options.scale,
                mean,
                elapsed.len()
            ),
        }
        println!("digest md5 {}", planes.md5());
        if options.references.is_empty() {
            println!("compare nothing: a decode that should succeed needs a --reference");
            return Ok(false);
        }
        let copied = options.scale == 1
            && match (subsampling.chroma_divisors(), format.chroma_divisors()) {
                (_, None) => false,
                (None, Some(_)) => true,
                (Some(sampled), Some(layout)) => sampled == layout,
            };
        if !copied {
            let output = planes.to_planes(format, output_width, output_height)?;
            let dump = |suffix: &str, planes: &[(&str, Vec<u8>)]| match &options.dump {
                Some(prefix) => {
                    std::fs::write(format!("{prefix}.{suffix}"), planes.concat_planes())
                        .map_err(|e| format!("{prefix}.{suffix}: {e}"))
                }
                None => Ok(()),
            };
            dump("ours", &output)?;
            for (name, path) in &options.references {
                let reference = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
                let expected = |mode| {
                    expected_output(
                        &reference,
                        subsampling,
                        (width, height),
                        region,
                        format,
                        (output_width, output_height),
                        mode,
                    )
                };
                let odd = output_width % 2 == 1 || output_height % 2 == 1;
                let floor = match (odd, options.scale) {
                    (true, _) => 28.0,
                    (false, 1) => 34.0,
                    (false, _) => 30.0,
                };
                let centres = expected(ChromaMode::Centres)?;
                dump(name, &centres)?;
                let mut matched = compare_psnr(name, &output, &centres, floor);
                // RGB from chroma taken either way is right: core's
                // reference interpolates it, Apple's hardware looks to take
                // the nearest sample.
                if format == JpegOutputFormat::Bgra {
                    let nearest = expected(ChromaMode::Nearest)?;
                    let named = format!("{name}, nearest chroma,");
                    matched |= compare_psnr(&named, &output, &nearest, floor);
                }
                pass &= matched;
            }
            return Ok(pass);
        }
        let decoded = planes.to_planar(format, subsampling, region.width, region.height)?;
        for (name, path) in &options.references {
            let reference = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
            pass &= compare(
                name,
                &decoded,
                &reference,
                subsampling,
                (width, height),
                region,
                options.tolerance,
            )?;
        }
        Ok(pass)
    }

    fn kind(error: &JpegError) -> &'static str {
        match error {
            JpegError::BadData(_) => "BadData",
            JpegError::ScanEndsEarly => "ScanEndsEarly",
            JpegError::UnsupportedStream(_) => "UnsupportedStream",
            JpegError::SizeMismatch { .. } => "SizeMismatch",
            JpegError::SubsamplingMismatch => "SubsamplingMismatch",
            JpegError::UnsupportedFormat(_) => "UnsupportedFormat",
            JpegError::UnsupportedScale(_) => "UnsupportedScale",
            JpegError::Malfunction(_) => "Malfunction",
        }
    }

    fn core_subsampling(read: jpeg::Subsampling) -> Option<JpegSubsampling> {
        Some(match read {
            jpeg::Subsampling::Yuv420 => JpegSubsampling::Yuv420,
            jpeg::Subsampling::Yuv422 => JpegSubsampling::Yuv422,
            jpeg::Subsampling::Yuv444 => JpegSubsampling::Yuv444,
            jpeg::Subsampling::Yuv411 => JpegSubsampling::Yuv411,
            jpeg::Subsampling::Grey => JpegSubsampling::Grey,
            jpeg::Subsampling::Yuv440 => return None,
        })
    }

    /// The stream without its DHT segments.
    fn strip_dht(data: &[u8]) -> Result<Vec<u8>, String> {
        let mut out = data[..2].to_vec();
        let mut at = 2;
        loop {
            let header = data
                .get(at..at + 4)
                .ok_or("the stream ends before its scan")?;
            let marker = header[1];
            let length = usize::from(u16::from_be_bytes([header[2], header[3]]));
            if marker == 0xda {
                out.extend_from_slice(&data[at..]);
                return Ok(out);
            }
            if marker != 0xc4 {
                out.extend_from_slice(
                    data.get(at..at + 2 + length)
                        .ok_or("a segment runs past the end")?,
                );
            }
            at += 2 + length;
        }
    }

    /// Whether the stream's tables are Annex K's, by class and index.
    fn check_annex_k(image: &jpeg::Image<'_>) -> bool {
        let expected: [(&str, Option<HuffmanTable<'_>>, HuffmanTable<'_>); 4] = [
            ("DC 0", image.dc_tables[0], jpeg::DEFAULT_DC_LUMINANCE),
            ("DC 1", image.dc_tables[1], jpeg::DEFAULT_DC_CHROMINANCE),
            ("AC 0", image.ac_tables[0], jpeg::DEFAULT_AC_LUMINANCE),
            ("AC 1", image.ac_tables[1], jpeg::DEFAULT_AC_CHROMINANCE),
        ];
        let mut all = true;
        for (name, table, annex_k) in expected {
            let verdict = match table {
                None => "absent",
                Some(table) if table == annex_k => "Annex K's",
                Some(_) => {
                    all = false;
                    "not Annex K's"
                }
            };
            println!("tables {name}: {verdict}");
        }
        all
    }

    /// The destination's planes, rows 64-byte aligned.
    struct Planes {
        planes: Vec<(Vec<u8>, usize, u32)>,
    }

    impl Planes {
        fn for_layout(format: JpegOutputFormat, width: u32, height: u32) -> Self {
            let w = usize::try_from(width).unwrap_or(0);
            let shapes: Vec<(usize, u32)> = match format {
                JpegOutputFormat::Biplanar420 => {
                    vec![(w, height), (2 * w.div_ceil(2), height.div_ceil(2))]
                }
                JpegOutputFormat::Biplanar422 => vec![(w, height), (2 * w.div_ceil(2), height)],
                JpegOutputFormat::Biplanar444 => vec![(w, height), (2 * w, height)],
                JpegOutputFormat::Yuyv | JpegOutputFormat::Uyvy => {
                    vec![(4 * w.div_ceil(2), height)]
                }
                JpegOutputFormat::Bgra => vec![(4 * w, height)],
            };
            Self {
                planes: shapes
                    .into_iter()
                    .map(|(row, rows)| {
                        let stride = row.next_multiple_of(64).max(64);
                        (
                            vec![0xee; stride * usize::try_from(rows).unwrap_or(0)],
                            stride,
                            rows,
                        )
                    })
                    .collect(),
            }
        }

        fn destinations(&mut self) -> Vec<PlaneDestination<'_>> {
            self.planes
                .iter_mut()
                .map(|(bytes, stride, rows)| PlaneDestination {
                    bytes,
                    bytes_per_row: *stride,
                    rows: *rows,
                })
                .collect()
        }

        /// The MD5 of every plane's bytes, padding included, one after
        /// another.
        fn md5(&self) -> String {
            let mut md5 = crate::md5::Md5::new();
            for (bytes, _, _) in &self.planes {
                md5.update(bytes);
            }
            md5.hex()
        }

        /// The output's planes, each tightly packed and named: Y, Cb and Cr
        /// at the layout's sampling, or B, G, R and A.
        fn to_planes(
            &self,
            format: JpegOutputFormat,
            width: u32,
            height: u32,
        ) -> Result<Vec<(&'static str, Vec<u8>)>, String> {
            if format != JpegOutputFormat::Bgra {
                let names = ["Y", "Cb", "Cr"];
                let sampling = match format.chroma_divisors() {
                    Some((2, 2)) => JpegSubsampling::Yuv420,
                    Some((2, 1)) => JpegSubsampling::Yuv422,
                    _ => JpegSubsampling::Yuv444,
                };
                return Ok(names
                    .into_iter()
                    .zip(self.to_planar(format, sampling, width, height)?)
                    .collect());
            }
            let (w, h) = (
                usize::try_from(width).unwrap_or(0),
                usize::try_from(height).unwrap_or(0),
            );
            let (bytes, stride, _) = &self.planes[0];
            Ok(["B", "G", "R", "A"]
                .into_iter()
                .enumerate()
                .map(|(channel, name)| {
                    let mut plane = Vec::with_capacity(w * h);
                    for row in 0..h {
                        let line = &bytes[row * stride..];
                        plane.extend((0..w).map(|x| line[4 * x + channel]));
                    }
                    (name, plane)
                })
                .collect())
        }

        /// Y, then Cb and Cr at the stream's sampling (none for grey, whose
        /// chroma must be 0x80), each tightly packed.
        fn to_planar(
            &self,
            format: JpegOutputFormat,
            subsampling: JpegSubsampling,
            width: u32,
            height: u32,
        ) -> Result<Vec<Vec<u8>>, String> {
            let (w, h) = (
                usize::try_from(width).unwrap_or(0),
                usize::try_from(height).unwrap_or(0),
            );
            let divisors = format.chroma_divisors().ok_or("BGRA is not compared")?;
            let (cw, ch) = (
                w.div_ceil(usize::try_from(divisors.0).unwrap_or(1)),
                h.div_ceil(usize::try_from(divisors.1).unwrap_or(1)),
            );
            let mut y = Vec::with_capacity(w * h);
            let mut cb = Vec::with_capacity(cw * ch);
            let mut cr = Vec::with_capacity(cw * ch);
            match format {
                JpegOutputFormat::Yuyv | JpegOutputFormat::Uyvy => {
                    let (bytes, stride, _) = &self.planes[0];
                    let (luma, chroma) = if format == JpegOutputFormat::Yuyv {
                        (0, 1)
                    } else {
                        (1, 0)
                    };
                    for row in 0..h {
                        let line = &bytes[row * stride..];
                        y.extend((0..w).map(|x| line[2 * x + luma]));
                        cb.extend((0..cw).map(|x| line[4 * x + chroma]));
                        cr.extend((0..cw).map(|x| line[4 * x + chroma + 2]));
                    }
                }
                _ => {
                    let (luma, luma_stride, _) = &self.planes[0];
                    let (chroma, chroma_stride, _) = &self.planes[1];
                    for row in 0..h {
                        y.extend_from_slice(&luma[row * luma_stride..row * luma_stride + w]);
                    }
                    for row in 0..ch {
                        let line = &chroma[row * chroma_stride..];
                        cb.extend((0..cw).map(|x| line[2 * x]));
                        cr.extend((0..cw).map(|x| line[2 * x + 1]));
                    }
                }
            }
            if subsampling == JpegSubsampling::Grey {
                if cb.iter().chain(&cr).any(|&c| c != 0x80) {
                    return Err("a grey image's chroma is not 0x80".into());
                }
                println!("compare chroma of grey: all 0x80");
                return Ok(vec![y]);
            }
            Ok(vec![y, cb, cr])
        }
    }

    /// Compares the decoded planes with the reference's region, plane by
    /// plane, and prints the largest and mean differences.
    fn compare(
        name: &str,
        decoded: &[Vec<u8>],
        reference: &[u8],
        subsampling: JpegSubsampling,
        (width, height): (u32, u32),
        region: PixelRect,
        tolerance: u8,
    ) -> Result<bool, String> {
        let size = |v: u32| usize::try_from(v).unwrap_or(0);
        let (w, h) = (size(width), size(height));
        let (across, down) = subsampling.chroma_divisors().unwrap_or((1, 1));
        let (across, down) = (size(across), size(down));
        let (cw, ch) = (w.div_ceil(across), h.div_ceil(down));
        let mut shapes = vec![(
            w,
            h,
            size(region.x),
            size(region.y),
            size(region.width),
            size(region.height),
        )];
        if decoded.len() == 3 {
            let chroma = (
                cw,
                ch,
                size(region.x) / across,
                size(region.y) / down,
                size(region.width).div_ceil(across),
                size(region.height).div_ceil(down),
            );
            shapes.push(chroma);
            shapes.push(chroma);
        }
        let expected_len = w * h + if decoded.len() == 3 { 2 * cw * ch } else { 0 };
        let grey_with_chroma = decoded.len() == 1 && reference.len() == w * h + 2 * cw * ch;
        if reference.len() != expected_len && !grey_with_chroma {
            return Err(format!(
                "{name}: {} bytes, where {expected_len} were expected",
                reference.len()
            ));
        }
        let mut offset = 0;
        let mut pass = true;
        let mut report = Vec::new();
        for ((plane_name, plane), (pw, ph, x0, y0, rw, rh)) in
            ["Y", "Cb", "Cr"].iter().zip(decoded).zip(shapes)
        {
            let mut largest = 0u8;
            let mut sum = 0u64;
            let mut differing = 0u64;
            for row in 0..rh {
                for column in 0..rw {
                    let ours = plane[row * rw + column];
                    let theirs = reference[offset + (y0 + row) * pw + x0 + column];
                    let difference = ours.abs_diff(theirs);
                    largest = largest.max(difference);
                    sum += u64::from(difference);
                    differing += u64::from(difference != 0);
                }
            }
            offset += pw * ph;
            let count = (rw * rh).max(1) as f64;
            pass &= largest <= tolerance;
            report.push(format!(
                "{plane_name} max {largest} mean {:.4} differing {:.3}%",
                sum as f64 / count,
                100.0 * differing as f64 / count
            ));
        }
        println!("compare {name}: {}", report.join("; "));
        Ok(pass)
    }

    /// How a conversion to RGB finds a pixel's chroma.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ChromaMode {
        /// Interpolated from the sample centres around it, as JFIF sites
        /// a JPEG's chroma.
        Centres,
        /// The sample it is in, as Apple's hardware's BGRA looks to.
        Nearest,
    }

    /// What a scaled or converted decode should be, by core's scaler
    /// reference's resampling (`reference::resample`) of a reference
    /// decoder's planes: each plane named as [`Planes::to_planes`] names
    /// it. Each plane is resampled at its own resolution, as
    /// `reference::run` does between Y'CbCr formats; into BGRA, the chroma
    /// is then brought to every pixel as `chroma_mode` says and converted
    /// with BT.601 at full range.
    fn expected_output(
        reference: &[u8],
        subsampling: JpegSubsampling,
        (width, height): (u32, u32),
        region: PixelRect,
        format: JpegOutputFormat,
        (output_width, output_height): (u32, u32),
        chroma_mode: ChromaMode,
    ) -> Result<Vec<(&'static str, Vec<u8>)>, String> {
        let size = |v: u32| usize::try_from(v).unwrap_or(0);
        let (w, h) = (size(width), size(height));
        let (ow, oh) = (size(output_width), size(output_height));
        let sampled = subsampling.chroma_divisors();
        let (sx, sy) = sampled.map_or((1, 1), |(a, d)| (size(a), size(d)));
        let (cw, ch) = (w.div_ceil(sx), h.div_ceil(sy));
        let chroma_len = if sampled.is_some() { 2 * cw * ch } else { 0 };
        if reference.len() < w * h + chroma_len {
            return Err(format!(
                "a reference of {} bytes, where {} were expected",
                reference.len(),
                w * h + chroma_len
            ));
        }
        let luma = Picture {
            width: w,
            height: h,
            pixels: reference[..w * h]
                .iter()
                .map(|&v| [f64::from(v) / 255.0, 0.0, 0.0, 1.0])
                .collect(),
        };
        let chroma = sampled.map(|_| {
            let (cb, cr) = reference[w * h..w * h + chroma_len].split_at(cw * ch);
            Picture {
                width: cw,
                height: ch,
                pixels: cb
                    .iter()
                    .zip(cr)
                    .map(|(&b, &r)| {
                        let c = |v: u8| (f64::from(v) - 128.0) / 255.0;
                        [0.0, c(b), c(r), 1.0]
                    })
                    .collect(),
            }
        });
        let f = |v: u32| f64::from(v);
        let window = (f(region.x), f(region.y), f(region.width), f(region.height));
        let code = |v: f64| (v * 255.0).round().clamp(0.0, 255.0) as u8;
        let code_chroma = |v: f64| (128.0 + v * 255.0).round().clamp(0.0, 255.0) as u8;
        let smooth = |picture: &Picture, window, ow, oh| {
            resample(
                picture,
                window,
                ow,
                oh,
                Orientation::Identity,
                Filter::Smooth,
            )
        };
        let Some((dx, dy)) = format.chroma_divisors() else {
            // Each plane resampled to the output, then converted pixel by
            // pixel. With the chroma's centres, it is resampled straight to
            // every pixel: interpolated where it grows, averaged where it
            // shrinks. With the nearest, it is resampled at its own
            // sampling, and each pixel takes the sample it is in.
            let out_luma = smooth(&luma, window, ow, oh);
            let (sxf, syf) = (sx as f64, sy as f64);
            let chroma_window = (
                window.0 / sxf,
                window.1 / syf,
                window.2 / sxf,
                window.3 / syf,
            );
            let out_chroma = chroma.as_ref().map(|chroma| match chroma_mode {
                ChromaMode::Centres => (smooth(chroma, chroma_window, ow, oh), (1, 1)),
                ChromaMode::Nearest => (
                    smooth(chroma, chroma_window, ow.div_ceil(sx), oh.div_ceil(sy)),
                    (sx, sy),
                ),
            });
            let mut rgb = Vec::with_capacity(ow * oh);
            for y in 0..oh {
                for x in 0..ow {
                    let mut pixel = [out_luma.pixels[y * ow + x][0], 0.0, 0.0, 1.0];
                    if let Some((chroma, (across, down))) = &out_chroma {
                        let c = chroma.pixels[(y / down) * chroma.width + x / across];
                        (pixel[1], pixel[2]) = (c[1], c[2]);
                    }
                    rgb.push(to_rgb(pixel, Matrix::Bt601));
                }
            }
            return Ok([("B", 2), ("G", 1), ("R", 0), ("A", 3)]
                .into_iter()
                .map(|(name, channel)| (name, rgb.iter().map(|p| code(p[channel])).collect()))
                .collect());
        };
        let mut planes = vec![(
            "Y",
            smooth(&luma, window, ow, oh)
                .pixels
                .iter()
                .map(|p| code(p[0]))
                .collect::<Vec<u8>>(),
        )];
        let (dx, dy) = (size(dx), size(dy));
        let (ocw, och) = (ow.div_ceil(dx), oh.div_ceil(dy));
        match &chroma {
            Some(chroma) => {
                let (sxf, syf) = (sx as f64, sy as f64);
                let chroma_window = (
                    window.0 / sxf,
                    window.1 / syf,
                    window.2 / sxf,
                    window.3 / syf,
                );
                let out = smooth(chroma, chroma_window, ocw, och);
                planes.push(("Cb", out.pixels.iter().map(|p| code_chroma(p[1])).collect()));
                planes.push(("Cr", out.pixels.iter().map(|p| code_chroma(p[2])).collect()));
            }
            None => {
                planes.push(("Cb", vec![0x80; ocw * och]));
                planes.push(("Cr", vec![0x80; ocw * och]));
            }
        }
        Ok(planes)
    }

    /// Named planes, one after another.
    trait ConcatPlanes {
        fn concat_planes(&self) -> Vec<u8>;
    }

    impl ConcatPlanes for [(&str, Vec<u8>)] {
        fn concat_planes(&self) -> Vec<u8> {
            self.iter()
                .flat_map(|(_, plane)| plane.iter().copied())
                .collect()
        }
    }

    /// Compares each plane of `ours` with `expected`'s of its name by
    /// PSNR, with our mean difference from it, and prints them; passes
    /// where every plane is `floor` dB or more.
    fn compare_psnr(
        name: &str,
        ours: &[(&'static str, Vec<u8>)],
        expected: &[(&'static str, Vec<u8>)],
        floor: f64,
    ) -> bool {
        let mut pass = ours.len() == expected.len();
        let mut report = Vec::new();
        for ((plane, a), (other, b)) in ours.iter().zip(expected) {
            if plane != other || a.len() != b.len() {
                report.push(format!("{plane}: not {other}'s size"));
                pass = false;
                continue;
            }
            let squares: f64 = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
                .sum();
            let bias = a
                .iter()
                .zip(b)
                .map(|(&x, &y)| f64::from(x) - f64::from(y))
                .sum::<f64>()
                / a.len().max(1) as f64;
            let psnr = if squares == 0.0 {
                f64::INFINITY
            } else {
                10.0 * (255.0f64.powi(2) * a.len().max(1) as f64 / squares).log10()
            };
            pass &= psnr >= floor;
            report.push(format!("{plane} {psnr:.1} dB, mean {bias:+.2}"));
        }
        println!(
            "compare {name} by the scaler's reference (pass at {floor} dB): {}",
            report.join("; ")
        );
        pass
    }
}
