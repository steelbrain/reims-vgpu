//! The scaler suite: a scaler backend against core's CPU reference.
//!
//! [`check`] prints the capabilities the guest would be given, then runs
//! every case through the backend's `Accelerator`, as core's scaler device
//! does: copies and conversions between each pair of formats, scaling up
//! and down, crops, destination windows, the eight orientations, the
//! nearest filter and odd sizes.
//!
//! Each case's source is core's test picture, drawn from scratch; its
//! destination starts mid grey. The output is compared with
//! `reference::run` of the same plan by `reference::psnr`, per plane over
//! every byte, and passes at 34 dB for a copy or a conversion, 30 dB when
//! it scales, 28 dB where a size or a window's position is odd; each case's
//! line also gives the largest difference of a sample (a code) and how many
//! samples differ. Bytes outside the window's coverage must be untouched.
//! Where RGB and Y'CbCr meet the reference is also made with BT.709, where
//! a case asks for the nearest filter, with the smooth one, and where a
//! case fails, with the other seven orientations, and the best is named,
//! so a wrong matrix, filter or orientation shows as such. A refusal (kIOReturnUnsupported) is counted
//! apart where Apple's scaler refuses too, a pair with 2vuy or 32ARGB
//! either way, which the plan refuses, or a side under the capabilities'
//! smallest; any other refusal fails, and so does a run in which no case
//! passes.
//!
//! Before the cases, `siting` lines give the chroma kernels the backend's
//! conversions have, measured as [`siting`](mod@siting) says; and `map` lines where
//! its scales take their pixels from, beside where the plan's alignment
//! puts them ([`mapping`](mod@mapping)).
//!
//! Lines start with a word to grep: `caps`, `siting`, `map`, `case`,
//! `engines`, `bench`, `result`. The engines line is the GPU driver's
//! per-client busy time for this process's render-node descriptors over
//! the cases (`/proc/self/fdinfo`, amdgpu's `drm-engine-*`; nothing where
//! there is none).
//!
//! For looking into a case: `RVGV_SCALER_BIAS=1` prints each plane's mean
//! signed difference from the reference by byte position (a channel's
//! bias), `RVGV_SCALER_COORDS=1` draws the source as its own coordinates
//! and prints where a grid of output pixels came from, and
//! `RVGV_SCALER_DUMP=DIR` writes the source, output and reference as BMPs.

pub use cases::{check, name as format_name};
pub use mapping::{Mapped, measure_all as mapping};
pub use siting::{Kernel, Measured, measure as siting};

mod mapping;
mod siting;

#[cfg(test)]
mod tests {
    use reims_vgpu_video_core::scaler::backend::{
        Accelerator, ReferenceBackend, ScalerBackend, TransformJob,
    };
    use reims_vgpu_video_core::scaler::capabilities::{Capabilities, FormatClass};
    use reims_vgpu_video_core::scaler::io_return::{self, IoReturn};

    /// Core's CPU stand-in is the reference itself, so it passes: the
    /// suite's own check, on every host. One case of each kind, since all
    /// of them take two minutes in a debug build; the backends' runs take
    /// them all.
    fn reference() -> ReferenceBackend {
        ReferenceBackend {
            capabilities: Capabilities {
                scaler_version: 0,
                formats: vec![FormatClass::TwoPlanes420, FormatClass::Argb8888],
                min_source: 4,
                max_source: (4096, 4096),
                min_destination: 4,
                max_destination: (4096, 4096),
                max_scale: [8, 8, 8, 4],
                xy_offsets_with_tiling: (false, false),
            },
        }
    }

    /// The reference sites chroma as Apple's scaler does, as measured on
    /// the M3 Max's. Up, every Y'CbCr source's chroma sits on its first luma
    /// column and row and is interpolated between; down, each sample is the
    /// pixel or the mean of the two pixels the destination's siting names,
    /// the first pixel where it names none.
    #[test]
    fn the_reference_sites_chroma_as_apples_scaler() {
        use reims_vgpu_video_core::scaler::backend::ScalerBackend;
        const BETWEEN: [f64; 6] = [0.0, 0.25, 0.5, 0.25, 0.0, 0.0];
        const FIRST: [f64; 6] = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        const BOTH: [f64; 6] = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0];
        const SECOND: [f64; 6] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut accelerator = reference()
            .create_accelerator()
            .unwrap_or_else(|_| unreachable!());
        let measured = super::siting(accelerator.as_mut());
        assert_eq!(measured.len(), 2 + 2 + 1 + 7 * 2 + 1);
        for m in measured {
            // Bits 15 to 17: 1 left, 2 centre, 3 top left, 4 top, 5 bottom
            // left, 6 bottom.
            let want = match (m.direction, m.vertical, m.cosite) {
                ("up", _, _) => BETWEEN,
                ("down", false, 2 | 4 | 6) => BOTH,
                ("down", false, _) => FIRST,
                ("down", true, 1 | 2) => BOTH,
                ("down", true, 5 | 6) => SECOND,
                _ => FIRST,
            };
            assert!(
                m.kernel.is_ok_and(|k| k.near(want, 0.02)),
                "{}, where Apple's is {want:?}",
                m.line()
            );
        }
    }

    /// The reference scales as its plan's alignment says: each axis's
    /// positions within its rounding of where the plan puts them, through
    /// TransferSurface's flags and TransformSurface's.
    #[test]
    fn the_reference_samples_where_its_plan_says() {
        use reims_vgpu_video_core::scaler::backend::ScalerBackend;
        let mut accelerator = reference()
            .create_accelerator()
            .unwrap_or_else(|_| unreachable!());
        let measured = super::mapping(accelerator.as_mut());
        assert!(measured.len() >= 40);
        for m in measured {
            let (mean, spread) = m.less[0];
            assert!(
                m.status.is_none() && m.samples > 50 && mean.abs() < 0.02 && spread < 0.1,
                "{}",
                m.line()
            );
        }
    }

    #[test]
    fn the_reference_backend_passes_a_case_of_each_kind() {
        let reference = reference();
        let kinds = [
            "copy-420v-420v",
            "csc-x420-BGRA",
            "csc-yuvs-420f",
            "down2-420v",
            "up4-420v",
            "crop-fraction-BGRA",
            "rect-odd-yuvs",
            "nearest-down-420v",
            "odd-BGRA-420v",
            "turn-aniso-BGRA",
            "rotate-csc-420v-BGRA",
        ]
        .map(String::from);
        assert!(super::check(&reference, "reference", &kinds, false));
    }

    /// A scaler that refuses every transform, as a backend with no video
    /// processing would.
    struct Refusing(ReferenceBackend);

    impl ScalerBackend for Refusing {
        fn create_accelerator(&self) -> Result<Box<dyn Accelerator>, String> {
            struct Refuses(Box<dyn Accelerator>);
            impl Accelerator for Refuses {
                fn capabilities(&self) -> Capabilities {
                    self.0.capabilities()
                }
                fn transform(&mut self, _: &mut TransformJob<'_>) -> IoReturn {
                    io_return::UNSUPPORTED
                }
            }
            Ok(Box::new(Refuses(self.0.create_accelerator()?)))
        }
    }

    #[test]
    fn a_check_of_refusals_alone_or_of_no_case_fails() {
        let kinds = ["copy-420v-420v", "csc-x420-BGRA"].map(String::from);
        assert!(!super::check(
            &Refusing(reference()),
            "refusing",
            &kinds,
            false
        ));
        let none = ["no-such-case".to_owned()];
        assert!(!super::check(&reference(), "reference", &none, false));
    }
}

mod cases {

    use reims_vgpu_video_core::device_wait::GuestHold;
    use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
    use reims_vgpu_video_core::scaler::backend::{Accelerator, ScalerBackend, TransformJob};
    use reims_vgpu_video_core::scaler::coverage::{Window, covered};
    use reims_vgpu_video_core::scaler::format::{Packing, ScalerFormat, apple_refuses};
    use reims_vgpu_video_core::scaler::io_return;
    use reims_vgpu_video_core::scaler::plan::{
        Filter, FormatOf, Matrix, Orientation, Plan, Siting,
    };
    use reims_vgpu_video_core::scaler::reference::{self, Space};
    use reims_vgpu_video_core::scaler::request::{
        FIXED_ONE, FixedRect, Rect, Transform, TransformFlags,
    };
    use reims_vgpu_video_core::surfaces::{GuestSurface, PlaneLayout, SurfaceError, SurfaceLayout};
    use reims_vgpu_video_core::sync::Mutex;

    /// A guest surface in this process's memory.
    pub(super) struct Memory {
        layout: SurfaceLayout,
        bytes: Mutex<Vec<u8>>,
    }

    impl Memory {
        pub(super) fn new(layout: SurfaceLayout, bytes: Vec<u8>) -> Memory {
            Memory {
                layout,
                bytes: Mutex::new(bytes),
            }
        }

        /// A copy of its bytes.
        pub(super) fn bytes(&self) -> Vec<u8> {
            self.bytes.lock().clone()
        }
    }

    impl GuestSurface for Memory {
        fn layout(&self) -> SurfaceLayout {
            self.layout.clone()
        }

        fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
            write(&mut self.bytes.lock());
            Ok(())
        }

        fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
            read(&self.bytes.lock());
            Ok(())
        }
    }

    /// A layout of `format`, `width` by `height`, each row padded past
    /// 64-byte alignment by 64 bytes, as a guest's IOSurface may be.
    pub(super) fn layout(format: ScalerFormat, width: u32, height: u32) -> SurfaceLayout {
        let (w, h) = (width as usize, height as usize);
        let pad = |bytes: usize| bytes.div_ceil(64) * 64 + 64;
        let s = format.bytes_per_sample();
        let planes = match format.packing() {
            Packing::Biplanar420 => {
                let luma = pad(w * s);
                let chroma = pad(w.div_ceil(2) * 2 * s);
                vec![
                    PlaneLayout {
                        offset: 0,
                        bytes_per_row: luma,
                        width,
                        height,
                    },
                    PlaneLayout {
                        offset: luma * h,
                        bytes_per_row: chroma,
                        width: width.div_ceil(2),
                        height: height.div_ceil(2),
                    },
                ]
            }
            Packing::Packed422(_) => vec![PlaneLayout {
                offset: 0,
                bytes_per_row: pad(w.div_ceil(2) * 4),
                width,
                height,
            }],
            Packing::Rgb32(_) => vec![PlaneLayout {
                offset: 0,
                bytes_per_row: pad(w * 4),
                width,
                height,
            }],
        };
        SurfaceLayout {
            pixel_format: format.fourcc(),
            width,
            height,
            planes,
            bytes_per_row: 0,
        }
    }

    pub(super) fn length(layout: &SurfaceLayout) -> usize {
        layout
            .planes
            .iter()
            .map(|p| p.offset + p.bytes_per_row * p.height as usize)
            .max()
            .unwrap_or(0)
    }

    /// A format as the lines print it.
    pub fn name(format: ScalerFormat) -> String {
        match format {
            ScalerFormat::Argb => "ARGB".to_owned(),
            other => other.fourcc().to_string(),
        }
    }

    /// One case.
    #[derive(Clone, Debug)]
    struct Case {
        name: String,
        src: ScalerFormat,
        dst: ScalerFormat,
        src_size: (u32, u32),
        dst_size: (u32, u32),
        /// The source window in pixels, 16.16 after multiplying; zero for
        /// all.
        crop: (f64, f64, f64, f64),
        /// The destination window; zero for all.
        rect: (u32, u32, u32, u32),
        transform: u32,
        nearest: bool,
        /// `FixUpscaling` (flag bit 12): set, as every `TransformSurface`
        /// sets it, but in the `transfer-` cases, which are what a guest's
        /// `TransferSurface` sends: whole windows and no bit 12.
        fix: bool,
    }

    impl Case {
        fn new(name: impl Into<String>, src: ScalerFormat, dst: ScalerFormat) -> Case {
            Case {
                name: name.into(),
                src,
                dst,
                src_size: (320, 240),
                dst_size: (320, 240),
                crop: (0.0, 0.0, 0.0, 0.0),
                rect: (0, 0, 0, 0),
                transform: 0,
                nearest: false,
                fix: true,
            }
        }
        fn sizes(mut self, src: (u32, u32), dst: (u32, u32)) -> Case {
            self.src_size = src;
            self.dst_size = dst;
            self
        }
        fn crop(mut self, x: f64, y: f64, w: f64, h: f64) -> Case {
            self.crop = (x, y, w, h);
            self
        }
        fn rect(mut self, x: u32, y: u32, w: u32, h: u32) -> Case {
            self.rect = (x, y, w, h);
            self
        }
        fn transform(mut self, bits: u32) -> Case {
            self.transform = bits;
            self
        }
        fn nearest(mut self) -> Case {
            self.nearest = true;
            self
        }
        fn transfer(mut self) -> Case {
            self.fix = false;
            self
        }
    }

    fn cases() -> Vec<Case> {
        use ScalerFormat::*;
        let mut list = Vec::new();
        for src in ScalerFormat::ALL {
            for dst in ScalerFormat::ALL {
                let kind = if src == dst { "copy" } else { "csc" };
                list.push(Case::new(
                    format!("{kind}-{}-{}", name(src), name(dst)),
                    src,
                    dst,
                ));
            }
        }
        let n = Nv12VideoRange;
        list.extend([
            Case::new("down2-420v", n, n).sizes((1280, 720), (640, 360)),
            Case::new("down4-BGRA", Bgra, Bgra).sizes((1280, 720), (320, 180)),
            Case::new("down8-420v", n, n).sizes((1920, 1080), (240, 136)),
            Case::new("down8-BGRA", Bgra, Bgra).sizes((1920, 1080), (240, 136)),
            Case::new("up4-420v", n, n).sizes((320, 180), (1280, 720)),
            Case::new("up8-BGRA", Bgra, Bgra).sizes((160, 90), (1280, 720)),
            Case::new("scale-csc-420v-BGRA", n, Bgra).sizes((1920, 1080), (1280, 720)),
            Case::new("scale-csc-BGRA-420v", Bgra, n).sizes((1920, 1080), (1280, 720)),
            Case::new("scale-x420-420v", P010VideoRange, n).sizes((1920, 1080), (960, 540)),
            Case::new("crop-420v", n, n)
                .sizes((1280, 720), (320, 180))
                .crop(160.0, 90.0, 640.0, 360.0),
            Case::new("crop-fraction-BGRA", Bgra, Bgra)
                .sizes((640, 360), (320, 180))
                .crop(80.5, 40.25, 320.0, 180.0),
            Case::new("rect-420v", n, n)
                .sizes((640, 360), (1280, 720))
                .rect(64, 32, 640, 360),
            Case::new("rect-scale-BGRA", Bgra, Bgra)
                .sizes((640, 360), (1280, 720))
                .rect(320, 180, 320, 180),
            Case::new("rect-odd-420v", n, n)
                .sizes((320, 240), (640, 480))
                .rect(63, 31, 320, 240),
            Case::new("rect-odd-yuvs", Yuyv, Yuyv)
                .sizes((320, 240), (640, 480))
                .rect(63, 31, 320, 240),
            Case::new("rect-csc-420v-BGRA", n, Bgra)
                .sizes((640, 360), (1280, 720))
                .rect(100, 50, 640, 360),
            Case::new("nearest-up-BGRA", Bgra, Bgra)
                .sizes((160, 120), (480, 360))
                .nearest(),
            Case::new("nearest-down-420v", n, n)
                .sizes((640, 480), (320, 240))
                .nearest(),
            Case::new("odd-420v", n, n).sizes((321, 241), (161, 121)),
            Case::new("odd-copy-420v", n, n).sizes((321, 241), (321, 241)),
            Case::new("odd-yuvs-BGRA", Yuyv, Bgra).sizes((321, 241), (161, 121)),
            Case::new("odd-BGRA-420v", Bgra, n).sizes((321, 241), (321, 241)),
            Case::new("turn-aniso-BGRA", Bgra, Bgra)
                .sizes((320, 240), (240, 160))
                .transform(4),
            Case::new("turn-1080-BGRA", Bgra, Bgra)
                .sizes((1920, 1080), (1080, 1920))
                .transform(4),
            Case::new("turn-1080-half-420v", n, n)
                .sizes((1920, 1080), (540, 960))
                .transform(4),
            Case::new("turn-1080-flip-420v", n, n)
                .sizes((1920, 1080), (1920, 1080))
                .transform(1),
            Case::new("odd-rotate-420v", n, n)
                .sizes((321, 241), (241, 321))
                .transform(4),
            Case::new("odd-flip-BGRA", Bgra, Bgra)
                .sizes((321, 241), (321, 241))
                .transform(1),
            Case::new("odd-crop-rotate-BGRA", Bgra, Bgra)
                .sizes((320, 240), (101, 99))
                .crop(13.0, 7.0, 99.0, 101.0)
                .transform(6),
            Case::new("rotate-scale-BGRA", Bgra, Bgra)
                .sizes((320, 240), (120, 160))
                .transform(4),
            Case::new("rotate-crop-BGRA", Bgra, Bgra)
                .sizes((320, 240), (240, 160))
                .crop(0.0, 0.0, 160.0, 240.0)
                .rect(0, 0, 240, 160)
                .transform(4),
            Case::new("rotate-rect-BGRA", Bgra, Bgra)
                .sizes((320, 240), (480, 480))
                .rect(100, 50, 240, 320)
                .transform(4),
            Case::new("rotate-csc-420v-BGRA", n, Bgra)
                .sizes((320, 240), (240, 320))
                .transform(4),
            // Sides under Apple's scaler's smallest, 4, which a backend
            // that offers them transforms, and a test may have one take.
            Case::new("tiny-copy-BGRA", Bgra, Bgra).sizes((1, 1), (1, 1)),
            Case::new("tiny-down-BGRA", Bgra, Bgra).sizes((4, 2), (2, 1)),
            Case::new("tiny-csc-BGRA-420v", Bgra, n).sizes((2, 2), (2, 2)),
            Case::new("tiny-up-BGRA", Bgra, Bgra).sizes((2, 2), (4, 4)),
            Case::new("tiny-copy-420v", n, n).sizes((2, 2), (2, 2)),
            Case::new("tiny-csc-420v-BGRA", n, Bgra).sizes((2, 2), (2, 2)),
            Case::new("tiny-turn-BGRA", Bgra, Bgra)
                .sizes((3, 1), (1, 3))
                .transform(4),
        ]);
        // As TransferSurface sends them: whole windows, no FixUpscaling, so
        // every axis puts the first pixels on each other.
        list.extend(
            [
                Case::new("transfer-up2-420v", n, n).sizes((320, 180), (640, 360)),
                Case::new("transfer-up2-BGRA", Bgra, Bgra).sizes((160, 120), (320, 240)),
                Case::new("transfer-up1.5-BGRA", Bgra, Bgra).sizes((320, 240), (480, 360)),
                Case::new("transfer-up4-420v", n, n).sizes((320, 180), (1280, 720)),
                Case::new("transfer-up8-BGRA", Bgra, Bgra).sizes((160, 90), (1280, 720)),
                Case::new("transfer-down2-BGRA", Bgra, Bgra).sizes((640, 480), (320, 240)),
                Case::new("transfer-csc-up-420v-BGRA", n, Bgra).sizes((640, 360), (1280, 720)),
                Case::new("transfer-turn-up-BGRA", Bgra, Bgra)
                    .sizes((160, 120), (240, 320))
                    .transform(4),
                Case::new("transfer-odd-up-BGRA", Bgra, Bgra).sizes((321, 241), (642, 482)),
            ]
            .map(Case::transfer),
        );
        for bits in 1..8 {
            // A source window that is not the whole source, turned.
            let turned = bits >= 4;
            let dst = if turned { (120, 160) } else { (160, 120) };
            list.push(
                Case::new(format!("crop-transform-{bits}-BGRA"), Bgra, Bgra)
                    .sizes((320, 240), dst)
                    .crop(40.0, 30.0, 160.0, 120.0)
                    .transform(bits),
            );
            list.push(
                Case::new(format!("crop-transform-{bits}-420v"), n, n)
                    .sizes((320, 240), dst)
                    .crop(40.0, 30.0, 160.0, 120.0)
                    .transform(bits),
            );
        }
        for bits in 1..8 {
            let turned = bits >= 4;
            let dst = if turned { (240, 320) } else { (320, 240) };
            list.push(
                Case::new(format!("transform-{bits}-BGRA"), Bgra, Bgra)
                    .sizes((320, 240), dst)
                    .transform(bits),
            );
            list.push(
                Case::new(format!("transform-{bits}-420v"), n, n)
                    .sizes((320, 240), dst)
                    .transform(bits),
            );
        }
        list
    }

    /// A surface of `format` holding `picture` (RGB), encoded with BT.601.
    fn surface(format: ScalerFormat, size: (u32, u32), rgb: Option<[f64; 4]>) -> Memory {
        let layout = layout(format, size.0, size.1);
        let mut bytes = vec![0u8; length(&layout)];
        if let Ok(of) = FormatOf::of(&layout) {
            let (w, h) = (size.0 as usize, size.1 as usize);
            let picture = match rgb {
                Some(rgb) => reference::flat_picture(w, h, rgb),
                None if std::env::var_os("RVGV_SCALER_COORDS").is_some() => {
                    let mut p = reference::flat_picture(w, h, [0.0; 4]);
                    for y in 0..h {
                        for x in 0..w {
                            p.pixels[y * w + x] = [
                                x as f64 / (w - 1) as f64,
                                y as f64 / (h - 1) as f64,
                                0.0,
                                1.0,
                            ];
                        }
                    }
                    p
                }
                None => reference::test_picture(w, h),
            };
            reference::encode(
                &of,
                &mut bytes,
                &picture,
                0,
                0,
                Space::Rgb,
                Matrix::Bt601,
                Siting::CENTRED,
            );
        }
        Memory {
            layout,
            bytes: Mutex::new(bytes),
        }
    }

    fn transform_of(case: &Case) -> Transform {
        let fixed = |v: f64| (v * FIXED_ONE as f64).round() as u64;
        let (x, y, w, h) = case.crop;
        let (rx, ry, rw, rh) = case.rect;
        let mut flags = case.transform as u64;
        if case.nearest {
            flags |= 0x80;
        }
        if case.fix {
            flags |= 0x1000;
        }
        Transform {
            source_rect: FixedRect {
                x: fixed(x),
                y: fixed(y),
                width: fixed(w),
                height: fixed(h),
            },
            destination_rect: Rect {
                x: rx,
                y: ry,
                width: rw,
                height: rh,
            },
            flags: TransformFlags(flags),
            ..Transform::default()
        }
    }

    /// What a case found.
    enum Outcome {
        Ok,
        Bad,
        Refused,
    }

    fn worst(planes: &[f64]) -> f64 {
        planes.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// The largest difference between two surfaces of `of`, a sample at a time over each
    /// plane's samples (a 10-bit sample's code, not its 16 bits), and how many samples differ.
    fn largest_difference(of: &FormatOf, a: &[u8], b: &[u8]) -> (u32, usize) {
        let (mut largest, mut count) = (0u32, 0usize);
        let s = of.format.bytes_per_sample();
        for span in &of.planes {
            for row in 0..span.rows {
                let at = span.offset + row * span.bytes_per_row;
                for k in (0..span.row_bytes).step_by(s) {
                    let sample = |bytes: &[u8]| match s {
                        2 => u32::from(u16::from_le_bytes([bytes[at + k], bytes[at + k + 1]]) >> 6),
                        _ => u32::from(bytes[at + k]),
                    };
                    let d = sample(a).abs_diff(sample(b));
                    largest = largest.max(d);
                    count += usize::from(d != 0);
                }
            }
        }
        (largest, count)
    }

    fn planes_text(planes: &[f64]) -> String {
        planes
            .iter()
            .map(|p| {
                if p.is_infinite() {
                    "inf".to_owned()
                } else {
                    format!("{p:.1}")
                }
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    fn expected(plan: &Plan, source: &[u8], before: &[u8]) -> Option<Vec<u8>> {
        let mut out = before.to_vec();
        reference::run(plan, source, &mut out).ok().map(|()| out)
    }

    /// Runs `case`. A pair of formats Apple's scaler refuses must be
    /// refused, by the plan, and a side under `smallest` may be, by the
    /// backend, both with kIOReturnUnsupported; any other refusal is bad.
    fn run_case(accelerator: &mut dyn Accelerator, case: &Case, smallest: u32) -> Outcome {
        let src = surface(case.src, case.src_size, None);
        let dst = surface(case.dst, case.dst_size, Some([0.5, 0.5, 0.5, 1.0]));
        let transform = transform_of(case);
        let refused_as_apple = apple_refuses(case.src, case.dst);
        let plan = match Plan::new(&transform, &src.layout, &dst.layout) {
            Err(io_return::UNSUPPORTED) if refused_as_apple => {
                println!(
                    "case {}: refused, as Apple's scaler refuses {} into {}; ok",
                    case.name,
                    name(case.src),
                    name(case.dst)
                );
                return Outcome::Refused;
            }
            Ok(_) if refused_as_apple => {
                println!(
                    "case {}: planned, where Apple's scaler refuses {} into {}; bad",
                    case.name,
                    name(case.src),
                    name(case.dst)
                );
                return Outcome::Bad;
            }
            Ok(plan) => plan,
            Err(status) => {
                println!("case {}: no plan, {status:#x}; bad", case.name);
                return Outcome::Bad;
            }
        };
        let source = src.bytes.lock().clone();
        let before = dst.bytes.lock().clone();
        let mut job = TransformJob {
            transform: &transform,
            plan: plan.clone(),
            source: &src,
            destination: &dst,
            hold: GuestHold::nothing(),
            inputs: Default::default(),
            outputs: Default::default(),
        };
        let started = StdRuntime.now();
        let status = accelerator.transform(&mut job);
        let ms = StdRuntime.elapsed_since(started).as_secs_f64() * 1000.0;
        let head = format!(
            "case {}: {} {}x{} -> {} {}x{}",
            case.name,
            name(case.src),
            case.src_size.0,
            case.src_size.1,
            name(case.dst),
            case.dst_size.0,
            case.dst_size.1
        );
        let small = [case.src_size, case.dst_size]
            .iter()
            .any(|&(w, h)| w.min(h) < smallest);
        if status == io_return::UNSUPPORTED && small {
            println!("{head}: status {status:#x} ({ms:.1} ms); refused, a side under {smallest}");
            return Outcome::Refused;
        }
        if status != io_return::SUCCESS {
            println!("{head}: status {status:#x} ({ms:.1} ms); bad");
            return Outcome::Bad;
        }
        let out = dst.bytes.lock().clone();
        let of = &plan.destination_format;
        let Some(reference) = expected(&plan, &source, &before) else {
            println!("{head}: the reference refused; bad");
            return Outcome::Bad;
        };
        let planes = reference::psnr(of, &out, &reference);
        if let Some(dir) = std::env::var_os("RVGV_SCALER_DUMP") {
            for (what, bytes) in [("out", &out), ("ref", &reference), ("src", &source)] {
                let of = if what == "src" {
                    &plan.source_format
                } else {
                    of
                };
                let picture = reference::decode(of, bytes, Space::Rgb, Matrix::Bt601);
                let path = std::path::Path::new(&dir).join(format!("{}-{what}.bmp", case.name));
                let _ = std::fs::write(path, bmp(&picture));
            }
        }
        if std::env::var_os("RVGV_SCALER_COORDS").is_some() {
            let pic = reference::decode(of, &out, Space::Rgb, Matrix::Bt601);
            let refpic = reference::decode(of, &reference, Space::Rgb, Matrix::Bt601);
            let (sw, sh) = (case.src_size.0 as f64 - 1.0, case.src_size.1 as f64 - 1.0);
            let grid: Vec<(f64, f64)> = (0..5)
                .flat_map(|j| (0..5).map(move |i| (i as f64 * 0.245, j as f64 * 0.245)))
                .collect();
            for (fx, fy) in grid {
                let (x, y) = (
                    (fx * pic.width as f64) as usize,
                    (fy * pic.height as f64) as usize,
                );
                let o = pic.pixels[y * pic.width + x];
                let r = refpic.pixels[y * pic.width + x];
                println!(
                    "coords {} at ({x},{y}): source ({:.0},{:.0}), reference ({:.0},{:.0})",
                    case.name,
                    o[0] * sw,
                    o[1] * sh,
                    r[0] * sw,
                    r[1] * sh
                );
            }
        }
        if std::env::var_os("RVGV_SCALER_BIAS").is_some() {
            // The mean signed difference of each byte position within a
            // 4-byte group, per plane: a channel's bias.
            for (i, span) in of.planes.iter().enumerate() {
                let mut sums = [0f64; 4];
                let mut n = [0usize; 4];
                for row in 0..span.rows {
                    let at = span.offset + row * span.bytes_per_row;
                    for k in 0..span.row_bytes {
                        let d = f64::from(out[at + k]) - f64::from(reference[at + k]);
                        sums[k % 4] += d;
                        n[k % 4] += 1;
                    }
                }
                let bias: Vec<String> = sums
                    .iter()
                    .zip(n)
                    .map(|(s, n)| format!("{:+.2}", s / n.max(1) as f64))
                    .collect();
                println!("bias {} plane {i}: {}", case.name, bias.join(" "));
            }
        }
        // Where RGB and Y'CbCr meet, the other matrix too; where the case
        // turns, every orientation.
        let crosses = case.src.is_ycbcr() != case.dst.is_ycbcr();
        let mut notes = Vec::new();
        if crosses {
            let other = Plan {
                matrix: Matrix::Bt709,
                ..plan.clone()
            };
            if let Some(r) = expected(&other, &source, &before) {
                let p709 = worst(&reference::psnr(of, &out, &r));
                notes.push(if p709 > worst(&planes) {
                    format!("matrix 709 ({p709:.1})")
                } else {
                    "matrix 601".to_owned()
                });
            }
        }
        // An odd size or window position leaves a chroma sample shared
        // with what is outside, where scalers differ.
        let (rx, ry, rw, rh) = case.rect;
        let odd = [case.src_size, case.dst_size, (rx, ry), (rw, rh)]
            .iter()
            .any(|&(w, h)| w % 2 == 1 || h % 2 == 1);
        if case.nearest {
            let smooth = Plan {
                filter: Filter::Smooth,
                ..plan.clone()
            };
            if let Some(r) = expected(&smooth, &source, &before) {
                let p = worst(&reference::psnr(of, &out, &r));
                notes.push(if p > worst(&planes) {
                    format!("filter smooth ({p:.1})")
                } else {
                    format!("filter nearest (smooth {p:.1})")
                });
            }
        }
        let threshold = if odd {
            28.0
        } else if plan.scales() || case.crop.0.fract() != 0.0 || case.crop.1.fract() != 0.0 {
            30.0
        } else {
            34.0
        };
        if worst(&planes) < threshold {
            let mut best = (f64::NEG_INFINITY, 0);
            for bits in 0..8 {
                let turned = Plan {
                    orientation: Orientation::from_bits(bits),
                    ..plan.clone()
                };
                if let Some(r) = expected(&turned, &source, &before) {
                    let p = worst(&reference::psnr(of, &out, &r));
                    if p > best.0 {
                        best = (p, bits);
                    }
                }
            }
            notes.push(format!("best orientation {} ({:.1})", best.1, best.0));
        }
        // Bytes outside what the window covers must be as they were.
        let d = plan.destination_rect;
        let window = Window {
            x: d.x as usize,
            y: d.y as usize,
            width: d.width as usize,
            height: d.height as usize,
        };
        let mut mask = vec![false; out.len()];
        for run in covered(of.format, window) {
            if let Some(span) = of.planes.get(run.plane) {
                let at = span.offset + run.row * span.bytes_per_row + run.start;
                for m in mask.iter_mut().skip(at).take(run.length) {
                    *m = true;
                }
            }
        }
        let changed = out
            .iter()
            .zip(&before)
            .zip(&mask)
            .filter(|((a, b), inside)| !**inside && a != b)
            .count();
        if changed > 0 {
            notes.push(format!("{changed} bytes outside the window changed"));
        }
        let ok = worst(&planes) >= threshold && changed == 0;
        let notes = if notes.is_empty() {
            String::new()
        } else {
            format!(", {}", notes.join(", "))
        };
        let (largest, differ) = largest_difference(of, &out, &reference);
        println!(
            "{head}: {ms:.1} ms, psnr {} dB (worst {:.1}, at least {threshold:.0}), largest \
             difference {largest} ({differ} samples){notes}; {}",
            planes_text(&planes),
            worst(&planes),
            if ok { "ok" } else { "bad" }
        );
        if ok { Outcome::Ok } else { Outcome::Bad }
    }

    /// A 24-bit BMP of an RGB picture, for looking at a case.
    fn bmp(picture: &reference::Picture) -> Vec<u8> {
        let (w, h) = (picture.width, picture.height);
        let row = (w * 3).div_ceil(4) * 4;
        let size = 54 + row * h;
        let mut out = Vec::with_capacity(size);
        out.extend_from_slice(b"BM");
        out.extend_from_slice(&(size as u32).to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&54u32.to_le_bytes());
        out.extend_from_slice(&40u32.to_le_bytes());
        out.extend_from_slice(&(w as i32).to_le_bytes());
        out.extend_from_slice(&(-(h as i32)).to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&24u16.to_le_bytes());
        out.extend_from_slice(&[0; 24]);
        for y in 0..h {
            for x in 0..w {
                let p = picture.pixels[y * w + x];
                for c in [2, 1, 0] {
                    out.push((p[c].clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            out.resize(54 + (y + 1) * row, 0);
        }
        out
    }

    fn bench(accelerator: &mut dyn Accelerator) {
        use ScalerFormat::*;
        let benches = [
            Case::new("1080p 420v to 720p BGRA", Nv12VideoRange, Bgra)
                .sizes((1920, 1080), (1280, 720)),
            Case::new("2160p 420v to 1080p 420v", Nv12VideoRange, Nv12VideoRange)
                .sizes((3840, 2160), (1920, 1080)),
            Case::new("720p BGRA to 720p 420v", Bgra, Nv12VideoRange)
                .sizes((1280, 720), (1280, 720)),
            Case::new("1080p BGRA turned 90", Bgra, Bgra)
                .sizes((1920, 1080), (1080, 1920))
                .transform(4),
            Case::new("2160p 420v to 2160p BGRA", Nv12VideoRange, Bgra)
                .sizes((3840, 2160), (3840, 2160)),
        ];
        for case in benches {
            let src = surface(case.src, case.src_size, None);
            let dst = surface(case.dst, case.dst_size, Some([0.5, 0.5, 0.5, 1.0]));
            let transform = transform_of(&case);
            let Ok(plan) = Plan::new(&transform, &src.layout, &dst.layout) else {
                continue;
            };
            let mut times = Vec::new();
            for _ in 0..21 {
                let mut job = TransformJob {
                    transform: &transform,
                    plan: plan.clone(),
                    source: &src,
                    destination: &dst,
                    hold: GuestHold::nothing(),
                    inputs: Default::default(),
                    outputs: Default::default(),
                };
                let started = StdRuntime.now();
                let status = accelerator.transform(&mut job);
                if status != io_return::SUCCESS {
                    println!("bench {}: status {status:#x}", case.name);
                    break;
                }
                times.push(StdRuntime.elapsed_since(started).as_secs_f64() * 1000.0);
            }
            if times.is_empty() {
                continue;
            }
            let first = times[0];
            let mut rest = times[1..].to_vec();
            rest.sort_by(f64::total_cmp);
            let median = rest.get(rest.len() / 2).copied().unwrap_or(first);
            println!(
                "bench {}: first {first:.1} ms, median of {} {median:.1} ms",
                case.name,
                rest.len()
            );
        }
    }

    /// Runs every case, or those whose names start with one of `only`,
    /// through an accelerator of `scaler`, after its capabilities, and
    /// with `bench` its timings; `label` names the device in the result
    /// line. Whether every case that ran matched the reference, and at
    /// least one did. A refusal (`kIOReturnUnsupported`) is no failure
    /// where Apple's scaler refuses too: a pair of formats it does not
    /// convert, which the plan refuses, and a side under the capabilities'
    /// smallest. Any other refusal is a failure, and a scaler that refuses
    /// every case, or a selection with no case, checks nothing.
    pub fn check(scaler: &dyn ScalerBackend, label: &str, only: &[String], do_bench: bool) -> bool {
        let mut accelerator = match scaler.create_accelerator() {
            Ok(a) => a,
            Err(e) => {
                println!("result: {label} no accelerator: {e}");
                return false;
            }
        };
        let caps = accelerator.capabilities();
        println!(
            "caps: classes {:?}, source {}..{:?}, destination {}..{:?}, scale {:?}, version {}",
            caps.formats,
            caps.min_source,
            caps.max_source,
            caps.min_destination,
            caps.max_destination,
            caps.max_scale,
            caps.scaler_version
        );
        if only.is_empty() || only.iter().any(|p| p == "siting") {
            for measured in super::siting::measure(accelerator.as_mut()) {
                println!("{}", measured.line());
            }
        }
        if only.is_empty() || only.iter().any(|p| p == "map") {
            for mapped in super::mapping::measure_all(accelerator.as_mut()) {
                println!("{}", mapped.line());
            }
        }
        let before = crate::engines::busy();
        let (mut ok, mut bad, mut refused) = (0, 0, 0);
        let started = StdRuntime.now();
        for case in cases() {
            if !only.is_empty() && !only.iter().any(|p| case.name.starts_with(p.as_str())) {
                continue;
            }
            let smallest = caps.min_source.max(caps.min_destination);
            match run_case(accelerator.as_mut(), &case, smallest) {
                Outcome::Ok => ok += 1,
                Outcome::Bad => bad += 1,
                Outcome::Refused => refused += 1,
            }
        }
        let wall = StdRuntime.elapsed_since(started).as_secs_f64() * 1000.0;
        let busy: Vec<String> = crate::engines::spent(&before, &crate::engines::busy())
            .into_iter()
            .map(|(engine, ns)| format!("{engine} {:.1} ms", ns as f64 / 1e6))
            .collect();
        println!("engines: {} over {wall:.0} ms of cases", busy.join(", "));
        if do_bench {
            bench(accelerator.as_mut());
        }
        let passed = bad == 0 && ok > 0;
        println!(
            "result: {label} {} ({ok} of {} cases, {refused} refused)",
            if passed { "yes" } else { "no" },
            ok + bad
        );
        passed
    }
}
