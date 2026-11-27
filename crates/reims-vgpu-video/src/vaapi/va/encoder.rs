//! Encoding on VA-API's `EncSlice` entrypoint: H.264 and HEVC, with the
//! parameter sets and slice headers ours, as packed headers.
//! [`VideoEncoder`] is [`crate::codec::encode`]'s core over
//! [`accelerator::VaAccelerator`].
//!
//! A dropped encoder's session stays open until a later encoder on the
//! render node has coded a picture and is about to code its next
//! (`accelerator`'s docs). On radeonsi, an encode session begun on the VCN
//! after the last one was closed, with none open between, hangs the VCN at
//! its first picture, faulting on a write to an address not mapped, and the
//! kernel resets the ring, or the whole GPU where that fails. A pause of 3 s
//! after the close does not avoid it, nor does a display of the new
//! session's own while the process holds another display of the GPU, whose
//! address space it shares. Sessions never closed, or each closed once the
//! next has coded its first picture, do not hang. (Measured with a libva
//! program of our own, Mesa 26.2.6, Linux 7.3.1: the hang on a Radeon 780M
//! of Phoenix and one of Hawk Point, within 2 to 4 sessions; 40 sessions in
//! a row each way without it on the Phoenix one.) iHD (Iris Xe) codes 90
//! sessions in a row, each closed before the next.
//!
//! # Imported inputs
//!
//! An input imported for another device to write frames into
//! (`VideoEncoder::import_input`) is laid out as the driver lays out its
//! own input surfaces where those are in Intel's Y tiles, and linear
//! otherwise (`VideoEncoder::input_layout`). iHD codes a frame from a linear
//! import differently from the same frame uploaded into a surface of its
//! own, and as the upload does from a Y-tiled import laid out as its own.
//! (Measured with a libva program of our own on an Iris Xe of Raptor Lake-P,
//! iHD 26.3.5: its own inputs are in Y tiles, `I915_FORMAT_MOD_Y_TILED`, in
//! one object, the luma plane at its first byte, the pitch the coded row's
//! bytes rounded up to 128, the chroma plane at the pitch times the coded
//! height rounded up to 32 rows, as at 384 and 86016 for 336x208 NV12. At
//! 320x192, 336x208, 640x360, 1280x720 and 1920x1080 in H.264 Constrained
//! Baseline, Main and High and HEVC Main and Main 10, 6 frames from a Y-tiled
//! import so laid out coded as their upload did, byte for byte, wherever the
//! upload coded the same bytes twice: everywhere but HEVC Main at 640x360
//! and 1920x1080, where two uploads differ too. Linear imports, at the row's
//! bytes rounded up to 64, 128 or 256 and the chroma plane right after the
//! coded rows, differed from the upload in H.264's P pictures at 640x360 and
//! smaller, and in every HEVC Main 10 picture, intra ones too; at the
//! driver's own pitch and offsets, in every H.264 picture at 1280x720 and
//! smaller; at a pitch not a multiple of 64, even in intra pictures. P010
//! samples are in the upper 10 bits: in the lower, the pictures came out
//! near blank.) radeonsi's own inputs are in AMD's 64 KiB swizzle, which no
//! import here declares; a linear import at rows 256 bytes apart codes as
//! its upload does there (a Radeon 780M, Mesa 26.2.6). Intel's Tile4, which
//! later Intel GPUs' drivers may give their own, is not measured, and an
//! input there is linear.

mod accelerator;
mod caps;
mod params;
mod surfaces;

#[cfg(unix)]
use std::os::fd::BorrowedFd;
use std::sync::Arc;

pub use caps::{EncoderCapabilities, Feature, HevcBlockSizes, HevcFeatures};
pub use surfaces::UploadMethod;

use super::caps::{Entrypoint, Profile};
use super::display::{Display, Error};
use super::export::SurfaceExport;
use super::ffi::{self, encode as enc};
#[cfg(unix)]
use super::target::TargetPlane;
use super::target::{Target, TargetFormat, TargetShape, TargetTiling};
use super::target_check::ALIGNMENT as TARGET_ALIGNMENT;
use super::wait::{self, Deadline};
use crate::codec::encode::encoder::{DriverLimits, Encoder};
use crate::codec::encode::{
    self, Codec, Coded, EncoderSettings, FrameRate, InputFormat, InputFrame, RateControl,
    Unsupported, hevc,
};
use accelerator::{Input, VaAccelerator};

impl From<Unsupported> for Error {
    fn from(e: Unsupported) -> Self {
        Self::Unsupported(e.0)
    }
}

/// The VA profiles that encode a profile, best first: a Constrained
/// Baseline stream is within Main's and High's tools too.
fn va_profiles(profile: encode::Profile) -> &'static [Profile] {
    const CB: Profile = Profile(ffi::VAProfileH264ConstrainedBaseline);
    const MAIN: Profile = Profile(ffi::VAProfileH264Main);
    const HIGH: Profile = Profile(ffi::VAProfileH264High);
    const HEVC_MAIN: Profile = Profile(enc::VAProfileHEVCMain);
    const HEVC_MAIN10: Profile = Profile(enc::VAProfileHEVCMain10);
    match profile {
        encode::Profile::H264ConstrainedBaseline => &[CB, MAIN, HIGH],
        encode::Profile::H264Main => &[MAIN, HIGH],
        encode::Profile::H264High => &[HIGH],
        encode::Profile::HevcMain => &[HEVC_MAIN],
        encode::Profile::HevcMain10 => &[HEVC_MAIN10],
    }
}

/// The encode entrypoints, full first, then low power.
const ENTRYPOINTS: [Entrypoint; 2] = [
    Entrypoint(enc::VAEntrypointEncSlice),
    Entrypoint(enc::VAEntrypointEncSliceLP),
];

/// The packed headers the encoder writes: parameter sets and slice
/// headers.
const PACKED_HEADERS: u32 = enc::VA_ENC_PACKED_HEADER_SEQUENCE
    | enc::VA_ENC_PACKED_HEADER_PICTURE
    | enc::VA_ENC_PACKED_HEADER_SLICE;

/// One profile the driver encodes, and its limits.
#[derive(Clone, Debug)]
pub struct EncoderSupport {
    /// The profile.
    pub profile: encode::Profile,
    /// The VA profile and entrypoint that encode it.
    pub capabilities: EncoderCapabilities,
    /// The largest picture.
    pub max_width: u32,
    /// The tallest picture.
    pub max_height: u32,
}

/// The driver's encoder for `profile`: the first VA profile and entrypoint
/// it has, with their attributes.
fn find(display: &Display, profile: encode::Profile) -> Result<Option<EncoderCapabilities>, Error> {
    for &va_profile in va_profiles(profile) {
        for entrypoint in ENTRYPOINTS {
            if let Some(caps) = EncoderCapabilities::query(display, va_profile, entrypoint)? {
                let rt = caps.rt_formats.unwrap_or(0);
                let wanted = surfaces::va_format(profile.input_format()).0;
                if rt & wanted != 0 {
                    return Ok(Some(caps));
                }
            }
        }
    }
    Ok(None)
}

/// The profiles the driver encodes, of the five the encoders write.
pub fn encoder_support(display: &Display) -> Vec<EncoderSupport> {
    [
        encode::Profile::H264ConstrainedBaseline,
        encode::Profile::H264Main,
        encode::Profile::H264High,
        encode::Profile::HevcMain,
        encode::Profile::HevcMain10,
    ]
    .into_iter()
    .filter_map(|profile| {
        let capabilities = find(display, profile).ok()??;
        let max_width = capabilities.max_width.unwrap_or(0);
        let max_height = capabilities.max_height.unwrap_or(0);
        Some(EncoderSupport {
            profile,
            capabilities,
            max_width,
            max_height,
        })
    })
    .collect()
}

/// HEVC's tools from the driver's features and block sizes: what it
/// requires, and what it allows that the encoder wants. A tool it requires
/// that the headers cannot say is refused.
fn hevc_tools(caps: &EncoderCapabilities) -> Result<hevc::Tools, Unsupported> {
    let features = caps
        .hevc_feature_set()
        .ok_or_else(|| Unsupported("the driver reports no HEVC features".into()))?;
    let sizes = caps
        .hevc_block_size_set()
        .ok_or_else(|| Unsupported("the driver reports no HEVC block sizes".into()))?;
    let required = [
        ("separate colour planes", features.separate_colour_planes),
        ("scaling lists", features.scaling_lists),
        ("PCM", features.pcm),
        ("temporal MVP", features.temporal_mvp),
        ("dependent slices", features.dependent_slices),
        ("sign data hiding", features.sign_data_hiding),
        ("weighted prediction", features.weighted_prediction),
        ("transquant bypass", features.transquant_bypass),
    ];
    for (name, feature) in required {
        if feature == Feature::Required {
            return Err(Unsupported(format!("the driver requires {name}")));
        }
    }
    Ok(hevc::Tools {
        log2_ctb_size: 3 + sizes.log2_max_coding_tree_block_size_minus3,
        log2_min_cb_size: 3 + sizes.log2_min_luma_coding_block_size_minus3,
        log2_min_tb_size: 2 + sizes.log2_min_luma_transform_block_size_minus2,
        log2_max_tb_size: 2 + sizes.log2_max_luma_transform_block_size_minus2,
        max_transform_hierarchy_depth_inter: sizes.max_max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra: sizes.max_max_transform_hierarchy_depth_intra,
        amp: features.amp.resolve(true),
        sao: features.sao.resolve(true),
        strong_intra_smoothing: features.strong_intra_smoothing.resolve(true),
        transform_skip: features.transform_skip.resolve(false),
        cu_qp_delta: features.cu_qp_delta.resolve(true),
        cu_qp_delta_required: features.cu_qp_delta == Feature::Required,
        constrained_intra_pred: features.constrained_intra_pred.resolve(false),
    })
}

fn limits(
    display: &Display,
    caps: &EncoderCapabilities,
    codec: Codec,
) -> Result<DriverLimits, Error> {
    let rc = caps.rate_control.unwrap_or(0);
    let (max_l0, max_l1) = caps.max_references();
    let surfaces = display.surface_capabilities(caps.profile, caps.entrypoint)?;
    Ok(DriverLimits {
        constant_qp: rc & enc::VA_RC_CQP != 0,
        constant_bit_rate: rc & enc::VA_RC_CBR != 0,
        variable_bit_rate: rc & enc::VA_RC_VBR != 0,
        quality_variable_bit_rate: rc & enc::VA_RC_QVBR != 0,
        // Ours: VA-API has no way to ask. radeonsi passes the HRD buffer it
        // is given, and our rate control holds it on any driver that takes
        // a constant QP.
        keeps_buffer: false,
        max_l0,
        max_l1,
        min_width: surfaces.min_width.unwrap_or(1),
        min_height: surfaces.min_height.unwrap_or(1),
        max_width: caps.max_width.or(surfaces.max_width).unwrap_or(u32::MAX),
        max_height: caps.max_height.or(surfaces.max_height).unwrap_or(u32::MAX),
        hevc_tools: match codec {
            Codec::Hevc => Some(hevc_tools(caps)?),
            Codec::H264 => None,
        },
    })
}

/// An H.264 or HEVC encoder on a VA display.
///
/// Frames go in in display order, as host-memory NV12 or P010 planes;
/// samples come out in decode order, as length-prefixed NAL units with
/// their time stamps and sync flags, and the configuration record with the
/// first.
pub struct VideoEncoder {
    inner: Encoder<VaAccelerator>,
}

// Encoders move between threads: libva is thread-safe (va.h,
// "Multithreading Guide"), and each encoder is used through `&mut`.
const _: fn() = || {
    fn send<T: Send>() {}
    send::<VideoEncoder>();
};

impl VideoEncoder {
    /// An encoder for `settings`, refused with [`Error::Unsupported`] when
    /// the driver lacks the profile, size or rate control they ask for.
    pub fn new(display: &Display, settings: EncoderSettings) -> Result<Self, Error> {
        let caps = find(display, settings.profile)?.ok_or_else(|| {
            Error::Unsupported(format!("the driver does not encode {:?}", settings.profile))
        })?;
        let limits = limits(display, &caps, settings.profile.codec())?;
        let packed = caps.packed_headers.unwrap_or(0) & PACKED_HEADERS;
        let accelerator = VaAccelerator::new(display, caps.profile, caps.entrypoint, packed);
        Ok(Self {
            inner: Encoder::new(accelerator, settings, &limits)?,
        })
    }

    /// Encodes one frame, and returns each frame that comes out, in decode
    /// order, with its own result: none while it waits to be a B picture,
    /// several when it ends a run of them. `Err` is a frame not taken.
    pub fn encode(&mut self, frame: &InputFrame<'_>) -> Result<Vec<Coded<Error>>, Error> {
        // One deadline for the call's driver calls (`super::wait`).
        wait::by(&Deadline::default(), || self.inner.encode(frame))
    }

    /// [`Self::encode`] of `frame` in place of the last frame encoded, whose
    /// sample the caller drops: the same frame again, coded as the last one
    /// was, from the references it was coded from, at the QP the settings give
    /// now ([`Encoder::encode_in_place`]). `Err` is a frame not taken, after
    /// which the next frame is coded as an IDR picture.
    pub fn encode_in_place(&mut self, frame: &InputFrame<'_>) -> Result<Vec<Coded<Error>>, Error> {
        wait::by(&Deadline::default(), || self.inner.encode_in_place(frame))
    }

    /// Starts the stream again where the caller dropped the last frame's
    /// sample ([`Encoder::start_again`]): the next frame is coded as an IDR
    /// picture.
    pub fn start_again(&mut self) {
        self.inner.start_again();
    }

    /// Encodes the frames still waiting, and returns each with its own
    /// result. More frames may follow.
    pub fn flush(&mut self) -> Vec<Coded<Error>> {
        wait::by(&Deadline::default(), || self.inner.flush())
    }

    /// The configuration record (`avcC` or `hvcC`) of the parameter sets in
    /// use; the first sample carries it too.
    pub fn configuration(&self) -> Arc<[u8]> {
        self.inner.configuration()
    }

    /// The settings, with the changes made since.
    pub fn settings(&self) -> &EncoderSettings {
        self.inner.settings()
    }

    /// Changes the bitrate, or the QPs, from the next picture, within the
    /// same mode.
    pub fn set_rate_control(&mut self, rate_control: RateControl) -> Result<(), Error> {
        self.inner.set_rate_control(rate_control)
    }

    /// Changes the frame rate rate control expects, from the next picture.
    pub fn set_frame_rate(&mut self, frame_rate: FrameRate) -> Result<(), Error> {
        self.inner.set_frame_rate(frame_rate)
    }

    /// Changes the key-frame interval, in frames and in ticks, from the
    /// next frame.
    pub fn set_key_frame_interval(&mut self, frames: Option<u32>, duration: Option<i64>) {
        self.inner.set_key_frame_interval(frames, duration);
    }

    /// The size the hardware encodes before the SPS crops it.
    pub fn coded_size(&self) -> (u32, u32) {
        self.inner.coded_size()
    }

    /// B pictures in a run, 0 when frames keep their order.
    pub fn b_frames(&self) -> u32 {
        self.inner.b_frames()
    }

    /// The level the SPS says.
    pub fn level_idc(&self) -> u8 {
        self.inner.level_idc()
    }

    /// The VA profile and entrypoint in use.
    pub fn va_profile(&self) -> (Profile, Entrypoint) {
        let a = self.inner.accelerator();
        (a.profile(), a.entrypoint())
    }

    /// The `VA_ENC_PACKED_HEADER_*` bits the driver takes from us.
    pub fn packed_headers(&self) -> u32 {
        self.inner.accelerator().packed_headers()
    }

    /// How frames reach the driver, once the first has.
    pub fn upload_method(&self) -> Option<UploadMethod> {
        self.inner.accelerator().upload_method()
    }

    /// Uploads frames by `method` from now on.
    pub fn force_upload_method(&mut self, method: UploadMethod) {
        self.inner.accelerator_mut().force_upload_method(method);
    }

    /// A free input surface of the coded size, for another device to write the next frame into
    /// through its memory as dma-bufs ([`InputSurface::export`]), then encoded by
    /// [`Self::encode_surface`]. It goes back to its pool where it is dropped instead.
    pub fn input_surface(&mut self) -> Result<InputSurface, Error> {
        let (lent, export) = self.inner.accelerator_mut().lend_input()?;
        Ok(InputSurface { lent, export })
    }

    /// [`Self::encode`] of the frame written into `surface`, whose whole coded size holds it:
    /// the picture's rows and columns, and the margin to the coded size filled as an upload
    /// fills it, its last column and row repeated. `frame`'s times, key-frame demand and tag are
    /// taken, and its planes are not read.
    pub fn encode_surface(
        &mut self,
        surface: InputSurface,
        frame: &InputFrame<'_>,
    ) -> Result<Vec<Coded<Error>>, Error> {
        wait::by(&Deadline::default(), || {
            self.inner.encode_input(Input::Lent(surface.lent), frame)
        })
    }

    /// The shape of the encoder's input: its format, NV12 or P010, and its coded size, which an
    /// imported input holds (`import_input`, on Unix).
    pub fn input_shape(&self) -> TargetShape {
        TargetShape {
            format: match self.settings().input_format() {
                InputFormat::Nv12 => TargetFormat::Nv12,
                InputFormat::P010 => TargetFormat::P010,
            },
            coded: self.coded_size(),
        }
    }

    /// `dma_buf`, of `size` bytes, imported on the encoder's display as an input of
    /// [`Self::input_shape`] whose planes lie at `planes`, luma then chroma, as `tiling` lays them
    /// out, keeping `keep` while it lives ([`Target::import_as`]), for another device to write
    /// frames into and [`Self::encode_target`] to code. `Err` where the driver refuses the import,
    /// or the planes do not fit the dma-buf or the tiling. [`Self::input_layout`] says how to lay
    /// it out.
    #[cfg(unix)]
    pub fn import_input(
        &self,
        dma_buf: BorrowedFd<'_>,
        size: u64,
        planes: [TargetPlane; 2],
        tiling: TargetTiling,
        keep: Box<dyn Send + Sync>,
    ) -> Result<Target, Error> {
        let display = self.inner.accelerator().display();
        Target::import_as(
            display,
            dma_buf,
            size,
            self.input_shape(),
            planes,
            tiling,
            keep,
        )
    }

    /// How an input imported for the encoder ([`Self::import_input`]) lays out its planes (module
    /// docs, "Imported inputs"): as the driver lays out its own input surfaces, where it lays them
    /// out in Intel's Y tiles; linear otherwise, at a pitch of the coded row's bytes rounded up to
    /// [`TARGET_ALIGNMENT`], the chroma plane right after the luma plane's coded rows. The driver's
    /// is read from a surface made as the encoder's are, exported and let go.
    #[cfg(unix)]
    pub fn input_layout(&self) -> InputLayout {
        let shape = self.input_shape();
        let display = self.inner.accelerator().display();
        driver_layout(display, self.settings().input_format(), shape)
            .unwrap_or_else(|| InputLayout::linear(shape))
    }

    /// [`Self::encode`] of the frame written into `input`, an import of the encoder's display of
    /// [`Self::input_shape`] (`import_input`, on Unix), whose whole coded size holds it: the
    /// picture's rows and columns, and the margin to the coded size filled as an upload fills it,
    /// its last column and row repeated. `frame`'s times, key-frame demand and tag are taken, and
    /// its planes are not read. The encoder reads `input` until the frame's samples come out:
    /// with no B pictures, before this returns. `Err` is a frame not taken: an input of another
    /// display or shape, or a time stamp that does not follow the last.
    pub fn encode_target(
        &mut self,
        input: &Target,
        frame: &InputFrame<'_>,
    ) -> Result<Vec<Coded<Error>>, Error> {
        let input = self.input_of(input)?;
        wait::by(&Deadline::default(), || {
            self.inner.encode_input(input, frame)
        })
    }

    /// [`Self::encode_in_place`] of the frame written into `input`, as
    /// [`Self::encode_target`] takes it.
    pub fn encode_target_in_place(
        &mut self,
        input: &Target,
        frame: &InputFrame<'_>,
    ) -> Result<Vec<Coded<Error>>, Error> {
        let input = match self.input_of(input) {
            Ok(input) => input,
            Err(error) => {
                self.inner.start_again();
                return Err(error);
            }
        };
        wait::by(&Deadline::default(), || {
            self.inner.encode_input_in_place(input, frame)
        })
    }

    /// `input` as the encoder takes it ([`Self::encode_target`]).
    fn input_of(&self, input: &Target) -> Result<Input, Error> {
        let shape = self.input_shape();
        if !input.is_on(self.inner.accelerator().display()) || input.shape() != shape {
            return Err(Error::Unsupported(format!(
                "an input of {:?} on {} for an encoder of {shape:?}",
                input.shape(),
                if input.is_on(self.inner.accelerator().display()) {
                    "its display"
                } else {
                    "another display"
                }
            )));
        }
        Ok(Input::Imported(input.clone()))
    }
}

/// How an input imported for an encoder lays out its planes in its memory, from the luma plane's
/// first byte (`VideoEncoder::input_layout`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InputLayout {
    /// How the planes' samples lie.
    pub tiling: TargetTiling,
    /// The bytes from one row of either plane to the next.
    pub pitch: u64,
    /// The chroma plane's first byte, from the luma plane's.
    pub chroma_at: u64,
    /// The bytes the input's memory holds, from the luma plane's first byte: the planes' span,
    /// or the driver's own surface's where its planes are laid out as it lays them out.
    pub length: u64,
}

impl InputLayout {
    /// The linear layout of an input of `shape` (`VideoEncoder::input_layout`).
    pub fn linear(shape: TargetShape) -> InputLayout {
        let pitch = (u64::from(shape.coded.0) * shape.format.bytes_per_sample())
            .next_multiple_of(TARGET_ALIGNMENT);
        let rows = u64::from(shape.coded.1);
        InputLayout {
            tiling: TargetTiling::Linear,
            pitch,
            chroma_at: pitch * rows,
            length: pitch * rows + pitch * rows.div_ceil(2),
        }
    }
}

/// The layout the driver gives an input surface of `format` and `shape`'s coded size, made as
/// the encoder's pool makes its own and exported, where it lays it out in Intel's Y tiles as an
/// import in them is laid out (`Target::import_as`): both planes in one dma-buf at one pitch, the
/// luma plane from its first byte, the chroma plane a whole number of rows of tiles after it, at
/// least the coded height, and the dma-buf's bytes, taken as the layout's length, holding the
/// chroma plane's rows of tiles. `None` otherwise, and where the driver makes or exports none.
#[cfg(unix)]
fn driver_layout(
    display: &Display,
    format: InputFormat,
    shape: TargetShape,
) -> Option<InputLayout> {
    use super::target::{Y_TILE_ROWS, Y_TILE_WIDTH};
    let (width, height) = shape.coded;
    let pool = surfaces::Pool::new(display, format, width, height, 1).ok()?;
    let id = *pool.ids().first()?;
    let export = super::export::export_for_write(display, id, shape.coded).ok()?;
    let (luma, chroma) = (export.plane(0), export.plane(1));
    drop(export);
    drop(pool);
    let pitch = luma.pitch;
    let rows = chroma.offset.checked_div(pitch)?;
    let chroma_end = chroma.offset + pitch * (rows / 2).next_multiple_of(Y_TILE_ROWS);
    let tiled = TargetTiling::of_modifier(luma.modifier) == Some(TargetTiling::IntelY)
        && chroma.modifier == luma.modifier
        && chroma.object == luma.object
        && luma.offset == 0
        && chroma.pitch == pitch
        && pitch.is_multiple_of(Y_TILE_WIDTH)
        && pitch >= u64::from(width) * shape.format.bytes_per_sample()
        && chroma.offset.is_multiple_of(pitch * Y_TILE_ROWS)
        && rows >= u64::from(height)
        && luma.object_size >= chroma_end;
    tiled.then_some(InputLayout {
        tiling: TargetTiling::IntelY,
        pitch,
        chroma_at: chroma.offset,
        length: luma.object_size,
    })
}

/// An input surface of an encoder's, lent for another device to write a frame into
/// ([`VideoEncoder::input_surface`]).
pub struct InputSurface {
    lent: surfaces::Lent,
    export: Arc<SurfaceExport>,
}

impl InputSurface {
    /// The surface's memory as dma-bufs, exported for writing: the encoder's coded size, its
    /// planes where the export says, kept while the encoder's pool lives.
    pub fn export(&self) -> &Arc<SurfaceExport> {
        &self.export
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;
    use crate::codec::encode::Plane;
    use crate::vaapi::va::gated::GatedRuntime;

    /// On the driver, 24 encoders made from one display one after another, each dropped before
    /// the next, the profiles the driver encodes in turn, code each of their 4 frames: a dropped
    /// encoder's session is closed only once the next one has coded a picture (module docs).
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn encoders_made_one_after_another_from_one_display_code_every_frame() {
        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        let display = Display::open(&node, Arc::new(GatedRuntime)).expect("a VA display");
        let profiles: Vec<encode::Profile> = (encoder_support(&display).into_iter())
            .map(|s| s.profile)
            .collect();
        assert!(!profiles.is_empty(), "the driver encodes nothing");
        let (width, height) = (320, 192);
        for made in 0..24 {
            let profile = profiles[made % profiles.len()];
            let settings = EncoderSettings::new(profile, width, height);
            let mut encoder = VideoEncoder::new(&display, settings)
                .unwrap_or_else(|e| panic!("encoder {made} ({profile:?}): {e}"));
            let row = width as usize * profile.input_format().bytes_per_sample();
            let luma = vec![0x40; row * height as usize];
            let chroma = vec![0x80; row * height as usize / 2];
            let mut coded = Vec::new();
            for id in 0..4 {
                let frame = InputFrame {
                    planes: [
                        Plane {
                            data: &luma,
                            bytes_per_row: row,
                        },
                        Plane {
                            data: &chroma,
                            bytes_per_row: row,
                        },
                    ],
                    pts: 3000 * id as i64,
                    duration: 3000,
                    force_key_frame: false,
                    id,
                };
                let out = (encoder.encode(&frame))
                    .unwrap_or_else(|e| panic!("encoder {made} ({profile:?}) frame {id}: {e}"));
                coded.extend(out);
            }
            coded.extend(encoder.flush());
            let samples = (coded.into_iter())
                .map(|c| c.map_err(|f| (f.id, f.error.map(|e| e.to_string()))))
                .collect::<Result<Vec<_>, _>>()
                .unwrap_or_else(|e| panic!("encoder {made} ({profile:?}): {e:?}"));
            assert_eq!(samples.len(), 4, "encoder {made} ({profile:?})");
        }
    }

    /// Writes `plane`, `rows` rows of `row` bytes, into `bytes` from `at`, laid out as `layout`
    /// says, out to `coded_rows` rows of `coded_bytes`, as the upload fills its surface: each row
    /// past the plane's its last row, each byte past a row's its last `unit` (a sample, or a
    /// Cb/Cr pair) repeated.
    #[allow(clippy::too_many_arguments)]
    fn fill_as_uploaded(
        bytes: &mut [u8],
        at: usize,
        layout: InputLayout,
        plane: &[u8],
        (row, rows): (usize, usize),
        (coded_bytes, coded_rows): (usize, usize),
        unit: usize,
    ) {
        let pitch = layout.pitch;
        let mut put = |x: usize, y: usize, byte: u8| {
            let to = layout.tiling.byte_at(pitch, x as u64, y as u64);
            bytes[at + to as usize] = byte;
        };
        for r in 0..coded_rows {
            let from = &plane[r.min(rows - 1) * row..][..row];
            for x in 0..coded_bytes {
                let byte = if x < row {
                    from[x]
                } else {
                    from[row - unit + (x - row) % unit]
                };
                put(x, r, byte);
            }
        }
    }

    /// On the driver, for each profile it encodes, 6 frames of a moving 330x202 picture (a
    /// margin of columns and rows to the coded size) at a constant QP with no B pictures, written
    /// by the CPU into an input imported over host memory (`import_input`: a memfd made a dma-buf
    /// by udmabuf, laid out as `input_layout` says: in iHD's Y tiles, linear on radeonsi), its
    /// margin filled as an upload fills it, code as the same frames uploaded, byte for byte:
    /// two encoders of the same settings code them, one from host bytes, one from the import
    /// (`encode_target`), and every sample of the second is the first's. The one import holds
    /// each frame in turn, so no frame may wait as a B picture past the next one's write
    /// (`encode_target`'s docs).
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn a_frame_in_an_imported_input_codes_as_its_upload_byte_for_byte() {
        use super::super::host_memory::HostMemory;

        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        let display = Display::open(&node, Arc::new(GatedRuntime)).expect("a VA display");
        let profiles: Vec<encode::Profile> = (encoder_support(&display).into_iter())
            .map(|s| s.profile)
            .collect();
        assert!(!profiles.is_empty(), "the driver encodes nothing");
        let (width, height) = (330u32, 202u32);
        for profile in profiles {
            let mut settings = EncoderSettings::new(profile, width, height);
            settings.rate_control = RateControl::ConstantQp {
                intra: 26,
                predicted: 28,
                bidirectional: 30,
            };
            settings.allow_frame_reordering = false;
            let mut by_upload = VideoEncoder::new(&display, settings.clone())
                .unwrap_or_else(|e| panic!("{profile:?}: an encoder: {e}"));
            let mut by_import = VideoEncoder::new(&display, settings)
                .unwrap_or_else(|e| panic!("{profile:?}: an encoder: {e}"));
            let shape = by_import.input_shape();
            let sample = profile.input_format().bytes_per_sample();
            let (coded_width, coded_height) = (shape.coded.0 as usize, shape.coded.1 as usize);
            assert!(
                coded_width > width as usize && coded_height > height as usize,
                "{profile:?}: a coded size of {:?} leaves no margin",
                shape.coded
            );
            let layout = by_import.input_layout();
            eprintln!("{profile:?}: the input laid out as {layout:?}");
            let chroma_at = layout.chroma_at as usize;
            let mut memory = HostMemory::new(layout.length as usize)
                .unwrap_or_else(|e| panic!("host memory: {e}"));
            let planes = [
                TargetPlane {
                    offset: 0,
                    bytes_per_row: layout.pitch,
                },
                TargetPlane {
                    offset: layout.chroma_at,
                    bytes_per_row: layout.pitch,
                },
            ];
            let input = (by_import.import_input(
                memory.dma_buf_fd(),
                memory.len() as u64,
                planes,
                layout.tiling,
                Box::new(()),
            ))
            .unwrap_or_else(|e| panic!("{profile:?}: the import: {e}"));
            let (w, h) = (width as usize, height as usize);
            let row = w * sample;
            let (mut uploaded, mut imported) = (Vec::new(), Vec::new());
            for id in 0..6u64 {
                let value = |x: usize, y: usize, shift: usize| ((x + y + shift) % 220 + 16) as u16;
                let mut luma = Vec::with_capacity(row * h);
                let mut chroma = Vec::with_capacity(row * (h / 2));
                for y in 0..h {
                    for x in 0..w {
                        let v = value(x, y, id as usize * 4);
                        match sample {
                            2 => luma.extend_from_slice(&(v << 6).to_le_bytes()),
                            _ => luma.push(v as u8),
                        }
                    }
                }
                for y in 0..h / 2 {
                    for x in 0..w {
                        let v = 128 + value(x, y, id as usize) % 32;
                        match sample {
                            2 => chroma.extend_from_slice(&(v << 6).to_le_bytes()),
                            _ => chroma.push(v as u8),
                        }
                    }
                }
                let bytes = memory.bytes_mut();
                let coded_bytes = coded_width * sample;
                fill_as_uploaded(
                    bytes,
                    0,
                    layout,
                    &luma,
                    (row, h),
                    (coded_bytes, coded_height),
                    sample,
                );
                fill_as_uploaded(
                    bytes,
                    chroma_at,
                    layout,
                    &chroma,
                    (row, h / 2),
                    (coded_bytes, coded_height / 2),
                    sample * 2,
                );
                let frame = InputFrame {
                    planes: [
                        Plane {
                            data: &luma,
                            bytes_per_row: row,
                        },
                        Plane {
                            data: &chroma,
                            bytes_per_row: row,
                        },
                    ],
                    pts: 3000 * id as i64,
                    duration: 3000,
                    force_key_frame: false,
                    id,
                };
                uploaded.extend(
                    (by_upload.encode(&frame))
                        .unwrap_or_else(|e| panic!("{profile:?} frame {id}: the upload: {e}")),
                );
                imported.extend(
                    (by_import.encode_target(&input, &frame))
                        .unwrap_or_else(|e| panic!("{profile:?} frame {id}: the import: {e}")),
                );
            }
            uploaded.extend(by_upload.flush());
            imported.extend(by_import.flush());
            let data = |coded: Vec<Coded<Error>>| -> Vec<Vec<u8>> {
                (coded.into_iter())
                    .map(|c| {
                        c.unwrap_or_else(|f| panic!("{profile:?}: {:?}", f.error))
                            .data
                    })
                    .collect()
            };
            let (a, b) = (data(uploaded), data(imported));
            assert_eq!(a.len(), 6, "{profile:?}: samples");
            assert!(a == b, "{profile:?}: a sample of the import differs");
            eprintln!("{profile:?}: 6 frames from an import coded as uploaded");
        }
    }

    #[test]
    fn a_linear_input_is_laid_out_at_rows_of_256_bytes_its_chroma_plane_after_the_coded_rows() {
        let shape = |format, coded| TargetShape { format, coded };
        assert_eq!(
            InputLayout::linear(shape(TargetFormat::Nv12, (336, 208))),
            InputLayout {
                tiling: TargetTiling::Linear,
                pitch: 512,
                chroma_at: 512 * 208,
                length: 512 * (208 + 104),
            }
        );
        assert_eq!(
            InputLayout::linear(shape(TargetFormat::P010, (1920, 1081))),
            InputLayout {
                tiling: TargetTiling::Linear,
                pitch: 3840,
                chroma_at: 3840 * 1081,
                length: 3840 * (1081 + 541),
            }
        );
    }

    /// An input of another shape is not taken: a frame for an encoder of 330x202 is refused from
    /// an import of 320x192's coded size.
    #[test]
    #[ignore = "needs a VA-API GPU: run on the VA-API host"]
    fn an_input_of_another_shape_is_refused() {
        use super::super::host_memory::HostMemory;

        let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
        let display = Display::open(&node, Arc::new(GatedRuntime)).expect("a VA display");
        let support = (encoder_support(&display).into_iter())
            .find(|s| s.profile.input_format() == InputFormat::Nv12)
            .expect("an NV12 profile");
        let small = VideoEncoder::new(&display, EncoderSettings::new(support.profile, 320, 192))
            .expect("an encoder");
        let mut large =
            VideoEncoder::new(&display, EncoderSettings::new(support.profile, 330, 202))
                .expect("an encoder");
        let (coded_width, coded_height) = small.coded_size();
        let pitch = u64::from(coded_width).next_multiple_of(256);
        let chroma_at = pitch * u64::from(coded_height);
        let memory = HostMemory::new((chroma_at + pitch * u64::from(coded_height) / 2) as usize)
            .expect("host memory");
        let planes = [
            TargetPlane {
                offset: 0,
                bytes_per_row: pitch,
            },
            TargetPlane {
                offset: chroma_at,
                bytes_per_row: pitch,
            },
        ];
        let input = (small.import_input(
            memory.dma_buf_fd(),
            memory.len() as u64,
            planes,
            TargetTiling::Linear,
            Box::new(()),
        ))
        .expect("the import");
        let none = Plane {
            data: &[],
            bytes_per_row: 0,
        };
        let frame = InputFrame {
            planes: [none, none],
            pts: 0,
            duration: 3000,
            force_key_frame: false,
            id: 0,
        };
        let refused = large.encode_target(&input, &frame);
        assert!(
            matches!(refused, Err(Error::Unsupported(_))),
            "{:?}",
            refused.map(|c| c.len())
        );
    }
}
