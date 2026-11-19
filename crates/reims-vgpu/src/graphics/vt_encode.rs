//! What the video arms' VideoToolbox paravirtual encoders share (`super::vt`): the encoder list
//! of the profiles an arm encodes ([`encoders_of`]), `Quality` as the device's quality
//! ([`quality_percent`]), and a source frame's planes in the surface the guest lent
//! ([`source_planes`]), as the device's GPU reads them ([`buffer_planes`]) or the CPU does
//! ([`host_planes`]).

use reims_vgpu_video::codec::encode::{
    Codec as EncodeCodec, EncoderSettings, InputFormat, Profile,
};
use reims_vgpu_video::core::backend::BackendError as VideoError;
use reims_vgpu_video::core::device_memory::HostView;
use reims_vgpu_video::core::surfaces::PlaneLayout;
use reims_vgpu_video::paravirt::{Codec, Encoder};
use reims_vgpu_vulkan::core::num::{ToU64 as _, ToUsize as _};
use reims_vgpu_vulkan::core::platform::{Handle as _, Platform};

use super::decode::GuestSource;
use super::video_host::OutputPlane;

/// The encoders of `supports`, each a profile and the largest picture its encoder takes: one
/// for each codec of which a profile is there, with those profiles and the largest picture any
/// of them takes.
pub(super) fn encoders_of(supports: impl IntoIterator<Item = (Profile, u32, u32)>) -> Vec<Encoder> {
    let mut encoders: Vec<Encoder> = Vec::new();
    for (profile, max_width, max_height) in supports {
        let codec = match profile.codec() {
            EncodeCodec::H264 => Codec::H264,
            EncodeCodec::Hevc => Codec::Hevc,
        };
        match encoders.iter_mut().find(|e| e.codec == codec) {
            Some(encoder) => {
                encoder.profiles.push(profile);
                encoder.max_width = encoder.max_width.max(max_width);
                encoder.max_height = encoder.max_height.max(max_height);
            }
            None => encoders.push(Encoder {
                codec,
                profiles: vec![profile],
                max_width,
                max_height,
            }),
        }
    }
    encoders
}

/// `quality`, `Quality`'s 0 to 1, as the device's quality, a whole percentage: 0 the fastest to
/// 100 the best.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a float within 0 to 100 to an integer: `as` saturates, NaN to 0"
)]
pub(super) fn quality_percent(quality: f32) -> u32 {
    (quality.clamp(0.0, 1.0) * 100.0).round() as u32
}

/// The luma and chroma planes of `source`, where it holds a frame of a stream of `settings`: a
/// surface in the input format the stream's profile takes (`420v` or `420f` for 8 bits, `x420`
/// or `xf20` for 10), of two planes, luma's bytes running to chroma's first and chroma's to the
/// memory's end, each holding the stream's rows at its pitch. `Unsupported` says what does not.
pub(super) fn source_planes<P: Platform>(
    source: &GuestSource<'_, P>,
    settings: &EncoderSettings,
) -> Result<[PlaneLayout; 2], VideoError> {
    let layout = &source.layout;
    let format = match &layout.pixel_format.to_be_bytes() {
        b"420v" | b"420f" => InputFormat::Nv12,
        b"x420" | b"xf20" => InputFormat::P010,
        _ => {
            return Err(VideoError::Unsupported(format!(
                "a source frame in '{}'",
                layout.pixel_format
            )));
        }
    };
    if format != settings.input_format() {
        return Err(VideoError::Unsupported(format!(
            "a source frame in '{}' for a {:?} stream",
            layout.pixel_format, settings.profile
        )));
    }
    let [luma, chroma] = layout.planes[..] else {
        return Err(VideoError::Unsupported(format!(
            "a source of {} planes",
            layout.planes.len()
        )));
    };
    // Luma's bytes run to chroma's first, chroma's to the surface's end: each plane the
    // encoder's rows of at least the stream's size, at the plane's pitch.
    let sample = format.bytes_per_sample();
    let holds = |plane: &PlaneLayout, bytes: usize, rows: u32, row_bytes: usize| {
        let rows = rows.to_usize();
        rows <= plane.height.to_usize()
            && row_bytes <= plane.bytes_per_row
            && (plane.bytes_per_row.checked_mul(rows.saturating_sub(1)))
                .and_then(|n| n.checked_add(row_bytes))
                .is_some_and(|n| n <= bytes)
    };
    let (width, height) = (settings.width.to_usize(), settings.height);
    let length = source.memory.length();
    if !(luma.offset < chroma.offset && chroma.offset < length)
        || !holds(&luma, chroma.offset - luma.offset, height, width * sample)
        || !holds(&chroma, length - chroma.offset, height / 2, width * sample)
    {
        return Err(VideoError::Unsupported(
            "a source whose planes do not hold the stream's picture".to_owned(),
        ));
    }
    Ok([luma, chroma])
}

/// The host bytes of `planes`, a source's as [`source_planes`] gave them, in `view`, the host's
/// view of its memory: luma's from its first to chroma's first, chroma's from its first to the
/// view's end.
pub(super) fn host_planes<'v>(
    view: &'v HostView,
    planes: &[PlaneLayout; 2],
) -> Result<[&'v [u8]; 2], VideoError> {
    let [luma, chroma] = planes;
    let y_at = view.offset.checked_add(luma.offset);
    let uv_at = view.offset.checked_add(chroma.offset);
    let (Some(y_at), Some(uv_at)) = (y_at, uv_at) else {
        return Err(VideoError::Unsupported("planes past the view".to_owned()));
    };
    // SAFETY: the view's `size` bytes from `first` are the host's mapping of the surface's
    // pages, which `view.hold` keeps mapped for as long as `view`, which the bytes borrow, lives;
    // the guest's kext holds the surface lent, its pages wired, until the host's `rels`, sent
    // once the frame is encoded and emitted. Each plane is taken of the view's bytes by `get`.
    let all = unsafe { std::slice::from_raw_parts(view.first.as_ptr(), view.size) };
    let (Some(y), Some(uv)) = (all.get(y_at..uv_at), all.get(uv_at..)) else {
        return Err(VideoError::Unsupported("planes past the view".to_owned()));
    };
    Ok([y, uv])
}

/// `planes`, a source's as [`source_planes`] gave them, as the device's GPU reads them in
/// `buffer`, a Shared buffer over the surface's pages from its first byte: luma's rows of a
/// picture of `width` by `height` samples of `sample` bytes, and chroma's, half as many, each
/// of `width / 2` pairs of samples.
pub(super) fn buffer_planes<P: Platform>(
    buffer: &P::Buffer,
    planes: &[PlaneLayout; 2],
    (width, height): (u32, u32),
    sample: u32,
) -> [OutputPlane<P>; 2] {
    // A plane of `w` samples (a chroma plane's of two each) by `h` rows.
    let plane = |plane: &PlaneLayout, w: u32, h: u32, samples: u32| OutputPlane {
        buffer: buffer.clone_ref(),
        offset: plane.offset.to_u64(),
        bytes_per_row: plane.bytes_per_row.to_u64(),
        row_bytes: u64::from(w) * u64::from(samples * sample),
        width: u64::from(w),
        height: u64::from(h),
        rows_held: plane.height.into(),
        cached: false,
        host: None,
        pages: None,
        cache: None,
        #[cfg(unix)]
        surface: None,
    };
    [
        plane(&planes[0], width, height, 1),
        plane(&planes[1], width / 2, height / 2, 2),
    ]
}
