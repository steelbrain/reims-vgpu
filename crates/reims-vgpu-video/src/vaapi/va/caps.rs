//! What a VA driver says it can do: its profiles and entrypoints, and for
//! each pair the render-target formats, sizes, surface formats and, for
//! video processing, filters.
//!
//! A display keeps the driver's first answer to each of these queries
//! ([`Answers`]), and [`Display::open`] asks them all on the thread that
//! opens it ([`Display::prime`]). A session, on a thread of its own, then
//! reads what the driver said there, and never asks it again: NVIDIA's
//! driver answers some of them only on the opening thread.

use core::ffi::c_uint;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::core::sync::MutexGuard;

use super::display::{Display, Error, check};
use super::ffi;
use super::ffi::encode as enc;

/// A `VAProfile`. Drivers may report profiles newer than this crate, so
/// any value is kept, and [`Profile::name`] names the ones va.h 1.24 has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Profile(pub i32);

impl Profile {
    /// `VAProfileNone`: video processing's profile.
    pub const NONE: Self = Self(ffi::VAProfileNone);
    /// `VAProfileH264ConstrainedBaseline`.
    pub const H264_CONSTRAINED_BASELINE: Self = Self(ffi::VAProfileH264ConstrainedBaseline);
    /// `VAProfileH264Main`.
    pub const H264_MAIN: Self = Self(ffi::VAProfileH264Main);
    /// `VAProfileH264High`.
    pub const H264_HIGH: Self = Self(ffi::VAProfileH264High);
    /// `VAProfileHEVCMain`.
    pub const HEVC_MAIN: Self = Self(ffi::VAProfileHEVCMain);
    /// `VAProfileHEVCMain10`.
    pub const HEVC_MAIN10: Self = Self(ffi::VAProfileHEVCMain10);

    /// va.h's name for the profile, without its `VAProfile` prefix.
    pub fn name(self) -> Option<&'static str> {
        let index = usize::try_from(self.0.checked_add(1)?).ok()?;
        PROFILE_NAMES.get(index).copied().filter(|n| !n.is_empty())
    }
}

/// va.h's `VAProfile` names from -1 (`None`) to 40.
const PROFILE_NAMES: [&str; 42] = [
    "None",
    "MPEG2Simple",
    "MPEG2Main",
    "MPEG4Simple",
    "MPEG4AdvancedSimple",
    "MPEG4Main",
    "", // 5: VAProfileH264Baseline, removed from va.h
    "H264Main",
    "H264High",
    "VC1Simple",
    "VC1Main",
    "VC1Advanced",
    "H263Baseline",
    "JPEGBaseline",
    "H264ConstrainedBaseline",
    "VP8Version0_3",
    "H264MultiviewHigh",
    "H264StereoHigh",
    "HEVCMain",
    "HEVCMain10",
    "VP9Profile0",
    "VP9Profile1",
    "VP9Profile2",
    "VP9Profile3",
    "HEVCMain12",
    "HEVCMain422_10",
    "HEVCMain422_12",
    "HEVCMain444",
    "HEVCMain444_10",
    "HEVCMain444_12",
    "HEVCSccMain",
    "HEVCSccMain10",
    "HEVCSccMain444",
    "AV1Profile0",
    "AV1Profile1",
    "HEVCSccMain444_10",
    "Protected",
    "H264High10",
    "VVCMain10",
    "VVCMultilayerMain10",
    "AV1Profile2",
    "H264High422",
];

/// A `VAEntrypoint`, any value kept as for [`Profile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Entrypoint(pub i32);

impl Entrypoint {
    /// `VAEntrypointVLD`: decoding.
    pub const VLD: Self = Self(ffi::VAEntrypointVLD);
    /// `VAEntrypointVideoProc`: scaling and colour conversion.
    pub const VIDEO_PROC: Self = Self(ffi::VAEntrypointVideoProc);

    /// va.h's name for the entrypoint, without its `VAEntrypoint` prefix.
    pub fn name(self) -> Option<&'static str> {
        Some(match self.0 {
            1 => "VLD",
            2 => "IZZ",
            3 => "IDCT",
            4 => "MoComp",
            5 => "Deblocking",
            6 => "EncSlice",
            7 => "EncPicture",
            8 => "EncSliceLP",
            10 => "VideoProc",
            11 => "FEI",
            12 => "Stats",
            13 => "ProtectedTEEComm",
            14 => "ProtectedContent",
            _ => return None,
        })
    }

    /// Whether the entrypoint encodes.
    pub fn is_encode(self) -> bool {
        matches!(self.0, 6..=8 | 11)
    }
}

/// A FOURCC code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fourcc(pub u32);

impl Fourcc {
    /// `VA_FOURCC_NV12`.
    pub const NV12: Self = Self(ffi::VA_FOURCC_NV12);
}

impl std::fmt::Display for Fourcc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0.to_le_bytes() {
            let c = if byte.is_ascii_graphic() || byte == b' ' {
                char::from(byte)
            } else {
                '?'
            };
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

/// The names of the `VA_RT_FORMAT_*` bits set in `formats` (va.h).
pub fn rt_format_names(formats: u32) -> Vec<&'static str> {
    const BITS: [(u32, &str); 16] = [
        (0x0000_0001, "YUV420"),
        (0x0000_0002, "YUV422"),
        (0x0000_0004, "YUV444"),
        (0x0000_0008, "YUV411"),
        (0x0000_0010, "YUV400"),
        (0x0000_0100, "YUV420_10"),
        (0x0000_0200, "YUV422_10"),
        (0x0000_0400, "YUV444_10"),
        (0x0000_1000, "YUV420_12"),
        (0x0000_2000, "YUV422_12"),
        (0x0000_4000, "YUV444_12"),
        (0x0001_0000, "RGB16"),
        (0x0002_0000, "RGB32"),
        (0x0010_0000, "RGBP"),
        (0x0020_0000, "RGB32_10"),
        (0x8000_0000, "PROTECTED"),
    ];
    BITS.iter()
        .filter(|(bit, _)| formats & bit != 0)
        .map(|&(_, name)| name)
        .collect()
}

/// va_vpp.h's name for a `VAProcFilterType`.
pub fn proc_filter_name(filter: i32) -> Option<&'static str> {
    Some(match filter {
        1 => "NoiseReduction",
        2 => "Deinterlacing",
        3 => "Sharpening",
        4 => "ColorBalance",
        5 => "SkinToneEnhancement",
        6 => "TotalColorCorrection",
        7 => "HVSNoiseReduction",
        8 => "HighDynamicRangeToneMapping",
        9 => "3DLUT",
        _ => return None,
    })
}

/// What the driver reports for one profile and entrypoint.
#[derive(Clone, Debug)]
pub struct ConfigCapabilities {
    /// The profile.
    pub profile: Profile,
    /// The entrypoint.
    pub entrypoint: Entrypoint,
    /// `VAConfigAttribRTFormat`: `VA_RT_FORMAT_*` bits.
    pub rt_formats: Option<u32>,
    /// `VAConfigAttribMaxPictureWidth`.
    pub max_width: Option<u32>,
    /// `VAConfigAttribMaxPictureHeight`.
    pub max_height: Option<u32>,
    /// `VAConfigAttribDecSliceMode` (decoding).
    pub dec_slice_mode: Option<u32>,
    /// `VAConfigAttribRateControl`: `VA_RC_*` bits (encoding).
    pub rate_control: Option<u32>,
    /// `VAConfigAttribEncPackedHeaders` (encoding).
    pub packed_headers: Option<u32>,
    /// `VAConfigAttribEncMaxRefFrames` (encoding): list 0 in the low 16
    /// bits, list 1 in the high.
    pub max_ref_frames: Option<u32>,
    /// `VAConfigAttribEncMaxSlices` (encoding).
    pub max_slices: Option<u32>,
    /// The surface attributes of a config of this pair, or the error
    /// making one gave.
    pub surfaces: Result<SurfaceCapabilities, String>,
    /// For video processing, `vaQueryVideoProcFilters`.
    pub proc_filters: Option<Result<Vec<i32>, String>>,
}

/// `vaQuerySurfaceAttributes` of one config.
#[derive(Clone, Debug, Default)]
pub struct SurfaceCapabilities {
    /// `VASurfaceAttribPixelFormat`, one per format.
    pub pixel_formats: Vec<Fourcc>,
    /// `VASurfaceAttribMinWidth`.
    pub min_width: Option<u32>,
    /// `VASurfaceAttribMaxWidth`.
    pub max_width: Option<u32>,
    /// `VASurfaceAttribMinHeight`.
    pub min_height: Option<u32>,
    /// `VASurfaceAttribMaxHeight`.
    pub max_height: Option<u32>,
    /// `VASurfaceAttribMemoryType`: `VA_SURFACE_ATTRIB_MEM_TYPE_*` bits.
    pub memory_types: Option<u32>,
}

/// Everything a driver reports, as [`Display::capabilities`] reads it.
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// `vaQueryVendorString`.
    pub vendor: String,
    /// The VA-API version.
    pub api_version: (i32, i32),
    /// Each profile and entrypoint pair, in the driver's order.
    pub configs: Vec<ConfigCapabilities>,
    /// `vaQueryImageFormats`.
    pub image_formats: Vec<Fourcc>,
}

impl Capabilities {
    /// The pair's capabilities, if the driver has it.
    pub fn find(&self, profile: Profile, entrypoint: Entrypoint) -> Option<&ConfigCapabilities> {
        self.configs
            .iter()
            .find(|c| c.profile == profile && c.entrypoint == entrypoint)
    }
}

/// The attributes [`ConfigCapabilities`] keeps, in the order it asks.
const ATTRIBUTES: [ffi::VAConfigAttribType; 8] = [
    ffi::VAConfigAttribRTFormat,
    ffi::VAConfigAttribMaxPictureWidth,
    ffi::VAConfigAttribMaxPictureHeight,
    ffi::VAConfigAttribDecSliceMode,
    ffi::VAConfigAttribRateControl,
    ffi::VAConfigAttribEncPackedHeaders,
    ffi::VAConfigAttribEncMaxRefFrames,
    ffi::VAConfigAttribEncMaxSlices,
];

/// Every attribute any query of this crate asks of a pair: [`ATTRIBUTES`],
/// the decoders', and the encoders' (`encoder::caps`),
/// so that [`Display::prime`] has the answer to each before a session asks.
const ASKED_ATTRIBUTES: [ffi::VAConfigAttribType; 18] = [
    ffi::VAConfigAttribRTFormat,
    ffi::VAConfigAttribMaxPictureWidth,
    ffi::VAConfigAttribMaxPictureHeight,
    ffi::VAConfigAttribDecSliceMode,
    ffi::VAConfigAttribRateControl,
    ffi::VAConfigAttribEncPackedHeaders,
    ffi::VAConfigAttribEncMaxRefFrames,
    ffi::VAConfigAttribEncMaxSlices,
    enc::VAConfigAttribEncSliceStructure,
    enc::VAConfigAttribEncQualityRange,
    enc::VAConfigAttribEncIntraRefresh,
    enc::VAConfigAttribEncSkipFrame,
    enc::VAConfigAttribEncROI,
    enc::VAConfigAttribEncRateControlExt,
    enc::VAConfigAttribMaxFrameSize,
    enc::VAConfigAttribPredictionDirection,
    enc::VAConfigAttribEncHEVCFeatures,
    enc::VAConfigAttribEncHEVCBlockSizes,
];

/// The driver's answers to a display's queries, each kept from the first
/// time it answered (module doc). Only answers are kept: a query that
/// failed is asked again.
#[derive(Default)]
pub(crate) struct Answers {
    /// `vaQueryConfigProfiles`.
    profiles: Option<Vec<Profile>>,
    /// `vaQueryConfigEntrypoints`, by profile.
    entrypoints: HashMap<Profile, Vec<Entrypoint>>,
    /// `vaGetConfigAttributes`' raw value, by pair and attribute.
    attributes: HashMap<(Profile, Entrypoint, ffi::VAConfigAttribType), u32>,
    /// `vaQueryImageFormats`.
    image_formats: Option<Vec<ffi::VAImageFormat>>,
    /// `vaQuerySurfaceAttributes` of a config of the pair.
    surfaces: HashMap<(Profile, Entrypoint), SurfaceCapabilities>,
}

impl Display {
    /// The answers kept so far. Held only to read or store one, never
    /// across a call to the driver.
    fn answers(&self) -> MutexGuard<'_, Answers> {
        self.answers_lock().lock()
    }

    /// Asks the driver, on this thread, every query a session may ask
    /// later: the profiles, each one's entrypoints, every pair's
    /// [`ASKED_ATTRIBUTES`] and surface attributes, and the image formats.
    /// [`Display::open`] calls it, so the answers are the opening thread's.
    /// A query that fails here is left to be asked again where it is next
    /// needed, which says why.
    pub(crate) fn prime(&self) {
        let Ok(profiles) = self.profiles() else {
            return;
        };
        for profile in profiles {
            let Ok(entrypoints) = self.entrypoints(profile) else {
                continue;
            };
            for entrypoint in entrypoints {
                if self
                    .config_attributes(profile, entrypoint, &ASKED_ATTRIBUTES)
                    .is_err()
                {
                    // A driver that refuses one of them all at once may
                    // still answer each alone.
                    for attribute in ASKED_ATTRIBUTES {
                        self.config_attributes(profile, entrypoint, &[attribute])
                            .ok();
                    }
                }
                self.surface_capabilities(profile, entrypoint).ok();
            }
        }
        self.image_format_list().ok();
    }

    /// The profiles the driver reports (`vaQueryConfigProfiles`).
    pub fn profiles(&self) -> Result<Vec<Profile>, Error> {
        if let Some(list) = self.answers().profiles.clone() {
            return Ok(list);
        }
        // SAFETY: an initialised display.
        let max = self.call(|raw| unsafe { ffi::vaMaxNumProfiles(raw) })?;
        let mut list = vec![0; listed(max)];
        let mut count = 0;
        if !list.is_empty() {
            check(
                "vaQueryConfigProfiles",
                // SAFETY: `list` holds vaMaxNumProfiles entries, as va.h asks.
                self.call(|raw| unsafe {
                    ffi::vaQueryConfigProfiles(raw, list.as_mut_ptr(), &mut count)
                })?,
            )?;
        }
        list.truncate(listed(count));
        let list: Vec<Profile> = list.into_iter().map(Profile).collect();
        self.answers().profiles = Some(list.clone());
        Ok(list)
    }

    /// The entrypoints the driver has for `profile`.
    pub fn entrypoints(&self, profile: Profile) -> Result<Vec<Entrypoint>, Error> {
        if let Some(list) = self.answers().entrypoints.get(&profile) {
            return Ok(list.clone());
        }
        // SAFETY: an initialised display.
        let max = self.call(|raw| unsafe { ffi::vaMaxNumEntrypoints(raw) })?;
        let mut list = vec![0; listed(max)];
        let mut count = 0;
        if !list.is_empty() {
            check(
                "vaQueryConfigEntrypoints",
                // SAFETY: `list` holds vaMaxNumEntrypoints entries.
                self.call(|raw| unsafe {
                    ffi::vaQueryConfigEntrypoints(raw, profile.0, list.as_mut_ptr(), &mut count)
                })?,
            )?;
        }
        list.truncate(listed(count));
        let list: Vec<Entrypoint> = list.into_iter().map(Entrypoint).collect();
        self.answers().entrypoints.insert(profile, list.clone());
        Ok(list)
    }

    /// Whether the driver has `entrypoint` for `profile`.
    pub fn supports(&self, profile: Profile, entrypoint: Entrypoint) -> bool {
        self.entrypoints(profile)
            .is_ok_and(|list| list.contains(&entrypoint))
    }

    /// Config attributes of a pair (`vaGetConfigAttributes`): `None` for
    /// each the driver marks `VA_ATTRIB_NOT_SUPPORTED`. Answered from what
    /// the driver said before where it has said all of them.
    pub(crate) fn config_attributes(
        &self,
        profile: Profile,
        entrypoint: Entrypoint,
        types: &[ffi::VAConfigAttribType],
    ) -> Result<Vec<Option<u32>>, Error> {
        let known: Option<Vec<u32>> = {
            let answers = self.answers();
            (types.iter())
                .map(|&t| answers.attributes.get(&(profile, entrypoint, t)).copied())
                .collect()
        };
        let values = match known {
            Some(values) => values,
            None => {
                let values = self.ask_config_attributes(profile, entrypoint, types)?;
                let mut answers = self.answers();
                for (&t, &value) in types.iter().zip(&values) {
                    answers.attributes.insert((profile, entrypoint, t), value);
                }
                values
            }
        };
        Ok(values
            .into_iter()
            .map(|v| (v != ffi::VA_ATTRIB_NOT_SUPPORTED).then_some(v))
            .collect())
    }

    /// `vaGetConfigAttributes` itself: each attribute's raw value.
    fn ask_config_attributes(
        &self,
        profile: Profile,
        entrypoint: Entrypoint,
        types: &[ffi::VAConfigAttribType],
    ) -> Result<Vec<u32>, Error> {
        let mut attribs: Vec<ffi::VAConfigAttrib> = types
            .iter()
            .map(|&type_| ffi::VAConfigAttrib { type_, value: 0 })
            .collect();
        let count = i32::try_from(attribs.len()).unwrap_or(0);
        check(
            "vaGetConfigAttributes",
            // SAFETY: `attribs` is a valid array of its length.
            self.call(|raw| unsafe {
                ffi::vaGetConfigAttributes(
                    raw,
                    profile.0,
                    entrypoint.0,
                    attribs.as_mut_ptr(),
                    count,
                )
            })?,
        )?;
        Ok(attribs.iter().map(|a| a.value).collect())
    }

    /// The image formats `vaGetImage` and `vaCreateImage` take.
    pub fn image_formats(&self) -> Result<Vec<Fourcc>, Error> {
        Ok((self.image_format_list()?.iter())
            .map(|f| Fourcc(f.fourcc))
            .collect())
    }

    /// The image formats whole (`vaQueryImageFormats`).
    pub(crate) fn image_format_list(&self) -> Result<Vec<ffi::VAImageFormat>, Error> {
        if let Some(list) = self.answers().image_formats.clone() {
            return Ok(list);
        }
        // SAFETY: an initialised display.
        let max = self.call(|raw| unsafe { ffi::vaMaxNumImageFormats(raw) })?;
        let mut list = vec![ffi::VAImageFormat::default(); listed(max)];
        let mut count = 0;
        if !list.is_empty() {
            check(
                "vaQueryImageFormats",
                // SAFETY: `list` holds vaMaxNumImageFormats entries.
                self.call(|raw| unsafe {
                    ffi::vaQueryImageFormats(raw, list.as_mut_ptr(), &mut count)
                })?,
            )?;
        }
        list.truncate(listed(count));
        self.answers().image_formats = Some(list.clone());
        Ok(list)
    }

    /// Everything the driver reports: each profile's entrypoints with
    /// their attributes, surface formats and filters, and the image
    /// formats. Makes and destroys a config (and, for video processing, a
    /// context) per pair.
    pub fn capabilities(&self) -> Result<Capabilities, Error> {
        let mut configs = Vec::new();
        for profile in self.profiles()? {
            for entrypoint in self.entrypoints(profile)? {
                let values = self.config_attributes(profile, entrypoint, &ATTRIBUTES)?;
                let surfaces = self
                    .surface_capabilities(profile, entrypoint)
                    .map_err(|e| e.to_string());
                let proc_filters = (entrypoint == Entrypoint::VIDEO_PROC).then(|| {
                    self.proc_filters(profile, entrypoint)
                        .map_err(|e| e.to_string())
                });
                configs.push(ConfigCapabilities {
                    profile,
                    entrypoint,
                    rt_formats: values[0],
                    max_width: values[1],
                    max_height: values[2],
                    dec_slice_mode: values[3],
                    rate_control: values[4],
                    packed_headers: values[5],
                    max_ref_frames: values[6],
                    max_slices: values[7],
                    surfaces,
                    proc_filters,
                });
            }
        }
        Ok(Capabilities {
            vendor: self.vendor().to_owned(),
            api_version: self.api_version(),
            configs,
            image_formats: self.image_formats()?,
        })
    }

    /// `vaQuerySurfaceAttributes` of a config of the pair, made and
    /// destroyed for the query the first time it is asked.
    pub(crate) fn surface_capabilities(
        &self,
        profile: Profile,
        entrypoint: Entrypoint,
    ) -> Result<SurfaceCapabilities, Error> {
        if let Some(caps) = self.answers().surfaces.get(&(profile, entrypoint)) {
            return Ok(caps.clone());
        }
        let config = Config::new(self, profile, entrypoint, &mut [])?;
        let mut count: c_uint = 0;
        check(
            "vaQuerySurfaceAttributes",
            // SAFETY: a null list asks for the count (va.h,
            // vaQuerySurfaceAttributes).
            self.call(|raw| unsafe {
                ffi::vaQuerySurfaceAttributes(raw, config.id, core::ptr::null_mut(), &mut count)
            })?,
        )?;
        let mut attribs = vec![ffi::VASurfaceAttrib::zeroed(); usize::try_from(count).unwrap_or(0)];
        check(
            "vaQuerySurfaceAttributes",
            // SAFETY: `attribs` holds `count` entries.
            self.call(|raw| unsafe {
                ffi::vaQuerySurfaceAttributes(raw, config.id, attribs.as_mut_ptr(), &mut count)
            })?,
        )?;
        attribs.truncate(usize::try_from(count).unwrap_or(0));
        let mut caps = SurfaceCapabilities::default();
        for a in &attribs {
            if a.value.type_ != ffi::VAGenericValueTypeInteger {
                continue;
            }
            // SAFETY: the value's type says the union holds `i`.
            let value = unsafe { a.value.value.i };
            let unsigned = u32::from_ne_bytes(value.to_ne_bytes());
            match a.type_ {
                ffi::VASurfaceAttribPixelFormat => caps.pixel_formats.push(Fourcc(unsigned)),
                ffi::VASurfaceAttribMinWidth => caps.min_width = Some(unsigned),
                ffi::VASurfaceAttribMaxWidth => caps.max_width = Some(unsigned),
                ffi::VASurfaceAttribMinHeight => caps.min_height = Some(unsigned),
                ffi::VASurfaceAttribMaxHeight => caps.max_height = Some(unsigned),
                ffi::VASurfaceAttribMemoryType => caps.memory_types = Some(unsigned),
                _ => {}
            }
        }
        (self.answers().surfaces).insert((profile, entrypoint), caps.clone());
        Ok(caps)
    }

    fn proc_filters(&self, profile: Profile, entrypoint: Entrypoint) -> Result<Vec<i32>, Error> {
        let config = Config::new(self, profile, entrypoint, &mut [])?;
        let context = Context::new(self, config, 1920, 1080, &mut [])?;
        // va_vpp.h: VAProcFilterCount entries are enough.
        let mut filters = vec![0; 16];
        let mut count = c_uint::try_from(filters.len()).unwrap_or(0);
        check(
            "vaQueryVideoProcFilters",
            // SAFETY: `filters` holds `count` entries.
            self.call(|raw| unsafe {
                ffi::vaQueryVideoProcFilters(raw, context.id, filters.as_mut_ptr(), &mut count)
            })?,
        )?;
        filters.truncate(usize::try_from(count).unwrap_or(0));
        Ok(filters)
    }
}

/// The length of a list the driver gives a count or a bound of: 0 for a
/// negative one, so that no list is asked for into no room.
fn listed(count: i32) -> usize {
    usize::try_from(count).unwrap_or(0)
}

/// A `VAConfigID`, destroyed on drop.
pub(crate) struct Config {
    display: Display,
    pub(crate) id: ffi::VAConfigID,
}

impl Config {
    pub(crate) fn new(
        display: &Display,
        profile: Profile,
        entrypoint: Entrypoint,
        attribs: &mut [ffi::VAConfigAttrib],
    ) -> Result<Self, Error> {
        let mut id = ffi::VA_INVALID_ID;
        let count = i32::try_from(attribs.len()).unwrap_or(0);
        check(
            "vaCreateConfig",
            // SAFETY: `attribs` is a valid array of its length (empty is a
            // null-free empty array, which libva reads none of).
            display.call(|raw| unsafe {
                ffi::vaCreateConfig(
                    raw,
                    profile.0,
                    entrypoint.0,
                    attribs.as_mut_ptr(),
                    count,
                    &mut id,
                )
            })?,
        )?;
        Ok(Self {
            display: display.clone_ref(),
            id,
        })
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        let id = self.id;
        // SAFETY: made by vaCreateConfig on this display, destroyed once,
        // here, behind the destroy of every context on it, as each holds
        // its config.
        self.display.release(move |raw| unsafe {
            ffi::vaDestroyConfig(raw, id);
        });
    }
}

/// A `VAContextID`, destroyed on drop, before its config.
pub(crate) struct Context {
    display: Display,
    pub(crate) id: ffi::VAContextID,
    /// Held so that the config outlives the context: it drops after
    /// [`Context`]'s `drop` has destroyed the context.
    _config: Arc<Config>,
    /// Whether a picture of the context was ended (`vaEndPicture`), which
    /// for a decoder's context on radeonsi begins its session
    /// (`super::buffer::render_decode`).
    ended: AtomicBool,
}

impl Context {
    /// A context on `config`, which it holds: a config of its own, or one
    /// it shares with other contexts.
    pub(crate) fn new(
        display: &Display,
        config: impl Into<Arc<Config>>,
        width: u32,
        height: u32,
        render_targets: &mut [ffi::VASurfaceID],
    ) -> Result<Self, Error> {
        let config = config.into();
        let mut id = ffi::VA_INVALID_ID;
        let size = |v: u32| i32::try_from(v).unwrap_or(i32::MAX);
        let count = i32::try_from(render_targets.len()).unwrap_or(0);
        check(
            "vaCreateContext",
            // SAFETY: `render_targets` is a valid array of its length.
            display.call(|raw| unsafe {
                ffi::vaCreateContext(
                    raw,
                    config.id,
                    size(width),
                    size(height),
                    ffi::VA_PROGRESSIVE,
                    render_targets.as_mut_ptr(),
                    count,
                    &mut id,
                )
            })?,
        )?;
        Ok(Self {
            display: display.clone_ref(),
            id,
            _config: config,
            ended: AtomicBool::new(false),
        })
    }

    /// Whether no picture of the context was ended before: `true` once,
    /// for its first `vaEndPicture`.
    pub(crate) fn first_end(&self) -> bool {
        !self.ended.swap(true, Ordering::Relaxed)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let id = self.id;
        // SAFETY: made by vaCreateContext on this display, destroyed once,
        // here, before its config, whose destroy is made behind this.
        self.display.release(move |raw| unsafe {
            ffi::vaDestroyContext(raw, id);
        });
    }
}
