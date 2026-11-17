//! An encoder session's properties: which it has, what `esup` lists of them, what `ecop`
//! answers, what `eset` and `emul` take, and the [`EncoderConfig`] they make for the backend.
//!
//! The keys are VideoToolbox's `kVTCompressionPropertyKey_*` strings. Each description takes the
//! shape Apple's hardware encoder gives its own: `PropertyType` (`Boolean`, `Number` or
//! `Enumeration`), `ReadWriteStatus` (`ReadWrite` or `ReadOnly`) and an enumeration's
//! `SupportedValueList`. The properties are the ones the host's encoders honour:
//!
//! - `ProfileLevel`: each profile the encoder lists, at each level H.264 names and at
//!   `AutoLevel`, which leaves the level to the encoder; none set is the encoder's highest 8-bit
//!   profile, at `AutoLevel`.
//! - `Quality` with no `DataRateLimits`: every picture at one QP, the M3's for that `Quality`
//!   ([`QUALITY_QPS`]), an `AverageBitRate` left out, as the M3's encoder leaves it.
//! - Else `AverageBitRate` and `DataRateLimits` (pairs of bytes and seconds): a variable bitrate
//!   around the average, under the tightest limit, which then holds as a hard limit over its
//!   window. A limit with no average makes the average the limit; neither is the codec's
//!   default rate ([`EncoderSettings::new`]).
//! - `MaxKeyFrameInterval` (0 for none), `MaxKeyFrameIntervalDuration` and
//!   `AllowTemporalCompression` (false makes every frame a key frame).
//! - `AllowFrameReordering` (true where unset, as VideoToolbox's default is) and
//!   `MaxFrameDelayCount` (-1 for no limit).
//! - `ExpectedFrameRate`, `Quality` (the backend also maps it onto its driver's quality levels),
//!   `H264EntropyMode` for H.264, and `ColorPrimaries`, `TransferFunction` and `YCbCrMatrix`,
//!   which the VUI carries.
//! - `RealTime`, `MaximizePowerEfficiency` and `PrioritizeEncodingSpeedOverQuality` are kept
//!   and answered, and change nothing.
//! - `UsingHardwareAcceleratedVideoEncoder` reads true and is read-only; `Paravirtualized` reads
//!   true and is the guest's own, which `eset` and `emul` never set.
//!
//! An unknown key, a value of another type, and a number outside the range Apple's hardware
//! encoder takes ([`in_range`]) are refused with `kVTPropertyNotSupportedErr`, and the property
//! keeps its value.

use crate::codec::encode::{
    ColourDescription, EncoderSettings, Entropy, FrameRate, Profile, RateControl,
};
use crate::paravirt::host::Codec;
use crate::paravirt::plist::Value;
use crate::paravirt::status::{self, OsStatus};

use super::{EncoderConfig, TICKS_PER_SECOND};

/// `kVTCompressionPropertyKey_Paravirtualized`: the guest's own.
pub(super) const PARAVIRTUALIZED: &str = "Paravirtualized";
const USING_HARDWARE: &str = "UsingHardwareAcceleratedVideoEncoder";
const ALLOW_FRAME_REORDERING: &str = "AllowFrameReordering";
const ALLOW_TEMPORAL_COMPRESSION: &str = "AllowTemporalCompression";
const AVERAGE_BIT_RATE: &str = "AverageBitRate";
const DATA_RATE_LIMITS: &str = "DataRateLimits";
const EXPECTED_FRAME_RATE: &str = "ExpectedFrameRate";
const MAX_FRAME_DELAY_COUNT: &str = "MaxFrameDelayCount";
const MAX_KEY_FRAME_INTERVAL: &str = "MaxKeyFrameInterval";
const MAX_KEY_FRAME_INTERVAL_DURATION: &str = "MaxKeyFrameIntervalDuration";
const PROFILE_LEVEL: &str = "ProfileLevel";
const QUALITY: &str = "Quality";
const REAL_TIME: &str = "RealTime";
const ENTROPY_MODE: &str = "H264EntropyMode";
const COLOR_PRIMARIES: &str = "ColorPrimaries";
const TRANSFER_FUNCTION: &str = "TransferFunction";
const YCBCR_MATRIX: &str = "YCbCrMatrix";
const MAXIMIZE_POWER_EFFICIENCY: &str = "MaximizePowerEfficiency";
const PRIORITIZE_SPEED: &str = "PrioritizeEncodingSpeedOverQuality";

/// How a property's value is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Boolean,
    Number,
    /// One of a list of strings.
    Enumeration,
    /// An array of numbers: `DataRateLimits`, which `esup` gives no type.
    NumberArray,
}

/// Every property, in the order `esup` lists them, and whether it is read-only.
const PROPERTIES: &[(&str, Kind, bool)] = &[
    (ALLOW_FRAME_REORDERING, Kind::Boolean, false),
    (ALLOW_TEMPORAL_COMPRESSION, Kind::Boolean, false),
    (AVERAGE_BIT_RATE, Kind::Number, false),
    (COLOR_PRIMARIES, Kind::Enumeration, false),
    (DATA_RATE_LIMITS, Kind::NumberArray, false),
    (EXPECTED_FRAME_RATE, Kind::Number, false),
    (ENTROPY_MODE, Kind::Enumeration, false),
    (MAX_FRAME_DELAY_COUNT, Kind::Number, false),
    (MAX_KEY_FRAME_INTERVAL, Kind::Number, false),
    (MAX_KEY_FRAME_INTERVAL_DURATION, Kind::Number, false),
    (MAXIMIZE_POWER_EFFICIENCY, Kind::Boolean, false),
    (PARAVIRTUALIZED, Kind::Boolean, false),
    (PRIORITIZE_SPEED, Kind::Boolean, false),
    (PROFILE_LEVEL, Kind::Enumeration, false),
    (QUALITY, Kind::Number, false),
    (REAL_TIME, Kind::Boolean, false),
    (TRANSFER_FUNCTION, Kind::Enumeration, false),
    (USING_HARDWARE, Kind::Boolean, true),
    (YCBCR_MATRIX, Kind::Enumeration, false),
];

/// `ColorPrimaries` values and their ITU-T H.273 `colour_primaries`.
const PRIMARIES: &[(&str, u8)] = &[
    ("ITU_R_709_2", 1),
    ("EBU_3213", 5),
    ("SMPTE_C", 6),
    ("ITU_R_2020", 9),
    ("DCI_P3", 11),
    ("P3_D65", 12),
];

/// `TransferFunction` values and their H.273 `transfer_characteristics`.
const TRANSFERS: &[(&str, u8)] = &[
    ("ITU_R_709_2", 1),
    ("SMPTE_240M_1995", 7),
    ("Linear", 8),
    ("IEC_sRGB", 13),
    ("ITU_R_2020", 14),
    ("SMPTE_ST_2084_PQ", 16),
    ("SMPTE_ST_428_1", 17),
    ("ITU_R_2100_HLG", 18),
];

/// `YCbCrMatrix` values and their H.273 `matrix_coefficients`.
const MATRICES: &[(&str, u8)] = &[
    ("ITU_R_709_2", 1),
    ("ITU_R_601_4", 6),
    ("SMPTE_240M_1995", 7),
    ("ITU_R_2020", 9),
];

/// H.273's "unspecified", for a colour code point the guest did not set.
const UNSPECIFIED: u8 = 2;

/// `H264EntropyMode` values (`kVTH264EntropyMode_CAVLC`, `_CABAC`).
const ENTROPY_MODES: &[(&str, Entropy)] = &[("CAVLC", Entropy::Cavlc), ("CABAC", Entropy::Cabac)];

/// The H.264 levels `ProfileLevel` names, as it spells them, and `level_idc`.
const H264_LEVELS: &[(&str, u8)] = &[
    ("3_0", 30),
    ("3_1", 31),
    ("3_2", 32),
    ("4_0", 40),
    ("4_1", 41),
    ("4_2", 42),
    ("5_0", 50),
    ("5_1", 51),
    ("5_2", 52),
];

/// A profile as `kVTProfileLevel_*` names it.
fn profile_name(profile: Profile) -> &'static str {
    match profile {
        Profile::H264ConstrainedBaseline => "H264_Baseline",
        Profile::H264Main => "H264_Main",
        Profile::H264High => "H264_High",
        Profile::HevcMain => "HEVC_Main",
        Profile::HevcMain10 => "HEVC_Main10",
    }
}

/// The `ProfileLevel` values of `profiles`: each at H.264's levels, then at `AutoLevel`.
fn profile_levels(profiles: &[Profile]) -> Vec<(String, Profile, Option<u8>)> {
    let mut values = Vec::new();
    for &profile in profiles {
        let name = profile_name(profile);
        if profile.codec() == crate::codec::encode::Codec::H264 {
            for &(level, number) in H264_LEVELS {
                values.push((format!("{name}_{level}"), profile, Some(number)));
            }
        }
        values.push((format!("{name}_AutoLevel"), profile, None));
    }
    values
}

/// The properties of one session: its encoder's profiles and the values the guest set.
pub(super) struct Properties {
    codec: Codec,
    profiles: Vec<Profile>,
    values: Vec<(&'static str, Value)>,
}

impl Properties {
    /// The properties of an encoder of `codec` that writes `profiles`, none set.
    pub(super) fn new(codec: Codec, profiles: Vec<Profile>) -> Self {
        Self {
            codec,
            profiles,
            values: Vec::new(),
        }
    }

    fn find(&self, key: &str) -> Option<(&'static str, Kind, bool)> {
        (PROPERTIES.iter().copied()).find(|&(known, _, _)| {
            known == key && (known != ENTROPY_MODE || self.codec == Codec::H264)
        })
    }

    fn value(&self, key: &str) -> Option<&Value> {
        (self.values.iter())
            .find(|(k, _)| *k == key)
            .map(|(_, value)| value)
    }

    /// The values an enumeration takes.
    fn enumeration(&self, key: &str) -> Vec<String> {
        let names = |table: &[(&str, u8)]| table.iter().map(|(n, _)| (*n).to_owned()).collect();
        match key {
            COLOR_PRIMARIES => names(PRIMARIES),
            TRANSFER_FUNCTION => names(TRANSFERS),
            YCBCR_MATRIX => names(MATRICES),
            ENTROPY_MODE => ENTROPY_MODES.iter().map(|(n, _)| (*n).to_owned()).collect(),
            PROFILE_LEVEL => (profile_levels(&self.profiles).into_iter())
                .map(|(name, _, _)| name)
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The profile a session with no `ProfileLevel` set writes: the highest 8-bit one listed.
    fn default_profile(&self) -> Option<Profile> {
        let rank = |p: &Profile| match p {
            Profile::H264High => 3,
            Profile::H264Main => 2,
            Profile::H264ConstrainedBaseline => 1,
            Profile::HevcMain => 1,
            Profile::HevcMain10 => 0,
        };
        self.profiles.iter().copied().max_by_key(rank)
    }

    /// `esup`'s `,sup`: each property's description, by its key.
    pub(super) fn supported(&self) -> Value {
        let mut supported = Vec::new();
        for &(key, kind, read_only) in PROPERTIES {
            if self.find(key).is_none() {
                continue;
            }
            let mut description = Vec::new();
            let kind = match kind {
                Kind::Boolean => Some("Boolean"),
                Kind::Number => Some("Number"),
                Kind::Enumeration => Some("Enumeration"),
                Kind::NumberArray => None,
            };
            if let Some(kind) = kind {
                description.push(("PropertyType".to_owned(), Value::string(kind)));
            }
            let status = if read_only { "ReadOnly" } else { "ReadWrite" };
            description.push(("ReadWriteStatus".to_owned(), Value::string(status)));
            let values = self.enumeration(key);
            if !values.is_empty() {
                description.push((
                    "SupportedValueList".to_owned(),
                    Value::Array(values.into_iter().map(Value::String).collect()),
                ));
            }
            supported.push((key.to_owned(), Value::Dictionary(description)));
        }
        Value::Dictionary(supported)
    }

    /// `ecop`: the status, and the value where there is one. One the guest has not set answers
    /// its default: true for `AllowFrameReordering` and `AllowTemporalCompression`, false for
    /// other booleans, 0 for numbers, the default profile at `AutoLevel`, and no value for the
    /// other enumerations.
    pub(super) fn copy(&self, key: &str) -> (OsStatus, Option<Value>) {
        let Some((key, kind, _)) = self.find(key) else {
            return (status::PROPERTY_NOT_SUPPORTED, None);
        };
        if matches!(key, USING_HARDWARE | PARAVIRTUALIZED) {
            return (status::NO_ERROR, Some(Value::Boolean(true)));
        }
        if let Some(value) = self.value(key) {
            return (status::NO_ERROR, Some(value.clone()));
        }
        let value = match kind {
            Kind::Boolean => Value::Boolean(matches!(
                key,
                ALLOW_FRAME_REORDERING | ALLOW_TEMPORAL_COMPRESSION
            )),
            Kind::Number => Value::Integer(0),
            Kind::NumberArray => Value::Array(Vec::new()),
            Kind::Enumeration if key == PROFILE_LEVEL => match self.default_profile() {
                Some(profile) => Value::String(format!("{}_AutoLevel", profile_name(profile))),
                None => return (status::PROPERTY_NOT_SUPPORTED, None),
            },
            Kind::Enumeration => return (status::NO_ERROR, None),
        };
        (status::NO_ERROR, Some(value))
    }

    /// `eset`, and each of `emul`'s: takes `value`, NULL clearing the property, or refuses it.
    pub(super) fn set(&mut self, key: &str, value: Option<Value>) -> OsStatus {
        let Some((key, kind, read_only)) = self.find(key) else {
            return status::PROPERTY_NOT_SUPPORTED;
        };
        if read_only {
            return status::PROPERTY_READ_ONLY;
        }
        let Some(value) = value else {
            self.values.retain(|(k, _)| *k != key);
            return status::NO_ERROR;
        };
        let fits = match kind {
            Kind::Boolean => value.as_boolean().is_some(),
            Kind::Number => number(&value).is_some(),
            Kind::NumberArray => matches!(&value, Value::Array(values)
                if values.iter().all(|v| number(v).is_some())),
            Kind::Enumeration => (value.as_str())
                .is_some_and(|name| self.enumeration(key).iter().any(|known| known == name)),
        };
        if !fits || !in_range(key, &value) {
            return status::PROPERTY_NOT_SUPPORTED;
        }
        self.values.retain(|(k, _)| *k != key);
        self.values.push((key, value));
        status::NO_ERROR
    }

    /// Every value the guest set, by key, for a set that the backend may refuse to be undone.
    pub(super) fn snapshot(&self) -> Vec<(&'static str, Value)> {
        self.values.clone()
    }

    /// Puts back what [`Self::snapshot`] took.
    pub(super) fn restore(&mut self, values: Vec<(&'static str, Value)>) {
        self.values = values;
    }

    fn boolean(&self, key: &str) -> Option<bool> {
        self.value(key).and_then(Value::as_boolean)
    }

    fn number(&self, key: &str) -> Option<f64> {
        self.value(key).and_then(number)
    }

    fn code(&self, key: &str, table: &[(&str, u8)]) -> Option<u8> {
        let name = self.value(key)?.as_str()?;
        (table.iter())
            .find(|(known, _)| *known == name)
            .map(|(_, code)| *code)
    }

    /// The colour code points the guest set, the rest unspecified; none where it set none.
    fn colour(&self) -> Option<ColourDescription> {
        let primaries = self.code(COLOR_PRIMARIES, PRIMARIES);
        let transfer = self.code(TRANSFER_FUNCTION, TRANSFERS);
        let matrix = self.code(YCBCR_MATRIX, MATRICES);
        (primaries.is_some() || transfer.is_some() || matrix.is_some()).then(|| ColourDescription {
            primaries: primaries.unwrap_or(UNSPECIFIED),
            transfer: transfer.unwrap_or(UNSPECIFIED),
            matrix: matrix.unwrap_or(UNSPECIFIED),
        })
    }

    /// The colour properties the guest set, as the format description's extensions name them
    /// (`CVImageBufferColorPrimaries` and the others), with their values.
    pub(super) fn colour_extensions(&self) -> Vec<(&'static str, String)> {
        [
            (COLOR_PRIMARIES, "CVImageBufferColorPrimaries"),
            (TRANSFER_FUNCTION, "CVImageBufferTransferFunction"),
            (YCBCR_MATRIX, "CVImageBufferYCbCrMatrix"),
        ]
        .into_iter()
        .filter_map(|(key, extension)| Some((extension, self.value(key)?.as_str()?.to_owned())))
        .collect()
    }

    /// The tightest of `DataRateLimits`' pairs, each a byte count then seconds: its rate in bits
    /// a second and its window in milliseconds.
    fn data_rate_limit(&self) -> Option<(u32, u32)> {
        let Some(Value::Array(limits)) = self.value(DATA_RATE_LIMITS) else {
            return None;
        };
        let (pairs, _) = limits.as_chunks::<2>();
        (pairs.iter())
            .filter_map(|[bytes, seconds]| {
                let (bytes, seconds) = (number(bytes)?, number(seconds)?);
                let rate = bytes * 8.0 / seconds;
                (bytes > 0.0 && seconds > 0.0 && rate.is_finite()).then_some((rate, seconds))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .and_then(|(rate, seconds)| {
                let rate = u32::try_from(whole_number(rate)?).ok().filter(|&r| r > 0)?;
                let window = u32::try_from(whole_number(seconds * 1000.0)?).unwrap_or(u32::MAX);
                Some((rate, window.max(1)))
            })
    }

    /// The configuration the properties make for an encoder at `width` by `height`, even;
    /// `None` where the encoder lists no profile.
    pub(super) fn config(&self, width: u32, height: u32) -> Option<EncoderConfig> {
        let (profile, level) = match self.value(PROFILE_LEVEL).and_then(Value::as_str) {
            Some(name) => (profile_levels(&self.profiles).into_iter())
                .find(|(known, _, _)| known == name)
                .map(|(_, profile, level)| (profile, level))?,
            None => (self.default_profile()?, None),
        };
        let mut settings = EncoderSettings::new(profile, width, height);
        settings.level = level;
        settings.timescale = TICKS_PER_SECOND;
        settings.colour = self.colour();
        if let Some(rate) = self.number(EXPECTED_FRAME_RATE).and_then(frame_rate) {
            settings.frame_rate = rate;
        }
        let average = (self.number(AVERAGE_BIT_RATE).and_then(whole_number))
            .and_then(|bits| u32::try_from(bits).ok())
            .filter(|&bits| bits > 0);
        let limit = self.data_rate_limit();
        if let (Some(quality), None) = (self.number(QUALITY), limit) {
            let qp = quality_qp(quality);
            settings.rate_control = RateControl::ConstantQp {
                intra: qp,
                predicted: qp,
                bidirectional: qp,
            };
        } else if average.is_some() || limit.is_some() {
            let peak = limit.map(|(rate, _)| rate);
            let average = average.or(peak).unwrap_or(1);
            settings.rate_control = RateControl::VariableBitRate {
                average_bits_per_second: average,
                peak_bits_per_second: peak
                    .unwrap_or_else(|| average.saturating_add(average / 2))
                    .max(average),
            };
        }
        if let Some((_, window)) = limit {
            settings.buffer_milliseconds = window;
            settings.set_hard_limit(true);
        }
        settings.key_frame_interval =
            match self.number(MAX_KEY_FRAME_INTERVAL).and_then(whole_number) {
                Some(0) => None,
                Some(frames) => Some(u32::try_from(frames).unwrap_or(u32::MAX)),
                None => settings.key_frame_interval,
            };
        if self.boolean(ALLOW_TEMPORAL_COMPRESSION) == Some(false) {
            settings.key_frame_interval = Some(1);
        }
        settings.key_frame_interval_duration = self
            .number(MAX_KEY_FRAME_INTERVAL_DURATION)
            .filter(|&seconds| seconds.is_finite() && seconds > 0.0)
            .and_then(|seconds| whole_number(seconds * f64::from(TICKS_PER_SECOND)))
            .and_then(|ticks| i64::try_from(ticks).ok());
        settings.allow_frame_reordering = self.boolean(ALLOW_FRAME_REORDERING).unwrap_or(true);
        settings.max_frame_delay = match self.number(MAX_FRAME_DELAY_COUNT) {
            Some(count) if count >= 0.0 => whole_number(count).and_then(|c| u32::try_from(c).ok()),
            _ => None,
        };
        settings.entropy = (self.value(ENTROPY_MODE).and_then(Value::as_str)).and_then(|name| {
            (ENTROPY_MODES.iter())
                .find(|(known, _)| *known == name)
                .map(|(_, mode)| *mode)
        });
        let quality = self.number(QUALITY).map(narrow);
        Some(EncoderConfig { settings, quality })
    }
}

/// Whether `value`, of `key`'s type, is in the range Apple's hardware encoder takes, which
/// refuses any other with `kVTPropertyNotSupportedErr` (measured on macOS 27, 2026-10-24):
/// `Quality` from 0 to 1, `ExpectedFrameRate` from 1 to under 100000, and
/// `MaxKeyFrameIntervalDuration` 0 or more, none NaN; `AverageBitRate` and
/// `MaxKeyFrameInterval` from 0 to 2^31 - 1, a real cut toward zero. Every other number is taken.
fn in_range(key: &str, value: &Value) -> bool {
    let real = number(value).unwrap_or(f64::NAN);
    match key {
        QUALITY => (0.0..=1.0).contains(&real),
        EXPECTED_FRAME_RATE => (1.0..100_000.0).contains(&real),
        MAX_KEY_FRAME_INTERVAL_DURATION => real >= 0.0,
        AVERAGE_BIT_RATE | MAX_KEY_FRAME_INTERVAL => (0.0..2_147_483_648.0).contains(&real.trunc()),
        _ => true,
    }
}

/// A number value, integer or real.
fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Real(real) => Some(*real),
        Value::Integer(integer) => Some(nearest_f64(*integer)),
        _ => None,
    }
}

/// `rate` frames a second as a fraction in thousandths, in lowest terms.
fn frame_rate(rate: f64) -> Option<FrameRate> {
    let thousandths = u32::try_from(whole_number(rate * 1000.0)?).ok()?;
    Some(FrameRate::new(thousandths, 1000).reduced()).filter(|r| r.numerator > 0)
}

/// `value` rounded to the nearest whole number, half away from zero, where it is finite and not
/// negative; past `u64::MAX`, `u64::MAX`.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a float to an integer: `as` saturates, and the value is finite and not negative"
)]
fn whole_number(value: f64) -> Option<u64> {
    (value.is_finite() && value >= 0.0).then(|| value.round() as u64)
}

/// The QP an M3's encoder codes every picture at, I, P and B alike, H.264's and HEVC's, at a
/// `Quality` with no `DataRateLimits`, beside an `AverageBitRate` or not: from each percent of
/// `Quality` on, its QP (macOS 27.0.1, measured at each percent from 0 to 100).
const QUALITY_QPS: [(u8, u8); 49] = [
    (0, 48),
    (11, 47),
    (14, 46),
    (17, 45),
    (19, 44),
    (22, 43),
    (25, 42),
    (28, 41),
    (30, 40),
    (33, 39),
    (36, 38),
    (38, 37),
    (41, 36),
    (43, 35),
    (46, 34),
    (48, 33),
    (51, 32),
    (53, 31),
    (55, 30),
    (57, 29),
    (60, 28),
    (62, 27),
    (64, 26),
    (66, 25),
    (68, 24),
    (70, 23),
    (72, 22),
    (74, 21),
    (76, 20),
    (78, 19),
    (79, 18),
    (81, 17),
    (83, 16),
    (84, 15),
    (86, 14),
    (87, 13),
    (88, 12),
    (89, 11),
    (90, 10),
    (91, 9),
    (92, 8),
    (93, 7),
    (94, 6),
    (95, 5),
    (96, 4),
    (97, 3),
    (98, 2),
    (99, 1),
    (100, 0),
];

/// The QP of `quality`, 0 to 1 ([`QUALITY_QPS`]).
fn quality_qp(quality: f64) -> u8 {
    // A millionth of a percent over, so that a percent's own value, 0.29 for one, reads as it.
    let percent = quality * 100.0 + 1e-6;
    (QUALITY_QPS.iter().rev())
        .find(|(from, _)| f64::from(*from) <= percent)
        .map_or(QUALITY_QPS[0].1, |(_, qp)| *qp)
}

/// `quality`, 0 to 1, as an `f32`.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "a float to a narrower float: within 0 to 1 it only rounds"
)]
fn narrow(quality: f64) -> f32 {
    quality as f32
}

/// The nearest `f64` to `integer`: exact up to 2^53.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "an integer to a float, the nearest one"
)]
fn nearest_f64(integer: i64) -> f64 {
    integer as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264() -> Properties {
        Properties::new(
            Codec::H264,
            vec![
                Profile::H264ConstrainedBaseline,
                Profile::H264Main,
                Profile::H264High,
            ],
        )
    }

    /// `esup` gives each property its type and status, and `ProfileLevel` its values: nine
    /// levels and `AutoLevel` for each H.264 profile.
    #[test]
    fn each_property_is_listed_with_its_type_and_status() {
        let supported = h264().supported();
        assert_eq!(
            supported.get(AVERAGE_BIT_RATE),
            Some(&Value::dictionary([
                ("PropertyType", Value::string("Number")),
                ("ReadWriteStatus", Value::string("ReadWrite")),
            ]))
        );
        assert_eq!(
            supported
                .get(USING_HARDWARE)
                .and_then(|d| d.get("ReadWriteStatus")),
            Some(&Value::string("ReadOnly"))
        );
        assert_eq!(
            supported.get(DATA_RATE_LIMITS),
            Some(&Value::dictionary([(
                "ReadWriteStatus",
                Value::string("ReadWrite")
            )]))
        );
        let Some(Value::Array(levels)) = supported
            .get(PROFILE_LEVEL)
            .and_then(|d| d.get("SupportedValueList"))
        else {
            panic!("ProfileLevel is an enumeration");
        };
        assert_eq!(levels.len(), 3 * 10);
        assert!(levels.contains(&Value::string("H264_Baseline_3_0")));
        assert!(levels.contains(&Value::string("H264_High_AutoLevel")));
        let hevc = Properties::new(Codec::Hevc, vec![Profile::HevcMain]).supported();
        assert_eq!(hevc.get(ENTROPY_MODE), None, "H.264's alone");
    }

    /// What the guest sets becomes the encoder's settings.
    #[test]
    fn the_values_set_become_the_settings() {
        let mut properties = h264();
        for (key, value) in [
            (PROFILE_LEVEL, Value::string("H264_Main_4_1")),
            (AVERAGE_BIT_RATE, Value::Integer(2_000_000)),
            (
                DATA_RATE_LIMITS,
                Value::Array(vec![
                    Value::Integer(500_000),
                    Value::Integer(1),
                    Value::Integer(600_000),
                    Value::Real(2.0),
                ]),
            ),
            (ALLOW_FRAME_REORDERING, Value::Boolean(false)),
            (MAX_KEY_FRAME_INTERVAL, Value::Real(30.0)),
            (MAX_KEY_FRAME_INTERVAL_DURATION, Value::Real(0.5)),
            (EXPECTED_FRAME_RATE, Value::Real(29.97)),
            (ENTROPY_MODE, Value::string("CAVLC")),
            (COLOR_PRIMARIES, Value::string("ITU_R_709_2")),
            (QUALITY, Value::Real(0.25)),
        ] {
            assert_eq!(properties.set(key, Some(value)), 0, "{key}");
        }
        let EncoderConfig { settings, quality } = properties.config(640, 480).unwrap();
        assert_eq!(
            (settings.profile, settings.level),
            (Profile::H264Main, Some(41))
        );
        assert_eq!((settings.width, settings.height), (640, 480));
        assert_eq!(
            settings.rate_control,
            RateControl::VariableBitRate {
                average_bits_per_second: 2_000_000,
                peak_bits_per_second: 2_400_000,
            }
        );
        assert_eq!(settings.buffer_milliseconds, 2000);
        assert!(!settings.allow_frame_reordering);
        assert_eq!(settings.key_frame_interval, Some(30));
        assert_eq!(settings.key_frame_interval_duration, Some(45_000));
        assert_eq!(settings.frame_rate, FrameRate::new(29_970, 1000).reduced());
        assert_eq!(settings.entropy, Some(Entropy::Cavlc));
        assert_eq!(
            settings.colour,
            Some(ColourDescription {
                primaries: 1,
                transfer: 2,
                matrix: 2,
            })
        );
        assert_eq!(quality, Some(0.25));
        assert_eq!(
            properties.colour_extensions(),
            vec![("CVImageBufferColorPrimaries", "ITU_R_709_2".to_owned())]
        );
    }

    /// A `Quality` codes every picture at the M3's QP for it, beside an `AverageBitRate` too; a
    /// `DataRateLimits` beside it keeps the bitrate's rate control.
    #[test]
    fn a_quality_is_the_m3s_constant_qp_unless_a_limit_is_set() {
        let qp = |quality: f64, others: &[(&str, Value)]| {
            let mut properties = h264();
            assert_eq!(properties.set(QUALITY, Some(Value::Real(quality))), 0);
            for (key, value) in others {
                assert_eq!(properties.set(key, Some(value.clone())), 0, "{key}");
            }
            properties.config(640, 480).unwrap().settings.rate_control
        };
        let constant = |qp| RateControl::ConstantQp {
            intra: qp,
            predicted: qp,
            bidirectional: qp,
        };
        for (quality, want) in [
            (0.0, 48),
            (0.1, 48),
            (0.11, 47),
            (0.29, 41),
            (0.5, 33),
            (0.75, 21),
            (0.9, 10),
            (0.99, 1),
            (1.0, 0),
        ] {
            assert_eq!(qp(quality, &[]), constant(want), "{quality}");
        }
        let average = [(AVERAGE_BIT_RATE, Value::Integer(1_000_000))];
        assert_eq!(qp(0.9, &average), constant(10));
        let limit = [(
            DATA_RATE_LIMITS,
            Value::Array(vec![Value::Integer(20_000), Value::Integer(1)]),
        )];
        assert!(matches!(
            qp(0.9, &limit),
            RateControl::VariableBitRate { .. }
        ));
    }

    /// With nothing set, the encoder writes its highest profile at the codec's defaults, frames
    /// reordered.
    #[test]
    fn nothing_set_is_the_highest_profile_at_the_defaults() {
        let properties = h264();
        let config = properties.config(64, 48).unwrap();
        let defaults = EncoderSettings::new(Profile::H264High, 64, 48);
        assert_eq!(config.settings.profile, Profile::H264High);
        assert_eq!(config.settings.rate_control, defaults.rate_control);
        assert_eq!(
            config.settings.key_frame_interval,
            defaults.key_frame_interval
        );
        assert!(config.settings.allow_frame_reordering);
        assert_eq!(config.settings.timescale, TICKS_PER_SECOND);
        assert_eq!(config.quality, None);
        assert_eq!(
            properties.copy(PROFILE_LEVEL),
            (0, Some(Value::string("H264_High_AutoLevel")))
        );
        assert_eq!(
            properties.copy(ALLOW_FRAME_REORDERING),
            (0, Some(Value::Boolean(true)))
        );
        assert_eq!(
            properties.copy(AVERAGE_BIT_RATE),
            (0, Some(Value::Integer(0)))
        );
        assert_eq!(properties.copy(COLOR_PRIMARIES), (0, None));
    }

    /// An unknown key, another type and a number out of range are refused, and the property
    /// keeps nothing; the read-only one is refused as read-only.
    #[test]
    fn what_does_not_fit_is_refused() {
        let mut properties = h264();
        let refused = status::PROPERTY_NOT_SUPPORTED;
        assert_eq!(
            properties.set("NoSuchKey", Some(Value::Integer(1))),
            refused
        );
        assert_eq!(
            properties.set(ALLOW_FRAME_REORDERING, Some(Value::Integer(1))),
            refused
        );
        assert_eq!(
            properties.set(PROFILE_LEVEL, Some(Value::string("HEVC_Main_AutoLevel"))),
            refused
        );
        assert_eq!(properties.set(QUALITY, Some(Value::Real(1.5))), refused);
        assert_eq!(
            properties.set(EXPECTED_FRAME_RATE, Some(Value::Real(0.5))),
            refused
        );
        assert_eq!(
            properties.set(AVERAGE_BIT_RATE, Some(Value::Integer(-1))),
            refused
        );
        assert_eq!(
            properties.set(AVERAGE_BIT_RATE, Some(Value::Integer(1 << 31))),
            refused
        );
        assert_eq!(
            properties.set(USING_HARDWARE, Some(Value::Boolean(false))),
            status::PROPERTY_READ_ONLY
        );
        assert_eq!(properties.copy(QUALITY), (0, Some(Value::Integer(0))));
        assert_eq!(
            properties.copy(USING_HARDWARE),
            (0, Some(Value::Boolean(true)))
        );
        assert_eq!(properties.copy("NoSuchKey"), (refused, None));
    }

    /// `AllowTemporalCompression` false makes every frame a key frame; `MaxKeyFrameInterval` 0
    /// sets no limit.
    #[test]
    fn key_frames_follow_the_interval_and_temporal_compression() {
        let mut properties = h264();
        properties.set(MAX_KEY_FRAME_INTERVAL, Some(Value::Integer(0)));
        assert_eq!(
            properties
                .config(64, 48)
                .unwrap()
                .settings
                .key_frame_interval,
            None
        );
        properties.set(ALLOW_TEMPORAL_COMPRESSION, Some(Value::Boolean(false)));
        assert_eq!(
            properties
                .config(64, 48)
                .unwrap()
                .settings
                .key_frame_interval,
            Some(1)
        );
    }
}
