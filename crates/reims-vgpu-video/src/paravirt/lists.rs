//! The host's own answers: the codec lists (`decs`, `encs`, `meps`), a decoder's capabilities
//! (`dcap`) and a create's reply.
//!
//! The lists say what the guest may ask for. The decoder list holds one decoder for each codec the
//! host's video arm decodes, with the profiles and the level the arm answers: the guest's
//! VideoToolbox registers each as `paravirtualized:<ID>`, hardware-accelerated. The encoder list
//! holds one encoder for each codec the host's video arm encodes, by the ID of Apple's hardware
//! encoder of the codec ([`apple_encoder_id`]), and none where it has no encoder backend; the
//! motion-estimation list holds none, as Apple's host answers where it has none.

use crate::paravirt::message::{Message, MessageFlags, MessageWriter};
use crate::paravirt::names;
use crate::paravirt::plist::Value;
use crate::paravirt::status::OsStatus;
use crate::paravirt::{Codec, Decoder, Encoder, WireError};

/// The protocol version each list and create reply carries: 6, for the decoders, the encoders
/// (`kVTVESHostProtocolVersion`), motion estimation and the capabilities
/// (`kVTVDCapabilitesHostProtocolVersion`) alike.
pub(crate) const PROTOCOL_VERSION: u32 = 6;

/// The rating each listed decoder and encoder carries. Apple's hardware decoders and encoders
/// carry 400 (measured on macOS 27, 26A434, 2026-10-24: `decs` lists
/// `com.apple.videotoolbox.videodecoder.avd.avc` and `encs`
/// `com.apple.videotoolbox.videoencoder.ave.avc` with VTRating 400); the guest prefers the
/// higher-rated of two for a codec.
const RATING: i64 = 400;

/// What `,hin`'s `VTHostBuildVersion` says of this host. Apple's says the host's macOS build; the
/// guest only logs it. The value is ours.
const HOST_BUILD_VERSION: &str = concat!("reims-vgpu-video ", env!("CARGO_PKG_VERSION"));

/// The reply to `request`: its type, UUID and reply identifier, flagged as a reply to the guest.
pub(crate) fn reply(request: &Message<'_>) -> MessageWriter {
    MessageWriter::reply_to(&request.header(), MessageFlags::REPLY_TO_GUEST)
}

/// The `CMClassImplementationID` of this host's decoder for `codec`, which the guest registers as
/// `paravirtualized:<ID>` and sends back as `,cid` when it opens one. The IDs and names are ours.
fn decoder_id(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "reims-vgpu-video.videodecoder.avc",
        Codec::Hevc => "reims-vgpu-video.videodecoder.hevc",
        Codec::Av1 => "reims-vgpu-video.videodecoder.av1",
        Codec::Vp9 => "reims-vgpu-video.videodecoder.vp9",
    }
}

fn decoder_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "reims-vgpu-video Decoder - AVC",
        Codec::Hevc => "reims-vgpu-video Decoder - HEVC",
        Codec::Av1 => "reims-vgpu-video Decoder - AV1",
        Codec::Vp9 => "reims-vgpu-video Decoder - VP9",
    }
}

fn fourcc_string(codec: Codec) -> Value {
    Value::String(String::from_utf8_lossy(&codec.fourcc().to_be_bytes()).into_owned())
}

fn profiles(decoder: &Decoder) -> Value {
    Value::Array(
        decoder
            .profiles
            .iter()
            .map(|&p| Value::Integer(i64::from(p)))
            .collect(),
    )
}

/// One entry of the decoder list, in the keys and shapes Apple's hardware decoders' entries have
/// (measured on macOS 27, 26A434, 2026-10-24: VTCodecType an array of FourCC strings,
/// VTPerProfileDetails keyed by the profile as a decimal string, under VTDecoderCapabilities).
fn decoder_entry(decoder: &Decoder) -> Value {
    let details = decoder
        .profiles
        .iter()
        .map(|p| {
            let detail = Value::dictionary([(
                "VTMaxDecodeLevel",
                Value::Integer(i64::from(decoder.max_level)),
            )]);
            (p.to_string(), detail)
        })
        .collect();
    Value::dictionary([
        (
            "CMClassImplementationID",
            Value::string(decoder_id(decoder.codec)),
        ),
        (
            "CMClassImplementationName",
            Value::string(decoder_name(decoder.codec)),
        ),
        ("IsPreferredInternalPlugin", Value::Boolean(false)),
        (
            "VTCodecType",
            Value::Array(vec![fourcc_string(decoder.codec)]),
        ),
        ("VTRating", Value::Integer(RATING)),
        (
            "VTDecoderCapabilities",
            Value::dictionary([
                ("VTPerProfileDetails", Value::Dictionary(details)),
                ("VTSupportedProfileArray", profiles(decoder)),
            ]),
        ),
    ])
}

/// `decs`: the decoders, then the version. The request's `,ver` is read but not checked.
pub(crate) fn decoder_list(
    request: &Message<'_>,
    decoders: &[Decoder],
) -> Result<Vec<u8>, WireError> {
    request.u32(names::VERSION)?;
    let entries = decoders.iter().map(decoder_entry).collect();
    let mut reply = reply(request);
    reply.property_list(names::DECODERS, Some(&Value::Array(entries)));
    reply.u32(names::VERSION, PROTOCOL_VERSION);
    Ok(reply.finish())
}

/// A decoder's capabilities as `dcap` answers them, in the shape Apple's host answers for `avc1`
/// (measured on macOS 27, 26A434, 2026-10-24): per profile, that it is hardware-accelerated and
/// its highest decode and playback level, then the profiles. VTIsHDRAllowedOnDevice is true, as
/// Apple's host answers on a device that allows HDR: the arm decodes an HDR stream as any other,
/// and the profiles say which it takes.
fn capabilities(decoder: &Decoder) -> Value {
    let level = Value::Integer(i64::from(decoder.max_level));
    let per_profile = decoder
        .profiles
        .iter()
        .map(|p| {
            let support = Value::dictionary([
                ("VTIsHardwareAccelerated", Value::Boolean(true)),
                ("VTMaxDecodeLevel", level.clone()),
                ("VTMaxPlaybackLevel", level.clone()),
            ]);
            (p.to_string(), support)
        })
        .collect();
    Value::dictionary([
        ("VTIsHDRAllowedOnDevice", Value::Boolean(true)),
        ("VTPerProfileSupport", Value::Dictionary(per_profile)),
        ("VTSupportedProfiles", profiles(decoder)),
    ])
}

/// `dcap`: the capabilities of the codec `,cod` (`=s32`) names, NULL for one the host does not
/// decode, then the version.
pub(crate) fn decoder_capabilities(
    request: &Message<'_>,
    decoders: &[Decoder],
) -> Result<Vec<u8>, WireError> {
    let codec = request.s32(names::CODEC)?.unwrap_or(0).cast_unsigned();
    let found = decoders.iter().find(|d| d.codec.fourcc().0 == codec);
    let mut reply = reply(request);
    reply.property_list(
        names::DECODER_CAPABILITIES,
        found.map(capabilities).as_ref(),
    );
    reply.u32(names::VERSION, PROTOCOL_VERSION);
    Ok(reply.finish())
}

/// The `CMClassImplementationID` of Apple's hardware encoder for `codec`, which this host's
/// encoder of it is listed by. The guest registers the entry as `paravirtualized:<ID>`; its own
/// two paravirtual encoders, the hardware encoders its apps get
/// (`VTVPParavirtualizedH264VideoEncoder` and its HEVC twin), open the one of this ID, sending
/// it as `,cid`, and fail (`kVTInvalidSessionErr`) where the host lists none (read from code,
/// VideoToolbox of macOS 27, 2026-11-16; seen on a macOS 26.6.2 guest: a session that requires
/// a hardware encoder fails so, and one that allows one is given a software encoder, while the
/// list names this host's encoders by IDs of its own).
/// No arm encodes AV1 or VP9, so no encoder of either is listed: their IDs are ours, and never
/// sent.
pub(crate) fn apple_encoder_id(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "com.apple.videotoolbox.videoencoder.ave.avc",
        Codec::Hevc => "com.apple.videotoolbox.videoencoder.ave.hevc",
        Codec::Av1 => "reims-vgpu-video.videoencoder.av1",
        Codec::Vp9 => "reims-vgpu-video.videoencoder.vp9",
    }
}

fn encoder_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "reims-vgpu-video Encoder - AVC",
        Codec::Hevc => "reims-vgpu-video Encoder - HEVC",
        Codec::Av1 => "reims-vgpu-video Encoder - AV1",
        Codec::Vp9 => "reims-vgpu-video Encoder - VP9",
    }
}

/// One entry of the encoder list, in the keys Apple's hardware encoders' entries have (measured
/// on macOS 27, 2026-10-24): the codec type as a number, the codec's name, the encoder's name
/// (ours) and ID (Apple's, [`apple_encoder_id`]), and the rating.
fn encoder_entry(encoder: &Encoder) -> Value {
    let codec = encoder.codec;
    Value::dictionary([
        ("VTCodecType", Value::Integer(i64::from(codec.fourcc().0))),
        ("VTRating", Value::Integer(RATING)),
        (
            "VTCodecName",
            Value::string(match codec {
                Codec::H264 => "H.264",
                Codec::Hevc => "HEVC",
                Codec::Av1 => "AV1",
                Codec::Vp9 => "VP9",
            }),
        ),
        (
            "CMClassImplementationName",
            Value::string(encoder_name(codec)),
        ),
        (
            "CMClassImplementationID",
            Value::string(apple_encoder_id(codec)),
        ),
    ])
}

/// `encs`: the encoders, then the version. The request's `,ver` is read but not checked.
pub(crate) fn encoder_list(
    request: &Message<'_>,
    encoders: &[Encoder],
) -> Result<Vec<u8>, WireError> {
    request.u32(names::VERSION)?;
    let entries = encoders.iter().map(encoder_entry).collect();
    let mut reply = reply(request);
    reply.property_list(names::ENCODERS, Some(&Value::Array(entries)));
    reply.u32(names::VERSION, PROTOCOL_VERSION);
    Ok(reply.finish())
}

/// What `meps` answers: a reply, or a status and no reply.
pub(crate) enum MotionEstimationAnswer {
    /// The reply to send.
    Reply(Vec<u8>),
    /// The status to answer, with nothing sent.
    Refused(OsStatus),
}

/// `meps`: a guest older than protocol 6 is refused with no reply, any other told of no
/// processor, then the version.
pub(crate) fn motion_estimation_list(
    request: &Message<'_>,
) -> Result<MotionEstimationAnswer, WireError> {
    let version = request.u32(names::VERSION)?.unwrap_or(0);
    if version < PROTOCOL_VERSION {
        return Ok(MotionEstimationAnswer::Refused(
            crate::paravirt::status::UNSUPPORTED_GUEST,
        ));
    }
    let mut reply = reply(request);
    reply.property_list(names::MOTION_ESTIMATORS, Some(&Value::Array(Vec::new())));
    reply.u32(names::VERSION, PROTOCOL_VERSION);
    Ok(MotionEstimationAnswer::Reply(reply.finish()))
}

/// A create's reply, with `status` in its `,err`, then the version and the host's info, as Apple's
/// host answers a create it makes no codec for: the guest fails its create on the `,err`.
pub(crate) fn create_reply(request: &Message<'_>, status: OsStatus) -> Vec<u8> {
    let mut reply = reply(request);
    reply.s32(names::ERROR, status);
    reply.u32(names::VERSION, PROTOCOL_VERSION);
    reply.property_list(
        names::HOST_INFO,
        Some(&Value::dictionary([(
            "VTHostBuildVersion",
            Value::string(HOST_BUILD_VERSION),
        )])),
    );
    reply.finish()
}

/// A JPEG request's reply: `,err` alone, as the host has no JPEG engine.
pub(crate) fn jpeg_reply(request: &Message<'_>, status: OsStatus) -> Vec<u8> {
    let mut reply = reply(request);
    reply.s32(names::ERROR, status);
    reply.finish()
}
