//! The host's side of the guest's messages: what each is answered with, and what the host sends
//! the guest for it.

use crate::core::fourcc::FourCc;
use crate::paravirt::lists::{self, MotionEstimationAnswer};
use crate::paravirt::message::{Message, read_header};
use crate::paravirt::names::{self, HostMessage};
use crate::paravirt::status::{self, OsStatus};

/// A codec the host's decoders decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    /// H.264, `avc1`.
    H264,
    /// HEVC, `hvc1`.
    Hevc,
}

impl Codec {
    /// The CoreMedia codec type the guest names it by.
    pub const fn fourcc(self) -> FourCc {
        FourCc::new(match self {
            Self::H264 => *b"avc1",
            Self::Hevc => *b"hvc1",
        })
    }
}

/// A decoder the host's video arm runs, as the decoder list names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoder {
    /// Its codec.
    pub codec: Codec,
    /// The profiles it decodes, as VideoToolbox numbers them (H.264's `profile_idc`, HEVC's
    /// `general_profile_idc`), in ascending order.
    pub profiles: Vec<u32>,
    /// The highest level it decodes, as the codec numbers it (H.264's `level_idc`, HEVC's
    /// `general_level_idc`).
    pub max_level: u32,
}

/// What a guest message is answered with: the status the guest's send returns, and the messages
/// the host sends the guest for it, in order, each at most the guest's 16 KiB dequeue buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Answer {
    /// The status.
    pub status: OsStatus,
    /// The messages to the guest.
    pub replies: Vec<Vec<u8>>,
}

impl Answer {
    fn status(status: OsStatus) -> Self {
        Self {
            status,
            replies: Vec::new(),
        }
    }

    fn reply(reply: Vec<u8>) -> Self {
        Self {
            status: status::NO_ERROR,
            replies: vec![reply],
        }
    }
}

/// The host's side: the decoders the host's video arm runs, and the answers the module doc of
/// [`crate::paravirt`] lists. It keeps no session.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Host {
    decoders: Vec<Decoder>,
}

impl Host {
    /// A host whose decoder list holds `decoders`.
    pub fn new(decoders: Vec<Decoder>) -> Self {
        Self { decoders }
    }

    /// The decoders the list holds.
    pub fn decoders(&self) -> &[Decoder] {
        &self.decoders
    }

    /// Answers the guest's message `bytes`, as the kext took it from its client: a message that
    /// does not read is malformed, and one for a session is answered as for a UUID with none.
    pub fn deliver(&self, bytes: &[u8]) -> Answer {
        if let Err(error) = read_header(bytes, 0) {
            return Answer::status(error.status());
        }
        let message = match Message::parse(bytes) {
            Ok(message) => message,
            Err(error) => return Answer::status(error.status()),
        };
        let header = message.header();
        let Some(kind) = HostMessage::from_fourcc(header.message_type) else {
            return Answer::status(status::SESSION_NOT_FOUND);
        };
        let replied = |reply: Result<Vec<u8>, crate::paravirt::WireError>| match reply {
            Ok(reply) => Answer::reply(reply),
            Err(error) => Answer::status(error.status()),
        };
        match kind {
            HostMessage::DecoderList => replied(lists::decoder_list(&message, &self.decoders)),
            HostMessage::DecoderCapabilities => {
                replied(lists::decoder_capabilities(&message, &self.decoders))
            }
            HostMessage::EncoderList => replied(lists::encoder_list(&message)),
            HostMessage::MotionEstimationList => match lists::motion_estimation_list(&message) {
                Ok(MotionEstimationAnswer::Reply(reply)) => Answer::reply(reply),
                Ok(MotionEstimationAnswer::Refused(status)) => Answer::status(status),
                Err(error) => Answer::status(error.status()),
            },
            HostMessage::CreateDecoder | HostMessage::CreateEncoder => {
                // A decoder or encoder create needs a nonzero `,cod` and reads `,ver`, as
                // Apple's create reads them.
                match (message.s32(names::CODEC), message.u32(names::VERSION)) {
                    (Ok(Some(codec)), Ok(_)) if codec != 0 => {
                        let refused = if kind == HostMessage::CreateDecoder {
                            status::COULD_NOT_FIND_VIDEO_DECODER
                        } else {
                            status::COULD_NOT_FIND_VIDEO_ENCODER
                        };
                        Answer::reply(lists::create_reply(&message, refused))
                    }
                    (Err(error), _) | (_, Err(error)) => Answer::status(error.status()),
                    _ => Answer::status(status::MALFORMED_MESSAGE),
                }
            }
            HostMessage::CreateMotionEstimation => match message.u32(names::VERSION) {
                Ok(_) => Answer::reply(lists::create_reply(
                    &message,
                    status::MOTION_ESTIMATION_PROCESSOR_NOT_FOUND,
                )),
                Err(error) => Answer::status(error.status()),
            },
            // The host keeps no session, so a close finds none to end.
            HostMessage::Destroy => Answer::status(status::NO_ERROR),
            HostMessage::Jpeg => {
                Answer::reply(lists::jpeg_reply(&message, status::NO_JPEG_HARDWARE))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paravirt::message::{GuestUuid, MessageFlags, MessageWriter};
    use crate::paravirt::plist::{Value, encode};

    fn decoders() -> Vec<Decoder> {
        vec![
            Decoder {
                codec: Codec::H264,
                profiles: vec![66, 77, 100],
                max_level: 52,
            },
            Decoder {
                codec: Codec::Hevc,
                profiles: vec![1, 2, 3],
                max_level: 153,
            },
        ]
    }

    fn request(kind: &[u8; 4]) -> MessageWriter {
        MessageWriter::new(
            FourCc::new(*kind),
            MessageFlags::GUEST_EXPECTS_REPLY,
            Some(GuestUuid([0x5a; 16])),
            9,
        )
    }

    /// The one reply `answer` holds, checked as the reply to a request of `kind`, and its status.
    fn one_reply(answer: &Answer, kind: &[u8; 4]) -> Vec<u8> {
        assert_eq!(answer.status, 0);
        let [reply] = answer.replies.as_slice() else {
            panic!("one reply, not {}", answer.replies.len())
        };
        let header = Message::parse(reply).unwrap().header();
        assert_eq!(header.message_type, FourCc::new(*kind));
        assert_eq!(header.flags, MessageFlags::REPLY_TO_GUEST);
        assert_eq!(header.guest_uuid, GuestUuid([0x5a; 16]));
        assert_eq!(header.reply_id, 9);
        reply.clone()
    }

    fn levels(level: i64, keys: &[&str]) -> Value {
        Value::Dictionary(
            keys.iter()
                .map(|k| ((*k).to_owned(), Value::Integer(level)))
                .collect(),
        )
    }

    fn integers(values: &[i64]) -> Value {
        Value::Array(values.iter().map(|&v| Value::Integer(v)).collect())
    }

    /// `decs` lists each decoder, its profiles and its level, as the module doc of `lists` says.
    #[test]
    fn the_decoder_list_names_each_decoder_with_its_profiles_and_level() {
        let mut decs = request(b"decs");
        decs.u32(names::VERSION, 6);
        let reply = one_reply(&Host::new(decoders()).deliver(&decs.finish()), b"decs");
        let message = Message::parse(&reply).unwrap();
        assert_eq!(message.u32(names::VERSION).unwrap(), Some(6));
        let entry = |id: &str, name: &str, codec: &str, profiles: &[i64], level: i64| {
            let details = profiles
                .iter()
                .map(|p| (p.to_string(), levels(level, &["VTMaxDecodeLevel"])))
                .collect();
            Value::dictionary([
                ("CMClassImplementationID", Value::string(id)),
                ("CMClassImplementationName", Value::string(name)),
                ("IsPreferredInternalPlugin", Value::Boolean(false)),
                ("VTCodecType", Value::Array(vec![Value::string(codec)])),
                ("VTRating", Value::Integer(400)),
                (
                    "VTDecoderCapabilities",
                    Value::dictionary([
                        ("VTPerProfileDetails", Value::Dictionary(details)),
                        ("VTSupportedProfileArray", integers(profiles)),
                    ]),
                ),
            ])
        };
        let expected = Value::Array(vec![
            entry(
                "reims-vgpu-video.videodecoder.avc",
                "reims-vgpu-video Decoder - AVC",
                "avc1",
                &[66, 77, 100],
                52,
            ),
            entry(
                "reims-vgpu-video.videodecoder.hevc",
                "reims-vgpu-video Decoder - HEVC",
                "hvc1",
                &[1, 2, 3],
                153,
            ),
        ]);
        assert_eq!(
            message.property_list_bytes(names::DECODERS).unwrap(),
            Some(encode(&expected).as_slice())
        );
    }

    /// A host with no decoder lists none.
    #[test]
    fn a_host_with_no_decoder_lists_none() {
        let mut decs = request(b"decs");
        decs.u32(names::VERSION, 6);
        let reply = one_reply(&Host::default().deliver(&decs.finish()), b"decs");
        let message = Message::parse(&reply).unwrap();
        assert_eq!(
            message.property_list_bytes(names::DECODERS).unwrap(),
            Some(encode(&Value::Array(Vec::new())).as_slice())
        );
    }

    /// `dcap` answers the capabilities of a codec the host decodes, and NULL for any other.
    #[test]
    fn a_decoders_capabilities_are_its_profiles_and_level() {
        let host = Host::new(decoders());
        let mut dcap = request(b"dcap");
        dcap.s32(names::CODEC, FourCc::new(*b"hvc1").0.cast_signed());
        let reply = one_reply(&host.deliver(&dcap.finish()), b"dcap");
        let message = Message::parse(&reply).unwrap();
        let support = |p: &str| {
            (
                p.to_owned(),
                Value::dictionary([
                    ("VTIsHardwareAccelerated", Value::Boolean(true)),
                    ("VTMaxDecodeLevel", Value::Integer(153)),
                    ("VTMaxPlaybackLevel", Value::Integer(153)),
                ]),
            )
        };
        let expected = Value::dictionary([
            ("VTIsHDRAllowedOnDevice", Value::Boolean(true)),
            (
                "VTPerProfileSupport",
                Value::Dictionary(vec![support("1"), support("2"), support("3")]),
            ),
            ("VTSupportedProfiles", integers(&[1, 2, 3])),
        ]);
        assert_eq!(
            message
                .property_list_bytes(names::DECODER_CAPABILITIES)
                .unwrap(),
            Some(encode(&expected).as_slice())
        );
        assert_eq!(message.u32(names::VERSION).unwrap(), Some(6));
        let mut vp9 = request(b"dcap");
        vp9.s32(names::CODEC, FourCc::new(*b"vp09").0.cast_signed());
        let reply = one_reply(&host.deliver(&vp9.finish()), b"dcap");
        let message = Message::parse(&reply).unwrap();
        assert_eq!(
            message
                .property_list_bytes(names::DECODER_CAPABILITIES)
                .unwrap(),
            Some(&[][..])
        );
    }

    /// `encs` and `meps` list nothing; `meps` from a guest older than protocol 6 is refused with
    /// no reply.
    #[test]
    fn no_encoder_and_no_motion_estimation_are_listed() {
        let host = Host::new(decoders());
        let mut encs = request(b"encs");
        encs.u32(names::VERSION, 6);
        let reply = one_reply(&host.deliver(&encs.finish()), b"encs");
        let message = Message::parse(&reply).unwrap();
        let empty = encode(&Value::Array(Vec::new()));
        assert_eq!(
            message.property_list_bytes(names::ENCODERS).unwrap(),
            Some(empty.as_slice())
        );
        let mut meps = request(b"meps");
        meps.u32(names::VERSION, 6);
        let reply = one_reply(&host.deliver(&meps.finish()), b"meps");
        let message = Message::parse(&reply).unwrap();
        assert_eq!(
            message
                .property_list_bytes(names::MOTION_ESTIMATORS)
                .unwrap(),
            Some(empty.as_slice())
        );
        let mut old = request(b"meps");
        old.u32(names::VERSION, 5);
        assert_eq!(
            host.deliver(&old.finish()),
            Answer::status(status::UNSUPPORTED_GUEST)
        );
    }

    /// A create is refused in its reply's `,err`, as for a codec the host lacks, and its close
    /// taken; a create with no codec type is malformed, with no reply.
    #[test]
    fn a_create_is_refused_in_its_reply_and_its_close_taken() {
        let host = Host::new(decoders());
        for (kind, refused) in [
            (b"+dec", status::COULD_NOT_FIND_VIDEO_DECODER),
            (b"+enc", status::COULD_NOT_FIND_VIDEO_ENCODER),
        ] {
            let mut create = request(kind);
            create.s32(names::CODEC, FourCc::new(*b"avc1").0.cast_signed());
            create.u32(names::VERSION, 6);
            let reply = one_reply(&host.deliver(&create.finish()), kind);
            let message = Message::parse(&reply).unwrap();
            assert_eq!(message.s32(names::ERROR).unwrap(), Some(refused));
            assert_eq!(message.u32(names::VERSION).unwrap(), Some(6));
            assert!(
                message
                    .property_list_bytes(names::HOST_INFO)
                    .unwrap()
                    .is_some()
            );
            let mut bare = request(kind);
            bare.u32(names::VERSION, 6);
            assert_eq!(
                host.deliver(&bare.finish()),
                Answer::status(status::MALFORMED_MESSAGE)
            );
        }
        let mut mep = request(b"+mep");
        mep.u32(names::VERSION, 6);
        let reply = one_reply(&host.deliver(&mep.finish()), b"+mep");
        let message = Message::parse(&reply).unwrap();
        assert_eq!(
            message.s32(names::ERROR).unwrap(),
            Some(status::MOTION_ESTIMATION_PROCESSOR_NOT_FOUND)
        );
        for kind in [b"-dec", b"-enc", b"-mep", b"-uui", b"-all"] {
            assert_eq!(
                host.deliver(&request(kind).finish()),
                Answer::status(status::NO_ERROR)
            );
        }
    }

    /// A JPEG request is answered with no JPEG hardware; a session's message, as for a UUID with
    /// none; bytes with no message header, as malformed.
    #[test]
    fn jpeg_session_and_unreadable_messages() {
        let host = Host::new(decoders());
        let reply = one_reply(&host.deliver(&request(b"jcap").finish()), b"jcap");
        let message = Message::parse(&reply).unwrap();
        assert_eq!(
            message.s32(names::ERROR).unwrap(),
            Some(status::NO_JPEG_HARDWARE)
        );
        assert_eq!(
            host.deliver(&request(b"dfrm").finish()),
            Answer::status(status::SESSION_NOT_FOUND)
        );
        assert_eq!(
            host.deliver(&[0; 48]),
            Answer::status(status::MALFORMED_MESSAGE)
        );
    }
}
