//! The message types and parameter keys the host reads and writes, as the wire spells them. Each
//! is Apple's, as the appender or getter that uses it spells it; the Rust names are ours.

use crate::core::fourcc::FourCc;

/// A message the host answers itself, before any session's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostMessage {
    /// `decs`: list the host's decoders.
    DecoderList,
    /// `dcap`: a decoder's capabilities.
    DecoderCapabilities,
    /// `encs`: list the host's encoders.
    EncoderList,
    /// `meps`: list the host's motion-estimation processors.
    MotionEstimationList,
    /// `+dec`: open a decoder session.
    CreateDecoder,
    /// `+enc`: open an encoder session.
    CreateEncoder,
    /// `+mep`: open a motion-estimation session.
    CreateMotionEstimation,
    /// `-all`, `-dec`, `-enc`, `-mep`, `-uui`: close sessions.
    Destroy,
    /// `jcap`, `jdec`, `jenc`: the JPEG engine's.
    Jpeg,
}

impl HostMessage {
    const ALL: [(Self, [u8; 4]); 15] = [
        (Self::DecoderList, *b"decs"),
        (Self::DecoderCapabilities, *b"dcap"),
        (Self::EncoderList, *b"encs"),
        (Self::MotionEstimationList, *b"meps"),
        (Self::CreateDecoder, *b"+dec"),
        (Self::CreateEncoder, *b"+enc"),
        (Self::CreateMotionEstimation, *b"+mep"),
        (Self::Destroy, *b"-all"),
        (Self::Destroy, *b"-dec"),
        (Self::Destroy, *b"-enc"),
        (Self::Destroy, *b"-mep"),
        (Self::Destroy, *b"-uui"),
        (Self::Jpeg, *b"jcap"),
        (Self::Jpeg, *b"jdec"),
        (Self::Jpeg, *b"jenc"),
    ];

    /// The message `message_type` names, if the host answers it itself.
    pub(crate) fn from_fourcc(message_type: FourCc) -> Option<Self> {
        Self::ALL
            .iter()
            .find(|(_, code)| FourCc::new(*code) == message_type)
            .map(|(message, _)| *message)
    }
}

/// `,ver`: a protocol version, `=u32`.
pub(crate) const VERSION: FourCc = FourCc::new(*b",ver");
/// `,err`: a status, `=s32`.
pub(crate) const ERROR: FourCc = FourCc::new(*b",err");
/// `,cod`: a codec type, `=s32`.
pub(crate) const CODEC: FourCc = FourCc::new(*b",cod");
/// `,dec`: the decoder list, `=pls`.
pub(crate) const DECODERS: FourCc = FourCc::new(*b",dec");
/// `,enc`: the encoder list, `=pls`.
pub(crate) const ENCODERS: FourCc = FourCc::new(*b",enc");
/// `,mep`: the motion-estimation processor list, `=pls`.
pub(crate) const MOTION_ESTIMATORS: FourCc = FourCc::new(*b",mep");
/// `,dca`: a decoder's capabilities, `=pls`.
pub(crate) const DECODER_CAPABILITIES: FourCc = FourCc::new(*b",dca");
/// `,hin`: the host's info, `=pls`.
pub(crate) const HOST_INFO: FourCc = FourCc::new(*b",hin");
