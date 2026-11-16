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

/// `Dpba`: the attributes the guest's output buffers are to have.
pub(crate) const SET_PIXEL_BUFFER_ATTRIBUTES: FourCc = FourCc::new(*b"Dpba");
/// `Dbuf`: ask the guest for an output buffer.
pub(crate) const CREATE_PIXEL_BUFFER: FourCc = FourCc::new(*b"Dbuf");
/// `Demt`: emit a decoded frame.
pub(crate) const EMIT_DECODED_FRAME: FourCc = FourCc::new(*b"Demt");
/// `Dfub`: output buffers let go of unemitted.
pub(crate) const FREE_UNUSED_PIXEL_BUFFERS: FourCc = FourCc::new(*b"Dfub");

/// `,fds`: a format description, `=fds`.
pub(crate) const FORMAT_DESCRIPTION: FourCc = FourCc::new(*b",fds");
/// `,blu`: the guest's buffer pool, `=s32`.
pub(crate) const GUEST_POOL: FourCc = FourCc::new(*b",blu");
/// `,frm`: a frame's identifier, `=s64`.
pub(crate) const FRAME: FourCc = FourCc::new(*b",frm");
/// `,sbf`: a sample buffer, `=sbf`.
pub(crate) const SAMPLE_BUFFER: FourCc = FourCc::new(*b",sbf");
/// `,dfl`: the decode flags, `=u32`.
pub(crate) const DECODE_FLAGS: FourCc = FourCc::new(*b",dfl");
/// `,inf`: the decode info flags, `=u32`.
pub(crate) const INFO_FLAGS: FourCc = FourCc::new(*b",inf");
/// `,uui`: a pixel buffer's UUID, `=uui`.
pub(crate) const PIXEL_BUFFER_UUID: FourCc = FourCc::new(*b",uui");
/// `,ui+`: pixel buffers' UUIDs, `=ui+`.
pub(crate) const PIXEL_BUFFER_UUIDS: FourCc = FourCc::new(*b",ui+");
/// `,pba`: pixel buffer attributes, `=pls`.
pub(crate) const PIXEL_BUFFER_ATTRIBUTES: FourCc = FourCc::new(*b",pba");
/// `,key`: a property's key, `=pls`.
pub(crate) const PROPERTY_KEY: FourCc = FourCc::new(*b",key");
/// `,val`: a property's value, `=pls`.
pub(crate) const PROPERTY_VALUE: FourCc = FourCc::new(*b",val");
/// `,sup`: the supported properties, `=pls`.
pub(crate) const SUPPORTED_PROPERTIES: FourCc = FourCc::new(*b",sup");
/// `,pro`: properties, `=pls`.
pub(crate) const PROPERTIES: FourCc = FourCc::new(*b",pro");
/// `,can`: whether the decoder goes on with a format description, `=s32`.
pub(crate) const CAN_ACCEPT: FourCc = FourCc::new(*b",can");
