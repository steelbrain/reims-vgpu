//! The statuses the guest sees: the `OSStatus` the host answers each message with, and the `,err`
//! of its replies. Each value is Apple's: the VideoToolbox ones are public (`VTErrors.h` in the
//! macOS SDK), the paravirtual protocol's are its host's and guest's own. The names are ours, from
//! Apple's constant names where it has one.

/// The status of an operation: 0 is success, anything else an `OSStatus` error.
pub type OsStatus = i32;

/// Success.
pub const NO_ERROR: OsStatus = 0;
/// `kVTPropertyNotSupportedErr`.
pub const PROPERTY_NOT_SUPPORTED: OsStatus = -12900;
/// `kVTPropertyReadOnlyErr`.
pub const PROPERTY_READ_ONLY: OsStatus = -12901;
/// `kVTParameterErr`.
pub const PARAMETER: OsStatus = -12902;
/// `kVTCouldNotFindVideoDecoderErr`: a decoder create for a codec the host has no decoder of.
pub const COULD_NOT_FIND_VIDEO_DECODER: OsStatus = -12906;
/// `kVTCouldNotFindVideoEncoderErr`: likewise, an encoder create.
pub const COULD_NOT_FIND_VIDEO_ENCODER: OsStatus = -12908;
/// `kVTVideoDecoderBadDataErr`: a sample that does not decode, a second start.
pub const VIDEO_DECODER_BAD_DATA: OsStatus = -12909;
/// `kVTVideoDecoderUnsupportedDataFormatErr`: a format description no decoder here takes.
pub const VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT: OsStatus = -12910;
/// `kVTVideoDecoderMalfunctionErr`.
pub const VIDEO_DECODER_MALFUNCTION: OsStatus = -12911;
/// `kVTUnimplementedErr`: a request the decoder does not carry out.
pub const UNIMPLEMENTED: OsStatus = -12782;
/// A message that does not parse, or that the receiver cannot take (kVTPVMalformedMessageErr).
pub const MALFORMED_MESSAGE: OsStatus = -19093;
/// A message for a UUID with no session.
pub const SESSION_NOT_FOUND: OsStatus = -19094;
/// No motion-estimation processor matches (kVTMESProcessorNotFoundErr).
pub const MOTION_ESTIMATION_PROCESSOR_NOT_FOUND: OsStatus = -19350;
/// A motion-estimation list request from a guest older than protocol 6
/// (kVTMESUnsupportedGuestErr).
pub const UNSUPPORTED_GUEST: OsStatus = -19960;
/// CMPhoto's status where it has no JPEG hardware, which a JPEG request is answered with. The
/// name is ours.
pub const NO_JPEG_HARDWARE: OsStatus = -16994;
