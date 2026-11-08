//! VA-API: decode, and encode where the driver has it, on the host's GPU
//! through libva alone, with no FFmpeg or other layer between, over the
//! codecs of [`crate::codec`].
//!
//! - The libva binding and the decoders on it: [`Display`],
//!   [`H264Decoder`], [`HevcDecoder`], [`Frame`].
//! - [`VideoEncoder`] runs [`crate::codec::encode`]'s encoders on VA-API.

mod va;

pub use va::{
    Capabilities, ConfigCapabilities, Display, Entrypoint, Error, ExportedPlane, Fourcc, Frame,
    FrameHold, GuestWrites, H264_MAX_MACROBLOCKS, H264Decoder, HevcDecoder, ImportWrite,
    ImportWrites, LoadError, PlaneRows, Profile, ReadBack, ReadbackTimes, ReadbackWay,
    SurfaceCapabilities, SurfaceExport, proc_filter_name, read_back_by, rt_format_names,
};

pub use va::{
    EncoderCapabilities, EncoderSupport, Feature, HevcBlockSizes, HevcFeatures, UploadMethod,
    VideoEncoder, encoder_support,
};
