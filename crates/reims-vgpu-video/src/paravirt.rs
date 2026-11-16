//! VideoToolbox's paravirtual protocol, the host's side (docs/graphics-kext/design.md, "Decided",
//! 17): the `vtpv` messages an arm64 guest's VideoToolbox sends its host, which the arm64e kext
//! carries to the device as `VT_MESSAGE`s, and the host's answers.
//!
//! - [`atom`] and [`message`] read and write the messages, [`plist`] the property lists they
//!   carry, and [`image_description`] and [`sample_buffer`] the format descriptions and samples a
//!   decoder session's carry.
//! - [`Host`] answers them: the decoder list (`decs`) and a decoder's capabilities (`dcap`) from
//!   the decoders the host's video arm runs ([`Decoder`]), no encoder (`encs`) and no
//!   motion-estimation processor (`meps`). A decoder create (`+dec`) of a codec the list holds
//!   opens a session ([`decoder`]) where the host has a [`Backend`] to decode on; any other
//!   create is refused, as for a codec the host lacks. A close ends its session.
//! - [`status`] is the statuses the guest sees.

// The library's lint set, as `core`'s: no unsafe code, no indexing that can panic, and no silent
// `as` conversion outside tests.
#![deny(
    unsafe_code,
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unreachable,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    clippy::mod_module_files
)]

pub mod atom;
pub mod decoder;
mod host;
pub mod image_description;
mod known_keys;
mod lists;
pub mod message;
mod names;
pub mod plist;
pub mod sample_buffer;
pub mod status;

pub use decoder::{Backend, BackendDecoder, Decoded, DecoderConfig, LentSurface, Link};
pub use host::{Answer, Codec, Decoder, Host};

/// Why guest bytes could not be read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    /// The bytes do not follow the format: an atom's size, a parameter's length or type, a
    /// missing header.
    #[error("malformed: {0}")]
    Malformed(&'static str),
}

impl WireError {
    /// The status the guest sees for this failure.
    pub fn status(&self) -> status::OsStatus {
        match self {
            Self::Malformed(_) => status::MALFORMED_MESSAGE,
        }
    }
}
