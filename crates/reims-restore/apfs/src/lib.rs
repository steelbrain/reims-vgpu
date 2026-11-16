//! Reading APFS: a container (bare, or the APFS partition of a GPT disk), its volumes, and a
//! file's content by path, compressed files decoded.
//!
//! reims-restore reads files out of an IPSW's system image with it, the image decoded by its own
//! readers into a [`dev::Source`]. A path is resolved by searching the volume's trees from their
//! roots, so a read touches only the nodes on its way. Everything read is checked: each object's
//! Fletcher-64 checksum, and its identifier, type and transaction against what led to it; each
//! node's keys ascend, and sit under its parent's key for it.
//!
//! ```no_run
//! # fn main() -> reims_apfs::Result<()> {
//! let image = std::sync::Arc::new(std::fs::File::open("os.raw")?);
//! let c = reims_apfs::open(image)?;
//! let v = c.system_volume()?;
//! let booter = v.read_file("/System/Library/Frameworks/Virtualization.framework/Resources/AVPBooter.vmapple2.bin")?;
//! # let _ = booter; Ok(()) }
//! ```

pub mod canon;
pub mod decmpfs;
pub mod dev;
pub mod file;
pub mod file_at;
pub mod fletcher;
pub mod fs;
pub mod gpt;
pub mod names;
pub mod ondisk;
pub mod read;
pub mod seal;
pub mod write;

mod btree;
mod le;

pub use file::{Entry, Extent, open};
pub use read::{Container, NodeShape, SealFacts, Volume};

/// What goes wrong reading a disk.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// An object whose checksum does not match its bytes.
    #[error("block {block}: checksum mismatch (stored {stored:#018x}, computed {computed:#018x})")]
    Checksum {
        block: u64,
        stored: u64,
        computed: u64,
    },
    /// A structure that breaks the format.
    #[error("block {block}: {what}")]
    Corrupt { block: u64, what: String },
    /// Something the format allows that this crate does not handle.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A request that cannot be met: a path that is not there, a file of the wrong kind.
    #[error("{0}")]
    Invalid(String),
    #[error("GPT: {0}")]
    Gpt(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

pub(crate) fn corrupt(block: u64, what: impl Into<String>) -> Error {
    Error::Corrupt {
        block,
        what: what.into(),
    }
}
