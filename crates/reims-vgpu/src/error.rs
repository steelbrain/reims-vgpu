//! [`Error`]: why a call refused.

/// Why a call refused.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A value outside its range.
    InvalidArgument,
    /// The platform refused an object the call needs.
    Metal,
    /// The call's GPU work ended with an error.
    Gpu,
    /// The host refused the memory, a pipe or a thread a device needs: the graphics device's
    /// region, or its threads.
    HostResources,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Error::InvalidArgument => "invalid argument",
            Error::Metal => "the platform refused an object",
            Error::Gpu => "the GPU work failed",
            Error::HostResources => "the host refused memory, a pipe or a thread",
        })
    }
}

impl std::error::Error for Error {}
