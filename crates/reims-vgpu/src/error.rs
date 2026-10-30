//! [`Error`]: why a call refused, and [`ResumeError`]: why a resume refused its state.

use reims_vgpu_core::device::DeviceRefusal;
use reims_vgpu_core::display::DisplayRefusal;
use reims_vgpu_core::iosurface::IoSurfaceRefusal;

/// Why a call refused. Each create checks in core's order (docs/api.md section 1): the
/// variant names the check that failed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// A value outside its range, or `add_trace_range` without `remove_trace_range`.
    InvalidArgument,
    /// A display port count of 0, or above [`crate::MAX_DISPLAY_PORTS`].
    DisplayPortCount,
    /// No Metal device has the registry ID, or the system has none.
    NoMetalDevice,
    /// A memory map together with a task or memory callback.
    MemoryConflict,
    /// The memory map was refused: ranges that overlap or overflow.
    MemoryMap,
    /// Process isolation without a memory map.
    IsolationWithoutMap,
    /// Metal refused an object the device or display needs.
    Metal,
    /// The display's port is not below the device's port count.
    PortOutOfRange,
    /// Another display holds the port.
    PortInUse,
    /// A display of the process holds the serial number.
    SerialInUse,
    /// An IOSurface device without a memory map.
    NoMemoryMap,
    /// The device has no suspend state to give.
    Suspend,
    /// The device refused the resume state, with its reason when it stored one.
    Resume(Option<ResumeError>),
    /// The frame's GPU work ended with an error.
    Gpu,
    /// A frame committed twice, or waited on before its commit.
    FrameState,
    /// The host refused the memory, a pipe or a thread a device needs: the graphics device's
    /// region, or its threads.
    HostResources,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Error::InvalidArgument => "invalid argument",
            Error::DisplayPortCount => "display port count out of range",
            Error::NoMetalDevice => "no such Metal device",
            Error::MemoryConflict => "a memory map together with a task or memory callback",
            Error::MemoryMap => "the memory map was refused",
            Error::IsolationWithoutMap => "process isolation needs a memory map",
            Error::Metal => "Metal refused an object",
            Error::PortOutOfRange => "the port is out of range",
            Error::PortInUse => "the port is in use",
            Error::SerialInUse => "the serial number is in use",
            Error::NoMemoryMap => "the IOSurface device needs a memory map",
            Error::Suspend => "no suspend state",
            Error::Resume(_) => "the resume state was refused",
            Error::Gpu => "the frame's GPU work failed",
            Error::FrameState => "the frame is not in a state for this call",
            Error::HostResources => "the host refused memory, a pipe or a thread",
        })
    }
}

impl std::error::Error for Error {}

impl From<DeviceRefusal> for Error {
    fn from(r: DeviceRefusal) -> Error {
        match r {
            DeviceRefusal::DisplayPortCount => Error::DisplayPortCount,
            DeviceRefusal::NoMetalDevice => Error::NoMetalDevice,
            DeviceRefusal::MemoryConflict => Error::MemoryConflict,
            DeviceRefusal::MemoryMap => Error::MemoryMap,
            DeviceRefusal::TraceWithoutRemove => Error::InvalidArgument,
            DeviceRefusal::Metal => Error::Metal,
            DeviceRefusal::IsolationWithoutMap => Error::IsolationWithoutMap,
        }
    }
}

impl From<DisplayRefusal> for Error {
    fn from(r: DisplayRefusal) -> Error {
        match r {
            DisplayRefusal::PortOutOfRange => Error::PortOutOfRange,
            DisplayRefusal::Metal => Error::Metal,
            DisplayRefusal::SerialInUse => Error::SerialInUse,
            DisplayRefusal::PortInUse => Error::PortInUse,
        }
    }
}

impl From<IoSurfaceRefusal> for Error {
    fn from(r: IoSurfaceRefusal) -> Error {
        match r {
            IoSurfaceRefusal::NoMemoryMap => Error::NoMemoryMap,
            IoSurfaceRefusal::MemoryMap => Error::MemoryMap,
        }
    }
}

/// Why a resume refused its state: the codes the device stores, which the guest ABI fixes.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResumeError {
    /// An internal fault.
    InternalFault = 0,
    /// The state's version is not one the device reads.
    InvalidSuspendStateVersion = 1,
    /// The state is malformed, or absent.
    InvalidContent = 2,
    /// The guest's binary version in the state is not one the device supports.
    InvalidGuestVersion = 3,
    /// The state was taken on a device this one cannot stand in for.
    IncompatibleDevice = 4,
    /// The state's display port count differs from the device's.
    InvalidDisplayPortCount = 5,
}

impl ResumeError {
    /// The reason a device stored as `code`, or `None` for a code it never stores.
    pub fn from_code(code: u64) -> Option<ResumeError> {
        Some(match code {
            0 => ResumeError::InternalFault,
            1 => ResumeError::InvalidSuspendStateVersion,
            2 => ResumeError::InvalidContent,
            3 => ResumeError::InvalidGuestVersion,
            4 => ResumeError::IncompatibleDevice,
            5 => ResumeError::InvalidDisplayPortCount,
            _ => return None,
        })
    }

    /// The code.
    pub fn code(self) -> u32 {
        self as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each core refusal names its variant; `add_trace_range` alone is an invalid argument
    #[test]
    fn refusals_map_to_their_variant() {
        assert_eq!(
            Error::from(DeviceRefusal::TraceWithoutRemove),
            Error::InvalidArgument
        );
        assert_eq!(
            Error::from(DeviceRefusal::DisplayPortCount),
            Error::DisplayPortCount
        );
        assert_eq!(Error::from(DisplayRefusal::PortInUse), Error::PortInUse);
        assert_eq!(Error::from(IoSurfaceRefusal::MemoryMap), Error::MemoryMap);
        assert!(!Error::Resume(None).to_string().is_empty());
    }

    /// The six codes round-trip; any other is no reason
    #[test]
    fn resume_codes() {
        for code in 0..6u32 {
            assert_eq!(
                ResumeError::from_code(u64::from(code)).map(ResumeError::code),
                Some(code)
            );
        }
        assert_eq!(ResumeError::from_code(6), None);
    }
}
