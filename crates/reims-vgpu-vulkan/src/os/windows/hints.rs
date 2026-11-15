//! The scheduling hint a thread that walks the guest's command buffers gives Windows
//! ([`hint_walk_thread`]): it opts out of execution-speed throttling.
//!
//! A walk thread of the graphics device's lanes walks a command buffer's ops one after another, so
//! wherever the GPU keeps up its speed is the frame rate; it gives itself this hint as it starts
//! (`Threads::give_walk_hints`). Windows picks a thread's quality of service itself, and may run
//! a thread of a process whose window is minimized or hidden at its efficiency level (EcoQoS:
//! lower clocks, and the efficiency cores of a hybrid processor). A thread may opt out of that
//! throttling (`SetThreadInformation` with `ThreadPowerThrottling`, the execution-speed bit in
//! the control mask and none in the state mask), so the guest's frames keep their pace while
//! QEMU's window is not in front (read, Microsoft's documentation of `SetThreadInformation` and
//! of quality of service).
//!
//! - **What is not given.** The thread's priority stays its own: a raised priority would take the
//!   CPU from the host's other work, which macOS's and Linux's hints never do. No CPU is chosen.
//! - **The threads the library starts** set no throttling state of their own. A lane's thread
//!   gives itself the hint as it starts. The runtime's threads are its own.
//!
//! The first hint's outcome is logged once.

use std::sync::atomic::{AtomicBool, Ordering};

use windows_sys::Win32::System::Threading::{
    GetCurrentThread, SetThreadInformation, THREAD_POWER_THROTTLING_CURRENT_VERSION,
    THREAD_POWER_THROTTLING_EXECUTION_SPEED, THREAD_POWER_THROTTLING_STATE, ThreadPowerThrottling,
};

use crate::core::num::ToU64 as _;

/// The state a walk thread asks for: execution-speed throttling controlled, and off.
const UNTHROTTLED: THREAD_POWER_THROTTLING_STATE = THREAD_POWER_THROTTLING_STATE {
    Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
    ControlMask: THREAD_POWER_THROTTLING_EXECUTION_SPEED,
    StateMask: 0,
};

/// Gives the calling thread, a walk thread, its scheduling hint (the module docs). The first call
/// logs what Windows took.
pub(crate) fn hint_walk_thread() {
    let size = std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>().to_u64();
    // SAFETY: the running thread's pseudo-handle, and a live state of the size given.
    let ok = unsafe {
        SetThreadInformation(
            GetCurrentThread(),
            ThreadPowerThrottling,
            std::ptr::from_ref(&UNTHROTTLED).cast(),
            u32::try_from(size).unwrap_or(u32::MAX),
        )
    };
    let line = if ok != 0 {
        "walk threads opt out of Windows's execution-speed throttling".to_owned()
    } else {
        format!(
            "walk threads give the scheduler no hint: SetThreadInformation refused it ({})",
            std::io::Error::last_os_error()
        )
    };
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::info!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::System::Threading::GetThreadInformation;

    use super::*;

    /// The calling thread's throttling state, as Windows reads it back.
    fn state() -> THREAD_POWER_THROTTLING_STATE {
        let mut state = THREAD_POWER_THROTTLING_STATE {
            Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: 0,
            StateMask: 0,
        };
        let size = std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>().to_u64();
        // SAFETY: the running thread's pseudo-handle, and a live state of the size given.
        let ok = unsafe {
            GetThreadInformation(
                GetCurrentThread(),
                ThreadPowerThrottling,
                std::ptr::from_mut(&mut state).cast(),
                u32::try_from(size).unwrap(),
            )
        };
        assert_ne!(ok, 0, "{}", std::io::Error::last_os_error());
        state
    }

    /// A walk thread opts out of execution-speed throttling, where a fresh thread had set no
    /// state of its own
    #[test]
    fn a_walk_thread_opts_out_of_throttling() {
        let [before, control, state] = std::thread::scope(|s| {
            s.spawn(|| {
                let before = state();
                hint_walk_thread();
                let after = state();
                [before.ControlMask, after.ControlMask, after.StateMask]
            })
            .join()
            .unwrap()
        });
        assert_eq!(before, 0, "a fresh thread's control mask");
        assert_eq!(control, THREAD_POWER_THROTTLING_EXECUTION_SPEED);
        assert_eq!(state, 0);
    }
}
