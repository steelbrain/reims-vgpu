//! A helper's watch on Windows: its private commit, a Job Object that ends it with the library's
//! process, and the host's memory and CPUs.

use std::io;
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt as _;
use std::process::{Child, Command};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
use windows_sys::Win32::System::Threading::{
    ALL_PROCESSOR_GROUPS, CREATE_NO_WINDOW, GetActiveProcessorCount,
};

/// The host's memory status; `None` where Windows does not say.
fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: a MEMORYSTATUSEX is plain data, which zero bytes make a valid one.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = u32::try_from(size_of::<MEMORYSTATUSEX>()).ok()?;
    // SAFETY: `status` is a MEMORYSTATUSEX whose length is set, as the call asks.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

/// The host's physical memory, in bytes; `None` where Windows does not say.
pub(super) fn host_memory() -> Option<u64> {
    memory_status().map(|status| status.ullTotalPhys)
}

/// What the host can give now, in bytes: the smaller of its available physical memory and the
/// physical memory no commit charge has taken, as a helper's footprint is its commit; `None`
/// where Windows does not say.
pub(super) fn free_memory() -> Option<u64> {
    memory_status().map(|status| {
        let charged = status
            .ullTotalPageFile
            .saturating_sub(status.ullAvailPageFile);
        (status.ullAvailPhys).min(status.ullTotalPhys.saturating_sub(charged))
    })
}

/// Nothing: the guest's RAM is a pagefile-backed section, charged in full at its creation, which
/// the free memory ([`free_memory`]) leaves out already.
pub(super) fn unresident(_: &[(u64, u64)]) -> u64 {
    0
}

/// The host's logical CPUs, in every processor group; `None` where Windows does not say.
pub(super) fn host_cpus() -> Option<u64> {
    // SAFETY: GetActiveProcessorCount takes a group number, and no pointer.
    let count = unsafe { GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) };
    (count > 0).then_some(u64::from(count))
}

/// Has the child run with no console window of its own, so a library's process with no console
/// (one started detached) shows none for its helper; its standard handles are the ones the
/// command names all the same. The Job Object ends it ([`Watched::new`]).
pub(super) fn prepare(command: &mut Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

/// Nothing: a write to a helper that has ended fails, and Windows raises no signal.
pub(super) fn mask_broken_pipes() {}

/// The helper's end comes with its parent's by the Job Object: nothing of the helper's is needed.
pub(super) fn watch_parent() -> io::Result<()> {
    Ok(())
}

/// A running helper in a Job Object of its own, which kills it once the job's one handle, the
/// library's, closes: at the end of the watch, or of the library's process.
#[derive(Debug)]
pub(super) struct Watched {
    job: HANDLE,
    /// The child's handle, which outlives the watch (`run_watched` drops the watch first).
    process: HANDLE,
}

impl Watched {
    pub(super) fn new(child: &Child) -> io::Result<Watched> {
        // SAFETY: no attributes, and no name: a new job.
        let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if job.is_null() {
            return Err(io::Error::last_os_error());
        }
        let watched = Watched {
            job,
            process: child.as_raw_handle(),
        };
        // SAFETY: a JOBOBJECT_EXTENDED_LIMIT_INFORMATION is plain data, which zero bytes make a
        // valid one: no limits.
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let size = u32::try_from(size_of_val(&limits)).map_err(io::Error::other)?;
        // SAFETY: the job is ours, and `limits` is the class's struct, of `size` bytes.
        let set = unsafe {
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const limits).cast(),
                size,
            )
        };
        if set == 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the job is ours, and the process handle the child's, which is open.
        if unsafe { AssignProcessToJobObject(job, watched.process) } == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(watched)
    }

    /// Its private commit, in bytes (`PrivateUsage`).
    pub(super) fn footprint(&self) -> io::Result<u64> {
        // SAFETY: a PROCESS_MEMORY_COUNTERS_EX is plain data, which zero bytes make a valid one.
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        let size =
            u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS_EX>()).map_err(io::Error::other)?;
        counters.cb = size;
        // SAFETY: the process handle is open, and `counters` the extended struct of `size` bytes,
        // which the call fills where `cb` names its size.
        let read = unsafe {
            K32GetProcessMemoryInfo(
                self.process,
                (&raw mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
                size,
            )
        };
        if read == 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(u64::try_from(counters.PrivateUsage).unwrap_or(u64::MAX))
    }
}

impl Drop for Watched {
    /// Closes the job: a process still in it ends.
    fn drop(&mut self) {
        // SAFETY: the job handle is ours, closed once.
        unsafe { CloseHandle(self.job) };
    }
}

/// Whether the process `pid` runs.
#[cfg(test)]
pub(super) fn alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SYNCHRONIZE, WaitForSingleObject,
    };
    // SAFETY: a handle opened for waiting only, closed below.
    let process = unsafe { OpenProcess(PROCESS_SYNCHRONIZE, 0, pid) };
    if process.is_null() {
        return false;
    }
    // SAFETY: the handle is open; a wait of 0 ms only looks.
    let ended = unsafe { WaitForSingleObject(process, 0) } == WAIT_OBJECT_0;
    // SAFETY: the handle is ours, closed once.
    unsafe { CloseHandle(process) };
    !ended
}
