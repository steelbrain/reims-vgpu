//! The clock on Windows: the performance counter (`QueryPerformanceCounter`), the host time domain
//! Vulkan's calibrated timestamps name there (`VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR`), so
//! the GPU's times the device reports and this clock are one.

use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

/// The host's monotonic clock, in nanoseconds: the counter's ticks scaled by its frequency, which
/// is fixed at boot.
pub(crate) fn now_ns() -> u64 {
    let mut ticks = 0i64;
    let mut frequency = 0i64;
    // SAFETY: each call writes the one integer it is given, and cannot fail on Windows XP or
    // later.
    unsafe {
        QueryPerformanceCounter(&mut ticks);
        QueryPerformanceFrequency(&mut frequency);
    }
    let ticks = u128::from(u64::try_from(ticks).unwrap_or(0));
    let Some(frequency) = u64::try_from(frequency)
        .ok()
        .and_then(std::num::NonZeroU64::new)
    else {
        return 0;
    };
    let nanos = ticks.saturating_mul(1_000_000_000) / u128::from(frequency.get());
    u64::try_from(nanos).unwrap_or(u64::MAX)
}
