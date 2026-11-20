//! The trace's clock ([`trace_clock`]).

/// The trace's clock: rt's host clock, in nanoseconds (`metal2vulkan_runtime::host_time`), the clock
/// rt's command buffers' times count, so a command buffer's times and the trace's events are of
/// one clock.
pub fn trace_clock() -> u64 {
    metal2vulkan_runtime::host_time()
}
