//! The catch on Linux: a Rust unwind out of the host's code that no panic hook ended first ends
//! the process at the call.

use reims_vgpu_vulkan::core::testkit::capture::catch_death;

use super::*;

/// An unwind the panic hook never saw (`resume_unwind`) is fatal at the call
#[test]
fn a_rust_unwind_from_the_host_is_fatal_at_the_call() {
    let d = catch_death(|| host_call("raise_interrupt", || std::panic::resume_unwind(Box::new(7))));
    assert_eq!(d.unwrap_err(), host_raised("raise_interrupt"));
}
