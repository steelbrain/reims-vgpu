//! The `Backend` trait implementation for device lifecycle.
//!
//! Probing the device is [`super::runtime`]'s job, not this module's. Two
//! wrappers here used to say otherwise: a `MetalRuntime` unit struct whose one
//! associated function forwarded `system_device`, and a `system_device_name`
//! that forwarded the identically-named function it imported. Neither was
//! constructed or called anywhere outside this file's own test, and the second
//! put one name on two functions in two modules — so a `grep` for it reported
//! two producers and the arm a reader landed on was arbitrary.

use crate::backend::metal::runtime::system_device;
use crate::backend::Backend;

/// Device lifecycle handle; product encode is the C ABI in `ffi`.
///
/// `ready` and `name` have no caller outside this file's test, and `ready` is
/// kept anyway because `new`'s `system_device()` call is the side effect that
/// first creates the process-global `MTLDevice`. Dropping the field to quiet
/// the lint would move when that happens.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct MetalBackend {
    ready: bool,
}

#[allow(dead_code)] // `ready` and `name` — see the type's doc.
impl MetalBackend {
    pub fn new() -> Self {
        Self {
            ready: system_device().is_some(),
        }
    }

    pub fn ready(&self) -> bool {
        self.ready
    }

    pub fn name(&self) -> &'static str {
        "metal"
    }
}

impl Backend for MetalBackend {
    fn reset(&mut self) {
        crate::runtime::icb::clear_icb_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::runtime::system_device_name;

    /// Named for what it asserts. It was called `system_device`, which shadowed
    /// the imported function of that name inside the test module.
    #[test]
    fn the_probe_finds_a_device_and_the_backend_reports_it_ready() {
        assert!(system_device().is_some());
        assert!(system_device_name().is_some());
        assert!(MetalBackend::new().ready());
    }
}
