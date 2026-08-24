//! Observation-only output port for Vulkan execution measurements.
//!
//! The backend may publish facts through this port. It never reads them back,
//! so telemetry cannot influence guest-visible execution policy.

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadbackPhase {
    Submit,
    Fence,
    Map,
    Write,
    Vouch,
    Resolve,
}

pub trait BackendTelemetry: Send + Sync {
    fn route(&self, _name: &'static str) {}
    fn route_n(&self, _name: &'static str, _count: u64) {}
    fn route_us(&self, _name: &'static str, _micros: u64) {}
    fn readback_phase(&self, _phase: ReadbackPhase, _micros: u64) {}
    fn readback_gpu_us(&self, _barrier: u64, _copy: u64) {}
    fn guest_imports_invalidated(&self) {}
}

static TELEMETRY: OnceLock<&'static dyn BackendTelemetry> = OnceLock::new();

/// Install the process-wide observation consumer. A second install is benign.
pub fn install(telemetry: &'static dyn BackendTelemetry) {
    let _ = TELEMETRY.set(telemetry);
}

fn with_telemetry(f: impl FnOnce(&dyn BackendTelemetry)) {
    if let Some(telemetry) = TELEMETRY.get() {
        f(*telemetry);
    }
}

pub fn note_route(name: &'static str) {
    with_telemetry(|telemetry| telemetry.route(name));
}

pub fn note_route_n(name: &'static str, count: u64) {
    with_telemetry(|telemetry| telemetry.route_n(name, count));
}

pub fn note_route_us(name: &'static str, micros: u64) {
    with_telemetry(|telemetry| telemetry.route_us(name, micros));
}

pub fn note_readback_phase(phase: ReadbackPhase, micros: u64) {
    with_telemetry(|telemetry| telemetry.readback_phase(phase, micros));
}

pub fn note_readback_gpu_us(barrier: u64, copy: u64) {
    with_telemetry(|telemetry| telemetry.readback_gpu_us(barrier, copy));
}

pub fn guest_imports_invalidated() {
    with_telemetry(|telemetry| telemetry.guest_imports_invalidated());
}
