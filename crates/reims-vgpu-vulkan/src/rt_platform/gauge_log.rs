//! rt's memory gauges in the host's log, so a boot's reader can tell the memory rt holds from the
//! memory its objects use while the boot runs: every [`PERIOD`], while the target [`TARGET`] is on
//! at debug, one line of the gauges of the process's latest device (`super::device::gauges`)
//! where they changed since the line before. A diagnostic: the gauges are rt's own, which no
//! Metal call answers, and which lines the host writes is free under the contract.
//!
//! - **Asked for by name.** [`start`] makes the thread at a device's open where `REIMS_VGPU_LOG`
//!   names the target itself at debug (`info,reims_vgpu::rt_gauges=debug`:
//!   `crate::core::logging::names`), as `crate::work_log` does. A lone level, such as a bug
//!   report's `debug`, starts none, nor does a test's capture of the log, so no line comes
//!   between the lines a test reads. Off, no thread is made. Reading the gauges is a load of each
//!   of rt's counters, which costs the work they count nothing.
//! - **One thread a process.** It reads the latest device's gauges at each look, so a device made
//!   after a let-go is read with no thread of its own, and a device torn down reads as zeros.
//! - **A line** is `rt gauges: NAME VALUE, ...` ([`line`]), each name the field of rt's `Gauges`
//!   it reads, bytes and counts as rt keeps them: the Private and the Shared pool's allocations,
//!   their idle allocations and reserve, the blocks taken from them, the live buffers in them and
//!   the long allocations held again, the textures' places in texture blocks, which are Private
//!   blocks, and outside the pools, the textures' allocations of their own, the spills, the
//!   imports and the surface imports; then the live `VkDeviceMemory` allocations, the imports,
//!   surface imports and exports among them, and the live buffers and textures. A look at which
//!   none changed writes no line ([`Lines`]).

use std::sync::Once;
use std::time::Duration;

use metal2vulkan_rt as rt;

/// The log target of the lines, at debug, which [`start`] asks the log for.
pub(crate) const TARGET: &str = "reims_vgpu::rt_gauges";

/// How often the thread looks at the gauges.
pub(crate) const PERIOD: Duration = Duration::from_secs(5);

/// The gauges a line holds, in its order, each by its field's name.
macro_rules! named {
    ($gauges:expr; $($field:ident),* $(,)?) => {
        [$((stringify!($field), $gauges.$field)),*]
    };
}

/// The line of `gauges` (the module docs).
pub(crate) fn line(gauges: &rt::Gauges) -> String {
    let named = named!(gauges;
        private_allocation_bytes,
        private_idle_bytes,
        private_reserve_bytes,
        private_placed_bytes,
        private_buffer_bytes,
        texture_block_bytes,
        private_long_reused_bytes,
        shared_allocation_bytes,
        shared_idle_bytes,
        shared_reserve_bytes,
        shared_placed_bytes,
        shared_buffer_bytes,
        shared_long_reused_bytes,
        texture_own_bytes,
        spilled_bytes,
        managed_spilled_bytes,
        import_bytes,
        surface_import_bytes,
        blocks,
        imports,
        surface_imports,
        exports,
        buffers,
        textures,
    );
    let pairs: Vec<String> = (named.iter())
        .map(|(name, value)| format!("{name} {value}"))
        .collect();
    format!("rt gauges: {}", pairs.join(", "))
}

/// The line before.
#[derive(Default)]
pub(crate) struct Lines {
    last: Option<String>,
}

impl Lines {
    /// The line of `gauges`, or None where it is the line before's.
    pub(crate) fn next(&mut self, gauges: &rt::Gauges) -> Option<String> {
        let now = line(gauges);
        if self.last.as_ref() == Some(&now) {
            return None;
        }
        self.last = Some(now.clone());
        Some(now)
    }
}

/// Starts the thread that logs the gauges (the module docs), once a process, where
/// `REIMS_VGPU_LOG` names [`TARGET`] at debug and its lines are written.
pub(super) fn start() {
    static STARTED: Once = Once::new();
    if !(crate::core::logging::names(TARGET, log::Level::Debug)
        && log::log_enabled!(target: TARGET, log::Level::Debug))
    {
        return;
    }
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu-gauges".into())
            .spawn(|| {
                let mut lines = Lines::default();
                loop {
                    std::thread::sleep(PERIOD);
                    if !log::log_enabled!(target: TARGET, log::Level::Debug) {
                        continue;
                    }
                    let Some(gauges) = super::device::gauges() else {
                        continue;
                    };
                    if let Some(line) = lines.next(&gauges) {
                        log::debug!(target: TARGET, "{line}");
                    }
                }
            });
        if let Err(error) = spawned {
            log::debug!(target: TARGET, "the gauges' thread was not made: {error}");
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gauges_line_names_each_gauge_and_comes_again_only_on_a_change() {
        let mut gauges = rt::Gauges::default();
        gauges.private_allocation_bytes = 33_554_432;
        gauges.private_idle_bytes = 16_777_216;
        gauges.private_reserve_bytes = 16_777_216;
        gauges.private_placed_bytes = 8_388_608;
        gauges.private_buffer_bytes = 1_048_576;
        gauges.texture_block_bytes = 4_194_304;
        gauges.shared_allocation_bytes = 16_777_216;
        gauges.shared_placed_bytes = 65_536;
        gauges.shared_buffer_bytes = 65_536;
        gauges.texture_own_bytes = 2_097_152;
        gauges.import_bytes = 3_145_728;
        gauges.blocks = 6;
        gauges.imports = 3;
        gauges.buffers = 4;
        gauges.textures = 5;
        let expected = "rt gauges: private_allocation_bytes 33554432, private_idle_bytes 16777216, \
                        private_reserve_bytes 16777216, private_placed_bytes 8388608, \
                        private_buffer_bytes 1048576, texture_block_bytes 4194304, \
                        private_long_reused_bytes 0, shared_allocation_bytes 16777216, \
                        shared_idle_bytes 0, shared_reserve_bytes 0, shared_placed_bytes 65536, \
                        shared_buffer_bytes 65536, shared_long_reused_bytes 0, \
                        texture_own_bytes 2097152, spilled_bytes 0, managed_spilled_bytes 0, \
                        import_bytes 3145728, surface_import_bytes 0, blocks 6, imports 3, \
                        surface_imports 0, exports 0, buffers 4, textures 5";
        assert_eq!(line(&gauges), expected);

        let mut lines = Lines::default();
        assert_eq!(lines.next(&gauges).as_deref(), Some(expected));
        assert_eq!(lines.next(&gauges), None, "nothing changed");
        gauges.textures = 6;
        let changed = lines.next(&gauges).expect("a gauge changed");
        assert!(changed.ends_with("buffers 4, textures 6"), "{changed}");
        let zeros = line(&rt::Gauges::default());
        assert_eq!(lines.next(&rt::Gauges::default()), Some(zeros), "a torn-down device's");
    }
}
