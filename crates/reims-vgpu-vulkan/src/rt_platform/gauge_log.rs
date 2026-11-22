//! rt's gauges looked at while a boot runs: every [`PERIOD`], the gauges of the process's latest
//! device (`super::device::gauges`) go to the counts line's highest, and where the target
//! [`TARGET`] is on at debug, one line of its memory gauges goes to the host's log where they
//! changed since the line before, so a boot's reader can tell the memory rt holds from the memory
//! its objects use. A diagnostic: the gauges are rt's own, which no Metal call answers, and which
//! lines the host writes is free under the contract.
//!
//! - **Always looking.** [`start`] makes the thread at the process's first device's open, whatever
//!   the log, and each look gives the gauges to the counts line's highest
//!   (`super::device::note_gauges`), so that the counts line's `rt-<gauge>-most` are the gauges'
//!   highest at a 5 s look, not only at the counts' folds (`super::device`'s docs, "Its memory at
//!   its highest"). A look costs the work the gauges count nothing, once every 5 s: a lock of the
//!   library's own (the latest device's counters, `super::device::gauges`) around rt's read of
//!   its gauges, a relaxed load of each of its counters and a read of its clock, which takes no
//!   lock of rt's and calls no driver; a lock of the gauges' highest; and the degradation counts'
//!   lock for each gauge that rose past its highest.
//! - **Lines asked for by name.** A look writes its lines only where `REIMS_VGPU_LOG` names the
//!   target itself at debug (`info,reims_vgpu::rt_gauges=debug`: `crate::core::logging::names`),
//!   as reims-vgpu's work lines (`reims_vgpu::work`) are, and the log writes them ([`logs`]). A
//!   lone level, such as a bug report's `debug`, writes none, nor does a test's capture of the
//!   log, so no line comes between the lines a test reads.
//! - **One thread a process.** It reads the latest device's gauges at each look, so a device made
//!   after a let-go is read with no thread of its own, and a device torn down reads as zeros, which
//!   leave the highest as it was.
//! - **A line** is `rt gauges: NAME VALUE, ...` ([`line()`]), each name the field of rt's `Gauges`
//!   it reads, bytes and counts as rt keeps them: the Private and the Shared pool's allocations,
//!   their idle allocations and reserve, the blocks taken from them, the live buffers in them and
//!   the long allocations held again, the textures' places in texture blocks, which are Private
//!   blocks, and outside the pools, the textures' allocations of their own, the spills, the
//!   imports, the surface imports, and the preprocess regions, each a buffer on an allocation of
//!   its own, that executions of indirect command buffers hold (each taken at an execution's
//!   record and given back when its command buffer completes) and that no execution holds, kept
//!   for the next ones, and the textures' resolve images; then the live `VkDeviceMemory`
//!   allocations, the imports, surface imports and exports among them, the live buffers and
//!   textures, the resolve images, and the releases that wait for the command buffers committed
//!   before them. A look at which none changed writes no line ([`Lines`]).
//! - **The completed handlers' line.** Each look that writes lines also writes the line of the
//!   completed handlers that held rt's completion thread (`super::handler_times`), where it
//!   changed since the line before, whether or not a device is made.

use std::sync::Once;
use std::time::Duration;

use metal2vulkan_runtime as rt;

/// The log target of the lines, at debug, which [`start`] asks the log for.
pub(crate) const TARGET: &str = "reims_vgpu::rt_gauges";

/// How often the thread looks at the gauges.
#[cfg(not(test))]
pub(crate) const PERIOD: Duration = Duration::from_secs(5);

/// How often the thread looks at the gauges in the crate's tests, so that a test of a look sees
/// one without a wait of 5 s.
#[cfg(test)]
pub(crate) const PERIOD: Duration = Duration::from_millis(50);

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
        preprocess_used_bytes,
        preprocess_idle_bytes,
        resolve_image_bytes,
        blocks,
        imports,
        surface_imports,
        exports,
        buffers,
        textures,
        resolve_images,
        releases_waiting,
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

/// Whether a look writes its lines (the module docs): `REIMS_VGPU_LOG` names [`TARGET`] at debug
/// and the log writes its lines.
fn logs() -> bool {
    crate::core::logging::names(TARGET, log::Level::Debug)
        && log::log_enabled!(target: TARGET, log::Level::Debug)
}

/// What the thread keeps from one look to the next: the lines before.
#[derive(Default)]
struct Looks {
    /// The gauges' line before.
    lines: Lines,
    /// The completed handlers' line before.
    handlers: String,
}

impl Looks {
    /// A look (the module docs): the latest device's gauges, where one was made, to the counts
    /// line's highest, whatever the log; then, where `logs`, the completed handlers' line and the
    /// gauges' line, each where it changed.
    fn look(&mut self, logs: bool) {
        let gauges = super::device::gauges();
        if let Some(gauges) = &gauges {
            super::device::note_gauges(gauges);
        }
        if !logs {
            return;
        }
        let now = super::handler_times::line();
        if now != self.handlers {
            log::debug!(target: TARGET, "{now}");
            self.handlers = now;
        }
        if let Some(line) = gauges.and_then(|gauges| self.lines.next(&gauges)) {
            log::debug!(target: TARGET, "{line}");
        }
    }
}

/// Starts the thread that looks at the gauges (the module docs), once a process, whatever the
/// log: each [`PERIOD`] a look, which writes its lines only where [`logs`].
pub(super) fn start() {
    static STARTED: Once = Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu-gauges".into())
            .spawn(|| {
                let mut looks = Looks::default();
                loop {
                    std::thread::sleep(PERIOD);
                    looks.look(logs());
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

    /// The gauges' thread, made at a device's open, gives the gauges to the counts line's highest
    /// at its looks with its lines off, on the GPU the test holds: a compute state made and kept
    /// moves rt's kernel gauges (`Gauges::kernels`, `Gauges::kernel_entries`), and a look of the
    /// thread's counts each at least at its value with the state kept, with no fold of the counts
    /// (`super::super::device::fold_metrics`) and no let-go before it.
    #[test]
    fn on_vulkan_the_gauges_thread_notes_the_gauges_with_its_lines_off() {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::os::clock::Clock as _;
        use crate::core::platform::resource::Library as _;
        use crate::core::raise::OnFault;
        use crate::rt_platform::RtPlatform;
        use crate::rt_platform::device::{KERNEL_ENTRIES_MOST_CODE, KERNELS_MOST_CODE};

        /// `tests/shaders/compute_tgstatic.metal`'s kernel.
        const TGSTATIC: &[u8] =
            include_bytes!("../../../../tests/shaders/compute_tgstatic.metallib");

        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        assert!(!logs(), "the test's log asks for no gauges' lines");
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU");
        let counters = device.rt().counters();
        let library = device.new_library(TGSTATIC).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let made = library.new_function(OnFault::Return, "tg_static");
        let kernel = made.expect("never a raise").expect("the kernel");
        let made = device.new_compute_pipeline_state_for_function(Some(&kernel));
        let state = made.expect("a compute state").expect("a compute state");
        let kept = counters.gauges();
        assert!(kept.kernels > 0 && kept.kernel_entries > 0, "{kept:?}");
        let noted = || {
            counted(KERNELS_MOST_CODE) >= kept.kernels
                && counted(KERNEL_ENTRIES_MOST_CODE) >= kept.kernel_entries
        };
        let clock = crate::os::OsClock::new();
        let start = clock.now_mono();
        while !noted() {
            let waited = clock.now_mono().saturating_duration_since(start);
            assert!(
                waited < Duration::from_secs(10),
                "a look of the thread's, every {PERIOD:?}, within 10 s"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(state);
    }

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
        gauges.preprocess_used_bytes = 1_179_648;
        gauges.preprocess_idle_bytes = 524_288;
        gauges.resolve_image_bytes = 3_686_400;
        gauges.blocks = 6;
        gauges.imports = 3;
        gauges.buffers = 4;
        gauges.textures = 5;
        gauges.resolve_images = 1;
        gauges.releases_waiting = 2;
        let expected = "rt gauges: private_allocation_bytes 33554432, private_idle_bytes 16777216, \
                        private_reserve_bytes 16777216, private_placed_bytes 8388608, \
                        private_buffer_bytes 1048576, texture_block_bytes 4194304, \
                        private_long_reused_bytes 0, shared_allocation_bytes 16777216, \
                        shared_idle_bytes 0, shared_reserve_bytes 0, shared_placed_bytes 65536, \
                        shared_buffer_bytes 65536, shared_long_reused_bytes 0, \
                        texture_own_bytes 2097152, spilled_bytes 0, managed_spilled_bytes 0, \
                        import_bytes 3145728, surface_import_bytes 0, \
                        preprocess_used_bytes 1179648, preprocess_idle_bytes 524288, \
                        resolve_image_bytes 3686400, blocks 6, imports 3, surface_imports 0, \
                        exports 0, buffers 4, textures 5, resolve_images 1, releases_waiting 2";
        assert_eq!(line(&gauges), expected);

        let mut lines = Lines::default();
        assert_eq!(lines.next(&gauges).as_deref(), Some(expected));
        assert_eq!(lines.next(&gauges), None, "nothing changed");
        gauges.textures = 6;
        let changed = lines.next(&gauges).expect("a gauge changed");
        assert!(
            changed.ends_with("buffers 4, textures 6, resolve_images 1, releases_waiting 2"),
            "{changed}"
        );
        let zeros = line(&rt::Gauges::default());
        assert_eq!(
            lines.next(&rt::Gauges::default()),
            Some(zeros),
            "a torn-down device's"
        );
    }
}
