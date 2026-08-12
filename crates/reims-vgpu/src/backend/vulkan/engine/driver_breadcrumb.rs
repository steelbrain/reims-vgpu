//! What the device was inside when the process died.
//!
//! # Why a file and not a log line
//!
//! A driver that segmentation-faults inside a call we made ends the VM process
//! with no unwinding, no `Drop`, and no chance to write anything afterwards. The
//! fail log names what ran *before*, which on a shader-compile crash is a
//! translation that succeeded — the module the compiler choked on is a value in
//! this process's memory and dies with it. A core dump holds it, and digging a
//! `Vec<u32>` out of a 2 GiB core is not a workflow.
//!
//! So the bytes go to disk *before* the call and are removed *after* it. Three
//! calls compile a module this device assembled — `vkCreateShaderModule`,
//! `vkCreateComputePipelines` and `vkCreateGraphicsPipelines` — and on a healthy
//! boot the files exist for the duration of each and nothing is left behind. On
//! a crash they are still there, and they are exactly the modules that killed
//! the process.
//!
//! # What it costs
//!
//! One write per pipeline *miss*. Pipeline creation is cached by content digest
//! and a driven boot of a macOS guest takes single-digit misses per second at
//! its very busiest and none at all once the caches are warm, so this is not on
//! any hot path. A hit writes nothing, which is why the caller asks the cache
//! first rather than arming unconditionally.
//!
//! # What it is not
//!
//! It is not a dump facility and it is not an operator switch. There is one path
//! per stage the call consumes, they are overwritten by the next arming, and
//! their whole contract is "if these files exist, the process died in a driver
//! call with these inputs". A graphics pipeline compiles two modules in one call
//! and neither can be ruled out from the outside, so both are written and both
//! are removed together — a single file would have to guess which stage killed
//! the process, and a guess in an evidence file is worse than no file.
//!
//! # The other failure this arming carries: a call that does not return
//!
//! Arming also starts [`crate::observe::driver_watch`], because "the driver died
//! in this call" and "the driver has not come back from this call" are the same
//! bracket around the same call. A hang is the worse of the two — the process
//! survives, so nothing is written, and the drain thread holds the device lock
//! for the whole VM while every census in the crate stays silent because each
//! one reports at the end of a tranche that will not end. The watch is what puts
//! a line in the log while it is happening.
//!
//! # The third use: not doing it again
//!
//! A breadcrumb left on disk is a completed experiment — this driver, these
//! exact inputs, one dead process. [`quarantine`] is what makes the next boot
//! act on it: the surviving files are folded into a persistent list of driver
//! calls that killed a process, and a call whose inputs match one is refused by
//! name instead of made.
//!
//! This is evidence, not a heuristic, and the distinction is the whole licence
//! for it. Nothing here inspects a module to guess whether a driver will like
//! it — there is no size threshold, no opcode census, no shape rule. The only
//! input is "the last process that made this exact call did not return from it",
//! which no amount of reading the module can tell you and which
//! `AGENTS.md`'s ban on overfitting is not about.
//!
//! The cost is that the first crash still happens: this cannot predict a driver
//! bug, only remember one. The benefit is that a VM stops dying on every boot
//! for the same reason, and the loss becomes one refused pipeline with a typed
//! decline naming it — which is the difference between a rail nobody can work on
//! and a rail with one known gap.
//!
//! There is deliberately **no environment switch to turn it off.** A switch that
//! let a known process-killing module through would widen what the device does
//! on the strength of an operator's optimism, which `AGENTS.md` forbids for
//! exactly this reason. Recovery is `rm` on the file, which is a decision with
//! the evidence in front of you rather than a flag set months earlier.
//!
//! # The other half: a module the validator refused
//!
//! [`keep_rejected_module`] lives here because it answers the same question with
//! the same mechanism, for the failure one step earlier. The breadcrumb catches
//! a module the *driver* could not survive; that one catches a module this
//! device assembled and `spirv-val` would not accept — which is a defect in the
//! translation rather than in the driver, and is equally unreadable from a log
//! line alone. The difference is that its file is **kept**: the event has
//! already happened when it is written, so there is nothing to disarm.

use std::path::PathBuf;

pub(crate) mod quarantine;

/// The prefix every file here shares.
///
/// Fixed in production, because the whole mechanism turns on the *next* process
/// finding what this one left — a per-process name would make a crash
/// unreadable by the boot that follows it.
///
/// Under test it carries the test binary's pid, and that is load-bearing rather
/// than tidy. These paths are global to the machine, so a `cargo test` run
/// beside a live VM would otherwise delete the boot's breadcrumb out from under
/// it (observed: a test's `disarm` removed a wedged boot's 1 MB fragment module
/// mid-wedge) and could write a quarantine entry a real boot would then obey.
/// The sink's `redirect_logs_for_tests` splits the same way for the same reason.
#[cfg(not(test))]
fn prefix() -> String {
    String::from("reims-vgpu-driver")
}

#[cfg(test)]
fn prefix() -> String {
    format!("reims-vgpu-driver-test-{}", std::process::id())
}

/// One path per stage. Fixed rather than per-pipeline so a crash leaves exactly
/// one set of files to look at, and so a previous boot's leftovers cannot
/// accumulate.
fn path(stage: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{}-breadcrumb-{stage}.spv", prefix()))
}

/// The metadata file beside them, so a reader knows what the modules were for
/// without disassembling anything.
fn meta_path() -> PathBuf {
    std::env::temp_dir().join(format!("{}-breadcrumb.txt", prefix()))
}

/// Live for the duration of a driver call that could take the process down or
/// fail to return from.
///
/// Dropping it removes the files and stops the watch, so a crash is the only way
/// the files survive and a return is the only way the watch stops.
pub(crate) struct DriverBreadcrumb {
    /// The stage tags whose files this guard owns, empty when nothing was
    /// written.
    stages: Vec<&'static str>,
    /// Whether this guard owns [`crate::observe::driver_watch`]'s slot. False
    /// when an outer call already held it — see that module's `enter`.
    watching: bool,
}

impl DriverBreadcrumb {
    /// Clear the quarantine, write every module the call consumes plus a
    /// description of it, and hold them until dropped.
    ///
    /// **This is the only way to enter one of those calls.** The guard the
    /// caller must hold to make the call is the thing that checks whether the
    /// last process survived it, so a new driver entry point cannot be added
    /// without meeting the rule — which is the only enforcement there is, since
    /// nothing scans for call sites.
    ///
    /// A write that fails is reported once and costs that stage its file: losing
    /// the breadcrumb is a lost diagnostic, never a reason to skip the work the
    /// guest asked for. The clock watch is armed regardless, because it needs
    /// nothing from the filesystem.
    pub(crate) fn arm(
        what: &str,
        modules: &[(&'static str, &[u32])],
    ) -> Result<Self, quarantine::Quarantined> {
        if let Some(hit) = quarantine::check(modules) {
            return Err(hit);
        }
        let watching = crate::observe::driver_watch::enter(what.to_string());
        let mut stages = Vec::with_capacity(modules.len());
        let mut meta = format!("what={what}\n");
        for (stage, spirv) in modules {
            let mut bytes = Vec::with_capacity(spirv.len() * 4);
            for word in *spirv {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            match std::fs::write(path(stage), &bytes) {
                Ok(()) => stages.push(*stage),
                Err(e) => crate::observe::fail(format!(
                    "driver_breadcrumb reason=write_failed what={what} stage={stage} err={e}"
                )),
            }
            meta.push_str(&format!(
                "stage={stage} words={} bytes={}\n",
                spirv.len(),
                spirv.len() * 4
            ));
        }
        // Last, and written whether or not a module reached disk: the key is
        // what the next process needs to recognise this call, and it costs
        // nothing that a failed `.spv` write should be allowed to take away.
        meta.push_str(&format!("key={}\n", quarantine::key_of(modules)));
        let _ = std::fs::write(meta_path(), meta);
        Ok(Self { stages, watching })
    }

    /// The call returned, so the input is not the one that kills the process and
    /// the call is not the one holding the device.
    pub(crate) fn disarm(mut self) {
        self.clear();
    }

    fn clear(&mut self) {
        if self.watching {
            self.watching = false;
            crate::observe::driver_watch::leave();
        }
        if self.stages.is_empty() {
            return;
        }
        for stage in std::mem::take(&mut self.stages) {
            let _ = std::fs::remove_file(path(stage));
        }
        let _ = std::fs::remove_file(meta_path());
    }
}

impl Drop for DriverBreadcrumb {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Keep a module the validator refused, beside the line that refused it.
///
/// The same argument as the breadcrumb above and a different failure. A module
/// this device assembled and `spirv-val` rejected is a **device defect**: the
/// guest asked for something translatable and the translation came out invalid.
/// The refusal line carries the validator's one-sentence complaint, and that
/// sentence names an instruction by result id — `%214 = OpCompositeInsert
/// %_struct_52 %101 %56 0 0` — which is unreadable without the module those ids
/// belong to. Reconstructing it means re-running the boot with a probe, which is
/// two rebuilds and a guest login to recover evidence the device was holding.
///
/// Unlike the breadcrumb this file is **kept**, because the event it records has
/// already happened by the time it is written. Named by content digest so
/// several distinct bad modules in one boot do not overwrite each other, and so
/// the name matches nothing else if the same module is refused again — the
/// validator's verdict is cached per digest, so it is written once per distinct
/// module per boot regardless of how many pipelines want it.
///
/// Disassemble with `spirv-dis`. A failed write costs a diagnostic and nothing
/// else; the refusal has already been emitted by the caller.
pub(crate) fn keep_rejected_module(digest: &str, spirv: &[u32]) {
    let mut bytes = Vec::with_capacity(spirv.len() * 4);
    for word in spirv {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let path = std::env::temp_dir().join(format!("reims-vgpu-rejected-{digest}.spv"));
    match std::fs::write(&path, &bytes) {
        Ok(()) => crate::observe::off(format!(
            "spirv_rejected_module_kept path={} words={}",
            path.display(),
            spirv.len()
        )),
        Err(e) => crate::observe::fail(format!(
            "spirv_rejected_module reason=write_failed path={} err={e}",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    /// A graphics compile consumes two modules and both reach disk under their
    /// own stage names, then both go away when the call returns.
    ///
    /// The `arm`/`disarm` pair is the whole contract — a file left behind after
    /// a healthy call would accuse the next reader's boot of a crash it did not
    /// have, and a stage that never reached disk would leave a real crash half
    /// explained.
    #[test]
    fn both_stages_of_a_graphics_compile_reach_disk_and_are_taken_back() {
        let vert = [0x0723_0203u32, 0x0001_0000, 1];
        let frag = [0x0723_0203u32, 0x0001_0000, 2, 3];
        let crumb = super::DriverBreadcrumb::arm("test_graphics", &[("vert", &vert), ("frag", &frag)])
            .expect("a module nothing has crashed on is not quarantined");

        let vert_path = super::path("vert");
        let frag_path = super::path("frag");
        assert_eq!(
            std::fs::read(&vert_path).expect("the vertex module is on disk"),
            vert.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>()
        );
        assert_eq!(
            std::fs::read(&frag_path).expect("the fragment module is on disk"),
            frag.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>()
        );
        let meta = std::fs::read_to_string(super::meta_path()).expect("the meta line is on disk");
        assert!(meta.contains("vert words=3"), "{meta}");
        assert!(meta.contains("frag words=4"), "{meta}");

        crumb.disarm();
        assert!(!vert_path.exists(), "a returned call leaves no vertex file");
        assert!(!frag_path.exists(), "a returned call leaves no fragment file");
    }

    /// Arming a breadcrumb also puts the call under the clock watch, and
    /// disarming takes it back out.
    ///
    /// These are two modules coupled by one line, and the coupling is the whole
    /// reason a hang gets reported at all: the breadcrumb answers "the driver
    /// died here" and the watch answers "the driver has not come back", and only
    /// the arming site knows where the call is.
    #[test]
    fn an_armed_breadcrumb_puts_the_call_under_the_clock_watch() {
        crate::observe::driver_watch::leave();
        let words = [0x0723_0203u32, 0x0001_0000];
        let crumb = super::DriverBreadcrumb::arm("test_watched", &[("module", &words)])
            .expect("a module nothing has crashed on is not quarantined");
        assert_eq!(
            crate::observe::driver_watch::watching().as_deref(),
            Some("test_watched"),
            "the watch names the call this breadcrumb is bracketing"
        );
        crumb.disarm();
        assert_eq!(
            crate::observe::driver_watch::watching(),
            None,
            "a returned call is no longer outstanding"
        );
    }

    /// The module reaches disk, little-endian and whole.
    ///
    /// Worth a test rather than trusting the write: the point of the file is to
    /// be handed to `spirv-dis`, and a word order or a truncation that made it
    /// undisassemblable would show up only on the boot where somebody needed it.
    #[test]
    fn a_rejected_module_reaches_disk_word_for_word() {
        // SPIR-V magic first, so the bytes on disk are a plausible module and a
        // reversed word order is visible rather than merely different.
        let words = [0x0723_0203u32, 0x0001_0000, 42, 0xdead_beef];
        let digest = "test0123456789ab";
        let path = std::env::temp_dir().join(format!("reims-vgpu-rejected-{digest}.spv"));
        let _ = std::fs::remove_file(&path);

        super::keep_rejected_module(digest, &words);

        let got = std::fs::read(&path).expect("the rejected module is kept");
        let _ = std::fs::remove_file(&path);
        let want: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(got, want, "little-endian words, in order, nothing added");
        assert_eq!(&got[..4], &[0x03, 0x02, 0x23, 0x07], "reads as SPIR-V magic");
    }
}
