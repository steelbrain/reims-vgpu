//! Driver calls a process did not return from, remembered across boots.
//!
//! The parent module's doc says why this is evidence rather than a heuristic and
//! why it has no off switch. This one is about the mechanism.
//!
//! # "Did not return" includes being killed
//!
//! The breadcrumb records that a process **ended while inside** the call, and it
//! cannot tell a `SIGSEGV` from a `SIGTERM`. Both were observed: macOS 15's
//! compile segmentation-faults at ~11.5 minutes and macOS 26's was still running
//! when the boot script's own timeout killed it at 25. Both are "this call does
//! not come back", so both quarantine.
//!
//! The cost is the case at the other end of that scale: killing a boot by hand
//! during an ordinary healthy compile would quarantine a healthy call. The arm
//! window is a few hundred microseconds of a boot and the next process says so
//! loudly, so the answer is `rm` on the list rather than a rule trying to
//! distinguish the two — which it cannot, because the information is not there.
//!
//! # The key is the call, not the module
//!
//! A graphics compile consumes a vertex module and a fragment module in one
//! call, and when it takes the process down nothing outside the driver can say
//! which of the two it choked on. Quarantining both would cost every other
//! pipeline that shares the innocent one — on a real guest the vertex stage is
//! shared widely and the fragment stage is not, so quarantining by module would
//! turn one bad shader into a dead rail.
//!
//! So the key is the ordered list of the call's module digests. That is exactly
//! what the evidence supports: *this call, with these inputs, did not return*.
//! A later call reusing one of the modules with a different partner is a
//! different experiment and is allowed to run.
//!
//! # Keying on the content makes the list self-invalidating, which is a reading
//!
//! The key is computed from the modules the caller is holding *now* and matched
//! against the file; it is never stored alongside them. So when the thing that
//! produced those modules changes — a translator bump is the case that arises —
//! the emitted bytes change, the digest changes, and the entry simply stops
//! matching. The call is re-attempted with no list surgery and no rev field.
//!
//! That has a consequence worth stating, because it saves a boot. **A firing is
//! evidence about the present, not about when the line was written.** If a
//! quarantine recorded before a translator bump still fires after it, the module
//! being handed to the driver is byte-identical to the one that killed a
//! process, and every count recorded on that line — word counts included —
//! describes what is being produced today.
//!
//! The converse is where the list goes quiet rather than wrong: a line nothing
//! matches any more is indistinguishable from a defect somebody fixed. Do not
//! read stale entries as a census of live defects; read the ones that *fire*.
//!
//! # The file
//!
//! One text file, one line per quarantined call:
//!
//! ```text
//! <key>\t<what>
//! ```
//!
//! It is appended to, never rewritten, and it is read once per process. A
//! corrupt or unreadable line costs that entry and nothing else — a quarantine
//! this device cannot parse must not stop the boot, because the failure it
//! guards against is rarer than the filesystem being odd.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// A driver call this device will not make again, and the description it had
/// when the process that made it ended inside it.
#[derive(Debug)]
pub(crate) struct Quarantined {
    pub(crate) key: String,
    pub(crate) previously: String,
}

/// The persistent list. In the temp dir beside the breadcrumb it is derived
/// from, so one `rm reims-vgpu-driver-*` clears the whole mechanism.
pub(crate) fn list_path() -> PathBuf {
    std::env::temp_dir().join(format!("{}-quarantine", super::prefix()))
}

/// The digest list identifying one driver call's inputs.
///
/// Built from the same [`super::super::digest::Digest128`] the shader cache keys
/// on, so a module that hashes equal here is the module the cache would have
/// reused — a second digest function would be a second opinion about identity
/// and there is no reason to have one.
pub(crate) fn key_of(modules: &[(&'static str, &[u32])]) -> String {
    let mut key = String::new();
    for (stage, spirv) in modules {
        if !key.is_empty() {
            key.push('+');
        }
        let d = super::super::digest::Digest128::of_u32_words(spirv);
        key.push_str(&format!("{stage}:{:016x}{:016x}{:x}", d.a, d.b, d.len));
    }
    key
}

/// The quarantine as this process sees it: loaded once, folding in whatever the
/// previous process left behind.
fn entries() -> &'static HashMap<String, String> {
    static ENTRIES: OnceLock<HashMap<String, String>> = OnceLock::new();
    ENTRIES.get_or_init(|| {
        fold_surviving_breadcrumb();
        let map = std::fs::read_to_string(list_path())
            .map(|text| parse_list(&text))
            .unwrap_or_default();
        if !map.is_empty() {
            reims_vgpu_observe::fail(format!(
                "driver_quarantine reason=driver_quarantine_loaded calls={} path={} \
                 (each one ended a process that was inside the driver; delete the file to try \
                 them again)",
                map.len(),
                list_path().display()
            ));
        }
        map
    })
}

/// A breadcrumb still on disk means the process that armed it never returned
/// from the call. Fold it into the list and take the files away, so the next
/// process reads a quarantine rather than a fresh ending.
///
/// Called from the one-time load, before the file is read, so a crash and the
/// boot that follows it are separated by nothing the caller has to sequence.
fn fold_surviving_breadcrumb() {
    let Ok(meta) = std::fs::read_to_string(super::meta_path()) else {
        return;
    };
    let (what, key, stages) = parse_meta(&meta);
    // Take the evidence away whatever happens next: a breadcrumb that stays
    // would be folded again by every later process, and a meta file with no
    // `key=` is one this device cannot act on either way.
    for stage in stages {
        let _ = std::fs::remove_file(super::path(stage));
    }
    let _ = std::fs::remove_file(super::meta_path());
    if key.is_empty() {
        reims_vgpu_observe::fail(format!(
            "driver_quarantine reason=driver_quarantine_crash_unkeyed what={what} \
             (a breadcrumb survived, so the last process ended inside this call, but its meta \
             carried no key= line and the call cannot be recognised again)"
        ));
        return;
    }
    reims_vgpu_observe::fail(format!(
        "driver_quarantine reason=driver_quarantine_ended_in_call what={what} key={key} \
         (the last process ended while inside this driver call — a crash, or a kill of a call that \
         was not coming back; it will be refused from now on)"
    ));
    append(key, what);
}

/// Read a surviving breadcrumb's meta file: what the call was, the key that
/// identifies it, and the stage files to take away.
///
/// Pure so the format has a test that does not need a crashed process. Every
/// field is optional in the sense that a truncated write — the process died
/// while this was being written, which is the exact scenario — must still be
/// readable for whatever reached disk.
fn parse_meta(meta: &str) -> (&str, &str, Vec<&str>) {
    let what = meta
        .lines()
        .find_map(|l| l.strip_prefix("what="))
        .unwrap_or("unknown");
    let key = meta
        .lines()
        .find_map(|l| l.strip_prefix("key="))
        .unwrap_or_default();
    let stages = meta
        .lines()
        .filter_map(|l| l.strip_prefix("stage="))
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    (what, key, stages)
}

/// Read the persistent list. A line without the separator is skipped rather than
/// failing the load: the file is appended to by processes that may die mid-write
/// and a torn last line must not cost the entries above it.
fn parse_list(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter_map(|l| l.split_once('\t'))
        .map(|(key, what)| (key.to_string(), what.to_string()))
        .collect()
}

fn append(key: &str, what: &str) {
    use std::io::Write;
    let opened = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(list_path());
    match opened {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{key}\t{what}") {
                reims_vgpu_observe::fail(format!(
                    "driver_quarantine reason=driver_quarantine_write_failed key={key} err={e}"
                ));
            }
        }
        Err(e) => reims_vgpu_observe::fail(format!(
            "driver_quarantine reason=driver_quarantine_open_failed key={key} err={e}"
        )),
    }
}

/// Whether this device has already lost a process to this exact call.
pub(crate) fn check(modules: &[(&'static str, &[u32])]) -> Option<Quarantined> {
    let key = key_of(modules);
    entries().get(&key).map(|previously| Quarantined {
        key,
        previously: previously.clone(),
    })
}

#[cfg(test)]
mod tests {
    /// The key names the stage as well as the bytes.
    ///
    /// A vertex module and a fragment module that happened to be byte-identical
    /// are two different inputs to the driver, and a key that could not tell
    /// them apart would quarantine a call this device never made.
    #[test]
    fn the_key_distinguishes_stage_and_content() {
        let a = [0x0723_0203u32, 1, 2];
        let b = [0x0723_0203u32, 1, 3];
        let vert_a = super::key_of(&[("vert", &a)]);
        let frag_a = super::key_of(&[("frag", &a)]);
        let vert_b = super::key_of(&[("vert", &b)]);
        assert_ne!(vert_a, frag_a, "the stage is part of the call's identity");
        assert_ne!(vert_a, vert_b, "so is the module's content");
        assert_eq!(vert_a, super::key_of(&[("vert", &a)]), "and it is stable");
    }

    /// The meta a live arming writes is the meta a later process reads back.
    ///
    /// The two halves are written and parsed in different modules and nothing
    /// else compares them, so this is where the format is pinned: a `stage=`
    /// line the reader cannot see leaves a stale `.spv` on disk forever, and a
    /// `key=` line it cannot see turns a recorded crash into an unkeyed one that
    /// can never be recognised again.
    #[test]
    fn a_breadcrumbs_meta_round_trips_through_the_reader() {
        let meta = "what=create_graphics_pipelines vert_words=1761 frag_words=261597\n\
                    stage=vert words=1761 bytes=7044\n\
                    stage=frag words=261597 bytes=1046388\n\
                    key=vert:aabb1+frag:ccdd2\n";
        let (what, key, stages) = super::parse_meta(meta);
        assert_eq!(
            what,
            "create_graphics_pipelines vert_words=1761 frag_words=261597"
        );
        assert_eq!(key, "vert:aabb1+frag:ccdd2");
        assert_eq!(stages, vec!["vert", "frag"]);
    }

    /// A meta file the dying process only got halfway through is still read for
    /// what it has. The crash this records happens *inside* the call the meta
    /// describes, so a torn write is the ordinary case, not the exotic one.
    #[test]
    fn a_truncated_meta_still_yields_what_reached_disk() {
        let (what, key, stages) = super::parse_meta("what=create_shader_module\nstage=mod");
        assert_eq!(what, "create_shader_module");
        assert_eq!(stages, vec!["mod"], "the stage file can still be removed");
        assert!(key.is_empty(), "and the caller can see there is no key");
    }

    /// A torn last line costs its own entry and none of the ones above it. The
    /// list is appended to by processes that die by definition.
    #[test]
    fn a_torn_list_keeps_the_entries_above_the_tear() {
        let list = "k1\twhat one\nk2\twhat two\nk3-no-tab";
        let map = super::parse_list(list);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get("k1").map(String::as_str), Some("what one"));
        assert_eq!(map.get("k2").map(String::as_str), Some("what two"));
    }

    /// A two-module call has a key of its own that neither half carries.
    ///
    /// This is the property that keeps a quarantined graphics compile from
    /// costing every other pipeline built on its vertex stage.
    #[test]
    fn a_pair_is_not_either_of_its_halves() {
        let v = [0x0723_0203u32, 7];
        let f = [0x0723_0203u32, 8];
        let pair = super::key_of(&[("vert", &v), ("frag", &f)]);
        assert_ne!(pair, super::key_of(&[("vert", &v)]));
        assert_ne!(pair, super::key_of(&[("frag", &f)]));
        assert_ne!(
            pair,
            super::key_of(&[("frag", &f), ("vert", &v)]),
            "the order the driver was given them in is part of the call"
        );
    }
}
