//! Every environment variable this device reads, and the one way they parse.
//!
//! # Why they all live here
//!
//! An override is a rule the operator states from outside the process, so it has
//! the same problem the ABI header has: nothing in the toolchain finds the second
//! copy. A variable read at its point of use is invisible to everyone who does not
//! already know it exists, two sites spelling one variable's "off" differently is
//! a divergence no test can see, and a name that gets renamed in one place keeps
//! working in the other. Naming them here makes the set greppable and makes the
//! parse shared.
//!
//! # What an override may do
//!
//! **An override may only narrow what this device does. It may never widen it.**
//!
//! A switch can turn a rail *off* that the host was capable of running, because
//! that is a statement about policy and is always satisfiable. A switch may not
//! turn a rail *on* that the host reported it cannot run: capability is measured
//! from the device, and a variable that could override the measurement would turn
//! "this host has no such extension" into a crash or, worse, undefined behavior
//! inside a driver. Every gate stays where it is; a switch can only add a reason
//! to refuse.
//!
//! That rule is why [`Switch::On`] exists but is nowhere sufficient on its own.
//! Reading it is how a caller notices an operator asked for something the host
//! cannot give and says so, rather than ignoring the request in silence.

/// Guest RAM reaches the GPU as a host-pointer import over whole RAMBlocks.
/// Setting this off makes the device take the copying rails on a host that
/// could have imported — see
/// [`crate::backend::vulkan::caps::host_pointer`].
///
/// This is the switch that matters for verification. Where the import works
/// every guest window takes it and the copying rails run zero times, so a green
/// boot says nothing about them — and they are the only rails on a host without
/// the extension, and the rails a discrete GPU takes regardless.
pub const GUEST_IMPORT: &str = "REIMS_VGPU_GUEST_IMPORT";

/// Verbose per-draw logging on top of the always-on fail sink.
pub const DRAW_LOG: &str = "REIMS_VGPU_DRAW_LOG";

/// Setting this off makes a completion stamp that follows a guest-page writeback
/// block the drain worker on that writeback and then write the stamp word
/// itself, instead of recording the word into the same GPU queue behind the
/// copy and letting the completion thread raise the interrupt.
///
/// A narrowing, like every switch here: the GPU-ordered stamp needs a
/// host-pointer import to reach the stamp page and `timelineSemaphore` to be
/// waited off-thread, so `off` selects the rail a host lacking either takes
/// regardless. It exists because the two rails answer "when may the guest
/// observe this stamp" with different mechanisms — a CPU wait versus a pipeline
/// barrier plus a thread — and a hang or a torn frame has to be attributable to
/// one of them without rebuilding.
pub const GPU_STAMP: &str = "REIMS_VGPU_GPU_STAMP";

/// What one variable says, including the two ways it says nothing usable.
///
/// Four states rather than a `bool` because "unset", "explicitly on" and
/// "spelled wrong" are three different operator intents and a `bool` collapses
/// them into the default. The last one matters most: a typo that silently reads
/// as the default is how an operator concludes a switch does not work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Switch {
    /// Not in the environment, or exported empty — which is how a shell says
    /// "not set" when a variable is assigned from an unset variable.
    Unset,
    /// An affirmative spelling. Never sufficient by itself; see the module doc.
    On,
    /// A negative spelling. This is the state that may change behavior.
    Off,
    /// Present, non-empty, and not one of the spellings below. Carries nothing:
    /// the value is handed back by [`read`] for the caller to name in its own
    /// refusal, because only the caller knows which variable this was.
    Unrecognized,
}

/// The spellings accepted for each state, ASCII-case-insensitively.
///
/// The conventional shell set rather than a chosen one, so an operator does not
/// have to look up which of `0`/`false`/`no` this particular program wanted. The
/// two lists are disjoint and every entry is lowercase, which
/// `the_spellings_are_disjoint_and_lowercase` pins.
const ON_SPELLINGS: [&str; 4] = ["1", "on", "true", "yes"];
const OFF_SPELLINGS: [&str; 4] = ["0", "off", "false", "no"];

/// Classify `name`'s value, and hand back the raw value for a caller that needs
/// to quote it.
///
/// Pure: it reads the environment and parses, and emits nothing. Deliberately —
/// [`crate::observe`] itself reads a variable through here, so an emit on this
/// path would recurse through the sink that is asking whether it is enabled.
/// The caller emits, and it is better placed to: it knows which rail the answer
/// gates and what the consequence of refusing is.
pub fn read(name: &str) -> (Switch, Option<String>) {
    let Some(raw) = std::env::var_os(name) else {
        return (Switch::Unset, None);
    };
    let value = raw.to_string_lossy().into_owned();
    let folded = value.trim().to_ascii_lowercase();
    if folded.is_empty() {
        return (Switch::Unset, None);
    }
    let state = if ON_SPELLINGS.contains(&folded.as_str()) {
        Switch::On
    } else if OFF_SPELLINGS.contains(&folded.as_str()) {
        Switch::Off
    } else {
        Switch::Unrecognized
    };
    (state, Some(value))
}

/// [`read`] for a caller that has nothing to say about the value.
pub fn switch(name: &str) -> Switch {
    read(name).0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One process-wide lock for every test that mutates the environment.
    /// `set_var` is process-global and unsynchronized; two tests setting
    /// different variables concurrently is fine, but two setting the *same* one
    /// is not, and these all touch the same probe name.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Set `PROBE` to `value` (or unset it), run `body`, and restore.
    fn with_probe<R>(value: Option<&str>, body: impl FnOnce() -> R) -> R {
        const PROBE: &str = "REIMS_VGPU_TEST_PROBE";
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock above serializes every mutation of this variable in
        // this process, and nothing outside these tests reads it.
        unsafe {
            match value {
                Some(v) => std::env::set_var(PROBE, v),
                None => std::env::remove_var(PROBE),
            }
        }
        let out = body();
        unsafe { std::env::remove_var(PROBE) };
        out
    }

    fn probe(value: Option<&str>) -> Switch {
        with_probe(value, || switch("REIMS_VGPU_TEST_PROBE"))
    }

    /// Both directions, in every spelling the module claims to accept. A
    /// spelling that silently reads as `Unrecognized` is a switch an operator
    /// sets and watches do nothing.
    #[test]
    fn every_documented_spelling_parses() {
        for on in ON_SPELLINGS {
            assert_eq!(probe(Some(on)), Switch::On, "{on}");
            assert_eq!(probe(Some(&on.to_ascii_uppercase())), Switch::On, "{on}");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(probe(Some(off)), Switch::Off, "{off}");
            assert_eq!(probe(Some(&off.to_ascii_uppercase())), Switch::Off, "{off}");
        }
    }

    /// An unset variable and one exported empty are the same answer. `FOO=$BAR`
    /// with `BAR` unset produces the second, and reading it as a value would
    /// make an unrelated typo elsewhere in a boot script silently flip a rail.
    #[test]
    fn unset_and_empty_are_the_same_answer() {
        assert_eq!(probe(None), Switch::Unset);
        assert_eq!(probe(Some("")), Switch::Unset);
        assert_eq!(probe(Some("   ")), Switch::Unset);
    }

    /// A typo is its own answer and keeps its value, so the caller's refusal can
    /// quote what was actually written. Collapsing this into `Unset` is how a
    /// misspelled switch reads as working.
    #[test]
    fn a_value_that_is_neither_keeps_itself_for_the_message() {
        let (state, value) = with_probe(Some("mabye"), || read("REIMS_VGPU_TEST_PROBE"));
        assert_eq!(state, Switch::Unrecognized);
        assert_eq!(value.as_deref(), Some("mabye"));
    }

    /// Surrounding whitespace is not a value. A trailing space picked up from a
    /// heredoc or a `docker run -e` line would otherwise read as a typo.
    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(probe(Some(" off ")), Switch::Off);
        assert_eq!(probe(Some("\t1\n")), Switch::On);
    }

    /// The two lists cannot overlap and are compared lowercased, so an entry
    /// with a capital in it would never match anything.
    #[test]
    fn the_spellings_are_disjoint_and_lowercase() {
        for on in ON_SPELLINGS {
            assert!(!OFF_SPELLINGS.contains(&on), "{on} is in both lists");
            assert_eq!(on, on.to_ascii_lowercase(), "{on} would never match");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(off, off.to_ascii_lowercase(), "{off} would never match");
        }
    }

    /// Every variable the crate honors is named here, spelled consistently. A
    /// name that does not carry the crate prefix is one an operator cannot find
    /// by grepping their own environment.
    #[test]
    fn every_name_carries_the_crate_prefix() {
        for name in [DRAW_LOG, GUEST_IMPORT, GPU_STAMP] {
            assert!(name.starts_with("REIMS_VGPU_"), "{name}");
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
                "{name}"
            );
        }
    }
}
