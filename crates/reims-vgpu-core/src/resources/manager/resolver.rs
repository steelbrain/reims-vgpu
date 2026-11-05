//! The function resolver Metal calls, synchronously and on the caller's thread, while it
//! deserializes a pipeline descriptor: each function reference is looked up re-entrantly through
//! the manager (the creation lock may already be held by the same thread).
//!
//! The resolver's context object is the metal crate's, one per pipeline-descriptor send,
//! holding [`function_with_function_ref`] as its resolver.
//!
//! A failed lookup is nil to Metal, as Apple's framework answers it, except two, which raise the
//! function lookup's failure: Metal aborts on a nil vertex or compute function, and the raise
//! faults the exec instead, as every other failed lookup of the decode does.
//! - A function whose entry is empty (type 0)
//!   (`interplay_lifetime_held_pipeline_function_retyped`). A guest empties a task's entries when
//!   the process that owns it ends, without waiting for the execs that process queued, so an
//!   exec the host runs late makes its pipelines from empty entries (x86-64 macOS 26 guests'
//!   icon services, docs/qemu.md).
//! - A function whose IR sits over a hole in the task's page table, IR the guest has not
//!   mapped, which its creator raises (`create/function.rs`, `mutgap_svc_function_over_hole`;
//!   docs/qemu.md, "x86-64 macOS guests: WebGPU render bundles").

use crate::num::Truncate;
use crate::platform::Platform;
use crate::raise::{Raise, RaiseResult};
use crate::resources::manager::states::Functions;
use crate::resources::manager::{ObjectKind, RaiseKind, ResourceManager};
use crate::rt::Hold;

/// The function `function_ref` names: its low 32 bits are the reference. A lookup error is
/// logged and nil to Metal, unless the reference's entry is empty, which raises the function
/// lookup's failure; a raise from the lookup (a task cursor, or a creator raising, as one does
/// for IR over a hole) is `Err` too.
/// The platform carries an `Err` through Metal's frames to the caller's handler.
pub fn function_with_function_ref<P: Platform>(
    hold: Hold<'_>,
    manager: &ResourceManager<P>,
    function_ref: u64,
) -> RaiseResult<Option<P::Function>> {
    let reference = function_ref.low_u32();
    Ok(match manager.get_object::<Functions>(hold, reference)? {
        // Nil for reference 0, with no log
        Ok(function) => function,
        Err(error) => {
            let emptied = manager
                .get_object_entry(reference)
                .is_ok_and(|entry| entry.object_type() == 0);
            if emptied {
                log::warn!("pipeline function {reference} not found: {error}");
                let kind = ObjectKind::Function;
                return Err(Raise::new(RaiseKind::Lookup(kind)).with_detail(kind.failure()));
            }
            log::warn!("pipeline function {reference} not found, given to Metal as nil: {error}");
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::super::fixture::*;
    use super::*;
    use crate::platform::Handle;
    use crate::testkit::calllog::Arg;
    use crate::testkit::{Step, kind};

    #[test]
    fn reference_zero_and_its_truncations_are_nil_with_no_lookup() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        let mark = fx.clear_log();
        assert!(
            function_with_function_ref(Hold::NONE, &fx.m, 0)
                .unwrap()
                .is_none()
        );
        assert!(
            function_with_function_ref(Hold::NONE, &fx.m, 0x1_0000_0000)
                .unwrap()
                .is_none()
        );
        assert!(fx.sels_since(mark).is_empty());
    }

    #[test]
    fn the_reference_is_truncated_to_u32() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        let f = fx
            .platform
            .world()
            .new_obj::<kind::Function>(Default::default());
        assert!(fx.m.register_object::<Functions>(1, f.clone_ref()));
        let got = function_with_function_ref(Hold::NONE, &fx.m, 0xffff_0000_0000_0001)
            .unwrap()
            .unwrap();
        assert_eq!(got.id(), f.id());
    }

    #[test]
    fn a_lookup_error_is_nil() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        assert!(
            function_with_function_ref(Hold::NONE, &fx.m, 5)
                .unwrap()
                .is_none()
        );
        // Reference 1 needs a cursor: there is none, so its creation is refused.
        assert!(
            function_with_function_ref(Hold::NONE, &fx.m, 1)
                .unwrap()
                .is_none()
        );
        assert_eq!(
            fx.log().calls_to("cursor_from_virtual_offset")[0].args,
            vec![Arg::U64(REC), Arg::U64(0x10)]
        );
    }

    /// An entry the guest emptied (type 0 and offset all ones, as a task's end leaves each of
    /// its entries) raises the function lookup's failure, so the exec faults where Metal would
    /// abort on the nil.
    #[test]
    fn an_empty_entry_raises() {
        use crate::raise::RaiseId;
        use crate::resources::manager::{ObjectKind, RaiseKind};
        let fx = Fx::new(
            &[(0, 0, 0), (6, 0x10, REC), (0, 0, u64::MAX)],
            Placement::Off,
        );
        let (r, lines) = fx.logged(|| function_with_function_ref(Hold::NONE, &fx.m, 2));
        let r = r.err().unwrap();
        assert_eq!(
            r.id(),
            RaiseId::Manager(RaiseKind::Lookup(ObjectKind::Function))
        );
        assert_eq!(r.detail(), Some("function lookup failed"));
        assert_eq!(
            lines,
            vec!["pipeline function 2 not found: object 2 has an empty entry"]
        );
    }

    #[test]
    fn a_lookup_error_is_logged_with_the_truncated_reference() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        let (r, lines) = fx.logged(|| function_with_function_ref(Hold::NONE, &fx.m, 0x2_0000_0005));
        assert!(r.unwrap().is_none());
        assert_eq!(
            lines,
            vec![
                "pipeline function 5 not found, given to Metal as nil: reference 5 is past the \
                 object list's 2 entries"
            ]
        );
    }

    /// The resolver's re-entrant lookup: called while the same thread holds the creation lock
    /// (a creator's nested send), it creates without waiting.
    #[test]
    fn a_lookup_under_the_held_creation_lock_reenters() {
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        let _creation = fx.m.creation.lock();
        fx.next_cursor(0, 0x10);
        // The record is not a function's, so the kind's creator refuses it: the lookup ran.
        let (r, lines) = fx.logged(|| function_with_function_ref(Hold::NONE, &fx.m, 1));
        assert!(r.unwrap().is_none());
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn a_raise_goes_to_metal() {
        // The raise leaves the resolver; no log.
        let fx = Fx::new(&[(0, 0, 0), (6, 0x10, REC)], Placement::Off);
        fx.platform.script().push(
            "cursor_from_virtual_offset",
            Step::raise("NSGenericException", "x"),
        );
        let hold = Hold::NONE;
        let (r, lines) = fx.logged(|| function_with_function_ref(hold, &fx.m, 1));
        assert!(r.is_err());
        assert!(lines.is_empty());
    }
}
