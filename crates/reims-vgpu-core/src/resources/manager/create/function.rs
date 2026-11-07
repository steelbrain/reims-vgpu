//! Functions (object type 6, command 15): the record names where the function's IR sits in the
//! task's memory; Metal makes a library of it, and the library's one function is the object.
//!
//! IR the task cannot copy refuses the function, except IR over a hole in the task's page table,
//! which raises the function lookup's failure: a guest commits a buffer's pages to the task's
//! page table as it prepares a command buffer that lists the buffer, so a hole is IR the guest
//! has not mapped. A nil function would reach Metal, which ends the process at a nil vertex or
//! compute function; the raise faults the exec instead, as an empty entry's lookup does
//! (`resolver`), and caches nothing, so a lookup once the guest has mapped the IR makes the
//! function (docs/qemu.md, "x86-64 macOS guests: WebGPU render bundles").
//!
//! A library or function the platform refuses for a gap of its own or of its device's, where
//! Metal makes it, raises the same failure (`platform::device::relaying_gaps`): the Vulkan
//! platform answers such a library its runtime's refusal as the `NSError`, and such a function
//! nil, and either would reach Metal's abort at the state that names the function, one create
//! later. The state is never made, and the exec or info that names it faults, as where the
//! platform refuses the state itself for a gap. Nothing is cached, so each lookup asks the
//! platform again. Metal's own refusal of the IR still refuses the function, and Metal's nil
//! function still reaches the registration's assert.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::FUNCTION_OBJECT_TYPE;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::num::{ToU64, Truncate};
use crate::platform::Platform;
use crate::platform::device::{GpuDevice, relaying_gaps};
use crate::platform::resource::Library;
use crate::raise::RaiseResult;
use crate::raise::{OnFault, Raise};
use crate::resources::manager::cache::nil_object;
use crate::resources::manager::icb::ManagerOps;
use crate::resources::manager::records::OperationHeader;
use crate::resources::manager::{DeathKind, ObjectKind, RaiseKind};
use crate::rt::Hold;

/// The command type that makes a function from IR.
const NEW_FUNCTION_WITH_IR: u32 = 15;

/// A function's record (packed, 12 bytes): where the function's IR sits in the task's memory.
#[derive(Debug, Clone, Copy, Default, FromBytes, IntoBytes, Immutable, KnownLayout)]
#[repr(C, packed)]
pub(crate) struct ObjectFunction {
    /// The IR's task virtual offset.
    pub offset: u64,
    /// The IR's length.
    pub length: u32,
}

const _: () = assert!(size_of::<ObjectFunction>() == 12);

/// Why no function was made of the IR.
#[derive(Debug, PartialEq, Eq)]
enum NotMade {
    /// Metal's refusal: the function's creation refuses with this text.
    Refused(String),
    /// The platform's refusal for a gap of its own or of its device's, where Metal makes the
    /// library or function ([`relaying_gaps`]): this text says which.
    Gap(String),
}

/// The function Metal makes of `data` (the IR, handed over unmodified). What Metal hands back
/// `+0` meanwhile (the error, the names) goes in the unit's real pool. `Ok(None)` is Metal's nil
/// function, unchecked here.
fn new_function_with_ir<P: Platform, M: ManagerOps<P>>(
    m: &M,
    data: &[u8],
) -> RaiseResult<Result<Option<P::Function>, NotMade>> {
    let (made, gap) = relaying_gaps(|| m.device().map_or(Ok(Ok(None)), |d| d.new_library(data)));
    let library = match made? {
        Ok(Some(library)) => library,
        Ok(None) => {
            log::error!("Metal made no library from the function's IR, and gave no error");
            crate::fatal!(DeathKind::NilLibraryWithoutError)
        }
        Err(error) if gap => {
            return Ok(Err(NotMade::Gap(format!(
                "the platform refused its library for a gap: {}",
                error.description
            ))));
        }
        Err(error) => {
            log::warn!("Metal refused the function's IR: {error}");
            return Ok(Err(NotMade::Refused(format!(
                "Metal made no library of the function's IR: {}",
                error.description
            ))));
        }
    };
    // The names are asked for at each use
    if library.function_names().len() != 1 {
        let count = library.function_names().len().to_u64().low_u32();
        return Ok(Err(NotMade::Refused(format!(
            "the function's library holds {count} functions, not 1"
        ))));
    }
    let names = library.function_names();
    let Some(name) = names.first() else {
        // The list emptied between the two asks: indexing it raises
        return Err(Raise::foreign("objectAtIndexedSubscript:"));
    };
    let (made, gap) = relaying_gaps(|| library.new_function(OnFault::Return, name));
    match made? {
        None if gap => Ok(Err(NotMade::Gap(format!(
            "the platform answered its function {name:?} nil for a gap"
        )))),
        made => Ok(Ok(made)),
    }
}

/// The function `reference` names, from `entry` and its record. IR over a hole in the task's
/// page table raises the function lookup's failure, and so does a library or function the
/// platform refuses for a gap; IR the task cannot otherwise copy refuses it.
pub(crate) fn create<P: Platform, M: ManagerOps<P>>(
    hold: Hold<'_>,
    m: &M,
    reference: u32,
    entry: &ObjectEntry,
    mut cursor: Cursor,
) -> RaiseResult<Result<P::Function, String>> {
    let object_type = entry.object_type();
    if object_type != FUNCTION_OBJECT_TYPE {
        return Ok(Err(format!(
            "object of type {object_type} is not a function"
        )));
    }
    let function = refuse!(
        cursor
            .read::<ObjectFunction>()
            .map_err(|e| format!("function record not readable: {e}"))
    );
    let header = refuse!(
        cursor
            .read::<OperationHeader>()
            .map_err(|e| format!("command header not readable: {e}"))
    );
    // The IR is copied before the command type is checked
    let (offset, length) = (function.offset, function.length);
    let data = match m.get_data(hold, offset, u64::from(length)) {
        Ok(data) => data,
        Err(missing) if missing.hole => {
            log::warn!(
                "function {reference} not made: its IR, {length:#x} bytes at task offset \
                 {offset:#x}, is not mapped: {missing}"
            );
            let kind = ObjectKind::Function;
            return Err(Raise::new(RaiseKind::Lookup(kind)).with_detail(kind.failure()));
        }
        Err(missing) => {
            return Ok(Err(format!(
                "the function's IR, {length:#x} bytes at task offset {offset:#x}, is not in \
                 task memory: {missing}"
            )));
        }
    };
    if header.command_type != NEW_FUNCTION_WITH_IR {
        let ct = header.command_type;
        return Ok(Err(format!("unknown command type {ct}")));
    }
    match new_function_with_ir(m, &data)? {
        Ok(Some(f)) => Ok(Ok(f)),
        // A nil function reaches the registration's assert
        Ok(None) => nil_object(),
        Err(NotMade::Refused(why)) => Ok(Err(why)),
        Err(NotMade::Gap(why)) => {
            log::warn!("function {reference} not made: {why}");
            let kind = ObjectKind::Function;
            Err(Raise::new(RaiseKind::Lookup(kind)).with_detail(kind.failure()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::types::NsErrorText;
    use crate::raise::RaiseId;
    use crate::resources::manager::icb::testing::*;
    use crate::resources::manager::lookup::NoData;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    fn record(offset: u64, length: u32, command_type: u32) -> Vec<u8> {
        let mut v = bytes(&ObjectFunction { offset, length });
        v.extend(header(command_type));
        v
    }

    fn make(o: &FakeOps, ty: u32, data: &[u8]) -> Result<Fk<kind::Function>, String> {
        create::<FakePlatform, _>(Hold::NONE, o, 3, &entry(ty), cursor(data)).unwrap()
    }

    fn error(description: &str) -> Step {
        Step::Error(NsErrorText {
            domain: "MTLLibraryErrorDomain".to_owned(),
            code: 1,
            description: description.to_owned(),
        })
    }

    fn last(o: &FakeOps, sel: &str) -> Vec<Arg> {
        o.log().calls_to(sel).last().unwrap().args.clone()
    }

    mod creation {
        use super::*;

        #[test]
        fn the_object_type_must_be_6() {
            let o = FakeOps::new();
            assert_eq!(
                make(&o, 7, &record(0, 4, 15)).unwrap_err(),
                "object of type 7 is not a function"
            );
        }

        #[test]
        fn short_headers_fail() {
            let o = FakeOps::new();
            let r = record(0, 4, 15);
            let e = make(&o, 6, &r[..11]).unwrap_err();
            assert!(e.starts_with("function record not readable: "), "{e}");
            let e = make(&o, 6, &r[..15]).unwrap_err();
            assert!(e.starts_with("command header not readable: "), "{e}");
            assert!(o.calls().is_empty());
        }

        #[test]
        fn the_data_is_copied_before_the_command_type_is_checked() {
            let mut o = FakeOps::new();
            let e = make(&o, 6, &record(0x40, 4, 3)).unwrap_err();
            assert_eq!(
                e,
                "the function's IR, 0x4 bytes at task offset 0x40, is not in task memory: no data"
            );
            assert_eq!(o.calls(), vec!["get_data(0x40, 0x4)"]);
            o.data = Some(vec![1, 2, 3, 4]);
            let e = make(&o, 6, &record(0x40, 4, 3)).unwrap_err();
            assert_eq!(e, "unknown command type 3");
            assert!(o.log().is_empty());
        }

        #[test]
        fn ir_over_a_hole_raises_the_function_lookup_s_failure() {
            let mut o = FakeOps::new();
            o.no_data = NoData {
                hole: true,
                why: "a hole in the page table".to_owned(),
            };
            let (r, lines) = logged(|| {
                create::<FakePlatform, _>(
                    Hold::NONE,
                    &o,
                    3,
                    &entry(6),
                    cursor(&record(0x40, 4, 15)),
                )
            });
            let raised = r.err().unwrap();
            assert_eq!(
                raised.id(),
                RaiseId::Manager(RaiseKind::Lookup(ObjectKind::Function))
            );
            assert_eq!(raised.detail(), Some("function lookup failed"));
            assert_eq!(
                lines,
                vec![
                    "function 3 not made: its IR, 0x4 bytes at task offset 0x40, is not mapped: a \
                     hole in the page table"
                ]
            );
            assert_eq!(o.calls(), vec!["get_data(0x40, 0x4)"]);
            assert!(o.log().is_empty());
        }

        #[test]
        fn one_function_is_made_from_the_ir() {
            let mut o = FakeOps::new();
            o.data = Some(vec![9, 8, 7]);
            o.p.world().set_library_functions(vec!["main0".to_owned()]);
            make(&o, 6, &record(0x40, 3, 15)).unwrap();
            assert_eq!(
                o.log().sels(),
                vec!["newLibraryWithData:error:", "newFunctionWithName:"]
            );
            assert_eq!(
                last(&o, "newLibraryWithData:error:"),
                vec![Arg::Bytes(vec![9, 8, 7])]
            );
            assert_eq!(
                last(&o, "newFunctionWithName:"),
                vec![Arg::Str("main0".to_owned())]
            );
        }

        #[test]
        fn a_nil_function_reaches_the_registrations_assert() {
            let mut o = FakeOps::new();
            o.data = Some(vec![0]);
            o.p.world().set_library_functions(vec!["f".to_owned()]);
            o.p.script().push("newFunctionWithName:", Step::Nil);
            let d = catch_death(|| make(&o, 6, &record(0, 1, 15)).is_ok()).unwrap_err();
            assert_eq!(d, DeathKind::NilObject.site());
        }

        /// A library the platform refuses for a gap of its own (its runtime's refusal as the
        /// `NSError`), or the library's function it answers nil for one, raises the function
        /// lookup's failure, where Metal's refusal refuses the function and Metal's nil reaches
        /// the registration's assert
        #[test]
        fn a_library_or_function_refused_for_a_gap_raises_the_function_lookup_s_failure() {
            let gap = Step::Gap(NsErrorText {
                domain: "runtime".to_owned(),
                code: 0,
                description: "not implemented: inline assembly".to_owned(),
            });
            for (refused, line, sels) in [
                (
                    "newLibraryWithData:error:",
                    "function 3 not made: the platform refused its library for a gap: not \
                     implemented: inline assembly",
                    vec!["newLibraryWithData:error:"],
                ),
                (
                    "newFunctionWithName:",
                    "function 3 not made: the platform answered its function \"main0\" nil for a \
                     gap",
                    vec!["newLibraryWithData:error:", "newFunctionWithName:"],
                ),
            ] {
                let mut o = FakeOps::new();
                o.data = Some(vec![9, 8, 7]);
                o.p.world().set_library_functions(vec!["main0".to_owned()]);
                o.p.script().push(refused, gap.clone());
                let (r, lines) = logged(|| {
                    create::<FakePlatform, _>(
                        Hold::NONE,
                        &o,
                        3,
                        &entry(6),
                        cursor(&record(0x40, 3, 15)),
                    )
                });
                let raised = r.err().unwrap();
                assert_eq!(
                    raised.id(),
                    RaiseId::Manager(RaiseKind::Lookup(ObjectKind::Function))
                );
                assert_eq!(raised.detail(), Some("function lookup failed"));
                assert_eq!(lines, vec![line]);
                assert_eq!(o.log().sels(), sels);
                // A refusal noted inside one creation is not the next one's
                let made = relaying_gaps(|| make(&o, 6, &record(0x40, 3, 15)));
                assert!(made.0.is_ok() && !made.1);
            }
        }
    }

    mod library {
        use super::*;

        #[test]
        fn a_library_error_is_logged_and_returned() {
            let o = FakeOps::new();
            o.p.script()
                .push("newLibraryWithData:error:", error("bad IR"));
            let (e, lines) = logged(|| new_function_with_ir::<FakePlatform, _>(&o, &[1]));
            assert_eq!(
                e.unwrap().unwrap_err(),
                NotMade::Refused("Metal made no library of the function's IR: bad IR".to_owned())
            );
            assert_eq!(lines, vec!["Metal refused the function's IR: bad IR"]);
        }

        /// The library goes when the creation returns.
        #[test]
        fn the_library_is_released_when_the_function_returns() {
            let o = FakeOps::new();
            o.p.world().set_library_functions(vec!["main0".to_owned()]);
            let f = new_function_with_ir::<FakePlatform, _>(&o, &[1])
                .unwrap()
                .unwrap();
            assert!(f.is_some());
            assert!(o.p.world().live("MTLLibrary").is_empty());
        }

        #[test]
        fn a_nil_library_without_an_error_logs_and_dies() {
            let o = FakeOps::new();
            o.p.script().push("newLibraryWithData:error:", Step::Nil);
            let (d, lines) = logged(|| {
                catch_death(|| new_function_with_ir::<FakePlatform, _>(&o, &[1]).is_ok())
            });
            assert_eq!(d.unwrap_err(), DeathKind::NilLibraryWithoutError.site());
            assert_eq!(
                lines,
                vec!["Metal made no library from the function's IR, and gave no error"]
            );
        }

        #[test]
        fn a_count_other_than_one_fails() {
            for names in [vec![], vec!["a".to_owned(), "b".to_owned()]] {
                let n = names.len();
                let o = FakeOps::new();
                o.p.world().set_library_functions(names);
                let e = new_function_with_ir::<FakePlatform, _>(&o, &[1])
                    .unwrap()
                    .unwrap_err();
                assert_eq!(
                    e,
                    NotMade::Refused(format!("the function's library holds {n} functions, not 1"))
                );
                assert_eq!(o.log().count("newFunctionWithName:"), 0);
            }
        }
    }
}
