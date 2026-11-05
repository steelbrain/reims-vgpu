//! Functions (object type 6, command 15): the record names where the function's IR sits in the
//! task's memory; Metal makes a library of it, and the library's one function is the object.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use super::FUNCTION_OBJECT_TYPE;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::Cursor;
use crate::num::{ToU64, Truncate};
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Library;
use crate::raise::RaiseResult;
use crate::raise::{OnFault, Raise};
use crate::resources::manager::DeathKind;
use crate::resources::manager::cache::nil_object;
use crate::resources::manager::icb::ManagerOps;
use crate::resources::manager::records::OperationHeader;
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

/// The function Metal makes of `data` (the IR, handed over unmodified). What Metal hands back
/// `+0` meanwhile (the error, the names) goes in the unit's real pool. `Ok(None)` is Metal's nil
/// function, unchecked here.
fn new_function_with_ir<P: Platform, M: ManagerOps<P>>(
    m: &M,
    data: &[u8],
) -> RaiseResult<Result<Option<P::Function>, String>> {
    let made = m.device().map_or(Ok(Ok(None)), |d| d.new_library(data))?;
    let library = match made {
        Ok(Some(library)) => library,
        Ok(None) => {
            log::error!("Metal made no library from the function's IR, and gave no error");
            crate::fatal!(DeathKind::NilLibraryWithoutError)
        }
        Err(error) => {
            log::warn!("Metal refused the function's IR: {error}");
            return Ok(Err(format!(
                "Metal made no library of the function's IR: {}",
                error.description
            )));
        }
    };
    // The names are asked for at each use
    if library.function_names().len() != 1 {
        let count = library.function_names().len().to_u64().low_u32();
        return Ok(Err(format!(
            "the function's library holds {count} functions, not 1"
        )));
    }
    let names = library.function_names();
    let Some(name) = names.first() else {
        // The list emptied between the two asks: indexing it raises
        return Err(Raise::foreign("objectAtIndexedSubscript:"));
    };
    Ok(Ok(library.new_function(OnFault::Return, name)?))
}

/// A function from `entry` and its record.
pub(crate) fn create<P: Platform, M: ManagerOps<P>>(
    hold: Hold<'_>,
    m: &M,
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
    match refuse!(new_function_with_ir(m, &data)?) {
        Some(f) => Ok(Ok(f)),
        // A nil function reaches the registration's assert
        None => nil_object(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::types::NsErrorText;
    use crate::resources::manager::icb::testing::*;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{FakePlatform, Fk, Step, kind};

    fn record(offset: u64, length: u32, command_type: u32) -> Vec<u8> {
        let mut v = bytes(&ObjectFunction { offset, length });
        v.extend(header(command_type));
        v
    }

    fn make(o: &FakeOps, ty: u32, data: &[u8]) -> Result<Fk<kind::Function>, String> {
        create::<FakePlatform, _>(Hold::NONE, o, &entry(ty), cursor(data)).unwrap()
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
                "Metal made no library of the function's IR: bad IR"
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
                    format!("the function's library holds {n} functions, not 1")
                );
                assert_eq!(o.log().count("newFunctionWithName:"), 0);
            }
        }
    }
}
