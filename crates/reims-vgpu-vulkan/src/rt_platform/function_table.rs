//! The rt platform's function handles ([`RtFunctionHandle`]) and visible function tables
//! ([`RtFunctionTable`]): rt's, of a compute pipeline state (`super::pipeline`'s
//! `functionHandleWithFunction:` and `newVisibleFunctionTableWithDescriptor:`), which a compute
//! encoder binds at a buffer index (`super::compute`).
//!
//! - **A handle** is rt's word of a function among those the state's kernel links in `functions`
//!   or `binary_functions`; a function it does not link (a private one, the kernel itself, any
//!   other) has none, Metal's nil.
//! - **A table** is rt's: its words, a header and one word an entry, in a Shared block of rt's
//!   own, which a dispatch reads as they stand when it runs, so an entry set after a dispatch is
//!   encoded and before its command buffer commits is the one it calls. A null entry, or a handle
//!   none of the dispatch's state's linked functions has, calls nothing and yields zero.
//! - **A set** (`setFunctions:withRange:`, which the guest's `setFunction:atIndex:` sends too) is
//!   rt's. A range past the table is rt's refusal (`Refusal::FunctionTableIndex`, Metal's
//!   refusal of an index past the table), counted under its kind (`rt-refused-ours`), and sets
//!   none of the entries.
//! - **Its answers.** `gpuResourceID` is rt's, the device address of the table's words, which the
//!   guest's argument encoder writes where an argument buffer holds the table; rt refuses the
//!   state of a kernel that reads a table through an argument buffer (`rt-refused-translation`).
//!   `allocatedSize` is the bytes of the table's words in whole pages of 4 KiB ([`PAGE`]), as the
//!   guest counts a buffer's, and `resourceOptions` is `Shared` and tracked ([`OPTIONS`]), as
//!   rt's words are.

use metal2vulkan::bind::function_table_entry_offset;
use metal2vulkan_runtime as rt;

use crate::core::platform::Platform;
use crate::core::platform::function_table::VisibleFunctionTable;
use crate::core::platform::types::NsRange;
use crate::core::raise::RaiseResult;

use super::unbuilt::{RtRefusal, refused_as, rt_refused};
use super::{RtObj, RtPlatform};

/// `setFunctions:withRange:`.
const SET_FUNCTIONS: &str = "setFunctions:withRange:";

/// The page a table's `allocatedSize` counts its words in: the guest's page, which its buffers'
/// `allocatedSize` counts in (the bundle's `RgfxBuffer.mm`). On the AMD Radeon Pro 5300M and the
/// Intel UHD Graphics 630, a table of 0, 1 or 3 entries answers 4096 (measured: macOS 26.7.1,
/// `oracle_fnptr`, docs/graphics-kext/dgpu-oracle.md Q29).
pub const PAGE: u64 = 4096;

/// A table's `resourceOptions`: `MTLResourceStorageModeShared` (0) and
/// `MTLResourceHazardTrackingModeTracked` (512), as rt's words are Shared and a command buffer
/// that binds a table keeps them. The Intel UHD Graphics 630 and an Apple M3 Max answer 512
/// (measured: docs/graphics-kext/dgpu-oracle.md Q29).
pub const OPTIONS: u64 = 512;

/// `id<MTLFunctionHandle>`: rt's handle of a function a compute pipeline state links.
pub struct RtFunctionHandle {
    rt: rt::FunctionHandle,
}

impl RtFunctionHandle {
    /// The handle `rt`.
    pub fn new(rt: rt::FunctionHandle) -> RtFunctionHandle {
        RtFunctionHandle { rt }
    }

    /// rt's handle.
    pub fn rt(&self) -> rt::FunctionHandle {
        self.rt
    }
}

/// `id<MTLVisibleFunctionTable>`: rt's table of a compute pipeline state, of its entries.
pub struct RtFunctionTable {
    rt: rt::VisibleFunctionTable,
    /// Its entries, `functionCount`.
    count: u32,
}

impl RtFunctionTable {
    /// The table `rt` of `count` entries.
    pub fn new(rt: rt::VisibleFunctionTable, count: u32) -> RtFunctionTable {
        RtFunctionTable { rt, count }
    }

    /// rt's table.
    pub fn rt(&self) -> &rt::VisibleFunctionTable {
        &self.rt
    }
}

impl VisibleFunctionTable<RtPlatform> for RtObj<RtFunctionTable> {
    /// rt's: the device address of the table's words.
    fn gpu_resource_id(&self) -> u64 {
        crate::recorder::plain_resource_id(self, || self.rt.gpu_resource_id())
    }

    /// The bytes of the table's words in whole pages ([`PAGE`]).
    fn allocated_size(&self) -> u64 {
        function_table_entry_offset(self.count).next_multiple_of(PAGE)
    }

    /// [`OPTIONS`].
    fn resource_options(&self) -> u64 {
        OPTIONS
    }

    /// rt's set of the entries from `range.location`. A list of another length than the range is
    /// Metal's refusal, and a range past the table rt's (the module's docs), each counted and
    /// setting none of the entries.
    fn set_functions(
        &self,
        functions: &[Option<&<RtPlatform as Platform>::FunctionHandle>],
        range: NsRange,
    ) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!(SET_FUNCTIONS, self, functions, range);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            if u64::try_from(functions.len()).ok() != Some(range.length) {
                let why = "a list of another length than its range";
                refused_as(SET_FUNCTIONS, RtRefusal::Refused, &why);
                return Ok(());
            }
            // A first entry past rt's 32 bits is past every table, which rt refuses at its word's
            // maximum.
            let first = u32::try_from(range.location).unwrap_or(u32::MAX);
            let words: Vec<Option<rt::FunctionHandle>> = (functions.iter())
                .map(|function| function.map(|f| f.rt()))
                .collect();
            if let Err(error) = self.rt.set_functions(&words, first) {
                rt_refused(SET_FUNCTIONS, &error);
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::num::ToU64 as _;
    use crate::core::platform::Handle as _;
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{Buffer as _, ComputePipeline as _, Library as _};
    use crate::core::platform::types::{CommandBufferStatus, MtlSize};
    use crate::core::platform::values::FullComputePipelineDesc;
    use crate::core::raise::OnFault;
    use crate::rt_platform::unbuilt::RtRefusal;
    use crate::rt_platform::{RtBuffer, RtFunction, RtQueue};

    /// `visible_tables.metal`'s functions, compiled by build.rs: `table_kernel`'s thread `i`
    /// writes what entry `which[i]` of the table at buffer 0 answers of `i + 10`, `table_add1`
    /// answering its argument plus 1 and `table_mul2` twice it.
    const TABLES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/visible_tables.metallib"));

    /// The threads of each dispatch.
    const THREADS: usize = 8;

    /// The entry each thread calls, `which[i]`: the four entries of a table, twice.
    const WHICH: [u32; THREADS] = [0, 1, 2, 3, 0, 1, 2, 3];

    /// What an output holds before its dispatch writes it.
    const UNWRITTEN: i32 = 0x5a5a_5a5a;

    /// A function an entry calls, as the CPU computes it.
    type Op = fn(i32) -> i32;

    const ADD1: Op = |x| x + 1;
    const MUL2: Op = |x| x * 2;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// What `table_kernel` writes through a table of `entries`, each the function an entry calls
    /// (`None` for a nil entry, which calls nothing and answers 0 on rt).
    fn expected(entries: [Option<Op>; 4]) -> Vec<i32> {
        (WHICH.iter().zip(10..))
            .map(|(&which, x)| entries[which as usize].map_or(0, |f| f(x)))
            .collect()
    }

    /// The first [`THREADS`] words of the Shared `buffer`.
    fn words(buffer: &RtObj<RtBuffer>) -> Vec<i32> {
        assert!((4 * THREADS).to_u64() <= buffer.length());
        // SAFETY: a Shared buffer's contents are mapped, at least `THREADS` words long.
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr::<i32>(), THREADS) }.to_vec()
    }

    /// Encodes `cmds` into one compute encoder of a command buffer of `queue` and ends it, runs
    /// `between`, then commits the command buffer and waits for it: its status then.
    fn run(
        queue: &RtObj<RtQueue>,
        cmds: Vec<ComputeCmd<'_, RtPlatform>>,
        between: impl FnOnce(),
    ) -> CommandBufferStatus {
        let on = OnFault::Return;
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let encoder = made.compute_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a compute encoder");
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(encoder.encode(on, ComputeCmd::EndEncoding), Ok(()));
        between();
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        made.status()
    }

    /// `buffers`, the entries each thread calls and its output, bound at buffers 1 and 2, then a
    /// dispatch of [`THREADS`] threads.
    fn dispatch<'a>(
        buffers: &'a [Option<&'a RtObj<RtBuffer>>; 2],
    ) -> [ComputeCmd<'a, RtPlatform>; 2] {
        [
            ComputeCmd::SetBuffers {
                buffers,
                offsets: &[0, 0],
                range: NsRange::new(1, 2),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(THREADS.to_u64(), 1, 1),
                threads_per_threadgroup: MtlSize::new(THREADS.to_u64(), 1, 1),
            },
        ]
    }

    /// A compute pipeline state's function handles and visible function tables on rt: a handle of
    /// each function the state links, of the same function made again too, and none of the
    /// kernel or of a function it does not link; tables made, their entries set one and many,
    /// bound at a buffer index and in a range, and unbound with nil. A dispatch calls the function
    /// each entry holds, an entry set after the encoding and before the commit the one it calls;
    /// a nil entry, and every entry of an unbound slot, calls nothing and answers 0 (rt's
    /// answer; Metal's is undefined). A set past the table is rt's refusal, and a list of another
    /// length than its range Metal's, each counted and setting none of the entries. The tables'
    /// resource IDs are rt's, two tables' apart, and their sizes the words' pages.
    #[test]
    fn function_tables_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let library = device.new_library(TABLES).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let function = |name: &str| {
            let made = library.new_function(on, name).expect("never a raise");
            made.expect("a function of the library")
        };
        let kernel = function("table_kernel");
        let (add1, mul2) = (function("table_add1"), function("table_mul2"));
        let mut desc = FullComputePipelineDesc::new(&kernel);
        desc.linked.functions = vec![add1.clone_ref(), mul2.clone_ref()];
        let state = device.new_compute_pipeline_state_full(&desc);
        let state = state.expect("a compute state").expect("a compute state");

        // The handles of the linked functions, as made again too; none of any other.
        let handle = |f: &RtObj<RtFunction>| state.function_handle(f).expect("never a raise");
        let h_add = handle(&add1).expect("table_add1's handle");
        let h_mul = handle(&mul2).expect("table_mul2's handle");
        assert_ne!(h_add.rt().word(), h_mul.rt().word());
        let again = handle(&function("table_add1")).map(|h| h.rt().word());
        assert_eq!(again, Some(h_add.rt().word()));
        assert!(handle(&kernel).is_none());
        assert!(handle(&function("table_arg_kernel")).is_none());

        let table = |count: u64| {
            let made = state.new_visible_function_table(count);
            made.expect("never a raise").expect("a table")
        };
        let set = |table: &RtObj<RtFunctionTable>,
                   entries: &[Option<&RtObj<RtFunctionHandle>>],
                   first: u64| {
            let range = NsRange::new(first, entries.len().to_u64());
            table.set_functions(entries, range).expect("never a raise");
        };
        // One entry, then a range of a handle and a nil entry; entry 3 stays nil.
        let held = table(4);
        set(&held, &[Some(&h_add)], 0);
        set(&held, &[Some(&h_mul), None], 1);
        let swapped = table(4);
        set(
            &swapped,
            &[Some(&h_mul), Some(&h_add), Some(&h_mul), Some(&h_add)],
            0,
        );
        let late = table(4);
        set(&late, &[Some(&h_add); 4], 0);

        // Past the table, and past rt's 32 bits: rt's refusals. A list of another length than its
        // range: Metal's. None sets an entry (the dispatches below).
        let ours = (RtRefusal::Ours.code(), counted(RtRefusal::Ours.code()));
        let metal = (
            RtRefusal::Refused.code(),
            counted(RtRefusal::Refused.code()),
        );
        set(&held, &[Some(&h_add), Some(&h_add)], 3);
        set(&held, &[Some(&h_add)], u64::MAX);
        let short = held.set_functions(&[Some(&h_add)], NsRange::new(2, 2));
        assert_eq!(short, Ok(()));
        assert_eq!(counted(ours.0), ours.1 + 2);
        assert_eq!(counted(metal.0), metal.1 + 1);

        let (a, b) = (held.gpu_resource_id(), swapped.gpu_resource_id());
        assert!(a != 0 && b != 0 && a != b, "{a:#x} {b:#x}");
        assert_eq!(
            (held.allocated_size(), held.resource_options()),
            (PAGE, OPTIONS)
        );
        let sizes = [0, 1022, 1023].map(|count| table(count).allocated_size());
        assert_eq!(sizes, [4096, 4096, 8192]);

        let buffer = || {
            let made = device.new_buffer_of_length(on, (4 * THREADS).to_u64(), 0);
            let made = made.expect("never a raise").expect("a buffer");
            // SAFETY: the Shared buffer's contents are mapped, `THREADS` words long.
            unsafe {
                let at = made.contents().as_ptr::<i32>();
                std::slice::from_raw_parts_mut(at, THREADS).fill(UNWRITTEN);
            }
            made
        };
        let which = buffer();
        // SAFETY: as above.
        unsafe {
            let at = which.contents().as_ptr::<u32>();
            at.copy_from_nonoverlapping(WHICH.as_ptr(), THREADS);
        }
        let outs: Vec<RtObj<RtBuffer>> = (0..4).map(|_| buffer()).collect();
        let bound: Vec<[Option<&RtObj<RtBuffer>>; 2]> =
            (outs.iter()).map(|out| [Some(&which), Some(out)]).collect();
        let [bind0, run0] = dispatch(&bound[0]);
        let [bind1, run1] = dispatch(&bound[1]);
        let [bind2, run2] = dispatch(&bound[2]);
        let [bind3, run3] = dispatch(&bound[3]);
        let status = run(
            &queue,
            vec![
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state),
                },
                ComputeCmd::SetVisibleFunctionTable {
                    function_table: Some(&held),
                    buffer_index: 0,
                },
                bind0,
                run0,
                ComputeCmd::SetVisibleFunctionTables {
                    function_tables: &[Some(&swapped)],
                    range: NsRange::new(0, 1),
                },
                bind1,
                run1,
                ComputeCmd::SetVisibleFunctionTable {
                    function_table: None,
                    buffer_index: 0,
                },
                bind2,
                run2,
                ComputeCmd::SetVisibleFunctionTables {
                    function_tables: &[Some(&late)],
                    range: NsRange::new(0, 1),
                },
                bind3,
                run3,
            ],
            || set(&late, &[Some(&h_mul)], 1),
        );
        assert_eq!(status, CommandBufferStatus::COMPLETED);
        let (add, mul) = (Some(ADD1), Some(MUL2));
        assert_eq!(words(&outs[0]), expected([add, mul, None, None]));
        assert_eq!(words(&outs[1]), expected([mul, add, mul, add]));
        assert_eq!(words(&outs[2]), [0; THREADS]);
        assert_eq!(words(&outs[3]), expected([add, mul, add, add]));
    }
}
