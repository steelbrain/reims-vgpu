//! `id<MTLIndirectCommandBuffer>` and its commands: core's `IndirectCommandBuffer`,
//! `IndirectRenderCommand` and `IndirectComputeCommand`, over the runtime's.
//!
//! - **The commands** are the runtime's: each setter writes the command's record, and each
//!   getter reads it back typed (`IndirectCommandBuffer::commands`, the runtime's stand-in for
//!   Metal's private getters), as the records stand now, whoever wrote them. A buffer slot's
//!   getters are the runtime's (`EncodedComputeCommand::kernel_buffer_address` and its kin), past
//!   the 32 slots a record holds too, which the runtime keeps beside the records and its blits
//!   move as Metal's GPU does: the buffer's address plus the offset's low 32 bits, or those bits
//!   alone for nil, and the stride, all ones after a plain set; 0 and 0 for a slot never set or
//!   reset. An execution the GPU replays binds a nil buffer's bits as the address Metal's function
//!   sees, and one replayed on the host binds none and announces `indirect-nil-buffer-offset`,
//!   which dies ([`crate::answer::DIES`]).
//!   A threadgroup memory length's setter and getter are the runtime's too
//!   (`IndirectComputeCommand::threadgroup_memory_length`), past the 31 a record holds as well,
//!   which Metal keeps for its getter alone, as the runtime does: 38 a command, an index of 38 or
//!   more reaching a later command's, as Metal's, which the runtime's blits move with the
//!   records, and those past the last command's, where Metal keeps each command's words of its
//!   pipeline and its dispatch, which the runtime's copies and resets move as Metal's do; the
//!   index's low 32 bits, as Metal's take them. A length there that Metal's setters, a copy or,
//!   over a buffer that inherits its pipeline state, a blit wrote otherwise than the runtime's is
//!   relayed at its getter, a death in strict mode, until core sets it again ([`Tail`]). A copy
//!   into a command of a
//!   buffer that does not inherit its pipeline state clears its lengths 36 and 37 in the
//!   runtime, where Metal's carries them from a command that names no pipeline state: their
//!   getter is relayed alike, until core sets them or a reset clears them
//!   ([`Sides::copied_lengths`]).
//! - **The pipeline state's `uniqueIdentifier`** a command's getter answers is the runtime's
//!   (`pipeline_unique_identifier`): that of the state its record names by the state's resource
//!   ID, which a record the GPU wrote (a kernel's, or a blit's copy) names with no setter's, as
//!   Metal's answers that state's own (measured: a kernel's set of a compute and of a render
//!   state read from an argument buffer); 0 for none, as Metal's. Metal answers the identifier
//!   of the state the record was written with, a released one included, though a state made
//!   later took its ID (measured), and so does the runtime: the state the CPU set, unless the GPU
//!   may have written the record while another state held the ID, and for a record the GPU
//!   wrote, the state that held the ID as the GPU last could write it. Where it cannot tell
//!   which state the record names, it answers none, and the getter dies. The runtime's command
//!   names its state without holding it, as Metal's does: the guest's release gives the ID back,
//!   and a state made later takes it, with the released one's identifier too, as on Metal
//!   (measured), which the command then answers.
//! - **A blit's copy past the destination's end** lands the commands that fit, as Metal's does,
//!   and takes every command before the index as written over ([`IcbObject::overrun`]): Metal
//!   writes over some of those records in its own layout, by a shape this platform does not
//!   follow, and no other buffer changes. A getter, an execution or a copy of one dies until a
//!   reset or a copy writes its whole record again, and so does a `useResource` form that makes
//!   the buffer resident, as a function of the encoder may copy any of its commands
//!   (`copy_command`), where this platform's copy would carry the intact record
//!   ([`IcbObject::resident`]); a copy past the end in a shape nothing measured dies at the
//!   copy (`encode_blit::overrun_measured`).
//! - **Getters Metal reads raw** answer the values Metal's do: a nil patch or control-point index
//!   buffer is Metal's address `0x1eeee1111` ([`NIL_BUFFER_ADDRESS`]), a base vertex is its low 32
//!   bits, zero-extended, every stride of a buffer without dynamic attribute strides is all ones,
//!   and a patch draw's factor scale is 1. Metal reads past the descriptor's buffers and
//!   lengths, which core never asks for; they answer 0 here (a stride all ones without dynamic
//!   attribute strides). Metal's arguments getters
//!   read a record of any kind as their own; core reads each only for a command of its kind, and
//!   another kind's dies here ([`another_kind`]).
//! - **Metal's private primitive types** 5 to 8, a line loop, a triangle fan, quads and a quad
//!   strip, which a command's draws take, direct and over indices, go to the runtime, whose
//!   executions draw each as an encoder's draw of the type draws ([`rt::DrawPrimitiveType`];
//!   compared with Metal's pixel for pixel, inheriting the encoder's state or not,
//!   `private_types_draw_over_indices_and_arguments_as_metals`); the arguments getters answer
//!   the runtime's, the type set but 0 for quads and a quad strip, as Metal's do
//!   ([`rt::IndirectRenderCommand::draw`]), and `getCommandType` the runtime's too. A draw of
//!   type 9, at which Metal's GPU fails the command buffer that runs it, dies at its setter, as
//!   does a type past it ([`primitive`]). A kernel's draw of a type past Metal's five writes
//!   points, as Metal's encoding does (the runtime's).
//! - **An indexed patch draw of no control-point index buffer** is a patch draw, as Metal records
//!   it (measured: `getCommandType` answers `DrawPatches` and `drawPatchesArguments` its
//!   arguments, whatever the descriptor's types); a nil index or tessellation factor buffer,
//!   which Metal crashes at, dies.
//! - **Numbers the guest sees** are Metal's, the runtime's: the `gpuResourceID`, which is also
//!   the `resourceIndex`, from Metal's pool of IDs of indirect command buffers, and the
//!   `uniqueIdentifier`, Metal's address of the buffer in its range of the GPU's address space
//!   (`rt::IndirectCommandBuffer::unique_identifier`). Metal's own encoders take ranges of that
//!   range, which the runtime takes as an M3 Max's take them where that is fixed (a blit
//!   encoder's and a render encoder's at their opening, a dispatch's or a blit command's as its
//!   command buffer is scheduled), and not a render pass's run, which takes more by its targets
//!   and what the device drew before: of a buffer made after a command buffer that encoded a
//!   render encoder was scheduled, after a resource state encoder opened, or once Metal's
//!   encoders' ranges passed what the runtime models (`identifier_unmeasured`), the read is
//!   relayed, a death in strict mode ([`crate::answer::Inexact::IcbIdentifierAfterEncoder`]).
//!   The runtime's `gpu_resource_id` is the ID a translated kernel encoding through an argument
//!   buffer reaches a buffer's records by, so the kernel reaches the records of the buffer the
//!   guest names. A reservation is the runtime's, as Metal takes, ignores or drops it, to
//!   2^31 - 1, and the runtime makes a buffer at every ID below 2^31 (`RECORDS_ID_LIMIT`), each
//!   ID's records at an address of their own, as Metal makes them; the one Metal spins at, which
//!   the runtime refuses as one, parks the thread holding the lock creations wait on
//!   ([`RESERVING`], [`reserve`], [`new_indirect_command_buffer`]).
//! - **A released buffer's ID comes back after the completed handlers** of the command buffers
//!   that execute it or make it resident, which hold it while they may run ([`RESERVING`]):
//!   Metal gives it back once those handlers returned, and not always before
//!   `waitUntilCompleted` returns (measured on an Apple M3 Max, macOS 27.0.1, 2026-10-26, of a
//!   buffer released while a command buffer blocked on an event executed it or made it resident:
//!   a buffer made in a completed handler took another ID in 4,800 of 4,800, and so did one a
//!   thread made while the handler that woke it ran on, in 2,942 of 2,942; after the handlers,
//!   the ID was back when the wait returned in 6,000 of 6,000 without added load, and in 8,390 of
//!   8,400 under 16 CPU spinners and a GPU load, the other 10 back up to 884 us after it; a
//!   thread a completed handler woke found it still held in 8 of 1,400 there, all of buffers
//!   made resident). The runtime gives it back as its command buffer lets go of what it
//!   retained, after the completed handlers and before its waiters return, Metal's order where no
//!   load delays it: held in a handler, back once the wait returned, and maybe still held for a
//!   thread a handler woke, as on Metal.
//! - **Refusals** die at the selector, naming the runtime's error: Metal without its validation
//!   layer raises at none of them, and goes on at those measured (an index at or past the size,
//!   a command of the other kind, and a Private buffer's command each answer a command object).

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use metal2vulkan::metal::{IndexType, IndirectCommandType};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::Truncate;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::icb::{
    DispatchThreadgroupsArgs, DispatchThreadsArgs, DrawArgs, DrawIndexedArgs,
    DrawIndexedPatchesArgs, DrawPatchesArgs, IcbComputeCmd, IcbRenderCmd, ImageBlockArgs,
    IndirectCommandBuffer, IndirectComputeCommand, IndirectRenderCommand, TessellationFactorArgs,
};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
use reims_vgpu_core::platform::values::IcbDesc;
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::sync::{Mutex, MutexGuard};

use crate::answer::{
    Inexact, Policy, decide, get, hang, inexact, metal_aborts, plain, refused, send,
    unrepresentable, unstopped_at,
};
use crate::gpu::command::{Progress, Stage};
use crate::gpu::device::VulkanDevice;
use crate::gpu::metal_value;
use crate::platform::{Obj, VulkanPlatform};

/// The address Metal's patch and control-point index getters answer for a nil buffer (measured:
/// a patch draw of a nil patch index buffer, and an indexed one of nil buffers, on a fresh
/// command and on one that held a buffer).
pub const NIL_BUFFER_ADDRESS: u64 = 0x1_eeee_1111;

// ---- Metal's numbers.

/// The first resource ID a placement cannot take: Metal writes past its table of IDs at 16384 or
/// more, which ends its process at random (measured: 16383 always made, 16384 to 50367 made or
/// crashed from run to run, 100000 crashed).
const ID_LIMIT: u64 = 16384;

/// How many threadgroup memory lengths Metal keeps a command of a compute buffer, in one array
/// of the buffer's, so that an index of 38 or more reaches a later command's length, as the
/// runtime keeps them (`IndirectComputeCommand::set_threadgroup_memory_length`).
const LENGTHS_A_COMMAND: u64 = 38;

/// The lock of Metal's pool of indirect command buffer IDs, of the process (one device's, as the
/// host has one), which a reservation Metal spins at holds for good ([`reserve`]) and every
/// creation takes ([`new_indirect_command_buffer`]). Isolated tasks share it, as every task's
/// device is the one object of the process's device ([`crate::gpu::device::adopt`]).
///
/// Measured on this host: plain allocation hands out IDs from Metal's pool, which the runtime
/// follows (the tests compare it with Metal: its first 32 IDs, the smallest free range first, a
/// reservation's split); a placement (`setResourceIndex:` of a nonzero index) takes the index's
/// low 32 bits whoever holds them, reserved or not, and takes no ID from the pool.
///
/// A reservation (`reserveResourceIndicesForResourceType:indices:indexCount:` of type 5) keeps
/// its IDs from plain allocation for good. Only the process's first call with an ID counts, and
/// only before a buffer was made, placed or not: Metal ignores any other, whatever it holds,
/// saying "reserve() must be called before any allocation takes place". A call holding an ID of
/// 2^32 or more is dropped whole, and a later call is the reservation. 0 is no ID: beside others
/// the call is theirs (0 and 20 hand out what 20 alone does: 21 to 31, 1 to 19, then 32 on), and
/// alone it reserves nothing (plain IDs from 1), but a later reservation is taken beside the
/// pool's first range, and Metal then hands out IDs twice (measured, in order of making: 0 then
/// 20, 21 to 31, 1 to 19, then 1 to 10; 0 then 10, 1 to 9, 11 to 31, then 1 to 15; 0 then 100,
/// 101 to 127, 1 to 31, then 1 to 82; 0 then 2^31 - 1, 1 to 31, then 1 to 31 and 32 on). The
/// runtime does each of these, its late call logged as Metal's
/// (`metal2vulkan-late-reservation`). A call Metal takes holding an ID of 2^31 to 2^32 - 1 never
/// returns, which the runtime refuses, reserving nothing, as a call Metal hangs at
/// (`Problem::ReservationHangs`): Metal spins in
/// `AGX::Mempool<32u,0u,true,0u,0u,unsigned long long>::reserve` at full CPU (measured at 2^31,
/// 2^31 + 5 and 2^32 - 1, the longest run 5 minutes, and at 2^31 + 5 after a call of 0 alone),
/// where 2^31 - 1 is taken and hands out 1 next. While it spins it holds its pool's lock
/// (measured from a second thread, 5 s a case, the waits sampled): another buffer's creation,
/// placed or not, waits on it in `AGX::Mempool<32>::RangeAllocation`, one placed at 16384 or
/// 100000 too, past the table of IDs Metal ends its process at ([`ID_LIMIT`]), and another
/// reservation of type 5 in `AGX::Mempool<32>::reserve`. A device texture and a command queue
/// wait too where they are the process's first, in `-[AGXG15XFamilyDevice setupDeferred]`, which
/// core's command queues, made with the device, run before any reservation; buffers, textures
/// over a buffer, heaps and their textures, sampler states, command buffers' creation, commit and
/// wait, and texture reservations go on. No buffer lives while it spins: one made first makes
/// Metal ignore the reservation. At the runtime's refusal the thread parks for good holding this
/// lock ([`answered`], [`crate::answer::hang`]), as a hang stays a hang, so every later creation
/// and reservation of type 5, on any thread, an isolated task's among them, waits on it for good,
/// before any death of its own ([`new_indirect_command_buffer`]).
///
/// A buffer released is not free while a command buffer that uses it may run (measured, a
/// command buffer blocked on an event, then completed): a compute or render encoder's execution
/// or `useResource` of it keeps the buffer and its ID until the command buffer completed and its
/// completed handlers returned (the module docs), or goes uncommitted; a blit's copy (from it or
/// into it) or reset keeps neither, its ID free at once. The runtime holds its IDs so too.
static RESERVING: Mutex<()> = Mutex::new(());

/// `reserveResourceIndicesForResourceType:indices:indexCount:` (`sel`) of indirect command
/// buffers (type 5) on `device`: the runtime's reservation of `ids`, which takes, ignores or
/// drops it as Metal does ([`RESERVING`]); the thread parks for good where Metal spins, which the
/// runtime's refusal tells ([`answered`]). Another refusal of its would die at the selector.
pub(crate) fn reserve(device: &VulkanDevice, sel: &'static str, ids: &[u64]) -> RaiseResult<()> {
    // The pool's lock stays held until the runtime answered, as a call Metal spins at holds it;
    // no stop of an ended task comes between the two, so a call that took the lock reaches the
    // runtime.
    let held = plain(|| RESERVING.lock());
    #[cfg(test)]
    if let Some(amid) = AMID_RESERVATION.take() {
        amid();
    }
    let ids: Vec<rt::ResourceId> = ids.iter().map(|&id| rt::ResourceId(id)).collect();
    let answer = unstopped_at(sel, || {
        device
            .raw
            .reserve_resource_ids(rt::ResourceIdKind::IndirectCommandBuffer, &ids)
    });
    answered(sel, held, answer)
}

/// What the runtime's `answer` to a reservation at `sel` makes of it, `held` the pool's lock:
/// where it refuses the call as one Metal spins at (`Problem::ReservationHangs`, [`spins`]), this
/// thread parks for good holding the lock, as Metal's spinning thread holds its pool's, so a
/// later creation or reservation waits on it for good; another refusal dies at `sel`.
fn answered(
    sel: &'static str,
    held: MutexGuard<'_, ()>,
    answer: Result<(), rt::Error>,
) -> RaiseResult<()> {
    match answer {
        Err(e) if spins(&e) => {
            let _held = held;
            hang()
        }
        Err(e) => decide(sel, Policy::Dies, e),
        Ok(()) => Ok(()),
    }
}

/// Whether `e` is the runtime's refusal of a reservation Metal spins at for good: an ID of 2^31
/// to 2^32 - 1 (`Problem::ReservationHangs`).
fn spins(e: &rt::Error) -> bool {
    matches!(
        e,
        rt::Error::Invalid(rt::Invalid {
            problem: rt::Problem::ReservationHangs { .. },
            ..
        })
    )
}

#[cfg(test)]
thread_local! {
    /// What a test runs in [`reserve`] on this thread once it holds [`RESERVING`], before the
    /// runtime takes the reservation.
    static AMID_RESERVATION: std::cell::Cell<Option<fn()>> = const { std::cell::Cell::new(None) };
}

// ---- The buffer.

/// What an indirect command buffer's commands take of the encoder that executes them: the
/// textures always, as no command sets one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inherits {
    /// The pipeline state (`inheritPipelineState`).
    pub pipeline: bool,
    /// The buffers (`inheritBuffers`).
    pub buffers: bool,
}

/// An indirect command buffer: the runtime's, with the commands Metal's copies past its end may
/// have written over ([`IcbObject::overrun`]).
#[derive(Debug)]
pub struct IcbObject {
    /// The runtime's buffer.
    pub raw: rt::IndirectCommandBuffer,
    /// Whether its commands take attribute strides.
    strides: bool,
    /// What its commands take of the encoder that executes them.
    pub(crate) inherits: Inherits,
    /// The commands Metal's copies past its end may have written over.
    sides: Arc<Mutex<Sides>>,
    /// Which shape of those Metal's copies past a buffer's end were measured at it is, if any
    /// ([`OverrunShape::of`]).
    pub(crate) overrun_shape: Option<OverrunShape>,
    /// How many committed command buffers that have not ended execute it as their commit read
    /// its records ([`crate::gpu::command::executions::reads`]).
    pub(crate) readers: Arc<AtomicUsize>,
}

/// A buffer of a shape Metal's copies of commands past a buffer's end were measured at, on this
/// host (`encode_blit::overrun_measured`, which also takes only a source and a destination of
/// one count of commands, as every measurement's were), made with options 0 (Shared storage),
/// of one of two descriptors with nothing else set, each of 2 buffers a stage and dynamic
/// attribute strides (two runs a shape, buffers of 2 to 64 commands, 1 to 63 commands past the
/// end). Where Metal's copy past the end writes depends on its layout of the descriptor's
/// records, which nothing here models, so a buffer of any other descriptor is of none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OverrunShape {
    /// Both dispatch kinds (`CONCURRENT_DISPATCH | CONCURRENT_DISPATCH_THREADS`: threadgroups
    /// and threads, both concurrent), 2 kernel buffers and 2 threadgroup memory lengths; the
    /// buffers measured held thread dispatches alone.
    Compute,
    /// The four draw kinds, 2 vertex and 2 fragment buffers.
    Render,
}

impl OverrunShape {
    /// The fewest and the most commands of a buffer of [`OverrunShape::Compute`] or
    /// [`OverrunShape::Render`] measured.
    const COMMANDS: std::ops::RangeInclusive<u64> = 2..=64;

    /// The shape of a buffer of `desc`, `max_count` commands and raw `options`, where it is one
    /// that was measured.
    fn of(desc: &IcbDesc, max_count: u64, options: u64) -> Option<OverrunShape> {
        let compute = IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 2,
            max_kernel_threadgroup_memory_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        };
        let render = IcbDesc {
            command_types: 0xf,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        };
        if options != 0 {
            return None;
        }
        match desc {
            d if *d == compute && Self::COMMANDS.contains(&max_count) => {
                Some(OverrunShape::Compute)
            }
            d if *d == render && Self::COMMANDS.contains(&max_count) => Some(OverrunShape::Render),
            _ => None,
        }
    }
}

/// The commands of a buffer that Metal's copies past its end may have written over in its own
/// layout, and what decides it ([`IcbObject::overrun`]).
#[derive(Debug, Default)]
struct Sides {
    /// Each command taken as written over: a getter, an execution or a copy of it dies until a
    /// reset or a copy writes its whole record again.
    corrupted: BTreeSet<u32>,
    /// The copies past the buffer's end of command buffers the GPU may still run.
    blits: Vec<Blit>,
    /// The progress of each command buffer that executes the buffer or makes it resident, once
    /// each, while the GPU may run it ([`IcbObject::executed`], [`IcbObject::resident`]).
    executions: Vec<std::sync::Weak<Progress>>,
    /// What Metal answers of the threadgroup memory lengths past the last command's that the
    /// runtime's may not.
    tail: Tail,
    /// An encoder made the buffer resident for writing: its functions may write its records,
    /// which Metal's GPU may write words past the last command's lengths of ([`Tail`]).
    kernel_written: bool,
    /// The threadgroup memory lengths 36 and 37 of the commands a copy wrote into, by their
    /// place in the buffer's array of them ([`LENGTHS_A_COMMAND`]), in a buffer that does not
    /// inherit its pipeline state, that core has not set since and no reset has cleared: the
    /// runtime's copy clears them, where Metal's carries them from a command that names no
    /// pipeline state (measured on the M3 Max, macOS 27.0.1:
    /// `blits_move_lengths_past_the_record_as_metals`). Their getter is relayed, a death in strict
    /// mode ([`Inexact::IcbCopiedLength`]).
    copied_lengths: BTreeSet<u64>,
}

impl Sides {
    /// The places of lengths 36 and 37 of each of commands `range` ([`Sides::copied_lengths`]).
    fn copied_places(range: Range<u64>) -> impl Iterator<Item = u64> {
        range.flat_map(|i| [36, 37].map(|slot| i * LENGTHS_A_COMMAND + slot))
    }
}

/// The threadgroup memory lengths past the last command's ([`LENGTHS_A_COMMAND`]) of a buffer of
/// `n` commands, which Metal and the runtime read back as core set them, where Metal keeps each
/// command `i`'s words: 2 of its pipeline from `2 i` past the last command's lengths
/// ([`Tail::pipeline`]), 6 of its dispatch from `2 n + 8 i` and 2 of its imageblock size after
/// them ([`Tail::dispatch`], [`Tail::imageblock`]), and one word at `10 n` that a copy writes
/// ([`Tail::copied`]); the runtime's copies and resets move the same words as Metal's (measured
/// on the M3 Max, macOS 27.0.1, buffers of 2 commands, every place up to 24 past the last
/// command's, compared with Metal's: `lengths_past_the_last_command_follow_metals_words`). The
/// runtime's answer of a word differs from Metal's
/// - after a pipeline, dispatch or imageblock setter of the command, whose words Metal's writes
///   and the runtime's does not;
/// - after a copy into the command: Metal's keeps the destination's pipeline words where the
///   source's are 0, where the runtime's copies the 0, and writes the word at `10 n`, which the
///   runtime's leaves; the dispatch words are the source's, as they were;
/// - after a blit's copy into a buffer, or reset of one, that inherits its pipeline state,
///   whose words the runtime keeps without the pipeline words, so that its blits move the
///   dispatch words at other places than Metal's;
/// - and in a buffer an encoder made resident for writing ([`Sides::kernel_written`]), whose
///   functions' encodings Metal's GPU may write words of, unmeasured.
///
/// A reset matches Metal's: on the CPU it zeroes the command's pipeline words, and a blit's its
/// pipeline and dispatch words, in a buffer that does not inherit its pipeline state. A word that
/// differs, or may, is relayed at its getter, a death in strict mode
/// ([`Inexact::IcbLengthPastTheLastCommand`]), until core sets it again.
#[derive(Debug, Default)]
struct Tail {
    /// The places in the buffer's array of lengths whose answer may not be Metal's.
    unknown: BTreeSet<u64>,
}

impl Tail {
    /// Of a buffer of `n` commands, the places of command `i`'s pipeline words.
    fn pipeline(n: u64, i: u64) -> Range<u64> {
        let at = n * LENGTHS_A_COMMAND + 2 * i;
        at..at + 2
    }

    /// Of a buffer of `n` commands, the places of command `i`'s dispatch words.
    fn dispatch(n: u64, i: u64) -> Range<u64> {
        let at = n * LENGTHS_A_COMMAND + 2 * n + 8 * i;
        at..at + 6
    }

    /// Of a buffer of `n` commands, the places of command `i`'s imageblock words.
    fn imageblock(n: u64, i: u64) -> Range<u64> {
        let at = n * LENGTHS_A_COMMAND + 2 * n + 8 * i + 6;
        at..at + 2
    }

    /// Of a buffer of `n` commands, the place of the word a copy writes.
    fn copied(n: u64) -> u64 {
        n * LENGTHS_A_COMMAND + 10 * n
    }

    /// Of a buffer of `n` commands, the places of every command's words.
    fn commands(n: u64) -> Range<u64> {
        n * LENGTHS_A_COMMAND..Self::copied(n)
    }

    /// Of a buffer of `n` commands, the places of every command's words and the copy's.
    fn words(n: u64) -> std::ops::RangeInclusive<u64> {
        n * LENGTHS_A_COMMAND..=Self::copied(n)
    }

    /// The dispatch and imageblock words of a buffer of `n` commands, command `i`'s.
    fn dispatch_and_imageblock(n: u64, i: u64) -> Range<u64> {
        Self::dispatch(n, i).start..Self::imageblock(n, i).end
    }

    /// The places of `range` that may not answer Metal's, as offsets from its start.
    fn within(&self, range: Range<u64>) -> Vec<u64> {
        let start = range.start;
        self.unknown
            .range(range)
            .map(|place| place - start)
            .collect()
    }

    /// The places of `range` may not answer Metal's.
    fn mark(&mut self, range: impl IntoIterator<Item = u64>) {
        self.unknown.extend(range);
    }

    /// The places of `range` answer Metal's.
    fn clear(&mut self, range: Range<u64>) {
        let gone: Vec<u64> = self.unknown.range(range).copied().collect();
        for place in gone {
            self.unknown.remove(&place);
        }
    }
}

/// The death of a getter, an execution or a copy of a command whose record Metal's copy past the
/// buffer's end may have written over ([`IcbObject::overrun`]).
const OVERRUN: &str = "a command whose record Metal's copy of commands past the indirect command \
                       buffer's end may have written over in its own layout, which this \
                       platform's copy does not";

/// What a `useResource` form that makes such a buffer resident dies naming
/// ([`IcbObject::resident`]).
const RESIDENT_OVERRUN: &str = "an indirect command buffer made resident for the encoder's \
                                functions, which may copy a command (copy_command) whose record \
                                Metal's copy of commands past the buffer's end may have written \
                                over in its own layout, where this platform's copy carries the \
                                intact record";

/// A blit's copy of commands past a buffer's end, kept by that buffer's [`Sides`] while its
/// command buffer may run it, which may write over `commands` in Metal's own layout
/// ([`IcbObject::overrun`]).
#[derive(Debug)]
struct Blit {
    /// Its command buffer's progress.
    progress: Arc<Progress>,
    /// The commands it may write over.
    commands: Range<u64>,
}

impl Blit {
    /// Whether it may still write over them: its command buffer may run it. Once that completed
    /// or failed, they are taken as written over ([`Sides::corrupted`]).
    fn matters(&self) -> bool {
        matches!(self.progress.stage(), Stage::Encoding | Stage::Committed)
    }
}

impl Sides {
    /// Whether each of commands `from` is taken as written over.
    fn copies(&self, from: Range<u64>) -> Vec<(u64, bool)> {
        from.map(|i| {
            let corrupted = u32::try_from(i).is_ok_and(|i| self.corrupted.contains(&i));
            (i, corrupted)
        })
        .collect()
    }

    /// Commands from `to` are `moved`, copies of commands from `from`: each is written over
    /// where its source was, and whole where it was not.
    fn take(&mut self, from: u64, to: u64, moved: Vec<(u64, bool)>) {
        for (i, corrupted) in moved {
            let Ok(at) = u32::try_from(to + (i - from)) else {
                continue;
            };
            if corrupted {
                self.corrupted.insert(at);
            } else {
                self.corrupted.remove(&at);
            }
        }
    }

    /// Commands from `to` may or may not be `moved`, copies of commands from `from`: a blit of a
    /// failed command buffer may have copied them. Each whose source was written over may be
    /// too, and one written over stays so.
    fn unsettle(&mut self, from: u64, to: u64, moved: Vec<(u64, bool)>) {
        for (i, corrupted) in moved {
            if let Ok(at) = u32::try_from(to + (i - from))
                && corrupted
            {
                self.corrupted.insert(at);
            }
        }
    }

    /// Whether Metal's copy past the buffer's end may have written over a record of `range` (all
    /// of them, `None`), or may yet: one that completed or failed, or one of a command buffer the
    /// GPU may still run ([`IcbObject::overrun`]).
    fn overrun(&mut self, range: Option<&Range<u64>>) -> bool {
        self.blits.retain(Blit::matters);
        let reaches = |commands: &Range<u64>| {
            range.is_none_or(|r| r.start < commands.end && commands.start < r.end)
        };
        let marked = (self.corrupted.iter()).any(|&i| reaches(&(u64::from(i)..u64::from(i) + 1)));
        marked || self.blits.iter().any(|b| reaches(&b.commands))
    }
}

impl std::ops::Deref for IcbObject {
    type Target = rt::IndirectCommandBuffer;
    fn deref(&self) -> &rt::IndirectCommandBuffer {
        &self.raw
    }
}

impl IcbObject {
    /// The stride getter's answer where the runtime's getter has none (past the descriptor's
    /// count, which core never asks for): all ones on a buffer without dynamic attribute
    /// strides, as Metal answers every stride there, else 0.
    fn unset_stride(&self) -> u64 {
        if self.strides { 0 } else { u64::MAX }
    }

    /// A blit of the command buffer of `progress` (none once it went) copies commands `from` over
    /// the commands from `to` of `destination`: once it completed, each landed command is taken
    /// as written over where its source was ([`IcbObject::overrun`]), and as whole where it was
    /// not; once it failed, each whose source was written over may be.
    pub(crate) fn copied(
        &self,
        progress: Option<Arc<Progress>>,
        from: Range<u64>,
        destination: &IcbObject,
        to: u64,
    ) {
        let Some(progress) = &progress else {
            return;
        };
        let landed = to..to + (from.end - from.start);
        let cleared = !destination.inherits.pipeline;
        let (m, n) = plain(|| (self.raw.size(), destination.raw.size()));
        // The words a copy may write otherwise than Metal's, until it ran ([`Tail`]): over a
        // buffer that inherits its pipeline state, every command's; else the landed commands'
        // pipeline and dispatch words and the copy's word, the dispatch words taking the
        // source's once it ran.
        let misplaced = self.inherits.pipeline || destination.inherits.pipeline;
        let written = plain(|| self.sides.lock().kernel_written);
        plain(|| {
            let mut sides = destination.sides.lock();
            if cleared {
                sides
                    .copied_lengths
                    .extend(Sides::copied_places(landed.clone()));
            }
            match misplaced {
                true => sides.tail.mark(Tail::words(n)),
                false => {
                    for i in landed.clone() {
                        sides.tail.mark(Tail::pipeline(n, i));
                        sides.tail.mark(Tail::dispatch_and_imageblock(n, i));
                    }
                    sides.tail.mark([Tail::copied(n)]);
                }
            }
        });
        type Land = fn(&mut Sides, u64, u64, Vec<(u64, bool)>);
        let copy = |land: Land, ran: bool| {
            let (source, destination) = (Arc::clone(&self.sides), Arc::clone(&destination.sides));
            let (from, landed) = (from.clone(), landed.clone());
            move || {
                let (moved, unknown) = {
                    let source = source.lock();
                    let unknown: Vec<Vec<u64>> = (from.clone())
                        .map(|i| source.tail.within(Tail::dispatch_and_imageblock(m, i)))
                        .collect();
                    (source.copies(from.clone()), unknown)
                };
                let mut destination = destination.lock();
                land(&mut destination, from.start, to, moved);
                if cleared {
                    let places = Sides::copied_places(landed.clone());
                    destination.copied_lengths.extend(places);
                }
                // The dispatch words landed as the source's were, where the copy ran.
                if ran && !misplaced && !written {
                    for (i, unknown) in landed.zip(unknown) {
                        let words = Tail::dispatch_and_imageblock(n, i);
                        destination.tail.clear(words.clone());
                        destination
                            .tail
                            .mark(unknown.into_iter().map(|o| words.start + o));
                    }
                }
            }
        };
        progress.at_completion(copy(Sides::take, true));
        progress.at_failure(copy(Sides::unsettle, false));
    }

    /// A blit of the command buffer of `progress` (none once it went) resets commands `range`:
    /// once it completed, their whole records are written again.
    pub(crate) fn reset(&self, progress: Option<Arc<Progress>>, range: Range<u64>) {
        let Some(progress) = &progress else {
            return;
        };
        let n = plain(|| self.raw.size());
        // Over a buffer that inherits its pipeline state, the runtime's reset zeroes other words
        // than Metal's ([`Tail`]).
        let misplaced = self.inherits.pipeline;
        if misplaced {
            plain(|| self.sides.lock().tail.mark(Tail::commands(n)));
        }
        let sides = Arc::clone(&self.sides);
        progress.at_completion(move || {
            let mut sides = sides.lock();
            for place in Sides::copied_places(range.clone()) {
                sides.copied_lengths.remove(&place);
            }
            for i in range {
                if let Ok(at) = u32::try_from(i) {
                    sides.corrupted.remove(&at);
                }
                if !misplaced {
                    sides.tail.clear(Tail::pipeline(n, i));
                    sides.tail.clear(Tail::dispatch_and_imageblock(n, i));
                }
            }
        });
    }

    /// Command `index` was reset on the CPU: its whole record is written again, and, in a buffer
    /// that does not inherit its pipeline state, its pipeline words are zeroed, as Metal's are
    /// ([`Tail`]).
    fn rewritten(&self, index: u32) {
        let n = plain(|| self.raw.size());
        plain(|| {
            let mut sides = self.sides.lock();
            sides.corrupted.remove(&index);
            let at = u64::from(index);
            for place in Sides::copied_places(at..at + 1) {
                sides.copied_lengths.remove(&place);
            }
            if !self.inherits.pipeline {
                sides.tail.clear(Tail::pipeline(n, at));
            }
        });
    }

    /// Commands `range` were reset on the CPU at once: each as [`rewritten`](Self::rewritten)
    /// leaves one, in one pass over what the sides hold, not one per command.
    fn rewritten_range(&self, range: Range<u64>) {
        let n = plain(|| self.raw.size());
        plain(|| {
            let mut sides = self.sides.lock();
            sides
                .corrupted
                .retain(|&at| !range.contains(&u64::from(at)));
            sides.copied_lengths.retain(|&place| {
                !(range.contains(&(place / LENGTHS_A_COMMAND))
                    && matches!(place % LENGTHS_A_COMMAND, 36 | 37))
            });
            if !self.inherits.pipeline && !range.is_empty() {
                let words =
                    Tail::pipeline(n, range.start).start..Tail::pipeline(n, range.end).start;
                sides.tail.clear(words);
            }
        });
    }

    /// A setter of command `index` wrote the words `words` gives of a buffer of its size, as
    /// Metal's does and the runtime's does not ([`Tail`]).
    fn words_set(&self, index: u32, words: fn(u64, u64) -> Range<u64>) {
        let n = plain(|| self.raw.size());
        plain(|| (self.sides.lock().tail).mark(words(n, u64::from(index))));
    }

    /// Core set the threadgroup memory length `flat` places into the buffer's array of them,
    /// past the last command's ([`Tail`]): it answers as set again.
    fn tail_set(&self, flat: u64) {
        plain(|| self.sides.lock().tail.unknown.remove(&flat));
    }

    /// Core set the threadgroup memory length `flat` places into the buffer's array of them, of a
    /// command: a length 36 or 37 a copy cleared answers as set again
    /// ([`Sides::copied_lengths`]).
    fn record_set(&self, flat: u64) {
        plain(|| self.sides.lock().copied_lengths.remove(&flat));
    }

    /// `sel` reads the threadgroup memory length `flat` places into the buffer's array of them,
    /// of a command: relayed, a death in strict mode, where it is a length 36 or 37 a copy
    /// cleared, where Metal's carries it from a command that names no pipeline state
    /// ([`Sides::copied_lengths`]).
    fn record_read(&self, sel: &'static str, flat: u64) {
        if plain(|| self.sides.lock().copied_lengths.contains(&flat)) {
            inexact(sel, Inexact::IcbCopiedLength, || {
                "a threadgroup memory length 36 or 37 of a command a copy wrote into, which the \
                 runtime's copy clears and Metal's carries from a command that names no pipeline \
                 state: it answers the runtime's"
                    .to_owned()
            });
        }
    }

    /// `sel` reads the threadgroup memory length `flat` places into the buffer's array of them,
    /// past the last command's ([`Tail`]): relayed, a death in strict mode, where Metal's may
    /// differ, a word of a command or of a copy that Metal wrote otherwise than the runtime, or
    /// may have, since core set it.
    fn tail_read(&self, sel: &'static str, flat: u64) {
        let n = plain(|| self.raw.size());
        let unknown = plain(|| {
            let sides = self.sides.lock();
            let written = sides.kernel_written && Tail::words(n).contains(&flat);
            written || sides.tail.unknown.contains(&flat)
        });
        if unknown {
            inexact(sel, Inexact::IcbLengthPastTheLastCommand, || {
                "a threadgroup memory length past the last command's, a word of a command's \
                 pipeline or dispatch or of a copy, which Metal's setters, copies or GPU wrote \
                 otherwise than the runtime's, or may have: it answers the runtime's"
                    .to_owned()
            });
        }
    }

    /// A blit `sel` of the command buffer of `progress` (none once it went) copies commands into
    /// this buffer past its end, the commands that fit landing from `range`'s end on: Metal's
    /// writes over the records before them, `range`, in its own layout (measured on this host,
    /// two runs a shape, compute and render buffers of 2 to 64 commands, 1 to 63 commands past the
    /// end: some of those records turn into a dispatch of another kind, at shapes where others
    /// stay whole, and no other buffer changes). Every command of `range` is taken as overwritten
    /// once the command buffer completed or failed: a getter of one dies, as do an execution and
    /// a copy that reach one, until a reset or a copy writes its whole record again. While the
    /// GPU may run the blit, a getter of one races it and dies; a setter checks nothing, as each
    /// command is taken as written over once the blit ran, one reset before too. An execution or a
    /// copy that
    /// reaches one dies too, which Metal may run before or after it. An execution of the buffer,
    /// or a residency ([`IcbObject::resident`]), that another command buffer holds while the GPU
    /// may run it, which may run after the blit, dies here.
    pub(crate) fn overrun(
        &self,
        sel: &'static str,
        progress: Option<Arc<Progress>>,
        range: Range<u64>,
    ) {
        let Some(progress) = &progress else {
            return;
        };
        let elsewhere = plain(|| {
            let mut sides = self.sides.lock();
            sides
                .executions
                .retain(|p| p.upgrade().is_some_and(|p| p.running()));
            let elsewhere = (sides.executions.iter())
                .any(|p| p.upgrade().is_some_and(|p| !Arc::ptr_eq(&p, progress)));
            sides.blits.retain(Blit::matters);
            sides.blits.push(Blit {
                progress: Arc::clone(progress),
                commands: range.clone(),
            });
            elsewhere
        });
        if elsewhere {
            unrepresentable(
                sel,
                "a copy of commands past an indirect command buffer's end, which Metal writes \
                 over its other commands' records with, while another command buffer that may run \
                 after it executes the buffer or makes it resident",
            );
        }
        let mark = |sides: Arc<Mutex<Sides>>, range: Range<u64>| {
            move || {
                let mut sides = sides.lock();
                for i in range {
                    if let Ok(i) = u32::try_from(i) {
                        sides.corrupted.insert(i);
                    }
                }
            }
        };
        progress.at_completion(mark(Arc::clone(&self.sides), range.clone()));
        progress.at_failure(mark(Arc::clone(&self.sides), range));
    }

    /// Dies at `sel`, which reads command `index`'s record, where Metal's copy past the buffer's
    /// end may have written over it ([`IcbObject::overrun`]): one that completed or failed, or
    /// one of a committed command buffer the GPU may be running. Before the commit the GPU ran
    /// none of it, and a read answers what stands.
    fn intact(&self, sel: &'static str, index: u32) {
        let at = u64::from(index);
        let overrun = plain(|| {
            let mut sides = self.sides.lock();
            sides.blits.retain(Blit::matters);
            let committed =
                |b: &Blit| b.progress.stage() == Stage::Committed && b.commands.contains(&at);
            sides.corrupted.contains(&index) || sides.blits.iter().any(committed)
        });
        if overrun {
            unrepresentable(sel, OVERRUN);
        }
    }

    /// `sel` executes commands `range` of the buffer (all of them, `None`: an execution whose
    /// range a buffer holds) in the command buffer of `progress`: it dies where Metal's copy past
    /// the buffer's end may have written over one of their records, or may yet before they run
    /// ([`IcbObject::overrun`]); else the execution is kept while the GPU may run it.
    pub(crate) fn executed(
        &self,
        sel: &'static str,
        range: Option<Range<u64>>,
        progress: Option<&Arc<Progress>>,
    ) {
        self.reached(sel, range, progress, OVERRUN);
    }

    /// `sel`, a `useResource` form, makes the buffer resident for the functions of an encoder of
    /// the command buffer of `progress`, which may copy any of its commands (`copy_command`), as
    /// an execution of the whole buffer reaches them: it dies where Metal's copy past the
    /// buffer's end may have written over one of their records, or may yet before the functions
    /// run ([`IcbObject::overrun`]), as this platform's copy would carry the intact record; else
    /// the residency is kept while the GPU may run it, so such a copy in another command buffer
    /// dies at the copy. Where it is for writing (`writes`), the functions may write the records,
    /// for good ([`Sides::kernel_written`]).
    pub(crate) fn resident(
        &self,
        sel: &'static str,
        progress: Option<&Arc<Progress>>,
        writes: bool,
    ) {
        self.reached(sel, None, progress, RESIDENT_OVERRUN);
        if writes {
            plain(|| self.sides.lock().kernel_written = true);
        }
    }

    /// [`IcbObject::executed`] and [`IcbObject::resident`], whose death names `why`.
    fn reached(
        &self,
        sel: &'static str,
        range: Option<Range<u64>>,
        progress: Option<&Arc<Progress>>,
        why: &str,
    ) {
        let overrun = plain(|| {
            let mut sides = self.sides.lock();
            if let Some(progress) = progress {
                sides
                    .executions
                    .retain(|p| p.upgrade().is_some_and(|p| p.running()));
                // Once a command buffer: another execution or residency in it changes nothing.
                let listed = (sides.executions.iter()).any(|p| p.as_ptr() == Arc::as_ptr(progress));
                if !listed {
                    sides.executions.push(Arc::downgrade(progress));
                }
            }
            sides.overrun(range.as_ref())
        });
        if overrun {
            unrepresentable(sel, why);
        }
    }

    /// Dies at `sel`, which copies commands `range` of the buffer, where Metal's copy past the
    /// buffer's end may have written over one of their records, or may yet before the copy runs
    /// ([`IcbObject::overrun`]).
    pub(crate) fn copied_intact(&self, sel: &'static str, range: &Range<u64>) {
        if plain(|| self.sides.lock().overrun(Some(range))) {
            unrepresentable(sel, OVERRUN);
        }
    }
}

/// `id<MTLIndirectCommandBuffer>`.
pub type VulkanIcb = Obj<IcbObject>;

/// Another reference to each indirect command buffer of `list`, which a `useResource` form made
/// resident: its command buffer holds them, and their IDs, until it has run (the docs of
/// [`RESERVING`]).
pub(crate) fn made_resident(list: &[Option<ResourceRef<'_, VulkanPlatform>>]) -> Vec<VulkanIcb> {
    list.iter()
        .flatten()
        .filter_map(|r| match r {
            ResourceRef::Icb(icb) => Some(icb.clone_ref()),
            _ => None,
        })
        .collect()
}

/// A count of a descriptor's, sent with `sel`, as the runtime's `u32`.
fn count(sel: &'static str, value: u64) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| unrepresentable(sel, &format!("a count of {value}")))
}

/// The runtime's descriptor of `desc`, its fields set in the value's order: the object and mesh
/// buffer counts among them, which mesh draws a kernel encodes take, every inheritance bit,
/// Apple's private ones among them (stencil reference values, depth test bounds, scissor
/// rectangles, viewports, blend colour), the scissor rectangle and viewport counts and the
/// render state overrides. Not sent: ray tracing, which the runtime has no form for, and the
/// object threadgroup memory count, which no command core encodes reaches and the runtime reads
/// nowhere, and which the runtime's `u32` might not hold. Core's descriptor has no
/// colour-attachment mapping.
///
/// A render command runs with the encoder's render state, as the runtime's do on a device that
/// claims no Apple10, MoltenVK among them, whatever the bits say (measured: an ICB without
/// `inheritCullMode`, `inheritFrontFacingWinding`, `inheritScissorRects` or `inheritViewports`
/// draws with the encoder's cull mode, winding, scissor rectangle and viewport, as one with them
/// does; the runtime's counts and overrides are kept and read by nothing, as Metal's draws are
/// the encoder's under any). Core's render commands set none of those states, nor object
/// threadgroup lengths.
fn descriptor(desc: &IcbDesc) -> rt::IndirectCommandBufferDescriptor {
    let mut d = rt::IndirectCommandBufferDescriptor::new();
    d.command_types = metal_value("setCommandTypes:", "command types", desc.command_types);
    d.max_vertex_buffer_bind_count = count(
        "setMaxVertexBufferBindCount:",
        desc.max_vertex_buffer_bind_count,
    );
    d.max_fragment_buffer_bind_count = count(
        "setMaxFragmentBufferBindCount:",
        desc.max_fragment_buffer_bind_count,
    );
    d.max_kernel_buffer_bind_count = count(
        "setMaxKernelBufferBindCount:",
        desc.max_kernel_buffer_bind_count,
    );
    d.max_object_buffer_bind_count = count(
        "setMaxObjectBufferBindCount:",
        desc.max_object_buffer_bind_count,
    );
    d.max_mesh_buffer_bind_count = count(
        "setMaxMeshBufferBindCount:",
        desc.max_mesh_buffer_bind_count,
    );
    d.max_kernel_threadgroup_memory_bind_count = count(
        "setMaxKernelThreadgroupMemoryBindCount:",
        desc.max_kernel_threadgroup_memory_bind_count,
    );
    d.inherit_pipeline_state = desc.inherit_pipeline_state;
    d.inherit_buffers = desc.inherit_buffers;
    d.support_dynamic_attribute_stride = desc.support_dynamic_attribute_stride;
    d.max_scissor_rect_count = desc.max_scissor_rect_count;
    d.max_viewport_count = desc.max_viewport_count;
    d.inherit_stencil_reference_values = desc.inherit_stencil_reference_values;
    d.inherit_depth_test_bounds = desc.inherit_depth_test_bounds;
    d.inherit_scissor_rects = desc.inherit_scissor_rects;
    d.inherit_viewports = desc.inherit_viewports;
    d.inherit_blend_color = desc.inherit_blend_color;
    d.allow_override_render_states = desc.allow_override_render_states;
    d.inherit_depth_stencil_state = desc.inherit_depth_stencil_state;
    d.inherit_depth_bias = desc.inherit_depth_bias;
    d.inherit_depth_clip_mode = desc.inherit_depth_clip_mode;
    d.inherit_cull_mode = desc.inherit_cull_mode;
    d.inherit_front_facing_winding = desc.inherit_front_facing_winding;
    d.inherit_triangle_fill_mode = desc.inherit_triangle_fill_mode;
    d
}

/// `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:`: the runtime's buffer, with
/// Metal's numbers (the module docs), placed at the descriptor's resource index when its low 32
/// bits are nonzero. Metal answers no nil here. A placement past Metal's table of IDs dies, as
/// Metal ends its process there; so does every refusal of the runtime's, as Metal crashes at
/// memory it cannot back. A buffer whose own ID is 2^23 or more, which plain allocation hands out
/// past a reservation that large, the runtime makes as Metal does, at every ID below 2^31
/// (`RECORDS_ID_LIMIT`) and any count of commands. The runtime makes what Metal makes of
/// the descriptors its validation stops (no command types, render and compute types together,
/// more than 31 buffers of a stage).
pub(crate) fn new_indirect_command_buffer(
    device: &VulkanDevice,
    desc: &IcbDesc,
    max_count: u64,
    options: u64,
) -> RaiseResult<Option<VulkanIcb>> {
    let sel = "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:";
    let overrun_shape = OverrunShape::of(desc, max_count, options);
    // The pool's lock first, so every death here comes after the wait: Metal's creation waits on
    // it while a reservation spins, a placement past its table of IDs too ([`RESERVING`]). The
    // runtime's buffer is made under it, so no reservation comes between the two.
    let held = plain(|| RESERVING.lock());
    let mut d = descriptor(desc);
    let options = crate::gpu::resource_options(sel, options);
    let placed = desc
        .resource_index
        .map(|index| index & 0xffff_ffff)
        .filter(|&id| id != 0);
    if let Some(id) = placed {
        d.placement = rt::IdPlacement::At(rt::ResourceId(id));
    }
    if let Some(id) = placed
        && id >= ID_LIMIT
    {
        metal_aborts(
            sel,
            &format!("an indirect command buffer placed at ID {id}"),
        );
    }
    let raw = send(sel, Policy::Dies, || {
        device
            .raw
            .new_indirect_command_buffer(&d, max_count, options)
    })?;
    drop(held);
    Ok(Some(Obj::new(IcbObject {
        raw,
        strides: d.support_dynamic_attribute_stride,
        inherits: Inherits {
            pipeline: d.inherit_pipeline_state,
            buffers: d.inherit_buffers,
        },
        sides: Arc::default(),
        overrun_shape,
        readers: Arc::default(),
    })))
}

impl IndirectCommandBuffer<VulkanPlatform> for VulkanIcb {
    /// The runtime's ID of the buffer, Metal's (the module docs).
    fn gpu_resource_id(&self) -> u64 {
        plain(|| self.raw.gpu_resource_id().0)
    }

    /// The runtime's setters of the command, which Metal hands out at any index and of a buffer
    /// of either kind: the runtime's refusal of those dies.
    fn compute_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<VulkanIndirectComputeCommand>> {
        let sel = "indirectComputeCommandAtIndex:";
        let raw = send(sel, Policy::Dies, || {
            self.raw.indirect_compute_command_at(command_index)
        })?;
        Ok(Some(Obj::new(CommandObject {
            icb: self.clone_ref(),
            index: command_index.low_u32(),
            raw,
        })))
    }

    /// As [`compute_command`](Self::compute_command), for a render command.
    fn render_command(
        &self,
        command_index: u64,
    ) -> RaiseResult<Option<VulkanIndirectRenderCommand>> {
        let sel = "indirectRenderCommandAtIndex:";
        let raw = send(sel, Policy::Dies, || {
            self.raw.indirect_render_command_at(command_index)
        })?;
        Ok(Some(Obj::new(CommandObject {
            icb: self.clone_ref(),
            index: command_index.low_u32(),
            raw,
        })))
    }

    /// The runtime's `reset_with_range`, one call for the range, which refuses a range past the
    /// size as Metal's validation does (a death: core passes one inside it); then the commands'
    /// sides as a CPU reset leaves them ([`IcbObject::rewritten_range`]).
    fn reset_with_range(&self, location: u64, length: u64) -> RaiseResult<()> {
        let range = location..location.saturating_add(length);
        send("resetWithRange:", Policy::Dies, || {
            self.raw.reset_with_range(range.clone())
        })?;
        self.rewritten_range(range);
        Ok(())
    }

    /// Metal's `resourceIndex` of an indirect command buffer is its `gpuResourceID` (measured,
    /// placed or not), as the runtime's is.
    fn resource_index(&self) -> u64 {
        plain(|| self.raw.resource_index())
    }

    fn size(&self) -> u64 {
        plain(|| self.raw.size())
    }

    /// The runtime's, Metal's base plus the buffer's address in Metal's range (the module docs).
    /// Of a buffer the runtime made after encoders whose take of Metal's range it does not model
    /// (`rt::IndirectCommandBuffer::identifier_unmeasured`) it is relayed, a death in strict mode.
    fn unique_identifier(&self) -> u64 {
        if plain(|| self.raw.identifier_unmeasured()) {
            inexact(
                "uniqueIdentifier",
                Inexact::IcbIdentifierAfterEncoder,
                || {
                    "the uniqueIdentifier of an indirect command buffer made after encoders whose \
                     take of Metal's range the runtime does not model: it answers the runtime's"
                        .to_owned()
                },
            );
        }
        plain(|| self.raw.unique_identifier())
    }
}

// ---- The commands.

/// One command of an indirect command buffer: the runtime's setters, and the buffer.
#[derive(Debug)]
pub struct CommandObject<C> {
    /// The buffer it is a command of.
    icb: VulkanIcb,
    /// Its index there.
    index: u32,
    /// The runtime's setters.
    raw: C,
}

/// `id<MTLIndirectRenderCommand>`.
pub type VulkanIndirectRenderCommand = Obj<CommandObject<rt::IndirectRenderCommand>>;

/// `id<MTLIndirectComputeCommand>`.
pub type VulkanIndirectComputeCommand = Obj<CommandObject<rt::IndirectComputeCommand>>;

/// A runtime setter's answer, for `sel`: every refusal dies (the module docs).
fn set(sel: &'static str, call: impl FnOnce() -> Result<(), rt::Error>) -> RaiseResult<()> {
    send(sel, Policy::Dies, call)
}

/// A binding index core passes as `NSUInteger`, which the runtime takes as `u32`.
fn slot(sel: &'static str, index: u64) -> u32 {
    u32::try_from(index).unwrap_or_else(|_| unrepresentable(sel, &format!("index {index}")))
}

/// `object`, which core may pass as nil, for `sel`: Metal crashes at a nil one (measured: an
/// indexed draw's index buffer, a patch draw's tessellation factor buffer).
fn some<'a, T>(sel: &'static str, object: Option<&'a T>, what: &str) -> &'a T {
    object.unwrap_or_else(|| unrepresentable(sel, &format!("a nil {what}, which Metal crashes at")))
}

/// The death of a getter of another kind's arguments: Metal reads the record as its own kind,
/// which no record of the runtime's holds, and core asks only for a command's own kind.
fn another_kind(sel: &'static str) -> ! {
    unrepresentable(sel, "Metal's reading of a command of another kind")
}

/// What a pipeline state identifier's getter dies naming where the runtime answers none of it.
const UNTOLD_STATE: &str = "the pipeline state a record names by a resource ID two or more states \
                            held, where the runtime cannot tell which of them the record was \
                            written with";

/// The address an arguments getter answers for a buffer of a record: Metal's for nil.
fn address_or_nil(address: Option<rt::GpuAddress>) -> u64 {
    address.map_or(NIL_BUFFER_ADDRESS, |a| a.0)
}

impl CommandObject<rt::IndirectRenderCommand> {
    /// The command's record, read back, for `sel`.
    fn record(&self, sel: &'static str) -> rt::EncodedRenderCommand {
        self.icb.intact(sel, self.index);
        let i = u64::from(self.index);
        let commands = get(sel, || self.icb.raw.commands(i..i + 1));
        match commands.into_iter().next() {
            Some(rt::IndirectCommand::Render(c)) => c,
            _ => unrepresentable(sel, "a render command the runtime reads as none"),
        }
    }

    /// The command's draw as an arguments getter `sel` answers it: the runtime's, whose primitive
    /// type is the one set but 0 for quads and a quad strip (7 and 8), as Metal's getters answer
    /// of a command set with either, direct or over indices, though its execution draws the type
    /// set (measured: `render_commands_read_back_as_on_metal`,
    /// `private_types_draw_over_indices_and_arguments_as_metals`).
    fn draw(&self, sel: &'static str) -> Option<rt::IndirectDraw> {
        self.icb.intact(sel, self.index);
        get(sel, || self.raw.draw())
    }
}

impl CommandObject<rt::IndirectComputeCommand> {
    /// The place of the command's threadgroup memory length at `index`'s low 32 bits in the
    /// buffer's array of them ([`LENGTHS_A_COMMAND`]).
    fn flat(&self, index: u64) -> u64 {
        u64::from(self.index) * LENGTHS_A_COMMAND + u64::from(index.low_u32())
    }

    /// Where Metal keeps the command's threadgroup memory length at `index`'s low 32 bits
    /// ([`LENGTHS_A_COMMAND`]): the command whose record holds it, or its place in the buffer's
    /// array of them, past the last command's ([`Tail`]).
    fn length_place(&self, index: u64) -> Result<u32, u64> {
        let flat = self.flat(index);
        let command = flat / LENGTHS_A_COMMAND;
        match command < plain(|| self.icb.raw.size()) {
            true => Ok(command.low_u32()),
            false => Err(flat),
        }
    }

    /// The command's record, read back, for `sel`.
    fn record(&self, sel: &'static str) -> rt::EncodedComputeCommand {
        self.icb.intact(sel, self.index);
        let i = u64::from(self.index);
        let commands = get(sel, || self.icb.raw.commands(i..i + 1));
        match commands.into_iter().next() {
            Some(rt::IndirectCommand::Compute(c)) => c,
            _ => unrepresentable(sel, "a compute command the runtime reads as none"),
        }
    }
}

/// A draw's raw primitive type, sent with `sel`: a public `MTLPrimitiveType`, or one of Metal's
/// private types 5 to 8, which the runtime's executions draw as an encoder's draw of the type
/// draws ([`rt::DrawPrimitiveType`]; the module docs). Type 9 dies ([`TYPE_9`]), and so does a
/// type past it, which no `MTLPrimitiveType` is.
fn primitive(sel: &'static str, raw: u64) -> rt::DrawPrimitiveType {
    match raw {
        5..=8 => rt::DrawPrimitiveType::from_raw(raw).unwrap_or_else(|e| refused(sel, &e)),
        9 => unrepresentable(sel, TYPE_9),
        _ => rt::DrawPrimitiveType::Public(metal_value(sel, "primitive type", raw)),
    }
}

/// Why a command's draw of primitive type 9 dies at its setter: the runtime fails the command
/// buffer at one only among a range given at the execution's call, as the records stand then,
/// and nothing here tells the command buffer, whose commit dies where it fails another way too
/// (`crate::gpu::command`); one written after the call, or among a range a buffer holds, runs
/// nothing, where Metal's GPU fails the command buffer.
const TYPE_9: &str = "a draw of primitive type 9 in an indirect command, which Metal's GPU \
     fails the command buffer at as it runs the command: the runtime fails it only at one among \
     a range given at the execution's call, as the records stand then, and runs nothing of one \
     written after the call or among a range a buffer holds";

impl IndirectRenderCommand<VulkanPlatform> for VulkanIndirectRenderCommand {
    /// One runtime setter per command. A nil pipeline state, index buffer or tessellation factor
    /// buffer, which the runtime's setters take none of, dies. A nil buffer the runtime holds at
    /// its offset (the module docs).
    fn encode(&self, cmd: IcbRenderCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        let c = &self.raw;
        match cmd {
            IcbRenderCmd::ClearBarrier => set(sel, || c.clear_barrier()),
            IcbRenderCmd::SetBarrier => set(sel, || c.set_barrier()),
            IcbRenderCmd::DrawIndexedPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                control_point_index_buffer,
                control_point_index_buffer_offset,
                instance_count,
                base_instance,
                buffer,
                offset,
                instance_stride,
            } => {
                let factors = some(sel, buffer, "tessellation factor buffer");
                let patch_index = patch_index_buffer.map(|b| &b.raw);
                set(sel, || match control_point_index_buffer {
                    Some(control_points) => c.draw_indexed_patches(
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index,
                        patch_index_buffer_offset,
                        Some(&control_points.raw),
                        control_point_index_buffer_offset,
                        instance_count,
                        base_instance,
                        &factors.raw,
                        offset,
                        instance_stride,
                    ),
                    None => c.draw_patches(
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index,
                        patch_index_buffer_offset,
                        instance_count,
                        base_instance,
                        &factors.raw,
                        offset,
                        instance_stride,
                    ),
                })
            }
            IcbRenderCmd::DrawIndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                index_buffer_offset,
                instance_count,
                base_vertex,
                base_instance,
            } => {
                let primitive = primitive(sel, primitive_type);
                let index_type: IndexType = metal_value(sel, "index type", index_type);
                let indices = some(sel, index_buffer, "index buffer");
                set(sel, || {
                    c.draw_indexed_primitives(
                        primitive,
                        index_count,
                        index_type,
                        &indices.raw,
                        index_buffer_offset,
                        instance_count,
                        base_vertex,
                        base_instance,
                    )
                })
            }
            IcbRenderCmd::DrawPatches {
                number_of_patch_control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                patch_index_buffer_offset,
                instance_count,
                base_instance,
                buffer,
                offset,
                instance_stride,
            } => {
                let factors = some(sel, buffer, "tessellation factor buffer");
                set(sel, || {
                    c.draw_patches(
                        number_of_patch_control_points,
                        patch_start,
                        patch_count,
                        patch_index_buffer.map(|b| &b.raw),
                        patch_index_buffer_offset,
                        instance_count,
                        base_instance,
                        &factors.raw,
                        offset,
                        instance_stride,
                    )
                })
            }
            IcbRenderCmd::DrawPrimitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            } => {
                let primitive = primitive(sel, primitive_type);
                set(sel, || {
                    c.draw_primitives(
                        primitive,
                        vertex_start,
                        vertex_count,
                        instance_count,
                        base_instance,
                    )
                })
            }
            IcbRenderCmd::Reset => {
                set(sel, || c.reset())?;
                self.icb.rewritten(self.index);
                Ok(())
            }
            IcbRenderCmd::SetFragmentBuffer {
                buffer,
                offset,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    c.set_fragment_buffer(buffer.map(|b| &b.raw), offset, at)
                })?;
                Ok(())
            }
            IcbRenderCmd::SetRenderPipelineState { pipeline_state } => {
                let state = some(sel, pipeline_state, "pipeline state");
                set(sel, || c.set_render_pipeline_state(&state.raw))
            }
            IcbRenderCmd::SetVertexBuffer {
                buffer,
                offset,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    c.set_vertex_buffer(buffer.map(|b| &b.raw), offset, at)
                })?;
                Ok(())
            }
            IcbRenderCmd::SetVertexBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    let raw = buffer.map(|b| &b.raw);
                    c.set_vertex_buffer_with_stride(raw, offset, stride, at)
                })?;
                Ok(())
            }
        }
    }

    /// The record's draw kind, as `MTLIndirectCommandType`; 0 for none: the runtime's.
    fn command_type(&self) -> u64 {
        let sel = "getCommandType";
        self.icb.intact(sel, self.index);
        u64::from(get(sel, || self.raw.command_type()).as_raw())
    }

    fn fragment_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getFragmentBufferAddressAtIndex:";
        Ok(self.record(sel).fragment_buffer_address(index).unwrap_or(0))
    }

    /// The runtime's identifier of the pipeline state the record names, live or released; 0 for
    /// none (the module docs).
    fn pipeline_state_unique_identifier(&self) -> u64 {
        let sel = "getPipelineStateUniqueIdentifier";
        self.icb.intact(sel, self.index);
        let answer = get(sel, || self.raw.pipeline_unique_identifier());
        answer.unwrap_or_else(|| unrepresentable(sel, UNTOLD_STATE))
    }

    fn vertex_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getVertexAttributeStrideAtIndex:";
        let stride = self.record(sel).vertex_attribute_stride(index);
        Ok(stride.unwrap_or_else(|| self.icb.unset_stride()))
    }

    fn vertex_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getVertexBufferAddressAtIndex:";
        Ok(self.record(sel).vertex_buffer_address(index).unwrap_or(0))
    }

    fn has_barrier(&self) -> bool {
        self.record("hasBarrier").barrier
    }

    fn draw_args(&self) -> Option<DrawArgs> {
        let sel = "drawArguments";
        match self.draw(sel) {
            Some(rt::IndirectDraw::Primitives {
                primitive_type,
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            }) => Some(DrawArgs {
                primitive_type: primitive_type.raw(),
                vertex_start,
                vertex_count,
                instance_count,
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    /// Metal's `baseVertex` answers the record's 32 bits, zero-extended (measured: -2 reads
    /// 4294967294).
    fn draw_indexed_args(&self) -> Option<DrawIndexedArgs> {
        let sel = "drawIndexedArguments";
        match self.draw(sel) {
            Some(rt::IndirectDraw::IndexedPrimitives {
                primitive_type,
                index_count,
                index_type,
                index_buffer,
                instance_count,
                base_vertex,
                base_instance,
            }) => Some(DrawIndexedArgs {
                index_buffer_gpu_virtual_address: index_buffer.0,
                primitive_type: primitive_type.raw(),
                index_type: u64::from(index_type.as_raw()),
                index_count,
                instance_count,
                base_vertex: u64::from(base_vertex.cast_unsigned().low_u32()),
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    fn draw_patches_args(&self) -> Option<DrawPatchesArgs> {
        let sel = "drawPatchesArguments";
        match self.draw(sel) {
            Some(rt::IndirectDraw::Patches {
                control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                instance_count,
                base_instance,
                ..
            }) => Some(DrawPatchesArgs {
                patch_index_buffer_gpu_virtual_address: address_or_nil(patch_index_buffer),
                number_of_patch_control_points: control_points,
                patch_start,
                patch_count,
                instance_count,
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    fn draw_indexed_patches_args(&self) -> Option<DrawIndexedPatchesArgs> {
        let sel = "drawIndexedPatchesArguments";
        match self.draw(sel) {
            Some(rt::IndirectDraw::IndexedPatches {
                control_points,
                patch_start,
                patch_count,
                patch_index_buffer,
                control_point_index_buffer,
                instance_count,
                base_instance,
                ..
            }) => Some(DrawIndexedPatchesArgs {
                patch_index_buffer_gpu_virtual_address: address_or_nil(patch_index_buffer),
                control_point_index_buffer_gpu_virtual_address: address_or_nil(
                    control_point_index_buffer,
                ),
                number_of_patch_control_points: control_points,
                patch_start,
                patch_count,
                instance_count,
                base_instance,
            }),
            _ => another_kind(sel),
        }
    }

    /// A patch draw's factors: the buffer's address plus its offset, the instance stride, and
    /// the scale 1, which every patch draw a command's setter encodes holds.
    fn tessellation_factor_args(&self) -> Option<TessellationFactorArgs> {
        let sel = "getTessellationFactorArguments";
        match self.draw(sel) {
            Some(
                rt::IndirectDraw::Patches {
                    factor_buffer,
                    factor_instance_stride,
                    ..
                }
                | rt::IndirectDraw::IndexedPatches {
                    factor_buffer,
                    factor_instance_stride,
                    ..
                },
            ) => Some(TessellationFactorArgs {
                gpu_virtual_address: factor_buffer.0,
                instance_stride: factor_instance_stride,
                scale: 1.0,
            }),
            _ => another_kind(sel),
        }
    }
}

impl IndirectComputeCommand<VulkanPlatform> for VulkanIndirectComputeCommand {
    /// One runtime setter per command. A nil pipeline state, which the runtime's setter takes
    /// none of, dies.
    fn encode(&self, cmd: IcbComputeCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        let c = &self.raw;
        match cmd {
            IcbComputeCmd::ClearBarrier => set(sel, || c.clear_barrier()),
            IcbComputeCmd::ConcurrentDispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => {
                set(sel, || {
                    c.concurrent_dispatch_threadgroups(
                        crate::gpu::size(threadgroups_per_grid),
                        crate::gpu::size(threads_per_threadgroup),
                    )
                })?;
                self.icb.words_set(self.index, Tail::dispatch);
                Ok(())
            }
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => {
                set(sel, || {
                    c.concurrent_dispatch_threads(
                        crate::gpu::size(threads_per_grid),
                        crate::gpu::size(threads_per_threadgroup),
                    )
                })?;
                self.icb.words_set(self.index, Tail::dispatch);
                Ok(())
            }
            IcbComputeCmd::Reset => {
                set(sel, || c.reset())?;
                self.icb.rewritten(self.index);
                Ok(())
            }
            IcbComputeCmd::SetBarrier => set(sel, || c.set_barrier()),
            IcbComputeCmd::SetComputePipelineState { pipeline_state } => {
                let state = some(sel, pipeline_state, "pipeline state");
                set(sel, || c.set_compute_pipeline_state(&state.raw))?;
                self.icb.words_set(self.index, Tail::pipeline);
                Ok(())
            }
            IcbComputeCmd::SetImageblockWidth { width, height } => {
                set(sel, || c.set_imageblock_size(width, height))?;
                self.icb.words_set(self.index, Tail::imageblock);
                Ok(())
            }
            IcbComputeCmd::SetKernelBuffer {
                buffer,
                offset,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    c.set_kernel_buffer(buffer.map(|b| &b.raw), offset, at)
                })?;
                Ok(())
            }
            IcbComputeCmd::SetKernelBufferAttributeStride {
                buffer,
                offset,
                stride,
                index,
            } => {
                let at = slot(sel, index);
                set(sel, || {
                    let raw = buffer.map(|b| &b.raw);
                    c.set_kernel_buffer_with_stride(raw, offset, stride, at)
                })?;
                Ok(())
            }
            IcbComputeCmd::SetStageInRegion { region } => {
                set(sel, || c.set_stage_in_region(crate::gpu::region(region)))
            }
            IcbComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                let length = u32::try_from(length)
                    .unwrap_or_else(|_| unrepresentable(sel, &format!("a length of {length}")));
                set(sel, || {
                    c.set_threadgroup_memory_length(length, index.low_u32())
                })?;
                match self.length_place(index) {
                    Ok(_) => self.icb.record_set(self.flat(index)),
                    Err(flat) => self.icb.tail_set(flat),
                }
                Ok(())
            }
        }
    }

    /// The record's dispatch kind, as `MTLIndirectCommandType`; 0 for none.
    fn command_type(&self) -> u64 {
        let sel = "getCommandType";
        let kind = match self.record(sel).dispatch {
            None => IndirectCommandType::empty(),
            Some(rt::IndirectDispatch::Threadgroups { .. }) => {
                IndirectCommandType::CONCURRENT_DISPATCH
            }
            Some(rt::IndirectDispatch::Threads { .. }) => {
                IndirectCommandType::CONCURRENT_DISPATCH_THREADS
            }
            Some(_) => unrepresentable(sel, "a dispatch of a kind Metal has no type for"),
        };
        u64::from(kind.as_raw())
    }

    fn kernel_attribute_stride(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getKernelAttributeStrideAtIndex:";
        let stride = self.record(sel).kernel_attribute_stride(index);
        Ok(stride.unwrap_or_else(|| self.icb.unset_stride()))
    }

    fn kernel_buffer_address(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getKernelBufferAddressAtIndex:";
        Ok(self.record(sel).kernel_buffer_address(index).unwrap_or(0))
    }

    /// The runtime's identifier of the pipeline state the record names, live or released; 0 for
    /// none, a reset's retained one included, as Metal's getter reports it (the module docs).
    fn pipeline_state_unique_identifier(&self) -> u64 {
        let sel = "getPipelineStateUniqueIdentifier";
        self.icb.intact(sel, self.index);
        let answer = get(sel, || self.raw.pipeline_unique_identifier());
        answer.unwrap_or_else(|| unrepresentable(sel, UNTOLD_STATE))
    }

    fn stage_in_region(&self) -> MtlRegion {
        let r = self.record("getStageInRegion").stage_in_region;
        MtlRegion::new(
            MtlOrigin::new(r.origin.x, r.origin.y, r.origin.z),
            MtlSize::new(r.size.width, r.size.height, r.size.depth),
        )
    }

    /// The runtime's length where Metal keeps it, at the index's low 32 bits, as Metal's getter
    /// takes them: the command's below 38, a later command's at 38 or more
    /// ([`LENGTHS_A_COMMAND`]), past the last command's too. A command whose record Metal's copy
    /// past the buffer's end may have written over dies ([`IcbObject::overrun`]); one past the
    /// last command's that Metal wrote otherwise than the runtime, or may have, is relayed
    /// ([`Tail`]).
    fn threadgroup_memory_length(&self, index: u64) -> RaiseResult<u64> {
        let sel = "getThreadgroupMemoryLengthAtIndex:";
        match self.length_place(index) {
            Ok(command) => {
                self.icb.intact(sel, command);
                self.icb.record_read(sel, self.flat(index));
            }
            Err(flat) => self.icb.tail_read(sel, flat),
        }
        let length = get(sel, || self.raw.threadgroup_memory_length(index));
        let length = length.unwrap_or_else(|| {
            unrepresentable(
                sel,
                "a threadgroup memory length the runtime answers none of",
            )
        });
        Ok(u64::from(length))
    }

    fn has_barrier(&self) -> bool {
        self.record("hasBarrier").barrier
    }

    fn dispatch_threadgroups_args(&self) -> Option<DispatchThreadgroupsArgs> {
        let sel = "dispatchThreadgroupsArguments";
        match self.record(sel).dispatch {
            Some(rt::IndirectDispatch::Threadgroups {
                threadgroups_per_grid: g,
                threads_per_threadgroup: t,
            }) => Some(DispatchThreadgroupsArgs {
                threadgroups_per_grid: MtlSize::new(g.width, g.height, g.depth),
                threads_per_threadgroup: MtlSize::new(t.width, t.height, t.depth),
            }),
            _ => another_kind(sel),
        }
    }

    fn dispatch_threads_args(&self) -> Option<DispatchThreadsArgs> {
        let sel = "dispatchThreadsArguments";
        match self.record(sel).dispatch {
            Some(rt::IndirectDispatch::Threads {
                threads_per_grid: g,
                threads_per_threadgroup: t,
            }) => Some(DispatchThreadsArgs {
                threads_per_grid: MtlSize::new(g.width, g.height, g.depth),
                threads_per_threadgroup: MtlSize::new(t.width, t.height, t.depth),
            }),
            _ => another_kind(sel),
        }
    }

    fn image_block_args(&self) -> Option<ImageBlockArgs> {
        let c = self.record("getImageBlockSize");
        Some(ImageBlockArgs {
            width: c.imageblock_width,
            height: c.imageblock_height,
        })
    }
}

/// The tests of identifiers need a process each, as `cargo nextest` (the repository's runner,
/// `cargo xtask check`) gives them: what a record names is decided by process-wide state (the
/// runtime's holders of each resource ID, and its pools of IDs) over the one runtime device of
/// the process, so under `cargo test`'s threads one test's states and releases reach another's.
#[cfg(test)]
mod tests {
    use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _, ResourceRef};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::{Buffer as _, ComputePipeline as _};
    use reims_vgpu_core::platform::types::NsRange;
    use reims_vgpu_core::raise::OnFault;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::command::VulkanCommandBuffer;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::{VulkanBuffer, VulkanComputePipeline};

    const ADD: &[u8] =
        include_bytes!("../../../../tests/suite/fixtures/shaders/compute_add.metallib");

    /// `MTLCommandBufferStatusCompleted`.
    const COMPLETED: u64 = 4;

    /// A descriptor of both dispatch kinds, two kernel buffers with dynamic strides and two
    /// threadgroup lengths.
    fn compute_desc() -> IcbDesc {
        IcbDesc {
            command_types: 0x60,
            max_kernel_buffer_bind_count: 2,
            max_kernel_threadgroup_memory_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        }
    }

    /// A descriptor of the four draw kinds, two vertex buffers with dynamic strides and two
    /// fragment buffers.
    fn render_desc() -> IcbDesc {
        IcbDesc {
            command_types: 0xf,
            max_vertex_buffer_bind_count: 2,
            max_fragment_buffer_bind_count: 2,
            support_dynamic_attribute_stride: true,
            ..IcbDesc::default()
        }
    }

    /// A buffer of `n` commands of `desc`, Shared.
    fn make<P: Platform>(d: &P::Device, desc: &IcbDesc, n: u64) -> P::Icb {
        d.new_indirect_command_buffer(desc, n, 0).unwrap().unwrap()
    }

    /// The `uniqueIdentifier` of a buffer made after a command buffer that encoded a render
    /// encoder was scheduled, whose pass's take of Metal's range the runtime does not model
    /// (`identifier_unmeasured`), is relayed, answering the runtime's, and dies in strict mode,
    /// naming `inexact-icb-identifier-after-encoder`, its creation and its other numbers gone on;
    /// one made before it answers Metal's first, `0x1000018000`, with no relay, then and after.
    /// Where the runtime models the encoders' take, its number is Metal's
    /// (`metal_tests::identifiers_after_are_metals`)
    #[test]
    fn identifiers_after_a_scheduled_render_encoder_are_relayed() {
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        let d = dev();
        let before = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        assert_eq!(before.unique_identifier(), 0x10_0001_8000);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let p = crate::platform::tests::platform();
        let (_target, pass) = crate::gpu::encode_render::tests::pass_of(&p, &d, (4, 4));
        let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
        e.encode(OnFault::Return, RenderCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let after = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        assert_eq!(before.unique_identifier(), 0x10_0001_8000);
        assert_eq!((after.gpu_resource_id(), after.resource_index()), (2, 2));
        *crate::answer::LAST_LINE.lock() = None;
        let site = catch_death(|| after.unique_identifier()).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "uniqueIdentifier")
        );
        let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
        let kind = crate::answer::Inexact::IcbIdentifierAfterEncoder;
        assert!(line.contains(kind.code()), "{line}");
        let relayed = crate::answer::relaxed(|| after.unique_identifier());
        assert_eq!(relayed, plain(|| after.raw.unique_identifier()));
        assert_eq!(crate::answer::inexact_counts(), [(kind, 1)]);
    }

    /// An isolated task that ends while its reservation is under way, once it holds
    /// [`RESERVING`], still gives it to the runtime, which then hands out Metal's IDs to the
    /// process's later buffers: 1 then 3, 2 reserved
    #[test]
    fn a_task_that_ends_amid_its_reservation_leaves_the_runtimes_ids_metals() {
        use reims_vgpu_core::tasks::isolation::life::TaskLife;

        let d = dev();
        let life = TaskLife::new();
        AMID_RESERVATION.set(Some(|| {
            assert!(TaskLife::current().unwrap().exit());
        }));
        let reserved = life.run(|| d.reserve_resource_indices(5, &[2]));
        assert!(matches!(reserved, Some(Ok(()))), "{reserved:?}");
        let made: Vec<VulkanIcb> = (0..2)
            .map(|_| make::<VulkanPlatform>(&d, &compute_desc(), 1))
            .collect();
        let seen: Vec<u64> = made.iter().map(|b| b.gpu_resource_id()).collect();
        assert_eq!(seen, [1, 3]);
    }

    /// A device the host releases and opens again is the one runtime device, whose IDs go on with
    /// these, as Metal's device lives for the process: 2 reserved on a handle dropped after, the
    /// device opened again hands out 1 then 3, where a runtime device made again would hand out 1
    /// then 2. It holds no other handle ([`dev`]) before the drop.
    #[test]
    fn a_device_opened_again_keeps_its_ids() {
        let open = || VulkanPlatform::system_default_device().expect("a Vulkan device (MoltenVK)");
        let first = open();
        first.reserve_resource_indices(5, &[2]).unwrap();
        drop(first);
        let d = open();
        let made: Vec<VulkanIcb> = (0..2)
            .map(|_| make::<VulkanPlatform>(&d, &compute_desc(), 1))
            .collect();
        let seen: Vec<u64> = made.iter().map(|b| b.gpu_resource_id()).collect();
        assert_eq!(seen, [1, 3]);
    }

    /// The runtime refuses a reservation Metal spins at, of an ID of 2^31 to 2^32 - 1, as a hang
    /// (`Problem::ReservationHangs`), reserving nothing (plain IDs from 1), and at that refusal
    /// the thread parks for good holding the lock it is given, as Metal's spinning thread holds
    /// its pool's ([`RESERVING`]), so a creation or another reservation would wait on it for good:
    /// here the lock stays held past 500 ms. One the runtime takes lets the lock go. The test
    /// leaves the parked thread behind, holding a lock of its own (leaked, not the process's),
    /// until the test's process exits
    #[test]
    fn a_reservation_the_runtime_says_metal_spins_at_holds_the_pools_lock() {
        use std::time::Duration;

        const SEL: &str = "reserveResourceIndicesForResourceType:indices:indexCount:";
        let d = dev();
        let ask = |id: u64| {
            let ids = [rt::ResourceId(id)];
            d.raw
                .reserve_resource_ids(rt::ResourceIdKind::IndirectCommandBuffer, &ids)
        };
        let refusal = ask(u64::from(u32::MAX)).unwrap_err();
        assert!(spins(&refusal), "{refusal:?}");
        let refusal = ask((1 << 31) + 5).unwrap_err();
        assert!(spins(&refusal), "{refusal:?}");
        let made = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        assert_eq!(made.gpu_resource_id(), 1);
        let lock = || -> &'static Mutex<()> { Box::leak(Box::new(Mutex::new(()))) };
        let taken = lock();
        answered(SEL, taken.lock(), Ok(())).unwrap();
        assert!(!taken.is_locked());
        let spun = lock();
        std::thread::Builder::new()
            .name("spun reservation".into())
            .spawn(move || {
                // Parks for good: it never answers.
                let _never = answered(SEL, spun.lock(), Err(refusal));
            })
            .unwrap();
        // Up to 10 s for the thread to take the lock, which it then holds for good.
        let held = (0..10_000).any(|_| {
            std::thread::sleep(Duration::from_millis(1));
            spun.is_locked()
        });
        assert!(held, "never held");
        assert!(spun.try_lock_for(Duration::from_millis(500)).is_none());
    }

    /// The runtime's pipeline state of `add_one`, with indirect command buffer support, as core's
    /// handle.
    fn icb_pipeline(d: &VulkanDevice) -> VulkanComputePipeline {
        let library = d.raw.new_library_with_data(ADD).unwrap();
        let function = library.new_function_with_name("add_one").unwrap();
        let mut desc =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        desc.support_indirect_command_buffers = true;
        let raw = d
            .raw
            .new_compute_pipeline_state(&desc, &rt::PipelineOptions::default())
            .unwrap();
        Obj::new(crate::gpu::resource::ComputePipelineObject::new(raw))
    }

    /// A Shared buffer holding `w`.
    fn holding(d: &VulkanDevice, w: &[u32]) -> VulkanBuffer {
        let b = d
            .new_buffer_of_length(OnFault::Return, u64::try_from(w.len() * 4).unwrap(), 0)
            .unwrap()
            .unwrap();
        // SAFETY: the buffer's Shared bytes, as long as `w`, written before any command uses them.
        unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), b.contents().as_ptr::<u32>(), w.len()) };
        b
    }

    /// The words of a Shared buffer.
    fn words(b: &VulkanBuffer) -> Vec<u32> {
        let n = usize::try_from(b.length() / 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u32>(), n) }.to_vec()
    }

    /// A compute command names the pipeline state core set by its handle's identifier, 0 after
    /// a reset; a blit's copy carries the command, its pipeline and barrier, to another buffer,
    /// whose execution runs it; a blit's reset leaves a fresh command
    #[test]
    fn commands_run_copy_and_reset() {
        let d = dev();
        let pipeline = icb_pipeline(&d);
        let desc = IcbDesc {
            command_types: 0x40,
            max_kernel_buffer_bind_count: 2,
            ..IcbDesc::default()
        };
        let (a, b) = (
            make::<VulkanPlatform>(&d, &desc, 1),
            make::<VulkanPlatform>(&d, &desc, 2),
        );
        let input = holding(&d, &[10, 20, 30, 40]);
        let out = holding(&d, &[0; 4]);
        let c = a.compute_command(0).unwrap().unwrap();
        for cmd in [
            IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&out),
                offset: 0,
                index: 0,
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&input),
                offset: 0,
                index: 1,
            },
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
            IcbComputeCmd::SetBarrier,
        ] {
            c.encode(cmd).unwrap();
        }
        assert_eq!(
            c.pipeline_state_unique_identifier(),
            pipeline.unique_identifier()
        );
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyIndirectCommandBuffer {
                source: &a,
                source_range: NsRange::new(0, 1),
                destination: &b,
                destination_index: 1,
            },
            BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer: &b,
                range: NsRange::new(0, 2),
            },
            BlitCmd::ResetCommandsInBuffer {
                buffer: &a,
                range: NsRange::new(0, 1),
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Return, cmd).unwrap();
        }
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            ComputeCmd::UseResources {
                resources: &[
                    Some(ResourceRef::Buffer(&input)),
                    Some(ResourceRef::Buffer(&out)),
                    Some(ResourceRef::Icb(&b)),
                ],
                usage: 3,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &b,
                execution_range: NsRange::new(1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
        assert_eq!(words(&out), [11, 21, 31, 41]);
        let copied = b.compute_command(1).unwrap().unwrap();
        assert_eq!(
            (
                copied.command_type(),
                copied.pipeline_state_unique_identifier(),
                copied.has_barrier()
            ),
            (0x40, pipeline.unique_identifier(), true)
        );
        let reset = a.compute_command(0).unwrap().unwrap();
        assert_eq!(
            (
                reset.command_type(),
                reset.pipeline_state_unique_identifier(),
                reset.has_barrier()
            ),
            (0, 0, false)
        );
        c.encode(IcbComputeCmd::SetComputePipelineState {
            pipeline_state: Some(&pipeline),
        })
        .unwrap();
        c.encode(IcbComputeCmd::Reset).unwrap();
        assert_eq!(c.pipeline_state_unique_identifier(), 0);
    }

    /// Executes command 0 of `icb`, `resident` made resident for reading and writing, in a
    /// command buffer of its own, which completes with no error.
    fn execute_first(
        d: &VulkanDevice,
        icb: &VulkanIcb,
        resident: &[Option<ResourceRef<'_, VulkanPlatform>>],
    ) {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            ComputeCmd::UseResources {
                resources: resident,
                usage: 3,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: icb,
                execution_range: NsRange::new(0, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    }

    /// Command 0 of a buffer of `size` dispatches `add_one` of `input` into `out`.
    fn add_into(
        d: &VulkanDevice,
        size: u64,
        out: &VulkanBuffer,
        input: &VulkanBuffer,
    ) -> (VulkanIcb, VulkanComputePipeline) {
        let pipeline = icb_pipeline(d);
        let desc = IcbDesc {
            command_types: 0x40,
            max_kernel_buffer_bind_count: 2,
            ..IcbDesc::default()
        };
        let icb = make::<VulkanPlatform>(d, &desc, size);
        let c = icb.compute_command(0).unwrap().unwrap();
        for cmd in [
            IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&pipeline),
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(out),
                offset: 0,
                index: 0,
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(input),
                offset: 0,
                index: 1,
            },
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
        ] {
            c.encode(cmd).unwrap();
        }
        (icb, pipeline)
    }

    /// A CPU reset of a range, one call however many commands it spans, leaves each command of it
    /// fresh, as a command's reset leaves one: of 2^20 commands, command 0 dispatches nothing
    /// after it, and its getters read a fresh command's
    #[test]
    fn a_cpu_reset_of_a_range_leaves_its_commands_fresh() {
        let d = dev();
        let input = holding(&d, &[10, 20, 30, 40]);
        let out = holding(&d, &[0; 4]);
        let (icb, _pipeline) = add_into(&d, 1 << 20, &out, &input);
        let resident = [
            Some(ResourceRef::Buffer(&input)),
            Some(ResourceRef::Buffer(&out)),
            Some(ResourceRef::Icb(&icb)),
        ];
        execute_first(&d, &icb, &resident);
        assert_eq!(words(&out), [11, 21, 31, 41]);
        icb.reset_with_range(0, 1 << 20).unwrap();
        let reset = icb.compute_command(0).unwrap().unwrap();
        assert_eq!(
            (
                reset.command_type(),
                reset.pipeline_state_unique_identifier(),
                reset.has_barrier()
            ),
            (0, 0, false)
        );
        let out_again = holding(&d, &[0; 4]);
        let resident = [
            Some(ResourceRef::Buffer(&input)),
            Some(ResourceRef::Buffer(&out_again)),
            Some(ResourceRef::Icb(&icb)),
        ];
        execute_first(&d, &icb, &resident);
        assert_eq!(words(&out_again), [0; 4]);
    }

    /// The tests' kernels that encode indirect commands, which `build.rs` compiled from
    /// `icb_encode.metal`.
    const ICB_ENCODE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/icb_encode.metallib"));

    /// Two states of `nothing` and two of `a_vertex` and `a_colour`, each made from a guest
    /// descriptor that supports indirect command buffers (compute key 7, render key 25).
    fn twin_states<P: Platform>(
        d: &P::Device,
        library: &P::Library,
    ) -> (Vec<P::ComputePipeline>, Vec<P::RenderPipeline>) {
        use crate::gpu::serialized::tests::{hex, pipeline_blob};
        use reims_vgpu_core::platform::resource::Library as _;
        use reims_vgpu_core::rt::Hold;
        let function = |name: &str| {
            library
                .new_function(OnFault::Return, name)
                .unwrap()
                .unwrap()
        };
        let (kernel, vertex, fragment) = (
            function("nothing"),
            function("a_vertex"),
            function("a_colour"),
        );
        let compute = hex("02 07 04 01000000 00 04 05000000");
        // The basic blob with one more entry, key 25 set, before its three: the colour
        // attachments' block, at the offset its key 8 holds (byte 3 on), moves past it.
        let mut render = pipeline_blob("basic", 0x21, 0x22, 0);
        render[0] += 1;
        render[3] += 6;
        render.splice(1..1, [25, 4, 1, 0, 0, 0]);
        let computes = (0..2)
            .map(|_| {
                let mut resolve = |_: Hold<'_>, r: u64| Ok((r == 5).then(|| kernel.clone_ref()));
                let desc = d
                    .deserialize_compute_pipeline_descriptor(Hold::NONE, &compute, &mut resolve)
                    .unwrap()
                    .unwrap();
                d.new_compute_pipeline_state(Some(&desc), 0)
                    .unwrap()
                    .unwrap()
                    .unwrap()
            })
            .collect();
        let renders = (0..2)
            .map(|_| {
                let mut resolve = |_: Hold<'_>, r: u64| {
                    Ok(match r {
                        0x21 => Some(vertex.clone_ref()),
                        0x22 => Some(fragment.clone_ref()),
                        _ => None,
                    })
                };
                let desc = d
                    .deserialize_render_pipeline_descriptor(Hold::NONE, &render, &mut resolve)
                    .unwrap()
                    .unwrap();
                d.new_render_pipeline_state_from_descriptor(Some(&desc), 0)
                    .unwrap()
                    .unwrap()
                    .unwrap()
            })
            .collect();
        (computes, renders)
    }

    /// Runs the kernel `name` of `library` once on `d`, its argument buffer holding `icb`'s ID
    /// and `state`, a pipeline state's ID, which it encodes `icb`'s command 0 with.
    fn encode_by_kernel<P: Platform>(
        d: &P::Device,
        library: &P::Library,
        name: &str,
        icb: &P::Icb,
        state: u64,
    ) {
        use reims_vgpu_core::platform::resource::Library as _;
        let f = library
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap();
        let kernel = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let held = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let words = [icb.gpu_resource_id(), state];
        // SAFETY: the buffer's 16 Shared bytes, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), held.contents().as_ptr::<u64>(), 2)
        };
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let buffers = [Some(&held)];
        let resident = [Some(ResourceRef::Icb(icb))];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&kernel),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::UseResources {
                resources: &resident,
                usage: 3,
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    }

    /// A command a kernel may have encoded after a state made later took the resource ID of the
    /// one its argument buffer held, released while the dispatch waited to be committed, answers
    /// the identifier both held: the later state took the released one's identifier too, as
    /// Metal's does, so the states that held the ID while the GPU may have written the record
    /// answer one identifier
    #[test]
    fn a_command_written_while_two_states_held_its_id_answers_their_identifier() {
        use reims_vgpu_core::platform::resource::{ComputePipeline as _, Library as _};
        let d = dev();
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let (mut computes, renders) = twin_states::<VulkanPlatform>(&d, &library);
        let icb = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let id = computes[1].gpu_resource_id();
        let f = library
            .new_function(OnFault::Return, "set_compute_state")
            .unwrap()
            .unwrap();
        let kernel = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let held = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let words = [icb.gpu_resource_id(), id];
        // SAFETY: the buffer's 16 Shared bytes, written before any command uses them.
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), held.contents().as_ptr::<u64>(), 2)
        };
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let buffers = [Some(&held)];
        let resident = [Some(ResourceRef::Icb(&icb))];
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&kernel),
            },
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::UseResources {
                resources: &resident,
                usage: 3,
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(1, 1, 1),
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        let released = computes[1].unique_identifier();
        computes.truncate(1);
        let (later, later_renders) = twin_states::<VulkanPlatform>(&d, &library);
        let holder = later.iter().find(|state| state.gpu_resource_id() == id);
        assert_eq!(
            holder.map(|state| state.unique_identifier()),
            Some(released)
        );
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let answer = icb.compute_command(0).unwrap().unwrap();
        assert_eq!(answer.pipeline_state_unique_identifier(), released);
        drop((renders, later_renders));
    }

    /// A state of `mark` of `library`, made from a guest descriptor that supports indirect
    /// command buffers (compute key 7).
    fn marking_state<P: Platform>(d: &P::Device, library: &P::Library) -> P::ComputePipeline {
        use crate::gpu::serialized::tests::hex;
        use reims_vgpu_core::platform::resource::Library as _;
        use reims_vgpu_core::rt::Hold;
        let mark = library
            .new_function(OnFault::Return, "mark")
            .unwrap()
            .unwrap();
        let compute = hex("02 07 04 01000000 00 04 05000000");
        let mut resolve = |_: Hold<'_>, r: u64| Ok((r == 5).then(|| mark.clone_ref()));
        let desc = d
            .deserialize_compute_pipeline_descriptor(Hold::NONE, &compute, &mut resolve)
            .unwrap()
            .unwrap();
        d.new_compute_pipeline_state(Some(&desc), 0)
            .unwrap()
            .unwrap()
            .unwrap()
    }

    /// Commits, on `d`, an execution of `icb`'s command 0, which writes `out`, and waits for it.
    fn execute_writing<P: Platform>(d: &P::Device, icb: &P::Icb, out: &P::Buffer) {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Icb(icb))],
                usage: 1,
            },
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(out))],
                usage: 3,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: icb,
                execution_range: NsRange::new(0, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    }

    /// What `out`'s first word reads on `d` after an execution of a command a kernel encoded with
    /// a state of `mark` the program released since, whose ID nothing took: 1 where the
    /// execution runs the released state's program.
    pub(super) fn released_state_marks<P: Platform>(d: &P::Device) -> u32 {
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let state = marking_state::<P>(d, &library);
        let out = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let icb = make::<P>(d, &compute_desc(), 1);
        let command = icb.compute_command(0).unwrap().unwrap();
        command
            .encode(IcbComputeCmd::SetKernelBuffer {
                buffer: Some(&out),
                offset: 0,
                index: 0,
            })
            .unwrap();
        let id = state.gpu_resource_id();
        encode_by_kernel::<P>(d, &library, "set_compute_state", &icb, id);
        drop(state);
        execute_writing::<P>(d, &icb, &out);
        // SAFETY: the buffer's first Shared word, read after the execution completed.
        unsafe { *out.contents().as_ptr::<u32>() }
    }

    /// An execution of a command a kernel encoded with a state the program released since, whose
    /// ID nothing took, runs the released state's program: the runtime keeps the program of a
    /// state of indirect command buffer support released after a command buffer that may write
    /// the buffer's records was encoded, and announces nothing (`ReleasedCommandState` covers
    /// what it does not keep, [`crate::answer::DIES_BY_DEFAULT`]). Metal's is the same
    /// (`metal_tests`)
    #[test]
    fn an_execution_of_a_record_the_gpu_wrote_naming_a_released_state_runs_it() {
        assert_eq!(released_state_marks::<VulkanPlatform>(&dev()), 1);
    }

    /// A state the guest released while a command core set it on still names it goes, as Metal's
    /// does: a state made after it takes its identifier, as Metal's does (measured), which the
    /// guest reads (info 460), and so does the command's getter, by whose answer core's map
    /// reaches the later state, as Metal's does
    #[test]
    fn a_state_a_command_names_goes_at_the_guests_release() {
        use reims_vgpu_core::platform::resource::ComputePipeline as _;
        let d = dev();
        let library = d.new_library(ICB_ENCODE).unwrap().unwrap().unwrap();
        let (mut computes, renders) = twin_states::<VulkanPlatform>(&d, &library);
        let icb = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let command = icb.compute_command(0).unwrap().unwrap();
        command
            .encode(IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(&computes[1]),
            })
            .unwrap();
        let released = computes[1].unique_identifier();
        computes.truncate(1);
        let (later, later_renders) = twin_states::<VulkanPlatform>(&d, &library);
        drop((renders, later_renders));
        let read = crate::answer::relaxed(|| later[0].guest_unique_identifier());
        assert_eq!(read, released);
        assert_eq!(command.pipeline_state_unique_identifier(), released);
    }

    /// A command's draw of primitive type 9, direct or over indices, dies at its setter
    /// ([`TYPE_9`]), and so does one of a type past it, which no `MTLPrimitiveType` is; types 5
    /// to 8 are set (`private_types_draw_over_indices_and_arguments_as_metals` draws them)
    #[test]
    fn a_command_of_type_9_or_past_it_dies_at_its_setter() {
        let d = dev();
        let icb = make::<VulkanPlatform>(&d, &render_desc(), 1);
        let indices = d
            .new_buffer_of_length(OnFault::Return, 16, 0)
            .unwrap()
            .unwrap();
        let c = icb.render_command(0).unwrap().unwrap();
        let direct = |primitive_type| IcbRenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
            base_instance: 0,
        };
        let indexed = |primitive_type| IcbRenderCmd::DrawIndexedPrimitives {
            primitive_type,
            index_count: 3,
            index_type: 0,
            index_buffer: Some(&indices),
            index_buffer_offset: 0,
            instance_count: 1,
            base_vertex: 0,
            base_instance: 0,
        };
        for primitive_type in 5..=8 {
            c.encode(direct(primitive_type)).unwrap();
            c.encode(indexed(primitive_type)).unwrap();
        }
        let sels = [
            "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
            "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
             instanceCount:baseVertex:baseInstance:",
        ];
        for primitive_type in [9, 10] {
            for (cmd, sel) in [direct(primitive_type), indexed(primitive_type)]
                .into_iter()
                .zip(sels)
            {
                let site = catch_death(|| c.encode(cmd)).unwrap_err();
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
            }
        }
    }

    /// An execution range past the buffer's size, which Metal runs without validation, dies at
    /// its selector, naming the runtime's refusal
    #[test]
    fn an_execution_past_the_size_dies() {
        let d = dev();
        let icb = make::<VulkanPlatform>(&d, &compute_desc(), 1);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let site = catch_death(|| {
            e.encode(
                OnFault::Return,
                ComputeCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: &icb,
                    execution_range: NsRange::new(0, 2),
                },
            )
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "executeCommandsInBuffer:withRange:")
        );
    }

    /// The copy's selector.
    const COPY: &str = "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:";

    /// Buffers of `n` threads dispatches each on `d`, of 10 threads up, 20 up and 30 up (draws of
    /// that many vertices where `render`): a source, a destination and a neighbour made after it.
    fn three_buffers<P: Platform>(d: &P::Device, render: bool, n: u64) -> [P::Icb; 3] {
        let desc = if render {
            render_desc()
        } else {
            compute_desc()
        };
        let buffers = [(); 3].map(|()| make::<P>(d, &desc, n));
        for (icb, base) in buffers.iter().zip([10, 20, 30]) {
            for i in 0..n {
                if render {
                    let draw = IcbRenderCmd::DrawPrimitives {
                        primitive_type: 3,
                        vertex_start: 0,
                        vertex_count: base + i,
                        instance_count: 1,
                        base_instance: 0,
                    };
                    icb.render_command(i)
                        .unwrap()
                        .unwrap()
                        .encode(draw)
                        .unwrap();
                } else {
                    let dispatch = IcbComputeCmd::ConcurrentDispatchThreads {
                        threads_per_grid: MtlSize::new(base + i, 1, 1),
                        threads_per_threadgroup: MtlSize::new(1, 1, 1),
                    };
                    icb.compute_command(i)
                        .unwrap()
                        .unwrap()
                        .encode(dispatch)
                        .unwrap();
                }
            }
        }
        buffers
    }

    /// Command `i` of `icb` as its type and width: a dispatch's threads or threadgroups across, a
    /// draw's vertex count, and `u64::MAX` for a command of no draw or dispatch (type 0, a reset
    /// one), whose arguments core never asks for (a getter of another kind's dies here).
    fn kind_and_width<P: Platform>(icb: &P::Icb, render: bool, i: u64) -> (u64, u64) {
        if render {
            let c = icb.render_command(i).unwrap().unwrap();
            let width = match c.command_type() {
                0 => None,
                _ => c.draw_args().map(|a| a.vertex_count),
            };
            (c.command_type(), width.unwrap_or(u64::MAX))
        } else {
            let c = icb.compute_command(i).unwrap().unwrap();
            let width = match c.command_type() {
                0 => None,
                0x40 => c.dispatch_threads_args().map(|a| a.threads_per_grid.width),
                _ => c
                    .dispatch_threadgroups_args()
                    .map(|a| a.threadgroups_per_grid.width),
            };
            (c.command_type(), width.unwrap_or(u64::MAX))
        }
    }

    /// A buffer an encoder made resident for writing, whose functions' encodings Metal's GPU may
    /// write the words past the last command's lengths of, relays its getter of every such word,
    /// a death in strict mode, though core set it again since; one made resident for reading
    /// alone does not, nor does a length past the words (`Tail`)
    #[test]
    fn a_buffer_made_resident_for_writing_relays_its_words() {
        let d = dev();
        let (read, written) = (
            make::<VulkanPlatform>(&d, &compute_desc(), 2),
            make::<VulkanPlatform>(&d, &compute_desc(), 2),
        );
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for (icb, usage) in [(&read, 1), (&written, 2)] {
            let resources = [Some(ResourceRef::Icb(icb))];
            let cmd = ComputeCmd::UseResources {
                resources: &resources,
                usage,
            };
            e.encode(OnFault::Return, cmd).unwrap();
        }
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let command = |icb: &VulkanIcb| icb.compute_command(1).unwrap().unwrap();
        for icb in [&read, &written] {
            for index in [38, 58, 62] {
                let length = IcbComputeCmd::SetThreadgroupMemoryLength { length: 16, index };
                command(icb).encode(length).unwrap();
            }
        }
        let length = |icb: &VulkanIcb, index| command(icb).threadgroup_memory_length(index);
        for index in [38, 58, 62] {
            assert_eq!(length(&read, index).unwrap(), 16, "read {index}");
        }
        assert_eq!(length(&written, 62).unwrap(), 16);
        for index in [38, 58] {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| length(&written, index)).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            let sel = "getThreadgroupMemoryLengthAtIndex:";
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{line}");
            assert!(
                line.contains(Inexact::IcbLengthPastTheLastCommand.code()),
                "{line}"
            );
            let relayed = crate::answer::relaxed(|| length(&written, index));
            assert_eq!(relayed.unwrap(), 16, "{index}");
        }
    }

    /// `blit` of `cmds` on a new command buffer of `d`'s, committed and run.
    fn blits<P: Platform>(d: &P::Device, cmds: Vec<BlitCmd<'_, P>>) -> P::CommandBuffer {
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in cmds {
            blit.encode(OnFault::Return, cmd).unwrap();
        }
        blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        cb
    }

    /// `k` commands of `source` copied to index `to` of `destination`.
    fn copy_to<'a, P: Platform>(
        source: &'a P::Icb,
        k: u64,
        destination: &'a P::Icb,
        to: u64,
    ) -> BlitCmd<'a, P> {
        BlitCmd::CopyIndirectCommandBuffer {
            source,
            source_range: NsRange::new(0, k),
            destination,
            destination_index: to,
        }
    }

    /// What a copy past the end may have written over stays so until its whole record is
    /// written again: an execution or a copy that reaches it dies, and an execution of the
    /// commands that landed does not; a reset of the command on the CPU, a blit's reset of it and
    /// a blit's copy into it make it answer again. Until its command buffer commits, an execution
    /// of one, which Metal runs after the copy, dies; and a copy past the end while another
    /// command buffer that has not run yet executes the buffer dies at the copy
    #[test]
    fn what_a_copy_past_the_end_overwrote_dies_until_written_again() {
        const EXECUTE: &str = "executeCommandsInBuffer:withRange:";
        let d = dev();
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 4);
        blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 3)]);
        let q = d.new_command_queue().unwrap();
        let executes = |range: NsRange| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            let execution = ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &b,
                execution_range: range,
            };
            e.encode(OnFault::Return, execution).unwrap();
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        };
        for range in [NsRange::new(0, 1), NsRange::new(2, 2)] {
            let site = catch_death(|| executes(range)).unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, EXECUTE),
                "{range:?}"
            );
        }
        executes(NsRange::new(3, 1));
        let other = make::<VulkanPlatform>(&d, &compute_desc(), 4);
        let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![copy_to(&b, 1, &other, 0)]));
        assert_eq!(site.unwrap_err().file, COPY);
        // Each way the whole record is written again.
        let command = b.compute_command(0).unwrap().unwrap();
        command.encode(IcbComputeCmd::Reset).unwrap();
        assert_eq!(command.command_type(), 0);
        let reset = BlitCmd::ResetCommandsInBuffer {
            buffer: &b,
            range: NsRange::new(1, 1),
        };
        let copied = BlitCmd::CopyIndirectCommandBuffer {
            source: &a,
            source_range: NsRange::new(3, 1),
            destination: &b,
            destination_index: 2,
        };
        blits::<VulkanPlatform>(&d, vec![reset, copied]);
        assert_eq!(kind_and_width::<VulkanPlatform>(&b, false, 1).0, 0);
        assert_eq!(kind_and_width::<VulkanPlatform>(&b, false, 2), (0x40, 13));
        executes(NsRange::new(0, 4));
        // Before the commit: an execution after the copy, in its command buffer.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let site = catch_death(|| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            blit.encode(OnFault::Return, copy_to(&a, 2, &b, 1)).unwrap();
            blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            let execution = ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: &b,
                execution_range: NsRange::new(0, 1),
            };
            e.encode(OnFault::Return, execution).unwrap();
        });
        assert_eq!(site.unwrap_err().file, EXECUTE);
        // Another command buffer, not run yet, executes the buffer.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let pending = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = pending.compute_encoder(OnFault::Return).unwrap().unwrap();
        let execution = ComputeCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &b,
            execution_range: NsRange::new(0, 2),
        };
        e.encode(OnFault::Return, execution).unwrap();
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 1)]));
        assert_eq!(site.unwrap_err().file, COPY);
        drop(pending);
    }

    /// A `useResource` form that makes a buffer resident while a copy past its end may have
    /// written over one of its records dies, as a function of the encoder may copy that command
    /// (`copy_command`), where this platform's copy would carry the intact record: in a compute
    /// encoder at each usage and in a render encoder, once the copy completed, and after the copy
    /// in its command buffer, before the commit. A copy past the end while another command buffer
    /// that has not run yet makes the buffer resident dies at the copy. A buffer whose records are
    /// intact is made resident
    #[test]
    fn a_buffer_a_copy_past_its_end_overwrote_dies_where_made_resident() {
        use reims_vgpu_core::platform::render::{RenderCmd, RenderEncoder as _};
        const USE: &str = "useResources:count:usage:";
        let d = dev();
        let p = crate::platform::tests::platform();
        let q = d.new_command_queue().unwrap();
        /// `icb` made resident in `e` with `usage`.
        fn use_in(e: &<VulkanPlatform as Platform>::ComputeEncoder, icb: &VulkanIcb, usage: u64) {
            let resources = [Some(ResourceRef::Icb(icb))];
            let used = ComputeCmd::UseResources {
                resources: &resources,
                usage,
            };
            e.encode(OnFault::Return, used).unwrap();
        }
        let resident = |icb: &VulkanIcb, usage| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            use_in(&e, icb, usage);
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        };
        let [a, b, c] = three_buffers::<VulkanPlatform>(&d, false, 2);
        resident(&c, 1);
        blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 1)]);
        for usage in [1, 2, 3] {
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| resident(&b, usage)).unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::Abort, USE),
                "{usage}: {line}"
            );
            assert!(line.contains("copy_command"), "{line}");
        }
        let (_target, pass) = crate::gpu::encode_render::tests::pass_of(&p, &d, (4, 4));
        let site = catch_death(|| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let e = cb.render_encoder(OnFault::Return, &pass).unwrap().unwrap();
            let resources = [Some(ResourceRef::Icb(&b))];
            let used = RenderCmd::UseResources {
                resources: &resources,
                usage: 1,
            };
            e.encode(OnFault::Return, used).unwrap();
        });
        assert_eq!(site.unwrap_err().file, USE);
        // After the copy in its command buffer, before the commit.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let site = catch_death(|| {
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            blit.encode(OnFault::Return, copy_to(&a, 2, &b, 1)).unwrap();
            blit.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            use_in(
                &cb.compute_encoder(OnFault::Return).unwrap().unwrap(),
                &b,
                1,
            );
        });
        assert_eq!(site.unwrap_err().file, USE);
        // Another command buffer, not run yet, makes the buffer resident.
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 2);
        let pending = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = pending.compute_encoder(OnFault::Return).unwrap().unwrap();
        use_in(&e, &b, 1);
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![copy_to(&a, 2, &b, 1)]));
        assert_eq!(site.unwrap_err().file, COPY);
        drop(pending);
    }

    /// A copy past the destination's end in a shape nothing measured dies at the copy: more
    /// commands than the destination holds, an index at its end, more than 63 commands past it,
    /// and a buffer into itself; and, each in an otherwise measured shape, a destination of 65
    /// commands, one of 3 kernel buffers, one of Private storage, a render buffer's commands
    /// into a compute buffer, and a source of 64 commands into a destination of 4
    #[test]
    fn copies_past_the_end_in_unmeasured_shapes_die() {
        let d = dev();
        let [a, b, _] = three_buffers::<VulkanPlatform>(&d, false, 4);
        let [big, ..] = three_buffers::<VulkanPlatform>(&d, false, 128);
        let [wide, ..] = three_buffers::<VulkanPlatform>(&d, false, 128);
        let [render, ..] = three_buffers::<VulkanPlatform>(&d, true, 4);
        let long = make::<VulkanPlatform>(&d, &compute_desc(), 65);
        let sixty_four = make::<VulkanPlatform>(&d, &compute_desc(), 64);
        let three = IcbDesc {
            max_kernel_buffer_bind_count: 3,
            ..compute_desc()
        };
        let wider = make::<VulkanPlatform>(&d, &three, 4);
        let private = d
            .new_indirect_command_buffer(&compute_desc(), 4, 0x20)
            .unwrap()
            .unwrap();
        let cases: [BlitCmd<'_, VulkanPlatform>; 9] = [
            copy_to(&big, 5, &b, 0),
            copy_to(&a, 1, &b, 4),
            copy_to(&big, 65, &wide, 127),
            copy_to(&b, 2, &b, 3),
            copy_to(&a, 2, &long, 64),
            copy_to(&a, 2, &wider, 3),
            copy_to(&a, 2, &private, 3),
            copy_to(&render, 2, &b, 3),
            copy_to(&sixty_four, 2, &b, 3),
        ];
        for cmd in cases {
            let what = format!("{cmd:?}");
            let site = catch_death(|| blits::<VulkanPlatform>(&d, vec![cmd])).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, COPY), "{what}");
        }
    }

    /// A way a command buffer uses an indirect command buffer.
    #[derive(Debug, Clone, Copy)]
    enum Use {
        Execution,
        Resident,
        CopyFrom,
        CopyTo,
        Optimize,
        Reset,
    }

    /// Encodes `how` on `cb`, of `icb`, a copy's other buffer `other`, and an execution's
    /// `pipeline` and `out`.
    fn encode_use(
        cb: &VulkanCommandBuffer,
        how: Use,
        icb: &VulkanIcb,
        other: &VulkanIcb,
        pipeline: &VulkanComputePipeline,
        out: &VulkanBuffer,
    ) {
        let one = NsRange::new(0, 1);
        let blit = match how {
            Use::Execution | Use::Resident => None,
            Use::CopyFrom => Some(BlitCmd::CopyIndirectCommandBuffer {
                source: icb,
                source_range: one,
                destination: other,
                destination_index: 0,
            }),
            Use::CopyTo => Some(BlitCmd::CopyIndirectCommandBuffer {
                source: other,
                source_range: one,
                destination: icb,
                destination_index: 0,
            }),
            Use::Optimize => Some(BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer: icb,
                range: one,
            }),
            Use::Reset => Some(BlitCmd::ResetCommandsInBuffer {
                buffer: icb,
                range: one,
            }),
        };
        if let Some(cmd) = blit {
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            e.encode(OnFault::Return, cmd).unwrap();
            e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            return;
        }
        if let Use::Resident = how {
            let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
            e.encode(
                OnFault::Return,
                ComputeCmd::UseResources {
                    resources: &[Some(ResourceRef::Icb(icb))],
                    usage: 1,
                },
            )
            .unwrap();
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            return;
        }
        let c = icb.compute_command(0).unwrap().unwrap();
        for cmd in [
            IcbComputeCmd::SetComputePipelineState {
                pipeline_state: Some(pipeline),
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(out),
                offset: 0,
                index: 0,
            },
            IcbComputeCmd::SetKernelBuffer {
                buffer: Some(out),
                offset: 0,
                index: 1,
            },
            IcbComputeCmd::ConcurrentDispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
        ] {
            c.encode(cmd).unwrap();
        }
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(out))],
                usage: 3,
            },
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer: icb,
                execution_range: one,
            },
            ComputeCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
    }

    /// A buffer core released while a command buffer that uses it may still run keeps its ID
    /// until that command buffer's completed handlers returned where an encoder executes it or
    /// makes it resident, held in a handler and back after them, and not where a blit copies from
    /// it or into it, resets it or optimizes it, as Metal's does (the module docs); an execution
    /// in a command buffer never committed keeps it until the command buffer goes. The
    /// `uniqueIdentifier` of a buffer made after them reads with no relay: the runtime models what
    /// their encoders took of Metal's range
    #[test]
    fn a_buffer_keeps_its_id_while_a_command_buffer_uses_it() {
        use std::sync::mpsc;
        use std::time::Duration;

        use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};
        use reims_vgpu_core::platform::os::Clock as _;

        let d = dev();
        let pipeline = icb_pipeline(&d);
        let desc = IcbDesc {
            command_types: 0x40,
            max_kernel_buffer_bind_count: 2,
            ..IcbDesc::default()
        };
        let out = holding(&d, &[0; 4]);
        let q = d.new_command_queue().unwrap();
        let numbers = |b: &VulkanIcb| b.gpu_resource_id();
        for (how, keeps_id) in [
            (Use::Execution, true),
            (Use::Resident, true),
            (Use::CopyFrom, false),
            (Use::CopyTo, false),
            (Use::Optimize, false),
            (Use::Reset, false),
        ] {
            let icb = make::<VulkanPlatform>(&d, &desc, 1);
            let other = make::<VulkanPlatform>(&d, &desc, 1);
            let old = numbers(&icb);
            let event = d.new_shared_event().unwrap();
            let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
            cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
            encode_use(&cb, how, &icb, &other, &pipeline, &out);
            let (tx, rx) = mpsc::channel();
            let handlers = d.clone_ref();
            cb.add_completed_handler(
                OnFault::Return,
                Box::new(move |_| {
                    let inside = numbers(&make::<VulkanPlatform>(&handlers, &desc, 1));
                    tx.send(inside).unwrap();
                }),
            )
            .unwrap();
            cb.commit(OnFault::Return).unwrap();
            drop(icb);
            let during = numbers(&make::<VulkanPlatform>(&d, &desc, 1));
            assert_eq!(during != old, keeps_id, "{how:?}");
            event.set_signaled_value(1).unwrap();
            let inside = rx.recv().unwrap();
            assert_eq!(inside != old, keeps_id, "{how:?}");
            assert_eq!((cb.status().0, cb.error()), (COMPLETED, None), "{how:?}");
            // The ID comes back once the handlers returned, which this thread, woken from one,
            // may run ahead of, as it may on Metal, before or after `waitUntilCompleted` returns
            // (the module docs).
            let clock = crate::os::OsClock::new();
            let began = clock.now_mono();
            while numbers(&make::<VulkanPlatform>(&d, &desc, 1)) != old {
                let waited = clock.now_mono().saturating_duration_since(began);
                assert!(waited < Duration::from_secs(5), "{how:?}");
                std::thread::sleep(Duration::from_micros(100));
            }
        }
        let icb = make::<VulkanPlatform>(&d, &desc, 1);
        let other = make::<VulkanPlatform>(&d, &desc, 1);
        let old = numbers(&icb);
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        encode_use(&cb, Use::Execution, &icb, &other, &pipeline, &out);
        drop(icb);
        assert_ne!(numbers(&make::<VulkanPlatform>(&d, &desc, 1)), old);
        drop(cb);
        assert_eq!(numbers(&make::<VulkanPlatform>(&d, &desc, 1)), old);
        let after = make::<VulkanPlatform>(&d, &desc, 1);
        assert_eq!(
            after.unique_identifier(),
            plain(|| after.raw.unique_identifier())
        );
        assert_eq!(crate::answer::inexact_counts(), []);
    }

    /// A descriptor of [`compute_desc`] whose commands take 41 threadgroup lengths, past the 31
    /// a record holds.
    fn lengths_desc() -> IcbDesc {
        IcbDesc {
            max_kernel_threadgroup_memory_bind_count: 41,
            ..compute_desc()
        }
    }

    /// A blit's reset and copy move a threadgroup length past the 31 a record holds as they move
    /// the record, as the GPU runs them: while the blit waits on an event, the getters answer
    /// what stands and a setter of a command it copies goes, the copy carrying what it set; once
    /// it ran, the copy's destination answers its source's, and the reset command 0
    #[test]
    fn blits_move_lengths_past_the_record_as_they_run() {
        use std::sync::mpsc;

        use reims_vgpu_core::platform::event::{EventRef, SharedEvent as _};

        let d = dev();
        let (a, b) = (
            make::<VulkanPlatform>(&d, &lengths_desc(), 2),
            make::<VulkanPlatform>(&d, &lengths_desc(), 2),
        );
        let command = |icb: &VulkanIcb, i| icb.compute_command(i).unwrap().unwrap();
        let length = |length| IcbComputeCmd::SetThreadgroupMemoryLength { length, index: 33 };
        for i in [0, 1] {
            command(&a, i).encode(IcbComputeCmd::SetBarrier).unwrap();
            command(&a, i).encode(length(20)).unwrap();
        }
        let q = d.new_command_queue().unwrap();
        let event = d.new_shared_event().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.encode_wait(Some(EventRef::Shared(&event)), 1).unwrap();
        let blit = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyIndirectCommandBuffer {
                source: &a,
                source_range: NsRange::new(0, 1),
                destination: &b,
                destination_index: 1,
            },
            BlitCmd::ResetCommandsInBuffer {
                buffer: &a,
                range: NsRange::new(1, 1),
            },
            BlitCmd::EndEncoding,
        ] {
            blit.encode(OnFault::Return, cmd).unwrap();
        }
        let barriers = |icb: &VulkanIcb| [0, 1].map(|i| command(icb, i).has_barrier());
        let lengths = |icb: &VulkanIcb| {
            [0, 1].map(|i| command(icb, i).threadgroup_memory_length(33).unwrap())
        };
        let (tx, rx) = mpsc::channel();
        cb.add_completed_handler(OnFault::Return, Box::new(move |_| tx.send(()).unwrap()))
            .unwrap();
        cb.commit(OnFault::Return).unwrap();
        assert_eq!((barriers(&a), barriers(&b)), ([true, true], [false, false]));
        assert_eq!((lengths(&a), lengths(&b)), ([20, 20], [0, 0]));
        command(&a, 0).encode(length(24)).unwrap();
        assert_eq!(lengths(&a), [24, 20]);
        event.set_signaled_value(1).unwrap();
        rx.recv().unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
        assert_eq!((barriers(&a), barriers(&b)), ([true, false], [false, true]));
        assert_eq!((lengths(&a), lengths(&b)), ([24, 0], [0, 24]));
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
