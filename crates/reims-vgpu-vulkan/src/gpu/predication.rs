//! Command predication: `encodeStartIf:offset:comparison:referenceValue:`, `encodeStartElse`,
//! `encodeEndIf`, `encodeStartWhile:offset:comparison:referenceValue:`, `encodeEndWhile`,
//! `encodeStartDoWhile` and `encodeEndDoWhile:offset:comparison:referenceValue:`. Metal's private
//! API, built here over the runtime's compute encoder; the runtime's own predication is never
//! asked.
//!
//! # What Metal does
//!
//! Measured on an M3 Max (macOS 26.7):
//! - **The condition** is `word OP reference`, unsigned, with the word on the left. The word is
//!   the 32 bits at the buffer's `offset & !3`: an offset of 1, 2, 3 or 6 reads the aligned word
//!   at or below it, and the next word plays no part. `comparison` is not an
//!   `MTLCompareFunction`:
//!
//!   | `comparison` | holds when |
//!   |---|---|
//!   | 0 | `word == reference` |
//!   | 1 | `word != reference` |
//!   | 2 | `word > reference` |
//!   | 3 | `word >= reference` |
//!   | 4 | `word < reference` |
//!   | 5 | `word <= reference` |
//!   | any other value | `word == reference` |
//!
//!   So `compute_conditionals`' comparison 2 is `>` (7 > 7 and 7 > 8 fail, and the Else runs)
//!   and its 5 is `<=` (7 <= 7 holds, and the If runs). `compute_loops`' 1 is `!=`: its loops
//!   run while the word is not 5 and not 3.
//! - **When the word is read**: when the GPU reaches the record. Writes from an earlier dispatch
//!   or an earlier encoder, and the CPU's before the commit, all count, with no barrier.
//! - **What a false branch skips**: direct and indirect dispatches, and indirect command buffer
//!   executions.
//! - **Setters are never skipped**: they take effect in encode order whichever branch runs.
//! - **Nesting**: 4096 levels were measured, so `maxPredicatedNestingDepth` (device-info key
//!   36) is `u32::MAX`.
//! - **Loops**: a While tests its condition before each iteration and a DoWhile after, reading
//!   the word again each time. Each dispatch of each iteration runs with the bindings it was
//!   encoded with, not the ones the body ends with.
//! - **A unit** is a run of compute encoders of one command buffer. It ends at a blit or render
//!   encoder, `encodeSignalEvent:value:` or `encodeWaitForEvent:value:`; a scheduled handler
//!   does not end it. Blocks can span a unit's encoders but not a unit's end: an End after the
//!   end is unbalanced.
//! - **A unit's first indirect dispatch** (`dispatchThreadgroupsWithIndirectBuffer:...` or
//!   `dispatchThreadsWithIndirectBuffer:...`) ends the process at the call (a segmentation
//!   fault) when its encoder's dispatch type is not that of the unit's first encoder, serial
//!   or concurrent; a type neither is counts as serial. The first encoder counts with no
//!   dispatch of its own, and once an indirect dispatch went through, later ones of either type
//!   do (`interplay_decodewalk_27`'s ending; 25 shapes of 1 to 3 encoders, threadgroups and
//!   threads alike). A blit or render encoder, an event signal or wait, or a new command buffer
//!   between ends the unit, and no shape with one there dies.
//! - **Unbalanced records empty the unit**. These answer NO, and every dispatch of the unit
//!   writes nothing, including its earlier encoders'; the command buffer still completes:
//!   - an End with no open block, or one of another block's kind;
//!   - an Else with no open If;
//!   - a Start on a nil buffer, which opens nothing.
//! - **A block still open at the unit's end**: a DoWhile, or an If or While whose condition held,
//!   is harmless. An If or While whose condition failed fails the command buffer, in a GPU
//!   recovery.
//!
//! # What the adapter builds
//!
//! - **Block structure at encode time.** Each command buffer models its current unit at encode
//!   time ([`Predication`]): its open blocks, whether an unbalanced record emptied it, and
//!   whether it sent guest work. The End answers and the emptying come from this model.
//! - **A skip counter on the GPU.** Each command buffer holds a `u32` counter, which the first
//!   step of each unit resets. It is 0 while commands run and above 0 while they are skipped,
//!   and a skipped block's nested blocks count on from it.
//! - **Steps.** Each Start, Else and End is a step: one thread of the adapter's kernel
//!   (`predication.metal`) reads and writes the counter. A memory barrier comes before the step,
//!   so it reads the guest's word after the work before it, and another after it.
//! - **Predicated work.** A dispatch or execution inside a block that can skip runs as its
//!   indirect form. A step first writes the form's arguments: the direct arguments, a copy of the
//!   guest's indirect arguments, or the same with the counts at 0 while skipping (a dispatch's
//!   grid, an execution's length).
//! - **The step's bindings.** A step binds its own pipeline and buffers 0 to 2, then binds the
//!   guest's pipeline and those buffers again. A guest with no pipeline set has none to bind, so
//!   its next work first takes a fresh runtime encoder, which is given back the guest's bindings
//!   and residency. Without that, an execution that inherits the pipeline would run the step's
//!   kernel.
//! - **Loops are laid down.** Vulkan has no jump in a command stream, so a loop's body is
//!   recorded as the guest sends it and encoded again for each iteration up to a fixed count
//!   ([`UNROLL`], or [`NESTED_UNROLL`] inside another loop). Each copy starts from the bindings
//!   the loop started with, and is predicated on the loop's condition. After the last copy, a
//!   loop still running sets a flag. At the command buffer's completion, before the handlers core
//!   adds after the encoding, a set flag ends the process, naming the loop: Metal would run it
//!   on.
//!
//! # What dies
//!
//! What the adapter has no form for dies at the record that shows it, naming it:
//! - an If or While still open where its unit ends, at the blit or render encoder, the event or
//!   the commit that ends the unit. Its fate depends on the GPU's reading of the word. A DoWhile
//!   left open ran once, as on Metal, and is kept. A decode fault ends the unit without either,
//!   and commits nothing.
//! - a record that empties a unit after the unit sent work: Metal drops that work too.
//! - a second Else in one If, an Else whose innermost open block is a loop, and a nil
//!   EndDoWhile inside an open DoWhile. Metal's answers to the first and last are not modelled,
//!   and the second's is not measured.
//! - a fence inside a block, or in an emptied unit, which is not measured.
//! - a loop inside two others, which the counts laid down bound.
//! - a condition word past its buffer's end, which is not measured.
//! - a loop that runs past its iterations laid down, at completion.
//! - a unit's first indirect dispatch in an encoder of another dispatch type than the unit's
//!   first encoder's, as Metal ends the process there.
//!
//! # Where the indirect form differs
//!
//! - **Encode-time checks.** A dispatch whose threadgroup its direct form runs nothing for, or
//!   refuses, is sent as that form, whatever the condition; a predicated one's grid and
//!   threadgroup count are taken modulo `2^32`, as the direct form takes them. A kernel the
//!   runtime records at a host step is bound there, so a binding its direct form refuses at the
//!   encode is refused by the host step instead, which dies at `commit` naming the refusal
//!   ([`crate::gpu::command`]).
//! - **Executions.** A predicated execution runs as the runtime's indirect execution, by a host
//!   step: its commands are read as the GPU reaches them, not as they stand at the commit.
//! - **Residency.** The guest's buffers a step reads are bound to the step, which makes them
//!   resident as any bound buffer is.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, Range};
use std::sync::{Arc, Weak};

use metal2vulkan::metal::{BarrierScope, DispatchType, ResourceUsage};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::Truncate;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_core::platform::types::{MtlRegion, MtlSize, NsRange};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::rt::BlockEnv;
use reims_vgpu_core::sync::Mutex;

use crate::answer::{Policy, metal_aborts, plain, refused, send};
use crate::gpu::command::{CommandBufferObject, VulkanCommandBuffer};
use crate::gpu::device::VulkanDevice;
use crate::gpu::encode_blit::EncoderObject;
use crate::gpu::encode_compute::VulkanComputeEncoder;
use crate::gpu::icb::{Inherits, VulkanIcb};
use crate::gpu::resource::{
    VulkanBuffer, VulkanComputePipeline, VulkanHeap, VulkanSamplerState, VulkanTexture,
};
use crate::gpu::writes::{Bound, Slot, Whether, texture_bound};
use crate::gpu::{metal_value, region, resource_options, size};
use crate::platform::{Obj, VulkanPlatform};

/// The step kernel's metallib, which `build.rs` compiled from `predication.metal`.
const METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/predication.metallib"));

/// The step kernel.
const KERNEL: &str = "predication_step";

/// The iterations of a loop that is inside no other loop, laid down: past them it dies.
pub const UNROLL: u32 = 64;

/// The iterations of a loop inside another, laid down: each of the outer loop's copies holds
/// them all.
pub const NESTED_UNROLL: u32 = 16;

/// How many loops a loop may be inside.
const LOOP_DEPTH: usize = 1;

/// A step's bytes in its chunk.
const STEP_BYTES: u64 = 128;

/// Where in a step the arguments of its indirect form are: `Step::out`.
const OUT: u64 = 64;

/// The steps of one chunk buffer.
const CHUNK_STEPS: u64 = 512;

/// The state buffer's bytes: the counter, then one flag per loop.
const STATE_BYTES: u64 = 4096;

/// The loops one command buffer can flag.
const FLAGS: usize = 1023;

/// The buffer slots a step binds, which are the guest's again after it.
const STEP_SLOTS: u32 = 3;

/// `MTLAttributeStrideStatic`: no stride of the bind's own.
const STRIDE_STATIC: u64 = u64::MAX;

/// A step: `Step` in `predication.metal`, as the CPU writes it.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Step {
    op: u32,
    comparison: u32,
    reference: u32,
    flag: u32,
    mask: u32,
    count: u32,
    unused: [u32; 2],
    args: [u32; 8],
    out: [u32; 8],
}

/// The step's counter is 0 before it.
const RESET: u32 = 0x100;
/// The steps `predication.metal` names.
const IF: u32 = 1;
const ELSE: u32 = 2;
const END: u32 = 3;
const ENTER: u32 = 4;
const CHECK: u32 = 5;
const OVERRUN: u32 = 6;
const ARGS: u32 = 7;
const GUEST_ARGS: u32 = 8;

impl Step {
    /// A step that reads no condition.
    fn of(op: u32) -> Step {
        Step {
            op,
            ..Step::default()
        }
    }

    /// A step that tests `cond`.
    fn testing(op: u32, cond: &Cond) -> Step {
        Step {
            op,
            comparison: cond.comparison,
            reference: cond.reference,
            ..Step::default()
        }
    }

    /// A step that writes `args`, or the guest's, with the words of `mask` 0 while skipping.
    fn arguments(op: u32, args: &[u32], mask: u32) -> Step {
        let mut step = Step::of(op);
        step.mask = mask;
        step.count = u32::try_from(args.len()).unwrap_or(0);
        for (to, from) in step.args.iter_mut().zip(args) {
            *to = *from;
        }
        step
    }
}

/// The death of `sel`, a record whose shape the adapter's predication has no form for.
fn dies(sel: &'static str, what: &str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        &format!("command predication: {what}"),
    )
}

/// `call`'s result, or the death of `sel` naming the runtime's refusal.
fn made<T>(sel: &'static str, call: impl FnOnce() -> Result<T, rt::Error>) -> T {
    plain(call).unwrap_or_else(|e| refused(sel, &e))
}

/// A condition: the guest's word and how it is compared.
struct Cond {
    buffer: VulkanBuffer,
    /// The word's offset: the aligned word at or below the record's offset.
    word: u64,
    /// The comparison as the step kernel takes it: 0 to 5.
    comparison: u32,
    reference: u32,
}

impl Cond {
    /// The condition of `sel`'s record; a word past the buffer's end dies.
    fn new(
        sel: &'static str,
        buffer: &VulkanBuffer,
        offset: u64,
        comparison: u64,
        reference: u32,
    ) -> Cond {
        let word = offset & !3;
        let length = plain(|| buffer.raw.length());
        if word.checked_add(4).is_none_or(|end| end > length) {
            dies(
                sel,
                &format!(
                    "a condition word at {offset:#x} of a buffer of {length:#x} bytes, where what \
                     Metal reads is not measured"
                ),
            );
        }
        Cond {
            buffer: buffer.clone_ref(),
            word,
            comparison: u32::try_from(comparison)
                .ok()
                .filter(|c| *c <= 5)
                .unwrap_or(0),
            reference,
        }
    }

    fn copy(&self) -> Cond {
        Cond {
            buffer: self.buffer.clone_ref(),
            ..*self
        }
    }
}

/// An open block.
struct Frame {
    kind: Kind,
    /// Whether the counter may be above 0 inside it, so its work is predicated.
    predicated: bool,
    /// What it is, for a death that names it.
    name: String,
}

enum Kind {
    If { orelse: bool },
    While(Loop),
    DoWhile(Loop),
}

/// What a loop's End lays down its body again with.
struct Loop {
    /// The log index its body starts at.
    body: usize,
    /// The bindings it started with, which each copy of its body starts from.
    start: Bindings,
    /// Its flag in the state buffer, which an overrun sets.
    flag: usize,
    /// The iterations laid down.
    unroll: u32,
    /// A While's condition; a DoWhile's comes at its End.
    cond: Option<Cond>,
}

/// A stage-in region, as the guest set it.
enum StageIn {
    Region(MtlRegion),
    Indirect(VulkanBuffer, u64),
}

/// The bindings of the open encoder, as the guest set them: what each copy of a loop's body
/// starts from, and what a step binds again after it. The runtime keeps the same, but for the
/// slots a step bound and, where the guest set none, the pipeline.
#[derive(Default)]
struct Bindings {
    pipeline: Option<VulkanComputePipeline>,
    /// Each buffer with its offset and attribute stride.
    buffers: BTreeMap<u32, (VulkanBuffer, u64, u64)>,
    textures: BTreeMap<u32, VulkanTexture>,
    /// Each sampler state with the LOD clamps its bind set.
    samplers: BTreeMap<u32, (VulkanSamplerState, Option<(f32, f32)>)>,
    threadgroup_lengths: BTreeMap<u32, u32>,
    /// The imageblock size, where one was set (1 by 1 until then).
    imageblock: Option<(u64, u64)>,
    stage_in: Option<StageIn>,
}

/// An index or a range's index past `u32`, which the arm then dies at.
fn index(raw: u64) -> Option<u32> {
    u32::try_from(raw).ok()
}

/// The indices `range` names, from its location.
fn indices(range: NsRange) -> impl Iterator<Item = u32> {
    (0..).map_while(move |i: u64| index(range.location.checked_add(i)?))
}

impl Bindings {
    fn copy(&self) -> Bindings {
        Bindings {
            pipeline: self.pipeline.as_ref().map(Handle::clone_ref),
            buffers: self
                .buffers
                .iter()
                .map(|(i, (b, o, s))| (*i, (b.clone_ref(), *o, *s)))
                .collect(),
            textures: self
                .textures
                .iter()
                .map(|(i, t)| (*i, t.clone_ref()))
                .collect(),
            samplers: self
                .samplers
                .iter()
                .map(|(i, (s, c))| (*i, (s.clone_ref(), *c)))
                .collect(),
            threadgroup_lengths: self.threadgroup_lengths.clone(),
            imageblock: self.imageblock,
            stage_in: match &self.stage_in {
                None => None,
                Some(StageIn::Region(r)) => Some(StageIn::Region(*r)),
                Some(StageIn::Indirect(b, o)) => Some(StageIn::Indirect(b.clone_ref(), *o)),
            },
        }
    }

    fn bind_buffer(&mut self, at: u32, buffer: Option<&VulkanBuffer>, offset: u64, stride: u64) {
        match buffer {
            Some(b) => {
                self.buffers.insert(at, (b.clone_ref(), offset, stride));
            }
            None => {
                self.buffers.remove(&at);
            }
        }
    }

    fn bind_sampler(
        &mut self,
        at: u32,
        sampler: Option<&VulkanSamplerState>,
        clamps: Option<(f32, f32)>,
    ) {
        match sampler {
            Some(s) => {
                self.samplers.insert(at, (s.clone_ref(), clamps));
            }
            None => {
                self.samplers.remove(&at);
            }
        }
    }

    fn bind_texture(&mut self, at: u32, texture: Option<&VulkanTexture>) {
        match texture {
            Some(t) => {
                self.textures.insert(at, t.clone_ref());
            }
            None => {
                self.textures.remove(&at);
            }
        }
    }

    /// The bindings after `cmd`, as the runtime's argument tables take it. A setter the runtime
    /// refuses dies in its arm, after this.
    fn apply(&mut self, cmd: &ComputeCmd<'_, VulkanPlatform>) {
        match *cmd {
            ComputeCmd::SetBufferOffset { offset, index: at } => {
                if let Some(at) = index(at)
                    && let Some(bound) = self.buffers.get_mut(&at)
                {
                    bound.1 = offset;
                    bound.2 = STRIDE_STATIC;
                }
            }
            ComputeCmd::SetBufferOffsetAttributeStride {
                offset,
                stride,
                index: at,
            } => {
                if let Some(at) = index(at)
                    && let Some(bound) = self.buffers.get_mut(&at)
                {
                    bound.1 = offset;
                    bound.2 = stride;
                }
            }
            ComputeCmd::SetBuffers {
                buffers,
                offsets,
                range,
            } => {
                for ((at, b), o) in indices(range).zip(buffers).zip(offsets) {
                    self.bind_buffer(at, *b, *o, STRIDE_STATIC);
                }
            }
            ComputeCmd::SetBuffersAttributeStrides {
                buffers,
                offsets,
                strides,
                range,
            } => {
                for (((at, b), o), s) in indices(range).zip(buffers).zip(offsets).zip(strides) {
                    self.bind_buffer(at, *b, *o, *s);
                }
            }
            ComputeCmd::SetComputePipelineState { state: Some(p) } => {
                self.pipeline = Some(p.clone_ref());
            }
            ComputeCmd::SetImageblockWidth { width, height } => {
                self.imageblock = Some((width, height));
            }
            ComputeCmd::SetSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                for (((at, s), lo), hi) in indices(range)
                    .zip(samplers)
                    .zip(lod_min_clamps)
                    .zip(lod_max_clamps)
                {
                    self.bind_sampler(at, *s, Some((*lo, *hi)));
                }
            }
            ComputeCmd::SetSamplerStates { samplers, range } => {
                for (at, s) in indices(range).zip(samplers) {
                    self.bind_sampler(at, *s, None);
                }
            }
            ComputeCmd::SetStageInRegion { region: r } => {
                self.stage_in = Some(StageIn::Region(r));
            }
            ComputeCmd::SetStageInRegionIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                self.stage_in = Some(StageIn::Indirect(
                    indirect_buffer.clone_ref(),
                    indirect_buffer_offset,
                ));
            }
            ComputeCmd::SetTexture { texture, index: at } => {
                if let Some(at) = index(at) {
                    self.bind_texture(at, texture);
                }
            }
            ComputeCmd::SetTextures { textures, range } => {
                for (at, t) in indices(range).zip(textures) {
                    self.bind_texture(at, *t);
                }
            }
            ComputeCmd::SetThreadgroupMemoryLength { length, index: at } => {
                if let (Ok(length), Some(at)) = (u32::try_from(length), index(at)) {
                    self.threadgroup_lengths.insert(at, length);
                }
            }
            _ => {}
        }
    }
}

/// Whether two optional handles are the same object.
fn same<T: Handle>(a: Option<&T>, b: Option<&T>) -> bool {
    a.map(Handle::id) == b.map(Handle::id)
}

/// Binds in `e` what moves it from `from` to `to`, and, where a step bound them (`stepped`), the
/// pipeline and the step's slots whatever `from` holds. A slot `to` leaves empty is unbound; a
/// threadgroup length it leaves unset is 0. A pipeline or a stage-in region it leaves unset stays
/// as it was, which the runtime has no unset of.
fn restore(
    e: &mut rt::ComputeCommandEncoder,
    from: &Bindings,
    to: &Bindings,
    stepped: bool,
) -> Result<(), rt::Error> {
    if let Some(p) = &to.pipeline
        && (stepped || !same(from.pipeline.as_ref(), Some(p)))
    {
        e.set_compute_pipeline_state(p)?;
    }
    let step_slots = if stepped { 0..STEP_SLOTS } else { 0..0 };
    let slots: BTreeSet<u32> = from
        .buffers
        .keys()
        .chain(to.buffers.keys())
        .copied()
        .chain(step_slots)
        .collect();
    for at in slots {
        let (was, now) = (from.buffers.get(&at), to.buffers.get(&at));
        let moved = match (was, now) {
            (Some((a, ao, a_s)), Some((b, bo, b_s))) => a.id() != b.id() || ao != bo || a_s != b_s,
            (None, None) => false,
            _ => true,
        };
        if moved || (stepped && at < STEP_SLOTS) {
            match now {
                Some((b, offset, stride)) => {
                    e.set_buffer_with_stride(Some(&b.raw), *offset, *stride, at)?;
                }
                None => e.set_buffer(None, 0, at)?,
            }
        }
    }
    let slots: BTreeSet<u32> = from
        .textures
        .keys()
        .chain(to.textures.keys())
        .copied()
        .collect();
    for at in slots {
        let now = to.textures.get(&at);
        if !same(from.textures.get(&at), now) {
            e.set_texture(now.map(|t| &t.raw), at)?;
        }
    }
    let slots: BTreeSet<u32> = from
        .samplers
        .keys()
        .chain(to.samplers.keys())
        .copied()
        .collect();
    for at in slots {
        let (was, now) = (from.samplers.get(&at), to.samplers.get(&at));
        let moved = match (was, now) {
            (Some((a, ac)), Some((b, bc))) => {
                let bits = |c: &Option<(f32, f32)>| c.map(|(lo, hi)| (lo.to_bits(), hi.to_bits()));
                a.id() != b.id() || bits(ac) != bits(bc)
            }
            (None, None) => false,
            _ => true,
        };
        if moved {
            match now {
                Some((s, Some((lo, hi)))) => {
                    e.set_sampler_state_with_lod_clamps(Some(&**s), *lo, *hi, at)?;
                }
                Some((s, None)) => e.set_sampler_state(Some(&**s), at)?,
                None => e.set_sampler_state(None, at)?,
            }
        }
    }
    let slots: BTreeSet<u32> = from
        .threadgroup_lengths
        .keys()
        .chain(to.threadgroup_lengths.keys())
        .copied()
        .collect();
    for at in slots {
        let now = to.threadgroup_lengths.get(&at).copied();
        if from.threadgroup_lengths.get(&at).copied() != now {
            e.set_threadgroup_memory_length(now.unwrap_or(0), at)?;
        }
    }
    let imageblock = |b: &Bindings| b.imageblock.unwrap_or((1, 1));
    if imageblock(from) != imageblock(to) {
        let (width, height) = imageblock(to);
        e.set_imageblock_size(width, height)?;
    }
    match (&from.stage_in, &to.stage_in) {
        (Some(StageIn::Region(a)), Some(StageIn::Region(b))) if a == b => {}
        (Some(StageIn::Indirect(a, ao)), Some(StageIn::Indirect(b, bo)))
            if a.id() == b.id() && ao == bo => {}
        (_, Some(StageIn::Region(r))) => e.set_stage_in_region(region(*r))?,
        (_, Some(StageIn::Indirect(b, o))) => e.set_stage_in_region_indirect(&b.raw, *o)?,
        (_, None) => {}
    }
    Ok(())
}

/// A resource of a residency or barrier list, held.
enum Resource {
    Buffer(VulkanBuffer),
    Texture(VulkanTexture),
    Icb(VulkanIcb),
}

impl Resource {
    fn of(r: &ResourceRef<'_, VulkanPlatform>) -> Resource {
        match r {
            ResourceRef::Buffer(b) => Resource::Buffer(b.clone_ref()),
            ResourceRef::Texture(t) => Resource::Texture(t.clone_ref()),
            ResourceRef::Icb(i) => Resource::Icb(i.clone_ref()),
        }
    }

    fn as_ref(&self) -> ResourceRef<'_, VulkanPlatform> {
        match self {
            Resource::Buffer(b) => ResourceRef::Buffer(b),
            Resource::Texture(t) => ResourceRef::Texture(t),
            Resource::Icb(i) => ResourceRef::Icb(i),
        }
    }

    fn raw(&self) -> rt::ResourceRef<'_> {
        match self {
            Resource::Buffer(b) => rt::ResourceRef::Buffer(&b.raw),
            Resource::Texture(t) => rt::ResourceRef::Texture(&t.raw),
            Resource::Icb(i) => rt::ResourceRef::IndirectCommandBuffer(&i.raw),
        }
    }
}

/// What the open encoder made resident, which a fresh runtime encoder is given again.
enum Use {
    Resources(Vec<Resource>, ResourceUsage),
    Heaps(Vec<VulkanHeap>),
}

impl Use {
    fn apply(&self, e: &mut rt::ComputeCommandEncoder) -> Result<(), rt::Error> {
        match self {
            Use::Resources(list, usage) => {
                let refs: Vec<rt::ResourceRef<'_>> = list.iter().map(Resource::raw).collect();
                e.use_resources(&refs, *usage)
            }
            Use::Heaps(heaps) => {
                let heaps: Vec<&rt::Heap> = heaps.iter().map(|h| &h.raw).collect();
                e.use_heaps(&heaps)
            }
        }
    }
}

/// A command a loop's body holds, as core sent it: encoded again for each copy of the body.
enum Owned {
    DispatchThreadgroups(MtlSize, MtlSize),
    DispatchThreadgroupsIndirect(Option<VulkanBuffer>, u64, MtlSize),
    DispatchThreads(MtlSize, MtlSize),
    DispatchThreadsIndirect(Option<VulkanBuffer>, u64),
    StartDoWhile,
    StartElse,
    StartIf(Option<VulkanBuffer>, u64, u64, u32),
    StartWhile(Option<VulkanBuffer>, u64, u64, u32),
    ExecuteIndirect(VulkanIcb, VulkanBuffer, u64),
    Execute(VulkanIcb, NsRange),
    Flush,
    BarrierResources(Vec<Option<Resource>>),
    BarrierScope(u64),
    SetBufferOffset(u64, u64),
    SetBufferOffsetStride(u64, u64, u64),
    SetBuffersStrides(Vec<Option<VulkanBuffer>>, Vec<u64>, Vec<u64>, NsRange),
    SetBuffers(Vec<Option<VulkanBuffer>>, Vec<u64>, NsRange),
    SetPipeline(Option<VulkanComputePipeline>),
    SetImageblock(u64, u64),
    SetSamplersClamps(Vec<Option<VulkanSamplerState>>, Vec<f32>, Vec<f32>, NsRange),
    SetSamplers(Vec<Option<VulkanSamplerState>>, NsRange),
    SetStageInRegion(MtlRegion),
    SetStageInRegionIndirect(VulkanBuffer, u64),
    SetTexture(Option<VulkanTexture>, u64),
    SetTextures(Vec<Option<VulkanTexture>>, NsRange),
    SetThreadgroupMemoryLength(u64, u64),
    UseHeaps(Vec<Option<VulkanHeap>>),
    UseResources(Vec<Option<Resource>>, u64),
}

/// Each of `list`, held.
fn held<T: Handle>(list: &[Option<&T>]) -> Vec<Option<T>> {
    list.iter().map(|o| o.map(Handle::clone_ref)).collect()
}

/// Each of `list`, borrowed.
fn lent<T>(list: &[Option<T>]) -> Vec<Option<&T>> {
    list.iter().map(Option::as_ref).collect()
}

/// Each resource of `list`, borrowed.
fn resources(list: &[Option<Resource>]) -> Vec<Option<ResourceRef<'_, VulkanPlatform>>> {
    list.iter()
        .map(|r| r.as_ref().map(Resource::as_ref))
        .collect()
}

impl Owned {
    /// `cmd`, held; `None` for what a body never holds (`endEncoding`, which ends no body, and a
    /// fence, which dies in one).
    fn of(cmd: &ComputeCmd<'_, VulkanPlatform>) -> Option<Owned> {
        let resources = |list: &[Option<ResourceRef<'_, VulkanPlatform>>]| {
            list.iter().map(|r| r.as_ref().map(Resource::of)).collect()
        };
        Some(match *cmd {
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => Owned::DispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup),
            ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
                threads_per_threadgroup,
            } => Owned::DispatchThreadgroupsIndirect(
                indirect_buffer.map(Handle::clone_ref),
                indirect_buffer_offset,
                threads_per_threadgroup,
            ),
            ComputeCmd::DispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => Owned::DispatchThreads(threads_per_grid, threads_per_threadgroup),
            ComputeCmd::DispatchThreadsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => Owned::DispatchThreadsIndirect(
                indirect_buffer.map(Handle::clone_ref),
                indirect_buffer_offset,
            ),
            ComputeCmd::EncodeStartDoWhile => Owned::StartDoWhile,
            ComputeCmd::EncodeStartElse => Owned::StartElse,
            ComputeCmd::EncodeStartIf {
                buffer,
                offset,
                comparison,
                reference_value,
            } => Owned::StartIf(
                buffer.map(Handle::clone_ref),
                offset,
                comparison,
                reference_value,
            ),
            ComputeCmd::EncodeStartWhile {
                buffer,
                offset,
                comparison,
                reference_value,
            } => Owned::StartWhile(
                buffer.map(Handle::clone_ref),
                offset,
                comparison,
                reference_value,
            ),
            ComputeCmd::EndEncoding
            | ComputeCmd::UpdateFence { .. }
            | ComputeCmd::WaitForFence { .. } => return None,
            ComputeCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer,
                indirect_range_buffer,
                indirect_buffer_offset,
            } => Owned::ExecuteIndirect(
                indirect_commandbuffer.clone_ref(),
                indirect_range_buffer.clone_ref(),
                indirect_buffer_offset,
            ),
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                execution_range,
            } => Owned::Execute(indirect_command_buffer.clone_ref(), execution_range),
            ComputeCmd::CompressedTextureReinterpretationFlush => Owned::Flush,
            ComputeCmd::MemoryBarrierWithResources { resources: list } => {
                Owned::BarrierResources(resources(list))
            }
            ComputeCmd::MemoryBarrierWithScope { scope } => Owned::BarrierScope(scope),
            ComputeCmd::SetBufferOffset { offset, index } => Owned::SetBufferOffset(offset, index),
            ComputeCmd::SetBufferOffsetAttributeStride {
                offset,
                stride,
                index,
            } => Owned::SetBufferOffsetStride(offset, stride, index),
            ComputeCmd::SetBuffersAttributeStrides {
                buffers,
                offsets,
                strides,
                range,
            } => Owned::SetBuffersStrides(held(buffers), offsets.to_vec(), strides.to_vec(), range),
            ComputeCmd::SetBuffers {
                buffers,
                offsets,
                range,
            } => Owned::SetBuffers(held(buffers), offsets.to_vec(), range),
            ComputeCmd::SetComputePipelineState { state } => {
                Owned::SetPipeline(state.map(Handle::clone_ref))
            }
            ComputeCmd::SetImageblockWidth { width, height } => Owned::SetImageblock(width, height),
            ComputeCmd::SetSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => Owned::SetSamplersClamps(
                held(samplers),
                lod_min_clamps.to_vec(),
                lod_max_clamps.to_vec(),
                range,
            ),
            ComputeCmd::SetSamplerStates { samplers, range } => {
                Owned::SetSamplers(held(samplers), range)
            }
            ComputeCmd::SetStageInRegion { region: r } => Owned::SetStageInRegion(r),
            ComputeCmd::SetStageInRegionIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                Owned::SetStageInRegionIndirect(indirect_buffer.clone_ref(), indirect_buffer_offset)
            }
            ComputeCmd::SetTexture { texture, index } => {
                Owned::SetTexture(texture.map(Handle::clone_ref), index)
            }
            ComputeCmd::SetTextures { textures, range } => {
                Owned::SetTextures(held(textures), range)
            }
            ComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                Owned::SetThreadgroupMemoryLength(length, index)
            }
            ComputeCmd::UseHeaps { heaps } => Owned::UseHeaps(held(heaps)),
            ComputeCmd::UseResources {
                resources: list,
                usage,
            } => Owned::UseResources(resources(list), usage),
        })
    }

    /// Sends the command again to `enc`, as core sent it.
    fn send(&self, enc: &VulkanComputeEncoder) -> RaiseResult<()> {
        let on = OnFault::Die;
        match self {
            Owned::DispatchThreadgroups(groups, threads) => enc.encode(
                on,
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid: *groups,
                    threads_per_threadgroup: *threads,
                },
            ),
            Owned::DispatchThreadgroupsIndirect(buffer, offset, threads) => enc.encode(
                on,
                ComputeCmd::DispatchThreadgroupsIndirect {
                    indirect_buffer: buffer.as_ref(),
                    indirect_buffer_offset: *offset,
                    threads_per_threadgroup: *threads,
                },
            ),
            Owned::DispatchThreads(threads, per_group) => enc.encode(
                on,
                ComputeCmd::DispatchThreads {
                    threads_per_grid: *threads,
                    threads_per_threadgroup: *per_group,
                },
            ),
            Owned::DispatchThreadsIndirect(buffer, offset) => enc.encode(
                on,
                ComputeCmd::DispatchThreadsIndirect {
                    indirect_buffer: buffer.as_ref(),
                    indirect_buffer_offset: *offset,
                },
            ),
            Owned::StartDoWhile => enc.encode(on, ComputeCmd::EncodeStartDoWhile),
            Owned::StartElse => enc.encode(on, ComputeCmd::EncodeStartElse),
            Owned::StartIf(buffer, offset, comparison, reference) => enc.encode(
                on,
                ComputeCmd::EncodeStartIf {
                    buffer: buffer.as_ref(),
                    offset: *offset,
                    comparison: *comparison,
                    reference_value: *reference,
                },
            ),
            Owned::StartWhile(buffer, offset, comparison, reference) => enc.encode(
                on,
                ComputeCmd::EncodeStartWhile {
                    buffer: buffer.as_ref(),
                    offset: *offset,
                    comparison: *comparison,
                    reference_value: *reference,
                },
            ),
            Owned::ExecuteIndirect(icb, buffer, offset) => enc.encode(
                on,
                ComputeCmd::ExecuteCommandsIndirect {
                    indirect_commandbuffer: icb,
                    indirect_range_buffer: buffer,
                    indirect_buffer_offset: *offset,
                },
            ),
            Owned::Execute(icb, range) => enc.encode(
                on,
                ComputeCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: icb,
                    execution_range: *range,
                },
            ),
            Owned::Flush => enc.encode(on, ComputeCmd::CompressedTextureReinterpretationFlush),
            Owned::BarrierResources(list) => {
                let list = resources(list);
                enc.encode(
                    on,
                    ComputeCmd::MemoryBarrierWithResources { resources: &list },
                )
            }
            Owned::BarrierScope(scope) => {
                enc.encode(on, ComputeCmd::MemoryBarrierWithScope { scope: *scope })
            }
            Owned::SetBufferOffset(offset, index) => enc.encode(
                on,
                ComputeCmd::SetBufferOffset {
                    offset: *offset,
                    index: *index,
                },
            ),
            Owned::SetBufferOffsetStride(offset, stride, index) => enc.encode(
                on,
                ComputeCmd::SetBufferOffsetAttributeStride {
                    offset: *offset,
                    stride: *stride,
                    index: *index,
                },
            ),
            Owned::SetBuffersStrides(buffers, offsets, strides, range) => {
                let buffers = lent(buffers);
                enc.encode(
                    on,
                    ComputeCmd::SetBuffersAttributeStrides {
                        buffers: &buffers,
                        offsets,
                        strides,
                        range: *range,
                    },
                )
            }
            Owned::SetBuffers(buffers, offsets, range) => {
                let buffers = lent(buffers);
                enc.encode(
                    on,
                    ComputeCmd::SetBuffers {
                        buffers: &buffers,
                        offsets,
                        range: *range,
                    },
                )
            }
            Owned::SetPipeline(state) => enc.encode(
                on,
                ComputeCmd::SetComputePipelineState {
                    state: state.as_ref(),
                },
            ),
            Owned::SetImageblock(width, height) => enc.encode(
                on,
                ComputeCmd::SetImageblockWidth {
                    width: *width,
                    height: *height,
                },
            ),
            Owned::SetSamplersClamps(samplers, lo, hi, range) => {
                let samplers = lent(samplers);
                enc.encode(
                    on,
                    ComputeCmd::SetSamplerStatesLodClamps {
                        samplers: &samplers,
                        lod_min_clamps: lo,
                        lod_max_clamps: hi,
                        range: *range,
                    },
                )
            }
            Owned::SetSamplers(samplers, range) => {
                let samplers = lent(samplers);
                enc.encode(
                    on,
                    ComputeCmd::SetSamplerStates {
                        samplers: &samplers,
                        range: *range,
                    },
                )
            }
            Owned::SetStageInRegion(r) => {
                enc.encode(on, ComputeCmd::SetStageInRegion { region: *r })
            }
            Owned::SetStageInRegionIndirect(buffer, offset) => enc.encode(
                on,
                ComputeCmd::SetStageInRegionIndirect {
                    indirect_buffer: buffer,
                    indirect_buffer_offset: *offset,
                },
            ),
            Owned::SetTexture(texture, index) => enc.encode(
                on,
                ComputeCmd::SetTexture {
                    texture: texture.as_ref(),
                    index: *index,
                },
            ),
            Owned::SetTextures(textures, range) => {
                let textures = lent(textures);
                enc.encode(
                    on,
                    ComputeCmd::SetTextures {
                        textures: &textures,
                        range: *range,
                    },
                )
            }
            Owned::SetThreadgroupMemoryLength(length, index) => enc.encode(
                on,
                ComputeCmd::SetThreadgroupMemoryLength {
                    length: *length,
                    index: *index,
                },
            ),
            Owned::UseHeaps(heaps) => {
                let heaps = lent(heaps);
                enc.encode(on, ComputeCmd::UseHeaps { heaps: &heaps })
            }
            Owned::UseResources(list, usage) => {
                let list = resources(list);
                enc.encode(
                    on,
                    ComputeCmd::UseResources {
                        resources: &list,
                        usage: *usage,
                    },
                )
            }
        }
    }
}

/// An entry of a loop body's log.
enum Logged {
    Cmd(Owned),
    EndIf,
    EndWhile,
    EndDoWhile(Option<VulkanBuffer>, u64, u64, u32),
    /// The open encoder ended and another of the unit opened, with no bindings.
    Encoder,
}

/// The GPU objects of a command buffer's steps, made at its first.
struct Gpu {
    pipeline: rt::ComputePipelineState,
    /// The counter, then the loops' flags.
    state: rt::Buffer,
    /// The steps, [`CHUNK_STEPS`] to a buffer.
    chunks: Vec<rt::Buffer>,
    steps: u64,
    /// Each flagged loop's End selector and name, by flag.
    loops: Arc<Mutex<Vec<(&'static str, String)>>>,
    /// Whether the completion's overrun check is added.
    checked: bool,
}

/// The step kernel's pipeline of each device, by registry ID: made once.
static PIPELINES: Mutex<Vec<(u64, rt::ComputePipelineState)>> = Mutex::new(Vec::new());

/// The step kernel's pipeline of `device`; for `sel`, whose step it is first made for.
fn pipeline(device: &VulkanDevice, sel: &'static str) -> rt::ComputePipelineState {
    let mut pipelines = PIPELINES.lock();
    if let Some((_, p)) = pipelines.iter().find(|(id, _)| *id == device.registry_id) {
        return p.clone();
    }
    let p = made(sel, || {
        crate::gpu::timing::creation(
            "newComputePipelineStateWithFunction:",
            crate::gpu::timing::DISPATCH_LATER,
            || {
                let library = device.raw.new_library_with_data(METALLIB)?;
                let function = library.new_function_with_name(KERNEL)?;
                device.raw.new_compute_pipeline_state_with_function(
                    &function,
                    &rt::PipelineOptions::default(),
                )
            },
            || format!("kernel {KERNEL:?}, the predication step's, with its library"),
            |made| match made {
                Ok(_) => "made".to_owned(),
                Err(e) => format!("refused: {e}"),
            },
        )
    });
    pipelines.push((device.registry_id, p.clone()));
    p
}

/// A Shared buffer of `length` zero bytes of `device`, for `sel`'s step, outside the Metal address
/// space ([`crate::gpu::own_buffer`]).
fn buffer(device: &VulkanDevice, sel: &'static str, length: u64) -> rt::Buffer {
    made(sel, || {
        let options = resource_options(sel, 0);
        crate::gpu::own_buffer(&device.raw, length, options, rt::BufferContents::Zero)
    })
}

/// The word `at` of `buffer`, a Shared buffer.
fn word(buffer: &rt::Buffer, at: usize) -> u32 {
    let Some(base) = buffer.contents() else {
        return 0;
    };
    // SAFETY: word `at` lies inside the buffer (the callers' indices stay under its length),
    // and its CPU address is live as long as the buffer.
    unsafe { base.as_ptr().cast::<u32>().add(at).read_volatile() }
}

/// The completion's check of `state`'s flags: a loop that ran past its copies ends the process,
/// naming it.
fn overruns(state: &rt::Buffer, loops: &Mutex<Vec<(&'static str, String)>>) {
    let loops = loops.lock();
    for (flag, (sel, name)) in loops.iter().enumerate() {
        if word(state, 1 + flag) != 0 {
            dies(
                sel,
                &format!(
                    "{name} ran past the iterations laid down for it: Metal runs a loop until \
                     its condition fails, and the adapter encodes a fixed number of copies of \
                     its body"
                ),
            );
        }
    }
}

/// A command buffer's command predication: the block structure of its current unit of compute
/// encoders, the open encoder's bindings, and the GPU objects its steps use.
pub struct Predication {
    unit: Mutex<Unit>,
    device: VulkanDevice,
}

impl std::fmt::Debug for Predication {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Predication").finish_non_exhaustive()
    }
}

impl Predication {
    /// A command buffer's predication, of a queue of `device`.
    pub fn new(device: &VulkanDevice) -> Arc<Predication> {
        Arc::new(Predication {
            unit: Mutex::new(Unit::default()),
            device: device.clone_ref(),
        })
    }

    /// Whether a unit of compute encoders is open: one began since its last end, and a compute
    /// encoder after it would go on in it.
    pub(crate) fn in_unit(&self) -> bool {
        self.unit.lock().first_type.is_some()
    }
}

/// The state of a command buffer's current unit, and of its open compute encoder.
#[derive(Default)]
struct Unit {
    stack: Vec<Frame>,
    /// Whether an unbalanced record emptied the unit.
    emptied: bool,
    /// Whether the unit sent guest work: a dispatch or an execution.
    work: bool,
    /// Whether a step ran in the unit: the first resets the counter.
    stepped: bool,
    /// The block records numbered so far, which the names deaths give count.
    records: u32,
    /// What the open loops' bodies hold, from the outermost's start.
    log: Vec<Arc<Logged>>,
    /// The log index of the entry being encoded again, while a body is laid down.
    cursor: Option<usize>,
    /// The flag of each inner loop whose Start the log holds, by its index.
    flags: BTreeMap<usize, usize>,
    /// The open compute encoder's number, and the count numbered so far.
    encoder: Option<u64>,
    encoders: u64,
    bindings: Bindings,
    used: Vec<Use>,
    /// Whether a texture `used` holds lies on a surface whose GPU writes never reach its memory,
    /// or is one whose writes the runtime refuses ([`Unit::reaches_dropped_texels`]), which a
    /// texture answers alike for its life: asked once, as it is made resident.
    used_dropped: bool,
    /// Whether the runtime encoder holds a pipeline the guest did not set.
    foreign: bool,
    /// The dispatch type of the unit's first compute encoder, once it has one.
    first_type: Option<DispatchType>,
    /// Whether an indirect dispatch went through in the unit.
    indirect: bool,
    gpu: Option<Gpu>,
}

impl Unit {
    /// A new unit: the command buffer's GPU objects, and its numbering, stay.
    fn reset(&mut self) {
        self.stack.clear();
        self.emptied = false;
        self.work = false;
        self.stepped = false;
        self.log.clear();
        self.cursor = None;
        self.flags.clear();
        self.first_type = None;
        self.indirect = false;
    }

    /// Whether work sent now is predicated.
    fn predicated(&self) -> bool {
        self.stack.last().is_some_and(|f| f.predicated)
    }

    fn loops(&self) -> usize {
        self.stack
            .iter()
            .filter(|f| !matches!(f.kind, Kind::If { .. }))
            .count()
    }

    /// Whether what the guest sends now goes into the log: a loop is open, and its body is not
    /// being laid down again.
    fn recording(&self) -> bool {
        !self.emptied && self.cursor.is_none() && self.loops() > 0
    }

    /// The name of the block `sel` opens.
    fn name(&mut self, what: &str, sel: &'static str) -> String {
        if self.cursor.is_some() {
            return format!("a copy of a {what} ({sel})");
        }
        self.records += 1;
        format!(
            "the {what} of {sel} (block record {} of its command buffer)",
            self.records
        )
    }

    /// An unbalanced record `sel`: the unit is emptied, as Metal empties it. After the unit's
    /// work it dies: Metal drops that work too, which the adapter cannot take back.
    fn empty(&mut self, sel: &'static str) {
        if self.emptied {
            return;
        }
        if self.work {
            dies(
                sel,
                "an unbalanced block record after its unit of compute encoders sent work: Metal \
                 then drops every dispatch of the unit, the earlier ones too",
            );
        }
        self.emptied = true;
    }

    /// After a block closed: with no loop left open, and no body being laid down, the log goes.
    fn closed(&mut self) {
        if self.cursor.is_none() && self.loops() == 0 {
            self.log.clear();
            self.flags.clear();
        }
    }

    fn gpu(&mut self, predication: &Predication, sel: &'static str) -> &mut Gpu {
        self.gpu.get_or_insert_with(|| Gpu {
            pipeline: pipeline(&predication.device, sel),
            state: buffer(&predication.device, sel, STATE_BYTES),
            chunks: Vec::new(),
            steps: 0,
            loops: Arc::default(),
            checked: false,
        })
    }

    /// A new loop's flag, named `name`, whose End is `end`; past [`FLAGS`] it dies at `sel`.
    fn flag(
        &mut self,
        predication: &Predication,
        sel: &'static str,
        end: &'static str,
        name: &str,
        unroll: u32,
    ) -> usize {
        let gpu = self.gpu(predication, sel);
        let mut loops = gpu.loops.lock();
        if loops.len() == FLAGS {
            dies(
                sel,
                &format!("more than {FLAGS} loops in one command buffer"),
            );
        }
        loops.push((end, format!("{name}, laid down {unroll} times")));
        loops.len() - 1
    }

    /// Encodes `step` in `enc`'s runtime encoder for `sel`, reading `guest` (a buffer and an
    /// offset) where it reads the guest's: barriers around it, then the guest's bindings again.
    /// The step's buffer and offset.
    fn step(
        &mut self,
        predication: &Predication,
        enc: &VulkanComputeEncoder,
        sel: &'static str,
        mut step: Step,
        guest: Option<(&rt::Buffer, u64)>,
    ) -> (rt::Buffer, u64) {
        if !self.stepped {
            step.op |= RESET;
            self.stepped = true;
        }
        let gpu = self.gpu(predication, sel);
        let at = gpu.steps;
        gpu.steps += 1;
        let (chunk, offset) = (at / CHUNK_STEPS, at % CHUNK_STEPS * STEP_BYTES);
        if u64::try_from(gpu.chunks.len()).is_ok_and(|n| n == chunk) {
            gpu.chunks
                .push(buffer(&predication.device, sel, CHUNK_STEPS * STEP_BYTES));
        }
        let Some(chunk) = gpu.chunks.last().cloned() else {
            dies(sel, "a step with no buffer")
        };
        let Some(base) = chunk.contents() else {
            dies(sel, "a step buffer with no CPU address")
        };
        let Ok(at) = usize::try_from(offset) else {
            dies(sel, "a step past the address space")
        };
        // SAFETY: the step's bytes lie inside the chunk (`offset` + `STEP_BYTES` is at most its
        // length, and a `Step` is shorter), whose CPU address is live as long as the chunk; the
        // GPU reads them only once the command buffer is committed, after this write.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const step).cast::<u8>(),
                base.as_ptr().add(at),
                size_of::<Step>(),
            );
        }
        let (state, kernel) = (gpu.state.clone(), gpu.pipeline.clone());
        let bindings = &self.bindings;
        made(sel, || {
            enc.with(sel, |e| {
                let all = BarrierScope::BUFFERS.union(BarrierScope::TEXTURES);
                e.memory_barrier_with_scope(all);
                e.set_compute_pipeline_state(&kernel)?;
                e.set_buffer(Some(&state), 0, 0)?;
                e.set_buffer(Some(&chunk), offset, 1)?;
                let (read, from) = guest.unwrap_or((&state, 0));
                e.set_buffer(Some(read), from, 2)?;
                let one = rt::Size::new(1, 1, 1);
                e.dispatch_threadgroups(one, one)?;
                e.memory_barrier_with_scope(all);
                restore(e, bindings, bindings, true)
            })
        });
        self.foreign = self.bindings.pipeline.is_none();
        (chunk, offset)
    }

    /// A step that tests `cond`.
    fn test(
        &mut self,
        predication: &Predication,
        enc: &VulkanComputeEncoder,
        sel: &'static str,
        op: u32,
        cond: &Cond,
    ) {
        let guest = (&cond.buffer.raw, cond.word);
        self.step(predication, enc, sel, Step::testing(op, cond), Some(guest));
    }

    /// The overrun step of the loop flagged `flag`, and the completion's check, added with the
    /// command buffer's first.
    fn overrun(
        &mut self,
        predication: &Predication,
        enc: &VulkanComputeEncoder,
        sel: &'static str,
        flag: usize,
    ) {
        let mut step = Step::of(OVERRUN);
        step.flag = u32::try_from(flag).unwrap_or(0);
        self.step(predication, enc, sel, step, None);
        let gpu = self.gpu(predication, sel);
        // A command buffer core let go of commits no more, so nothing of it runs on.
        if !gpu.checked
            && let Some(cb) = Obj::upgrade(&enc.command_buffer)
        {
            gpu.checked = true;
            let (state, loops) = (gpu.state.clone(), Arc::clone(&gpu.loops));
            let check: rt::Handler = Box::new(move |_: &rt::CommandBuffer| {
                // A callout on the runtime's thread: marked ours, so a panic here aborts.
                let _env = BlockEnv::foreign();
                overruns(&state, &loops);
            });
            made(sel, || cb.raw.add_completed_handler(check));
        }
    }

    /// Whether a texture the open encoder's next work binds or made resident lies on a surface
    /// whose GPU writes never reach its memory ([`crate::os::gpu_read_only`]). A kernel's stores
    /// into a buffer on one are dropped, as on Metal, and are sent ([`crate::os::dropped_write`]).
    fn reaches_dropped_texels(&self) -> bool {
        self.used_dropped || self.bindings.textures.values().any(drops_texels)
    }

    /// The open encoder made `list` resident with `usage`: kept until the encoder ends, as a fresh
    /// runtime encoder is given it again ([`Unit::refresh`]), with whether one of its textures
    /// drops texels ([`Unit::reaches_dropped_texels`]).
    fn made_resident(&mut self, list: Vec<Resource>, usage: ResourceUsage) {
        self.used_dropped |=
            (list.iter()).any(|r| matches!(r, Resource::Texture(t) if drops_texels(t)));
        self.used.push(Use::Resources(list, usage));
    }

    /// The open encoder ended or went, and what it made resident with it.
    fn let_go(&mut self) {
        self.encoder = None;
        self.bindings = Bindings::default();
        self.used.clear();
        self.used_dropped = false;
    }

    /// What the open encoder's next work may write through textures, `inherits` what an
    /// execution's commands take of the encoder, one entry an object, with whether it does: the
    /// textures bound at slots its kernel writes, by its pipeline's slot writes (every bound
    /// texture where the kernel writes any and they cannot tell; none where the
    /// commands run pipeline states of their own, as Metal makes no state that supports
    /// indirect command buffers of a kernel with a texture argument, measured: the render
    /// encoder's `Reach::at_slots`), and the textures made resident with write usage.
    fn texel_writes(&self, inherits: Option<Inherits>) -> Vec<(Whether, Bound)> {
        let mut written = Vec::new();
        let kernel = match inherits {
            Some(i) if !i.pipeline => None,
            _ => self.bindings.pipeline.as_ref(),
        };
        if inherits.is_none_or(|i| i.pipeline) {
            let writes_textures = kernel.is_some_and(|k| plain(|| k.raw.writes_textures()));
            for (&slot, t) in &self.bindings.textures {
                let whether = match kernel {
                    Some(k) => Whether::Slot(
                        Arc::clone(&k.reflection),
                        Slot {
                            stage: 0,
                            texture: true,
                            slot: u64::from(slot),
                            untold: writes_textures,
                        },
                    ),
                    None => Whether::Writes,
                };
                written.push((whether, texture_bound(t)));
            }
        }
        for u in &self.used {
            if let Use::Resources(list, usage) = u
                && usage.contains(ResourceUsage::WRITE)
            {
                let textures = list.iter().filter_map(|r| match r {
                    Resource::Texture(t) => Some(texture_bound(t)),
                    Resource::Buffer(_) | Resource::Icb(_) => None,
                });
                written.extend(textures.map(|m| (Whether::Writes, m)));
            }
        }
        written.retain(|(_, m)| m.meets);
        written
    }

    /// How the open encoder's next work, `inherits` what an execution's commands take of the
    /// encoder, writes through an object over memory the runtime imports through a writable twin
    /// of read-only pages ([`crate::gpu::writes::kept_write`]), the first found: a texture bound
    /// at a slot its kernel writes, as [`Self::texel_writes`] reads them; a buffer bound at a slot
    /// the kernel writes, by its slot writes (every slot where they cannot tell, and
    /// every bound buffer an execution's commands take where they run pipeline states of their
    /// own); and a texture or buffer made resident with write usage (a heap made resident is
    /// read, as Metal's are). `None` where it writes none, and where no pipeline state is set for
    /// it, which the runtime refuses.
    fn kept_write_through(&self, inherits: Option<Inherits>) -> Option<Whether> {
        use crate::gpu::writes::{buffer_kept, texture_kept};
        let kernel = match inherits {
            Some(i) if !i.pipeline => None,
            _ => Some(self.bindings.pipeline.as_ref()?),
        };
        let at = |texture: bool, slot: u32, untold: bool| match kernel {
            Some(k) => Whether::Slot(
                Arc::clone(&k.reflection),
                Slot {
                    stage: 0,
                    texture,
                    slot: u64::from(slot),
                    untold,
                },
            ),
            None => Whether::Writes,
        };
        let mut written = Vec::new();
        if inherits.is_none_or(|i| i.pipeline) {
            let writes_textures = kernel.is_some_and(|k| plain(|| k.raw.writes_textures()));
            let textures = (self.bindings.textures.iter()).filter(|(_, t)| texture_kept(t));
            written.extend(textures.map(|(&slot, _)| at(true, slot, writes_textures)));
        }
        if inherits.is_none_or(|i| i.buffers) {
            let buffers = (self.bindings.buffers.iter()).filter(|(_, (b, ..))| buffer_kept(b));
            written.extend(buffers.map(|(&slot, _)| at(false, slot, true)));
        }
        for u in &self.used {
            if let Use::Resources(list, usage) = u
                && usage.contains(ResourceUsage::WRITE)
                && list.iter().any(|r| match r {
                    Resource::Texture(t) => texture_kept(t),
                    Resource::Buffer(b) => buffer_kept(b),
                    Resource::Icb(_) => false,
                })
            {
                written.push(Whether::Writes);
            }
        }
        written.into_iter().find(Whether::holds)
    }

    /// The runtime encoder's bindings moved to `to`, which the guest's become.
    fn rebind(&mut self, enc: &VulkanComputeEncoder, sel: &'static str, to: Bindings) {
        made(sel, || {
            enc.with(sel, |e| restore(e, &self.bindings, &to, false))
        });
        self.foreign = match &to.pipeline {
            Some(_) => false,
            None => self.foreign || self.bindings.pipeline.is_some(),
        };
        self.bindings = to;
    }

    /// The open encoder's runtime encoder ended and a fresh one in its place, with the guest's
    /// bindings and residency: the runtime has no unset of a pipeline, and the guest set none.
    fn refresh(&mut self, enc: &VulkanComputeEncoder, sel: &'static str) {
        // A command buffer core let go of commits no more, so nothing encoded here runs.
        let Some(cb) = Obj::upgrade(&enc.command_buffer) else {
            return;
        };
        let held = enc.raw.lock();
        let mut raw = held.borrow_mut();
        let Some(old) = raw.take() else {
            return;
        };
        let dispatch_type = old.dispatch_type();
        made(sel, || old.end_encoding());
        let mut fresh = made(sel, || {
            cb.raw
                .compute_command_encoder_with_dispatch_type(dispatch_type)
        });
        cb.made_encoder(false);
        made(sel, || {
            restore(&mut fresh, &Bindings::default(), &self.bindings, false)?;
            self.used.iter().try_for_each(|u| u.apply(&mut fresh))
        });
        *raw = Some(fresh);
        self.foreign = false;
    }
}

/// Whether `t` lies on a surface whose GPU writes never reach its memory, or is one whose writes
/// the runtime refuses ([`Unit::reaches_dropped_texels`]).
fn drops_texels(t: &VulkanTexture) -> bool {
    t.gpu_read_only() || t.writes_refused()
}

/// `id<MTLComputeCommandEncoder>`'s object: the runtime's encoder, with its command buffer's
/// predication.
#[derive(Debug)]
pub struct PredicatedEncoder {
    /// The runtime's encoder.
    pub encoder: EncoderObject<rt::ComputeCommandEncoder>,
    predication: Arc<Predication>,
    /// Its command buffer, held weakly as the encoder holds it: once the runtime's encoder ended,
    /// an encoder that outlives its command buffer holds no slot of its queue.
    command_buffer: Weak<CommandBufferObject>,
    /// Its number among its command buffer's compute encoders.
    number: u64,
}

impl Deref for PredicatedEncoder {
    type Target = EncoderObject<rt::ComputeCommandEncoder>;
    fn deref(&self) -> &EncoderObject<rt::ComputeCommandEncoder> {
        &self.encoder
    }
}

impl Drop for PredicatedEncoder {
    /// An encoder that goes without `endEncoding` lets go of what it bound and made resident,
    /// as one that ended does.
    fn drop(&mut self) {
        let mut u = self.predication.unit.lock();
        if u.encoder == Some(self.number) {
            u.let_go();
        }
    }
}

/// A compute encoder of `command_buffer`, `raw`: its bindings start empty, and a loop's body
/// open across encoders holds the change.
pub fn encoder(
    raw: rt::ComputeCommandEncoder,
    command_buffer: &VulkanCommandBuffer,
) -> VulkanComputeEncoder {
    let predication = Arc::clone(&command_buffer.predication);
    let dispatch_type = raw.dispatch_type();
    let number = {
        let mut u = predication.unit.lock();
        u.first_type.get_or_insert(dispatch_type);
        if u.recording() {
            u.log.push(Arc::new(Logged::Encoder));
        }
        u.let_go();
        u.encoders += 1;
        u.encoder = Some(u.encoders);
        u.foreign = false;
        u.encoders
    };
    Obj::new(PredicatedEncoder {
        encoder: EncoderObject::new(raw, command_buffer),
        predication,
        command_buffer: command_buffer.downgrade(),
        number,
    })
}

/// The end of `predication`'s unit, at `sel`: a blit or render encoder, an event, or the commit.
/// An If or While still open dies.
pub fn boundary(predication: &Predication, sel: &'static str) {
    let mut u = predication.unit.lock();
    if !u.emptied
        && let Some(open) = u.stack.iter().find(|f| !matches!(f.kind, Kind::DoWhile(_)))
    {
        dies(
            sel,
            &format!(
                "{} is still open where its unit of compute encoders ends: Metal fails the \
                 command buffer there (a GPU recovery) when its condition failed",
                open.name
            ),
        );
    }
    u.reset();
}

/// Dies at `sel` after `endEncoding`, as every command does.
fn open(enc: &VulkanComputeEncoder, sel: &'static str) {
    plain(|| enc.with(sel, |_| ()));
}

/// Every command core sends `enc`, before it is encoded: the bindings it sets, and the log of an
/// open loop's body. A fence inside a block dies.
pub fn observe(enc: &VulkanComputeEncoder, cmd: &ComputeCmd<'_, VulkanPlatform>) {
    let mut u = enc.predication.unit.lock();
    if u.encoder != Some(enc.number) {
        return;
    }
    match cmd {
        // What the encoder bound and made resident goes with it: what the runtime's command
        // buffer holds until it completes is the runtime's to hold.
        ComputeCmd::EndEncoding => {
            u.let_go();
            return;
        }
        ComputeCmd::UpdateFence { .. } | ComputeCmd::WaitForFence { .. }
            if !u.stack.is_empty() || u.emptied =>
        {
            dies(
                cmd.selector(),
                "a fence inside a block of command predication, or in a unit an unbalanced \
                 record emptied, which Metal is not measured at",
            );
        }
        ComputeCmd::SetComputePipelineState { state: Some(_) } => u.foreign = false,
        ComputeCmd::UseResources {
            resources: list,
            usage,
        } => {
            let usage: ResourceUsage = metal_value(cmd.selector(), "resource usage", *usage);
            let list = list.iter().flatten().map(Resource::of).collect();
            u.made_resident(list, usage);
        }
        ComputeCmd::UseHeaps { heaps } => {
            let heaps = heaps.iter().flatten().map(|h| h.clone_ref()).collect();
            u.used.push(Use::Heaps(heaps));
        }
        _ => {}
    }
    if u.recording()
        && let Some(owned) = Owned::of(cmd)
    {
        u.log.push(Arc::new(Logged::Cmd(owned)));
    }
    u.bindings.apply(cmd);
}

/// A dispatch or an indirect command buffer execution, in the runtime's terms: each variant's
/// fields are its selector's arguments, and an execution's what its buffer's commands take of
/// the encoder and the check its host step runs on them, which the one send takes
/// (`crate::gpu::command::executions::Execution::host_step_check`).
#[expect(missing_docs, reason = "the fields are the selector's arguments")]
pub enum Work<'a> {
    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    Threadgroups { groups: MtlSize, threads: MtlSize },
    /// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`.
    ThreadgroupsIndirect {
        buffer: &'a rt::Buffer,
        offset: u64,
        threads: MtlSize,
    },
    /// `dispatchThreads:threadsPerThreadgroup:`.
    Threads {
        threads: MtlSize,
        per_group: MtlSize,
    },
    /// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:`.
    ThreadsIndirect { buffer: &'a rt::Buffer, offset: u64 },
    /// `executeCommandsInBuffer:withRange:`, the range's indices.
    Execute {
        icb: &'a rt::IndirectCommandBuffer,
        range: Range<u64>,
        inherits: Inherits,
        check: Cell<Option<(rt::IndirectCommandCheck, rt::IndirectCommandCheckNeeds)>>,
    },
    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`.
    ExecuteIndirect {
        icb: &'a rt::IndirectCommandBuffer,
        buffer: &'a rt::Buffer,
        offset: u64,
        inherits: Inherits,
        check: Cell<Option<(rt::IndirectCommandCheck, rt::IndirectCommandCheckNeeds)>>,
    },
}

impl Work<'_> {
    /// What an execution's commands take of the encoder; `None` for a dispatch.
    fn inherits(&self) -> Option<Inherits> {
        match *self {
            Work::Execute { inherits, .. } | Work::ExecuteIndirect { inherits, .. } => {
                Some(inherits)
            }
            _ => None,
        }
    }
}

/// The 32-bit words of `s`: each axis modulo `2^32`, as the runtime and Metal take a grid and a
/// threadgroup count.
fn axes(s: MtlSize) -> [u32; 3] {
    [s.width, s.height, s.depth].map(Truncate::low_u32)
}

/// Where `sel` reads `length` bytes of arguments in `buffer`, as the runtime reads them: a
/// dispatch's (`rounds`) at `offset` rounded down to a multiple of 4, as Metal reads them, and an
/// execution's at `offset`, which dies unless it is a multiple of 4. Arguments past the buffer
/// from there die, as the runtime refuses them.
fn arguments_in(
    sel: &'static str,
    buffer: &rt::Buffer,
    offset: u64,
    length: u64,
    rounds: bool,
) -> u64 {
    let size = plain(|| buffer.length());
    let at = if rounds { offset & !3 } else { offset };
    if !at.is_multiple_of(4) || at.checked_add(length).is_none_or(|end| end > size) {
        dies(
            sel,
            &format!(
                "indirect arguments of {length} bytes at {offset:#x} of a buffer of {size:#x} \
                 bytes, which the runtime refuses"
            ),
        );
    }
    at
}

impl Work<'_> {
    /// Whether the runtime takes this work's direct form, with `pipeline`, as running nothing
    /// whatever the condition, or refuses it: a threadgroup of no threads or more than the
    /// pipeline's most, as Metal runs nothing for, or one whose axes past 32 bits change its
    /// size. The direct form then stands for the predicated one.
    fn direct_only(&self, pipeline: Option<&VulkanComputePipeline>) -> bool {
        let (
            Some(p),
            Work::Threadgroups { threads: group, .. }
            | Work::ThreadgroupsIndirect { threads: group, .. }
            | Work::Threads {
                per_group: group, ..
            },
        ) = (pipeline, self)
        else {
            return false;
        };
        let total = group
            .width
            .wrapping_mul(group.height)
            .wrapping_mul(group.depth);
        let limit = plain(|| p.raw.max_total_threads_per_threadgroup());
        let product = axes(*group).iter().map(|&a| u64::from(a)).product::<u64>();
        total == 0 || total > u64::from(limit) || product != total
    }

    /// The direct send.
    fn direct(&self, e: &mut rt::ComputeCommandEncoder) -> Result<(), rt::Error> {
        match *self {
            Work::Threadgroups { groups, threads } => {
                e.dispatch_threadgroups(size(groups), size(threads))
            }
            Work::ThreadgroupsIndirect {
                buffer,
                offset,
                threads,
            } => e.dispatch_threadgroups_indirect(buffer, offset, size(threads)),
            Work::Threads { threads, per_group } => {
                e.dispatch_threads(size(threads), size(per_group))
            }
            Work::ThreadsIndirect { buffer, offset } => e.dispatch_threads_indirect(buffer, offset),
            Work::Execute {
                icb,
                ref range,
                ref check,
                ..
            } => match check.take() {
                Some((check, needs)) => {
                    e.execute_commands_in_buffer_checked(icb, range.clone(), check, needs)
                }
                None => e.execute_commands_in_buffer(icb, range.clone()),
            },
            Work::ExecuteIndirect {
                icb,
                buffer,
                offset,
                ref check,
                ..
            } => match check.take() {
                Some((check, _)) => e.execute_commands_in_buffer_indirect_checked(
                    icb,
                    buffer,
                    offset,
                    check,
                    rt::IndirectCommandCheckNeeds::ALL,
                ),
                None => e.execute_commands_in_buffer_indirect(icb, buffer, offset),
            },
        }
    }

    /// The step that writes the indirect form's arguments, with the guest's buffer it reads;
    /// what the direct form refuses at the encode dies here.
    fn arguments<'b>(&'b self, sel: &'static str) -> (Step, Option<(&'b rt::Buffer, u64)>) {
        match *self {
            Work::Threadgroups { groups, .. } => {
                (Step::arguments(ARGS, &axes(groups), 0b111), None)
            }
            Work::Threads { threads, per_group } => {
                let (grid, group) = (axes(threads), axes(per_group));
                let args = [grid[0], grid[1], grid[2], group[0], group[1], group[2]];
                (Step::arguments(ARGS, &args, 0b111), None)
            }
            Work::ThreadgroupsIndirect { buffer, offset, .. } => {
                let at = arguments_in(sel, buffer, offset, 12, true);
                (
                    Step::arguments(GUEST_ARGS, &[0; 3], 0b111),
                    Some((buffer, at)),
                )
            }
            Work::ThreadsIndirect { buffer, offset } => {
                let at = arguments_in(sel, buffer, offset, 24, true);
                (
                    Step::arguments(GUEST_ARGS, &[0; 6], 0b111),
                    Some((buffer, at)),
                )
            }
            Work::Execute { icb, ref range, .. } => {
                let commands = plain(|| icb.size());
                let (Ok(location), Ok(length)) = (
                    u32::try_from(range.start),
                    u32::try_from(range.end.saturating_sub(range.start)),
                ) else {
                    dies(
                        sel,
                        &format!("the range {range:?}, past an indirect command buffer"),
                    )
                };
                if range.end > commands {
                    dies(
                        sel,
                        &format!(
                            "the range {range:?} of an indirect command buffer of {commands} \
                             commands, which the runtime refuses"
                        ),
                    );
                }
                (Step::arguments(ARGS, &[location, length], 0b10), None)
            }
            Work::ExecuteIndirect { buffer, offset, .. } => {
                arguments_in(sel, buffer, offset, 8, false);
                (
                    Step::arguments(GUEST_ARGS, &[0; 2], 0b10),
                    Some((buffer, offset)),
                )
            }
        }
    }

    /// The indirect form, its arguments at `at` of `chunk`.
    fn indirect(
        &self,
        e: &mut rt::ComputeCommandEncoder,
        chunk: &rt::Buffer,
        at: u64,
    ) -> Result<(), rt::Error> {
        match *self {
            Work::Threadgroups { threads, .. } | Work::ThreadgroupsIndirect { threads, .. } => {
                e.dispatch_threadgroups_indirect(chunk, at, size(threads))
            }
            Work::Threads { .. } | Work::ThreadsIndirect { .. } => {
                e.dispatch_threads_indirect(chunk, at)
            }
            Work::Execute { icb, ref check, .. } | Work::ExecuteIndirect { icb, ref check, .. } => {
                match check.take() {
                    Some((check, _)) => e.execute_commands_in_buffer_indirect_checked(
                        icb,
                        chunk,
                        at,
                        check,
                        rt::IndirectCommandCheckNeeds::ALL,
                    ),
                    None => e.execute_commands_in_buffer_indirect(icb, chunk, at),
                }
            }
        }
    }
}

/// `work`, `sel`: sent as it is outside a block that can skip, as its indirect form inside one,
/// and not at all in a unit an unbalanced record emptied, as Metal drops it there.
pub fn work(enc: &VulkanComputeEncoder, sel: &'static str, work: Work<'_>) -> RaiseResult<()> {
    open(enc, sel);
    let indirect = matches!(
        work,
        Work::ThreadgroupsIndirect { .. } | Work::ThreadsIndirect { .. }
    );
    let dispatch_type = indirect.then(|| plain(|| enc.with(sel, |e| e.dispatch_type())));
    let predication = &*enc.predication;
    let mut u = predication.unit.lock();
    if let Some(dispatch_type) = dispatch_type
        && !u.indirect
    {
        if u.first_type.is_some_and(|first| first != dispatch_type) {
            drop(u);
            metal_aborts(
                sel,
                "a unit of compute encoders' first indirect dispatch, in an encoder of another \
                 dispatch type than the unit's first (a segmentation fault)",
            );
        }
        u.indirect = true;
    }
    if u.emptied {
        return Ok(());
    }
    // An execution of no command runs nothing: it writes and samples nothing.
    let empty = matches!(&work, Work::Execute { range, .. } if range.is_empty());
    // A write through an object over memory the runtime imports through a writable twin of
    // read-only pages dies, at every extent; with no such object, none is looked for.
    if crate::os::TWINS_READ_ONLY
        && !empty
        && crate::gpu::writes::any_kept()
        && let Some(whether) = u.kept_write_through(work.inherits())
    {
        drop(u);
        let what = format!("a dispatch's write through {}, into", whether.through());
        crate::gpu::writes::kept_write(sel, &what);
    }
    // A write of a texture on a surface whose GPU writes never reach its memory dies, and so does
    // one of a 4:4:4 two-plane YCbCr texture whose writes the runtime refuses; a kernel's stores
    // into a buffer on such a surface are sent and dropped, as Metal drops them.
    if !empty && u.reaches_dropped_texels() {
        let written = u.texel_writes(work.inherits());
        let hit = (written.iter()).find(|(w, m)| (m.dropped || m.refused.is_some()) && w.holds());
        if let Some((whether, m)) = hit {
            let (through, refused) = (whether.through(), m.refused.filter(|_| !m.dropped));
            drop(u);
            let what = format!("a dispatch's write through {through}");
            match refused {
                None => crate::os::dropped_write(sel, &format!("{what}, on")),
                Some(pf) => crate::gpu::ycbcr::refused_write(sel, &format!("{what}, into"), pf),
            }
        }
    }
    if u.foreign && u.bindings.pipeline.is_none() {
        u.refresh(enc, sel);
    }
    if matches!(&work, Work::Execute { range, .. } if range.is_empty())
        || work.direct_only(u.bindings.pipeline.as_ref())
    {
        drop(u);
        return send(sel, Policy::Dies, || enc.with(sel, |e| work.direct(e)));
    }
    u.work = true;
    if !u.predicated() {
        drop(u);
        return send(sel, Policy::Dies, || enc.with(sel, |e| work.direct(e)));
    }
    let (step, guest) = work.arguments(sel);
    let (chunk, at) = u.step(predication, enc, sel, step, guest);
    drop(u);
    send(sel, Policy::Dies, || {
        enc.with(sel, |e| work.indirect(e, &chunk, at + OUT))
    })
}

/// `encodeStartIf:offset:comparison:referenceValue:`: a nil buffer opens nothing and empties the
/// unit.
pub fn start_if(
    enc: &VulkanComputeEncoder,
    sel: &'static str,
    buffer: Option<&VulkanBuffer>,
    offset: u64,
    comparison: u64,
    reference: u32,
) -> RaiseResult<()> {
    open(enc, sel);
    let predication = &*enc.predication;
    let mut u = predication.unit.lock();
    let Some(buffer) = buffer else {
        u.empty(sel);
        return Ok(());
    };
    let cond = Cond::new(sel, buffer, offset, comparison, reference);
    let name = u.name("If", sel);
    u.stack.push(Frame {
        kind: Kind::If { orelse: false },
        predicated: true,
        name,
    });
    if !u.emptied {
        u.test(predication, enc, sel, IF, &cond);
    }
    Ok(())
}

/// `encodeStartElse`: with no open If it empties the unit.
pub fn start_else(enc: &VulkanComputeEncoder, sel: &'static str) -> RaiseResult<()> {
    open(enc, sel);
    let predication = &*enc.predication;
    let mut u = predication.unit.lock();
    let emptied = u.emptied;
    match u.stack.last_mut().map(|f| &mut f.kind) {
        Some(Kind::If { orelse }) => {
            if *orelse && !emptied {
                dies(
                    sel,
                    "a second encodeStartElse in one If: Metal then skips the If's work before \
                     it too, which the adapter does not model",
                );
            }
            *orelse = true;
        }
        Some(_) if !emptied => dies(
            sel,
            "an encodeStartElse whose innermost open block is a loop, which Metal is not \
             measured at",
        ),
        Some(_) => return Ok(()),
        None => {
            u.empty(sel);
            return Ok(());
        }
    }
    if !emptied {
        u.step(predication, enc, sel, Step::of(ELSE), None);
    }
    Ok(())
}

/// `encodeEndIf`: YES where it closes an If; with none open it empties the unit, and answers NO.
pub fn end_if(enc: &VulkanComputeEncoder, sel: &'static str) -> RaiseResult<bool> {
    open(enc, sel);
    let predication = &*enc.predication;
    let mut u = predication.unit.lock();
    if u.recording() {
        u.log.push(Arc::new(Logged::EndIf));
    }
    if !matches!(
        u.stack.last(),
        Some(Frame {
            kind: Kind::If { .. },
            ..
        })
    ) {
        u.empty(sel);
        return Ok(false);
    }
    u.stack.pop();
    if !u.emptied {
        u.step(predication, enc, sel, Step::of(END), None);
    }
    u.closed();
    Ok(true)
}

/// A loop's start, `sel`, while or do-while (with no condition), whose End is `end`.
fn start_loop(
    enc: &VulkanComputeEncoder,
    sel: &'static str,
    end: &'static str,
    cond: Option<Cond>,
) {
    let predication = &*enc.predication;
    let mut u = predication.unit.lock();
    if u.loops() > LOOP_DEPTH {
        dies(
            sel,
            &format!(
                "a loop inside {} others, past the {LOOP_DEPTH} the adapter lays down: each \
                 copy of an outer loop's body holds every copy of an inner one's",
                u.loops()
            ),
        );
    }
    let unroll = if u.loops() == 0 {
        UNROLL
    } else {
        NESTED_UNROLL
    };
    let outer = u.predicated();
    let what = if cond.is_some() { "While" } else { "DoWhile" };
    // The Start entry the log took, where it is recording.
    let logged = u.recording();
    if u.cursor.is_none() && !logged {
        u.log.clear();
        u.flags.clear();
    }
    let name = u.name(what, sel);
    let (body, flag) = match u.cursor {
        Some(at) => (at + 1, u.flags.get(&at).copied().unwrap_or(0)),
        None if u.emptied => (u.log.len(), 0),
        None => {
            let flag = u.flag(predication, sel, end, &name, unroll);
            if logged {
                let at = u.log.len().saturating_sub(1);
                u.flags.insert(at, flag);
            }
            (u.log.len(), flag)
        }
    };
    let start = u.bindings.copy();
    let emptied = u.emptied;
    let (kind, predicated) = match cond {
        Some(cond) => {
            if !emptied {
                if outer {
                    u.step(predication, enc, sel, Step::of(ENTER), None);
                }
                u.test(predication, enc, sel, CHECK, &cond);
            }
            let l = Loop {
                body,
                start,
                flag,
                unroll,
                cond: Some(cond),
            };
            (Kind::While(l), true)
        }
        None => {
            if !emptied && outer {
                u.step(predication, enc, sel, Step::of(ENTER), None);
            }
            let l = Loop {
                body,
                start,
                flag,
                unroll,
                cond: None,
            };
            (Kind::DoWhile(l), outer)
        }
    };
    u.stack.push(Frame {
        kind,
        predicated,
        name,
    });
}

/// `encodeStartWhile:offset:comparison:referenceValue:`: a nil buffer opens nothing and empties
/// the unit.
pub fn start_while(
    enc: &VulkanComputeEncoder,
    sel: &'static str,
    buffer: Option<&VulkanBuffer>,
    offset: u64,
    comparison: u64,
    reference: u32,
) -> RaiseResult<()> {
    open(enc, sel);
    let Some(buffer) = buffer else {
        enc.predication.unit.lock().empty(sel);
        return Ok(());
    };
    let cond = Cond::new(sel, buffer, offset, comparison, reference);
    start_loop(enc, sel, "encodeEndWhile", Some(cond));
    Ok(())
}

/// `encodeStartDoWhile`.
pub fn start_do_while(enc: &VulkanComputeEncoder, sel: &'static str) -> RaiseResult<()> {
    open(enc, sel);
    start_loop(
        enc,
        sel,
        "encodeEndDoWhile:offset:comparison:referenceValue:",
        None,
    );
    Ok(())
}

/// Encodes the log's entries `body` again.
fn replay(enc: &VulkanComputeEncoder, sel: &'static str, body: Range<usize>) -> RaiseResult<()> {
    let predication = &*enc.predication;
    for at in body {
        let (entry, before) = {
            let mut u = predication.unit.lock();
            let Some(entry) = u.log.get(at).cloned() else {
                dies(sel, "a loop body past its log")
            };
            (entry, u.cursor.replace(at))
        };
        match &*entry {
            Logged::Cmd(cmd) => cmd.send(enc)?,
            Logged::EndIf => {
                enc.encode_end_if()?;
            }
            Logged::EndWhile => {
                enc.encode_end_while()?;
            }
            Logged::EndDoWhile(buffer, offset, comparison, reference) => {
                enc.encode_end_do_while(buffer.as_ref(), *offset, *comparison, *reference)?;
            }
            Logged::Encoder => {
                predication
                    .unit
                    .lock()
                    .rebind(enc, sel, Bindings::default());
            }
        }
        predication.unit.lock().cursor = before;
    }
    Ok(())
}

/// The rest of the loop on top of the stack, laid down at its End `sel` with `cond`: its body's
/// other copies, each from the bindings the loop started with and, for a While, after its
/// condition's test (a DoWhile's follows each copy), then its overrun step and its end.
fn lay_down(
    enc: &VulkanComputeEncoder,
    sel: &'static str,
    end: usize,
    cond: &Cond,
    before_each: bool,
) -> RaiseResult<()> {
    let predication = &*enc.predication;
    let (body, start, flag, unroll) = {
        let mut u = predication.unit.lock();
        let Some(Frame {
            kind: Kind::While(l) | Kind::DoWhile(l),
            predicated,
            ..
        }) = u.stack.last_mut()
        else {
            dies(sel, "a loop's end with no loop open")
        };
        *predicated = true;
        (l.body, l.start.copy(), l.flag, l.unroll)
    };
    for _ in 1..unroll {
        {
            let mut u = predication.unit.lock();
            u.rebind(enc, sel, start.copy());
            if before_each {
                u.test(predication, enc, sel, CHECK, cond);
            }
        }
        replay(enc, sel, body..end)?;
        if !before_each {
            predication
                .unit
                .lock()
                .test(predication, enc, sel, CHECK, cond);
        }
    }
    let mut u = predication.unit.lock();
    if before_each {
        u.test(predication, enc, sel, CHECK, cond);
    }
    u.overrun(predication, enc, sel, flag);
    u.step(predication, enc, sel, Step::of(END), None);
    u.stack.pop();
    u.closed();
    Ok(())
}

/// `encodeEndWhile`: YES where it closes a While; with none open it empties the unit, and
/// answers NO.
pub fn end_while(enc: &VulkanComputeEncoder, sel: &'static str) -> RaiseResult<bool> {
    open(enc, sel);
    let predication = &*enc.predication;
    let (end, cond) = {
        let mut u = predication.unit.lock();
        let end = u.cursor.unwrap_or(u.log.len());
        if u.recording() {
            u.log.push(Arc::new(Logged::EndWhile));
        }
        let cond = match u.stack.last() {
            Some(Frame {
                kind: Kind::While(l),
                ..
            }) => l.cond.as_ref().map(Cond::copy),
            _ => {
                u.empty(sel);
                return Ok(false);
            }
        };
        if u.emptied {
            u.stack.pop();
            u.closed();
            return Ok(true);
        }
        (end, cond)
    };
    let Some(cond) = cond else {
        dies(sel, "a While with no condition")
    };
    lay_down(enc, sel, end, &cond, true)?;
    Ok(true)
}

/// `encodeEndDoWhile:offset:comparison:referenceValue:`: YES where it closes a DoWhile; with
/// none open it empties the unit, and answers NO. A nil buffer inside an open DoWhile dies.
pub fn end_do_while(
    enc: &VulkanComputeEncoder,
    sel: &'static str,
    buffer: Option<&VulkanBuffer>,
    offset: u64,
    comparison: u64,
    reference: u32,
) -> RaiseResult<bool> {
    open(enc, sel);
    let predication = &*enc.predication;
    let (end, cond) = {
        let mut u = predication.unit.lock();
        let end = u.cursor.unwrap_or(u.log.len());
        let closes = matches!(
            u.stack.last(),
            Some(Frame {
                kind: Kind::DoWhile(_),
                ..
            })
        );
        if closes && buffer.is_none() && !u.emptied {
            dies(
                sel,
                "an encodeEndDoWhile of a nil buffer inside an open DoWhile: Metal then skips \
                 the body and what follows, which the adapter does not model",
            );
        }
        if u.recording() {
            u.log.push(Arc::new(Logged::EndDoWhile(
                buffer.map(Handle::clone_ref),
                offset,
                comparison,
                reference,
            )));
        }
        if !closes {
            u.empty(sel);
            return Ok(false);
        }
        if u.emptied {
            u.stack.pop();
            u.closed();
            return Ok(true);
        }
        let Some(buffer) = buffer else {
            dies(sel, "a nil condition")
        };
        let cond = Cond::new(sel, buffer, offset, comparison, reference);
        u.test(predication, enc, sel, CHECK, &cond);
        (end, cond)
    };
    lay_down(enc, sel, end, &cond, false)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
    use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::event::EventRef;
    use reims_vgpu_core::platform::icb::{
        IcbComputeCmd, IndirectCommandBuffer as _, IndirectComputeCommand as _,
    };
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _};
    use reims_vgpu_core::platform::values::{ComputePassDesc, IcbDesc};
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::command::{QueueObject, VulkanCommandBuffer};
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::ComputePipelineObject;
    use crate::gpu::resource::tests::shared;

    /// `inc_k`: one thread adds 1 to `c[0]`, and marks `c[n]` with `n * 0x11` below 16.
    const INC: &[u8] =
        include_bytes!("../../../../tests/suite/fixtures/shaders/compute_inc.metallib");
    /// `double_gid`: `out[gid] = gid * 2`, so one thread zeroes `out[0]`.
    const DOUBLE: &[u8] =
        include_bytes!("../../../../tests/suite/fixtures/shaders/kernel.metallib");

    /// `MTLCommandBufferStatusCompleted`.
    const COMPLETED: u64 = 4;

    fn kernel(bytes: &[u8], name: &str) -> VulkanComputePipeline {
        let d = dev();
        let f = d
            .new_library(bytes)
            .unwrap()
            .unwrap()
            .unwrap()
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap();
        d.new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    }

    fn command_buffer() -> VulkanCommandBuffer {
        let q = dev().new_command_queue().unwrap();
        q.command_buffer(OnFault::Return).unwrap().unwrap()
    }

    /// A Shared buffer holding `w`.
    fn holding(w: &[u32]) -> VulkanBuffer {
        let b = shared(u64::try_from(w.len() * 4).unwrap());
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

    /// One thread of the pipeline set.
    fn one() -> ComputeCmd<'static, VulkanPlatform> {
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        }
    }

    /// `buffers` at `offsets`, from slot 0.
    fn bind<'a>(
        buffers: &'a [Option<&'a VulkanBuffer>],
        offsets: &'a [u64],
    ) -> ComputeCmd<'a, VulkanPlatform> {
        ComputeCmd::SetBuffers {
            buffers,
            offsets,
            range: NsRange::new(0, u64::try_from(buffers.len()).unwrap()),
        }
    }

    fn start_if(
        b: &VulkanBuffer,
        offset: u64,
        comparison: u64,
        reference: u32,
    ) -> ComputeCmd<'_, VulkanPlatform> {
        ComputeCmd::EncodeStartIf {
            buffer: Some(b),
            offset,
            comparison,
            reference_value: reference,
        }
    }

    fn start_while(
        b: &VulkanBuffer,
        offset: u64,
        comparison: u64,
        reference: u32,
    ) -> ComputeCmd<'_, VulkanPlatform> {
        ComputeCmd::EncodeStartWhile {
            buffer: Some(b),
            offset,
            comparison,
            reference_value: reference,
        }
    }

    /// Sends each of `cmds` to `e`.
    fn encode_all<const N: usize>(
        e: &VulkanComputeEncoder,
        cmds: [ComputeCmd<'_, VulkanPlatform>; N],
    ) {
        for cmd in cmds {
            e.encode(OnFault::Return, cmd).unwrap();
        }
    }

    fn set(p: &VulkanComputePipeline) -> ComputeCmd<'_, VulkanPlatform> {
        ComputeCmd::SetComputePipelineState { state: Some(p) }
    }

    fn compute(cb: &VulkanCommandBuffer) -> VulkanComputeEncoder {
        cb.compute_encoder(OnFault::Return).unwrap().unwrap()
    }

    /// The death of `f`, at `sel`.
    fn dies_at(sel: &str, f: impl FnOnce()) {
        let site = catch_death(f).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    }

    /// Commits `cb` and waits: it completes with no error.
    fn completes(cb: &VulkanCommandBuffer) {
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    }

    /// The step kernel loads, once per device
    #[test]
    fn the_step_kernel_loads() {
        let d = dev();
        let a = pipeline(&d, "test");
        let b = pipeline(&d, "test");
        assert_eq!(
            a.max_total_threads_per_threadgroup(),
            b.max_total_threads_per_threadgroup()
        );
    }

    /// The comparison table: `word OP reference` with the word on the left, unsigned, 2 as `>`,
    /// 5 as `<=` and any other value as `==`; the word is the aligned one at or below the offset,
    /// and the next word plays no part. The If's work runs where it holds, the Else's where not
    #[test]
    fn conditions_compare_as_metal_does() {
        let inc = kernel(INC, "inc_k");
        let cond = holding(&[0x8000_0007, 0xffff_ffff]);
        let rows: [(u64, u64, u32, bool); 16] = [
            (0, 0, 0x8000_0007, true),
            (0, 1, 0x8000_0006, false),
            (1, 2, 0x8000_0007, false),
            (1, 3, 0x8000_0006, true),
            (2, 0, 0x8000_0006, true),
            (2, 0, 0x8000_0007, false),
            (3, 2, 0x8000_0007, true),
            (3, 0, 0x8000_0008, false),
            (4, 0, 0x8000_0008, true),
            (4, 3, 0x8000_0007, false),
            (5, 0, 0x8000_0007, true),
            (5, 1, 0x8000_0006, false),
            (6, 2, 0x8000_0007, true),
            (9, 3, 0x8000_0006, false),
            (0x1_0000_0001, 0, 0x8000_0007, true),
            (u64::MAX, 1, 0x8000_0008, false),
        ];
        let out = shared(64 * 16);
        let cb = command_buffer();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        e.encode(
            OnFault::Return,
            ComputeCmd::SetComputePipelineState { state: Some(&inc) },
        )
        .unwrap();
        for (i, (comparison, offset, reference, _)) in rows.iter().enumerate() {
            let at = 64 * u64::try_from(i).unwrap();
            for cmd in [
                start_if(&cond, *offset, *comparison, *reference),
                bind(&[Some(&out)], &[at]),
                one(),
                ComputeCmd::EncodeStartElse,
                bind(&[Some(&out)], &[at + 32]),
                one(),
            ] {
                e.encode(OnFault::Return, cmd).unwrap();
            }
            assert!(e.encode_end_if().unwrap());
        }
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        completes(&cb);
        let w = words(&out);
        let ran: Vec<(u32, u32)> = (0..16).map(|i| (w[16 * i], w[16 * i + 8])).collect();
        let expected: Vec<(u32, u32)> = rows
            .iter()
            .map(|r| if r.3 { (1, 0) } else { (0, 1) })
            .collect();
        assert_eq!(ran, expected);
    }

    /// compute_loops: a DoWhile while c != 5 runs its body five times, a While while c != 3
    /// three, and none where the word starts at 3; each iteration marks its count
    #[test]
    fn loops_run_until_their_condition_fails() {
        let inc = kernel(INC, "inc_k");
        let (c1, c2, c3) = (shared(64), shared(64), holding(&[3; 16]));
        let cb = command_buffer();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        e.encode(
            OnFault::Return,
            ComputeCmd::SetComputePipelineState { state: Some(&inc) },
        )
        .unwrap();
        for cmd in [
            bind(&[Some(&c1)], &[0]),
            ComputeCmd::EncodeStartDoWhile,
            one(),
            ComputeCmd::MemoryBarrierWithScope { scope: 1 },
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        assert!(e.encode_end_do_while(Some(&c1), 0, 1, 5).unwrap());
        for (c, value) in [(&c2, 3), (&c3, 3)] {
            for cmd in [bind(&[Some(c)], &[0]), start_while(c, 0, 1, value), one()] {
                e.encode(OnFault::Return, cmd).unwrap();
            }
            assert!(e.encode_end_while().unwrap());
        }
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        completes(&cb);
        assert_eq!(words(&c1)[..7], [5, 0x11, 0x22, 0x33, 0x44, 0x55, 0]);
        assert_eq!(words(&c2)[..5], [3, 0x11, 0x22, 0x33, 0]);
        assert_eq!(words(&c3)[..3], [3, 3, 3]);
    }

    /// Nested loops: an outer While (a != 3) whose body counts a, zeroes b and runs an inner
    /// While (b != 2) that counts b and t runs t six times; an If inside a DoWhile runs on the
    /// iteration where it holds
    #[test]
    fn loops_nest_and_hold_ifs() {
        let (inc, zero) = (kernel(INC, "inc_k"), kernel(DOUBLE, "double_gid"));
        let (a, b, t) = (shared(64), shared(64), shared(64));
        let cb = command_buffer();
        let e = compute(&cb);
        encode_all(
            &e,
            [
                start_while(&a, 0, 1, 3),
                set(&inc),
                bind(&[Some(&a)], &[0]),
                one(),
                set(&zero),
                bind(&[Some(&b)], &[0]),
                one(),
                set(&inc),
                start_while(&b, 0, 1, 2),
                bind(&[Some(&b)], &[0]),
                one(),
                bind(&[Some(&t)], &[0]),
                one(),
            ],
        );
        assert!(e.encode_end_while().unwrap());
        assert!(e.encode_end_while().unwrap());
        let (c, d) = (shared(64), shared(64));
        encode_all(
            &e,
            [
                ComputeCmd::EncodeStartDoWhile,
                bind(&[Some(&c)], &[0]),
                one(),
                start_if(&c, 0, 0, 2),
                bind(&[Some(&d)], &[0]),
                one(),
            ],
        );
        assert!(e.encode_end_if().unwrap());
        assert!(e.encode_end_do_while(Some(&c), 0, 1, 4).unwrap());
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        completes(&cb);
        let first = |b: &VulkanBuffer| words(b)[0];
        assert_eq!((first(&a), first(&b), first(&t)), (3, 2, 6));
        assert_eq!((first(&c), first(&d)), (4, 1));
    }

    /// Each copy of a loop's body starts from the bindings the loop started with, as each of
    /// Metal's iterations runs with the bindings its dispatches were encoded with: a body that
    /// moves its buffer's offset and binds another between its dispatches counts each once an
    /// iteration, and after the loop the body's last bindings stand
    #[test]
    fn each_copy_starts_from_the_loops_bindings() {
        let inc = kernel(INC, "inc_k");
        let (first, w) = (shared(128), shared(64));
        let cb = command_buffer();
        let e = compute(&cb);
        encode_all(
            &e,
            [
                set(&inc),
                bind(&[Some(&first)], &[0]),
                ComputeCmd::EncodeStartDoWhile,
                one(),
                ComputeCmd::SetBufferOffset {
                    offset: 64,
                    index: 0,
                },
                one(),
                bind(&[Some(&w)], &[0]),
                one(),
            ],
        );
        assert!(e.encode_end_do_while(Some(&w), 0, 1, 3).unwrap());
        encode_all(&e, [one(), ComputeCmd::EndEncoding]);
        completes(&cb);
        assert_eq!(
            (words(&first)[0], words(&first)[16], words(&w)[0]),
            (3, 3, 4)
        );
    }

    /// Blocks span the compute encoders of a unit: an If opened in one encoder and closed in the
    /// next skips both encoders' work while its condition fails, and the work after it runs; a
    /// DoWhile opened in one encoder loops over both
    #[test]
    fn blocks_span_a_units_encoders() {
        let inc = kernel(INC, "inc_k");
        let cond = holding(&[0]);
        for (reference, ran) in [(0, 1), (1, 0)] {
            let w = shared(192);
            let cb = command_buffer();
            let e = compute(&cb);
            encode_all(
                &e,
                [
                    set(&inc),
                    start_if(&cond, 0, 0, reference),
                    bind(&[Some(&w)], &[0]),
                    one(),
                    ComputeCmd::EndEncoding,
                ],
            );
            let e = compute(&cb);
            encode_all(&e, [set(&inc), bind(&[Some(&w)], &[64]), one()]);
            assert!(e.encode_end_if().unwrap());
            encode_all(
                &e,
                [bind(&[Some(&w)], &[128]), one(), ComputeCmd::EndEncoding],
            );
            completes(&cb);
            let w = words(&w);
            assert_eq!([w[0], w[16], w[32]], [ran, ran, 1], "If (0 == {reference})");
        }
        let (c, d) = (shared(64), shared(64));
        let cb = command_buffer();
        let e = compute(&cb);
        encode_all(
            &e,
            [
                set(&inc),
                bind(&[Some(&c)], &[0]),
                ComputeCmd::EncodeStartDoWhile,
                one(),
                ComputeCmd::EndEncoding,
            ],
        );
        let e = compute(&cb);
        encode_all(&e, [set(&inc), bind(&[Some(&d)], &[0]), one()]);
        assert!(e.encode_end_do_while(Some(&c), 0, 1, 3).unwrap());
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        completes(&cb);
        assert_eq!((words(&c)[0], words(&d)[0]), (3, 3));
    }

    /// compute_unbalanced_jumps: an End with nothing open, an Else with no If and a Start of a nil
    /// buffer empty their unit, whose dispatch then writes nothing, and the End answers NO; a
    /// DoWhile never closed runs its body once
    #[test]
    fn unbalanced_records_empty_their_unit() {
        let inc = kernel(INC, "inc_k");
        type Records = fn(&VulkanComputeEncoder, &VulkanBuffer) -> Option<bool>;
        let passes: [(&str, Records, Option<bool>, u32); 7] = [
            (
                "EndIf",
                |e, _| Some(e.encode_end_if().unwrap()),
                Some(false),
                0,
            ),
            (
                "EndWhile",
                |e, _| Some(e.encode_end_while().unwrap()),
                Some(false),
                0,
            ),
            (
                "EndDoWhile",
                |e, w| Some(e.encode_end_do_while(Some(w), 0x3f00, 1, 5).unwrap()),
                Some(false),
                0,
            ),
            (
                "StartElse",
                |e, _| {
                    encode_all(e, [ComputeCmd::EncodeStartElse]);
                    None
                },
                None,
                0,
            ),
            (
                "StartIf nil",
                |e, _| {
                    let nil = ComputeCmd::EncodeStartIf {
                        buffer: None,
                        offset: 0,
                        comparison: 2,
                        reference_value: 0,
                    };
                    encode_all(e, [nil]);
                    Some(e.encode_end_if().unwrap())
                },
                Some(false),
                0,
            ),
            (
                "StartWhile nil",
                |e, _| {
                    let nil = ComputeCmd::EncodeStartWhile {
                        buffer: None,
                        offset: 0,
                        comparison: 2,
                        reference_value: 1,
                    };
                    encode_all(e, [nil]);
                    Some(e.encode_end_while().unwrap())
                },
                Some(false),
                0,
            ),
            (
                "StartDoWhile, never closed",
                |e, _| {
                    encode_all(e, [ComputeCmd::EncodeStartDoWhile]);
                    None
                },
                None,
                1,
            ),
        ];
        for (label, records, answer, count) in passes {
            let w = shared(64);
            let cb = command_buffer();
            let e = compute(&cb);
            encode_all(&e, [set(&inc), bind(&[Some(&w)], &[0])]);
            let answered = records(&e, &w);
            encode_all(&e, [one(), ComputeCmd::EndEncoding]);
            let e = compute(&cb);
            encode_all(
                &e,
                [
                    set(&inc),
                    bind(&[Some(&w)], &[32]),
                    one(),
                    ComputeCmd::EndEncoding,
                ],
            );
            completes(&cb);
            let w = words(&w);
            assert_eq!((answered, w[0], w[8]), (answer, count, count), "{label}");
        }
    }

    /// An unbalanced record after its unit's work dies at its selector: Metal would drop that
    /// work too
    #[test]
    fn emptying_after_work_dies() {
        let inc = kernel(INC, "inc_k");
        let w = shared(64);
        let cb = command_buffer();
        let e = compute(&cb);
        encode_all(
            &e,
            [
                set(&inc),
                bind(&[Some(&w)], &[0]),
                one(),
                ComputeCmd::EndEncoding,
            ],
        );
        let e = compute(&cb);
        dies_at("encodeEndIf", || {
            e.encode_end_if().unwrap();
        });
    }

    /// An If or While still open where its unit ends dies there, at a blit encoder, an event
    /// and the commit; a DoWhile left open does not
    #[test]
    fn an_if_or_while_open_at_a_units_end_dies_there() {
        let cond = holding(&[0]);
        let event = dev().new_shared_event().unwrap();
        type End = fn(&VulkanCommandBuffer, &crate::gpu::event::VulkanSharedEvent);
        let ends: [(&str, End); 4] = [
            ("blitCommandEncoder", |cb, _| {
                cb.blit_encoder(OnFault::Return).unwrap();
            }),
            ("encodeSignalEvent:value:", |cb, ev| {
                cb.encode_signal(Some(EventRef::Shared(ev)), 1).unwrap();
            }),
            ("encodeWaitForEvent:value:", |cb, ev| {
                cb.encode_wait(Some(EventRef::Shared(ev)), 0).unwrap();
            }),
            ("commit", |cb, _| cb.commit(OnFault::Return).unwrap()),
        ];
        for (i, (sel, end)) in ends.into_iter().enumerate() {
            let cb = command_buffer();
            let e = compute(&cb);
            let start = if i % 2 == 0 {
                start_if(&cond, 0, 0, 0)
            } else {
                start_while(&cond, 0, 1, 0)
            };
            encode_all(&e, [start, ComputeCmd::EndEncoding]);
            dies_at(sel, || end(&cb, &event));
        }
        let cb = command_buffer();
        let e = compute(&cb);
        encode_all(
            &e,
            [ComputeCmd::EncodeStartDoWhile, ComputeCmd::EndEncoding],
        );
        completes(&cb);
    }

    /// What the adapter has no form for dies at the record that shows it: a second Else in one If,
    /// an Else inside a loop, a nil EndDoWhile inside a DoWhile, a fence inside a block, a loop
    /// inside two others, and a condition word past its buffer
    #[test]
    fn shapes_without_a_form_die_at_their_record() {
        let cond = holding(&[1, 0]);
        let fence = dev().new_fence().unwrap().unwrap();
        type Records = fn(&VulkanComputeEncoder, &VulkanBuffer, &crate::gpu::resource::VulkanFence);
        let shapes: [(&str, Records); 6] = [
            ("encodeStartElse", |e, c, _| {
                encode_all(
                    e,
                    [
                        start_if(c, 0, 0, 1),
                        ComputeCmd::EncodeStartElse,
                        ComputeCmd::EncodeStartElse,
                    ],
                );
            }),
            ("encodeStartElse", |e, _, _| {
                encode_all(
                    e,
                    [ComputeCmd::EncodeStartDoWhile, ComputeCmd::EncodeStartElse],
                );
            }),
            (
                "encodeEndDoWhile:offset:comparison:referenceValue:",
                |e, _, _| {
                    encode_all(e, [ComputeCmd::EncodeStartDoWhile]);
                    e.encode_end_do_while(None, 0, 0, 0).unwrap();
                },
            ),
            ("updateFence:", |e, c, f| {
                encode_all(
                    e,
                    [start_if(c, 0, 0, 1), ComputeCmd::UpdateFence { fence: f }],
                );
            }),
            ("encodeStartDoWhile", |e, c, _| {
                encode_all(
                    e,
                    [
                        start_while(c, 0, 1, 0),
                        ComputeCmd::EncodeStartDoWhile,
                        ComputeCmd::EncodeStartDoWhile,
                    ],
                );
            }),
            (
                "encodeStartIf:offset:comparison:referenceValue:",
                |e, c, _| {
                    encode_all(e, [start_if(c, 6, 0, 1), start_if(c, 8, 0, 1)]);
                },
            ),
        ];
        for (sel, records) in shapes {
            let cb = command_buffer();
            let e = compute(&cb);
            dies_at(sel, || records(&e, &cond, &fence));
        }
    }

    /// Every dispatch form is predicated: a false If runs none of the four, a true one runs each,
    /// the indirect ones with the guest's arguments, the threads form's at an offset read
    /// rounded down to a multiple of 4
    #[test]
    fn every_dispatch_form_is_predicated() {
        let inc = kernel(INC, "inc_k");
        let args = holding(&[1, 1, 1, 0, 1, 1, 1, 1, 1, 1]);
        let cond = holding(&[0]);
        let group = MtlSize::new(1, 1, 1);
        for (reference, ran) in [(0, 1), (1, 0)] {
            let w = shared(256);
            let cb = command_buffer();
            let e = compute(&cb);
            encode_all(
                &e,
                [
                    set(&inc),
                    start_if(&cond, 0, 0, reference),
                    bind(&[Some(&w)], &[0]),
                    ComputeCmd::DispatchThreadgroups {
                        threadgroups_per_grid: group,
                        threads_per_threadgroup: group,
                    },
                    bind(&[Some(&w)], &[64]),
                    ComputeCmd::DispatchThreadgroupsIndirect {
                        indirect_buffer: Some(&args),
                        indirect_buffer_offset: 0,
                        threads_per_threadgroup: group,
                    },
                    bind(&[Some(&w)], &[128]),
                    one(),
                    bind(&[Some(&w)], &[192]),
                    ComputeCmd::DispatchThreadsIndirect {
                        indirect_buffer: Some(&args),
                        indirect_buffer_offset: 18,
                    },
                ],
            );
            assert!(e.encode_end_if().unwrap());
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            completes(&cb);
            let w = words(&w);
            assert_eq!(
                [w[0], w[16], w[32], w[48]],
                [ran; 4],
                "If (0 == {reference})"
            );
        }
    }

    /// A predicated dispatch takes its sizes as its direct form does: a grid or threadgroup count
    /// past 32 bits modulo `2^32`, and a threadgroup of more threads than the pipeline's most as
    /// running nothing
    #[test]
    fn predicated_sizes_are_taken_as_the_direct_forms_take_them() {
        let inc = kernel(INC, "inc_k");
        let cond = holding(&[0]);
        let unit = MtlSize::new(1, 1, 1);
        let wide = MtlSize::new((1 << 32) + 1, 1, 1);
        let w = shared(256);
        let cb = command_buffer();
        let e = compute(&cb);
        encode_all(
            &e,
            [
                set(&inc),
                start_if(&cond, 0, 0, 0),
                bind(&[Some(&w)], &[0]),
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid: wide,
                    threads_per_threadgroup: unit,
                },
                bind(&[Some(&w)], &[64]),
                ComputeCmd::DispatchThreads {
                    threads_per_grid: wide,
                    threads_per_threadgroup: unit,
                },
                bind(&[Some(&w)], &[128]),
                ComputeCmd::DispatchThreads {
                    threads_per_grid: unit,
                    threads_per_threadgroup: MtlSize::new(1 << 16, 1, 1),
                },
            ],
        );
        assert!(e.encode_end_if().unwrap());
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        completes(&cb);
        let w = words(&w);
        assert_eq!([w[0], w[16], w[32]], [1, 1, 0]);
    }

    /// `inc_k`'s pipeline with indirect command buffer support.
    fn icb_kernel() -> VulkanComputePipeline {
        let d = dev();
        let library = d.raw.new_library_with_data(INC).unwrap();
        let function = library.new_function_with_name("inc_k").unwrap();
        let mut desc =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        desc.support_indirect_command_buffers = true;
        let raw = d
            .raw
            .new_compute_pipeline_state(&desc, &rt::PipelineOptions::default())
            .unwrap();
        Obj::new(ComputePipelineObject::new(raw))
    }

    /// Executions are predicated, direct and indirect, in an encoder whose guest set no pipeline:
    /// the steps' pipeline never stands for one, so an execution of commands of their own
    /// pipeline runs them where the If holds, and none where it fails
    #[test]
    fn executions_are_predicated_with_no_pipeline_set() {
        let kernel = icb_kernel();
        let cond = holding(&[0]);
        let range = holding(&[0, 1]);
        let desc = IcbDesc {
            command_types: 0x40,
            max_kernel_buffer_bind_count: 1,
            ..IcbDesc::default()
        };
        for (reference, ran) in [(0, 2), (1, 0)] {
            let w = shared(64);
            let icb = dev()
                .new_indirect_command_buffer(&desc, 1, 0)
                .unwrap()
                .unwrap();
            let c = icb.compute_command(0).unwrap().unwrap();
            for cmd in [
                IcbComputeCmd::SetComputePipelineState {
                    pipeline_state: Some(&kernel),
                },
                IcbComputeCmd::SetKernelBuffer {
                    buffer: Some(&w),
                    offset: 0,
                    index: 0,
                },
                IcbComputeCmd::ConcurrentDispatchThreads {
                    threads_per_grid: MtlSize::new(1, 1, 1),
                    threads_per_threadgroup: MtlSize::new(1, 1, 1),
                },
            ] {
                c.encode(cmd).unwrap();
            }
            let cb = command_buffer();
            let e = compute(&cb);
            encode_all(
                &e,
                [
                    ComputeCmd::UseResources {
                        resources: &[Some(ResourceRef::Buffer(&w))],
                        usage: 3,
                    },
                    start_if(&cond, 0, 0, reference),
                    ComputeCmd::ExecuteCommandsInBuffer {
                        indirect_command_buffer: &icb,
                        execution_range: NsRange::new(0, 1),
                    },
                    ComputeCmd::MemoryBarrierWithScope { scope: 1 },
                    ComputeCmd::ExecuteCommandsIndirect {
                        indirect_commandbuffer: &icb,
                        indirect_range_buffer: &range,
                        indirect_buffer_offset: 0,
                    },
                ],
            );
            assert!(e.encode_end_if().unwrap());
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            completes(&cb);
            assert_eq!(words(&w)[0], ran, "If (0 == {reference})");
        }
    }

    /// A dispatch with no pipeline set dies at its selector after a step, as the runtime refuses
    /// it before any
    #[test]
    fn a_dispatch_with_no_pipeline_dies_after_a_step() {
        let cond = holding(&[0]);
        let cb = command_buffer();
        let e = compute(&cb);
        encode_all(&e, [start_if(&cond, 0, 0, 0)]);
        assert!(e.encode_end_if().unwrap());
        dies_at("dispatchThreads:threadsPerThreadgroup:", || {
            encode_all(&e, [one()]);
        });
    }

    /// A concurrent encoder's work is predicated as a serial one's
    #[test]
    fn a_concurrent_encoder_is_predicated() {
        let inc = kernel(INC, "inc_k");
        let cond = holding(&[0]);
        for (reference, ran) in [(0, 1), (1, 0)] {
            let w = shared(64);
            let cb = command_buffer();
            let pass = ComputePassDesc {
                dispatch_type: Some(1),
            };
            let e = cb.compute_encoder_for_pass(&pass).unwrap().unwrap();
            encode_all(
                &e,
                [
                    set(&inc),
                    bind(&[Some(&w)], &[0]),
                    start_if(&cond, 0, 0, reference),
                    one(),
                ],
            );
            assert!(e.encode_end_if().unwrap());
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            completes(&cb);
            assert_eq!(words(&w)[0], ran);
        }
    }

    /// An ended encoder that outlives its command buffer holds none of its queue's slots: the
    /// runtime's command buffer goes with core's, uncommitted, so the queue's next one is made
    #[test]
    fn an_ended_encoder_outliving_its_command_buffer_holds_no_slot() {
        let d = dev();
        let raw = d
            .raw
            .new_command_queue_with_max_command_buffer_count(1)
            .unwrap();
        let q = Obj::new(QueueObject::new(raw, d.clone_ref()));
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = compute(&cb);
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        drop(cb);
        let (made, next) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
                made.send(()).unwrap();
                drop(cb);
            });
            let waited = next
                .recv_timeout(std::time::Duration::from_secs(10))
                .is_err();
            // The encoder's release lets a waiting command buffer through, so the scope ends.
            drop(e);
            assert!(
                !waited,
                "the queue's next command buffer waited for the slot of one core let go of"
            );
        });
    }

    reims_vgpu_core::death_test!(
        /// A loop still running after its iterations laid down ends the process at the command
        /// buffer's completion, at its End's selector: Metal would run it on
        a_loop_past_its_copies_dies_at_completion,
        |_child| {
            let inc = kernel(INC, "inc_k");
            let c = shared(64);
            let cb = command_buffer();
            let e = compute(&cb);
            encode_all(
                &e,
                [
                    set(&inc),
                    bind(&[Some(&c)], &[0]),
                    start_while(&c, 0, 1, 1000),
                    one(),
                ],
            );
            e.encode_end_while().unwrap();
            e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
        },
        |report| {
            report.assert_died(
                DeathId::Platform {
                    kind: DeathKind::Abort,
                    call: "encodeEndWhile",
                }
                .site(),
            );
        }
    );
}
