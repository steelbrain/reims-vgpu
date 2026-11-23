//! The rt platform's compute pipeline states ([`RtComputePipeline`]): rt's, made of a function
//! (`newComputePipelineStateWithFunction:error:`) or of core's whole descriptor
//! (`newComputePipelineStateWithDescriptor:options:reflection:error:`), with the sampler indices
//! their samplers take ([`Numbering`]).
//!
//! - **The descriptor** is rt's of core's values: the kernel as its function was made
//!   (`super::library`), the linked functions as theirs were, the stage-input descriptor, and the
//!   descriptor's numbers, a value past rt's word held at the word's maximum, which rt refuses as
//!   Metal refuses a value past its limit. rt refuses a state for indirect command buffers
//!   (`rt-refused-gap`). Every refusal is the creator's `NSError`, counted under its kind. A
//!   kernel made unspecialized (`newFunctionWithName:`) that declares function constants is
//!   refused before rt compiles it (`rt-refused-metal`): Metal's creation ends the process at it.
//! - **Its samplers' indices.** rt numbers the samplers a state samples through as Metal's
//!   creation does: its constexpr samplers, then one for a texture read and one for a texture
//!   write, and takes an index of a live sampler grant for each. A sampler equal to one a live
//!   state holds takes its index, as Metal's numbering does, and any other an index of the
//!   platform's own (`super::texture`'s `OwnIndices`, the sampler states' own, which never takes
//!   an index the guest reserved). rt places a constexpr sampler's device samplers at its index;
//!   the read and write samplers take nothing there. A state gives its indices back as it goes,
//!   after rt's state: an index whose last holder goes is the platform's again, taken once every
//!   command buffer committed before then has completed, as any of its own
//!   (`super::texture`'s `OwnIndices::give_back`). Every command buffer that set the state keeps
//!   it until it completes (`super::command`'s `Kept`). Where none is free, the creation is
//!   refused (`rt-refused-limit`).
//! - **Its answers.** `maxTotalThreadsPerThreadgroup`, `staticThreadgroupMemoryLength`,
//!   `threadExecutionWidth` (32, the SIMD-group width rt runs every kernel's SIMD groups at),
//!   `supportIndirectCommandBuffers` (NO: rt refuses a compute state for indirect command
//!   buffers, `Gap::IndirectCommandBuffers` of kind `ComputeCommands`) and
//!   `imageblockMemoryLengthForDimensions:` are rt's. rt answers the last 0 for a kernel that
//!   declares no imageblock; for one that does, it refuses (`Gap::ImageblockLength`), and the
//!   answer is 0, counted (`rt-refused-gap`). `gpuResourceID` and `resourceIndex` are 0, as
//!   Metal's are for a state without indirect command buffer support; and `uniqueIdentifier` is
//!   one of a process-wide count from 1, where Metal's is an address that differs from run to
//!   run. Function handles and visible function tables are rt's (`super::function_table`).
//! - **Its slots' writes.** Whether the kernel may write the texture bound at a slot is rt's
//!   interface of it (`texture_accesses`): a slot it writes, or reads and writes, as its
//!   operations do, not as its argument's `access::` qualifier says. Every other slot it only
//!   samples, reads or queries, or does not use. The state keeps the written slots as it is made.
//! - **Its dispatches made ahead.** rt's make of the state lowers every variant a dispatch can
//!   run, but a kernel's `VkPipeline` depends on the dispatch's threadgroup size, threadgroup
//!   memory lengths and grid tiles, so rt makes it at the dispatch's first use of them;
//!   `prepare_dispatch` is rt's make of it ahead, at the size the dispatch gives, on the caller's
//!   thread, for the device's pre-pass of a committed command buffer (reims-vgpu's
//!   `graphics::decode::prepass`). It lowers nothing. A dispatch that needs what such a call is
//!   making waits for that call alone, and gets every result and error of its own. The pre-pass
//!   makes each pipeline once by rt's key of it (`dispatch_variant`), of the compiled content:
//!   the states of one content share its `VkPipeline`s, and answer one key for one dispatch. rt
//!   answers with the key whether its `VkPipeline` is made, so the pre-pass hands none made by
//!   the state's finish or by an earlier make.
//! - **Its main pipeline.** rt's make of the state makes everything up to the driver's create of
//!   the state's main `VkPipeline` (a tiling kernel's tiles', an image-only kernel's square
//!   workgroups', or an observing kernel's whole and ragged grids' at its
//!   `requiredThreadsPerThreadgroup`, or at `[N, 1, 1]` for the bound `N` of threads its state
//!   declares), and answers. The state's finish ([`ComputePipeline::finish`]) makes that
//!   pipeline, rt's `finish_pipeline`, on the thread that made the state: the device's compile
//!   thread, after the create's reply and before its store (reims-vgpu's `graphics::compile`,
//!   "Order"). rt counts what the finishes made and their time (`Metrics::finished_pipelines`,
//!   `finished_pipeline_nanos`). A finish rt fails, the driver's create, keeps nothing: the
//!   state's first dispatch, or a make ahead of it, makes the pipeline again, the dispatch
//!   waiting for the make (`Metrics::first_use_unfinished_waits`), and fails as a dispatch rt
//!   refuses where that make fails too.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use metal2vulkan::Budget;
use metal2vulkan::metal::{AttributeFormat, IndexType, StepFunction};
use metal2vulkan::pipeline::{
    LinkedFunction, LinkedFunctions, StageEntries, StageInputAttributeDescriptor,
    StageInputDescriptor, StageInputLayoutDescriptor,
};
use metal2vulkan_runtime as rt;

use crate::core::platform::compute::Dispatch;
use crate::core::platform::resource::ComputePipeline;
use crate::core::platform::stage::{self, Site, StageFills};
use crate::core::platform::types::{MtlSize, NsErrorText};
use crate::core::platform::values::{FullComputePipelineDesc, LinkedFunctionsDesc, StageInputDesc};
use crate::core::platform::{Handle, Platform};
use crate::core::raise::RaiseResult;
use crate::core::sync::Mutex;
use crate::core::trace;

use super::device::RtDevice;
use super::function_table::{RtFunctionHandle, RtFunctionTable};
use super::library::RtFunction;
use super::stage::{StageKey, UNFILLED_AT_MAKE, counted, stage_of};
use super::state::value;
use super::texture::OwnIndices;
use super::unbuilt::{
    RtRefusal, naming_functions, refused_error_as, rt_refused_create, rt_refused_error,
    rt_refused_imageblock, stage_function, translation,
};
use super::{RtObj, RtPlatform};

/// `newComputePipelineStateWithFunction:error:`.
pub const WITH_FUNCTION: &str = "newComputePipelineStateWithFunction:error:";

/// `newComputePipelineStateWithDescriptor:options:reflection:error:`.
pub const WITH_DESCRIPTOR: &str = "newComputePipelineStateWithDescriptor:options:reflection:error:";

/// `imageblockMemoryLengthForDimensions:`, a compute or render state's.
pub(super) const IMAGEBLOCK_MEMORY_LENGTH: &str = "imageblockMemoryLengthForDimensions:";

/// The dimensions `size` as rt takes them, each past 32 bits held at the word's maximum.
pub(super) fn dimensions(size: MtlSize) -> [u32; 3] {
    [size.width, size.height, size.depth].map(|n| u32::try_from(n).unwrap_or(u32::MAX))
}

/// `MTLBufferLayoutStrideDynamic`: a layout whose stride the encoder binds the buffer with.
const STRIDE_DYNAMIC: u64 = u64::MAX;

/// The next `uniqueIdentifier` a state takes.
static NEXT_IDENTIFIER: AtomicU64 = AtomicU64::new(1);

/// The sampler indices the live pipeline states' samplers hold (the module's docs): each sampler
/// a live state numbers, with its index and how many of the live states' samplers hold it.
#[derive(Default)]
pub struct Numbering {
    held: Mutex<HashMap<rt::NumberedSampler, Held>>,
}

/// A numbered sampler's index and its holders.
struct Held {
    index: u32,
    holders: usize,
}

impl Numbering {
    /// How many indices live states' samplers hold.
    pub fn held(&self) -> usize {
        self.held.lock().len()
    }

    /// An index for each of `samplers`, in their order, held until [`give_back`](Self::give_back)
    /// of the same samplers: a sampler equal to one held takes its index, and any other one of
    /// `own`, granted on `device` where it is new. `Ok(None)` where none of `own` is free for one;
    /// on `None` or `Err` it holds nothing.
    fn take(
        &self,
        own: &OwnIndices,
        device: &rt::Device,
        samplers: &[rt::NumberedSampler],
    ) -> Result<Option<Vec<u32>>, rt::Error> {
        let mut held = self.held.lock();
        let mut indices = Vec::with_capacity(samplers.len());
        for (at, sampler) in samplers.iter().enumerate() {
            if let Some(h) = held.get_mut(sampler) {
                h.holders = h.holders.saturating_add(1);
                indices.push(h.index);
                continue;
            }
            let taken = own.take(device);
            let Ok(Some(index)) = taken else {
                give_back(&mut held, own, samplers.get(..at).unwrap_or_default());
                return taken.map(|_| None);
            };
            held.insert(*sampler, Held { index, holders: 1 });
            indices.push(index);
        }
        Ok(Some(indices))
    }

    /// Gives back one hold of each of `samplers`: an index whose last holder goes is `own`'s.
    fn give_back(&self, own: &OwnIndices, samplers: &[rt::NumberedSampler]) {
        give_back(&mut self.held.lock(), own, samplers);
    }
}

/// [`Numbering::give_back`] under its lock.
fn give_back(
    held: &mut HashMap<rt::NumberedSampler, Held>,
    own: &OwnIndices,
    samplers: &[rt::NumberedSampler],
) {
    for sampler in samplers {
        let Some(h) = held.get_mut(sampler) else {
            continue;
        };
        h.holders = h.holders.saturating_sub(1);
        if h.holders == 0 {
            own.give_back(h.index);
            held.remove(sampler);
        }
    }
}

/// The samplers a state holds indices for, given back at its drop: held after rt's state, which
/// gives up a constexpr sampler's device samplers before its index is the platform's again.
pub(super) struct Lease {
    device: RtObj<RtDevice>,
    samplers: Vec<rt::NumberedSampler>,
    /// Each one's index.
    indices: Vec<u32>,
}

impl Lease {
    /// The samplers, in rt's order.
    pub(super) fn samplers(&self) -> &[rt::NumberedSampler] {
        &self.samplers
    }

    /// The index each of [`samplers`](Self::samplers) holds.
    pub(super) fn indices(&self) -> &[u32] {
        &self.indices
    }
}

impl Drop for Lease {
    /// Counts a pipeline state's release where a completed handler's run is timed, as each holds
    /// one lease (`super::handler_times`), then gives back its samplers' indices.
    fn drop(&mut self) {
        super::handler_times::released("pipeline-state", || 0);
        let device = &self.device;
        (device.numbering()).give_back(device.sampler_indices(), &self.samplers);
    }
}

/// `id<MTLComputePipelineState>` on metal2vulkan-runtime.
pub struct RtComputePipeline {
    /// rt's state. Declared before `lease`, so it goes first: rt gives up a constexpr sampler's
    /// device samplers before its index is the platform's again.
    rt: rt::ComputePipelineState,
    /// Its samplers' indices.
    lease: Lease,
    /// Its `uniqueIdentifier`.
    unique_identifier: u64,
    /// The texture slots the kernel may write ([`written_slots`]).
    written: Box<[u32]>,
    /// Its device's stage fills, which its makes ahead start.
    fills: Arc<StageFills<StageKey>>,
    /// Its kernel, which the first line of each kind of rt's refusals of its dispatches names
    /// (`super::compute`).
    kernel: Option<metal2vulkan::library::Function>,
}

impl RtComputePipeline {
    /// rt's state.
    pub fn rt(&self) -> &rt::ComputePipelineState {
        &self.rt
    }

    /// Its kernel (`super::unbuilt::stage_function`).
    pub(super) fn kernel(&self) -> Option<&metal2vulkan::library::Function> {
        self.kernel.as_ref()
    }

    /// The samplers it numbers, in rt's order.
    pub fn samplers(&self) -> &[rt::NumberedSampler] {
        self.lease.samplers()
    }

    /// The index each of [`samplers`](Self::samplers) holds.
    pub fn sampler_indices(&self) -> &[u32] {
        self.lease.indices()
    }
}

/// The state of `function` alone, as `newComputePipelineStateWithFunction:error:` makes it, with
/// Metal's defaults elsewhere.
pub fn of_function(
    device: &RtObj<RtDevice>,
    function: &RtObj<RtFunction>,
) -> Result<RtObj<RtComputePipeline>, NsErrorText> {
    let stage = function.stage();
    specialized(WITH_FUNCTION, &stage)?;
    let descriptor = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(stage));
    make(device, WITH_FUNCTION, &descriptor)
}

/// The state of core's whole descriptor `desc` (the module's docs).
pub fn of_descriptor(
    device: &RtObj<RtDevice>,
    desc: &FullComputePipelineDesc<'_, RtPlatform>,
) -> Result<RtObj<RtComputePipeline>, NsErrorText> {
    if let Some(library) = desc.libraries.preloaded.first() {
        library.unreachable()
    }
    if let Some((_, library)) = desc.libraries.loaded.first() {
        library.unreachable()
    }
    let mut stage = desc.function.stage();
    specialized(WITH_DESCRIPTOR, &stage)?;
    stage.linked = linked_functions(&desc.linked);
    let mut d = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(stage));
    if let Some(input) = &desc.stage_input {
        match stage_input(input) {
            Ok(input) => d.stage_input = Some(input),
            Err(why) => return Err(refused_error_as(WITH_DESCRIPTOR, RtRefusal::Refused, &why)),
        }
    }
    d.max_total_threads_per_threadgroup = (desc.max_total_threads_per_threadgroup != 0)
        .then(|| word(desc.max_total_threads_per_threadgroup));
    d.threadgroup_size_is_multiple_of_execution_width =
        desc.threadgroup_size_is_multiple_of_execution_width;
    d.support_indirect_command_buffers = desc.support_indirect_command_buffers;
    d.max_call_stack_depth = word(desc.max_call_stack_depth);
    d.open_cl_mode_enabled = desc.open_cl_mode;
    make(device, WITH_DESCRIPTOR, &d)
}

/// Whether Metal makes a state of `stage` by the creator `sel`: none of a function made
/// unspecialized that declares function constants (`FunctionDescriptor::needs_specialization`),
/// whose creation ends the process on Metal. `Err` is the creator's `NSError`, counted: Metal's
/// refusal, or the translator's of the function's constants. Neither names the function, whose
/// name is the guest's.
pub(super) fn specialized(
    sel: &'static str,
    stage: &rt::FunctionDescriptor,
) -> Result<(), NsErrorText> {
    match stage.needs_specialization(&Budget::standard(), &StageEntries::default()) {
        Ok(false) => Ok(()),
        Ok(true) => {
            let why = "a function that declares function constants was made unspecialized";
            Err(refused_error_as(sel, RtRefusal::Refused, &why))
        }
        Err(error) => Err(rt_refused_error(sel, &translation(error))),
    }
}

/// `v` as rt's word, held at the word's maximum.
fn word(v: u64) -> u32 {
    u32::try_from(v).unwrap_or(u32::MAX)
}

/// The translator's linked functions of `linked`, each as it was made.
pub(super) fn linked_functions(linked: &LinkedFunctionsDesc<RtPlatform>) -> LinkedFunctions {
    fn each(functions: &[RtObj<RtFunction>]) -> Vec<LinkedFunction> {
        (functions.iter())
            .map(|function| function.linked().clone())
            .collect()
    }
    let mut out = LinkedFunctions::new();
    out.functions = each(&linked.functions);
    out.binary_functions = each(&linked.binary_functions);
    out.private_functions = each(&linked.private_functions);
    out.groups = (linked.groups.iter())
        .map(|(name, functions)| (name.clone(), each(functions)))
        .collect();
    out
}

/// The translator's stage-input descriptor of `input`.
///
/// Err: the first value it has no value of, named.
fn stage_input(input: &StageInputDesc) -> Result<StageInputDescriptor, String> {
    let mut out = StageInputDescriptor::new();
    out.index_type = value::<IndexType>("index type", input.index_type)?;
    out.index_buffer_index = u32::try_from(input.index_buffer_index)
        .map_err(|_| format!("index buffer index {}", input.index_buffer_index))?;
    let mut layouts = BTreeMap::new();
    for l in &input.layouts {
        let index =
            u32::try_from(l.buffer_index).map_err(|_| format!("layout {}", l.buffer_index))?;
        let stride = match l.stride {
            STRIDE_DYNAMIC => None,
            stride => Some(u32::try_from(stride).map_err(|_| format!("stride {stride}"))?),
        };
        let mut layout = StageInputLayoutDescriptor::new(stride);
        layout.step_function = value::<StepFunction>("step function", l.step_function)?;
        layout.step_rate =
            u32::try_from(l.step_rate).map_err(|_| format!("step rate {}", l.step_rate))?;
        layouts.insert(index, layout);
    }
    out.layouts = layouts;
    for a in &input.attributes {
        let index = (u32::try_from(a.attribute_index))
            .map_err(|_| format!("attribute {}", a.attribute_index))?;
        let format = value::<AttributeFormat>("attribute format", a.format)?;
        let offset = u32::try_from(a.offset).map_err(|_| format!("offset {}", a.offset))?;
        let buffer = (u32::try_from(a.buffer_index))
            .map_err(|_| format!("attribute buffer {}", a.buffer_index))?;
        let attribute = StageInputAttributeDescriptor::new(format, offset, buffer);
        out.attributes.insert(index, attribute);
    }
    Ok(out)
}

/// rt's state of `descriptor`, by the creator `sel`, its samplers' indices taken (the module's
/// docs). `Err` is the creator's `NSError`, counted under the refusal's kind, the first line of
/// each kind naming the kernel (`super::unbuilt::naming_functions`).
fn make(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    descriptor: &rt::ComputePipelineDescriptor,
) -> Result<RtObj<RtComputePipeline>, NsErrorText> {
    let mut taking = Taking::new(device);
    let started = trace::on().then(trace::now);
    let made =
        (device.rt()).new_compute_pipeline_state(descriptor, |samplers| taking.ids(samplers));
    if let Some(started) = started {
        // rtcpso: the nanoseconds rt's make of the state took
        trace::event("rtcpso", &[trace::now().saturating_sub(started)]);
    }
    if let Err(error) = &made {
        super::diag::keep_refused_compute(sel, error, descriptor);
        // The kernel's compile for its helper starts now, for the state's next make, which the
        // make's caller makes at the fill's end (`stage::making`); counted apart from the make's
        // refusal (`super::stage`).
        if let Some(stage) = counted(UNFILLED_AT_MAKE, stage_of(error)) {
            stage::refused_for(&device.stage_fills().fill(stage, Site::Make));
        }
    }
    let kernel = stage_function(&descriptor.kernel);
    let named = match (&made, &kernel) {
        (Err(_), Some(kernel)) => vec![("kernel", kernel.clone())],
        _ => Vec::new(),
    };
    let (rt, lease) = naming_functions(named, || taking.made(sel, made))?;
    let written = written_slots(rt.texture_accesses());
    Ok(RtObj::new(RtComputePipeline {
        rt,
        lease,
        unique_identifier: next_identifier(),
        written,
        fills: Arc::clone(device.stage_fills()),
        kernel,
    }))
}

/// The texture slots of `accesses`, a function's slots and their access as rt's interface of it
/// answers (`texture_accesses`), that the function may write: those it writes, or reads and
/// writes.
pub(super) fn written_slots(accesses: Vec<(u32, rt::BindingAccess)>) -> Box<[u32]> {
    (accesses.into_iter())
        .filter(|&(_, access)| access != rt::BindingAccess::ReadOnly)
        .map(|(slot, _)| slot)
        .collect()
}

/// Whether slot `index` is one of `written` ([`written_slots`]).
pub(super) fn writes(written: &[u32], index: u64) -> bool {
    u32::try_from(index).is_ok_and(|index| written.contains(&index))
}

/// The next `uniqueIdentifier` of a pipeline state, compute or render: one of a process-wide count
/// from 1.
pub(super) fn next_identifier() -> u64 {
    NEXT_IDENTIFIER.fetch_add(1, Ordering::Relaxed)
}

/// The sampler indices one creation of a state takes (the module's docs): the answer rt's
/// pipeline creators ask the state's samplers' IDs by ([`ids`](Self::ids)), and what it took,
/// which [`made`](Self::made) leases to the state.
pub(super) struct Taking<'a> {
    device: &'a RtObj<RtDevice>,
    samplers: Vec<rt::NumberedSampler>,
    indices: Vec<u32>,
    /// Whether no index was free for a sampler.
    exhausted: bool,
}

impl<'a> Taking<'a> {
    /// Nothing taken yet, of `device`'s indices.
    pub(super) fn new(device: &'a RtObj<RtDevice>) -> Self {
        Taking {
            device,
            samplers: Vec::new(),
            indices: Vec::new(),
            exhausted: false,
        }
    }

    /// An index for each of `samplers`, in their order ([`Numbering`]): rt's sampler IDs. Where
    /// none is free for one, rt's refusal of its index, which [`made`](Self::made) answers as the
    /// limit.
    pub(super) fn ids(&mut self, samplers: &[rt::NumberedSampler]) -> Result<Vec<u32>, rt::Error> {
        let device = self.device;
        let indices = (device.numbering()).take(device.sampler_indices(), device.rt(), samplers)?;
        let Some(indices) = indices else {
            self.exhausted = true;
            return Err(rt::Error::Refused(rt::Refusal::IndicesNotGranted {
                class: rt::IndexClass::Sampler,
                index: 0,
            }));
        };
        self.samplers = samplers.to_vec();
        self.indices = indices.clone();
        Ok(indices)
    }

    /// rt's answer `made` of the creator `sel`, with the lease of the indices it took. `Err` is the
    /// creator's `NSError`, counted under the refusal's kind; the indices go back.
    pub(super) fn made<S>(
        self,
        sel: &'static str,
        made: Result<S, rt::Error>,
    ) -> Result<(S, Lease), NsErrorText> {
        let lease = Lease {
            device: self.device.clone_ref(),
            samplers: self.samplers,
            indices: self.indices,
        };
        match made {
            Ok(state) => Ok((state, lease)),
            Err(_) if self.exhausted => Err(refused_error_as(
                sel,
                RtRefusal::Limit,
                &"live states and sampler states hold every sampler index",
            )),
            Err(error) => Err(rt_refused_error(sel, &error)),
        }
    }
}

impl ComputePipeline<RtPlatform> for RtObj<RtComputePipeline> {
    /// 0: rt makes no state for indirect command buffers.
    fn gpu_resource_id(&self) -> u64 {
        crate::recorder::plain_resource_id(self, || 0)
    }

    /// rt's handle of `function` among the functions the kernel links in `functions` or
    /// `binary_functions`; nil, Metal's, for any other (`super::function_table`).
    fn function_handle(
        &self,
        function: &RtObj<RtFunction>,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::FunctionHandle>> {
        let recorded_call =
            crate::recorder::rec_call!("functionHandleWithFunction:", self, function);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::FunctionHandle>> {
                let handle = self.rt.function_handle(function.linked());
                Ok(handle.map(|handle| RtObj::new(RtFunctionHandle::new(handle))))
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's table of `function_count` null entries (`super::function_table`); nil where rt
    /// refuses it, counted under its kind.
    fn new_visible_function_table(
        &self,
        function_count: u64,
    ) -> RaiseResult<Option<<RtPlatform as Platform>::VisibleFunctionTable>> {
        const SEL: &str = "newVisibleFunctionTableWithDescriptor:";
        let recorded_call = crate::recorder::rec_call!(SEL, self, function_count);
        let recorded_answer = crate::recorder::run(
            || -> RaiseResult<Option<<RtPlatform as Platform>::VisibleFunctionTable>> {
                let descriptor = rt::VisibleFunctionTableDescriptor::new(function_count);
                match self.rt.new_visible_function_table(&descriptor) {
                    // rt makes no table past `MAX_BUFFER_LENGTH`, whose entries fit 32 bits.
                    Ok(table) => Ok(u32::try_from(function_count)
                        .ok()
                        .map(|count| RtObj::new(RtFunctionTable::new(table, count)))),
                    Err(error) => {
                        rt_refused_create(SEL, &error);
                        Ok(None)
                    }
                }
            },
        );
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's answer; 0 where rt refuses it, counted (`rt_refused_imageblock`).
    fn imageblock_memory_length(&self, imageblock_dimensions: MtlSize) -> RaiseResult<u64> {
        let asked = dimensions(imageblock_dimensions);
        let made = self.rt.imageblock_memory_length_for_dimensions(asked);
        Ok(made.unwrap_or_else(|error| {
            rt_refused_imageblock(IMAGEBLOCK_MEMORY_LENGTH, &error);
            0
        }))
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        u64::from(self.rt.max_total_threads_per_threadgroup())
    }

    /// 0, as [`gpu_resource_id`](Self::gpu_resource_id).
    fn resource_index(&self) -> u64 {
        0
    }

    fn static_threadgroup_memory_length(&self) -> u64 {
        u64::from(self.rt.static_threadgroup_memory_length())
    }

    fn support_indirect_command_buffers(&self) -> bool {
        self.rt.support_indirect_command_buffers()
    }

    fn thread_execution_width(&self) -> u64 {
        u64::from(self.rt.thread_execution_width())
    }

    fn unique_identifier(&self) -> u64 {
        self.unique_identifier
    }

    /// `Some(true)` where the kernel writes the texture bound at slot `index`, or reads and
    /// writes it, as rt's interface of the kernel says; `Some(false)` for every other slot.
    fn texture_slot_writes(&self, index: u64) -> Option<bool> {
        Some(writes(&self.written, index))
    }

    /// rt's make ahead (`prepare_dispatch`) of the `VkPipeline` the dispatch would make at its
    /// first use, at its threadgroup size, threadgroup memory lengths and grid tiles. A value past
    /// rt's 32 bits makes nothing: the dispatch, or the length's set, refuses it
    /// (`super::compute`). rt's one refusal here is of a kernel past its stage bound whose compile
    /// the cache directory does not hold, and its fill starts (`super::stage`): the dispatch waits
    /// for that fill in its walk.
    fn prepare_dispatch(&self, dispatch: Dispatch, threads: MtlSize, lengths: &[(u64, u64)]) {
        let Some(args) = RtDispatch::of(dispatch, threads, lengths) else {
            return;
        };
        let state = &self.rt;
        let prepared = state.prepare_dispatch(args.dispatch, args.threads, &args.lengths);
        if let Err(error) = prepared
            && let Some(stage) = stage_of(&error)
        {
            let _fill = self.fills.fill(stage, Site::MakeAhead);
        }
    }

    /// rt's key (`dispatch_variant`) of the `VkPipeline` the dispatch runs, and whether rt made
    /// it: of the compiled content, so the states of one content answer one key for one
    /// dispatch. rt reads made from the kernel's variants, false for a make that ends after the
    /// read. `None` where rt would make nothing for it, and for a value past rt's 32 bits, as
    /// [`prepare_dispatch`](Self::prepare_dispatch) makes nothing there.
    fn dispatch_variant(
        &self,
        dispatch: Dispatch,
        threads: MtlSize,
        lengths: &[(u64, u64)],
    ) -> Option<(rt::DispatchVariant, bool)> {
        let args = RtDispatch::of(dispatch, threads, lengths)?;
        self.rt
            .dispatch_variant(args.dispatch, args.threads, &args.lengths)
    }

    /// The driver's create of the state's main pipeline, which rt's make of the state (`make`)
    /// leaves to `ComputePipelineState::finish_pipeline`, on this thread (the module's docs, "Its
    /// main pipeline"); nothing where the state has none or it is made. `Err` is rt's one error
    /// here, a Vulkan one, as its text ([`finished`]): rt keeps nothing, and the state's first
    /// dispatch makes the pipeline again.
    fn finish(&self) -> Result<(), String> {
        finished(self.rt.finish_pipeline())
    }
}

/// rt's answer `finish` of a state's `finish_pipeline`, compute or render, as the platform's
/// finish answers it: `Err` is rt's error's text, for the host's log, where the device counts the
/// create's late failure (reims-vgpu's `pipeline-finish-failed`). Nothing more is counted here:
/// rt counts the failure as its own `Metrics::failed_finishes`.
pub(super) fn finished(finish: Result<(), rt::Error>) -> Result<(), String> {
    finish.map_err(|error| error.to_string())
}

/// A dispatch's arguments as rt's make ahead and key take them.
struct RtDispatch {
    /// Its grid.
    dispatch: rt::Dispatch,
    /// Its threads per threadgroup.
    threads: [u32; 3],
    /// Its threadgroup memory lengths, by index.
    lengths: Vec<(u32, u32)>,
}

impl RtDispatch {
    /// The dispatch's, of its `threads` per threadgroup and its threadgroup memory `lengths`;
    /// `None` where a value is past rt's 32 bits.
    fn of(dispatch: Dispatch, threads: MtlSize, lengths: &[(u64, u64)]) -> Option<Self> {
        let small = |n: u64| u32::try_from(n).ok();
        let size = |s: MtlSize| Some([small(s.width)?, small(s.height)?, small(s.depth)?]);
        let threads = size(threads)?;
        let dispatch = match dispatch {
            Dispatch::Threads(grid) => rt::Dispatch::Threads(size(grid)?),
            Dispatch::Threadgroups(groups) => rt::Dispatch::Threadgroups(size(groups)?),
            Dispatch::ThreadgroupsIndirect => rt::Dispatch::Indirect,
        };
        let lengths = (lengths.iter())
            .map(|&(index, length)| Some((small(index)?, small(length)?)))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            dispatch,
            threads,
            lengths,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{FunctionConstant, Library as _};
    use crate::core::platform::values::{AttributeDesc, BufferLayoutDesc};
    use crate::core::raise::OnFault;

    /// `numbered_samplers.metal`'s functions, compiled by build.rs.
    const NUMBERED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/numbered_samplers.metallib"));

    /// `optional_constant.metal`'s kernel, compiled by build.rs.
    const OPTIONAL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/optional_constant.metallib"));

    /// `tests/shaders/compute_tgstatic.metal`'s kernel, of 64 words of static threadgroup memory.
    const TGSTATIC: &[u8] = include_bytes!("../../../../tests/shaders/compute_tgstatic.metallib");

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A kernel made unspecialized that declares function constants is refused, and neither the
    /// refusal's line nor its `NSError` names the function, whose name is the guest's (in a child,
    /// whose stderr is its own)
    #[test]
    fn an_unspecialized_function_s_refusal_names_no_function() {
        use crate::core::testkit::death::{Expect, Options, Outcome, run};
        const NAME: &str = "optional_constant";
        run(
            Expect::Any,
            Options::default(),
            module_path!(),
            "an_unspecialized_function_s_refusal_names_no_function",
            |_| {
                let library = rt::Library::from_metallib(OPTIONAL).expect("a library");
                let kernel = library.function(NAME).expect("the kernel");
                let stage = rt::FunctionDescriptor::new(kernel);
                let refused = specialized(WITH_FUNCTION, &stage).expect_err("a refusal");
                super::super::unbuilt::say(&format!("test: NSError: {}\n", refused.description));
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::Returned, "{}", r.stderr());
                let out = r.stderr();
                let said = "a function that declares function constants was made unspecialized";
                let line = format!(
                    "reims-vgpu: {WITH_FUNCTION}: metal2vulkan-runtime refused it ({said}); \
                     refused (rt-refused-metal), the first of its kind"
                );
                let answered = format!(
                    "test: NSError: metal2vulkan-runtime refused it: {said} \
                     (rt-refused-metal)"
                );
                // A line ends with what was written: the child's first line starts with
                // libtest's own "test NAME ... ".
                let lines: Vec<&str> = out.lines().collect();
                assert!(lines.iter().any(|l| l.ends_with(&line)), "{out}");
                assert!(lines.iter().any(|l| l.ends_with(&answered)), "{out}");
                assert!(!out.contains(NAME), "{out}");
            },
        );
    }

    /// Each value of a stage-input descriptor is the translator's, a dynamic stride none; a value
    /// it has none of is named.
    #[test]
    fn stage_input_values_are_the_translator_s() {
        let input = StageInputDesc {
            layouts: vec![
                BufferLayoutDesc {
                    buffer_index: 1,
                    stride: 16,
                    step_function: 5,
                    step_rate: 1,
                },
                BufferLayoutDesc {
                    buffer_index: 2,
                    stride: u64::MAX,
                    step_function: 0,
                    step_rate: 0,
                },
            ],
            attributes: vec![AttributeDesc {
                attribute_index: 3,
                format: 31,
                offset: 4,
                buffer_index: 1,
            }],
            index_type: 1,
            index_buffer_index: 7,
        };
        let out = stage_input(&input).expect("the translator's values");
        assert_eq!(
            (out.index_type, out.index_buffer_index),
            (IndexType::UInt32, 7)
        );
        let one = out.layouts.get(&1).expect("layout 1");
        assert_eq!(one.stride, Some(16));
        assert_eq!(one.step_function, StepFunction::ThreadPositionInGridX);
        let two = out.layouts.get(&2).expect("layout 2");
        assert_eq!(
            (two.stride, two.step_function),
            (None, StepFunction::Constant)
        );
        let attribute = StageInputAttributeDescriptor::new(AttributeFormat::Float4, 4, 1);
        assert_eq!(out.attributes.get(&3), Some(&attribute));

        let mut bad = input.clone();
        bad.layouts[0].step_function = 99;
        assert_eq!(stage_input(&bad).err(), Some("step function 99".to_owned()));
    }

    /// Libraries of metallibs and the platform's shaders, functions as they were made, and compute
    /// states of them: samplers equal to a live state's take its index and give it back with the
    /// last; every refusal is counted.
    #[test]
    fn libraries_and_compute_pipelines_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let translation = RtRefusal::Translation.code();
        let metal = RtRefusal::Refused.code();

        let before = counted(translation);
        let junk = device.new_library(b"no metallib").expect("never a raise");
        assert!(junk.is_err());
        assert_eq!(counted(translation), before + 1);

        let shaders = device
            .new_shader_library()
            .expect("our shaders")
            .expect("a library");
        let again = device
            .new_shader_library()
            .expect("our shaders")
            .expect("a library");
        assert_eq!(shaders.id(), again.id());
        assert!(
            shaders
                .function_names()
                .iter()
                .any(|n| n == "displayMirrorTile")
        );

        let library = device.new_library(NUMBERED).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let names = library.function_names();
        for name in ["numbered_read", "numbered_sample", "numbered_write"] {
            assert!(names.iter().any(|n| n == name), "{names:?}");
        }
        assert!(
            library
                .new_function(on, "absent")
                .expect("never a raise")
                .is_none()
        );
        let function = |name: &str| {
            let made = library.new_function(on, name).expect("never a raise");
            made.expect("a function of the table")
        };
        let state = |f: &RtObj<RtFunction>| {
            let made = device.new_compute_pipeline_state_for_function(Some(f));
            made.expect("a compute state").expect("a compute state")
        };

        let last = rt::IndexClass::Sampler.range().end - 1;
        let held = device.numbering().held();
        let sample = function("numbered_sample");
        let first = state(&sample);
        let second = state(&sample);
        assert!(matches!(
            first.samplers(),
            [rt::NumberedSampler::Constexpr(_)]
        ));
        assert_eq!(first.sampler_indices(), [last]);
        assert_eq!(second.sampler_indices(), [last]);
        assert_eq!(device.numbering().held(), held + 1);
        let read = state(&function("numbered_read"));
        assert_eq!(read.samplers(), [rt::NumberedSampler::Read]);
        assert_eq!(read.sampler_indices(), [last - 1]);
        assert_eq!(device.numbering().held(), held + 2);

        // A slot the kernel writes is written; one it reads or samples, or does not use, is not.
        let write = state(&function("numbered_write"));
        let slots = |s: &RtObj<RtComputePipeline>| -> Vec<Option<bool>> {
            (0..2).map(|i| s.texture_slot_writes(i)).collect()
        };
        assert_eq!(slots(&write), [Some(true), Some(false)]);
        assert_eq!(slots(&read), [Some(false); 2]);
        assert_eq!(slots(&first), [Some(false); 2]);
        assert_eq!(write.texture_slot_writes(u64::MAX), Some(false));
        drop(write);

        assert!(first.max_total_threads_per_threadgroup() > 0);
        assert_eq!(first.thread_execution_width(), 32);
        assert_eq!((first.gpu_resource_id(), first.resource_index()), (0, 0));
        assert!(!first.support_indirect_command_buffers());
        assert_ne!(first.unique_identifier(), 0);
        assert_ne!(first.unique_identifier(), second.unique_identifier());

        // The kernel links no function, so it has no handle, itself included; a table of it is
        // made all the same (`super::function_table`'s tests call through one).
        assert!(
            first
                .function_handle(&sample)
                .expect("never a raise")
                .is_none()
        );
        assert!(
            first
                .new_visible_function_table(2)
                .expect("never a raise")
                .is_some()
        );
        assert_eq!(first.static_threadgroup_memory_length(), 0);
        let length = first.imageblock_memory_length(MtlSize::new(8, 8, 1));
        assert_eq!(length.expect("never a raise"), 0);

        drop((first, second, read));
        assert_eq!(device.numbering().held(), held);

        let mut desc = FullComputePipelineDesc::new(&sample);
        desc.max_total_threads_per_threadgroup = 64;
        let limited = device.new_compute_pipeline_state_full(&desc);
        let limited = limited.expect("a compute state").expect("a compute state");
        assert_eq!(limited.max_total_threads_per_threadgroup(), 64);

        // A kernel's static threadgroup memory is rt's answer: Metal's for this kernel on an M3
        // Max.
        let tgstatic = device.new_library(TGSTATIC).expect("never a raise");
        let tgstatic = tgstatic.expect("a metallib").expect("a library");
        let kernel = tgstatic
            .new_function(on, "tg_static")
            .expect("never a raise");
        let kernel = state(&kernel.expect("a function of the table"));
        assert_eq!(kernel.static_threadgroup_memory_length(), 256);
        assert_eq!(kernel.thread_execution_width(), 32);
        assert!(!kernel.support_indirect_command_buffers());

        let optional = device.new_library(OPTIONAL).expect("never a raise");
        let optional = optional.expect("a metallib").expect("a library");
        let plain = optional
            .new_function(on, "optional_constant")
            .expect("never a raise");
        let plain = plain.expect("a function of the table");
        let before = counted(metal);
        let refused = device.new_compute_pipeline_state_for_function(Some(&plain));
        assert!(refused.is_err());
        assert_eq!(counted(metal), before + 1);
        let five = FunctionConstant {
            index: 0,
            data_type: 29,
            value: [5; FunctionConstant::MAX_BYTES],
        };
        let made = optional.new_function_with_constants(on, "optional_constant", &[five]);
        let specialized = made
            .expect("never a raise")
            .expect("a specialized function");
        assert!(state(&specialized).max_total_threads_per_threadgroup() > 0);
        let made = optional.new_function_with_options(on, "optional_constant", &[], 0, Some("o"));
        let named = made
            .expect("never a raise")
            .expect("a function made by a descriptor");
        assert_eq!(named.name(), "o");
        assert!(state(&named).max_total_threads_per_threadgroup() > 0);

        let before = counted(metal);
        let unknown = FunctionConstant {
            data_type: 7,
            ..five
        };
        let made = optional.new_function_with_constants(on, "optional_constant", &[unknown]);
        assert!(made.expect("never a raise").is_none());
        let made = optional.new_function_with_options(on, "optional_constant", &[], 0x100, None);
        assert!(made.expect("never a raise").is_none());
        assert_eq!(counted(metal), before + 2);
    }

    /// rt's key of the `VkPipeline` a dispatch runs (`dispatch_variant`): one for the states of
    /// one kernel, and for grids that differ only in how many whole threadgroups they hold;
    /// another for another threadgroup size of a kernel that observes its threadgroups, or for
    /// another kernel; none where rt would make nothing (a threadgroup size past the state's
    /// most, a grid of no threadgroups) and for a value past rt's 32 bits. A key is made once
    /// its pipeline is made ahead, for every state of its kernel, and not before
    #[test]
    fn dispatch_variants_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let function = |metallib: &[u8], name: &str| {
            let library = device.new_library(metallib).expect("never a raise");
            let library = library.expect("a metallib").expect("a library");
            let made = library.new_function(on, name).expect("never a raise");
            made.expect("a function of the table")
        };
        let state = |f: &RtObj<RtFunction>| {
            let made = device.new_compute_pipeline_state_for_function(Some(f));
            made.expect("a compute state").expect("a compute state")
        };
        // The key and the made flag of `state`'s dispatch of `groups` threadgroups of `threads`
        // threads, with the threadgroup memory `lengths`.
        let variant =
            |state: &RtObj<RtComputePipeline>, groups, threads, lengths: &[(u64, u64)]| {
                let grid = Dispatch::Threadgroups(MtlSize::new(groups, 1, 1));
                state.dispatch_variant(grid, MtlSize::new(threads, 1, 1), lengths)
            };
        let key = |state: &RtObj<RtComputePipeline>, groups, threads, lengths: &[(u64, u64)]| {
            variant(state, groups, threads, lengths).map(|(key, _)| key)
        };

        let tgstatic = function(TGSTATIC, "tg_static");
        let (first, second) = (state(&tgstatic), state(&tgstatic));
        let made = key(&first, 2, 32, &[]);
        assert!(
            made.is_some(),
            "no key of a dispatch rt makes a pipeline for"
        );
        assert_eq!(key(&second, 2, 32, &[]), made);
        // `tg_static` declares no bound of threads: its state makes no `[32, 1, 1]` pipeline,
        // and a make ahead of the dispatch makes the key's, for both states.
        assert_eq!(variant(&first, 2, 32, &[]), made.map(|key| (key, false)));
        let grid = Dispatch::Threadgroups(MtlSize::new(2, 1, 1));
        first.prepare_dispatch(grid, MtlSize::new(32, 1, 1), &[]);
        assert_eq!(variant(&first, 2, 32, &[]), made.map(|key| (key, true)));
        assert_eq!(variant(&second, 4, 32, &[]), made.map(|key| (key, true)));
        assert_eq!(key(&first, 4, 32, &[]), made);
        let wider = key(&first, 2, 64, &[]);
        assert!(wider.is_some(), "no key of a wider threadgroup");
        assert_ne!(wider, made);
        let read = state(&function(NUMBERED, "numbered_read"));
        let other = key(&read, 2, 32, &[]);
        assert!(other.is_some(), "no key of another kernel's dispatch");
        assert_ne!(other, made);

        let most = first.max_total_threads_per_threadgroup();
        assert_eq!(key(&first, 2, most * 2, &[]), None);
        assert_eq!(key(&first, 0, 32, &[]), None);
        assert_eq!(key(&first, 1 << 32, 32, &[]), None);
        assert_eq!(key(&first, 2, 32, &[(1 << 32, 16)]), None);
    }

    /// A state's finish, on the GPU the test holds, is the driver's create of its main pipeline
    /// (rt's `finish_pipeline`): rt's make of a tiling kernel's state makes no `VkPipeline`, its
    /// finish makes one, counted as finished, and a second finish makes nothing; the finish of a
    /// state with no main pipeline, of a kernel that observes its threadgroups and declares no
    /// size, makes nothing either
    #[test]
    fn a_finish_makes_the_state_s_main_pipeline_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let counters = device.rt().counters();
        let on = OnFault::Return;
        let function = |metallib: &[u8], name: &str| {
            let library = device.new_library(metallib).expect("never a raise");
            let library = library.expect("a metallib").expect("a library");
            let made = library.new_function(on, name).expect("never a raise");
            made.expect("a function of the table")
        };
        let state = |f: &RtObj<RtFunction>| {
            let made = device.new_compute_pipeline_state_for_function(Some(f));
            made.expect("a compute state").expect("a compute state")
        };
        // The pipelines made, and those the finishes made.
        let made = || {
            let m = counters.metrics();
            (m.pipelines_created, m.finished_pipelines)
        };

        // `numbered_read` reads its position in the grid alone: a kernel of tiles.
        let read = function(NUMBERED, "numbered_read");
        let before = made();
        let tiles = state(&read);
        assert_eq!(made(), before, "the state's make made a VkPipeline");
        assert_eq!(tiles.finish(), Ok(()));
        assert_eq!(made(), (before.0 + 1, before.1 + 1));
        assert_eq!(tiles.finish(), Ok(()));
        assert_eq!(made(), (before.0 + 1, before.1 + 1));

        let observing = state(&function(TGSTATIC, "tg_static"));
        let before = made();
        assert_eq!(observing.finish(), Ok(()));
        assert_eq!(made(), before, "a state with no main pipeline made one");
    }

    /// A finish rt fails, the driver's create after the state answered (`Error::Vulkan`), is the
    /// finish's `Err` with rt's text, which the device takes as the create's late failure; one
    /// that made the pipeline is `Ok`
    #[test]
    fn a_failed_finish_answers_rt_s_error_as_its_text() {
        let failure = || rt::Error::Vulkan {
            call: "vkCreateComputePipelines",
            result: ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
        };
        let text = failure().to_string();
        assert!(
            text.starts_with("vkCreateComputePipelines failed: "),
            "{text}"
        );
        assert_eq!(finished(Err(failure())), Err(text));
        assert_eq!(finished(Ok(())), Ok(()));
    }
}
