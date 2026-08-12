//! Compute segment sequencing: control-flow encode + ICB fail-closed.
//!
//! ## Control-flow (`0xdc`–`0xe2`)
//!
//! Host Metal exposes SPI (`encodeStartIf` / `While` / `DoWhile` family) on the
//! real AGX compute encoder (runtime-probed; not in public headers). Product
//! path opens a multi-record [`ComputeSession`] encoder and records those SPI
//! calls with condition buffers staged from guest GVA.
//!
//! Nested dispatches under an open session encode onto the **same** encoder via
//! [`crate::backend::metal::compute::compute_encode_on_encoder`] so they sit
//! inside the SPI region. The session commits once at segment end; GVA
//! writeback is deferred until then.
//!
//! ## ICB (`0xe4` / `0xe5`)
//!
//! Type-7 tag `0x36` materializes a host `MTLIndirectCommandBuffer` (cached per
//! task/ref). Command fills use the host Metal fill API in
//! [`crate::runtime::icb`] — the stream carries no fill opcodes. Execute
//! applies **parent-encoder inheritance** from stream [`ComputeAccum`] (Metal:
//! buffers when `inheritBuffers`, pipeline when `inheritPipelineState`; textures/
//! samplers are never recordable into classic `MTLIndirectComputeCommand` and
//! always come from the encoder when present), then encodes
//! `executeCommandsInBuffer` SPI. Buffer writebacks from ICB fills and from
//! inherited encoder binds flush after session commit. Failures latch
//! [`SequencingBlock::IndirectCommandBuffer`].

use crate::model::DeviceState;
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
use crate::runtime::compute_exec;
use crate::runtime::compute_exec::{ComputeAccum, ComputeStatus};
use crate::runtime::decode::compute::{Command as ComputeCommand, Kind};
use crate::runtime::host::{HostMemory, HostOps};

/// Latched reason that blocks later dispatches in the same compute segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SequencingBlock {
    ControlFlow,
    IndirectCommandBuffer,
}

/// Multi-record Metal compute encoder for a single compute segment.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
pub struct ComputeSession {
    pub(crate) device: metal::Device,
    command_buffer: metal::CommandBuffer,
    pub(crate) encoder: metal::ComputeCommandEncoder,
    pub(crate) retained: Vec<metal::Buffer>,
    /// PSOs kept alive for parent-encoder inheritance until session commit.
    retained_psos: Vec<metal::ComputePipelineState>,
    /// Textures staged for ICB parent-encoder / argument-buffer inheritance.
    retained_textures: Vec<metal::Texture>,
    /// Samplers encoded into argument buffers (must outlive the command buffer).
    retained_samplers: Vec<metal::SamplerState>,
    /// Materialized ICBs kept alive until session commit.
    retained_icbs: Vec<metal::IndirectCommandBuffer>,
    /// Nested dispatches encoded on this session; flushed after GPU completion.
    pub(crate) nested_jobs: Vec<compute_exec::NestedDispatchJob>,
    pub control_depth: i32,
    ended: bool,
}

#[cfg(feature = "backend-vulkan")]
pub struct ComputeSession {
    pub control_depth: i32,
}

#[cfg(all(feature = "backend-metal", target_os = "macos"))]
impl Drop for ComputeSession {
    fn drop(&mut self) {
        if !self.ended {
            // Abandoned session (test early-return / panic): close encoder cleanly.
            self.encoder.end_encoding();
            self.ended = true;
        }
    }
}

impl ComputeSession {
    pub fn open(dispatch_type: u32) -> Result<Self, ComputeStatus> {
        #[cfg(feature = "backend-vulkan")]
        {
            let _ = dispatch_type;
            Err(ComputeStatus::NoMetal("compute_session_no_vulkan_path"))
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        {
            use crate::backend::metal::abi::REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT;
            use crate::backend::metal::runtime::{system_device, thread_queue};
            use metal::MTLDispatchType;

            let Some(device) = system_device() else {
                return Err(ComputeStatus::NoMetal("compute_session_no_metal_device"));
            };
            let queue = thread_queue(device);
            let metal_dt = if dispatch_type == REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT {
                MTLDispatchType::Concurrent
            } else {
                MTLDispatchType::Serial
            };
            let Some(command_buffer) = crate::backend::metal::raw_metal::new_command_buffer(&queue)
            else {
                return Err(ComputeStatus::MetalFailed(
                    "compute_session_command_buffer_unavailable",
                ));
            };
            let command_buffer = command_buffer.to_owned();
            let Some(encoder) =
                crate::backend::metal::raw_metal::new_compute_command_encoder_with_dispatch_type(
                    &command_buffer,
                    metal_dt,
                )
            else {
                return Err(ComputeStatus::MetalFailed(
                    "compute_session_encoder_unavailable",
                ));
            };
            let encoder = encoder.to_owned();
            Ok(Self {
                device: device.to_owned(),
                command_buffer,
                encoder,
                retained: Vec::new(),
                retained_psos: Vec::new(),
                retained_textures: Vec::new(),
                retained_samplers: Vec::new(),
                retained_icbs: Vec::new(),
                nested_jobs: Vec::new(),
                control_depth: 0,
                ended: false,
            })
        }
    }

    pub fn encode_control<M: HostMemory + HostOps>(
        &mut self,
        state: &DeviceState,
        host: &M,
        task_id: u32,
        cmd: &ComputeCommand,
    ) -> ComputeStatus {
        #[cfg(feature = "backend-vulkan")]
        {
            let _ = (state, host, task_id, cmd);
            ComputeStatus::NoMetal("compute_control_no_vulkan_path")
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        {
            use crate::backend::metal::raw_metal::{
                encode_end_do_while, encode_end_if, encode_end_while, encode_start_do_while,
                encode_start_else, encode_start_if, encode_start_while,
            };
            use metal::MTLResourceOptions;

            let stage_cond = |this: &mut Self,
                              buffer_ref: u32,
                              offset: u64|
             -> Result<(metal::Buffer, u64), ComputeStatus> {
                let end = offset.checked_add(4).ok_or(ComputeStatus::MissingBuffer(
                    "compute_control_cond_offset_overflow",
                ))?;
                let bytes = compute_exec::read_buffer_window(
                    state,
                    host,
                    task_id,
                    buffer_ref,
                    0,
                    end as usize,
                )?;
                let mtl = unsafe {
                    crate::backend::metal::raw_metal::new_buffer_with_data(
                        &this.device,
                        bytes.as_ptr() as *const _,
                        bytes.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or(ComputeStatus::MetalFailed(
                    "compute_session_control_buffer_alloc_failed",
                ))?;
                this.retained.push(mtl.clone());
                Ok((mtl, offset))
            };

            match cmd.kind {
                Kind::ControlStartDoWhile => {
                    encode_start_do_while(&self.encoder);
                    self.control_depth = self.control_depth.saturating_add(1);
                    ComputeStatus::Ok
                }
                Kind::ControlEndDoWhile => {
                    let (buf, off) = match stage_cond(
                        self,
                        cmd.condition_buffer_ref,
                        cmd.condition_buffer_offset,
                    ) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    let ok = encode_end_do_while(
                        &self.encoder,
                        buf.as_ref(),
                        off,
                        cmd.condition_comparison as u64,
                        cmd.condition_reference_value,
                    );
                    self.control_depth = self.control_depth.saturating_sub(1);
                    if ok {
                        ComputeStatus::Ok
                    } else {
                        ComputeStatus::MetalFailed("compute_control_end_do_while")
                    }
                }
                Kind::ControlStartWhile => {
                    let (buf, off) = match stage_cond(
                        self,
                        cmd.condition_buffer_ref,
                        cmd.condition_buffer_offset,
                    ) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    encode_start_while(
                        &self.encoder,
                        buf.as_ref(),
                        off,
                        cmd.condition_comparison as u64,
                        cmd.condition_reference_value,
                    );
                    self.control_depth = self.control_depth.saturating_add(1);
                    ComputeStatus::Ok
                }
                Kind::ControlEndWhile => {
                    let ok = encode_end_while(&self.encoder);
                    self.control_depth = self.control_depth.saturating_sub(1);
                    if ok {
                        ComputeStatus::Ok
                    } else {
                        ComputeStatus::MetalFailed("compute_control_end_while")
                    }
                }
                Kind::ControlStartIf => {
                    let (buf, off) = match stage_cond(
                        self,
                        cmd.condition_buffer_ref,
                        cmd.condition_buffer_offset,
                    ) {
                        Ok(v) => v,
                        Err(e) => return e,
                    };
                    encode_start_if(
                        &self.encoder,
                        buf.as_ref(),
                        off,
                        cmd.condition_comparison as u64,
                        cmd.condition_reference_value,
                    );
                    self.control_depth = self.control_depth.saturating_add(1);
                    ComputeStatus::Ok
                }
                Kind::ControlStartElse => {
                    encode_start_else(&self.encoder);
                    ComputeStatus::Ok
                }
                Kind::ControlEndIf => {
                    let ok = encode_end_if(&self.encoder);
                    self.control_depth = self.control_depth.saturating_sub(1);
                    if ok {
                        ComputeStatus::Ok
                    } else {
                        ComputeStatus::MetalFailed("compute_control_end_if")
                    }
                }
                _ => ComputeStatus::Unsupported("control_flow_unknown_kind"),
            }
        }
    }

    pub fn encode_icb<M: HostMemory + HostOps>(
        &mut self,
        state: &mut DeviceState,
        host: &mut M,
        task_id: u32,
        cmd: &ComputeCommand,
        acc: &ComputeAccum,
    ) -> ComputeStatus {
        if cmd.indirect_command_buffer_ref == 0 {
            return ComputeStatus::MissingBuffer("compute_icb_ref_zero");
        }
        #[cfg(feature = "backend-vulkan")]
        {
            let _ = (state, host, task_id, cmd, acc);
            ComputeStatus::NoMetal("compute_icb_no_vulkan_path")
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        {
            use crate::backend::metal::raw_metal::{
                execute_commands_in_buffer, execute_commands_in_buffer_indirect,
            };
            use crate::runtime::compute_exec::read_buffer_window;
            use crate::runtime::icb::{
                export_icb_writeback_job, fill_icb_from_command_memory, resolve_metal_icb,
            };
            use metal::MTLResourceOptions;

            let icb_ref = cmd.indirect_command_buffer_ref;
            // The ICB rail names its own check; forwarding keeps that name
            // instead of the four coarse literals this match used to invent.
            let (desc, icb) = match resolve_metal_icb(state, host, task_id, icb_ref) {
                Ok(v) => v,
                Err(e) => return e.into(),
            };

            let status = match cmd.kind {
                Kind::ExecuteCommandsInBuffer => {
                    let loc = cmd.indirect_command_range_location;
                    let len = cmd.indirect_command_range_length;
                    // Bounds: range must fit the materialized ICB size (maxCommandCount).
                    let size = icb.size();
                    if loc.saturating_add(len) > size {
                        return ComputeStatus::Unsupported("icb_range_exceeds_size");
                    }
                    // Buffer-backed fills: re-decode guest ICB command memory into host slots.
                    // `icb_fill_outcome` owns what an unfilled ICB costs and
                    // which outcomes an execute carries on from; the render arm
                    // in `runtime::draw::metal_icb` asks the same function.
                    match crate::runtime::icb::icb_fill_outcome(
                        fill_icb_from_command_memory(state, host, task_id, icb_ref, loc, len),
                        task_id,
                        icb_ref,
                    ) {
                        Ok(()) => {}
                        Err(e) => return e.into(),
                    }
                    // Parent-encoder inheritance after slot fill, before execute
                    // (Metal: inheritBuffers / inheritPipelineState / AB textures).
                    let inherit_job = match apply_icb_compute_encoder_inheritance(
                        self,
                        state,
                        host,
                        task_id,
                        acc,
                        &desc,
                        Some(icb.as_ref()),
                        loc,
                        len,
                    ) {
                        Ok(job) => job,
                        Err(e) => return e,
                    };
                    execute_commands_in_buffer(&self.encoder, icb.as_ref(), loc, len);
                    self.retained_icbs.push(icb);
                    if let Some(job) = inherit_job {
                        self.nested_jobs.push(job);
                    }
                    ComputeStatus::Ok
                }
                Kind::ExecuteCommandsInBufferIndirect => {
                    // Stage 8-byte MTLIndirectCommandBufferExecutionRange from guest buffer.
                    const EXEC_RANGE_LEN: usize = 8;
                    let raw = match read_buffer_window(
                        state,
                        host,
                        task_id,
                        cmd.indirect_command_arguments_buffer_ref,
                        cmd.indirect_command_arguments_buffer_offset,
                        EXEC_RANGE_LEN,
                    ) {
                        Ok(b) => b,
                        Err(e) => return e,
                    };
                    let Some(mtl) = (unsafe {
                        crate::backend::metal::raw_metal::new_buffer_with_data(
                            &self.device,
                            raw.as_ptr() as *const _,
                            raw.len() as u64,
                            MTLResourceOptions::StorageModeShared,
                        )
                    }) else {
                        return ComputeStatus::MetalFailed(
                            "compute_session_icb_buffer_alloc_failed",
                        );
                    };
                    self.retained.push(mtl.clone());
                    // Indirect range size unknown until GPU reads it — apply inheritance
                    // with parent-encoder binds only (no ICB slot patch of AB buffer).
                    let inherit_job = match apply_icb_compute_encoder_inheritance(
                        self, state, host, task_id, acc, &desc, None, 0, 0,
                    ) {
                        Ok(job) => job,
                        Err(e) => return e,
                    };
                    execute_commands_in_buffer_indirect(
                        &self.encoder,
                        icb.as_ref(),
                        mtl.as_ref(),
                        0,
                    );
                    self.retained_icbs.push(icb);
                    if let Some(job) = inherit_job {
                        self.nested_jobs.push(job);
                    }
                    ComputeStatus::Ok
                }
                _ => ComputeStatus::Unsupported("icb_encode_unknown_kind"),
            };
            if status == ComputeStatus::Ok {
                if let Some(job) = export_icb_writeback_job(task_id, icb_ref) {
                    self.nested_jobs.push(job);
                }
            }
            status
        }
    }

    /// End encoding, commit, wait, then flush nested dispatch writebacks to GVA.
    #[allow(
        unused_mut,
        reason = "the Apple Metal branch mutates session state before committing"
    )]
    pub fn finish<M: HostMemory + HostOps>(
        mut self,
        host: &mut M,
        state: &mut DeviceState,
        task_id: u32,
    ) -> ComputeStatus {
        #[cfg(feature = "backend-vulkan")]
        {
            let _ = (host, state, task_id);
            ComputeStatus::Ok
        }
        #[cfg(all(feature = "backend-metal", target_os = "macos"))]
        {
            use metal::MTLCommandBufferStatus;

            if !self.ended {
                self.encoder.end_encoding();
                self.ended = true;
            }
            self.command_buffer.commit();
            self.command_buffer.wait_until_completed();
            if self.command_buffer.status() == MTLCommandBufferStatus::Error {
                return ComputeStatus::MetalFailed("compute_session_command_buffer_error");
            }
            compute_exec::flush_nested_jobs(state, host, task_id, &mut self.nested_jobs)
        }
    }
}

/// The mutable state of one `SEGMENT_TYPE_COMPUTE` segment.
///
/// These three share a single lifetime: they come into existence when the
/// segment opens, every record in the segment reads and mutates them together,
/// and the session commits when the segment ends. Passing them as one value
/// keeps that lifetime visible at each call site.
#[derive(Default)]
pub struct ComputeSegment {
    /// Pipeline / bind state accumulated across the segment's records.
    pub acc: ComputeAccum,
    /// Multi-record encoder, opened on demand by the first control-flow or ICB
    /// record and committed at segment end.
    pub session: Option<ComputeSession>,
    /// Latched sequencing failure; once set it refuses later dispatches.
    pub block: Option<SequencingBlock>,
}

pub fn ensure_session(
    session: &mut Option<ComputeSession>,
    dispatch_type: u32,
) -> Result<&mut ComputeSession, ComputeStatus> {
    if session.is_none() {
        *session = Some(ComputeSession::open(dispatch_type)?);
    }
    Ok(session.as_mut().unwrap())
}

pub fn apply_sequencing<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    cmd: &ComputeCommand,
    seg: &mut ComputeSegment,
) -> ComputeStatus {
    if seg.block.is_some() {
        return ComputeStatus::Unsupported("sequencing_block_active");
    }
    match cmd.kind {
        Kind::ControlStartDoWhile
        | Kind::ControlEndDoWhile
        | Kind::ControlStartWhile
        | Kind::ControlEndWhile
        | Kind::ControlStartIf
        | Kind::ControlStartElse
        | Kind::ControlEndIf => {
            let sess = match ensure_session(&mut seg.session, seg.acc.dispatch_type) {
                Ok(s) => s,
                Err(e) => {
                    seg.block = Some(SequencingBlock::ControlFlow);
                    return e;
                }
            };
            let st = sess.encode_control(state, host, task_id, cmd);
            if !matches!(st, ComputeStatus::Ok) {
                seg.block = Some(SequencingBlock::ControlFlow);
            }
            st
        }
        Kind::ExecuteCommandsInBuffer | Kind::ExecuteCommandsInBufferIndirect => {
            let sess = match ensure_session(&mut seg.session, seg.acc.dispatch_type) {
                Ok(s) => s,
                Err(e) => {
                    seg.block = Some(SequencingBlock::IndirectCommandBuffer);
                    return e;
                }
            };
            let st = sess.encode_icb(state, host, task_id, cmd, &seg.acc);
            // Latch only on failure so successful materialize+execute does not
            // block later dispatches in the segment.
            if !matches!(st, ComputeStatus::Ok) {
                seg.block = Some(SequencingBlock::IndirectCommandBuffer);
            }
            st
        }
        _ => ComputeStatus::Unsupported("sequencing_unknown_kind"),
    }
}

/// Apply stream-accumulated state to the parent compute encoder before
/// `executeCommandsInBuffer`.
///
/// Metal contract:
/// - **Kernel buffers** when create `inheritBuffers`.
/// - **Pipeline** when create `inheritPipelineState` (ICB-capable PSO).
/// - **Textures/samplers:** classic `MTLIndirectComputeCommand` has no
///   setTexture/setSampler. Kernels with **direct** texture args cannot set
///   `supportIndirectCommandBuffers` on this stack — textures must live in an
///   **argument buffer** (kernel `constant Struct &args [[buffer(N)]]` with
///   `texture2d`/`sampler` members). Product packages stream textures/samplers
///   into that AB via BindingInfo reflection, binds the AB as a kernel buffer,
///   and `useResource`s the textures on the parent encoder.
///
/// When `icb` + range are provided and `!inheritBuffers`, the AB buffer is also
/// recorded onto each ICB command in range (`setKernelBuffer`).
///
/// Returns a deferred writeback job for inherited kernel buffers and storage
/// textures (flushed after session commit), or `None` when nothing needs
/// writeback.
#[cfg(all(feature = "backend-metal", target_os = "macos"))]
// Session, device state, host, task, ICB and range: the inheritance rule needs
// all of them, and each is already a distinct borrow.
#[allow(clippy::too_many_arguments)]
fn apply_icb_compute_encoder_inheritance<M: HostMemory + HostOps>(
    session: &mut ComputeSession,
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    acc: &ComputeAccum,
    desc: &crate::runtime::decode::resource::IndirectCommandBufferDescriptor,
    icb: Option<&metal::IndirectCommandBufferRef>,
    range_location: u64,
    range_length: u64,
) -> Result<Option<compute_exec::NestedDispatchJob>, ComputeStatus> {
    use crate::backend::metal::abi::{
        texture_binds_as_storage, ReimsVgpuSampler, REIMS_VGPU_BINDING_SAMPLER_BASE,
        REIMS_VGPU_BINDING_TEXTURE_BASE,
    };
    use crate::backend::metal::compute::{
        bind_compute_sampled_images, bind_compute_samplers, bind_storage_images,
        reflect_compute_textures_mtlb,
    };
    use crate::backend::metal::format::storage_image_format;
    use crate::backend::metal::raw_metal::{
        reflect_argument_buffer_layout, set_buffer_with_attribute_stride, BINDING_ACCESS_READ_ONLY,
        BINDING_ACCESS_WRITE_ONLY,
    };
    use crate::backend::metal::runtime::new_buffer_from_host;
    use crate::backend::metal::samplers::make_explicit_sampler;
    use crate::backend::metal::util::valid_buffer_binding;
    use crate::contract::endian::ld32;
    use crate::contract::extent::tight_image_bytes;
    use crate::runtime::compute_exec::{
        load_compute_pipeline, nested_job_from_icb_resources, split_staged_textures, stage_buffer,
        stage_texture_raw,
    };
    use crate::runtime::decode::resource::{
        decode_sampler_descriptor, OBJECT_TYPE_TYPE7, TYPE7_OBJECT_SAMPLER,
    };
    use crate::runtime::icb::new_icb_compute_pso;
    use crate::runtime::mtlb::{load_mtlb, AirLoadRail};
    use crate::runtime::objects;
    use metal::{
        MTLRegion, MTLResourceUsage, MTLStorageMode, MTLTextureType, MTLTextureUsage,
        TextureDescriptor,
    };

    // Pipeline when inheritPipelineState — not recorded into the ICB slot.
    if desc.inherit_pipeline_state() {
        if acc.pipeline_ref == 0 {
            return Err(ComputeStatus::MissingPipeline(
                "compute_icb_inherit_pipeline_ref_zero",
            ));
        }
        let pipeline = load_compute_pipeline(state, host, task_id, acc.pipeline_ref).ok_or(
            ComputeStatus::MissingPipeline("compute_icb_inherit_pipeline_load"),
        )?;
        let mtlb = load_mtlb(
            state,
            host,
            task_id,
            pipeline.kernel_func_ref,
            AirLoadRail::Compute,
        )
        .ok_or(ComputeStatus::MissingMtlb("compute_icb_inherit_mtlb_load"))?;
        let pso = new_icb_compute_pso(&session.device, &mtlb).map_err(ComputeStatus::from)?;
        session.encoder.set_compute_pipeline_state(&pso);
        session.retained_psos.push(pso);
    }

    let mut staged_bufs = Vec::new();
    let mut mtl_buffers = Vec::new();

    // Kernel buffers when inheritBuffers — Metal uses parent-encoder binds.
    if desc.inherit_buffers() {
        for b in &acc.buffers {
            if b.buffer_ref == 0 {
                continue;
            }
            // Metal's kernel buffer argument table has
            // `REIMS_VGPU_METAL_MAX_BUFFERS` entries, and
            // `setBuffer:offset:atIndex:` past the end raises an out-of-range
            // exception — which aborts the process instead of declining, so the
            // bound has to be checked before the call and not after. `b.index`
            // comes from the decoded stream, so nothing upstream constrains it.
            //
            // The three sibling bind paths all gate on this limit already: direct
            // compute through `valid_buffer_binding`, and both render paths
            // (direct draw and ICB inheritance) through
            // `draw::MAX_BUFFER_BIND_SLOTS`, which a `const` assertion beside
            // `REIMS_VGPU_METAL_MAX_BUFFERS` pins equal to it. This path had no device-limit gate at all — only the
            // descriptor check below, which the guest disables outright by leaving
            // `max_kernel_buffer_bind_count` at 0.
            if !valid_buffer_binding(b.index) {
                return Err(ComputeStatus::Unsupported(
                    "icb_inherit_buffer_binding_out_of_range",
                ));
            }
            // Narrower than the device limit and separate from it: the guest's own
            // declared per-command bind count for this ICB. Kept as it stands —
            // whether `maxKernelBufferBindCount` is meant to bound *parent-encoder*
            // binds under `inheritBuffers`, as opposed to binds recorded into an
            // ICB command, is not settled from the decoded fields.
            if desc.max_kernel_buffer_bind_count > 0
                && b.index as u64 >= desc.max_kernel_buffer_bind_count as u64
            {
                return Err(ComputeStatus::Unsupported("icb_buffer_index_exceeds_max"));
            }
            let staged = stage_buffer(state, host, task_id, b)?;
            let mtl =
                new_buffer_from_host(&session.device, staged.bytes.as_ptr(), staged.bytes.len())
                    .ok_or(ComputeStatus::MetalFailed(
                        "compute_icb_inherit_buffer_alloc",
                    ))?;
            if b.has_attribute_stride {
                set_buffer_with_attribute_stride(
                    &session.encoder,
                    &mtl,
                    0,
                    b.attribute_stride,
                    b.index as u64,
                );
            } else {
                session
                    .encoder
                    .set_buffer(b.index as u64, Some(mtl.as_ref()), 0);
            }
            session.retained.push(mtl.clone());
            staged_bufs.push(staged);
            mtl_buffers.push(mtl);
        }
    }

    let mut storage_tex = Vec::new();
    let mut mtl_storage = Vec::new();

    // Textures / samplers — prefer argument-buffer packaging (ICB-capable path).
    if !acc.textures.is_empty() || !acc.samplers.is_empty() {
        if acc.pipeline_ref == 0 {
            return Err(ComputeStatus::MissingPipeline(
                "compute_icb_inherit_tex_pipeline_ref_zero",
            ));
        }
        let pipeline = load_compute_pipeline(state, host, task_id, acc.pipeline_ref).ok_or(
            ComputeStatus::MissingPipeline("compute_icb_inherit_tex_pipeline_load"),
        )?;
        let mtlb = load_mtlb(
            state,
            host,
            task_id,
            pipeline.kernel_func_ref,
            AirLoadRail::Compute,
        )
        .ok_or(ComputeStatus::MissingMtlb(
            "compute_icb_inherit_tex_mtlb_load",
        ))?;

        let library = session
            .device
            .new_library_with_data(&mtlb)
            .map_err(|_| ComputeStatus::MetalFailed("compute_icb_inherit_library"))?;
        let names = library.function_names();
        if names.len() != 1 {
            return Err(ComputeStatus::Unsupported("icb_library_function_count"));
        }
        let function = library
            .get_function(&names[0], None)
            .map_err(|_| ComputeStatus::MetalFailed("compute_icb_inherit_function"))?;

        let reflected_layout = match reflect_argument_buffer_layout(&session.device, &function) {
            Ok(layout) => layout,
            Err(error) => {
                use crate::observe::Decline as _;
                crate::observe::Emit::decline("compute_icb_argument_buffer_reflection", &error)
                    .field("task", task_id)
                    .field("pipe", acc.pipeline_ref)
                    .fail_once(acc.pipeline_ref as u64);
                return Err(ComputeStatus::MetalFailed(error.slug()));
            }
        };
        if let Some(ab_layout) = reflected_layout {
            // --- Argument-buffer path (real texture-using ICB kernels) ---
            let mut stream_tex: Vec<_> = acc
                .textures
                .iter()
                .filter(|t| t.texture_ref != 0)
                .cloned()
                .collect();
            stream_tex.sort_by_key(|t| t.index);
            let mut stream_samp: Vec<_> = acc
                .samplers
                .iter()
                .filter(|s| s.sampler_ref != 0)
                .cloned()
                .collect();
            stream_samp.sort_by_key(|s| s.index);

            if stream_tex.len() != ab_layout.textures.len()
                || stream_samp.len() != ab_layout.samplers.len()
            {
                // Structural mismatch: stream bind count must match AB members.
                return Err(ComputeStatus::Unsupported("icb_ab_bind_count_mismatch"));
            }

            // Stage guest textures; access from AB member reflection.
            let mut mtl_texs: Vec<metal::Texture> = Vec::new();
            for (st, abm) in stream_tex.iter().zip(ab_layout.textures.iter()) {
                let is_storage = abm.access != BINDING_ACCESS_READ_ONLY;
                let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + st.index;
                let staged =
                    stage_texture_raw(state, host, task_id, st.texture_ref, binding, is_storage)?;
                // Materialize Metal texture (not set_texture on encoder — only AB).
                let selector = staged.storage_selector_or_refuse(task_id, acc.pipeline_ref)?;
                let (pixel_format, bpp) = storage_image_format(selector);
                let Some(expected_len) = tight_image_bytes(staged.width, staged.height, bpp) else {
                    return Err(ComputeStatus::Unsupported("icb_texture_image_len_overflow"));
                };
                if staged.bytes.len() < expected_len {
                    return Err(ComputeStatus::GuestIo("compute_icb_inherit_texture_short"));
                }
                let td = TextureDescriptor::new();
                td.set_texture_type(MTLTextureType::D2);
                td.set_pixel_format(pixel_format);
                td.set_width(staged.width as u64);
                td.set_height(staged.height as u64);
                td.set_storage_mode(MTLStorageMode::Shared);
                let mut usage = MTLTextureUsage::ShaderRead;
                if is_storage {
                    usage |= MTLTextureUsage::ShaderWrite;
                }
                td.set_usage(usage);
                let tex = crate::backend::metal::raw_metal::new_texture(&session.device, &td)
                    .ok_or(ComputeStatus::MetalFailed(
                        "compute_session_inherit_texture_alloc_failed",
                    ))?;
                let region = MTLRegion {
                    origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
                    size: metal::MTLSize {
                        width: staged.width as u64,
                        height: staged.height as u64,
                        depth: 1,
                    },
                };
                tex.replace_region(
                    region,
                    0,
                    staged.bytes.as_ptr() as *const _,
                    (staged.width as u64) * (bpp as u64),
                );
                // Residency for resources referenced through the AB.
                let res_usage = if is_storage {
                    MTLResourceUsage::Write | MTLResourceUsage::Read
                } else {
                    MTLResourceUsage::Read
                };
                session.encoder.use_resource(tex.as_ref(), res_usage);
                session.retained_textures.push(tex.clone());
                if is_storage {
                    mtl_storage.push(tex.clone());
                    storage_tex.push(staged);
                }
                mtl_texs.push(tex);
            }

            // Samplers for AB.
            let mut mtl_samps: Vec<metal::SamplerState> = Vec::new();
            for s in &stream_samp {
                let (_entry, desc_bytes) = objects::resolve_descriptor(
                    state,
                    host,
                    task_id,
                    s.sampler_ref,
                    &[OBJECT_TYPE_TYPE7],
                )
                .map_err(|rung| {
                    ComputeStatus::MissingSampler(crate::observe::ladder_slugs!(
                        "compute_icb_inherit_ab_sampler"
                    )(rung))
                })?;
                if desc_bytes.len() < 4 || ld32(&desc_bytes) != TYPE7_OBJECT_SAMPLER {
                    return Err(ComputeStatus::MissingSampler(
                        "compute_icb_inherit_ab_sampler_bad_tag",
                    ));
                }
                let sd = decode_sampler_descriptor(&desc_bytes).map_err(|_| {
                    ComputeStatus::MissingSampler(crate::observe::ladder_slug!(
                        "compute_icb_inherit_ab_sampler",
                        desc_decode
                    ))
                })?;
                // AB-resident samplers must support argument buffers.
                let reims_vgpu = crate::runtime::draw::sampler_record(
                    REIMS_VGPU_BINDING_SAMPLER_BASE + s.index,
                    &sd,
                    s.has_lod_clamp.then_some((s.lod_min_bits, s.lod_max_bits)),
                    true,
                );
                let mut err_buf = [0i8; 256];
                let samp = make_explicit_sampler(
                    &session.device,
                    &reims_vgpu,
                    (err_buf.as_mut_ptr(), err_buf.len()),
                )
                .map_err(|_| ComputeStatus::MetalFailed("compute_icb_inherit_ab_sampler_make"))?;
                mtl_samps.push(samp);
            }

            // Encode argument buffer.
            let arg_enc = crate::backend::metal::raw_metal::new_argument_encoder(
                &function,
                ab_layout.buffer_index,
            )
            .ok_or(ComputeStatus::MetalFailed(
                "compute_session_argument_encoder_unavailable",
            ))?;
            let ab_len = arg_enc.encoded_length();
            if ab_len == 0 {
                return Err(ComputeStatus::MetalFailed(
                    "compute_icb_inherit_ab_zero_len",
                ));
            }
            let ab = crate::backend::metal::raw_metal::new_buffer(
                &session.device,
                ab_len,
                metal::MTLResourceOptions::StorageModeShared,
            )
            .ok_or(ComputeStatus::MetalFailed(
                "compute_session_argument_buffer_alloc_failed",
            ))?;
            arg_enc.set_argument_buffer(&ab, 0);
            for (tex, abm) in mtl_texs.iter().zip(ab_layout.textures.iter()) {
                arg_enc.set_texture(abm.argument_index, tex.as_ref());
            }
            for (samp, abm) in mtl_samps.iter().zip(ab_layout.samplers.iter()) {
                arg_enc.set_sampler_state(abm.argument_index, samp.as_ref());
            }
            session.retained.push(ab.clone());
            // Keep sampler states alive.
            for s in mtl_samps {
                session.retained_samplers.push(s);
            }

            // Bind AB as kernel buffer: parent encoder always (residency / inherit path).
            session
                .encoder
                .set_buffer(ab_layout.buffer_index, Some(ab.as_ref()), 0);
            // When ICB does not inherit buffers, also patch each command in range.
            if !desc.inherit_buffers() {
                if let Some(icb_ref) = icb {
                    for i in 0..range_length {
                        let idx = range_location.saturating_add(i);
                        if idx >= icb_ref.size() {
                            break;
                        }
                        let cmd = icb_ref.indirect_compute_command_at_index(idx);
                        cmd.set_kernel_buffer(ab_layout.buffer_index, Some(ab.as_ref()), 0);
                    }
                }
            }
            let _ = BINDING_ACCESS_WRITE_ONLY;
        } else {
            // --- Direct encoder binds (non-AB kernels; ICB-capable buffer-only PSO) ---
            if !acc.textures.is_empty() {
                let mut err_buf = [0i8; 256];
                let usages = match reflect_compute_textures_mtlb(
                    &mtlb,
                    (err_buf.as_mut_ptr(), err_buf.len()),
                ) {
                    Ok(u) => u,
                    Err(st) => return Err(ComputeStatus::MetalBackend(st)),
                };
                let mut staged_tex = Vec::new();
                for t in &acc.textures {
                    if t.texture_ref == 0 {
                        continue;
                    }
                    let binding = REIMS_VGPU_BINDING_TEXTURE_BASE + t.index;
                    let is_storage = texture_binds_as_storage(&usages, binding);
                    staged_tex.push(stage_texture_raw(
                        state,
                        host,
                        task_id,
                        t.texture_ref,
                        binding,
                        is_storage,
                    )?);
                }

                let (mut storage, sampled) =
                    split_staged_textures(&mut staged_tex, task_id, acc.pipeline_ref)?;

                let mut err_buf = [0i8; 256];
                let err = (err_buf.as_mut_ptr(), err_buf.len());
                let mut mtl_images = Vec::new();
                let rc = bind_storage_images(
                    &session.device,
                    &session.encoder,
                    &mut storage,
                    &mut mtl_images,
                    err,
                );
                if !rc.is_ok() {
                    return Err(ComputeStatus::MetalFailed(
                        "compute_icb_inherit_bind_storage",
                    ));
                }
                let mut mtl_sampled = Vec::new();
                let rc = bind_compute_sampled_images(
                    &session.device,
                    &session.encoder,
                    &sampled,
                    &mut mtl_sampled,
                    err,
                );
                if !rc.is_ok() {
                    return Err(ComputeStatus::MetalFailed(
                        "compute_icb_inherit_bind_sampled",
                    ));
                }
                let mut mtl_img_iter = mtl_images.into_iter();
                for t in staged_tex {
                    if t.is_storage {
                        let mtl = mtl_img_iter.next().ok_or(ComputeStatus::MetalFailed(
                            "compute_icb_inherit_storage_image_missing",
                        ))?;
                        session.retained_textures.push(mtl.clone());
                        mtl_storage.push(mtl);
                        storage_tex.push(t);
                    }
                }
                for mtl in mtl_sampled {
                    session.retained_textures.push(mtl);
                }
            }

            if !acc.samplers.is_empty() {
                let mut reims_vgpu_samplers: Vec<ReimsVgpuSampler> = Vec::new();
                for s in &acc.samplers {
                    if s.sampler_ref == 0 {
                        continue;
                    }
                    let (_entry, desc_bytes) = objects::resolve_descriptor(
                        state,
                        host,
                        task_id,
                        s.sampler_ref,
                        &[OBJECT_TYPE_TYPE7],
                    )
                    .map_err(|rung| {
                        ComputeStatus::MissingSampler(crate::observe::ladder_slugs!(
                            "compute_icb_inherit_sampler"
                        )(rung))
                    })?;
                    if desc_bytes.len() < 4 || ld32(&desc_bytes) != TYPE7_OBJECT_SAMPLER {
                        return Err(ComputeStatus::MissingSampler(
                            "compute_icb_inherit_sampler_bad_tag",
                        ));
                    }
                    let sd = decode_sampler_descriptor(&desc_bytes).map_err(|_| {
                        ComputeStatus::MissingSampler(crate::observe::ladder_slug!(
                            "compute_icb_inherit_sampler",
                            desc_decode
                        ))
                    })?;
                    reims_vgpu_samplers.push(crate::runtime::draw::sampler_record(
                        REIMS_VGPU_BINDING_SAMPLER_BASE + s.index,
                        &sd,
                        s.has_lod_clamp.then_some((s.lod_min_bits, s.lod_max_bits)),
                        false,
                    ));
                }
                let mut err_buf = [0i8; 256];
                let err = (err_buf.as_mut_ptr(), err_buf.len());
                let rc = bind_compute_samplers(
                    &session.device,
                    &session.encoder,
                    &reims_vgpu_samplers,
                    err,
                );
                if !rc.is_ok() {
                    return Err(ComputeStatus::MetalFailed(
                        "compute_icb_inherit_bind_samplers",
                    ));
                }
            }
        }
    }

    if staged_bufs.is_empty() && storage_tex.is_empty() {
        return Ok(None);
    }
    Ok(Some(nested_job_from_icb_resources(
        staged_bufs,
        mtl_buffers,
        storage_tex,
        mtl_storage,
    )))
}

/// Finish an open session at compute-segment end (no-op if none).
pub fn finish_session<M: HostMemory + HostOps>(
    session: &mut Option<ComputeSession>,
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
) -> Option<ComputeStatus> {
    session.take().map(|s| s.finish(host, state, task_id))
}

#[cfg(test)]
mod tests {

    use super::*;
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::contract::endian::{st32, st64};
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::runtime::decode::resource::{
        list_object_entry_offset, OBJECT_LIST_ENTRY_LEN, OBJECT_TYPE_BUFFER, RESOURCE_PAGE_SHIFT,
    };
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::runtime::gva_mem;
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    use crate::runtime::gva_mem::write_task_gva_arm64e;

    use crate::runtime::host::FakeHost;

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    #[test]
    fn metal_reflection_status_survives_the_session_handoff() {
        use crate::observe::{Emit, Refusal as _};

        let status = crate::backend::metal::error::Status::execute(
            "metal_compute_reflection_pso_create_failed",
        );
        let carried = ComputeStatus::MetalBackend(status);
        assert_eq!(
            carried.refusal(),
            Some("metal_compute_reflection_pso_create_failed")
        );
        assert_eq!(
            Emit::refusal("compute_session", &carried)
                .expect("the session must preserve the backend refusal")
                .render(),
            "compute_session reason=metal_compute_reflection_pso_create_failed \
             class=execute recovery=metal_failed"
        );
    }

    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    #[test]
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    fn control_if_else_spi_session_commits() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));

        // Condition buffer: u32 == 5 at offset 0.
        let cond = 5u32.to_le_bytes();
        let buf_gva = 5u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &cond);
        let mut bdesc = vec![0u8; 16];
        st64(&mut bdesc[0..], 4);
        st32(&mut bdesc[8..], 5);
        let bdesc_gva = 0x180u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
        {
            let off = list_object_entry_offset(7, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }

        let mut session = ComputeSession::open(0).expect("metal session");
        let start = ComputeCommand {
            kind: Kind::ControlStartIf,
            condition_buffer_ref: 7,
            condition_buffer_offset: 0,
            condition_comparison: 2, // Equal
            condition_reference_value: 5,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &start),
            ComputeStatus::Ok
        );
        assert_eq!(session.control_depth, 1);

        let els = ComputeCommand {
            kind: Kind::ControlStartElse,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &els),
            ComputeStatus::Ok
        );

        let end = ComputeCommand {
            kind: Kind::ControlEndIf,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &end),
            ComputeStatus::Ok
        );
        assert_eq!(session.control_depth, 0);
        assert_eq!(session.finish(&mut host, &mut state, 1), ComputeStatus::Ok);
    }

    #[test]
    #[cfg(all(feature = "backend-metal", target_os = "macos"))]
    fn nested_dispatch_under_if_writeback() {
        use crate::runtime::compute_exec::{execute_dispatch_nested, ComputeBufferBind};
        use crate::runtime::decode::compute::Size3;
        use crate::runtime::decode::resource::{
            OBJECT_TYPE_FUNCTION, OBJECT_TYPE_TYPE7, PIPELINE_TAG_KERNEL_FUNC, TYPE7_FIRST_TLVS,
            TYPE7_OBJECT_COMPUTE_PIPELINE,
        };
        use std::path::PathBuf;

        let mtlb_paths =
            [PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/compute_mul3add1.mtlb")];
        let mtlb = mtlb_paths
            .iter()
            .find_map(|p| std::fs::read(p).ok())
            .expect("compute_mul3add1.mtlb fixture");

        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        gva_mem::define_task_pages_arm64e(&mut host, &mut state, 4, 8);
        assert!(state.set_object_list(1, 0, 32));

        // Condition == 1 at buffer ref 8.
        let cond = 1u32.to_le_bytes();
        let cond_gva = 4u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], cond_gva, &cond);
        let mut cdesc = vec![0u8; 16];
        st64(&mut cdesc[0..], 4);
        st32(&mut cdesc[8..], 4);
        let cdesc_gva = 0x100u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], cdesc_gva, &cdesc);
        {
            let off = list_object_entry_offset(8, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&cdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }

        // Kernel function + pipeline + data buffer (same shape as mul3add1 unit).
        let blob_gva = 5u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], blob_gva, &mtlb);
        let mut fdesc = vec![0u8; 32];
        st64(&mut fdesc[0..], blob_gva);
        st32(&mut fdesc[8..], mtlb.len() as u32);
        let fdesc_gva = 0x140u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], fdesc_gva, &fdesc);
        {
            let off = list_object_entry_offset(5, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_FUNCTION as u32) | (32u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&fdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }
        let mut pdesc = vec![0u8; 32];
        st32(&mut pdesc[0..], TYPE7_OBJECT_COMPUTE_PIPELINE);
        st32(&mut pdesc[4..], 32);
        pdesc[TYPE7_FIRST_TLVS] = 1;
        pdesc[TYPE7_FIRST_TLVS + 1] = PIPELINE_TAG_KERNEL_FUNC;
        pdesc[TYPE7_FIRST_TLVS + 2] = 4;
        st32(&mut pdesc[TYPE7_FIRST_TLVS + 3..], 5);
        let pdesc_gva = 0x180u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], pdesc_gva, &pdesc);
        {
            let off = list_object_entry_offset(6, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_TYPE7 as u32) | (32u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&pdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }
        let data = [1u32, 2, 3, 4];
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buf_gva = 6u64 << RESOURCE_PAGE_SHIFT;
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);
        let mut bdesc = vec![0u8; 16];
        st64(&mut bdesc[0..], 16);
        st32(&mut bdesc[8..], 6);
        let bdesc_gva = 0x1c0u64;
        write_task_gva_arm64e(&mut host, &state.tasks[1], bdesc_gva, &bdesc);
        {
            let off = list_object_entry_offset(7, 32).unwrap();
            let mut le = [0u8; OBJECT_LIST_ENTRY_LEN];
            let packed = (OBJECT_TYPE_BUFFER as u32) | (16u32 << 8);
            st32(&mut le[0..], packed);
            le[4..12].copy_from_slice(&bdesc_gva.to_le_bytes());
            write_task_gva_arm64e(&mut host, &state.tasks[1], off, &le);
        }

        // Phase A: nested dispatch alone on a session (no control SPI).
        {
            let mut session = ComputeSession::open(0).expect("session");
            let mut acc = ComputeAccum::default();
            acc.set_pipeline(6);
            acc.buffers.push(ComputeBufferBind {
                index: 0,
                buffer_ref: 7,
                offset: 0,
                attribute_stride: 0,
                has_attribute_stride: false,
            });
            let dcmd = ComputeCommand {
                kind: Kind::DispatchThreadgroups,
                grid: Size3 { x: 1, y: 1, z: 1 },
                threads_per_threadgroup: Size3 { x: 4, y: 1, z: 1 },
                ..Default::default()
            };
            assert_eq!(
                execute_dispatch_nested(&mut state, &mut host, 1, &acc, &dcmd, &mut session),
                ComputeStatus::Ok
            );
            assert_eq!(session.nested_jobs.len(), 1);
            assert_eq!(session.finish(&mut host, &mut state, 1), ComputeStatus::Ok);
            let mut back = [0u8; 16];
            assert!(gva_mem::read_task_gva(
                &host,
                &state.tasks[1],
                buf_gva,
                &mut back,
                PAGE_SHIFT_ARM64E
            )
            .is_ok());
            let out: Vec<u32> = back
                .chunks(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(out, vec![4, 7, 10, 13], "session-only nested writeback");
        }

        // Reset data for phase B (if-wrapped).
        let data = [1u32, 2, 3, 4];
        let data_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_task_gva_arm64e(&mut host, &state.tasks[1], buf_gva, &data_bytes);

        // Phase B: if wraps nested dispatch. Concurrent encoder is the intended
        // SPI host for encodeStartIf. Wire comparison is the Reims VGPU encoder's enum
        // (not MTLCompareFunction): Equal=0 for buffer==reference (probed).
        let mut session = ComputeSession::open(1).expect("session");
        let start = ComputeCommand {
            kind: Kind::ControlStartIf,
            condition_buffer_ref: 8,
            condition_comparison: 0, // SPI Equal (buffer == reference)
            condition_reference_value: 1,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &start),
            ComputeStatus::Ok
        );
        let mut acc = ComputeAccum::default();
        acc.set_pipeline(6);
        acc.buffers.push(ComputeBufferBind {
            index: 0,
            buffer_ref: 7,
            offset: 0,
            attribute_stride: 0,
            has_attribute_stride: false,
        });
        let dcmd = ComputeCommand {
            kind: Kind::DispatchThreadgroups,
            grid: Size3 { x: 1, y: 1, z: 1 },
            threads_per_threadgroup: Size3 { x: 4, y: 1, z: 1 },
            ..Default::default()
        };
        assert_eq!(
            execute_dispatch_nested(&mut state, &mut host, 1, &acc, &dcmd, &mut session),
            ComputeStatus::Ok
        );
        let end = ComputeCommand {
            kind: Kind::ControlEndIf,
            ..Default::default()
        };
        assert_eq!(
            session.encode_control(&state, &host, 1, &end),
            ComputeStatus::Ok
        );
        assert_eq!(session.finish(&mut host, &mut state, 1), ComputeStatus::Ok);
        let mut back = [0u8; 16];
        assert!(gva_mem::read_task_gva(
            &host,
            &state.tasks[1],
            buf_gva,
            &mut back,
            PAGE_SHIFT_ARM64E
        )
        .is_ok());
        let out: Vec<u32> = back
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(out, vec![4, 7, 10, 13], "if-wrapped nested writeback");
    }

    #[test]
    fn icb_latches_sequencing_block() {
        let mut host = FakeHost::new();
        let mut state = DeviceState::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        let mut seg = ComputeSegment::default();
        let cmd = ComputeCommand {
            kind: Kind::ExecuteCommandsInBuffer,
            indirect_command_buffer_ref: 1,
            ..ComputeCommand::default()
        };
        let st = apply_sequencing(&mut state, &mut host, 1, &cmd, &mut seg);
        // Missing list entry → MissingBuffer; latches sequencing block.
        // Non-Apple metal stubs may short-circuit to NoMetal (Linux product).
        assert!(
            matches!(
                st,
                ComputeStatus::MissingBuffer(_)
                    | ComputeStatus::Unsupported(_)
                    | ComputeStatus::NoMetal(_)
            ),
            "unexpected {st:?}"
        );
        assert_eq!(seg.block, Some(SequencingBlock::IndirectCommandBuffer));
        if let Some(s) = seg.session.take() {
            let _ = s.finish(&mut host, &mut state, 1);
        }
    }
}
