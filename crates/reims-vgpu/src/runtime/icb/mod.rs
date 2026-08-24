//! Product-path MTLIndirectCommandBuffer materialization + compute command fills.
//!
//! ## Create (wire)
//!
//! Guest create is serialized by
//! `PGSerializer newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:`
//! into an ICB construction body including a 52-byte command
//! **layout** at `+0x1c`. Product records the semantic declaration per device,
//! task, and ICB reference.
//!
//! ## Command fills — buffer-backed, not stream opcodes
//!
//! There is **no** Reims VGPU compute-stream opcode for
//! `indirectComputeCommandAtIndex` fills. Guest CPU
//! writes into an ICB backing buffer via
//! `PGSerializerIndirectComputeCommand` (setPSO / setKernelBuffer /
//! concurrentDispatch*). Command slots use the layout from create:
//! - type `0x20` = concurrentDispatchThreadgroups, `0x40` = …Threads
//! - pipeline object-list ref at `pipelineStateOffset`
//! - kernel binds at `kernelBufferBindOffset` (0x14 B: ref@0, va@4, gpuva@0xc)
//! - dispatch args at `commandArgumentsOffset` (3×u64 grid + 3×u64 tptg)
//!
//! [`decode_icb_command_range`] decodes registered command memory into typed
//! compute or render fills. The render executor replays render fills through
//! Vulkan. Compute ICB execution remains a typed unsupported operation; the
//! decoder does not imply that a backend executes every returned command kind.

use crate::runtime::Device;
use reims_vgpu_core::endian::{ld32, ld64}; // ld64: 0x1d1 gpu_address + dispatch args

use crate::runtime::decode::resource::{
    decode_serializer_resource, icb_layout_attribute_stride_slot_count,
    icb_layout_kernel_tg_slot_count, icb_layout_table_len, Descriptor as ResourceDescriptor,
    IcbCommandLayout, IndirectCommandBufferDescriptor, ObjectKind, ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE,
    ICB_BUFFER_BIND_STRIDE, ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS,
    ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS, ICB_CMD_TYPE_DRAW, ICB_CMD_TYPE_DRAW_INDEXED,
    ICB_CMD_TYPE_DRAW_INDEXED_PATCHES, ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS,
    ICB_CMD_TYPE_DRAW_MESH_THREADS, ICB_CMD_TYPE_DRAW_PATCHES, ICB_CONCURRENT_DISPATCH_ARGS_LEN,
    ICB_DRAW_INDEXED_PATCHES_ARGS_LEN, ICB_DRAW_MESH_ARGS_LEN, ICB_DRAW_PATCHES_ARGS_LEN,
    ICB_TESSELLATION_FACTOR_LEN, ICB_TG_MEMORY_STRIDE,
}; // ICB_TG_MEMORY_STRIDE: object + kernel TG length tables
#[cfg(test)]
use crate::runtime::decode::resource::{
    MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES, MTL_INDIRECT_CMD_DRAW_PATCHES,
}; // slot-encoder fixtures only
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::objects;

/// A refusal on the indirect-command-buffer rail.
///
/// Every variant carries the registered slug naming **which** check refused.
/// Before that payload existed the five variants spoke for 153 checks — `Args`
/// alone for 84 — so a guest whose ICB never executed produced a log line
/// indistinguishable from thirty other bugs. The variant is the class; the slug
/// is the check.
///
/// There is deliberately no `Ok`: every function on this rail returns
/// `Result<_, IcbStatus>`, so success is `Ok(..)` and this type is *always* a
/// refusal. The old `Ok` variant was never constructed anywhere in the crate —
/// it survived only as an unreachable `Err(IcbStatus::Ok)` match arm — and
/// keeping it would have forced [`crate::observe::Refusal`] where
/// [`crate::observe::Decline`] is the honest shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbStatus {
    /// A guest object, a cache entry, or a required buffer is not there.
    Missing(&'static str),
    /// A descriptor was found but is the wrong type or does not decode.
    BadDescriptor(&'static str),
    /// A host backend call failed.
    BackendFailed(&'static str),
    /// The decoded arguments do not satisfy the contract: a span past the end,
    /// a zero count, an unknown wire tag.
    Args(&'static str),
    /// The record decoded and is well-formed, and this device does not
    /// implement what it asks for. Distinct from [`Self::Args`], which says the
    /// guest's bytes were the problem — here the guest is blameless and the
    /// answer is simply not built.
    Unsupported(&'static str),
}

impl crate::observe::Decline for IcbStatus {
    fn slug(&self) -> &'static str {
        match self {
            Self::Missing(s)
            | Self::BadDescriptor(s)
            | Self::BackendFailed(s)
            | Self::Args(s)
            | Self::Unsupported(s) => s,
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "class",
            match self {
                Self::Missing(_) => "missing",
                Self::BadDescriptor(_) => "bad_descriptor",
                Self::BackendFailed(_) => "backend_failed",
                Self::Args(_) => "args",
                Self::Unsupported(_) => "unsupported",
            }
            .to_string(),
        )]
    }
}

/// Carry an ICB refusal onto the compute rail without losing its name.
///
/// The compute rail is where an ICB refusal actually reaches the sink: the
/// session hands a [`crate::runtime::compute_exec::ComputeStatus`] back to
/// `exec`, which logs it as `compute_record reason=<slug>`. Before this
/// conversion existed each boundary invented its own coarse literal —
/// `icb_resolve_bad_descriptor_or_args` spoke for `BadDescriptor` *and* `Args`,
/// i.e. for 93 of this file's checks at once — so the reason died one frame
/// short of the log. Forwarding `self.slug()` means the check that refused is
/// the reason that prints.
impl From<IcbStatus> for crate::runtime::compute_exec::ComputeStatus {
    fn from(e: IcbStatus) -> Self {
        use crate::observe::Decline;
        let slug = e.slug();
        match e {
            IcbStatus::Missing(_) => Self::MissingBuffer(slug),
            IcbStatus::BadDescriptor(_) | IcbStatus::Args(_) | IcbStatus::Unsupported(_) => {
                Self::Unsupported(slug)
            }
            IcbStatus::BackendFailed(_) => Self::BackendFailed(slug),
        }
    }
}

/// One kernel-buffer bind for an ICB compute command fill (Metal setKernelBuffer).
#[derive(Clone, Debug, Default)]
pub struct IcbKernelBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    /// Byte offset into the type-1 buffer (host fill API, or resolved from [`Self::wire_va`]).
    pub offset: u64,
    /// Absolute guest VA from bind record `va@+4` (PGSerializer: base+offset).
    /// `0` means host-only fill / ref-at-base. Resolved to [`Self::offset`] before stage.
    pub wire_va: u64,
    /// Dynamic attribute stride (`setKernelBuffer:offset:attributeStride:atIndex:`).
    /// Wire: u64 at `attributeStrideOffset + index*8`; 0 = no stride API / default.
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

/// Dispatch form recorded into an ICB compute command (Metal concurrent* only).
#[derive(Clone, Copy, Debug)]
pub enum IcbFillDispatch {
    /// `concurrentDispatchThreadgroups:threadsPerThreadgroup:`
    ConcurrentThreadgroups {
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
    },
    /// `concurrentDispatchThreads:threadsPerThreadgroup:`
    ConcurrentThreads {
        threads_x: u32,
        threads_y: u32,
        threads_z: u32,
        tg_x: u32,
        tg_y: u32,
        tg_z: u32,
    },
}

/// One kernel-threadgroup-memory length for an ICB compute command fill.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IcbThreadgroupMemory {
    pub index: u32,
    /// Byte length (`setThreadgroupMemoryLength:atIndex:`); 0 clears the slot.
    pub length: u64,
}

/// Arguments for product-path fill of one compute command slot.
#[derive(Clone, Debug)]
pub struct IcbComputeFill {
    pub command_index: u32,
    /// Compute-pipeline object-list ref (kernel function + optional stage-in).
    pub pipeline_ref: u32,
    pub buffers: Vec<IcbKernelBufferBind>,
    /// `setThreadgroupMemoryLength:atIndex:` entries (wire: u64 lengths table).
    pub threadgroup_memory: Vec<IcbThreadgroupMemory>,
    /// `setBarrier` when true, `clearBarrier` when false (wire: u32 at barrierOffset).
    pub barrier: bool,
    pub dispatch: IcbFillDispatch,
}

/// Which encoder state an indirect command inherits instead of declaring in
/// its own slot. Inherited and command-local state are exclusive sources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct IcbInheritance {
    pipeline_state: bool,
    buffers: bool,
}

/// Stage for a render ICB buffer bind (layout table + Metal set*Buffer API).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IcbRenderBindStage {
    #[default]
    Vertex,
    Fragment,
    /// Object-shader stage (`setObjectBuffer`); wire at `objectBufferBindOffset`.
    Object,
    /// Mesh-shader stage (`setMeshBuffer`); wire at `meshBufferBindOffset`.
    Mesh,
}

/// One buffer bind for a render ICB command fill.
#[derive(Clone, Debug, Default)]
pub struct IcbRenderBufferBind {
    pub index: u32,
    pub buffer_ref: u32,
    /// Byte offset into the type-1 buffer (host fill API, or resolved from [`Self::wire_va`]).
    pub offset: u64,
    /// Absolute guest VA from bind record `va@+4` (PGSerializer: base+offset).
    /// `0` means host-only fill / ref-at-base. Resolved to [`Self::offset`] before stage.
    pub wire_va: u64,
    /// Dynamic attribute stride for vertex binds
    /// (`setVertexBuffer:offset:attributeStride:atIndex:`). Wire: u64 at
    /// `attributeStrideOffset + index*8`. Non-vertex stages ignore this field.
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
    /// Legacy convenience: `true` means [`IcbRenderBindStage::Fragment`].
    /// Prefer [`Self::stage`]; when `stage` is default Vertex and this is true,
    /// treat as Fragment (older call sites).
    pub is_fragment: bool,
    /// Bind stage (vertex / fragment / object / mesh). When `Object` or `Mesh`,
    /// overrides `is_fragment`.
    pub stage: IcbRenderBindStage,
}

impl IcbRenderBufferBind {
    /// Effective stage after reconciling `stage` and legacy `is_fragment`.
    pub fn effective_stage(&self) -> IcbRenderBindStage {
        match self.stage {
            IcbRenderBindStage::Object | IcbRenderBindStage::Mesh => self.stage,
            IcbRenderBindStage::Fragment => IcbRenderBindStage::Fragment,
            IcbRenderBindStage::Vertex if self.is_fragment => IcbRenderBindStage::Fragment,
            IcbRenderBindStage::Vertex => IcbRenderBindStage::Vertex,
        }
    }
}

/// Tessellation-factor buffer recorded at layout `tessellationFactorOffset`.
/// u32 ref@0 · u64 va@4 · u64 gpuva@0xc · u64 instanceStride@0x14.
#[derive(Clone, Copy, Debug, Default)]
pub struct IcbTessellationFactor {
    pub buffer_ref: u32,
    pub offset: u64,
    pub wire_va: u64,
    pub instance_stride: u64,
}

/// Draw form recorded into a render ICB command.
#[derive(Clone, Copy, Debug)]
pub enum IcbRenderDraw {
    /// command type `0x1` — drawPrimitives
    Primitives {
        primitive_type: u16,
        vertex_start: u64,
        vertex_count: u64,
        instance_count: u64,
        base_instance: u64,
    },
    /// command type `0x2` — drawIndexedPrimitives (PGSerializer layout).
    Indexed {
        primitive_type: u16,
        /// MTLIndexType (UInt16=0, UInt32=1).
        index_type: u16,
        index_buffer_ref: u32,
        index_count: u64,
        /// Byte offset into the index type-1 buffer (host fill or resolved from wire VA).
        index_buffer_offset: u64,
        /// Absolute guest VA of the index range (`va@+0x10` in DrawIndexed args); `0` = base.
        index_wire_va: u64,
        instance_count: u64,
        base_vertex: i64,
        base_instance: u64,
    },
    /// command type `0x4` — drawPatches (host RE PGSerializerIndirectRenderCommand).
    Patches {
        number_of_patch_control_points: u16,
        patch_start: u64,
        patch_count: u64,
        /// Optional patch-index buffer object-list ref (`0` = none / null Metal buffer).
        patch_index_buffer_ref: u32,
        patch_index_buffer_offset: u64,
        patch_index_wire_va: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor: IcbTessellationFactor,
    },
    /// command type `0x8` — drawIndexedPatches.
    IndexedPatches {
        number_of_patch_control_points: u16,
        patch_start: u64,
        patch_count: u64,
        patch_index_buffer_ref: u32,
        patch_index_buffer_offset: u64,
        patch_index_wire_va: u64,
        control_point_index_buffer_ref: u32,
        control_point_index_buffer_offset: u64,
        control_point_index_wire_va: u64,
        instance_count: u64,
        base_instance: u64,
        tessellation_factor: IcbTessellationFactor,
    },
    /// command type `0x100` — drawMeshThreads. `grid` is threadsPerGrid.
    MeshThreads(IcbMeshDraw),
    /// command type `0x80` — drawMeshThreadgroups. `grid` is
    /// threadgroupsPerGrid.
    MeshThreadgroups(IcbMeshDraw),
}

/// The record both mesh draw commands serialize.
///
/// Wire: three MTLSize as 3×u64 each at `commandArgumentsOffset`, total
/// [`ICB_DRAW_MESH_ARGS_LEN`] (`0x48` from host `setupCommandLayout`). Field
/// order matches Metal SPI `MTLIndirectDrawMesh*Arguments` — grid @0, object TG
/// @0x18, mesh TG @0x30. Fill IMPs are stubs; the layout follows
/// `setupCommandLayout` + concurrent-dispatch packing + SPI field order.
///
/// `drawMeshThreads` (`0x100`) and `drawMeshThreadgroups` (`0x80`) write byte-
/// identical records; the only difference is what the first MTLSize counts,
/// which the two [`IcbRenderDraw`] variants carry. One record, two meanings.
#[derive(Clone, Copy, Debug)]
pub struct IcbMeshDraw {
    /// threadsPerGrid or threadgroupsPerGrid, per the owning variant.
    pub grid: [u32; 3],
    pub object_tg: [u32; 3],
    pub mesh_tg: [u32; 3],
}

impl IcbMeshDraw {
    /// Read the nine u64 dimensions at `args` within `slot`. The caller has
    /// already proved `args + ICB_DRAW_MESH_ARGS_LEN <= slot.len()`.
    fn decode(slot: &[u8], args: usize) -> Self {
        let at = |off: usize| ld64(&slot[args + off..]) as u32;
        Self {
            grid: [at(0), at(8), at(0x10)],
            object_tg: [at(0x18), at(0x20), at(0x28)],
            mesh_tg: [at(0x30), at(0x38), at(0x40)],
        }
    }

    /// Write the nine dimensions at `args` within `slot`. The caller has
    /// already proved `args + ICB_DRAW_MESH_ARGS_LEN <= size`. Test-only, like
    /// its only caller [`encode_render_command_slot`].
    #[cfg(test)]
    fn encode(&self, slot: &mut [u8], args: usize) {
        use reims_vgpu_core::endian::st64;
        for (i, v) in self
            .grid
            .iter()
            .chain(&self.object_tg)
            .chain(&self.mesh_tg)
            .enumerate()
        {
            st64(&mut slot[args + i * 8..], u64::from(*v));
        }
    }
}

/// Arguments for product-path fill of one render command slot.
#[derive(Clone, Debug)]
pub struct IcbRenderFill {
    pub command_index: u32,
    pub pipeline_ref: u32,
    pub buffers: Vec<IcbRenderBufferBind>,
    /// Object-stage TG memory lengths (`setObjectThreadgroupMemoryLength:atIndex:`).
    /// Wire: u64 table at `objectThreadgroupMemoryLengthOffset + index*8`.
    pub object_threadgroup_memory: Vec<IcbThreadgroupMemory>,
    pub draw: IcbRenderDraw,
}

/// Guest ICB command-memory association (backing buffer for CPU fills).
pub use reims_vgpu_protocol::IcbCommandMemory;

/// Decode one filled compute command slot from ICB backing bytes.
///
/// Returns `None` if the slot is empty/reset (command type 0).
pub fn decode_compute_command_slot(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_kernel_binds: u16,
) -> Result<Option<IcbComputeFill>, IcbStatus> {
    decode_compute_command_slot_with_inheritance(
        layout,
        slot,
        max_kernel_binds,
        IcbInheritance::default(),
    )
}

fn decode_compute_command_slot_with_inheritance(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_kernel_binds: u16,
    inheritance: IcbInheritance,
) -> Result<Option<IcbComputeFill>, IcbStatus> {
    let cmd_size = layout.command_size as usize;
    if cmd_size == 0 || slot.len() < cmd_size {
        return Err(IcbStatus::Args("icb_dcs_slot_short"));
    }
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > slot.len() {
        return Err(IcbStatus::Args("icb_dcs_type_offset_oob"));
    }
    let cmd_type = ld32(&slot[type_off..]);
    if cmd_type == 0 {
        return Ok(None);
    }
    let dispatch = match cmd_type {
        ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS
        | ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS => {
            // Both commands serialize the same two MTLSize (grid, threadgroup)
            // as 6xu64; only what the first counts differs.
            let threadgroups = cmd_type == ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS;
            let args = layout.command_arguments_offset as usize;
            if args + ICB_CONCURRENT_DISPATCH_ARGS_LEN > slot.len() {
                return Err(IcbStatus::Args(if threadgroups {
                    "icb_dcs_tg_args_oob"
                } else {
                    "icb_dcs_threads_args_oob"
                }));
            }
            let d = |off: usize| ld64(&slot[args + off..]) as u32;
            let (x, y, z, tg_x, tg_y, tg_z) = (d(0), d(8), d(16), d(24), d(32), d(40));
            if threadgroups {
                IcbFillDispatch::ConcurrentThreadgroups {
                    grid_x: x,
                    grid_y: y,
                    grid_z: z,
                    tg_x,
                    tg_y,
                    tg_z,
                }
            } else {
                IcbFillDispatch::ConcurrentThreads {
                    threads_x: x,
                    threads_y: y,
                    threads_z: z,
                    tg_x,
                    tg_y,
                    tg_z,
                }
            }
        }
        _ => return Err(IcbStatus::Args("icb_dcs_unknown_command_type")),
    };

    // Pipeline ref may be 0 when the ICB was created with inheritPipelineState —
    // PSO then comes from the parent compute encoder at execute (Metal contract).
    let mut pipeline_ref = 0u32;
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > slot.len() {
            return Err(IcbStatus::Args("icb_dcs_pipeline_offset_oob"));
        }
        pipeline_ref = ld32(&slot[off..]);
    }
    if pipeline_ref == 0 && !inheritance.pipeline_state {
        return Err(IcbStatus::Missing("icb_dcs_pipeline_ref_zero"));
    }
    if pipeline_ref != 0 && inheritance.pipeline_state {
        return Err(IcbStatus::Args("icb_dcs_inherited_pipeline_ref_nonzero"));
    }

    let mut buffers = Vec::new();
    if !inheritance.buffers && layout.kernel_buffer_bind_offset != 0 && max_kernel_binds > 0 {
        let base = layout.kernel_buffer_bind_offset as usize;
        for i in 0..max_kernel_binds as usize {
            let off = base + i * ICB_BUFFER_BIND_STRIDE;
            if off + 4 > slot.len() {
                break;
            }
            let buffer_ref = ld32(&slot[off..]);
            if buffer_ref == 0 {
                continue;
            }
            // Bind record 0x14 B: ref@0, va@4, gpuva@0xc. Offset is not a
            // separate field — guest writes base+offset into va (resolved later).
            let wire_va = if off + 0xc <= slot.len() {
                ld64(&slot[off + 4..])
            } else {
                0
            };
            // Attribute stride table (separate from bind record).
            let (attribute_stride, has_attribute_stride) =
                read_attribute_stride(layout, slot, i as u32);
            buffers.push(IcbKernelBufferBind {
                index: i as u32,
                buffer_ref,
                offset: 0,
                wire_va,
                attribute_stride,
                has_attribute_stride,
            });
        }
    }

    // Barrier: u32 at barrierOffset (setBarrier writes 1, clearBarrier writes 0).
    // PGSerializerIndirectComputeCommand setBarrier/clearBarrier.
    let mut barrier = false;
    if layout.barrier_offset != 0 {
        let bo = layout.barrier_offset as usize;
        if bo + 4 <= slot.len() {
            barrier = ld32(&slot[bo..]) != 0;
        }
    }

    // Threadgroup memory: u64 length table at threadgroupMemoryLengthOffset,
    // entry i at + i*8 (setThreadgroupMemoryLength:atIndex:).
    let mut threadgroup_memory = Vec::new();
    let tg_slots = icb_layout_kernel_tg_slot_count(layout);
    if tg_slots > 0 && layout.threadgroup_memory_length_offset != 0 {
        let base = layout.threadgroup_memory_length_offset as usize;
        for i in 0..tg_slots as usize {
            let off = base + i * ICB_TG_MEMORY_STRIDE;
            if off + 8 > slot.len() {
                break;
            }
            let length = ld64(&slot[off..]);
            if length != 0 {
                threadgroup_memory.push(IcbThreadgroupMemory {
                    index: i as u32,
                    length,
                });
            }
        }
    }

    Ok(Some(IcbComputeFill {
        command_index: 0, // caller sets index
        pipeline_ref,
        buffers,
        threadgroup_memory,
        barrier,
        dispatch,
    }))
}

/// Decode one filled **render** command slot (Draw / DrawIndexed) from ICB backing.
pub fn decode_render_command_slot(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_vertex_binds: u16,
    max_fragment_binds: u16,
) -> Result<Option<IcbRenderFill>, IcbStatus> {
    decode_render_command_slot_with_inheritance(
        layout,
        slot,
        max_vertex_binds,
        max_fragment_binds,
        IcbInheritance::default(),
    )
}

fn decode_render_command_slot_with_inheritance(
    layout: &IcbCommandLayout,
    slot: &[u8],
    max_vertex_binds: u16,
    max_fragment_binds: u16,
    inheritance: IcbInheritance,
) -> Result<Option<IcbRenderFill>, IcbStatus> {
    let cmd_size = layout.command_size as usize;
    if cmd_size == 0 || slot.len() < cmd_size {
        return Err(IcbStatus::Args("icb_drs_slot_short"));
    }
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > slot.len() {
        return Err(IcbStatus::Args("icb_drs_type_offset_oob"));
    }
    let cmd_type = ld32(&slot[type_off..]);
    if cmd_type == 0 {
        return Ok(None);
    }

    let mut pipeline_ref = 0u32;
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > slot.len() {
            return Err(IcbStatus::Args("icb_drs_pipeline_offset_oob"));
        }
        pipeline_ref = ld32(&slot[off..]);
    }
    if pipeline_ref == 0 && !inheritance.pipeline_state {
        return Err(IcbStatus::Missing("icb_drs_pipeline_ref_zero"));
    }
    if pipeline_ref != 0 && inheritance.pipeline_state {
        return Err(IcbStatus::Args("icb_drs_inherited_pipeline_ref_nonzero"));
    }

    let tessellation_factor = read_tessellation_factor(layout, slot);

    let mut buffers = Vec::new();
    let push_binds = |buffers: &mut Vec<IcbRenderBufferBind>,
                      base_off: u32,
                      count: u32,
                      stage: IcbRenderBindStage| {
        if base_off == 0 || count == 0 {
            return;
        }
        let base = base_off as usize;
        for i in 0..count as usize {
            let off = base + i * ICB_BUFFER_BIND_STRIDE;
            if off + 4 > slot.len() {
                break;
            }
            let buffer_ref = ld32(&slot[off..]);
            if buffer_ref == 0 {
                continue;
            }
            let wire_va = if off + 0xc <= slot.len() {
                ld64(&slot[off + 4..])
            } else {
                0
            };
            let (attribute_stride, has_attribute_stride) = if stage == IcbRenderBindStage::Vertex {
                read_attribute_stride(layout, slot, i as u32)
            } else {
                (0, false)
            };
            buffers.push(IcbRenderBufferBind {
                index: i as u32,
                buffer_ref,
                offset: 0,
                wire_va,
                attribute_stride,
                has_attribute_stride,
                is_fragment: stage == IcbRenderBindStage::Fragment,
                stage,
            });
        }
    };
    if !inheritance.buffers {
        push_binds(
            &mut buffers,
            layout.vertex_buffer_bind_offset,
            u32::from(max_vertex_binds),
            IcbRenderBindStage::Vertex,
        );
        push_binds(
            &mut buffers,
            layout.fragment_buffer_bind_offset,
            u32::from(max_fragment_binds),
            IcbRenderBindStage::Fragment,
        );
        // Object/mesh bind table sizes from layout offsets (setupCommandLayout order).
        let max_object = icb_layout_stage_bind_count(
            layout.object_buffer_bind_offset,
            layout.mesh_buffer_bind_offset,
        );
        let max_mesh = icb_layout_stage_bind_count(
            layout.mesh_buffer_bind_offset,
            layout.kernel_buffer_bind_offset,
        );
        push_binds(
            &mut buffers,
            layout.object_buffer_bind_offset,
            max_object,
            IcbRenderBindStage::Object,
        );
        push_binds(
            &mut buffers,
            layout.mesh_buffer_bind_offset,
            max_mesh,
            IcbRenderBindStage::Mesh,
        );
    }

    let args = layout.command_arguments_offset as usize;
    let draw = match cmd_type {
        ICB_CMD_TYPE_DRAW => {
            // Pack: u16 prim @0, u64 vertexStart @2, u64 vertexCount @0xa,
            // u64 instanceCount @0x12, u64 baseInstance @0x1a.
            if args + 0x22 > slot.len() {
                return Err(IcbStatus::Args("icb_drs_draw_args_oob"));
            }
            let prim = u16::from_le_bytes([slot[args], slot[args + 1]]);
            IcbRenderDraw::Primitives {
                primitive_type: prim,
                vertex_start: ld64(&slot[args + 2..]),
                vertex_count: ld64(&slot[args + 0xa..]),
                instance_count: ld64(&slot[args + 0x12..]),
                base_instance: ld64(&slot[args + 0x1a..]),
            }
        }
        ICB_CMD_TYPE_DRAW_INDEXED => {
            // DrawIndexed (PGSerializer): u16 prim @0, u16 indexType @2,
            // u32 indexBufferRef @4, u64 indexCount @8, u64 va @0x10, u64 gpuva @0x18,
            // u64 instanceCount @0x20, u64 baseVertex @0x28 (signed bit pattern),
            // u64 baseInstance @0x30.
            if args + 0x38 > slot.len() {
                return Err(IcbStatus::Args("icb_drs_indexed_args_oob"));
            }
            let prim = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let index_type = u16::from_le_bytes([slot[args + 2], slot[args + 3]]);
            let index_buffer_ref = ld32(&slot[args + 4..]);
            if index_buffer_ref == 0 {
                return Err(IcbStatus::Missing("icb_drs_index_buffer_ref_zero"));
            }
            let index_wire_va = ld64(&slot[args + 0x10..]);
            IcbRenderDraw::Indexed {
                primitive_type: prim,
                index_type,
                index_buffer_ref,
                index_count: ld64(&slot[args + 8..]),
                index_buffer_offset: 0, // resolved from index_wire_va when non-zero
                index_wire_va,
                instance_count: ld64(&slot[args + 0x20..]),
                base_vertex: ld64(&slot[args + 0x28..]) as i64,
                base_instance: ld64(&slot[args + 0x30..]),
            }
        }
        ICB_CMD_TYPE_DRAW_PATCHES => {
            // u16 controlPoints@0, u64 patchStart@2, u64 patchCount@0xa,
            // u32 patchIndexRef@0x12, u64 va@0x16, u64 gpuva@0x1e,
            // u64 instanceCount@0x26, u64 baseInstance@0x2e.
            if args + ICB_DRAW_PATCHES_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args("icb_drs_patches_args_oob"));
            }
            let cps = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let patch_index_buffer_ref = ld32(&slot[args + 0x12..]);
            let patch_index_wire_va = ld64(&slot[args + 0x16..]);
            IcbRenderDraw::Patches {
                number_of_patch_control_points: cps,
                patch_start: ld64(&slot[args + 2..]),
                patch_count: ld64(&slot[args + 0xa..]),
                patch_index_buffer_ref,
                patch_index_buffer_offset: 0,
                patch_index_wire_va,
                instance_count: ld64(&slot[args + 0x26..]),
                base_instance: ld64(&slot[args + 0x2e..]),
                tessellation_factor,
            }
        }
        ICB_CMD_TYPE_DRAW_INDEXED_PATCHES => {
            // like DrawPatches through patchIndex, then
            // u32 controlPointIndexRef@0x26, u64 va@0x2a, u64 gpuva@0x32,
            // u64 instanceCount@0x3a, u64 baseInstance@0x42.
            if args + ICB_DRAW_INDEXED_PATCHES_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args("icb_drs_indexed_patches_args_oob"));
            }
            let cps = u16::from_le_bytes([slot[args], slot[args + 1]]);
            let patch_index_buffer_ref = ld32(&slot[args + 0x12..]);
            let patch_index_wire_va = ld64(&slot[args + 0x16..]);
            let control_point_index_buffer_ref = ld32(&slot[args + 0x26..]);
            if control_point_index_buffer_ref == 0 {
                return Err(IcbStatus::Missing("icb_drs_control_point_ref_zero"));
            }
            let control_point_index_wire_va = ld64(&slot[args + 0x2a..]);
            IcbRenderDraw::IndexedPatches {
                number_of_patch_control_points: cps,
                patch_start: ld64(&slot[args + 2..]),
                patch_count: ld64(&slot[args + 0xa..]),
                patch_index_buffer_ref,
                patch_index_buffer_offset: 0,
                patch_index_wire_va,
                control_point_index_buffer_ref,
                control_point_index_buffer_offset: 0,
                control_point_index_wire_va,
                instance_count: ld64(&slot[args + 0x3a..]),
                base_instance: ld64(&slot[args + 0x42..]),
                tessellation_factor,
            }
        }
        ICB_CMD_TYPE_DRAW_MESH_THREADS | ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS => {
            let threads = cmd_type == ICB_CMD_TYPE_DRAW_MESH_THREADS;
            if args + ICB_DRAW_MESH_ARGS_LEN as usize > slot.len() {
                return Err(IcbStatus::Args(if threads {
                    "icb_drs_mesh_threads_args_oob"
                } else {
                    "icb_drs_mesh_threadgroups_args_oob"
                }));
            }
            let mesh = IcbMeshDraw::decode(slot, args);
            if threads {
                IcbRenderDraw::MeshThreads(mesh)
            } else {
                IcbRenderDraw::MeshThreadgroups(mesh)
            }
        }
        _ => return Err(IcbStatus::Args("icb_drs_unknown_command_type")),
    };

    // Object TG memory length table (setupCommandLayout: before kernel TG).
    let mut object_threadgroup_memory = Vec::new();
    let obj_tg_slots = icb_layout_object_tg_slot_count(layout);
    if obj_tg_slots > 0 && layout.object_threadgroup_memory_length_offset != 0 {
        let base = layout.object_threadgroup_memory_length_offset as usize;
        for i in 0..obj_tg_slots as usize {
            let off = base + i * ICB_TG_MEMORY_STRIDE;
            if off + 8 > slot.len() {
                break;
            }
            let length = ld64(&slot[off..]);
            if length != 0 {
                object_threadgroup_memory.push(IcbThreadgroupMemory {
                    index: i as u32,
                    length,
                });
            }
        }
    }

    Ok(Some(IcbRenderFill {
        command_index: 0,
        pipeline_ref,
        buffers,
        object_threadgroup_memory,
        draw,
    }))
}

/// Object-TG length table slot count between layout offsets.
fn icb_layout_object_tg_slot_count(layout: &IcbCommandLayout) -> u32 {
    icb_layout_table_len(
        layout.object_threadgroup_memory_length_offset,
        layout.threadgroup_memory_length_offset,
        ICB_TG_MEMORY_STRIDE,
    )
}

/// Read tessellation-factor table at `tessellationFactorOffset` (host RE).
fn read_tessellation_factor(layout: &IcbCommandLayout, slot: &[u8]) -> IcbTessellationFactor {
    if layout.tessellation_factor_offset == 0 {
        return IcbTessellationFactor::default();
    }
    let off = layout.tessellation_factor_offset as usize;
    if off + ICB_TESSELLATION_FACTOR_LEN > slot.len() {
        return IcbTessellationFactor::default();
    }
    IcbTessellationFactor {
        buffer_ref: ld32(&slot[off..]),
        wire_va: ld64(&slot[off + 4..]),
        offset: 0,
        instance_stride: ld64(&slot[off + 0x14..]),
    }
}

#[cfg(test)]
fn write_tessellation_factor(
    layout: &IcbCommandLayout,
    slot: &mut [u8],
    tf: &IcbTessellationFactor,
) -> Result<(), IcbStatus> {
    if layout.tessellation_factor_offset == 0 {
        return Ok(());
    }
    let off = layout.tessellation_factor_offset as usize;
    if off + ICB_TESSELLATION_FACTOR_LEN > slot.len() {
        return Err(IcbStatus::Args("icb_write_tess_factor_oob"));
    }
    use reims_vgpu_core::endian::{st32, st64};
    st32(&mut slot[off..], tf.buffer_ref);
    let va = if tf.wire_va != 0 { tf.wire_va } else { 0 };
    st64(&mut slot[off + 4..], va);
    st64(&mut slot[off + 0xc..], va);
    st64(&mut slot[off + 0x14..], tf.instance_stride);
    Ok(())
}

/// Bind-table slot count between two layout offsets (`count × 0x14`).
fn icb_layout_stage_bind_count(start: u32, end: u32) -> u32 {
    icb_layout_table_len(start, end, ICB_BUFFER_BIND_STRIDE)
}

/// Encode one render Draw / DrawIndexed command slot (tests / fixtures).
#[cfg(test)]
pub fn encode_render_command_slot(
    layout: &IcbCommandLayout,
    fill: &IcbRenderFill,
) -> Result<Vec<u8>, IcbStatus> {
    use reims_vgpu_core::endian::{st16, st32, st64};
    let size = layout.command_size as usize;
    if size == 0 {
        return Err(IcbStatus::Args("icb_ers_zero_command_size"));
    }
    let mut slot = vec![0u8; size];
    let type_off = layout.command_type_offset as usize;
    if layout.pipeline_state_offset != 0 {
        st32(
            &mut slot[layout.pipeline_state_offset as usize..],
            fill.pipeline_ref,
        );
    }
    for b in &fill.buffers {
        let base = match b.effective_stage() {
            IcbRenderBindStage::Vertex => layout.vertex_buffer_bind_offset,
            IcbRenderBindStage::Fragment => layout.fragment_buffer_bind_offset,
            IcbRenderBindStage::Object => layout.object_buffer_bind_offset,
            IcbRenderBindStage::Mesh => layout.mesh_buffer_bind_offset,
        } as usize;
        let off = base + (b.index as usize) * ICB_BUFFER_BIND_STRIDE;
        if off + ICB_BUFFER_BIND_STRIDE > size {
            return Err(IcbStatus::Args("icb_ers_bind_offset_oob"));
        }
        st32(&mut slot[off..], b.buffer_ref);
        // Wire VA = absolute GVA (base+offset). Prefer explicit wire_va; else 0
        // (host fill uses offset without a wire VA). Same 0x14 packing as
        // setVertexBuffer / setFragmentBuffer (ref@0 · va@4 · gpuva@0xc).
        let va = if b.wire_va != 0 { b.wire_va } else { 0 };
        st64(&mut slot[off + 4..], va);
        st64(&mut slot[off + 0xc..], va); // gpuva same as va for fixtures
        if b.effective_stage() == IcbRenderBindStage::Vertex && b.has_attribute_stride {
            write_attribute_stride(layout, &mut slot, b.index, b.attribute_stride)?;
        }
    }
    // Object TG memory lengths (u64 at objectThreadgroupMemoryLengthOffset + i*8).
    for tg in &fill.object_threadgroup_memory {
        if layout.object_threadgroup_memory_length_offset == 0 {
            return Err(IcbStatus::Args("icb_ers_no_object_tg_table"));
        }
        let off = layout.object_threadgroup_memory_length_offset as usize
            + (tg.index as usize) * ICB_TG_MEMORY_STRIDE;
        if off + 8 > size {
            return Err(IcbStatus::Args("icb_ers_object_tg_offset_oob"));
        }
        st64(&mut slot[off..], tg.length);
    }
    let args = layout.command_arguments_offset as usize;
    match fill.draw {
        IcbRenderDraw::Primitives {
            primitive_type,
            vertex_start,
            vertex_count,
            instance_count,
            base_instance,
        } => {
            if args + 0x22 > size {
                return Err(IcbStatus::Args("icb_ers_draw_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW);
            st16(&mut slot[args..], primitive_type);
            st64(&mut slot[args + 2..], vertex_start);
            st64(&mut slot[args + 0xa..], vertex_count);
            st64(&mut slot[args + 0x12..], instance_count);
            st64(&mut slot[args + 0x1a..], base_instance);
        }
        IcbRenderDraw::Indexed {
            primitive_type,
            index_type,
            index_buffer_ref,
            index_count,
            index_buffer_offset: _,
            index_wire_va,
            instance_count,
            base_vertex,
            base_instance,
        } => {
            if args + 0x38 > size {
                return Err(IcbStatus::Args("icb_ers_indexed_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_INDEXED);
            st16(&mut slot[args..], primitive_type);
            st16(&mut slot[args + 2..], index_type);
            st32(&mut slot[args + 4..], index_buffer_ref);
            st64(&mut slot[args + 8..], index_count);
            st64(&mut slot[args + 0x10..], index_wire_va);
            st64(&mut slot[args + 0x18..], index_wire_va);
            st64(&mut slot[args + 0x20..], instance_count);
            st64(&mut slot[args + 0x28..], base_vertex as u64);
            st64(&mut slot[args + 0x30..], base_instance);
        }
        IcbRenderDraw::Patches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            patch_index_buffer_ref,
            patch_index_buffer_offset: _,
            patch_index_wire_va,
            instance_count,
            base_instance,
            tessellation_factor,
        } => {
            if args + ICB_DRAW_PATCHES_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args("icb_ers_patches_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_PATCHES);
            st16(&mut slot[args..], number_of_patch_control_points);
            st64(&mut slot[args + 2..], patch_start);
            st64(&mut slot[args + 0xa..], patch_count);
            st32(&mut slot[args + 0x12..], patch_index_buffer_ref);
            st64(&mut slot[args + 0x16..], patch_index_wire_va);
            st64(&mut slot[args + 0x1e..], patch_index_wire_va);
            st64(&mut slot[args + 0x26..], instance_count);
            st64(&mut slot[args + 0x2e..], base_instance);
            write_tessellation_factor(layout, &mut slot, &tessellation_factor)?;
        }
        IcbRenderDraw::IndexedPatches {
            number_of_patch_control_points,
            patch_start,
            patch_count,
            patch_index_buffer_ref,
            patch_index_buffer_offset: _,
            patch_index_wire_va,
            control_point_index_buffer_ref,
            control_point_index_buffer_offset: _,
            control_point_index_wire_va,
            instance_count,
            base_instance,
            tessellation_factor,
        } => {
            if args + ICB_DRAW_INDEXED_PATCHES_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args("icb_ers_indexed_patches_args_oob"));
            }
            st32(&mut slot[type_off..], ICB_CMD_TYPE_DRAW_INDEXED_PATCHES);
            st16(&mut slot[args..], number_of_patch_control_points);
            st64(&mut slot[args + 2..], patch_start);
            st64(&mut slot[args + 0xa..], patch_count);
            st32(&mut slot[args + 0x12..], patch_index_buffer_ref);
            st64(&mut slot[args + 0x16..], patch_index_wire_va);
            st64(&mut slot[args + 0x1e..], patch_index_wire_va);
            st32(&mut slot[args + 0x26..], control_point_index_buffer_ref);
            st64(&mut slot[args + 0x2a..], control_point_index_wire_va);
            st64(&mut slot[args + 0x32..], control_point_index_wire_va);
            st64(&mut slot[args + 0x3a..], instance_count);
            st64(&mut slot[args + 0x42..], base_instance);
            write_tessellation_factor(layout, &mut slot, &tessellation_factor)?;
        }
        IcbRenderDraw::MeshThreads(mesh) | IcbRenderDraw::MeshThreadgroups(mesh) => {
            let threads = matches!(fill.draw, IcbRenderDraw::MeshThreads(_));
            if args + ICB_DRAW_MESH_ARGS_LEN as usize > size {
                return Err(IcbStatus::Args(if threads {
                    "icb_ers_mesh_threads_args_oob"
                } else {
                    "icb_ers_mesh_threadgroups_args_oob"
                }));
            }
            st32(
                &mut slot[type_off..],
                if threads {
                    ICB_CMD_TYPE_DRAW_MESH_THREADS
                } else {
                    ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS
                },
            );
            mesh.encode(&mut slot, args);
        }
    }
    let _ = (
        MTL_INDIRECT_CMD_DRAW,
        MTL_INDIRECT_CMD_DRAW_PATCHES,
        MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES,
    );
    Ok(slot)
}

/// Encode one compute command slot into ICB backing bytes (tests / fixtures).
#[cfg(test)]
pub fn encode_compute_command_slot(
    layout: &IcbCommandLayout,
    fill: &IcbComputeFill,
) -> Result<Vec<u8>, IcbStatus> {
    use reims_vgpu_core::endian::{st32, st64};
    let size = layout.command_size as usize;
    if size == 0 {
        return Err(IcbStatus::Args("icb_ecs_zero_command_size"));
    }
    let mut slot = vec![0u8; size];
    let (cmd_type, gx, gy, gz, tx, ty, tz) = match fill.dispatch {
        IcbFillDispatch::ConcurrentThreadgroups {
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        } => (
            ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS,
            grid_x,
            grid_y,
            grid_z,
            tg_x,
            tg_y,
            tg_z,
        ),
        IcbFillDispatch::ConcurrentThreads {
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        } => (
            ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS,
            threads_x,
            threads_y,
            threads_z,
            tg_x,
            tg_y,
            tg_z,
        ),
    };
    let type_off = layout.command_type_offset as usize;
    if type_off + 4 > size {
        return Err(IcbStatus::Args("icb_ecs_type_offset_oob"));
    }
    st32(&mut slot[type_off..], cmd_type);
    if layout.pipeline_state_offset != 0 {
        let off = layout.pipeline_state_offset as usize;
        if off + 4 > size {
            return Err(IcbStatus::Args("icb_ecs_pipeline_offset_oob"));
        }
        st32(&mut slot[off..], fill.pipeline_ref);
    }
    for b in &fill.buffers {
        let off =
            layout.kernel_buffer_bind_offset as usize + (b.index as usize) * ICB_BUFFER_BIND_STRIDE;
        if off + ICB_BUFFER_BIND_STRIDE > size {
            return Err(IcbStatus::Args("icb_ecs_bind_offset_oob"));
        }
        st32(&mut slot[off..], b.buffer_ref);
        let va = if b.wire_va != 0 { b.wire_va } else { 0 };
        st64(&mut slot[off + 4..], va);
        st64(&mut slot[off + 0xc..], va);
        if b.has_attribute_stride {
            write_attribute_stride(layout, &mut slot, b.index, b.attribute_stride)?;
        }
    }
    // Barrier u32 (1 = setBarrier, 0 = clear).
    if layout.barrier_offset != 0 {
        let bo = layout.barrier_offset as usize;
        if bo + 4 > size {
            return Err(IcbStatus::Args("icb_ecs_barrier_offset_oob"));
        }
        st32(&mut slot[bo..], if fill.barrier { 1 } else { 0 });
    }
    // Threadgroup memory length table (u64 per index).
    for tg in &fill.threadgroup_memory {
        let off = layout.threadgroup_memory_length_offset as usize
            + (tg.index as usize) * ICB_TG_MEMORY_STRIDE;
        if off + 8 > size {
            return Err(IcbStatus::Args("icb_ecs_tg_offset_oob"));
        }
        st64(&mut slot[off..], tg.length);
    }
    let args = layout.command_arguments_offset as usize;
    if args + ICB_CONCURRENT_DISPATCH_ARGS_LEN > size {
        return Err(IcbStatus::Args("icb_ecs_dispatch_args_oob"));
    }
    st64(&mut slot[args..], gx as u64);
    st64(&mut slot[args + 8..], gy as u64);
    st64(&mut slot[args + 16..], gz as u64);
    st64(&mut slot[args + 24..], tx as u64);
    st64(&mut slot[args + 32..], ty as u64);
    st64(&mut slot[args + 40..], tz as u64);
    Ok(slot)
}

/// Load and decode an ICB descriptor for `icb_ref` on the task object list.
pub fn load_icb_descriptor<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    icb_ref: u32,
) -> Result<IndirectCommandBufferDescriptor, IcbStatus> {
    if icb_ref == 0 {
        return Err(IcbStatus::Missing("icb_desc_ref_zero"));
    }
    // The two statuses this rail splits the ladder into, stated once: a tag that
    // is not a serializer resource means the guest described something wrongly, while a
    // missing entry or unreadable bytes mean it described nothing this device
    // can see yet.
    let (_entry, desc) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        icb_ref,
        &[ObjectKind::SerializerResource],
    )
    .map_err(|rung| {
        let slug = crate::observe::ladder_slugs!("icb")(rung);
        match rung {
            objects::LadderRung::NoListEntry | objects::LadderRung::DescRead { .. } => {
                IcbStatus::Missing(slug)
            }
            objects::LadderRung::WrongType { .. } => IcbStatus::BadDescriptor(slug),
        }
    })?;
    match decode_serializer_resource(&desc) {
        Ok(ResourceDescriptor::IndirectCommandBuffer(icb)) => {
            note_unapplied_icb_flags(task_id, icb_ref, &icb);
            Ok(icb)
        }
        Ok(_) => Err(IcbStatus::BadDescriptor("icb_desc_not_icb_body")),
        Err(_) => Err(IcbStatus::BadDescriptor(crate::observe::ladder_slug!(
            "icb",
            desc_decode
        ))),
    }
}

/// A flag the guest set on its indirect command buffer that this device decodes
/// and does not apply.
struct IcbFlagDropped(crate::runtime::decode::resource::IcbUnappliedFlag);

impl crate::observe::Decline for IcbFlagDropped {
    fn slug(&self) -> &'static str {
        self.0.slug()
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Report every decoded ICB flag this device drops on the floor.
///
/// Eight of the ten attributed bits in the create body's flag word reach no
/// executor: `supportRayTracing`, `supportDynamicAttributeStride` and the six
/// inherit-state flags added in macOS 26. Each one therefore remains unapplied.
///
/// Counted rather than executed, deliberately. Six of the eight default *on* at
/// both ends, so on a descriptor the guest left alone nothing is lost and every
/// counter here is a healthy zero — which means a **non**-zero reading is the
/// measured argument for implementing that flag, and says which one.
///
/// This sits in [`load_icb_descriptor`] so the count reflects what the guest
/// asked for even before an executor exists.
fn note_unapplied_icb_flags(task_id: u32, icb_ref: u32, desc: &IndirectCommandBufferDescriptor) {
    use crate::observe::Decline as _;
    for flag in desc.unapplied_flags() {
        let decline = IcbFlagDropped(flag);
        crate::runtime::drain::note_store_route(decline.slug());
        crate::observe::Emit::decline("icb_desc_flag", &decline)
            .field("task", task_id)
            .field("icb", icb_ref)
            .field("flags", format!("{:#06x}", desc.flags))
            // The slug is already per flag, so the buffer is the only thing
            // left to key on: one line per ICB per flag, however many times a
            // cache miss reloads the descriptor.
            .fail_once(icb_ref as u64);
    }
}

/// Load the guest's ICB descriptor and record it, on every pathway.
///
/// Returns the descriptor the caller should build against. When the create body
/// no longer matches what was recorded, the recorded command memory is dropped
/// with it: a re-created ICB of a different shape is not the one whose slots the
/// old span held, and decoding the old bytes at the new layout would read
/// whatever happened to be at those offsets rather than refusing.
pub fn resolve_icb_record<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    icb_ref: u32,
) -> Result<IndirectCommandBufferDescriptor, IcbStatus> {
    let desc = load_icb_descriptor(state, host, task_id, icb_ref)?;
    state
        .task_objects
        .indirect_command_buffers
        .record(task_id, icb_ref, desc)
        .map_err(|_| IcbStatus::Args("icb_identity_space_exhausted"))
}

// ---------------------------------------------------------------------------
// ICB auxiliary records and command-memory association
// ---------------------------------------------------------------------------

/// Info-segment opcode for `PGSerializerInfoCommandEncoder icbHostResourceInfo:info:`.
///
/// Full wire record length `0x18` (8 B header + 16 B payload). Payload:
/// `icb_ref:u32 @0`, `buffer_ref:u32 @4`, `gpu_address:u64 @8`.
///
/// **The offsets are right and the last two names are wrong.** Apple's own bytes
/// say this record is a *query*: `+4` is the reply staging buffer and `+8` is
/// the offset into it where the host is being asked to write two `u64`s, which
/// is what the selector's `^{?=QQ}` out-parameter means. The reading below —
/// the ICB's backing buffer and its GPU address — has no derivation behind it;
/// it arrived with the initial import and nothing has tested it against a
/// captured record, because `PGSerializerInfoCommandEncoder` sits in the
/// divergence instrument's `UNCOVERED_CLASSES`.
///
/// The evidence is in [`reims_vgpu_wire::ops::info::Query`], which declares the
/// same three offsets under the other names. Shortest form: `+4` reads the same
/// value in all ten query fixtures, *including* the one whose queried object is
/// itself a buffer with a different ref — so it cannot be that object's backing
/// buffer.
///
/// Repaired: [`apply_icb_host_resource_info`] now declines by name rather than
/// binding the reply pair, and [`IcbHostResourceInfo`] carries the wire crate's
/// field names. The device still never writes the answer the guest asked for —
/// the two `u64`s are unattributed, and `runtime::heap_query` shows the shape a
/// reply takes. The rail is dormant, which is why the wrong reading survived
/// as long as it did: `runtime::icb` reads 0.00% on a driven boot.
///
/// The three constants below are the wire crate's, aliased rather than spelled,
/// so this file cannot drift from the declaration the fixtures pin.
pub const INFO_OP_ICB_HOST_RESOURCE: u32 =
    reims_vgpu_wire::ops::info::OPCODE_ICB_HOST_RESOURCE_INFO;
pub const INFO_OP_ICB_HOST_RESOURCE_RECORD_LEN: u32 = reims_vgpu_wire::ops::info::QUERY_TOTAL_LEN;
pub const INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN: usize =
    std::mem::size_of::<reims_vgpu_wire::ops::info::Query>();

/// Decoded `0x1d1` `icbHostResourceInfo:info:` payload.
///
/// The field names are [`reims_vgpu_wire::ops::info::Query`]'s, because this
/// record *is* that record — ten selectors write the identical 24 bytes and
/// differ only in opcode. This device used to declare the same three offsets a
/// second time under two other names, `buffer_ref` and `gpu_address`, which is
/// the drift the wire crate exists to catch: the offsets agreed and the meanings
/// did not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IcbHostResourceInfo {
    /// The ICB being asked about.
    pub icb_ref: u32,
    /// Where the *answer* goes: the scratch buffer the guest's command stream
    /// returned from `-getBufferBytes:alignment:buffer:offset:`.
    pub reply_buffer_ref: u32,
    /// Offset into [`Self::reply_buffer_ref`] for the two `u64`s the guest is
    /// asking the host to write.
    pub reply_offset: u64,
}

/// Decode `0x1d1` payload (16 bytes) or full record (24 bytes including header).
pub fn decode_icb_host_resource_info(bytes: &[u8]) -> Result<IcbHostResourceInfo, IcbStatus> {
    let payload = if bytes.len() >= INFO_OP_ICB_HOST_RESOURCE_RECORD_LEN as usize
        && ld32(&bytes[0..]) == INFO_OP_ICB_HOST_RESOURCE
    {
        &bytes[8..8 + INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN]
    } else if bytes.len() >= INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN {
        &bytes[..INFO_OP_ICB_HOST_RESOURCE_PAYLOAD_LEN]
    } else {
        return Err(IcbStatus::Args("icb_host_resource_info_short"));
    };
    // The three offsets are taken from the wire declaration rather than spelled
    // again, so a layout change there fails this build instead of silently
    // re-slicing the same bytes into different fields.
    use reims_vgpu_wire::ops::info::Query;
    let icb_ref = ld32(&payload[std::mem::offset_of!(Query, object_ref)..]);
    let reply_buffer_ref = ld32(&payload[std::mem::offset_of!(Query, reply_buffer_ref)..]);
    let reply_offset = ld64(&payload[std::mem::offset_of!(Query, reply_offset)..]);
    if icb_ref == 0 {
        return Err(IcbStatus::Args("icb_host_resource_info_ref_zero"));
    }
    Ok(IcbHostResourceInfo {
        icb_ref,
        reply_buffer_ref,
        reply_offset,
    })
}

/// The check [`decode_icb_command_range`] fails when an ICB has no command
/// memory bound.
pub const ICB_FILL_NO_COMMAND_MEMORY: &str = "icb_fill_no_command_memory";

/// Register guest command-memory GVA for an ICB (backing buffer for CPU fills).
///
/// Fills are not stream opcodes; the guest writes this buffer via
/// `PGSerializerIndirectComputeCommand`. Product re-decodes it at execute.
pub fn bind_icb_command_memory(
    state: &Device,
    task_id: u32,
    icb_ref: u32,
    mem: IcbCommandMemory,
) -> Result<(), IcbStatus> {
    if icb_ref == 0 || mem.gva == 0 || mem.byte_len == 0 {
        return Err(IcbStatus::Args("icb_bind_memory_bad_args"));
    }
    state
        .task_objects
        .indirect_command_buffers
        .bind(task_id, icb_ref, mem)
        .then_some(())
        .ok_or(IcbStatus::Missing("icb_bind_memory_not_cached"))
}

/// Refuse info-segment `0x1d1` (`icbHostResourceInfo:info:`) by name.
///
/// This record is a query, not a backing association. It names an ICB, a reply
/// buffer, and a byte offset at which the host must write two `u64` result
/// words. Their semantics are not yet established, so manufacturing an answer
/// or treating the reply destination as command memory would violate the wire
/// contract. `runtime::heap_query` shows the shape a real reply takes once the
/// two result words are known.
pub fn apply_icb_host_resource_info<M: HostMemory + HostOps>(
    _state: &Device,
    _host: &M,
    _task_id: u32,
    _info: &IcbHostResourceInfo,
) -> Result<IcbCommandMemory, IcbStatus> {
    Err(IcbStatus::Unsupported("icb_info_query_unanswered"))
}

/// Read attribute-stride u64 at `attributeStrideOffset + index*8`.
///
/// Returns `(stride, has)` — `has` is true when a stride table slot exists and
/// the stored value is non-zero, or when the slot exists and we treat any
/// stored value as authoritative (including 0 from host encode of
/// `has_attribute_stride` with stride 0 is rare; product uses non-zero for has).
fn read_attribute_stride(layout: &IcbCommandLayout, slot: &[u8], index: u32) -> (u64, bool) {
    let slots = icb_layout_attribute_stride_slot_count(layout);
    if slots == 0 || index >= slots || layout.attribute_stride_offset == 0 {
        return (0, false);
    }
    let off = layout.attribute_stride_offset as usize
        + (index as usize) * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE;
    if off + 8 > slot.len() {
        return (0, false);
    }
    let stride = ld64(&slot[off..]);
    // Non-zero stride means the attributeStride API was used. Zero means unset
    // (plain setKernelBuffer/setVertexBuffer does not touch this table).
    if stride != 0 {
        (stride, true)
    } else {
        (0, false)
    }
}

#[cfg(test)]
fn write_attribute_stride(
    layout: &IcbCommandLayout,
    slot: &mut [u8],
    index: u32,
    stride: u64,
) -> Result<(), IcbStatus> {
    use reims_vgpu_core::endian::st64;
    let slots = icb_layout_attribute_stride_slot_count(layout);
    if slots == 0 || index >= slots || layout.attribute_stride_offset == 0 {
        return Err(IcbStatus::Args("icb_attribute_stride_no_slot"));
    }
    let off = layout.attribute_stride_offset as usize
        + (index as usize) * ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE;
    if off + 8 > slot.len() {
        return Err(IcbStatus::Args("icb_attribute_stride_offset_oob"));
    }
    st64(&mut slot[off..], stride);
    Ok(())
}

fn type1_buffer_gva_size<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<(u64, u64), IcbStatus> {
    objects::resolve_buffer_span(state, host, task_id, buffer_ref).map_err(
        |refusal| match refusal {
            objects::BufferSpanRefusal::Rung(rung) => {
                let slug = crate::observe::ladder_slugs!("icb_type1")(rung);
                match rung {
                    objects::LadderRung::NoListEntry | objects::LadderRung::DescRead { .. } => {
                        IcbStatus::Missing(slug)
                    }
                    objects::LadderRung::WrongType { .. } => IcbStatus::BadDescriptor(slug),
                }
            }
            objects::BufferSpanRefusal::Decode => {
                IcbStatus::BadDescriptor(crate::observe::ladder_slug!("icb_type1", desc_decode))
            }
            objects::BufferSpanRefusal::NoBacking => IcbStatus::Missing("icb_type1_no_backing"),
        },
    )
}

/// Convert absolute bind VA → offset into type-1 allocation (`handle << page_shift`).
///
/// PGSerializer stores `base+offset` in the bind VA field (not a separate offset).
/// `wire_va == 0` means base (offset 0). Fail-closed if VA is below base or past size.
fn offset_from_wire_va<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    buffer_ref: u32,
    wire_va: u64,
) -> Result<u64, IcbStatus> {
    if wire_va == 0 {
        return Ok(0);
    }
    let (base, size) = type1_buffer_gva_size(state, host, task_id, buffer_ref)?;
    if wire_va < base {
        return Err(IcbStatus::Args("icb_wire_va_below_base"));
    }
    let off = wire_va - base;
    if off >= size {
        return Err(IcbStatus::Args("icb_wire_va_past_end"));
    }
    Ok(off)
}

/// Resolve wire VAs on a compute fill into type-1 bind offsets (mutates in place).
pub fn resolve_compute_fill_offsets<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    fill: &mut IcbComputeFill,
) -> Result<(), IcbStatus> {
    for b in &mut fill.buffers {
        if b.wire_va != 0 {
            b.offset = offset_from_wire_va(state, host, task_id, b.buffer_ref, b.wire_va)?;
        }
    }
    Ok(())
}

/// Resolve wire VAs on a render fill into type-1 bind / index offsets (mutates in place).
pub fn resolve_render_fill_offsets<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    fill: &mut IcbRenderFill,
) -> Result<(), IcbStatus> {
    for b in &mut fill.buffers {
        if b.wire_va != 0 {
            b.offset = offset_from_wire_va(state, host, task_id, b.buffer_ref, b.wire_va)?;
        }
    }
    match &mut fill.draw {
        IcbRenderDraw::Indexed {
            index_buffer_ref,
            index_wire_va,
            index_buffer_offset,
            ..
        } => {
            if *index_wire_va != 0 {
                *index_buffer_offset =
                    offset_from_wire_va(state, host, task_id, *index_buffer_ref, *index_wire_va)?;
            }
        }
        IcbRenderDraw::Patches {
            patch_index_buffer_ref,
            patch_index_wire_va,
            patch_index_buffer_offset,
            tessellation_factor,
            ..
        } => {
            if *patch_index_wire_va != 0 && *patch_index_buffer_ref != 0 {
                *patch_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *patch_index_buffer_ref,
                    *patch_index_wire_va,
                )?;
            }
            if tessellation_factor.wire_va != 0 && tessellation_factor.buffer_ref != 0 {
                tessellation_factor.offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    tessellation_factor.buffer_ref,
                    tessellation_factor.wire_va,
                )?;
            }
        }
        IcbRenderDraw::IndexedPatches {
            patch_index_buffer_ref,
            patch_index_wire_va,
            patch_index_buffer_offset,
            control_point_index_buffer_ref,
            control_point_index_wire_va,
            control_point_index_buffer_offset,
            tessellation_factor,
            ..
        } => {
            if *patch_index_wire_va != 0 && *patch_index_buffer_ref != 0 {
                *patch_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *patch_index_buffer_ref,
                    *patch_index_wire_va,
                )?;
            }
            if *control_point_index_wire_va != 0 {
                *control_point_index_buffer_offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    *control_point_index_buffer_ref,
                    *control_point_index_wire_va,
                )?;
            }
            if tessellation_factor.wire_va != 0 && tessellation_factor.buffer_ref != 0 {
                tessellation_factor.offset = offset_from_wire_va(
                    state,
                    host,
                    task_id,
                    tessellation_factor.buffer_ref,
                    tessellation_factor.wire_va,
                )?;
            }
        }
        IcbRenderDraw::Primitives { .. }
        | IcbRenderDraw::MeshThreads(_)
        | IcbRenderDraw::MeshThreadgroups(_) => {}
    }
    Ok(())
}

/// One decoded, offset-resolved ICB command slot, ready for a backend to apply.
///
/// [`decode_icb_command_range`] returns these; what a backend does with one is
/// real `MTLIndirectCommandBuffer` from them, the Vulkan arm replays them as
/// draws. Empty slots are not represented — the decoders skip them.
#[derive(Clone, Debug)]
pub enum IcbCommandFill {
    Compute(IcbComputeFill),
    Render(IcbRenderFill),
}

/// Decode guest command memory into host ICB fills for the given index range.
///
/// Dispatches compute vs render fills from wire `commandTypes` / slot
/// command-type tags, and resolves every wire VA into a type-1 bind offset, so
/// the result names only refs and offsets. Nothing here touches a backend: the
/// same decoder serves both product pathways and gives a future Vulkan executor
/// typed commands to replay.
pub fn decode_icb_command_range<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    icb_ref: u32,
    range_location: u64,
    range_length: u64,
) -> Result<Vec<IcbCommandFill>, IcbStatus> {
    use crate::runtime::decode::resource::{
        MTL_INDIRECT_CMD_CONCURRENT_DISPATCH, MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS,
        MTL_INDIRECT_CMD_DRAW, MTL_INDIRECT_CMD_DRAW_INDEXED,
        MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADS, MTL_INDIRECT_CMD_DRAW_PATCHES,
    };
    use crate::runtime::gva_mem;

    let (layout, max_kernel, max_vertex, max_fragment, inheritance, command_types, max_cmds, mem) = {
        let rec = state
            .task_objects
            .indirect_command_buffers
            .snapshot(task_id, icb_ref)
            .ok_or(IcbStatus::Missing("icb_fill_not_cached"))?;
        (
            rec.descriptor.layout,
            rec.descriptor.max_kernel_buffer_bind_count,
            rec.descriptor.max_vertex_buffer_bind_count,
            rec.descriptor.max_fragment_buffer_bind_count,
            IcbInheritance {
                pipeline_state: rec.descriptor.inherit_pipeline_state(),
                buffers: rec.descriptor.inherit_buffers(),
            },
            rec.descriptor.command_types,
            rec.descriptor.max_command_count as u64,
            rec.command_memory,
        )
    };
    if layout.command_size == 0 {
        return Err(IcbStatus::Args("icb_fill_zero_command_size"));
    }
    let end = range_location
        .checked_add(range_length)
        .ok_or(IcbStatus::Args("icb_fill_range_overflow"))?;
    if end > max_cmds {
        return Err(IcbStatus::Args("icb_fill_range_past_capacity"));
    }
    // An empty execute range is a true no-op. It neither resolves nor reads
    // command backing, but its location still belongs to the declared ICB
    // range and was validated above.
    if range_length == 0 {
        return Ok(Vec::new());
    }
    let mem = mem.ok_or(IcbStatus::Missing(ICB_FILL_NO_COMMAND_MEMORY))?;
    let command_size = u64::from(layout.command_size);
    let byte_start = range_location
        .checked_mul(command_size)
        .ok_or(IcbStatus::Args("icb_fill_range_byte_start_overflow"))?;
    let byte_end = end
        .checked_mul(command_size)
        .ok_or(IcbStatus::Args("icb_fill_range_byte_end_overflow"))?;
    if byte_end > mem.byte_len {
        return Err(IcbStatus::Args("icb_fill_range_past_memory"));
    }
    let byte_len = byte_end - byte_start;
    let host_len = usize::try_from(byte_len)
        .map_err(|_| IcbStatus::Args("icb_fill_range_host_size_overflow"))?;
    let read_gva = mem
        .gva
        .checked_add(byte_start)
        .ok_or(IcbStatus::Args("icb_fill_range_gva_overflow"))?;
    let mut bytes = vec![0u8; host_len];
    gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        read_gva,
        &mut bytes,
        state.page_shift,
    )
    .map_err(|_| IcbStatus::BackendFailed("icb_fill_command_memory_read"))?;

    let mut out = Vec::new();
    // Concurrent-dispatch capability selects the compute command domain. A
    // descriptor may retain render bits and still construct, but its render
    // command setters are not valid; the extra bits do not make the backing
    // heterogeneous.
    let compute_command_domain = command_types
        & (MTL_INDIRECT_CMD_CONCURRENT_DISPATCH | MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS)
        != 0;
    for (local_index, i) in (range_location..end).enumerate() {
        let off = local_index * (layout.command_size as usize);
        let slot = &bytes[off..off + layout.command_size as usize];
        let type_offset = layout.command_type_offset as usize;
        if type_offset + 4 > slot.len() {
            return Err(IcbStatus::Args("icb_fill_command_type_offset_oob"));
        }
        let command_type = ld32(&slot[type_offset..]);
        if command_type == 0 {
            continue;
        }
        if command_types & command_type == 0 {
            return Err(IcbStatus::Args("icb_fill_command_type_not_declared"));
        }
        match command_type {
            MTL_INDIRECT_CMD_CONCURRENT_DISPATCH | MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS => {
                let Some(mut fill) = decode_compute_command_slot_with_inheritance(
                    &layout,
                    slot,
                    max_kernel,
                    inheritance,
                )?
                else {
                    continue;
                };
                fill.command_index = i as u32;
                resolve_compute_fill_offsets(state, host, task_id, &mut fill)?;
                out.push(IcbCommandFill::Compute(fill));
            }
            MTL_INDIRECT_CMD_DRAW
            | MTL_INDIRECT_CMD_DRAW_INDEXED
            | MTL_INDIRECT_CMD_DRAW_PATCHES
            | MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES
            | MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS
            | MTL_INDIRECT_CMD_DRAW_MESH_THREADS => {
                if compute_command_domain {
                    return Err(IcbStatus::Args("icb_fill_render_command_in_compute_domain"));
                }
                let Some(mut fill) = decode_render_command_slot_with_inheritance(
                    &layout,
                    slot,
                    max_vertex,
                    max_fragment,
                    inheritance,
                )?
                else {
                    continue;
                };
                fill.command_index = i as u32;
                resolve_render_fill_offsets(state, host, task_id, &mut fill)?;
                out.push(IcbCommandFill::Render(fill));
            }
            _ => return Err(IcbStatus::Args("icb_fill_unknown_command_type")),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
