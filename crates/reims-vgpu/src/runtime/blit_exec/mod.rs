//! Product-path execution of blit fill/copy commands against guest backings.
//!
//! Supported now:
//! - `fillBuffer` (0x132) on type-1 buffers
//! - `copyFromBuffer:toBuffer:` (0x12d) on type-1 buffers
//! - Rectangular buffer↔texture / texture↔texture copies on linear type-2/3
//! - Same rectangular copies with **IOSurface** texture endpoints
//!   (level 0, slice 0, depth 1) via mapping page tables; multi-plane (biplanar)
//!   sample windows from cached `sIOSurfaceDeviceDescriptor` selected by texture
//!   geometry (width/height/bpe), not a wire plane index
//! - **Type-8 texture views** as copy endpoints: unswizzled views over type-2/3
//!   or IOSurface texture bases; multi-level / array / non-2D Metal types when geometry matches
//!   (IOSurface texture bases remain single-level / single-slice — see below)
//! - **`MTLBlitOption`**: None; DepthFromDepthStencil / StencilFromDepthStencil;
//!   combined DS plane packing on linear GVA; unknown bits / RowLinearPVRTC fail
//! - **`0x13e` whole-surface** texture→texture: for each level in
//!   `[sourceLevel, sourceLevel+levelCount)`:
//!   - **depth-1 (array/2D):** full `width×height` across `sliceCount` consecutive
//!     slices
//!   - **depth>1 (3D volume):** Metal requires `sliceCount==1` and slices 0;
//!     copies full `width×height×depth` of that mip (depth planes via
//!     `bytes_per_image`); linear type-2/3 only
//!   - zero `sliceCount`/`levelCount` are Metal no-ops
//! - **Fences** `0x13c` update / `0x13d` wait: operations on the shared fence
//!   object via [`reims_vgpu_core::synchronization`]; waits that are not yet satisfied are
//!   soft-pending (do not block drain), matching the unified-memory in-order path
//!
//! Not executed (fail visibly / soft miss):
//! - swizzled type-8 views (contract: blit rejects remapped swizzle materialization)
//! - multisample view types
//! - RowLinearPVRTC / unknown option bits
//! - overlapping same-buffer B2B windows
//! - IOSurface texture multi-mip / non-zero level or slice — **not a missing feature**: Metal
//!   forbids mipmapped IOSurface textures (`newTextureWithDescriptor:iosurface:`
//!   rejects `mipmapLevelCount > 1`). Product path fail-closes; do not invent a
//!   pyramid layout in the mapping.
//! - 3D whole-surface with `sliceCount!=1`, non-zero slices, or IOSurface texture endpoint

use crate::observe::Decline;
use crate::runtime::decode::blit::{self, BlitAspect, Command, CopyKind, Kind, Point};
use crate::runtime::decode::resource::{
    texture_view_type_is_3d, texture_view_type_uses_slices, Descriptor as ResourceDescriptor,
    ObjectKind,
};
use crate::runtime::draw::{self, host_alloc_len};
use crate::runtime::fence_exec::{self, FenceStatus};
use crate::runtime::gva_mem;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::mapper;
use crate::runtime::mapper::RectStride;
use crate::runtime::mapping_write;
use crate::runtime::objects;
use crate::runtime::Device;
use reims_vgpu_core::pixel_format::{self, MTL_FORMAT_BGRA8_UNORM};
use reims_vgpu_core::{
    BlitCompletion, BufferFillPattern, CommandExecution, ContentStamp, ExecutionOutput,
    ResolvedBlit, ResolvedBufferRange, ResolvedBufferToTextureBlit, ResolvedCommand,
    ResolvedLinearTextureLevel as LinearTextureLevel, ResolvedSubmission,
    ResolvedSurfaceTextureBacking as IOSurfaceTextureBacking,
    ResolvedTextureBacking as TextureBacking, ResolvedTextureCopyBatch, ResolvedTextureEndpoint,
    ResolvedTextureLevelCopy, ResolvedTextureToBufferBlit, ResolvedTextureToTextureBlit,
    TextureExtent, TextureOrigin,
};
use reims_vgpu_core::{FenceAction, SynchronizationDomain as FenceDomain};
use reims_vgpu_protocol::{ByteLength, GuestVirtualAddress, MappingId};
use reims_vgpu_wire::ops::blit as wire_blit;

/// Chunk size for fill/copy host staging (bounded guest IO).
const CHUNK: usize = 64 * 1024;

/// Outcome of a product-path blit fill/copy/fence attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitStatus {
    Ok,
    /// Missing object, wrong kind, or unreadable descriptor.
    MissingResource,
    /// Opcode / options / view / slice / 3D / IOSurface texture not on this path.
    Unsupported,
    /// Offset/length/extent outside allocation or level bounds.
    Bounds,
    /// Guest GVA read/write failed.
    GuestIo,
    /// Pathological size or host staging cap.
    Capacity,
    /// Zero-size fill or zero-extent rectangular copy (Metal no-op → soft ok).
    ZeroExtent,
    /// Same buffer, overlapping source/destination windows.
    Overlap,
    /// Fence wait not yet satisfied (soft; does not block drain).
    FencePending,
}

impl crate::observe::Refusal for BlitStatus {
    /// The reason comes from the thread-local channel, not from the variant.
    ///
    /// This rail is the crate's largest refusal surface — **177 distinct checks
    /// across 182 sites**, collapsing into eight coarse statuses — so the specific
    /// cause has always travelled beside the value in [`BLIT_FAIL_REASON`] rather
    /// than inside it. That is a legitimate shape (a 177-arm `slug()` is not a
    /// thing anyone writes) and the registry reads the vocabulary at the `br(`
    /// sites, so every one of the 177 is counted and unique crate-wide.
    ///
    /// What was *not* legitimate: an uninstrumented site returned a coarse status
    /// with the channel still empty, and the dispatch line rendered a bare
    /// `reason=` with nothing after it — unfindable by grep and indistinguishable
    /// from a missing field. That case is now the registered `blit_unattributed`,
    /// which names the gap instead of hiding it.
    ///
    /// Read on the same thread that ran the blit, which both dispatch sites do
    /// immediately after the call. `Ok`, `ZeroExtent` and `FencePending` are
    /// control flow — the first two are the dispatch site's success arm, the third
    /// is a soft wait the guest re-polls — and this reproduces exactly the two
    /// sites' previous log conditions.
    fn refusal(&self) -> Option<&'static str> {
        match self {
            Self::Ok | Self::ZeroExtent | Self::FencePending => None,
            _ => Some(match blit_fail_reason() {
                "" => "blit_unattributed",
                slug => slug,
            }),
        }
    }
}

thread_local! {
    /// The specific reason slug for the most recent non-`Ok` [`BlitStatus`], set at
    /// the failing site so the single dispatch-site failure line can name *which* of
    /// the many checks that collapse into a coarse status actually fired. Cleared at
    /// the start of every `execute_blit`/`execute_blit_fence` so an uninstrumented
    /// site reports empty rather than a stale value from a prior command. Genuine
    /// failures only reach the dispatch log, so this never floods a healthy boot.
    static BLIT_FAIL_REASON: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Record `reason` for a non-`Ok` [`BlitStatus`] at the failing site and return
/// that status unchanged. Use at every `return Err(..)` / `.ok_or_else(..)` site that
/// collapses a distinct cause into a coarse status.
#[inline]
fn br(status: BlitStatus, reason: &'static str) -> BlitStatus {
    BLIT_FAIL_REASON.with(|r| r.set(reason));
    status
}

/// Read the last recorded blit-failure reason without clearing it, so several call
/// sites (a path-specific line plus the dispatch summary) can name the same cause.
/// The channel is reset at the start of the next command via [`clear_blit_fail_reason`],
/// so a stale reason cannot leak across commands. Read this only on the failure path.
pub fn blit_fail_reason() -> &'static str {
    BLIT_FAIL_REASON.with(|r| r.get())
}

/// Reset the reason channel at entry to a blit command so an uninstrumented failure
/// reports empty rather than a stale reason from a prior command.
#[inline]
fn clear_blit_fail_reason() {
    BLIT_FAIL_REASON.with(|r| r.set(""));
}

/// Dedup set for the `tex_wrong_type` enrichment line, keyed by
/// `(task_id, texture_ref, object_type)`. A blit that binds a non-texture ref
/// fails once per draw (observed ~67/six-app-launch), so the enrichment must
/// dedup — but the bare `reason=tex_wrong_type` dispatch slug hides *what* the
/// object actually is (buffer bound as texture = a decode/tracking bug vs. a
/// legit guest race), which is the load-bearing field for diagnosis.
static TEX_WRONG_TYPE_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(u32, u32, ObjectKind)>>,
> = std::sync::OnceLock::new();

/// Emit ONE always-on `blit tex_wrong_type` line per distinct
/// `(task, ref, object_type)` naming the actual object type a blit tried to use
/// as a texture. Deduped so a per-draw repeat cannot flood. Returns whether it
/// emitted (tests use it). Diagnostic only.
fn note_tex_wrong_type(
    task_id: u32,
    texture_ref: u32,
    object_type: ObjectKind,
    level: u16,
    slice: u16,
) -> bool {
    let set = TEX_WRONG_TYPE_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert((task_id, texture_ref, object_type)) {
            return false;
        }
    }
    crate::observe::fail(format!(
        "blit tex_wrong_type task={task_id} ref={texture_ref} object_type={object_type} level={level} slice={slice}"
    ));
    true
}

#[cfg(test)]
fn reset_tex_wrong_type_dedup_for_test() {
    if let Some(set) = TEX_WRONG_TYPE_SEEN.get() {
        if let Ok(mut g) = set.lock() {
            g.clear();
        }
    }
}

/// Dedup set for the `t5_view_decode` diagnostic, keyed by surface id.
static T5_DECODE_FAIL_SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<u32>>> =
    std::sync::OnceLock::new();

/// One always-on diagnostic per surface id when a IOSurface plane view RefTexture's view
/// record fails to decode: dumps `desc_len` + head hex so the exact blit-path
/// IOSurface plane view layout can be read offline (the decoder wants tag 0x42 at +0x14, 2D
/// nonzero geom, depth==1). Deduped so a per-draw repeat cannot flood.
fn note_t5_decode_fail(sid: u32, bytes: &[u8]) {
    let set = T5_DECODE_FAIL_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert(sid) {
            return;
        }
    }
    let n = bytes.len().min(40);
    let mut hex = String::with_capacity(n * 2);
    for b in &bytes[..n] {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    crate::observe::fail(format!(
        "blit t5_view_decode sid={sid} desc_len={} head_hex={hex}",
        bytes.len()
    ));
}

/// `(task, side, format)` — what `repack_storage_assumed` reports once per.
type RepackAssumedKey = (u32, &'static str, u16);

/// Dedup set for `blit repack_storage_assumed`.
static REPACK_STORAGE_ASSUMED_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<RepackAssumedKey>>,
> = std::sync::OnceLock::new();

/// Name an aspect repack whose storage stride could not be derived.
///
/// `bytes_per_pixel` refused this texture's format — in practice a zero, which
/// the texture-to-texture format check deliberately admits — so the repack runs
/// with the copied aspect's own width instead. That is right whenever the two
/// widths agree and reads the wrong bytes when they do not, and nothing
/// downstream can tell which happened. This is a **healthy zero**: a firing says
/// a real guest workload reached a combined depth/stencil repack against a
/// texture whose format this device never learned, which is the evidence that
/// would justify deriving the storage width from the backing instead of
/// assuming it. Deduped per task, side and format so a per-blit repeat cannot
/// flood.
fn note_repack_storage_assumed(
    task_id: u32,
    side: &'static str,
    format: u16,
    assumed_bpp: u32,
) -> bool {
    let set = REPACK_STORAGE_ASSUMED_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    if let Ok(mut g) = set.lock() {
        if !g.insert((task_id, side, format)) {
            return false;
        }
    }
    crate::observe::fail(format!(
        "blit repack_storage_assumed task={task_id} side={side} format={format} \
         assumed_bpp={assumed_bpp} (storage width underivable; aspect width used)"
    ));
    true
}

/// Dedup set for the `copy_region_*_io` enrichment, keyed by
/// `(task, gva_page, is_write)`.
static COPY_REGION_IO_SEEN: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<(u32, u64, bool)>>,
> = std::sync::OnceLock::new();

/// Emit ONE always-on `blit copy_region_io` line per distinct
/// `(task, failing-gva-page, is_write)` naming the exact guest address a
/// rectangular copy row could not read/write. A guest that tears down the
/// destination surface mid-copy (teardown race) shows a plausible-but-unmapped
/// gva; a decode/geometry bug shows a wild gva. Deduped per page so a strided
/// multi-row failure cannot flood. Diagnostic only.
fn note_copy_region_io(
    task_id: u32,
    is_write: bool,
    gva: u64,
    row: u64,
    image: u64,
    row_bytes: u64,
    page_shift: u32,
) -> bool {
    let set = COPY_REGION_IO_SEEN.get_or_init(|| std::sync::Mutex::new(Default::default()));
    let key = (task_id, gva >> page_shift, is_write);
    if let Ok(mut g) = set.lock() {
        if !g.insert(key) {
            return false;
        }
    }
    let dir = if is_write { "write" } else { "read" };
    crate::observe::fail(format!(
        "blit copy_region_io dir={dir} task={task_id} gva={gva:#x} row={row} image={image} row_bytes={row_bytes}"
    ));
    true
}

struct LinearBuffer {
    content: ContentStamp,
    gva: u64,
    size: u64,
}

impl LinearBuffer {
    fn range(&self, offset: u64, length: u64) -> Option<ResolvedBufferRange> {
        if !range_fits(offset, length, self.size) {
            return None;
        }
        Some(ResolvedBufferRange {
            content: self.content,
            address: GuestVirtualAddress::new(self.gva.checked_add(offset)?),
            length: ByteLength::new(length),
        })
    }
}

fn resolve_buffer<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
) -> Result<LinearBuffer, BlitStatus> {
    if buffer_ref == 0 {
        return Err(br(BlitStatus::MissingResource, "buf_ref_zero"));
    }
    let resource = objects::resolve_resource(state, host, task_id, buffer_ref).map_err(|rung| {
        br(
            BlitStatus::MissingResource,
            crate::observe::ladder_slugs!("buf")(rung),
        )
    })?;
    if resource.entry().kind != ObjectKind::Buffer {
        return Err(br(
            BlitStatus::MissingResource,
            crate::observe::ladder_slug!("buf", wrong_type),
        ));
    }
    let Ok(ResourceDescriptor::Buffer(buf)) = objects::decoded_resource(&resource) else {
        return Err(br(
            BlitStatus::MissingResource,
            crate::observe::ladder_slug!("buf", desc_decode),
        ));
    };
    let Some((gva, size)) = buf.backing_gva_size(state.page_shift) else {
        return Err(br(BlitStatus::MissingResource, "buf_no_backing"));
    };
    let Some((resource, version)) = state
        .task_objects
        .resources
        .content_stamp(task_id, buffer_ref)
    else {
        return Err(br(BlitStatus::MissingResource, "buf_no_semantic_identity"));
    };
    Ok(LinearBuffer {
        content: ContentStamp { resource, version },
        gva,
        size,
    })
}

fn resolve_texture_backing<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
) -> Result<TextureBacking, BlitStatus> {
    resolve_texture_backing_depth(
        state,
        host,
        task_id,
        TextureResolveRequest {
            texture_ref,
            level,
            slice,
            view_depth: 0,
            settle_guest_bytes: true,
        },
    )
}

/// Resolve immutable texture storage without making its guest bytes current.
///
/// Planning uses this only when execution owns the decision between a resident
/// GPU copy and a guest-byte fallback. The latter must explicitly settle the
/// endpoint before reading or partially overwriting it.
fn resolve_texture_backing_unsettled<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
) -> Result<TextureBacking, BlitStatus> {
    resolve_texture_backing_depth(
        state,
        host,
        task_id,
        TextureResolveRequest {
            texture_ref,
            level,
            slice,
            view_depth: 0,
            settle_guest_bytes: false,
        },
    )
}

fn resolve_texture_endpoint<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
) -> Result<ResolvedTextureEndpoint, BlitStatus> {
    let backing = resolve_texture_backing(state, host, task_id, texture_ref, level, slice)?;
    let Some((resource, version)) = state
        .task_objects
        .resources
        .content_stamp(task_id, texture_ref)
    else {
        return Err(br(BlitStatus::MissingResource, "tex_no_semantic_identity"));
    };
    Ok(ResolvedTextureEndpoint {
        content: ContentStamp { resource, version },
        backing,
    })
}

fn resolve_texture_endpoint_unsettled<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u16,
    slice: u16,
) -> Result<ResolvedTextureEndpoint, BlitStatus> {
    let backing =
        resolve_texture_backing_unsettled(state, host, task_id, texture_ref, level, slice)?;
    let Some((resource, version)) = state
        .task_objects
        .resources
        .content_stamp(task_id, texture_ref)
    else {
        return Err(br(BlitStatus::MissingResource, "tex_no_semantic_identity"));
    };
    Ok(ResolvedTextureEndpoint {
        content: ContentStamp { resource, version },
        backing,
    })
}

fn resolve_texture_copy_batch<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> Result<ResolvedBlit, BlitStatus> {
    if cmd.slice_count == 0 || cmd.level_count == 0 {
        return Err(BlitStatus::ZeroExtent);
    }
    if cmd.source == 0 || cmd.destination == 0 {
        return Err(br(BlitStatus::MissingResource, "sl_missing_ref"));
    }
    let mut levels = Vec::with_capacity(usize::from(cmd.level_count));
    for level_delta in 0..cmd.level_count {
        let sl_resolve_started = std::time::Instant::now();
        let source_level = cmd
            .source_level
            .checked_add(level_delta)
            .ok_or_else(|| br(BlitStatus::Bounds, "sl_src_level_overflow"))?;
        let destination_level = cmd
            .destination_level
            .checked_add(level_delta)
            .ok_or_else(|| br(BlitStatus::Bounds, "sl_dst_level_overflow"))?;
        let mut slices = Vec::with_capacity(usize::from(cmd.slice_count));
        for slice_delta in 0..cmd.slice_count {
            let source_slice = cmd
                .source_slice
                .checked_add(slice_delta)
                .ok_or_else(|| br(BlitStatus::Bounds, "sl_src_slice_overflow"))?;
            let destination_slice = cmd
                .destination_slice
                .checked_add(slice_delta)
                .ok_or_else(|| br(BlitStatus::Bounds, "sl_dst_slice_overflow"))?;
            let source = resolve_texture_endpoint_unsettled(
                state,
                host,
                task_id,
                cmd.source,
                source_level,
                source_slice,
            )?;
            let destination = resolve_texture_endpoint_unsettled(
                state,
                host,
                task_id,
                cmd.destination,
                destination_level,
                destination_slice,
            )?;
            if slice_delta == 0
                && (source.backing.depth() > 1 || destination.backing.depth() > 1)
                && (cmd.slice_count != 1 || cmd.source_slice != 0 || cmd.destination_slice != 0)
            {
                return Err(br(BlitStatus::Unsupported, "sl_volume_slice_constraint"));
            }
            slices.push((source, destination));
        }
        let first_slice = slices.remove(0);
        levels.push(ResolvedTextureLevelCopy {
            first_slice,
            remaining_slices: slices.into_boxed_slice(),
        });
        crate::runtime::drain::note_store_route_us(
            "sl_resolve_us",
            sl_resolve_started.elapsed().as_micros() as u64,
        );
    }
    let first_level = levels.remove(0);
    Ok(ResolvedBlit::TextureCopyBatch(ResolvedTextureCopyBatch {
        source_base_slice: cmd.source_slice,
        destination_base_slice: cmd.destination_slice,
        first_level,
        remaining_levels: levels.into_boxed_slice(),
    }))
}

fn resolve_buffer_to_texture_blit<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> Result<ResolvedBlit, BlitStatus> {
    let source = resolve_buffer(state, host, task_id, cmd.source)?;
    let remaining = source
        .size
        .checked_sub(cmd.source_offset)
        .ok_or_else(|| br(BlitStatus::Bounds, "b2t_src_offset_oob"))?;
    let source = source
        .range(cmd.source_offset, remaining)
        .ok_or_else(|| br(BlitStatus::Bounds, "b2t_src_range_oob"))?;
    let destination = resolve_texture_endpoint(
        state,
        host,
        task_id,
        cmd.destination,
        cmd.destination_level,
        cmd.destination_slice,
    )?;
    let (aspect, _) = copy_aspect_for_options(destination.backing.pixel_format(), cmd)?;
    Ok(ResolvedBlit::BufferToTexture(ResolvedBufferToTextureBlit {
        source,
        source_bytes_per_row: cmd.source_bytes_per_row,
        source_bytes_per_image: cmd.source_bytes_per_image,
        destination,
        destination_origin: TextureOrigin {
            x: cmd.destination_origin.x,
            y: cmd.destination_origin.y,
            z: cmd.destination_origin.z,
        },
        extent: TextureExtent {
            width: cmd.source_size.width,
            height: cmd.source_size.height,
            depth: cmd.source_size.depth,
        },
        aspect,
    }))
}

fn resolve_texture_to_buffer_blit<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> Result<ResolvedBlit, BlitStatus> {
    let source = resolve_texture_endpoint(
        state,
        host,
        task_id,
        cmd.source,
        cmd.source_level,
        cmd.source_slice,
    )?;
    let (aspect, _) = copy_aspect_for_options(source.backing.pixel_format(), cmd)?;
    let destination = resolve_buffer(state, host, task_id, cmd.destination)?;
    let remaining = destination
        .size
        .checked_sub(cmd.destination_offset)
        .ok_or_else(|| br(BlitStatus::Bounds, "t2b_dst_offset_oob"))?;
    let destination = destination
        .range(cmd.destination_offset, remaining)
        .ok_or_else(|| br(BlitStatus::Bounds, "t2b_dst_range_oob"))?;
    Ok(ResolvedBlit::TextureToBuffer(ResolvedTextureToBufferBlit {
        source,
        source_origin: TextureOrigin {
            x: cmd.source_origin.x,
            y: cmd.source_origin.y,
            z: cmd.source_origin.z,
        },
        extent: TextureExtent {
            width: cmd.source_size.width,
            height: cmd.source_size.height,
            depth: cmd.source_size.depth,
        },
        destination,
        destination_bytes_per_row: cmd.destination_bytes_per_row,
        destination_bytes_per_image: cmd.destination_bytes_per_image,
        aspect,
    }))
}

fn resolve_texture_to_texture_blit<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> Result<ResolvedBlit, BlitStatus> {
    let source = resolve_texture_endpoint(
        state,
        host,
        task_id,
        cmd.source,
        cmd.source_level,
        cmd.source_slice,
    )?;
    let destination = resolve_texture_endpoint(
        state,
        host,
        task_id,
        cmd.destination,
        cmd.destination_level,
        cmd.destination_slice,
    )?;
    let (aspect, source_bpp) = copy_aspect_for_options(source.backing.pixel_format(), cmd)?;
    let (_, destination_bpp) = copy_aspect_for_options(destination.backing.pixel_format(), cmd)?;
    if source_bpp != destination_bpp {
        return Err(br(BlitStatus::Unsupported, "t2t_bpp_mismatch"));
    }
    Ok(ResolvedBlit::TextureToTexture(
        ResolvedTextureToTextureBlit {
            source,
            source_origin: TextureOrigin {
                x: cmd.source_origin.x,
                y: cmd.source_origin.y,
                z: cmd.source_origin.z,
            },
            destination,
            destination_origin: TextureOrigin {
                x: cmd.destination_origin.x,
                y: cmd.destination_origin.y,
                z: cmd.destination_origin.z,
            },
            extent: TextureExtent {
                width: cmd.source_size.width,
                height: cmd.source_size.height,
                depth: cmd.source_size.depth,
            },
            aspect,
        },
    ))
}

#[derive(Clone, Copy)]
struct TextureResolveRequest {
    texture_ref: u32,
    level: u16,
    slice: u16,
    view_depth: u32,
    settle_guest_bytes: bool,
}

fn resolve_texture_backing_depth<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    request: TextureResolveRequest,
) -> Result<TextureBacking, BlitStatus> {
    let TextureResolveRequest {
        texture_ref,
        level,
        slice,
        // How many texture-view hops deep this recursion is — **not** a texture's
        // depth. Keeping it in this typed request prevents it from being confused
        // with the level's declared depth.
        view_depth,
        settle_guest_bytes,
    } = request;
    if texture_ref == 0 {
        return Err(br(BlitStatus::MissingResource, "tex_ref_zero"));
    }
    // Shared with the draw/sample walk on purpose: the two arms read one wire
    // form, and a chain that resolver would follow must not be a copy this one
    // drops. See `MAX_TEXTURE_VIEW_CHAIN` for why the contract's own depth is
    // the right number for both.
    //
    // `>` and not `>=` because this recursion counts *calls* and the other walk
    // counts *hops*: the last call here is the non-view base, one level below
    // the deepest view. A chain of MAX views arrives at its base at `depth ==
    // MAX`, so admitting that depth is what makes the two arms accept and
    // refuse the same chains.
    if view_depth as usize > crate::runtime::draw::MAX_TEXTURE_VIEW_CHAIN {
        return Err(br(BlitStatus::Unsupported, "tex_view_depth_cap"));
    }
    // **This function is the whole blit rail's definition of "the guest bytes of
    // a texture", and guest bytes are only a resource's content once everything
    // this device rendered into it has landed.** Every endpoint of every blit —
    // source and destination, texture-to-texture, texture-to-buffer and back —
    // arrives here, so this is the one place the rail has to state that.
    //
    // A render pass whose colour attachment is an ordinary private `MTLTexture`
    // resolves on `render_target`'s fourth rung to a linear guest VA, and
    // `writeback_debt::arm_gva` then keeps the result in the engine's resident
    // and arms a debt against `(task_id, texture_ref)` instead of copying it
    // out. Nothing lands until a **guest-byte reader** names the resource. The
    // two readers that did name it are `draw::texture_view` and `compute_exec`;
    // a blit is exactly as much a guest-byte reader as either, and it was
    // silent. Its copy read whatever the pages held before the pass — which for
    // a freshly allocated private target is zeros, at scale, with no error, the
    // "Never Fail Silently" class in its worst form.
    //
    // `pay_for_texture` is the call that covers both spellings a debt can have
    // (the task-local GVA resource and the surface mapping), and it early-returns
    // on an empty ledger, so the cost on a rail with nothing owed is one
    // `is_empty`. Paying here rather than at the copy also covers the view chain:
    // the recursion below re-enters with the base ref, and the debt may be armed
    // against either spelling of the pair.
    //
    // Paying a *destination*'s debt before overwriting it is deliberate, not
    // waste. The blit writes a rect, not the plane, so the pixels outside that
    // rect are the resource's content and must be real; and leaving the debt
    // armed would let it land the pre-blit resident over the blit's own bytes
    // later.
    if settle_guest_bytes {
        note_blit_endpoint_debt(state, task_id, texture_ref);
        crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    }
    // Resolve through the retained resource aggregate, not directly through
    // mutable object-list bytes.  A successful endpoint therefore has one
    // generational identity and one descriptor snapshot for its whole guest
    // lifetime, just like buffers and every other canonical resource family.
    let resource =
        objects::resolve_resource(state, host, task_id, texture_ref).map_err(|rung| {
            br(
                BlitStatus::MissingResource,
                crate::observe::ladder_slugs!("tex")(rung),
            )
        })?;
    let entry = resource.entry();
    let bytes = resource.descriptor().as_ref();

    // Type-8 view → base texture (unswizzled; multi-level / array / non-2D allowed).
    if entry.kind == ObjectKind::TextureView {
        let view = draw::resolve_texture_view_reasoned(state, host, task_id, texture_ref).map_err(
            |reason| {
                crate::observe::Emit::decline("blit_tex_view_resolve", &reason)
                    .field("task", task_id)
                    .field("ref", texture_ref)
                    .fail_once(u64::from(task_id) << 32 | u64::from(texture_ref));
                br(BlitStatus::Unsupported, "view_resolve")
            },
        )?;
        // Blit rejects swizzled materialization (contract).
        if view
            .swizzle
            .as_ref()
            .is_some_and(|plan| !pixel_format::swizzle_is_identity(plan))
        {
            return Err(br(BlitStatus::Unsupported, "view_swizzle_nonident"));
        }
        // The command's level/slice are relative to the outermost view. The
        // shared resolver has already composed every nested range into the
        // final base namespace.
        let (abs_level, abs_slice) = view
            .select(u64::from(level), u64::from(slice))
            .ok_or_else(|| br(BlitStatus::Bounds, "view_subresource_oob"))?;
        if abs_level > u16::MAX as u64 {
            return Err(br(BlitStatus::Bounds, "view_level_u16"));
        }
        if abs_slice > u16::MAX as u64 {
            return Err(br(BlitStatus::Bounds, "view_slice_u16"));
        }
        // 3D views use depth planes, not array slices.
        if let Some(view_type) = view.texture_type {
            if texture_view_type_is_3d(view_type) && abs_slice != 0 {
                return Err(br(BlitStatus::Unsupported, "view_3d_slice"));
            }
            // Non-array 2D/1D: only slice 0.
            if !texture_view_type_uses_slices(view_type)
                && !texture_view_type_is_3d(view_type)
                && abs_slice != 0
            {
                return Err(br(BlitStatus::Unsupported, "view_nonarray_slice"));
            }
        }
        let mut backing = resolve_texture_backing_depth(
            state,
            host,
            task_id,
            TextureResolveRequest {
                texture_ref: view.base_texture_ref,
                level: abs_level as u16,
                slice: abs_slice as u16,
                view_depth: view_depth + 1,
                settle_guest_bytes,
            },
        )?;
        // Geometry constraints for non-2D types.
        match (&backing, view.texture_type) {
            (TextureBacking::Linear(t), Some(view_type)) => {
                if matches!(
                    view_type,
                    crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D
                        | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
                ) && t.height != 1
                {
                    return Err(br(BlitStatus::Unsupported, "view_1d_height"));
                }
            }
            (TextureBacking::Surface(_), Some(view_type)) => {
                // Metal forbids mipmapped / multi-slice IOSurface textures; see
                // IOSurfaceTextureBacking. Fail closed rather than inventing layout.
                if abs_level != 0 || abs_slice != 0 {
                    return Err(br(BlitStatus::Unsupported, "view_iosurface_level_slice"));
                }
                if texture_view_type_uses_slices(view_type) || texture_view_type_is_3d(view_type) {
                    return Err(br(BlitStatus::Unsupported, "view_iosurface_type"));
                }
            }
            _ => {}
        }
        // View pixel_format overrides base when bpp-compatible.
        if let Some(declared) = view.pixel_format {
            let base_fmt = backing.pixel_format();
            let eff = draw::effective_view_sample_format(base_fmt, Some(declared))
                .ok_or_else(|| br(BlitStatus::Unsupported, "view_fmt_incompat"))?;
            match &mut backing {
                TextureBacking::Linear(t) => {
                    t.pixel_format = eff;
                    t.bpp = pixel_format::bytes_per_pixel(eff)
                        .ok_or_else(|| br(BlitStatus::Unsupported, "view_fmt_bpp"))?;
                }
                TextureBacking::Surface(t) => {
                    t.pixel_format = eff;
                    t.bpp = pixel_format::bytes_per_pixel(eff)
                        .ok_or_else(|| br(BlitStatus::Unsupported, "view_fmt_bpp"))?;
                }
            }
        }
        return Ok(backing);
    }

    // IOSurface: single level, 2D, mapping page table.
    // Non-zero level/slice is fail-closed (Metal disallows mipmapped IOSurfaces).
    // Texture object dims/format select the plane when the mapping is multi-plane.
    if entry.kind == ObjectKind::IOSurfaceTexture {
        if level != 0 || slice != 0 {
            return Err(br(BlitStatus::Unsupported, "iosurface_level_slice"));
        }
        let Ok(ResourceDescriptor::MapperIOSurfaceTextureView(view)) =
            objects::decoded_resource(&resource)
        else {
            return Err(br(
                BlitStatus::MissingResource,
                crate::observe::ladder_slug!("iosurface", desc_decode),
            ));
        };
        let tex_w = view.declaration.width;
        let tex_h = view.declaration.height;
        let tex_fmt = view.declaration.pixel_format;
        if tex_w == 0 || tex_h == 0 {
            return Err(br(BlitStatus::MissingResource, "iosurface_zero_geom"));
        }
        // Latch texture→mapping and refresh pages / device desc.
        let Some(mapping_id) =
            objects::resolve_iosurface_texture_ref(state, host, task_id, texture_ref)
        else {
            return Err(br(
                BlitStatus::MissingResource,
                "iosurface_mapper_surface_unresolved",
            ));
        };
        let _ = mapper::ensure_resolved_for_scanout(state, host, mapping_id);
        let Some(m) = state.surfaces.mappings.get(&mapping_id) else {
            return Err(br(BlitStatus::MissingResource, "iosurface_no_mapping"));
        };
        if !m.lifecycle.active || m.pages.entries.is_empty() {
            return Err(br(BlitStatus::MissingResource, "iosurface_unmapped"));
        }
        let format = if tex_fmt != 0 {
            tex_fmt
        } else if m.format_or_zero() != 0 {
            m.format_or_zero()
        } else {
            MTL_FORMAT_BGRA8_UNORM
        };
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            return Err(br(BlitStatus::Unsupported, "iosurface_fmt_bpp"));
        };
        let Some((surface_offset, surface_bpr, span_end)) =
            mapping_write::iosurface_texture_sample_window(m, tex_w, tex_h, format)
        else {
            return Err(br(BlitStatus::Bounds, "iosurface_sample_window"));
        };
        note_blit_iosurface_resident(state, mapping_id);
        return Ok(TextureBacking::Surface(IOSurfaceTextureBacking {
            mapping_id: MappingId::new(mapping_id),
            width: tex_w,
            height: tex_h,
            surface_offset,
            row_stride: surface_bpr,
            span_end,
            bpp,
            pixel_format: format,
        }));
    }

    // IOSurface plane view RefTexture: a serialized Metal texture VIEW over an IOSurface
    // (surfaceID at +0). The compute stage path already resolves these; the
    // blit path previously dropped every one as `tex_wrong_type` (~99/six-app
    // launch, all object_type=5), so a blit COPY from a video/biplanar plane
    // or a row-byte-equivalent reinterpretation view (e.g. RGBA32Uint over
    // BGRA8) never landed.
    //
    // Resolve it with the view's own geometry, format **and plane index**. The
    // plane is on the wire here (record `+0x20`) and must be used: it is the
    // whole difference between this and IOSurface texture, whose window resolves the plane
    // by matching geometry and bytes-per-element and so cannot tell two planes
    // that share both apart. A biplanar COPY names exactly such a pair, so this
    // is the path where dropping the index lands. `iosurface_plane_view_sample_window` states
    // the case; a plane it cannot resolve declines here rather than binding
    // whichever plane shares the geometry.
    if entry.kind == ObjectKind::IOSurfacePlaneView {
        if level != 0 || slice != 0 {
            return Err(br(BlitStatus::Unsupported, "t5_level_slice"));
        }
        let Ok(ResourceDescriptor::IOSurfacePlaneView(t5)) = objects::decoded_resource(&resource)
        else {
            return Err(br(BlitStatus::MissingResource, "t5_desc_short"));
        };
        let sid = t5.surface.get();
        if sid == 0 {
            return Err(br(BlitStatus::MissingResource, "t5_no_sid"));
        }
        let Some(view) = t5.view else {
            // A short/zero-geom record fails closed — no fallback to base geom.
            // Capture why (len/tag/geom) deduped per sid so the exact blit-path
            // IOSurface plane view layout can be decoded without flooding.
            note_t5_decode_fail(sid, bytes);
            return Err(br(BlitStatus::Unsupported, "t5_view_decode"));
        };
        // Surface id IS the surface backing mapping mid (never the task object-list ref —
        // those id spaces collide). Resolve the backing, then the mapping.
        let _ = objects::ensure_surface_for_present(state, host, sid);
        let _ = mapper::ensure_resolved_for_scanout(state, host, sid);
        let Some(m) = state.surfaces.mappings.get(&sid) else {
            return Err(br(BlitStatus::MissingResource, "t5_no_mapping"));
        };
        if !m.lifecycle.active || m.pages.entries.is_empty() {
            return Err(br(BlitStatus::MissingResource, "t5_unmapped"));
        }
        let format = view.pixel_format;
        let Some(bpp) = pixel_format::bytes_per_pixel(format) else {
            return Err(br(BlitStatus::Unsupported, "t5_fmt_bpp"));
        };
        let Some((surface_offset, surface_bpr, span_end)) =
            mapping_write::iosurface_plane_view_sample_window(
                m,
                view.plane_index,
                view.width,
                view.height,
                format,
            )
        else {
            return Err(br(BlitStatus::Bounds, "t5_sample_window"));
        };
        // Whether this arm runs at all. Without it a change to the window this
        // arm resolves cannot be attributed: an unchanged screen and an arm that
        // never executed look identical, and so do a repaired blit and a blit
        // that never happened.
        //
        // Read on a driven x86/Vulkan boot (Safari window drag + two
        // web-content-probe runs): **0** — this arm does not execute on that
        // workload at all, while `blit_dest_bound` reads 26, so the blit path
        // itself does run and it is the IOSurface plane view source that is absent. The
        // plane-index resolution above is therefore contract fidelity, not a
        // repair of anything this workload does, and a screen that looks the
        // same after changing it says nothing either way.
        crate::runtime::drain::note_store_route("blit_t5_plane_device");
        note_blit_iosurface_resident(state, sid);
        return Ok(TextureBacking::Surface(IOSurfaceTextureBacking {
            mapping_id: MappingId::new(sid),
            width: view.width,
            height: view.height,
            surface_offset,
            row_stride: surface_bpr,
            span_end,
            bpp,
            pixel_format: format,
        }));
    }

    if entry.kind != ObjectKind::Texture {
        let _ = note_tex_wrong_type(task_id, texture_ref, entry.kind, level, slice);
        return Err(br(
            BlitStatus::MissingResource,
            crate::observe::ladder_slug!("tex", wrong_type),
        ));
    }
    let Ok(ResourceDescriptor::Texture(tex)) = objects::decoded_resource(&resource) else {
        return Err(br(
            BlitStatus::MissingResource,
            crate::observe::ladder_slug!("tex", desc_decode),
        ));
    };
    let Some(declared_format) = tex.declared_pixel_format() else {
        crate::observe::fail(format!(
            "blit tex no_pixel_format ref={texture_ref} w={} h={} fmt={}",
            tex.width, tex.height, 0
        ));
        return Err(br(BlitStatus::Unsupported, "tex_no_pixel_format"));
    };
    let Some(bpp) = pixel_format::bytes_per_pixel(declared_format) else {
        crate::observe::fail(format!(
            "blit tex bad_bpp ref={texture_ref} fmt={}",
            declared_format
        ));
        return Err(br(BlitStatus::Unsupported, "tex_bad_bpp"));
    };
    let Some((layout_gva, layout)) = tex.level_gva(level as u32, state.page_shift) else {
        crate::observe::fail(format!(
            "blit tex level_gva_shift fail ref={texture_ref} lvl={level} handle={} alloc={} mips={} page_shift={} w={} h={} fmt={:#x}",
            tex.handle,
            tex.allocation_size,
            tex.mipmap_level_count,
            state.page_shift,
            tex.width,
            tex.height,
            declared_format
        ));
        return Err(br(BlitStatus::Bounds, "tex_level_gva"));
    };
    let Some(base_gva) = tex.allocation_base_gva(state.page_shift) else {
        return Err(br(BlitStatus::MissingResource, "tex_no_base_gva"));
    };
    // level_gva already applied offset; keep offset relative to base for plane math.
    let level_offset = match layout_gva.checked_sub(base_gva) {
        Some(v) => v,
        None => {
            crate::observe::fail(format!(
                "blit tex level_offset underflow layout_gva={layout_gva:#x} base={base_gva:#x} page_shift={}",
                state.page_shift
            ));
            return Err(br(BlitStatus::Bounds, "tex_level_offset_underflow"));
        }
    };
    if layout.width == 0 || layout.height == 0 {
        crate::observe::fail(format!(
            "blit tex zero_geom ref={texture_ref} lvl={level} layout={}x{}x{}",
            layout.width, layout.height, layout.depth
        ));
        return Err(br(BlitStatus::Bounds, "tex_zero_geom"));
    }
    let declaration = tex
        .declaration
        .ok_or_else(|| br(BlitStatus::Unsupported, "tex_no_declaration"))?;
    let storage_type = u16::from(declaration.texture_type);
    let is_cube = matches!(
        storage_type,
        crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE
            | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
    );
    if tex.slice_count != u32::from(declaration.array_length)
        || tex.cube_faces != is_cube
        || !tex.declared_packing_fits_allocation()
        || !tex.level_fits_slice(layout)
    {
        return Err(br(BlitStatus::Unsupported, "tex_packing_mismatch"));
    }
    let arrayed = matches!(
        storage_type,
        crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
            | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE
            | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
    );
    let physical_slices = tex
        .physical_slice_count()
        .ok_or_else(|| br(BlitStatus::Bounds, "tex_slice_count_overflow"))?;
    if (!arrayed && slice != 0) || u32::from(slice) >= physical_slices {
        return Err(br(BlitStatus::Bounds, "tex_slice_bounds"));
    }
    let slice_stride = if arrayed { tex.bytes_per_slice } else { 0 };
    let tight_row = pixel_format::tight_row_bytes(layout.width, declared_format)
        .ok_or_else(|| br(BlitStatus::Unsupported, "tex_slice_tight_row"))?;
    let slice_read = layout
        .slice_read_span(tight_row)
        .ok_or_else(|| br(BlitStatus::Bounds, "tex_slice_read_span"))?;
    let slice_start = tex
        .subresource_offset(u32::from(slice), u32::from(level))
        .ok_or_else(|| br(BlitStatus::Bounds, "tex_slice_offset"))?;
    let slice_end = slice_start
        .checked_add(slice_read)
        .ok_or_else(|| br(BlitStatus::Bounds, "tex_slice_overflow"))?;
    if slice_end > tex.allocation_size {
        return Err(br(BlitStatus::Bounds, "tex_slice_bounds"));
    }
    Ok(TextureBacking::Linear(LinearTextureLevel {
        base_gva,
        alloc_size: tex.allocation_size,
        level_offset,
        row_stride: layout.row_stride,
        slice_stride,
        slice_index: slice as u32,
        width: layout.width,
        height: layout.height,
        depth: layout.planes(),
        bpp,
        pixel_format: declared_format,
    }))
}

/// Read one texture row (tight `row_bytes`) at texel (ox, oy+row_i) plane z into `buf`.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper still names the plane geometry a row walk needs"
)]
fn read_texture_row<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    row_bytes: u64,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    if row_bytes as usize > buf.len() {
        return Err(br(BlitStatus::Capacity, "rd_row_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let off = t
                .texel_offset(
                    ox,
                    oy.checked_add(row_i)
                        .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_y_overflow"))?,
                    oz,
                )
                .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_texel_oob"))?;
            let gva = t
                .base_gva
                .checked_add(off)
                .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_gva_overflow"))?;
            if gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                gva,
                &mut buf[..row_bytes as usize],
                state.page_shift,
            )
            .is_err()
            {
                return Err(br(BlitStatus::GuestIo, "rd_row_linear_io"));
            }
            Ok(())
        }
        TextureBacking::Surface(t) => {
            if oz != 0 {
                return Err(br(BlitStatus::Unsupported, "rd_row_iosurface_z"));
            }
            let y = oy
                .checked_add(row_i)
                .ok_or_else(|| br(BlitStatus::Bounds, "rd_row_iosurface_y_overflow"))?;
            if y > u32::MAX as u64 || ox > u32::MAX as u64 {
                return Err(br(BlitStatus::Bounds, "rd_row_iosurface_coord_range"));
            }
            let pixels = (row_bytes / t.bpp as u64) as u32;
            if !mapping_write::read_rect_raw_at(
                state,
                host,
                t.mapping_id.get(),
                mapping_write::SurfaceWindow {
                    base_off: t.surface_offset,
                    bpr: t.row_stride,
                    span_end: t.span_end,
                    bpp: t.bpp,
                },
                mapping_write::Rect {
                    origin_x: ox as u32,
                    origin_y: y as u32,
                    width: pixels,
                    height: 1,
                },
                &mut buf[..row_bytes as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitStatus::GuestIo, "rd_row_iosurface_io"));
            }
            Ok(())
        }
    }
}

/// Write one texture row from `buf`, bounded to the pages the copy's whole
/// destination region resolved to before its row loop started
/// ([`texture_region_window`]).
///
/// `allowed` is not consulted on the IOSurface texture arm: that write goes through the
/// mapping rail, whose authorisation is the page list the guest declared for
/// the mapping itself. That is a different and equally explicit model, not an
/// unbounded one.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper still names the plane geometry a row walk needs"
)]
fn write_texture_row<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    row_bytes: u64,
    buf: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    if row_bytes as usize > buf.len() {
        return Err(br(BlitStatus::Capacity, "wr_row_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let off = t
                .texel_offset(
                    ox,
                    oy.checked_add(row_i)
                        .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_y_overflow"))?,
                    oz,
                )
                .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_texel_oob"))?;
            let gva = t
                .base_gva
                .checked_add(off)
                .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_gva_overflow"))?;
            if gva_mem::write_task_gva_product_within(
                state,
                host,
                task_id,
                gva,
                &buf[..row_bytes as usize],
                allowed,
            )
            .is_err()
            {
                return Err(br(BlitStatus::GuestIo, "wr_row_linear_io"));
            }
            Ok(())
        }
        TextureBacking::Surface(t) => {
            if oz != 0 {
                return Err(br(BlitStatus::Unsupported, "wr_row_iosurface_z"));
            }
            let y = oy
                .checked_add(row_i)
                .ok_or_else(|| br(BlitStatus::Bounds, "wr_row_iosurface_y_overflow"))?;
            if y > u32::MAX as u64 || ox > u32::MAX as u64 {
                return Err(br(BlitStatus::Bounds, "wr_row_iosurface_coord_range"));
            }
            let pixels = (row_bytes / t.bpp as u64) as u32;
            if !mapping_write::write_rect_raw_at(
                state,
                host,
                t.mapping_id.get(),
                mapping_write::SurfaceWindow {
                    base_off: t.surface_offset,
                    bpr: t.row_stride,
                    span_end: t.span_end,
                    bpp: t.bpp,
                },
                mapping_write::Rect {
                    origin_x: ox as u32,
                    origin_y: y as u32,
                    width: pixels,
                    height: 1,
                },
                &buf[..row_bytes as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitStatus::GuestIo, "wr_row_iosurface_io"));
            }
            Ok(())
        }
    }
}

/// Read a whole `row_bytes`-wide, `row_count`-tall rectangle at `origin` into
/// `buf`, rows packed `row_bytes` apart.
///
/// # A rect is the unit the mapping rail is built for, and a row is not
///
/// [`mapping_write::read_rect_raw_at`] takes a height because every per-call
/// cost it carries is per *rect*, not per row: a writeback settle, a mapping
/// lookup, a window revalidation, and — on a fragmented mapping, which is the
/// arm a driven x86 boot takes — a fresh QEMU memory-region import per guest
/// page run. Handing it `height: 1` in a loop pays all of that `row_count`
/// times to move one row of texels.
///
/// That is not a small constant. A driven macos-13 Maps leg measured the
/// slice/level copy's row loop at **30.15 s of a 30.28 s blit rail** while
/// moving 14.6 MB — 0.48 MB/s, against 0.22 s for every strided guest-RAM copy
/// in the device put together. The bytes were never the cost; the per-row
/// re-entry into the mapping rail was.
///
/// The linear arm keeps its row loop, because there the per-row call is a bare
/// guest-RAM read with none of that preamble, and rows of a strided level are
/// genuinely discontiguous.
#[allow(
    clippy::too_many_arguments,
    reason = "the rect helper names the same geometry its row counterpart does"
)]
fn read_texture_rect<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    let need = row_bytes
        .checked_mul(row_count)
        .ok_or_else(|| br(BlitStatus::Capacity, "rd_rect_span_overflow"))?;
    if need as usize > buf.len() {
        return Err(br(BlitStatus::Capacity, "rd_rect_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let (gva, rect) = linear_rect(t, origin, row_bytes, row_count, "rd_rect_linear_shape")?;
            crate::runtime::gva_view::read_rect(state, host, task_id, gva, rect, buf)
                .map_err(|_| br(BlitStatus::GuestIo, "rd_rect_linear_io"))?;
            crate::runtime::drain::note_store_route("blit_rect_linear_read_walk");
            crate::runtime::drain::note_store_route_n(
                "blit_rect_linear_read_rows_hoisted",
                row_count.saturating_sub(1),
            );
            Ok(())
        }
        TextureBacking::Surface(t) => {
            let (pixels, height, origin_x, origin_y) =
                iosurface_rect_extent(t, origin, row_bytes, row_count)?;
            if !mapping_write::read_rect_raw_at(
                state,
                host,
                t.mapping_id.get(),
                iosurface_window(t),
                mapping_write::Rect {
                    origin_x,
                    origin_y,
                    width: pixels,
                    height,
                },
                &mut buf[..need as usize],
                row_bytes as u32,
            ) {
                return Err(br(BlitStatus::GuestIo, "rd_rect_iosurface_io"));
            }
            Ok(())
        }
    }
}

/// A linear level's rectangle as the GVA rail's own shape: where it starts and
/// how its rows are laid out.
///
/// This is the linear endpoint's missing rect description. The IOSurface texture endpoint
/// has had one since it landed — [`mapping_write::write_rect_raw_at`] and
/// friends — while the linear endpoint had only [`write_texture_row`] and
/// [`read_texture_row`], so [`write_texture_rect`] and [`read_texture_rect`]
/// each served a rectangle by re-entering the GVA rail `row_count` times. Every
/// one of those re-entries walks the task page table afresh for a row of the
/// same allocation, so all but the first re-derive an answer already in hand. A
/// driven macos-13 boot charged that loop 906.7 ms of a 916.6 ms
/// texture-to-texture rail across 118 464 rows — about 7.6 us for a 4 KiB row,
/// which is the walk and not the bytes.
///
/// **The stride is the contract's, not an observation.**
/// [`LinearTextureLevel::texel_offset`]'s `y` term is exactly `y * row_stride`,
/// so consecutive rows of one rectangle are `row_stride` apart by construction.
/// That makes the whole rectangle one [`RectStride`] over one span, which is
/// what lets a single walk place every row.
///
/// Only the last row's offset is resolved alongside the first. That is not a
/// two-endpoint sample of a range: `texel_offset` is affine and increasing in
/// `y`, and its only `y` bound is `y < height`, so the largest `y` is the one
/// that can fail and checking it checks them all.
fn linear_rect(
    t: &LinearTextureLevel,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
    site: &'static str,
) -> Result<(u64, RectStride), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let last_y = oy
        .checked_add(row_count.saturating_sub(1))
        .ok_or_else(|| br(BlitStatus::Bounds, site))?;
    let first = t
        .texel_offset(ox, oy, oz)
        .ok_or_else(|| br(BlitStatus::Bounds, site))?;
    t.texel_offset(ox, last_y, oz)
        .ok_or_else(|| br(BlitStatus::Bounds, site))?;
    let gva = t
        .base_gva
        .checked_add(first)
        .ok_or_else(|| br(BlitStatus::Bounds, site))?;
    let rect = RectStride::new(t.row_stride, row_bytes, row_count)
        .ok_or_else(|| br(BlitStatus::Bounds, site))?;
    Ok((gva, rect))
}

/// Write a whole `row_bytes`-wide, `row_count`-tall rectangle at `origin` from
/// `buf`, rows packed `row_bytes` apart. The rect counterpart of
/// [`write_texture_row`]; see [`read_texture_rect`] for why the rect is the
/// unit.
///
/// A rect that covers an IOSurface texture plane entirely goes through
/// [`mapping_write::write_full_rect_raw_at`], whose fragmented arm imports each
/// maximal packed GPA run once instead of once per row. The two calls address
/// identical guest bytes; only the fragmented staging differs.
#[allow(
    clippy::too_many_arguments,
    reason = "the rect helper names the same geometry its row counterpart does"
)]
fn write_texture_rect<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
    buf: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    let need = row_bytes
        .checked_mul(row_count)
        .ok_or_else(|| br(BlitStatus::Capacity, "wr_rect_span_overflow"))?;
    if need as usize > buf.len() {
        return Err(br(BlitStatus::Capacity, "wr_rect_buf_cap"));
    }
    match tex {
        TextureBacking::Linear(t) => {
            let (gva, rect) = linear_rect(t, origin, row_bytes, row_count, "wr_rect_linear_shape")?;
            crate::runtime::gva_view::write_rect_within(
                state, host, task_id, gva, rect, buf, allowed,
            )
            .map_err(|_| br(BlitStatus::GuestIo, "wr_rect_linear_io"))?;
            crate::runtime::drain::note_store_route("blit_rect_linear_walk");
            crate::runtime::drain::note_store_route_n(
                "blit_rect_linear_rows_hoisted",
                row_count.saturating_sub(1),
            );
            Ok(())
        }
        TextureBacking::Surface(t) => {
            let (pixels, height, origin_x, origin_y) =
                iosurface_rect_extent(t, origin, row_bytes, row_count)?;
            let src = &buf[..need as usize];
            let ok = if origin_x == 0 && origin_y == 0 && pixels == t.width && height == t.height {
                mapping_write::write_full_rect_raw_at(
                    state,
                    host,
                    t.mapping_id.get(),
                    t.surface_offset,
                    t.row_stride,
                    t.span_end,
                    pixels,
                    height,
                    t.bpp,
                    src,
                    row_bytes as u32,
                )
            } else {
                mapping_write::write_rect_raw_at(
                    state,
                    host,
                    t.mapping_id.get(),
                    iosurface_window(t),
                    mapping_write::Rect {
                        origin_x,
                        origin_y,
                        width: pixels,
                        height,
                    },
                    src,
                    row_bytes as u32,
                )
            };
            if !ok {
                return Err(br(BlitStatus::GuestIo, "wr_rect_iosurface_io"));
            }
            Ok(())
        }
    }
}

/// The mapping-rail sample window an IOSurface texture backing names.
///
/// Spelled once so the four rect/row call sites cannot drift on which of the
/// four fields a copy presents.
fn iosurface_window(t: &IOSurfaceTextureBacking) -> mapping_write::SurfaceWindow {
    mapping_write::SurfaceWindow {
        base_off: t.surface_offset,
        bpr: t.row_stride,
        span_end: t.span_end,
        bpp: t.bpp,
    }
}

/// Narrow a rect's texel geometry to the `u32` the mapping rail's [`mapping_write::Rect`]
/// is expressed in, refusing by name rather than truncating.
fn iosurface_rect_extent(
    t: &IOSurfaceTextureBacking,
    origin: Point,
    row_bytes: u64,
    row_count: u64,
) -> Result<(u32, u32, u32, u32), BlitStatus> {
    if origin.z != 0 {
        return Err(br(BlitStatus::Unsupported, "rect_iosurface_z"));
    }
    if t.bpp == 0 {
        return Err(br(BlitStatus::Bounds, "rect_iosurface_bpp_zero"));
    }
    let origin_x =
        u32::try_from(origin.x).map_err(|_| br(BlitStatus::Bounds, "rect_iosurface_x_range"))?;
    let origin_y =
        u32::try_from(origin.y).map_err(|_| br(BlitStatus::Bounds, "rect_iosurface_y_range"))?;
    let height = u32::try_from(row_count)
        .map_err(|_| br(BlitStatus::Bounds, "rect_iosurface_height_range"))?;
    let pixels = u32::try_from(row_bytes / t.bpp as u64)
        .map_err(|_| br(BlitStatus::Bounds, "rect_iosurface_width_range"))?;
    Ok((pixels, height, origin_x, origin_y))
}

/// Census: does the surface this blit is about to copy through its **guest
/// pages** have live GPU-resident content instead?
///
/// The sampled rail and the blit rail consume the same wire form — an IOSurface texture
/// IOSurface, named directly or through a IOSurface plane view view — and they resolve it
/// completely differently. `draw::execution`'s sampled resolver runs a four-rung
/// ladder whose top rung is `iosurfacerung_resident`, the engine image, and a driven
/// session puts 64-93 % of its binds there. This resolver has no ladder at all:
/// it returns a [`IOSurfaceTextureBacking`] over the mapping's guest pages every time, and
/// the copy then reads and writes those pages on the CPU.
///
/// That is only sound while the guest pages hold the surface's newest content.
/// The writeback debt is what is supposed to make that true, and
/// `mapping_write`'s settle pays it before every read — but a resident carrying
/// `gpu_only_content` with no debt armed owes nothing, so nothing lands, and the
/// copy reads whatever the pages held before. A blit is not a decode failure and
/// not a refusal: it succeeds, and the pixels are simply not the guest's.
///
/// So this counts rather than branches. `blit_iosurface_resident_ready` above zero
/// says this device is copying a surface whose authoritative content is on the
/// GPU, which is the reading that decides whether the blit rail needs the
/// sampled rail's ladder. `_not_ready` beside it is the denominator, so a zero
/// can be told from an arm that never ran.
/// What a texture-to-texture copy is actually made of: which pair of backings it
/// joins, and how many bytes it moves through the host.
///
/// `walk_blit_us` says this rail costs 33.6 s of a 45 s driven Maps window
/// against 0.45 s for 1.49 M render records, and a per-record average cannot say
/// whether that is a few enormous copies or many small ones, nor which pair of
/// endpoint kinds is paying it. A GPU-side rail can only serve pairs whose
/// **both** ends this device can hold as images, so the pair split is what
/// decides whether such a rail would serve the workload or a corner of it.
///
/// `blit_t2t_bytes` is the denominator for every later claim about this rail: a
/// copy that is genuinely moving a gigabyte a second through `memcpy` is a
/// bandwidth problem, and one that is not is a per-row overhead problem, and the
/// two have opposite repairs. The two reverted attempts in `09a45414` and
/// `81f99f4f` were both aimed at granularity without this number in hand.
fn note_t2t_shape(
    src: &TextureBacking,
    dst: &TextureBacking,
    copy_w: u64,
    copy_h: u64,
    copy_d: u64,
    copy_bpp: u32,
) {
    use crate::runtime::drain::{note_store_route, note_store_route_n};
    note_store_route(match (src.is_surface(), dst.is_surface()) {
        (false, false) => "blit_t2t_linear_linear",
        (false, true) => "blit_t2t_linear_iosurface",
        (true, false) => "blit_t2t_iosurface_linear",
        (true, true) => "blit_t2t_iosurface_to_iosurface",
    });
    let bytes = copy_w
        .saturating_mul(copy_h)
        .saturating_mul(copy_d)
        .saturating_mul(u64::from(copy_bpp));
    note_store_route_n("blit_t2t_bytes", bytes);
    // Banded rather than averaged, because the mean of a full-window copy and a
    // 16x16 icon says nothing about either and this rail issues both.
    note_store_route(match bytes {
        0..=4_095 => "blit_t2t_band_tiny",
        4_096..=262_143 => "blit_t2t_band_small",
        262_144..=4_194_303 => "blit_t2t_band_medium",
        _ => "blit_t2t_band_large",
    });
}

/// Whether a blit endpoint arrived owing this device a writeback, split by which
/// spelling of the debt named it.
///
/// The payment itself is unconditional and silent — `pay_for_texture` early-exits
/// on an empty ledger — so without this the repair is unattributable: a rail that
/// never had a debt to pay and a rail that pays thousands look identical from
/// outside, and the screen cannot tell them apart either.
///
/// `gva` is the interesting one. It counts blit endpoints whose real content was
/// sitting in an engine resident behind an armed
/// [`crate::runtime::writeback_debt::GvaWritebackDebt`], i.e. copies that read
/// transfer backing the render never reached.
fn note_blit_endpoint_debt(state: &Device, task_id: u32, texture_ref: u32) {
    if state.content.pending_writebacks.is_empty() {
        return;
    }
    if crate::runtime::writeback_debt::resource_key(state, task_id, texture_ref)
        .is_some_and(|key| state.content.pending_writebacks.has_gva(key))
    {
        crate::runtime::drain::note_store_route("blit_endpoint_owed_gva");
    }
}

fn note_blit_iosurface_resident(state: &Device, mapping_id: u32) {
    {
        // This census asks the engine a question, and asking takes the engine
        // lock — the same lock the draw rail holds while it encodes and submits.
        // A probe that blocks is not a probe, so time it: if this reads anywhere
        // near `walk_blit_us`, the blit rail's cost is this instrument waiting
        // for the renderer rather than anything the blit itself does.
        let probe_started = std::time::Instant::now();
        let _probe = ProbeClock(probe_started);
        struct ProbeClock(std::time::Instant);
        impl Drop for ProbeClock {
            fn drop(&mut self) {
                crate::runtime::drain::note_store_route_us(
                    "blit_resident_probe_us",
                    self.0.elapsed().as_micros() as u64,
                );
            }
        }
        let Some(m) = state.surfaces.mappings.get(&mapping_id) else {
            return;
        };
        if !m.has_geometry() || m.width_or_zero() == 0 || m.height_or_zero() == 0 {
            crate::runtime::drain::note_store_route("blit_iosurface_resident_no_geom");
            return;
        }
        let (w, h) = (m.width_or_zero(), m.height_or_zero());
        let id = crate::runtime::present_identity::surface_identity(state, mapping_id, w, h);
        crate::runtime::drain::note_store_route(
            match state.executor.resident_read_plan(&id).backing {
                reims_vgpu_core::ResidentContentBacking::NotReady => {
                    "blit_iosurface_resident_not_ready"
                }
                _ => "blit_iosurface_resident_ready",
            },
        );
    }
}

fn range_fits(offset: u64, length: u64, size: u64) -> bool {
    offset <= size && length <= size - offset
}

fn ranges_overlap(a0: u64, a_len: u64, b0: u64, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a1 = a0.saturating_add(a_len);
    let b1 = b0.saturating_add(b_len);
    a0 < b1 && b0 < a1
}

use gva_mem::dest_window;

/// [`dest_window`] over the region of a texture a row loop is about to write.
///
/// The span is measured with the texture's own [`LinearTextureLevel::texel_offset`],
/// first row to last, so a level, slice or plane stride this bound does not
/// model cannot place a row outside the set it authorises.
///
/// `Ok(None)` for an IOSurface texture: that write goes through the mapping rail,
/// authorised by the page list the guest declared for the mapping. Walking a
/// GVA span for it would bound the wrong address space.
///
/// Every other way this can fail to produce a bound is an `Err`, never a
/// `None`. `None` reaches [`write_texture_row`] as "authorised by the command",
/// so an arithmetic failure answered with `None` would *widen* the write from
/// the region the guest named to the whole address space — the opposite of what
/// failing to measure that region should do. The geometry here is the copy the
/// command decoded; if it does not resolve, the command does not execute.
#[allow(
    clippy::too_many_arguments,
    reason = "the window mirrors the copy extent the row loop walks"
)]
fn texture_region_window<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    copy_w: u32,
    copy_h: u64,
    copy_d: u64,
    bpp: u32,
) -> Result<Option<std::collections::HashSet<u64>>, BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let TextureBacking::Linear(t) = tex else {
        return Ok(None);
    };
    // An empty extent authorises no page, which is exact: every caller's row
    // loop is `for z in 0..copy_d { for y in 0..copy_h`, so it writes nothing.
    let (Some(last_row), Some(last_plane)) = (copy_h.checked_sub(1), copy_d.checked_sub(1)) else {
        return Ok(Some(std::collections::HashSet::new()));
    };
    let oob = |slug| br(BlitStatus::Bounds, slug);
    let first = t
        .texel_offset(ox, oy, oz)
        .ok_or_else(|| oob("tex_window_first_texel_oob"))?;
    let last = oy
        .checked_add(last_row)
        .zip(oz.checked_add(last_plane))
        .and_then(|(y, z)| t.texel_offset(ox, y, z))
        .ok_or_else(|| oob("tex_window_last_texel_oob"))?;
    let row_bytes = (copy_w as u64)
        .checked_mul(bpp as u64)
        .ok_or_else(|| oob("tex_window_row_bytes_overflow"))?;
    let span = last
        .checked_add(row_bytes)
        .and_then(|end| end.checked_sub(first))
        .ok_or_else(|| oob("tex_window_span_overflow"))?;
    let base = t
        .base_gva
        .checked_add(first)
        .ok_or_else(|| oob("tex_window_base_overflow"))?;
    Ok(dest_window(state, host, task_id, base, span))
}

/// Highest byte offset past `base` a strided plane/row walk reaches.
///
/// The bound wants the span the command names, not the resource's whole
/// allocation: a copy into the top-left corner of a 64 MiB texture must not be
/// authorised for the other 63. Derived from the walk's own geometry — last
/// plane, last row, one row of bytes — so it cannot drift from the loop it
/// bounds.
fn strided_span(
    row_bytes: u64,
    row_stride: u64,
    row_count: u64,
    image_stride: u64,
    image_count: u64,
) -> Option<u64> {
    let last_image = image_count.checked_sub(1)?.checked_mul(image_stride)?;
    let last_row = row_count.checked_sub(1)?.checked_mul(row_stride)?;
    last_image.checked_add(last_row)?.checked_add(row_bytes)
}

/// Write `length` bytes at `gva` by repeating `pattern` from its first byte.
///
/// One body for both fill records rather than a copy per record. The byte fill
/// is this with a one-byte pattern, which is not a generalisation for its own
/// sake: the bounds check, the `dest_window` authorisation, the chunked write
/// and the GVA advance are the four things a second copy would have to keep in
/// step, and this rail has already produced three bugs from two arms of one
/// guest-memory write drifting apart.
///
/// Phase is preserved across chunks because the tile is a whole number of
/// patterns, so every chunk begins on pattern byte 0 exactly as the first one
/// does. That is asserted rather than assumed — a `CHUNK` that stopped being a
/// multiple of the pattern width would shift every byte after the first tile.
fn write_fill_pattern<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut Device,
    task_id: u32,
    gva: u64,
    length: u64,
    pattern: &[u8],
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    debug_assert!(!pattern.is_empty(), "a fill with no pattern writes nothing");
    debug_assert_eq!(
        CHUNK % pattern.len(),
        0,
        "the staging tile must hold a whole number of patterns, or the phase \
         shifts at every chunk boundary"
    );
    let allowed = dest_window(state, host, task_id, gva, length);
    let mut remaining = length;
    let mut cur = gva;
    let mut chunk = vec![0u8; CHUNK];
    for (i, b) in chunk.iter_mut().enumerate() {
        *b = pattern[i % pattern.len()];
    }
    while remaining > 0 {
        let n = remaining.min(CHUNK as u64) as usize;
        if gva_mem::write_task_gva_product_within(
            state,
            host,
            task_id,
            cur,
            &chunk[..n],
            allowed.as_ref(),
        )
        .is_err()
        {
            return Err(br(BlitStatus::GuestIo, "fill_write_io"));
        }
        cur = cur
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitStatus::Capacity, "fill_gva_advance_overflow"))?;
        remaining -= n as u64;
    }
    Ok(())
}

fn copy_bytes<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut Device,
    task_id: u32,
    src_gva: u64,
    dst_gva: u64,
    length: u64,
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    let allowed = dest_window(state, host, task_id, dst_gva, length);
    copy_bytes_within(
        host,
        state,
        task_id,
        src_gva,
        dst_gva,
        length,
        allowed.as_ref(),
    )
}

/// Execute an immutable buffer operation through the shared command-buffer seam.
///
/// The capability handler owns the host-memory transfer and produces the
/// destination's canonical content version as its completion fact. Ordering,
/// submission identity and completion assembly remain core-owned.
fn execute_resolved_blit<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    operation: ResolvedBlit,
) -> BlitStatus {
    let destination = operation.destination_content().resource;
    let context = crate::runtime::executor::context_for(state, task_id);
    let submission =
        ResolvedSubmission::<(), ()>::single(context, ResolvedCommand::Blit(Box::new(operation)));
    let completion = reims_vgpu_core::execute_resolved_submission(
        submission,
        |_, ()| -> Result<CommandExecution<()>, BlitStatus> { unreachable!() },
        |_, ()| -> Result<CommandExecution<()>, BlitStatus> { unreachable!() },
        |_, operation| {
            match operation {
                ResolvedBlit::Fill {
                    destination,
                    pattern,
                } => write_fill_pattern(
                    host,
                    state,
                    task_id,
                    destination.address.get(),
                    destination.length.get(),
                    pattern.bytes(),
                ),
                ResolvedBlit::Copy {
                    source,
                    destination,
                } => copy_bytes(
                    host,
                    state,
                    task_id,
                    source.address.get(),
                    destination.address.get(),
                    source.length.get(),
                ),
                ResolvedBlit::BufferToTexture(operation) => {
                    execute_resolved_buffer_to_texture(state, host, task_id, operation)
                }
                ResolvedBlit::TextureToBuffer(operation) => {
                    execute_resolved_texture_to_buffer(state, host, task_id, operation)
                }
                ResolvedBlit::TextureToTexture(operation) => {
                    match execute_resolved_texture_to_texture(state, host, task_id, operation) {
                        BlitStatus::Ok => Ok(()),
                        status => Err(status),
                    }
                }
                ResolvedBlit::TextureCopyBatch(operation) => {
                    match execute_resolved_texture_copy_batch(state, host, task_id, operation) {
                        BlitStatus::Ok => Ok(()),
                        status => Err(status),
                    }
                }
            }?;
            let Some(version) = state
                .task_objects
                .resources
                .note_guest_write_by_id(destination)
            else {
                return Err(br(
                    BlitStatus::MissingResource,
                    "blit_completion_resource_gone",
                ));
            };
            Ok(CommandExecution::without_gpu_materialization(
                BlitCompletion {
                    written: Some(ContentStamp {
                        resource: destination,
                        version,
                    }),
                },
            ))
        },
        |_, _| -> Result<CommandExecution<_>, BlitStatus> { unreachable!() },
    );
    match completion {
        Ok(completion)
            if matches!(
                completion.output.as_ref(),
                [ExecutionOutput::Blit(BlitCompletion {
                    written: Some(ContentStamp { resource, .. })
                })] if *resource == destination
            ) =>
        {
            BlitStatus::Ok
        }
        Ok(_) => br(BlitStatus::Unsupported, "blit_completion_mismatch"),
        Err(status) => status,
    }
}

/// Semantic destination of a command which can complete a guest-memory write.
///
/// This is deliberately independent of backing class: unified and discrete
/// placement may choose different transports, but both complete the same
/// resource transition. Zero-extent and refused operations never call it.
fn blit_write_destination(cmd: &Command) -> Option<u32> {
    match cmd.kind {
        // Buffer destinations and buffer-to-texture destinations transition
        // inside the immutable command's completion handler. The remaining
        // texture families settle here temporarily.
        Kind::FillBuffer | Kind::FillBufferPattern4 => None,
        Kind::Copy => match cmd.copy_kind {
            CopyKind::BufferToBuffer
            | CopyKind::BufferToTexture
            | CopyKind::TextureToBuffer
            | CopyKind::TextureToTexture
            | CopyKind::TextureToTextureSliceLevel => None,
            CopyKind::None => None,
        },
        Kind::Fence
        | Kind::Resource
        | Kind::Image
        | Kind::Unknown
        | Kind::IcbRange
        | Kind::IcbCopy
        | Kind::FillTexture
        | Kind::InvalidateCompressedTexture => None,
    }
}

/// Apply the content transition named by a successful synchronous blit.
fn complete_blit_write(state: &Device, task_id: u32, cmd: &Command) -> BlitStatus {
    let Some(object_ref) = blit_write_destination(cmd) else {
        return BlitStatus::Ok;
    };
    let Some(resource) = state.task_objects.resources.identity(task_id, object_ref) else {
        return br(BlitStatus::MissingResource, "blit_completion_resource_gone");
    };
    if state
        .task_objects
        .resources
        .note_guest_write_by_id(resource)
        .is_none()
    {
        return br(BlitStatus::MissingResource, "blit_completion_resource_gone");
    }
    BlitStatus::Ok
}

/// [`copy_bytes`] with the destination window supplied rather than captured.
///
/// The split exists so a test can run the identical loop unbounded and show
/// where the bytes land without it. Product code calls [`copy_bytes`]; there is
/// no product caller that should be choosing its own window here, because the
/// window this copy is entitled to is the destination span it was given.
#[allow(
    clippy::too_many_arguments,
    reason = "the bounded form adds the window to the copy's own geometry"
)]
fn copy_bytes_within<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut Device,
    task_id: u32,
    src_gva: u64,
    dst_gva: u64,
    length: u64,
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    if length == 0 {
        return Ok(());
    }
    let mut remaining = length;
    let mut s = src_gva;
    let mut d = dst_gva;
    let mut buf = vec![0u8; CHUNK.min(length as usize).max(1)];
    while remaining > 0 {
        let n = remaining.min(buf.len() as u64) as usize;
        if gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            s,
            &mut buf[..n],
            state.page_shift,
        )
        .is_err()
        {
            return Err(br(BlitStatus::GuestIo, "copy_bytes_read_io"));
        }
        if gva_mem::write_task_gva_product_within(state, host, task_id, d, &buf[..n], allowed)
            .is_err()
        {
            return Err(br(BlitStatus::GuestIo, "copy_bytes_write_io"));
        }
        s = s
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_bytes_src_overflow"))?;
        d = d
            .checked_add(n as u64)
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_bytes_dst_overflow"))?;
        remaining -= n as u64;
    }
    Ok(())
}

/// Copy a rectangular multi-plane region with independent source/dest strides.
#[allow(
    clippy::too_many_arguments,
    reason = "the copy helper mirrors independent source and destination row geometry"
)]
fn copy_row_region<M: HostMemory + HostOps>(
    host: &mut M,
    state: &mut Device,
    task_id: u32,
    src_base: u64,
    src_row_stride: u64,
    src_image_stride: u64,
    dst_base: u64,
    dst_row_stride: u64,
    dst_image_stride: u64,
    row_bytes: u64,
    row_count: u64,
    image_count: u64,
) -> Result<(), BlitStatus> {
    if row_bytes == 0 || row_count == 0 || image_count == 0 {
        return Ok(());
    }
    // Stride/row contract only — no host MiB byte budget (chunked row I/O).
    if row_bytes > src_row_stride || row_bytes > dst_row_stride {
        return Err(br(BlitStatus::Bounds, "copy_region_row_gt_stride"));
    }
    let _total = row_bytes
        .checked_mul(row_count)
        .and_then(|v| v.checked_mul(image_count))
        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_total_overflow"))?;
    let row_len = host_alloc_len(row_bytes)
        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_row_alloc"))?;
    let dst_span = strided_span(
        row_bytes,
        dst_row_stride,
        row_count,
        dst_image_stride,
        image_count,
    )
    .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_span_overflow"))?;
    // The window is built once and the rows run against it, so a single total
    // over the pair cannot say which is the cost. `dest_window` walks the whole
    // destination span's guest page table into a `HashSet`, which is per-record
    // work that does not shrink with the copy; the row loop is per-row work that
    // does. Timed apart because the repair differs.
    let window_started = std::time::Instant::now();
    let allowed = dest_window(state, host, task_id, dst_base, dst_span);
    crate::runtime::drain::note_store_route_us(
        "blit_window_us",
        window_started.elapsed().as_micros() as u64,
    );
    let rows_started = std::time::Instant::now();
    crate::runtime::drain::note_store_route_n("blit_rows_n", row_count.saturating_mul(image_count));
    let mut row_buf = vec![0u8; row_len];
    for z in 0..image_count {
        let src_plane = src_base
            .checked_add(
                z.checked_mul(src_image_stride)
                    .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_plane_overflow"))?,
            )
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_plane_overflow"))?;
        let dst_plane = dst_base
            .checked_add(
                z.checked_mul(dst_image_stride)
                    .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_plane_overflow"))?,
            )
            .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_plane_overflow"))?;
        for y in 0..row_count {
            let s = src_plane
                .checked_add(
                    y.checked_mul(src_row_stride)
                        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_row_overflow"))?,
                )
                .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_src_row_overflow"))?;
            let d = dst_plane
                .checked_add(
                    y.checked_mul(dst_row_stride)
                        .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_row_overflow"))?,
                )
                .ok_or_else(|| br(BlitStatus::Capacity, "copy_region_dst_row_overflow"))?;
            if gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                s,
                &mut row_buf,
                state.page_shift,
            )
            .is_err()
            {
                note_copy_region_io(task_id, false, s, y, z, row_bytes, state.page_shift);
                return Err(br(BlitStatus::GuestIo, "copy_region_read_io"));
            }
            if gva_mem::write_task_gva_product_within(
                state,
                host,
                task_id,
                d,
                &row_buf,
                allowed.as_ref(),
            )
            .is_err()
            {
                note_copy_region_io(task_id, true, d, y, z, row_bytes, state.page_shift);
                return Err(br(BlitStatus::GuestIo, "copy_region_write_io"));
            }
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_rows_us",
        rows_started.elapsed().as_micros() as u64,
    );
    Ok(())
}

fn exec_fill_buffer<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    if cmd.range_length == 0 {
        return BlitStatus::ZeroExtent;
    }
    let buf = match resolve_buffer(state, host, task_id, cmd.buffer) {
        Ok(b) => b,
        Err(st) => return st,
    };
    let Some(destination) = buf.range(cmd.range_location, cmd.range_length) else {
        return br(BlitStatus::Bounds, "fill_range_oob");
    };
    execute_resolved_blit(
        state,
        host,
        task_id,
        ResolvedBlit::Fill {
            destination,
            pattern: BufferFillPattern::Byte(cmd.fill_value),
        },
    )
}

/// `fillBuffer:range:pattern4:` — the byte fill with a repeating 32-bit unit.
///
/// Same resolve, same bounds check and same write path as
/// [`exec_fill_buffer`]; only the chunk this repeats differs. That is the whole
/// difference between the two records on the wire too — one length, one layout,
/// and a last field that is one byte wide in `0x132` and four in `0x13f`.
///
/// # Why an unaligned range is refused rather than filled
///
/// The record settles *what* repeats and says nothing about the **phase** it
/// repeats on, and there are two readings that differ:
///
/// - the pattern restarts at `range.location`, so byte `i` of the range takes
///   `pattern[i % 4]`;
/// - the pattern is anchored to the buffer, so the byte at buffer offset `o`
///   takes `pattern[o % 4]`.
///
/// The two agree for every byte exactly when `range.location % 4 == 0`, and
/// nothing this project can reach decides between them otherwise: the
/// serializer forwards the arguments untouched, so the choice belongs to
/// Apple's host implementation and no capture of the command stream can see it.
///
/// So the aligned case is executed — no guess is being made there — and the
/// unaligned case is refused by name. Filling it under either reading would put
/// bytes in guest memory that are plausible and possibly wrong, which is worse
/// than a refusal the fail log explains. `fill_pattern4_unaligned_range` is a
/// healthy zero, and a non-zero reading is the measured argument for going and
/// deriving the phase rule.
fn exec_fill_buffer_pattern4<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    if cmd.range_length == 0 {
        return BlitStatus::ZeroExtent;
    }
    let pattern = cmd.fill_pattern.to_le_bytes();
    if !cmd.range_location.is_multiple_of(pattern.len() as u64) {
        return br(BlitStatus::Unsupported, "fill_pattern4_unaligned_range");
    }
    let buf = match resolve_buffer(state, host, task_id, cmd.buffer) {
        Ok(b) => b,
        Err(st) => return st,
    };
    let Some(destination) = buf.range(cmd.range_location, cmd.range_length) else {
        return br(BlitStatus::Bounds, "fill_pattern4_range_oob");
    };
    execute_resolved_blit(
        state,
        host,
        task_id,
        ResolvedBlit::Fill {
            destination,
            pattern: BufferFillPattern::Word(pattern),
        },
    )
}

fn exec_copy_buffer_to_buffer<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    if cmd.size == 0 {
        return BlitStatus::ZeroExtent;
    }
    let src = match resolve_buffer(state, host, task_id, cmd.source) {
        Ok(b) => b,
        Err(st) => return st,
    };
    let dst = match resolve_buffer(state, host, task_id, cmd.destination) {
        Ok(b) => b,
        Err(st) => return st,
    };
    if !range_fits(cmd.source_offset, cmd.size, src.size)
        || !range_fits(cmd.destination_offset, cmd.size, dst.size)
    {
        return br(BlitStatus::Bounds, "b2b_range_oob");
    }
    // Same allocation (same GVA base + size): reject overlapping windows.
    if src.gva == dst.gva
        && src.size == dst.size
        && ranges_overlap(
            cmd.source_offset,
            cmd.size,
            cmd.destination_offset,
            cmd.size,
        )
    {
        return br(BlitStatus::Overlap, "b2b_overlap");
    }
    let (Some(source), Some(destination)) = (
        src.range(cmd.source_offset, cmd.size),
        dst.range(cmd.destination_offset, cmd.size),
    ) else {
        return br(BlitStatus::Bounds, "b2b_resolved_range_oob");
    };
    execute_resolved_blit(
        state,
        host,
        task_id,
        ResolvedBlit::Copy {
            source,
            destination,
        },
    )
}

/// A copy extent the guest asked for, checked against what the texture holds.
///
/// `None` means the region reaches past the edge and the caller must refuse.
///
/// **This used to clamp**, at nine sites here plus three destination-side cuts
/// in the texture-to-texture path that did not even come through it, and it
/// reported nothing. A truncation there was a smaller copy returned as `Ok`:
/// the guest asked Metal to move a W x H region, got fewer texels, and the
/// texels outside the cut kept whatever the destination held before, with no
/// status saying so.
///
/// Three things say refusing is the faithful answer, and no one of them would
/// have been enough alone:
///
/// * **Metal refuses.** A region reaching past a texture is a validation
///   failure there, not a clipped copy. Emulating the clip emulates a device
///   Apple does not ship.
/// * **The origin check beside every call site already refuses**
///   (`t2t_origin_oob` and its siblings). Origin and extent are two halves of
///   one region out of one wire record and were handled opposite ways — which
///   is exactly the divergence `AGENTS.md` says to look for by diffing two arms
///   that consume one wire form, and which nobody had diffed.
/// * **The cut never fires.** Measured before it was changed, on a driven x86
///   boot with Safari composited on a Ventura desktop: `blit_extent_fits` 66,
///   `blit_extent_cut` **0**. The path is reached and the clamp is not.
///
/// The counters stay, and `blit_extent_cut` keeps its name so a boot series
/// spanning this change stays comparable — it is now the refusal's volume
/// rather than a silent truncation's. A firing is a workload this rig has not
/// seen, and it will be loud instead of a wrong texture.
fn copy_extent(kind: &'static str, axis: &'static str, requested: u64, max: u64) -> Option<u64> {
    if requested == 0 {
        // Metal size 0 is a no-op extent; keep 0. The callers below turn an
        // all-zero extent into `ZeroExtent`, which is not a refusal.
        Some(0)
    } else if requested > max {
        note_extent_over(kind, axis, requested, max);
        None
    } else {
        crate::runtime::drain::note_store_route("blit_extent_fits");
        Some(requested)
    }
}

/// The always-on half of [`copy_extent`], shared with the destination-side
/// checks in the texture-to-texture path that do not go through it.
///
/// Latched per `(kind, axis)` rather than per size: what a reader needs first is
/// which copy was refused and on which dimension, and a per-size latch on a
/// window drag would emit once per distinct window width.
fn note_extent_over(kind: &'static str, axis: &'static str, requested: u64, max: u64) {
    crate::runtime::drain::note_store_route("blit_extent_cut");
    let key = kind
        .bytes()
        .chain(axis.bytes())
        .fold(0u64, |acc, b| acc.rotate_left(7) ^ u64::from(b));
    if !crate::observe::first_sight("blit_extent_cut", key) {
        return;
    }
    crate::observe::fail(format!(
        "blit_extent reason=blit_extent_over kind={kind} axis={axis}          requested={requested} available={max} (the guest asked to copy past          the edge of a texture; Metal refuses that region and so does this,          where it used to copy less and report Ok)"
    ));
}

/// Resolve `MTLBlitOption` → aspect flags + buffer-side plane bpp.
fn copy_aspect_for_options(
    texture_format: u16,
    cmd: &Command,
) -> Result<(BlitAspect, u32), BlitStatus> {
    // The three option checks used to collapse into a bare `Unsupported` with
    // the reason discarded by `map_err(|_| ..)`. The blit reason channel carries
    // the specific slug to the dispatch-site line, so an unknown option bit and
    // a depth+stencil conflict no longer read identically.
    let aspect = blit::parse_blit_options(cmd.has_options, cmd.options)
        .map_err(|e: blit::BlitOptionError| br(BlitStatus::Unsupported, e.slug()))?;
    let bpp = pixel_format::blit_aspect_bytes_per_pixel(texture_format, aspect)
        .ok_or(BlitStatus::Unsupported)?;
    Ok((aspect, bpp))
}

/// Texture-side full texel bpp (storage). Plane copies use this for GVA strides.
fn texture_storage_bpp(format: u16) -> Result<u32, BlitStatus> {
    pixel_format::bytes_per_pixel(format).ok_or(BlitStatus::Unsupported)
}

/// Read one packed texture row (tight `width * storage_bpp`) at (ox, oy+row_i, oz).
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps packed texture coordinates and format explicit"
)]
fn read_texture_storage_row<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    width: u32,
    storage_bpp: u32,
    buf: &mut [u8],
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let row_bytes = (width as u64)
        .checked_mul(storage_bpp as u64)
        .ok_or(BlitStatus::Capacity)?;
    if row_bytes as usize > buf.len() {
        return Err(BlitStatus::Capacity);
    }
    // Reuse read_texture_row but with storage row size (not plane size).
    // Temporarily: call the same GVA path with storage row_bytes.
    read_texture_row(
        state,
        host,
        task_id,
        tex,
        Point {
            x: ox,
            y: oy,
            z: oz,
        },
        row_i,
        row_bytes,
        buf,
    )
}

/// Write one packed texture row.
#[allow(
    clippy::too_many_arguments,
    reason = "the row helper keeps packed texture coordinates and format explicit"
)]
fn write_texture_storage_row<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    tex: &TextureBacking,
    origin: Point,
    row_i: u64,
    width: u32,
    storage_bpp: u32,
    buf: &[u8],
    allowed: crate::runtime::gva_view::WindowPages<'_>,
) -> Result<(), BlitStatus> {
    let Point {
        x: ox,
        y: oy,
        z: oz,
    } = origin;
    let row_bytes = (width as u64)
        .checked_mul(storage_bpp as u64)
        .ok_or(BlitStatus::Capacity)?;
    write_texture_row(
        state,
        host,
        task_id,
        tex,
        Point {
            x: ox,
            y: oy,
            z: oz,
        },
        row_i,
        row_bytes,
        buf,
        allowed,
    )
}

/// Copy buffer plane rows ↔ texture with optional combined-DS plane repack.
#[allow(
    clippy::too_many_arguments,
    reason = "the blit executor mirrors the decoded buffer, texture, and aspect fields"
)]
fn copy_buffer_texture_rows_aspect<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    buf_base_gva: u64,
    buf_row_stride: u64,
    buf_image_stride: u64,
    tex: &TextureBacking,
    tex_origin: Point,
    copy_w: u32,
    copy_h: u64,
    copy_d: u64,
    plane_bpp: u32,
    aspect: BlitAspect,
    to_texture: bool,
) -> Result<(), BlitStatus> {
    let Point {
        x: tex_ox,
        y: tex_oy,
        z: tex_oz,
    } = tex_origin;
    let fmt = tex.pixel_format();
    let repack = pixel_format::blit_aspect_needs_repack(fmt, aspect);
    let storage_bpp = if repack {
        texture_storage_bpp(fmt)?
    } else {
        plane_bpp
    };
    let plane_row = (copy_w as u64)
        .checked_mul(plane_bpp as u64)
        .ok_or(BlitStatus::Capacity)? as usize;
    let storage_row = (copy_w as u64)
        .checked_mul(storage_bpp as u64)
        .ok_or(BlitStatus::Capacity)? as usize;
    // The other half of the rail. `note_t2t_shape` covers texture-to-texture
    // only, and that left 4 243 of a driven Maps leg's 26 234 blit records
    // uncounted — a population big enough to hold the whole per-record cost if
    // its copies are large, and invisible in a census that stops at one copy
    // kind. Same three readings, so the two halves are comparable line for line.
    {
        use crate::runtime::drain::{note_store_route, note_store_route_n};
        note_store_route(match (to_texture, tex.is_surface()) {
            (true, false) => "blit_b2t_linear",
            (true, true) => "blit_b2t_iosurface",
            (false, false) => "blit_t2b_linear",
            (false, true) => "blit_t2b_iosurface",
        });
        let bytes = (plane_row as u64)
            .saturating_mul(copy_h)
            .saturating_mul(copy_d);
        note_store_route_n("blit_bt_bytes", bytes);
        note_store_route_n("blit_bt_rows_n", copy_h.saturating_mul(copy_d));
        note_store_route(match bytes {
            0..=4_095 => "blit_bt_band_tiny",
            4_096..=262_143 => "blit_bt_band_small",
            262_144..=4_194_303 => "blit_bt_band_medium",
            _ => "blit_bt_band_large",
        });
    }
    let mut plane = vec![0u8; plane_row];
    let mut packed = vec![0u8; storage_row.max(plane_row)];
    // Destination pages, taken before the loop below rather than per row. The
    // loop *is* the copy -- up to `copy_d * copy_h` guest reads and writes -- and
    // it re-derives its destination from `tex`/`buf_base_gva` on every one of
    // them, which is the drift `dest_window` exists to close.
    let allowed = if to_texture {
        texture_region_window(
            state,
            host,
            task_id,
            tex,
            Point {
                x: tex_ox,
                y: tex_oy,
                z: tex_oz,
            },
            copy_w,
            copy_h,
            copy_d,
            storage_bpp,
        )?
    } else {
        let span = strided_span(
            plane_row as u64,
            buf_row_stride,
            copy_h,
            buf_image_stride,
            copy_d,
        )
        .ok_or(BlitStatus::Capacity)?;
        dest_window(state, host, task_id, buf_base_gva, span)
    };
    // The half `blit_rows_us` cannot see. That counter sits in `copy_row_region`,
    // which only the linear-to-linear fast path reaches; every IOSurface texture and
    // IOSurface plane view endpoint stages through this loop instead, and each of its rows
    // re-vouches the mapping's guest page table. `mapw_pages_vouched` reads over
    // a million on a driven Maps leg and nothing timed the loop that spends them.
    let bt_rows_started = std::time::Instant::now();
    for z in 0..copy_d {
        for y in 0..copy_h {
            let buf_gva = buf_base_gva
                .checked_add(
                    z.checked_mul(buf_image_stride)
                        .ok_or(BlitStatus::Capacity)?,
                )
                .ok_or(BlitStatus::Capacity)?
                .checked_add(y.checked_mul(buf_row_stride).ok_or(BlitStatus::Capacity)?)
                .ok_or(BlitStatus::Capacity)?;
            if to_texture {
                if gva_mem::read_task_gva_by_id(
                    host,
                    &state.tasks,
                    task_id,
                    buf_gva,
                    &mut plane,
                    state.page_shift,
                )
                .is_err()
                {
                    return Err(BlitStatus::GuestIo);
                }
                if repack {
                    // RMW: load existing packed row, insert plane, store.
                    read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        tex,
                        Point {
                            x: tex_ox,
                            y: tex_oy,
                            z: tex_oz + z,
                        },
                        y,
                        copy_w,
                        storage_bpp,
                        &mut packed,
                    )?;
                    if !pixel_format::insert_plane_row(
                        fmt,
                        aspect,
                        &plane,
                        copy_w,
                        &mut packed[..storage_row],
                    ) {
                        return Err(BlitStatus::Unsupported);
                    }
                    write_texture_storage_row(
                        state,
                        host,
                        task_id,
                        tex,
                        Point {
                            x: tex_ox,
                            y: tex_oy,
                            z: tex_oz + z,
                        },
                        y,
                        copy_w,
                        storage_bpp,
                        &packed[..storage_row],
                        allowed.as_ref(),
                    )?;
                } else {
                    write_texture_row(
                        state,
                        host,
                        task_id,
                        tex,
                        Point {
                            x: tex_ox,
                            y: tex_oy,
                            z: tex_oz + z,
                        },
                        y,
                        plane_row as u64,
                        &plane,
                        allowed.as_ref(),
                    )?;
                }
            } else if repack {
                read_texture_storage_row(
                    state,
                    host,
                    task_id,
                    tex,
                    Point {
                        x: tex_ox,
                        y: tex_oy,
                        z: tex_oz + z,
                    },
                    y,
                    copy_w,
                    storage_bpp,
                    &mut packed,
                )?;
                if !pixel_format::extract_plane_row(
                    fmt,
                    aspect,
                    &packed[..storage_row],
                    copy_w,
                    &mut plane,
                ) {
                    return Err(BlitStatus::Unsupported);
                }
                if gva_mem::write_task_gva_product_within(
                    state,
                    host,
                    task_id,
                    buf_gva,
                    &plane,
                    allowed.as_ref(),
                )
                .is_err()
                {
                    return Err(BlitStatus::GuestIo);
                }
            } else {
                read_texture_row(
                    state,
                    host,
                    task_id,
                    tex,
                    Point {
                        x: tex_ox,
                        y: tex_oy,
                        z: tex_oz + z,
                    },
                    y,
                    plane_row as u64,
                    &mut plane,
                )?;
                if gva_mem::write_task_gva_product_within(
                    state,
                    host,
                    task_id,
                    buf_gva,
                    &plane,
                    allowed.as_ref(),
                )
                .is_err()
                {
                    return Err(BlitStatus::GuestIo);
                }
            }
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_bt_rows_us",
        bt_rows_started.elapsed().as_micros() as u64,
    );
    Ok(())
}

fn execute_resolved_buffer_to_texture<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    operation: ResolvedBufferToTextureBlit,
) -> Result<(), BlitStatus> {
    let ResolvedBufferToTextureBlit {
        source,
        source_bytes_per_row,
        source_bytes_per_image,
        destination,
        destination_origin,
        extent,
        aspect,
    } = operation;
    let src = LinearBuffer {
        content: source.content,
        gva: source.address.get(),
        size: source.length.get(),
    };
    let dst = destination.backing;
    let copy_bpp = pixel_format::blit_aspect_bytes_per_pixel(dst.pixel_format(), aspect)
        .ok_or_else(|| br(BlitStatus::Unsupported, "blit_options_aspect_format"))?;
    let repack = pixel_format::blit_aspect_needs_repack(dst.pixel_format(), aspect);
    // IOSurface texture is 2D only.
    if dst.is_surface() && (destination_origin.z != 0 || extent.depth > 1) {
        if extent.depth == 0 {
            return Err(BlitStatus::ZeroExtent);
        }
        if destination_origin.z != 0 || extent.depth != 1 {
            return Err(br(BlitStatus::Unsupported, "b2t_iosurface_z_or_depth"));
        }
    }
    let ox = destination_origin.x;
    let oy = destination_origin.y;
    let oz = destination_origin.z;
    if ox > dst.width() as u64 || oy > dst.height() as u64 || oz > dst.depth() as u64 {
        return Err(br(BlitStatus::Bounds, "b2t_origin_oob"));
    }
    // Refused rather than clipped, and the origin check directly above is why
    // the two now agree: one wire record names a region, and both halves of it
    // are checked the same way.
    let (Some(copy_w), Some(copy_h)) = (
        copy_extent("b2t", "w", extent.width, dst.width() as u64 - ox),
        copy_extent("b2t", "h", extent.height, dst.height() as u64 - oy),
    ) else {
        return Err(br(BlitStatus::Bounds, "b2t_extent_oob"));
    };
    let copy_d = if extent.depth == 0 {
        0
    } else {
        match copy_extent("b2t", "d", extent.depth, dst.depth() as u64 - oz) {
            Some(d) => d,
            None => return Err(br(BlitStatus::Bounds, "b2t_extent_oob")),
        }
    };
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return Err(BlitStatus::ZeroExtent);
    }
    // Buffer-side plane bpp (aspect-aware).
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return Err(br(BlitStatus::Capacity, "b2t_row_bytes_overflow")),
    };
    let src_bpr = if source_bytes_per_row != 0 {
        source_bytes_per_row
    } else {
        row_bytes
    };
    if src_bpr < row_bytes {
        return Err(br(BlitStatus::Bounds, "b2t_src_bpr_lt_row"));
    }
    let src_bpi = if source_bytes_per_image != 0 {
        source_bytes_per_image
    } else {
        match src_bpr.checked_mul(copy_h) {
            Some(v) => v,
            None => return Err(br(BlitStatus::Capacity, "b2t_src_bpi_overflow")),
        }
    };
    // Combined DS + aspect: plane repack path (not raw GVA span).
    if repack {
        return copy_buffer_texture_rows_aspect(
            state,
            host,
            task_id,
            src.gva,
            src_bpr,
            src_bpi,
            &dst,
            Point {
                x: ox,
                y: oy,
                z: oz,
            },
            copy_w as u32,
            copy_h,
            copy_d,
            copy_bpp,
            aspect,
            true,
        );
    }
    // Prefer direct GVA row-span when both sides linear (dst only texture here).
    if let TextureBacking::Linear(ref lt) = dst {
        let dst_off = match lt.texel_offset(ox, oy, oz) {
            Some(v) => v,
            None => return Err(br(BlitStatus::Bounds, "b2t_dst_texel_oob")),
        };
        let dst_bpi = match lt.bytes_per_image() {
            Some(v) => v,
            None => return Err(br(BlitStatus::Capacity, "b2t_dst_bpi_overflow")),
        };
        let last = match dst_off
            .checked_add((copy_d - 1).saturating_mul(dst_bpi))
            .and_then(|v| v.checked_add((copy_h - 1).saturating_mul(lt.row_stride)))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return Err(br(BlitStatus::Bounds, "b2t_dst_span_overflow")),
        };
        if lt.alloc_size != 0 && last > lt.alloc_size {
            return Err(br(BlitStatus::Bounds, "b2t_dst_alloc_oob"));
        }
        let src_span = match (copy_d - 1)
            .saturating_mul(src_bpi)
            .checked_add((copy_h - 1).saturating_mul(src_bpr))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return Err(br(BlitStatus::Bounds, "b2t_src_span_overflow")),
        };
        if src_span > src.size {
            return Err(br(BlitStatus::Bounds, "b2t_src_span_oob"));
        }
        let dst_gva = match lt.base_gva.checked_add(dst_off) {
            Some(v) => v,
            None => return Err(br(BlitStatus::Bounds, "b2t_dst_gva_overflow")),
        };
        return copy_row_region(
            host,
            state,
            task_id,
            src.gva,
            src_bpr,
            src_bpi,
            dst_gva,
            lt.row_stride,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        );
    }
    // IOSurface texture destination: row-stage from buffer GVA.
    let src_span = match (copy_d - 1)
        .saturating_mul(src_bpi)
        .checked_add((copy_h - 1).saturating_mul(src_bpr))
        .and_then(|v| v.checked_add(row_bytes))
    {
        Some(v) => v,
        None => return Err(br(BlitStatus::Bounds, "b2t_iosurface_src_span_overflow")),
    };
    if src_span > src.size {
        return Err(br(BlitStatus::Bounds, "b2t_iosurface_src_span_oob"));
    }
    // `None` for the IOSurface texture destination this arm is for, which the mapping rail
    // authorises instead; a linear destination reaching here is still bounded.
    let allowed = texture_region_window(
        state,
        host,
        task_id,
        &dst,
        Point {
            x: ox,
            y: oy,
            z: oz,
        },
        copy_w as u32,
        copy_h,
        copy_d,
        copy_bpp,
    )?;
    let mut row = vec![0u8; row_bytes as usize];
    for z in 0..copy_d {
        for y in 0..copy_h {
            let s = match src
                .gva
                .checked_add(z.saturating_mul(src_bpi))
                .and_then(|b| b.checked_add(y.saturating_mul(src_bpr)))
            {
                Some(v) => v,
                None => return Err(br(BlitStatus::Bounds, "b2t_iosurface_src_gva_overflow")),
            };
            if gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                s,
                &mut row,
                state.page_shift,
            )
            .is_err()
            {
                return Err(br(BlitStatus::GuestIo, "b2t_iosurface_read_io"));
            }
            write_texture_row(
                state,
                host,
                task_id,
                &dst,
                Point {
                    x: ox,
                    y: oy,
                    z: oz + z,
                },
                y,
                row_bytes,
                &row,
                allowed.as_ref(),
            )?;
        }
    }
    Ok(())
}

fn execute_resolved_texture_to_buffer<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    operation: ResolvedTextureToBufferBlit,
) -> Result<(), BlitStatus> {
    let ResolvedTextureToBufferBlit {
        source,
        source_origin,
        extent,
        destination,
        destination_bytes_per_row,
        destination_bytes_per_image,
        aspect,
    } = operation;
    let src = source.backing;
    let copy_bpp = pixel_format::blit_aspect_bytes_per_pixel(src.pixel_format(), aspect)
        .ok_or_else(|| br(BlitStatus::Unsupported, "blit_options_aspect_format"))?;
    let dst = LinearBuffer {
        content: destination.content,
        gva: destination.address.get(),
        size: destination.length.get(),
    };
    let repack = pixel_format::blit_aspect_needs_repack(src.pixel_format(), aspect);
    if src.is_surface() && (source_origin.z != 0 || extent.depth > 1) {
        if extent.depth == 0 {
            return Err(BlitStatus::ZeroExtent);
        }
        if source_origin.z != 0 || extent.depth != 1 {
            return Err(br(BlitStatus::Unsupported, "t2b_iosurface_z_or_depth"));
        }
    }
    let ox = source_origin.x;
    let oy = source_origin.y;
    let oz = source_origin.z;
    if ox > src.width() as u64 || oy > src.height() as u64 || oz > src.depth() as u64 {
        return Err(br(BlitStatus::Bounds, "t2b_origin_oob"));
    }
    // Refused rather than clipped, and the origin check directly above is why
    // the two now agree: one wire record names a region, and both halves of it
    // are checked the same way.
    let (Some(copy_w), Some(copy_h)) = (
        copy_extent("t2b", "w", extent.width, src.width() as u64 - ox),
        copy_extent("t2b", "h", extent.height, src.height() as u64 - oy),
    ) else {
        return Err(br(BlitStatus::Bounds, "t2b_extent_oob"));
    };
    let copy_d = if extent.depth == 0 {
        0
    } else {
        match copy_extent("t2b", "d", extent.depth, src.depth() as u64 - oz) {
            Some(d) => d,
            None => return Err(br(BlitStatus::Bounds, "t2b_extent_oob")),
        }
    };
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return Err(BlitStatus::ZeroExtent);
    }
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return Err(br(BlitStatus::Capacity, "t2b_row_bytes_overflow")),
    };
    let dst_bpr = if destination_bytes_per_row != 0 {
        destination_bytes_per_row
    } else {
        row_bytes
    };
    if dst_bpr < row_bytes {
        return Err(br(BlitStatus::Bounds, "t2b_dst_bpr_lt_row"));
    }
    let dst_bpi = if destination_bytes_per_image != 0 {
        destination_bytes_per_image
    } else {
        match dst_bpr.checked_mul(copy_h) {
            Some(v) => v,
            None => return Err(br(BlitStatus::Capacity, "t2b_dst_bpi_overflow")),
        }
    };
    if repack {
        return copy_buffer_texture_rows_aspect(
            state,
            host,
            task_id,
            dst.gva,
            dst_bpr,
            dst_bpi,
            &src,
            Point {
                x: ox,
                y: oy,
                z: oz,
            },
            copy_w as u32,
            copy_h,
            copy_d,
            copy_bpp,
            aspect,
            false,
        );
    }
    if let TextureBacking::Linear(ref lt) = src {
        let src_off = match lt.texel_offset(ox, oy, oz) {
            Some(v) => v,
            None => return Err(br(BlitStatus::Bounds, "t2b_src_texel_oob")),
        };
        let src_bpi = match lt.bytes_per_image() {
            Some(v) => v,
            None => return Err(br(BlitStatus::Capacity, "t2b_src_bpi_overflow")),
        };
        let dst_span = match (copy_d - 1)
            .saturating_mul(dst_bpi)
            .checked_add((copy_h - 1).saturating_mul(dst_bpr))
            .and_then(|v| v.checked_add(row_bytes))
        {
            Some(v) => v,
            None => return Err(br(BlitStatus::Bounds, "t2b_dst_span_overflow")),
        };
        if dst_span > dst.size {
            return Err(br(BlitStatus::Bounds, "t2b_dst_span_oob"));
        }
        let src_gva = match lt.base_gva.checked_add(src_off) {
            Some(v) => v,
            None => return Err(br(BlitStatus::Bounds, "t2b_src_gva_overflow")),
        };
        return copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            lt.row_stride,
            src_bpi,
            dst.gva,
            dst_bpr,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        );
    }
    let dst_span = match (copy_d - 1)
        .saturating_mul(dst_bpi)
        .checked_add((copy_h - 1).saturating_mul(dst_bpr))
        .and_then(|v| v.checked_add(row_bytes))
    {
        Some(v) => v,
        None => return Err(br(BlitStatus::Bounds, "t2b_stage_dst_span_overflow")),
    };
    if dst_span > dst.size {
        return Err(br(BlitStatus::Bounds, "t2b_stage_dst_span_oob"));
    }
    let dst_base = dst.gva;
    let allowed = dest_window(state, host, task_id, dst_base, dst_span);
    // `blit_rows_us` lives in `copy_row_region`, which only the linear-to-linear
    // fast path reaches. A texture-to-buffer copy stages every row through
    // `read_texture_row` instead, and for an IOSurface texture or IOSurface plane view source that
    // re-vouches the mapping's guest page table per row.
    let stage_rows_started = std::time::Instant::now();
    let mut row = vec![0u8; row_bytes as usize];
    for z in 0..copy_d {
        for y in 0..copy_h {
            read_texture_row(
                state,
                host,
                task_id,
                &src,
                Point {
                    x: ox,
                    y: oy,
                    z: oz + z,
                },
                y,
                row_bytes,
                &mut row,
            )?;
            let d = match dst_base
                .checked_add(z.saturating_mul(dst_bpi))
                .and_then(|b| b.checked_add(y.saturating_mul(dst_bpr)))
            {
                Some(v) => v,
                None => return Err(br(BlitStatus::Bounds, "t2b_stage_dst_gva_overflow")),
            };
            if gva_mem::write_task_gva_product_within(
                state,
                host,
                task_id,
                d,
                &row,
                allowed.as_ref(),
            )
            .is_err()
            {
                return Err(br(BlitStatus::GuestIo, "t2b_stage_write_io"));
            }
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_t2b_stage_us",
        stage_rows_started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route_n("blit_t2b_stage_rows", copy_h.saturating_mul(copy_d));
    Ok(())
}

fn resolved_object(
    state: &Device,
    task_id: u32,
    resource: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
) -> Result<u32, BlitStatus> {
    let Some((owner_task, object)) = state.task_objects.resources.owner(resource) else {
        return Err(br(BlitStatus::MissingResource, "resolved_resource_retired"));
    };
    if owner_task.get() != task_id {
        return Err(br(
            BlitStatus::MissingResource,
            "resolved_resource_task_mismatch",
        ));
    }
    Ok(object.get())
}

fn execute_resolved_texture_to_texture<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    operation: ResolvedTextureToTextureBlit,
) -> BlitStatus {
    // `walk_blit_us` says this call costs ~1.1 ms and `blit_t2t_bytes` says it
    // moves ~800 of them, so the cost is not in the copy and a single total
    // cannot say where it is instead. The three phases below are the whole body:
    // resolving both endpoints, arming the destination's page window, and the
    // row loop. Whichever of them holds the millisecond is the one to repair.
    let phase_started = std::time::Instant::now();
    let ResolvedTextureToTextureBlit {
        source,
        source_origin,
        destination,
        destination_origin,
        extent,
        aspect,
    } = operation;
    let destination_resource = destination.content.resource;
    let destination_object = match resolved_object(state, task_id, destination_resource) {
        Ok(object) => object,
        Err(status) => return status,
    };
    let src = source.backing;
    let dst = destination.backing;
    // Options apply to both ends; plane bpp must agree under the selected aspect.
    let Some(src_bpp) = pixel_format::blit_aspect_bytes_per_pixel(src.pixel_format(), aspect)
    else {
        return br(BlitStatus::Unsupported, "blit_options_aspect_format");
    };
    if src.pixel_format() != 0
        && dst.pixel_format() != 0
        && src.pixel_format() != dst.pixel_format()
    {
        return br(BlitStatus::Unsupported, "t2t_format_mismatch");
    }
    let copy_bpp = src_bpp;
    let repack_src = pixel_format::blit_aspect_needs_repack(src.pixel_format(), aspect);
    let repack_dst = pixel_format::blit_aspect_needs_repack(dst.pixel_format(), aspect);
    let any_iosurface = src.is_surface() || dst.is_surface();
    if any_iosurface && (source_origin.z != 0 || destination_origin.z != 0) {
        return br(BlitStatus::Unsupported, "t2t_iosurface_z");
    }
    let sox = source_origin.x;
    let soy = source_origin.y;
    let soz = source_origin.z;
    let dox = destination_origin.x;
    let doy = destination_origin.y;
    let doz = destination_origin.z;
    if sox > src.width() as u64
        || soy > src.height() as u64
        || soz > src.depth() as u64
        || dox > dst.width() as u64
        || doy > dst.height() as u64
        || doz > dst.depth() as u64
    {
        return br(BlitStatus::Bounds, "t2t_origin_oob");
    }
    // One region, checked against both textures. The source and destination
    // halves are separate `kind`s so a refusal says which end was too small,
    // and the destination half used to have no instrument at all — it did not
    // even come through the extent helper, which made it the quieter of two
    // quiet paths.
    let (Some(copy_w), Some(_)) = (
        copy_extent("t2t_src", "w", extent.width, src.width() as u64 - sox),
        copy_extent("t2t_dst", "w", extent.width, dst.width() as u64 - dox),
    ) else {
        return br(BlitStatus::Bounds, "t2t_extent_oob");
    };
    let (Some(copy_h), Some(_)) = (
        copy_extent("t2t_src", "h", extent.height, src.height() as u64 - soy),
        copy_extent("t2t_dst", "h", extent.height, dst.height() as u64 - doy),
    ) else {
        return br(BlitStatus::Bounds, "t2t_extent_oob");
    };
    let copy_d = if extent.depth == 0 {
        0
    } else {
        let (Some(d), Some(_)) = (
            copy_extent("t2t_src", "d", extent.depth, src.depth() as u64 - soz),
            copy_extent("t2t_dst", "d", extent.depth, dst.depth() as u64 - doz),
        ) else {
            return br(BlitStatus::Bounds, "t2t_extent_oob");
        };
        d
    };
    if any_iosurface && copy_d > 1 {
        return br(BlitStatus::Unsupported, "t2t_iosurface_volume");
    }
    if copy_w == 0 || copy_h == 0 || copy_d == 0 {
        return BlitStatus::ZeroExtent;
    }
    note_t2t_shape(&src, &dst, copy_w, copy_h, copy_d, copy_bpp);
    crate::runtime::drain::note_store_route_us(
        "blit_t2t_resolve_us",
        phase_started.elapsed().as_micros() as u64,
    );
    let row_bytes = match copy_w.checked_mul(copy_bpp as u64) {
        Some(v) => v,
        None => return br(BlitStatus::Capacity, "t2t_row_bytes_overflow"),
    };
    // Combined DS + aspect: extract plane from src, insert into dst (RMW).
    if repack_src || repack_dst {
        // The repack strides are the two textures' *storage* widths, which for a
        // combined depth/stencil format is wider than the aspect being copied.
        // Either side may still be format 0 here: the mismatch check above lets a
        // zero through, so a format-less texture can pair with a combined one and
        // only the combined side sets its `repack_*`. `bytes_per_pixel` cannot
        // answer for a zero, and the aspect's own `copy_bpp` is then a guess at a
        // stride this device is about to read guest bytes with. It stays the
        // guess — refusing would drop a copy that is correct whenever the two
        // widths agree — but it is a guess about guest data, so it says so.
        let src_storage = texture_storage_bpp(src.pixel_format()).unwrap_or_else(|_| {
            note_repack_storage_assumed(task_id, "src", src.pixel_format(), copy_bpp);
            copy_bpp
        });
        let dst_storage = texture_storage_bpp(dst.pixel_format()).unwrap_or_else(|_| {
            note_repack_storage_assumed(task_id, "dst", dst.pixel_format(), copy_bpp);
            copy_bpp
        });
        let plane_row = row_bytes as usize;
        let mut plane = vec![0u8; plane_row];
        let mut src_packed = vec![0u8; (copy_w as usize).saturating_mul(src_storage as usize)];
        let mut dst_packed = vec![0u8; (copy_w as usize).saturating_mul(dst_storage as usize)];
        let allowed = match texture_region_window(
            state,
            host,
            task_id,
            &dst,
            Point {
                x: dox,
                y: doy,
                z: doz,
            },
            copy_w as u32,
            copy_h,
            copy_d,
            // This loop has two write arms: the repack one stores a packed
            // `dst_storage` row, the other a `copy_bpp` plane row. The window
            // has to cover whichever runs, so it is measured on the wider.
            dst_storage.max(copy_bpp),
        ) {
            Ok(v) => v,
            Err(st) => return st,
        };
        for z in 0..copy_d {
            for y in 0..copy_h {
                if repack_src {
                    if let Err(st) = read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &src,
                        Point {
                            x: sox,
                            y: soy,
                            z: soz + z,
                        },
                        y,
                        copy_w as u32,
                        src_storage,
                        &mut src_packed,
                    ) {
                        return st;
                    }
                    if !pixel_format::extract_plane_row(
                        src.pixel_format(),
                        aspect,
                        &src_packed,
                        copy_w as u32,
                        &mut plane,
                    ) {
                        return br(BlitStatus::Unsupported, "t2t_extract_plane");
                    }
                } else if let Err(st) = read_texture_row(
                    state,
                    host,
                    task_id,
                    &src,
                    Point {
                        x: sox,
                        y: soy,
                        z: soz + z,
                    },
                    y,
                    row_bytes,
                    &mut plane,
                ) {
                    return st;
                }
                if repack_dst {
                    if let Err(st) = read_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &dst,
                        Point {
                            x: dox,
                            y: doy,
                            z: doz + z,
                        },
                        y,
                        copy_w as u32,
                        dst_storage,
                        &mut dst_packed,
                    ) {
                        return st;
                    }
                    if !pixel_format::insert_plane_row(
                        dst.pixel_format(),
                        aspect,
                        &plane,
                        copy_w as u32,
                        &mut dst_packed,
                    ) {
                        return br(BlitStatus::Unsupported, "t2t_insert_plane");
                    }
                    if let Err(st) = write_texture_storage_row(
                        state,
                        host,
                        task_id,
                        &dst,
                        Point {
                            x: dox,
                            y: doy,
                            z: doz + z,
                        },
                        y,
                        copy_w as u32,
                        dst_storage,
                        &dst_packed,
                        allowed.as_ref(),
                    ) {
                        return st;
                    }
                } else if let Err(st) = write_texture_row(
                    state,
                    host,
                    task_id,
                    &dst,
                    Point {
                        x: dox,
                        y: doy,
                        z: doz + z,
                    },
                    y,
                    row_bytes,
                    &plane,
                    allowed.as_ref(),
                ) {
                    return st;
                }
            }
        }
        return BlitStatus::Ok;
    }
    // Fast path: both linear → existing GVA span copy.
    if let (TextureBacking::Linear(ref sl), TextureBacking::Linear(ref dl)) = (&src, &dst) {
        let src_off = match sl.texel_offset(sox, soy, soz) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_src_texel_oob"),
        };
        let dst_off = match dl.texel_offset(dox, doy, doz) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_dst_texel_oob"),
        };
        let src_bpi = match sl.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "t2t_src_bpi_overflow"),
        };
        let dst_bpi = match dl.bytes_per_image() {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "t2t_dst_bpi_overflow"),
        };
        let src_gva = match sl.base_gva.checked_add(src_off) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_src_gva_overflow"),
        };
        let dst_gva = match dl.base_gva.checked_add(dst_off) {
            Some(v) => v,
            None => return br(BlitStatus::Bounds, "t2t_dst_gva_overflow"),
        };
        // Exact identity self-copy is a no-op: source and destination name the
        // same guest bytes with the same layout, so every row reads and writes
        // the same address — the destination already holds the source content.
        // Observed live (Ventura x86, media apps): the guest issues
        // copyFromTexture:X toTexture:X sourceOrigin==destinationOrigin on small
        // window textures (src_ref==dst_ref, src_off==dst_off). Copying bytes
        // onto themselves changes nothing, so succeed without work rather than
        // rejecting it as Overlap (which returned a spurious error to the guest
        // blit encoder and dropped a copy the guest treats as complete). A
        // genuinely-shifted overlap falls through to source-before-destination
        // staging below.
        if src_gva == dst_gva && sl.row_stride == dl.row_stride && src_bpi == dst_bpi {
            return BlitStatus::Ok;
        }
        // Same allocation (self-copy or aliased view) but a different region:
        // direct row-by-row execution is safe only when the source and
        // destination texel rectangles do not intersect. Two axis-aligned
        // rectangles overlap iff they overlap on every axis. If the layouts
        // differ, their texel grids are incomparable, so the byte spans give
        // the conservative answer.
        if sl.base_gva == dl.base_gva {
            let same_layout = sl.row_stride == dl.row_stride && src_bpi == dst_bpi;
            let overlaps = if same_layout {
                let x = sox < dox + copy_w && dox < sox + copy_w;
                let y = soy < doy + copy_h && doy < soy + copy_h;
                let z = soz < doz + copy_d && doz < soz + copy_d;
                x && y && z
            } else {
                let s_end =
                    src_off.saturating_add(row_bytes.saturating_mul(copy_h).saturating_mul(copy_d));
                let d_end =
                    dst_off.saturating_add(row_bytes.saturating_mul(copy_h).saturating_mul(copy_d));
                ranges_overlap(
                    src_off,
                    s_end.saturating_sub(src_off),
                    dst_off,
                    d_end.saturating_sub(dst_off),
                )
            };
            if overlaps {
                // The serialized copy record carries the complete source and
                // destination regions and does not define an overlap refusal.
                // Snapshot every source plane before writing any destination
                // plane: a row-at-a-time loop can overwrite bytes a later row
                // or depth plane still has to read.
                let plane_bytes = match row_bytes.checked_mul(copy_h) {
                    Some(v) => v,
                    None => return br(BlitStatus::Capacity, "t2t_overlap_plane_overflow"),
                };
                let total_bytes = match plane_bytes.checked_mul(copy_d) {
                    Some(v) => v,
                    None => return br(BlitStatus::Capacity, "t2t_overlap_total_overflow"),
                };
                let Some(total_len) = host_alloc_len(total_bytes) else {
                    return br(BlitStatus::Capacity, "t2t_overlap_alloc");
                };
                let allowed = match texture_region_window(
                    state,
                    host,
                    task_id,
                    &dst,
                    Point {
                        x: dox,
                        y: doy,
                        z: doz,
                    },
                    copy_w as u32,
                    copy_h,
                    copy_d,
                    copy_bpp,
                ) {
                    Ok(v) => v,
                    Err(st) => return st,
                };
                let mut staged = vec![0u8; total_len];
                for z in 0..copy_d {
                    let start = (z * plane_bytes) as usize;
                    let end = start + plane_bytes as usize;
                    if let Err(st) = read_texture_rect(
                        state,
                        host,
                        task_id,
                        &src,
                        Point {
                            x: sox,
                            y: soy,
                            z: soz + z,
                        },
                        row_bytes,
                        copy_h,
                        &mut staged[start..end],
                    ) {
                        return st;
                    }
                }
                for z in 0..copy_d {
                    let start = (z * plane_bytes) as usize;
                    let end = start + plane_bytes as usize;
                    if let Err(st) = write_texture_rect(
                        state,
                        host,
                        task_id,
                        &dst,
                        Point {
                            x: dox,
                            y: doy,
                            z: doz + z,
                        },
                        row_bytes,
                        copy_h,
                        &staged[start..end],
                        allowed.as_ref(),
                    ) {
                        return st;
                    }
                }
                crate::runtime::drain::note_store_route("blit_t2t_overlap_staged");
                crate::runtime::drain::note_store_route_n(
                    "blit_t2t_overlap_staged_bytes",
                    total_bytes,
                );
                return BlitStatus::Ok;
            }
        }
        return match copy_row_region(
            host,
            state,
            task_id,
            src_gva,
            sl.row_stride,
            src_bpi,
            dst_gva,
            dl.row_stride,
            dst_bpi,
            row_bytes,
            copy_h,
            copy_d,
        ) {
            Ok(()) => BlitStatus::Ok,
            Err(st) => st,
        };
    }
    // Mixed or IOSurface texture↔IOSurface texture: stage rows.
    let allowed = match texture_region_window(
        state,
        host,
        task_id,
        &dst,
        Point {
            x: dox,
            y: doy,
            z: doz,
        },
        copy_w as u32,
        copy_h,
        copy_d,
        copy_bpp,
    ) {
        Ok(v) => v,
        Err(st) => return st,
    };
    // The last untimed loop on the rail: a texture-to-texture copy with a
    // IOSurface texture or IOSurface plane view end on either side stages through here rather than
    // through `copy_row_region`, so `blit_rows_us` reports nothing for it.
    //
    // A plane at a time, not a row at a time: this is the same staging shape the
    // slice/level form carried, and there a driven Maps leg charged the row loop
    // 30.15 s of a 30.28 s rail. See [`read_texture_rect`] for what a per-row
    // call into the mapping rail re-pays.
    // Whether a GPU-side copy serves this pair, and if not, which term stops it.
    // `engine::copy_target_to_guest_pages` takes no source rectangle: it copies
    // level 0 of the resident whole, at origin zero, into a destination whose
    // geometry is the resident's own. So an IOSurface texture source going to a linear
    // destination is reachable only when both ends are the whole plane at the
    // origin, and the three counters partition the population so a reading says
    // how much of it that is. See [`try_copy_iosurface_plane_to_linear_on_gpu`] for
    // what the arm below is instead of, which is the settle the staging loop
    // pays to make the source's guest bytes readable.
    if src.is_surface() && !dst.is_surface() {
        let whole_src =
            sox == 0 && soy == 0 && copy_w == src.width() as u64 && copy_h == src.height() as u64;
        let whole_dst =
            dox == 0 && doy == 0 && copy_w == dst.width() as u64 && copy_h == dst.height() as u64;
        crate::runtime::drain::note_store_route(match (whole_src, whole_dst) {
            (true, true) => "blit_t2t_iosurface_whole_plane",
            (true, false) => "blit_t2t_iosurface_dst_partial",
            (false, _) => "blit_t2t_iosurface_src_partial",
        });
        if whole_src && whole_dst {
            if let (TextureBacking::Surface(s), TextureBacking::Linear(d)) = (&src, &dst) {
                if let Some(status) = try_copy_iosurface_plane_to_linear_on_gpu(
                    state,
                    host,
                    task_id,
                    destination_object,
                    s,
                    d,
                ) {
                    return status;
                }
            }
        }
    }
    let t2t_stage_started = std::time::Instant::now();
    let mut staged = vec![0u8; row_bytes.saturating_mul(copy_h) as usize];
    for z in 0..copy_d {
        if let Err(st) = read_texture_rect(
            state,
            host,
            task_id,
            &src,
            Point {
                x: sox,
                y: soy,
                z: soz + z,
            },
            row_bytes,
            copy_h,
            &mut staged,
        ) {
            return st;
        }
        if let Err(st) = write_texture_rect(
            state,
            host,
            task_id,
            &dst,
            Point {
                x: dox,
                y: doy,
                z: doz + z,
            },
            row_bytes,
            copy_h,
            &staged,
            allowed.as_ref(),
        ) {
            return st;
        }
    }
    crate::runtime::drain::note_store_route_us(
        "blit_t2t_stage_us",
        t2t_stage_started.elapsed().as_micros() as u64,
    );
    crate::runtime::drain::note_store_route_n("blit_t2t_stage_rows", copy_h.saturating_mul(copy_d));
    BlitStatus::Ok
}

/// `0x13e copyFromTexture:…sliceCount:levelCount:` — whole-surface multi-slice/level.
///
/// For each level offset in `0..level_count`:
/// - **depth == 1:** copies full `width×height` across `slice_count` consecutive
///   array slices (`origin (0,0,0)`, size `w×h×1`).
/// - **depth > 1 (3D volume):** Metal requires `sliceCount == 1` and source/
///   destination slices 0; copies full `width×height×depth` of that mip with
///   depth planes strided by `bytes_per_image`. Linear type-2/3 only.
///
/// One resolved blit endpoint, reduced to what the GPU whole-plane arm reads.
///
/// A [`TextureBacking`] says where a texture's *guest bytes* are, which is the
/// question every other consumer in this module asks. This is the other one:
/// which plane of which surface a GPU-side copy would land in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuPlane {
    width: u32,
    height: u32,
    /// Byte offset of this texture's plane within its mapping's allocation.
    surface_offset: u64,
    row_stride: u32,
    pixel_format: u16,
}

/// The plane `mapping_write::write_bgra8_from_resident_gpu` will address, from
/// the destination mapping's own declaration rather than from the texture
/// descriptor the blit named.
///
/// The two are independent derivations of one plane and the whole safety of the
/// GPU arm is that they agree. See [`mapping_write::resident_gpu_plane`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuMappingWindow {
    surface_offset: u64,
    row_stride: u32,
    pixel_format: u16,
}

/// The source's real content, as the engine holds it behind an armed
/// [`crate::runtime::writeback_debt::GvaWritebackDebt`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GpuResidentSource {
    width: u32,
    height: u32,
    pixel_format: u16,
}

/// Why one whole-surface texture-to-texture copy is not the GPU arm's.
///
/// Every variant is a **fall-through and not a loss**: the host path below runs
/// unchanged and lands the same pixels. They are counters rather than fail-log
/// records for that reason, and they partition the whole-surface population with
/// `sl_gpu_landed`, so a census that does not add up is the bug.
///
/// There is deliberately no partial-rect variant. `0x13e` is the *whole-surface*
/// form: its origins are (0,0,0) and its extent is the endpoints' full
/// width/height by construction of the opcode, so a rect check here would be an
/// arm no guest command can take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuPlaneRefusal {
    /// More than one level or slice: the GPU arm copies one plane.
    MultiLevel,
    /// Source and destination are the same reference, so resolving the
    /// destination would pay away the very debt holding the source's content.
    SelfCopy,
    /// The source's bytes are its guest pages' bytes already — nothing to copy
    /// from a resident, and the host path is the cheap one.
    ///
    /// **Says which bytes the host path names, and nothing about when it reads
    /// them.** This arm is selected by the target-import rail: a source the
    /// device rendered into directly owes no writeback debt, so there is no
    /// resident to copy from and the fall-through is a `memcpy` over pages a
    /// submitted render Store may still be writing. Reading that as "not a
    /// loss" is what hid a whole missing layer — `gva_view`'s reads settle
    /// against outstanding guest writes now, and 716 of 716 on one driven boot
    /// genuinely overlapped. The variant is still a fall-through and still not
    /// a loss; the ordering that makes that true lives in the read.
    SrcNotResident,
    /// The destination is a linear guest allocation, which has no mapping for a
    /// GPU-side copy to name.
    DstNotIOSurface,
    /// The destination mapping declines a resident-to-guest-pages copy at this
    /// extent, so there is no window to write.
    DstWindowUnresolved,
    /// The two derivations of the destination plane disagree. A IOSurface plane view view's
    /// wire plane index lands here: it can name a plane the mapping's own
    /// geometry scan does not resolve to, and the GPU rail takes no index.
    PlaneOffset,
    /// The resident is not the destination's size, so a full-plane copy would
    /// be a resize.
    GeometryDiffers,
    /// A copy converts nothing, so all three of source, destination and mapping
    /// must already agree on the texel.
    FormatDiffers,
}

impl GpuPlaneRefusal {
    fn route(self) -> &'static str {
        match self {
            Self::MultiLevel => "sl_gpu_multi_level",
            Self::SelfCopy => "sl_gpu_self_copy",
            Self::SrcNotResident => "sl_gpu_src_not_resident",
            Self::DstNotIOSurface => "sl_gpu_dst_not_iosurface",
            Self::DstWindowUnresolved => "sl_gpu_dst_window",
            Self::PlaneOffset => "sl_gpu_plane_offset",
            Self::GeometryDiffers => "sl_gpu_geometry_differs",
            Self::FormatDiffers => "sl_gpu_format_differs",
        }
    }
}

/// Everything the GPU arm can decide before resolving anything.
///
/// Split from [`gpu_whole_plane_destination`] because resolving the destination
/// is the expensive half and it is also the half with the side effect: it pays
/// the destination's own debt. This decides whether that is worth doing.
fn gpu_whole_plane_admissible(
    level_count: u16,
    slice_count: u16,
    source_ref: u32,
    destination_ref: u32,
    source_is_resident: bool,
) -> Result<(), GpuPlaneRefusal> {
    if level_count != 1 || slice_count != 1 {
        return Err(GpuPlaneRefusal::MultiLevel);
    }
    if source_ref == destination_ref {
        return Err(GpuPlaneRefusal::SelfCopy);
    }
    if !source_is_resident {
        return Err(GpuPlaneRefusal::SrcNotResident);
    }
    Ok(())
}

/// Whether the resolved destination is the plane the GPU rail will actually
/// write, and whether the resident is a whole-plane copy into it.
///
/// `dst` is `None` for a linear endpoint and `window` is `None` when the mapping
/// declines the extent, so the two `Option`s are the caller's two resolution
/// steps and not defensive wrapping.
fn gpu_whole_plane_destination(
    dst: Option<GpuPlane>,
    window: Option<GpuMappingWindow>,
    src: GpuResidentSource,
) -> Result<(), GpuPlaneRefusal> {
    let Some(dst) = dst else {
        return Err(GpuPlaneRefusal::DstNotIOSurface);
    };
    let Some(window) = window else {
        return Err(GpuPlaneRefusal::DstWindowUnresolved);
    };
    // The plane the guest's descriptor named against the plane the rail will
    // write. `mapping_write/mod.rs`'s own record of a bound error landing in the
    // next plane's pixels of a multi-plane IOSurface is what this is here for:
    // that failure is silent at every layer, so it has to be refused before the
    // copy rather than detected after it.
    if window.surface_offset != dst.surface_offset || window.row_stride != dst.row_stride {
        return Err(GpuPlaneRefusal::PlaneOffset);
    }
    if src.width != dst.width || src.height != dst.height {
        return Err(GpuPlaneRefusal::GeometryDiffers);
    }
    // All three, not two. The engine refuses a resident whose texel is not the
    // destination's (`copy_target_to_guest_pages`), and the mapping's declared
    // format is what the guest will read these bytes back as — a chain that
    // agrees pairwise but not as a whole would land a converted-looking frame
    // with nothing having converted anything.
    //
    // **Stored texels, not declared formats.** A copy converts nothing, so what
    // has to agree is what the bytes *are*, and the transfer function is not part
    // of that: it says how a sampler reads a texel, not how one is stored.
    // `store_texel_order` is this crate's single answer to that question and its
    // own doc states the fold ("only the storage matters to a copy"), which is
    // also what `translate::pixel::stored_bytes_agree` encodes one layer down for
    // the Vulkan spelling of the same comparison.
    //
    // Equality here is not a stricter version of that rule, it is a different and
    // wrong one, and it cost this arm every record it was written for: a guest
    // render target declared `BGRA8Unorm_sRGB` meets an IOSurface mapping that
    // declares plain `BGRA8Unorm` for the same four stored bytes, so the triple
    // reads 81/81/80 and equality calls that a disagreement forever. A `None`
    // still refuses — a format with no byte-copy layout is one where the copy
    // would have to convert, which this arm does not do.
    let texel = reims_vgpu_core::pixel_format::store_texel_order(dst.pixel_format);
    if texel.is_none()
        || reims_vgpu_core::pixel_format::store_texel_order(src.pixel_format) != texel
        || reims_vgpu_core::pixel_format::store_texel_order(window.pixel_format) != texel
    {
        return Err(GpuPlaneRefusal::FormatDiffers);
    }
    Ok(())
}

/// Land the source's engine resident straight into the destination's guest pages
/// with the GPU, for the one shape where that is exactly the copy the guest asked
/// for.
///
/// # Why a blit is not a guest-byte reader here
///
/// `resolve_texture_backing` pays every endpoint's writeback debt, because its
/// answer is "where are this texture's guest bytes" and guest bytes are only a
/// resource's content once everything rendered into it has landed. That is right
/// for every endpoint the host row loops read or write. It is wrong for *this*
/// shape, and expensively so: a whole-plane copy out of a resident makes the
/// device read the resident back into the source's guest pages, then memcpy those
/// pages into the destination's — two crossings of a frame to move content the
/// GPU already holds.
///
/// Planning resolves the source's immutable storage geometry, but this arm
/// **never pays the source's debt**.
/// The source's own guest pages stay stale and stay owed; the debt stays armed,
/// and the next genuine guest-byte reader — a sample, a compute bind, a
/// `CmdSynchronizeResources` — is what lands them. That is what the Metal
/// contract says: `copyFromTexture:toTexture:` is a blit-encoder command with no
/// host visibility, and `synchronizeResource:` is the separate call that means
/// "make this CPU-visible".
///
/// The *destination*'s debt is still paid before this policy runs, and must be:
/// leaving it armed would let a pre-blit resident land over this copy's bytes
/// later.
///
/// Returns `None` for every fall-through, having named it on a counter. The
/// caller then runs the host path unchanged, so nothing here can lose a frame —
/// only spend one.
struct WholePlaneGpuCopy<'a> {
    source_object: u32,
    destination_object: u32,
    level_count: u16,
    slice_count: u16,
    destination: &'a TextureBacking,
}

fn try_copy_whole_plane_on_gpu<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    copy: WholePlaneGpuCopy<'_>,
) -> Option<BlitStatus> {
    use crate::runtime::drain::note_store_route;
    let WholePlaneGpuCopy {
        source_object,
        destination_object,
        level_count,
        slice_count,
        destination,
    } = copy;
    let key = crate::runtime::writeback_debt::resource_key(state, task_id, source_object)?;
    let debt = state.content.pending_writebacks.get_gva(key);
    if let Err(refusal) = gpu_whole_plane_admissible(
        level_count,
        slice_count,
        source_object,
        destination_object,
        debt.is_some(),
    ) {
        note_store_route(refusal.route());
        return None;
    }
    let debt = debt?;
    let TextureBacking::Surface(t) = destination else {
        note_store_route(GpuPlaneRefusal::DstNotIOSurface.route());
        return None;
    };
    let plane = GpuPlane {
        width: t.width,
        height: t.height,
        surface_offset: t.surface_offset,
        row_stride: t.row_stride,
        pixel_format: t.pixel_format,
    };
    let mapping_id = t.mapping_id.get();
    let window = state
        .surfaces
        .mappings
        .get(&mapping_id)
        .and_then(|m| mapping_write::resident_gpu_plane(m, plane.width, plane.height))
        .map(
            |(surface_offset, row_stride, pixel_format)| GpuMappingWindow {
                surface_offset,
                row_stride,
                pixel_format,
            },
        );
    let src = GpuResidentSource {
        width: debt.width,
        height: debt.height,
        pixel_format: debt.format,
    };
    if let Err(refusal) = gpu_whole_plane_destination(Some(plane), window, src) {
        note_store_route(refusal.route());
        // "The formats differ" does not say which of the three does, and the
        // three have different answers: a source that disagrees is a converting
        // copy the contract does not describe, while a *mapping* that disagrees
        // with a destination the source already matches is this device's own
        // declaration being narrower than the texture it describes. Name all
        // three once per distinct triple so the reading is the diagnosis.
        if refusal == GpuPlaneRefusal::FormatDiffers {
            let discriminant = (u64::from(src.pixel_format) << 32)
                | (u64::from(plane.pixel_format) << 16)
                | u64::from(window.map_or(u16::MAX, |w| w.pixel_format));
            if crate::observe::first_sight("sl_gpu_format_differs", discriminant) {
                crate::observe::fail(format!(
                    "blit_gpu_plane reason=sl_gpu_format_differs src_format={} \
                     dst_format={} mapping_format={} width={} height={}",
                    src.pixel_format,
                    plane.pixel_format,
                    window.map_or(u16::MAX, |w| w.pixel_format),
                    plane.width,
                    plane.height
                ));
            }
        }
        return None;
    }
    let identity = crate::runtime::writeback_debt::gva_identity(debt);
    match mapping_write::write_bgra8_from_resident_gpu(
        state,
        host,
        mapping_id,
        &identity,
        plane.width,
        plane.height,
    ) {
        Ok(_) => {
            note_store_route("sl_gpu_landed");
            Some(BlitStatus::Ok)
        }
        Err(decline) => {
            // A decline is a routing answer, so the counter is the record and the
            // off channel carries which check answered — the engine's format
            // comparison in particular, which is the one that can refuse every
            // payment on one texture and leave a canvas black.
            note_store_route("sl_gpu_engine_declined");
            crate::observe::off(format!(
                "blit_gpu_plane mid={mapping_id} {}x{} decline={decline:?}",
                plane.width, plane.height
            ));
            None
        }
    }
}

/// Why one whole-plane IOSurface texture to guest-linear copy is not the GPU arm's, for
/// the terms that can be decided from the two endpoints alone.
///
/// Every variant is a **fall-through and not a loss**: the staging loop runs
/// unchanged and lands the same pixels. They are counters for that reason, and
/// with `t2t_gpu_src_not_resident`, `t2t_gpu_dst_unbounded`,
/// `t2t_gpu_engine_declined` and `t2t_gpu_landed` they partition
/// `blit_t2t_iosurface_whole_plane`, so a census that does not add up is the bug.
#[derive(Clone, Debug, PartialEq, Eq)]
enum T2tGvaRefusal {
    /// The source names no mapping this device holds, or one that has not
    /// declared its geometry, so there is no surface identity to ask about.
    NoSurface,
    /// The source is a plane of a larger surface, or disagrees with the mapping
    /// about the surface's size. The resident is keyed by the mapping's own
    /// geometry and this copy lands it whole, so anything but the whole surface
    /// would land it into a window that is not the whole of it.
    SrcNotWholeSurface,
    /// The destination level's own base does not resolve.
    DstOffsetOverflow,
    /// The destination's pitch does not fit the guest's own 32-bit declaration
    /// of one.
    DstStrideWide,
    /// The destination has no byte-copy geometry — the plane's own typed
    /// reason, carried whole so a reading names the same check the copy would
    /// have named.
    DstPlane(crate::runtime::render_writeback::GvaWritebackDecline),
    /// The plane runs past the allocation the level lives in.
    DstExtentOob,
}

impl T2tGvaRefusal {
    fn route(&self) -> &'static str {
        match self {
            Self::NoSurface => "t2t_gpu_no_surface",
            Self::SrcNotWholeSurface => "t2t_gpu_src_not_whole_surface",
            Self::DstOffsetOverflow => "t2t_gpu_dst_offset_overflow",
            Self::DstStrideWide => "t2t_gpu_dst_stride_wide",
            Self::DstPlane(_) => "t2t_gpu_dst_plane",
            Self::DstExtentOob => "t2t_gpu_dst_extent_oob",
        }
    }
}

/// The destination plane a whole-plane IOSurface texture to guest-linear copy would write,
/// and its span, or the typed reason there is none.
///
/// Everything [`try_copy_iosurface_plane_to_linear_on_gpu`] can decide before it asks
/// the engine anything or walks the guest's page table, which is also everything
/// about it that a test can reach without a GPU. `surface` is the mapping's own
/// declared geometry and `None` when it has none.
fn gpu_t2t_gva_plane(
    surface: Option<(u32, u32)>,
    src: &IOSurfaceTextureBacking,
    dst: &LinearTextureLevel,
    destination_ref: u32,
) -> Result<
    (
        crate::runtime::render_writeback::GvaPlaneDestination,
        crate::runtime::render_writeback::GvaPlaneGeometry,
    ),
    T2tGvaRefusal,
> {
    let Some((sw, sh)) = surface else {
        return Err(T2tGvaRefusal::NoSurface);
    };
    if sw != src.width || sh != src.height || src.surface_offset != 0 || sw == 0 || sh == 0 {
        return Err(T2tGvaRefusal::SrcNotWholeSurface);
    }
    // The destination plane, from the level the blit resolved. Origin zero on
    // both ends is the caller's admission, so this is the level's own base.
    let Some(level_base) = dst.texel_offset(0, 0, 0) else {
        return Err(T2tGvaRefusal::DstOffsetOverflow);
    };
    let Some(target_gva) = dst.base_gva.checked_add(level_base) else {
        return Err(T2tGvaRefusal::DstOffsetOverflow);
    };
    let Ok(row_stride) = u32::try_from(dst.row_stride) else {
        return Err(T2tGvaRefusal::DstStrideWide);
    };
    let plane = crate::runtime::render_writeback::GvaPlaneDestination {
        target_gva,
        width: dst.width,
        height: dst.height,
        row_stride,
        format: dst.pixel_format,
        texture_ref: destination_ref,
    };
    // The span to walk, from the destination's own terms, so the licence covers
    // exactly the bytes the copy writes and not one page more.
    let geometry = plane.geometry().map_err(T2tGvaRefusal::DstPlane)?;
    // Against the allocation and not against the span: a copy that runs off the
    // level's own bytes is the class `texture_region_window` bounds the host
    // path with, and this arm owes the same check before it walks anything.
    if !range_fits(level_base, geometry.extent, dst.alloc_size) {
        return Err(T2tGvaRefusal::DstExtentOob);
    }
    Ok((plane, geometry))
}

/// The whole-plane copy out of an IOSurface the GPU already holds, into a
/// guest-linear destination, issued by the GPU.
///
/// # What this is instead of
///
/// The host path below reads the source through the mapping rail and writes the
/// destination through the GVA rail. Reading the source's *guest bytes* is what
/// makes it expensive, and the cost is not the copy: a mapping read must first
/// settle, which pays the surface's writeback debt and then waits for this
/// device's own submitted writes to land in those pages. On a driven macos-13
/// x86 boot that settle was 91 % of the staging window and the memcpy behind it
/// was 4.5 %.
///
/// None of it is owed here. The source's authoritative content is the engine's
/// resident, the destination is a plane of guest pages, and
/// `engine::copy_target_to_guest_pages` moves exactly that — so this arm never
/// touches the source's guest bytes and has nothing to wait for. What the guest
/// asked for is `copyFromTexture:toTexture:`, a blit-encoder command with no
/// host visibility; making the source CPU-readable is `synchronizeResource:`,
/// which is a different call the guest did not make.
///
/// # Why it is only the whole plane
///
/// `copy_target_to_guest_pages` takes no source rectangle: it copies level 0 of
/// the resident whole, at origin zero. So a partial rect on either end is not
/// this arm's, and the caller's census — which partitions the population — is
/// what says how much of it that leaves. On the boot above it left all of it:
/// 511 of 511.
///
/// Returns `None` for every fall-through, having named it on a counter. The
/// caller then runs the host path unchanged, so nothing here can lose a frame —
/// only spend one.
fn try_copy_iosurface_plane_to_linear_on_gpu<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    destination_ref: u32,
    src: &IOSurfaceTextureBacking,
    dst: &LinearTextureLevel,
) -> Option<BlitStatus> {
    use crate::runtime::drain::note_store_route;
    let surface = state
        .surfaces
        .mappings
        .get(&src.mapping_id.get())
        .filter(|m| m.has_geometry())
        .map(|m| (m.width_or_zero(), m.height_or_zero()));
    let (plane, geometry) = match gpu_t2t_gva_plane(surface, src, dst, destination_ref) {
        Ok(v) => v,
        Err(refusal) => {
            note_store_route(refusal.route());
            return None;
        }
    };
    let identity = crate::runtime::present_identity::surface_identity(
        state,
        src.mapping_id.get(),
        src.width,
        src.height,
    );
    if matches!(
        state.executor.resident_read_plan(&identity).backing,
        reims_vgpu_core::ResidentContentBacking::NotReady
    ) {
        // The source's bytes are its guest pages' bytes already, so the host
        // path is reading what it should and is the cheap arm rather than the
        // wasteful one.
        note_store_route("t2t_gpu_src_not_resident");
        return None;
    }
    // The destination's pages, captured once. The host path's `dest_window`
    // takes the same walk for the same reason — the guest's vCPUs run
    // throughout, so the licence must be the walk the command itself was
    // authorised by rather than whatever the address names later.
    let gpas = gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        plane.target_gva,
        geometry.extent,
        state.page_shift,
    );
    if gpas.is_empty() {
        note_store_route("t2t_gpu_dst_unbounded");
        return None;
    }
    let pages = crate::runtime::draw::StoreTargetPages::from_ordered(&gpas, geometry.extent);
    match crate::runtime::render_writeback::copy_resident_into_gva_plane(
        state,
        host,
        task_id,
        &identity,
        &plane,
        Some(&pages),
    ) {
        Ok(_) => {
            note_store_route("t2t_gpu_landed");
            Some(BlitStatus::Ok)
        }
        Err(decline) => {
            note_store_route("t2t_gpu_engine_declined");
            crate::observe::off(format!(
                "blit_gpu_gva mid={} {}x{} decline={decline:?}",
                src.mapping_id.get(),
                dst.width,
                dst.height
            ));
            None
        }
    }
}

/// Execute one non-empty, fully resolved multi-slice/multi-level copy.
fn execute_resolved_texture_copy_batch<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    operation: ResolvedTextureCopyBatch,
) -> BlitStatus {
    let ResolvedTextureCopyBatch {
        source_base_slice,
        destination_base_slice,
        first_level,
        remaining_levels,
    } = operation;
    let source_resource = first_level.first_slice.0.content.resource;
    let destination_resource = first_level.first_slice.1.content.resource;
    let source_object = match resolved_object(state, task_id, source_resource) {
        Ok(object) => object,
        Err(status) => return status,
    };
    let destination_object = match resolved_object(state, task_id, destination_resource) {
        Ok(object) => object,
        Err(status) => return status,
    };
    let level_count = u16::try_from(1usize.saturating_add(remaining_levels.len()))
        .expect("wire level count fits u16");
    let slice_count = u16::try_from(1usize.saturating_add(first_level.remaining_slices.len()))
        .expect("wire slice count fits u16");

    // A partial destination overwrite must preserve bytes outside the copied
    // region before either execution policy runs.
    note_blit_endpoint_debt(state, task_id, destination_object);
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, destination_object);
    if let Some(status) = try_copy_whole_plane_on_gpu(
        state,
        host,
        task_id,
        WholePlaneGpuCopy {
            source_object,
            destination_object,
            level_count,
            slice_count,
            destination: &first_level.first_slice.1.backing,
        },
    ) {
        return status;
    }
    note_blit_endpoint_debt(state, task_id, source_object);
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, source_object);
    for level in std::iter::once(&first_level).chain(remaining_levels.iter()) {
        // `blit_kind_t2t_sl_us` charges this function 28.8 s of a 29.1 s rail
        // while `blit_rows_us` — its linear arm's whole copy — reads 0.275 s.
        // Between those two numbers sit the resolves below, and this form runs
        // them once per level and again per slice, so their count is the
        // multiplier nobody has measured. `sl_levels_n` is the denominator that
        // says whether a level loop of two or of twelve is being paid for.
        crate::runtime::drain::note_store_route("sl_levels_n");
        let src0 = &level.first_slice.0.backing;
        let dst0 = &level.first_slice.1.backing;
        if src0.bpp() != dst0.bpp() {
            return br(BlitStatus::Unsupported, "sl_bpp_mismatch");
        }
        if src0.pixel_format() != 0
            && dst0.pixel_format() != 0
            && src0.pixel_format() != dst0.pixel_format()
        {
            return br(BlitStatus::Unsupported, "sl_format_mismatch");
        }
        if src0.width() != dst0.width() || src0.height() != dst0.height() {
            return br(BlitStatus::Bounds, "sl_dim_mismatch");
        }
        if src0.depth() != dst0.depth() {
            return br(BlitStatus::Bounds, "sl_depth_mismatch");
        }
        let w = src0.width();
        let h = src0.height();
        let d = src0.depth();
        if w == 0 || h == 0 || d == 0 {
            return br(BlitStatus::Bounds, "sl_zero_geom");
        }
        let is_volume = d > 1;
        // Metal 3D whole-surface: sliceCount must be 1, slices 0; full depth of mip.
        if is_volume {
            if slice_count != 1 || source_base_slice != 0 || destination_base_slice != 0 {
                return br(BlitStatus::Unsupported, "sl_volume_slice_constraint");
            }
            // IOSurface texture is 2D (depth 1); volume endpoints are linear only.
            if src0.is_surface() || dst0.is_surface() {
                return br(BlitStatus::Unsupported, "sl_volume_iosurface");
            }
        }
        let bpp = src0.bpp();
        let row_bytes = match (w as u64).checked_mul(bpp as u64) {
            Some(v) => v,
            None => return br(BlitStatus::Capacity, "sl_row_bytes_overflow"),
        };

        // Linear: multi-slice (depth-1) or full volume (depth>1).
        if let (TextureBacking::Linear(sl), TextureBacking::Linear(dl)) = (src0, dst0) {
            if !is_volume && slice_count > 1 && (sl.slice_stride == 0 || dl.slice_stride == 0) {
                return br(BlitStatus::Unsupported, "sl_slice_stride_zero");
            }
            let src_off = match sl.texel_offset(0, 0, 0) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_src_texel_oob"),
            };
            let dst_off = match dl.texel_offset(0, 0, 0) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_dst_texel_oob"),
            };
            let src_gva = match sl.base_gva.checked_add(src_off) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_src_gva_overflow"),
            };
            let dst_gva = match dl.base_gva.checked_add(dst_off) {
                Some(v) => v,
                None => return br(BlitStatus::Bounds, "sl_dst_gva_overflow"),
            };
            // Volume: image_count = depth, stride = bytes_per_image (z planes).
            // Array: image_count = slice_count, stride = slice_stride when multi.
            let (src_img_stride, dst_img_stride, image_count) = if is_volume {
                let src_bpi = match sl.bytes_per_image() {
                    Some(v) if v > 0 => v,
                    _ => return br(BlitStatus::Bounds, "sl_src_bpi_zero"),
                };
                let dst_bpi = match dl.bytes_per_image() {
                    Some(v) if v > 0 => v,
                    _ => return br(BlitStatus::Bounds, "sl_dst_bpi_zero"),
                };
                (src_bpi, dst_bpi, d as u64)
            } else if slice_count <= 1 {
                // One image, so neither stride is ever stepped: both consumers
                // scale it by `image_count - 1`, which is zero here —
                // `strided_span`'s `last_image` term and `copy_region`'s `z`
                // loop. That is why this arm may fall back on `row_bytes` where
                // the volume arm above refuses a missing `bytes_per_image` by
                // name: there the stride selects z planes and a wrong one reads
                // the wrong plane, here it is multiplied away. The fallback is
                // inert rather than a second opinion about the layout.
                (
                    sl.bytes_per_image().unwrap_or(row_bytes),
                    dl.bytes_per_image().unwrap_or(row_bytes),
                    1u64,
                )
            } else {
                (sl.slice_stride, dl.slice_stride, u64::from(slice_count))
            };
            // Same allocation overlap check (conservative).
            if sl.base_gva == dl.base_gva {
                let span = row_bytes
                    .saturating_mul(h as u64)
                    .saturating_mul(image_count);
                if ranges_overlap(src_off, span, dst_off, span) {
                    return br(BlitStatus::Overlap, "sl_overlap");
                }
            }
            if let Err(st) = copy_row_region(
                host,
                state,
                task_id,
                src_gva,
                sl.row_stride,
                src_img_stride,
                dst_gva,
                dl.row_stride,
                dst_img_stride,
                row_bytes,
                h as u64,
                image_count,
            ) {
                return st;
            }
            continue;
        }

        // IOSurface texture / mixed: depth-1 only (IOSurface texture is 2D); per-slice whole-surface.
        if is_volume {
            return br(BlitStatus::Unsupported, "sl_volume_mixed");
        }
        // The slice/level form's IOSurface texture arm. It used to stage one row at a
        // time, and a driven Maps leg charged that loop 30.15 s of a 30.28 s
        // blit rail to move 14.6 MB, against 0.12 s for the resolves beside it
        // and 0.22 s for every strided guest-RAM copy in the device. The bytes
        // were never the cost — re-entering the mapping rail per row was. It
        // stages the slice whole now; see [`read_texture_rect`].
        let sl_mixed_started = std::time::Instant::now();
        // This loop is a whole-surface **CPU** copy, and it should not exist:
        // `copyFromTexture:...toTexture:` is a blit-encoder command, so the
        // contract puts this copy on the GPU queue, where ordering against
        // earlier GPU writes is free and no host byte moves.
        //
        // What it costs, from one driven fullscreen Maps boot: `sl_mixed_bytes`
        // 7.57 GB staged in 45 s over 968 levels, 7.8 MB a level -- a
        // full-screen surface each time. The four clocks below sum to 99.9 % of
        // `sl_mixed_us` and say the bytes were never the problem:
        //
        // | part | share | note |
        // |---|---|---|
        // | `settle` inside the read | **51.0 %** | the CPU blocking on the GPU |
        // | read copy | 21.1 % | 9.9 GB/s |
        // | write copy | 18.4 % | 11.3 GB/s |
        // | zeroing `staged` | 9.4 % | overwritten in full immediately after |
        //
        // The read costs 3.9x the write on identical byte counts for one reason:
        // `gva_view::read_rect` calls `settle_before_read` and
        // `write_rect_within` does not. `settle_gva_rect_read` equals
        // `sl_levels_n` exactly, and its `_overlap` count equals it too against
        // two `_disjoint`, so **every** read here waits on this device's own
        // submitted GPU writes -- 1.9 ms a call, 1.85 s of a 45 s boot with the
        // drain thread stalled. The staging copy is what creates the stall it
        // then pays for.
        //
        // Encoding the copy on the queue removes all four rows at once, and
        // stops ~500 MB/s of DRAM traffic that a unified-memory part's GPU is
        // contending for and that `proc_us` charges nowhere. Widening
        // `try_copy_whole_plane_on_gpu` is not the route: its `source_is_resident`
        // predicate asks whether the GPU holds newer content than the guest
        // pages, which under a host-pointer import is never true, and its
        // plane-to-plane shape is not this arm's buffer-to-image one.
        //
        // Banked per level rather than per slice so a sub-microsecond part is
        // not truncated to nothing by `as_micros`.
        let mut alloc_ns = 0u64;
        let mut window_ns = 0u64;
        let mut read_ns = 0u64;
        let mut write_ns = 0u64;
        let alloc_started = std::time::Instant::now();
        let mut staged = vec![0u8; (row_bytes.saturating_mul(h as u64)) as usize];
        alloc_ns += alloc_started.elapsed().as_nanos() as u64;
        crate::runtime::drain::note_store_route_n("sl_mixed_bytes", staged.len() as u64);
        for (source, destination) in
            std::iter::once(&level.first_slice).chain(level.remaining_slices.iter())
        {
            let src = &source.backing;
            let dst = &destination.backing;
            if src.width() != w || src.height() != h || dst.width() != w || dst.height() != h {
                return br(BlitStatus::Bounds, "sl_inner_dim_mismatch");
            }
            let window_started = std::time::Instant::now();
            let allowed = match texture_region_window(
                state,
                host,
                task_id,
                dst,
                Point { x: 0, y: 0, z: 0 },
                w,
                h as u64,
                1,
                bpp,
            ) {
                Ok(v) => v,
                Err(st) => return st,
            };
            window_ns += window_started.elapsed().as_nanos() as u64;
            let read_started = std::time::Instant::now();
            if let Err(st) = read_texture_rect(
                state,
                host,
                task_id,
                src,
                Point { x: 0, y: 0, z: 0 },
                row_bytes,
                h as u64,
                &mut staged,
            ) {
                return st;
            }
            read_ns += read_started.elapsed().as_nanos() as u64;
            let write_started = std::time::Instant::now();
            if let Err(st) = write_texture_rect(
                state,
                host,
                task_id,
                dst,
                Point { x: 0, y: 0, z: 0 },
                row_bytes,
                h as u64,
                &staged,
                allowed.as_ref(),
            ) {
                return st;
            }
            write_ns += write_started.elapsed().as_nanos() as u64;
        }
        crate::runtime::drain::note_store_route_us("sl_mixed_alloc_us", alloc_ns / 1000);
        crate::runtime::drain::note_store_route_us("sl_mixed_window_us", window_ns / 1000);
        crate::runtime::drain::note_store_route_us("sl_mixed_read_us", read_ns / 1000);
        crate::runtime::drain::note_store_route_us("sl_mixed_write_us", write_ns / 1000);
        crate::runtime::drain::note_store_route_us(
            "sl_mixed_us",
            sl_mixed_started.elapsed().as_micros() as u64,
        );
    }
    BlitStatus::Ok
}

/// Execute blit fence update (`0x13c`) or wait (`0x13d`) on the named fence object.
///
/// See [`fence_exec::execute_fence`].
pub fn execute_blit_fence(state: &mut Device, task_id: u32, cmd: &Command) -> BlitStatus {
    clear_blit_fail_reason();
    if cmd.kind != Kind::Fence {
        return br(BlitStatus::Unsupported, "fence_wrong_kind");
    }
    let action = match cmd.opcode {
        wire_blit::OPCODE_UPDATE_FENCE => FenceAction::Update,
        wire_blit::OPCODE_WAIT_FOR_FENCE => FenceAction::Wait,
        _ => return br(BlitStatus::Unsupported, "fence_bad_opcode"),
    };
    blit_status_from_fence(fence_exec::execute_fence(
        state,
        task_id,
        FenceDomain::BlitFence,
        cmd.fence,
        action,
    ))
}

/// Re-express a fence outcome as a blit outcome, carrying the reason across.
///
/// Named rather than inlined so the reason-forwarding is directly testable: the
/// `Unsupported` arm used to write a flat `fence_unsupported`, so all seven
/// fence/event refusals — bad domain, event-on-fence-path, either timeout form,
/// either invalid plan, unknown event kind — reached the blit dispatch line as
/// one indistinguishable reason. The forwarded slug is registered by
/// `FenceStatus`, not by this file.
pub(crate) fn blit_status_from_fence(status: FenceStatus) -> BlitStatus {
    match status {
        FenceStatus::Ok => BlitStatus::Ok,
        FenceStatus::Pending => BlitStatus::FencePending,
        FenceStatus::Missing => br(BlitStatus::MissingResource, "fence_missing"),
        FenceStatus::Unsupported(why) => br(BlitStatus::Unsupported, why),
    }
}

/// Execute a decoded blit command on the product path.
///
/// Returns [`BlitStatus::Unsupported`] for resource/image/mipmap opcodes
/// that other modules own or that are protocol no-ops (caller should not count
/// those as copy/fill failures). Fences use [`execute_blit_fence`].
///
/// Nothing follows a successful copy. A blit into an IOSurface texture destination writes
/// the guest pages directly and no GPU object caches those bytes, so the
/// content is coherent by construction and there is nothing to invalidate.
/// Each arm resolved its own destination in order to write it, so a second
/// resolve afterwards only repeats the page walk.
pub fn execute_blit<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    cmd: &Command,
) -> BlitStatus {
    // Fresh reason channel per command: an uninstrumented failure reports empty
    // rather than a stale slug left by a prior blit (see `br` / `blit_fail_reason`).
    clear_blit_fail_reason();
    // One clock over the whole dispatch, attributed by the arm that ran.
    //
    // The per-loop clocks added alongside this are the ones that say *why* an
    // arm is slow, but each of them had to be placed by hand and between them
    // they accounted for 0.7 % of `walk_blit_us`. Being exhaustive by
    // construction is what this one buys: every record entering `execute_blit`
    // leaves through exactly one arm and is charged to it, so the sum of
    // `blit_kind_*_us` cannot be less than the rail's cost the way a hand-placed
    // set can. A family that turns out to hold the wall clock and has no inner
    // clock yet is then a known gap rather than an invisible one.
    let kind_started = std::time::Instant::now();
    let kind_route = match cmd.kind {
        Kind::FillBuffer => "blit_kind_fill_us",
        Kind::FillBufferPattern4 => "blit_kind_fill4_us",
        Kind::Copy => match cmd.copy_kind {
            CopyKind::BufferToBuffer => "blit_kind_b2b_us",
            CopyKind::BufferToTexture => "blit_kind_b2t_us",
            CopyKind::TextureToBuffer => "blit_kind_t2b_us",
            CopyKind::TextureToTexture => "blit_kind_t2t_us",
            CopyKind::TextureToTextureSliceLevel => "blit_kind_t2t_sl_us",
            CopyKind::None => "blit_kind_none_us",
        },
        Kind::Fence => "blit_kind_fence_us",
        _ => "blit_kind_other_us",
    };
    let mut status = match cmd.kind {
        Kind::FillBuffer => exec_fill_buffer(state, host, task_id, cmd),
        Kind::FillBufferPattern4 => exec_fill_buffer_pattern4(state, host, task_id, cmd),
        Kind::Copy => match cmd.copy_kind {
            CopyKind::BufferToBuffer => exec_copy_buffer_to_buffer(state, host, task_id, cmd),
            CopyKind::BufferToTexture => {
                match resolve_buffer_to_texture_blit(state, host, task_id, cmd) {
                    Ok(operation) => execute_resolved_blit(state, host, task_id, operation),
                    Err(status) => status,
                }
            }
            CopyKind::TextureToBuffer => {
                match resolve_texture_to_buffer_blit(state, host, task_id, cmd) {
                    Ok(operation) => execute_resolved_blit(state, host, task_id, operation),
                    Err(status) => status,
                }
            }
            CopyKind::TextureToTexture => {
                match resolve_texture_to_texture_blit(state, host, task_id, cmd) {
                    Ok(operation) => execute_resolved_blit(state, host, task_id, operation),
                    Err(status) => status,
                }
            }
            CopyKind::TextureToTextureSliceLevel => {
                if cmd.slice_count == 0 || cmd.level_count == 0 {
                    BlitStatus::ZeroExtent
                } else {
                    match resolve_texture_copy_batch(state, host, task_id, cmd) {
                        Ok(operation) => execute_resolved_blit(state, host, task_id, operation),
                        Err(status) => status,
                    }
                }
            }
            CopyKind::None => br(BlitStatus::Unsupported, "copy_kind_none"),
        },
        Kind::Fence => execute_blit_fence(state, task_id, cmd),
        Kind::Resource | Kind::Image | Kind::Unknown => {
            br(BlitStatus::Unsupported, "blit_kind_unsupported")
        }
        // The three indirect-command-buffer records never reach here:
        // `handle_blit_record` answers them itself, counting the two that are
        // lost work and treating the optimize hint as the no-op it is. A
        // sighting means the dispatch there stopped routing them, so it gets a
        // reason of its own rather than joining the unsupported kinds above.
        Kind::IcbRange | Kind::IcbCopy => br(BlitStatus::Unsupported, "blit_kind_icb_misrouted"),
        // The two `BlitEncoderSPI` records this device decodes and does not
        // apply. Like the ICB pair above they are answered by
        // `handle_blit_record`, which counts the texture fill as lost work and
        // the compressed-texture invalidate as the no-op it is — so reaching
        // here means that dispatch stopped routing them, which is a different
        // defect from a kind nobody implemented and gets its own reason.
        Kind::FillTexture | Kind::InvalidateCompressedTexture => {
            br(BlitStatus::Unsupported, "blit_kind_spi_misrouted")
        }
    };
    if status == BlitStatus::Ok {
        status = complete_blit_write(state, task_id, cmd);
    }
    crate::runtime::drain::note_store_route_us(
        kind_route,
        kind_started.elapsed().as_micros() as u64,
    );
    status
}

#[cfg(test)]
mod tests;
