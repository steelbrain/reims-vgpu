//! Type-8 texture-view chain resolution and the CPU linear-texture loads that
//! consume it: swizzle application, native upload-format selection, and the
//! tight-row RGBA/BGRA readers shared by the sample and seed paths.
//!
//! Backend-independent, so [`super`] declares this module ungated and
//! re-exports its items flat — callers keep addressing them as
//! `crate::runtime::draw::<name>`. The two items that really are arm- or
//! test-specific carry their own cfg. `use super::*` pulls in the parent's
//! imports, which this module shares.

use super::*;

/// Type-8 view resolution for sample/seed paths.
#[derive(Clone, Debug)]
pub(crate) struct ViewResolve {
    /// Non-view base texture ref after walking the view chain (archive
    /// `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN` walk).
    pub(crate) base_texture_ref: u32,
    pub(crate) level: u32,
    /// Present when the view carries a swizzle form (opcode 0x1b); selectors already validated.
    pub(crate) swizzle: Option<pixel_format::SwizzlePlan>,
    /// Non-zero view pixel format from the descriptor (`@16`); `None` inherits the base format.
    pub(crate) pixel_format: Option<u16>,
}

/// A specific refusal while resolving one type-8 texture-view chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TextureViewDecline {
    HopEntryMissing {
        texture_ref: u32,
    },
    HopObjectNotView {
        texture_ref: u32,
        object_type: u8,
    },
    HopDescriptorMissing {
        texture_ref: u32,
        descriptor_length: u32,
    },
    HopDecode {
        texture_ref: u32,
        opcode: u32,
        declared: u32,
        descriptor_len: usize,
        bytes_hex: String,
        reason: DecodeStatus,
    },
    HopZeroBase {
        texture_ref: u32,
        opcode: u32,
    },
    HopLevelOverflow {
        texture_ref: u32,
        level_base: u64,
    },
    HopSwizzleInvalid {
        texture_ref: u32,
        selectors: [u8; 4],
    },
    ChainSelfOrZero {
        base: u32,
        next: u32,
        depth: u32,
    },
    ChainOverflow {
        base: u32,
        depth: u32,
    },
}

impl Decline for TextureViewDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::HopEntryMissing { .. } => {
                crate::observe::ladder_slug!("texture_view_hop", no_list_entry)
            }
            Self::HopObjectNotView { .. } => {
                crate::observe::ladder_slug!("texture_view_hop", wrong_type)
            }
            Self::HopDescriptorMissing { .. } => {
                crate::observe::ladder_slug!("texture_view_hop", desc_read)
            }
            // Keep the descriptor decoder's exact registered reason primary.
            Self::HopDecode { reason, .. } => reason.slug(),
            Self::HopZeroBase { .. } => "texture_view_hop_zero_base",
            Self::HopLevelOverflow { .. } => "texture_view_hop_level_overflow",
            Self::HopSwizzleInvalid { .. } => "texture_view_hop_swizzle_invalid",
            Self::ChainSelfOrZero { .. } => "texture_view_chain_self_or_zero",
            Self::ChainOverflow { .. } => "texture_view_chain_overflow",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::HopEntryMissing { texture_ref } => {
                vec![("texture_ref", texture_ref.to_string())]
            }
            Self::HopObjectNotView {
                texture_ref,
                object_type,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("object_type", object_type.to_string()),
            ],
            Self::HopDescriptorMissing {
                texture_ref,
                descriptor_length,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("descriptor_length", descriptor_length.to_string()),
            ],
            Self::HopDecode {
                texture_ref,
                opcode,
                declared,
                descriptor_len,
                bytes_hex,
                reason,
            } => {
                let mut fields = vec![
                    ("texture_ref", texture_ref.to_string()),
                    ("opcode", format!("{opcode:#x}")),
                    ("declared", declared.to_string()),
                    ("descriptor_len", descriptor_len.to_string()),
                    ("bytes", bytes_hex.clone()),
                ];
                fields.extend(reason.fields());
                fields
            }
            Self::HopZeroBase {
                texture_ref,
                opcode,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("opcode", format!("{opcode:#x}")),
            ],
            Self::HopLevelOverflow {
                texture_ref,
                level_base,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("level_base", level_base.to_string()),
            ],
            Self::HopSwizzleInvalid {
                texture_ref,
                selectors,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                (
                    "selectors",
                    selectors
                        .iter()
                        .map(u8::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
            Self::ChainSelfOrZero { base, next, depth } => vec![
                ("base", base.to_string()),
                ("next", next.to_string()),
                ("depth", depth.to_string()),
            ],
            Self::ChainOverflow { base, depth } => {
                vec![("base", base.to_string()), ("depth", depth.to_string())]
            }
        }
    }
}

crate::observe::decline_display!(TextureViewDecline);

impl std::error::Error for TextureViewDecline {}

/// Archive `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN` — nested type-8 views collapse
/// to a non-view base (`apple_pv_gpu_resource_resolve_texture` chain walk).
///
/// This is the decoded contract's own bound, not a budget of ours: a chain that
/// needs a ninth hop is one the guest's own resolver would not have followed
/// either, so refusing it is fidelity rather than a shortfall. That makes it the
/// number **every** arm that walks a type-8 chain must use, and it is `pub(crate)`
/// for exactly that reason. Two arms walk one: this module's
/// [`resolve_texture_view_reasoned`] for the draw/sample path, and
/// `blit_exec::resolve_texture_backing_depth` for copies. They used to disagree —
/// blit stopped at five hops on a number its own comment called "not a contract
/// limit" — so a six-deep chain sampled correctly and had its blit dropped as
/// `tex_view_depth_cap`. Both now count hops against this constant.
///
/// It is also what terminates a guest-built cycle. `A views B views A` is
/// expressible and neither walk carries a visited set; the chain simply runs out
/// of hops and refuses visibly. A cycle is malformed, so a refusal is the right
/// answer — the bound only has to stop the recursion, and the contract's own
/// depth already does.
pub(crate) const MAX_TEXTURE_VIEW_CHAIN: usize = 8;

/// Report a slice range the render path decodes and does not apply.
///
/// [`decode_texture_view_hop_reasoned`] resolves a type-8 view to four things —
/// base ref, mip level, swizzle and format override — and the ranged forms
/// (opcodes `0x08` and `0x1b`) carry two more that nothing on this path reads:
/// `slice_base` and `slice_count`. `blit_exec` consumes them; the draw and
/// sample path does not. So a guest that views slices `[5, 9)` of a texture
/// array samples slice **0** of the base, silently, and the wrong texels reach
/// the frame with no refusal anywhere.
///
/// Reported rather than declined, because declining would break every view
/// that asks for the default. A view whose range *is* the default —
/// `slice_base == 0` with at most one slice — is asking for what this path
/// already does, so it says nothing. That makes this a healthy zero, and a
/// non-zero reading is the measured argument for threading the slice through.
///
/// # The reading, and what it says about doing that work
///
/// **Zero.** Driven x86/PCI boot, `web-content-probe -n 10 --churn 1`: not one
/// `slice_dropped` line over the whole run, against 10/10 visual-gate regions
/// on colour. This guest binds no texture-array or cube view with a non-default
/// slice range on this workload, so threading `slice_base`/`slice_count`
/// through the draw and sample path would cost a `ViewResolve` field, every
/// consumer of it, and a `baseArrayLayer`/`layerCount` on the sample view — for
/// no measured benefit on the pathway that can measure it.
///
/// So it stays reported. The gap is real and the wrong texels really would reach
/// the frame if a guest asked; the argument for closing it has to come from a
/// workload that puts a non-zero reading here. `blit_exec` already consumes both
/// fields, so that arm is the reference when one does.
///
/// Keyed by texture ref through [`crate::observe::state_changed`] rather than
/// latched once: this runs per bind, so an undeduped line floods and a
/// first-sight latch goes quiet after the first view and never reports the
/// second. A transition report is bounded by the number of real changes.
fn note_view_slice_range_dropped(
    texture_ref: u32,
    opcode: u32,
    view: &crate::runtime::decode::resource::TextureViewDescriptor,
) {
    if !view.carries_range() || (view.slice_base == 0 && view.slice_count <= 1) {
        return;
    }
    let state = view.slice_base.rotate_left(32) ^ view.slice_count;
    if !crate::observe::state_changed("view_slice_dropped", texture_ref as u64, state) {
        return;
    }
    crate::observe::fail(format!(
        "texture_view slice_dropped ref={texture_ref} opcode={opcode:#x} \
         base={} count={} note=render path samples slice 0",
        view.slice_base, view.slice_count
    ));
}

/// Decode one type-8 hop (does not walk nested bases).
///
/// The `Result` carries a specific failure slug for the always-on fail log. No
/// wrapper collapses it at this level: the slug travels up through
/// [`resolve_texture_view_reasoned`]'s `?`, and [`resolve_texture_view`] is what
/// turns the whole walk into an `Option` for the hot path.
fn decode_texture_view_hop_reasoned<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Result<(u32, u32, Option<pixel_format::SwizzlePlan>, Option<u16>), TextureViewDecline> {
    use crate::runtime::decode::resource::{
        decode_texture_view_descriptor, texture_type8_header, OBJECT_TYPE_TEXTURE_VIEW,
    };
    let (_entry, desc) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        texture_ref,
        &[OBJECT_TYPE_TEXTURE_VIEW],
    )
    .map_err(|rung| match rung {
        objects::LadderRung::NoListEntry => TextureViewDecline::HopEntryMissing { texture_ref },
        objects::LadderRung::WrongType { got } => TextureViewDecline::HopObjectNotView {
            texture_ref,
            object_type: got,
        },
        objects::LadderRung::DescRead { declared_len } => {
            TextureViewDecline::HopDescriptorMissing {
                texture_ref,
                descriptor_length: declared_len,
            }
        }
    })?;
    // Bytes visible before decode, for the len-mismatch / bad-opcode census.
    let (opcode, declared) = texture_type8_header(&desc).unwrap_or((0, 0));
    let view = decode_texture_view_descriptor(&desc).map_err(|reason| {
        // Dump the full wire blob for an unknown texture-view opcode: this is the
        // only signal that reveals a new serializer variant (off the hot path —
        // fires only on a genuine decode failure).
        let hex: String = desc.iter().map(|b| format!("{b:02x}")).collect();
        TextureViewDecline::HopDecode {
            texture_ref,
            opcode,
            declared,
            descriptor_len: desc.len(),
            bytes_hex: hex,
            reason,
        }
    })?;
    if view.base_texture_ref == 0 {
        return Err(TextureViewDecline::HopZeroBase {
            texture_ref,
            opcode,
        });
    }
    note_view_slice_range_dropped(texture_ref, opcode, &view);
    let level = if view.carries_range() {
        // level_base is a mip index (u64 on wire); reject pathological values.
        if view.level_base > u32::MAX as u64 {
            return Err(TextureViewDecline::HopLevelOverflow {
                texture_ref,
                level_base: view.level_base,
            });
        }
        view.level_base as u32
    } else {
        0
    };
    let swizzle = if view.carries_swizzle() {
        // Malformed selectors (not in 0..5) fail the resolve — visible soft miss on sample.
        Some(pixel_format::swizzle_plan(&view.swizzle).ok_or(
            TextureViewDecline::HopSwizzleInvalid {
                texture_ref,
                selectors: view.swizzle,
            },
        )?)
    } else {
        None
    };
    // Zero pixel_format means inherit base (serializer always writes a real format when set).
    let pixel_format = view.declared_pixel_format();
    Ok((view.base_texture_ref, level, swizzle, pixel_format))
}

/// Resolve type-8 view to non-view base + mip + format override + swizzle.
///
/// The `Result` carries a specific failure slug (`reason=view_resolve` sub-case)
/// for the always-on fail log; [`resolve_texture_view`] collapses it to `Option`
/// for the hot path. Walks nested type-8 bases up to [`MAX_TEXTURE_VIEW_CHAIN`]
/// (archive `apple_pv_gpu_resource_resolve_texture` chain). Outer-most view
/// supplies level / format / swizzle (inner hops only extend the base ref),
/// matching the product RT path which materializes a single selected level.
pub(crate) fn resolve_texture_view_reasoned<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Result<ViewResolve, TextureViewDecline> {
    use crate::runtime::decode::resource::OBJECT_TYPE_TEXTURE_VIEW;

    let (mut base, level, swizzle, pixel_format) =
        decode_texture_view_hop_reasoned(state, host, task_id, texture_ref)?;

    // Collapse nested type-8 bases to a non-view texture (type-11 / type-2/3).
    let mut depth = 0u32;
    for _ in 1..MAX_TEXTURE_VIEW_CHAIN {
        let Some(entry) = objects::lookup_list_entry(state, host, task_id, base) else {
            // Base missing from the list — leave the ref for the caller to fail
            // visibly (same as a one-hop miss on an unmapped base).
            break;
        };
        if entry.object_type != OBJECT_TYPE_TEXTURE_VIEW {
            break;
        }
        depth += 1;
        let (next, _lvl, _sw, _fmt) = decode_texture_view_hop_reasoned(state, host, task_id, base)?;
        if next == 0 || next == base {
            return Err(TextureViewDecline::ChainSelfOrZero { base, next, depth });
        }
        base = next;
    }

    // Final base must not still be a type-8 view past the chain cap.
    if let Some(entry) = objects::lookup_list_entry(state, host, task_id, base) {
        if entry.object_type == OBJECT_TYPE_TEXTURE_VIEW {
            return Err(TextureViewDecline::ChainOverflow { base, depth });
        }
    }

    Ok(ViewResolve {
        base_texture_ref: base,
        level,
        swizzle,
        pixel_format,
    })
}

/// Resolve type-8 view to non-view base + mip + format override + swizzle.
///
/// Returns `None` if the ref is not a type-8 view, a hop is short/unsupported,
/// the chain exceeds the max depth without a non-view base, a base ref is zero,
/// or swizzle selectors are malformed. See [`resolve_texture_view_reasoned`] for
/// the specific reason on the fail path.
pub(super) fn resolve_texture_view<M: HostMemory + HostOps>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Option<ViewResolve> {
    resolve_texture_view_reasoned(state, host, task_id, texture_ref).ok()
}

/// Which term of [`effective_view_sample_format`] refused.
///
/// Three different bugs, and only one of them is this crate's:
///
/// * `BaseUndeclared` — a format the guest's own texture is in that
///   `contract::pixel_format` has no row for. **Ours.** Nothing about the bind
///   is wrong; this table is short.
/// * `ViewUndeclared` — the guest named an override this table has no row for.
///   Also ours, one argument over.
/// * `WidthMismatch` — the guest asked to view N bytes of storage as M, which
///   Metal itself forbids. **The guest's**, and the only one of the three the
///   old single slug actually described.
///
/// They were one `None`, printed as `format_incompatible` — a name that asserts
/// the third. Two readings of a macos-26 log spent their time in the guest's
/// descriptor over `MTLPixelFormatR8Uint`, which was the first case: a row this
/// table was simply missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ViewSampleRefusal {
    BaseUndeclared { base: u16 },
    ViewUndeclared { view: u16 },
    WidthMismatch { base_bpp: u32, view_bpp: u32 },
}

impl std::fmt::Display for ViewSampleRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BaseUndeclared { base } => {
                write!(f, "base_undeclared_here base={base:#x}")
            }
            Self::ViewUndeclared { view } => {
                write!(f, "view_undeclared_here view={view:#x}")
            }
            Self::WidthMismatch { base_bpp, view_bpp } => {
                write!(
                    f,
                    "view_width_mismatch base_bpp={base_bpp} view_bpp={view_bpp}"
                )
            }
        }
    }
}

/// Pick the sample format for a type-8 view over base storage.
///
/// Metal texture views require the view format to be bpp-compatible with the base.
/// Unknown formats (no `bytes_per_pixel`) fail visibly. `None` override inherits base.
///
/// Almost every caller only needs "may I", which is what this answers. A caller
/// that has to *print* a refusal wants [`effective_view_sample_format_reasoned`]
/// instead — same rule, one implementation, following the
/// [`resolve_texture_view`] / [`resolve_texture_view_reasoned`] pair above.
pub(crate) fn effective_view_sample_format(base_fmt: u16, view_fmt: Option<u16>) -> Option<u16> {
    effective_view_sample_format_reasoned(base_fmt, view_fmt).ok()
}

/// [`effective_view_sample_format`], naming the term that refused.
pub(crate) fn effective_view_sample_format_reasoned(
    base_fmt: u16,
    view_fmt: Option<u16>,
) -> Result<u16, ViewSampleRefusal> {
    let sample = view_fmt.unwrap_or(base_fmt);
    let base_bpp = pixel_format::bytes_per_pixel(base_fmt)
        .ok_or(ViewSampleRefusal::BaseUndeclared { base: base_fmt })?;
    let sample_bpp = pixel_format::bytes_per_pixel(sample)
        .ok_or(ViewSampleRefusal::ViewUndeclared { view: sample })?;
    if base_bpp != sample_bpp {
        return Err(ViewSampleRefusal::WidthMismatch {
            base_bpp,
            view_bpp: sample_bpp,
        });
    }
    Ok(sample)
}

/// Apply a type-8 view swizzle by rewriting tight RGBA8 texels. Identity plans
/// are no-ops. Returns `None` only if the buffer length is not a multiple of 4
/// (corrupt load).
///
/// **This is the slow way and it is counted.** Vulkan performs the same remap
/// for free on the image view, so the Vulkan pathway uses
/// `SampledImageResource::swizzle` and never calls this; every invocation here
/// is a texture that gave up its zero-copy crossing to be remapped by hand. The
/// Metal-direct pathway still needs it, so it reports itself rather than being
/// deleted.
#[cfg(any(test, all(feature = "backend-metal", target_os = "macos")))]
pub(super) fn apply_view_swizzle_rgba8(
    rgba: &mut [u8],
    plan: Option<&pixel_format::SwizzlePlan>,
    texture_ref: u32,
) -> Option<()> {
    let Some(plan) = plan else {
        return Some(());
    };
    if pixel_format::swizzle_is_identity(plan) {
        return Some(());
    }
    if !rgba.len().is_multiple_of(4) {
        return None;
    }
    crate::runtime::census::view_swizzle_census::note_cpu_remap(texture_ref);
    for px in rgba.chunks_exact_mut(4) {
        let input = [px[0], px[1], px[2], px[3]];
        let out = pixel_format::apply_swizzle_rgba8(plan, input);
        px.copy_from_slice(&out);
    }
    Some(())
}

/// Why a linear (buffer-backed) texture could not be loaded for sampling.
///
/// [`load_linear_texture_impl`] runs fourteen distinct checks and used to
/// return a bare `Option`, so its one caller on the sampling path printed a
/// single label — `linear_sample_miss reason=guest_load` — for all of them.
/// That label is the caller's assumption at full confidence: it says "the guest
/// load failed" whether the object-list lookup missed, the descriptor would not
/// decode, the format has no conversion, or a row's guest page is unmapped.
/// Those have four different fixes.
///
/// The reason now comes *from* the check that refused. It is worth the type:
/// this path failing drops a whole draw (`draw_vk_nothing_stored`), and the
/// draw it was observed dropping is the WindowServer's full-screen composite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LinearLoadRefusal {
    /// The texture ref is not in this task's object list.
    ObjectListMiss,
    /// The list entry is not a texture or texture variant.
    NotATexture { object_type: u8 },
    /// The entry's descriptor bytes could not be read from guest memory.
    DescriptorUnreadable,
    /// The descriptor bytes are not a decodable texture descriptor.
    DescriptorUndecodable,
    /// The descriptor carries no pixel format, so nothing names its layout.
    NoPixelFormat,
    /// A type-8 view format that is not the same bytes-per-pixel as the base,
    /// which would reinterpret the allocation rather than re-read it.
    ViewFormatBppMismatch { base: u16, view: u16 },
    /// The requested mip level has no address/layout in the descriptor.
    NoLevelGva { level: u32 },
    /// This format has no bytes-per-pixel in the contract table, so no row
    /// length can be computed for it.
    FormatBppUnknown { format: u16 },
    /// The declared row stride is below one tight row of this format.
    RowStrideBelowTight { stride: u64, tight: u32 },
    /// Zero width or height.
    ZeroExtent { width: u32, height: u32 },
    /// A size computation overflowed, or the image exceeds the host allocation
    /// ceiling.
    SizeOverflow,
    /// The level's rows run past the descriptor's own allocation size.
    SpanExceedsAllocation { end: u64, allocation: u64 },
    /// A tight-row bulk read of the whole image did not resolve in the task's
    /// page table.
    TightImageUnreadable,
    /// One padded row did not resolve in the task's page table. `row` is the
    /// first that failed, which says whether the allocation is partly mapped.
    PaddedRowUnreadable { row: u32 },
    /// The format has a bytes-per-pixel but no row conversion to RGBA8.
    RowConvertUnsupported { format: u16 },
}

impl crate::observe::Decline for LinearLoadRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::ObjectListMiss => "linear_load_object_list_miss",
            Self::NotATexture { .. } => "linear_load_not_a_texture",
            Self::DescriptorUnreadable => "linear_load_descriptor_unreadable",
            Self::DescriptorUndecodable => "linear_load_descriptor_undecodable",
            Self::NoPixelFormat => "linear_load_no_pixel_format",
            Self::ViewFormatBppMismatch { .. } => "linear_load_view_bpp_mismatch",
            Self::NoLevelGva { .. } => "linear_load_no_level_gva",
            Self::FormatBppUnknown { .. } => "linear_load_format_bpp_unknown",
            Self::RowStrideBelowTight { .. } => "linear_load_stride_below_tight",
            Self::ZeroExtent { .. } => "linear_load_zero_extent",
            Self::SizeOverflow => "linear_load_size_overflow",
            Self::SpanExceedsAllocation { .. } => "linear_load_span_exceeds_alloc",
            Self::TightImageUnreadable => "linear_load_tight_image_unreadable",
            Self::PaddedRowUnreadable { .. } => "linear_load_padded_row_unreadable",
            Self::RowConvertUnsupported { .. } => "linear_load_row_convert_unsupported",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NotATexture { object_type } => vec![("objtype", object_type.to_string())],
            Self::ViewFormatBppMismatch { base, view } => vec![
                ("base_fmt", format!("{base:#x}")),
                ("view_fmt", format!("{view:#x}")),
            ],
            Self::NoLevelGva { level } => vec![("level", level.to_string())],
            Self::FormatBppUnknown { format } | Self::RowConvertUnsupported { format } => {
                vec![("fmt", format!("{format:#x}"))]
            }
            Self::RowStrideBelowTight { stride, tight } => {
                vec![("stride", stride.to_string()), ("tight", tight.to_string())]
            }
            Self::ZeroExtent { width, height } => vec![("extent", format!("{width}x{height}"))],
            Self::SpanExceedsAllocation { end, allocation } => {
                vec![("end", end.to_string()), ("alloc", allocation.to_string())]
            }
            Self::PaddedRowUnreadable { row } => vec![("row", row.to_string())],
            _ => Vec::new(),
        }
    }
}

pub(super) fn load_linear_texture_rgba_host<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
    site: crate::runtime::render_writeback::SettleSite,
) -> Option<Vec<u8>> {
    load_linear_texture_impl(
        state,
        host,
        task_id,
        texture_ref,
        level,
        format_override,
        false,
        site,
    )
    .ok()
    .map(|(bytes, _)| bytes)
}

/// When a sampled format's guest bytes are ALREADY in the final upload order —
/// so the loader can read padded source rows straight into the tight output
/// with no intermediate buffer and no per-row convert — this returns the engine
/// upload format. `RGBA8` always qualifies (its convert is an identity copy);
/// `BGRA8` qualifies only when the caller opts into a native BGRA8 upload
/// (`native_bgra8`), otherwise it must be swapped to RGBA8. Every other format
/// needs a real convert pass and returns `None`.
pub(super) fn linear_native_upload_format(
    sample_format: u16,
    native_bgra8: bool,
) -> Option<TexelLayout> {
    use pixel_format::SampledClass;
    // The decode contract's sampled class is the one rule for "which 8-bit
    // channel order is this"; it folds each sRGB format onto its linear
    // sibling's layout, which is right — they share a layout — but loses the
    // qualifier, so the census records what the fold cost.
    let upload = match pixel_format::sampled_class(sample_format)? {
        SampledClass::Rgba8Unorm => TexelLayout::Rgba8,
        SampledClass::Bgra8Unorm if native_bgra8 => TexelLayout::Bgra8,
        _ => return None,
    };
    note_srgb_upload_downgrade(srgb_census::site::LINEAR_NATIVE_UPLOAD, sample_format);
    Some(upload)
}

/// Record an sRGB downgrade on a byte-layout rail, if this format had a
/// qualifier to lose. One helper so the two CPU upload paths cannot drift on
/// when they report.
fn note_srgb_upload_downgrade(site: &'static str, sample_format: u16) {
    if pixel_format::is_srgb(sample_format) {
        srgb_census::note_downgrade(site, sample_format);
    }
}

// Eight, because the last one is the caller's identity and the whole point of
// it is that this leaf cannot derive it. Grouping the descriptor selectors into
// a struct would hide which of them a call site varies, which is what the
// wrappers above exist to make obvious.
#[allow(clippy::too_many_arguments)]
fn load_linear_texture_impl<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
    native_bgra8: bool,
    site: crate::runtime::render_writeback::SettleSite,
) -> Result<(Vec<u8>, TexelLayout), LinearLoadRefusal> {
    use LinearLoadRefusal as R;
    let (_entry, desc_bytes) = objects::resolve_descriptor(
        state,
        host,
        task_id,
        texture_ref,
        &[OBJECT_TYPE_TEXTURE, OBJECT_TYPE_TEXTURE_VARIANT],
    )
    .map_err(|rung| match rung {
        objects::LadderRung::NoListEntry => R::ObjectListMiss,
        objects::LadderRung::WrongType { got } => R::NotATexture { object_type: got },
        objects::LadderRung::DescRead { .. } => R::DescriptorUnreadable,
    })?;
    let tex = decode_texture_descriptor(&desc_bytes).map_err(|_| R::DescriptorUndecodable)?;
    if tex.declared_pixel_format().is_none() {
        return Err(R::NoPixelFormat);
    }
    let base_fmt = tex.pixel_format;
    let sample_fmt = effective_view_sample_format(base_fmt, format_override).ok_or(
        R::ViewFormatBppMismatch {
            base: base_fmt,
            view: format_override.unwrap_or(base_fmt),
        },
    )?;
    let (gva, layout) = tex
        .level_gva(level, state.page_shift)
        .ok_or(R::NoLevelGva { level })?;
    let w = layout.width;
    let h = layout.height;
    let bpr = layout.row_stride;
    if bpr > u32::MAX as u64 {
        return Err(R::SizeOverflow);
    }
    let bpr_u32 = bpr as u32;
    let tight = pixel_format::tight_row_bytes(w, base_fmt)
        .ok_or(R::FormatBppUnknown { format: base_fmt })?;
    if w == 0 || h == 0 {
        return Err(R::ZeroExtent {
            width: w,
            height: h,
        });
    }
    if bpr_u32 < tight {
        return Err(R::RowStrideBelowTight { stride: bpr, tight });
    }
    let need_rgba = (w as u64)
        .checked_mul(h as u64)
        .and_then(|n| n.checked_mul(RGBA8_BPP as u64))
        .and_then(host_alloc_len)
        .ok_or(R::SizeOverflow)?;
    // The extent actually read, not `bpr * h` — see `TextureLevelLayout::read_span`.
    let span = layout.read_span(tight).ok_or(R::SizeOverflow)?;
    let end = layout.offset.saturating_add(span);
    if tex.allocation_size != 0 && end > tex.allocation_size {
        return Err(R::SpanExceedsAllocation {
            end,
            allocation: tex.allocation_size,
        });
    }
    // Deferred-writeback flush-on-access: the reads below walk raw task GVAs
    // and bypass the mapping-keyed hooks — land any resident-authoritative
    // window whose physical pages alias the sampled span first, and only then.
    //
    // Narrowed on this read's own pages, as `load_linear_guest_memoized` is: the
    // walk runs only when something is outstanding, and the pages read are the
    // ones `read_span` names — not `bpr * h`, so a padded source does not claim
    // the trailing padding it never touches.
    // The reads below walk a raw task GVA, but the reference names a resource,
    // and a debt is keyed by mapping id — so only what this reference resolves
    // to is paid. `note_unnamed_reach` stays as the standing alarm for the one
    // thing the naming cannot see, raw page aliasing; it samples this read's own
    // walk against every owed surface and must stay at zero overlap.
    {
        let (tasks, page_shift) = (&state.tasks, state.page_shift);
        let page_size = state.page_size();
        crate::runtime::writeback_debt::note_unnamed_reach(state, || {
            let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
            let gpas = gva_mem::task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift);
            (gpas.len() as u64 == want).then_some(gpas)
        });
    }
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    let (tasks, page_shift, page_size) = (&state.tasks, state.page_shift, state.page_size());
    crate::runtime::render_writeback::settle_guest_writes_unless_disjoint(
        site,
        || {
            let want = reims_vgpu_paging::span::pages_spanned(gva, span, page_size);
            let gpas = gva_mem::task_gva_page_gpas(host, tasks, task_id, gva, span, page_shift);
            (gpas.len() as u64 == want).then_some(gpas)
        },
    );
    // Tight display textures are the common compositor source. Read the whole
    // image with one task-root/cache lifetime: the row loop below otherwise
    // rebuilds the GVA walker cache once per row (1,080 times for the live
    // Safari source). Padded rows retain the conservative disjoint reads so we
    // never touch padding that the guest did not make readable.
    if bpr_u32 == tight {
        let (rgba, fmt) = load_tight_linear_rgba_with(w, h, sample_fmt, native_bgra8, |native| {
            gva_mem::read_task_gva_by_id(host, &state.tasks, task_id, gva, native, state.page_shift)
                .is_ok()
        })?;
        return Ok((rgba, fmt));
    }
    // Padded rows. When the source bytes are already in the final upload order
    // (RGBA8 always; BGRA8 under a native upload) AND the guest rows are 4-byte
    // tight, read each padded source row STRAIGHT into the tight output — no
    // intermediate row buffer, no per-row convert/swizzle pass. This is the
    // Safari-scroll fallback hot path (`lin_guest_fb`), so the elided convert
    // pass is a full second walk over the sampled bytes off the drain worker.
    let tight_4bpp = tight as u64
        == (w as u64)
            .checked_mul(RGBA8_BPP as u64)
            .ok_or(R::SizeOverflow)?;
    if let Some(fmt) = linear_native_upload_format(sample_fmt, native_bgra8).filter(|_| tight_4bpp)
    {
        let row_bytes = tight as usize;
        let mut rgba = vec![0u8; need_rgba];
        for y in 0..h {
            let row_gva = (y as u64)
                .checked_mul(bpr)
                .and_then(|off| gva.checked_add(off))
                .ok_or(R::SizeOverflow)?;
            let dst_off = (y as usize).checked_mul(row_bytes).ok_or(R::SizeOverflow)?;
            let dst = rgba
                .get_mut(dst_off..dst_off + row_bytes)
                .ok_or(R::SizeOverflow)?;
            gva_mem::read_task_gva_by_id(
                host,
                &state.tasks,
                task_id,
                row_gva,
                dst,
                state.page_shift,
            )
            .map_err(|_| R::PaddedRowUnreadable { row: y })?;
        }
        return Ok((rgba, fmt));
    }
    let mut rgba = vec![0u8; need_rgba];
    let mut row = vec![0u8; tight as usize];
    for y in 0..h {
        let row_gva = (y as u64)
            .checked_mul(bpr)
            .and_then(|off| gva.checked_add(off))
            .ok_or(R::SizeOverflow)?;
        gva_mem::read_task_gva_by_id(
            host,
            &state.tasks,
            task_id,
            row_gva,
            &mut row,
            state.page_shift,
        )
        .map_err(|_| R::PaddedRowUnreadable { row: y })?;
        let dst_off = (y as usize) * (w as usize) * 4;
        if !pixel_format::convert_row_to_rgba8(sample_fmt, &row, w, &mut rgba[dst_off..]) {
            return Err(R::RowConvertUnsupported { format: sample_fmt });
        }
    }
    Ok((rgba, TexelLayout::Rgba8))
}

pub(super) fn load_tight_linear_rgba_with<F>(
    width: u32,
    height: u32,
    sample_format: u16,
    native_bgra8: bool,
    mut read: F,
) -> Result<(Vec<u8>, TexelLayout), LinearLoadRefusal>
where
    F: FnMut(&mut [u8]) -> bool,
{
    use LinearLoadRefusal as R;
    let tight = pixel_format::tight_row_bytes(width, sample_format).ok_or(R::FormatBppUnknown {
        format: sample_format,
    })?;
    let native_len = (tight as u64)
        .checked_mul(height as u64)
        .and_then(host_alloc_len)
        .ok_or(R::SizeOverflow)?;
    let rgba_stride = width.checked_mul(RGBA8_BPP).ok_or(R::SizeOverflow)?;
    let rgba_len = (rgba_stride as u64)
        .checked_mul(height as u64)
        .and_then(host_alloc_len)
        .ok_or(R::SizeOverflow)?;
    let mut native = vec![0u8; native_len];
    if !read(&mut native) {
        return Err(R::TightImageUnreadable);
    }
    // The compositor's common BGRA8/RGBA8 sources already have the output
    // allocation size. Convert them in place so the bulk page walk does not
    // add a second display-sized allocation and copy.
    if native_len == rgba_len {
        // Same single rule as `linear_native_upload_format`: the contract's
        // sampled class names the channel order, and an sRGB source is reported
        // to the census rather than folded away unnoticed.
        match pixel_format::sampled_class(sample_format) {
            Some(pixel_format::SampledClass::Rgba8Unorm) => {
                note_srgb_upload_downgrade(srgb_census::site::TIGHT_LINEAR_LOAD, sample_format);
                return Ok((native, TexelLayout::Rgba8));
            }
            Some(pixel_format::SampledClass::Bgra8Unorm) => {
                note_srgb_upload_downgrade(srgb_census::site::TIGHT_LINEAR_LOAD, sample_format);
                if native_bgra8 {
                    // Upload the guest's native BGRA8 order; the engine binds a
                    // BGRA8 image and the sampler swizzles in hardware. Elides
                    // the full-image CPU channel-swap pass over the read bytes.
                    return Ok((native, TexelLayout::Bgra8));
                }
                for pixel in native.chunks_exact_mut(RGBA8_BPP as usize) {
                    pixel.swap(0, 2);
                }
                return Ok((native, TexelLayout::Rgba8));
            }
            _ => {}
        }
    }
    let mut rgba = vec![0u8; rgba_len];
    for y in 0..height as usize {
        let src_off = y.checked_mul(tight as usize).ok_or(R::SizeOverflow)?;
        let dst_off = y.checked_mul(rgba_stride as usize).ok_or(R::SizeOverflow)?;
        if !pixel_format::convert_row_to_rgba8(
            sample_format,
            &native[src_off..src_off + tight as usize],
            width,
            &mut rgba[dst_off..dst_off + rgba_stride as usize],
        ) {
            return Err(R::RowConvertUnsupported {
                format: sample_format,
            });
        }
    }
    Ok((rgba, TexelLayout::Rgba8))
}

#[cfg(test)]
mod texture_view_split_tests {
    use super::*;
    use crate::runtime::decode::resource::TextureViewDescriptor;

    #[test]
    fn view_pixel_format_override_effective() {
        use crate::contract::pixel_format::{
            MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
            MTL_FORMAT_RGBA8_UNORM,
        };
        assert_eq!(
            effective_view_sample_format(MTL_FORMAT_BGRA8_UNORM, None),
            Some(MTL_FORMAT_BGRA8_UNORM)
        );
        assert_eq!(
            effective_view_sample_format(MTL_FORMAT_BGRA8_UNORM, Some(MTL_FORMAT_RGBA8_UNORM)),
            Some(MTL_FORMAT_RGBA8_UNORM)
        );
        assert!(
            effective_view_sample_format(MTL_FORMAT_BGRA8_UNORM, Some(MTL_FORMAT_R8_UNORM))
                .is_none()
        );
        assert!(effective_view_sample_format(
            MTL_FORMAT_BGRA8_UNORM,
            Some(MTL_FORMAT_RGBA16_FLOAT)
        )
        .is_none());
        assert!(effective_view_sample_format(0, Some(MTL_FORMAT_RGBA8_UNORM)).is_none());
    }

    /// A format this crate has never declared and one the guest may not
    /// reinterpret are different refusals, and this gate could only say the
    /// second.
    ///
    /// It answers `None` the moment either side has no known texel width, so an
    /// undeclared format arrives at the compute and draw binds as
    /// `format_incompatible` — which reads as "the guest asked for an illegal
    /// view" and sends the next reader to the guest's descriptor rather than to
    /// this crate's own table. `R8Uint` is the worked example: a macOS 26 guest
    /// stages one into compute dispatches, `bytes_per_pixel` had no arm for it,
    /// and 51 dispatches a boot were refused under the wrong name.
    ///
    /// Declaring the width is all this asserts. Whether an *integer* texel may
    /// then be sampled is a separate question, answered separately and by name —
    /// see `an_integer_texel_is_declared_but_has_no_sampled_rail`.
    #[test]
    fn a_declared_format_clears_the_width_gate_whether_or_not_a_rail_takes_it() {
        use crate::contract::pixel_format::{
            MTL_FORMAT_R8_UINT, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RGBA8_UNORM,
        };
        assert_eq!(
            effective_view_sample_format(MTL_FORMAT_R8_UINT, None),
            Some(MTL_FORMAT_R8_UINT)
        );
        // One byte wide, so it views as the other one-byte formats and not as
        // anything wider. The mismatch arm is what stays load-bearing here.
        assert_eq!(
            effective_view_sample_format(MTL_FORMAT_R8_UINT, Some(MTL_FORMAT_R8_UNORM)),
            Some(MTL_FORMAT_R8_UNORM)
        );
        assert!(
            effective_view_sample_format(MTL_FORMAT_R8_UINT, Some(MTL_FORMAT_RGBA8_UNORM))
                .is_none()
        );
    }

    /// The three ways this gate can refuse are three different bugs, and only
    /// the third is the guest's. They shared one slug, and that slug asserted
    /// the third.
    ///
    /// The `is_none()`/`is_err()` agreement is what keeps this from becoming a
    /// second implementation of the rule: the plain gate is written in terms of
    /// the reasoned one, and this walks a table across all three outcomes plus
    /// the success case to say so.
    #[test]
    fn the_view_gate_names_which_of_its_three_terms_refused() {
        use crate::contract::pixel_format::{
            MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_R8_UNORM, MTL_FORMAT_RGBA8_UNORM,
        };
        // A value Metal does not define and this table therefore has no row for.
        const UNDECLARED: u16 = 0xfff0;

        let cases: &[(u16, Option<u16>, Result<u16, ViewSampleRefusal>)] = &[
            (
                MTL_FORMAT_BGRA8_UNORM,
                Some(MTL_FORMAT_RGBA8_UNORM),
                Ok(MTL_FORMAT_RGBA8_UNORM),
            ),
            (
                UNDECLARED,
                None,
                Err(ViewSampleRefusal::BaseUndeclared { base: UNDECLARED }),
            ),
            (
                MTL_FORMAT_BGRA8_UNORM,
                Some(UNDECLARED),
                Err(ViewSampleRefusal::ViewUndeclared { view: UNDECLARED }),
            ),
            (
                MTL_FORMAT_BGRA8_UNORM,
                Some(MTL_FORMAT_R8_UNORM),
                Err(ViewSampleRefusal::WidthMismatch {
                    base_bpp: 4,
                    view_bpp: 1,
                }),
            ),
        ];

        for &(base, view, ref want) in cases {
            let got = effective_view_sample_format_reasoned(base, view);
            assert_eq!(&got, want, "base={base:#x} view={view:?}");
            // The `Option` gate is this one with the reason dropped, and every
            // caller that only asks "may I" must keep agreeing with it.
            assert_eq!(
                effective_view_sample_format(base, view),
                got.ok(),
                "base={base:#x} view={view:?}: the two gates disagree"
            );
        }

        // Each refusal prints its own text, so the fail log can be ranked on it.
        let printed: Vec<String> = cases
            .iter()
            .filter_map(|(b, v, _)| effective_view_sample_format_reasoned(*b, *v).err())
            .map(|r| r.to_string())
            .collect();
        assert_eq!(printed.len(), 3);
        let distinct: std::collections::HashSet<&str> =
            printed.iter().map(String::as_str).collect();
        assert_eq!(
            distinct.len(),
            3,
            "two refusals share a spelling: {printed:?}"
        );
    }

    /// The point of typing this refusal is that the sink can tell the fifteen
    /// checks apart. Two sharing a slug would put them back behind one label —
    /// which is the state this replaced, where all fifteen printed
    /// `reason=guest_load`.
    #[test]
    fn every_linear_load_refusal_has_its_own_slug() {
        use crate::observe::Decline;
        let all = [
            LinearLoadRefusal::ObjectListMiss,
            LinearLoadRefusal::NotATexture { object_type: 9 },
            LinearLoadRefusal::DescriptorUnreadable,
            LinearLoadRefusal::DescriptorUndecodable,
            LinearLoadRefusal::NoPixelFormat,
            LinearLoadRefusal::ViewFormatBppMismatch { base: 1, view: 2 },
            LinearLoadRefusal::NoLevelGva { level: 3 },
            LinearLoadRefusal::FormatBppUnknown { format: 4 },
            LinearLoadRefusal::RowStrideBelowTight {
                stride: 5,
                tight: 6,
            },
            LinearLoadRefusal::ZeroExtent {
                width: 0,
                height: 7,
            },
            LinearLoadRefusal::SizeOverflow,
            LinearLoadRefusal::SpanExceedsAllocation {
                end: 8,
                allocation: 9,
            },
            LinearLoadRefusal::TightImageUnreadable,
            LinearLoadRefusal::PaddedRowUnreadable { row: 11 },
            LinearLoadRefusal::RowConvertUnsupported { format: 12 },
        ];
        let mut slugs: Vec<&str> = all.iter().map(|r| r.slug()).collect();
        let before = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "two refusals share a slug: {slugs:?}");
        // Every one that carries data must print it, or the slug names the
        // check without saying what it saw.
        for r in &all {
            let carries_data = !matches!(
                r,
                LinearLoadRefusal::ObjectListMiss
                    | LinearLoadRefusal::DescriptorUnreadable
                    | LinearLoadRefusal::DescriptorUndecodable
                    | LinearLoadRefusal::NoPixelFormat
                    | LinearLoadRefusal::SizeOverflow
                    | LinearLoadRefusal::TightImageUnreadable
            );
            assert_eq!(
                carries_data,
                !r.fields().is_empty(),
                "{} disagrees with itself about carrying fields",
                r.slug()
            );
        }
    }

    /// A padded-row refusal names the first row that failed, which is what
    /// separates "the allocation is not mapped at all" from "it is mapped up to
    /// row N" — different bugs with different fixes.
    #[test]
    fn a_padded_row_refusal_names_the_row() {
        use crate::observe::Decline;
        let r = LinearLoadRefusal::PaddedRowUnreadable { row: 11 };
        assert_eq!(r.fields(), vec![("row", "11".to_string())]);
    }

    /// Start capturing the always-on log; returns the offset to slice from.
    fn log_mark() -> usize {
        crate::observe::redirect_logs_for_tests();
        std::fs::read_to_string(crate::observe::fail_log_path())
            .unwrap_or_default()
            .len()
    }

    fn log_since(mark: usize) -> String {
        let body = std::fs::read_to_string(crate::observe::fail_log_path()).unwrap_or_default();
        body[mark.min(body.len())..].to_string()
    }

    fn ranged_view(slice_base: u64, slice_count: u64) -> TextureViewDescriptor {
        TextureViewDescriptor {
            view_opcode: crate::runtime::decode::resource::TEXTURE_VIEW_OPCODE_RANGED,
            base_texture_ref: 9,
            slice_base,
            slice_count,
            ..Default::default()
        }
    }

    /// A slice range the render path cannot honour is guest work lost, and the
    /// device must say so.
    ///
    /// The loss is real: `decode_texture_view_hop_reasoned` resolves a view to
    /// base ref, mip level, swizzle and format, and nothing downstream of it
    /// reads `slice_base`. A guest viewing slices `[5, 9)` samples slice 0.
    #[test]
    fn a_slice_range_the_render_path_drops_is_reported() {
        let mark = log_mark();
        // Distinct refs, because the report is keyed by ref and would
        // otherwise dedup the second away.
        note_view_slice_range_dropped(0x1001, 8, &ranged_view(5, 4));
        note_view_slice_range_dropped(0x1002, 0x1b, &ranged_view(0, 6));
        let log = log_since(mark);
        assert!(
            log.contains("slice_dropped ref=4097") && log.contains("base=5 count=4"),
            "a non-default slice base went unreported:\n{log}"
        );
        assert!(
            log.contains("slice_dropped ref=4098") && log.contains("base=0 count=6"),
            "a multi-slice range went unreported:\n{log}"
        );
    }

    /// ...and a view asking for what this path already does says nothing.
    ///
    /// This is the half that makes the counter usable: without it the line
    /// fires on every ordinary 2D view and a real loss is invisible in the
    /// volume. A healthy zero here is what makes a non-zero reading the
    /// measured argument for threading the slice through.
    #[test]
    fn a_default_slice_range_is_not_reported() {
        let mark = log_mark();
        note_view_slice_range_dropped(0x2001, 8, &ranged_view(0, 1));
        note_view_slice_range_dropped(0x2002, 8, &ranged_view(0, 0));
        // The format-only form carries no slice range at all.
        note_view_slice_range_dropped(
            0x2003,
            7,
            // Slice words set to what a ranged record would put there: the
            // gate must turn on the opcode alone, not on the words being
            // non-zero.
            &TextureViewDescriptor {
                view_opcode: crate::runtime::decode::resource::TEXTURE_VIEW_OPCODE_SIMPLE,
                base_texture_ref: 9,
                slice_base: 5,
                slice_count: 4,
                ..Default::default()
            },
        );
        let log = log_since(mark);
        assert!(
            !log.contains("slice_dropped"),
            "a default slice range was reported as a loss:\n{log}"
        );
    }

    /// The report is bounded by real changes, not by binds.
    ///
    /// It sits on the per-bind path, so an undeduped line floods the log and a
    /// once-latch goes quiet after the first view and never reports a second.
    #[test]
    fn the_same_view_bound_twice_reports_once_and_a_changed_range_reports_again() {
        let mark = log_mark();
        for _ in 0..5 {
            note_view_slice_range_dropped(0x3001, 8, &ranged_view(5, 4));
        }
        assert_eq!(
            log_since(mark).matches("slice_dropped").count(),
            1,
            "the per-bind path reported more than once for an unchanged view"
        );

        let mark = log_mark();
        note_view_slice_range_dropped(0x3001, 8, &ranged_view(6, 4));
        assert_eq!(
            log_since(mark).matches("slice_dropped").count(),
            1,
            "a view whose slice range changed did not report again"
        );
    }
}
