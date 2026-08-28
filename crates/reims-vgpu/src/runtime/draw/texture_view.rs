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
    /// The two formats occupy the same bytes per addressable unit but address a
    /// different **grid** of texels with them — one is block-compressed and the
    /// other is not.
    ///
    /// Distinct from [`Self::WidthMismatch`] because the two numbers that
    /// variant prints would be *equal* here and read as a contradiction:
    /// `BC3_RGBA` and `RGBA32Float` are both sixteen bytes a unit, and one of
    /// them spends those bytes on sixteen texels. Reinterpreting either as the
    /// other is not a view of one allocation.
    GridMismatch {
        base_compressed: bool,
        view_compressed: bool,
    },
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
            Self::GridMismatch {
                base_compressed,
                view_compressed,
            } => {
                write!(
                    f,
                    "view_grid_mismatch base_compressed={base_compressed} \
                     view_compressed={view_compressed}"
                )
            }
        }
    }
}

/// Pick the sample format for a type-8 view over base storage.
///
/// Metal texture views require the view format to be storage-compatible with the
/// base. Compatibility is compared as a whole [`pixel_format::BlockGeometry`] —
/// the grid *and* the bytes — rather than as a bytes-per-texel, because a
/// block-compressed format has no bytes-per-texel at all and comparing two
/// `None`s would have admitted a `BC1`-as-`BC3` reinterpretation while refusing
/// every compressed texture outright.
///
/// That refusal was not theoretical: before this took the block form, **every**
/// BC texture failed here as `linear_load_view_bpp_mismatch` — with no view
/// override in play, because the base alone has no texel width — so a compressed
/// bind never reached the loader at all.
///
/// Unknown formats (no [`pixel_format::block_geometry`]) fail visibly. `None`
/// override inherits base.
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
    let base_block = pixel_format::block_geometry(base_fmt)
        .ok_or(ViewSampleRefusal::BaseUndeclared { base: base_fmt })?;
    let sample_block = pixel_format::block_geometry(sample)
        .ok_or(ViewSampleRefusal::ViewUndeclared { view: sample })?;
    if base_block.bytes != sample_block.bytes {
        return Err(ViewSampleRefusal::WidthMismatch {
            base_bpp: base_block.bytes,
            view_bpp: sample_block.bytes,
        });
    }
    // Equal bytes over a different grid is its own refusal — see
    // [`ViewSampleRefusal::GridMismatch`].
    if (base_block.width, base_block.height) != (sample_block.width, sample_block.height) {
        return Err(ViewSampleRefusal::GridMismatch {
            base_compressed: base_block.is_compressed(),
            view_compressed: sample_block.is_compressed(),
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

/// Load a linear (buffer-backed) texture for sampling, keeping the layout its
/// bytes are in.
///
/// A caller that passes anything but [`NativeUploads::NONE`] must carry the
/// returned layout all the way to the bind: the bytes are then the guest's own
/// and their length is `width * height * layout.bytes_per_texel()`, which for
/// the half-float arms is not `width * height * 4`.
// Eight, for [`load_linear_texture_impl`]'s reason: the last two are the
// caller's own answers — which native layouts it carries, and which rung it is
// — and neither is derivable here. Bundling the descriptor selectors into a
// struct would hide which of them a call site varies.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_linear_texture_host<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    level: u32,
    format_override: Option<u16>,
    native: NativeUploads,
    site: crate::runtime::render_writeback::SettleSite,
) -> Option<(Vec<u8>, SampledByteFormat)> {
    load_linear_texture_impl(
        state,
        host,
        task_id,
        texture_ref,
        level,
        0,
        format_override,
        native,
        site,
    )
    .ok()
}

/// Load all six faces of a cube texture, tightly packed face after face — the
/// layer order `VkBufferImageCopy` consumes with `layerCount = 6`.
///
/// Metal stores a cube as a six-slice array, and this device's measured
/// array-packing rule (see [`load_linear_texture_impl`]'s `face` parameter) puts
/// each face one face-stride after the last. Each face rides the ordinary 2D
/// loader, so every conversion arm — native BC blocks, BGRA8, the RGBA8 convert
/// fallback — serves a cube exactly as it serves the equivalent 2D texture, and
/// a divergence between the two is not expressible.
///
/// Level 0 only, which is what the sampled bind path uploads for every shape
/// (the engine creates its sampled images with one mip). The faces must agree
/// on their byte layout by construction — one descriptor, one format — so the
/// per-face formats are asserted equal rather than reconciled.
// The Vulkan bind path is the only caller; the Metal arm's cube support is
// native and never reloads faces on the CPU, so ungated this is dead code
// there — which the cross-compiled clippy run is what catches.
#[cfg(feature = "backend-vulkan")]
pub(crate) fn load_cube_faces<M: HostMemory + HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    native: NativeUploads,
    site: crate::runtime::render_writeback::SettleSite,
) -> Result<(Vec<u8>, SampledByteFormat), LinearLoadRefusal> {
    /// Faces of a cube. Not configurable: five is not a cube and neither is
    /// seven, and the engine refuses `cube` images whose layer count is not
    /// exactly this.
    const CUBE_FACES: u32 = 6;
    let mut packed: Vec<u8> = Vec::new();
    let mut format: Option<SampledByteFormat> = None;
    for face in 0..CUBE_FACES {
        let (bytes, byte_format) = load_linear_texture_impl(
            state,
            host,
            task_id,
            texture_ref,
            0,
            face,
            None,
            native,
            site,
        )?;
        match format {
            None => {
                // Sized once, from the first face: the five remaining reads of
                // one descriptor cannot legally differ in length.
                packed.reserve_exact(bytes.len() * (CUBE_FACES as usize - 1));
                format = Some(byte_format);
            }
            Some(first) => {
                // One descriptor, one format — a disagreement here is this
                // loader's own bug, not the guest's, so it is a refusal rather
                // than a debug assertion that vanishes in release.
                if first.layout() != byte_format.layout() {
                    return Err(LinearLoadRefusal::RowConvertUnsupported {
                        format: 0,
                    });
                }
            }
        }
        packed.extend_from_slice(&bytes);
    }
    let format = format.ok_or(LinearLoadRefusal::ZeroExtent {
        width: 0,
        height: 0,
    })?;
    Ok((packed, format))
}

/// Which non-RGBA8 sampled layouts a caller of the linear loaders will carry.
///
/// A loader that hands back the guest's own bytes hands back a layout whose
/// bytes-per-texel need not be four, and not every caller can take one: the
/// colour-LOAD seed rail discards the layout entirely and reads the bytes as
/// RGBA8, so it must be able to say it takes none of them. That is what
/// [`Self::NONE`] is for, and it is why this is a parameter rather than a fact
/// this module could work out for itself.
///
/// Two independent questions decide each field, and both belong to the caller:
/// whether it can carry the layout at all, and whether this host can sample and
/// filter the matching format — which `engine::supports_sampled_layout_linear_filter`
/// answers and a backend-independent module cannot ask.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct NativeUploads {
    /// Upload guest BGRA8 as `B8G8R8A8_UNORM` and let the sampler read the
    /// channels in the order the guest stored them, instead of running a
    /// full-image CPU channel swap.
    pub bgra8: bool,
    /// Upload guest half-float colour (`RGBA16Float`, `RG16Float`) at its own
    /// eight- or four-byte footprint, as `R16G16B16A16_SFLOAT` /
    /// `R16G16_SFLOAT`. Unlike the BGRA8 case this is not a saved pass — it is
    /// the only exact path. The CPU arm for these goes through
    /// `f16_to_unorm8_lut`, which clamps to `[0, 1]` and quantizes to 256
    /// levels; see [`pixel_format::TexelLayout::cpu_loader_arm_is_lossy`].
    pub float16: bool,
    /// Upload the guest's BC (DXT / S3TC) blocks verbatim as the matching
    /// `VK_FORMAT_BC*_BLOCK`.
    ///
    /// Unlike the two above this is not a saved pass or an exactness question —
    /// it is the **only** path. There is no CPU decompressor here and there will
    /// not be one, so a host that clears this flag loses the texture and says
    /// so; see [`pixel_format::TexelLayout::has_cpu_loader_arm`], which answers
    /// `false` for every BC layout. Set from
    /// `engine::supports_block_compressed_sampled`, which is one Vulkan feature
    /// for the whole family. Desktop GPUs have it; Apple GPUs carry ASTC
    /// instead and read `false`.
    pub block_compressed: bool,
}

impl NativeUploads {
    /// Everything converts to RGBA8. The answer for any caller that keeps the
    /// bytes and drops the layout.
    pub const NONE: Self = Self {
        bgra8: false,
        float16: false,
        block_compressed: false,
    };

    /// Native BGRA8 only — the answer this parameter carried when it was a
    /// `bool`, kept so a caller that has not thought about the wider layouts
    /// says so rather than inheriting them.
    ///
    /// Gated because the only rail that opts into a native upload is the
    /// Vulkan one; the Metal arm's single caller drops the layout and takes
    /// [`Self::NONE`]. Ungated it is dead code on `backend-metal`, which the
    /// cross-compiled clippy run is what catches.
    #[cfg(any(feature = "backend-vulkan", test))]
    pub const BGRA8: Self = Self {
        bgra8: true,
        float16: false,
        block_compressed: false,
    };

    /// Every native layout the loaders can produce.
    ///
    /// Test-only on purpose. Production reaches this answer through
    /// `native_uploads_for_host`, which asks the host whether it can filter the
    /// half-float formats; a constant that says yes without asking is exactly
    /// the shape that would let a capability go unchecked.
    #[cfg(test)]
    pub const ALL: Self = Self {
        bgra8: true,
        float16: true,
        block_compressed: true,
    };
}

/// When a sampled format's guest bytes are ALREADY in the final upload order —
/// so the loader can read padded source rows straight into the tight output
/// with no intermediate buffer and no per-row convert — this returns the engine
/// upload format. `RGBA8` always qualifies (its convert is an identity copy);
/// the other classes qualify only where `native` says the caller carries them.
/// Every other format needs a real convert pass and returns `None`.
///
/// The returned layout's [`TexelLayout::bytes_per_texel`] is what sizes the
/// output — it is **not** four for the half-float arms, so a caller reading
/// this must size its rows from the layout and never from `RGBA8_BPP`.
pub(crate) fn linear_native_upload_format(
    sample_format: u16,
    native: NativeUploads,
) -> Option<TexelLayout> {
    use pixel_format::SampledClass;
    // The compressed families are answered before the sampled class, because
    // that vocabulary has no block in it and because the answer here is not an
    // optimisation: a BC bind is native or it is a refusal. `None` when the host
    // cannot sample the family sends it to the convert path, which declines by
    // name for a format with no CPU loader arm — which is what a refusal looks
    // like on this rail.
    if let Some(layout) = pixel_format::block_compressed_layout(sample_format) {
        return native.block_compressed.then_some(layout);
    }
    // The decode contract's sampled class is the one rule for "which channel
    // order and width is this"; it folds each sRGB format onto its linear
    // sibling's layout, which is right — they share a layout. The qualifier is
    // no longer lost by that fold: every caller pairs the layout it gets here
    // with `sample_format` into a [`SampledByteFormat`], and the bind applies
    // the transfer function from there.
    Some(match pixel_format::sampled_class(sample_format)? {
        SampledClass::Rgba8Unorm => TexelLayout::Rgba8,
        SampledClass::Bgra8Unorm if native.bgra8 => TexelLayout::Bgra8,
        SampledClass::Rgba16Float if native.float16 => TexelLayout::Rgba16Float,
        SampledClass::Rg16Float if native.float16 => TexelLayout::Rg16Float,
        _ => return None,
    })
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
    // Array slice / cube face to read. Slices pack as contiguous images at each
    // mip — the rule `blit_exec`'s array copies measured against live guest
    // traffic (opcode 0x12c, slices 1 and 2 at exact one-image offsets) — so a
    // face is the level's own read one face-stride further in. 0 is every
    // pre-existing caller, and everything below is unchanged for it.
    face: u32,
    format_override: Option<u16>,
    native: NativeUploads,
    site: crate::runtime::render_writeback::SettleSite,
) -> Result<(Vec<u8>, SampledByteFormat), LinearLoadRefusal> {
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
    let planes = layout.planes();
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
        .and_then(|n| n.checked_mul(u64::from(planes)))
        .and_then(|n| n.checked_mul(RGBA8_BPP as u64))
        .and_then(host_alloc_len)
        .ok_or(R::SizeOverflow)?;
    // Every depth plane belongs to the level; only the final plane's final-row
    // padding lies outside the bytes this loader reads.
    //
    // Counted in rows of storage rather than rows of texels, so a
    // block-compressed level does not claim four times its own extent and get
    // refused against the allocation the guest sized correctly.
    let storage_rows = pixel_format::tight_row_count(h, base_fmt).ok_or(R::FormatBppUnknown {
        format: base_fmt,
    })?;
    // One face-stride per slice past the level base. The stride is a whole
    // image *including* its final row's padding — the packing rule is
    // contiguous images, so face N+1 starts where face N's allocation ends,
    // not where its last read ends. Zero for face 0, which is every 2D caller.
    let face_off = if face == 0 {
        0
    } else {
        bpr.checked_mul(u64::from(storage_rows))
            .and_then(|stride| stride.checked_mul(u64::from(planes)))
            .and_then(|stride| stride.checked_mul(u64::from(face)))
            .ok_or(R::SizeOverflow)?
    };
    let gva = gva.checked_add(face_off).ok_or(R::SizeOverflow)?;
    let span = layout
        .slice_read_span_rows(storage_rows, tight)
        .ok_or(R::SizeOverflow)?;
    let end = layout.offset.saturating_add(face_off).saturating_add(span);
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
    // ones `slice_read_span` names — not `bpr * h * planes`, so a padded source
    // does not claim the final trailing padding it never touches.
    // Census, pay, settle — the whole obligation of a CPU read of one named
    // resource's guest bytes. See `writeback_debt::settle_for_texture`.
    crate::runtime::writeback_debt::settle_for_texture(
        state, host, task_id, texture_ref, gva, span, site,
    );
    // Tight display textures are the common compositor source. Read the whole
    // image with one task-root/cache lifetime: the row loop below otherwise
    // rebuilds the GVA walker cache once per row (1,080 times for the live
    // Safari source). Padded rows retain the conservative disjoint reads so we
    // never touch padding that the guest did not make readable.
    if bpr_u32 == tight {
        let (rgba, fmt) =
            load_tight_linear_rgba_with(w, h, planes, sample_fmt, native, site.route(), |dst| {
                gva_mem::read_task_gva_by_id(
                    host,
                    &state.tasks,
                    task_id,
                    gva,
                    dst,
                    state.page_shift,
                )
                .is_ok()
            })?;
        return Ok((rgba, fmt));
    }
    // Padded rows. When the source bytes are already in the final upload order
    // (RGBA8 always; BGRA8 and half-float colour under a native upload), read
    // each padded source row STRAIGHT into the tight output — no intermediate
    // row buffer, no per-row convert/swizzle pass. This is the Safari-scroll
    // fallback hot path (`lin_guest_fb`), so the elided convert pass is a full
    // second walk over the sampled bytes off the drain worker.
    //
    // The output is sized from the layout's own texel width and not from
    // `RGBA8_BPP`: the half-float arms are eight and four bytes a texel, and
    // `need_rgba` is the RGBA8 figure. The tight-row check is the same
    // agreement one step earlier — a source row that is not exactly one tight
    // row of the upload layout cannot be copied straight through.
    if let Some(fmt) = linear_native_upload_format(sample_fmt, native)
        .filter(|fmt| fmt.tight_row_bytes(w) == Some(tight))
    {
        let row_bytes = tight as usize;
        let rows = (fmt.tight_row_count(h) as usize)
            .checked_mul(planes as usize)
            .ok_or(R::SizeOverflow)?;
        let out_len = row_bytes.checked_mul(rows).ok_or(R::SizeOverflow)?;
        let mut rgba = vec![0u8; out_len];
        for row_index in 0..rows {
            let row_gva = (row_index as u64)
                .checked_mul(bpr)
                .and_then(|off| gva.checked_add(off))
                .ok_or(R::SizeOverflow)?;
            let dst_off = row_index.checked_mul(row_bytes).ok_or(R::SizeOverflow)?;
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
            .map_err(|_| R::PaddedRowUnreadable {
                row: u32::try_from(row_index).unwrap_or(u32::MAX),
            })?;
        }
        return Ok((rgba, SampledByteFormat::from_source(fmt, sample_fmt)));
    }
    let mut rgba = vec![0u8; need_rgba];
    let mut row = vec![0u8; tight as usize];
    let rows = storage_rows.checked_mul(planes).ok_or(R::SizeOverflow)?;
    for row_index in 0..rows {
        let row_gva = (row_index as u64)
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
        .map_err(|_| R::PaddedRowUnreadable { row: row_index })?;
        crate::runtime::draw::note_sampled_narrowing(site.route(), 0, sample_fmt, w, h);
        let dst_off = (row_index as usize) * (w as usize) * 4;
        if !pixel_format::convert_row_to_rgba8(sample_fmt, &row, w, &mut rgba[dst_off..]) {
            return Err(R::RowConvertUnsupported { format: sample_fmt });
        }
    }
    Ok((rgba, SampledByteFormat::from_source(TexelLayout::Rgba8, sample_fmt)))
}

///
/// `site` names the calling rung and is what a narrowing is reported under, so
/// a `RGBA8`-by-design seed read and a sampled bind that lost precision are two
/// census lines rather than one.
pub(crate) fn load_tight_linear_rgba_with<F>(
    width: u32,
    height: u32,
    planes: u32,
    sample_format: u16,
    native: NativeUploads,
    site: &'static str,
    mut read: F,
) -> Result<(Vec<u8>, SampledByteFormat), LinearLoadRefusal>
where
    F: FnMut(&mut [u8]) -> bool,
{
    use LinearLoadRefusal as R;
    let tight = pixel_format::tight_row_bytes(width, sample_format).ok_or(R::FormatBppUnknown {
        format: sample_format,
    })?;
    // Rows of storage, not rows of texels: a BC level is a quarter as tall in
    // blocks as it is in texels, rounded up. `tight_row_count` is the one
    // spelling of that and it answers `height` for every uncompressed format,
    // so this is the same read it always was for them.
    let rows = pixel_format::tight_row_count(height, sample_format)
        .ok_or(R::FormatBppUnknown {
            format: sample_format,
        })?
        .checked_mul(planes)
        .ok_or(R::SizeOverflow)?;
    let native_len = (tight as u64)
        .checked_mul(rows as u64)
        .and_then(host_alloc_len)
        .ok_or(R::SizeOverflow)?;
    let rgba_stride = width.checked_mul(RGBA8_BPP).ok_or(R::SizeOverflow)?;
    let rgba_len = (rgba_stride as u64)
        .checked_mul(rows as u64)
        .and_then(host_alloc_len)
        .ok_or(R::SizeOverflow)?;
    let mut bytes = vec![0u8; native_len];
    if !read(&mut bytes) {
        return Err(R::TightImageUnreadable);
    }
    // A format whose guest bytes are already the upload bytes is returned as
    // read, and the layout says what they are. This is decided from the format
    // alone and NOT from `native_len == rgba_len`, which is the shape this gate
    // used to have: that comparison is true only for a four-byte texel, so a
    // half-float source could never reach the arm that keeps it exact however
    // the classes were extended, and fell through to the quantizing convert
    // below every time.
    //
    // The BGRA8 arm additionally converts in place when the caller will not
    // take a native BGRA8 image, which is free of a second display-sized
    // allocation precisely because the two lengths agree there.
    if let Some(layout) = linear_native_upload_format(sample_format, native) {
        return Ok((bytes, SampledByteFormat::from_source(layout, sample_format)));
    }
    if native_len == rgba_len
        && pixel_format::sampled_class(sample_format) == Some(pixel_format::SampledClass::Bgra8Unorm)
    {
        // A channel exchange, which moves no value across the transfer
        // function: the bytes stay encoded exactly as the guest stored them.
        for pixel in bytes.chunks_exact_mut(RGBA8_BPP as usize) {
            pixel.swap(0, 2);
        }
        return Ok((
            bytes,
            SampledByteFormat::from_source(TexelLayout::Rgba8, sample_format),
        ));
    }
    // Keyed by the calling rung, not by "tight rows". Which rung narrowed is
    // the question a boot's census leaves open otherwise: the colour LOAD seed
    // reads RGBA8 by design and a narrowing there is expected, while one on the
    // sampled rung is a texture the guest is about to read back through a
    // shader. One shared slug reported both as the same event.
    crate::runtime::draw::note_sampled_narrowing(site, 0, sample_format, width, height);
    let mut rgba = vec![0u8; rgba_len];
    for y in 0..rows as usize {
        let src_off = y.checked_mul(tight as usize).ok_or(R::SizeOverflow)?;
        let dst_off = y.checked_mul(rgba_stride as usize).ok_or(R::SizeOverflow)?;
        if !pixel_format::convert_row_to_rgba8(
            sample_format,
            &bytes[src_off..src_off + tight as usize],
            width,
            &mut rgba[dst_off..dst_off + rgba_stride as usize],
        ) {
            return Err(R::RowConvertUnsupported {
                format: sample_format,
            });
        }
    }
    Ok((
        rgba,
        SampledByteFormat::from_source(TexelLayout::Rgba8, sample_format),
    ))
}

#[cfg(test)]
mod texture_view_split_tests {
    use super::*;
    use crate::runtime::decode::resource::TextureViewDescriptor;

    /// View compatibility is a whole storage grid, which is what lets a
    /// compressed texture past this gate at all.
    ///
    /// The blocker this pins: the check compared `bytes_per_pixel`, and a BC
    /// format has none — so **every** compressed texture was refused here as
    /// `linear_load_view_bpp_mismatch` with no view override in play, because
    /// the base format alone could not answer. A compressed bind never reached
    /// the loader, and it would have read as "BC is unsupported" rather than as
    /// this one gate.
    ///
    /// The two mismatch directions matter separately and the second is why this
    /// is a grid and not a byte count: `BC3_RGBA` and `RGBA32Float` are both
    /// sixteen bytes per addressable unit, and one of them spends them on
    /// sixteen texels.
    #[test]
    fn a_view_is_compatible_by_storage_grid_not_by_texel_width() {
        use crate::contract::pixel_format as pf;

        // No override: the base format alone must pass, which is the case that
        // was refusing every compressed texture.
        assert_eq!(
            effective_view_sample_format(pf::MTL_FORMAT_BC3_RGBA, None),
            Some(pf::MTL_FORMAT_BC3_RGBA)
        );
        for &format in &[
            pf::MTL_FORMAT_BC1_RGBA,
            pf::MTL_FORMAT_BC1_RGBA_SRGB,
            pf::MTL_FORMAT_BC4_R_SNORM,
            pf::MTL_FORMAT_BC6H_RGB_FLOAT,
            pf::MTL_FORMAT_BC7_RGBA_UNORM_SRGB,
        ] {
            assert_eq!(
                effective_view_sample_format(format, None),
                Some(format),
                "{format:#x} must pass its own compatibility gate"
            );
        }
        // The transfer-function view of one allocation is compatible: same grid,
        // same bytes.
        assert_eq!(
            effective_view_sample_format(
                pf::MTL_FORMAT_BC3_RGBA,
                Some(pf::MTL_FORMAT_BC3_RGBA_SRGB)
            ),
            Some(pf::MTL_FORMAT_BC3_RGBA_SRGB)
        );
        // Different weight class in the same grid: eight bytes a block is not
        // sixteen.
        assert!(matches!(
            effective_view_sample_format_reasoned(
                pf::MTL_FORMAT_BC1_RGBA,
                Some(pf::MTL_FORMAT_BC3_RGBA)
            ),
            Err(ViewSampleRefusal::WidthMismatch {
                base_bpp: 8,
                view_bpp: 16
            })
        ));
        // Same bytes, different grid — the case a byte-count comparison would
        // have admitted, and the reason `GridMismatch` exists.
        assert!(matches!(
            effective_view_sample_format_reasoned(
                pf::MTL_FORMAT_BC3_RGBA,
                Some(pf::MTL_FORMAT_RGBA32_FLOAT)
            ),
            Err(ViewSampleRefusal::GridMismatch {
                base_compressed: true,
                view_compressed: false
            })
        ));
        assert!(matches!(
            effective_view_sample_format_reasoned(
                pf::MTL_FORMAT_RGBA32_FLOAT,
                Some(pf::MTL_FORMAT_BC3_RGBA)
            ),
            Err(ViewSampleRefusal::GridMismatch {
                base_compressed: false,
                view_compressed: true
            })
        ));
        // And the uncompressed rule is unchanged in both directions.
        assert_eq!(
            effective_view_sample_format(
                pf::MTL_FORMAT_BGRA8_UNORM,
                Some(pf::MTL_FORMAT_RGBA8_UNORM)
            ),
            Some(pf::MTL_FORMAT_RGBA8_UNORM)
        );
        assert!(matches!(
            effective_view_sample_format_reasoned(
                pf::MTL_FORMAT_BGRA8_UNORM,
                Some(pf::MTL_FORMAT_RGBA16_FLOAT)
            ),
            Err(ViewSampleRefusal::WidthMismatch { .. })
        ));
    }

    /// A BC bind is native or it is nothing, and the host capability is what
    /// decides which.
    ///
    /// There is no CPU decompressor here, so unlike the BGRA8 and half-float
    /// flags this one does not choose between a fast path and a slow one — it
    /// chooses between the guest keeping its texture and losing it. That makes
    /// the gate load-bearing rather than a performance switch, which is why it
    /// is asserted in both directions: a `Some` on a host that cannot sample the
    /// family would create an image in a format the driver never advertised.
    #[test]
    fn a_compressed_bind_is_native_or_refused() {
        use crate::contract::pixel_format::{self as pf, TexelLayout};

        let capable = NativeUploads {
            block_compressed: true,
            ..NativeUploads::BGRA8
        };
        assert_eq!(
            linear_native_upload_format(pf::MTL_FORMAT_BC3_RGBA, capable),
            Some(TexelLayout::Bc3Rgba)
        );
        // The sRGB spelling folds onto the same layout — identical blocks — and
        // the qualifier is carried by `SampledByteFormat`'s source format.
        assert_eq!(
            linear_native_upload_format(pf::MTL_FORMAT_BC3_RGBA_SRGB, capable),
            Some(TexelLayout::Bc3Rgba)
        );
        // Every family, so a member admitted to the contract and forgotten here
        // fails rather than silently refusing at run time.
        for format in [
            pf::MTL_FORMAT_BC1_RGBA,
            pf::MTL_FORMAT_BC2_RGBA,
            pf::MTL_FORMAT_BC4_R_UNORM,
            pf::MTL_FORMAT_BC5_RG_SNORM,
            pf::MTL_FORMAT_BC6H_RGB_UFLOAT,
            pf::MTL_FORMAT_BC7_RGBA_UNORM,
        ] {
            assert_eq!(
                linear_native_upload_format(format, capable),
                pf::block_compressed_layout(format),
                "{format:#x} must reach the native rail on a capable host"
            );
            // A host without the family refuses, and `NativeUploads::BGRA8` is
            // exactly such a host: the flag defaults off.
            assert_eq!(
                linear_native_upload_format(format, NativeUploads::BGRA8),
                None,
                "{format:#x} must not be bound on a host that cannot sample it"
            );
            assert_eq!(linear_native_upload_format(format, NativeUploads::NONE), None);
        }
        // The gate is per family, not per call: an uncompressed format is
        // unaffected by it in either direction.
        assert_eq!(
            linear_native_upload_format(pf::MTL_FORMAT_BGRA8_UNORM, capable),
            Some(TexelLayout::Bgra8)
        );
        assert_eq!(
            linear_native_upload_format(pf::MTL_FORMAT_BGRA8_UNORM, NativeUploads::NONE),
            None
        );
    }

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
