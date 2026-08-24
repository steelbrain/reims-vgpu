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
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ViewResolve {
    /// Non-view base texture ref after walking the view chain (archive
    /// `REIMS_VGPU_RESOURCE_RESOLVE_MAX_VIEW_CHAIN` walk).
    pub(crate) base_texture_ref: u32,
    /// Range exposed by the resolved view, expressed in the final base
    /// texture's mip/layer namespace. `None` is the simple format-only form,
    /// which does not narrow the immediate base's range.
    pub(crate) range: Option<TextureViewRange>,
    /// Declared output texture type for a ranged view. A simple format-only
    /// view inherits the first ranged type beneath it, or the base texture's
    /// declaration when the whole chain is simple.
    pub(crate) texture_type: Option<u16>,
    /// Present when the view carries a swizzle form (opcode 0x1b); selectors already validated.
    pub(crate) swizzle: Option<pixel_format::SwizzlePlan>,
    /// Non-zero view pixel format from the descriptor (`@16`); `None` inherits the base format.
    pub(crate) pixel_format: Option<u16>,
}

impl ViewResolve {
    /// Translate a level/slice relative to this view into the final base
    /// texture. A simple view does not narrow either namespace.
    pub(crate) fn select(&self, level: u64, slice: u64) -> Option<(u64, u64)> {
        self.range
            .map_or_else(|| Some((level, slice)), |range| range.select(level, slice))
    }

    /// Return the one mip selected by a non-array view consumer.
    ///
    /// Consumers which cannot carry layers must ask this explicitly; resolving
    /// a view no longer discards the other subresources on their behalf.
    pub(crate) fn single_non_array_level(&self) -> Option<u32> {
        self.range
            .map_or(Some(0), TextureViewRange::single_non_array_subresource)
    }
}

/// A texture view's exact subresource range in its final base texture.
///
/// Mip levels and array slices remain separate because a 3D texture's depth
/// planes are not array slices. The wire carries all four terms as 64-bit
/// values and resolution preserves that width until a concrete consumer has a
/// narrower API boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TextureViewRange {
    pub(crate) level_base: u64,
    pub(crate) level_count: u64,
    pub(crate) slice_base: u64,
    pub(crate) slice_count: u64,
}

impl TextureViewRange {
    fn compose_over(self, inner: Self) -> Result<Self, RangeCompositionError> {
        let level_end = self
            .level_base
            .checked_add(self.level_count)
            .ok_or(RangeCompositionError::LevelOverflow)?;
        if level_end > inner.level_count {
            return Err(RangeCompositionError::LevelOutOfRange);
        }
        let slice_end = self
            .slice_base
            .checked_add(self.slice_count)
            .ok_or(RangeCompositionError::SliceOverflow)?;
        if slice_end > inner.slice_count {
            return Err(RangeCompositionError::SliceOutOfRange);
        }
        Ok(Self {
            level_base: inner
                .level_base
                .checked_add(self.level_base)
                .ok_or(RangeCompositionError::LevelOverflow)?,
            level_count: self.level_count,
            slice_base: inner
                .slice_base
                .checked_add(self.slice_base)
                .ok_or(RangeCompositionError::SliceOverflow)?,
            slice_count: self.slice_count,
        })
    }

    /// Select one subresource relative to this view.
    pub(crate) fn select(self, level: u64, slice: u64) -> Option<(u64, u64)> {
        if level >= self.level_count || slice >= self.slice_count {
            return None;
        }
        Some((
            self.level_base.checked_add(level)?,
            self.slice_base.checked_add(slice)?,
        ))
    }

    /// The currently supported sampled/compute import shape: one mip and the
    /// complete non-array slice domain.
    pub(crate) fn single_non_array_subresource(self) -> Option<u32> {
        (self.level_count == 1 && self.slice_base == 0 && self.slice_count == 1)
            .then(|| u32::try_from(self.level_base).ok())
            .flatten()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RangeCompositionError {
    LevelOverflow,
    LevelOutOfRange,
    SliceOverflow,
    SliceOutOfRange,
}

/// A specific refusal while resolving one type-8 texture-view chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TextureViewDecline {
    HopEntryMissing {
        texture_ref: u32,
    },
    HopObjectNotView {
        texture_ref: u32,
        object_type: ObjectKind,
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
    HopTextureTypeUnsupported {
        texture_ref: u32,
        texture_type: u16,
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
    ChainLevelOutOfRange {
        texture_ref: u32,
        outer_base: u64,
        outer_count: u64,
        inner_count: u64,
    },
    ChainLevelOverflow {
        texture_ref: u32,
    },
    ChainSliceOutOfRange {
        texture_ref: u32,
        outer_base: u64,
        outer_count: u64,
        inner_count: u64,
    },
    ChainSliceOverflow {
        texture_ref: u32,
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
            Self::HopTextureTypeUnsupported { .. } => "texture_view_hop_texture_type_unsupported",
            Self::HopSwizzleInvalid { .. } => "texture_view_hop_swizzle_invalid",
            Self::ChainSelfOrZero { .. } => "texture_view_chain_self_or_zero",
            Self::ChainLevelOutOfRange { .. } => "texture_view_chain_level_out_of_range",
            Self::ChainLevelOverflow { .. } => "texture_view_chain_level_overflow",
            Self::ChainSliceOutOfRange { .. } => "texture_view_chain_slice_out_of_range",
            Self::ChainSliceOverflow { .. } => "texture_view_chain_slice_overflow",
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
            Self::HopTextureTypeUnsupported {
                texture_ref,
                texture_type,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("texture_type", texture_type.to_string()),
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
            Self::ChainLevelOutOfRange {
                texture_ref,
                outer_base,
                outer_count,
                inner_count,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("outer_base", outer_base.to_string()),
                ("outer_count", outer_count.to_string()),
                ("inner_count", inner_count.to_string()),
            ],
            Self::ChainLevelOverflow { texture_ref } | Self::ChainSliceOverflow { texture_ref } => {
                vec![("texture_ref", texture_ref.to_string())]
            }
            Self::ChainSliceOutOfRange {
                texture_ref,
                outer_base,
                outer_count,
                inner_count,
            } => vec![
                ("texture_ref", texture_ref.to_string()),
                ("outer_base", outer_base.to_string()),
                ("outer_count", outer_count.to_string()),
                ("inner_count", inner_count.to_string()),
            ],
            Self::ChainOverflow { base, depth } => {
                vec![("base", base.to_string()), ("depth", depth.to_string())]
            }
        }
    }
}

crate::observe::decline_display!(TextureViewDecline);

impl std::error::Error for TextureViewDecline {}

/// Contract view-chain limit: nested type-8 views collapse to a non-view base.
///
/// This is the decoded contract's own bound, not a budget of ours: a chain that
/// needs a ninth hop is one the guest's own resolver would not have followed
/// either, so refusing it is fidelity rather than a shortfall. That makes it the
/// number **every** consumer uses through [`resolve_texture_view_reasoned`].
/// Blits, sampling, compute, and render-target resolution must not grow their
/// own chain walkers: doing so gives one wire object multiple meanings.
///
/// It is also what terminates a guest-built cycle. `A views B views A` is
/// expressible and neither walk carries a visited set; the chain simply runs out
/// of hops and refuses visibly. A cycle is malformed, so a refusal is the right
/// answer — the bound only has to stop the recursion, and the contract's own
/// depth already does.
pub(crate) const MAX_TEXTURE_VIEW_CHAIN: usize = 8;

#[derive(Clone, Copy, Debug)]
struct ViewHop {
    base_texture_ref: u32,
    range: Option<TextureViewRange>,
    texture_type: Option<u16>,
    swizzle: Option<pixel_format::SwizzlePlan>,
    pixel_format: Option<u16>,
}

/// Decode one type-8 hop (does not walk nested bases).
///
/// The `Result` carries a specific failure slug for the always-on fail log. No
/// wrapper collapses it at this level: the slug travels up through
/// [`resolve_texture_view_reasoned`]'s `?`, and [`resolve_texture_view`] is what
/// turns the whole walk into an `Option` for the hot path.
fn decode_texture_view_hop_reasoned<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Result<ViewHop, TextureViewDecline> {
    use crate::runtime::decode::resource::texture_type8_header;
    let resource = objects::resolve_resource(state, host, task_id, texture_ref).map_err(
        |rung| match rung {
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
        },
    )?;
    if resource.entry().kind != ObjectKind::TextureView {
        return Err(TextureViewDecline::HopObjectNotView {
            texture_ref,
            object_type: resource.entry().kind,
        });
    }
    let desc = resource.descriptor().as_ref();
    // Bytes visible before decode, for the len-mismatch / bad-opcode census.
    let (opcode, declared) = texture_type8_header(desc).unwrap_or((0, 0));
    let view = match objects::decoded_resource(&resource) {
        Ok(crate::runtime::decode::resource::Descriptor::TextureView(view)) => view,
        Err(reason) => {
            // Dump the full wire blob for an unknown texture-view opcode: this is the
            // only signal that reveals a new serializer variant (off the hot path —
            // fires only on a genuine decode failure).
            let hex: String = desc.iter().map(|b| format!("{b:02x}")).collect();
            return Err(TextureViewDecline::HopDecode {
                texture_ref,
                opcode,
                declared,
                descriptor_len: desc.len(),
                bytes_hex: hex,
                reason: *reason,
            });
        }
        Ok(_) => {
            return Err(TextureViewDecline::HopDecode {
                texture_ref,
                opcode,
                declared,
                descriptor_len: desc.len(),
                bytes_hex: String::new(),
                reason: crate::runtime::decode::resource::DecodeStatus::ErrUnsupported(
                    "res_texture_view_semantic_kind",
                ),
            });
        }
    };
    if view.base_texture_ref == 0 {
        return Err(TextureViewDecline::HopZeroBase {
            texture_ref,
            opcode,
        });
    }
    if view.carries_range()
        && !crate::runtime::decode::resource::texture_view_type_supported(view.texture_type)
    {
        return Err(TextureViewDecline::HopTextureTypeUnsupported {
            texture_ref,
            texture_type: view.texture_type,
        });
    }
    let range = view.carries_range().then_some(TextureViewRange {
        level_base: view.level_base,
        level_count: view.level_count,
        slice_base: view.slice_base,
        slice_count: view.slice_count,
    });
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
    Ok(ViewHop {
        base_texture_ref: view.base_texture_ref,
        range,
        texture_type: view.carries_range().then_some(view.texture_type),
        swizzle,
        pixel_format,
    })
}

/// Resolve a type-8 view to its non-view base, exact subresource range, format,
/// declared type, and swizzle.
///
/// The `Result` carries a specific failure slug (`reason=view_resolve` sub-case)
/// for the always-on fail log; [`resolve_texture_view`] collapses it to `Option`
/// for the hot path. Walks nested type-8 bases up to [`MAX_TEXTURE_VIEW_CHAIN`]
/// Each view is constructed on its immediate base object. Nested mip/layer
/// ranges and swizzles therefore compose; a walk that merely skips inner
/// descriptors changes the view's semantics.
pub(crate) fn resolve_texture_view_reasoned<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Result<ViewResolve, TextureViewDecline> {
    let outer = decode_texture_view_hop_reasoned(state, host, task_id, texture_ref)?;
    let mut base = outer.base_texture_ref;
    let mut range = outer.range;
    let mut texture_type = outer.texture_type;
    let mut swizzle = outer.swizzle;
    let mut pixel_format = outer.pixel_format;

    // Collapse nested type-8 bases to a non-view texture (IOSurface texture / type-2/3).
    let mut depth = 0u32;
    for _ in 1..MAX_TEXTURE_VIEW_CHAIN {
        let Some(entry) = objects::lookup_list_entry(state, host, task_id, base) else {
            // Base missing from the list — leave the ref for the caller to fail
            // visibly (same as a one-hop miss on an unmapped base).
            break;
        };
        if entry.kind != ObjectKind::TextureView {
            break;
        }
        depth += 1;
        let inner = decode_texture_view_hop_reasoned(state, host, task_id, base)?;
        let next = inner.base_texture_ref;
        if next == 0 || next == base {
            return Err(TextureViewDecline::ChainSelfOrZero { base, next, depth });
        }
        range = match (range, inner.range) {
            (Some(outer), Some(inner)) => {
                Some(outer.compose_over(inner).map_err(|error| match error {
                    RangeCompositionError::LevelOverflow => {
                        TextureViewDecline::ChainLevelOverflow { texture_ref: base }
                    }
                    RangeCompositionError::LevelOutOfRange => {
                        TextureViewDecline::ChainLevelOutOfRange {
                            texture_ref: base,
                            outer_base: outer.level_base,
                            outer_count: outer.level_count,
                            inner_count: inner.level_count,
                        }
                    }
                    RangeCompositionError::SliceOverflow => {
                        TextureViewDecline::ChainSliceOverflow { texture_ref: base }
                    }
                    RangeCompositionError::SliceOutOfRange => {
                        TextureViewDecline::ChainSliceOutOfRange {
                            texture_ref: base,
                            outer_base: outer.slice_base,
                            outer_count: outer.slice_count,
                            inner_count: inner.slice_count,
                        }
                    }
                })?)
            }
            (outer @ Some(_), None) => outer,
            (None, inner) => inner,
        };
        swizzle = match (swizzle, inner.swizzle) {
            (Some(outer), Some(inner)) => Some(outer.after(&inner)),
            (outer @ Some(_), None) => outer,
            (None, inner) => inner,
        };
        if pixel_format.is_none() {
            pixel_format = inner.pixel_format;
        }
        if texture_type.is_none() {
            texture_type = inner.texture_type;
        }
        base = next;
    }

    // Final base must not still be a type-8 view past the chain cap.
    if let Some(entry) = objects::lookup_list_entry(state, host, task_id, base) {
        if entry.kind == ObjectKind::TextureView {
            return Err(TextureViewDecline::ChainOverflow { base, depth });
        }
    }

    Ok(ViewResolve {
        base_texture_ref: base,
        range,
        texture_type,
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
    state: &Device,
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
///   `reims_vgpu_core::pixel_format` has no row for. **Ours.** Nothing about the bind
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

impl Decline for ViewSampleRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::BaseUndeclared { .. } => "texture_view_base_format_undeclared",
            Self::ViewUndeclared { .. } => "texture_view_format_undeclared",
            Self::WidthMismatch { .. } => "texture_view_format_width_mismatch",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::BaseUndeclared { base } => vec![("base_format", format!("{base:#x}"))],
            Self::ViewUndeclared { view } => vec![("view_format", format!("{view:#x}"))],
            Self::WidthMismatch { base_bpp, view_bpp } => vec![
                ("base_bpp", base_bpp.to_string()),
                ("view_bpp", view_bpp.to_string()),
            ],
        }
    }
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
/// deleted.
#[cfg(test)]
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
    NotATexture { object_type: ObjectKind },
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
    NoLevelGva { slice: u32, level: u32 },
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
            Self::NoLevelGva { slice, level } => {
                vec![("slice", slice.to_string()), ("level", level.to_string())]
            }
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

/// Load one `(slice, level)` subresource of a linear (buffer-backed) texture for
/// sampling, keeping the layout its bytes are in.
///
/// One subresource, never a whole texture: a caller that needs every array layer
/// or every cube face calls this once per physical slice and concatenates, which
/// is the only order the engine's tightly-packed layered expectation accepts.
/// The slice count belongs to the caller because it is the *bind* that says how
/// many layers it declared, and `LinearTextureDescriptor::physical_slice_count`
/// is what answers it.
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
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    slice: u32,
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
        slice,
        level,
        format_override,
        native,
        site,
    )
    .ok()
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
}

impl NativeUploads {
    /// Everything converts to RGBA8. The answer for any caller that keeps the
    /// bytes and drops the layout.
    pub const NONE: Self = Self {
        bgra8: false,
        float16: false,
    };

    /// Native BGRA8 only — the answer this parameter carried when it was a
    /// `bool`, kept so a caller that has not thought about the wider layouts
    /// says so rather than inheriting them.
    ///
    /// Gated because the only rail that opts into a native upload is the
    /// cross-compiled clippy run is what catches.
    pub const BGRA8: Self = Self {
        bgra8: true,
        float16: false,
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
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    slice: u32,
    level: u32,
    format_override: Option<u16>,
    native: NativeUploads,
    site: crate::runtime::render_writeback::SettleSite,
) -> Result<(Vec<u8>, SampledByteFormat), LinearLoadRefusal> {
    use LinearLoadRefusal as R;
    let resource = objects::resolve_resource(state, host, task_id, texture_ref).map_err(
        |rung| match rung {
            objects::LadderRung::NoListEntry => R::ObjectListMiss,
            objects::LadderRung::WrongType { got } => R::NotATexture { object_type: got },
            objects::LadderRung::DescRead { .. } => R::DescriptorUnreadable,
        },
    )?;
    if resource.entry().kind != ObjectKind::Texture {
        return Err(R::NotATexture {
            object_type: resource.entry().kind,
        });
    }
    let Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) =
        objects::decoded_resource(&resource)
    else {
        return Err(R::DescriptorUndecodable);
    };
    let base_fmt = tex.declared_pixel_format().ok_or(R::NoPixelFormat)?;
    let sample_fmt = effective_view_sample_format(base_fmt, format_override).ok_or(
        R::ViewFormatBppMismatch {
            base: base_fmt,
            view: format_override.unwrap_or(base_fmt),
        },
    )?;
    let (gva, layout) = tex
        .subresource_gva(slice, level, state.page_shift)
        .ok_or(R::NoLevelGva { slice, level })?;
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
    let span = layout.slice_read_span(tight).ok_or(R::SizeOverflow)?;
    // Bounded against the subresource this call actually reads. `base_offset +
    // layout.offset` is the same arithmetic only for slice 0; for any later
    // face or array layer it omits the inter-slice advance and would bound a
    // read of the last slice against the extent of the first.
    let end = tex
        .subresource_offset(slice, level)
        .and_then(|offset| offset.checked_add(span))
        .ok_or(R::SizeOverflow)?;
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
        state,
        host,
        task_id,
        texture_ref,
        gva,
        span,
        site,
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
        .filter(|fmt| (tight as u64) == (w as u64).saturating_mul(fmt.bytes_per_texel() as u64))
    {
        let row_bytes = tight as usize;
        let rows = (h as usize)
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
    let rows = h.checked_mul(planes).ok_or(R::SizeOverflow)?;
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
    Ok((
        rgba,
        SampledByteFormat::from_source(TexelLayout::Rgba8, sample_fmt),
    ))
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
    let rows = height.checked_mul(planes).ok_or(R::SizeOverflow)?;
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
        && pixel_format::sampled_class(sample_format)
            == Some(pixel_format::SampledClass::Bgra8Unorm)
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

    #[test]
    fn view_pixel_format_override_effective() {
        use reims_vgpu_core::pixel_format::{
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
        use reims_vgpu_core::pixel_format::{
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
        use reims_vgpu_core::pixel_format::{
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
            LinearLoadRefusal::NotATexture {
                object_type: ObjectKind::MemorylessTexture,
            },
            LinearLoadRefusal::DescriptorUnreadable,
            LinearLoadRefusal::DescriptorUndecodable,
            LinearLoadRefusal::NoPixelFormat,
            LinearLoadRefusal::ViewFormatBppMismatch { base: 1, view: 2 },
            LinearLoadRefusal::NoLevelGva { slice: 0, level: 3 },
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

    #[test]
    fn nested_ranges_compose_in_the_final_base_namespace() {
        let outer = TextureViewRange {
            level_base: 1,
            level_count: 2,
            slice_base: 2,
            slice_count: 3,
        };
        let inner = TextureViewRange {
            level_base: 4,
            level_count: 5,
            slice_base: 8,
            slice_count: 7,
        };
        assert_eq!(
            outer.compose_over(inner),
            Ok(TextureViewRange {
                level_base: 5,
                level_count: 2,
                slice_base: 10,
                slice_count: 3,
            })
        );
    }

    #[test]
    fn nested_ranges_refuse_each_out_of_bounds_axis_independently() {
        let inner = TextureViewRange {
            level_base: 4,
            level_count: 2,
            slice_base: 8,
            slice_count: 3,
        };
        assert_eq!(
            TextureViewRange {
                level_base: 1,
                level_count: 2,
                slice_base: 0,
                slice_count: 1,
            }
            .compose_over(inner),
            Err(RangeCompositionError::LevelOutOfRange)
        );
        assert_eq!(
            TextureViewRange {
                level_base: 0,
                level_count: 1,
                slice_base: 2,
                slice_count: 2,
            }
            .compose_over(inner),
            Err(RangeCompositionError::SliceOutOfRange)
        );
    }

    #[test]
    fn range_selection_never_flattens_an_array_or_mip_range() {
        let range = TextureViewRange {
            level_base: 3,
            level_count: 2,
            slice_base: 5,
            slice_count: 4,
        };
        assert_eq!(range.select(1, 3), Some((4, 8)));
        assert_eq!(range.select(2, 0), None);
        assert_eq!(range.select(0, 4), None);
        assert_eq!(range.single_non_array_subresource(), None);
    }
}
