//! Sampled resource resolution and guest-to-executor materialization.
//!
//! This module resolves guest resource identities and authoritative content into
//! semantic sampled-source requests. It may choose an executor capability path,
//! but it does not construct or submit a native draw.

use super::*;

pub(super) enum SampledSourceRequest {
    /// Shared texel bytes + optional producer identity (see
    /// [`LinearSampleIdentity`]) + what those texels are; the Arc lets memoized
    /// repeat binds skip the per-draw copy and the engine skip re-hashing.
    ///
    /// The third field is a [`SampledByteFormat`] and not a bare `TexelLayout`
    /// because a layout is linear by construction. While it was one, every CPU
    /// upload of an sRGB guest texture reached the sampler through a `_UNORM`
    /// view and was never decoded, while the zero-copy rails beside it — which
    /// carry a semantic image-view format — preserved sRGB and were. One
    /// guest texture, two colours, and which one it got decided by a cost
    /// threshold. Each producer answers from the format it *loaded from*, so a
    /// convert that reorders channels keeps the transfer function it never
    /// touched.
    Bytes(
        std::sync::Arc<Vec<u8>>,
        Option<LinearSampleIdentity>,
        SampledByteFormat,
        reims_vgpu_core::SampledByteOrigin,
    ),
    /// Engine-resident allocation plus the exact view format this sampled
    /// texture declared. Allocation identity and view interpretation are
    /// separate parts of the texture contract.
    Target(
        crate::model::TargetIdentity,
        reims_vgpu_protocol::ImageFormat,
    ),
    /// The render attachment and fragment binding name the same serialized
    /// texture. The engine either binds that image through native feedback or
    /// produces the capability fallback entirely on the GPU.
    Attachment(
        crate::model::TargetIdentity,
        reims_vgpu_core::AttachmentInitial,
        reims_vgpu_protocol::ImageFormat,
    ),
    /// Zero-copy guest gather: the engine copies the texel bytes from
    /// imported guest RAM inside the draw CB — no CPU read, no memo, no
    /// hash. Carries the native texel layout the image is created with.
    /// Guest-RAM runs the engine gathers from, the byte layout of those texels,
    /// an optional copied-content identity, and what the guest-write witness
    /// says that identity is worth (see [`crate::runtime::gather_witness`]).
    ///
    /// The last field is the **format's own** channel plan, not the guest's
    /// view swizzle: this rail binds guest bytes untouched, so a format whose
    /// Metal channels do not sit identically on the backend format carrying them
    /// needs that difference expressed on the image view. It is composed with
    /// the type-8 view swizzle at the push site. Identity for every format but
    /// `A8Unorm`.
    GuestRuns(
        reims_vgpu_memory::GuestRunSource,
        TexelLayout,
        /// Exact semantic image/view format. This is distinct from the texel
        /// layout because linear and sRGB formats carry the same bytes while
        /// applying different fixed-function sampling conversions.
        reims_vgpu_protocol::ImageFormat,
        /// Consecutive depth planes carried by the source window.
        u32,
        Option<LinearSampleIdentity>,
        crate::runtime::gather_witness::GatherVouch,
        pixel_format::SwizzlePlan,
    ),
    /// One mapped image backing with both its direct-image and transfer
    /// materializations. The backend selects between them from its image-layout
    /// contract; the runtime does not turn storage identity into page runs.
    GuestImage(
        reims_vgpu_memory::GuestImageSource,
        reims_vgpu_protocol::ImageFormat,
        Option<LinearSampleIdentity>,
        crate::runtime::gather_witness::GatherVouch,
        pixel_format::SwizzlePlan,
    ),
}

/// Producer identity + generation for CPU-sourced sampled bytes, so that equal
/// identity implies equal bytes under the same coherence model the producing
/// cache already relies on.
///
/// `key` is namespaced by its top two bits, because four producers share one
/// keyspace and a raw id would alias between them:
///
/// | bit 63 | bit 62 | producer | low bits |
/// |---|---|---|---|
/// | 0 | 0 | guest linear | the texture's authoritative GVA (`host_gva_surfaces`) |
/// | 1 | 0 | IOSurface plane view view | `plane_index << 32 \| mapping_id` |
/// | 0 | 1 | IOSurface texture host cache | `mapping_id` (`host_surfaces`) |
/// | 1 | 1 | IOSurface texture guest memo | `mapping_id` (`iosurface_texture_memo`) |
///
/// GVAs are well under 2^62, so the unflagged row cannot collide with a flagged
/// one. `generation` comes from
/// [`crate::runtime::Device::next_sampled_content_generation`] for every one
/// of them — a device-global counter, never per-entry — so a `(key, generation)`
/// pair names one content for the life of the device and content cannot alias
/// even if two producers did collide on a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LinearSampleIdentity {
    pub(super) key: u64,
    pub(super) generation: u64,
}

impl From<crate::runtime::gather_witness::GatheredIdentity> for LinearSampleIdentity {
    /// The zero-copy gather rail's key is a hash of the window's name rather
    /// than a bit-namespaced id like the four rows above, so it can collide with
    /// any of them. That is harmless for the same reason the table's last
    /// paragraph gives: the generation comes from the one device-global counter,
    /// which issues a value once and never again, so a `(key, generation)` pair
    /// names one content even when two producers agree on a key.
    fn from(id: crate::runtime::gather_witness::GatheredIdentity) -> Self {
        Self {
            key: id.key,
            generation: id.generation,
        }
    }
}

pub(super) type LoadedIOSurfacePlaneView = (
    u32,
    u32,
    std::sync::Arc<Vec<u8>>,
    LinearSampleIdentity,
    SampledByteFormat,
);
pub(super) type LoadedLinearSample = (
    u32,
    u32,
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    SampledByteFormat,
);

/// Resolve a fragment binding that names one of this pass's color textures.
///
/// The render-pass descriptor and the fragment binding serialize the same
/// texture reference. The attachment's load action is therefore the complete
/// statement of its initial contents; CPU seed availability, backing kind and
/// chain position do not participate in deciding whether it aliases.
pub(super) fn fragment_attachment_alias_initial(
    req: &DrawEncodeRequest,
    texture_index: u32,
    texture_ref: u32,
) -> Option<(u32, u32, reims_vgpu_core::AttachmentInitial)> {
    let color = req
        .colors
        .iter()
        .find(|color| color.slot == texture_index && color.texture_ref == texture_ref)?;
    use reims_vgpu_core::AttachmentInitial;
    match color.load_action {
        reims_vgpu_protocol::pass_action::LoadAction::Clear => Some((
            color.width,
            color.height,
            AttachmentInitial::Clear([
                color.clear_color[0] as f32,
                color.clear_color[1] as f32,
                color.clear_color[2] as f32,
                color.clear_color[3] as f32,
            ]),
        )),
        reims_vgpu_protocol::pass_action::LoadAction::Load => {
            Some((color.width, color.height, AttachmentInitial::Seed))
        }
        reims_vgpu_protocol::pass_action::LoadAction::DontCare => {
            Some((color.width, color.height, AttachmentInitial::DontCare))
        }
    }
}

pub(super) fn attachment_alias_source(
    identity: crate::model::TargetIdentity,
    format: reims_vgpu_protocol::ImageFormat,
    initial: reims_vgpu_core::AttachmentInitial,
) -> SampledSourceRequest {
    SampledSourceRequest::Attachment(identity, initial, format)
}

/// Resolve the texture construction object behind a sampled `texture_ref`.
///
/// A texture bind names a texture object. Descriptor length or byte shape does
/// not widen that contract: another object kind with a sufficiently long body
/// is still not a texture, and interpreting it as one would let unrelated
/// fields become image geometry. This is the same typed object-list rung used
/// by mipmap, texture-view, compute, and render-target resolution.
pub(super) fn sampled_texture_descriptor<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    texture_ref: u32,
) -> Option<(
    crate::runtime::decode::resource::ListObjectEntry,
    reims_vgpu_protocol::LinearTextureDescriptor,
)> {
    let resource = objects::resolve_resource(state, host, task_id, texture_ref).ok()?;
    let entry = resource.entry();
    if entry.kind != ObjectKind::Texture {
        return None;
    }
    let Ok(crate::runtime::decode::resource::Descriptor::Texture(descriptor)) =
        objects::decoded_resource(&resource)
    else {
        return None;
    };
    Some((entry, descriptor.clone()))
}

/// Resolve a sampled texture ref to `(width, height, mapping_id, source)`.
///
/// Backend-neutral: the returned [`SampledSourceRequest`] is either an engine
/// target to bind directly (zero-copy) or CPU bytes to upload, so this is the
/// resolver the engine draw path uses. Distinct from [`load_sampled_rgba_static`],
/// which always materializes RGBA8 bytes.
///
/// # The IOSurface texture ladder is measured, and every rung carries load
///
/// Four rungs offer the same IOSurface texture surface, and the obvious reading is that
/// three of them are redundant with the first. They are not. A DRIVEN x86/Vulkan
/// session — four Safari page loads, each scrolled six pages and then dragged by
/// its title bar — split as:
///
///   iosurfacerung_resident         31 916   93.0 %   engine image, taken zero-copy
///   iosurfacerung_host_cache        1 694    4.9 %   surface_cache's BGRA mirror
///   iosurfacerung_zero_copy           705    2.1 %   guest pages, gathered
///   iosurfacerung_guest_memo          150    0.4 %   guest pages, CPU convert
///   iosurfacerung_miss                  0             no source at all
///   iosurfacerung_resident_refused        2            guest overwrote the resident
///
/// Measure this on a DRIVEN session or not at all. The same census on an
/// undriven boot to the desktop reported 12 / 5 / 8 / 0, which is far too quiet
/// to tell a rung that never fires from one the boot never reached — and quiet
/// enough to talk someone into deleting it.
///
/// A second driven session with a different drive — Chess, Maps, Safari on the
/// WebGL aquarium, Wikipedia and apple.com, page scrolls and two title-bar
/// drags — reproduced the shape on a smaller population:
///
///   iosurfacerung_resident         15 992   64.5 %
///   iosurfacerung_host_cache        5 777   23.3 %
///   iosurfacerung_zero_copy         3 036   12.2 %
///   iosurfacerung_guest_memo           62    0.25 %
///   iosurfacerung_miss                  0
///   iosurfacerung_resident_refused      0
///
/// The order is the same and no rung is empty, so the two runs agree on which
/// rungs carry load. The share does move with the drive — live 3D and a WebGL
/// canvas push work down off the resident rung — so treat the percentages as a
/// range and not as a constant of the design. The bottom rung is the one to
/// keep watching: 150 binds in one session and 62 in the other is small enough
/// to read as noise, and both are a fallback nothing below would correct.
///
/// Two facts to weigh before touching the order:
///
/// - The host-cache rung is NOT a duplicate of the guest-page rungs below it.
///   A render Store defers its writeback into guest pages, so between the Store
///   and its flush the cache is the only host-side copy that holds the new
///   pixels; the pages still hold the old ones. Its 1 694 binds are that window.
/// - `iosurfacerung_resident_refused` firing twice is not evidence the guest-write
///   witness is dead weight. Those are the binds where the guest CPU painted
///   over a surface the engine still claimed to hold, and the rung sits above
///   both page-reading rungs, so nothing below would have corrected it. Two
///   uncorrected stale binds on a repainted surface is the "renders correctly
///   for a few frames, then stays corrupted" report.
///
/// The currency ladder distinguishes an engine allocation from a resident over
/// the guest allocation. A guest write stales the former; it updates the same
/// bytes in the latter and changes the next Vulkan dependency to `HOST_WRITE`.
struct IOSurfaceResidentCurrency {
    identity: crate::model::TargetIdentity,
    read: reims_vgpu_core::ResidentReadPlan,
    ready: bool,
    current: bool,
}

fn iosurface_resident_currency(
    state: &mut Device,
    resource: Option<&std::sync::Arc<crate::model::TaskResource>>,
    mapping_id: u32,
    width: u32,
    height: u32,
    resident_identity: Option<crate::model::TargetIdentity>,
) -> IOSurfaceResidentCurrency {
    let resident_id = resident_identity.unwrap_or_else(|| {
        crate::runtime::present_identity::surface_identity(state, mapping_id, width, height)
    });
    let resident_read = state.executor.resident_read_plan(&resident_id);
    let resident_backing = resource
        .filter(|resource| resource_type_owns_surface_resident(resource.entry().kind))
        .map(|resource| {
            state
                .executor
                .retain_resident_resource(resource.lifetime_ref(), &resident_id)
        })
        .unwrap_or(resident_read.backing);
    let resident_ready = resident_backing != reims_vgpu_core::ResidentContentBacking::NotReady;
    if resident_ready {
        crate::runtime::drain::note_store_route(match resident_backing {
            reims_vgpu_core::ResidentContentBacking::GuestAllocation => {
                "iosurfacesample_ready_guest_allocation"
            }
            reims_vgpu_core::ResidentContentBacking::DeviceAllocation => {
                "iosurfacesample_ready_device_allocation"
            }
            reims_vgpu_core::ResidentContentBacking::NotReady => {
                unreachable!("resident_ready excludes this arm")
            }
        });
    }
    let mapping_epoch = state
        .surfaces
        .mappings
        .get(&mapping_id)
        .map(|mapping| mapping.content.surface_epoch);
    let mut resident_epoch = resident_ready
        .then_some(resident_read.content_epoch)
        .flatten();
    if let Some(epoch) = mapping_epoch.filter(|epoch| {
        resident_backing == reims_vgpu_core::ResidentContentBacking::GuestAllocation
            && Some(*epoch) != resident_epoch
    }) {
        if state
            .executor
            .note_resident_guest_write(&resident_id, epoch)
        {
            resident_epoch = Some(epoch);
            note_iosurface_texture_sample_rung("iosurfacerung_guest_allocation_resynced");
        } else {
            crate::observe::fail(format!(
                "iosurface_sample_fail reason=guest_allocation_resync_refused mapping={mapping_id} epoch={epoch}"
            ));
        }
    }
    IOSurfaceResidentCurrency {
        identity: resident_id,
        read: resident_read,
        ready: resident_ready,
        current: resident_ready
            && iosurface_texture_resident_is_current(mapping_epoch, resident_epoch),
    }
}

pub(crate) fn compute_iosurface_resident_sample<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    mapping_id: u32,
    width: u32,
    height: u32,
) -> Option<crate::model::TargetIdentity> {
    let resource = objects::resolve_resource(state, host, task_id, texture_ref).ok();
    let plane_identity = resource
        .as_ref()
        .filter(|resource| resource.entry().kind == ObjectKind::IOSurfacePlaneView)
        .and_then(|resource| match objects::decoded_resource(resource) {
            Ok(crate::runtime::decode::resource::Descriptor::IOSurfacePlaneView(descriptor)) => {
                descriptor.view
            }
            _ => None,
        })
        .and_then(|view| {
            reims_vgpu_core::pixel_format::store_texel_order(view.pixel_format).map(|format| {
                crate::runtime::present_identity::surface_plane_identity(
                    state,
                    mapping_id,
                    view.plane_index,
                    width,
                    height,
                    format,
                )
            })
        });
    let currency = iosurface_resident_currency(
        state,
        resource.as_ref(),
        mapping_id,
        width,
        height,
        plane_identity,
    );
    currency.current.then_some(currency.identity)
}

pub(super) fn resolve_sampled_source<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    resource: Option<std::sync::Arc<crate::model::TaskResource>>,
    may_bind_resident: bool,
    sampled_shape: reims_vgpu_core::SampledImageShape,
) -> Option<(u32, u32, u32, SampledSourceRequest)> {
    if texture_ref == 0 {
        return None;
    }
    let direct_image_2d = sampled_shape.layers == 1
        && !sampled_shape.arrayed
        && !sampled_shape.volume
        && !sampled_shape.cube
        && !sampled_shape.one_dim
        && !sampled_shape.multisampled;

    // Opcode-9 buffer-backed texture (type-8): the sampled bytes are an MTLBuffer's
    // guest storage, not a view over another texture. Resolve it directly before
    // the view/surface paths (which would mis-decode the opcode-9 descriptor).
    // `resource` (when supplied by the caller) serves every classification and
    // descriptor consumer below from the one retained object.
    if let Some(bt) =
        buffer_texture_descriptor(state, host, task_id, texture_ref, resource.as_deref())
    {
        // The opcode-9 descriptor's own pixel format. The loader converts to
        // RGBA8 order and decodes nothing, so the transfer function the guest
        // declared is still the one these bytes carry.
        let source = bt.desc.pixel_format;
        let (w, h, rgba) = load_buffer_texture_rgba(state, host, task_id, texture_ref, &bt)?;
        return Some((
            w,
            h,
            0,
            SampledSourceRequest::Bytes(
                std::sync::Arc::new(rgba),
                None,
                SampledByteFormat::from_source(TexelLayout::Rgba8, source),
                reims_vgpu_core::SampledByteOrigin::BufferBackedTexture,
            ),
        ));
    }

    // The object list names exactly ONE surface for a sampled ref. Which one is
    // decided by the entry's `object_type`, and the cases below are distinct
    // values of that single u8 field, so they cannot both apply:
    //
    //   IOSurface plane view RefTextureHandle — carries the surface backing surface id in its
    //                             descriptor, alongside the Metal texture view
    //   surface backing Surface         — the ref *is* the surface id
    //   IOSurface      — resolves to the mapping id it was created on
    //
    // The retained resource already carries the total typed decode. The
    // IOSurface texture wire tag can
    // therefore fill the slot directly from that object rather than looking up
    // and decoding the same reference a second time. It returns `None` for every
    // other type, so it can only fill a slot the classification above left empty.
    // Hence an `Option`, not a list of candidates to choose between.
    let mut is_linear_tex = false;
    let mut is_iosurface_plane_view = false;
    let mut iosurface_plane_view: Option<objects::IOSurfacePlaneViewDescriptor> = None;
    let mut surface: Option<u32> = None;
    let resolved_resource =
        resource.or_else(|| objects::resolve_resource(state, host, task_id, texture_ref).ok());

    // A texture view is a semantic image over its retained base, not a byte
    // loader that happens to know how to find that base. Resolve the chain once
    // and offer the selected mip to the same exact-layout import rail as a base
    // texture. The view object remains the draw's lifetime owner in the caller;
    // the packed allocation and content witness are owned by the final base.
    if resolved_resource.as_ref().is_some_and(|resource| {
        resource.entry().kind == ObjectKind::TextureView
            && matches!(
                objects::decoded_resource(resource),
                Ok(crate::runtime::decode::resource::Descriptor::TextureView(_))
            )
    }) {
        if let Ok(view) = resolve_texture_view_reasoned(state, host, task_id, texture_ref) {
            let view_level = view
                .range
                .map_or(Some(0), |range| u32::try_from(range.level_base).ok())?;
            if let Ok(base_resource) =
                objects::resolve_resource(state, host, task_id, view.base_texture_ref)
            {
                if let Ok(crate::runtime::decode::resource::Descriptor::Texture(base)) =
                    objects::decoded_resource(&base_resource)
                {
                    let selection = LinearSampleSelection {
                        level: view_level,
                        pixel_format: view.pixel_format,
                        texture_type: view.texture_type,
                        range: view.range,
                    };
                    let exact_shape = base.level(view_level).is_some_and(|level| {
                        declared_guest_image_selection(
                            sampled_shape,
                            base,
                            level,
                            view.texture_type,
                            view.range,
                        )
                        .is_some()
                    });
                    if exact_shape {
                        if let Some((w, h, source)) = try_linear_sample_zero_copy(
                            state,
                            host,
                            task_id,
                            view.base_texture_ref,
                            base,
                            sampled_shape,
                            selection,
                        ) {
                            crate::runtime::drain::note_store_route("zc_lin_texture_view");
                            return Some((w, h, 0, source));
                        }
                    }
                    if refuse_unmaterialized_mip_range(texture_ref, base, selection) {
                        return None;
                    }
                }
            }
        }
    }
    if let Some(resource) = resolved_resource.as_ref() {
        let entry = resource.entry();
        if entry.kind == ObjectKind::IOSurfacePlaneView {
            is_iosurface_plane_view = true;
            if let Ok(crate::runtime::decode::resource::Descriptor::IOSurfacePlaneView(t5)) =
                objects::decoded_resource(resource)
            {
                let sid = t5.surface.get();
                if sid != 0 {
                    iosurface_plane_view = t5.view;
                    surface = Some(sid);
                }
            }
        }
        if entry.kind == ObjectKind::SurfaceBacking {
            surface = Some(texture_ref);
        }
        if entry.kind == ObjectKind::Texture {
            is_linear_tex = true;
        }
    }
    if !is_iosurface_plane_view {
        // Runs for the linear and unclassified types too, not only for the
        // IOSurface texture hit it can return: the resolve records the ref as live in the
        // task's object set and reports a typed failure when the descriptor is
        // unreadable. Both are wanted for any ref a draw sampled.
        surface = surface.or_else(|| {
            resolved_resource.as_ref().and_then(|resource| {
                objects::resolve_iosurface_texture_resource(state, task_id, texture_ref, resource)
            })
        });
    }

    if let Some(mid) = surface {
        // Ensure surface backing pages exist for this surface id.
        let _ = objects::ensure_surface_for_texture_bind(state, host, mid);
        // A IOSurface plane view serialized record is the exact Metal texture view over the
        // IOSurface bytes. Materialize it only when it differs from (or cannot
        // be inferred from) the base mapping. Exact base views keep the fast
        // resident/cache path below; an unknown 2-B/texel base FourCC exposed
        // as RG8 must instead use the serialized view's native interpretation.
        // `iosurface_plane_view` is set only on the branch that also set `surface` to that
        // view's own surface id, so reaching here with a view in hand already
        // means `mid` is the surface it describes.
        if let Some(view) = iosurface_plane_view {
            let needs_materialization = state
                .surfaces
                .mappings
                .get(&mid)
                .map(|m| {
                    iosurface_plane_view_requires_materialization(
                        m.has_geometry(),
                        m.width_or_zero(),
                        m.height_or_zero(),
                        m.format_or_zero(),
                        view,
                    )
                })
                .unwrap_or(true);
            if needs_materialization {
                // Zero-copy the decoded plane straight from guest pages when
                // it samples byte-identically (video NV12 R8/RG8, BGRA8/
                // RGBA8). This bypasses the ~1.5 MB/plane/frame CPU read +
                // upload the CPU loader below would pay every decoded frame.
                if let Some(src) = resolved_resource.as_ref().and_then(|_| {
                    try_iosurface_plane_view_sample_zero_copy(
                        state,
                        host,
                        mid,
                        view,
                        direct_image_2d,
                    )
                }) {
                    // Success path: a healthy video decodes ~2 planes/frame,
                    // so this fires per-bind (~99k lines/boot). The aggregate
                    // lives in `sampled_branch_census` (`t5_zc=count:bytes`),
                    // which is the always-on signal; keep the per-bind detail
                    // for deep debugging behind REIMS_VGPU_DRAW_LOG (observe::line)
                    // rather than flooding the always-on fail sink.
                    crate::observe::line(format!(
                        "iosurface_plane_view_zc ref={texture_ref} sid={mid} view={}x{} fmt={:#x} plane={}",
                        view.width, view.height, view.pixel_format, view.plane_index
                    ));
                    return Some((view.width, view.height, mid, src));
                }
                let (w, h, rgba, identity, byte_format) =
                    load_iosurface_plane_view_rgba(state, host, task_id, texture_ref, mid, view)?;
                return Some((
                    w,
                    h,
                    mid,
                    SampledSourceRequest::Bytes(
                        rgba,
                        Some(identity),
                        byte_format,
                        reims_vgpu_core::SampledByteOrigin::SerializedSurfaceView,
                    ),
                ));
            }
        }
        if let Some(m) = state.surfaces.mappings.get(&mid) {
            if m.has_geometry() && m.width_or_zero() > 0 && m.height_or_zero() > 0 {
                let (w, h) = (m.width_or_zero(), m.height_or_zero());
                let declared_format = m.format_or_zero();
                // Compute the resident-surface identity once and reuse it for
                // both the readiness check and the direct bind.
                let currency =
                    iosurface_resident_currency(state, resolved_resource.as_ref(), mid, w, h, None);
                let resident_id = currency.identity;
                let resident_read = currency.read;
                // `content_ready` only. The obvious strengthening — also require
                // the resident's `content_epoch` to match the mapping's, as the
                // attachment LOAD elision does — was tried and reverted, because
                // `content_epoch` is not free for this rung to reinterpret. The
                // deferred flush uses the same field as its own identity check
                // ("is this still the resident my window was armed on"), so
                // withholding the stamp to disqualify a resident *here* made
                // every later window on that resident report
                // `resident_epoch_drift` and drop its frame: `deferred_flush_lost`
                // went from 17 to 3 161 on one boot and the screen stayed black.
                //
                // Separating the two meanings is worth doing and is not this
                // change. What guards this rung today is the guest-write witness
                // below, which is the witness the LOAD elision's epoch pair
                // cannot supply anyway.
                let resident_ready = currency.ready;
                // The serialized validity quad advances the mapping epoch when
                // the guest declares a CPU write. A resident is current only
                // when its Store stamp still equals that contract-owned epoch.
                let resident_current = currency.current;

                // A bind whose view remaps channels cannot take a resident
                // directly — the engine hands the swizzle to the image view and
                // the direct bind has none — so it falls straight to a byte rung
                // that can apply it.
                if resident_current && !may_bind_resident {
                    note_iosurface_texture_sample_rung("iosurfacerung_resident_swizzled");
                } else if resident_current {
                    note_iosurface_texture_sample_rung("iosurfacerung_resident");
                    let declared = if declared_format == 0 {
                        pixel_format::MTL_FORMAT_BGRA8_UNORM
                    } else {
                        declared_format
                    };
                    let format = pixel_format::sampled_image_format(declared)?;
                    return Some((w, h, mid, SampledSourceRequest::Target(resident_id, format)));
                } else if resident_ready {
                    note_iosurface_texture_sample_rung("iosurfacerung_resident_refused");
                }

                // Falling through because the resident is *gone* is not the same
                // as falling through because it is stale, and until this line
                // the two were indistinguishable here —
                // `resident_content_ready` is `is_some_and(content_ready)`, so
                // absent and not-ready-yet are both `false`.
                //
                // The stale case above is sound because it merges the
                // resident's half into the pages first, and refuses when that
                // merge does not land, for the reason its own comment gives: the
                // pages below then hold only the guest's half, "which for a
                // composite the Store deliberately left GPU-side is nothing at
                // all". A reclaimed resident has exactly that property and there
                // is nothing left to merge from — the image is destroyed — yet
                // it takes this fall-through with no merge and no refusal.
                //
                // For most surfaces that is correct: an IOSurface texture surface's pages
                // are its content, the flush rails write them, and reading them
                // back is what `resolve_iosurface_texture_load_seed` already calls "a cache
                // miss is a reason to read them".
                //
                // # The unsound case this line was added to count is closed
                //
                // It used to be reachable. A resident whose pixels were never
                // written to those pages at all — an MRT secondary attachment,
                // never pinned, never written back, and still carrying a real
                // `Gva`/`Surface` identity — could be aged out, and serving it
                // from its pages substituted an unrelated earlier frame.
                //
                // `ResidentTargetSlot::gpu_only_content` closed it at the
                // reclaim end, which is the only end that can be closed: both
                // allocation-pressure recovery skips such a slot at any
                // population. So **a resident that reaches this arm was, by
                // construction, not the sole copy of its pixels when it was
                // reclaimed** — something had copied them out, which is what
                // cleared the flag.
                //
                // The guarantee therefore does not live here and cannot be
                // asserted here; it lives beside those two selectors, held by
                // `elapsed_time_never_reclaims_a_live_resident`,
                // `the_capacity_walk_finds_no_victim_rather_than_destroy_the_only_copy`
                // and `no_reclaim_cause_may_take_the_only_copy_of_a_frame` —
                // the last of which is exhaustive over `ResidentReclaim`, so a
                // fourth way to lose a resident has to answer this question
                // before it compiles.
                //
                // What the line below still reports is a **cost**, not a
                // soundness risk: this device paid for the reclaim by re-reading
                // guest pages. It stays on the fail channel because the reclaim
                // cutoff is a measured trade (see `IDLE_MAINTENANCE_START_MS`) and the
                // reliance on it should stay visible, not because a firing means
                // something was lost.
                if !resident_ready {
                    if let Some((cause, since_ms)) = resident_read.absent_after_reclaim {
                        crate::runtime::drain::note_store_route(
                            "iosurfacesample_reclaimed_from_pages",
                        );
                        // How long after we destroyed it the guest came back.
                        // This is the half `resident_resample_peak_ms` cannot
                        // see: that peak only observes residents that survived
                        // to be read, so every gap longer than the cutoff is
                        // censored out of it and a reclaim policy tuned from it
                        // is tuned from data it destroyed the tail of. A
                        // resident read here had gone at least
                        // `IDLE_MAINTENANCE_START_MS + since_ms` between uses.
                        crate::runtime::drain::note_store_route(reclaimed_resample_band(
                            since_ms,
                            state.executor.idle_reclaim_start_ms(),
                        ));
                        if crate::observe::first_sight("sampled_resident_reclaimed", u64::from(mid))
                        {
                            crate::observe::fail(format!(
                                "sampled_resident_reclaimed reason=sampled_resident_reclaimed \
                                 mid={mid} {w}x{h} prior={} since_reclaim_ms={since_ms} \
                                 (reclaimed after its pixels were copied out; re-reading \
                                 its guest pages costs an upload, not a frame)",
                                cause.slug()
                            ));
                        }
                    }
                }

                // 1) Host cache — the other host-side copy of these pages, and
                // so gated on exactly the same witness as the resident above.
                // It sits above both rungs that read the guest's own pages, so a
                // stale hit is never corrected by anything below it; falling
                // through costs a re-read and reaches content that is
                // authoritative by construction.
                //
                // No content scan gates this. What stood here counted non-black
                // pixels (2 073 600 per bind at 1080p) and let the count decide
                // which image got bound — `runtime/census/README.md` forbids
                // exactly that, and an all-black frame is a legal frame, so the
                // test mistook a correct black surface for an empty one.
                if let Some((bgra, host_gen)) =
                    crate::runtime::surface_cache::get_shared_with_gen(state, mid, w, h)
                {
                    // Uploaded in the order the cache already holds. This rung's
                    // bytes are BGRA8 by construction — it is a surface backing scanout
                    // cache — and `B8G8R8A8_UNORM` is a Vulkan-mandatory sampled
                    // format with linear filtering, so declaring the layout costs
                    // nothing and the hardware reads the channels the guest
                    // stored. What stood here rebuilt the whole frame into RGBA8
                    // first: a 1.7 MB allocation plus a full read+write pass on
                    // every bind, ~116 binds a second live, to reach bytes the
                    // sampler could already address. The linear rail reached the
                    // same conclusion (`linear_native_upload_format(.., true)`).
                    //
                    // The view swizzle applied at bind is a *logical* channel
                    // remap from the guest descriptor and composes with the
                    // physical format rather than substituting for it, so this
                    // does not double-swap.
                    //
                    // The generation is the cache entry's own `host_gen`, which
                    // every writer of `host_surfaces` re-takes from
                    // `next_sampled_content_generation` in the same breath as it
                    // changes the bytes — so an unchanged pair is a statement
                    // that the frame has not moved, and the engine can bind what
                    // it already holds instead of re-digesting 1.7 MB to find
                    // out. See [`LinearSampleIdentity`] for the key namespace.
                    //
                    // A 0 generation is an entry never stored into.
                    // `get_shared_with_gen` already refuses those — it requires
                    // bytes — but a false "unchanged" is the one wrong answer
                    // here that binds a stale frame, so it is not left to that
                    // alone.
                    let identity = (host_gen != 0).then_some(LinearSampleIdentity {
                        key: (1u64 << 62) | mid as u64,
                        generation: host_gen,
                    });
                    note_iosurface_texture_sample_rung("iosurfacerung_host_cache");
                    // BGRA8 by construction — it is a surface backing scanout cache — but
                    // the *values* in it are the surface's, and this cache is
                    // filled from a writeback that reorders channels and decodes
                    // nothing. So the transfer function is the mapping's declared
                    // one, exactly as it is on the guest-page rungs below.
                    let source = crate::runtime::draw::mapping_declared_format(state, mid, None);
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(
                            bgra,
                            identity,
                            SampledByteFormat::from_source(TexelLayout::Bgra8, source),
                            reims_vgpu_core::SampledByteOrigin::SurfaceHostCache,
                        ),
                    ));
                }

                // 2) Guest pages, which are what the surface *is*. Reached only
                // when no host-side copy served the bind — no resident, or one
                // the guest has written over — so the gather always runs and the
                // guest bytes are taken unconditionally. Declining the gather is
                // expected control flow — the CPU byte loader below serves the
                // same pixels — so it stays quiet, like the type-2/3 rail's.
                if let Some(src) = resolved_resource.as_ref().and_then(|_| {
                    try_iosurface_texture_sample_zero_copy(state, host, mid, w, h, direct_image_2d)
                }) {
                    note_iosurface_texture_sample_rung("iosurfacerung_zero_copy");
                    return Some((w, h, mid, src));
                }
                // The memo skips the convert/alloc on unchanged content and
                // returns a content identity so the engine skips re-hash+upload;
                // its census (IOSurfaceMemo hit / IOSurfaceGuest fill) is emitted internally.
                let memo_source = crate::runtime::draw::mapping_declared_format(state, mid, None);
                if let Some((rgba, identity)) =
                    load_iosurface_texture_rgba_memoized(state, host, mid)
                {
                    note_iosurface_texture_sample_rung("iosurfacerung_guest_memo");
                    return Some((
                        w,
                        h,
                        mid,
                        SampledSourceRequest::Bytes(
                            rgba,
                            Some(identity),
                            SampledByteFormat::from_source(TexelLayout::Rgba8, memo_source),
                            reims_vgpu_core::SampledByteOrigin::SurfaceGuestFallback,
                        ),
                    ));
                }

                {
                    // A sample that resolved to no bytes anywhere is a lost
                    // guest command at any geometry: an app-window layer paints
                    // blank exactly as a full-screen one does. Latched per
                    // (mid, geometry) so a steady repeat stays at one line.
                    use std::collections::HashSet;
                    use std::sync::Mutex;
                    note_iosurface_texture_sample_rung("iosurfacerung_miss");
                    static SEEN: Mutex<Option<HashSet<(u32, u32, u32)>>> = Mutex::new(None);
                    let mut guard = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                    if guard.get_or_insert_with(HashSet::new).insert((mid, w, h)) {
                        crate::observe::fail(format!(
                            "sample_src=miss ref={texture_ref} mid={mid} {w}x{h} resident_ready={} (no guest/cache/resident bytes)",
                            resident_ready as u8
                        ));
                    }
                }
            }
        }
    }

    // Type-2/3: GVA-keyed encode, then texture_ref with **descriptor** geom match.
    if is_linear_tex {
        // Zero-copy gather for large Vulkan-native linear textures: replaces
        // the CPU host-cache/memo byte paths below for eligible formats (the
        // lin_memo full-window re-read + memcmp per bind was the dominant
        // per-draw cost under compositor load).
        // The object map retains the typed construction descriptor for the
        // resource lifetime. Both linear loaders consume that same object here;
        // neither needs to revisit guest construction bytes.
        if let Some((resource, tex)) = resolved_resource.as_ref().and_then(|resource| {
            match crate::runtime::objects::decoded_resource(resource) {
                Ok(crate::runtime::decode::resource::Descriptor::Texture(tex)) => {
                    Some((resource.as_ref(), tex))
                }
                _ => None,
            }
        }) {
            let selection = LinearSampleSelection::default();
            let (_, requested_mip_count) = requested_linear_mip_range(tex, selection);
            // Above the gather: a span whose pages a render Store published and
            // nothing has written since is already an engine image, so there is
            // nothing to gather and — the point of the rung — no writeback to
            // wait for. See [`try_gva_resident_sample`].
            if may_bind_resident && requested_mip_count == 1 {
                if let Some((w, h, src)) =
                    try_gva_resident_sample(state, host, task_id, texture_ref, resource, tex)
                {
                    note_linear_sample_geometry(
                        state,
                        task_id,
                        texture_ref,
                        tex,
                        LinearSampleRung::Resident,
                    );
                    return Some((w, h, 0, src));
                }
            }
            if let Some((w, h, src)) = try_linear_sample_zero_copy(
                state,
                host,
                task_id,
                texture_ref,
                tex,
                sampled_shape,
                selection,
            ) {
                note_linear_sample_geometry(
                    state,
                    task_id,
                    texture_ref,
                    tex,
                    LinearSampleRung::ZeroCopy,
                );
                return Some((w, h, 0, src));
            }
            if refuse_unmaterialized_mip_range(texture_ref, tex, selection) {
                return None;
            }
            if let Some((w, h, rgba, identity, byte_format)) =
                load_linear_from_host_caches(state, host, task_id, texture_ref, tex)
            {
                note_linear_sample_geometry(
                    state,
                    task_id,
                    texture_ref,
                    tex,
                    LinearSampleRung::HostRead,
                );
                return Some((
                    w,
                    h,
                    0,
                    SampledSourceRequest::Bytes(
                        rgba,
                        identity,
                        byte_format,
                        reims_vgpu_core::SampledByteOrigin::LinearTexture,
                    ),
                ));
            }
            // Every rung of the linear fork declined. The bind is not lost --
            // the last-resort rung below still reads the guest's bytes -- but
            // it reads them on the CPU, per bind, and none of the three rails
            // this fork exists to choose between served it. Recorded with the
            // same geometry line as the three that do, so a boot can say which
            // textures land here rather than only how many.
            note_linear_sample_geometry(
                state,
                task_id,
                texture_ref,
                tex,
                LinearSampleRung::Unserved,
            );
        }
    }

    // The last-resort sampled rung. The geometry comes from the decoded texture
    // descriptor and from nowhere else. Neither a payload shorter than the
    // descriptor's own extent nor a descriptor naming no extent at all is a
    // geometry this call may invent one for: the caller turns `None` into a
    // typed `DrawPreparationDecline::TextureResolveMissing`, which names the ref
    // and the stage. `TextureDescriptor::extent` owns the second check and says
    // what clamping the two fields up would have bound instead.
    //
    // The layout is carried rather than assumed. This rung answered
    // `TexelLayout::Rgba8` unconditionally and sized its own length check at
    // four bytes a texel, so it was the one place a half-float texture could
    // still be quantised after every rail above it learned not to — and it is
    // the rung a `RGBA16Float` display-profile LUT actually lands on, because
    // the three rungs above are reached only for a resource the draw already
    // knows is a linear texture.
    // Every physical slice the descriptor declares, concatenated in slice order.
    // A cube is six faces and a texture array is its declared length, both of
    // them at `bytes_per_slice` advances rather than tightly packed, which is
    // why this loops instead of scaling one read up: a six-times-longer read
    // from face 0's offset returns five faces of whatever the allocation holds
    // next. The engine expects `width * height * layers * texel` with the slices
    // packed end to end, so the concatenation here is the layout it validates
    // against.
    //
    // `physical_slice_count` is the contract term — `slice_count` expanded by
    // the six faces a cube dimension record declares — and it is 1 for an
    // ordinary 2D texture, which is what every caller of this rung used to get
    // by reading slice 0 alone.
    let (_entry, tex) = sampled_texture_descriptor(state, host, task_id, texture_ref)?;
    let (w, h) = tex.extent()?;
    let planes = tex.levels.first()?.planes();
    let slices = tex.physical_slice_count()?;
    let mut bytes: Vec<u8> = Vec::new();
    let mut layout = None;
    for slice in 0..slices {
        let (face, face_layout) = load_sampled_rgba_static(
            state,
            host,
            task_id,
            texture_ref,
            slice,
            native_uploads_asking_host(state.executor.as_ref()),
            crate::runtime::render_writeback::SettleSite::LinearTextureSampled,
        )?;
        // Every slice of one texture shares its format; a rung that answered a
        // different layout for a later face would silently splice two texel
        // sizes into one buffer.
        if *layout.get_or_insert(face_layout) != face_layout {
            return None;
        }
        bytes.extend_from_slice(&face);
    }
    let layout = layout?;
    let need = (w as usize)
        .saturating_mul(h as usize)
        .saturating_mul(planes as usize)
        .saturating_mul(slices as usize)
        .saturating_mul(layout.layout().bytes_per_texel() as usize);
    if bytes.len() < need {
        return None;
    }
    bytes.truncate(need);
    Some((
        w,
        h,
        0,
        SampledSourceRequest::Bytes(
            std::sync::Arc::new(bytes),
            None,
            layout,
            reims_vgpu_core::SampledByteOrigin::LinearTexture,
        ),
    ))
}

/// Whether a decoded resource object owns the resident reached by the surface
/// branch above.
///
/// Base surfaces, texture views, and IOSurface textures are three construction
/// forms of a texture object. Each carries one stable resource reference for
/// its lifetime; a view additionally names its parent, but that does not make
/// its own reference transient. Other object kinds can reach this resolver as
/// probes, but cannot produce the `surface` value whose resident is retained.
fn resource_type_owns_surface_resident(kind: ObjectKind) -> bool {
    matches!(
        kind,
        ObjectKind::SurfaceBacking | ObjectKind::IOSurfacePlaneView | ObjectKind::IOSurfaceTexture
    )
}

/// Whether a decoded resource object owns a linear GVA texture resident.
///
/// Normal textures and their serialized variants carry one stable resource
/// reference from construction until deletion. Their level-zero GVA identity
/// may therefore retain the engine allocation for that same lifetime. Surface
/// texture forms use [`resource_type_owns_surface_resident`] instead, while an
/// anonymous attachment keeps the registry-query fallback.
fn resource_type_owns_gva_resident(kind: ObjectKind) -> bool {
    kind == ObjectKind::Texture
}

#[cfg(test)]
mod resource_resident_ownership_tests {
    use super::*;

    #[test]
    fn every_surface_texture_construction_form_owns_its_resident() {
        for object_type in [
            ObjectKind::SurfaceBacking,
            ObjectKind::IOSurfacePlaneView,
            ObjectKind::IOSurfaceTexture,
        ] {
            assert!(resource_type_owns_surface_resident(object_type));
        }
        for object_type in [
            ObjectKind::Buffer,
            ObjectKind::Texture,
            ObjectKind::Function,
        ] {
            assert!(!resource_type_owns_surface_resident(object_type));
        }
    }

    #[test]
    fn linear_texture_construction_forms_own_their_gva_resident() {
        assert!(resource_type_owns_gva_resident(ObjectKind::Texture));
        for object_type in [
            ObjectKind::SurfaceBacking,
            ObjectKind::IOSurfacePlaneView,
            ObjectKind::IOSurfaceTexture,
            ObjectKind::Buffer,
        ] {
            assert!(!resource_type_owns_gva_resident(object_type));
        }
    }
}

#[inline]
pub(super) fn iosurface_plane_view_requires_materialization(
    base_has_geom: bool,
    base_width: u32,
    base_height: u32,
    base_format: u16,
    view: objects::IOSurfacePlaneViewDescriptor,
) -> bool {
    !base_has_geom
        || view.depth != 1
        || base_format == 0
        || base_width != view.width
        || base_height != view.height
        || base_format != view.pixel_format
}

/// The decoded device-surface fields a failed sample-window derivation dumps
/// for diagnosis: `(width, height, pixel_format, bytes_per_row, alloc_size)`.
type SampleWindowDesc = (u32, u32, u32, u32, u32);

/// Why the IOSurface plane view serialized-view loader refused to materialize a plane.
///
/// # Why these slugs are prefixed `iosurface_plane_view_`
///
/// The blit rail's `BlitStatus` already owns a `t5_*` vocabulary for the IOSurface plane view
/// *copy* path (`t5_no_mapping`, `t5_sample_window`, `t5_fmt_bpp`,
/// `t5_unmapped`), and four of this loader's checks are conceptually the same
/// words. A bare `no_mapping` was in fact one of three claimants — console
/// capture, guest-page import and this loader — that the last present-rail
/// migration recorded as still sharing the word. The `iosurface_plane_view_` prefix keeps
/// `grep reason=iosurface_plane_view_…` answerable against the copy path that shares the
/// surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IOSurfacePlaneViewDecline {
    /// The serialized view is volumetric; only `depth == 1` planes materialize.
    UnsupportedDepth { depth: u32 },
    /// The mapping's page table is not resident for scanout.
    Unresolved,
    /// The view's MTLPixelFormat has no known bytes-per-pixel.
    FormatBpp,
    /// The mapping id has no live entry.
    NoMapping,
    /// No sample window could be derived from the device descriptor for this
    /// plane geometry. Carries the base geometry and the decoded descriptor (or
    /// its absence) that disagreed.
    SampleWindow {
        base_w: u32,
        base_h: u32,
        base_fmt: u16,
        desc: Option<SampleWindowDesc>,
    },
    /// The mapping's resident pages span fewer bytes than the sample window
    /// ends at.
    Span {
        pages: usize,
        page_bytes: u64,
        span_end: u64,
        bpr: u32,
    },
    /// `width * bpp` overflowed a u32, so a tight row is unrepresentable.
    TightOverflow { bpp: u32 },
    /// The native plane byte length overflowed the host allocation cap.
    NativeLen { tight: u32 },
    /// The native plane window could not be read from guest memory.
    Read {
        base_w: u32,
        base_h: u32,
        base_fmt: u16,
        off: u64,
        bpr: u32,
        span_end: u64,
        pages: usize,
    },
    /// `width * 4` overflowed a u32, so the RGBA row is unrepresentable.
    RgbaStride,
    /// The RGBA buffer length overflowed the host allocation cap.
    RgbaLen { stride: u32 },
    /// A row failed to convert from the native format into RGBA8.
    Convert { row: usize, bpp: u32 },
}

impl crate::observe::Decline for IOSurfacePlaneViewDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::UnsupportedDepth { .. } => "iosurface_plane_view_unsupported_depth",
            Self::Unresolved => "iosurface_plane_view_unresolved",
            Self::FormatBpp => "iosurface_plane_view_format_bpp",
            Self::NoMapping => "iosurface_plane_view_no_mapping",
            Self::SampleWindow { .. } => "iosurface_plane_view_sample_window",
            Self::Span { .. } => "iosurface_plane_view_span",
            Self::TightOverflow { .. } => "iosurface_plane_view_tight_overflow",
            Self::NativeLen { .. } => "iosurface_plane_view_native_len",
            Self::Read { .. } => "iosurface_plane_view_read",
            Self::RgbaStride => "iosurface_plane_view_rgba_stride",
            Self::RgbaLen { .. } => "iosurface_plane_view_rgba_len",
            Self::Convert { .. } => "iosurface_plane_view_convert",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnsupportedDepth { depth } => vec![("depth", depth.to_string())],
            Self::SampleWindow {
                base_w,
                base_h,
                base_fmt,
                desc,
            } => {
                let mut v = vec![
                    ("base", format!("{base_w}x{base_h}")),
                    ("base_fmt", format!("{base_fmt:#x}")),
                ];
                match desc {
                    Some((w, h, fmt, bpr, alloc)) => {
                        v.push(("desc", format!("{w}x{h}")));
                        v.push(("desc_fmt", format!("{fmt:#x}")));
                        v.push(("bpr", bpr.to_string()));
                        v.push(("alloc", alloc.to_string()));
                    }
                    None => v.push(("desc", "missing".to_string())),
                }
                v
            }
            Self::Span {
                pages,
                page_bytes,
                span_end,
                bpr,
            } => vec![
                ("pages", pages.to_string()),
                ("page_bytes", page_bytes.to_string()),
                ("span_end", span_end.to_string()),
                ("bpr", bpr.to_string()),
            ],
            Self::TightOverflow { bpp } => vec![("bpp", bpp.to_string())],
            Self::NativeLen { tight } => vec![("tight", tight.to_string())],
            Self::Read {
                base_w,
                base_h,
                base_fmt,
                off,
                bpr,
                span_end,
                pages,
            } => vec![
                ("base", format!("{base_w}x{base_h}")),
                ("base_fmt", format!("{base_fmt:#x}")),
                ("off", off.to_string()),
                ("bpr", bpr.to_string()),
                ("span_end", span_end.to_string()),
                ("pages", pages.to_string()),
            ],
            Self::RgbaLen { stride } => vec![("stride", stride.to_string())],
            Self::Convert { row, bpp } => {
                vec![("row", row.to_string()), ("bpp", bpp.to_string())]
            }
            Self::Unresolved | Self::FormatBpp | Self::NoMapping | Self::RgbaStride => Vec::new(),
        }
    }
}

/// Why an IOSurface texture attachment `LOAD` could not be seeded with the surface's own
/// prior contents.
///
/// This is not a degradation the caller absorbs. `exec` resolves the pass load
/// action as "explicit `load_op` > `target_rgba8` > **Clear**", so a seed of
/// `None` makes `PassKey::single(load = false)` and the render pass begins with
/// `LoadOp::CLEAR` against the hardcoded `[0,0,0,0]` primary clear value. The
/// guest asked for its surface to be preserved and got a transparent-black wipe,
/// and the matching Store then reads that wipe back and publishes it. On a
/// compositor doing a damage-rect redraw that is one whole layer rendering solid
/// black — the reported black-rectangle class, whose screenshots show sharp
/// axis-aligned rectangles at layer boundaries.
///
/// It had no report of any kind. `surface_cache::get_shared` returns `Option` and
/// the arm simply left `target_rgba8` unset, so the loss was invisible on the
/// always-on channel. Measured on one x86/Vulkan boot before the guest-pages rung
/// existed: **121 distinct (mapping, geometry) wipes** in ~170 s, four of them at
/// the full 1920x1080 composite extent, against 0 in the idle phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IOSurfaceSeedDecline {
    /// The cache holds no entry for this mapping id and the mapping's own pages
    /// could not be read at the requested extent either.
    ///
    /// This is the whole population the pre-fix boot measured: every one of the
    /// 121 lines carried `hostgen=0`, and every one had `want == mapgeom`, which
    /// is what said the guest pages were readable and made the fallback rung the
    /// fix rather than a guess.
    NoEntry,
    /// An entry exists but at a different geometry, so the exact-geometry hit
    /// rule refuses it. `host_surfaces` keeps exactly one entry per mapping and
    /// every Store replaces it, so a Store at another geometry orphans every
    /// window still living at this one.
    ///
    /// Fired **0** times on that boot. Kept because it is a different check with
    /// a different fix (the entry is stale, not missing), and folding it into
    /// `NoEntry` would hide which one a future boot hit.
    GeomMismatch { have_w: u32, have_h: u32 },
}

impl crate::observe::Decline for IOSurfaceSeedDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoEntry => "iosurface_texture_seed_cache_absent",
            Self::GeomMismatch { .. } => "iosurface_texture_seed_cache_geom",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoEntry => Vec::new(),
            Self::GeomMismatch { have_w, have_h } => vec![("have", format!("{have_w}x{have_h}"))],
        }
    }
}

/// Which rung of the IOSurface texture `LOAD` seed ladder produced the attachment's prior
/// contents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IOSurfaceSeedRung {
    /// The host render cache held this mapping at exactly this geometry.
    Cache,
    /// The cache missed and the surface's own guest IOSurface pages were read.
    GuestPages,
}

/// An IOSurface texture attachment's prior contents, kept in the representation of the
/// freshest rung that supplied them.
pub(super) enum IOSurfaceLoadSeed {
    /// Host-owned bytes from the render cache, or the universal converted
    /// fallback when the native guest-page view cannot be described.
    Host(std::sync::Arc<Vec<u8>>, reims_vgpu_core::SeedOrder),
    /// The mapping's native texels as bounded guest-RAM runs. The engine imports
    /// them when possible and uses the runs themselves for its CPU fallback.
    Guest(reims_vgpu_memory::GuestTargetSeed),
}

impl IOSurfaceSeedRung {
    fn name(self) -> &'static str {
        match self {
            Self::Cache => "cache_hit",
            Self::GuestPages => "guest_pages",
        }
    }
}

/// Report which way the IOSurface texture `LOAD` seed branch went, once per
/// `(mapping, requested geometry, outcome)`.
///
/// Every outcome reports, because a zero on the miss arm has to be readable. A
/// probe that only fires on failure cannot separate "the cache always hit" from
/// "this branch never ran", and the branch is reached only for `mapping_id != 0`
/// under `MTL_LOAD_ACTION_LOAD` with no caller-supplied seed. With the served
/// arms beside it, an absent miss line next to present hit lines is evidence
/// rather than silence.
///
/// Naming the *rung* rather than just hit/miss is what prices the fallback: a
/// `guest_pages` line is a cache miss that was recovered, and its rate is the
/// only thing that says whether the recovery is cheap. Fusing it into `cache_hit`
/// would make the fix unmeasurable the moment it worked.
///
/// The mapping's own latched geometry and generation ride along on every arm:
/// `want == mapgeom` is the condition under which the guest-pages rung can serve
/// at all, so the pair says whether a miss was recoverable.
pub(super) fn note_iosurface_texture_load_seed(
    state: &Device,
    mapping_id: u32,
    w: u32,
    h: u32,
    served: Option<IOSurfaceSeedRung>,
) {
    let (map_w, map_h, map_gen) = state
        .surfaces
        .mappings
        .get(&mapping_id)
        .map(|m| {
            (
                m.width_or_zero(),
                m.height_or_zero(),
                m.lifecycle.generation,
            )
        })
        .unwrap_or((0, 0, 0));
    let cached = state.host_replicas.surface(mapping_id);
    let have = cached.map(|e| (e.width, e.height));
    let host_gen = cached.map(|e| e.host_gen).unwrap_or(0);
    // Latch before building the line: `Emit::field` renders eagerly, and this
    // sits on a branch the census measures at 28-111 entries a second.
    let outcome_bits = match served {
        None => 0u64,
        Some(IOSurfaceSeedRung::Cache) => 1,
        Some(IOSurfaceSeedRung::GuestPages) => 2,
    };
    let disc =
        (u64::from(mapping_id) << 40) | (u64::from(w) << 20) | u64::from(h) | (outcome_bits << 62);
    if let Some(rung) = served {
        if !crate::observe::first_sight("iosurface_texture_load_seed_served", disc) {
            return;
        }
        crate::observe::off(format!(
            "iosurface_texture_load_seed outcome={} mid={mapping_id} want={w}x{h} \
             mapgeom={map_w}x{map_h} mapgen={map_gen} hostgen={host_gen}",
            rung.name()
        ));
        return;
    }
    let d = match have {
        Some((have_w, have_h)) => IOSurfaceSeedDecline::GeomMismatch { have_w, have_h },
        None => IOSurfaceSeedDecline::NoEntry,
    };
    if !crate::observe::first_sight(crate::observe::Decline::slug(&d), disc) {
        return;
    }
    crate::observe::Emit::decline("iosurface_texture_load_seed", &d)
        .field("mid", mapping_id)
        .field("want", format!("{w}x{h}"))
        .field("mapgeom", format!("{map_w}x{map_h}"))
        .field("mapgen", map_gen)
        .field("hostgen", host_gen)
        .fail();
}

/// The prior contents of an IOSurface texture attachment under `MTL_LOAD_ACTION_LOAD`,
/// with the byte order they are in.
///
/// Two rungs, in freshness order:
///
/// 1. **The host render cache.** The hot one: `store_routes` measures 28-111 of
///    these a second under a browser workload. It holds guest scanout order and
///    the pooled target is RGBA, so the buffer is handed over behind an `Arc` and
///    the R/B exchange rides the engine's single copy into mapped staging rather
///    than materializing a converted frame here.
/// 2. **The surface's own guest IOSurface pages.** The cache is an accelerator,
///    not the surface. What an IOSurface texture attachment *contains* is its pages, so a
///    cache miss is a reason to read them — not a reason to drop the guest's
///    LOAD. Without this rung the pass began with `LoadOp::CLEAR` against the
///    hardcoded `[0,0,0,0]` primary clear and the matching Store published that
///    wipe, which is a whole compositing layer going solid black.
///
/// The guest-pages rung preserves the mapping's native texels as bounded runs.
/// The engine imports or gathers those runs and copies them straight into the
/// same-format attachment; when the host cannot expose stable aliases, the
/// existing RGBA reader remains the universal fallback. Both arms use the
/// mapping's latched geometry, and the engine validates the exact strided span
/// before recording the copy. Any writeback debt is paid before the page view is
/// built, so it observes this device's latest Store rather than pre-Store bytes.
///
/// IOSurface texture `seed_color_load` falls through to the same reader via
/// `load_sampled_rgba_static`.
///
/// `None` means the guest's LOAD could not be honoured at all, and
/// [`note_iosurface_texture_load_seed`] has already said which check refused.
/// Band how long after pressure recovery the guest wanted the resident again.
/// The fixed reference interval keeps existing census buckets comparable; it
/// has no role in deciding whether the resident remains alive.
fn reclaimed_resample_band(since_ms: u64, cutoff: Option<u64>) -> &'static str {
    let Some(cutoff) = cutoff else {
        return "iosurfacesample_reclaimed_cutoff_unavailable";
    };
    if since_ms < cutoff {
        "iosurfacesample_reclaimed_within_1x_cutoff"
    } else if since_ms < cutoff * 2 {
        "iosurfacesample_reclaimed_within_2x_cutoff"
    } else if since_ms < cutoff * 4 {
        "iosurfacesample_reclaimed_within_4x_cutoff"
    } else {
        "iosurfacesample_reclaimed_past_4x_cutoff"
    }
}

fn try_iosurface_texture_target_guest_seed<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    w: u32,
    h: u32,
    target_format: reims_vgpu_core::pixel_format::TexelLayout,
) -> Option<reims_vgpu_memory::GuestTargetSeed> {
    use crate::runtime::mapping_write::iosurface_texture_sample_window;
    use reims_vgpu_memory::GuestRunSource;
    use reims_vgpu_memory::GuestTargetSeed;

    if w == 0 || h == 0 || !mapper::ensure_resolved_for_scanout(state, host, mapping_id) {
        return None;
    }
    let (base_off, bpr, layout) = {
        let mapping = state.surfaces.mappings.get(&mapping_id)?;
        if !mapping.lifecycle.active
            || mapping.pages.entries.is_empty()
            || !mapping.has_geometry()
            || mapping.width_or_zero() != w
            || mapping.height_or_zero() != h
        {
            return None;
        }
        let format = if mapping.format_or_zero() == 0 {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        } else {
            mapping.format_or_zero()
        };
        let layout = pixel_format::store_texel_order(format)?;
        let (base_off, bpr, _) = iosurface_texture_sample_window(mapping, w, h, format)?;
        (base_off, u64::from(bpr), layout)
    };
    if target_format != layout {
        return None;
    }
    let (span, row_length_texels) =
        strided_window_extent(w, h, u64::from(layout.bytes_per_texel()), bpr)?;

    let (gpas, runs) = mapping_window_guest_runs(state, host, mapping_id, base_off, span)?;
    let page = state.page_size();
    let physical_pages = reims_vgpu_memory::GuestPageSet::new(&gpas);
    Some(GuestTargetSeed {
        source: GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 0,
            total_len: span,
            row_length_texels,
            pages: guest_page_window(host, gpas, page, base_off % page, span),
            physical_pages,
        },
        format: layout,
    })
}

/// Stable mapping allocation behind an IOSurface texture attachment.
///
/// The mapping is the allocation and the decoded surface geometry names the
/// plane inside it. Vulkan may bind that allocation directly only if its own
/// image-layout query later agrees with these exact offsets and pitch; a
/// missing stable alias or any disagreement leaves the ordinary copied
/// resident as the complete fallback.
pub(super) fn try_iosurface_texture_target_guest_memory<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    w: u32,
    h: u32,
    target_format: reims_vgpu_core::pixel_format::TexelLayout,
) -> Option<reims_vgpu_memory::GuestTargetMemory> {
    use crate::runtime::mapping_write::iosurface_texture_sample_window;

    if w == 0 || h == 0 || !mapper::ensure_resolved_for_scanout(state, host, mapping_id) {
        return None;
    }
    let (base_off, bpr) = {
        let mapping = state.surfaces.mappings.get(&mapping_id)?;
        if !mapping.lifecycle.active
            || mapping.pages.entries.is_empty()
            || !mapping.has_geometry()
            || mapping.width_or_zero() != w
            || mapping.height_or_zero() != h
        {
            return None;
        }
        let format = if mapping.format_or_zero() == 0 {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        } else {
            mapping.format_or_zero()
        };
        if pixel_format::store_texel_order(format)? != target_format {
            return None;
        }
        let (base_off, bpr, _) = iosurface_texture_sample_window(mapping, w, h, format)?;
        (base_off, u64::from(bpr))
    };
    let (import, footprint) = mapper::ensure_contig_import_with_footprint(state, host, mapping_id)?;
    let backing = reims_vgpu_memory::GuestTargetBacking {
        allocation_host_ptr: import.host_base(),
        allocation_len: import.len(),
        resource_offset: 0,
        resource_len: import.len(),
        plane_offset: base_off,
        row_pitch: bpr,
    };
    backing.visible_window(w, h, u64::from(target_format.bytes_per_texel()))?;
    Some(reims_vgpu_memory::GuestTargetMemory {
        backing,
        import,
        footprint,
    })
}

pub(super) fn resolve_iosurface_texture_load_seed<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mapping_id: u32,
    w: u32,
    h: u32,
    target_format: reims_vgpu_core::pixel_format::TexelLayout,
) -> Option<IOSurfaceLoadSeed> {
    use reims_vgpu_core::SeedOrder;
    // `clear_host_valid` removes this mapping's cache entry at the contract
    // boundary. A hit therefore already means no later guest validity
    // statement superseded the stored bytes.
    let cached = crate::runtime::surface_cache::get_shared(state, mapping_id, w, h);
    let served = if let Some(bgra) = cached {
        Some((
            IOSurfaceLoadSeed::Host(bgra, SeedOrder::Bgra8),
            IOSurfaceSeedRung::Cache,
        ))
    } else {
        try_iosurface_texture_target_guest_seed(state, host, mapping_id, w, h, target_format)
            .map(|seed| {
                (
                    IOSurfaceLoadSeed::Guest(seed),
                    IOSurfaceSeedRung::GuestPages,
                )
            })
            .or_else(|| {
                load_iosurface_mapping_rgba(state, host, mapping_id, None)
                    .map(|(_, _, r)| r)
                    .filter(|rgba| rgba.len() == (w as usize) * (h as usize) * 4)
                    .map(|rgba| {
                        (
                            IOSurfaceLoadSeed::Host(std::sync::Arc::new(rgba), SeedOrder::Rgba8),
                            IOSurfaceSeedRung::GuestPages,
                        )
                    })
            })
    };
    note_iosurface_texture_load_seed(state, mapping_id, w, h, served.as_ref().map(|s| s.1));
    served.map(|(seed, _)| seed)
}

/// Materialize the exact serialized Metal view carried by a IOSurface plane view object.
///
/// The underlying surface backing FourCC is allocation metadata, not necessarily the
/// sampled Metal format. The view's format/geometry define the native row
/// interpretation; the surface backing device descriptor supplies its base/BPR/span.
/// Materialize a IOSurface plane view serialized texture view through the byte-exact
/// revalidated memo (same contract as [`load_linear_guest_memoized`]): every
/// bind re-reads the native plane window so a guest write is always observed;
/// conversion, allocation, and — via the returned content identity — the
/// engine upload are skipped when the bytes are unchanged.
pub(super) fn load_iosurface_plane_view_rgba<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    mapping_id: u32,
    view: objects::IOSurfacePlaneViewDescriptor,
) -> Option<LoadedIOSurfacePlaneView> {
    let fail = |d: IOSurfacePlaneViewDecline| -> Option<LoadedIOSurfacePlaneView> {
        crate::observe::Emit::decline("iosurface_plane_view_draw_view", &d)
            .field("task", task_id)
            .field("ref", texture_ref)
            .field("sid", mapping_id)
            .field("view", format!("{}x{}", view.width, view.height))
            .field("fmt", format!("{:#x}", view.pixel_format))
            .fail();
        None
    };

    if view.depth != 1 {
        return fail(IOSurfacePlaneViewDecline::UnsupportedDepth { depth: view.depth });
    }
    if !mapper::ensure_resolved_for_scanout(state, host, mapping_id) {
        return fail(IOSurfacePlaneViewDecline::Unresolved);
    }
    let Some(bpp) = pixel_format::bytes_per_pixel(view.pixel_format) else {
        return fail(IOSurfacePlaneViewDecline::FormatBpp);
    };
    let (base_off, surface_bpr, span_end, pages_n, base_w, base_h, base_fmt, map_gen) = {
        let Some(m) = state.surfaces.mappings.get(&mapping_id) else {
            return fail(IOSurfacePlaneViewDecline::NoMapping);
        };
        let Some((base_off, surface_bpr, span_end)) =
            mapping_write::iosurface_plane_view_sample_window(
                m,
                view.plane_index,
                view.width,
                view.height,
                view.pixel_format,
            )
        else {
            let desc = reims_vgpu_protocol::decode_device_surface(m.device_desc_bytes()).map(|d| {
                (
                    d.width,
                    d.height,
                    d.pixel_format,
                    d.bytes_per_row,
                    d.alloc_size,
                )
            });
            return fail(IOSurfacePlaneViewDecline::SampleWindow {
                base_w: m.width_or_zero(),
                base_h: m.height_or_zero(),
                base_fmt: m.format_or_zero(),
                desc,
            });
        };
        (
            base_off,
            surface_bpr,
            span_end,
            m.pages.entries.len(),
            m.width_or_zero(),
            m.height_or_zero(),
            m.format_or_zero(),
            m.lifecycle.generation,
        )
    };
    let page_bytes = (pages_n as u64).saturating_mul(1u64 << state.page_shift);
    if page_bytes < span_end {
        return fail(IOSurfacePlaneViewDecline::Span {
            pages: pages_n,
            page_bytes,
            span_end,
            bpr: surface_bpr,
        });
    }
    let Some(tight) = view.width.checked_mul(bpp) else {
        return fail(IOSurfacePlaneViewDecline::TightOverflow { bpp });
    };
    let Some(native_len) = (tight as u64)
        .checked_mul(view.height as u64)
        .and_then(host_alloc_len)
    else {
        return fail(IOSurfacePlaneViewDecline::NativeLen { tight });
    };
    let mut native = vec![0u8; native_len];
    if !mapping_write::read_rect_raw_at(
        state,
        host,
        mapping_id,
        mapping_write::SurfaceWindow {
            base_off,
            bpr: surface_bpr,
            span_end,
            bpp,
        },
        mapping_write::Rect {
            origin_x: 0,
            origin_y: 0,
            width: view.width,
            height: view.height,
        },
        &mut native,
        tight,
    ) {
        return fail(IOSurfacePlaneViewDecline::Read {
            base_w,
            base_h,
            base_fmt,
            off: base_off,
            bpr: surface_bpr,
            span_end,
            pages: pages_n,
        });
    }
    // Identity key namespace: bit 63 marks IOSurface plane view view content (guest linear
    // identities use the raw sampled GVA as key). Every producer draws its
    // generation from `Device::next_sampled_content_generation`, so a
    // (key, generation) pair cannot alias content even on a key collision.
    let identity_key = (1u64 << 63) | ((view.plane_index as u64) << 32) | mapping_id as u64;
    let memo_key = (
        mapping_id,
        view.plane_index,
        view.width,
        view.height,
        view.pixel_format,
    );
    // A single/dual-channel plane (biplanar video Y = R8, CbCr = RG8) uploads at
    // its native footprint: `texel_to_rgba8` places R8→(r,0,0,255) and
    // RG8→(r,g,0,255), which is exactly what an R8_UNORM / R8G8_UNORM Vulkan
    // image samples to (`.r` / `.rg`, zero-filled tail). Skipping the CPU expand
    // and uploading native cuts 4×/2× the staging bytes with byte-exact texels.
    // The ten-bit pair (`'x420'`, `R16Unorm` / `RG16Unorm`) takes the same
    // native rail for the same reason and one more: `texel_to_rgba8` has no arm
    // for them, because an arm would have to narrow ten bits of graded luma to
    // eight. `TexelLayout::has_cpu_loader_arm` is where that is stated.
    //
    // The half-float colour pair is deliberately **not** here yet. It belongs
    // by the same argument the linear rails took — `texel_to_rgba8`'s arm for
    // it clamps to `[0, 1]` and quantizes to 256 levels — but nothing has ever
    // measured a IOSurface plane view view arriving in one, and this rail is the video-plane
    // rail. `iosurface_plane_view_narrowed` below is the measurement; add the arm when it
    // fires, not before.
    //
    // The packed 32-bit colour formats take the native rail for the same
    // reason and a sharper one: their channel boundaries are not byte
    // boundaries, so `TexelLayout::Rgba8` would not merely quantize them, it
    // would read the word as four unrelated bytes. Four bytes wide is exactly
    // what the default arm below tests for and exactly what makes that wrong,
    // which is why they are named here rather than left to it.
    // The layout is the view's own; the transfer function travels with it,
    // because the default arm below converts to RGBA8 order and decodes nothing.
    let byte_format = SampledByteFormat::from_source(
        match view.pixel_format {
            pixel_format::MTL_FORMAT_RG8_UNORM => TexelLayout::Rg8,
            pixel_format::MTL_FORMAT_R16_UNORM => TexelLayout::R16Unorm,
            pixel_format::MTL_FORMAT_RG16_UNORM => TexelLayout::Rg16Unorm,
            pixel_format::MTL_FORMAT_RGB10A2_UNORM => TexelLayout::Rgb10a2Unorm,
            pixel_format::MTL_FORMAT_BGR10A2_UNORM => TexelLayout::Bgr10a2Unorm,
            pixel_format::MTL_FORMAT_RG11B10_FLOAT => TexelLayout::Rg11b10Float,
            _ => TexelLayout::Rgba8,
        },
        view.pixel_format,
    );
    let ok_line = |generation_source: &str, rgba: &[u8]| {
        // Per-draw success echo — fires on EVERY IOSurface plane view plane bind (thousands/sec
        // under video → ~36k lines/boot, 61% of the fail log), burying real
        // failures. The always-on health signal is the `sampled_branch_census`
        // aggregate (IOSurfacePlaneView / T5Memo, noted on both paths below), so this
        // per-bind detail — and its O(w*h) `rgba_rgb_stats` scan — is diagnostic
        // only: gate both behind REIMS_VGPU_DRAW_LOG so a normal boot stays uncluttered.
        if !crate::observe::draw_log_enabled() {
            return;
        }
        let (nz, max, _) = crate::observe::rgba_rgb_stats(rgba);
        crate::observe::line(format!(
            "iosurface_plane_view_draw_view ok task={task_id} ref={texture_ref} sid={mapping_id} map_gen={map_gen} view={}x{} fmt={:#x} bpp={bpp} base={base_w}x{base_h} base_fmt={base_fmt:#x} off={base_off} bpr={surface_bpr} span_end={span_end} src={generation_source} rgb_nz={nz} max_rgb={max}",
            view.width,
            view.height,
            view.pixel_format,
        ));
    };
    if let Some(m) = state
        .content
        .sampled
        .iosurface_plane_view_memo
        .get_touch(&memo_key)
    {
        // Vec equality is length + byte memcmp with early exit on change.
        if m.native == native {
            let rgba = m.rgba.clone();
            let generation = m.generation;
            ok_line("memo", &rgba);
            return Some((
                view.width,
                view.height,
                rgba,
                LinearSampleIdentity {
                    key: identity_key,
                    generation,
                },
                byte_format,
            ));
        }
    }
    // RGBA8 formats expand per-pixel into a fresh RGBA8 buffer; native R8/RG8
    // upload the plane bytes verbatim (the memo stores those bytes as both the
    // memcmp key and the upload payload).
    let rgba: std::sync::Arc<Vec<u8>> = if byte_format.layout() == TexelLayout::Rgba8 {
        let Some(rgba_stride) = view.width.checked_mul(RGBA8_BPP) else {
            return fail(IOSurfacePlaneViewDecline::RgbaStride);
        };
        let Some(rgba_len) = (rgba_stride as u64)
            .checked_mul(view.height as u64)
            .and_then(host_alloc_len)
        else {
            return fail(IOSurfacePlaneViewDecline::RgbaLen {
                stride: rgba_stride,
            });
        };
        let mut rgba = vec![0u8; rgba_len];
        // The third CPU convert in this crate, and the third that said nothing
        // when it lost precision. `byte_format`'s table above names the video
        // plane formats natively and folds everything else here, so a half-float
        // IOSurface plane view view is quantized on the same terms the linear rails were.
        crate::runtime::draw::note_sampled_narrowing(
            "iosurface_plane_view_narrowed",
            texture_ref,
            view.pixel_format,
            view.width,
            view.height,
        );
        for y in 0..view.height as usize {
            let src_off = y.saturating_mul(tight as usize);
            let dst_off = y.saturating_mul(rgba_stride as usize);
            if !pixel_format::convert_row_to_rgba8(
                view.pixel_format,
                &native[src_off..src_off + tight as usize],
                view.width,
                &mut rgba[dst_off..dst_off + rgba_stride as usize],
            ) {
                return fail(IOSurfacePlaneViewDecline::Convert { row: y, bpp });
            }
        }
        std::sync::Arc::new(rgba)
    } else {
        std::sync::Arc::new(native.clone())
    };
    let generation = state.next_sampled_content_generation();
    ok_line("fill", &rgba);
    let entry_bytes = native.len() + rgba.len();
    state.content.sampled.iosurface_plane_view_memo.insert(
        memo_key,
        crate::model::GuestLinearMemo {
            native,
            rgba: rgba.clone(),
            // The IOSurface plane view view path re-derives this from the view's own pixel
            // format on every call, hit or miss, so storing it is a statement of
            // what the bytes are rather than the source anything reads.
            layout: byte_format.layout(),
            generation,
        },
        entry_bytes,
    );
    Some((
        view.width,
        view.height,
        rgba,
        LinearSampleIdentity {
            key: identity_key,
            generation,
        },
        byte_format,
    ))
}

/// Does this host promise a guest-page alias that stays valid indefinitely?
///
/// Every guest-run producer below needs that promise, and needs it for a reason
/// that survived the removal of the host-pointer import: the engine gathers from
/// these pointers when the submission it armed them for reaches the GPU, which is
/// after this call returns, so a pointer with a bounded lifetime would be read
/// after its view was released.
///
/// A `false` is expected control flow — the caller falls through to the CPU
/// byte loader and the guest gets correct pixels — so it is not a decline. But
/// it is answered by the host once and then forever, and the whole rail
/// disappearing is not something a reader should have to infer from an absence,
/// so the first refusal of the process says so by name.
///
/// This is where the arm64 pathway diverges: its MMIO shim can return a
/// `mach_vm_remap` view for a fragmented page list, and since that view is
/// released on `unmap_pages` rather than retained until teardown, the shim
/// answers 0. The x86 PCI shim can assemble scattered file-backed guest pages
/// into one packed alias and retains every such address until teardown, so it
/// answers 1.
fn guest_run_alias_available<M: HostOps>(host: &M) -> bool {
    if host.map_pages_stable() {
        return true;
    }
    static NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::observe::fail(String::from(
            "guest_run_rail off reason=host_page_alias_not_stable \
             (draw binds take the CPU byte loader)",
        ));
    }
    false
}

/// Walk `span` bytes of `task_id`'s GVA space from `gva` and return the guest
/// pages covering it alongside the packed guest-RAM runs over them
/// (GPA-contiguous stretches coalesced and mapped to stable host pointers).
/// `None` when any page is unmapped or the mapping is incomplete. Shared by the
/// sampled and buffer zero-copy rails; callers must land intersecting deferred
/// stores first and verify import coverage per run.
///
/// The page list rides out with the runs because a caller that wants to say
/// anything about the window's *contents* over time needs the pages, not the
/// host pointers: guest-write tracking is registered per page set.
pub(super) fn task_gva_guest_run_window<M: HostMemory + HostOps>(
    state: &Device,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Result<(Vec<u64>, Vec<reims_vgpu_memory::GuestRun>), WindowRefusal> {
    if !guest_run_alias_available(host) {
        return Err(WindowRefusal::NoAlias);
    }
    let page = state.page_size();
    let gpas =
        gva_mem::task_gva_page_gpas(host, &state.tasks, task_id, gva, span, state.page_shift);
    let wanted = reims_vgpu_paging::span::pages_spanned(gva, span, page);
    if gpas.len() as u64 != wanted {
        return Err(WindowRefusal::SpanUnmapped);
    }
    let runs = coalesce_pages_to_runs(host, &gpas, page, gva % page, span)
        .ok_or(WindowRefusal::Untileable)?;
    Ok((gpas, runs))
}

/// Resolve one complete GVA allocation into the transfer source shared by
/// render and compute sampled-image staging.
///
/// This is the copy-backed counterpart of a packed resource: it preserves the
/// allocation's complete byte range and physical-page identity without
/// requiring Vulkan host-pointer import. Callers retain the separately decoded
/// image allocation/view contract beside it.
pub(crate) fn task_gva_guest_run_source<M: HostMemory + HostOps>(
    state: &Device,
    host: &mut M,
    task_id: u32,
    gva: u64,
    span: u64,
) -> Result<(Vec<u64>, reims_vgpu_memory::GuestRunSource), WindowRefusal> {
    let (gpas, runs) = task_gva_guest_run_window(state, host, task_id, gva, span)?;
    let page = state.page_size();
    let physical_pages = reims_vgpu_memory::GuestPageSet::new(&gpas);
    let pages = guest_page_window(host, gpas.clone(), page, gva % page, span);
    Ok((
        gpas,
        reims_vgpu_memory::GuestRunSource {
            runs: std::sync::Arc::new(runs),
            source_offset: 0,
            total_len: span,
            row_length_texels: 0,
            pages,
            physical_pages,
        },
    ))
}

/// Why a guest-page window could not be built.
///
/// Typed rather than a bare `None` because these are **degradations that
/// repeat**. A bind that lands here is not cached — only resolutions are held
/// (see [`crate::runtime::bound_buffers`]) — so the same reference re-walks the
/// task page table and re-pays the CPU staging read on every draw for as long
/// as the guest keeps binding it. That is the part of the per-draw cost the
/// held-resolution registry does not reach, and until these were counted there
/// was no way to say how large it is: the two silent `None`s this replaces were
/// the only unnamed exits in a rail whose every other outcome has a route.
///
/// Each caller maps these onto its own route prefix rather than counting them
/// here. The buffer rail and the linear sampled rail differ by two orders of
/// magnitude in volume, and one shared counter would report their sum as though
/// it were either — the same conflation [`band_runs`] already carries and says
/// so about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WindowRefusal {
    /// The host will not promise a stable page alias, so no rail here can run.
    ///
    /// Latched once per process by [`guest_run_alias_available`], which names
    /// it on the failure channel; the per-caller route is what gives it a rate.
    NoAlias,
    /// Some page of the span does not resolve under the task's page table.
    ///
    /// The one a mapped-range record could answer without walking.
    SpanUnmapped,
    /// Every page resolved, but a GPA-contiguous stretch would not import.
    ///
    /// A walk that finished and still could not bind, so no range record would
    /// have saved it.
    Untileable,
}

/// The bounded guest-memory references behind a set of runs, one per maximal
/// GPA-contiguous stretch, when this host can import guest RAM at all.
///
/// `None` is the routing answer for every host that cannot import — no
/// `VK_EXT_external_memory_host`, an operator who turned the rail off, a shim
/// that cannot say where guest RAM lives, or a GPA the imports do not cover —
/// and the caller gathers on the CPU exactly as it did before. Every refusal is
/// named on the always-on sink by [`crate::runtime::guest_ram_map`], so a fall
/// back to the copy is never silent.
///
/// # Why this asks for runs rather than one bind range
///
/// It asked for one, and a driven boot priced what that cost. `zc_buffer_gathered`
/// read 371 422 against `zc_buffer_imported` at **zero** on a host whose
/// `vk_caps` said `host_pointer_import=supported`, and the banded census said
/// why with no ambiguity left in it: not one window in the boot was refused for
/// a missing import, a declined pointer, an unbacked GPA or a range outside
/// one. Every single one was refused for being scattered, 98.5 % of them into
/// 9-32 stretches, and **nothing at all** at one or two. The guest backs a
/// surface in 16 KiB physically-contiguous granules, so a rail that takes only
/// one stretch is not a rail that rarely fires — it is one that cannot fire.
///
/// The bytes have to be gathered somewhere regardless: a vertex or storage bind
/// must name one contiguous range and these windows are not one in GPA space.
/// So the only question was whether the CPU or the GPU does it, and the CPU was
/// answering it at 3.6 GB/s of `memcpy` — 105 ms per second of wall clock, two
/// thirds of every draw's staging phase. Handing the runs to the caller lets it
/// submit one `VkBufferCopy` per stretch into device-local memory instead, which
/// crosses the bus once where the CPU path crossed it once and paid a full
/// core's memcpy on top.
///
/// # Four call sites, and the counters are shared
///
/// This serves the draw-time buffer rail and three sampled ones. From the boot
/// above, through `engine_delta`:
///
/// | rail | gathers | bytes the CPU moved |
/// |---|---:|---:|
/// | buffer (`stage_phase`'s `runs`) | 15 758 per second | 3.6 GB **per second** |
/// | sampled (`sampled_gathers`) | 211 for the boot | 254 MB for the boot |
///
/// So a reading of `zc_buf_runs_*` is both populations, and the sampled one is
/// around two orders of magnitude smaller. Only the buffer rail consumes the
/// runs; the sampled rail binds a one-run window directly and otherwise still
/// gathers on the CPU, which its own volume does not justify changing.
fn guest_page_window<M: HostOps>(
    host: &mut M,
    gpas: Vec<u64>,
    page: u64,
    head_offset: u64,
    span: u64,
) -> Option<std::sync::Arc<Vec<crate::runtime::guest_ram_map::GuestWindowRun>>> {
    use crate::runtime::guest_ram_map::MapRefusal;
    match crate::runtime::guest_ram_map::references_for_runs(host, &gpas, page, head_offset, span) {
        Ok(runs) => {
            // Banded on the way through as well as on the refusals below,
            // because the count is what decides whether a window binds straight
            // into the draw (one run) or costs a copy region per stretch — and
            // a rail whose regions grew without anyone noticing would read here
            // first.
            crate::runtime::drain::note_store_route(band_runs(runs.len()));
            Some(std::sync::Arc::new(runs))
        }
        Err(refusal) => {
            crate::runtime::drain::note_store_route(match refusal {
                MapRefusal::NoBackendImport => "zc_buf_no_import",
                MapRefusal::HostRefused(_) => "zc_buf_host_refused",
                MapRefusal::NoUsableRegion { .. } => "zc_buf_no_region",
                // Its own band and not folded into `zc_buf_no_import`: this
                // host has the extension and would import, and what refused is
                // the size of the guest against the size of its heaps. A boot
                // reading this is one where raising the heap or lowering `-m`
                // would restore the rail, which is not true of any other band.
                MapRefusal::ImportExceedsHeap { .. } => "zc_buf_over_heap",
                MapRefusal::GpaNotInAnyImport { .. } => "zc_buf_gpa_unbacked",
                MapRefusal::OutsideImport(_) => "zc_buf_outside_import",
                // `references_for_runs` reaches this only for a window it could
                // not tile at all — an empty page list, a zero length, an
                // overflowing range. A merely scattered window is now a success
                // with several runs, counted above.
                MapRefusal::Scattered { .. } => "zc_buf_untileable",
            });
            None
        }
    }
}

/// Band a window's stretch count for the census.
///
/// Banded, not exact: what these decide is how many copy regions a window costs
/// the GPU gather, which is a question about the order of magnitude. An exact
/// count would also need an unbounded set of static strings, which
/// `note_store_route` does not take.
///
/// The low bands are the ones a driven boot was first measured in, kept so a
/// later reading is comparable with that one: 42 windows at 3-4 stretches,
/// 4 322 at 5-8, **370 716 at 9-32** and 1 261 above — and nothing at all at one
/// or two, which is what made the single-reference rail unreachable.
///
/// # Why they reach past 64
///
/// These bands stopped at `>32` while the Vulkan engine capped its GPU gather at
/// 64 regions, so every window that cap turned away landed in one bucket that
/// *starts below the cap* — a reading of it could not say whether a refused
/// window overshot by one region or by five hundred, and the cap's own
/// justification was written from exactly that bucket.
///
/// Widening them answered it and retired the cap: on a driven boot the
/// distribution is bimodal, 99.66 % of windows at 1-32 stretches and a second
/// population of full-screen surfaces at 257-512, with **nothing between 33 and
/// 256**. 64 was not a threshold between two regimes; it sat in the empty space
/// between them, and any value from 33 to 256 would have refused the same 1 162
/// windows. The bands stay wide because that shape is what a future cap
/// proposal has to be argued against.
fn band_runs(runs: usize) -> &'static str {
    match runs {
        0..=1 => "zc_buf_runs_1",
        2 => "zc_buf_runs_2",
        3..=4 => "zc_buf_runs_3_4",
        5..=8 => "zc_buf_runs_5_8",
        9..=32 => "zc_buf_runs_9_32",
        33..=64 => "zc_buf_runs_33_64",
        65..=128 => "zc_buf_runs_65_128",
        129..=256 => "zc_buf_runs_129_256",
        257..=512 => "zc_buf_runs_257_512",
        513..=1024 => "zc_buf_runs_513_1024",
        _ => "zc_buf_runs_gt1024",
    }
}

/// Coalesce GPA-contiguous stretches of `window` into packed host-VA runs
/// covering `span` bytes from `head_off` into the first page.
///
/// The stretch arithmetic is `reims_vgpu_paging::runs::coalesce_window`; what
/// this adds is the host side — one `map_pages` per stretch. `map_pages` hands
/// back a direct RAMBlock alias, so the import is a lookup and `unmap` is a
/// no-op.
///
/// `None` if any stretch fails to import, or if the window runs out before
/// `span` — a partial gather would hand the GPU a short buffer, which is a
/// wrong frame rather than a slow one.
fn coalesce_pages_to_runs<M: HostOps>(
    host: &mut M,
    window: &[u64],
    page: u64,
    head_off: u64,
    span: u64,
) -> Option<Vec<reims_vgpu_memory::GuestRun>> {
    let stretches = reims_vgpu_paging::runs::coalesce_window(window, page, head_off, span)?;
    let mut runs: Vec<reims_vgpu_memory::GuestRun> = Vec::with_capacity(stretches.len());
    for s in stretches {
        let base = host.map_pages(&window[s.pages], page as usize)? as u64;
        runs.push(reims_vgpu_memory::GuestRun {
            host_ptr: (base + s.start_offset) as usize,
            len: s.len,
        });
    }
    Some(runs)
}

fn linear_guest_image_allocation_memory(
    packed: &crate::runtime::bound_buffers::PackedBufferAccess,
    allocation: &reims_vgpu_memory::GuestImageAllocationLayout,
    bytes_per_texel: u64,
) -> Option<reims_vgpu_memory::GuestTargetMemory> {
    let base = allocation.base()?;
    let backing = reims_vgpu_memory::GuestTargetBacking {
        allocation_host_ptr: packed.import.host_base(),
        allocation_len: packed.import.len(),
        resource_offset: packed.head,
        resource_len: packed.size,
        plane_offset: packed.head.checked_add(base.resource_relative_offset)?,
        row_pitch: base.row_pitch,
    };
    for mip in allocation.mips.iter() {
        mip.plane_in(backing)?
            .visible_image_window(mip.layout, bytes_per_texel)?;
    }
    Some(reims_vgpu_memory::GuestTargetMemory {
        backing,
        import: packed.import.clone(),
        footprint: packed.footprint.clone(),
    })
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_sample_from_packed(
    packed: &crate::runtime::bound_buffers::PackedBuffer,
    level_offset: u64,
    span: u64,
    row_length_texels: u32,
    layout: reims_vgpu_memory::GuestImageLayout,
    native: TexelLayout,
    format: reims_vgpu_protocol::ImageFormat,
    identity: LinearSampleIdentity,
    vouch: crate::runtime::gather_witness::GatherVouch,
    native_components: pixel_format::SwizzlePlan,
) -> Option<SampledSourceRequest> {
    let transfer = packed.texel_source(level_offset, span, row_length_texels)?;
    let bytes_per_texel = u64::from(native.bytes_per_texel());
    let row_pitch = if row_length_texels == 0 {
        u64::from(layout.width()).checked_mul(bytes_per_texel)?
    } else {
        u64::from(row_length_texels).checked_mul(bytes_per_texel)?
    };
    let backing = reims_vgpu_memory::GuestTargetBacking {
        allocation_host_ptr: packed.import.host_base(),
        allocation_len: packed.import.len(),
        resource_offset: packed.head,
        resource_len: packed.size,
        plane_offset: packed.head.checked_add(level_offset)?,
        row_pitch,
    };
    backing.visible_image_window(layout, bytes_per_texel)?;
    let memory = reims_vgpu_memory::GuestTargetMemory {
        backing,
        import: packed.import.clone(),
        footprint: packed.footprint.clone(),
    };
    Some(SampledSourceRequest::GuestImage(
        reims_vgpu_memory::GuestImageSource::single_mip(memory, layout, transfer)?,
        format,
        Some(identity),
        vouch,
        native_components,
    ))
}

/// Build one GPU-copy source from the mapping's retained allocation.
#[derive(Clone, Copy)]
struct MappedSamplePlane {
    mapping_id: u32,
    base_off: u64,
    span: u64,
    row_length_texels: u32,
}

fn mapped_sampled_source<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    plane: MappedSamplePlane,
) -> Option<reims_vgpu_memory::GuestRunSource> {
    let MappedSamplePlane {
        mapping_id,
        base_off,
        span,
        row_length_texels,
    } = plane;
    crate::runtime::mapper::guest_texel_source(
        state,
        host,
        mapping_id,
        base_off,
        span,
        row_length_texels,
    )
}

/// Build either retained-allocation or page-run execution state for a mapped
/// plane, with one content identity shared by both representations. The source
/// choice changes how the GPU reaches the bytes; it must not decide whether a
/// retained copied image can be recognised on the next bind.
fn witnessed_mapping_sampled_source<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    rail: crate::runtime::gather_witness::GatherRail,
    plane: MappedSamplePlane,
) -> Option<(
    reims_vgpu_memory::GuestRunSource,
    LinearSampleIdentity,
    crate::runtime::gather_witness::GatherVouch,
)> {
    let retained = mapped_sampled_source(state, host, plane);
    let (gpas, runs) =
        mapping_window_guest_runs(state, host, plane.mapping_id, plane.base_off, plane.span)?;
    let page = state.page_size() as usize;
    // A mapping's page generation, not a texture's content version, and the two
    // do not have the same grounds. The storage-mode gate the linear rail
    // applies is unreachable here: a mapping is named by many textures and this
    // device keeps no reverse lookup, so there is no single declaration to ask.
    // What stands in for it is the surface's own flush-on-access rule
    // (`iosurface_texture_sample_window`), under which a guest CPU write to the
    // surface reaches this generation. If that ever stops holding, this is the
    // site that loses content, and the sampled content audit is its only
    // instrument.
    let stated = Some(crate::runtime::gather_witness::StatedGeneration::Mapping(
        state
            .surfaces
            .mappings
            .get(&plane.mapping_id)?
            .content
            .guest_page_generation,
    ));
    let seen = crate::runtime::gather_witness::note_gather(
        state,
        rail,
        crate::runtime::gather_witness::GatherKey::Mapping {
            mapping: reims_vgpu_protocol::MappingId::new(plane.mapping_id),
            base_offset: reims_vgpu_protocol::ByteOffset::new(plane.base_off),
        },
        stated,
        crate::runtime::gather_witness::GatherWindow {
            gpas: &gpas,
            runs: &runs,
            span: plane.span,
            page_size: page,
        },
    );
    let physical_pages = reims_vgpu_memory::GuestPageSet::new(&gpas);
    let source = retained.unwrap_or_else(|| reims_vgpu_memory::GuestRunSource {
        runs: std::sync::Arc::new(runs),
        source_offset: 0,
        total_len: plane.span,
        row_length_texels: plane.row_length_texels,
        pages: guest_page_window(
            host,
            gpas,
            page as u64,
            plane.base_off % page as u64,
            plane.span,
        ),
        physical_pages,
    });
    Some((
        source,
        LinearSampleIdentity::from(seen.identity),
        seen.vouch,
    ))
}

/// Resolve the stable allocation that owns one mapped image plane.
///
/// The plane offset and pitch are decoded resource geometry. Vulkan still has
/// to prove its subresource layout agrees before it may bind this allocation as
/// an image; this function only preserves the resource's backing identity up to
/// that admission point.
fn mapped_guest_image_memory<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    plane: MappedSamplePlane,
    row_pitch: u64,
    extent: (u32, u32),
    bytes_per_texel: u64,
) -> Option<reims_vgpu_memory::GuestTargetMemory> {
    let (import, footprint) =
        mapper::ensure_contig_import_with_footprint(state, host, plane.mapping_id)?;
    let backing = reims_vgpu_memory::GuestTargetBacking {
        allocation_host_ptr: import.host_base(),
        allocation_len: import.len(),
        resource_offset: 0,
        resource_len: import.len(),
        plane_offset: plane.base_off,
        row_pitch,
    };
    backing.visible_window(extent.0, extent.1, bytes_per_texel)?;
    Some(reims_vgpu_memory::GuestTargetMemory {
        backing,
        import,
        footprint,
    })
}

/// The byte extent of a `w × h` image at `bpr` bytes per row and `bpp` bytes per
/// texel, and the `bufferRowLength` in texels the copy needs to stride the
/// padding (0 when rows are tight).
///
/// `None` when the stride cannot describe the image: narrower than one tight
/// row, or not a whole number of texels — `bufferRowLength` is a texel count, so
/// a byte-granular stride has no representation. Padded strides otherwise ride
/// the same rail. The extent stops after the last row's texels because trailing
/// padding may not be mapped.
pub(super) fn strided_window_extent(w: u32, h: u32, bpp: u64, bpr: u64) -> Option<(u64, u32)> {
    let tight = (w as u64).checked_mul(bpp)?;
    if bpr < tight || bpp == 0 || !bpr.is_multiple_of(bpp) {
        return None;
    }
    let span = bpr
        .checked_mul(h.checked_sub(1)? as u64)?
        .checked_add(tight)?;
    let row_length_texels = if bpr == tight {
        0
    } else {
        u32::try_from(bpr / bpp).ok()?
    };
    Some((span, row_length_texels))
}

/// Byte window and Vulkan row pitch for one complete linear texture level.
/// Depth planes are consecutive at `row_stride * height`; only the final
/// plane's final-row padding is outside the sampled window.
pub(super) fn strided_level_extent(
    layout: &crate::runtime::decode::resource::TextureLevelLayout,
    bpp: u64,
) -> Option<(u64, u32)> {
    let (last_plane, row_length_texels) =
        strided_window_extent(layout.width, layout.height, bpp, layout.row_stride)?;
    let preceding_planes = u64::from(layout.planes() - 1)
        .checked_mul(layout.row_stride)?
        .checked_mul(u64::from(layout.height))?;
    Some((preceding_planes.checked_add(last_plane)?, row_length_texels))
}

/// Project the decoded resource layout and reflected shader dimension into one
/// complete image-layout contract.
///
/// Array count comes from the texture declaration, while 3D depth comes from
/// the level record. They are deliberately not interchangeable: Vulkan uses
/// different create fields and different subresource pitches for each.
#[cfg(test)]
pub(super) fn declared_guest_image_layout(
    shape: reims_vgpu_core::SampledImageShape,
    texture: &TextureDescriptor,
    level: &crate::runtime::decode::resource::TextureLevelLayout,
    view_texture_type: Option<u16>,
) -> Option<reims_vgpu_memory::GuestImageLayout> {
    declared_guest_image_selection(shape, texture, level, view_texture_type, None)
        .map(|selection| selection.0)
}

/// Exact image shape and byte displacement selected by a texture view.
///
/// An array view may expose a layer subrange or one layer as a non-array
/// image. Geometry and allocation displacement are one answer here so no
/// caller can update one while continuing to sample the old layer.
pub(super) fn declared_guest_image_selection(
    shape: reims_vgpu_core::SampledImageShape,
    texture: &TextureDescriptor,
    level: &crate::runtime::decode::resource::TextureLevelLayout,
    view_texture_type: Option<u16>,
    view_range: Option<TextureViewRange>,
) -> Option<(reims_vgpu_memory::GuestImageLayout, u64)> {
    use crate::runtime::decode::resource::{
        TEXTURE_VIEW_MTL_TYPE_1D, TEXTURE_VIEW_MTL_TYPE_1D_ARRAY, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY, TEXTURE_VIEW_MTL_TYPE_3D, TEXTURE_VIEW_MTL_TYPE_CUBE,
    };
    // Face counts are compared against `slice_base`/`slice_count`, which the
    // decoded view range carries as `u64`.
    const CUBE_FACES: u64 = reims_vgpu_protocol::CUBE_FACES as u64;
    if shape.multisampled {
        return None;
    }
    let declaration = texture.declaration?;
    let storage_type = u16::from(declaration.texture_type);
    let storage_is_cube = matches!(
        storage_type,
        crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE
            | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
    );
    if texture.slice_count != u32::from(declaration.array_length)
        || texture.cube_faces != storage_is_cube
        || !texture.declared_packing_fits_allocation()
        || !texture.level_fits_slice(level)
    {
        return None;
    }
    let declared_type = view_texture_type.unwrap_or(storage_type);
    let (slice_base, slice_count) = view_range
        .map(|range| (range.slice_base, range.slice_count))
        .unwrap_or_else(|| match storage_type {
            TEXTURE_VIEW_MTL_TYPE_1D_ARRAY | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY => {
                (0, u64::from(declaration.array_length))
            }
            // Six, by the definition of the type the guest declared: a cube
            // texture always has six slices, one per face. That is the same
            // count [`physical_slice_count`] expands `slice_count` by, and it
            // is why nothing here needs a cube layout of its own.
            //
            // [`physical_slice_count`]: reims_vgpu_protocol::LinearTextureDescriptor::physical_slice_count
            TEXTURE_VIEW_MTL_TYPE_CUBE => (0, CUBE_FACES),
            _ => (0, 1),
        });
    let storage_layers = match storage_type {
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY => {
            u64::from(declaration.array_length)
        }
        TEXTURE_VIEW_MTL_TYPE_CUBE => CUBE_FACES,
        _ => 1,
    };
    if slice_count == 0 || slice_base.checked_add(slice_count)? > storage_layers {
        return None;
    }
    let layer_offset = match storage_type {
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
        | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
        | TEXTURE_VIEW_MTL_TYPE_CUBE => texture.bytes_per_slice.checked_mul(slice_base)?,
        _ if slice_base == 0 && slice_count == 1 => 0,
        _ => return None,
    };
    // Dispatch on the type the guest's own descriptor declares, and let the
    // shader's reflected shape veto. Metal's texture object carries its type;
    // a shader that declares a different one is a mismatch the guest is
    // responsible for, never a choice this device gets to make. Written the
    // other way round — a `match` on `shape` that then checks the declaration
    // agrees — the two are the same function, which
    // `the_declaration_and_the_shader_shape_select_the_same_layout` proves over
    // the whole input space. This way round is the one whose answer does not
    // depend on which pipeline happens to be bound, so it can be taken once
    // when the guest sets the texture instead of once per draw.
    match declared_type {
        TEXTURE_VIEW_MTL_TYPE_CUBE => {
            // Only a cube storage, never a cube array: `sampled_image_shape`
            // refuses `CubeArray` outright, so a `shape.cube` over cube-array
            // storage could only be the first six faces of a longer array,
            // chosen silently. And only the whole cube -- a face subrange is a
            // shape the reflected declaration cannot be, so it is a refusal
            // rather than a clamp.
            (shape.cube && storage_type == TEXTURE_VIEW_MTL_TYPE_CUBE && slice_count == CUBE_FACES)
                .then(|| {
                    Some((
                        reims_vgpu_memory::GuestImageLayout::D2Array {
                            width: level.width,
                            height: level.height,
                            layers: u32::try_from(CUBE_FACES).ok()?,
                            array_pitch: texture.bytes_per_slice,
                        },
                        layer_offset,
                    ))
                })
                .flatten()
        }
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY => {
            if !shape.one_dim || !shape.arrayed || level.height != 1 {
                return None;
            }
            if storage_type != TEXTURE_VIEW_MTL_TYPE_1D_ARRAY {
                return None;
            }
            let layers = u32::try_from(slice_count).ok()?;
            (layers != 0).then_some((
                reims_vgpu_memory::GuestImageLayout::D1Array {
                    width: level.width,
                    layers,
                    array_pitch: texture.bytes_per_slice,
                },
                layer_offset,
            ))
        }
        TEXTURE_VIEW_MTL_TYPE_1D => {
            if !shape.one_dim || shape.arrayed || level.height != 1 {
                return None;
            }
            (matches!(
                storage_type,
                TEXTURE_VIEW_MTL_TYPE_1D | TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            ) && slice_count == 1)
                .then_some((
                    reims_vgpu_memory::GuestImageLayout::D1 { width: level.width },
                    layer_offset,
                ))
        }
        TEXTURE_VIEW_MTL_TYPE_3D => {
            if !shape.volume || shape.cube || shape.one_dim {
                return None;
            }
            if storage_type != TEXTURE_VIEW_MTL_TYPE_3D || slice_base != 0 || slice_count != 1 {
                return None;
            }
            let depth = level.planes();
            let depth_pitch = level.row_stride.checked_mul(u64::from(level.height))?;
            Some((
                reims_vgpu_memory::GuestImageLayout::D3 {
                    width: level.width,
                    height: level.height,
                    depth,
                    depth_pitch,
                },
                0,
            ))
        }
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY => {
            if shape.cube || shape.one_dim || shape.volume || !shape.arrayed {
                return None;
            }
            if storage_type != TEXTURE_VIEW_MTL_TYPE_2D_ARRAY {
                return None;
            }
            let layers = u32::try_from(slice_count).ok()?;
            (layers != 0).then_some((
                reims_vgpu_memory::GuestImageLayout::D2Array {
                    width: level.width,
                    height: level.height,
                    layers,
                    array_pitch: texture.bytes_per_slice,
                },
                layer_offset,
            ))
        }
        TEXTURE_VIEW_MTL_TYPE_2D => {
            if shape.cube || shape.one_dim || shape.volume || shape.arrayed {
                return None;
            }
            (matches!(
                storage_type,
                TEXTURE_VIEW_MTL_TYPE_2D | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
            ) && slice_count == 1)
                .then_some((
                    reims_vgpu_memory::GuestImageLayout::D2 {
                        width: level.width,
                        height: level.height,
                    },
                    layer_offset,
                ))
        }
        _ => None,
    }
}

/// The pre-inversion implementation of [`declared_guest_image_selection`],
/// kept as the differential oracle for
/// `the_declaration_and_the_shader_shape_select_the_same_layout`.
///
/// It dispatches on the *shader's* declared shape and then checks the guest's
/// texture declaration agrees. That is the wrong way round — the texture
/// object declares its own type and a shader must match it, so the shape can
/// only ever veto — but the two must select identically, and the only way to
/// prove that over the whole input space is to keep both and compare.
#[cfg(test)]
pub(super) fn selection_dispatched_on_the_shader_shape(
    shape: reims_vgpu_core::SampledImageShape,
    texture: &TextureDescriptor,
    level: &crate::runtime::decode::resource::TextureLevelLayout,
    view_texture_type: Option<u16>,
    view_range: Option<TextureViewRange>,
) -> Option<(reims_vgpu_memory::GuestImageLayout, u64)> {
    use crate::runtime::decode::resource::{
        TEXTURE_VIEW_MTL_TYPE_1D, TEXTURE_VIEW_MTL_TYPE_1D_ARRAY, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY, TEXTURE_VIEW_MTL_TYPE_3D, TEXTURE_VIEW_MTL_TYPE_CUBE,
    };
    // Face counts are compared against `slice_base`/`slice_count`, which the
    // decoded view range carries as `u64`.
    const CUBE_FACES: u64 = reims_vgpu_protocol::CUBE_FACES as u64;
    if shape.multisampled {
        return None;
    }
    let declaration = texture.declaration?;
    let storage_type = u16::from(declaration.texture_type);
    let storage_is_cube = matches!(
        storage_type,
        crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE
            | crate::runtime::decode::resource::TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
    );
    if texture.slice_count != u32::from(declaration.array_length)
        || texture.cube_faces != storage_is_cube
        || !texture.declared_packing_fits_allocation()
        || !texture.level_fits_slice(level)
    {
        return None;
    }
    let declared_type = view_texture_type.unwrap_or(storage_type);
    let (slice_base, slice_count) = view_range
        .map(|range| (range.slice_base, range.slice_count))
        .unwrap_or_else(|| match storage_type {
            TEXTURE_VIEW_MTL_TYPE_1D_ARRAY | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY => {
                (0, u64::from(declaration.array_length))
            }
            // Six, by the definition of the type the guest declared: a cube
            // texture always has six slices, one per face. That is the same
            // count [`physical_slice_count`] expands `slice_count` by, and it
            // is why nothing here needs a cube layout of its own.
            //
            // [`physical_slice_count`]: reims_vgpu_protocol::LinearTextureDescriptor::physical_slice_count
            TEXTURE_VIEW_MTL_TYPE_CUBE => (0, CUBE_FACES),
            _ => (0, 1),
        });
    let storage_layers = match storage_type {
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY => {
            u64::from(declaration.array_length)
        }
        TEXTURE_VIEW_MTL_TYPE_CUBE => CUBE_FACES,
        _ => 1,
    };
    if slice_count == 0 || slice_base.checked_add(slice_count)? > storage_layers {
        return None;
    }
    let layer_offset = match storage_type {
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
        | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
        | TEXTURE_VIEW_MTL_TYPE_CUBE => texture.bytes_per_slice.checked_mul(slice_base)?,
        _ if slice_base == 0 && slice_count == 1 => 0,
        _ => return None,
    };
    if shape.cube {
        // Only a cube storage, never a cube array: `sampled_image_shape`
        // refuses `CubeArray` outright, so a `shape.cube` over cube-array
        // storage could only be the first six faces of a longer array, chosen
        // silently. And only the whole cube -- a face subrange is a shape the
        // reflected declaration cannot be, so it is a refusal rather than a
        // clamp.
        if declared_type != TEXTURE_VIEW_MTL_TYPE_CUBE
            || storage_type != TEXTURE_VIEW_MTL_TYPE_CUBE
            || slice_count != CUBE_FACES
        {
            return None;
        }
        return Some((
            reims_vgpu_memory::GuestImageLayout::D2Array {
                width: level.width,
                height: level.height,
                layers: u32::try_from(CUBE_FACES).ok()?,
                array_pitch: texture.bytes_per_slice,
            },
            layer_offset,
        ));
    }
    if shape.one_dim {
        if level.height != 1 {
            return None;
        }
        if shape.arrayed {
            if declared_type != TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
                || storage_type != TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            {
                return None;
            }
            let layers = u32::try_from(slice_count).ok()?;
            return (layers != 0).then_some((
                reims_vgpu_memory::GuestImageLayout::D1Array {
                    width: level.width,
                    layers,
                    array_pitch: texture.bytes_per_slice,
                },
                layer_offset,
            ));
        }
        return (declared_type == TEXTURE_VIEW_MTL_TYPE_1D
            && matches!(
                storage_type,
                TEXTURE_VIEW_MTL_TYPE_1D | TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            )
            && slice_count == 1)
            .then_some((
                reims_vgpu_memory::GuestImageLayout::D1 { width: level.width },
                layer_offset,
            ));
    }
    if shape.volume {
        if declared_type != TEXTURE_VIEW_MTL_TYPE_3D
            || storage_type != TEXTURE_VIEW_MTL_TYPE_3D
            || slice_base != 0
            || slice_count != 1
        {
            return None;
        }
        let depth = level.planes();
        let depth_pitch = level.row_stride.checked_mul(u64::from(level.height))?;
        return Some((
            reims_vgpu_memory::GuestImageLayout::D3 {
                width: level.width,
                height: level.height,
                depth,
                depth_pitch,
            },
            0,
        ));
    }
    if shape.arrayed {
        if declared_type != TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
            || storage_type != TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
        {
            return None;
        }
        let layers = u32::try_from(slice_count).ok()?;
        return (layers != 0).then_some((
            reims_vgpu_memory::GuestImageLayout::D2Array {
                width: level.width,
                height: level.height,
                layers,
                array_pitch: texture.bytes_per_slice,
            },
            layer_offset,
        ));
    }
    (declared_type == TEXTURE_VIEW_MTL_TYPE_2D
        && matches!(
            storage_type,
            TEXTURE_VIEW_MTL_TYPE_2D | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
        )
        && slice_count == 1)
        .then_some((
            reims_vgpu_memory::GuestImageLayout::D2 {
                width: level.width,
                height: level.height,
            },
            layer_offset,
        ))
}

/// Complete Vulkan-compatible allocation layout and the view selected from it.
///
/// Linear texture storage is slice-major: every array slice owns one complete
/// mip chain and `bytes_per_slice` advances between equal mip levels in
/// adjacent slices. Vulkan expresses that same relation as `arrayPitch` on
/// each mip. A volume is different: its shrinking depth remains inside the
/// mip and is represented by `depthPitch`, never by array layers.
pub(crate) fn declared_guest_image_allocation(
    shape: reims_vgpu_core::SampledImageShape,
    texture: &TextureDescriptor,
    view_texture_type: Option<u16>,
    view_range: Option<TextureViewRange>,
    bytes_per_texel: u64,
) -> Option<(
    reims_vgpu_memory::GuestImageAllocationLayout,
    reims_vgpu_memory::GuestImageViewRange,
)> {
    use crate::runtime::decode::resource::{
        TEXTURE_VIEW_MTL_TYPE_1D, TEXTURE_VIEW_MTL_TYPE_1D_ARRAY, TEXTURE_VIEW_MTL_TYPE_2D,
        TEXTURE_VIEW_MTL_TYPE_2D_ARRAY, TEXTURE_VIEW_MTL_TYPE_3D, TEXTURE_VIEW_MTL_TYPE_CUBE,
    };

    if shape.multisampled
        || texture.compressed_layout
        || bytes_per_texel == 0
        || !texture.declared_packing_fits_allocation()
    {
        return None;
    }
    let declaration = texture.declaration?;
    let storage_type = u16::from(declaration.texture_type);
    let declared_mips = texture.mipmap_level_count.max(1);
    if texture.levels.len() != usize::try_from(declared_mips).ok()?
        || texture.slice_count != u32::from(declaration.array_length)
    {
        return None;
    }

    let storage_layers = match storage_type {
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY => {
            u32::from(declaration.array_length)
        }
        // A cube's faces are its physical slices, which is exactly what
        // `physical_slice_count` reports for the same descriptor -- this is
        // that count, restricted to the one declared array element a
        // `shape.cube` bind can name.
        TEXTURE_VIEW_MTL_TYPE_CUBE => reims_vgpu_protocol::CUBE_FACES,
        TEXTURE_VIEW_MTL_TYPE_1D | TEXTURE_VIEW_MTL_TYPE_2D | TEXTURE_VIEW_MTL_TYPE_3D => 1,
        _ => return None,
    };
    if storage_layers == 0 {
        return None;
    }

    let (base_mip_level, mip_level_count, base_array_layer, array_layer_count) = match view_range {
        Some(range) => (
            u32::try_from(range.level_base).ok()?,
            u32::try_from(range.level_count).ok()?,
            u32::try_from(range.slice_base).ok()?,
            u32::try_from(range.slice_count).ok()?,
        ),
        None => (0, declared_mips, 0, storage_layers),
    };
    let view = reims_vgpu_memory::GuestImageViewRange {
        base_mip_level,
        mip_level_count,
        base_array_layer,
        array_layer_count,
    };

    let base_level = texture.level(base_mip_level)?;
    declared_guest_image_selection(shape, texture, base_level, view_texture_type, view_range)?;

    let first = texture.level(0)?;
    let mut mips = Vec::with_capacity(texture.levels.len());
    for (index, level) in texture.levels.iter().enumerate() {
        let mip = u32::try_from(index).ok()?;
        let reduced = |base: u32| base.checked_shr(mip).unwrap_or(0).max(1);
        let expected_width = reduced(first.width);
        let expected_height = reduced(first.height);
        let expected_depth = reduced(first.planes());
        if !texture.level_fits_slice(level)
            || level.width != expected_width
            || level.row_stride < u64::from(level.width).checked_mul(bytes_per_texel)?
            || !level.row_stride.is_multiple_of(bytes_per_texel)
        {
            return None;
        }

        let layout = match storage_type {
            TEXTURE_VIEW_MTL_TYPE_1D if level.height == 1 && level.planes() == 1 => {
                reims_vgpu_memory::GuestImageLayout::D1 { width: level.width }
            }
            TEXTURE_VIEW_MTL_TYPE_1D_ARRAY if level.height == 1 && level.planes() == 1 => {
                reims_vgpu_memory::GuestImageLayout::D1Array {
                    width: level.width,
                    layers: storage_layers,
                    array_pitch: texture.bytes_per_slice,
                }
            }
            TEXTURE_VIEW_MTL_TYPE_2D if level.height == expected_height && level.planes() == 1 => {
                reims_vgpu_memory::GuestImageLayout::D2 {
                    width: level.width,
                    height: level.height,
                }
            }
            // A cube joins the 2-D array arm rather than getting one of its
            // own: its six faces are six slices of the same slice-major
            // packing, at the same `bytes_per_slice` advance, in the order
            // both APIs define. See [`reims_vgpu_protocol::CUBE_FACES`].
            TEXTURE_VIEW_MTL_TYPE_2D_ARRAY | TEXTURE_VIEW_MTL_TYPE_CUBE
                if level.height == expected_height && level.planes() == 1 =>
            {
                reims_vgpu_memory::GuestImageLayout::D2Array {
                    width: level.width,
                    height: level.height,
                    layers: storage_layers,
                    array_pitch: texture.bytes_per_slice,
                }
            }
            TEXTURE_VIEW_MTL_TYPE_3D
                if level.height == expected_height && level.planes() == expected_depth =>
            {
                reims_vgpu_memory::GuestImageLayout::D3 {
                    width: level.width,
                    height: level.height,
                    depth: level.planes(),
                    depth_pitch: level.row_stride.checked_mul(u64::from(level.height))?,
                }
            }
            _ => return None,
        };
        let tight_row = u32::try_from(u64::from(level.width).checked_mul(bytes_per_texel)?).ok()?;
        if level.slice_read_span(tight_row)? > level.size {
            return None;
        }
        mips.push(reims_vgpu_memory::GuestImageMipLayout {
            resource_relative_offset: texture.base_offset.checked_add(level.offset)?,
            row_pitch: level.row_stride,
            layout,
        });
    }
    let allocation = reims_vgpu_memory::GuestImageAllocationLayout {
        mips: std::sync::Arc::from(mips),
    };
    view.fits(&allocation).then_some((allocation, view))
}

/// Gather `span` bytes from `base_off` into mapping `mid`'s guest pages as host
/// runs, landing any deferred writeback that aliases them first. Returns the
/// window's own page list beside the runs, for the same reason
/// [`task_gva_guest_run_window`] does.
///
/// Shared by the IOSurface texture attachment seed and the IOSurface texture/IOSurface plane view sampled rails,
/// which reach the same pages through different window math.
///
/// # No settle, for the reason its linear twin already states
///
/// This produced a settle until it was measured at 945 waits and 0.63 s on a
/// driven boot, justified as "the coherence rule the CPU loaders obey: a
/// resident-authoritative window covering this mapping must land before the GPU
/// reads, or the gather sees the pre-Store bytes". That rule is the CPU
/// loaders', and this is not one of them. Nothing here reads a pixel byte: it
/// resolves a page list and coalesces it into runs, and the *GPU* reads those
/// runs when the draw's command buffer executes.
///
/// A guest-page writeback is a GPU command on the same single queue, submitted
/// before this call can return, so queue order already puts it ahead of the
/// gather. [`try_linear_sample_zero_copy`] states the same argument for the
/// linear gather and has never taken a settle; this is the arm that diverged.
fn mapping_window_guest_runs<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mid: u32,
    base_off: u64,
    span: u64,
) -> Option<(Vec<u64>, Vec<reims_vgpu_memory::GuestRun>)> {
    if !guest_run_alias_available(host) {
        return None;
    }
    let gpas = mapper::mapping_page_gpas(state, host, mid)?;
    let page = state.page_size();
    if (gpas.len() as u64).saturating_mul(page) < base_off.checked_add(span)? {
        return None;
    }
    let first_page = (base_off / page) as usize;
    let head_off = base_off % page;
    let need_pages = (head_off + span).div_ceil(page) as usize;
    let window = gpas.get(first_page..first_page + need_pages)?;
    let runs = coalesce_pages_to_runs(host, window, page, head_off, span)?;
    Some((window.to_vec(), runs))
}

/// Zero-copy draw-time buffer bind: resolve a type-1 buffer object's backing
/// span (from `offset`) to guest-RAM runs and hand the engine a
/// [`engine::BufferContent::GuestRuns`] — the GPU gathers the bytes from
/// imported guest RAM inside the draw's own CB. Replaces the per-draw CPU
/// re-read + double memcpy of the same ~50–260 KB vertex/SSBO buffers.
/// Guest CPU writes are still observed: the gather re-executes every draw
/// and reads at execute time (at least as fresh as the CPU path).
///
/// Every non-empty declared window is eligible. A miss means its pages cannot
/// be represented as stable host mappings, and the caller stays on the CPU
/// staging read. Deferred stores intersecting the span are landed first,
/// exactly like the CPU path.
///
/// The guest bind itself has no length, so `size - offset` remains the admission
/// window and the fallback whenever reflection cannot prove a tighter answer.
/// A reflected bounded object or invocation-bounded footprint narrows only the
/// bytes walked and moved. Unbounded pointers, unknown access, and indexed
/// vertex access keep the full window.
fn try_buffer_zero_copy_resolved<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    backing: &BufferBacking,
    offset: u64,
    extent_cap: Option<u64>,
) -> Option<crate::runtime::bound_buffers::BoundBuffer> {
    let (gva, size) = (backing.gva, backing.size);
    if offset >= size {
        // The guest bound past the end of the allocation it named. Counted
        // rather than dropped: it is the guest disagreeing with its own
        // descriptor, and it is the one route here that is not about paging.
        crate::runtime::drain::note_store_route("zc_buffer_offset_past_end");
        return None;
    }
    let Some(span) = host_alloc_len(size - offset)
        .filter(|&n| n > 0)
        .map(|n| n as u64)
    else {
        // A declared length this process cannot address. `offset < size` above
        // makes the `n > 0` arm unreachable, so this is the width check alone.
        crate::runtime::drain::note_store_route("zc_buffer_span_unusable");
        return None;
    };
    // The shader's proven reach, when it has one. `min` and not the cap alone:
    // a declared object larger than what is left of the allocation is the guest
    // and the shader disagreeing, and the allocation is the side that bounds
    // what this device may read.
    let full = span;
    let span = extent_cap.map_or(full, |cap| full.min(cap));
    // Counted only once the rail has actually taken the bind. Counting at the
    // narrowing instead credited this rail with bytes the bind then went and
    // read on the CPU path anyway, which is a saving that did not happen.
    if span < full {
        crate::runtime::drain::note_store_route("zc_buffer_extent_narrowed");
        crate::runtime::drain::note_store_route_n("zc_buffer_extent_saved_bytes", full - span);
    }
    // No settle here, for the reason `try_linear_sample_zero_copy` states at
    // length: this rail hands the engine guest-RAM runs and the *GPU* reads them
    // when the draw's command buffer executes, so a guest-page writeback — a GPU
    // command already on the same single queue — is ordered ahead of it by
    // submission order. Only the CPU readers, which touch the pages with this
    // thread, owe the block.
    // Walk exactly the bound range. Resolving the whole backing and slicing out
    // the bind would translate every page of the allocation to serve one bind,
    // and would refuse a bind whose allocation has an unmapped tail page even
    // though the bind itself resolves.
    let (gpas, runs) = match task_gva_guest_run_window(state, host, task_id, gva + offset, span) {
        Ok(window) => window,
        Err(refusal) => {
            crate::runtime::drain::note_store_route(match refusal {
                WindowRefusal::NoAlias => "zc_buffer_no_alias",
                WindowRefusal::SpanUnmapped => "zc_buffer_span_unmapped",
                WindowRefusal::Untileable => "zc_buffer_untileable",
            });
            return None;
        }
    };
    let page = state.page_size();
    let physical_pages = reims_vgpu_memory::GuestPageSet::new(&gpas);
    let pages = guest_page_window(host, gpas, page, (gva + offset) % page, span);
    crate::runtime::drain::note_store_route(if pages.is_some() {
        "zc_buffer_imported"
    } else {
        "zc_buffer_gathered"
    });
    Some(crate::runtime::bound_buffers::BoundBuffer {
        gva: gva + offset,
        span,
        source_offset: 0,
        runs: std::sync::Arc::new(runs),
        pages,
        physical_pages,
    })
}

/// The engine's view of a held resolution.
///
/// One spelling for the fresh walk and the lookup, so a resolution cannot mean
/// one thing on the draw that built it and another on every draw after.
fn bound_buffer_content(
    bound: &crate::runtime::bound_buffers::BoundBuffer,
) -> reims_vgpu_core::BufferContent {
    reims_vgpu_core::BufferContent::GuestRuns(reims_vgpu_memory::GuestRunSource {
        runs: std::sync::Arc::clone(&bound.runs),
        source_offset: bound.source_offset,
        total_len: bound.span,
        row_length_texels: 0,
        pages: bound.pages.clone(),
        physical_pages: bound.physical_pages.clone(),
    })
}

/// Derive one buffer-plus-offset bind from the retained whole-resource
/// allocation. The registry owns address resolution; the returned source owns
/// only the execution references needed to keep that allocation live.
fn packed_buffer_content(
    state: &mut Device,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    extent_cap: Option<u64>,
) -> Option<reims_vgpu_core::BufferContent> {
    let packed = match state.bound_buffers.packed(task_id, buffer_ref)? {
        crate::runtime::bound_buffers::PackedBufferResolution::Available(packed) => packed,
        crate::runtime::bound_buffers::PackedBufferResolution::Unavailable { .. } => return None,
    };
    let full = packed.size.checked_sub(offset).filter(|&span| span != 0)?;
    let span = extent_cap.map_or(full, |cap| full.min(cap));
    let source = state
        .bound_buffers
        .packed_buffer_source(task_id, buffer_ref, offset, span)?;
    Some(reims_vgpu_core::BufferContent::GuestRuns(source))
}

/// Load one draw-time buffer bind: the zero-copy rail when allowed and
/// eligible, else the CPU staging read. `allow_zero_copy` is false for
/// buffers feeding Constant-step attributes (the engine prepends a CPU
/// base-instance prefix to those).
///
/// # The `zc_buffer_*` route family, and what it is for
///
/// Every bind that reaches here with `allow_zero_copy` and a resolvable
/// backing takes **exactly one** route, so the family sums to the attempts:
///
/// ```text
/// held                                    the registry answered
/// offset_past_end + span_unusable         the descriptor disagrees with itself
/// no_alias + span_unmapped + untileable   the rail was tried and refused
/// imported + gathered                     the rail ran
/// ```
///
/// The split exists to answer one question the held-resolution registry cannot:
/// **how much of the per-draw cost is being paid over and over.** A resolution
/// is cached; a refusal is not. So a reference in the last-but-one group
/// re-walks the task page table *and* re-pays the CPU staging read on every
/// draw the guest binds it, for as long as it keeps binding it — and before
/// these routes existed, that path was the only outcome in this function with
/// no name at all. A steady rate there is repeats, because the guest's live
/// reference set is bounded and the bind rate is not.
///
/// Only `span_unmapped` is a refusal a mapped-range record could answer without
/// walking. `untileable` walked successfully and still could not bind, so
/// nothing upstream of the walk would have saved it — which is why the two are
/// counted apart rather than as one "the window failed".
///
/// `extent_cap` is the byte extent the shader on this draw proved it cannot read
/// past, from the executor's reflected-buffer extent service. `None`
/// keeps the whole-allocation window this function has always bound. It is part
/// of the registry key rather than of the resolution, because it describes the
/// shader and not the bind — see [`crate::runtime::bound_buffers`].
///
/// Answer a retained zero-copy resolution without touching the object table or
/// walking the task page table.
///
/// # The packed rail answers first, and it re-derives what the registry holds
///
/// The two rungs are not equivalent. [`bound_buffer_content`] hands back a
/// `BoundBuffer`'s prebuilt fields — including its `physical_pages`, already
/// canonicalised — as a few `Arc` clones. [`packed_buffer_content`] derives
/// the window from the retained packed allocation on every call, and that
/// derivation ends in `GuestPageSet::new`, which copies, sorts, deduplicates
/// and hashes the window's guest-physical page list and then copies it again
/// into an `Arc<[u64]>`.
///
/// Retaining every derived window was measured and reverted. One fullscreen
/// Maps boot did halve vertex resolution (1.555 to 0.794 µs/draw), but it held
/// 30 705 windows over 2 549 resources, including 3 313 offsets for one live
/// resource. Visibility/barrier work doubled from 0.82 to 1.62 µs/draw, slot
/// work nearly doubled from 1.06 to 2.03, and the whole drain regressed from
/// 18.73 to 29.18 µs/draw. The derived window is therefore the wrong ownership
/// unit even though its content stays live; a repair has to avoid materialising
/// one entry per offset rather than move that population into this registry.
fn held_buffer_content(
    state: &mut Device,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    extent_cap: Option<u64>,
) -> Option<reims_vgpu_core::BufferContent> {
    if let Some(content) = packed_buffer_content(state, task_id, buffer_ref, offset, extent_cap) {
        crate::runtime::drain::note_store_route("zc_buffer_held_packed");
        return Some(content);
    }
    // The registry is keyed on the same cap the walk uses, or a lookup could
    // answer with a shorter span than this shader needs.
    // A held resolution answers before anything is resolved at all: the walk
    // below produces the same runs until the guest moves the addresses, and it
    // announces every such move. This is the whole point of the registry — see
    // `crate::runtime::bound_buffers`.
    if let Some(bound) = state
        .bound_buffers
        .get(task_id, buffer_ref, offset, extent_cap)
    {
        let content = bound_buffer_content(bound);
        crate::runtime::drain::note_store_route("zc_buffer_held");
        return Some(content);
    }
    None
}

/// Whether this bind may take the zero-copy buffer rail at all.
///
/// The caller's own answer, narrowed by the operator's. `REIMS_VGPU_BUFFER_IMPORT=off`
/// may only take a bind *off* the rail; it can never put one on that the caller
/// refused, which is the direction `AGENTS.md` requires of every override.
fn buffer_zero_copy_allowed(requested: bool) -> bool {
    requested
        && !matches!(
            crate::env::switch(crate::env::BUFFER_IMPORT),
            crate::env::Switch::Off
        )
}

/// Resolve a previously validated backing through the zero-copy ladder, with
/// the CPU read as the capability fallback.
// Hot buffer-load path: the arguments are the decoded bind plus the host and
// device state it resolves against, and threading a struct through would only
// rename them.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_buffer_content_resolved<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    offset: u64,
    allow_zero_copy: bool,
    extent_cap: Option<u64>,
    backing: &BufferBacking,
) -> Option<reims_vgpu_core::BufferContent> {
    let allow_zero_copy = buffer_zero_copy_allowed(allow_zero_copy);
    // Resolve the backing (object-list entry + descriptor) once and share it
    // between the zero-copy attempt and the CPU fallback.
    if allow_zero_copy {
        if offset < backing.size
            && crate::runtime::bound_buffers::ensure_packed_resource(
                state,
                host,
                task_id,
                buffer_ref,
                backing.gva,
                backing.size,
                crate::runtime::bound_buffers::PackedResourceUse::Buffer,
            )
        {
            if let Some(content) =
                packed_buffer_content(state, task_id, buffer_ref, offset, extent_cap)
            {
                crate::runtime::drain::note_store_route("zc_buffer_imported_packed");
                return Some(content);
            }
        }
        if let Some(bound) =
            try_buffer_zero_copy_resolved(state, host, task_id, backing, offset, extent_cap)
        {
            let content = bound_buffer_content(&bound);
            state
                .bound_buffers
                .insert(task_id, buffer_ref, offset, extent_cap, bound);
            return Some(content);
        }
    }
    let bytes = read_buffer_bytes_resolved(
        state, host, task_id, buffer_ref, backing, offset, extent_cap,
    )?;
    Some(reims_vgpu_core::BufferContent::from(bytes))
}

// Same shape as `load_buffer_content_resolved`, which this forwards to.
#[allow(clippy::too_many_arguments)]
pub(super) fn load_buffer_content<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    buffer_ref: u32,
    resource: Option<&crate::model::TaskResource>,
    offset: u64,
    allow_zero_copy: bool,
    extent_cap: Option<u64>,
) -> Option<reims_vgpu_core::BufferContent> {
    let allow_zero_copy = buffer_zero_copy_allowed(allow_zero_copy);
    if allow_zero_copy {
        if let Some(content) = held_buffer_content(state, task_id, buffer_ref, offset, extent_cap) {
            return Some(content);
        }
    }
    // Resolve the backing (object-list entry + descriptor) once and share it
    // between the zero-copy attempt and the CPU fallback.
    let backing = resolve_buffer_backing(state, host, task_id, buffer_ref, resource)?;
    load_buffer_content_resolved(
        state,
        host,
        task_id,
        buffer_ref,
        offset,
        allow_zero_copy,
        extent_cap,
        &backing,
    )
}

/// Retain an indexed draw's exact guest-buffer window for the Vulkan vertex
/// input stage. Unlike the Metal fallback, this does not materialize the index
/// array on the CPU: Vulkan consumes the bounded resource directly when the
/// command buffer executes.
pub(super) fn load_index_content_reason<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    info: &IndexedDrawInfo,
) -> Result<reims_vgpu_core::BufferContent, IndexLoadReason> {
    let window = resolve_index_window_reason(state, host, task_id, info)?;
    let extent = Some(window.len as u64);
    if let Some(content) = held_buffer_content(
        state,
        task_id,
        info.index_buffer_ref,
        window.byte_offset,
        extent,
    ) {
        return Ok(content);
    }
    load_buffer_content_resolved(
        state,
        host,
        task_id,
        info.index_buffer_ref,
        window.byte_offset,
        true,
        extent,
        &window.backing,
    )
    .ok_or(IndexLoadReason::ReadFail)
}

/// The guest bytes one GVA render target occupies, as the rails that ask about
/// it name them.
///
/// One value rather than five parameters because the five only mean anything
/// together — a stride belongs to a height, and a format decides the channel
/// order the registry keys a resident on — and because two callers assembling
/// the same five by hand is how they come to disagree about one of them.
#[derive(Clone, Copy, Debug)]
pub(super) struct GvaSpan {
    pub texture_ref: u32,
    pub gva: u64,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    /// The guest's declared pixel format, not a host one:
    /// [`gva_resident_format`] turns it into the `format` half of the key.
    pub format: u16,
}

/// Why a GVA span's resident may not stand in for its guest pages.
///
/// Kept apart from a bare `None` because the three have nothing in common: one
/// is a span with no identity at all, one is a target something has written
/// since the Store, and one is a target the engine no longer holds. Each caller
/// names them on its own census routes — the rule is shared, the vocabulary is
/// not, so a reading says which rung refused as well as why.
pub(super) enum GvaResidentRefusal {
    /// The resource has no complete initial transfer backing, so it has no
    /// usable resident identity yet.
    NoGeneration,
    /// The witness will not call the pages quiet.
    Wrote(crate::runtime::gva_store_witness::GvaWriteReach),
    /// Quiet, but the engine is not holding an image under this identity.
    NoResident,
}

fn retained_resident_is_ready(
    backing: Option<reims_vgpu_core::ResidentContentBacking>,
    registry_query: impl FnOnce() -> bool,
) -> bool {
    use reims_vgpu_core::ResidentContentBacking;

    match backing {
        Some(ResidentContentBacking::GuestAllocation) => true,
        Some(ResidentContentBacking::DeviceAllocation) => true,
        Some(ResidentContentBacking::NotReady) => false,
        None => registry_query(),
    }
}

/// Whether the resident named by a GVA texture is still usable.
///
/// A named texture owns a retained allocation lease, so warm binds answer from
/// the texture object without re-entering the global engine. Anonymous and
/// unclassified spans have no protocol lifetime to hold such a lease and keep
/// the fail-closed registry query.
pub(super) fn gva_resident_ready(
    state: &Device,
    task_id: u32,
    texture_ref: u32,
    identity: &crate::model::TargetIdentity,
) -> bool {
    let backing = (texture_ref != 0)
        .then(|| state.task_objects.resources.get(task_id, texture_ref))
        .flatten()
        .filter(|resource| resource_type_owns_gva_resident(resource.entry().kind))
        .map(|resource| {
            state
                .executor
                .retain_resident_resource(resource.lifetime_ref(), identity)
        });
    let retained = backing.is_some();
    let ready = retained_resident_is_ready(backing, || {
        state.executor.resident_read_plan(identity).backing
            != reims_vgpu_core::ResidentContentBacking::NotReady
    });
    crate::runtime::drain::note_store_route(match (retained, ready) {
        (true, true) => "gva_ready_resource",
        (true, false) => "gva_not_ready_resource",
        (false, true) => "gva_ready_registry",
        (false, false) => "gva_not_ready_registry",
    });
    ready
}

/// The one currency test behind every GVA resident shortcut: does the engine
/// still hold, under this span's own identity, what the render Store published
/// into these guest pages?
///
/// Two rails ask it — the sampled bind below and the colour LOAD seed — and it
/// is written once because a copied version of this rule is the next
/// divergence. The callers differ only in what they do with the answer.
///
/// A named resource uses its stable host-texture generation and retained
/// transfer backing. The page-set fallback remains only for an attachment with
/// no resource reference and therefore no protocol lifetime to carry.
pub(super) fn gva_resident_if_current<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    span: GvaSpan,
) -> Result<crate::model::TargetIdentity, GvaResidentRefusal> {
    use crate::runtime::gva_store_witness::{reach, GvaTargetKey};

    let GvaSpan {
        texture_ref,
        gva,
        row_stride,
        width: w,
        height: h,
        format,
    } = span;
    if gva == 0 || w == 0 || h == 0 {
        return Err(GvaResidentRefusal::NoGeneration);
    }
    let span_bytes = u64::from(row_stride).saturating_mul(u64::from(h));
    let generation = if texture_ref != 0 {
        crate::runtime::writeback_debt::resource_key(state, task_id, texture_ref)
            .map(|key| {
                crate::runtime::writeback_debt::gva_resource_generation(
                    state, host, key, gva, span_bytes,
                )
            })
            .unwrap_or(0)
    } else {
        gva_span_alloc_generation(state, host, task_id, gva, row_stride, h)
    };
    if generation == 0 {
        return Err(GvaResidentRefusal::NoGeneration);
    }
    let resident_format = gva_resident_format(state.executor.as_ref(), format);
    let identity = crate::model::TargetIdentity::Gva {
        gva,
        width: w,
        height: h,
        generation,
        format: resident_format,
    };
    // An unpaid Store says the guest pages are deliberately stale and this
    // image is authoritative. The older witness below answers the opposite
    // state: a Store was copied out and both locations still agree. Keeping
    // those states distinct prevents a skipped copy from masquerading as a
    // statement about bytes that were never written.
    if crate::runtime::writeback_debt::gva_resident_authoritative(state, &identity) {
        return gva_resident_ready(state, task_id, texture_ref, &identity)
            .then_some(identity)
            .ok_or(GvaResidentRefusal::NoResident);
    }
    let key = crate::runtime::writeback_debt::resource_key(state, task_id, texture_ref)
        .ok_or(GvaResidentRefusal::NoGeneration)?;
    let verdict = reach(
        state,
        GvaTargetKey {
            task_id,
            resource: key.resource,
            gva,
            generation,
            width: w,
            height: h,
            // From the identity built just above, not from `resident_format`
            // again. `GvaTargetKey::of` builds this same key from the same
            // identity on the other side of the witness, and the two must
            // agree — a channel order written by hand at two sites is how they
            // stop agreeing.
            bgra: identity.is_bgra(),
        },
    );
    if !verdict.is_quiet() {
        return Err(GvaResidentRefusal::Wrote(verdict));
    }
    gva_resident_ready(state, task_id, texture_ref, &identity)
        .then_some(identity)
        .ok_or(GvaResidentRefusal::NoResident)
}

#[cfg(test)]
mod gva_resident_ownership_tests {
    use super::*;
    use reims_vgpu_core::ResidentContentBacking;
    use std::cell::Cell;

    #[test]
    fn a_retained_texture_answers_readiness_without_a_registry_query() {
        assert!(retained_resident_is_ready(
            Some(ResidentContentBacking::GuestAllocation),
            || panic!("a retained guest allocation must not query the registry")
        ));
        assert!(retained_resident_is_ready(
            Some(ResidentContentBacking::DeviceAllocation),
            || panic!("a retained allocation must not query the registry")
        ));
        assert!(!retained_resident_is_ready(
            Some(ResidentContentBacking::NotReady),
            || panic!("a named resource's failed retain is already authoritative")
        ));
    }

    #[test]
    fn an_anonymous_gva_span_keeps_the_registry_fallback() {
        let queries = Cell::new(0_u32);
        assert!(retained_resident_is_ready(None, || {
            queries.set(queries.get() + 1);
            true
        }));
        assert_eq!(queries.get(), 1);
    }
}

/// May the colour LOAD seed at this GVA attachment be skipped, because the
/// engine still holds what the render Store published into these pages?
///
/// Answering `true` **obliges the encode side** to chain or to re-seed:
/// `colors[0].target_seed_rgba` goes out `None` while the attachment still says
/// LOAD, so a pass that does neither loads an undefined attachment.
/// `try_metal2vulkan_draw` owns that obligation.
///
/// # Why the rung this replaces was thought not to exist
///
/// `settle_linear_texture_seed` is the device's largest remaining wait — 4 701
/// per driven drag, **4 692 of them genuine overlaps**, because a
/// `MTLLoadActionLoad` over a GVA target reads the attachment's own guest pages
/// on the CPU while the render Store that published them is still writing. The
/// sampled twin of that wait was removed by `try_gva_resident_sample`, and the
/// same currency test applies here.
///
/// A cross-pass version of this rung existed and was **deleted for reading
/// zero**. That zero was an artifact of where it was sampled: it sat downstream
/// of `mrt_draw_request`, which produced the seed *eagerly* for every GVA LOAD,
/// so by the time it ran no draw had a seedless GVA LOAD target and its
/// denominator was empty by construction. It could not have fired whatever the
/// guest did.
///
/// Asked here instead — at the production site, before `seed_color_load` reads
/// anything — the same question answers, on one driven Safari drag against
/// `load_seed_ok_color` 4 862, which was every colour LOAD seed that boot
/// produced:
///
/// ```text
/// gvaseed_elided       4 849   99.7 %
/// gvaseed_not_quiet       11
/// gvaseed_no_resident      2
/// gvaseed_no_generation    0
/// ```
///
/// # What eliding them did
///
/// ```text
///                                    before    after
/// load_seed_ok_color                  4 862       11
/// settle_linear_texture_seed (waits)  4 792        3
/// settle_linear_texture_seed_us        1.69 s   11 ms
/// fence (waits)                       6 403    3 136
/// fence_us                             6.17 s   4.88 s
/// ```
///
/// `gvaseed_chained` equalled `gvaseed_elided` exactly — 4 475 of each — so
/// every elision was honoured at encode time and `gvaseed_reseeded` never
/// fired. The race is real and must stay handled; it is simply not hot.
///
/// Correctness was not taken from a screenshot, because this class renders a
/// *plausible* frame when it is wrong and this file records a reverted attempt
/// at it that gave a black screen with orange fragments. The multi-round
/// recomposite run over a live Wikipedia article scored **PATCHED none,
/// UNSCOREABLE none** with both its gates satisfied, on five CLEAN offsets and
/// one CHURN.
///
/// # The copying arm never reaches this, and that is the design
///
/// `gva_store_witness` is armed only by the GPU-direct Store rail, so a host
/// without the guest-RAM import stamps nothing and this can never answer yes.
/// Confirmed rather than argued: a `REIMS_VGPU_GUEST_IMPORT=off` boot reads
/// `gvaseed_not_quiet` **3 246 against `load_seed_ok_color` 3 246** — every
/// seed built, none elided — with `gvaseed_elided` and `gvarung_resident` both
/// absent and zero bound imports. That arm keeps the behaviour it had before
/// either rung existed.
pub(super) fn gva_load_seed_elidable<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    span: GvaSpan,
) -> bool {
    use crate::runtime::drain::note_store_route;
    let answer = gva_resident_if_current(state, host, task_id, span);
    note_store_route(match answer {
        Ok(_) => "gvaseed_elided",
        Err(GvaResidentRefusal::NoGeneration) => "gvaseed_no_generation",
        Err(GvaResidentRefusal::Wrote(_)) => "gvaseed_not_quiet",
        Err(GvaResidentRefusal::NoResident) => "gvaseed_no_resident",
    });
    answer.is_ok()
}

pub(super) fn try_gva_resident_sample<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    resource: &crate::model::TaskResource,
    tex: &TextureDescriptor,
) -> Option<(u32, u32, SampledSourceRequest)> {
    use crate::runtime::drain::note_store_route;

    if !resource.was_render_target() {
        note_store_route("gvarung_sampled_only");
        return None;
    }

    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let (w, h) = (layout.width, layout.height);
    let row_stride = u32::try_from(layout.row_stride).ok()?;
    let declared_format = tex.declared_pixel_format()?;
    let identity = match gva_resident_if_current(
        state,
        host,
        task_id,
        GvaSpan {
            texture_ref,
            gva,
            row_stride,
            width: w,
            height: h,
            format: declared_format,
        },
    ) {
        Ok(identity) => identity,
        Err(GvaResidentRefusal::NoGeneration) => return None,
        Err(GvaResidentRefusal::Wrote(verdict)) => {
            note_store_route(verdict.route());
            return None;
        }
        Err(GvaResidentRefusal::NoResident) => {
            note_store_route("gvarung_resident_absent");
            return None;
        }
    };
    let format = pixel_format::sampled_image_format(declared_format)?;
    note_store_route("gvarung_resident");
    Some((w, h, SampledSourceRequest::Target(identity, format)))
}

/// Which repair would let the linear zero-copy rung carry this format, as a
/// route name.
///
/// The two named arms are the two ways [`pixel_format::sampled_texel_checked`]
/// declines, and each points at different work: teach the decode contract an
/// ordinal, name a byte layout for a format the contract already defines, or
/// give the image view a component mapping. `_other` is a healthy zero — a
/// firing means `sampled_pixels` grew a fourth decline that this split does not
/// name, not that a format was lost.
fn zc_lin_no_layout_route(reason: pixel_format::SampledTexelDecline) -> &'static str {
    use pixel_format::SampledTexelDecline as R;
    match reason {
        R::UnknownPixelFormat { .. } => "zc_lin_no_layout_undefined_format",
        R::NoSampledLayout { .. } => "zc_lin_no_layout_no_texel_layout",
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LinearSampleSelection {
    pub(super) level: u32,
    pub(super) pixel_format: Option<u16>,
    pub(super) texture_type: Option<u16>,
    pub(super) range: Option<TextureViewRange>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearMipRangeDecline {
    /// The requested view needs a mip chain, but no materialization preserving
    /// that complete chain was available. A one-level byte upload or resident
    /// render target is not an equivalent representation.
    CompleteAllocationUnavailable {
        texture_ref: u32,
        base_mip_level: u64,
        mip_level_count: u64,
    },
}

impl crate::observe::Decline for LinearMipRangeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::CompleteAllocationUnavailable { .. } => {
                "linear_mip_complete_allocation_unavailable"
            }
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::CompleteAllocationUnavailable {
                texture_ref,
                base_mip_level,
                mip_level_count,
            } => vec![
                ("ref", texture_ref.to_string()),
                ("base_mip", base_mip_level.to_string()),
                ("mip_count", mip_level_count.to_string()),
            ],
        }
    }
}

reims_vgpu_observe::decline_display!(LinearMipRangeDecline);

pub(super) fn requested_linear_mip_range(
    tex: &TextureDescriptor,
    selection: LinearSampleSelection,
) -> (u64, u64) {
    selection.range.map_or_else(
        || (0, u64::from(tex.mipmap_level_count.max(1))),
        |range| (range.level_base, range.level_count),
    )
}

/// What a linear sampled bind learned when it asked whether its guest
/// allocation can be bound directly.
///
/// This is the largest routing decision in the sampled path. A bind admitted
/// here aliases the guest allocation and reads the live storage; one that is not
/// copies the whole texture every time it is bound, and then relies on a gather
/// vouch — which `runtime/gather_witness.rs` says in its own first paragraph is
/// not a statement about bytes.
///
/// It is an enum rather than an `Option` because three unrelated things used to
/// collapse into one absence: the backend answering that it cannot represent the
/// layout, the backend answering nothing at all, and there being no packed
/// allocation to form a question about. None of the three was counted, so the
/// copy rail's dominant cause was invisible — a boot could report zero refusals
/// from every named term and still run 71 % of its sampled binds through a copy.
/// Naming them separately is what makes the next reading say *which*.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum LinearDirectAdmission {
    /// The backend will bind this allocation directly, subject to `requirement`.
    Admitted(reims_vgpu_memory::GuestImageBindingRequirement),
    /// The backend was asked and answered that it cannot represent this layout.
    BackendRefused,
    /// The backend was asked and returned no disposition at all.
    BackendSilent,
    /// There was no question to ask: this bind has no packed guest allocation or
    /// no image contract, so no binding request could be formed.
    NoBindingRequest,
}

impl LinearDirectAdmission {
    /// The census route this outcome is counted under.
    ///
    /// The three refusing arms sum with `lin_direct_admitted` to the number of
    /// binds that reached the admission question at all, which is what makes a
    /// missing term visible rather than absorbed.
    pub(super) fn route(&self) -> &'static str {
        match self {
            Self::Admitted(_) => "lin_direct_admitted",
            Self::BackendRefused => "lin_direct_backend_refused",
            Self::BackendSilent => "lin_direct_backend_silent",
            Self::NoBindingRequest => "lin_direct_no_binding_request",
        }
    }

    /// The requirement to satisfy, when this bind was admitted.
    pub(super) fn requirement(self) -> Option<reims_vgpu_memory::GuestImageBindingRequirement> {
        match self {
            Self::Admitted(requirement) => Some(requirement),
            Self::BackendRefused | Self::BackendSilent | Self::NoBindingRequest => None,
        }
    }
}

pub(super) fn direct_binding_requirement(
    disposition: Option<reims_vgpu_memory::GuestImageBindingDisposition>,
    asked: bool,
) -> LinearDirectAdmission {
    match disposition {
        Some(reims_vgpu_memory::GuestImageBindingDisposition::Direct(requirement)) => {
            LinearDirectAdmission::Admitted(requirement)
        }
        Some(reims_vgpu_memory::GuestImageBindingDisposition::Refused) => {
            LinearDirectAdmission::BackendRefused
        }
        None if asked => LinearDirectAdmission::BackendSilent,
        None => LinearDirectAdmission::NoBindingRequest,
    }
}

fn refuse_unmaterialized_mip_range(
    texture_ref: u32,
    tex: &TextureDescriptor,
    selection: LinearSampleSelection,
) -> bool {
    let (base_mip_level, mip_level_count) = requested_linear_mip_range(tex, selection);
    if mip_level_count == 1 {
        return false;
    }
    crate::runtime::drain::note_store_route("lin_mip_complete_allocation_unavailable");
    crate::observe::Emit::decline(
        "linear_sample",
        &LinearMipRangeDecline::CompleteAllocationUnavailable {
            texture_ref,
            base_mip_level,
            mip_level_count,
        },
    )
    .fail_once(u64::from(texture_ref));
    true
}

/// The guest's own statement about a linear texture's content, or `None` where
/// the guest never makes one.
///
/// The resource's content version advances on a decoded validity transition —
/// the guest telling this device it wrote. That statement exists only where
/// `MTLStorageMode` obliges the guest to make it, which is `Managed` alone. On
/// every other mode the guest still CPU-writes the texture (the mode is an
/// announcement contract, not an access contract) and the version sits still
/// while it does, so handing that version to the witness would let an unmoved
/// number be read as "no write happened" when it only ever meant "nobody told
/// me". The witness fails closed on `None`, so the silent modes lose the
/// memoization and re-read their bytes.
///
/// This is what a CPU-rasterized glyph atlas needs: it is written by the guest
/// between draws with no announcement, and a vouch taken over it freezes the
/// sampled copy at whatever the atlas held when it was first gathered.
fn stated_task_resource_generation(
    state: &Device,
    tex: &TextureDescriptor,
    resource: reims_vgpu_protocol::ResourceId<reims_vgpu_protocol::ResourceObject>,
) -> Option<crate::runtime::gather_witness::StatedGeneration> {
    match tex.guest_write_announcement() {
        reims_vgpu_protocol::GuestWriteAnnouncement::Announced => Some(
            crate::runtime::gather_witness::StatedGeneration::TaskResource(
                state.resource_write_stamp_for(resource)?,
            ),
        ),
        reims_vgpu_protocol::GuestWriteAnnouncement::Silent => {
            crate::runtime::drain::note_store_route("gw_stated_silent_mode");
            None
        }
    }
}

/// Which rung of the linear sampled fork served one guest texture.
///
/// The three arms below are the three rungs in `resolve_sampled_source`'s
/// linear branch, in the order it tries them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum LinearSampleRung {
    /// A render Store already published these pages as an engine image.
    Resident,
    /// The zero-copy rail: the engine reads the guest's own pages.
    ZeroCopy,
    /// The CPU rung: this device reads the guest bytes and converts them.
    HostRead,
    /// None of the three rungs served this bind, so it falls past the linear
    /// branch to the descriptor-driven last-resort rung below.
    ///
    /// The other three are outcomes of the fork; this is the absence of one,
    /// and it was invisible until a boot needed it. A linear bind that reaches
    /// the last-resort rung has read guest bytes on the CPU and rebuilt them
    /// per bind, so a rail sitting here is both the slow answer and the one
    /// whose content nothing above compared.
    Unserved,
}

impl LinearSampleRung {
    fn route(self) -> &'static str {
        match self {
            Self::Resident => "lingeom_resident",
            Self::ZeroCopy => "lingeom_zero_copy",
            Self::HostRead => "lingeom_host_read",
            Self::Unserved => "lingeom_unserved",
        }
    }

    fn slug(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::ZeroCopy => "zero_copy",
            Self::HostRead => "host_read",
            Self::Unserved => "unserved",
        }
    }
}

/// Record the geometry one linear sampled bind resolved, and which rung took it.
///
/// This is an instrument and decides nothing. It exists because the two
/// guest-memory arms — the zero-copy rail and the CPU rung the copying host
/// runs — are known to render Maps' type layer differently while every counter
/// in the tree reads clean on both, and no reading in the tree says whether the
/// two arms compute the *same geometry* for the same guest texture. Both arms
/// pass through this one fork, so a record emitted here is directly comparable
/// between two boots: run each arm to the same scene and diff the two multisets.
///
/// Deduped per `(gva, declared format)`, which is a handful of lines a boot,
/// while the route counters beside it carry the volume. The pair is deliberate:
/// a count says how much traffic a rung took and the line says what shape it
/// took, and `AGENTS.md` records that quoting one as the other is a mistake this
/// repository has already made.
fn note_linear_sample_geometry(
    state: &Device,
    task_id: u32,
    texture_ref: u32,
    tex: &TextureDescriptor,
    rung: LinearSampleRung,
) {
    crate::runtime::drain::note_store_route(rung.route());
    let Some((gva, layout)) = tex.level_gva(0, state.page_shift) else {
        return;
    };
    let Some(format) = tex.declared_pixel_format() else {
        return;
    };
    if !crate::observe::first_sight("lin_geom", gva ^ (u64::from(format) << 48)) {
        return;
    }
    crate::observe::off(format!(
        "lin_geom rung={} task={task_id} ref={texture_ref} gva={gva:#x} \
         {}x{} pitch={} fmt={format:#x} mips={} usage={:#x} announce={:?}",
        rung.slug(),
        layout.width,
        layout.height,
        layout.row_stride,
        tex.mipmap_level_count.max(1),
        tex.declared_usage().unwrap_or(0),
        tex.guest_write_announcement(),
    ));
}

/// Record the byte window one linear sampled bind resolved for its gather.
///
/// Three sites build that window — a packed import plus an image contract, a
/// packed import without one, and a task-GVA walk when no import exists — and
/// they are reached by different hosts, so no boot exercises more than two of
/// them. `AGENTS.md` names that shape as the one to diff arm against arm rather
/// than read alone, and the levels of an image contract are *allocation*
/// relative, so a window based anywhere but the allocation's own start applies
/// those offsets to the wrong bytes.
///
/// Deduped per `(gva, site)` so the arms can be compared source by source.
/// Purely an observation: nothing here reaches a decision.
/// Judge a **reused** packed view's recorded page GPAs against a live walk of
/// the task page table.
///
/// `bound_buffers::audit_view_gpas_against_page_table` asks the same question at
/// *construction*, and says in its own doc that it therefore "cannot see a page
/// table edited after an alias was built". That blind spot is not a detail: the
/// construction audit's large `agree` beside a zero `differs` was read as closing
/// the question of *where* the sampled bytes come from, and it cannot see the one
/// way the answer goes wrong after the fact. A guest that frees a small texture
/// and places different physical pages at the same address — which is what a
/// window server does continuously with rasterized type — moves the page table
/// under a view this device already built and caches.
///
/// So this is the reuse-time half. It decides nothing and it is not a gate: a
/// disagreement is reported and the bind proceeds exactly as before, because the
/// point is to measure whether the class occurs at all before anything is built
/// on the answer.
///
/// It walks the guest page table on every bind it runs for, so it runs only when
/// the operator has already asked for content-level scrutiny with
/// `REIMS_VGPU_GATHER_AUDIT_ALL=on`. There is deliberately **no stride**: a
/// sampling rate here would report a rate nobody could interpret, and
/// `AGENTS.md` bans one for exactly that reason.
fn audit_packed_view_on_reuse<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    gva: u64,
    view_gpas: &[u64],
) {
    use crate::runtime::gather_witness::AuditDensity;
    if matches!(
        crate::runtime::gather_witness::audit_density(),
        AuditDensity::Disabled
    ) {
        return;
    }
    let page = state.page_size();
    if page == 0 || view_gpas.is_empty() {
        return;
    }
    let page_base = gva - (gva % page);
    let map_len = (view_gpas.len() as u64) * page;
    let table = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        page_base,
        map_len,
        state.page_shift,
    );
    if table.len() < view_gpas.len() {
        crate::runtime::drain::note_store_route("linview_reuse_short");
        return;
    }
    match view_gpas
        .iter()
        .zip(table.iter())
        .enumerate()
        .find(|(_, (view, live))| view != live)
    {
        None => crate::runtime::drain::note_store_route("linview_reuse_agree"),
        Some((index, (view_gpa, live_gpa))) => {
            crate::runtime::drain::note_store_route("linview_reuse_differs");
            if crate::observe::first_sight("linview_reuse", gva) {
                crate::observe::fail(format!(
                    "linview_reuse_differs task={task_id} gva={gva:#x} page={index} \
                     view_gpa={view_gpa:#x} live_gpa={live_gpa:#x} pages={} \
                     (the packed view this bind samples names a physical page the \
                     guest no longer places at this address)",
                    view_gpas.len()
                ));
            }
        }
    }
}

/// Compare the **bytes** the import path would hand the gather against a live
/// page-table read of the same guest window.
///
/// [`audit_packed_view_on_reuse`] compares page *numbers* and reads 1 364 726
/// agree against zero. That is a real closure of one question and it does not
/// close this one: a packed view's host pointer is
/// `import.host_base() + head + offset`, and every one of those terms can be
/// wrong while the page list is right. The remapped alias the PCI shim builds
/// places each shared page at its *resource* offset inside a reserved range, so
/// an error in `head`, in the within-page offset, or in which reserved range the
/// resource was given, moves the read without moving a single GPA.
///
/// So this reads the window both ways and compares the bytes themselves. The
/// CPU walk is the same one the copying arm uses for every bind, which makes
/// this an arm-against-arm comparison inside one boot rather than across two.
///
/// An instrument: it decides nothing, it is not a gate, and the bind proceeds
/// unchanged either way. It reads the whole window on the CPU, so it runs only
/// under `REIMS_VGPU_GATHER_AUDIT_ALL=on`, and never with a stride.
fn audit_import_bytes_against_page_table<M: HostMemory>(
    state: &Device,
    host: &M,
    task_id: u32,
    gva: u64,
    host_ptr: usize,
    span: u64,
) {
    use crate::runtime::gather_witness::AuditDensity;
    if matches!(
        crate::runtime::gather_witness::audit_density(),
        AuditDensity::Disabled
    ) {
        return;
    }
    // Bound the compare so one enormous window cannot dominate the boot. This
    // is a prefix, not a sample: every window is judged, over its first pages.
    const COMPARE_LIMIT: u64 = 16 * 1024;
    let len = span.min(COMPARE_LIMIT);
    let Ok(len) = usize::try_from(len) else {
        return;
    };
    if len == 0 {
        return;
    }
    let mut walked = vec![0u8; len];
    if crate::runtime::gva_mem::try_read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut walked,
        state.page_shift,
    )
    .is_err()
    {
        crate::runtime::drain::note_store_route("linbytes_walk_unavailable");
        return;
    }
    // SAFETY: `host_ptr` is the import's own mapping of this window, obtained
    // from the packed view that produced `span`, and the guest RAM import is
    // held for the lifetime of the VM.
    let imported = unsafe { std::slice::from_raw_parts(host_ptr as *const u8, len) };
    match imported.iter().zip(walked.iter()).position(|(a, b)| a != b) {
        None => crate::runtime::drain::note_store_route("linbytes_agree"),
        Some(offset) => {
            crate::runtime::drain::note_store_route("linbytes_differ");
            if crate::observe::first_sight("linbytes_differ", gva) {
                crate::observe::fail(format!(
                    "linbytes_differ task={task_id} gva={gva:#x} span={span} \
                     first_diff={offset} imported={:#04x} walked={:#04x} \
                     (the import mapping and the task page table disagree about \
                     the bytes at this address)",
                    imported[offset], walked[offset]
                ));
            }
        }
    }
}

fn note_linear_sample_window(site: &'static str, gva: u64, base: u64, span: u64, row_len: u32) {
    crate::runtime::drain::note_store_route(match site {
        "packed_contract" => "linwin_packed_contract",
        "packed_plane" => "linwin_packed_plane",
        _ => "linwin_gva_walk",
    });
    if !crate::observe::first_sight("lin_window", gva ^ (span << 24)) {
        return;
    }
    crate::observe::off(format!(
        "lin_window site={site} gva={gva:#x} base={base} span={span} rowlen={row_len}"
    ));
}

fn try_linear_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    tex: &TextureDescriptor,
    sampled_shape: reims_vgpu_core::SampledImageShape,
    selection: LinearSampleSelection,
) -> Option<(u32, u32, SampledSourceRequest)> {
    // The object-list entry + descriptor are resolved+decoded once by the caller
    // (`resolve_sampled_source`'s linear branch) and threaded in as `tex`; the
    // cache fallback shares the same decode.
    let Some(base_format) = tex.declared_pixel_format() else {
        crate::runtime::drain::note_store_route("zc_lin_no_declared_format");
        return None;
    };
    let declared_format =
        match effective_view_sample_format_reasoned(base_format, selection.pixel_format) {
            Ok(format) => format,
            Err(reason) => {
                crate::observe::Emit::decline("zc_lin_view_format", &reason)
                    .field("ref", texture_ref)
                    .fail_once(u64::from(texture_ref) << 32 | u64::from(selection.level));
                return None;
            }
        };
    // sRGB variants ride the same rail as their linear siblings: the layout is
    // identical and the CPU loaders never decoded either. The qualifier is
    // still lost, so the census records it rather than letting the fold be
    // silent.
    // **Every layout `sampled_pixels` returns is admitted**, which is the same
    // rule `try_iosurface_plane_view_sample_zero_copy` states and applies: that function is
    // the answer to "which guest bytes sample byte-identically through the
    // matching Vulkan image", and a layout it hands back has already passed
    // the identity-components test inside it. The engine creates the image
    // with `vk_texel_layout(native)`, so the texel size and channel order come
    // from the same answer and cannot disagree with it.
    //
    // This rail used to narrow that set again, to four-byte colour plus the
    // single-channel floats, on the stated grounds that "R8/Rg8 video planes
    // keep their existing CPU/IOSurface plane view rails". The premise was that R8 and Rg8
    // only ever arrive as video; they do not. A Safari window drag with no
    // video playing produced 37 704 `Rg8` binds, 49 % of every linear sampled
    // bind in the boot, and each one fell to `load_linear_guest_memoized`'s
    // full-span guest re-read plus memcmp. The narrowing was never a
    // correctness rule — `texel_to_rgba8` expands `Rg8` to `(r, g, 0, 255)`
    // and an `R8G8_UNORM` image samples `(r, g, 0, 1)`, which is the same
    // texel — so what it bought was the CPU path on half the binds.
    //
    // `R32_SFLOAT` keeps its extra condition, and it is a host capability
    // rather than a layout one: LUTs are sampled with interpolation and that
    // format's linear-filter feature is optional (absent on Apple/MoltenVK).
    //
    // The two ways a format declines are separated, because they want opposite
    // fixes and a single count reads the same for both. `sampled_pixels`
    // answering `Err` means the *contract* carries no [`TexelLayout`] for the
    // format at all — either it is undefined, or its Metal channels do not sit
    // identically on their Vulkan ones, which is a component mapping this rail
    // does not yet carry. Answering `Ok` with a layout this rail does not admit
    // now means only the host filter gate below.
    //
    // The plan beside the layout is the format's own channel mapping — identity
    // for all but `A8Unorm`, whose byte rides in `R8_UNORM`. It is composed with
    // the guest's type-8 view swizzle where the image view is built, so this
    // rail no longer has to refuse a format for having one.
    let (native, sampled_format, native_components) =
        match pixel_format::sampled_texel_checked(declared_format) {
            // Deduped per declared format, which is a handful of values a boot
            // enumerates in a handful of lines. The number is the guest's own
            // `MTLPixelFormat` ordinal, so it names the format without this device
            // having to hold a second spelling of Apple's table.
            // The reason is kept, not discarded. `Err` here has three causes that
            // want three different repairs — the format is outside the decode
            // contract, the contract defines it but no rail names a byte layout for
            // it, or the layout exists but its channels need a swizzle — and a
            // single count reads the same for all three. The sub-route is the
            // reason's own slug, so it cannot drift from the core taxonomy, and
            // the total is still recorded beside it so the
            // split adds up.
            Err(reason) => {
                crate::runtime::drain::note_store_route("zc_lin_format_no_layout");
                crate::runtime::drain::note_store_route(zc_lin_no_layout_route(reason));
                if crate::observe::first_sight(
                    "zc_lin_format_no_layout",
                    u64::from(declared_format),
                ) {
                    crate::observe::off(format!(
                        "zc_lin_format_no_layout fmt={declared_format:#x} {reason} \
                     (no sampled TexelLayout; the bind falls to the CPU \
                     re-read + memcmp rung)"
                    ));
                }
                return None;
            }
            Ok((layout, components)) => {
                // Every layout is asked about, not just the one that was known to
                // be optional. This rail hands the guest's bytes to a sampler that
                // interpolates them, so "can this host filter this format" is a
                // question about the layout, and a table indexed by the layout
                // cannot be missing an entry for one that was added later.
                if !state
                    .executor
                    .sampled_layout_linear_filter_supported(layout)
                {
                    crate::runtime::drain::note_store_route("zc_lin_layout_unfilterable");
                    return None;
                }
                let format = pixel_format::sampled_image_format(declared_format)?;
                (layout, format, components)
            }
        };
    let bpp = native.bytes_per_texel();
    // Derive the allocation/view contract once. The same answer governs the
    // direct binding query, direct materialization and transfer fallback; none
    // of those layers may independently reinterpret the texture descriptor.
    let image_contract = declared_guest_image_allocation(
        sampled_shape,
        tex,
        selection.texture_type,
        selection.range,
        u64::from(bpp),
    );
    let Some((level_gva, layout)) = tex.level_gva(selection.level, state.page_shift) else {
        crate::runtime::drain::note_store_route("zc_lin_no_level_gva");
        return None;
    };
    let (w, h) = (layout.width, layout.height);
    if w == 0 || h == 0 {
        crate::runtime::drain::note_store_route("zc_lin_no_extent");
        return None;
    }
    let guest_selection = declared_guest_image_selection(
        sampled_shape,
        tex,
        layout,
        selection.texture_type,
        selection.range,
    );
    let guest_layout = guest_selection.map(|selection| selection.0);
    let subresource_offset = guest_selection.map_or(0, |selection| selection.1);
    let plane_offset = tex
        .base_offset
        .checked_add(layout.offset)?
        .checked_add(subresource_offset)?;
    let gva = level_gva.checked_add(subresource_offset)?;
    let row_length_texels = if layout.row_stride == u64::from(w).checked_mul(u64::from(bpp))? {
        0
    } else {
        u32::try_from(layout.row_stride.checked_div(u64::from(bpp))?).ok()?
    };
    if !layout.row_stride.is_multiple_of(u64::from(bpp)) {
        crate::runtime::drain::note_store_route("zc_lin_unstrideable");
        return None;
    }
    let span = match guest_layout {
        Some(image) => image.visible_span(layout.row_stride, u64::from(bpp)),
        None => strided_level_extent(layout, u64::from(bpp)).map(|(span, _)| span),
    }?;
    let planes = match guest_layout {
        Some(reims_vgpu_memory::GuestImageLayout::D1Array { layers, .. })
        | Some(reims_vgpu_memory::GuestImageLayout::D2Array { layers, .. }) => layers,
        Some(reims_vgpu_memory::GuestImageLayout::D3 { depth, .. }) => depth,
        _ => layout.planes(),
    };
    if tex.allocation_size != 0 && plane_offset.saturating_add(span) > tex.allocation_size {
        crate::runtime::drain::note_store_route("zc_lin_past_allocation");
        return None;
    }
    // No settle here, and that is the difference between this rail and the CPU
    // loaders it replaces.
    //
    // A CPU loader reads the guest's pages with this thread, which nothing
    // orders against a submitted-but-unexecuted writeback, so it has to block
    // until the writeback has landed. This rail does not read anything: it hands
    // the engine guest-RAM runs and the *GPU* reads them when the draw's command
    // buffer executes. A guest-page writeback is a GPU command on the same
    // single queue, and `copy_image_level0_to_buffer` submits it before
    // returning — it is already on the queue by the time the debt flag that a
    // settle consults is even set. Queue order therefore already puts the
    // writeback ahead of this gather, and a CPU fence wait buys an ordering that
    // holds without it.
    //
    // `try_iosurface_texture_sample_zero_copy` and `try_iosurface_plane_view_sample_zero_copy` are the two
    // rails that were already written this way, and this one is now consistent
    // with them.
    //
    // That argument is about a **submitted** writeback, and it does not extend
    // to an owed one: a writeback debt is a frame this device rendered and
    // deliberately did not write down, so there is no command on any queue for
    // queue order to order and the pages hold the frame before it. The payment
    // is what puts it on the queue; then the paragraph above applies again.
    //
    // Paid through the texture ref, the same call this rail's CPU twin makes,
    // because a linear texture's bytes may alias a surface this device owes a
    // frame and only `pay_for_texture` resolves one id namespace to the other.
    crate::runtime::writeback_debt::pay_for_texture(state, host, task_id, texture_ref);
    // Retain the texture's complete allocation once. A sampled image needs the
    // allocation base, level offset and row pitch together; reducing it to the
    // level's page runs would throw away the resource shape and force a copy.
    let allocation = tex
        .allocation_base_gva(state.page_shift)
        .filter(|_| tex.allocation_size != 0)
        .map(|allocation_gva| BufferBacking {
            gva: allocation_gva,
            size: tex.allocation_size,
        });

    // The witness recorder mutably borrows `state`, so the request owns its
    // lightweight retained-allocation handles across that call. This applies to
    // a direct-image candidate too: exact image admission belongs to the
    // backend, and if it declines the copied fallback needs the same identity
    let resource = state
        .task_objects
        .resources
        .identity(task_id, texture_ref)?;
    // every other packed source uses or it would upload on every bind.
    let packed_span = crate::runtime::sampled_phase::Span::open(
        crate::runtime::sampled_phase::Part::LinearPacked,
    );
    let mut packed = allocation.as_ref().and_then(|backing| {
        state
            .bound_buffers
            .packed_available(task_id, texture_ref, backing.gva, backing.size)
            .map(crate::runtime::bound_buffers::PackedBuffer::access)
    });
    if packed.is_none() {
        packed = allocation.as_ref().and_then(|backing| {
            crate::runtime::bound_buffers::ensure_packed_resource(
                state,
                host,
                task_id,
                texture_ref,
                backing.gva,
                backing.size,
                crate::runtime::bound_buffers::PackedResourceUse::LinearSample,
            )
            .then(|| {
                state
                    .bound_buffers
                    .packed_available(task_id, texture_ref, backing.gva, backing.size)
                    .map(crate::runtime::bound_buffers::PackedBuffer::access)
            })
            .flatten()
        });
    }
    drop(packed_span);

    // Vulkan's image memory requirement may extend beyond the guest's final
    // visible byte. Ask the backend for that exact extent before publishing
    // the direct image. The resource owner may then replace its host view with
    // guest pages followed by host-only padding; its footprint and writeback
    // remain bounded by the guest allocation.
    let mut direct_admitted = false;
    if guest_layout.is_none() {
        // No decoded guest image layout, so there is nothing to offer the direct
        // rail and the admission question below is never asked. Counted beside
        // the three admission outcomes so the four sum to every linear sampled
        // bind: a term missing from that sum is a route nobody is watching.
        crate::runtime::drain::note_store_route("lin_direct_no_guest_layout");
    }
    if guest_layout.is_some() {
        let _admission_span = crate::runtime::sampled_phase::Span::open(
            crate::runtime::sampled_phase::Part::LinearAdmission,
        );
        let binding_request = packed.as_ref().and_then(|current| {
            let (image_allocation, _) = image_contract.as_ref()?;
            let memory =
                linear_guest_image_allocation_memory(current, image_allocation, u64::from(bpp))?;
            Some(reims_vgpu_memory::GuestImageBindingRequest {
                backing: memory.backing,
                allocation: image_allocation.clone(),
                format: sampled_format,
            })
        });
        let binding_key = binding_request.as_ref().map(|request| request.key());
        let known_disposition = binding_key.as_ref().and_then(|key| {
            let backing = allocation.as_ref()?;
            state
                .bound_buffers
                .packed_available(task_id, texture_ref, backing.gva, backing.size)?
                .sampled_image_requirements
                .get(key)
                .copied()
        });
        let disposition = known_disposition.or_else(|| {
            state
                .executor
                .sampled_image_binding_requirement(binding_request.clone()?)
        });
        if known_disposition.is_none() {
            if let (Some(key), Some(disposition), Some(backing)) =
                (binding_key, disposition, allocation.as_ref())
            {
                state.bound_buffers.note_sampled_image_requirement(
                    task_id,
                    texture_ref,
                    backing.gva,
                    backing.size,
                    key.clone(),
                    disposition,
                );
            }
        }
        let admission = direct_binding_requirement(disposition, binding_request.is_some());
        crate::runtime::drain::note_store_route(admission.route());
        let requirement = admission.requirement();
        direct_admitted = requirement.is_some();
        if let (Some(requirement), Some(current), Some(backing)) =
            (requirement, packed.as_ref(), allocation.as_ref())
        {
            if requirement.allocation_len > current.import.len()
                && crate::runtime::bound_buffers::ensure_packed_resource_for_image(
                    state,
                    host,
                    crate::runtime::bound_buffers::PackedImageBinding {
                        task_id,
                        resource_ref: texture_ref,
                        gva: backing.gva,
                        size: backing.size,
                        required_import_len: requirement.allocation_len,
                        usage: crate::runtime::bound_buffers::PackedResourceUse::LinearSample,
                    },
                )
            {
                packed = state
                    .bound_buffers
                    .packed_available(task_id, texture_ref, backing.gva, backing.size)
                    .map(crate::runtime::bound_buffers::PackedBuffer::access);
            }
        }
    }

    if let Some(packed) = packed {
        if let Some((image_allocation, view)) = image_contract.as_ref() {
            let page = state.page_size();
            let witness = packed.witness_window(0, packed.size)?;
            let witness_runs = [reims_vgpu_memory::GuestRun {
                host_ptr: witness.host_ptr,
                len: packed.size,
            }];
            let stated = stated_task_resource_generation(state, tex, resource);
            let allocation_gva = allocation.as_ref()?.gva;
            audit_packed_view_on_reuse(state, host, task_id, allocation_gva, witness.gpas);
            audit_import_bytes_against_page_table(
                state,
                host,
                task_id,
                allocation_gva,
                witness.host_ptr,
                packed.size,
            );
            let seen = crate::runtime::gather_witness::note_gather(
                state,
                crate::runtime::gather_witness::GatherRail::Linear,
                crate::runtime::gather_witness::GatherKey::TaskGva {
                    task_id,
                    resource,
                    gva: allocation_gva,
                },
                stated,
                crate::runtime::gather_witness::GatherWindow {
                    gpas: witness.gpas,
                    runs: &witness_runs,
                    span: packed.size,
                    page_size: page as usize,
                },
            );
            let memory =
                linear_guest_image_allocation_memory(&packed, image_allocation, u64::from(bpp))?;
            let transfer = packed.texel_source(0, packed.size, 0)?;
            note_linear_sample_window("packed_contract", allocation_gva, 0, packed.size, 0);
            let request = SampledSourceRequest::GuestImage(
                reims_vgpu_memory::GuestImageSource {
                    direct: direct_admitted.then_some(memory),
                    allocation: image_allocation.clone(),
                    view: *view,
                    transfer,
                },
                sampled_format,
                Some(LinearSampleIdentity::from(seen.identity)),
                seen.vouch,
                native_components,
            );
            return Some((w, h, request));
        }

        let page = state.page_size();
        let witness = packed.witness_window(plane_offset, span)?;
        audit_packed_view_on_reuse(state, host, task_id, gva, witness.gpas);
        audit_import_bytes_against_page_table(state, host, task_id, gva, witness.host_ptr, span);
        let witness_runs = [reims_vgpu_memory::GuestRun {
            host_ptr: witness.host_ptr,
            len: span,
        }];
        let stated = stated_task_resource_generation(state, tex, resource);
        let seen = crate::runtime::gather_witness::note_gather(
            state,
            crate::runtime::gather_witness::GatherRail::Linear,
            crate::runtime::gather_witness::GatherKey::TaskGva {
                task_id,
                resource,
                gva,
            },
            stated,
            crate::runtime::gather_witness::GatherWindow {
                gpas: witness.gpas,
                runs: &witness_runs,
                span,
                page_size: page as usize,
            },
        );
        note_linear_sample_window("packed_plane", gva, plane_offset, span, row_length_texels);
        let source = reims_vgpu_memory::GuestRunSource {
            runs: std::sync::Arc::clone(&packed.runs),
            source_offset: plane_offset,
            total_len: span,
            row_length_texels,
            pages: Some(std::sync::Arc::clone(&packed.pages)),
            physical_pages: reims_vgpu_memory::GuestPageSet::new(witness.gpas),
        };
        return Some((
            w,
            h,
            SampledSourceRequest::GuestRuns(
                source,
                native,
                sampled_format,
                planes,
                Some(LinearSampleIdentity::from(seen.identity)),
                seen.vouch,
                native_components,
            ),
        ));
    }

    // A complete image contract does not depend on host-pointer import. When
    // no stable parent alias exists, retain the same allocation/view shape and
    // expose its full resource as runs; the executor's ordinary gather path
    // materializes every selected mip from that description.
    if let (Some(backing), Some((image_allocation, view))) =
        (allocation.as_ref(), image_contract.as_ref())
    {
        let (gpas, transfer) =
            match task_gva_guest_run_source(state, host, task_id, backing.gva, backing.size) {
                Ok(window) => window,
                Err(refusal) => {
                    crate::runtime::drain::note_store_route(match refusal {
                        WindowRefusal::NoAlias => "zc_lin_no_alias",
                        WindowRefusal::SpanUnmapped => "zc_lin_span_unmapped",
                        WindowRefusal::Untileable => "zc_lin_untileable",
                    });
                    return None;
                }
            };
        let page = state.page_size() as usize;
        let stated = stated_task_resource_generation(state, tex, resource);
        let seen = crate::runtime::gather_witness::note_gather(
            state,
            crate::runtime::gather_witness::GatherRail::Linear,
            crate::runtime::gather_witness::GatherKey::TaskGva {
                task_id,
                resource,
                gva: backing.gva,
            },
            stated,
            crate::runtime::gather_witness::GatherWindow {
                gpas: &gpas,
                runs: &transfer.runs,
                span: backing.size,
                page_size: page,
            },
        );
        note_linear_sample_window("gva_walk", backing.gva, 0, backing.size, 0);
        return Some((
            w,
            h,
            SampledSourceRequest::GuestImage(
                reims_vgpu_memory::GuestImageSource {
                    direct: None,
                    allocation: image_allocation.clone(),
                    view: *view,
                    transfer,
                },
                sampled_format,
                Some(LinearSampleIdentity::from(seen.identity)),
                seen.vouch,
                native_components,
            ),
        ));
    }

    // The copy-backed fallback still covers exactly the bound level window.
    let (gpas, runs) = match task_gva_guest_run_window(state, host, task_id, gva, span) {
        Ok(window) => window,
        Err(refusal) => {
            crate::runtime::drain::note_store_route(match refusal {
                WindowRefusal::NoAlias => "zc_lin_no_alias",
                WindowRefusal::SpanUnmapped => "zc_lin_span_unmapped",
                WindowRefusal::Untileable => "zc_lin_untileable",
            });
            return None;
        }
    };
    let page = state.page_size() as usize;
    let stated = stated_task_resource_generation(state, tex, resource);
    let seen = crate::runtime::gather_witness::note_gather(
        state,
        crate::runtime::gather_witness::GatherRail::Linear,
        crate::runtime::gather_witness::GatherKey::TaskGva {
            task_id,
            resource,
            gva,
        },
        stated,
        crate::runtime::gather_witness::GatherWindow {
            gpas: &gpas,
            runs: &runs,
            span,
            page_size: page,
        },
    );
    let physical_pages = reims_vgpu_memory::GuestPageSet::new(&gpas);
    let source = reims_vgpu_memory::GuestRunSource {
        runs: std::sync::Arc::new(runs),
        source_offset: 0,
        total_len: span,
        row_length_texels,
        pages: guest_page_window(host, gpas, page as u64, gva % page as u64, span),
        physical_pages,
    };
    Some((
        w,
        h,
        SampledSourceRequest::GuestRuns(
            source,
            native,
            sampled_format,
            planes,
            Some(LinearSampleIdentity::from(seen.identity)),
            seen.vouch,
            native_components,
        ),
    ))
}

/// Zero-copy rail for IOSurface texture mapping-backed sampled binds. Eligible when
/// the mapping's raw bytes sample byte-identically through a native UNORM
/// image (BGRA8/RGBA8 families — the CPU loader's `texel_to_rgba8` is a
/// byte pass-through/swizzle for exactly these) and the caller established
/// the resident is not authoritative. Mirrors `paint_mapping`'s window math
/// (`iosurface_texture_sample_window`) and its flush-on-access rule; any gate miss
/// falls back to the CPU byte path.
pub(super) fn try_iosurface_texture_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mid: u32,
    w: u32,
    h: u32,
    direct_image_2d: bool,
) -> Option<SampledSourceRequest> {
    use crate::runtime::mapping_write::iosurface_texture_sample_window;
    if w == 0 || h == 0 {
        return None;
    }
    let (native, sampled_format, base_off, bpr) = {
        let m = state.surfaces.mappings.get(&mid)?;
        if !m.lifecycle.active || m.pages.entries.is_empty() {
            return None;
        }
        let format = if m.format_or_zero() != 0 {
            m.format_or_zero()
        } else {
            pixel_format::MTL_FORMAT_BGRA8_UNORM
        };
        // This rail binds the mapping's raw bytes with the view mapping it was
        // given, so it carries no format plan. Identity is required rather than
        // assumed: `is_four_byte_color` happens to exclude the one non-identity
        // format (`A8Unorm` is a single byte), but that is a coincidence of
        // widths and not the rule this line depends on.
        let native = match pixel_format::sampled_texel_checked(format) {
            Ok((layout, components)) if layout.is_four_byte_color() => {
                if !pixel_format::swizzle_is_identity(&components) {
                    crate::runtime::drain::note_store_route("zc_iosurface_needs_swizzle");
                    return None;
                }
                layout
            }
            _ => return None,
        };
        let sampled_format = pixel_format::sampled_image_format(format)?;
        let (base_off, bpr_u32, _span_end) = iosurface_texture_sample_window(m, w, h, format)?;
        (native, sampled_format, base_off, bpr_u32 as u64)
    };
    // From the layout the translation chose, as the IOSurface plane view rail does, so the
    // texel size cannot disagree with the image the engine creates. The
    // `is_four_byte_color` gate above already fixes it at four.
    let (span, row_length_texels) =
        strided_window_extent(w, h, native.bytes_per_texel() as u64, bpr)?;
    let plane = MappedSamplePlane {
        mapping_id: mid,
        base_off,
        span,
        row_length_texels,
    };
    let (source, identity, vouch) = witnessed_mapping_sampled_source(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::IOSurface,
        plane,
    )?;
    let memory = direct_image_2d
        .then(|| {
            mapped_guest_image_memory(
                state,
                host,
                plane,
                bpr,
                (w, h),
                u64::from(native.bytes_per_texel()),
            )
        })
        .flatten();
    Some(match memory {
        Some(memory) => SampledSourceRequest::GuestImage(
            reims_vgpu_memory::GuestImageSource::single_mip(
                memory,
                reims_vgpu_memory::GuestImageLayout::D2 {
                    width: w,
                    height: h,
                },
                source,
            )?,
            sampled_format,
            Some(identity),
            vouch,
            pixel_format::swizzle_identity(),
        ),
        None => SampledSourceRequest::GuestRuns(
            source,
            native,
            sampled_format,
            1,
            Some(identity),
            vouch,
            pixel_format::swizzle_identity(),
        ),
    })
}

/// Zero-copy rail for a IOSurface plane view serialized IOSurface plane view — the video
/// hot path. VideoToolbox decodes to NV12 (Y = R8, CbCr = RG8; also
/// BGRA8/RGBA8 surfaces), sampled through the IOSurface plane view view path whose CPU
/// loader (`load_iosurface_plane_view_rgba`) read + uploaded ~1.5 MB per plane per
/// decoded frame (census `t5_view`). This gathers the plane's guest pages
/// directly in the draw CB so the decoded frame never materializes CPU bytes.
/// Mirrors `try_iosurface_texture_sample_zero_copy`'s page coalescing over the plane
/// window from `iosurface_plane_view_sample_window` (which carries the wire plane index +
/// biplanar offset); any gate miss falls back to the CPU byte path.
pub(super) fn try_iosurface_plane_view_sample_zero_copy<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    mid: u32,
    view: objects::IOSurfacePlaneViewDescriptor,
    direct_image_2d: bool,
) -> Option<SampledSourceRequest> {
    use crate::runtime::mapping_write::iosurface_plane_view_sample_window;
    let (w, h) = (view.width, view.height);
    if w == 0 || h == 0 || view.depth != 1 {
        return None;
    }
    // Match the CPU path's resolution before reading the plane pages.
    if !mapper::ensure_resolved_for_scanout(state, host, mid) {
        return None;
    }
    let (native, sampled_format, bpp, base_off, bpr) = {
        let m = state.surfaces.mappings.get(&mid)?;
        if !m.lifecycle.active || m.pages.entries.is_empty() {
            return None;
        }
        // Native formats whose guest bytes sample byte-identically through the
        // matching Vulkan image (the CPU loader's `texel_to_rgba8` is a
        // pass-through/swizzle for exactly these); everything else stays CPU.
        // The texel size comes from the layout the translation chose, so it can
        // never disagree with the image the engine creates.
        // A multiplanar view's planes are the video luma/chroma formats, all of
        // which sit identically on their Vulkan spellings. Required rather than
        // assumed, for the reason the IOSurface texture rail states.
        let (native, bpp) = match pixel_format::sampled_texel_checked(view.pixel_format) {
            Ok((layout, components)) => {
                if !pixel_format::swizzle_is_identity(&components) {
                    crate::runtime::drain::note_store_route("zc_t5_needs_swizzle");
                    return None;
                }
                (layout, layout.bytes_per_texel())
            }
            Err(_) => return None,
        };
        let (base_off, bpr_u32, _span_end) =
            iosurface_plane_view_sample_window(m, view.plane_index, w, h, view.pixel_format)?;
        let sampled_format = pixel_format::sampled_image_format(view.pixel_format)?;
        (native, sampled_format, bpp, base_off, bpr_u32 as u64)
    };
    let (span, row_length_texels) = strided_window_extent(w, h, bpp as u64, bpr)?;
    let plane = MappedSamplePlane {
        mapping_id: mid,
        base_off,
        span,
        row_length_texels,
    };
    let (source, identity, vouch) = witnessed_mapping_sampled_source(
        state,
        host,
        crate::runtime::gather_witness::GatherRail::IOSurfacePlaneView,
        plane,
    )?;
    let memory = direct_image_2d
        .then(|| {
            mapped_guest_image_memory(
                state,
                host,
                plane,
                bpr,
                (w, h),
                u64::from(native.bytes_per_texel()),
            )
        })
        .flatten();
    Some(match memory {
        Some(memory) => SampledSourceRequest::GuestImage(
            reims_vgpu_memory::GuestImageSource::single_mip(
                memory,
                reims_vgpu_memory::GuestImageLayout::D2 {
                    width: w,
                    height: h,
                },
                source,
            )?,
            sampled_format,
            Some(identity),
            vouch,
            pixel_format::swizzle_identity(),
        ),
        None => SampledSourceRequest::GuestRuns(
            source,
            native,
            sampled_format,
            1,
            Some(identity),
            vouch,
            pixel_format::swizzle_identity(),
        ),
    })
}

/// Serve a guest-CPU-produced linear texture (tight OR padded row stride)
/// through the byte-exact revalidated memo. Every call re-reads the native
/// guest rows (a guest write is always observed); only the swizzle/gather +
/// allocation — and, via the returned generation identity, the engine's
/// content hash + upload — are skipped when the bytes are unchanged. Returns
/// the upload byte format (native BGRA8 when eligible, else RGBA8). Measured
/// on Safari fast-scroll: the padded-stride glyph/tile atlases re-present only
/// ~59 distinct gva keys with ~99% recurrence (`fallback_gva_churn`), so this
/// memo now serves that former `lin_guest_fb` hot path instead of a per-bind
/// re-read+re-upload. Returns `None` (no logging: a fast-path miss, not a
/// failure) only for sub-tight strides or formats `convert_row_to_rgba8`
/// cannot decode, which fall through to the general loader.
/// Convert the raw native rows read for a guest-linear texture (row stride
/// `bpr`, `tight` = the packed row byte count) into the tight upload buffer.
/// A straight upload — RGBA8, BGRA8 kept native, or half-float colour kept at
/// its own width — gathers each row with a plain copy (padding skipped, no
/// swizzle) and reports its native format; every other format converts to
/// RGBA8 per row. Shared by the guest-linear memo's miss-fill so its padded and
/// tight branches agree byte-for-byte with the direct loader.
///
/// The straight-upload output is sized from the chosen layout's own texel
/// width. It was sized from `RGBA8_BPP` while only four-byte layouts could be
/// chosen, and the half-float arms are eight and four bytes a texel — so a
/// hard-coded four here would under-allocate an `RGBA16Float` image by half and
/// the row copy would refuse rather than write past it, which is a lost bind
/// dressed as a decline.
struct NativeScratchUpload<'a> {
    scratch: &'a [u8],
    w: u32,
    h: u32,
    planes: u32,
    bpr: u64,
    sample_fmt: u16,
    tight: u64,
    executor: &'a dyn crate::runtime::executor::Executor,
}

fn native_scratch_to_upload(request: NativeScratchUpload<'_>) -> Option<(Vec<u8>, TexelLayout)> {
    let NativeScratchUpload {
        scratch,
        w,
        h,
        planes,
        bpr,
        sample_fmt,
        tight,
        executor,
    } = request;
    let native = native_uploads_for(sample_fmt, executor);
    let bpr = bpr as usize;
    if let Some(fmt) = linear_native_upload_format(sample_fmt, native)
        .filter(|fmt| tight == (w as u64).saturating_mul(fmt.bytes_per_texel() as u64))
    {
        let row_bytes = tight as usize;
        let rows = (h as usize).checked_mul(planes as usize)?;
        let mut out = vec![0u8; row_bytes.checked_mul(rows)?];
        for row_index in 0..rows {
            let src = row_index.checked_mul(bpr)?;
            let dst = row_index * row_bytes;
            out.get_mut(dst..dst + row_bytes)?
                .copy_from_slice(scratch.get(src..src + row_bytes)?);
        }
        return Some((out, fmt));
    }
    let out_row = (w as usize).checked_mul(RGBA8_BPP as usize)?;
    let rows = (h as usize).checked_mul(planes as usize)?;
    let out_len = out_row.checked_mul(rows)?;
    let trow = tight as usize;
    let mut out = vec![0u8; out_len];
    // This rung carries nearly all of the pathway's sampled traffic, so a
    // narrowing taken here is the one most likely to be the narrowing that
    // matters — and until this line existed the rung reported none at all while
    // the general loader reported its own. Same key as the others, so a format
    // narrowed on both rails is two lines and not one.
    crate::runtime::draw::note_sampled_narrowing("linear_memo_narrowed", 0, sample_fmt, w, h);
    for row_index in 0..rows {
        let src = row_index.checked_mul(bpr)?;
        if !pixel_format::convert_row_to_rgba8(
            sample_fmt,
            scratch.get(src..src + trow)?,
            w,
            &mut out[row_index * out_row..],
        ) {
            return None;
        }
    }
    Some((out, TexelLayout::Rgba8))
}

/// Which native sampled layouts the CPU byte rails may hand the engine for this
/// guest format on this host.
///
/// [`NativeUploads`] is a parameter of the loaders and not a constant because
/// the answer has a capability half that `runtime/draw/texture_view.rs` cannot
/// ask: an image is created at the layout's own `VkFormat`, and a host that
/// cannot linearly filter that format would sample it through a sampler that
/// asks for filtering anyway. This is the one place that asks, so the two
/// halves of the answer are decided together.
///
/// `Bgra8` is unconditional: `B8G8R8A8_UNORM` carries
/// `SAMPLED_IMAGE_FILTER_LINEAR` on every Vulkan implementation by mandate, and
/// the rail that first took it argues the same.
///
/// **Keyed on the format so the common one never asks an irrelevant question.**
/// The host capability is a lock-free device-lifetime snapshot, but this sits
/// on the rung carrying essentially all of the pathway's sampled traffic and
/// only two guest formats can make use of the answer.
/// [`pixel_format::narrows_to_unorm8`] is exactly the set of formats whose CPU
/// arm is lossy, which is exactly the set the half-float flag can change the
/// answer for — so keying on it is the same rule stated once, not a fast path
/// that could disagree with the slow one.
fn native_uploads_for(
    sample_format: u16,
    executor: &dyn crate::runtime::executor::Executor,
) -> NativeUploads {
    if !pixel_format::narrows_to_unorm8(sample_format) {
        return NativeUploads::BGRA8;
    }
    native_uploads_asking_host(executor)
}

/// The same answer for a caller that does not yet know the guest format.
///
/// The last-resort sampled rung resolves the format inside the loader, so it
/// cannot key the question the way the hot rung does — and it does not need to:
/// `settle_linear_texture_sampled` read **0** across a four-rail sweep, because
/// every rung above it serves. A lock on a path that does not run is not a cost.
fn native_uploads_asking_host(executor: &dyn crate::runtime::executor::Executor) -> NativeUploads {
    NativeUploads {
        // One flag for both half-float layouts, so the answer is the
        // conjunction: a host that filters one and not the other keeps neither
        // on the native rail. Nothing on record separates them — both carry
        // `SAMPLED_IMAGE_FILTER_LINEAR` by mandate — and a per-layout flag
        // would be two fields nobody could point at a host that needed them.
        float16: executor.sampled_layout_linear_filter_supported(TexelLayout::Rgba16Float)
            && executor.sampled_layout_linear_filter_supported(TexelLayout::Rg16Float),
        ..NativeUploads::BGRA8
    }
}

/// The sampled linear ladder's hot rung: read the guest's own rows for this
/// texture, reuse the converted copy when the bytes have not changed.
///
/// It carries essentially all of this pathway's sampled traffic — 725 231 of
/// 725 233 loads on a driven boot — so it is where a wrong-content defect would
/// have to live, and it is worth stating plainly that nothing here guesses:
///
/// - The address chain is resolved fresh on every call.
///   `objects::lookup_list_entry` re-reads the object-list entry out of guest
///   memory, `read_descriptor` re-reads the descriptor, and `level_gva` derives
///   the span from that. No step caches, so a recycled `texture_ref` cannot
///   hand this the previous resource's address.
/// - The staleness check is exact, not sampled. The full
///   `bpr * h * depth_planes` native span
///   is re-read every call and compared byte for byte against the memo
///   (`m.native == scratch`), padding included, so a guest write anywhere in
///   the span misses the memo.
/// - The read does not go through a cached host view. `gva_view`'s registered
///   views measured a zero hit rate on this pathway (`view_reuse` = 0 over four
///   boots), so `read_task_gva_by_id` walks the page table here.
///
/// That is measured, not asserted, and it is why the surviving Finder icon
/// class is not a wrong-bytes defect on this rung: the bytes served are the
/// bytes at the address the guest named, checked afresh each time.
#[allow(clippy::too_many_arguments)]
fn load_linear_guest_memoized<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    tex: &TextureDescriptor,
    gva: u64,
    w: u32,
    h: u32,
) -> Option<(
    std::sync::Arc<Vec<u8>>,
    Option<LinearSampleIdentity>,
    SampledByteFormat,
)> {
    let declared_format = tex.declared_pixel_format()?;
    let sample_fmt = effective_view_sample_format(declared_format, None)?;
    let (_, layout) = tex.level_gva(0, state.page_shift)?;
    let bpr = layout.row_stride;
    let planes = layout.planes();
    let tight = pixel_format::tight_row_bytes(w, declared_format)? as u64;
    // Padded strides ride the same memo now — the native read below covers the
    // full `bpr*h*planes` span (padding included, so a write anywhere is
    // observed) and
    // `native_scratch_to_upload` gathers the tight rows. A sub-tight stride
    // (impossible geometry) or a zero dimension declines to the fallback.
    if bpr < tight || w == 0 || h == 0 {
        return None;
    }
    // `bpr*h*planes` and not `TextureLevelLayout::slice_read_span`, which every
    // reader that walks only the tight rows uses instead. This one really does
    // read the last row's padding — that is what makes the memo's byte-for-byte
    // compare able to notice a guest write into it — so it is charged for what
    // it touches.
    //
    // The consequence is a third way to decline, and the one most worth knowing:
    // an image the guest sized to `offset + read_span` exactly is refused here
    // and served by the general loader below, which uses the tighter rule. That
    // is a slower path, not lost work.
    let span = bpr.checked_mul(h as u64)?.checked_mul(u64::from(planes))?;
    let native_len = host_alloc_len(span)?;
    if tex.allocation_size != 0
        && tex
            .base_offset
            .checked_add(layout.offset)
            .and_then(|offset| offset.checked_add(span))
            .is_none_or(|end| end > tex.allocation_size)
    {
        return None;
    }
    // Same coherence rule as the general loader: land any resident-authoritative
    // writeback *aliasing the sampled span* before reading it — and only then.
    //
    // This is the device's largest single wait, 11.5 s across a driven
    // Safari-drag boot, and almost none of it was owed: a writeback lands in one
    // surface's pages while this reader is usually somewhere else entirely. The
    // walk below runs only when something is outstanding, so the binds that
    // dominate this rail — the ones with a clear debt flag — still pay one
    // atomic load.
    //
    // A short walk is `None` and settles. `pages_spanned` is the count the
    // resolver would have produced with nothing dropped, and a dropped page is
    // one this reader cannot rule out.
    // Census, pay, settle — the whole obligation of a CPU read of one named
    // resource's guest bytes. See `writeback_debt::settle_for_texture`.
    crate::runtime::writeback_debt::settle_for_texture(
        state,
        host,
        task_id,
        texture_ref,
        gva,
        span,
        crate::runtime::render_writeback::SettleSite::LinearMemoRead,
    );
    let mut scratch = std::mem::take(&mut state.content.sampled.guest_linear_scratch);
    scratch.resize(native_len, 0);
    let read = gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        gva,
        &mut scratch,
        state.page_shift,
    );
    if read.is_err() {
        state.content.sampled.guest_linear_scratch = scratch;
        return None;
    }
    let key = (task_id, gva, w, h, planes, sample_fmt);
    // Three-way, because "the memo did not answer" has two causes that want
    // opposite fixes and a hit/miss pair cannot tell them apart.
    //
    // This memo cannot skip the guest read — the read is *how* it knows the
    // content is unchanged — so a hit buys exactly one thing: it skips
    // `native_scratch_to_upload`, the pixel-format conversion. Everything else
    // it costs is paid on every bind regardless: the read, the memcmp against
    // `native`, and a cap charged for storing the native bytes *and* their
    // converted copy.
    //
    // So the memo is worth its keep only if `lin_memo_hit` dominates.
    // `lin_memo_changed` is the memcmp running in full and buying nothing — the
    // guest rewrote the plane, which is the case the memo cannot help. And
    // `lin_memo_absent` is the key never repeating, where the cap is holding
    // bytes no bind will ask for again.
    //
    // **Measured, and it earns its keep.** One driven x86 / Vulkan boot (two
    // Safari page loads, scrolls, three title-bar drags): hit 7221, changed 557,
    // absent 179 — a **90.8 %** hit rate, so nine binds in ten skip the format
    // conversion entirely. The three arms sum to 7957, which is exactly
    // `lin_rung_guest_memo` for the same boot; that reconciliation is what says
    // the census is complete rather than merely quiet.
    //
    // This was instrumented to decide whether to delete the memo, on the
    // suspicion it was another cache paying its own miss on every hit. It is
    // not: unlike a walk memo it cannot skip the guest read, but the read was
    // never what it claimed to save. Keep it. The counters stay because the
    // answer is workload-dependent — a guest that rewrites its planes every
    // frame would push `changed` up and invert the conclusion — so this is a
    // ratio worth re-reading, not a settled fact worth deleting.
    let hit = match state.content.sampled.guest_linear_memo.get_touch(&key) {
        None => {
            crate::runtime::drain::note_store_route("lin_memo_absent");
            None
        }
        // Vec equality is length + byte memcmp with early exit on change.
        Some(m) if m.native == scratch => {
            crate::runtime::drain::note_store_route("lin_memo_hit");
            Some((m.rgba.clone(), m.generation, m.layout))
        }
        Some(_) => {
            crate::runtime::drain::note_store_route("lin_memo_changed");
            None
        }
    };
    if let Some((rgba, generation, fmt)) = hit {
        state.content.sampled.guest_linear_scratch = scratch;
        return Some((
            rgba,
            Some(LinearSampleIdentity {
                key: gva,
                generation,
            }),
            // The memo stores the layout it converted to; the transfer function
            // is `sample_fmt`'s and is re-derived on hit and miss alike, so a
            // retained entry cannot carry a stale one.
            SampledByteFormat::from_source(fmt, sample_fmt),
        ));
    }
    // First sight or native bytes changed: convert fresh, new generation.
    let Some((rgba, fmt)) = native_scratch_to_upload(NativeScratchUpload {
        scratch: &scratch,
        w,
        h,
        planes,
        bpr,
        sample_fmt,
        tight,
        executor: state.executor.as_ref(),
    }) else {
        state.content.sampled.guest_linear_scratch = scratch;
        return None;
    };
    let generation = state.next_sampled_content_generation();
    let rgba = std::sync::Arc::new(rgba);
    let entry_bytes = scratch.len() + rgba.len();
    state.content.sampled.guest_linear_memo.insert(
        key,
        crate::model::GuestLinearMemo {
            native: scratch,
            rgba: rgba.clone(),
            layout: fmt,
            generation,
        },
        entry_bytes,
    );
    Some((
        rgba,
        Some(LinearSampleIdentity {
            key: gva,
            generation,
        }),
        SampledByteFormat::from_source(fmt, sample_fmt),
    ))
}

/// Report a sampled texture served entirely as zeroes out of the guest's pages
/// while the host cache holds an entry for the same address.
///
/// A type-2/3 texture's guest GVA pages are a *pageable alias* of a body this
/// device owns (`surface_cache::store_linear_texture`), so a blank read here
/// could mean the device rendered the span, cached it, and its own writeback
/// never landed in the guest's pages — a silent loss, since the draw then
/// succeeds and paints a blank cell with nothing declining.
///
/// Three questions separate that defect from its lookalikes, and this function
/// asks all three:
///
/// - **Does the cache hold this span at all?** `lin_rung_host_entry` against
///   `lin_rung_guest_memo` is the denominator. Without it a bare count cannot
///   tell "300 of 300" from "300 of 300 000".
/// - **Are the zeroed pages still the pages the entry was produced over?**
///   [`crate::runtime::surface_cache::gva_backing_state`]: `Same` means the
///   cache entry is live over these pages, `Moved`/`Unmapped` means the guest
///   handed the address to another allocation and the *cache* is the stale
///   side — where serving it would be the corruption, not the repair.
/// - **Does the entry hold any pixels?** The question the class was named for
///   and never asked. A span the device CLEARed and cached blank reads blank
///   off blank pages with nothing lost, and `draw_partial_clear` runs in the
///   thousands a boot.
///
/// ## Measured, and the class is not a loss
///
/// One driven x86/Vulkan boot — 30 s Safari window drag plus two web-content
/// probe runs, all declared regions measuring their colour — summed over its
/// `store_routes` windows:
///
/// ```text
/// lin_rung_guest_memo             79898   sampled serves off the guest's pages
/// lin_rung_host_entry             18988   …of which the cache also held the span (23.8 %)
/// lin_rung_guest_blank             1859   …that came back all zeroes (2.3 %)
/// lin_rung_blank_with_host_entry     22   …of those, with a host entry (1.2 % of blanks)
/// lin_rung_blank_host_agrees         22   …where the cache is blank too: nothing lost
/// lin_rung_blank_host_content         0   …where the cache holds pixels: the defect
///
/// 13 distinct spans, backing=Same and fmt=Bgra8 on every one
/// ```
///
/// A second driven boot on the same workload read 28 / 28 / 0 and a third
/// 32 / 32 / 0 — the same partition three times, so the zero is not one boot's
/// luck. The identity to check is `with_host_entry == host_agrees +
/// host_content`; it is what catches a miscount before the zero is believed.
///
/// So the two rails agree on every occurrence. The dominant blank class is
/// elsewhere — 98.8 % of blank samples have no cache entry for the span at all,
/// which is "we do not have the pixels", not "we lost them". `fmt=Bgra8`
/// throughout also excludes a conversion artifact: the blank test runs on
/// converted RGBA, and a layout whose conversion zeroed the buffer would show
/// up as a different `fmt`.
///
/// `lin_rung_blank_host_content` is therefore a healthy zero, and a non-zero
/// reading is the alarm: it is the only arm that means guest work was lost, and
/// the place to repair it is the GVA writeback rail upstream of this rung.
///
/// ## What this does not license
///
/// Serving the cache on the whole rung — making the order match
/// [`crate::runtime::draw::seed_color_load`]'s stated rule, "exact target
/// GVA is the strongest identity … Guest memory is last" — would change ~19 000
/// serves to repair nothing. The two rails are not the same case: the seed's
/// entry is for an attachment the pass is about to draw *onto*, while a sampled
/// span may be guest-CPU-produced between the encode and the sample with
/// nothing here able to witness it. Serving the cache only when the sample came
/// back blank is not available either — that is selecting on content.
///
/// The `fail` line is behind `first_sight` on `(gva, w, h)`, so it fires once
/// per distinct span for the life of the boot while the counters beside it are
/// per-occurrence; the two are not comparable. The `gva_backing_state` walk
/// sits under that latch, so it is one page-table walk per distinct span rather
/// than per sample.
///
/// `span` is `(gva, width, height)` — the GVA cache's key, taken as one value
/// because every lookup below needs all three and none of them means anything
/// apart.
pub(super) fn note_guest_rung_blank<H: HostMemory>(
    state: &Device,
    host: &H,
    task_id: u32,
    texture_ref: u32,
    span: (u64, u32, u32),
    rgba: &[u8],
    byte_format: SampledByteFormat,
) {
    let (gva, w, h) = span;
    crate::runtime::drain::note_store_route("lin_rung_guest_memo");
    // The denominator for the loss below: every serve off the guest's pages for
    // a span the cache also holds, whatever came back. Taken before the blank
    // test so the blank ones are a subset of a population, not a bare count.
    //
    // The bytes, not just the presence: a blank guest read only means pixels
    // were lost if the cache holds pixels to lose. `has_gva` cannot tell the two
    // apart, so it counted "we cached a blank frame" as loss.
    let host_bytes = crate::runtime::surface_cache::get_gva(state, gva, w, h);
    let host_entry = host_bytes.is_some();
    if host_entry {
        crate::runtime::drain::note_store_route("lin_rung_host_entry");
    }
    if rgba.is_empty() || rgba.iter().any(|&b| b != 0) {
        return;
    }
    crate::runtime::drain::note_store_route("lin_rung_guest_blank");
    // Identity, latched per span, because the count alone cannot say whether a
    // blank sample is a transparent layer doing its job or the icon cell that
    // came out empty. 99.5 % of loads on this rung return content, so the
    // population that matters is small enough to name each member of, and the
    // geometry is what joins one of these to something on screen.
    if crate::observe::first_sight("lin_rung_guest_blank", gva ^ ((w as u64) << 32) ^ h as u64) {
        crate::observe::off(format!(
            "lin_rung_guest_blank task={task_id} ref={texture_ref} gva={gva:#x} {w}x{h}"
        ));
    }
    let Some(host_bytes) = host_bytes else {
        return;
    };
    crate::runtime::drain::note_store_route("lin_rung_blank_with_host_entry");
    // Which of the two cases this span is. A cache entry that is itself all
    // zeroes agrees with the guest's pages, so nothing was lost and there is
    // nothing upstream to repair; only a cache entry holding content while the
    // guest alias reads zero is a coherence loss.
    let host_blank = host_bytes.iter().all(|&b| b == 0);
    crate::runtime::drain::note_store_route(if host_blank {
        "lin_rung_blank_host_agrees"
    } else {
        "lin_rung_blank_host_content"
    });
    if crate::observe::first_sight(
        "lin_rung_blank_with_host_entry",
        gva ^ ((w as u64) << 32) ^ h as u64,
    ) {
        // Under `first_sight`, so this walk is once per distinct blank span for
        // the life of the boot and not once per sample.
        let backing = crate::runtime::surface_cache::gva_backing_state(state, host, gva);
        crate::observe::fail(format!(
            "lin_rung_blank_with_host_entry task={task_id} ref={texture_ref} \
             gva={gva:#x} {w}x{h} bytes={} fmt={byte_format:?} host_blank={} \
             backing={backing:?} (guest alias is zero and the host cache has this span; \
             host_blank=true means the cache agrees and nothing was lost, false means \
             the cache holds content this read did not return; backing=Same means the \
             cache entry is still over these pages, Moved/Unmapped means the address \
             was handed on and the cache entry is the stale one)",
            rgba.len(),
            u8::from(host_blank)
        ));
    }
}

pub(super) fn load_linear_from_host_caches<M: HostMemory + HostOps>(
    state: &mut Device,
    host: &mut M,
    task_id: u32,
    texture_ref: u32,
    tex: &TextureDescriptor,
) -> Option<LoadedLinearSample> {
    // The descriptor is resolved+decoded once by the caller
    // (`resolve_sampled_source`'s linear branch) and threaded in; the zero-copy
    // attempt above shares the same decode.
    let (gva, layout) = tex.level_gva(0, state.page_shift)?;
    let w = layout.width;
    let h = layout.height;
    if w == 0 || h == 0 {
        return None;
    }
    // No address- or content-derived cache decides this resource's bytes. A
    // named current resident is handled above; this fallback reads the guest's
    // declared linear resource and memoizes only the derived conversion.
    // Guest-CPU-produced linear textures (wallpaper, glyph atlases) have no
    // host producer generation. Re-read the native rows and byte-compare
    // against the memo: unchanged content reuses the retained swizzled Arc
    // and carries a generation identity so the engine skips hash+memcmp too.
    if let Some((rgba, identity, byte_format)) =
        load_linear_guest_memoized(state, host, task_id, texture_ref, tex, gva, w, h)
    {
        note_guest_rung_blank(
            state,
            host,
            task_id,
            texture_ref,
            (gva, w, h),
            &rgba,
            byte_format,
        );
        // The other half of a channel-order reading. A GVA span is *written* by
        // the render Store at the attachment's declared format and *read* here
        // at the sampled descriptor's, and on the copying rail those are two
        // interpretations of one buffer rather than one typed image — so the
        // pair has to be joinable. `gva_flush_gpu_declined` names the write's
        // format against the same `gva=`; this names the read's. Latched per
        // (gva, format) so a steady bind stays at one line and a *change* of
        // interpretation still surfaces.
        let declared_format = tex.declared_pixel_format()?;
        if crate::observe::first_sight("lin_serve_fmt", gva ^ (u64::from(declared_format) << 48)) {
            crate::observe::off(format!(
                "lin_serve_fmt task={task_id} ref={texture_ref} gva={gva:#x} {w}x{h} \
                 fmt={:#x} bytes={byte_format:?}",
                declared_format
            ));
        }
        return Some((w, h, rgba, identity, byte_format));
    }
    // There is deliberately no second guest rung under the memo. One used to
    // re-read through `load_linear_texture_native_host` when the memo declined
    // and never ran: every `None` the memo returns is a decode, geometry,
    // bounds or guest-read failure that the re-read meets on the same
    // descriptor and the same pages.
    //
    // This counter takes its place because `load_linear_guest_memoized` emits
    // on none of its refusal paths, so the deleted rung's decline was the only
    // line that ever named one. A sample refused here falls to
    // `load_sampled_rgba_static` and then to the caller's typed
    // `TextureResolveMissing` — visible, but without the memo's own reason.
    // `lin_rung_memo_declined` against `lin_rung_guest_memo` says whether that
    // gap is worth closing; while it reads zero there is nothing to name.
    crate::runtime::drain::note_store_route("lin_rung_memo_declined");
    None
}
