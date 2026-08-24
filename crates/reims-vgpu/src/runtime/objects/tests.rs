use reims_vgpu_wire::device_desc::{IOSurfacePlaneViewBuilder, SurfaceBackingBuilder};

use super::*;
use crate::model::{DeviceId, PAGE_SHIFT_ARM64E, PAGE_SHIFT_X86};
use crate::runtime::decode::resource::SERIALIZER_RESOURCE_OBJECT_SAMPLER;
use crate::runtime::host::FakeHost;
use reims_vgpu_core::endian::{ld32, st16, st32, st64};
use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};
use reims_vgpu_protocol::DEVICE_DESC_PLANE_COUNT;

fn mapper_texture_descriptor(
    mapper_ref: u64,
    object_ref: u32,
    pixel_format: u16,
    width: u32,
    height: u32,
) -> [u8; 56] {
    let mut desc = [0u8; 56];
    st64(&mut desc[0..], mapper_ref);
    st32(
        &mut desc[8..],
        reims_vgpu_wire::ops::backed_texture::OPCODE_IOSURFACE_TEXTURE,
    );
    st32(
        &mut desc[12..],
        reims_vgpu_wire::ops::backed_texture::IOSURFACE_TEXTURE_TOTAL_LEN,
    );
    st32(&mut desc[16..], object_ref);
    st32(&mut desc[20..], u32::from(pixel_format) << 16 | 2);
    st32(&mut desc[24..], width);
    st32(&mut desc[28..], height);
    st32(&mut desc[32..], 1);
    st16(&mut desc[36..], 1);
    st16(&mut desc[38..], 1);
    st16(&mut desc[40..], 1);
    desc
}

#[test]
fn iosurface_texture_fail_latch_dedups_per_task_ref_and_rearms_on_clear() {
    // Flood guard for the per-draw-per-ref resolve path: a genuinely-broken
    // IOSurface texture ref logs each reason once, isolates per (task,ref), and
    // re-arms on resolve. Unique ids so this never races real refs across
    // the process-global latch.
    let (t, r, r2) = (0xAB01u32, 0xCD01u32, 0xCD02u32);
    clear_iosurface_texture_fail(t, r);
    clear_iosurface_texture_fail(t, r2);
    let seen = |task: u32, rf: u32, reason: &'static str| {
        iosurface_texture_fail_latch()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&(task, rf, reason))
    };
    note_iosurface_texture_fail(t, r, "iosurface_texture_register", "x".into());
    assert!(seen(t, r, "iosurface_texture_register"));
    // Distinct reason on the same ref tracked independently.
    note_iosurface_texture_fail(t, r, "iosurface_texture_desc_read", "x".into());
    assert!(seen(t, r, "iosurface_texture_desc_read"));
    // A different ref is untouched.
    assert!(!seen(t, r2, "iosurface_texture_register"));
    note_iosurface_texture_fail(t, r2, "iosurface_texture_register", "x".into());
    // Clearing r re-arms only r, leaves r2.
    clear_iosurface_texture_fail(t, r);
    assert!(!seen(t, r, "iosurface_texture_register"));
    assert!(!seen(t, r, "iosurface_texture_desc_read"));
    assert!(seen(t, r2, "iosurface_texture_register"));
    clear_iosurface_texture_fail(t, r2);
}

fn setup_task_with_list(host: &mut FakeHost, state: &mut Device) {
    assert!(state.map_mapper_surface(
        reims_vgpu_protocol::MapperSurfaceRef::new(9),
        reims_vgpu_protocol::MapperResolvedSurfaceId::new(9)
    ));
    assert!(state.map_mapper_surface(
        reims_vgpu_protocol::MapperSurfaceRef::new(10),
        reims_vgpu_protocol::MapperResolvedSurfaceId::new(10)
    ));
    // Same 1-level map as gva_mem test: GVA page 0 → data pfn 4.
    let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
    let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x4000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);

    state.define_task(1, 0x1000, 2);
    // list base GVA 0 (pfn field 0 allowed)
    assert!(state.set_object_list(1, 0, 8));
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 11u32 | (56u32 << 8));
    entry[4..12].copy_from_slice(&0x40u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 12, &entry);
    let desc = mapper_texture_descriptor(9, 1, 0x50, 64, 32);
    let _ = host.write_gpa(data_gpa + 0x40, &desc);
}

#[test]
fn resolve_iosurface_texture_from_list() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    // Sanity: list entry readable
    let e = lookup_list_entry(&state, &host, 1, 1).expect("list entry");
    assert_eq!(e.kind, ObjectKind::IOSurfaceTexture);
    assert_eq!(e.descriptor_gva, 0x40);
    let mid = resolve_iosurface_texture_ref(&mut state, &host, 1, 1).expect("iosurface_texture");
    assert_eq!(mid, 9);
    let m = state.surfaces.mappings.get(&9).unwrap();
    assert!(m.has_geometry());
    assert_eq!(
        m.geometry().map(|g| (g.width, g.height, g.format)),
        Some((64, 32, 0x50))
    );
}

/// Registering an IOSurface texture is construction, not bind-time repair.
///
/// Once the task owns the texture object, later binds retrieve that object and
/// must not replay its serialized descriptor over mutable mapping state. A new
/// descriptor can take effect only after the resource lifetime ends.
#[test]
fn a_retained_iosurface_texture_runs_construction_side_effects_once() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let resource = resolve_resource(&state, &host, 1, 1).expect("construction");

    assert_eq!(
        resolve_iosurface_texture_resource(&mut state, 1, 1, &resource),
        Some(9)
    );
    {
        let mapping = state
            .surfaces
            .mappings
            .get_mut(&9)
            .expect("registered mapping");
        mapping.publish_geometry_for_test(17, 19, 0x71);
    }

    assert_eq!(
        resolve_iosurface_texture_resource(&mut state, 1, 1, &resource),
        Some(9)
    );
    let mapping = &state.surfaces.mappings[&9];
    assert_eq!(
        mapping
            .geometry()
            .map(|g| (g.width, g.height, g.format))
            .expect("geometry"),
        (17, 19, 0x71),
        "a warm bind must not replay immutable construction input"
    );
}

/// Physical replacement is the event that re-arms backing resolution for a
/// retained texture. A warm bind accepts the already-latched page plan; once
/// invalidation clears that plan, the same bind must enter the resolver again
/// rather than treating object retention as proof that old pages remain live.
#[test]
fn a_texture_bind_reuses_backing_until_physical_invalidation() {
    let host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    assert!(state.map_surface(9));
    {
        let mapping = state
            .surfaces
            .mappings
            .get_mut(&9)
            .expect("surface mapping");
        mapping.lifecycle.active = true;
        mapping.publish_geometry_for_test(64, 32, 0);
        mapping.pages.entries = vec![0x1234_5001];
    }

    assert!(ensure_surface_for_texture_bind(&mut state, &host, 9));
    assert!(state.invalidate_mapping_pages(9).had_page_state);
    assert!(
        !ensure_surface_for_texture_bind(&mut state, &host, 9),
        "the invalidated page plan must be rebuilt before the texture binds"
    );
}

/// A list entry and descriptor are construction input for a resource, not
/// mutable bind-time state.
///
/// Moving the task's object list changes where future resources are
/// constructed from; it does not retarget an object that is already live. An
/// explicit delete ends that lifetime, and reusing the reference constructs a
/// new object from the then-current descriptor.
#[test]
fn resources_keep_construction_input_until_explicit_delete() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;

    let first = resolve_resource(&state, &host, 1, 1).expect("first construction");
    assert_eq!(ld32(first.descriptor()), 9);

    // Rewrite the descriptor and move the list somewhere unreadable. Neither
    // operation changes the already-registered object.
    let _ = host.write_gpa(data_gpa + 0x40, &10u32.to_le_bytes());
    assert!(state.set_object_list(1, 0xdead, 8));
    let retained = resolve_resource(&state, &host, 1, 1).expect("registered object");
    assert!(Arc::ptr_eq(&first, &retained));
    assert_eq!(ld32(retained.descriptor()), 9);
    let Ok(crate::runtime::decode::resource::Descriptor::MapperIOSurfaceTextureView(view)) =
        decoded_resource(&retained)
    else {
        panic!("retained descriptor lost its semantic IOSurface texture shape");
    };
    assert_eq!(view.mapper_surface.get(), 9);
    assert_eq!((view.declaration.width, view.declaration.height), (64, 32));
    assert_eq!(
        resolve_iosurface_texture_resource(&mut state, 1, 1, &retained),
        Some(9),
        "the retained typed object resolves after its construction bytes become unreadable"
    );

    // Delete and reuse is the lifecycle edge that permits the same reference
    // to name a newly-constructed resource.
    assert!(state.delete_object(1, 1));
    assert!(state.set_object_list(1, 0, 8));
    let replacement = resolve_resource(&state, &host, 1, 1).expect("replacement construction");
    assert!(!Arc::ptr_eq(&first, &replacement));
    assert_eq!(ld32(replacement.descriptor()), 10);
    assert_eq!(
        resolve_iosurface_texture_resource(&mut state, 1, 1, &replacement),
        Some(10),
        "the replacement lifetime runs its own construction side effects"
    );
}

#[test]
fn a_retained_view_retries_its_parent_relation_after_the_parent_appears() {
    let host = FakeHost::new();
    let state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let descriptor = IOSurfacePlaneViewBuilder::new(
        7,
        0,
        2,
        reims_vgpu_wire::device_desc::IOSURFACE_PLANE_VIEW_RECORD_TAG_COLOR_VIEW,
    );
    let view = state.task_objects.resources.register(
        1,
        2,
        Arc::new(TaskResource::new(
            ListObjectEntry::new(ObjectKind::IOSurfacePlaneView, 0, 0),
            Arc::from(descriptor.bytes()),
        )),
    );
    let view_id = view.semantic_id().unwrap();

    ensure_resource_relations(&state, &host, 1, 2, &view);
    assert_eq!(
        state
            .task_objects
            .resources
            .resource_node(view_id)
            .unwrap()
            .storage,
        None
    );

    let surface = state.task_objects.resources.register(
        0,
        7,
        Arc::new(TaskResource::new(
            ListObjectEntry::new(ObjectKind::SurfaceBacking, 0, 0),
            Arc::from([]),
        )),
    );
    ensure_resource_relations(&state, &host, 1, 2, &view);

    let surface_node = state
        .task_objects
        .resources
        .resource_node(surface.semantic_id().unwrap())
        .unwrap();
    let view_node = state.task_objects.resources.resource_node(view_id).unwrap();
    assert_eq!(view_node.storage, surface_node.storage);
    assert!(view_node.parents.contains(&surface_node.id));
}

fn heap_texture_descriptor(
    object_ref: u32,
    heap_ref: u32,
    use_offset: bool,
    offset: u64,
) -> [u8; 60] {
    use reims_vgpu_wire::ops::heap_texture as heap;
    let mut bytes = [0u8; heap::NEW_HEAP_TEXTURE_TOTAL_LEN as usize];
    st32(&mut bytes[0..], heap::OPCODE_NEW_HEAP_TEXTURE);
    st32(&mut bytes[4..], heap::NEW_HEAP_TEXTURE_TOTAL_LEN);
    st32(&mut bytes[8..], object_ref);
    st32(&mut bytes[12..], heap_ref);
    // 2D, GPU-optimized, shaderRead|shaderWrite, RGBA32Float (125).
    st32(&mut bytes[16..], 0x007d_0342);
    st32(&mut bytes[20..], 180);
    st32(&mut bytes[24..], 135);
    st32(&mut bytes[28..], 1);
    st16(&mut bytes[32..], 1);
    st16(&mut bytes[34..], 1);
    st16(&mut bytes[36..], 1);
    st16(&mut bytes[38..], 0x20);
    bytes[48] = u8::from(use_offset);
    st64(&mut bytes[52..], offset);
    bytes
}

#[test]
fn heap_texture_relations_publish_the_generational_heap_and_exact_alias() {
    let host = FakeHost::new();
    let state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let first = state.task_objects.resources.register(
        1,
        2,
        Arc::new(TaskResource::new(
            ListObjectEntry::new(ObjectKind::TextureView, 0, 0),
            Arc::from(heap_texture_descriptor(2, 7, true, 0)),
        )),
    );
    let second = state.task_objects.resources.register(
        1,
        3,
        Arc::new(TaskResource::new(
            ListObjectEntry::new(ObjectKind::TextureView, 0, 0),
            Arc::from(heap_texture_descriptor(3, 7, true, 0)),
        )),
    );

    ensure_resource_relations(&state, &host, 1, 2, &first);
    ensure_resource_relations(&state, &host, 1, 3, &second);

    let first_node = state
        .task_objects
        .resources
        .resource_node(first.semantic_id().unwrap())
        .unwrap();
    let second_node = state
        .task_objects
        .resources
        .resource_node(second.semantic_id().unwrap())
        .unwrap();
    assert_eq!(first_node.storage, second_node.storage);
    let storage = state
        .task_objects
        .resources
        .storage_node(first_node.storage.expect("heap storage"))
        .unwrap();
    let reims_vgpu_core::StorageBacking::HeapPlacement {
        heap,
        offset,
        length,
    } = storage.backing
    else {
        panic!("explicit heap texture did not publish explicit heap storage");
    };
    assert_eq!(
        heap,
        state
            .task_objects
            .heaps
            .identity(1, reims_vgpu_protocol::SerializerRef::new(7))
            .unwrap()
    );
    assert_eq!(offset.get(), 0);
    assert_ne!(length.get(), 0);
    assert!(first_node.content.same_authority(&second_node.content));
}

#[test]
fn task_lifetime_retires_all_of_its_resource_objects() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let resource = resolve_resource(&state, &host, 1, 1).expect("construction");
    assert!(state.task_objects.resources.get(1, 1).is_some());

    assert!(state.delete_task(1).is_some());
    assert!(state.task_objects.resources.get(1, 1).is_none());
    assert_eq!(
        ld32(resource.descriptor()),
        9,
        "an outstanding host owner remains valid"
    );
}

#[test]
fn the_resource_registry_accepts_exactly_the_resource_constructor_kinds() {
    let accepted: Vec<ObjectKind> = [
        ObjectKind::Buffer,
        ObjectKind::Texture,
        ObjectKind::SurfaceBacking,
        ObjectKind::IOSurfacePlaneView,
        ObjectKind::Function,
        ObjectKind::SerializerResource,
        ObjectKind::TextureView,
        ObjectKind::MemorylessTexture,
        ObjectKind::IOSurfaceTexture,
        ObjectKind::DualPlaneTexture,
        ObjectKind::ResourceHandle,
        ObjectKind::HeapBuffer,
        ObjectKind::ExternalBuffer,
    ]
    .into_iter()
    .filter(|&kind| object_kind_is_resource(kind))
    .collect();
    assert_eq!(
        accepted,
        [
            ObjectKind::Buffer,
            ObjectKind::Texture,
            ObjectKind::SurfaceBacking,
            ObjectKind::IOSurfacePlaneView,
            ObjectKind::TextureView,
            ObjectKind::MemorylessTexture,
            ObjectKind::IOSurfaceTexture,
            ObjectKind::DualPlaneTexture,
            ObjectKind::ResourceHandle,
            ObjectKind::HeapBuffer,
            ObjectKind::ExternalBuffer,
        ]
    );
}

/// Serializer state has its own lifetime. A `DeleteResource`-scoped registry
/// must not retain its descriptor and hide a later update.
#[test]
fn non_resource_descriptors_are_read_again() {
    use crate::runtime::decode::resource::OBJECT_TYPE_SERIALIZER_RESOURCE;

    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut entry = [0u8; 12];
    st32(
        &mut entry,
        u32::from(OBJECT_TYPE_SERIALIZER_RESOURCE) | (4u32 << 8),
    );
    entry[4..].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 24, &entry);
    let _ = host.write_gpa(data_gpa + 0x80, &1u32.to_le_bytes());

    let (_, first) = resolve_descriptor(&state, &host, 1, 2, &[ObjectKind::SerializerResource])
        .expect("first serializer descriptor");
    assert_eq!(ld32(&first), 1);
    assert!(state.task_objects.resources.get(1, 2).is_none());

    let _ = host.write_gpa(data_gpa + 0x80, &2u32.to_le_bytes());
    let (_, second) = resolve_descriptor(&state, &host, 1, 2, &[ObjectKind::SerializerResource])
        .expect("updated serializer descriptor");
    assert_eq!(ld32(&second), 2);
    assert!(state.task_objects.resources.get(1, 2).is_none());
}

fn put_sampler_object(host: &mut FakeHost, ref_: u32, descriptor_gva: u64, lod_min: f32) {
    use crate::runtime::decode::resource::OBJECT_TYPE_SERIALIZER_RESOURCE;
    use reims_vgpu_wire::ops::sampler::NEW_SAMPLER_TOTAL_LEN;

    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry,
        u32::from(OBJECT_TYPE_SERIALIZER_RESOURCE) | (NEW_SAMPLER_TOTAL_LEN << 8),
    );
    st64(&mut entry[4..], descriptor_gva);
    let _ = host.write_gpa(
        data_gpa + u64::from(ref_) * OBJECT_LIST_ENTRY_LEN as u64,
        &entry,
    );

    let mut descriptor = vec![0u8; NEW_SAMPLER_TOTAL_LEN as usize];
    st32(&mut descriptor, SERIALIZER_RESOURCE_OBJECT_SAMPLER);
    st32(&mut descriptor[4..], NEW_SAMPLER_TOTAL_LEN);
    st32(&mut descriptor[8..], ref_);
    st32(&mut descriptor[12..], 0x8400_0000);
    st32(&mut descriptor[20..], lod_min.to_bits());
    let _ = host.write_gpa(data_gpa + descriptor_gva, &descriptor);
}

#[test]
fn sampler_construction_is_retained_until_its_own_explicit_delete() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    put_sampler_object(&mut host, 2, 0x80, 1.25);

    let first = resolve_sampler_state(&state, &host, 1, 2).expect("first sampler");
    assert_eq!(first.lod_min_clamp, 1.25);

    // Neither mutable descriptor bytes nor a moved object-list pointer mutate
    // an already-constructed sampler object.
    put_sampler_object(&mut host, 2, 0x80, 7.5);
    assert!(state.set_object_list(1, 0xdead, 8));
    let retained = resolve_sampler_state(&state, &host, 1, 2).expect("retained sampler");
    assert!(Arc::ptr_eq(&first, &retained));
    assert_eq!(retained.lod_min_clamp, 1.25);

    // The sampler API's delete edge, not resource deletion, permits ref reuse.
    assert!(state
        .task_objects
        .samplers
        .delete(1, reims_vgpu_protocol::SerializerRef::new(2)));
    assert!(state.set_object_list(1, 0, 8));
    let replacement = resolve_sampler_state(&state, &host, 1, 2).expect("replacement sampler");
    assert!(!Arc::ptr_eq(&first, &replacement));
    assert_eq!(replacement.lod_min_clamp, 7.5);
}

#[test]
fn failed_sampler_construction_is_not_retained_and_can_retry() {
    use crate::runtime::decode::resource::OBJECT_TYPE_SERIALIZER_RESOURCE;

    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut short_entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut short_entry,
        u32::from(OBJECT_TYPE_SERIALIZER_RESOURCE) | (4 << 8),
    );
    st64(&mut short_entry[4..], 0x80);
    let _ = host.write_gpa(data_gpa + 24, &short_entry);
    let _ = host.write_gpa(
        data_gpa + 0x80,
        &SERIALIZER_RESOURCE_OBJECT_SAMPLER.to_le_bytes(),
    );

    assert!(matches!(
        resolve_sampler_state(&state, &host, 1, 2),
        Err(SamplerResolveError::Decode { .. })
    ));
    assert!(state
        .task_objects
        .samplers
        .get(1, reims_vgpu_protocol::SerializerRef::new(2))
        .is_none());

    put_sampler_object(&mut host, 2, 0x80, 3.0);
    let sampler = resolve_sampler_state(&state, &host, 1, 2).expect("published retry");
    assert_eq!(sampler.lod_min_clamp, 3.0);
}

#[test]
fn task_teardown_retires_sampler_objects_without_touching_outstanding_owners() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    put_sampler_object(&mut host, 2, 0x80, 2.0);
    let sampler = resolve_sampler_state(&state, &host, 1, 2).expect("sampler");

    assert!(state.delete_task(1).is_some());
    assert!(state
        .task_objects
        .samplers
        .get(1, reims_vgpu_protocol::SerializerRef::new(2))
        .is_none());
    assert_eq!(sampler.lod_min_clamp, 2.0);
}

/// The surface backing decoder refuses a descriptor it cannot decode as declared, and
/// says which check refused.
///
/// All three of these bounds are correct — IOSurface caps `getPlaneCount` at
/// eight, a plane record the blob does not reach cannot be decoded, and the
/// device descriptor's `allocSize` really is 32 bits.
///
/// Naming them was the first half. The second is that the first two must not
/// publish a *partial* surface, which is what they did: truncating to the cap
/// handed every later reader a surface that simply has eight planes, and
/// defaulting an unreachable record handed them a 0x0 plane at pitch 0 in a slot
/// the guest declared. Neither reads as a decode failure downstream — both are
/// well-formed surfaces — so the loss appears as a layer that samples blank,
/// which is what content that is genuinely empty also looks like. `None` reaches
/// the caller's `surface_backing_fail reason=desc_decode`, which names the surface
/// id.
///
/// Fails without the fix: both decodes return `Some`.
#[test]
fn the_surface_backing_decoder_refuses_what_it_cannot_decode_and_reports_why() {
    // Twelve planes against IOSurface's own ceiling of eight.
    let over_cap = SurfaceBackingBuilder::new(0x1000, 0x100, 0x4247_5241, 12).with_len(0x24); // 'BGRA'
                                                                                              // A legal plane count whose records the blob does not reach: `with_len`
                                                                                              // stops after plane 0, so planes 1..=3 are declared and unreachable.
    let short_records = SurfaceBackingBuilder::new(0x1000, 0x100, 0x4247_5241, 4).with_len(0x24);

    reset_surface_backing_decode_drops();
    let cap = crate::observe::FailCapture::start();
    assert!(
        decode_surface_backing(over_cap.bytes()).is_none(),
        "a plane count past IOSurface's own ceiling is a malformed descriptor, \
         and there is no correct prefix of it to publish"
    );
    let over = cap
        .lines()
        .into_iter()
        .find(|l| l.contains("reason=plane_count_over_cap"))
        .expect("an over-cap plane count must be reported");
    assert!(
        over.contains("declared=12") && over.contains("cap=8"),
        "the line must name what the guest asked for and what the device holds: {over}"
    );

    // Same reason twice is one line — the latch is what keeps a per-surface
    // stream from flooding the always-on channel. The *refusal* still applies
    // every time; only the line is deduped.
    let cap2 = crate::observe::FailCapture::start();
    assert!(
        decode_surface_backing(over_cap.bytes()).is_none(),
        "the latch must not turn the second refusal into an acceptance"
    );
    assert!(
        cap2.lines()
            .iter()
            .all(|l| !l.contains("reason=plane_count_over_cap")),
        "a repeat must not spend a second line: {:?}",
        cap2.lines()
    );

    // A declared plane whose record the blob does not reach.
    reset_surface_backing_decode_drops();
    let cap3 = crate::observe::FailCapture::start();
    assert!(
        decode_surface_backing(short_records.bytes()).is_none(),
        "a declared plane the blob does not reach must refuse the surface, \
         not publish a 0x0 plane in its slot"
    );
    let short = cap3
        .lines()
        .into_iter()
        .find(|l| l.contains("reason=plane_record_short"))
        .expect("an unreachable plane record must be reported");
    assert!(
        short.contains("plane=1"),
        "the line names the first plane that could not be reached: {short}"
    );

    // A surface larger than the 32-bit `allocSize` field can express.
    reset_surface_backing_decode_drops();
    let big = SurfaceBackingBuilder::new((u32::MAX as u64) + 1, 0x100, 0x4247_5241, 1)
        .plane(0, 0, 64, 32, 256, 0);
    let surf = decode_surface_backing(big.bytes()).expect("surface_backing decodes");
    let cap4 = crate::observe::FailCapture::start();
    let _ = synthesize_device_desc_from_surface_backing(&surf);
    let sat = cap4
        .lines()
        .into_iter()
        .find(|l| l.contains("reason=alloc_size_over_u32"))
        .expect("a length the 32-bit allocSize cannot hold must be reported");
    assert!(sat.contains("length=4294967296"), "{sat}");
}

#[test]
fn decode_surface_backing_plane0() {
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x100); // backing pfn
    st32(&mut desc[0xc..], 0x4247_5241); // 'BGRA'
    desc[0x10] = 1;
    st32(&mut desc[0x14..], 0); // plane offset
    st32(&mut desc[0x18..], 64);
    st32(&mut desc[0x1c..], 32);
    st32(&mut desc[0x20..], 256); // bpr
    let s = decode_surface_backing(&desc).expect("surface_backing");
    assert_eq!(s.length, 0x1000);
    assert_eq!(s.backing_pfn, 0x100);
    assert_eq!((s.width, s.height, s.bytes_per_row), (64, 32, 256));
    assert_eq!(s.plane_count, 1);
    assert_eq!(s.planes[0].offset, 0);
    assert!(!surface_backing_is_multiplanar(&s));
    assert_eq!(
        iosurface_pixel_format_to_mtl(s.pixel_format),
        reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM
    );
}

#[test]
fn fourcc_420f_not_bgra_and_multiplanar() {
    assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420F), 0);
    assert_eq!(iosurface_pixel_format_to_mtl(IOSURFACE_FOURCC_420V), 0);
    assert!(iosurface_fourcc_is_biplanar(IOSURFACE_FOURCC_420F));
    // Unknown FourCC must not invent BGRA.
    assert_eq!(iosurface_pixel_format_to_mtl(0xdead_beef), 0);
}

/// A small value is not an MTLPixelFormat ordinal in disguise.
///
/// The converter used to return `pixel_format as u16` for anything at or
/// below 0x200, deciding which encoding the field was in from how big the
/// number was. Every caller passes a surface backing `pixelFormat` (+0x0c), which is
/// an IOSurface OSType and therefore never below `'    '` (0x20202020), so
/// a small value arriving here is a bad read — and passing it through
/// published a format the guest never named. Fail closed instead, which is
/// what this function already does for every FourCC it does not know.
#[test]
fn a_small_value_is_not_read_as_an_mtl_ordinal() {
    // 0x50 is MTLPixelFormatBGRA8Unorm. As a surface backing OSType it is nonsense,
    // and the old magnitude test would have handed it back as a format.
    assert_eq!(iosurface_pixel_format_to_mtl(0x50), 0);
    assert_eq!(iosurface_pixel_format_to_mtl(0x200), 0);
    // Known FourCCs are unaffected — this is the boundary the old test sat
    // on, not a narrowing of what the converter accepts.
    assert_eq!(
        iosurface_pixel_format_to_mtl(0x4247_5241),
        reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM
    );
}

#[test]
fn decode_surface_backing_biplanar_420f_planes() {
    // Wire: plane0 Y 1024×1024 bpr=1024 bpe=1; plane1 UV 512×512 bpr=1024 bpe=2.
    // Live boot: fmt='420f' len=0x180000 plane0 bpr=1024.
    let mut desc = vec![0u8; 0x14 + 2 * 0x10];
    st64(&mut desc[0..], 0x180000);
    st32(&mut desc[8..], 0x200);
    st32(&mut desc[0xc..], IOSURFACE_FOURCC_420F);
    desc[0x10] = 2;
    // plane0
    st32(&mut desc[0x14..], 0); // offset
    st32(&mut desc[0x18..], 1024);
    st32(&mut desc[0x1c..], 1024);
    st32(&mut desc[0x20..], 1024 | (1 << 24)); // bpr | bpe<<24
                                               // plane1
    st32(&mut desc[0x24..], 1024 * 1024); // offset after Y
    st32(&mut desc[0x28..], 512);
    st32(&mut desc[0x2c..], 512);
    st32(&mut desc[0x30..], 1024 | (2 << 24));
    let s = decode_surface_backing(&desc).expect("surface_backing 420f");
    assert!(surface_backing_is_multiplanar(&s));
    assert_eq!(s.plane_count, 2);
    assert_eq!(
        (
            s.planes[0].width,
            s.planes[0].height,
            s.planes[0].bytes_per_row
        ),
        (1024, 1024, 1024)
    );
    assert_eq!(s.planes[0].bytes_per_element, 1);
    assert_eq!(
        (
            s.planes[1].width,
            s.planes[1].height,
            s.planes[1].bytes_per_element
        ),
        (512, 512, 2)
    );
    let dev = synthesize_device_desc_from_surface_backing(&s);
    assert_eq!(dev[DEVICE_DESC_PLANE_COUNT], 2);
    use reims_vgpu_protocol::{
        decode_device_surface, mapping_span_bound, sample_window_from_device_desc,
        DEVICE_DESC_PIXEL_FORMAT,
    };
    assert_eq!(
        ld32(&dev[DEVICE_DESC_PIXEL_FORMAT..]),
        IOSURFACE_FOURCC_420F
    );
    let surf = decode_device_surface(&dev).expect("device");
    assert_eq!(surf.plane_count, 2);
    assert_eq!(surf.alloc_size, 0x180000);
    // IOSurface texture Y plane: R8 1024×1024 matches plane0 (contract geometry key).
    let y = sample_window_from_device_desc(
        Some(&dev),
        None,
        reims_vgpu_core::pixel_format::MTL_FORMAT_R8_UNORM,
        1024,
        1024,
    )
    .expect("Y window");
    assert_eq!(y.0, 0); // offset
    assert_eq!(y.1, 1024); // bpr
                           // UV plane: RG8 half res.
    let uv = sample_window_from_device_desc(
        Some(&dev),
        None,
        reims_vgpu_core::pixel_format::MTL_FORMAT_RG8_UNORM,
        512,
        512,
    )
    .expect("UV window");
    assert_eq!(uv.0, 1024 * 1024);
    assert_eq!(uv.1, 1024);
    // A full 1024² BGRA matches no plane record, so it binds nothing, and
    // its page-sizing estimate still rejects on the wire allocation.
    assert!(sample_window_from_device_desc(
        Some(&dev),
        None,
        reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        1024,
        1024,
    )
    .is_none());
    assert!(mapping_span_bound(
        Some(&dev),
        reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM,
        1024,
        1024,
    )
    .is_none());
}

/// A failed page-table walk is not an address. The device used to answer it
/// with the backing *virtual* address used as a physical one whenever that
/// number happened to be RAM, which put a fabricated PFN into
/// `page_entries` — the list every later reader and writer resolves
/// through. Here the walk cannot resolve the backing GVA and the identity
/// candidate *is* mapped RAM, so the old path would have accepted it.
#[test]
fn resolve_surface_backing_refuses_to_substitute_the_gva_when_the_walk_fails() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    // The identity candidate is backed RAM: `read_gpa` succeeds on it, which
    // is the whole of what the old gate checked.
    host.map_range(0x20u64 << PAGE_SHIFT_X86, 0x2000, 0x5a);
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x200, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // root[0] carries the object list and descriptors. root[0x20] — the
    // backing GVA page — is left unmapped, so the backing walk fails.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, 8));
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x20); // backing GVA page — unmapped in this task
    st32(&mut desc[0xc..], 0x50);
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);

    assert!(
        !resolve_surface_backing(&mut state, &host, 3),
        "an untranslatable backing must not resolve"
    );
    // The refusal happens before any mutation, so no fabricated entry is
    // left behind for a later writer to aim at.
    let fabricated = state
        .surfaces
        .mappings
        .get(&3)
        .map(|m| m.lifecycle.active || !m.pages.entries.is_empty())
        .unwrap_or(false);
    assert!(!fabricated, "refusal must not cache a fabricated backing");
}

/// `resolve_surface_backing_ex` probes task 0 first and returns on the first
/// task whose backing applies. The identity guess made task 0 succeed for
/// surfaces it could not translate, so the search stopped there and the
/// owning task was never tried — the surface was then backed by an address
/// derived from a virtual one. Refusing is what lets the loop continue.
///
/// Both tasks list the surface, as task 0 (the kernel/global list) and the
/// owner do in production; only the owner can translate the backing.
#[test]
fn the_task_search_reaches_the_owner_when_task_zero_cannot_translate() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    let dir0_gpa = 2u64 << PAGE_SHIFT_X86;
    let root0_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    let dir1_gpa = 7u64 << PAGE_SHIFT_X86;
    let root1_gpa = 8u64 << PAGE_SHIFT_X86;
    let real_page = 9u64 << PAGE_SHIFT_X86;
    for (gpa, len) in [
        (dir0_gpa, 0x20),
        (root0_gpa, 0x1000),
        (data_gpa, 0x200),
        (dir1_gpa, 0x20),
        (root1_gpa, 0x1000),
        (real_page, 0x1000),
    ] {
        host.map_range(gpa, len, 0);
    }
    // The identity candidate for the backing GVA is RAM, so the old path
    // would have taken it on task 0 rather than moving on.
    host.map_range(0x20u64 << PAGE_SHIFT_X86, 0x1000, 0x5a);

    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir0_gpa, &d);
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 8);
    let _ = host.write_gpa(dir1_gpa, &d);
    // Both roots reach the object list at GVA 0; only task 1's maps the
    // backing GVA page 0x20, and it maps it to `real_page`.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root0_gpa, &d[..4]);
    let _ = host.write_gpa(root1_gpa, &d[..4]);
    st32(&mut d[..4], 9);
    let _ = host.write_gpa(root1_gpa + 0x20 * 4, &d[..4]);

    state.define_task(0, 0x1000, 2);
    assert!(state.set_object_list(0, 0, 8));
    state.define_task(1, 0x1000, 7);
    assert!(state.set_object_list(1, 0, 8));

    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 0x20);
    st32(&mut desc[0xc..], 0x50);
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);

    assert!(
        resolve_surface_backing(&mut state, &host, 3),
        "the owning task can translate the backing, so the resolve must succeed"
    );
    let m = state.surfaces.mappings.get(&3).unwrap();
    assert_eq!(m.pages.entries.len(), 1);
    assert_eq!(
        entry_gpa_shift(m.pages.entries[0], PAGE_SHIFT_X86),
        Some(real_page),
        "the backing must come from the task that could translate it, \
         not from task 0's untranslatable GVA"
    );
}

/// The search stops on the first task that can back a surface, so whether
/// that choice was ever a choice is the thing to count. Nothing on the wire
/// can verify a candidate — the object-list entry carries no identity and
/// the surface backing descriptor is fully decoded — so the claimant count is the
/// only available reading of the search's exposure, and it has to
/// distinguish "one task lists this id" from "two do".
#[test]
fn a_surface_id_claimed_by_two_tasks_is_counted_as_two() {
    // Two tasks, each with its own directory and root, both listing eight
    // object slots at GVA 0. Task 0's list page holds a surface backing surface at
    // slot 3; task 1's holds a IOSurface plane view there until the second half of the
    // test rewrites it.
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    let dir0_gpa = 2u64 << PAGE_SHIFT_X86;
    let root0_gpa = 3u64 << PAGE_SHIFT_X86;
    let list0_gpa = 4u64 << PAGE_SHIFT_X86;
    let dir1_gpa = 7u64 << PAGE_SHIFT_X86;
    let root1_gpa = 8u64 << PAGE_SHIFT_X86;
    let list1_gpa = 9u64 << PAGE_SHIFT_X86;
    for (gpa, len) in [
        (dir0_gpa, 0x20),
        (root0_gpa, 0x1000),
        (list0_gpa, 0x200),
        (dir1_gpa, 0x20),
        (root1_gpa, 0x1000),
        (list1_gpa, 0x200),
    ] {
        host.map_range(gpa, len, 0);
    }

    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir0_gpa, &d);
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 8);
    let _ = host.write_gpa(dir1_gpa, &d);
    // Each task's GVA page 0 reaches its own list page.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root0_gpa, &d[..4]);
    st32(&mut d[..4], 9);
    let _ = host.write_gpa(root1_gpa, &d[..4]);

    state.define_task(0, 0x1000, 2);
    assert!(state.set_object_list(0, 0, 8));
    state.define_task(1, 0x1000, 7);
    assert!(state.set_object_list(1, 0, 8));

    // Slot 3 of task 0 is the surface. Both entries carry a descriptor GVA
    // and length, which is what `lookup_list_entry` requires before the type
    // is even looked at.
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], OBJECT_TYPE_SURFACE as u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(list0_gpa + 3 * 12, &entry);

    // Task 1 lists a *different object type* at the same slot, so it is not
    // a claimant even though the slot is populated.
    let mut other = [0u8; 12];
    st32(
        &mut other[0..],
        OBJECT_TYPE_REF_TEXTURE as u32 | (0x30u32 << 8),
    );
    other[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(list1_gpa + 3 * 12, &other);

    assert_eq!(
        surface_backing_claimant_tasks(&state, &host, 3),
        vec![0],
        "a populated slot of another object type is not a claim on this id"
    );

    // Now task 1 lists a surface backing surface at the same slot. The id spaces are
    // per task, so this is a second, unrelated surface wearing the same id —
    // and the search would have to break the tie by probe order alone.
    let _ = host.write_gpa(list1_gpa + 3 * 12, &entry);
    assert_eq!(
        surface_backing_claimant_tasks(&state, &host, 3),
        vec![0, 1],
        "both tasks list a surface backing surface at slot 3, so both are claimants"
    );

    // An inactive task cannot be the one the search stops on, so it is not
    // counted either.
    state.tasks[1].active = false;
    assert_eq!(
        surface_backing_claimant_tasks(&state, &host, 3),
        vec![0],
        "an inactive task is not a claimant"
    );
}

/// Force-resolve must rebuild the cached page table when the task PT
/// translation of the backing GVA moved (same surface id, same geometry,
/// new physical pages — the early-boot FB vs WindowServer reallocation).
#[test]
fn resolve_surface_backing_force_rebuilds_when_task_translation_moves() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    let old_page = 5u64 << PAGE_SHIFT_X86;
    let new_page = 6u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x200, 0);
    host.map_range(old_page, 0x1000, 0x11);
    host.map_range(new_page, 0x1000, 0x22);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // root[0] = data page (object list + descriptors), root[1] = old backing.
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    st32(&mut d[..4], 5);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, 8));
    // Surface backing entry at surface_id=3, descriptor at GVA 0x80.
    let mut entry = [0u8; 12];
    st32(&mut entry[0..], 4u32 | (0x30u32 << 8));
    entry[4..12].copy_from_slice(&0x80u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 3 * 12, &entry);
    let mut desc = vec![0u8; 0x30];
    st64(&mut desc[0..], 0x1000);
    st32(&mut desc[8..], 1); // backing_pfn = GVA page 1
    st32(&mut desc[0xc..], 0x50);
    desc[0x10] = 1;
    st32(&mut desc[0x18..], 16);
    st32(&mut desc[0x1c..], 16);
    st32(&mut desc[0x20..], 64);
    let _ = host.write_gpa(data_gpa + 0x80, &desc);

    assert!(resolve_surface_backing(&mut state, &host, 3));
    {
        let m = state.surfaces.mappings.get(&3).unwrap();
        assert_eq!(m.pages.entries.len(), 1);
        assert_eq!(
            entry_gpa_shift(m.pages.entries[0], PAGE_SHIFT_X86),
            Some(old_page)
        );
        assert_eq!(m.lifecycle.generation, 1);
    }
    // Guest remaps GVA page 1 onto a new physical page (same id/geometry).
    st32(&mut d[..4], 6);
    let _ = host.write_gpa(root_gpa + 4, &d[..4]);
    assert!(resolve_surface_backing_force(&mut state, &host, 3));
    {
        let m = state.surfaces.mappings.get(&3).unwrap();
        assert_eq!(
            entry_gpa_shift(m.pages.entries[0], PAGE_SHIFT_X86),
            Some(new_page),
            "force-resolve must follow the moved translation"
        );
        assert_eq!(m.lifecycle.generation, 2, "page move bumps map_generation");
    }
    // Unchanged translation: force keeps the table without a rebuild.
    assert!(resolve_surface_backing_force(&mut state, &host, 3));
    let m = state.surfaces.mappings.get(&3).unwrap();
    assert_eq!(m.lifecycle.generation, 2);
    assert_eq!(
        entry_gpa_shift(m.pages.entries[0], PAGE_SHIFT_X86),
        Some(new_page)
    );
}

/// A genuine backing failure (a surface whose descriptor decoded fine but
/// whose page-backing construction fails) must be fail-visible with a
/// `reason=` slug, deduped per `(surface_id, reason)`, and re-armed when the
/// surface next backs cleanly — never a silent `return false` that paints
/// stale/black with no log. Locks the surface backing backing blind-spot closure.
#[test]
fn apply_surface_backing_fail_latches_reason_and_rearms() {
    let host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    state.page_shift = PAGE_SHIFT_X86;
    // A surface_id other surface backing tests do not touch (they use 3).
    let sid = 11u32;
    clear_surface_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    assert!(!surface_backing_fail_latch()
        .lock()
        .unwrap()
        .contains_key(&(sid, "task_inactive")));
    // Small valid length (page_count = 1) so the alloc-guard passes, then an
    // undefined/inactive task_id hits the `task_inactive` site — the drain
    // race where a decoded surface's owning task died before backing landed.
    let surf = SurfaceBackingDescriptor {
        length: 0x1000,
        backing_pfn: 0x20,
        pixel_format: 0,
        plane_count: 1,
        planes: [SurfaceBackingPlane::default(); SURFACE_BACKING_PLANE_CAP],
        width: 16,
        height: 16,
        bytes_per_row: 64,
    };
    assert!(!apply_surface_backing(&mut state, &host, 5, sid, &surf));
    assert!(
        !surface_backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "task_inactive")),
        "one task's probe is not a backing failure: the search has other \
         tasks to try, and reporting here is what put `reason=translate` \
         lines under surfaces that then backed cleanly"
    );
    // The search running out of tasks is what turns the probe's reason into
    // a reported failure.
    flush_surface_backing_fail(sid);
    assert!(
        surface_backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "task_inactive")),
        "an exhausted search must report the first probe's reason slug"
    );
    // A clean backing on the same surface re-arms the latch.
    clear_surface_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    assert!(
        !surface_backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "task_inactive")),
        "clear_surface_backing_fail must re-arm so a later failure logs again"
    );
}

/// A refusal that the next attach resolves is reported as the recovery it is,
/// and only when the backing that landed is the one the refusal named.
///
/// `surface_backing_fail reason=translate` reads as lost guest work and usually is
/// not: `st=zero-pfn` means the guest had not finished mapping when the
/// per-present path walked it, and the refusal exists so the device asks again
/// rather than substituting a guess. Every one of the six on a driven boot
/// recovered, within 1-21 ms, and nothing in the log said so — see
/// [`super::clear_surface_backing_fail`].
///
/// The match is on the **backing address**, never on `surface_id`: ids recycle
/// within a boot and across geometries, so a clean attach on a recycled id must
/// re-arm the latch (it does, as the test above locks) without claiming that the
/// earlier, different surface recovered.
/// A refusal leaves the latch three ways, and all three are now countable.
///
/// Recovered, superseded, or still there. The third is the only one that can be
/// lost guest work, and before `surface_backing_superseded` and
/// `surface_backing_outstanding` it was indistinguishable from the second: a
/// driven boot with five `surface_backing_fail` lines and four
/// `surface_backing_recovered` lines gave a reader no way to tell a refusal the
/// guest walked away from apart from one that never came back, short of
/// hand-matching backing GVAs across the log. That is how this gap was found.
///
/// The identity is what makes the census readable, so it is what this asserts:
/// every refusal that leaves the latch is accounted for by exactly one of the
/// three, and the residue is the census line's `n`.
#[test]
fn every_surface_backing_refusal_leaves_the_latch_recovered_superseded_or_counted() {
    use crate::runtime::drain::store_route_count;

    // Ids no other test in this module uses; the latch is process-global.
    let sid = 0x4d2u32;
    let gva = 0x4222000u64;
    clear_surface_backing_fail(sid, gva);

    // Superseded: refused at `gva`, backed somewhere else. Not a recovery, and
    // it used to be the silent one.
    let before = store_route_count("surface_backing_superseded");
    let recovered_before = store_route_count("surface_backing_recovered");
    defer_surface_backing_fail(
        sid,
        "translate",
        Some(gva),
        "surface_backing_fail probe".into(),
    );
    flush_surface_backing_fail(sid);
    clear_surface_backing_fail(sid, gva + 0x1000);
    assert_eq!(
        store_route_count("surface_backing_superseded"),
        before + 1,
        "a refusal dropped because the surface backed elsewhere must be counted"
    );
    assert_eq!(
        store_route_count("surface_backing_recovered"),
        recovered_before,
        "and must not be claimed as a recovery"
    );

    // Recovered: refused at `gva`, backed at `gva`. Counts on the other route
    // and leaves the superseded count alone — the two must not double-count one
    // refusal, which is what would make the identity stop holding.
    let before = store_route_count("surface_backing_superseded");
    let recovered_before = store_route_count("surface_backing_recovered");
    defer_surface_backing_fail(
        sid,
        "translate",
        Some(gva),
        "surface_backing_fail probe".into(),
    );
    flush_surface_backing_fail(sid);
    clear_surface_backing_fail(sid, gva);
    assert_eq!(
        store_route_count("surface_backing_recovered"),
        recovered_before + 1,
        "an attach on the backing the refusal named is a recovery"
    );
    assert_eq!(
        store_route_count("surface_backing_superseded"),
        before,
        "and is not also a supersede"
    );
}

/// The census names an outstanding refusal, and says nothing when there is none.
///
/// `oldest_ms` is the field that distinguishes a retry caught mid-flight from a
/// surface this device never backed, so a line without it would report the same
/// thing in both cases — which is the state this replaced.
#[test]
fn the_outstanding_census_names_the_oldest_refusal_and_is_otherwise_silent() {
    let sid = 0x4d3u32;
    let gva = 0x4333000u64;
    clear_surface_backing_fail(sid, gva);

    // Other tests in this module share the latch, so assert about *this* sid
    // rather than about emptiness — a bare `is_none()` would be order-dependent.
    let mine = |line: &Option<String>| {
        line.as_deref()
            .is_some_and(|l| l.contains(&format!("sid={sid}")))
    };
    assert!(
        !mine(&surface_backing_outstanding_census()),
        "nothing is latched for this surface yet"
    );

    defer_surface_backing_fail(
        sid,
        "translate",
        Some(gva),
        "surface_backing_fail probe".into(),
    );
    flush_surface_backing_fail(sid);
    let line = surface_backing_outstanding_census().expect("a latched refusal must be censused");
    assert!(
        line.starts_with("surface_backing_outstanding n=") && line.contains("oldest_ms="),
        "the line must carry both the count and the age: {line}"
    );
    assert!(
        line.contains(&format!("gva={gva:#x}")) || !mine(&Some(line.clone())),
        "when this surface is the oldest, the line must name its backing: {line}"
    );

    // Retiring it removes it from the census, by either route.
    clear_surface_backing_fail(sid, gva);
    assert!(
        !mine(&surface_backing_outstanding_census()),
        "a recovered refusal is no longer outstanding"
    );
}

/// A retried refusal and an abandoned one must not read the same.
///
/// The distinction the census exists to make, and the one it could not make for
/// two sessions. A surface the device asks for every frame and is refused every
/// frame is losing guest work; a surface it asked for once and never again is
/// one the guest stopped presenting. Both sit in the latch as `n=1`.
///
/// The trap this pins: `note_surface_backing_fail` refreshes its timestamp on a repeat, so
/// `oldest_ms` alone reads **backwards** — a live retry holds it near zero and
/// an abandoned refusal lets it grow with the clock. `attempts` is what makes
/// the line state which of the two it is without anyone re-deriving that.
#[test]
fn a_retried_surface_backing_refusal_counts_its_attempts_and_an_abandoned_one_does_not() {
    let sid = 0x4d3u32;
    let gva = 0x4188000u64;
    clear_surface_backing_fail(sid, gva);
    let is_mine = |line: &Option<String>| {
        line.as_ref()
            .is_some_and(|l| l.contains(&format!("sid={sid} ")))
    };

    // Asked once and refused.
    defer_surface_backing_fail(sid, "translate", Some(gva), "first refusal".into());
    flush_surface_backing_fail(sid);
    let line = surface_backing_outstanding_census().expect("a latched refusal is censused");
    if is_mine(&Some(line.clone())) {
        assert!(
            line.contains("attempts=1"),
            "one refusal is one attempt: {line}"
        );
        assert!(
            line.contains("since_last_ms=") && line.contains("oldest_ms="),
            "both ages travel, or `attempts` cannot be placed in time: {line}"
        );
    }

    // Asked again and refused again, four more times. The fail channel stays
    // quiet — this is the per-present path and one line a frame would flood it
    // — so the count is the only thing saying the device is still trying.
    for _ in 0..4 {
        defer_surface_backing_fail(sid, "translate", Some(gva), "retry refusal".into());
        flush_surface_backing_fail(sid);
    }
    let line = surface_backing_outstanding_census().expect("still latched");
    if is_mine(&Some(line.clone())) {
        assert!(
            line.contains("attempts=5"),
            "four retries after the first must be counted, not deduped away: {line}"
        );
    }

    clear_surface_backing_fail(sid, gva);
    assert!(
        !is_mine(&surface_backing_outstanding_census()),
        "the latch re-arms once the surface backs"
    );
}

#[test]
fn a_surface_backing_refusal_the_next_attach_resolves_is_reported_as_recovered() {
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

    // A surface id no other test in this module uses.
    let sid = 0x4d1u32;
    let gva = 0x4112000u64;
    clear_surface_backing_fail(sid, gva);

    // A reported refusal naming this backing...
    let mark = log_mark();
    defer_surface_backing_fail(
        sid,
        "translate",
        Some(gva),
        "surface_backing_fail probe".into(),
    );
    flush_surface_backing_fail(sid);
    assert!(
        log_since(mark).contains("surface_backing_fail probe"),
        "the exhausted search must report the probe's reason"
    );

    // ...that a later attach on a *different* backing must not claim.
    let mark = log_mark();
    clear_surface_backing_fail(sid, gva + 0x1000);
    assert!(
        !log_since(mark).contains("surface_backing_recovered"),
        "a recycled surface id is not evidence that the earlier backing landed"
    );
    assert!(
        !surface_backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "translate")),
        "the latch must still re-arm, or a later genuine failure goes unlogged"
    );

    // The same refusal, then an attach on the backing it named, is a recovery.
    let mark = log_mark();
    defer_surface_backing_fail(
        sid,
        "translate",
        Some(gva),
        "surface_backing_fail probe".into(),
    );
    flush_surface_backing_fail(sid);
    clear_surface_backing_fail(sid, gva);
    let log = log_since(mark);
    assert!(
        log.contains("surface_backing_recovered")
            && log.contains(&format!("sid={sid}"))
            && log.contains("reason=translate")
            && log.contains(&format!("gva={gva:#x}")),
        "a refusal whose backing then landed must say so:\n{log}"
    );
}

/// A refused walk must say **which** of the walk's checks refused.
///
/// The walk distinguishes fifteen refusals and this rail reported one word,
/// `translate`, for all of them — so "the guest has not filled in this leaf
/// PTE yet" and "this device could not read the task root at all" produced
/// identical log lines while wanting opposite responses. Both halves are
/// locked here: the walk names its failing check, and the detail line
/// carries that name verbatim.
///
/// The fixture maps GVA page 0 and nothing else, so the *same* task walks
/// clean for one address and refuses for the next. Asserting the clean case
/// too is what keeps this from passing vacuously: a fixture in which every
/// walk fails would satisfy the refusal assertions on its own.
#[test]
fn a_refused_surface_backing_walk_names_the_check_that_refused() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);
    let task = state.tasks.get(1).expect("fixture defines task 1");

    // Control: the address the fixture does map walks all the way down.
    let mapped = gva_mem::diagnose_task_slot(&host, task, 1, 0, PAGE_SHIFT_ARM64E);
    assert!(
        mapped.contains("st=ok"),
        "fixture must be able to translate, got {mapped:?}"
    );

    // The case the rig produces: a backing whose leaf entry the guest has
    // not written. Page 1 shares the fixture's root and has no PTE.
    let gva = 1u64 << PAGE_SHIFT_ARM64E;
    let walk = gva_mem::diagnose_task_slot(&host, task, 1, gva, PAGE_SHIFT_ARM64E);
    assert!(
        walk.contains("st=zero-pfn"),
        "an unwritten leaf must be reported as zero-pfn, got {walk:?}"
    );
    assert!(
        walk.contains("lvl=") && walk.contains("idx="),
        "the refusal must name where in the walk it stopped, got {walk:?}"
    );

    let line = surface_backing_translate_fail_detail(202, 1, 0, 640, gva, &walk);
    assert!(line.contains("reason=translate"), "{line}");
    assert!(line.contains("sid=202"), "{line}");
    assert!(line.contains("page=0/640"), "{line}");
    assert!(
        line.contains(&format!("walk=[{walk}]")),
        "the refusal must carry the walk diagnosis verbatim, got {line}"
    );
}

/// A refused object-list entry read names the three inputs its address came
/// from, not just the address.
///
/// `gva_mem`'s own refusal can only print the gva, because it is generic over
/// every caller. Here the gva is derived — `(list_pfn << page_shift) +
/// ref * entry_len` — and a driven x86 boot emits ten of these all reading
/// `gva=0x11b0`, which is `pfn = 1, ref = 36` and is equally consistent with
/// the guest not having mapped its list yet and with this device resolving a
/// ref against the wrong task. The address alone cannot separate those; the
/// inputs can, which is why they have to be on the line.
///
/// Asserts the fields rather than the prose, so rewording the parenthetical
/// does not fail it.
#[test]
fn a_refused_object_list_entry_names_the_geometry_behind_its_address() {
    let task = crate::model::TaskEntry {
        active: true,
        length: 0x1000,
        directory_pfn: 2,
        object_list_pfn: 1,
        object_list_count: 64,
    };
    let entry_gva = (1u64 << PAGE_SHIFT_X86) + 36 * OBJECT_LIST_ENTRY_LEN as u64;
    let line = list_entry_unreadable_detail(3, 36, &task, entry_gva);

    assert!(line.contains("task=3"), "{line}");
    assert!(line.contains("ref=36"), "{line}");
    assert!(line.contains("gva=0x11b0"), "{line}");
    assert!(
        line.contains("list_pfn=1"),
        "the pfn the address was built from must be on the line: {line}"
    );
    assert!(
        line.contains("list_count=64"),
        "the count that admitted this ref must be on the line: {line}"
    );
    assert!(
        line.contains("entry_len=12"),
        "the stride the offset was scaled by must be on the line: {line}"
    );
}

/// A task the guest has defined but never given an object list to must
/// resolve **nothing** — not another task's list.
///
/// This reproduces, at unit scale, what the rail was measured doing on every
/// boot. `TaskEntry::define` used to invent `object_list_pfn = 1` and
/// `count = 0x100000`, so a task with no `SetObjectList` still computed an
/// entry address of `0x1000 + off`. Nothing is mapped there for that task,
/// the walk failed `gva_zero_pfn`, and `read_task_gva_by_id` then walked
/// task `5 >> 1 == 2`'s page table at the same address — where task 2's
/// object list genuinely lives — and decoded task 2's entry as task 5's.
///
/// Task 2's own lookup is asserted first so the fixture is known to be real:
/// a test where the donor list is unreadable would pass for the wrong reason.
#[test]
fn a_task_with_no_object_list_resolves_nothing_not_its_neighbours_list() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    // PTE for GVA page 1 (0x1000) → pfn 4, so task 2's list is readable.
    let mut pte = [0u8; 4];
    st32(&mut pte, 4);
    let _ = host.write_gpa(root_gpa + 4, &pte);

    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_SURFACE as u32) | (0x40u32 << 8),
    );
    entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa, &entry);

    // Task 2 owns a real list at pfn 1. Task 5 has a directory that maps
    // nothing, and `5 >> 1 == 2`.
    state.define_task(2, 0x1000, 2);
    assert!(state.set_object_list(2, 1, 4));
    state.define_task(5, 0x1000, 9);

    let donor = lookup_list_entry(&state, &host, 2, 0);
    assert!(
        donor.is_some(),
        "fixture is not real: task 2's own list must be readable"
    );

    // The behavioural claim first, so a regression fails on the corruption
    // itself rather than on the field that causes it.
    assert_eq!(
        lookup_list_entry(&state, &host, 5, 0),
        None,
        "task 5 has no object list, so it must resolve nothing — returning \
         Some here is task 2's entry answering for task 5"
    );
    assert_eq!(
        state.tasks[5].object_list_pfn, 0,
        "a defined task has no list until SetObjectList says so"
    );
    assert_eq!(state.tasks[5].object_list_count, 0);
}

/// The probe and the named lookup must give the **same answer**. Only whether a
/// miss is reportable differs.
///
/// This is the half a regression would break. `probe_list_entry` exists because
/// `surface_backing_probe_order` walks every live task asking who owns a surface, so it
/// misses on every task before the owner — 18 `gva_read_refused` lines per
/// driven boot, all of them the search working. Quietening that is only correct
/// while it still *answers* identically; a probe that skipped the liveness test,
/// or read a different address, would pass a "no line was emitted" check and
/// still be wrong.
///
/// Same fixture as the test above: task 2 owns a readable list at pfn 1, task 5
/// has a directory that maps nothing, and `5 >> 1 == 2` — so a probe that fell
/// through to a neighbour's page table would answer `Some` for task 5.
#[test]
fn the_probe_and_the_named_lookup_answer_identically() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    let mut pte = [0u8; 4];
    st32(&mut pte, 4);
    let _ = host.write_gpa(root_gpa + 4, &pte);
    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_SURFACE as u32) | (0x40u32 << 8),
    );
    entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa, &entry);

    state.define_task(2, 0x1000, 2);
    assert!(state.set_object_list(2, 1, 4));
    state.define_task(5, 0x1000, 9);

    for (task, ref_, what) in [
        (2u32, 0u32, "the owner's own entry"),
        (5, 0, "a task whose list does not translate"),
        (2, 3, "a slot inside the list the guest never filled"),
        (77, 0, "a task nothing defined"),
    ] {
        assert_eq!(
            probe_list_entry(&state, &host, task, ref_),
            lookup_list_entry(&state, &host, task, ref_),
            "probe and named lookup disagree on {what} (task {task}, ref {ref_})"
        );
    }

    // And the fixture is real in both directions, so the loop above cannot be
    // passing by answering `None` to everything.
    assert!(
        probe_list_entry(&state, &host, 2, 0).is_some(),
        "the owner must be found through the probe — that is what the search is for"
    );
    assert_eq!(probe_list_entry(&state, &host, 5, 0), None);
}

fn setup_surface_backing_candidate(
    host: &mut FakeHost,
    state: &mut Device,
    surface_id: u32,
    desc_gva: u64,
    desc_len: u32,
) -> u64 {
    let dir_gpa = 2u64 << PAGE_SHIFT_X86;
    let root_gpa = 3u64 << PAGE_SHIFT_X86;
    let data_gpa = 4u64 << PAGE_SHIFT_X86;
    host.map_range(dir_gpa, 0x20, 0);
    host.map_range(root_gpa, 0x1000, 0);
    host.map_range(data_gpa, 0x1000, 0);
    let mut d = [0u8; 8];
    st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
    st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
    let _ = host.write_gpa(dir_gpa, &d);
    st32(&mut d[..4], 4);
    let _ = host.write_gpa(root_gpa, &d[..4]);
    state.define_task(1, 0x1000, 2);
    assert!(state.set_object_list(1, 0, surface_id + 1));

    let mut entry = [0u8; OBJECT_LIST_ENTRY_LEN];
    st32(
        &mut entry[0..],
        (OBJECT_TYPE_SURFACE as u32) | (desc_len << 8),
    );
    entry[4..12].copy_from_slice(&desc_gva.to_le_bytes());
    let entry_gpa = data_gpa + surface_id as u64 * OBJECT_LIST_ENTRY_LEN as u64;
    let _ = host.write_gpa(entry_gpa, &entry);
    data_gpa
}

/// Once task-scan lookup finds an actual surface backing candidate, descriptor read
/// failure is no longer speculative: the surface has an owner but cannot get
/// backing. It must be fail-visible with a stable reason slug.
#[test]
fn resolve_surface_backing_candidate_logs_descriptor_read_failure() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let sid = 17u32;
    clear_surface_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    let _ = setup_surface_backing_candidate(&mut host, &mut state, sid, 0x3000, 0x30);

    assert!(!resolve_surface_backing(&mut state, &host, sid));
    assert!(
        surface_backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "desc_read")),
        "surface-type candidate with unreadable descriptor must name desc_read"
    );
    clear_surface_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
}

/// A readable but invalid surface backing descriptor used to fall through to the
/// resolver tail with no site reason. Keep it fail-visible without logging
/// absent/non-surface speculative probes.
#[test]
fn resolve_surface_backing_candidate_logs_descriptor_decode_failure() {
    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let sid = 18u32;
    clear_surface_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
    let data_gpa = setup_surface_backing_candidate(&mut host, &mut state, sid, 0x80, 0x30);
    let bad_desc = vec![0u8; 0x30];
    let _ = host.write_gpa(data_gpa + 0x80, &bad_desc);

    assert!(!resolve_surface_backing(&mut state, &host, sid));
    assert!(
        surface_backing_fail_latch()
            .lock()
            .unwrap()
            .contains_key(&(sid, "desc_decode")),
        "surface-type candidate with invalid descriptor must name desc_decode"
    );
    clear_surface_backing_fail(sid, 0x20u64 << PAGE_SHIFT_X86);
}

/// Live wire bytes (boot 093019 `compute_stage_tex iosurface_plane_view … args_hex`):
/// R8 1024×1024 = Y plane view of a biplanar 1024×1024 surface.
#[test]
fn decode_iosurface_plane_view_live_r8_y_plane() {
    let mut desc = vec![0u8; 8];
    st32(&mut desc[IOSURFACE_PLANE_VIEW_SURFACE_ID..], 8);
    // args blob: kind 0x2f, len 0x30, own_ref 0x15, record R8 1024×1024 d=1.
    let args = [
        0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x15, 0, 0, 0, // kind, blob_len, own_ref
        0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
        0x00, 0x04, 0x00, 0x00, // width 1024
        0x00, 0x04, 0x00, 0x00, // height 1024
        0x01, 0x00, 0x00, 0x00, // depth 1
        0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer (unconsumed)
    ];
    desc.extend_from_slice(&args);
    let rec = decode_iosurface_plane_view(&desc).expect("live R8 record decodes");
    assert_eq!(rec.pixel_format, 0x0a);
    assert_eq!((rec.width, rec.height, rec.depth), (1024, 1024, 1));
    // Short record (no +0x20 field) defaults to plane 0.
    assert_eq!(rec.plane_index, 0);
}

/// Live 56-byte wire blob from the BLIT copy-source path (x86 Ventura
/// 13.7.8, 2026-07-19 `blit t5_view_decode sid=34`): a full-color
/// texture view (BGRA8_sRGB 1024×768 window backing) carries the sibling
/// record tag `0x62`, not the biplanar `0x42`. Same field layout — must
/// decode, or the blit path drops the copy.
#[test]
fn decode_iosurface_plane_view_live_0x62_color_window_view() {
    // Exact leading 40 bytes observed, zero-padded to the 56-byte desc_len.
    let head: [u8; 40] = [
        0x22, 0x00, 0x00, 0x00, // surface_id = 34
        0x00, 0x00, 0x00, 0x00, // field
        0x2f, 0x00, 0x00, 0x00, // kind 0x2f
        0x30, 0x00, 0x00, 0x00, // blob_len 0x30
        0x0b, 0x00, 0x00, 0x00, // own_ref 0x0b
        0x62, 0x00, 0x51, 0x00, // tag=0x62, unk, fmt=0x51 BGRA8_sRGB
        0x00, 0x04, 0x00, 0x00, // width 1024
        0x00, 0x03, 0x00, 0x00, // height 768
        0x01, 0x00, 0x00, 0x00, // depth 1
        0x01, 0x00, 0x01, 0x00, // trailer
    ];
    let mut desc = head.to_vec();
    desc.resize(56, 0); // plane field (+0x20 in record) reads 0
    let rec = decode_iosurface_plane_view(&desc).expect("0x62 color view must decode");
    assert_eq!(rec.pixel_format, 0x51);
    assert_eq!((rec.width, rec.height, rec.depth), (1024, 768, 1));
    assert_eq!(rec.plane_index, 0);
}

/// Live 56-byte wire blob (boot 20260717-063043, v0a8 hero): the record
/// carries the `newTextureWithDescriptor:iosurface:plane:` plane at
/// `+0x20` — Y views carry 0, the RG8 chroma view 1, the same-geometry
/// alpha view 2. Geometry cannot separate Y from alpha; this field does.
#[test]
fn decode_iosurface_plane_view_live_v0a8_alpha_plane_index() {
    let mut desc = vec![0u8; 8];
    st32(&mut desc[IOSURFACE_PLANE_VIEW_SURFACE_ID..], 0x6d);
    let args = [
        0x2fu8, 0, 0, 0, 0x30, 0, 0, 0, 0x82, 0x01, 0, 0, // kind, blob_len, own_ref
        0x42, 0x01, 0x0a, 0x00, // tag, unk, fmt=R8
        0xb2, 0x03, 0x00, 0x00, // width 946
        0x5e, 0x01, 0x00, 0x00, // height 350
        0x01, 0x00, 0x00, 0x00, // depth 1
        0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x10, 0x00, // trailer
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
        0x02, 0x00, 0x00, 0x00, // IOSurface plane index = 2 (alpha)
    ];
    desc.extend_from_slice(&args);
    let rec = decode_iosurface_plane_view(&desc).expect("live v0a8 alpha record decodes");
    assert_eq!(rec.pixel_format, 0x0a);
    assert_eq!((rec.width, rec.height, rec.depth), (946, 350, 1));
    assert_eq!(rec.plane_index, 2);
}

/// The owner-task census must read the dword the guest wrote, and must be
/// able to tell 0 from anything else.
///
/// A census whose extraction is wrong reports 0 forever whatever the wire
/// says, and 0 is the answer this device already assumes — so the failing
/// case would be indistinguishable from the healthy one, which is the whole
/// point of having it. Pinning the offset against a descriptor whose *other*
/// leading dword is non-zero is what makes an off-by-four visible.
#[test]
fn the_iosurface_plane_view_owner_task_is_read_from_its_own_dword() {
    let mut desc = [0u8; IOSURFACE_PLANE_VIEW_MIN_LEN];
    st32(&mut desc[IOSURFACE_PLANE_VIEW_SURFACE_ID..], 0xabcd);
    assert_eq!(
        ld32(&desc[IOSURFACE_PLANE_VIEW_OWNER_TASK..]),
        0,
        "the surface id must not be read as the owner task"
    );
    st32(&mut desc[IOSURFACE_PLANE_VIEW_OWNER_TASK..], 7);
    assert_eq!(ld32(&desc[IOSURFACE_PLANE_VIEW_OWNER_TASK..]), 7);
    assert_eq!(
        ld32(&desc[IOSURFACE_PLANE_VIEW_SURFACE_ID..]),
        0xabcd,
        "writing the owner task must not disturb the surface id"
    );
    // Both fields sit inside the minimum descriptor — the array above is
    // exactly `IOSURFACE_PLANE_VIEW_MIN_LEN` and indexing it proves that — so the census can
    // never be silently skipped on a well-formed record.
    assert_eq!(
        IOSURFACE_PLANE_VIEW_OWNER_TASK,
        IOSURFACE_PLANE_VIEW_SURFACE_ID + 4
    );
}

#[test]
fn decode_iosurface_plane_view_fail_closed() {
    // Short descriptor (no record).
    let mut short = vec![0u8; 8];
    st32(&mut short[IOSURFACE_PLANE_VIEW_SURFACE_ID..], 8);
    assert!(decode_iosurface_plane_view(&short).is_none());
    // Wrong record tag.
    let mut bad_tag = vec![0u8; 8];
    st32(&mut bad_tag[IOSURFACE_PLANE_VIEW_SURFACE_ID..], 8);
    bad_tag.extend_from_slice(&[0u8; 12]);
    bad_tag.extend_from_slice(&[
        0x41, 0x01, 0x0a, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x01, 0, 0, 0,
    ]);
    assert!(decode_iosurface_plane_view(&bad_tag).is_none());
    // Non-2D (depth != 1) fails closed.
    let mut vol = vec![0u8; 8];
    st32(&mut vol[IOSURFACE_PLANE_VIEW_SURFACE_ID..], 8);
    vol.extend_from_slice(&[0u8; 12]);
    vol.extend_from_slice(&[
        0x42, 0x07, 0x50, 0x00, 0x40, 0, 0, 0, 0x40, 0, 0, 0, 0x40, 0, 0, 0,
    ]);
    assert!(decode_iosurface_plane_view(&vol).is_none());
    // Zero width fails closed.
    let mut zw = vec![0u8; 8];
    st32(&mut zw[IOSURFACE_PLANE_VIEW_SURFACE_ID..], 8);
    zw.extend_from_slice(&[0u8; 12]);
    zw.extend_from_slice(&[
        0x42, 0x01, 0x0a, 0x00, 0, 0, 0, 0, 0x00, 0x04, 0, 0, 0x01, 0, 0, 0,
    ]);
    assert!(decode_iosurface_plane_view(&zw).is_none());
}

/// The probe's notion of "undecoded" must be exactly the bytes
/// `decode_surface_backing` skips, and it must distinguish two surfaces on
/// those bytes alone.
///
/// This is the measurement that blocks the largest deletion in the present
/// path: nothing decoded at surface-create time separates a desktop
/// swapchain buffer from a same-geometry offscreen tile, so membership is
/// reconstructed by half a dozen downstream mechanisms. If the guest is
/// telling us in the undecoded span, the probe has to be able to see it.
/// The two arms of the surface backing freshness test must accept exactly the same
/// backings, because only one of them rebuilds when it says no.
///
/// The force arm returns through `win_surface_backing_search` **without** calling
/// `apply_surface_backing`, so `set_mapping_geom` and
/// `synthesize_device_desc_from_surface_backing` are both skipped. It used to compare
/// width alone while the non-force arm compared width and height, and
/// `ensure_surface_for_present` calls the force arm precisely to catch a
/// wire geometry change — so a height change that stayed inside the same
/// page count left the mapping describing the previous incarnation, on the
/// path whose job was to notice.
///
/// Neither arm compared format, and a surface id recycled at identical
/// dimensions with a different pixel format keeps the old bytes-per-pixel
/// for every read window built over it.
#[test]
fn a_latched_backing_is_stale_when_any_of_geometry_or_format_moved() {
    use reims_vgpu_core::pixel_format::{MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA8_UNORM};
    let surf = |w: u32, h: u32, fourcc: u32| SurfaceBackingDescriptor {
        length: 0x1000,
        backing_pfn: 1,
        pixel_format: fourcc,
        plane_count: 1,
        planes: Default::default(),
        width: w,
        height: h,
        bytes_per_row: w * 4,
    };
    // 'BGRA' and 'RGBA' are distinct single-plane FourCCs at one bpp, so a
    // swap between them is invisible to a dimensions-only test.
    const BGRA: u32 = 0x4247_5241;
    const RGBA: u32 = 0x5247_4241;
    assert_eq!(
        latched_mapping_format(&surf(8, 4, BGRA)),
        MTL_FORMAT_BGRA8_UNORM
    );
    assert_eq!(
        latched_mapping_format(&surf(8, 4, RGBA)),
        MTL_FORMAT_RGBA8_UNORM
    );

    let m = SurfaceMappingEntry::default().with_geometry_for_test(8, 4, MTL_FORMAT_BGRA8_UNORM);
    assert!(backing_matches_latched_geom(&m, &surf(8, 4, BGRA)));
    assert!(
        !backing_matches_latched_geom(&m, &surf(8, 5, BGRA)),
        "a height change must be stale on both arms"
    );
    assert!(!backing_matches_latched_geom(&m, &surf(9, 4, BGRA)));
    assert!(
        !backing_matches_latched_geom(&m, &surf(8, 4, RGBA)),
        "same dimensions, different format: every read window's bpp comes from it"
    );
}

/// A multi-plane backing must compare equal to itself.
///
/// The latch stores `0` for it — the decoder's refusal to name a single
/// colour format — while the raw FourCC conversion may well return a real
/// format. A freshness test that compared the raw conversion would find
/// `0 != BGRA8` on every present and rebuild the backing forever, which is
/// the failure a shared `latched_mapping_format` exists to make impossible.
#[test]
fn a_multiplane_backing_compares_equal_to_the_zero_it_latched() {
    let mut surf = SurfaceBackingDescriptor {
        length: 0x1000,
        backing_pfn: 1,
        pixel_format: 0x4247_5241, // 'BGRA' — a format the converter knows
        plane_count: 2,
        planes: Default::default(),
        width: 8,
        height: 4,
        bytes_per_row: 32,
    };
    assert_ne!(
        iosurface_pixel_format_to_mtl(surf.pixel_format),
        0,
        "the fixture only means something if the raw conversion resolves"
    );
    assert_eq!(latched_mapping_format(&surf), 0, "multi-plane latches 0");

    let m = SurfaceMappingEntry::default().with_geometry_for_test(8, 4, 0);
    assert!(backing_matches_latched_geom(&m, &surf));
    // Dropping to one plane makes it a single-plane BGRA8 surface, which is
    // a real change of what the mapping describes.
    surf.plane_count = 1;
    assert!(!backing_matches_latched_geom(&m, &surf));
}

/// A single-plane surface must publish plane 0's offset, because both its
/// consumers fold it in and one of them is the other pathway.
///
/// `decode_surface_backing_plane` reads four fields; the surface-level convenience
/// copies on `SurfaceBackingDescriptor` take three, and the synthesizer's single-plane
/// arm used to publish only those three. A surface whose pixels start past
/// the base of its allocation was then read and written at 0 — the
/// multi-plane arm has always published each plane's offset, and
/// `sample_window_from_device_surface` treats `base_offset` exactly as
/// `sample_window_from_device_plane` treats a plane's.
#[test]
fn a_single_plane_backing_publishes_the_offset_its_pixels_start_at() {
    use reims_vgpu_core::pixel_format::MTL_FORMAT_BGRA8_UNORM;
    use reims_vgpu_protocol::{decode_device_surface, sample_window_from_device_desc};
    const BASE: u32 = 0x800;
    let (w, h, bpr) = (8u32, 4u32, 32u32);
    let mut surf = SurfaceBackingDescriptor {
        length: 0x4000,
        backing_pfn: 1,
        pixel_format: 0x4247_5241, // 'BGRA'
        plane_count: 1,
        planes: Default::default(),
        width: w,
        height: h,
        bytes_per_row: bpr,
    };
    surf.planes[0] = SurfaceBackingPlane {
        offset: BASE,
        width: w,
        height: h,
        bytes_per_row: bpr,
        bytes_per_element: 4,
    };
    assert!(
        !surface_backing_is_multiplanar(&surf),
        "the single-plane arm is the one under test"
    );

    let desc = synthesize_device_desc_from_surface_backing(&surf);
    let decoded = decode_device_surface(&desc).expect("device descriptor");
    assert_eq!(
        decoded.plane_count, 0,
        "single-plane publishes no plane records"
    );
    assert_eq!(decoded.base_offset, BASE);

    // The consumer, not just the field: the sample window must start at the
    // offset and its span must end past it, or publishing it bought nothing.
    let (off, got_bpr, end) =
        sample_window_from_device_desc(Some(&desc), None, MTL_FORMAT_BGRA8_UNORM, w, h)
            .expect("surface-level window");
    assert_eq!(off, BASE as u64);
    assert_eq!(got_bpr, bpr);
    assert_eq!(
        end,
        BASE as u64 + (h as u64 - 1) * bpr as u64 + (w as u64 * 4)
    );

    // Zero stays zero — the ordinary case must not gain an offset.
    surf.planes[0].offset = 0;
    let zero = synthesize_device_desc_from_surface_backing(&surf);
    assert_eq!(decode_device_surface(&zero).expect("desc").base_offset, 0);
}

/// The device descriptor's format word must survive both of the encodings
/// it is written in.
///
/// The x86 synthesizer writes an MTL ordinal for a known single-plane
/// surface and the raw OSType otherwise; the arm64 mapper reads the guest's
/// own descriptor, where media surfaces carry a FourCC. Narrowing with
/// `as u16` is correct for one of those and destroys the other — `'BGRA'`
/// becomes `0x5241`, which no format table accepts, so the mapping ends up
/// with a format that refuses every sample window and every render target.
#[test]
fn the_device_descriptor_format_word_survives_both_of_its_encodings() {
    use reims_vgpu_core::pixel_format::{
        bytes_per_pixel, MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_RGBA16_FLOAT,
    };

    const BGRA_FOURCC: u32 = 0x4247_5241;

    // The failure the narrowing produced, stated as the thing not to return.
    assert!(
        bytes_per_pixel((BGRA_FOURCC & 0xffff) as u16).is_none(),
        "the truncation's output is not a format, which is why it was a bug"
    );
    assert_eq!(
        device_desc_format_to_mtl(BGRA_FOURCC),
        MTL_FORMAT_BGRA8_UNORM
    );

    // An ordinal fits in the descriptor's own 16-bit format fields and is
    // passed through as itself — including one above the old 0x200
    // magnitude boundary, which is why the test is width and not size.
    assert_eq!(
        device_desc_format_to_mtl(MTL_FORMAT_BGRA8_UNORM as u32),
        MTL_FORMAT_BGRA8_UNORM
    );
    assert_eq!(
        device_desc_format_to_mtl(MTL_FORMAT_RGBA16_FLOAT as u32),
        MTL_FORMAT_RGBA16_FLOAT
    );
    // MTLPixelFormatBGRA10_XR is 552, above the 0x200 boundary an earlier
    // magnitude test used and which `iosurface_pixel_format_to_mtl` records
    // as having been wrong for exactly this format. It still fits in 16
    // bits, so the width test carries it where a size test did not.
    assert_eq!(device_desc_format_to_mtl(552), 552);

    // Fail closed, not BGRA8: a multi-plane OSType and an unknown one.
    assert_eq!(device_desc_format_to_mtl(IOSURFACE_FOURCC_420F), 0);
    assert_eq!(device_desc_format_to_mtl(0x5A5A_5A5A), 0);
    assert_eq!(device_desc_format_to_mtl(0), 0);
}

/// The surface backing probe order must visit task 0 first, the hint next, and every
/// **live** task exactly once.
///
/// It is the thing that makes the search terminate on the first probe for
/// every surface this device has ever resolved, so its shape is the whole
/// cost of the search. Two properties are load-bearing and neither is
/// obvious from the iterator chain: no task may be probed **twice** (a
/// duplicate is a wasted guest read on the hot present path), and no live task
/// may be **missed** (a missed one is a surface that cannot be found at all).
///
/// The tail used to be `1..MAX_TASKS`, and this test asserted a length of
/// exactly 256. It now walks the live ids, so the assertions are about the
/// task set rather than about a constant — which is the point of the change,
/// and is why a length assertion against a number would silently stop meaning
/// anything.
#[test]
fn the_surface_backing_probe_order_visits_task_zero_first_and_every_live_task_once() {
    use std::collections::HashSet;

    let mut state = Device::new(DeviceId(1), crate::model::PAGE_SHIFT_X86);
    // Deliberately sparse, and one id far past the retired 256 ceiling: the
    // probe must reach a task the old fixed range could not even name.
    let live = [0u32, 1, 7, 300, 70_000];
    for id in live {
        state.define_task(id, 0x1000, 2);
    }

    for hint in [0u32, 1, 7, 70_000] {
        let order = surface_backing_probe_order(&state.tasks, hint);
        assert_eq!(order[0], 0, "task 0 leads for hint {hint}");
        if hint != 0 {
            assert_eq!(order[1], hint, "the hint is probed second");
        }
        let seen: HashSet<u32> = order.iter().copied().collect();
        assert_eq!(
            seen.len(),
            order.len(),
            "no task probed twice for hint {hint}: {order:?}"
        );
        assert!(
            live.iter().all(|t| seen.contains(t)),
            "every live task probed for hint {hint}: {order:?}"
        );
    }

    // A dead task is not probed. It never could be — the probe's own liveness
    // test refused it — so yielding it only ever cost a guest read's worth of
    // work per present.
    let order = surface_backing_probe_order(&state.tasks, 0);
    assert!(
        !order.contains(&9),
        "an id nothing defined must not be probed: {order:?}"
    );

    // A hint naming no live task adds a probe that the liveness test at the
    // probe then refuses, and must not lose a live one or duplicate task 0.
    let order = surface_backing_probe_order(&state.tasks, u32::MAX);
    let seen: HashSet<u32> = order.iter().copied().collect();
    assert_eq!(seen.len(), order.len(), "{order:?}");
    assert!(live.iter().all(|t| seen.contains(t)), "{order:?}");
}

#[test]
fn undecoded_surface_backing_span_is_exactly_what_the_decoder_skips() {
    // One plane: the decoder consumes 0x14..0x24, so the tail starts there.
    let built = SurfaceBackingBuilder::new(0x800000, 0x1234, 0x4247_5241, 1) // 'BGRA'
        .plane(0, 0, 1920, 1080, 1920 * 4, 0)
        .with_len(0x40);
    let a = built.bytes().to_vec();

    // Every decoded field can change without moving the undecoded span.
    let b = SurfaceBackingBuilder::new(0x900000, 0x9999, 0x4c31_3062, 1)
        .plane(0, 0, 1280, 720, 1280 * 4, 0)
        .with_len(0x40);
    assert_eq!(
        undecoded_surface_backing_bytes(&a),
        undecoded_surface_backing_bytes(b.bytes()),
        "changing only decoded fields must not look like a new shape"
    );

    // The span covers the three bytes after plane_count and the whole tail
    // past the plane records the decoder consumed.
    for probe in [0x11usize, 0x13, 0x24, 0x3f] {
        let mut c = a.clone();
        c[probe] ^= 0xff;
        assert_ne!(
            undecoded_surface_backing_bytes(&a),
            undecoded_surface_backing_bytes(&c),
            "byte {probe:#x} is undecoded and must be visible to the probe"
        );
    }

    // Bytes the decoder DOES read must not be in the span, or ordinary
    // surface-to-surface variation would look like a new shape forever.
    // `plane_count` (+0x10) is excluded on purpose: it is decoded AND it
    // moves the span's own boundary, which the two-plane case below pins.
    for probe in [0x00usize, 0x08, 0x0c, 0x14, 0x23] {
        let mut c = a.clone();
        c[probe] ^= 0xff;
        assert_eq!(
            undecoded_surface_backing_bytes(&a),
            undecoded_surface_backing_bytes(&c),
            "byte {probe:#x} is decoded and must stay out of the span"
        );
    }

    // A second plane moves the boundary: 0x24..0x34 becomes decoded.
    let two = SurfaceBackingBuilder::new(0x800000, 0x1234, 0x4247_5241, 2)
        .plane(0, 0, 1920, 1080, 1920 * 4, 0)
        .with_len(0x40);
    assert_eq!(
        undecoded_surface_backing_bytes(two.bytes()).len(),
        undecoded_surface_backing_bytes(&a).len() - SURFACE_BACKING_PLANE_STRIDE,
        "the span shrinks by exactly one plane record"
    );

    // A record too short to decode reports nothing rather than a partial
    // span that would compare unequal against every real one.
    assert!(undecoded_surface_backing_bytes(&a[..SURFACE_BACKING_MIN_LEN - 1]).is_empty());
}

/// The shared ladder asks its three questions in the only order they can be
/// asked, and names which one refused.
///
/// The order is the point, not an implementation detail: a type tag cannot be
/// checked before the entry is found, and a descriptor cannot be read before the
/// entry says where it is. Twenty rails wrote this out by hand and any of them
/// could have reordered or dropped a rung without the compiler noticing — which
/// is what [`LadderRung`] being a value rather than three separate `else`
/// branches removes.
#[test]
fn the_shared_ladder_names_the_rung_that_refused() {
    use crate::runtime::decode::resource::OBJECT_TYPE_IOSURFACE;

    let mut host = FakeHost::new();
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    setup_task_with_list(&mut host, &mut state);

    // Ref 1 is an IOSurface texture entry whose descriptor is mapped: all three rungs pass.
    let (entry, bytes) = resolve_descriptor(&state, &host, 1, 1, &[ObjectKind::IOSurfaceTexture])
        .expect("all rungs pass");
    assert_eq!(entry.kind, ObjectKind::IOSurfaceTexture);
    assert!(!bytes.is_empty(), "the descriptor bytes come back with it");

    // Same ref, asked for as a buffer: the tag it found travels with the
    // refusal, so a rail no longer re-formats `ot=` from an entry it has
    // already dropped.
    assert_eq!(
        resolve_descriptor(&state, &host, 1, 1, &[ObjectKind::Buffer]),
        Err(LadderRung::WrongType {
            got: ObjectKind::IOSurfaceTexture
        })
    );

    // A ref past the end of the list. Asked for *no* acceptable type at all, so
    // a resolver that checked the tag first would have to answer `WrongType`;
    // answering `NoListEntry` is what proves the lookup runs first.
    assert_eq!(
        resolve_descriptor(&state, &host, 1, 9999, &[]),
        Err(LadderRung::NoListEntry)
    );

    // An entry whose descriptor GVA is not mapped: found, right type, unreadable
    // — the rung that separates "the guest never registered this" from "the
    // guest registered it and its descriptor is not resident right now".
    let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
    let mut entry = [0u8; 12];
    st32(
        &mut entry[0..],
        u32::from(OBJECT_TYPE_IOSURFACE) | (0x20u32 << 8),
    );
    entry[4..12].copy_from_slice(&0xdead_0000u64.to_le_bytes());
    let _ = host.write_gpa(data_gpa + 24, &entry);
    // The declared length travels with the rung: the entry above says 0x20
    // bytes, and by the time a rail reports this the entry is gone.
    assert_eq!(
        resolve_descriptor(&state, &host, 1, 2, &[ObjectKind::IOSurfaceTexture]),
        Err(LadderRung::DescRead { declared_len: 0x20 })
    );
}

/// A re-point over a ref-keyed host copy drops that copy, so the next resolve
/// reads the pages the guest just rewired instead of bytes read from the old
/// ones.
///
/// `ReplacePhysical` says by its own contract that the PFNs under this object
/// have changed. The mapping rail discharges that through
/// `invalidate_mapping_pages`, but `host_texture_surfaces` and
/// `host_linear_textures` are keyed by object-list ref and carry no page list,
/// so nothing in them can notice — and this device holds a copy under exactly
/// those keys for resources that own no mapping. Measured on a driven x86/PCI boot under
/// `web-content-probe`: 7 texture and 1 linear against 32 that held nothing, so
/// the guest was being served stale content on an ordinary browsing workload.
///
/// Fails without the fix: both entries survive the packet.
#[test]
fn a_repoint_drops_the_ref_keyed_host_copies_of_the_object() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let (task, object) = (7u32, 4242u32);

    state.host_replicas.texture_surfaces.insert(
        (task, object),
        crate::model::HostSurface {
            width: 4,
            height: 4,
            bgra: std::sync::Arc::new(vec![0xAB; 4 * 4 * 4]),
            host_gen: 1,
            producer_object_type: 0,
            last_touch: 0,
            backing: None,
            guest_holds_bytes: false,
            source_gva: 0,
        },
    );
    state.host_replicas.texture_surfaces.insert(
        (task + 1, object),
        crate::model::HostSurface {
            width: 4,
            height: 4,
            bgra: std::sync::Arc::new(vec![0xEF; 4 * 4 * 4]),
            host_gen: 2,
            producer_object_type: 0,
            last_touch: 0,
            backing: None,
            guest_holds_bytes: false,
            source_gva: 0,
        },
    );
    state.host_replicas.linear_textures.insert(
        (task, object),
        crate::model::HostLinearTexture {
            gva: 0x1000,
            pixel_format: 0,
            width: 4,
            height: 4,
            row_stride: 16,
            bytes: vec![0xCD; 64],
            host_gen: 1,
            resident_gen: 0,
        },
    );
    // No mapping owns the id, which is the route this covers: three quarters of
    // the re-points on a driven boot take it.
    assert!(!state.surfaces.mappings.contains_key(&object));

    super::replace_physical(&mut state, &mut host, task, object);

    assert!(
        !state
            .host_replicas
            .texture_surfaces
            .contains_key(&(task, object)),
        "the ref-keyed texture copy was read from pages the guest has re-pointed"
    );
    assert!(
        state
            .host_replicas
            .texture_surfaces
            .contains_key(&(task + 1, object)),
        "a re-point must not evict another task's same-numbered texture copy"
    );
    assert!(
        !state
            .host_replicas
            .linear_textures
            .contains_key(&(task, object)),
        "the ref-keyed linear copy was read from pages the guest has re-pointed"
    );
}

/// ReplacePhysical's object id is local to the task carried beside it. A
/// mapping id is a different namespace even when the integers happen to be
/// equal.
///
/// This is the compositor failure class: task 0 owns surface backing surface 1 while
/// task 1 owns IOSurface texture resource 1, which resolves to mapping 9. Re-pointing the
/// latter must retire mapping 9 and leave task 0's surface intact. The old
/// global-id-first route did the opposite.
#[test]
fn a_repoint_resolves_the_resource_in_its_task_before_touching_a_mapping() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
    let mut host = FakeHost::new();
    setup_task_with_list(&mut host, &mut state);

    assert_eq!(
        resolve_iosurface_texture_ref(&mut state, &host, 1, 1),
        Some(9)
    );

    assert!(state.map_surface(1));
    {
        let surface = state
            .surfaces
            .mappings
            .get_mut(&1)
            .expect("surface mapping");
        surface.lifecycle.active = true;
        surface.pages.entries = vec![0x1234_5001];
        surface.pages.surface_walk = Some(crate::model::SurfaceBackingWalk {
            task_id: 0,
            backing_pfn: 0x20,
            page_generation: surface.pages.generation,
        });
    }
    {
        let resource = state
            .surfaces
            .mappings
            .get_mut(&9)
            .expect("IOSurface texture mapping");
        resource.lifecycle.active = true;
        resource.pages.entries = vec![0x6789_a001];
    }

    super::replace_physical(&mut state, &mut host, 1, 1);

    assert_eq!(
        state.surfaces.mappings[&1].pages.entries,
        vec![0x1234_5001],
        "a same-number resource in another task does not own this surface"
    );
    assert!(
        state.surfaces.mappings[&9].pages.entries.is_empty(),
        "the task-local IOSurface texture association names the mapping to invalidate"
    );
}

/// A direct surface backing resource is routed by the task provenance latched with its
/// page walk, so tightening the namespace must not suppress genuine surface
/// re-points.
#[test]
fn a_repoint_retires_a_surface_backing_mapping_owned_by_the_packet_task() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    assert!(state.map_surface(7));
    let prior_generation = {
        let surface = state
            .surfaces
            .mappings
            .get_mut(&7)
            .expect("surface mapping");
        surface.lifecycle.active = true;
        surface.pages.entries = vec![0x1234_5001];
        surface.pages.surface_walk = Some(crate::model::SurfaceBackingWalk {
            task_id: 3,
            backing_pfn: 0x20,
            page_generation: surface.pages.generation,
        });
        surface.lifecycle.generation
    };

    super::replace_physical(&mut state, &mut host, 3, 7);

    assert!(state.surfaces.mappings[&7].pages.entries.is_empty());
    assert_ne!(
        state.surfaces.mappings[&7].lifecycle.generation,
        prior_generation
    );
    assert_ne!(
        state.surfaces.mappings[&7]
            .pages
            .surface_walk
            .expect("the old walk remains only as provenance")
            .page_generation,
        state.surfaces.mappings[&7].pages.generation,
        "the generation bump makes the retired walk unusable as currency"
    );
}

/// A re-point that reaches nothing changes nothing, and does not invent a
/// removal for a neighbouring ref.
///
/// The counterpart to the test above and the reason the repair is keyed on
/// `(task, ref)` rather than on the ref alone for the linear map: 32 of 40
/// re-points on the measured boot held no state at all, and one that started
/// evicting its neighbours would turn a benign majority into a new loss.
#[test]
fn a_repoint_of_an_object_this_device_holds_nothing_for_touches_no_neighbour() {
    let mut state = Device::new(DeviceId(1), PAGE_SHIFT_X86);
    let mut host = FakeHost::new();
    let (task, object, neighbour) = (7u32, 4242u32, 4243u32);

    state.host_replicas.linear_textures.insert(
        (task, neighbour),
        crate::model::HostLinearTexture {
            gva: 0x1000,
            pixel_format: 0,
            width: 4,
            height: 4,
            row_stride: 16,
            bytes: vec![0xCD; 64],
            host_gen: 1,
            resident_gen: 0,
        },
    );
    // The same ref under a different task must also survive.
    state.host_replicas.linear_textures.insert(
        (task + 1, object),
        crate::model::HostLinearTexture {
            gva: 0x2000,
            pixel_format: 0,
            width: 4,
            height: 4,
            row_stride: 16,
            bytes: vec![0xEF; 64],
            host_gen: 1,
            resident_gen: 0,
        },
    );

    super::replace_physical(&mut state, &mut host, task, object);

    assert!(
        state
            .host_replicas
            .linear_textures
            .contains_key(&(task, neighbour)),
        "a different ref in the same task is a different object"
    );
    assert!(
        state
            .host_replicas
            .linear_textures
            .contains_key(&(task + 1, object)),
        "the same ref in a different task is a different object"
    );
}

/// Each way an object-list lookup comes back empty gets its own route.
///
/// The whole reason [`super::ListMiss`] exists is that eight causes shared one
/// `reason=no_list_entry`, so a boot losing draws could not say whether this
/// device had cleared a task's list under the guest or the guest had not
/// published the object yet. Two variants sharing a route string — the obvious
/// copy-paste when a ninth is added — rebuilds exactly that, and rebuilds it
/// invisibly, because a merged population still reads as a clean count.
#[test]
fn every_object_list_miss_names_a_different_check() {
    let routes: Vec<&'static str> = super::ListMiss::ALL.iter().map(|m| m.route()).collect();
    let mut unique = routes.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(
        unique.len(),
        routes.len(),
        "two object-list misses share a route, so their counts add up as one: {routes:?}"
    );
    assert!(
        routes.iter().all(|r| r.starts_with("list_miss_")),
        "the family shares a prefix so a boot can rank it in one grep: {routes:?}"
    );
}

/// The claimant banding must separate a real ownership signal from the confound
/// that nearly buried it.
///
/// Every task registers its object list at the same `pfn = 1`, so on a busy
/// guest "some other task has something at slot 3" is close to a tautology. The
/// first version of this instrument was a yes/no and answered yes to every miss
/// on macos-26, which reads as a finding and is not one. The band against the
/// live task count is what makes the difference visible, so each boundary is
/// pinned here:
///
/// - nobody has it — the guest has not published it, and the fix is to wait;
/// - exactly one other task has it — a real ownership signal, the object is in
///   a list this device did not look in;
/// - all of the others have it — the slot index is just populated everywhere and
///   this search cannot tell ownership from coincidence.
///
/// The asking task is excluded from the count, so "all" must compare against
/// `live - 1`. Comparing against `live` would make "all" unreachable and silently
/// demote every genuine all-claim to "many".
#[test]
fn a_claimant_count_is_banded_against_the_tasks_that_could_have_claimed() {
    use super::slot_empty_claim_route as band;

    assert_eq!(band(0, 8), "list_miss_slot_empty_claimed_nowhere");
    assert_eq!(band(1, 8), "list_miss_slot_empty_claimed_by_one");
    assert_eq!(band(4, 8), "list_miss_slot_empty_claimed_by_many");
    assert_eq!(
        band(7, 8),
        "list_miss_slot_empty_claimed_by_all",
        "seven others out of eight live tasks is every task that could have claimed"
    );

    // Two tasks total: the one asking and one other. That other claiming is
    // both "one" and "all", and "one" is the reading that matters — it is the
    // ownership signal, while "all" only ever means the search is uninformative.
    assert_eq!(band(1, 2), "list_miss_slot_empty_claimed_by_one");

    // A single live task has nobody else to claim, and must not be reported as
    // a unanimous claim over an empty population.
    assert_eq!(band(0, 1), "list_miss_slot_empty_claimed_nowhere");
    assert_eq!(band(0, 0), "list_miss_slot_empty_claimed_nowhere");
}
