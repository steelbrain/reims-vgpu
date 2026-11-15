//! `id<MTLDevice>`: core's `GpuDevice` over the runtime's `Device`.
//!
//! - **One object per device.** Every handle to one runtime device is a reference to one
//!   [`DeviceObject`] ([`adopt`]), so a device opened twice, and the `device` of every object made
//!   from it, have the same identity, as one `MTLDevice` does. The objects made from a device hold
//!   its handle.
//! - **Capabilities** are read once per device object ([`crate::gpu::caps`]) and shared.
//! - **The registry ID** is Metal's default device's for the process's default device
//!   (`crate::platform::default_device`),
//!   asked of Metal at run time on macOS ([`registry_id`]), which is what a host names it by; the
//!   runtime's own on Linux.

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::{Arc, OnceLock};

use crate::core::host::HostPtr;
use crate::core::platform::caps::{DeviceCaps, IdentityCaps};
use crate::core::platform::device::{GpuDevice, NoCopyDeallocator};
use crate::core::platform::os::vm::HostVm;
use crate::core::platform::types::{
    ArgumentKind, ArgumentLayout, ArgumentMember, NsErrorText, PixelFormatInfo, SizeAndAlign,
    TextureLevelInfo, TextureLevelQuery,
};
use crate::core::platform::values::{
    ArgumentDesc, BufferDeallocator, BufferDesc, DepthStencilDesc, FullComputePipelineDesc,
    FullMeshRenderPipelineDesc, FullRenderPipelineDesc, HeapDesc, HeapMade, IcbDesc, OptionSet,
    PagesInPlace, PagesPlane, RenderPipelineDesc, SamplerDesc,
};
use crate::core::platform::{Handle, HostRange};
use crate::core::raise::{Kept, OnFault, RaiseResult, keep_forever};
use crate::core::rt::BlockEnv;
use crate::core::sync::Mutex;
use metal2vulkan::pipeline::ComputePipelineDescriptor;
use metal2vulkan_runtime as rt;

use crate::answer::{
    Nil, Policy, create, created_or_decided, decide, error_text, get, get_describing, hearing,
    metal_aborts, plain, plain_at, relay_gap, send, unrepresentable,
};
use crate::gpu::backing::{self, Backing};
use crate::gpu::caps;
use crate::gpu::command::{QueueObject, VulkanCommandQueue};
use crate::gpu::desc::{
    Maker, TextureFields, VulkanTextureDescriptor, created_format_refused, format_refused,
};
use crate::gpu::event::{VulkanEvent, VulkanSharedEvent};
use crate::gpu::icb::VulkanIcb;
use crate::gpu::metal_space::{self, PinRefused, Placing, TextureKind};
use crate::gpu::rate_map::{VulkanRateMap, VulkanRateMapDescriptor};
use crate::gpu::resource::{
    BufferObject, ComputePipelineObject, HeapObject, LibraryObject, OverSurface, TextureObject,
    Under, VulkanBuffer, VulkanComputePipeline, VulkanDepthStencilState, VulkanFence,
    VulkanFunction, VulkanHeap, VulkanLibrary, VulkanRenderPipeline, VulkanSamplerState,
    VulkanSharedTextureHandle, VulkanTexture,
};
use crate::gpu::timing;
use crate::os::OsVm;
use crate::platform::{Obj, VulkanPlatform};

/// Core's layout of the runtime's `layout`, nested layouts with it.
fn layout_of(layout: &metal2vulkan::bind::ArgumentBufferLayout) -> ArgumentLayout {
    ArgumentLayout {
        encoded_length: layout.encoded_length,
        alignment: layout.alignment,
        members: layout
            .members
            .iter()
            .map(|m| ArgumentMember {
                id: m.id,
                count: m.count,
                ids_per_element: m.ids_per_element,
                offset: m.offset,
                stride: m.stride,
                size: m.size,
                kind: argument_kind(m.ty),
                layout: m.layout.as_deref().map(|l| Box::new(layout_of(l))),
            })
            .collect(),
    }
}

/// Core's kind of the runtime's argument type.
fn argument_kind(ty: metal2vulkan::bind::ArgumentType) -> ArgumentKind {
    use metal2vulkan::bind::ArgumentType as T;
    match ty {
        T::Buffer => ArgumentKind::Buffer,
        T::Texture => ArgumentKind::Texture,
        T::Sampler => ArgumentKind::Sampler,
        T::Constant => ArgumentKind::Constant,
        T::IndirectCommandBuffer => ArgumentKind::IndirectCommandBuffer,
        T::ComputePipelineState => ArgumentKind::ComputePipelineState,
        T::RenderPipelineState => ArgumentKind::RenderPipelineState,
        T::VisibleFunctionTable => ArgumentKind::VisibleFunctionTable,
        T::IntersectionFunctionTable => ArgumentKind::IntersectionFunctionTable,
        T::AccelerationStructure => ArgumentKind::AccelerationStructure,
        T::DepthStencilState => ArgumentKind::DepthStencilState,
        T::Struct => ArgumentKind::Struct,
        _ => ArgumentKind::Constant,
    }
}

/// A runtime device, as core holds it.
#[derive(Debug)]
pub struct DeviceObject {
    /// The runtime's device.
    pub raw: rt::Device,
    /// The registry ID this platform answers for it ([`registry_id`]).
    pub registry_id: u64,
    /// Its capabilities, once read.
    caps: OnceLock<Arc<DeviceCaps>>,
    /// Its keeping awake while the guest draws ([`crate::awake`]).
    pub(crate) awake: crate::awake::Awake,
    /// Its other threads of the library's that hold its runtime device ([`crate::let_go`]).
    pub(crate) threads: crate::let_go::DeviceThreads,
    /// Its answers of [`GpuDevice::linear_rows_pitch`] so far, by row bytes and rows.
    pub(crate) rows_pitches: Mutex<BTreeMap<(u64, u64), Option<u64>>>,
    /// The guest's texture resource IDs it reserved as it was adopted, where the process declared
    /// them first (`crate::platform::present_guest_texture_ids`).
    guest_texture_ids: Option<std::ops::RangeInclusive<u64>>,
}

impl Drop for DeviceObject {
    /// Ends the device's threads of the library's and waits for them, before its runtime device's
    /// handle goes: none holds that device past it, so where this holds its last handle, the
    /// runtime's teardown runs here ([`crate::let_go`]).
    fn drop(&mut self) {
        self.awake.stop();
        self.threads.end();
    }
}

/// How many uncompleted command buffers a queue of `newCommandQueue` holds at most, as Metal's
/// and the runtime's `new_command_queue`'s.
const METAL_MAX_COMMAND_BUFFER_COUNT: usize = 64;

/// The runtime's queue of at most `count` uncompleted command buffers for a command queue the
/// graphics lane's kext asks unshared (`RGFX_QUEUE_UNSHARED`, WindowServer's queues): the
/// runtime's `new_command_queue_unshared`, whose `VkQueue` no other command queue shares where the
/// runtime shares `VkQueue`s, so no other process's long command buffer queues its frames on the
/// GPU. On a device that shares none it is the queue every create makes. Its refusals are the
/// runtime's: a `count` of 0 (`Invalid`, `MaxCommandBufferCount`) and a lost device
/// (`DeviceLost`).
pub(crate) fn unshared_queue(
    raw: &rt::Device,
    count: usize,
) -> Result<rt::CommandQueue, rt::Error> {
    raw.new_command_queue_unshared(count)
}

/// The registry ID this platform answers for `raw`: for the process's default device
/// ([`crate::platform::default_device`]),
/// Metal's for its system default device, asked at run time on macOS
/// ([`crate::os::default_registry_id`]), and the runtime's own for any other device, or where
/// there is no Metal default device.
pub fn registry_id(raw: &rt::Device) -> u64 {
    if crate::platform::default_device_registry_id() == Some(raw.registry_id())
        && let Some(id) = crate::os::default_registry_id()
    {
        return id;
    }
    raw.registry_id()
}

/// `id<MTLDevice>`.
pub type VulkanDevice = Obj<DeviceObject>;

/// The `NSError` Metal hands back for bytes that are no metallib (measured).
fn invalid_library_file() -> NsErrorText {
    NsErrorText {
        domain: "MTLLibraryErrorDomain".to_owned(),
        code: 1,
        description: "Error Domain=MTLLibraryErrorDomain Code=1 \"Invalid library file\" \
                      UserInfo={NSLocalizedDescription=Invalid library file}"
            .to_owned(),
    }
}

/// The runtime's refusal `e` of the pipeline creator `sel`: the `NSError` where Metal answers one
/// ([`super::states::pipeline_error`]), the runtime's refusal as the `NSError` for a gap of its
/// own or the device's, relayed by default ([`relay_gap`]), else a death at the send, whose line
/// names `descriptor`'s formats ([`super::states::compute_formats`]). An unspecialized function
/// constant is a death, as Metal aborts for one.
fn compile_error(
    sel: &'static str,
    e: rt::Error,
    descriptor: Option<&ComputePipelineDescriptor>,
) -> NsErrorText {
    super::states::pipeline_error(&e).unwrap_or_else(|| {
        let formats = descriptor.map(super::states::compute_formats);
        match relay_gap(sel, &e, formats.as_deref()) {
            true => error_text(&e),
            // Decided as a getter's refusal: none of them returns.
            false => get_describing(sel, formats.as_deref(), || Err(e)),
        }
    })
}

/// The devices made in the process, by registry ID, each the one object every handle to it is.
/// Each is kept for the process, as Metal's device is: a device the host releases and opens again
/// is the same runtime device, whose spaces of IDs (textures', sampler states', pipeline states'
/// and indirect command buffers') live on, as Metal's do for the process. A runtime device whose
/// last handle dropped would tear down, and one opened again would number from a fresh pool where
/// Metal's and [`crate::gpu::icb`]'s go on. A device the runtime lost stays lost, kept, where one
/// made again would not be; a call the runtime refuses as lost dies at the send anyway
/// ([`crate::answer`]). The platform's let-go of its devices ([`crate::let_go`]), once nothing
/// else holds one (the C API's release of the process's last graphics device), takes them all:
/// a device adopted after it is a new runtime device.
static DEVICES: Mutex<Vec<(u64, VulkanDevice)>> = Mutex::new(Vec::new());

/// The runtime device of the first device the process adopted, if it adopted one.
pub(crate) fn first_adopted() -> Option<rt::Device> {
    let devices = DEVICES.lock();
    devices.first().map(|(_, device)| device.raw.clone())
}

/// Takes every device adopted ([`DEVICES`]) and the libraries of the platform's own shaders loaded
/// of them (`gpu::shaders`), for the platform's let-go of its devices ([`crate::let_go`]), where no
/// handle beyond those holds one: `Err` naming each device a handle beyond them holds, and how
/// many, nothing taken. Under [`DEVICES`]' lock, so no handle is made from it meanwhile.
#[allow(clippy::type_complexity)]
pub(crate) fn take_if_only_kept()
-> Result<(Vec<(u64, VulkanDevice)>, Vec<(u64, VulkanLibrary)>), String> {
    let mut devices = DEVICES.lock();
    let held: Vec<String> = (devices.iter())
        .filter_map(|(id, device)| {
            let (libraries, library_held) = crate::gpu::shaders::kept_of(device);
            let beyond = device.strong_count().saturating_sub(1 + libraries);
            (beyond > 0 || library_held).then(|| {
                let library = if library_held {
                    ", and a library of the platform's shaders of it"
                } else {
                    ""
                };
                format!(
                    "{} (registry ID {id:#x}) by {beyond} handles beyond the platform's{library}",
                    device.raw.properties().name
                )
            })
        })
        .collect();
    if !held.is_empty() {
        return Err(held.join("; "));
    }
    let libraries = crate::gpu::shaders::take_loaded();
    Ok((std::mem::take(&mut *devices), libraries))
}

/// The handle of `raw`, a device the runtime made: the object of the same device made before, or a
/// new one, kept. A new device first reserves the guest's texture resource IDs where the process
/// declared them ([`reserve_guest_texture_ids`]), before any work of the library's below makes a
/// command queue or a texture on it. The process's first device's sampler ID pool is the one
/// Metal's sampler mempool follows ([`metal_space::count_sampler_ids`]). A new device's class joins
/// the process's placement of degradations ([`crate::answer::native::device_made`]), and its warm
/// pass runs over the pipeline records earlier processes kept ([`crate::records`]), in the
/// background, but in a test build. Its work counts go to the log where its filter names their
/// target ([`crate::work_log`]), its GPU is kept awake while the guest draws ([`crate::awake`]),
/// and the runtime's check of textures over host pages starts ([`start_in_place_checks`]). The exit
/// gate's hook is registered, after the device's driver was loaded ([`crate::exit_gate`]): every
/// process that holds a device of the process's, a test's or a host's, waits at its exit for the
/// library's GPU work in progress.
pub fn adopt(raw: rt::Device) -> VulkanDevice {
    let id = registry_id(&raw);
    let mut devices = DEVICES.lock();
    if let Some((_, kept)) = devices.iter().find(|(at, _)| *at == id) {
        return kept.clone_ref();
    }
    let guest_texture_ids = reserve_guest_texture_ids(&raw);
    crate::exit_gate::arm();
    metal_space::count_sampler_ids(&raw);
    crate::answer::native::device_made(raw.properties());
    #[cfg(target_os = "linux")]
    crate::os::learn_view_imports(&raw);
    #[cfg(windows)]
    crate::os::hand_ram_to(&raw);
    let mut threads = crate::let_go::DeviceThreads::default();
    if crate::records::KEPT
        && let Some(home) = std::env::var_os("HOME")
    {
        let path = crate::records::path_in(home.as_ref());
        threads.keep(crate::records::warm(&raw, path, &threads.stop));
    }
    threads.keep(crate::work_log::start(&raw, &threads.stop));
    start_in_place_checks(&raw);
    let made = Obj::new(DeviceObject {
        awake: crate::awake::Awake::start(&raw),
        raw,
        registry_id: id,
        caps: OnceLock::new(),
        threads,
        rows_pitches: Mutex::default(),
        guest_texture_ids,
    });
    devices.push((id, made.clone_ref()));
    made
}

/// A device of a window's own over `raw`, a runtime device of another GPU than the process's
/// default device's (`VulkanPlatform::window_device`): not adopted, so no open of the process
/// answers it, and none of what an adopted device starts runs on it (the guest's texture IDs,
/// the answers of the process's placement, the imports' probe, its records, work log and checks
/// of textures over host pages). The window that made it draws on it alone, and it goes with
/// its last handle.
pub fn window_device(raw: rt::Device) -> VulkanDevice {
    let id = raw.registry_id();
    Obj::new(DeviceObject {
        awake: crate::awake::Awake::start(&raw),
        raw,
        registry_id: id,
        caps: OnceLock::new(),
        threads: crate::let_go::DeviceThreads::default(),
        rows_pitches: Mutex::default(),
        guest_texture_ids: None,
    })
}

/// Reserves the guest's texture resource IDs on `raw` where the process declared them
/// (`crate::platform::present_guest_texture_ids`), the device's first call, before the library's
/// own work on it makes a command queue or a texture ([`adopt`]): the IDs it reserved, or `None`
/// where none were declared or the runtime refused them, which is logged at error.
fn reserve_guest_texture_ids(raw: &rt::Device) -> Option<std::ops::RangeInclusive<u64>> {
    let ids = crate::platform::guest_texture_ids()?;
    let reserved: Vec<rt::ResourceId> = ids.clone().map(rt::ResourceId).collect();
    match raw.reserve_resource_ids(rt::ResourceIdKind::Texture, &reserved) {
        Ok(()) => Some(ids),
        Err(error) => {
            log::error!(
                "the guest's texture resource ids {} to {} are not reserved as the device is \
                 made ({error:?}): its textures may share ids with the host's",
                ids.start(),
                ids.end()
            );
            None
        }
    }
}

/// Starts `raw`'s check of textures over host pages on the runtime's thread for it, where its
/// verdicts are pending: where no record of an earlier process of the same driver and runtime
/// holds them, and the policy takes such textures in place (`REIMS_VGPU_EXACT` names no
/// `host_page_textures`). Waits for nothing. A texture the guest makes over its pages (a
/// framebuffer of its compositor's) before its class's verdict is known is a shadow, copied in
/// before each use and merged back after each write, where the verdict would have the texture in
/// place (the runtime's `in_place.rs`); the check otherwise starts at the first such texture,
/// which is then a shadow. The check's buffers lie outside Metal's address space and its textures
/// take no resource ID (the runtime's `in_place_check.rs`), so no value the guest sees moves with
/// it.
fn start_in_place_checks(raw: &rt::Device) {
    let _ = raw.settle_in_place_checks(std::time::Duration::ZERO);
}

/// The handle of the device made in the process whose registry ID, as this platform answers it
/// ([`registry_id`]), is `registry_id`, where one was made ([`DEVICES`]): the object [`adopt`]
/// answers for that device again.
pub fn kept(registry_id: u64) -> Option<VulkanDevice> {
    let devices = DEVICES.lock();
    let (_, device) = devices.iter().find(|(at, _)| *at == registry_id)?;
    Some(device.clone_ref())
}

/// Tells every device made in the process that the pages under the `length` host bytes at
/// `bytes` were replaced (`rt::Device::host_range_remapped`, [`crate::remap`]): a no-copy buffer
/// made over any of them after this imports the pages there then. Callable from any thread, a no-copy buffer's deallocator among
/// them: the runtime's lock under this one is a leaf. A null address or a length of 0 tells
/// nothing.
pub(crate) fn host_range_remapped(bytes: HostPtr, length: u64) {
    let Some(bytes) = NonNull::new(bytes.as_ptr::<u8>()) else {
        return;
    };
    if length == 0 {
        return;
    }
    for (_, device) in DEVICES.lock().iter() {
        device.raw.host_range_remapped(bytes, length);
    }
}

/// A deallocator of core's the runtime may run, once, or drop unrun when it refuses the buffer:
/// whichever comes first takes it from the slot.
type Slot<D> = Arc<Mutex<Option<D>>>;

/// The runtime's deallocator over `slot`: it takes core's and runs it, marked ours, with the
/// buffer's pointer and length.
fn runtime_deallocator<D: Send + 'static>(
    slot: &Slot<D>,
    run: impl FnOnce(D, NonNull<u8>, u64) + Send + 'static,
) -> rt::Deallocator {
    let slot = Arc::clone(slot);
    Box::new(move |pointer, length| {
        // A callout on the runtime's thread: marked ours, so a panic here aborts.
        let _env = BlockEnv::foreign();
        if let Some(d) = slot.lock().take() {
            run(d, pointer, length);
        }
    })
}

/// Core's deallocator of a buffer the runtime did not make: never run, and its captures kept for
/// good (what they hold stays mapped).
fn keep_unrun<D, T>(made: &RaiseResult<Option<T>>, slot: Option<Slot<D>>) {
    if !matches!(made, Ok(Some(_)))
        && let Some(d) = slot.and_then(|slot| slot.lock().take())
    {
        keep_forever(d, Kept::AddressSpace);
    }
}

/// What the runtime is given for a no-copy buffer over the `length` bytes at `start`, which
/// core's `HostRange` vouches for, with `run`, the runtime's form of core's deallocator: the first
/// byte it imports (the bytes, read-only ones too, which the runtime copies,
/// [`crate::os::import_view`]), the deallocator,
/// which runs at Metal's moment, and the keep-alive, which the
/// runtime drops once it freed its import: it holds what the view keeps of the pages
/// (`os::ImportView::release`), and an unmap hold on the bytes' pages, so that core's
/// unmap of its mapping of them through the platform's VM, which its deallocator may make, waits
/// for the free ([`crate::remap::UnmapHold`]); whether the pages are mapped read-only
/// ([`Backing::imported`]); whether the runtime may be told the pages are writable
/// (`os::ImportView::vouched`); and the word that the import was made, which the caller
/// gives once the runtime's call returned ([`crate::os::ImportMade`]).
fn no_copy_release(
    start: NonNull<u8>,
    length: u64,
    run: Option<rt::Deallocator>,
    site: backing::ImportSite,
) -> NoCopyImport {
    let view = crate::os::import_view(start, length);
    let imported = view.at();
    let (read_only, vouched) = (view.read_only(), view.vouched());
    let made = view.made();
    let (run, kept) = view.release(run);
    let hold = crate::remap::UnmapHold::new(start, length);
    let traced = backing::ImportTrace::new(site, start, length, imported);
    NoCopyImport {
        at: imported,
        run,
        keep: backing::keep_alive((kept, hold, traced)),
        read_only,
        vouched,
        made,
    }
}

/// What [`no_copy_release`] gives the runtime for a no-copy object.
struct NoCopyImport {
    /// The first byte the runtime imports.
    at: NonNull<u8>,
    /// The runtime's deallocator.
    run: Option<rt::Deallocator>,
    /// What keeps the import's pages mapped until the runtime freed it.
    keep: rt::KeepAlive,
    /// The pages are mapped read-only (`os::ImportView::read_only`).
    read_only: bool,
    /// The runtime may be told the pages are writable.
    vouched: bool,
    /// The word that the runtime's import was made, given once its call returned
    /// ([`crate::os::ImportMade`]).
    made: crate::os::ImportMade,
}

/// [`create`] of a buffer, its nil answer logged with its reason, once ([`no_buffer`]): `pinned`
/// where Metal's space refused the guest's pin ([`Placing::pin_refused`]), the span the pin needs
/// and the held byte it meets or the granule it is off; else the runtime's refusal (no memory,
/// an import that failed). The guest learns only that the buffer is nil, and its later uses of it
/// fail far from the cause.
fn create_buffer<T>(
    sel: &'static str,
    nil: Nil,
    pinned: Option<&PinRefused>,
    call: impl FnOnce() -> Result<T, rt::Error>,
) -> RaiseResult<Option<T>> {
    let mut refused = None;
    let made = create(sel, Policy::Dies, nil, || {
        call().inspect_err(|e| refused = Some(e.to_string()))
    });
    if matches!(made, Ok(None)) {
        match (pinned, refused) {
            (Some(pinned), _) => {
                no_buffer(
                    sel,
                    &format_args!("Metal's space refused the pin: {pinned:x?}"),
                );
            }
            (None, Some(refused)) => no_buffer(sel, &refused),
            (None, None) => {}
        }
    }
    made
}

/// Logs that `sel` answered nil, its buffer refused for `why`.
fn no_buffer(sel: &'static str, why: &dyn std::fmt::Display) {
    log::warn!(target: "reims_vgpu_vulkan::gpu", kind:% = why; "{sel} answered nil: {why}");
}

/// What Metal answers where the runtime refuses a buffer or heap made at `placing`'s address:
/// `nil`, which answers nil for the runtime's refusal of the address, for a guest pin and for one
/// Metal's space refused ([`Placing::nil_at_conflict`]); for a range the space placed, nil only
/// for memory the device cannot allocate, and a death for the runtime's refusal of the address,
/// where Metal makes the resource ([`metal_space`]).
fn placed_nil(placing: &Placing, nil: Nil) -> Nil {
    match placing.nil_at_conflict() {
        true => nil,
        false => Nil::Memory,
    }
}

/// `descriptor` with `token`, the range token of its range of Metal's space
/// ([`metal_space::Hold`]), where it has one.
fn with_token(
    descriptor: rt::BufferDescriptor<'_>,
    token: Option<rt::RangeToken>,
) -> rt::BufferDescriptor<'_> {
    match token {
        Some(token) => descriptor.with_range_token(token),
        None => descriptor,
    }
}

/// The slot of a buffer `sel` makes on `raw` that Metal suballocates, where `placing` holds one
/// ([`Placing::pool_slot`]): the runtime's reservation of its pool and the slot's offset in it,
/// which the buffer's descriptor takes (`BufferDescriptor::with_pool_slot`). A refusal of the
/// reservation dies at `sel`, naming it, with the space unlocked.
fn pool_slot(
    sel: &'static str,
    raw: &rt::Device,
    placing: &mut Placing,
) -> RaiseResult<Option<(Arc<rt::AddressPool>, u64)>> {
    match placing.pool_slot(raw) {
        None => Ok(None),
        Some(Ok(slot)) => Ok(Some(slot)),
        Some(Err(e)) => decide(sel, Policy::Dies, e),
    }
}

/// `descriptor` in `pool`'s slot, the one Metal suballocates it in ([`pool_slot`]), where it has
/// one.
fn in_slot<'a>(
    descriptor: rt::BufferDescriptor<'a>,
    pool: &'a Option<(Arc<rt::AddressPool>, u64)>,
) -> rt::BufferDescriptor<'a> {
    match pool {
        Some((pool, offset)) => descriptor.with_pool_slot(pool, *offset),
        None => descriptor,
    }
}

/// What Metal's space holds a buffer of its own memory of `options` as: private or not
/// ([`metal_space::buffer_space`]).
fn own_bytes(options: rt::ResourceOptions) -> metal_space::Bytes {
    metal_space::Bytes::Own {
        class: metal_space::PoolClass::of(options.storage_mode().ok()),
    }
}

/// The first byte of `bytes`, which the runtime's no-copy buffers take non-null; a NULL one dies
/// at `sel`.
fn pointer(sel: &'static str, bytes: HostPtr) -> NonNull<u8> {
    NonNull::new(bytes.as_ptr::<u8>()).unwrap_or_else(|| unrepresentable(sel, "a NULL pointer"))
}

/// A buffer descriptor's `resourceOptions` before any setter: `Shared`.
const BUFFER_DESCRIPTOR_OPTIONS: u64 = 0;

/// A heap descriptor's `resourceOptions` before any setter: `Private` (measured).
const HEAP_DESCRIPTOR_OPTIONS: u64 = 0x20;

/// `MTLResourceOptions` bit 24, which makes a buffer or heap read-only, and which the runtime
/// takes and drops ([`crate::gpu::backing`] keeps it).
const READ_ONLY_OPTION: u64 = 0x100_0000;

/// The resource options a descriptor holds after core's storage properties, in order, from
/// `initial`: a storage mode replaces bits 4 to 7 and a hazard tracking mode bits 8 and 9, as
/// their setters do (measured on a heap descriptor: storage 2 then tracking 2 holds 0x220). A
/// buffer descriptor's value sends no hazard tracking mode.
fn descriptor_options(initial: u64, options: &[OptionSet]) -> u64 {
    options.iter().fold(initial, |held, option| match *option {
        OptionSet::ResourceOptions(options) => options,
        OptionSet::ResourceOptionsOr(bits) => held | bits,
        OptionSet::StorageMode(mode) => (held & !0xf0) | ((mode & 0xf) << 4),
        OptionSet::HazardTrackingMode(mode) => (held & !0x300) | ((mode & 0x3) << 8),
    })
}

impl VulkanDevice {
    /// The runtime's answer to `heapTextureSizeAndAlignWithDescriptor:` for `desc`, asked inside
    /// `sel`; `None` where it refuses it. What [`metal_space::own_heap_size`] makes Metal's
    /// answer of where the runtime models none.
    pub(crate) fn size_and_align_at(
        &self,
        sel: &'static str,
        desc: &rt::TextureDescriptor,
    ) -> Option<SizeAndAlign> {
        let answer = plain_at(sel, || self.raw.heap_texture_size_and_align(desc)).ok()?;
        Some(SizeAndAlign {
            size: answer.size,
            align: answer.align,
        })
    }

    /// The runtime's model of Metal's answer to `heapTextureSizeAndAlignWithDescriptor:` for
    /// `desc` (`Device::metal_heap_texture_size_and_align`), asked inside `sel`: the
    /// same on every device, whatever its image needs, with no image made and nothing announced. `None` where the model has no answer (a format or a sample count not
    /// measured), and where the runtime refuses the descriptor. What
    /// [`metal_space::own_heap_size`] takes first.
    pub(crate) fn metal_answer_at(
        &self,
        sel: &'static str,
        desc: &rt::TextureDescriptor,
    ) -> Option<SizeAndAlign> {
        let answer = plain_at(sel, || self.raw.metal_heap_texture_size_and_align(desc));
        let answer = answer.ok()??;
        Some(SizeAndAlign {
            size: answer.size,
            align: answer.align,
        })
    }

    /// The handle of a texture this device made, answering the usage bits `private_usage` beyond
    /// the runtime's ([`TextureObject::private_usage`]), holding `space` of Metal's address space.
    pub(crate) fn texture_of(
        &self,
        raw: rt::Texture,
        private_usage: u64,
        space: Arc<metal_space::Hold>,
    ) -> VulkanTexture {
        Obj::new(TextureObject {
            raw,
            device: self.clone_ref(),
            under: None,
            surface: None,
            private_usage,
            space,
        })
    }

    /// The handle of a texture this device made over a buffer's bytes or placed in a heap, which
    /// holds that buffer or heap while it lives ([`Under`]), answering the usage bits
    /// `private_usage` beyond the runtime's ([`TextureObject::private_usage`]). It takes none of
    /// Metal's space but its descriptor ([`metal_space::texture`]).
    pub(crate) fn texture_in(
        &self,
        raw: rt::Texture,
        under: Under,
        private_usage: u64,
    ) -> VulkanTexture {
        self.texture_under(raw, under, private_usage, None)
    }

    /// The handle of a texture this device made over `under`, a buffer that stands for an
    /// IOSurface, lying over `surface` of it ([`crate::gpu::resource`]'s
    /// `new_texture_over_surface_plane`), as [`Self::texture_in`]'s.
    pub(crate) fn texture_over_surface(
        &self,
        raw: rt::Texture,
        under: Under,
        surface: OverSurface,
    ) -> VulkanTexture {
        self.texture_under(raw, under, 0, Some(surface))
    }

    /// [`Self::texture_in`]'s handle, over `surface` of an IOSurface where given.
    fn texture_under(
        &self,
        raw: rt::Texture,
        under: Under,
        private_usage: u64,
        surface: Option<OverSurface>,
    ) -> VulkanTexture {
        let kind = match under {
            Under::Buffer(..) => metal_space::TextureKind::OverBuffer,
            Under::Heap(..) => metal_space::TextureKind::InHeap,
        };
        let space = Arc::new(metal_space::texture(kind));
        Obj::new(TextureObject {
            raw,
            device: self.clone_ref(),
            under: Some(under),
            surface,
            private_usage,
            space,
        })
    }

    /// A texture of `descriptor` of the platform's own, for a helper the guest never reaches
    /// (the window's cursor glyph): it takes no range of Metal's space ([`metal_space::Hold`]'s
    /// `NONE`) and no resource ID (the runtime's `IdPlacement::Outside`, whose texture answers 0),
    /// so no guest-visible address or ID depends on it. `None` where the runtime refuses it.
    pub fn new_own_texture(&self, descriptor: &VulkanTextureDescriptor) -> Option<VulkanTexture> {
        let fields = descriptor.fields();
        let mut desc = fields.descriptor("newTextureWithDescriptor:", Maker::Other);
        desc.placement = rt::IdPlacement::Outside;
        let raw = plain(|| self.raw.new_texture(&desc)).ok()?;
        Some(self.texture_of(raw, 0, Arc::new(metal_space::Hold::NONE)))
    }

    /// `newBufferWithLength:options:`'s buffer (`new_buffer_of_length`), declaring the bytes
    /// between a texture's rows over it unobserved where `padding_unobserved`, and its byte
    /// `aligned_byte` on a multiple of 256 bytes of its memory where given
    /// (`new_buffer_of_length_padding_unobserved`), in memory
    /// exportable as `export` where given (`new_dma_buf_buffer_of_length_padding_unobserved`). A
    /// buffer to export is no buffer of Metal's: any refusal of it answers `None`, logged at debug,
    /// and the caller makes the buffer as any other, where one Metal makes refuses as
    /// `newBufferWithLength:options:` does.
    fn buffer_of_length(
        &self,
        length: u64,
        options: u64,
        padding_unobserved: bool,
        aligned_byte: Option<u64>,
        export: Option<rt::MemoryExport>,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        let sel = "newBufferWithLength:options:";
        let options = crate::gpu::resource_options(sel, options);
        let (placement, mut placing) = metal_space::placing(length, own_bytes(options), None);
        let pool = pool_slot(sel, &self.raw, &mut placing)?;
        let descriptor = rt::BufferDescriptor::new(length, options, rt::BufferContents::Zero)
            .with_placement(placement);
        let descriptor = match padding_unobserved {
            true => descriptor.with_padding_unobserved(),
            false => descriptor,
        };
        let descriptor = match aligned_byte {
            Some(byte) => descriptor.with_aligned_byte(byte),
            None => descriptor,
        };
        let descriptor = match export {
            Some(export) => descriptor.with_exported_as(export),
            None => descriptor,
        };
        let descriptor = with_token(in_slot(descriptor, &pool), placing.token());
        let made = match export {
            None => create_buffer(
                sel,
                placed_nil(&placing, Nil::BufferLength),
                placing.pin_refused(),
                || self.raw.new_buffer_with_descriptor(descriptor),
            ),
            Some(export) => match plain(|| self.raw.new_buffer_with_descriptor(descriptor)) {
                Ok(raw) => Ok(Some(raw)),
                Err(error) => {
                    log::debug!("no buffer of {length} bytes exportable as {export:?}: {error}");
                    Ok(None)
                }
            },
        };
        metal_space::let_go(pool);
        made.map(|b| b.map(|raw| self.buffer(raw)))
    }

    /// The handle of a buffer this device made.
    pub(crate) fn buffer(&self, raw: rt::Buffer) -> VulkanBuffer {
        Obj::new(BufferObject {
            raw,
            device: self.clone_ref(),
            backing: Backing::new(false),
            heap: None,
        })
    }

    /// The handle of a buffer placed in `heap`.
    pub(crate) fn placed_buffer(&self, raw: rt::Buffer, heap: &VulkanHeap) -> VulkanBuffer {
        Obj::new(BufferObject {
            raw,
            device: self.clone_ref(),
            backing: Backing::new(false),
            heap: Some(heap.clone_ref()),
        })
    }
}

impl GpuDevice<VulkanPlatform> for VulkanDevice {
    /// Nothing to allow: the runtime makes a library of a metallib whatever platform it was
    /// built for (`newLibraryWithData:error:` checks no platform).
    fn allow_foreign_libraries(&self) {
        plain(|| ());
    }

    /// The runtime's encoder's layout (`ArgumentEncoder::layout`), Metal's rules for descriptors
    /// as the runtime measured them. A descriptor of a data type, access or texture type the
    /// runtime has no enumerator of answers nil, as one of a type no argument has; what the
    /// runtime refuses as Metal raises (no arguments, a misplaced constant block alignment) is the
    /// caller's to catch.
    fn argument_layout(&self, arguments: &[ArgumentDesc]) -> RaiseResult<Option<ArgumentLayout>> {
        let sel = "newArgumentEncoderWithArguments:";
        let mut descriptors = Vec::with_capacity(arguments.len());
        for a in arguments {
            let (
                Some(data_type),
                Some(access),
                Some(texture_type),
                Ok(index),
                Ok(array_length),
                Ok(alignment),
            ) = (
                u32::try_from(a.data_type)
                    .ok()
                    .and_then(|raw| metal2vulkan::metal::DataType::try_from(raw).ok()),
                u32::try_from(a.access)
                    .ok()
                    .and_then(|raw| metal2vulkan::metal::BindingAccess::try_from(raw).ok()),
                crate::gpu::runtime_texture_type(a.texture_type),
                u32::try_from(a.index),
                u32::try_from(a.array_length),
                u32::try_from(a.constant_block_alignment),
            )
            else {
                return Ok(None);
            };
            let mut d = metal2vulkan::bind::ArgumentDescriptor::new(data_type, index);
            d.array_length = array_length;
            d.access = access;
            d.texture_type = texture_type;
            d.constant_block_alignment = alignment;
            descriptors.push(d);
        }
        let encoder = create(sel, Policy::Caught, Nil::Never, || {
            self.raw.new_argument_encoder_with_arguments(&descriptors)
        })?;
        Ok(encoder.map(|e| layout_of(e.layout())))
    }

    /// The runtime's encoder of the function's argument buffer, laid out as its AIR lays it out,
    /// nested argument buffers with it. No argument buffer at the index is the caller's to catch,
    /// as Metal raises.
    fn function_argument_layout(
        &self,
        function: &VulkanFunction,
        buffer_index: u64,
    ) -> RaiseResult<Option<ArgumentLayout>> {
        let sel = "newArgumentEncoderWithBufferIndex:";
        let Ok(index) = u32::try_from(buffer_index) else {
            return Ok(None);
        };
        let encoder = create(sel, Policy::Caught, Nil::Never, || {
            function.new_argument_encoder_with_buffer_index(index)
        })?;
        Ok(encoder.map(|e| layout_of(e.layout())))
    }

    /// The runtime's answer, Metal's but for a depth or stencil texture with
    /// `allowGPUOptimizedContents` off, which MoltenVK keeps GPU-optimized: the runtime answers
    /// 66048 bytes for a 128 by 128 Depth32Float texture and 98816 for a Depth32Float_Stencil8
    /// one, where Metal answers 65536 and 81920 (measured on the M3 Max; with it on or unset,
    /// the sizes are Metal's; [`crate::gpu::caps`]' list). The query hears the runtime announce
    /// it (`OptimizedDepthStencilTexture`, [`crate::answer::HEARD`]) and answers Metal's: the
    /// runtime's model of Metal's answer ([`VulkanDevice::metal_answer_at`]), the size
    /// `newTextureWithDescriptor:` gives such a texture's range of Metal's space, which answers
    /// `Depth16Unorm`, `Depth32Float`, `Stencil8` and `Depth32Float_Stencil8` textures, and an
    /// `X32_Stencil8` one as a `Stencil8` one of its shape: Metal's answer at each
    /// shape measured (`resource`'s
    /// `heap_depth_stencil_shapes_with_optimized_contents_off_are_metals` and
    /// `x32_stencil8_size_queries_are_metals`). It dies at the selector where the model has
    /// none. One of
    /// `ShaderAtomic` usage, at which Metal asserts, the runtime refuses first (`Invalid`), which
    /// dies at the selector too. A heap's placement of such a texture still
    /// dies where the runtime's size decides it, as its image needs that size in the heap
    /// (`crate::gpu::resource::Placements`).
    /// It relays, naming the kind and the texture, and in strict mode dies at the selector, for a
    /// texture whose layout in Metal's heaps the runtime does not model, whose size and alignment the runtime answers as its device's image's, larger or smaller than
    /// Metal's, and announces so (`unmeasured-heap-texture-size`, [`crate::answer::HEARD`];
    /// [`metal_space::unmeasured_size`]): the guest reads the answer, and no suite case or guest
    /// run asks it of such a texture. The runtime refuses 8 samples first on every device, as
    /// Metal stops them, so the query dies at that refusal. The runtime models a mipmapped texture
    /// Metal keeps losslessly compressed (a seven-level 64x64 `R32Uint` render target answers
    /// Metal's 0x5800 bytes) and a `Depth32Float_Stencil8` one of one sample. Where the device's
    /// image takes more than the model's answer, the runtime's answer
    /// is the device's (`heap-texture-larger-than-metal`, [`crate::answer`]), as a heap places
    /// the texture by it.
    /// Metal's 0 bytes, at Metal's alignment, for a format it makes no texture of from a
    /// descriptor alone, which the runtime answers as Metal does
    /// ([`crate::gpu::no_texture::descriptor`]).
    fn heap_texture_size_and_align(
        &self,
        desc: &VulkanTextureDescriptor,
    ) -> RaiseResult<SizeAndAlign> {
        let sel = "heapTextureSizeAndAlignWithDescriptor:";
        let fields = desc.fields();
        let desc = fields.descriptor(sel, Maker::SizeQuery);
        let raw = fields.raw_pixel_format();
        let (answer, heard) = hearing(|| {
            send(sel, Policy::Dies, || {
                (self.raw.heap_texture_size_and_align(&desc))
                    .map_err(|e| format_refused(sel, "a texture of", raw, e))
            })
        });
        if heard.contains(&rt::Degradation::OptimizedDepthStencilTexture) {
            match self.metal_answer_at(sel, &desc) {
                Some(metal) => return Ok(metal),
                None => crate::gpu::resource::optimized_size(sel, "the size and alignment of"),
            }
        }
        if heard.contains(&rt::Degradation::UnmeasuredHeapTextureSize) {
            metal_space::unmeasured_size(sel, "the size and alignment of", &desc);
        }
        let answer = answer?;
        Ok(SizeAndAlign {
            size: answer.size,
            align: answer.align,
        })
    }

    /// Any pointer and any length of at least 1 byte; a length of 0, or past the device's largest
    /// buffer, is nil, as Metal answers. So is memory with a page not mapped, or mapped without
    /// read access, which Metal makes no buffer over (measured: PROT_NONE pages, an unmapped
    /// page, and a range of a mapped and an unmapped page are nil; read-only pages make a
    /// buffer): the runtime imports the pages as it makes the buffer, as Metal wires them, and
    /// refuses those it cannot import, which [`OsVm::readable`] tells from its other
    /// refusals, which die. The deallocator runs once, when the runtime releases the buffer; one
    /// the runtime refuses never runs, and its captures are kept. At the address Metal's space
    /// gives it, keeping the pointer's offset into its page ([`metal_space`]), so made through
    /// the runtime's descriptor, which takes an address.
    fn new_buffer_over_host_memory(
        &self,
        _on: OnFault,
        bytes: HostRange<'_>,
        options: u64,
        deallocator: Option<NoCopyDeallocator>,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        let sel = "newBufferWithBytesNoCopy:length:options:deallocator:";
        let bit_24 = options & READ_ONLY_OPTION != 0;
        let options = crate::gpu::resource_options(sel, options);
        let slot = deallocator.map(|d| Arc::new(Mutex::new(Some(d))));
        let run = slot
            .as_ref()
            .map(|slot| runtime_deallocator(slot, |d: NoCopyDeallocator, _, _| d()));
        let start = pointer(sel, bytes.ptr());
        let offset = u64::try_from(start.as_ptr().addr()).unwrap_or(0);
        let (placement, mut placing) =
            metal_space::placing(bytes.len(), metal_space::Bytes::NoCopy(offset), None);
        let token = placing.token();
        // The bytes, and what keeps them mapped until the runtime freed its import.
        let NoCopyImport {
            at: imported,
            run,
            keep,
            read_only,
            vouched,
            made: import_made,
        } = no_copy_release(start, bytes.len(), run, backing::ImportSite::NoCopyBytes);
        let made = create_buffer(sel, placed_nil(&placing, Nil::BufferLength), None, || {
            // SAFETY: `HostRange::new`'s contract keeps the bytes mapped until the deallocator
            // runs, or for the buffer's life with none, and an unmap of core's through the
            // platform's VM waits for the keep-alive; and a vouch is the table's record of the
            // pages as they are mapped now.
            let memory = unsafe { crate::os::host_memory_vouched(imported, vouched) };
            let contents = rt::BufferContents::NoCopy {
                memory,
                deallocator: run,
                keep_alive: Some(keep),
            };
            let descriptor =
                rt::BufferDescriptor::new(bytes.len(), options, contents).with_placement(placement);
            match self
                .raw
                .new_buffer_with_descriptor(with_token(descriptor, token))
            {
                Ok(raw) => Ok(Some(raw)),
                // Pages Metal makes no buffer over: nil.
                Err(e) if !OsVm::new().readable(bytes.ptr(), bytes.len()) => {
                    no_buffer(sel, &format_args!("pages it cannot read: {e}"));
                    Ok(None)
                }
                Err(e) => Err(e),
            }
        })
        .map(Option::flatten);
        keep_unrun(&made, slot);
        if matches!(made, Ok(Some(_))) {
            import_made.made(&self.raw);
        }
        made.map(|b| {
            b.map(|raw| {
                let buffer = self.buffer(raw);
                buffer.backing.imported(read_only, bit_24);
                buffer
            })
        })
    }

    /// [`VulkanDevice::buffer_of_length`]'s, with neither of its declarations.
    fn new_buffer_of_length(
        &self,
        _on: OnFault,
        length: u64,
        options: u64,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        self.buffer_of_length(length, options, false, None, None)
    }

    /// As `new_buffer_of_length`'s, with the runtime's declaration that the bytes between the
    /// rows and slices of a texture over the buffer are unobserved
    /// (`BufferDescriptor::with_padding_unobserved`): a texture whose rows lie with bytes between
    /// them is then a linear image over the buffer's memory that holds them, where the device
    /// lays one out at the texture's rows (`WorkCounts::unobserved_padding_aliases`), and not a
    /// shadow (`TextureShadowReason::BytesBetweenRows`). Its byte `aligned_byte`, where given, lies
    /// on a multiple of 256 bytes of its memory (`BufferDescriptor::with_aligned_byte`), which
    /// takes at most 255 bytes more of it.
    fn new_buffer_of_length_padding_unobserved(
        &self,
        _on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        self.buffer_of_length(length, options, true, aligned_byte, None)
    }

    /// As `new_buffer_of_length_padding_unobserved`'s, in memory the runtime makes exportable as a
    /// dma-buf (`BufferDescriptor::with_exported_as`, `MemoryExport::DmaBuf`), at the address
    /// Metal's space gives it, as that buffer's. Any refusal (a device that exports no dma-buf of
    /// a buffer's memory, `RuntimeGap::DmaBufExport`, among them) answers `None`, logged at debug,
    /// and the caller makes the buffer as any other.
    #[cfg(unix)]
    fn new_dma_buf_buffer_of_length_padding_unobserved(
        &self,
        _on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        let export = Some(rt::MemoryExport::DmaBuf);
        self.buffer_of_length(length, options, true, aligned_byte, export)
    }

    /// The runtime's buffer of zeros in memory exportable as an opaque file descriptor
    /// (`BufferDescriptor::exported_as`), its byte `aligned_byte` on a multiple of 256 bytes of
    /// that memory where given (`BufferDescriptor::with_aligned_byte`), its GPU address placed by
    /// the runtime rather than as Metal places a new buffer's, as nothing of Metal's makes one.
    /// Any refusal (a device that exports no buffer's memory, `RuntimeGap::ExportableMemory`,
    /// among them) answers `None`, logged at debug, and the caller makes the buffer as any other.
    #[cfg(unix)]
    fn new_exportable_buffer_of_length(
        &self,
        _on: OnFault,
        length: u64,
        options: u64,
        aligned_byte: Option<u64>,
    ) -> RaiseResult<Option<VulkanBuffer>> {
        let options = crate::gpu::resource_options("newBufferWithLength:options:", options);
        let descriptor = rt::BufferDescriptor::new(length, options, rt::BufferContents::Zero)
            .with_exported_as(rt::MemoryExport::OpaqueFd);
        let descriptor = match aligned_byte {
            Some(byte) => descriptor.with_aligned_byte(byte),
            None => descriptor,
        };
        match plain(|| self.raw.new_buffer_with_descriptor(descriptor)) {
            Ok(raw) => Ok(Some(self.buffer(raw))),
            Err(error) => {
                log::debug!("no exportable buffer of {length} bytes: {error}");
                Ok(None)
            }
        }
    }

    /// A queue of at most 64 uncompleted command buffers, as Metal's. A first use of Metal's
    /// device ([`metal_space::FirstUse::Queue`]).
    fn new_command_queue(&self) -> Option<VulkanCommandQueue> {
        let raw = get("newCommandQueue", || self.raw.new_command_queue());
        metal_space::first_use(metal_space::FirstUse::Queue);
        Some(Obj::new(QueueObject::new(raw, self.clone_ref())))
    }

    /// A queue of at most 64 uncompleted command buffers, as `newCommandQueue`'s, made by
    /// [`unshared_queue`]; a refusal dies at the call, as `newCommandQueue`'s does. A first use
    /// of Metal's device, as `newCommandQueue`'s.
    fn new_command_queue_unshared(&self) -> Option<VulkanCommandQueue> {
        let raw = get("newCommandQueue", || {
            unshared_queue(&self.raw, METAL_MAX_COMMAND_BUFFER_COUNT)
        });
        metal_space::first_use(metal_space::FirstUse::Queue);
        Some(Obj::new(QueueObject::new(raw, self.clone_ref())))
    }

    /// The recoveries the runtime announced in the process ([`crate::answer::recoveries`]).
    fn recoveries(&self) -> u64 {
        crate::answer::recoveries()
    }

    /// On Linux and Windows, where the device uses the guest's RAM whole and the runtime copies
    /// by its runs (`os/host_ram.rs`); never on macOS, whose devices are handed no RAM.
    fn copies_guest_ram(&self) -> bool {
        #[cfg(any(target_os = "linux", windows))]
        {
            crate::os::copies_guest_ram(&self.raw)
        }
        #[cfg(not(any(target_os = "linux", windows)))]
        {
            false
        }
    }

    /// The runtime's placements of textures over the buffer's pages, plane by plane
    /// (`crate::gpu::pages_in_place`).
    fn surface_pages_in_place(&self, pages: &VulkanBuffer, planes: &[PagesPlane]) -> PagesInPlace {
        crate::gpu::pages_in_place::surface_pages_in_place(&pages.raw, planes)
    }

    /// The runtime's placements of textures over the pages before any import of them
    /// (`crate::gpu::pages_in_place::surface_pages_never_in_place`).
    fn surface_pages_never_in_place(
        &self,
        planes: &[PagesPlane],
        pages: &[std::ops::Range<u64>],
    ) -> Option<String> {
        crate::gpu::pages_in_place::surface_pages_never_in_place(&self.raw, planes, pages)
    }

    /// The runtime's placements of textures over a fresh surface's pages before any import of
    /// them, each page a piece of its own
    /// (`crate::gpu::pages_in_place::fresh_surface_never_in_place`).
    fn fresh_surface_never_in_place(&self, planes: &[PagesPlane], length: u64) -> Option<String> {
        crate::gpu::pages_in_place::fresh_surface_never_in_place(&self.raw, planes, length)
    }

    /// The runtime's alignment of a BGRA8 texture's offset and rows in a buffer at which the
    /// device's images hold it in place (`rt::Device::minimum_linear_texture_alignment_for_pixel_format`,
    /// the device's own answer, which the guest's `minimumLinearTextureAlignmentForPixelFormat:`
    /// does not take: `caps::linear_alignment`).
    fn linear_row_alignment(&self) -> u64 {
        let pf = metal2vulkan::metal::PixelFormat::BGRA8Unorm;
        plain(|| {
            self.raw
                .minimum_linear_texture_alignment_for_pixel_format(pf)
        })
        .unwrap_or(0)
    }

    /// The runtime's pitch of a `Private` 2D texture over a buffer of its own memory that only
    /// samples (`rt::Device::linear_texture_bytes_per_row`, the driver's `rowPitch` of its linear
    /// image), asked of one format of each texel size a buffer backs that divides `row_bytes`
    /// (`R8Unorm`, `RG8Unorm`, `BGRA8Unorm`, `RGBA16Float` and `RGBA32Float`), `row_bytes` over
    /// it wide and `rows` high, where those that answer agree: their pitch where it is a multiple
    /// of Metal's 16-byte linear alignment ([`caps::METAL_LINEAR_ALIGNMENT`]), at which alone the
    /// runtime makes a texture over a buffer or over a plane of an IOSurface without planes, as
    /// Metal does (`BufferTextureProblem::RowPitchAlignment`), and a texture at it is the
    /// driver's own linear image; else that pitch rounded up to the device's line of memory and
    /// to Metal's 16 (`rt::Device::in_place_check_line`, `nonCoherentAtomSize`), at which the
    /// runtime makes a `DRM_FORMAT_MOD_LINEAR` image over the buffer's memory, and `rows` rows
    /// to their pitch fill whole lines. A device whose stores write a row's last line back whole
    /// (the Iris Xe) renders in place only into rows that share no line with bytes past them
    /// (the runtime's `in_place::Padding::of`): rounded up to 16 alone, ANV's 1100 bytes for 275
    /// `BGRA8Unorm` texels is 1104, and 54 rows of it end 32 bytes into a line. ANV's linear
    /// images lay rows of 280 bytes (70 `BGRA8Unorm` texels) side by side (**measured**
    /// 2026-11-12, Iris Xe, Mesa 26.2.6), and textures over device caches laid out at such
    /// pitches (292 and 280 bytes) were refused, hundreds a boot (macOS 14.8.9 and 26.7.2 guests
    /// on Iris Xe). Each pair is asked once a device: a new answer makes and releases an image of
    /// each such format.
    fn linear_rows_pitch(&self, row_bytes: u64, rows: u64) -> Option<u64> {
        use metal2vulkan::metal::PixelFormat as Pf;
        /// The answers a device keeps; past them they are asked again.
        const KEPT: usize = 4096;
        if let Some(&answer) = self.rows_pitches.lock().get(&(row_bytes, rows)) {
            return answer;
        }
        let formats = [
            (1, Pf::R8Unorm),
            (2, Pf::RG8Unorm),
            (4, Pf::BGRA8Unorm),
            (8, Pf::RGBA16Float),
            (16, Pf::RGBA32Float),
        ];
        let mut answers = formats
            .into_iter()
            .filter(|&(size, _)| row_bytes != 0 && rows != 0 && row_bytes.is_multiple_of(size))
            .filter_map(|(size, pixel_format)| {
                let mut descriptor = rt::TextureDescriptor::default();
                descriptor.pixel_format = pixel_format;
                descriptor.width = row_bytes / size;
                descriptor.height = rows;
                descriptor.resource_options = rt::ResourceOptions::private();
                plain(|| self.raw.linear_texture_bytes_per_row(&descriptor))
                    .ok()
                    .flatten()
            });
        let first = answers.next();
        let answer = first
            .filter(|&pitch| answers.all(|other| other == pitch))
            .and_then(|pitch| {
                if pitch.is_multiple_of(caps::METAL_LINEAR_ALIGNMENT) {
                    return Some(pitch);
                }
                let line = plain(|| self.raw.in_place_check_line());
                (line.is_power_of_two())
                    .then(|| pitch.next_multiple_of(line.max(caps::METAL_LINEAR_ALIGNMENT)))
            });
        let mut kept = self.rows_pitches.lock();
        if kept.len() >= KEPT {
            kept.clear();
        }
        kept.insert((row_bytes, rows), answer);
        answer
    }

    /// Every refusal dies but the compile's of a function or descriptor it finds wrong, which is
    /// the `NSError`: an unspecialized function constant dies, as Metal aborts for one
    /// (measured), and so does a nil function, which the runtime has no form for.
    fn new_compute_pipeline_state_for_function(
        &self,
        compute_function: Option<&VulkanFunction>,
    ) -> Result<Option<VulkanComputePipeline>, NsErrorText> {
        let sel = "newComputePipelineStateWithFunction:error:";
        let function = compute_function.unwrap_or_else(|| unrepresentable(sel, "a nil function"));
        let options = rt::PipelineOptions::default();
        timing::creation(
            sel,
            timing::DISPATCH_LATER,
            || match plain_at(sel, || {
                self.raw
                    .new_compute_pipeline_state_with_function(function, &options)
            }) {
                Ok(state) => Ok(Some(Obj::new(ComputePipelineObject::new(state)))),
                Err(e) => Err(compile_error(sel, e, None)),
            },
            || format!("kernel {:?}", function.name()),
            |made| timing::answer(made, |s| timing::state(s.unique_identifier)),
        )
    }

    /// The one creator whose result is kept: the library's own shaders, built into a metallib
    /// with this crate, are loaded once per device, and a later call answers the same library
    /// ([`super::shaders`]).
    fn new_shader_library(&self) -> Result<Option<VulkanLibrary>, NsErrorText> {
        super::shaders::library(self)
    }

    fn new_event(&self, _on: OnFault) -> RaiseResult<Option<VulkanEvent>> {
        send("newEvent", Policy::Dies, || self.raw.new_event()).map(|e| Some(Obj::new(e)))
    }

    /// The runtime makes a fence for every call.
    fn new_fence(&self) -> RaiseResult<Option<VulkanFence>> {
        Ok(Some(Obj::new(plain(|| self.raw.new_fence()))))
    }

    /// A metallib only, as Metal and the runtime take: other bytes, raw AIR and a bitcode wrapper
    /// among them, are Metal's `NSError` (measured). A gap of the runtime's or the
    /// device's is its refusal as the `NSError`, relayed by default ([`relay_gap`]).
    fn new_dynamic_library(
        &self,
        library: &VulkanLibrary,
    ) -> RaiseResult<Result<Option<crate::gpu::dynamic_library::VulkanDynamicLibrary>, NsErrorText>>
    {
        crate::gpu::dynamic_library::new_dynamic_library(&self.raw, library)
    }

    fn new_library(&self, data: &[u8]) -> RaiseResult<Result<Option<VulkanLibrary>, NsErrorText>> {
        let sel = "newLibraryWithData:error:";
        let made = timing::creation(
            sel,
            "",
            || {
                send(sel, Policy::Dies, || {
                    match self.raw.new_library_with_data(data) {
                        Ok(raw) => Ok(Ok(raw)),
                        Err(
                            e @ rt::Error::Compile(metal2vulkan::error::Error::InvalidInput(_)),
                        ) => {
                            // Metal's NSError names no cause; the runtime's names its check and
                            // the offset in the file. At debug: a guest call Metal refuses too.
                            log::debug!(
                                target: "reims_vgpu_vulkan::gpu",
                                "{sel} refused {} bytes as no metallib: {e}",
                                data.len()
                            );
                            Ok(Err(invalid_library_file()))
                        }
                        Err(e) if relay_gap(sel, &e, None) => Ok(Err(error_text(&e))),
                        Err(e) => Err(e),
                    }
                })
            },
            || format!("{} bytes", data.len()),
            |made| match made {
                Ok(Ok(raw)) => format!("made, {} functions", raw.function_names().len()),
                Ok(Err(e)) => format!("refused: {}", e.description),
                Err(_) => "raised".to_owned(),
            },
        )?;
        Ok(made.map(|raw| {
            Some(Obj::new(LibraryObject {
                raw,
                device: self.clone_ref(),
            }))
        }))
    }

    fn new_rate_map(
        &self,
        descriptor: &VulkanRateMapDescriptor,
    ) -> RaiseResult<Option<VulkanRateMap>> {
        super::rate_map::new_map(self, descriptor)
    }

    fn new_shared_event(&self) -> Option<VulkanSharedEvent> {
        Some(Obj::new(get("newSharedEvent", || {
            self.raw.new_shared_event()
        })))
    }

    /// A texture with in-process handles; nil for storage other than `Private`, as Metal answers,
    /// a size of 0 included, which it aborts at in `Private` ([`crate::gpu::desc`]); one the
    /// runtime makes at such a storage dies, as it would hold no range of Metal's space. Nil for
    /// a format Metal makes no texture of from a descriptor alone
    /// ([`crate::gpu::desc::TextureFields::formatless`]), as Metal answers before its validation,
    /// whatever the descriptor holds (measured: [`crate::gpu::no_texture`]'s docs, and
    /// [`crate::gpu::ycbcr`]'s), where the runtime validates first. One of protection options
    /// other than none is made as without them, relayed, where Metal answers nil
    /// ([`crate::gpu::desc::TextureFields::made_unprotected`]).
    fn new_shared_texture(
        &self,
        descriptor: &VulkanTextureDescriptor,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newSharedTextureWithDescriptor:";
        let fields = descriptor.fields();
        if fields.formatless().is_some() {
            return Ok(None);
        }
        // Metal answers nil for a value the runtime keeps raw at a storage other than `Private`,
        // as for every format, and at `Private` for 630 to 650 where it describes no format for
        // them; it makes a texture of a format it describes, and asserts at 255 and 262
        // (measured: `formats_the_runtime_lacks_die_where_textures_are_made`). It answers
        // nil for `MTLPixelFormatInvalid` at either storage (measured:
        // `invalid_formats_are_metals`).
        const PRIVATE: u64 = 2;
        let storage = (fields.desc.resource_options.raw() >> 4) & 0xf;
        if let Some(raw) = fields.lacked_format
            && (storage != PRIVATE
                || (matches!(raw, 630..=650) && !crate::gpu::format_info::described(sel, raw)))
        {
            return Ok(None);
        }
        if fields.desc.pixel_format == metal2vulkan::metal::PixelFormat::Invalid
            && fields.lacked_format.is_none()
        {
            return Ok(None);
        }
        let desc = fields.descriptor(sel, Maker::Shared);
        // Its range of Metal's space, placed before the runtime makes it, at `Private` storage,
        // where it would be made: its allocation is sized by Apple's layout from Metal's
        // description of its format ([`crate::gpu::format_info::block`]). At another
        // storage the runtime answers nil, as Metal does, and nothing is placed; a texture the
        // runtime made there would hold no range of Metal's space where Metal makes none, so it
        // dies. The runtime's range token holds it until the runtime frees the texture, and its
        // object shares it ([`metal_space::Hold`]).
        let space = match storage == PRIVATE {
            true => metal_space::texture(metal_space::shared_backing(sel, &desc)),
            false => metal_space::Hold::NONE,
        };
        let space = Arc::new(space);
        let token: rt::RangeToken = Box::new(Arc::clone(&space));
        let raw = fields.raw_pixel_format();
        let made = create(sel, Policy::Dies, Nil::StorageMode, || {
            (self.raw.new_shared_texture_with_range_token(&desc, token))
                .map_err(|e| created_format_refused(sel, "a texture of", raw, e))
        })?;
        if made.is_some() && storage != PRIVATE {
            unrepresentable(
                sel,
                &format!(
                    "a shareable texture of storage mode {storage}, which the runtime makes where \
                     Metal answers nil, and which takes no range of Metal's space"
                ),
            );
        }
        if made.is_some() {
            fields.made_unprotected(sel);
        }
        Ok(made.map(|raw| self.texture_of(raw, 0, space)))
    }

    /// The texture of a handle of this process, its resource ID at `resource_index` (0: one the
    /// runtime picks), answering its usage as the handle's texture does. It takes no range of
    /// Metal's space and shares the handle's texture's (`vaprobe surfaces`), which the shared
    /// texture's range token holds while it lives, and while a command buffer that holds it
    /// runs, and its object shares ([`metal_space::Hold`]; `vaprobe sharedrelease`: with the
    /// shared texture and its handle released, Metal kept the range while the opened texture
    /// lived, and freed it once that was released too). The runtime takes no nil handle: a nil
    /// one dies.
    fn open_shared_texture(
        &self,
        handle: Option<&VulkanSharedTextureHandle>,
        resource_index: u64,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newSharedTextureWithHandle:withResourceIndex:";
        let handle = handle.unwrap_or_else(|| unrepresentable(sel, "a nil handle"));
        let placement = rt::IdPlacement::At(rt::ResourceId(resource_index));
        let made = plain_at(sel, || {
            self.raw
                .new_shared_texture_with_handle(&handle.raw, placement)
        });
        let raw = match made {
            Ok(raw) => raw,
            Err(e) => return created_or_decided(sel, Policy::Dies, e),
        };
        let usage = handle.private_usage;
        Ok(Some(self.texture_of(
            raw,
            usage,
            Arc::new(metal_space::Hold::sharing(Arc::clone(&handle.space))),
        )))
    }

    /// Nil for memory the device cannot allocate, and for a descriptor Metal makes no texture
    /// of, as Metal answers, which the runtime answers as Metal does: a type Metal makes none of
    /// in the format (a two-plane extended-range cube or cube array that passes every check
    /// Metal's validation makes), and a format Metal makes no texture of from a descriptor alone
    /// ([`crate::gpu::no_texture::descriptor`]). It holds what Metal's space
    /// gives a texture of memory of its own ([`metal_space::texture_space`]): a range of
    /// Metal's heap size for it ([`metal_space::own_heap_size`]: the runtime's model of Metal's
    /// answer, [`VulkanDevice::metal_answer_at`], on every device, where the device's own answer
    /// may be larger), a slot for one Metal suballocates (of a heap size under a page), none for
    /// a memoryless one, placed before the runtime makes it, which the runtime's range token holds
    /// until it frees the texture, and its object shares ([`metal_space::Hold`]). Where the model
    /// has no answer, a depth or stencil texture without `allowGPUOptimizedContents` whose size the
    /// runtime answers larger than Metal's (`optimized-depth-stencil-texture`) and Metal's is not
    /// measured for dies here, at every shape; so does one whose layout in Metal's heaps the
    /// runtime does not model (`unmeasured-heap-texture-size`), at every shape ([`metal_space::unmeasured_size`]). One of protection options other than none
    /// is made as without them, relayed, where Metal answers nil
    /// ([`crate::gpu::desc::TextureFields::made_unprotected`]).
    fn new_texture(
        &self,
        _on: OnFault,
        descriptor: &VulkanTextureDescriptor,
    ) -> RaiseResult<Option<VulkanTexture>> {
        let sel = "newTextureWithDescriptor:";
        let fields = descriptor.fields();
        let desc = fields.descriptor(sel, Maker::Other);
        let raw = fields.raw_pixel_format();
        let kind = match desc.resource_options.storage_mode() {
            Ok(rt::StorageMode::Memoryless) => TextureKind::Memoryless,
            _ => {
                let metal = self.metal_answer_at(sel, &desc);
                // A type Metal makes no texture of in the format, which the runtime's model of
                // Metal's size answers 0 bytes for, is nil before any of Metal's space is taken
                // for it; a format Metal makes no texture of from a descriptor alone goes on to
                // the runtime's nil.
                if metal.is_some_and(|metal| metal.size == 0) && fields.formatless().is_none() {
                    return Ok(None);
                }
                let answer = |desc: &rt::TextureDescriptor| self.size_and_align_at(sel, desc);
                let (heap_size, heard) =
                    hearing(|| metal_space::own_heap_size(&desc, metal, answer));
                if heard.contains(&rt::Degradation::OptimizedDepthStencilTexture) {
                    crate::gpu::resource::optimized_size(sel, "the range of Metal's space of");
                }
                if heard.contains(&rt::Degradation::UnmeasuredHeapTextureSize) {
                    metal_space::unmeasured_size(sel, "the range of Metal's space of", &desc);
                }
                metal_space::own_memory(sel, &desc, heap_size)
            }
        };
        let space = Arc::new(metal_space::texture(kind));
        let token: rt::RangeToken = Box::new(Arc::clone(&space));
        let made = create(sel, Policy::Dies, Nil::Memory, || {
            (self.raw.new_texture_with_range_token(&desc, token))
                .map_err(|e| created_format_refused(sel, "a texture of", raw, e))
        })?;
        if made.is_some() {
            fields.made_unprotected(sel);
        }
        Ok(made.map(|raw| self.texture_of(raw, 0, space)))
    }

    /// Keeps `indices` from plain allocation of IDs of the type for the device's life: textures
    /// (type 0) and sampler states (1) in the runtime's spaces, which number them as Metal does,
    /// indirect command buffers (5) in the numbers this platform gives them and, below 2^23, in the
    /// runtime's, which park the thread for good where Metal spins and die where Metal goes on to
    /// hand out IDs twice ([`crate::gpu::icb`]). The runtime takes only a space's first call with
    /// an ID, before the space handed out one, and ignores any other, as Metal does; a call holding
    /// an ID of 2^32 or more it drops whole, leaving the reservation to a later call, as Metal
    /// does. An ID the runtime refuses dies: a texture or sampler state ID of 2^28 or more, which
    /// ended Metal's process, or a texture ID past what the device's texture arrays hold, which
    /// Metal takes; so does any other type, which core never reserves. Textures of exactly the
    /// guest's IDs the device reserved as it was adopted ([`adopt`]) are answered at once: they
    /// are reserved, and the runtime would ignore the call as a late one.
    fn reserve_resource_indices(&self, resource_type: u64, indices: &[u64]) -> RaiseResult<()> {
        let sel = "reserveResourceIndicesForResourceType:indices:indexCount:";
        if resource_type == 0
            && let Some(ids) = &self.guest_texture_ids
            && indices.iter().copied().eq(ids.clone())
        {
            return Ok(());
        }
        let kind = match resource_type {
            0 => rt::ResourceIdKind::Texture,
            1 => rt::ResourceIdKind::SamplerState,
            5 => return crate::gpu::icb::reserve(self, sel, indices),
            other => unrepresentable(sel, &format!("resource type {other}")),
        };
        let ids: Vec<rt::ResourceId> = indices.iter().map(|&id| rt::ResourceId(id)).collect();
        send(sel, Policy::Dies, || {
            self.raw.reserve_resource_ids(kind, &ids)
        })
    }

    /// Metal's description of the format, the runtime's
    /// ([`crate::gpu::format_info::description`]).
    fn pixel_format_info(&self, format: u64) -> PixelFormatInfo {
        plain(|| crate::gpu::format_info::description(&self.raw, format))
    }

    /// Metal's layout of the level, the runtime's ([`crate::gpu::format_info::level_layout`]).
    fn texture_level_info(&self, q: &TextureLevelQuery) -> TextureLevelInfo {
        plain(|| crate::gpu::format_info::level_layout(&self.raw, q))
    }

    /// Read on the first call for this device object, then shared: the heap-buffer query's length
    /// is the host's page size, as on Metal.
    fn caps(&self) -> Arc<DeviceCaps> {
        Arc::clone(
            self.caps
                .get_or_init(|| caps::read(&self.raw, self.registry_id, OsVm::new().page_size())),
        )
    }

    /// Read on each call ([`crate::gpu::identity`]).
    fn identity_caps(&self) -> IdentityCaps {
        crate::gpu::identity::read(&self.raw, &self.caps())
    }

    /// Always: the runtime's render pipeline descriptors hold the private state
    /// (`states::new_render_pipeline_state_full`).
    fn takes_private_render_state(&self) -> bool {
        true
    }

    fn own_memory_budget(&self) -> u64 {
        crate::gpu::identity::device_local_budget(&self.raw)
    }

    /// Always: the runtime links a kernel of OpenCL mode as Metal does, behind its three buffers
    /// (`states::full_compute_descriptor`).
    fn makes_opencl_mode_pipelines(&self) -> bool {
        true
    }

    fn pipeline_answers_identity(&self) -> Option<Vec<u8>> {
        Some(crate::gpu::identity::pipeline_answers_identity(&self.raw))
    }

    /// The runtime's buffer descriptor of the value: zero bytes, a copy of the contents, or,
    /// with `no_copy`, the contents themselves with the deallocator, at the pinned address when
    /// there is one. Nil as `newBufferWithLength:options:` is, for a pinned address the space
    /// cannot honour, and for no-copy contents of length 0, whose deallocator never runs, as
    /// Metal answers (measured: NULL contents of length 0). What the runtime's descriptor has no form for (no-copy without contents, a
    /// deallocator of a buffer that is not no-copy, copied contents of another length) dies.
    fn new_buffer(&self, desc: BufferDesc<'_>) -> RaiseResult<Option<VulkanBuffer>> {
        let sel = "newBufferWithDescriptor:";
        let held = descriptor_options(BUFFER_DESCRIPTOR_OPTIONS, &desc.options);
        let options = crate::gpu::resource_options(sel, held);
        let slot = desc.deallocator.map(|d| Arc::new(Mutex::new(Some(d))));
        if desc.no_copy && desc.length == 0 {
            let made = Ok(None);
            keep_unrun(&made, slot);
            return made;
        }
        let bytes = match (desc.contents, desc.no_copy) {
            (Some(bytes), true) => {
                metal_space::Bytes::NoCopy(u64::try_from(bytes.ptr().addr()).unwrap_or(0))
            }
            (None, true) => metal_space::Bytes::NoCopy(0),
            (_, false) => own_bytes(options),
        };
        let (placement, mut placing) =
            metal_space::placing(desc.length, bytes, desc.pinned_gpu_address);
        // Whether a no-copy buffer's pages are mapped read-only.
        let mut read_only = false;
        // The word that a no-copy buffer's import was made ([`crate::os::ImportMade`]).
        let mut import_made = None;
        let contents = match (desc.contents, desc.no_copy, &slot) {
            (None, false, None) => rt::BufferContents::Zero,
            (Some(bytes), false, None) if bytes.len() == desc.length => {
                // A NULL pointer dies whatever the length, as the no-copy creators' does.
                pointer(sel, bytes.ptr());
                // Read once, through the pointer, straight into the buffer's memory: guest memory
                // may change while it is read.
                rt::BufferContents::CopyHost(crate::gpu::resource::guest_bytes(sel, bytes))
            }
            (Some(bytes), true, slot) => {
                let start = pointer(sel, bytes.ptr());
                // The bytes, and what keeps them mapped until the runtime freed its import; the
                // deallocator is given the bytes' own address, at Metal's moment.
                let run = slot.as_ref().map(|slot| {
                    runtime_deallocator(slot, move |d: BufferDeallocator, p, len| {
                        d(HostPtr::from_addr(p.as_ptr().addr()), len);
                    })
                });
                let NoCopyImport {
                    at: imported,
                    run,
                    keep,
                    read_only: mapped_read_only,
                    vouched,
                    made,
                } = no_copy_release(
                    start,
                    bytes.len(),
                    run,
                    backing::ImportSite::NoCopyDescriptor,
                );
                read_only = mapped_read_only;
                import_made = Some(made);
                rt::BufferContents::NoCopy {
                    // SAFETY: `HostRange::new`'s contract keeps the bytes mapped until the
                    // deallocator runs, or for the buffer's life with none, and an unmap of
                    // core's through the platform's VM waits for the keep-alive; and a vouch is the
                    // table's record of the pages as they are mapped now.
                    memory: unsafe { crate::os::host_memory_vouched(imported, vouched) },
                    deallocator: run,
                    keep_alive: Some(keep),
                }
            }
            _ => unrepresentable(sel, "that descriptor's contents"),
        };
        let pool = pool_slot(sel, &self.raw, &mut placing)?;
        let descriptor =
            rt::BufferDescriptor::new(desc.length, options, contents).with_placement(placement);
        let descriptor = with_token(in_slot(descriptor, &pool), placing.token());
        let made = create_buffer(
            sel,
            placed_nil(&placing, Nil::BufferLength),
            placing.pin_refused(),
            || self.raw.new_buffer_with_descriptor(descriptor),
        );
        metal_space::let_go(pool);
        keep_unrun(&made, slot);
        if let (Ok(Some(_)), Some(import_made)) = (&made, import_made) {
            import_made.made(&self.raw);
        }
        made.map(|b| {
            b.map(|raw| {
                let backing = Backing::new(held & READ_ONLY_OPTION != 0);
                backing.imported(read_only, false);
                Obj::new(BufferObject {
                    raw,
                    device: self.clone_ref(),
                    backing,
                    heap: None,
                })
            })
        })
    }

    /// A heap of the value, at the pinned address when there is one. The descriptor's options
    /// are those Metal's descriptor holds after the value's setters ([`descriptor_options`]),
    /// and the heap's those of its modes, an untracked mode where the value names the default,
    /// as Metal's are (measured). Nil for a size of 0, for memory the device cannot allocate and
    /// for a pinned address the space cannot honour, as Metal answers; a `Managed` or
    /// `Memoryless` heap dies, as Metal asserts at it ("Heap Descriptor Validation"). A type the
    /// runtime lacks dies at `setType:`.
    fn new_heap(&self, desc: &HeapDesc) -> RaiseResult<HeapMade<VulkanPlatform>> {
        let sel = "newHeapWithDescriptor:";
        let heap_type = u64::try_from(desc.heap_type)
            .ok()
            .and_then(|raw| rt::HeapType::from_raw(raw).ok())
            .unwrap_or_else(|| {
                unrepresentable("setType:", &format!("heap type {}", desc.heap_type))
            });
        let held = descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options);
        let options = crate::gpu::resource_options(sel, held);
        // A sparse heap's ranges are its range token's, which the runtime drops once the heap,
        // its textures and the command buffers that hold any of them are gone; the runtime holds
        // none of the space for it. One the space refuses dies: the runtime would make it, and
        // Metal's answer is not measured ([`metal_space::placing_sparse`]).
        let (placement, placing, token) = match heap_type {
            rt::HeapType::Sparse => {
                let (placement, mut placing) =
                    metal_space::placing_sparse(desc.size, desc.pinned_gpu_address);
                if placing.refused() {
                    let at = match desc.pinned_gpu_address {
                        Some(pin) => format!("at the pin {pin:#x}"),
                        None => "with no room for its range".to_owned(),
                    };
                    unrepresentable(
                        sel,
                        &format!(
                            "a sparse heap of {:#x} bytes {at}, which the model of Metal's space \
                             refuses and nothing measured",
                            desc.size
                        ),
                    );
                }
                let token = placing.token();
                (placement, placing, token)
            }
            _ => {
                let (placement, mut placing) = metal_space::placing(
                    desc.size,
                    metal_space::Bytes::Heap,
                    desc.pinned_gpu_address,
                );
                let token = placing.token();
                (placement, placing, token)
            }
        };
        let nil = placed_nil(&placing, Nil::Heap);
        let descriptor =
            rt::HeapDescriptor::new(heap_type, desc.size, options).with_placement(placement);
        let storage = |made: Result<rt::Heap, rt::Error>| match made {
            Err(rt::Error::Invalid(rt::Invalid {
                problem: rt::Problem::StorageMode,
                ..
            })) => metal_aborts(sel, &format!("a heap of resource options {held:#x}")),
            made => made,
        };
        let heap = create(sel, Policy::Dies, nil, || {
            storage(match token {
                Some(token) => self.raw.new_heap_with_range_token(&descriptor, token),
                None => self.raw.new_heap(&descriptor),
            })
        })?;
        Ok(HeapMade {
            heap: heap.map(|raw| {
                Obj::new(HeapObject {
                    raw,
                    device: self.clone_ref(),
                    descriptor_size: desc.size,
                    backing: Backing::new(false),
                    placed: Mutex::default(),
                    space: placing.made(),
                })
            }),
            descriptor_resource_options: held,
        })
    }

    /// A heap whose memory is `bytes`, imported in place as a no-copy buffer's is
    /// ([`no_copy_release`]), at the address Metal's space gives it or the value's pin: its
    /// resources lie in `bytes` from their offsets, so the guest's writes there and the GPU's are
    /// each other's. Nil for a size of 0, for bytes
    /// short of the size rounded up to the runtime's granule, and as [`Self::new_heap`] answers
    /// for its address; storage other than `Shared` dies, as there. A type the runtime lacks
    /// dies at `setType:`.
    fn new_heap_over_host_memory(
        &self,
        bytes: HostRange<'_>,
        desc: &HeapDesc,
    ) -> RaiseResult<HeapMade<VulkanPlatform>> {
        let sel = "newHeapWithDescriptor:";
        let heap_type = u64::try_from(desc.heap_type)
            .ok()
            .and_then(|raw| rt::HeapType::from_raw(raw).ok())
            .unwrap_or_else(|| {
                unrepresentable("setType:", &format!("heap type {}", desc.heap_type))
            });
        let held = descriptor_options(HEAP_DESCRIPTOR_OPTIONS, &desc.options);
        let options = crate::gpu::resource_options(sel, held);
        let bit_24 = held & READ_ONLY_OPTION != 0;
        let covered = desc.size.checked_next_multiple_of(backing::HEAP_GRANULE);
        if desc.size == 0 || covered.is_none_or(|need| bytes.len() < need) {
            return Ok(HeapMade {
                heap: None,
                descriptor_resource_options: held,
            });
        }
        let (placement, mut placing) =
            metal_space::placing(desc.size, metal_space::Bytes::Heap, desc.pinned_gpu_address);
        let token = placing.token();
        let nil = placed_nil(&placing, Nil::Heap);
        let descriptor =
            rt::HeapDescriptor::new(heap_type, desc.size, options).with_placement(placement);
        let start = pointer(sel, bytes.ptr());
        // The bytes, and what keeps them mapped until the runtime freed its import.
        let NoCopyImport {
            at: imported,
            keep,
            read_only,
            vouched,
            made: import_made,
            ..
        } = no_copy_release(start, bytes.len(), None, backing::ImportSite::Heap);
        let heap = create(sel, Policy::Dies, nil, || {
            // SAFETY: `HostRange::new`'s contract keeps the bytes mapped for the heap's life, and
            // an unmap of core's through the platform's VM waits for the keep-alive; they cover
            // the heap's size rounded up to the runtime's granule (checked above) and start on a
            // page, the runtime's import unit on this host; and a vouch is the table's record of
            // the pages as they are mapped now.
            let made = unsafe {
                let memory = crate::os::host_memory_vouched(imported, vouched);
                match token {
                    Some(token) => self.raw.new_heap_with_bytes_no_copy_and_range_token(
                        memory,
                        &descriptor,
                        None,
                        Some(keep),
                        token,
                    ),
                    None => {
                        self.raw
                            .new_heap_with_bytes_no_copy(memory, &descriptor, None, Some(keep))
                    }
                }
            };
            match made {
                Err(rt::Error::Invalid(rt::Invalid {
                    problem: rt::Problem::StorageMode,
                    ..
                })) => metal_aborts(sel, &format!("a heap of resource options {held:#x}")),
                made => made,
            }
        })?;
        if heap.is_some() {
            import_made.made(&self.raw);
        }
        Ok(HeapMade {
            heap: heap.map(|raw| {
                let backing = Backing::new(false);
                backing.imported(read_only, bit_24);
                Obj::new(HeapObject {
                    raw,
                    device: self.clone_ref(),
                    descriptor_size: desc.size,
                    backing,
                    placed: Mutex::default(),
                    space: placing.made(),
                })
            }),
            descriptor_resource_options: held,
        })
    }

    /// One object for equal requests while it lives, placed at the value's resource index
    /// ([`super::states`]).
    fn new_sampler_state(&self, desc: &SamplerDesc) -> RaiseResult<Option<VulkanSamplerState>> {
        super::states::new_sampler_state(self, desc)
    }

    /// A state of any descriptor ([`super::states`]).
    fn new_depth_stencil_state(
        &self,
        desc: &DepthStencilDesc,
    ) -> RaiseResult<Option<VulkanDepthStencilState>> {
        super::states::new_depth_stencil_state(self, desc)
    }

    fn new_indirect_command_buffer(
        &self,
        desc: &IcbDesc,
        max_count: u64,
        options: u64,
    ) -> RaiseResult<Option<VulkanIcb>> {
        crate::gpu::icb::new_indirect_command_buffer(self, desc, max_count, options)
    }

    /// The display's present pipeline ([`super::states`]).
    fn new_render_pipeline_state(
        &self,
        desc: &RenderPipelineDesc<'_, VulkanPlatform>,
    ) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
        super::states::new_render_pipeline_state(self, desc)
    }

    /// The graphics lane's whole descriptor ([`super::states::new_render_pipeline_state_full`]).
    fn new_render_pipeline_state_full(
        &self,
        desc: &FullRenderPipelineDesc<'_, VulkanPlatform>,
    ) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
        super::states::new_render_pipeline_state_full(self, desc)
    }

    /// The graphics lane's whole descriptor: the compile's refusal of a function or descriptor it
    /// finds wrong, and a payload past Metal's, are the `NSError`; every other refusal dies at
    /// the send ([`super::states::new_mesh_render_pipeline_state_full`]).
    fn new_mesh_render_pipeline_state_full(
        &self,
        desc: &FullMeshRenderPipelineDesc<'_, VulkanPlatform>,
    ) -> Result<Option<VulkanRenderPipeline>, NsErrorText> {
        super::states::new_mesh_render_pipeline_state_full(self, desc)
    }

    /// The graphics lane's whole descriptor: the compile's refusal of a function or descriptor it
    /// finds wrong is the `NSError`, every other refusal dies at the send ([`compile_error()`]).
    fn new_compute_pipeline_state_full(
        &self,
        desc: &FullComputePipelineDesc<'_, VulkanPlatform>,
    ) -> Result<Option<VulkanComputePipeline>, NsErrorText> {
        let sel = "newComputePipelineStateWithDescriptor:options:reflection:error:";
        let d = super::states::full_compute_descriptor(sel, desc);
        let mut options = rt::PipelineOptions::default();
        options.label = desc.label.map(str::to_owned);
        timing::creation(
            sel,
            timing::DISPATCH_LATER,
            || match plain_at(sel, || self.raw.new_compute_pipeline_state(&d, &options)) {
                Ok(state) => Ok(Some(Obj::new(ComputePipelineObject::new(state)))),
                Err(e) => Err(compile_error(sel, e, Some(&d))),
            },
            || format!("kernel {}", timing::stage(&d.kernel)),
            |made| timing::answer(made, |s| timing::state(s.unique_identifier)),
        )
    }
}

/// The runtime's descriptor of a texture of `fields` over a surface's plane, made by `sel` on
/// `device`: Metal's `newTextureWithDescriptor:iosurface:plane:`, over a buffer that stands for
/// the surface ([`crate::gpu::resource`]'s `new_texture_over_surface_plane`).
/// `None` for a format of [`crate::gpu::no_texture`], nil once Metal's validation passes, before
/// the plane and the surface are looked at: the runtime's heap size query validates its
/// descriptor as the runtime's creators do (over a surface's plane too), and answers 0 bytes for
/// it. Any other format, the two-plane ones among them, is the runtime's to make
/// ([`Maker::Surface`]).
pub(crate) fn surface_plane_descriptor(
    device: &rt::Device,
    sel: &'static str,
    fields: &TextureFields,
) -> RaiseResult<Option<rt::TextureDescriptor>> {
    if fields.no_texture_format.is_some()
        && let Some(pf) = fields.formatless()
    {
        let desc = crate::gpu::no_texture::descriptor(sel, fields, pf);
        send(sel, Policy::Dies, || {
            device.heap_texture_size_and_align(&desc)
        })?;
        return Ok(None);
    }
    Ok(Some(fields.descriptor(sel, Maker::Surface)))
}

/// What a texture over a surface's plane reads of the surface besides the plane
/// ([`plane_facts_taken`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlaneFacts {
    /// The surface's YCbCr facts: its OSType, its size, its plane 1, and its YCbCr matrix.
    YCbCr,
    /// Its OSType, its size and its plane 1 alone.
    Planes,
}

/// The facts of its surface a texture of `pf` over a plane of it takes: the YCbCr facts for a
/// two-plane YCbCr format the runtime reads ([`crate::gpu::ycbcr::runtime_makes`]), the planes'
/// for a two-plane XR10 format ([`two_plane_xr10`]), which Metal reads alike ('b3a8'), and none
/// for any other.
pub(crate) fn plane_facts_taken(pf: metal2vulkan::metal::PixelFormat) -> Option<PlaneFacts> {
    if crate::gpu::ycbcr::runtime_makes(pf) {
        Some(PlaneFacts::YCbCr)
    } else if two_plane_xr10(pf) {
        Some(PlaneFacts::Planes)
    } else {
        None
    }
}

/// The runtime's texture of `desc` (of `fields`) over `plane` of `buffer`, which stands for the
/// surface (`Buffer::new_texture_with_iosurface_plane`), made by `sel`: what the runtime refuses
/// dies here, naming the refusal, and its nil is nil.
pub(crate) fn texture_over_plane(
    sel: &'static str,
    fields: &TextureFields,
    buffer: &rt::Buffer,
    desc: &rt::TextureDescriptor,
    plane: rt::IoSurfacePlane,
) -> RaiseResult<Option<rt::Texture>> {
    let format = fields.raw_pixel_format();
    create(sel, Policy::Dies, Nil::Never, || {
        (buffer.new_texture_with_iosurface_plane(desc, plane))
            .map_err(|e| created_format_refused(sel, "a texture of", format, e))
    })
}

/// Whether `pf` is a two-plane XR10 format: `RGB10A8_2P_XR10` (550) or its sRGB twin (551),
/// which Metal makes over a surface's two planes, the color words in plane 0 and the alpha bytes
/// in plane 1.
pub(crate) fn two_plane_xr10(pf: metal2vulkan::metal::PixelFormat) -> bool {
    use metal2vulkan::metal::PixelFormat;
    matches!(
        pf,
        PixelFormat::RGB10A82PXR10 | PixelFormat::RGB10A82PXR10SRGB
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::LazyLock;

    use crate::core::platform::Handle;
    use crate::core::platform::device::GpuDevice;
    use crate::core::raise::death::DeathKind;
    use crate::core::testkit::capture::catch_death;

    use super::*;

    /// The system default device, one handle to the runtime device the platform keeps for the
    /// process ([`DEVICES`]).
    static DEVICE: LazyLock<VulkanDevice> = LazyLock::new(|| {
        VulkanPlatform::system_default_device().expect("a Vulkan device (MoltenVK)")
    });

    /// The system default device (MoltenVK on the suite's host).
    pub(crate) fn dev() -> VulkanDevice {
        DEVICE.clone_ref()
    }

    /// Whether `d` stores a `Depth32Float` depth outside `[0, 1]` clamped into `[0, 1]`
    /// ([`clamped_depth`]), where Metal keeps it: a device without
    /// `VK_EXT_depth_range_unrestricted` (MoltenVK, ANV). The runtime announces
    /// `restricted-depth-upload` at every upload into such a depth there, whatever values it
    /// writes: a blit from a buffer, and a heap texture's copy in of the heap's bytes as it is
    /// made over the guest's pages or in core's legacy heap.
    #[cfg(target_os = "macos")]
    pub(crate) fn clamps_depth(d: &VulkanDevice) -> bool {
        use metal2vulkan::device::VkExtension;
        !(d.raw.enabled().extensions).contains(&VkExtension::ExtDepthRangeUnrestricted)
    }

    /// The `Depth32Float` depth `bits` as a device that clamps ([`clamps_depth`]) stores it: a
    /// NaN of either sign as 1.0, -0.0 as it is, any other negative value (a negative denormal
    /// included) as 0, one above 1.0 as 1.0, and any other as it is, a positive denormal included.
    #[cfg(target_os = "macos")]
    pub(crate) fn clamped_depth(bits: u32) -> u32 {
        let depth = f32::from_bits(bits);
        if depth.is_nan() || depth > 1.0 {
            1.0f32.to_bits()
        } else if depth.is_sign_negative() && bits != (-0.0f32).to_bits() {
            0
        } else {
            bits
        }
    }

    /// Runs `call`, which uploads into a `Depth32Float` depth, on `d`: as it is where `d` keeps a
    /// depth outside `[0, 1]`. On a device that clamps ([`clamps_depth`]), `call` dies of
    /// `restricted-depth-upload` at `site` in strict mode, and runs again under the default
    /// ([`crate::answer::relaxed`]), which answers here: it relays that kind of
    /// [`crate::answer::DIES`] and no other.
    #[cfg(target_os = "macos")]
    pub(crate) fn uploading_depth<T>(d: &VulkanDevice, site: &str, call: impl Fn() -> T) -> T {
        if !clamps_depth(d) {
            return call();
        }
        let restricted = rt::Degradation::RestrictedDepthUpload;
        let died = crate::answer::dies_of(restricted, &call);
        assert_eq!(died.file, site, "where it dies of {}", restricted.code());
        let before = crate::answer::relayed_counts();
        let answer = crate::answer::relaxed(&call);
        let relayed: Vec<rt::Degradation> = (crate::answer::relayed_counts().into_iter())
            .filter(|count| !before.contains(count))
            .map(|(kind, _)| kind)
            .collect();
        assert_eq!(relayed, [restricted], "the kinds of DIES relayed");
        answer
    }

    /// Waits, up to a minute each, for the checks the device starts on threads of its own as it
    /// is adopted: its in-place, queue-switch and submission checks. They make, import and submit
    /// through the program's paths, the runtime counts their imports with the program's, and the
    /// device's first makings and imports can wait behind them. A test that counts the device's
    /// work, or waits for the runtime's thread, settles them first. Linux's and Windows's, as are
    /// those tests.
    #[cfg(any(target_os = "linux", windows))]
    pub(crate) fn settle_device_checks(d: &VulkanDevice) {
        let wait = std::time::Duration::from_secs(60);
        let _ = d.raw.settle_in_place_checks(wait);
        let _ = d.raw.settle_queue_switch_check(wait);
        let _ = d.raw.settle_submit_checks(wait);
    }

    /// The one function of a fixture metallib.
    pub(crate) fn fixture_function(bytes: &[u8]) -> VulkanFunction {
        use crate::core::platform::resource::Library as _;
        let library = dev().new_library(bytes).unwrap().unwrap().unwrap();
        let name = library.function_names().remove(0);
        library
            .new_function(OnFault::Return, &name)
            .unwrap()
            .unwrap()
    }

    pub(crate) const STAGEIN: &[u8] =
        include_bytes!("../../../../tests/shaders/compute_stagein.metallib");

    /// Colour attachments, each given by its properties' numbers and values: 0 its index, 1 its
    /// pixel format, 2 whether it blends, 3 to 8 its source RGB factor, destination RGB factor,
    /// RGB operation, source alpha factor, destination alpha factor and alpha operation, and 9
    /// its write mask; Metal's defaults for those unnamed.
    pub(crate) fn colours_of(
        colours: &[&[(u8, u32)]],
    ) -> Vec<(u64, crate::core::platform::values::ColorAttachmentDesc)> {
        use crate::core::platform::values::ColorAttachmentDesc;
        colours
            .iter()
            .map(|properties| {
                let (mut index, mut c) = (0, ColorAttachmentDesc::default());
                for &(key, value) in *properties {
                    let value = u64::from(value);
                    match key {
                        0 => index = value,
                        1 => c.pixel_format = value,
                        2 => c.blending_enabled = value != 0,
                        3 => c.source_rgb_blend_factor = value,
                        4 => c.destination_rgb_blend_factor = value,
                        5 => c.rgb_blend_operation = value,
                        6 => c.source_alpha_blend_factor = value,
                        7 => c.destination_alpha_blend_factor = value,
                        8 => c.alpha_blend_operation = value,
                        9 => c.write_mask = value,
                        _ => panic!("no colour attachment property {key}"),
                    }
                }
                (index, c)
            })
            .collect()
    }

    /// The whole render value of `vertex` and `fragment` into `colours` ([`colours_of`]) at
    /// `samples` samples, Metal's defaults elsewhere.
    #[cfg(target_os = "macos")]
    pub(crate) fn render_value<'a, P: crate::core::platform::Platform>(
        (vertex, fragment): (&'a P::Function, &'a P::Function),
        samples: u64,
        colours: &[&[(u8, u32)]],
    ) -> crate::core::platform::values::FullRenderPipelineDesc<'a, P> {
        use crate::core::platform::values::FullRenderPipelineDesc;
        FullRenderPipelineDesc {
            colors: colours_of(colours),
            raster_sample_count: samples,
            ..FullRenderPipelineDesc::new(vertex, Some(fragment))
        }
    }

    /// The platform's device options with a shader cache of their own, as
    /// `metal_tests::device_with_own_cache` makes its device with.
    pub(crate) fn options_with_own_cache() -> (rt::DeviceOptions, OwnCache) {
        static MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let home = std::env::var_os("HOME").expect("a home directory");
        let n = MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::path::PathBuf::from(home).join(format!(
            ".cache/reims-vgpu-vulkan-tests-own/{}-{n}",
            std::process::id()
        ));
        // One a process of the same ID left: this process's own, and emptied.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut options = crate::platform::device_options().clone();
        options.shader_cache = rt::ShaderCacheOptions::in_directory(&dir);
        (options, OwnCache(dir))
    }

    /// The directory of [`options_with_own_cache`]'s shader cache, removed as it drops.
    pub(crate) struct OwnCache(std::path::PathBuf);

    impl Drop for OwnCache {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A device the platform adopts starts the runtime's check of textures over host pages, with
    /// no texture made ([`start_in_place_checks`]): a device of a shader cache of its own, empty,
    /// so no record holds the verdicts, adopted as the first of its process (nextest's, one a
    /// test), ends the check. The platform then lets go of it before the process exits: kept to
    /// the exit right after its check ended, it ended the exit with `SIGSEGV` in NVIDIA's driver
    /// in 6 of 80 runs on an RTX 3090 ([`crate::let_go`]).
    #[test]
    fn an_adopted_device_starts_its_in_place_check() {
        let (options, _cache) = options_with_own_cache();
        let raw = rt::Device::system_default_with(&options).unwrap();
        assert!(raw.in_place_check_took().is_none(), "no check ran before");
        assert!(
            kept(registry_id(&raw)).is_none(),
            "the test's process adopted a device before"
        );
        use crate::core::platform::os::Clock as _;
        let device = adopt(raw);
        let clock = crate::os::OsClock::new();
        let began = clock.now_mono();
        while device.raw.in_place_check_took().is_none() {
            let waited = clock.now_mono().saturating_duration_since(began);
            assert!(
                waited < std::time::Duration::from_secs(120),
                "no check ended in 120 s: {:?}",
                device.raw.in_place_check_report()
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        drop(device);
        let bound = std::time::Duration::from_secs(30);
        assert_eq!(crate::let_go::let_go_of_devices(bound), Ok(1));
    }

    /// The platform lets go of its devices once nothing beyond its own keeps holds one
    /// ([`crate::let_go`]). While a buffer of the device lives, the let-go waits out its bound,
    /// names the device, and lets go of nothing. With the buffer gone it lets go of the device,
    /// the library of its shaders and the no-draw pipeline the platform made of it, and the device
    /// whose answers describe formats, and ends the device's threads: the device object goes. The
    /// next adopt of the same GPU makes a new device, and that one is let go of in turn.
    #[test]
    fn the_platform_lets_go_of_its_devices_once_nothing_else_holds_one() {
        use crate::core::platform::values::OptionSet;
        use std::time::Duration;

        let (options, _cache) = options_with_own_cache();
        let raw = rt::Device::system_default_with(&options).unwrap();
        let id = registry_id(&raw);
        assert!(
            kept(id).is_none(),
            "the test's process adopted a device before"
        );
        let device = adopt(raw);
        let object = device.downgrade();
        drop(crate::gpu::shaders::library(&device).unwrap());
        drop(crate::gpu::rate_map::no_draw("a test's no-draw", &device));
        crate::gpu::format_info::described("a test's format", 70);
        let buffer = (device.new_buffer(BufferDesc {
            length: 0x1000,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: None,
            contents: None,
            no_copy: false,
            deallocator: None,
        }))
        .unwrap()
        .unwrap();
        drop(device);
        let held = crate::let_go::let_go_of_devices(Duration::from_millis(50)).unwrap_err();
        assert!(held.contains(&format!("registry ID {id:#x}")), "{held}");
        assert!(kept(id).is_some(), "nothing let go");
        drop(buffer);
        let bound = Duration::from_secs(30);
        assert_eq!(crate::let_go::let_go_of_devices(bound), Ok(1));
        assert!(object.upgrade().is_none(), "the device object went");
        assert!(kept(id).is_none());
        assert!(crate::gpu::shaders::take_loaded().is_empty());
        assert!(crate::gpu::rate_map::take_no_draws().is_empty());
        assert!(crate::platform::take_answering_device().is_none());
        assert!(crate::gpu::metal_space::take_sampler_ids().is_none());
        let again = adopt(rt::Device::system_default_with(&options).unwrap());
        assert!(kept(id).is_some());
        drop(again);
        assert_eq!(crate::let_go::let_go_of_devices(bound), Ok(1));
    }

    /// Descriptors and a function's argument buffer are laid out as Metal's own encoder lays them
    /// out (measured on the M3 Max with the same descriptors and the same probe kernel): the
    /// runtime's layout, Metal's rules, reaches the guest unchanged
    #[test]
    fn argument_layouts_are_metals() {
        use crate::core::platform::resource::Library as _;
        use crate::core::platform::types::ArgumentKind as K;
        let d = dev();
        let arg = |data_type: u64, index: u64, array_length: u64| ArgumentDesc {
            data_type,
            index,
            array_length,
            texture_type: 2,
            ..ArgumentDesc::default()
        };
        let layout = d
            .argument_layout(&[arg(58, 0, 0), arg(60, 1, 4), arg(6, 5, 0), arg(3, 6, 0)])
            .unwrap()
            .unwrap();
        let got: Vec<_> = layout
            .members
            .iter()
            .map(|m| (m.id, m.count, m.offset, m.stride, m.size, m.kind))
            .collect();
        assert_eq!(
            got,
            [
                (0, 1, 0, 8, 8, K::Texture),
                (1, 4, 8, 8, 8, K::Buffer),
                (5, 1, 48, 16, 16, K::Constant),
                (6, 1, 64, 4, 4, K::Constant),
            ]
        );
        assert_eq!((layout.encoded_length, layout.alignment), (80, 16));

        const PROBE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/argument_probe.metallib"));
        let library = d.new_library(PROBE).unwrap().unwrap().unwrap();
        let function = library
            .new_function(OnFault::Die, "reims_vgpu_argument_probe")
            .unwrap()
            .unwrap();
        let layout = d.function_argument_layout(&function, 0).unwrap().unwrap();
        let got: Vec<_> = layout
            .members
            .iter()
            .map(|m| (m.id, m.offset, m.kind, m.layout.is_some()))
            .collect();
        assert_eq!(
            got,
            [
                (0, 0, K::Texture, false),
                (1, 8, K::Buffer, true),
                (2, 16, K::Sampler, false),
                (3, 32, K::Constant, false),
                (4, 48, K::Buffer, false),
            ]
        );
        assert_eq!((layout.encoded_length, layout.alignment), (64, 16));
        let inner = layout.members[1].layout.as_ref().unwrap();
        let got: Vec<_> = inner
            .members
            .iter()
            .map(|m| (m.id, m.offset, m.kind))
            .collect();
        assert_eq!(got, [(0, 0, K::Texture), (1, 8, K::Sampler)]);
        assert_eq!(inner.encoded_length, 16);
    }

    /// The crate's directory, which cargo and nextest give a test process: read at run time, so
    /// the tests are the same binary in every worktree (docs/development.md, "Building through
    /// kache").
    pub(crate) fn crate_dir() -> std::path::PathBuf {
        std::env::var_os("CARGO_MANIFEST_DIR")
            .map(std::path::PathBuf::from)
            .expect("CARGO_MANIFEST_DIR: run the tests with cargo or nextest")
    }

    /// The directory of the test shaders, `tests/shaders`.
    pub(crate) fn fixture_shaders() -> std::path::PathBuf {
        crate_dir().join("../../tests/shaders")
    }

    /// The build script's `OUT_DIR`, which cargo and nextest give a test process of a package
    /// with a build script too: read at run time, as [`crate_dir`] is. What the tests compile in
    /// from it, they include (`include_bytes!`), which kache keys by its bytes; a file whose
    /// bytes name the directory (an import metallib's install name, `build.rs`) is read here.
    #[cfg(target_os = "macos")]
    pub(crate) fn out_dir() -> std::path::PathBuf {
        std::env::var_os("OUT_DIR")
            .map(std::path::PathBuf::from)
            .expect("OUT_DIR: run the tests with cargo or nextest")
    }

    /// A device of the system default's physical device made with `options`, after the shared
    /// device ([`dev`]) and not kept for [`adopt`].
    pub(crate) fn device_of_options(options: &rt::DeviceOptions) -> VulkanDevice {
        drop(dev());
        let raw = rt::Device::system_default_with(options).unwrap();
        Obj::new(DeviceObject {
            registry_id: registry_id(&raw),
            raw,
            caps: OnceLock::new(),
            awake: Default::default(),
            threads: Default::default(),
            rows_pitches: Mutex::default(),
            guest_texture_ids: None,
        })
    }

    /// A buffer pinned where another's span lies is nil, and the line it logs names the runtime's
    /// refusal: the held range the pin met.
    #[test]
    fn a_pinned_buffer_the_runtime_refuses_logs_why() {
        use crate::core::platform::resource::Buffer;
        use crate::core::platform::values::OptionSet;
        use crate::core::testkit::logs;
        let d = dev();
        let pin = metal_space::END / 4 + 0x100_0000;
        let pinned = || BufferDesc {
            length: 0x1000,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: Some(pin),
            contents: None,
            no_copy: false,
            deallocator: None,
        };
        let first = d.new_buffer(pinned()).unwrap().unwrap();
        assert_eq!(first.gpu_address(), pin);
        let logs = logs::capture();
        assert!(d.new_buffer(pinned()).unwrap().is_none());
        // The device's own threads log beside the test: its awake thread's first line, written
        // as the thread starts, can land after the capture began. The lines of the creator's
        // target alone.
        let lines: Vec<_> = (logs.take().into_iter())
            .filter(|(_, target, _)| target == "reims_vgpu_vulkan::gpu")
            .collect();
        assert_eq!(lines.len(), 1, "{lines:?}");
        let (level, _, line) = &lines[0];
        assert_eq!(*level, log::Level::Warn);
        assert!(
            line.starts_with(
                "newBufferWithDescriptor: answered nil: Metal's space refused the pin: Held {"
            ) && line.contains(&format!("held: {pin:x}")),
            "{line}"
        );
    }

    /// A Shared heap over host memory lies in it, in place: a buffer placed at an offset is the
    /// bytes there, at the heap's pin plus the offset, and the CPU's writes through either and
    /// the GPU's fill of the buffer land in them. Bytes short of the heap's size rounded up to
    /// the runtime's granule make none.
    #[test]
    fn a_heap_over_host_memory_lies_in_its_bytes() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder};
        use crate::core::platform::command::{CommandBuffer, CommandQueue};
        use crate::core::platform::resource::{Buffer, Heap};
        use crate::core::platform::types::NsRange;
        use crate::core::platform::values::{HeapDesc, OptionSet};
        let d = dev();
        // Private anonymous memory, as the graphics device's region is, kept for the process.
        let layout = std::alloc::Layout::from_size_align(0x2_0000, 0x4000).unwrap();
        // SAFETY: a layout of non-zero size.
        let mem = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!mem.is_null());
        let at = |offset: usize| {
            // SAFETY: inside the allocation, which the process keeps.
            unsafe { mem.add(offset) }
        };
        let over = |length: u64| {
            // SAFETY: the allocation stays for the process.
            unsafe { HostRange::new(HostPtr::from_ptr(mem), length) }
        };
        let pin = metal_space::END / 4;
        let desc = HeapDesc {
            size: 0x1_0001,
            heap_type: 1,
            options: vec![OptionSet::StorageMode(0)],
            pinned_gpu_address: Some(pin),
        };
        let short = d.new_heap_over_host_memory(over(0x1_0000), &desc).unwrap();
        assert!(short.heap.is_none(), "short of the granule");
        let heap = d
            .new_heap_over_host_memory(over(0x1_4000), &desc)
            .unwrap()
            .heap
            .unwrap();
        assert_eq!(heap.gpu_address(), pin);
        let buffer = heap
            .new_buffer_of_length(0x1000, 0, 0x4000)
            .unwrap()
            .unwrap();
        assert_eq!(buffer.contents().addr(), at(0x4000).addr());
        assert_eq!(buffer.gpu_address(), pin + 0x4000);
        // SAFETY: the buffer's bytes are the allocation's, which nothing else touches.
        unsafe {
            at(0x4000).write_bytes(0xa1, 0x800);
            assert_eq!(buffer.contents().as_ptr::<u8>().read(), 0xa1);
        }
        let queue = d.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        for command in [
            BlitCmd::FillBuffer {
                buffer: &buffer,
                range: NsRange::new(0x800, 0x800),
                value: 0x5c,
            },
            BlitCmd::EndEncoding,
        ] {
            e.encode(OnFault::Return, command).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status().0, 4, "completed");
        // SAFETY: as above, the command buffer done.
        unsafe {
            assert_eq!(at(0x4000).read(), 0xa1, "the CPU's bytes stay");
            assert_eq!(at(0x47ff).read(), 0xa1);
            assert_eq!(
                at(0x4800).read(),
                0x5c,
                "the GPU's fill lands in the heap's bytes"
            );
            assert_eq!(at(0x4fff).read(), 0x5c);
            assert_eq!(at(0x5000).read(), 0, "past the buffer");
        }
    }

    /// A texture whose rows lie with bytes between them, over a Private buffer that declares those
    /// bytes unobserved (`new_buffer_of_length_padding_unobserved`, the graphics lane's laid-out
    /// device caches), is an image over the buffer's memory that holds them, where the device's
    /// own linear images lay their rows at the texture's pitch: the runtime counts it among the
    /// images that do (`unobserved_padding_aliases`) and shadows nothing. Over a buffer made as
    /// any other the same texture, which renders, is an image over the buffer's memory that keeps
    /// the padding (`kept_padding_aliases`) where the device's check of renders over its pooled
    /// memory passed for rows that end as these do (its report's `Pooled renders` line, read
    /// before the textures are made, as a verdict still pending makes a shadow), and a shadow
    /// (`BytesBetweenRows`, counted among the shadowed textures) where it did not. The rows
    /// lie as the guest's surfaces' do on a host that answers its linear row
    /// alignment (`linear_row_alignment`): one texel past the alignment, a row of `BGRA8Unorm`
    /// texels 4 bytes past it, in rows twice it apart. Where it is past Metal's 16 bytes the
    /// device's images lay their rows at its multiples (the runtime's probe of them: 256 on
    /// RADV), so at the texture's pitch. At 16 bytes a device may lay them tighter (ANV lays them
    /// 4 bytes apart), its image of the texture lying at another pitch, and the texture is a
    /// shadow over either buffer
    #[test]
    fn a_texture_over_a_buffer_with_padding_unobserved_lies_over_its_memory() {
        use crate::core::platform::Platform as _;
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::values::TextureDesc;

        use crate::platform::tests::platform;

        const PRIVATE: u64 = 0x20;
        let d = dev();
        let align = d.linear_row_alignment();
        assert!(align >= 16 && align.is_power_of_two(), "{align}");
        let (width, height, pitch) = (align / 4 + 1, 8, 2 * align);
        // The checks the device starts make images of their own: they end before the counts are
        // read, so only this test's textures move them.
        #[cfg(any(target_os = "linux", windows))]
        settle_device_checks(&d);
        let desc = platform().texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width,
            height,
            usage: 0x5,
            resource_options: Some(PRIVATE),
            ..TextureDesc::default()
        });
        let counts = || {
            let counts = d.raw.work_counts();
            (
                counts.unobserved_padding_aliases,
                counts.shadowed_textures + counts.shadows_left_unmade - counts.shadows_made_at_use,
                counts.kept_padding_aliases,
            )
        };
        let texture_over = |buffer: Option<VulkanBuffer>| {
            let before = counts();
            let texture = (buffer.expect("a buffer")).new_texture(OnFault::Return, &desc, 0, pitch);
            assert!(matches!(texture, Ok(Some(_))), "a texture");
            let after = counts();
            (after.0 - before.0, after.1 - before.1, after.2 - before.2)
        };
        // The check's verdict for renders over pooled memory whose rows end as these do: in the
        // lines of the device's line size, or on them.
        let line = d.raw.in_place_check_line();
        let row = width * 4;
        let rows_on_lines = line > 0
            && (0..height).all(|k| (k * pitch) % line == 0 && (k * pitch + row) % line == 0);
        let verdict = if rows_on_lines {
            "Pooled renders, rows on lines: "
        } else {
            "Pooled renders: "
        };
        let renders_kept = d
            .raw
            .in_place_check_report()
            .iter()
            .any(|l| l.strip_prefix(verdict) == Some("Passed"));
        let length = pitch * height;
        let declared =
            d.new_buffer_of_length_padding_unobserved(OnFault::Return, length, PRIVATE, None);
        let declared = texture_over(declared.unwrap());
        let plain = texture_over(
            d.new_buffer_of_length(OnFault::Return, length, PRIVATE)
                .unwrap(),
        );
        let kept_or_shadow = if renders_kept { (0, 0, 1) } else { (0, 1, 0) };
        assert_eq!(
            plain, kept_or_shadow,
            "over a buffer made as any other: its memory where the check of pooled renders \
             passed ({verdict}{renders_kept}), else a shadow"
        );
        if align > 16 {
            assert_eq!(declared, (1, 0, 0), "over the buffer's memory, at {align}");
        } else {
            assert!(
                declared == (1, 0, 0) || declared == (0, 1, 0),
                "over its memory or a shadow: {declared:?}"
            );
        }
    }

    /// The device's pitch of a surface plane's rows (`linear_rows_pitch`) is the runtime's pitch
    /// of its linear image of a texture of each texel size that divides the row, and a texture
    /// whose rows lie that far apart over a Private buffer that declares the bytes between them
    /// unobserved (a device cache laid out at the GPU's pitch) is an image over the buffer's
    /// memory, not a shadow: rows of 88 texels of `RGBA16Float` (704 bytes, CoreAnimation's
    /// rows of such a surface) and of `BGRA8Unorm` (352), 8 high, that only sample, as the
    /// textures over macOS 26.7.2's caches the runtime shadowed for their pitch on an RX 7900 XTX
    /// do. It lies over the padding as unobserved only where the pitch passes the row: a device
    /// whose images lay such rows side by side (MoltenVK's on an M3 Max and ANV's on Iris Xe lay
    /// the 704-byte rows so, measured) lays the texture over the buffer as it is. Asked again it answers the same. A device whose
    /// formats' images lay such rows apart otherwise answers none, and promises nothing. A pitch
    /// off Metal's 16-byte linear alignment, at which the runtime makes no texture over a buffer,
    /// it answers rounded up to the device's line of memory, at which the texture lies over the
    /// buffer: ANV's images lay the rows of 70 `BGRA8Unorm` texels (280 bytes) side by side
    /// (measured), and 73 (292 bytes) was the pitch of the device caches whose textures a macOS
    /// 14.8.9 guest's login had refused on Iris Xe
    #[test]
    fn a_texture_at_the_device_s_pitch_of_a_plane_s_rows_lies_over_its_memory() {
        use crate::core::platform::Platform as _;
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::values::TextureDesc;
        use metal2vulkan::metal::PixelFormat as Pf;

        use crate::platform::tests::platform;

        const PRIVATE: u64 = 0x20;
        let d = dev();
        #[cfg(any(target_os = "linux", windows))]
        settle_device_checks(&d);
        let counts = || {
            let counts = d.raw.work_counts();
            (
                counts.unobserved_padding_aliases,
                counts.shadowed_textures + counts.shadows_left_unmade - counts.shadows_made_at_use,
            )
        };
        let unit = d
            .raw
            .in_place_check_line()
            .max(caps::METAL_LINEAR_ALIGNMENT);
        for (pixel_format, texel, width) in
            [(115u64, 8u64, 88u64), (80, 4, 88), (80, 4, 70), (80, 4, 73)]
        {
            let height = 8;
            let row = width * texel;
            let Some(pitch) = d.linear_rows_pitch(row, height) else {
                continue;
            };
            assert!(pitch >= row, "{pitch} for {row}");
            assert!(
                pitch.is_multiple_of(caps::METAL_LINEAR_ALIGNMENT),
                "{pitch} for {row}: off Metal's linear alignment"
            );
            assert_eq!(d.linear_rows_pitch(row, height), Some(pitch), "asked again");
            for (size, pf) in [
                (1, Pf::R8Unorm),
                (2, Pf::RG8Unorm),
                (4, Pf::BGRA8Unorm),
                (8, Pf::RGBA16Float),
                (16, Pf::RGBA32Float),
            ] {
                if !row.is_multiple_of(size) {
                    continue;
                }
                let mut descriptor = rt::TextureDescriptor::default();
                descriptor.pixel_format = pf;
                descriptor.width = row / size;
                descriptor.height = height;
                descriptor.resource_options = rt::ResourceOptions::private();
                let runtime = d.raw.linear_texture_bytes_per_row(&descriptor).unwrap();
                let off = |p: u64| !p.is_multiple_of(caps::METAL_LINEAR_ALIGNMENT);
                assert!(
                    runtime
                        .is_none_or(|p| p == pitch || off(p) && p.next_multiple_of(unit) == pitch),
                    "{pf:?}: {runtime:?}, not {pitch}"
                );
            }
            let desc = platform().texture_descriptor(&TextureDesc {
                pixel_format,
                width,
                height,
                usage: 0x1,
                resource_options: Some(PRIVATE),
                ..TextureDesc::default()
            });
            let buffer = d
                .new_buffer_of_length_padding_unobserved(
                    OnFault::Return,
                    pitch * height,
                    PRIVATE,
                    None,
                )
                .unwrap()
                .expect("a buffer");
            let before = counts();
            let texture = buffer.new_texture(OnFault::Return, &desc, 0, pitch);
            assert!(matches!(texture, Ok(Some(_))), "a texture");
            let after = counts();
            assert_eq!(
                (after.0 - before.0, after.1 - before.1),
                (u64::from(pitch > row), 0),
                "{pixel_format}: over the buffer's memory at {pitch}, its rows {row}"
            );
        }
    }

    /// A surface tile that renders and samples (`BGRA8Unorm`, `ShaderRead | RenderTarget`) over
    /// a Private buffer that declares the bytes between its rows unobserved, its rows at the
    /// device's pitch of them (`linear_rows_pitch`), is an image over the buffer's memory: at the
    /// driver's own pitch its linear image, at a pitch rounded up from one off Metal's 16 bytes
    /// (ANV's) an image at that pitch whose rows to their pitch fill whole lines of the device's
    /// memory, so share no line with bytes past them, where the device's check of renders over
    /// such padding passed. The sizes are those of textures over a macOS 14.8.9 guest's caches
    /// that were shadows on Iris Xe (275x54, 2618x1024, 73x59, 99x67, 87x55), whose rows of
    /// 1100, 10472, 292, 396 and 348 bytes ANV lays side by side (measured, the runtime's shadow
    /// log), and 70x59 (280 bytes): rounded up to 16 alone, 54 rows at 1104 bytes end 32 bytes
    /// into a line. A device that answers no pitch promises nothing, and one whose images lie at
    /// Metal's 16 bytes may make the tile a shadow at its own pitch
    #[test]
    fn a_surface_tile_at_the_device_s_pitch_renders_over_its_memory() {
        use crate::core::platform::Platform as _;
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::values::TextureDesc;
        use metal2vulkan::metal::PixelFormat as Pf;

        use crate::platform::tests::platform;

        const PRIVATE: u64 = 0x20;
        let d = dev();
        #[cfg(any(target_os = "linux", windows))]
        settle_device_checks(&d);
        let line = d.raw.in_place_check_line();
        let align = d.linear_row_alignment();
        let report = d.raw.in_place_check_report();
        let passed = |verdict: &str| {
            report
                .iter()
                .any(|l| l.strip_prefix(verdict) == Some("Passed"))
        };
        for (width, height) in [
            (275u64, 54u64),
            (2618, 1024),
            (70, 59),
            (73, 59),
            (99, 67),
            (87, 55),
        ] {
            let row = width * 4;
            let Some(pitch) = d.linear_rows_pitch(row, height) else {
                continue;
            };
            let mut descriptor = rt::TextureDescriptor::default();
            descriptor.pixel_format = Pf::BGRA8Unorm;
            descriptor.width = width;
            descriptor.height = height;
            descriptor.resource_options = rt::ResourceOptions::private();
            let own = d.raw.linear_texture_bytes_per_row(&descriptor).unwrap() == Some(pitch);
            if !own {
                assert!(
                    line.is_power_of_two(),
                    "a pitch off the driver's needs the device's line"
                );
                assert!(
                    (pitch * height).is_multiple_of(line) && pitch.is_multiple_of(16),
                    "{width}x{height} at {pitch}: rows to their pitch end inside a {line}-byte line"
                );
            }
            let desc = platform().texture_descriptor(&TextureDesc {
                pixel_format: 80,
                width,
                height,
                usage: 0x5,
                resource_options: Some(PRIVATE),
                ..TextureDesc::default()
            });
            let buffer = d
                .new_buffer_of_length_padding_unobserved(
                    OnFault::Return,
                    pitch * height,
                    PRIVATE,
                    None,
                )
                .unwrap()
                .expect("a buffer");
            let shadows = || {
                let c = d.raw.work_counts();
                c.shadowed_textures + c.shadows_left_unmade - c.shadows_made_at_use
            };
            let before = shadows();
            let texture = buffer.new_texture(OnFault::Return, &desc, 0, pitch);
            assert!(
                matches!(texture, Ok(Some(_))),
                "{width}x{height}: a texture"
            );
            let shadows = shadows() - before;
            let rows_on_lines = line > 0 && pitch.is_multiple_of(line) && row.is_multiple_of(line);
            let verdict = match rows_on_lines {
                true => "Pooled unobserved renders, rows on lines: ",
                false => "Pooled unobserved renders: ",
            };
            let in_place = match own {
                true => align > caps::METAL_LINEAR_ALIGNMENT,
                false => passed(verdict),
            };
            eprintln!(
                "{width}x{height}: rows {row} at {pitch} (own {own}, line {line}, {verdict}{}): \
                 {shadows} shadows",
                passed(verdict)
            );
            if in_place {
                assert_eq!(
                    shadows, 0,
                    "{width}x{height} at {pitch}: over the buffer's memory"
                );
            }
        }
    }

    /// A sparse heap takes its range where the model of Metal's space places it: unpinned or at a
    /// pin it honours it is made; at a pin it refuses (one off a 32-KiB granule) it dies naming
    /// the pin, as the runtime would make it and Metal's answer there is not measured
    #[test]
    fn sparse_heaps_die_at_pins_the_space_refuses() {
        use crate::core::platform::values::{HeapDesc, OptionSet};
        let d = dev();
        let sparse = |pin: Option<u64>| HeapDesc {
            size: 0x40000,
            heap_type: 2,
            options: vec![OptionSet::StorageMode(2)],
            pinned_gpu_address: pin,
        };
        let made = d.new_heap(&sparse(None)).unwrap();
        assert!(made.heap.is_some(), "unpinned");
        let far = metal_space::END / 2;
        let made = d.new_heap(&sparse(Some(far))).unwrap();
        assert!(
            made.heap.is_some(),
            "pinned far from every range, on a granule"
        );
        // A granule's second page, far from the first heap's range.
        let refused = far + 0x10_0000 + metal_space::PAGE;
        let site = catch_death(|| {
            let _ = d.new_heap(&sparse(Some(refused)));
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newHeapWithDescriptor:")
        );
        let line = crate::answer::LAST_LINE.lock().clone().unwrap();
        assert!(line.contains(&format!("{refused:#x}")), "{line}");
    }

    /// A no-copy buffer over pages the table shows mapped writable is imported with the runtime
    /// told so, which asks the kernel nothing of them (its count of the mappings it asked about
    /// stays); one over memory the table does not know (anonymous, with no file behind it) is
    /// not vouched for, and the runtime asks. The device's in-place check, which asks the kernel
    /// of its own pages, is settled first
    #[cfg(target_os = "linux")]
    #[test]
    fn a_buffer_is_vouched_for_while_its_pages_are_writable() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::HostVm as _;

        const PAGE: u64 = 0x1000;
        let d = dev();
        // The in-place check the device starts as it is adopted asks the kernel of its own
        // pages on another thread: it ends before the count is read, so only this test's
        // imports move it.
        settle_device_checks(&d);
        let vm = crate::remap::VulkanVm::new();
        let at = crate::os::allocate_aliasable(2 * PAGE).unwrap();
        let queried = || d.raw.work_counts().host_mappings_queried;
        let buffer = |at: HostPtr| {
            // SAFETY: the pages stay mapped until their unmapping below, after the buffer's drop.
            let bytes = unsafe { HostRange::new(at, 2 * PAGE) };
            d.new_buffer_over_host_memory(OnFault::Return, bytes, 0, None)
                .unwrap()
                .unwrap()
        };
        let before = queried();
        let vouched = buffer(at);
        assert_eq!(queried(), before, "vouched for: nothing asked");
        drop(vouched);
        // SAFETY: a fresh anonymous mapping, unmapped below.
        let anon = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                (2 * PAGE).to_usize(),
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(anon, libc::MAP_FAILED);
        let checked = buffer(HostPtr::from_ptr(anon.cast::<u8>()));
        assert!(queried() > before, "not vouched for: the runtime asked");
        drop(checked);
        assert!(
            vm.deallocate(HostPtr::from_ptr(anon.cast::<u8>()), 2 * PAGE)
                .is_success()
        );
        assert!(vm.deallocate(at, 2 * PAGE).is_success());
    }

    /// A no-copy buffer over a mapping of core's, released while a command buffer that reads it
    /// runs: its deallocator runs as the command buffer completes, Metal's moment, and core's
    /// unmap of the mapping inside it, through the platform's VM, answers success and leaves the
    /// pages mapped, as the device still imports them; the runtime frees the import after the
    /// deallocator and then drops the keep-alive, whose hold unmaps them. Nothing waits on another
    /// thread: the command buffer completes, the deallocator returns, and the unmap lands. The
    /// command buffer reads the buffer's bytes, and no deallocator is delayed. The release lands
    /// while the kernel runs, its end mark unset, or the test fails: a kernel waiting for a word
    /// the CPU writes after the release cannot hold the command buffer instead, as Metal's GPU saw
    /// no CPU write to a word it polled (`tests/new_images.rs`). The kernel runs about 60 ms on
    /// any device ([`crate::gpu::spin`]): 1,000 rounds, 60 ms on the M3 Max, are about 0.3 ms on
    /// an RTX 5080 Laptop GPU, where the release came 0.7 ms after the commit, once the thread
    /// waiting for the command buffer's scheduling woke (measured, 2026-10-26, beside the crate's
    /// whole run)
    #[test]
    fn an_unmap_in_a_deallocator_waits_for_the_import() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::mpsc;
        use std::time::Duration;

        use crate::core::platform::blit::{BlitCmd, BlitEncoder};
        use crate::core::platform::command::{CommandBuffer, CommandQueue};
        use crate::core::platform::resource::Buffer;
        use crate::core::platform::types::CommandBufferStatus;

        use crate::gpu::spin;

        /// How long the kernel that keeps the GPU busy runs, sized on the device
        /// ([`spin::rounds_to_hold`]): long enough to release the buffer while it runs, and short
        /// of the half second Metal fails under other GPU work (`tests/new_images.rs`'s kernel).
        /// Its fewest rounds, about as long on the M3 Max.
        const HOLD: Duration = Duration::from_millis(60);
        const FEWEST: u32 = 1_000;
        const LEN: u64 = 0x8000;
        const BYTE: u8 = 0xa5;
        let wait = Duration::from_secs(30);
        // Whether `done` came true within `wait`, polled each millisecond.
        let within = |done: &dyn Fn() -> bool| {
            (0..wait.as_millis()).any(|_| {
                done() || {
                    std::thread::sleep(Duration::from_millis(1));
                    false
                }
            }) || done()
        };
        let d = dev();
        let vm = crate::remap::VulkanVm::new();
        let va = vm.allocate_anywhere(LEN).unwrap();
        let len = usize::try_from(LEN).unwrap();
        // SAFETY: pages just mapped, readable and writable.
        unsafe { std::ptr::write_bytes(va.as_ptr::<u8>(), BYTE, len) };
        // What the deallocator saw after core's unmap: its answer, whether the first page was
        // still mapped, and whether the unmap waited.
        let (ran, seen) = mpsc::channel();
        let deallocator: NoCopyDeallocator = Box::new(move || {
            // Core's mapping's release (`Teardown::Vm`).
            let freed = crate::remap::VulkanVm::new().deallocate(va, LEN);
            let waits = crate::remap::unmap_waits(va, LEN);
            let _ = ran.send((freed.is_success(), crate::os::mapped(va), waits));
        });
        // SAFETY: the pages stay mapped until the deallocator runs, which unmaps them.
        let bytes = unsafe { HostRange::new(va, LEN) };
        let buffer = d
            .new_buffer_over_host_memory(OnFault::Die, bytes, 0, Some(deallocator))
            .unwrap()
            .unwrap();
        let state = spin::state(&d);
        let queue = d.new_command_queue().unwrap();
        let words = d
            .new_buffer_of_length(OnFault::Die, 16, 0)
            .unwrap()
            .unwrap();
        let word = |i: usize| {
            // SAFETY: one of the buffer's four words, which the kernel reads and writes.
            unsafe { AtomicU32::from_ptr(words.contents().as_ptr::<u32>().add(i)) }
        };
        word(0).store(
            spin::rounds_to_hold(&d, &queue, &state, HOLD, FEWEST),
            Ordering::SeqCst,
        );
        let out = d
            .new_buffer_of_length(OnFault::Die, LEN, 0)
            .unwrap()
            .unwrap();
        let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
        spin::encode(&cb, &state, &words);
        let b = cb.blit_encoder(OnFault::Die).unwrap().unwrap();
        for cmd in [
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&buffer),
                source_offset: 0,
                destination_buffer: Some(&out),
                destination_offset: 0,
                size: LEN,
            },
            BlitCmd::EndEncoding,
        ] {
            b.encode(OnFault::Die, cmd).unwrap();
        }
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_scheduled().unwrap();
        let running = word(1).load(Ordering::SeqCst) == 0;
        drop(buffer);
        assert!(running, "released while the kernel ran");
        // The command buffer holds the buffer: no deallocator while the kernel runs.
        let early = seen.try_recv().ok();
        if early.is_some() {
            assert_ne!(
                word(1).load(Ordering::SeqCst),
                0,
                "ran before the kernel ended"
            );
        }
        let ended = [CommandBufferStatus::COMPLETED, CommandBufferStatus::ERROR];
        assert!(
            within(&|| ended.contains(&cb.status())),
            "the command buffer did not complete"
        );
        assert_eq!(
            cb.status(),
            CommandBufferStatus::COMPLETED,
            "{:?}",
            cb.error()
        );
        let (freed, mapped, waits) = early
            .or_else(|| seen.recv_timeout(wait).ok())
            .expect("the deallocator ran");
        assert!(freed, "the unmap answered success");
        assert!(
            mapped && waits,
            "the device still imported the pages: the unmap waited"
        );
        assert!(
            within(&|| !crate::remap::unmap_waits(va, LEN)),
            "the keep-alive was not dropped"
        );
        // SAFETY: a read of the first page where it is mapped; another test's thread may have
        // mapped it again since, without these bytes.
        let gone = !crate::os::mapped(va) || unsafe { va.as_ptr::<u8>().read_volatile() } != BYTE;
        assert!(gone, "unmapped once the import was freed");
        // SAFETY: a Shared buffer the GPU's writes completed.
        let read = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), len) };
        assert!(
            read.iter().all(|&b| b == BYTE),
            "the command buffer read the bytes"
        );
        assert_eq!(d.raw.work_counts().host_deallocators_delayed, 0);
    }

    /// A no-copy buffer over a mapping of core's, released while a narrower one made over its
    /// first page lives, which the device binds inside the wider one's import (the runtime's "One
    /// page, one import") and which holds that import past the release: the runtime delays a
    /// deallocator given no keep-alive until the narrower one went, announcing
    /// `HostDeallocatorDelayed`, where Metal's runs at the release (the runtime's probe
    /// `dealloc-probe4`, the M3 Max under macOS 27.0.1). Every no-copy buffer and heap this
    /// platform makes gives one ([`no_copy_release`]), so the deallocator runs at the release, the
    /// keep-alive keeping the pages mapped for the narrower one, and nothing is announced, which
    /// in strict mode, this test's, would die there. The narrower one's runs at its own release.
    /// The narrower one is made under the default: a device that did not bind it inside would make
    /// it over a copy of its page (`copied-host-pages`), which the test then fails, as it would
    /// reach nothing
    #[test]
    fn a_release_whose_import_another_buffer_holds_runs_its_deallocator_at_once() {
        use std::sync::mpsc;
        use std::time::Duration;
        let d = dev();
        let vm = crate::remap::VulkanVm::new();
        let page = crate::os::PAGE;
        let va = vm.allocate_anywhere(2 * page).unwrap();
        let made = |len: u64, ran: mpsc::Sender<()>| {
            let deallocator: NoCopyDeallocator = Box::new(move || {
                let _ = ran.send(());
            });
            // SAFETY: the pages stay mapped until both buffers went and their imports were freed.
            let bytes = unsafe { HostRange::new(va, len) };
            crate::answer::relaxed(|| {
                d.new_buffer_over_host_memory(OnFault::Die, bytes, 0, Some(deallocator))
            })
            .unwrap()
            .expect("a no-copy buffer")
        };
        let base = d.raw.work_counts();
        let (wide_ran, wide_seen) = mpsc::channel();
        let (narrow_ran, narrow_seen) = mpsc::channel();
        let wide = made(2 * page, wide_ran);
        let narrow = made(page, narrow_ran);
        let bound = |c: &rt::WorkCounts| c.object_churn.buffers_bound_in_import;
        let inside = bound(&d.raw.work_counts()) > bound(&base);
        assert!(
            inside,
            "the narrower one lies inside the wider one's import"
        );
        drop(wide);
        assert!(
            wide_seen.try_recv().is_ok(),
            "the wider one's deallocator ran at its release"
        );
        let counts = d.raw.work_counts();
        assert_eq!(
            counts.host_deallocators_delayed,
            base.host_deallocators_delayed
        );
        assert_eq!(
            counts.host_releases_held_inside - base.host_releases_held_inside,
            1,
            "the narrower one held the wider one's import past its release"
        );
        drop(narrow);
        assert!(
            narrow_seen.recv_timeout(Duration::from_secs(30)).is_ok(),
            "the narrower one's deallocator ran"
        );
        assert!(vm.deallocate(va, 2 * page).is_success());
    }

    #[test]
    fn a_device_is_one_object() {
        let d = dev();
        let again = adopt(d.raw.clone());
        assert_eq!(again.id(), d.id());
        assert!(!d.raw.name().is_empty());
    }

    /// A device made in the process is found by the registry ID the platform answers for it, as
    /// the one object every handle to it is, which an open answers with no enumeration; an ID no
    /// device made answers none
    #[test]
    fn a_made_device_is_kept_by_its_registry_id() {
        use crate::core::platform::Platform as _;
        let d = dev();
        assert_eq!(kept(d.registry_id).map(|k| k.id()), Some(d.id()));
        assert!(kept(d.registry_id ^ 0xdead_beef).is_none());
        let opened = crate::platform::tests::platform().open_device(d.registry_id);
        assert_eq!(opened.map(|o| o.id()), Some(d.id()));
    }

    /// A compute pipeline the runtime refuses for a gap of its own or of the device's is, by
    /// default, nil and the runtime's refusal as the `NSError`, relayed; in strict mode it dies
    /// at its selector as before. The creators' decision, without a device
    #[test]
    fn a_compute_pipeline_gap_is_its_nserror_by_default() {
        use metal2vulkan::error::{Feature, Unsupported};
        let sel = "newComputePipelineStateWithFunction:error:";
        let e = rt::Error::Unsupported(Unsupported::not_implemented(Feature::SubgroupOperations));
        let before = crate::answer::inexact_counts();
        let error = crate::answer::relaxed(|| compile_error(sel, e.clone(), None));
        assert_eq!(error, error_text(&e));
        assert_eq!(error.domain, crate::answer::RUNTIME_DOMAIN);
        let counted = |counts: Vec<(crate::answer::Inexact, u64)>| {
            (counts.into_iter())
                .find(|&(k, _)| k == crate::answer::Inexact::RefusedCreate)
                .map_or(0, |(_, n)| n)
        };
        assert_eq!(
            counted(crate::answer::inexact_counts()),
            counted(before) + 1
        );
        let site = catch_death(|| compile_error(sel, e.clone(), None)).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
    }

    /// A multisample texture of the padded 10-bit formats (576 to 579), which the runtime stops
    /// as Metal's validation does (`Invalid`, `TextureProblem::TypeForFormat`), dies at its
    /// creator, where Metal aborts in the descriptor's validation: measured on this host (macOS
    /// 26.7), plain and under `MTL_DEBUG_LAYER`, 16 by 16 at 4 samples, `Private`, render target
    /// usage, a 2D multisample texture and an array of 2 ("-[MTLTextureDescriptorInternal
    /// validateWithDevice:]:1416: failed assertion `Texture Descriptor Validation
    /// MTLTextureDescriptor pixelFormat (MTLPixelFormatR10Unorm_X6) does not support multisample
    /// textures.'"). The creator's decision of the runtime's stop, without a device
    #[test]
    fn a_stopped_multisample_padded_texture_dies_at_its_creator() {
        let sel = "newTextureWithDescriptor:";
        for raw in 576..=579 {
            let stop = rt::Error::Invalid(rt::Invalid::new(
                "new_texture",
                rt::Problem::Texture(rt::TextureProblem::TypeForFormat),
            ));
            *crate::answer::LAST_LINE.lock() = None;
            let site = catch_death(|| {
                create(sel, Policy::Dies, Nil::Memory, || {
                    Err::<(), _>(format_refused(sel, "a texture of", raw, stop))
                })
            })
            .unwrap_err();
            let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{raw}");
            assert!(line.contains("TypeForFormat"), "{raw}: {line}");
        }
    }

    /// The same at the device: a 2D multisample texture of 576 to 579, and an array of 2, at 4
    /// samples, die at `newTextureWithDescriptor:`, named by the runtime's stop, where Metal
    /// aborts ([`a_stopped_multisample_padded_texture_dies_at_its_creator`])
    #[test]
    fn multisample_padded_textures_die_where_metal_aborts() {
        use crate::core::platform::Platform as _;
        use crate::core::platform::values::TextureDesc;
        let p = crate::platform::tests::platform();
        let sel = "newTextureWithDescriptor:";
        for raw in 576..=579 {
            for (texture_type, array_length) in [(4, 1), (8, 2)] {
                let desc = p.texture_descriptor(&TextureDesc {
                    texture_type,
                    pixel_format: raw,
                    width: 16,
                    height: 16,
                    array_length,
                    sample_count: 4,
                    usage: 4,
                    resource_options: Some(2 << 4),
                    ..TextureDesc::default()
                });
                *crate::answer::LAST_LINE.lock() = None;
                let site =
                    catch_death(|| drop(dev().new_texture(OnFault::Return, &desc))).unwrap_err();
                let line = crate::answer::LAST_LINE.lock().take().unwrap_or_default();
                let case = format!("{raw} type {texture_type}");
                assert_eq!((site.kind, site.file), (DeathKind::Abort, sel), "{case}");
                assert!(line.contains("TypeForFormat"), "{case}: {line}");
            }
        }
    }

    /// The platform's own texture takes no resource ID and none of Metal's space: it answers 0,
    /// and a guest texture and buffer made while it lives take the ID and the address they take
    /// without it, where a guest texture held in its place moves them
    #[test]
    fn an_own_texture_takes_no_resource_id_or_metal_space() {
        use crate::core::platform::Platform as _;
        use crate::core::platform::resource::{Buffer as _, Texture as _};
        use crate::core::platform::values::TextureDesc;
        let p = crate::platform::tests::platform();
        let d = dev();
        let desc = p.texture_descriptor(&TextureDesc {
            width: 32,
            height: 32,
            ..TextureDesc::default()
        });
        let guest = || {
            let texture = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
            let buffer = d
                .new_buffer_of_length(OnFault::Return, 64, 0)
                .unwrap()
                .unwrap();
            (texture.unique_identifier(), buffer.gpu_address())
        };
        let without = guest();
        let own = d.new_own_texture(&desc).unwrap();
        assert_eq!(own.unique_identifier(), 0);
        assert_eq!(guest(), without);
        let held = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let (id, address) = guest();
        assert_ne!(id, without.0);
        assert_ne!(address, without.1);
        drop((own, held));
    }

    /// A Core Image kernel library (`ci_kernel.metallib`, its header's file type 1) is made, as
    /// Metal makes it, and the device goes on to make others
    #[test]
    fn a_core_image_kernel_library_is_made() {
        const CI_KERNEL: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ci_kernel.metallib"));
        assert_eq!(CI_KERNEL[10], 1, "the header's file type");
        assert!(dev().new_library(CI_KERNEL).unwrap().unwrap().is_some());
        assert!(
            dev()
                .new_library(include_bytes!("../../../../tests/shaders/kernel.metallib"))
                .unwrap()
                .unwrap()
                .is_some()
        );
    }

    /// Every compute pipeline creation takes a new nonzero identifier, even of one function twice,
    /// and every reference to a state answers its own
    #[test]
    fn each_compute_pipeline_creation_takes_a_new_identifier() {
        use crate::core::platform::resource::ComputePipeline as _;
        use crate::core::platform::resource::Library as _;
        let d = dev();
        let library = d
            .new_library(include_bytes!("../../../../tests/shaders/kernel.metallib"))
            .unwrap()
            .unwrap()
            .unwrap();
        let name = &library.function_names()[0];
        let f = library
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap();
        let a = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let b = d
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        assert_ne!(a.unique_identifier(), 0);
        assert_ne!(a.unique_identifier(), b.unique_identifier());
        assert_eq!(a.clone_ref().unique_identifier(), a.unique_identifier());
    }

    /// The creators of queues, events and fences each make a new object of the device
    #[test]
    fn creators_make_objects_of_the_device() {
        let d = dev();
        d.allow_foreign_libraries();
        let (a, b) = (
            d.new_command_queue().unwrap(),
            d.new_command_queue().unwrap(),
        );
        assert_ne!(a.id(), b.id());
        assert_eq!(a.device.id(), d.id());
        let e = d.new_event(OnFault::Return).unwrap().unwrap();
        assert_ne!(e.id(), d.new_event(OnFault::Die).unwrap().unwrap().id());
        assert!(d.new_fence().unwrap().is_some());
    }

    /// An unshared queue runs a command buffer as `newCommandQueue`'s does, and its maker passes
    /// the runtime's refusals on as they are: a count of 0 is the runtime's `Invalid`
    /// (`MaxCommandBufferCount`), where the platform asks for Metal's 64.
    #[test]
    fn an_unshared_queue_runs_and_its_refusals_are_the_runtimes() {
        use crate::core::platform::command::{CommandBuffer, CommandQueue};
        let d = dev();
        match unshared_queue(&d.raw, 0) {
            Err(rt::Error::Invalid(i)) => assert_eq!(i.problem, rt::Problem::MaxCommandBufferCount),
            Err(e) => panic!("refused otherwise: {e}"),
            Ok(_) => panic!("a count of 0 made a queue"),
        }
        let queue = d.new_command_queue_unshared().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!(cb.status().0, 4, "completed");
    }

    /// A buffer made as a dma-buf device cache (`new_dma_buf_buffer_of_length_padding_unobserved`)
    /// lies, on Linux, in memory the runtime exports as a dma-buf: its aligned byte on 256 bytes of
    /// the allocation, its bytes within it, and the acquire of another API's writes taken. A
    /// buffer made as any other exports no dma-buf. On macOS, where MoltenVK exports no dma-buf of
    /// a buffer's memory, none is made, and the caller makes the cache as any other.
    #[test]
    #[cfg(unix)]
    fn a_dma_buf_cache_exports_its_memory_with_its_aligned_byte_on_256_bytes() {
        use crate::core::platform::resource::Buffer as _;
        let d = dev();
        let (length, aligned) = (0x10_0000, 0x20);
        let private = 2 << 4;
        let made = d.new_dma_buf_buffer_of_length_padding_unobserved(
            OnFault::Return,
            length,
            private,
            Some(aligned),
        );
        let made = made.expect("no raise");
        if cfg!(target_os = "macos") {
            assert!(made.is_none(), "MoltenVK made a dma-buf buffer");
            return;
        }
        let buffer = made.expect("a dma-buf buffer");
        let exported = buffer.export_dma_buf().expect("its dma-buf");
        assert_eq!((exported.offset + aligned) % 256, 0, "{exported:?}");
        assert!(
            exported.offset + length <= exported.allocation_size,
            "{exported:?}"
        );
        buffer.external_write_done().expect("the acquire");
        let plain = (d.new_buffer_of_length_padding_unobserved(
            OnFault::Return,
            length,
            private,
            Some(aligned),
        ))
        .expect("no raise")
        .expect("a buffer");
        assert!(plain.export_dma_buf().is_err());
    }

    #[cfg(target_os = "macos")]
    pub(crate) mod metal_tests;
}
