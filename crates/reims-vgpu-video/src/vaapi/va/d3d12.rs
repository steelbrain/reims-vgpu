//! Direct3D 12, on Windows, as much of it as the module needs: a device on the display's DXGI
//! adapter, the NV12 and P010 textures that the display's pools are made of, and copies between
//! them.
//!
//! A pool's surfaces on Windows are textures of ours, made here on the adapter's device, each
//! handed to Mesa's VA driver (`vaon12`) as its resource
//! (`VA_SURFACE_ATTRIB_MEM_TYPE_D3D12_RESOURCE`), so that what decodes into them can be read by
//! the plane draws (`super::d3d11`), which wrap the same resource. Direct3D 12 makes one device a
//! process and adapter (`D3D12CreateDevice` answers the one that lives), so the driver's, the
//! draws' and the pools' devices on the decode's adapter are one, and each takes the resource as
//! its own. vaon12's own surfaces cannot be read so: their export fails (measured on the UHD 630,
//! Mesa 26.2.7, 2026-11-16). A copy of one texture into another (a stand-in for a lost frame,
//! `super::surface`) runs on a copy queue of this device's, which the module waits for on its
//! fence.
//!
//! No pool texture is shared by an NT handle. An open of a texture's NT handle on the device that
//! made it (`ID3D12Device::OpenSharedHandle`) adds the texture's size to the process's Shared
//! Usage in the "GPU Process Memory" counters (what Task Manager shows): a charge in those
//! counters that never comes down, even past the device's last release; available physical
//! memory, the commit charge and DXGI's usage (`QueryVideoMemoryInfo`) did not move with it
//! (over 1440 MiB of charge, measured on the lab's UHD 630 and on the software adapter, WARP,
//! potluck job m3-max-75692). It stays past the open's release, the texture's and its handle's,
//! and the display's `vaTerminate`. vaon12's import of an NT handle
//! (`VA_SURFACE_ATTRIB_MEM_TYPE_NTHANDLE`) is such an open, charged again at each import of the
//! same texture. A pool of six 3840x2160 NV12 textures was charged 72 MiB at each import and at
//! each of the draws' opens, with no VA call for the opens alone; handed over and wrapped as
//! resources, none (measured on an Intel UHD Graphics 630, driver 31.0.101.2145, Mesa 26.2.7,
//! 2026-11-26, with programs of our own on D3D12 and libva alone).
//!
//! The binding is written by hand: the functions are loaded from `d3d12.dll` and `dxgi.dll` with
//! `LoadLibraryW` and `GetProcAddress`, as libva is (`super::libva`), and each COM method is
//! called through its object's vtable at the index the Windows SDK's headers give it (MSYS2
//! CLANG64's `d3d12.h` and `dxgi1_4.h`, read 2026-11-16). Only methods that return nothing, a
//! scalar or an `HRESULT` are called: one that returns a structure has another calling
//! convention. `super::d3d11`'s draws of a picture's planes call through the same binding.

use std::ffi::{CStr, c_void};
use std::ptr::{NonNull, null, null_mut};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use crate::core::sync::Mutex;

use super::surface::SurfaceFormat;

pub(super) type Hresult = i32;

#[repr(C)]
pub(super) struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

pub(super) const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
    Guid {
        data1,
        data2,
        data3,
        data4,
    }
}

const IID_IDXGI_FACTORY4: Guid = guid(
    0x1bc6_ea02,
    0xef36,
    0x464f,
    [0xbf, 0x0c, 0x21, 0xca, 0x39, 0xe5, 0x16, 0x8a],
);
const IID_IDXGI_ADAPTER: Guid = guid(
    0x2411_e7e1,
    0x12ac,
    0x4ccf,
    [0xbd, 0x14, 0x97, 0x98, 0xe8, 0x53, 0x4d, 0xc0],
);
pub(super) const IID_ID3D12_DEVICE: Guid = guid(
    0x1898_19f1,
    0x1db6,
    0x4b57,
    [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
);
pub(super) const IID_ID3D12_RESOURCE: Guid = guid(
    0x6964_42be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);
pub(super) const IID_ID3D12_COMMAND_QUEUE: Guid = guid(
    0x0ec8_70a6,
    0x5d7e,
    0x4c22,
    [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
);
const IID_ID3D12_COMMAND_ALLOCATOR: Guid = guid(
    0x6102_dee4,
    0xaf59,
    0x4b09,
    [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
);
const IID_ID3D12_GRAPHICS_COMMAND_LIST: Guid = guid(
    0x5b16_0d0f,
    0xac1b,
    0x4185,
    [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
);
pub(super) const IID_ID3D12_FENCE: Guid = guid(
    0x0a75_3dcf,
    0xc4d8,
    0x4b91,
    [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
);

// The vtable indices of the methods called (the SDK's `<interface>Vtbl` structures).
const QUERY_INTERFACE: usize = 0;
const ADD_REF: usize = 1;
const RELEASE: usize = 2;
const FACTORY4_ENUM_ADAPTER_BY_LUID: usize = 26;
pub(super) const DEVICE_CREATE_COMMAND_QUEUE: usize = 8;
const DEVICE_CREATE_COMMAND_ALLOCATOR: usize = 9;
const DEVICE_CREATE_COMMAND_LIST: usize = 12;
const DEVICE_CREATE_COMMITTED_RESOURCE: usize = 27;
pub(super) const DEVICE_CREATE_SHARED_HANDLE: usize = 31;
pub(super) const DEVICE_CREATE_FENCE: usize = 36;
const QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;
pub(super) const QUEUE_SIGNAL: usize = 14;
const ALLOCATOR_RESET: usize = 8;
const LIST_CLOSE: usize = 9;
const LIST_RESET: usize = 10;
const LIST_COPY_RESOURCE: usize = 17;
pub(super) const FENCE_GET_COMPLETED_VALUE: usize = 8;
pub(super) const FENCE_SET_EVENT_ON_COMPLETION: usize = 9;

const DXGI_FORMAT_NV12: u32 = 103;
const DXGI_FORMAT_P010: u32 = 104;
const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
const D3D12_HEAP_TYPE_DEFAULT: u32 = 1;
const D3D12_HEAP_FLAG_NONE: u32 = 0;
const D3D12_RESOURCE_DIMENSION_TEXTURE2D: u32 = 3;
const D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET: u32 = 0x1;
const D3D12_RESOURCE_STATE_COMMON: u32 = 0;
const D3D12_COMMAND_LIST_TYPE_COPY: u32 = 3;
pub(super) const WAIT_OBJECT_0: u32 = 0;

/// `D3D12_HEAP_PROPERTIES`.
#[repr(C)]
struct HeapProperties {
    type_: u32,
    cpu_page_property: u32,
    memory_pool_preference: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
}

/// `D3D12_RESOURCE_DESC`: 56 bytes, `Width` at 16 and `Flags` at 48, as the SDK lays it out.
#[repr(C)]
struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

/// `D3D12_COMMAND_QUEUE_DESC`.
#[repr(C)]
pub(super) struct CommandQueueDesc {
    pub(super) type_: u32,
    pub(super) priority: i32,
    pub(super) flags: u32,
    pub(super) node_mask: u32,
}

/// `LUID`, passed by value.
#[repr(C)]
#[derive(Clone, Copy)]
struct Luid {
    low: u32,
    high: i32,
}

#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const std::ffi::c_char) -> *mut c_void;
    fn CloseHandle(handle: *mut c_void) -> i32;
    pub(super) fn CreateEventW(
        attributes: *const c_void,
        manual_reset: i32,
        initial_state: i32,
        name: *const u16,
    ) -> *mut c_void;
    pub(super) fn WaitForSingleObject(handle: *mut c_void, milliseconds: u32) -> u32;
}

type CreateDxgiFactory1 = unsafe extern "system" fn(*const Guid, *mut *mut c_void) -> Hresult;
type D3d12CreateDevice =
    unsafe extern "system" fn(*mut c_void, u32, *const Guid, *mut *mut c_void) -> Hresult;

/// The address of `symbol` in the system library `library`, loaded once and never unloaded.
pub(super) fn system_function(library: &str, symbol: &CStr) -> Result<NonNull<c_void>, String> {
    let wide: Vec<u16> = library.encode_utf16().chain([0]).collect();
    // SAFETY: `wide` is NUL-terminated UTF-16 naming a system library.
    let module = unsafe { LoadLibraryW(wide.as_ptr()) };
    if module.is_null() {
        return Err(format!(
            "cannot load {library}: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: `module` is a loaded module, never freed, and `symbol` is NUL-terminated.
    NonNull::new(unsafe { GetProcAddress(module, symbol.as_ptr()) }).ok_or_else(|| {
        format!(
            "{library} has no {}: {}",
            symbol.to_string_lossy(),
            std::io::Error::last_os_error()
        )
    })
}

/// `call`'s `HRESULT` as a result: `Err` naming `what` where it failed.
pub(super) fn check(what: &str, result: Hresult) -> Result<(), String> {
    if result < 0 {
        // The HRESULT's bits, as Windows writes them.
        Err(format!("{what} failed: {:#010x}", result.cast_unsigned()))
    } else {
        Ok(())
    }
}

/// A COM object we hold one reference of, released when dropped.
#[derive(Debug)]
pub(super) struct Com(NonNull<c_void>);

// SAFETY: the objects held are Direct3D 12's, Direct3D 11's and DXGI's. Direct3D 12's and DXGI's,
// and Direct3D 11's devices, are free-threaded: their methods may be called from any thread. A
// Direct3D 11 device context is not, and `super::d3d11` calls one only under the lock of its
// state, from one thread at a time.
unsafe impl Send for Com {}
// SAFETY: as above.
unsafe impl Sync for Com {}

impl Com {
    /// The object `out` names, made by a call that returned `result`.
    pub(super) fn made(what: &str, result: Hresult, out: *mut c_void) -> Result<Com, String> {
        check(what, result)?;
        NonNull::new(out)
            .map(Com)
            .ok_or_else(|| format!("{what} made nothing"))
    }

    pub(super) fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    /// Another reference of the object (`AddRef`, every COM object's method 1), released when
    /// dropped.
    pub(super) fn another(&self) -> Com {
        // SAFETY: `AddRef` is every COM object's method 1; the reference it adds is the new
        // `Com`'s, which releases it once.
        unsafe {
            let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = self.method(ADD_REF);
            add_ref(self.raw());
        }
        Com(self.0)
    }

    /// The method at `index` of the object's vtable, as `F`.
    ///
    /// # Safety
    ///
    /// `F` must be the method's signature, with the object as its first argument.
    pub(super) unsafe fn method<F: Copy>(&self, index: usize) -> F {
        // SAFETY: a COM object's first field is its vtable, an array of function pointers, of
        // which the caller names one of the object's interface by its index and signature.
        unsafe {
            let vtable = *self.0.as_ptr().cast::<*const *const c_void>();
            std::mem::transmute_copy::<*const c_void, F>(&*vtable.add(index))
        }
    }

    /// The object's interface of `iid` (`QueryInterface`, every COM object's method 0), named
    /// `what` where it has none.
    pub(super) fn query(&self, iid: &Guid, what: &str) -> Result<Com, String> {
        let mut out = null_mut();
        // SAFETY: `QueryInterface` is every COM object's method 0, given an IID and where to
        // write the interface.
        let result = unsafe {
            let query: unsafe extern "system" fn(
                *mut c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.method(QUERY_INTERFACE);
            query(self.raw(), iid, &mut out)
        };
        Com::made(what, result, out)
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        // SAFETY: `Release` is every COM object's method 2, and this reference is ours.
        unsafe {
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = self.method(RELEASE);
            release(self.raw());
        }
    }
}

/// A Direct3D 12 device on the display's adapter, and its copy queue. While it lives, the device
/// makes one video decode queue, which every ask for one gets (`super::decode_queue`).
pub(crate) struct Device {
    pub(super) device: Com,
    copies: Mutex<Option<Copies>>,
    _decode_queue: super::decode_queue::Hold,
}

/// The objects of the device's copies, made at its first copy.
struct Copies {
    queue: Com,
    allocator: Com,
    list: Com,
    fence: Com,
    /// The fence's last value signalled.
    value: u64,
    /// The event the fence's completion sets.
    event: Handle,
}

impl Copies {
    /// Whether the fence has reached the last value signalled: `Removed` where it answers
    /// `UINT64_MAX`, as a removed device's fence does.
    fn completed(&self) -> Completed {
        // SAFETY: the method is ID3D12Fence's `GetCompletedValue`.
        let value = unsafe {
            let completed: unsafe extern "system" fn(*mut c_void) -> u64 =
                self.fence.method(FENCE_GET_COMPLETED_VALUE);
            completed(self.fence.raw())
        };
        match value {
            u64::MAX => Completed::Removed,
            value if value >= self.value => Completed::Done,
            _ => Completed::Not,
        }
    }
}

/// What a fence answers of the last copy's ([`Copies::completed`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Completed {
    Done,
    Not,
    Removed,
}

/// A copy sent and not seen done ([`Device::copy`]): its queue's objects and the textures it
/// reads and writes, to be kept until its fence answers it done
/// ([`Unfinished`](crate::core::device_wait::Unfinished)). Dropped in field order: the copy's
/// objects, then the textures.
pub(crate) struct InFlight {
    copies: Copies,
    _textures: [Arc<SharedTexture>; 2],
}

impl crate::core::device_wait::Unfinished for InFlight {
    fn finished(&mut self) -> bool {
        // A removed device's answer says nothing of its writes.
        self.copies.completed() == Completed::Done
    }
}

/// Why [`Device::copy`] did not copy.
pub(crate) enum CopyError {
    /// A call failed, with why; and the copy where one was sent.
    Failed(String, Option<InFlight>),
    /// The copy was not done by its wait.
    Late(InFlight),
}

/// An NT handle, closed when dropped.
#[derive(Debug)]
pub(super) struct Handle(pub(super) NonNull<c_void>);

// SAFETY: a handle is the process's, usable from any thread.
unsafe impl Send for Handle {}
// SAFETY: as above.
unsafe impl Sync for Handle {}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: the handle is ours, and closed once.
        unsafe { CloseHandle(self.0.as_ptr()) };
    }
}

/// A texture of a pool's, shared by the driver's decodes and the draws (module doc), which each
/// take its resource ([`SharedTexture::raw_resource`], [`SharedTexture::resource`]).
///
/// An import of it is let go once [`SharedTexture::alive`] no longer upgrades, and a device that
/// keeps one (the draws' wrap of its resource) asks to be told as the texture goes
/// ([`SharedTexture::on_let_go`]), since it keeps the texture's memory while it lives.
#[derive(Debug)]
pub struct SharedTexture {
    resource: Com,
    /// The device that made it, which the draws' device is to be (module doc): its pointer, as
    /// the device's identity, never dereferenced.
    device: usize,
    format: SurfaceFormat,
    size: (u32, u32),
    /// Its life, which another device's imports of it are let go by.
    life: Arc<()>,
    /// What the devices that keep an open of it do as it goes.
    let_go: LetGoes,
}

/// What a device that keeps an open of a [`SharedTexture`] does as the texture goes, told the
/// texture's life ([`SharedTexture::alive`]) its open was made by: let go of the open.
pub(crate) type LetGo = Box<dyn FnOnce(&Weak<()>) + Send>;

/// A [`SharedTexture`]'s [`LetGo`]s, run once, at its drop.
#[derive(Default)]
struct LetGoes(Mutex<Vec<LetGo>>);

impl std::fmt::Debug for LetGoes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} let go", self.0.lock().len())
    }
}

impl Drop for SharedTexture {
    /// Tells each device that keeps an open of the texture that it goes, on this thread, before
    /// its resource and handle are let go: a device then lets go of its open, so that the
    /// texture's memory goes with it, not at the device's next use of another texture.
    fn drop(&mut self) {
        let life = Arc::downgrade(&self.life);
        for let_go in std::mem::take(self.let_go.0.get_mut()) {
            let_go(&life);
        }
    }
}

impl SharedTexture {
    /// The texture's resource (an `ID3D12Resource`), for the driver's import, which takes a
    /// reference of its own.
    pub(crate) fn raw_resource(&self) -> *mut c_void {
        self.resource.raw()
    }

    /// Another reference of the texture's resource, for a device of `device`'s to wrap: `Err`
    /// where `device` did not make it, as only the device that made a resource uses it.
    pub(super) fn resource(&self, device: &Com) -> Result<Com, String> {
        if device.raw().addr() != self.device {
            return Err("a texture of another Direct3D 12 device".to_owned());
        }
        Ok(self.resource.another())
    }

    /// Whether its samples are 10-bit in 16 (P010), else 8-bit (NV12).
    pub fn is_10_bit(&self) -> bool {
        self.format == SurfaceFormat::P010
    }

    /// Its width and height in luma samples: the pool's, the decoded (coded) picture's.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// A reference that lives as long as the texture: an import of it is let go once it no
    /// longer upgrades.
    pub fn alive(&self) -> Weak<()> {
        Arc::downgrade(&self.life)
    }

    /// Asks that `let_go` run as the texture goes, on the thread that lets go of it last, told
    /// its life: a device that keeps an open of it lets go of the open there. `let_go` takes no
    /// lock that a holder of the texture may hold as it lets go of it.
    pub(crate) fn on_let_go(&self, let_go: LetGo) {
        self.let_go.0.lock().push(let_go);
    }
}

impl Device {
    /// The device on the DXGI adapter of `luid`, or on the default adapter for `None` (the one a
    /// display opened with no LUID decodes on: the first DXGI lists).
    pub(crate) fn new(luid: Option<[u8; 8]>) -> Result<Device, String> {
        let create_device = system_function("d3d12.dll", c"D3D12CreateDevice")?;
        // SAFETY: the function is `D3D12CreateDevice` (d3d12.h).
        let create_device =
            unsafe { std::mem::transmute::<NonNull<c_void>, D3d12CreateDevice>(create_device) };
        let adapter = match luid {
            Some(luid) => Some(adapter_of(luid)?),
            None => None,
        };
        let mut device = null_mut();
        // SAFETY: the adapter is null or one DXGI made; the IID is ID3D12Device's.
        let result = unsafe {
            create_device(
                adapter.as_ref().map_or(null_mut(), Com::raw),
                D3D_FEATURE_LEVEL_11_0,
                &IID_ID3D12_DEVICE,
                &mut device,
            )
        };
        let device = Com::made("D3D12CreateDevice", result, device)?;
        Ok(Device {
            _decode_queue: super::decode_queue::Hold::new(&device)?,
            device,
            copies: Mutex::new(None),
        })
    }

    /// A texture of `format`, `width` by `height`, for a pool (module doc).
    pub(crate) fn shared_texture(
        &self,
        format: SurfaceFormat,
        width: u32,
        height: u32,
    ) -> Result<SharedTexture, String> {
        let dxgi_format = match format {
            SurfaceFormat::Nv12 => DXGI_FORMAT_NV12,
            SurfaceFormat::P010 => DXGI_FORMAT_P010,
            SurfaceFormat::Xyuv => return Err("no shared texture of XYUV".to_owned()),
        };
        let heap = HeapProperties {
            type_: D3D12_HEAP_TYPE_DEFAULT,
            cpu_page_property: 0,
            memory_pool_preference: 0,
            creation_node_mask: 0,
            visible_node_mask: 0,
        };
        let desc = ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            alignment: 0,
            width: u64::from(width),
            height,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: dxgi_format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        };
        let mut resource = null_mut();
        // SAFETY: the method is `CreateCommittedResource`, with valid descriptions, no clear
        // value, and ID3D12Resource's IID.
        let result = unsafe {
            let create: unsafe extern "system" fn(
                *mut c_void,
                *const HeapProperties,
                u32,
                *const ResourceDesc,
                u32,
                *const c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_COMMITTED_RESOURCE);
            create(
                self.device.raw(),
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                null(),
                &IID_ID3D12_RESOURCE,
                &mut resource,
            )
        };
        let resource = Com::made("CreateCommittedResource", result, resource)?;
        Ok(SharedTexture {
            resource,
            device: self.device.raw().addr(),
            format,
            size: (width, height),
            life: Arc::new(()),
            let_go: LetGoes::default(),
        })
    }

    /// Copies the whole of `from` into `to`, textures of the same format and size, on the
    /// device's copy queue, and waits for the copy until `wait` has passed.
    ///
    /// A copy not seen done, by then or because a call failed once the copy was sent, comes back
    /// with what it may still use ([`CopyError::Late`], [`CopyError::Failed`]), for the caller to
    /// keep until the GPU is done with it ([`InFlight`]); its queue's objects are the copy's from
    /// then on, and the next copy makes others.
    pub(crate) fn copy(
        &self,
        from: &Arc<SharedTexture>,
        to: &Arc<SharedTexture>,
        wait: Duration,
    ) -> Result<(), CopyError> {
        let taken = self.copies.lock().take();
        let mut copies = match taken {
            Some(copies) => copies,
            None => self.copies().map_err(|why| CopyError::Failed(why, None))?,
        };
        // SAFETY: each method is the named one of its object's interface, called on objects of
        // this device's: the allocator and list are reset once the last copy's fence completed
        // (the wait below; a copy not seen done takes the objects with it), the list records one
        // copy between two resources of this device's, which are in the common state, promoted
        // to the copy's states and back on a copy queue.
        let sent = unsafe {
            let reset: unsafe extern "system" fn(*mut c_void) -> Hresult =
                copies.allocator.method(ALLOCATOR_RESET);
            let reset_list: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
            ) -> Hresult = copies.list.method(LIST_RESET);
            let copy: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) =
                copies.list.method(LIST_COPY_RESOURCE);
            let close: unsafe extern "system" fn(*mut c_void) -> Hresult =
                copies.list.method(LIST_CLOSE);
            check(
                "ID3D12CommandAllocator::Reset",
                reset(copies.allocator.raw()),
            )
            .and_then(|()| {
                check(
                    "ID3D12GraphicsCommandList::Reset",
                    reset_list(copies.list.raw(), copies.allocator.raw(), null_mut()),
                )
            })
            .and_then(|()| {
                copy(copies.list.raw(), to.resource.raw(), from.resource.raw());
                check("ID3D12GraphicsCommandList::Close", close(copies.list.raw()))
            })
        };
        // Nothing is sent yet: the objects go with the error.
        sent.map_err(|why| CopyError::Failed(why, None))?;
        copies.value += 1;
        let in_flight = |copies| InFlight {
            copies,
            _textures: [Arc::clone(from), Arc::clone(to)],
        };
        // SAFETY: as above; the list is closed, the fence and event are the copies'.
        let done = unsafe {
            let execute: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) =
                copies.queue.method(QUEUE_EXECUTE_COMMAND_LISTS);
            let lists = [copies.list.raw()];
            execute(copies.queue.raw(), 1, lists.as_ptr());
            let signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> Hresult =
                copies.queue.method(QUEUE_SIGNAL);
            let on_completion: unsafe extern "system" fn(*mut c_void, u64, *mut c_void) -> Hresult =
                copies.fence.method(FENCE_SET_EVENT_ON_COMPLETION);
            check(
                "ID3D12CommandQueue::Signal",
                signal(copies.queue.raw(), copies.fence.raw(), copies.value),
            )
            .and_then(|()| match copies.completed() {
                Completed::Done => Ok(true),
                Completed::Removed => Err("the device was removed".to_owned()),
                Completed::Not => check(
                    "ID3D12Fence::SetEventOnCompletion",
                    on_completion(copies.fence.raw(), copies.value, copies.event.0.as_ptr()),
                )
                .map(|()| {
                    let millis = u32::try_from(wait.as_millis()).unwrap_or(u32::MAX - 1);
                    WaitForSingleObject(copies.event.0.as_ptr(), millis) == WAIT_OBJECT_0
                }),
            })
        };
        match done {
            Ok(true) if copies.completed() == Completed::Done => {
                let mut kept = self.copies.lock();
                if kept.is_none() {
                    *kept = Some(copies);
                }
                Ok(())
            }
            Ok(_) => Err(CopyError::Late(in_flight(copies))),
            Err(why) => Err(CopyError::Failed(why, Some(in_flight(copies)))),
        }
    }

    /// The objects of the device's copies.
    fn copies(&self) -> Result<Copies, String> {
        let desc = CommandQueueDesc {
            type_: D3D12_COMMAND_LIST_TYPE_COPY,
            priority: 0,
            flags: 0,
            node_mask: 0,
        };
        let device = self.device.raw();
        // SAFETY: each method is the named one of ID3D12Device's, with valid descriptions and the
        // IID of the interface made; the list is made on the allocator, closed until a copy.
        unsafe {
            let mut out = null_mut();
            let create_queue: unsafe extern "system" fn(
                *mut c_void,
                *const CommandQueueDesc,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_COMMAND_QUEUE);
            let result = create_queue(device, &desc, &IID_ID3D12_COMMAND_QUEUE, &mut out);
            let queue = Com::made("CreateCommandQueue", result, out)?;
            let mut out = null_mut();
            let create_allocator: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_COMMAND_ALLOCATOR);
            let result = create_allocator(
                device,
                D3D12_COMMAND_LIST_TYPE_COPY,
                &IID_ID3D12_COMMAND_ALLOCATOR,
                &mut out,
            );
            let allocator = Com::made("CreateCommandAllocator", result, out)?;
            let mut out = null_mut();
            let create_list: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *mut c_void,
                *mut c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_COMMAND_LIST);
            let result = create_list(
                device,
                0,
                D3D12_COMMAND_LIST_TYPE_COPY,
                allocator.raw(),
                null_mut(),
                &IID_ID3D12_GRAPHICS_COMMAND_LIST,
                &mut out,
            );
            let list = Com::made("CreateCommandList", result, out)?;
            // A list is made open: closed, so that each copy resets it.
            let close: unsafe extern "system" fn(*mut c_void) -> Hresult = list.method(LIST_CLOSE);
            check("ID3D12GraphicsCommandList::Close", close(list.raw()))?;
            let mut out = null_mut();
            let create_fence: unsafe extern "system" fn(
                *mut c_void,
                u64,
                u32,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_FENCE);
            let result = create_fence(device, 0, 0, &IID_ID3D12_FENCE, &mut out);
            let fence = Com::made("CreateFence", result, out)?;
            let event = NonNull::new(CreateEventW(null(), 0, 0, null()))
                .map(Handle)
                .ok_or_else(|| format!("CreateEventW: {}", std::io::Error::last_os_error()))?;
            Ok(Copies {
                queue,
                allocator,
                list,
                fence,
                value: 0,
                event,
            })
        }
    }

    /// Writes `planes`, the luma's bytes and the chroma's, each row after row with nothing
    /// between them, into `to`, a texture of the device's, from an upload buffer on a copy queue
    /// of its own (`CopyTextureRegion` of each plane's footprint, as `GetCopyableFootprints`
    /// places it), and returns once the copy is done: a picture of a test's own. `Err` where a
    /// plane's length is not its footprint's, a call failed, or the copy was not done in 10 s.
    #[cfg(test)]
    pub(crate) fn upload(&self, to: &SharedTexture, planes: [&[u8]; 2]) -> Result<(), String> {
        const D3D12_HEAP_TYPE_UPLOAD: u32 = 2;
        const D3D12_RESOURCE_DIMENSION_BUFFER: u32 = 1;
        const D3D12_TEXTURE_LAYOUT_ROW_MAJOR: u32 = 1;
        const D3D12_RESOURCE_STATE_GENERIC_READ: u32 = 0xac3;
        const D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX: u32 = 0;
        const D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT: u32 = 1;
        const DEVICE_GET_COPYABLE_FOOTPRINTS: usize = 38;
        const RESOURCE_MAP: usize = 8;
        const RESOURCE_UNMAP: usize = 9;
        const LIST_COPY_TEXTURE_REGION: usize = 16;
        /// `D3D12_PLACED_SUBRESOURCE_FOOTPRINT`: where a subresource's rows start in a buffer, its
        /// format, width, height and depth, and how far apart its rows are.
        #[repr(C)]
        #[derive(Clone, Copy, Default)]
        struct PlacedFootprint {
            offset: u64,
            format: u32,
            width: u32,
            height: u32,
            depth: u32,
            row_pitch: u32,
        }
        /// `D3D12_TEXTURE_COPY_LOCATION`: a resource, the kind of location, and the union of a
        /// placed footprint and a subresource's index, which is the union's first 4 bytes.
        #[repr(C)]
        struct CopyLocation {
            resource: *mut c_void,
            type_: u32,
            footprint: PlacedFootprint,
        }
        if to.device != self.device.raw().addr() {
            return Err("a texture of another Direct3D 12 device".to_owned());
        }
        let format = match to.format {
            SurfaceFormat::Nv12 => DXGI_FORMAT_NV12,
            SurfaceFormat::P010 => DXGI_FORMAT_P010,
            SurfaceFormat::Xyuv => return Err("no shared texture of XYUV".to_owned()),
        };
        let desc = ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            alignment: 0,
            width: u64::from(to.size.0),
            height: to.size.1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        };
        let mut layouts = [PlacedFootprint::default(); 2];
        let mut rows = [0_u32; 2];
        let mut row_sizes = [0_u64; 2];
        let mut total = 0_u64;
        // SAFETY: the method is ID3D12Device's `GetCopyableFootprints`, of the texture's
        // description, for its two planes' subresources, with room for each's answers.
        unsafe {
            let footprints: unsafe extern "system" fn(
                *mut c_void,
                *const ResourceDesc,
                u32,
                u32,
                u64,
                *mut PlacedFootprint,
                *mut u32,
                *mut u64,
                *mut u64,
            ) = self.device.method(DEVICE_GET_COPYABLE_FOOTPRINTS);
            footprints(
                self.device.raw(),
                &desc,
                0,
                2,
                0,
                layouts.as_mut_ptr(),
                rows.as_mut_ptr(),
                row_sizes.as_mut_ptr(),
                &mut total,
            );
        }
        let shapes: Vec<(usize, usize)> = (rows.iter().zip(&row_sizes))
            .map(|(&rows, &size)| (rows as usize, usize::try_from(size).unwrap_or(usize::MAX)))
            .collect();
        for (plane, (&(rows, size), bytes)) in shapes.iter().zip(&planes).enumerate() {
            if bytes.len() != rows * size {
                return Err(format!(
                    "plane {plane}: {} bytes for {rows} rows of {size}",
                    bytes.len()
                ));
            }
        }
        let heap = HeapProperties {
            type_: D3D12_HEAP_TYPE_UPLOAD,
            cpu_page_property: 0,
            memory_pool_preference: 0,
            creation_node_mask: 0,
            visible_node_mask: 0,
        };
        let buffer_desc = ResourceDesc {
            dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            alignment: 0,
            width: total,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            flags: 0,
        };
        let mut out = null_mut();
        // SAFETY: the method is `CreateCommittedResource`, with valid descriptions of a buffer
        // on an upload heap, in the state such a buffer is made in, and ID3D12Resource's IID.
        let result = unsafe {
            let create: unsafe extern "system" fn(
                *mut c_void,
                *const HeapProperties,
                u32,
                *const ResourceDesc,
                u32,
                *const c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_COMMITTED_RESOURCE);
            create(
                self.device.raw(),
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &buffer_desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                null(),
                &IID_ID3D12_RESOURCE,
                &mut out,
            )
        };
        let buffer = Com::made("CreateCommittedResource", result, out)?;
        let mut data = null_mut::<c_void>();
        // SAFETY: the methods are ID3D12Resource's `Map` and `Unmap`, of the buffer's one
        // subresource, which holds `total` bytes: each plane's rows are written at its
        // footprint's offset, `row_pitch` apart, each no longer than the pitch.
        unsafe {
            let map: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *const c_void,
                *mut *mut c_void,
            ) -> Hresult = buffer.method(RESOURCE_MAP);
            check(
                "ID3D12Resource::Map",
                map(buffer.raw(), 0, null(), &mut data),
            )?;
            for ((layout, &(rows, size)), bytes) in layouts.iter().zip(&shapes).zip(&planes) {
                for row in 0..rows {
                    let at = data
                        .cast::<u8>()
                        .add(layout.offset as usize + row * layout.row_pitch as usize);
                    std::ptr::copy_nonoverlapping(bytes[row * size..].as_ptr(), at, size);
                }
            }
            let unmap: unsafe extern "system" fn(*mut c_void, u32, *const c_void) =
                buffer.method(RESOURCE_UNMAP);
            unmap(buffer.raw(), 0, null());
        }
        let mut copies = self.copies()?;
        copies.value = 1;
        // SAFETY: each method is the named one of its object's interface, on objects of this
        // device's: the list, made closed, is reset on its allocator, records a copy of each
        // plane from its footprint in the buffer into the texture's subresource of that plane
        // (the texture in the common state, promoted to the copy's and back on a copy queue),
        // and is closed, then executed, and the fence signalled after it; the buffer lives until
        // the wait below has seen the fence pass, or the copies' objects and it are let go with
        // the error.
        let done = unsafe {
            let reset: unsafe extern "system" fn(*mut c_void) -> Hresult =
                copies.allocator.method(ALLOCATOR_RESET);
            let reset_list: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
            ) -> Hresult = copies.list.method(LIST_RESET);
            let copy: unsafe extern "system" fn(
                *mut c_void,
                *const CopyLocation,
                u32,
                u32,
                u32,
                *const CopyLocation,
                *const c_void,
            ) = copies.list.method(LIST_COPY_TEXTURE_REGION);
            let close: unsafe extern "system" fn(*mut c_void) -> Hresult =
                copies.list.method(LIST_CLOSE);
            let execute: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) =
                copies.queue.method(QUEUE_EXECUTE_COMMAND_LISTS);
            let signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> Hresult =
                copies.queue.method(QUEUE_SIGNAL);
            let on_completion: unsafe extern "system" fn(*mut c_void, u64, *mut c_void) -> Hresult =
                copies.fence.method(FENCE_SET_EVENT_ON_COMPLETION);
            check(
                "ID3D12CommandAllocator::Reset",
                reset(copies.allocator.raw()),
            )?;
            check(
                "ID3D12GraphicsCommandList::Reset",
                reset_list(copies.list.raw(), copies.allocator.raw(), null_mut()),
            )?;
            for (plane, layout) in layouts.iter().enumerate() {
                let into = CopyLocation {
                    resource: to.resource.raw(),
                    type_: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    footprint: PlacedFootprint {
                        offset: plane as u64,
                        ..PlacedFootprint::default()
                    },
                };
                let from = CopyLocation {
                    resource: buffer.raw(),
                    type_: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    footprint: *layout,
                };
                copy(copies.list.raw(), &into, 0, 0, 0, &from, null());
            }
            check("ID3D12GraphicsCommandList::Close", close(copies.list.raw()))?;
            let lists = [copies.list.raw()];
            execute(copies.queue.raw(), 1, lists.as_ptr());
            check(
                "ID3D12CommandQueue::Signal",
                signal(copies.queue.raw(), copies.fence.raw(), copies.value),
            )?;
            check(
                "ID3D12Fence::SetEventOnCompletion",
                on_completion(copies.fence.raw(), copies.value, copies.event.0.as_ptr()),
            )?;
            WaitForSingleObject(copies.event.0.as_ptr(), 10_000) == WAIT_OBJECT_0
        };
        match copies.completed() {
            Completed::Done if done => Ok(()),
            Completed::Removed => Err("the device was removed".to_owned()),
            _ => Err("the upload was not done in 10 s".to_owned()),
        }
    }

    /// What the device answers as why it was removed (`ID3D12Device::GetDeviceRemovedReason`):
    /// `S_OK` (0) where it was not.
    #[cfg(test)]
    pub(super) fn removed_reason(&self) -> Hresult {
        const DEVICE_GET_DEVICE_REMOVED_REASON: usize = 37;
        // SAFETY: the method is ID3D12Device's `GetDeviceRemovedReason`.
        unsafe {
            let reason: unsafe extern "system" fn(*mut c_void) -> Hresult =
                self.device.method(DEVICE_GET_DEVICE_REMOVED_REASON);
            reason(self.device.raw())
        }
    }
}

/// A DXGI factory, as `IDXGIFactory4`.
fn factory() -> Result<Com, String> {
    let create_factory = system_function("dxgi.dll", c"CreateDXGIFactory1")?;
    // SAFETY: the function is `CreateDXGIFactory1` (dxgi.h).
    let create_factory =
        unsafe { std::mem::transmute::<NonNull<c_void>, CreateDxgiFactory1>(create_factory) };
    let mut factory = null_mut();
    // SAFETY: the IID is IDXGIFactory4's.
    let result = unsafe { create_factory(&IID_IDXGI_FACTORY4, &mut factory) };
    Com::made("CreateDXGIFactory1", result, factory)
}

/// The DXGI adapter of `luid`.
pub(super) fn adapter_of(luid: [u8; 8]) -> Result<Com, String> {
    let factory = factory()?;
    let [l0, l1, l2, l3, h0, h1, h2, h3] = luid;
    let luid = Luid {
        low: u32::from_le_bytes([l0, l1, l2, l3]),
        high: i32::from_le_bytes([h0, h1, h2, h3]),
    };
    let mut adapter = null_mut();
    // SAFETY: the method is IDXGIFactory4's `EnumAdapterByLuid`, which takes the LUID by value.
    let result = unsafe {
        let by_luid: unsafe extern "system" fn(
            *mut c_void,
            Luid,
            *const Guid,
            *mut *mut c_void,
        ) -> Hresult = factory.method(FACTORY4_ENUM_ADAPTER_BY_LUID);
        by_luid(factory.raw(), luid, &IID_IDXGI_ADAPTER, &mut adapter)
    };
    Com::made("EnumAdapterByLuid", result, adapter)
}

/// What this process holds on the first adapter DXGI lists, as DXGI counts it, in bytes: in the
/// adapter's local segment group and in its non-local one (`IDXGIAdapter3::QueryVideoMemoryInfo`'s
/// `CurrentUsage`).
#[cfg(test)]
pub(super) fn video_memory_used() -> Result<[u64; 2], String> {
    /// `DXGI_QUERY_VIDEO_MEMORY_INFO`.
    #[repr(C)]
    #[derive(Default)]
    struct VideoMemoryInfo {
        budget: u64,
        current_usage: u64,
        available_for_reservation: u64,
        current_reservation: u64,
    }
    const IID_IDXGI_ADAPTER3: Guid = guid(
        0x6459_67a4,
        0x1392,
        0x4310,
        [0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd],
    );
    const FACTORY1_ENUM_ADAPTERS1: usize = 12;
    const ADAPTER3_QUERY_VIDEO_MEMORY_INFO: usize = 14;
    /// `DXGI_MEMORY_SEGMENT_GROUP_LOCAL` and `DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL`.
    const SEGMENT_GROUPS: [u32; 2] = [0, 1];

    let factory = factory()?;
    let mut adapter = null_mut();
    // SAFETY: the method is IDXGIFactory1's `EnumAdapters1`, which IDXGIFactory4 inherits, given
    // the adapter's index and where to write it.
    let result = unsafe {
        let enum_adapters: unsafe extern "system" fn(
            *mut c_void,
            u32,
            *mut *mut c_void,
        ) -> Hresult = factory.method(FACTORY1_ENUM_ADAPTERS1);
        enum_adapters(factory.raw(), 0, &mut adapter)
    };
    let adapter =
        Com::made("EnumAdapters1", result, adapter)?.query(&IID_IDXGI_ADAPTER3, "IDXGIAdapter3")?;
    let mut used = [0; 2];
    for (group, used) in SEGMENT_GROUPS.into_iter().zip(&mut used) {
        let mut info = VideoMemoryInfo::default();
        // SAFETY: the method is IDXGIAdapter3's `QueryVideoMemoryInfo`, given the adapter's node
        // 0, a segment group and where to write what it holds.
        let result = unsafe {
            let query: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *mut VideoMemoryInfo,
            ) -> Hresult = adapter.method(ADAPTER3_QUERY_VIDEO_MEMORY_INFO);
            query(adapter.raw(), 0, group, &mut info)
        };
        check("QueryVideoMemoryInfo", result)?;
        *used = info.current_usage;
    }
    Ok(used)
}

/// What Windows charges this process on its GPUs, in bytes: the "GPU Process Memory" counters'
/// Shared Usage and Dedicated Usage of the process's instances (one an adapter, named
/// `pid_<pid>_luid_...`), summed, read once. It counts what an open of a texture's NT handle left
/// charged, which DXGI's `QueryVideoMemoryInfo` does not (module doc). Tests only.
#[cfg(test)]
pub(super) fn process_gpu_memory() -> Result<u64, String> {
    /// `PDH_FMT_COUNTERVALUE_ITEM_W` read as `PDH_FMT_LARGE`: the instance's name, then
    /// `PDH_FMT_COUNTERVALUE`'s status and its value.
    #[repr(C)]
    struct Item {
        name: *const u16,
        status: u32,
        value: i64,
    }
    type OpenQuery = unsafe extern "system" fn(*const u16, usize, *mut *mut c_void) -> u32;
    type AddCounter =
        unsafe extern "system" fn(*mut c_void, *const u16, usize, *mut *mut c_void) -> u32;
    type Collect = unsafe extern "system" fn(*mut c_void) -> u32;
    type CounterArray =
        unsafe extern "system" fn(*mut c_void, u32, *mut u32, *mut u32, *mut Item) -> u32;
    type CloseQuery = unsafe extern "system" fn(*mut c_void) -> u32;
    const PDH_FMT_LARGE: u32 = 0x400;
    const PDH_MORE_DATA: u32 = 0x8000_07d2;
    const COUNTERS: [&str; 2] = [
        "\\GPU Process Memory(*)\\Shared Usage",
        "\\GPU Process Memory(*)\\Dedicated Usage",
    ];
    let wide = |text: &str| -> Vec<u16> { text.encode_utf16().chain([0]).collect() };
    let function = |name: &CStr| system_function("pdh.dll", name);
    // SAFETY: each function is pdh.dll's of that name (pdh.h), as the type says.
    let (open, add, collect, array, close) = unsafe {
        (
            std::mem::transmute::<NonNull<c_void>, OpenQuery>(function(c"PdhOpenQueryW")?),
            std::mem::transmute::<NonNull<c_void>, AddCounter>(function(c"PdhAddEnglishCounterW")?),
            std::mem::transmute::<NonNull<c_void>, Collect>(function(c"PdhCollectQueryData")?),
            std::mem::transmute::<NonNull<c_void>, CounterArray>(function(
                c"PdhGetFormattedCounterArrayW",
            )?),
            std::mem::transmute::<NonNull<c_void>, CloseQuery>(function(c"PdhCloseQuery")?),
        )
    };
    let failed = |what: &str, status: u32| format!("{what} failed: {status:#010x}");
    let mut query = null_mut();
    // SAFETY: no data source (the live counters), and where to write the query.
    let status = unsafe { open(null(), 0, &mut query) };
    if status != 0 {
        return Err(failed("PdhOpenQueryW", status));
    }
    let prefix = format!("pid_{}_", std::process::id());
    let read = || -> Result<u64, String> {
        let mut counters = [null_mut(); 2];
        for (path, counter) in COUNTERS.iter().zip(&mut counters) {
            let path = wide(path);
            // SAFETY: the query is open, the path NUL-terminated.
            let status = unsafe { add(query, path.as_ptr(), 0, counter) };
            if status != 0 {
                return Err(failed("PdhAddEnglishCounterW", status));
            }
        }
        // SAFETY: the query is open.
        let status = unsafe { collect(query) };
        if status != 0 {
            return Err(failed("PdhCollectQueryData", status));
        }
        let mut bytes = 0u64;
        for counter in counters {
            let (mut size, mut count) = (0u32, 0u32);
            // SAFETY: a null buffer of no size asks for the size.
            let status =
                unsafe { array(counter, PDH_FMT_LARGE, &mut size, &mut count, null_mut()) };
            if status != PDH_MORE_DATA {
                return Err(failed("PdhGetFormattedCounterArrayW", status));
            }
            let mut items: Vec<Item> =
                Vec::with_capacity(usize::try_from(size).unwrap_or(0) / size_of::<Item>() + 1);
            // SAFETY: the buffer holds `size` bytes, aligned for the items, which the call writes
            // with the names they point at after them.
            let status = unsafe {
                array(
                    counter,
                    PDH_FMT_LARGE,
                    &mut size,
                    &mut count,
                    items.as_mut_ptr(),
                )
            };
            if status != 0 {
                return Err(failed("PdhGetFormattedCounterArrayW", status));
            }
            // SAFETY: the call wrote `count` items.
            unsafe { items.set_len(usize::try_from(count).unwrap_or(0)) };
            for item in &items {
                // SAFETY: each name is NUL-terminated UTF-16 in the buffer, which lives.
                let name = unsafe {
                    let length = (0..).take_while(|&i| *item.name.add(i) != 0).count();
                    String::from_utf16_lossy(std::slice::from_raw_parts(item.name, length))
                };
                if item.status == 0 && name.starts_with(&prefix) {
                    bytes += u64::try_from(item.value).unwrap_or(0);
                }
            }
        }
        Ok(bytes)
    };
    let bytes = read();
    // SAFETY: the query is open, and closed once.
    unsafe { close(query) };
    bytes
}

/// [`process_gpu_memory`] once it has moved by less than 1 MiB over its last 2 s, read every
/// 250 ms, or as it is 30 s on. Tests only.
#[cfg(test)]
pub(super) fn process_gpu_memory_settled() -> u64 {
    use crate::core::runtime::{Runtime as _, StdRuntime};
    // The tests' clock: std's, which the library does not read.
    let clock = StdRuntime;
    let start = clock.now();
    let mut reads = std::collections::VecDeque::new();
    loop {
        let bytes = process_gpu_memory().expect("the GPU Process Memory counters");
        reads.push_back(bytes);
        if reads.len() > 8 {
            reads.pop_front();
        }
        let (low, high) = (reads.iter().min(), reads.iter().max());
        let settled = reads.len() == 8 && high.zip(low).is_some_and(|(h, l)| h - l < 1 << 20);
        if settled || clock.elapsed_since(start) >= Duration::from_secs(30) {
            return bytes;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// The device of the display opened on `luid`, made at its first use and kept; `Err` with why
/// where none is made, for good.
pub(crate) fn device_of(
    kept: &OnceLock<Result<Device, String>>,
    luid: Option<[u8; 8]>,
) -> Result<&Device, String> {
    kept.get_or_init(|| Device::new(luid))
        .as_ref()
        .map_err(Clone::clone)
}

#[cfg(test)]
impl Com {
    /// The references the object has beside the one this holds: `AddRef`, then `Release`, which
    /// answers the object's count after it, ours among them.
    pub(super) fn references_beside_ours(&self) -> u32 {
        // SAFETY: `AddRef` and `Release` are every COM object's methods 1 and 2: the reference
        // added is released at once, and the one this holds stays.
        let count = unsafe {
            let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = self.method(ADD_REF);
            let release: unsafe extern "system" fn(*mut c_void) -> u32 = self.method(RELEASE);
            add_ref(self.raw());
            release(self.raw())
        };
        count - 1
    }
}

/// The references the default adapter's Direct3D 12 device has beside one of ours: the device
/// `D3D12CreateDevice` answers while one lives (one a process and adapter), with what holds it,
/// else a new one, with none. Tests only.
#[cfg(test)]
pub(super) fn default_adapter_device_references() -> u32 {
    let device = Device::new(None).expect("a Direct3D 12 device");
    device.device.references_beside_ours()
}

/// [`default_adapter_device_references`] once they are none, or as they are 5 s on, for a test
/// whose last release may be made on another thread (a display's `vaTerminate`, ordered behind
/// its call thread's calls). Tests only.
#[cfg(test)]
pub(super) fn default_adapter_device_references_once_let_go() -> u32 {
    use crate::core::runtime::{Runtime as _, StdRuntime};
    // The tests' clock: std's, which the library does not read.
    let clock = StdRuntime;
    let start = clock.now();
    loop {
        let left = default_adapter_device_references();
        if left == 0 || clock.elapsed_since(start) >= Duration::from_secs(5) {
            return left;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}
