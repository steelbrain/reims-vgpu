//! Direct3D 12, on Windows, as much of it as the module needs: a device on the display's DXGI
//! adapter, shared NV12 and P010 textures that the display's pools are made of, and copies
//! between them.
//!
//! A pool's surfaces on Windows are textures of ours, made here on the adapter's device with a
//! shared heap (`D3D12_HEAP_FLAG_SHARED`), each opened by Mesa's VA driver (`vaon12`) through its
//! NT handle (`VA_SURFACE_ATTRIB_MEM_TYPE_NTHANDLE`), so that what decodes into them can be read
//! by others that open the same handle: the Vulkan platform's copy of a picture into the guest's
//! planes. vaon12's own surfaces are not shared, and their export fails (measured on the UHD 630,
//! Mesa 26.2.7, 2026-11-16). A copy of one texture into another runs on a copy queue of this
//! device's, which the module waits for on its fence.
//!
//! The binding is written by hand: the functions are loaded from `d3d12.dll` and `dxgi.dll` with
//! `LoadLibraryW` and `GetProcAddress`, as libva is (`super::libva`), and each COM method is
//! called through its object's vtable at the index the Windows SDK's headers give it (MSYS2
//! CLANG64's `d3d12.h` and `dxgi1_4.h`, read 2026-11-16). Only methods that return nothing, a
//! scalar or an `HRESULT` are called: one that returns a structure has another calling
//! convention.

use std::ffi::{CStr, c_void};
use std::ptr::{NonNull, null, null_mut};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crate::core::sync::Mutex;

use super::surface::SurfaceFormat;

type Hresult = i32;

#[repr(C)]
struct Guid {
    data1: u32,
    data2: u16,
    data3: u16,
    data4: [u8; 8],
}

const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
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
const IID_ID3D12_DEVICE: Guid = guid(
    0x1898_19f1,
    0x1db6,
    0x4b57,
    [0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7],
);
const IID_ID3D12_RESOURCE: Guid = guid(
    0x6964_42be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);
const IID_ID3D12_COMMAND_QUEUE: Guid = guid(
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
const IID_ID3D12_FENCE: Guid = guid(
    0x0a75_3dcf,
    0xc4d8,
    0x4b91,
    [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
);

// The vtable indices of the methods called (the SDK's `<interface>Vtbl` structures).
const RELEASE: usize = 2;
const FACTORY4_ENUM_ADAPTER_BY_LUID: usize = 26;
const DEVICE_CREATE_COMMAND_QUEUE: usize = 8;
const DEVICE_CREATE_COMMAND_ALLOCATOR: usize = 9;
const DEVICE_CREATE_COMMAND_LIST: usize = 12;
const DEVICE_CREATE_COMMITTED_RESOURCE: usize = 27;
const DEVICE_CREATE_SHARED_HANDLE: usize = 31;
const DEVICE_CREATE_FENCE: usize = 36;
const QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;
const QUEUE_SIGNAL: usize = 14;
const ALLOCATOR_RESET: usize = 8;
const LIST_CLOSE: usize = 9;
const LIST_RESET: usize = 10;
const LIST_COPY_RESOURCE: usize = 17;
const FENCE_GET_COMPLETED_VALUE: usize = 8;
const FENCE_SET_EVENT_ON_COMPLETION: usize = 9;

const DXGI_FORMAT_NV12: u32 = 103;
const DXGI_FORMAT_P010: u32 = 104;
const D3D_FEATURE_LEVEL_11_0: u32 = 0xb000;
const D3D12_HEAP_TYPE_DEFAULT: u32 = 1;
const D3D12_HEAP_FLAG_SHARED: u32 = 0x1;
const D3D12_RESOURCE_DIMENSION_TEXTURE2D: u32 = 3;
const D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET: u32 = 0x1;
const D3D12_RESOURCE_STATE_COMMON: u32 = 0;
const D3D12_COMMAND_LIST_TYPE_COPY: u32 = 3;
const GENERIC_ALL: u32 = 0x1000_0000;
const WAIT_OBJECT_0: u32 = 0;

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
struct CommandQueueDesc {
    type_: u32,
    priority: i32,
    flags: u32,
    node_mask: u32,
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
    fn CreateEventW(
        attributes: *const c_void,
        manual_reset: i32,
        initial_state: i32,
        name: *const u16,
    ) -> *mut c_void;
    fn WaitForSingleObject(handle: *mut c_void, milliseconds: u32) -> u32;
}

type CreateDxgiFactory1 = unsafe extern "system" fn(*const Guid, *mut *mut c_void) -> Hresult;
type D3d12CreateDevice =
    unsafe extern "system" fn(*mut c_void, u32, *const Guid, *mut *mut c_void) -> Hresult;

/// The address of `symbol` in the system library `library`, loaded once and never unloaded.
fn system_function(library: &str, symbol: &CStr) -> Result<NonNull<c_void>, String> {
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
fn check(what: &str, result: Hresult) -> Result<(), String> {
    if result < 0 {
        // The HRESULT's bits, as Windows writes them.
        Err(format!("{what} failed: {:#010x}", result.cast_unsigned()))
    } else {
        Ok(())
    }
}

/// A COM object we hold one reference of, released when dropped.
struct Com(NonNull<c_void>);

// SAFETY: the objects held are Direct3D 12's and DXGI's, which are free-threaded: their methods
// may be called from any thread.
unsafe impl Send for Com {}
// SAFETY: as above.
unsafe impl Sync for Com {}

impl Com {
    /// The object `out` names, made by a call that returned `result`.
    fn made(what: &str, result: Hresult, out: *mut c_void) -> Result<Com, String> {
        check(what, result)?;
        NonNull::new(out)
            .map(Com)
            .ok_or_else(|| format!("{what} made nothing"))
    }

    fn raw(&self) -> *mut c_void {
        self.0.as_ptr()
    }

    /// The method at `index` of the object's vtable, as `F`.
    ///
    /// # Safety
    ///
    /// `F` must be the method's signature, with the object as its first argument.
    unsafe fn method<F: Copy>(&self, index: usize) -> F {
        // SAFETY: a COM object's first field is its vtable, an array of function pointers, of
        // which the caller names one of the object's interface by its index and signature.
        unsafe {
            let vtable = *self.0.as_ptr().cast::<*const *const c_void>();
            std::mem::transmute_copy::<*const c_void, F>(&*vtable.add(index))
        }
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

/// A Direct3D 12 device on the display's adapter, and its copy queue.
pub(crate) struct Device {
    device: Com,
    copies: Mutex<Option<Copies>>,
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
struct Handle(NonNull<c_void>);

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

/// A shared texture of a pool's: the resource, and the NT handle that opens it elsewhere.
pub(crate) struct SharedTexture {
    resource: Com,
    handle: Handle,
}

impl SharedTexture {
    /// The NT handle that opens the texture, which stays the texture's.
    pub(crate) fn handle(&self) -> *mut c_void {
        self.handle.0.as_ptr()
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
        Ok(Device {
            device: Com::made("D3D12CreateDevice", result, device)?,
            copies: Mutex::new(None),
        })
    }

    /// A texture of `format`, `width` by `height`, in a shared heap, and its NT handle.
    pub(crate) fn shared_texture(
        &self,
        format: SurfaceFormat,
        width: u32,
        height: u32,
    ) -> Result<SharedTexture, String> {
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
            format: match format {
                SurfaceFormat::Nv12 => DXGI_FORMAT_NV12,
                SurfaceFormat::P010 => DXGI_FORMAT_P010,
            },
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
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                null(),
                &IID_ID3D12_RESOURCE,
                &mut resource,
            )
        };
        let resource = Com::made("CreateCommittedResource", result, resource)?;
        let mut handle = null_mut();
        // SAFETY: the method is `CreateSharedHandle`, of a resource of this device's in a shared
        // heap, with no security attributes and no name.
        let result = unsafe {
            let share: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const c_void,
                u32,
                *const u16,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_SHARED_HANDLE);
            share(
                self.device.raw(),
                resource.raw(),
                null(),
                GENERIC_ALL,
                null(),
                &mut handle,
            )
        };
        check("CreateSharedHandle", result)?;
        let handle = NonNull::new(handle)
            .map(Handle)
            .ok_or("CreateSharedHandle made no handle")?;
        Ok(SharedTexture { resource, handle })
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
}

/// The DXGI adapter of `luid`.
fn adapter_of(luid: [u8; 8]) -> Result<Com, String> {
    let create_factory = system_function("dxgi.dll", c"CreateDXGIFactory1")?;
    // SAFETY: the function is `CreateDXGIFactory1` (dxgi.h).
    let create_factory =
        unsafe { std::mem::transmute::<NonNull<c_void>, CreateDxgiFactory1>(create_factory) };
    let mut factory = null_mut();
    // SAFETY: the IID is IDXGIFactory4's.
    let result = unsafe { create_factory(&IID_IDXGI_FACTORY4, &mut factory) };
    let factory = Com::made("CreateDXGIFactory1", result, factory)?;
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
