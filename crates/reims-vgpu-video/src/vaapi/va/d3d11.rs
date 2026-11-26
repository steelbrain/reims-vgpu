//! Direct3D 11 on Windows: a picture's planes drawn from the pool texture it decoded into
//! (`Frame::texture`) into two textures of one plane each, which another device imports as
//! images of one plane ([`PlaneDraw`], [`PlaneTargets`]).
//!
//! A pool's texture is one NV12 or P010 resource of two planes, which Vulkan on the UHD 630
//! imports only as a whole (its plane views refused), while it imports a Direct3D 11 texture of
//! one plane exactly (measured 2026-11-19, Intel UHD 630, Vulkan driver 101.2145). So each
//! picture is drawn, once its decode is done, by the GPU into two textures of the native
//! Direct3D 11 device on the decode's adapter, shared by NT handle with no keyed mutex
//! (`D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`): the luma as `R8_UNORM` (`R16_UNORM` for 10-bit samples) and the chroma's Cb and Cr samples packed into
//! one texel, `Cb | Cr << 8` as `R16_UINT` (`<< 16`, `R32_UINT`), which reads as `RG8Unorm`
//! (`RG16Unorm`) in a view of the same bits. The draws run on a Direct3D 11 device over a
//! Direct3D 12 device of ours (D3D11On12), the pools' own device (one a process and adapter,
//! `super::d3d12`), which wraps each pool texture's resource (`CreateWrappedResource`) rather
//! than open an NT handle of it, an open that would stay charged for good (`super::d3d12`), and
//! reads its planes through views of `R8_UINT` and `R8G8_UINT` (`R16_UINT` and `R16G16_UINT`):
//! one triangle a plane over the crop's extent, each pixel its texel from the crop's origin, so
//! the targets hold what the CPU's copy writes, at their first texel, and nothing past it.
//!
//! A picture's targets are made without the [`PlaneDraw`]'s lock, as both devices' methods are
//! free-threaded (neither is made single-threaded); the draws' device opens each by its NT handle
//! (`OpenSharedResource1`) at its make. A draw records its writes and submits them with the
//! queue's fence signalled after them, under the lock; it then waits on its caller's thread for
//! the fence without the lock, so that another picture's draw is recorded while the GPU
//! writes. Once the wait returns the importer may read the targets: the caller says so to it
//! (`external_writes_done`). Nothing else orders the writes before the reads: the flush, the
//! fence's wait and then the importer's acquire do, and no device takes a keyed mutex.
//!
//! A driver may refuse the draws' device's open of a target shared so. [`PlaneDraw::targets`]
//! then answers the call that refused and its HRESULT (`OpenSharedResource1 failed:
//! 0x80070057`, as `super::d3d12`'s calls write theirs), and makes no targets of another kind in
//! their place: the caller lands those pictures by the CPU's copy.
//!
//! What a picture's draw would otherwise pay the first time is made with the [`PlaneDraw`], at a
//! decoder's open: the devices, the shaders, compiled by `D3DCompile` at run time, and the
//! pipelines of the four target formats, by a draw into a texture of each. A pool texture's open
//! (the wrap of its resource) and views are made at its first picture, and kept while it lives:
//! the texture tells the draws as it goes (`SharedTexture::on_let_go`), and they let go of its
//! open then, on the thread that let go of the texture, and flush as a pair of targets let go does
//! (below). An open keeps the texture's memory, so a pool's goes as the pool does, not at the
//! draws' next picture of another pool, which may never come.
//!
//! Direct3D 11 destroys an object whose last reference goes only at a later flush of its
//! device's immediate context ("deferred destruction", `ID3D11DeviceContext::Flush`'s
//! remarks). The native device's context records nothing in a draw, so its targets' textures
//! would never be destroyed, and the draws' device would destroy their opens only at its next
//! draw, all of a held session's at once. So a pair of targets let go flushes both contexts,
//! under the lock, once its objects are released ([`PlaneTargets`]' last field): its textures
//! and their memory go as it does. A pool texture's open let go flushes them the same way.
//!
//! Pairs let go of together, as a session's are at its end on several threads, share their
//! flushes ([`Flushes`]), and so do pool textures' opens: a pair let go of while another's flush
//! runs waits for the next flush, which the first of the waiting pairs makes, once, for every
//! pair released before it began, and each pair's drop returns once a flush begun after its
//! release is done. So a burst takes the lock and flushes once for the pairs let go of
//! meanwhile, not once a pair, and a flush destroys no more than the pairs whose drops wait for
//! it, one a thread: a picture's draw waiting for the lock waits for no more than that.
//!
//! A target's bytes are undefined at its make. Its owner may say they are to read zeros wherever
//! no draw writes them ([`PlaneTargets::owe_zeros`]): the next draw whose extent leaves part of
//! them unwritten then clears both to zeros first, in the same submission and the same wait,
//! and a draw that writes them whole clears nothing; where no draw is to come,
//! [`PlaneDraw::zero_owed`] clears them alone. Either way they owe no zeros after, nor once their
//! owner says another device wrote them whole ([`PlaneTargets::written`]). The zeros are a
//! clear of each target's render target view to 0, which every target format stores as zero
//! bits.
//!
//! The form is made only where both Direct3D 11 devices answer `ExtendedResourceSharing`
//! (`D3D11_FEATURE_D3D11_OPTIONS`), which sharing the targets' formats needs; elsewhere
//! [`PlaneDraw::new`] says why not, and the caller copies by the CPU.
//!
//! **Measured** (2026-11-19, Intel UHD 630, Mesa 26.2.7's `vaon12`): drawn so and read back by
//! Vulkan, 600 pictures of 1080p and 4K NV12 and 4K P010 matched `vaGetImage`'s exactly; a draw,
//! its flush and the wait took 1.48 ms at the mean for a 1080p picture, 3.74 ms for a 4K NV12
//! one and 3.87 ms for a 4K P010 one, and the first picture of a device with no pipelines made
//! 24.8 ms. With targets shared with no keyed mutex (2026-11-23, the same GPU and driver, job
//! m3-max-70852), 360 pictures of 1080p NV12 and 4K P010 read back by the native device matched
//! `vaGetImage`'s exactly, and a draw, its flush and the wait took 1.511 ms at the mean for a
//! 1080p picture and 3.970 ms for a 4K P010 one.
//!
//! The binding is written by hand, as `super::d3d12`'s, whose COM objects it shares: the
//! functions are loaded from `d3d11.dll` and `d3dcompiler_47.dll`, and each method is called
//! through its object's vtable at the index the Windows SDK's headers give it (MinGW-w64's
//! `d3d11.h`, `d3d11_1.h`, `d3d11on12.h`, `dxgi.h` and `d3dcommon.h`, read 2026-11-19).

use std::ffi::{CStr, c_void};
use std::ptr::{NonNull, null, null_mut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use crate::core::sync::{Condvar, Mutex};

use super::d3d12::{
    Com, CommandQueueDesc, CreateEventW, DEVICE_CREATE_COMMAND_QUEUE, DEVICE_CREATE_FENCE, Device,
    FENCE_GET_COMPLETED_VALUE, FENCE_SET_EVENT_ON_COMPLETION, Guid, Handle, Hresult,
    IID_ID3D12_COMMAND_QUEUE, IID_ID3D12_FENCE, LetGo, QUEUE_SIGNAL,
    SharedTexture, WAIT_OBJECT_0, WaitForSingleObject, adapter_of, check, guid, system_function,
};

const IID_ID3D11_ON12_DEVICE: Guid = guid(
    0x8561_1e73,
    0x70a9,
    0x490e,
    [0x96, 0x14, 0xa9, 0xe3, 0x02, 0x77, 0x79, 0x04],
);
const IID_ID3D11_DEVICE1: Guid = guid(
    0xa04b_fb29,
    0x08ef,
    0x43d6,
    [0xa4, 0x9c, 0xa9, 0xbd, 0xbd, 0xcb, 0xe6, 0x86],
);
const IID_ID3D11_TEXTURE2D: Guid = guid(
    0x6f15_aaf2,
    0xd208,
    0x4e89,
    [0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c],
);
const IID_IDXGI_RESOURCE1: Guid = guid(
    0x3096_1379,
    0x4609,
    0x4a41,
    [0x99, 0x8e, 0x54, 0xfe, 0x56, 0x7e, 0xe0, 0xc1],
);

// The vtable indices of the methods called (the SDK's `<interface>Vtbl` structures).
const DEVICE_CREATE_BUFFER: usize = 3;
const DEVICE_CREATE_TEXTURE_2D: usize = 5;
const DEVICE_CREATE_SHADER_RESOURCE_VIEW: usize = 7;
const DEVICE_CREATE_RENDER_TARGET_VIEW: usize = 9;
const DEVICE_CREATE_VERTEX_SHADER: usize = 12;
const DEVICE_CREATE_PIXEL_SHADER: usize = 15;
const DEVICE_CHECK_FEATURE_SUPPORT: usize = 33;
const DEVICE1_OPEN_SHARED_RESOURCE_1: usize = 48;
const CONTEXT_PS_SET_SHADER_RESOURCES: usize = 8;
const CONTEXT_PS_SET_SHADER: usize = 9;
const CONTEXT_VS_SET_SHADER: usize = 11;
const CONTEXT_DRAW: usize = 13;
const CONTEXT_MAP: usize = 14;
const CONTEXT_UNMAP: usize = 15;
const CONTEXT_PS_SET_CONSTANT_BUFFERS: usize = 16;
const CONTEXT_IA_SET_INPUT_LAYOUT: usize = 17;
const CONTEXT_IA_SET_PRIMITIVE_TOPOLOGY: usize = 24;
const CONTEXT_OM_SET_RENDER_TARGETS: usize = 33;
const CONTEXT_RS_SET_VIEWPORTS: usize = 44;
const CONTEXT_COPY_RESOURCE: usize = 47;
const CONTEXT_UPDATE_SUBRESOURCE: usize = 48;
const CONTEXT_CLEAR_RENDER_TARGET_VIEW: usize = 50;
const CONTEXT_FLUSH: usize = 111;
const ON12_CREATE_WRAPPED_RESOURCE: usize = 3;
const ON12_RELEASE_WRAPPED_RESOURCES: usize = 4;
const ON12_ACQUIRE_WRAPPED_RESOURCES: usize = 5;
const RESOURCE1_CREATE_SHARED_HANDLE: usize = 13;
const BLOB_GET_BUFFER_POINTER: usize = 3;
const BLOB_GET_BUFFER_SIZE: usize = 4;

const DXGI_FORMAT_R32_UINT: u32 = 42;
const DXGI_FORMAT_R16G16_UINT: u32 = 36;
const DXGI_FORMAT_R8G8_UINT: u32 = 50;
const DXGI_FORMAT_R16_UNORM: u32 = 56;
const DXGI_FORMAT_R16_UINT: u32 = 57;
const DXGI_FORMAT_R8_UNORM: u32 = 61;
const DXGI_FORMAT_R8_UINT: u32 = 62;
const DXGI_SHARED_RESOURCE_READ_WRITE: u32 = 0x8000_0001;
const D3D_DRIVER_TYPE_UNKNOWN: u32 = 0;
const D3D_DRIVER_TYPE_HARDWARE: u32 = 1;
const D3D11_SDK_VERSION: u32 = 7;
const D3D11_FEATURE_D3D11_OPTIONS: u32 = 5;
const D3D11_BIND_CONSTANT_BUFFER: u32 = 0x4;
const D3D11_BIND_SHADER_RESOURCE: u32 = 0x8;
const D3D11_BIND_RENDER_TARGET: u32 = 0x20;
/// The `BindFlags` a [`Target`] is made with: drawn into and read.
const TARGET_BIND: u32 = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
const D3D11_RESOURCE_MISC_SHARED: u32 = 0x2;
const D3D11_RESOURCE_MISC_SHARED_NTHANDLE: u32 = 0x800;
const D3D11_SRV_DIMENSION_TEXTURE2D: u32 = 4;
const D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST: u32 = 4;
const D3D12_RESOURCE_STATE_COMMON: u32 = 0;
const D3D12_COMMAND_LIST_TYPE_DIRECT: u32 = 0;
const INFINITE: u32 = u32::MAX;

/// The vertex shader: one triangle over the viewport, from the vertex's index alone.
const VERTEX_SHADER: &str = "float4 main(uint id : SV_VertexID) : SV_Position {
    float2 t = float2((id << 1) & 2, id & 2);
    return float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
}";

/// The luma's pixel shader: the texel at the pixel from the crop's origin (`o.xy`), as a
/// normalised value of `MAX`, which the target's `UNORM` format stores back as the same bits.
const LUMA_SHADER: &str = "Texture2D<uint> t : register(t0);
cbuffer origin : register(b0) { uint4 o; };
float main(float4 p : SV_Position) : SV_Target {
    return t.Load(int3(int2(p.xy) + int2(o.xy), 0)) / MAX;
}";

/// The chroma's pixel shader: the Cb and Cr texel at the pixel from half the crop's origin
/// (`o.zw`), packed into one value, Cb in the low `SHIFT` bits.
const CHROMA_SHADER: &str = "Texture2D<uint2> t : register(t0);
cbuffer origin : register(b0) { uint4 o; };
uint main(float4 p : SV_Position) : SV_Target {
    uint2 c = t.Load(int3(int2(p.xy) + int2(o.zw), 0));
    return c.x | (c.y << SHIFT);
}";

type D3d11CreateDevice = unsafe extern "system" fn(
    *mut c_void,
    u32,
    *mut c_void,
    u32,
    *const u32,
    u32,
    u32,
    *mut *mut c_void,
    *mut u32,
    *mut *mut c_void,
) -> Hresult;
type D3d11On12CreateDevice = unsafe extern "system" fn(
    *mut c_void,
    u32,
    *const u32,
    u32,
    *const *mut c_void,
    u32,
    u32,
    *mut *mut c_void,
    *mut *mut c_void,
    *mut u32,
) -> Hresult;
type D3dCompile = unsafe extern "system" fn(
    *const c_void,
    usize,
    *const std::ffi::c_char,
    *const c_void,
    *mut c_void,
    *const std::ffi::c_char,
    *const std::ffi::c_char,
    u32,
    u32,
    *mut *mut c_void,
    *mut *mut c_void,
) -> Hresult;

/// `D3D11_TEXTURE2D_DESC`.
#[repr(C)]
struct Texture2dDesc {
    width: u32,
    height: u32,
    mip_levels: u32,
    array_size: u32,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    usage: u32,
    bind_flags: u32,
    cpu_access_flags: u32,
    misc_flags: u32,
}

/// `D3D11_SHADER_RESOURCE_VIEW_DESC` of a `TEXTURE2D` view: its format, dimension, first mip
/// and mips, and the rest of the union, 16 bytes in all.
#[repr(C)]
struct ShaderResourceViewDesc {
    format: u32,
    dimension: u32,
    most_detailed_mip: u32,
    mip_levels: u32,
    rest: [u32; 2],
}

/// `D3D11_BUFFER_DESC`.
#[repr(C)]
struct BufferDesc {
    byte_width: u32,
    usage: u32,
    bind_flags: u32,
    cpu_access_flags: u32,
    misc_flags: u32,
    structure_byte_stride: u32,
}

/// `D3D11_VIEWPORT`.
#[repr(C)]
struct Viewport {
    top_left_x: f32,
    top_left_y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
}

/// `D3D11_RESOURCE_FLAGS`, of a wrapped Direct3D 12 resource.
#[repr(C)]
struct ResourceFlags {
    bind_flags: u32,
    misc_flags: u32,
    cpu_access_flags: u32,
    structure_byte_stride: u32,
}

/// The formats of a target pair and of the views a pool texture's planes are read through, by
/// the samples' depth: luma's and chroma's.
#[derive(Clone, Copy)]
struct Formats {
    targets: [u32; 2],
    views: [u32; 2],
}

/// [`Formats`] of 8-bit samples (NV12) and of 10-bit ones in 16 (P010).
const FORMATS: [Formats; 2] = [
    Formats {
        targets: [DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R16_UINT],
        views: [DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8G8_UINT],
    },
    Formats {
        targets: [DXGI_FORMAT_R16_UNORM, DXGI_FORMAT_R32_UINT],
        views: [DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16_UINT],
    },
];

/// The [`FORMATS`] of samples 10-bit in 16, else 8-bit.
fn formats(ten_bit: bool) -> Formats {
    FORMATS[usize::from(ten_bit)]
}

/// The draws of pictures' planes on one adapter (module doc): its devices, shaders and fence,
/// made at a decoder's open, and the pool textures opened so far, under one lock, so that one
/// picture's draw is recorded at a time; and, outside it, the devices again, which make a
/// picture's targets, and the fence, which a draw waits for.
pub struct PlaneDraw {
    /// Shared with each pair of targets made, whose release flushes its contexts (module doc).
    state: Arc<Mutex<State>>,
    /// The flushes the pairs let go of share (module doc), shared with each pair made.
    flushes: Arc<Flushes>,
    makers: Makers,
    fence: Com,
}

/// A [`PlaneDraw`]'s objects.
struct State {
    /// The Direct3D 12 device the draws run on, which opens the pool textures.
    d3d12: Device,
    /// Its direct queue, which the Direct3D 11 device records onto.
    queue: Com,
    /// The Direct3D 11 device over it, its immediate context, and its `ID3D11On12Device`.
    device: Com,
    context: Com,
    on12: Com,
    /// The native Direct3D 11 device on the adapter, which makes the targets, and its context.
    native: Com,
    native_context: Com,
    vertex: Com,
    /// The pixel shaders of 8-bit and 10-bit samples: luma's and chroma's.
    pixel: [[Com; 2]; 2],
    /// The constant buffer of the crop's origin: its column and row, and half of each.
    origin: Com,
    fence: Com,
    /// The fence's last value signalled.
    value: u64,
    /// The pool textures opened, each kept while it lives, and let go as it goes (module doc).
    opened: Vec<Opened>,
}

/// The devices a picture's targets are made on, without the [`PlaneDraw`]'s lock (module doc).
struct Makers {
    /// The native Direct3D 11 device, which makes the targets.
    native: Com,
    /// The draws' Direct3D 11 device, which opens them.
    device: Com,
}

/// A pool texture opened on the draws' device (module doc).
struct Opened {
    /// The texture's life, which the texture's drop names the entry by (module doc).
    texture: Weak<()>,
    /// The texture's Direct3D 12 resource, a reference of the draws' own.
    _resource: Com,
    /// The resource wrapped for the Direct3D 11 device, and its luma and chroma views.
    wrapped: Com,
    views: [Com; 2],
}

/// The two textures a picture's planes are drawn into (module doc): the luma and the chroma.
pub struct PlaneTargets {
    planes: [Target; 2],
    ten_bit: bool,
    size: (u32, u32),
    /// They are to read zeros wherever no draw writes them, and no draw or clear has since
    /// (module doc).
    zeros_owed: AtomicBool,
    /// The flushes that destroy their objects once released (module doc). Last: a struct's
    /// fields drop in their order, so this drops after the planes.
    _destroyed: Destroyed,
}

/// The flushes of a [`PlaneDraw`]'s two immediate contexts, under its lock, as a pair of targets
/// goes, once the pair's objects are released (module doc).
struct Destroyed {
    state: Arc<Mutex<State>>,
    flushes: Arc<Flushes>,
}

impl Drop for Destroyed {
    /// The pair's objects are released by now (its last field): it flushes them
    /// ([`flush_released`]).
    fn drop(&mut self) {
        flush_released(&self.state, &self.flushes);
    }
}

/// Destroys what the caller released of `state`'s objects (module doc): asks for a flush and
/// returns once one begun after its ask is done. One that runs already began before it, so it
/// waits for the next, which it makes where no other caller waiting began it first.
fn flush_released(state: &Mutex<State>, flushes: &Flushes) {
    let mut owed = flushes.owed.lock();
    owed.asked += 1;
    let mine = owed.asked;
    while owed.covered < mine {
        if owed.flushing {
            flushes.done.wait(&mut owed);
            continue;
        }
        owed.flushing = true;
        drop(owed);
        let locked = state.lock();
        // Every caller that asked by now released its objects before it asked: this flush,
        // begun after, destroys them.
        let covers = flushes.owed.lock().asked;
        locked.flush_destroyed();
        drop(locked);
        owed = flushes.owed.lock();
        owed.covered = owed.covered.max(covers);
        owed.flushing = false;
        #[cfg(test)]
        {
            owed.made += 1;
        }
        flushes.done.notify_all();
    }
}

/// What the draws do as a pool texture they opened goes (`SharedTexture::on_let_go`, module
/// doc): let go of its open, under the lock, and flush it as a pair of targets let go is
/// ([`flush_released`]); nothing once the draws' objects are gone.
fn let_go_open(state: Weak<Mutex<State>>, flushes: Weak<Flushes>) -> LetGo {
    Box::new(move |life: &Weak<()>| {
        let (Some(state), Some(flushes)) = (state.upgrade(), flushes.upgrade()) else {
            return;
        };
        let mut locked = state.lock();
        let had = locked.opened.len();
        locked
            .opened
            .retain(|opened| !Weak::ptr_eq(&opened.texture, life));
        let released = locked.opened.len() < had;
        drop(locked);
        if released {
            flush_released(&state, &flushes);
        }
    })
}

/// The flushes the pairs of a [`PlaneDraw`]'s let go of share, and its opens of pool textures
/// let go (module doc).
#[derive(Default)]
struct Flushes {
    owed: Mutex<Owed>,
    /// Told as a flush is done.
    done: Condvar,
}

/// The pairs and opens let go of, and the flushes of them.
#[derive(Default)]
struct Owed {
    /// The pairs and opens that asked for a flush so far, each once its objects were released.
    asked: u64,
    /// The asks the last flush done covered: every one made as it began.
    covered: u64,
    /// A pair's drop, or an open's let go, is making a flush.
    flushing: bool,
    /// The flushes made (tests only).
    #[cfg(test)]
    made: u64,
}

/// One of [`PlaneTargets`]: a texture of the native device, shared by NT handle with no keyed
/// mutex, and opened on the draws' device with a render target view.
struct Target {
    /// The native device's texture, which [`PlaneDraw::read_back`] reads.
    native: Com,
    handle: Handle,
    _opened: Com,
    view: Com,
    size: (u32, u32),
    /// The `BindFlags` the native device made the texture with.
    bind_flags: u32,
}

impl PlaneTargets {
    /// The NT handle of the luma's texture, which stays the targets'.
    pub fn luma_handle(&self) -> std::os::windows::io::BorrowedHandle<'_> {
        self.planes[0].handle()
    }

    /// The NT handle of the chroma's texture, which stays the targets'.
    pub fn chroma_handle(&self) -> std::os::windows::io::BorrowedHandle<'_> {
        self.planes[1].handle()
    }

    /// Whether they hold 10-bit samples in 16 (`R16_UNORM` and `R32_UINT`), else 8-bit
    /// (`R8_UNORM` and `R16_UINT`).
    pub fn is_10_bit(&self) -> bool {
        self.ten_bit
    }

    /// The luma's width and height in samples; the chroma's are half of each, rounded up.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The `BindFlags` (`D3D11_TEXTURE2D_DESC`) the luma's texture and the chroma's were made
    /// with, which an import of them as images of another API's takes.
    pub fn bind_flags(&self) -> [u32; 2] {
        self.planes.each_ref().map(|plane| plane.bind_flags)
    }

    /// Says that they are to read zeros wherever no draw writes them, until a draw or
    /// [`PlaneDraw::zero_owed`] writes them, or [`Self::written`] (module doc).
    pub fn owe_zeros(&self) {
        self.zeros_owed.store(true, Ordering::Release);
    }

    /// Says that another device wrote them whole: they owe no zeros (module doc).
    pub fn written(&self) {
        self.zeros_owed.store(false, Ordering::Release);
    }

    /// Whether they owe zeros ([`Self::owe_zeros`]).
    pub fn owes_zeros(&self) -> bool {
        self.zeros_owed.load(Ordering::Acquire)
    }
}

/// Whether a picture's draw of `extent` luma samples, from their first texel, writes targets of
/// `size` luma samples whole: the extent is at least the size each way, and so the chroma's
/// half of it, rounded up, at least theirs.
fn covers(size: (u32, u32), extent: (u32, u32)) -> bool {
    extent.0 >= size.0 && extent.1 >= size.1
}

impl Target {
    fn handle(&self) -> std::os::windows::io::BorrowedHandle<'_> {
        // SAFETY: the handle is open while the target lives, and the borrow ends with it.
        unsafe { std::os::windows::io::BorrowedHandle::borrow_raw(self.handle.0.as_ptr()) }
    }
}

impl PlaneDraw {
    /// The draws on the DXGI adapter of `luid`, or on the default adapter for `None`, as
    /// `super::d3d12::Device::new` takes it, with their pipelines made (module doc); `Err` with
    /// why where a device answers no `ExtendedResourceSharing` or a call fails.
    pub fn new(luid: Option<[u8; 8]>) -> Result<PlaneDraw, String> {
        let d3d12 = Device::new(luid)?;
        let queue = direct_queue(&d3d12.device)?;
        let (device, context, on12) = on12_device(&d3d12.device, &queue)?;
        let (native, native_context) = native_device(luid)?;
        for (name, device) in [
            ("the D3D11On12 device", &device),
            ("the native one", &native),
        ] {
            if !extended_resource_sharing(device)? {
                return Err(format!("{name} answers no ExtendedResourceSharing"));
            }
        }
        let vertex = shader(
            &device,
            DEVICE_CREATE_VERTEX_SHADER,
            &compile(VERTEX_SHADER, c"vs_5_0")?,
        )?;
        let pixel = |source: String| {
            shader(
                &device,
                DEVICE_CREATE_PIXEL_SHADER,
                &compile(&source, c"ps_5_0")?,
            )
        };
        let pair = |max: &str, shift: &str| -> Result<[Com; 2], String> {
            Ok([
                pixel(LUMA_SHADER.replace("MAX", max))?,
                pixel(CHROMA_SHADER.replace("SHIFT", shift))?,
            ])
        };
        let pixel = [pair("255.0", "8")?, pair("65535.0", "16")?];
        let origin = constant_buffer(&device)?;
        let fence = new_fence(&d3d12.device)?;
        let waits = fence.query(&IID_ID3D12_FENCE, "ID3D12Fence")?;
        let makers = Makers {
            native: native.query(&IID_ID3D11_DEVICE1, "ID3D11Device1")?,
            device: device.query(&IID_ID3D11_DEVICE1, "ID3D11Device1")?,
        };
        let mut state = State {
            d3d12,
            queue,
            device,
            context,
            on12,
            native,
            native_context,
            vertex,
            pixel,
            origin,
            fence,
            value: 0,
            opened: Vec::new(),
        };
        state.make_pipelines()?;
        Ok(PlaneDraw {
            state: Arc::new(Mutex::new(state)),
            flushes: Arc::default(),
            makers,
            fence: waits,
        })
    }

    /// Targets of a picture `size` big in luma samples, of 10-bit samples in 16 or 8-bit ones
    /// (module doc).
    pub fn targets(&self, size: (u32, u32), ten_bit: bool) -> Result<PlaneTargets, String> {
        let (width, height) = size;
        let chroma = (width.div_ceil(2), height.div_ceil(2));
        let formats = formats(ten_bit);
        // Made first, so that a luma target made before the chroma's make fails is destroyed too.
        let destroyed = Destroyed {
            state: Arc::clone(&self.state),
            flushes: Arc::clone(&self.flushes),
        };
        let planes = [
            self.makers.target(formats.targets[0], size)?,
            self.makers.target(formats.targets[1], chroma)?,
        ];
        Ok(PlaneTargets {
            planes,
            ten_bit,
            size,
            zeros_owed: AtomicBool::new(false),
            _destroyed: destroyed,
        })
    }

    /// Draws the picture in `texture`, a pool's, once its decode is done, into `to`: the
    /// `extent.0` by `extent.1` luma samples from `origin` (the crop's first column and row,
    /// even), at the targets' first texel, and half as many chroma samples of each, rounded up,
    /// from half the origin (module doc), cleared to zeros first where they owe zeros and the
    /// extent leaves part of them unwritten ([`PlaneTargets::owe_zeros`]). Returns once the GPU
    /// wrote them, however long it takes, owing no zeros; `Err` with why where a call failed, or
    /// the device was removed, owing what they owed.
    pub fn draw(
        &self,
        texture: &Arc<SharedTexture>,
        origin: (u32, u32),
        to: &PlaneTargets,
        extent: (u32, u32),
    ) -> Result<(), String> {
        if texture.is_10_bit() != to.ten_bit {
            return Err("a picture of other samples than its targets'".to_owned());
        }
        let chroma = (extent.0.div_ceil(2), extent.1.div_ceil(2));
        let [luma_target, chroma_target] = &to.planes;
        if extent.0 > luma_target.size.0
            || extent.1 > luma_target.size.1
            || chroma.0 > chroma_target.size.0
            || chroma.1 > chroma_target.size.1
        {
            return Err(format!(
                "a picture's {}x{} past its targets' {}x{}",
                extent.0, extent.1, to.size.0, to.size.1
            ));
        }
        let mut state = self.state.lock();
        let (at, made) = state.opened(texture, to.ten_bit)?;
        if made {
            // Asked once, as the open is made: the caller holds the texture for the call, so
            // it goes only after.
            texture.on_let_go(let_go_open(
                Arc::downgrade(&self.state),
                Arc::downgrade(&self.flushes),
            ));
        }
        let (x, y) = origin;
        state.set_origin([x, y, x / 2, y / 2]);
        // The owed zeros are taken as the draw is recorded, under the state's lock, as
        // `zero_owed` takes them: no clear of theirs is recorded after a draw that wrote them.
        let owed = to.zeros_owed.swap(false, Ordering::AcqRel);
        if owed && !covers(to.size, extent) {
            state.clear(&to.planes, [0.0; 4]);
        }
        let opened = &state.opened[at];
        let shaders = &state.pixel[usize::from(to.ten_bit)];
        let draws = [
            (luma_target, &opened.views[0], &shaders[0], extent),
            (chroma_target, &opened.views[1], &shaders[1], chroma),
        ];
        state.encode(&opened.wrapped, &draws);
        // The flush submits the draws, and the queue signals the fence after them.
        let signalled = state.flush_and_signal();
        drop(state);
        // The wait for the GPU's writes holds no lock (module doc).
        let drawn = signalled.and_then(|value| wait_for(&self.fence, value));
        if drawn.is_err() && owed {
            to.owe_zeros();
        }
        drawn
    }

    /// Clears `to` to zeros where it owes them ([`PlaneTargets::owe_zeros`]), and returns once
    /// the GPU did, however long it takes; nothing where it owes none. It owes none after,
    /// `Err` with why where a call failed, or the device was removed, included.
    pub fn zero_owed(&self, to: &PlaneTargets) -> Result<(), String> {
        let mut state = self.state.lock();
        if !to.zeros_owed.swap(false, Ordering::AcqRel) {
            return Ok(());
        }
        state.clear(&to.planes, [0.0; 4]);
        state.flush_and_wait()
    }

    /// Clears `to` to `colour`, as a target's format converts it, and returns once the GPU did:
    /// bytes a test's draw is to keep or replace.
    #[cfg(test)]
    fn clear_to(&self, to: &PlaneTargets, colour: [f32; 4]) -> Result<(), String> {
        let mut state = self.state.lock();
        state.clear(&to.planes, colour);
        state.flush_and_wait()
    }

    /// What `targets` hold, for checks: the luma's bytes and the chroma's, row after row with
    /// nothing between them, as the native device reads them back once their last draw is done.
    pub fn read_back(&self, targets: &PlaneTargets) -> Result<[Vec<u8>; 2], String> {
        let state = self.state.lock();
        let sample = if targets.ten_bit { 2 } else { 1 };
        let [luma, chroma] = &targets.planes;
        Ok([
            state.read_back(luma, sample)?,
            state.read_back(chroma, 2 * sample)?,
        ])
    }
}

impl Makers {
    /// A target of `format`, `size` big (`PlaneTargets`).
    fn target(&self, format: u32, size: (u32, u32)) -> Result<Target, String> {
        let native = new_texture(
            &self.native,
            format,
            size,
            TARGET_BIND,
            D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
        )?;
        let handle = share(&native)?;
        let device1 = &self.device;
        let mut out = null_mut();
        // SAFETY: the method is ID3D11Device1's `OpenSharedResource1`, of an NT handle of ours,
        // with ID3D11Texture2D's IID.
        let result = unsafe {
            let open: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = device1.method(DEVICE1_OPEN_SHARED_RESOURCE_1);
            open(
                device1.raw(),
                handle.0.as_ptr(),
                &IID_ID3D11_TEXTURE2D,
                &mut out,
            )
        };
        let opened = Com::made("OpenSharedResource1", result, out)?;
        let mut out = null_mut();
        // SAFETY: the method is ID3D11Device's `CreateRenderTargetView`, of a texture of the
        // device's, in its own format.
        let result = unsafe {
            let create: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const c_void,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_RENDER_TARGET_VIEW);
            create(self.device.raw(), opened.raw(), null(), &mut out)
        };
        let view = Com::made("CreateRenderTargetView", result, out)?;
        Ok(Target {
            native,
            handle,
            _opened: opened,
            view,
            size,
            bind_flags: TARGET_BIND,
        })
    }
}

impl State {
    /// The index in `opened` of `texture`'s open, and whether it is made now, where there was
    /// none.
    fn opened(
        &mut self,
        texture: &Arc<SharedTexture>,
        ten_bit: bool,
    ) -> Result<(usize, bool), String> {
        let life = texture.alive();
        if let Some(at) = (self.opened.iter()).position(|o| Weak::ptr_eq(&o.texture, &life)) {
            return Ok((at, false));
        }
        let resource = texture.resource(&self.d3d12.device)?;
        let flags = ResourceFlags {
            bind_flags: D3D11_BIND_SHADER_RESOURCE,
            misc_flags: 0,
            cpu_access_flags: 0,
            structure_byte_stride: 0,
        };
        let mut out = null_mut();
        // SAFETY: the method is ID3D11On12Device's `CreateWrappedResource`, of a resource of
        // the device's Direct3D 12 device, kept in the common state before and after each use,
        // with ID3D11Texture2D's IID.
        let result = unsafe {
            let wrap: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const ResourceFlags,
                u32,
                u32,
                *const Guid,
                *mut *mut c_void,
            ) -> Hresult = self.on12.method(ON12_CREATE_WRAPPED_RESOURCE);
            wrap(
                self.on12.raw(),
                resource.raw(),
                &flags,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COMMON,
                &IID_ID3D11_TEXTURE2D,
                &mut out,
            )
        };
        let wrapped = Com::made("CreateWrappedResource", result, out)?;
        let formats = formats(ten_bit);
        let views = [
            self.view(&wrapped, formats.views[0])?,
            self.view(&wrapped, formats.views[1])?,
        ];
        self.opened.push(Opened {
            texture: life,
            _resource: resource,
            wrapped,
            views,
        });
        Ok((self.opened.len() - 1, true))
    }

    /// A view of `resource`'s plane of `format`, as the plane's format names it.
    fn view(&self, resource: &Com, format: u32) -> Result<Com, String> {
        let desc = ShaderResourceViewDesc {
            format,
            dimension: D3D11_SRV_DIMENSION_TEXTURE2D,
            most_detailed_mip: 0,
            mip_levels: 1,
            rest: [0; 2],
        };
        let mut out = null_mut();
        // SAFETY: the method is ID3D11Device's `CreateShaderResourceView`, of a resource of the
        // device's, with a valid description.
        let result = unsafe {
            let create: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const ShaderResourceViewDesc,
                *mut *mut c_void,
            ) -> Hresult = self.device.method(DEVICE_CREATE_SHADER_RESOURCE_VIEW);
            create(self.device.raw(), resource.raw(), &desc, &mut out)
        };
        Com::made("CreateShaderResourceView", result, out)
    }

    /// Writes `origin` into the constant buffer the pixel shaders read.
    fn set_origin(&self, origin: [u32; 4]) {
        // SAFETY: the method is ID3D11DeviceContext's `UpdateSubresource`, of the whole of a
        // 16-byte constant buffer of the device's from 16 bytes; the context is called under the
        // state's lock.
        unsafe {
            let update: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                u32,
                *const c_void,
                *const c_void,
                u32,
                u32,
            ) = self.context.method(CONTEXT_UPDATE_SUBRESOURCE);
            update(
                self.context.raw(),
                self.origin.raw(),
                0,
                null(),
                origin.as_ptr().cast(),
                0,
                0,
            );
        }
    }

    /// Records a clear of each of `targets` to `colour`, as its format converts it: 0 is zero bits
    /// in each.
    fn clear(&self, targets: &[Target], colour: [f32; 4]) {
        // SAFETY: the method is ID3D11DeviceContext's `ClearRenderTargetView`, of render target
        // views of the device's, from four floats; the context is called under the state's lock.
        unsafe {
            let clear: unsafe extern "system" fn(*mut c_void, *mut c_void, *const f32) =
                self.context.method(CONTEXT_CLEAR_RENDER_TARGET_VIEW);
            for target in targets {
                clear(self.context.raw(), target.view.raw(), colour.as_ptr());
            }
        }
    }

    /// Records the `draws`, each into its target from its view with its pixel shader over its
    /// extent, from `wrapped`, acquired for them and released after them.
    fn encode(&self, wrapped: &Com, draws: &[(&Target, &Com, &Com, (u32, u32))]) {
        let resources = [wrapped.raw()];
        // SAFETY: the methods are ID3D11On12Device's and ID3D11DeviceContext's named, with
        // objects of the device's; the context is called under the state's lock.
        unsafe {
            let acquire: unsafe extern "system" fn(*mut c_void, *const *mut c_void, u32) =
                self.on12.method(ON12_ACQUIRE_WRAPPED_RESOURCES);
            acquire(self.on12.raw(), resources.as_ptr(), 1);
            self.encode_draws(
                draws.iter().map(|&(target, view, shader, extent)| {
                    (&target.view, Some(view), shader, extent)
                }),
            );
            let release: unsafe extern "system" fn(*mut c_void, *const *mut c_void, u32) =
                self.on12.method(ON12_RELEASE_WRAPPED_RESOURCES);
            release(self.on12.raw(), resources.as_ptr(), 1);
        }
    }

    /// Records one triangle a draw: into its render target view, from its shader resource view
    /// where it names one (none bound otherwise), with its pixel shader, over its extent; then
    /// unbinds the views.
    ///
    /// # Safety
    ///
    /// The views and shaders are the device's, and the caller holds the state's lock.
    unsafe fn encode_draws<'a>(
        &self,
        draws: impl Iterator<Item = (&'a Com, Option<&'a Com>, &'a Com, (u32, u32))>,
    ) {
        let context = self.context.raw();
        // SAFETY: each method is ID3D11DeviceContext's named, with objects of the device's.
        unsafe {
            let set_layout: unsafe extern "system" fn(*mut c_void, *mut c_void) =
                self.context.method(CONTEXT_IA_SET_INPUT_LAYOUT);
            let set_topology: unsafe extern "system" fn(*mut c_void, u32) =
                self.context.method(CONTEXT_IA_SET_PRIMITIVE_TOPOLOGY);
            let set_vertex: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const *mut c_void,
                u32,
            ) = self.context.method(CONTEXT_VS_SET_SHADER);
            let set_constants: unsafe extern "system" fn(
                *mut c_void,
                u32,
                u32,
                *const *mut c_void,
            ) = self.context.method(CONTEXT_PS_SET_CONSTANT_BUFFERS);
            let set_targets: unsafe extern "system" fn(
                *mut c_void,
                u32,
                *const *mut c_void,
                *mut c_void,
            ) = self.context.method(CONTEXT_OM_SET_RENDER_TARGETS);
            let set_viewports: unsafe extern "system" fn(*mut c_void, u32, *const Viewport) =
                self.context.method(CONTEXT_RS_SET_VIEWPORTS);
            let set_views: unsafe extern "system" fn(*mut c_void, u32, u32, *const *mut c_void) =
                self.context.method(CONTEXT_PS_SET_SHADER_RESOURCES);
            let set_pixel: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                *const *mut c_void,
                u32,
            ) = self.context.method(CONTEXT_PS_SET_SHADER);
            let draw: unsafe extern "system" fn(*mut c_void, u32, u32) =
                self.context.method(CONTEXT_DRAW);
            set_layout(context, null_mut());
            set_topology(context, D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            set_vertex(context, self.vertex.raw(), null(), 0);
            let constants = [self.origin.raw()];
            set_constants(context, 0, 1, constants.as_ptr());
            for (target, view, shader, (width, height)) in draws {
                let targets = [target.raw()];
                set_targets(context, 1, targets.as_ptr(), null_mut());
                // The viewport bounds the triangle's pixels: the extent, from the first texel. A
                // texture's sides are at most 16384, which an f32 holds exactly.
                let viewport = Viewport {
                    top_left_x: 0.0,
                    top_left_y: 0.0,
                    width: width as f32,
                    height: height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                set_viewports(context, 1, &viewport);
                let views = [view.map_or(null_mut(), Com::raw)];
                set_views(context, 0, 1, views.as_ptr());
                set_pixel(context, shader.raw(), null(), 0);
                draw(context, 3, 0);
            }
            let none = [null_mut::<c_void>()];
            set_views(context, 0, 1, none.as_ptr());
            set_targets(context, 0, null(), null_mut());
        }
    }

    /// Flushes what the context recorded, and waits on this thread until the GPU did it,
    /// however long it takes; `Err` where a call failed, or the device was removed.
    fn flush_and_wait(&mut self) -> Result<(), String> {
        let value = self.flush_and_signal()?;
        wait_for(&self.fence, value)
    }

    /// Flushes what the context recorded, and signals the fence after it on the queue: the
    /// value it reaches once the GPU did it. `Err` where a call failed.
    fn flush_and_signal(&mut self) -> Result<u64, String> {
        self.value += 1;
        // SAFETY: each method is the named one of its object's interface: the context's flush
        // under the state's lock, and the queue's signal of the state's fence.
        unsafe {
            let flush: unsafe extern "system" fn(*mut c_void) = self.context.method(CONTEXT_FLUSH);
            flush(self.context.raw());
            let signal: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> Hresult =
                self.queue.method(QUEUE_SIGNAL);
            check(
                "ID3D12CommandQueue::Signal",
                signal(self.queue.raw(), self.fence.raw(), self.value),
            )?;
        }
        Ok(self.value)
    }

    /// Flushes the draws' device's immediate context and the native device's, so that the
    /// objects released since are destroyed (module doc).
    fn flush_destroyed(&self) {
        for context in [&self.context, &self.native_context] {
            // SAFETY: the method is ID3D11DeviceContext's `Flush`, of one of the state's
            // immediate contexts, under the state's lock.
            unsafe {
                let flush: unsafe extern "system" fn(*mut c_void) = context.method(CONTEXT_FLUSH);
                flush(context.raw());
            }
        }
    }

    /// Makes the pipelines of the draws (module doc): a draw of each pixel shader into a
    /// texture of its target's format, with no view bound, and the wait for them.
    fn make_pipelines(&mut self) -> Result<(), String> {
        let mut kept = Vec::new();
        for (depth, formats) in FORMATS.iter().enumerate() {
            for (plane, &format) in formats.targets.iter().enumerate() {
                let texture =
                    new_texture(&self.device, format, (2, 2), D3D11_BIND_RENDER_TARGET, 0)?;
                let mut out = null_mut();
                // SAFETY: the method is ID3D11Device's `CreateRenderTargetView`, of a texture of
                // the device's, in its own format.
                let result = unsafe {
                    let create: unsafe extern "system" fn(
                        *mut c_void,
                        *mut c_void,
                        *const c_void,
                        *mut *mut c_void,
                    ) -> Hresult = self.device.method(DEVICE_CREATE_RENDER_TARGET_VIEW);
                    create(self.device.raw(), texture.raw(), null(), &mut out)
                };
                let view = Com::made("CreateRenderTargetView", result, out)?;
                kept.push((texture, view, depth, plane));
            }
        }
        self.set_origin([0; 4]);
        let draws = kept
            .iter()
            .map(|(_, view, depth, plane)| (view, None, &self.pixel[*depth][*plane], (2, 2)));
        // SAFETY: the views and shaders are the device's, under the state's lock (`&mut self`).
        unsafe { self.encode_draws(draws) };
        self.flush_and_wait()
    }

    /// The bytes of `target`'s texels, of `texel` bytes each, read back by the native device once
    /// its last draw's wait has returned, row after row with nothing between them.
    fn read_back(&self, target: &Target, texel: u32) -> Result<Vec<u8>, String> {
        let (width, height) = target.size;
        let staging = {
            let desc = Texture2dDesc {
                width,
                height,
                mip_levels: 1,
                array_size: 1,
                format: target_format(&target.native),
                sample_count: 1,
                sample_quality: 0,
                usage: 3,
                bind_flags: 0,
                cpu_access_flags: 0x2_0000,
                misc_flags: 0,
            };
            texture_of(&self.native, &desc)?
        };
        // SAFETY: the method is ID3D11DeviceContext's `CopyResource`, between textures of the
        // native device of one format and size.
        unsafe {
            let copy: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) =
                self.native_context.method(CONTEXT_COPY_RESOURCE);
            copy(
                self.native_context.raw(),
                staging.raw(),
                target.native.raw(),
            );
        }
        #[repr(C)]
        struct Mapped {
            data: *const u8,
            row_pitch: u32,
            depth_pitch: u32,
        }
        let mut mapped = Mapped {
            data: null(),
            row_pitch: 0,
            depth_pitch: 0,
        };
        // SAFETY: the methods are ID3D11DeviceContext's `Map` and `Unmap`, of a staging texture
        // of the device's for reading, whose rows are `row_pitch` apart while it is mapped.
        unsafe {
            let map: unsafe extern "system" fn(
                *mut c_void,
                *mut c_void,
                u32,
                u32,
                u32,
                *mut Mapped,
            ) -> Hresult = self.native_context.method(CONTEXT_MAP);
            check(
                "ID3D11DeviceContext::Map",
                map(
                    self.native_context.raw(),
                    staging.raw(),
                    0,
                    1,
                    0,
                    &mut mapped,
                ),
            )?;
            let row = (width * texel) as usize;
            let mut bytes = Vec::with_capacity(row * height as usize);
            for y in 0..height as usize {
                let at = mapped.data.add(y * mapped.row_pitch as usize);
                bytes.extend_from_slice(std::slice::from_raw_parts(at, row));
            }
            let unmap: unsafe extern "system" fn(*mut c_void, *mut c_void, u32) =
                self.native_context.method(CONTEXT_UNMAP);
            unmap(self.native_context.raw(), staging.raw(), 0);
            Ok(bytes)
        }
    }
}

/// A direct queue on `device`, which the Direct3D 11 device records onto.
fn direct_queue(device: &Com) -> Result<Com, String> {
    let desc = CommandQueueDesc {
        type_: D3D12_COMMAND_LIST_TYPE_DIRECT,
        priority: 0,
        flags: 0,
        node_mask: 0,
    };
    let mut out = null_mut();
    // SAFETY: the method is ID3D12Device's `CreateCommandQueue`, with a valid description and
    // ID3D12CommandQueue's IID.
    let result = unsafe {
        let create: unsafe extern "system" fn(
            *mut c_void,
            *const CommandQueueDesc,
            *const Guid,
            *mut *mut c_void,
        ) -> Hresult = device.method(DEVICE_CREATE_COMMAND_QUEUE);
        create(device.raw(), &desc, &IID_ID3D12_COMMAND_QUEUE, &mut out)
    };
    Com::made("CreateCommandQueue", result, out)
}

/// A fence of `device`'s, at 0.
fn new_fence(device: &Com) -> Result<Com, String> {
    let mut out = null_mut();
    // SAFETY: the method is ID3D12Device's `CreateFence`, with no flags and ID3D12Fence's IID.
    let result = unsafe {
        let create: unsafe extern "system" fn(
            *mut c_void,
            u64,
            u32,
            *const Guid,
            *mut *mut c_void,
        ) -> Hresult = device.method(DEVICE_CREATE_FENCE);
        create(device.raw(), 0, 0, &IID_ID3D12_FENCE, &mut out)
    };
    Com::made("CreateFence", result, out)
}

/// The Direct3D 11 device over `device` that records onto `queue`, its immediate context, and
/// its `ID3D11On12Device`.
fn on12_device(device: &Com, queue: &Com) -> Result<(Com, Com, Com), String> {
    let create = system_function("d3d11.dll", c"D3D11On12CreateDevice")?;
    // SAFETY: the function is `D3D11On12CreateDevice` (d3d11on12.h).
    let create = unsafe { std::mem::transmute::<NonNull<c_void>, D3d11On12CreateDevice>(create) };
    let queues = [queue.raw()];
    let (mut made, mut context, mut level) = (null_mut(), null_mut(), 0);
    // SAFETY: the device and queue are Direct3D 12's; the device's feature levels are its
    // defaults; the outputs are written once each.
    let result = unsafe {
        create(
            device.raw(),
            0,
            null(),
            0,
            queues.as_ptr(),
            1,
            0,
            &mut made,
            &mut context,
            &mut level,
        )
    };
    let made = Com::made("D3D11On12CreateDevice", result, made)?;
    let context = Com::made("D3D11On12CreateDevice's context", result, context)?;
    let on12 = made.query(&IID_ID3D11_ON12_DEVICE, "ID3D11On12Device")?;
    Ok((made, context, on12))
}

/// The native Direct3D 11 device on the DXGI adapter of `luid`, or on the default adapter for
/// `None`, and its immediate context.
fn native_device(luid: Option<[u8; 8]>) -> Result<(Com, Com), String> {
    let create = system_function("d3d11.dll", c"D3D11CreateDevice")?;
    // SAFETY: the function is `D3D11CreateDevice` (d3d11.h).
    let create = unsafe { std::mem::transmute::<NonNull<c_void>, D3d11CreateDevice>(create) };
    let adapter = luid.map(adapter_of).transpose()?;
    let driver = match adapter {
        Some(_) => D3D_DRIVER_TYPE_UNKNOWN,
        None => D3D_DRIVER_TYPE_HARDWARE,
    };
    let (mut made, mut context, mut level) = (null_mut(), null_mut(), 0);
    // SAFETY: the adapter is null or DXGI's; no software module, no flags, the default feature
    // levels; the outputs are written once each.
    let result = unsafe {
        create(
            adapter.as_ref().map_or(null_mut(), Com::raw),
            driver,
            null_mut(),
            0,
            null(),
            0,
            D3D11_SDK_VERSION,
            &mut made,
            &mut level,
            &mut context,
        )
    };
    let made = Com::made("D3D11CreateDevice", result, made)?;
    let context = Com::made("D3D11CreateDevice's context", result, context)?;
    Ok((made, context))
}

/// Whether `device` answers `ExtendedResourceSharing` (`D3D11_FEATURE_DATA_D3D11_OPTIONS`'s
/// last of 14 `BOOL`s).
fn extended_resource_sharing(device: &Com) -> Result<bool, String> {
    let mut options = [0i32; 14];
    // SAFETY: the method is ID3D11Device's `CheckFeatureSupport`, of the options' 56 bytes.
    let result = unsafe {
        let check_support: unsafe extern "system" fn(
            *mut c_void,
            u32,
            *mut c_void,
            u32,
        ) -> Hresult = device.method(DEVICE_CHECK_FEATURE_SUPPORT);
        check_support(
            device.raw(),
            D3D11_FEATURE_D3D11_OPTIONS,
            options.as_mut_ptr().cast(),
            56,
        )
    };
    check("CheckFeatureSupport (D3D11_OPTIONS)", result)?;
    Ok(options[13] != 0)
}

/// `source`'s `main`, compiled for `target` by `D3DCompile`.
fn compile(source: &str, target: &CStr) -> Result<Vec<u8>, String> {
    let compile = system_function("d3dcompiler_47.dll", c"D3DCompile")?;
    // SAFETY: the function is `D3DCompile` (d3dcompiler.h).
    let compile = unsafe { std::mem::transmute::<NonNull<c_void>, D3dCompile>(compile) };
    let (mut code, mut errors) = (null_mut(), null_mut());
    // SAFETY: the source is `source.len()` bytes; no name, defines or includes; the entry point
    // and target are NUL-terminated; the outputs are written once each.
    let result = unsafe {
        compile(
            source.as_ptr().cast(),
            source.len(),
            null(),
            null(),
            null_mut(),
            c"main".as_ptr(),
            target.as_ptr(),
            0,
            0,
            &mut code,
            &mut errors,
        )
    };
    let errors = Com::made("D3DCompile's errors", 0, errors).ok();
    if let Err(why) = check("D3DCompile", result) {
        let said = errors.as_ref().map(blob_bytes).unwrap_or_default();
        return Err(format!(
            "{why} for {}: {}",
            target.to_string_lossy(),
            String::from_utf8_lossy(&said)
        ));
    }
    Ok(blob_bytes(&Com::made("D3DCompile", result, code)?))
}

/// A blob's bytes (`ID3DBlob`).
fn blob_bytes(blob: &Com) -> Vec<u8> {
    // SAFETY: the methods are ID3DBlob's `GetBufferPointer` and `GetBufferSize`, whose bytes
    // live as long as the blob.
    unsafe {
        let pointer: unsafe extern "system" fn(*mut c_void) -> *const u8 =
            blob.method(BLOB_GET_BUFFER_POINTER);
        let size: unsafe extern "system" fn(*mut c_void) -> usize =
            blob.method(BLOB_GET_BUFFER_SIZE);
        let (pointer, size) = (pointer(blob.raw()), size(blob.raw()));
        if pointer.is_null() {
            return Vec::new();
        }
        std::slice::from_raw_parts(pointer, size).to_vec()
    }
}

/// A shader of `device`'s from `code`, by its method `index` (`CreateVertexShader` or
/// `CreatePixelShader`, which take the same arguments).
fn shader(device: &Com, index: usize, code: &[u8]) -> Result<Com, String> {
    let mut out = null_mut();
    // SAFETY: the method is ID3D11Device's `CreateVertexShader` or `CreatePixelShader`, of
    // compiled code, with no class linkage.
    let result = unsafe {
        let create: unsafe extern "system" fn(
            *mut c_void,
            *const c_void,
            usize,
            *mut c_void,
            *mut *mut c_void,
        ) -> Hresult = device.method(index);
        create(
            device.raw(),
            code.as_ptr().cast(),
            code.len(),
            null_mut(),
            &mut out,
        )
    };
    Com::made("CreateVertexShader or CreatePixelShader", result, out)
}

/// A constant buffer of `device`'s of 16 bytes, in the GPU's memory.
fn constant_buffer(device: &Com) -> Result<Com, String> {
    let desc = BufferDesc {
        byte_width: 16,
        usage: 0,
        bind_flags: D3D11_BIND_CONSTANT_BUFFER,
        cpu_access_flags: 0,
        misc_flags: 0,
        structure_byte_stride: 0,
    };
    let mut out = null_mut();
    // SAFETY: the method is ID3D11Device's `CreateBuffer`, with a valid description and no
    // first bytes.
    let result = unsafe {
        let create: unsafe extern "system" fn(
            *mut c_void,
            *const BufferDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> Hresult = device.method(DEVICE_CREATE_BUFFER);
        create(device.raw(), &desc, null(), &mut out)
    };
    Com::made("CreateBuffer", result, out)
}

/// A texture of `device`'s of `format`, `size` big, in the GPU's memory, bound as `bind` with
/// `misc`.
fn new_texture(
    device: &Com,
    format: u32,
    size: (u32, u32),
    bind: u32,
    misc: u32,
) -> Result<Com, String> {
    let desc = Texture2dDesc {
        width: size.0,
        height: size.1,
        mip_levels: 1,
        array_size: 1,
        format,
        sample_count: 1,
        sample_quality: 0,
        usage: 0,
        bind_flags: bind,
        cpu_access_flags: 0,
        misc_flags: misc,
    };
    texture_of(device, &desc)
}

/// A texture of `device`'s as `desc` describes it, with no first texels.
fn texture_of(device: &Com, desc: &Texture2dDesc) -> Result<Com, String> {
    let mut out = null_mut();
    // SAFETY: the method is ID3D11Device's `CreateTexture2D`, with a valid description and no
    // first texels.
    let result = unsafe {
        let create: unsafe extern "system" fn(
            *mut c_void,
            *const Texture2dDesc,
            *const c_void,
            *mut *mut c_void,
        ) -> Hresult = device.method(DEVICE_CREATE_TEXTURE_2D);
        create(device.raw(), desc, null(), &mut out)
    };
    Com::made("CreateTexture2D", result, out)
}

/// The format of `texture`, an `ID3D11Texture2D` (`GetDesc`).
fn target_format(texture: &Com) -> u32 {
    let mut desc = Texture2dDesc {
        width: 0,
        height: 0,
        mip_levels: 0,
        array_size: 0,
        format: 0,
        sample_count: 0,
        sample_quality: 0,
        usage: 0,
        bind_flags: 0,
        cpu_access_flags: 0,
        misc_flags: 0,
    };
    // SAFETY: the method is ID3D11Texture2D's `GetDesc` (method 10), which writes the
    // description.
    unsafe {
        let get: unsafe extern "system" fn(*mut c_void, *mut Texture2dDesc) = texture.method(10);
        get(texture.raw(), &mut desc);
    }
    desc.format
}

/// An NT handle of `texture`, a shared texture, for reading and writing.
fn share(texture: &Com) -> Result<Handle, String> {
    let resource = texture.query(&IID_IDXGI_RESOURCE1, "IDXGIResource1")?;
    let mut handle = null_mut();
    // SAFETY: the method is IDXGIResource1's `CreateSharedHandle`, with no security attributes
    // and no name.
    let result = unsafe {
        let create: unsafe extern "system" fn(
            *mut c_void,
            *const c_void,
            u32,
            *const u16,
            *mut *mut c_void,
        ) -> Hresult = resource.method(RESOURCE1_CREATE_SHARED_HANDLE);
        create(
            resource.raw(),
            null(),
            DXGI_SHARED_RESOURCE_READ_WRITE,
            null(),
            &mut handle,
        )
    };
    check("IDXGIResource1::CreateSharedHandle", result)?;
    NonNull::new(handle)
        .map(Handle)
        .ok_or_else(|| "IDXGIResource1::CreateSharedHandle made no handle".to_owned())
}

/// Waits on this thread until `fence`, an `ID3D12Fence`, reaches `value`, however long it takes,
/// with an event of the wait's own, so that waits on several threads need no lock; `Err` where a
/// call failed, or the device was removed.
fn wait_for(fence: &Com, value: u64) -> Result<(), String> {
    if completed(fence) < value {
        // SAFETY: an auto-reset event, unnamed, with no security attributes; closed as the
        // handle drops.
        let event = NonNull::new(unsafe { CreateEventW(null(), 0, 0, null()) })
            .map(Handle)
            .ok_or_else(|| format!("CreateEventW: {}", std::io::Error::last_os_error()))?;
        // SAFETY: the method is ID3D12Fence's `SetEventOnCompletion`, with the wait's event,
        // which is open until the wait returns.
        unsafe {
            let on_completion: unsafe extern "system" fn(*mut c_void, u64, *mut c_void) -> Hresult =
                fence.method(FENCE_SET_EVENT_ON_COMPLETION);
            check(
                "ID3D12Fence::SetEventOnCompletion",
                on_completion(fence.raw(), value, event.0.as_ptr()),
            )?;
            if WaitForSingleObject(event.0.as_ptr(), INFINITE) != WAIT_OBJECT_0 {
                return Err(format!(
                    "the wait for the draw: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
    }
    match completed(fence) {
        // A removed device's fence answers `UINT64_MAX`, which says nothing of its writes.
        u64::MAX => Err("the device was removed".to_owned()),
        reached if reached >= value => Ok(()),
        reached => Err(format!(
            "the fence answers {reached} after the wait for {value}"
        )),
    }
}

/// `fence`'s value, an `ID3D12Fence`'s.
fn completed(fence: &Com) -> u64 {
    // SAFETY: the method is ID3D12Fence's `GetCompletedValue`.
    unsafe {
        let completed: unsafe extern "system" fn(*mut c_void) -> u64 =
            fence.method(FENCE_GET_COMPLETED_VALUE);
        completed(fence.raw())
    }
}

#[cfg(test)]
mod tests;
