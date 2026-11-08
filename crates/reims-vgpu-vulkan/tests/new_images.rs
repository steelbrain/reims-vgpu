//! Textures made while the GPU runs earlier work, on both platforms.
//!
//! Metal makes a texture with no GPU work: a texture over an IOSurface (as a guest display's
//! mapped present makes one) and a texture of memory of its own come back while a committed
//! kernel still runs. So does the runtime's: it records a new image's move to its first layout
//! in the device's next submission.
//!
//! The kernel ([`SPIN`]) runs [`ROUNDS`] rounds of busy work, about half a second, then marks its
//! end in a word the test reads as each creation returns: still unmarked where the creation did
//! not wait for it. Its rounds bound every wait here. A kernel that polls a word the CPU writes
//! cannot stand in for the guest's release: on the M3 Max, Metal's GPU saw no CPU write to such a
//! word while the kernel polled it (`fresh_res_spin`'s loop, released 100 µs into its run, ended
//! at its cap).

#![cfg(target_os = "macos")]
// Compares with Metal, which is macOS's.
// Every platform object is `Send` and `Sync` (core's `Handle`, and `Obj`'s bound, which an
// encoder's drop asks of `PredicatedEncoder`), and the proof walks the object's fields: an
// encoder's command buffer, its predication, and the runtime objects that holds, 141 to 150
// levels deep (measured: rustc warns at a limit of 140, not at 150). That is deeper than
// rustc's default of 128, as in the library (`src/lib.rs`).
#![recursion_limit = "256"]
#![allow(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    missing_docs,
    unsafe_code
)]

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::os::clock::{Clock, MonoTime};
use reims_vgpu_core::platform::os::surface::SurfaceSpec;
use reims_vgpu_core::platform::resource::{Buffer, Library};
use reims_vgpu_core::platform::types::{CommandBufferStatus, MtlSize, NsRange};
use reims_vgpu_core::platform::values::TextureDesc;
use reims_vgpu_core::raise::OnFault;
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_vulkan::VulkanPlatform;

/// The kernel that keeps the GPU busy (`src/gpu/spin.metal`).
const SPIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/spin.metallib"));

/// The kernel's rounds: each takes about 60 µs on the M3 Max (4096 steps of busy work), so about
/// half a second, where Metal makes each texture in about 100 µs.
const ROUNDS: u32 = 8_000;

/// How long the test waits for the kernel's command buffer to complete once the textures are made.
const END: Duration = Duration::from_secs(30);

/// `MTLPixelFormatBGRA8Unorm` and `MTLPixelFormatRGBA8Unorm`; `MTLTextureUsageShaderRead`;
/// shared and private storage.
const BGRA8: u64 = 80;
const RGBA8: u64 = 70;
const SHADER_READ: u64 = 1;
const SHARED: u64 = 0;
const PRIVATE: u64 = 0x20;

/// The kernel's words: its rounds, and its end mark, 0 until it writes [`ENDED`].
const ROUNDS_WORD: usize = 0;
const END_WORD: usize = 1;
const ENDED: u32 = 0x600d;

/// What each creation saw as it returned: the kernel's end mark, its command buffer's status,
/// and how long the creation took.
#[derive(Debug)]
struct Made {
    end_mark: u32,
    status: CommandBufferStatus,
    took: Duration,
}

/// What a run saw: [`Made`] for a texture over an IOSurface, then one of memory of its own.
#[derive(Debug)]
struct Seen {
    over_surface: Made,
    own_memory: Made,
}

/// The kernel committed, a texture made over an IOSurface and one of memory of its own, then the
/// kernel's command buffer waited for (at most [`END`]).
fn textures_while_a_kernel_runs<P: Platform>(p: &P, d: &P::Device) -> Seen {
    let library = d.new_library(SPIN).unwrap().unwrap().unwrap();
    let function = library.new_function(OnFault::Die, "spin").unwrap().unwrap();
    let state = d
        .new_compute_pipeline_state_for_function(Some(&function))
        .unwrap()
        .unwrap();
    let words = d
        .new_buffer_of_length(OnFault::Die, 16, SHARED)
        .unwrap()
        .unwrap();
    let word = |i: usize| {
        // SAFETY: one of the buffer's four words, which the kernel reads and writes as atomics.
        unsafe { AtomicU32::from_ptr(words.contents().as_ptr::<u32>().add(i)) }
    };
    word(ROUNDS_WORD).store(ROUNDS, Ordering::SeqCst);
    let queue = d.new_command_queue().unwrap();
    let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
    let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
    for cmd in [
        ComputeCmd::SetComputePipelineState {
            state: Some(&state),
        },
        ComputeCmd::SetBuffers {
            buffers: &[Some(&words)],
            offsets: &[0],
            range: NsRange::new(0, 1),
        },
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        },
        ComputeCmd::EndEncoding,
    ] {
        e.encode(OnFault::Die, cmd).unwrap();
    }
    cb.commit(OnFault::Die).unwrap();
    cb.wait_until_scheduled().unwrap();

    let (width, height) = (64, 16);
    let surface = p
        .surfaces()
        .create(&SurfaceSpec {
            pixel_format: u32::from_be_bytes(*b"BGRA"),
            alloc_size: 0x4000,
            width,
            height,
            bytes_per_element: 4,
            bytes_per_row: width * 4,
            element_width: 1,
            element_height: 1,
            ..SurfaceSpec::default()
        })
        .unwrap();
    let on_surface = p.texture_descriptor(&TextureDesc {
        pixel_format: BGRA8,
        width,
        height,
        usage: SHADER_READ,
        resource_options: Some(SHARED),
        ..TextureDesc::default()
    });
    let clock = p.clock();
    let since = |at: MonoTime| Duration::from_nanos(clock.now_mono().as_nanos() - at.as_nanos());
    let at = clock.now_mono();
    let over = d
        .new_texture_on_surface(&on_surface, Some(&surface), 0)
        .unwrap()
        .unwrap();
    let made = |at: MonoTime| Made {
        end_mark: word(END_WORD).load(Ordering::SeqCst),
        status: cb.status(),
        took: since(at),
    };
    let over_surface = made(at);
    let own = p.texture_descriptor(&TextureDesc {
        pixel_format: RGBA8,
        width: 64,
        height: 64,
        usage: SHADER_READ,
        resource_options: Some(PRIVATE),
        ..TextureDesc::default()
    });
    let at = clock.now_mono();
    let own = d.new_texture(OnFault::Die, &own).unwrap().unwrap();
    let own_memory = made(at);

    let waited = clock.now_mono();
    while ![CommandBufferStatus::COMPLETED, CommandBufferStatus::ERROR].contains(&cb.status()) {
        assert!(
            since(waited) < END,
            "the kernel's command buffer did not complete within {END:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        cb.status(),
        CommandBufferStatus::COMPLETED,
        "{:?}",
        cb.error()
    );
    assert_eq!(word(END_WORD).load(Ordering::SeqCst), ENDED);
    drop((over, own));
    Seen {
        over_surface,
        own_memory,
    }
}

/// Each creation came back while the kernel still ran: before it marked its end.
fn assert_made_while_running(seen: &Seen) {
    for (what, made) in [
        ("over an IOSurface", &seen.over_surface),
        ("of memory of its own", &seen.own_memory),
    ] {
        assert_eq!(
            made.end_mark, 0,
            "the texture {what} came back after the kernel ended: made in {:?}, the kernel's \
             command buffer {:?}",
            made.took, made.status
        );
    }
}

#[test]
fn metal_makes_textures_while_the_gpu_runs() {
    let p = MetalPlatform::new();
    let seen = textures_while_a_kernel_runs(&p, &MetalPlatform::system_default_device().unwrap());
    assert_made_while_running(&seen);
}

#[test]
fn this_platform_makes_textures_while_the_gpu_runs() {
    let p = VulkanPlatform::new();
    let seen = textures_while_a_kernel_runs(&p, &VulkanPlatform::system_default_device().unwrap());
    assert_made_while_running(&seen);
}
