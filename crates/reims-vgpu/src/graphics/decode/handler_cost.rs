//! How long after a command buffer's GPU end the host's threads learn of it: through the
//! platform's completion-seen handler, which runs on the thread that saw the completion, and
//! through its completed handler, which on the Vulkan platform runs on a handler thread the
//! runtime wakes after that. The device caches answer a download as the first sees its copy done
//! and tell a fill's waiters there; this measures what that saves against the second.
//!
//! A run commits [`ROUNDS`] command buffers of one 4 KiB blit fill each, one at a time, with the
//! host's threads asleep for 2 ms before each, as between a client's frames, after [`WARM`] to
//! warm up. Each buffer carries both handlers, which read the host's clock. Printed: p50, p90
//! and p99 in microseconds of the GPU's start after the commit call began, of each handler's
//! time after the buffer's GPU end, which both platforms count on the host's clock
//! (`CLOCK_UPTIME_RAW` on macOS, `mach_absolute_time`'s; `CLOCK_MONOTONIC_RAW` on Linux), and of
//! the completed handler's after the completion-seen one's. On Metal the two run on Metal's one
//! handler thread, the completion-seen one first. A measurement, run by hand in a release build
//! holding the host's GPU (`cargo test --release -p reims-vgpu --lib [--features vulkan]
//! handler_cost -- --ignored --nocapture`).

use std::sync::mpsc;
use std::time::Duration;

use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _, CompletedInfo};
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::types::NsRange;
use reims_vgpu_core::raise::OnFault;

use crate::platform::ApiPlatform;

/// The buffers timed, and those before them.
const ROUNDS: usize = 2000;
const WARM: usize = 100;

/// The host's clock, in nanoseconds: the one both platforms give GPU times in.
fn host_ns() -> u64 {
    #[cfg(target_os = "macos")]
    let clock = libc::CLOCK_UPTIME_RAW;
    #[cfg(not(target_os = "macos"))]
    let clock = libc::CLOCK_MONOTONIC_RAW;
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `t` is the timespec the call writes.
    unsafe { libc::clock_gettime(clock, &mut t) };
    (t.tv_sec as u64) * 1_000_000_000 + t.tv_nsec as u64
}

/// Which handler told, when on the host's clock, and the buffer's GPU times it learned.
type Told = (bool, u64, Option<(u64, u64)>);

/// p50, p90 and p99 of `us`, sorted here.
fn spread(us: &mut [f64]) -> (f64, f64, f64) {
    us.sort_by(f64::total_cmp);
    let n = us.len();
    (us[n / 2], us[n * 9 / 10], us[n * 99 / 100])
}

/// One run on `P`'s system device, printed under `name`.
#[allow(clippy::cast_precision_loss)] // microseconds of a measurement
fn measure<P: ApiPlatform>(name: &str) {
    let o = OnFault::Die;
    let platform = P::process();
    let device = platform.metal_device(None).unwrap();
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer_of_length(o, 4096, 0).unwrap().unwrap();
    let (mut started, mut seen, mut completed, mut between) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for round in 0..WARM + ROUNDS {
        std::thread::sleep(Duration::from_millis(2));
        let cb = queue.command_buffer(o).unwrap().unwrap();
        let blit = cb.blit_encoder(o).unwrap().unwrap();
        blit.encode(
            o,
            BlitCmd::FillBuffer {
                buffer: &buffer,
                range: NsRange::new(0, 4096),
                value: (round % 251) as u8,
            },
        )
        .unwrap();
        blit.encode(o, BlitCmd::EndEncoding).unwrap();
        let (tx, rx) = mpsc::channel::<Told>();
        let tell = |first: bool, tx: mpsc::Sender<Told>| {
            Box::new(move |info: CompletedInfo| {
                let at = host_ns();
                let _ = tx.send((first, at, info.gpu_times_ns));
            })
        };
        cb.add_completion_seen_handler(o, tell(true, tx.clone()))
            .unwrap();
        cb.add_completed_handler(o, tell(false, tx)).unwrap();
        let committed = host_ns();
        cb.commit(o).unwrap();
        let mut at = [0u64; 2];
        let mut times = None;
        for _ in 0..2 {
            let (first, when, gpu_times) = rx
                .recv_timeout(Duration::from_secs(10))
                .expect("a command buffer never completed");
            at[usize::from(!first)] = when;
            times = times.or(gpu_times);
        }
        if round < WARM {
            continue;
        }
        let us = |from: u64, to: u64| (to as f64 - from as f64) / 1000.0;
        if let Some((start, end)) = times {
            started.push(us(committed, start));
            seen.push(us(end, at[0]));
            completed.push(us(end, at[1]));
        }
        between.push(us(at[0], at[1]));
    }
    println!("handler_cost on {name}, {ROUNDS} buffers of one 4 KiB fill, 2 ms apart:");
    for (what, us) in [
        ("GPU start after the commit", &mut started),
        ("completion-seen after the GPU end", &mut seen),
        ("completed after the GPU end", &mut completed),
        ("completed after completion-seen", &mut between),
    ] {
        if us.is_empty() {
            println!("  {what:<36} no GPU times");
            continue;
        }
        let (p50, p90, p99) = spread(us);
        println!("  {what:<36} p50 {p50:>8.1} p90 {p90:>8.1} p99 {p99:>8.1} us");
    }
}

/// On the Vulkan platform: the runtime's completion thread, then its handler thread.
#[test]
#[ignore = "a measurement, run by hand in a release build holding the GPU"]
fn on_vulkan_handler_cost() {
    measure::<reims_vgpu_vulkan::VulkanPlatform>("vulkan");
}
