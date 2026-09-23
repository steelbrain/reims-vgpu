//! Answers one design question and nothing else: **may the host-window present
//! run on a thread that is not the window's event-loop thread?**
//!
//! That question gates the last large piece of the replacement cutover. The
//! present is the only path that still needs `&mut ResourcePools` from the
//! window thread, and `pools` cannot follow `caches` onto the device until the
//! whole present is a transaction run by the thread that owns the device — the
//! drain. `window_present_frame`'s own docs record what that move would cost in
//! schedule and lock time (nothing, and about a millisecond a second), and then
//! say what was *not* established: the acquire has to happen before the blit can
//! be recorded, and it is the window's surface that owns the swapchain. Whether
//! the platform's WSI tolerates being driven from another thread was left open,
//! and designing around a guess is how a rail acquires an intermittent hang.
//!
//! # What this runs
//!
//! The window is opened by [`spawn`] exactly as a boot opens it, on its own
//! event-loop thread, with a frame slot that is **never published to**. The
//! loop's `needs_present` gate therefore lets it present once — the first-frame
//! forced redraw — and then goes quiet for the rest of the run. Every present
//! after that comes from this file's own thread, through the same
//! `Backend::window_present` entry the window thread calls.
//!
//! So a clean run is a rail that acquired a drawable, recorded and submitted a
//! blit, and called `vkQueuePresentKHR` several hundred times from a thread that
//! is not the one that created the surface — which is the shape the move needs.
//!
//! It names the Vulkan rail on purpose. The question is about one rail's WSI,
//! and `AGENTS.md`'s rule for a test whose answer is a rail's is to name that
//! rail rather than ask a `cfg` which arm was compiled.
//!
//! ```text
//! REIMS_VGPU_PROBE_SECONDS=6 cargo run -p reims-vgpu \
//!     --example window_present_off_thread \
//!     --no-default-features --features backend-vulkan,host-window
//! ```
//!
//! # What a clean run does *not* establish
//!
//! One host, one WSI platform, one driver. The pathway table carries a Linux
//! Vulkan rail and two Apple ones, and MoltenVK's drawable acquisition is a
//! different implementation with different thread rules — a pass here is
//! evidence about the platform it ran on and about nothing else. Run it on the
//! other host before designing for the other host.
//!
//! # Swapchain recreation, which is the other half of the question
//!
//! Presenting is the frequent WSI call; recreating is the dangerous one. It
//! destroys and rebuilds a `VkSwapchainKHR` against a surface the compositor and
//! the event loop both still own, and a present-only probe would have missed it
//! entirely. So every `REIMS_VGPU_PROBE_RECREATE_EVERY` frames (default 30) the
//! probe thread toggles the presenter's desired extent, which arms
//! `recreate_pending`, and the *next off-thread present* is the one that
//! destroys and rebuilds the swapchain. The reported `recreates` count is how
//! many times that was asked for.
//!
//! Whether the requested extent is honoured is not the point and is not asserted
//! — Wayland reports a `currentExtent` the driver clamps to, and the native
//! window is never actually resized here. The point is that the destroy/create
//! pair ran on the wrong thread and the rail kept presenting afterwards.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reims_vgpu::backend::vulkan::engine::{
    window_present_attached, window_present_frame, window_present_resize,
};
use reims_vgpu::backend::window::{WindowCpuFrame, WindowPresentOutcome};
use reims_vgpu::host_window::present::{
    spawn, CursorSlot, FrameSlot, FullscreenStrategy, GuestCursorState, WindowConfig, WindowMode,
    WindowWaker,
};

const W: u32 = 640;
const H: u32 = 400;

fn main() {
    let seconds: u64 = std::env::var("REIMS_VGPU_PROBE_SECONDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(6);

    // Deliberately empty and never published to. See the module docs: this is
    // what keeps the window thread from presenting after its first frame, so a
    // present that lands is one this file's thread made.
    let frames: FrameSlot = Arc::new(Mutex::new(None));
    let wake = WindowWaker::new();
    let cursor_slot: CursorSlot = Arc::new(Mutex::new(GuestCursorState::default()));
    let stop = Arc::new(AtomicBool::new(false));

    let recreate_every: u64 = std::env::var("REIMS_VGPU_PROBE_RECREATE_EVERY")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(30);

    let presented = Arc::new(AtomicU64::new(0));
    let busy = Arc::new(AtomicU64::new(0));
    let failed = Arc::new(AtomicU64::new(0));
    let recreates = Arc::new(AtomicU64::new(0));

    let probe_stop = Arc::clone(&stop);
    let probe_presented = Arc::clone(&presented);
    let probe_busy = Arc::clone(&busy);
    let probe_failed = Arc::clone(&failed);
    let probe_recreates = Arc::clone(&recreates);
    let probe = std::thread::spawn(move || {
        // The window thread has to have built its loop, created the surface and
        // attached the presenter before there is anything to present through.
        let attached_by = Instant::now() + Duration::from_secs(20);
        while !window_present_attached() {
            if Instant::now() > attached_by {
                eprintln!("probe: no presenter attached after 20 s; nothing was measured");
                probe_stop.store(true, Ordering::Release);
                return;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        eprintln!(
            "probe: presenting from thread {:?}, which is not the event loop's",
            std::thread::current().id()
        );

        let until = Instant::now() + Duration::from_secs(seconds);
        let mut seq = 0u64;
        let mut first_error: Option<String> = None;
        while Instant::now() < until && !probe_stop.load(Ordering::Acquire) {
            seq += 1;
            // Arm a swapchain rebuild for the present two lines below. The
            // toggle only has to make the extent *differ* from the last one; see
            // the module docs on why the value itself is not the claim.
            if recreate_every != 0 && seq % recreate_every == 0 {
                let height = if (seq / recreate_every) % 2 == 0 {
                    H
                } else {
                    H - 1
                };
                window_present_resize(W, height);
                probe_recreates.fetch_add(1, Ordering::Relaxed);
            }
            let bgra = gradient(W, H, (seq as u32).wrapping_mul(3));
            let frame = WindowCpuFrame {
                bgra: &bgra,
                width: W,
                height: H,
                seq,
            };
            match window_present_frame(None, Some(frame)) {
                Ok(WindowPresentOutcome::Presented { .. }) => {
                    probe_presented.fetch_add(1, Ordering::Relaxed);
                }
                Ok(WindowPresentOutcome::Busy) => {
                    probe_busy.fetch_add(1, Ordering::Relaxed);
                }
                Err(error) => {
                    probe_failed.fetch_add(1, Ordering::Relaxed);
                    first_error.get_or_insert_with(|| error.to_string());
                }
            }
            std::thread::sleep(Duration::from_millis(16));
        }
        if let Some(error) = first_error {
            eprintln!("probe: first off-thread present failure: {error}");
        }
        probe_stop.store(true, Ordering::Release);
    });

    let handle = spawn(
        WindowConfig {
            title: "reims_vgpu off-thread present probe".to_string(),
            width: W,
            height: H,
            mode: WindowMode::requested(),
            strategy: FullscreenStrategy::requested(WindowMode::requested()),
        },
        Arc::new(|_action| {}),
        frames,
        cursor_slot,
        stop,
        wake,
    );

    let _ = probe.join();
    match handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => eprintln!("probe: window error: {e}"),
        Err(_) => eprintln!("probe: window thread panicked"),
    }

    let presented = presented.load(Ordering::Relaxed);
    let busy = busy.load(Ordering::Relaxed);
    let failed = failed.load(Ordering::Relaxed);
    let recreates = recreates.load(Ordering::Relaxed);
    println!(
        "off_thread_present presented={presented} busy={busy} failed={failed} \
         recreates={recreates}"
    );
    if failed > 0 || presented == 0 || (recreate_every != 0 && recreates == 0) {
        println!("off_thread_present verdict=refused");
        std::process::exit(1);
    }
    println!("off_thread_present verdict=accepted");
}

/// A moving BGRA8 gradient (tightly packed `w*h*4`), phase `t`. Different pixels
/// every frame, so a swapchain that stopped updating is visible rather than
/// merely uncounted.
fn gradient(w: u32, h: u32, t: u32) -> Vec<u8> {
    let mut bgra = vec![0u8; (w as usize) * (h as usize) * 4];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            bgra[i] = ((x + t) % 256) as u8;
            bgra[i + 1] = ((y + t) % 256) as u8;
            bgra[i + 2] = (((x + y) / 2 + t) % 256) as u8;
            bgra[i + 3] = 0xff;
        }
    }
    bgra
}
