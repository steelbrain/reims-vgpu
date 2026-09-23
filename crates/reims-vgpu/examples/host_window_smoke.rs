//! Standalone smoke test for the host-owned window ([[host-window]]) — opens the
//! window and presents an animated BGRA gradient, printing every input action
//! the window produces. Verifies a rail's whole CPU present path (staging
//! upload, aspect-fit blit, drawable present) and the input mapping, on any host
//! with a display, without booting the VM.
//!
//! Which rail presents is the same run-time answer it is in a boot, so this
//! smoke-tests whichever rail the build carries and `REIMS_VGPU_RAIL` selects:
//!
//! ```text
//! # Metal (Apple hosts)
//! cargo run -p reims-vgpu --example host_window_smoke \
//!     --features backend-metal,host-window
//! # Vulkan (needs an ICD; MoltenVK on Apple, a native one on Linux)
//! cargo run -p reims-vgpu --example host_window_smoke \
//!     --no-default-features --features backend-vulkan,host-window
//! ```
//!
//! It publishes CPU frames and no resident, which is every frame on the Metal
//! rail and the fallback path on the Vulkan one — so a gradient here says
//! nothing about the zero-copy path, which needs a guest.
//!
//! Click / scroll / type in the window to see the mapped `Input*` HostActions on
//! stdout; close the window to exit.
//!
//! **Linux only in practice.** [`spawn`] runs the loop on a dedicated thread,
//! and macOS requires AppKit work on the process main thread — a boot gets there
//! through `device_window_run_main`, which this example has no device to ask.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reims_vgpu::host_window::present::{
    spawn, CursorSlot, Frame, FrameSlot, FullscreenStrategy, GuestCursorState, WindowConfig,
    WindowMode, WindowWaker,
};

fn main() {
    let (w, h) = (960u32, 600u32);
    let frames: FrameSlot = Arc::new(Mutex::new(Some(Arc::new(gradient(w, h, 0)))));
    // The window sleeps until something says a frame landed, so this stands in
    // for the device's publisher — without it the gradient would advance only at
    // the window's backstop rate rather than at the 62 Hz below.
    let wake = WindowWaker::new();
    let cursor_slot: CursorSlot = Arc::new(Mutex::new(GuestCursorState::default()));

    // Animate the gradient on a helper thread so the window shows live updates.
    let anim = frames.clone();
    let anim_wake = Arc::clone(&wake);
    let _animator = std::thread::spawn(move || {
        let mut t = 0u32;
        loop {
            t = t.wrapping_add(2);
            if let Ok(mut slot) = anim.lock() {
                *slot = Some(Arc::new(gradient(w, h, t)));
            }
            anim_wake.wake();
            std::thread::sleep(Duration::from_millis(16));
        }
    });

    let on_input = Arc::new(|action| {
        println!("input: {action:?}");
    });

    let stop = Arc::new(AtomicBool::new(false));
    let handle = spawn(
        WindowConfig {
            title: "reims_vgpu host-window smoke".to_string(),
            width: w,
            height: h,
            mode: WindowMode::requested(),
            strategy: FullscreenStrategy::requested(WindowMode::requested()),
        },
        on_input,
        frames,
        cursor_slot,
        stop,
        wake,
    );
    match handle.join() {
        Ok(Ok(())) => println!("window closed"),
        Ok(Err(e)) => eprintln!("window error: {e}"),
        Err(_) => eprintln!("window thread panicked"),
    }
}

/// A moving BGRA8 gradient (tightly packed `w*h*4`), phase `t`.
fn gradient(w: u32, h: u32, t: u32) -> Frame {
    let mut bgra = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            bgra[i] = ((x + t) & 0xff) as u8; // B
            bgra[i + 1] = ((y + t) & 0xff) as u8; // G
            bgra[i + 2] = ((x ^ y).wrapping_add(t) & 0xff) as u8; // R
            bgra[i + 3] = 0xff; // A
        }
    }
    Frame {
        // Phase `t` is the frame identity: each animation step bumps it, so the
        // window sees a new seq and re-uploads (a static seq would freeze the
        // gradient under the seq-gated upload fast path).
        seq: t as u64,
        width: w,
        height: h,
        bgra,
        resident: None,
    }
}
