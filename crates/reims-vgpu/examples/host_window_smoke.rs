//! Standalone smoke test for the host-owned window ([[host-window]]) — opens the
//! `winit` + Vulkan swapchain window and presents an animated BGRA gradient,
//! printing every input action the window produces. Verifies the present path
//! (swapchain + staging upload + scale-blit) and input mapping on any host with
//! a display, without booting the VM.
//!
//! Run (needs a display + Vulkan ICD):
//!   cargo run -p reims-vgpu --example host_window_smoke \
//!       --no-default-features --features host-window
//!
//! Click / scroll / type in the window to see the mapped `Input*` HostActions on
//! stdout; close the window to exit.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use reims_vgpu::host_window::present::{spawn, Frame, FrameSlot, WindowConfig, WindowWaker};

fn main() {
    let (w, h) = (960u32, 600u32);
    let frames: FrameSlot = Arc::new(Mutex::new(Some(Arc::new(gradient(w, h, 0)))));
    // The window sleeps until something says a frame landed, so this stands in
    // for the device's publisher — without it the gradient would advance only at
    // the window's backstop rate rather than at the 62 Hz below.
    let wake = WindowWaker::new();

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
        },
        on_input,
        frames,
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
