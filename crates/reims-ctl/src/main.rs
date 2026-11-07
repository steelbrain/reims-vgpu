//! reims-ctl: creates, boots and manages macOS VMs on QEMU's apple-gfx, on the QEMU
//! scripts/install-ctl.sh builds with reims-vgpu. With no command it opens its interactive
//! interface.

mod cli;
mod create;
mod form;
mod gpu;
mod host;
mod install;
mod logs;
mod macos;
mod opencore;
mod paths;
mod prebuilt;
mod qemu;
mod qmp;
mod reims_util;
mod runtime;
#[cfg(test)]
mod testenv;
mod tui;
mod update;
mod usb;
mod view;
mod vm;
mod vz;

use clap::Parser;

fn main() {
    quiet_on_closed_stdout();
    let cli = cli::Cli::parse();
    match cli::main(cli) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("reims-ctl: {e:#}");
            std::process::exit(1);
        }
    }
}

/// Output piped into a reader that stops early (`reims-ctl list | head -1`) ends the process
/// quietly with 141, the status of a death by SIGPIPE, rather than as println!'s panic. SIGPIPE
/// itself stays ignored, as Rust leaves it, so a write to a QMP socket whose QEMU just exited
/// is an error the caller handles rather than a signal that ends the interactive interface.
fn quiet_on_closed_stdout() {
    let default = std::panic::take_hook();
    let closed = format!("(os error {})", libc::EPIPE);
    std::panic::set_hook(Box::new(move |info| {
        let msg = info
            .payload()
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        if msg.starts_with("failed printing to stdout") && msg.contains(&closed) {
            std::process::exit(141);
        }
        default(info);
    }));
}
