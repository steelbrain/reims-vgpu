//! Lists what the VA-API driver of each render node reports: vendor and
//! version, each profile's entrypoints with their render-target formats,
//! maximum sizes, surface formats and memory types, video processing's
//! filters, and the image formats: `vainfo`'s answers, for a host without
//! it.
//!
//! ```sh
//! cargo run --example vaapi-probe -- /dev/dri/renderD128 /dev/dri/renderD129
//! ```
//!
//! With no arguments it probes every `/dev/dri/renderD*`.

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("vaapi-probe needs Linux and libva");
}

#[cfg(target_os = "linux")]
mod linux {
    use reims_vgpu_video_core::runtime::StdRuntime;
    use std::process::ExitCode;

    use reims_vgpu_video_vaapi::{Display, ImportWrite, proc_filter_name, rt_format_names};

    pub(crate) fn main() -> ExitCode {
        let mut nodes: Vec<String> = std::env::args().skip(1).collect();
        if nodes.is_empty() {
            let mut found: Vec<String> = std::fs::read_dir("/dev/dri")
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path().display().to_string())
                .filter(|p| p.contains("renderD"))
                .collect();
            found.sort();
            nodes = found;
        }
        let mut failed = false;
        for node in &nodes {
            if let Err(e) = probe(node) {
                eprintln!("{node}: {e}");
                failed = true;
            }
        }
        if failed {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }

    fn sysfs(node: &str, file: &str) -> String {
        let name = node.rsplit('/').next().unwrap_or(node);
        std::fs::read_to_string(format!("/sys/class/drm/{name}/device/{file}"))
            .map(|s| s.trim().to_owned())
            .unwrap_or_else(|_| "?".into())
    }

    fn opt(value: Option<u32>) -> String {
        value.map_or_else(|| "-".into(), |v| v.to_string())
    }

    fn probe(node: &str) -> Result<(), Box<dyn std::error::Error>> {
        let runtime = StdRuntime::shared();
        // Opened twice: the first open in the process, and one after it, each with the time
        // its import check took.
        let mut opens = Vec::new();
        let mut display = None;
        for _ in 0..2 {
            let start = runtime.now();
            let opened = Display::open(node, StdRuntime::shared())?;
            let took = runtime.now().saturating_duration_since(start);
            opens.push((took, opened.import_writes().took_each()));
            display = Some(opened);
        }
        let Some(display) = display else {
            return Ok(());
        };
        let caps = display.capabilities()?;
        println!("== {node}");
        println!(
            "pci {}:{} driver {}",
            sysfs(node, "vendor"),
            sysfs(node, "device"),
            sysfs(node, "uevent")
                .lines()
                .find_map(|l| l.strip_prefix("DRIVER="))
                .unwrap_or("?")
        );
        println!("vendor {}", caps.vendor);
        println!("va-api {}.{}", caps.api_version.0, caps.api_version.1);
        for c in &caps.configs {
            let profile = c
                .profile
                .name()
                .map_or_else(|| format!("profile {}", c.profile.0), str::to_owned);
            let entrypoint = c
                .entrypoint
                .name()
                .map_or_else(|| format!("entrypoint {}", c.entrypoint.0), str::to_owned);
            let rt = c
                .rt_formats
                .map_or_else(|| "-".into(), |f| rt_format_names(f).join(","));
            print!(
                "{profile:<24} {entrypoint:<11} rt={rt} max={}x{}",
                opt(c.max_width),
                opt(c.max_height)
            );
            if c.entrypoint.is_encode() {
                print!(
                    " rc={} packed={} refs={} slices={}",
                    c.rate_control
                        .map_or_else(|| "-".into(), |v| format!("{v:#x}")),
                    c.packed_headers
                        .map_or_else(|| "-".into(), |v| format!("{v:#x}")),
                    c.max_ref_frames
                        .map_or_else(|| "-".into(), |v| format!("{v:#x}")),
                    opt(c.max_slices)
                );
            } else if let Some(mode) = c.dec_slice_mode {
                print!(" slice-mode={mode:#x}");
            }
            match &c.surfaces {
                Ok(s) => {
                    let formats: Vec<String> =
                        s.pixel_formats.iter().map(ToString::to_string).collect();
                    print!(
                        " surfaces={} size={}x{}..{}x{} mem={}",
                        formats.join(","),
                        opt(s.min_width),
                        opt(s.min_height),
                        opt(s.max_width),
                        opt(s.max_height),
                        s.memory_types
                            .map_or_else(|| "-".into(), |v| format!("{v:#x}"))
                    );
                }
                Err(e) => print!(" surfaces=({e})"),
            }
            if let Some(filters) = &c.proc_filters {
                match filters {
                    Ok(list) => {
                        let names: Vec<String> = list
                            .iter()
                            .map(|&f| {
                                proc_filter_name(f).map_or_else(|| f.to_string(), str::to_owned)
                            })
                            .collect();
                        print!(
                            " filters={}",
                            if names.is_empty() {
                                "none".into()
                            } else {
                                names.join(",")
                            }
                        );
                    }
                    Err(e) => print!(" filters=({e})"),
                }
            }
            println!();
        }
        let images: Vec<String> = caps.image_formats.iter().map(ToString::to_string).collect();
        println!("images {}", images.join(","));
        let writes = display.import_writes();
        let said = |answer: Result<ImportWrite, String>| match answer {
            Ok(ImportWrite {
                alignment,
                whole_rows: false,
            }) => format!("at {alignment}-byte rows and planes"),
            Ok(ImportWrite {
                alignment,
                whole_rows: true,
            }) => format!("at {alignment}-byte rows and planes, rows of whole {alignment}s"),
            Err(why) => format!("none ({why})"),
        };
        println!(
            "import writes: NV12 {}; P010 {}",
            said(writes.nv12()),
            said(writes.p010())
        );
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        for (n, (open, [nv12, p010])) in opens.iter().enumerate() {
            println!(
                "open {}: {:.2} ms, the import check {:.2} ms of it (NV12 {:.2}, P010 {:.2})",
                n + 1,
                ms(*open),
                ms(*nv12 + *p010),
                ms(*nv12),
                ms(*p010)
            );
        }
        Ok(())
    }
}
