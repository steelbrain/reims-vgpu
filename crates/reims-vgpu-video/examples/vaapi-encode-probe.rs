//! Lists what each render node's VA-API driver reports for its H.264 and
//! HEVC encoders: every encode attribute the encoders read, with HEVC's
//! features and block sizes decoded.
//!
//! ```sh
//! cargo run --example vaapi-encode-probe -- /dev/dri/renderD128 /dev/dri/renderD129
//! ```

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("vaapi-encode-probe needs Linux and libva");
}

#[cfg(target_os = "linux")]
mod linux {
    use reims_vgpu_video::core::runtime::StdRuntime;
    use std::process::ExitCode;

    use reims_vgpu_video::vaapi::{Display, EncoderCapabilities, Entrypoint, Profile};

    /// The VA profiles the guest's two encoders could use (va.h).
    const PROFILES: [(i32, &str); 5] = [
        (13, "H264ConstrainedBaseline"),
        (6, "H264Main"),
        (7, "H264High"),
        (17, "HEVCMain"),
        (18, "HEVCMain10"),
    ];
    const ENTRYPOINTS: [(i32, &str); 2] = [(6, "EncSlice"), (8, "EncSliceLP")];

    pub(crate) fn main() -> ExitCode {
        let nodes: Vec<String> = std::env::args().skip(1).collect();
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

    fn show(value: Option<u32>) -> String {
        value.map_or_else(|| "-".to_owned(), |v| format!("{v:#x}"))
    }

    fn probe(node: &str) -> Result<(), Box<dyn std::error::Error>> {
        let display = Display::open(node, StdRuntime::shared())?;
        println!("{node}: {}", display.vendor());
        for (profile, profile_name) in PROFILES {
            for (entrypoint, entrypoint_name) in ENTRYPOINTS {
                let Some(caps) =
                    EncoderCapabilities::query(&display, Profile(profile), Entrypoint(entrypoint))?
                else {
                    continue;
                };
                println!("  {profile_name} / {entrypoint_name}");
                println!(
                    "    rt {} rc {} packed {} refs {} slices {} structure {} quality {}",
                    show(caps.rt_formats),
                    show(caps.rate_control),
                    show(caps.packed_headers),
                    show(caps.max_ref_frames),
                    show(caps.max_slices),
                    show(caps.slice_structure),
                    show(caps.quality_range),
                );
                println!(
                    "    intra-refresh {} skip {} roi {} rc-ext {} max-frame-size {} \
                     prediction {} max {}x{}",
                    show(caps.intra_refresh),
                    show(caps.skip_frame),
                    show(caps.roi),
                    show(caps.rate_control_ext),
                    show(caps.max_frame_size),
                    show(caps.prediction_direction),
                    show(caps.max_width),
                    show(caps.max_height),
                );
                if let Some(features) = caps.hevc_feature_set() {
                    println!(
                        "    hevc features {}: {features:?}",
                        show(caps.hevc_features)
                    );
                }
                if let Some(sizes) = caps.hevc_block_size_set() {
                    println!(
                        "    hevc block sizes {}: {sizes:?}",
                        show(caps.hevc_block_sizes)
                    );
                }
            }
        }
        Ok(())
    }
}
