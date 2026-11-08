//! Lists what the NVIDIA driver answers on each GPU: CUDA's devices, NVDEC's
//! decode capabilities for H.264 and HEVC at every chroma format and bit
//! depth, and
//! NVENC's codecs, profiles, presets, input formats and capabilities. The
//! NVIDIA host has no `nvidia-smi` query for these, and this stands in for
//! the SDK's samples.
//!
//! ```sh
//! cargo run --example nvidia-probe            # every GPU
//! cargo run --example nvidia-probe -- 0000:01:00.0
//! ```

#[cfg(target_os = "linux")]
mod linux {
    use std::process::ExitCode;
    use std::sync::Arc;

    use reims_vgpu_video::nvidia::{
        ChromaFormat, Gpu, GpuSelector, NvdecCodec, decode_capabilities, devices, encoder_report,
    };

    pub(super) fn main() -> ExitCode {
        let selectors: Vec<GpuSelector> = std::env::args()
            .skip(1)
            .map(|name| GpuSelector::parse(&name))
            .collect();
        let selectors = if selectors.is_empty() {
            match devices() {
                Ok(devices) => devices
                    .iter()
                    .map(|d| GpuSelector::Ordinal(d.ordinal))
                    .collect(),
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            }
        } else {
            selectors
        };
        let mut failed = false;
        for selector in &selectors {
            if let Err(e) = probe(selector) {
                eprintln!("{selector:?}: {e}");
                failed = true;
            }
        }
        if failed {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }

    fn probe(selector: &GpuSelector) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = Arc::new(Gpu::open(selector)?);
        let info = gpu.info();
        println!(
            "GPU {}: {} at {}, compute {}.{}, CUDA driver {}.{}",
            info.ordinal,
            info.name,
            info.pci_bus_id,
            info.compute_capability.0,
            info.compute_capability.1,
            gpu.driver_version() / 1000,
            gpu.driver_version() % 1000 / 10
        );
        println!("NVDEC (cuvidGetDecoderCaps):");
        for codec in NvdecCodec::ALL {
            for chroma in ChromaFormat::ALL {
                for bit_depth in [8, 10, 12] {
                    let caps = decode_capabilities(&gpu, codec, chroma, bit_depth)?;
                    if !caps.supported {
                        continue;
                    }
                    println!(
                        "  {codec:?} {chroma:?} {bit_depth}-bit: {}x{} to {}x{}, {} macroblocks, \
                     {} engines, out {:?}",
                        caps.min_width,
                        caps.min_height,
                        caps.max_width,
                        caps.max_height,
                        caps.max_macroblocks,
                        caps.engines,
                        caps.output_formats
                    );
                }
            }
        }
        let report = encoder_report(Arc::clone(&gpu))?;
        println!(
            "NVENC: API {}.{} at most (NvEncodeAPIGetMaxSupportedVersion)",
            report.max_api_version.0, report.max_api_version.1
        );
        for codec in &report.codecs {
            println!("  {}:", codec.codec);
            println!("    profiles: {}", codec.profiles.join(" "));
            println!("    presets: {}", codec.presets.join(" "));
            println!("    input formats: {}", codec.input_formats.join(" "));
            let capabilities: Vec<String> = codec
                .capabilities
                .iter()
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
            for line in capabilities.chunks(6) {
                println!("    {}", line.join(" "));
            }
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn main() -> std::process::ExitCode {
    linux::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("nvidia-probe needs Linux and NVIDIA's driver");
}
