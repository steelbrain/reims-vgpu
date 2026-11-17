//! The arm on Windows, on the runtime device's GPU: `vaon12` on its DXGI adapter decodes into the
//! VA crate's shared textures, and the Windows copier (`gpu_copy`) lands each picture from its
//! texture.

use std::sync::Arc;

use reims_vgpu_video::vaapi::{Display, H264Decoder, HevcDecoder};
use reims_vgpu_vulkan::VulkanPlatform;

use super::both_ways::land_both_ways;
use super::{Codec, GpuCopies as _, gpu_copy};

/// A picture copied by the GPU from the shared texture it decoded into is the CPU's copy of the
/// same picture, byte for byte: each plane's rows of the crop, and every byte before the planes,
/// past each row and past the planes as it was ([`land_both_ways`]), into host pages a buffer of
/// the runtime device lies over or a buffer of its own memory made as a device cache is. Every
/// picture of each H.264 and HEVC stream in the directories `RVG_VA_STREAMS` lists (as `PATH`
/// lists them), 8 and 10 bits, on the runtime device's adapter. A stream the decoder refuses is
/// decoded up to its refusal.
#[test]
#[ignore = "needs vaon12 on the runtime device's GPU and test streams: run on the Windows host"]
fn a_picture_copied_by_the_gpu_from_its_texture_is_the_cpus_copy_byte_for_byte() {
    use reims_vgpu_video::codec::{h264, hevc};

    let dirs =
        std::env::var_os("RVG_VA_STREAMS").expect("RVG_VA_STREAMS: the streams' directories");
    let mut streams: Vec<std::path::PathBuf> = std::env::split_paths(&dirs)
        .flat_map(|dir| std::fs::read_dir(dir).expect("a streams directory"))
        .map(|entry| entry.expect("a directory entry").path())
        .filter(|path| {
            let name = path.to_string_lossy();
            name.ends_with(".264") || name.ends_with(".265")
        })
        .collect();
    streams.sort();
    assert!(!streams.is_empty(), "no streams in {dirs:?}");
    let platform = Arc::new(VulkanPlatform::new());
    crate::entry::run(|_| {
        let device = VulkanPlatform::system_default_device().expect("a Vulkan device");
        let luid = reims_vgpu_vulkan::host_gpus::adapter_luid_of(&device.raw).expect("a LUID");
        let runtime = crate::video::runtime::PlatformRuntime::of(&platform);
        let display =
            Display::open_adapter(Some(luid), runtime).expect("a VA display on the device's GPU");
        let copies = gpu_copy::RuntimeCopies::new(&device);
        let mut copier = copies.copier().expect("a GPU copier on the device");
        let (mut pictures, mut refused) = (0usize, Vec::new());
        for stream in &streams {
            let file = std::fs::read(stream).expect("a stream");
            let (units, mut decoder) = if stream.to_string_lossy().ends_with(".265") {
                let decoder = HevcDecoder::new_annex_b(&display);
                (hevc::nal::access_units(&file), Codec::Hevc(decoder))
            } else {
                let decoder = H264Decoder::new_annex_b(&display);
                (h264::nal::access_units(&file), Codec::H264(decoder))
            };
            for (id, unit) in units.iter().enumerate() {
                let id = u64::try_from(id).expect("an id");
                let decoded = match &mut decoder {
                    Codec::H264(d) => d.decode(unit, id),
                    Codec::Hevc(d) => d.decode(unit, id),
                };
                match decoded {
                    Ok(frames) => {
                        for frame in &frames {
                            assert!(frame.texture().is_some(), "a decode not into a texture");
                            land_both_ways(&mut *copier, &device, frame, pictures, stream);
                            pictures += 1;
                        }
                    }
                    Err(error) => {
                        refused.push(format!("{}: {error}", stream.display()));
                        break;
                    }
                }
            }
            let flushed = match &mut decoder {
                Codec::H264(d) => d.flush(),
                Codec::Hevc(d) => d.flush(),
            };
            for frame in &flushed.unwrap_or_default() {
                land_both_ways(&mut *copier, &device, frame, pictures, stream);
                pictures += 1;
            }
        }
        eprintln!(
            "{pictures} pictures of {} streams copied by the GPU as by the CPU; refused: {refused:?}",
            streams.len()
        );
        assert!(refused.is_empty(), "refused: {refused:?}");
        assert!(pictures > 0);
    });
}
