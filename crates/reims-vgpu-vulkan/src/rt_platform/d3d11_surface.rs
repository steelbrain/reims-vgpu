//! The rt platform's surfaces over Direct3D 11 textures (`GpuDevice::import_d3d11_surface`), on
//! Windows. Each plane of a picture is rt's texture over a shared texture that the host's video
//! draws the picture into (reims-vgpu-video's `vaapi::PlaneDraw`). The IOSurface device caches
//! keep a two-plane surface's planes in such textures, so a picture lands in the images that the
//! guest's textures are views of (reims-vgpu's `surface_cache::exported`).
//!
//! **The call** is rt's `Device::import_d3d11_surface`, one `rt::D3d11Plane` a plane. Where rt
//! refuses it (a device that imports no Direct3D 11 texture, a plane whose formats or usage its
//! import does not take), the surface is refused and counted, and the caller keeps the planes in
//! images of its own, where the host's video lands its pictures by the CPU's copy.
//!
//! **The planes.** Each plane is a 2D texture of the plane's format and size, at no resource
//! index, over the Direct3D texture its handle names. rt imports each one as a dedicated
//! `D3D11_TEXTURE` image of the texture's element format (`image_format`). A chroma plane's
//! element is one packed pair (`R16Uint`, `R32Uint`), and its texture is viewed in the plane's
//! two-channel format.
//!
//! **The acquire.** `ImportedSurfaceMemory::external_writes_done` is rt's
//! `Surface::external_writes_done`, which records nothing for a Direct3D 11 surface: each
//! submission whose command buffers or uploads use a plane acquires it from
//! `VK_QUEUE_FAMILY_EXTERNAL` at its start and releases it to that family at its end, so Direct3D
//! 11 owns the planes' memory between rt's submissions (`super::device`, "Its Direct3D 11 planes'
//! hand-overs"). rt takes no keyed mutex, and the writer's textures have none. The writer waits
//! for its write to be done on the GPU, and only then calls it.
//!
//! **Refusals.** Where rt makes no surface, the call answers `None`, counted
//! ([`D3D11_SURFACE_REFUSED_CODE`]). The process's first refusal also writes a line naming rt's
//! error and the planes.
//!
//! **Shared Usage counts every import.** Each import adds its textures' size to the process's
//! `GPU Process Memory\Shared Usage` counter, which Task Manager shows as shared GPU memory, and
//! nothing takes it off: not the import's free, and not the textures' release. No memory is
//! behind that growth, so nothing here keeps imports alive to hold the counter down.
//! **Measured** (2026-11-20, Intel UHD 630, Windows 11). In a test of three rounds of 32 pairs
//! of 4K targets (64 textures, 385 MiB a round), the round that imported none ended with
//! Shared Usage back at its start (149 MiB). In the two that imported theirs, Shared Usage less
//! the counter `Total Committed` grew by each round's 64 imports and never fell:
//! - 384 MiB after the first;
//! - 768 MiB after the second;
//! - still 768 MiB once the first round's imports were freed.
//!
//! Over the run, `\Memory\Available MBytes` fell only 165 MiB and `\Memory\Committed Bytes`
//! rose only 123 MiB. In a macOS guest's boot of 3940 imports (fps passes, then 8 rounds of a
//! held and a released 4K H.264 check):
//! - Shared Usage rose from 1.7 to 52.6 GiB, passing the process's DXGI budget of 48230 MiB in
//!   the sixth round.
//! - The process's DXGI usage (`QueryVideoMemoryInfo`, local; the non-local group answers 0 and
//!   a budget of 0 on this GPU), read every 40th pair of targets made, rose to 9.6-19.2 GiB in
//!   each held pass, and its lowest read in each pass after the first was 1.6-3.0 GiB.
//! - The budget read 48230 MiB in all 130 reads.
//! - Available MBytes went from 46443 to 48237, Committed Bytes from 19.5 to 17.9 GiB, and the
//!   process's handles from 1117 to 971.
//! - No import was refused, and every check matched all 240 of its pictures.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;
use metal2vulkan_runtime as rt;

use crate::core::platform::resource::{D3d11PlaneDesc, ImportedSurface, ImportedSurfaceMemory};
use crate::core::platform::values::TextureDesc;

use super::descriptor::rt_texture_descriptor;
use super::device::RtDevice;
use super::{RtObj, RtPlatform, texture};

/// The log target of the surfaces' lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The code of a surface over Direct3D 11 textures that rt made, counted per make.
pub const D3D11_SURFACES_CODE: &str = "rt-d3d11-surfaces";

/// The code of a surface over Direct3D 11 textures that rt refused (the module doc's
/// "Refusals"), counted per refusal.
pub const D3D11_SURFACE_REFUSED_CODE: &str = "rt-refused-d3d11-surface";

/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;

/// `MTLResourceStorageModePrivate`, in `MTLResourceOptions`: the planes' storage.
const STORAGE_PRIVATE: u64 = 2 << 4;

/// Whether the process has refused a surface yet.
static REFUSED_SAID: AtomicBool = AtomicBool::new(false);

/// What a surface over Direct3D 11 textures keeps: rt's surface, which holds the planes'
/// images.
struct RtImportedMemory {
    /// rt's surface.
    surface: rt::Surface,
}

impl ImportedSurfaceMemory for RtImportedMemory {
    fn external_writes_done(&self) -> Result<(), String> {
        (self.surface.external_writes_done()).map_err(|error| error.to_string())
    }
}

/// The texture of `plane`: 2D and Private, of the plane's format, size and usage.
fn plane_texture(plane: &D3d11PlaneDesc<'_>) -> TextureDesc {
    TextureDesc {
        texture_type: TEXTURE_TYPE_2D,
        pixel_format: plane.pixel_format,
        width: plane.width,
        height: plane.height,
        usage: plane.usage,
        resource_options: Some(STORAGE_PRIVATE),
        ..TextureDesc::default()
    }
}

/// rt's pixel format of the `MTLPixelFormat` `format`, as `what`.
fn rt_format(what: &str, format: u64) -> Result<rt::PixelFormat, String> {
    let word = u32::try_from(format).map_err(|_| format!("{what} {format}"))?;
    rt::PixelFormat::try_from(word).map_err(|_| format!("{what} {format}"))
}

/// rt's plane of `plane`, whose texture is `texture`; `Err` says why rt takes none.
fn plane_spec<'a>(
    plane: &D3d11PlaneDesc<'a>,
    texture: &TextureDesc,
) -> Result<rt::D3d11Plane<'a>, String> {
    Ok(rt::D3d11Plane {
        handle: plane.handle,
        image_format: rt_format("element format", plane.image_format)?,
        descriptor: rt_texture_descriptor(texture)?,
        index: None,
    })
}

/// Counts rt's refusal `why` of a surface of `planes` ([`D3D11_SURFACE_REFUSED_CODE`]). The
/// process's first refusal also writes its line.
fn refused(planes: &[D3d11PlaneDesc<'_>], why: &dyn std::fmt::Display) {
    crate::answer::count_kind(D3D11_SURFACE_REFUSED_CODE);
    if !REFUSED_SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: TARGET,
            kind:% = D3D11_SURFACE_REFUSED_CODE;
            "a surface over Direct3D 11 textures of {planes:?}: metal2vulkan-runtime makes none \
             ({why}); its planes stay in images of their own, and the host's video lands its \
             pictures another way ({D3D11_SURFACE_REFUSED_CODE}), counted from here"
        );
    }
}

/// The surface over the Direct3D 11 textures `planes` name, on `device` (the module doc).
/// `None` where rt makes none, counted, or where a plane's texture is not one rt takes.
pub(super) fn imported_surface(
    device: &RtObj<RtDevice>,
    planes: &[D3d11PlaneDesc<'_>],
) -> Option<ImportedSurface<RtPlatform>> {
    let sel = "importD3D11Surface (a plane)";
    let mut textures = Vec::with_capacity(planes.len());
    let mut specs = Vec::with_capacity(planes.len());
    for plane in planes {
        let texture = plane_texture(plane);
        match plane_spec(plane, &texture) {
            Ok(spec) => specs.push(spec),
            Err(why) => {
                refused(planes, &why);
                return None;
            }
        }
        textures.push(texture);
    }
    let surface = match device.rt().import_d3d11_surface(&specs) {
        Ok(surface) => surface,
        Err(error) => {
            refused(planes, &error);
            return None;
        }
    };
    crate::answer::count_kind(D3D11_SURFACES_CODE);
    let made = (surface.planes().iter().zip(&textures))
        .map(|(plane, texture)| texture::surface_plane(device, sel, texture, plane.clone()))
        .collect::<Option<Vec<_>>>()?;
    Some(ImportedSurface {
        planes: made,
        memory: Arc::new(RtImportedMemory { surface }),
    })
}

/// The import's usages that rt's `d3d11_plane` wants (metal2vulkan-runtime at fd0771d944,
/// `d3d11.rs`'s `WANTED`), each kept where the plane's view format holds it and the import takes
/// it ([`import_answers`]).
const WANTED: [vk::ImageUsageFlags; 5] = [
    vk::ImageUsageFlags::SAMPLED,
    vk::ImageUsageFlags::COLOR_ATTACHMENT,
    vk::ImageUsageFlags::TRANSFER_SRC,
    vk::ImageUsageFlags::TRANSFER_DST,
    vk::ImageUsageFlags::STORAGE,
];

/// What the driver answers rt's import of each of `planes` on `device`, for the line of the
/// draws' check at a video's open (reims-vgpu's `graphics::drawn_check`), which is a diagnostic:
/// nothing decides by it. Each plane's image format and view format; the usage rt's
/// `d3d11_plane` (metal2vulkan-runtime at fd0771d944) takes, by the same queries: each usage it
/// wants that the view format's optimal features hold and the import takes with the pair of the
/// two formats (none where they are one); and the formats the image lists. rt lists every format
/// Metal views the plane's as, where the import takes them with that usage, else the pair. rt
/// keeps its own list private, so that list is rebuilt here from metal2vulkan's
/// `views_with_usage` and `direct_format`, said as such.
pub fn import_answers(device: &RtObj<RtDevice>, planes: &[D3d11PlaneDesc<'_>]) -> String {
    let rt_device = device.rt();
    let raw = rt_device.raw();
    let answers: Vec<String> = (planes.iter())
        .map(|plane| plane_answers(&raw, plane).unwrap_or_else(|why| why))
        .collect();
    answers.join(", ")
}

/// The Vulkan format and its compatibility class that metal2vulkan maps `pf` to directly.
fn direct(pf: rt::PixelFormat) -> Option<(vk::Format, Option<&'static str>)> {
    let format = metal2vulkan::metal::format::direct_format(pf)?;
    Some((
        vk::Format::from_raw(format.as_raw()),
        format.block().map(|block| block.class),
    ))
}

/// [`import_answers`]' answers for `plane`; `Err` names a format it does not map.
fn plane_answers(raw: &rt::RawDevice<'_>, plane: &D3d11PlaneDesc<'_>) -> Result<String, String> {
    use metal2vulkan::metal::format::views_with_usage;
    let image_pf = rt_format("element format", plane.image_format)?;
    let view_pf = rt_format("pixel format", plane.pixel_format)?;
    let unmapped = |pf: rt::PixelFormat| format!("{pf:?} maps to no Vulkan format");
    let (format, _) = direct(image_pf).ok_or_else(|| unmapped(image_pf))?;
    let (view, class) = direct(view_pf).ok_or_else(|| unmapped(view_pf))?;
    let pair = match format == view {
        true => Vec::new(),
        false => vec![format, view],
    };
    // SAFETY: the physical device is the instance's.
    let held =
        unsafe { (raw.instance).get_physical_device_format_properties(raw.physical_device, view) }
            .optimal_tiling_features;
    let holds = |bit: vk::ImageUsageFlags| {
        let feature = match bit {
            vk::ImageUsageFlags::SAMPLED => vk::FormatFeatureFlags::SAMPLED_IMAGE,
            vk::ImageUsageFlags::COLOR_ATTACHMENT => vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            vk::ImageUsageFlags::TRANSFER_SRC => vk::FormatFeatureFlags::TRANSFER_SRC,
            vk::ImageUsageFlags::TRANSFER_DST => vk::FormatFeatureFlags::TRANSFER_DST,
            _ => vk::FormatFeatureFlags::STORAGE_IMAGE,
        };
        held.contains(feature)
    };
    let takes = |usage, list: &[vk::Format]| imports(raw, format, usage, list);
    let usage = (WANTED.iter())
        .filter(|&&bit| holds(bit) && takes(bit, &pair))
        .fold(vk::ImageUsageFlags::empty(), |all, &bit| all | bit);
    let whole = !usage.is_empty() && takes(usage, &pair);
    let mut listed = vec![view];
    for &other in rt::PixelFormat::ALL {
        if other == view_pf || !views_with_usage(view_pf, other) {
            continue;
        }
        if let Some((other, other_class)) = direct(other)
            && other_class == class
            && !listed.contains(&other)
        {
            listed.push(other);
        }
    }
    let mut full = vec![format];
    for listed in listed {
        if !full.contains(&listed) {
            full.push(listed);
        }
    }
    if full.len() == 1 {
        full.clear();
    }
    let takes_full = full.len() > pair.len() && takes(usage, &full);
    let list = match takes_full {
        true => format!("Metal's views (rebuilt) {}", names(&full)),
        false => format!(
            "the pair {}, refused Metal's views (rebuilt) {}",
            names(&pair),
            names(&full)
        ),
    };
    let whole = if whole { "" } else { " (refused together)" };
    let usage = metal2vulkan::vk::ImageUsageFlags::from_raw(usage.as_raw());
    Ok(format!(
        "{} as {}: usage {usage:?}{whole}, listing {list}",
        names(&[format]),
        names(&[view])
    ))
}

/// `formats` by their Vulkan names, in brackets where there are several or none.
fn names(formats: &[vk::Format]) -> String {
    let named: Vec<String> = (formats.iter())
        .map(|format| format!("{:?}", metal2vulkan::vk::Format::from_raw(format.as_raw())))
        .collect();
    match named.as_slice() {
        [one] => one.clone(),
        _ => format!("[{}]", named.join(", ")),
    }
}

/// Whether the driver imports a Direct3D 11 texture as an optimal 2D image of `format` and
/// `usage`, mutable and listing `list` where it is not empty, as rt's `import_query` asks it.
fn imports(
    raw: &rt::RawDevice<'_>,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    list: &[vk::Format],
) -> bool {
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE);
    let mut listed = vk::ImageFormatListCreateInfo::default().view_formats(list);
    let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .push_next(&mut external);
    if !list.is_empty() {
        info = info
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
            .push_next(&mut listed);
    }
    let mut external_out = vk::ExternalImageFormatProperties::default();
    let mut out = vk::ImageFormatProperties2::default().push_next(&mut external_out);
    // SAFETY: the physical device is the instance's, which rt made with the handle's extension.
    let answered = unsafe {
        (raw.instance).get_physical_device_image_format_properties2(
            raw.physical_device,
            &info,
            &mut out,
        )
    };
    answered.is_ok()
        && (external_out
            .external_memory_properties
            .external_memory_features)
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
}

#[cfg(test)]
mod tests {
    use std::os::windows::io::BorrowedHandle;

    use super::*;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::raise::OnFault;

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A surface rt refuses is counted, for the host's video to land its pictures another way
    /// (the module doc): here a luma plane of 2-byte elements viewed as `R8Unorm`, whose texel
    /// is 1 byte, which rt refuses before it reads the plane's handle.
    #[test]
    fn a_surface_rt_refuses_is_counted_for_another_landing_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        // SAFETY: -1 is the current process's pseudo-handle, which is never closed; rt refuses
        // the luma plane's formats, or a device that imports no Direct3D 11 texture, before it
        // reads a handle.
        let handle =
            unsafe { BorrowedHandle::borrow_raw(std::ptr::without_provenance_mut(usize::MAX)) };
        let plane = |image_format, pixel_format, width, height| D3d11PlaneDesc {
            handle,
            image_format,
            pixel_format,
            width,
            height,
            usage: 0x1 | 0x4 | 0x10,
        };
        let planes = [plane(20, 10, 64, 64), plane(23, 30, 32, 32)];
        let before = counted(D3D11_SURFACE_REFUSED_CODE);
        let made = device.import_d3d11_surface(OnFault::Return, &planes);
        assert!(matches!(made, Ok(None)));
        assert_eq!(counted(D3D11_SURFACE_REFUSED_CODE), before + 1);
    }
}
