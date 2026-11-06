//! The library's own shaders on this platform: the metal crate's `shaders.metal` (the display's
//! present functions and the XR10 paging kernels), compiled by Apple's tools when this crate is
//! built (`build.rs`) into [`METALLIB`], which [`library`] loads with the runtime's
//! `new_library_with_data`. Core asks for it as `newLibraryWithSource:options:error:`.
//!
//! As on the Metal platform, a library belongs to the device that made it, so the metallib is
//! loaded once per device, the first time core asks for it (a device's blit pipelines, a
//! display's pipeline cache, a service task's blit pipelines), and every later ask answers the
//! same library. Devices are told apart by the registry ID the platform answers for them
//! (`device::registry_id`). A loaded library stays for the life of the process; one that fails to
//! load is not kept, and the next ask loads again.

use metal2vulkan_runtime::Error;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::types::NsErrorText;
use reims_vgpu_core::sync::Mutex;

use crate::answer::{error_text, plain, refused};
use crate::gpu::device::VulkanDevice;
use crate::gpu::resource::{LibraryObject, VulkanLibrary};
use crate::platform::Obj;

/// The metallib `build.rs` compiled `shaders.metal` into.
pub const METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));

/// The selector core names the shader library's creation by.
const SELECTOR: &str = "newLibraryWithSource:options:error:";

/// The libraries loaded so far, each with its device's registry ID.
static LOADED: Mutex<Vec<(u64, VulkanLibrary)>> = Mutex::new(Vec::new());

/// The shader library of `device` (a new reference), loaded on the first call for its registry
/// ID. `Err` is the runtime's refusal of the metallib's contents (`Compile`), which stands where
/// Metal's compiler hands back an `NSError`; any other refusal dies at the selector, as a raise
/// inside Metal's compile does, since no path reaches it under a handler.
pub(crate) fn library(device: &VulkanDevice) -> Result<Option<VulkanLibrary>, NsErrorText> {
    let id = device.registry_id;
    // Held through the load, so two first asks load once.
    let mut loaded = LOADED.lock();
    if let Some((_, library)) = loaded.iter().find(|(r, _)| *r == id) {
        return Ok(Some(library.clone_ref()));
    }
    let made = load(device);
    if let Ok(Some(library)) = &made {
        loaded.push((id, library.clone_ref()));
    }
    made
}

/// [`METALLIB`] as a library of `device`.
fn load(device: &VulkanDevice) -> Result<Option<VulkanLibrary>, NsErrorText> {
    let made = crate::gpu::timing::creation(
        SELECTOR,
        "",
        || plain(|| device.raw.new_library_with_data(METALLIB)),
        || "the platform's shaders".to_owned(),
        |made| match made {
            Ok(_) => "made".to_owned(),
            Err(e) => format!("refused: {e}"),
        },
    );
    match made {
        Ok(raw) => Ok(Some(Obj::new(LibraryObject {
            raw,
            device: device.clone_ref(),
        }))),
        Err(e @ Error::Compile(_)) => Err(error_text(&e)),
        Err(e) => refused(SELECTOR, &e),
    }
}

#[cfg(test)]
mod tests {
    use metal2vulkan::metal::PixelFormat;
    use metal2vulkan::pipeline::{
        ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource,
    };
    use metal2vulkan_runtime as rt;

    use super::*;
    use crate::gpu::device::tests::dev;

    /// The display's and the blits' nine entry points.
    const FUNCTIONS: [&str; 9] = [
        "BlitInRGB_2P_XR10_A8",
        "BlitOutRGB_2P_XR10_A8",
        "displayPresentVertex",
        "displayPresentFragment",
        "displayPresentFragmentWithGammaTableAndColorMatrix",
        "displayPipeColorFragment",
        "displayPipeColorScanoutFragment",
        "displayPipeLutFragment",
        "displayPipeLutScanoutFragment",
    ];

    fn function(lib: &VulkanLibrary, name: &str) -> rt::Function {
        lib.raw
            .new_function_with_name(name)
            .unwrap_or_else(|e| panic!("no function {name}: {e}"))
    }

    /// The metallib loads as a library of the device holding the nine functions
    #[test]
    fn the_metallib_loads_with_the_nine_functions() {
        let d = dev();
        let lib = library(&d).unwrap().unwrap();
        assert_eq!(lib.device.id(), d.id());
        let mut names = lib.raw.function_names();
        names.sort();
        let mut nine = FUNCTIONS.map(str::to_owned);
        nine.sort();
        assert_eq!(names, nine);
    }

    /// The metallibs the repository holds for a build without Apple's tools (`metallibs/`,
    /// `build.rs`) are this tree's: their record is the one `build.rs` writes of this tree's
    /// commands and sources, and the directory holds every metallib it names and no other. So a
    /// source changed on a Mac without `cargo xtask metallibs` fails here, not on Linux.
    #[test]
    fn the_committed_metallibs_are_this_trees() {
        let dir = crate::gpu::device::tests::crate_dir().join("metallibs");
        let mend = "run `cargo xtask metallibs` on a Mac and commit what it writes";
        let tree = include_str!(concat!(env!("OUT_DIR"), "/metallibs.toml"));
        let committed = std::fs::read_to_string(dir.join("metallibs.toml"))
            .unwrap_or_else(|e| panic!("{}: {e}: {mend}", dir.display()));
        assert_eq!(
            committed,
            tree,
            "{}'s record is not this tree's: {mend}",
            dir.display()
        );
        let mut named: Vec<&str> = tree
            .lines()
            .filter_map(|line| line.split_once(" = "))
            .map(|(file, _)| file.trim_matches('"'))
            .collect();
        named.sort_unstable();
        let mut held: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "metallibs.toml")
            .collect();
        held.sort_unstable();
        assert_eq!(held, named, "{}: {mend}", dir.display());
    }

    /// A later ask answers the library the first one loaded
    #[test]
    fn loaded_once_per_device() {
        let d = dev();
        let a = library(&d).unwrap().unwrap();
        let b = library(&d).unwrap().unwrap();
        assert_eq!(a.id(), b.id());
    }

    /// The present pipelines are made as `display/pipeline.rs` makes them, and the display
    /// colour's as `reims-vgpu`'s `graphics/sink.rs` does: the vertex function, one of the six
    /// fragment functions, colour 0 a frame target's format, blending off, no vertex descriptor;
    /// for each format a frame target takes
    #[test]
    fn both_present_pipelines_are_made() {
        let d = dev();
        let lib = library(&d).unwrap().unwrap();
        let vertex = function(&lib, "displayPresentVertex");
        for fragment in [
            "displayPresentFragment",
            "displayPresentFragmentWithGammaTableAndColorMatrix",
            "displayPipeColorFragment",
            "displayPipeColorScanoutFragment",
            "displayPipeLutFragment",
            "displayPipeLutScanoutFragment",
        ] {
            let fragment = function(&lib, fragment);
            for format in [
                PixelFormat::BGRA8Unorm,
                PixelFormat::BGRA8UnormSRGB,
                PixelFormat::RGBA8Unorm,
                PixelFormat::RGBA8UnormSRGB,
            ] {
                let mut desc = RenderPipelineDescriptor::new(StageSource::Function(
                    vertex.function_descriptor(),
                ));
                desc.fragment = Some(StageSource::Function(fragment.function_descriptor()));
                desc.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
                d.raw
                    .new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
                    .unwrap_or_else(|e| panic!("{format:?}: {e}"));
            }
        }
    }
}
