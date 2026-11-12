//! The Vulkan platform's dynamic libraries (`MTLDynamicLibrary`): the runtime's (metal2vulkan's
//! `dynamic_library.rs`), each made of a library the guest made (`newDynamicLibrary:error:`). A
//! compute pipeline's kernel links them through its function descriptor's libraries
//! ([`libraries`]): the preloaded in their order, and by install name the library each name loads,
//! which the guest read from the file the name names. Creating the state loads from a file only an
//! install name its search reaches that the descriptor does not hold, so a guest that names every
//! install name its kernel's libraries and these depend on has the runtime read no file.

use crate::answer::{Policy, error_text, relay_gap, send};
use crate::core::platform::types::NsErrorText;
use crate::core::platform::values::DynamicLibrariesDesc;
use crate::core::raise::RaiseResult;
use crate::gpu::resource::VulkanLibrary;
use crate::platform::{Obj, VulkanPlatform};
use metal2vulkan::pipeline::DynamicLibraries;
use metal2vulkan_runtime as rt;

/// `id<MTLDynamicLibrary>`.
pub type VulkanDynamicLibrary = Obj<rt::DynamicLibrary>;

/// `newDynamicLibrary:error:` of `library` on `device`: the runtime's dynamic library, or the
/// `NSError` of a library it refuses as one, as Metal refuses it (of another type, of no install
/// name, or whose code reads a function constant). A library of another device, which no guest
/// names, dies.
pub(crate) fn new_dynamic_library(
    device: &rt::Device,
    library: &VulkanLibrary,
) -> RaiseResult<Result<Option<VulkanDynamicLibrary>, NsErrorText>> {
    let sel = "newDynamicLibrary:error:";
    let made = send(sel, Policy::Dies, || {
        match device.new_dynamic_library(&library.raw) {
            Ok(raw) => Ok(Ok(raw)),
            Err(e @ rt::Error::Compile(metal2vulkan::error::Error::InvalidDescriptor(_))) => {
                Ok(Err(error_text(&e)))
            }
            Err(e) if relay_gap(sel, &e, None) => Ok(Err(error_text(&e))),
            Err(e) => Err(e),
        }
    })?;
    Ok(made.map(|raw| Some(Obj::new(raw))))
}

/// The runtime's dynamic libraries of a whole value's: the preloaded in their order, and each
/// install name with the library it loads.
pub(crate) fn libraries(l: &DynamicLibrariesDesc<VulkanPlatform>) -> DynamicLibraries {
    let mut out = DynamicLibraries::new();
    for library in &l.preloaded {
        out = out.with_preloaded(library.preloaded_library());
    }
    for (name, library) in &l.loaded {
        out = out.with_loaded(name.clone(), library.preloaded_library());
    }
    out
}
