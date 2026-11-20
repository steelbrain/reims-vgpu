//! The rt platform's instance on macOS ([`rt_instance`]): rt's `Instance::with_drivers`, with the
//! Vulkan loader Homebrew's `vulkan-loader` installs and Mesa's KosmicKrisp, the driver the rt
//! platform runs on a Mac, given by its `vk_icdGetInstanceProcAddr`. rt's floor refuses MoltenVK,
//! the driver the system's loader finds there, and rt names no driver itself: the consumer loads
//! the one it ships and gives it (rt's README, "Instance and device").
//!
//! **Where the driver lies**: `libvulkan_kosmickrisp.dylib` beside the process's executable, or in
//! `../lib` from it, as a distribution lays out its binaries and their libraries (a link to it
//! there is followed). Where neither
//! loads, the instance is rt's `Instance::new` over the system's drivers, and the device is
//! refused at rt's floor as before, its reason logged.
//!
//! **Where the loader lies**: rt searches only `dlopen`'s default paths for it by name, which hold
//! no loader on a Mac with Homebrew under `/opt/homebrew`; so the first of `$VULKAN_SDK/lib`,
//! `/opt/homebrew/lib` and `/usr/local/lib` that has `libvulkan.1.dylib` is given, as the Vulkan
//! platform searches (`super::open_loader`), and rt's own search where none has.
//!
//! **The driver stays loaded** for the rest of the process once loaded: rt's contract asks that it
//! stay loaded while the instance or anything made from it lives, and the process's devices live
//! up to its exit.

use std::ffi::{CStr, CString, c_void};
use std::os::unix::ffi::OsStrExt as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ash::vk;
use metal2vulkan_rt as rt;

/// The log target.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The driver's library.
const DRIVER: &str = "libvulkan_kosmickrisp.dylib";

/// The entry point a driver built for the Vulkan loader exports.
const ENTRY: &CStr = c"vk_icdGetInstanceProcAddr";

/// The loader's library in the directories searched.
const LOADER: &str = "libvulkan.1.dylib";

/// The driver as loaded once a process: its library's path and its entry point, or why none
/// loaded.
static LOADED: OnceLock<Result<(PathBuf, vk::PFN_vkGetInstanceProcAddr), String>> = OnceLock::new();

/// The rt platform's instance with `extensions` (the module's docs).
pub(crate) fn rt_instance(extensions: &[&CStr]) -> Result<rt::Instance, rt::Error> {
    let loaded = LOADED.get_or_init(|| {
        let mut why = Vec::new();
        for path in driver_paths() {
            match load(&path) {
                Ok(entry) => return Ok((path, entry)),
                Err(e) => why.push(format!("{}: {e}", path.display())),
            }
        }
        Err(why.join("; "))
    });
    let loader = loader_path();
    match loaded {
        Ok((path, entry)) => {
            let through = (loader.as_ref())
                .map_or_else(|| "rt's own search".to_owned(), |p| p.display().to_string());
            log::info!(
                target: TARGET,
                "the rt platform's Vulkan driver: {}, through the loader {through}",
                path.display()
            );
            // SAFETY: the loader, where given, is the Vulkan loader's library; the entry point is
            // the driver's `vkGetInstanceProcAddr`, whose library stays loaded for the rest of
            // the process (the module's docs).
            unsafe { rt::Instance::with_drivers(loader.as_deref(), extensions, &[*entry]) }
        }
        Err(why) => {
            log::warn!(
                target: TARGET,
                "no Vulkan driver of the rt platform's loads ({why}): the system's drivers"
            );
            // SAFETY: as above, with no driver given.
            unsafe { rt::Instance::with_drivers(loader.as_deref(), extensions, &[]) }
        }
    }
}

/// Where the driver's library is looked for, in order (the module's docs).
fn driver_paths() -> Vec<PathBuf> {
    let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_owned))
    else {
        return Vec::new();
    };
    vec![dir.join(DRIVER), dir.join("../lib").join(DRIVER)]
}

/// The loader's library given to rt, the first of the directories searched that has one (the
/// module's docs); `None` for rt's own search.
fn loader_path() -> Option<PathBuf> {
    let sdk = std::env::var_os("VULKAN_SDK").map(|sdk| PathBuf::from(sdk).join("lib"));
    (sdk.into_iter())
        .chain(["/opt/homebrew/lib", "/usr/local/lib"].map(PathBuf::from))
        .map(|directory| directory.join(LOADER))
        .find(|path| path.exists())
}

/// Loads the driver's library at `path` for the rest of the process and finds its entry point;
/// `dlerror`'s text where either fails.
fn load(path: &Path) -> Result<vk::PFN_vkGetInstanceProcAddr, String> {
    let name = CString::new(path.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    // SAFETY: a C string; loading the driver runs its initialisers, which set up the driver alone.
    let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        return Err(dl_error());
    }
    // SAFETY: the handle is open; the name a C string.
    let symbol = unsafe { libc::dlsym(handle, ENTRY.as_ptr()) };
    if symbol.is_null() {
        let why = dl_error();
        // SAFETY: the handle is open, and nothing of the library is used.
        unsafe { libc::dlclose(handle) };
        return Err(why);
    }
    // SAFETY: the symbol is the driver's `vk_icdGetInstanceProcAddr`, of the type of
    // `vkGetInstanceProcAddr`; its library is never closed.
    Ok(unsafe { std::mem::transmute::<*mut c_void, vk::PFN_vkGetInstanceProcAddr>(symbol) })
}

/// `dlerror`'s text, or a stand-in where it has none.
fn dl_error() -> String {
    // SAFETY: `dlerror` answers null or a C string, valid until the next `dl*` call here.
    let text = unsafe { libc::dlerror() };
    if text.is_null() {
        return "no error text".to_owned();
    }
    // SAFETY: as above.
    unsafe { CStr::from_ptr(text) }
        .to_string_lossy()
        .into_owned()
}
