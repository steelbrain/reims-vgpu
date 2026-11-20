//! The rt platform's instance on macOS ([`rt_instance`]): over Mesa's KosmicKrisp where our
//! distribution ships it, else rt's own instance, as on other hosts. rt's floor refuses MoltenVK,
//! the driver the system's loader finds on a Mac, and rt names no driver itself: the consumer
//! loads the one it ships and gives it by its `vk_icdGetInstanceProcAddr`
//! (rt's `Instance::with_drivers`, its README's "Instance and device").
//!
//! **The driver** is `libvulkan_kosmickrisp.dylib` in our distribution: beside the process's
//! executable, or in `../lib` from it, the first present (a link there is followed). Nothing else
//! chooses it: no search path and no environment variable. Where neither is present, or the one
//! present does not load, the instance is rt's own (`Instance::new`), as it was before this
//! module, and the device the system's drivers give, or none, refused at rt's floor. Either way
//! one line at info names the driver taken, and why rt's own instance was made.
//!
//! **The loader** is found the same way, once the driver is: `libvulkan.1.dylib` beside the
//! executable or in `../lib` from it, the first present; where neither is, rt's own search, which
//! asks `dlopen` for the loader by name.
//!
//! **Where the build puts them**: nowhere yet. A build that runs QEMU on rt puts each in QEMU's
//! build directory as a link, beside `qemu-system-aarch64`: the driver to a Mesa release's
//! KosmicKrisp, the loader to Homebrew's `vulkan-loader`. Shipping them with QEMU is owed.
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

/// The loader's library.
const LOADER: &str = "libvulkan.1.dylib";

/// The instance a process's rt platform makes (the module's docs), chosen once.
static ROUTE: OnceLock<Route<vk::PFN_vkGetInstanceProcAddr>> = OnceLock::new();

/// The instance the rt platform makes, of a driver's entry point `E`.
#[derive(Debug, PartialEq)]
enum Route<E> {
    /// rt's instance over the driver our distribution ships.
    Shipped {
        /// The driver's library, loaded.
        driver: PathBuf,
        /// Its entry point.
        entry: E,
        /// The loader our distribution ships; `None` for rt's own search.
        loader: Option<PathBuf>,
    },
    /// rt's own instance, over the system's drivers, and why.
    Own(String),
}

impl<E> Route<E> {
    /// The line that names the driver taken.
    fn line(&self) -> String {
        match self {
            Self::Shipped { driver, loader, .. } => {
                let loader = (loader.as_ref())
                    .map_or_else(|| "rt's own search".to_owned(), |p| p.display().to_string());
                let driver = driver.display();
                format!("the rt platform's Vulkan driver: KosmicKrisp, {driver}; loader: {loader}")
            }
            Self::Own(why) => {
                format!(
                    "the rt platform's Vulkan driver: the system's, by rt's own instance ({why})"
                )
            }
        }
    }
}

/// The rt platform's instance with `extensions` (the module's docs).
pub(crate) fn rt_instance(extensions: &[&CStr]) -> Result<rt::Instance, rt::Error> {
    let route = ROUTE.get_or_init(|| {
        let exe = std::env::current_exe().ok();
        route(exe.as_deref(), |path| path.exists(), load)
    });
    log::info!(target: TARGET, "{}", route.line());
    match route {
        Route::Shipped { entry, loader, .. } => {
            // SAFETY: the loader, where given, is the Vulkan loader's library; the entry point is
            // the driver's `vkGetInstanceProcAddr`, whose library stays loaded for the rest of
            // the process (the module's docs).
            unsafe { rt::Instance::with_drivers(loader.as_deref(), extensions, &[*entry]) }
        }
        Route::Own(_) => rt::Instance::new(extensions),
    }
}

/// The instance for the executable at `exe` (`None` where its path is unknown): over the driver
/// our distribution ships where `exists` finds it and `load` loads it, else rt's own (the module's
/// docs).
fn route<E>(
    exe: Option<&Path>,
    exists: impl Fn(&Path) -> bool,
    load: impl FnOnce(&Path) -> Result<E, String>,
) -> Route<E> {
    let Some(exe) = exe else {
        return Route::Own("the executable's path is unknown".to_owned());
    };
    let Some(driver) = shipped(exe, DRIVER, &exists) else {
        return Route::Own(format!(
            "no {DRIVER} beside the executable or in ../lib from it"
        ));
    };
    match load(&driver) {
        Ok(entry) => Route::Shipped {
            loader: shipped(exe, LOADER, &exists),
            driver,
            entry,
        },
        Err(e) => Route::Own(format!("{} does not load: {e}", driver.display())),
    }
}

/// `name` in our distribution for the executable at `exe`: beside it, or in `../lib` from it, the
/// first `exists` finds.
fn shipped(exe: &Path, name: &str, exists: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let dir = exe.parent()?;
    [dir.join(name), dir.join("../lib").join(name)]
        .into_iter()
        .find(|path| exists(path))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The executable the path tests run as.
    const EXE: &str = "/opt/reims/bin/qemu-system-aarch64";

    /// An `exists` that finds `present` alone.
    fn only(present: &[&str]) -> impl Fn(&Path) -> bool {
        let present: Vec<PathBuf> = present.iter().map(PathBuf::from).collect();
        move |path| present.iter().any(|p| p == path)
    }

    /// A load that loads anything, answering its path.
    fn loads(path: &Path) -> Result<PathBuf, String> {
        Ok(path.to_owned())
    }

    /// The driver beside the executable is taken, with the loader beside it.
    #[test]
    fn rt_driver_beside_the_executable() {
        let exists = only(&[
            "/opt/reims/bin/libvulkan_kosmickrisp.dylib",
            "/opt/reims/bin/libvulkan.1.dylib",
        ]);
        let driver = PathBuf::from("/opt/reims/bin/libvulkan_kosmickrisp.dylib");
        assert_eq!(
            route(Some(Path::new(EXE)), exists, loads),
            Route::Shipped {
                driver: driver.clone(),
                entry: driver,
                loader: Some(PathBuf::from("/opt/reims/bin/libvulkan.1.dylib")),
            }
        );
    }

    /// The driver in `../lib` is taken where none lies beside the executable, and the loader
    /// there too; with no loader shipped, rt's own search.
    #[test]
    fn rt_driver_in_lib() {
        let driver = PathBuf::from("/opt/reims/bin/../lib/libvulkan_kosmickrisp.dylib");
        let exists = only(&[
            "/opt/reims/bin/../lib/libvulkan_kosmickrisp.dylib",
            "/opt/reims/bin/../lib/libvulkan.1.dylib",
        ]);
        assert_eq!(
            route(Some(Path::new(EXE)), exists, loads),
            Route::Shipped {
                driver: driver.clone(),
                entry: driver.clone(),
                loader: Some(PathBuf::from("/opt/reims/bin/../lib/libvulkan.1.dylib")),
            }
        );
        let exists = only(&["/opt/reims/bin/../lib/libvulkan_kosmickrisp.dylib"]);
        assert_eq!(
            route(Some(Path::new(EXE)), exists, loads),
            Route::Shipped {
                driver: driver.clone(),
                entry: driver,
                loader: None,
            }
        );
    }

    /// The driver beside the executable is taken over one in `../lib`.
    #[test]
    fn rt_driver_beside_before_lib() {
        let exists = only(&[
            "/opt/reims/bin/../lib/libvulkan_kosmickrisp.dylib",
            "/opt/reims/bin/libvulkan_kosmickrisp.dylib",
        ]);
        assert_eq!(
            shipped(Path::new(EXE), DRIVER, exists),
            Some(PathBuf::from("/opt/reims/bin/libvulkan_kosmickrisp.dylib"))
        );
    }

    /// With no driver shipped, rt's own instance, and nothing loaded; the same where the
    /// executable's path is unknown. A loader shipped without the driver is not looked at.
    #[test]
    fn rt_driver_absent_is_rts_own_instance() {
        let exists = only(&["/opt/reims/bin/libvulkan.1.dylib"]);
        let never = |_: &Path| -> Result<(), String> { panic!("nothing to load") };
        let own = route(Some(Path::new(EXE)), exists, never);
        assert_eq!(
            own,
            Route::Own(
                "no libvulkan_kosmickrisp.dylib beside the executable or in ../lib from it"
                    .to_owned()
            )
        );
        assert!(own.line().contains("the system's, by rt's own instance"));
        let unknown = route(None, |_: &Path| true, never);
        assert_eq!(
            unknown,
            Route::Own("the executable's path is unknown".to_owned())
        );
    }

    /// With no driver on disk beside a real executable's path, rt's own instance, through the
    /// real file checks and loader; with a driver there that does not load, rt's own instance too,
    /// with `dlopen`'s reason. No KosmicKrisp is needed.
    #[test]
    fn rt_driver_absent_or_broken_on_disk_is_rts_own_instance() {
        let dir = std::env::temp_dir().join(format!("rvg-rt-driver-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the directory");
        let exe = dir.join("qemu-system-aarch64");
        assert!(matches!(
            route(Some(&exe), |p| p.exists(), load),
            Route::Own(why) if why.starts_with("no libvulkan_kosmickrisp.dylib")
        ));
        std::fs::write(dir.join(DRIVER), b"not a library").expect("the file");
        let broken = route(Some(&exe), |p| p.exists(), load);
        std::fs::remove_dir_all(&dir).expect("removed");
        assert!(matches!(
            &broken,
            Route::Own(why) if why.contains(DRIVER) && why.contains("does not load")
        ));
    }

    /// The shipped route's line names KosmicKrisp, its library and the loader.
    #[test]
    fn rt_driver_line_names_the_driver() {
        let route = Route::Shipped {
            driver: PathBuf::from("/opt/reims/bin/libvulkan_kosmickrisp.dylib"),
            entry: (),
            loader: None,
        };
        assert_eq!(
            route.line(),
            "the rt platform's Vulkan driver: KosmicKrisp, \
             /opt/reims/bin/libvulkan_kosmickrisp.dylib; loader: rt's own search"
        );
    }
}
