//! [`run`]: what every public entry point runs inside, its last release included.
//!
//! An entry marks its thread as the library's (a panic then aborts at its op, 6.5), pushes a real
//! autorelease pool and makes a holder ([`Held`]), which its work holds to ([`Hold::of`]). No
//! handler is live in it, so a raise in it dies at the raise. When the entry's work returns, the
//! holder releases what was held to it, most recent first, and then the pool pops. Both come
//! before the call returns to the host, after the entry's own work:
//! - a value core holds to the entry's holder (the kernel task a lookup reached, a restore's
//!   mapper surface and texture) is released there, not in the host's pool, which a host in C
//!   does not have;
//! - what Metal autoreleases on the host's thread (a `+0` result, such as a frame's command
//!   buffer) goes at the pool's pop, so no uncommitted buffer outlives the call and holds one of
//!   its queue's 64 slots.

use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::rt::{Held, Hold, Pool, enter_ours};

/// A raise that reaches an entry point: nothing catches it there, so it ends the process.
pub(crate) fn no_catch<T>(r: RaiseResult<T>) -> T {
    r.unwrap_or_else(|raised| raised.fatal())
}

/// Runs one entry's `body` marked as ours, inside a real pool and with a holder; `body` gets a
/// hold of that holder. The holder releases what it holds when `body` returns, then the pool pops
/// (module doc). An entry reached inside another (a release of the last reference inside a call)
/// makes its own.
pub(crate) fn run<T>(body: impl for<'u> FnOnce(Hold<'u>) -> T) -> T {
    let _ours = enter_ours();
    let pool = Pool::here();
    let held = Held::new();
    let r = body(Hold::of(&held));
    drop(held);
    drop(pool);
    r
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::rt::is_our_thread;
    use reims_vgpu_core::sync::Mutex;

    use super::*;

    /// Records its name in a shared log when dropped.
    struct Rel(&'static str, Arc<Mutex<Vec<String>>>);

    impl Drop for Rel {
        fn drop(&mut self) {
            self.1.lock().push(format!("release {}", self.0));
        }
    }

    /// The body runs marked as ours; what it holds to the entry's holder is released most
    /// recent first when it returns, before `run` does, and the mark goes after
    #[test]
    fn an_entry_releases_what_it_holds_before_it_returns() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let got = run(|hold| {
            assert!(is_our_thread());
            hold.keep(Rel("first", log.clone()));
            hold.keep(Rel("second", log.clone()));
            log.lock().push("body".into());
            7
        });
        assert_eq!(got, 7);
        assert!(!is_our_thread());
        assert_eq!(*log.lock(), ["body", "release second", "release first"]);
    }

    /// With the real pool: what the body autoreleases goes when the pool pops, after the holder
    /// has released, and both before `run` returns. In a child, since the pool is installed for
    /// the process.
    #[cfg(target_os = "macos")]
    #[test]
    fn an_entry_pops_its_pool_after_its_holder_and_before_it_returns() {
        use objc2::rc::Retained;
        use objc2::runtime::{AnyObject, NSObject};

        fn retain_count(o: &NSObject) -> usize {
            // SAFETY: `-retainCount` of a live object.
            unsafe { objc2::msg_send![o, retainCount] }
        }

        /// Records the object's retain count when dropped.
        struct Probe(Retained<NSObject>, Arc<Mutex<Vec<usize>>>);
        // SAFETY: the probe is dropped on the thread that made it (the entry's).
        unsafe impl Send for Probe {}
        impl Drop for Probe {
            fn drop(&mut self) {
                self.1.lock().push(retain_count(&self.0));
            }
        }

        if std::env::var_os("RVG_ENTRY_POOL_CHILD").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "entry::tests::an_entry_pops_its_pool_after_its_holder_and_before_it_returns",
                    "--test-threads=1",
                ])
                .env("RVG_ENTRY_POOL_CHILD", "1")
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            assert!(
                String::from_utf8_lossy(&out.stdout).contains(" 1 passed"),
                "{out:?}"
            );
            return;
        }
        reims_vgpu_metal::os::autorelease::install();
        let object = NSObject::new();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let alone = retain_count(&object);
        run(|hold| {
            let pooled = Retained::autorelease_ptr(object.clone());
            assert!(!pooled.cast::<AnyObject>().is_null());
            hold.keep(Probe(object.clone(), seen.clone()));
        });
        // The probe saw the pool's reference, its own and ours; after `run` only ours is left.
        assert_eq!(*seen.lock(), [alone + 2]);
        assert_eq!(retain_count(&object), alone);
    }

    /// An entry inside another releases its own holds at its own end, and restores the outer
    /// mark
    #[test]
    fn a_nested_entry_releases_at_its_own_end() {
        let log = Arc::new(Mutex::new(Vec::new()));
        run(|hold| {
            hold.keep(Rel("outer", log.clone()));
            run(|hold| hold.keep(Rel("inner", log.clone())));
            log.lock().push("inner returned".into());
            assert!(is_our_thread());
        });
        assert_eq!(
            *log.lock(),
            ["release inner", "inner returned", "release outer"]
        );
    }
}
