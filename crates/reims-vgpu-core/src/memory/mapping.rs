//! A host range over guest memory, and what its last release does.

use std::sync::Arc;

use crate::host::HostPtr;
use crate::platform::Platform;

/// What a mapping's last release does.
pub enum Teardown<P: Platform> {
    /// Nothing: the range belongs to someone else.
    None,
    /// `mach_vm_deallocate(virtual_address, length)` through `P`'s `HostVm`: the map made the
    /// region.
    Vm {
        /// The platform whose `vm()` deallocates.
        platform: Arc<P>,
    },
}

/// A host range over guest memory. Shared as `Arc<Mapping<P>>`; the last drop runs the
/// teardown, so where the last reference goes decides when the host sees the unmap.
pub struct Mapping<P: Platform> {
    virtual_address: HostPtr,
    length: u64,
    writable: bool,
    teardown: Teardown<P>,
}

impl<P: Platform> std::fmt::Debug for Mapping<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mapping")
            .field("virtual_address", &self.virtual_address)
            .field("length", &self.length)
            .field("writable", &self.writable)
            .finish_non_exhaustive()
    }
}

impl<P: Platform> Mapping<P> {
    /// A mapping of `length` bytes at `virtual_address`.
    pub fn new(
        virtual_address: HostPtr,
        length: u64,
        writable: bool,
        teardown: Teardown<P>,
    ) -> Arc<Self> {
        Arc::new(Mapping {
            virtual_address,
            length,
            writable,
            teardown,
        })
    }

    /// Where the range is mapped.
    pub fn virtual_address(&self) -> HostPtr {
        self.virtual_address
    }

    /// The range's length.
    pub fn length(&self) -> u64 {
        self.length
    }

    /// Whether the range may be written.
    pub fn writable(&self) -> bool {
        self.writable
    }
}

impl<P: Platform> Drop for Mapping<P> {
    fn drop(&mut self) {
        match &self.teardown {
            Teardown::None => {}
            Teardown::Vm { platform } => {
                // The result is ignored.
                let _ = platform.vm().deallocate(self.virtual_address, self.length);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::calllog::Arg;
    use crate::testkit::platform::FakePlatform;

    mod plain {
        use super::*;

        #[test]
        fn keeps_address_length_and_flag_and_tears_down_nothing() {
            let p = Arc::new(FakePlatform::new());
            for w in [false, true] {
                let m =
                    Mapping::<FakePlatform>::new(HostPtr::from_addr(0x1000), 7, w, Teardown::None);
                assert_eq!(m.virtual_address(), HostPtr::from_addr(0x1000));
                assert_eq!(m.length(), 7);
                assert_eq!(m.writable(), w);
                drop(m);
            }
            assert!(p.call_log().is_empty());
        }

        #[test]
        fn plain_mapping_allows_null_and_zero() {
            let m = Mapping::<FakePlatform>::new(HostPtr::NULL, 0, true, Teardown::None);
            assert!(m.virtual_address().is_null());
            assert_eq!(m.length(), 0);
        }
    }

    mod vm {
        use super::*;

        #[test]
        fn deallocates_at_last_release_only() {
            let p = Arc::new(FakePlatform::new());
            let m = Mapping::new(
                HostPtr::from_addr(0x8000),
                0x4000,
                false,
                Teardown::Vm {
                    platform: p.clone(),
                },
            );
            assert!(!m.writable());
            let m2 = m.clone();
            drop(m);
            assert!(p.call_log().is_empty());
            drop(m2);
            let calls = p.call_log().calls();
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].sel, "mach_vm_deallocate");
            assert_eq!(calls[0].args, vec![Arg::Ptr(0x8000), Arg::U64(0x4000)]);
        }
    }
}
