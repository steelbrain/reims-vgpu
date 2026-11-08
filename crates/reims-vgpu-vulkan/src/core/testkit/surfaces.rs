//! Fake: `FakeSurfaces` with retain counting (architecture 13.2).
//!
//! An IOSurface is a [`FakeSurface`] handle whose `CFGetRetainCount` is its live handle count: the
//! creator's reference, every `.clone_ref()` copy core makes, and one for each live fake texture or
//! buffer made over it (their models hold a handle, as Metal retains the surface). So the
//! guest-visible count falls out of the object lifetimes. Retains and releases are logged as
//! `CFRetain`/`CFRelease`.

use std::sync::Arc;

use super::calllog::Arg;
use super::guest_ram::RawRegion;
use super::objects::{FAKE_MAX_BYTES, Fk, Model, SurfaceModel, World, kind};
use super::platform::FakePlatform;
use crate::core::host::HostPtr;
use crate::core::num::ToU64;
use crate::core::platform::os::surface::{BackingSurfaceSpec, IoSurfaces, Surface, SurfaceSpec};
use crate::core::platform::os::vm::KernReturn;

/// A fake `IOSurfaceRef`.
pub type FakeSurface = Fk<kind::Surface>;

/// IOSurface creation over a fake world.
#[derive(Debug, Clone)]
pub struct FakeSurfaces {
    world: Arc<World>,
}

impl FakeSurfaces {
    /// Surfaces of `world`.
    pub fn new(world: Arc<World>) -> FakeSurfaces {
        FakeSurfaces { world }
    }

    fn make(&self, m: SurfaceModel) -> FakeSurface {
        let mem = (m.alloc_size <= FAKE_MAX_BYTES)
            .then(|| RawRegion::new(m.alloc_size))
            .flatten();
        self.world
            .new_obj(Model::Surface(SurfaceModel { mem, ..m }))
    }

    /// A surface with the given geometry, as a test's stand-in for one the host made.
    pub fn make_simple(&self, width: u64, height: u64, pixel_format: u32, bpe: u64) -> FakeSurface {
        let bpr = width.wrapping_mul(bpe);
        self.make(SurfaceModel {
            pixel_format,
            width,
            height,
            bytes_per_element: bpe,
            bytes_per_row: bpr,
            plane_count: 0,
            alloc_size: bpr.wrapping_mul(height),
            ..SurfaceModel::default()
        })
    }
}

impl IoSurfaces<FakePlatform> for FakeSurfaces {
    fn create(&self, spec: &SurfaceSpec) -> Option<FakeSurface> {
        self.world.log.push(
            crate::core::platform::ObjId(0),
            "IOSurface",
            "IOSurfaceCreate",
            vec![Arg::val(spec)],
        );
        let step = self
            .world
            .script
            .next(crate::core::platform::ObjId(0), "IOSurfaceCreate");
        step.getter("IOSurfaceCreate");
        if step.is_nil() {
            return None;
        }
        Some(self.make(SurfaceModel {
            pixel_format: spec.pixel_format,
            width: spec.width,
            height: spec.height,
            bytes_per_element: spec.bytes_per_element,
            bytes_per_row: spec.bytes_per_row,
            plane_count: spec.planes.as_ref().map_or(0, |p| p.len().to_u64()),
            alloc_size: spec.alloc_size,
            protection_options: spec.protection_options,
            spec: Some(spec.clone()),
            ..SurfaceModel::default()
        }))
    }

    /// The fake has no dictionary: the spec's Rust form stands in for its `-description`.
    fn describe(&self, spec: &SurfaceSpec) -> String {
        format!("{spec:?}")
    }

    fn create_backing(&self, spec: &BackingSurfaceSpec) -> Option<FakeSurface> {
        self.world.log.push(
            crate::core::platform::ObjId(0),
            "IOSurface",
            "initWithProperties:",
            vec![Arg::val(spec)],
        );
        let step = self
            .world
            .script
            .next(crate::core::platform::ObjId(0), "initWithProperties:");
        step.getter("initWithProperties:");
        if step.is_nil() {
            return None;
        }
        let first = spec.planes.first().copied().unwrap_or_default();
        let alloc = spec
            .planes
            .iter()
            .map(|p| {
                u64::from(p.offset).wrapping_add(
                    i64::from(p.bytes_per_row)
                        .cast_unsigned()
                        .wrapping_mul(u64::from(p.height)),
                )
            })
            .max()
            .unwrap_or(0);
        Some(self.make(SurfaceModel {
            pixel_format: spec.pixel_format,
            width: u64::from(spec.width),
            height: u64::from(spec.height),
            bytes_per_element: i64::from(first.bytes_per_element).cast_unsigned(),
            bytes_per_row: i64::from(first.bytes_per_row).cast_unsigned(),
            plane_count: spec.planes.len().to_u64(),
            alloc_size: alloc,
            backing_spec: Some(spec.clone()),
            ..SurfaceModel::default()
        }))
    }
}

impl Fk<kind::Surface> {
    fn with_surface<T>(&self, f: impl FnOnce(&mut SurfaceModel) -> T) -> Option<T> {
        match &mut self.state().model {
            Model::Surface(m) => Some(f(m)),
            _ => None,
        }
    }

    fn get_u64(&self, sel: &'static str, f: impl FnOnce(&SurfaceModel) -> u64) -> u64 {
        let step = self.call(sel, vec![]);
        step.getter_value(sel, || self.with_surface(|m| f(m)).unwrap_or(0))
    }

    /// The surface's memory, for a test to fill or read.
    pub fn memory(&self) -> Option<Arc<RawRegion>> {
        self.with_surface(|m| m.mem.clone()).flatten()
    }

    /// Outstanding locks.
    pub fn lock_count(&self) -> i64 {
        self.with_surface(|m| m.lock_count).unwrap_or(0)
    }
}

impl Surface for Fk<kind::Surface> {
    fn retain_count(&self) -> i64 {
        let step = self.call("CFGetRetainCount", vec![]);
        step.getter_value("CFGetRetainCount", || Fk::retain_count(self))
    }

    fn pixel_format(&self) -> u32 {
        let step = self.call("IOSurfaceGetPixelFormat", vec![]);
        step.getter_value("IOSurfaceGetPixelFormat", || {
            self.with_surface(|m| m.pixel_format).unwrap_or(0)
        })
    }

    fn width(&self) -> u64 {
        self.get_u64("IOSurfaceGetWidth", |m| m.width)
    }

    fn height(&self) -> u64 {
        self.get_u64("IOSurfaceGetHeight", |m| m.height)
    }

    fn bytes_per_element(&self) -> u64 {
        self.get_u64("IOSurfaceGetBytesPerElement", |m| m.bytes_per_element)
    }

    fn bytes_per_row(&self) -> u64 {
        self.get_u64("IOSurfaceGetBytesPerRow", |m| m.bytes_per_row)
    }

    fn plane_count(&self) -> u64 {
        self.get_u64("IOSurfaceGetPlaneCount", |m| m.plane_count)
    }

    fn allows_pixel_size_casting(&self) -> bool {
        let step = self.call("IOSurfaceAllowsPixelSizeCasting", vec![]);
        step.getter_value("IOSurfaceAllowsPixelSizeCasting", || {
            self.with_surface(|m| m.allows_pixel_size_casting)
                .unwrap_or(false)
        })
    }

    fn protection_options(&self) -> u64 {
        self.get_u64("IOSurfaceGetProtectionOptions", |m| m.protection_options)
    }

    fn lock(&self, options: u32) -> KernReturn {
        let step = self.call("IOSurfaceLock", vec![Arg::U32(options)]);
        step.getter_value("IOSurfaceLock", || {
            let _ = self.with_surface(|m| m.lock_count = m.lock_count.wrapping_add(1));
            KernReturn::SUCCESS
        })
    }

    fn unlock(&self, options: u32) -> KernReturn {
        let step = self.call("IOSurfaceUnlock", vec![Arg::U32(options)]);
        step.getter_value("IOSurfaceUnlock", || {
            let _ = self.with_surface(|m| m.lock_count = m.lock_count.wrapping_sub(1));
            KernReturn::SUCCESS
        })
    }

    fn base_address(&self) -> HostPtr {
        let step = self.call("IOSurfaceGetBaseAddress", vec![]);
        step.getter_value("IOSurfaceGetBaseAddress", || {
            self.memory().map_or(HostPtr::NULL, |r| r.addr())
        })
    }

    fn alloc_size(&self) -> u64 {
        self.get_u64("IOSurfaceGetAllocSize", |m| m.alloc_size)
    }

    /// A new fake surface over the same memory and geometry, with its own handle count.
    fn lookup_by_port(&self) -> Option<Self> {
        let step = self.call("IOSurfaceLookupFromMachPort", vec![]);
        step.getter("IOSurfaceLookupFromMachPort");
        if step.is_nil() {
            return None;
        }
        let model = self.with_surface(|m| SurfaceModel {
            mem: m.mem.clone(),
            pixel_format: m.pixel_format,
            width: m.width,
            height: m.height,
            bytes_per_element: m.bytes_per_element,
            bytes_per_row: m.bytes_per_row,
            plane_count: m.plane_count,
            alloc_size: m.alloc_size,
            protection_options: m.protection_options,
            allows_pixel_size_casting: m.allows_pixel_size_casting,
            lock_count: m.lock_count,
            spec: m.spec.clone(),
            backing_spec: m.backing_spec.clone(),
        })?;
        Some(self.world().new_obj(Model::Surface(model)))
    }
}

#[cfg(test)]
mod tests {
    use super::super::script::Step;
    use super::*;
    use crate::core::platform::Handle;
    use crate::core::platform::os::surface::BackingPlaneSpec;

    fn surfaces() -> (Arc<World>, FakeSurfaces) {
        let w = World::new();
        (w.clone(), FakeSurfaces::new(w))
    }

    /// IOSurfaceCreate models the spec; nil when scripted
    #[test]
    fn create_from_spec() {
        let (w, s) = surfaces();
        let spec = SurfaceSpec {
            pixel_format: 0x4247_5241,
            width: 16,
            height: 8,
            bytes_per_element: 4,
            bytes_per_row: 64,
            alloc_size: 512,
            protection_options: 3,
            ..SurfaceSpec::default()
        };
        let sf = s.create(&spec).unwrap();
        assert_eq!(sf.pixel_format(), 0x4247_5241);
        assert_eq!((Surface::width(&sf), Surface::height(&sf)), (16, 8));
        assert_eq!((sf.bytes_per_element(), sf.bytes_per_row()), (4, 64));
        assert_eq!((sf.alloc_size(), sf.plane_count()), (512, 0));
        assert_eq!(Surface::protection_options(&sf), 3);
        assert!(!sf.allows_pixel_size_casting());
        assert!(!sf.base_address().is_null());
        w.script.push("IOSurfaceCreate", Step::Nil);
        assert!(s.create(&spec).is_none());
        assert_eq!(w.log.count("IOSurfaceCreate"), 2);
    }

    /// The retain count follows handles, including a model's
    #[test]
    fn retain_count_follows_handles() {
        let (w, s) = surfaces();
        let sf = s.make_simple(4, 4, 0, 4);
        assert_eq!(Surface::retain_count(&sf), 1);
        let r = sf.clone_ref();
        assert_eq!(Surface::retain_count(&sf), 2);
        drop(r);
        // a texture over it holds one
        let tex: Fk<kind::Texture> =
            w.new_obj(Model::Texture(super::super::objects::TextureModel {
                surface: Some(sf.clone_ref()),
                ..Default::default()
            }));
        assert_eq!(Surface::retain_count(&sf), 2);
        drop(tex);
        assert_eq!(Surface::retain_count(&sf), 1);
        assert!(w.log.count("CFRetain") >= 2);
        w.script.push("CFGetRetainCount", Step::value(9i64));
        assert_eq!(Surface::retain_count(&sf), 9);
    }

    /// Locks count; initWithProperties computes the allocation from its planes
    #[test]
    fn locks_and_backing_surfaces() {
        let (_, s) = surfaces();
        let spec = BackingSurfaceSpec {
            pixel_format: 1,
            planes: vec![
                BackingPlaneSpec {
                    width: 4,
                    height: 4,
                    bytes_per_element: 4,
                    bytes_per_row: 16,
                    base: 0,
                    offset: 0,
                },
                BackingPlaneSpec {
                    width: 2,
                    height: 2,
                    bytes_per_element: 2,
                    bytes_per_row: 8,
                    base: 64,
                    offset: 64,
                },
            ],
            width: 4,
            height: 4,
        };
        let sf = s.create_backing(&spec).unwrap();
        assert_eq!(sf.alloc_size(), 80);
        assert_eq!(sf.plane_count(), 2);
        assert_eq!(sf.bytes_per_row(), 16);
        assert_eq!(sf.lock(1), KernReturn::SUCCESS);
        assert_eq!(sf.lock_count(), 1);
        assert_eq!(sf.unlock(1), KernReturn::SUCCESS);
        assert_eq!(sf.lock_count(), 0);
        assert_eq!(sf.memory().unwrap().len(), 80);
    }
}
