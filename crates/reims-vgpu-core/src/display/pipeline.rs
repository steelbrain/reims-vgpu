//! A display's render pipelines, by the target's pixel format and whether there is a gamma
//! table. The table is looked up and set under its mutex, never iterated.

use crate::collections::LookupMap;
use crate::display::DeathKind;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Library;
use crate::platform::values::RenderPipelineDesc;
use crate::platform::{Handle, Platform};
use crate::raise::OnFault;

/// The name of the vertex function in the platform's shader library.
pub const VERTEX_FUNCTION: &str = "displayPresentVertex";
/// The fragment function without a gamma table.
pub const FRAGMENT_FUNCTION: &str = "displayPresentFragment";
/// The fragment function with the gamma table and the colour matrix.
pub const FRAGMENT_FUNCTION_GAMMA_COLOR_MATRIX: &str =
    "displayPresentFragmentWithGammaTableAndColorMatrix";
/// The pipeline label.
pub const PIPELINE_LABEL: &str = "reims-vgpu.display-present";

/// Which pipeline: the render target's pixel format, and whether it applies the gamma table and
/// the colour matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PipelineKey {
    /// The render target's `MTLPixelFormat`.
    pub format: u64,
    /// Whether the gamma table and the colour matrix apply.
    pub gamma: bool,
}

/// A display's render pipelines.
pub struct DisplayPipelineCache<P: Platform> {
    /// The host's `MTLDevice` (retained).
    pub mtl_device: P::Device,
    /// The vertex function.
    pub vertex_function: P::Function,
    /// The fragment function without a gamma table.
    pub fragment_function: P::Function,
    /// The fragment function with the gamma table and the colour matrix.
    pub fragment_function_gamma_color_matrix: P::Function,
    /// The pipelines made so far.
    pub table: parking_lot::Mutex<LookupMap<PipelineKey, P::RenderPipeline>>,
}

impl<P: Platform> DisplayPipelineCache<P> {
    /// Loads the platform's shader library and its three display functions; Metal giving none
    /// of one dies. Always `Some`. Only a display's creation, an API entry with no handler,
    /// makes one, so a Metal raise dies at its send.
    pub fn new(device: P::Device) -> Option<Self> {
        // The error is never read.
        let library = device.new_shader_library().ok().flatten();
        let Some(library) = library else {
            crate::fatal!(DeathKind::NoLibrary)
        };
        // Each made with `OnFault::Die`: its raise has died at the send, so `Err` never comes.
        let function = |name| match library.new_function(OnFault::Die, name) {
            Ok(function) => function,
            Err(raised) => raised.fatal(),
        };
        // The three functions, then the three checks.
        let vertex = function(VERTEX_FUNCTION);
        let fragment = function(FRAGMENT_FUNCTION);
        let fragment_gamma = function(FRAGMENT_FUNCTION_GAMMA_COLOR_MATRIX);
        let Some(vertex_function) = vertex else {
            crate::fatal!(DeathKind::NoVertexFunction)
        };
        let Some(fragment_function) = fragment else {
            crate::fatal!(DeathKind::NoFragmentFunction)
        };
        let Some(fragment_function_gamma_color_matrix) = fragment_gamma else {
            crate::fatal!(DeathKind::NoGammaFragmentFunction)
        };
        let made = DisplayPipelineCache {
            mtl_device: device,
            vertex_function,
            fragment_function,
            fragment_function_gamma_color_matrix,
            table: parking_lot::Mutex::new(LookupMap::new()),
        };
        drop(library);
        Some(made)
    }

    /// The pipeline for `key` (a new reference), made and kept on a miss. A failed creation
    /// answers nil and keeps nothing, so it is tried again next time. No path reaches it under a
    /// handler: a raise making the pipeline dies at its send.
    pub fn pipeline_for(&self, key: PipelineKey) -> Option<P::RenderPipeline> {
        let mut table = self.table.lock();
        if let Some(pipeline) = table.get(&key) {
            return Some(pipeline.clone_ref());
        }
        let pipeline = self
            .mtl_device
            .new_render_pipeline_state(&RenderPipelineDesc {
                label: Some(PIPELINE_LABEL),
                vertex: &self.vertex_function,
                fragment: if key.gamma {
                    &self.fragment_function_gamma_color_matrix
                } else {
                    &self.fragment_function
                },
                color0_pixel_format: key.format,
                resource_index: None,
            })
            .ok()
            .flatten();
        if let Some(p) = &pipeline {
            table.insert(key, p.clone_ref());
        }
        // The table's lock is let go first.
        drop(table);
        pipeline
    }
}

impl<P: Platform> Drop for DisplayPipelineCache<P> {
    /// The pipelines first, then the fields in their order.
    fn drop(&mut self) {
        self.table.get_mut().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::calllog::Arg;
    use crate::testkit::capture::catch_death;
    use crate::testkit::{FakePlatform, Step};

    fn cache(p: &FakePlatform) -> DisplayPipelineCache<FakePlatform> {
        DisplayPipelineCache::new(p.device()).unwrap()
    }

    mod cache_init {
        use super::*;

        /// The shader library, then the three functions by name, in order.
        #[test]
        fn loads_library_and_three_functions() {
            let p = FakePlatform::new();
            let _c = cache(&p);
            let log = p.call_log();
            assert!(log.contains_in_order(&[
                "newLibraryWithSource:options:error:",
                "newFunctionWithName:",
                "newFunctionWithName:",
                "newFunctionWithName:",
            ]));
            let names: Vec<_> = log
                .calls_to("newFunctionWithName:")
                .into_iter()
                .map(|c| c.args[0].clone())
                .collect();
            assert_eq!(
                names,
                vec![
                    Arg::Str("displayPresentVertex".into()),
                    Arg::Str("displayPresentFragment".into()),
                    Arg::Str("displayPresentFragmentWithGammaTableAndColorMatrix".into()),
                ]
            );
        }

        /// No library (nil or an error): the assert.
        #[test]
        fn no_library_asserts() {
            for step in [Step::Nil, Step::Error(Default::default())] {
                let p = FakePlatform::new();
                p.script().push("newLibraryWithSource:options:error:", step);
                let r = catch_death(|| {
                    DisplayPipelineCache::<FakePlatform>::new(p.device()).map(|_| ())
                });
                assert_eq!(r.err(), Some(DeathKind::NoLibrary.site()));
            }
        }

        /// A missing function asserts at its line, after all three were asked for.
        #[test]
        fn missing_function_asserts_at_its_line() {
            for (skip, kind) in [
                (0u64, DeathKind::NoVertexFunction),
                (1, DeathKind::NoFragmentFunction),
                (2, DeathKind::NoGammaFragmentFunction),
            ] {
                let p = FakePlatform::new();
                p.script().after("newFunctionWithName:", skip, Step::Nil);
                let r = catch_death(|| {
                    DisplayPipelineCache::<FakePlatform>::new(p.device()).map(|_| ())
                });
                assert_eq!(r.err(), Some(kind.site()));
                assert_eq!(p.call_log().count("newFunctionWithName:"), 3);
            }
        }

        /// A Metal raise at the library load dies at the send: no handler is live on any path
        /// into the cache's creation.
        #[test]
        fn a_raise_at_the_library_load_dies_at_the_send() {
            let p = FakePlatform::new();
            p.script().push(
                "newLibraryWithSource:options:error:",
                Step::raise("NSGenericException", "x"),
            );
            let r =
                catch_death(|| DisplayPipelineCache::<FakePlatform>::new(p.device()).map(|_| ()));
            assert_eq!(
                r.err().map(|s| (s.kind, s.file, s.line)),
                Some((
                    crate::raise::DeathKind::UncaughtObjc,
                    "newLibraryWithSource:options:error:",
                    0
                ))
            );
        }
    }

    mod pipeline_for_descriptor {
        use super::*;

        /// The descriptor a miss makes the pipeline from: the label, the vertex function, the
        /// fragment function by `gamma`, the key's pixel format and no resource index.
        fn desc<'a>(
            c: &'a DisplayPipelineCache<FakePlatform>,
            key: PipelineKey,
        ) -> RenderPipelineDesc<'a, FakePlatform> {
            RenderPipelineDesc {
                label: Some("reims-vgpu.display-present"),
                vertex: &c.vertex_function,
                fragment: if key.gamma {
                    &c.fragment_function_gamma_color_matrix
                } else {
                    &c.fragment_function
                },
                color0_pixel_format: key.format,
                resource_index: None,
            }
        }

        /// A miss makes the pipeline from its descriptor: label, vertex function, the fragment
        /// function by `gamma`, the pixel format; then a hit returns the same pipeline without
        /// Metal.
        #[test]
        fn miss_builds_then_hit_reuses() {
            let p = FakePlatform::new();
            let c = cache(&p);
            let key = PipelineKey {
                format: 80,
                gamma: true,
            };
            let m = p.call_log().mark();
            let a = c.pipeline_for(key).unwrap();
            let calls = p.call_log().since(m);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].sel, "newRenderPipelineStateWithDescriptor:error:");
            assert_eq!(calls[0].args, vec![Arg::val(&desc(&c, key))]);
            let m = p.call_log().mark();
            let b = c.pipeline_for(key).unwrap();
            assert!(p.call_log().since(m).is_empty());
            assert_eq!(a.id(), b.id());
        }

        /// Without gamma the plain fragment function; each key its own pipeline.
        #[test]
        fn no_gamma_uses_plain_fragment() {
            let p = FakePlatform::new();
            let c = cache(&p);
            let with = PipelineKey {
                format: 80,
                gamma: true,
            };
            let without = PipelineKey {
                format: 80,
                gamma: false,
            };
            let a = c.pipeline_for(with).unwrap();
            let m = p.call_log().mark();
            let b = c.pipeline_for(without).unwrap();
            let made = p.call_log().since(m);
            assert_eq!(made[0].args, vec![Arg::val(&desc(&c, without))]);
            assert_ne!(a.id(), b.id());
        }

        /// A failed creation returns nil, caches nothing, and is tried again.
        #[test]
        fn failure_is_not_cached() {
            let p = FakePlatform::new();
            let c = cache(&p);
            let key = PipelineKey {
                format: 124,
                gamma: false,
            };
            p.script().push(
                "newRenderPipelineStateWithDescriptor:error:",
                Step::Error(Default::default()),
            );
            assert!(c.pipeline_for(key).is_none());
            assert!(c.table.lock().is_empty());
            assert!(c.pipeline_for(key).is_some());
            assert_eq!(
                p.call_log()
                    .count("newRenderPipelineStateWithDescriptor:error:"),
                2
            );
        }
    }

    /// The table's pipelines are released
    #[test]
    fn drop_releases_pipelines() {
        let p = FakePlatform::new();
        let c = cache(&p);
        let key = PipelineKey::default();
        let a = c.pipeline_for(key).unwrap();
        let held = a.retain_count();
        drop(c);
        assert_eq!(a.retain_count(), held - 1);
    }
}
