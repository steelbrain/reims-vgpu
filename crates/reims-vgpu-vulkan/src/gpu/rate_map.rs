//! `id<MTLRasterizationRateMap>`, its descriptor and its layers: core's `RateMap`,
//! `RateMapDescriptor` and `RateMapLayerDescriptor`, over the runtime's map.
//!
//! What Metal answers, measured on this host (macOS 26.7, M3 Max) without the API validation
//! layer:
//!
//! - **A layer descriptor** (`initWithSampleCount:`) is nil for a count of 0 on either axis, and
//!   otherwise holds that many zeroed samples on each axis, the horizontal then the vertical in
//!   one allocation; its `sampleCount` is the count with depth 0, whatever depth was asked.
//! - **A map descriptor** holds its layers by reference: the layer `layers` answers at an index is
//!   the one set there, nil past them. Its `layerCount` counts the layers set from index 0 up to
//!   the first index with none. Its mutability is Immutable (2) until set.
//! - **The minimum factor** (`setMinFactor:`) changes nothing a map answers (physical sizes,
//!   coordinate maps, parameter data), set on the map's descriptor or a reset's, so it is not
//!   kept.
//! - **An immutable map** is compiled at creation: the runtime's compile, whose parameter data
//!   is Metal's bytes and whose coordinate maps are Apple's arithmetic over them, bit for bit.
//!   Where the runtime refuses the descriptor it dies. Metal crashes there too,
//!   at three layers and at a screen side past 16384. It makes a map of a screen side of 0.
//! - **A map of no layers** is made, at the creation and at a reset, as Metal makes it, and the
//!   runtime makes it too: its layer count is 0, its parameter data Metal's (no
//!   horizontal tables, a vertical axis of the least quality). A pass over it loads, clears and
//!   stores as Metal's does. Metal's draws in such a pass rasterize by what earlier maps left
//!   (measured: nothing in one run, a corner of 1,018 fragments of a 64-pixel target after other
//!   maps' passes in another; the runtime's measurement records rows 0 to 15 drawn as with no map
//!   after a released map of one layer, here too), where the runtime's draw nothing, so a draw and
//!   an execution of indirect commands in it die, and a tile dispatch, which only draws were
//!   measured against, relays it, and dies in strict mode ([`crate::gpu::encode_render`]).
//!   Its coordinate maps end Metal's process, and the runtime refuses them, so they die here
//!   ([`mapped`]).
//! - **A mutable map** (`setMutability:` Mutable, 1) is the runtime's map of that mutability,
//!   not compiled until it is reset. Until then it answers, as Metal's does, the descriptor's
//!   screen size and layer count, a physical size of 0 by 0 for every layer, screen-to-physical
//!   (0, 0), and parameter data of zeros. Metal makes it whatever its descriptor holds (three
//!   layers, a screen side of 0 or 40000), as the runtime does, and crashes at
//!   physical-to-screen, which the runtime refuses, so it dies here ([`mapped`]).
//!   `resetUsingDescriptor:` is the runtime's: it compiles
//!   the descriptor given into the map over the tables it holds, keeping the map's own
//!   mutability, and past the layers the widths the map held below layer 2 (Metal's answer). A
//!   reset the runtime refuses dies, as no path reaches it under a handler.
//! - **`parameterBufferSizeAndAlign`** is 24,624 bytes aligned to 4 for every map, compiled or
//!   not. A compiled map's parameter data is the runtime's, which is Metal's, in Apple's layout,
//!   byte for byte, and which the translated decoder of a function reads with
//!   Apple's arithmetic; its copy writes all of it. A map not yet compiled writes 24,624 zeros,
//!   as Metal's does.
//! - **The granularity** is the runtime's, Metal's 32 by 32, depth 1, compiled or not; a
//!   physical size has depth 1.
//!
//! **In a pass.** A pass descriptor holds the map by reference (`desc.rs`).
//! - **A pass of a map whose rates vary** (a tile spanning other than as many screen pixels as
//!   physical ones) loads, clears and stores as Metal's does, and the map's parameter data and
//!   coordinate maps are Metal's, but the runtime's draws in it rasterize by regions whose lines
//!   are the map's tiles' exact lines, where Metal's are its fixed-point ones. The runtime
//!   announces it as it compiles the map, at its creation or a
//!   reset (`rasterization-rate-map`, [`Class::Varying`]), which is relayed, and a draw, an
//!   execution of indirect commands that is not empty and a tile dispatch in such a pass relay it
//!   again, and die in strict mode ([`crate::answer::Inexact::RateMap`],
//!   [`crate::gpu::encode_render`]).
//! - **A pass of a mutable map not yet reset** rasterizes nothing, as on Metal (measured: a
//!   full-screen triangle's vertex function writes its buffer, its fragment function never runs,
//!   and the colour, depth and stencil attachments keep their clears), as the runtime's pass
//!   does. A tile dispatch in it, which only draws were measured against, dies.
//! - **A pass over a mutable map**, reset or not, that no draw of the guest's took the place of
//!   ends with a draw of no vertices the guest did not encode ([`no_draw`],
//!   [`crate::gpu::encode_render`]), as the Metal platform's does. Without a draw call Metal's
//!   GPU hangs at it (measured on the M3 Max, macOS 27.0.1: its command buffer fails with code 2
//!   and stores nothing), and the runtime fails it so too; with the draw, both run it and store
//!   what it loaded or cleared. A guest's draw takes its place where it takes the Metal
//!   platform's: at least one whole triangle of a triangle list or strip, once or more, which the
//!   runtime took and counts as the pass's draw call. The draw's pipeline is made only for a pass
//!   that needs it, at the pass's end, as making it moves identifiers the guest reads: the
//!   runtime gives a pipeline state made without `supportIndirectCommandBuffers`, as this one
//!   is, the next number of one process-wide count as its `uniqueIdentifier` (read from the
//!   runtime's code), so every such state made after it answers one more than it would have
//!   (`no_draw_moves_later_plain_identifiers_only_for_a_pass_that_drew_nothing`). It takes no
//!   `gpuResourceID` and no identifier of a state with indirect command buffer support.
//! - **When the map is read.** Metal's pass reads the map as the GPU runs it (measured: a reset
//!   after the encoder is made, after its draw, after `endEncoding` and just after `commit` each
//!   shows in the pass, from a map not yet reset and from a compiled one). So does the runtime's:
//!   a reset encodes again each pass that took the map and whose command buffer is not yet
//!   submitted, by the map as reset. One it cannot encode again (an encoder that kept no log,
//!   such as one that left work for the runtime's host step) draws by the map as it took it,
//!   which it announces (`rate-map-reset-unencoded`), relayed, dying in strict mode
//!   ([`crate::answer::DIES`]). What the pass's encoder decides by the map (the deaths and relays
//!   of [`Class`]) it decides by the map's class at its creation, and the map keeps the command
//!   buffers whose passes took it, by their progress ([`crate::gpu::command::Progress`]), which
//!   outlives core's handle, with the class and the screen size each took. A reset while one of
//!   them may still run on the GPU (not committed, or committed and not complete) dies where the
//!   map as reset is of a class the one it took does not hold over ([`Class::holds_over`]), as
//!   the encoder's decisions would no longer be Metal's. A reset to the identity of another
//!   screen size draws as Metal's: Metal's GPU draws the pass over the screen the map had at the
//!   encoder's creation, to the 32-pixel tiles of the new screen where the new screen spans one
//!   such tile fewer, and so does the runtime's
//!   (`rt::RasterizationRateMap::reset_using_descriptor`; measured: a full-screen triangle in a
//!   pass of a 16-pixel target over a mutable map of a 16-pixel screen reset, before or after the
//!   commit, to the identity over 8 pixels shades all 256 pixels, over a map of 8 reset to 16 the
//!   8 by 8 corner, and in a pass of a 64-pixel target over a map of 64 reset to 32 or to 24 the
//!   32 by 32 corner and to 48 all of it, on both,
//!   `a_pass_over_a_mutable_map_draws_by_the_map_as_reset_before_it_runs`). A reset to a map
//!   other than the identity of another screen size dies there, where no measurement says what
//!   Metal's GPU draws ([`RateMapObject`]'s `reset`). One that goes uncommitted runs nothing, and
//!   no longer counts. What an encoder takes and the
//!   registration of its command buffer, once the encoder exists, are under the map's lock, which
//!   a reset's compile and check hold too.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use metal2vulkan::bind;
use metal2vulkan::metal::Mutability;
use metal2vulkan_runtime as rt;
use reims_vgpu_core::host::HostPtr;
use reims_vgpu_core::platform::Handle;
use reims_vgpu_core::platform::rate_map::{RateMap, RateMapDescriptor, RateMapLayerDescriptor};
use reims_vgpu_core::platform::types::{MtlCoordinate2D, MtlSize, SizeAndAlign};
use reims_vgpu_core::platform::values::RateMapDesc;
use reims_vgpu_core::raise::{OnFault, RaiseResult};
use reims_vgpu_core::sync::Mutex;

use crate::answer::{
    Policy, created_or_decided, get, hearing, plain, plain_at, refused, send, unrepresentable,
    unstopped_at,
};
use crate::gpu::command::{Progress, VulkanCommandBuffer};
use crate::gpu::device::VulkanDevice;
use crate::gpu::resource::VulkanBuffer;
use crate::platform::{Obj, VulkanPlatform};

/// `MTLMutabilityImmutable`: a descriptor's mutability until it is set.
const IMMUTABLE: u64 = 2;

/// `MTLMutabilityMutable`: a map made of a descriptor of it is compiled only at its reset.
const MUTABLE: u64 = 1;

/// Metal's `parameterBufferSizeAndAlign` of every map on this host, which the runtime answers
/// too.
const PARAMETER_DATA: SizeAndAlign = SizeAndAlign {
    size: 24624,
    align: 4,
};

// ---- Layers.

/// `MTLRasterizationRateLayerDescriptor *`: Metal's sample count and its sample storage.
#[derive(Debug)]
pub struct LayerObject {
    /// The horizontal samples' count, then the vertical's.
    counts: (usize, usize),
    /// The horizontal samples, then the vertical, as `f32` bits: core writes them in place
    /// through the storage getters' pointers.
    samples: Box<[AtomicU32]>,
}

/// `MTLRasterizationRateLayerDescriptor *`.
pub type VulkanRateMapLayerDescriptor = Obj<LayerObject>;

impl LayerObject {
    /// The samples from `at` on, `n` of them.
    fn read(&self, at: usize, n: usize) -> Vec<f32> {
        self.samples[at..at + n]
            .iter()
            .map(|s| f32::from_bits(s.load(Ordering::Relaxed)))
            .collect()
    }

    /// The compile layer's form of the layer, its samples as they stand now.
    fn samples(&self) -> bind::RateMapLayerDescriptor {
        let (w, h) = self.counts;
        bind::RateMapLayerDescriptor::new(self.read(0, w), self.read(w, h))
    }

    /// The address of sample `at`.
    fn storage(&self, at: usize) -> HostPtr {
        HostPtr::from_ptr(self.samples[at..].as_ptr().cast::<u32>().cast_mut())
    }
}

/// `initWithSampleCount:` of `sample_count`: nil for a count of 0 on either axis, or one of
/// samples it has no memory for, as Metal answers.
pub(crate) fn new_layer(sample_count: MtlSize) -> Option<VulkanRateMapLayerDescriptor> {
    let w = usize::try_from(sample_count.width).ok()?;
    let h = usize::try_from(sample_count.height).ok()?;
    if w == 0 || h == 0 {
        return None;
    }
    let n = w.checked_add(h)?;
    let mut samples = Vec::new();
    samples.try_reserve_exact(n).ok()?;
    samples.resize_with(n, || AtomicU32::new(0));
    Some(Obj::new(LayerObject {
        counts: (w, h),
        samples: samples.into_boxed_slice(),
    }))
}

impl RateMapLayerDescriptor for VulkanRateMapLayerDescriptor {
    fn horizontal_sample_storage(&self) -> HostPtr {
        plain(|| self.storage(0))
    }

    /// The counts, depth 0, as Metal answers them.
    fn sample_count(&self) -> MtlSize {
        let (w, h) = self.counts;
        MtlSize::new(w as u64, h as u64, 0)
    }

    fn vertical_sample_storage(&self) -> HostPtr {
        plain(|| self.storage(self.counts.0))
    }
}

// ---- Descriptors.

/// `MTLRasterizationRateMapDescriptor *`: its screen size, mutability and layers.
#[derive(Debug)]
pub struct DescriptorObject {
    /// `screenSize`.
    screen_size: MtlSize,
    /// `mutability`.
    mutability: u64,
    /// The layers set, by index.
    layers: Mutex<Vec<Option<VulkanRateMapLayerDescriptor>>>,
}

/// `MTLRasterizationRateMapDescriptor *`.
pub type VulkanRateMapDescriptor = Obj<DescriptorObject>;

/// A descriptor of `desc`'s screen size and mutability (Immutable when unset), with no layers;
/// its minimum factor, which changes nothing a map answers (the module docs), is not kept.
pub(crate) fn new_descriptor(desc: &RateMapDesc) -> VulkanRateMapDescriptor {
    Obj::new(DescriptorObject {
        screen_size: desc.screen_size,
        mutability: desc.mutability.unwrap_or(IMMUTABLE),
        layers: Mutex::new(Vec::new()),
    })
}

impl DescriptorObject {
    /// The layers set from index 0 up to the first index with none: Metal's `layerCount`.
    fn counted(&self) -> Vec<VulkanRateMapLayerDescriptor> {
        plain(|| {
            self.layers
                .lock()
                .iter()
                .map_while(|l| l.as_ref().map(Handle::clone_ref))
                .collect()
        })
    }

    /// The compile layer's form of the descriptor, its layers' samples as they stand now, for
    /// `sel`.
    fn compiled(&self, sel: &'static str) -> bind::RateMapDescriptor {
        let side = |v: u64| {
            u32::try_from(v)
                .unwrap_or_else(|_| unrepresentable(sel, &format!("a screen side of {v}")))
        };
        let mut d = bind::RateMapDescriptor::new([
            side(self.screen_size.width),
            side(self.screen_size.height),
        ]);
        d.layers = self.counted().iter().map(|l| l.samples()).collect();
        d.mutability = mutability(self.mutability);
        d
    }
}

impl RateMapDescriptor<VulkanPlatform> for VulkanRateMapDescriptor {
    /// The layer is kept by reference, as Metal keeps it.
    fn set_layer(
        &self,
        _on: OnFault,
        layer: &VulkanRateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()> {
        let sel = "setLayer:atIndex:";
        let at = usize::try_from(layer_index)
            .ok()
            .filter(|&at| at < 1 << 16)
            .unwrap_or_else(|| unrepresentable(sel, &format!("layer {layer_index}")));
        plain(|| {
            let mut layers = self.layers.lock();
            if layers.len() <= at {
                layers.resize_with(at + 1, || None);
            }
            layers[at] = Some(layer.clone_ref());
        });
        Ok(())
    }

    fn layer_count(&self) -> u64 {
        self.counted().len() as u64
    }

    /// The layer set at the index, nil where none is.
    fn layer(
        &self,
        _on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<VulkanRateMapLayerDescriptor>> {
        Ok(plain(|| {
            let layers = self.layers.lock();
            usize::try_from(layer_index)
                .ok()
                .and_then(|at| layers.get(at))
                .and_then(|l| l.as_ref().map(Handle::clone_ref))
        }))
    }
}

// ---- Maps.

/// What a map is, as far as a pass's deaths and relays tell maps apart
/// (`crate::gpu::encode_render`): a pass takes the class of the map as it is at its encoder's
/// creation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    /// A mutable map not yet reset, whose pass rasterizes nothing, as Metal's.
    Unset,
    /// A compiled map of no layers, whose draws Metal rasterizes by what earlier maps left.
    NoLayers,
    /// A compiled map of layers that the runtime takes as the identity inside the screen: it
    /// announced nothing as it compiled it.
    Identity,
    /// A compiled map whose rates vary (a tile spanning other than as many screen pixels as
    /// physical ones), which the runtime announced as it compiled it (`rasterization-rate-map`,
    /// [`crate::answer::HEARD`]): a pass over it draws by regions whose lines are its tiles'
    /// exact lines, where Apple's are its fixed-point ones, so a draw, an execution of indirect
    /// commands that is not empty and a tile dispatch in it relay it, and die in strict mode
    /// ([`crate::answer::Inexact::RateMap`], [`crate::gpu::encode_render`]).
    Varying,
}

impl Class {
    /// The class of `map`, just compiled, the runtime having announced `heard` as it compiled
    /// it.
    fn compiled(map: &rt::RasterizationRateMap, heard: &[rt::Degradation]) -> Class {
        if plain(|| map.layer_count()) == 0 {
            Class::NoLayers
        } else if heard.contains(&rt::Degradation::RasterizationRateMap) {
            Class::Varying
        } else {
            Class::Identity
        }
    }

    /// Whether a pass that took a map of this class, and may still run, draws as Metal's, or
    /// relayed where it does not, once the map is reset to one of class `to`: the runtime encodes
    /// it again by the map as reset, as Metal's GPU reads the map as it runs, but what the pass's
    /// encoder decided by the class it took stands (a tile dispatch's death in a pass that
    /// rasterizes nothing, a draw's in a pass of no layers, a resolve dropped or refused, a relay
    /// where the rates vary). Under any class a draw is exact, relayed or dead, and a resolve of
    /// another format is dropped only under the identity, so a reset to the identity, or to the
    /// same class, draws as Metal's or as relayed.
    fn holds_over(self, to: Class) -> bool {
        self == to || to == Class::Identity
    }
}

/// `id<MTLRasterizationRateMap>`: the runtime's map, made with its descriptor's mutability, and
/// that mutability as the guest set it.
#[derive(Debug)]
pub struct RateMapObject {
    /// The runtime's map, which answers every query and takes the resets.
    raw: rt::RasterizationRateMap,
    /// `mutability`: its descriptor's raw value, which a reset keeps.
    mutability: u64,
    /// What it is now, and the passes that took it, under one lock.
    inner: Mutex<Inner>,
}

/// A map's class and its passes.
#[derive(Debug)]
struct Inner {
    /// What it is now.
    class: Class,
    /// The passes that took the map, while the GPU may run them.
    passes: Vec<Took>,
}

/// What a pass took of its map, at its encoder's creation.
#[derive(Debug)]
struct Took {
    /// The progress of the pass's command buffer.
    progress: Arc<Progress>,
    /// The map's class.
    class: Class,
    /// The map's screen size, width and height: Metal's GPU draws the pass over it, reset or not.
    screen: (u64, u64),
}

/// The screen size of the runtime's map `raw`, width and height.
fn screen(raw: &rt::RasterizationRateMap) -> (u64, u64) {
    let s = plain(|| raw.screen_size());
    (s.width, s.height)
}

/// `id<MTLRasterizationRateMap>`.
pub type VulkanRateMap = Obj<RateMapObject>;

/// The runtime's mutability of a descriptor's raw `mutability`: a map of any value but
/// `MTLMutabilityMutable` is compiled at its creation, on Metal (measured) as in the runtime.
fn mutability(raw: u64) -> Mutability {
    match raw {
        MUTABLE => Mutability::Mutable,
        0 => Mutability::Default,
        _ => Mutability::Immutable,
    }
}

/// `newRasterizationRateMapWithDescriptor:`: the runtime's map of the descriptor, a mutable one
/// not yet compiled, dying where the runtime refuses it (the module docs), but at a gap of the
/// runtime's or the device's, which by default answers nil ([`crate::answer::relay_gap`]). Metal
/// answers no nil here.
pub(crate) fn new_map(
    device: &VulkanDevice,
    descriptor: &VulkanRateMapDescriptor,
) -> RaiseResult<Option<VulkanRateMap>> {
    let sel = "newRasterizationRateMapWithDescriptor:";
    let d = descriptor.compiled(sel);
    let (made, heard) =
        hearing(
            || match plain_at(sel, || device.raw.new_rasterization_rate_map(&d)) {
                Ok(raw) => Ok(Some(raw)),
                Err(e) => created_or_decided(sel, Policy::Dies, e),
            },
        );
    let Some(raw) = made? else {
        return Ok(None);
    };
    let class = match descriptor.mutability {
        MUTABLE => Class::Unset,
        _ => Class::compiled(&raw, &heard),
    };
    Ok(Some(Obj::new(RateMapObject {
        raw,
        mutability: descriptor.mutability,
        inner: Mutex::new(Inner {
            class,
            passes: Vec::new(),
        }),
    })))
}

impl RateMapObject {
    /// `open`, the creation of an encoder of a pass on `command_buffer` that takes the map, given
    /// the runtime's map and its class now, under the map's lock. Once it answers an encoder, the
    /// map keeps the command buffer's progress and the class the pass took while the GPU may run
    /// it; one that answers none leaves it unregistered. No reset comes between what the encoder
    /// takes and the registration.
    pub(crate) fn for_pass<T>(
        &self,
        command_buffer: &VulkanCommandBuffer,
        open: impl FnOnce(rt::RasterizationRateMap, Class) -> RaiseResult<Option<T>>,
    ) -> RaiseResult<Option<T>> {
        let mut inner = plain(|| self.inner.lock());
        let class = inner.class;
        let opened = open(self.raw.clone(), class)?;
        if opened.is_some() {
            let progress = command_buffer.progress("renderCommandEncoderWithDescriptor:");
            let screen = screen(&self.raw);
            inner.passes.retain(|t| t.progress.running());
            let known = (inner.passes.iter()).any(|t| {
                Arc::ptr_eq(&t.progress, &progress) && t.class == class && t.screen == screen
            });
            if !known {
                inner.passes.push(Took {
                    progress,
                    class,
                    screen,
                });
            }
        }
        Ok(opened)
    }

    /// Whether the map was made of a `Mutable` descriptor, a pass over which ends with an empty
    /// draw where no draw of the guest's took its place ([`no_draw`]).
    pub(crate) fn is_mutable(&self) -> bool {
        self.mutability == MUTABLE
    }
}

/// The metallib `build.rs` compiled `no_draw.metal` into: [`no_draw`]'s vertex function.
const NO_DRAW_LIBRARY: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/no_draw.metallib"));

/// The no-draw pipelines made so far, each with its device's registry ID. Each stays for the life
/// of the process.
static NO_DRAW: Mutex<Vec<(u64, rt::RenderPipelineState)>> = Mutex::new(Vec::new());

/// The pipeline a pass over a mutable map ends with a draw of no vertices through, as the Metal
/// platform's does (its `gpu::rate_map`), made on the first call for `device`'s registry ID. The
/// runtime fails the command buffer of a pass over a mutable map with no draw call as a GPU
/// hang, as Metal's GPU does, until a command buffer of the device with a draw call over such a
/// map was committed (`rate_mapped_pass.rs`), and counts a draw of no vertices as a draw call.
/// Its vertex function returns nothing ([`NO_DRAW_LIBRARY`]), its rasterization is off, it names
/// no attachment formats, and its largest vertex amplification count is the device's, which is
/// 1 on a device without multiview. A refusal of the library, the function or the pipeline dies
/// at `sel`, as the Metal platform's ends the process where Metal does not make one. Nothing here
/// stops a thread of a task that ended: the pass's end, which asks for it, stopped one first
/// (`crate::gpu::encode_render`). The device's caps it reads were read as core started the
/// device, before any task ran.
pub(crate) fn no_draw(sel: &'static str, device: &VulkanDevice) -> rt::RenderPipelineState {
    use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
    use reims_vgpu_core::platform::device::GpuDevice as _;

    let id = device.registry_id;
    // Held through the make, so two first asks make one.
    let mut made = NO_DRAW.lock();
    if let Some((_, pipeline)) = made.iter().find(|(r, _)| *r == id) {
        return pipeline.clone();
    }
    let library = unstopped_at(sel, || device.raw.new_library_with_data(NO_DRAW_LIBRARY))
        .unwrap_or_else(|e| refused(sel, &e));
    let function = unstopped_at(sel, || library.new_function_with_name("reims_vgpu_no_draw"))
        .unwrap_or_else(|e| refused(sel, &e));
    let mut desc =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    desc.rasterization_enabled = false;
    desc.max_vertex_amplification_count =
        u32::try_from(device.caps().max_vertex_amplification_count).unwrap_or(u32::MAX);
    let pipeline = unstopped_at(sel, || {
        (device.raw).new_render_pipeline_state(&desc, &rt::PipelineOptions::default())
    })
    .unwrap_or_else(|e| refused(sel, &e));
    made.push((id, pipeline.clone()));
    pipeline
}

/// Core's coordinate as the runtime's.
fn coordinate(c: MtlCoordinate2D) -> rt::Coordinate2D {
    rt::Coordinate2D::new(c.x, c.y)
}

/// The runtime's coordinate as core's.
fn mtl_coordinate(c: rt::Coordinate2D) -> MtlCoordinate2D {
    MtlCoordinate2D { x: c.x, y: c.y }
}

/// The runtime's size as core's.
fn mtl_size(s: rt::Size) -> MtlSize {
    MtlSize::new(s.width, s.height, s.depth)
}

/// The runtime's coordinate map `sel`, `map`: where it refuses as Metal ends the process (a map
/// of no layers, and physical-to-screen of a mutable map not yet reset) it dies naming why, and
/// any other refusal dies naming it.
fn mapped(
    sel: &'static str,
    map: impl FnOnce() -> Result<rt::Coordinate2D, rt::Error>,
) -> MtlCoordinate2D {
    let ends = |e: &rt::Error| match e {
        rt::Error::Invalid(i) => match i.problem {
            rt::Problem::RateMap(rt::RateMapProblem::NoLayers) => Some(
                "a coordinate map of a rasterization rate map of no layers, which ends Metal's \
                 process",
            ),
            rt::Problem::RateMap(rt::RateMapProblem::NotCompiled) => {
                Some("a mutable rate map not yet reset, which Metal crashes at")
            }
            _ => None,
        },
        _ => None,
    };
    match plain_at(sel, map) {
        Ok(c) => mtl_coordinate(c),
        Err(e) => match ends(&e) {
            Some(why) => unrepresentable(sel, why),
            None => refused(sel, &e),
        },
    }
}

impl RateMap<VulkanPlatform> for VulkanRateMap {
    /// Metal's 24,624 bytes at `offset`, the runtime's, which are Metal's: a compiled map's
    /// parameter data, and zeros for a mutable map not yet reset. Core checks the buffer's room
    /// for them first. A refusal of the runtime's (a `Private` buffer, where Metal's copy ends the
    /// process) dies.
    fn copy_parameter_data(&self, buffer: &VulkanBuffer, offset: u64) -> RaiseResult<()> {
        let sel = "copyParameterDataToBuffer:offset:";
        send(sel, Policy::Dies, || {
            self.raw.copy_parameter_data_to_buffer(&buffer.raw, offset)
        })
    }

    fn layer_count(&self) -> u64 {
        plain(|| self.raw.layer_count())
    }

    /// The runtime's map; Metal ends its process at a map of no layers and at a mutable map not
    /// yet reset, which die ([`mapped`]).
    fn physical_to_screen(
        &self,
        physical_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let sel = "mapPhysicalToScreenCoordinates:forLayer:";
        Ok(mapped(sel, || {
            self.raw
                .map_physical_to_screen_coordinates(coordinate(physical_coordinates), layer_index)
        }))
    }

    /// The runtime's map, (0, 0) for a mutable map not yet reset, as Metal's; Metal ends its
    /// process at a map of no layers, which dies ([`mapped`]).
    fn screen_to_physical(
        &self,
        screen_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        let sel = "mapScreenToPhysicalCoordinates:forLayer:";
        Ok(mapped(sel, || {
            self.raw
                .map_screen_to_physical_coordinates(coordinate(screen_coordinates), layer_index)
        }))
    }

    fn mutability(&self) -> u64 {
        self.mutability
    }

    /// Metal's (the module docs), which the runtime answers too: a runtime answer of another
    /// size dies, as its copy would write other bytes than Metal's.
    fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        let size = plain(|| self.raw.parameter_buffer_size_and_align()).size;
        if size != PARAMETER_DATA.size {
            unrepresentable(
                "parameterBufferSizeAndAlign",
                &format!(
                    "parameter data of {size} bytes, not Metal's {}",
                    PARAMETER_DATA.size
                ),
            );
        }
        PARAMETER_DATA
    }

    fn physical_granularity(&self) -> MtlSize {
        mtl_size(plain(|| self.raw.physical_granularity()))
    }

    /// The runtime's, which is Metal's: 0 by 0 for a mutable map not yet reset, and past the
    /// layers the width the map last held for that layer below 2, which a reset keeps.
    fn physical_size(&self, layer_index: u64) -> RaiseResult<MtlSize> {
        Ok(mtl_size(plain(|| {
            self.raw.physical_size_for_layer(layer_index)
        })))
    }

    /// The runtime's reset, which compiles the descriptor over the map's tables, keeps its
    /// mutability, and encodes again each pass that took the map whose command buffer is not yet
    /// submitted (the module docs); a refusal dies. So does a reset while a pass that took the
    /// map may still run, where what its encoder decided by the class it took does not hold over
    /// the map as reset ([`Class::holds_over`]), or where the map's screen size changes to one of
    /// a map other than the identity: Metal's GPU draws the pass over the screen as it was at the
    /// encoder's creation, as the runtime's does over the identity, and over another map by the
    /// map as reset within it, which no measurement compared with Metal's. The reset holds the
    /// map's lock, which a pass's creation holds too.
    fn reset(&self, descriptor: &VulkanRateMapDescriptor) {
        let sel = "resetUsingDescriptor:";
        let mut inner = plain(|| self.inner.lock());
        let d = descriptor.compiled(sel);
        let ((), heard) = hearing(|| get(sel, || self.raw.reset_using_descriptor(&d)));
        let class = Class::compiled(&self.raw, &heard);
        let (width, height) = screen(&self.raw);
        inner.passes.retain(|t| t.progress.running());
        if let Some(took) = inner.passes.iter().find(|t| !t.class.holds_over(class)) {
            unrepresentable(
                sel,
                &format!(
                    "a reset to a map of class {class:?} before a pass that took the map as \
                     {:?} has run: the runtime draws the pass by the map as reset, as Metal's \
                     GPU reads it, but what its encoder decided by the map it took stands",
                    took.class
                ),
            );
        }
        let resized = inner.passes.iter().find(|t| t.screen != (width, height));
        if let Some(took) = resized.filter(|_| class != Class::Identity) {
            let (w, h) = took.screen;
            unrepresentable(
                sel,
                &format!(
                    "a reset to a map of class {class:?} of a {width} by {height} screen before a \
                     pass that took the map of a {w} by {h} screen has run: Metal's GPU draws the \
                     pass over the screen the map had at the encoder's creation, and the runtime's \
                     by the map as reset within it, which no measurement compared with Metal's"
                ),
            );
        }
        inner.class = class;
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::platform::Platform;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::device::tests::dev;
    use crate::platform::tests::platform;

    /// A layer of `horizontal` and `vertical` samples, written through its storage as core
    /// writes them.
    fn layer<P: Platform>(
        p: &P,
        horizontal: &[f32],
        vertical: &[f32],
    ) -> P::RateMapLayerDescriptor {
        let l = p
            .new_rate_map_layer(
                OnFault::Die,
                MtlSize::new(horizontal.len() as u64, vertical.len() as u64, 0),
            )
            .unwrap()
            .unwrap();
        let write = |at: HostPtr, samples: &[f32]| {
            // SAFETY: the layer's storage holds as many samples as it was made with.
            unsafe {
                std::ptr::copy_nonoverlapping(samples.as_ptr(), at.as_ptr::<f32>(), samples.len())
            }
        };
        write(l.horizontal_sample_storage(), horizontal);
        write(l.vertical_sample_storage(), vertical);
        l
    }

    /// A descriptor of a 256 by 128 screen, mutable when asked, of one layer whose samples
    /// are `quality` then 1 across and `quality` down.
    fn descriptor<P: Platform>(p: &P, quality: f32, mutable: bool) -> P::RateMapDescriptor {
        let d = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(256, 128, 1),
            mutability: mutable.then_some(MUTABLE),
            min_factor: mutable.then_some(0.5),
        });
        d.set_layer(OnFault::Die, &layer(p, &[quality, 1.0], &[quality]), 0)
            .unwrap();
        d
    }

    /// What Metal crashes at dies at the selector: a map of three layers, which the runtime
    /// refuses, and physical-to-screen of a mutable map not yet reset
    #[test]
    fn what_metal_crashes_at_dies() {
        let p = platform();
        let d = dev();
        let three = p.rate_map_descriptor(&RateMapDesc {
            screen_size: MtlSize::new(64, 32, 1),
            ..RateMapDesc::default()
        });
        for i in 0..3 {
            three
                .set_layer(OnFault::Die, &layer(&p, &[1.0], &[1.0]), i)
                .unwrap();
        }
        assert_eq!(three.layer_count(), 3);
        let site = catch_death(|| d.new_rate_map(&three)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "newRasterizationRateMapWithDescriptor:")
        );
        let unset = d.new_rate_map(&descriptor(&p, 1.0, true)).unwrap().unwrap();
        let site = catch_death(|| unset.physical_to_screen(MtlCoordinate2D { x: 1.0, y: 1.0 }, 0))
            .unwrap_err();
        assert_eq!(site.file, "mapPhysicalToScreenCoordinates:forLayer:");
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
