//! The answer layer: where the runtime's typed errors become core's outcomes, as the metal
//! crate's message layer (`reims_vgpu_metal::msg`) decides a raise inside Metal.
//!
//! Every runtime call core asks for runs here, named by the Metal selector it stands for, which a
//! raise and a death name:
//! - [`send`]: the call's [`Policy`], its label, decides an [`Error::Invalid`]. By default
//!   (`Dies`, every encoder command and creator) it dies at the send naming the runtime's error,
//!   as every other refusal does: what Metal's validation stops, Metal without it goes on, nils
//!   or asserts at. Only where the evidence shows that Metal raises is it answered as the Metal
//!   platform answers a raise at that send: `Caught` (and `On(Return)`) as
//!   `Err(Raise::platform(sel))`, the caller's to catch, `Fatal` (and `On(Die)`) as a death at
//!   the send.
//! - [`get`]: a call whose every refusal dies at the call (a getter, a property setter, a send
//!   no path reaches under a handler); [`plain`], a call the runtime cannot refuse.
//! - [`create`]: a creator, where Metal answers nil: every refusal the runtime types
//!   [`Error::Nil`], what plain Metal answers nil for, and those of [`Nil`]'s set, which the
//!   runtime types otherwise where plain Metal was measured to answer nil, are `None`; a gap of
//!   the runtime's or the device's is relayed as `None` ([`relay_gap`]); the rest are as [`send`]
//!   decides them.
//! - [`error_text`]: a runtime error where Metal hands back an `NSError`.
//!
//! **What the runtime refuses where Metal would go on** (`NotImplemented`, `Unsupported`,
//! `AppleOnly`, a Vulkan failure, a lost device, a wait it will not make) is never a raise the
//! guest could catch: it is a death at the send, naming the selector and the runtime's error
//! ([`refused`]). **A method not built yet** dies the same way ([`unbuilt`]), and so does a raw
//! value the runtime's types cannot hold ([`unrepresentable`]).
//!
//! **A gap at a creator** is the exception ([`gap`], [`relay_gap`]): where the runtime refuses
//! an object Metal makes for a feature it or the device lacks (`Unsupported`, but by a GPU
//! family or a device Metal itself refuses it on), a piece it has not built (`NotImplemented`),
//! an Apple-only call (`AppleOnly`) or a compile it does not support (`Compile(Unsupported)`),
//! the creator answers nil, as Metal's does for an object it cannot make, and a pipeline or
//! library creator hands back the runtime's refusal as its `NSError` ([`error_text`]). The guest
//! sees the creation fail, as when Metal answers nil: Apple's framework then fails the work that
//! names the object, and the command buffer faults. It is relayed as [`Inexact::RefusedCreate`],
//! the first of its kind writing a line naming the refusal and every one counted, and dies in
//! strict mode as before. Nothing here makes the missing feature: the gap stays the runtime's
//! to close. A lost device, a Vulkan failure, an answer the runtime cannot prove and every other
//! refusal still die at the creator.
//!
//! **A call Metal never returns from** parks the calling thread for good ([`hang`]), as the
//! contract keeps a hang a hang.
//!
//! **A write the runtime refuses where Metal's GPU writes it** dies at the command: a
//! dispatch's or a draw's write through a texture over a surface of the 4:4:4 two-plane YCbCr
//! formats whose writes the runtime does not store as Metal does, the `_sRGB` twins 523 and 527
//! ([`crate::gpu::ycbcr::refused_write`]). The runtime stores the writes of 503, 507, 542 and
//! 545 into the surface's planes as Metal's GPU does, drops those of the packed 510 and 530 as
//! Metal's GPU drops them on this host, and refuses
//! where the command is encoded the writes of those it does not store: through a texture over
//! plane 1, which dies there naming the refusal (`writes_of_444_textures_over_plane_1_die`), and
//! through a slot that also reads the texture, or by a command that writes the planes through the
//! texture and another way, which the runtime refuses alike.
//!
//! **A degradation of a kind [`DIES`] lists is relayed, and dies in strict mode** ([`notice`],
//! the runtime's message sink), but those of [`DIES_BY_DEFAULT`], which die in either mode: the
//! runtime announces each kind of degradation once per device, but these kinds at every
//! occurrence in strict and census mode, and those of [`DIES_BY_DEFAULT`] in every mode
//! (`DeviceOptions::announce_each`, [`crate::platform`]).
//! By default the process goes on: the first occurrence of each kind writes the runtime's line,
//! and each occurrence announced is counted ([`relayed_counts`]), as a native device's relay is
//! (docs/accuracy-ledger.md: a difference from Metal's answer is logged, never a death of its
//! own). **Strict mode** ([`strict`], `REIMS_VGPU_STRICT=1`, read once per process) restores the
//! deaths, for tests and audits: the call that announced it dies once it returns, at its selector,
//! naming the notice, and a notice announced on a thread outside every call here (the runtime's
//! scheduler thread, at a host step) dies where it is announced. Where a kind's docs below say it
//! dies, they say what strict mode does. This crate's tests run strict, as each asserts the
//! deaths (the tests' `relaxed` runs a call under the default).
//!
//! **A difference this platform finds itself** ([`Inexact`], [`inexact`]): a value it computes
//! otherwise than Metal, which no runtime notice announces as a death (a `half` colour output's
//! blend, a bfloat attachment's codes, a `textureWriteRoundingMode` the
//! runtime does not take), or an answer it gives otherwise (a texture of protection options made
//! unprotected, the identifier of a pipeline state made with indirect command buffer support, and
//! of an indirect command buffer made after encoders whose take of Metal's range the runtime does
//! not model), is relayed alike, its first occurrence
//! writing a line naming the kind, and dies in strict mode. A death where Metal itself ends the
//! process, or where the runtime cannot perform the call at all, is a death in either mode.
//!
//! **On a native device** (off macOS, a driver other than MoltenVK, [`native`]), the kinds of
//! [`native::NATIVE_RELAYED`] are relayed in strict mode too, each with its ground, the first
//! occurrence's line written and each occurrence announced counted; every other kind is placed
//! as above, as it is on MoltenVK. The placement is the process's: native only while every device
//! it adopted is.
//!
//! **In census mode** (`REIMS_VGPU_CENSUS_DIR`, a diagnostic for inventory runs on Linux alone,
//! [`census`]), a degradation of [`DIES`] is recorded with the selector of the call it was
//! announced inside, and relayed where it would die, so one sweep finds every kind. The mode
//! says on stderr, once per process as it turns on, that the run is not faithful to Metal. On
//! macOS the variable is ignored: every such degradation dies there.
//!
//! **Every death here is a platform death** (`DeathId::Platform`): it ends the process, or, inside
//! a refusal scope, refuses the op that met it. The runtime catches a panic in its
//! sink, which only the test capture's death is: under the capture, a notice kept before it died
//! dies again at the thread's next call here.
//!
//! **A degradation of a kind [`HEARD`] lists is heard** by the call here it is announced inside
//! ([`hearing`]), at every occurrence too: the call dies where an answer it gives would differ
//! from Metal's, and otherwise goes on, the notice relayed (the first time alone, for
//! [`RELAYED_ONCE`]'s kinds).
//!
//! **Every other kind is relayed**: it goes to stderr as the runtime writes it without a sink
//! ([`relay`]), as every other message of the runtime's own does (its pipeline-creation log among
//! them; a `Verbose` one, which the runtime gives a sink alone, goes to the log at debug), each
//! line naming its kind. Each is relayed for one of these reasons, named at the kind
//! in the placement test (`every_degradation_kind_is_placed_once`):
//! - Metal's answer at a cost Metal does not pay, in time or memory (the performance kinds);
//! - measured equal to Metal's answer on this host;
//! - a difference the device itself forces, a stated fact of its hardware (as close as the
//!   hardware allows, the rest logged, no interpreter);
//! - an accepted class, within its stated bound: 8-bit linear filtering
//!   within one code of Metal's (`device-unorm8-linear-filter`), fast-math rewrites within what
//!   the function's AIR contract and reassociation flags allow (`fast-math-rewrites`), maths
//!   functions whose bound Metal does not state, each device's deviation measured
//!   (`device-transcendentals`), and a `half` colour output's rounding within its measured bound
//!   (`half-output-precision`, which past it is relayed as [`Inexact::HalfOutputBlend`], and dies
//!   in strict mode, [`HEARD`]); and a native device's own maths and sampling, each bound
//!   recorded ([`native::NATIVE_RELAYED`]);
//! - Metal's own answer there is not one value: undefined, memory-dependent or racing;
//! - no value the guest or the host sees;
//! - the difference dies where it shows, at a later call;
//! - pending a decision: `dont-care-load-tiles` and `extended-range-bits-kept`, and
//!   `heap-texture-lossless-layout` ([`HEARD`]).
//!
//! That a guest's boot or a suite case reaches a kind is no reason: a kind none of these covers
//! dies ([`DIES`]) until the runtime makes it exact or it is measured equal. The other heap
//! layout, `heap-texture-block-compressed-layout`, dies.

use std::cell::{Cell, RefCell};
use std::ptr::NonNull;

use metal2vulkan_runtime as rt;
use metal2vulkan_runtime::Error;
use reims_vgpu_core::platform::types::NsErrorText;
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};

pub mod census;
pub mod native;

/// What a refusal the runtime types `Invalid` does at a call: the call's label, as the metal
/// crate's `msg::Policy` labels the same send.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Policy {
    /// A call where Metal raises at nothing: the refusal dies at the send, naming the runtime's
    /// error ([`refused`]). The default for an encoder command and a creator.
    Dies,
    /// A call Metal raises at, which no path reaches without a handler: the refusal is `Err`,
    /// the caller's to catch.
    Caught,
    /// A call whose refusal dies whatever handler is live.
    Fatal,
    /// A call in a platform method reached both ways, whose caller says by the [`OnFault`] which
    /// way this call is.
    On(OnFault),
}

impl Policy {
    /// The way this call is reached: `Return` under a handler, `Die` with none.
    fn on(self) -> OnFault {
        match self {
            Policy::Caught => OnFault::Return,
            Policy::Fatal | Policy::Dies => OnFault::Die,
            Policy::On(on) => on,
        }
    }
}

/// Which refusals of a creator Metal answers with nil, beside every [`Error::Nil`] ([`create`]):
/// those the runtime types otherwise, where plain Metal was measured to answer nil, which are
/// the runtime's memory refusals. A buffer's length, a heap's size and a buffer's place in a
/// heap, which plain Metal answers nil for, are [`Error::Nil`]s.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Nil {
    /// None: every refusal is decided as [`send`] decides it.
    Never,
    /// A buffer's length or memory: a length of 0 or past the device's largest buffer, no memory
    /// for it, or a pinned address the space cannot honour (measured on the Metal platform:
    /// `newBufferWithLength:options:` of `u64::MAX`, and a no-copy buffer of length 0 or 2^46,
    /// answer nil).
    BufferLength,
    /// What a library lacks: `newFunctionWithName:` of a name the library does not hold.
    MissingFunction,
    /// Memory the device cannot allocate.
    Memory,
    /// No memory for a shared texture: one of storage other than `Private` is an [`Error::Nil`].
    StorageMode,
    /// A heap's size or memory: a size of 0, no memory for it, or a pinned address the space
    /// cannot honour (measured on the Metal platform: a heap of size 0 is nil).
    Heap,
    /// Where a resource is placed in a heap: an offset off the resource's alignment, bytes past
    /// the heap's end, or no memory for it (measured on the Metal platform: a buffer at an offset
    /// off 256 bytes, and a buffer or texture past the heap's end, are nil).
    Placed,
}

impl Nil {
    /// Whether Metal answers nil where the runtime refuses with `e`.
    fn covers(self, e: &Error) -> bool {
        match self {
            Nil::Never => false,
            Nil::BufferLength => matches!(e, Error::OutOfMemory(_) | Error::PlacementConflict(_)),
            Nil::MissingFunction => matches!(
                e,
                Error::Compile(metal2vulkan::error::Error::InvalidDescriptor(_))
            ),
            Nil::Memory => matches!(e, Error::OutOfMemory(_)),
            Nil::StorageMode => matches!(e, Error::OutOfMemory(_)),
            Nil::Heap => matches!(e, Error::OutOfMemory(_) | Error::PlacementConflict(_)),
            Nil::Placed => matches!(e, Error::OutOfMemory(_)),
        }
    }
}

/// Runs `call`, the runtime's form of `sel`: `policy` decides a refusal Metal's validation stops,
/// and any other refusal dies at the send ([`decide`]).
pub fn send<T>(
    sel: &'static str,
    policy: Policy,
    call: impl FnOnce() -> Result<T, Error>,
) -> RaiseResult<T> {
    inside(Some(sel), call).or_else(|e| decide(sel, policy, e))
}

/// [`send`], the line of a death or a raise also carrying what `describe` says of the call
/// ([`decide_describing`]), asked only where the runtime refuses it.
pub fn send_describing<T>(
    sel: &'static str,
    policy: Policy,
    describe: impl FnOnce() -> String,
    call: impl FnOnce() -> Result<T, Error>,
) -> RaiseResult<T> {
    inside(Some(sel), call).or_else(|e| decide_describing(sel, policy, e, Some(&describe())))
}

/// Runs `call`, the runtime's form of `sel`, whose every refusal dies at the call.
pub fn get<T>(sel: &'static str, call: impl FnOnce() -> Result<T, Error>) -> T {
    get_describing(sel, None, call)
}

/// [`get`], the line of a death also carrying `descriptor` ([`decide_describing`]).
pub fn get_describing<T>(
    sel: &'static str,
    descriptor: Option<&str>,
    call: impl FnOnce() -> Result<T, Error>,
) -> T {
    got(sel, descriptor, call)
}

/// [`get_describing`]'s call.
fn got<T>(
    sel: &'static str,
    descriptor: Option<&str>,
    call: impl FnOnce() -> Result<T, Error>,
) -> T {
    inside(Some(sel), call).unwrap_or_else(|e| {
        match decide_describing::<T>(sel, Policy::Fatal, e, descriptor) {
            Ok(v) => v,
            Err(r) => r.fatal(),
        }
    })
}

/// Runs `call`, a runtime call that cannot refuse and stands for no selector: a degradation of
/// [`DIES`] announced inside it dies at the kind's code. A call that stands for a selector is
/// [`plain_at`]'s.
pub fn plain<T>(call: impl FnOnce() -> T) -> T {
    inside(None, call)
}

/// Runs `call`, the runtime's form of `sel`, whose result the caller decides: a degradation of
/// [`DIES`] announced inside it dies at `sel`, as one inside [`send`]'s call does.
pub fn plain_at<T>(sel: &'static str, call: impl FnOnce() -> T) -> T {
    inside(Some(sel), call)
}

/// Runs `call`, a creator standing for `sel`: `None` for an [`Error::Nil`] and for the refusals
/// `nil` names, as Metal answers nil, and for a gap the creator relays ([`relay_gap`]); the rest
/// as [`send`] decides them.
pub fn create<T>(
    sel: &'static str,
    policy: Policy,
    nil: Nil,
    call: impl FnOnce() -> Result<T, Error>,
) -> RaiseResult<Option<T>> {
    match inside(Some(sel), call) {
        Ok(v) => Ok(Some(v)),
        Err(Error::Nil(_)) => Ok(None),
        Err(e) if nil.covers(&e) => Ok(None),
        Err(e) => created_or_decided(sel, policy, e),
    }
}

/// A creator's answer at `sel` to the runtime's refusal `e`, of a creator for which Metal answers
/// nil for none of the runtime's refusals: `None` for a gap the creator relays ([`relay_gap`]),
/// the rest as [`send`] decides them.
pub fn created_or_decided<T>(
    sel: &'static str,
    policy: Policy,
    e: Error,
) -> RaiseResult<Option<T>> {
    match relay_gap(sel, &e, None) {
        true => Ok(None),
        false => decide(sel, policy, e),
    }
}

/// The degradations whose cost reaches what the guest sees, which die ([`notice`]), each
/// announced at every occurrence:
/// - `ReleasedCommandState`: an execution of an indirect command whose record the GPU wrote
///   naming a state of its kind the program released whose program the buffer does not keep
///   (released before any command buffer that may write the buffer's records was encoded, or
///   past its ID's last four holders), which runs nothing where Metal runs the released state's
///   program; and, at a host step of records the GPU may write, one naming the ID of the state
///   the CPU set, released, where the GPU could write the records while another state held the
///   ID, which runs the set state's program where Metal runs the other's for the GPU's write.
///   Announced by the execution alone: at `commit` for records the CPU
///   wrote, a pass whose pipelines were still being made then included, which `commit` waits for
///   before it records the pass; and for records the GPU may write or a range a buffer holds, at
///   the host step, which runs on the runtime's scheduler thread alone, outside every call here:
///   it dies there and ends the process, never at `commit` (a pass one of whose pipelines failed
///   to be made goes to the host step too, which fails the command buffer);
/// - `TwoPlaneTextureCopy` and `TwoPlaneBufferBlit`: a blit of the two-plane XR10 formats whose
///   answer is not Metal's measured one ([`crate::gpu::encode_blit`]);
/// - `YCbCrFilterNotExact`: a pipeline's filtered sample of a texture slot bound to a two-plane
///   YCbCr texture (any of the 24 formats of [`crate::gpu::ycbcr`]'s family) that does not take
///   Metal's exact filter of the format, which reads off Metal's grid (of 1/2048 for a converted
///   format, of 1/8192 for an `_sRGB` twin), naming the slot and the format: the runtime's generic
///   sample filter notes (`HalfSampleFilter`, `FloatSampleFilter`) cover no slot bound to a YCbCr
///   texture. A slot of a `YCBCR8_420_2P` (500) or `YCBCR10_420_2P` (505) texture whose planes the
///   runtime reads in place (on a device that reads host memory with no bus between, MoltenVK
///   among them: the runtime's `YCbCrInPlace`, its samples and gathers alike) takes Metal's exact
///   filter and taps, and announces none of this kind. On MoltenVK, under the runtime's default
///   `ycbcr_filter`, every other such sample: Metal's
///   own sampler filters the slot's converted texels (an `RGBA16Float` view on the grid of
///   1/2048, an `RGBA32Float` one on 1/8192), which hold Metal's reads exactly, at Metal's weights
///   but without the format's truncated sum on its grid, within one step of the grid of Metal's
///   answer into floats and 1/2048 into halves (`ycbcr::sampling`'s
///   `the_default_filter_samples_within_its_bound_of_metals`), announced at each draw or dispatch
///   that binds such a texture, through a runtime sampler whatever its state. A native device
///   makes such a texture's texels by its own sampler's Y′CbCr conversion where it converts the
///   format, announced where the texture is made, and filters it as its own sampler does
///   (`DeviceSampling`).
///   Under the default a device that samples natively filters such a slot itself, under
///   `DeviceSampling`, and announces none of this kind.
///   Announced at each dispatch, tile dispatch and draw that binds the pipeline's variant, on the
///   encoding thread, inside its call, which dies at its selector: not where the variant is made,
///   which may be the runtime's pipeline-making thread, off every call here. An indirect command's
///   is announced where its execution records it: at `commit`, or at the host step on the
///   scheduler's thread alone, which dies there and ends the process, as `ReleasedCommandState`
///   does. A variant made where the command does not know its formats (an execution set's,
///   `VK_EXT_device_generated_commands`, which MoltenVK lacks) binds no texture slot, so its
///   samples are none of these: a texture it reaches through handles is refused
///   (`MappedTextureSampling`, [`refusal`]);
/// - `RotatedBlendRounding`: a pipeline's blend into an `ARGB8Unorm` attachment in none of the
///   forms measured to blend as Metal's, or in one of them under a fragment output that is not
///   32-bit float, which may round a code apart from Metal's (the runtime's
///   `PrecisionNote::RotatedBlendRounding`). Announced at each draw that binds the variant, on the
///   encoding thread, inside its call, which dies at its selector, as `YCbCrFilterNotExact` is.
///   It dies on every device, a native one included: Apple's blend arithmetic for those forms is
///   not measured, and the exact form is a measured one or the blend in the function;
/// - `EncodedUseRebound`: a command buffer's command named a buffer or heap over the guest's
///   pages that was rebound (a replace, `crate::gpu::backing`) after the command was encoded and
///   before the command buffer's first submission, in an encoder the rebind could not encode again
///   (a compute encoder of the runtime's command predication that shares its unit, holds a loop
///   or leaves work out, which no record of ours reaches, a render pass that executes an indirect
///   command buffer, which the runtime records later, or whose end captured its draws' tile
///   reach, a call that answers otherwise than at its encoding; every other compute, blit and
///   render encoder, one with a pipeline still being made at the commit among them, it encodes
///   again over the new pages, as Metal runs it, a call refused at its
///   encoding refused again and the calls after it running, and a resource state encoder takes no
///   pages), and neither the submission check nor the failure the command buffer was committed with
///   (`commit_failed`) failed the submission holding it: the runtime runs nothing of that
///   submission and fails the command buffer, where Metal runs the command over the pages the
///   object lies in as it submits. A command buffer an earlier submission failed already (a use
///   detached behind an event wait, Metal's code 8) keeps that code. A submission the check fails
///   keeps that code and announces nothing. Announced once per device, whatever `announce_each`
///   says, as the submission is made, on the thread that makes it: the runtime's scheduler thread,
///   outside every call here, or a thread inside a later command buffer's `commit`, in which the
///   runtime submits every batch ready then. It dies there and ends the process. The failure it
///   leaves dies where it is first read too (`crate::gpu::command`), so a later one, which the
///   device does not announce again, dies;
/// - `ZeroLayerMapAfterRelease`: a draw in a pass over a rasterization rate map of no layers made
///   after the device released a map of layers, which writes nothing where Metal draws rows of
///   the pass. Announced at the draw, inside its call, which dies at its selector. No draw over a
///   map of no layers reaches the runtime here ([`crate::gpu::rate_map`]);
/// - `RateMapResetUnencoded`: a render pass over a rasterization rate map reset after the pass
///   took it and before its command buffer's submission, in an encoder the runtime could not
///   encode again (an encoder opened while a buffer of the device was detached, or one that left
///   work for a host step), which draws by the map as
///   it took it, where Metal's GPU draws by the map as reset. Announced at the commit, inside its
///   call, or at the reset of a map whose pass's command buffer was committed, inside
///   `resetUsingDescriptor:`, which die at their selectors ([`crate::gpu::rate_map`]);
/// - `HeapTextureBlockCompressedLayout`: a texture made in a heap over the guest's pages of a
///   block-compressed format whose placement the runtime does not measure (the BC, ETC2, EAC,
///   ASTC and PVRTC families are measured, so no format the runtime has), whose blocks the
///   runtime lays out as its own where Metal tiles them. Its reads, writes and copies
///   are Metal's, but a buffer or texture over its bytes meets other bytes than Metal's would.
///   Announced once per texture, inside the call that made it (the heap's
///   `newTextureWithDescriptor:offset:`), which dies at its selector;
/// - `BiasedDepthPastUnitRange`: a draw of triangles in depth clip mode under a depth bias, on a
///   device that enabled `VK_EXT_depth_range_unrestricted` but cannot clamp to `[0, 1]` while it
///   clips, which stores a depth the bias takes past `[0, 1]` unclamped, where Metal stores 0
///   or 1. Announced at the draw, inside its call, which dies at its selector. MoltenVK does not
///   enable the extension, so no draw here reaches it;
/// - the kinds whose answer this host's driver gives as Metal's, which another device's may not,
///   so that no device here announces them, each dying where a device announces it, inside the
///   guest's call that met it (a draw, a pass, an upload, a resource's or the device's creation;
///   a note at each draw or dispatch that binds the pipeline's variant), at its selector:
///   - `UnormDepthBias` and `UnormDepthRounding`, a draw biasing or testing a `Depth16Unorm`
///     depth, or a pass clearing one, which MoltenVK stores as Metal does (the runtime's driver
///     deviations `MetalDepth16BiasedStores` and `MetalDepth16Stores`), where another device
///     biases, converts and rounds the depth by its own rules. Each is announced at the draw or
///     the pass;
///   - `RestrictedDepthUpload` and the note `RestrictedDepthRange`, a `Depth32Float` value outside
///     `[0, 1]` that a device stores clamped, which MoltenVK keeps
///     (`DepthImageKeepsValuesOutsideUnitRange`);
///   - `UnmeasuredResolve`, a resolve past 4 samples, of a texture neither this device nor Metal's
///     here makes;
///   - `MoltenVkArgumentBuffersUnread` and the note
///     `FunctionStorageAssumed`: a driver release, a MoltenVK configuration or a device's private
///     memory the runtime has not measured, where this host's are measured. A device whose
///     driver, GPU and release the runtime has not measured takes no driver fact measured
///     elsewhere, only the exact path, and says so once as it is made in an informational message
///     (`metal2vulkan-unmeasured-driver`), which is relayed ([`relay`]), no degradation. Function
///     storage has no exact path: on a device whose private memory it has not measured, the
///     runtime takes its floor, 16 KiB a thread, below every size a measured device failed at,
///     and runs a stage within it unnoted. A vertex or fragment function past the floor, up to
///     the 192 KiB a thread Metal was measured to hold, runs all the same and records
///     `FunctionStorageAssumed`, naming its private bytes and the floor, as a driver past what
///     it holds can answer wrongly or hang the GPU with no error (probed: an RTX 5080 Laptop GPU
///     hung at 48 KiB, an RTX 3090 at 192 KiB, an Iris Xe answered wrongly at 48 KiB and a Radeon
///     780M at 128 KiB); a kernel, object or mesh function past the floor is refused as the
///     runtime translates it;
///   - the notes `HalfSampleFilter`, `SrgbTexelDecode` and `FloatSampleFilter`: a sample a
///     device makes otherwise than
///     Metal's texture unit, which MoltenVK's is (the runtime requires
///     `VkProperty::MetalSampler` there instead). A native device relays `HalfSampleFilter`,
///     `SrgbTexelDecode` and `FloatSampleFilter`, its sampling, as the class of a native
///     device's own maths and sampling ([`native::NATIVE_RELAYED`]);
///   - the notes `AnisotropicFilter` and `DeviceAnisotropicFilter`: a sample through a sampler
///     of `maxAnisotropy` above 1 on a device without Metal's own sampler, which takes Metal's
///     anisotropic filter emulated by the device's samples at explicit levels, a near-equivalent
///     whose level and tap spacing are a fitted model's (a 2D
///     texture or 2D array sampled without comparison through a constexpr sampler, or through a
///     runtime sampler slot under a bound sampler that states anisotropy), or, for a shape the
///     emulation does not reach (a cube, a 3D texture, a comparison, a sample whose residency
///     the function reads, a texture of another class, a swizzled storage slot), the device's
///     own anisotropic filter, up to 62 codes off Metal's on an RTX 3090's 2D samples and not
///     measured on those shapes. MoltenVK samples through Metal's own sampler, where the runtime
///     emulates nothing and notes neither. Both die on every device, a native one included:
///     the emulation's level and its tap spacing past two taps are a fitted model, not Metal's
///     rule, and each tap is the device's own filter;
///   - the notes `DeviceInterpolation`, `QuadsSplitByClipping`, `VertexSimdgroupGrouping` and
///     `FragmentSimdgroupGrouping`: the interpolation of every device but MoltenVK, where Metal
///     interpolates (the runtime's `MetalInterpolation`; Vulkan states no precision for
///     interpolation, and RADV, ANV, lavapipe and NVIDIA's driver each rounded most interpolated
///     lanes otherwise than Metal on its probes), llvmpipe's clipping, ANV's vertex subgroups, and
///     the fragment subgroups of a device whose are not Metal's simdgroups;
///   - `NonStrictResidency` and the note `SparseResidency`: a sparse texture's unmapped texels as
///     the device reads them, where MoltenVK binds no sparse memory;
///   - the note `DeviceRayIntersection`: a ray query, which MoltenVK's profile lacks;
/// - the note `ViewBaseTexelOffset`: a sample or gather of an argument buffer's texture with a
///   constant texel offset on an axis that may repeat or mirror, which the runtime places by
///   Metal's rule on the bound view's own base-level size, where Metal reads its parent's: on a
///   view whose first level lies above its parent's base, an axis of size 2 or 4 under a parent
///   whose base-level size is not a power of two keeps the offset Metal drops, and reads one
///   texel or footprint away. A texture slot's access reads the parent's and announces
///   nothing. MoltenVK's own sampler places the offset as Metal's, so there only the forms the
///   runtime emulates on every device announce it: a bicubic filter, and a sample or gather of a
///   texture a heap holds by its raw texels (an extended-range or bfloat format); no guest run on
///   the Mac has announced it. Announced at each draw or dispatch that binds the
///   pipeline's variant, whatever texture it binds, inside its call, which dies at its selector.
///   It dies on MoltenVK: the hardware allows the offset placed by the parent's base level, as a
///   texture slot's is, which the runtime has not built for an argument buffer's texture. A
///   native device relays it, its sampling, as the class of a native device's own maths and
///   sampling ([`native::NATIVE_RELAYED`]);
/// - the depth a draw stores where the runtime's depth state answers otherwise than Metal's,
///   each announced at the draw, inside its call, which dies at its selector (an indirect
///   command's where its execution records it, as `ReleasedCommandState`'s):
///   - `PointGuardBand`: a point whose centre lies past its viewport, within Metal's guard band
///     of 1.5 times the viewport's half-size, where no Vulkan viewport the device's limits take
///     holds the band, which the device culls where Metal draws it;
///   - `NanDepth`: a NaN depth the fragment function writes, which stores the depth clamp
///     range's low end where Metal stores 0, or runs no stencil operation where Metal runs the
///     depth-failure one;
///   - `ClippedDepthPieces` and `SampleDepthClamp`: a primitive past Metal's near or far plane
///     or its guard band, or a biased one in depth clamp mode, whose depths the device clamps at
///     its vertices or its samples, where Metal clamps the pieces it clips the primitive into,
///     or each vertex's depth before the interpolation;
///   - `DepthBiasPastOne`: a depth a bias moves past 1 in depth clamp mode, which stores 1
///     where Metal clamps it to the range's larger depth;
/// - `SignedDepthTest`: a render target's 32-bit float depth that may hold a negative value or a
///   NaN, which Metal's depth test compares by its magnitude, a NaN above +inf, and a pass with
///   depth writes stores without its sign (the runtime's probes `depthkeep` and `depthstore` on
///   the M3 Max under macOS 27.0.1), where Vulkan's compares and keeps the value as it is. A
///   single-sample `Depth32Float` target renders such a depth over its magnitude stand-in, whose
///   tests and stores are Metal's, and announces only the corners: a pass that writes over the
///   stand-in and keeps a stored NaN where a draw wrote exactly +inf, a pass that loads the depth
///   through another image (separate depth and stencil textures' scratch image, a stand-in for
///   custom sample positions), and a write that switches the stand-in on while another command
///   buffer is uncommitted, whose passes encoded before it test and store the value as it is. A
///   multisampled or `Depth32Float_Stencil8` target takes no stand-in: its words are stored
///   without their signs after a pass with depth writes, and its test, which compares the value
///   as it is, is announced at the write that may store such a value. A render pass's NaN clear
///   depth the device cannot write is cleared to 1.0, announced alike. MoltenVK's depth test is
///   Metal's own (the runtime's `DepthTestComparesMagnitudes`), so there only the switch while
///   another command buffer is uncommitted announces it. Announced inside the call that encodes
///   the write or the pass (a blit, `replaceRegion`, a pass's creation or its `endEncoding`),
///   which dies at its selector;
/// - `InPlaceVertexReads` and `FragmentsBeforeGeometry`: a draw's vertex, object or mesh stage
///   that reads in place bytes an earlier fragment stage of its pass wrote, or writes bytes one
///   read or wrote, which the device may run after those fragments, and does across a split of
///   the pass, where Metal's vertex, object and mesh functions ran before the pass's fragments
///   in every run measured. Announced at the draw, inside its call, which dies at its selector;
/// - `AttachmentReadInPlace`: a draw whose functions read or write, other than as an attachment,
///   memory an attachment of its render pass instance renders, where no copy or snapshot the
///   runtime makes serves the access (a texture reached through a handle or an argument buffer, a
///   storage binding of a colour attachment's texture, the depth or stencil attachment's texture,
///   another texture or a buffer over its bytes, a texture whose copy is not served, an
///   execution's or a capture's command): the access sees the attachment as the device holds it
///   inside the pass, which Vulkan leaves undefined, where Metal's read at the draw's own pixel
///   gives the texel from before the pass. The hardware allows the exact form, the runtime's copy
///   or snapshot as for a sampled slot's binding, which it has not built for those routes, so it
///   dies on every device. Announced at the draw, inside its call, which dies at its selector;
/// - `HeapDepthStencilApart` and `HeapTextureHidesAliasedWrites`: a heap's texture whose texels
///   do not alias the heap's bytes as Metal's do: a depth or stencil image a driver lays apart
///   from its memory (MoltenVK, the runtime's `DepthStencilImagesLieApart`), where the runtime
///   keeps no shadow of it in the heap's bytes, or a colour image a driver may compress (ANV's
///   `CompressedImagesHideAliasedWrites`, which no device here has). A write through a buffer
///   or texture over its bytes changes none of its texels, or not all, and its writes change
///   none of those resources' bytes. Announced inside the heap's creator of the texture, which
///   dies at its selector;
/// - `HeapTextureCompressed`: on a driver whose compressed images may keep their texels apart
///   from the memory they are bound to (ANV's `CompressedImagesHideAliasedWrites`), under the
///   runtime's `HeapCompressionPolicy::Compressed`, a heap's colour texture that Metal keeps
///   losslessly compressed too (`allowGPUOptimizedContents`, no `ShaderWrite`, `PixelFormatView`
///   or `ShaderAtomic` usage, 16 texels or more a side), made as the driver's compressed image: a
///   write through a buffer or texture over its bytes may leave its texels as they were, where
///   Metal's texture reads opaque magenta (`0xffff00ff`) in every texel of drawn content the write
///   covered and the written bytes where it held only a clear (an M3 Max), and its own writes may
///   show in those bytes compressed otherwise than Metal's. Every device here takes
///   `HeapCompressionPolicy::Uncompressed` ([`crate::platform`]'s `exact_policy`), which makes its
///   image one the driver keeps uncompressed in the heap's bytes, where the device can (else
///   `HeapTextureHidesAliasedWrites`), so the runtime announces it for none of them. No device on
///   Metal announces it, and a native device does not relay it: a texture's layout is neither
///   maths nor sampling. Announced inside the heap's creator of the texture, which would die at
///   its selector;
/// - a resource a function may reach through its ID or address without the command naming it,
///   which the read would find other than Metal's, each announced where the reach becomes
///   possible, not where a function reads it, inside the call that makes it possible, which
///   dies at its selector:
///   - `UnlistedArgumentSampler`: a live sampler state the device's sampler tables do not map
///     for some commands: one no set-1 sampler array can hold, as more states live than it
///     holds, one at an ID the tables' windows do not reach, one whose Vulkan sampler the device
///     cannot make for an array's use, or a cubic state for a stage that does not test for one (a
///     fragment stage), which a function reading its ID through an argument buffer reads as the
///     neutral sampler, where Metal's reads the state. Announced as the state is made, or as a
///     command is encoded while it lives;
///   - `UnreadableHandleTexture`: a PVRTC or 4:2:2 texture a command that reads texture
///     handles reaches, which reads the neutral texture where Metal decodes it;
///   - `UnnamedTextureUnviewable` and `UnnamedTexturesPastCapacity`: a texture a heap array
///     cannot view, or one past a pipeline's heap arrays, which a function reading its ID
///     unnamed reads as the neutral texture;
///   - `HeapSetCopied`: a heap set a command buffer binds on a device that cannot write a set a
///     pending command buffer binds, whose functions read each ID's texture as it stood at the
///     commit;
///   - `DetachedStorageKeepsWrites`: a storage texture over detached bytes a committed command
///     buffer's heap set reaches, whose unnamed storage reads read what an earlier such write
///     left, where Metal drops those writes and reads zero. Announced at `commit`;
/// - what a pass or a copy leaves in its texture otherwise than Metal, each announced inside the
///   call that encodes it, which dies at its selector:
///   - `MixedDepthStencilResolve`: a pass that stores and resolves one texture's depth beside
///     its stencil resolved alone, which Metal's validation stops and plain Metal answers with a
///     layout pattern of the resolved depth, where the runtime resolves both and keeps both
///     aspects' samples. Announced at the pass's creation;
///   - `UncoveredColorKept`: a draw of a state that lacks a colour attachment of the pass, whose
///     texels it keeps where Metal's compiler may fold the output to a constant that stores zero.
///     Announced at the draw;
///   - `SnormMinimumKept` and `SharedExponentKept`: under the runtime's default `tile_store`, a
///     draw in a pass that loads and stores a single-sampled colour attachment of a signed
///     normalized format or of `RGB9E5Float`, whose texels the pass's draws do not write keep
///     their bytes, where Metal's pass writes each 32 by 32 tile a draw reaches back through its
///     values: a channel's minimum code (-128, or -32768) as the code above it, which reads as the
///     same -1.0, and a shared-exponent word whose largest mantissa is below 256 under an exponent
///     above 0 normalized, of the same value. Only uploaded or copied bytes hold such a code or
///     word. MoltenVK's pass is a Metal pass (the runtime's
///     `TileStoreIsMetals`), so no device on Metal announces either, and a native device does not
///     relay them: a pass's tile store is neither maths nor sampling. Announced at the draw;
///   - `DeviceRoundedResolve`: a pass's resolve of a multisampled colour attachment that the
///     device averages and rounds: every one but an attachment a fetching pass holds in sample
///     planes. A
///     resolved channel is up to one code off Metal's (an sRGB one two) near a rounding boundary,
///     a float channel off in its last bits, and wholly where a sample is a NaN or an infinity,
///     measured at 2 and 4 samples on RADV, ANV, NVIDIA and lavapipe, each rounding an 8-bit unorm
///     tie down. The runtime's notice names the attachment. MoltenVK's average is Metal's own (the
///     runtime's `ColorResolveAverageIsMetals`), so no device on Metal announces it for a resolve
///     the device averages. Also a resolve of an 8-bit UNORM attachment into its format's sRGB
///     twin, which no device's average writes, on a device that renders the twin: the runtime's
///     resolve after the pass decodes the samples and encodes their average by the device's own
///     sRGB conversions, where Metal's resolve takes its 12-bit tables, and announces it on every
///     device, MoltenVK among them (the runtime's `color_resolve.rs`; measured for `R8Unorm` and
///     `RG8Unorm` at 4 samples on the M3 Max under macOS 27.0.1, MoltenVK 1.4.2,
///     `encode_render`'s `resolves_into_twins_are_the_runtimes`). An sRGB attachment's resolve
///     into its UNORM twin averages its codes by the unorm rule and announces nothing. A native
///     device relays it, the device's own rounding of an average and the runtime's default off
///     Apple's GPUs, as the class of a native device's own maths and sampling
///     ([`native::NATIVE_RELAYED`]), so a strict run there, this crate's tests on Linux among them,
///     goes on past it. Announced as the pass ends, inside `endEncoding`;
/// - `CappedAnisotropy`: a sampler state of a `maxAnisotropy` past 16, which Metal samples with
///   32, where the device filters with its largest (16), so a footprint more than 16 times as
///   long as it is wide reads a coarser level. Announced at the state's creation, which dies at
///   its selector;
/// - `NotMipmappedAnisotropicSampler` and the note `NotMipmappedAnisotropicFilter`: on a device
///   that samples through Metal's own sampler (MoltenVK on Apple's GPU), a sampler state, or a
///   constexpr sampler a sample at an implicit level or with gradients goes through, with
///   anisotropy, no mip filter and different minification and magnification filters. Vulkan has
///   no sampler without mipmapping, so the device's sampler takes a nearest mip filter with the
///   level clamp `[0, 0.25]`, which keeps Metal's choice of filter, and Metal's anisotropic filter
///   then reads another footprint than Metal's `NotMipmapped` at `maxAnisotropy` 2 to 16. Metal's
///   footprint for such a sampler is not known as a state MoltenVK takes, so nothing here holds it
///   Metal's, and it dies there; no other device announces either. The state's announced at its
///   creation, which dies at its selector, the note at each draw or dispatch that binds the
///   pipeline's variant, inside its call, which dies at its selector;
/// - `CombinedArchiveStages`: a render pipeline state the runtime makes from binary archives
///   where Metal may refuse it with an archive miss (its two adds' fragment functions reading
///   different vertex outputs). Announced at the state's creation, which dies at its selector;
/// - `ImplementationDefinedReconvergence`: a stage that reads simdgroup state after divergent
///   control flow on a device that does not reconverge maximally, whose simdgroup operations
///   there can see other lanes active than Metal's. Announced at each draw or dispatch that binds
///   the pipeline's variant, inside its call, which dies at its selector. The runtime notes every
///   stage holding a simdgroup or quad operation, a derivative, an implicit-level sample or a
///   barrier on a driver without `VK_KHR_shader_maximal_reconvergence`, which decides which lanes
///   run together after divergence: other sets of lanes than Metal's can be active together, so
///   a ballot or a sum can cover another set of lanes, and a `simd_is_first` election can pick
///   several lanes. A native device relays it, as it does the precision notes below, on the
///   rule for native devices stated there ([`native::NATIVE_RELAYED`]). A fragment stage with
///   simdgroup operations still dies there, of `FragmentSimdgroupGrouping`, which the runtime
///   records on every driver but MoltenVK;
/// - the precision notes of a pipeline's stage whose arithmetic, filtering or ordering can give
///   other bits than Metal's, as the runtime's documentation of each says, which no measurement
///   holds equal here: `F16FmaDoubleRounding` and `F16IntConvDoubleRounding` (a widened half fused
///   multiply-add or half-integer conversion rounding twice), `GradFromBias` (a biased sample
///   emulated through gradients), `ReductionOrder` (a float reduction's association),
///   `DeviceSignedZeroInfNan` (float operations run without `SignedZeroInfNanPreserve`),
///   `TessellationFactorScale` and `TessellationFactorInstance` (a scaled factor's rounding, a
///   per-instance factor's first instance), `Fp8Value` (an 8-bit float converted to an integer),
///   `MultisamplePosition` (a value Metal reads at the pixel's centre, evaluated elsewhere),
///   `CubeGradientFootprint` (a cube sample's level), `QuadsCountedBeforeDiscard` (a simdgroup
///   operation before a discard counting a quad Metal's may not) and `BlendedWriteRounding` (a
///   blended half-float write under `textureWriteRoundingMode` 2). Announced at each draw or
///   dispatch that binds the pipeline's variant, inside its call, which dies at its selector, as
///   `RotatedBlendRounding` is. MoltenVK's placement is as measured on the Mac GPUs: every one
///   dies there. A native device relays those whose difference is the device's own rounding,
///   evaluation position or reduction order, a fact of the device ([`native::NATIVE_RELAYED`],
///   each with its ground): `GradFromBias`, `ReductionOrder`, `DeviceSignedZeroInfNan` and
///   `MultisamplePosition`, and, as the class of the device's own maths and sampling,
///   `F16FmaDoubleRounding`, `F16IntConvDoubleRounding` and `CubeGradientFootprint`, on the rule
///   for native devices: as close to Metal's answer as the hardware allows, a difference that
///   cannot match logged as a degradation, with no fallback to an interpreter for correctness.
///   Every other one dies there too, as no fact of
///   the device decides it: `TessellationFactorInstance` and `Fp8Value` are other values, not
///   other bits of the same arithmetic (the factors of a draw whose base instance is not 0, read
///   at an offset counted from it; an 8-bit float's integer from the format's value, where Metal
///   on an Apple M3 answers its carrier byte's); `QuadsCountedBeforeDiscard` differs where
///   Apple's compiler moves the operation past the discard, which that compiler decides shape by
///   shape; the hardware allows `BlendedWriteRounding` an exact form the runtime has not built
///   (the blend in the function under the pixel interlock); and Metal's rule for
///   `TessellationFactorScale` is not read (whether its fractional partitions take the scaled
///   factor in `float` or rounded up to `half`). On MoltenVK, `CubeGradientFootprint` dies too:
///   a vertex or fragment function's cube sample with explicit gradients on a `y` face under
///   macOS 27, which SPIRV-Cross's gradient form hands Metal with its `z` tangent lost, so the
///   runtime hands Metal's length of the two tangents (an isotropic sample's footprint kept up to
///   one rounding of `64/91`, every lane measured Metal's; an anisotropic one's ellipse laid on
///   the face's diagonal), and an implicit-level cube sample's level, where Metal's rule is not
///   measured. A kernel hands Metal's own gradients there and notes nothing;
/// - the note `UnmeasuredGroupSpeculation`: a stage with a branch whose arms hold a quad or
///   simdgroup operation that Metal's backend makes answer over every lane of the branch's block
///   where it speculates the arms, of a shape no measurement says it speculates (an arm of an
///   instruction, size or shape no probe covers): the branch keeps the AIR's lanes, so the
///   operation may answer over the lanes that take the arm where Metal's answers over the
///   block's. It dies on every device, a native one included, as `QuadsCountedBeforeDiscard`
///   does: where Apple's compiler speculates decides the answer, not the device. Announced at each
///   draw or dispatch that binds the pipeline's variant, inside its call, which dies at its
///   selector;
/// - the note `DeviceSampleLevel`: on a native device, a sample whose level the device's level
///   arithmetic chooses (a fragment function's implicit-level sample, its bias within the device's limit; a
///   sample with explicit gradients; a fragment function's LOD query; an explicit level Metal's
///   level rule is not handed on, such as an integer texture's), of a texture other than a cube,
///   without comparison, where no sample-filter note covers it: Vulkan's level of detail may
///   approximate its scale factor and rounds a nearest mip level otherwise than Metal, so near a
///   boundary the sample may read another level or filter than Metal's, and the query answers the
///   device's level. MoltenVK samples through Metal's own sampler, and the default's native
///   sampling records `DeviceSampling` in its place, so neither announces it. A native device
///   relays it, the device's level arithmetic, as the class of a native device's own maths and
///   sampling ([`native::NATIVE_RELAYED`]). Announced at each draw or dispatch that binds the
///   pipeline's variant, inside its call, which dies at its selector;
/// - the note `CoverageReadsRoundedAlpha`: a render pipeline that covers by alpha under
///   `textureWriteRoundingMode` 2, whose fragment function writes a `float` alpha to a half-float
///   attachment 0 that stores it as written: the function rounds the alpha to the nearest-even
///   `half` Metal stores, and the coverage reads the rounded alpha where Metal's reads the alpha
///   before rounding, so a fragment whose alpha lies within half a `half` step of a coverage
///   threshold may cover one sample more or fewer. Announced at each draw that binds the
///   pipeline's variant, inside its call, which dies at its selector;
/// - the note `UnmeasuredFetchBlend`: on a device that fetches colour attachments through the
///   pixel interlock's storage form (MoltenVK, RADV and ANV, the runtime's statement),
///   the blend of an attachment the fragment function reads, of a format whose Metal blend
///   arithmetic the runtime has not built (a 32-bit float, 16-bit normalized, 8-bit signed
///   normalized, 10-bit, packed float or `A8Unorm` format, among others, an 8-bit unorm one
///   written by fewer than four lanes, or a 16-bit float or 8-bit unorm one under a blend of the
///   second source): the shader's generic blend, in `float`, each step rounded, the sources and
///   the blend colour clamped to a normalized format's range, the device's conversion storing the
///   result; or of a four-channel 8-bit `_sRGB` format under a blend of the second source in a
///   module compiled with fast math, in the 8-bit unorm blend's rounded forms over the decoded
///   texel, where Metal's blend for these is not measured (a four-channel 8-bit `_sRGB` one's
///   other blends are Metal's, exact).
///   Announced at each draw that binds the pipeline's variant, inside its call, which dies at its
///   selector;
/// - the note `LinkedSamplerNumbering`: the constexpr sampler IDs of a stage that links visible
///   functions, where the runtime does not number them as Metal does. It notes a stage that
///   links two or more with constexpr samplers, whose order Metal itself does not decide (two
///   machines numbered one pipeline in different orders), a dynamic library's, a binary
///   function's or a function-pointer group's, which nothing measured, and a function a table
///   can hold whose samplers the runtime cannot read, which Metal numbers and the runtime does
///   not, so every sampler state made while the state lives may answer another `gpuResourceID`.
///   One linked function alone, of `functions` or `privateFunctions`, called or not, is
///   numbered as Metal numbers it and noted for nothing. One note
///   covers them all, so it dies. No guest form here links a function: no serialized key
///   carries one, and a dynamic library's import is refused. Announced at each draw or dispatch
///   that binds the pipeline's variant, inside its call, which dies at its selector;
/// - the note `PixelSamplerScaling`: a stage that samples through a pixel-coordinate sampler where
///   none of the runtime's exact paths reaches and the device filters a coordinate the runtime
///   scaled: a linear sample or gather through a sampler state bound at a slot, a linear sample of
///   an array or 3D texture, a comparison, a level query, or a sample that reads a sparse residency
///   status. A constexpr pixel sampler's sample or gather of a 1D or 2D texture reads through the
///   device's own unnormalized-coordinate sampler, and every nearest pixel sample reads at its
///   texel's centre, so both read Metal's texel and note nothing (the runtime's matrix of texel
///   edges and centres: no difference from an M3 Max's Metal on an RTX 5080, on MoltenVK or on
///   lavapipe). Announced at each draw or dispatch that binds the pipeline's variant (a slot's
///   where the state bound has pixel coordinates), inside its call, which dies at its selector;
/// - the note `ArgumentPixelSamplerScaling`: a stage that samples through an argument buffer's
///   sampler, which may hand the device a coordinate the runtime scaled from pixels, where the
///   state the buffer names has pixel coordinates. A nearest state reads Metal's texel at its
///   centre, but the runtime announces the note for it too, as the shader cannot tell its states
///   apart. Announced at each draw, dispatch, tile dispatch or execution of indirect commands of
///   the pipeline while the device holds a live sampler state of pixel coordinates, inside its
///   call, which dies at its selector;
/// - the note `DiscardDropsSimdgroupStores`: on MoltenVK on an Apple GPU, a fragment function
///   that samples, and after a discard computes a floating-point value and writes storage,
///   where Apple's compiler drops the texture writes of every live thread of a SIMD group holding
///   a discarded thread, whole 8 by 4 blocks of a storage-form fragment's colour writes. The
///   runtime waits with such a discard until after the writes, which announces nothing; it
///   keeps the discard and announces the note where a SIMD-group operation follows
///   it, so the function's storage writes may be lost. A store, atomic or image write that
///   follows a discard it keeps otherwise runs only where the thread was not discarded,
///   announcing nothing. No VM boot or suite run is known to make such a function. Announced at
///   each draw or dispatch that binds the pipeline's variant (the runtime announces a note of a
///   kind in `announce_each` there, not where the variant is made), inside its call, which dies at
///   its selector;
/// - `FreshPatternZeroed`: a new `Private` texture Metal keeps losslessly compressed, whose
///   fixed pattern the device cannot write exactly (a NaN depth a copy may store clamped, a
///   multisampled `Depth32Float` one's NaN, a NaN colour of a multisampled image of a format
///   the device makes no integer scratch image of, or an image of other bytes than Metal's
///   texel), which reads zero where Metal reads the pattern. The runtime copies a multisampled
///   NaN pattern from its integer scratch through a view of the image's format, so MoltenVK's
///   copies hold it (`rgbx`'s `new_textures_read_as_metals`). No VM boot or suite run is known
///   to read such a texture before writing it, so nothing grounds a relay. Announced where a
///   command first reads a texel of that content that no clear, resolve or blit wrote (a pass that
///   loads it, a blit that copies from it, a shader that reads it), inside its call, which dies at
///   its selector. The runtime counts what writes the texels by box and aspect: a blit that writes
///   a whole level, blits of parts that together cover one (an upload in pieces), a resolve into
///   the texture, separate depth and stencil attachments and a stand-in's own texture, and a pass's
///   load or a blit reads only its region; a copy of one aspect of a `Depth32Float_Stencil8`
///   texture writes that aspect alone. So a guest that copies a buffer into a new multisampled
///   `Depth32Float` target and reads it with no clearing pass between reads Metal's texels
///   (`encode_blit`'s `multisampled_depth_and_stencil_buffer_copies_are_the_metal_platforms`). A
///   pass's clear counts as a write, as the pass ends, only where it reaches the texture: an
///   attachment the pass renders itself, or one it renders through a stand-in or a separate
///   depth or stencil scratch that it stores. A new separate depth or stencil texture, or a
///   stand-in's own, that a pass cleared through its scratch and stored `DontCare` keeps its
///   fresh pattern, as Metal's keeps its texels there, and a later read of it announces the kind
///   and dies;
/// - the note `ExtendedRangeFloatBlendSource`: a fragment function's `float` colour output
///   blended into an extended-range attachment where Metal may round the blend's source otherwise
///   than the runtime's model: under fast math a lane of the output that is not a load, or a
///   module that names no math mode (a note of its own, beside the `half` output's
///   `ExtendedRangeBlendSource`). A computed `float4` under fast math misses Metal's by
///   a code of a channel under 17 of the 66 blended configurations `encode_render`'s sweeps draw
///   on this host's M3 Max, and no VM boot or suite run is known to make such a state, so nothing
///   grounds a relay. Announced at each draw that binds the state's variant, inside its call,
///   which dies at its selector; the state is made (`encode_render`'s
///   `float4_extended_range_blends_die_naming_their_source`);
/// - the note `ExtendedRangeBlendSource`, the same for a `half` colour output: Metal's compiler
///   folds the blend into the function, so its rounding follows how the function computed the
///   output, where the runtime rounds each product to `half`. A module
///   compiled without fast math, and an output each of whose lanes is a load under fast math,
///   announce nothing and blend as Metal's; a computed `half4` under fast math, or a module that
///   names no math mode, may store a channel a code apart from Metal's. The function can take
///   Metal's rounding once the runtime reads how Apple's compiler folds the blend, so nothing in
///   the hardware forces the difference, and the blend is not in the function's own arithmetic,
///   whose fast-math rewrites `FastMathRewrites` covers. It dies on every device. Announced at
///   each draw that binds the state's variant, inside its call, which dies at its selector;
/// - `PrivateResourceOption`: a resource made with `MTLResourceOptions` bit 24, a private option
///   Metal takes, which the runtime makes as without it: on Metal a no-copy buffer made with it
///   keeps the GPU's writes from the CPU (the runtime's probe `nc_bit24`), where the runtime's
///   reach the CPU. On macOS the runtime imports a no-copy buffer's or heap's pages of the option
///   through a read-only view of them, which drops the GPU's writes as Metal's does, and
///   announces nothing there; it announces a heap of its own memory, an indirect command buffer
///   and, off macOS, the no-copy ones. A device can hold the GPU's writes in memory of its own,
///   apart from the guest's pages, so nothing in the hardware forces the difference, and it dies
///   on every device. Announced where the resource is made, inside the creator, which dies at its
///   selector;
/// - `StaleDetachedReach`: a command buffer that bound a heap set, whose commands may reach a
///   texture through its ID without naming it, submitted while the texture's bytes were
///   detached after a submitted command buffer made them resident: a read that does not name
///   it reads zeros, where Metal's reads the pages the guest gave back. Those pages hold what
///   they held until the guest writes them again, so Metal's answer is one value, and no fact of
///   the hardware forces the difference, so it dies on every device. Announced once per device,
///   at the first such submission, inside its call, which dies at its selector;
/// - `CrossClassTextureNeutral`: a draw or dispatch binding, at a texture slot declared of
///   another numeric class (`float` or `half`, `int`, `uint`) than its texture's format, one the
///   runtime has no exact answer for (a storage slot, a texture buffer, a multisampled, sparse,
///   depth, extended-range or YCbCr texture, or a read of a `short`, `ushort` or `ulong`
///   result), whose slot reads its neutral texture, zeros, where Metal reads the texture in its
///   format's own class and hands back its words as the declared type. A sampled colour texture
///   read with `float`, `half` or 32-bit integer results, at a slot or held by an argument buffer,
///   the runtime answers as Metal does and announces nothing (`rgbx`'s
///   `cross_class_samples_are_metals`, `mistyped_reads_are_metals`). Only a guest
///   shader that mistypes a texture reaches it, and no boot or suite case is known to make one,
///   so nothing grounds a relay. Announced at the draw or dispatch that binds it, inside its
///   call, which dies at its selector;
/// - `IndirectNilBufferOffset`: an execution of indirect commands the runtime replays on the
///   host, of a command that holds a nil buffer set at an offset whose low 32 bits are not 0,
///   which Metal's function sees as the buffer's address, where the replay binds no buffer, a
///   null address (an execution the GPU replays passes the bits, as Metal does). Only the address
///   differs, but a function may read through it or keep it, and no boot or suite case is known to
///   set such a buffer, so nothing grounds a relay. Announced where the runtime replays the
///   execution, for records the GPU may write or a range a buffer holds: at the host step, on its
///   scheduler thread alone, outside every call here, which dies there and ends the process, as
///   `ReleasedCommandState` does;
/// - `CompressedTexelDecode`: a texture of a BC1 to BC5 format (the sRGB forms too) on a device
///   whose driver is not measured to decode their blocks as Metal does, which keeps no decoded
///   twin of Metal's texels: one over the guest's pages in a heap, laid out apart from a heap's
///   bytes, memoryless or sparse, over a surface, a view of another decode than its texture's
///   twin or a shadowed view, or one whose twin's format the device cannot sample, filter and
///   copy into. Its reads, samples and gathers answer the device's decode,
///   where Metal interpolates with 6-bit weights (NVIDIA 615.71.09: up to 3/255 off for BC1 to
///   BC3, 7.3/255 for BC4 and BC5). MoltenVK samples the blocks through Metal itself, and the
///   runtime holds its decode as Metal's on every Apple GPU, so no device on Metal announces it.
///   A texture with a twin reads Metal's texels and announces `decoded-block-twin`, a
///   performance kind, relayed, and `cpu-block-decode`, another,
///   where a CPU copy into it (`replaceRegion:`) while a command buffer that may read it is
///   submitted and not seen complete is decoded on the CPU into the twin before the call
///   returns, so that the command buffer samples the copied texels, as Metal's does
///   (`WorkCounts::cpu_block_decodes` counts them). The hardware allows an exact answer, a decode
///   of the raw blocks in the shader by Metal's rules, which the runtime has not built; a native
///   device relays the device's decode, its sampling, as the class of a native device's own
///   maths and sampling ([`native::NATIVE_RELAYED`]). Announced where the texture or view is made,
///   inside its creator's call, which dies at its selector elsewhere;
/// - `DecodedTexelPrecision`: under the runtime's default `decoded_texels`, a texture of an ASTC
///   LDR format (not an sRGB one) on a device without that ASTC format, or of an EAC 11-bit
///   format on one without ETC2, which the runtime holds in a carrier, its reads, samples and
///   gathers reading a decoded twin of compact codes: an ASTC channel the 8-bit code nearest
///   Metal's 12-bit value, at most 1/510 off it, an EAC 11-bit one the 16-bit normalized code
///   nearest Metal's value, at most half a 16-bit step off it. Apple's GPUs have the formats, so no device on
///   Metal announces it; a native device relays it, the runtime's default sampling off Apple's
///   GPUs, as the class of a native device's own maths and sampling
///   ([`native::NATIVE_RELAYED`]). Announced where the texture is made, inside its creator's
///   call, which dies at its selector elsewhere;
/// - `HeapTextureLargerThanMetal`: a texture sized or placed in a heap whose image on the device
///   takes more bytes or a coarser alignment than the runtime's model of Metal's answer, not one
///   `OptimizedDepthStencilTexture` covers (on NVIDIA 615.71.09, a 4x4
///   `Depth32Float` texture takes 1024 bytes, where Metal answers 128). The runtime answers the
///   device's size and alignment, so the guest's heap size query differs from Metal's; the range
///   of Metal's space a texture of memory of its own takes is of the runtime's model of Metal's
///   answer ([`crate::gpu::metal_space::own_heap_size`]), which asks no image, so it is Metal's.
///   A native device relays it, on the hardware's ground: the device's memory granularity decides
///   it, and the guest sizes and places its heaps by the same answers, so a heap it sizes by them
///   is larger, and the buffers and heaps made after it can lie at other gpuAddresses than on Metal
///   ([`native::NATIVE_RELAYED`]). Announced
///   inside the call that sizes or places the texture (`heapTextureSizeAndAlignWithDescriptor:`,
///   a heap's `newTextureWithDescriptor:offset:`, and `newTextureWithDescriptor:` of a texture
///   the model has no answer for), which dies at its selector;
/// - `HostDeallocatorDelayed`: a no-copy buffer or heap released with a deallocator while a
///   command buffer that may still reach its pages holds them, whose deallocator the runtime runs
///   once that command buffer lets go of them, where Metal's runs at the release (and the command
///   buffer's later accesses reach nothing: the runtime's probe `dealloc-probe4` on the M3 Max
///   under macOS 27.0.1), as Vulkan frees no memory that submitted commands may still use. Core's
///   deallocator is what the guest and the host see of the release (core's mapping's release, an
///   IOSurface's), so its moment is Metal's. The runtime delays it only for a release with no
///   keep-alive: with one, it runs the deallocator at the release, as Metal's, and the import
///   keeps the keep-alive, which keeps the pages mapped, until it is freed, wherever they are held
///   (an image over them holding their import, a live buffers' table that lists them, MoltenVK's
///   freed memory waiting for the batches in flight), and announces nothing. Every no-copy buffer,
///   heap and rebind this platform makes gives one ([`crate::gpu::backing::keep_alive`]), so the
///   runtime announces it for none of them. Announced once per device, on the thread that lets go
///   of the buffer, inside its release here or on one of the runtime's as a command buffer
///   completes, outside every call here: it dies where it is announced;
/// - `GroupedCompletion`: on a driver whose queue merges a submission that signals nothing into the
///   next one (Mesa's: ANV, RADV, lavapipe), under the runtime's `CompletionPolicy::Grouped`, up
///   to four batches of command buffers of one queue that its scheduler found ready together, or
///   held while two or more earlier submissions of their queue ran, went to the driver in one
///   submission, only the last signalling: each command buffer of it but the last is seen complete
///   (its status, its completed handlers, which core's completion of the guest's work runs in, and
///   its waits), and the event values it signals are reached, once the last completed, late by the
///   GPU time of the batches after it, at most three, never early and in commit order, where
///   Metal's host sees each complete at its own end. Every device here takes
///   `CompletionPolicy::Exact` ([`crate::platform`]'s `exact_policy`), which submits each batch
///   alone, so the runtime announces it for none of them, and a strict run on Mesa's drivers, this
///   crate's tests among them, never dies of it. MoltenVK merges no submissions, and no fact of
///   the hardware forces it, so a native device does not relay it. Announced once per device
///   whatever `announce_each` asks, on the runtime's scheduler thread, outside every call here: it
///   would die where it is announced;
/// - `DrawDroppedStillMaking` and `DispatchDroppedStillMaking`: under the runtime's
///   `PipelineWaitPolicy::Bounded`, its default, a draw, compute dispatch or tile dispatch whose
///   `VkPipeline` the driver was still making 4.5 s after its command buffer's commit ran nothing,
///   the command buffer running on, where Metal's draw renders and its dispatch writes: a cold
///   compile (seconds on MoltenVK with a cold Metal compiler service) then costs that command's
///   work, not a stall past the guest's 10 s check of its FIFO. A strict process takes
///   `PipelineWaitPolicy::Exact` ([`crate::platform`]'s device options), which waits until the
///   pipeline is made, so a strict run, this crate's tests among them, never dies of a slow
///   compile. Announced where the command is recorded: in the commit, or the command buffer's
///   host step;
/// - `DrawDroppedUnmade`: a draw whose `VkPipeline` the driver failed to make after the draw
///   returned ran nothing, the command buffer running on, where Metal's draw renders, and Metal
///   fails the state's creation instead. Announced where the draw is recorded: in the commit, or
///   the command buffer's host step;
/// - six kinds no device on Metal announces, each dying where a device announces it, at its
///   selector, as no fact of the hardware forces it:
///   - `VisibilityCountPerColor`: a draw counted by a `Counting` visibility result whose
///     fragment variant writes more than one colour attachment, on a device whose driver counts
///     each sample once per colour output (the runtime's `OcclusionCountPerColorOutput`: ANV),
///     whose result is Metal's count multiplied by the colours the draw writes; a `Boolean`
///     result is Metal's. Announced at the draw, inside its call, or where the runtime records an
///     execution's command;
///   - the note `ColorAttachmentGroups`: a render pipeline whose colour attachments the device
///     draws in more than one group, on a device with fewer colour attachments than the
///     pipeline's: each draw runs once per group, each group's variant over its own copy of the
///     pass's depth and stencil, so groups past the first may differ at discard or depth edges if
///     the driver compiles the variants differently, the runtime's recorded bound. Announced at
///     each draw that binds the pipeline's variant, inside its call;
///   - the note `DevicePointSize`: a vertex function's `[[point_size]]`, or a mesh function's
///     `set_point_size`, on a device whose `pointSizeRange` ends below Metal's largest point,
///     511 pixels (255 on ANV, 256 on lavapipe), which draws a point between its bound and 511
///     pixels smaller than Metal's. The device's limit is on its point primitives alone: drawing
///     such a point as two triangles is the exact form, which the runtime has not built.
///     Announced at each draw that binds the pipeline's variant, inside its call;
///   - the note `ClippedCoverageSplit`: a multisampled pipeline's fragment function that reads
///     `[[sample_mask]]` on a device whose rasterizer is not Apple's, which hands a pixel's
///     fragment the whole primitive's coverage where Apple's GPU runs it once per piece its
///     clipper splits a triangle past the viewport into, each reading that piece's coverage.
///     Announced at each draw that binds the pipeline's variant, inside
///     its call;
///   - the note `TessellationTriangleOrder`: a tessellation pipeline drawn through the
///     device's tessellation stages, on a device whose tessellator is not Apple's (every one but
///     MoltenVK on an Apple GPU), which orders each patch's triangles, picks their first corners
///     and joins its rings where Vulkan leaves that to the implementation, not as Metal's
///     tessellator does. The domain points are Metal's; flat varyings, interpolation across
///     another triangulation and the order a patch's triangles blend in may differ. The
///     runtime's emulation, which it runs on a device without `tessellationShader`, draws
///     Metal's order: the exact form, which the hardware allows. Announced at each draw that
///     binds the pipeline's variant, inside its call;
/// - the note `DeviceNanEncoding`: a NaN whose bits the device decides, where Metal answers its
///   own: the canonical `0x7fc00000` for its float arithmetic and functions, a 16-bit float
///   texel's own NaN for its read, sample, gather or pixel fetch and the canonical one for a
///   32-bit texel's, and the half's own bits for a `half` written to a 16-bit float texel.
///   MoltenVK answers the canonical NaN for a 16-bit texel's read and a 32-bit texel's own. A
///   written `half` keeps its bits through a 16-bit float texel's unsigned integer view, but
///   where a heap's handle reaches a texture that view cannot hold (a
///   swizzled one, or a wrapped image made without `MUTABLE_FORMAT`) and in a packed small float
///   texel (`RG11B10Float`, `RGB9E5Float`), where the device stores its own conversion of the
///   NaN; another device may answer another sign or payload for its arithmetic (llvmpipe:
///   `0xffc00000`). A `half` colour output to a half-float channel reaches the attachment through
///   the device's widening of the `half`, whose NaN is the device's, on every device (NVIDIA's
///   615.71.09 on an RTX 5080 Laptop GPU stores each as `0x7fff`, ANV and RADV as `0xfe00`, and
///   MoltenVK canonicalizes it), where Metal stores the half's own bits; an output no return of
///   which can hold a NaN announces nothing. The hardware allows Metal's bits, the function
///   selecting them itself, as the runtime does for its own functions and a 32-bit texel's write,
///   so nothing in the hardware forces the difference, and it dies on MoltenVK; a native device
///   relays it, its NaNs' bits, as the class of a native device's own maths and sampling
///   ([`native::NATIVE_RELAYED`]): there the runtime gives the device's NaN, its native form, and
///   builds no path to Metal's bits where the device has one, and only a NaN's bits differ.
///   Announced at each draw or dispatch that binds the pipeline's variant (where it binds a
///   texture whose format may hold a NaN), inside its call, which dies at its selector. Also
///   a GPU blit into, out of or between textures of which one is an `RG11B10Float` texture Metal
///   keeps losslessly compressed: Metal's blit through the compression makes a NaN field the
///   quiet NaN, where the device's copy keeps the texels' bytes, as MoltenVK's does where the
///   texture's image takes `STORAGE` usage or `MUTABLE_FORMAT`, which Metal keeps uncompressed
///   (the runtime's `compressed_blit.rs` and `CompressedImageCopiesRewriteTexels`; measured on
///   the M3 Max under macOS 27.0.1, MoltenVK 1.4.2, `encode_render`'s
///   `drawn_nan_and_infinite_samples_resolve_as_metals`). Announced inside the blit's call, which
///   dies at its selector;
/// - the note `DeviceArithmetic`: a `float` division, remainder, square root, fused multiply-add
///   or dot product at Vulkan's precision bounds (a division within 2.5 ulp, an `Fma` that may run
///   unfused) where Metal answers the correctly rounded value, and a fast division's reciprocal,
///   which Metal rounds within about 0.64 ulp, as the device's or as `RN(1 / y)`, an ulp apart
///   near rounding midpoints. The device's integer and `float`
///   operations can compute Metal's values in the function, as the runtime already does for
///   `sqrt` and `fmod`, so nothing in the hardware forces the difference, and it dies on MoltenVK;
///   a native device relays it, its maths, as the class of a native device's own maths and
///   sampling ([`native::NATIVE_RELAYED`]). Not announced where the device answers as Metal's own
///   function (MoltenVK, under fast math). Announced at each draw or dispatch that binds the
///   pipeline's variant, inside its call, which dies at its selector;
/// - the note `DeviceDenormals`: an `f32` denormal the device keeps or flushes by its own float
///   controls, where Metal reads a denormal operand of its float arithmetic, comparisons and
///   functions as the zero of its sign and flushes a denormal result to one: at a remainder, a
///   conversion to or from `double`, `dot`'s fused sum and a function the translation does not
///   compute itself. An exponent-field select gives Metal's reading on every device, as the runtime
///   already does for `+ - *`, the comparisons, the fused multiply-adds, `ldexp`, `sqrt`, `floor`
///   and `ceil`, so nothing in the hardware forces the difference, and it dies on MoltenVK; a
///   native device relays it, its maths, as the class of a native device's own maths and sampling
///   ([`native::NATIVE_RELAYED`]). Not announced where the device answers as Metal's own function
///   (MoltenVK, under fast math). A nearest pixel sample also records it where the runtime floors
///   its coordinate to the texel's centre (a constexpr sampler with an offset, of an array or
///   fetching the edge with an offset, and a slot's or an argument buffer's nearest pixel state):
///   at a negative denormal coordinate Metal reads the texel before 0, where the device's floor
///   reads texel 0 (M3 Max). Announced at each draw or dispatch that binds the pipeline's
///   variant, inside its call, which dies at its selector;
/// - the note `HalfTexelConversion`: a read, gather or sample with a `half` result of a texture
///   bound in a format the translation does not convert as Metal does (an sRGB format otherwise
///   than its 8-bit slot, a packed or depth format and the rest) or of an argument buffer's
///   element, which keeps the runtime's rounding of the device's `f32` texel to nearest even,
///   where Metal rounds a 32-bit float texel toward zero and an 8- or 16-bit normalized one onto
///   its format's grid first; and, on MoltenVK under the runtime's default `ycbcr_filter`, a
///   `half` sample through a runtime sampler of a texture bound in a two-plane YCbCr format on the
///   grid of 1/8192 (an `_sRGB` twin), whose nearest texel Metal's own sampler converts as a
///   filtered value, to nearest even, where Metal rounds a tie away from zero (one `half` unit in
///   the last place at a tie), announced at each draw or dispatch that binds such a texture
///   whatever the sampler's state. The function can convert by Metal's
///   rule, as the runtime already does for the formats it measured, so nothing in the hardware
///   forces the difference, and it dies on MoltenVK; a native device relays it, its sampling, as
///   the class of a native device's own maths and sampling ([`native::NATIVE_RELAYED`]). Announced
///   at each draw or dispatch that binds the pipeline's variant, inside its call, which dies at its
///   selector;
/// - the note `BicubicFilterArithmetic`: Metal's bicubic filter, emulated by fetches, whose sums
///   differ from Metal's where the runtime has not pinned Metal's rule (a 2D normalized sum near
///   a rounding tie, one grid step apart; an argument buffer's sRGB element; the formats no case
///   reads; a denormal tap and the sign of a zero sum; a coordinate past `2^23` texels). The
///   fetches can compute Metal's sums once the rule is read, so
///   nothing in the hardware forces the difference, and it dies on MoltenVK; a native device
///   relays it, its sampling, as the class of a native device's own maths and sampling
///   ([`native::NATIVE_RELAYED`]). Announced at each draw or dispatch that binds the pipeline's
///   variant, wherever the function samples through a bicubic filter, whatever the format, inside
///   its call, which dies at its selector;
/// - the notes `TessellationWinding` and `TessellationFactorEdgeCases`: a tessellation
///   pipeline's output winding, declared as MoltenVK maps Metal's under Vulkan's upper-left
///   domain origin, where Metal's mapping is not read, and the other one reverses the facing of
///   every generated triangle; and a subnormal edge factor that does not cull and a NaN inside
///   factor at the range's lower end, which Metal answers so under the `integer` partition and is
///   not read under the others. Metal's answers can be
///   read and taken, so nothing in the hardware forces the difference, and each dies on every
///   device. Announced at each draw that binds the pipeline's variant, inside its call, which
///   dies at its selector;
/// - the note `ComputedSampleDepth`: a fragment function that runs its draw's depth test itself
///   at more than one sample, on a device whose `FragCoord.z` is the pixel centre's there
///   (MoltenVK), computes each sample's depth from its vertex function's window depth, which
///   the runtime measured on the M3 Max and the M4 exact in 10 of 17 depth cases and otherwise up
///   to 2 ulp from the depth Metal stores, so a depth test between surfaces
///   that close may decide otherwise. The runtime defers an exact path, and no hardware fact
///   forces the difference, so it dies. Announced at each draw that binds the pipeline's
///   variant, inside its call, which dies at its selector;
/// - the note `LoopedSampleOutputs`: a multisampled pipeline's fragment function that reads
///   `[[sample_mask]]`, which Metal runs once per covered sample (it reads `[[sample_id]]` or a
///   sample-qualified varying, or writes memory), and returns an output, which the runtime runs
///   once per pixel, its body once per covered sample from a loop over the coverage, and whose
///   outputs land on the samples Metal's runs keep, each the last pass's, where Metal writes each
///   sample's from that sample's run: equal where every pass computes the same outputs. The
///   runtime's ground is Vulkan's rule that an invocation under sample shading sees only its own
///   sample's bit of the coverage, no fact of a device's hardware, as `ClippedCoverageSplit`'s is
///   none, so it dies on every device. It replaces the runtime's refusal of such a function
///   (`CoverageLoopRefusal::Outputs`), which died where the pipeline was made. On MoltenVK on
///   Apple's GPU a function that writes memory runs as written, once per covered sample, exact,
///   and announces nothing. Announced at each draw that binds the pipeline's variant, inside its
///   call, which dies at its selector;
/// - the note `HelperBranchDropsSimdgroupStores`: on MoltenVK on an Apple GPU, a fragment
///   function where only helper invocations may run an operation SPIRV-Cross writes as an
///   `optnone` call before a storage write, which the runtime cannot keep them out of exactly,
///   where Apple's compiler may drop the texture writes of the live threads of such a SIMD group
///   (a branch every invocation of the draw takes alike leaves no helper alone and announces
///   nothing). The runtime can keep the helpers away from that call by
///   another form of the function, and defers it, so nothing in the hardware forces the
///   difference, and it dies. Announced at each draw that binds the pipeline's variant, inside
///   its call, which dies at its selector;
/// - `ExtendedRangeBlendsStoredCodes`: a draw that blends into, or reads, an extended-range
///   colour attachment an earlier draw of its pass may have written, or whose `Clear` left a
///   colour between two codes, and an execution of render commands in a pass with such an
///   attachment, where the runtime holds no `half` shadow of the colour: Metal's tile memory
///   holds the colour as a `half` through the pass and rounds it to its codes once, at the store,
///   where the runtime stores the codes at each draw, a value a code or two off that can grow
///   over draws (measured on an M4 and the M3 Max). The runtime holds the
///   shadow, exactly, through a pass of one sample and one layer without a rate map on a device
///   that fetches in the pixel interlock, announcing nothing; where it does
///   not yet (another pass, an execution of render commands or a tile dispatch before the draw,
///   an earlier render pass instance of the pass, a pipeline with no shadowed form) nothing in
///   the hardware forces the difference, and it dies on every device. Announced once per
///   device, at the first such draw or execution, inside its call, which dies at its selector;
/// - `RotatedLoadTiles`: a pass that loads a texture of rotation 4 and draws, whose 32 by 32
///   tiles no draw reaches the runtime lands mirrored as the others, where Metal keeps them in
///   place ([`crate::gpu::desc`]). That difference is one value on each side: Metal's M3 Max
///   left a block of a reached tile in place in two of 88 passes, a different block each time,
///   which is not the tiles the kind names. The runtime can load the unreached tiles in place, so
///   nothing in the hardware forces the difference, and it dies on every device. Announced once
///   per device, at the first such pass, inside its call, which dies at its selector;
/// - `YCbCrLoadTiles`: a pass that loads and stores a two-plane YCbCr colour attachment (500 to
///   507 or 540 to 545 over plane 0, the runtime's `ycbcr_renders`) and draws, which re-encodes
///   the planes' codes of every 32 by 32 tile of the rectangle bounding its draws' scissors from
///   the texels it loaded (each chroma pair from its filtered pairs), where Metal stores only the
///   tiles a draw's primitives reach and keeps the planes' bytes of the others (the M3 Max, macOS
///   27.0.1). The runtime can find the tiles a pass's primitives reach, as it does for an
///   extended-range attachment's load, so nothing in the hardware forces the difference, and it
///   dies on every device. Announced as the pass ends, inside `endEncoding`, which dies at its
///   selector;
/// - `YCbCrBlendRounding`: a draw that blends into such an attachment, which the device blends in
///   the floats the runtime holds the attachment in, where Metal's tile blends its halves, an
///   8-bit format's in `half` arithmetic and a 10-bit one's in `float` rounded toward zero to
///   `half`: a blended texel's codes can miss Metal's by one (every code of 4096 blended texels
///   matched by those rules on the M3 Max, macOS 27.0.1), and, as `RotatedBlendRounding`, the
///   exact form is the blend in the function. Announced at the draw, inside its call, which dies
///   at its selector;
/// - `ExtendedRangeResolve`: a pass's resolve of a multisampled extended-range colour attachment
///   whose resolve Metal averages from values finer than the codes the attachment holds
///   (`BGRA10_XR`'s alpha, an `_sRGB` twin's colour, `RGB10A2Unorm_sRGB`'s channels), which the
///   runtime's own resolve after the pass, on every device, writes from the codes: Metal's code
///   where a pixel's samples hold one, and where they differ a near-equivalent, measured against
///   Metal's resolve on the M3 Max (macOS 27.0.1, 4096 texels a format at 2 and 4 samples):
///   `BGRA10_XR`'s alpha within a code in every texel, an `_sRGB` twin's colour within a code in
///   99.5% of channels and up to 7 codes off where the samples' values nearly cancel,
///   `RGB10A2Unorm_sRGB`'s colour within 2 and its alpha within 1. A linear form's colour,
///   `BGR10_XR`'s and `BGRA10_XR`'s, is Metal's rule in every texel, which announces nothing.
///   The runtime's resolve is no device's rounding, so a native device does not relay it either.
///   Announced as the pass ends, inside `endEncoding`, which dies at its selector;
/// - `WordResolve`: a pass's resolve of a multisampled packed 16-bit colour attachment (the formats
///   the device renders only through the view of its word), which the runtime's own resolve
///   writes from each field's codes, where Metal averages its tile's values finer than the codes:
///   Metal's code where a pixel's samples hold one word, and within one code of Metal's in every
///   channel where they differ (M3 Max, macOS 27.0.1). Such multisampled textures were refused
///   before the runtime held them in their word's format, so placing it here ends no run that
///   went on. Announced as the pass ends, inside `endEncoding`, which dies at its selector;
/// - `AstcStorageNeutral`: a draw or dispatch binding an ASTC texture at a storage slot its function
///   writes, or reads and writes, which binds the slot's neutral, so its writes leave the blocks
///   as they were and its reads answer zero, where Metal completes the writes and stores bytes no
///   format defines into the blocks. Such bindings died before the runtime bound the neutral, so
///   placing it here ends no run that went on. Announced at the draw or dispatch, inside its call,
///   which dies at its selector;
/// - the notes of the device's own answers on a GPU that is not Apple's, which the runtime gives
///   in place of Metal's, each recording its bound: the device's maths (`NativeTranscendentals`,
///   `NativeDivision`, `NativeFma`, `NativeNanEncoding`, `NativeMinMax`, `NativeDot`, `NativeMix`,
///   and `DevicePackConversion`, its `air.pack` and `air.unpack` of a normalized format), its
///   rounding of a `half` result (`NativeHalfConversion` and `NativeHalfQuantize`), its sampling
///   (`DeviceSampling` and `DeviceTexelConversion`), its vertex input's conversion of a normalized
///   attribute (`DeviceVertexConversion`) and its alpha to coverage (`DeviceAlphaCoverage`). On a
///   device that runs on Metal (MoltenVK) the runtime answers most as Metal's own functions,
///   sampler and conversions, which announce nothing there, and one announced there dies (its
///   filter of a two-plane YCbCr texture announces `YCbCrFilterNotExact` instead). The runtime
///   announces `DevicePackConversion` on every device, MoltenVK among them, for a function that
///   converts the texels of a slot or attachment of an extended-range, padded or bfloat form,
///   which it holds in its codes and converts by the device's arithmetic (a read multiplies a
///   code by an `f32` reciprocal, a write rounds an `f32` product, where Metal answers the
///   correctly rounded quotient and encodes by the exact product), so on a device on Metal each
///   draw of such a function dies of it (the runtime's `lower/arith/extended_range.rs`; measured
///   on the M3 Max under macOS 27.0.1, MoltenVK 1.4.2: each unblended and blended draw of
///   `encode_render`'s `half4_extended_range_blends_announce_their_source`). A native device
///   relays each as the
///   class of a native device's own answers ([`native::NATIVE_RELAYED`]). Announced at each draw
///   or dispatch that binds the pipeline's variant, inside its call, which dies at its selector;
/// - `ProvokingVertexLines` and `ProvokingVertexUnmatched`: a draw under the last provoking vertex
///   (`setProvokingVertexMode:` 1) of a pipeline state that reads it (a flat input, a layer or
///   viewport its vertex function selects, or `[[barycentric_coord]]`), which the runtime draws
///   otherwise than Metal. On a device without its last vertex (`VK_EXT_provoking_vertex`'s
///   `provokingVertexLast` with `provokingVertexModePerPipeline`, which MoltenVK lacks), each line
///   is drawn from its last vertex to its first, its flat inputs Metal's but its pixels and its
///   smooth inputs not always (`ProvokingVertexLines`); of the weights of a triangle, or of an
///   emulated mesh's, whose fragment function also reads a flat input the first is Metal's and
///   the other two Metal's but for rounding, and a patch or a mesh draw through
///   `VK_EXT_mesh_shader` takes its flat inputs from its first vertex
///   (`ProvokingVertexUnmatched`). On a device with it, a strip whose vertex function computes
///   `[[primitive_id]]` into its varying reads each primitive's last vertex's, and a layer or
///   viewport selected in a synthesized geometry stage comes from each primitive's first vertex
///   (`ProvokingVertexUnmatched`). On every device a mode from 2 up, its top bit set or not,
///   which Metal draws a third way, is drawn as mode 1, patches as mode 0, as Metal's: a draw
///   whose shape the third way draws otherwise than mode 1, as far as its state reads the
///   difference (a strip of triangles, lines, sized points, the triangles of a state that selects
///   a layer or viewport, and mesh draws, not measured there), announces it
///   (`ProvokingVertexUnmatched`); the setter announces nothing. Announced at each such draw,
///   inside its call, which dies at its selector;
/// - `DeviceWideLines`: a draw of lines at a width other than 1 (`setLineWidth:`), of a
///   triangle's edges in fill mode `Lines` or of a mesh draw's lines, that the runtime draws by
///   the device's own lines where they are not Metal's parallelogram: a strict device's rectangle
///   with square ends (NVIDIA, lavapipe), lines Vulkan does not state as parallelograms (ANV, and
///   every width below 1), or a width the device rounds to its `lineWidthGranularity`. They can
///   cover pixels Metal's parallelogram does not, and miss others, at each end and along each
///   edge. The runtime's default (`WideLinePolicy::Device`) takes them; `REIMS_VGPU_EXACT`'s
///   `wide_lines` ([`crate::platform::EXACT_FORMS`]) takes the band instead for lines, two
///   triangles over Metal's parallelogram, on a device with `tessellationShader`, but not for a
///   triangle's edges or a mesh draw's lines, which the band does not draw. MoltenVK has no wide
///   lines, so on the Mac nothing announces it: the band draws lines at every such width.
///   Announced at each such draw, inside its call, which dies at its selector;
/// - `LinesAtNearestWidth`: a draw of lines at a width other than 1, of a triangle's edges in
///   fill mode `Lines` or of a mesh draw's lines, that neither the device's own lines at the width
///   nor the band draws, drawn by the device's lines at the width nearest it they draw: the width
///   clamped into `lineWidthRange` where the device has `wideLines`, else 1 pixel. They cover
///   fewer or more pixels across each line than Metal's. A triangle's edges and a mesh draw's
///   lines where the device's lines do not draw the width (every such draw on MoltenVK, 1 pixel
///   wide), a line on a device without `tessellationShader` whose lines do not draw it, a band
///   under vertex amplification above 1, and every line in a pass of a rasterization rate map or
///   of an execution's command, whose width Metal's answer is not measured for. Announced at each
///   such draw, inside its call, which dies at its selector;
/// - `WideEdgeDepthBias`: a draw of a triangle's edges in fill mode `Lines` at a width other than
///   1, drawn as the band, under a depth bias the device applies otherwise than Metal (a slope
///   factor, or a constant factor over a float depth): Metal biases the edges by the triangle's
///   depth slope and its largest depth's exponent, the device each of the band's triangles by its
///   own. Such draws died before the runtime drew triangles' edges as bands (`DeviceWideLines` or
///   `LinesAtNearestWidth`), so placing it here ends no run that went on. Announced at each such
///   draw, inside its call, which dies at its selector;
/// - `DeviceLineTies`: a draw of lines, or of a triangle's edges in fill mode `Lines`, that the
///   device's own lines draw at width 1 (Metal's default width, or the nearest width a device
///   without `wideLines` draws), on a driver not measured to break a tie as Metal does (the
///   runtime's `DriverFact::LineTiesAreMetals`, measured on MoltenVK on the Apple9 GPUs under
///   macOS 27.0, the M3 Max's, which announce nothing): a line through a vertex two pixels'
///   diamonds share exits neither diamond, which Vulkan's open diamond leaves to the driver with
///   no control over it, where Metal covers the lower pixel, so such a pixel, and the rest of a
///   line that runs along such vertices, can lie a row or a column from Metal's (NVIDIA's
///   615.71.09 on an RTX 5080 Laptop GPU covers the other pixel; lavapipe leaves out a strip's
///   first pixel where it starts on such a vertex; RADV and ANV are not measured). On MoltenVK
///   Metal's own lines draw, but where the runtime has not measured them it dies, as no other
///   reason covers it; a native device relays it, its rasterizer's tie, a fact of its hardware
///   whose only exact form is the line's rule run in the fragment function, an interpreter's
///   ([`native::NATIVE_RELAYED`]). Announced at each such draw, inside its call, which dies at its
///   selector;
/// - `DeviceSubpixelSnap`: a draw of lines, or of a triangle's edges in fill mode `Lines`, that
///   the band draws, on a driver not measured to snap a vertex half a step off its 1/256-pixel
///   grid up, as Metal snaps a line's endpoint (the runtime's `DriverFact::SubpixelSnapIsMetals`,
///   measured as `DeviceLineTies`' is): the band's corners, its endpoints offset by a multiple of
///   1/32 pixel, are snapped by the device's rasterizer, where Metal snaps the endpoints first, so
///   where an endpoint lies half a step off the grid a pixel within that half step of the band's
///   edge can be drawn, or left out, a row or a column from Metal's (NVIDIA's 615.71.09 on an RTX
///   5080 Laptop GPU and lavapipe round the half step down). The band is the runtime's own:
///   snapping its endpoints to Metal's grid before it offsets them leaves the rasterizer nothing
///   to round, so the hardware allows the exact form, and it dies on every device, as
///   `WideEdgeDepthBias` does. On a native device under the default only a line at a width its
///   own lines do not draw (below 1/16, which Metal draws at 1/16, or past its `lineWidthRange`)
///   takes the band, and under `REIMS_VGPU_EXACT`'s `wide_lines` every line its lines would draw
///   otherwise than Metal's. Announced at each such draw, inside its call, which dies at its
///   selector;
/// - the note `DeviceInterpolationOffset`: a fragment function's interpolant read at an offset
///   (`interpolate_at_offset`), at the device's interpolation offset, which Vulkan rounds to a grid
///   of at least sixteenths and clamps to the device's range, where Metal's grid and range are not
///   measured. Such reads died before the runtime lowered Metal's `interpolant` inputs, so placing
///   it here ends no run that went on. Announced at each draw that binds such a pipeline;
/// - `PredicationWordPastBuffer`: a command predication record of the runtime's whose condition
///   word lies past its buffer's end, past the 256 bytes Metal rounds a lone buffer up to, or past
///   the end of the heap the buffer lies in, which reads zero where Metal reads what its own
///   address space holds there. Announced at the record, inside its call, which dies at its
///   selector. No record reaches the runtime's predication here: this platform's own carries
///   command predication ([`crate::gpu::predication`]);
/// - `HeapTwoPlaneLayout`: a heap's two-plane XR10 texture of more than one level, or a 1D array,
///   whose first level's texels lie in the heap's bytes as Metal's CPU copies and GPU writes lay
///   them, and whose later levels lie apart from the heap's bytes, a 1D array's GPU reads reading
///   each texel, where Metal's paths address those texels at bytes that differ by path. Announced
///   inside the heap's creator of the texture, which dies at its selector;
/// - `CopiedHostPages`: a no-copy buffer, heap or rebind made over a copy of the guest's pages in
///   memory of the device's own, as their import failed (out of memory, or a driver out of the
///   mappings the host's kernel allows), or the views' gate refused it on Linux, where mapping the
///   pages from their files would have run the views of memory within the headroom of their
///   budget of kernel mappings, or the pages are a guest's read-only object's (a replace
///   `readOnly:YES`, a heap or no-copy buffer made with `MTLResourceOptions` bit 24, pages mapped
///   read-only), whose copy drops the GPU's writes, as Metal's GPU drops them (`crate::os`'s
///   import gate, `crate::gpu::writes`), which each commit fills from the pages and each
///   completion writes back: a CPU write the GPU reads mid-command-buffer, or a GPU write the CPU
///   reads before the batch completes, is not seen, where Metal's GPU uses the pages themselves.
///   Announced inside the call that makes it, which dies at its selector.
///
/// Every kind not listed here or in [`HEARD`] is relayed, those the guest sees included, each for
/// one of the reasons the module's docs give: logged once per device and not dying. Each relayed
/// kind is reviewed, its reason named in
/// the test `every_degradation_kind_is_placed_once`, which a kind the runtime adds fails until it
/// is placed; a kind whose answer the runtime states is Metal's (`rt::Exactness::Exact`: a
/// performance or validity kind) or one of Metal's where Metal's own is not one value
/// (`rt::Exactness::OneOfMetals`) is placed by that statement.
pub const DIES: [rt::Degradation; 140] = [
    rt::Degradation::ReleasedCommandState,
    rt::Degradation::TwoPlaneTextureCopy,
    rt::Degradation::TwoPlaneBufferBlit,
    rt::Degradation::YCbCrFilterNotExact,
    rt::Degradation::RotatedBlendRounding,
    rt::Degradation::EncodedUseRebound,
    rt::Degradation::ZeroLayerMapAfterRelease,
    rt::Degradation::RateMapResetUnencoded,
    rt::Degradation::HeapTextureBlockCompressedLayout,
    rt::Degradation::BiasedDepthPastUnitRange,
    rt::Degradation::UnormDepthBias,
    rt::Degradation::UnormDepthRounding,
    rt::Degradation::RestrictedDepthUpload,
    rt::Degradation::RestrictedDepthRange,
    rt::Degradation::UnmeasuredResolve,
    rt::Degradation::MoltenVkArgumentBuffersUnread,
    rt::Degradation::FunctionStorageAssumed,
    rt::Degradation::HalfSampleFilter,
    rt::Degradation::SrgbTexelDecode,
    rt::Degradation::FloatSampleFilter,
    rt::Degradation::AnisotropicFilter,
    rt::Degradation::DeviceAnisotropicFilter,
    rt::Degradation::ViewBaseTexelOffset,
    rt::Degradation::DeviceInterpolation,
    rt::Degradation::DeviceInterpolationOffset,
    rt::Degradation::QuadsSplitByClipping,
    rt::Degradation::VertexSimdgroupGrouping,
    rt::Degradation::FragmentSimdgroupGrouping,
    rt::Degradation::NonStrictResidency,
    rt::Degradation::SparseResidency,
    rt::Degradation::DeviceRayIntersection,
    rt::Degradation::PointGuardBand,
    rt::Degradation::NanDepth,
    rt::Degradation::ClippedDepthPieces,
    rt::Degradation::SampleDepthClamp,
    rt::Degradation::DepthBiasPastOne,
    rt::Degradation::SignedDepthTest,
    rt::Degradation::InPlaceVertexReads,
    rt::Degradation::FragmentsBeforeGeometry,
    rt::Degradation::AttachmentReadInPlace,
    rt::Degradation::HeapDepthStencilApart,
    rt::Degradation::HeapTextureHidesAliasedWrites,
    rt::Degradation::HeapTextureCompressed,
    rt::Degradation::UnlistedArgumentSampler,
    rt::Degradation::UnreadableHandleTexture,
    rt::Degradation::UnnamedTextureUnviewable,
    rt::Degradation::UnnamedTexturesPastCapacity,
    rt::Degradation::HeapSetCopied,
    rt::Degradation::DetachedStorageKeepsWrites,
    rt::Degradation::MixedDepthStencilResolve,
    rt::Degradation::UncoveredColorKept,
    rt::Degradation::SnormMinimumKept,
    rt::Degradation::SharedExponentKept,
    rt::Degradation::DeviceRoundedResolve,
    rt::Degradation::CappedAnisotropy,
    rt::Degradation::NotMipmappedAnisotropicSampler,
    rt::Degradation::NotMipmappedAnisotropicFilter,
    rt::Degradation::CombinedArchiveStages,
    rt::Degradation::ImplementationDefinedReconvergence,
    rt::Degradation::F16FmaDoubleRounding,
    rt::Degradation::F16IntConvDoubleRounding,
    rt::Degradation::GradFromBias,
    rt::Degradation::ReductionOrder,
    rt::Degradation::DeviceSignedZeroInfNan,
    rt::Degradation::TessellationFactorScale,
    rt::Degradation::TessellationFactorInstance,
    rt::Degradation::Fp8Value,
    rt::Degradation::MultisamplePosition,
    rt::Degradation::CubeGradientFootprint,
    rt::Degradation::QuadsCountedBeforeDiscard,
    rt::Degradation::BlendedWriteRounding,
    rt::Degradation::UnmeasuredGroupSpeculation,
    rt::Degradation::DeviceSampleLevel,
    rt::Degradation::LinkedSamplerNumbering,
    rt::Degradation::PixelSamplerScaling,
    rt::Degradation::ArgumentPixelSamplerScaling,
    rt::Degradation::DiscardDropsSimdgroupStores,
    rt::Degradation::FreshPatternZeroed,
    rt::Degradation::ExtendedRangeFloatBlendSource,
    rt::Degradation::CrossClassTextureNeutral,
    rt::Degradation::IndirectNilBufferOffset,
    rt::Degradation::CompressedTexelDecode,
    rt::Degradation::DecodedTexelPrecision,
    rt::Degradation::HeapTextureLargerThanMetal,
    rt::Degradation::HostDeallocatorDelayed,
    rt::Degradation::GroupedCompletion,
    rt::Degradation::DrawDroppedStillMaking,
    rt::Degradation::DispatchDroppedStillMaking,
    rt::Degradation::DrawDroppedUnmade,
    rt::Degradation::VisibilityCountPerColor,
    rt::Degradation::ColorAttachmentGroups,
    rt::Degradation::DevicePointSize,
    rt::Degradation::ClippedCoverageSplit,
    rt::Degradation::TessellationTriangleOrder,
    rt::Degradation::DeviceNanEncoding,
    rt::Degradation::DeviceArithmetic,
    rt::Degradation::DeviceDenormals,
    rt::Degradation::HalfTexelConversion,
    rt::Degradation::BicubicFilterArithmetic,
    rt::Degradation::TessellationWinding,
    rt::Degradation::TessellationFactorEdgeCases,
    rt::Degradation::ComputedSampleDepth,
    rt::Degradation::LoopedSampleOutputs,
    rt::Degradation::HelperBranchDropsSimdgroupStores,
    rt::Degradation::ExtendedRangeBlendsStoredCodes,
    rt::Degradation::ExtendedRangeBlendSource,
    rt::Degradation::PrivateResourceOption,
    rt::Degradation::StaleDetachedReach,
    rt::Degradation::RotatedLoadTiles,
    rt::Degradation::YCbCrLoadTiles,
    rt::Degradation::YCbCrBlendRounding,
    rt::Degradation::ExtendedRangeResolve,
    rt::Degradation::WordResolve,
    rt::Degradation::AstcStorageNeutral,
    rt::Degradation::NativeTranscendentals,
    rt::Degradation::NativeDivision,
    rt::Degradation::NativeFma,
    rt::Degradation::NativeNanEncoding,
    rt::Degradation::NativeMinMax,
    rt::Degradation::NativeDot,
    rt::Degradation::NativeMix,
    rt::Degradation::NativeHalfConversion,
    rt::Degradation::NativeHalfQuantize,
    rt::Degradation::DeviceSampling,
    rt::Degradation::DeviceTexelConversion,
    rt::Degradation::DevicePackConversion,
    rt::Degradation::DeviceVertexConversion,
    rt::Degradation::DeviceAlphaCoverage,
    rt::Degradation::ProvokingVertexLines,
    rt::Degradation::ProvokingVertexUnmatched,
    rt::Degradation::DeviceWideLines,
    rt::Degradation::LinesAtNearestWidth,
    rt::Degradation::WideEdgeDepthBias,
    rt::Degradation::DeviceLineTies,
    rt::Degradation::DeviceSubpixelSnap,
    rt::Degradation::PredicationWordPastBuffer,
    rt::Degradation::HeapTwoPlaneLayout,
    rt::Degradation::CopiedHostPages,
    rt::Degradation::CoverageReadsRoundedAlpha,
    rt::Degradation::UnmeasuredFetchBlend,
];

/// The degradations a creator here learns of, announced at every occurrence too, which do not
/// die ([`hearing`]):
/// - `HeapTwoPlaneTexture`, a heap's two-plane XR10 texture whose texels lie in memory of their
///   own, apart from the heap's bytes: only one whose moves pass the device's limits (a one-level
///   one keeps its planes in the heap's bytes, a 3D one its first depth slice's, over which Metal
///   lays every other, and a mipmapped one and a 1D array its first level's, announcing
///   `HeapTwoPlaneLayout` instead, of [`DIES`]), announced on the caller's thread inside the
///   placement's call, which dies where a buffer or texture of the heap that lives
///   shares a byte with it: the heap's cover, the buffer over all of it core makes with each
///   heap, included, so in every heap core makes ([`crate::gpu::resource::Placements`]);
/// - `OptimizedDepthStencilTexture`, a depth or stencil texture without
///   `allowGPUOptimizedContents`, whose layout MoltenVK keeps GPU-optimized: the runtime sizes it
///   as Metal's texture with the property on, larger than Metal's with it off (16512 bytes for a 64
///   by 64 `Depth32Float` texture, where Metal answers 16384). Announced where such a texture is
///   sized or placed. Its reads and writes are Metal's, so its creation relays it, a device's
///   texture's, whose range or slot of Metal's space is of Metal's size, the runtime's model of
///   Metal's answer, which asks no image and hears nothing
///   ([`crate::gpu::metal_space::own_heap_size`]), and a heap's placement's that fits it. Where
///   the model has none (one of `ShaderAtomic` usage the runtime refuses first, as Metal
///   asserts), a device's texture dies at `newTextureWithDescriptor:`, at every shape; an
///   `X32_Stencil8` texture takes the model's size, a `Stencil8` one's of its shape.
///   `heapTextureSizeAndAlignWithDescriptor:` hears it and answers Metal's, the runtime's model of
///   it, as the device's texture takes it, and dies at its selector where the model has none
///   ([`crate::gpu::device`]). A placement that depends on the runtime's size dies: a heap's
///   placement that hears it dies where it answers nil inside the heap and Metal's smaller size
///   fits there or is not modelled (where Metal's does not fit either, the nil is Metal's),
///   where it is made but Metal answers nil or a size the runtime's falls short of, and in a heap
///   over no address ranges at every shape (`crate::gpu::resource::HeapObject`'s placement).
///   Elsewhere the heap keeps the texture apart: a placement over a byte of it dies while it lives,
///   and so does such a texture placed over a byte of a buffer or texture that lives, or while its
///   heap's cover lives, the buffer over all of the heap's bytes core makes with each heap, whose
///   bytes the guest reads with no call to die at (`crate::gpu::resource::Placements`). A heap's
///   `usedSize`, `currentAllocatedSize` and `maxAvailableSize`, which would answer it too, are no
///   answer core asks for. Relayed the first time alone ([`RELAYED_ONCE`]);
/// - `RasterizationRateMap`, a rasterization rate map whose rates vary, which the runtime compiles
///   to other than the identity: a pass over it draws by regions whose lines are its tiles' exact
///   lines, where Apple's are its fixed-point ones. Its parameter data and coordinate maps are
///   Metal's, and so are a pass's loads, clears and stores over it. Announced as the runtime
///   compiles the map, inside `newRasterizationRateMapWithDescriptor:` or `resetUsingDescriptor:`,
///   which hear it ([`crate::gpu::rate_map`]): a draw or an execution of indirect commands that is
///   not empty, and a tile dispatch, in a pass over such a map relay it as [`Inexact::RateMap`]
///   (in strict mode die at their selector, [`crate::gpu::encode_render`]). Relayed the first
///   time alone ([`RELAYED_ONCE`]);
/// - `HeapTextureUntiledLayout`, a texture in a heap over the guest's pages that the runtime does
///   not lay out in Metal's tiles: a layout not measured (a format whose layout in Metal's heaps is
///   not probed), an image that holds other bytes than Metal's elements, moves past the device's
///   limits, or a multisample texture whose samples the device does not move to and from a buffer
///   (a 16-bit depth where its multisampled copies go by draws), in memory of its own; a
///   multisample depth or stencil texture Metal keeps uncompressed lies in Metal's tiles, its
///   samples side by side, on MoltenVK as elsewhere. The runtime announces it too of a texture in a
///   heap over the device's memory whose bytes there are not Metal's layout: its image in the
///   driver's layout (any colour image on a driver other than MoltenVK, a depth or stencil image
///   the runtime does not tile), a shadow of the bytes in rows, a multisample texture (but a colour
///   one Metal keeps uncompressed on MoltenVK, whose multisample colour images the runtime measured
///   lying as Metal's heap textures), or a layout not measured. In a heap over no address ranges,
///   core's legacy heap, whose bytes core copies to and from the guest's pages, it is relayed alone
///   ([`crate::gpu::resource::Placements`]). Its reads, writes and copies are Metal's, but its
///   bytes in the heap are the runtime's layout, where Metal's are its tiles, so a buffer or
///   texture over them would meet other bytes than on Metal. Announced at each such texture, inside
///   `newTextureWithDescriptor:offset:`, which hears it. Over the guest's pages the heap keeps it
///   apart, as an optimized one: a buffer or texture placed over a byte of it while it lives dies
///   at its placement, and so does such a texture placed over a byte of a buffer or texture that
///   lives, or while its heap's cover lives, the buffer over all of the heap's bytes core makes
///   with each heap, so in every heap core makes: the cover's bytes and the guest's pages under
///   them would hold the runtime's layout where Metal's hold its tiles, which the guest reads with
///   no call to die at (`crate::gpu::resource::Placements`). Relayed the first time alone
///   ([`RELAYED_ONCE`]), where nothing shares its bytes;
/// - `HeapTextureLosslessLayout`, a texture in a heap over the guest's pages that Metal keeps
///   losslessly compressed, whose bytes on Metal are no layout of its texels: a single-sample one
///   laid out in its uncompressed tiles, which guests make by the tens of thousands a boot, and a
///   2- or 4-sample one, its samples in its uncompressed tiles, or, where the device does not
///   move them to and from a buffer, its image in memory of its own, the heap's bytes keeping what
///   they held. The runtime announces it too of a depth or stencil texture it lays out in Metal's
///   uncompressed tiles in any heap on MoltenVK, and of such a texture in a heap over the device's
///   memory, its image there of the runtime's usage, which need not be compressed as Metal's: in
///   a heap over no address ranges, core's legacy heap, it is relayed alone
///   ([`crate::gpu::resource::Placements`]). Its reads, writes, resolves and copies are Metal's:
///   only another resource over its bytes meets other bytes. Announced at each such texture,
///   inside `newTextureWithDescriptor:offset:`, which hears it. A single-sample one is made and the
///   notice relayed, the first time alone ([`RELAYED_ONCE`]), pending a decision. A multisample
///   one over the guest's pages is made too, and relayed as [`Inexact::HeapLayout`] (in strict
///   mode it dies there, named): where its image lies in memory of its own, a buffer or texture
///   over its bytes, which on Metal sees and changes them, would neither see nor change it. The
///   notice does not tell that one from one whose samples lie in the heap's bytes, which a
///   buffer over them meets uncompressed, as a single-sample one's texels: both relay so;
/// - `UnmeasuredHeapTextureSize`, a texture whose layout in Metal's heaps the runtime does not
///   model, whose size and alignment it answers as its device's image's, larger or smaller than
///   Metal's: the runtime names 8 samples, whose layout it does not measure, but no texture of 8
///   samples reaches it: the runtime takes Apple's sample counts alone (1, 2 and 4) on every
///   device, a native one included, so its descriptor check refuses 8 at every
///   creator and the size query first (`SampleCount`), which dies there, as Metal stops them on the
///   M3 Max and the M4 (the runtime's measurements). Announced where such a texture is sized or
///   placed, inside `heapTextureSizeAndAlignWithDescriptor:`, a heap's
///   `newTextureWithDescriptor:offset:` and `newTextureWithDescriptor:`, whose texture of memory of
///   its own takes a range of Metal's space of the runtime's size
///   ([`crate::gpu::metal_space::own_heap_size`]). Each hears it and relays it as
///   [`Inexact::HeapLayout`], naming the kind and the texture (in strict mode it dies at its
///   selector, [`crate::gpu::metal_space::unmeasured_size`]), but for a heap's nil at or past its
///   end, Metal's whatever the size. Relayed the first time alone ([`RELAYED_ONCE`]) where nothing takes the size;
/// - `HalfOutputPrecision`, a fragment function's `half` colour output, of a module compiled
///   without fast math or naming no math mode (one compiled with fast math notes
///   `HalfOutputFastMathBlend` in its place, relayed in the accepted class of fast-math
///   rewrites), which the runtime rounds to `half` before the blend and the
///   format's conversion, where Metal may carry it at `float` precision, and a `float` colour
///   output of such a module into a 16-bit float attachment blended by `One` and `Zero`, added,
///   which the runtime blends by the device's blend; a module without fast math notes it only
///   where an attachment the function writes blends. Announced at each draw that binds such a state,
///   inside its call, which hears it
///   and relays it as [`Inexact::HalfOutputBlend`] (in strict mode dies at its selector) where an
///   attachment the function writes stores a value past the
///   bound the rounding is relayed within: a 16-bit float
///   attachment blended otherwise than by `Min` or `Max` (by `One` and `Zero`, added, too, where
///   Metal keeps the zero factor's product, a NaN over an infinite texel, and the device's blend
///   answers its own, which the runtime notes for a `float` output too), a 32-bit
///   float one blended by a factor that reads the source beside one neither `Zero` nor `One`, and
///   an 8-bit `_sRGB` one blended by `DestinationColor` and `OneMinusDestinationColor`
///   (`crate::gpu::encode_render`'s `half_apart`, each measured against Metal on the M3 Max).
///   Elsewhere the draw relays it: within one code of a unorm, snorm or packed float format, a
///   16-bit float format's `half` unblended or blended by `Min` or `Max`, and a 32-bit float
///   format's value, as measured. An execution's commands announce it at the commit, which hears
///   nothing and relays it. Relayed the first time alone ([`RELAYED_ONCE`]).
pub const HEARD: [rt::Degradation; 7] = [
    rt::Degradation::HeapTwoPlaneTexture,
    rt::Degradation::OptimizedDepthStencilTexture,
    rt::Degradation::RasterizationRateMap,
    rt::Degradation::HeapTextureUntiledLayout,
    rt::Degradation::HeapTextureLosslessLayout,
    rt::Degradation::UnmeasuredHeapTextureSize,
    rt::Degradation::HalfOutputPrecision,
];

/// The kinds of [`HEARD`] the runtime would announce once per device but for `announce_each`,
/// which asks for each occurrence only so a call here can hear it: relayed the first time alone,
/// as without a sink, so a guest's every pass does not log one ([`notice`]).
const RELAYED_ONCE: [rt::Degradation; 6] = [
    rt::Degradation::OptimizedDepthStencilTexture,
    rt::Degradation::RasterizationRateMap,
    rt::Degradation::HeapTextureUntiledLayout,
    rt::Degradation::HeapTextureLosslessLayout,
    rt::Degradation::UnmeasuredHeapTextureSize,
    rt::Degradation::HalfOutputPrecision,
];

/// The kinds of [`RELAYED_ONCE`] this process relayed, a bit each at the kind's place there: a
/// notice of one is announced at every occurrence (a heap's texture, a draw's bind), on whatever
/// thread meets it, so no lock is taken for it.
static RELAYED_KINDS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
const _: () = assert!(RELAYED_ONCE.len() <= 8, "a bit of RELAYED_KINDS each");

/// Whether `kind` is of [`RELAYED_ONCE`] and this is the process's first notice of it, which is
/// relayed ([`RELAYED_KINDS`]).
fn first_relayed_once(kind: rt::Degradation) -> Option<bool> {
    use std::sync::atomic::Ordering;
    let at = RELAYED_ONCE.iter().position(|&k| k == kind)?;
    let bit = 1u8 << at;
    Some(RELAYED_KINDS.fetch_or(bit, Ordering::Relaxed) & bit == 0)
}

/// What every call here reads of this thread on its way in and out ([`inside`]): one lookup of
/// a value with no destructor, so the success path reads no other thread-local and never touches
/// the notice itself. Measured on the M3 Max, a release build: [`plain`] of a call that does
/// nothing takes 5.9 ns, where it took 11.3 ns with the depth and the notice each a thread-local
/// looked up on every call, and the notice moved out of its own at every return.
struct Calls {
    /// How many calls here this thread is inside.
    depth: Cell<u32>,
    /// Whether [`PENDING`] holds a notice.
    pending: Cell<bool>,
}

thread_local! {
    /// This thread's calls here.
    static CALLS: Calls = const {
        Calls {
            depth: Cell::new(0),
            pending: Cell::new(false),
        }
    };
    /// A notice of [`DIES`] the runtime announced on this thread inside a call here, which the
    /// call dies at once it returns: read only where [`Calls::pending`] says it is there.
    static PENDING: RefCell<Option<rt::DegradeNotice>> = const { RefCell::new(None) };
    /// The list of the innermost [`hearing`] this thread runs, where one runs: the kinds of
    /// [`HEARD`] the runtime announced on this thread inside it. The list lies on the hearing's
    /// stack, and the hearing puts back the one outside it before it reads its own.
    static HEARING: Cell<Option<NonNull<Vec<rt::Degradation>>>> = const { Cell::new(None) };
    /// Whether this thread runs an [`unheard`] call.
    static UNHEARD: Cell<bool> = const { Cell::new(false) };
}

/// Keeps `notice` for the call here this thread is inside to die at once it returns.
fn keep_pending(notice: &rt::DegradeNotice) {
    PENDING.with(|p| *p.borrow_mut() = Some(notice.clone()));
    CALLS.with(|c| c.pending.set(true));
}

/// The notice this thread keeps for a call here to die at, taken: `None` where it keeps none.
fn take_pending(calls: &Calls) -> Option<rt::DegradeNotice> {
    if !calls.pending.replace(false) {
        return None;
    }
    PENDING.with(RefCell::take)
}

/// How many calls here this thread is inside.
fn depth() -> u32 {
    CALLS.with(|c| c.depth.get())
}

/// Whether this process dies at a difference from Metal's answer it could go on past: a
/// degradation of [`DIES`] or one of [`Inexact`] (the module's docs). `REIMS_VGPU_STRICT` set to
/// `1`, read once per process, turns it on; by default the process relays them and goes on. A
/// test of this crate runs strict, but where the tests' `relaxed` runs it on its thread.
pub fn strict() -> bool {
    #[cfg(test)]
    if let Some(strict) = TEST_STRICT.with(Cell::get) {
        return strict;
    }
    strict_process()
}

/// [`strict`] as the process is set, whatever a test's thread runs under (the tests' `relaxed`).
pub(crate) fn strict_process() -> bool {
    static STRICT: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        cfg!(test) || std::env::var_os("REIMS_VGPU_STRICT").is_some_and(|v| v == "1")
    });
    *STRICT
}

#[cfg(test)]
thread_local! {
    /// The strictness a test runs its thread's calls under ([`relaxed`]), in place of the
    /// process's.
    static TEST_STRICT: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Runs `call` with the calls and notices on this thread under the default, not strict: a test's
/// of what the process does without `REIMS_VGPU_STRICT`.
#[cfg(test)]
pub(crate) fn relaxed<T>(call: impl FnOnce() -> T) -> T {
    /// Puts back the outer strictness on drop, an unwinding call's included.
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_STRICT.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(TEST_STRICT.with(|s| s.replace(Some(false))));
    call()
}

/// How many times this process relayed each kind of [`DIES`], in its order, outside strict mode.
static RELAYED_DIES: [std::sync::atomic::AtomicU64; DIES.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; DIES.len()];

/// Counts an occurrence of `kind`, of [`DIES`], relayed: whether it is the process's first, whose
/// line is written.
fn count_relayed(kind: rt::Degradation) -> bool {
    use std::sync::atomic::Ordering;
    DIES.iter()
        .position(|&k| k == kind)
        .and_then(|at| RELAYED_DIES.get(at))
        .is_some_and(|c| c.fetch_add(1, Ordering::Relaxed) == 0)
}

/// Each kind of [`DIES`] this process relayed outside strict mode, with how many times the runtime
/// announced it: once per device by default, but in census mode or in a strict process whose
/// thread runs the tests' `relaxed` ([`crate::platform`]'s `announced_each`).
pub fn relayed_counts() -> Vec<(rt::Degradation, u64)> {
    use std::sync::atomic::Ordering;
    (DIES.iter().zip(&RELAYED_DIES))
        .map(|(&k, c)| (k, c.load(Ordering::Relaxed)))
        .filter(|&(_, n)| n > 0)
        .collect()
}

/// Runs `call` on this thread, and the kinds of [`HEARD`] the runtime announced on this thread
/// inside it, in order, each as many times as announced.
pub fn hearing<T>(call: impl FnOnce() -> T) -> (T, Vec<rt::Degradation>) {
    /// Puts back the outer hearing on drop, an unwinding call's included.
    struct Restore(Option<NonNull<Vec<rt::Degradation>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            HEARING.with(|h| h.set(self.0));
        }
    }
    let mut heard = Vec::new();
    let out = {
        let _outer = Restore(HEARING.with(|h| h.replace(Some(NonNull::from(&mut heard)))));
        call()
    };
    (out, heard)
}

/// Runs `call`, a probe of the device's own outside any guest's work (`crate::os`'s
/// `learn_view_imports`), with what the runtime announces on this thread inside it heard by no
/// one: neither relayed, counted, kept nor dying. A kind the runtime announces once per device
/// is announced at every occurrence where a probe may meet it ([`crate::platform`]'s
/// `announced_each`), so that the probe's takes no later occurrence's line.
#[cfg(target_os = "linux")]
pub(crate) fn unheard<T>(call: impl FnOnce() -> T) -> T {
    /// Puts back the outer state on drop, an unwinding call's included.
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            UNHEARD.with(|u| u.set(self.0));
        }
    }
    let _restore = Restore(UNHEARD.with(|u| u.replace(true)));
    call()
}

/// Runs `call`, a runtime call standing for `sel` (none for [`plain`]'s), and dies at `sel`, or
/// at the notice's code, where the runtime announced a degradation of [`DIES`] inside it, or on
/// this thread before it ([`notice`]). A call that unwinds, a death inside it, lets its notice
/// go: the death that unwinds it is the call's, and the thread's next call here must not die of
/// a notice of this one.
#[inline]
fn inside<T>(sel: Option<&'static str>, call: impl FnOnce() -> T) -> T {
    // The call runs inside the one lookup, which a nested call here makes again.
    CALLS.with(|calls| inside_of(calls, sel, call))
}

/// [`inside`], of this thread's `calls`.
#[inline]
fn inside_of<T>(calls: &Calls, sel: Option<&'static str>, call: impl FnOnce() -> T) -> T {
    /// Leaves the call on drop, an unwinding one included, whose notice it drops.
    struct Leave<'a> {
        calls: &'a Calls,
        /// Whether the call returned.
        returned: bool,
    }
    impl Drop for Leave<'_> {
        fn drop(&mut self) {
            self.calls.depth.set(self.calls.depth.get() - 1);
            if !self.returned {
                drop(take_pending(self.calls));
            }
        }
    }
    calls.depth.set(calls.depth.get() + 1);
    let out = {
        let mut leave = Leave {
            calls,
            returned: false,
        };
        let out = call();
        leave.returned = true;
        out
    };
    if let Some(notice) = take_pending(calls) {
        noticed_inside(sel, &notice);
    }
    out
}

/// A run of runtime calls this thread makes one after another ([`run`]): this thread's calls here,
/// looked up once for them all, and the list of the hearing around them.
pub(crate) struct Run<'a> {
    calls: &'a Calls,
    /// The list of the run's hearing ([`HEARING`]), on [`run`]'s stack.
    heard: NonNull<Vec<rt::Degradation>>,
}

/// Runs `run`, whose runtime calls go through [`Run::at`]: each as [`unstopped_at`] runs it, and
/// what the runtime announces inside each heard as by a [`hearing`] around them all, for the run
/// to read between them ([`Run::heard`], [`Run::heard_since`]). A hearing inside the run hears
/// what is announced inside it, as one inside another does.
pub(crate) fn run<T>(run: impl FnOnce(&Run<'_>) -> T) -> T {
    /// Puts back the outer hearing on drop, an unwinding run's included.
    struct Restore(Option<NonNull<Vec<rt::Degradation>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            HEARING.with(|h| h.set(self.0));
        }
    }
    let mut heard = Vec::new();
    let list = NonNull::from(&mut heard);
    let _outer = Restore(HEARING.with(|h| h.replace(Some(list))));
    CALLS.with(|calls| run(&Run { calls, heard: list }))
}

impl Run<'_> {
    /// Runs `call`, the runtime's form of `sel`, as [`unstopped_at`] does.
    #[inline]
    pub(crate) fn at<T>(&self, sel: &'static str, call: impl FnOnce() -> T) -> T {
        inside_of(self.calls, Some(sel), call)
    }

    /// How many kinds of [`HEARD`] the runtime announced inside the run's calls so far.
    #[inline]
    pub(crate) fn heard(&self) -> usize {
        // SAFETY: the run's list, on `run`'s stack for the run's life, which [`notice`] writes
        // only inside a call on this thread, and none runs while the run reads it.
        unsafe { self.heard.as_ref() }.len()
    }

    /// Whether the runtime announced `kind` inside the run's calls after the first `at` of what
    /// it announced ([`Self::heard`]).
    pub(crate) fn heard_since(&self, at: usize, kind: rt::Degradation) -> bool {
        // SAFETY: as in [`Self::heard`].
        let heard = unsafe { self.heard.as_ref() };
        heard.get(at..).is_some_and(|since| since.contains(&kind))
    }
}

/// What [`inside`] does with `notice`, a degradation of [`DIES`] announced inside its call of
/// `sel`: out of line, as nearly every call announces none.
#[cold]
#[inline(never)]
fn noticed_inside(sel: Option<&'static str>, notice: &rt::DegradeNotice) {
    let call = sel.unwrap_or(notice.degradation.code());
    if census::on() {
        census::selector(notice.degradation, call);
    } else {
        degraded(call, notice);
    }
}

/// The kinds of [`DIES`] that die by default too, not only in strict mode, as the guest or the
/// host would otherwise go on past a break of what it relies on, which no log line repairs:
/// - `ReleasedCommandState`: a command whose record the GPU wrote naming a released state whose
///   program the buffer does not keep runs nothing, and one naming the ID of a released state the
///   CPU set, which another state held while the GPU could write the record, runs the set state's
///   program, where Metal runs the state the GPU's write named in both;
/// - `PrivateResourceOption`: the GPU's writes reach the guest's pages, which Metal's never do
///   for such a resource (announced off macOS for no-copy memory, as `DIES` says);
/// - `HostDeallocatorDelayed`: the deallocator, which the host sees as the release, runs after
///   the release, out of the host's release order.
pub const DIES_BY_DEFAULT: [rt::Degradation; 3] = [
    rt::Degradation::ReleasedCommandState,
    rt::Degradation::PrivateResourceOption,
    rt::Degradation::HostDeallocatorDelayed,
];

/// What [`notice`] does with a degradation of [`DIES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fate {
    /// Strict mode ([`strict`]), or a kind of [`DIES_BY_DEFAULT`]: it dies at the call it was
    /// announced inside, or where it was announced outside every call.
    Dies,
    /// A native device relays it ([`native::NATIVE_RELAYED`]): counted, the first relayed.
    RelayedNatively,
    /// Census mode records it ([`census`]), with the call it was announced inside, and relays it.
    Counted,
    /// The default: counted, the first relayed ([`relayed_counts`]), and the process goes on.
    Relayed,
}

/// The fate of `kind`, a kind of [`DIES`], under `placement`, with census mode and strict mode
/// each on or off.
fn fate(kind: rt::Degradation, placement: native::Placement, census: bool, strict: bool) -> Fate {
    if placement == native::Placement::Native && native::relayed(kind) {
        Fate::RelayedNatively
    } else if census {
        Fate::Counted
    } else if strict || DIES_BY_DEFAULT.contains(&kind) {
        Fate::Dies
    } else {
        Fate::Relayed
    }
}

/// Whether every occurrence of every degradation is written to the trace
/// (`REIMS_VGPU_TRACE_DEGRADE=1`, read once per process, with `REIMS_VGPU_TRACE` and its file
/// open): a diagnostic, which only the sink's relaying and the trace see. The runtime announces
/// a kind asked for at every occurrence only in a call of the program's or a command buffer's
/// host step, and a kind it meets elsewhere once, whatever it is asked
/// (`DeviceOptions::announce_each`). A breach of that rule is the runtime's: in a build with
/// debug assertions it ends the process there (its `MISPLACED_ENDS`), and in other builds, the
/// dist's among them, the notice reaches the sink from the runtime's notice thread, with a
/// `misplaced-notice` after it.
pub fn trace_degradations() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("REIMS_VGPU_TRACE_DEGRADE").is_some_and(|v| v == "1")
    });
    *ON
}

/// The kinds the trace saw, which the runtime announces once without it.
static TRACED_KINDS: reims_vgpu_core::sync::Mutex<Vec<rt::Degradation>> =
    reims_vgpu_core::sync::Mutex::new(Vec::new());

/// The runtime's message sink (`DeviceOptions::messages`, [`crate::platform`]): a degradation
/// of [`DIES`] is relayed and counted, or in strict mode kept for the call here it was announced
/// inside to die at, or, announced outside every call here, dies now, ending the process (the
/// module's docs), but where a native device relays it or census mode records it ([`Fate`]); one
/// of [`HEARD`] is kept for the
/// [`hearing`] it was announced inside; every other message, and every one of [`HEARD`] too,
/// goes to [`relay`], to stderr (a `Verbose` one of the runtime's own to the log at debug), but
/// a kind of [`RELAYED_ONCE`] after its first.
pub fn notice(message: &rt::DebugMessage) {
    let Some(notice) = &message.degradation else {
        relay(message);
        return;
    };
    if UNHEARD.with(Cell::get) {
        return;
    }
    if trace_degradations() && reims_vgpu_core::trace::on() {
        let kind = notice.degradation;
        // deg: an occurrence of a degradation, its code
        reims_vgpu_core::trace::event_text("deg", &[], kind.code());
        // A kind announced once without the trace goes on only at its first occurrence.
        if !DIES.contains(&kind) && !HEARD.contains(&kind) {
            let mut seen = TRACED_KINDS.lock();
            if seen.contains(&kind) {
                return;
            }
            seen.push(kind);
        }
    }
    if !DIES.contains(&notice.degradation) {
        if HEARD.contains(&notice.degradation) {
            hear(notice.degradation);
        }
        #[cfg(test)]
        RELAYED.lock().push(notice.degradation);
        if first_relayed_once(notice.degradation) == Some(false) {
            return;
        }
        relay(message);
        return;
    }
    let kind = notice.degradation;
    match fate(kind, native::placement(), census::on(), strict()) {
        Fate::Dies => {}
        Fate::Relayed => {
            if count_relayed(kind) {
                relay(message);
            }
            return;
        }
        Fate::RelayedNatively => {
            census::record(kind, &notice.object, true);
            if native::count(kind) {
                relay(message);
            }
            return;
        }
        Fate::Counted => {
            if census::record(kind, &notice.object, false) {
                relay(message);
            }
            if depth() == 0 {
                census::selector(kind, census::OUTSIDE);
            } else {
                keep_pending(notice);
            }
            return;
        }
    }
    // Kept first: the death below ends the process, but the test capture's unwinds into the
    // runtime, which catches it, and the thread then dies at its next call here.
    keep_pending(notice);
    if depth() == 0 {
        degraded(notice.degradation.code(), notice);
    }
}

/// Keeps `kind`, of [`HEARD`], for the innermost [`hearing`] or [`run`] this thread runs, where it
/// runs one ([`notice`]).
fn hear(kind: rt::Degradation) {
    if let Some(mut heard) = HEARING.with(Cell::get) {
        // SAFETY: the list of the innermost hearing or run this thread runs, on its stack until it
        // puts back the one outside it, and read by it only between its calls.
        unsafe { heard.as_mut() }.push(kind);
    }
}

/// The degradations not of [`DIES`] the runtime announced in this process, in order, which
/// [`notice`] relays and a test reads.
#[cfg(test)]
pub(crate) static RELAYED: reims_vgpu_core::sync::Mutex<Vec<rt::Degradation>> =
    reims_vgpu_core::sync::Mutex::new(Vec::new());

/// The least severity the sink hears of the messages that are no degradation
/// (`MessageSink::with_least_severity`): `Verbose` where the log takes this module's debug lines,
/// which [`relay_to`] writes such a message to, else `Info`. The runtime then formats none of the
/// `Verbose` lines of its own that would go nowhere: a host import freed, on a device that keeps
/// MoltenVK's residency set, comes at every buffer's or texture's release over the guest's pages.
/// Its degrade notices reach the sink whatever the severity.
pub(crate) fn least_severity() -> rt::MessageSeverity {
    match log::log_enabled!(log::Level::Debug) {
        true => rt::MessageSeverity::Verbose,
        false => rt::MessageSeverity::Info,
    }
}

/// What the id names of the runtime's own messages start with: its degradations'
/// (`metal2vulkan-degrade-`), its pipeline-creation log's (`metal2vulkan-pipeline-creation`) and
/// its other diagnostics' (`metal2vulkan-late-reservation`, `metal2vulkan-host-import-freed`, the
/// shader cache's `metal2vulkan-shader-cache-evicted` and
/// `metal2vulkan-shader-cache-write-failed`, each said once per device, and
/// `metal2vulkan-unmeasured-driver`, said as a device whose driver, GPU and release the runtime
/// has not measured is made).
const RUNTIME_MESSAGES: &str = "metal2vulkan-";

/// Writes `message`, which the runtime gave the sink, on stderr ([`relay_to`]).
fn relay(message: &rt::DebugMessage) {
    relay_to(
        &mut std::io::stderr(),
        message.severity,
        &message.id_name,
        &message.text,
    );
}

/// Writes `text`, a message of `severity` and id name `id_name` the runtime gave the sink, on `w`
/// as the runtime writes it on stderr without a sink, one line, unchanged, where it is one of the
/// runtime's own ([`RUNTIME_MESSAGES`]). One of its own of severity `Verbose`, which the runtime
/// gives a sink alone, and only one that hears it ([`least_severity`]), goes to the log at debug
/// instead, as the library's other diagnostics do: the runtime's line for each host import freed
/// on a device that keeps MoltenVK's residency set (`metal2vulkan-host-import-freed`; this host's
/// MoltenVK keeps it). A driver's or a layer's (`VK_EXT_debug_utils`), which the runtime reports
/// to a sink alone, is written only at severity `Error`, after its id name: MoltenVK's reason for
/// a refusal the runtime answers with a bare `VkResult` (`mvk-error`, a shader the Metal compiler
/// did not take, say) reaches stderr with nothing else; one of a lower severity is dropped. The
/// runtime hands the sink none of the driver's messages its own self-checks provoke.
fn relay_to(w: &mut impl std::io::Write, severity: rt::MessageSeverity, id_name: &str, text: &str) {
    if !id_name.starts_with(RUNTIME_MESSAGES) {
        if severity == rt::MessageSeverity::Error {
            // One write, so the line never interleaves with another thread's.
            let _ = w.write_all(format!("{id_name}: {text}\n").as_bytes());
        }
        return;
    }
    if severity == rt::MessageSeverity::Verbose {
        log::debug!("{text}");
        return;
    }
    let mut line = text.to_owned();
    line.push('\n');
    // One write, so the line never interleaves with another thread's.
    let _ = w.write_all(line.as_bytes());
}

/// The death at `call` of a degradation of [`DIES`], its line naming the notice.
fn degraded(call: &'static str, notice: &rt::DegradeNotice) -> ! {
    dies(
        call,
        &format!(
            "the Vulkan runtime degraded it ({}): {}: {}; {}",
            notice.degradation.code(),
            notice.object,
            notice.reason,
            notice.cost
        ),
    )
}

/// The runtime's refusal `e` of `sel`, decided: what Metal's validation stops (`Invalid`) by the
/// call's label, anything else, and anything at a call labelled `Dies`, a death here
/// ([`refused`]). A pipeline state of an unspecialized function (`UnspecializedFunction`) dies
/// whatever the label, as plain Metal aborts at each pipeline creator for one.
pub fn decide<T>(sel: &'static str, policy: Policy, e: Error) -> RaiseResult<T> {
    decide_describing(sel, policy, e, None)
}

/// [`decide`], the line of a death or a raise also carrying `descriptor`
/// ([`refusal_describing`]): what the refused call's descriptor holds that the runtime's error
/// does not say, a pipeline creator's formats (`gpu::states::formats`).
pub fn decide_describing<T>(
    sel: &'static str,
    policy: Policy,
    e: Error,
    descriptor: Option<&str>,
) -> RaiseResult<T> {
    let line = refusal_describing(&e, descriptor);
    match e {
        Error::Invalid(ref i) if matches!(i.problem, rt::Problem::UnspecializedFunction { .. }) => {
            dies(sel, &line)
        }
        Error::Invalid(_) if policy != Policy::Dies => {
            policy.on().throw(Raise::platform(sel).with_detail(line))
        }
        _ => dies(sel, &line),
    }
}

/// The domain of every `NSError` this platform answers but a mesh render pipeline's of a stage
/// function of another stage's type, which is the compiler's, as on the Metal platform
/// (`crate::gpu::states::wrong_mesh_stage`): Metal's is its driver's name, which differs between
/// GPU families, and only core's log line reads it ([`crate::gpu::states`]' module docs).
pub const RUNTIME_DOMAIN: &str = "metal2vulkan-runtime";

/// The runtime's error as the `NSError` Metal hands back for a creator with an `error:`
/// out-parameter: the runtime's domain ([`RUNTIME_DOMAIN`]), no code of Metal's (0), and its
/// text.
pub fn error_text(e: &Error) -> NsErrorText {
    NsErrorText {
        domain: RUNTIME_DOMAIN.to_owned(),
        code: 0,
        description: e.to_string(),
    }
}

/// The death of a send the runtime refuses where Metal goes on: named by the selector, the line
/// carrying the runtime's error ([`refusal`]).
pub fn refused(sel: &'static str, e: &Error) -> ! {
    dies(sel, &refusal(e))
}

/// The line of the last death [`dies`] made in this process, which a test reads: the test
/// capture takes a death before its line is written.
#[cfg(test)]
pub(crate) static LAST_LINE: reims_vgpu_core::sync::Mutex<Option<String>> =
    reims_vgpu_core::sync::Mutex::new(None);

/// The death at `sel` of [`refused`], [`metal_aborts`] and [`unrepresentable`], its line
/// `line`.
fn dies(sel: &'static str, line: &str) -> ! {
    #[cfg(test)]
    {
        *LAST_LINE.lock() = Some(line.to_owned());
    }
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        line,
    )
}

/// The line of [`refused`]: the runtime's error, and where it is a command sampling a texture whose
/// address modes are mapped (one of the two-plane YCbCr family the runtime makes over a surface,
/// [`crate::gpu::ycbcr::runtime_makes`], whose modes it maps as Metal addresses the format:
/// `YCBCR8_420_2P`'s repeat and mirror modes read as clamp-to-edge), which of the paths the runtime
/// cannot map it is: a sampler slot shared with a texture of another map, sampling through
/// handles (an argument buffer's or a heap's), or a constexpr sampler whose modes the runtime
/// cannot map, which it maps where its modes are stated and the map sends each to one a constexpr
/// sampler addresses, and refuses otherwise; and where it is a refusal of SIMD-group operations
/// in a stage, the stage and the class ([`simd_stage`]); and where it is a refusal of host memory
/// after the import's dma-buf source made none, why it made none ([`host_memory_refusal`]).
pub fn refusal(e: &Error) -> String {
    use metal2vulkan_runtime::{MappedSampling, RuntimeGap};
    let path = match e {
        Error::NotImplemented(RuntimeGap::MappedTextureSampling { through, .. }) => match through {
            MappedSampling::ConstexprSampler => {
                Some("through a constexpr sampler whose address modes the map changes")
            }
            MappedSampling::SharedSamplerSlot => Some(
                "through a sampler slot that samples, in the same stage, a texture the map \
                 addresses otherwise",
            ),
            MappedSampling::Handles => Some(
                "by a function that samples through handles (an argument buffer's or a heap's) \
                 able to reach it",
            ),
            _ => Some("where the runtime cannot map its sampler"),
        },
        _ => None,
    };
    let named = path
        .map(|path| {
            format!(
                "a texture whose address modes are mapped to Metal's format's (a member of \
                 gpu::ycbcr's family) is sampled {path}"
            )
        })
        .or_else(|| simd_stage(e))
        .or_else(|| host_memory_refusal(e));
    match named {
        Some(named) => format!("the Vulkan runtime refused it: {e}; {named}"),
        None => format!("the Vulkan runtime refused it: {e}"),
    }
}

/// What a refusal of host memory names, for [`refusal`]: why the import's dma-buf source on this
/// thread made no dma-buf of its pages (`crate::os::host_memory`), such as udmabuf's limits, which
/// the device would have imported where it refuses the pages as host memory. `None` for any other
/// refusal, and where the source made one or was not asked.
fn host_memory_refusal(e: &Error) -> Option<String> {
    use metal2vulkan::error::Feature;
    match e {
        Error::Unsupported(u) if u.feature == Feature::HostMemory => crate::os::dma_buf_refusal(),
        _ => None,
    }
}

/// What a refusal of SIMD-group or quad-group operations names, for [`refusal`]: the stage the
/// runtime does not take them in on this device, and the class. Device info's key 33 answers
/// the SIMD reduction and shuffle-and-fill bits no where the runtime does not take every function
/// they promise in every stage ([`crate::gpu::caps`]), but a guest may still use the class in
/// such a stage, which Metal runs: it reaches this refusal where its pipeline state is made, and
/// dies there. The line keys on the refusal, never on the stage: where the runtime takes the
/// class in a stage, no refusal comes. `None` for any other refusal.
fn simd_stage(e: &Error) -> Option<String> {
    use metal2vulkan::device::VkProperty;
    use metal2vulkan::error::{Capability, Feature, Needs};
    use metal2vulkan::pipeline::Stage;
    let Error::Unsupported(u) = e else {
        return None;
    };
    let class = match u.feature {
        Feature::SubgroupOperations => "SIMD-group operations",
        Feature::QuadOperations => "quad-group operations",
        _ => return None,
    };
    let Needs::AnyOf(alternatives) = &u.needs else {
        return None;
    };
    let mut stage = None;
    let mut ops = Vec::new();
    for capability in alternatives.iter().flatten() {
        match capability {
            Capability::Property(VkProperty::SubgroupStage(s)) => stage = Some(*s),
            // Quad operations outside the fragment and kernel stages are a vertex function's.
            Capability::Property(VkProperty::SubgroupQuadOperationsInAllStages) => {
                stage = Some(Stage::Vertex)
            }
            Capability::Property(VkProperty::SubgroupOperation(op)) => ops.push(format!("{op:?}")),
            _ => {}
        }
    }
    let whose = match stage {
        Some(Stage::Vertex) => "a vertex function's",
        Some(Stage::Fragment) => "a fragment function's",
        Some(Stage::Kernel) => "a kernel's",
        Some(Stage::Object) => "an object function's",
        Some(Stage::Mesh) => "a mesh function's",
        Some(_) | None => "a function's",
    };
    let of = if ops.is_empty() {
        String::new()
    } else {
        format!(" (the classes {})", ops.join(", "))
    };
    Some(format!(
        "{whose} {class}{of}, which the runtime does not take in its stage on this device, \
         where Metal runs them (device info's key 33 answers no SIMD bits for a class some \
         stage lacks: gpu::caps)"
    ))
}

/// [`refusal`]'s line, then `descriptor`, what the refused call's descriptor holds, after a
/// semicolon.
pub fn refusal_describing(e: &Error, descriptor: Option<&str>) -> String {
    match descriptor {
        Some(descriptor) => format!("{}; {descriptor}", refusal(e)),
        None => refusal(e),
    }
}

/// The death of a method this platform has not built yet: its line names the selector.
pub fn unbuilt(sel: &'static str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: sel,
        },
        "not built on the Vulkan platform",
    )
}

/// The death of a send Metal itself aborts at (an assertion inside Metal, whatever handler is
/// live), where the runtime would refuse or go on: `what` says what Metal stops.
pub fn metal_aborts(sel: &'static str, what: &str) -> ! {
    dies(sel, &format!("Metal aborts at {what}"))
}

/// The death of a send at which Metal reads `what` at a NULL or near-NULL address, in the zero
/// region every arm64 macOS executable maps (`__PAGEZERO`, below core's `ZERO_REGION_END`), where
/// the read ends the process with a segmentation fault.
pub fn metal_reads_near_null(sel: &'static str, what: &str) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::NullDeref,
            call: sel,
        },
        &format!("Metal reads {what} in the zero region (a segmentation fault)"),
    )
}

/// The death of a send whose raw value the runtime's types cannot hold: `what` names it (a pixel
/// format no `MTLPixelFormat` is, an option bit the runtime does not define). Metal itself aborts
/// for some of these (a pixel format it refuses); for the others the runtime has no form.
pub fn unrepresentable(sel: &'static str, what: &str) -> ! {
    dies(sel, &format!("the Vulkan runtime cannot hold {what}"))
}

/// A value this platform computes otherwise than Metal, or an answer it gives otherwise, which no
/// runtime notice of [`DIES`] announces: relayed by default, a death in strict mode ([`inexact`],
/// [`relay_gap`]). Each kind is placed once, under a code of its own
/// (`every_inexact_kind_is_placed_once`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inexact {
    /// A `half` colour output stored past the bound its rounding was measured within, as blended
    /// or converted at its attachment (`HalfOutputPrecision`, `crate::gpu::encode_render`'s
    /// `half_apart`).
    HalfOutputBlend,
    /// A bfloat attachment loaded and stored by a pass that draws, whose codes the runtime moves
    /// through `half` otherwise than Metal's tile store.
    BfloatLoaded,
    /// A colour attachment write under `textureWriteRoundingMode` 2 the runtime is not measured
    /// to store as Metal does: a resolve at a sample count nothing measured.
    WriteRounding,
    /// A heap texture's size, alignment or bytes in the heap the runtime answers or lays out as
    /// its device's image, where Metal's are not measured or are its own: a texture is made, and
    /// its size answered, by the runtime's.
    HeapLayout,
    /// A draw or tile dispatch in a pass of a rasterization rate map whose rates vary, or a tile
    /// dispatch in one of no layers, which the runtime runs by its own regions and tiles.
    RateMap,
    /// A colour fill packed in a pixel format other than the texture's, whose bytes the runtime
    /// repeats where Metal's follow no rule measured.
    FillFormat,
    /// A late resolve texture beside a resolve the pass dropped, a shape Metal is not measured
    /// at: the dropped resolve stays dropped, and a depth or stencil resolve goes on.
    LateResolve,
    /// A texture of protection options other than none, made by `newTextureWithDescriptor:` or
    /// `newSharedTextureWithDescriptor:` as without them, as Metal's other creators make one
    /// (it answers protection options 0), where Metal's two answer nil to a process without the
    /// protected-content entitlement, QEMU among them ([`crate::gpu::desc`]). Nothing here holds
    /// protected memory, so its contents are not kept from the host.
    ProtectedContent,
    /// The `uniqueIdentifier` the guest reads (info command 460) of a pipeline state made with
    /// indirect command buffer support: the runtime's, `2^36` plus an offset of its own
    /// allocation of such states, where Metal's is an address of its allocation that no measured
    /// rule gives (`unmeasured-unique-identifier`). The runtime keeps the relations a guest can
    /// rely on, as Metal's: one value for the state's life, no other live state's, the last
    /// released taken again first by kind (`crate::gpu::resource`'s `identified`).
    UniqueIdentifier,
    /// The `uniqueIdentifier` the guest reads (info command 460) of an indirect command buffer
    /// made after encoders whose take of Metal's range the runtime does not model
    /// (`identifier_unmeasured`: after a command buffer that encoded a render encoder was
    /// scheduled, whose pass's run takes more, after a resource state encoder opened, or once
    /// Metal's encoders' ranges passed what it models): the runtime's
    /// (`unmeasured-unique-identifier`, which the runtime announces once a device, as such a
    /// buffer is read). Relayed at every such buffer, wider than the runtime's notice
    /// ([`crate::gpu::icb`]).
    IcbIdentifierAfterEncoder,
    /// A threadgroup memory length of a compute indirect command buffer past its last command's,
    /// a word of a command's pipeline, dispatch or imageblock size or of a copy, that Metal wrote
    /// otherwise than the runtime, or may have, until core sets it again: after a setter of the
    /// command that writes it, a copy into the command, a blit's copy or reset over a buffer that
    /// inherits its pipeline state, and in a buffer made resident for writing, so the getter
    /// answers the runtime's ([`crate::gpu::icb`]'s `Tail`).
    IcbLengthPastTheLastCommand,
    /// A threadgroup memory length 36 or 37 of a command of a compute indirect command buffer
    /// that does not inherit its pipeline state, from the encoding of a copy into the command
    /// on, until core sets it or a reset clears it: the runtime's copy clears both, where
    /// Metal's carries them from a command that names no pipeline state, so the getter answers
    /// the runtime's 0 ([`crate::gpu::icb`]'s `Sides::copied_lengths`).
    IcbCopiedLength,
    /// An object a creator asked for that the runtime refuses for a gap of its own or of the
    /// device's, where Metal makes it ([`gap`]): answered nil, a pipeline state or a library with
    /// the runtime's refusal as its `NSError` ([`relay_gap`]). The guest's creation fails, as
    /// where Metal answers nil, and the work that names the object faults.
    RefusedCreate,
}

impl Inexact {
    /// Every kind, once.
    pub const ALL: [Self; 13] = [
        Self::HalfOutputBlend,
        Self::BfloatLoaded,
        Self::WriteRounding,
        Self::HeapLayout,
        Self::RateMap,
        Self::FillFormat,
        Self::LateResolve,
        Self::ProtectedContent,
        Self::UniqueIdentifier,
        Self::IcbIdentifierAfterEncoder,
        Self::IcbLengthPastTheLastCommand,
        Self::IcbCopiedLength,
        Self::RefusedCreate,
    ];

    /// The kind's code, which its line names.
    pub const fn code(self) -> &'static str {
        match self {
            Self::HalfOutputBlend => "inexact-half-output-blend",
            Self::BfloatLoaded => "inexact-bfloat-loaded",
            Self::WriteRounding => "inexact-write-rounding",
            Self::HeapLayout => "inexact-heap-layout",
            Self::RateMap => "inexact-rate-map",
            Self::FillFormat => "inexact-fill-format",
            Self::LateResolve => "inexact-late-resolve",
            Self::ProtectedContent => "inexact-protected-content",
            Self::UniqueIdentifier => "inexact-unique-identifier",
            Self::IcbIdentifierAfterEncoder => "inexact-icb-identifier-after-encoder",
            Self::IcbLengthPastTheLastCommand => "inexact-icb-length-past-last-command",
            Self::IcbCopiedLength => "inexact-icb-copied-length",
            Self::RefusedCreate => "inexact-refused-create",
        }
    }
}

/// How many times this process relayed each kind of [`Inexact`], in [`Inexact::ALL`]'s order.
static RELAYED_INEXACT: [std::sync::atomic::AtomicU64; Inexact::ALL.len()] =
    [const { std::sync::atomic::AtomicU64::new(0) }; Inexact::ALL.len()];

/// A value of kind `kind` at `sel` that the runtime computes otherwise than Metal, `what` saying
/// which: in strict mode the death of [`unrepresentable`], its line naming the kind's code, as a
/// degradation's death names its; by default relayed, the process's first of the kind writing a
/// line on stderr, and counted ([`inexact_counts`]). `what` is asked only for a line, the death's
/// or the first relay's, so a relay past the first of its kind, as a draw's can be at every draw
/// of a pass, makes none.
pub fn inexact(sel: &'static str, kind: Inexact, what: impl FnOnce() -> String) {
    if strict() {
        unrepresentable(sel, &format!("{} ({})", what(), kind.code()));
    }
    if counted_first(kind) {
        let line = format!(
            "reims-vgpu: {sel}: the Vulkan runtime cannot hold {}; relayed ({}), the first of \
             its kind: REIMS_VGPU_STRICT=1 makes it a death\n",
            what(),
            kind.code()
        );
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
    }
}

/// Counts one relay of `kind` ([`inexact_counts`]): whether it is the process's first of the
/// kind.
fn counted_first(kind: Inexact) -> bool {
    use std::sync::atomic::Ordering;
    let at = Inexact::ALL.iter().position(|&k| k == kind);
    (at.and_then(|at| RELAYED_INEXACT.get(at)))
        .is_some_and(|c| c.fetch_add(1, Ordering::Relaxed) == 0)
}

/// Whether the runtime's refusal `e` of an object is a gap of its own or of the device's, where
/// Metal makes the object: a feature the device or the runtime lacks (`Unsupported`, and a
/// compile's `Unsupported`), but one Metal itself refuses by a GPU family the device lacks or on
/// the devices it was measured to refuse it on (`Needs::MetalGpuFamily`, `Needs::MetalRefuses`),
/// which are Metal's refusals; a piece the runtime has not built (`NotImplemented`); and an
/// Apple-only call it refuses (`AppleOnly`). A lost device, a Vulkan failure, no memory, an
/// answer the runtime cannot prove ([`Error::Unproven`]), a misuse Metal refuses and the rest
/// are not.
pub fn gap(e: &Error) -> bool {
    use metal2vulkan::error::{Error as Compile, Needs};
    let metal_s =
        |needs: &Needs| matches!(needs, Needs::MetalGpuFamily(_) | Needs::MetalRefuses(_));
    match e {
        Error::Unsupported(u) | Error::Compile(Compile::Unsupported(u)) => !metal_s(&u.needs),
        Error::NotImplemented(_) | Error::AppleOnly(_) => true,
        _ => false,
    }
}

/// Whether a creator relays the runtime's refusal `e` ([`relay_gap`]): a [`gap`], outside strict
/// mode. A step that would die at the refusal before the creator decides it (the death of a
/// pixel format the runtime refuses, [`crate::gpu::desc::format_refused`]) leaves such a
/// refusal to the creator.
pub fn relays_gap(e: &Error) -> bool {
    gap(e) && !strict()
}

/// Whether the creator `sel` answers nil for the runtime's refusal `e` ([`relays_gap`]): a gap of
/// the runtime's or the device's where Metal makes the object ([`gap`]), relayed as
/// [`Inexact::RefusedCreate`], counted ([`inexact_counts`]) and the process's first of the kind
/// writing a line on stderr naming the refusal and `descriptor` ([`refusal_describing`]). The
/// creator then answers nil, a pipeline or library creator with [`error_text`] as its
/// `NSError`, and the relay is noted for core on the creator's thread
/// ([`reims_vgpu_core::platform::device::note_relayed_gap`]), which faults the work that names a
/// function so refused where Metal would end the process at the nil one create later. `false` in
/// strict mode and for every other refusal, which the creator decides as before: a death at
/// `sel`.
pub fn relay_gap(sel: &'static str, e: &Error, descriptor: Option<&str>) -> bool {
    if !relays_gap(e) {
        return false;
    }
    reims_vgpu_core::platform::device::note_relayed_gap();
    if counted_first(Inexact::RefusedCreate) {
        let line = format!(
            "reims-vgpu: {sel}: {}; answered nil, relayed ({}), the first of its kind: \
             REIMS_VGPU_STRICT=1 makes it a death\n",
            refusal_describing(e, descriptor),
            Inexact::RefusedCreate.code()
        );
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
    }
    true
}

/// How many times this process answered a call as when memory runs out ([`out_of_host_memory`]).
static OUT_OF_HOST_MEMORY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A map, a file or memory that this platform's host refused at `call` for want of mappings
/// (`vm.max_map_count`), of open files or of memory, where Metal's call needs none: in strict
/// mode the process ends there, naming `detail`, as it did before; by default the caller answers
/// as Metal's call does when memory runs out (`KERN_NO_SPACE`, no memory, a nil object), and the
/// answer is counted ([`out_of_host_memory_count`]) and logged at the first and at each power of
/// two. `detail` is asked only for a line.
pub(crate) fn out_of_host_memory(call: &'static str, detail: impl FnOnce() -> String) {
    use std::sync::atomic::Ordering;
    if strict() {
        fatal_with_detail(
            DeathId::Platform {
                kind: DeathKind::Abort,
                call,
            },
            &detail(),
        );
    }
    let count = OUT_OF_HOST_MEMORY.fetch_add(1, Ordering::Relaxed) + 1;
    if count.is_power_of_two() {
        log::warn!(
            target: "reims_vgpu_vulkan::os",
            "{count} calls answered as when memory runs out, the latest at {call}: {}; \
             REIMS_VGPU_STRICT=1 makes each a death",
            detail()
        );
    }
}

/// How many times this process answered a call as when memory runs out ([`out_of_host_memory`]).
pub fn out_of_host_memory_count() -> u64 {
    OUT_OF_HOST_MEMORY.load(std::sync::atomic::Ordering::Relaxed)
}

/// Each kind of [`Inexact`] this process relayed, with how many times.
pub fn inexact_counts() -> Vec<(Inexact, u64)> {
    use std::sync::atomic::Ordering;
    (Inexact::ALL.iter().zip(&RELAYED_INEXACT))
        .map(|(&k, c)| (k, c.load(Ordering::Relaxed)))
        .filter(|&(_, n)| n > 0)
        .collect()
}

/// Parks this thread for good, where Metal's call never returns: a wait Metal never returns
/// from ([`crate::gpu::command`]), and a reservation Metal spins in
/// ([`crate::gpu::icb`]).
pub fn hang() -> ! {
    loop {
        std::thread::park();
    }
}

/// Runs `call`, a test's, which must die of a degradation of `kind` ([`DIES`]): the site it
/// died at, its line naming the kind's code. A call that returns, or dies of another, fails the
/// test.
#[cfg(test)]
pub(crate) fn dies_of<T>(
    kind: rt::Degradation,
    call: impl FnOnce() -> T,
) -> reims_vgpu_core::raise::death::DeathSite {
    *LAST_LINE.lock() = None;
    let Err(site) = reims_vgpu_core::testkit::capture::catch_death(call) else {
        panic!("it did not die, where {} dies", kind.code());
    };
    let line = LAST_LINE.lock().take().unwrap_or_default();
    assert!(
        line.contains(&format!("({})", kind.code())),
        "it died of another than {}: {line}",
        kind.code()
    );
    site
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan_runtime::{Invalid, Problem, RuntimeGap};
    use reims_vgpu_core::platform::device::relaying_gaps;
    use reims_vgpu_core::raise::{Origin, RaiseId};
    use reims_vgpu_core::testkit::capture::catch_death;

    fn invalid() -> Error {
        Error::Invalid(Invalid::new("fill_buffer", Problem::StorageMode))
    }

    /// The runtime's own messages that are no degradation, its pipeline-creation log, its late
    /// reservation's warning and its shader cache's warnings of an entry evicted and one not
    /// written, reach stderr's writer as the runtime writes them without a sink, a line each,
    /// unchanged, and so does a degradation it announces; its own `Verbose` line of a host import
    /// freed goes to the log alone; a layer's and a driver's error reaches it after its id name,
    /// and their messages of a lower severity are dropped
    #[test]
    fn the_runtimes_own_messages_are_relayed_unchanged() {
        use rt::MessageSeverity as S;
        let creation = "metal2vulkan: pipeline creation: state 3 \"blit\" pipeline 2: 4.1 ms";
        let late = "metal2vulkan: reserve() must be called before any allocation takes place";
        let degraded =
            "metal2vulkan: texture 7: its rows are copied; slower [metal2vulkan-degrade-x]";
        let freed = "metal2vulkan: a host import of 16384 bytes freed with 2 submissions in flight";
        let evicted = "metal2vulkan: the shader cache directory /c reached its cap of 26843545600";
        let unwritten = "metal2vulkan: an entry of the shader cache directory /c was not written";
        let mut out = Vec::new();
        for (severity, id_name, text) in [
            (S::Info, "metal2vulkan-pipeline-creation", creation),
            (
                S::Error,
                "VUID-RuntimeSpirv-None-06343",
                "Validation Error: [ VUID-RuntimeSpirv-None-06343 ]",
            ),
            (S::Warning, "metal2vulkan-late-reservation", late),
            (S::Verbose, "metal2vulkan-host-import-freed", freed),
            (S::Warning, "metal2vulkan-shader-cache-evicted", evicted),
            (
                S::Warning,
                "metal2vulkan-shader-cache-write-failed",
                unwritten,
            ),
            (S::Warning, "", "MoltenVK: a message of the driver's"),
            (
                S::Verbose,
                "",
                "MoltenVK: a verbose message of the driver's",
            ),
            (
                S::Error,
                "mvk-error",
                "VK_ERROR_INITIALIZATION_FAILED: Shader library compile failed (Error code 3):",
            ),
            (
                S::Warning,
                "metal2vulkan-degrade-shadowed-texture",
                degraded,
            ),
        ] {
            relay_to(&mut out, severity, id_name, text);
        }
        let validation = "VUID-RuntimeSpirv-None-06343: Validation Error: \
                          [ VUID-RuntimeSpirv-None-06343 ]";
        let compile = "mvk-error: VK_ERROR_INITIALIZATION_FAILED: Shader library compile \
                       failed (Error code 3):";
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format!(
                "{creation}\n{validation}\n{late}\n{evicted}\n{unwritten}\n{compile}\n{degraded}\n"
            )
        );
    }

    /// Every kind of degradation the runtime has is placed here once: it dies ([`DIES`]), a
    /// creator hears it ([`HEARD`]), or it is relayed, in one of the reviewed lists below. A kind
    /// the runtime adds fails this until it is placed, so none is relayed unreviewed
    #[test]
    fn every_degradation_kind_is_placed_once() {
        use rt::Degradation as D;
        // Metal's answer, at a cost Metal does not pay in time or memory (a performance kind), or
        // through a use of the device its driver does not advertise but runs as Metal does (a
        // validity kind, `UnadvertisedVertexSimdgroups` alone: a vertex function's simdgroup
        // functions): relayed, as the runtime states (`rt::Exactness::Exact`). Among them,
        // `SynthesizedGeometryStage` (a geometry stage that hands the fragment function its
        // barycentric weights, or a primitive's slice and viewport, where the device cannot).
        // Three performance kinds are Metal's answer while a pool of the command buffer holds
        // the work (`rt::Exactness::ExactWithinPool`): `IndirectIndexBoundsPass`,
        // `EmulatedMeshDraws` and `EmulatedTessellationDraws` (a tessellation pipeline's patch
        // draws on a device without `tessellationShader`), whose overflow fails the command
        // buffer, which dies where it is read (`crate::gpu::command`'s `pool_overflow`,
        // `mesh_records_overflow` and `tessellation_records_overflow`): the difference dies
        // where it shows.
        let exact = D::ALL.iter().copied().filter(|k| {
            matches!(
                k.exactness(),
                rt::Exactness::Exact | rt::Exactness::ExactWithinPool
            )
        });
        // One of Metal's answers where Metal's own is not one value, as the runtime states
        // (`rt::Exactness::OneOfMetals`), unless it dies here: relayed. Among them:
        // - `DontCareStore`: the texels a `DontCare` store leaves are the device's undefined ones
        //   here, where Apple's GPU keeps the earlier texels unless its tiler overflows, which no
        //   call sees coming. On MoltenVK 1.4.1 on the M3 Max, as on
        //   the M4, the runtime does not announce it (its `DontCareStoreIsMetals`):
        //   the device's stores there kept every kept case of its lane.
        // - `ZeroedIndexFetch`: an indexed draw from past its index buffer draws index 0 here,
        //   where Metal reads whatever memory follows the buffer.
        // - `RepeatedExecutionStart`: executions of indirect commands from a start an earlier one
        //   ran with another length each run their own range here, where Metal's run one with the
        //   other's length, differently from run to run (the runtime's probes on the M3 Max and
        //   the M4).
        // - `NarrowSurfaceRows`: a GPU read of a row past the first of a surface plane's texture
        //   under 16 bytes a row reads what the GPU's writes put there, where Metal's GPU reads
        //   memory-dependent bytes (the runtime's `linear_plan`: they differ between the M3 Max and
        //   the M4). The suite runs it twice (`textures_backing_records`, which reads no row past
        //   the first).
        // - `CollidingSurfaceRowWrites`: a kernel's writes through a texture of a surface plane
        //   whose rows share bytes on Metal's GPU (a `bytesPerRow` whose multiple of 16 below it is
        //   shorter than the row) leave the earlier row's bytes where two rows' writes meet, where
        //   Metal's leave whichever row its command wrote last, which moves with the threadgroup
        //   shape, the GPU pitch and the row count (the runtime's probe on the M3 Max): one of
        //   Metal's. A kernel is the only writer such a plane takes; the runtime refuses a blit
        //   into one, which dies at the blit.
        // - `OverlappingTextureCopyAsMemmove`: a blit copy between overlapping regions of one
        //   single-sampled texture's texels, which the runtime gives `memmove`'s answer through a
        //   scratch buffer. That is Metal's for regions of up to a few hundred texels (measured
        //   on the M3 Max and an M4); past that Metal's copy
        //   races, some texels reading what it already wrote, differently from run to run.
        //   Announced once per device. A multisampled texture's overlapping copy is still refused,
        //   which dies at the copy.
        // - `PlainStateUniqueIdentifier`: the `uniqueIdentifier` of a pipeline state made without
        //   indirect command buffer support, which Metal answers as an address that differs from
        //   run to run, and the runtime as one of a process-wide count. The guest reads it (info
        //   command 460, `crate::gpu::resource`'s `identified`): Metal's own answer is not one
        //   value.
        // - `AbsentArgumentRead`: a pipeline state whose function may read an argument its
        //   function constants removed, announced as it is made. The runtime takes zero for such a
        //   read and the command buffer completes, where Metal's compiler drops or folds the read
        //   and its debug layer reads zero.
        let one_of_metals = D::ALL
            .iter()
            .copied()
            .filter(|k| k.exactness() == rt::Exactness::OneOfMetals && !DIES.contains(k));
        // Among the exact kinds, `MisplacedNotice` (`misplaced-notice`): in a
        // build without debug assertions, a notice the runtime announced where its rule announces
        // none (inside a pipeline pool's piece, or a kind of `announce_each` outside the
        // program's call and outside a host step) reaches the sink unchanged from the runtime's
        // notice thread, after the call that met it, and this kind follows it there, naming the
        // kind and the place (`rt::DegradeNotice::misplaced`). Metal's answers are unchanged, as
        // the runtime states. Relayed: the notice it follows is placed by its own kind, outside
        // every call here, where a kind of `DIES` dies as it is announced and a kind of `HEARD`
        // reaches no hearing. A build with debug assertions, every test here, ends the process at
        // such a notice instead, with the runtime's line.
        // A recorded near-equivalent of Metal's answer (`rt::Exactness::NearEquivalent`), or work
        // the runtime fails instead (`FailsInstead`): relayed and logged, each for one of the
        // module's reasons, named below.
        // - `UncalibratedGpuTimes`, `UntimedCommandBuffer` and `GpuTimesFromHost`: a command
        //   buffer's GPU times, which only the graphics lane's guest sees, as the length of the
        //   buffer's run (its bundle's `GPUEndTime` less `GPUStartTime`, the end where the
        //   completion reached the guest), and debug lines read. `gpu_times_from_host`
        //   (`crate::platform`'s `device_options`) leaves the first two unreachable, and makes
        //   the run's length the time from the end of its submission, or the end of its queue's
        //   run before it, to the runtime's seeing it complete: within Metal's envelope of a
        //   run's length, which counts from the GPU's start. VM boots relay `GpuTimesFromHost`,
        //   which the runtime also takes on MoltenVK 1.4.1 whatever the option (its
        //   `TimestampsWaitForEveryQueue`).
        // - `UnmeasuredUniqueIdentifier`: the difference dies where it shows. The runtime
        //   announces it where a call answers an identifier it has not measured to be Metal's
        //   number: a state's with indirect command buffer support, which this platform reads as
        //   each state is made (`crate::gpu::resource`'s `identified`), a buffer's made after an
        //   encoder, and
        //   an indirect command's state identifier, which its read-back of the commands answers at
        //   every getter of a command (`IndirectCommandBuffer::commands`, [`crate::gpu::icb`]).
        //   The guest sees a command's only as the state core's map reaches by it, which the
        //   runtime's relations keep Metal's: one value for a state's life, no other live
        //   state's, the last released taken again first by kind. It reads a state's own (info
        //   command 460), which is relayed there as [`Inexact::UniqueIdentifier`], dying in strict
        //   mode, and a buffer's made after encoders whose take of Metal's range the runtime does
        //   not model, relayed as [`Inexact::IcbIdentifierAfterEncoder`].
        // Among the others:
        // - `DontCareLoadTiles` and `ExtendedRangeBitsKept`: the bytes a pass leaves where its
        //   draws write nothing, announced at a pass Metal runs. Metal's `DontCare` load zeroes
        //   every tile in a pass of one layer with a draw call that clears and stores an
        //   attachment, none in a pass whose draws draw nothing and that clears and stores none,
        //   and else the tiles a draw's primitive reaches within that draw's viewport and scissor
        //   rectangle, keeping the rest (an M4). The runtime takes the first two exactly, and in
        //   the third keeps the tiles outside its draws' viewports and scissors, clearing the
        //   tiles inside them that no primitive touches. Its tiles are the pass's, as Metal's grid
        //   is: its `tileWidth` and `tileHeight`, or the default for its samples. A pass at a
        //   sample count with no measured default takes every tile. A rotated frame, a colour held
        //   in sample planes, and a pass through a rate map, with an execution of render commands
        //   or with a tile dispatch, still take every tile. A tile it clears takes the texel a
        //   `Clear` of (0, 0, 0, 0) writes, an extended-range format's own zero, which reads 0, as
        //   Metal's. Pending a decision: the runtime's in-place load of the tiles no draw reaches
        //   is the exact form. A pass that loads and stores an extended-range, padded 10-bit or
        //   bfloat colour and draws keeps the bytes its draws do not write, the device's own load
        //   and store, where Metal's writes each tile a draw reaches back with the bits its codes
        //   leave unused made canonical and a bfloat code moved through `half`: each draw
        //   announces `ExtendedRangeBitsKept`. A bfloat attachment's codes are values, so a draw in
        //   a pass that loads and stores one dies first (`encode_render`'s `bfloat_loaded`).
        // - `DrawnDepthCopy`: a copy into a multisampled depth texture, or a pass's copies of a
        //   separate depth texture through its scratch image, which a fragment-depth draw writes
        //   where Metal copies the bytes: no copy of the device keeps a multisampled depth's
        //   samples (Vulkan copies a buffer into a single-sampled image alone, and a depth aspect
        //   to or from another format only under maintenance8, which MoltenVK 1.4.1 and NVIDIA's
        //   GA10x lack or get wrong), so a draw writes them through the fragment's depth. Measured
        //   equal: the runtime measured each of its four cases exact at 1 and 4 samples over a
        //   4-sample `Depth32Float_Stencil8` on the M4, the M3 Max's 4-sample row equal to the
        //   M4's. The runtime announces it at a copy from a buffer into a
        //   multisampled 32-bit float depth texture too, whose samples the same draw writes where
        //   the device's maintenance8 copies are missing or wrong (MoltenVK 1.4.1, NVIDIA's GA10x):
        //   every value Metal's but the edges, which Metal keeps bit for bit, where MoltenVK stores
        //   a value with the sign bit set (-0, a negative value, -inf, a negative NaN) as +0 and
        //   one past 1 or +inf as 1, a positive NaN writing nothing, and NVIDIA flushes denormals
        //   and stores NaN as -inf (the notice names which), and on a driver whose fragment depth
        //   is not measured (any other, and lavapipe releases not measured) the values Vulkan
        //   leaves to it (outside `[0, 1]`, denormals, -0) may land changed: the device forces
        //   those edges, as a fragment's depth reaches the attachment through the device's depth
        //   unit, which clamps and flushes them, and no other write of a multisampled depth exists
        //   there. Relayed on the rule for native devices: as close as the hardware allows, the
        //   rest logged.
        // - `UnnamedShadowedTexture`: a shadowed texture (its texels a buffer's, a surface's
        //   among them) a heap set holds, which a function reads through its ID as the texels an
        //   encoder that named it last copied in. Announced at the first command that binds a
        //   heap set holding its view while neither naming it nor making it resident, once per
        //   device, not at a read; a set no command binds announces nothing. The read Metal
        //   defines is by ID with the texture resident, and is exact: an
        //   encoder that names it (binds it, or makes it resident by `useResource` or by `useHeap`,
        //   a texture over a heap's buffer included) copies its texels in, and so does each later
        //   compute or render encoder of a command buffer that made it resident, as Metal's
        //   residency lasts the command buffer (the runtime's lanes against the M4's answers).
        //   Measured here: by ID through an argument buffer under `useResource`
        //   (`ycbcr::sampling`'s `the_guests_id_is_metals`, a 505 texture's reads and IDs as
        //   Metal's). What stays stale is outside Metal's residency rule: a read in an encoder
        //   before the first that makes the texture resident in its command buffer, which Metal
        //   orders against nothing either, and a read in a command buffer that never makes it
        //   resident, which on the M4 read the older texels, or zeros for a heap's buffer's
        //   texture. Metal's own answer there is not one value, though the runtime states
        //   the kind a near-equivalent.
        // - `DepthDenormals`: a write that may store a denormal in a 32-bit float depth (a
        //   pass's clear to one, a blit into a `Depth32Float` or `Depth32Float_Stencil8`
        //   texture's depth whatever bytes it copies, a `replace_region` whose bytes hold one) on
        //   a device whose driver may store it as a zero, or whose depth test may compare it
        //   otherwise than Metal's, which keeps it and compares a positive one above zero and a
        //   negative one not below zero. Announced once per device, where
        //   the denormal may be stored, not where a depth test or a read meets one. Grounds:
        //   - this host's answer, measured equal to Metal's: MoltenVK keeps a denormal in a depth
        //     image on the M3 Max (the runtime's `DepthImageKeepsDenormals`),
        //     and its depth test compares one as Metal's does there, positive and negative,
        //     written by a blit or a pass's clear, under every compare function but `Never` and
        //     `Always` (`serialized::draws`'
        //     `denormal_depths_store_and_compare_as_metals`, which announces the kind); the
        //     runtime measured that depth test on an Apple M4 alone (`DepthTestKeepsDenormals`),
        //     so it announces the kind on this host too;
        //   - on another device, a fact of the device's depth unit, which the runtime measured:
        //     NVIDIA's depth test compares a stored positive denormal as zero, and RADV and ANV
        //     store one a GPU write gives as a zero. Where the driver stores it and only its depth
        //     test differs, a single-sample `Depth32Float` render target's passes test the
        //     smallest normal in its place (the runtime's magnitude stand-in, as for
        //     `SignedDepthTest`): Metal's answer but for a fragment at exactly 2^-126, which
        //     passes `Equal` and fails `Greater` against it where Metal's fails and passes, as
        //     the notice says. No exact form is the device's but a depth test run in the shader,
        //     an interpreter's, which the rule for native devices does not ask for.
        // - `DeviceFilteredMipmaps`: `generateMipmapsForTexture:` of a filtered format, whose
        //   levels are the device's linear filter of the level above (its blit chain, or the
        //   runtime's filter in `float` where its blits do not filter the format), announced once
        //   per device, MoltenVK's included. Apple's GPUs answer otherwise among themselves (an M3
        //   Max's and an M4's levels up to a unit in the last place apart), so no one value is
        //   Metal's, and the runtime answers such a value with the device's own arithmetic,
        //   recording a note where it can differ from Metal's, with no exact form to take.
        // - `FloatTininessAfterRounding`: the device's maths, as `NativeFma`'s: a `float` product or fused multiply-add within half a unit below
        //   `2^-126` answers `2^-126` of its sign where Metal flushes it to the zero of its sign.
        //   MoltenVK's float functions are Metal's own, so a device that runs on Metal never
        //   announces it, and on a native device it would be relayed as `NativeFma` is
        //   ([`native::NATIVE_RELAYED`]). It is announced once, as the runtime makes the device,
        //   before this platform adopts the device's class ([`crate::gpu::device`]'s `adopt`), where
        //   a kind of [`DIES`] would die in strict mode on every native device's first creation:
        //   relayed instead, the same answer on every device.
        // - `DeviceBooleanVisibility`: a `Boolean` visibility result where a sample passed is the
        //   device's nonzero occlusion value, which Metal's own devices differ in too (an M3 Max
        //   writes 1, the AMD Radeon Pro 5300M and the Intel UHD Graphics 630 the samples that
        //   passed, as the runtime records), so no one nonzero value is Metal's; 0 where none
        //   passed, as Metal's.
        const PRECISION: [D; 12] = [
            D::UncalibratedGpuTimes,
            D::UntimedCommandBuffer,
            D::GpuTimesFromHost,
            D::DontCareLoadTiles,
            D::ExtendedRangeBitsKept,
            D::DrawnDepthCopy,
            D::UnnamedShadowedTexture,
            D::UnmeasuredUniqueIdentifier,
            D::DepthDenormals,
            D::DeviceFilteredMipmaps,
            D::FloatTininessAfterRounding,
            D::DeviceBooleanVisibility,
        ];
        // A pipeline stage's precision notes, each arithmetic, filtering or ordering of the device
        // that its documentation says can differ from Metal's: relayed, but those [`DIES`] and
        // [`HEARD`] list, each for one of the module's reasons:
        // - `FastMathRewrites`: an accepted class, fast-math rewrites
        //   within what the function's AIR contract and reassociation flags allow, which Metal's
        //   compiler may make too.
        // - `DeviceTranscendentals`: an accepted class, maths functions
        //   whose bound Metal does not state, each device's deviation measured and recorded, and
        //   fixed where far off. Later runtimes do not record it where MoltenVK's float functions
        //   are Metal's own.
        // - `HalfOutputFastMathBlend`: the accepted class of fast-math rewrites, extended to
        //   blends. A fast-math module's `half` colour output to an attachment, whose blend
        //   and conversion Metal's compiler folds into the function and regroups per pipeline in
        //   no form the AIR and the blend state decide: the device's blend of the output's `half`
        //   stores a value at most three `half` units in the last place at the largest operand's
        //   magnitude apart from Metal's into a 16-bit float attachment, and two `float` ones into
        //   a 32-bit float one (MoltenVK on the M3 Max against plain Metal). A
        //   module without fast math records `HalfOutputPrecision` instead ([`HEARD`]).
        // And `DeviceColorConversion`: a colour channel the device converts, which on MoltenVK,
        // whose device stores Metal's codes, is a half-float channel the function does not round
        // itself: the alpha that alpha to coverage reads at colour 0 at the default rounding
        // mode, and under `textureWriteRoundingMode` 2 where the write mask leaves it out (under
        // mode 2 the function rounds one the mask stores, and the coverage reads it rounded,
        // `CoverageReadsRoundedAlpha` of [`DIES`]), and under mode 2 a half-float
        // attachment blended by other factors than one and zero (`BlendedWriteRounding`, of
        // [`DIES`], too). Its conversion is measured as Metal's, byte for byte over sweeps
        // through unorm values, past 1 by float steps and past the largest half, by
        // `encode_render`'s
        // `mode_2_half_float_attachments_no_other_test_compares_round_as_metals`, at `R16Float`,
        // `RG16Float` and `RGBA16Float`. A `bfloat` colour output no longer reaches the note: the
        // runtime refuses one as invalid input, as Apple's compiler rejects the return
        // type, which the pipeline creators answer as an `NSError` of code 0
        // (`crate::gpu::states::pipeline_error`); what Metal does with one is not measured.
        const NOTES: [D; 4] = [
            D::DeviceTranscendentals,
            D::FastMathRewrites,
            D::HalfOutputFastMathBlend,
            D::DeviceColorConversion,
        ];
        // No value the guest or the host sees: `PrivatePropertiesUnknown`, Apple's private device
        // properties the runtime has no answer of the device's for, announced where
        // `rt::Device::private_properties` is first read, as a device's caps are. The runtime
        // gives none there (`rt::PrivateAnswer::value` is `None`), never a value of its own, and
        // this platform answers each of those from a stand-in its table names
        // ([`crate::gpu::caps`]).
        const UNSEEN: [D; 1] = [D::PrivatePropertiesUnknown];
        let relayed: Vec<D> = exact
            .chain(one_of_metals)
            .chain(PRECISION)
            .chain(NOTES)
            .chain(UNSEEN)
            .collect();
        for kind in D::ALL {
            let places = [
                DIES.contains(kind),
                HEARD.contains(kind),
                relayed.contains(kind),
            ];
            let placed = places.into_iter().filter(|&p| p).count();
            assert_eq!(placed, 1, "{kind:?} ({}): {places:?}", kind.code());
        }
        assert_eq!(DIES.len() + HEARD.len() + relayed.len(), D::ALL.len());
        // On a native device, the kinds of `native::NATIVE_RELAYED` are relayed where they die
        // on MoltenVK: each a kind of DIES, listed once, with its ground at the entry. They are
        // the device's interpolation and level arithmetic, the heap size the
        // device's memory granularity decides, which the guest reads, and, on the rule for native
        // devices, the precision notes whose difference is the device's rounding, evaluation
        // position or order, and its line rasterizer's tie at a vertex two pixels' diamonds share
        // (`DeviceLineTies`); then, as the class of a native device's own maths and sampling, the
        // device's maths
        // and sampling, each with its bound or the runtime's record of it, the notes of the
        // runtime's native defaults off Apple's GPUs last, the device's resolve
        // (`DeviceRoundedResolve`) among them. With these, and every device on the exact
        // completion (`GroupedCompletion` never announced), a strict run on Mesa's drivers, this
        // crate's tests on Linux among them, dies neither at a grouped submission nor at a
        // resolve the device averages.
        let natively: Vec<D> = native::NATIVE_RELAYED.iter().map(|r| r.kind).collect();
        assert_eq!(
            natively,
            [
                D::DeviceInterpolation,
                D::GradFromBias,
                D::HeapTextureLargerThanMetal,
                D::ReductionOrder,
                D::DeviceSignedZeroInfNan,
                D::MultisamplePosition,
                D::ImplementationDefinedReconvergence,
                D::DeviceLineTies,
                D::F16FmaDoubleRounding,
                D::F16IntConvDoubleRounding,
                D::DeviceArithmetic,
                D::DeviceDenormals,
                D::DeviceNanEncoding,
                D::FloatSampleFilter,
                D::HalfSampleFilter,
                D::CubeGradientFootprint,
                D::CompressedTexelDecode,
                D::ViewBaseTexelOffset,
                D::SrgbTexelDecode,
                D::BicubicFilterArithmetic,
                D::HalfTexelConversion,
                D::DeviceSampleLevel,
                D::NativeTranscendentals,
                D::NativeDivision,
                D::NativeFma,
                D::NativeNanEncoding,
                D::NativeMinMax,
                D::NativeDot,
                D::NativeMix,
                D::NativeHalfConversion,
                D::NativeHalfQuantize,
                D::DeviceSampling,
                D::DeviceTexelConversion,
                D::DecodedTexelPrecision,
                D::DeviceRoundedResolve,
                D::DevicePackConversion,
                D::DeviceVertexConversion,
                D::DeviceAlphaCoverage,
            ]
        );
        // These stay deaths on native devices too, as neither a fact of the device nor the
        // class of the device's own maths and sampling covers them: two precision notes
        // are other values, not other bits, the next two are where Apple's compiler places or
        // speculates an operation, Metal's rule for tessellation factors is not read, blending,
        // storage, an attachment's memory read in place, a heap texture's compressed image, a
        // pass's tile store and when a command buffer is seen complete are neither maths nor
        // sampling, the anisotropic emulation is a fitted model, and the band's corners are the
        // runtime's own, whose snap to Metal's grid the hardware allows (`DeviceSubpixelSnap`).
        for kind in [
            D::TessellationFactorInstance,
            D::Fp8Value,
            D::QuadsCountedBeforeDiscard,
            D::UnmeasuredGroupSpeculation,
            D::AttachmentReadInPlace,
            D::TessellationFactorScale,
            D::BlendedWriteRounding,
            D::RotatedBlendRounding,
            D::AnisotropicFilter,
            D::TessellationWinding,
            D::TessellationFactorEdgeCases,
            D::HeapTextureCompressed,
            D::SnormMinimumKept,
            D::SharedExponentKept,
            D::GroupedCompletion,
            D::DeviceSubpixelSnap,
        ] {
            assert!(DIES.contains(&kind) && !native::relayed(kind), "{kind:?}");
        }
        for (at, r) in native::NATIVE_RELAYED.iter().enumerate() {
            let code = r.kind.code();
            assert!(
                DIES.contains(&r.kind),
                "{code} relayed natively, but not of DIES"
            );
            assert!(
                !native::NATIVE_RELAYED[..at]
                    .iter()
                    .any(|o| o.kind == r.kind),
                "{code} relayed natively twice"
            );
            assert!(
                !r.ground.is_empty(),
                "{code} relayed natively without a ground"
            );
        }
    }

    /// In strict mode a kind of DIES dies on a device that runs on Metal, MoltenVK's, as before;
    /// on a native device the kinds of `native::NATIVE_RELAYED` are relayed and every other one
    /// dies; census mode counts and relays what would die, and leaves a native relay a native
    /// relay. By default every kind is relayed, counted, on every device, and the process goes
    /// on, but those of `DIES_BY_DEFAULT`, which die
    #[test]
    fn a_kind_of_dies_is_placed_by_the_device_class() {
        use native::Placement::{Native, OnMetal};
        for kind in DIES {
            let natively = native::relayed(kind);
            assert_eq!(fate(kind, OnMetal, false, true), Fate::Dies, "{kind:?}");
            assert_eq!(fate(kind, OnMetal, true, true), Fate::Counted, "{kind:?}");
            let by_default = if DIES_BY_DEFAULT.contains(&kind) {
                Fate::Dies
            } else {
                Fate::Relayed
            };
            assert_eq!(fate(kind, OnMetal, false, false), by_default, "{kind:?}");
            assert_eq!(fate(kind, OnMetal, true, false), Fate::Counted, "{kind:?}");
            let (native, census, relayed) = if natively {
                (
                    Fate::RelayedNatively,
                    Fate::RelayedNatively,
                    Fate::RelayedNatively,
                )
            } else {
                (Fate::Dies, Fate::Counted, by_default)
            };
            assert_eq!(fate(kind, Native, false, true), native, "{kind:?}");
            assert_eq!(fate(kind, Native, true, true), census, "{kind:?}");
            assert_eq!(fate(kind, Native, false, false), relayed, "{kind:?}");
        }
        assert_eq!(
            fate(rt::Degradation::DeviceInterpolation, Native, false, true),
            Fate::RelayedNatively
        );
        // A precision note the device's evaluation position decides: relayed on a native device,
        // a death on MoltenVK in strict mode, as before; so is the device's sampling, as the class
        // of the device's own maths and sampling. One whose Metal rule is not read, and not the
        // device's maths or sampling, dies on both in strict mode, and is relayed on both by
        // default.
        let position = rt::Degradation::MultisamplePosition;
        assert_eq!(fate(position, Native, false, true), Fate::RelayedNatively);
        assert_eq!(fate(position, OnMetal, false, true), Fate::Dies);
        let cube = rt::Degradation::CubeGradientFootprint;
        assert_eq!(fate(cube, Native, false, true), Fate::RelayedNatively);
        assert_eq!(fate(cube, OnMetal, false, true), Fate::Dies);
        let scale = rt::Degradation::TessellationFactorScale;
        assert_eq!(fate(scale, Native, false, true), Fate::Dies);
        assert_eq!(fate(scale, OnMetal, false, true), Fate::Dies);
        assert_eq!(fate(scale, Native, false, false), Fate::Relayed);
        assert_eq!(fate(scale, OnMetal, false, false), Fate::Relayed);
        // A kind whose difference breaks what the guest or host relies on dies by default too,
        // on every device; each is one of DIES, and none a native relay.
        for kind in DIES_BY_DEFAULT {
            assert!(DIES.contains(&kind) && !native::relayed(kind), "{kind:?}");
            assert_eq!(fate(kind, OnMetal, false, false), Fate::Dies, "{kind:?}");
            assert_eq!(fate(kind, Native, false, false), Fate::Dies, "{kind:?}");
        }
    }

    /// A test of this crate runs strict; [`relaxed`] runs its call under the default, on its
    /// thread alone, and puts strict mode back after, an unwinding call's included
    #[test]
    fn tests_run_strict_but_where_relaxed() {
        assert!(strict());
        assert!(!relaxed(strict));
        assert!(strict());
        let unwound = catch_death(|| relaxed(|| dies("sel:", "a death inside")));
        assert!(unwound.is_err() && strict());
        let elsewhere = relaxed(|| std::thread::scope(|s| s.spawn(strict).join().unwrap()));
        assert!(elsewhere);
    }

    /// Every kind this platform relays as computed otherwise than Metal is placed once: listed
    /// once, under a code of its own, none a runtime kind's
    #[test]
    fn every_inexact_kind_is_placed_once() {
        let codes: Vec<&str> = Inexact::ALL.iter().map(|k| k.code()).collect();
        for (at, code) in codes.iter().enumerate() {
            assert!(!codes[..at].contains(code), "{code} placed twice");
            assert!(
                rt::Degradation::ALL.iter().all(|d| d.code() != *code),
                "{code} is another kind's"
            );
        }
        for kind in Inexact::ALL {
            match kind {
                Inexact::HalfOutputBlend
                | Inexact::BfloatLoaded
                | Inexact::WriteRounding
                | Inexact::HeapLayout
                | Inexact::RateMap
                | Inexact::FillFormat
                | Inexact::LateResolve
                | Inexact::ProtectedContent
                | Inexact::UniqueIdentifier
                | Inexact::IcbIdentifierAfterEncoder
                | Inexact::IcbLengthPastTheLastCommand
                | Inexact::IcbCopiedLength
                | Inexact::RefusedCreate => {}
            }
        }
    }

    /// By default a value computed otherwise than Metal is relayed and counted, and the call
    /// returns; in strict mode it dies at its selector, as before, its line saying what and
    /// naming the kind's code
    #[test]
    fn an_inexact_value_is_relayed_but_in_strict_mode() {
        let counted = || {
            (inexact_counts().into_iter())
                .find(|&(k, _)| k == Inexact::HeapLayout)
                .map_or(0, |(_, n)| n)
        };
        let before = counted();
        relaxed(|| inexact("sel:", Inexact::HeapLayout, || "a value".into()));
        relaxed(|| inexact("sel:", Inexact::HeapLayout, || "a value".into()));
        assert_eq!(counted(), before + 2);
        *LAST_LINE.lock() = None;
        let site = reims_vgpu_core::testkit::capture::catch_death(|| {
            inexact("sel:", Inexact::HeapLayout, || "a value".into());
        })
        .unwrap_err();
        assert_eq!(site.file, "sel:");
        let line = LAST_LINE.lock().take().unwrap_or_default();
        assert!(line.contains("cannot hold a value"), "{line}");
        assert!(line.contains("(inexact-heap-layout)"), "{line}");
        assert_eq!(counted(), before + 2);
    }

    /// A relay past the first of its kind writes no line, so it says nothing of what it relays:
    /// what the value is goes unasked, and only the count moves
    #[test]
    fn a_relay_past_the_first_of_its_kind_asks_nothing() {
        let asked = std::cell::Cell::new(0);
        let what = || {
            asked.set(asked.get() + 1);
            String::from("a value")
        };
        relaxed(|| inexact("sel:", Inexact::HeapLayout, what));
        let first = asked.get();
        assert!(first <= 1, "asked {first} times for one line");
        relaxed(|| inexact("sel:", Inexact::HeapLayout, what));
        relaxed(|| inexact("sel:", Inexact::HeapLayout, what));
        assert_eq!(asked.get(), first);
    }

    /// A run hears what is announced inside each of its calls, in order ([`Run::heard_since`]),
    /// and a hearing inside the run hears what is announced inside it alone; the hearing outside
    /// the run hears none of it, and has its own back after it
    #[test]
    fn a_run_hears_what_each_of_its_calls_heard() {
        let kind = rt::Degradation::HalfOutputPrecision;
        let ((), outer) = hearing(|| {
            run(|r| {
                assert_eq!(r.heard(), 0);
                r.at("a", || hear(kind));
                assert!(r.heard_since(0, kind));
                let at = r.heard();
                r.at("b", || {});
                assert!(!r.heard_since(at, kind));
                let ((), inner) = hearing(|| hear(kind));
                assert_eq!(inner, [kind]);
                assert_eq!(r.heard(), at);
                r.at("c", || hear(kind));
                assert!(r.heard_since(at, kind) && !r.heard_since(r.heard(), kind));
            });
            hear(rt::Degradation::RasterizationRateMap);
        });
        assert_eq!(outer, [rt::Degradation::RasterizationRateMap]);
        assert_eq!(depth(), 0);
    }

    /// A call here that the runtime announced a degradation of [`DIES`] inside, and that then
    /// unwinds, a death of its own inside it, dies of that death alone, and lets the notice go:
    /// the thread's next call here returns. A copy within one texture of a two-plane XR10 format
    /// announces `TwoPlaneTextureCopy` (`gpu::encode_blit`'s
    /// `two_plane_xr10_blits_metal_was_not_measured_at_die`)
    #[test]
    fn a_call_that_unwinds_lets_its_notice_go() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::values::TextureDesc;
        use reims_vgpu_core::raise::OnFault;
        let d = crate::gpu::device::tests::dev();
        let desc = crate::platform::tests::platform().texture_descriptor(&TextureDesc {
            pixel_format: 550,
            width: 16,
            height: 4,
            usage: 0x3,
            ..TextureDesc::default()
        });
        let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let queue = d.new_command_queue().unwrap();
        let cb = queue.raw.command_buffer().unwrap();
        let mut e = cb.blit_command_encoder().unwrap();
        let site = catch_death(|| {
            inside(Some("the call"), || {
                let texels = t.copied("the call");
                let copied = e.copy_from_texture_to_texture(
                    texels,
                    0,
                    0,
                    rt::Origin::new(0, 0, 0),
                    rt::Size::new(4, 1, 1),
                    texels,
                    0,
                    0,
                    rt::Origin::new(8, 0, 0),
                );
                assert!(
                    CALLS.with(|c| c.pending.get()) && PENDING.with(|p| p.borrow().is_some()),
                    "{copied:?}"
                );
                dies("its own death", "a death inside the call")
            })
        })
        .unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "its own death"));
        assert!(PENDING.with(|p| p.borrow().is_none()));
        assert!(!CALLS.with(|c| c.pending.get()));
        assert_eq!(depth(), 0);
        assert_eq!(plain(|| 7), 7);
        drop((e, cb, t));
    }

    /// By default a degradation of [`DIES`] announced inside a call is relayed: the call
    /// returns, nothing is kept for it to die at, and the kind is counted at each occurrence.
    /// The copy of `a_call_that_unwinds_lets_its_notice_go` announces `TwoPlaneTextureCopy`
    #[test]
    fn a_kind_of_dies_is_relayed_and_counted_by_default() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::values::TextureDesc;
        use reims_vgpu_core::raise::OnFault;
        let kind = rt::Degradation::TwoPlaneTextureCopy;
        let counted = || {
            (relayed_counts().into_iter())
                .find(|&(k, _)| k == kind)
                .map_or(0, |(_, n)| n)
        };
        let d = crate::gpu::device::tests::dev();
        let desc = crate::platform::tests::platform().texture_descriptor(&TextureDesc {
            pixel_format: 550,
            width: 16,
            height: 4,
            usage: 0x3,
            ..TextureDesc::default()
        });
        let t = d.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let queue = d.new_command_queue().unwrap();
        let cb = queue.raw.command_buffer().unwrap();
        let mut e = cb.blit_command_encoder().unwrap();
        let before = counted();
        for at in [8, 12] {
            let copied = relaxed(|| {
                inside(Some("the call"), || {
                    let texels = t.copied("the call");
                    e.copy_from_texture_to_texture(
                        texels,
                        0,
                        0,
                        rt::Origin::new(0, 0, 0),
                        rt::Size::new(4, 1, 1),
                        texels,
                        0,
                        0,
                        rt::Origin::new(at, 0, 0),
                    )
                })
            });
            assert!(copied.is_ok(), "{copied:?}");
            assert!(PENDING.with(|p| p.borrow().is_none()));
            assert!(!CALLS.with(|c| c.pending.get()));
        }
        assert_eq!(counted(), before + 2);
        drop((e, cb, t));
    }

    /// A kind relayed once is relayed at the process's first notice of it alone, whatever thread
    /// announces it; a kind of no such set is not decided there
    #[test]
    fn a_kind_relayed_once_is_relayed_at_its_first_notice_alone() {
        let kind = rt::Degradation::RasterizationRateMap;
        assert!(first_relayed_once(kind).is_some());
        assert_eq!(first_relayed_once(kind), Some(false));
        let elsewhere = std::thread::scope(|s| s.spawn(|| first_relayed_once(kind)).join());
        assert_eq!(elsewhere.unwrap(), Some(false));
        for kind in RELAYED_ONCE {
            assert!(first_relayed_once(kind).is_some(), "{kind:?}");
        }
        assert_eq!(
            first_relayed_once(rt::Degradation::ReleasedCommandState),
            None
        );
    }

    /// What Metal's validation stops is the caller's raise under a handler and a death with none,
    /// as the send's label says
    #[test]
    fn an_invalid_refusal_is_decided_by_the_label() {
        let r = send("fillBuffer:range:value:", Policy::Caught, || {
            Err::<(), _>(invalid())
        })
        .unwrap_err();
        assert_eq!(r.id(), RaiseId::Platform);
        assert_eq!(r.origin(), Origin::Platform("fillBuffer:range:value:"));
        assert!(
            r.detail().unwrap_or_default().contains("StorageMode"),
            "{r}"
        );
        let r = send(
            "fillBuffer:range:value:",
            Policy::On(OnFault::Return),
            || Err::<(), _>(invalid()),
        );
        assert!(r.is_err());
        for policy in [Policy::Fatal, Policy::On(OnFault::Die)] {
            let site = catch_death(|| {
                send(
                    "fillBuffer:range:value:",
                    policy,
                    || Err::<(), _>(invalid()),
                )
            })
            .unwrap_err();
            assert_eq!(
                (site.kind, site.file),
                (DeathKind::UncaughtObjc, "fillBuffer:range:value:")
            );
        }
        assert_eq!(send("length", Policy::Caught, || Ok(3)), Ok(3));
    }

    /// A runtime gap is a death naming the selector whatever the label: never a raise the guest
    /// could catch
    #[test]
    fn a_runtime_gap_dies_at_the_send() {
        let site = catch_death(|| {
            send("generateMipmapsForTexture:", Policy::Caught, || {
                Err::<(), _>(Error::NotImplemented(RuntimeGap::Textures))
            })
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::Abort, "generateMipmapsForTexture:", 0)
        );
        let site = catch_death(|| get("length", || Err::<u64, _>(Error::DeviceLost))).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "length"));
        let site = catch_death(|| get("length", || Err::<u64, _>(invalid()))).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::UncaughtObjc, "length"));
    }

    /// A command the runtime refuses for sampling a texture of an address-mode map dies naming
    /// which path it samples it by
    #[test]
    fn a_mapped_texture_sampling_refusal_names_its_path() {
        use metal2vulkan::metal::SamplerAddressMode;
        use metal2vulkan_runtime::{AddressModeMap, MappedSampling};
        let map = AddressModeMap::IDENTITY
            .with(SamplerAddressMode::Repeat, SamplerAddressMode::ClampToEdge);
        for (through, named) in [
            (
                MappedSampling::ConstexprSampler,
                "through a constexpr sampler",
            ),
            (
                MappedSampling::SharedSamplerSlot,
                "through a sampler slot that samples",
            ),
            (MappedSampling::Handles, "samples through handles"),
        ] {
            let e = Error::NotImplemented(RuntimeGap::MappedTextureSampling { map, through });
            let text = refusal(&e);
            assert!(text.contains(named), "{text}");
            assert!(text.contains(&format!("{through:?}")), "{text}");
        }
        let gap = refusal(&Error::NotImplemented(RuntimeGap::Textures));
        assert!(!gap.contains("address modes"), "{gap}");
    }

    /// A refusal of SIMD-group or quad-group operations names the stage and the class the
    /// refusal names, and no other refusal names one
    #[test]
    fn a_simd_stage_refusal_names_its_stage_and_class() {
        use metal2vulkan::device::{SubgroupOp, VkProperty};
        use metal2vulkan::error::{Capability, Feature, Unsupported};
        use metal2vulkan::pipeline::Stage;
        let property = |p| Capability::Property(p);
        for (unsupported, named) in [
            (
                Unsupported::needs(
                    Feature::SubgroupOperations,
                    property(VkProperty::SubgroupStage(Stage::Vertex)),
                ),
                "a vertex function's SIMD-group operations, which",
            ),
            (
                Unsupported::needs(
                    Feature::QuadOperations,
                    property(VkProperty::SubgroupQuadOperationsInAllStages),
                ),
                "a vertex function's quad-group operations, which",
            ),
            (
                Unsupported::needs_any_of(
                    Feature::SubgroupOperations,
                    vec![vec![
                        property(VkProperty::SubgroupOperation(SubgroupOp::Arithmetic)),
                        property(VkProperty::SubgroupOperation(SubgroupOp::ShuffleRelative)),
                    ]],
                ),
                "a function's SIMD-group operations (the classes Arithmetic, ShuffleRelative)",
            ),
        ] {
            let e = Error::Unsupported(unsupported);
            let text = refusal(&e);
            assert!(text.contains(named), "{text}");
            assert!(text.contains("key 33"), "{text}");
            // A pipeline's death line carries the clause, then its descriptor's formats.
            let formats = "the descriptor's formats: no colour attachments";
            let line = refusal_describing(&e, Some(formats));
            assert_eq!(line, format!("{text}; {formats}"));
        }
        for other in [
            Error::Unsupported(Unsupported::not_implemented(Feature::SubgroupOperations)),
            Error::Unsupported(Unsupported::needs(
                Feature::SimdgroupMatrix,
                property(VkProperty::SubgroupStage(Stage::Vertex)),
            )),
            Error::NotImplemented(RuntimeGap::Textures),
        ] {
            let text = refusal(&other);
            assert!(!text.contains("key 33"), "{text}");
        }
    }

    /// A creator answers nil for the refusals Metal answers nil for, every `Error::Nil` and
    /// those its set names, and decides the rest; an `Error::Nil` at a send that makes nothing
    /// dies naming the selector
    #[test]
    fn a_creator_answers_nil_where_metal_does() {
        let nil_of = || {
            Err::<(), _>(Error::Nil(metal2vulkan_runtime::Nil::new(
                "new_texture",
                Problem::StorageMode,
            )))
        };
        for policy in [Policy::Dies, Policy::Caught] {
            assert_eq!(
                create("newTextureWithDescriptor:", policy, Nil::Never, nil_of),
                Ok(None)
            );
        }
        let site =
            catch_death(|| send("fillBuffer:range:value:", Policy::Caught, nil_of)).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "fillBuffer:range:value:")
        );
        let nil = create(
            "newBufferWithLength:options:",
            Policy::Caught,
            Nil::BufferLength,
            || Err::<(), _>(Error::OutOfMemory(metal2vulkan_runtime::MemoryKind::Device)),
        );
        assert_eq!(nil, Ok(None));
        let r = create(
            "newBufferWithLength:options:",
            Policy::Caught,
            Nil::Never,
            || Err::<(), _>(invalid()),
        );
        assert!(r.is_err());
        assert_eq!(
            create("newFence", Policy::Caught, Nil::Never, || Ok(1)),
            Ok(Some(1))
        );
    }

    /// How many times this process relayed a gap at a creator ([`Inexact::RefusedCreate`]).
    fn refused_creates() -> u64 {
        (inexact_counts().into_iter())
            .find(|&(k, _)| k == Inexact::RefusedCreate)
            .map_or(0, |(_, n)| n)
    }

    /// The gaps of the runtime's or the device's, where Metal makes the object: `Unsupported`
    /// (the runtime's and a compile's) but by a GPU family or a device Metal refuses it on, and
    /// `NotImplemented`; no other refusal
    #[test]
    fn a_gap_is_a_feature_the_runtime_or_the_device_lacks() {
        use metal2vulkan::error::{Error as Compile, Feature, Needs, Unsupported};
        let unsupported = Unsupported::not_implemented(Feature::Format);
        let mut metal_s = unsupported.clone();
        metal_s.needs = Needs::MetalGpuFamily(1009);
        for (e, is) in [
            (Error::Unsupported(unsupported.clone()), true),
            (Error::Compile(Compile::Unsupported(unsupported)), true),
            (Error::NotImplemented(RuntimeGap::Textures), true),
            (Error::Unsupported(metal_s), false),
            (Error::DeviceLost, false),
            (Error::OutOfMemory(rt::MemoryKind::Device), false),
            (invalid(), false),
        ] {
            assert_eq!(gap(&e), is, "{e:?}");
        }
    }

    /// A creator answers nil by default for a gap of the runtime's or the device's (a sampler of
    /// an address mode the device lacks, a depth-stencil state the runtime has not built, a
    /// texture of a format it refuses), relayed and counted, and dies at its selector in strict
    /// mode, its line the refusal's as before; any other refusal dies in either mode
    #[test]
    fn a_gap_at_a_creator_is_nil_by_default_and_a_death_in_strict_mode() {
        use metal2vulkan::error::{Feature, Unsupported};
        let gaps = [
            (
                "newSamplerStateWithDescriptor:",
                Error::Unsupported(Unsupported::not_implemented(Feature::SamplerAddressMode)),
            ),
            (
                "newDepthStencilStateWithDescriptor:",
                Error::NotImplemented(RuntimeGap::IndirectRenderCommands),
            ),
            (
                "newTextureWithDescriptor:",
                Error::Unsupported(Unsupported::not_implemented(Feature::Format)),
            ),
        ];
        for (sel, e) in gaps {
            let before = refused_creates();
            let refused = || Err::<(), _>(e.clone());
            let (made, noted) =
                relaying_gaps(|| relaxed(|| create(sel, Policy::Dies, Nil::Never, refused)));
            assert_eq!(made, Ok(None), "{sel}");
            // Noted for core, which faults the work naming a function so refused
            assert!(noted, "{sel}");
            assert_eq!(refused_creates(), before + 1, "{sel}");
            *LAST_LINE.lock() = None;
            let site = catch_death(|| create(sel, Policy::Dies, Nil::Never, refused)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
            assert_eq!(LAST_LINE.lock().take(), Some(refusal(&e)), "{sel}");
            assert_eq!(refused_creates(), before + 1, "{sel}");
        }
        let before = refused_creates();
        let site = catch_death(|| {
            relaxed(|| {
                create(
                    "newTextureWithDescriptor:",
                    Policy::Dies,
                    Nil::Never,
                    || Err::<(), _>(Error::DeviceLost),
                )
            })
        })
        .unwrap_err();
        assert_eq!(site.file, "newTextureWithDescriptor:");
        assert_eq!(refused_creates(), before);
        // Metal's nil is no gap
        let (made, noted) = relaying_gaps(|| {
            create(
                "newTextureWithDescriptor:",
                Policy::Dies,
                Nil::Never,
                || {
                    Err::<(), _>(Error::Nil(metal2vulkan_runtime::Nil::new(
                        "new_texture",
                        Problem::StorageMode,
                    )))
                },
            )
        });
        assert_eq!((made, noted), (Ok(None), false));
    }

    /// An unbuilt method, and a value the runtime cannot hold, die naming the selector
    #[test]
    fn unbuilt_and_unrepresentable_die_at_the_selector() {
        let site = catch_death(|| unbuilt("newHeapWithDescriptor:")).unwrap_err();
        assert_eq!(
            (site.kind, site.file, site.line),
            (DeathKind::Abort, "newHeapWithDescriptor:", 0)
        );
        let site =
            catch_death(|| unrepresentable("setPixelFormat:", "pixel format 1")).unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "setPixelFormat:")
        );
    }

    /// A runtime error as an NSError carries the runtime's text
    #[test]
    fn error_text_is_the_runtimes() {
        let t = error_text(&Error::DeviceLost);
        assert_eq!(t.description, "the Vulkan device was lost");
        assert_eq!(t.to_string(), t.description);
    }
}
