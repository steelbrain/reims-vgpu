//! The pieces of the rt platform not built yet ([`Unbuilt`]), and their refusals; and the
//! refusals rt itself answers a built piece with ([`rt_refused`]).
//!
//! A call that reaches an unbuilt piece answers as Metal's call answers where it fails: a nil
//! object, a failed creation, zeros from a getter. Never a death or a panic. Each refusal is
//! counted under its piece's code, `rt-unbuilt-<piece>` ([`Unbuilt::code`]), in the process's
//! degradation counts ([`crate::answer::count_kind`], which the `reims-vgpu: degradation counts:`
//! line gives, docs/conventions.md). The first refusal of each kind in the process writes one line
//! on stderr naming the call, the piece and the code. A boot's counts line therefore lists every
//! unbuilt piece the boot reached, with how many times.
//!
//! A call of a built piece that rt refuses (its `Error`: a limit, a refusal of Metal's, a known
//! gap, a failed Vulkan call) answers the same way, counted under the error's kind,
//! `rt-refused-<kind>` ([`RtRefusal::code`]), its first line naming the call and rt's error.
//!
//! [`Unbuilt::ALL`] is the list of what is still unbuilt. A piece leaves it when it is built.

use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_rt as rt;

use crate::core::platform::types::NsErrorText;

/// The domain of the `NSError` a refused creator with an `error:` out-parameter answers
/// ([`refused_error`]).
pub const RT_DOMAIN: &str = "metal2vulkan-rt";

/// A piece of the rt platform not built yet, which the calls that reach it refuse ([`refuse`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unbuilt {
    /// Buffers the device makes (`newBufferWithLength:options:` and the others but the host
    /// memory's): over rt's blocks. Each answers nil.
    Buffer,
    /// A buffer or a heap over the host's memory (`newBufferWithBytesNoCopy:…`, a heap over the
    /// guest's pages): rt's import of host memory, which the pinned rt has not. Each answers nil.
    HostImport,
    /// Textures the device makes, and their views. Each answers nil.
    Texture,
    /// A texture over a plane of an IOSurface (`newTextureWithDescriptor:iosurface:plane:`), which
    /// rt makes none of yet. Each answers nil.
    IosurfaceTexture,
    /// Shared textures and their handles (`newSharedTextureWithDescriptor:`), which rt has none
    /// of. Each answers nil.
    SharedTexture,
    /// Heaps (`newHeapWithDescriptor:`). Each answers nil.
    Heap,
    /// Sampler states. Each answers nil.
    Sampler,
    /// Depth-stencil states. Each answers nil.
    DepthStencil,
    /// Libraries and their functions (`newLibraryWithData:error:`): over the translator's
    /// library. Each answers nil.
    Library,
    /// Compute and render pipeline states. Each answers nil, with an `NSError` where Metal's
    /// creation fills one.
    Pipeline,
    /// Argument buffers' layouts (`newArgumentEncoderWithArguments:` and a function's). Each
    /// answers nil.
    ArgumentLayout,
    /// Command buffers, and so their encoders. `commandBuffer` answers nil.
    CommandBuffer,
    /// Events (`newEvent`). Each answers nil.
    Event,
    /// Fences (`newFence`). Each answers nil.
    Fence,
    /// Shared events (`newSharedEvent`). Each answers nil.
    SharedEvent,
    /// A render pass descriptor's attachments and properties (`RenderPassDescriptor`): its sets
    /// keep nothing, and its reads answer Metal's reading of a nil attachment.
    RenderPass,
    /// Rasterization rate maps, their descriptors and layers: rt has none. A layer's creation
    /// answers nil, and a descriptor keeps no layer.
    RateMap,
    /// Indirect command buffers, and the reservation of their resource indices (type 5): rt has
    /// none. A creation answers nil.
    Icb,
    /// Dynamic libraries (`newDynamicLibrary:error:`): rt has none. A creation answers nil.
    DynamicLibrary,
    /// A reservation of resource indices of a type core never reserves (other than 0, 1 and 5).
    /// It reserves nothing.
    ResourceType,
}

impl Unbuilt {
    /// Every piece, once, in the order of their codes' places.
    pub const ALL: [Unbuilt; 20] = [
        Unbuilt::Buffer,
        Unbuilt::HostImport,
        Unbuilt::Texture,
        Unbuilt::IosurfaceTexture,
        Unbuilt::SharedTexture,
        Unbuilt::Heap,
        Unbuilt::Sampler,
        Unbuilt::DepthStencil,
        Unbuilt::Library,
        Unbuilt::Pipeline,
        Unbuilt::ArgumentLayout,
        Unbuilt::CommandBuffer,
        Unbuilt::Event,
        Unbuilt::Fence,
        Unbuilt::SharedEvent,
        Unbuilt::RenderPass,
        Unbuilt::RateMap,
        Unbuilt::Icb,
        Unbuilt::DynamicLibrary,
        Unbuilt::ResourceType,
    ];

    /// The piece's code in the degradation counts.
    pub const fn code(self) -> &'static str {
        match self {
            Unbuilt::Buffer => "rt-unbuilt-buffer",
            Unbuilt::HostImport => "rt-unbuilt-host-import",
            Unbuilt::Texture => "rt-unbuilt-texture",
            Unbuilt::IosurfaceTexture => "rt-unbuilt-iosurface-texture",
            Unbuilt::SharedTexture => "rt-unbuilt-shared-texture",
            Unbuilt::Heap => "rt-unbuilt-heap",
            Unbuilt::Sampler => "rt-unbuilt-sampler",
            Unbuilt::DepthStencil => "rt-unbuilt-depth-stencil",
            Unbuilt::Library => "rt-unbuilt-library",
            Unbuilt::Pipeline => "rt-unbuilt-pipeline",
            Unbuilt::ArgumentLayout => "rt-unbuilt-argument-layout",
            Unbuilt::CommandBuffer => "rt-unbuilt-command-buffer",
            Unbuilt::Event => "rt-unbuilt-event",
            Unbuilt::Fence => "rt-unbuilt-fence",
            Unbuilt::SharedEvent => "rt-unbuilt-shared-event",
            Unbuilt::RenderPass => "rt-unbuilt-render-pass",
            Unbuilt::RateMap => "rt-unbuilt-rate-map",
            Unbuilt::Icb => "rt-unbuilt-icb",
            Unbuilt::DynamicLibrary => "rt-unbuilt-dynamic-library",
            Unbuilt::ResourceType => "rt-unbuilt-resource-type",
        }
    }

    /// What the piece is, for the first refusal's line.
    pub const fn what(self) -> &'static str {
        match self {
            Unbuilt::Buffer => "a buffer the device makes",
            Unbuilt::HostImport => "a buffer or heap over the host's memory (rt's import)",
            Unbuilt::Texture => "a texture",
            Unbuilt::IosurfaceTexture => "a texture over an IOSurface's plane",
            Unbuilt::SharedTexture => "a shared texture",
            Unbuilt::Heap => "a heap",
            Unbuilt::Sampler => "a sampler state",
            Unbuilt::DepthStencil => "a depth-stencil state",
            Unbuilt::Library => "a library",
            Unbuilt::Pipeline => "a pipeline state",
            Unbuilt::ArgumentLayout => "an argument buffer's layout",
            Unbuilt::CommandBuffer => "a command buffer",
            Unbuilt::Event => "an event",
            Unbuilt::Fence => "a fence",
            Unbuilt::SharedEvent => "a shared event",
            Unbuilt::RenderPass => "a render pass descriptor's attachments",
            Unbuilt::RateMap => "a rasterization rate map",
            Unbuilt::Icb => "an indirect command buffer",
            Unbuilt::DynamicLibrary => "a dynamic library",
            Unbuilt::ResourceType => "a reservation of resource indices of another type",
        }
    }

    /// The piece's place in [`Unbuilt::ALL`].
    fn at(self) -> usize {
        Unbuilt::ALL
            .iter()
            .position(|&piece| piece == self)
            .unwrap_or(0)
    }
}

/// Whether the process refused a call of each piece yet, in [`Unbuilt::ALL`]'s order.
static SAID: [AtomicBool; Unbuilt::ALL.len()] =
    [const { AtomicBool::new(false) }; Unbuilt::ALL.len()];

/// Refuses the call `sel` for the unbuilt `piece`: counts it under the piece's code, and the
/// process's first refusal of the piece writes its line on stderr. The caller answers as Metal's
/// call answers where it fails.
pub fn refuse(piece: Unbuilt, sel: &'static str) {
    crate::answer::count_kind(piece.code());
    if first(&SAID, piece.at()) {
        say(&refusal_line(piece, sel));
    }
}

/// Refuses the creator `sel` for the unbuilt `piece` ([`refuse`]), which answers nil or an
/// `NSError` where Metal makes the object: noted for core on the creator's thread as a gap of the
/// platform's own (`core::platform::device::note_relayed_gap`), so core faults the work that names
/// the nil object where Metal would end the process at it.
pub fn refuse_create(piece: Unbuilt, sel: &'static str) {
    crate::core::platform::device::note_relayed_gap();
    refuse(piece, sel);
}

/// Refuses the creator `sel` for the unbuilt `piece` ([`refuse_create`]) where it has an `error:`
/// out-parameter: the `NSError` it answers, in [`RT_DOMAIN`], code 0, naming the piece and its
/// code.
pub fn refused_error(piece: Unbuilt, sel: &'static str) -> NsErrorText {
    refuse_create(piece, sel);
    NsErrorText {
        domain: RT_DOMAIN.to_owned(),
        code: 0,
        description: format!(
            "{} is not built on metal2vulkan-rt ({})",
            piece.what(),
            piece.code()
        ),
    }
}

/// The line of the process's first refusal of `piece`, at `sel`.
fn refusal_line(piece: Unbuilt, sel: &'static str) -> String {
    format!(
        "reims-vgpu: {sel}: {} is not built on metal2vulkan-rt; refused ({}), the first of its \
         kind\n",
        piece.what(),
        piece.code()
    )
}

/// The kind of an error rt answered a built piece's call with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtRefusal {
    /// No Vulkan loader (`Error::Loader`, `Error::NoDirectDriverLoading`).
    Loader,
    /// A device below rt's floor (`Error::BelowFloor`).
    BelowFloor,
    /// A failed Vulkan call (`Error::Vulkan`).
    Vulkan,
    /// A call Metal refuses (`Error::Refused`).
    Refused,
    /// One of the device's or rt's fixed limits (`Error::Limit`).
    Limit,
    /// A block no handle holds (`Error::NoSuchBlock`).
    NoSuchBlock,
    /// A thread rt could not start (`Error::Thread`).
    Thread,
    /// The translator's refusal (`Error::Translation`).
    Translation,
    /// A Metal feature rt does not build yet, a known gap of its own (`Error::Unsupported`).
    Gap,
    /// A call after the device was quiesced (`Error::Quiesced`).
    Quiesced,
    /// No physical device the loader lists meets rt's floor, so the platform has no device.
    NoDevice,
    /// An error of a kind this platform does not name.
    Other,
}

impl RtRefusal {
    /// Every kind, once.
    pub const ALL: [RtRefusal; 12] = [
        RtRefusal::Loader,
        RtRefusal::BelowFloor,
        RtRefusal::Vulkan,
        RtRefusal::Refused,
        RtRefusal::Limit,
        RtRefusal::NoSuchBlock,
        RtRefusal::Thread,
        RtRefusal::Translation,
        RtRefusal::Gap,
        RtRefusal::Quiesced,
        RtRefusal::NoDevice,
        RtRefusal::Other,
    ];

    /// The kind of `error`.
    pub fn of(error: &rt::Error) -> RtRefusal {
        match error {
            rt::Error::Loader(_) | rt::Error::NoDirectDriverLoading => RtRefusal::Loader,
            rt::Error::BelowFloor(_) => RtRefusal::BelowFloor,
            rt::Error::Vulkan { .. } => RtRefusal::Vulkan,
            rt::Error::Refused(_) => RtRefusal::Refused,
            rt::Error::Limit(_) => RtRefusal::Limit,
            rt::Error::NoSuchBlock(_) => RtRefusal::NoSuchBlock,
            rt::Error::Thread(_) => RtRefusal::Thread,
            rt::Error::Translation(_) => RtRefusal::Translation,
            rt::Error::Unsupported(_) => RtRefusal::Gap,
            rt::Error::Quiesced => RtRefusal::Quiesced,
            _ => RtRefusal::Other,
        }
    }

    /// The kind's code in the degradation counts.
    pub const fn code(self) -> &'static str {
        match self {
            RtRefusal::Loader => "rt-refused-loader",
            RtRefusal::BelowFloor => "rt-refused-below-floor",
            RtRefusal::Vulkan => "rt-refused-vulkan",
            RtRefusal::Refused => "rt-refused-metal",
            RtRefusal::Limit => "rt-refused-limit",
            RtRefusal::NoSuchBlock => "rt-refused-no-such-block",
            RtRefusal::Thread => "rt-refused-thread",
            RtRefusal::Translation => "rt-refused-translation",
            RtRefusal::Gap => "rt-refused-gap",
            RtRefusal::Quiesced => "rt-refused-quiesced",
            RtRefusal::NoDevice => "rt-refused-no-device",
            RtRefusal::Other => "rt-refused-other",
        }
    }

    /// The kind's place in [`RtRefusal::ALL`].
    fn at(self) -> usize {
        RtRefusal::ALL
            .iter()
            .position(|&kind| kind == self)
            .unwrap_or(0)
    }
}

/// Whether the process refused a call of each kind yet, in [`RtRefusal::ALL`]'s order.
static SAID_RT: [AtomicBool; RtRefusal::ALL.len()] =
    [const { AtomicBool::new(false) }; RtRefusal::ALL.len()];

/// Counts rt's refusal `error` of the call `sel` under its kind's code, and the process's first of
/// the kind writes its line on stderr, naming the call and the error. The caller answers as
/// Metal's call answers where it fails. Returns the kind.
pub fn rt_refused(sel: &'static str, error: &rt::Error) -> RtRefusal {
    let kind = RtRefusal::of(error);
    refused_as(sel, kind, error);
    kind
}

/// Counts a refusal of the call `sel` of the kind `kind` under its code, and the process's first
/// of the kind writes its line on stderr, naming the call and `why`.
pub fn refused_as(sel: &'static str, kind: RtRefusal, why: &dyn std::fmt::Display) {
    crate::answer::count_kind(kind.code());
    if first(&SAID_RT, kind.at()) {
        say(&format!(
            "reims-vgpu: {sel}: metal2vulkan-rt refused it ({why}); refused ({}), the first of \
             its kind\n",
            kind.code()
        ));
    }
}

/// Counts a creator's refusal of the kind `kind` again, with no line: a call answered from a
/// refusal the process met before (the device's, which is made once), noted for core as a gap
/// of the platform's own as [`refuse_create`] does.
pub fn refused_again(kind: RtRefusal) {
    crate::core::platform::device::note_relayed_gap();
    crate::answer::count_kind(kind.code());
}

/// [`rt_refused`] of a creator, which answers nil where Metal makes the object: noted for core as
/// a gap of the platform's own, as [`refuse_create`] does.
pub fn rt_refused_create(sel: &'static str, error: &rt::Error) -> RtRefusal {
    crate::core::platform::device::note_relayed_gap();
    rt_refused(sel, error)
}

/// Whether this is the first time `said[at]` is asked: sets it.
fn first(said: &[AtomicBool], at: usize) -> bool {
    said.get(at)
        .is_some_and(|said| !said.swap(true, Ordering::Relaxed))
}

/// Writes `line` on stderr in one write, so it never interleaves with another thread's.
fn say(line: &str) {
    let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::relaying_gaps;

    /// The count of `code` in the process's degradation counts.
    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// Each piece is in the list once, at its place, with a code of its own that names it as
    /// unbuilt on rt, and no digit, so the logger's class of it is the whole code; so is each
    /// kind of rt's refusals.
    #[test]
    fn every_unbuilt_piece_has_a_code_of_its_own() {
        let mut codes: Vec<&str> = Unbuilt::ALL.iter().map(|p| p.code()).collect();
        for (at, piece) in Unbuilt::ALL.iter().enumerate() {
            assert_eq!(piece.at(), at, "{piece:?}");
            let code = piece.code();
            assert!(code.starts_with("rt-unbuilt-"), "{code}");
            assert!(!code.chars().any(|c| c.is_ascii_digit()), "{code}");
        }
        for (at, kind) in RtRefusal::ALL.iter().enumerate() {
            assert_eq!(kind.at(), at, "{kind:?}");
            let code = kind.code();
            assert!(code.starts_with("rt-refused-"), "{code}");
            assert!(!code.chars().any(|c| c.is_ascii_digit()), "{code}");
            codes.push(code);
        }
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), Unbuilt::ALL.len() + RtRefusal::ALL.len());
    }

    /// A refusal is counted under its piece's code in the degradation counts, every time; a
    /// creator's refusal is noted for core as a gap of the platform's.
    #[test]
    fn a_refusal_is_counted_under_its_piece_s_code() {
        let code = Unbuilt::RateMap.code();
        let before = counted(code);
        refuse(Unbuilt::RateMap, "test");
        refuse(Unbuilt::RateMap, "test");
        assert!(counted(code) >= before + 2, "{code}");
        let ((), relayed) = relaying_gaps(|| refuse_create(Unbuilt::RateMap, "test"));
        assert!(relayed);
        let ((), relayed) = relaying_gaps(|| refuse(Unbuilt::RateMap, "test"));
        assert!(!relayed);
    }

    /// rt's refusal is counted under its kind's code; a creator's is noted for core as a gap.
    #[test]
    fn rt_s_refusal_is_counted_under_its_kind() {
        let error = rt::Error::Quiesced;
        let code = RtRefusal::Quiesced.code();
        let before = counted(code);
        assert_eq!(rt_refused("test", &error), RtRefusal::Quiesced);
        let (kind, relayed) = relaying_gaps(|| rt_refused_create("test", &error));
        assert_eq!(kind, RtRefusal::Quiesced);
        assert!(relayed);
        assert!(counted(code) >= before + 2, "{code}");
    }

    /// A creator with an `error:` out-parameter answers an `NSError` naming the piece and its
    /// code, counted and noted as a gap.
    #[test]
    fn a_refused_creator_answers_an_nserror() {
        let (error, relayed) =
            relaying_gaps(|| refused_error(Unbuilt::Library, "newLibraryWithData:error:"));
        assert!(relayed);
        assert_eq!((error.domain.as_str(), error.code), (RT_DOMAIN, 0));
        assert_eq!(
            error.description,
            "a library is not built on metal2vulkan-rt (rt-unbuilt-library)"
        );
        let code = RtRefusal::NoDevice.code();
        let before = counted(code);
        let ((), relayed) = relaying_gaps(|| refused_again(RtRefusal::NoDevice));
        assert!(relayed);
        assert!(counted(code) > before, "{code}");
    }

    /// The first refusal's line names the call, the piece and its code.
    #[test]
    fn the_first_refusal_s_line_names_the_call_and_the_code() {
        assert_eq!(
            refusal_line(Unbuilt::Buffer, "newBufferWithLength:options:"),
            "reims-vgpu: newBufferWithLength:options:: a buffer the device makes is not built on \
             metal2vulkan-rt; refused (rt-unbuilt-buffer), the first of its kind\n"
        );
    }
}
