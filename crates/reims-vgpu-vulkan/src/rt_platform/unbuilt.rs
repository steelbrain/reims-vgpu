//! The pieces of the rt platform not built yet ([`Unbuilt`]), and their refusals; and the
//! refusals rt itself answers a built piece with ([`rt_refused`]).
//!
//! A call that reaches an unbuilt piece answers as Metal's call answers where it fails: a nil
//! object, a failed creation, zeros from a getter. Never a death or a panic. Each refusal is
//! counted under its piece's code, `rt-unbuilt-<piece>` ([`Unbuilt::code`]), in the process's
//! degradation counts ([`crate::answer::count_kind`], which the `reims-vgpu: degradation counts:`
//! line gives, docs/conventions.md). The first refusal of each kind in the process writes one line
//! on stderr naming the call, the piece and the code, and what the call asked of the piece where
//! its site names it, as structure alone ([`refuse_with`]: a render pass's sample positions). A
//! boot's counts line therefore lists every unbuilt piece the boot reached, with how many times.
//!
//! A call of a built piece that rt refuses (its `Error`: a limit, a refusal of Metal's, a known
//! gap, a failed Vulkan call) answers the same way, counted under the error's kind,
//! `rt-refused-<kind>` ([`RtRefusal::code`]), its first line naming the call and rt's error as
//! structure alone ([`said`]), as the `NSError` a creator answers names it. A refusal of rt's
//! (`Error::Refused`) is counted by the basis rt declares for it (`Error::basis`, in
//! metal2vulkan-runtime's `error/basis.rs`, [`RtRefusal::of_basis`]): Metal's own, documented or
//! measured, under `rt-refused-metal`, and a rule of rt's own under `rt-refused-ours`, but for a
//! heap texture's place that Metal's own rule refuses too, which is Metal's. The first of each of
//! rt's kinds by its name in the boot diagnostics (`Draw-AttachmentRead`, `TextureCopy`,
//! `RenderPass`, `Limit-TextureSamples`, `Translation-Lowering`), whichever count it goes to,
//! writes a line of its own too: rt's error as [`said`] gives it and, but for the translator's
//! refusal, as it debugs, which carries the gap's fields, and what the call asked of rt where its
//! site names it, as structure alone: a texture's descriptor, a view's and the texture it views, a
//! copy's kind, textures, places, buffer layout and what its two textures share, a pass's
//! attachments and store actions, a draw's primitive type and its pipeline state's rasterization
//! fields ([`refusal_detail`]). rt's error is in a line or an `NSError` of this module's as
//! structure alone: no name of the guest's it carries (a function's, a symbol's, a library's) is in
//! one, nor a label. Where a refused dispatch, a refused draw or a pipeline state's refused make
//! names the functions it runs ([`naming_functions`]), the first line of each kind written there
//! names them, each by its role, its library's file, its index and its name, as the boot
//! diagnostics name a refused state's AIR, so the user's log says which of the guest's functions rt
//! refused; no `NSError`, and no line past the first of its kind, names them. The boot
//! diagnostics' refusal-keys line counts every refusal of rt's they count by the functions its
//! site names, each by its role, its library's file and its index, with no name
//! ([`named_keys`], `super::diag::RefusalKeys`), so the log names every function refused, not
//! only the first of each kind. A texture rt makes none of for its pixel format
//! (`Gap::TextureFormat`) is counted under its format's own code
//! instead, `rt-refused-texture-format-<n>` ([`rt_refused_texture`]), so a boot's counts line gives
//! each format number with its count. A pipeline state's imageblock length rt refuses
//! (`Gap::ImageblockLength`) is counted apart from rt's other gaps too,
//! `rt-refused-imageblock-length` ([`rt_refused_imageblock`]). So is a heap texture rt refuses on
//! its place in its block where Metal places it in its heap, `rt-refused-heap-texture-place`
//! ([`HEAP_TEXTURE_PLACE`]).
//!
//! [`Unbuilt::ALL`] is the list of what is still unbuilt. A piece leaves it when it is built.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};

use metal2vulkan_runtime as rt;

use crate::core::platform::types::NsErrorText;
use crate::core::sync::Mutex;

/// The domain of the `NSError` a refused creator with an `error:` out-parameter answers
/// ([`refused_error`]).
pub const RT_DOMAIN: &str = "metal2vulkan-runtime";

/// A piece of the rt platform not built yet, which the calls that reach it refuse ([`refuse`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unbuilt {
    /// What of buffers is not built yet (`super::buffer`): a Private buffer of bytes
    /// (`newBufferWithDescriptor:` with contents), which a copy on the GPU fills. It answers nil.
    Buffer,
    /// A texture over a plane of an IOSurface (`newTextureWithDescriptor:iosurface:plane:`) of a
    /// format that reads beside its plane (a two-plane YCbCr or XR10 format over a surface of two
    /// planes or more), which rt makes none of yet. Each answers nil.
    IosurfaceTexture,
    /// Shared textures and their handles (`newSharedTextureWithDescriptor:`), which rt has none
    /// of. Each answers nil.
    SharedTexture,
    /// What of heaps is not built yet (`super::heap`): a sparse heap, whose ranges rt maps no
    /// tiles of. Each answers nil.
    Heap,
    /// Mesh render pipeline states
    /// (`newRenderPipelineStateWithMeshDescriptor:options:reflection:error:`): rt has none. Each
    /// answers nil and an `NSError`.
    MeshPipeline,
    /// Render commands rt has no form of: indexed patch draws, mesh draws, tile dispatches,
    /// resolve textures, the private settings of other than Metal's default rt has none of, and a
    /// line width other than 1, at which rt draws no line (`super::render`). Each encodes nothing.
    RenderCommand,
    /// Compute commands rt has no form of: an indirect `dispatchThreads`
    /// (`dispatchThreadsWithIndirectBuffer:indirectBufferOffset:`). It encodes nothing.
    ComputeCommand,
    /// Command predication (`encodeStart*`, `encodeEnd*`): rt has none. A start encodes nothing,
    /// and an end answers NO.
    Predication,
    /// Protection options other than 0 (`setProtectionOptions:`): rt protects no work. The
    /// command buffer runs as one with none.
    Protection,
    /// `fillTexture:level:slice:region:bytes:length:`, which rt has no form of. It encodes
    /// nothing.
    FillTexture,
    /// What of blits rt has no form of but [`Unbuilt::FillTexture`]: a fill of a texture's region
    /// with a colour (`fillTexture:level:slice:region:color:pixelFormat:`), a fill of a buffer with
    /// a 4-byte pattern of more than one byte value (`fillBuffer:range:pattern4:`), a copy between
    /// textures with blit options, and a copy between a buffer and a texture with blit options
    /// other than one plane of depth and stencil (`MTLBlitOption`: PVRTC rows, or more than one
    /// option). Each encodes nothing.
    Blit,
    /// A render pass descriptor's properties rt's pass has no field of: imageblock and threadgroup
    /// memory lengths, tile sizes and sample positions. A set of one other than Metal's default
    /// (sample positions other than Metal's default positions of their count, `super::descriptor`)
    /// keeps nothing, and the pass renders without it.
    RenderPass,
    /// Rasterization rate maps, their descriptors and layers: rt has none. A layer's creation
    /// answers nil, and a descriptor keeps no layer.
    RateMap,
    /// What of indirect command buffers rt does not build (`super::icb`): commands the CPU
    /// encodes (`indirectRenderCommandAtIndex:`, `indirectComputeCommandAtIndex:`), which answer
    /// nil; a CPU reset (`resetWithRange:`), which resets nothing; a blit's copy between buffers,
    /// which encodes nothing; a buffer of a resource index, which answers nil; and the
    /// reservation of their resource indices (type 5), which reserves nothing.
    Icb,
    /// Dynamic libraries (`newDynamicLibrary:error:`): rt has none. A creation answers nil.
    DynamicLibrary,
    /// A reservation of resource indices of a type core never reserves (other than 0, 1 and 5).
    /// It reserves nothing.
    ResourceType,
    /// A tessellation factor scale other than 1 (`setTessellationFactorScale:`), which rt has no
    /// call of: its patch draws read the factors as the factor buffer holds them, unscaled.
    TessellationFactorScale,
}

impl Unbuilt {
    /// Every piece, once, in the order of their codes' places.
    pub const ALL: [Unbuilt; 17] = [
        Unbuilt::Buffer,
        Unbuilt::IosurfaceTexture,
        Unbuilt::SharedTexture,
        Unbuilt::Heap,
        Unbuilt::MeshPipeline,
        Unbuilt::RenderCommand,
        Unbuilt::ComputeCommand,
        Unbuilt::Predication,
        Unbuilt::Protection,
        Unbuilt::FillTexture,
        Unbuilt::Blit,
        Unbuilt::RenderPass,
        Unbuilt::RateMap,
        Unbuilt::Icb,
        Unbuilt::DynamicLibrary,
        Unbuilt::ResourceType,
        Unbuilt::TessellationFactorScale,
    ];

    /// The piece's code in the degradation counts.
    pub const fn code(self) -> &'static str {
        match self {
            Unbuilt::Buffer => "rt-unbuilt-buffer",
            Unbuilt::IosurfaceTexture => "rt-unbuilt-iosurface-texture",
            Unbuilt::SharedTexture => "rt-unbuilt-shared-texture",
            Unbuilt::Heap => "rt-unbuilt-heap",
            Unbuilt::MeshPipeline => "rt-unbuilt-mesh-pipeline",
            Unbuilt::RenderCommand => "rt-unbuilt-render-command",
            Unbuilt::ComputeCommand => "rt-unbuilt-compute-command",
            Unbuilt::Predication => "rt-unbuilt-predication",
            Unbuilt::Protection => "rt-unbuilt-protection",
            Unbuilt::FillTexture => "rt-unbuilt-fill-texture",
            Unbuilt::Blit => "rt-unbuilt-blit",
            Unbuilt::RenderPass => "rt-unbuilt-render-pass",
            Unbuilt::RateMap => "rt-unbuilt-rate-map",
            Unbuilt::Icb => "rt-unbuilt-icb",
            Unbuilt::DynamicLibrary => "rt-unbuilt-dynamic-library",
            Unbuilt::ResourceType => "rt-unbuilt-resource-type",
            Unbuilt::TessellationFactorScale => "rt-unbuilt-tessellation-factor-scale",
        }
    }

    /// What the piece is, for the first refusal's line.
    pub const fn what(self) -> &'static str {
        match self {
            Unbuilt::Buffer => "a Private buffer of bytes",
            Unbuilt::IosurfaceTexture => "a texture over an IOSurface's plane",
            Unbuilt::SharedTexture => "a shared texture",
            Unbuilt::Heap => "a sparse heap",
            Unbuilt::MeshPipeline => "a mesh render pipeline state",
            Unbuilt::RenderCommand => "an indexed patch, mesh or tile draw, resolve or GL setting",
            Unbuilt::ComputeCommand => "an indirect dispatchThreads",
            Unbuilt::Predication => "command predication",
            Unbuilt::Protection => "a command buffer's protection options",
            Unbuilt::FillTexture => "a fill of a texture with bytes",
            Unbuilt::Blit => "a colour fill, a pattern fill or a copy with blit options",
            Unbuilt::RenderPass => "a render pass's memory lengths, tiles or samples",
            Unbuilt::RateMap => "a rasterization rate map",
            Unbuilt::Icb => "an indirect command buffer's CPU encode, reset, copy or index",
            Unbuilt::DynamicLibrary => "a dynamic library",
            Unbuilt::ResourceType => "a reservation of resource indices of another type",
            Unbuilt::TessellationFactorScale => "a tessellation factor scale other than 1",
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
    refused_piece(piece, sel, None);
}

/// Refuses the call `sel` for the unbuilt `piece` as [`refuse`] does, the line of the process's
/// first refusal of the piece, where this call writes it, naming `what` too: what the call asked
/// of the piece, as structure alone (a render pass's sample positions, `super::descriptor`).
pub fn refuse_with(piece: Unbuilt, sel: &'static str, what: &dyn std::fmt::Display) {
    refused_piece(piece, sel, Some(what));
}

/// Counts a refusal of the call `sel` under `piece`'s code; the process's first of the piece
/// writes its line ([`refusal_line`]), naming `what` where the caller names it.
fn refused_piece(piece: Unbuilt, sel: &'static str, what: Option<&dyn std::fmt::Display>) {
    crate::answer::count_kind(piece.code());
    if first(&SAID, piece.at()) {
        say(&refusal_line(piece, sel, what));
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
            "{} is not built on metal2vulkan-runtime ({})",
            piece.what(),
            piece.code()
        ),
    }
}

/// The line of the process's first refusal of `piece`, at `sel`, and what the call asked of the
/// piece, `what`, where it names it, after the line's `: ` as [`detail_line`] gives it; then the
/// functions its site names ([`named`]).
fn refusal_line(piece: Unbuilt, sel: &str, what: Option<&dyn std::fmt::Display>) -> String {
    let (what_piece, code, named) = (piece.what(), piece.code(), named());
    match what {
        Some(what) => format!(
            "reims-vgpu: {sel}: {what_piece} is not built on metal2vulkan-runtime; refused \
             ({code}), the first of its kind: {what}{named}\n"
        ),
        None => format!(
            "reims-vgpu: {sel}: {what_piece} is not built on metal2vulkan-runtime; refused \
             ({code}), the first of its kind{named}\n"
        ),
    }
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
    /// A call rt refuses as Metal's refusal (`Error::Refused` of a basis of Metal's, documented
    /// or measured, [`RtRefusal::of_basis`]), and a heap texture's place rt refuses that Metal's
    /// own rule refuses too, off the texture's alignment or past the heap's end; and what the
    /// platform refuses before rt as Metal's refusal.
    Refused,
    /// A call rt refuses on a rule of its own, not Metal's (`Error::Refused` of rt's own basis,
    /// [`RtRefusal::of_basis`]), such as a texture buffer not over a buffer its consumer places
    /// (`Refusal::TextureBufferNotOverBuffer`), which the device's texture buffer no longer
    /// reaches, made over a buffer of the platform's own (`super::texture`'s
    /// `make_texture_buffer`), or an indirect argument offset that leaves the buffer fewer bytes
    /// than the arguments' (`Refusal::IndirectArgumentsPastEnd`).
    Ours,
    /// One of the device's or rt's fixed limits (`Error::Limit`).
    Limit,
    /// A block no handle holds (`Error::NoSuchBlock`).
    NoSuchBlock,
    /// A thread rt could not start (`Error::Thread`).
    Thread,
    /// The translator's refusal (`Error::Translation`).
    Translation,
    /// A Metal feature rt does not build yet, a known gap of its own (`Error::Unsupported`), or a
    /// kernel past rt's stage bound that no helper has compiled (`Error::StageUnfilled`), whose
    /// fill the platform starts in its helper process (`super::stage`).
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
    pub const ALL: [RtRefusal; 13] = [
        RtRefusal::Loader,
        RtRefusal::BelowFloor,
        RtRefusal::Vulkan,
        RtRefusal::Refused,
        RtRefusal::Ours,
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
            rt::Error::Refused(rt::Refusal::HeapTexturePlace { .. }) if !metal_places(error) => {
                RtRefusal::Refused
            }
            rt::Error::Refused(refusal) => RtRefusal::of_basis(refusal.basis()),
            rt::Error::Limit(_) => RtRefusal::Limit,
            rt::Error::NoSuchBlock(_) => RtRefusal::NoSuchBlock,
            rt::Error::Thread(_) => RtRefusal::Thread,
            rt::Error::Translation { .. } => RtRefusal::Translation,
            rt::Error::Unsupported(_) | rt::Error::StageUnfilled(_) => RtRefusal::Gap,
            rt::Error::Quiesced => RtRefusal::Quiesced,
            _ => RtRefusal::Other,
        }
    }

    /// The kind of a refusal of rt's (`Error::Refused`) of the basis `basis`, as rt declares it
    /// (`Error::basis`): a rule of rt's own, [`RtRefusal::Ours`]; Metal's, documented or
    /// measured, [`RtRefusal::Refused`]. So is the device's basis, which no refusal of rt's
    /// declares at its pin (rt's `Limit` and `Gap` declare it, counted under their own kinds), and
    /// a basis rt adds later, each counted where it was before rt declared bases.
    pub fn of_basis(basis: rt::RefusalBasis) -> RtRefusal {
        match basis {
            rt::RefusalBasis::Ours => RtRefusal::Ours,
            _ => RtRefusal::Refused,
        }
    }

    /// The kind's code in the degradation counts.
    pub const fn code(self) -> &'static str {
        match self {
            RtRefusal::Loader => "rt-refused-loader",
            RtRefusal::BelowFloor => "rt-refused-below-floor",
            RtRefusal::Vulkan => "rt-refused-vulkan",
            RtRefusal::Refused => "rt-refused-metal",
            RtRefusal::Ours => "rt-refused-ours",
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
/// the kind writes its line on stderr, naming the call and the error as [`said`] gives it. The
/// caller answers as Metal's call answers where it fails. Returns the kind.
pub fn rt_refused(sel: &'static str, error: &rt::Error) -> RtRefusal {
    let kind = RtRefusal::of(error);
    super::diag::refusal(error);
    refused_as(sel, kind, &said(error));
    refusal_detail(sel, error, None);
    kind
}

/// Counts rt's refusal `error` of the call `sel` in the boot diagnostics by its name
/// (`super::diag::refusal`), and the process's first of the name writes its line
/// ([`refusal_detail`]), as [`rt_refused`] does: for the refusals counted under a code of their
/// own (a pixel format's, an imageblock length's, a heap texture's place).
fn diagnosed(sel: &'static str, error: &rt::Error) {
    super::diag::refusal(error);
    refusal_detail(sel, error, None);
}

/// Writes on stderr the process's first line of each kind of rt's refusals by its name in the
/// boot diagnostics (`super::diag::refusal_name`: `Draw-AttachmentRead`, `TextureCopy`,
/// `RenderPass`, `Limit-TextureSamples`, `Translation-Lowering`), which the line of the first of
/// its [`RtRefusal`] kind ([`refused_as`]) leaves out past that kind's first: [`detail_line`] of
/// the call `sel`, rt's `error` and `what`, what the call asked of rt that rt's error does not
/// name, as structure alone (a texture's descriptor; a view's and its texture's; a copy's kind,
/// textures, places, buffer layout and what its textures share; a pass's attachments and store
/// actions; a draw's primitive type and its pipeline state's rasterization fields). A caller that
/// names `what` calls it before it counts the refusal, whose own call here ([`rt_refused`],
/// [`diagnosed`]) then writes nothing. Every count of the diagnostics' calls here, so each name
/// the counts give has its line. Counts nothing.
pub fn refusal_detail(sel: &'static str, error: &rt::Error, what: Option<&dyn std::fmt::Display>) {
    let Some(name) = super::diag::refusal_name(error) else {
        return;
    };
    if first_named(name.clone()) {
        say(&detail_line(sel, error, &name, what));
    }
}

/// Whether this is the process's first ask of the refusal kind `name` ([`refusal_detail`]):
/// notes it.
fn first_named(name: String) -> bool {
    static SAID: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    SAID.lock().insert(name)
}

/// The line [`refusal_detail`] writes of rt's `error` of the call `sel`, of the kind `name`, and
/// what the call asked of rt, `what`, where it names it: rt's error as [`said`] gives it and, but
/// for the translator's refusal, whose fields can name the guest's functions and symbols, as it
/// debugs (its variant's fields), the name, and the code of [`counted_code`]; then the functions
/// its site names ([`named`]).
fn detail_line(
    sel: &str,
    error: &rt::Error,
    name: &str,
    what: Option<&dyn std::fmt::Display>,
) -> String {
    let (code, said, named) = (counted_code(error), said(error), named());
    let fields = match error {
        rt::Error::Translation { .. } => String::new(),
        _ => format!("; {error:?}"),
    };
    match what {
        Some(what) => format!(
            "reims-vgpu: {sel}: metal2vulkan-runtime refused it ({said}{fields}); the first of \
             {name} ({code}): {what}{named}\n"
        ),
        None => format!(
            "reims-vgpu: {sel}: metal2vulkan-runtime refused it ({said}{fields}); the first of \
             {name} ({code}){named}\n"
        ),
    }
}

/// rt's `error` as structure alone ([`Said`]), for a line or an `NSError` that names it, which
/// may name an error of any kind.
pub fn said(error: &rt::Error) -> Said<'_> {
    Said(error)
}

/// rt's error as [`said`] gives it: as it displays, but the translator's refusal, whose display
/// and fields can name the guest's functions, symbols and libraries, as the feature it refused and
/// what it needs (their variants' fields, the translator's enums and numbers) and the attachment
/// it is tied to, else as a refusal of the translator's alone. rt's other errors name nothing of
/// the guest's: their fields are rt's own enums and numbers, a Vulkan call and its result, and the
/// host's own I/O and loader errors.
pub struct Said<'a>(&'a rt::Error);

impl std::fmt::Display for Said<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use metal2vulkan::error::Error as Translator;
        match self.0 {
            rt::Error::Translation {
                error: Translator::Unsupported(u),
                ..
            } => {
                write!(
                    f,
                    "the translator refused the feature {:?}, needing {:?}",
                    u.feature, u.needs
                )?;
                match (u.location.as_ref()).and_then(|l| l.attachment()) {
                    Some(at) => write!(f, ", at the {at}"),
                    None => Ok(()),
                }
            }
            rt::Error::Translation { .. } => f.write_str("the translator refused it"),
            error => write!(f, "{error}"),
        }
    }
}

/// The code [`detail_line`] names for rt's `error`: a pixel format's own for `Gap::TextureFormat`
/// and [`IMAGEBLOCK_LENGTH`] for `Gap::ImageblockLength`, which [`rt_refused_texture`] and
/// [`rt_refused_imageblock`] count them under, [`HEAP_TEXTURE_PLACE`] for a heap texture's place
/// Metal places ([`metal_places`]), and its kind's code for any other.
fn counted_code(error: &rt::Error) -> std::borrow::Cow<'static, str> {
    match *error {
        rt::Error::Unsupported(rt::Gap::TextureFormat { pixel_format }) => {
            format!("rt-refused-texture-format-{}", pixel_format.as_raw()).into()
        }
        rt::Error::Unsupported(rt::Gap::ImageblockLength) => IMAGEBLOCK_LENGTH.into(),
        _ if metal_places(error) => HEAP_TEXTURE_PLACE.into(),
        _ => RtRefusal::of(error).code().into(),
    }
}

/// Counts a refusal of the call `sel` of the kind `kind` under its code, and under the code of the
/// make that runs on the thread where one names one (`crate::answer::refusing_as`), and the
/// process's first of the kind writes its line on stderr ([`kind_line`]), naming the call and
/// `why`.
pub fn refused_as(sel: &'static str, kind: RtRefusal, why: &dyn std::fmt::Display) {
    count_refusal(kind);
    if first(&SAID_RT, kind.at()) {
        say(&kind_line(sel, kind, why));
    }
}

/// The line of the process's first refusal of the kind `kind` ([`refused_as`]): the call `sel`,
/// `why`, the kind's code, and the functions its site names ([`named`]).
fn kind_line(sel: &str, kind: RtRefusal, why: &dyn std::fmt::Display) -> String {
    format!(
        "reims-vgpu: {sel}: metal2vulkan-runtime refused it ({why}); refused ({}), the first of \
         its kind{}\n",
        kind.code(),
        named()
    )
}

/// The functions a refusal's site names on the first line of each kind it writes, each by its
/// role in what runs it (`kernel`, `vertex`, `fragment`; `object` and `mesh` of a mesh state)
/// ([`naming_functions`]).
pub type Named = Vec<(&'static str, metal2vulkan::library::Function)>;

thread_local! {
    /// The functions the first lines written on the thread name, while a site that names them
    /// runs ([`naming_functions`]).
    static NAMING: RefCell<Named> = const { RefCell::new(Vec::new()) };
}

/// Runs `refuse` with the first line of each kind it writes on this thread naming `functions`
/// ([`named`]): a refused dispatch's kernel, a refused draw's or a refused state's functions, so
/// the user's log says which of the guest's functions rt refused. With none, `refuse` runs under
/// what a site around it names. It counts nothing, and no `NSError` names them.
pub fn naming_functions<T>(functions: Named, refuse: impl FnOnce() -> T) -> T {
    /// Puts back what the site around it named, however `refuse` returns.
    struct Restore(Named);
    impl Drop for Restore {
        fn drop(&mut self) {
            let was = std::mem::take(&mut self.0);
            NAMING.with(|n| *n.borrow_mut() = was);
        }
    }
    if functions.is_empty() {
        return refuse();
    }
    let _restore = Restore(NAMING.with(|n| n.replace(functions)));
    refuse()
}

/// The function of a stage's `source` that [`naming_functions`] names: none of a compiled
/// function's.
pub fn stage_function(source: &rt::StageSource) -> Option<metal2vulkan::library::Function> {
    match source {
        rt::StageSource::Function(f) => Some(f.function.clone()),
        rt::StageSource::Compiled(_) => None,
    }
}

/// The tail of a first line of the functions [`naming_functions`] names on this thread: `; ` and
/// each one's role and name as the boot diagnostics name a refused state's functions
/// (`super::diag::function_name`: its library's file, `<its SHA-256's first 16 hex
/// digits>.metallib`, its index and its name), split by `, `; nothing where none is named.
fn named() -> String {
    NAMING.with(|n| {
        let mut out = String::new();
        for (at, (role, function)) in n.borrow().iter().enumerate() {
            let split = if at == 0 { "; " } else { ", " };
            let name = super::diag::function_name(function);
            let _ = write!(out, "{split}{role} {name}");
        }
        out
    })
}

/// The functions [`naming_functions`] names on this thread as the boot diagnostics' refusal keys
/// hold them (`super::diag::KeyFunction`): each by its role, its library's hash and its index, and
/// no name; none where none is named.
pub(super) fn named_keys() -> Vec<super::diag::KeyFunction> {
    NAMING.with(|n| {
        (n.borrow().iter())
            .map(|(role, function)| super::diag::KeyFunction::of(role, function))
            .collect()
    })
}

/// Counts a creator's refusal of the kind `kind` again, with no line: a call answered from a
/// refusal the process met before (the device's, which is made once), noted for core as a gap
/// of the platform's own as [`refuse_create`] does.
pub fn refused_again(kind: RtRefusal) {
    crate::core::platform::device::note_relayed_gap();
    count_refusal(kind);
}

/// Counts a refusal of the kind `kind` under its code, and under the code of the make that runs on
/// the thread where one names one (`crate::answer::refusing_as`).
fn count_refusal(kind: RtRefusal) {
    crate::answer::count_kind(kind.code());
    if let Some(code) = crate::answer::refusing() {
        crate::answer::count_kind(code);
    }
}

/// [`rt_refused`] of a creator, which answers nil where Metal makes the object: noted for core as
/// a gap of the platform's own, as [`refuse_create`] does.
pub fn rt_refused_create(sel: &'static str, error: &rt::Error) -> RtRefusal {
    crate::core::platform::device::note_relayed_gap();
    rt_refused(sel, error)
}

/// [`rt_refused`] of a call that makes or sizes a texture: a refusal of the texture for its pixel
/// format (`Gap::TextureFormat`) is counted under its format's code ([`format_code`]), and the
/// process's first of each format writes its line on stderr; a heap texture's place that Metal
/// places under [`HEAP_TEXTURE_PLACE`]; any other under its kind's. Returns the kind
/// ([`RtRefusal::Gap`] for a format).
pub fn rt_refused_texture(sel: &'static str, error: &rt::Error) -> RtRefusal {
    if let Some(kind) = heap_texture_place(sel, error) {
        return kind;
    }
    let rt::Error::Unsupported(rt::Gap::TextureFormat { pixel_format }) = error else {
        return rt_refused(sel, error);
    };
    let (code, first) = format_code(pixel_format.as_raw());
    crate::answer::count_kind(code);
    diagnosed(sel, error);
    if first {
        say(&format!(
            "reims-vgpu: {sel}: metal2vulkan-runtime refused it ({error}); refused ({code}), the \
             first of its format\n"
        ));
    }
    RtRefusal::Gap
}

/// The code a pipeline state's imageblock length rt refuses is counted under
/// ([`rt_refused_imageblock`]).
pub const IMAGEBLOCK_LENGTH: &str = "rt-refused-imageblock-length";

/// [`rt_refused`] of a pipeline state's imageblock length: rt's refusal of the length
/// (`Gap::ImageblockLength`) is counted under [`IMAGEBLOCK_LENGTH`], apart from rt's other gaps,
/// and the process's first writes its line on stderr; any other under its kind's.
pub fn rt_refused_imageblock(sel: &'static str, error: &rt::Error) {
    static SAID_IMAGEBLOCK: AtomicBool = AtomicBool::new(false);
    if !matches!(error, rt::Error::Unsupported(rt::Gap::ImageblockLength)) {
        rt_refused(sel, error);
        return;
    }
    crate::answer::count_kind(IMAGEBLOCK_LENGTH);
    diagnosed(sel, error);
    if !SAID_IMAGEBLOCK.swap(true, Ordering::Relaxed) {
        say(&format!(
            "reims-vgpu: {sel}: metal2vulkan-runtime refused it ({error}); refused \
             ({IMAGEBLOCK_LENGTH}), the first of its kind\n"
        ));
    }
}

/// The code a heap texture rt refuses on its place in its block, where Metal places it in its heap,
/// is counted under ([`heap_texture_place`]).
pub const HEAP_TEXTURE_PLACE: &str = "rt-refused-heap-texture-place";

/// [`rt_refused_texture`] of rt's `HeapTexturePlace` where Metal's own rule places the texture:
/// its offset a multiple of the alignment rt answers for it (as
/// `heapTextureSizeAndAlignWithDescriptor:` answers it), and its bytes inside the heap. rt binds
/// the image at its offset in the heap's block, the heap's offset there added, and refuses one
/// whose place there is off the image's alignment: a heap pinned in a block at an offset that is
/// not a multiple of it. Counted under
/// [`HEAP_TEXTURE_PLACE`] and under the code of the make that runs on the thread where one names
/// one, apart from Metal's refusals, and the process's first writes its line on stderr; its kind
/// rt's own rule's, [`RtRefusal::Ours`]. `None` for any other error, which the caller counts.
fn heap_texture_place(sel: &'static str, error: &rt::Error) -> Option<RtRefusal> {
    static SAID_HEAP_PLACE: AtomicBool = AtomicBool::new(false);
    if !metal_places(error) {
        return None;
    }
    crate::answer::count_kind(HEAP_TEXTURE_PLACE);
    diagnosed(sel, error);
    if let Some(code) = crate::answer::refusing() {
        crate::answer::count_kind(code);
    }
    if !SAID_HEAP_PLACE.swap(true, Ordering::Relaxed) {
        say(&format!(
            "reims-vgpu: {sel}: metal2vulkan-runtime refused it ({error}), where Metal places it; \
             refused ({HEAP_TEXTURE_PLACE}), the first of its kind\n"
        ));
    }
    Some(RtRefusal::Ours)
}

/// Whether `error` is rt's `HeapTexturePlace` of a place Metal's own rule places
/// ([`heap_texture_place`]): its offset a multiple of the alignment, and its bytes inside the
/// heap.
fn metal_places(error: &rt::Error) -> bool {
    let rt::Error::Refused(rt::Refusal::HeapTexturePlace {
        offset,
        size,
        align,
        heap_len,
    }) = *error
    else {
        return false;
    };
    let aligned = align != 0 && offset.is_multiple_of(align);
    aligned && offset.checked_add(size).is_some_and(|end| end <= heap_len)
}

/// rt's error of the translator's `error` where rt's own call did not give it, as rt makes one
/// for a sampler state or a device: naming no function.
pub fn translation(error: metal2vulkan::error::Error) -> rt::Error {
    rt::Error::Translation {
        error,
        functions: Box::default(),
    }
}

/// [`rt_refused_create`] of a creator with an `error:` out-parameter: the `NSError` it answers, in
/// [`RT_DOMAIN`], code 0, naming rt's error as [`said`] gives it and its kind's code.
pub fn rt_refused_error(sel: &'static str, error: &rt::Error) -> NsErrorText {
    let kind = rt_refused_create(sel, error);
    rt_error(kind, &said(error))
}

/// [`refused_as`] of a creator with an `error:` out-parameter, noted for core as a gap of the
/// platform's own as [`refuse_create`] does: the `NSError` it answers, as [`rt_refused_error`]'s.
pub fn refused_error_as(
    sel: &'static str,
    kind: RtRefusal,
    why: &dyn std::fmt::Display,
) -> NsErrorText {
    crate::core::platform::device::note_relayed_gap();
    refused_as(sel, kind, why);
    rt_error(kind, why)
}

/// The `NSError` of a refusal of the kind `kind` for `why`.
fn rt_error(kind: RtRefusal, why: &dyn std::fmt::Display) -> NsErrorText {
    NsErrorText {
        domain: RT_DOMAIN.to_owned(),
        code: 0,
        description: format!("metal2vulkan-runtime refused it: {why} ({})", kind.code()),
    }
}

/// [`rt_refused_texture`] of a creator, which answers nil where Metal makes the texture: noted
/// for core as a gap of the platform's own, as [`refuse_create`] does.
pub fn rt_refused_texture_create(sel: &'static str, error: &rt::Error) -> RtRefusal {
    crate::core::platform::device::note_relayed_gap();
    rt_refused_texture(sel, error)
}

/// The code of a texture refused for its pixel format `raw`, `rt-refused-texture-format-<raw>`,
/// and whether this is the process's first ask of it. Each format's code is made once and kept
/// for the process (the counts keep codes as `&'static str`): at most one per pixel format rt
/// names.
fn format_code(raw: u32) -> (&'static str, bool) {
    static CODES: Mutex<BTreeMap<u32, &'static str>> = Mutex::new(BTreeMap::new());
    let mut codes = CODES.lock();
    if let Some(&code) = codes.get(&raw) {
        return (code, false);
    }
    let code: &'static str = Box::leak(format!("rt-refused-texture-format-{raw}").into_boxed_str());
    codes.insert(raw, code);
    (code, true)
}

/// Whether this is the first time `said[at]` is asked: sets it.
fn first(said: &[AtomicBool], at: usize) -> bool {
    said.get(at)
        .is_some_and(|said| !said.swap(true, Ordering::Relaxed))
}

/// Writes `line` on stderr in one write, so it never interleaves with another thread's.
pub(super) fn say(line: &str) {
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

    /// rt's refusal is counted under its kind's code, and under the code of a make running on the
    /// thread that names one; a creator's is noted for core as a gap.
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
        let making = "test-refused-making";
        let (made, ours) = (counted(code), counted(making));
        crate::answer::refusing_as(making, || rt_refused("test", &error));
        crate::answer::refusing_as(making, || refused_again(RtRefusal::Quiesced));
        assert_eq!(counted(making), ours + 2);
        assert!(counted(code) >= made + 2, "{code}");
    }

    /// rt's refusal of a texture for its format is counted under the format's own code, each
    /// format its own; any other texture refusal under its kind's.
    #[test]
    fn a_texture_s_format_refusal_is_counted_by_format() {
        let gap = rt::Error::Unsupported(rt::Gap::TextureFormat {
            pixel_format: rt::PixelFormat::GBGR422,
        });
        let code = "rt-refused-texture-format-240";
        let (before, gaps) = (counted(code), counted(RtRefusal::Gap.code()));
        assert_eq!(rt_refused_texture("test", &gap), RtRefusal::Gap);
        let (kind, relayed) = relaying_gaps(|| rt_refused_texture_create("test", &gap));
        assert_eq!(kind, RtRefusal::Gap);
        assert!(relayed);
        assert!(counted(code) >= before + 2, "{code}");
        assert_eq!(format_code(240), (code, false));
        assert_eq!(format_code(241).0, "rt-refused-texture-format-241");
        assert_eq!(counted(RtRefusal::Gap.code()), gaps);
        let code = RtRefusal::Quiesced.code();
        let before = counted(code);
        assert_eq!(
            rt_refused_texture("test", &rt::Error::Quiesced),
            RtRefusal::Quiesced
        );
        assert!(counted(code) > before, "{code}");
    }

    /// rt's refusal of an imageblock length is counted under its own code, not with rt's other
    /// gaps; any other refusal of the length under its kind's.
    #[test]
    fn an_imageblock_length_refusal_is_counted_apart() {
        let gap = rt::Error::Unsupported(rt::Gap::ImageblockLength);
        let (before, gaps) = (counted(IMAGEBLOCK_LENGTH), counted(RtRefusal::Gap.code()));
        rt_refused_imageblock("imageblockSampleLength", &gap);
        rt_refused_imageblock("imageblockSampleLength", &gap);
        assert_eq!(counted(IMAGEBLOCK_LENGTH), before + 2);
        assert_eq!(counted(RtRefusal::Gap.code()), gaps);
        let code = RtRefusal::Quiesced.code();
        let before = counted(code);
        rt_refused_imageblock("imageblockSampleLength", &rt::Error::Quiesced);
        assert_eq!(counted(code), before + 1);
    }

    /// rt's refusal of a heap texture's place that Metal's own rule places (its offset a multiple
    /// of the image's alignment, its bytes inside the heap) is counted under its own code, not as
    /// Metal's refusal, and under the code of a make running on the thread, its kind rt's own
    /// rule's; one off the alignment or past the heap's end is Metal's refusal, though rt
    /// declares its own basis for every heap texture's place.
    #[test]
    fn a_heap_texture_s_place_is_counted_apart_where_metal_places_it() {
        let place = |offset, size| {
            rt::Error::Refused(rt::Refusal::HeapTexturePlace {
                offset,
                size,
                align: 0x10000,
                heap_len: 0x20_0000,
            })
        };
        let metal = RtRefusal::Refused.code();
        let (before, metals) = (counted(HEAP_TEXTURE_PLACE), counted(metal));
        let sel = "newTextureWithDescriptor:offset:";
        assert_eq!(
            rt_refused_texture(sel, &place(0, 0xc_0000)),
            RtRefusal::Ours
        );
        let (kind, relayed) =
            relaying_gaps(|| rt_refused_texture_create(sel, &place(0x14_0000, 0xc_0000)));
        assert_eq!(kind, RtRefusal::Ours);
        assert!(relayed);
        let making = "test-refused-heap-place";
        let ours = counted(making);
        crate::answer::refusing_as(making, || rt_refused_texture(sel, &place(0x10000, 0x1000)));
        assert_eq!(counted(making), ours + 1);
        assert_eq!(counted(HEAP_TEXTURE_PLACE), before + 3);
        assert_eq!(counted(metal), metals);
        // Metal's own refusals: off the alignment, past the heap's end.
        assert_eq!(
            rt_refused_texture(sel, &place(0x8000, 0x1000)),
            RtRefusal::Refused
        );
        assert_eq!(
            rt_refused_texture(sel, &place(0x18_0000, 0xc_0000)),
            RtRefusal::Refused
        );
        assert_eq!(counted(HEAP_TEXTURE_PLACE), before + 3);
        assert!(counted(metal) >= metals + 2, "{metal}");
        // rt declares its own basis for each; the platform's kind tells Metal's own rule apart.
        let placed = rt::Refusal::HeapTexturePlace {
            offset: 0,
            size: 0x1000,
            align: 0x10000,
            heap_len: 0x20_0000,
        };
        assert_eq!(placed.basis(), rt::RefusalBasis::Ours);
        assert_eq!(RtRefusal::of(&place(0, 0x1000)), RtRefusal::Ours);
        assert_eq!(RtRefusal::of(&place(0x8000, 0x1000)), RtRefusal::Refused);
    }

    /// rt's refusal is counted by the basis rt declares for it: a rule of rt's own under
    /// `rt-refused-ours`, Metal's, documented or measured, under `rt-refused-metal`, and the
    /// device's, which no refusal of rt's declares, where every refusal of rt's was counted
    /// before, as Metal's.
    #[test]
    fn rt_s_refusal_is_counted_by_its_basis() {
        use rt::RefusalBasis as Basis;
        let arms = [
            (Basis::Ours, RtRefusal::Ours, "rt-refused-ours"),
            (
                Basis::MetalDocumented,
                RtRefusal::Refused,
                "rt-refused-metal",
            ),
            (Basis::MetalMeasured, RtRefusal::Refused, "rt-refused-metal"),
            (Basis::Device, RtRefusal::Refused, "rt-refused-metal"),
        ];
        for (basis, kind, code) in arms {
            assert_eq!(RtRefusal::of_basis(basis), kind, "{basis:?}");
            assert_eq!(kind.code(), code, "{basis:?}");
        }
        // One refusal of rt's of each basis it declares for refusals, each counted under its
        // basis's code.
        let refusals = [
            (rt::Refusal::WaitOnOwnThread, Basis::Ours),
            (rt::Refusal::TextureBufferNotOverBuffer, Basis::Ours),
            (rt::Refusal::Committed, Basis::MetalDocumented),
            (rt::Refusal::EmptyBuffer, Basis::MetalMeasured),
        ];
        for (refusal, basis) in refusals {
            assert_eq!(refusal.basis(), basis, "{refusal:?}");
            let error = rt::Error::Refused(refusal);
            let kind = RtRefusal::of_basis(basis);
            assert_eq!(error.basis(), Some(basis), "{refusal:?}");
            assert_eq!(RtRefusal::of(&error), kind, "{refusal:?}");
            let (before, other) = match kind {
                RtRefusal::Ours => (counted("rt-refused-ours"), counted("rt-refused-metal")),
                _ => (counted("rt-refused-metal"), counted("rt-refused-ours")),
            };
            assert_eq!(rt_refused("test", &error), kind, "{refusal:?}");
            let (after, other_after) = match kind {
                RtRefusal::Ours => (counted("rt-refused-ours"), counted("rt-refused-metal")),
                _ => (counted("rt-refused-metal"), counted("rt-refused-ours")),
            };
            assert_eq!(after, before + 1, "{refusal:?}");
            assert_eq!(other_after, other, "{refusal:?}");
        }
        // A limit and a gap of rt's own basis stay under their own kinds.
        let limit = rt::Error::Limit(rt::Limit::BlockIds);
        assert_eq!(limit.basis(), Some(Basis::Ours));
        assert_eq!(RtRefusal::of(&limit), RtRefusal::Limit);
        let gap = rt::Error::Unsupported(rt::Gap::ImageblockLength);
        assert_eq!(gap.basis(), Some(Basis::Ours));
        assert_eq!(RtRefusal::of(&gap), RtRefusal::Gap);
    }

    /// A creator with an `error:` out-parameter answers an `NSError` naming the piece and its
    /// code, counted and noted as a gap.
    #[test]
    fn a_refused_creator_answers_an_nserror() {
        let (error, relayed) =
            relaying_gaps(|| refused_error(Unbuilt::DynamicLibrary, "newDynamicLibrary:error:"));
        assert!(relayed);
        assert_eq!(
            (error.domain.as_str(), error.code),
            ("metal2vulkan-runtime", 0)
        );
        assert_eq!(
            error.description,
            "a dynamic library is not built on metal2vulkan-runtime (rt-unbuilt-dynamic-library)"
        );
        let (error, relayed) = relaying_gaps(|| rt_refused_error("test", &rt::Error::Quiesced));
        assert!(relayed);
        assert_eq!(error.domain, RT_DOMAIN);
        assert!(
            error.description.ends_with("(rt-refused-quiesced)"),
            "{error}"
        );
        let (error, relayed) =
            relaying_gaps(|| refused_error_as("test", RtRefusal::Limit, &"full"));
        assert!(relayed);
        assert_eq!(
            error.description,
            "metal2vulkan-runtime refused it: full (rt-refused-limit)"
        );
        let code = RtRefusal::NoDevice.code();
        let before = counted(code);
        let ((), relayed) = relaying_gaps(|| refused_again(RtRefusal::NoDevice));
        assert!(relayed);
        assert!(counted(code) > before, "{code}");
    }

    /// The first refusal of each of rt's kinds by name has a line of its own, naming the call,
    /// rt's error as it displays and as it debugs, the kind, its code and what the call asked of
    /// rt where its site names it; the kind's next refusal writes none.
    #[test]
    fn the_first_refusal_of_each_named_kind_has_its_own_line() {
        let samples = rt::Error::Limit(rt::Limit::TextureSamples {
            sample_count: 4,
            counts: 0x1,
        });
        let name = crate::rt_platform::diag::refusal_name(&samples).expect("a limit's name");
        assert_eq!(name, "Limit-TextureSamples");
        let what = "a texture of format 70";
        assert_eq!(
            detail_line("newTextureWithDescriptor:", &samples, &name, Some(&what)),
            format!(
                "reims-vgpu: newTextureWithDescriptor:: metal2vulkan-runtime refused it \
                 ({samples}; {samples:?}); the first of Limit-TextureSamples (rt-refused-limit): a \
                 texture of format 70\n"
            )
        );
        assert!(
            format!("{samples:?}").contains("sample_count: 4"),
            "{samples:?}"
        );
        let wide = rt::Error::Unsupported(rt::Gap::Draw {
            kind: rt::DrawKind::LineWidth,
        });
        assert_eq!(
            detail_line(
                "drawPrimitives:vertexStart:vertexCount:",
                &wide,
                "Draw-LineWidth",
                None
            ),
            "reims-vgpu: drawPrimitives:vertexStart:vertexCount:: metal2vulkan-runtime refused it \
             (a draw with LineWidth is not built; Unsupported(Draw { kind: LineWidth \
             })); the first of Draw-LineWidth (rt-refused-gap)\n"
        );
        let format = rt::Error::Unsupported(rt::Gap::TextureFormat {
            pixel_format: rt::PixelFormat::BGRA8Unorm,
        });
        let line = detail_line("newTextureWithDescriptor:", &format, "TextureFormat", None);
        assert!(
            line.ends_with("the first of TextureFormat (rt-refused-texture-format-80)\n"),
            "{line}"
        );
        let kind = "test-first-named";
        assert!(first_named(kind.to_owned()));
        assert!(!first_named(kind.to_owned()));
    }

    /// The first of each feature the translator refuses has its line, naming the feature and what
    /// it needs, as structure alone, and the attachment it is tied to; no name of the guest's (a
    /// function's, a symbol's) is in it. An error of the translator's that refuses no feature is
    /// named by its variant.
    #[test]
    fn a_translator_refusal_s_line_names_its_feature_and_no_name() {
        use metal2vulkan::error::{Attachment, DiagName, Error as Translator, Feature, Location};
        use metal2vulkan::error::{LowerFamily, Unsupported};
        let at = Location::function(DiagName::new("a_guest_function"), Some(3), Some(9));
        let refused = Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture))
            .at(at.with_attachment(Attachment::Color(1)))
            .naming(DiagName::new("air.a_guest_symbol"));
        let error = rt::Error::Translation {
            error: Translator::Unsupported(refused),
            functions: Box::default(),
        };
        let name = crate::rt_platform::diag::refusal_name(&error).expect("a translator's name");
        assert_eq!(name, "Translation-Lowering");
        let sel = "newRenderPipelineStateWithDescriptor:error:";
        assert_eq!(
            detail_line(sel, &error, &name, None),
            "reims-vgpu: newRenderPipelineStateWithDescriptor:error:: metal2vulkan-runtime refused \
             it (the translator refused the feature Lowering(Texture), needing NotImplemented, at \
             the color attachment 1); the first of Translation-Lowering (rt-refused-translation)\n"
        );
        let cancelled = rt::Error::Translation {
            error: Translator::Cancelled,
            functions: Box::default(),
        };
        let name = crate::rt_platform::diag::refusal_name(&cancelled).expect("a translator's name");
        assert_eq!(
            detail_line(sel, &cancelled, &name, None),
            "reims-vgpu: newRenderPipelineStateWithDescriptor:error:: metal2vulkan-runtime refused \
             it (the translator refused it); the first of Translation-Cancelled \
             (rt-refused-translation)\n"
        );
    }

    /// No line the library writes of the translator's refusals, nor the `NSError` a creator
    /// answers of one, names what the guest named in the refusal: refusals whose display names a
    /// function, a symbol and an entry point (`secret_fn`, a name of our own) write the first line
    /// of their kind, the first line of each name and an `NSError` each, which give the refusal as
    /// structure alone (in a child, whose stderr is its own). What a site names of the guest's is
    /// its functions alone, on its first lines (`a_site_s_functions_are_named_on_its_first_lines`).
    #[test]
    fn no_line_of_a_translator_refusal_names_the_guest_s_function() {
        use crate::core::testkit::death::{Expect, Options, Outcome, run};
        use metal2vulkan::error::{DescriptorError, DiagName, Error as Translator, Feature};
        use metal2vulkan::error::{Location, LowerFamily, RuntimeInput, Unsupported};
        const SECRET: &str = "secret_fn";
        let refusals = || {
            let at = Location::function(DiagName::new(SECRET), Some(3), Some(9));
            let lowering = Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture))
                .at(at)
                .naming(DiagName::new(SECRET));
            [
                Translator::Unsupported(lowering),
                Translator::InvalidDescriptor(DescriptorError::EntryNotFound {
                    requested: DiagName::new(SECRET),
                    available: 2,
                }),
                Translator::NeedsInput(RuntimeInput::LinkedFunction {
                    symbol: DiagName::new(SECRET),
                }),
            ]
            .map(|error| rt::Error::Translation {
                error,
                functions: Box::default(),
            })
        };
        for error in refusals() {
            assert!(error.to_string().contains(SECRET), "{error}");
        }
        run(
            Expect::Any,
            Options::default(),
            module_path!(),
            "no_line_of_a_translator_refusal_names_the_guest_s_function",
            |_| {
                for error in refusals() {
                    rt_refused("newRenderPipelineStateWithDescriptor:error:", &error);
                    let sel = "newComputePipelineStateWithFunction:error:";
                    let answered = rt_refused_error(sel, &error);
                    say(&format!("test: NSError: {}\n", answered.description));
                }
            },
            |r| {
                assert_eq!(r.outcome(), &Outcome::Returned, "{}", r.stderr());
                let out = r.stderr();
                // A line ends with what was written: the child's first line starts with
                // libtest's own "test NAME ... ".
                let lines: Vec<&str> = out.lines().collect();
                let first_of_kind = "reims-vgpu: newRenderPipelineStateWithDescriptor:error:: \
                                     metal2vulkan-runtime refused it (the translator refused the \
                                     feature Lowering(Texture), needing NotImplemented); refused \
                                     (rt-refused-translation), the first of its kind";
                assert!(lines.iter().any(|l| l.ends_with(first_of_kind)), "{out}");
                for name in ["Lowering", "InvalidDescriptor", "NeedsInput"] {
                    let first = format!("the first of Translation-{name} (rt-refused-translation)");
                    assert!(lines.iter().any(|l| l.ends_with(&first)), "{first}: {out}");
                }
                let answered = |said: &str| {
                    format!(
                        "test: NSError: metal2vulkan-runtime refused it: {said} \
                         (rt-refused-translation)"
                    )
                };
                let lowering = "the translator refused the feature Lowering(Texture), needing \
                                NotImplemented";
                let answers = [
                    lowering,
                    "the translator refused it",
                    "the translator refused it",
                ]
                .map(answered);
                let said: Vec<&str> = (lines.iter())
                    .filter_map(|l| l.find("test: NSError: ").map(|at| &l[at..]))
                    .collect();
                assert_eq!(said, answers, "{out}");
                assert!(!out.contains(SECRET), "{out}");
            },
        );
    }

    /// The refusals counted under a code of their own (a pixel format's, an imageblock length's, a
    /// heap texture's place Metal places) write their name's line too, as every refusal the boot
    /// diagnostics count by name does.
    #[test]
    fn a_refusal_of_a_code_of_its_own_writes_its_name_s_line() {
        let format = rt::Error::Unsupported(rt::Gap::TextureFormat {
            pixel_format: rt::PixelFormat::BGRA8Unorm,
        });
        rt_refused_texture("newTextureWithDescriptor:", &format);
        assert!(!first_named("TextureFormat".to_owned()));
        rt_refused_imageblock(
            "imageblockSampleLength",
            &rt::Error::Unsupported(rt::Gap::ImageblockLength),
        );
        assert!(!first_named("ImageblockLength".to_owned()));
        let place = rt::Error::Refused(rt::Refusal::HeapTexturePlace {
            offset: 0,
            size: 0x1000,
            align: 0x1000,
            heap_len: 0x2000,
        });
        assert!(metal_places(&place));
        rt_refused_texture("newTextureWithDescriptor:offset:", &place);
        assert!(!first_named("Refused-HeapTexturePlace".to_owned()));
        assert_eq!(counted_code(&place), HEAP_TEXTURE_PLACE);
    }

    /// The first refusal's line names the call, the piece and its code, and what the call asked
    /// of the piece where its site names it.
    #[test]
    fn the_first_refusal_s_line_names_the_call_and_the_code() {
        assert_eq!(
            refusal_line(Unbuilt::Buffer, "newBufferWithDescriptor:", None),
            "reims-vgpu: newBufferWithDescriptor:: a Private buffer of bytes is not built on \
             metal2vulkan-runtime; refused (rt-unbuilt-buffer), the first of its kind\n"
        );
        assert_eq!(
            refusal_line(Unbuilt::RateMap, "test", Some(&"a map of 2 layers")),
            "reims-vgpu: test: a rasterization rate map is not built on metal2vulkan-runtime; \
             refused (rt-unbuilt-rate-map), the first of its kind: a map of 2 layers\n"
        );
    }

    /// Where a site names functions ([`naming_functions`]), the first line of each kind written
    /// there names each after what the line names, by its role, its library's file, its index and
    /// its name; a site inside it names its own, and one that names none leaves its names. No line
    /// past the site, and no `NSError`, names them (functions of our own: `add_one`, and
    /// `render_small` and `render_color` of two libraries).
    #[test]
    fn a_site_s_functions_are_named_on_its_first_lines() {
        const ADD: &[u8] = include_bytes!("../../../../tests/shaders/compute_add.metallib");
        const SMALL: &[u8] = include_bytes!("../../../../tests/shaders/render_small.metallib");
        const COLOR: &[u8] = include_bytes!("../../../../tests/shaders/render_color.metallib");
        let function = |bytes: &[u8], name: &str| {
            let library = rt::Library::from_metallib(bytes).expect("a library");
            let hash: String = (library.content_hash().iter().take(8))
                .map(|b| format!("{b:02x}"))
                .collect();
            let function = library.function(name).expect("the function");
            let said = format!("{hash}.metallib #{} {name}", function.index());
            (function, said)
        };
        let (kernel, kernel_said) = function(ADD, "add_one");
        let (vertex, vertex_said) = function(SMALL, "render_small");
        let (fragment, fragment_said) = function(COLOR, "render_color");
        let sel = "dispatchThreads:threadsPerThreadgroup:";
        let wide = rt::Error::Unsupported(rt::Gap::Draw {
            kind: rt::DrawKind::LineWidth,
        });
        let lines = || {
            [
                kind_line(sel, RtRefusal::Gap, &said(&wide)),
                detail_line(sel, &wide, "Draw-LineWidth", Some(&"a draw")),
                refusal_line(Unbuilt::ComputeCommand, sel, None),
            ]
        };
        let plain = lines();
        assert_eq!(
            plain[0],
            "reims-vgpu: dispatchThreads:threadsPerThreadgroup:: metal2vulkan-runtime refused it \
             (a draw with LineWidth is not built); refused (rt-refused-gap), the first of its \
             kind\n"
        );
        let with = |tail: &str| {
            let mut lines = plain.clone();
            for line in &mut lines {
                line.insert_str(line.len() - 1, tail);
            }
            lines
        };
        let kernel_tail = format!("; kernel {kernel_said}");
        assert_eq!(
            naming_functions(vec![("kernel", kernel.clone())], lines),
            with(&kernel_tail)
        );
        let drawn = vec![("vertex", vertex), ("fragment", fragment)];
        let draw_tail = format!("; vertex {vertex_said}, fragment {fragment_said}");
        let (inner, around) = naming_functions(vec![("kernel", kernel.clone())], || {
            let inner = naming_functions(drawn, lines);
            assert_eq!(naming_functions(Vec::new(), lines), with(&kernel_tail));
            (inner, lines())
        });
        assert_eq!(inner, with(&draw_tail));
        assert_eq!(around, with(&kernel_tail));
        assert_eq!(lines(), plain);
        let sel = "newComputePipelineStateWithFunction:error:";
        let answered = naming_functions(vec![("kernel", kernel)], || rt_refused_error(sel, &wide));
        assert_eq!(
            answered.description,
            "metal2vulkan-runtime refused it: a draw with LineWidth is not built \
             (rt-refused-gap)"
        );
    }
}
