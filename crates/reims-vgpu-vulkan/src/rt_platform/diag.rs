//! The rt platform's boot diagnostics, which a boot that checks it reads: lines
//! `reims-vgpu: rt-diag ...` on stderr, written with the process's counts ([`write()`], which
//! `crate::answer::write_counts` runs). They hold:
//!
//! - **Gaps by kind.** Each refusal of rt's `Gap`, by the gap's kind, a draw's by its draw kind
//!   and a linear plane pass's by its shape ([`refusal`]), which the counts line holds as one
//!   `rt-refused-gap`.
//! - **Refusals by variant.** Each refusal of Metal's that rt answers (`Error::Refused`) and each of
//!   its limits (`Error::Limit`), by its variant, `Refused-<variant>` and `Limit-<variant>`
//!   ([`refusal`]), which the counts line holds as `rt-refused-metal` and `rt-refused-limit` with
//!   the platform's own refusals of those kinds; and each of the translator's refusals
//!   (`Error::Translation`), by the feature it refused, `Translation-<feature>` (the feature's
//!   variant), or the translator's error's variant where it refused no feature, which the counts
//!   line holds as one `rt-refused-translation`.
//! - **Attachment reads through a texture.** Each draw rt refused as one
//!   (`Gap::Draw { kind: AttachmentRead }`), by its class ([`attachment_read`]): for each texture
//!   slot of the draw's state whose texture's image a target of the pass renders, the slot's stage
//!   and declared access, how the texture was made, the usage its maker declared, its format and
//!   storage mode, whether it is a view of a surface's shape image, the attachment it meets and
//!   whether a second target renders the image, whether the texture is the attachment's own or a
//!   twin view of one surface's image, the depth write state of a depth or stencil attachment's
//!   draw, the usage its image was made with (a surface's shape image: storage where 0, plain
//!   where `ShaderRead | RenderTarget`), whether a live texture that takes storage lies over the
//!   texture's surface plane ([`surface_texture`]), and whether the draw's process presents to the
//!   display (the compositor) or not (an app). Class A, which rt builds, is the texture and the
//!   attachment's being one texture or twin views of one surface's image. The first refused draw
//!   of each pipeline state, ten at most, writes its line as it is refused.
//! - **Fetched colour attachments.** Each draw rt refused as reading a colour attachment through
//!   its texture (`Gap::Draw { kind: FetchedAttachment }`, framebuffer fetch), by its class
//!   ([`fetched_attachment`]): for each colour target of its pass, as rt does not say which the
//!   fragment function reads, how its texture was made, its texture type and sample count, the
//!   level and slice the pass renders and a view's first, whether it has an index (rt reads a
//!   fetched attachment through its texture's index), whether its image's usage lets shaders read
//!   it and write it (Unknown, 0, as both), the usage its maker declared and its image's, and
//!   whether another target of the pass renders its image. The first refused draw of each
//!   pipeline state, ten at most, writes its line as it is refused.
//! - **Imageblock asks.** Each distinct combination of colour attachment formats (every index)
//!   and sample count of the render states asked `imageblockSampleLength`, with its asks and rt's
//!   answer, or its refusal ([`imageblock`]); its first ask writes its line.
//! - **Over-copies a presented frame.** rt's over-copies, their bytes and its submissions between
//!   two of the display's presents ([`presented`]): mean, worst, and how many frames copied none,
//!   1 to 9, 10 to 99 and 100 or more.
//! - **Memory allocations by site.** rt's `vkAllocateMemory` calls inside the platform's calls
//!   into it (any thread but rt's own block maker), by the site in rt that made them (rt's
//!   `SiteAllocations`, eight sites): for each, the calls, the bytes they asked for and the
//!   longest call since the device was made; the site of the longest call comes first.
//! - **Memory.** The textures rt cleared at their make and the bytes it cleared (rt's
//!   `texture_clears`, `texture_clear_bytes`), and the Private memory it holds now, allocated and
//!   idle, with its blocks (rt's `Gauges`): what a placed shape's image skips (`texture`, "Placed
//!   images for views") and what it holds.
//! - **Textures over an imported block**, which rt keeps in an image of their own and copies the
//!   bytes of into it at each submission that uses them (rt's `over` module), by kind: format,
//!   size, rows, usage, storage mode, and where their block was imported (the selector, the
//!   decoder's message the import was made under, the thread's name without its number, and
//!   whether rt took the pages themselves or a copy of them, `pages` or `copy`: [`imported`]).
//!   For each kind, the textures made, and their uses in submissions
//!   ([`Uses::submitted`]): their bytes, and how many sampled, rendered or blitted it. Their bytes
//!   are not read: the committing thread runs the guest's call, and a hash of them at each commit
//!   held it in proportion to them.
//! - **Imports by origin**, as the textures' kinds name it.
//! - **Imports by path.** Each import of the host's pages a no-copy buffer or heap asked
//!   (`super::buffer::import_pages`), by the kind of the make that asked it
//!   (`crate::answer::making_as`): how many were made by their pointer and as a dma-buf of the
//!   files behind them, and how many were refused as too large for one dma-buf or otherwise, with
//!   their bytes; of those made by their pointer, how many were mapped from their files as too
//!   large for one dma-buf, with their bytes and the most runs of one; the most runs and bytes of
//!   one refused as too large, and the limits of one dma-buf; and every import asked, by size
//!   ([`import_path`]).
//! - **Guest surface textures by declared usage.** Each texture the decoder made, or was refused,
//!   over a surface's plane (`TEXTURE_FROM_IOSURFACE`), by the usage the guest declared, how it
//!   was made (a view of a shape's image, or over the cache's buffer and why) and whether its
//!   process presents to the display ([`surface_texture`]); each kind says whether it takes
//!   storage: usage 0 (Unknown), `ShaderWrite` (0x2) or `ShaderAtomic` (0x20).
//! - **Two formats over one plane.** Each time a guest surface texture is made over a plane while
//!   one of another format is live over it, by the pair of formats, with the most bytes of such a
//!   texture; and each write noted through a texture of one of them while one of the other is
//!   live ([`shape_written`]).
//! - **Views refused over a plain shape.** Each texture over a plain shape (no texture over its
//!   plane takes storage) in a format that is neither the shape's nor its sRGB sibling, by the
//!   pair ([`plain_view_refused`]).
//! - **Render passes over a video plane.** Each render pass the decoder began that takes a view of
//!   a video plane image (a plane shape of a video cache's surface, its exported planes) as an
//!   attachment, by its shape ([`plane_pass`]): one plane alone, the two planes of one picture
//!   alone, with a depth or stencil attachment, with a colour attachment that is not a plane, a
//!   plane as a multisample resolve target, and planes of more than one picture or one plane
//!   twice. Beside each shape some pass had, how many of its passes the compositor's process
//!   (one that presents to the display) began and how many another's, an app's: the host is told
//!   no process's name, nor whether a system framework in it encoded the pass. Its line is
//!   written whether or not a pass was counted.
//!
//! - **Sampler states by border.** Each sampler state made (`newSamplerStateWithDescriptor:`), and
//!   of them each that clamps to its border colour in some direction, by its border colour
//!   ([`sampler`]). A function's constexpr samplers are not counted.
//! - **Exported block makes.** Each of rt's `Device::new_exported_block` calls the platform made
//!   (`super::export`, on Unix; Windows makes none), timed alone with rt's host clock
//!   (`exported_block_made`): their count, those over 5 ms, and the longest with its bytes. Each
//!   is an allocation of its own: rt dedicates an exported block's memory to it.
//! - **rt's timed take parts.** rt's `Timed` metrics of its block takes and makes (`takes`,
//!   `take_lock_waits`, `take_makes`, `take_maps`, `maker_makes`, `zero_writes`,
//!   `long_take_zero_writes`): for each, its calls, those over 5 ms, and its longest call with
//!   that call's bytes and whether it made an allocation of its own.
//! - **rt's pipeline makes.** rt's `VkPipeline`s created; its first-use waits (encoding calls that
//!   waited while the pipeline they needed was made), with their milliseconds and the longest, and
//!   of them those for a compute kernel's pipeline at a threadgroup size, threadgroup memory
//!   lengths or grid tiles none of its states had run with, and those for a render content's first
//!   draw of lines; and the compute pipelines made ahead of their dispatches by the device's
//!   pre-pass (rt's `prepare_dispatch`, reims-vgpu's `graphics::decode::prepass`), with their
//!   milliseconds ([`pipelines_line`]).
//! - **Own indices taken again.** The platform's own texture and sampler indices
//!   (`super::texture::OwnIndices`) taken again after they were given back
//!   ([`own_index_taken_again`]), and those given back ([`own_index_given_back`]), each class's
//!   total, in one line written on the rt platform. An index given back is taken again only once
//!   every command buffer its device committed before the give-back has completed, by the
//!   device's commit order, so none is taken while work committed before its release reads it.
//!
//! Totals, numbers, formats, sizes and usages only: no name, label or byte of the guest's leaves
//! the process, but through the switch below. A diagnostic, which the contract leaves free:
//! nothing the guest observes changes, and a texture's bytes are only read.
//!
//! **Refused pipelines' AIR**, a diagnostic switch that is off by default: where
//! `REIMS_VGPU_REFUSED_AIR_DIR` ([`REFUSED_AIR_VAR`]) names a directory, each pipeline state the
//! translator refused (`Error::Translation`) has the metallib of each of its stages' functions
//! written into `air-<pid>` there, once per library, as `<its SHA-256's first 16 hex
//! digits>.metallib`, and a line in `refused.txt` there naming the creator, each function (its
//! library's file, its index and name) and the refusal, once per creator and functions
//! ([`keep_refused_compute`], [`keep_refused_render`]). Each refused state also has a record in
//! `records.txt` there, once per record: its creator and refusal, then for each stage its
//! function, whether it was made specialized, each function-constant value it was made with (its
//! index, the name its function declares at that index, its type and its bytes in hex) and each
//! constant it declares with no value, the same of each function it links, and each field of
//! the descriptor rt was given, as rt's descriptor holds it. The lines give how many it kept.
//! The AIR and the records are the guest's: they stay on the host that made them, are never
//! written to stderr or the log, and nothing commits them or copies them off.
//!
//! One count is not a diagnostic but a degradation code on the counts line: each use, by a
//! committed command buffer, of a guest surface texture over a buffer in an imported block,
//! which rt copies into an image of its own at each submission where Metal reads the surface's
//! bytes in place, and its bytes ([`SURFACE_PLANE_COPIES_CODE`]). A texture over a device
//! cache's buffer, in a block of rt's own, is not counted: rt copies one only where its rows or
//! place are off the device's linear lines, which the cache lays its planes on
//! (`crate::gpu::resource`). Two more are degradation codes too, so a boot shows whether a view
//! of a plane's texture ever asks a format the plane's image may not hold: each view made of a
//! guest surface texture, or of a view of one, by the view's pixel format, and each one rt
//! refused, by its format ([`plane_view`]); and each guest surface texture made whose declared
//! usage has `PixelFormatView` (0x10) ([`PLANE_PIXEL_FORMAT_VIEW_CODE`]).

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

use metal2vulkan_runtime as rt;

use crate::core::sync::Mutex;

use super::RtObj;
use super::texture::RtTexture;

/// A surface view's declared usage before [`declared`] notes it.
const NONE: u64 = u64::MAX;

/// The most first lines of refused attachment reads a process writes.
const FIRST_LINES: usize = 10;

/// How a texture was made.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Call {
    /// `newTextureWithDescriptor:`.
    Plain,
    /// A shape's image of a surface's plane (the device's `new_texture_for_views`).
    SurfaceImage,
    /// A view of a texture.
    View,
    /// Over a buffer's bytes (`newTextureWithDescriptor:offset:bytesPerRow:`).
    OverBuffer,
    /// A tiled texture of more than one level, which takes a buffer's bytes
    /// (`newTiledTextureWithDescriptor:offset:bytesPerRow:`, `super::tiled`).
    Tiled,
    /// In a heap (`newTextureWithDescriptor:offset:`).
    InHeap,
    /// Over a window swapchain's image (`super::swapchain`).
    Drawable,
}

impl Call {
    fn name(self) -> &'static str {
        match self {
            Call::Plain => "ordinary",
            Call::SurfaceImage => "surface-image",
            Call::View => "view",
            Call::OverBuffer => "over-buffer",
            Call::Tiled => "tiled",
            Call::InHeap => "in-heap",
            Call::Drawable => "drawable",
        }
    }
}

/// What a texture is, kept as it is made (the module's docs).
pub struct Made {
    /// The process's number of it, from 1.
    id: u64,
    /// Its image's number: its own; a view's, its texture's image's; a texture over a buffer's,
    /// the buffer's GPU address with the top bit set, as rt keys the hazards of one in place by its
    /// buffer.
    image: u64,
    call: Call,
    /// Its image is a surface's shape image.
    surface: bool,
    format: u32,
    width: u32,
    height: u32,
    /// Its storage mode, the resource options' bits 4 to 7.
    storage: u64,
    /// The usage its descriptor asked; a view's, its texture's.
    usage: u64,
    /// A surface view's usage as the guest declared it ([`declared`]); [`NONE`] until noted.
    declared: AtomicU64,
    /// A view's first level and slice; 0 for any other.
    level: u32,
    slice: u32,
    /// Where it lies over an imported block: what of it rt copies.
    over: Option<Over>,
    /// The block of the buffer it lies over; a view's, its texture's; `None` for any other.
    block: Option<rt::BlockId>,
    /// A guest surface texture's plane ([`surface_texture`]).
    on_plane: OnceLock<OnPlane>,
    /// It is a view of a guest surface texture, or of a view of one ([`plane_view`]).
    of_plane: AtomicBool,
}

/// The texture of a descriptor, before it is made: what [`Made`] takes of it.
pub struct Shape {
    /// Its `MTLPixelFormat`.
    pub format: u32,
    /// Its width in texels.
    pub width: u32,
    /// Its height in texels.
    pub height: u32,
    /// Its storage mode, the resource options' bits 4 to 7.
    pub storage: u64,
    /// Its descriptor's `MTLTextureUsage`.
    pub usage: u64,
}

/// The process's next texture number.
static NEXT: AtomicU64 = AtomicU64::new(1);

impl Made {
    /// A texture made by `call` (not a view nor over a buffer) of `shape`.
    pub fn new(call: Call, shape: &Shape) -> Arc<Made> {
        let id = NEXT.fetch_add(1, Relaxed);
        Made::of(
            id,
            id,
            call,
            call == Call::SurfaceImage,
            shape,
            (0, 0),
            (None, None),
        )
    }

    /// A view of `of`'s image, in `format`, from its level `level` and slice `slice`.
    pub fn view(of: &Made, format: u32, (level, slice): (u32, u32)) -> Arc<Made> {
        let id = NEXT.fetch_add(1, Relaxed);
        let shape = Shape {
            format,
            width: of.width >> level.min(31),
            height: of.height >> level.min(31),
            storage: of.storage,
            usage: of.usage,
        };
        let made = Made::of(
            id,
            of.image,
            Call::View,
            of.surface,
            &shape,
            (level, slice),
            (None, of.block),
        );
        if of.of_a_plane() {
            made.of_plane.store(true, Relaxed);
        }
        made
    }

    /// The block of the buffer it lies over, a view's its texture's: `None` for any other.
    pub fn block(&self) -> Option<rt::BlockId> {
        self.block
    }

    /// Whether it is a guest surface texture, or a view of one ([`plane_view`]).
    fn of_a_plane(&self) -> bool {
        self.on_plane.get().is_some() || self.of_plane.load(Relaxed)
    }

    /// A texture of `shape` over `buffer`'s bytes from `offset`, rows `bytes_per_row` apart:
    /// with what rt copies of it where the buffer's block was imported ([`imported`]).
    pub fn over_buffer(
        buffer: &rt::Buffer,
        shape: &Shape,
        offset: u64,
        bytes_per_row: u64,
    ) -> Arc<Made> {
        let id = NEXT.fetch_add(1, Relaxed);
        let image = buffer.gpu_address() | (1 << 63);
        let over = Over::of(buffer, shape, offset, bytes_per_row);
        let block = Some(buffer.block());
        Made::of(
            id,
            image,
            Call::OverBuffer,
            false,
            shape,
            (0, 0),
            (over, block),
        )
    }

    fn of(
        id: u64,
        image: u64,
        call: Call,
        surface: bool,
        shape: &Shape,
        (level, slice): (u32, u32),
        (over, block): (Option<Over>, Option<rt::BlockId>),
    ) -> Arc<Made> {
        Arc::new(Made {
            id,
            image,
            call,
            surface,
            format: shape.format,
            width: shape.width,
            height: shape.height,
            storage: shape.storage,
            usage: shape.usage,
            declared: AtomicU64::new(NONE),
            level,
            slice,
            over,
            block,
            on_plane: OnceLock::new(),
            of_plane: AtomicBool::new(false),
        })
    }

    /// How it was made, a view of a surface's image named as one.
    fn call_name(&self) -> &'static str {
        match (self.call, self.surface) {
            (Call::View, true) => "surface-view",
            (call, _) => call.name(),
        }
    }

    /// How it was made, as the boot diagnostics' lines name it: `ordinary`, `surface-image`,
    /// `view`, `surface-view` (a view of a surface's shape image, as a guest surface texture is),
    /// `over-buffer`, `tiled`, `in-heap` or `drawable` (over a window swapchain's image).
    pub fn made_as(&self) -> &'static str {
        self.call_name()
    }

    /// What it and `other`, two textures one call names (a copy's ends), share, as structure
    /// alone: `one texture`; `one image` (a texture and a view of it, or two views of one texture,
    /// which rt takes as one texture, so the same level and slice of both is one subresource);
    /// `one buffer` (two textures over one buffer's bytes, or views of them); `one surface plane`
    /// (two guest surface textures over one IOSurface plane, of two images); else `no image`.
    pub fn shared_with(&self, other: &Made) -> &'static str {
        let plane = |m: &Made| m.on_plane.get().map(|on| on.plane);
        if self.id == other.id {
            "one texture"
        } else if self.image == other.image && self.image >> 63 == 0 {
            "one image"
        } else if self.image == other.image {
            "one buffer"
        } else if plane(self).is_some() && plane(self) == plane(other) {
            "one surface plane"
        } else {
            "no image"
        }
    }

    /// The usage its maker declared: a surface view's, the guest's; any other's, its
    /// descriptor's.
    fn declared_usage(&self) -> u64 {
        match self.declared.load(Relaxed) {
            NONE => self.usage,
            usage => usage,
        }
    }
}

impl Drop for Made {
    /// A guest surface texture leaves its plane's live ones.
    fn drop(&mut self) {
        let Some(on) = self.on_plane.get() else {
            return;
        };
        let mut planes = PLANES.lock();
        let Some(p) = planes.get_mut(&on.plane) else {
            return;
        };
        if on.storage {
            p.storage = p.storage.saturating_sub(1);
        }
        if let Some(n) = p.formats.get_mut(&self.format) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                p.formats.remove(&self.format);
            }
        }
        if p.formats.is_empty() {
            planes.remove(&on.plane);
        }
    }
}

/// Notes the guest's declared `usage` of `texture`, a view of a surface's shape image, where it is
/// one of the rt platform's (the decoder's surface shapes).
pub fn declared<T: 'static>(texture: &T, usage: u64) {
    let any: &dyn std::any::Any = texture;
    if let Some(t) = any.downcast_ref::<RtObj<RtTexture>>() {
        t.made().declared.store(usage, Relaxed);
    }
}

// ----- Guest surface textures.

/// The usages that take storage on a shape's image, beside usage 0 (Unknown): `ShaderWrite`
/// and `ShaderAtomic`.
const STORAGE_USAGES: u64 = 0x2 | 0x20;

/// A surface's plane: the block of its device cache, else its buffer's object id with the top
/// bit set; and the plane's index.
type Plane = (u64, u32);

/// A guest surface texture's plane, and whether it takes storage.
#[derive(Clone, Copy)]
struct OnPlane {
    plane: Plane,
    storage: bool,
}

/// A plane's live guest textures: how many take storage, and how many there are of each format.
#[derive(Default)]
struct PlaneLive {
    storage: u64,
    formats: BTreeMap<u32, u64>,
}

/// The planes with live guest textures.
static PLANES: Mutex<BTreeMap<Plane, PlaneLive>> = Mutex::new(BTreeMap::new());

/// Two formats live over one plane at once: the meetings, the most bytes of a texture at one, and
/// the writes through a texture of one while one of the other was live.
#[derive(Default)]
struct PairTotals {
    met: u64,
    bytes_most: u64,
    writes: u64,
}

/// Two formats over one plane, the lesser first, with their totals.
static PAIRS: Mutex<BTreeMap<(u32, u32), PairTotals>> = Mutex::new(BTreeMap::new());

/// A kind of guest surface texture, the key of their totals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct SurfaceKind {
    context: u32,
    usage: u64,
    form: &'static str,
    made: bool,
}

/// The guest surface textures made or refused, by kind.
static SURFACES: Mutex<BTreeMap<SurfaceKind, u64>> = Mutex::new(BTreeMap::new());

/// The views refused over a plain shape, by (shape's format, view's format).
static PLAIN_REFUSED: Mutex<BTreeMap<(u64, u64), u64>> = Mutex::new(BTreeMap::new());

/// Whether a texture of declared `usage` takes storage on a shape's image.
fn takes_storage(usage: u64) -> bool {
    usage == 0 || usage & STORAGE_USAGES != 0
}

/// The texture the decoder made of a guest surface texture, for [`shape_written`]: where it is
/// one of the rt platform's.
#[derive(Clone)]
pub struct Token(Arc<Made>);

impl std::fmt::Debug for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Token({})", self.0.id)
    }
}

/// [`Token`] of `texture`, where it is one of the rt platform's.
pub fn token<T: 'static>(texture: &T) -> Option<Token> {
    let any: &dyn std::any::Any = texture;
    let t = any.downcast_ref::<RtObj<RtTexture>>()?;
    Some(Token(Arc::clone(t.made())))
}

/// Counts a guest surface texture the decoder made (`texture`) or was refused (`None`) for
/// context `context` over plane `plane` of a surface (the block of its device cache, else its
/// buffer's object id with the top bit set, and the plane's index), of the `usage` the guest
/// declared and `bytes` bytes, made as `form` says (a view of a shape's image, or over the
/// buffer and why). Where it is one of the rt platform's, notes its plane, whose live textures it
/// is one of until it is let go, and counts each other format live over it ([`PairTotals`]).
pub fn surface_texture<T: 'static>(
    texture: Option<&T>,
    context: u32,
    plane: (u64, u32),
    (usage, bytes): (u64, u64),
    form: &'static str,
) {
    *SURFACES
        .lock()
        .entry(SurfaceKind {
            context,
            usage,
            form,
            made: texture.is_some(),
        })
        .or_insert(0) += 1;
    if let Some(Token(made)) = texture.and_then(token) {
        on_plane(&made, plane, usage, bytes);
        if usage & PIXEL_FORMAT_VIEW != 0 {
            crate::answer::count_kind(PLANE_PIXEL_FORMAT_VIEW_CODE);
        }
    }
}

/// `MTLTextureUsagePixelFormatView`.
const PIXEL_FORMAT_VIEW: u64 = 0x10;

/// The code of the guest surface textures made whose declared usage has `PixelFormatView`.
pub const PLANE_PIXEL_FORMAT_VIEW_CODE: &str = "rt-plane-textures-pixel-format-view";

/// The family of the codes of the views made of a guest surface texture, or of a view of one,
/// by the view's pixel format: `rt-plane-views-format-<raw>`.
pub const PLANE_VIEWS_CODE: &str = "rt-plane-views-format";

/// As [`PLANE_VIEWS_CODE`], of the views rt refused: `rt-plane-views-refused-format-<raw>`.
pub const PLANE_VIEWS_REFUSED_CODE: &str = "rt-plane-views-refused-format";

/// A view in pixel format `format` of `of` was made (`made`), or rt refused it: counted by the
/// format where `of` is a guest surface texture or a view of one ([`PLANE_VIEWS_CODE`],
/// [`PLANE_VIEWS_REFUSED_CODE`]).
pub fn plane_view(of: &Made, format: u32, made: bool) {
    if !of.of_a_plane() {
        return;
    }
    let family = if made {
        PLANE_VIEWS_CODE
    } else {
        PLANE_VIEWS_REFUSED_CODE
    };
    crate::answer::count_kind(format_code(family, format));
}

/// The code `<family>-<format>`, made once a pair and kept for the process (the counts keep
/// codes as `&'static str`).
fn format_code(family: &'static str, format: u32) -> &'static str {
    static CODES: Mutex<BTreeMap<(&'static str, u32), &'static str>> = Mutex::new(BTreeMap::new());
    let mut codes = CODES.lock();
    codes
        .entry((family, format))
        .or_insert_with(|| Box::leak(format!("{family}-{format}").into_boxed_str()))
}

/// Notes `made`'s plane `plane`, of declared `usage` and `bytes`, once.
fn on_plane(made: &Made, plane: Plane, usage: u64, bytes: u64) {
    let storage = takes_storage(usage);
    if made.on_plane.set(OnPlane { plane, storage }).is_err() {
        return;
    }
    let mut planes = PLANES.lock();
    let p = planes.entry(plane).or_default();
    let others: Vec<u32> = (p.formats.keys().copied())
        .filter(|&f| f != made.format)
        .collect();
    *p.formats.entry(made.format).or_insert(0) += 1;
    if storage {
        p.storage += 1;
    }
    drop(planes);
    let mut pairs = PAIRS.lock();
    for other in others {
        let pair = pairs.entry(pair_of(made.format, other)).or_default();
        pair.met += 1;
        pair.bytes_most = pair.bytes_most.max(bytes);
    }
}

/// `a` and `b`, the lesser first.
fn pair_of(a: u32, b: u32) -> (u32, u32) {
    (a.min(b), a.max(b))
}

/// The decoder noted a write through the guest surface texture `token` is: counted for each
/// other format live over its plane.
pub fn shape_written(token: &Token) {
    let made = &token.0;
    let Some(on) = made.on_plane.get() else {
        return;
    };
    let others: Vec<u32> = match PLANES.lock().get(&on.plane) {
        Some(p) => (p.formats.keys().copied())
            .filter(|&f| f != made.format)
            .collect(),
        None => return,
    };
    let mut pairs = PAIRS.lock();
    for other in others {
        pairs.entry(pair_of(made.format, other)).or_default().writes += 1;
    }
}

/// Counts a view in `view`'s format of a plain shape of `shape`'s format, which rt refuses.
pub fn plain_view_refused(shape: u64, view: u64) {
    *PLAIN_REFUSED.lock().entry((shape, view)).or_insert(0) += 1;
}

/// The plane of `texture`, a guest surface texture, as an attachment read's class names it: one
/// with a live texture that takes storage, or one with none.
fn plane_storage(texture: &Made) -> &'static str {
    let Some(on) = texture.on_plane.get() else {
        return "no-surface-plane";
    };
    match PLANES.lock().get(&on.plane) {
        Some(p) if p.storage != 0 => "plane-storage-live",
        _ => "plane-no-storage",
    }
}

// ----- Where a block was imported.

thread_local! {
    /// The decoder's message this thread carries out ([`message`]); 0 for none.
    static MESSAGE: Cell<u32> = const { Cell::new(0) };
    /// The context whose command buffer this thread walks ([`walk_context`]).
    static CONTEXT: Cell<Option<u32>> = const { Cell::new(None) };
}

/// The decoder carries out a message of type `r#type` on this thread.
pub fn message(r#type: u32) {
    MESSAGE.set(r#type);
}

/// The decoder walks a command buffer of `context` on this thread.
pub fn walk_context(context: Option<u32>) {
    CONTEXT.set(context);
}

/// Where a block was imported: the selector, the decoder's message carried out on the thread
/// (0: none), the thread's name without its number, and whether the block is a copy of the
/// pages ([`copy_made`]) rather than the pages themselves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Origin {
    sel: &'static str,
    message: u32,
    thread: &'static str,
    copy: bool,
}

/// An import block as [`imported`] noted it: where it was imported, and its range and storage
/// mode, for a lost device's fault (`super::fault`, [`import_range`]).
#[derive(Clone, Copy)]
struct Imported {
    origin: Origin,
    gpu_address: u64,
    len: u64,
    class: rt::MemoryClass,
}

/// The blocks imported, by rt's number, which rt never reuses, with where and their ranges.
static IMPORTED: Mutex<BTreeMap<rt::BlockId, Imported>> = Mutex::new(BTreeMap::new());

/// The imports by origin.
static IMPORTS: Mutex<BTreeMap<Origin, u64>> = Mutex::new(BTreeMap::new());

/// The blocks made as copies of an import's pages ([`copy_made`]) whose import is yet to be
/// noted ([`imported`]).
static COPIES: Mutex<BTreeSet<rt::BlockId>> = Mutex::new(BTreeSet::new());

/// Notes that block `block`, made now, is a copy of an import's pages (`super::buffer`'s
/// `copied`: pages whose GPU writes Metal drops, or pages rt took neither by their pointer nor
/// as a dma-buf), for its import's origin ([`imported`]).
pub fn copy_made(block: rt::BlockId) {
    COPIES.lock().insert(block);
}

/// Notes block `block`, imported now for `sel` (`super::buffer::import_pages`): a copy of the
/// pages where [`copy_made`] noted it, else the pages themselves.
pub fn imported(block: &rt::BlockInfo, sel: &'static str) {
    let origin = Origin {
        sel,
        message: MESSAGE.get(),
        thread: thread_class(),
        copy: COPIES.lock().remove(&block.id),
    };
    let imported = Imported {
        origin,
        gpu_address: block.gpu_address,
        len: block.len,
        class: block.class,
    };
    IMPORTED.lock().insert(block.id, imported);
    *IMPORTS.lock().entry(origin).or_insert(0) += 1;
}

/// An import block's range and storage mode, and whether it is a copy of its pages
/// ([`import_range`]).
pub(super) struct ImportRange {
    /// Its first byte's GPU address.
    pub(super) gpu_address: u64,
    /// Its length in bytes.
    pub(super) len: u64,
    /// Its storage mode.
    pub(super) class: rt::MemoryClass,
    /// Whether it is a copy of its pages ([`copy_made`]) rather than the pages themselves.
    pub(super) copy: bool,
}

/// Import block `block`'s range, where it is an import the device made ([`imported`]), for a lost
/// device's fault (`super::fault`). The record is kept for the process's life, as the origins'
/// are: it says nothing of whether the import lives, so the fault asks only of blocks it knows
/// live (a live buffer's, a live dma-buf import).
pub(super) fn import_range(block: rt::BlockId) -> Option<ImportRange> {
    let imported = *IMPORTED.lock().get(&block)?;
    Some(ImportRange {
        gpu_address: imported.gpu_address,
        len: imported.len,
        class: imported.class,
        copy: imported.origin.copy,
    })
}

/// This thread's name without the number that ends it (`-0x...` or `-N`), kept once.
fn thread_class() -> &'static str {
    static NAMES: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let current = std::thread::current();
    let name = current.name().unwrap_or("unnamed");
    let class = match name.rsplit_once('-') {
        Some((head, tail))
            if tail.starts_with("0x") || tail.chars().all(|c| c.is_ascii_digit()) =>
        {
            head
        }
        _ => name,
    };
    let mut names = NAMES.lock();
    if let Some(&kept) = names.get(class) {
        return kept;
    }
    let kept: &'static str = Box::leak(class.to_owned().into_boxed_str());
    names.insert(kept);
    kept
}

/// The context of the draw this thread encodes: the walk's, else the queue's in the thread's name
/// (`reims-vgpu.graphics-queue-0x...`, `-lane-0x...`), whose object id's bits from 40 are its
/// context.
fn context_now() -> Option<u32> {
    CONTEXT.get().or_else(|| {
        let current = std::thread::current();
        let (_, tail) = current.name()?.rsplit_once("-0x")?;
        let id = u64::from_str_radix(tail, 16).ok()?;
        u32::try_from(id >> 40).ok()
    })
}

// ----- Textures over an imported block.

/// What rt copies of a texture over an imported block (the module's docs).
struct Over {
    /// How many bytes it lies over.
    len: u64,
    /// Its kind's totals.
    totals: Arc<OverTotals>,
}

/// A kind of textures over an imported block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OverKind {
    format: u32,
    width: u32,
    height: u32,
    bytes_per_row: u64,
    usage: u64,
    storage: u64,
    origin: Origin,
}

/// A kind's totals (the module's docs).
#[derive(Default)]
struct OverTotals {
    made: AtomicU64,
    submissions: AtomicU64,
    bytes: AtomicU64,
    sampled: AtomicU64,
    rendered: AtomicU64,
    blitted: AtomicU64,
}

/// The kinds' totals.
static OVERS: Mutex<BTreeMap<OverKind, Arc<OverTotals>>> = Mutex::new(BTreeMap::new());

impl Over {
    /// What rt copies of a texture of `shape` over `buffer` from `offset`, rows `bytes_per_row`
    /// apart: `None` where the buffer's block was not imported. rt copies one whether or not it
    /// maps the block for the host: a buffer without contents (a dma-buf import) is counted too.
    fn of(buffer: &rt::Buffer, shape: &Shape, offset: u64, bytes_per_row: u64) -> Option<Over> {
        let origin = IMPORTED.lock().get(&buffer.block())?.origin;
        let kind = OverKind {
            format: shape.format,
            width: shape.width,
            height: shape.height,
            bytes_per_row,
            usage: shape.usage,
            storage: shape.storage,
            origin,
        };
        let totals = Arc::clone(OVERS.lock().entry(kind).or_default());
        totals.made.fetch_add(1, Relaxed);
        let rows = bytes_per_row.saturating_mul(u64::from(shape.height));
        let len = rows.min(buffer.length().saturating_sub(offset));
        Some(Over { len, totals })
    }

    /// A submission used it as `how` says: counts it under its kind.
    fn submitted(&self, how: How) {
        let t = &self.totals;
        t.submissions.fetch_add(1, Relaxed);
        t.bytes.fetch_add(self.len, Relaxed);
        for (used, count) in [
            (how.sampled, &t.sampled),
            (how.rendered, &t.rendered),
            (how.blitted || how.blitted_into, &t.blitted),
        ] {
            if used {
                count.fetch_add(1, Relaxed);
            }
        }
    }
}

/// How a command buffer used a texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    /// Bound to a shader's slot.
    Sampled,
    /// Rendered into, an attachment of a pass.
    Rendered,
    /// A blit's source.
    Blitted,
    /// A blit's destination.
    BlittedInto,
}

/// A submission's uses of one texture.
#[derive(Clone, Copy, Default)]
struct How {
    sampled: bool,
    rendered: bool,
    blitted: bool,
    blitted_into: bool,
}

/// A texture a command buffer used, and how.
type Used = (Arc<Made>, Use);

/// A command buffer's uses of textures over an imported block, until its commit
/// ([`Uses::submitted`]).
#[derive(Clone, Default)]
pub struct Uses(Arc<Mutex<Vec<Used>>>);

impl Uses {
    /// The command buffer uses `texture` as `how` says: kept where it lies over an imported
    /// block.
    pub fn add(&self, texture: &RtObj<RtTexture>, how: Use) {
        self.add_made(texture.made(), how);
    }

    /// [`Uses::add`] of the texture `made` is.
    pub fn add_made(&self, made: &Arc<Made>, how: Use) {
        if made.over.is_some() {
            self.0.lock().push((Arc::clone(made), how));
        }
    }

    /// The command buffer is committed: each texture it used over an imported block is counted
    /// once, with how it was used ([`Over::submitted`]).
    pub fn submitted(&self) {
        let uses = std::mem::take(&mut *self.0.lock());
        if uses.is_empty() {
            return;
        }
        let mut by_texture: HashMap<u64, (Arc<Made>, How)> = HashMap::new();
        for (made, used) in uses {
            let (_, how) = by_texture
                .entry(made.id)
                .or_insert_with(|| (Arc::clone(&made), How::default()));
            match used {
                Use::Sampled => how.sampled = true,
                Use::Rendered => how.rendered = true,
                Use::Blitted => how.blitted = true,
                Use::BlittedInto => how.blitted_into = true,
            }
        }
        let (mut plane_copies, mut plane_bytes) = (0, 0);
        for (made, how) in by_texture.into_values() {
            if let Some(over) = &made.over {
                over.submitted(how);
                if made.on_plane.get().is_some() {
                    plane_copies += 1;
                    plane_bytes += over.len;
                }
            }
        }
        crate::answer::count_kind_by(SURFACE_PLANE_COPIES_CODE, plane_copies);
        crate::answer::count_kind_by(SURFACE_PLANE_COPY_BYTES_CODE, plane_bytes);
    }
}

/// The code of the command buffers' uses of a guest surface texture over a buffer in an imported
/// block, which rt copies the bytes of into the texture's image of its own at each submission
/// that uses it: one for each texture a committed command buffer used ([`Uses::submitted`]).
pub const SURFACE_PLANE_COPIES_CODE: &str = "rt-surface-plane-copies";

/// The code of those uses' bytes: the texture's rows in the buffer, its rows apart times its
/// height, as far as the buffer goes.
pub const SURFACE_PLANE_COPY_BYTES_CODE: &str = "rt-surface-plane-copy-bytes";

// ----- Gaps.

/// The refusals of rt's gaps by kind.
static GAPS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// For the tests: the refusals of rt's gaps counted under `name` ([`refusal_name`]).
#[cfg(test)]
pub(crate) fn gaps_named(name: &str) -> u64 {
    GAPS.lock().get(name).copied().unwrap_or(0)
}

/// The refusals of Metal's that rt answers, its limits, and the translator's refusals, by variant.
static REFUSALS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// Counts a refusal of rt's `error`: a gap under its gap's kind, a kernel's past the stage bound
/// that no helper has compiled under `StageUnfilled`, a draw's by its draw kind; a
/// refusal of Metal's and a limit under their variant, `Refused-<variant>` and `Limit-<variant>`;
/// the translator's under the feature it refused, `Translation-<feature>`; any other error not.
pub fn refusal(error: &rt::Error) {
    let Some(name) = refusal_name(error) else {
        return;
    };
    let map = match error {
        rt::Error::Unsupported(_) | rt::Error::StageUnfilled(_) => &GAPS,
        _ => &REFUSALS,
    };
    *map.lock().entry(name).or_insert(0) += 1;
}

/// The kind [`refusal`] counts rt's `error` under: a gap's kind, a kernel's past the stage bound
/// that no helper has compiled `StageUnfilled`, a draw's `Draw-<kind>`, a linear plane pass's
/// `LinearPlanePass-<kind>`, an indirect command buffer's
/// `IndirectCommandBuffers-<kind>`, a refusal of Metal's `Refused-<variant>`, a limit
/// `Limit-<variant>`, and the translator's `Translation-<feature>`, the variant of the feature it
/// refused, or of its error where it refused no feature (`Translation-InvalidInput`); `None` for
/// any other error. Each is a variant's name alone, no name of the guest's.
pub fn refusal_name(error: &rt::Error) -> Option<String> {
    use metal2vulkan::error::Error as Translator;
    Some(match error {
        rt::Error::Unsupported(rt::Gap::Draw { kind }) => format!("Draw-{kind:?}"),
        rt::Error::Unsupported(rt::Gap::LinearPlanePass { kind }) => {
            format!("LinearPlanePass-{kind:?}")
        }
        rt::Error::Unsupported(rt::Gap::IndirectCommandBuffers { kind }) => {
            format!("IndirectCommandBuffers-{kind:?}")
        }
        rt::Error::Unsupported(gap) => variant(&format!("{gap:?}")).to_owned(),
        rt::Error::StageUnfilled(_) => "StageUnfilled".to_owned(),
        rt::Error::Refused(r) => format!("Refused-{}", variant(&format!("{r:?}"))),
        rt::Error::Limit(l) => format!("Limit-{}", variant(&format!("{l:?}"))),
        rt::Error::Translation {
            error: Translator::Unsupported(u),
            ..
        } => {
            format!("Translation-{}", variant(&format!("{:?}", u.feature)))
        }
        rt::Error::Translation { error: t, .. } => {
            format!("Translation-{}", variant(&format!("{t:?}")))
        }
        _ => return None,
    })
}

/// The variant `debug` names, the `{:?}` of an enum's value: up to its first character that is not
/// alphanumeric.
fn variant(debug: &str) -> &str {
    let end = debug.find(|c: char| !c.is_alphanumeric());
    &debug[..end.unwrap_or(debug.len())]
}

// ----- Attachment reads through a texture.

/// A pass's attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Point {
    /// The colour attachment of this index.
    Color(usize),
    /// The depth attachment.
    Depth,
    /// The stencil attachment.
    Stencil,
}

/// A target of a pass: its attachment, level and slice, and its texture.
pub struct Target {
    /// Its attachment.
    pub point: Point,
    /// The level it renders.
    pub level: u32,
    /// The slice it renders.
    pub slice: u32,
    /// Its texture's `MTLTextureType`.
    pub texture_type: u32,
    /// Its texture's sample count.
    pub samples: u32,
    /// Its texture's index, 0 for one with none: rt reads a fetched attachment through it.
    pub index: u32,
    /// Its texture's guest object id, 0 for none (`super::fault`'s `Named`).
    pub id: u64,
    /// Its texture.
    pub made: Arc<Made>,
}

/// The textures a render encoder bound, by stage (vertex, fragment) and slot.
#[derive(Default)]
pub struct Bound([Vec<Option<Arc<Made>>>; 2]);

impl Bound {
    /// Slot `slot` of stage `stage` (0 vertex, 1 fragment) holds `texture`.
    pub fn set(&mut self, stage: usize, slot: u32, texture: Option<&RtObj<RtTexture>>) {
        let (Some(slots), Ok(slot)) = (self.0.get_mut(stage), usize::try_from(slot)) else {
            return;
        };
        if slot >= 256 {
            return;
        }
        if slots.len() <= slot {
            slots.resize(slot + 1, None);
        }
        slots[slot] = texture.map(|t| Arc::clone(t.made()));
    }
}

/// A class of refused draws, the key of their totals.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ReadClass {
    context: Option<u32>,
    class: String,
}

/// The refused draws by class.
static READS: Mutex<BTreeMap<ReadClass, u64>> = Mutex::new(BTreeMap::new());

/// The pipeline states whose first refused draw wrote its line, and the lines written.
static FIRSTS: Mutex<(BTreeSet<u64>, usize)> = Mutex::new((BTreeSet::new(), 0));

/// Counts a draw rt refused as an attachment read through a texture, by its class (the module's
/// docs): the draw of pipeline state `pipeline`, whose stages declared `accesses` (vertex,
/// fragment) of their slots, with `bound` bound, in a pass of `targets`, its depth write state
/// `depth_write`.
pub fn attachment_read(
    pipeline: u64,
    accesses: [&[(u32, rt::BindingAccess)]; 2],
    bound: &Bound,
    targets: &[Target],
    depth_write: bool,
) {
    let context = context_now();
    let mut classes = Vec::new();
    let mut lines = Vec::new();
    for (stage, (slots, bound)) in accesses.iter().zip(&bound.0).enumerate() {
        let stage_name = ["vertex", "fragment"][stage];
        for &(slot, access) in *slots {
            let Some(Some(texture)) = usize::try_from(slot).ok().and_then(|s| bound.get(s)) else {
                continue;
            };
            let meets: Vec<&Target> = (targets.iter())
                .filter(|t| t.made.image == texture.image)
                .collect();
            let Some(&target) = meets.first() else {
                continue;
            };
            let relation = if target.made.id == texture.id {
                "same-texture"
            } else if texture.surface && target.made.surface {
                "twin-surface-views"
            } else if texture.call == Call::OverBuffer {
                "over-one-buffer"
            } else {
                "views-of-one-image"
            };
            let class_a = matches!(relation, "same-texture" | "twin-surface-views");
            let point = match target.point {
                Point::Color(i) => format!("color{i}"),
                Point::Depth => "depth".to_owned(),
                Point::Stencil => "stencil".to_owned(),
            };
            let depth = match target.point {
                Point::Color(_) => String::new(),
                _ => format!(" depth-write {depth_write}"),
            };
            let class = format!(
                "class {} {relation} {stage_name} {access:?} {} declared {:#x} format {} storage {} \
                 at {point}{} second {} image-usage {:#x} {}",
                if class_a { "A" } else { "other" },
                texture.call_name(),
                texture.declared_usage(),
                texture.format,
                texture.storage,
                depth,
                meets.len() > 1,
                texture.usage,
                plane_storage(texture),
            );
            lines.push(format!(
                "{class}; slot {slot}, texture level {} slice {}, attachment level {} slice {}, \
                 attachment {} declared {:#x}",
                texture.level,
                texture.slice,
                target.level,
                target.slice,
                target.made.call_name(),
                target.made.declared_usage(),
            ));
            classes.push(class);
        }
    }
    if classes.is_empty() {
        classes.push("class unmatched: no slot's texture lies in a target's image".to_owned());
        lines.push(format!(
            "{}; {} targets, {} vertex and {} fragment slots bound",
            classes[0],
            targets.len(),
            bound.0[0].iter().flatten().count(),
            bound.0[1].iter().flatten().count(),
        ));
    }
    let class = classes.join(" + ");
    let draw = Refused {
        totals: &READS,
        firsts: &FIRSTS,
        what: "attachment-read",
    };
    draw.count(context, pipeline, class, &lines.join(" + "));
}

/// The totals of a kind of refused draws, and its first lines.
struct Refused {
    /// Its refused draws by class.
    totals: &'static Mutex<BTreeMap<ReadClass, u64>>,
    /// The pipeline states whose first refused draw wrote its line, and the lines written.
    firsts: &'static Mutex<(BTreeSet<u64>, usize)>,
    /// The kind's name in its lines.
    what: &'static str,
}

impl Refused {
    /// Counts a refused draw of class `class` of a process of context `context`, and writes
    /// `line`, as the kind's first of a state, where it is the first of pipeline state `pipeline`
    /// and fewer than [`FIRST_LINES`] were written.
    fn count(&self, context: Option<u32>, pipeline: u64, class: String, line: &str) {
        *self
            .totals
            .lock()
            .entry(ReadClass { context, class })
            .or_insert(0) += 1;
        let mut firsts = self.firsts.lock();
        if firsts.1 < FIRST_LINES && firsts.0.insert(pipeline) {
            firsts.1 += 1;
            drop(firsts);
            let context = context.map_or("unknown".to_owned(), |c| c.to_string());
            say(&format!(
                "reims-vgpu: rt-diag {} first of a state: context {context}: {line}\n",
                self.what
            ));
        }
    }
}

/// Each process's totals of `totals`, by class and by whether the process presents (`presenters`).
fn by_process(
    totals: &Mutex<BTreeMap<ReadClass, u64>>,
    presenters: &BTreeSet<u32>,
) -> BTreeMap<(String, &'static str), u64> {
    let mut out = BTreeMap::new();
    for (key, n) in totals.lock().iter() {
        let process = match key.context {
            Some(c) if presenters.contains(&c) => "compositor",
            Some(_) => "app",
            None => "unknown",
        };
        *out.entry((key.class.clone(), process)).or_insert(0) += n;
    }
    out
}

// ----- Fetched colour attachments.

/// The draws rt refused as reading a colour attachment through its texture, by class.
static FETCHES: Mutex<BTreeMap<ReadClass, u64>> = Mutex::new(BTreeMap::new());

/// The pipeline states whose first such refused draw wrote its line, and the lines written.
static FETCH_FIRSTS: Mutex<(BTreeSet<u64>, usize)> = Mutex::new((BTreeSet::new(), 0));

/// Counts a draw of pipeline state `pipeline` in a pass of `targets` that rt refused as reading a
/// colour attachment through its texture, by its class (the module's docs): each colour target's.
pub fn fetched_attachment(pipeline: u64, targets: &[Target]) {
    let colors: Vec<String> = (targets.iter())
        .filter(|t| matches!(t.point, Point::Color(_)))
        .map(|t| fetched_class(t, targets))
        .collect();
    let class = match colors.is_empty() {
        true => "no colour target".to_owned(),
        false => colors.join(" + "),
    };
    let draw = Refused {
        totals: &FETCHES,
        firsts: &FETCH_FIRSTS,
        what: "fetched-attachment",
    };
    draw.count(context_now(), pipeline, class.clone(), &class);
}

/// The class of colour target `t` of a pass of `targets` (the module's docs).
fn fetched_class(t: &Target, targets: &[Target]) -> String {
    let m = &t.made;
    let point = match t.point {
        Point::Color(i) => format!("color{i}"),
        Point::Depth => "depth".to_owned(),
        Point::Stencil => "stencil".to_owned(),
    };
    let shared = (targets.iter()).filter(|o| o.made.image == m.image).count() > 1;
    format!(
        "{point} {} type {} samples {} level {} slice {} view-first {} {} index {} reads {} \
         writes {} declared {:#x} image-usage {:#x} shares-image {shared}",
        m.call_name(),
        t.texture_type,
        t.samples,
        t.level,
        t.slice,
        m.level,
        m.slice,
        if t.index == 0 { "none" } else { "yes" },
        m.usage == 0 || m.usage & TEXTURE_USAGE_SHADER_READ != 0,
        m.usage == 0 || m.usage & TEXTURE_USAGE_SHADER_WRITE != 0,
        m.declared_usage(),
        m.usage,
    )
}

/// `MTLTextureUsageShaderRead`.
const TEXTURE_USAGE_SHADER_READ: u64 = 0x1;

/// `MTLTextureUsageShaderWrite`.
const TEXTURE_USAGE_SHADER_WRITE: u64 = 0x2;

// ----- Sampler states by border.

/// The sampler states made, and of them those that clamp to their border colour in some direction,
/// by border colour: transparent black, opaque black, opaque white, and any other.
static SAMPLERS: Mutex<(u64, [u64; 4])> = Mutex::new((0, [0; 4]));

/// Counts a sampler state made of `d` (the module's docs).
pub fn sampler(d: &rt::SamplerDescriptor) {
    let border = [d.s_address_mode, d.t_address_mode, d.r_address_mode]
        .contains(&rt::SamplerAddressMode::ClampToBorderColor);
    let mut samplers = SAMPLERS.lock();
    samplers.0 += 1;
    if border {
        let at = usize::try_from(d.border_color.as_raw()).map_or(3, |raw| raw.min(3));
        samplers.1[at] += 1;
    }
}

// ----- Exported block makes.

/// rt's exported block makes (the module's docs): their count, those over 5 ms, and the longest's
/// nanoseconds and bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ExportMakes {
    count: u64,
    over_5_ms: u64,
    longest_nanos: u64,
    longest_bytes: u64,
}

impl ExportMakes {
    /// Adds a make of `bytes` that took `nanos`.
    #[cfg(any(unix, test))]
    fn add(&mut self, nanos: u64, bytes: u64) {
        self.count += 1;
        if nanos > 5_000_000 {
            self.over_5_ms += 1;
        }
        if nanos > self.longest_nanos {
            self.longest_nanos = nanos;
            self.longest_bytes = bytes;
        }
    }
}

/// The process's exported block makes.
static EXPORT_MAKES: Mutex<ExportMakes> = Mutex::new(ExportMakes {
    count: 0,
    over_5_ms: 0,
    longest_nanos: 0,
    longest_bytes: 0,
});

/// Counts one of rt's `Device::new_exported_block` calls, of `bytes`, which took `nanos` (the
/// module's docs).
#[cfg(unix)]
pub(super) fn exported_block_made(nanos: u64, bytes: u64) {
    EXPORT_MAKES.lock().add(nanos, bytes);
}

/// The exported block makes' line.
fn export_makes_line(e: &ExportMakes) -> String {
    format!(
        "reims-vgpu: rt-diag exported-block makes: {} calls, {} over 5 ms, longest {:.3} ms of {} \
         bytes (an allocation of its own)",
        e.count,
        e.over_5_ms,
        e.longest_nanos as f64 / 1e6,
        e.longest_bytes
    )
}

/// rt's timed take parts' line (the module's docs), from `m`.
fn timed_parts_line(m: &rt::Metrics) -> String {
    let parts = [
        ("takes", m.takes),
        ("take-lock-waits", m.take_lock_waits),
        ("take-makes", m.take_makes),
        ("take-maps", m.take_maps),
        ("maker-makes", m.maker_makes),
        ("zero-writes", m.zero_writes),
        ("long-take-zero-writes", m.long_take_zero_writes),
    ];
    let all: Vec<String> = parts
        .iter()
        .map(|(part, t)| {
            format!(
                "{part} {} calls {} over 5 ms longest {:.3} ms of {} bytes{}",
                t.count,
                t.over_5_ms,
                t.longest_nanos as f64 / 1e6,
                t.longest_bytes,
                if t.longest_made { " made" } else { "" }
            )
        })
        .collect();
    format!("reims-vgpu: rt-diag timed take parts: {}", all.join(", "))
}

/// The line of rt's pipeline makes in `m` (the module's docs, "rt's pipeline makes").
fn pipelines_line(m: &rt::Metrics) -> String {
    let ms = |nanos: u64| nanos as f64 / 1e6;
    format!(
        "reims-vgpu: rt-diag pipelines: {} created; first-use waits {} ({:.1} ms, longest {:.1} \
         ms), of a compute variant {} ({:.1} ms, longest {:.1} ms), of lines {} ({:.1} ms, \
         longest {:.1} ms); made ahead {} ({:.1} ms)",
        m.pipelines_created,
        m.first_use_waits,
        ms(m.first_use_wait_nanos),
        ms(m.first_use_wait_max_nanos),
        m.first_use_variant_waits,
        ms(m.first_use_variant_wait_nanos),
        ms(m.first_use_variant_wait_max_nanos),
        m.first_use_line_waits,
        ms(m.first_use_line_wait_nanos),
        ms(m.first_use_line_wait_max_nanos),
        m.made_ahead_variants,
        ms(m.made_ahead_variant_nanos),
    )
}

// ----- Own indices taken again.

/// The platform's own indices taken again after they were given back, textures' then sampler
/// states'.
static OWN_TAKEN_AGAIN: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// The platform's own indices given back, textures' then sampler states'.
static OWN_GIVEN_BACK: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Where a count of own indices of objects `what` names is kept: 0 for textures, 1 for sampler
/// states.
fn own_class(what: &'static str) -> usize {
    usize::from(what == "sampler")
}

/// Each of the platform's own indices of objects `what` names taken again after it was given
/// back (`super::texture::OwnIndices::take`).
pub fn own_index_taken_again(what: &'static str) {
    OWN_TAKEN_AGAIN[own_class(what)].fetch_add(1, Relaxed);
}

/// Each of the platform's own indices of objects `what` names given back
/// (`super::texture::OwnIndices::give_back`).
pub fn own_index_given_back(what: &'static str) {
    OWN_GIVEN_BACK[own_class(what)].fetch_add(1, Relaxed);
}

/// The own-index-reuse line (the module's docs, "Own indices taken again"): the textures' and
/// sampler states' indices taken again after they were given back, and those given back.
fn own_reused_line() -> String {
    let [again_textures, again_samplers] = OWN_TAKEN_AGAIN.each_ref().map(|n| n.load(Relaxed));
    let [back_textures, back_samplers] = OWN_GIVEN_BACK.each_ref().map(|n| n.load(Relaxed));
    format!(
        "reims-vgpu: rt-diag own-index-reuse: textures {again_textures}, samplers \
         {again_samplers} taken again after they were given back, once the command buffers \
         committed before were complete; given back: textures {back_textures}, samplers \
         {back_samplers}"
    )
}

// ----- Refused pipelines' AIR.

/// The variable that names the directory refused pipelines' AIR is written into (the module's
/// docs).
pub const REFUSED_AIR_VAR: &str = "REIMS_VGPU_REFUSED_AIR_DIR";

/// What the process kept of refused pipelines' AIR.
struct KeptAir {
    /// The libraries written, by their files' names.
    libraries: BTreeSet<String>,
    /// The refusals' lines written, by their creator and functions.
    refusals: BTreeSet<String>,
    /// The records written, by their text.
    records: BTreeSet<String>,
}

/// What the process kept of refused pipelines' AIR.
static KEPT_AIR: Mutex<KeptAir> = Mutex::new(KeptAir {
    libraries: BTreeSet::new(),
    refusals: BTreeSet::new(),
    records: BTreeSet::new(),
});

/// [`REFUSED_AIR_VAR`]'s directory, read once: `None` where it is unset or empty.
fn refused_air_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    let dir = DIR.get_or_init(|| {
        let dir = std::env::var_os(REFUSED_AIR_VAR).filter(|v| !v.is_empty())?;
        Some(PathBuf::from(dir).join(format!("air-{}", std::process::id())))
    });
    dir.as_deref()
}

/// The translator refused the compute state of `d` by the creator `sel` with `error`: where
/// [`REFUSED_AIR_VAR`] names a directory, keeps its kernel's AIR and the state's record there
/// (the module's docs).
pub fn keep_refused_compute(sel: &str, error: &rt::Error, d: &rt::ComputePipelineDescriptor) {
    keep_refused_air(sel, error, &[("kernel", Some(&d.kernel))], || {
        compute_fields(d)
    });
}

/// The translator refused the render state of `d` by the creator `sel` with `error`: where
/// [`REFUSED_AIR_VAR`] names a directory, keeps its functions' AIR and the state's record there
/// (the module's docs).
pub fn keep_refused_render(sel: &str, error: &rt::Error, d: &rt::RenderPipelineDescriptor) {
    let stages = [
        ("vertex", Some(&d.vertex)),
        ("fragment", d.fragment.as_ref()),
    ];
    keep_refused_air(sel, error, &stages, || render_fields(d));
}

/// [`keep_refused_compute`] and [`keep_refused_render`]: the state's stages by their role, and
/// its descriptor's fields' lines. A write that fails keeps nothing more of the refusal, and the
/// next refusal tries again.
fn keep_refused_air(
    sel: &str,
    error: &rt::Error,
    stages: &[(&str, Option<&rt::StageSource>)],
    fields: impl FnOnce() -> Vec<String>,
) {
    let rt::Error::Translation { .. } = error else {
        return;
    };
    let Some(dir) = refused_air_dir() else {
        return;
    };
    let (functions, record) = refused_record(stages, fields);
    let _ = keep_air_in(dir, sel, error, &functions, &record);
}

/// The functions of `stages`, by their role, whose AIR is kept, and the state's record's lines:
/// each stage's, then `fields`'.
fn refused_record<'a>(
    stages: &[(&str, Option<&'a rt::StageSource>)],
    fields: impl FnOnce() -> Vec<String>,
) -> (Vec<&'a metal2vulkan::library::Function>, Vec<String>) {
    let mut functions = Vec::new();
    let mut record = Vec::new();
    for &(role, stage) in stages {
        match stage {
            Some(rt::StageSource::Function(f)) => {
                functions.push(&f.function);
                stage_lines(&mut record, role, f);
            }
            Some(_) => record.push(format!("{role}: a compiled function")),
            None => record.push(format!("{role}: none")),
        }
    }
    record.extend(fields());
    (functions, record)
}

/// [`keep_refused_air`] into `dir`, of the refusal `why` of the functions `functions` by `sel`,
/// with the state's record `record`, its lines.
fn keep_air_in(
    dir: &Path,
    sel: &str,
    why: &dyn std::fmt::Display,
    functions: &[&metal2vulkan::library::Function],
    record: &[String],
) -> std::io::Result<()> {
    let mut kept = KEPT_AIR.lock();
    let mut named = Vec::new();
    for f in functions {
        let file = air_file(f.library());
        if !kept.libraries.contains(&file) {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join(&file), f.library().bytes())?;
            kept.libraries.insert(file);
        }
        named.push(function_name(f));
    }
    let key = format!("{sel}; {}", named.join(", "));
    if !kept.refusals.contains(&key) {
        append(
            &dir.join("refused.txt"),
            &format!("{key}; refused: {why}\n"),
        )?;
        kept.refusals.insert(key.clone());
    }
    let body: String = (record.iter()).map(|line| format!("  {line}\n")).collect();
    let body = format!("{key}; refused: {why}\n{body}");
    if !kept.records.contains(&body) {
        let n = kept.records.len() + 1;
        append(&dir.join("records.txt"), &format!("record {n}: {body}"))?;
        kept.records.insert(body);
    }
    Ok(())
}

/// Appends `text` to the file at `path`, making it and its directory where they are not.
fn append(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut file = (std::fs::OpenOptions::new().create(true).append(true)).open(path)?;
    std::io::Write::write_all(&mut file, text.as_bytes())
}

/// The file `library`'s AIR is kept in: its SHA-256's first 16 hex digits.
fn air_file(library: &rt::Library) -> String {
    let hash: String = (library.content_hash().iter().take(8))
        .map(|b| format!("{b:02x}"))
        .collect();
    format!("{hash}.metallib")
}

/// `f` as the lines name it: its library's file, its index and its name.
fn function_name(f: &metal2vulkan::library::Function) -> String {
    format!("{} #{} {}", air_file(f.library()), f.index(), f.info().name)
}

/// The record's lines of the stage `role` of `f`: its function, the constants it was made with,
/// and the functions it links with theirs.
fn stage_lines(out: &mut Vec<String>, role: &str, f: &rt::FunctionDescriptor) {
    let function = function_name(&f.function);
    out.push(format!("{role}: {function}; specialized {}", f.specialized));
    constant_lines(out, role, &f.function, &f.constants);
    let linked = &f.linked;
    let groups = (linked.groups.iter()).flat_map(|(name, fs)| fs.iter().map(move |l| (name, l)));
    let kinds = [
        ("functions", &linked.functions),
        ("private functions", &linked.private_functions),
        ("binary functions", &linked.binary_functions),
    ];
    let listed = kinds
        .into_iter()
        .flat_map(|(kind, fs)| fs.iter().map(move |l| (kind.to_owned(), l)));
    let grouped = groups.map(|(name, l)| (format!("group {name}"), l));
    for (kind, l) in listed.chain(grouped) {
        let prefix = format!("{role} linked {kind}");
        out.push(format!(
            "{prefix}: {}; specialized {}; specialized name {:?}; precompiled {}",
            function_name(&l.function),
            l.specialized,
            l.specialized_name,
            l.precompiled
        ));
        constant_lines(out, &prefix, &l.function, &l.constants);
    }
    let libraries = &f.libraries;
    if !libraries.is_empty() {
        out.push(format!(
            "{role} dynamic libraries: {} preloaded, {} loaded",
            libraries.preloaded.len(),
            libraries.loaded.len()
        ));
    }
}

/// The record's lines of the constants of `function` made with `values`, under `prefix`: each
/// value, by index then by name, and each constant the function declares with no value.
fn constant_lines(
    out: &mut Vec<String>,
    prefix: &str,
    function: &metal2vulkan::library::Function,
    values: &metal2vulkan::library::FunctionConstantValues,
) {
    let declared = function.info().constants;
    let hex = |bytes: &[u8]| -> String { bytes.iter().map(|b| format!("{b:02x}")).collect() };
    for (index, v) in &values.by_index {
        let name = (declared.iter()).find(|c| c.index == Some(*index));
        let name = name.map_or("?", |c| c.name.as_str());
        out.push(format!(
            "{prefix} constant #{index} {name}: {:?} {}",
            v.ty,
            hex(&v.bytes)
        ));
    }
    for (name, v) in &values.by_name {
        let index = (declared.iter())
            .find(|c| c.name == *name)
            .and_then(|c| c.index);
        let index = index.map_or_else(|| "?".to_owned(), |i| i.to_string());
        out.push(format!(
            "{prefix} constant #{index} {name} (by name): {:?} {}",
            v.ty,
            hex(&v.bytes)
        ));
    }
    for c in declared {
        let valued = c.index.is_some_and(|i| values.by_index.contains_key(&i))
            || values.by_name.contains_key(&c.name);
        if !valued {
            let index = c.index.map_or_else(|| "?".to_owned(), |i| i.to_string());
            let ty = c.ty.map_or_else(|| "?".to_owned(), |t| format!("{t:?}"));
            out.push(format!(
                "{prefix} constant #{index} {}: {ty} undefined",
                c.name
            ));
        }
    }
}

/// `descriptor <field>: <value>` of each named field of `d`.
macro_rules! fields {
    ($d:expr, $($field:ident),+ $(,)?) => {
        vec![$(format!(concat!("descriptor ", stringify!($field), ": {:?}"), $d.$field)),+]
    };
}

/// The record's lines of the compute descriptor `d`'s fields, its kernel's aside.
fn compute_fields(d: &rt::ComputePipelineDescriptor) -> Vec<String> {
    fields!(
        d,
        stage_input,
        max_total_threads_per_threadgroup,
        threadgroup_size_is_multiple_of_execution_width,
        support_indirect_command_buffers,
        max_call_stack_depth,
        required_threads_per_threadgroup,
        support_adding_binary_functions,
        buffers,
        needs_custom_border_color_samplers,
        open_cl_mode_enabled,
    )
}

/// The record's lines of the render descriptor `d`'s fields, its functions' aside.
fn render_fields(d: &rt::RenderPipelineDescriptor) -> Vec<String> {
    fields!(
        d,
        vertex_descriptor,
        color_attachments,
        depth_format,
        stencil_format,
        raster_sample_count,
        alpha_to_coverage,
        alpha_to_one,
        rasterization_enabled,
        input_primitive_topology,
        max_vertex_amplification_count,
        tessellation,
        support_indirect_command_buffers,
        max_vertex_call_stack_depth,
        max_fragment_call_stack_depth,
        support_adding_vertex_binary_functions,
        support_adding_fragment_binary_functions,
        vertex_buffers,
        fragment_buffers,
        texture_write_rounding_mode,
        private,
    )
}

// ----- Imageblock asks.

/// Imageblock asks by colour formats and sample count: how many, and rt's answer.
type Asks = BTreeMap<([u32; 8], u32), (u64, Option<u32>)>;

/// The render states' imageblock asks.
static IMAGEBLOCKS: Mutex<Asks> = Mutex::new(BTreeMap::new());

/// Counts an ask of `imageblockSampleLength` of a render state of colour formats `formats` and
/// sample count `samples`, which rt answered `answer`, `None` where it refused; the first of the
/// combination writes its line.
pub fn imageblock(formats: [u32; 8], samples: u32, answer: Option<u32>) {
    let mut asks = IMAGEBLOCKS.lock();
    let (n, _) = asks.entry((formats, samples)).or_insert((0, answer));
    *n += 1;
    if *n == 1 {
        drop(asks);
        say(&format!(
            "reims-vgpu: rt-diag imageblock first: formats {formats:?} samples {samples}: {}\n",
            answered(answer)
        ));
    }
}

/// rt's imageblock answer `answer` as a line gives it.
fn answered(answer: Option<u32>) -> String {
    answer.map_or_else(|| "refused".to_owned(), |n| format!("{n} bytes"))
}

// ----- Import paths.

/// How an import of the host's pages was made, or why it was refused
/// (`super::buffer::import_pages`).
#[derive(Clone, Copy, Debug)]
pub enum ImportPath {
    /// Made by the pages' pointer (rt's `import_host_memory`).
    Pointer,
    /// Made as a dma-buf of the files behind the pages (rt's `import_dma_buf`).
    DmaBuf,
    /// Refused as more runs of file pages, or more bytes, than one dma-buf takes: `pieces` runs,
    /// where one takes `most_pieces` runs and `most_bytes` bytes.
    TooLarge {
        /// The pages' runs of file pages.
        pieces: u64,
        /// One dma-buf's most runs.
        most_pieces: u64,
        /// One dma-buf's most bytes.
        most_bytes: u64,
    },
    /// Refused otherwise.
    Refused,
    /// Made by the pages' pointer once they were mapped from their files, as more runs of file
    /// pages, or more bytes, than one dma-buf takes (`super::buffer`'s `unpinnable`): `pieces`
    /// runs, where one takes `most_pieces` runs and `most_bytes` bytes. Counted as made by their
    /// pointer too.
    FromFilesTooLarge {
        /// The pages' runs of file pages.
        pieces: u64,
        /// One dma-buf's most runs.
        most_pieces: u64,
        /// One dma-buf's most bytes.
        most_bytes: u64,
    },
}

impl ImportPath {
    /// The path of an import made by the pages' pointer once they were mapped from their files,
    /// where one dma-buf of them was refused as `self` gives.
    pub fn from_files(self) -> ImportPath {
        match self {
            ImportPath::TooLarge {
                pieces,
                most_pieces,
                most_bytes,
            } => ImportPath::FromFilesTooLarge {
                pieces,
                most_pieces,
                most_bytes,
            },
            _ => ImportPath::Pointer,
        }
    }
}

/// The most bytes of each size bucket of the imports asked but the last, which takes the rest.
const SIZES: [u64; 5] = [64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20];

/// The imports one kind of make asked, by path.
#[derive(Default)]
struct PathTotals {
    /// The imports and their bytes, by path: pointer, dma-buf, too large, refused.
    counts: [u64; 4],
    bytes: [u64; 4],
    /// The most runs and the most bytes of an import refused as too large.
    most_pieces: u64,
    most_bytes: u64,
    /// Of the imports made by their pointer, those mapped from their files as too large for one
    /// dma-buf: how many, their bytes, and the most runs of one.
    from_files: (u64, u64, u64),
    /// One dma-buf's most runs and bytes, as the last import too large for one gave them.
    limits: (u64, u64),
    /// The imports asked, on every path, by size ([`SIZES`]).
    sizes: [u64; 6],
}

/// The imports asked, by the kind of the make that asked them.
static PATHS: Mutex<BTreeMap<&'static str, PathTotals>> = Mutex::new(BTreeMap::new());

/// Notes an import of `bytes` bytes of the host's pages for a make of kind `kind`, made or refused
/// as `path` (`super::buffer::import_pages`).
pub fn import_path(kind: &'static str, path: ImportPath, bytes: u64) {
    let i = match path {
        ImportPath::Pointer | ImportPath::FromFilesTooLarge { .. } => 0,
        ImportPath::DmaBuf => 1,
        ImportPath::TooLarge { .. } => 2,
        ImportPath::Refused => 3,
    };
    let mut paths = PATHS.lock();
    let t = paths.entry(kind).or_default();
    t.counts[i] += 1;
    t.bytes[i] += bytes;
    match path {
        ImportPath::TooLarge {
            pieces,
            most_pieces,
            most_bytes,
        } => {
            t.most_pieces = t.most_pieces.max(pieces);
            t.most_bytes = t.most_bytes.max(bytes);
            t.limits = (most_pieces, most_bytes);
        }
        ImportPath::FromFilesTooLarge {
            pieces,
            most_pieces,
            most_bytes,
        } => {
            let (n, b, most) = t.from_files;
            t.from_files = (n + 1, b + bytes, most.max(pieces));
            t.limits = (most_pieces, most_bytes);
        }
        ImportPath::Pointer | ImportPath::DmaBuf | ImportPath::Refused => {}
    }
    let size = SIZES.iter().position(|&most| bytes <= most);
    t.sizes[size.unwrap_or(SIZES.len())] += 1;
}

// ----- Frames.

/// The contexts that presented to the display.
static PRESENTERS: Mutex<BTreeSet<u32>> = Mutex::new(BTreeSet::new());

/// rt's metrics at the last present, and the totals between presents.
#[derive(Default)]
struct Frames {
    /// Over-copies, their bytes, those over an imported block, and submissions, at the last
    /// present.
    last: Option<[u64; 4]>,
    frames: u64,
    sums: [u64; 4],
    worst: [u64; 4],
    /// Frames of no over-copy, 1 to 9, 10 to 99, and 100 or more.
    spread: [u64; 4],
}

static FRAMES: Mutex<Frames> = Mutex::new(Frames {
    last: None,
    frames: 0,
    sums: [0; 4],
    worst: [0; 4],
    spread: [0; 4],
});

/// The display shows a frame of context `owner`'s: counts rt's over-copies since the last.
pub fn presented(owner: u32) {
    PRESENTERS.lock().insert(owner);
    let Some(m) = super::device::metrics() else {
        return;
    };
    let now = [
        m.over_copies,
        m.over_copy_bytes,
        m.over_copies_imported,
        m.submissions,
    ];
    let mut f = FRAMES.lock();
    if let Some(last) = f.last {
        f.frames += 1;
        for i in 0..4 {
            let d = now[i].saturating_sub(last[i]);
            f.sums[i] += d;
            f.worst[i] = f.worst[i].max(d);
        }
        let copies = now[0].saturating_sub(last[0]);
        let bucket = match copies {
            0 => 0,
            1..=9 => 1,
            10..=99 => 2,
            _ => 3,
        };
        f.spread[bucket] += 1;
    }
    f.last = Some(now);
}

// ----- Render passes over a video plane.

/// A view of a video plane image (a plane shape of a video cache's surface): the block of its
/// cache and its plane.
pub type VideoPlane = (u64, u64);

/// A render pass's attachments, as [`PlanePass::of`] classes them: each texture that is a view of
/// a video plane image, and of the others only what the shape needs.
#[derive(Debug, Default)]
pub struct PassAttachments {
    /// Each attachment's texture that is a view of a video plane image: the plane, and whether it
    /// is a resolve texture.
    planes: Vec<(VideoPlane, bool)>,
    /// The colour attachments' textures and resolve textures that are not.
    other_colours: u32,
    /// Whether the pass has a depth or a stencil attachment.
    depth_stencil: bool,
}

impl PassAttachments {
    /// A colour attachment's texture, or where `resolve` its resolve texture: a view of the video
    /// plane `plane` where `Some`.
    pub fn colour(&mut self, plane: Option<VideoPlane>, resolve: bool) {
        match plane {
            Some(plane) => self.planes.push((plane, resolve)),
            None => self.other_colours += 1,
        }
    }

    /// A depth or stencil attachment, its resolve texture a view of the video plane `resolve`
    /// where `Some`.
    pub fn depth_or_stencil(&mut self, resolve: Option<VideoPlane>) {
        self.depth_stencil = true;
        if let Some(plane) = resolve {
            self.planes.push((plane, true));
        }
    }
}

/// The shape of a render pass that takes a view of a video plane image as an attachment
/// ([`Self::of`]). A pass may have several of the last four; `alone` and `both` hold only where
/// none of those does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlanePass {
    /// One plane, a colour attachment, is its only attachment.
    pub alone: bool,
    /// The two planes of one picture, colour attachments, are its only attachments.
    pub both: bool,
    /// A depth or stencil attachment beside a plane.
    pub depth_stencil: bool,
    /// A colour attachment, or a colour attachment's resolve texture, that is not a plane.
    pub other_colour: bool,
    /// A plane as a resolve texture: the target of a multisample resolve.
    pub resolve: bool,
    /// Planes as colour attachments that are neither one plane nor the two planes of one
    /// picture: planes of more than one picture, or one plane twice.
    pub apart: bool,
}

/// The shapes as the plane-passes line names them, in [`PlanePass::shapes`]' order.
const PLANE_PASS_SHAPES: [&str; 6] = [
    "one plane alone",
    "both planes of one picture",
    "with a depth or stencil attachment",
    "with a colour attachment not a plane",
    "a plane as a resolve target",
    "planes of more than one picture",
];

impl PlanePass {
    /// The shape of a pass of the attachments `a`; `None` where none of them is a view of a video
    /// plane image.
    pub fn of(a: &PassAttachments) -> Option<PlanePass> {
        if a.planes.is_empty() {
            return None;
        }
        let resolve = a.planes.iter().any(|&(_, resolve)| resolve);
        let mut colours = (a.planes.iter())
            .filter(|&&(_, resolve)| !resolve)
            .map(|&(plane, _)| plane);
        let (first, second, more) = (colours.next(), colours.next(), colours.next().is_some());
        let one_picture = matches!(
            (first, second, more),
            (Some((b0, p0)), Some((b1, p1)), false) if b0 == b1 && p0 != p1
        );
        let only = !a.depth_stencil && a.other_colours == 0 && !resolve;
        Some(PlanePass {
            alone: only && first.is_some() && second.is_none(),
            both: only && one_picture,
            depth_stencil: a.depth_stencil,
            other_colour: a.other_colours != 0,
            resolve,
            apart: second.is_some() && !one_picture,
        })
    }

    /// Whether it has each shape, in [`PLANE_PASS_SHAPES`]' order.
    fn shapes(&self) -> [bool; 6] {
        [
            self.alone,
            self.both,
            self.depth_stencil,
            self.other_colour,
            self.resolve,
            self.apart,
        ]
    }
}

/// The render passes that took a view of a video plane image as an attachment, by the context
/// whose command buffer began them and their shape.
static PLANE_PASSES: Mutex<BTreeMap<(Option<u32>, PlanePass), u64>> = Mutex::new(BTreeMap::new());

/// A render pass of the attachments `a` begins in the command buffer this thread walks: counted
/// by its shape and the walk's context where it takes a view of a video plane image as an
/// attachment ([`PlanePass::of`]).
pub fn plane_pass(a: &PassAttachments) {
    if let Some(shape) = PlanePass::of(a) {
        *PLANE_PASSES
            .lock()
            .entry((context_now(), shape))
            .or_insert(0) += 1;
    }
}

/// The plane-passes line (the module's docs, "Render passes over a video plane"): the passes, and
/// each shape's, with those some pass had split between the compositor's process (a context of
/// `presenters`), an app's, and a context not known.
fn plane_passes_line(presenters: &BTreeSet<u32>) -> String {
    let mut all = [0u64; 3];
    let mut shapes = [[0u64; 3]; 6];
    for (&(context, shape), &n) in PLANE_PASSES.lock().iter() {
        let process = match context {
            Some(c) if presenters.contains(&c) => 0,
            Some(_) => 1,
            None => 2,
        };
        all[process] += n;
        for (i, has) in shape.shapes().into_iter().enumerate() {
            if has {
                shapes[i][process] += n;
            }
        }
    }
    // The passes of `n`, and where there are any, how many each process began.
    let counted = |n: [u64; 3]| {
        let total: u64 = n.iter().sum();
        let by: Vec<String> = (["compositor", "app", "unknown"].iter().zip(n))
            .filter(|&(_, n)| n != 0)
            .map(|(class, n)| format!("{class} {n}"))
            .collect();
        let by = if by.is_empty() {
            String::new()
        } else {
            format!(" ({})", by.join(", "))
        };
        (total, by)
    };
    let each: Vec<String> = (PLANE_PASS_SHAPES.iter().zip(shapes))
        .map(|(name, n)| {
            let (total, by) = counted(n);
            format!("{name} {total}{by}")
        })
        .collect();
    let (total, by) = counted(all);
    format!(
        "reims-vgpu: rt-diag plane-passes: {total} in all{by}; {}",
        each.join("; ")
    )
}

// ----- The lines.

/// Writes the lines (the module's docs) on stderr, in one write.
pub fn write() {
    let mut out = String::new();
    let gaps = GAPS.lock();
    if !gaps.is_empty() {
        let all: Vec<String> = gaps.iter().map(|(k, n)| format!("{k} {n}")).collect();
        let _ = writeln!(out, "reims-vgpu: rt-diag gaps: {}", all.join(", "));
    }
    drop(gaps);
    let refusals = REFUSALS.lock();
    if !refusals.is_empty() {
        let all: Vec<String> = refusals.iter().map(|(k, n)| format!("{k} {n}")).collect();
        let _ = writeln!(out, "reims-vgpu: rt-diag refusals: {}", all.join(", "));
    }
    drop(refusals);
    let presenters = PRESENTERS.lock().clone();
    for ((class, process), n) in &by_process(&READS, &presenters) {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag attachment-read: {process}: {class}: {n}"
        );
    }
    for ((class, process), n) in &by_process(&FETCHES, &presenters) {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag fetched-attachment: {process}: {class}: {n}"
        );
    }
    let (made, borders) = *SAMPLERS.lock();
    if made != 0 {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag samplers: {made} made; clamping to the border colour, by it: \
             transparent-black {}, opaque-black {}, opaque-white {}, other {}",
            borders[0], borders[1], borders[2], borders[3]
        );
    }
    let kept = KEPT_AIR.lock();
    if !kept.refusals.is_empty() {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag refused-air: {} refusals, {} libraries, {} records kept",
            kept.refusals.len(),
            kept.libraries.len(),
            kept.records.len()
        );
    }
    drop(kept);
    let mut surfaces: BTreeMap<(&str, u64, &str, bool), u64> = BTreeMap::new();
    for (kind, n) in SURFACES.lock().iter() {
        let process = if presenters.contains(&kind.context) {
            "compositor"
        } else {
            "app"
        };
        *surfaces
            .entry((process, kind.usage, kind.form, kind.made))
            .or_insert(0) += n;
    }
    for ((process, usage, form, made), n) in &surfaces {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag surface-texture: {process}: usage {usage:#x} ({}) {form}{}: {n}",
            if takes_storage(*usage) {
                "takes storage"
            } else {
                "plain"
            },
            if *made { "" } else { " refused" },
        );
    }
    for ((a, b), t) in PAIRS.lock().iter() {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag plane-formats: {a} and {b}: met {}, most bytes {}, writes while \
             both live {}",
            t.met, t.bytes_most, t.writes
        );
    }
    for ((shape, view), n) in PLAIN_REFUSED.lock().iter() {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag plain-view-refused: shape format {shape}, view format {view}: {n}"
        );
    }
    for ((formats, samples), (n, answer)) in IMAGEBLOCKS.lock().iter() {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag imageblock: formats {formats:?} samples {samples}: {n} asks, {}",
            answered(*answer)
        );
    }
    let f = FRAMES.lock();
    if f.frames != 0 {
        let mean = |i: usize| f.sums[i] as f64 / f.frames as f64;
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag frames: {} presented after the first; over-copies a frame mean \
             {:.2} worst {}; over an import mean {:.2} worst {}; bytes mean {:.0} worst {}; \
             submissions mean {:.2} worst {}; frames of 0, 1-9, 10-99, 100+ over-copies: {:?}",
            f.frames,
            mean(0),
            f.worst[0],
            mean(2),
            f.worst[2],
            mean(1),
            f.worst[1],
            mean(3),
            f.worst[3],
            f.spread,
        );
    }
    drop(f);
    if let Some(m) = super::device::metrics() {
        let mut sites = [
            ("texture", m.texture_allocations),
            ("texture-block", m.texture_block_allocations),
            ("ring-block", m.ring_block_allocations),
            ("staging-region", m.staging_region_allocations),
            ("staging-buffer", m.staging_buffer_allocations),
            ("zero-source", m.zero_source_allocations),
            ("shared-take", m.shared_take_allocations),
            ("private-take", m.private_take_allocations),
        ];
        sites.sort_by_key(|(_, s)| std::cmp::Reverse(s.longest_nanos));
        let all: Vec<String> = sites
            .iter()
            .map(|(site, s)| {
                format!(
                    "{site} {} calls {} bytes longest {:.3} ms of {} bytes, takes ahead {} of {} \
                     bytes",
                    s.count,
                    s.bytes,
                    s.longest_nanos as f64 / 1e6,
                    s.longest_bytes,
                    s.takes_ahead,
                    s.takes_ahead_bytes
                )
            })
            .collect();
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag allocation sites, longest call first: {}",
            all.join(", ")
        );
        let _ = writeln!(out, "{}", timed_parts_line(&m));
        let _ = writeln!(out, "{}", pipelines_line(&m));
        // The textures rt cleared at their make, which placed images skip
        // (`texture::PLACED_IMAGES_CODE`), and the Private memory it holds now.
        let (private, idle, blocks) = super::device::gauges().map_or((0, 0, 0), |g| {
            (g.private_allocation_bytes, g.private_idle_bytes, g.blocks)
        });
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag memory: {} textures cleared ({} MiB); Private {} MiB allocated, \
             {} MiB idle; {} blocks",
            m.texture_clears,
            m.texture_clear_bytes >> 20,
            private >> 20,
            idle >> 20,
            blocks
        );
        let _ = writeln!(out, "{}", own_reused_line());
    }
    let _ = writeln!(out, "{}", export_makes_line(&EXPORT_MAKES.lock()));
    for (kind, t) in OVERS.lock().iter() {
        let n = |a: &AtomicU64| a.load(Relaxed);
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag over-texture: format {} {}x{} rows {} usage {:#x} storage {} over \
             {}: made {}, submissions {} ({} bytes), sampled {}, rendered {}, blitted {}",
            kind.format,
            kind.width,
            kind.height,
            kind.bytes_per_row,
            kind.usage,
            kind.storage,
            origin_name(&kind.origin),
            n(&t.made),
            n(&t.submissions),
            n(&t.bytes),
            n(&t.sampled),
            n(&t.rendered),
            n(&t.blitted),
        );
    }
    for (origin, n) in IMPORTS.lock().iter() {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag imports: {}: {n}",
            origin_name(origin)
        );
    }
    for (kind, t) in PATHS.lock().iter() {
        let _ = writeln!(out, "{}", paths_line(kind, t));
    }
    let _ = writeln!(out, "{}", plane_passes_line(&presenters));
    if !presenters.is_empty() {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag presenters: contexts {presenters:?}"
        );
    }
    if !out.is_empty() {
        say(&out);
    }
}

/// The import-paths line of the imports a make of kind `kind` asked (the module's docs, "Imports
/// by path").
fn paths_line(kind: &str, t: &PathTotals) -> String {
    let (c, kib) = (&t.counts, t.bytes.map(|b| b >> 10));
    let (files, files_bytes, files_most) = t.from_files;
    format!(
        "reims-vgpu: rt-diag import-paths: {kind}: pointer {} ({} KiB; from files as too large {} \
         ({} KiB; most {} runs)), dma-buf {} ({} KiB), too large {} ({} KiB; most {} runs, {} \
         KiB; one dma-buf takes {} runs, {} KiB), refused {} ({} KiB); asked up to 64 KiB, 256 \
         KiB, 1 MiB, 4 MiB, 16 MiB, more: {:?}",
        c[0],
        kib[0],
        files,
        files_bytes >> 10,
        files_most,
        c[1],
        kib[1],
        c[2],
        kib[2],
        t.most_pieces,
        t.most_bytes >> 10,
        t.limits.0,
        t.limits.1 >> 10,
        c[3],
        kib[3],
        t.sizes,
    )
}

/// `origin` as a line names it: the selector, the message's type in hex, the thread, and
/// `copy` for a copy of the pages or `pages` for the pages themselves.
fn origin_name(origin: &Origin) -> String {
    let made = if origin.copy { "copy" } else { "pages" };
    format!(
        "{} message {:#x} thread {}, {made}",
        origin.sel, origin.message, origin.thread
    )
}

/// Writes `line` on stderr in one write, so it never interleaves with another thread's.
fn say(line: &str) {
    let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A thread's name loses the number that ends it, and keeps any other end.
    #[test]
    fn a_thread_s_class_drops_its_number() {
        let class = |name: &str| {
            let name = name.to_owned();
            std::thread::Builder::new()
                .name(name)
                .spawn(thread_class)
                .expect("a thread")
                .join()
                .expect("it returns")
        };
        assert_eq!(
            class("reims-vgpu.graphics-lane-0x10000000001"),
            "reims-vgpu.graphics-lane"
        );
        assert_eq!(
            class("reims-vgpu.graphics-compile-3"),
            "reims-vgpu.graphics-compile"
        );
        assert_eq!(
            class("reims-vgpu.graphics-surfaces"),
            "reims-vgpu.graphics-surfaces"
        );
    }

    /// A pass is classed by its attachments only where one is a view of a video plane image: one
    /// plane or one picture's two planes as its only attachments are `alone` and `both`; a depth
    /// or stencil attachment, a colour that is not a plane, a plane as a resolve target, and
    /// planes of two pictures or one plane twice each say so, and leave both of those unset.
    #[test]
    fn a_pass_over_a_video_plane_is_classed_by_its_shape() {
        let (y, uv, other) = ((0x10_0000, 0), (0x10_0000, 1), (0x20_0000, 0));
        let of = |colours: &[(Option<VideoPlane>, bool)], depth: Option<Option<VideoPlane>>| {
            let mut a = PassAttachments::default();
            for &(plane, resolve) in colours {
                a.colour(plane, resolve);
            }
            if let Some(resolve) = depth {
                a.depth_or_stencil(resolve);
            }
            PlanePass::of(&a)
        };
        let none = PlanePass::default();
        assert_eq!(of(&[], None), None);
        assert_eq!(of(&[(None, false), (None, true)], Some(None)), None);
        let alone = PlanePass {
            alone: true,
            ..none
        };
        assert_eq!(of(&[(Some(y), false)], None), Some(alone));
        assert_eq!(of(&[(Some(uv), false)], None), Some(alone));
        let both = PlanePass { both: true, ..none };
        assert_eq!(of(&[(Some(y), false), (Some(uv), false)], None), Some(both));
        assert_eq!(of(&[(Some(uv), false), (Some(y), false)], None), Some(both));
        let depth = PlanePass {
            depth_stencil: true,
            ..none
        };
        assert_eq!(of(&[(Some(y), false)], Some(None)), Some(depth));
        assert_eq!(
            of(&[(Some(y), false), (Some(uv), false)], Some(None)),
            Some(depth)
        );
        let colour = PlanePass {
            other_colour: true,
            ..none
        };
        assert_eq!(of(&[(None, false), (Some(y), false)], None), Some(colour));
        assert_eq!(
            of(&[(Some(y), false), (Some(uv), false), (None, false)], None),
            Some(colour)
        );
        assert_eq!(
            of(&[(None, false), (Some(y), true)], None),
            Some(PlanePass {
                other_colour: true,
                resolve: true,
                ..none
            })
        );
        assert_eq!(
            of(&[], Some(Some(uv))),
            Some(PlanePass {
                depth_stencil: true,
                resolve: true,
                ..none
            })
        );
        let apart = PlanePass {
            apart: true,
            ..none
        };
        assert_eq!(
            of(&[(Some(y), false), (Some(other), false)], None),
            Some(apart)
        );
        assert_eq!(of(&[(Some(y), false), (Some(y), false)], None), Some(apart));
        assert_eq!(
            of(
                &[(Some(y), false), (Some(uv), false), (Some(other), false)],
                None
            ),
            Some(apart)
        );
    }

    /// The plane-passes line gives every shape, and beside each some pass had, its passes by the
    /// process that began them: the compositor's (a context that presented), an app's, or one
    /// whose context the walk did not name.
    #[test]
    fn the_plane_passes_line_splits_each_shape_by_process() {
        let mut one = PassAttachments::default();
        one.colour(Some((0x30_0000, 0)), false);
        let mut beside = PassAttachments::default();
        beside.colour(Some((0x30_0000, 1)), false);
        beside.depth_or_stencil(None);
        walk_context(Some(7));
        plane_pass(&one);
        plane_pass(&one);
        plane_pass(&PassAttachments::default());
        walk_context(Some(8));
        plane_pass(&beside);
        walk_context(None);
        plane_pass(&one);
        assert_eq!(
            plane_passes_line(&BTreeSet::from([8])),
            "reims-vgpu: rt-diag plane-passes: 4 in all (compositor 1, app 2, unknown 1); one \
             plane alone 3 (app 2, unknown 1); both planes of one picture 0; with a depth or \
             stencil attachment 1 (compositor 1); with a colour attachment not a plane 0; a plane \
             as a resolve target 0; planes of more than one picture 0"
        );
    }

    /// A guest surface texture is one of its plane's live textures, and of those that take storage
    /// where its usage is 0, `ShaderWrite` or `ShaderAtomic`, until it is let go; a texture of
    /// another format made, or written, while it is live counts the pair.
    #[test]
    fn a_plane_counts_its_live_textures_and_formats() {
        let shape = |format, usage| Shape {
            format,
            width: 4,
            height: 4,
            storage: 2,
            usage,
        };
        let plane = (0x5a5a_0000, 1);
        let plain = Made::new(Call::View, &shape(80, 0x5));
        on_plane(&plain, plane, 0x5, 64);
        assert_eq!(plane_storage(&plain), "plane-no-storage");
        for usage in [0, 0x2, 0x20] {
            let storage = Made::new(Call::View, &shape(80, usage));
            on_plane(&storage, plane, usage, 64);
            assert_eq!(plane_storage(&plain), "plane-storage-live");
            drop(storage);
            assert_eq!(plane_storage(&plain), "plane-no-storage");
        }
        let other = Made::new(Call::View, &shape(53, 0));
        on_plane(&other, plane, 0, 128);
        shape_written(&Token(Arc::clone(&other)));
        drop(other);
        shape_written(&Token(Arc::clone(&plain)));
        let pairs = PAIRS.lock();
        let pair = pairs.get(&(53, 80)).expect("the pair is counted");
        assert_eq!((pair.met, pair.bytes_most, pair.writes), (1, 128, 1));
        drop(pairs);
        let unplaned = Made::new(Call::Plain, &shape(80, 0x5));
        assert_eq!(plane_storage(&unplaned), "no-surface-plane");
    }

    /// A view of a guest surface texture, or of a view of one, is counted by its format, made or
    /// refused; a view of any other texture is not. A surface texture made with `PixelFormatView`
    /// usage is counted; one without, or refused, is not.
    #[test]
    fn plane_views_count_by_format() {
        let counted = |code: &str| {
            (crate::answer::degradation_counts().into_iter())
                .find(|(c, _)| c == code)
                .map_or(0, |(_, n)| n)
        };
        let shape = Shape {
            format: 80,
            width: 4,
            height: 4,
            storage: 2,
            usage: 0x11,
        };
        let plane = Made::new(Call::View, &shape);
        on_plane(&plane, (0x5b5b_0000, 0), 0x11, 64);
        let view = Made::view(&plane, 81, (0, 0));
        let plain = Made::new(Call::Plain, &shape);
        let (made, refused) = (
            "rt-plane-views-format-81",
            "rt-plane-views-refused-format-71",
        );
        let before = (counted(made), counted(refused));
        plane_view(&plane, 81, true);
        plane_view(&view, 71, false);
        plane_view(&plain, 81, true);
        plane_view(&Made::view(&plain, 71, (0, 0)), 71, false);
        assert_eq!(
            (counted(made), counted(refused)),
            (before.0 + 1, before.1 + 1)
        );
        assert_eq!(format_code(PLANE_VIEWS_CODE, 81), made);

        let before = counted(PLANE_PIXEL_FORMAT_VIEW_CODE);
        surface_texture::<()>(
            None,
            7,
            (0x5c5c_0000, 0),
            (0x11, 64),
            "over-buffer: no cache",
        );
        assert_eq!(counted(PLANE_PIXEL_FORMAT_VIEW_CODE), before);
    }

    /// Two textures one call names say what they share, by their makes: one texture, one image (a
    /// texture and a view of it, two views of one texture), one buffer, one surface plane, or no
    /// image; and each says how it was made.
    #[test]
    fn two_textures_say_what_they_share() {
        let shape = Shape {
            format: 80,
            width: 8,
            height: 8,
            storage: 2,
            usage: 0x5,
        };
        let texture = Made::new(Call::Plain, &shape);
        let view = Made::view(&texture, 81, (1, 0));
        let other_view = Made::view(&texture, 80, (0, 0));
        let apart = Made::new(Call::Plain, &shape);
        assert_eq!(texture.shared_with(&texture), "one texture");
        assert_eq!(texture.shared_with(&view), "one image");
        assert_eq!(view.shared_with(&other_view), "one image");
        assert_eq!(texture.shared_with(&apart), "no image");
        let over = || {
            let id = NEXT.fetch_add(1, Relaxed);
            let image = 0x4000 | (1 << 63);
            Made::of(
                id,
                image,
                Call::OverBuffer,
                false,
                &shape,
                (0, 0),
                (None, None),
            )
        };
        assert_eq!(over().shared_with(&over()), "one buffer");
        let (s, t) = (Made::new(Call::View, &shape), Made::new(Call::View, &shape));
        on_plane(&s, (0x5d5d_0000, 0), 0x5, 64);
        on_plane(&t, (0x5d5d_0000, 0), 0x5, 64);
        assert_eq!(s.shared_with(&t), "one surface plane");
        assert_eq!(s.shared_with(&apart), "no image");
        let drawable = Made::new(Call::Drawable, &shape);
        let made_as = [&texture, &view, &over(), &drawable].map(|m| m.made_as());
        assert_eq!(made_as, ["ordinary", "view", "over-buffer", "drawable"]);
    }

    /// An import made by its pointer once its pages were mapped from their files, as too large for
    /// one dma-buf, counts as made by its pointer, once on every count, and its bytes and runs as
    /// such, which its line shows; an import refused as too large counts apart.
    #[test]
    fn imports_from_files_as_too_large_count_by_pointer() {
        let kind = "test-from-files";
        let too_large = |pieces| ImportPath::TooLarge {
            pieces,
            most_pieces: 1024,
            most_bytes: 64 << 20,
        };
        import_path(kind, ImportPath::Pointer, 0x1000);
        import_path(kind, too_large(2025).from_files(), 8100 << 10);
        import_path(kind, too_large(1500).from_files(), 6000 << 10);
        import_path(kind, ImportPath::Refused.from_files(), 0x2000);
        import_path(kind, too_large(3000), 12 << 20);
        let paths = PATHS.lock();
        let t = paths.get(kind).expect("the kind's totals");
        assert_eq!(t.counts, [4, 0, 1, 0]);
        assert_eq!(t.bytes[0], 0x3000 + (14100 << 10));
        assert_eq!(t.from_files, (2, 14100 << 10, 2025));
        assert_eq!((t.most_pieces, t.most_bytes), (3000, 12 << 20));
        assert_eq!(t.limits, (1024, 64 << 20));
        assert_eq!(t.sizes, [2, 0, 0, 0, 3, 0]);
        let line = paths_line(kind, t);
        let shown = ": pointer 4 (14112 KiB; from files as too large 2 (14100 KiB; most 2025 \
                     runs)), dma-buf 0 (0 KiB), too large 1 (12288 KiB; most 3000 runs, 12288 \
                     KiB; one dma-buf takes 1024 runs, 65536 KiB), refused 0 (0 KiB);";
        assert!(line.contains(shown), "{line}");
    }

    /// A gap is counted by its kind, a draw's by its draw kind, a linear plane pass's by its shape
    /// and an indirect command buffer's by its kind, a refusal of Metal's and a limit by their
    /// variant, and the translator's by the feature it refused, else its error's variant, with no
    /// name of the guest's; another error is not.
    #[test]
    fn refusals_count_by_kind() {
        use metal2vulkan::error::{
            DiagName, Error as Translator, Feature, LowerFamily, Unsupported,
        };
        refusal(&rt::Error::Unsupported(rt::Gap::Draw {
            kind: rt::DrawKind::AttachmentRead,
        }));
        refusal(&rt::Error::Unsupported(rt::Gap::CubicSampler));
        refusal(&rt::Error::Unsupported(rt::Gap::LinearPlanePass {
            kind: rt::LinearPlanePassKind::DepthStencil,
        }));
        refusal(&rt::Error::Unsupported(rt::Gap::IndirectCommandBuffers {
            kind: rt::IndirectCommandsKind::Tokens,
        }));
        refusal(&rt::Error::Refused(rt::Refusal::BufferTooLong {
            length: 8,
            max: 4,
        }));
        refusal(&rt::Error::Limit(rt::Limit::BlockLen { len: 8, max: 4 }));
        let translator = |u: Unsupported| rt::Error::Translation {
            error: Translator::Unsupported(u),
            functions: Box::default(),
        };
        let lowering = Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture));
        refusal(&translator(lowering.naming(DiagName::new("a_function"))));
        refusal(&translator(Unsupported::not_implemented(
            Feature::Imageblocks,
        )));
        refusal(&rt::Error::Translation {
            error: Translator::Cancelled,
            functions: Box::default(),
        });
        refusal(&rt::Error::Quiesced);
        let gaps = GAPS.lock();
        assert!(gaps.get("Draw-AttachmentRead").is_some_and(|&n| n >= 1));
        assert!(gaps.get("CubicSampler").is_some_and(|&n| n >= 1));
        assert!(
            gaps.get("LinearPlanePass-DepthStencil")
                .is_some_and(|&n| n >= 1)
        );
        assert!(
            gaps.get("IndirectCommandBuffers-Tokens")
                .is_some_and(|&n| n >= 1)
        );
        drop(gaps);
        let refusals = REFUSALS.lock();
        assert!(
            refusals
                .get("Refused-BufferTooLong")
                .is_some_and(|&n| n >= 1)
        );
        assert!(refusals.get("Limit-BlockLen").is_some_and(|&n| n >= 1));
        for name in [
            "Translation-Lowering",
            "Translation-Imageblocks",
            "Translation-Cancelled",
        ] {
            assert!(refusals.get(name).is_some_and(|&n| n >= 1), "{name}");
        }
        drop(refusals);
        for map in [&GAPS, &REFUSALS] {
            let keys = map.lock();
            assert!(!keys.keys().any(|k| k.contains("Quiesced")));
            assert!(!keys.keys().any(|k| k.contains("a_function")));
        }
    }

    /// A draw rt refused as reading a colour attachment through its texture is counted by each
    /// colour target's class: how its texture was made, its type, samples, level and slice, whether
    /// it has an index, whether its image's usage lets shaders read and write it (Unknown as both),
    /// and whether another target renders its image. The depth target is not in it.
    #[test]
    fn fetched_attachments_count_by_each_colour_target() {
        let shape = |usage| Shape {
            format: 80,
            width: 4,
            height: 4,
            storage: 2,
            usage,
        };
        let target = |point, made: &Arc<Made>, index| Target {
            point,
            level: 0,
            slice: 0,
            texture_type: 2,
            samples: 1,
            index,
            id: 0,
            made: Arc::clone(made),
        };
        let render_only = Made::new(Call::Plain, &shape(0x4));
        let unknown = Made::new(Call::Plain, &shape(0));
        let view = Made::view(&render_only, 80, (0, 0));
        let depth = Made::new(Call::Plain, &shape(0x4));
        let targets = [
            target(Point::Color(0), &render_only, 0),
            target(Point::Color(1), &unknown, 7),
            target(Point::Color(2), &view, 3),
            target(Point::Depth, &depth, 0),
        ];
        fetched_attachment(0x5eed, &targets);
        fetched_attachment(0x5eed, &targets);
        let common = "type 2 samples 1 level 0 slice 0 view-first 0 0";
        let class = [
            format!(
                "color0 ordinary {common} index none reads false writes false declared 0x4 \
                 image-usage 0x4 shares-image true"
            ),
            format!(
                "color1 ordinary {common} index yes reads true writes true declared 0x0 \
                 image-usage 0x0 shares-image false"
            ),
            format!(
                "color2 view {common} index yes reads false writes false declared 0x4 \
                 image-usage 0x4 shares-image true"
            ),
        ]
        .join(" + ");
        let fetches = FETCHES.lock();
        let counted = (fetches.iter()).find(|(k, _)| k.class == class);
        assert_eq!(counted.map(|(_, &n)| n), Some(2), "{:?}", fetches.keys());
    }

    /// Each sampler state made is counted, and one that clamps to its border colour in some
    /// direction by its border colour.
    #[test]
    fn samplers_count_by_their_border() {
        let (made, borders) = *SAMPLERS.lock();
        let mut d = rt::SamplerDescriptor::new();
        sampler(&d);
        d.t_address_mode = rt::SamplerAddressMode::ClampToBorderColor;
        d.border_color = rt::SamplerBorderColor::OpaqueBlack;
        sampler(&d);
        d.t_address_mode = rt::SamplerAddressMode::Repeat;
        d.r_address_mode = rt::SamplerAddressMode::ClampToBorderColor;
        d.border_color = rt::SamplerBorderColor::OpaqueWhite;
        sampler(&d);
        let (now, by) = *SAMPLERS.lock();
        assert_eq!(now - made, 3);
        let added: Vec<u64> = by.iter().zip(borders).map(|(n, was)| n - was).collect();
        assert_eq!(added, [0, 1, 1, 0]);
    }

    /// The exported block makes keep the longest call's bytes and count those over 5 ms; the
    /// line says each.
    #[test]
    fn exported_block_makes_keep_the_longest_calls_bytes() {
        let mut e = ExportMakes::default();
        e.add(2_000_000, 1 << 20);
        e.add(30_400_000, 12 << 20);
        e.add(6_000_000, 4 << 20);
        assert_eq!(
            e,
            ExportMakes {
                count: 3,
                over_5_ms: 2,
                longest_nanos: 30_400_000,
                longest_bytes: 12 << 20
            }
        );
        assert_eq!(
            export_makes_line(&e),
            "reims-vgpu: rt-diag exported-block makes: 3 calls, 2 over 5 ms, longest 30.400 ms \
             of 12582912 bytes (an allocation of its own)"
        );
    }

    /// rt's timed take parts' line names each part with its longest call's bytes, and `made`
    /// where that call made an allocation of its own.
    #[test]
    fn the_timed_take_parts_line_names_each_longest_call() {
        let mut m = rt::Metrics::default();
        m.takes.count = 9;
        m.takes.over_5_ms = 1;
        m.takes.longest_nanos = 7_250_000;
        m.takes.longest_bytes = 1 << 20;
        m.takes.longest_made = true;
        m.zero_writes.count = 2;
        m.zero_writes.longest_nanos = 1_000;
        m.zero_writes.longest_bytes = 4096;
        assert_eq!(
            timed_parts_line(&m),
            "reims-vgpu: rt-diag timed take parts: takes 9 calls 1 over 5 ms longest 7.250 ms \
             of 1048576 bytes made, take-lock-waits 0 calls 0 over 5 ms longest 0.000 ms of 0 \
             bytes, take-makes 0 calls 0 over 5 ms longest 0.000 ms of 0 bytes, take-maps 0 calls \
             0 over 5 ms longest 0.000 ms of 0 bytes, maker-makes 0 calls 0 over 5 ms longest \
             0.000 ms of 0 bytes, zero-writes 2 calls 0 over 5 ms longest 0.001 ms of 4096 bytes, \
             long-take-zero-writes 0 calls 0 over 5 ms longest 0.000 ms of 0 bytes"
        );
    }

    /// rt's pipeline makes' line gives its first-use waits by kind and its makes ahead, each
    /// with its milliseconds.
    #[test]
    fn the_pipelines_line_gives_first_use_waits_and_makes_ahead() {
        let mut m = rt::Metrics::default();
        m.pipelines_created = 40;
        m.first_use_waits = 5;
        m.first_use_wait_nanos = 250_000_000;
        m.first_use_wait_max_nanos = 120_500_000;
        m.first_use_variant_waits = 4;
        m.first_use_variant_wait_nanos = 240_000_000;
        m.first_use_variant_wait_max_nanos = 120_500_000;
        m.first_use_line_waits = 1;
        m.first_use_line_wait_nanos = 10_000_000;
        m.first_use_line_wait_max_nanos = 10_000_000;
        m.made_ahead_variants = 13;
        m.made_ahead_variant_nanos = 1_052_340_000;
        assert_eq!(
            pipelines_line(&m),
            "reims-vgpu: rt-diag pipelines: 40 created; first-use waits 5 (250.0 ms, longest \
             120.5 ms), of a compute variant 4 (240.0 ms, longest 120.5 ms), of lines 1 (10.0 ms, \
             longest 10.0 ms); made ahead 13 (1052.3 ms)"
        );
    }

    /// A refused pipeline's functions' library is written once into the directory, as its hash
    /// names it, and a line of the creator, its functions and the refusal once.
    #[test]
    fn a_refused_pipelines_air_is_kept_once() {
        let library = rt::Library::from_metallib(crate::gpu::shaders::METALLIB).expect("shaders");
        let f = library.function_at(0).expect("a function");
        let dir = std::env::temp_dir().join(format!("rvg-refused-air-{}", std::process::id()));
        for _ in 0..2 {
            keep_air_in(&dir, "sel:", &"a refusal", &[&f, &f], &[]).expect("written");
        }
        let hash: String = (library.content_hash()[..8].iter())
            .map(|b| format!("{b:02x}"))
            .collect();
        let kept = std::fs::read(dir.join(format!("{hash}.metallib"))).expect("the library");
        assert_eq!(kept, library.bytes());
        let lines = std::fs::read_to_string(dir.join("refused.txt")).expect("the lines");
        let name = f.info().name;
        let function = format!("{hash}.metallib #0 {name}");
        assert_eq!(
            lines,
            format!("sel:; {function}, {function}; refused: a refusal\n")
        );
        let records = std::fs::read_to_string(dir.join("records.txt")).expect("the records");
        let header = format!(": sel:; {function}, {function}; refused: a refusal\n");
        assert_eq!(records.lines().count(), 1, "{records}");
        assert!(
            records.starts_with("record ") && records.ends_with(&header),
            "{records}"
        );
        std::fs::remove_dir_all(&dir).expect("removed");
    }

    /// A refused compute state's record holds the function-constant values its kernel was made
    /// with, each by its index, its declared name and its type, and its descriptor's fields; a
    /// state of the kernel made with no value, its constant undefined, has a record of its own.
    #[test]
    fn a_refused_states_record_holds_its_constants_and_fields() {
        use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
        const OPTIONAL: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/optional_constant.metallib"));
        let library = rt::Library::from_metallib(OPTIONAL).expect("the kernel's library");
        let f = library.function("optional_constant").expect("the kernel");
        let int = metal2vulkan::metal::DataType::try_from(29u32).expect("MTLDataTypeInt");
        let int_name = format!("{int:?}");
        let five = FunctionConstantValue::new(int, 5i32.to_le_bytes().to_vec());
        let mut stage = rt::FunctionDescriptor::new(f.clone()).with_specialized(true);
        stage.constants = FunctionConstantValues::defaults().with_index(0, five);
        let mut d = rt::ComputePipelineDescriptor::new(rt::StageSource::Function(stage));
        d.threadgroup_size_is_multiple_of_execution_width = true;
        d.max_call_stack_depth = 3;
        let mut undefined = d.clone();
        let rt::StageSource::Function(stage) = &mut undefined.kernel else {
            unreachable!("made of a function")
        };
        stage.constants = FunctionConstantValues::defaults();

        let dir = std::env::temp_dir().join(format!("rvg-refused-record-{}", std::process::id()));
        for d in [&d, &d, &undefined] {
            let stages = [("kernel", Some(&d.kernel))];
            let (functions, record) = refused_record(&stages, || compute_fields(d));
            keep_air_in(&dir, "sel:", &"a refusal", &functions, &record).expect("written");
        }
        let records = std::fs::read_to_string(dir.join("records.txt")).expect("the records");
        std::fs::remove_dir_all(&dir).expect("removed");
        let function = function_name(&f);
        let header = format!(": sel:; {function}; refused: a refusal");
        let starts: Vec<&str> = records
            .lines()
            .filter(|l| l.starts_with("record "))
            .collect();
        assert_eq!(starts.len(), 2, "{records}");
        assert!(starts.iter().all(|l| l.ends_with(&header)), "{records}");
        let lines: Vec<&str> = records.lines().map(str::trim).collect();
        let has = |line: &str| assert!(lines.contains(&line), "no {line:?} in {records}");
        has(&format!("kernel: {function}; specialized true"));
        has(&format!(
            "kernel constant #0 optional_value: {int_name} 05000000"
        ));
        has(&format!(
            "kernel constant #0 optional_value: {int_name} undefined"
        ));
        has("descriptor threadgroup_size_is_multiple_of_execution_width: true");
        has("descriptor max_call_stack_depth: 3");
        has("descriptor max_total_threads_per_threadgroup: None");
        has(&format!("descriptor buffers: {:?}", d.buffers));
    }
}
