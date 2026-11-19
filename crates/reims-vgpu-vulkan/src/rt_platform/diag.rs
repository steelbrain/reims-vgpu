//! The rt platform's boot diagnostics, which a boot that checks it reads: lines
//! `reims-vgpu: rt-diag ...` on stderr, written with the process's counts ([`write()`], which
//! `crate::answer::write_counts` runs). They hold:
//!
//! - **Gaps by kind.** Each refusal of rt's `Gap`, by the gap's kind, a draw's by its draw kind
//!   ([`gap`]), which the counts line holds as one `rt-refused-gap`.
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
//! - **Textures over an imported block**, which rt keeps in an image of their own and copies the
//!   bytes of into it at each submission that uses them (rt's `over` module), by kind: format,
//!   size, rows, usage, storage mode, and where their block was imported (the selector, the
//!   decoder's message the import was made under and the thread's name without its number,
//!   [`imported`]). For each kind, the textures made, and their uses in submissions
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
//!
//! Totals, numbers, formats, sizes and usages only: no name, label or byte of the guest's leaves
//! the process. A diagnostic, which the contract leaves free: nothing the guest observes changes,
//! and a texture's bytes are only read.
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

use metal2vulkan_rt as rt;

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
            None,
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
            None,
        );
        if of.of_a_plane() {
            made.of_plane.store(true, Relaxed);
        }
        made
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
        Made::of(id, image, Call::OverBuffer, false, shape, (0, 0), over)
    }

    fn of(
        id: u64,
        image: u64,
        call: Call,
        surface: bool,
        shape: &Shape,
        (level, slice): (u32, u32),
        over: Option<Over>,
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
/// (0: none), and the thread's name without its number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Origin {
    sel: &'static str,
    message: u32,
    thread: &'static str,
}

/// The blocks imported, by rt's number, which rt never reuses, with where.
static IMPORTED: Mutex<BTreeMap<rt::BlockId, Origin>> = Mutex::new(BTreeMap::new());

/// The imports by origin.
static IMPORTS: Mutex<BTreeMap<Origin, u64>> = Mutex::new(BTreeMap::new());

/// Notes block `block`, imported now for `sel` (`super::buffer::import_pages`).
pub fn imported(block: rt::BlockId, sel: &'static str) {
    let origin = Origin {
        sel,
        message: MESSAGE.get(),
        thread: thread_class(),
    };
    IMPORTED.lock().insert(block, origin);
    *IMPORTS.lock().entry(origin).or_insert(0) += 1;
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
        let origin = *IMPORTED.lock().get(&buffer.block())?;
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

/// Counts a refusal of rt's `error` under its gap's kind, where it is a gap.
pub fn gap(error: &rt::Error) {
    let rt::Error::Unsupported(gap) = error else {
        return;
    };
    let name = match gap {
        rt::Gap::Draw { kind } => format!("Draw-{kind:?}"),
        other => {
            let debug = format!("{other:?}");
            let end = debug.find(|c: char| !c.is_alphanumeric());
            debug[..end.unwrap_or(debug.len())].to_owned()
        }
    };
    *GAPS.lock().entry(name).or_insert(0) += 1;
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
    *READS
        .lock()
        .entry(ReadClass { context, class })
        .or_insert(0) += 1;
    let mut firsts = FIRSTS.lock();
    if firsts.1 < FIRST_LINES && firsts.0.insert(pipeline) {
        firsts.1 += 1;
        drop(firsts);
        let context = context.map_or("unknown".to_owned(), |c| c.to_string());
        say(&format!(
            "reims-vgpu: rt-diag attachment-read first of a state: context {context}: {}\n",
            lines.join(" + ")
        ));
    }
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
    let presenters = PRESENTERS.lock().clone();
    let mut reads: BTreeMap<(String, &str), u64> = BTreeMap::new();
    for (key, n) in READS.lock().iter() {
        let process = match key.context {
            Some(c) if presenters.contains(&c) => "compositor",
            Some(_) => "app",
            None => "unknown",
        };
        *reads.entry((key.class.clone(), process)).or_insert(0) += n;
    }
    for ((class, process), n) in &reads {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag attachment-read: {process}: {class}: {n}"
        );
    }
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
            ("take", m.take_allocations),
        ];
        sites.sort_by_key(|(_, s)| std::cmp::Reverse(s.longest_nanos));
        let all: Vec<String> = sites
            .iter()
            .map(|(site, s)| {
                format!(
                    "{site} {} calls {} bytes longest {:.3} ms",
                    s.count,
                    s.bytes,
                    s.longest_nanos as f64 / 1e6
                )
            })
            .collect();
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag allocation sites, longest call first: {}",
            all.join(", ")
        );
    }
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

/// `origin` as a line names it: the selector, the message's type in hex, and the thread.
fn origin_name(origin: &Origin) -> String {
    format!(
        "{} message {:#x} thread {}",
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

    /// A gap is counted by its kind, a draw's by its draw kind; another error is not.
    #[test]
    fn gaps_count_by_kind() {
        gap(&rt::Error::Unsupported(rt::Gap::Draw {
            kind: rt::DrawKind::AttachmentRead,
        }));
        gap(&rt::Error::Unsupported(rt::Gap::CubicSampler));
        gap(&rt::Error::Quiesced);
        let gaps = GAPS.lock();
        assert!(gaps.get("Draw-AttachmentRead").is_some_and(|&n| n >= 1));
        assert!(gaps.get("CubicSampler").is_some_and(|&n| n >= 1));
        assert!(!gaps.keys().any(|k| k.contains("Quiesced")));
    }
}
