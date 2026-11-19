//! The rt platform's boot diagnostics, which a boot that checks it reads: lines
//! `reims-vgpu: rt-diag ...` on stderr, written with the process's counts ([`write`], which
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
//!   draw, and whether the draw's process presents to the display (the compositor) or not (an
//!   app). Class A, which rt builds, is the texture and the attachment's being one texture or
//!   twin views of one surface's image. The first refused draw of each pipeline state, ten at
//!   most, writes its line as it is refused.
//! - **Imageblock asks.** Each distinct combination of colour attachment formats (every index)
//!   and sample count of the render states asked `imageblockSampleLength`, with its asks
//!   ([`imageblock`]); its first ask writes its line.
//! - **Over-copies a presented frame.** rt's over-copies, their bytes and its submissions between
//!   two of the display's presents ([`presented`]): mean, worst, and how many frames copied none,
//!   1 to 9, 10 to 99 and 100 or more.
//! - **Textures over an imported block**, which rt keeps in an image of their own and copies the
//!   bytes of into it at each submission that uses them (rt's `over` module), by kind: format,
//!   size, rows, usage, storage mode, and where their block was imported (the selector, the
//!   decoder's message the import was made under and the thread's name without its number,
//!   [`imported`]). For each kind, the textures made, and their uses in submissions
//!   ([`Uses::submitted`]): their bytes, how many found the bytes as the submission before that
//!   used the texture left them (a hash of them at each commit, read while the guest may write
//!   them), how many found them changed after a submission that rendered into the texture or
//!   blitted into it, and how many sampled, rendered or blitted it.
//! - **Imports by origin**, as the textures' kinds name it.
//!
//! Totals, numbers, formats, sizes and usages only: no name, label or byte of the guest's leaves
//! the process. A diagnostic, which the contract leaves free: nothing the guest observes changes,
//! and a texture's bytes are only read.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

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
    /// In a heap (`newTextureWithDescriptor:offset:`).
    InHeap,
}

impl Call {
    fn name(self) -> &'static str {
        match self {
            Call::Plain => "ordinary",
            Call::SurfaceImage => "surface-image",
            Call::View => "view",
            Call::OverBuffer => "over-buffer",
            Call::InHeap => "in-heap",
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
        Made::of(
            id,
            of.image,
            Call::View,
            of.surface,
            &shape,
            (level, slice),
            None,
        )
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

/// Notes the guest's declared `usage` of `texture`, a view of a surface's shape image, where it is
/// one of the rt platform's (the decoder's surface shapes).
pub fn declared<T: 'static>(texture: &T, usage: u64) {
    let any: &dyn std::any::Any = texture;
    if let Some(t) = any.downcast_ref::<RtObj<RtTexture>>() {
        t.made().declared.store(usage, Relaxed);
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
    /// The buffer, which holds its block, and so the bytes, mapped.
    buffer: rt::Buffer,
    /// The bytes it lies over, from their first's offset in the buffer.
    offset: u64,
    len: u64,
    /// The hash of its bytes at the last submission that used it; 0 before any.
    last: AtomicU64,
    /// The last submission that used it rendered or blitted into it.
    written: AtomicBool,
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
    unchanged: AtomicU64,
    changed_after_write: AtomicU64,
    sampled: AtomicU64,
    rendered: AtomicU64,
    blitted: AtomicU64,
}

/// The kinds' totals.
static OVERS: Mutex<BTreeMap<OverKind, Arc<OverTotals>>> = Mutex::new(BTreeMap::new());

impl Over {
    /// What rt copies of a texture of `shape` over `buffer` from `offset`, rows `bytes_per_row`
    /// apart: `None` where the buffer's block was not imported, or the buffer has no contents.
    fn of(buffer: &rt::Buffer, shape: &Shape, offset: u64, bytes_per_row: u64) -> Option<Over> {
        let origin = *IMPORTED.lock().get(&buffer.block())?;
        buffer.contents()?;
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
        Some(Over {
            buffer: buffer.clone(),
            offset,
            len,
            last: AtomicU64::new(0),
            written: AtomicBool::new(false),
            totals,
        })
    }

    /// A hash of its bytes as they are now, never 0: FNV-1a of them, read one at a time through
    /// volatile reads, as the guest may write them meanwhile.
    fn hash(&self) -> u64 {
        let Some(contents) = self.buffer.contents() else {
            return 1;
        };
        let (Ok(offset), Ok(len)) = (usize::try_from(self.offset), usize::try_from(self.len))
        else {
            return 1;
        };
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        // SAFETY: the bytes lie in the buffer (`len` is cut at its length), whose block the buffer
        // holds mapped while it lives, and `self` holds it.
        let first = unsafe { contents.as_ptr().add(offset) };
        for i in 0..len {
            // SAFETY: as above, `i` below `len`.
            let byte = unsafe { first.add(i).read_volatile() };
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
        hash | 1
    }

    /// A submission used it as `how` says: counts it under its kind.
    fn submitted(&self, how: How) {
        let t = &self.totals;
        t.submissions.fetch_add(1, Relaxed);
        t.bytes.fetch_add(self.len, Relaxed);
        let hash = self.hash();
        let last = self.last.swap(hash, Relaxed);
        let wrote = self.written.swap(how.rendered || how.blitted_into, Relaxed);
        if last == hash {
            t.unchanged.fetch_add(1, Relaxed);
        } else if last != 0 && wrote {
            t.changed_after_write.fetch_add(1, Relaxed);
        }
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
        for (made, how) in by_texture.into_values() {
            if let Some(over) = &made.over {
                over.submitted(how);
            }
        }
    }
}

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
                 at {point}{} second {}",
                if class_a { "A" } else { "other" },
                texture.call_name(),
                texture.declared_usage(),
                texture.format,
                texture.storage,
                depth,
                meets.len() > 1,
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

/// The render states' imageblock asks by colour formats and sample count.
static IMAGEBLOCKS: Mutex<BTreeMap<([u32; 8], u32), u64>> = Mutex::new(BTreeMap::new());

/// Counts an ask of `imageblockSampleLength` of a render state of colour formats `formats` and
/// sample count `samples`; the first of the combination writes its line.
pub fn imageblock(formats: [u32; 8], samples: u32) {
    let mut asks = IMAGEBLOCKS.lock();
    let n = asks.entry((formats, samples)).or_insert(0);
    *n += 1;
    if *n == 1 {
        drop(asks);
        say(&format!(
            "reims-vgpu: rt-diag imageblock first: formats {formats:?} samples {samples}\n"
        ));
    }
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
    for ((formats, samples), n) in IMAGEBLOCKS.lock().iter() {
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag imageblock: formats {formats:?} samples {samples}: {n} asks"
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
    for (kind, t) in OVERS.lock().iter() {
        let n = |a: &AtomicU64| a.load(Relaxed);
        let _ = writeln!(
            out,
            "reims-vgpu: rt-diag over-texture: format {} {}x{} rows {} usage {:#x} storage {} over \
             {}: made {}, submissions {} ({} bytes), unchanged {}, changed after its GPU write {}, \
             sampled {}, rendered {}, blitted {}",
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
            n(&t.unchanged),
            n(&t.changed_after_write),
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
