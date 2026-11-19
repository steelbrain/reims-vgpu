//! A no-copy buffer's escaped identity (docs/graphics-kext/design.md, "A buffer's escaped
//! identity"). The host reads a no-copy buffer's bytes in place (`BUFFER_OVER_PAGES`,
//! `BUFFER_OVER_HOST_MEMORY`, `Managed` neither), and the guest's Metal hands the buffer's
//! identity out with no message naming the buffer at four acts: the app's read of its
//! `gpuAddress`, an argument encoder's write of it or of a texture over it, the app's read of such
//! a texture's or view's `gpuResourceID`, and the buffer, or such a texture, joining a residency
//! set. From then on the GPU may reach the buffer through memory alone. The guest notes the first
//! such act of each buffer with `BUFFER_ESCAPED`, which ring 0 carries out before any command
//! buffer of the process that could use the identity: the guest posts it, or holds it to go
//! before its next commit's messages, before the identity leaves its Metal, and a command buffer
//! is carried out only after every ring-0 message its process posted before it.
//!
//! The decoder marks the buffer's entry escaped ([`Escape`]), once for its life: a later note for
//! it changes nothing. The mark changes nothing the guest observes. It is counted, by the act
//! that noted it ([`ESCAPED_BY`]), with the escaped buffers made over a copy of their pages
//! ([`ESCAPED_COPIES`]) and the no-copy buffers made, which the escapes are a share of
//! ([`NO_COPY_BUFFERS`]), in the degradation counts' line (docs/conventions.md), and each act's
//! first escape is logged at info. A note for an object that is no such buffer is a bad
//! reference, and one for an object whose create the host refused is dropped, as any naming of
//! one is ([`super::refused`]).
//!
//! Where the process records its platform's calls (`REIMS_VGPU_RECORD`), the first note is also
//! a record on the buffer, `m2v.bufferEscaped` `[act]`, queued at the note
//! ([`recorder::buffer_escaped`]): the stream holds it after the memory writes of each command
//! buffer committed before it and before those of each committed after. A replay player stops
//! there, typed, where a later write holds an address in the buffer. A later note, a note for a
//! refused create and a bad reference write none.
//!
//! An escaped buffer over a copy of its pages is one the GPU may read through memory, with no
//! binding naming it, after the guest's CPU changed its pages. Metal requires a function's reach
//! of a buffer through memory to be made resident by its encoder (`useResource:usage:`), which
//! names it as a use, and the runtime brings every copy a command buffer's uses name up to date at
//! its commit, and every copy where its work may reach memory no use names (metal2vulkan's
//! `command_buffer/mod.rs`, "Every byte the CPU changed in the pages of a copied host buffer"),
//! so the decoder adds nothing to a commit for one, and the guest observes nothing of this: it
//! keeps each process's count of its live escaped copies ([`Context::escaped_copies`]), and
//! counts each command buffer begun of a process that holds one ([`ESCAPED_COPY_COMMITS`]), the
//! first logged at info. The tests pin the runtime's copy in of such a copy.
//!
//! [`Context::escaped_copies`]: super::tables::Context::escaped_copies

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use reims_vgpu_graphics::wire::RgfxBufferEscaped;
use reims_vgpu_vulkan::answer::count_kind;
use reims_vgpu_vulkan::core::num::ToU64 as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::recorder;

use super::tables::Entry;
use super::{DecodeError, GraphicsDecoder, ObjHandle, context_of, read};

/// The kinds a buffer's escape is counted as, by the act that noted it, at its
/// `RGFX_BUFFER_ESCAPED_*` value: the app's read of its `gpuAddress`, an argument encoder's write,
/// the app's read of a texture's `gpuResourceID`, a residency set's add.
pub(crate) const ESCAPED_BY: [&str; 4] = [
    "buffer-escaped-gpu-address",
    "buffer-escaped-argument-encoder",
    "buffer-escaped-resource-id",
    "buffer-escaped-residency-set",
];

/// The kind an escaped buffer made over a copy of its pages ([`NoCopy::ReadOnlyPages`]) is
/// counted as too, beside its act's.
pub(crate) const ESCAPED_COPIES: &str = "buffer-escaped-read-only-pages";

/// The kind each no-copy buffer made ([`Escape::made`]) is counted as: those the escapes are a
/// share of.
pub(crate) const NO_COPY_BUFFERS: &str = "no-copy-buffers";

/// The kind each command buffer begun of a process that holds an escaped buffer over a copy of
/// its pages is counted as ([`GraphicsDecoder::escaped_copy_commit`]).
pub(crate) const ESCAPED_COPY_COMMITS: &str = "escaped-copy-commits";

/// Whether this process logged a command buffer of [`ESCAPED_COPY_COMMITS`]: the first is
/// logged, the rest counted.
static COMMIT_LOGGED: AtomicBool = AtomicBool::new(false);

/// Whether this process logged an escape by each act of [`ESCAPED_BY`], at its index: the first
/// is logged, the rest counted.
static LOGGED: [AtomicBool; ESCAPED_BY.len()] =
    [const { AtomicBool::new(false) }; ESCAPED_BY.len()];

/// What a no-copy buffer lies over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NoCopy {
    /// A block of the region's (`BUFFER_OVER_HOST_MEMORY`), in place.
    HostMemory,
    /// A pages block its process maps writable (`BUFFER_OVER_PAGES`), in place.
    Pages,
    /// A pages block its process maps read-only (`BUFFER_OVER_PAGES`): a copy of its pages on
    /// Linux and Windows, which read the block's view ([`super::surface_pages`]).
    ReadOnlyPages,
}

/// A no-copy buffer's escape: what it lies over, and whether the guest noted that its identity
/// escaped. Set once, never cleared.
pub(crate) struct Escape {
    over: NoCopy,
    escaped: AtomicBool,
    /// Where it lies over a copy of its pages and escaped, its context's count of such buffers
    /// ([`super::tables::Context::escaped_copies`]), which it added itself to at its first note
    /// and leaves at its drop.
    copies: OnceLock<Arc<AtomicUsize>>,
}

impl Escape {
    /// A no-copy buffer made over `over`, whose identity has not escaped, counted among those
    /// made ([`NO_COPY_BUFFERS`]).
    pub(crate) fn made(over: NoCopy) -> Self {
        count_kind(NO_COPY_BUFFERS);
        Escape {
            over,
            escaped: AtomicBool::new(false),
            copies: OnceLock::new(),
        }
    }

    /// What the buffer lies over.
    pub(crate) fn over(&self) -> NoCopy {
        self.over
    }

    /// Whether the guest noted that the buffer's identity escaped.
    #[cfg(test)]
    pub(crate) fn escaped(&self) -> bool {
        self.escaped.load(Ordering::Relaxed)
    }

    /// The guest's note that buffer `obj_id`'s identity escaped by the act at `act` of
    /// [`ESCAPED_BY`]: the first marks it, records it where the process records, on `buffer`,
    /// the platform's (`m2v.bufferEscaped`, [`recorder::buffer_escaped`]), and counts it, a
    /// buffer over a copy of its pages added to `copies`, its context's count of its escaped
    /// copies, and the first of its act in this process is logged; a later one changes nothing.
    fn note(&self, act: usize, obj_id: u64, buffer: &dyn Any, copies: Option<Arc<AtomicUsize>>) {
        if self.escaped.swap(true, Ordering::Relaxed) {
            return;
        }
        recorder::buffer_escaped(buffer, act.to_u64());
        count_kind(ESCAPED_BY[act]);
        if self.over == NoCopy::ReadOnlyPages {
            count_kind(ESCAPED_COPIES);
            if let Some(copies) = copies {
                copies.fetch_add(1, Ordering::Relaxed);
                let _ = self.copies.set(copies);
            }
        }
        if !LOGGED[act].swap(true, Ordering::Relaxed) {
            log::info!(
                target: "reims_vgpu::graphics",
                "the identity of no-copy buffer {obj_id:#x}, over {:?}, left the guest's Metal \
                 with no message naming the buffer ({}): the host takes it as reachable through \
                 memory for its life (logged once per act, counted as {})",
                self.over,
                ESCAPED_BY[act],
                ESCAPED_BY[act]
            );
        }
    }
}

impl Drop for Escape {
    /// An escaped buffer over a copy of its pages leaves its context's count of them.
    fn drop(&mut self) {
        if let Some(copies) = self.copies.get() {
            copies.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// `BUFFER_ESCAPED`: the buffer it names marked escaped ([`Escape`]), once for its life. A
    /// message too short or of an act it does not know is malformed; an object the sender does
    /// not hold, or one that is no no-copy buffer, is a bad reference; and one whose create the
    /// host refused is named as such ([`super::refused`]) and nothing more.
    pub(super) fn buffer_escaped(&self, message: &[u8]) -> Result<(), DecodeError> {
        let Some(m) = read::<RgfxBufferEscaped>(message) else {
            return Err(DecodeError::Malformed);
        };
        let Some(act) = usize::try_from(m.kind)
            .ok()
            .filter(|&a| a < ESCAPED_BY.len())
        else {
            return Err(DecodeError::Malformed);
        };
        match self.entry(m.obj_id) {
            Some(Entry {
                handle: ObjHandle::Refused(refused),
                ..
            }) => {
                refused.used(m.obj_id);
                Ok(())
            }
            Some(Entry {
                handle: ObjHandle::Buffer(buffer),
                escape: Some(escape),
                ..
            }) => {
                let copies = self.escaped_copies(context_of(m.obj_id), Arc::clone);
                escape.note(act, m.obj_id, buffer, copies);
                Ok(())
            }
            _ => Err(DecodeError::BadReference),
        }
    }

    /// What `f` answers of context `context`'s count of its live escaped copies ([`Escape`]),
    /// where the message holds a view of its table: it does once it named an object of it.
    fn escaped_copies<T>(&self, context: u32, f: impl FnOnce(&Arc<AtomicUsize>) -> T) -> Option<T> {
        let views = self.views.borrow();
        let view = views.iter().find(|v| v.context().id == context)?;
        Some(f(&view.context().escaped_copies))
    }

    /// A command buffer of context `context` begun, once it named the buffer's queue: counted
    /// ([`ESCAPED_COPY_COMMITS`]) where the context holds an escaped buffer over a copy of its
    /// pages, the first logged. The runtime brings the copy up to date at its commit where its
    /// work may reach it, as it does any copy; nothing is added to it.
    pub(super) fn escaped_copy_commit(&self, context: u32) {
        let held = self.escaped_copies(context, |copies| copies.load(Ordering::Relaxed));
        let Some(held @ 1..) = held else {
            return;
        };
        count_kind(ESCAPED_COPY_COMMITS);
        if !COMMIT_LOGGED.swap(true, Ordering::Relaxed) {
            log::info!(
                target: "reims_vgpu::graphics",
                "a command buffer of context {context}, which holds {held} escaped no-copy \
                 buffers over a copy of their pages, begun: the runtime brings the copies up to \
                 date at its commit where its work may reach them (logged once, counted as {})",
                ESCAPED_COPY_COMMITS
            );
        }
    }

    /// The escape of the no-copy buffer `obj_id` names, where the sender holds one (for the
    /// tests).
    #[cfg(test)]
    pub(super) fn escape_of(&self, obj_id: u64) -> Option<&Escape> {
        self.entry(obj_id).and_then(|e| e.escape.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_graphics::ops::{RGFX_COMMAND_BUFFER_OPS_OFFSET, RgfxCommandBuffer};
    use reims_vgpu_graphics::wire::{
        RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER, RGFX_BUFFER_ESCAPED_GPU_ADDRESS,
        RGFX_BUFFER_ESCAPED_RESIDENCY_SET, RGFX_BUFFER_ESCAPED_RESOURCE_ID, RGFX_CMD_BUFFER_CREATE,
        RGFX_CMD_BUFFER_ESCAPED, RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_BUFFER_OVER_PAGES,
        RGFX_CMD_COMMAND_BUFFER, RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_QUEUE_CREATE, RGFX_STATUS_OK,
        RGFX_SURFACE_PAGES_PRIVATE, RGFX_SURFACE_PAGES_READ_ONLY, RgfxBufferCreate,
        RgfxBufferOverHostMemory, RgfxBufferOverPages, RgfxMsgHeader, RgfxObjectDestroy,
        RgfxQueueCreate,
    };
    use zerocopy::IntoBytes;

    use super::super::surface_pages::READ_ONLY_OPTION;
    use super::super::surface_pages::tests::{BLOCK, Rig, rig};
    use super::*;

    fn header(r#type: u32) -> RgfxMsgHeader {
        RgfxMsgHeader {
            r#type,
            size: 0,
            seqno: 0,
        }
    }

    /// A `BUFFER_ESCAPED` of `obj_id` by the act `kind`, carried out.
    fn note(r: &mut Rig, obj_id: u64, kind: u32) -> Result<(), DecodeError> {
        let m = RgfxBufferEscaped {
            header: header(RGFX_CMD_BUFFER_ESCAPED),
            obj_id,
            kind,
            reserved_1c: 0,
        };
        r.dec.decode(&r.device, m.as_bytes()).unwrap()
    }

    /// A `BUFFER_OVER_HOST_MEMORY` of `obj_id`, a page at region offset `at`, of `options`.
    fn host_memory(r: &mut Rig, obj_id: u64, at: u64, options: u32) -> Result<(), DecodeError> {
        let m = RgfxBufferOverHostMemory {
            header: header(RGFX_CMD_BUFFER_OVER_HOST_MEMORY),
            obj_id,
            region_offset: at,
            length: r.page,
            options,
            ..RgfxBufferOverHostMemory::default()
        };
        r.dec.decode(&r.device, m.as_bytes()).unwrap()
    }

    /// This process's count of the kind `code`, which another test in it may have counted
    /// before.
    fn counted(code: &str) -> u64 {
        (reims_vgpu_vulkan::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// The counts of every kind this module counts: [`ESCAPED_BY`]'s, then [`ESCAPED_COPIES`]'s
    /// and [`NO_COPY_BUFFERS`]'.
    fn counts() -> [u64; 6] {
        let mut all = [0; 6];
        for (n, code) in all.iter_mut().zip(
            ESCAPED_BY
                .iter()
                .copied()
                .chain([ESCAPED_COPIES, NO_COPY_BUFFERS]),
        ) {
            *n = counted(code);
        }
        all
    }

    /// The three kinds of no-copy buffer, each made and counted, and each act's note: the first
    /// for a buffer marks it escaped and counts its act (an escaped one over a copy of its pages
    /// counted as one too), and a later note for it, by any act, changes nothing
    #[test]
    fn a_note_marks_a_no_copy_buffer_escaped_once_by_its_act() {
        let mut r = rig(16);
        let page = r.page;
        let writable = BLOCK + 0x8000;
        let in_region = BLOCK + 0x20000;
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(0, 4)], 4 * page, RGFX_SURFACE_PAGES_PRIVATE),
            RGFX_STATUS_OK
        );
        let read_only = RGFX_SURFACE_PAGES_PRIVATE | RGFX_SURFACE_PAGES_READ_ONLY;
        assert_eq!(
            r.pages_flagged(writable, 0, &[(8, 4)], 4 * page, read_only),
            RGFX_STATUS_OK
        );
        let before = counts();
        assert_eq!(r.pages_buffer_of(10, 0, BLOCK, 0, page, 0), Ok(()));
        assert_eq!(
            r.pages_buffer_of(11, 0, writable, 0, page, READ_ONLY_OPTION),
            Ok(())
        );
        assert_eq!(host_memory(&mut r, 12, in_region, 0), Ok(()));
        assert_eq!(r.pages_buffer_of(13, 0, BLOCK, page, page, 0), Ok(()));
        let made = counts();
        assert_eq!(made[5], before[5] + 4, "four no-copy buffers made");
        assert_eq!(made[..5], before[..5], "none escaped");
        let over = |r: &Rig, obj_id| r.dec.escape_of(obj_id).map(Escape::over);
        assert_eq!(over(&r, 10), Some(NoCopy::Pages));
        assert_eq!(over(&r, 11), Some(NoCopy::ReadOnlyPages));
        assert_eq!(over(&r, 12), Some(NoCopy::HostMemory));
        let escaped = |r: &Rig, obj_id| r.dec.escape_of(obj_id).is_some_and(Escape::escaped);
        assert!(!(10..14).any(|id| escaped(&r, id)));

        let acts = [
            (10, RGFX_BUFFER_ESCAPED_GPU_ADDRESS),
            (11, RGFX_BUFFER_ESCAPED_RESIDENCY_SET),
            (12, RGFX_BUFFER_ESCAPED_RESOURCE_ID),
            (13, RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER),
        ];
        for (obj_id, kind) in acts {
            assert_eq!(note(&mut r, obj_id, kind), Ok(()));
            assert!(escaped(&r, obj_id));
        }
        let noted = counts();
        let one_each: Vec<u64> = (0..4).map(|i| noted[i] - made[i]).collect();
        assert_eq!(one_each, [1, 1, 1, 1], "one of each act");
        assert_eq!(noted[4], made[4] + 1, "the copy's escape");
        assert_eq!(noted[5], made[5]);

        // A repeat, by its act or another, changes nothing.
        for (obj_id, _) in acts {
            for kind in 0..4 {
                assert_eq!(note(&mut r, obj_id, kind), Ok(()));
            }
        }
        assert_eq!(counts(), noted);
    }

    /// A note for an id the sender holds no object under, or for an object that is no no-copy
    /// buffer (a device buffer, a `Managed` one over host memory), is a bad reference; one too
    /// short, or of an act the host does not know, is malformed; none marks or counts anything,
    /// and none kills the decoder
    #[test]
    fn a_note_for_no_no_copy_buffer_is_a_bad_reference() {
        let mut r = rig(16);
        let device_buffer = RgfxBufferCreate {
            header: header(RGFX_CMD_BUFFER_CREATE),
            obj_id: 20,
            length: 0x1000,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(
            r.dec.decode(&r.device, device_buffer.as_bytes()).unwrap(),
            Ok(())
        );
        // Bits 4 and 5 of the options: Managed.
        let managed = 1 << 4;
        assert_eq!(host_memory(&mut r, 21, BLOCK + 0x20000, managed), Ok(()));
        assert_eq!(host_memory(&mut r, 22, BLOCK + 0x30000, 0), Ok(()));
        let before = counts();
        for obj_id in [20, 21, 0x99] {
            assert_eq!(
                note(&mut r, obj_id, RGFX_BUFFER_ESCAPED_GPU_ADDRESS),
                Err(DecodeError::BadReference),
                "object {obj_id:#x}"
            );
        }
        assert_eq!(note(&mut r, 22, 4), Err(DecodeError::Malformed));
        let whole = RgfxBufferEscaped {
            header: header(RGFX_CMD_BUFFER_ESCAPED),
            obj_id: 22,
            ..RgfxBufferEscaped::default()
        };
        let short = &whole.as_bytes()[..size_of::<RgfxBufferEscaped>() - 4];
        assert_eq!(
            r.dec.decode(&r.device, short).unwrap(),
            Err(DecodeError::Malformed)
        );
        assert_eq!(counts(), before);
        assert!(!r.dec.escape_of(22).is_some_and(Escape::escaped));
        assert_eq!(note(&mut r, 22, RGFX_BUFFER_ESCAPED_GPU_ADDRESS), Ok(()));
        assert!(r.dec.escape_of(22).is_some_and(Escape::escaped));
    }

    /// A note for a no-copy buffer whose create the host refused names the refused object, as
    /// any naming of one does, and is carried out: nothing is marked or counted
    #[test]
    fn a_note_for_a_refused_no_copy_buffer_is_dropped() {
        let mut r = rig(16);
        let page = r.page;
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(0, 4)], 4 * page, RGFX_SURFACE_PAGES_PRIVATE),
            RGFX_STATUS_OK
        );
        // Past the block's view: refused, and its id stands for the refused buffer.
        let past = RgfxBufferOverPages {
            header: header(RGFX_CMD_BUFFER_OVER_PAGES),
            obj_id: 30,
            block_offset: BLOCK,
            block_size: 0x2000,
            length: 8 * page,
            ..RgfxBufferOverPages::default()
        };
        assert_eq!(
            r.dec.decode_refusing(&r.device, past.as_bytes()).unwrap(),
            Err(DecodeError::Refused)
        );
        let before = counts();
        assert_eq!(note(&mut r, 30, RGFX_BUFFER_ESCAPED_GPU_ADDRESS), Ok(()));
        assert_eq!(counts(), before);
        assert!(r.dec.escape_of(30).is_none());
    }

    /// A `QUEUE_CREATE` of queue 1, carried out.
    fn queue(r: &mut Rig) {
        let m = RgfxQueueCreate {
            header: header(RGFX_CMD_QUEUE_CREATE),
            obj_id: 1,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(r.dec.decode(&r.device, m.as_bytes()).unwrap(), Ok(()));
    }

    /// An empty `COMMAND_BUFFER` `cb_obj_id` of queue 1, carried out.
    fn commit(r: &mut Rig, cb_obj_id: u64) {
        let m = RgfxCommandBuffer {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMMAND_BUFFER,
                size: RGFX_COMMAND_BUFFER_OPS_OFFSET,
                seqno: cb_obj_id,
            },
            queue_obj_id: 1,
            cb_obj_id,
            op_bytes: 0,
        };
        assert_eq!(r.dec.decode(&r.device, m.as_bytes()).unwrap(), Ok(()));
    }

    /// Each command buffer begun of a process that holds an escaped buffer over a copy of its
    /// pages is counted, and only while it holds one: none before the copy's note, none for an
    /// escaped buffer over pages in place, and none once the copy is destroyed
    #[test]
    fn a_command_buffer_of_a_process_holding_an_escaped_copy_is_counted() {
        let mut r = rig(16);
        let page = r.page;
        let copied = BLOCK + 0x8000;
        assert_eq!(
            r.pages_flagged(BLOCK, 0, &[(0, 4)], 4 * page, RGFX_SURFACE_PAGES_PRIVATE),
            RGFX_STATUS_OK
        );
        let read_only = RGFX_SURFACE_PAGES_PRIVATE | RGFX_SURFACE_PAGES_READ_ONLY;
        assert_eq!(
            r.pages_flagged(copied, 0, &[(8, 4)], 4 * page, read_only),
            RGFX_STATUS_OK
        );
        queue(&mut r);
        assert_eq!(r.pages_buffer_of(10, 0, BLOCK, 0, page, 0), Ok(()));
        assert_eq!(
            r.pages_buffer_of(11, 0, copied, 0, page, READ_ONLY_OPTION),
            Ok(())
        );
        // Two command buffers from `first`: the count after.
        let two = |r: &mut Rig, first: u64| {
            commit(r, first);
            commit(r, first + 1);
            counted(ESCAPED_COPY_COMMITS)
        };
        let before = counted(ESCAPED_COPY_COMMITS);
        assert_eq!(two(&mut r, 20), before, "nothing escaped");
        assert_eq!(note(&mut r, 10, RGFX_BUFFER_ESCAPED_GPU_ADDRESS), Ok(()));
        assert_eq!(two(&mut r, 22), before, "escaped in place");
        assert_eq!(
            note(&mut r, 11, RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER),
            Ok(())
        );
        assert_eq!(two(&mut r, 24), before + 2, "the copy escaped");
        let destroy = RgfxObjectDestroy {
            header: header(RGFX_CMD_OBJECT_DESTROY),
            obj_id: 11,
        };
        assert_eq!(r.dec.decode(&r.device, destroy.as_bytes()).unwrap(), Ok(()));
        assert_eq!(two(&mut r, 26), before + 2, "the copy destroyed");
    }

    /// The Vulkan crate's metallib of the kernel that reaches a buffer through an address it
    /// loads from another (its `reach.metal`), which a host without Metal takes as a Mac does.
    const REACH: &[u8] = include_bytes!("../../../../reims-vgpu-vulkan/metallibs/reach.metallib");

    /// On the Vulkan platform, a buffer over a copy of its read-only pages whose identity escaped
    /// (an argument encoder's write of its address) is read by a kernel that reaches it only
    /// through that address, the encoder having made it resident (`useResource:usage:`), as Metal
    /// requires of a buffer a function reaches through memory: before the guest's CPU writes its
    /// pages the kernel reads their old bytes, and after, their new ones. The decoder adds nothing
    /// to the command buffer for the escaped copy: this holds the runtime's own copy in, at a
    /// commit, of each copy its encoders' uses name (metal2vulkan's `command_buffer/mod.rs`, "Every
    /// byte the CPU changed in the pages of a copied host buffer"). A reach of a buffer no encoder
    /// made resident is undefined in Metal, and reads zero on the runtime (its `residency.rs`).
    #[test]
    fn on_vulkan_an_escaped_copy_reached_by_its_address_reads_the_cpus_writes() {
        use reims_vgpu_graphics::wire::{
            RGFX_CMD_SURFACE_PAGES, RGFX_CURSOR_SPAN, RgfxPageRun, RgfxSurfacePages,
        };
        use reims_vgpu_vulkan::VulkanPlatform;
        use reims_vgpu_vulkan::core::memory::map::MemoryMap as CoreMemoryMap;
        use reims_vgpu_vulkan::core::platform::blit::ResourceRef;
        use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Library as _};
        use reims_vgpu_vulkan::core::platform::types::{MtlSize, NsRange};
        use reims_vgpu_vulkan::core::raise::OnFault;

        use super::super::surface_pages::Lease;
        use super::super::surface_pages::costs::{RAM, guest_ram};
        use crate::platform::ApiPlatform as _;

        let _pool = reims_vgpu_vulkan::core::rt::Pool::here();
        let platform = VulkanPlatform::process();
        let device = platform.metal_device(None).unwrap();
        let page = platform.vm().page_size().max(4096);
        // The guest's RAM, which outlives the decoder and its views.
        let ram = guest_ram(16 * page, true);
        let lease: Lease = Arc::from(platform.guest_memory(&ram.map));
        let core =
            CoreMemoryMap::new(Arc::clone(&platform), &ram.map.to_core()).expect("a memory map");
        let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(crate::platform::test_region(&*platform, size as usize));
        let mut dec = GraphicsDecoder::<VulkanPlatform>::new(Arc::clone(&platform), region);
        dec.set_guest_memory(core, lease);
        assert!(
            dec.surface_pages_available(),
            "no views of guest RAM on a host page of {page:#x}"
        );
        // The buffer's first two words, three pages into the RAM.
        // SAFETY: three pages into the RAM's `16 * page` bytes, which `ram` keeps mapped.
        let words = unsafe { ram.at.add((3 * page) as usize).cast::<u32>() };
        let write = |at: usize, value: u32| {
            // SAFETY: a word of the buffer's pages, written while no command runs.
            unsafe { words.add(at).write_volatile(value) }
        };
        const OLD: u32 = 0x1111_1111;
        const NEW: u32 = 0x2222_2222;
        // The kernel writes the word after the one it reads with what it read there: no change.
        const AFTER: u32 = 0x5a5a_5a5a;
        write(0, OLD);
        write(1, AFTER);

        let blocks = RgfxSurfacePages {
            header: header(RGFX_CMD_SURFACE_PAGES),
            context: 0,
            run_count: 1,
            block_offset: BLOCK,
            block_size: 0x2000,
            runs_offset: BLOCK + 0x1000,
            length: page,
        };
        let runs = [RgfxPageRun {
            address: RAM + 3 * page,
            length: page,
        }];
        let read_only = RGFX_SURFACE_PAGES_PRIVATE | RGFX_SURFACE_PAGES_READ_ONLY;
        assert_eq!(
            dec.surface_pages(&device, &blocks, &runs, read_only),
            RGFX_STATUS_OK
        );
        let over = RgfxBufferOverPages {
            header: header(RGFX_CMD_BUFFER_OVER_PAGES),
            obj_id: 10,
            block_offset: BLOCK,
            block_size: 0x2000,
            length: page,
            options: READ_ONLY_OPTION,
            ..RgfxBufferOverPages::default()
        };
        assert_eq!(dec.decode(&device, over.as_bytes()).unwrap(), Ok(()));
        let note = RgfxBufferEscaped {
            header: header(RGFX_CMD_BUFFER_ESCAPED),
            obj_id: 10,
            kind: RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER,
            reserved_1c: 0,
        };
        assert_eq!(dec.decode(&device, note.as_bytes()).unwrap(), Ok(()));
        assert_eq!(
            dec.escape_of(10).map(|e| (e.over(), e.escaped())),
            Some((NoCopy::ReadOnlyPages, true))
        );
        let Some(ObjHandle::Buffer(copy)) = dec.object(10) else {
            panic!("no buffer 10");
        };

        let library = device.new_library(REACH).unwrap().unwrap().unwrap();
        let f = library
            .new_function(OnFault::Die, "reach")
            .unwrap()
            .unwrap();
        let state = device
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap();
        let shared = |length: u64| {
            device
                .new_buffer_of_length(OnFault::Die, length, 0)
                .unwrap()
                .unwrap()
        };
        // The address the argument encoder wrote, and the kernel's two words.
        let (addresses, out) = (shared(8), shared(8));
        // SAFETY: Shared buffers of 8 bytes, which no command uses yet.
        unsafe {
            addresses
                .contents()
                .as_ptr::<u64>()
                .write(copy.gpu_address());
            out.contents().as_ptr::<u32>().add(1).write(AFTER);
        }
        let queue = device.new_command_queue().unwrap();
        // What the kernel reads through the address alone.
        let reach = || {
            let cb = queue.command_buffer(OnFault::Die).unwrap().unwrap();
            let e = cb.compute_encoder(OnFault::Die).unwrap().unwrap();
            let one = MtlSize::new(1, 1, 1);
            for cmd in [
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state),
                },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&addresses), Some(&out)],
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
                // MTLResourceUsageRead.
                ComputeCmd::UseResources {
                    resources: &[Some(ResourceRef::Buffer(copy))],
                    usage: 1,
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: one,
                    threads_per_threadgroup: one,
                },
                ComputeCmd::EndEncoding,
            ] {
                e.encode(OnFault::Die, cmd).unwrap();
            }
            cb.commit(OnFault::Die).unwrap();
            cb.wait_until_completed(OnFault::Die).unwrap();
            assert!(cb.error().is_none(), "{:?}", cb.error());
            // SAFETY: a Shared buffer's first word, read after the command buffer completed.
            unsafe { out.contents().as_ptr::<u32>().read() }
        };
        assert_eq!(reach(), OLD, "before the CPU's write");
        write(0, NEW);
        assert_eq!(reach(), NEW, "after the CPU's write");
    }

    /// The marker that makes a run of [`on_vulkan_a_recording_holds_each_buffers_first_escape`]
    /// the child, which records.
    const RECORD_CHILD: &str = "REIMS_VGPU_ESCAPE_RECORD_CHILD";

    /// With the process recording, each no-copy buffer's first note is one `m2v.bufferEscaped`
    /// record on the buffer, `[act]`, its act's `RGFX_BUFFER_ESCAPED_*` value; a later note for
    /// it, by any act, a note for a buffer whose create the host refused, and a bad reference (a
    /// device buffer, an id the sender holds no object under) write none. The recorder is the
    /// process's, so the test runs in a child of its own, recording into a scratch directory.
    #[test]
    fn on_vulkan_a_recording_holds_each_buffers_first_escape() {
        use reims_vgpu_vulkan::recorder::ENV_VAR;
        if std::env::var_os(RECORD_CHILD).is_some() {
            let streams = std::env::var_os(ENV_VAR).expect("the child records");
            recorded_escapes(std::path::Path::new(&streams));
            return;
        }
        let dir = std::env::temp_dir().join(format!("rvg-recorder-escape-{}", std::process::id()));
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "graphics::decode::escape::tests::on_vulkan_a_recording_holds_each_buffers_first_escape",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(RECORD_CHILD, "1")
            .env(ENV_VAR, dir.join("streams"))
            .output()
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(out.status.success(), "{out:?}");
        let ran = String::from_utf8_lossy(&out.stdout).contains("1 passed");
        assert!(ran, "the child ran no test: {out:?}");
    }

    /// [`on_vulkan_a_recording_holds_each_buffers_first_escape`]'s child, recording into
    /// `streams`.
    fn recorded_escapes(streams: &std::path::Path) {
        use reims_vgpu_graphics::wire::RGFX_CURSOR_SPAN;
        use reims_vgpu_vulkan::VulkanPlatform as V;
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::recorder::BUFFER_ESCAPED;
        use reims_vgpu_vulkan::recorder::format::{self, Call, Value};

        use crate::platform::ApiPlatform as _;

        let platform = V::process();
        assert!(
            reims_vgpu_vulkan::recorder::on(),
            "the child does not record"
        );
        let device = platform.metal_device(None).unwrap();
        let page = platform.vm().page_size().max(4096);
        let size = BLOCK + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(crate::platform::test_region(&*platform, size as usize));
        let mut dec = GraphicsDecoder::<V>::new(Arc::clone(&platform), region);
        let over = |obj_id, region_offset, length| RgfxBufferOverHostMemory {
            header: header(RGFX_CMD_BUFFER_OVER_HOST_MEMORY),
            obj_id,
            region_offset,
            length,
            ..RgfxBufferOverHostMemory::default()
        };
        // Two no-copy buffers, of three pages and of five: their creates' records tell them
        // apart.
        for (obj_id, at, pages) in [(10, BLOCK + 0x20000, 3), (11, BLOCK + 0x40000, 5)] {
            let m = over(obj_id, at, pages * page);
            assert_eq!(dec.decode(&device, m.as_bytes()).unwrap(), Ok(()));
        }
        let device_buffer = RgfxBufferCreate {
            header: header(RGFX_CMD_BUFFER_CREATE),
            obj_id: 20,
            length: 0x1000,
            ..RgfxBufferCreate::default()
        };
        assert_eq!(
            dec.decode(&device, device_buffer.as_bytes()).unwrap(),
            Ok(())
        );
        // Of no bytes: refused, and its id stands for the refused buffer.
        let refused = over(30, BLOCK + 0x60000, 0);
        assert_eq!(
            dec.decode_refusing(&device, refused.as_bytes()).unwrap(),
            Err(DecodeError::Refused)
        );
        let mut note = |obj_id, kind| {
            let m = RgfxBufferEscaped {
                header: header(RGFX_CMD_BUFFER_ESCAPED),
                obj_id,
                kind,
                reserved_1c: 0,
            };
            dec.decode(&device, m.as_bytes()).unwrap()
        };
        assert_eq!(note(10, RGFX_BUFFER_ESCAPED_RESOURCE_ID), Ok(()));
        for kind in 0..4 {
            assert_eq!(note(10, kind), Ok(()));
        }
        assert_eq!(note(11, RGFX_BUFFER_ESCAPED_RESIDENCY_SET), Ok(()));
        assert_eq!(note(11, RGFX_BUFFER_ESCAPED_GPU_ADDRESS), Ok(()));
        assert_eq!(note(30, RGFX_BUFFER_ESCAPED_GPU_ADDRESS), Ok(()));
        for obj_id in [20, 0x99] {
            assert_eq!(
                note(obj_id, RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER),
                Err(DecodeError::BadReference)
            );
        }
        // A call after the notes: the stream holds their records once it holds its.
        let _queue = device.new_command_queue().unwrap();

        let segment = streams.join(format!("{}.0000.m2vr", std::process::id()));
        let mut calls = Vec::new();
        for _ in 0..200 {
            let read = std::fs::read(&segment)
                .ok()
                .and_then(|b| format::read(&b, &mut Vec::new()).ok());
            calls = (read.into_iter().flat_map(|r| r.records))
                .filter_map(|r| match r {
                    format::Record::Call(c) => Some(c),
                    _ => None,
                })
                .collect::<Vec<Call>>();
            if calls.iter().any(|c| c.selector == "newCommandQueue") {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        // Each no-copy buffer's stream id, by its create's length.
        let made = |length: u64| {
            let length_of = |c: &Call| match c.args.first() {
                Some(Value::Map(fields)) => (fields.iter())
                    .find(|(k, _)| k == "length")
                    .map(|(_, v)| v.clone()),
                _ => None,
            };
            let create = calls.iter().find(|c| {
                c.selector == "newBufferWithDescriptor:"
                    && c.result != 0
                    && length_of(c) == Some(Value::Uint(length))
            });
            create
                .unwrap_or_else(|| panic!("no create of {length:#x} bytes: {calls:?}"))
                .result
        };
        let (ten, eleven) = (made(3 * page), made(5 * page));
        let escape = |receiver, kind: u32| Call {
            selector: BUFFER_ESCAPED.to_owned(),
            receiver,
            args: vec![Value::Uint(kind.into())],
            ..Call::default()
        };
        let escapes: Vec<&Call> = (calls.iter())
            .filter(|c| c.selector == BUFFER_ESCAPED)
            .collect();
        assert_eq!(
            escapes,
            [
                &escape(ten, RGFX_BUFFER_ESCAPED_RESOURCE_ID),
                &escape(eleven, RGFX_BUFFER_ESCAPED_RESIDENCY_SET)
            ]
        );
    }
}
