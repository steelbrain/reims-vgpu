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

use std::sync::atomic::{AtomicBool, Ordering};

use reims_vgpu_graphics::wire::RgfxBufferEscaped;
use reims_vgpu_vulkan::answer::count_kind;
use reims_vgpu_vulkan::core::platform::Platform;

use super::tables::Entry;
use super::{DecodeError, GraphicsDecoder, ObjHandle, read};

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
}

impl Escape {
    /// A no-copy buffer made over `over`, whose identity has not escaped, counted among those
    /// made ([`NO_COPY_BUFFERS`]).
    pub(crate) fn made(over: NoCopy) -> Self {
        count_kind(NO_COPY_BUFFERS);
        Escape {
            over,
            escaped: AtomicBool::new(false),
        }
    }

    /// What the buffer lies over.
    #[cfg(test)]
    pub(crate) fn over(&self) -> NoCopy {
        self.over
    }

    /// Whether the guest noted that the buffer's identity escaped.
    #[cfg(test)]
    pub(crate) fn escaped(&self) -> bool {
        self.escaped.load(Ordering::Relaxed)
    }

    /// The guest's note that buffer `obj_id`'s identity escaped by the act at `act` of
    /// [`ESCAPED_BY`]: the first marks it and counts it, and the first of its act in this
    /// process is logged; a later one changes nothing.
    fn note(&self, act: usize, obj_id: u64) {
        if self.escaped.swap(true, Ordering::Relaxed) {
            return;
        }
        count_kind(ESCAPED_BY[act]);
        if self.over == NoCopy::ReadOnlyPages {
            count_kind(ESCAPED_COPIES);
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
                escape: Some(escape),
                ..
            }) => {
                escape.note(act, m.obj_id);
                Ok(())
            }
            _ => Err(DecodeError::BadReference),
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
    use reims_vgpu_graphics::wire::{
        RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER, RGFX_BUFFER_ESCAPED_GPU_ADDRESS,
        RGFX_BUFFER_ESCAPED_RESIDENCY_SET, RGFX_BUFFER_ESCAPED_RESOURCE_ID, RGFX_CMD_BUFFER_CREATE,
        RGFX_CMD_BUFFER_ESCAPED, RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_BUFFER_OVER_PAGES,
        RGFX_STATUS_OK, RGFX_SURFACE_PAGES_PRIVATE, RGFX_SURFACE_PAGES_READ_ONLY, RgfxBufferCreate,
        RgfxBufferOverHostMemory, RgfxBufferOverPages, RgfxMsgHeader,
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
}
