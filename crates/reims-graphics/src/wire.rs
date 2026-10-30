//! The device's wire format, as the guest and the host share it: the PCI identity, the registers
//! of BAR 0, the layout of BAR 2 (the region), the rings' header, and the control ring's commands
//! and events. cbindgen writes it into `include/reims_graphics_wire.h`, the header the guest's
//! kext and option ROM include (`cargo xtask header`).
//!
//! Every value is little-endian, as the guest and every host the library builds for are. A
//! message's struct is the whole message, its header included, so a writer fills in one struct;
//! a `u64` field that would follow a `u32` at an offset that is not a multiple of 8 has a
//! `reserved` field before it, so the structs are laid out the same in C and in Rust, with no
//! padding the compiler adds.
//!
//! **Rings.** A ring is a [`RgfxRingHeader`] and a byte ring of [`RGFX_CONTROL_RING_DATA_SIZE`]
//! bytes after it. `producer` and `consumer` are free-running byte counts: a message is at
//! `count % size` in the byte ring. A message starts with an [`RgfxMsgHeader`], its `size` a
//! multiple of 8 that includes the header, and never wraps: where the bytes left before the
//! ring's end are fewer than the message, the producer pads them with a message of type
//! [`RGFX_MSG_PAD`], whose `size` covers them. The bytes left are a multiple of 8, and may be 8,
//! so a pad's header is its `type` and `size` alone: its `seqno` is not written or read.
//!
//! **Doorbell and interrupts.** The producer of a submission ring (the guest) writes a message,
//! publishes `producer` (a release store), issues a full fence and reads `consumer_idle`; if it
//! reads 1 it writes the ring's id to [`RGFX_REG_DOORBELL`]. The consumer (the host) sets
//! `consumer_idle` before it sleeps, issues a full fence and reads `producer` again, so no wake
//! is lost, and clears it when it wakes. The producer of a completion ring (the host) publishes a
//! message, issues a full fence and reads `armed`; if it reads 1, it clears it and raises the
//! ring's MSI-X vector. The guest sets `armed` after it has drained the ring, issues a full fence
//! and reads `producer` again.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// The device's PCI vendor ID: a placeholder until the owner picks the device's own.
pub const RGFX_PCI_VENDOR_ID: u16 = 0x1234;
/// The device's PCI device ID: a placeholder, as the vendor ID is.
pub const RGFX_PCI_DEVICE_ID: u16 = 0x5247;
/// The device's PCI class: a display controller of class "other" (0x0380).
pub const RGFX_PCI_CLASS: u16 = 0x0380;
/// The BAR of the registers: 4 KiB of MMIO, with the MSI-X table and PBA from
/// [`RGFX_MSIX_OFFSET`].
pub const RGFX_PCI_BAR_REGISTERS: u32 = 0;
/// The BAR of the region: 64-bit and prefetchable, over host memory the guest maps write-back.
pub const RGFX_PCI_BAR_REGION: u32 = 2;
/// The size of BAR 0.
pub const RGFX_REGISTERS_SIZE: u32 = 0x1000;
/// Where in BAR 0 the MSI-X table starts, the PBA after it.
pub const RGFX_MSIX_OFFSET: u32 = 0x800;
/// The device's MSI-X vectors: vector 0 is the control completion ring's.
pub const RGFX_MSIX_VECTORS: u32 = 1;

/// `MAGIC`, RO: [`RGFX_MAGIC`].
pub const RGFX_REG_MAGIC: u32 = 0x000;
/// `VERSION`, RO: [`RGFX_VERSION`].
pub const RGFX_REG_VERSION: u32 = 0x004;
/// `FEATURES_LO`, RO: feature bits 0 to 31, none of them set yet.
pub const RGFX_REG_FEATURES_LO: u32 = 0x008;
/// `FEATURES_HI`, RO: feature bits 32 to 63.
pub const RGFX_REG_FEATURES_HI: u32 = 0x00c;
/// `STATUS`, RW: writing [`RGFX_STATUS_RESET`] resets the device; the guest writes
/// [`RGFX_STATUS_DRIVER_READY`] once it has read the rings' headers.
pub const RGFX_REG_STATUS: u32 = 0x010;
/// `REGION_SIZE_LO`, RO: BAR 2's size, its low 32 bits.
pub const RGFX_REG_REGION_SIZE_LO: u32 = 0x014;
/// `REGION_SIZE_HI`, RO: BAR 2's size, its high 32 bits.
pub const RGFX_REG_REGION_SIZE_HI: u32 = 0x018;
/// `DOORBELL`, WO: writing a ring's id wakes its consumer.
pub const RGFX_REG_DOORBELL: u32 = 0x020;

/// What `MAGIC` reads: "RGFX", little-endian.
pub const RGFX_MAGIC: u32 = 0x5846_4752;
/// What `VERSION` reads.
pub const RGFX_VERSION: u32 = 1;
/// `STATUS`: the device is reset, and takes no message.
pub const RGFX_STATUS_RESET: u32 = 0;
/// `STATUS`: the guest's driver has read the rings' headers; the device takes its messages.
pub const RGFX_STATUS_DRIVER_READY: u32 = 1;

/// Where the control submission ring is in the region: ring id [`RGFX_CONTROL_RING_ID`], guest
/// to host.
pub const RGFX_CONTROL_SUBMIT_OFFSET: u64 = 0x0_0000;
/// Where the control completion ring is in the region: host to guest, MSI-X vector
/// [`RGFX_CONTROL_VECTOR`].
pub const RGFX_CONTROL_COMPLETE_OFFSET: u64 = 0x1_0000;
/// The room each control ring has in the region, its header included.
pub const RGFX_CONTROL_RING_SPAN: u64 = 0x1_0000;
/// Where the guest's allocations start in the region (the scanout buffers and the cursor
/// image): everything from here to the region's end is the guest's to place.
pub const RGFX_ALLOCATIONS_OFFSET: u64 = 0x2_0000;
/// The control submission ring's id, which the guest writes to `DOORBELL`.
pub const RGFX_CONTROL_RING_ID: u32 = 0;
/// The control completion ring's MSI-X vector.
pub const RGFX_CONTROL_VECTOR: u32 = 0;

/// The size of a ring's header ([`RgfxRingHeader`]); its byte ring follows it.
pub const RGFX_RING_HEADER_SIZE: u32 = 256;
/// The size of a control ring's byte ring: the largest power of two that fits in its span after
/// the header, 32 KiB. The rest of the span is unused.
pub const RGFX_CONTROL_RING_DATA_SIZE: u32 = 0x8000;

/// A ring's header, at the start of the ring: each field in a 64-byte line of its own, so the
/// producer's and the consumer's writes never share a cache line.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct RgfxRingHeader {
    /// The producer's free-running byte count, written by the producer.
    pub producer: u64,
    /// The rest of `producer`'s line.
    pub reserved_08: [u8; 56],
    /// The consumer's free-running byte count, written by the consumer.
    pub consumer: u64,
    /// The rest of `consumer`'s line.
    pub reserved_48: [u8; 56],
    /// 1 while the consumer sleeps until a doorbell, written by the consumer.
    pub consumer_idle: u32,
    /// The rest of `consumer_idle`'s line.
    pub reserved_84: [u8; 60],
    /// 1 when the guest wants an interrupt, set by the guest on a completion ring and cleared by
    /// the host as it raises one.
    pub armed: u32,
    /// The rest of `armed`'s line.
    pub reserved_c4: [u8; 60],
}

/// A pad to the ring's end: its `size` covers the bytes left, 8 or more, and it has no `seqno`.
pub const RGFX_MSG_PAD: u32 = 0;
/// Command `GET_DISPLAYS` ([`RgfxGetDisplays`]): answered by [`RGFX_EVT_DISPLAYS`].
pub const RGFX_CMD_GET_DISPLAYS: u32 = 1;
/// Command `SET_SCANOUT` ([`RgfxSetScanout`]): answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_SET_SCANOUT: u32 = 2;
/// Command `SET_CURSOR` ([`RgfxSetCursor`]): answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_SET_CURSOR: u32 = 3;
/// Command `MOVE_CURSOR` ([`RgfxMoveCursor`]): not answered.
pub const RGFX_CMD_MOVE_CURSOR: u32 = 4;
/// Command `SET_VBLANK` ([`RgfxSetVblank`]): answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_SET_VBLANK: u32 = 5;
/// Event `COMPLETE` ([`RgfxComplete`]): a command's answer, with its `seqno`.
pub const RGFX_EVT_COMPLETE: u32 = 0x100;
/// Event `DISPLAYS` ([`RgfxDisplays`]): `GET_DISPLAYS`'s answer, with its `seqno`.
pub const RGFX_EVT_DISPLAYS: u32 = 0x101;
/// Event `DISPLAY_CHANGED` ([`RgfxDisplayChanged`]): the guest asks `GET_DISPLAYS` again.
pub const RGFX_EVT_DISPLAY_CHANGED: u32 = 0x102;
/// Event `VBLANK` ([`RgfxVblank`]): a refresh of the display's host screen.
pub const RGFX_EVT_VBLANK: u32 = 0x103;

/// `COMPLETE`'s status: the command was done.
pub const RGFX_STATUS_OK: u32 = 0;
/// `COMPLETE`'s status: the device has no command of that type; nothing was done.
pub const RGFX_STATUS_UNKNOWN_COMMAND: u32 = 1;
/// `COMPLETE`'s status: the device has no display of that id; nothing was done.
pub const RGFX_STATUS_NO_DISPLAY: u32 = 2;
/// `COMPLETE`'s status: an argument is out of range (an image that does not lie inside the
/// guest's allocations, a stride or a size the device refuses, a message shorter than its
/// command); nothing was done.
pub const RGFX_STATUS_INVALID_ARGUMENT: u32 = 3;

/// Scanout pixels: 32-bit BGRA8, `kIO32BGRAPixelFormat`, the one format there is.
pub const RGFX_SCANOUT_BYTES_PER_PIXEL: u32 = 4;
/// The largest cursor image, in pixels each way.
pub const RGFX_CURSOR_MAX_SIZE: u32 = 256;
/// The displays the device has: display 0.
pub const RGFX_DISPLAY_COUNT: u32 = 1;

/// Every message's header.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxMsgHeader {
    /// The message's type: an `RGFX_CMD_*`, an `RGFX_EVT_*` or [`RGFX_MSG_PAD`].
    pub r#type: u32,
    /// The message's bytes, this header included: a multiple of 8.
    pub size: u32,
    /// A command's number, from 1 up on each ring, which the guest picks; an answer carries its
    /// command's, and an event that answers nothing carries 0.
    pub seqno: u64,
}

/// `GET_DISPLAYS`: no payload.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxGetDisplays {
    /// [`RGFX_CMD_GET_DISPLAYS`].
    pub header: RgfxMsgHeader,
}

/// `SET_SCANOUT`: the display shows `height` rows of `width` BGRA8 pixels, each row `stride`
/// bytes after the one before, from `offset` in the region. A `width` of 0 blanks the display.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSetScanout {
    /// [`RGFX_CMD_SET_SCANOUT`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Pixels a row; 0 blanks the display.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Bytes from one row's start to the next's: a multiple of 4, at least `width * 4`.
    pub stride: u32,
    /// Where the first row starts in the region: a multiple of 4, in the guest's allocations.
    pub offset: u64,
}

/// `SET_CURSOR`: the display's cursor is `height` rows of `width` premultiplied BGRA8 pixels,
/// `width * 4` bytes a row, from `offset` in the region, its hot spot at (`hot_x`, `hot_y`). A
/// `width` of 0 removes the image.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSetCursor {
    /// [`RGFX_CMD_SET_CURSOR`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Pixels a row, up to [`RGFX_CURSOR_MAX_SIZE`]; 0 removes the image.
    pub width: u32,
    /// Rows, up to [`RGFX_CURSOR_MAX_SIZE`].
    pub height: u32,
    /// The hot spot's column, below `width`.
    pub hot_x: u32,
    /// The hot spot's row, below `height`.
    pub hot_y: u32,
    /// Keeps `offset` 8-aligned; 0.
    pub reserved_24: u32,
    /// Where the image starts in the region: a multiple of 4, in the guest's allocations.
    pub offset: u64,
}

/// `MOVE_CURSOR`: the cursor's hot spot is at (`x`, `y`) in the display's pixels, and shows
/// where `visible` is not 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxMoveCursor {
    /// [`RGFX_CMD_MOVE_CURSOR`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// The hot spot's column.
    pub x: i32,
    /// The hot spot's row.
    pub y: i32,
    /// Whether the cursor shows: 0 hides it.
    pub visible: u32,
}

/// `SET_VBLANK`: the device sends `VBLANK` at each refresh of the display's host screen while
/// `enabled` is not 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSetVblank {
    /// [`RGFX_CMD_SET_VBLANK`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Whether `VBLANK` is sent: 0 stops it.
    pub enabled: u32,
}

/// `COMPLETE`: a command's answer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComplete {
    /// [`RGFX_EVT_COMPLETE`], with the command's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`: 0 for success.
    pub status: u32,
    /// Pads the message to a multiple of 8; 0.
    pub reserved_14: u32,
}

/// `DISPLAYS`: `GET_DISPLAYS`'s answer. `count` [`RgfxDisplayEntry`] records follow it, at
/// [`RGFX_DISPLAYS_ENTRIES_OFFSET`], each followed by its `mode_count` [`RgfxModeEntry`] records;
/// the message's `size` is rounded up to a multiple of 8 after the last. The struct is the
/// message's start: C's `sizeof` pads it past `count`, where the first record already is.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RgfxDisplays {
    /// [`RGFX_EVT_DISPLAYS`], with `GET_DISPLAYS`' `seqno`.
    pub header: RgfxMsgHeader,
    /// The displays that follow.
    pub count: u32,
}

/// Where a `DISPLAYS` message's first display record starts: right after `count`.
pub const RGFX_DISPLAYS_ENTRIES_OFFSET: u32 = 20;

/// One display in a `DISPLAYS` message.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDisplayEntry {
    /// The display's id, which the commands name.
    pub id: u32,
    /// 1 where a display is connected, else 0.
    pub connected: u32,
    /// The index of the mode the display prefers, among the modes that follow.
    pub preferred_mode: u32,
    /// The modes that follow this record.
    pub mode_count: u32,
}

/// One mode of a display in a `DISPLAYS` message.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxModeEntry {
    /// Pixels a row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// The refresh rate, in thousandths of a hertz.
    pub refresh_mhz: u32,
}

/// `DISPLAY_CHANGED`: what the display offers has changed; the guest asks `GET_DISPLAYS` again.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDisplayChanged {
    /// [`RGFX_EVT_DISPLAY_CHANGED`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Pads the message to a multiple of 8; 0.
    pub reserved_14: u32,
}

/// `VBLANK`: the display's host screen refreshed.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVblank {
    /// [`RGFX_EVT_VBLANK`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Keeps `count` 8-aligned; 0.
    pub reserved_14: u32,
    /// The display's refreshes since it was made, this one included.
    pub count: u64,
    /// The host's monotonic clock at the refresh, in nanoseconds.
    pub host_time_ns: u64,
}

// The layout the tables of the design give.
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<RgfxRingHeader>() == RGFX_RING_HEADER_SIZE as usize);
    assert!(offset_of!(RgfxRingHeader, consumer) == 0x40);
    assert!(offset_of!(RgfxRingHeader, consumer_idle) == 0x80);
    assert!(offset_of!(RgfxRingHeader, armed) == 0xc0);
    assert!(size_of::<RgfxMsgHeader>() == 16);
    assert!(size_of::<RgfxGetDisplays>() == 16);
    assert!(size_of::<RgfxSetScanout>() == 40);
    assert!(size_of::<RgfxSetCursor>() == 48);
    assert!(size_of::<RgfxMoveCursor>() == 32);
    assert!(size_of::<RgfxSetVblank>() == 24);
    assert!(size_of::<RgfxComplete>() == 24);
    assert!(offset_of!(RgfxDisplays, count) + 4 == RGFX_DISPLAYS_ENTRIES_OFFSET as usize);
    assert!(size_of::<RgfxDisplayEntry>() == 16);
    assert!(size_of::<RgfxModeEntry>() == 12);
    assert!(size_of::<RgfxDisplayChanged>() == 24);
    assert!(size_of::<RgfxVblank>() == 40);
    assert!(RGFX_CONTROL_RING_DATA_SIZE.is_power_of_two());
    assert!(
        RGFX_RING_HEADER_SIZE as u64 + RGFX_CONTROL_RING_DATA_SIZE as u64 <= RGFX_CONTROL_RING_SPAN
    );
    assert!(RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN <= RGFX_ALLOCATIONS_OFFSET);
};
