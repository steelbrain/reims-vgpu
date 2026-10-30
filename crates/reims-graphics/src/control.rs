//! The control commands (`wire`'s `RGFX_CMD_*`), as the control thread takes them: each checked,
//! done, and answered with a `COMPLETE` carrying its `seqno` and a status, or with `DISPLAYS`
//! for `GET_DISPLAYS`; `MOVE_CURSOR` is not answered. A command that is refused changes nothing.
//! A message of a type the device has no command for is answered
//! [`RGFX_STATUS_UNKNOWN_COMMAND`], and one shorter than its command
//! [`RGFX_STATUS_INVALID_ARGUMENT`].

use zerocopy::{FromBytes, IntoBytes};

use crate::device::Shared;
use crate::display::{CursorImage, Scanout, straight_rgba};
use crate::ring::{Message, set_size, sized};
use crate::wire::*;

/// What answers `message`: the bytes of its answer, or `None` for a command that has none.
pub(crate) fn answer(shared: &Shared, message: &Message) -> Option<Vec<u8>> {
    let seqno = message.header.seqno;
    let status = match message.header.r#type {
        RGFX_CMD_GET_DISPLAYS => return Some(displays(shared, seqno)),
        RGFX_CMD_SET_SCANOUT => read(message).map_or(RGFX_STATUS_INVALID_ARGUMENT, |command| {
            set_scanout(shared, &command)
        }),
        RGFX_CMD_SET_CURSOR => read(message).map_or(RGFX_STATUS_INVALID_ARGUMENT, |command| {
            set_cursor(shared, &command)
        }),
        RGFX_CMD_MOVE_CURSOR => {
            if let Some(command) = read::<RgfxMoveCursor>(message)
                && command.display == 0
            {
                shared
                    .screen
                    .cursor_moved(command.x, command.y, command.visible != 0);
            }
            return None;
        }
        RGFX_CMD_SET_VBLANK => read(message).map_or(RGFX_STATUS_INVALID_ARGUMENT, |command| {
            set_vblank(shared, &command)
        }),
        RGFX_CMD_PRESENT_SURFACE => read(message)
            .map_or(RGFX_STATUS_INVALID_ARGUMENT, |command| {
                present_surface(shared, &command)
            }),
        // Stage 3 (every type at or above `QUEUE_CREATE`): the bundle rides the device ring now,
        // but a stage-3 message that reaches the control ring is still handled, on the device ring's
        // terms. Its replies — a pipeline reflection, a command buffer's completion — come back on
        // the device completion ring through the device's poster, not from here, so there is no
        // answer to post synchronously.
        t if t >= RGFX_CMD_QUEUE_CREATE => {
            device_message(shared, message);
            return None;
        }
        _ => RGFX_STATUS_UNKNOWN_COMMAND,
    };
    Some(complete(seqno, status))
}

/// Dispatches one stage-3 message taken from the device submission ring (the Metal bundle's own):
/// the object-lifecycle creates and destroy, the pipeline creates and any `COMMAND_BUFFER` not on a
/// per-queue ring. Every message goes to the host's decoder; a `QUEUE_CREATE` also makes the queue's
/// own submission+completion ring pair and service thread (where its ring parameters are valid), and
/// an `OBJECT_DESTROY` of a command queue ends that queue's per-queue service. The decoder posts any
/// reply it owes on the device completion ring itself, so nothing is posted synchronously here.
pub(crate) fn device_message(shared: &Shared, message: &Message) {
    match message.header.r#type {
        // The host's decoder makes the queue handle first (so a command buffer can name it), then
        // the device makes the queue's own ring pair and service thread.
        RGFX_CMD_QUEUE_CREATE => {
            shared.host.stage3_message(&message.bytes);
            if let Some(command) = read::<RgfxQueueCreate>(message) {
                shared.create_queue(&command);
            }
        }
        // The host's decoder drops the handle; where the object is a command queue, the device ends
        // its per-queue service too.
        RGFX_CMD_OBJECT_DESTROY => {
            shared.host.stage3_message(&message.bytes);
            if let Some(command) = read::<RgfxObjectDestroy>(message) {
                shared.destroy_queue(command.obj_id);
            }
        }
        _ => {
            shared.host.stage3_message(&message.bytes);
        }
    }
}

/// `message`'s bytes as command `T`, or `None` where they are fewer than it.
fn read<T: FromBytes>(message: &Message) -> Option<T> {
    T::read_from_prefix(&message.bytes).ok().map(|(t, _)| t)
}

/// A `COMPLETE` for command `seqno`.
pub(crate) fn complete(seqno: u64, status: u32) -> Vec<u8> {
    sized(&RgfxComplete {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_COMPLETE,
            size: 0,
            seqno,
        },
        status,
        reserved_14: 0,
    })
}

/// `GET_DISPLAYS`' answer: display 0, connected, with the screen's mode as its one mode, which
/// it prefers. The guest is told that mode from now on (`device`'s refresh).
fn displays(shared: &Shared, seqno: u64) -> Vec<u8> {
    let mode = shared.screen.mode();
    shared.display.lock().told = Some(mode);
    let mut bytes = RgfxMsgHeader {
        r#type: RGFX_EVT_DISPLAYS,
        size: 0,
        seqno,
    }
    .as_bytes()
    .to_vec();
    bytes.extend_from_slice(&RGFX_DISPLAY_COUNT.to_le_bytes());
    let entry = RgfxDisplayEntry {
        id: 0,
        connected: 1,
        preferred_mode: 0,
        mode_count: 1,
    };
    bytes.extend_from_slice(entry.as_bytes());
    let mode = RgfxModeEntry {
        width: mode.width,
        height: mode.height,
        refresh_mhz: mode.refresh_mhz,
    };
    bytes.extend_from_slice(mode.as_bytes());
    bytes.resize(bytes.len().next_multiple_of(8), 0);
    set_size(&mut bytes);
    bytes
}

/// Whether `len` bytes at `offset` lie in the guest's allocations, `offset` a multiple of 4.
fn in_allocations(shared: &Shared, offset: u64, len: u64) -> bool {
    offset.is_multiple_of(4)
        && offset >= RGFX_ALLOCATIONS_OFFSET
        && shared.region.contains(offset, len)
}

/// `SET_SCANOUT`.
fn set_scanout(shared: &Shared, command: &RgfxSetScanout) -> u32 {
    if command.display != 0 {
        return RGFX_STATUS_NO_DISPLAY;
    }
    if command.width == 0 {
        shared.display.lock().scanout = None;
        shared.screen.scanout(None);
        return RGFX_STATUS_OK;
    }
    let row = u64::from(command.width) * u64::from(RGFX_SCANOUT_BYTES_PER_PIXEL);
    if command.height == 0 || !command.stride.is_multiple_of(4) || u64::from(command.stride) < row {
        return RGFX_STATUS_INVALID_ARGUMENT;
    }
    let scanout = Scanout {
        width: command.width,
        height: command.height,
        stride: command.stride,
        offset: command.offset,
    };
    if !in_allocations(shared, scanout.offset, scanout.span()) {
        return RGFX_STATUS_INVALID_ARGUMENT;
    }
    shared.display.lock().scanout = Some(scanout);
    shared.screen.scanout(Some(scanout));
    RGFX_STATUS_OK
}

/// `SET_CURSOR`: the image is copied out of the region now; the guest may reuse its bytes once
/// it has the answer.
fn set_cursor(shared: &Shared, command: &RgfxSetCursor) -> u32 {
    if command.display != 0 {
        return RGFX_STATUS_NO_DISPLAY;
    }
    if command.width == 0 {
        shared.screen.cursor_image(None);
        return RGFX_STATUS_OK;
    }
    let (width, height) = (command.width, command.height);
    if width > RGFX_CURSOR_MAX_SIZE
        || height == 0
        || height > RGFX_CURSOR_MAX_SIZE
        || command.hot_x >= width
        || command.hot_y >= height
    {
        return RGFX_STATUS_INVALID_ARGUMENT;
    }
    let len = u64::from(width) * u64::from(height) * 4;
    if !in_allocations(shared, command.offset, len) {
        return RGFX_STATUS_INVALID_ARGUMENT;
    }
    let Some(bgra) = shared.region.read(command.offset, len as usize) else {
        return RGFX_STATUS_INVALID_ARGUMENT;
    };
    shared.screen.cursor_image(Some(CursorImage {
        width,
        height,
        hot_spot: (command.hot_x, command.hot_y),
        rgba: straight_rgba(&bgra),
    }));
    RGFX_STATUS_OK
}

/// `PRESENT_SURFACE`: display 0 only, and a surface the host holds a texture over.
fn present_surface(shared: &Shared, command: &RgfxPresentSurface) -> u32 {
    if command.display != 0 {
        return RGFX_STATUS_NO_DISPLAY;
    }
    if shared.host.present_surface(command.iosurface_id) {
        RGFX_STATUS_OK
    } else {
        log::debug!(
            target: "reims_graphics",
            "PRESENT_SURFACE of transaction {}: no texture over IOSurface {}",
            command.transaction_id,
            command.iosurface_id
        );
        RGFX_STATUS_INVALID_ARGUMENT
    }
}

/// `SET_VBLANK`.
fn set_vblank(shared: &Shared, command: &RgfxSetVblank) -> u32 {
    if command.display != 0 {
        return RGFX_STATUS_NO_DISPLAY;
    }
    shared.display.lock().vblank = command.enabled != 0;
    RGFX_STATUS_OK
}
