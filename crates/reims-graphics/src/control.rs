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

/// What answers `message`, a command carried out with the ring held (any but those [`is_unheld`]
/// names): the bytes of its answer, or `None` for a command that has none.
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
        // A command buffer rides a command queue's own ring, never the control ring.
        _ => RGFX_STATUS_UNKNOWN_COMMAND,
    };
    Some(complete(seqno, status))
}

/// Whether `message` is carried out with no ring held ([`unheld`]): the object lifecycle the
/// accelerator kext posts on ring 0 (every stage-3 type from `QUEUE_CREATE` below
/// `COMMAND_BUFFER`), and `PRESENT_SURFACE`, which the host may answer at once, raising the
/// interrupt, or carry out under locks of its own.
pub(crate) fn is_unheld(message: &Message) -> bool {
    (RGFX_CMD_QUEUE_CREATE..RGFX_CMD_COMMAND_BUFFER).contains(&message.header.r#type)
        || message.header.r#type == RGFX_CMD_PRESENT_SURFACE
}

/// Carries out one message [`is_unheld`] names, with no ring held, taken at reset count `epoch`:
/// what answers it, or `None` for one the host answers itself or not at all. The answers the host
/// gives are at `epoch`, so they are dropped where a reset came after the message was taken.
pub(crate) fn unheld(shared: &Shared, message: &Message, epoch: u64) -> Option<Vec<u8>> {
    if message.header.r#type == RGFX_CMD_PRESENT_SURFACE {
        present(shared, message, epoch)
    } else {
        lifecycle(shared, message, epoch)
    }
}

/// Carries out one object-lifecycle message from ring 0, with no ring held: what answers it, or
/// `None` for one the host answers itself (a pipeline's reflection, through the control poster)
/// or not at all. A `QUEUE_CREATE` makes the queue's own ring pair and service thread, answered
/// with whether its flags and rings were taken; an `OBJECT_DESTROY` of a command queue ends the queue's
/// service, which drops its handle; a `CONTEXT_CLOSE` ends the context's queues and has the host
/// drop its objects, and the host answers it once the context's command buffers have finished; a
/// `BLOCK_RELEASE` is refused where a live queue's ring lies in the block, and else the host answers
/// it; a `BLOCK_GRANT` is kept with the region (`Region::grant`), and so are a `BLOCK_SHARE` and a
/// `BLOCK_UNSHARE` (`Region::share`, `Region::unshare`); a `SURFACE_PAGES` hands its block's runs
/// to the host (`Shared::surface_pages`), which answers with whether it mapped them; a
/// `SURFACE_DOWNLOAD` the host answers once the surface's pages hold its device cache
/// (`Host::surface_download`). Every other message goes to the host's decoder.
fn lifecycle(shared: &Shared, message: &Message, epoch: u64) -> Option<Vec<u8>> {
    let seqno = message.header.seqno;
    match message.header.r#type {
        RGFX_CMD_QUEUE_CREATE => {
            let flags = read_at::<RgfxQueueCreateFlags>(message, RGFX_QUEUE_CREATE_FLAGS_OFFSET)
                .map_or(0, |tail| tail.flags);
            let status = read::<RgfxQueueCreate>(message)
                .map_or(RGFX_STATUS_INVALID_ARGUMENT, |command| {
                    shared.create_queue(&command, flags, epoch)
                });
            Some(complete(seqno, status))
        }
        RGFX_CMD_OBJECT_DESTROY => {
            let queue = read::<RgfxObjectDestroy>(message)
                .is_some_and(|command| shared.destroy_queue(command.obj_id));
            if !queue {
                shared
                    .host
                    .stage3_message(&shared.control_poster(epoch), &message.bytes);
            }
            None
        }
        RGFX_CMD_CONTEXT_CLOSE => match read::<RgfxContextClose>(message) {
            Some(command) => {
                shared.close_context(command.context, shared.complete_answer(seqno, epoch));
                None
            }
            None => Some(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT)),
        },
        RGFX_CMD_BLOCK_GRANT => {
            if let Some(grant) = read::<RgfxBlockGrant>(message) {
                shared.grant_block(&grant);
            }
            None
        }
        RGFX_CMD_BLOCK_SHARE | RGFX_CMD_BLOCK_UNSHARE => {
            if let Some(share) = read::<RgfxBlockGrant>(message) {
                shared.share_block(&share, message.header.r#type == RGFX_CMD_BLOCK_SHARE);
            }
            None
        }
        RGFX_CMD_SURFACE_PAGES => {
            let flags = read_at::<RgfxSurfacePagesCache>(message, RGFX_SURFACE_PAGES_CACHE_OFFSET)
                .map_or(0, |c| c.flags);
            let status = read::<RgfxSurfacePages>(message)
                .map_or(Some(RGFX_STATUS_INVALID_ARGUMENT), |pages| {
                    shared.surface_pages(&pages, flags, seqno, epoch)
                });
            status.map(|status| complete(seqno, status))
        }
        RGFX_CMD_SURFACE_DOWNLOAD => match read::<RgfxSurfaceDownload>(message) {
            Some(download) => {
                shared
                    .host
                    .surface_download(download.block_offset, shared.complete_answer(seqno, epoch));
                None
            }
            None => Some(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT)),
        },
        RGFX_CMD_BLOCK_RELEASE => match read::<RgfxBlockRelease>(message) {
            // Whatever the answer, the block is no longer its process's to name.
            Some(release) => {
                let holders = shared.region.revoke(release.offset);
                if shared.rings_in(release.offset, release.size) {
                    Some(complete(seqno, RGFX_STATUS_REFUSED))
                } else {
                    // Another holder of a shared block whose BLOCK_UNSHARE has not come is let go
                    // of here as by it, so the release waits for that context's command buffers
                    // too: the kext posts an unshare out of its lock, so another holder's last
                    // let-go may post the block's release first, and gives the block's memory
                    // back at this answer.
                    for context in holders.into_iter().filter(|&c| c != release.context) {
                        shared.host.block_unshared(&RgfxBlockGrant {
                            header: RgfxMsgHeader {
                                r#type: RGFX_CMD_BLOCK_UNSHARE,
                                size: size_of::<RgfxBlockGrant>() as u32,
                                seqno: 0,
                            },
                            context,
                            offset: release.offset,
                            size: release.size,
                            ..RgfxBlockGrant::default()
                        });
                    }
                    shared
                        .host
                        .block_released(&release, shared.complete_answer(seqno, epoch));
                    None
                }
            }
            None => Some(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT)),
        },
        _ => {
            shared
                .host
                .stage3_message(&shared.control_poster(epoch), &message.bytes);
            None
        }
    }
}

/// `message`'s bytes as command `T`, or `None` where they are fewer than it.
fn read<T: FromBytes>(message: &Message) -> Option<T> {
    T::read_from_prefix(&message.bytes).ok().map(|(t, _)| t)
}

/// The `T` that follows a command's own struct at byte `offset` of `message`, or `None` where
/// the message ends before it (a message that does not carry it).
fn read_at<T: FromBytes>(message: &Message, offset: u32) -> Option<T> {
    let bytes = message.bytes.get(offset as usize..)?;
    T::read_from_prefix(bytes).ok().map(|(t, _)| t)
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

/// Whether `len` bytes at `offset` lie in the scanout's room: the [`RGFX_SCANOUT_SPAN`] bytes at
/// [`RGFX_ALLOCATIONS_OFFSET`], before the accelerator kext's blocks
/// ([`RGFX_CONTEXT_ALLOCATIONS_OFFSET`]), `offset` a multiple of 4. A scanout past the room would
/// show the processes' blocks on the display.
fn in_scanout_room(shared: &Shared, offset: u64, len: u64) -> bool {
    offset.is_multiple_of(4)
        && offset >= RGFX_ALLOCATIONS_OFFSET
        && len <= RGFX_SCANOUT_SPAN
        && offset - RGFX_ALLOCATIONS_OFFSET <= RGFX_SCANOUT_SPAN - len
        && shared.region.contains(offset, len)
}

/// `SET_SCANOUT`: the scanout lies in its room ([`in_scanout_room`]).
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
    if !in_scanout_room(shared, scanout.offset, scanout.span()) {
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

/// `PRESENT_SURFACE`, with no ring held, taken at reset count `epoch`: what answers it where it is
/// refused here, else `None`.
fn present(shared: &Shared, message: &Message, epoch: u64) -> Option<Vec<u8>> {
    let seqno = message.header.seqno;
    let status = match read::<RgfxPresentSurface>(message) {
        Some(command) if command.display == 0 => {
            let generation =
                read_at::<RgfxPresentSurfaceGen>(message, RGFX_PRESENT_SURFACE_GEN_OFFSET)
                    .map(|g| g.cpu_gen);
            present_surface(shared, &command, generation, seqno, epoch);
            return None;
        }
        Some(_) => RGFX_STATUS_NO_DISPLAY,
        None => RGFX_STATUS_INVALID_ARGUMENT,
    };
    Some(complete(seqno, status))
}

/// `PRESENT_SURFACE` on display 0: the host answers once it shows the surface, or refuses it
/// (`Host::present_surface`), at once or from any thread.
fn present_surface(
    shared: &Shared,
    command: &RgfxPresentSurface,
    generation: Option<u64>,
    seqno: u64,
    epoch: u64,
) {
    log::trace!(
        target: "reims_graphics",
        "PRESENT_SURFACE of transaction {}: IOSurface {}",
        command.transaction_id,
        command.iosurface_id
    );
    shared.host.present_surface(
        command.iosurface_id,
        generation,
        shared.complete_answer(seqno, epoch),
    );
}

/// `SET_VBLANK`.
fn set_vblank(shared: &Shared, command: &RgfxSetVblank) -> u32 {
    if command.display != 0 {
        return RGFX_STATUS_NO_DISPLAY;
    }
    shared.display.lock().vblank = command.enabled != 0;
    RGFX_STATUS_OK
}
