//! The control commands (`wire`'s `RGFX_CMD_*`), as the control thread takes them: each checked,
//! done, and answered with a `COMPLETE` carrying its `seqno` and a status, or with `DISPLAYS`
//! for `GET_DISPLAYS`; `MOVE_CURSOR`, `DISPLAY_BOUNDS`, `CURSOR_COLOR` and `SCROLL_DEVICES` are
//! not answered. A command that is refused changes nothing. A message of a type the device has no command for is
//! answered [`RGFX_STATUS_UNKNOWN_COMMAND`], and one shorter than its command
//! [`RGFX_STATUS_INVALID_ARGUMENT`]. A display command naming a display the guest does not see
//! (`Shared::seen_display`: past the device's count, or past display 0 where the guest did not
//! take [`RGFX_FEATURE_MULTI_DISPLAY`]) is refused [`RGFX_STATUS_NO_DISPLAY`].

use zerocopy::{FromBytes, IntoBytes};

use crate::device::{CursorShape, Shared};
use crate::display::cursor_color::CursorColor;
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
                && let Some(display) = shared.seen_display(command.display)
            {
                // The device knows first, so a window the move reaches asks it of a cursor
                // shown or hidden already.
                shared.cursor_visible(command.display, command.visible != 0);
                display
                    .screen
                    .cursor_moved(command.x, command.y, command.visible != 0);
                if command.visible != 0 {
                    shared.cursor_shown_on(command.display, command.x, command.y);
                }
            }
            return None;
        }
        RGFX_CMD_DISPLAY_BOUNDS => {
            if let Some(command) = read::<RgfxDisplayBounds>(message) {
                shared.display_bounds(&command);
            }
            return None;
        }
        RGFX_CMD_CURSOR_COLOR => {
            cursor_color(shared, message);
            return None;
        }
        RGFX_CMD_SCROLL_DEVICES => {
            if let Some(command) = read::<RgfxScrollDevices>(message) {
                shared.scroll_devices(command.devices);
            }
            return None;
        }
        RGFX_CMD_SET_VBLANK => read(message).map_or(RGFX_STATUS_INVALID_ARGUMENT, |command| {
            set_vblank(shared, &command)
        }),
        RGFX_CMD_SET_DISPLAY_POWER => read(message)
            .map_or(RGFX_STATUS_INVALID_ARGUMENT, |command| {
                set_display_power(shared, &command)
            }),
        // A command buffer rides a command queue's own ring, never the control ring.
        _ => RGFX_STATUS_UNKNOWN_COMMAND,
    };
    Some(complete(seqno, status))
}

/// Whether `message` is carried out with no ring held ([`unheld`]): the object lifecycle the
/// accelerator kext posts on ring 0 (every stage-3 type from `QUEUE_CREATE` below
/// `COMMAND_BUFFER`), and `PRESENT_SURFACE`, which the host may answer at once, raising the
/// interrupt, or carry out under locks of its own, with the `DISPLAY_COLOR`s that stage its
/// colour, which ride the presents' lane with it.
pub(crate) fn is_unheld(message: &Message) -> bool {
    (RGFX_CMD_QUEUE_CREATE..RGFX_CMD_COMMAND_BUFFER).contains(&message.header.r#type)
        || message.header.r#type == RGFX_CMD_PRESENT_SURFACE
        || message.header.r#type == RGFX_CMD_DISPLAY_COLOR
}

/// Carries out one message [`is_unheld`] names, with no ring held, taken at reset count `epoch`:
/// what answers it, or `None` for one the host answers itself or not at all. The answers the host
/// gives are at `epoch`, so they are dropped where a reset came after the message was taken.
pub(crate) fn unheld(shared: &Shared, message: &Message, epoch: u64) -> Option<Vec<u8>> {
    match message.header.r#type {
        RGFX_CMD_PRESENT_SURFACE => present(shared, message, epoch),
        RGFX_CMD_DISPLAY_COLOR => {
            display_color(shared, message);
            None
        }
        _ => lifecycle(shared, message, epoch),
    }
}

/// Carries out one object-lifecycle message from ring 0, with no ring held: what answers it, or
/// `None` for one the host answers itself (a pipeline's reflection, through the control poster)
/// or not at all. A `QUEUE_CREATE` makes the queue's own ring pair and service thread, answered
/// with whether its flags and rings were taken; an `OBJECT_DESTROY` of a command queue ends the queue's
/// service, which drops its handle; a `CONTEXT_CLOSE` ends the context's queues and has the host
/// drop its objects, and the host answers it once the context's command buffers have finished; a
/// `BLOCK_RELEASE` is refused where a live queue's ring lies in the block, and else the host answers
/// it; a `POOL_TAKE` and a `POOL_GIVE` go to the host, which answers them
/// (`Host::pool_take`, `Host::pool_give`), where the guest took `POOL`, and else to its decoder as
/// any message it does not know, and so do a `PRIVATE_TAKE` and a `PRIVATE_GIVE`
/// (`Host::private_take`, `Host::private_give`), where the guest took `PRIVATE_BLOCKS`; a `BLOCK_GRANT` is kept with the region (`Region::grant`), and so are a `BLOCK_SHARE` and a
/// `BLOCK_UNSHARE` (`Region::share`, `Region::unshare`); a `SURFACE_PAGES` hands its block's runs
/// to the host (`Shared::surface_pages`), which answers with whether it mapped them; a
/// `SURFACE_DOWNLOAD` the host answers once the surface's pages hold its device cache
/// (`Host::surface_download`); a `VT_MESSAGE`, its runs and length checked, goes to the host's
/// side of VideoToolbox's protocol, which answers it (`Host::vt_message`). Every other message goes
/// to the host's decoder.
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
                let answer = shared.complete_answer(seqno, epoch, RGFX_CMD_CONTEXT_CLOSE);
                shared.close_context(command.context, answer);
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
            // A block that asks whether it needs a device cache carries its surface's layout; a
            // cached one may, a message long enough for it.
            let layout =
                read_at::<RgfxSurfacePagesLayout>(message, RGFX_SURFACE_PAGES_LAYOUT_OFFSET);
            let ask = flags & RGFX_SURFACE_PAGES_ASK_CACHE != 0;
            let cached = flags & RGFX_SURFACE_PAGES_CACHED != 0;
            let status = match (read::<RgfxSurfacePages>(message), ask, layout) {
                (None, _, _) | (_, true, None) => Some(RGFX_STATUS_INVALID_ARGUMENT),
                (Some(pages), ask, layout) => {
                    let layout = layout.filter(|_| ask || cached);
                    shared.surface_pages(&pages, flags, layout, seqno, epoch)
                }
            };
            // A pages block's answer carries its address where the guest took them, a refusal's
            // too, so every answer to such a block reads alike.
            let pages = flags & RGFX_SURFACE_PAGES_PRIVATE != 0
                && shared.driver_took(RGFX_FEATURE_IMPORT_ADDRESSES);
            status.map(|status| {
                if pages {
                    pages_answer(seqno, status, 0)
                } else {
                    complete(seqno, status)
                }
            })
        }
        RGFX_CMD_VT_MESSAGE => {
            // A guest that took no VT_MESSAGES sends none; one it sends anyway is answered as
            // one the device could not read, as is one whose runs or length do not hold.
            let read = shared
                .driver_took(RGFX_FEATURE_VT_MESSAGES)
                .then(|| crate::device::vt_read(message))
                .flatten();
            match read {
                Some(vt) => {
                    shared.host.vt_message(
                        vt,
                        shared.vt_answer(seqno, epoch),
                        shared.vt_link(epoch),
                    );
                    None
                }
                None => Some(crate::device::vt_status(seqno, RGFX_VT_STATUS_MALFORMED)),
            }
        }
        RGFX_CMD_SURFACE_DOWNLOAD => match read::<RgfxSurfaceDownload>(message) {
            Some(download) => {
                let answer = shared.complete_answer(seqno, epoch, RGFX_CMD_SURFACE_DOWNLOAD);
                shared.host.surface_download(download.block_offset, answer);
                None
            }
            None => Some(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT)),
        },
        RGFX_CMD_POOL_TAKE if shared.driver_took(RGFX_FEATURE_POOL) => {
            match read::<RgfxPoolTake>(message) {
                Some(take) => {
                    shared
                        .host
                        .pool_take(&take, shared.pool_answer(seqno, epoch));
                    None
                }
                None => Some(pool_block(seqno, RGFX_STATUS_INVALID_ARGUMENT, 0, 0, 0)),
            }
        }
        RGFX_CMD_POOL_GIVE if shared.driver_took(RGFX_FEATURE_POOL) => {
            match read::<RgfxPoolGive>(message) {
                Some(give) => {
                    let answer = shared.complete_answer(seqno, epoch, RGFX_CMD_POOL_GIVE);
                    shared.host.pool_give(&give, answer);
                    None
                }
                None => Some(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT)),
            }
        }
        RGFX_CMD_PRIVATE_TAKE if shared.driver_took(RGFX_FEATURE_PRIVATE_BLOCKS) => {
            match read::<RgfxPrivateTake>(message) {
                Some(take) => {
                    shared
                        .host
                        .private_take(&take, shared.private_answer(seqno, epoch));
                    None
                }
                None => Some(private_block(seqno, RGFX_STATUS_INVALID_ARGUMENT, 0, 0)),
            }
        }
        RGFX_CMD_PRIVATE_GIVE if shared.driver_took(RGFX_FEATURE_PRIVATE_BLOCKS) => {
            match read::<RgfxPrivateGive>(message) {
                Some(give) => {
                    let answer = shared.complete_answer(seqno, epoch, RGFX_CMD_PRIVATE_GIVE);
                    shared.host.private_give(&give, answer);
                    None
                }
                None => Some(complete(seqno, RGFX_STATUS_INVALID_ARGUMENT)),
            }
        }
        RGFX_CMD_BLOCK_RELEASE => match read::<RgfxBlockRelease>(message) {
            // Whatever the answer, the block is no longer its process's to name.
            Some(release) => {
                let holders = shared.region.revoke(release.offset);
                if shared.rings_in(release.offset, release.size) {
                    Some(complete(seqno, RGFX_STATUS_REFUSED))
                } else {
                    // Another holder of a shared block whose BLOCK_UNSHARE has not come is let go
                    // of here as by it, so the release waits for that context's command buffers
                    // too, as the kext gives the block's memory back at this answer. The kext
                    // posts each hold's unshare before the block's release, which waits for the
                    // holders' lanes, so with it none is left here but the releasing context.
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
                    let answer = shared.complete_answer(seqno, epoch, RGFX_CMD_BLOCK_RELEASE);
                    shared.host.block_released(&release, answer);
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

/// A `POOL_BLOCK` for `POOL_TAKE` `seqno`.
pub(crate) fn pool_block(
    seqno: u64,
    status: u32,
    pool_offset: u64,
    length: u64,
    gpu_address: u64,
) -> Vec<u8> {
    sized(&RgfxPoolBlock {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_POOL_BLOCK,
            size: 0,
            seqno,
        },
        status,
        reserved_14: 0,
        pool_offset,
        length,
        gpu_address,
    })
}

/// A `PRIVATE_BLOCK` for `PRIVATE_TAKE` `seqno`.
pub(crate) fn private_block(seqno: u64, status: u32, gpu_address: u64, length: u64) -> Vec<u8> {
    sized(&RgfxPrivateBlock {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_PRIVATE_BLOCK,
            size: 0,
            seqno,
        },
        status,
        reserved_14: 0,
        gpu_address,
        length,
    })
}

/// A pages block's `SURFACE_PAGES` answer for command `seqno` where the guest took
/// `IMPORT_ADDRESSES`: a `COMPLETE` of 32 bytes, with `gpu_address` where `status` is
/// [`RGFX_STATUS_OK`] (0 otherwise).
pub(crate) fn pages_answer(seqno: u64, status: u32, gpu_address: u64) -> Vec<u8> {
    sized(&RgfxPagesAnswer {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_COMPLETE,
            size: 0,
            seqno,
        },
        status,
        reserved_14: 0,
        gpu_address: if status == RGFX_STATUS_OK {
            gpu_address
        } else {
            0
        },
    })
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

/// `GET_DISPLAYS`' answer: each display the guest sees (`Shared::displays_seen`: display 0
/// alone where it did not take [`RGFX_FEATURE_MULTI_DISPLAY`]), in order of id, connected, with
/// the mode it offers (`Shared::offered_mode`) as its one mode, which it prefers, and, where the
/// guest took [`RGFX_FEATURE_DISPLAY_SCALE`], its screen's scale between the display's record and
/// its mode. The guest is told each display's mode from now on (`device`'s refresh).
fn displays(shared: &Shared, seqno: u64) -> Vec<u8> {
    let seen = shared.displays_seen();
    let mut bytes = RgfxMsgHeader {
        r#type: RGFX_EVT_DISPLAYS,
        size: 0,
        seqno,
    }
    .as_bytes()
    .to_vec();
    bytes.extend_from_slice(&seen.to_le_bytes());
    for (id, display) in (0..seen).zip(shared.displays.iter()) {
        let mode = shared.offered_mode(display);
        display.state.lock().told = Some(mode);
        let entry = RgfxDisplayEntry {
            id,
            connected: 1,
            preferred_mode: 0,
            mode_count: 1,
        };
        bytes.extend_from_slice(entry.as_bytes());
        if shared.driver_took(RGFX_FEATURE_DISPLAY_SCALE) {
            let scale = RgfxDisplayScale {
                scale_120ths: mode.scale_120ths,
                reserved_4: 0,
            };
            bytes.extend_from_slice(scale.as_bytes());
        }
        let mode = RgfxModeEntry {
            width: mode.width,
            height: mode.height,
            refresh_mhz: mode.refresh_mhz,
        };
        bytes.extend_from_slice(mode.as_bytes());
    }
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

/// `SET_SCANOUT`: the scanout lies in its display's room (`Shared::in_scanout_room`).
fn set_scanout(shared: &Shared, command: &RgfxSetScanout) -> u32 {
    let Some(display) = shared.seen_display(command.display) else {
        return RGFX_STATUS_NO_DISPLAY;
    };
    if command.width == 0 {
        display.state.lock().scanout = None;
        display.screen.scanout(None);
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
    if !shared.in_scanout_room(command.display, scanout.offset, scanout.span()) {
        return RGFX_STATUS_INVALID_ARGUMENT;
    }
    display.state.lock().scanout = Some(scanout);
    display.screen.scanout(Some(scanout));
    RGFX_STATUS_OK
}

/// `SET_CURSOR`: the image is copied out of the region now; the guest may reuse its bytes once
/// it has the answer.
fn set_cursor(shared: &Shared, command: &RgfxSetCursor) -> u32 {
    let Some(display) = shared.seen_display(command.display) else {
        return RGFX_STATUS_NO_DISPLAY;
    };
    if command.width == 0 {
        display.state.lock().cursor_image = None;
        display.screen.cursor_image(None);
        shared.cursor_shape(command.display, None);
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
    let image = CursorImage {
        width,
        height,
        hot_spot: (command.hot_x, command.hot_y),
        rgba: straight_rgba(&bgra),
    };
    let shown = {
        let mut state = display.state.lock();
        let shown = converted(&image, state.cursor_color.as_ref());
        state.cursor_image = Some(image);
        shown
    };
    display.screen.cursor_image(Some(shown));
    shared.cursor_shape(
        command.display,
        Some(CursorShape {
            width,
            height,
            hot_spot: (command.hot_x, command.hot_y),
        }),
    );
    RGFX_STATUS_OK
}

/// `image` as the screen shows it: converted through `color` where the cursor has a chain.
fn converted(image: &CursorImage, color: Option<&CursorColor>) -> CursorImage {
    let mut shown = image.clone();
    if let Some(color) = color {
        color.convert(&mut shown.rgba);
    }
    shown
}

/// `CURSOR_COLOR`, where the guest took [`RGFX_FEATURE_CURSOR_COLOR`]: the display's cursor
/// converts its image through the chain from now on, and the screen is handed the last image
/// again where the chain changed. A message the device cannot read, of a display the guest does
/// not see, or from a guest that did not take the feature, is dropped, and the cursor keeps its
/// chain (logged once a kind).
fn cursor_color(shared: &Shared, message: &Message) {
    let Some(command) = read::<RgfxCursorColor>(message) else {
        log::warn!(
            target: "reims_vgpu_graphics",
            "a CURSOR_COLOR shorter than its fields was dropped"
        );
        return;
    };
    if !shared.driver_took(RGFX_FEATURE_CURSOR_COLOR) {
        log::warn!(
            target: "reims_vgpu_graphics",
            "a CURSOR_COLOR from a guest that did not take CURSOR_COLOR was dropped"
        );
        return;
    }
    let Some(display) = shared.seen_display(command.display) else {
        log::warn!(
            target: "reims_vgpu_graphics",
            "a CURSOR_COLOR of display {} was dropped: there is no such display",
            command.display
        );
        return;
    };
    let color = match CursorColor::read(&command) {
        Ok(color) => color,
        Err(dropped) => {
            log::warn!(
                target: "reims_vgpu_graphics", kind:? = dropped;
                "a CURSOR_COLOR of display {} was dropped ({dropped:?}): the cursor keeps its \
                 colour",
                command.display
            );
            return;
        }
    };
    let (shown, probe) = {
        let mut state = display.state.lock();
        if state.cursor_color == color {
            return;
        }
        state.cursor_color = color;
        let color = state.cursor_color.as_ref();
        let probe = color.map(|color| {
            [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]].map(|rgb| color.pixel(rgb))
        });
        let shown = (state.cursor_image.as_ref()).map(|image| converted(image, color));
        (shown, probe)
    };
    // A state change, which the guest makes at its login and at each change of the display's
    // profile: what the chain does to sRGB's primaries and to white, the boots' read-back.
    match probe {
        Some([red, green, blue, white]) => log::info!(
            target: "reims_vgpu_graphics",
            "display {}'s cursor converts its colours through the guest's chain: red (255, 0, 0) \
             shows as {red:?}, green as {green:?}, blue as {blue:?}, white as {white:?}",
            command.display
        ),
        None => log::info!(
            target: "reims_vgpu_graphics",
            "display {}'s cursor shows its colours as they are",
            command.display
        ),
    }
    if let Some(shown) = shown {
        display.screen.cursor_image(Some(shown));
    }
}

/// `PRESENT_SURFACE`, with no ring held, taken at reset count `epoch`: what answers it where it is
/// refused here, else `None`.
fn present(shared: &Shared, message: &Message, epoch: u64) -> Option<Vec<u8>> {
    let seqno = message.header.seqno;
    let status = match read::<RgfxPresentSurface>(message) {
        // A bit the wire does not name, or a kept present that names a surface.
        Some(command)
            if command.flags & !RGFX_PRESENT_KEEP != 0
                || (command.flags & RGFX_PRESENT_KEEP != 0 && command.iosurface_id != 0) =>
        {
            RGFX_STATUS_INVALID_ARGUMENT
        }
        Some(command) if shared.seen_display(command.display).is_some() => {
            if command.flags & RGFX_PRESENT_KEEP != 0 {
                log::trace!(
                    target: "reims_vgpu_graphics",
                    "PRESENT_SURFACE of transaction {}: display {} keeps what it shows",
                    command.transaction_id,
                    command.display
                );
                let answer = shared.complete_answer(seqno, epoch, RGFX_CMD_PRESENT_SURFACE);
                shared.host.present_keep(command.display, answer);
                return None;
            }
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

/// `DISPLAY_COLOR`, on the presents' lane: its fields and entries to the host, which stages them
/// (`Host::display_color`). Not answered: one for a display the guest does not see, or shorter
/// than the entries it names, or naming more than [`RGFX_COLOR_PART_ENTRIES`], is logged and
/// dropped.
fn display_color(shared: &Shared, message: &Message) {
    let Some(color) = read::<RgfxDisplayColor>(message) else {
        log::warn!(
            target: "reims_vgpu_graphics",
            "a DISPLAY_COLOR shorter than its fields was dropped"
        );
        return;
    };
    let fields = size_of::<RgfxDisplayColor>();
    let entries = (color.part_entries <= RGFX_COLOR_PART_ENTRIES)
        .then(|| {
            const ENTRY: usize = size_of::<RgfxColorEntry>();
            let length = color.part_entries as usize * ENTRY;
            let bytes = message.bytes.get(fields..fields.checked_add(length)?)?;
            let (entries, _) = bytes.as_chunks::<ENTRY>();
            entries
                .iter()
                .map(|entry| RgfxColorEntry::read_from_bytes(entry).ok())
                .collect::<Option<Vec<_>>>()
        })
        .flatten();
    match entries {
        Some(entries) if shared.seen_display(color.display).is_some() => {
            shared.host.display_color(&color, &entries);
        }
        Some(_) => log::warn!(
            target: "reims_vgpu_graphics",
            "a DISPLAY_COLOR of display {} (transaction {}) was dropped: there is no such display",
            color.display,
            color.transaction_id
        ),
        None => log::warn!(
            target: "reims_vgpu_graphics",
            "a DISPLAY_COLOR of transaction {} naming {} entries was dropped: the message does \
             not carry them, or they are past {RGFX_COLOR_PART_ENTRIES}",
            color.transaction_id,
            color.part_entries
        ),
    }
}

/// `PRESENT_SURFACE` on a display the guest sees: the host answers once it shows the surface, or
/// refuses it (`Host::present_surface`), at once or from any thread.
fn present_surface(
    shared: &Shared,
    command: &RgfxPresentSurface,
    generation: Option<u64>,
    seqno: u64,
    epoch: u64,
) {
    log::trace!(
        target: "reims_vgpu_graphics",
        "PRESENT_SURFACE of transaction {}: IOSurface {} on display {}",
        command.transaction_id,
        command.iosurface_id,
        command.display
    );
    shared.host.present_surface(
        command.display,
        command.iosurface_id,
        generation,
        shared.complete_answer(seqno, epoch, RGFX_CMD_PRESENT_SURFACE),
    );
}

/// `SET_VBLANK`.
fn set_vblank(shared: &Shared, command: &RgfxSetVblank) -> u32 {
    let Some(display) = shared.seen_display(command.display) else {
        return RGFX_STATUS_NO_DISPLAY;
    };
    display.state.lock().vblank = command.enabled != 0;
    RGFX_STATUS_OK
}

/// `SET_DISPLAY_POWER`: the screen is told where the panel's state changes. The display's
/// scanout, presents and `VBLANK`s go on either way (`display`'s module doc).
fn set_display_power(shared: &Shared, command: &RgfxSetDisplayPower) -> u32 {
    let Some(display) = shared.seen_display(command.display) else {
        return RGFX_STATUS_NO_DISPLAY;
    };
    let asleep = command.on == 0;
    let changed = {
        let mut state = display.state.lock();
        std::mem::replace(&mut state.asleep, asleep) != asleep
    };
    if changed {
        if command.display == 0 {
            log::info!(
                target: "reims_vgpu_graphics",
                "the display's panel is {}: the guest's display {}",
                if asleep { "off" } else { "on" },
                if asleep { "sleeps" } else { "woke" }
            );
        } else {
            log::info!(
                target: "reims_vgpu_graphics",
                "display {}'s panel is {}: the guest's display {}",
                command.display,
                if asleep { "off" } else { "on" },
                if asleep { "sleeps" } else { "woke" }
            );
        }
        display.screen.power(!asleep);
    }
    RGFX_STATUS_OK
}
