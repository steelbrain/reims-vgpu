//! The window's mirror on the compositor's GPU: the presented frame in host memory, which a draw
//! on a device of the window's own on that GPU reads (`sink`, "The compositor's GPU").
//!
//! The presented surface's texture is the runtime's device's, and on a GPU with memory of its own
//! it lies in a device cache there, which another GPU cannot read. So at each draw the runtime's
//! GPU brings the mirror up to the frame, and sends only what changed:
//! - it copies the presented texture into a buffer of its own memory (`now`), as linear rows of
//!   the mirror's pitch;
//! - it compares `now` with the frame it sent last (`sent`, the `now` of the last sync) in tiles
//!   of 64 × 64 pixels, and copies each tile that differs into the mirror, in host memory
//!   (`displayMirrorTile`, the display's shaders): a tile that changed nothing crosses no link;
//! - `now` then becomes `sent` for the next sync.
//!
//! A mirror made again (its first sync, a frame of another format or size) sends every tile. The
//! mirror holds the frame's bytes as the texture holds them, so a draw from it draws what a draw
//! from the texture draws.
//!
//! **Ordering.** The draw that reads the mirror on the compositor's GPU comes after the sync that
//! wrote it, whose command buffer makes its writes available to the host as its work ends:
//! - where the platform exports the sync's completion as a sync file
//!   (`CommandBuffer::export_completion_sync_file`), the draw waits for it on its own GPU
//!   ([`Synced::completion`], `CommandBuffer::encode_wait_for_sync_file`), and the window's
//!   thread waits for no GPU work of the sync's;
//! - else, and where the draw's device cannot wait for the file, the window's thread waits until
//!   the sync has finished before it encodes the draw.
//!
//! A sync writes the mirror only once the last draw that read it has finished
//! ([`Mirror::read_by`]), and the sync before it too, whose count of tiles it then reads: the
//! draws come one at a time on the window's thread, so the last draw was committed or dropped by
//! then, and it waited for the sync before it.
//!
//! **On a copy engine.** Where the platform gives a queue whose copies run on a copy engine of
//! the runtime's GPU (`GpuDevice::new_command_queue_for_copies`), the tiles cross the link there,
//! apart from the engine that runs the guest's render and compute work:
//! - the sync's command buffer, on the mirror's own queue, copies the presented texture into
//!   `now` and compares it with `sent` as above, but only lists the tiles that differ
//!   (`displayMirrorTileList`), and writes nothing into host memory;
//! - the window's thread commits it at once, then waits for it, for the last draw that read the
//!   mirror and for the last copy into it, so the compare runs beside those two;
//! - then it copies the listed tiles out of `now` into the mirror on the copy engine's queue: a
//!   run of tiles side by side is one copy in each of its pixel rows, a run of whole rows of
//!   tiles one copy, and a copy that meets the one before it joins it. With no tile changed,
//!   nothing is copied and the draw waits for nothing;
//! - the copy's completion goes to the draw as a sync's does above, and `now` becomes `sent`.
//!
//! The kernel's own writes into host memory hold the GPU engine that every queue of its
//! graphics family runs on for as long as they take to cross the link, and no other queue of that
//! family runs meanwhile. On an RTX 3090 on PCIe 3.0 x4, with the compositor on a Radeon 780M, a
//! GPU-bound guest drew 11% fewer frames with the window drawn from the mirror than with the
//! window presenting from the runtime's GPU, and each sync waited 71 ms for the one before it,
//! behind the guest's command buffers (`docs/graphics-kext/design.md`, Decided 14). Where the
//! platform gives no such queue (the Vulkan platform's runtime gives a program none), or the
//! runtime refuses the list's kernel, the sync writes the tiles itself, as above.
//!
//! **Its pages.** The mirror's host memory is the platform's
//! (`reims_vgpu_vulkan::os::PrivateMemory`), whose unmap waits until the runtime has freed every
//! import of its pages. The runtime runs a no-copy buffer's deallocator at the buffer's release,
//! Metal's moment, and frees the buffer's import after it, on a handler thread: the old mirror's
//! release, as a mirror is made again, runs its deallocator on the window's thread just before
//! the draw's commit. An unmap of the OS's there takes the pages from under a live import, and
//! RADV, which hands every pointer import of its device to each submission, rejected the draw's
//! submission (`-EFAULT`), losing the device (`docs/lessons.md`).
//!
//! **What it refuses.** A surface of a pixel format the mirror does not carry (not 4 or 8 bytes a
//! pixel), and any object the runtime refuses to make or a sync that fails, answer `None`: the
//! window then presents from the runtime's GPU.

use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};

use reims_vgpu_vulkan::core::host::HostPtr;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::os::clock::MonoTime;
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Library as _, Texture as _};
use reims_vgpu_vulkan::core::platform::types::{CommandBufferStatus, MtlOrigin, MtlSize, NsRange};
use reims_vgpu_vulkan::core::platform::{Handle, HostRange, Platform};
use reims_vgpu_vulkan::core::raise::{OnFault, RaiseResult};
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_vulkan::os::PrivateMemory;

use super::sink::{Open, end_blit};
use crate::entry::no_catch;
use crate::platform::ApiPlatform;

/// The display's kernel that sends a tile ([`Mirror`]).
const TILE_FUNCTION: &str = "displayMirrorTile";

/// The display's kernel that lists the tiles that changed, where the copies run on a copy engine
/// (the module doc, "On a copy engine").
const LIST_FUNCTION: &str = "displayMirrorTileList";

/// A tile's side, in pixels.
const TILE: u64 = 64;

/// The threads of a tile's threadgroup (`display_mirror_threads` in the display's shaders).
const TILE_THREADS: u64 = 256;

/// `MTLResourceStorageModePrivate`: the runtime GPU's own memory.
const PRIVATE: u64 = 0x20;

/// The bytes of a pixel of `format` (an `MTLPixelFormat`) the mirror carries: the formats of 4
/// and 8 bytes a pixel a presented surface may have; `None` for any other.
fn pixel_bytes(format: u64) -> Option<u64> {
    match format {
        // RGBA8Unorm(_sRGB), BGRA8Unorm(_sRGB), RGB10A2Unorm, RG11B10Float, RGB9E5Float,
        // BGR10A2Unorm, BGR10_XR(_sRGB).
        70 | 71 | 80 | 81 | 90 | 92 | 93 | 94 | 554 | 555 => Some(4),
        // RGBA16Unorm, RGBA16Float, BGRA10_XR(_sRGB).
        110 | 115 | 552 | 553 => Some(8),
        _ => None,
    }
}

/// The window's mirror (the module doc), on the runtime's device.
pub(crate) struct Mirror<P: Platform> {
    /// The runtime's device.
    device: P::Device,
    /// The queue its syncs run on.
    queue: P::CommandQueue,
    /// `displayMirrorTile`'s pipeline.
    pipeline: P::ComputePipeline,
    /// The count of tiles a sync sent, which the kernel adds to (4 bytes, shared).
    tiles: P::Buffer,
    /// Where the platform gives a queue for copies: that queue, and `displayMirrorTileList`'s
    /// pipeline (the module doc, "On a copy engine").
    copies: Option<(P::CommandQueue, P::ComputePipeline)>,
    /// The frame the mirror holds, made at the first sync.
    frame: Mutex<Option<Frame<P>>>,
    /// The syncs and the tiles they sent since the last log line, and when that was.
    counts: Mutex<Counts>,
    /// The device's recoveries when the mirror was made ([`Self::recovered`]).
    made_at: u64,
    /// The tiles the last sync sent.
    last: AtomicU64,
}

/// What a mirror holds of one frame's shape.
struct Frame<P: Platform> {
    /// The pixel format, width and height it holds.
    shape: (u64, u64, u64),
    /// Bytes from one of its rows to the next.
    bytes_per_row: u64,
    /// The frame copied at the sync under way, in the runtime GPU's memory.
    now: P::Buffer,
    /// The frame the last sync sent, in the runtime GPU's memory.
    sent: P::Buffer,
    /// The mirror: the runtime's buffer over host memory of its own.
    mirror: P::Buffer,
    /// The kernel's arguments (`DisplayMirrorFrame`), shared.
    arguments: P::Buffer,
    /// Where the copies run on a copy engine: the list of the tiles that changed, shared, its
    /// first word their count and their numbers after it.
    list: Option<P::Buffer>,
    /// Every tile goes at the next sync: the mirror was made since the last.
    fresh: bool,
    /// The last draw that read the mirror, on the compositor's GPU.
    reader: Option<P::CommandBuffer>,
    /// The last sync, where the draw waits for it on its GPU: the next sync waits for it, and
    /// reads its count of tiles. Where the copies run on a copy engine, the last copy.
    pending: Option<P::CommandBuffer>,
    /// The last sync asked for its completion as a sync file.
    asked: bool,
}

/// The syncs and tiles counted for the debug log.
#[derive(Default)]
struct Counts {
    syncs: u64,
    /// The syncs the draw waited for on the compositor's GPU.
    on_gpu: u64,
    tiles: u64,
    bytes: u64,
    /// Nanoseconds the window's thread waited on the syncs.
    waited: u64,
    since: Option<MonoTime>,
}

/// The mirror after a sync: the runtime's buffer over it, and where the frame lies in it.
pub(crate) struct Synced<P: Platform> {
    /// The runtime's buffer over the mirror's host memory.
    pub(crate) buffer: P::Buffer,
    /// Bytes from one of the frame's rows to the next.
    pub(crate) bytes_per_row: u64,
    /// The frame's width in pixels.
    pub(crate) width: u64,
    /// Its height in pixels.
    pub(crate) height: u64,
    /// The sync's completion, where the draw waits for it on its GPU (the module doc,
    /// "Ordering"); `None` where the sync has finished.
    pub(crate) completion: Option<Completion<P>>,
}

/// A sync's completion that the draw waits for on its GPU.
pub(crate) struct Completion<P: Platform> {
    /// The sync file of the sync's completion.
    pub(crate) file: OwnedFd,
    /// The sync, which the window's thread waits for where the draw's device cannot wait for the
    /// file.
    pub(crate) sync: P::CommandBuffer,
}

/// A compute encoder's end.
fn end_compute<P: Platform>(encoder: &P::ComputeEncoder) -> RaiseResult<()> {
    encoder.encode(OnFault::Die, ComputeCmd::EndEncoding)
}

/// A copy of `length` bytes from byte `at`, which lie at the same place in `now` and the mirror.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    at: u64,
    length: u64,
}

/// The copies that carry the tiles `tiles` (their numbers, a row of tiles after another, sorted)
/// of a frame of `width` by `height` pixels of `bytes` bytes, its rows `bytes_per_row` apart: a
/// run of tiles side by side is a copy of its bytes in each of its rows, and a run across a
/// whole row of tiles a copy of its rows whole; a copy that starts where the one before it ends
/// joins it. No byte of a tile not listed is copied, but the bytes past a row's last pixel, up to
/// the next row, in a whole row's copy.
fn spans(tiles: &[u32], width: u64, height: u64, bytes: u64, bytes_per_row: u64) -> Vec<Span> {
    let across = width.div_ceil(TILE);
    let mut spans: Vec<Span> = Vec::new();
    let mut push = |span: Span| match spans.last_mut() {
        Some(before) if before.at + before.length == span.at => before.length += span.length,
        _ => spans.push(span),
    };
    let mut at = 0;
    while at < tiles.len() {
        // The run from tile `first` to `last`, in one row of tiles.
        let first = u64::from(tiles[at]);
        let mut last = first;
        at += 1;
        while at < tiles.len()
            && u64::from(tiles[at]) == last + 1
            && u64::from(tiles[at]) % across != 0
        {
            last += 1;
            at += 1;
        }
        let (row, x0, x1) = (first / across, first % across, last % across + 1);
        let (y0, y1) = (row * TILE, ((row + 1) * TILE).min(height));
        if x0 == 0 && x1 == across {
            push(Span {
                at: y0 * bytes_per_row,
                length: (y1 - y0) * bytes_per_row,
            });
        } else {
            let (from, to) = (x0 * TILE * bytes, (x1 * TILE).min(width) * bytes);
            for y in y0..y1 {
                push(Span {
                    at: y * bytes_per_row + from,
                    length: to - from,
                });
            }
        }
    }
    spans
}

impl<P: ApiPlatform> Mirror<P> {
    /// The mirror on `device`, the runtime's: its queue, the tile kernel's pipeline and the
    /// count of tiles, and the platform's queue for copies where it gives one (the module doc,
    /// "On a copy engine"); `None` where the runtime refuses one, logged.
    pub(crate) fn new(device: P::Device) -> Option<Mirror<P>> {
        let copies = device.new_command_queue_for_copies();
        Self::with_copies(device, copies)
    }

    /// The mirror on `device`, whose copies into host memory run on `copies` where it is given
    /// (the module doc, "On a copy engine"), else in its syncs; `None` where the runtime refuses
    /// one of its objects, logged.
    pub(crate) fn with_copies(
        device: P::Device,
        copies: Option<P::CommandQueue>,
    ) -> Option<Mirror<P>> {
        let made = (|| {
            // On a present's path, as the window's sink's queue is: a GPU queue of its own where
            // the runtime has one to give.
            let queue =
                (device.new_command_queue_unshared()).or_else(|| device.new_command_queue())?;
            let library = device.new_shader_library().ok().flatten()?;
            let function = no_catch(library.new_function(OnFault::Die, TILE_FUNCTION))?;
            let pipeline = device
                .new_compute_pipeline_state_for_function(Some(&function))
                .ok()
                .flatten()?;
            let tiles = no_catch(device.new_buffer_of_length(OnFault::Die, 4, 0))?;
            Some((queue, library, pipeline, tiles))
        })();
        let Some((queue, library, pipeline, tiles)) = made else {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the runtime refused the window's mirror (its queue, its kernel's pipeline or its \
                 count): the window presents from the runtime's GPU"
            );
            return None;
        };
        let copies = copies.and_then(|copies| {
            let listing = (no_catch(library.new_function(OnFault::Die, LIST_FUNCTION)))
                .and_then(|function| {
                    (device.new_compute_pipeline_state_for_function(Some(&function)))
                        .ok()
                        .flatten()
                });
            if listing.is_none() {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "the runtime refused the window's mirror's list of tiles (its kernel's \
                     pipeline): the mirror's syncs copy the tiles themselves, as on a platform \
                     with no queue for copies"
                );
            }
            Some((copies, listing?))
        });
        Some(Mirror {
            made_at: device.recoveries(),
            last: AtomicU64::new(0),
            device,
            queue,
            pipeline,
            tiles,
            copies,
            frame: Mutex::new(None),
            counts: Mutex::default(),
        })
    }

    /// The mirror brought up to `surface` (the module doc), its rows `align` bytes apart at the
    /// least: the runtime's buffer over it, once the sync has finished; `None` where the mirror
    /// does not carry the surface's format, or the runtime refuses an object or the sync. `now`
    /// reads the platform's monotonic clock, for the time the window's thread waits on the sync,
    /// which the debug log counts.
    pub(crate) fn sync(
        &self,
        surface: &P::Texture,
        align: u64,
        now: &dyn Fn() -> MonoTime,
    ) -> Option<Synced<P>> {
        let started = now();
        let format = surface.pixel_format();
        let (width, height) = (surface.width(), surface.height());
        let bytes = pixel_bytes(format)?;
        if width == 0 || height == 0 {
            return None;
        }
        let mut kept = self.frame.lock();
        let reshaped = (kept.as_ref()).is_none_or(|f| f.shape != (format, width, height));
        // Where the sync writes the mirror itself, or the mirror is made again, the last draw
        // that read it and the last sync finish first; where a copy engine copies, the compare
        // runs beside them (the module doc, "On a copy engine").
        if (self.copies.is_none() || reshaped)
            && let Some(frame) = kept.as_mut()
            && !self.settle(frame, bytes)
        {
            return None;
        }
        if reshaped {
            *kept = None;
            *kept = Some(self.frame(format, width, height, bytes, align)?);
        }
        let frame = kept.as_mut()?;
        let bytes_per_row = frame.bytes_per_row;
        let words = |b: u64| u32::try_from(b / 4).ok();
        let arguments = [
            words(bytes_per_row)?,
            words(width * bytes)?,
            u32::try_from(height).ok()?,
            words(TILE * bytes)?,
            u32::from(frame.fresh),
        ];
        // The count the kernel adds to: the tiles' count, or the list's first word.
        let count = frame.list.as_ref().unwrap_or(&self.tiles);
        // SAFETY: shared buffers of 20 bytes and of at least 4, mapped while they live, which no
        // command buffer reads: the last sync has finished, and so has the last compare.
        unsafe {
            let at: *mut u32 = frame.arguments.contents().as_ptr();
            std::ptr::copy_nonoverlapping(arguments.as_ptr(), at, arguments.len());
            count.contents().as_ptr::<u32>().write(0);
        }
        let command_buffer = no_catch(self.queue.command_buffer(OnFault::Die))?;
        let blit = Open::new(
            no_catch(command_buffer.blit_encoder(OnFault::Die))?,
            end_blit::<P>,
        );
        no_catch(blit.get().encode(
            OnFault::Die,
            BlitCmd::CopyTextureToBuffer {
                source_texture: surface,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(width, height, 1),
                destination_buffer: &frame.now,
                destination_offset: 0,
                destination_bytes_per_row: bytes_per_row,
                destination_bytes_per_image: bytes_per_row * height,
            },
        ));
        blit.end();
        let compute = Open::new(
            no_catch(command_buffer.compute_encoder(OnFault::Die))?,
            end_compute::<P>,
        );
        let encoder = compute.get();
        let (pipeline, buffers) = match (&self.copies, &frame.list) {
            (Some((_, listing)), Some(list)) => (
                listing,
                vec![
                    Some(&frame.now),
                    Some(&frame.sent),
                    Some(list),
                    Some(&frame.arguments),
                ],
            ),
            _ => (
                &self.pipeline,
                vec![
                    Some(&frame.now),
                    Some(&frame.sent),
                    Some(&frame.mirror),
                    Some(&frame.arguments),
                    Some(&self.tiles),
                ],
            ),
        };
        no_catch(encoder.encode(
            OnFault::Die,
            ComputeCmd::SetComputePipelineState {
                state: Some(pipeline),
            },
        ));
        no_catch(encoder.encode(
            OnFault::Die,
            ComputeCmd::SetBuffers {
                buffers: &buffers,
                offsets: &vec![0; buffers.len()],
                range: NsRange {
                    location: 0,
                    length: buffers.len() as u64,
                },
            },
        ));
        no_catch(encoder.encode(
            OnFault::Die,
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(width.div_ceil(TILE), height.div_ceil(TILE), 1),
                threads_per_threadgroup: MtlSize::new(TILE_THREADS, 1, 1),
            },
        ));
        compute.end();
        let completion = match &self.copies {
            None => {
                // `now` is the frame sent from here.
                std::mem::swap(&mut frame.now, &mut frame.sent);
                frame.fresh = false;
                self.hand_over(frame, command_buffer, Some(bytes))?
            }
            Some((copies, _)) => self.copy_listed(frame, copies, &command_buffer, bytes)?,
        };
        self.count_sync(completion.is_some(), started, now());
        Some(Synced {
            buffer: frame.mirror.clone_ref(),
            bytes_per_row,
            width,
            height,
            completion,
        })
    }

    /// The last draw that read `frame`'s mirror and the last sync (or copy) into it, of a frame
    /// of `bytes` bytes a pixel, finished: the sync's count of tiles read, where the sync's
    /// kernel counted them; `false`, logged, where it failed.
    fn settle(&self, frame: &mut Frame<P>, bytes: u64) -> bool {
        // The last draw that read the mirror has finished before this sync writes it, or makes it
        // again.
        if let Some(reader) = frame.reader.take()
            && reader.status() != CommandBufferStatus::NOT_ENQUEUED
        {
            no_catch(reader.wait_until_completed(OnFault::Die));
        }
        // And the last sync, where the draw waited for it on its GPU: its count is read here.
        if let Some(last) = frame.pending.take() {
            no_catch(last.wait_until_completed(OnFault::Die));
            let held = pixel_bytes(frame.shape.0).unwrap_or(bytes);
            return self.finished(&last, self.copies.is_none().then_some(held));
        }
        true
    }

    /// The command buffer `sync`, encoded, which writes `frame`'s mirror, committed for the draw:
    /// its completion as a sync file where the platform exports one and it has not finished
    /// already, which the draw waits for on its GPU (`Some(Some(_))`), else finished on the host
    /// (`Some(None)`), its count of tiles read where `count_of` says the bytes of the frame's
    /// pixels; `None` where it failed, logged.
    fn hand_over(
        &self,
        frame: &mut Frame<P>,
        sync: P::CommandBuffer,
        count_of: Option<u64>,
    ) -> Option<Option<Completion<P>>> {
        let exported = sync.add_completion_sync_file();
        frame.asked = exported;
        no_catch(sync.commit(OnFault::Die));
        // The sync file, where the platform exports one and the sync has not finished already.
        let file = exported
            .then(|| sync.export_completion_sync_file())
            .flatten()
            .flatten();
        match file {
            Some(file) => {
                frame.pending = Some(sync.clone_ref());
                Some(Some(Completion { file, sync }))
            }
            None => {
                no_catch(sync.wait_until_completed(OnFault::Die));
                self.finished(&sync, count_of).then_some(None)
            }
        }
    }

    /// Where a copy engine copies (the module doc, "On a copy engine"): the compare `compare`,
    /// encoded, committed and finished on the window's thread, beside the last draw and the last
    /// copy, which then finish too; then the tiles it listed copied out of `frame`'s `now` into
    /// its mirror on `copies`, a frame of `bytes` bytes a pixel, committed for the draw as
    /// [`Self::hand_over`] does (`Some(None)` where no tile changed, and nothing is copied);
    /// `None` where the compare or the last copy failed, or the runtime refused an object,
    /// logged.
    fn copy_listed(
        &self,
        frame: &mut Frame<P>,
        copies: &P::CommandQueue,
        compare: &P::CommandBuffer,
        bytes: u64,
    ) -> Option<Option<Completion<P>>> {
        no_catch(compare.commit(OnFault::Die));
        no_catch(compare.wait_until_completed(OnFault::Die));
        if !self.finished(compare, None) || !self.settle(frame, bytes) {
            return None;
        }
        let (width, height) = (frame.shape.1, frame.shape.2);
        let all = width.div_ceil(TILE) * height.div_ceil(TILE);
        let list = frame.list.as_ref()?;
        // SAFETY: the shared list, one word of count and one for each tile, which the finished
        // compare wrote, and no command buffer since.
        let mut listed: Vec<u32> = unsafe {
            let at = list.contents().as_ptr::<u32>();
            let count = u64::from(at.read()).min(all);
            std::slice::from_raw_parts(at.add(1), count as usize).to_vec()
        };
        listed.sort_unstable();
        // `now` is the frame sent from here.
        std::mem::swap(&mut frame.now, &mut frame.sent);
        frame.fresh = false;
        let tiles = listed.len() as u64;
        self.last.store(tiles, Ordering::Relaxed);
        self.count_tiles(tiles, tiles * TILE * TILE * bytes);
        if listed.is_empty() {
            frame.asked = false;
            return Some(None);
        }
        let copy = no_catch(copies.command_buffer(OnFault::Die))?;
        let blit = Open::new(no_catch(copy.blit_encoder(OnFault::Die))?, end_blit::<P>);
        for span in spans(&listed, width, height, bytes, frame.bytes_per_row) {
            no_catch(blit.get().encode(
                OnFault::Die,
                BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(&frame.sent),
                    source_offset: span.at,
                    destination_buffer: Some(&frame.mirror),
                    destination_offset: span.at,
                    size: span.length,
                },
            ));
        }
        blit.end();
        self.hand_over(frame, copy, None)
    }

    /// The command buffer `sync`, finished, which wrote the mirror: where `count_of` says the
    /// bytes of the frame's pixels, its kernel's count of tiles read and counted (where a copy
    /// engine copies, the tiles were counted from the list); `false`, logged, where it failed.
    fn finished(&self, sync: &P::CommandBuffer, count_of: Option<u64>) -> bool {
        if sync.status() != CommandBufferStatus::COMPLETED {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the window's mirror's sync failed ({:?}): the window presents from the \
                 runtime's GPU",
                sync.error()
            );
            return false;
        }
        if let Some(bytes) = count_of {
            // SAFETY: the 4-byte shared count, which the finished sync wrote, and no sync since.
            let tiles = u64::from(unsafe { self.tiles.contents().as_ptr::<u32>().read() });
            self.last.store(tiles, Ordering::Relaxed);
            self.count_tiles(tiles, tiles * TILE * TILE * bytes);
        }
        true
    }

    /// The tiles the last sync sent, once it has finished.
    #[cfg(test)]
    pub(crate) fn last_sent(&self) -> u64 {
        let pending = self.frame.lock().as_mut().and_then(|f| f.pending.take());
        if let Some(last) = pending {
            no_catch(last.wait_until_completed(OnFault::Die));
            let bytes =
                (self.frame.lock().as_ref()).map_or(4, |f| pixel_bytes(f.shape.0).unwrap_or(4));
            assert!(
                self.finished(&last, self.copies.is_none().then_some(bytes)),
                "the mirror's last sync failed"
            );
        }
        self.last.load(Ordering::Relaxed)
    }

    /// Whether the mirror's copies run on a queue of their own (the module doc, "On a copy
    /// engine").
    #[cfg(test)]
    pub(crate) fn copies_on_their_queue(&self) -> bool {
        self.copies.is_some()
    }

    /// Whether the last sync asked for its completion as a sync file.
    #[cfg(test)]
    pub(crate) fn last_asked(&self) -> bool {
        self.frame.lock().as_ref().is_some_and(|f| f.asked)
    }

    /// Whether the last sync's completion went to the draw as a sync file.
    #[cfg(test)]
    pub(crate) fn last_on_gpu(&self) -> bool {
        self.frame
            .lock()
            .as_ref()
            .is_some_and(|f| f.pending.is_some())
    }

    /// Whether the runtime's device recovered since the mirror was made: its objects then carry
    /// nothing on, and the mirror is made again.
    pub(crate) fn recovered(&self) -> bool {
        self.device.recoveries() != self.made_at
    }

    /// The draw `reader`, on the compositor's GPU, reads the mirror until it finishes: the next
    /// sync waits for it where it was committed.
    pub(crate) fn read_by(&self, reader: &P::CommandBuffer) {
        if let Some(frame) = self.frame.lock().as_mut() {
            frame.reader = Some(reader.clone_ref());
        }
    }

    /// A frame of `format`, `width` by `height` pixels of `bytes` bytes, its rows a multiple of
    /// `align` bytes apart: its buffers and the mirror's host memory; `None`, logged, where the
    /// host or the runtime refuses one.
    fn frame(
        &self,
        format: u64,
        width: u64,
        height: u64,
        bytes: u64,
        align: u64,
    ) -> Option<Frame<P>> {
        let made = (|| {
            let bytes_per_row = (width * bytes).checked_next_multiple_of(align.max(4))?;
            let length = bytes_per_row.checked_mul(height)?;
            // SAFETY: sysconf reads a constant of the host's.
            let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
            let span = length.checked_next_multiple_of(page)?;
            // The platform's memory, whose unmap waits for the runtime's free of every import of
            // its pages (the module doc, "Its pages").
            let pages = PrivateMemory::new(span)?;
            // SAFETY: the memory's `span` bytes, whole pages from a page, are mapped from here
            // until the deallocator, which owns them, drops them once the buffer has gone, and
            // past that until the runtime freed the buffer's import.
            let range = unsafe { HostRange::new(HostPtr::from_ptr(pages.base()), span) };
            let mirror = no_catch(self.device.new_buffer_over_host_memory(
                OnFault::Die,
                range,
                0,
                Some(Box::new(move || drop(pages))),
            ))?;
            let private = |length| {
                no_catch(
                    self.device
                        .new_buffer_of_length(OnFault::Die, length, PRIVATE),
                )
            };
            // A word of count, and one for each tile.
            let list = match self.copies {
                Some(_) => {
                    let words = 1 + width.div_ceil(TILE) * height.div_ceil(TILE);
                    Some(no_catch(self.device.new_buffer_of_length(
                        OnFault::Die,
                        words * 4,
                        0,
                    ))?)
                }
                None => None,
            };
            Some(Frame {
                shape: (format, width, height),
                bytes_per_row,
                now: private(length)?,
                sent: private(length)?,
                mirror,
                arguments: no_catch(self.device.new_buffer_of_length(OnFault::Die, 20, 0))?,
                list,
                fresh: true,
                reader: None,
                pending: None,
                asked: false,
            })
        })();
        if made.is_none() {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the window's mirror of a {width} x {height} frame (pixel format {format}) was \
                 refused: the window presents from the runtime's GPU"
            );
        }
        made
    }

    /// A finished sync sent `tiles` tiles, `bytes` bytes at most (a tile at the frame's edge is
    /// smaller): counted.
    fn count_tiles(&self, tiles: u64, bytes: u64) {
        let mut counts = self.counts.lock();
        counts.tiles += tiles;
        counts.bytes += bytes;
    }

    /// A sync the window's thread spent from `started` to `now` on, which the draw waits for on
    /// its GPU where `on_gpu`: counted, and the counts logged at debug about once a second.
    fn count_sync(&self, on_gpu: bool, started: MonoTime, now: MonoTime) {
        let mut counts = self.counts.lock();
        counts.syncs += 1;
        counts.on_gpu += u64::from(on_gpu);
        counts.waited += now.saturating_duration_since(started).as_nanos() as u64;
        let since = *counts.since.get_or_insert(now);
        let over = now.saturating_duration_since(since).as_secs_f64();
        if over >= 1.0 {
            log::debug!(
                target: "reims_vgpu::graphics::mirror",
                "the window's mirror: {} syncs sent {} tiles, at most {} bytes, over {:.1} s, {} \
                 of them waited for on the compositor's GPU; the window's thread waited {} us on \
                 them",
                counts.syncs,
                counts.tiles,
                counts.bytes,
                over,
                counts.on_gpu,
                counts.waited / 1000
            );
            *counts = Counts {
                since: Some(now),
                ..Counts::default()
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::time::Duration;

    use reims_vgpu_vulkan::VulkanPlatform;
    use reims_vgpu_vulkan::core::platform::ObjId;
    use reims_vgpu_vulkan::core::platform::os::clock::Clock as _;
    use reims_vgpu_vulkan::core::platform::values::TextureDesc;
    use reims_vgpu_vulkan::core::testkit::calllog::Call;
    use reims_vgpu_vulkan::core::testkit::impls::device::COPY_QUEUE;
    use reims_vgpu_vulkan::core::testkit::objects::{CompleteOn, kind};
    use reims_vgpu_vulkan::core::testkit::{Arg, FakePlatform};

    use super::*;

    /// A span of `length` bytes from `at`.
    fn span(at: u64, length: u64) -> Span {
        Span { at, length }
    }

    /// A run of tiles side by side is a copy in each of its pixel rows, of its tiles' bytes
    /// alone, and the tiles' order in the row of tiles is theirs: 256 x 128 pixels of 4 bytes,
    /// rows 1024 bytes apart, tiles 1 and 2 of the first row of four
    #[test]
    fn a_run_of_tiles_is_a_copy_in_each_of_its_rows() {
        let spans = spans(&[1, 2], 256, 128, 4, 1024);
        let rows: Vec<_> = (0..64).map(|y| span(y * 1024 + 256, 512)).collect();
        assert_eq!(spans, rows);
    }

    /// A run across a whole row of tiles is one copy of its rows, joined with the next row of
    /// tiles where that one is whole too: 256 x 128 pixels, both rows of four tiles; the second
    /// row alone
    #[test]
    fn whole_rows_of_tiles_are_one_copy() {
        assert_eq!(
            spans(&[0, 1, 2, 3, 4, 5, 6, 7], 256, 128, 4, 1024),
            [span(0, 128 * 1024)]
        );
        assert_eq!(
            spans(&[4, 5, 6, 7], 256, 128, 4, 1024),
            [span(64 * 1024, 64 * 1024)]
        );
    }

    /// A tile at the frame's right or bottom edge copies only the pixels the frame has there:
    /// 200 x 130 pixels of 4 bytes, rows 1024 bytes apart, four tiles by three; the last tile of
    /// the first row is 8 pixels wide, the last row of tiles 2 pixels high
    #[test]
    fn a_tile_at_the_edge_copies_the_frames_pixels_alone() {
        let rows: Vec<_> = (0..64).map(|y| span(y * 1024 + 768, 32)).collect();
        assert_eq!(spans(&[3], 200, 130, 4, 1024), rows);
        assert_eq!(
            spans(&[11], 200, 130, 4, 1024),
            [span(128 * 1024 + 768, 32), span(129 * 1024 + 768, 32)]
        );
        assert_eq!(
            spans(&[8, 9, 10, 11], 200, 130, 4, 1024),
            [span(128 * 1024, 2 * 1024)]
        );
    }

    /// A copy that starts where the one before it ends joins it: 128 x 128 pixels of 4 bytes,
    /// rows 512 bytes apart with nothing past their pixels, two tiles a row; the second tile of
    /// the first row ends its last pixel row where the first tile of the second row starts its
    /// first. A run does not go on from one row of tiles into the next
    #[test]
    fn copies_that_meet_are_one() {
        let spans = spans(&[1, 2], 128, 128, 4, 512);
        let mut rows: Vec<_> = (0..63).map(|y| span(y * 512 + 256, 256)).collect();
        rows.push(span(63 * 512 + 256, 512));
        rows.extend((65..128).map(|y| span(y * 512, 256)));
        assert_eq!(spans, rows);
    }

    /// A mirror on the fake platform, whose devices give a queue for copies where `copies`, with
    /// a 256 x 128 surface of 4 bytes a pixel; command buffers complete only as the test
    /// completes them.
    fn fake(
        copies: bool,
    ) -> (
        Arc<FakePlatform>,
        Arc<Mirror<FakePlatform>>,
        <FakePlatform as Platform>::Texture,
    ) {
        let platform = Arc::new(FakePlatform::new());
        platform.world().give_copy_queues(copies);
        let device = platform.metal_device(None).unwrap();
        let mirror = Arc::new(Mirror::new(device.clone_ref()).expect("the mirror"));
        let descriptor = platform.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 256,
            height: 128,
            ..TextureDesc::default()
        });
        let surface = no_catch(device.new_texture(OnFault::Die, &descriptor)).unwrap();
        (platform, mirror, surface)
    }

    /// The sync of `mirror` to `surface`, on a thread of its own.
    fn sync_on_a_thread(
        platform: &Arc<FakePlatform>,
        mirror: &Arc<Mirror<FakePlatform>>,
        surface: &<FakePlatform as Platform>::Texture,
    ) -> std::thread::JoinHandle<Option<(u64, bool)>> {
        let (platform, mirror, surface) = (Arc::clone(platform), Arc::clone(mirror), surface.clone_ref());
        std::thread::spawn(move || {
            let synced = mirror.sync(&surface, 256, &|| platform.clock().now_mono())?;
            Some((synced.bytes_per_row, synced.completion.is_some()))
        })
    }

    /// The committed command buffers not yet completed, once there are `count` of them.
    fn committed(platform: &FakePlatform, count: usize) -> Vec<ObjId> {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let committed = platform.world().committed();
            if committed.len() >= count {
                assert_eq!(committed.len(), count, "more command buffers than expected");
                return committed;
            }
            assert!(std::time::Instant::now() < deadline, "{committed:?} of {count}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// `id`, a command buffer, completed.
    fn complete(platform: &FakePlatform, id: ObjId) {
        assert!(platform.world().complete(id, CommandBufferStatus::COMPLETED, None, CompleteOn::Thread));
    }

    /// The buffers the last kernel among `calls` was handed.
    fn bound(calls: &[Call]) -> Vec<Option<ObjId>> {
        let set = (calls.iter().rev())
            .find(|c| c.sel == "setBuffers:offsets:withRange:")
            .expect("buffers bound");
        match &set.args[0] {
            Arg::Objs(objs) => objs.clone(),
            other => panic!("{other:?}"),
        }
    }

    /// The compare's list of tiles, `tiles`, written as its kernel writes it, into the buffer
    /// `list`.
    fn list(platform: &FakePlatform, list: Option<ObjId>, tiles: &[u32]) {
        let buffer = (platform.world().find(list.unwrap()))
            .and_then(|b| b.cast::<kind::Buffer>())
            .unwrap();
        let words: Vec<u32> = std::iter::once(tiles.len() as u32).chain(tiles.iter().copied()).collect();
        assert!(buffer.length() >= 4 * words.len() as u64);
        // SAFETY: the fake buffer's contents, which no command buffer reads now.
        unsafe {
            let at: *mut u32 = buffer.contents().as_ptr();
            std::ptr::copy_nonoverlapping(words.as_ptr(), at, words.len());
        }
    }

    /// The copies among `calls`: each one's source, its offset, its destination and its length,
    /// its offset there the same.
    fn copies(calls: &[Call]) -> Vec<(Option<ObjId>, u64, Option<ObjId>, u64)> {
        (calls.iter())
            .filter(|c| c.sel == "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:")
            .map(|c| match c.args.as_slice() {
                [Arg::Obj(from), Arg::U64(at), Arg::Obj(to), Arg::U64(there), Arg::U64(length)] => {
                    assert_eq!(at, there, "a copy to another place in the mirror");
                    (*from, *at, *to, *length)
                }
                other => panic!("{other:?}"),
            })
            .collect()
    }

    /// Where the platform gives a queue for copies, the sync's command buffer copies the surface
    /// into `now` and lists the tiles that changed, writing nothing into the mirror; once it has
    /// finished, only the listed tiles are copied out of `now` into the mirror, on the queue for
    /// copies, a run of them in one copy a row, and the draw waits for that copy. The next sync's
    /// compare is committed while the last draw still runs, and its copy only once both have
    /// finished, out of the frame that compare copied. A sync that lists no tile copies nothing,
    /// and the draw waits for nothing
    #[test]
    fn with_a_queue_for_copies_the_listed_tiles_are_copied_there_after_the_compare_and_the_draw() {
        let (platform, mirror, surface) = fake(true);
        let log = platform.call_log();
        let copy_queue = mirror.copies.as_ref().expect("a queue for copies").0.id();
        let mark = log.mark();
        // The first sync: the compare, then the copy of tiles 5 and 4, listed in no order.
        let first = sync_on_a_thread(&platform, &mirror, &surface);
        let compare = committed(&platform, 1)[0];
        let calls = log.since(mark);
        assert_eq!(
            (calls.iter())
                .filter(|c| c.sel.starts_with("copyFromTexture") && c.sel.contains("toBuffer"))
                .count(),
            1
        );
        let buffers = bound(&calls);
        assert_eq!(buffers.len(), 4, "now, sent, the list and the arguments");
        let now = buffers[0];
        list(&platform, buffers[2], &[5, 4]);
        let mark = log.mark();
        complete(&platform, compare);
        let copy = committed(&platform, 1)[0];
        let calls = log.since(mark);
        let queued = (calls.iter()).find(|c| c.sel == "commandBuffer").expect("the copy's buffer");
        assert_eq!(queued.obj, copy_queue, "the copy on the queue for copies");
        let rows: Vec<_> = (64..128).map(|y| (now, y * 1024, 512)).collect();
        let copied = copies(&calls);
        assert!(copied.iter().all(|c| c.2 != now && c.2.is_some()));
        assert_eq!(
            copied.iter().map(|c| (c.0, c.1, c.3)).collect::<Vec<_>>(),
            rows,
            "tiles 4 and 5, the first two of the second row, a copy a pixel row"
        );
        let mirrored = copied[0].2;
        complete(&platform, copy);
        assert_eq!(first.join().unwrap(), Some((1024, false)));
        assert_eq!(mirror.last_sent(), 2);
        // The next sync, while a draw that read the mirror still runs.
        let device = platform.metal_device(None).unwrap();
        let draw = device.new_command_queue().unwrap();
        let reader = no_catch(draw.command_buffer(OnFault::Die)).unwrap();
        no_catch(reader.commit(OnFault::Die));
        mirror.read_by(&reader);
        let reading = committed(&platform, 1)[0];
        let mark = log.mark();
        let second = sync_on_a_thread(&platform, &mirror, &surface);
        let running = committed(&platform, 2);
        assert_eq!(running[0], reading, "the draw still runs as the compare is committed");
        let compare = running[1];
        let buffers = bound(&log.since(mark));
        assert_ne!(buffers[0], now, "the compare copies into the other frame");
        assert_eq!(buffers[1], now, "and compares with the one sent");
        let now = buffers[0];
        list(&platform, buffers[2], &[0]);
        complete(&platform, compare);
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            platform.world().committed(),
            [reading],
            "no copy while the draw runs"
        );
        let mark = log.mark();
        complete(&platform, reading);
        let copy = committed(&platform, 1)[0];
        let rows: Vec<_> = (0..64).map(|y| (now, y * 1024, mirrored, 256)).collect();
        assert_eq!(copies(&log.since(mark)), rows, "tile 0, out of the frame just compared");
        complete(&platform, copy);
        assert_eq!(second.join().unwrap(), Some((1024, false)));
        assert_eq!(mirror.last_sent(), 1);
        assert!(!mirror.last_asked());
        // A sync that lists nothing.
        let mark = log.mark();
        let third = sync_on_a_thread(&platform, &mirror, &surface);
        let compare = committed(&platform, 1)[0];
        list(&platform, bound(&log.since(mark))[2], &[]);
        complete(&platform, compare);
        assert_eq!(third.join().unwrap(), Some((1024, false)));
        assert!(platform.world().committed().is_empty());
        assert!(copies(&log.since(mark)).is_empty());
        assert_eq!(mirror.last_sent(), 0);
        assert!(!mirror.last_asked());
    }

    /// Where the platform gives no queue for copies (the default), the mirror asks for none in
    /// the log, makes no list, and each sync is one command buffer on its own queue that copies
    /// the surface into `now` and runs the tile kernel, which writes the mirror and counts the
    /// tiles, as before the queue for copies: no copy of a buffer, and the sync committed only
    /// once the last draw has finished
    #[test]
    fn with_no_queue_for_copies_a_sync_is_one_command_buffer_that_writes_the_mirror() {
        let (platform, mirror, surface) = fake(false);
        assert!(mirror.copies.is_none());
        let log = platform.call_log();
        assert!(!log.calls().iter().any(|c| c.sel == COPY_QUEUE));
        assert!(
            !log.calls()
                .iter()
                .any(|c| c.args.contains(&Arg::Str(LIST_FUNCTION.to_owned())))
        );
        let device = platform.metal_device(None).unwrap();
        for at in 0..2 {
            let reader = (at == 1).then(|| {
                let draw = device.new_command_queue().unwrap();
                let reader = no_catch(draw.command_buffer(OnFault::Die)).unwrap();
                no_catch(reader.commit(OnFault::Die));
                mirror.read_by(&reader);
                committed(&platform, 1)[0]
            });
            let mark = log.mark();
            let sync = sync_on_a_thread(&platform, &mirror, &surface);
            if let Some(reader) = reader {
                std::thread::sleep(Duration::from_millis(100));
                assert_eq!(platform.world().committed(), [reader], "no sync while the draw runs");
                complete(&platform, reader);
            }
            let made = committed(&platform, 1)[0];
            let calls = log.since(mark);
            // From the sync's command buffer on: the first sync makes the frame's buffers
            // before it.
            let sels: Vec<_> = (calls.iter())
                .map(|c| c.sel)
                .skip_while(|s| *s != "commandBuffer")
                .filter(|s| !s.starts_with("waitUntil"))
                .collect();
            let expected = [
                "commandBuffer",
                "blitCommandEncoder",
                "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
                "endEncoding",
                "computeCommandEncoder",
                "setComputePipelineState:",
                "setBuffers:offsets:withRange:",
                "dispatchThreadgroups:threadsPerThreadgroup:",
                "endEncoding",
                "commit",
            ];
            assert_eq!(sels, expected, "sync {at}");
            let buffers = bound(&calls);
            assert_eq!(buffers.len(), 5, "now, sent, the mirror, the arguments and the count");
            complete(&platform, made);
            assert_eq!(sync.join().unwrap(), Some((1024, false)));
        }
    }

    /// On Vulkan, a mirror made again while an import of the old mirror's pages lives leaves each
    /// of those pages mapped, holding what it held, until the runtime has freed the import: the
    /// old mirror's deallocator runs at the old buffer's release, before that free (the module
    /// doc, "Its pages"). A buffer over the old mirror's first page, made while the mirror lived,
    /// lies in its import and holds the import until it goes
    #[test]
    fn on_vulkan_a_mirror_made_again_keeps_the_old_pages_mapped_while_their_import_lives() {
        let platform = Arc::new(VulkanPlatform::new());
        crate::entry::run(|_| {
            // Whether the page at `at` is mapped, read without touching it.
            let mapped = |at: HostPtr| {
                // SAFETY: `msync` of the page at `at`, which reads no byte and fails where no
                // page is mapped.
                unsafe { libc::msync(at.as_ptr(), 1, libc::MS_ASYNC) == 0 }
            };
            // SAFETY: sysconf reads a constant of the host's.
            let page = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).unwrap();
            let device = platform.metal_device(None).unwrap();
            let mirror = Mirror::<VulkanPlatform>::new(device.clone_ref()).expect("the mirror");
            let surface = |side: u64| {
                let descriptor = platform.texture_descriptor(&TextureDesc {
                    pixel_format: 80,
                    width: side,
                    height: side,
                    usage: 0x1,
                    resource_options: Some(0),
                    ..TextureDesc::default()
                });
                no_catch(device.new_texture(OnFault::Die, &descriptor)).unwrap()
            };
            // The mirror brought up to a `side` by `side` surface, its sync finished: the
            // mirror's first byte and length.
            let sync = |side: u64| {
                let synced = (mirror.sync(&surface(side), 256, &|| platform.clock().now_mono()))
                    .expect("a sync");
                if let Some(completion) = &synced.completion {
                    no_catch(completion.sync.wait_until_completed(OnFault::Die));
                }
                (synced.buffer.contents(), synced.buffer.length())
            };
            // 128 x 128 pixels of 4 bytes: 64 KiB, four pages of 16 KiB or sixteen of 4 KiB.
            let (old, length) = sync(128);
            let pages = length / page;
            assert!(pages > 1, "the old mirror is {length:#x} bytes");
            let at = |p: u64| old.wrapping_add(p * page);
            let mark = |p: u64| 0xa0 | u8::try_from(p & 0xf).unwrap();
            for p in 0..pages {
                // SAFETY: a byte of the old mirror's pages, which no GPU work reaches now.
                unsafe { at(p).as_ptr::<u8>().write_volatile(mark(p)) };
            }
            // SAFETY: the old mirror's first page, mapped while an import of it lives.
            let bytes = unsafe { HostRange::new(old, page) };
            let held = no_catch(device.new_buffer_over_host_memory(OnFault::Die, bytes, 0, None))
                .expect("a buffer over the old mirror's first page");
            // A frame of another size: the mirror is made again, and the old one goes.
            let (new, _) = sync(32);
            assert_ne!(new.addr(), old.addr(), "the mirror made again");
            for p in 0..pages {
                assert!(
                    mapped(at(p)),
                    "page {p} of the old mirror's {pages} stays mapped while the import lives"
                );
                // SAFETY: a byte of the old mirror's pages, mapped (above).
                let byte = unsafe { at(p).as_ptr::<u8>().read_volatile() };
                assert_eq!(byte, mark(p), "page {p} holds what it held");
            }
            drop(held);
            assert!(
                reims_vgpu_vulkan::remap::wait_unimported(
                    NonNull::new(old.as_ptr::<u8>()).unwrap(),
                    length,
                    Duration::from_secs(30)
                ),
                "the runtime freed the import of the old mirror's pages"
            );
        });
    }
}
