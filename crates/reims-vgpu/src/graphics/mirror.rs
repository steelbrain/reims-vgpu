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
    /// Every tile goes at the next sync: the mirror was made since the last.
    fresh: bool,
    /// The last draw that read the mirror, on the compositor's GPU.
    reader: Option<P::CommandBuffer>,
    /// The last sync, where the draw waits for it on its GPU: the next sync waits for it, and
    /// reads its count of tiles.
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

impl<P: ApiPlatform> Mirror<P> {
    /// The mirror on `device`, the runtime's: its queue, the tile kernel's pipeline and the
    /// count of tiles; `None` where the runtime refuses one, logged.
    pub(crate) fn new(device: P::Device) -> Option<Mirror<P>> {
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
            Some((queue, pipeline, tiles))
        })();
        let Some((queue, pipeline, tiles)) = made else {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the runtime refused the window's mirror (its queue, its kernel's pipeline or its \
                 count): the window presents from the runtime's GPU"
            );
            return None;
        };
        Some(Mirror {
            made_at: device.recoveries(),
            last: AtomicU64::new(0),
            device,
            queue,
            pipeline,
            tiles,
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
        if let Some(frame) = kept.as_mut() {
            // The last draw that read the mirror has finished before this sync writes it, or
            // makes it again.
            if let Some(reader) = frame.reader.take()
                && reader.status() != CommandBufferStatus::NOT_ENQUEUED
            {
                no_catch(reader.wait_until_completed(OnFault::Die));
            }
            // And the last sync, where the draw waited for it on its GPU: its count is read here.
            if let Some(last) = frame.pending.take() {
                no_catch(last.wait_until_completed(OnFault::Die));
                let held = pixel_bytes(frame.shape.0).unwrap_or(bytes);
                if !self.finished(&last, held) {
                    return None;
                }
            }
        }
        if kept
            .as_ref()
            .is_none_or(|f| f.shape != (format, width, height))
        {
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
        // SAFETY: shared buffers of 20 and 4 bytes, mapped while they live, which no command
        // buffer reads: the last sync has finished.
        unsafe {
            let at: *mut u32 = frame.arguments.contents().as_ptr();
            std::ptr::copy_nonoverlapping(arguments.as_ptr(), at, arguments.len());
            self.tiles.contents().as_ptr::<u32>().write(0);
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
        no_catch(encoder.encode(
            OnFault::Die,
            ComputeCmd::SetComputePipelineState {
                state: Some(&self.pipeline),
            },
        ));
        no_catch(encoder.encode(
            OnFault::Die,
            ComputeCmd::SetBuffers {
                buffers: &[
                    Some(&frame.now),
                    Some(&frame.sent),
                    Some(&frame.mirror),
                    Some(&frame.arguments),
                    Some(&self.tiles),
                ],
                offsets: &[0; 5],
                range: NsRange {
                    location: 0,
                    length: 5,
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
        let exported = command_buffer.add_completion_sync_file();
        frame.asked = exported;
        no_catch(command_buffer.commit(OnFault::Die));
        // The sync file, where the platform exports one and the sync has not finished already.
        let file = exported
            .then(|| command_buffer.export_completion_sync_file())
            .flatten()
            .flatten();
        // `now` is the frame sent from here.
        std::mem::swap(&mut frame.now, &mut frame.sent);
        frame.fresh = false;
        let completion = match file {
            Some(file) => {
                frame.pending = Some(command_buffer.clone_ref());
                Some(Completion {
                    file,
                    sync: command_buffer,
                })
            }
            None => {
                no_catch(command_buffer.wait_until_completed(OnFault::Die));
                if !self.finished(&command_buffer, bytes) {
                    return None;
                }
                None
            }
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

    /// The sync `sync`, finished, of a frame of `bytes` bytes a pixel: its count of tiles read
    /// and counted; `false`, logged, where it failed.
    fn finished(&self, sync: &P::CommandBuffer, bytes: u64) -> bool {
        if sync.status() != CommandBufferStatus::COMPLETED {
            log::warn!(
                target: "reims_vgpu::graphics",
                "the window's mirror's sync failed ({:?}): the window presents from the \
                 runtime's GPU",
                sync.error()
            );
            return false;
        }
        // SAFETY: the 4-byte shared count, which the finished sync wrote, and no sync since.
        let tiles = u64::from(unsafe { self.tiles.contents().as_ptr::<u32>().read() });
        self.last.store(tiles, Ordering::Relaxed);
        self.count_tiles(tiles, tiles * TILE * TILE * bytes);
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
            assert!(self.finished(&last, bytes), "the mirror's last sync failed");
        }
        self.last.load(Ordering::Relaxed)
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
            Some(Frame {
                shape: (format, width, height),
                bytes_per_row,
                now: private(length)?,
                sent: private(length)?,
                mirror,
                arguments: no_catch(self.device.new_buffer_of_length(OnFault::Die, 20, 0))?,
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
    use reims_vgpu_vulkan::core::platform::os::clock::Clock as _;
    use reims_vgpu_vulkan::core::platform::values::TextureDesc;

    use super::*;

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
            let mirror = Mirror::new(device.clone_ref()).expect("the mirror");
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
