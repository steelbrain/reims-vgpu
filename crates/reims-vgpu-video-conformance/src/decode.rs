//! The decode suite: a stand-in for the guest's decoder decodes an H.264,
//! HEVC, AV1 or VP9 MP4 through the host session, on any backend's
//! decoders, the way a guest's VideoToolbox would, and hashes each emitted
//! frame's planes, packed, as the guest's buffer holds them: NV12, or P010
//! (`x420`) when the host asks for it, as for Main 10, 10-bit AV1 and VP9
//! Profile 2. The hashes are ffmpeg's `framemd5` of the same format, so a
//! reference decode's list checks them ([`crate::suite`]).
//!
//! The stand-in drives a `HostSession` as the guest's paravirtualized
//! decoder does: `decs`, then `+dec` with the stream's codec, as `,cid`
//! the ID of the entry `decs` listed for it, and the guest's info as
//! `,gin`, then `dsta` with the track's `avcC`, `hvcC`, `av1C` or `vpcC`,
//! then one `dfrm` per sample in decode order, cut into 0x200000-byte
//! fragments, each waited for, then `dfin` and `-dec`. A stream whose
//! codec `decs` does not list is no session's: the guest decodes it
//! itself. A second thread plays the guest's handler: it lends a buffer
//! for each `Dbuf`, laid out as `Dpba` asks, its `BytesPerRowAlignment`
//! and `PlaneAlignment` among it, with rows wider than the frame, and takes
//! it back at `Demt`. A third delivers queue 0 in order, as the device
//! does. Asynchronous runs set
//! `kVTDecodeFrame_EnableAsynchronousDecompression` on each frame.
//!
//! The guest's handler answers `Demt` at once and leaves the hashing to a
//! few threads of their own, so that the run's rate is the host's and not
//! the hash's. Its buffers come from a pool, as a guest's pixel buffer pool
//! lends the same ones again; a buffer back in the pool is filled with
//! `POISON` first, so that a frame the host leaves partly unwritten
//! still hashes wrong.
//!
//! The messages, holds and time go to stderr.
//!
//! [`decode_file_described`] gives the format description another size
//! than the stream's, as a guest may. Its buffers are of the description's
//! size, their luma filled with a mark, and each frame hashes the stream's
//! size at the top left, or as much of it as the buffer holds, or a size
//! asked for, and says how much of the luma the host wrote.
//!
//! [`decode_file_sized`] hashes each frame at a size of its own, given by
//! its frame ID, at the buffer's top left: for an AV1 stream whose frames
//! change size, each of which the host writes at its own size into a
//! buffer of the session's.

pub use stand_in::{
    DecodeRun, EmittedFrame, decode_file, decode_file_described, decode_file_keeping_pixels,
    decode_file_sized, set_zero_copy,
};

#[cfg(test)]
mod tests;

mod stand_in {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::mpsc::{self, Sender, SyncSender};
    use std::time::Duration;

    use reims_vgpu_video_codec::{av1, h264, hevc, vp9};
    use reims_vgpu_video_core::backend::Backend;
    #[cfg(not(target_os = "macos"))]
    use reims_vgpu_video_core::device_memory::HostView;
    use reims_vgpu_video_core::device_memory::{DeviceMemory, MemoryIdentity, MemoryLife};
    use reims_vgpu_video_core::fourcc::FourCc;
    use reims_vgpu_video_core::host::{GuestSink, HostSession};
    use reims_vgpu_video_core::plist;
    use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
    use reims_vgpu_video_core::surfaces::{
        GuestSurface, MappingId, PlaneLayout, SurfaceError, SurfaceLayout, Surfaces,
    };
    use reims_vgpu_video_core::sync::Mutex;
    use reims_vgpu_video_core::wire::image_description::{
        AV1_CONFIGURATION, AVC_CONFIGURATION, HEVC_CONFIGURATION, ImageDescription,
        VP9_CONFIGURATION,
    };
    use reims_vgpu_video_core::wire::message::{
        GuestUuid, MediaTime, Message, MessageFlags, MessageWriter, read_header,
    };
    use reims_vgpu_video_core::wire::sample_buffer::{
        MEDIA_TYPE_VIDEO, SampleBuffer, SampleTiming, SerializedFormatDescription, SerializedValue,
    };
    use reims_vgpu_video_core::wire::surface_boxes::read_surface_release;

    #[cfg(not(target_os = "macos"))]
    use crate::guest_pages::GuestPages;
    #[cfg(target_os = "macos")]
    use crate::guest_surface::GuestIoSurface;
    use crate::md5::{Md5, self_test};
    use crate::mp4;

    const fn key(name: &[u8; 4]) -> FourCc {
        FourCc::new(*name)
    }
    const CODEC: FourCc = key(b",cod");
    const VERSION: FourCc = key(b",ver");
    const ERROR: FourCc = key(b",err");
    const FORMAT: FourCc = key(b",fds");
    const POOL: FourCc = key(b",blu");
    const FRAME: FourCc = key(b",frm");
    const FLAGS: FourCc = key(b",dfl");
    const SAMPLE: FourCc = key(b",sbf");
    const INFO: FourCc = key(b",inf");
    const UUID: FourCc = key(b",uui");
    const UUIDS: FourCc = key(b",ui+");
    const ATTRIBUTES: FourCc = key(b",pba");
    const DECODERS: FourCc = key(b",dec");
    const CODEC_ID: FourCc = key(b",cid");
    const GUEST_INFO: FourCc = key(b",gin");
    const FRAGMENT_LIMIT: usize = 0x20_0000;
    /// `kVTParavirtualizationGuestInfoKey_BuildVersion`: the key of the
    /// guest's build in `+dec`'s `,gin`.
    const GUEST_BUILD_VERSION_KEY: &str = "VTGuestBuildVersion";
    /// The guest build the stand-in names there: the x86_64 26.7.1 guest's
    /// (docs/video/devices.md, "The decoder session"). Core reads `,gin` as
    /// a dictionary, and nothing in it.
    const GUEST_BUILD_VERSION: &str = "25G241";
    /// Padding past each row of the guest's buffers, before the row is
    /// rounded up to `Dpba`'s `BytesPerRowAlignment`: a copy at the wrong
    /// stride shows. 256 bytes, so that at the 64 every backend's `Dpba`
    /// asks, rows a 256-byte multiple wide stay one, as radeonsi imports
    /// them, and a buffer a guest's rows would let a device write, this
    /// one's do.
    const ROW_PADDING: usize = 256;
    /// What a buffer back in the pool is filled with before it is lent
    /// again.
    const POISON: u8 = 0xa5;

    /// The guest's IOSurfaces by mapping ID, and the host's holds on them.
    #[derive(Default)]
    struct GuestMemory {
        surfaces: Mutex<HashMap<u64, Arc<Buffer>>>,
        holds: Arc<AtomicUsize>,
        most_held: AtomicUsize,
    }

    /// One buffer the guest lent the host, for one `Dbuf`: its layout and
    /// its storage, which goes back to the pool when the host has let it go
    /// and the frame in it is hashed.
    struct Buffer {
        layout: SurfaceLayout,
        storage: Arc<Storage>,
        /// The pool its storage goes back to.
        pool: Arc<Mutex<Vec<Arc<Storage>>>>,
    }

    impl Drop for Buffer {
        fn drop(&mut self) {
            self.pool.lock().push(self.storage.clone());
        }
    }

    impl Buffer {
        fn bytes(&self) -> Bytes<'_> {
            self.storage.bytes()
        }
    }

    /// One of the guest's pixel buffers, lent again and again under one
    /// mapping ID, as a guest's pool lends its IOSurfaces: the guest maps
    /// each once, and the host's mapping of it lives as long as it does.
    /// Its pages lie as a guest's do where the host can have them so
    /// (`GuestPages`, off macOS), and are lent to a backend's device then.
    struct Storage {
        mapping_id: u64,
        len: usize,
        /// On macOS, the layout its IOSurface was made at, which a buffer
        /// takes it again only for.
        layout: Option<SurfaceLayout>,
        memory: Mutex<Memory>,
        life: MemoryLife,
    }

    impl Storage {
        /// Its bytes, until the result drops: an IOSurface's locked for the
        /// CPU while it lives.
        #[cfg_attr(
            target_os = "macos",
            expect(
                clippy::disallowed_methods,
                reason = "the stand-in guest ends its run where its own buffer cannot be read: \
                          a panic on its handler's thread would leave the run to time out"
            )
        )]
        fn bytes(&self) -> Bytes<'_> {
            let memory = self.memory.lock();
            #[cfg(target_os = "macos")]
            let surface = match &*memory {
                Memory::Surface(surface) => Some(surface.lock().unwrap_or_else(|why| {
                    eprintln!("the guest's buffer: {why}");
                    std::process::abort()
                })),
                _ => None,
            };
            Bytes {
                #[cfg(target_os = "macos")]
                surface,
                memory,
            }
        }

        /// Whether a device may be lent its memory: all but plain memory.
        fn lent_to_devices(&self) -> bool {
            !matches!(*self.memory.lock(), Memory::Plain(_))
        }
    }

    /// A buffer's bytes: in pages as a guest's lie on Linux, in an
    /// IOSurface laid out as a guest's on macOS, or plain memory.
    enum Memory {
        #[cfg(not(target_os = "macos"))]
        Pages(GuestPages),
        #[cfg(target_os = "macos")]
        Surface(GuestIoSurface),
        Plain(Vec<u8>),
    }

    /// A buffer's bytes as the stand-in reaches them ([`Storage::bytes`]).
    struct Bytes<'a> {
        /// An IOSurface's lock, dropped before the memory's.
        #[cfg(target_os = "macos")]
        surface: Option<crate::guest_surface::Locked>,
        memory: reims_vgpu_video_core::sync::MutexGuard<'a, Memory>,
    }

    impl std::ops::Deref for Bytes<'_> {
        type Target = [u8];
        fn deref(&self) -> &[u8] {
            #[cfg(target_os = "macos")]
            if let Some(surface) = &self.surface {
                return surface.as_bytes();
            }
            match &*self.memory {
                #[cfg(not(target_os = "macos"))]
                Memory::Pages(pages) => pages.as_bytes(),
                Memory::Plain(bytes) => bytes,
                #[cfg(target_os = "macos")]
                Memory::Surface(_) => &[],
            }
        }
    }

    impl std::ops::DerefMut for Bytes<'_> {
        fn deref_mut(&mut self) -> &mut [u8] {
            #[cfg(target_os = "macos")]
            if let Some(surface) = &mut self.surface {
                return surface.bytes();
            }
            match &mut *self.memory {
                #[cfg(not(target_os = "macos"))]
                Memory::Pages(pages) => pages.bytes(),
                Memory::Plain(bytes) => bytes,
                #[cfg(target_os = "macos")]
                Memory::Surface(_) => &mut [],
            }
        }
    }

    /// A held buffer's memory as the stand-in host lends it to a device:
    /// its pages' view, and a dma-buf of them, as reims-vgpu lends a
    /// guest's.
    struct StandInDevice {
        storage: Arc<Storage>,
    }

    impl DeviceMemory for StandInDevice {
        fn identity(&self) -> MemoryIdentity {
            self.storage.life.identity()
        }

        fn length(&self) -> usize {
            self.storage.len
        }

        #[cfg(target_os = "linux")]
        fn dma_buf_of(
            &self,
            ranges: &[(usize, usize)],
        ) -> Result<
            reims_vgpu_video_core::device_memory::SurfaceDmaBuf,
            reims_vgpu_video_core::device_memory::DmaBufError,
        > {
            let memory = self.storage.memory.lock();
            let Memory::Pages(pages) = &*memory else {
                return Err(reims_vgpu_video_core::device_memory::DmaBufError::Refused(
                    "plain memory".into(),
                ));
            };
            pages.dma_buf_of(ranges)
        }

        #[cfg(not(target_os = "macos"))]
        fn host_view(&self) -> Result<HostView, SurfaceError> {
            let memory = self.storage.memory.lock();
            let Memory::Pages(pages) = &*memory else {
                return Err(SurfaceError::Unreachable("plain memory".into()));
            };
            let (first, size) = pages.view();
            Ok(HostView {
                first,
                size,
                offset: 0,
                hold: Box::new(self.storage.clone()),
            })
        }

        /// The buffer's IOSurface, writable, as reims-vgpu lends a guest's
        /// that is not read-only for its clients.
        #[cfg(target_os = "macos")]
        fn io_surface(
            &self,
        ) -> Result<reims_vgpu_video_core::device_memory::LentIoSurface<'_>, SurfaceError> {
            let memory = self.storage.memory.lock();
            let Memory::Surface(surface) = &*memory else {
                return Err(SurfaceError::Unreachable("no IOSurface".into()));
            };
            // SAFETY: the surface lives as long as its storage, which this
            // holds, and is never replaced.
            Ok(unsafe {
                reims_vgpu_video_core::device_memory::LentIoSurface::new(surface.as_ptr(), true)
            })
        }
    }

    /// A frame emitted with an image, for the hashing threads: its place
    /// among the frames emitted, its frame ID and its buffer.
    type ToHash = (usize, i64, Arc<Buffer>);

    struct Held {
        buffer: Arc<Buffer>,
        holds: Arc<AtomicUsize>,
        device: StandInDevice,
    }

    impl GuestSurface for Held {
        fn layout(&self) -> SurfaceLayout {
            self.buffer.layout.clone()
        }

        fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
            write(&mut self.buffer.bytes());
            Ok(())
        }

        fn device_memory(&self) -> Option<&dyn DeviceMemory> {
            (self.buffer.storage.lent_to_devices()).then_some(&self.device as &dyn DeviceMemory)
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            self.holds.fetch_sub(1, Ordering::SeqCst);
        }
    }

    impl Surfaces for GuestMemory {
        fn retain(&self, mapping_id: MappingId) -> Result<Box<dyn GuestSurface>, SurfaceError> {
            let buffer = self
                .surfaces
                .lock()
                .get(&mapping_id.0)
                .cloned()
                .ok_or(SurfaceError::Unknown(mapping_id.0))?;
            let held = self.holds.fetch_add(1, Ordering::SeqCst) + 1;
            self.most_held.fetch_max(held, Ordering::SeqCst);
            Ok(Box::new(Held {
                device: StandInDevice {
                    storage: buffer.storage.clone(),
                },
                buffer,
                holds: self.holds.clone(),
            }))
        }
    }

    struct ChannelSink(Mutex<Sender<Vec<u8>>>);

    impl GuestSink for ChannelSink {
        fn send(&self, message: Vec<u8>) {
            let _ = self.0.lock().send(message);
        }
    }

    /// What the guest's handler keeps and counts.
    struct Guest {
        /// The stream's size, which the hashes take at the top left.
        width: u32,
        height: u32,
        /// The format description's size, and so the buffers' at first.
        described: (u32, u32),
        /// The size the last `Dpba` asked the buffers to have: `dsta`'s
        /// gives the description's, and a later one an AV1 frame's own.
        pool_size: Mutex<(u32, u32)>,
        /// The last `Dpba`'s `BytesPerRowAlignment` and `PlaneAlignment`,
        /// in bytes: each row is rounded up to the first, and the chroma
        /// plane's offset to the second. A key the `Dpba` leaves out, or
        /// one of 0, asks for none (1).
        alignment: Mutex<Alignment>,
        /// The size at the top left that the MD5s take: the stream's, or
        /// as much of it as the buffers hold, unless asked for another.
        hashed: (u32, u32),
        /// Where given, the size each frame ID's MD5 takes at the top left
        /// in place of `hashed`, or as much of it as its buffer holds.
        sizes: Option<HashMap<i64, (u32, u32)>>,
        memory: Arc<GuestMemory>,
        lent: Mutex<HashMap<GuestUuid, u64>>,
        waiting: Mutex<HashMap<u64, Sender<Vec<u8>>>>,
        next_surface: AtomicU64,
        emitted: AtomicUsize,
        failed: AtomicUsize,
        released: AtomicUsize,
        counts: Mutex<HashMap<String, usize>>,
        /// The host's messages in the order they came ([`DecodeRun::sequence`]).
        sequence: Mutex<Vec<String>>,
        frames: Mutex<Vec<EmittedFrame>>,
        /// Whether each frame keeps its packed planes.
        keep_pixels: bool,
        /// The buffers, poisoned, to lend again.
        pool: Arc<Mutex<Vec<Arc<Storage>>>>,
        /// The next `Dbuf`'s pixel buffer, for its UUID.
        next_buffer: AtomicU64,
        /// Each lent buffer's mapping ID's lend, the `Dbuf`'s count from 1,
        /// which [`DecodeRun::sequence`] names a `rels` by: the same in
        /// every run, whichever of the pool's buffers each lend took.
        lends: Mutex<HashMap<u64, u64>>,
        /// The hashing threads' queue, closed at the end of the run.
        to_hash: Mutex<Option<SyncSender<ToHash>>>,
    }

    /// One frame the host emitted with an image.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct EmittedFrame {
        /// Its place among the frames emitted, from 0.
        pub index: usize,
        /// The frame ID of the sample it was decoded from: the sample's
        /// index in decode order.
        pub frame_id: i64,
        /// The MD5 of its planes, packed.
        pub md5: String,
        /// Its planes, packed as the MD5 takes them: kept only by
        /// [`decode_file_keeping_pixels`].
        pub pixels: Option<Vec<u8>>,
        /// From [`decode_file_described`] with another size: the columns
        /// and rows of the luma the host wrote, up to the last sample that
        /// is not the mark.
        pub written: Option<(usize, usize)>,
    }

    /// What one run of a stream gave.
    #[derive(Clone, Debug)]
    pub struct DecodeRun {
        /// The frames emitted with an image, in the order they came.
        pub frames: Vec<EmittedFrame>,
        /// The samples sent.
        pub samples: usize,
        /// The frames emitted without an image.
        pub failed: usize,
        /// `dfin`'s status.
        pub finished: i32,
        /// From the first `dfrm` to `dfin`'s reply.
        pub elapsed: Duration,
        /// Surfaces the host's `rels` named.
        pub released: usize,
        /// The most surfaces the host held at once.
        pub most_held: usize,
        /// The host's messages, by type.
        pub counts: HashMap<String, usize>,
        /// The host's messages in the order they came, one line each: its
        /// type, a request's frame ID, the buffer a `Dbuf` was lent, a
        /// `Demt`'s status, info flags and buffer, the buffers a `Dfub`
        /// gives back, and a `Dpba`'s size, as `Dpba 640x360`, `Dbuf 3 -> 7`,
        /// `reply dfrm`, `Demt 3 0 0x1 7`, `Dfub [7]`, `rels [3]`, then
        /// `-dec` where the session's end begins. Buffers are named by the
        /// `Dbuf` that lent them, counted from 1 in the order lent: the
        /// stand-in's own mapping IDs are its pool's, which lends a buffer
        /// again under the same one, as a guest's pool does. Two runs of a
        /// stream on one backend give the same sequence, so a change to the
        /// host session that must not show to the guest is checked by
        /// comparing them.
        pub sequence: Vec<String>,
    }

    impl DecodeRun {
        /// The run in one line, as the harness's log has it.
        pub fn summary(&self) -> String {
            let emitted = self.frames.len();
            format!(
                "{} samples, {emitted} frames emitted, {} without an image, dfin {}, \
                 {:.3} s ({:.1} fps), {} surfaces released, at most {} held at once, \
                 messages {:?}",
                self.samples,
                self.failed,
                self.finished,
                self.elapsed.as_secs_f64(),
                emitted as f64 / self.elapsed.as_secs_f64().max(f64::MIN_POSITIVE),
                self.released,
                self.most_held,
                self.counts,
            )
        }
    }

    impl Guest {
        /// Whether the description's size is not the stream's.
        fn marked(&self) -> bool {
            self.described != (self.width, self.height)
        }

        /// A buffer as the last `Dpba` asks ([`buffer_layout`]). Where
        /// that is not the stream's size, the luma is [`LUMA_MARK`].
        fn buffer(&self, format: FourCc) -> Buffer {
            let (layout, len) =
                buffer_layout(format, *self.pool_size.lock(), *self.alignment.lock());
            let storage = self.pooled(&layout, len);
            if self.marked() {
                storage.bytes()[..layout.planes[1].offset].fill(LUMA_MARK);
            }
            Buffer {
                layout,
                storage,
                pool: self.pool.clone(),
            }
        }

        /// The columns and rows of `buffer`'s luma up to the last sample
        /// that is not [`LUMA_MARK`] (both bytes, for 16-bit samples).
        fn written(&self, buffer: &Buffer) -> (usize, usize) {
            let bytes = buffer.bytes();
            let sample = bytes_per_sample(buffer.layout.pixel_format);
            let plane = &buffer.layout.planes[0];
            let width = usize::try_from(plane.width).unwrap_or(0);
            let (mut columns, mut rows) = (0, 0);
            for row in 0..usize::try_from(plane.height).unwrap_or(0) {
                let start = plane.offset + row * plane.bytes_per_row;
                let line = bytes.get(start..start + width * sample).unwrap_or_default();
                if let Some(at) = line
                    .chunks(sample)
                    .rposition(|s| s.iter().any(|&b| b != LUMA_MARK))
                {
                    columns = columns.max(at + 1);
                    rows = row + 1;
                }
            }
            (columns, rows)
        }

        /// A buffer of `len` bytes for `layout` from the pool, or a new one
        /// under a mapping ID of its own, all [`POISON`]: in pages as a
        /// guest's lie on Linux, in an IOSurface laid out as `layout` on
        /// macOS, where the host can have them so, else plain memory.
        fn pooled(&self, layout: &SurfaceLayout, len: usize) -> Arc<Storage> {
            let mut pool = self.pool.lock();
            if let Some(at) = pool.iter().position(|storage| {
                storage.len == len && storage.layout.as_ref().is_none_or(|made| made == layout)
            }) {
                return pool.swap_remove(at);
            }
            drop(pool);
            let said = |why: String| {
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    eprintln!("the guest's buffers are plain memory, lent to no device: {why}");
                });
            };
            let mut made_at = None;
            #[cfg(target_os = "macos")]
            let memory = match GuestIoSurface::new(layout, len) {
                Ok(surface) => {
                    made_at = Some(layout.clone());
                    Memory::Surface(surface)
                }
                Err(why) => {
                    said(why);
                    Memory::Plain(vec![POISON; len])
                }
            };
            #[cfg(not(target_os = "macos"))]
            let memory = match GuestPages::new(len) {
                Ok(pages) => Memory::Pages(pages),
                Err(why) => {
                    said(why);
                    Memory::Plain(vec![POISON; len])
                }
            };
            let storage = Arc::new(Storage {
                mapping_id: self.next_surface.fetch_add(1, Ordering::SeqCst),
                len,
                layout: made_at.take(),
                memory: Mutex::new(memory),
                life: MemoryLife::new(),
            });
            storage.bytes().fill(POISON);
            storage
        }

        /// A hashing thread: each frame's MD5, its pixels where the run
        /// keeps them, and what the host wrote of a marked buffer; then its
        /// buffer is poisoned for the pool.
        fn hash(&self, (index, frame_id, buffer): ToHash) {
            let (width, height) = (self.sizes.as_ref())
                .and_then(|sizes| sizes.get(&frame_id).copied())
                .unwrap_or(self.hashed);
            let size = (
                width.min(buffer.layout.width),
                height.min(buffer.layout.height),
            );
            let md5 = self.md5(&buffer, size);
            let pixels = self.keep_pixels.then(|| self.packed(&buffer));
            let written = self.marked().then(|| self.written(&buffer));
            self.frames.lock().push(EmittedFrame {
                index,
                frame_id,
                md5,
                pixels,
                written,
            });
            buffer.bytes().fill(POISON);
        }

        fn count(&self, what: &str) {
            *self.counts.lock().entry(what.to_owned()).or_default() += 1;
        }

        /// The guest's handler for one host message.
        fn handle(&self, message: &[u8], queue0: &Sender<Vec<u8>>, format: &Mutex<FourCc>) {
            if let Some((ids, _)) = read_surface_release(message) {
                self.count("rels");
                let lends = self.lends.lock();
                let named: Vec<u64> = (ids.iter())
                    .map(|id| lends.get(&id.0).copied().unwrap_or(id.0))
                    .collect();
                drop(lends);
                self.sequence.lock().push(format!("rels {named:?}"));
                self.released.fetch_add(ids.len(), Ordering::SeqCst);
                for id in ids {
                    self.memory.surfaces.lock().remove(&id.0);
                }
                return;
            }
            let Ok(header) = read_header(message, 0) else {
                self.count("unreadable");
                return;
            };
            if header.flags.contains(MessageFlags::REPLY_TO_GUEST) {
                self.sequence
                    .lock()
                    .push(format!("reply {}", header.message_type));
                if let Some(waiter) = self.waiting.lock().remove(&header.reply_id) {
                    let _ = waiter.send(message.to_vec());
                }
                return;
            }
            self.count(&header.message_type.to_string());
            let Ok(parsed) = Message::parse(message) else {
                self.sequence
                    .lock()
                    .push(format!("{} unreadable", header.message_type));
                return;
            };
            let mut line = header.message_type.to_string();
            if let Some(frame) = parsed.s64(FRAME).ok().flatten() {
                line.push_str(&format!(" {frame}"));
            }
            // The buffers each names, by the `Dbuf` that lent them, which
            // each buffer's UUID carries.
            let mapping_id_of = |uuid: GuestUuid| {
                let mut id = [0; 8];
                id.copy_from_slice(&uuid.0[..8]);
                u64::from_be_bytes(id)
            };
            let lent_id = (&header.message_type.to_be_bytes() == b"Dbuf")
                .then(|| self.next_buffer.fetch_add(1, Ordering::SeqCst));
            match &header.message_type.to_be_bytes() {
                b"Dbuf" => line.push_str(&format!(" -> {}", lent_id.unwrap_or(0))),
                b"Demt" => {
                    let status = parsed.s32(ERROR).ok().flatten().unwrap_or(-1);
                    let info = parsed.u32(INFO).ok().flatten().unwrap_or(0);
                    let buffer = parsed.uuid(UUID).ok().flatten();
                    let buffer = buffer.map_or("-".into(), |uuid| mapping_id_of(uuid).to_string());
                    line.push_str(&format!(" {status} {info:#x} {buffer}"));
                }
                b"Dfub" => {
                    let uuids = parsed.uuid_array(UUIDS).ok().flatten().unwrap_or_default();
                    let ids: Vec<u64> = uuids.into_iter().map(mapping_id_of).collect();
                    line.push_str(&format!(" {ids:?}"));
                }
                b"Dpba" => {
                    let attributes = parsed.dictionary(ATTRIBUTES).ok().flatten();
                    let read = |key| {
                        (attributes.as_ref())
                            .and_then(|a| a.get(key))
                            .and_then(plist::Value::as_unsigned_integer)
                            .unwrap_or(0)
                    };
                    line.push_str(&format!(" {}x{}", read("Width"), read("Height")));
                }
                _ => {}
            }
            self.sequence.lock().push(line);
            let reply = |status: i32| {
                let mut reply = MessageWriter::reply_to(&header, MessageFlags::REPLY_TO_HOST);
                reply.s32(ERROR, status);
                reply
            };
            match &header.message_type.to_be_bytes() {
                b"Dpba" => {
                    let attributes = parsed.dictionary(ATTRIBUTES).ok().flatten();
                    let pixel_format = attributes
                        .as_ref()
                        .and_then(|a| a.get("PixelFormatType"))
                        .and_then(|f| f.as_array())
                        .and_then(|f| f.first())
                        .and_then(|f| f.as_unsigned_integer())
                        .and_then(|f| u32::try_from(f).ok());
                    if let Some(pixel_format) = pixel_format {
                        *format.lock() = FourCc(pixel_format);
                    }
                    let dimension = |key| {
                        (attributes.as_ref())
                            .and_then(|a| a.get(key))
                            .and_then(plist::Value::as_unsigned_integer)
                            .and_then(|v| u32::try_from(v).ok())
                    };
                    if let (Some(width), Some(height)) = (dimension("Width"), dimension("Height")) {
                        *self.pool_size.lock() = (width, height);
                    }
                    let alignment = |key| {
                        (attributes.as_ref())
                            .and_then(|a| a.get(key))
                            .and_then(plist::Value::as_unsigned_integer)
                            .and_then(|v| usize::try_from(v).ok())
                            .unwrap_or(1)
                            .max(1)
                    };
                    *self.alignment.lock() = Alignment {
                        bytes_per_row: alignment("BytesPerRowAlignment"),
                        plane: alignment("PlaneAlignment"),
                    };
                    eprintln!("Dpba: {attributes:?}");
                }
                b"Dbuf" => {
                    let buffer = self.buffer(*format.lock());
                    let mapping_id = buffer.storage.mapping_id;
                    let mut uuid = [0; 16];
                    let pixel_buffer = lent_id.unwrap_or(0);
                    self.lends.lock().insert(mapping_id, pixel_buffer);
                    uuid[..8].copy_from_slice(&pixel_buffer.to_be_bytes());
                    let uuid = GuestUuid(uuid);
                    self.memory
                        .surfaces
                        .lock()
                        .insert(mapping_id, Arc::new(buffer));
                    self.lent.lock().insert(uuid, mapping_id);
                    let mut reply = reply(0);
                    reply.uuid(UUID, uuid);
                    let mut bytes = vec![0, 0, 0, 0x30];
                    bytes.extend_from_slice(b"subs");
                    bytes.extend_from_slice(&mapping_id.to_be_bytes());
                    bytes.extend_from_slice(&[0; 32]);
                    bytes.extend_from_slice(&reply.finish());
                    let _ = queue0.send(bytes);
                }
                b"Demt" => {
                    let frame = parsed.s64(FRAME).ok().flatten().unwrap_or(-1);
                    let status = parsed.s32(ERROR).ok().flatten().unwrap_or(-1);
                    let uuid = parsed.uuid(UUID).ok().flatten();
                    let lent = uuid.and_then(|uuid| self.lent.lock().remove(&uuid));
                    if uuid.is_some() && lent.is_none() {
                        // The guest emits a frame of a buffer it does not
                        // know with an error, and does not reply.
                        self.failed.fetch_add(1, Ordering::SeqCst);
                        eprintln!(
                            "frame {frame}: status {status}, a buffer the guest did not lend, \
                             no image and no reply"
                        );
                        return;
                    }
                    let buffer = lent.and_then(|id| self.memory.surfaces.lock().get(&id).cloned());
                    match buffer {
                        Some(buffer) if status == 0 => {
                            let index = self.emitted.fetch_add(1, Ordering::SeqCst);
                            let to_hash = self.to_hash.lock().clone();
                            if let Some(to_hash) = to_hash {
                                let _ = to_hash.send((index, frame, buffer));
                            }
                        }
                        _ => {
                            self.failed.fetch_add(1, Ordering::SeqCst);
                            let info = parsed.u32(INFO).ok().flatten().unwrap_or(0);
                            eprintln!("frame {frame}: status {status}, info {info:#x}, no image");
                        }
                    }
                    let _ = queue0.send(reply(0).finish());
                }
                b"Dfub" => {
                    for uuid in parsed.uuid_array(UUIDS).ok().flatten().unwrap_or_default() {
                        self.lent.lock().remove(&uuid);
                    }
                }
                _ => {}
            }
        }

        /// The frame's planes, packed, as [`Guest::md5`] hashes them.
        fn packed(&self, buffer: &Buffer) -> Vec<u8> {
            let bytes = buffer.bytes();
            let sample = bytes_per_sample(buffer.layout.pixel_format);
            let width = usize::try_from(self.width).unwrap_or(0);
            let row_bytes = [width * sample, width.div_ceil(2) * 2 * sample];
            let mut out = Vec::new();
            for ((plane, rows), row_bytes) in (buffer.layout.planes.iter())
                .zip([self.height, self.height.div_ceil(2)])
                .zip(row_bytes)
            {
                for row in 0..usize::try_from(rows).unwrap_or(0) {
                    let start = plane.offset + row * plane.bytes_per_row;
                    out.extend_from_slice(bytes.get(start..start + row_bytes).unwrap_or_default());
                }
            }
            out
        }

        /// The frame's planes, packed, as `ffmpeg -pix_fmt nv12 -f
        /// framemd5` (or `p010le`) hashes them, `(width, height)` of them
        /// at the top left.
        fn md5(&self, buffer: &Buffer, (width, height): (u32, u32)) -> String {
            let bytes = buffer.bytes();
            let sample = bytes_per_sample(buffer.layout.pixel_format);
            let width = usize::try_from(width).unwrap_or(0);
            // A chroma row holds a Cb/Cr pair for every two luma samples,
            // the last one too when the width is odd.
            let row_bytes = [width * sample, width.div_ceil(2) * 2 * sample];
            let mut md5 = Md5::new();
            let planes = &buffer.layout.planes;
            for ((plane, rows), row_bytes) in planes
                .iter()
                .zip([height, height.div_ceil(2)])
                .zip(row_bytes)
            {
                for row in 0..usize::try_from(rows).unwrap_or(0) {
                    let start = plane.offset + row * plane.bytes_per_row;
                    md5.update(bytes.get(start..start + row_bytes).unwrap_or_default());
                }
            }
            md5.hex()
        }
    }

    /// The alignments a `Dpba` asks of the guest's buffers, in bytes.
    #[derive(Clone, Copy)]
    pub(super) struct Alignment {
        /// `BytesPerRowAlignment`: each plane's bytes a row.
        pub(super) bytes_per_row: usize,
        /// `PlaneAlignment`: each plane's offset.
        pub(super) plane: usize,
    }

    /// No alignment, before a `Dpba` asks for one.
    const UNALIGNED: Alignment = Alignment {
        bytes_per_row: 1,
        plane: 1,
    };

    /// The layout of a buffer of `(width, height)` in `format`, 4:2:0, and
    /// its length: each plane's rows `ROW_PADDING` past its samples, rounded
    /// up to `alignment.bytes_per_row`, and the chroma plane at the luma's
    /// end rounded up to `alignment.plane`.
    pub(super) fn buffer_layout(
        format: FourCc,
        (width, height): (u32, u32),
        alignment: Alignment,
    ) -> (SurfaceLayout, usize) {
        let row = usize::try_from(width).unwrap_or(0) * bytes_per_sample(format);
        let rows = usize::try_from(height).unwrap_or(0);
        let bytes_per_row = (row + ROW_PADDING).next_multiple_of(alignment.bytes_per_row);
        let chroma_offset = (bytes_per_row * rows).next_multiple_of(alignment.plane);
        let layout = SurfaceLayout {
            pixel_format: format,
            width,
            height,
            planes: vec![
                PlaneLayout {
                    offset: 0,
                    bytes_per_row,
                    width,
                    height,
                },
                PlaneLayout {
                    offset: chroma_offset,
                    bytes_per_row,
                    width: width.div_ceil(2),
                    height: height.div_ceil(2),
                },
            ],
            bytes_per_row: 0,
        };
        (layout, chroma_offset + bytes_per_row * rows.div_ceil(2))
    }

    /// The `CMClassImplementationID` of the first entry of a `decs` reply
    /// whose `VTCodecType` lists `codec`, as the guest registers each entry
    /// for its codec types and sends the ID back in `+dec`.
    pub(super) fn listed_decoder(reply: &[u8], codec: [u8; 4]) -> Result<String, String> {
        let codec = String::from_utf8_lossy(&codec).into_owned();
        let decoders = Message::parse(reply)
            .and_then(|message| message.property_list(DECODERS))
            .map_err(|e| format!("decs: {e}"))?;
        let entries = (decoders.as_ref())
            .and_then(plist::Value::as_array)
            .ok_or("decs: no ,dec array")?;
        entries
            .iter()
            .filter_map(plist::Value::as_dictionary)
            .find(|entry| {
                (entry.get("VTCodecType"))
                    .and_then(plist::Value::as_array)
                    .is_some_and(|types| types.iter().any(|t| t.as_string() == Some(&codec)))
            })
            .and_then(|entry| entry.get("CMClassImplementationID"))
            .and_then(plist::Value::as_string)
            .map(str::to_owned)
            .ok_or_else(|| format!("decs lists no decoder for {codec}"))
    }

    /// The bytes of one sample of a buffer in `format`: 2 for the 16-bit
    /// `x420` and `xf20`, else 1.
    fn bytes_per_sample(format: FourCc) -> usize {
        if format == FourCc::new(*b"x420") || format == FourCc::new(*b"xf20") {
            2
        } else {
            1
        }
    }

    /// The track's codec as the guest names it, its record's atom, and the
    /// visible size its first SPS gives (for AV1 and VP9, its sample
    /// entry's size, as CoreMedia's description takes it).
    struct Stream {
        codec: [u8; 4],
        atom: [u8; 4],
        record: Vec<u8>,
        width: u32,
        height: u32,
    }

    fn stream(file: &[u8]) -> Result<(mp4::Track, Stream), String> {
        if let Ok(track) = mp4::h264_track(file) {
            let record =
                h264::avcc::DecoderConfiguration::parse(&track.avcc).map_err(|e| e.to_string())?;
            let sps = record
                .sequence_parameter_sets
                .first()
                .ok_or("the avcC has no SPS")?;
            let (_, _, width, height) = h264::sps::Sps::parse(sps)
                .map_err(|e| e.to_string())?
                .crop_rect();
            let record = track.avcc.clone();
            return Ok((
                track,
                Stream {
                    codec: *b"avc1",
                    atom: *b"avcC",
                    record,
                    width,
                    height,
                },
            ));
        }
        if let Ok(track) = mp4::av1_track(file) {
            av1::av1c::ConfigurationRecord::parse(&track.avcc).map_err(|e| e.to_string())?;
            let ((width, height), record) = (track.size, track.avcc.clone());
            return Ok((
                track,
                Stream {
                    codec: *b"av01",
                    atom: *b"av1C",
                    record,
                    width,
                    height,
                },
            ));
        }
        if let Ok(track) = mp4::vp9_track(file) {
            vp9::vpcc::ConfigurationRecord::parse(&track.avcc).map_err(|e| e.to_string())?;
            let ((width, height), record) = (track.size, track.avcc.clone());
            return Ok((
                track,
                Stream {
                    codec: *b"vp09",
                    atom: *b"vpcC",
                    record,
                    width,
                    height,
                },
            ));
        }
        let track = mp4::hevc_track(file)?;
        let record =
            hevc::hvcc::DecoderConfiguration::parse(&track.avcc).map_err(|e| e.to_string())?;
        let sps = record
            .nal_units
            .iter()
            .find(|unit| {
                hevc::nal::NalUnit::new(unit)
                    .is_ok_and(|nal| nal.unit_type() == hevc::nal::nal_type::SPS)
            })
            .ok_or("the hvcC has no SPS")?;
        let (_, _, width, height) = hevc::sps::Sps::parse(sps)
            .map_err(|e| e.to_string())?
            .crop_rect();
        let record = track.avcc.clone();
        Ok((
            track,
            Stream {
                codec: *b"hvc1",
                atom: *b"hvcC",
                record,
                width,
                height,
            },
        ))
    }

    fn format_description(
        stream: &Stream,
        (width, height): (u32, u32),
    ) -> SerializedFormatDescription {
        SerializedFormatDescription {
            media_type: MEDIA_TYPE_VIDEO,
            codec: FourCc::new(stream.codec),
            width: i32::try_from(width).unwrap_or(0),
            height: i32::try_from(height).unwrap_or(0),
            extensions: Some(vec![(
                "SampleDescriptionExtensionAtoms".into(),
                SerializedValue::Dictionary(vec![(
                    String::from_utf8_lossy(&stream.atom).into_owned(),
                    SerializedValue::PropertyList(plist::Value::Data(stream.record.clone())),
                )]),
            )]),
        }
    }

    /// Whether the host sessions of the decodes from now on let the
    /// backend's device write each frame into the guest's buffer itself
    /// (`HostSession::set_zero_copy`): on, as in the product, unless
    /// [`set_zero_copy`] turned it off.
    static ZERO_COPY: AtomicBool = AtomicBool::new(true);

    /// Turns zero copy on or off for the host sessions of the decodes from
    /// now on: off, the CPU copies every frame, to compare the two
    /// (`framemd5 --no-zero-copy`).
    pub fn set_zero_copy(on: bool) {
        ZERO_COPY.store(on, Ordering::Relaxed);
    }

    /// Decodes the MP4 `file` through a host session on `backend`'s
    /// decoders, each frame synchronously or, with `asynchronous`,
    /// asynchronously, and returns what the guest got. An error is a
    /// stream that does not read, a message the host refused, or a
    /// surface still held at the end.
    pub fn decode_file(
        backend: &Backend,
        file: &[u8],
        asynchronous: bool,
    ) -> Result<DecodeRun, String> {
        decode(backend, file, asynchronous, false, None, None, None)
    }

    /// [`decode_file`] with each frame's MD5 taken at the size `sizes`
    /// gives its frame ID, at the buffer's top left, or as much of it as
    /// the buffer holds; a frame ID it does not name takes the stream's
    /// size. For a stream whose frames change size.
    pub fn decode_file_sized(
        backend: &Backend,
        file: &[u8],
        asynchronous: bool,
        sizes: HashMap<i64, (u32, u32)>,
    ) -> Result<DecodeRun, String> {
        decode(backend, file, asynchronous, false, None, None, Some(sizes))
    }

    /// [`decode_file`] with a format description of `described`'s size,
    /// where that is given, in place of the stream's.
    /// The MD5s take `hashed` at the top left where it is given, else the
    /// stream's size, or as much of it as the description's holds.
    pub fn decode_file_described(
        backend: &Backend,
        file: &[u8],
        asynchronous: bool,
        described: Option<(u32, u32)>,
        hashed: Option<(u32, u32)>,
    ) -> Result<DecodeRun, String> {
        decode(backend, file, asynchronous, false, described, hashed, None)
    }

    /// [`decode_file`], each frame keeping its packed planes
    /// ([`EmittedFrame::pixels`]), as the encode suite's decode back needs.
    pub fn decode_file_keeping_pixels(
        backend: &Backend,
        file: &[u8],
        asynchronous: bool,
    ) -> Result<DecodeRun, String> {
        decode(backend, file, asynchronous, true, None, None, None)
    }

    /// The byte a described buffer's luma is filled with.
    const LUMA_MARK: u8 = 0x11;

    fn decode(
        backend: &Backend,
        file: &[u8],
        asynchronous: bool,
        keep_pixels: bool,
        described: Option<(u32, u32)>,
        hashed: Option<(u32, u32)>,
        sizes: Option<HashMap<i64, (u32, u32)>>,
    ) -> Result<DecodeRun, String> {
        if !self_test() {
            return Err("MD5 fails its own test vectors".into());
        }
        let (track, stream) = stream(file)?;
        let (width, height) = (stream.width, stream.height);

        let memory = Arc::new(GuestMemory::default());
        let (to_guest, from_host) = mpsc::channel::<Vec<u8>>();
        let session = Arc::new(HostSession::with_backend(
            StdRuntime::shared(),
            Arc::new(ChannelSink(Mutex::new(to_guest))),
            memory.clone(),
            backend,
        ));
        session.set_zero_copy(ZERO_COPY.load(Ordering::Relaxed));
        let (queue0, device_queue) = mpsc::channel::<Vec<u8>>();
        let device = {
            let session = session.clone();
            StdRuntime::thread("conformance-device", move || {
                while let Ok(message) = device_queue.recv() {
                    session.deliver(&message);
                }
            })
            .map_err(|e| e.to_string())?
        };
        let described = described.unwrap_or((width, height));
        let hashed = hashed.unwrap_or((width.min(described.0), height.min(described.1)));
        let guest = Arc::new(Guest {
            width,
            height,
            described,
            pool_size: Mutex::new(described),
            alignment: Mutex::new(UNALIGNED),
            hashed,
            sizes,
            memory: memory.clone(),
            lent: Mutex::new(HashMap::new()),
            waiting: Mutex::new(HashMap::new()),
            next_surface: AtomicU64::new(1),
            next_buffer: AtomicU64::new(1),
            lends: Mutex::new(HashMap::new()),
            emitted: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
            released: AtomicUsize::new(0),
            counts: Mutex::new(HashMap::new()),
            sequence: Mutex::new(Vec::new()),
            frames: Mutex::new(Vec::new()),
            keep_pixels,
            pool: Arc::new(Mutex::new(Vec::new())),
            to_hash: Mutex::new(None),
        });
        let hashers = {
            let threads = std::thread::available_parallelism()
                .map_or(2, |n| n.get() / 4)
                .clamp(2, 6);
            let (to_hash, queue) = mpsc::sync_channel::<ToHash>(threads * 2);
            *guest.to_hash.lock() = Some(to_hash);
            let queue = Arc::new(Mutex::new(queue));
            (0..threads)
                .map(|_| {
                    let guest = guest.clone();
                    let queue = queue.clone();
                    StdRuntime::thread("conformance-hasher", move || {
                        loop {
                            let next = queue.lock().recv();
                            let Ok(next) = next else { break };
                            guest.hash(next);
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        };
        let handler = {
            let guest = guest.clone();
            let queue0 = queue0.clone();
            StdRuntime::thread("conformance-guest", move || {
                let format = Mutex::new(FourCc::new(*b"420v"));
                while let Ok(message) = from_host.recv() {
                    guest.handle(&message, &queue0, &format);
                }
            })
            .map_err(|e| e.to_string())?
        };

        let uuid = GuestUuid([0x42; 16]);
        let next_reply_id = AtomicU64::new(100);
        let call = |mut message: Vec<u8>, timeout: Duration| -> Result<Vec<u8>, String> {
            let reply_id = next_reply_id.fetch_add(1, Ordering::SeqCst);
            if let Some(field) = message.get_mut(32..40) {
                field.copy_from_slice(&reply_id.to_be_bytes());
            }
            let (arrived, reply) = mpsc::channel();
            guest.waiting.lock().insert(reply_id, arrived);
            queue0.send(message).map_err(|e| e.to_string())?;
            let answer = reply
                .recv_timeout(timeout)
                .map_err(|_| "no reply".to_string());
            guest.waiting.lock().remove(&reply_id);
            answer
        };
        let status = |reply: &[u8]| -> Result<i32, String> {
            Message::parse(reply)
                .and_then(|m| m.s32(ERROR))
                .map_err(|e| e.to_string())?
                .ok_or("no ,err".to_string())
        };
        let message = |kind: &[u8; 4]| {
            MessageWriter::new(
                FourCc::new(*kind),
                MessageFlags::GUEST_EXPECTS_REPLY,
                Some(uuid),
                0,
            )
        };
        let short = Duration::from_millis(1100);
        let long = Duration::from_secs(10);

        // The guest asks for the host's decoders once, registers each, and
        // opens a session with the ID of the one for the stream's codec.
        let mut list = MessageWriter::new(
            FourCc::new(*b"decs"),
            MessageFlags::GUEST_EXPECTS_REPLY,
            None,
            0,
        );
        list.u32(VERSION, 6);
        let decoder_id = listed_decoder(&call(list.finish(), short)?, stream.codec)?;
        let guest_info = plist::Value::Dictionary(
            [(
                GUEST_BUILD_VERSION_KEY.to_owned(),
                plist::Value::String(GUEST_BUILD_VERSION.to_owned()),
            )]
            .into_iter()
            .collect(),
        );
        let mut create = message(b"+dec");
        create.s32(CODEC, i32::from_be_bytes(stream.codec));
        create.u32(VERSION, 6);
        create
            .property_list(CODEC_ID, Some(&plist::Value::String(decoder_id)))
            .and_then(|create| create.property_list(GUEST_INFO, Some(&guest_info)))
            .map_err(|e| e.to_string())?;
        let created = status(&call(create.finish(), short)?)?;
        if created != 0 {
            return Err(format!("+dec: {created}"));
        }
        let description = ImageDescription {
            codec: FourCc::new(stream.codec),
            width: u16::try_from(described.0).map_err(|e| e.to_string())?,
            height: u16::try_from(described.1).map_err(|e| e.to_string())?,
            depth: 24,
            compressor_name: Vec::new(),
            extensions: vec![(
                match &stream.codec {
                    b"hvc1" => HEVC_CONFIGURATION,
                    b"av01" => AV1_CONFIGURATION,
                    b"vp09" => VP9_CONFIGURATION,
                    _ => AVC_CONFIGURATION,
                },
                stream.record.clone(),
            )],
        };
        let mut start = message(b"dsta");
        start.image_description(FORMAT, &description.to_bytes());
        start.s32(POOL, 0);
        let started = status(&call(start.finish(), short)?)?;
        if started != 0 {
            return Err(format!("dsta: {started}"));
        }

        let clock = StdRuntime.now();
        let mut written_format = false;
        for (index, &(offset, size)) in track.samples.iter().enumerate() {
            let data = file
                .get(offset..offset + size)
                .ok_or("sample past the file's end")?;
            let frame_id = i64::try_from(index).map_err(|e| e.to_string())?;
            let tick = MediaTime {
                value: frame_id,
                timescale: 30,
                flags: MediaTime::FLAG_VALID,
                epoch: 0,
            };
            let sample = SampleBuffer {
                sample_count: 1,
                timings: vec![SampleTiming {
                    duration: MediaTime { value: 1, ..tick },
                    presentation: tick,
                    decode: tick,
                }],
                sizes: vec![u32::try_from(size).map_err(|e| e.to_string())?],
                data: Some(data.to_vec()),
                // The guest's writer leaves out a format description equal
                // to the last one it wrote.
                format_description: (!written_format)
                    .then(|| format_description(&stream, described)),
                ..SampleBuffer::default()
            };
            written_format = true;
            let serialization = sample.serialize();
            let mut fragment = message(b"dfrm");
            fragment.s64(FRAME, frame_id);
            fragment.u32(FLAGS, u32::from(asynchronous));
            let mut rest =
                fragment.sample_buffer_with_limit(SAMPLE, FRAGMENT_LIMIT, &serialization);
            while !rest.is_empty() {
                fragment.change_flags(MessageFlags::MORE_FRAGMENTS, MessageFlags(0x11));
                queue0.send(fragment.finish()).map_err(|e| e.to_string())?;
                fragment = message(b"dfrm");
                rest = fragment.sample_buffer_with_limit(SAMPLE, FRAGMENT_LIMIT, rest);
            }
            let reply = call(fragment.finish(), if asynchronous { short } else { long })?;
            let decoded = status(&reply)?;
            if decoded != 0 {
                return Err(format!("dfrm {index}: {decoded}"));
            }
        }
        let finished = status(&call(message(b"dfin").finish(), short)?)?;
        let elapsed = StdRuntime.elapsed_since(clock);
        // Asynchronous emissions may still be on their way.
        let settle = StdRuntime.now();
        while guest.emitted.load(Ordering::SeqCst) + guest.failed.load(Ordering::SeqCst)
            < track.samples.len()
            && StdRuntime.elapsed_since(settle) < Duration::from_secs(5)
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        // What follows is the session's end, which the recordings of
        // Apple's host do not hold.
        guest.sequence.lock().push("-dec".to_owned());
        queue0
            .send(
                MessageWriter::new(FourCc::new(*b"-dec"), MessageFlags::NONE, Some(uuid), 0)
                    .finish(),
            )
            .map_err(|e| e.to_string())?;
        // A decs round trip: the device thread has delivered the -dec.
        let mut list = MessageWriter::new(
            FourCc::new(*b"decs"),
            MessageFlags::GUEST_EXPECTS_REPLY,
            None,
            0,
        );
        list.u32(VERSION, 6);
        call(list.finish(), short)?;
        // Every frame emitted is hashed before the run is counted.
        guest.to_hash.lock().take();
        for hasher in hashers {
            let _ = hasher.join();
        }
        let mut frames = guest.frames.lock().clone();
        frames.sort_by_key(|frame| frame.index);
        let run = DecodeRun {
            frames,
            samples: track.samples.len(),
            failed: guest.failed.load(Ordering::SeqCst),
            finished,
            elapsed,
            released: guest.released.load(Ordering::SeqCst),
            most_held: memory.most_held.load(Ordering::SeqCst),
            counts: guest.counts.lock().clone(),
            sequence: guest.sequence.lock().clone(),
        };
        let held_after_close = memory.holds.load(Ordering::SeqCst);
        // The sink goes with invalidation, which ends the handler and its
        // sender on queue 0; then queue 0 closes and the device thread ends.
        session.invalidate();
        session.complete_invalidate();
        drop(queue0);
        let _ = handler.join();
        let _ = device.join();
        if held_after_close != 0 {
            return Err(format!("{held_after_close} surfaces still held after -dec"));
        }
        if memory.holds.load(Ordering::SeqCst) != 0 {
            return Err("surfaces still held after the session ended".into());
        }
        Ok(run)
    }
}
