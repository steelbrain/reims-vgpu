//! Frames copied into host memory (docs/api.md section 2): a [`FrameTarget`] the library draws
//! into, and a [`Frame`], one encode of a display's current frame into a target.
//!
//! A frame target is a shared-storage 2D texture, not mipmapped, whose usage is render target
//! and shader read. A frame is a command buffer of the process's frame queue: encoded
//! uncommitted, then committed, then waited on, as separate calls, so a host can hold a frame
//! in flight. Its completion handler holds the display until it has run; a frame released
//! uncommitted lets the display go with its command buffer, and stays counted in flight for the
//! display's life, as a host's dropped command buffer did (docs/api.md section 10, question 8).
//!
//! Each call, and each release of a frame or a target, runs as an entry ([`crate::entry`]).

use std::sync::Arc;

use reims_vgpu_core::platform::command::CommandBuffer as _;
use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::types::CommandBufferStatus;
use reims_vgpu_core::platform::{Handle as _, Platform};
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use crate::DefaultPlatform;
use crate::entry;
use crate::error::Error;
use crate::platform::ApiPlatform;
use crate::retained::Ours;

/// The largest width or height of a frame target.
pub const MAX_FRAME_TARGET_SIZE: u32 = 16384;

/// `MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget`.
const TARGET_USAGE: u64 = 0x1 | 0x4;
/// `MTLResourceStorageModeShared`, with the default CPU cache and hazard tracking modes.
const TARGET_RESOURCE_OPTIONS: u64 = 0;

/// A raise that reaches an entry point: nothing catches it there, so it ends the process.
fn no_catch<T>(r: RaiseResult<T>) -> T {
    r.unwrap_or_else(|raised| raised.fatal())
}

/// A frame target's pixel format: 8 bits a channel, 4 bytes a pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// `MTLPixelFormatRGBA8Unorm` (70).
    Rgba8Unorm,
    /// `MTLPixelFormatRGBA8Unorm_sRGB` (71).
    Rgba8UnormSrgb,
    /// `MTLPixelFormatBGRA8Unorm` (80): the byte order of a pixman x8r8g8b8 surface.
    Bgra8Unorm,
    /// `MTLPixelFormatBGRA8Unorm_sRGB` (81).
    Bgra8UnormSrgb,
}

impl PixelFormat {
    /// The format of Metal's number, or `None` for any other.
    pub fn from_metal(format: u32) -> Option<PixelFormat> {
        Some(match format {
            70 => PixelFormat::Rgba8Unorm,
            71 => PixelFormat::Rgba8UnormSrgb,
            80 => PixelFormat::Bgra8Unorm,
            81 => PixelFormat::Bgra8UnormSrgb,
            _ => return None,
        })
    }

    /// Metal's number for the format.
    pub fn metal(self) -> u32 {
        match self {
            PixelFormat::Rgba8Unorm => 70,
            PixelFormat::Rgba8UnormSrgb => 71,
            PixelFormat::Bgra8Unorm => 80,
            PixelFormat::Bgra8UnormSrgb => 81,
        }
    }
}

/// Where a frame is drawn in its target, in pixels. The whole target is cleared first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    /// Left.
    pub x: u64,
    /// Top.
    pub y: u64,
    /// Width.
    pub width: u64,
    /// Height.
    pub height: u64,
}

/// What a frame target holds.
struct Target<P: Platform> {
    texture: P::Texture,
    width: u32,
    height: u32,
    format: PixelFormat,
    /// The platform it was made on.
    platform: Arc<P>,
    /// The display's frame the last encode left for a CPU read in place of a draw
    /// (`DisplayCore::encode_current_frame_by_copy`), or `None` after a draw: what
    /// [`FrameTarget::read_pixels`] reads. Released inside the entry that replaces it, or the
    /// target's own release.
    source: reims_vgpu_core::sync::Mutex<Option<P::Texture>>,
    /// What [`FrameTarget::update_pixels`] reads a frame into whole where it cannot read it in
    /// place, kept from one call to the next so that its pages are not taken afresh each time.
    scratch: reims_vgpu_core::sync::Mutex<Vec<u8>>,
    /// The row [`FrameTarget::update_pixels`] copies each row of a frame it reads in place into
    /// before comparing it, kept from one call to the next.
    line: reims_vgpu_core::sync::Mutex<Vec<u8>>,
}

/// A 2D texture the library draws frames into and reads pixels back from: one counted
/// reference. It keeps only its texture, and may outlive the device it was made on.
pub struct FrameTarget<P: Platform = DefaultPlatform> {
    inner: Ours<Arc<Target<P>>>,
}

impl<P: Platform> std::fmt::Debug for FrameTarget<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameTarget")
            .field("width", &self.inner.width)
            .field("height", &self.inner.height)
            .field("format", &self.inner.format)
            .finish_non_exhaustive()
    }
}

impl<P: Platform> FrameTarget<P> {
    /// A target on `device`, the Metal device of a device of this platform.
    pub(crate) fn new(
        platform: &Arc<P>,
        device: &P::Device,
        width: u32,
        height: u32,
        format: PixelFormat,
    ) -> Result<FrameTarget<P>, Error> {
        entry::run(|_| {
            let size = 1..=MAX_FRAME_TARGET_SIZE;
            if !size.contains(&width) || !size.contains(&height) {
                return Err(Error::InvalidArgument);
            }
            // Metal's `+0` descriptor goes at the entry's pool.
            let td = no_catch(platform.texture_2d_descriptor(
                OnFault::Die,
                u64::from(format.metal()),
                u64::from(width),
                u64::from(height),
                false,
            ));
            td.set_usage(TARGET_USAGE);
            td.set_resource_options(TARGET_RESOURCE_OPTIONS);
            let texture = no_catch(device.new_texture(OnFault::Die, &td));
            drop(td);
            let texture = texture.ok_or(Error::Metal)?;
            let target = Target {
                texture,
                width,
                height,
                format,
                platform: platform.clone(),
                source: reims_vgpu_core::sync::Mutex::new(None),
                scratch: reims_vgpu_core::sync::Mutex::new(Vec::new()),
                line: reims_vgpu_core::sync::Mutex::new(Vec::new()),
            };
            Ok(FrameTarget {
                inner: Ours::new(Arc::new(target)),
            })
        })
    }

    /// Another reference to this target.
    #[must_use]
    pub fn clone_ref(&self) -> FrameTarget<P> {
        FrameTarget {
            inner: Ours::new(Arc::clone(&self.inner)),
        }
    }

    /// Pixels wide.
    pub fn width(&self) -> u32 {
        self.inner.width
    }

    /// Pixels high.
    pub fn height(&self) -> u32 {
        self.inner.height
    }

    /// The pixel format.
    pub fn format(&self) -> PixelFormat {
        self.inner.format
    }

    /// The texture.
    pub(crate) fn texture(&self) -> &P::Texture {
        &self.inner.texture
    }

    /// What the last encode into this target left for [`read_pixels`](Self::read_pixels): the
    /// display's frame to read on the CPU, or `None` for the texture the encode drew.
    pub(crate) fn set_source(&self, source: Option<P::Texture>) {
        let old = std::mem::replace(&mut *self.inner.source.lock(), source);
        drop(old);
    }

    /// Whether both are references to one target.
    pub fn ptr_eq(&self, other: &FrameTarget<P>) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<P: ApiPlatform> FrameTarget<P> {
    /// A copy's row of `width * 4` bytes, its rows (`height`) and its extent in a buffer of
    /// `len` bytes, rows `bytes_per_row` apart: [`Error::InvalidArgument`] when `bytes_per_row`
    /// is below the row or `len` below the extent.
    fn copy_extent(
        &self,
        len: usize,
        bytes_per_row: usize,
    ) -> Result<(usize, usize, usize), Error> {
        let row = usize::try_from(self.inner.width)
            .ok()
            .and_then(|w| w.checked_mul(4))
            .ok_or(Error::InvalidArgument)?;
        let rows = usize::try_from(self.inner.height).map_err(|_| Error::InvalidArgument)?;
        let extent = bytes_per_row
            .checked_mul(rows.saturating_sub(1))
            .and_then(|e| e.checked_add(row))
            .ok_or(Error::InvalidArgument)?;
        if bytes_per_row < row || len < extent {
            return Err(Error::InvalidArgument);
        }
        Ok((row, rows, extent))
    }

    /// Brings `dst` up to the target's pixels, laid out as [`read_pixels`](Self::read_pixels)
    /// lays them out: each row that differs from `dst`'s is written and `changed[row]` set to 1;
    /// each equal row is left as it is and `changed[row]` set to 0. A caller that keeps `dst` as
    /// it was left learns which rows changed since its last read, and the rows it did not change
    /// are never written. Where the platform reads the display's frame in place
    /// ([`ApiPlatform::read_in_place`]) nothing is copied whole: each row is read once,
    /// compared and written only where it differs. Otherwise the frame is first read whole into
    /// a buffer the target keeps, as `read_pixels` reads it. Under
    /// `REIMS_VGPU_CONSOLE_READ_CHECK=1` a read in place is checked against `read_pixels`'s, and
    /// a difference logged. [`Error::InvalidArgument`] as for `read_pixels`, and when `changed`
    /// holds fewer than `height` bytes.
    pub fn update_pixels(
        &self,
        dst: &mut [u8],
        bytes_per_row: usize,
        changed: &mut [u8],
    ) -> Result<(), Error> {
        entry::run(|_| {
            let (row, rows, extent) = self.copy_extent(dst.len(), bytes_per_row)?;
            let changed = changed.get_mut(..rows).ok_or(Error::InvalidArgument)?;
            let dst = dst.get_mut(..extent).ok_or(Error::InvalidArgument)?;
            let source = self.inner.source.lock();
            let texture = source.as_ref().unwrap_or(&self.inner.texture);
            let platform = &self.inner.platform;
            let mut in_place = false;
            platform.read_in_place(texture, &mut |first, len, from_row| {
                // The display's frame has the target's size (`encode_current_frame_by_copy`);
                // rows short of the target's are left to the copy.
                let reach = from_row
                    .checked_mul(rows.saturating_sub(1))
                    .and_then(|r| r.checked_add(row));
                if from_row < row || reach.is_none_or(|reach| reach > len) {
                    return;
                }
                // A row of the frame is copied out before it is compared: the guest's or the
                // GPU's writes may reach it meanwhile, so it is never read as a slice.
                let mut line = self.inner.line.lock();
                line.resize(row, 0);
                refresh_rows(dst, bytes_per_row, (row, rows), changed, |y, to| {
                    // SAFETY: row `y` lies in the `len` bytes from `first` (checked above), which
                    // the platform keeps mapped for the call; the copy reads them through the
                    // pointer, as the runtime's `getBytes` does, while writes may reach them.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            first.add(y * from_row),
                            line.as_mut_ptr(),
                            row,
                        );
                    }
                    write_if_differs(to, &line)
                });
                drop(line);
                in_place = true;
            });
            if !in_place {
                if !COPIED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::info!(
                        "console read: the display's frame is read by copying, not in place \
                         (logged once a process)"
                    );
                }
                let mut scratch = self.inner.scratch.lock();
                scratch.resize(extent, 0);
                platform.read_pixels(texture, &mut scratch, bytes_per_row);
                refresh_rows(dst, bytes_per_row, (row, rows), changed, |y, to| {
                    let at = y * bytes_per_row;
                    write_if_differs(to, &scratch[at..at + row])
                });
                drop(scratch);
            } else {
                if !IN_PLACE_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    log::info!(
                        "console read: the display's frame is read in place (logged once a \
                         process)"
                    );
                }
                if read_check_requested() {
                    let mut scratch = self.inner.scratch.lock();
                    scratch.resize(extent, 0);
                    platform.read_pixels(texture, &mut scratch, bytes_per_row);
                    read_check(dst, &scratch, bytes_per_row, (row, rows));
                    drop(scratch);
                }
            }
            drop(source);
            Ok(())
        })
    }

    /// Copies the whole target into `dst`: `height` rows of `width * 4` bytes, each
    /// `bytes_per_row` after the last, in the target's format. It reads what the GPU has
    /// written so far: wait for the frames that draw into it first. [`Error::InvalidArgument`]
    /// when `bytes_per_row` is below `width * 4` or `dst` is shorter than the copy's extent.
    pub fn read_pixels(&self, dst: &mut [u8], bytes_per_row: usize) -> Result<(), Error> {
        entry::run(|_| {
            let (_, _, extent) = self.copy_extent(dst.len(), bytes_per_row)?;
            let dst = dst.get_mut(..extent).ok_or(Error::InvalidArgument)?;
            // The display's frame where the last encode left it for a CPU read (its size and
            // format are the target's), else the texture the encode drew.
            let source = self.inner.source.lock();
            let texture = source.as_ref().unwrap_or(&self.inner.texture);
            self.inner.platform.read_pixels(texture, dst, bytes_per_row);
            drop(source);
            Ok(())
        })
    }
}

/// Brings `dst`'s `rows` rows of `row` bytes, `bytes_per_row` apart, up to the frame's, one at a
/// time: `update` gets row `y` of `dst`, writes the frame's row over it where they differ and
/// says whether it did. A row written is flagged 1 in `changed`, one left as it was 0.
fn refresh_rows(
    dst: &mut [u8],
    bytes_per_row: usize,
    (row, rows): (usize, usize),
    changed: &mut [u8],
    mut update: impl FnMut(usize, &mut [u8]) -> bool,
) {
    for (y, flag) in changed.iter_mut().enumerate().take(rows) {
        let at = y * bytes_per_row;
        *flag = u8::from(update(y, &mut dst[at..at + row]));
    }
}

/// Writes `from` over `to` where they differ; whether they did.
fn write_if_differs(to: &mut [u8], from: &[u8]) -> bool {
    let differs = *to != *from;
    if differs {
        to.copy_from_slice(from);
    }
    differs
}

/// `REIMS_VGPU_CONSOLE_READ_CHECK`: set to `1`, [`FrameTarget::update_pixels`] checks each read
/// in place against the copy `read_pixels` makes.
pub const CONSOLE_READ_CHECK_ENV: &str = "REIMS_VGPU_CONSOLE_READ_CHECK";

/// Whether [`CONSOLE_READ_CHECK_ENV`] is `1`, read once.
fn read_check_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(CONSOLE_READ_CHECK_ENV).is_some_and(|v| v == "1"))
}

/// Whether [`FrameTarget::update_pixels`] has logged a read by copying, and one in place.
static COPIED_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static IN_PLACE_LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The process's checks of a read in place: those made, and those that found rows differing.
static READ_CHECKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static READ_CHECKS_DIFFERING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Compares a read in place (`dst`) with `read_pixels`'s copy of the same frame (`copy`), both
/// `rows` rows of `row` bytes `bytes_per_row` apart: a difference is logged with the rows it
/// takes, and every 1,024th check the counts so far. A guest that writes the frame between the
/// two reads differs too.
fn read_check(dst: &[u8], copy: &[u8], bytes_per_row: usize, (row, rows): (usize, usize)) {
    use std::sync::atomic::Ordering::Relaxed;
    let differing = (0..rows)
        .filter(|y| {
            let at = y * bytes_per_row;
            dst[at..at + row] != copy[at..at + row]
        })
        .count();
    let checks = READ_CHECKS.fetch_add(1, Relaxed) + 1;
    if differing > 0 {
        let bad = READ_CHECKS_DIFFERING.fetch_add(1, Relaxed) + 1;
        log::warn!(
            "console read check: the read in place differs from getBytes in {differing} of \
             {rows} rows (check {checks}, {bad} differing so far)"
        );
    }
    if checks.is_multiple_of(1024) {
        log::info!(
            "console read check: {checks} reads in place checked, {} differing",
            READ_CHECKS_DIFFERING.load(Relaxed)
        );
    }
}

/// One encode of a display's current frame into a target: uncommitted, committed, or complete.
pub struct Frame<P: Platform = DefaultPlatform> {
    /// Its last release may be the display's: the command buffer's completion handler holds it.
    command_buffer: Ours<P::CommandBuffer>,
    /// The target it draws into, kept while the frame is.
    _target: FrameTarget<P>,
    committed: bool,
}

impl<P: Platform> std::fmt::Debug for Frame<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Frame")
            .field("command_buffer", &self.command_buffer.id())
            .field("committed", &self.committed)
            .finish_non_exhaustive()
    }
}

impl<P: Platform> Frame<P> {
    /// An encoded, uncommitted frame.
    pub(crate) fn new(command_buffer: P::CommandBuffer, target: FrameTarget<P>) -> Frame<P> {
        Frame {
            command_buffer: Ours::new(command_buffer),
            _target: target,
            committed: false,
        }
    }

    /// Submits the frame to the GPU. [`Error::FrameState`] if it was committed already.
    pub fn commit(&mut self) -> Result<(), Error> {
        entry::run(|_| {
            if self.committed {
                return Err(Error::FrameState);
            }
            self.committed = true;
            no_catch(self.command_buffer.commit(OnFault::Die));
            Ok(())
        })
    }

    /// Whether [`Frame::commit`] has run.
    pub fn is_committed(&self) -> bool {
        self.committed
    }

    /// Waits until the committed frame completes: [`Error::Gpu`] when its GPU work ended with
    /// an error, [`Error::FrameState`] before [`Frame::commit`].
    pub fn wait(&self) -> Result<(), Error> {
        entry::run(|_| {
            if !self.committed {
                return Err(Error::FrameState);
            }
            no_catch(self.command_buffer.wait_until_completed(OnFault::Die));
            if self.command_buffer.status() == CommandBufferStatus::ERROR {
                return Err(Error::Gpu);
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Metal's four numbers, and no other
    #[test]
    fn formats() {
        for f in [
            PixelFormat::Rgba8Unorm,
            PixelFormat::Rgba8UnormSrgb,
            PixelFormat::Bgra8Unorm,
            PixelFormat::Bgra8UnormSrgb,
        ] {
            assert_eq!(PixelFormat::from_metal(f.metal()), Some(f));
        }
        assert_eq!(PixelFormat::from_metal(73), None);
    }
}
