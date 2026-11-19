//! An NVENC encode session's frames written by the runtime device's GPU into memory NVENC reads
//! ([`Filler`]), as the VA-API arm's are written into its encoder's input surfaces: a buffer of
//! the device's made exportable, its memory exported and imported on the session's CUDA GPU (the
//! device's own, `super::super`'s module doc), which NVENC registers as input
//! (`NvencEncoder::encode_device`).
//!
//! - **The input** is one buffer a session, made on its first frame: the frame's luma rows from
//!   its first byte, then its chroma rows, half as many, rounded up, at one pitch, as NVENC reads
//!   registered memory. The pitch is the source's own where its planes share one NVENC takes (a
//!   multiple of 4 that holds a row), so that each plane is one copy; else the row's bytes
//!   rounded up to 256, and each row is a copy of its own. A frame of another size, which only a
//!   new encoder takes, makes the input again. The import holds the buffer, so the memory
//!   outlives every registration of it. NVENC reads it one frame at a time: the encoder's frames
//!   come out before the next goes in (B-frames are none, `super`'s module doc).
//! - **A frame** is copied from the source's planes, where they lie, in the surface's pages or
//!   its device caches, by a blit on a queue of the filler's own: each plane's rows of the
//!   frame's samples, as NVENC's upload copies them, so NVENC encodes the same bytes either way.
//!   The blit then hands the input to CUDA (`BlitEncoder::release_to_external`: rt's release of
//!   its block, nothing on the Vulkan platform), and the copy is waited for before NVENC maps the
//!   input, as the upload is. Nothing reads the input through the device: the next frame's copy
//!   writes it anew.
//! - **Refusals** ([`Filled::Declined`]) come before anything is submitted, and the frame is
//!   uploaded instead; an input the device or CUDA refuses to make is not tried again in the
//!   session. A copy that failed once submitted is an error: its input is kept out of
//!   use while the session lives, as the GPU may write it still.
//!
//! Each session logs at its end how many frames went this way, and the copies' times.

use std::sync::Arc;
use std::time::Duration;

use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::nvidia::{Gpu, ImportedBuffer};
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::blit::{BlitCmd, BlitEncoder as _};
use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::platform::types::CommandBufferStatus;
use reims_vgpu_vulkan::core::raise::OnFault;

use super::super::super::video_host::{BackendError, OutputPlane};
use super::super::{ENCODE_NAME, STORAGE_PRIVATE};

/// The alignment of the input's rows where the source's pitch is not taken: 256 bytes.
const PITCH_ALIGNMENT: u64 = 256;

/// What came of a frame's fill.
pub(super) enum Filled {
    /// The frame lies in `input` at `pitch`, written and waited for.
    Written {
        input: Arc<ImportedBuffer>,
        pitch: u32,
    },
    /// Nothing was written, for the reason given: the frame is uploaded.
    Declined(String),
}

/// One session's input, made as its first frame was filled.
struct Input<P: Platform> {
    buffer: P::Buffer,
    /// Its import on the session's GPU, which holds the buffer.
    import: Arc<ImportedBuffer>,
    pitch: u64,
    /// The bytes of a row, and the luma rows, of the frames it was made for.
    row_bytes: u64,
    rows: u64,
}

/// An encode session's fills of its frames into memory NVENC reads (module doc).
pub(in super::super) struct Filler<P: Platform> {
    device: P::Device,
    gpu: Arc<Gpu>,
    queue: Option<P::CommandQueue>,
    input: Option<Input<P>>,
    /// Why no input is made, once a make was refused: the session's later frames are declined
    /// with it, and no make is tried again.
    refused: Option<String>,
    /// Inputs whose copy failed on the GPU, which may write them still.
    failed: Vec<Input<P>>,
    /// Frames filled, and the time of their copies from submission to their wait's end.
    filled: u64,
    /// The clock the copies are timed on: the video's runtime's.
    runtime: Arc<dyn Runtime>,
    total: Duration,
    longest: Duration,
}

impl<P: Platform> Filler<P> {
    /// A filler of `device`'s, whose memory `gpu`, the device's own GPU, imports, its copies timed
    /// on `runtime`'s clock.
    pub(in super::super) fn new(
        device: &P::Device,
        gpu: &Arc<Gpu>,
        runtime: &Arc<dyn Runtime>,
    ) -> Self {
        Self {
            device: device.clone_ref(),
            gpu: Arc::clone(gpu),
            queue: None,
            input: None,
            refused: None,
            failed: Vec::new(),
            filled: 0,
            runtime: Arc::clone(runtime),
            total: Duration::ZERO,
            longest: Duration::ZERO,
        }
    }

    /// Frames filled so far.
    #[cfg(test)]
    pub(super) fn filled(&self) -> u64 {
        self.filled
    }

    /// Writes the frame of `size`, of samples of `sample` bytes, in `luma` and `chroma` where
    /// they lie, into the session's input (module doc), and returns once the GPU wrote it.
    /// `Err` where the copy failed once submitted: the input is kept out of use, and the caller
    /// fills no more frames of the session.
    pub(super) fn fill(
        &mut self,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
        size: (u32, u32),
        sample: u32,
    ) -> Result<Filled, BackendError> {
        let (width, height) = (u64::from(size.0), u64::from(size.1));
        let row_bytes = width * u64::from(sample);
        let chroma_rows = height.div_ceil(2);
        if row_bytes == 0 || height == 0 {
            return Ok(Filled::Declined("an empty frame".to_owned()));
        }
        for (plane, rows, which) in [(luma, height, "luma"), (chroma, chroma_rows, "chroma")] {
            if plane.row_bytes < row_bytes || plane.height < rows {
                return Ok(Filled::Declined(format!(
                    "a {which} plane of {} rows of {} bytes for a frame of {rows} of {row_bytes}",
                    plane.height, plane.row_bytes
                )));
            }
        }
        let input = match self.input_for(luma, chroma, row_bytes, height) {
            Ok(input) => input,
            Err(why) => return Ok(Filled::Declined(why)),
        };
        let (pitch, import) = (input.pitch, Arc::clone(&input.import));
        let Ok(pitch_u32) = u32::try_from(pitch) else {
            return Ok(Filled::Declined(format!("a pitch of {pitch}")));
        };
        let destination = input.buffer.clone_ref();
        let copies = [(luma, 0, height), (chroma, pitch * height, chroma_rows)];
        let start = self.runtime.now();
        match self.copy(&destination, &copies, pitch, row_bytes)? {
            Some(why) => return Ok(Filled::Declined(why)),
            None => {
                let took = self.runtime.elapsed_since(start);
                self.filled += 1;
                self.total += took;
                self.longest = self.longest.max(took);
            }
        }
        Ok(Filled::Written {
            input: import,
            pitch: pitch_u32,
        })
    }

    /// The session's input for frames of `rows` rows of `row_bytes` from `luma` and `chroma`: the
    /// one made, where it holds them, else a new one (module doc). `Err` says why none is made.
    fn input_for(
        &mut self,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
        row_bytes: u64,
        rows: u64,
    ) -> Result<&Input<P>, String> {
        if let Some(why) = &self.refused {
            return Err(why.clone());
        }
        let kept = self
            .input
            .take()
            .filter(|i| i.row_bytes == row_bytes && i.rows == rows);
        let input = match kept {
            Some(input) => input,
            None => match self.make_input(luma, chroma, row_bytes, rows) {
                Ok(input) => input,
                Err(why) => return Err(self.refused.insert(why).clone()),
            },
        };
        Ok(self.input.insert(input))
    }

    /// A new input for frames of `rows` rows of `row_bytes`, at `luma`'s and `chroma`'s pitch
    /// where NVENC takes it.
    fn make_input(
        &self,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
        row_bytes: u64,
        rows: u64,
    ) -> Result<Input<P>, String> {
        let source = luma.bytes_per_row;
        let pitch = if source == chroma.bytes_per_row && source % 4 == 0 && source >= row_bytes {
            source
        } else {
            row_bytes.next_multiple_of(PITCH_ALIGNMENT)
        };
        let length = (rows + rows.div_ceil(2))
            .checked_mul(pitch)
            .ok_or_else(|| format!("an input of {rows} rows at {pitch}"))?;
        let made = (self.device).new_exportable_buffer_of_length(
            OnFault::Return,
            length,
            STORAGE_PRIVATE,
            None,
        );
        let buffer = match made {
            Ok(Some(buffer)) => buffer,
            Ok(None) => return Err("the device makes no exportable buffer".to_owned()),
            Err(raise) => return Err(format!("an exportable buffer's make: {raise}")),
        };
        let exported = buffer.export_memory()?;
        let import = ImportedBuffer::import(
            &self.gpu,
            exported.fd,
            exported.allocation_size,
            exported.offset,
            exported.length,
            exported.dedicated,
        )
        .map_err(|error| format!("CUDA's import of an exportable buffer: {error}"))?;
        let import = Arc::new(import.holding(Box::new(buffer.clone_ref())));
        Ok(Input {
            buffer,
            import,
            pitch,
            row_bytes,
            rows,
        })
    }

    /// Copies each of `copies`, a plane, the input's byte its first row goes to and its rows, into
    /// `destination`, rows `pitch` apart, `row_bytes` of each, on the filler's queue, and waits
    /// for the copy. `Some` says why nothing was submitted; `Err` is a copy that failed once
    /// submitted, whose input is kept out of use.
    fn copy(
        &mut self,
        destination: &P::Buffer,
        copies: &[(&OutputPlane<P>, u64, u64); 2],
        pitch: u64,
        row_bytes: u64,
    ) -> Result<Option<String>, BackendError> {
        if self.queue.is_none() {
            let Some(queue) = self.device.new_command_queue() else {
                return Ok(Some("no command queue".to_owned()));
            };
            queue.set_label("reims-vgpu.graphics-video-fill");
            self.queue = Some(queue);
        }
        let Some(queue) = self.queue.as_ref() else {
            return Ok(Some("no command queue".to_owned()));
        };
        let cb = match queue.command_buffer(OnFault::Return) {
            Ok(Some(cb)) => cb,
            Ok(None) => return Ok(Some("no command buffer".to_owned())),
            Err(raise) => return Ok(Some(format!("its command buffer: {raise}"))),
        };
        let blit = match cb.blit_encoder(OnFault::Return) {
            Ok(Some(blit)) => blit,
            Ok(None) => return Ok(Some("no blit encoder".to_owned())),
            Err(raise) => return Ok(Some(format!("its blit encoder: {raise}"))),
        };
        for &(plane, at, rows) in copies {
            for (from, to, size) in row_copies(plane, at, rows, pitch, row_bytes) {
                let copy = BlitCmd::CopyBufferToBuffer {
                    source_buffer: Some(&plane.buffer),
                    source_offset: from,
                    destination_buffer: Some(destination),
                    destination_offset: to,
                    size,
                };
                if let Err(raise) = blit.encode(OnFault::Return, copy) {
                    return Ok(Some(format!("a copy: {raise}")));
                }
            }
        }
        if let Err(why) = blit.release_to_external(destination) {
            return Ok(Some(format!("its release to CUDA: {why}")));
        }
        if let Err(raise) = blit.encode(OnFault::Return, BlitCmd::EndEncoding) {
            return Ok(Some(format!("its end: {raise}")));
        }
        if let Err(raise) = cb.commit(OnFault::Return) {
            return Ok(Some(format!("its commit: {raise}")));
        }
        let failed = match cb.wait_until_completed(OnFault::Return) {
            Err(raise) => Some(format!("its wait: {raise}")),
            Ok(()) if cb.status() != CommandBufferStatus::COMPLETED => {
                let error = cb.error().map(|e| format!("{e:?}")).unwrap_or_default();
                Some(format!("the copy failed {error}"))
            }
            Ok(()) => None,
        };
        if let Some(failed) = failed {
            self.failed.extend(self.input.take());
            return Err(format!("the GPU's copy of a frame into NVENC's input: {failed}").into());
        }
        Ok(None)
    }
}

/// The copies of `rows` rows of `row_bytes` from `plane` into an input whose rows are `pitch`
/// apart from its byte `at`, each (source offset, input offset, size): one where the plane's
/// rows are `pitch` apart too, from its first row's first byte to its last row's last, else one a
/// row.
fn row_copies<P: Platform>(
    plane: &OutputPlane<P>,
    at: u64,
    rows: u64,
    pitch: u64,
    row_bytes: u64,
) -> Vec<(u64, u64, u64)> {
    if plane.bytes_per_row == pitch {
        return vec![(plane.offset, at, pitch * (rows - 1) + row_bytes)];
    }
    (0..rows)
        .map(|row| {
            (
                plane.offset + row * plane.bytes_per_row,
                at + row * pitch,
                row_bytes,
            )
        })
        .collect()
}

impl<P: Platform> Drop for Filler<P> {
    fn drop(&mut self) {
        if self.filled == 0 {
            return;
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let mean = self.total / u32::try_from(self.filled).unwrap_or(u32::MAX);
        log::info!(
            "hardware video: {ENCODE_NAME}: an encode session's frames copied by the GPU into \
             NVENC's input: {}, {:.3} ms each from submission to the wait's end, {:.3} ms at most",
            self.filled,
            ms(mean),
            ms(self.longest)
        );
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_vulkan::core::testkit::FakePlatform;

    use super::*;

    /// A plane of `bytes_per_row` at `offset`, of rows of 100 bytes, in a buffer of the fake's.
    fn plane(p: &FakePlatform, offset: u64, bytes_per_row: u64) -> OutputPlane<FakePlatform> {
        let buffer = (p.device().new_buffer_of_length(OnFault::Return, 1 << 16, 0))
            .ok()
            .flatten()
            .unwrap_or_else(|| unreachable!("the fake makes buffers"));
        OutputPlane {
            buffer,
            offset,
            bytes_per_row,
            row_bytes: 100,
            width: 100,
            height: 4,
            rows_held: 4,
            cached: false,
            host: None,
            pages: None,
            cache: None,
            #[cfg(unix)]
            surface: None,
        }
    }

    /// A plane at the input's pitch is one copy, from its first row to its last row's last byte;
    /// one at another pitch is a copy a row, of the row's bytes alone.
    #[test]
    fn a_plane_at_the_inputs_pitch_is_one_copy_else_one_a_row() {
        let p = FakePlatform::new();
        let same = plane(&p, 32, 128);
        assert_eq!(row_copies(&same, 0, 4, 128, 100), [(32, 0, 3 * 128 + 100)]);
        let other = plane(&p, 32, 192);
        assert_eq!(
            row_copies(&other, 512, 3, 128, 100),
            [(32, 512, 100), (224, 640, 100), (416, 768, 100)]
        );
    }
}
