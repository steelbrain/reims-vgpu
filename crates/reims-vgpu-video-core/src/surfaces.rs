//! Guest surfaces as the host reaches them: by the mapping ID the guest's
//! IOSurface mapper gave each one.
//!
//! The device that embeds this crate implements [`Surfaces`] over the GPU
//! device's mapper. A hold is visible to the guest: unmapping a surface the
//! host holds answers Busy, and the guest kext keeps its own mapping of the
//! surface until the host's `rels` names it. So the host session holds each
//! surface exactly as long as Apple's does: while a message that lent it is
//! being handled, while a decoder session writes into it or waits to hand
//! it back, and while an encoder session has a frame of it in flight.

pub use crate::wire::surface_boxes::MappingId;

use crate::device_memory::DeviceMemory;
use crate::fourcc::FourCc;

/// Reaches the guest's surfaces by mapping ID.
pub trait Surfaces: Send + Sync {
    /// Takes a hold on the surface with `mapping_id`. The surface stays
    /// held, and its memory valid, until the returned value is dropped.
    ///
    /// A failure is not reported to the guest by itself: the message that
    /// named the surface is delivered with none of its surfaces, as
    /// Apple's host does when it cannot map one.
    fn retain(&self, mapping_id: MappingId) -> Result<Box<dyn GuestSurface>, SurfaceError>;
}

/// One held guest surface. Dropping it releases the hold.
///
/// A lend, [`GuestSurface::write`] or [`GuestSurface::read`], must not
/// lend another surface from inside it: a host may lend one surface at a
/// time in the process, as reims-vgpu's mapper does, so a lend inside
/// another waits on itself for ever. Copy one surface out first, as the
/// scaler and the motion-estimation session do.
pub trait GuestSurface: Send + Sync {
    /// The surface's pixel format, size and planes, as the guest created
    /// it.
    fn layout(&self) -> SurfaceLayout;

    /// Lends the surface's memory for writing: `write` gets every byte of
    /// the surface, each plane at its [`PlaneLayout::offset`].
    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError>;

    /// Lends the surface's memory for reading, as [`GuestSurface::write`]
    /// lends it for writing: the encoder sessions read their source frames
    /// through it. A surface that cannot be read says so, and the frame is
    /// refused.
    fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
        let _ = read;
        Err(SurfaceError::Unreachable(
            "this host's surfaces cannot be read".to_owned(),
        ))
    }

    /// The surface's memory as a device writes it without a lend, for a
    /// backend whose GPU writes a frame into the guest's pages itself:
    /// valid while this hold lives. `None`, the default, where the host
    /// offers none, as for a surface the guest made read-only for its
    /// clients, which is never lent for writing either; its frames are then
    /// copied by the CPU.
    fn device_memory(&self) -> Option<&dyn DeviceMemory> {
        None
    }

    /// A device wrote the surface's memory ([`GuestSurface::device_memory`]):
    /// called once the write is in the guest's pages, before the guest is
    /// told of it. The host may keep a copy of the pages elsewhere that the
    /// write makes stale. The default does nothing.
    fn device_wrote(&self) {}
}

/// A surface's layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceLayout {
    /// The CoreVideo pixel format, such as `420v`.
    pub pixel_format: FourCc,
    /// The width in pixels.
    pub width: u32,
    /// The height in pixels.
    pub height: u32,
    /// The planes, in plane order; empty for a surface with one plane
    /// that starts at offset 0 (then [`SurfaceLayout::bytes_per_row`] and
    /// the height describe it).
    pub planes: Vec<PlaneLayout>,
    /// The bytes per row of a surface without planes.
    pub bytes_per_row: usize,
}

/// One plane of a surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLayout {
    /// Where the plane starts in the surface's memory.
    pub offset: usize,
    /// The bytes from one row to the next.
    pub bytes_per_row: usize,
    /// The width in samples (half the surface's for a 4:2:0 chroma plane).
    pub width: u32,
    /// The height in rows.
    pub height: u32,
}

/// One plane of a destination, lent to a decoded frame to copy into.
#[derive(Debug)]
pub struct PlaneDestination<'a> {
    /// The plane's rows, `bytes_per_row` apart; the last may be shorter
    /// than `bytes_per_row` but holds the row's samples.
    pub bytes: &'a mut [u8],
    /// The bytes from one row to the next.
    pub bytes_per_row: usize,
    /// The rows the plane has room for.
    pub rows: u32,
}

/// One plane of a source, lent to an encoder to read from.
#[derive(Clone, Copy, Debug)]
pub struct PlaneSource<'a> {
    /// The plane's rows, `bytes_per_row` apart; the last may be shorter
    /// than `bytes_per_row` but holds the row's samples.
    pub bytes: &'a [u8],
    /// The bytes from one row to the next.
    pub bytes_per_row: usize,
    /// The rows the plane has.
    pub rows: u32,
}

/// Why a surface could not be held or reached.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SurfaceError {
    /// No surface has this mapping ID.
    #[error("no surface with mapping ID {0:#x}")]
    Unknown(u64),
    /// The surface exists but its memory cannot be reached.
    #[error("surface memory unreachable: {0}")]
    Unreachable(String),
    /// A plane's offset and size do not fit the surface's memory, or two
    /// planes overlap.
    #[error("surface layout does not fit its memory")]
    Layout,
}

impl SurfaceLayout {
    /// Where each plane lies in a surface's memory of `length` bytes: its
    /// offset, its length to the end of its last row, its bytes per row and
    /// its rows. Planes must be in ascending order without overlapping, and
    /// each must fit the memory.
    fn plane_ranges(&self, length: usize) -> Result<Vec<(usize, usize, usize, u32)>, SurfaceError> {
        let planes = if self.planes.is_empty() {
            vec![PlaneLayout {
                offset: 0,
                bytes_per_row: self.bytes_per_row,
                width: self.width,
                height: self.height,
            }]
        } else {
            self.planes.clone()
        };
        let mut ranges = Vec::with_capacity(planes.len());
        let mut consumed = 0usize;
        for plane in planes {
            let plane_length = plane
                .bytes_per_row
                .checked_mul(usize::try_from(plane.height).map_err(|_| SurfaceError::Layout)?)
                .ok_or(SurfaceError::Layout)?;
            if plane.offset < consumed {
                return Err(SurfaceError::Layout);
            }
            let end = plane
                .offset
                .checked_add(plane_length)
                .ok_or(SurfaceError::Layout)?;
            if end > length {
                return Err(SurfaceError::Layout);
            }
            ranges.push((
                plane.offset,
                plane_length,
                plane.bytes_per_row,
                plane.height,
            ));
            consumed = end;
        }
        Ok(ranges)
    }

    /// Splits a surface's memory into its planes, each from its offset to
    /// the end of its last row. Planes must be in ascending order without
    /// overlapping, and each must fit the memory.
    pub fn split<'a>(
        &self,
        memory: &'a mut [u8],
    ) -> Result<Vec<PlaneDestination<'a>>, SurfaceError> {
        let ranges = self.plane_ranges(memory.len())?;
        let mut destinations = Vec::with_capacity(ranges.len());
        let mut rest = memory;
        let mut consumed = 0usize;
        for (offset, length, bytes_per_row, rows) in ranges {
            // Within the memory, as `plane_ranges` checked.
            let (_, after_skip) = offset
                .checked_sub(consumed)
                .and_then(|skip| rest.split_at_mut_checked(skip))
                .ok_or(SurfaceError::Layout)?;
            let (bytes, after) = after_skip
                .split_at_mut_checked(length)
                .ok_or(SurfaceError::Layout)?;
            destinations.push(PlaneDestination {
                bytes,
                bytes_per_row,
                rows,
            });
            rest = after;
            consumed = offset.saturating_add(length);
        }
        Ok(destinations)
    }

    /// Splits a surface's memory into its planes for reading, as
    /// [`SurfaceLayout::split`] does for writing.
    pub fn split_for_reading<'a>(
        &self,
        memory: &'a [u8],
    ) -> Result<Vec<PlaneSource<'a>>, SurfaceError> {
        self.plane_ranges(memory.len())?
            .into_iter()
            .map(|(offset, length, bytes_per_row, rows)| {
                let bytes = offset
                    .checked_add(length)
                    .and_then(|end| memory.get(offset..end))
                    .ok_or(SurfaceError::Layout)?;
                Ok(PlaneSource {
                    bytes,
                    bytes_per_row,
                    rows,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv12(offsets: [usize; 2]) -> SurfaceLayout {
        SurfaceLayout {
            pixel_format: FourCc::new(*b"420v"),
            width: 4,
            height: 2,
            planes: vec![
                PlaneLayout {
                    offset: offsets[0],
                    bytes_per_row: 8,
                    width: 4,
                    height: 2,
                },
                PlaneLayout {
                    offset: offsets[1],
                    bytes_per_row: 8,
                    width: 2,
                    height: 1,
                },
            ],
            bytes_per_row: 0,
        }
    }

    #[test]
    fn planes_split_at_their_offsets() -> Result<(), SurfaceError> {
        let mut memory = [0u8; 64];
        let mut planes = nv12([0, 32]).split(&mut memory)?;
        assert_eq!(planes.len(), 2);
        planes[0].bytes.fill(1);
        planes[1].bytes.fill(2);
        assert_eq!(&memory[..16], &[1; 16]);
        assert_eq!(&memory[16..32], &[0; 16]);
        assert_eq!(&memory[32..40], &[2; 8]);
        Ok(())
    }

    #[test]
    fn overlapping_or_oversized_planes_are_refused() {
        let mut memory = [0u8; 64];
        assert_eq!(
            nv12([0, 8]).split(&mut memory).err(),
            Some(SurfaceError::Layout)
        );
        assert_eq!(
            nv12([0, 60]).split(&mut memory).err(),
            Some(SurfaceError::Layout)
        );
        assert_eq!(
            nv12([0, 60]).split_for_reading(&memory).err(),
            Some(SurfaceError::Layout)
        );
    }

    #[test]
    fn planes_read_at_their_offsets() -> Result<(), SurfaceError> {
        let memory: Vec<u8> = (0..64).collect();
        let planes = nv12([0, 32]).split_for_reading(&memory)?;
        assert_eq!(planes.len(), 2);
        assert_eq!(planes[0].bytes, &memory[..16]);
        assert_eq!(planes[1].bytes, &memory[32..40]);
        assert_eq!((planes[1].bytes_per_row, planes[1].rows), (8, 1));
        Ok(())
    }
}
