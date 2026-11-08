//! Dma-bufs of host memory for the host's video devices, on Linux: what the reims-graphics
//! device's surface views and region blocks lend them (`graphics/decode`'s `PlanePages`), as
//! core's `DeviceMemory::dma_buf_of` answers, over the Vulkan crate's dma-buf of memory its
//! backing table resolves to files (`reims_vgpu_vulkan::os::dma_buf_of_host`).

use std::ptr::NonNull;

use reims_vgpu_core::num::ToU64 as _;
use reims_vgpu_video::core::device_memory::{DmaBufError, SurfaceDmaBuf};
use reims_vgpu_vulkan::os::{LendRefusal, dma_buf_of_host};

/// A dma-buf of `ranges` of the `length` bytes from `first`, memory the library mapped from files
/// or the host registered (module docs). The caller keeps the bytes mapped for the call.
pub(crate) fn host_dma_buf(
    first: NonNull<u8>,
    length: usize,
    ranges: &[(usize, usize)],
) -> Result<SurfaceDmaBuf, DmaBufError> {
    let ranges: Vec<(u64, u64)> = (ranges.iter())
        .map(|&(at, bytes)| (at.to_u64(), bytes.to_u64()))
        .collect();
    match dma_buf_of_host(first, length.to_u64(), &ranges) {
        Ok(made) => Ok(SurfaceDmaBuf {
            fd: made.fd,
            size: made.size,
            offsets: made.offsets,
        }),
        Err(LendRefusal::TooLarge {
            pieces,
            bytes,
            most_pieces,
            most_bytes,
            why,
        }) => Err(DmaBufError::TooLarge {
            pieces,
            bytes,
            most_pieces,
            most_bytes,
            why,
        }),
        Err(LendRefusal::Refused(why)) => Err(DmaBufError::Refused(why)),
    }
}
