//! Vulkan Video's std structures, which ash's bindings give no `Default`:
//! each starts as all zeros, then takes its fields.

use ash::vk::native::{
    StdVideoDecodeH264PictureInfo, StdVideoDecodeH264PictureInfoFlags,
    StdVideoDecodeH264ReferenceInfo, StdVideoDecodeH264ReferenceInfoFlags,
    StdVideoH264PictureParameterSet, StdVideoH264PpsFlags, StdVideoH264ScalingLists,
    StdVideoH264SequenceParameterSet, StdVideoH264SpsFlags,
};

/// A std structure whose fields are all integers, bitfield units, arrays
/// of them, or raw pointers: all zeros is a value of it, with each pointer
/// null.
///
/// # Safety
///
/// Every bit pattern of zeros is a valid value of the type.
pub(crate) unsafe trait Zeroable: Sized {}

// SAFETY: each is a C structure of integers, bitfield units, integer
// arrays and raw pointers (vulkan_video_codec_h264std*.h, as ash binds it).
unsafe impl Zeroable for StdVideoH264SpsFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH264PpsFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH264ScalingLists {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH264SequenceParameterSet {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH264PictureParameterSet {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH264PictureInfoFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH264PictureInfo {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH264ReferenceInfoFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH264ReferenceInfo {}

/// A `T` of all zeros.
pub(crate) fn zeroed<T: Zeroable>() -> T {
    // SAFETY: zeros are a value of `T` (`Zeroable`'s contract).
    unsafe { std::mem::zeroed() }
}
