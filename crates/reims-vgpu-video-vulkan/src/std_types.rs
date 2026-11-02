//! Vulkan Video's std structures, which ash's bindings give no `Default`:
//! each starts as all zeros, then takes its fields.

use ash::vk::native::{
    StdVideoDecodeH264PictureInfo, StdVideoDecodeH264PictureInfoFlags,
    StdVideoDecodeH264ReferenceInfo, StdVideoDecodeH264ReferenceInfoFlags,
    StdVideoDecodeH265PictureInfo, StdVideoDecodeH265PictureInfoFlags,
    StdVideoDecodeH265ReferenceInfo, StdVideoDecodeH265ReferenceInfoFlags,
    StdVideoH264PictureParameterSet, StdVideoH264PpsFlags, StdVideoH264ScalingLists,
    StdVideoH264SequenceParameterSet, StdVideoH264SpsFlags, StdVideoH265DecPicBufMgr,
    StdVideoH265LongTermRefPicsSps, StdVideoH265PictureParameterSet, StdVideoH265PpsFlags,
    StdVideoH265ProfileTierLevel, StdVideoH265ProfileTierLevelFlags, StdVideoH265ScalingLists,
    StdVideoH265SequenceParameterSet, StdVideoH265ShortTermRefPicSet,
    StdVideoH265ShortTermRefPicSetFlags, StdVideoH265SpsFlags, StdVideoH265VideoParameterSet,
    StdVideoH265VpsFlags,
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
// SAFETY: each is a C structure of integers, bitfield units, integer
// arrays and raw pointers (vulkan_video_codec_h265std*.h, as ash binds it).
unsafe impl Zeroable for StdVideoH265VpsFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265VideoParameterSet {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265ProfileTierLevelFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265ProfileTierLevel {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265DecPicBufMgr {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265SpsFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265SequenceParameterSet {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265ScalingLists {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265ShortTermRefPicSetFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265ShortTermRefPicSet {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265LongTermRefPicsSps {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265PpsFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoH265PictureParameterSet {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH265PictureInfoFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH265PictureInfo {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH265ReferenceInfoFlags {}
// SAFETY: as above.
unsafe impl Zeroable for StdVideoDecodeH265ReferenceInfo {}

/// A `T` of all zeros.
pub(crate) fn zeroed<T: Zeroable>() -> T {
    // SAFETY: zeros are a value of `T` (`Zeroable`'s contract).
    unsafe { std::mem::zeroed() }
}
