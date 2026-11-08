//! Metal's private classes the platform names (the buffer descriptor, the address-range array,
//! the ICB argument classes) with typed declarations of the methods it sends them.

use std::ffi::c_void;

use objc2::rc::{Allocated, Retained};
use objc2::runtime::NSObject;
use objc2::{extern_class, extern_methods};
use objc2_metal::MTLSize;
use reims_vgpu_core::platform::types::ResourceAddressRange;

use crate::msg::encode::Enc;
use crate::spi::methods::Deallocator;

extern_class!(
    /// The private `MTLBufferDescriptor` (the images import `_OBJC_CLASS_$_MTLBufferDescriptor`).
    #[unsafe(super(NSObject))]
    #[name = "MTLBufferDescriptor"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct MTLBufferDescriptor;
);

#[allow(non_snake_case, missing_docs, reason = "each method is its selector")]
impl MTLBufferDescriptor {
    extern_methods!(
        /// `+new`: `+1`.
        #[unsafe(method(new))]
        pub fn new() -> Retained<Self>;

        /// The buffer's bytes, as they are (no copy with `noCopy`).
        ///
        /// # Safety
        /// `contents` stays valid for the buffer made from the descriptor.
        #[unsafe(method(setContents:))]
        pub unsafe fn setContents(&self, contents: *mut c_void);

        /// Metal copies the block and calls it once, when the buffer it made is released.
        #[unsafe(method(setDeallocator:))]
        pub fn setDeallocator(&self, deallocator: Option<&Deallocator>);

        #[unsafe(method(setLength:))]
        pub fn setLength(&self, length: u64);

        #[unsafe(method(setNoCopy:))]
        pub fn setNoCopy(&self, no_copy: bool);

        #[unsafe(method(setPinnedGPUAddress:))]
        pub fn setPinnedGPUAddress(&self, address: u64);

        #[unsafe(method(setResourceOptions:))]
        pub fn setResourceOptions(&self, options: u64);

        #[unsafe(method(setStorageMode:))]
        pub fn setStorageMode(&self, mode: u64);

        #[unsafe(method(resourceOptions))]
        pub fn resourceOptions(&self) -> u64;
    );
}

extern_class!(
    /// The private `MTLResourceAddressRangeArray`.
    #[unsafe(super(NSObject))]
    #[name = "MTLResourceAddressRangeArray"]
    #[derive(Debug, PartialEq, Eq, Hash)]
    pub struct MTLResourceAddressRangeArray;
);

#[allow(non_snake_case, missing_docs, reason = "each method is its selector")]
impl MTLResourceAddressRangeArray {
    extern_methods!(
        /// The array copies the ranges (`platform`'s test `new_address_range_array`).
        ///
        /// # Safety
        /// `ranges` is `count` readable ranges.
        #[unsafe(method(initWithRanges:count:))]
        pub unsafe fn initWithRanges_count(
            this: Allocated<Self>,
            ranges: *const Enc<ResourceAddressRange>,
            count: u64,
        ) -> Option<Retained<Self>>;

        #[unsafe(method(count))]
        pub fn count(&self) -> u64;

        /// The array's own copy of the ranges.
        #[unsafe(method(ranges))]
        pub fn ranges(&self) -> *const Enc<ResourceAddressRange>;
    );
}

macro_rules! private_classes {
    ($($rust:ident = $name:literal { $($(#[$m:meta])* fn $method:ident -> $ret:ty;)* })*) => {
        $(
            extern_class!(
                #[doc = concat!("Metal's private `", $name, "`.")]
                #[unsafe(super(NSObject))]
                #[name = $name]
                #[derive(Debug, PartialEq, Eq, Hash)]
                pub struct $rust;
            );

            #[allow(non_snake_case, missing_docs, reason = "each getter is its selector")]
            impl $rust {
                extern_methods!(
                    $(
                        $(#[$m])*
                        #[unsafe(method($method))]
                        pub fn $method(&self) -> $ret;
                    )*
                );
            }
        )*
    };
}

private_classes! {
    IndirectDrawArguments = "_MTLIndirectDrawArguments" {
        fn baseInstance -> u64;
        fn instanceCount -> u64;
        fn primitiveType -> u64;
        fn vertexCount -> u64;
        fn vertexStart -> u64;
    }
    IndirectDrawIndexedArguments = "_MTLIndirectDrawIndexedArguments" {
        fn baseInstance -> u64;
        fn baseVertex -> u64;
        fn indexBufferGPUVirtualAddress -> u64;
        fn indexCount -> u64;
        fn indexType -> u64;
        fn instanceCount -> u64;
        fn primitiveType -> u64;
    }
    IndirectDrawPatchesArguments = "_MTLIndirectDrawPatchesArguments" {
        fn baseInstance -> u64;
        fn instanceCount -> u64;
        fn numberOfPatchControlPoints -> u64;
        fn patchCount -> u64;
        fn patchIndexBufferGPUVirtualAddress -> u64;
        fn patchStart -> u64;
    }
    IndirectDrawIndexedPatchesArguments = "_MTLIndirectDrawIndexedPatchesArguments" {
        fn baseInstance -> u64;
        fn controlPointIndexBufferGPUVirtualAddress -> u64;
        fn instanceCount -> u64;
        fn numberOfPatchControlPoints -> u64;
        fn patchCount -> u64;
        fn patchIndexBufferGPUVirtualAddress -> u64;
        fn patchStart -> u64;
    }
    IndirectTessellationFactorArguments = "_MTLIndirectTessellationFactorArguments" {
        fn GPUVirtualAddress -> u64;
        fn instanceStride -> u64;
        fn scale -> f32;
    }
    IndirectDispatchThreadsArguments = "_MTLIndirectDispatchThreadsArguments" {
        fn threadsPerGrid -> MTLSize;
        fn threadsPerThreadgroup -> MTLSize;
    }
    IndirectDispatchThreadgroupsArguments = "_MTLIndirectDispatchThreadgroupsArguments" {
        fn threadgroupsPerGrid -> MTLSize;
        fn threadsPerThreadgroup -> MTLSize;
    }
    ImageBlockArguments = "_MTLImageBlockArguments" {
        fn height -> u64;
        fn width -> u64;
    }
}

#[cfg(test)]
mod tests {
    use objc2::ClassType;
    use objc2::runtime::AnyClass;

    use super::*;

    /// The private classes resolve to the runtime's classes
    #[test]
    fn private_classes_exist() {
        for name in [
            c"MTLBufferDescriptor",
            c"MTLResourceAddressRangeArray",
            c"_MTLIndirectDrawArguments",
        ] {
            assert!(AnyClass::get(name).is_some(), "{name:?}");
        }
        assert_eq!(MTLBufferDescriptor::class().name(), c"MTLBufferDescriptor");
        assert_eq!(
            IndirectDrawArguments::class().name(),
            c"_MTLIndirectDrawArguments"
        );
    }
}
