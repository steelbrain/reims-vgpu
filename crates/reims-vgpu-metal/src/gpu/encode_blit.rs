//! `id<MTLBlitCommandEncoder>`: core's `BlitEncoder`, whose commands (`BlitCmd`) are one
//! exhaustive `match`, one typed call per variant.

use std::ffi::c_void;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLCommandEncoder as _};
use reims_vgpu_core::num::ToUsize;
use reims_vgpu_core::platform::blit::BlitCmd;
use reims_vgpu_core::raise::{OnFault, RaiseResult};

use crate::gpu::resource;
use crate::msg::encode::ToMtl;
use crate::msg::{self, Policy};
use crate::platform::{MetalPlatform, Obj};
use crate::spi::methods::BlitEncoderSpi;

/// `id<MTLBlitCommandEncoder>`.
pub type MetalBlitEncoder = Obj<ProtocolObject<dyn MTLBlitCommandEncoder>>;

impl reims_vgpu_core::platform::blit::BlitEncoder<MetalPlatform> for MetalBlitEncoder {
    /// Never: the platform copies no guest RAM by its host addresses
    /// (`GpuDevice::copies_guest_ram`), and encodes nothing.
    fn copy_from_guest_ram(
        &self,
        _on: OnFault,
        _runs: &[std::ops::Range<u64>],
        _destination: &crate::gpu::resource::MetalBuffer,
        _destination_offset: u64,
    ) -> RaiseResult<bool> {
        Ok(false)
    }

    /// Never, as `copy_from_guest_ram` above.
    fn copy_to_guest_ram(
        &self,
        _on: OnFault,
        _source: &crate::gpu::resource::MetalBuffer,
        _source_offset: u64,
        _runs: &[std::ops::Range<u64>],
    ) -> RaiseResult<bool> {
        Ok(false)
    }

    /// One call per command: a raise inside Metal is the caller's to catch. Every command but
    /// `EndEncoding` is encoded only under a handler (`Caught`); the resource flush also ends its
    /// encoder with none, so `EndEncoding` is `On(on)`, as the caller says. Raw guest values go
    /// to Metal unvalidated, which is what objc2-metal's `unsafe` marks on the copies: Metal
    /// checks them itself, and raises or faults the GPU on what it refuses.
    fn encode(&self, on: OnFault, cmd: BlitCmd<'_, MetalPlatform>) -> RaiseResult<()> {
        let e = self.get();
        // SAFETY (every `unsafe` call below but `FillTexture`): as above.
        match cmd {
            BlitCmd::CopyBufferToTexture {
                source_buffer,
                source_offset,
                source_bytes_per_row,
                source_bytes_per_image,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
            } => {
                let (source, destination) = (source_buffer.get(), destination_texture.get());
                let (size, origin) = (source_size.mtl(), destination_origin.mtl());
                msg::send(
                    "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                    Policy::Caught,
                    || unsafe {
                        e.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        source,
                        source_offset.to_usize(),
                        source_bytes_per_row.to_usize(),
                        source_bytes_per_image.to_usize(),
                        size,
                        destination,
                        destination_slice.to_usize(),
                        destination_level.to_usize(),
                        origin,
                    )
                    },
                )
            }
            BlitCmd::CopyBufferToTextureOptions {
                source_buffer,
                source_offset,
                source_bytes_per_row,
                source_bytes_per_image,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
                options,
            } => {
                let source = source_buffer.map(Obj::get);
                let destination = destination_texture.map(Obj::get);
                let (size, origin) = (source_size.mtl(), destination_origin.mtl());
                msg::send(
                    "copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
                    Policy::Caught,
                    || {
                        BlitEncoderSpi::copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                        e,
                        source,
                        source_offset,
                        source_bytes_per_row,
                        source_bytes_per_image,
                        size,
                        destination,
                        destination_slice,
                        destination_level,
                        origin,
                        options,
                    )
                    },
                )
            }
            BlitCmd::CopyBufferToBuffer {
                source_buffer,
                source_offset,
                destination_buffer,
                destination_offset,
                size,
            } => {
                let source = source_buffer.map(Obj::get);
                let destination = destination_buffer.map(Obj::get);
                msg::send(
                    "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:",
                    Policy::Caught,
                    || {
                        BlitEncoderSpi::copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            e,
                            source,
                            source_offset,
                            destination,
                            destination_offset,
                            size,
                        )
                    },
                )
            }
            BlitCmd::CopyTextureToBuffer {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_buffer,
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
            } => {
                let (source, destination) = (source_texture.get(), destination_buffer.get());
                let (origin, size) = (source_origin.mtl(), source_size.mtl());
                msg::send(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:",
                    Policy::Caught,
                    || unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage(
                        source,
                        source_slice.to_usize(),
                        source_level.to_usize(),
                        origin,
                        size,
                        destination,
                        destination_offset.to_usize(),
                        destination_bytes_per_row.to_usize(),
                        destination_bytes_per_image.to_usize(),
                    )
                    },
                )
            }
            BlitCmd::CopyTextureToBufferOptions {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_buffer,
                destination_offset,
                destination_bytes_per_row,
                destination_bytes_per_image,
                options,
            } => {
                let source = source_texture.map(Obj::get);
                let destination = destination_buffer.map(Obj::get);
                let (origin, size) = (source_origin.mtl(), source_size.mtl());
                msg::send(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:",
                    Policy::Caught,
                    || {
                        BlitEncoderSpi::copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_options(
                        e,
                        source,
                        source_slice,
                        source_level,
                        origin,
                        size,
                        destination,
                        destination_offset,
                        destination_bytes_per_row,
                        destination_bytes_per_image,
                        options,
                    )
                    },
                )
            }
            BlitCmd::CopyTextureRegion {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
            } => {
                let (source, destination) = (source_texture.get(), destination_texture.get());
                let (origin, size) = (source_origin.mtl(), source_size.mtl());
                let destination_origin = destination_origin.mtl();
                msg::send(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:",
                    Policy::Caught,
                    || unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                        source,
                        source_slice.to_usize(),
                        source_level.to_usize(),
                        origin,
                        size,
                        destination,
                        destination_slice.to_usize(),
                        destination_level.to_usize(),
                        destination_origin,
                    )
                    },
                )
            }
            BlitCmd::CopyTextureRegionOptions {
                source_texture,
                source_slice,
                source_level,
                source_origin,
                source_size,
                destination_texture,
                destination_slice,
                destination_level,
                destination_origin,
                options,
            } => {
                let (source, destination) = (source_texture.get(), destination_texture.get());
                let (origin, size) = (source_origin.mtl(), source_size.mtl());
                let destination_origin = destination_origin.mtl();
                msg::send(
                    "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:",
                    Policy::Caught,
                    || {
                        e.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_options(
                        Some(source),
                        source_slice,
                        source_level,
                        origin,
                        size,
                        Some(destination),
                        destination_slice,
                        destination_level,
                        destination_origin,
                        options,
                    )
                    },
                )
            }
            BlitCmd::CopyTextureSlices {
                source_texture,
                source_slice,
                source_level,
                destination_texture,
                destination_slice,
                destination_level,
                slice_count,
                level_count,
            } => {
                let (source, destination) = (source_texture.get(), destination_texture.get());
                msg::send(
                    "copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:sliceCount:levelCount:",
                    Policy::Caught,
                    || unsafe {
                        e.copyFromTexture_sourceSlice_sourceLevel_toTexture_destinationSlice_destinationLevel_sliceCount_levelCount(
                        source,
                        source_slice.to_usize(),
                        source_level.to_usize(),
                        destination,
                        destination_slice.to_usize(),
                        destination_level.to_usize(),
                        slice_count.to_usize(),
                        level_count.to_usize(),
                    )
                    },
                )
            }
            BlitCmd::CopyIndirectCommandBuffer {
                source,
                source_range,
                destination,
                destination_index,
            } => {
                let (source, destination) = (source.get(), destination.get());
                let (range, index) = (source_range.mtl(), destination_index.to_usize());
                msg::send(
                    "copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:",
                    Policy::Caught,
                    || unsafe {
                        e.copyIndirectCommandBuffer_sourceRange_destination_destinationIndex(
                            source,
                            range,
                            destination,
                            index,
                        )
                    },
                )
            }
            BlitCmd::EndEncoding => msg::send("endEncoding", Policy::On(on), || e.endEncoding()),
            BlitCmd::FillBufferPattern4 {
                buffer,
                range,
                pattern,
            } => {
                let (buffer, range) = (buffer.get(), range.mtl());
                msg::send("fillBuffer:range:pattern4:", Policy::Caught, || {
                    e.fillBuffer_range_pattern4(Some(buffer), range, pattern)
                })
            }
            BlitCmd::FillBuffer {
                buffer,
                range,
                value,
            } => {
                let (buffer, range) = (buffer.get(), range.mtl());
                msg::send("fillBuffer:range:value:", Policy::Caught, || {
                    e.fillBuffer_range_value(buffer, range, value)
                })
            }
            BlitCmd::FillTexture {
                texture,
                level,
                slice,
                region,
                bytes,
            } => {
                let (texture, region) = (texture.get(), region.mtl());
                let pointer = bytes.ptr().as_ptr::<c_void>().cast_const();
                let length = bytes.len();
                // SAFETY: `HostRange::new`'s contract covers the bytes for the call.
                msg::send(
                    "fillTexture:level:slice:region:bytes:length:",
                    Policy::Caught,
                    || unsafe {
                        e.fillTexture_level_slice_region_bytes_length(
                            texture, level, slice, region, pointer, length,
                        )
                    },
                )
            }
            BlitCmd::FillTextureColor {
                texture,
                level,
                slice,
                region,
                color,
                pixel_format,
            } => {
                let (texture, region, color) = (texture.get(), region.mtl(), color.mtl());
                msg::send(
                    "fillTexture:level:slice:region:color:pixelFormat:",
                    Policy::Caught,
                    || {
                        e.fillTexture_level_slice_region_color_pixelFormat(
                            texture,
                            level,
                            slice,
                            region,
                            color,
                            pixel_format,
                        )
                    },
                )
            }
            BlitCmd::GenerateMipmaps { texture } => {
                let texture = texture.map(Obj::get);
                msg::send("generateMipmapsForTexture:", Policy::Caught, || {
                    BlitEncoderSpi::generateMipmapsForTexture(e, texture)
                })
            }
            BlitCmd::InvalidateCompressedTexture { texture } => {
                let texture = texture.get();
                msg::send("invalidateCompressedTexture:", Policy::Caught, || {
                    e.invalidateCompressedTexture(texture)
                })
            }
            BlitCmd::InvalidateCompressedTextureSlice {
                texture,
                slice,
                level,
            } => {
                let texture = texture.get();
                msg::send(
                    "invalidateCompressedTexture:slice:level:",
                    Policy::Caught,
                    || e.invalidateCompressedTexture_slice_level(texture, slice, level),
                )
            }
            BlitCmd::OptimizeForCpu { texture } => {
                let texture = texture.get();
                msg::send("optimizeContentsForCPUAccess:", Policy::Caught, || unsafe {
                    e.optimizeContentsForCPUAccess(texture)
                })
            }
            BlitCmd::OptimizeForCpuSlice {
                texture,
                slice,
                level,
            } => {
                let texture = texture.get();
                let (slice, level) = (slice.to_usize(), level.to_usize());
                msg::send(
                    "optimizeContentsForCPUAccess:slice:level:",
                    Policy::Caught,
                    || unsafe { e.optimizeContentsForCPUAccess_slice_level(texture, slice, level) },
                )
            }
            BlitCmd::OptimizeForGpu { texture } => {
                let texture = texture.get();
                msg::send("optimizeContentsForGPUAccess:", Policy::Caught, || {
                    e.optimizeContentsForGPUAccess(texture)
                })
            }
            BlitCmd::OptimizeForGpuSlice {
                texture,
                slice,
                level,
            } => {
                let texture = texture.get();
                let (slice, level) = (slice.to_usize(), level.to_usize());
                msg::send(
                    "optimizeContentsForGPUAccess:slice:level:",
                    Policy::Caught,
                    || unsafe { e.optimizeContentsForGPUAccess_slice_level(texture, slice, level) },
                )
            }
            BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer,
                range,
            } => {
                let (icb, range) = (indirect_command_buffer.get(), range.mtl());
                msg::send(
                    "optimizeIndirectCommandBuffer:withRange:",
                    Policy::Caught,
                    || unsafe { e.optimizeIndirectCommandBuffer_withRange(icb, range) },
                )
            }
            BlitCmd::ResetCommandsInBuffer { buffer, range } => {
                let (icb, range) = (buffer.get(), range.mtl());
                msg::send(
                    "resetCommandsInBuffer:withRange:",
                    Policy::Caught,
                    || unsafe { e.resetCommandsInBuffer_withRange(icb, range) },
                )
            }
            BlitCmd::SynchronizeResource { resource } => {
                let resource = resource.map(self::resource);
                msg::send("synchronizeResource:", Policy::Caught, || {
                    BlitEncoderSpi::synchronizeResource(e, resource)
                })
            }
            BlitCmd::SynchronizeTexture {
                texture,
                slice,
                level,
            } => {
                let texture = texture.get();
                let (slice, level) = (slice.to_usize(), level.to_usize());
                msg::send(
                    "synchronizeTexture:slice:level:",
                    Policy::Caught,
                    || unsafe { e.synchronizeTexture_slice_level(texture, slice, level) },
                )
            }
            BlitCmd::UpdateFence { fence } => {
                let fence = fence.get();
                msg::send("updateFence:", Policy::Caught, || e.updateFence(fence))
            }
            BlitCmd::WaitForFence { fence } => {
                let fence = fence.get();
                msg::send("waitForFence:", Policy::Caught, || e.waitForFence(fence))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use reims_vgpu_core::raise::OnFault;

    use reims_vgpu_core::platform::blit::*;
    use reims_vgpu_core::platform::command::CommandBuffer as _;

    use objc2_metal::MTLStorageMode;
    use reims_vgpu_core::platform::resource::Buffer as _;

    use reims_vgpu_core::platform::HostRange;
    use reims_vgpu_core::platform::types::*;

    use crate::gpu::resource::testkit::*;

    /// Every BlitEncoder method but fill_texture, endEncoding last
    #[test]
    fn blit_encoder_sends() {
        let k = kit();
        let cb = command_buffer(&k.queue);
        let x = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&k.buffer),
                source_offset: 0,
                destination_buffer: Some(&k.buffer),
                destination_offset: 0,
                size: 16,
            },
        );
        assert!(r.is_ok(), "copy_from_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyBufferToTexture {
                source_buffer: &k.buffer,
                source_offset: 0,
                source_bytes_per_row: 64,
                source_bytes_per_image: 1024,
                source_size: MtlSize::new(1, 1, 1),
                destination_texture: &k.texture,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            },
        );
        assert!(
            r.is_ok(),
            "copy_from_buffer_source_bytes_per_row_source_bytes_per_image_source_size_to_texture_destination_slice_destination_level_destination_origin: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyBufferToTextureOptions {
                source_buffer: Some(&k.buffer),
                source_offset: 0,
                source_bytes_per_row: 64,
                source_bytes_per_image: 1024,
                source_size: MtlSize::new(1, 1, 1),
                destination_texture: Some(&k.texture),
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
                options: 0,
            },
        );
        assert!(
            r.is_ok(),
            "copy_from_buffer_source_bytes_per_row_source_bytes_per_image_source_size_to_texture_destination_slice_destination_level_destination_origin_options: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyTextureSlices {
                source_texture: &k.texture,
                source_slice: 0,
                source_level: 0,
                destination_texture: &k.texture,
                destination_slice: 0,
                destination_level: 0,
                slice_count: 1,
                level_count: 1,
            },
        );
        assert!(r.is_ok(), "copy_from_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyTextureRegion {
                source_texture: &k.texture,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(1, 1, 1),
                destination_texture: &k.texture,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
            },
        );
        assert!(
            r.is_ok(),
            "copy_from_texture_source_origin_source_size_destination_origin: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyTextureRegionOptions {
                source_texture: &k.texture,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(1, 1, 1),
                destination_texture: &k.texture,
                destination_slice: 0,
                destination_level: 0,
                destination_origin: MtlOrigin::new(0, 0, 0),
                options: 0,
            },
        );
        assert!(
            r.is_ok(),
            "copy_from_texture_source_origin_source_size_destination_origin_options: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyTextureToBuffer {
                source_texture: &k.texture,
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(1, 1, 1),
                destination_buffer: &k.buffer,
                destination_offset: 0,
                destination_bytes_per_row: 64,
                destination_bytes_per_image: 1024,
            },
        );
        assert!(
            r.is_ok(),
            "copy_from_texture_source_origin_source_size_to_buffer_destination_offset_destination_bytes_per_row_destination_bytes_per_image: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyTextureToBufferOptions {
                source_texture: Some(&k.texture),
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(1, 1, 1),
                destination_buffer: Some(&k.buffer),
                destination_offset: 0,
                destination_bytes_per_row: 64,
                destination_bytes_per_image: 1024,
                options: 0,
            },
        );
        assert!(
            r.is_ok(),
            "copy_from_texture_source_origin_source_size_to_buffer_destination_offset_destination_bytes_per_row_destination_bytes_per_image_options: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::CopyIndirectCommandBuffer {
                source: &k.icb_compute,
                source_range: NsRange::new(0, 1),
                destination: &k.icb_compute,
                destination_index: 0,
            },
        );
        assert!(r.is_ok(), "copy_indirect_command_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::FillBuffer {
                buffer: &k.buffer,
                range: NsRange::new(0, 1),
                value: 0,
            },
        );
        assert!(r.is_ok(), "fill_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::FillBufferPattern4 {
                buffer: &k.buffer,
                range: NsRange::new(0, 1),
                pattern: 0,
            },
        );
        assert!(r.is_ok(), "fill_buffer_pattern4: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::FillTextureColor {
                texture: &k.texture,
                level: 0,
                slice: 0,
                region: MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(1, 1, 1)),
                color: ClearColor::default(),
                pixel_format: 80,
            },
        );
        assert!(r.is_ok(), "fill_texture_color_pixel_format: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::GenerateMipmaps {
                texture: Some(&k.texture),
            },
        );
        assert!(r.is_ok(), "generate_mipmaps_for_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::InvalidateCompressedTexture {
                texture: &k.texture,
            },
        );
        assert!(r.is_ok(), "invalidate_compressed_texture: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::InvalidateCompressedTextureSlice {
                texture: &k.texture,
                slice: 0,
                level: 0,
            },
        );
        assert!(
            r.is_ok(),
            "invalidate_compressed_texture_slice_level: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::OptimizeForCpu {
                texture: &k.texture,
            },
        );
        assert!(r.is_ok(), "optimize_contents_for_cpu_access: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::OptimizeForCpuSlice {
                texture: &k.texture,
                slice: 0,
                level: 0,
            },
        );
        assert!(
            r.is_ok(),
            "optimize_contents_for_cpu_access_slice_level: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::OptimizeForGpu {
                texture: &k.texture,
            },
        );
        assert!(r.is_ok(), "optimize_contents_for_gpu_access: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::OptimizeForGpuSlice {
                texture: &k.texture,
                slice: 0,
                level: 0,
            },
        );
        assert!(
            r.is_ok(),
            "optimize_contents_for_gpu_access_slice_level: {r:?}"
        );
        let r = x.encode(
            OnFault::Return,
            BlitCmd::OptimizeIndirectCommandBuffer {
                indirect_command_buffer: &k.icb_compute,
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "optimize_indirect_command_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::ResetCommandsInBuffer {
                buffer: &k.icb_compute,
                range: NsRange::new(0, 1),
            },
        );
        assert!(r.is_ok(), "reset_commands_in_buffer: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::SynchronizeResource {
                resource: Some(ResourceRef::Buffer(&k.buffer)),
            },
        );
        assert!(r.is_ok(), "synchronize_resource: {r:?}");
        let r = x.encode(
            OnFault::Return,
            BlitCmd::SynchronizeTexture {
                texture: &k.texture,
                slice: 0,
                level: 0,
            },
        );
        assert!(r.is_ok(), "synchronize_texture: {r:?}");
        let r = x.encode(OnFault::Return, BlitCmd::UpdateFence { fence: &k.fence });
        assert!(r.is_ok(), "update_fence: {r:?}");
        let r = x.encode(OnFault::Return, BlitCmd::WaitForFence { fence: &k.fence });
        assert!(r.is_ok(), "wait_for_fence: {r:?}");
        let r = x.encode(OnFault::Return, BlitCmd::EndEncoding);
        assert!(r.is_ok(), "end_encoding: {r:?}");
    }

    /// Commits `cb` and waits for it.
    fn run(cb: &crate::gpu::command::MetalCommandBuffer) {
        cb.commit(OnFault::Die).unwrap();
        cb.wait_until_completed(OnFault::Die).unwrap();
        assert_eq!(cb.status(), CommandBufferStatus::COMPLETED);
    }

    /// fillBuffer:range:value:, fillBuffer:range:pattern4: and
    /// copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size: write what they say
    #[test]
    fn buffer_fills_and_copies_take_effect() {
        let k = kit();
        let src = buffer(&k.device, 256);
        let dst = buffer(&k.device, 256);
        // SAFETY: a live shared buffer of 256 bytes.
        let src_bytes =
            unsafe { std::slice::from_raw_parts_mut(src.contents().as_ptr::<u8>(), 256) };
        for (i, b) in src_bytes.iter_mut().enumerate() {
            *b = u8::try_from(i).unwrap();
        }
        let cb = command_buffer(&k.queue);
        let x = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        x.encode(
            OnFault::Return,
            BlitCmd::FillBuffer {
                buffer: &dst,
                range: NsRange::new(0, 16),
                value: 0xab,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(&src),
                source_offset: 4,
                destination_buffer: Some(&dst),
                destination_offset: 16,
                size: 16,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            BlitCmd::FillBufferPattern4 {
                buffer: &dst,
                range: NsRange::new(32, 8),
                pattern: 0x0403_0201,
            },
        )
        .unwrap();
        x.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        run(&cb);
        // SAFETY: as above.
        let out = unsafe { std::slice::from_raw_parts(dst.contents().as_ptr::<u8>(), 256) };
        assert_eq!(&out[..16], &[0xab; 16]);
        assert_eq!(&out[16..32], &src_bytes[4..20]);
        assert_eq!(&out[32..40], &[1, 2, 3, 4, 1, 2, 3, 4]);
    }

    /// fillTexture:level:slice:region:bytes:length: (SPI) and
    /// fillTexture:level:slice:region:color:pixelFormat: (SPI) fill the region, which a copy to a
    /// buffer (copyFromTexture:…toBuffer:…options:) reads back
    #[test]
    fn texture_fills_take_effect() {
        let k = kit();
        let t = texture(&k.device, MTLStorageMode::Private);
        let pattern = buffer(&k.device, 64);
        // SAFETY: a live shared buffer of 64 bytes.
        let p = unsafe { std::slice::from_raw_parts_mut(pattern.contents().as_ptr::<u8>(), 64) };
        p[..4].copy_from_slice(&[1, 2, 3, 4]);
        let out = buffer(&k.device, 16 * 16 * 4);
        let cb = command_buffer(&k.queue);
        let x = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let whole = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(16, 16, 1));
        let one = MtlRegion::new(MtlOrigin::new(1, 0, 0), MtlSize::new(1, 1, 1));
        let color = ClearColor {
            red: 0.0,
            green: 0.0,
            blue: 1.0,
            alpha: 1.0,
        };
        x.encode(
            OnFault::Return,
            BlitCmd::FillTextureColor {
                texture: &t,
                level: 0,
                slice: 0,
                region: whole,
                color,
                pixel_format: 80,
            },
        )
        .unwrap();
        // SAFETY: `pattern` holds 4 readable bytes at its start and outlives the send.
        let bytes = unsafe { HostRange::new(pattern.contents(), 4) };
        x.encode(
            OnFault::Return,
            BlitCmd::FillTexture {
                texture: &t,
                level: 0,
                slice: 0,
                region: one,
                bytes,
            },
        )
        .unwrap();
        x.encode(
            OnFault::Return,
            BlitCmd::CopyTextureToBufferOptions {
                source_texture: Some(&t),
                source_slice: 0,
                source_level: 0,
                source_origin: MtlOrigin::new(0, 0, 0),
                source_size: MtlSize::new(16, 16, 1),
                destination_buffer: Some(&out),
                destination_offset: 0,
                destination_bytes_per_row: 64,
                destination_bytes_per_image: 1024,
                options: 0,
            },
        )
        .unwrap();
        x.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        run(&cb);
        // SAFETY: a live shared buffer of 1024 bytes.
        let o = unsafe { std::slice::from_raw_parts(out.contents().as_ptr::<u8>(), 1024) };
        // BGRA8: blue 1.0 is byte 0.
        assert_eq!(&o[..4], &[255, 0, 0, 255]);
        assert_eq!(&o[4..8], &[1, 2, 3, 4]);
    }
}
