//! Golden test: every method `spi` declares matches the runtime.
//!
//! objc2's own checks run only in dev builds, at a send (D36), and objc2-metal's declarations come
//! from the SDK's headers, so the declarations of our own are checked here, all at once: [`SPI`]
//! lists each method `spi` declares, with its receiver and the runtime's type encoding, and the
//! test compares it with the method of the class Metal hands out for each receiver: the system
//! device's concrete class, the concrete class of a live object of the family (an AGX encoder,
//! texture, heap …), or the object `+new` makes for a descriptor class. The types compare as
//! objc2's `verify_sel` compares them.
//!
//! The render-encoder SPI the host's Metal does not implement ([`UNIMPLEMENTED_ON_HOST`]) must
//! stay unimplemented: the render decoder raises for those records without a send. The negative
//! cases show that the check is live: a wrong width or a wrong struct tag fails it.
//!
//! The check sends nothing. A mistyped send inside a live render encoder panics in objc2's check,
//! the panic unwinds past `endEncoding`, and the encoder's dealloc then hits Metal's assertion
//! `Command encoder released without endEncoding` (SIGTRAP), so a declaration error there would
//! show as that trap rather than as a failed test (measured, objc2 0.6.4, macOS 26.6).

#![cfg(target_os = "macos")] // Metal is macOS's: the crate is empty elsewhere
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    missing_docs
)]

use std::ffi::CString;

use objc2::AnyThread as _;
use objc2::encode::{Encoding, EncodingBox};
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Bool, Method, ProtocolObject, Sel};
use objc2::{Message, sel};
use objc2_foundation::NSString;
use objc2_metal::*;
use reims_vgpu_core::platform::types::ResourceAddressRange;
use reims_vgpu_metal::MetalPlatform;
use reims_vgpu_metal::msg::encode::Enc;

fn device() -> Retained<ProtocolObject<dyn MTLDevice>> {
    let d = MetalPlatform::system_default_device().expect("a Metal device");
    d.get().retain()
}

fn device_class() -> &'static AnyClass {
    let d = MetalPlatform::system_default_device().expect("a Metal device");
    let obj: &AnyObject = d.get().as_ref();
    obj.class()
}

fn class_of<T: Message + ?Sized>(o: &T) -> &'static AnyClass {
    // SAFETY: every `T: Message` is an Objective-C object.
    let o: &AnyObject = unsafe { &*std::ptr::from_ref(o).cast::<AnyObject>() };
    o.class()
}

/// Metal Shading Language for the pipeline states whose classes the check needs.
const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct V { float4 p [[position]]; };
kernel void k(device uint *b [[buffer(0)]], uint i [[thread_position_in_grid]]) { b[i] = i; }
vertex V v(uint i [[vertex_id]]) { V o; o.p = float4(0, 0, 0, 1); return o; }
fragment float4 f() { return float4(1); }
"#;

/// The runtime class of one live object of every protocol family the declarations name, and
/// of `+new`'s object for the descriptor classes (a class cluster hands out a subclass).
fn runtime_classes() -> Vec<(&'static str, &'static AnyClass)> {
    let d = device();
    let mut m: Vec<(&'static str, &'static AnyClass)> = Vec::new();
    let q = d.newCommandQueue().unwrap();
    m.push(("MTLCommandQueue", class_of(&*q)));
    let td = MTLTextureDescriptor::new();
    // SAFETY: plain property sets on a fresh descriptor.
    unsafe {
        td.setWidth(16);
        td.setHeight(16);
    }
    td.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    td.setStorageMode(MTLStorageMode::Private);
    td.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    let tex = d.newTextureWithDescriptor(&td).unwrap();
    m.push(("MTLTexture", class_of(&*tex)));
    let buf = d
        .newBufferWithLength_options(256, MTLResourceOptions::empty())
        .unwrap();
    m.push(("MTLBuffer", class_of(&*buf)));
    let cb = q.commandBuffer().unwrap();
    m.push(("MTLCommandBuffer", class_of(&*cb)));
    let e = cb.blitCommandEncoder().unwrap();
    e.endEncoding();
    m.push(("MTLBlitCommandEncoder", class_of(&*e)));
    let e = cb.computeCommandEncoder().unwrap();
    e.endEncoding();
    m.push(("MTLComputeCommandEncoder", class_of(&*e)));
    let rp = MTLRenderPassDescriptor::new();
    // SAFETY: index 0 of the colour attachments.
    let ca = unsafe { rp.colorAttachments().objectAtIndexedSubscript(0) };
    ca.setTexture(Some(&tex));
    ca.setLoadAction(MTLLoadAction::Clear);
    ca.setStoreAction(MTLStoreAction::Store);
    let e = cb.renderCommandEncoderWithDescriptor(&rp).unwrap();
    e.endEncoding();
    m.push(("MTLRenderCommandEncoder", class_of(&*e)));
    let hd = MTLHeapDescriptor::new();
    hd.setSize(1 << 20);
    hd.setStorageMode(MTLStorageMode::Private);
    m.push(("MTLHeap", class_of(&*d.newHeapWithDescriptor(&hd).unwrap())));
    let sd = MTLSamplerDescriptor::new();
    sd.setSupportArgumentBuffers(true);
    m.push((
        "MTLSamplerState",
        class_of(&*d.newSamplerStateWithDescriptor(&sd).unwrap()),
    ));
    m.push((
        "MTLDepthStencilState",
        class_of(
            &*d.newDepthStencilStateWithDescriptor(&MTLDepthStencilDescriptor::new())
                .unwrap(),
        ),
    ));
    let lib = d
        .newLibraryWithSource_options_error(&NSString::from_str(MSL), None)
        .unwrap();
    m.push(("MTLLibrary", class_of(&*lib)));
    let func = |n: &str| lib.newFunctionWithName(&NSString::from_str(n)).unwrap();
    m.push(("MTLFunction", class_of(&*func("k"))));
    let cd = MTLComputePipelineDescriptor::new();
    cd.setComputeFunction(Some(&func("k")));
    cd.setSupportIndirectCommandBuffers(true);
    let cps = d
        .newComputePipelineStateWithDescriptor_options_reflection_error(
            &cd,
            MTLPipelineOption::None,
            None,
        )
        .unwrap();
    m.push(("MTLComputePipelineState", class_of(&*cps)));
    let rd = MTLRenderPipelineDescriptor::new();
    rd.setVertexFunction(Some(&func("v")));
    rd.setFragmentFunction(Some(&func("f")));
    // SAFETY: index 0 of the colour attachments.
    unsafe { rd.colorAttachments().objectAtIndexedSubscript(0) }
        .setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    rd.setSupportIndirectCommandBuffers(true);
    m.push((
        "MTLRenderPipelineState",
        class_of(&*d.newRenderPipelineStateWithDescriptor_error(&rd).unwrap()),
    ));
    m.push(("MTLSharedEvent", class_of(&*d.newSharedEvent().unwrap())));
    let rmd = MTLRasterizationRateMapDescriptor::new();
    let size = MTLSize {
        width: 2,
        height: 2,
        depth: 0,
    };
    // SAFETY: `+alloc` then `-initWithSampleCount:` with a small count.
    let layer = unsafe {
        MTLRasterizationRateLayerDescriptor::initWithSampleCount(
            MTLRasterizationRateLayerDescriptor::alloc(),
            size,
        )
    };
    m.push(("MTLRasterizationRateLayerDescriptor", class_of(&*layer)));
    // SAFETY: the screen size and layer 0 of a fresh descriptor.
    unsafe {
        rmd.setScreenSize(MTLSize {
            width: 64,
            height: 64,
            depth: 0,
        });
        rmd.setLayer_atIndex(Some(&layer), 0);
    }
    m.push((
        "MTLRasterizationRateMap",
        class_of(&*d.newRasterizationRateMapWithDescriptor(&rmd).unwrap()),
    ));
    m.push(("MTLRasterizationRateLayerArray", class_of(&*rmd.layers())));
    let icb = |types: MTLIndirectCommandType| {
        let id = MTLIndirectCommandBufferDescriptor::new();
        id.setCommandTypes(types);
        id.setMaxVertexBufferBindCount(2);
        id.setMaxFragmentBufferBindCount(2);
        id.setMaxKernelBufferBindCount(2);
        id.setInheritPipelineState(false);
        id.setInheritBuffers(false);
        // SAFETY: a valid descriptor; 4 commands.
        unsafe {
            d.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &id,
                4,
                MTLResourceOptions::empty(),
            )
        }
        .unwrap()
    };
    let icb_r = icb(MTLIndirectCommandType::Draw | MTLIndirectCommandType::DrawIndexed);
    let icb_c = icb(MTLIndirectCommandType::ConcurrentDispatch);
    m.push(("MTLIndirectCommandBuffer", class_of(&*icb_r)));
    // SAFETY: index 0 of 4.
    m.push((
        "MTLIndirectRenderCommand",
        class_of(&*unsafe { icb_r.indirectRenderCommandAtIndex(0) }),
    ));
    // SAFETY: index 0 of 4.
    m.push((
        "MTLIndirectComputeCommand",
        class_of(&*unsafe { icb_c.indirectComputeCommandAtIndex(0) }),
    ));
    let rp = MTLRenderPassDescriptor::new();
    m.push((
        "MTLRenderPassColorAttachmentDescriptorArray",
        class_of(&*rp.colorAttachments()),
    ));
    // SAFETY: index 0.
    m.push((
        "MTLRenderPassColorAttachmentDescriptor",
        class_of(&*unsafe { rp.colorAttachments().objectAtIndexedSubscript(0) }),
    ));
    m.push((
        "MTLRenderPassDepthAttachmentDescriptor",
        class_of(&*rp.depthAttachment()),
    ));
    m.push((
        "MTLRenderPassStencilAttachmentDescriptor",
        class_of(&*rp.stencilAttachment()),
    ));
    let rpd = MTLRenderPipelineDescriptor::new();
    m.push((
        "MTLRenderPipelineColorAttachmentDescriptorArray",
        class_of(&*rpd.colorAttachments()),
    ));
    // SAFETY: index 0.
    m.push((
        "MTLRenderPipelineColorAttachmentDescriptor",
        class_of(&*unsafe { rpd.colorAttachments().objectAtIndexedSubscript(0) }),
    ));
    m
}

/// The runtime class of the receiver a row names: the device's for
/// `MTLDevice`, a live object's for a protocol family, `+new`'s object for a descriptor class.
fn class_named(name: &str, live: &[(&'static str, &'static AnyClass)]) -> &'static AnyClass {
    if name == "MTLDevice" {
        return device_class();
    }
    if let Some((_, c)) = live.iter().find(|(n, _)| *n == name) {
        return c;
    }
    let c = CString::new(name).unwrap();
    let cls = AnyClass::get(&c).unwrap_or_else(|| panic!("no class {name}"));
    if name.starts_with("_MTL") {
        return cls;
    }
    // A descriptor class: the object `+new` makes (a class cluster's subclass).
    // SAFETY: `+new` on an NSObject subclass returns +1 or nil.
    let o: Option<Retained<AnyObject>> = unsafe { objc2::msg_send![cls, new] };
    o.map_or(cls, |o| o.class())
}

/// `e` as a static `Encoding`, so that objc2's own equivalence compares it (leaked: a test's
/// few hundred rows).
fn to_static(e: &EncodingBox) -> Encoding {
    fn leak<T>(t: T) -> &'static T {
        Box::leak(Box::new(t))
    }
    fn fields(v: &[EncodingBox]) -> &'static [Encoding] {
        Box::leak(
            v.iter()
                .map(to_static)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    }
    fn name(n: &str) -> &'static str {
        Box::leak(n.to_owned().into_boxed_str())
    }
    match e {
        EncodingBox::Char => Encoding::Char,
        EncodingBox::Short => Encoding::Short,
        EncodingBox::Int => Encoding::Int,
        EncodingBox::Long => Encoding::Long,
        EncodingBox::LongLong => Encoding::LongLong,
        EncodingBox::UChar => Encoding::UChar,
        EncodingBox::UShort => Encoding::UShort,
        EncodingBox::UInt => Encoding::UInt,
        EncodingBox::ULong => Encoding::ULong,
        EncodingBox::ULongLong => Encoding::ULongLong,
        EncodingBox::Float => Encoding::Float,
        EncodingBox::Double => Encoding::Double,
        EncodingBox::LongDouble => Encoding::LongDouble,
        EncodingBox::Bool => Encoding::Bool,
        EncodingBox::Void => Encoding::Void,
        EncodingBox::String => Encoding::String,
        EncodingBox::Object => Encoding::Object,
        EncodingBox::Block => Encoding::Block,
        EncodingBox::Class => Encoding::Class,
        EncodingBox::Sel => Encoding::Sel,
        EncodingBox::Unknown => Encoding::Unknown,
        EncodingBox::Pointer(t) => Encoding::Pointer(leak(to_static(t))),
        EncodingBox::Atomic(t) => Encoding::Atomic(leak(to_static(t))),
        EncodingBox::Array(n, t) => Encoding::Array(*n, leak(to_static(t))),
        EncodingBox::Struct(n, f) => Encoding::Struct(name(n), fields(f)),
        EncodingBox::Union(n, f) => Encoding::Union(name(n), fields(f)),
        other => panic!("no encoding in SPI spells {other}"),
    }
}

/// objc2's comparison of a declared type with the runtime's (`verify_sel`): register-sized
/// integers interoperate whatever their sign, anything else by `equivalent_to_box`.
fn equivalent(declared: &Encoding, runtime: &EncodingBox) -> bool {
    let reg = |e: &Encoding| {
        matches!(
            e,
            Encoding::LongLong | Encoding::ULongLong | Encoding::Long | Encoding::ULong
        )
    };
    let reg_box = |e: &EncodingBox| {
        matches!(
            e,
            EncodingBox::LongLong | EncodingBox::ULongLong | EncodingBox::Long | EncodingBox::ULong
        )
    };
    (reg(declared) && reg_box(runtime)) || declared.equivalent_to_box(runtime)
}

/// One method `spi` declares, on one receiver.
#[derive(Debug, Clone, Copy)]
struct Row {
    /// The Objective-C class that receives it (the class for a class method).
    receiver: &'static str,
    /// The full selector.
    selector: &'static str,
    /// The runtime's type encoding without frame offsets (return, `@:`, arguments).
    encoding: &'static str,
}

/// Every method `spi` declares that core sends, one line per receiver: the class, the selector
/// and the encoding ([`Row`]). From `spi/methods.rs` (the private selectors and the re-declared
/// public ones) and `spi/classes.rs` (the private classes' methods).
const SPI: &str = "\
MTLBufferDescriptor new @@:
MTLRasterizationRateLayerDescriptor initWithSampleCount: @@:{?=QQQ}
MTLResourceAddressRangeArray initWithRanges:count: @@:^{MTLAddressRange=QQ}Q
MTLDevice allowLibrariesFromOtherPlatforms v@:
MTLDevice deviceLinearTextureAlignmentBytes Q@:
MTLDevice doubleFPConfig Q@:
MTLDevice featureProfile Q@:
MTLDevice gpuCoreCount I@:
MTLDevice halfFPConfig Q@:
MTLDevice iosurfaceReadOnlyTextureAlignmentBytes Q@:
MTLDevice iosurfaceTextureAlignmentBytes Q@:
MTLDevice isFramebufferReadSupported B@:
MTLDevice isLargeMRTSupported B@:
MTLDevice isRGB10A2GammaSupported B@:
MTLDevice linearTextureAlignmentBytes Q@:
MTLDevice maxComputeBuffers Q@:
MTLDevice maxComputeLocalMemorySizes Q@:
MTLDevice maxComputeSamplers Q@:
MTLDevice maxComputeTextures Q@:
MTLDevice maxComputeThreadgroupMemory Q@:
MTLDevice maxComputeThreadgroupMemoryAlignmentBytes Q@:
MTLDevice maxFragmentBuffers Q@:
MTLDevice maxFragmentSamplers Q@:
MTLDevice maxFragmentTextures Q@:
MTLDevice maxPredicatedNestingDepth Q@:
MTLDevice maxRasterizationRateLayerCount Q@:
MTLDevice maxTextureHeight2D Q@:
MTLDevice maxTextureLayers Q@:
MTLDevice maxTextureWidth2D Q@:
MTLDevice maxTileBuffers Q@:
MTLDevice maxTileSamplers Q@:
MTLDevice maxTileTextures Q@:
MTLDevice maxTotalComputeThreadsPerThreadgroup Q@:
MTLDevice maxVertexAmplificationCount Q@:
MTLDevice maxVertexBuffers Q@:
MTLDevice maxVertexSamplers Q@:
MTLDevice maxVertexTextures Q@:
MTLDevice maxViewportCount Q@:
MTLDevice name @@:
MTLDevice newBufferWithBytesNoCopy:length:options:deallocator: @@:^vQQ@?
MTLDevice newBufferWithDescriptor: @@:@
MTLDevice newBufferWithIOSurface: @@:^{__IOSurface=}
MTLDevice newComputePipelineStateWithDescriptor:options:reflection:error: @@:@Q^@^@
MTLDevice newComputePipelineStateWithFunction:error: @@:@^@
MTLDevice newLibraryWithData:error: @@:@^@
MTLDevice newLibraryWithSource:options:error: @@:@@^@
MTLDevice newRenderPipelineStateWithDescriptor:error: @@:@^@
MTLDevice newRenderPipelineStateWithDescriptor:options:reflection:error: @@:@Q^@^@
MTLDevice newRenderPipelineStateWithTileDescriptor:options:reflection:error: @@:@Q^@^@
MTLDevice newSharedEventWithHandle: @@:@
MTLDevice newSharedTextureWithHandle:withResourceIndex: @@:@Q
MTLDevice newTextureWithDescriptor:iosurface:plane: @@:@^{__IOSurface=}Q
MTLDevice reserveResourceIndicesForResourceType:indices:indexCount: v@:Q^QQ
MTLDevice singleFPConfig Q@:
MTLDevice supportsBufferWithAddressRanges B@:
MTLDevice supportsBufferWithIOSurface B@:
MTLDevice supportsCommandBufferJump B@:
MTLDevice supportsComputeCompressedTextureWrite B@:
MTLDevice supportsConditionalLoadStore B@:
MTLDevice supportsDynamicAttributeStride B@:
MTLDevice supportsExtendedXR10Formats B@:
MTLDevice supportsFloat16BCubicFiltering B@:
MTLDevice supportsHeapWithAddressRanges B@:
MTLDevice supportsImageBlocks B@:
MTLDevice supportsMemoryOrderAtomics B@:
MTLDevice supportsNativeHardwareFP16 B@:
MTLDevice supportsNonSquareTileShaders B@:
MTLDevice supportsPlacementSparse B@:
MTLDevice supportsPrimitiveType: B@:Q
MTLDevice supportsProgrammableSamplePositions B@:
MTLDevice supportsResourceDetachBacking B@:
MTLDevice supportsSharedTextureHandles B@:
MTLDevice supportsSIMDReduction B@:
MTLDevice supportsSIMDShuffleAndFill B@:
MTLDevice supportsTexture2DMultisampleArray B@:
MTLDevice supportsTextureWriteRoundingMode: B@:q
MTLDevice supportsTileShaders B@:
MTLCommandBuffer encodeSignalEvent:value: v@:@Q
MTLCommandBuffer encodeSignalEventScheduled:value: v@:@Q
MTLCommandBuffer encodeWaitForEvent:value: v@:@Q
MTLCommandBuffer setProtectionOptions: v@:Q
MTLTextureDescriptor setProtectionOptions: v@:Q
MTLBlitCommandEncoder copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options: v@:@QQQ{?=QQQ}@QQ{?=QQQ}Q
MTLBlitCommandEncoder copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size: v@:@Q@QQ
MTLBlitCommandEncoder copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options: v@:@QQ{?=QQQ}{?=QQQ}@QQQQ
MTLBlitCommandEncoder copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options: v@:@QQ{?=QQQ}{?=QQQ}@QQ{?=QQQ}Q
MTLBlitCommandEncoder fillBuffer:range:pattern4: v@:@{_NSRange=QQ}I
MTLBlitCommandEncoder fillTexture:level:slice:region:bytes:length: v@:@QQ{?={?=QQQ}{?=QQQ}}^vQ
MTLBlitCommandEncoder fillTexture:level:slice:region:color:pixelFormat: v@:@QQ{?={?=QQQ}{?=QQQ}}{?=dddd}Q
MTLBlitCommandEncoder generateMipmapsForTexture: v@:@
MTLBlitCommandEncoder invalidateCompressedTexture: v@:@
MTLBlitCommandEncoder invalidateCompressedTexture:slice:level: v@:@QQ
MTLBlitCommandEncoder synchronizeResource: v@:@
MTLComputeCommandEncoder dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup: v@:@Q{?=QQQ}
MTLComputeCommandEncoder dispatchThreadsWithIndirectBuffer:indirectBufferOffset: v@:@Q
MTLComputeCommandEncoder encodeEndDoWhile:offset:comparison:referenceValue: B@:@QQI
MTLComputeCommandEncoder encodeEndIf B@:
MTLComputeCommandEncoder encodeEndWhile B@:
MTLComputeCommandEncoder encodeStartDoWhile v@:
MTLComputeCommandEncoder encodeStartElse v@:
MTLComputeCommandEncoder encodeStartIf:offset:comparison:referenceValue: v@:@QQI
MTLComputeCommandEncoder encodeStartWhile:offset:comparison:referenceValue: v@:@QQI
MTLComputeCommandEncoder insertCompressedTextureReinterpretationFlush v@:
MTLComputeCommandEncoder setComputePipelineState: v@:@
MTLIndirectComputeCommand setComputePipelineState: v@:@
MTLRenderCommandEncoder dispatchThreadsPerTile:inRegion: v@:{?=QQQ}{?={?=QQQ}{?=QQQ}}
MTLRenderCommandEncoder dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex: v@:{?=QQQ}{?={?=QQQ}{?=QQQ}}I
MTLRenderCommandEncoder setAlphaTestReferenceValue: v@:f
MTLRenderCommandEncoder setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex: v@:@QQQBQ
MTLRenderCommandEncoder setDepthCleared v@:
MTLRenderCommandEncoder setDepthClipModeSPI: v@:Q
MTLRenderCommandEncoder setDepthResolveTexture:slice:depthPlane:level:yInvert: v@:@QQQB
MTLRenderCommandEncoder setLineWidth: v@:f
MTLRenderCommandEncoder setPointSize: v@:f
MTLRenderCommandEncoder setPrimitiveRestartEnabled:index: v@:BQ
MTLRenderCommandEncoder setProvokingVertexMode: v@:Q
MTLIndirectRenderCommand setRenderPipelineState: v@:@
MTLRenderCommandEncoder setRenderPipelineState: v@:@
MTLRenderCommandEncoder setStencilCleared v@:
MTLRenderCommandEncoder setStencilResolveTexture:slice:depthPlane:level:yInvert: v@:@QQQB
MTLRenderCommandEncoder setVertexAmplificationMode:value: v@:QQ
MTLRenderCommandEncoder setViewportTransformEnabled: v@:B
MTLBuffer contents ^v@:
MTLBuffer detachBacking B@:
MTLHeap detachBacking B@:
MTLBuffer iosurface ^{__IOSurface=}@:
MTLTexture iosurface ^{__IOSurface=}@:
MTLBuffer newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage: @@:@QQQ
MTLBuffer replaceBackingWithRanges:readOnly: B@:@B
MTLHeap replaceBackingWithRanges:readOnly: B@:@B
MTLTexture newTextureViewWithPixelFormat:resourceIndex: @@:QQ
MTLTexture newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:resourceIndex: @@:QQ{_NSRange=QQ}{_NSRange=QQ}{?=CCCC}Q
MTLTexture newTextureViewWithPixelFormat:textureType:levels:slices:resourceIndex: @@:QQ{_NSRange=QQ}{_NSRange=QQ}Q
MTLTexture protectionOptions Q@:
MTLTexture replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage: v@:{?={?=QQQ}{?=QQQ}}QQ^vQQ
MTLComputePipelineState resourceIndex Q@:
MTLDepthStencilState resourceIndex Q@:
MTLIndirectCommandBuffer resourceIndex Q@:
MTLRenderPipelineState resourceIndex Q@:
MTLSamplerState resourceIndex Q@:
MTLTexture resourceIndex Q@:
MTLComputePipelineState uniqueIdentifier Q@:
MTLIndirectCommandBuffer uniqueIdentifier Q@:
MTLRenderPipelineState uniqueIdentifier Q@:
MTLTexture uniqueIdentifier Q@:
MTLHeap gpuAddress Q@:
MTLLibrary functionNames @@:
MTLFunction newArgumentEncoderWithBufferIndex:reflection: @@:Q^@
MTLSharedEvent newSharedEventHandle @@:
MTLRasterizationRateMap mutability Q@:
MTLRasterizationRateMap resetUsingDescriptor: v@:@
MTLRasterizationRateMapDescriptor setMinFactor: v@:f
MTLRasterizationRateMapDescriptor setMutability: v@:Q
MTLRasterizationRateLayerDescriptor horizontalSampleStorage ^f@:
MTLRasterizationRateLayerDescriptor sampleCount {?=QQQ}@:
MTLRasterizationRateLayerDescriptor verticalSampleStorage ^f@:
MTLIndirectCommandBuffer indirectComputeCommandAtIndex: @@:Q
MTLIndirectCommandBuffer indirectRenderCommandAtIndex: @@:Q
MTLIndirectRenderCommand drawArguments @@:
MTLIndirectRenderCommand drawIndexedArguments @@:
MTLIndirectRenderCommand drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride: v@:QQQ@Q@QQQ@QQ
MTLIndirectRenderCommand drawIndexedPatchesArguments @@:
MTLIndirectRenderCommand drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance: v@:QQQ@QQqQ
MTLIndirectRenderCommand drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:tessellationFactorBuffer:tessellationFactorBufferOffset:tessellationFactorBufferInstanceStride: v@:QQQ@QQQ@QQ
MTLIndirectRenderCommand drawPatchesArguments @@:
MTLIndirectComputeCommand getCommandType Q@:
MTLIndirectRenderCommand getCommandType Q@:
MTLIndirectRenderCommand getFragmentBufferAddressAtIndex: Q@:Q
MTLIndirectComputeCommand getPipelineStateUniqueIdentifier Q@:
MTLIndirectRenderCommand getPipelineStateUniqueIdentifier Q@:
MTLIndirectRenderCommand getTessellationFactorArguments @@:
MTLIndirectRenderCommand getVertexAttributeStrideAtIndex: Q@:Q
MTLIndirectRenderCommand getVertexBufferAddressAtIndex: Q@:Q
MTLIndirectComputeCommand hasBarrier B@:
MTLIndirectRenderCommand hasBarrier B@:
MTLIndirectRenderCommand setFragmentBuffer:offset:atIndex: v@:@QQ
MTLIndirectRenderCommand setVertexBuffer:offset:atIndex: v@:@QQ
MTLIndirectRenderCommand setVertexBuffer:offset:attributeStride:atIndex: v@:@QQQ
MTLIndirectComputeCommand dispatchThreadgroupsArguments @@:
MTLIndirectComputeCommand dispatchThreadsArguments @@:
MTLIndirectComputeCommand getImageBlockSize @@:
MTLIndirectComputeCommand getKernelAttributeStrideAtIndex: Q@:Q
MTLIndirectComputeCommand getKernelBufferAddressAtIndex: Q@:Q
MTLIndirectComputeCommand getStageInRegion {?={?=QQQ}{?=QQQ}}@:
MTLIndirectComputeCommand getThreadgroupMemoryLengthAtIndex: Q@:Q
MTLIndirectComputeCommand setKernelBuffer:offset:atIndex: v@:@QQ
MTLIndirectComputeCommand setKernelBuffer:offset:attributeStride:atIndex: v@:@QQQ
_MTLIndirectDrawArguments baseInstance Q@:
_MTLIndirectDrawIndexedArguments baseInstance Q@:
_MTLIndirectDrawIndexedPatchesArguments baseInstance Q@:
_MTLIndirectDrawPatchesArguments baseInstance Q@:
_MTLIndirectDrawArguments instanceCount Q@:
_MTLIndirectDrawIndexedArguments instanceCount Q@:
_MTLIndirectDrawIndexedPatchesArguments instanceCount Q@:
_MTLIndirectDrawPatchesArguments instanceCount Q@:
_MTLIndirectDrawArguments primitiveType Q@:
_MTLIndirectDrawIndexedArguments primitiveType Q@:
_MTLIndirectDrawArguments vertexCount Q@:
_MTLIndirectDrawArguments vertexStart Q@:
_MTLIndirectDrawIndexedArguments baseVertex Q@:
_MTLIndirectDrawIndexedArguments indexBufferGPUVirtualAddress Q@:
_MTLIndirectDrawIndexedArguments indexCount Q@:
_MTLIndirectDrawIndexedArguments indexType Q@:
_MTLIndirectDrawIndexedPatchesArguments numberOfPatchControlPoints Q@:
_MTLIndirectDrawPatchesArguments numberOfPatchControlPoints Q@:
_MTLIndirectDrawIndexedPatchesArguments patchCount Q@:
_MTLIndirectDrawPatchesArguments patchCount Q@:
_MTLIndirectDrawIndexedPatchesArguments patchIndexBufferGPUVirtualAddress Q@:
_MTLIndirectDrawPatchesArguments patchIndexBufferGPUVirtualAddress Q@:
_MTLIndirectDrawIndexedPatchesArguments patchStart Q@:
_MTLIndirectDrawPatchesArguments patchStart Q@:
_MTLIndirectDrawIndexedPatchesArguments controlPointIndexBufferGPUVirtualAddress Q@:
_MTLIndirectTessellationFactorArguments GPUVirtualAddress Q@:
_MTLIndirectTessellationFactorArguments instanceStride Q@:
_MTLIndirectTessellationFactorArguments scale f@:
_MTLIndirectDispatchThreadgroupsArguments threadgroupsPerGrid {?=QQQ}@:
_MTLIndirectDispatchThreadgroupsArguments threadsPerThreadgroup {?=QQQ}@:
_MTLIndirectDispatchThreadsArguments threadsPerThreadgroup {?=QQQ}@:
_MTLIndirectDispatchThreadsArguments threadsPerGrid {?=QQQ}@:
_MTLImageBlockArguments height Q@:
_MTLImageBlockArguments width Q@:
MTLTextureDescriptor rotation Q@:
MTLComputePipelineDescriptor setForceResourceIndex: v@:B
MTLRenderPipelineDescriptor setForceResourceIndex: v@:B
MTLTextureDescriptor setForceResourceIndex: v@:B
MTLTextureDescriptor setFramebufferOnly: v@:B
MTLTextureDescriptor setIsDrawable: v@:B
MTLComputePipelineDescriptor setResourceIndex: v@:Q
MTLIndirectCommandBufferDescriptor setResourceIndex: v@:Q
MTLRenderPipelineDescriptor setResourceIndex: v@:Q
MTLSamplerDescriptor setResourceIndex: v@:Q
MTLTextureDescriptor setResourceIndex: v@:Q
MTLTextureDescriptor setRotation: v@:Q
MTLTextureDescriptor setWriteSwizzleEnabled: v@:B
MTLBufferDescriptor resourceOptions Q@:
MTLBufferDescriptor setContents: v@:^v
MTLBufferDescriptor setDeallocator: v@:@?
MTLBufferDescriptor setLength: v@:Q
MTLBufferDescriptor setNoCopy: v@:B
MTLBufferDescriptor setPinnedGPUAddress: v@:Q
MTLHeapDescriptor setPinnedGPUAddress: v@:Q
MTLBufferDescriptor setResourceOptions: v@:Q
MTLBufferDescriptor setStorageMode: v@:Q
MTLHeapDescriptor setAddressRanges: v@:@
MTLRenderPassDescriptor setDefaultColorSampleCount: v@:Q
MTLRenderPassColorAttachmentDescriptorArray objectAtIndexedSubscript: @@:Q
MTLRenderPipelineColorAttachmentDescriptorArray objectAtIndexedSubscript: @@:Q
MTLIndirectCommandBufferDescriptor setAllowOverrideRenderStates: v@:q
MTLIndirectCommandBufferDescriptor setInheritBlendColor: v@:B
MTLIndirectCommandBufferDescriptor setInheritDepthTestBounds: v@:B
MTLIndirectCommandBufferDescriptor setInheritScissorRects: v@:B
MTLIndirectCommandBufferDescriptor setInheritStencilReferenceValues: v@:B
MTLIndirectCommandBufferDescriptor setInheritViewports: v@:B
MTLIndirectCommandBufferDescriptor setMaxScissorRectCount: v@:Q
MTLIndirectCommandBufferDescriptor setMaxViewportCount: v@:Q
";

/// [`SPI`]'s rows.
fn spi() -> Vec<Row> {
    SPI.lines()
        .map(|line| {
            let f: Vec<&'static str> = line.split(' ').collect();
            assert_eq!(f.len(), 3, "{line}");
            Row {
                receiver: f[0],
                selector: f[1],
                encoding: f[2],
            }
        })
        .collect()
}

/// A row's encoding as its return type and argument types (receiver and `_cmd` dropped).
fn row_types(row: &Row) -> (Encoding, Vec<Encoding>) {
    let mut s = row.encoding;
    let mut all = Vec::new();
    while !s.is_empty() {
        all.push(to_static(&EncodingBox::from_start_of_str(&mut s).unwrap()));
    }
    assert!(all.len() >= 3, "{}: {}", row.selector, row.encoding);
    let ret = all.remove(0);
    all.drain(..2);
    (ret, all)
}

/// Compares a row's types with the runtime's method.
fn check_method(row: &Row, m: &Method) -> Result<(), String> {
    let (ret, args) = row_types(row);
    let parse = |s: &std::ffi::CStr| s.to_str().unwrap().parse::<EncodingBox>().unwrap();
    let rt = parse(&m.return_type());
    if !equivalent(&ret, &rt) {
        return Err(format!("returns {rt}, the row says {ret}"));
    }
    let count = m.arguments_count();
    if count != args.len() + 2 {
        return Err(format!(
            "takes {} arguments, the row says {}",
            count - 2,
            args.len()
        ));
    }
    for (i, a) in args.iter().enumerate() {
        let at = parse(&m.argument_type(i + 2).unwrap());
        if !equivalent(a, &at) {
            return Err(format!("argument {i} is {at}, the row says {a}"));
        }
    }
    Ok(())
}

/// The `MTLRenderCommandEncoderSPI` selectors the render decoder raises for without a send
/// (`deserializer/render/state.rs`), because the host's AGX render encoder does not implement them
/// (measured: `class_getInstanceMethod` is NULL on macOS 26.6).
const UNIMPLEMENTED_ON_HOST: &[&str] = &[
    "setClipPlane:p2:p3:p4:atIndex:",
    "setFragmentSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
    "setTransformFeedbackState:",
    "setTriangleFrontFillMode:backFillMode:",
    "setVertexSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
];

/// Every method `spi` declares matches the runtime's method on each of its receivers (D41)
#[test]
fn every_spi_declaration_matches_the_runtime() {
    let live = runtime_classes();
    let rows = spi();
    let mut failures = Vec::new();
    for row in &rows {
        let sel = Sel::register(&CString::new(row.selector).unwrap());
        let cls = class_named(row.receiver, &live);
        let Some(m) = cls.instance_method(sel).or_else(|| cls.class_method(sel)) else {
            failures.push(format!("{}: no method {}", row.receiver, row.selector));
            continue;
        };
        if let Err(e) = check_method(row, m) {
            failures.push(format!(
                "{} on {} ({}): {e}",
                row.selector,
                row.receiver,
                cls.name().to_string_lossy()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(rows.len(), 255);
}

/// The render-encoder SPI the host's AGX render encoder does not implement stays unimplemented
#[test]
fn unimplemented_render_spi_stays_unimplemented() {
    let live = runtime_classes();
    let cls = class_named("MTLRenderCommandEncoder", &live);
    for selector in UNIMPLEMENTED_ON_HOST {
        let sel = Sel::register(&CString::new(*selector).unwrap());
        assert!(
            cls.instance_method(sel).is_none(),
            "{selector}: now implemented"
        );
    }
}

/// The check is live: a row with a wrong width or a wrong struct tag fails against the runtime
#[test]
fn a_wrong_row_fails_the_check() {
    let dev = device_class();
    let m = dev.instance_method(sel!(gpuCoreCount)).unwrap();
    let row = |encoding| Row {
        receiver: "MTLDevice",
        selector: "gpuCoreCount",
        encoding,
    };
    assert!(check_method(&row("I@:"), m).is_ok());
    assert!(check_method(&row("Q@:"), m).is_err());
    let arr = AnyClass::get(c"MTLResourceAddressRangeArray").unwrap();
    let m = arr.instance_method(sel!(initWithRanges:count:)).unwrap();
    let row = |encoding| Row {
        receiver: "MTLResourceAddressRangeArray",
        selector: "initWithRanges:count:",
        encoding,
    };
    assert!(check_method(&row("@@:^{MTLAddressRange=QQ}Q"), m).is_ok());
    assert!(check_method(&row("@@:^{MTLResourceAddressRange=QQ}Q"), m).is_err());
    assert!(check_method(&row("@@:^{MTLAddressRange=QQ}"), m).is_err());
}

/// The check is live: a wrong width or a wrong struct tag is caught. objc2-encode compares the tag
/// of a struct behind one pointer, so `MetalSPI.h`'s `MTLResourceAddressRange` fails where the
/// runtime says `MTLAddressRange`
#[test]
fn wrong_declarations_are_caught() {
    let dev = device_class();
    // gpuCoreCount is `unsigned int` at run time, not NSUInteger.
    assert!(dev.verify_sel::<(), u64>(sel!(gpuCoreCount)).is_err());
    assert!(dev.verify_sel::<(), u32>(sel!(gpuCoreCount)).is_ok());
    // supportsFamily: takes NSInteger, and returns BOOL.
    assert!(
        dev.verify_sel::<(i64,), Bool>(sel!(supportsFamily:))
            .is_ok()
    );
    assert!(
        dev.verify_sel::<(i64,), u64>(sel!(supportsFamily:))
            .is_err()
    );
    // The range's runtime tag is MTLAddressRange; the SPI's name is refused.
    #[repr(C)]
    struct HeaderTag {
        _a: u64,
        _b: u64,
    }
    // SAFETY: the same layout under the header's tag, for the negative check only.
    unsafe impl objc2::encode::RefEncode for HeaderTag {
        const ENCODING_REF: objc2::encode::Encoding =
            objc2::encode::Encoding::Pointer(&objc2::encode::Encoding::Struct(
                "MTLResourceAddressRange",
                &[
                    objc2::encode::Encoding::ULongLong,
                    objc2::encode::Encoding::ULongLong,
                ],
            ));
    }
    let arr = AnyClass::get(c"MTLResourceAddressRangeArray").unwrap();
    assert!(
        arr.verify_sel::<(*const HeaderTag, u64), *mut AnyObject>(sel!(initWithRanges:count:))
            .is_err()
    );
    assert!(
        arr.verify_sel::<(*const Enc<ResourceAddressRange>, u64), *mut AnyObject>(
            sel!(initWithRanges:count:)
        )
        .is_ok()
    );
    // A selector the class does not implement.
    assert!(dev.verify_sel::<(), u64>(sel!(pgNoSuchSelector)).is_err());
}
