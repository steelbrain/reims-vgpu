// Can a guest's mip chain be a *linear* texture aliasing its own memory, and if
// so who chooses each level's row pitch and offset?
//
// This device now admits a mipmapped 2D allocation onto the sampled alias rail,
// where the alias is one Vulkan LINEAR image carrying the whole chain and the
// host driver picks that image's per-level layout. Admission then holds only
// when the driver's choice matches the level table the guest declared. On one
// driven boot the dominant refusal was exactly that mismatch, so the question
// this probe answers is which side of it is the contract:
//
//   - if Metal's linear-texture rail refuses mipmaps outright, then a guest mip
//     chain is never a linear texture, and every per-level pitch the guest
//     declares comes from the guest driver's own packing rather than from a
//     device alignment rule we could reproduce;
//   - if it accepts them, the API names a per-level pitch rule and a device
//     that cannot reproduce it must refuse by name rather than guess.
//
// Also reported: the linear alignment for the formats this device aliases most,
// and the size/align Metal itself assigns a mipmapped descriptor. Those are the
// terms an implementation is allowed to depend on; a level offset read off one
// allocation is not.
//
// Run through `metal-oracle.sh`. Prints `RESULT key=value` lines only.

import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!

// The formats a sampled alias is built for. `minimumLinearTextureAlignment` is
// the same term the device's explicit-linear plane layout has to satisfy, so a
// disagreement here would be a disagreement about the plane, not the chain.
let formats: [(String, MTLPixelFormat)] = [
    ("r8Unorm", .r8Unorm),
    ("rg8Unorm", .rg8Unorm),
    ("rgba8Unorm", .rgba8Unorm),
    ("bgra8Unorm", .bgra8Unorm),
]
for (name, fmt) in formats {
    print("RESULT linear_align_\(name)=\(dev.minimumLinearTextureAlignment(for: fmt))")
}

// Does the linear rail take a chain at all? One buffer, one descriptor that
// differs from the working single-level case only in its level count.
let W = 256, H = 256
let fmt = MTLPixelFormat.rgba8Unorm
let align = dev.minimumLinearTextureAlignment(for: fmt)
var bpr = W * 4
if bpr % align != 0 { bpr += align - (bpr % align) }
// Generous: a full chain packed tightest-case still fits twice over.
let buf = dev.makeBuffer(length: bpr * H * 2, options: .storageModeShared)!

let flat = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: fmt, width: W, height: H, mipmapped: false)
flat.storageMode = .shared
flat.usage = [.shaderRead]
print("RESULT linear_chain_1_level="
    + (buf.makeTexture(descriptor: flat, offset: 0, bytesPerRow: bpr) == nil
        ? "REFUSED" : "OK"))

let chain = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: fmt, width: W, height: H, mipmapped: true)
chain.storageMode = .shared
chain.usage = [.shaderRead]
print("RESULT declared_chain_levels=\(chain.mipmapLevelCount)")

// What Metal assigns a chain of its own. `heapTextureSizeAndAlign` is the only
// public statement about a chain's footprint; a per-level offset is not exposed
// by the API at all, which is itself the answer to "may we depend on one".
let sa = dev.heapTextureSizeAndAlign(descriptor: chain)
print("RESULT chain_heap_size=\(sa.size) chain_heap_align=\(sa.align)")
let flatSA = dev.heapTextureSizeAndAlign(descriptor: flat)
print("RESULT flat_heap_size=\(flatSA.size) flat_heap_align=\(flatSA.align)")
print("RESULT tight_level0_bytes=\(W * 4 * H)")

// A chain the device allocates itself still reports one bytesPerRow-shaped
// fact per level through `getBytes`, which is the only per-level pitch the API
// will state. Ask whether it accepts a tight pitch at every level -- that is
// the rule a device reproducing the chain would have to satisfy.
chain.storageMode = .shared
guard let owned = dev.makeTexture(descriptor: chain) else {
    print("RESULT owned_chain=REFUSED")
    exit(0)
}
print("RESULT owned_chain_levels=\(owned.mipmapLevelCount)")
var accepted = 0
for level in 0..<owned.mipmapLevelCount {
    let w = max(1, W >> level), h = max(1, H >> level)
    let tight = w * 4
    var row = [UInt8](repeating: 0, count: tight * h)
    row.withUnsafeMutableBytes { raw in
        owned.getBytes(
            raw.baseAddress!, bytesPerRow: tight,
            from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: level)
    }
    accepted += 1
}
print("RESULT owned_chain_tight_pitch_levels_accepted=\(accepted)")

// Last, because it does not return. Metal validates a linear texture's
// descriptor with an assertion rather than a nil result, so this call aborts
// the process: `Linear texture: cannot be mipmapped`. That abort IS the answer,
// and putting it here means every RESULT above is already on stdout when it
// lands. A run that reaches this line and survives would be the finding.
print("RESULT linear_chain_n_levels=ASKING")
_ = buf.makeTexture(descriptor: chain, offset: 0, bytesPerRow: bpr)
print("RESULT linear_chain_n_levels=OK")
