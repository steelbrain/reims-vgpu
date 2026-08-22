// Does a CPU store made BEFORE the aliasing texture exists survive that
// texture's creation?
//
// `linear-alias.swift` asks the other half of this: it creates the texture
// first and then writes, so it measures whether later stores are seen. It
// cannot see the case that decides whether this device owes a materialization
// step, which is a guest that rasterizes into its own memory and only then
// declares a texture over it.
//
// The answer matters because Vulkan gives us no choice on our side. An image
// bound to imported host memory must be created with `initialLayout =
// UNDEFINED` (VUID-vkBindImageMemory-memory-02989 composing with
// VUID-VkImageCreateInfo-pNext-01443), and a transition out of `UNDEFINED` is
// permitted to discard whatever was in the memory. So if Metal preserves
// pre-creation bytes, an aliasing rail here must put them back explicitly, and
// a rail that skips that step is not reproducing the API -- it is silently
// dropping content the guest is entitled to.
//
// A `false` from this probe would be the surprising result and would mean the
// materialization step is unnecessary. Treat it as a finding, not a pass.

import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!

let W = 256, H = 64
let fmt = MTLPixelFormat.rgba8Unorm
let align = dev.minimumLinearTextureAlignment(for: fmt)
var bpr = W * 4
if bpr % align != 0 { bpr += align - (bpr % align) }

let buf = dev.makeBuffer(length: bpr * H, options: .storageModeShared)!

// The whole point: fill through the CPU pointer while no texture exists over
// this buffer at all.
let fillValue: UInt8 = 7
do {
    let p = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * H)
    for y in 0..<H {
        for x in 0..<W { p[y * bpr + x * 4] = fillValue }
    }
}
print("RESULT wrote_before_texture_exists=true value=\(fillValue)")

// Only now does the alias come into being.
let d = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: fmt, width: W, height: H, mipmapped: false)
d.storageMode = .shared
d.usage = [.shaderRead]
guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
    print("RESULT linear_texture_from_buffer=UNSUPPORTED")
    exit(1)
}
print("RESULT linear_texture_from_buffer=OK align=\(align) bpr=\(bpr)")

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void sum(texture2d<float, access::read> t [[texture(0)]],
                device atomic_uint *out [[buffer(0)]],
                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= t.get_width() || gid.y >= t.get_height()) return;
    float4 c = t.read(gid);
    atomic_fetch_add_explicit(out, uint(c.r * 255.0 + 0.5), memory_order_relaxed);
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let pipe = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "sum")!)
let out = dev.makeBuffer(length: 4, options: .storageModeShared)!

func gpuSum() -> UInt32 {
    out.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pipe)
    e.setTexture(tex, index: 0)
    e.setBuffer(out, offset: 0, index: 0)
    e.dispatchThreads(
        MTLSize(width: W, height: H, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return out.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
}

let expect = UInt32(W * H) * UInt32(fillValue)
let got = gpuSum()
print("RESULT prewritten_survives_creation visible=\(got == expect) got=\(got) expect=\(expect)")

// A second alias over the same buffer, created later still, must see the same
// bytes -- this is the guest re-declaring a view over an atlas it filled once.
guard let tex2 = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
    print("RESULT second_alias=UNSUPPORTED")
    exit(1)
}
let out2 = dev.makeBuffer(length: 4, options: .storageModeShared)!
out2.contents().bindMemory(to: UInt32.self, capacity: 1).pointee = 0
let cb = queue.makeCommandBuffer()!
let e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(pipe)
e.setTexture(tex2, index: 0)
e.setBuffer(out2, offset: 0, index: 0)
e.dispatchThreads(
    MTLSize(width: W, height: H, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
e.endEncoding()
cb.commit()
cb.waitUntilCompleted()
let got2 = out2.contents().bindMemory(to: UInt32.self, capacity: 1).pointee
print("RESULT second_alias_sees_same_bytes visible=\(got2 == expect) got=\(got2) expect=\(expect)")
