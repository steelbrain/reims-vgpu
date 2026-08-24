// Does a CPU store into an MTLBuffer become visible to a later GPU read through
// a linear texture aliasing that buffer, with NO API call announcing the write?
//
// This is the contract question behind every "sampled resource" decision this
// device makes. Our guest declares a storage mode and then rasterizes into its
// own memory with the CPU; whether we owe it a re-upload, and when, depends
// entirely on what Metal promises about that store. Guessing produces the
// gather-witness class of bug -- a vouch given for bytes nobody looked at.
//
// Run through `metal-oracle.sh`, which needs a real Metal device. The answer is
// a property of the API, not of this program, so a `false` here would be the
// interesting result and should be treated as one.

import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!

let W = 256, H = 64
let fmt = MTLPixelFormat.rgba8Unorm
// The row pitch is the device's, not ours. `minimumLinearTextureAlignment` is
// the same term the device's own explicit-linear plane layout has to satisfy.
let align = dev.minimumLinearTextureAlignment(for: fmt)
var bpr = W * 4
if bpr % align != 0 { bpr += align - (bpr % align) }

let buf = dev.makeBuffer(length: bpr * H, options: .storageModeShared)!
let d = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: fmt, width: W, height: H, mipmapped: false)
d.storageMode = .shared
d.usage = [.shaderRead]
guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
    print("RESULT linear_texture_from_buffer=UNSUPPORTED")
    exit(1)
}
print("RESULT linear_texture_from_buffer=OK align=\(align) bpr=\(bpr)")
print("RESULT buffer_storage=\(buf.storageMode.rawValue) tex_storage=\(tex.storageMode.rawValue)")

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

// Write purely through the CPU pointer. No blit, no `didModifyRange`, no
// `synchronize` -- their absence is the whole measurement.
func cpuFill(_ v: UInt8) {
    let p = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * H)
    for y in 0..<H {
        for x in 0..<W { p[y * bpr + x * 4] = v }
    }
}

let expect = UInt32(W * H)
cpuFill(1)
let a = gpuSum()
print("RESULT first_write visible=\(a == expect) got=\(a) expect=\(expect)")

// The case a copy-and-revalidate rail cannot see: a store landing AFTER the
// resource is live and has already been sampled once.
cpuFill(2)
let b = gpuSum()
print("RESULT rewrite_after_use visible=\(b == expect * 2) got=\(b) expect=\(expect * 2)")

cpuFill(3)
let c = gpuSum()
print("RESULT second_rewrite visible=\(c == expect * 3) got=\(c) expect=\(expect * 3)")
