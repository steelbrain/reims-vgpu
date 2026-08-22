// Does a LIVE aliasing texture see CPU writes made between two GPU reads?
//
// The other two probes here bracket this one and neither reaches it.
// `linear-alias.swift` writes once after creating the texture and reads once.
// `prewritten-alias.swift` writes before the texture exists and reads once.
// Both measure a single transition into "the GPU has looked at these bytes".
//
// The pattern this device actually has to reproduce is neither. A guest that
// rasterizes glyphs builds its atlas *incrementally*: it draws with the atlas,
// then writes more glyphs into the unused part of the same allocation, then
// draws with it again -- through one texture that is never recreated and never
// re-declared. If Metal keeps later stores visible across an intervening GPU
// read, then every rail here that treats "the GPU has read this once" as
// licence to stop re-reading is dropping content the guest is entitled to, and
// the symptom is precisely a first tranche of glyphs that renders and a second
// that does not.
//
// The probe is written so the two halves cannot be confused. Region A is
// written before the first read and never touched again; region B is zero at
// the first read and written only afterwards. Reading A's sum unchanged across
// both dispatches is the control -- it proves the second dispatch really ran
// over the same texture -- and B's sum going from zero to its expected value is
// the answer.
//
// A `false` here would be the surprising result. It would mean a live alias may
// be treated as a snapshot, and it should be recorded as a finding rather than
// waved through.

import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!

let W = 256, H = 64
let halfH = H / 2                     // rows [0, halfH) are A, [halfH, H) are B
let fmt = MTLPixelFormat.rgba8Unorm
let align = dev.minimumLinearTextureAlignment(for: fmt)
var bpr = W * 4
if bpr % align != 0 { bpr += align - (bpr % align) }

let buf = dev.makeBuffer(length: bpr * H, options: .storageModeShared)!
let p = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * H)

let valueA: UInt8 = 5
let valueB: UInt8 = 9

func fill(rows: Range<Int>, with value: UInt8) {
    for y in rows {
        for x in 0..<W { p[y * bpr + x * 4] = value }
    }
}

// A is populated up front; B stays zero so its first reading is unambiguous.
fill(rows: 0..<halfH, with: valueA)

let d = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: fmt, width: W, height: H, mipmapped: false)
d.storageMode = .shared
d.usage = [.shaderRead]
guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
    print("RESULT linear_texture_from_buffer=UNSUPPORTED")
    exit(1)
}
print("RESULT linear_texture_from_buffer=OK align=\(align) bpr=\(bpr)")

// Sum each half separately in one dispatch, so a single GPU read answers both
// the control and the question.
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void sum(texture2d<float, access::read> t [[texture(0)]],
                device atomic_uint *out [[buffer(0)]],
                constant uint &split [[buffer(1)]],
                uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= t.get_width() || gid.y >= t.get_height()) return;
    float4 c = t.read(gid);
    uint v = uint(c.r * 255.0 + 0.5);
    atomic_fetch_add_explicit(&out[gid.y < split ? 0 : 1], v, memory_order_relaxed);
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let pipe = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "sum")!)
let out = dev.makeBuffer(length: 8, options: .storageModeShared)!
var split = UInt32(halfH)

// One texture, created once, used by every dispatch. Nothing below recreates
// it, re-declares it, or tells Metal that the buffer changed.
func gpuSums() -> (UInt32, UInt32) {
    let o = out.contents().bindMemory(to: UInt32.self, capacity: 2)
    o[0] = 0
    o[1] = 0
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pipe)
    e.setTexture(tex, index: 0)
    e.setBuffer(out, offset: 0, index: 0)
    e.setBytes(&split, length: 4, index: 1)
    e.dispatchThreads(
        MTLSize(width: W, height: H, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return (o[0], o[1])
}

let expectA = UInt32(W * halfH) * UInt32(valueA)
let expectB = UInt32(W * (H - halfH)) * UInt32(valueB)

let (a1, b1) = gpuSums()
print("RESULT first_read a=\(a1) expect_a=\(expectA) b=\(b1) expect_b=0")
print("RESULT first_read_correct=\(a1 == expectA && b1 == 0)")

// The whole point: the CPU appends to the same allocation, after the GPU has
// already read it, with no API call announcing the change.
fill(rows: halfH..<H, with: valueB)

let (a2, b2) = gpuSums()
print("RESULT incremental_write_visible=\(b2 == expectB) got=\(b2) expect=\(expectB)")
print("RESULT untouched_region_stable=\(a2 == expectA) got=\(a2) expect=\(expectA)")

// A third round, so "it works once" is not mistaken for "it works". The guest
// grows an atlas many times over a session, not twice.
let valueC: UInt8 = 3
fill(rows: 0..<halfH, with: valueC)
let expectC = UInt32(W * halfH) * UInt32(valueC)
let (a3, b3) = gpuSums()
print("RESULT rewrite_visible=\(a3 == expectC) got=\(a3) expect=\(expectC)")
print("RESULT prior_append_still_visible=\(b3 == expectB) got=\(b3) expect=\(expectB)")
