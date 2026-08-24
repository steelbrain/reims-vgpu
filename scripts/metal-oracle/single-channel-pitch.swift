// What does a *single-channel* linear texture promise about stride and channels?
//
// The three alias probes beside this one all use `rgba8Unorm`, which is the
// easy case: four bytes a texel, a natural pitch that is already aligned, and a
// component mapping with nothing to decide. A glyph atlas is none of those. It
// is one byte a texel, so its natural pitch is the width itself and almost never
// lands on an alignment boundary, and its single byte has to arrive in a
// specific channel for the shader that samples it to find anything there.
//
// Both halves are contract terms this device has to reproduce and neither is
// visible in the wire format, which carries a pitch and a format enum and says
// nothing about what the API does with them. So this probe asks the two
// questions directly:
//
//   1. **Is a declared `bytesPerRow` honoured when it exceeds the width?** A
//      consumer that assumed `width * bytesPerTexel` would read row `y` from
//      somewhere inside row `y-1` and drift further with every row. Over an
//      atlas that is glyphs sheared into each other -- shapes in roughly the
//      right places that resolve to nothing.
//
//   2. **Which channel does the byte arrive in, per format?** `r8Unorm` and
//      `a8Unorm` are the same byte in memory and are not the same texture. If
//      they differ here, a device that maps both onto one destination format
//      without reproducing the mapping renders one of them blank.
//
// Both are asked with a padded pitch and a deliberately awkward width, because
// a width that happens to be a multiple of the alignment cannot tell a correct
// implementation from one that ignores the pitch.

import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!

// 100 is chosen to be neither a power of two nor a multiple of any plausible
// alignment, so a tight-packing consumer disagrees on the very first row.
let W = 100, H = 64

let src = """
#include <metal_stdlib>
using namespace metal;
// Read the first texel of each row and report all four components, so the
// channel question and the stride question are answered by one dispatch.
kernel void probe(texture2d<float, access::read> t [[texture(0)]],
                  device float4 *out [[buffer(0)]],
                  uint y [[thread_position_in_grid]]) {
    if (y >= t.get_height()) return;
    out[y] = t.read(uint2(0, y));
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let pipe = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "probe")!)

func byte(_ f: Float) -> Int { Int(f * 255.0 + 0.5) }

// One row-distinct value per row, so a stride error shows up as the wrong row
// rather than as a plausible-looking constant.
func rowValue(_ y: Int) -> UInt8 { UInt8(1 + y * 3) }

func probe(_ fmt: MTLPixelFormat, _ name: String) {
    let align = dev.minimumLinearTextureAlignment(for: fmt)
    var bpr = W                                   // one byte a texel
    if align != 0 && bpr % align != 0 { bpr += align - (bpr % align) }
    print("RESULT \(name)_align=\(align) natural_bpr=\(W) declared_bpr=\(bpr) padded=\(bpr > W)")

    let buf = dev.makeBuffer(length: bpr * H, options: .storageModeShared)!
    let p = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * H)
    // Fill the whole allocation with a value no row uses, so a read that lands
    // in padding is distinguishable from a read that lands in another row.
    for i in 0..<(bpr * H) { p[i] = 0xFE }
    for y in 0..<H {
        for x in 0..<W { p[y * bpr + x] = rowValue(y) }
    }

    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: fmt, width: W, height: H, mipmapped: false)
    d.storageMode = .shared
    d.usage = [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        print("RESULT \(name)_linear_texture=UNSUPPORTED")
        return
    }
    print("RESULT \(name)_linear_texture=OK")

    let out = dev.makeBuffer(length: 16 * H, options: .storageModeShared)!
    let cb = queue.makeCommandBuffer()!
    let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(pipe)
    e.setTexture(tex, index: 0)
    e.setBuffer(out, offset: 0, index: 0)
    e.dispatchThreads(MTLSize(width: H, height: 1, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let o = out.contents().bindMemory(to: SIMD4<Float>.self, capacity: H)

    // Which channel carries the byte? Answered on row 0, whose value is 1 and
    // so cannot be confused with the 1.0 an implicit alpha would produce.
    let c0 = o[0]
    let want = Int(rowValue(0))
    let channels = ["r", "g", "b", "a"]
    let carrying = (0..<4).filter { byte(c0[$0]) == want }.map { channels[$0] }
    print("RESULT \(name)_row0_rgba=\(byte(c0.x)),\(byte(c0.y)),\(byte(c0.z)),\(byte(c0.w))"
        + " expect_byte=\(want) carried_in=\(carrying.isEmpty ? "NONE" : carrying.joined(separator: "+"))")

    // Is the declared pitch honoured? Every row must report its own value in
    // whichever channel row 0 used. A tight-packing reader drifts by
    // (bpr - W) bytes a row and disagrees almost immediately.
    guard let ch = (0..<4).first(where: { byte(c0[$0]) == want }) else {
        print("RESULT \(name)_pitch_honoured=INDETERMINATE reason=no_channel_carried_row0")
        return
    }
    var firstBad = -1
    for y in 0..<H where byte(o[y][ch]) != Int(rowValue(y)) { firstBad = y; break }
    if firstBad < 0 {
        print("RESULT \(name)_pitch_honoured=true rows=\(H)")
    } else {
        print("RESULT \(name)_pitch_honoured=false first_bad_row=\(firstBad)"
            + " got=\(byte(o[firstBad][ch])) expect=\(Int(rowValue(firstBad)))")
    }
}

probe(.r8Unorm, "r8unorm")
probe(.a8Unorm, "a8unorm")

// A third question -- what a `bytesPerRow` the alignment does not admit does --
// is deliberately not asked by running it, because asking ends the process. A
// misaligned pitch is not a soft decline returning nil: Metal's validation
// raises a hard assertion inside `makeTexture` naming the required alignment,
// so a probe that tries it reports nothing afterwards.
//
// That is itself the answer, in its strong form. An unaligned pitch is not
// merely unsupported; it is a programming error the API declines to model. The
// alignment printed above is therefore a floor a linear texture declaration
// must already meet, not a preference an implementation may round past.
