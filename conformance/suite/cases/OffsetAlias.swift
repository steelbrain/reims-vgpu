import Metal
import Foundation
import IOSurface

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// H. `replaceRegion:` into a sampled texture, sub-rect, across submissions.
//
// Section B writes an atlas through the `MTLBuffer` its texture aliases. That
// is one of the two ways a glyph atlas is filled and it is not the one a
// CPU-rasterized layer uses: CoreGraphics rasterizes into its own bitmap and
// the result reaches Metal through
// `-[MTLTexture replaceRegion:mipmapLevel:withBytes:bytesPerRow:]`, on a plain
// texture with no buffer behind it. That is a different record on the wire and
// a different rail in the device, and nothing in this file reached it except
// inside the mip cases, where a separate defect masks the result.
//
// The sub-rect is the point. An atlas grows by replacing the strip it just
// rasterized, leaving every glyph already in it untouched, so a rail that
// re-uploads the whole texture, or uploads the strip to the wrong origin, or
// drops the second replace because it already has the resource resident, loses
// exactly the glyphs that were there before — which is type that renders once
// and then does not.
// ---------------------------------------------------------------------------

func replaceRegionCase(_ f: Fmt, viaFragment: Bool) {
    let w = 64, h = 32, half = 16
    let stage = viaFragment ? "fragment" : "compute"
    let prefix = "replace_\(f.name)_\(stage)"
    let names = (first: "\(prefix)_first_read",
                 append: "\(prefix)_append_visible",
                 stable: "\(prefix)_untouched_stable",
                 rewrite: "\(prefix)_rewrite_visible")

    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    // Not buffer-backed, and not `.private`: `replaceRegion` is a CPU write and
    // needs storage the CPU can reach. This is what a rasterized atlas is.
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = dev.makeTexture(descriptor: d) else {
        report(names.first, false, "makeTexture nil"); return
    }

    let markA: UInt8 = 0x41, markB: UInt8 = 0x5A, markRewrite: UInt8 = 0x77
    // Every texel of the strip, tight — `replaceRegion` takes the caller's own
    // pitch and has no alignment rule of its own.
    func replace(_ y0: Int, _ rows: Int, _ value: UInt8) {
        let pitch = w * f.bpp
        let bytes = [UInt8](repeating: value, count: pitch * rows)
        bytes.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake2D(0, y0, w, rows),
                        mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: pitch)
        }
    }
    func texel(_ value: UInt8) -> UInt32 { f.expect([UInt8](repeating: value, count: f.bpp)) }
    func read() -> [UInt32]? {
        viaFragment ? fragmentSample(tex, w, h) : readBack(readPipe, tex, w, h)
    }
    func rows(_ got: [UInt32], _ range: Range<Int>, _ want: UInt32) -> Bool {
        range.allSatisfy { y in (0..<w).allSatisfy { x in got[y * w + x] == want } }
    }
    // Where a wrong sub-rect landed, which a count alone cannot say.
    func wrongRows(_ got: [UInt32], _ range: Range<Int>, _ want: UInt32) -> String {
        let bad = range.filter { y in !(0..<w).allSatisfy { x in got[y * w + x] == want } }
        guard let firstBad = bad.first else { return "" }
        return " wrong_rows=\(bad.count)/\(range.count) first=\(firstBad) " +
               "got=\(hex(got[firstBad * w])) want=\(hex(want))"
    }

    // A texture from `makeTexture` has undefined contents, so region B is
    // written once here to establish a known floor. Only the *later* writes are
    // the thing under test.
    replace(0, h, 0)
    replace(0, half, markA)
    guard let first = read() else {
        refused(names.first)
        for dependent in [names.append, names.stable, names.rewrite] {
            skipDependent(dependent, names.first)
        }
        return
    }
    let aOK = rows(first, 0..<half, texel(markA))
    let bZero = rows(first, half..<h, texel(0))
    report(names.first, aOK && bZero,
           aOK && bZero ? "the replaced strip is present and the rest is still zero"
                        : "A_ok=\(aOK) B_zero=\(bZero)"
                          + wrongRows(first, 0..<half, texel(markA))
                          + wrongRows(first, half..<h, texel(0)))

    // The atlas grows: a strip replaced after the GPU has already sampled it.
    replace(half, h - half, markB)
    guard let second = read() else {
        refused(names.append)
        skipDependent(names.stable, names.append)
        skipDependent(names.rewrite, names.append)
        return
    }
    report(names.append, rows(second, half..<h, texel(markB)),
           rows(second, half..<h, texel(markB))
             ? "a strip replaced after a read is visible"
             : "append not visible" + wrongRows(second, half..<h, texel(markB)))
    report(names.stable, rows(second, 0..<half, texel(markA)),
           rows(second, 0..<half, texel(markA))
             ? "the glyphs already in the atlas are untouched"
             : "the earlier strip moved" + wrongRows(second, 0..<half, texel(markA)))

    // And a strip rewritten in place, which is how an atlas recycles a slot.
    replace(0, half, markRewrite)
    guard let third = read() else { refused(names.rewrite); return }
    report(names.rewrite, rows(third, 0..<half, texel(markRewrite)),
           rows(third, 0..<half, texel(markRewrite))
             ? "a strip rewritten in place is visible"
             : "rewrite not visible" + wrongRows(third, 0..<half, texel(markRewrite)))
}
