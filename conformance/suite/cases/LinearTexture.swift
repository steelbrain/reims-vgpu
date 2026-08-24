import Metal
import Foundation
import IOSurface

let formats: [Fmt] = [
    Fmt(name: "r8Unorm", mtl: .r8Unorm, bpp: 1) { b in pack(b[0], 0, 0, 255) },
    Fmt(name: "a8Unorm", mtl: .a8Unorm, bpp: 1) { b in pack(0, 0, 0, b[0]) },
    Fmt(name: "rg8Unorm", mtl: .rg8Unorm, bpp: 2) { b in pack(b[0], b[1], 0, 255) },
    Fmt(name: "rgba8Unorm", mtl: .rgba8Unorm, bpp: 4) { b in pack(b[0], b[1], b[2], b[3]) },
    Fmt(name: "bgra8Unorm", mtl: .bgra8Unorm, bpp: 4) { b in pack(b[2], b[1], b[0], b[3]) },
]

func linearAliasCase(_ f: Fmt, _ w: Int, _ h: Int, pitch: Pitch, sampler: Bool) {
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = pitch.bytes(width: w, bpp: f.bpp, align: align)
    let label = "linear_\(f.name)_\(w)x\(h)_pitch_\(pitch.tag)\(sampler ? "_sampled" : "")"
    if bpr % align != 0 || bpr < w * f.bpp {
        skip(label, "pitch \(bpr) is not a multiple of this device's minimumLinearTextureAlignment=\(align)")
        return
    }
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(label, false, "buffer allocation failed"); return
    }
    // Fill every byte, padding included, with a position-dependent pattern.
    // Padding gets a distinct value so a rail that folds it into the image
    // shows up as a wrong texel rather than as a plausible one.
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    for y in 0..<h {
        for x in 0..<bpr {
            base[y * bpr + x] = UInt8((x &* 7 &+ y &* 13 &+ 11) & 0xFF)
        }
    }
    let d = MTLTextureDescriptor()
    d.textureType = .type2D
    d.pixelFormat = f.mtl
    d.width = w; d.height = h
    d.mipmapLevelCount = 1
    d.storageMode = .shared
    d.usage = sampler ? [.shaderRead] : [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        report(label, false, "makeTexture returned nil"); return
    }
    guard let got = readBack(sampler ? samplePipe : readPipe, tex, w, h) else {
        refused(label); return
    }
    var bad: [(Int, Int)] = []
    var firstDetail = ""
    for y in 0..<h {
        for x in 0..<w {
            var bytes = [UInt8](repeating: 0, count: f.bpp)
            for i in 0..<f.bpp { bytes[i] = base[y * bpr + x * f.bpp + i] }
            let want = f.expect(bytes)
            let have = got[y * w + x]
            if want != have {
                bad.append((x, y))
                if firstDetail.isEmpty {
                    firstDetail = "first_bad=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "\(w * h) texels exact"
                       : "\(bad.count)/\(w * h) wrong \(firstDetail) \(badMap(bad, w, h))")
}

// ---------------------------------------------------------------------------
// B. Incremental CPU writes around GPU reads.
//
// The glyph-atlas pattern: draw with the texture, write more into the unused
// part of the same allocation, draw again, through one texture that is never
// recreated and never re-declared. The contract says the later writes are
// visible; a rail that treats the first read as a snapshot fails here and
// nowhere else.
// ---------------------------------------------------------------------------

/// The glyph-atlas lifecycle, per format and per stage.
///
/// Region A is written before the texture is ever read; region B only after the
/// GPU has already read it once; then A is rewritten. One texture, never
/// recreated, never re-declared. A rail that treats the first read as a
/// snapshot fails here and nowhere else in this file.
///
/// **Run through both stages.** The type layer reaches the rasterizer as a
/// *fragment* texture, and section F exists because a device may route a
/// sampled guest image differently for the two stages. A compute-only version
/// of this case cannot see a defect that lives on the draw path — and on this
/// device `a8Unorm`, the format the type layer uses, is refused on the compute
/// sampled rail outright, so the compute arm of the one format that matters
/// most produces no reading at all. The fragment arm is the one that can.
func incrementalCase(_ f: Fmt, viaFragment: Bool) {
    let w = 64, h = 32, half = 16
    let stage = viaFragment ? "fragment" : "compute"
    let prefix = "incremental_\(f.name)_\(stage)"
    let names = (first: "\(prefix)_first_read",
                 append: "\(prefix)_append_visible",
                 stable: "\(prefix)_untouched_stable",
                 rewrite: "\(prefix)_rewrite_visible")
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = alignUp(w * f.bpp, align)
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(names.first, false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    memset(buf.contents(), 0, bpr * h)

    // Byte values chosen so no two phases share one, and none is zero: zero is
    // what an untouched region must read as, so a phase marker that collided
    // with it could not tell "not yet written" from "written and lost".
    let markA: UInt8 = 0x41, markB: UInt8 = 0x5A, markRewrite: UInt8 = 0x77
    func fill(_ rows: Range<Int>, _ value: UInt8) {
        for y in rows { for x in 0..<bpr { base[y * bpr + x] = value } }
    }
    func texel(_ value: UInt8) -> UInt32 { f.expect([UInt8](repeating: value, count: f.bpp)) }
    func read(_ tex: MTLTexture) -> [UInt32]? {
        viaFragment ? fragmentSample(tex, w, h) : readBack(readPipe, tex, w, h)
    }
    func rows(_ got: [UInt32], _ range: Range<Int>, _ want: UInt32) -> Bool {
        range.allSatisfy { y in (0..<w).allSatisfy { x in got[y * w + x] == want } }
    }

    fill(0..<half, markA)
    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        report(names.first, false, "makeTexture nil"); return
    }

    guard let first = read(tex) else {
        refused(names.first)
        for dependent in [names.append, names.stable, names.rewrite] {
            skipDependent(dependent, names.first)
        }
        return
    }
    let aOK = rows(first, 0..<half, texel(markA))
    let bZero = rows(first, half..<h, texel(0))
    report(names.first, aOK && bZero,
           aOK && bZero ? "region A present, region B still zero"
                        : "A_ok=\(aOK) B_zero=\(bZero) sample=\(hex(first[0])) "
                          + "want_A=\(hex(texel(markA))) want_B=\(hex(texel(0)))")

    // Region B: written only after the GPU has already read this texture once.
    fill(half..<h, markB)
    guard let second = read(tex) else {
        refused(names.append)
        skipDependent(names.stable, names.append)
        skipDependent(names.rewrite, names.append)
        return
    }
    let bNow = rows(second, half..<h, texel(markB))
    let aStill = rows(second, 0..<half, texel(markA))
    report(names.append, bNow,
           bNow ? "post-read append visible"
                : "append not visible, got=\(hex(second[half * w])) want=\(hex(texel(markB)))")
    report(names.stable, aStill,
           aStill ? "region A unchanged"
                  : "region A moved, got=\(hex(second[0])) want=\(hex(texel(markA)))")

    // And a rewrite of a region already read twice.
    fill(0..<half, markRewrite)
    guard let third = read(tex) else { refused(names.rewrite); return }
    let rewrite = rows(third, 0..<half, texel(markRewrite))
    report(names.rewrite, rewrite,
           rewrite ? "rewrite visible"
                   : "rewrite not visible, got=\(hex(third[0])) want=\(hex(texel(markRewrite)))")
}
