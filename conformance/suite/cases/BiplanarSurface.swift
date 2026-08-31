import Metal
import Foundation
import IOSurface

// ---------------------------------------------------------------------------
// A two-plane IOSurface, read one plane at a time.
//
// One guest allocation carrying two textures at declared offsets, each with its
// own extent, row pitch and pixel format. That is the shape the guest's own
// compositor registers, and it is the shape that separates "a surface is an
// image" from "a surface is an allocation and a plane is an image": a device
// that gives both planes one backing has one layout to choose from and one
// image to build, so it either renders the wrong plane's geometry or refuses
// the second plane outright.
//
// Both planes are seeded from the CPU with values that are wrong for the other
// plane, so a device that resolves plane 1 to plane 0's image fails on content
// rather than on a refusal, and a device that refuses the second plane fails on
// the readback. Neither can pass by accident.
func biplanarSurfaceCase(_ w: Int, _ h: Int) {
    let dims = "\(w)x\(h)"
    let names = (luma: "biplanar_luma_\(dims)", chroma: "biplanar_chroma_\(dims)")

    // '420f' — full-range biplanar 8-bit YCbCr. Plane 0 is one byte a texel at
    // the full extent; plane 1 is two bytes a texel at half extent in both
    // axes, which is why the two planes cannot share a layout.
    let biplanar: UInt32 = 0x3432_3066
    let cw = w / 2
    let ch = h / 2
    // The alignment is the framework's, asked for by name. A literal here
    // would be a number nobody derived and wrong on the next host.
    let lumaBytesPerRow = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, w)
    let chromaBytesPerRow = IOSurfaceAlignProperty(kIOSurfaceBytesPerRow, cw * 2)
    let lumaSize = IOSurfaceAlignProperty(kIOSurfaceAllocSize, lumaBytesPerRow * h)
    let chromaSize = IOSurfaceAlignProperty(kIOSurfaceAllocSize, chromaBytesPerRow * ch)
    let props: [IOSurfacePropertyKey: Any] = [
        .width: w,
        .height: h,
        .pixelFormat: biplanar,
        .planeInfo: [
            [
                IOSurfacePropertyKey.planeWidth: w,
                IOSurfacePropertyKey.planeHeight: h,
                IOSurfacePropertyKey.planeBytesPerRow: lumaBytesPerRow,
                IOSurfacePropertyKey.planeOffset: 0,
                IOSurfacePropertyKey.planeSize: lumaSize,
                IOSurfacePropertyKey.planeBytesPerElement: 1,
            ],
            [
                IOSurfacePropertyKey.planeWidth: cw,
                IOSurfacePropertyKey.planeHeight: ch,
                IOSurfacePropertyKey.planeBytesPerRow: chromaBytesPerRow,
                IOSurfacePropertyKey.planeOffset: lumaSize,
                IOSurfacePropertyKey.planeSize: chromaSize,
                IOSurfacePropertyKey.planeBytesPerElement: 2,
            ],
        ],
    ]
    guard let surface = IOSurface(properties: props), surface.planeCount == 2 else {
        report(names.luma, false, "no two-plane IOSurface for \(dims)")
        skipDependent(names.chroma, names.luma)
        return
    }

    func planeTexture(_ plane: Int, _ format: MTLPixelFormat,
                      _ pw: Int, _ ph: Int, _ label: String) -> MTLTexture? {
        let td = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format, width: pw, height: ph, mipmapped: false)
        td.usage = [.shaderRead]
        td.storageMode = .shared
        guard let tex = dev.makeTexture(descriptor: td, iosurface: surface, plane: plane) else {
            report(label, false, "makeTexture(iosurface:plane:\(plane)) nil for \(dims)")
            return nil
        }
        return tex
    }

    // The seed is a function of the plane and the coordinate, so plane 1's
    // bytes are never a value plane 0 holds at the same coordinate.
    func luma(_ x: Int, _ y: Int) -> UInt8 { UInt8((1 + x * 7 + y * 13) % 251) }
    func chromaR(_ x: Int, _ y: Int) -> UInt8 { UInt8((160 + x * 3 + y * 5) % 251) }
    func chromaG(_ x: Int, _ y: Int) -> UInt8 { UInt8((200 + x * 11 + y * 2) % 251) }

    // The per-plane accessors are the C entry points: Swift's `IOSurface`
    // exposes only the whole-surface `baseAddress` and `bytesPerRow` as
    // properties, and on a biplanar surface those describe plane 0 alone.
    let ref = surface as IOSurfaceRef
    surface.lock(options: [], seed: nil)
    let lumaBase = IOSurfaceGetBaseAddressOfPlane(ref, 0).assumingMemoryBound(to: UInt8.self)
    let lumaPitch = IOSurfaceGetBytesPerRowOfPlane(ref, 0)
    for y in 0..<h {
        for x in 0..<w { lumaBase[y * lumaPitch + x] = luma(x, y) }
    }
    let chromaBase = IOSurfaceGetBaseAddressOfPlane(ref, 1).assumingMemoryBound(to: UInt8.self)
    let chromaPitch = IOSurfaceGetBytesPerRowOfPlane(ref, 1)
    for y in 0..<ch {
        for x in 0..<cw {
            chromaBase[y * chromaPitch + x * 2] = chromaR(x, y)
            chromaBase[y * chromaPitch + x * 2 + 1] = chromaG(x, y)
        }
    }
    surface.unlock(options: [], seed: nil)

    if let tex = planeTexture(0, .r8Unorm, w, h, names.luma) {
        if let got = readBack(readPipe, tex, w, h) {
            var bad: [(Int, Int)] = []
            for y in 0..<h {
                for x in 0..<w where got[y * w + x] != pack(luma(x, y), 0, 0, 255) {
                    bad.append((x, y))
                }
            }
            report(names.luma, bad.isEmpty,
                   bad.isEmpty
                       ? "plane 0 reads its own \(w)x\(h) one-byte texels"
                       : "wrong=\(bad.count)/\(w * h) "
                         + "first=\(hex(got[bad[0].1 * w + bad[0].0])) "
                         + "want=\(hex(pack(luma(bad[0].0, bad[0].1), 0, 0, 255)))"
                         + badMap(bad, w, h))
        } else {
            refused(names.luma)
        }
    }

    if let tex = planeTexture(1, .rg8Unorm, cw, ch, names.chroma) {
        if let got = readBack(readPipe, tex, cw, ch) {
            var bad: [(Int, Int)] = []
            for y in 0..<ch {
                for x in 0..<cw
                where got[y * cw + x] != pack(chromaR(x, y), chromaG(x, y), 0, 255) {
                    bad.append((x, y))
                }
            }
            report(names.chroma, bad.isEmpty,
                   bad.isEmpty
                       ? "plane 1 reads its own \(cw)x\(ch) two-byte texels"
                       : "wrong=\(bad.count)/\(cw * ch) "
                         + "first=\(hex(got[bad[0].1 * cw + bad[0].0])) "
                         + "want=\(hex(pack(chromaR(bad[0].0, bad[0].1), chromaG(bad[0].0, bad[0].1), 0, 255)))"
                         + badMap(bad, cw, ch))
        } else {
            refused(names.chroma)
        }
    }
}
