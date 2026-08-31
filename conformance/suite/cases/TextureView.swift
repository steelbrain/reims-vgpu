import Metal
import Foundation
import IOSurface

// A texture view bound as a shader resource.
//
// `makeTextureView` produces a texture object that owns no storage of its own:
// it names a base texture's storage through a possibly different format, type,
// level range, slice range or channel swizzle. Nothing else in this battery
// binds one, and a view is not an incidental shape --- it is how a guest reads
// one slice of an array, one level of a chain, or one channel order of an
// allocation it does not own.
//
// The rail it exercises is the one a device gets wrong by building the view a
// second image instead of naming the base's. That failure is not subtle when it
// happens: the bind refuses, and every later command on the same channel refuses
// behind it. But a device can also *silently* substitute --- read the base at
// the wrong slice, or ignore the swizzle --- and only a predicted value catches
// that. Each case here fills its base from the CPU with a pattern that differs
// per texel and per slice, so a view that lands on the wrong storage, the wrong
// slice, or the wrong channel order reports which.

/// The colour this battery expects at one texel of one slice.
///
/// Distinct in x, in y and in slice, so a view that reads the right texels of
/// the wrong slice is not mistaken for a pass.
private func viewTexel(_ x: Int, _ y: Int, _ slice: Int) -> (UInt8, UInt8, UInt8, UInt8) {
    (UInt8((x &* 7 &+ 11) & 0xFF),
     UInt8((y &* 13 &+ 29) & 0xFF),
     UInt8((slice &* 61 &+ 5) & 0xFF),
     0xFF)
}

/// Fill one slice of a `.bgra8Unorm` texture from the CPU.
private func fillViewSlice(_ tex: MTLTexture, _ w: Int, _ h: Int, slice: Int) {
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    for y in 0..<h {
        for x in 0..<w {
            let (r, g, b, a) = viewTexel(x, y, slice)
            let o = (y * w + x) * 4
            // `.bgra8Unorm` orders the bytes blue, green, red, alpha.
            bytes[o] = b; bytes[o + 1] = g; bytes[o + 2] = r; bytes[o + 3] = a
        }
    }
    bytes.withUnsafeBytes { raw in
        tex.replace(region: MTLRegionMake2D(0, 0, w, h),
                    mipmapLevel: 0, slice: slice,
                    withBytes: raw.baseAddress!,
                    bytesPerRow: w * 4, bytesPerImage: w * h * 4)
    }
}

/// Read one texture through `read_texels` and score it against `viewTexel`.
private func checkViewReads(_ label: String, _ tex: MTLTexture,
                            _ w: Int, _ h: Int, slice: Int) {
    guard let got = readBack(readPipe, tex, w, h) else {
        report(label, false, "compute readback never ran"); return
    }
    var bad: [(Int, Int)] = []
    var firstDetail = ""
    for y in 0..<h {
        for x in 0..<w {
            let (r, g, b, a) = viewTexel(x, y, slice)
            let want = pack(r, g, b, a)
            if got[y * w + x] != want {
                if bad.isEmpty {
                    firstDetail = " first=(\(x),\(y)) want=\(hex(want)) got=\(hex(got[y * w + x]))"
                }
                bad.append((x, y))
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "\(w)x\(h) slice=\(slice) matched"
                       : "\(badMap(bad, w, h))\(firstDetail)")
}

/// A view over a plain private-storage 2D texture, identical in every
/// parameter to its base.
///
/// The narrowest possible view, and therefore the one whose failure can only be
/// the view mechanism: same format, same type, one level, one slice. A device
/// that gives the view an image of its own fails here and nowhere earlier.
func textureViewIdentityCase(_ w: Int, _ h: Int) {
    let label = "texture_view_identity_\(w)x\(h)"
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    d.usage = [.shaderRead]
    d.storageMode = .shared
    guard let base = dev.makeTexture(descriptor: d) else {
        report(label, false, "makeTexture nil for \(w)x\(h)"); return
    }
    fillViewSlice(base, w, h, slice: 0)
    guard let view = base.makeTextureView(pixelFormat: .bgra8Unorm) else {
        report(label, false, "makeTextureView nil"); return
    }
    // The base first: if the base itself does not read back, the view's result
    // says nothing about views.
    checkViewReads("\(label)_base", base, w, h, slice: 0)
    checkViewReads("\(label)_view", view, w, h, slice: 0)
}

/// A view of one slice of a 2D array, bound as a 2D texture.
///
/// Every slice carries a different pattern, so a view that reads the base's
/// slice zero whatever it was asked for fails with a full-surface mismatch
/// rather than passing on a texture whose slices happen to agree.
func textureViewArraySliceCase(_ w: Int, _ h: Int, slices: Int) {
    let label = "texture_view_array_slice_\(w)x\(h)x\(slices)"
    let d = MTLTextureDescriptor()
    d.textureType = .type2DArray
    d.pixelFormat = .bgra8Unorm
    d.width = w; d.height = h; d.arrayLength = slices; d.mipmapLevelCount = 1
    d.usage = [.shaderRead]
    d.storageMode = .shared
    guard let base = dev.makeTexture(descriptor: d) else {
        report(label, false, "makeTexture nil for \(w)x\(h)x\(slices)"); return
    }
    for slice in 0..<slices { fillViewSlice(base, w, h, slice: slice) }
    for slice in 0..<slices {
        guard let view = base.makeTextureView(
            pixelFormat: .bgra8Unorm, textureType: .type2D,
            levels: 0..<1, slices: slice..<(slice + 1)) else {
            report("\(label)_s\(slice)", false, "makeTextureView nil"); return
        }
        checkViewReads("\(label)_s\(slice)", view, w, h, slice: slice)
    }
}

/// A view over an IOSurface-backed texture.
///
/// The base owns a surface plane rather than an ordinary allocation, and the
/// image belongs to the plane. A device that decides which resource owns an
/// image by walking the view chain rather than by asking who owns the storage
/// lands the view on the surface, which owns no image at all, and refuses.
func textureViewOverSurfaceCase(_ w: Int, _ h: Int) {
    let label = "texture_view_over_surface_\(w)x\(h)"
    let bgra: UInt32 = 0x4247_5241   // 'BGRA' as an OSType
    let props: [IOSurfacePropertyKey: Any] = [
        .width: w, .height: h, .bytesPerElement: 4, .pixelFormat: bgra,
    ]
    guard let surface = IOSurface(properties: props) else {
        report(label, false, "IOSurface(properties:) nil for \(w)x\(h)"); return
    }
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    d.usage = [.shaderRead]
    d.storageMode = .shared
    guard let base = dev.makeTexture(descriptor: d, iosurface: surface, plane: 0) else {
        report(label, false, "makeTexture(iosurface:) nil for \(w)x\(h)"); return
    }
    fillViewSlice(base, w, h, slice: 0)
    guard let view = base.makeTextureView(pixelFormat: .bgra8Unorm) else {
        report(label, false, "makeTextureView nil"); return
    }
    checkViewReads("\(label)_base", base, w, h, slice: 0)
    checkViewReads("\(label)_view", view, w, h, slice: 0)
}
