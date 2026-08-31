import Metal
import Foundation

// ---------------------------------------------------------------------------
// One texture copied onto itself, with the source and destination rectangles
// overlapping.
//
// `copyFromTexture:toTexture:` naming one texture twice at different origins is
// a shape the guest issues: a driven Maps interaction on the macos-13 rail
// produces it several times a boot, both as a horizontal shift inside one row
// and as a one-texel column shifted down a dozen rows. Nothing in the encoder
// refuses it, so the guest treats the copy as complete.
//
// What the overlapping texels are afterwards is the question, and it has one
// answer rather than being undefined: measured on Apple silicon, the whole
// source region is read before any destination byte is written. A row-at-a-time
// implementation gets a different answer whenever a destination row precedes a
// source row it still has to read, which is exactly what both live shapes do.
//
// So the expectation below is computed against a pristine snapshot of the
// source. `disjoint` is the control: there the two readings agree, so a device
// that refuses every self-copy fails it too and a device that only mishandles
// the overlap does not — which is what tells the two apart.
// ---------------------------------------------------------------------------

private func texelPattern(_ x: Int, _ y: Int) -> UInt32 {
    // Distinct per texel within the sizes used here, so a copy that lands the
    // wrong source texel is visible rather than aliasing onto the right value.
    pack(UInt8(x & 0xFF), UInt8(y & 0xFF), UInt8((x &* 7 &+ y &* 31) & 0xFF), 255)
}

private func selfCopyCase(_ label: String,
                          width: Int, height: Int,
                          src: MTLOrigin, dst: MTLOrigin, region: MTLSize) {
    let bpr = width * 4
    let d = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
    d.storageMode = .shared
    d.usage = [.shaderRead, .shaderWrite]
    guard let tex = dev.makeTexture(descriptor: d) else {
        report(label, false, "makeTexture nil for the shared \(width)x\(height) image")
        return
    }

    // An ordinary image, not one declared over a buffer: the copy below is the
    // only thing this case asks the device to do with it, and the readback is
    // the suite's usual one rather than a buffer-backed bind.
    var seed = [UInt32](repeating: 0, count: width * height)
    for y in 0..<height {
        for x in 0..<width { seed[y * width + x] = texelPattern(x, y) }
    }
    seed.withUnsafeBytes { raw in
        tex.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                    withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }

    // Source-snapshot semantics: every destination texel takes the source as it
    // stood before the copy, never a byte this copy has already written.
    var want = [UInt32](repeating: 0, count: width * height)
    for y in 0..<height {
        for x in 0..<width { want[y * width + x] = texelPattern(x, y) }
    }
    for dy in 0..<region.height {
        for dx in 0..<region.width {
            want[(dst.y + dy) * width + dst.x + dx] = texelPattern(src.x + dx, src.y + dy)
        }
    }

    guard let cb = queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else {
        report(label, false, "no blit encoder"); return
    }
    blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: src, sourceSize: region,
              to: tex, destinationSlice: 0, destinationLevel: 0, destinationOrigin: dst)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let e = cb.error {
        report(label, false, "the blit encoder refused a copy the guest treats as "
                             + "complete: \(e.localizedDescription)")
        return
    }

    guard let got = readBack(readPipe, tex, width, height) else { refused(label); return }
    var bad: [(Int, Int)] = []
    var first = ""
    var unmoved = 0
    for y in 0..<height {
        for x in 0..<width where got[y * width + x] != want[y * width + x] {
            bad.append((x, y))
            // The destination still holding what it held before the copy is the
            // signature of a dropped copy, as against a mis-ordered one.
            if got[y * width + x] == texelPattern(x, y) { unmoved += 1 }
            if first.isEmpty {
                first = "first=(\(x),\(y)) want=\(hex(want[y * width + x])) "
                      + "got=\(hex(got[y * width + x]))"
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty
             ? "the overlapping copy read its source before writing its destination"
             : "wrong=\(bad.count)/\(width * height) still_pre_copy=\(unmoved) \(first)"
               + badMap(bad, width, height))
}

/// A row copied onto itself 14 texels to the right — the destination overwrites
/// source texels the same row still has to read.
func selfCopyOverlapRowShiftCase() {
    selfCopyCase("self_copy_overlap_row_shift_14",
                 width: 128, height: 32,
                 src: MTLOrigin(x: 0, y: 0, z: 0),
                 dst: MTLOrigin(x: 14, y: 0, z: 0),
                 region: MTLSize(width: 20, height: 1, depth: 1))
}

/// A one-texel column shifted twelve rows down — a top-down row loop overwrites
/// rows 12 through 16 before it reads them.
func selfCopyOverlapColumnShiftCase() {
    selfCopyCase("self_copy_overlap_column_shift_12",
                 width: 256, height: 32,
                 src: MTLOrigin(x: 0, y: 0, z: 0),
                 dst: MTLOrigin(x: 0, y: 12, z: 0),
                 region: MTLSize(width: 1, height: 17, depth: 1))
}

/// The control: the same call shape with rectangles that do not intersect, where
/// both readings agree. A device that refuses every self-copy fails this too.
func selfCopyDisjointCase() {
    selfCopyCase("self_copy_disjoint_control",
                 width: 128, height: 32,
                 src: MTLOrigin(x: 0, y: 0, z: 0),
                 dst: MTLOrigin(x: 64, y: 0, z: 0),
                 region: MTLSize(width: 20, height: 1, depth: 1))
}
