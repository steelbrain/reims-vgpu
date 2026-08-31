import Metal
import Foundation

// ---------------------------------------------------------------------------
// Two textures over one buffer allocation, reading its bytes at different
// widths.
//
// `makeTexture(descriptor:offset:bytesPerRow:)` on an `MTLBuffer` gives a
// texture whose storage *is* that buffer's bytes, so two textures declared over
// one buffer at formats of different bytes per texel are one set of bytes and
// two texel counts. They share nothing else: their texel blocks are different
// widths, so no copy between the images and no reinterpreting view relates
// them, and a device that keeps a tiled image per texture has to leave the
// texel domain entirely to make one current from the other.
//
// Whether that is a copy the guest is entitled to expect is the question the
// oracle answers, and it is the reason this file exists rather than a Rust
// test: `heap_texture_placement_overlap_incompatible_widths` covers the *heap*
// reading of the same shape, where Metal leaves an alias's contents undefined
// once another writes, and claims nothing about bytes crossing. Buffer-backed
// storage is the other reading and the two must not be assumed to agree.
//
// The values are chosen so the expectation is bytes and never arithmetic. A
// `rgba16Float` texel of (1, 0, 0, 1) is the four half words 0x3C00, 0x0000,
// 0x0000, 0x3C00, which is `00 3C 00 00 00 00 00 3C` in memory. Read back as
// `rgba8Unorm` those eight bytes are two texels, (0, 0x3C, 0, 0) and
// (0, 0, 0, 0x3C), so the narrow texture's rows alternate between exactly two
// values and a device that lands the bytes at the wrong offset produces
// neither.
// ---------------------------------------------------------------------------

private func fillTexture(_ texture: MTLTexture, _ value: SIMD4<Float>) -> Bool {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return false
    }
    var color = value
    encoder.setComputePipelineState(pipeline("heap_alias_fill"))
    encoder.setTexture(texture, index: 0)
    encoder.setBytes(&color, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
    encoder.dispatchThreads(
        MTLSize(width: texture.width, height: texture.height, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.status == .completed
}

func bufferBackedAliasWidthCase() {
    let label = "buffer_backed_alias_incompatible_widths"
    let wideWidth = 64, height = 64
    let narrowWidth = 128
    let rowBytes = wideWidth * 8
    let wideAlign = dev.minimumLinearTextureAlignment(for: .rgba16Float)
    let narrowAlign = dev.minimumLinearTextureAlignment(for: .rgba8Unorm)
    guard wideAlign > 0, narrowAlign > 0,
          rowBytes % wideAlign == 0, rowBytes % narrowAlign == 0 else {
        skip(label, "a \(rowBytes)-byte row does not satisfy this device's "
                    + "minimumLinearTextureAlignment (rgba16Float=\(wideAlign), "
                    + "rgba8Unorm=\(narrowAlign))")
        return
    }
    guard let buffer = dev.makeBuffer(length: rowBytes * height,
                                      options: .storageModeShared) else {
        report(label, false, "the shared buffer could not be created")
        return
    }

    let wide = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba16Float, width: wideWidth, height: height, mipmapped: false)
    wide.storageMode = .shared
    wide.usage = [.shaderRead, .shaderWrite]
    let narrow = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: narrowWidth, height: height, mipmapped: false)
    narrow.storageMode = .shared
    narrow.usage = [.shaderRead, .shaderWrite]

    guard let wideTexture = buffer.makeTexture(descriptor: wide, offset: 0,
                                               bytesPerRow: rowBytes) else {
        report(label, false, "the buffer refused its rgba16Float texture")
        return
    }
    guard let narrowTexture = buffer.makeTexture(descriptor: narrow, offset: 0,
                                                 bytesPerRow: rowBytes) else {
        report(label, false, "the buffer refused an overlapping rgba8Unorm texture")
        return
    }

    guard fillTexture(wideTexture, SIMD4<Float>(1, 0, 0, 1)) else {
        report(label, false, "commands using the rgba16Float alias did not complete")
        return
    }
    guard let wideGot = readBack(readPipe, wideTexture, wideWidth, height) else {
        refused(label)
        return
    }
    let red = pack(255, 0, 0, 255)
    let wideBad = wideGot.indices.filter { wideGot[$0] != red }
    guard wideBad.isEmpty else {
        report(label, false,
               "the rgba16Float alias did not read its own write: "
                 + "wrong=\(wideBad.count)/\(wideGot.count) first=\(hex(wideGot[wideBad[0]]))")
        return
    }

    guard let narrowGot = readBack(readPipe, narrowTexture, narrowWidth, height) else {
        refused(label)
        return
    }
    // The eight bytes of one wide texel are two narrow ones, in that order.
    let evenTexel = pack(0, 0x3C, 0, 0)
    let oddTexel = pack(0, 0, 0, 0x3C)
    var bad: [(Int, Int)] = []
    var first = ""
    for y in 0..<height {
        for x in 0..<narrowWidth {
            let want = x % 2 == 0 ? evenTexel : oddTexel
            let have = narrowGot[y * narrowWidth + x]
            if want != have {
                bad.append((x, y))
                if first.isEmpty {
                    first = "first_bad=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty
             ? "every byte the rgba16Float alias wrote reached the rgba8Unorm one"
             : "\(bad.count)/\(narrowWidth * height) wrong \(first) "
                 + badMap(bad, narrowWidth, height))
}

// ---------------------------------------------------------------------------
// The same pair, with the second write covering part of one row rather than
// the whole allocation.
//
// A byte range that starts and ends inside a row is a union of row segments and
// not a rectangle in either texture's coordinates, which is the shape a device
// translating between the two has to decompose rather than refuse. Writing the
// narrow texture's rows 4 through 7 leaves every other row holding what the
// wide one put there, so a device that widens the range loses those rows and a
// device that drops it loses these.
// ---------------------------------------------------------------------------

func bufferBackedAliasPartialCase() {
    let label = "buffer_backed_alias_partial_row_range"
    let wideWidth = 64, height = 64
    let narrowWidth = 128
    let rowBytes = wideWidth * 8
    let wideAlign = dev.minimumLinearTextureAlignment(for: .rgba16Float)
    let narrowAlign = dev.minimumLinearTextureAlignment(for: .rgba8Unorm)
    guard wideAlign > 0, narrowAlign > 0,
          rowBytes % wideAlign == 0, rowBytes % narrowAlign == 0 else {
        skip(label, "a \(rowBytes)-byte row does not satisfy this device's "
                    + "minimumLinearTextureAlignment (rgba16Float=\(wideAlign), "
                    + "rgba8Unorm=\(narrowAlign))")
        return
    }
    guard let buffer = dev.makeBuffer(length: rowBytes * height,
                                      options: .storageModeShared) else {
        report(label, false, "the shared buffer could not be created")
        return
    }
    let wide = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba16Float, width: wideWidth, height: height, mipmapped: false)
    wide.storageMode = .shared
    wide.usage = [.shaderRead, .shaderWrite]
    let narrow = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: narrowWidth, height: 4, mipmapped: false)
    narrow.storageMode = .shared
    narrow.usage = [.shaderRead, .shaderWrite]

    guard let wideTexture = buffer.makeTexture(descriptor: wide, offset: 0,
                                               bytesPerRow: rowBytes) else {
        report(label, false, "the buffer refused its rgba16Float texture")
        return
    }
    // Four rows starting at row 4, addressed as its own texture over the same
    // buffer. The offset has to satisfy the format's alignment too.
    let offset = rowBytes * 4
    guard offset % wideAlign == 0, offset % narrowAlign == 0,
          let narrowTexture = buffer.makeTexture(descriptor: narrow, offset: offset,
                                                 bytesPerRow: rowBytes) else {
        skip(label, "a \(offset)-byte offset does not satisfy this device's "
                    + "minimumLinearTextureAlignment")
        return
    }

    guard fillTexture(wideTexture, SIMD4<Float>(1, 0, 0, 1)),
          fillTexture(narrowTexture, SIMD4<Float>(0, 1, 0, 1)) else {
        report(label, false, "commands using one of the aliases did not complete")
        return
    }
    guard let wideGot = readBack(readPipe, wideTexture, wideWidth, height) else {
        refused(label)
        return
    }

    // Outside the strip the wide texture still holds its own write. Inside it,
    // the narrow one wrote (0, 1, 0, 1) as rgba8Unorm -- `00 FF 00 FF` --
    // and two of those are one wide texel, whose channels are then the half
    // words 0xFF00, 0xFF00, 0xFF00, 0xFF00. That is -0.0004... in half, which
    // clamps to 0 through the readback's unorm packing, so the strip reads
    // black and the rest reads red.
    let red = pack(255, 0, 0, 255)
    let strip = pack(0, 0, 0, 0)
    var bad: [(Int, Int)] = []
    var first = ""
    for y in 0..<height {
        for x in 0..<wideWidth {
            let want = (4..<8).contains(y) ? strip : red
            let have = wideGot[y * wideWidth + x]
            if want != have {
                bad.append((x, y))
                if first.isEmpty {
                    first = "first_bad=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty
             ? "the four-row write landed on exactly its own rows of the wide alias"
             : "\(bad.count)/\(wideWidth * height) wrong \(first) "
                 + badMap(bad, wideWidth, height))
}
