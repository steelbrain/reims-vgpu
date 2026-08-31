import Metal
import Foundation

// Shaderless integer attachment clears. These cases contain no guest shader,
// so a native PASS / guest FAIL cannot be a shader-translation defect. The
// private arm observes the stored Vulkan image through a blit; the shared arm
// observes the clear-only publication into guest-visible texture bytes.

private let integerClearRed: UInt16 = 1
private let integerClearGreen: UInt16 = 258

private func encodeIntegerClear(_ texture: MTLTexture, _ label: String) -> Bool {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(
        red: Double(integerClearRed),
        green: Double(integerClearGreen),
        blue: 0,
        alpha: 0)
    pass.colorAttachments[0].storeAction = .store

    guard let command = queue.makeCommandBuffer(),
          let encoder = command.makeRenderCommandEncoder(descriptor: pass) else {
        report(label, false, "clear-only render encoder creation failed")
        return false
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        report(label, false, "clear-only command failed: \(String(describing: command.error))")
        return false
    }
    return true
}

private func verifyIntegerClear(_ words: UnsafePointer<UInt16>,
                                texels: Int,
                                _ label: String) {
    var first: (index: Int, red: UInt16, green: UInt16)?
    var wrong = 0
    for index in 0..<texels {
        let red = words[index * 2]
        let green = words[index * 2 + 1]
        if red != integerClearRed || green != integerClearGreen {
            wrong += 1
            if first == nil { first = (index, red, green) }
        }
    }
    if let first {
        report(label, false,
               "wrong=\(wrong)/\(texels) first=\(first.index) "
               + "got=[\(first.red),\(first.green)] "
               + "want=[\(integerClearRed),\(integerClearGreen)]")
    } else {
        report(label, true,
               "\(texels) RG16Uint texels retained [\(integerClearRed),\(integerClearGreen)]")
    }
}

private func integerClearPrivateCase() {
    let label = "clear_only_rg16uint_private_store"
    let width = 64
    let height = 17
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rg16Uint, width: width, height: height, mipmapped: false)
    descriptor.storageMode = .private
    descriptor.usage = [.renderTarget]
    guard let texture = dev.makeTexture(descriptor: descriptor) else {
        report(label, false, "private RG16Uint render target creation failed")
        return
    }
    guard encodeIntegerClear(texture, label) else { return }

    let bytesPerRow = width * 4
    let length = bytesPerRow * height
    guard let output = dev.makeBuffer(length: length, options: .storageModeShared),
          let command = queue.makeCommandBuffer(),
          let blit = command.makeBlitCommandEncoder() else {
        report(label, false, "private clear readback resource creation failed")
        return
    }
    memset(output.contents(), 0xEE, length)
    blit.copy(from: texture,
              sourceSlice: 0,
              sourceLevel: 0,
              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: width, height: height, depth: 1),
              to: output,
              destinationOffset: 0,
              destinationBytesPerRow: bytesPerRow,
              destinationBytesPerImage: length)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
        report(label, false, "private clear blit failed: \(String(describing: command.error))")
        return
    }
    verifyIntegerClear(
        output.contents().bindMemory(to: UInt16.self, capacity: width * height * 2),
        texels: width * height,
        label)
}

private func integerClearSharedCase() {
    let label = "clear_only_rg16uint_shared_store"
    let width = 60
    let height = 17
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rg16Uint, width: width, height: height, mipmapped: false)
    descriptor.storageMode = .shared
    descriptor.usage = [.renderTarget]
    guard let texture = dev.makeTexture(descriptor: descriptor) else {
        report(label, false, "shared RG16Uint render target creation failed")
        return
    }

    let seed = [UInt16](repeating: 0xEEEE, count: width * height * 2)
    seed.withUnsafeBytes { bytes in
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: bytes.baseAddress!,
            bytesPerRow: width * 4)
    }
    guard encodeIntegerClear(texture, label) else { return }

    var output = [UInt16](repeating: 0xEEEE, count: width * height * 2)
    output.withUnsafeMutableBytes { bytes in
        texture.getBytes(
            bytes.baseAddress!,
            bytesPerRow: width * 4,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0)
    }
    output.withUnsafeBufferPointer { words in
        verifyIntegerClear(words.baseAddress!, texels: width * height, label)
    }
}

func integerClearCases() {
    integerClearPrivateCase()
    integerClearSharedCase()
}
