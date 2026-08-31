import Metal
import Foundation

private let floatSamplePipe = pipeline("sample_float4")

private func floatTexture(_ values: [[Float]], label: String) -> MTLTexture? {
    let width = values.count
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float, width: width, height: 1, mipmapped: false)
    descriptor.usage = [.shaderRead]
    descriptor.storageMode = .shared
    guard let texture = dev.makeTexture(descriptor: descriptor) else {
        report(label, false, "makeTexture returned nil")
        return nil
    }
    let flat = values.flatMap { $0 }
    flat.withUnsafeBytes { raw in
        texture.replace(region: MTLRegionMake2D(0, 0, width, 1),
                        mipmapLevel: 0,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: width * 16)
    }
    return texture
}

private func computeFloatSampleCase(_ values: [[Float]]) {
    let width = values.count
    let name = "sample_rgba32float_compute_\(width)x1"
    guard let texture = floatTexture(values, label: name) else { return }
    let output = dev.makeBuffer(length: width * 16, options: .storageModeShared)!
    memset(output.contents(), 0xFF, width * 16)
    let ran = dev.makeBuffer(length: 4, options: .storageModeShared)!
    memset(ran.contents(), 0, 4)

    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(floatSamplePipe)
    encoder.setTexture(texture, index: 0)
    encoder.setBuffer(output, offset: 0, index: 0)
    var count = UInt32(width)
    encoder.setBytes(&count, length: 4, index: 1)
    encoder.setBuffer(ran, offset: 0, index: 4)
    encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    guard ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0] == 1 else {
        refused(name)
        return
    }
    let expected = values.flatMap { $0 }
    let pointer = output.contents().bindMemory(to: Float.self, capacity: expected.count)
    let got = Array(UnsafeBufferPointer(start: pointer, count: expected.count))
    let ok = got == expected
    report(name, ok, ok ? "\(width) float4 texels exact"
                        : "want=\(expected) got=\(got)")
}

private func vertexFloatSampleCase() {
    let name = "sample_rgba32float_vertex_positions_4x1"
    let values: [[Float]] = [
        [-0.75, -0.70, 2.0, -2.0],
        [0.75, -0.70, -4.0, 4.0],
        [0.0, 0.75, 3.5, -3.5],
        [1.25, -1.5, 5.0, 6.0],
    ]
    guard let texture = floatTexture(values, label: name) else { return }

    let pipelineDescriptor = MTLRenderPipelineDescriptor()
    pipelineDescriptor.vertexFunction = library.makeFunction(name: "float_positions_vs")
    pipelineDescriptor.fragmentFunction = library.makeFunction(name: "solid_fs")
    pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    pipelineDescriptor.inputPrimitiveTopology = .triangle
    guard let pipeline = try? dev.makeRenderPipelineState(descriptor: pipelineDescriptor) else {
        report(name, false, "render pipeline creation failed")
        return
    }

    let width = 64, height = 64
    let targetDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
    targetDescriptor.usage = [.renderTarget, .shaderRead]
    targetDescriptor.storageMode = .private
    let target = dev.makeTexture(descriptor: targetDescriptor)!

    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = target
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)!
    encoder.setRenderPipelineState(pipeline)
    encoder.setVertexTexture(texture, index: 0)
    var green: [Float] = [0, 1, 0, 1]
    encoder.setFragmentBytes(&green, length: 16, index: 0)
    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    guard let pixels = readBack(readPipe, target, width, height) else {
        refused(name)
        return
    }
    let greenPixel = pack(0, 255, 0, 255)
    let blackPixel = pack(0, 0, 0, 255)
    let lit = pixels.filter { $0 == greenPixel }.count
    let centre = pixels[32 * width + 32]
    let lowerLeftInterior = pixels[43 * width + 24]
    let corner = pixels[2 * width + 2]
    let ok = (850...1400).contains(lit)
        && centre == greenPixel
        && lowerLeftInterior == greenPixel
        && corner == blackPixel
    report(name, ok,
           "lit=\(lit) centre=\(hex(centre)) lower_left=\(hex(lowerLeftInterior)) corner=\(hex(corner))")
}

func floatSamplingCases() {
    computeFloatSampleCase([[-2.5, 0.25, 1.5, 17.0]])
    computeFloatSampleCase([
        [-0.75, -0.70, 2.0, -2.0],
        [0.75, -0.70, -4.0, 4.0],
        [0.0, 0.75, 3.5, -3.5],
        [1.25, -1.5, 5.0, 6.0],
    ])
    vertexFloatSampleCase()
}
