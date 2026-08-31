import Metal
import Foundation

private let indexedWidth = 64
private let indexedHeight = 64
private let indexedGreen = pack(0, 255, 0, 255)
private let indexedBlack = pack(0, 0, 0, 255)

private func indexedPipeline() -> MTLRenderPipelineState? {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "indexed_vs")
    descriptor.fragmentFunction = library.makeFunction(name: "solid_fs")
    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    descriptor.inputPrimitiveTopology = .triangle
    return try? dev.makeRenderPipelineState(descriptor: descriptor)
}

private func indexedPixels(name: String,
                           positions: [SIMD2<Float>],
                           encode: (MTLRenderCommandEncoder) -> Void) -> [UInt32]? {
    guard let pipeline = indexedPipeline() else {
        report(name, false, "render pipeline creation failed")
        return nil
    }
    let targetDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm,
        width: indexedWidth,
        height: indexedHeight,
        mipmapped: false)
    targetDescriptor.usage = [.renderTarget, .shaderRead]
    targetDescriptor.storageMode = .private
    guard let target = dev.makeTexture(descriptor: targetDescriptor),
          let vertices = dev.makeBuffer(
            bytes: positions,
            length: positions.count * MemoryLayout<SIMD2<Float>>.stride,
            options: .storageModeShared) else {
        report(name, false, "resource creation failed")
        return nil
    }

    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = target
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass)!
    encoder.setRenderPipelineState(pipeline)
    encoder.setVertexBuffer(vertices, offset: 0, index: 0)
    var green: [Float] = [0, 1, 0, 1]
    encoder.setFragmentBytes(&green, length: 16, index: 0)
    encode(encoder)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    guard let pixels = readBack(readPipe, target, indexedWidth, indexedHeight) else {
        refused(name)
        return nil
    }
    return pixels
}

private func instanceRelation(_ pixels: [UInt32]) -> (Bool, String) {
    let lit = pixels.filter { $0 == indexedGreen }.count
    let left = pixels[35 * indexedWidth + 18]
    let right = pixels[35 * indexedWidth + 46]
    let centre = pixels[35 * indexedWidth + 32]
    let ok = (300...520).contains(lit)
        && left == indexedGreen
        && right == indexedGreen
        && centre == indexedBlack
    return (ok,
            "lit=\(lit) left=\(hex(left)) right=\(hex(right)) centre=\(hex(centre))")
}

func indexedDrawCases() {
    let largeTriangle: [SIMD2<Float>] = [
        SIMD2(-0.75, -0.70), SIMD2(0.75, -0.70), SIMD2(0, 0.75),
    ]

    let controlName = "draw_nonindexed_shared_shader_control"
    guard let control = indexedPixels(name: controlName, positions: largeTriangle, encode: { encoder in
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }) else { return }
    let controlLit = control.filter { $0 == indexedGreen }.count
    let controlOK = (850...1400).contains(controlLit)
        && control[32 * indexedWidth + 32] == indexedGreen
        && control[2 * indexedWidth + 2] == indexedBlack
    report(controlName, controlOK, "lit=\(controlLit)")

    let uint16Name = "draw_indexed_uint16_offset_4"
    var indices16: [UInt16] = [99, 99, 0, 1, 2]
    let index16 = dev.makeBuffer(bytes: &indices16,
                                 length: indices16.count * MemoryLayout<UInt16>.stride,
                                 options: .storageModeShared)!
    if let pixels = indexedPixels(name: uint16Name, positions: largeTriangle, encode: { encoder in
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: 3,
                                      indexType: .uint16,
                                      indexBuffer: index16,
                                      indexBufferOffset: 4)
    }) {
        let ok = pixels == control
        report(uint16Name, ok, ok ? "exactly matches non-indexed control"
                                  : "pixel differences=\(zip(pixels, control).filter { $0 != $1 }.count)")
    }

    let uint32OffsetName = "draw_indexed_uint32_offset_4"
    var offsetIndices32: [UInt32] = [99, 0, 1, 2]
    let offsetIndex32 = dev.makeBuffer(bytes: &offsetIndices32,
                                       length: offsetIndices32.count * MemoryLayout<UInt32>.stride,
                                       options: .storageModeShared)!
    if let pixels = indexedPixels(name: uint32OffsetName, positions: largeTriangle, encode: { encoder in
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: 3,
                                      indexType: .uint32,
                                      indexBuffer: offsetIndex32,
                                      indexBufferOffset: 4)
    }) {
        let ok = pixels == control
        report(uint32OffsetName, ok, ok ? "exactly matches non-indexed control"
                                       : "pixel differences=\(zip(pixels, control).filter { $0 != $1 }.count)")
    }

    let baseVertexName = "draw_indexed_uint16_base_vertex_minus1"
    var baseIndices16: [UInt16] = [1, 2, 3]
    let baseIndex16 = dev.makeBuffer(bytes: &baseIndices16,
                                     length: baseIndices16.count * MemoryLayout<UInt16>.stride,
                                     options: .storageModeShared)!
    if let pixels = indexedPixels(name: baseVertexName, positions: largeTriangle, encode: { encoder in
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: 3,
                                      indexType: .uint16,
                                      indexBuffer: baseIndex16,
                                      indexBufferOffset: 0,
                                      instanceCount: 1,
                                      baseVertex: -1,
                                      baseInstance: 0)
    }) {
        let ok = pixels == control
        report(baseVertexName, ok, ok ? "exactly matches non-indexed control"
                                     : "pixel differences=\(zip(pixels, control).filter { $0 != $1 }.count)")
    }

    let uint32Name = "draw_indexed_uint32_offset_4_base_vertex_minus1"
    var indices32: [UInt32] = [99, 1, 2, 3]
    let index32 = dev.makeBuffer(bytes: &indices32,
                                 length: indices32.count * MemoryLayout<UInt32>.stride,
                                 options: .storageModeShared)!
    if let pixels = indexedPixels(name: uint32Name, positions: largeTriangle, encode: { encoder in
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: 3,
                                      indexType: .uint32,
                                      indexBuffer: index32,
                                      indexBufferOffset: 4,
                                      instanceCount: 1,
                                      baseVertex: -1,
                                      baseInstance: 0)
    }) {
        let ok = pixels == control
        report(uint32Name, ok, ok ? "exactly matches non-indexed control"
                                  : "pixel differences=\(zip(pixels, control).filter { $0 != $1 }.count)")
    }

    let instanceName = "draw_indexed_two_instances_base_instance_5"
    let smallTriangle: [SIMD2<Float>] = [
        SIMD2(-0.25, -0.40), SIMD2(0.25, -0.40), SIMD2(0, 0.40),
    ]
    let nonindexedInstanceName = "draw_nonindexed_two_instances_base_instance_5"
    let nonindexedInstances = indexedPixels(
        name: nonindexedInstanceName, positions: smallTriangle, encode: { encoder in
            encoder.drawPrimitives(type: .triangle,
                                   vertexStart: 0,
                                   vertexCount: 3,
                                   instanceCount: 2,
                                   baseInstance: 5)
        })
    if let pixels = nonindexedInstances {
        let verdict = instanceRelation(pixels)
        report(nonindexedInstanceName, verdict.0, verdict.1)
    }

    var instanceIndices: [UInt16] = [0, 1, 2]
    let instanceIndex = dev.makeBuffer(bytes: &instanceIndices,
                                       length: instanceIndices.count * MemoryLayout<UInt16>.stride,
                                       options: .storageModeShared)!
    if let pixels = indexedPixels(name: instanceName, positions: smallTriangle, encode: { encoder in
        encoder.drawIndexedPrimitives(type: .triangle,
                                      indexCount: 3,
                                      indexType: .uint16,
                                      indexBuffer: instanceIndex,
                                      indexBufferOffset: 0,
                                      instanceCount: 2,
                                      baseVertex: 0,
                                      baseInstance: 5)
    }) {
        let relation = instanceRelation(pixels)
        let matchesControl = nonindexedInstances.map { pixels == $0 } ?? false
        report(instanceName, relation.0 && matchesControl,
               "\(relation.1) matches_nonindexed=\(matchesControl)")
    }
}
