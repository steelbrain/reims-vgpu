import Metal
import Foundation

// Primitive topology has two independently supplied terms: the pipeline's
// topology class and the draw call's concrete primitive type. Each case keeps
// those terms consistent and checks a spatial relation that a different
// primitive cannot satisfy.

private let topologyWidth = 64
private let topologyHeight = 64
private let topologyGreen = pack(0, 255, 0, 255)
private let topologyBlack = pack(0, 0, 0, 255)

private func topologyPipeline(vertex: String,
                              class topologyClass: MTLPrimitiveTopologyClass)
    -> MTLRenderPipelineState? {
    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: vertex)
    descriptor.fragmentFunction = library.makeFunction(
        name: vertex == "point_vs" ? "point_fs" : "solid_fs")
    descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
    descriptor.inputPrimitiveTopology = topologyClass
    return try? dev.makeRenderPipelineState(descriptor: descriptor)
}

private func topologyPixels(name: String,
                            positions: [SIMD2<Float>],
                            primitive: MTLPrimitiveType,
                            topologyClass: MTLPrimitiveTopologyClass,
                            point: Bool = false) -> [UInt32]? {
    let vertexName = point ? "point_vs" : "topology_vs"
    guard let pipeline = topologyPipeline(vertex: vertexName, class: topologyClass) else {
        report(name, false, "render pipeline creation failed")
        return nil
    }

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm,
        width: topologyWidth,
        height: topologyHeight,
        mipmapped: false)
    textureDescriptor.usage = [.renderTarget, .shaderRead]
    textureDescriptor.storageMode = .private
    guard let target = dev.makeTexture(descriptor: textureDescriptor),
          let vertices = dev.makeBuffer(bytes: positions,
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
    if !point {
        var colour: [Float] = [0, 1, 0, 1]
        encoder.setFragmentBytes(&colour, length: 16, index: 0)
    }
    encoder.drawPrimitives(type: primitive, vertexStart: 0, vertexCount: positions.count)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    guard let pixels = readBack(readPipe, target, topologyWidth, topologyHeight) else {
        refused(name)
        return nil
    }
    return pixels
}

private func lit(_ pixel: UInt32) -> Bool { pixel == topologyGreen }

private func pointTopologyCase() {
    let name = "topology_point_three_islands"
    let positions: [SIMD2<Float>] = [
        SIMD2(-0.65, -0.65),
        SIMD2(0.65, -0.65),
        SIMD2(0.0, 0.65),
    ]
    guard let pixels = topologyPixels(name: name, positions: positions,
                                      primitive: .point, topologyClass: .point,
                                      point: true) else { return }
    let count = pixels.filter { lit($0) }.count
    let thirds = [
        pixels[53 * topologyWidth + 11],
        pixels[53 * topologyWidth + 52],
        pixels[11 * topologyWidth + 32],
    ].filter { lit($0) }.count
    let ok = (9...60).contains(count) && thirds == 3
    report(name, ok, "lit=\(count) point_centres=\(thirds)/3")
}

private func lineTopologyCase() {
    let name = "topology_line_horizontal"
    let positions = [SIMD2<Float>(-0.8, 0), SIMD2<Float>(0.8, 0)]
    guard let pixels = topologyPixels(name: name, positions: positions,
                                      primitive: .line, topologyClass: .line) else { return }
    let litRows = Set((0..<topologyHeight).filter { y in
        pixels[(y * topologyWidth)..<((y + 1) * topologyWidth)].contains(where: lit)
    })
    let count = pixels.filter { lit($0) }.count
    let ok = (32...128).contains(count) && (1...2).contains(litRows.count)
    report(name, ok, "lit=\(count) rows=\(litRows.sorted())")
}

private func lineStripTopologyCase() {
    let name = "topology_line_strip_two_arms"
    let positions: [SIMD2<Float>] = [
        SIMD2(-0.75, -0.7), SIMD2(0, 0.7), SIMD2(0.75, -0.7),
    ]
    guard let pixels = topologyPixels(name: name, positions: positions,
                                      primitive: .lineStrip, topologyClass: .line) else { return }
    var left = 0
    var right = 0
    for y in 0..<topologyHeight {
        for x in 0..<topologyWidth where lit(pixels[y * topologyWidth + x]) {
            if x < topologyWidth / 2 { left += 1 } else { right += 1 }
        }
    }
    let total = left + right
    let ok = left >= 20 && right >= 20 && (60...180).contains(total)
    report(name, ok, "lit=\(total) left=\(left) right=\(right)")
}

private func triangleTopologyCase() {
    let name = "topology_triangle_filled"
    let positions: [SIMD2<Float>] = [
        SIMD2(-0.8, -0.7), SIMD2(0.8, -0.7), SIMD2(0, 0.8),
    ]
    guard let pixels = topologyPixels(name: name, positions: positions,
                                      primitive: .triangle, topologyClass: .triangle) else { return }
    let count = pixels.filter { lit($0) }.count
    let centre = pixels[32 * topologyWidth + 32]
    let corner = pixels[2 * topologyWidth + 2]
    let ok = (900...1600).contains(count) && lit(centre) && corner == topologyBlack
    report(name, ok, "lit=\(count) centre=\(hex(centre)) corner=\(hex(corner))")
}

func topologyCases() {
    pointTopologyCase()
    lineTopologyCase()
    lineStripTopologyCase()
    triangleTopologyCase()
}
