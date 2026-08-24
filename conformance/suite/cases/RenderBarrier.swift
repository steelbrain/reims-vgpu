import Metal
import Foundation

// A producer and consumer in one render encoder, separated only by the
// resource barrier whose contract is under test. Alternating the token makes a
// stale value a defined wrong answer after the first round.
func renderBarrierCase() {
    let w = 256, h = 256
    let fmt = MTLPixelFormat.bgra8Unorm
    guard let writePipe = makeRenderPipeline("render_barrier_write_fs", fmt),
          let consumerPipe = makeRenderPipeline("render_barrier_read_fs", fmt) else {
        report("render_barrier_resource_fragment", false, "pipeline failed")
        return
    }

    let targetDesc = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: fmt, width: w, height: h, mipmapped: false)
    targetDesc.usage = [.renderTarget, .shaderRead]
    targetDesc.storageMode = .private
    guard let target = dev.makeTexture(descriptor: targetDesc),
          let words = dev.makeBuffer(length: w * h * MemoryLayout<UInt32>.size,
                                     options: .storageModePrivate),
          let vertices = dev.makeBuffer(bytes: quadVerts,
                                        length: quadVerts.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared) else {
        report("render_barrier_resource_fragment", false, "resource creation failed")
        return
    }

    let green = pack(0, 255, 0, 255)
    for round in 0..<8 {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        pass.colorAttachments[0].storeAction = .store

        var params = SIMD2<UInt32>(UInt32(w), round.isMultiple(of: 2) ? 0x13579BDF : 0x2468ACE0)
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setVertexBuffer(vertices, offset: 0, index: 0)
        enc.setFragmentBuffer(words, offset: 0, index: 0)
        enc.setFragmentBytes(&params, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)

        enc.setRenderPipelineState(writePipe)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.memoryBarrier(resources: [words], after: .fragment, before: .fragment)
        enc.setRenderPipelineState(consumerPipe)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        guard let got = readBack(readPipe, target, w, h) else {
            refused("render_barrier_resource_fragment")
            return
        }
        if let bad = got.firstIndex(where: { $0 != green }) {
            report("render_barrier_resource_fragment", false,
                   "round=\(round) pixel=\(bad) got=\(hex(got[bad])) want=\(hex(green))")
            return
        }
    }

    report("render_barrier_resource_fragment", true,
           "8 alternating producer/consumer rounds were visible")
}
