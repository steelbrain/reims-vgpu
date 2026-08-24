import Metal
import Foundation

/// A Store-only multisample attachment remains a multisample texture after its
/// encoder ends. Four unequal samples make preservation distinguishable from
/// an implicit resolve, and the later command buffer forces the observation
/// across the encoder/submission lifetime boundary.
func multisampleStoreCase() {
    let label = "msaa4_color_store_no_resolve"
    let w = 8, h = 8, samples = 4
    guard dev.supportsTextureSampleCount(samples) else {
        skip(label, "device does not report 4x texture sampling")
        return
    }

    let descriptor = MTLRenderPipelineDescriptor()
    descriptor.vertexFunction = library.makeFunction(name: "quad_vs")
    descriptor.fragmentFunction = library.makeFunction(name: "sample_id_fs")
    descriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
    descriptor.rasterSampleCount = samples
    guard let renderPipeline = try? dev.makeRenderPipelineState(descriptor: descriptor) else {
        report(label, false, "4x render pipeline creation failed")
        return
    }

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    textureDescriptor.textureType = .type2DMultisample
    textureDescriptor.sampleCount = samples
    textureDescriptor.storageMode = .private
    textureDescriptor.usage = [.renderTarget, .shaderRead]
    guard let texture = dev.makeTexture(descriptor: textureDescriptor),
          let vertices = dev.makeBuffer(bytes: quadVerts,
                                        length: quadVerts.count * MemoryLayout<Float>.size,
                                        options: .storageModeShared) else {
        report(label, false, "4x resource creation failed")
        return
    }

    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = texture
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let producer = queue.makeCommandBuffer()!
    let render = producer.makeRenderCommandEncoder(descriptor: pass)!
    render.setRenderPipelineState(renderPipeline)
    render.setVertexBuffer(vertices, offset: 0, index: 0)
    render.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    render.endEncoding()
    producer.commit()
    producer.waitUntilCompleted()

    guard let got = readBackMultisample(texture, w, h, samples: samples) else {
        refused(label)
        return
    }
    let expected = [
        pack(255, 0, 0, 255),
        pack(0, 255, 0, 255),
        pack(0, 0, 255, 255),
        pack(255, 255, 255, 255),
    ]
    var wrong = 0
    var first: (pixel: Int, sample: Int, got: UInt32, want: UInt32)?
    var badPixels: [(Int, Int)] = []
    for pixel in 0..<(w * h) {
        var pixelWrong = false
        for sample in 0..<samples {
            let index = pixel * samples + sample
            if got[index] != expected[sample] {
                wrong += 1
                pixelWrong = true
                if first == nil {
                    first = (pixel, sample, got[index], expected[sample])
                }
            }
        }
        if pixelWrong { badPixels.append((pixel % w, pixel / w)) }
    }

    if let first {
        report(label, false,
               "wrong=\(wrong)/\(got.count) first_pixel=\(first.pixel) "
               + "sample=\(first.sample) got=\(hex(first.got)) want=\(hex(first.want)) "
               + badMap(badPixels, w, h))
    } else {
        report(label, true, "\(w * h) pixels retained four distinct samples")
    }
}
