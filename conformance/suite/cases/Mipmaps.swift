import Metal
import Foundation
import IOSurface

/// The same chain, uploaded by a blit from a buffer rather than by
/// `replace(region:mipmapLevel:)`. Two different routes reach a level's
/// storage, and a device can lose one and keep the other -- which is the
/// difference between a broken texture-write path and a texture that has no
/// levels above zero at all.
func mipBlitCase() {
    let size = 64
    let levels = 7
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                     width: size, height: size, mipmapped: true)
    d.mipmapLevelCount = levels
    d.storageMode = .private
    d.usage = [.shaderRead, .shaderWrite]
    guard let tex = dev.makeTexture(descriptor: d) else {
        report("mip_blit_create", false, "makeTexture nil"); return
    }
    let cb = queue.makeCommandBuffer()!
    let blit = cb.makeBlitCommandEncoder()!
    var staging: [MTLBuffer] = []
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let bpr = dim * 4
        let buf = dev.makeBuffer(length: bpr * dim, options: .storageModeShared)!
        memset(buf.contents(), Int32(marker), bpr * dim)
        staging.append(buf)
        blit.copy(from: buf, sourceOffset: 0,
                  sourceBytesPerRow: bpr, sourceBytesPerImage: bpr * dim,
                  sourceSize: MTLSize(width: dim, height: dim, depth: 1),
                  to: tex, destinationSlice: 0, destinationLevel: l,
                  destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    }
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let want = pack(marker, marker, marker, marker)
        guard let got = readBack(fetchLevelPipe, tex, dim, dim, level: l) else {
            refused("mip_blit_level_\(l)_size\(dim)"); continue
        }
        let ok = got.allSatisfy { $0 == want }
        report("mip_blit_level_\(l)_size\(dim)", ok,
               ok ? "\(dim * dim) texels are level \(l)'s marker"
                  : "want=\(hex(want)) got=\(hex(got[0]))")
    }
}

// ---------------------------------------------------------------------------
// F. Vertex-buffer content across submissions.
//
// A draw's geometry comes out of a buffer the guest owns and rewrites between
// frames. If a device caches that window and serves a stale copy, the geometry
// is drawn in last frame's place -- which for a label layer means glyph quads
// landing where nothing is composited, i.e. absence over correct terrain.
// Two submissions with different vertex data, each verified.
// ---------------------------------------------------------------------------

func vertexBufferCase() {
    let w = 64, h = 64
    let fmt = MTLPixelFormat.bgra8Unorm
    guard let pipe = makeRenderPipeline("solid_fs", fmt) else {
        report("vb_pipeline", false, "pipeline failed"); return
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .private
    let rt = dev.makeTexture(descriptor: td)!
    let verts = dev.makeBuffer(length: 4 * 4 * 4, options: .storageModeShared)!
    var colour: [Float] = [1, 1, 1, 1]

    // Draw a strip covering only the requested x half, then check which half of
    // the target changed.
    func drawHalf(_ leftHalf: Bool, _ tag: String) {
        let x0: Float = leftHalf ? -1 : 0
        let x1: Float = leftHalf ? 0 : 1
        let data: [Float] = [x0, -1, 0, 1, x1, -1, 1, 1, x0, 1, 0, 0, x1, 1, 1, 0]
        memcpy(verts.contents(), data, data.count * 4)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        enc.setFragmentBytes(&colour, length: 16, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        guard let got = readBack(readPipe, rt, w, h) else {
            refused("vb_\(tag)"); return
        }
        let white = pack(255, 255, 255, 255)
        let black = pack(0, 0, 0, 255)
        // Sample well inside each half so rasterization edges are not the test.
        let leftPx = got[(h / 2) * w + (w / 4)]
        let rightPx = got[(h / 2) * w + (3 * w / 4)]
        let want = leftHalf ? (white, black) : (black, white)
        let ok = leftPx == want.0 && rightPx == want.1
        report("vb_\(tag)", ok,
               ok ? "geometry landed in the \(leftHalf ? "left" : "right") half"
                  : "left=\(hex(leftPx)) right=\(hex(rightPx)) wanted left=\(hex(want.0)) right=\(hex(want.1))")
    }
    drawHalf(true, "first_submission_left")
    drawHalf(false, "second_submission_right")
    drawHalf(true, "third_submission_left_again")
}
