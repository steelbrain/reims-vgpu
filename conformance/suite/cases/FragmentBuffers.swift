import Metal
import Foundation

// ---------------------------------------------------------------------------
// A buffer bound to the *fragment* stage, read at several offsets, and rebound
// between draws inside one encoder.
//
// Every other case that gives a fragment shader a constant hands it over with
// `setFragmentBytes`, which is inline bytes in the command stream and a wholly
// different path from a bound `MTLBuffer`. Nothing in the battery covered
// `setFragmentBuffer:offset:atIndex:` at all, so a device was free to route it
// however it liked -- and the byte source a fragment bind resolves to is not
// something a screenshot can name, because a wrong offset produces a plausible
// solid colour rather than a visible defect.
//
// Three things are checked, and the second and third are the ones with teeth:
//
//   1. offset 0 -- the bind resolves to the buffer's own bytes at all.
//   2. a non-zero offset -- the window starts where the guest said, not at the
//      start of the allocation. This is the failure mode of any path that
//      resolves a bind to its backing allocation and forgets the offset.
//   3. a rebind between two draws of one encoder -- the second draw sees the
//      second offset. A device that resolves the bind once per encoder, or
//      holds a stale resolution across the rebind, draws the first colour
//      twice and passes 1 and 2 while doing so.
//
// Offsets are multiples of 256 because the constant address space's minimum
// buffer offset alignment is 256 on some macOS devices and smaller on others;
// 256 is legal on every one of them, so the case is expressible everywhere
// rather than being a limit test.
// ---------------------------------------------------------------------------

func fragmentBufferCase() {
    let w = 64, h = 64
    let fmt = MTLPixelFormat.bgra8Unorm
    guard let pipe = makeRenderPipeline("solid_fs", fmt) else {
        report("fb_pipeline", false, "pipeline failed"); return
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: fmt, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .private
    guard let rt = dev.makeTexture(descriptor: td) else {
        report("fb_target", false, "no render target"); return
    }

    let verts = dev.makeBuffer(bytes: quadVerts,
                               length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // Three colours, one per 256-byte slot. Each channel is k/255 so the
    // unorm conversion is exact and the expectation is a literal byte rather
    // than a rounding argument.
    let stride = 256
    let slots: [(UInt8, UInt8, UInt8)] = [(64, 128, 192), (192, 64, 128), (128, 192, 64)]
    let colours = dev.makeBuffer(length: stride * slots.count, options: .storageModeShared)!
    for (i, c) in slots.enumerated() {
        var rgba: [Float] = [Float(c.0) / 255, Float(c.1) / 255, Float(c.2) / 255, 1]
        memcpy(colours.contents().advanced(by: i * stride), &rgba, 16)
    }
    func want(_ i: Int) -> UInt32 {
        pack(slots[i].0, slots[i].1, slots[i].2, 255)
    }

    // One pass, one draw, the colour taken from slot `i`.
    func drawOne(_ i: Int, _ tag: String) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        enc.setFragmentBuffer(colours, offset: i * stride, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        guard let got = readBack(readPipe, rt, w, h) else { refused("fb_\(tag)"); return }
        let px = got[(h / 2) * w + (w / 2)]
        let ok = px == want(i)
        report("fb_\(tag)", ok,
               ok ? "slot \(i) at offset \(i * stride) reached the fragment stage"
                  : "got=\(hex(px)) want=\(hex(want(i))) at offset \(i * stride)")
    }

    drawOne(0, "offset_0")
    drawOne(1, "offset_256")
    drawOne(2, "offset_512")

    // Two draws in ONE encoder, each covering half the target, with the
    // fragment buffer rebound to a different offset between them. The halves
    // are read well inside their own side so rasterization edges are not the
    // test.
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    for (n, leftHalf) in [true, false].enumerated() {
        let x0: Float = leftHalf ? -1 : 0
        let x1: Float = leftHalf ? 0 : 1
        var data: [Float] = [x0, -1, 0, 1, x1, -1, 1, 1, x0, 1, 0, 0, x1, 1, 1, 0]
        // A second buffer per half would let a device resolve each bind once
        // and still pass; one buffer rebound at two offsets cannot.
        let vb = dev.makeBuffer(bytes: &data, length: 64, options: .storageModeShared)!
        enc.setVertexBuffer(vb, offset: 0, index: 0)
        enc.setFragmentBuffer(colours, offset: n * stride, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    if let got = readBack(readPipe, rt, w, h) {
        let leftPx = got[(h / 2) * w + (w / 4)]
        let rightPx = got[(h / 2) * w + (3 * w / 4)]
        let ok = leftPx == want(0) && rightPx == want(1)
        report("fb_rebound_between_draws", ok,
               ok ? "each draw read the offset bound before it"
                  : "left=\(hex(leftPx)) right=\(hex(rightPx)) "
                    + "wanted left=\(hex(want(0))) right=\(hex(want(1)))")
    } else {
        refused("fb_rebound_between_draws")
    }
}
