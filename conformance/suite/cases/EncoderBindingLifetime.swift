import Metal
import Foundation

// ---------------------------------------------------------------------------
// Where an argument binding stops being in force.
//
// Metal scopes argument tables to the *encoder*: a binding set on one render
// command encoder is not in force on the next one, and every encoder starts
// with nothing bound. Every case in this battery so far rebinds inside one
// encoder, so nothing here has ever asked what happens at the boundary.
//
// # Why this is a case and not a comment
//
// A device that wants to stop re-resolving an argument table every draw has to
// decide when a table is "the same as last time". The cheap and exact test is
// pointer identity on the table the guest presented -- and pointer identity is
// blind to encoder boundaries. The same table re-presented in a *new* encoder
// is the same allocation and a wholly different binding state, so a device that
// compares only the table skips a bind that Metal requires, and the draw reads
// whatever the descriptor happened to hold.
//
// That failure is silent. It needs the same allocation to be presented twice
// across a boundary, which is exactly what a compositor does and exactly what
// no existing case here constructs. The three checks below are the boundaries a
// per-draw dirty flag can be wrong about, in increasing order of how likely a
// device is to carry state across them:
//
//   1. a second encoder in the SAME command buffer, identical bind re-issued
//   2. a second encoder in the same command buffer, DIFFERENT bind
//   3. a second encoder in a DIFFERENT command buffer, identical bind re-issued
//
// Each second encoder clears the target first, so the first encoder's pixels
// cannot stand in for the second's. A device that drops the second bind renders
// the clear colour or a stale slot, and both are distinguishable from the
// answer.
// ---------------------------------------------------------------------------

func encoderBindingLifetimeCase() {
    let w = 64, h = 64
    let fmt = MTLPixelFormat.bgra8Unorm
    guard let pipe = makeRenderPipeline("solid_fs", fmt) else {
        report("ebl_pipeline", false, "pipeline failed"); return
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: fmt, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .private
    guard let rt = dev.makeTexture(descriptor: td) else {
        report("ebl_target", false, "no render target"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts,
                               length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // Two slots, exact under the unorm conversion so the expectation is a byte
    // and not a rounding argument. The clear colour is a third value, so
    // "the second encoder did not draw" is distinguishable from either slot.
    let stride = 256
    let slots: [(UInt8, UInt8, UInt8)] = [(64, 128, 192), (192, 64, 128)]
    let colours = dev.makeBuffer(length: stride * slots.count, options: .storageModeShared)!
    for (i, c) in slots.enumerated() {
        var rgba: [Float] = [Float(c.0) / 255, Float(c.1) / 255, Float(c.2) / 255, 1]
        memcpy(colours.contents().advanced(by: i * stride), &rgba, 16)
    }
    func want(_ i: Int) -> UInt32 { pack(slots[i].0, slots[i].1, slots[i].2, 255) }
    let clearPixel = pack(255, 0, 255, 255)

    // One encoder that clears, binds `slot`, and covers the target.
    func encode(_ cb: MTLCommandBuffer, slot: Int) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        enc.setFragmentBuffer(colours, offset: slot * stride, index: 0)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
    }

    func judge(_ label: String, _ wantSlot: Int) {
        guard let got = readBack(readPipe, rt, w, h) else { refused(label); return }
        let px = got[(h / 2) * w + (w / 2)]
        let ok = px == want(wantSlot)
        let why: String
        if px == clearPixel {
            why = "the second encoder's draw produced nothing — its bind was dropped"
        } else if px == want(1 - wantSlot) {
            why = "the second encoder rendered the other encoder's slot"
        } else {
            why = "unrecognised"
        }
        report(label, ok,
               ok ? "the binding re-issued on the second encoder is the one that drew"
                  : "got=\(hex(px)) want=\(hex(want(wantSlot))) — \(why)")
    }

    // 1. Two encoders, one command buffer, the *same* bind re-issued. A device
    //    keyed on the table's identity alone sees "unchanged" and skips it.
    var cb = queue.makeCommandBuffer()!
    encode(cb, slot: 0)
    encode(cb, slot: 0)
    cb.commit(); cb.waitUntilCompleted()
    judge("ebl_same_bind_second_encoder", 0)

    // 2. Two encoders, one command buffer, a different bind on the second. The
    //    opposite error: state carried across the boundary rather than dropped.
    cb = queue.makeCommandBuffer()!
    encode(cb, slot: 0)
    encode(cb, slot: 1)
    cb.commit(); cb.waitUntilCompleted()
    judge("ebl_changed_bind_second_encoder", 1)

    // 3. The same bind re-issued from a *different* command buffer. Metal fixes
    //    the order at commit, so the second buffer's clear-and-draw is what the
    //    target holds; a device that carries per-draw state across command
    //    buffers is wrong here in the same way as 1 and harder to notice,
    //    because the two buffers are resolved at different times.
    let first = queue.makeCommandBuffer()!
    encode(first, slot: 1)
    first.commit()
    let second = queue.makeCommandBuffer()!
    encode(second, slot: 1)
    second.commit()
    second.waitUntilCompleted()
    judge("ebl_same_bind_second_command_buffer", 1)
}
