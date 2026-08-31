import Metal
import Foundation
import IOSurface

func blitAfterRenderCase(_ w: Int, _ h: Int) {
    let label = "srt_blit_after_render_\(w)x\(h)"
    guard let pipe = makeRenderPipeline("heavy_fs", .bgra8Unorm_srgb) else {
        report(label, false, "heavy pipeline unavailable for bgra8Unorm_srgb"); return
    }
    let sd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
    sd.usage = [.renderTarget, .shaderRead]
    sd.storageMode = .shared
    guard let source = dev.makeTexture(descriptor: sd) else {
        report(label, false, "makeTexture nil for the shared \(w)x\(h) source"); return
    }
    guard let dst = makeSrgbIOSurfaceTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func encodeFill(_ cb: MTLCommandBuffer, _ colour: SIMD4<Float>, draws: Int) {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = source
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let enc = cb.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        // Repeated whole-target draws, so the GPU is still working when the copy
        // behind them is decoded. One draw would leave the race to scheduling
        // luck and make the case flaky in the direction that reads as a pass.
        for _ in 0..<draws {
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
    }

    // 1. Red, landed. After this the source's bytes are red and nothing else.
    let first = queue.makeCommandBuffer()!
    encodeFill(first, SIMD4<Float>(1, 0, 0, 1), draws: 1)
    first.commit()
    first.waitUntilCompleted()

    // 2. Green, committed and not waited on, then the copy in its own command
    //    buffer. Queue order is what orders them; nothing else has to.
    let second = queue.makeCommandBuffer()!
    encodeFill(second, SIMD4<Float>(0, 1, 0, 1), draws: 256)
    second.commit()

    let copyCb = queue.makeCommandBuffer()!
    let blit = copyCb.makeBlitCommandEncoder()!
    blit.copy(from: source, sourceSlice: 0, sourceLevel: 0,
              to: dst, destinationSlice: 0, destinationLevel: 0,
              sliceCount: 1, levelCount: 1)
    blit.endEncoding()
    copyCb.commit()
    copyCb.waitUntilCompleted()

    guard let got = readBack(readPipe, dst, w, h) else { refused(label); return }
    let green = pack(0, 255, 0, 255)
    let red = pack(255, 0, 0, 255)
    var wrong: [(Int, Int)] = []
    var stale = 0
    for y in 0..<h {
        for x in 0..<w where got[y * w + x] != green {
            wrong.append((x, y))
            if got[y * w + x] == red { stale += 1 }
        }
    }
    let firstBad = wrong.first.map {
        "at=(\($0.0),\($0.1)) got=\(hex(got[$0.1 * w + $0.0]))"
    } ?? ""
    report(label, wrong.isEmpty,
           wrong.isEmpty
             ? "the copy moved what the render ahead of it had written"
             : "wrong=\(wrong.count)/\(w * h) stale_previous_frame=\(stale) "
               + "want=\(hex(green)) \(firstBad)" + badMap(wrong, w, h))
}

// The same copy, run the way a compositor runs it: many frames in flight.
//
// The single-shot case above reaches the rail and does not fail, because one
// render is not enough to keep the GPU busy past the point where the copy
// behind it is serviced. A compositor never has one frame outstanding — it
// renders a layer, copies it out, and starts the next without waiting for
// either, so a copy is decoded while earlier frames are still executing and the
// queue is never drained.
//
// Each frame gets its own colour and its own destination, so a stale read is not
// merely "wrong" — it is identifiably *an earlier frame*, which is what the
// report names. Nothing is waited on until every frame has been committed.
func blitPipelinedCase(_ w: Int, _ h: Int, frames: Int) {
    let label = "srt_blit_pipelined_\(w)x\(h)_x\(frames)"
    // Host-pointer import and copying configurations reach different counters
    // here. The exact pixel relation is portable between them, but no single
    // route counter is, so this case makes no route claim.
    guard let pipe = makeRenderPipeline("heavy_fs", .bgra8Unorm_srgb) else {
        report(label, false, "heavy pipeline unavailable for bgra8Unorm_srgb"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // One colour per frame, all full-intensity so the sRGB encode round-trips
    // exactly, and all distinguishable from one another.
    let palette: [(SIMD4<Float>, UInt32)] = [
        (SIMD4(1, 0, 0, 1), pack(255, 0, 0, 255)),
        (SIMD4(0, 1, 0, 1), pack(0, 255, 0, 255)),
        (SIMD4(0, 0, 1, 1), pack(0, 0, 255, 255)),
        (SIMD4(1, 1, 0, 1), pack(255, 255, 0, 255)),
        (SIMD4(1, 0, 1, 1), pack(255, 0, 255, 255)),
        (SIMD4(0, 1, 1, 1), pack(0, 255, 255, 255)),
        (SIMD4(1, 1, 1, 1), pack(255, 255, 255, 255)),
        (SIMD4(0, 0, 0, 1), pack(0, 0, 0, 255)),
    ]

    var sources: [MTLTexture] = []
    var dests: [MTLTexture] = []
    for i in 0..<frames {
        let sd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
        sd.usage = [.renderTarget, .shaderRead]
        sd.storageMode = .shared
        guard let src = dev.makeTexture(descriptor: sd) else {
            report(label, false, "makeTexture nil for frame \(i)'s source"); return
        }
        guard let dst = makeSrgbIOSurfaceTarget(w, h, label) else { return }
        sources.append(src)
        dests.append(dst)
    }

    var last: MTLCommandBuffer?
    for i in 0..<frames {
        let (colour, _) = palette[i % palette.count]
        let render = queue.makeCommandBuffer()!
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = sources[i]
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let enc = render.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        for _ in 0..<96 {
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
        render.commit()

        let copyCb = queue.makeCommandBuffer()!
        let blit = copyCb.makeBlitCommandEncoder()!
        blit.copy(from: sources[i], sourceSlice: 0, sourceLevel: 0,
                  to: dests[i], destinationSlice: 0, destinationLevel: 0,
                  sliceCount: 1, levelCount: 1)
        blit.endEncoding()
        copyCb.commit()
        last = copyCb
    }
    last?.waitUntilCompleted()

    var badFrames: [Int] = []
    var detail = ""
    for i in 0..<frames {
        guard let got = readBack(readPipe, dests[i], w, h) else { refused(label); return }
        let want = palette[i % palette.count].1
        var wrong = 0
        var sawOtherFrame = -1
        for t in 0..<(w * h) where got[t] != want {
            wrong += 1
            if sawOtherFrame < 0 {
                for (j, p) in palette.enumerated() where p.1 == got[t] { sawOtherFrame = j }
            }
        }
        if wrong > 0 {
            badFrames.append(i)
            if detail.isEmpty {
                detail = "frame=\(i) wrong=\(wrong)/\(w * h) want=\(hex(want)) "
                    + (sawOtherFrame >= 0
                        ? "got=another frame's colour (palette \(sawOtherFrame))"
                        : "got=\(hex(got[0]))")
            }
        }
    }
    report(label, badFrames.isEmpty,
           badFrames.isEmpty
             ? "every frame's copy moved that frame's own pixels"
             : "stale_frames=\(badFrames.count)/\(frames) \(detail)")
}

// A buffer-backed source, which is a linear guest allocation by construction.
//
// A texture made from an `MTLBuffer` names its own base and row stride, so
// unlike a plain `.shared` texture — whose layout the driver picks — it is
// unambiguously the linear form. Both are worth having: the pair above is what
// a compositor emits, and this is the shape that leaves the guest no room to
// choose something else.
//
// # This is a second defect, and it is not the unordered read
//
// The case fails with every texel zero and `stale_previous_frame=0`, which looks
// at first like the unordered read above — zero is also what pre-Store bytes
// look like on a freshly allocated source. It is not. It fails identically with
// the ordering in place, and the first pass here is waited on, so a racing copy
// would have found red rather than nothing.
//
// The device names it on the fail channel:
//
//     rt_resolve reason=rt_wrong_type object_type=texture_view
//
// A texture made from an `MTLBuffer` arrives as a texture *view*, and the
// render-target resolver does not accept that object type as a colour
// attachment. Every draw into one is dropped, so the guest reads back the
// allocation untouched. Metal renders into it happily, which is why this case
// passes natively and fails here.
//
// The refusal is correct behaviour for an unimplemented case — it costs the
// guest one command and says so by name, rather than guessing. The case stays
// red until the resolver accepts the type.
func makeLinearTarget(_ w: Int, _ h: Int, _ label: String,
                      _ format: MTLPixelFormat) -> MTLTexture? {
    let align = max(1, dev.minimumLinearTextureAlignment(for: format))
    let bpr = ((w * 4) + align - 1) / align * align
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(label, false, "makeBuffer nil for a linear \(w)x\(h) target"); return nil
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: format, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .shared
    guard let tex = buf.makeTexture(descriptor: td, offset: 0, bytesPerRow: bpr) else {
        report(label, false, "makeTexture nil for a buffer-backed \(w)x\(h) render target")
        return nil
    }
    return tex
}

func blitBufferBackedCase(_ w: Int, _ h: Int) {
    let label = "srt_blit_buffer_backed_\(w)x\(h)"
    guard let pipe = makeRenderPipeline("solid_fs", .bgra8Unorm) else {
        report(label, false, "solid pipeline unavailable for bgra8Unorm"); return
    }
    guard let source = makeLinearTarget(w, h, label, .bgra8Unorm) else { return }
    guard let dst = makeIOSurfaceTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func encodeFill(_ cb: MTLCommandBuffer, _ colour: SIMD4<Float>, draws: Int) {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = source
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let enc = cb.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        for _ in 0..<draws {
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
    }

    let first = queue.makeCommandBuffer()!
    encodeFill(first, SIMD4<Float>(1, 0, 0, 1), draws: 1)
    first.commit()
    first.waitUntilCompleted()

    let second = queue.makeCommandBuffer()!
    encodeFill(second, SIMD4<Float>(0, 1, 0, 1), draws: 64)
    second.commit()

    let copyCb = queue.makeCommandBuffer()!
    let blit = copyCb.makeBlitCommandEncoder()!
    blit.copy(from: source, sourceSlice: 0, sourceLevel: 0,
              to: dst, destinationSlice: 0, destinationLevel: 0,
              sliceCount: 1, levelCount: 1)
    blit.endEncoding()
    copyCb.commit()
    copyCb.waitUntilCompleted()

    guard let got = readBack(readPipe, dst, w, h) else { refused(label); return }
    let green = pack(0, 255, 0, 255)
    let red = pack(255, 0, 0, 255)
    var wrong: [(Int, Int)] = []
    var stale = 0
    var zero = 0
    for y in 0..<h {
        for x in 0..<w where got[y * w + x] != green {
            wrong.append((x, y))
            if got[y * w + x] == red { stale += 1 }
            if got[y * w + x] == 0 { zero += 1 }
        }
    }
    let firstBad = wrong.first.map {
        "at=(\($0.0),\($0.1)) got=\(hex(got[$0.1 * w + $0.0]))"
    } ?? ""
    report(label, wrong.isEmpty,
           wrong.isEmpty
             ? "the copy out of a buffer-backed source moved what the render had written"
             : "wrong=\(wrong.count)/\(w * h) stale_previous_frame=\(stale) "
               + "never_written=\(zero) want=\(hex(green)) \(firstBad)" + badMap(wrong, w, h))
}

// The same whole-plane copy with an IOSurface-backed *source*, which is the
// shape a compositor actually emits: it renders a layer into a surface it
// shares, then copies that surface into another one.
//
// # Why the source's backing is the whole case
//
// The two cases above give the copy a plain `.shared` source, and that is a
// different rail inside this device: a shared texture is a linear guest
// allocation, and the only account of its content the device had was a
// writeback debt — "the GPU holds newer bytes than the guest pages do". Under a
// host-pointer import there is never such a debt, because the render Store
// wrote into the guest pages themselves. So a copy out of a surface the guest
// had just rendered into found no GPU source, refused the GPU arm, and staged
// every row through the host CPU — on a driven fullscreen Maps boot, on every
// record of the boot.
//
// An IOSurface-backed texture is named by its surface identity instead, which
// needs no debt, so the copy `MTLBlitCommandEncoder` puts on the GPU queue runs
// on the GPU queue. `blit_t5_plane_device` is that account being the one that
// answered; a run where it never moves did not put this case on the rail it was
// written for, whatever it reported.
//
// Frames are pipelined for the same reason `blitPipelinedCase` pipelines them:
// nothing is waited on until every frame is committed, so each copy is decoded
// while the render feeding it is still executing. A device that names the wrong
// resident does not return garbage here — it returns *another frame's colour*,
// which is what the report says when it fails.
func blitIOSurfaceSourceCase(_ w: Int, _ h: Int, frames: Int) {
    let label = "srt_blit_iosurface_source_\(w)x\(h)_x\(frames)"
    claims(label, "blit_t5_plane_device", "sl_gpu_landed")
    guard let pipe = makeRenderPipeline("heavy_fs", .bgra8Unorm) else {
        report(label, false, "heavy pipeline unavailable for bgra8Unorm"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // Full-intensity colours only, so nothing here depends on a rounding rule,
    // and all eight distinguishable from one another.
    let palette: [(SIMD4<Float>, UInt32)] = [
        (SIMD4(1, 0, 0, 1), pack(255, 0, 0, 255)),
        (SIMD4(0, 1, 0, 1), pack(0, 255, 0, 255)),
        (SIMD4(0, 0, 1, 1), pack(0, 0, 255, 255)),
        (SIMD4(1, 1, 0, 1), pack(255, 255, 0, 255)),
        (SIMD4(1, 0, 1, 1), pack(255, 0, 255, 255)),
        (SIMD4(0, 1, 1, 1), pack(0, 255, 255, 255)),
        (SIMD4(1, 1, 1, 1), pack(255, 255, 255, 255)),
        (SIMD4(0, 0, 0, 1), pack(0, 0, 0, 255)),
    ]

    var sources: [MTLTexture] = []
    var dests: [MTLTexture] = []
    for _ in 0..<frames {
        guard let src = makeIOSurfaceTarget(w, h, label) else { return }
        guard let dst = makeIOSurfaceTarget(w, h, label) else { return }
        sources.append(src)
        dests.append(dst)
    }

    var last: MTLCommandBuffer?
    for i in 0..<frames {
        let (colour, _) = palette[i % palette.count]
        let render = queue.makeCommandBuffer()!
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = sources[i]
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let enc = render.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        for _ in 0..<96 {
            enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        enc.endEncoding()
        render.commit()

        let copyCb = queue.makeCommandBuffer()!
        let blit = copyCb.makeBlitCommandEncoder()!
        blit.copy(from: sources[i], sourceSlice: 0, sourceLevel: 0,
                  to: dests[i], destinationSlice: 0, destinationLevel: 0,
                  sliceCount: 1, levelCount: 1)
        blit.endEncoding()
        copyCb.commit()
        last = copyCb
    }
    last?.waitUntilCompleted()

    var badFrames: [Int] = []
    var detail = ""
    for i in 0..<frames {
        guard let got = readBack(readPipe, dests[i], w, h) else { refused(label); return }
        let want = palette[i % palette.count].1
        var wrong = 0
        var sawOtherFrame = -1
        for t in 0..<(w * h) where got[t] != want {
            wrong += 1
            if sawOtherFrame < 0 {
                for (j, p) in palette.enumerated() where p.1 == got[t] { sawOtherFrame = j }
            }
        }
        if wrong > 0 {
            badFrames.append(i)
            if detail.isEmpty {
                detail = "frame=\(i) wrong=\(wrong)/\(w * h) want=\(hex(want)) "
                    + (sawOtherFrame >= 0
                        ? "got=another frame's colour (palette \(sawOtherFrame))"
                        : "got=\(hex(got[0]))")
            }
        }
    }
    report(label, badFrames.isEmpty,
           badFrames.isEmpty
             ? "every surface-to-surface copy moved its own frame's pixels"
             : "stale_frames=\(badFrames.count)/\(frames) \(detail)")
}
