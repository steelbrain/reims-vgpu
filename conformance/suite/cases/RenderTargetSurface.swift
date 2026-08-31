import Metal
import Foundation
import IOSurface

// BGRA in memory, so the packed comparison value has to swap to match
// `read_texels`, which reports the sampled RGBA.
func iosurfaceCases(_ w: Int, _ h: Int) {
    let dims = "\(w)x\(h)"
    let names = (draw: "iosrt_draw_\(dims)",
                 load: "iosrt_load_preserves_\(dims)",
                 scissor: "iosrt_scissor_keeps_rest_\(dims)",
                 glyph: "iosrt_glyph_blend_\(dims)")
    guard let solid = makeRenderPipeline("solid_fs", .bgra8Unorm) else {
        for n in [names.draw, names.load, names.scissor, names.glyph] {
            report(n, false, "solid pipeline unavailable for bgra8Unorm")
        }
        return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func pass(_ rt: MTLTexture, _ pipe: MTLRenderPipelineState,
              _ colour: SIMD4<Float>, load: Bool, scissor: MTLScissorRect?,
              atlas: MTLTexture?) {
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = rt
        d.colorAttachments[0].loadAction = load ? .load : .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: d)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        if let atlas { enc.setFragmentTexture(atlas, index: 0) }
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        if let s = scissor { enc.setScissorRect(s) }
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // 1. A draw into an IOSurface-backed target.
    guard let rt = makeIOSurfaceTarget(w, h, names.draw) else {
        for n in [names.load, names.scissor, names.glyph] { skipDependent(n, names.draw) }
        return
    }
    pass(rt, solid, SIMD4<Float>(0, 1, 0, 1), load: false, scissor: nil, atlas: nil)
    guard let drawn = readBack(readPipe, rt, w, h) else {
        refused(names.draw)
        for n in [names.load, names.scissor, names.glyph] { skipDependent(n, names.draw) }
        return
    }
    var bad: [(Int, Int)] = []
    for y in 0..<h { for x in 0..<w where drawn[y * w + x] != greenTexel { bad.append((x, y)) } }
    report(names.draw, bad.isEmpty,
           bad.isEmpty ? "the whole IOSurface-backed target is the drawn colour"
                       : "wrong=\(bad.count)/\(w * h) "
                         + "first=\(hex(drawn[bad[0].1 * w + bad[0].0])) want=\(hex(greenTexel))"
                         + badMap(bad, w, h))

    // 2. A second pass over the bottom half must keep the first pass's top half.
    guard let rt2 = makeIOSurfaceTarget(w, h, names.load) else {
        for n in [names.scissor, names.glyph] { skipDependent(n, names.load) }
        return
    }
    let half = h / 2
    pass(rt2, solid, SIMD4<Float>(0, 1, 0, 1), load: false,
         scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half), atlas: nil)
    pass(rt2, solid, SIMD4<Float>(0, 0, 1, 1), load: true,
         scissor: MTLScissorRect(x: 0, y: half, width: w, height: h - half), atlas: nil)
    if let got = readBack(readPipe, rt2, w, h) {
        var top: [(Int, Int)] = []
        var bot: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let have = got[y * w + x]
                if y < half { if have != greenTexel { top.append((x, y)) } }
                else if have != blueTexel { bot.append((x, y)) }
            }
        }
        let ok = top.isEmpty && bot.isEmpty
        report(names.load, ok,
               ok ? "the second pass added to what the first pass left"
                  : "first_pass_lost=\(top.count) second_pass_wrong=\(bot.count)"
                    + badMap(top, w, h))
    } else { refused(names.load) }

    // 3. A scissored write into the middle of a loaded IOSurface target.
    guard let rt3 = makeIOSurfaceTarget(w, h, names.scissor) else {
        skipDependent(names.glyph, names.scissor); return
    }
    pass(rt3, solid, SIMD4<Float>(1, 0, 1, 1), load: false, scissor: nil, atlas: nil)
    let rx = w / 4, ry = h / 4, rw = max(1, w / 2), rh = max(1, h / 2)
    pass(rt3, solid, SIMD4<Float>(0, 1, 0, 1), load: true,
         scissor: MTLScissorRect(x: rx, y: ry, width: rw, height: rh), atlas: nil)
    if let got = readBack(readPipe, rt3, w, h) {
        var inside: [(Int, Int)] = []
        var outside: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let within = x >= rx && x < rx + rw && y >= ry && y < ry + rh
                let have = got[y * w + x]
                if within { if have != greenTexel { inside.append((x, y)) } }
                else if have != magentaTexel { outside.append((x, y)) }
            }
        }
        let ok = inside.isEmpty && outside.isEmpty
        report(names.scissor, ok,
               ok ? "the scissored write landed on exactly its own rows"
                  : "rect_wrong=\(inside.count) outside_clobbered=\(outside.count) "
                    + "rect=(\(rx),\(ry) \(rw)x\(rh))" + badMap(outside, w, h))
    } else { refused(names.scissor) }

    // 4. Coverage type blended into the IOSurface layer, twice, the second
    //    loading the first. This is the draw Maps' type layer actually makes.
    guard let glyph = makeGlyphPipeline(.bgra8Unorm) else {
        report(names.glyph, false, "glyph pipeline unavailable for bgra8Unorm"); return
    }
    let ad = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm, width: w, height: h, mipmapped: false)
    ad.usage = [.shaderRead]
    ad.storageMode = .shared
    guard let atlas = dev.makeTexture(descriptor: ad) else {
        report(names.glyph, false, "makeTexture nil for the coverage atlas"); return
    }
    var cov = [UInt8](repeating: 0, count: w * h)
    for y in 0..<h { for x in 0..<w { cov[y * w + x] = ((x / 3) + (y / 3)) % 2 == 0 ? 255 : 0 } }
    cov.withUnsafeBytes { raw in
        atlas.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                      withBytes: raw.baseAddress!, bytesPerRow: w)
    }
    guard let rt4 = makeIOSurfaceTarget(w, h, names.glyph) else { return }
    pass(rt4, glyph, SIMD4<Float>(0, 1, 0, 1), load: false, scissor: nil, atlas: atlas)
    pass(rt4, glyph, SIMD4<Float>(0, 0, 1, 1), load: true,
         scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half), atlas: atlas)
    if let got = readBack(readPipe, rt4, w, h) {
        var wrong: [(Int, Int)] = []
        var first = ""
        for y in 0..<h {
            for x in 0..<w {
                let on = cov[y * w + x] == 255
                // A texel the type does not cover keeps this section's clear,
                // which is magenta. Expecting magenta rather than black is also
                // what makes the check say something: black is what a target
                // nobody wrote looks like, so a case that expected it would pass
                // on a layer that was never rendered into at all.
                let want = !on ? magentaTexel
                    : (y < half ? pack(0, 0, 255, 255) : pack(0, 255, 0, 255))
                let have = got[y * w + x]
                if have != want {
                    wrong.append((x, y))
                    if first.isEmpty { first = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))" }
                }
            }
        }
        report(names.glyph, wrong.isEmpty,
               wrong.isEmpty
                 ? "coverage type blended into an IOSurface layer, and the second pass kept the first"
                 : "wrong=\(wrong.count)/\(w * h) \(first)" + badMap(wrong, w, h))
    } else { refused(names.glyph) }
}

// K. The CPU writes a layer the GPU already rendered into, and the GPU samples it.
//
// Section I and J both stop one step short of the shape a compositor actually
// runs. `cpu_write_after_render` writes the layer from the CPU and then reads
// it back with a compute kernel; `srt_sample_after_render` samples the layer but
// only ever sees texels the GPU itself drew. Neither one asks the question a
// text layer asks: after the CPU has written bytes into an allocation the GPU
// has already rendered into, does a *sampled bind* in a later render pass see
// them?
//
// The two reads are not interchangeable. A compute kernel reading a texture and
// a fragment shader sampling one are different binds, and a device that keeps a
// device-local image for an allocation it renders into may serve one from guest
// pages and the other from that image. A guest CPU store into its own RAM is
// invisible to the host -- no page fault, no command, nothing to observe -- so
// an image the device believes is authoritative stays stale with nothing
// anywhere reporting a loss.
//
// That is the whole failure mode this case exists for, and it is what a layer
// losing its type looks like: the GPU-drawn geometry in the layer is correct
// because the device drew it, and only the CPU-written half is missing.
func cpuWriteThenSampleCase(_ w: Int, _ h: Int, iosurface: Bool) {
    let rail = iosurface ? "iosrt" : "srt"
    let label = "\(rail)_cpu_write_then_sample_\(w)x\(h)"
    guard let pipe = iosurface ? makeRenderPipeline("solid_fs", .bgra8Unorm)
                               : solidPipeline else {
        report(label, false, "solid pipeline unavailable"); return
    }
    guard let texPipe = texPipeline else {
        report(label, false, "texture pipeline unavailable"); return
    }
    guard let layer = iosurface ? makeIOSurfaceTarget(w, h, label)
                                : sharedRenderTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // 1. The GPU renders the whole layer green. After this the device has an
    //    image for the allocation and every byte in it is the device's own.
    sharedTargetPass(layer, pipe, verts, SIMD4<Float>(0, 1, 0, 1),
                     load: false, clear: magenta, scissor: nil)

    // 2. The CPU writes the bottom half opaque red, the way CoreText rasterizes
    //    glyphs into a layer's bytes. `.shared` is BGRA in memory on the
    //    IOSurface rail and RGBA on the other, so the channel order follows the
    //    format rather than the case.
    let half = max(1, h / 2)
    let rows = h - half
    let bpr = w * 4
    var cpu = [UInt8](repeating: 0, count: bpr * rows)
    for i in 0..<(w * rows) {
        let o = i * 4
        if iosurface { cpu[o] = 0; cpu[o + 1] = 0; cpu[o + 2] = 255 }
        else { cpu[o] = 255; cpu[o + 1] = 0; cpu[o + 2] = 0 }
        cpu[o + 3] = 255
    }
    cpu.withUnsafeBytes { raw in
        layer.replace(region: MTLRegionMake2D(0, half, w, rows), mipmapLevel: 0,
                      withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }

    // 3. A later render pass samples the layer into a private destination --
    //    the compositor's read, not a compute read-back of the layer itself.
    let dd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    dd.usage = [.renderTarget, .shaderRead]
    dd.storageMode = .private
    guard let dst = dev.makeTexture(descriptor: dd) else {
        report(label, false, "makeTexture nil for the private destination"); return
    }
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = dst
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = magenta
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(texPipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentTexture(layer, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    guard let got = readBack(readPipe, dst, w, h) else { refused(label); return }
    let redTexel = pack(255, 0, 0, 255)
    var gpuBad: [(Int, Int)] = []
    var cpuBad: [(Int, Int)] = []
    var firstCpu = ""
    for y in 0..<h {
        for x in 0..<w {
            let have = got[y * w + x]
            if y < half {
                if have != greenTexel { gpuBad.append((x, y)) }
            } else if have != redTexel {
                cpuBad.append((x, y))
                if firstCpu.isEmpty {
                    firstCpu = "at=(\(x),\(y)) want=\(hex(redTexel)) got=\(hex(have))"
                }
            }
        }
    }
    let ok = gpuBad.isEmpty && cpuBad.isEmpty
    report(label, ok,
           ok ? "a sampled bind saw the texels the CPU wrote after the GPU rendered"
              : "cpu_written_unseen=\(cpuBad.count)/\(w * rows) "
                + "gpu_drawn_wrong=\(gpuBad.count)/\(w * half) \(firstCpu)"
                + badMap(cpuBad, w, h))
}
