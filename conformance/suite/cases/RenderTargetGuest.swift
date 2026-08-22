import Metal
import Foundation
import IOSurface

// One pass over `rt`. `scissor` nil means the whole target.
func sharedTargetPass(_ rt: MTLTexture, _ pipe: MTLRenderPipelineState,
                      _ verts: MTLBuffer, _ colour: SIMD4<Float>,
                      load: Bool, clear: MTLClearColor, scissor: MTLScissorRect?) {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = load ? .load : .clear
    pass.colorAttachments[0].clearColor = clear
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    var c = colour
    enc.setFragmentBytes(&c, length: 16, index: 0)
    if let s = scissor { enc.setScissorRect(s) }
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

let magenta = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)

let magentaTexel = pack(255, 0, 255, 255)

let greenTexel = pack(0, 255, 0, 255)

let blueTexel = pack(0, 0, 255, 255)

func sharedTargetCases(_ w: Int, _ h: Int) {
    let dims = "\(w)x\(h)"
    let names = (draw: "srt_draw_\(dims)",
                 load: "srt_load_preserves_\(dims)",
                 scissor: "srt_scissor_keeps_rest_\(dims)",
                 sample: "srt_sample_after_render_\(dims)")

    guard let pipe = solidPipeline else {
        for n in [names.draw, names.load, names.scissor, names.sample] {
            report(n, false, "solid pipeline unavailable")
        }
        return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // 1. Does a draw into guest-owned pages land at all.
    guard let rt = sharedRenderTarget(w, h, names.draw) else {
        for n in [names.load, names.scissor, names.sample] { skipDependent(n, names.draw) }
        return
    }
    sharedTargetPass(rt, pipe, verts, SIMD4<Float>(0, 1, 0, 1),
                     load: false, clear: magenta, scissor: nil)
    guard let drawn = readBack(readPipe, rt, w, h) else {
        refused(names.draw)
        for n in [names.load, names.scissor, names.sample] { skipDependent(n, names.draw) }
        return
    }
    var drawBad: [(Int, Int)] = []
    for y in 0..<h { for x in 0..<w where drawn[y * w + x] != greenTexel {
        drawBad.append((x, y)) } }
    report(names.draw, drawBad.isEmpty,
           drawBad.isEmpty ? "the whole guest-backed target is the drawn colour"
                           : "wrong=\(drawBad.count)/\(w * h) "
                             + "first=\(hex(drawn[drawBad[0].1 * w + drawBad[0].0])) "
                             + "want=\(hex(greenTexel))" + badMap(drawBad, w, h))

    // 2. A second pass with `.load` over the bottom half must leave the top
    //    half exactly as the first pass left it. A device that seeds `.load`
    //    from a copy taken before the first pass loses the top half here.
    guard let rt2 = sharedRenderTarget(w, h, names.load) else {
        for n in [names.scissor, names.sample] { skipDependent(n, names.load) }
        return
    }
    let half = h / 2
    sharedTargetPass(rt2, pipe, verts, SIMD4<Float>(0, 1, 0, 1), load: false,
                     clear: magenta, scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half))
    sharedTargetPass(rt2, pipe, verts, SIMD4<Float>(0, 0, 1, 1), load: true,
                     clear: magenta,
                     scissor: MTLScissorRect(x: 0, y: half, width: w, height: h - half))
    if let got = readBack(readPipe, rt2, w, h) {
        var topBad: [(Int, Int)] = []
        var botBad: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let have = got[y * w + x]
                if y < half {
                    if have != greenTexel { topBad.append((x, y)) }
                } else if have != blueTexel { botBad.append((x, y)) }
            }
        }
        let ok = topBad.isEmpty && botBad.isEmpty
        report(names.load, ok,
               ok ? "the second pass added to what the first pass left"
                  : "first_pass_lost=\(topBad.count) second_pass_wrong=\(botBad.count)"
                    + badMap(topBad, w, h))
    } else {
        refused(names.load)
    }

    // 3. A scissored write into the middle of a loaded target. Only that
    //    rectangle may change; a stride the guest never agreed to shows up as
    //    the green landing on the wrong rows, which the untouched-count says.
    guard let rt3 = sharedRenderTarget(w, h, names.scissor) else {
        skipDependent(names.sample, names.scissor); return
    }
    sharedTargetPass(rt3, pipe, verts, SIMD4<Float>(1, 0, 1, 1),
                     load: false, clear: magenta, scissor: nil)
    let rx = w / 4, ry = h / 4
    let rw = max(1, w / 2), rh = max(1, h / 2)
    sharedTargetPass(rt3, pipe, verts, SIMD4<Float>(0, 1, 0, 1), load: true, clear: magenta,
                     scissor: MTLScissorRect(x: rx, y: ry, width: rw, height: rh))
    if let got = readBack(readPipe, rt3, w, h) {
        var inBad: [(Int, Int)] = []
        var outBad: [(Int, Int)] = []
        for y in 0..<h {
            for x in 0..<w {
                let inside = x >= rx && x < rx + rw && y >= ry && y < ry + rh
                let have = got[y * w + x]
                if inside {
                    if have != greenTexel { inBad.append((x, y)) }
                } else if have != magentaTexel { outBad.append((x, y)) }
            }
        }
        let ok = inBad.isEmpty && outBad.isEmpty
        report(names.scissor, ok,
               ok ? "the scissored write landed on exactly its own rows"
                  : "rect_wrong=\(inBad.count) outside_clobbered=\(outBad.count) "
                    + "rect=(\(rx),\(ry) \(rw)x\(rh))" + badMap(outBad, w, h))
    } else {
        refused(names.scissor)
    }

    // 4. The crossover: the same allocation, rendered into by one pass and
    //    sampled by the next. A device that keeps a target and a sampled view
    //    of one allocation as two images has to make the render visible to the
    //    read, and this is the case that says whether it did.
    guard let texPipe = texPipeline else {
        report(names.sample, false, "texture pipeline unavailable"); return
    }
    guard let src = sharedRenderTarget(w, h, names.sample) else { return }
    sharedTargetPass(src, pipe, verts, SIMD4<Float>(0, 1, 0, 1),
                     load: false, clear: magenta, scissor: nil)
    let dd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    dd.usage = [.renderTarget, .shaderRead]
    dd.storageMode = .private
    guard let dst = dev.makeTexture(descriptor: dd) else {
        report(names.sample, false, "makeTexture nil for the private destination"); return
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
    enc.setFragmentTexture(src, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if let got = readBack(readPipe, dst, w, h) {
        var bad: [(Int, Int)] = []
        for y in 0..<h { for x in 0..<w where got[y * w + x] != greenTexel {
            bad.append((x, y)) } }
        report(names.sample, bad.isEmpty,
               bad.isEmpty ? "a later pass sampled what the earlier pass rendered"
                           : "wrong=\(bad.count)/\(w * h) "
                             + "first=\(hex(got[bad[0].1 * w + bad[0].0])) "
                             + "want=\(hex(greenTexel))" + badMap(bad, w, h))
    } else {
        refused(names.sample)
    }
}

// A guest-backed target whose content the CPU wrote before any GPU pass.
//
// This is the case the four above cannot reach. `srt_load_preserves` starts its
// first pass with `.clear`, so the target's prior content is this device's own
// and a rail that discards it on the way in still passes. A compositor's layer
// is the other order: CoreText rasterizes glyphs into the layer's bytes, and
// only then does the GPU composite over it with `.load`.
//
// A Vulkan image bound over memory that already holds data does not inherit
// that data -- the contents are undefined until something writes them -- so a
// device that aliases guest pages has to seed the first `.load` from those
// pages explicitly and may skip the seed on later ones. Both sides of that
// boundary are here: round 0 crosses the first-load seed, rounds 1 and 2 cross
// whatever the device does once it believes the image is authoritative.
//
// Everything CPU-written that a round did not draw over must survive every
// round. Losing it is a layer that renders its GPU geometry and drops the type
// the CPU put there, with no refusal anywhere.
func sharedTargetCpuSeedCase(_ w: Int, _ h: Int) {
    let label = "srt_cpu_seed_then_load_\(w)x\(h)"
    guard let pipe = solidPipeline else { report(label, false, "solid pipeline unavailable"); return }
    guard let rt = sharedRenderTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    // 1. The CPU writes every texel, before the GPU has touched the target.
    let bpr = w * 4
    var cpu = [UInt8](repeating: 0, count: bpr * h)
    for y in 0..<h {
        for x in 0..<w {
            let o = y * bpr + x * 4
            cpu[o] = UInt8((x &* 7 &+ y &* 13 &+ 1) & 0xFF)
            cpu[o + 1] = UInt8((x &+ y &* 5) & 0xFF)
            cpu[o + 2] = 0x40
            cpu[o + 3] = 0xFF
        }
    }
    cpu.withUnsafeBytes { raw in
        rt.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                   withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }

    // 2. Three `.load` passes, each over one horizontal band, leaving the last
    //    band CPU-only for the whole case.
    let band = max(1, h / 4)
    let drawn: [UInt32] = [greenTexel, blueTexel, pack(255, 0, 255, 255)]
    let colours: [SIMD4<Float>] = [SIMD4(0, 1, 0, 1), SIMD4(0, 0, 1, 1), SIMD4(1, 0, 1, 1)]
    for round in 0..<3 {
        sharedTargetPass(rt, pipe, verts, colours[round], load: true,
                         clear: magenta,
                         scissor: MTLScissorRect(x: 0, y: round * band,
                                                 width: w, height: band))
    }

    guard let got = readBack(readPipe, rt, w, h) else { refused(label); return }
    var gpuBad: [(Int, Int)] = []
    var cpuBad: [(Int, Int)] = []
    var firstCpu = ""
    for y in 0..<h {
        for x in 0..<w {
            let have = got[y * w + x]
            let round = y / band
            if round < 3 {
                if have != drawn[round] { gpuBad.append((x, y)) }
            } else {
                let o = y * bpr + x * 4
                let want = pack(cpu[o], cpu[o + 1], cpu[o + 2], cpu[o + 3])
                if have != want {
                    cpuBad.append((x, y))
                    if firstCpu.isEmpty {
                        firstCpu = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                    }
                }
            }
        }
    }
    let ok = gpuBad.isEmpty && cpuBad.isEmpty
    report(label, ok,
           ok ? "three loaded passes kept every texel the CPU wrote first"
              : "cpu_written_lost=\(cpuBad.count) gpu_bands_wrong=\(gpuBad.count) \(firstCpu) "
                + badMap(cpuBad, w, h))
}

// Type composited into a guest-backed layer.
//
// Everything in section I draws flat colour with blending off, and a driven
// Maps boot shows that is not where its type layer goes: it binds `R8Unorm`
// coverage atlases -- 128x128 and a scatter of 6x15, 10x1, 5x11 glyph
// bitmaps -- and blends them into a layer. Those atlases are sampled
// identically whether or not the device imports render targets, so whatever is
// lost is lost on the way *into* the layer, and this is the case shaped like
// that draw: a single-channel source, a premultiplied blend, and a
// guest-backed destination that accumulates across passes.
//
// The second pass is the half that a flat-colour case cannot stand in for. A
// text layer is built by blending over what is already there, so a rail that
// loses the destination on the way into a blended `.load` pass loses the type
// drawn before it while every solid draw in this file still passes.
func makeGlyphPipeline(_ fmt: MTLPixelFormat) -> MTLRenderPipelineState? {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "quad_vs")
    d.fragmentFunction = library.makeFunction(name: "glyph_fs")
    let a = d.colorAttachments[0]!
    a.pixelFormat = fmt
    a.isBlendingEnabled = true
    a.rgbBlendOperation = .add
    a.alphaBlendOperation = .add
    a.sourceRGBBlendFactor = .one
    a.sourceAlphaBlendFactor = .one
    a.destinationRGBBlendFactor = .oneMinusSourceAlpha
    a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
    return try? dev.makeRenderPipelineState(descriptor: d)
}

let glyphPipeline = makeGlyphPipeline(.rgba8Unorm)

func sharedTargetGlyphCase(_ w: Int, _ h: Int) {
    let label = "srt_glyph_blend_\(w)x\(h)"
    guard let pipe = glyphPipeline else {
        report(label, false, "glyph pipeline unavailable"); return
    }
    // A single-channel coverage atlas in guest-visible storage, binary so the
    // expected blend has no rounding to argue about and a shear shows up as a
    // moved checker rather than as a slightly wrong colour.
    let ad = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .r8Unorm, width: w, height: h, mipmapped: false)
    ad.usage = [.shaderRead]
    ad.storageMode = .shared
    guard let atlas = dev.makeTexture(descriptor: ad) else {
        report(label, false, "makeTexture nil for the coverage atlas"); return
    }
    var cov = [UInt8](repeating: 0, count: w * h)
    for y in 0..<h {
        for x in 0..<w { cov[y * w + x] = ((x / 3) + (y / 3)) % 2 == 0 ? 255 : 0 }
    }
    cov.withUnsafeBytes { raw in
        atlas.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0,
                      withBytes: raw.baseAddress!, bytesPerRow: w)
    }

    guard let rt = sharedRenderTarget(w, h, label) else { return }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func blend(_ colour: SIMD4<Float>, load: Bool, scissor: MTLScissorRect?) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = load ? .load : .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        enc.setFragmentTexture(atlas, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        if let s = scissor { enc.setScissorRect(s) }
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // 1. Green type over an opaque black layer.
    blend(SIMD4<Float>(0, 1, 0, 1), load: false, scissor: nil)
    // 2. Blue type over the top half, loading what pass 1 left.
    let half = h / 2
    blend(SIMD4<Float>(0, 0, 1, 1), load: true,
          scissor: MTLScissorRect(x: 0, y: 0, width: w, height: half))

    guard let got = readBack(readPipe, rt, w, h) else { refused(label); return }
    let black = pack(0, 0, 0, 255)
    var bad: [(Int, Int)] = []
    var first = ""
    for y in 0..<h {
        for x in 0..<w {
            let on = cov[y * w + x] == 255
            let want: UInt32
            if !on {
                want = black
            } else if y < half {
                want = pack(0, 0, 255, 255)
            } else {
                want = pack(0, 255, 0, 255)
            }
            let have = got[y * w + x]
            if have != want {
                bad.append((x, y))
                if first.isEmpty { first = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))" }
            }
        }
    }
    report(label, bad.isEmpty,
           bad.isEmpty ? "coverage type blended into a guest-backed layer, and the second pass kept the first"
                       : "wrong=\(bad.count)/\(w * h) \(first)" + badMap(bad, w, h))
}

// ---------------------------------------------------------------------------
// J. Render targets backed by an IOSurface.
//
// Section I creates its guest-backed targets as plain `.shared` textures, and a
// driven Maps boot says that is not what its layers are: every `pass_target`
// this device logs carries a mapping id, which a plain shared texture never
// has. A layer that another process composites is an IOSurface, the texture is
// created over it with `makeTexture(descriptor:iosurface:plane:)`, and the
// device routes it through a different rail from the one section I exercises --
// its own resident registry, its own sample rung, its own serialized plane
// view.
//
// So section I's fifteen green cases say nothing about the rail Maps' type
// layer is actually composited on. These are the same four questions plus the
// blend, asked of the target the app really uses.
//
// The surface picks its own `bytesPerRow`, which is the point of asking at
// these widths: 60 and 1000 texels are 240 and 4000 bytes and IOSurface will
// pad both, so the texture's stride is one the test never chose and any rail
// that assumes a tight row has somewhere to go wrong.
func makeIOSurfaceTarget(_ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
    // 'BGRA' as an OSType, which is what IOSurface's pixel-format key takes.
    let bgra: UInt32 = 0x4247_5241
    let props: [IOSurfacePropertyKey: Any] = [
        .width: w,
        .height: h,
        .bytesPerElement: 4,
        .pixelFormat: bgra,
    ]
    guard let surface = IOSurface(properties: props) else {
        report(label, false, "IOSurface(properties:) nil for \(w)x\(h)")
        return nil
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .shared
    guard let tex = dev.makeTexture(descriptor: td, iosurface: surface, plane: 0) else {
        report(label, false, "makeTexture(iosurface:) nil for \(w)x\(h)")
        return nil
    }
    return tex
}
