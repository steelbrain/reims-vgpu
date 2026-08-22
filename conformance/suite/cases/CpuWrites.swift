import Metal
import Foundation
import IOSurface

// The fragment arm runs in section F. `texPipeline` is a top-level binding and
// top-level code executes in order, so reading it from here answers nil and
// every fragment case reports a refusal it never made — which is what the
// native oracle caught when this loop ran both arms in place.

// ---------------------------------------------------------------------------
// C/D. Render-target round trips.
//
// Render into a texture, then (C) sample it in a later pass and (D) copy it out
// to a buffer and check the bytes. A device that renders correctly but cannot
// make the result readable again fails exactly one of these, which is the
// distinction a screenshot cannot draw.
// ---------------------------------------------------------------------------

let quadVerts: [Float] = [
    // x, y, u, v -- a triangle strip covering the whole target
    -1, -1, 0, 1,
     1, -1, 1, 1,
    -1,  1, 0, 0,
     1,  1, 1, 0,
]

func makeRenderPipeline(_ fragment: String, _ fmt: MTLPixelFormat) -> MTLRenderPipelineState? {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "quad_vs")
    d.fragmentFunction = library.makeFunction(name: fragment)
    d.colorAttachments[0].pixelFormat = fmt
    return try? dev.makeRenderPipelineState(descriptor: d)
}

func renderTargetCases() {
    let w = 64, h = 64
    let fmt = MTLPixelFormat.bgra8Unorm
    guard let pipe = makeRenderPipeline("solid_fs", fmt) else {
        report("rt_pipeline", false, "render pipeline creation failed"); return
    }
    report("rt_pipeline", true, "solid_fs pipeline built")

    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .private
    let rt = dev.makeTexture(descriptor: td)!

    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4, options: .storageModeShared)!
    // 0x40 red, 0x80 green, 0xC0 blue, opaque -- distinct in every channel so a
    // channel-order error is a different failure from a value error.
    var colour: [Float] = [64.0 / 255, 128.0 / 255, 192.0 / 255, 1.0]
    let wantPixel = pack(64, 128, 192, 255)

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

    // C: sample the rendered target in a separate submission.
    guard let sampled = readBack(readPipe, rt, w, h) else {
        refused("rt_render_then_sample"); return
    }
    let allRight = sampled.allSatisfy { $0 == wantPixel }
    report("rt_render_then_sample", allRight,
           allRight ? "\(w * h) texels are the drawn colour"
                    : "want=\(hex(wantPixel)) got=\(hex(sampled[0])) corner=\(hex(sampled[w * h - 1]))")

    // D: copy it out to a buffer and check the bytes the CPU sees.
    let bpr = w * 4
    let out = dev.makeBuffer(length: bpr * h, options: .storageModeShared)!
    memset(out.contents(), 0xEE, bpr * h)
    let cb2 = queue.makeCommandBuffer()!
    let blit = cb2.makeBlitCommandEncoder()!
    blit.copy(from: rt, sourceSlice: 0, sourceLevel: 0,
              sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: w, height: h, depth: 1),
              to: out, destinationOffset: 0,
              destinationBytesPerRow: bpr, destinationBytesPerImage: bpr * h)
    blit.endEncoding()
    cb2.commit()
    cb2.waitUntilCompleted()
    let p = out.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    // bgra8 in memory: B, G, R, A
    let okBytes = (0..<(w * h)).allSatisfy { i in
        p[i * 4] == 192 && p[i * 4 + 1] == 128 && p[i * 4 + 2] == 64 && p[i * 4 + 3] == 255
    }
    report("rt_blit_to_buffer", okBytes,
           okBytes ? "\(w * h) texels exact through blit"
                   : "first=[\(p[0]),\(p[1]),\(p[2]),\(p[3])] expect=[192,128,64,255]")
}

// ---------------------------------------------------------------------------
// E. A mip chain, level by level.
//
// The alias rail refuses a guest mip chain whose per-level offsets and pitches
// the host driver lays out differently, and that refusal is correct. What has
// never been checked is whether the levels the guest declares are *sampled*
// correctly once the copying rail carries them, which is a different question
// and the one a wrong level table would fail.
// ---------------------------------------------------------------------------

func mipCase() {
    let size = 64
    let levels = 7  // 64,32,16,8,4,2,1
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
                                                     width: size, height: size, mipmapped: true)
    d.mipmapLevelCount = levels
    d.storageMode = .shared
    d.usage = [.shaderRead]
    guard let tex = dev.makeTexture(descriptor: d) else {
        report("mip_chain_create", false, "makeTexture nil"); return
    }
    report("mip_chain_create", tex.mipmapLevelCount == levels, "levels=\(tex.mipmapLevelCount)")

    // Each level is filled with a constant that names the level, so a level
    // read at the wrong offset returns a neighbouring level's marker rather
    // than plausible-looking noise.
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let rows = [UInt8](repeating: marker, count: dim * dim * 4)
        rows.withUnsafeBytes { raw in
            tex.replace(region: MTLRegionMake2D(0, 0, dim, dim),
                        mipmapLevel: l,
                        withBytes: raw.baseAddress!,
                        bytesPerRow: dim * 4)
        }
    }
    for l in 0..<levels {
        let dim = max(1, size >> l)
        let marker = UInt8(0x10 + l * 0x11)
        let want = pack(marker, marker, marker, marker)
        // Fetched with the level named in the fetch, then sampled with the
        // level named as an explicit LOD. Which of the two fails says whether
        // the level's bytes are missing or the LOD is.
        guard let fetched = readBack(fetchLevelPipe, tex, dim, dim, level: l) else {
            refused("mip_fetch_level_\(l)_size\(dim)")
            skipDependent("mip_sample_level_\(l)_size\(dim)",
                          "mip_fetch_level_\(l)_size\(dim)")
            continue
        }
        let fOK = fetched.allSatisfy { $0 == want }
        report("mip_fetch_level_\(l)_size\(dim)", fOK,
               fOK ? "\(dim * dim) texels are level \(l)'s marker"
                   : "want=\(hex(want)) got=\(hex(fetched[0]))")
        guard let got = readBack(levelPipe, tex, dim, dim, level: l) else {
            refused("mip_sample_level_\(l)_size\(dim)"); continue
        }
        let ok = got.allSatisfy { $0 == want }
        report("mip_sample_level_\(l)_size\(dim)", ok,
               ok ? "\(dim * dim) texels are level \(l)'s marker"
                  : "want=\(hex(want)) got=\(hex(got[0]))")
    }
}
