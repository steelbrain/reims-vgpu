import Metal
import Foundation
import IOSurface

func fragmentAliasCase(_ f: Fmt, _ w: Int, _ h: Int, pitch: Pitch) {
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = pitch.bytes(width: w, bpp: f.bpp, align: align)
    let label = "fragsample_\(f.name)_\(w)x\(h)_pitch_\(pitch.tag)"
    // This case reports under two names, so every exit from it has to report
    // under both. A skip that names only `label` emits one line where a host
    // that runs the case emits two, and the two runs then pair up on neither.
    let arms = ["first_draw", "after_cpu_rewrite"]
    func bailAll(skipping: Bool, _ why: String) {
        for arm in arms {
            if skipping { skip("\(label)_\(arm)", why) }
            else { report("\(label)_\(arm)", false, why) }
        }
    }
    if bpr % align != 0 || bpr < w * f.bpp {
        bailAll(skipping: true, "pitch \(bpr) is not a multiple of this device's minimumLinearTextureAlignment=\(align)")
        return
    }
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        bailAll(skipping: false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    for y in 0..<h { for x in 0..<bpr { base[y * bpr + x] = UInt8((x &* 5 &+ y &* 17 &+ 3) & 0xFF) } }
    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        bailAll(skipping: false, "makeTexture nil"); return
    }

    func check(_ tag: String, _ expectByte: (Int, Int) -> [UInt8]) {
        guard let got = fragmentSample(tex, w, h) else {
            report("\(label)_\(tag)", false, "render pipeline unavailable"); return
        }
        var bad: [(Int, Int)] = []
        var first = ""
        for y in 0..<h {
            for x in 0..<w {
                let want = f.expect(expectByte(x, y))
                let have = got[y * w + x]
                if want != have {
                    bad.append((x, y))
                    if first.isEmpty { first = "first_bad=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))" }
                }
            }
        }
        report("\(label)_\(tag)", bad.isEmpty,
               bad.isEmpty ? "\(w * h) texels exact through the rasterizer"
                           : "\(bad.count)/\(w * h) wrong \(first) \(badMap(bad, w, h))")
    }

    check("first_draw") { x, y in
        (0..<f.bpp).map { base[y * bpr + x * f.bpp + $0] }
    }
    // Rewrite every byte, with the texture already drawn with once and never
    // re-declared. This is the glyph atlas being refilled between frames.
    for y in 0..<h { for x in 0..<bpr { base[y * bpr + x] = UInt8((x &* 3 &+ y &* 29 &+ 200) & 0xFF) } }
    check("after_cpu_rewrite") { x, y in
        (0..<f.bpp).map { base[y * bpr + x * f.bpp + $0] }
    }
}

let texPipeline = makeRenderPipeline("tex_fs", .rgba8Unorm)

// ---------------------------------------------------------------------------
// F2. The same alias, filled so that every byte names its own offset.
//
// A wrong-pitch read, a wrong-offset read and a byte that simply is not there
// are three different defects and the pattern fills above cannot tell them
// apart -- a mismatch says "not what I wrote" and stops. Here byte `i` holds
// `1 + (i % 251)`, so a returned value inverts to a source offset modulo 251,
// and the *difference* between the offset the contract names and the offset the
// device read is the finding. 251 is the largest prime below 256, so the cycle
// never divides a row pitch and an alias off by a row is not congruent to one
// that is exact. Zero is outside the fill's range entirely, so it means the
// device returned a byte nothing in this buffer ever held.
// ---------------------------------------------------------------------------

func offsetOracleCase(_ w: Int, _ h: Int, pitch: Pitch, viaFragment: Bool) {
    let f = formats[1]  // a8Unorm: one byte per texel, so a texel *is* an offset
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = pitch.bytes(width: w, bpp: f.bpp, align: align)
    let label = "offset_oracle_\(w)x\(h)_pitch_\(pitch.tag)\(viaFragment ? "_fragment" : "_compute")"
    if bpr % align != 0 || bpr < w {
        skip(label, "pitch \(bpr) is not a multiple of this device's minimumLinearTextureAlignment=\(align)")
        return
    }
    guard let buf = dev.makeBuffer(length: bpr * h, options: .storageModeShared) else {
        report(label, false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: bpr * h)
    for i in 0..<(bpr * h) { base[i] = UInt8(1 + (i % 251)) }
    let d = MTLTextureDescriptor()
    d.textureType = .type2D; d.pixelFormat = f.mtl
    d.width = w; d.height = h; d.mipmapLevelCount = 1
    d.storageMode = .shared; d.usage = [.shaderRead]
    guard let tex = buf.makeTexture(descriptor: d, offset: 0, bytesPerRow: bpr) else {
        report(label, false, "makeTexture nil"); return
    }
    let got: [UInt32]?
    if viaFragment { got = fragmentSample(tex, w, h) } else { got = readBack(readPipe, tex, w, h) }
    // The reading this whole case exists to make trustworthy: without the
    // kernel's own run witness a refused dispatch arrived here as a buffer full
    // of `0xEE`, which inverts to a valid-looking source offset and reported as
    // `absent=0 shifted=4080`. See `readBack`.
    guard let got else { refused(label); return }

    var absent: [(Int, Int)] = []
    var deltas: [Int: Int] = [:]   // (read offset - contract offset) mod 251 -> count
    for y in 0..<h {
        for x in 0..<w {
            let a = Int((got[y * w + x] >> 24) & 0xFF)
            if a == 0 { absent.append((x, y)); continue }
            let readOffset = (a - 1) % 251
            let wantOffset = (y * bpr + x) % 251
            let delta = ((readOffset - wantOffset) % 251 + 251) % 251
            if delta != 0 { deltas[delta, default: 0] += 1 }
        }
    }
    let ok = absent.isEmpty && deltas.isEmpty
    var detail = "\(w * h) texels at the offsets the contract names"
    if !ok {
        let top = deltas.sorted { $0.value > $1.value }.prefix(4)
            .map { "delta=\($0.key)x\($0.value)" }.joined(separator: " ")
        detail = "absent=\(absent.count) shifted=\(deltas.values.reduce(0, +)) \(top) "
            + badMap(absent, w, h)
    }
    report(label, ok, detail)
}

// ---------------------------------------------------------------------------
// F3. A render target whose width is not a multiple of eight.
//
// Every failing case above lost exactly `alignUp(w, 8) - w` columns, and the
// two that passed were the two whose width was already a multiple of eight.
// That is a property of the *target*, not of the alias being sampled, so this
// draws a flat colour with no texture bound at all: if it fails the same way,
// nothing about buffer-backed textures is involved and the finding is about
// render-target width.
// ---------------------------------------------------------------------------

func targetWidthCase(_ w: Int, _ h: Int) {
    guard let pipe = solidPipeline else {
        report("rt_width_\(w)", false, "solid pipeline unavailable"); return
    }
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    rd.storageMode = .private
    guard let rt = dev.makeTexture(descriptor: rd) else {
        report("rt_width_\(w)", false, "makeTexture nil"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = .clear
    // Cleared to a colour that is neither the drawn colour nor zero, so a
    // failing texel says which of "never drawn" and "never anything" it is.
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    var colour = SIMD4<Float>(0, 1, 0, 1)
    enc.setFragmentBytes(&colour, length: 16, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    guard let got = readBack(readPipe, rt, w, h) else {
        refused("rt_width_\(w)"); return
    }
    let want = pack(0, 255, 0, 255)
    let clear = pack(255, 0, 255, 255)
    var bad: [(Int, Int)] = []
    var zero = 0, cleared = 0
    for y in 0..<h {
        for x in 0..<w where got[y * w + x] != want {
            bad.append((x, y))
            if got[y * w + x] == 0 { zero += 1 }
            if got[y * w + x] == clear { cleared += 1 }
        }
    }
    let pad = alignUp(w, 8) - w
    report("rt_width_\(w)", bad.isEmpty,
           bad.isEmpty ? "\(w * h) texels are the drawn colour"
                       : "\(bad.count)/\(w * h) wrong zero=\(zero) still_clear=\(cleared) "
                         + "alignUp(w,8)-w=\(pad) \(badMap(bad, w, h))")
}

let solidPipeline = makeRenderPipeline("solid_fs", .rgba8Unorm)

// ---------------------------------------------------------------------------
// F4. `dispatchThreads` must run the grid it is given and not one thread more.
//
// `dispatchThreads:threadsPerThreadgroup:` takes a thread count, not a
// threadgroup count, and Metal is required to launch exactly that many threads
// however badly the count divides the threadgroup. A device that rounds the
// grid up to whole threadgroups instead runs extra threads with a
// `thread_position_in_grid` outside the grid, and every one of them is a stray
// write at whatever address the shader computes from it.
//
// Nothing about this is visible in a shader that is careful, which is why it
// hides: the damage lands in the *caller's* buffer, at the addresses just past
// each row, and those are the addresses the next row occupies. That is exactly
// the shape `rt_width_*` reports -- the first `alignUp(w, 8) - w` texels of a
// row, zeroed at random, with row zero always intact because nothing is
// dispatched before it.
//
// Here each thread owns a slot in a grid padded to the threadgroup size, so a
// stray thread cannot overwrite a real one and the evidence survives.
// ---------------------------------------------------------------------------

func gridBoundsCase(_ w: Int, _ h: Int, _ tg: Int) {
    let label = "dispatch_threads_grid_\(w)x\(h)_tg\(tg)"
    let pipe = pipeline("grid_bounds")
    let stride = alignUp(w, tg)
    let rows = alignUp(h, tg)
    let out = dev.makeBuffer(length: stride * rows * 4, options: .storageModeShared)!
    memset(out.contents(), 0, stride * rows * 4)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(out, offset: 0, index: 0)
    var strideU = UInt32(stride)
    enc.setBytes(&strideU, length: 4, index: 1)
    enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tg, height: tg, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let p = out.contents().bindMemory(to: UInt32.self, capacity: stride * rows)
    var missing = 0        // inside the grid and never written
    var strayX = 0         // x >= w
    var strayY = 0         // y >= h
    var firstStray = ""
    for y in 0..<rows {
        for x in 0..<stride {
            let v = p[y * stride + x]
            let inGrid = x < w && y < h
            if inGrid {
                if v != UInt32(1 + x) { missing += 1 }
            } else if v != 0 {
                if x >= w { strayX += 1 } else { strayY += 1 }
                if firstStray.isEmpty { firstStray = "first_stray=(\(x),\(y))=\(v)" }
            }
        }
    }
    let ok = missing == 0 && strayX == 0 && strayY == 0
    report(label, ok,
           ok ? "\(w * h) threads ran and nothing outside the grid did"
              : "missing=\(missing) stray_past_width=\(strayX) stray_past_height=\(strayY) \(firstStray)")
}

// ---------------------------------------------------------------------------
// F5. A CPU write into a texture the GPU has already rendered into.
//
// This is the shape of a compositor that draws its geometry on the GPU and
// rasterizes its type on the CPU, into one shared texture. Metal's contract is
// that both writers land: the render pass owns what it drew, the CPU owns what
// it wrote afterwards, and a later read sees each in the region it wrote.
//
// A device that keeps its own copy of the target and writes that copy back into
// the guest's pages after the fact destroys the second writer's bytes and
// nothing reports it. The failure is content, and content is what no counter in
// this project measures -- so it is asked here directly, and asked in both
// orders, because a writeback landing late and a writeback landing early are
// different bugs with the same symptom.
// ---------------------------------------------------------------------------

func cpuWriteAfterRenderCase(_ w: Int, _ h: Int, secondPass: Bool) {
    let label = "cpu_write_after_render_\(w)x\(h)\(secondPass ? "_then_second_pass" : "")"
    guard let pipe = solidPipeline else { report(label, false, "solid pipeline unavailable"); return }
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    // Shared, so the CPU can write it and the GPU can render into it -- which
    // is the whole point and is what a compositor's own surfaces are.
    rd.storageMode = .shared
    guard let rt = dev.makeTexture(descriptor: rd) else {
        report(label, false, "makeTexture nil"); return
    }
    let verts = dev.makeBuffer(bytes: quadVerts, length: quadVerts.count * 4,
                               options: .storageModeShared)!

    func draw(_ colour: SIMD4<Float>, load: Bool) {
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = rt
        pass.colorAttachments[0].loadAction = load ? .load : .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
        enc.setRenderPipelineState(pipe)
        enc.setVertexBuffer(verts, offset: 0, index: 0)
        var c = colour
        enc.setFragmentBytes(&c, length: 16, index: 0)
        // Only the top half, so the bottom half is the CPU's alone and a
        // whole-target writeback is the only thing that could reach it.
        enc.setScissorRect(MTLScissorRect(x: 0, y: 0, width: w, height: h / 2))
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    // 1. The GPU renders into the top half.
    draw(SIMD4<Float>(0, 1, 0, 1), load: false)

    // 2. The CPU rasterizes into the bottom half, after the pass completed.
    let bottom = h / 2
    let bpr = w * 4
    var rows = [UInt8](repeating: 0, count: bpr * (h - bottom))
    for y in 0..<(h - bottom) {
        for x in 0..<w {
            let o = y * bpr + x * 4
            rows[o] = UInt8((x &* 7 &+ y &* 3 &+ 1) & 0xFF)
            rows[o + 1] = 0x20
            rows[o + 2] = 0x40
            rows[o + 3] = 0xFF
        }
    }
    rows.withUnsafeBytes { raw in
        rt.replace(region: MTLRegionMake2D(0, bottom, w, h - bottom),
                   mipmapLevel: 0, withBytes: raw.baseAddress!, bytesPerRow: bpr)
    }

    // 3. Optionally another pass over the top half only. A device that reloads
    //    its own stale copy for `.load` and stores the whole target back is
    //    caught here and not by the first arm.
    if secondPass { draw(SIMD4<Float>(0, 0, 1, 1), load: true) }

    guard let got = readBack(readPipe, rt, w, h) else {
        refused(label); return
    }
    let drawn = secondPass ? pack(0, 0, 255, 255) : pack(0, 255, 0, 255)
    var topBad: [(Int, Int)] = []
    var cpuBad: [(Int, Int)] = []
    var cpuFirst = ""
    for y in 0..<h {
        for x in 0..<w {
            let have = got[y * w + x]
            if y < bottom {
                if have != drawn { topBad.append((x, y)) }
            } else {
                let o = (y - bottom) * bpr + x * 4
                let want = pack(rows[o], rows[o + 1], rows[o + 2], rows[o + 3])
                if have != want {
                    cpuBad.append((x, y))
                    if cpuFirst.isEmpty {
                        cpuFirst = "at=(\(x),\(y)) want=\(hex(want)) got=\(hex(have))"
                    }
                }
            }
        }
    }
    let ok = topBad.isEmpty && cpuBad.isEmpty
    report(label, ok,
           ok ? "the GPU kept its half and the CPU kept its half"
              : "gpu_half_wrong=\(topBad.count) cpu_half_wrong=\(cpuBad.count) \(cpuFirst) "
                + badMap(cpuBad, w, h))
}

// ---------------------------------------------------------------------------
// G. A linear alias at a non-zero offset into its allocation.
//
// A glyph atlas is a sub-range of a larger buffer, so the offset is part of the
// contract. A device that resolves the allocation but drops the offset reads
// the right pages and the wrong bytes, which looks exactly like corruption.
// ---------------------------------------------------------------------------

func offsetAliasCase(_ f: Fmt, _ w: Int, _ h: Int, tiles: Int) {
    let align = dev.minimumLinearTextureAlignment(for: f.mtl)
    let bpr = alignUp(w * f.bpp, align)
    let stride = alignUp(bpr * h, max(align, 256))
    let label = "offset_alias_\(f.name)_\(w)x\(h)_x\(tiles)"
    guard let buf = dev.makeBuffer(length: stride * tiles, options: .storageModeShared) else {
        report(label, false, "buffer allocation failed"); return
    }
    let base = buf.contents().bindMemory(to: UInt8.self, capacity: stride * tiles)
    for i in 0..<(stride * tiles) { base[i] = UInt8((i &* 31 &+ 7) & 0xFF) }

    var bad = 0
    var first = ""
    for t in 0..<tiles {
        let off = stride * t
        let d = MTLTextureDescriptor()
        d.textureType = .type2D; d.pixelFormat = f.mtl
        d.width = w; d.height = h; d.mipmapLevelCount = 1
        d.storageMode = .shared; d.usage = [.shaderRead]
        guard let tex = buf.makeTexture(descriptor: d, offset: off, bytesPerRow: bpr) else {
            report(label, false, "makeTexture nil at offset \(off)"); return
        }
        guard let got = readBack(readPipe, tex, w, h) else {
            refused(label); return
        }
        for y in 0..<h {
            for x in 0..<w {
                let bytes = (0..<f.bpp).map { base[off + y * bpr + x * f.bpp + $0] }
                let want = f.expect(bytes)
                if got[y * w + x] != want {
                    bad += 1
                    if first.isEmpty {
                        first = "tile=\(t) offset=\(off) at=(\(x),\(y)) want=\(hex(want)) got=\(hex(got[y * w + x]))"
                    }
                }
            }
        }
    }
    report(label, bad == 0,
           bad == 0 ? "\(tiles) tiles exact at their own offsets" : "\(bad) wrong \(first)")
}
