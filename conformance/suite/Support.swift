import Metal
import Foundation
import IOSurface

// Compiled on first use. `main.swift` forces it before reporting `shader_compile`
// so the report still lands where it always did, first.
let library: MTLLibrary = {
    do {
        return try dev.makeLibrary(source: shaderSource, options: nil)
    } catch {
        print("CASE shader_compile FAIL \(error)")
        fflush(stdout)
        exit(2)
    }
}()

func pipeline(_ name: String) -> MTLComputePipelineState {
    let fn = library.makeFunction(name: name)!
    return try! dev.makeComputePipelineState(function: fn)
}

let readPipe = pipeline("read_texels")

let readMultisamplePipe = pipeline("read_ms_texels")

let readMultisampleHostCountPipe = pipeline("read_ms_texels_host_count")

let samplePipe = pipeline("sample_texels")

let levelPipe = pipeline("read_level")

let fetchLevelPipe = pipeline("fetch_level")

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func alignUp(_ v: Int, _ a: Int) -> Int { (v + a - 1) / a * a }

/// How a linear case arrived at its bytes-per-row.
///
/// **The name has to be the derivation and not the number it came to.**
/// `minimumLinearTextureAlignment` is 16 on an M-series device and 256 on
/// Apple's paravirtual one, so a pitch derived from it lands on a different
/// integer per host -- and a case named after that integer has no counterpart
/// on the other host to be scored against. Sixty-two cases here were in that
/// state: they ran on both, tested the same thing on both, and the native/guest
/// comparison that makes a guest failure a finding could not pair a single one
/// of them.
///
/// Carrying the derivation also puts the alignment arithmetic in one place. It
/// used to be written out at each call site, which is the same value spelled
/// twice with nothing checking that the two agree.
enum Pitch {
    /// The smallest pitch this device will accept for the format.
    case tight
    /// That many whole alignment units above tight.
    case padded(rows: Int)
    /// A literal, from a guest census. Not every device can express one, and
    /// the case skips where Metal would reject the descriptor.
    case exact(Int)

    /// The label component. `.exact` keeps the bare number because a literal
    /// pitch means the same thing on every host.
    var tag: String {
        switch self {
        case .tight: return "tight"
        case .padded(let rows): return "tight_plus\(rows)"
        case .exact(let value): return "\(value)"
        }
    }

    func bytes(width: Int, bpp: Int, align: Int) -> Int {
        switch self {
        case .tight: return alignUp(width * bpp, align)
        case .padded(let rows): return alignUp(width * bpp, align) + rows * align
        case .exact(let value): return value
        }
    }
}

/// Run one of the texel-reading kernels over a texture and return the packed
/// RGBA of every texel, in row-major order.
/// `nil` means the dispatch produced nothing: the kernel never ran.
///
/// **A sentinel fill cannot answer this and must not be asked to.** This device
/// refuses a dispatch on the host side, so the guest's command buffer completes
/// clean and the output buffer keeps whatever was in it. Every case here then
/// compares the sentinel against what it wanted and reports a *content*
/// failure — which reads as "the device returned the wrong bytes" when the
/// truth is that it returned none. The `offset_oracle` cases showed how far
/// that misleads: their fill is `1 + (i % 251)`, zero means "a byte nothing in
/// this buffer ever held", and the sentinel `0xEE` is 238, squarely inside the
/// fill's own range. So a refused dispatch inverted to a constant read-offset
/// of 237, every texel landed in a different delta bucket, and four cases
/// reported `absent=0 shifted=4080` — a precise, plausible account of a defect
/// that did not exist. No sentinel value fixes this in general: a battery whose
/// cases cover many formats has no byte that is out of range for all of them.
///
/// So the kernel says so itself, in a buffer of its own.
func readBack(_ pipe: MTLComputePipelineState,
              _ tex: MTLTexture,
              _ w: Int, _ h: Int,
              level: Int? = nil) -> [UInt32]? {
    let out = dev.makeBuffer(length: w * h * 4, options: .storageModeShared)!
    memset(out.contents(), 0xEE, w * h * 4)
    let ran = dev.makeBuffer(length: 4, options: .storageModeShared)!
    memset(ran.contents(), 0, 4)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setTexture(tex, index: 0)
    enc.setBuffer(out, offset: 0, index: 0)
    var width = UInt32(w)
    enc.setBytes(&width, length: 4, index: 1)
    if var lvl = level.map({ UInt32($0) }) {
        enc.setBytes(&lvl, length: 4, index: 2)
    }
    var extent = SIMD2<UInt32>(UInt32(w), UInt32(h))
    enc.setBytes(&extent, length: 8, index: 3)
    enc.setBuffer(ran, offset: 0, index: 4)
    // Whole threadgroups plus an explicit guard in the kernel, deliberately
    // *not* `dispatchThreads`. The battery's own readback must not depend on
    // the thing `dispatch_threads_grid_*` is here to test: while a device runs
    // the surplus threads of a partial grid, an unguarded readback writes each
    // row's overrun into the next row's first entries and every other case in
    // this file reports that as its own failure.
    let tg = 8
    enc.dispatchThreadgroups(
        MTLSize(width: (w + tg - 1) / tg, height: (h + tg - 1) / tg, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tg, height: tg, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0] == 0 { return nil }
    let p = out.contents().bindMemory(to: UInt32.self, capacity: w * h)
    return Array(UnsafeBufferPointer(start: p, count: w * h))
}

/// Read every sample of a multisample texture without resolving it.
///
/// The output is pixel-major and then sample-major. Slots the bound image does
/// not expose retain the sentinel, so accidentally binding a single-sample
/// image cannot look like a valid four-sample store.
func readBackMultisample(_ tex: MTLTexture,
                         _ w: Int, _ h: Int,
                         samples: Int,
                         countFromTexture: Bool = true) -> [UInt32]? {
    let count = w * h * samples
    let out = dev.makeBuffer(length: count * 4, options: .storageModeShared)!
    memset(out.contents(), 0xEE, count * 4)
    let ran = dev.makeBuffer(length: 4, options: .storageModeShared)!
    memset(ran.contents(), 0, 4)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(
        countFromTexture ? readMultisamplePipe : readMultisampleHostCountPipe)
    enc.setTexture(tex, index: 0)
    enc.setBuffer(out, offset: 0, index: 0)
    var width = UInt32(w)
    var sampleCount = UInt32(samples)
    var extent = SIMD2<UInt32>(UInt32(w), UInt32(h))
    enc.setBytes(&width, length: 4, index: 1)
    enc.setBytes(&sampleCount, length: 4, index: 2)
    enc.setBytes(&extent, length: 8, index: 3)
    enc.setBuffer(ran, offset: 0, index: 4)
    let tg = 8
    enc.dispatchThreadgroups(
        MTLSize(width: (w + tg - 1) / tg, height: (h + tg - 1) / tg, depth: 1),
        threadsPerThreadgroup: MTLSize(width: tg, height: tg, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0] == 0 { return nil }
    let p = out.contents().bindMemory(to: UInt32.self, capacity: count)
    return Array(UnsafeBufferPointer(start: p, count: count))
}

/// A case that cannot be evaluated because the run it depends on did not
/// happen.
///
/// Reported rather than dropped. A battery whose case *count* moves between two
/// runs cannot be diffed against itself, and a name that simply stops appearing
/// reads as a case someone deleted — which is how a refusal of one case quietly
/// took three others out of the totals.
func skipDependent(_ name: String, _ on: String) {
    skip(name, "not evaluated — \(on) never ran")
}

/// One wording for every case, so a refusal is never mistaken for a mismatch.
func refused(_ label: String) {
    report(label, false,
           "the readback dispatch produced nothing — the device refused it, "
           + "or refused a bind in it; the texels below were never written")
}

/// A compact map of where a case's wrong texels are. A count alone cannot tell
/// a lost row from a lost page from scattered noise, and those are three
/// different defects: this prints `y=<row>:<first>-<last>x<count>` per affected
/// row so the shape is in the result line rather than in a follow-up run.
func badMap(_ bad: [(Int, Int)], _ w: Int, _ h: Int) -> String {
    if bad.isEmpty { return "" }
    var perRow: [Int: (Int, Int, Int)] = [:]   // y -> (minX, maxX, count)
    for (x, y) in bad {
        if let e = perRow[y] { perRow[y] = (min(e.0, x), max(e.1, x), e.2 + 1) }
        else { perRow[y] = (x, x, 1) }
    }
    let rows = perRow.keys.sorted()
    let shown = rows.prefix(12).map { y -> String in
        let e = perRow[y]!
        return "y=\(y):\(e.0)-\(e.1)x\(e.2)"
    }.joined(separator: " ")
    let more = rows.count > 12 ? " (+\(rows.count - 12) more rows)" : ""
    return "rows=\(rows.count)/\(h) \(shown)\(more)"
}

func pack(_ r: UInt8, _ g: UInt8, _ b: UInt8, _ a: UInt8) -> UInt32 {
    (UInt32(a) << 24) | (UInt32(b) << 16) | (UInt32(g) << 8) | UInt32(r)
}

func hex(_ v: UInt32) -> String { String(format: "0x%08x", v) }

// ---------------------------------------------------------------------------
// A. A linear texture over a shared buffer, per format, tight and padded pitch.
//
// This is the shape Maps' type layer actually has. The census of a driven boot
// found ~90 distinct `A8Unorm` sources a boot with padded rows -- 54x16 at
// pitch 64, 218x16 at pitch 256, 85x85 at pitch 128 -- so those exact
// geometries are cases here rather than round numbers.
// ---------------------------------------------------------------------------

struct Fmt {
    let name: String
    let mtl: MTLPixelFormat
    let bpp: Int
    /// What Metal must return for a texel whose bytes are `b`.
    let expect: ([UInt8]) -> UInt32
}
