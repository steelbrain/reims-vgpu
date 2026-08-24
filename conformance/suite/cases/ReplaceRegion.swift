import Metal
import Foundation
import IOSurface

// ---------------------------------------------------------------------------
// I. Render targets the guest allocated in its own memory.
//
// Every render target above this line is `.private`, so the device allocates it
// and the guest never names the pages behind it. That is one of the two kinds
// of render target a compositing app uses and it is not the interesting one: a
// layer that the CPU also rasterizes into, or that another process composites,
// is `.shared`, and its bytes are guest memory the device may bind a Vulkan
// image directly over instead of copying.
//
// Those are two different rails with two different failure modes, and until
// this section the battery had exactly one case on the second of them
// (`cpu_write_after_render`, section F5). A whole rail behind one case is not
// coverage -- it is a single sample that happens to pass, which is what let a
// live defect sit under 173 green cases.
//
// The four questions below are the ones a direct binding over guest pages can
// answer differently from a copy, in the order they get harder:
//
//   draw      -- does rendering into guest-owned pages land at all,
//   load      -- does a second pass see what the first one left, or does the
//                seed for `.load` overwrite it with a stale copy,
//   scissor   -- does a partial write land on the right rows, which is where a
//                row pitch the guest did not agree to shows up as a shear,
//   sample    -- does a later pass sampling that same texture read what was
//                rendered, which is the crossover between "this is a target"
//                and "this is a source" over one allocation.
//
// The widths are chosen so the row pitch cannot be assumed: 60 and 1000 texels
// are 240 and 4000 bytes, neither a multiple of the 256-byte linear alignment
// this device reports, so any rail that confuses the guest's stride with a
// padded one puts the pixels somewhere this section can see.
// ---------------------------------------------------------------------------

func sharedRenderTarget(_ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    rd.storageMode = .shared
    guard let rt = dev.makeTexture(descriptor: rd) else {
        report(label, false, "makeTexture nil for a shared \(w)x\(h) render target")
        return nil
    }
    return rt
}
