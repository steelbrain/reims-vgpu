import Metal
import Foundation
import IOSurface

// L. A whole-surface blit of a target the GPU has only just rendered into.
//
// The guest never reads the layer here — it hands both endpoints to a blit and
// asks the device to move the pixels. Metal orders that copy against the render
// before it: two command buffers on one queue execute in the order they were
// committed, so the copy sees everything the render wrote.
//
// A device that decomposes the pair does not get that ordering for free. If the
// render is submitted to the GPU and the copy is then serviced by reading the
// source's bytes on the CPU, the two are racing, and the reader wins whenever
// the GPU has not finished — which is most of the time, because submission is
// asynchronous and the copy is decoded immediately behind it.
//
// # The shape is measured, not chosen
//
// Which copies the guest driver emits as a whole-surface texture-to-texture
// copy, rather than staging through a buffer, is the driver's decision and not
// something this file can assume. Earlier attempts at this case used a
// `.bgra8Unorm` pair at 512x512 and never produced one: the driver staged every
// one of them, so the case exercised a different rail and passed for the wrong
// reason.
//
// The shape below is what a driven compositor actually emits — a linear source,
// an IOSurface-backed destination, `BGRA8Unorm_sRGB`, one level and one slice,
// at window and screen size. That is the pair a compositor produces when it
// draws a layer and then copies it into the surface the window server owns.
//
// # Why two passes
//
// The first pass lands red and is waited on, so the source's bytes are known
// and are *not* the answer. The second lands green and is not waited on before
// the copy. A correct device puts green in the destination; one that reads the
// source without ordering puts **red** there — the previous frame, whole and
// undamaged, which is why this class reads as a layer showing stale content
// rather than as corruption. `stale_previous_frame` in the failure counts
// exactly those texels, so the report distinguishes this defect from a copy
// that simply moved nothing.
//
// Full-intensity red and green round-trip an 8-bit sRGB encode exactly (0 and 1
// are both fixed points), so the expectation is unaffected by the transfer
// function on the attachment.
func makeSrgbIOSurfaceTarget(_ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
    let bgra: UInt32 = 0x4247_5241   // 'BGRA' as an OSType
    let props: [IOSurfacePropertyKey: Any] = [
        .width: w, .height: h, .bytesPerElement: 4, .pixelFormat: bgra,
    ]
    guard let surface = IOSurface(properties: props) else {
        report(label, false, "IOSurface(properties:) nil for \(w)x\(h)"); return nil
    }
    let td = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm_srgb, width: w, height: h, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .shared
    guard let tex = dev.makeTexture(descriptor: td, iosurface: surface, plane: 0) else {
        report(label, false, "makeTexture(iosurface:) nil for \(w)x\(h)"); return nil
    }
    return tex
}
