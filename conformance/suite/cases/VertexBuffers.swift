import Metal
import Foundation
import IOSurface

// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// F. The same linear alias, bound to a *fragment* stage in a render pass.
//
// Every case above reads the alias from a compute kernel. Type is not drawn
// that way: a glyph atlas reaches the rasterizer as a fragment texture, and a
// device may route a sampled guest image differently for the two stages. If a
// linear alias is exact under `read_texels` and wrong here, the seam is the
// stage binding and not the memory interpretation, which is a distinction no
// screenshot and no compute-only battery can draw.
// ---------------------------------------------------------------------------

/// Draw a full-target quad sampling `tex` into a fresh `rgba8Unorm` target of
/// the same size, and return the target's texels. With nearest filtering and a
/// target sized to the texture, pixel (x, y) samples texel (x, y) exactly.
func fragmentSample(_ tex: MTLTexture, _ w: Int, _ h: Int) -> [UInt32]? {
    guard let pipe = texPipeline else { return nil }
    let rd = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
    rd.usage = [.renderTarget, .shaderRead]
    rd.storageMode = .private
    guard let rt = dev.makeTexture(descriptor: rd) else { return nil }
    let verts = dev.makeBuffer(bytes: quadVerts,
                               length: quadVerts.count * 4,
                               options: .storageModeShared)!
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = rt
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 1, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    enc.setRenderPipelineState(pipe)
    enc.setVertexBuffer(verts, offset: 0, index: 0)
    enc.setFragmentTexture(tex, index: 0)
    enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return readBack(readPipe, rt, w, h)
}
