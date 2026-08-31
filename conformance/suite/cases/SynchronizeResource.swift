import Metal
import Foundation

// `synchronizeResource:` is a no-op on unified memory and the explicit
// GPU-to-CPU visibility operation for managed memory. The storage mode follows
// the device topology, but the contract asserted by the case is identical: CPU
// reads after command-buffer completion observe the preceding GPU writes.
func synchronizeResourceCase() {
    let label = "synchronize_resource_gpu_to_cpu"
    let count = 4096
    let bytes = count * MemoryLayout<UInt32>.size
    let options: MTLResourceOptions = dev.hasUnifiedMemory
        ? .storageModeShared
        : .storageModeManaged
    guard let words = dev.makeBuffer(length: bytes, options: options) else {
        report(label, false, "buffer creation failed")
        return
    }

    memset(words.contents(), 0, bytes)
    if !dev.hasUnifiedMemory {
        words.didModifyRange(0..<bytes)
    }
    var token: UInt32 = 0x6A09E667
    let cb = queue.makeCommandBuffer()!
    let compute = cb.makeComputeCommandEncoder()!
    compute.setComputePipelineState(pipeline("sync_resource_write"))
    compute.setBuffer(words, offset: 0, index: 0)
    compute.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 1)
    compute.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    compute.endEncoding()

    let blit = cb.makeBlitCommandEncoder()!
    blit.synchronize(resource: words)
    blit.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let got = words.contents().bindMemory(to: UInt32.self, capacity: count)
    if let bad = (0..<count).first(where: { got[$0] != token ^ UInt32($0) }) {
        report(label, false,
               "word=\(bad) got=\(hex(got[bad])) want=\(hex(token ^ UInt32(bad)))")
        return
    }
    report(label, true, "managed/shared GPU writes visible to CPU after synchronize")
}
