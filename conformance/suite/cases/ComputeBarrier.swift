import Metal
import Foundation

// A concurrent compute encoder promises no ordering between dispatches except
// the dependency named here. Alternating the token makes every stale word a
// deterministic wrong answer after the first round.
func computeBarrierCase(resourceBarrier: Bool) {
    let label = resourceBarrier ? "compute_barrier_resource" : "compute_barrier_scope_buffers"
    let count = 8192
    let bytes = count * MemoryLayout<UInt32>.size
    let writePipe = pipeline("compute_barrier_write")
    let readPipe = pipeline("compute_barrier_read")

    guard let words = dev.makeBuffer(length: bytes, options: .storageModePrivate),
          let verdict = dev.makeBuffer(length: bytes, options: .storageModeShared) else {
        report(label, false, "resource creation failed")
        return
    }

    for round in 0..<8 {
        memset(verdict.contents(), 0, bytes)
        var token: UInt32 = round.isMultiple(of: 2) ? 0x13579BDF : 0x2468ACE0
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent)!
        enc.setComputePipelineState(writePipe)
        enc.setBuffer(words, offset: 0, index: 0)
        enc.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 1)
        enc.dispatchThreadgroups(
            MTLSize(width: count / 64, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))

        if resourceBarrier {
            enc.memoryBarrier(resources: [words])
        } else {
            enc.memoryBarrier(scope: .buffers)
        }

        enc.setComputePipelineState(readPipe)
        enc.setBuffer(words, offset: 0, index: 0)
        enc.setBuffer(verdict, offset: 0, index: 1)
        enc.setBytes(&token, length: MemoryLayout<UInt32>.size, index: 2)
        enc.dispatchThreadgroups(
            MTLSize(width: count / 64, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let got = verdict.contents().bindMemory(to: UInt32.self, capacity: count)
        if let bad = (0..<count).first(where: { got[$0] != 1 }) {
            report(label, false,
                   "round=\(round) word=\(bad) got=\(got[bad]) want=1")
            return
        }
    }

    report(label, true, "8 alternating concurrent producer/consumer rounds were visible")
}
