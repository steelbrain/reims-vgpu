import Metal
import Foundation

// The indirect buffer begins one word into the allocation so this covers the
// byte offset as well as the three MTLDispatchThreadgroupsIndirectArguments
// counts. Metal multiplies those counts by threadsPerThreadgroup to form the
// shader-visible grid.
func indirectComputeDispatchCase() {
    let label = "compute_dispatch_threadgroups_indirect"
    let capacity = 64
    let expectedThreads = 12 * 4
    let byteCount = capacity * MemoryLayout<UInt32>.size

    guard let output = dev.makeBuffer(length: byteCount, options: .storageModeShared),
          let observedGrid = dev.makeBuffer(length: 3 * MemoryLayout<UInt32>.size,
                                            options: .storageModeShared),
          let ran = dev.makeBuffer(length: MemoryLayout<UInt32>.size,
                                   options: .storageModeShared),
          let indirect = dev.makeBuffer(length: 5 * MemoryLayout<UInt32>.size,
                                        options: .storageModeShared) else {
        report(label, false, "buffer creation failed")
        return
    }

    memset(output.contents(), 0, byteCount)
    memset(observedGrid.contents(), 0, 3 * MemoryLayout<UInt32>.size)
    memset(ran.contents(), 0, MemoryLayout<UInt32>.size)
    let arguments = indirect.contents().bindMemory(to: UInt32.self, capacity: 5)
    arguments[0] = 0xdeadbeef
    arguments[1] = 3
    arguments[2] = 2
    arguments[3] = 1
    arguments[4] = 0xcafebabe

    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipeline("indirect_threadgroups_write"))
    enc.setBuffer(output, offset: 0, index: 0)
    enc.setBuffer(observedGrid, offset: 0, index: 1)
    var outputCapacity = UInt32(capacity)
    enc.setBytes(&outputCapacity, length: MemoryLayout<UInt32>.size, index: 2)
    enc.setBuffer(ran, offset: 0, index: 4)
    enc.dispatchThreadgroups(
        indirectBuffer: indirect,
        indirectBufferOffset: MemoryLayout<UInt32>.size,
        threadsPerThreadgroup: MTLSize(width: 4, height: 2, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()

    let ranWord = ran.contents().bindMemory(to: UInt32.self, capacity: 1)[0]
    guard ranWord != 0 else {
        report(label, false, "the indirect dispatch produced nothing — the device refused it")
        return
    }

    let gridWords = observedGrid.contents().bindMemory(to: UInt32.self, capacity: 3)
    let grid = Array(UnsafeBufferPointer(start: gridWords, count: 3))
    let outputWords = output.contents().bindMemory(to: UInt32.self, capacity: capacity)
    let firstBad = (0..<capacity).first { index in
        let expected: UInt32 = index < expectedThreads ? 0xa5000000 | UInt32(index) : 0
        return outputWords[index] != expected
    }

    report(label, grid == [12, 4, 1] && firstBad == nil,
           "groups=[3,2,1] local=[4,2,1] grid=\(grid) first_bad=\(firstBad.map(String.init) ?? "none")")
}
