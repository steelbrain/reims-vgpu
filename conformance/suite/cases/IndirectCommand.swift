import Metal
import Foundation
import ObjectiveC

private typealias ICBRangeMessage = @convention(c) (
    AnyObject, Selector, AnyObject, NSRange
) -> Void
private typealias ICBCopyMessage = @convention(c) (
    AnyObject, Selector, AnyObject, NSRange, AnyObject, UInt
) -> Void

private let objcMessage = dlsym(dlopen(nil, RTLD_NOW), "objc_msgSend")!

private func makeIndirectPipeline() -> MTLComputePipelineState? {
    let descriptor = MTLComputePipelineDescriptor()
    descriptor.computeFunction = library.makeFunction(name: "indirect_command_write")
    descriptor.supportIndirectCommandBuffers = true
    return try? dev.makeComputePipelineState(descriptor: descriptor,
                                             options: [], reflection: nil)
}

private func resetCommands(_ encoder: MTLBlitCommandEncoder,
                           _ buffer: MTLIndirectCommandBuffer,
                           _ range: Range<Int>) {
    let send = unsafeBitCast(objcMessage, to: ICBRangeMessage.self)
    send(encoder as AnyObject, NSSelectorFromString("resetCommandsInBuffer:withRange:"),
         buffer as AnyObject, NSRange(location: range.lowerBound, length: range.count))
}

private func optimizeCommands(_ encoder: MTLBlitCommandEncoder,
                              _ buffer: MTLIndirectCommandBuffer,
                              _ range: Range<Int>) {
    let send = unsafeBitCast(objcMessage, to: ICBRangeMessage.self)
    send(encoder as AnyObject, NSSelectorFromString("optimizeIndirectCommandBuffer:withRange:"),
         buffer as AnyObject, NSRange(location: range.lowerBound, length: range.count))
}

private func copyCommands(_ encoder: MTLBlitCommandEncoder,
                          _ source: MTLIndirectCommandBuffer,
                          _ sourceRange: Range<Int>,
                          _ destination: MTLIndirectCommandBuffer,
                          _ destinationIndex: Int) {
    let send = unsafeBitCast(objcMessage, to: ICBCopyMessage.self)
    send(encoder as AnyObject,
         NSSelectorFromString("copyIndirectCommandBuffer:sourceRange:destination:destinationIndex:"),
         source as AnyObject,
         NSRange(location: sourceRange.lowerBound, length: sourceRange.count),
         destination as AnyObject,
         UInt(destinationIndex))
}

private func makeComputeICB(_ pipe: MTLComputePipelineState,
                            _ output: MTLBuffer,
                            _ values: MTLBuffer,
                            _ count: Int) -> MTLIndirectCommandBuffer? {
    let descriptor = MTLIndirectCommandBufferDescriptor()
    descriptor.commandTypes = .concurrentDispatch
    descriptor.inheritBuffers = false
    descriptor.inheritPipelineState = false
    descriptor.maxKernelBufferBindCount = 2
    guard let icb = dev.makeIndirectCommandBuffer(descriptor: descriptor,
                                                   maxCommandCount: count,
                                                   options: []) else {
        return nil
    }
    for index in 0..<count {
        let command = icb.indirectComputeCommandAt(index)
        command.setComputePipelineState(pipe)
        command.setKernelBuffer(output, offset: index * 4, at: 0)
        command.setKernelBuffer(values, offset: index * 4, at: 1)
        command.concurrentDispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                               threadsPerThreadgroup:
                                                   MTLSize(width: 1, height: 1, depth: 1))
    }
    return icb
}

private func executeComputeICB(_ icb: MTLIndirectCommandBuffer,
                               _ output: MTLBuffer,
                               _ values: MTLBuffer,
                               _ count: Int) -> Bool {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return false
    }
    encoder.useResource(output, usage: .write)
    encoder.useResource(values, usage: .read)
    encoder.executeCommandsInBuffer(icb, range: 0..<count)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.status == .completed
}

func indirectCommandMutationCase() {
    let label = "indirect_command_reset_copy_optimize"
    let count = 3
    let byteCount = count * MemoryLayout<UInt32>.size
    guard let pipe = makeIndirectPipeline() else {
        report(label, false, "indirect-capable compute pipeline unavailable")
        return
    }
    guard let output = dev.makeBuffer(length: byteCount, options: .storageModeShared),
          let sourceValues = dev.makeBuffer(length: byteCount, options: .storageModeShared),
          let destinationValues = dev.makeBuffer(length: byteCount, options: .storageModeShared)
    else {
        report(label, false, "buffers unavailable")
        return
    }
    let sourceWords = sourceValues.contents().bindMemory(to: UInt32.self, capacity: count)
    let destinationWords = destinationValues.contents().bindMemory(to: UInt32.self, capacity: count)
    for index in 0..<count {
        sourceWords[index] = UInt32(11 * (index + 1))
        destinationWords[index] = UInt32(90 + index)
    }
    guard let source = makeComputeICB(pipe, output, sourceValues, count),
          let destination = makeComputeICB(pipe, output, destinationValues, count) else {
        report(label, false, "indirect command buffers unavailable")
        return
    }

    memset(output.contents(), 0, byteCount)
    guard executeComputeICB(source, output, sourceValues, count) else {
        report(label, false, "baseline indirect execution failed")
        return
    }
    let outputWords = output.contents().bindMemory(to: UInt32.self, capacity: count)
    let baseline = Array(UnsafeBufferPointer(start: outputWords, count: count))

    guard let mutate = queue.makeCommandBuffer(),
          let blit = mutate.makeBlitCommandEncoder() else {
        report(label, false, "mutation blit encoder unavailable")
        return
    }
    resetCommands(blit, source, 1..<2)
    copyCommands(blit, source, 0..<count, destination, 0)
    optimizeCommands(blit, destination, 0..<count)
    blit.endEncoding()
    mutate.commit()
    mutate.waitUntilCompleted()
    guard mutate.status == .completed else {
        report(label, false, "mutation command buffer status=\(mutate.status.rawValue)")
        return
    }

    memset(output.contents(), 0, byteCount)
    guard executeComputeICB(destination, output, destinationValues, count) else {
        report(label, false, "destination indirect execution failed")
        return
    }
    let mutated = Array(UnsafeBufferPointer(start: outputWords, count: count))
    let expectedBaseline: [UInt32] = [11, 22, 33]
    let expectedMutated: [UInt32] = [11, 0, 33]
    report(label, baseline == expectedBaseline && mutated == expectedMutated,
           "baseline=\(baseline) mutated=\(mutated)")
}
