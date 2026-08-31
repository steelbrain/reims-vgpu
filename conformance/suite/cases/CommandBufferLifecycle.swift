import Metal
import Foundation
import Dispatch

// Two committed command buffers from one command queue execute in commit
// order. The first produces bytes and the second consumes them; waiting only
// for the consumer therefore observes both command buffers' effects.
func sameQueueCommandBufferOrderCase() {
    let label = "command_buffer_same_queue_order"
    guard let source = dev.makeBuffer(length: 64, options: .storageModeShared),
          let destination = dev.makeBuffer(length: 64, options: .storageModeShared),
          let producer = queue.makeCommandBuffer(),
          let consumer = queue.makeCommandBuffer(),
          let produce = producer.makeBlitCommandEncoder(),
          let consume = consumer.makeBlitCommandEncoder()
    else {
        report(label, false, "buffers, command buffers, or blit encoders unavailable")
        return
    }

    source.contents().initializeMemory(as: UInt8.self, repeating: 0, count: 64)
    destination.contents().initializeMemory(as: UInt8.self, repeating: 0, count: 64)
    produce.fill(buffer: source, range: 0..<64, value: 0xa5)
    produce.endEncoding()
    consume.copy(from: source, sourceOffset: 0,
                 to: destination, destinationOffset: 0, size: 64)
    consume.endEncoding()

    producer.commit()
    consumer.commit()
    consumer.waitUntilCompleted()

    let bytes = destination.contents().bindMemory(to: UInt8.self, capacity: 64)
    let ordered = (0..<64).allSatisfy { bytes[$0] == 0xa5 }
    let complete = producer.status == .completed && consumer.status == .completed
    report(label, ordered && complete,
           "producer=\(producer.status.rawValue) consumer=\(consumer.status.rawValue) ordered=\(ordered)")
}

// Completed handlers from command buffers committed to one queue are
// published in command-buffer order. This is a publication observation, not
// an inference from buffer contents.
func sameQueueCompletionPublicationCase() {
    let label = "command_buffer_same_queue_completion_publication"
    guard let first = queue.makeCommandBuffer(),
          let second = queue.makeCommandBuffer()
    else {
        report(label, false, "command buffers unavailable")
        return
    }

    let lock = NSLock()
    var published: [Int] = []
    first.addCompletedHandler { _ in
        lock.lock()
        published.append(1)
        lock.unlock()
    }
    second.addCompletedHandler { _ in
        lock.lock()
        published.append(2)
        lock.unlock()
    }

    first.commit()
    second.commit()
    second.waitUntilCompleted()

    lock.lock()
    let observed = published
    lock.unlock()
    report(label, observed == [1, 2], "published=\(observed)")
}

// A command buffer with no encoders still has a scheduled/completed lifecycle
// and callback publication obligation.
func emptyCommandBufferCompletionCase() {
    let label = "command_buffer_empty_completion"
    guard let commandBuffer = queue.makeCommandBuffer() else {
        report(label, false, "command buffer unavailable")
        return
    }
    let lock = NSLock()
    var callback = false
    commandBuffer.addCompletedHandler { _ in
        lock.lock()
        callback = true
        lock.unlock()
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    lock.lock()
    let published = callback
    lock.unlock()
    report(label, commandBuffer.status == .completed && published,
           "status=\(commandBuffer.status.rawValue) callback=\(published)")
}

// An explicit shared-event edge orders work across two command queues and
// carries the producer's writes into the consumer.
func crossQueueSharedEventVisibilityCase() {
    let label = "command_buffer_cross_queue_shared_event_visibility"
    guard let producerQueue = dev.makeCommandQueue(),
          let consumerQueue = dev.makeCommandQueue(),
          let event = dev.makeSharedEvent(),
          let source = dev.makeBuffer(length: 64, options: .storageModeShared),
          let destination = dev.makeBuffer(length: 64, options: .storageModeShared),
          let producer = producerQueue.makeCommandBuffer(),
          let consumer = consumerQueue.makeCommandBuffer(),
          let produce = producer.makeBlitCommandEncoder()
    else {
        report(label, false, "queues, event, buffers, command buffers, or encoders unavailable")
        return
    }

    source.contents().initializeMemory(as: UInt8.self, repeating: 0, count: 64)
    destination.contents().initializeMemory(as: UInt8.self, repeating: 0, count: 64)
    produce.fill(buffer: source, range: 0..<64, value: 0x6d)
    produce.endEncoding()
    producer.encodeSignalEvent(event, value: 1)
    consumer.encodeWaitForEvent(event, value: 1)
    guard let consume = consumer.makeBlitCommandEncoder() else {
        report(label, false, "consumer blit encoder unavailable")
        return
    }
    consume.copy(from: source, sourceOffset: 0,
                 to: destination, destinationOffset: 0, size: 64)
    consume.endEncoding()

    // Commit the waiter first: the explicit future dependency must remain
    // valid and must not be replaced with commit-arrival order.
    let completion = DispatchSemaphore(value: 0)
    consumer.addCompletedHandler { _ in completion.signal() }
    consumer.commit()
    producer.commit()
    let completedWithoutHostRelease = completion.wait(timeout: .now() + 2) == .success
    if !completedWithoutHostRelease {
        // A waiter-first scheduling defect can park the device-wide command
        // stream. Even publishing a CPU signal may synchronously wait behind
        // that stream, so report without another device call or a join. This
        // case runs last so a broken implementation cannot hide later cases.
        report(label, false, "completed_without_host_release=false visible=false")
        return
    }
    consumer.waitUntilCompleted()
    let bytes = destination.contents().bindMemory(to: UInt8.self, capacity: 64)
    let visible = (0..<64).allSatisfy { bytes[$0] == 0x6d }
    report(label,
           completedWithoutHostRelease && visible
               && producer.status == .completed && consumer.status == .completed,
           "producer=\(producer.status.rawValue) consumer=\(consumer.status.rawValue) "
               + "completed_without_host_release=\(completedWithoutHostRelease) visible=\(visible)")
}

// Publication is not globally serialized across independent API queues. A
// waiter on one queue cannot prevent an empty command buffer on another queue
// from completing and invoking its handler.
func independentQueueCompletionPublicationCase() {
    let label = "command_buffer_independent_queue_publication"
    guard let blockedQueue = dev.makeCommandQueue(),
          let independentQueue = dev.makeCommandQueue(),
          let event = dev.makeSharedEvent(),
          let blocked = blockedQueue.makeCommandBuffer(),
          let independent = independentQueue.makeCommandBuffer()
    else {
        report(label, false, "queues, event, or command buffers unavailable")
        return
    }

    blocked.encodeWaitForEvent(event, value: 1)
    let completed = DispatchSemaphore(value: 0)
    independent.addCompletedHandler { _ in completed.signal() }
    blocked.commit()
    independent.commit()
    let independentPublishedFirst = completed.wait(timeout: .now() + 2) == .success
    // Always release the waiter before joining either command buffer.
    event.signaledValue = 1
    blocked.waitUntilCompleted()
    independent.waitUntilCompleted()
    report(label, independentPublishedFirst,
           "independent_published_before_signal=\(independentPublishedFirst)")
}

// Scheduled publication precedes completed publication for one command
// buffer. Both handlers are obligations even when the host completes quickly.
func scheduledBeforeCompletedPublicationCase() {
    let label = "command_buffer_scheduled_before_completed"
    guard let commandBuffer = queue.makeCommandBuffer() else {
        report(label, false, "command buffer unavailable")
        return
    }
    let lock = NSLock()
    var publications: [String] = []
    commandBuffer.addScheduledHandler { _ in
        lock.lock()
        publications.append("scheduled")
        lock.unlock()
    }
    commandBuffer.addCompletedHandler { _ in
        lock.lock()
        publications.append("completed")
        lock.unlock()
    }
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    lock.lock()
    let observed = publications
    lock.unlock()
    report(label, observed == ["scheduled", "completed"], "publications=\(observed)")
}

// A normal command buffer retains resources referenced by encoded work. The
// event deliberately parks execution so the observation is made before GPU
// completion rather than after an accidentally fast fill.
func defaultCommandBufferRetainsResourceCase() {
    let label = "command_buffer_default_retains_resource"
    guard let event = dev.makeSharedEvent(),
          let commandBuffer = queue.makeCommandBuffer()
    else {
        report(label, false, "event or command buffer unavailable")
        return
    }
    var buffer: MTLBuffer? = dev.makeBuffer(length: 64, options: .storageModeShared)
    weak var retainedBuffer: MTLBuffer? = buffer
    guard buffer != nil else {
        report(label, false, "buffer unavailable")
        return
    }
    commandBuffer.encodeWaitForEvent(event, value: 1)
    var encoder: MTLBlitCommandEncoder? = commandBuffer.makeBlitCommandEncoder()
    guard encoder != nil else {
        report(label, false, "blit encoder unavailable")
        return
    }
    encoder!.fill(buffer: buffer!, range: 0..<64, value: 0x3c)
    encoder!.endEncoding()
    encoder = nil
    commandBuffer.commit()
    buffer = nil
    let retainedWhilePending = retainedBuffer != nil
    event.signaledValue = 1
    commandBuffer.waitUntilCompleted()
    report(label, retainedWhilePending && commandBuffer.status == .completed,
           "retained_while_pending=\(retainedWhilePending) status=\(commandBuffer.status.rawValue)")
}
