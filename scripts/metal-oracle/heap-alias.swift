import Foundation
import Metal

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() {
        fputs("FAIL \(message)\n", stderr)
        exit(1)
    }
}

guard let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else {
    fputs("FAIL no_metal_device_or_queue\n", stderr)
    exit(1)
}

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void fill_rgba8(texture2d<float, access::write> output [[texture(0)]],
                       constant float4 &value [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x < output.get_width() && gid.y < output.get_height()) {
        output.write(value, gid);
    }
}
"""
let library = try device.makeLibrary(source: source, options: nil)
guard let function = library.makeFunction(name: "fill_rgba8") else {
    fputs("FAIL no_fill_function\n", stderr)
    exit(1)
}
let pipeline = try device.makeComputePipelineState(function: function)

let width = 257
let height = 193
let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
    pixelFormat: .rgba8Unorm,
    width: width,
    height: height,
    mipmapped: false
)
textureDescriptor.storageMode = .private
textureDescriptor.usage = [.shaderRead, .shaderWrite]

let requirement = device.heapTextureSizeAndAlign(descriptor: textureDescriptor)
require(requirement.size > 0, "zero_heap_texture_size")
require(requirement.align > 0 && requirement.align.nonzeroBitCount == 1,
        "bad_heap_texture_alignment_\(requirement.align)")

let heapDescriptor = MTLHeapDescriptor()
heapDescriptor.type = .automatic
heapDescriptor.storageMode = .private
heapDescriptor.hazardTrackingMode = .untracked
heapDescriptor.size = requirement.size
guard let heap = device.makeHeap(descriptor: heapDescriptor),
      let first = heap.makeTexture(descriptor: textureDescriptor) else {
    fputs("FAIL heap_or_first_texture_unavailable\n", stderr)
    exit(1)
}
let firstOffset = first.heapOffset
require(heap.makeTexture(descriptor: textureDescriptor) == nil,
        "one_slot_heap_admitted_second_live_texture")

func fill(_ texture: MTLTexture, _ value: SIMD4<Float>) {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        fputs("FAIL no_compute_encoder\n", stderr)
        exit(1)
    }
    var color = value
    encoder.setComputePipelineState(pipeline)
    encoder.setTexture(texture, index: 0)
    encoder.setBytes(&color, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
    let group = MTLSize(width: 8, height: 8, depth: 1)
    encoder.dispatchThreads(
        MTLSize(width: width, height: height, depth: 1),
        threadsPerThreadgroup: group
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    require(commandBuffer.status == .completed, "fill_command_failed")
}

fill(first, SIMD4<Float>(1, 0, 0, 1))
first.makeAliasable()
require(first.isAliasable(), "first_texture_not_aliasable")
guard let second = heap.makeTexture(descriptor: textureDescriptor) else {
    fputs("FAIL alias_texture_unavailable\n", stderr)
    exit(1)
}
require(second.heapOffset == firstOffset,
        "alias_offset_moved_\(firstOffset)_\(second.heapOffset)")
fill(second, SIMD4<Float>(0, 1, 0, 1))

let bytesPerRow = ((width * 4 + 255) / 256) * 256
guard let readback = device.makeBuffer(
    length: bytesPerRow * height,
    options: .storageModeShared
), let commandBuffer = queue.makeCommandBuffer(),
   let blit = commandBuffer.makeBlitCommandEncoder() else {
    fputs("FAIL no_readback_resources\n", stderr)
    exit(1)
}
blit.copy(
    from: second,
    sourceSlice: 0,
    sourceLevel: 0,
    sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
    sourceSize: MTLSize(width: width, height: height, depth: 1),
    to: readback,
    destinationOffset: 0,
    destinationBytesPerRow: bytesPerRow,
    destinationBytesPerImage: bytesPerRow * height
)
blit.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
require(commandBuffer.status == .completed, "readback_command_failed")

let bytes = readback.contents().bindMemory(to: UInt8.self, capacity: bytesPerRow * height)
for y in 0..<height {
    for x in 0..<width {
        let at = y * bytesPerRow + x * 4
        require(bytes[at] == 0 && bytes[at + 1] == 255 &&
                bytes[at + 2] == 0 && bytes[at + 3] == 255,
                "wrong_texel_\(x)_\(y)_\(bytes[at])_\(bytes[at + 1])_\(bytes[at + 2])_\(bytes[at + 3])")
    }
}

print("RESULT heap_alias=OK")
print("RESULT texture_size=\(requirement.size) texture_align=\(requirement.align)")
print("RESULT reused_offset=\(second.heapOffset)")
