import Metal
import Foundation

private func fillHeapTexture(_ texture: MTLTexture, _ value: SIMD4<Float>) -> Bool {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
        return false
    }
    var color = value
    encoder.setComputePipelineState(pipeline("heap_alias_fill"))
    encoder.setTexture(texture, index: 0)
    encoder.setBytes(&color, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
    encoder.dispatchThreads(
        MTLSize(width: texture.width, height: texture.height, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    return commandBuffer.status == .completed
}

// A one-slot automatic heap has exactly one legal reuse transition: after the
// live texture occupying it becomes aliasable, a second compatible texture may
// take the same offset. The final readback covers both halves of the contract:
// the allocator reused the released range, and commands address the new
// resource rather than stale storage owned by the old one.
func heapTextureAliasCase() {
    let label = "heap_texture_alias_lifecycle"
    let width = 257
    let height = 193
    let descriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: width,
        height: height,
        mipmapped: false)
    descriptor.storageMode = .private
    descriptor.usage = [.shaderRead, .shaderWrite]

    let requirement = dev.heapTextureSizeAndAlign(descriptor: descriptor)
    guard requirement.size > 0, requirement.align > 0 else {
        skip(label, "the device reported no heap storage requirement for the texture")
        return
    }
    guard requirement.align.nonzeroBitCount == 1 else {
        report(label, false, "heap alignment is not a power of two: \(requirement.align)")
        return
    }

    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.type = .automatic
    heapDescriptor.storageMode = .private
    heapDescriptor.hazardTrackingMode = .untracked
    heapDescriptor.size = requirement.size
    guard let heap = dev.makeHeap(descriptor: heapDescriptor) else {
        report(label, false,
               "nonzero requirement size=\(requirement.size) align=\(requirement.align), but makeHeap returned nil")
        return
    }
    guard let first = heap.makeTexture(descriptor: descriptor) else {
        report(label, false, "the exact-size heap would not allocate its first texture")
        return
    }
    let firstOffset = first.heapOffset
    guard heap.makeTexture(descriptor: descriptor) == nil else {
        report(label, false, "a one-slot heap admitted two simultaneously live textures")
        return
    }
    guard fillHeapTexture(first, SIMD4<Float>(1, 0, 0, 1)) else {
        report(label, false, "commands using the first heap texture did not complete")
        return
    }

    first.makeAliasable()
    guard first.isAliasable() else {
        report(label, false, "makeAliasable did not change the first texture's lifecycle state")
        return
    }
    guard let second = heap.makeTexture(descriptor: descriptor) else {
        report(label, false, "the released heap range was not reusable")
        return
    }
    guard second.heapOffset == firstOffset else {
        report(label, false,
               "the replacement moved from offset=\(firstOffset) to offset=\(second.heapOffset)")
        return
    }
    guard fillHeapTexture(second, SIMD4<Float>(0, 1, 0, 1)) else {
        report(label, false, "commands using the replacement heap texture did not complete")
        return
    }
    guard let got = readBack(readPipe, second, width, height) else {
        refused(label)
        return
    }

    let want = pack(0, 255, 0, 255)
    let bad = got.indices.filter { got[$0] != want }
    report(label, bad.isEmpty,
           bad.isEmpty
             ? "one live range was released, reused at offset=\(second.heapOffset), and addressed as the replacement"
             : "wrong=\(bad.count)/\(got.count) first=\(hex(got[bad[0]])) want=\(hex(want))")
}

// Placement heaps let the caller name overlapping ranges directly. Merely
// creating both resources is defined; only concurrent GPU access needs an
// explicit synchronization discipline. Keep both objects live so an
// implementation that treats overlap as an allocator collision cannot pass by
// retiring the first one implicitly.
func heapTexturePlacementOverlapCase() {
    let label = "heap_texture_placement_overlap"
    let width = 64
    let height = 64
    let unorm = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: width,
        height: height,
        mipmapped: false)
    let bgra = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .bgra8Unorm,
        width: width,
        height: height,
        mipmapped: false)
    for descriptor in [unorm, bgra] {
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
    }

    let unormRequirement = dev.heapTextureSizeAndAlign(descriptor: unorm)
    let bgraRequirement = dev.heapTextureSizeAndAlign(descriptor: bgra)
    guard unormRequirement.size > 0, unormRequirement.align > 0,
          bgraRequirement.size > 0, bgraRequirement.align > 0 else {
        skip(label, "the device reported no heap storage requirement for one texture")
        return
    }

    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.type = .placement
    heapDescriptor.storageMode = .private
    heapDescriptor.hazardTrackingMode = .untracked
    heapDescriptor.size = max(unormRequirement.size, bgraRequirement.size)
    guard let heap = dev.makeHeap(descriptor: heapDescriptor) else {
        report(label, false, "the placement heap could not be created")
        return
    }
    guard let first = heap.makeTexture(descriptor: unorm, offset: 0) else {
        report(label, false, "the placement heap refused its first texture")
        return
    }
    guard let alias = heap.makeTexture(descriptor: bgra, offset: 0) else {
        report(label, false, "the placement heap refused a live overlapping texture")
        return
    }

    guard fillHeapTexture(first, SIMD4<Float>(1, 0, 0, 1)) else {
        report(label, false, "commands using the first placement texture did not complete")
        return
    }
    guard let aliased = readBack(readPipe, alias, width, height) else {
        refused(label)
        return
    }
    let blue = pack(0, 0, 255, 255)
    guard aliased.allSatisfy({ $0 == blue }) else {
        let wrong = aliased.filter { $0 != blue }
        report(label, false,
               "the BGRA alias did not observe RGBA bytes: wrong=\(wrong.count)/\(aliased.count)")
        return
    }
    guard fillHeapTexture(alias, SIMD4<Float>(0, 1, 0, 1)) else {
        report(label, false, "commands using the overlapping placement texture did not complete")
        return
    }
    guard let got = readBack(readPipe, alias, width, height) else {
        refused(label)
        return
    }

    let want = pack(0, 255, 0, 255)
    let bad = got.indices.filter { got[$0] != want }
    report(label, first.heapOffset == 0 && alias.heapOffset == 0 && bad.isEmpty,
           bad.isEmpty
             ? "two live definitions at offset=0 executed sequentially and addressed the replacement"
             : "wrong=\(bad.count)/\(got.count) first=\(hex(got[bad[0]])) want=\(hex(want))")
}

// The same overlap, between formats no single image can serve.
//
// `heap_texture_placement_overlap` aliases two 32-bit formats, which one
// native image plus a reinterpreting view can answer. Nothing here can be:
// rgba16Float and rgba8Unorm are different bit widths, so an implementation
// that gives a range of storage one image and reinterprets it for the other
// alias has no view to build and must refuse a binding the API defines.
//
// Metal leaves an alias's *contents* undefined after another alias writes, so
// this claims nothing about bytes crossing between them. It claims only that
// each live texture addresses its own definition: fill one, read it back, and
// the value written through that texture is the value that texture reads.
func heapTexturePlacementOverlapWidthCase() {
    let label = "heap_texture_placement_overlap_incompatible_widths"
    let width = 64
    let height = 64
    let wide = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba16Float,
        width: width,
        height: height,
        mipmapped: false)
    let narrow = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: width,
        height: height,
        mipmapped: false)
    for descriptor in [wide, narrow] {
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
    }

    let wideRequirement = dev.heapTextureSizeAndAlign(descriptor: wide)
    let narrowRequirement = dev.heapTextureSizeAndAlign(descriptor: narrow)
    guard wideRequirement.size > 0, wideRequirement.align > 0,
          narrowRequirement.size > 0, narrowRequirement.align > 0 else {
        skip(label, "the device reported no heap storage requirement for one texture")
        return
    }

    let heapDescriptor = MTLHeapDescriptor()
    heapDescriptor.type = .placement
    heapDescriptor.storageMode = .private
    heapDescriptor.hazardTrackingMode = .untracked
    heapDescriptor.size = max(wideRequirement.size, narrowRequirement.size)
    guard let heap = dev.makeHeap(descriptor: heapDescriptor) else {
        report(label, false, "the placement heap could not be created")
        return
    }
    guard let wideTexture = heap.makeTexture(descriptor: wide, offset: 0) else {
        report(label, false, "the placement heap refused its rgba16Float texture")
        return
    }
    guard let narrowTexture = heap.makeTexture(descriptor: narrow, offset: 0) else {
        report(label, false, "the placement heap refused a live overlapping rgba8Unorm texture")
        return
    }

    // Components of exactly 0 and 1 survive both storage formats and the
    // readback's fixed-point packing, so a mismatch is the alias and never the
    // arithmetic.
    guard fillHeapTexture(wideTexture, SIMD4<Float>(1, 0, 0, 1)) else {
        report(label, false, "commands using the rgba16Float alias did not complete")
        return
    }
    guard let wideGot = readBack(readPipe, wideTexture, width, height) else {
        refused(label)
        return
    }
    let red = pack(255, 0, 0, 255)
    let wideBad = wideGot.indices.filter { wideGot[$0] != red }
    guard wideBad.isEmpty else {
        report(label, false,
               "the rgba16Float alias did not read its own write: "
                 + "wrong=\(wideBad.count)/\(wideGot.count) first=\(hex(wideGot[wideBad[0]]))")
        return
    }

    guard fillHeapTexture(narrowTexture, SIMD4<Float>(0, 1, 0, 1)) else {
        report(label, false, "commands using the rgba8Unorm alias did not complete")
        return
    }
    guard let narrowGot = readBack(readPipe, narrowTexture, width, height) else {
        refused(label)
        return
    }
    let green = pack(0, 255, 0, 255)
    let narrowBad = narrowGot.indices.filter { narrowGot[$0] != green }
    report(label,
           wideTexture.heapOffset == 0 && narrowTexture.heapOffset == 0 && narrowBad.isEmpty,
           narrowBad.isEmpty
             ? "two live textures of different bit widths at offset=0 each addressed their own definition"
             : "the rgba8Unorm alias did not read its own write: "
                 + "wrong=\(narrowBad.count)/\(narrowGot.count) first=\(hex(narrowGot[narrowBad[0]]))")
}
