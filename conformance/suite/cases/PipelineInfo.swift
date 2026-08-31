import Metal
import Foundation

/// Pipeline information is construction-owned state. In particular, the
/// indirect-command support bit follows the descriptor used to compile the
/// object, while static threadgroup memory follows the compiled kernel rather
/// than a device-wide limit.
func pipelineInformationCase() {
    let name = "pipeline_information_lifetime"
    let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void plain_kernel(device uint *out [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
        if (tid == 0) out[0] = 1;
    }

    kernel void static_kernel(device uint *out [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
        threadgroup uint values[17];
        if (tid < 17) values[tid] = tid;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) out[0] = values[16];
    }

    struct VertexOut { float4 position [[position]]; };
    vertex VertexOut info_vertex(uint id [[vertex_id]]) {
        VertexOut out;
        out.position = float4(id == 0 ? -1.0 : 1.0,
                              id == 2 ? 1.0 : -1.0, 0.0, 1.0);
        return out;
    }
    fragment float4 info_fragment() { return float4(1.0); }
    """

    do {
        let infoLibrary = try dev.makeLibrary(source: source, options: nil)
        func compute(_ function: String, _ indirect: Bool) throws -> MTLComputePipelineState {
            let descriptor = MTLComputePipelineDescriptor()
            descriptor.computeFunction = infoLibrary.makeFunction(name: function)
            descriptor.supportIndirectCommandBuffers = indirect
            return try dev.makeComputePipelineState(
                descriptor: descriptor,
                options: [],
                reflection: nil)
        }
        func render(_ indirect: Bool) throws -> MTLRenderPipelineState {
            let descriptor = MTLRenderPipelineDescriptor()
            descriptor.vertexFunction = infoLibrary.makeFunction(name: "info_vertex")
            descriptor.fragmentFunction = infoLibrary.makeFunction(name: "info_fragment")
            descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
            descriptor.supportIndirectCommandBuffers = indirect
            return try dev.makeRenderPipelineState(descriptor: descriptor)
        }

        let plain = try compute("plain_kernel", false)
        let staticMemory = try compute("static_kernel", false)
        let indirectCompute = try compute("plain_kernel", true)
        let directRender = try render(false)
        let indirectRender = try render(true)
        let ok = plain.maxTotalThreadsPerThreadgroup > 0
            && plain.threadExecutionWidth > 0
            && plain.staticThreadgroupMemoryLength < staticMemory.staticThreadgroupMemoryLength
            && !plain.supportIndirectCommandBuffers
            && indirectCompute.supportIndirectCommandBuffers
            && !directRender.supportIndirectCommandBuffers
            && indirectRender.supportIndirectCommandBuffers
            && directRender.imageblockSampleLength > 0
        report(
            name,
            ok,
            "compute_max=\(plain.maxTotalThreadsPerThreadgroup) "
                + "width=\(plain.threadExecutionWidth) "
                + "static_plain=\(plain.staticThreadgroupMemoryLength) "
                + "static_declared=\(staticMemory.staticThreadgroupMemoryLength) "
                + "render_imageblock=\(directRender.imageblockSampleLength)")
    } catch {
        report(name, false, "pipeline creation failed: \(error)")
    }
}
