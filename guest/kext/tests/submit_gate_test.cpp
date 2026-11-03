// The native test of RgfxSubmitGate.hpp: submitMethod accepts only the object-lifecycle types
// strictly between QUEUE_CREATE and CONTEXT_CLOSE, and of those only OBJECT_DESTROY (a queue's)
// must be routed through the queue-destroy path rather than posted to the device raw.
#include "RgfxSubmitGate.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "submit_gate_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// The boundaries: QUEUE_CREATE is made through its own selector, CONTEXT_CLOSE and the
	// block messages above it are the kext's; neither end is accepted through submit.
	check(!rgfxSubmitAdmitted(RGFX_CMD_QUEUE_CREATE), "QUEUE_CREATE is not a submit message");
	check(!rgfxSubmitAdmitted(RGFX_CMD_CONTEXT_CLOSE), "CONTEXT_CLOSE is the kext's");
	check(!rgfxSubmitAdmitted(RGFX_CMD_RESOURCE_ID_CHUNK), "RESOURCE_ID_CHUNK is the kext's");
	check(!rgfxSubmitAdmitted(RGFX_CMD_BLOCK_RELEASE), "BLOCK_RELEASE is the kext's");
	check(!rgfxSubmitAdmitted(RGFX_CMD_BLOCK_GRANT), "BLOCK_GRANT is the kext's");
	check(!rgfxSubmitAdmitted(RGFX_CMD_GET_DISPLAYS), "a display command is not a submit message");
	check(!rgfxSubmitAdmitted(RGFX_CMD_PRESENT_SURFACE), "PRESENT_SURFACE is not one");
	check(!rgfxSubmitAdmitted(RGFX_CMD_COMMAND_BUFFER), "a command buffer rides a queue's ring");

	// The accepted range: every type strictly between QUEUE_CREATE and CONTEXT_CLOSE.
	check(rgfxSubmitAdmitted(RGFX_CMD_OBJECT_DESTROY), "OBJECT_DESTROY is accepted");
	check(rgfxSubmitAdmitted(RGFX_CMD_BUFFER_OVER_HOST_MEMORY), "a buffer over host memory is");
	check(rgfxSubmitAdmitted(RGFX_CMD_RENDER_PIPELINE_CREATE), "a pipeline create is");
	check(rgfxSubmitAdmitted(RGFX_CMD_VIDEO_SESSION_CREATE), "the last accepted type is");

	// Only OBJECT_DESTROY, with a tracked queue's id, routes through the queue-destroy path.
	// Every other accepted type names an object only the host tracks and is posted as it is.
	check(rgfxSubmitRoutesQueueDestroy(RGFX_CMD_OBJECT_DESTROY),
	      "OBJECT_DESTROY of a tracked queue routes");
	for (uint32_t type = RGFX_CMD_QUEUE_CREATE; type <= RGFX_CMD_CONTEXT_CLOSE; type++) {
		if (type == RGFX_CMD_OBJECT_DESTROY)
			continue;
		check(!rgfxSubmitRoutesQueueDestroy(type),
		      "no other type in the range routes as a queue destroy");
	}
	// A type the gate rejects never reaches the routing question, but is not a queue destroy.
	check(!rgfxSubmitRoutesQueueDestroy(RGFX_CMD_QUEUE_CREATE), "QUEUE_CREATE does not route");
	check(!rgfxSubmitRoutesQueueDestroy(RGFX_CMD_BLOCK_RELEASE), "BLOCK_RELEASE does not route");

	// Which accepted types make an object a destroy then drops: every type of the accepted
	// range is decided here, so a new wire type has to be placed.
	const uint32_t makes[] = {
	    RGFX_CMD_HEAP_CREATE, RGFX_CMD_HEAP_OVER_HOST_MEMORY, RGFX_CMD_BUFFER_CREATE,
	    RGFX_CMD_BUFFER_OVER_HOST_MEMORY, RGFX_CMD_BUFFER_AT_OFFSET, RGFX_CMD_TEXTURE_CREATE,
	    RGFX_CMD_TEXTURE_FROM_BUFFER, RGFX_CMD_TEXTURE_FROM_IOSURFACE, RGFX_CMD_SAMPLER_CREATE,
	    RGFX_CMD_DEPTH_STENCIL_CREATE, RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_RENDER_PIPELINE_CREATE,
	    RGFX_CMD_COMPUTE_PIPELINE_CREATE, RGFX_CMD_EVENT_CREATE, RGFX_CMD_SHARED_EVENT_CREATE,
	    RGFX_CMD_FENCE_CREATE, RGFX_CMD_LIBRARY_FROM_REGION, RGFX_CMD_LIBRARY_LOOKUP,
	    RGFX_CMD_MANAGED_TEXTURE_CREATE, RGFX_CMD_TEXTURE_VIEW, RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER,
	    RGFX_CMD_BUFFER_OVER_SURFACE_PAGES, RGFX_CMD_VIDEO_SESSION_CREATE, RGFX_CMD_ICB_CREATE,
	    RGFX_CMD_BUFFER_MANAGED_AT_OFFSET,
	};
	const uint32_t makesNone[] = {
	    RGFX_CMD_OBJECT_DESTROY, RGFX_CMD_BUFFER_DID_MODIFY_RANGE, RGFX_CMD_TEXTURE_REPLACE_REGION,
	    RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN, RGFX_CMD_ARGUMENT_LAYOUT,
	    RGFX_CMD_SHARED_EVENT_SET_VALUE, RGFX_CMD_ICB_WRITE,
	};
	for (uint32_t t : makes)
		check(rgfxSubmitAdmitted(t) && rgfxSubmitMakes(t), "a create makes an object");
	for (uint32_t t : makesNone)
		check(rgfxSubmitAdmitted(t) && !rgfxSubmitMakes(t),
		      "a destroy, a query or a write into an object makes none");
	for (uint32_t type = RGFX_CMD_QUEUE_CREATE + 1; type < RGFX_CMD_CONTEXT_CLOSE; type++) {
		bool listed = false;
		for (uint32_t t : makes)
			listed |= t == type;
		for (uint32_t t : makesNone)
			listed |= t == type;
		if (type <= RGFX_CMD_VIDEO_SESSION_CREATE || type == RGFX_CMD_BUFFER_MANAGED_AT_OFFSET)
			check(listed, "every wire type of the accepted range is placed");
		else if (!listed)
			check(!rgfxSubmitMakes(type), "a type the wire does not name makes none");
	}

	// A destroy follows a create only where it was posted and made something, and either timed
	// out or its context closed since the submit's check.
	check(rgfxSubmitDestroysBehind(RGFX_CMD_TEXTURE_CREATE, true, false, true),
	      "a span-less create posted across its context's close is destroyed behind");
	check(rgfxSubmitDestroysBehind(RGFX_CMD_RENDER_PIPELINE_CREATE, true, false, true),
	      "a pipeline create across the close too");
	check(rgfxSubmitDestroysBehind(RGFX_CMD_SAMPLER_CREATE, true, true, false),
	      "a create not answered in time is destroyed behind");
	check(!rgfxSubmitDestroysBehind(RGFX_CMD_TEXTURE_CREATE, true, false, false),
	      "an answered create of an open context is kept");
	check(!rgfxSubmitDestroysBehind(RGFX_CMD_TEXTURE_CREATE, false, false, true),
	      "nothing follows a message never posted");
	check(!rgfxSubmitDestroysBehind(RGFX_CMD_ARGUMENT_LAYOUT, true, true, true),
	      "a query makes nothing, so nothing follows it");
	check(!rgfxSubmitDestroysBehind(RGFX_CMD_OBJECT_DESTROY, true, false, true),
	      "a destroy is not destroyed behind");
	check(!rgfxSubmitDestroysBehind(RGFX_CMD_SHARED_EVENT_SET_VALUE, true, false, true),
	      "nor a write into an object");
	check(!rgfxSubmitDestroysBehind(RGFX_CMD_ICB_WRITE, true, true, true),
	      "nor an indirect command buffer's commands");
	// The timeout path is as before for every type that waits for an answer.
	const uint32_t answered[] = {RGFX_CMD_RENDER_PIPELINE_CREATE, RGFX_CMD_COMPUTE_PIPELINE_CREATE,
	                             RGFX_CMD_SAMPLER_CREATE, RGFX_CMD_ICB_CREATE, RGFX_CMD_LIBRARY_CREATE,
	                             RGFX_CMD_LIBRARY_LOOKUP, RGFX_CMD_LIBRARY_FROM_REGION};
	for (uint32_t t : answered)
		check(rgfxSubmitDestroysBehind(t, true, true, false),
		      "an answered create that times out is destroyed behind, as before");
	check(!rgfxSubmitDestroysBehind(RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN, true, true, false),
	      "a size query that times out is not, as before");

	std::printf("submit_gate_test: OK (%d checks)\n", checks);
	return 0;
}
