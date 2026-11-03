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

	std::printf("submit_gate_test: OK (%d checks)\n", checks);
	return 0;
}
