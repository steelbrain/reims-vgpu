// The native test of RgfxSharedEventPorts.hpp: a port's call speaks only for a client of the
// calling task that holds the context it names, its ids lie in that context, a port names one
// event, given once, and the messages the kext builds are the wire's.
#include "RgfxSharedEventPorts.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "shared_event_ports_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

uint64_t obj(uint64_t context, uint64_t n)
{
	return context << RGFX_CONTEXT_OBJ_ID_SHIFT | n;
}
} // namespace

int main()
{
	int a = 0, b = 0;
	const void *taskA = &a, *taskB = &b;
	// The caller's own client, holding the context it names.
	check(rgfxPortCallerHolds(taskA, taskA, true, 3, 3), "the caller's client of its context");
	// The port is a capability for the event, never for another process's context: a client of
	// another task, even holding the context named, does not speak for the caller.
	check(!rgfxPortCallerHolds(taskB, taskA, true, 3, 3), "another task's client of the context");
	check(!rgfxPortCallerHolds(taskA, taskA, true, 3, 4), "the caller's client of another context");
	check(!rgfxPortCallerHolds(taskA, taskA, false, 3, 3), "a client whose context is closed");
	check(!rgfxPortCallerHolds(taskA, taskA, true, 0, 0), "context 0 is no process's");
	check(!rgfxPortCallerHolds(nullptr, nullptr, true, 3, 3), "a client with no task");
	check(!rgfxPortCallerHolds(taskA, taskA, true, 3, (uint64_t{1} << 32) | 3),
	      "a context past 32 bits");

	check(rgfxPortIdsInContext(3, obj(3, 20), obj(3, 1)), "both ids in the context");
	check(!rgfxPortIdsInContext(3, obj(4, 20), obj(3, 1)), "the event of another context");
	check(!rgfxPortIdsInContext(3, obj(3, 20), obj(4, 1)), "the listen queue of another context");
	check(!rgfxPortIdsInContext(3, obj(3, 0), obj(3, 1)), "the context's base as the event");
	check(!rgfxPortIdsInContext(3, obj(3, 20), obj(3, 0)), "the context's base as the queue");
	check(!rgfxPortIdsInContext(0, obj(0, 20), obj(0, 1)), "context 0");

	// A port the process was handed is a share of the asking client's accelerator alone.
	int accelA = 0, accelB = 0;
	check(rgfxPortIsShareOf(&accelA, &accelA), "a port of the client's accelerator");
	check(!rgfxPortIsShareOf(&accelB, &accelA), "a port of another accelerator");
	check(!rgfxPortIsShareOf(nullptr, &accelA), "a port not started or freed");

	// A port names one event, given once; an import finds it only once the export is answered.
	RgfxPortName port;
	check(port.forImport() == 0, "a new port names nothing");
	check(port.beginExport(), "the first export begins");
	check(!port.beginExport(), "a second export while one is under way");
	check(port.forImport() == 0, "nothing to import while the export is under way");
	port.endExport(0);
	check(port.forImport() == 0, "a refused export names nothing");
	check(port.beginExport(), "an export after a refused one begins");
	port.endExport(7);
	check(port.forImport() == 7, "the export answered names the event");
	check(!port.beginExport(), "an export of a port that names an event");
	check(port.forImport() == 7, "and the name stands");

	RgfxSharedEventName name = rgfxSharedEventNameMessage(obj(3, 20), 7, obj(3, 1));
	check(name.header.type == RGFX_CMD_SHARED_EVENT_NAME && name.obj_id == obj(3, 20) &&
	          name.name == 7 && name.listen_queue_obj_id == obj(3, 1),
	      "SHARED_EVENT_NAME's fields");
	RgfxSharedEventImport import = rgfxSharedEventImportMessage(obj(5, 9), 7, obj(5, 2));
	check(import.header.type == RGFX_CMD_SHARED_EVENT_IMPORT && import.obj_id == obj(5, 9) &&
	          import.name == 7 && import.listen_queue_obj_id == obj(5, 2),
	      "SHARED_EVENT_IMPORT's fields");
	RgfxSharedEventUnname unname = rgfxSharedEventUnnameMessage(7);
	check(unname.header.type == RGFX_CMD_SHARED_EVENT_UNNAME && unname.reserved_10 == 0 &&
	          unname.name == 7,
	      "SHARED_EVENT_UNNAME's fields: context 0's");
	static_assert(sizeof(RgfxSharedEventName) == 40, "SHARED_EVENT_NAME is 40 bytes");
	static_assert(sizeof(RgfxSharedEventImport) == 40, "SHARED_EVENT_IMPORT is 40 bytes");
	static_assert(sizeof(RgfxSharedEventUnname) == 32, "SHARED_EVENT_UNNAME is 32 bytes");
	static_assert(sizeof(RgfxSharedEventValue) == 32, "SHARED_EVENT_VALUE is 32 bytes");

	std::printf("shared_event_ports_test: %d checks passed\n", checks);
	return 0;
}
