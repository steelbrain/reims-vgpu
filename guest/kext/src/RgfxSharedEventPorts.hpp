// RgfxSharedEventPorts: the rules of a shared event's port (RgfxSharedEventUserClient), the
// connection a process opens on the accelerator to hand its shared event to another process
// (design.md, "Shared events across processes").
//
// The kext can give a process no port of an object of its own: IOUserClient's
// copyPortNameForObjectInTask is in com.apple.kpi.private alone (the macOS 26 KDK's symbol sets),
// which no kext but Apple's may link. So the port is a connection: its send right is the event's
// eventPort, which Metal's handle carries over XPC as it carries Apple's, and a call on it proves
// that the caller holds it. Its calls:
//   - Export, by the process that opened it: the kext names the caller's shared event device-wide
//     (SHARED_EVENT_NAME, a name from 1 that the kext never gives again), once per port;
//   - Import, by any process that holds the port: the kext makes the caller's new shared event of
//     the name (SHARED_EVENT_IMPORT), answered the event's value.
// Before it calls on a port it was handed, the bundle asks its own connection whether the port is
// one of these (IOConnectAddClient, RgfxAcceleratorUserClient::connectClient), which the kernel
// answers without sending the port anything.
// Each names the caller's context and the ids in it. The kext finds the accelerator's client that
// holds that context open for the calling task (rgfxPortCallerHolds), so a process makes and names
// events in no context but its own, and posts through that client's asks, which its close waits
// for.
//
// The name holds the host's event until the port goes, as a Mac's port holds its kernel event:
// the connection's last send right gone, in any task, frees it, and its free posts the name's
// SHARED_EVENT_UNNAME. Its close and its creator's death (IOKit's clientClose and clientDied, which
// IOKit also calls where the creator dies while another task holds the port) let go of the
// accelerator's registry hold of it, and nothing else: the port works on for every task that
// holds it.
//
// Header-only and free of the kernel's headers, so its native test
// (tests/shared_event_ports_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_SHARED_EVENT_PORTS_HPP_
#define RGFX_SHARED_EVENT_PORTS_HPP_

#include "reims_graphics_wire.h"

#include <stdint.h>

// Whether a client of the accelerator speaks for a port's caller: its task, `clientTask`, is the
// caller's, and it holds context `asked` open (`hasContext`, `clientContext`). A context is
// numbered from 1.
inline bool rgfxPortCallerHolds(const void *clientTask, const void *callerTask, bool hasContext,
                                uint32_t clientContext, uint64_t asked)
{
	return clientTask && clientTask == callerTask && hasContext && asked != 0 &&
	       clientContext == asked;
}

// Whether an Export's or an Import's ids lie in the caller's context `ctx`: the shared event's
// and the listen queue's, neither the context's base, 0 within it.
inline bool rgfxPortIdsInContext(uint64_t ctx, uint64_t eventId, uint64_t listenQueue)
{
	const uint64_t within = (uint64_t{1} << RGFX_CONTEXT_OBJ_ID_SHIFT) - 1;
	return ctx != 0 && (eventId >> RGFX_CONTEXT_OBJ_ID_SHIFT) == ctx &&
	       (listenQueue >> RGFX_CONTEXT_OBJ_ID_SHIFT) == ctx && (eventId & within) != 0 &&
	       (listenQueue & within) != 0;
}

// Whether a port a process was handed is a shared event's port of the accelerator whose client
// asks (RgfxAcceleratorUserClient::connectClient): a port of it, started and not yet freed
// (`portsAccelerator` non-null).
inline bool rgfxPortIsShareOf(const void *portsAccelerator, const void *accelerator)
{
	return portsAccelerator && portsAccelerator == accelerator;
}

// What a port names, as its calls change it, under the port's lock.
struct RgfxPortName {
	// The name the port gives imports, 0 until an Export named the event.
	uint64_t name = 0;
	// An Export is under way.
	bool exporting = false;

	// Begins an Export: false where the port names an event already, or another Export is under
	// way.
	bool beginExport()
	{
		if (name != 0 || exporting)
			return false;
		exporting = true;
		return true;
	}
	// Ends the Export begun: `given` is the name the device took, 0 where it refused it or did
	// not answer.
	void endExport(uint64_t given)
	{
		exporting = false;
		name = given;
	}
	// The name an Import makes its event from: 0 where the port names none yet.
	uint64_t forImport() const { return exporting ? 0 : name; }
};

// The messages the kext builds, each from the port's name and the ids the caller gave; the
// kext stamps their size and seqno as it posts them.
inline RgfxSharedEventName rgfxSharedEventNameMessage(uint64_t eventId, uint64_t name,
                                                      uint64_t listenQueue)
{
	RgfxSharedEventName m = {};
	m.header.type = RGFX_CMD_SHARED_EVENT_NAME;
	m.obj_id = eventId;
	m.name = name;
	m.listen_queue_obj_id = listenQueue;
	return m;
}

inline RgfxSharedEventImport rgfxSharedEventImportMessage(uint64_t eventId, uint64_t name,
                                                          uint64_t listenQueue)
{
	RgfxSharedEventImport m = {};
	m.header.type = RGFX_CMD_SHARED_EVENT_IMPORT;
	m.obj_id = eventId;
	m.name = name;
	m.listen_queue_obj_id = listenQueue;
	return m;
}

// Context 0's, which no process is: the unname waits for no process's messages.
inline RgfxSharedEventUnname rgfxSharedEventUnnameMessage(uint64_t name)
{
	RgfxSharedEventUnname m = {};
	m.header.type = RGFX_CMD_SHARED_EVENT_UNNAME;
	m.reserved_10 = 0;
	m.name = name;
	return m;
}

#endif // RGFX_SHARED_EVENT_PORTS_HPP_
