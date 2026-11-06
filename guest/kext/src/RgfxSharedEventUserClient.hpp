/*
 * RgfxSharedEventUserClient: a shared event's port, user-client type 'RGSE'
 * (kRgfxSharedEventUserClientType) on the accelerator. The bundle opens one for each shared event
 * a process hands another, and its send right is the event's eventPort. Its two selectors,
 * kRgfxSharedEventMethodExport and kRgfxSharedEventMethodImport (RgfxUserClientABI.hpp), and the
 * rules they keep: RgfxSharedEventPorts.hpp.
 *
 * Its life is its port's. IOKit frees it once no task holds a send right of its connection and
 * the accelerator lets go of it, and its free lets go of the name it gave (SHARED_EVENT_UNNAME).
 * Its close, and IOKit's word that its creator died (clientDied, which IOKit sends at the
 * creator's end while another task holds the port too), only detach it from the accelerator,
 * whose hold would keep it past its port: every task that holds the port calls on it as before.
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_SHARED_EVENT_USER_CLIENT_HPP_
#define RGFX_SHARED_EVENT_USER_CLIENT_HPP_

#include <IOKit/IOLocks.h>
#include <IOKit/IOUserClient.h>

#include "RgfxSharedEventPorts.hpp"

class RgfxAccelerator;

class RgfxSharedEventUserClient : public IOUserClient {
	OSDeclareDefaultStructors(RgfxSharedEventUserClient);

public:
	bool initWithTask(task_t owningTask, void *securityID, UInt32 type,
	                  OSDictionary *properties) override;
	bool start(IOService *provider) override;
	IOReturn clientClose() override;
	IOReturn clientDied() override;
	void free() override;
	IOReturn externalMethod(uint32_t selector, IOExternalMethodArguments *arguments,
	                        IOExternalMethodDispatch *dispatch, OSObject *target,
	                        void *reference) override;
	// The accelerator the port is of, from its start until its free; nullptr outside them.
	const RgfxAccelerator *acceleratorOf() const { return accelerator; }

private:
	IOReturn exportMethod(IOExternalMethodArguments *arguments);
	IOReturn importMethod(IOExternalMethodArguments *arguments);
	// Detaches the client from the accelerator, once: nothing else of it ends.
	void letGoOfAccelerator();

	// The accelerator, retained from start until free, so a call after the detach still reaches
	// it.
	RgfxAccelerator *accelerator;
	// Under lock: the name (RgfxPortName), and whether the client is detached.
	IOLock *lock;
	RgfxPortName port;
	bool detached;
};

#endif // RGFX_SHARED_EVENT_USER_CLIENT_HPP_
