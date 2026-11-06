// oracle_events: what a Mac's Metal device answers for a shared event made in one process and
// used in another: its Mach port (eventPort, SPI), its handle (MTLSharedEventHandle), the events
// made back from either, their coding, and the value, listeners and waits two holders share.
// Each case runs in a child process of its own, since some may end the process; the parent
// prints how each child ended. It runs the same against the guest bundle's device, so it is the
// guest's check too.
//   oracle_events DEVICE [CASE...]   the cases named, or every case
// The cases, each line `CASE key=value ...`:
//   intro        the classes of a shared event and of its handle, up to NSObject: each class's
//                own instance and class methods with their type encodings, its instance
//                variables and protocols; and which shared-event creators the device answers
//   fake         MTLSharedEventHandle's initWithSharedEvent: (where it answers) over an object
//                of the probe's own that answers eventPort with a port the probe made: which
//                selectors the init sends it, the handle's eventPort and label, and the port's
//                send references at each step
//   refs         a shared event's eventPort: whether two asks give the same name, and the send
//                references the port holds after each step: the ask, a handle made and let go
//                of, an event made from the port (newSharedEventWithMachPort:, SPI) and from a
//                handle (newSharedEventWithHandle:) and let go of, and the event let go of; the
//                events made back: class, value, eventPort, and whether they are the event
//   values       one event and another made from its port in the same process: a CPU set of
//                either seen by the other at once, a listener of one fired by a set of the other
//                and by a GPU signal of the other, and a GPU wait on one released by a set of the
//                other
//   machport.*   newSharedEventWithMachPort: of a port that is no shared event's: .null
//                (MACH_PORT_NULL), .dead (MACH_PORT_DEAD), .recv (a send right of a receive right
//                the probe made), .task (mach_task_self()); the class answered, or nil
//   handle.nil   newSharedEventWithHandle: of nil
//   label        a labelled event's handle's label, and the label of the event made from it
//   coding       a shared event and its handle through keyed archivers: the convenience
//                archiver's data and error, whether encodeWithCoder: of an archiver raises (its
//                name and reason) or sets the archiver's error, and initWithCoder: of a keyed
//                unarchiver over an empty archive
//   xpc          a handle sent through an in-process NSXPC connection (an anonymous listener):
//                the event made from it on the far side, its value, and a set there seen here;
//                xpc.event, the shared event itself sent (what class arrives, and its value);
//                fake.xpc, a handle made over the fake's port (fake, above) sent: the port and
//                label the far side's handle answers
//   spawn        two processes: the probe spawns a peer of itself, which takes the event's port
//                over a Mach message and makes an event from it (newSharedEventWithMachPort:).
//                Each step's line: the value the peer reads; a CPU set in the peer as this
//                process reads it at once and once waited for; a GPU signal here as the peer's
//                listener and read see it; a GPU wait in the peer released by a CPU set here,
//                and the peer's signal after it seen here; a handle round trip in the peer; and
//                the event in the peer once this process has let go of it
// Build: see README.md.
#import "oracle_common.h"
#import <IOSurface/IOSurface.h>
#include <mach/mach.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static const char *g_argv0;
static const char *g_device;

static const char *cls(id o)
{
	return o ? class_getName(object_getClass(o)) : "nil";
}

static const char *err_text(NSError *e)
{
	if (e == nil)
		return "nil";
	return [[NSString stringWithFormat:@"%@ %ld", e.domain, (long)e.code] UTF8String];
}

// The send references `name` holds in this task, or -1 where it names no send right.
static int send_refs(mach_port_name_t name)
{
	mach_port_urefs_t refs = 0;
	if (name == MACH_PORT_NULL ||
	    mach_port_get_refs(mach_task_self(), name, MACH_PORT_RIGHT_SEND, &refs) != KERN_SUCCESS)
		return -1;
	return (int)refs;
}

static const char *port_type(mach_port_name_t name)
{
	mach_port_type_t t = 0;
	if (name == MACH_PORT_NULL)
		return "null";
	if (mach_port_type(mach_task_self(), name, &t) != KERN_SUCCESS)
		return "none";
	if (t & MACH_PORT_TYPE_DEAD_NAME)
		return "dead";
	if ((t & MACH_PORT_TYPE_SEND) && (t & MACH_PORT_TYPE_RECEIVE))
		return "send+receive";
	if (t & MACH_PORT_TYPE_SEND)
		return "send";
	if (t & MACH_PORT_TYPE_RECEIVE)
		return "receive";
	return "other";
}

static mach_port_t oe_receive_right(void);

// --- SPI by message ---------------------------------------------------------------------

static mach_port_t event_port(id e)
{
	SEL s = sel_registerName("eventPort");
	if (![e respondsToSelector:s])
		return MACH_PORT_NULL;
	return ((mach_port_t(*)(id, SEL))objc_msgSend)(e, s);
}

// newSharedEventWithMachPort:'s +1 event, which ARC takes as the new... family's.
typedef id __attribute__((ns_returns_retained)) (*oe_new_from_port)(id, SEL, mach_port_t);

static id<MTLSharedEvent> from_port(id<MTLDevice> d, mach_port_t port) NS_RETURNS_RETAINED;
static id<MTLSharedEvent> from_port(id<MTLDevice> d, mach_port_t port)
{
	SEL s = sel_registerName("newSharedEventWithMachPort:");
	if (![(id)d respondsToSelector:s])
		return nil;
	return ((oe_new_from_port)objc_msgSend)(d, s, port);
}

// --- intro --------------------------------------------------------------------------------

static void print_class(const char *tag, Class c)
{
	printf("%s class %s super %s\n", tag, class_getName(c),
	       class_getSuperclass(c) ? class_getName(class_getSuperclass(c)) : "-");
	unsigned int n = 0;
	Method *ms = class_copyMethodList(c, &n);
	for (unsigned int i = 0; i < n; i++)
		printf("%s   - %s %s\n", tag, sel_getName(method_getName(ms[i])),
		       method_getTypeEncoding(ms[i]));
	free(ms);
	ms = class_copyMethodList(object_getClass(c), &n);
	for (unsigned int i = 0; i < n; i++)
		printf("%s   + %s %s\n", tag, sel_getName(method_getName(ms[i])),
		       method_getTypeEncoding(ms[i]));
	free(ms);
	Ivar *iv = class_copyIvarList(c, &n);
	for (unsigned int i = 0; i < n; i++)
		printf("%s   ivar %s %s offset %td\n", tag, ivar_getName(iv[i]), ivar_getTypeEncoding(iv[i]),
		       ivar_getOffset(iv[i]));
	free(iv);
	Protocol *__unsafe_unretained *ps = class_copyProtocolList(c, &n);
	for (unsigned int i = 0; i < n; i++)
		printf("%s   protocol %s\n", tag, protocol_getName(ps[i]));
	free(ps);
}

static void print_chain(const char *tag, Class c)
{
	for (; c && c != [NSObject class]; c = class_getSuperclass(c))
		print_class(tag, c);
}

static void case_intro(id<MTLDevice> d)
{
	id<MTLSharedEvent> se = [d newSharedEvent];
	print_chain("intro event", object_getClass(se));
	MTLSharedEventHandle *h = [se newSharedEventHandle];
	print_chain("intro handle", h ? object_getClass(h) : [MTLSharedEventHandle class]);
	const char *sels[] = {"newSharedEvent", "newSharedEventWithHandle:",
	                      "newSharedEventWithMachPort:", "newSharedEventWithOptions:"};
	for (int i = 0; i < 4; i++) {
		SEL s = sel_registerName(sels[i]);
		Method m = class_getInstanceMethod(object_getClass(d), s);
		printf("intro device %s responds=%d types=%s\n", sels[i], (int)[(id)d respondsToSelector:s],
		       m ? method_getTypeEncoding(m) : "-");
	}
}

// --- fake ---------------------------------------------------------------------------------

// An object of the probe's own in place of a shared event: it answers eventPort with `port` and
// logs every selector sent to it that it does not implement, answering 0.
@interface OEFakeEvent : NSObject
@property(nonatomic) mach_port_t port;
@property(nonatomic, copy) NSString *label;
@end

@implementation OEFakeEvent
- (mach_port_t)eventPort
{
	printf("fake sent eventPort (refs %d)\n", send_refs(_port));
	return _port;
}
- (uint64_t)signaledValue
{
	printf("fake sent signaledValue\n");
	return 3;
}
- (NSString *)label
{
	printf("fake sent label\n");
	return _label;
}
- (BOOL)respondsToSelector:(SEL)s
{
	BOOL r = [super respondsToSelector:s];
	printf("fake asked respondsToSelector:%s -> %d\n", sel_getName(s), (int)r);
	return r;
}
- (NSMethodSignature *)methodSignatureForSelector:(SEL)s
{
	NSMethodSignature *sig = [super methodSignatureForSelector:s];
	return sig ? sig : [NSMethodSignature signatureWithObjCTypes:"Q@:"];
}
- (void)forwardInvocation:(NSInvocation *)inv
{
	printf("fake sent %s (not implemented; answers 0)\n", sel_getName(inv.selector));
	uint64_t zero = 0;
	if (inv.methodSignature.methodReturnLength <= sizeof zero)
		[inv setReturnValue:&zero];
}
@end

// A handle made by MTLSharedEventHandle's initWithSharedEvent: over a fake (OEFakeEvent) whose
// port is `port`: nil where the class has no such init.
static MTLSharedEventHandle *fake_handle(mach_port_t port)
{
	SEL init = sel_registerName("initWithSharedEvent:");
	printf("fake handle responds initWithSharedEvent:=%d\n",
	       (int)[MTLSharedEventHandle instancesRespondToSelector:init]);
	if (![MTLSharedEventHandle instancesRespondToSelector:init])
		return nil;
	OEFakeEvent *fake = [[OEFakeEvent alloc] init];
	fake.port = port;
	fake.label = @"fake-label";
	id h = [MTLSharedEventHandle alloc];
	h = ((id(*)(id, SEL, id))objc_msgSend)(h, init, fake);
	return h;
}

static void case_fake(void)
{
	mach_port_t rx = oe_receive_right();
	printf("fake port made refs %d\n", send_refs(rx));
	@autoreleasepool {
		MTLSharedEventHandle *h = fake_handle(rx);
		printf("fake handle=%s refs after init %d\n", cls(h), send_refs(rx));
		mach_port_t hp = event_port(h);
		printf("fake handle eventPort=%s same=%d label=%s refs %d\n", hp ? "nonzero" : "0",
		       (int)(hp == rx), h.label ? h.label.UTF8String : "nil", send_refs(rx));
		SEL tid = sel_registerName("labelTraceID");
		if ([h respondsToSelector:tid])
			printf("fake handle labelTraceID=%llu\n",
			       ((uint64_t(*)(id, SEL))objc_msgSend)(h, tid));
		h = nil;
	}
	printf("fake handle let go refs %d type %s\n", send_refs(rx), port_type(rx));
}

// --- refs ---------------------------------------------------------------------------------

static void case_refs(id<MTLDevice> d)
{
	mach_port_t p = MACH_PORT_NULL;
	@autoreleasepool {
		id<MTLSharedEvent> se = [d newSharedEvent];
		se.signaledValue = 7;
		p = event_port(se);
		printf("refs eventPort=%s refs %d type %s\n", p ? "nonzero" : "0", send_refs(p),
		       port_type(p));
		mach_port_t again = event_port(se);
		printf("refs eventPort again same=%d refs %d\n", (int)(again == p), send_refs(p));
		@autoreleasepool {
			MTLSharedEventHandle *h = [se newSharedEventHandle];
			mach_port_t hp = event_port(h);
			printf("refs handle made refs %d handle eventPort same=%d\n", send_refs(p),
			       (int)(hp == p));
			MTLSharedEventHandle *h2 = [se newSharedEventHandle];
			printf("refs second handle refs %d same handle=%d\n", send_refs(p), (int)(h2 == h));
			h = nil;
			h2 = nil;
		}
		printf("refs handles let go refs %d\n", send_refs(p));
		@autoreleasepool {
			id<MTLSharedEvent> e2 = from_port(d, p);
			printf("refs fromPort=%s value=%llu sameObject=%d eventPort same=%d refs %d\n", cls(e2),
			       e2 ? e2.signaledValue : 0ull, (int)(e2 == se), (int)(event_port(e2) == p),
			       send_refs(p));
			e2 = nil;
		}
		printf("refs fromPort let go refs %d\n", send_refs(p));
		@autoreleasepool {
			MTLSharedEventHandle *h = [se newSharedEventHandle];
			id<MTLSharedEvent> e3 = [d newSharedEventWithHandle:h];
			printf("refs fromHandle=%s value=%llu sameObject=%d eventPort same=%d refs %d\n",
			       cls(e3), e3 ? e3.signaledValue : 0ull, (int)(e3 == se),
			       (int)(event_port(e3) == p), send_refs(p));
			h = nil;
			printf("refs fromHandle handle let go refs %d\n", send_refs(p));
			e3 = nil;
		}
		printf("refs fromHandle let go refs %d\n", send_refs(p));
		// A port taken for good (+1), then the event let go of: whether the port lives on.
		mach_port_mod_refs(mach_task_self(), p, MACH_PORT_RIGHT_SEND, 1);
		se = nil;
	}
	printf("refs event let go refs %d type %s (one ref taken before)\n", send_refs(p), port_type(p));
	@autoreleasepool {
		id<MTLSharedEvent> e4 = from_port(d, p);
		printf("refs fromPort after the event went=%s value=%llu\n", cls(e4),
		       e4 ? e4.signaledValue : 0ull);
	}
}

// --- values -------------------------------------------------------------------------------

// A command buffer on `q` that signals `e` to `v`, committed; returned for its wait.
static id<MTLCommandBuffer> signal_cb(id<MTLCommandQueue> q, id<MTLSharedEvent> e, uint64_t v)
{
	id<MTLCommandBuffer> cb = [q commandBuffer];
	[cb encodeSignalEvent:e value:v];
	[cb commit];
	return cb;
}

static void case_values(id<MTLDevice> d)
{
	id<MTLSharedEvent> a = [d newSharedEvent];
	a.signaledValue = 1;
	id<MTLSharedEvent> b = from_port(d, event_port(a));
	printf("values b=%s value=%llu\n", cls(b), b ? b.signaledValue : 0ull);
	if (!b)
		return;
	b.signaledValue = 2;
	printf("values set b=2: a reads %llu at once\n", a.signaledValue);
	a.signaledValue = 3;
	printf("values set a=3: b reads %llu at once\n", b.signaledValue);
	MTLSharedEventListener *l =
	    [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("oe", NULL)];
	dispatch_semaphore_t fired = dispatch_semaphore_create(0);
	__block uint64_t at = 0;
	[a notifyListener:l
	          atValue:4
	            block:^(id<MTLSharedEvent> e, uint64_t v) {
		            at = v;
		            dispatch_semaphore_signal(fired);
	            }];
	b.signaledValue = 4;
	long w = dispatch_semaphore_wait(fired, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC));
	printf("values listener of a at 4, b set 4: fired=%d value=%llu\n", w == 0, at);
	id<MTLCommandQueue> q = [d newCommandQueue];
	[a notifyListener:l
	          atValue:5
	            block:^(id<MTLSharedEvent> e, uint64_t v) {
		            at = v;
		            dispatch_semaphore_signal(fired);
	            }];
	id<MTLCommandBuffer> cb = signal_cb(q, b, 5);
	[cb waitUntilCompleted];
	w = dispatch_semaphore_wait(fired, dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC));
	printf("values listener of a at 5, b signalled 5 on the GPU: fired=%d value=%llu a reads %llu "
	       "after the completion\n",
	       w == 0, at, a.signaledValue);
	id<MTLCommandBuffer> waits = [q commandBuffer];
	[waits encodeWaitForEvent:a value:6];
	[waits encodeSignalEvent:a value:7];
	[waits commit];
	usleep(50000);
	printf("values a GPU wait for a=6: status %ld before the set\n", (long)waits.status);
	double t0 = oracle_now_ms();
	b.signaledValue = 6;
	BOOL ok = [b waitUntilSignaledValue:7 timeoutMS:2000];
	printf("values b set 6: the wait on a released, b reached 7=%d in %.2f ms, status %ld\n",
	       (int)ok, oracle_now_ms() - t0, (long)waits.status);
}

// --- machport.* and handle.nil -----------------------------------------------------------

static void case_machport(id<MTLDevice> d, const char *which)
{
	mach_port_t p = MACH_PORT_NULL;
	if (!strcmp(which, "dead"))
		p = MACH_PORT_DEAD;
	else if (!strcmp(which, "recv")) {
		mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &p);
		mach_port_insert_right(mach_task_self(), p, p, MACH_MSG_TYPE_MAKE_SEND);
	} else if (!strcmp(which, "task"))
		p = mach_task_self();
	int before = send_refs(p);
	id<MTLSharedEvent> e = from_port(d, p);
	printf("machport.%s answered=%s value=%llu refs %d -> %d\n", which, cls(e),
	       e ? e.signaledValue : 0ull, before, send_refs(p));
}

static void case_handle_nil(id<MTLDevice> d)
{
	MTLSharedEventHandle *none = nil;
	id<MTLSharedEvent> e = [d newSharedEventWithHandle:none];
	printf("handle.nil answered=%s\n", cls(e));
}

// --- label --------------------------------------------------------------------------------

static void case_label(id<MTLDevice> d)
{
	id<MTLSharedEvent> se = [d newSharedEvent];
	se.label = @"oe-label";
	MTLSharedEventHandle *h = [se newSharedEventHandle];
	id<MTLSharedEvent> back = [d newSharedEventWithHandle:h];
	printf("label handle=%s fromHandle label=%s\n", h.label ? h.label.UTF8String : "nil",
	       back.label ? back.label.UTF8String : "nil");
	se.label = @"oe-label2";
	printf("label after relabel: handle=%s fromHandle=%s\n", h.label ? h.label.UTF8String : "nil",
	       back.label ? back.label.UTF8String : "nil");
	id<MTLSharedEvent> fromPort = from_port(d, event_port(se));
	printf("label fromPort label=%s\n", fromPort.label ? fromPort.label.UTF8String : "nil");
}

// --- coding -------------------------------------------------------------------------------

static void code_one(const char *tag, id o)
{
	for (int secure = 1; secure >= 0; secure--) {
		NSError *e = nil;
		NSData *data = nil;
		@try {
			data = [NSKeyedArchiver archivedDataWithRootObject:o requiringSecureCoding:secure error:&e];
			printf("coding %s archived secure=%d data=%lu error=%s\n", tag, secure,
			       (unsigned long)data.length, err_text(e));
		} @catch (NSException *x) {
			printf("coding %s archived secure=%d threw %s\n", tag, secure, x.name.UTF8String);
		}
	}
	NSKeyedArchiver *ar = [[NSKeyedArchiver alloc] initRequiringSecureCoding:YES];
	@try {
		[o encodeWithCoder:ar];
		printf("coding %s encodeWithCoder: returned, archiver error=%s\n", tag, err_text(ar.error));
	} @catch (NSException *x) {
		printf("coding %s encodeWithCoder: threw %s: %s\n", tag, x.name.UTF8String,
		       x.reason.UTF8String);
	}
	[ar finishEncoding];
	NSData *empty = [NSKeyedArchiver archivedDataWithRootObject:@{} requiringSecureCoding:YES
	                                                      error:nil];
	NSError *ue = nil;
	NSKeyedUnarchiver *un = [[NSKeyedUnarchiver alloc] initForReadingFromData:empty error:&ue];
	@try {
		id back = [(id)[[o class] alloc] initWithCoder:un];
		printf("coding %s initWithCoder: answered %s, unarchiver error=%s\n", tag, cls(back),
		       err_text(un.error));
	} @catch (NSException *x) {
		printf("coding %s initWithCoder: threw %s: %s\n", tag, x.name.UTF8String,
		       x.reason.UTF8String);
	}
}

static void case_coding(id<MTLDevice> d)
{
	id<MTLSharedEvent> se = [d newSharedEvent];
	printf("coding event conforms NSCoding=%d NSSecureCoding=%d supportsSecureCoding=%d\n",
	       (int)[(id)se conformsToProtocol:@protocol(NSCoding)],
	       (int)[(id)se conformsToProtocol:@protocol(NSSecureCoding)],
	       [[(id)se class] respondsToSelector:@selector(supportsSecureCoding)]
	           ? (int)[[(id)se class] supportsSecureCoding]
	           : -1);
	code_one("event", se);
	MTLSharedEventHandle *h = [se newSharedEventHandle];
	printf("coding handle conforms NSSecureCoding=%d supportsSecureCoding=%d\n",
	       (int)[h conformsToProtocol:@protocol(NSSecureCoding)],
	       (int)[MTLSharedEventHandle supportsSecureCoding]);
	code_one("handle", h);
}

// --- xpc ----------------------------------------------------------------------------------

@protocol OEPeer
- (void)takeHandle:(MTLSharedEventHandle *)handle reply:(void (^)(uint64_t value))reply;
- (void)takeEvent:(id)event reply:(void (^)(NSString *cls, uint64_t value))reply;
- (void)takeFakeHandle:(MTLSharedEventHandle *)handle reply:(void (^)(uint32_t port))reply;
@end

@interface OEPeerImpl : NSObject <OEPeer, NSXPCListenerDelegate>
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLSharedEvent> original;
@property(nonatomic, strong) NSXPCInterface *interface;
@end

@implementation OEPeerImpl
- (BOOL)listener:(NSXPCListener *)listener shouldAcceptNewConnection:(NSXPCConnection *)c
{
	c.exportedInterface = _interface;
	c.exportedObject = self;
	[c resume];
	return YES;
}
- (void)takeHandle:(MTLSharedEventHandle *)handle reply:(void (^)(uint64_t))reply
{
	id<MTLSharedEvent> e = [_device newSharedEventWithHandle:handle];
	printf("xpc far side: handle=%s port same as the event's=%d event=%s value=%llu "
	       "sameObject=%d\n",
	       cls(handle), (int)(event_port(handle) == event_port(_original)), cls(e),
	       e ? e.signaledValue : 0ull, (int)(e == _original));
	if (e)
		e.signaledValue = e.signaledValue + 1;
	printf("xpc far side set %llu: the original reads %llu at once\n", e ? e.signaledValue : 0ull,
	       _original.signaledValue);
	reply(e ? e.signaledValue : 0);
}
- (void)takeFakeHandle:(MTLSharedEventHandle *)handle reply:(void (^)(uint32_t))reply
{
	mach_port_t p = event_port(handle);
	printf("fake.xpc far side: handle=%s eventPort refs %d label=%s\n", cls(handle), send_refs(p),
	       handle.label ? handle.label.UTF8String : "nil");
	reply(p);
}
- (void)takeEvent:(id)event reply:(void (^)(NSString *, uint64_t))reply
{
	BOOL shared = [event conformsToProtocol:@protocol(MTLSharedEvent)];
	printf("xpc.event far side: %s conforms MTLSharedEvent=%d sameObject=%d\n", cls(event),
	       (int)shared, (int)(event == _original));
	reply(@(cls(event)), shared ? [(id<MTLSharedEvent>)event signaledValue] : 0);
}
@end

static void case_xpc(id<MTLDevice> d, BOOL sendEvent, BOOL fake)
{
	id<MTLSharedEvent> se = [d newSharedEvent];
	se.signaledValue = 7;
	NSXPCInterface *iface = [NSXPCInterface interfaceWithProtocol:@protocol(OEPeer)];
	NSMutableSet *classes = [NSMutableSet setWithObjects:[(id)se class], [MTLSharedEventHandle class],
	                                                     nil];
	Class ios = NSClassFromString(@"IOSurfaceSharedEvent");
	if (ios)
		[classes addObject:ios];
	[iface setClasses:classes forSelector:@selector(takeEvent:reply:) argumentIndex:0 ofReply:NO];
	OEPeerImpl *impl = [[OEPeerImpl alloc] init];
	impl.device = d;
	impl.original = se;
	impl.interface = iface;
	NSXPCListener *listener = [NSXPCListener anonymousListener];
	listener.delegate = impl;
	[listener resume];
	NSXPCConnection *c = [[NSXPCConnection alloc] initWithListenerEndpoint:listener.endpoint];
	c.remoteObjectInterface = iface;
	[c resume];
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	id<OEPeer> peer = [c remoteObjectProxyWithErrorHandler:^(NSError *e) {
		printf("xpc error %s\n", err_text(e));
		dispatch_semaphore_signal(done);
	}];
	if (fake) {
		mach_port_t rx = oe_receive_right();
		MTLSharedEventHandle *h = fake_handle(rx);
		printf("fake.xpc handle=%s refs %d\n", cls(h), send_refs(rx));
		[peer takeFakeHandle:h
		               reply:^(uint32_t p) {
			               printf("fake.xpc reply: the far side's port is the probe's=%d refs %d\n",
			                      (int)(p == rx), send_refs(rx));
			               dispatch_semaphore_signal(done);
		               }];
	} else if (sendEvent) {
		[peer takeEvent:se
		          reply:^(NSString *name, uint64_t v) {
			          printf("xpc.event arrived as %s value=%llu\n", name.UTF8String, v);
			          dispatch_semaphore_signal(done);
		          }];
	} else {
		MTLSharedEventHandle *h = [se newSharedEventHandle];
		[peer takeHandle:h
		           reply:^(uint64_t v) {
			           printf("xpc reply value=%llu: the original reads %llu\n", v, se.signaledValue);
			           dispatch_semaphore_signal(done);
		           }];
	}
	if (dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)))
		printf("xpc no reply in 5 s\n");
	[c invalidate];
}

// --- spawn: two processes -----------------------------------------------------------------

// One message between the probe and its peer: an op, two values and a port (or none).
typedef struct {
	mach_msg_header_t header;
	mach_msg_body_t body;
	mach_msg_port_descriptor_t port;
	uint64_t op, a, b;
} oe_msg;

typedef struct {
	oe_msg m;
	mach_msg_trailer_t trailer;
} oe_rcv;

enum { OE_HELLO = 1, OE_EVENT, OE_ACK, OE_SET, OE_READY, OE_SIGNALLED, OE_COMMITTED, OE_DONE, OE_GONE };

static kern_return_t oe_send(mach_port_t to, uint64_t op, uint64_t a, uint64_t b, mach_port_t port,
                             mach_msg_type_name_t disposition)
{
	oe_msg m;
	memset(&m, 0, sizeof m);
	m.header.msgh_bits =
	    MACH_MSGH_BITS_SET(MACH_MSG_TYPE_COPY_SEND, 0, 0, MACH_MSGH_BITS_COMPLEX);
	m.header.msgh_size = sizeof m;
	m.header.msgh_remote_port = to;
	m.body.msgh_descriptor_count = 1;
	m.port.name = port;
	m.port.disposition = port ? disposition : MACH_MSG_TYPE_COPY_SEND;
	m.port.type = MACH_MSG_PORT_DESCRIPTOR;
	m.op = op;
	m.a = a;
	m.b = b;
	return mach_msg(&m.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT, sizeof m, 0, MACH_PORT_NULL,
	                10000, MACH_PORT_NULL);
}

// The next message on `rx`, of op `want`; false (logged) on a timeout or another op.
static BOOL oe_recv(mach_port_t rx, uint64_t want, oe_rcv *r, const char *who)
{
	memset(r, 0, sizeof *r);
	kern_return_t kr = mach_msg(&r->m.header, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof *r, rx,
	                            20000, MACH_PORT_NULL);
	if (kr != KERN_SUCCESS || r->m.op != want) {
		printf("spawn %s: no message %llu (kr 0x%x, op %llu)\n", who, want, kr, r->m.op);
		return NO;
	}
	return YES;
}

static mach_port_t oe_receive_right(void)
{
	mach_port_t p = MACH_PORT_NULL;
	mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &p);
	mach_port_insert_right(mach_task_self(), p, p, MACH_MSG_TYPE_MAKE_SEND);
	return p;
}

// The peer: given the probe's port through its exception port (EXC_MASK_RPC_ALERT, which no
// exception raises), it takes the event's port and runs its half of each step.
static int run_peer(id<MTLDevice> d)
{
	exception_mask_t masks[EXC_TYPES_COUNT];
	mach_port_t ports[EXC_TYPES_COUNT];
	exception_behavior_t behaviors[EXC_TYPES_COUNT];
	thread_state_flavor_t flavors[EXC_TYPES_COUNT];
	mach_msg_type_number_t n = EXC_TYPES_COUNT;
	if (task_get_exception_ports(mach_task_self(), EXC_MASK_RPC_ALERT, masks, &n, ports, behaviors,
	                             flavors) != KERN_SUCCESS ||
	    n == 0 || !MACH_PORT_VALID(ports[0])) {
		printf("spawn peer: no port from the probe\n");
		return 1;
	}
	mach_port_t probe = ports[0];
	task_set_exception_ports(mach_task_self(), EXC_MASK_RPC_ALERT, MACH_PORT_NULL, EXCEPTION_DEFAULT,
	                         THREAD_STATE_NONE);
	mach_port_t rx = oe_receive_right();
	oe_send(probe, OE_HELLO, 0, 0, rx, MACH_MSG_TYPE_MAKE_SEND);
	oe_rcv r;
	if (!oe_recv(rx, OE_EVENT, &r, "peer"))
		return 1;
	mach_port_t port = r.m.port.name;
	id<MTLSharedEvent> e = from_port(d, port);
	printf("spawn peer: fromPort=%s value=%llu (the probe set %llu) port refs %d\n", cls(e),
	       e ? e.signaledValue : 0ull, r.m.a, send_refs(port));
	oe_send(probe, OE_ACK, e ? e.signaledValue : 0, 0, MACH_PORT_NULL, 0);
	if (!e)
		return 1;
	// A CPU set here, which the probe reads.
	e.signaledValue = 8;
	oe_send(probe, OE_SET, 8, 0, MACH_PORT_NULL, 0);
	// A listener at 9, which the probe's GPU signal fires.
	MTLSharedEventListener *l =
	    [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("oe", NULL)];
	dispatch_semaphore_t fired = dispatch_semaphore_create(0);
	__block double firedAt = 0;
	[e notifyListener:l
	          atValue:9
	            block:^(id<MTLSharedEvent> ev, uint64_t v) {
		            firedAt = oracle_now_ms();
		            dispatch_semaphore_signal(fired);
	            }];
	oe_send(probe, OE_READY, 0, 0, MACH_PORT_NULL, 0);
	if (!oe_recv(rx, OE_SIGNALLED, &r, "peer"))
		return 1;
	const double told = oracle_now_ms();
	const uint64_t atTell = e.signaledValue;
	long w = dispatch_semaphore_wait(fired, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC));
	printf("spawn peer: GPU signal 9 in the probe: read %llu when told, listener fired=%d %s the "
	       "message (%.2f ms)\n",
	       atTell, w == 0, firedAt <= told ? "before" : "after", firedAt - told);
	// A GPU wait here for 10, which the probe sets on the CPU, and a signal of 11 after it.
	id<MTLCommandQueue> q = [d newCommandQueue];
	id<MTLCommandBuffer> cb = [q commandBuffer];
	[cb encodeWaitForEvent:e value:10];
	[cb encodeSignalEvent:e value:11];
	[cb commit];
	oe_send(probe, OE_COMMITTED, 0, 0, MACH_PORT_NULL, 0);
	[cb waitUntilCompleted];
	printf("spawn peer: the buffer that waited for 10 completed, status %ld, value %llu\n",
	       (long)cb.status, e.signaledValue);
	oe_send(probe, OE_DONE, (uint64_t)cb.status, 0, MACH_PORT_NULL, 0);
	// A handle round trip here.
	MTLSharedEventHandle *h = [e newSharedEventHandle];
	id<MTLSharedEvent> back = [d newSharedEventWithHandle:h];
	printf("spawn peer: handle=%s fromHandle=%s value=%llu\n", cls(h), cls(back),
	       back ? back.signaledValue : 0ull);
	// The probe lets go of its event: this one lives on.
	if (!oe_recv(rx, OE_GONE, &r, "peer"))
		return 1;
	usleep(200000);
	e.signaledValue = 12;
	BOOL reached = [e waitUntilSignaledValue:12 timeoutMS:1000];
	printf("spawn peer: after the probe let go: set 12, reads %llu, wait=%d\n", e.signaledValue,
	       (int)reached);
	return 0;
}

static void case_spawn(id<MTLDevice> d)
{
	mach_port_t rx = oe_receive_right();
	posix_spawnattr_t attr;
	posix_spawnattr_init(&attr);
	posix_spawnattr_setexceptionports_np(&attr, EXC_MASK_RPC_ALERT, rx, EXCEPTION_DEFAULT,
	                                     THREAD_STATE_NONE);
	char *args[] = {(char *)g_argv0, (char *)g_device, "--peer", NULL};
	pid_t pid;
	int sr = posix_spawn(&pid, g_argv0, NULL, &attr, args, environ);
	posix_spawnattr_destroy(&attr);
	if (sr != 0) {
		printf("spawn: posix_spawn failed %d\n", sr);
		return;
	}
	oe_rcv r;
	id<MTLSharedEvent> se = nil;
	@autoreleasepool {
		se = [d newSharedEvent];
	}
	se.signaledValue = 7;
	id<MTLCommandQueue> q = [d newCommandQueue];
	if (!oe_recv(rx, OE_HELLO, &r, "probe"))
		goto out;
	mach_port_t peer = r.m.port.name;
	oe_send(peer, OE_EVENT, 7, 0, event_port(se), MACH_MSG_TYPE_COPY_SEND);
	if (!oe_recv(rx, OE_ACK, &r, "probe"))
		goto out;
	printf("spawn: the peer read %llu\n", r.m.a);
	if (!oe_recv(rx, OE_SET, &r, "probe"))
		goto out;
	{
		const uint64_t now = se.signaledValue;
		double t0 = oracle_now_ms();
		BOOL ok = [se waitUntilSignaledValue:8 timeoutMS:2000];
		printf("spawn: the peer set 8 on the CPU: read %llu at once, waited=%d %.2f ms\n", now,
		       (int)ok, oracle_now_ms() - t0);
	}
	if (!oe_recv(rx, OE_READY, &r, "probe"))
		goto out;
	{
		id<MTLCommandBuffer> cb = signal_cb(q, se, 9);
		[cb waitUntilCompleted];
		oe_send(peer, OE_SIGNALLED, 9, 0, MACH_PORT_NULL, 0);
	}
	if (!oe_recv(rx, OE_COMMITTED, &r, "probe"))
		goto out;
	{
		usleep(50000);
		const uint64_t before = se.signaledValue;
		double t0 = oracle_now_ms();
		se.signaledValue = 10;
		BOOL ok = [se waitUntilSignaledValue:11 timeoutMS:3000];
		printf("spawn: the peer's buffer waits for 10 (read %llu before): set 10, its signal of 11 "
		       "seen here=%d in %.2f ms\n",
		       before, (int)ok, oracle_now_ms() - t0);
	}
	if (!oe_recv(rx, OE_DONE, &r, "probe"))
		goto out;
	printf("spawn: the peer's buffer status %llu\n", r.m.a);
	se = nil;
	oe_send(peer, OE_GONE, 0, 0, MACH_PORT_NULL, 0);
out:;
	int st = 0;
	waitpid(pid, &st, 0);
	if (WIFSIGNALED(st))
		printf("spawn: the peer died(%d)\n", WTERMSIG(st));
	else
		printf("spawn: the peer exit %d\n", WEXITSTATUS(st));
}

// --- the cases ----------------------------------------------------------------------------

static const char *const kCases[] = {"intro",        "fake",         "refs",          "values",
                                     "machport.null", "machport.dead", "machport.recv", "machport.task",
                                     "handle.nil",   "label",        "coding",        "xpc",
                                     "xpc.event",    "fake.xpc",     "spawn"};

static int run_case(id<MTLDevice> d, const char *c)
{
	if (!strcmp(c, "intro"))
		case_intro(d);
	else if (!strcmp(c, "fake"))
		case_fake();
	else if (!strcmp(c, "refs"))
		case_refs(d);
	else if (!strcmp(c, "values"))
		case_values(d);
	else if (!strncmp(c, "machport.", 9))
		case_machport(d, c + 9);
	else if (!strcmp(c, "handle.nil"))
		case_handle_nil(d);
	else if (!strcmp(c, "label"))
		case_label(d);
	else if (!strcmp(c, "coding"))
		case_coding(d);
	else if (!strcmp(c, "xpc"))
		case_xpc(d, NO, NO);
	else if (!strcmp(c, "xpc.event"))
		case_xpc(d, YES, NO);
	else if (!strcmp(c, "fake.xpc"))
		case_xpc(d, NO, YES);
	else if (!strcmp(c, "spawn"))
		case_spawn(d);
	else {
		printf("%s unknown case\n", c);
		return 2;
	}
	return 0;
}

static void spawn_case(const char *c)
{
	printf("== %s\n", c);
	fflush(stdout);
	char *args[] = {(char *)g_argv0, (char *)g_device, "--one", (char *)c, NULL};
	pid_t pid;
	if (posix_spawn(&pid, g_argv0, NULL, NULL, args, environ) != 0) {
		printf("== %s spawn failed\n", c);
		return;
	}
	int st = 0;
	waitpid(pid, &st, 0);
	if (WIFSIGNALED(st))
		printf("== %s died(%d)\n", c, WTERMSIG(st));
	else
		printf("== %s exit %d\n", c, WEXITSTATUS(st));
	fflush(stdout);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	g_argv0 = argv[0];
	if (argc < 2) {
		fprintf(stderr, "usage: %s DEVICE [CASE...]\n", argv[0]);
		return 2;
	}
	g_device = argv[1];
	if (argc == 3 && !strcmp(argv[2], "--peer")) {
		int rc;
		@autoreleasepool {
			rc = run_peer(oracle_device(argv[1]));
		}
		fflush(stdout);
		return rc;
	}
	if (argc == 4 && !strcmp(argv[2], "--one")) {
		int rc;
		@autoreleasepool {
			rc = run_case(oracle_device(argv[1]), argv[3]);
		}
		fflush(stdout);
		return rc;
	}
	@autoreleasepool {
		oracle_header("oracle_events", oracle_device(argv[1]));
	}
	if (argc == 2)
		for (size_t i = 0; i < sizeof kCases / sizeof kCases[0]; i++)
			spawn_case(kCases[i]);
	else
		for (int i = 2; i < argc; i++)
			spawn_case(argv[i]);
	return 0;
}
