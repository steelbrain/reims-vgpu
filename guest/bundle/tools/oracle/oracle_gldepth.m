// oracle_gldepth: what Apple's OpenGL renderer over Metal (the GL the x86-64 guest gets on our
// device, and every Apple-silicon Mac's GL) asks of Metal for its command queues, and how many of
// its command buffers it keeps in flight in front of a busy GPU. Off screen: an FBO of a CGL
// context with no drawable. Run it where GL is over Metal (an Apple-silicon Mac); on an Intel
// Mac's own GL drivers nothing reaches Metal and every count is 0.
//   oracle_gldepth [TARGET_MS [FLUSHES]]
// Before the context is made, the probe hooks the Metal device's queue creators
// (newCommandQueue, newCommandQueueWithMaxCommandBufferCount:, newCommandQueueWithDescriptor:)
// and, on each queue made, its command-buffer creators, so every queue's asked count and every
// command buffer's creation, block (a creation over 1 ms) and completion are counted.
// A fragment shader's loop is calibrated so one full-screen draw takes about TARGET_MS (default 5)
// of wall time on the GPU (ten draws and a glFinish). Then, for each arm, FLUSHES (default 120)
// glFlush calls, each after:
//   arm draw1:  one heavy draw;
//   arm draw8:  eight heavy draws of TARGET_MS / 8, each after a glBufferSubData of the quad's
//               vertex buffer (an upload a frame's draws make);
// and a glFinish at the end.
// Lines: "queue N: <creator> asked A, answers maxCommandBufferCount M, class C"; per arm: the
// command buffers made on each queue, per flush, the most uncompleted at a creation, the creations
// that blocked and the uncompleted count at the first, the glFlush call's p50 and max, the
// uncompleted count right after each glFlush (p50, max), the GPU mean of a command buffer, and
// the arm's wall time against the sum of its buffers' GPU times.
// Q49 ran `oracle_gldepth 5 120` and `oracle_gldepth 10 80` on the M3; dgpu-oracle.md Q49 holds
// what it found. Build: see README.md (arm64, with -framework OpenGL).
#import "oracle_common.h"
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>
#include <stdatomic.h>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

enum { kQ = 16, kW = 1280, kH = 720 };

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
static void *gQueues[kQ];
static int gNQ;
static atomic_int gOut[kQ], gMade[kQ], gBlocks[kQ], gMaxOut[kQ], gFirstBlockOut[kQ];
static _Atomic double gGpuSum[kQ];
static atomic_int gDone[kQ];
static NSMutableSet *gHookedQueueClasses;

static IMP hook(Class c, SEL s, id block)
{
	Method m = class_getInstanceMethod(c, s);
	if (!m)
		return NULL;
	IMP orig = method_getImplementation(m);
	IMP nimp = imp_implementationWithBlock(block);
	if (!class_addMethod(c, s, nimp, method_getTypeEncoding(m)))
		method_setImplementation(m, nimp);
	return orig;
}

static int queueIndex(id q)
{
	pthread_mutex_lock(&gLock);
	int r = -1;
	for (int i = 0; i < gNQ; i++)
		if (gQueues[i] == (__bridge void *)q)
			r = i;
	pthread_mutex_unlock(&gLock);
	return r;
}

static void noteCB(id q, id<MTLCommandBuffer> cb, double dt)
{
	int i = queueIndex(q);
	if (i < 0 || !cb)
		return;
	int out = atomic_load(&gOut[i]);
	if (dt > 1.0) {
		if (atomic_fetch_add(&gBlocks[i], 1) == 0)
			atomic_store(&gFirstBlockOut[i], out);
	}
	int now = atomic_fetch_add(&gOut[i], 1) + 1;
	atomic_fetch_add(&gMade[i], 1);
	int mx = atomic_load(&gMaxOut[i]);
	while (now - 1 > mx && !atomic_compare_exchange_weak(&gMaxOut[i], &mx, now - 1)) {
	}
	[cb addCompletedHandler:^(id<MTLCommandBuffer> x) {
		atomic_fetch_sub(&gOut[i], 1);
		atomic_fetch_add(&gDone[i], 1);
		double g = (x.GPUEndTime - x.GPUStartTime) * 1e3, cur = atomic_load(&gGpuSum[i]);
		while (!atomic_compare_exchange_weak(&gGpuSum[i], &cur, cur + g)) {
		}
	}];
}

static void hookQueueClass(Class qc)
{
	pthread_mutex_lock(&gLock);
	NSString *n = NSStringFromClass(qc);
	BOOL done = [gHookedQueueClasses containsObject:n];
	if (!done)
		[gHookedQueueClasses addObject:n];
	pthread_mutex_unlock(&gLock);
	if (done)
		return;
	static IMP oCB, oCBU, oCBD;
	oCB = hook(qc, @selector(commandBuffer), ^id(id self) {
		double t0 = oracle_now_ms();
		id cb = ((id (*)(id, SEL))oCB)(self, @selector(commandBuffer));
		noteCB(self, cb, oracle_now_ms() - t0);
		return cb;
	});
	oCBU = hook(qc, @selector(commandBufferWithUnretainedReferences), ^id(id self) {
		double t0 = oracle_now_ms();
		id cb = ((id (*)(id, SEL))oCBU)(self, @selector(commandBufferWithUnretainedReferences));
		noteCB(self, cb, oracle_now_ms() - t0);
		return cb;
	});
	oCBD = hook(qc, @selector(commandBufferWithDescriptor:), ^id(id self, id d) {
		double t0 = oracle_now_ms();
		id cb = ((id (*)(id, SEL, id))oCBD)(self, @selector(commandBufferWithDescriptor:), d);
		noteCB(self, cb, oracle_now_ms() - t0);
		return cb;
	});
}

static void noteQueue(id q, const char *how, long asked)
{
	if (!q || queueIndex(q) >= 0)
		return;
	pthread_mutex_lock(&gLock);
	int i = gNQ < kQ ? gNQ++ : -1;
	if (i >= 0)
		gQueues[i] = (__bridge_retained void *)q;
	pthread_mutex_unlock(&gLock);
	if (i < 0)
		return;
	long ans = -1;
	if ([q respondsToSelector:NSSelectorFromString(@"maxCommandBufferCount")])
		ans = ((NSUInteger (*)(id, SEL))objc_msgSend)(q, NSSelectorFromString(@"maxCommandBufferCount"));
	printf("queue %d: %s asked %ld, answers maxCommandBufferCount %ld, class %s\n", i, how, asked, ans,
	       class_getName(object_getClass(q)));
	hookQueueClass(object_getClass(q));
}

static void hookDevice(void)
{
	gHookedQueueClasses = [NSMutableSet set];
	Class dc = object_getClass(MTLCreateSystemDefaultDevice());
	static IMP oQ, oQM, oQD;
	oQ = hook(dc, @selector(newCommandQueue), ^id(id self) {
		id q = ((id (*)(id, SEL))oQ)(self, @selector(newCommandQueue));
		noteQueue(q, "newCommandQueue", 0);
		return q;
	});
	oQM = hook(dc, @selector(newCommandQueueWithMaxCommandBufferCount:), ^id(id self, NSUInteger n) {
		id q = ((id (*)(id, SEL, NSUInteger))oQM)(self, @selector(newCommandQueueWithMaxCommandBufferCount:), n);
		noteQueue(q, "newCommandQueueWithMaxCommandBufferCount:", (long)n);
		return q;
	});
	oQD = hook(dc, @selector(newCommandQueueWithDescriptor:), ^id(id self, MTLCommandQueueDescriptor *d) {
		id q = ((id (*)(id, SEL, id))oQD)(self, @selector(newCommandQueueWithDescriptor:), d);
		noteQueue(q, "newCommandQueueWithDescriptor:", (long)d.maxCommandBufferCount);
		return q;
	});
	printf("# hooked device class %s\n", class_getName(dc));
}

static GLuint compile(GLenum type, const char *src)
{
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[1024];
		glGetShaderInfoLog(s, sizeof log, NULL, log);
		printf("  ! shader: %s\n", log);
		exit(2);
	}
	return s;
}

static GLint gLoops;
static GLuint gVbo;

static void draw(int loops, bool upload)
{
	static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
	if (upload)
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof quad, quad);
	glUniform1i(gLoops, loops);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

static void snapshot(int *made, double *gpu, int *done)
{
	for (int i = 0; i < gNQ; i++) {
		made[i] = atomic_load(&gMade[i]);
		gpu[i] = atomic_load(&gGpuSum[i]);
		done[i] = atomic_load(&gDone[i]);
		atomic_store(&gMaxOut[i], 0);
		atomic_store(&gBlocks[i], 0);
		atomic_store(&gFirstBlockOut[i], 0);
	}
}

static int cmpd(const void *a, const void *b)
{
	double x = *(const double *)a, y = *(const double *)b;
	return x < y ? -1 : x > y;
}

static void runArm(const char *name, int flushes, int draws, int loops, bool upload)
{
	int made0[kQ] = {0}, done0[kQ] = {0};
	double gpu0[kQ] = {0};
	glFinish();
	usleep(100000);
	snapshot(made0, gpu0, done0);
	double *ft = calloc(flushes, sizeof(double)), *at = calloc(flushes, sizeof(double));
	const double t0 = oracle_now_ms();
	for (int f = 0; f < flushes; f++) {
		for (int d = 0; d < draws; d++)
			draw(loops, upload);
		double a = oracle_now_ms();
		glFlush();
		ft[f] = oracle_now_ms() - a;
		int u = 0;
		for (int i = 0; i < gNQ; i++)
			u += atomic_load(&gOut[i]);
		at[f] = u;
	}
	glFinish();
	const double wall = oracle_now_ms() - t0;
	usleep(100000);
	qsort(ft, flushes, sizeof(double), cmpd);
	qsort(at, flushes, sizeof(double), cmpd);
	double gpuAll = 0;
	for (int i = 0; i < gNQ; i++) {
		int made = atomic_load(&gMade[i]) - made0[i], done = atomic_load(&gDone[i]) - done0[i];
		double g = atomic_load(&gGpuSum[i]) - gpu0[i];
		gpuAll += g;
		if (made == 0)
			continue;
		printf("arm %s queue %d: %d command buffers (%.2f a flush), uncompleted max %d, blocked creations %d "
		       "(uncompleted at the first %d), GPU mean %.2f ms\n",
		       name, i, made, (double)made / flushes, atomic_load(&gMaxOut[i]), atomic_load(&gBlocks[i]),
		       atomic_load(&gFirstBlockOut[i]), done ? g / done : 0);
	}
	printf("arm %s: %d flushes of %d draws; glFlush p50 %.3f p90 %.3f max %.3f ms; uncompleted after a flush p50 %.0f "
	       "max %.0f; wall %.1f ms (%.2f a flush), GPU %.1f ms\n",
	       name, flushes, draws, ft[flushes / 2], ft[flushes * 9 / 10], ft[flushes - 1], at[flushes / 2],
	       at[flushes - 1], wall, wall / flushes, gpuAll);
	free(ft);
	free(at);
}

int main(int argc, char **argv)
{
	setvbuf(stdout, NULL, _IOLBF, 0);
	@autoreleasepool {
		const double target = argc > 1 ? atof(argv[1]) : 5.0;
		const int flushes = argc > 2 ? atoi(argv[2]) : 120;
		oracle_header("oracle_gldepth", MTLCreateSystemDefaultDevice());
		hookDevice();
		CGLPixelFormatAttribute attrs[] = {kCGLPFAAccelerated, kCGLPFANoRecovery, kCGLPFAColorSize, 24,
		                                   kCGLPFAAlphaSize, 8, 0};
		CGLPixelFormatObj pf = NULL;
		GLint npf = 0;
		if (CGLChoosePixelFormat(attrs, &pf, &npf) != kCGLNoError || !pf) {
			printf("  ! no pixel format\n");
			return 2;
		}
		CGLContextObj ctx = NULL;
		CGLCreateContext(pf, NULL, &ctx);
		CGLSetCurrentContext(ctx);
		printf("# GL renderer \"%s\", version \"%s\"\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
		GLuint fbo, rb;
		glGenFramebuffersEXT(1, &fbo);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
		glGenRenderbuffersEXT(1, &rb);
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, rb);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_RGBA8, kW, kH);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_RENDERBUFFER_EXT, rb);
		glViewport(0, 0, kW, kH);
		const char *vs = "attribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }";
		const char *fs = "uniform int loops; void main() { vec2 x = gl_FragCoord.xy * 0.001;\n"
		                 "  for (int i = 0; i < loops; i++) x = fract(x * 1.0001 + vec2(0.123, 0.456)) * 1.7;\n"
		                 "  gl_FragColor = vec4(x, 0.0, 1.0); }";
		GLuint prog = glCreateProgram();
		glAttachShader(prog, compile(GL_VERTEX_SHADER, vs));
		glAttachShader(prog, compile(GL_FRAGMENT_SHADER, fs));
		glBindAttribLocation(prog, 0, "p");
		glLinkProgram(prog);
		glUseProgram(prog);
		gLoops = glGetUniformLocation(prog, "loops");
		static const float quad[] = {-1, -1, 1, -1, -1, 1, 1, 1};
		glGenBuffers(1, &gVbo);
		glBindBuffer(GL_ARRAY_BUFFER, gVbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		// Calibrate by wall time (an Apple GPU overlaps command buffers, so their GPU times do not add up): 2000
		// loops, one draw a flush, 10 flushes and a glFinish, after 3 to warm up.
		for (int k = 0; k < 3; k++) {
			draw(2000, false);
			glFlush();
		}
		glFinish();
		double c0 = oracle_now_ms();
		for (int k = 0; k < 10; k++) {
			draw(2000, false);
			glFlush();
		}
		glFinish();
		const double per = (oracle_now_ms() - c0) / 10;
		int n = 0;
		for (int i = 0; i < gNQ; i++)
			n += atomic_load(&gDone[i]);
		if (n == 0) {
			printf("  ! no Metal command buffers seen: GL is not over Metal here\n");
			return 0;
		}
		int loops = (int)(2000.0 * target / per), loops8 = loops / 8 ? loops / 8 : 1;
		printf("# calibrate: 2000 loops %.3f ms of wall a flush; heavy draw %d loops (target %.1f ms)\n", per, loops,
		       target);
		runArm("draw1", flushes, 1, loops, false);
		runArm("draw8", flushes, 8, loops8, true);
		CGLSetCurrentContext(NULL);
		CGLDestroyContext(ctx);
	}
	return 0;
}
