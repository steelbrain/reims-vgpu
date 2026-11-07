// gl_window: an on-screen OpenGL context, as a windowed GL app makes one. It opens a 256x256
// window holding an NSOpenGLView, clears it to (64, 128, 191) and draws a green triangle in
// it each frame, swaps, and after N frames (default 120) reads the back buffer's centre and a
// corner with glReadPixels, prints the renderer and the two pixels, and exits.
//   gl_window [-l] [-c] [-r] [-n FRAMES] [-t SECONDS]
//     -l   the view is layer-backed (wantsLayer, wantsBestResolutionOpenGLSurface), as an app
//          linked against a current SDK gets: CoreAnimation presents its frames;
//          without -l the window's surface is the context's drawable (CGL's surface path)
//     -c   a 3.2 core profile context (default legacy)
//     -r   at the middle frame the window's content resizes to 320x200, so the frames after it
//          and the read draw at the new size
//     -n   frames to draw
//     -t   seconds to keep the window open after the last frame (default 2), for a screen grab
// It prints the pixel format's renderer ID, GL_RENDERER and GL_VERSION, each frame's
// glGetError where it is not 0, the time per frame, and "window: pass" where the centre reads
// (0, 255, 0) and the corner (64, 128, 191). It must run in a GUI session (from SSH,
// `launchctl asuser $(id -u) ./gl_window`).
// Our own code.
#define GL_SILENCE_DEPRECATION 1
#import <Cocoa/Cocoa.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl3.h>
#include <getopt.h>
#include <mach/mach_time.h>

static BOOL gLayer = NO, gCore = NO, gResize = NO;
static int gFrames = 120;
static double gHold = 2;

static GLuint compile(GLenum type, const char *src) {
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char log[1024];
		glGetShaderInfoLog(s, sizeof log, NULL, log);
		printf("shader: %s\n", log);
	}
	return s;
}

@interface GLWView : NSOpenGLView
@property GLuint program, vao, vbo;
@property int frameIndex;
@end

@implementation GLWView
- (void)prepareOpenGL {
	[super prepareOpenGL];
	GLint swap = 1;
	[self.openGLContext setValues:&swap forParameter:NSOpenGLContextParameterSwapInterval];
	printf("GL_RENDERER %s\nGL_VERSION %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
	const char *vs = gCore ? "#version 150\nin vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }\n"
	                       : "#version 120\nattribute vec2 p; void main() { gl_Position = vec4(p, 0.0, 1.0); }\n";
	const char *fs = gCore ? "#version 150\nuniform vec4 c; out vec4 o; void main() { o = c; }\n"
	                       : "#version 120\nuniform vec4 c; void main() { gl_FragColor = c; }\n";
	GLuint p = glCreateProgram();
	glAttachShader(p, compile(GL_VERTEX_SHADER, vs));
	glAttachShader(p, compile(GL_FRAGMENT_SHADER, fs));
	glBindAttribLocation(p, 0, "p");
	glLinkProgram(p);
	self.program = p;
	const float v[6] = {-0.5f, -0.5f, 0.5f, -0.5f, 0.0f, 0.5f};
	GLuint vbo = 0, vao = 0;
	if (gCore) {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	self.vbo = vbo;
	self.vao = vao;
}

- (void)drawRect:(NSRect)rect {
	[self.openGLContext makeCurrentContext];
	NSRect px = [self convertRectToBacking:self.bounds];
	glViewport(0, 0, (GLsizei)px.size.width, (GLsizei)px.size.height);
	glClearColor(64 / 255.0f, 128 / 255.0f, 191 / 255.0f, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glUseProgram(self.program);
	glUniform4f(glGetUniformLocation(self.program, "c"), 0, 1, 0, 1);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	GLenum e = glGetError();
	if (e) printf("frame %d: glGetError 0x%x\n", self.frameIndex, e);
	if (self.frameIndex == gFrames - 1) {
		glFinish();
		uint8_t centre[4] = {0}, corner[4] = {0};
		glReadPixels((GLint)px.size.width / 2, (GLint)px.size.height / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, centre);
		glReadPixels(2, 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, corner);
		BOOL ok = centre[0] == 0 && centre[1] == 255 && centre[2] == 0 && corner[0] == 64 && corner[1] == 128 &&
		          corner[2] == 191;
		printf("back buffer %gx%g: centre (%u, %u, %u, %u), corner (%u, %u, %u, %u)\nwindow: %s\n", px.size.width,
		       px.size.height, centre[0], centre[1], centre[2], centre[3], corner[0], corner[1], corner[2], corner[3],
		       ok ? "pass" : "FAIL");
	}
	[self.openGLContext flushBuffer];
}
@end

@interface GLWDelegate : NSObject <NSApplicationDelegate>
@property(strong) NSWindow *window;
@property(strong) GLWView *view;
@end

@implementation GLWDelegate
- (void)applicationDidFinishLaunching:(NSNotification *)n {
	NSOpenGLPixelFormatAttribute attrs[] = {NSOpenGLPFAAccelerated, NSOpenGLPFADoubleBuffer, NSOpenGLPFAColorSize, 24,
	                                        NSOpenGLPFAAlphaSize, 8, NSOpenGLPFAOpenGLProfile,
	                                        gCore ? NSOpenGLProfileVersion3_2Core : NSOpenGLProfileVersionLegacy, 0};
	NSOpenGLPixelFormat *pf = [[NSOpenGLPixelFormat alloc] initWithAttributes:attrs];
	if (!pf) {
		printf("no accelerated pixel format\n");
		exit(1);
	}
	GLint renderer = 0;
	[pf getValues:&renderer forAttribute:NSOpenGLPFARendererID forVirtualScreen:0];
	printf("pixel format: renderer 0x%08x, layer-backed %d, core %d\n", renderer, gLayer, gCore);
	self.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(100, 100, 256, 256)
	                                          styleMask:NSWindowStyleMaskTitled
	                                            backing:NSBackingStoreBuffered
	                                              defer:NO];
	self.window.title = @"gl_window";
	self.view = [[GLWView alloc] initWithFrame:NSMakeRect(0, 0, 256, 256) pixelFormat:pf];
	self.view.wantsBestResolutionOpenGLSurface = gLayer;
	self.view.wantsLayer = gLayer;
	self.window.contentView = self.view;
	[self.window makeKeyAndOrderFront:nil];
	[NSApp activateIgnoringOtherApps:YES];
	uint64_t t0 = mach_absolute_time();
	__block int frame = 0;
	[NSTimer scheduledTimerWithTimeInterval:0
	                                repeats:YES
	                                  block:^(NSTimer *t) {
		                                  if (gResize && frame == gFrames / 2) {
			                                  [self.window setContentSize:NSMakeSize(320, 200)];
			                                  printf("frame %d: resized to 320x200\n", frame);
		                                  }
		                                  self.view.frameIndex = frame;
		                                  [self.view display];
		                                  if (++frame == gFrames) {
			                                  [t invalidate];
			                                  mach_timebase_info_data_t tb;
			                                  mach_timebase_info(&tb);
			                                  double ms = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;
			                                  printf("%d frames, %.2f ms each\n", gFrames, ms / gFrames);
			                                  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(gHold * 1e9)),
			                                                 dispatch_get_main_queue(), ^{
				                                                 exit(0);
			                                                 });
		                                  }
	                                  }];
}
@end

int main(int argc, char **argv) {
	@autoreleasepool {
		setvbuf(stdout, NULL, _IOLBF, 0);
		int c;
		while ((c = getopt(argc, argv, "lcrn:t:")) != -1) {
			if (c == 'l') gLayer = YES;
			else if (c == 'c') gCore = YES;
			else if (c == 'r') gResize = YES;
			else if (c == 'n') gFrames = atoi(optarg);
			else if (c == 't') gHold = atof(optarg);
			else return 2;
		}
		NSApplication *app = [NSApplication sharedApplication];
		[app setActivationPolicy:NSApplicationActivationPolicyRegular];
		GLWDelegate *d = [GLWDelegate new];
		app.delegate = d;
		[app run];
	}
	return 0;
}
