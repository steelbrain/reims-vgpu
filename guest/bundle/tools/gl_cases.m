// gl_cases: GL shaders of the features apps use, each drawn into a framebuffer object and its
// centre pixel checked against the value the GL spec gives for it. Its use is to compare a
// device's GL with another's: the same cases on Metal's own GL path and on the device's lowering
// (tools/gl_inject.mm) must give the same pixels.
//   gl_cases [-v] [CASE...]   runs the named cases (default all); -v prints each shader's info log
// Each case prints "NAME: pass" or "NAME: FAIL (got r,g,b,a, want r,g,b,a)" (with the link log
// where the program does not link), and the last line counts them. The cases run in a 4.1 core
// context, except those named legacy-*, which run in a legacy one. It makes no window.
// Our own code.
#define GL_SILENCE_DEPRECATION 1
#import <Foundation/Foundation.h>
#import <OpenGL/OpenGL.h>
#import <OpenGL/gl3.h>
#include <math.h>

// Legacy entry points gl3.h leaves out.
extern void glBegin(GLenum);
extern void glEnd(void);
extern void glVertex2f(GLfloat, GLfloat);
extern void glTexCoord2f(GLfloat, GLfloat);
extern void glColor4f(GLfloat, GLfloat, GLfloat, GLfloat);
extern void glEnable(GLenum);
#define GL_QUADS 0x0007
#define GL_POLYGON 0x0009
#define GL_POINT_SPRITE 0x8861
#define GL_TEXTURE_2D_LEGACY 0x0DE1

static BOOL gVerbose = NO;

typedef struct {
	const char *name;
	const char *vs, *fs;
	void (*setup)(GLuint program);
	GLenum primitive; // what the quad's draw uses: GL_TRIANGLE_STRIP by default
	uint8_t want[4];
	uint8_t tolerance;
} Case;

static GLuint compile(GLenum type, const char *src, BOOL *ok) {
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, NULL);
	glCompileShader(s);
	GLint status = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &status);
	if (!status || gVerbose) {
		char log[2048] = {0};
		glGetShaderInfoLog(s, sizeof log, NULL, log);
		if (log[0]) printf("    %s shader log: %s\n", type == GL_VERTEX_SHADER ? "vertex" : "fragment", log);
	}
	if (!status) *ok = NO;
	return s;
}

// --- setups ---------------------------------------------------------------------------------

static GLuint texture2D(GLenum internal, GLenum format, GLenum type, int w, int h, const void *pixels, GLenum filter) {
	GLuint t = 0;
	glGenTextures(1, &t);
	glBindTexture(GL_TEXTURE_2D, t);
	glTexImage2D(GL_TEXTURE_2D, 0, (GLint)internal, w, h, 0, format, type, pixels);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (GLint)filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (GLint)filter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	return t;
}

// A 2x2 texture: red, green / blue, white; the centre texel sampled with GL_NEAREST at (0.75, 0.75)
// is white.
static void setupTexture(GLuint p) {
	const uint8_t px[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
	glActiveTexture(GL_TEXTURE0);
	texture2D(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 2, 2, px, GL_NEAREST);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// Linear filtering of a 2x1 texture, black then white, sampled at its middle: grey.
static void setupLinear(GLuint p) {
	const uint8_t px[8] = {0, 0, 0, 255, 255, 255, 255, 255};
	glActiveTexture(GL_TEXTURE0);
	texture2D(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 2, 1, px, GL_LINEAR);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// Two textures on two units: their sum.
static void setupTwoUnits(GLuint p) {
	const uint8_t a[4] = {200, 0, 0, 255}, b[4] = {0, 100, 0, 255};
	glActiveTexture(GL_TEXTURE0);
	texture2D(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 1, 1, a, GL_NEAREST);
	glActiveTexture(GL_TEXTURE1);
	texture2D(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 1, 1, b, GL_NEAREST);
	glUniform1i(glGetUniformLocation(p, "a"), 0);
	glUniform1i(glGetUniformLocation(p, "b"), 1);
	glActiveTexture(GL_TEXTURE0);
}

// A float texture read with texelFetch.
static void setupFetch(GLuint p) {
	const float px[8] = {0.25f, 0.5f, 0.75f, 1.0f, 1, 0, 1, 1};
	glActiveTexture(GL_TEXTURE0);
	texture2D(GL_RGBA32F, GL_RGBA, GL_FLOAT, 2, 1, px, GL_NEAREST);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// A depth texture compared through a shadow sampler: 0.5 stored, compared at 0.25 (LEQUAL): 1.
static void setupShadow(GLuint p) {
	const float d = 0.5f;
	glActiveTexture(GL_TEXTURE0);
	texture2D(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 1, 1, &d, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// A cube map whose +X face is red and the rest blue, sampled along +X.
static void setupCube(GLuint p) {
	GLuint t = 0;
	glGenTextures(1, &t);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_CUBE_MAP, t);
	for (int f = 0; f < 6; f++) {
		const uint8_t px[4] = {f == 0 ? 255 : 0, 0, f == 0 ? 0 : 255, 255};
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + (GLenum)f, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// A 2D array texture of two layers, red and green: layer 1.
static void setupArray(GLuint p) {
	GLuint t = 0;
	glGenTextures(1, &t);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D_ARRAY, t);
	const uint8_t px[8] = {255, 0, 0, 255, 0, 255, 0, 255};
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 1, 1, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// A 3D texture of two slices, red and blue: the second.
static void setup3D(GLuint p) {
	GLuint t = 0;
	glGenTextures(1, &t);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, t);
	const uint8_t px[8] = {255, 0, 0, 255, 0, 0, 255, 255};
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA8, 1, 1, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// Uniforms of several types and an array.
static void setupUniforms(GLuint p) {
	glUniform1f(glGetUniformLocation(p, "f"), 0.5f);
	glUniform2f(glGetUniformLocation(p, "v2"), 0.25f, 0.75f);
	glUniform1i(glGetUniformLocation(p, "i"), 3);
	const float arr[4] = {0.1f, 0.2f, 0.3f, 0.4f};
	glUniform1fv(glGetUniformLocation(p, "arr"), 4, arr);
	const float m[16] = {0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
	glUniformMatrix4fv(glGetUniformLocation(p, "m"), 1, GL_FALSE, m);
}

// A uniform block in a buffer at binding 2.
static void setupBlock(GLuint p) {
	GLuint b = 0;
	glGenBuffers(1, &b);
	const float data[8] = {0.0f, 0.5f, 1.0f, 1.0f, 0.25f, 0, 0, 0};
	glBindBuffer(GL_UNIFORM_BUFFER, b);
	glBufferData(GL_UNIFORM_BUFFER, sizeof data, data, GL_STATIC_DRAW);
	GLuint index = glGetUniformBlockIndex(p, "B");
	glUniformBlockBinding(p, index, 2);
	glBindBufferBase(GL_UNIFORM_BUFFER, 2, b);
}

// A texture buffer of four RGBA32F texels, read at 2.
static void setupTexBuffer(GLuint p) {
	GLuint b = 0, t = 0;
	const float data[16] = {0, 0, 0, 1, 0, 0, 0, 1, 0.0f, 0.75f, 0.25f, 1.0f, 0, 0, 0, 1};
	glGenBuffers(1, &b);
	glBindBuffer(GL_TEXTURE_BUFFER, b);
	glBufferData(GL_TEXTURE_BUFFER, sizeof data, data, GL_STATIC_DRAW);
	glGenTextures(1, &t);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER, t);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, b);
	glUniform1i(glGetUniformLocation(p, "t"), 0);
}

// Blending: the quad's 50% red over the clear's blue.
static void setupBlend(GLuint p) {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

// Depth test: a far quad drawn first is covered by the near one the case draws.
static void setupDepth(GLuint p) {
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
}

// Stencil: the quad writes 1; the case's second draw passes only where the stencil is 1.
static void setupStencil(GLuint p) {
	glEnable(GL_STENCIL_TEST);
	glStencilFunc(GL_ALWAYS, 1, 0xff);
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
}

// Clip distance 0 on: the case's quad is cut at x = -0.5.
static void setupClip(GLuint p) {
	glEnable(GL_CLIP_DISTANCE0);
}

// Legacy fixed function: no program, the 2x2 texture of setupTexture on unit 0, which the quad's
// texture coordinate (0.75, 0.75) reads white.
static void setupLegacyTexture(GLuint p) {
	const uint8_t px[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
	glActiveTexture(GL_TEXTURE0);
	texture2D(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 2, 2, px, GL_NEAREST);
}

// --- the cases ------------------------------------------------------------------------------

#define VS41 "#version 410\nin vec2 p; out vec2 uv; void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.5, 1.0); }\n"
#define FS41(body) "#version 410\nin vec2 uv; out vec4 o;\n" body

static const Case kCases[] = {
    {"constant", VS41, FS41("void main() { o = vec4(0.25, 0.5, 0.75, 1.0); }"), NULL, 0, {64, 128, 191, 255}, 1},
    {"varying", VS41, FS41("void main() { o = vec4(uv, 0.0, 1.0); }"), NULL, 0, {128, 128, 0, 255}, 2},
    {"fragcoord", VS41, FS41("void main() { o = vec4(gl_FragCoord.xy / 64.0, 0.0, 1.0); }"), NULL, 0, {129, 129, 0, 255}, 2},
    {"math", VS41,
     FS41("void main() { float a = pow(2.0, 3.0) / 16.0; float b = smoothstep(0.0, 1.0, 0.5); float c = "
          "clamp(exp(0.0) - 0.75, 0.0, 1.0); o = vec4(a, b, c, 1.0); }"),
     NULL, 0, {128, 128, 64, 255}, 1},
    {"integers", VS41,
     FS41("void main() { int x = 13; int y = (x << 2) ^ 5; uint z = uint(y) % 7u; o = vec4(float(y & 255) / 255.0, "
          "float(z) / 6.0, float(x / 4) / 4.0, 1.0); }"),
     NULL, 0, {49, 0, 191, 255}, 1},
    {"loop", VS41,
     FS41("void main() { float s = 0.0; for (int i = 0; i < 10; i++) { if (i == 7) break; s += 0.05; } o = vec4(s, 0.0, "
          "0.0, 1.0); }"),
     NULL, 0, {89, 0, 0, 255}, 1},
    {"discard", VS41, FS41("void main() { if (uv.x > 0.25) discard; o = vec4(1.0); }"), NULL, 0, {64, 128, 191, 255}, 1},
    {"derivatives", VS41, FS41("void main() { o = vec4(dFdx(uv.x) * 64.0, abs(dFdy(uv.y)) * 64.0, fwidth(uv.x) * 64.0, 1.0); }"),
     NULL, 0, {255, 255, 255, 255}, 3},
    {"uniforms", VS41,
     "#version 410\nin vec2 uv; out vec4 o; uniform float f; uniform vec2 v2; uniform int i; uniform float arr[4]; uniform "
     "mat4 m;\nvoid main() { vec4 r = m * vec4(f, v2.x, 0.0, 1.0); o = vec4(r.x, r.y, arr[i], 1.0); }\n",
     setupUniforms, 0, {64, 128, 102, 255}, 1},
    {"block", VS41,
     "#version 410\nin vec2 uv; out vec4 o; layout(std140) uniform B { vec4 c; float k; };\nvoid main() { o = vec4(c.rgb * "
     "(1.0 - k), 1.0); }\n",
     setupBlock, 0, {0, 96, 191, 255}, 1},
    {"texture", VS41, FS41("uniform sampler2D t; void main() { o = texture(t, vec2(0.75, 0.75)); }"), setupTexture, 0,
     {255, 255, 255, 255}, 0},
    {"linear", VS41, FS41("uniform sampler2D t; void main() { o = texture(t, vec2(0.5, 0.5)); }"), setupLinear, 0,
     {128, 128, 128, 255}, 2},
    {"two-units", VS41, FS41("uniform sampler2D a; uniform sampler2D b; void main() { o = texture(a, uv) + texture(b, uv); }"),
     setupTwoUnits, 0, {200, 100, 0, 255}, 1},
    {"texelfetch", VS41, FS41("uniform sampler2D t; void main() { o = texelFetch(t, ivec2(0, 0), 0); }"), setupFetch, 0,
     {64, 128, 191, 255}, 1},
    {"shadow", VS41, FS41("uniform sampler2DShadow t; void main() { float s = texture(t, vec3(0.5, 0.5, 0.25)); o = vec4(s, 0.0, 0.0, 1.0); }"),
     setupShadow, 0, {255, 0, 0, 255}, 0},
    {"cube", VS41, FS41("uniform samplerCube t; void main() { o = texture(t, vec3(1.0, 0.0, 0.0)); }"), setupCube, 0,
     {255, 0, 0, 255}, 0},
    {"array", VS41, FS41("uniform sampler2DArray t; void main() { o = texture(t, vec3(0.5, 0.5, 1.0)); }"), setupArray, 0,
     {0, 255, 0, 255}, 0},
    {"3d", VS41, FS41("uniform sampler3D t; void main() { o = texture(t, vec3(0.5, 0.5, 0.75)); }"), setup3D, 0,
     {0, 0, 255, 255}, 0},
    {"texbuffer", VS41, FS41("uniform samplerBuffer t; void main() { o = texelFetch(t, 2); }"), setupTexBuffer, 0,
     {0, 191, 64, 255}, 1},
    {"flat", "#version 410\nin vec2 p; flat out int k; void main() { k = 5; gl_Position = vec4(p, 0.5, 1.0); }\n",
     "#version 410\nflat in int k; out vec4 o; void main() { o = vec4(float(k) / 5.0, 0.0, 0.0, 1.0); }\n", NULL, 0,
     {255, 0, 0, 255}, 0},
    {"fragdepth", VS41, FS41("void main() { gl_FragDepth = 0.25; o = vec4(gl_FragCoord.z, 0.0, 0.0, 1.0); }"), setupDepth, 0,
     {191, 0, 0, 255}, 2},
    {"blend", VS41, FS41("void main() { o = vec4(1.0, 0.0, 0.0, 0.5); }"), setupBlend, 0, {160, 64, 96, 191}, 2},
    {"depth", VS41, FS41("void main() { o = vec4(0.0, 1.0, 0.0, 1.0); }"), setupDepth, 0, {0, 255, 0, 255}, 0},
    {"stencil", VS41, FS41("void main() { o = vec4(1.0, 1.0, 0.0, 1.0); }"), setupStencil, 0, {255, 255, 0, 255}, 0},
    {"clipdistance",
     "#version 410\nin vec2 p; out float gl_ClipDistance[1]; void main() { gl_ClipDistance[0] = -p.x - 0.5; gl_Position = "
     "vec4(p, 0.5, 1.0); }\n",
     FS41("void main() { o = vec4(1.0); }"), setupClip, 0, {64, 128, 191, 255}, 1},
    {"legacy-color", "#version 120\nattribute vec2 p; varying vec4 c; void main() { c = vec4(0.25, 0.5, 0.75, 1.0); gl_Position = vec4(p, 0.5, 1.0); }\n",
     "#version 120\nvarying vec4 c; void main() { gl_FragColor = c; }\n", NULL, 0, {64, 128, 191, 255}, 1},
    {"legacy-texture", "#version 120\nattribute vec2 p; varying vec2 uv; void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.5, 1.0); }\n",
     "#version 120\nuniform sampler2D t; varying vec2 uv; void main() { gl_FragColor = texture2D(t, vec2(0.75, 0.75)); }\n",
     setupTexture, 0, {255, 255, 255, 255}, 0},
    {"legacy-fixed-quads", NULL, NULL, setupLegacyTexture, GL_QUADS, {255, 255, 255, 255}, 0},
    {"legacy-fixed-polygon", NULL, NULL, NULL, GL_POLYGON, {255, 128, 0, 255}, 1},
};

// --- running one ----------------------------------------------------------------------------

static CGLContextObj makeContext(BOOL legacy) {
	CGLPixelFormatAttribute attrs[] = {kCGLPFAAccelerated,
	                                   kCGLPFAOpenGLProfile,
	                                   (CGLPixelFormatAttribute)(legacy ? kCGLOGLPVersion_Legacy : kCGLOGLPVersion_GL4_Core),
	                                   kCGLPFAColorSize,
	                                   (CGLPixelFormatAttribute)24,
	                                   (CGLPixelFormatAttribute)0};
	CGLPixelFormatObj pf = NULL;
	GLint n = 0;
	if (CGLChoosePixelFormat(attrs, &pf, &n) != kCGLNoError || !pf) return NULL;
	CGLContextObj ctx = NULL;
	CGLCreateContext(pf, NULL, &ctx);
	CGLDestroyPixelFormat(pf);
	return ctx;
}

static BOOL runCase(const Case *c) {
	const BOOL legacy = strncmp(c->name, "legacy-", 7) == 0;
	CGLContextObj ctx = makeContext(legacy);
	if (!ctx) {
		printf("%s: FAIL (no context)\n", c->name);
		return NO;
	}
	CGLSetCurrentContext(ctx);
	// A 64x64 RGBA8 framebuffer with a depth-stencil buffer.
	GLuint fb = 0, colour = 0, ds = 0;
	glGenFramebuffers(1, &fb);
	glBindFramebuffer(GL_FRAMEBUFFER, fb);
	glGenRenderbuffers(1, &colour);
	glBindRenderbuffer(GL_RENDERBUFFER, colour);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 64, 64);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, colour);
	glGenRenderbuffers(1, &ds);
	glBindRenderbuffer(GL_RENDERBUFFER, ds);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 64, 64);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, ds);
	glViewport(0, 0, 64, 64);
	glClearColor(64 / 255.0f, 128 / 255.0f, 191 / 255.0f, 1);
	glClearDepth(1);
	glClearStencil(0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	GLuint program = 0;
	BOOL ok = YES;
	if (c->vs) {
		program = glCreateProgram();
		glAttachShader(program, compile(GL_VERTEX_SHADER, c->vs, &ok));
		glAttachShader(program, compile(GL_FRAGMENT_SHADER, c->fs, &ok));
		glBindAttribLocation(program, 0, "p");
		glLinkProgram(program);
		GLint linked = 0;
		glGetProgramiv(program, GL_LINK_STATUS, &linked);
		if (!linked) {
			char log[2048] = {0};
			glGetProgramInfoLog(program, sizeof log, NULL, log);
			printf("%s: FAIL (no link: %s)\n", c->name, log);
			CGLSetCurrentContext(NULL);
			CGLDestroyContext(ctx);
			return NO;
		}
		glUseProgram(program);
	}
	if (c->setup) c->setup(program);

	const float quad[8] = {-1, -1, 1, -1, -1, 1, 1, 1};
	GLuint vao = 0, vbo = 0;
	if (c->vs) {
		if (!legacy) {
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);
		}
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		if (strcmp(c->name, "stencil") == 0) {
			// The second draw passes where the stencil is 1: everywhere the first drew.
			glStencilFunc(GL_EQUAL, 1, 0xff);
			glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
		}
	} else {
		// Legacy fixed function: a full quad of glBegin's.
		if (c->setup) {
			glEnable(GL_TEXTURE_2D_LEGACY);
		}
		// White over the texture (modulated), else orange.
		glBegin(c->primitive);
		if (c->setup) {
			glTexCoord2f(0.75f, 0.75f);
			glColor4f(1, 1, 1, 1);
		} else
			glColor4f(1, 0.5f, 0, 1);
		glVertex2f(-1, -1);
		glVertex2f(1, -1);
		glVertex2f(1, 1);
		glVertex2f(-1, 1);
		glEnd();
	}
	glFinish();
	uint8_t got[4] = {0};
	// The centre for every case, but discard's and clip distance's, whose quads cover only the left
	// part: a pixel right of them.
	const GLint x = strcmp(c->name, "discard") == 0 || strcmp(c->name, "clipdistance") == 0 ? 48 : 32;
	glReadPixels(x, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);
	GLenum e = glGetError();
	BOOL pass = e == GL_NO_ERROR;
	for (int i = 0; i < 4; i++) pass = pass && abs((int)got[i] - (int)c->want[i]) <= c->tolerance;
	if (pass)
		printf("%s: pass\n", c->name);
	else
		printf("%s: FAIL (got %u,%u,%u,%u, want %u,%u,%u,%u, glGetError 0x%x)\n", c->name, got[0], got[1], got[2], got[3],
		       c->want[0], c->want[1], c->want[2], c->want[3], e);
	CGLSetCurrentContext(NULL);
	CGLDestroyContext(ctx);
	return pass;
}

int main(int argc, char **argv) {
	@autoreleasepool {
		setvbuf(stdout, NULL, _IOLBF, 0);
		int first = 1;
		if (argc > 1 && strcmp(argv[1], "-v") == 0) {
			gVerbose = YES;
			first = 2;
		}
		int passed = 0, ran = 0;
		for (size_t i = 0; i < sizeof kCases / sizeof kCases[0]; i++) {
			BOOL wanted = argc <= first;
			for (int a = first; a < argc; a++) wanted = wanted || strcmp(argv[a], kCases[i].name) == 0;
			if (!wanted) continue;
			ran++;
			passed += runCase(&kCases[i]) ? 1 : 0;
		}
		printf("gl_cases: %d of %d pass\n", passed, ran);
		return passed == ran ? 0 : 1;
	}
}
