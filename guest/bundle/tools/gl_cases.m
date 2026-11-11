// gl_cases: GL shaders of the features apps use, each drawn into a framebuffer object and its
// centre pixel checked against the value the GL spec gives for it. Its use is to compare a
// device's GL with another's: the same cases on Metal's own GL path and on the device's lowering
// (tools/gl_inject.mm) must give the same pixels.
//   gl_cases [-v] [CASE...]   runs the named cases (default all); -v prints each shader's info log
// Each case prints "NAME: pass" or "NAME: FAIL (got r,g,b,a, want r,g,b,a)" (with the link log
// where the program does not link), and the last line counts them. The cases run in a 4.1 core
// context, except those named legacy-*, which run in a legacy one. Those named msaa-* draw into a
// framebuffer of 4 samples, resolved into the one read, and those named point-* draw points. A
// case with a draw of its own (the provoking-vertex and primitive-restart cases) draws with it in
// place of the quad, and may check more pixels than the centre; one whose GL lacks what it draws
// prints "NAME: skip (why)" and is not counted. GL_CASES_RENDERER=TEXT runs the cases on the first
// renderer whose GL_RENDERER string holds TEXT (offline renderers included), and -v prints the
// renderer each context runs on. It makes no window.
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
extern void glAlphaFunc(GLenum, GLfloat);
extern void glShadeModel(GLenum);
#define GL_QUADS 0x0007
#define GL_POLYGON 0x0009
#define GL_POINT_SPRITE 0x8861
#define GL_TEXTURE_2D_LEGACY 0x0DE1
#define GL_ALPHA_TEST 0x0BC0
#define GL_VERTEX_PROGRAM_TWO_SIDE 0x8643
#define GL_FLAT_LEGACY 0x1D00
#define GL_PRIMITIVE_RESTART_FIXED_INDEX_CASES 0x8D69

static BOOL gVerbose = NO;

typedef struct {
	const char *name;
	const char *vs, *fs;
	void (*setup)(GLuint program);
	GLenum primitive; // what the quad's draw uses: GL_TRIANGLE_STRIP by default
	uint8_t want[4];
	uint8_t tolerance;
	// A draw of the case's own in place of the quad: 1 where it drew, 0 where the context lacks
	// what it needs (the case is skipped); `skip` says what.
	int (*draw)(GLuint program, const char **skip);
	// The pixel `want` is read at, where not the default; and up to two more, each with its value.
	GLint x, y;
	struct {
		GLint x, y;
		uint8_t want[4];
	} more[2];
	int moreCount;
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

// A logic operation: the quad's codes XORed into the clear's.
static void setupLogicOp(GLuint p) {
	glEnable(GL_COLOR_LOGIC_OP);
	glLogicOp(GL_XOR);
}

// The sample mask: two samples of four keep the quad's colour, so the resolve averages it with
// the clear's.
static void setupSampleMask(GLuint p) {
	glEnable(GL_SAMPLE_MASK);
	glSampleMaski(0, 0x3);
}

// Sample coverage of a half: two samples of four keep the quad's colour (where the coverage is
// spread over a quad of pixels, two in each).
static void setupSampleCoverage(GLuint p) {
	glEnable(GL_SAMPLE_COVERAGE);
	glSampleCoverage(0.5f, GL_FALSE);
}

// A point of the vertex function's size, its coordinates from the default upper-left origin.
static void setupPointSize(GLuint p) {
	glEnable(GL_PROGRAM_POINT_SIZE);
}

// A point of the vertex function's size, its coordinates from the lower-left origin.
static void setupPointLowerLeft(GLuint p) {
	glEnable(GL_PROGRAM_POINT_SIZE);
	glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
}

// Legacy alpha test: the quad's alpha of 0.25 fails a test of greater than 0.5, so the clear stays.
static void setupAlphaTest(GLuint p) {
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_GREATER, 0.5f);
}

// Legacy two-sided colour: the quad, wound clockwise here, is a back face, so it takes the back
// colour.
static void setupTwoSide(GLuint p) {
	glEnable(GL_VERTEX_PROGRAM_TWO_SIDE);
	glFrontFace(GL_CW);
}

// Legacy fixed function: no program, the 2x2 texture of setupTexture on unit 0, which the quad's
// texture coordinate (0.75, 0.75) reads white.
static void setupLegacyTexture(GLuint p) {
	const uint8_t px[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
	glActiveTexture(GL_TEXTURE0);
	texture2D(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 2, 2, px, GL_NEAREST);
}

// --- draws of their own: the provoking vertex and primitive restart --------------------------

// Vertices of position (x, y) and colour (r, g, b, a), from `v` (6 floats each), into a new vertex
// array: position at attribute 0, colour at the program's `col` where it has one.
static void vertices(GLuint program, const float *v, int count) {
	GLuint vao = 0, vbo = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(sizeof(float) * 6 * (size_t)count), v, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 0);
	const GLint col = glGetAttribLocation(program, "col");
	if (col >= 0) {
		glEnableVertexAttribArray((GLuint)col);
		glVertexAttribPointer((GLuint)col, 4, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (const void *)(2 * sizeof(float)));
	}
}

// One triangle over the centre, its vertices red, green and blue: its flat colour is the provoking
// vertex's, the third's (blue) under GL's default last-vertex convention, the first's (red) under
// the first-vertex one.
static const float kTriangle[18] = {-1, -1, 1, 0, 0, 1, 3, -1, 0, 1, 0, 1, -1, 3, 0, 0, 1, 1};
static int drawProvokingTriangle(GLuint p, const char **skip) {
	vertices(p, kTriangle, 3);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	return 1;
}

// A strip of two triangles over the framebuffer: triangle 0 (vertices 0, 1, 2) below the diagonal
// from (1, -1) to (-1, 1), triangle 1 (1, 2, 3) above it; vertices red, green, blue and white.
// Triangle i's provoking vertex is i + 2 under the last-vertex convention (blue below, white above)
// and i under the first-vertex one (red below, green above).
static const float kStrip[24] = {-1, -1, 1, 0, 0, 1, 1, -1, 0, 1, 0, 1, -1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
static int drawProvokingStrip(GLuint p, const char **skip) {
	vertices(p, kStrip, 4);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	return 1;
}

// A line along row 32, red at its left end and green at its right: its flat colour is the second
// vertex's (green) under the last-vertex convention, the first's (red) under the first-vertex one.
static const float kLine[12] = {-1, 0.015625f, 1, 0, 0, 1, 1, 0.015625f, 0, 1, 0, 1};
static int drawProvokingLine(GLuint p, const char **skip) {
	vertices(p, kLine, 2);
	glDrawArrays(GL_LINES, 0, 2);
	return 1;
}

static void setupFirstVertex(GLuint p) {
	glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
}

// The restart strips: two strips of one triangle each, A in the lower-left corner (vertices 0, 1,
// 2) and B in the upper-right (4, 5, 6), split by the restart index. Vertex 3, at (1, -1), is drawn
// only where the strip does not restart there: then triangle (2, 3, 4) covers the centre. White on
// the blue clear: the centre stays blue where the strip restarts, and pixels (4, 4) in A and (60,
// 60) in B are white.
static const float kRestart[7][2] = {{-1, -1}, {-0.5f, -1}, {-1, -0.5f}, {1, -1}, {1, 1}, {0.5f, 1}, {1, 0.5f}};

// The strip's vertices into a buffer of `count` (at least 7), white, with the restart strip's
// vertex 3 also at index `at3` where that is not 3.
static void restartVertices(GLuint p, int count, int at3) {
	float *v = calloc((size_t)count * 6, sizeof(float));
	for (int i = 0; i < count; i++) {
		const int k = i < 7 ? i : i == at3 ? 3 : 0;
		v[i * 6 + 0] = kRestart[k][0];
		v[i * 6 + 1] = kRestart[k][1];
		for (int c = 2; c < 6; c++) v[i * 6 + c] = 1;
	}
	vertices(p, v, count);
	free(v);
}

// The strip drawn over 16- or 32-bit indices 0, 1, 2, `split`, 4, 5, 6.
static void restartDraw(GLenum type, uint32_t split) {
	const uint32_t idx[7] = {0, 1, 2, split, 4, 5, 6};
	GLuint ibo = 0;
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
	if (type == GL_UNSIGNED_SHORT) {
		uint16_t idx16[7];
		for (int i = 0; i < 7; i++) idx16[i] = (uint16_t)idx[i];
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx16, idx16, GL_STATIC_DRAW);
	} else
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
	glDrawElements(GL_TRIANGLE_STRIP, 7, type, 0);
}

// Restart on at index 3, a vertex index the strip also uses: 16- and 32-bit.
static int drawRestartIndex16(GLuint p, const char **skip) {
	restartVertices(p, 7, 3);
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(3);
	restartDraw(GL_UNSIGNED_SHORT, 3);
	return 1;
}
static int drawRestartIndex32(GLuint p, const char **skip) {
	restartVertices(p, 7, 3);
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(3);
	restartDraw(GL_UNSIGNED_INT, 3);
	return 1;
}

// Restart on at the 16-bit all-ones index.
static int drawRestartAllOnes(GLuint p, const char **skip) {
	restartVertices(p, 7, 3);
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(0xffff);
	restartDraw(GL_UNSIGNED_SHORT, 0xffff);
	return 1;
}

// Restart off, a strip through vertex 0xffff (16-bit), which sits where vertex 3 does: GL draws it
// as a vertex, so the centre is covered.
static int drawRestartOff(GLuint p, const char **skip) {
	restartVertices(p, 0x10000, 0xffff);
	glDisable(GL_PRIMITIVE_RESTART);
	restartDraw(GL_UNSIGNED_SHORT, 0xffff);
	return 1;
}

// The strip drawn with restart on at 3, then again with restart off: the second draws vertex 3, so
// the centre is covered.
static int drawRestartOnThenOff(GLuint p, const char **skip) {
	restartVertices(p, 7, 3);
	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(3);
	restartDraw(GL_UNSIGNED_SHORT, 3);
	glDisable(GL_PRIMITIVE_RESTART);
	restartDraw(GL_UNSIGNED_SHORT, 3);
	return 1;
}

// GL_PRIMITIVE_RESTART_FIXED_INDEX (GL 4.3), where the context has it: the 32-bit strip restarts at
// 0xffffffff.
static int drawRestartFixed(GLuint p, const char **skip) {
	while (glGetError() != GL_NO_ERROR) {
	}
	glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX_CASES);
	if (glGetError() != GL_NO_ERROR) {
		*skip = "no GL_PRIMITIVE_RESTART_FIXED_INDEX";
		return 0;
	}
	restartVertices(p, 7, 3);
	restartDraw(GL_UNSIGNED_INT, 0xffffffffu);
	return 1;
}

// Legacy flat shading (glShadeModel GL_FLAT) of a fixed-function triangle, its vertices red, green
// and blue: the last vertex's colour, blue.
static int drawLegacyFlat(GLuint p, const char **skip) {
	glShadeModel(GL_FLAT_LEGACY);
	glBegin(GL_TRIANGLES);
	for (int i = 0; i < 3; i++) {
		glColor4f(kTriangle[i * 6 + 2], kTriangle[i * 6 + 3], kTriangle[i * 6 + 4], 1);
		glVertex2f(kTriangle[i * 6], kTriangle[i * 6 + 1]);
	}
	glEnd();
	return 1;
}

// --- the cases ------------------------------------------------------------------------------

#define VS41 "#version 410\nin vec2 p; out vec2 uv; void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.5, 1.0); }\n"
#define FS41(body) "#version 410\nin vec2 uv; out vec4 o;\n" body
// One point of 32 pixels centred at window (32, 24): the centre pixel's coordinate in it is
// (0.516, 0.234) from the upper-left origin and (0.516, 0.766) from the lower-left.
#define VSPOINT "#version 410\nin vec2 p; void main() { gl_PointSize = 32.0; gl_Position = vec4(0.0, -0.25, 0.5, 1.0); }\n"
#define FSPOINT "#version 410\nout vec4 o; void main() { o = vec4(gl_PointCoord, 0.0, 1.0); }\n"
// A flat colour from each vertex's.
#define VSFLAT "#version 410\nin vec2 p; in vec4 col; flat out vec4 c; void main() { c = col; gl_Position = vec4(p, 0.5, 1.0); }\n"
#define FSFLAT "#version 410\nflat in vec4 c; out vec4 o; void main() { o = c; }\n"
#define FSWHITE FS41("void main() { o = vec4(1.0); }")
#define RESTART_MORE {{4, 4, {255, 255, 255, 255}}, {60, 60, {255, 255, 255, 255}}}, 2

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
    {"logicop", VS41, FS41("void main() { o = vec4(1.0, 1.0, 0.0, 0.0); }"), setupLogicOp, 0, {191, 127, 191, 255}, 0},
    {"msaa-samplemask", VS41, FS41("void main() { o = vec4(1.0); }"), setupSampleMask, 0, {160, 192, 223, 255}, 2},
    {"msaa-samplecoverage", VS41, FS41("void main() { o = vec4(1.0); }"), setupSampleCoverage, 0, {160, 192, 223, 255},
     2},
    {"point-size", VSPOINT, FSPOINT, setupPointSize, 0, {131, 60, 0, 255}, 2},
    {"point-lowerleft", VSPOINT, FSPOINT, setupPointLowerLeft, 0, {131, 195, 0, 255}, 2},
    {"legacy-color", "#version 120\nattribute vec2 p; varying vec4 c; void main() { c = vec4(0.25, 0.5, 0.75, 1.0); gl_Position = vec4(p, 0.5, 1.0); }\n",
     "#version 120\nvarying vec4 c; void main() { gl_FragColor = c; }\n", NULL, 0, {64, 128, 191, 255}, 1},
    {"legacy-texture", "#version 120\nattribute vec2 p; varying vec2 uv; void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.5, 1.0); }\n",
     "#version 120\nuniform sampler2D t; varying vec2 uv; void main() { gl_FragColor = texture2D(t, vec2(0.75, 0.75)); }\n",
     setupTexture, 0, {255, 255, 255, 255}, 0},
    {"legacy-alphatest", "#version 120\nattribute vec2 p; void main() { gl_Position = vec4(p, 0.5, 1.0); }\n",
     "#version 120\nvoid main() { gl_FragColor = vec4(1.0, 1.0, 1.0, 0.25); }\n", setupAlphaTest, 0, {64, 128, 191, 255}, 0},
    {"legacy-twoside",
     "#version 120\nattribute vec2 p; void main() { gl_FrontColor = vec4(0.0, 1.0, 0.0, 1.0); gl_BackColor = vec4(1.0, "
     "0.0, 0.0, 1.0); gl_Position = vec4(p, 0.5, 1.0); }\n",
     "#version 120\nvoid main() { gl_FragColor = gl_Color; }\n", setupTwoSide, 0, {255, 0, 0, 255}, 0},
    {"legacy-fixed-quads", NULL, NULL, setupLegacyTexture, GL_QUADS, {255, 255, 255, 255}, 0},
    {"legacy-fixed-polygon", NULL, NULL, NULL, GL_POLYGON, {255, 128, 0, 255}, 1},
    {"provoking-last", VSFLAT, FSFLAT, NULL, 0, {0, 0, 255, 255}, 0, drawProvokingTriangle},
    {"provoking-first", VSFLAT, FSFLAT, setupFirstVertex, 0, {255, 0, 0, 255}, 0, drawProvokingTriangle},
    {"provoking-strip-last", VSFLAT, FSFLAT, NULL, 0, {255, 255, 255, 255}, 0, drawProvokingStrip, 40, 40,
     {{20, 20, {0, 0, 255, 255}}}, 1},
    {"provoking-strip-first", VSFLAT, FSFLAT, setupFirstVertex, 0, {0, 255, 0, 255}, 0, drawProvokingStrip, 40, 40,
     {{20, 20, {255, 0, 0, 255}}}, 1},
    {"provoking-line-last", VSFLAT, FSFLAT, NULL, 0, {0, 255, 0, 255}, 0, drawProvokingLine},
    {"provoking-line-first", VSFLAT, FSFLAT, setupFirstVertex, 0, {255, 0, 0, 255}, 0, drawProvokingLine},
    {"restart-index16", VS41, FSWHITE, NULL, 0, {64, 128, 191, 255}, 0, drawRestartIndex16, 0, 0, RESTART_MORE},
    {"restart-index32", VS41, FSWHITE, NULL, 0, {64, 128, 191, 255}, 0, drawRestartIndex32, 0, 0, RESTART_MORE},
    {"restart-allones", VS41, FSWHITE, NULL, 0, {64, 128, 191, 255}, 0, drawRestartAllOnes, 0, 0, RESTART_MORE},
    {"restart-off", VS41, FSWHITE, NULL, 0, {255, 255, 255, 255}, 0, drawRestartOff, 0, 0, RESTART_MORE},
    {"restart-on-then-off", VS41, FSWHITE, NULL, 0, {255, 255, 255, 255}, 0, drawRestartOnThenOff, 0, 0, RESTART_MORE},
    {"restart-fixed", VS41, FSWHITE, NULL, 0, {64, 128, 191, 255}, 0, drawRestartFixed, 0, 0, RESTART_MORE},
    {"legacy-flat", NULL, NULL, NULL, 0, {0, 0, 255, 255}, 0, drawLegacyFlat},
};

// --- running one ----------------------------------------------------------------------------

// A context of the legacy or the 4.1 core profile, on renderer `rendererID` where it is not 0.
static CGLContextObj contextOn(BOOL legacy, GLint rendererID) {
	CGLPixelFormatAttribute attrs[12];
	int a = 0;
	attrs[a++] = kCGLPFAAccelerated;
	attrs[a++] = kCGLPFAOpenGLProfile;
	attrs[a++] = (CGLPixelFormatAttribute)(legacy ? kCGLOGLPVersion_Legacy : kCGLOGLPVersion_GL4_Core);
	attrs[a++] = kCGLPFAColorSize;
	attrs[a++] = (CGLPixelFormatAttribute)24;
	if (rendererID) {
		attrs[a++] = kCGLPFAAllowOfflineRenderers;
		attrs[a++] = kCGLPFARendererID;
		attrs[a++] = (CGLPixelFormatAttribute)rendererID;
	}
	attrs[a] = (CGLPixelFormatAttribute)0;
	CGLPixelFormatObj pf = NULL;
	GLint n = 0;
	if (CGLChoosePixelFormat(attrs, &pf, &n) != kCGLNoError || !pf) return NULL;
	CGLContextObj ctx = NULL;
	CGLCreateContext(pf, NULL, &ctx);
	CGLDestroyPixelFormat(pf);
	return ctx;
}

// The renderer GL_CASES_RENDERER names (the first whose GL_RENDERER string holds it, among the
// virtual screens of a pixel format that allows offline renderers), or 0 for the default; -1
// where none does.
static GLint gRendererID = 0;
static GLint pickRenderer(void) {
	const char *want = getenv("GL_CASES_RENDERER");
	if (!want || !want[0]) return 0;
	CGLPixelFormatAttribute attrs[] = {kCGLPFAAccelerated, kCGLPFAAllowOfflineRenderers, kCGLPFAOpenGLProfile,
	                                   (CGLPixelFormatAttribute)kCGLOGLPVersion_GL4_Core, (CGLPixelFormatAttribute)0};
	CGLPixelFormatObj pf = NULL;
	GLint screens = 0;
	if (CGLChoosePixelFormat(attrs, &pf, &screens) != kCGLNoError || !pf) return -1;
	GLint found = -1;
	for (GLint i = 0; i < screens && found < 0; i++) {
		GLint id = 0;
		CGLDescribePixelFormat(pf, i, kCGLPFARendererID, &id);
		CGLContextObj ctx = contextOn(NO, id);
		if (!ctx) continue;
		CGLSetCurrentContext(ctx);
		const char *name = (const char *)glGetString(GL_RENDERER);
		printf("# renderer 0x%x: %s\n", id, name ? name : "?");
		if (name && strstr(name, want)) found = id;
		CGLSetCurrentContext(NULL);
		CGLDestroyContext(ctx);
	}
	CGLDestroyPixelFormat(pf);
	return found;
}

static CGLContextObj makeContext(BOOL legacy) {
	return contextOn(legacy, gRendererID);
}

// 1 where the case passes, 0 where it fails, -1 where it is skipped.
static int runCase(const Case *c) {
	const BOOL legacy = strncmp(c->name, "legacy-", 7) == 0;
	const GLsizei samples = strncmp(c->name, "msaa-", 5) == 0 ? 4 : 0;
	CGLContextObj ctx = makeContext(legacy);
	if (!ctx) {
		printf("%s: FAIL (no context)\n", c->name);
		return NO;
	}
	CGLSetCurrentContext(ctx);
	if (gVerbose) printf("  %s on %s, %s\n", c->name, (const char *)glGetString(GL_RENDERER), (const char *)glGetString(GL_VERSION));
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
	// An msaa-* case draws into a framebuffer of 4 samples, resolved into the one above at the end.
	GLuint msfb = 0, mscolour = 0, msds = 0;
	if (samples) {
		glGenFramebuffers(1, &msfb);
		glBindFramebuffer(GL_FRAMEBUFFER, msfb);
		glGenRenderbuffers(1, &mscolour);
		glBindRenderbuffer(GL_RENDERBUFFER, mscolour);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, 64, 64);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mscolour);
		glGenRenderbuffers(1, &msds);
		glBindRenderbuffer(GL_RENDERBUFFER, msds);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH24_STENCIL8, 64, 64);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, msds);
	}
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
	if (c->draw) {
		const char *skip = "";
		if (!c->draw(program, &skip)) {
			printf("%s: skip (%s)\n", c->name, skip);
			CGLSetCurrentContext(NULL);
			CGLDestroyContext(ctx);
			return -1;
		}
	} else if (c->vs) {
		if (!legacy) {
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);
		}
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glDrawArrays(strncmp(c->name, "point-", 6) == 0 ? GL_POINTS : GL_TRIANGLE_STRIP, 0, 4);
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
	if (samples) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, msfb);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb);
		glBlitFramebuffer(0, 0, 64, 64, 0, 0, 64, 64, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_FRAMEBUFFER, fb);
	}
	glFinish();
	uint8_t got[4] = {0};
	// The centre for every case, but discard's and clip distance's, whose quads cover only the left
	// part: a pixel right of them.
	const GLint x = c->x ? c->x : strcmp(c->name, "discard") == 0 || strcmp(c->name, "clipdistance") == 0 ? 48 : 32;
	const GLint y = c->y ? c->y : 32;
	glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, got);
	GLenum e = glGetError();
	BOOL pass = e == GL_NO_ERROR;
	for (int i = 0; i < 4; i++) pass = pass && abs((int)got[i] - (int)c->want[i]) <= c->tolerance;
	char detail[512] = "";
	if (!pass)
		snprintf(detail, sizeof detail, " (%d,%d) got %u,%u,%u,%u, want %u,%u,%u,%u, glGetError 0x%x;", x, y, got[0], got[1],
		         got[2], got[3], c->want[0], c->want[1], c->want[2], c->want[3], e);
	for (int m = 0; m < c->moreCount; m++) {
		uint8_t more[4] = {0};
		glReadPixels(c->more[m].x, c->more[m].y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, more);
		BOOL ok = YES;
		for (int i = 0; i < 4; i++) ok = ok && abs((int)more[i] - (int)c->more[m].want[i]) <= c->tolerance;
		if (!ok) {
			const size_t used = strlen(detail);
			snprintf(detail + used, sizeof detail - used, " (%d,%d) got %u,%u,%u,%u, want %u,%u,%u,%u;", c->more[m].x,
			         c->more[m].y, more[0], more[1], more[2], more[3], c->more[m].want[0], c->more[m].want[1],
			         c->more[m].want[2], c->more[m].want[3]);
		}
		pass = pass && ok;
	}
	if (pass)
		printf("%s: pass\n", c->name);
	else
		printf("%s: FAIL (%s)\n", c->name, detail + 1);
	CGLSetCurrentContext(NULL);
	CGLDestroyContext(ctx);
	return pass ? 1 : 0;
}

int main(int argc, char **argv) {
	@autoreleasepool {
		setvbuf(stdout, NULL, _IOLBF, 0);
		int first = 1;
		if (argc > 1 && strcmp(argv[1], "-v") == 0) {
			gVerbose = YES;
			first = 2;
		}
		gRendererID = pickRenderer();
		if (gRendererID < 0) {
			printf("gl_cases: no renderer's name holds %s\n", getenv("GL_CASES_RENDERER"));
			return 1;
		}
		int passed = 0, ran = 0, skipped = 0;
		for (size_t i = 0; i < sizeof kCases / sizeof kCases[0]; i++) {
			BOOL wanted = argc <= first;
			for (int a = first; a < argc; a++) wanted = wanted || strcmp(argv[a], kCases[i].name) == 0;
			if (!wanted) continue;
			const int r = runCase(&kCases[i]);
			if (r < 0) {
				skipped++;
				continue;
			}
			ran++;
			passed += r;
		}
		printf("gl_cases: %d of %d pass, %d skipped\n", passed, ran, skipped);
		return passed == ran ? 0 : 1;
	}
}
