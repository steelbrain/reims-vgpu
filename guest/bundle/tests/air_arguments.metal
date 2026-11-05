// The shaders of air_arguments_test.cpp: entry points of every stage the reader answers for,
// with binding arguments of each kind and access, arrays of them, argument buffers with and
// without a written texture, a function constant, arguments declared under function constants,
// binding indices function constants set, a vertex function's and a kernel's stage input, a
// post-tessellation vertex function's patch input, a function table and an acceleration
// structure, threadgroup memory, and a function of no texture; stitchable functions of every kind
// of type in their signatures; and function constants of several types. Built into a .metallib by
// Apple's Metal compiler at test time; never committed.
#include <metal_raytracing>
#include <metal_stdlib>
using namespace metal;
using namespace raytracing;

struct WritesOne {
	texture2d<float, access::write> out [[id(0)]];
	device float *values [[id(1)]];
};

struct ReadsOnly {
	texture2d<float> in [[id(0)]];
	constant float *values [[id(1)]];
};

constant bool kScale [[function_constant(0)]];

kernel void every_access(texture2d<float, access::read> r [[texture(0)]],
                         texture2d<float> s [[texture(1)]],
                         texture2d<float, access::read_write> rw [[texture(2)]],
                         texture2d<float, access::write> w [[texture(3)]],
                         array<texture2d<float>, 4> sampled [[texture(4)]],
                         array<texture2d<float, access::write>, 2> written [[texture(9)]],
                         device float *buf [[buffer(0)]],
                         constant float &k [[buffer(1)]],
                         device WritesOne &writesOne [[buffer(2)]],
                         constant ReadsOnly &readsOnly [[buffer(3)]],
                         sampler smp [[sampler(0)]],
                         uint2 g [[thread_position_in_grid]])
{
	float4 v = r.read(g) + s.sample(smp, float2(0)) + rw.read(g) + sampled[1].sample(smp, float2(0)) +
	           readsOnly.in.sample(smp, float2(0)) + readsOnly.values[0];
	rw.write(v, g);
	w.write(v, g);
	written[0].write(v, g);
	writesOne.out.write(v, g);
	writesOne.values[0] = v.x;
	buf[0] = kScale ? v.x * k : v.x;
}

kernel void no_texture(device float *b [[buffer(3)]], uint g [[thread_position_in_grid]])
{
	b[g] = 1;
}

struct VertexOut {
	float4 p [[position]];
};

vertex VertexOut reads_in_vertex(uint vid [[vertex_id]], texture2d<float> t [[texture(7)]])
{
	VertexOut o;
	o.p = t.read(uint2(vid, 0));
	return o;
}

fragment float4 writes_in_fragment(VertexOut in [[stage_in]],
                                   texture2d<float, access::write> w [[texture(5)]],
                                   texture2d<float> s [[texture(0)]], sampler smp [[sampler(0)]])
{
	float4 c = s.sample(smp, float2(0));
	w.write(c, uint2(in.p.xy));
	return c;
}

constant bool kHasLuma [[function_constant(1)]];
constant bool kHasChroma [[function_constant(2)]];
constant bool kHasPacked [[function_constant(3)]];

// A conversion kernel whose planes and output are declared under function constants, among
// four read buffers: each is its argument whatever the constants' values.
kernel void conditional_planes(device const uint *rows [[buffer(0)]],
                               constant uint &width [[buffer(1)]],
                               constant float4 &matrix [[buffer(2)]],
                               device const float *offsets [[buffer(3)]],
                               texture2d<half, access::read> luma [[texture(0), function_constant(kHasLuma)]],
                               texture2d<half, access::read> chroma
                                   [[texture(1), function_constant(kHasChroma)]],
                               texture2d<uint, access::write> packed
                                   [[texture(2), function_constant(kHasPacked)]],
                               device float *spill [[buffer(4), function_constant(kHasPacked)]],
                               uint2 g [[thread_position_in_grid]])
{
	float v = float(rows[g.y] + width) * matrix.x + offsets[0];
	if (kHasLuma)
		v += float(luma.read(g).x);
	if (kHasChroma)
		v += float(chroma.read(g / 2).x);
	if (kHasPacked) {
		packed.write(uint4(uint(v)), g);
		spill[g.x] = v;
	}
}

constant int kPlane [[function_constant(4)]];
constant int kOut [[function_constant(5)]];

// A plane read and an output written at binding indices function constants set, the plane's
// under a constant too: either may be at any index.
kernel void planes_where_constants_say(device const uint *rows [[buffer(0)]],
                                       texture2d<half, access::read> luma
                                           [[texture(kPlane), function_constant(kHasLuma)]],
                                       texture2d<half, access::write> out [[texture(kOut + 1)]],
                                       uint2 g [[thread_position_in_grid]])
{
	half v = half(rows[g.y]);
	if (kHasLuma)
		v += luma.read(g).x;
	out.write(half4(v), g);
}

// A plane read at an index a function constant sets, its output written at 3.
kernel void reads_where_constants_say(texture2d<half, access::read> luma [[texture(kPlane)]],
                                      texture2d<half, access::write> out [[texture(3)]],
                                      uint2 g [[thread_position_in_grid]])
{
	out.write(luma.read(g), g);
}

// Threadgroup memory at 0 beside a device buffer at 0 the kernel only reads: no buffer is bound
// at the memory's index.
kernel void shares_threadgroup_memory(threadgroup float *shared [[threadgroup(0)]],
                                      device const float *in [[buffer(0)]],
                                      device float *out [[buffer(1)]],
                                      uint t [[thread_position_in_threadgroup]])
{
	shared[t] = in[t];
	threadgroup_barrier(mem_flags::mem_threadgroup);
	out[t] = shared[t ^ 1];
}

[[visible]] float doubled(float x)
{
	return x * 2;
}

struct Attributes {
	float4 position [[attribute(0)]];
};

// Bindings of no texture besides buffers and samplers: a stage input's attribute, a function
// table and an acceleration structure.
vertex VertexOut reads_attributes(Attributes in [[stage_in]], constant float4 &scale [[buffer(1)]])
{
	VertexOut o;
	o.p = in.position * scale;
	return o;
}

// A post-tessellation vertex function's patch: its control points and its patch input, attributes
// fetched from buffers, bindings of no texture.
struct PatchPoint {
	float4 p [[attribute(0)]];
};

struct PatchIn {
	patch_control_point<PatchPoint> points;
	float4 color [[attribute(1)]];
};

[[patch(quad, 4)]] vertex float4 reads_patch(PatchIn in [[stage_in]],
                                            float2 uv [[position_in_patch]],
                                            texture2d<float> t [[texture(3)]])
{
	return in.points[0].p * uv.x + in.color + t.read(uint2(0));
}

// A kernel's stage input, its attributes fetched from buffers: a binding of no texture.
kernel void reads_stage_in(Attributes in [[stage_in]], device float4 *out [[buffer(1)]],
                           texture2d<float, access::write> w
                               [[texture(0), function_constant(kHasPacked)]],
                           uint g [[thread_position_in_grid]])
{
	out[g] = in.position;
	if (kHasPacked)
		w.write(in.position, uint2(g, 0));
}

kernel void calls_through_table(device float *out [[buffer(0)]],
                                visible_function_table<float(float)> table [[buffer(1)]],
                                instance_acceleration_structure scene [[buffer(2)]],
                                texture2d<float, access::write> w [[texture(3)]],
                                uint g [[thread_position_in_grid]])
{
	(void)scene;
	out[g] = table[0](out[g]);
	w.write(float4(out[g]), uint2(g, 0));
}

// Stitchable functions' signatures (rgfxReadAirVisible), of every kind of type a stitchable
// function can take: scalars (bfloat and every integer width among them), vectors (packed ones
// among them) and matrices; structs, nested; arrays of C's and metal::arrays, of scalars, vectors,
// matrices, structs, arrays, pointers, textures, samplers, enums and function tables; pointers
// and references in each address space, to each of those, to pointers and to void, as arguments
// and as fields; atomics; enums; textures of each shape and access, depth ones among them;
// samplers; function tables; an acceleration structure; an unnamed argument; a void return and
// none at all. `doubled` above is visible and not stitchable: it has no signature.
struct SigPair {
	float2 xy;
	float w;
	int n;
};

struct SigOuter {
	SigPair p;
	half4 h;
	uint u;
};

struct SigFields {
	float a[2];
	SigPair pairs[2];
	float grid[2][3];
	float4 v[2];
	float2x2 m[2];
	half hs[3];
	device float *f;
	constant SigPair *sp;
	constant float4x4 *transform;
	device float2 (*rows)[4];
	texture2d<float> t;
	sampler s;
	array<float, 3> af;
	texture2d<float> ts[2];
	sampler ss[2];
	device float *ps[2];
	float2 origin;
};

[[stitchable]] SigPair sig_struct(SigOuter o, int i, uint2 u, half h, bool z)
{
	SigPair r;
	r.xy = o.p.xy + float2(u) + float(h) + float(z);
	r.w = float(o.u);
	r.n = i;
	return r;
}

[[stitchable]] void sig_void(float a) {}

[[stitchable]] float sig_none() { return 1; }

[[stitchable]] uchar4 sig_scalars(char c, short s, ushort2 us, float4x4 m, packed_float3 pf, long l,
                                  ulong3 ul, bool2 b, half3x2 hm, float2x3 fm, packed_half2 ph, uint u,
                                  int4 i4, uchar uc, char3 c3)
{
	return uchar4(uchar(c + s + us.x), uchar(m[0][0] + pf.x + float(l) + float(ul.x) + float(b.x)),
	              uchar(float(hm[0][0]) + fm[0][0] + float(ph.x)), uchar(u + uint(i4.x) + uc + uchar(c3.x)));
}

[[stitchable]] float sig_pointers(device float *p, constant int4 *q, float a[3], device SigPair *sp,
                                  constant float4x4 *m, device float2 (*rows)[4], thread half *t)
{
	return p[0] + float(q[0].x) + a[0] + sp->w + (*m)[0][0] + rows[0][1].x + float(t[0]);
}

[[stitchable]] float4 sig_fields(SigFields f, array<float, 3> af, sampler s)
{
	return f.t.sample(f.s, f.origin) + f.a[0] + af[1] + f.f[0] + f.sp->w + (*f.transform)[0][0] +
	       f.ts[1].sample(s, f.origin);
}

[[stitchable]] float4 sig_textures(texture2d<half, access::read> r, texture2d<float, access::write> w,
                                   texture2d<float, access::read_write> rw, depth2d<float> d,
                                   texture3d<float> t3, texturecube_array<float> ca,
                                   texture1d_array<float> a1, texture_buffer<float> tb,
                                   texture2d_ms_array<float> msa, depth2d_ms_array<float> dmsa,
                                   texture2d<int> ti, texture2d<ushort> tu, sampler s)
{
	return float4(r.read(uint2(0))) + d.sample(s, float2(0)) + t3.sample(s, float3(0));
}

[[stitchable]] texture2d<float> sig_returns_texture(texture2d<float> t) { return t; }

// References, read as pointers: Metal reflects the two alike. Arguments of each kind of referent,
// and a struct's fields, as Core Image's CIKL kernels take a sampler in a struct with a reference
// field.
struct SigReferences {
	texture2d<float> t;
	sampler s;
	constant float4x4 &transform;
	float2 origin;
	device SigPair &pair;
	constant float3 (&rows)[2];
};

[[stitchable]] float4 sig_references(SigReferences r, thread const float4 &v, device float &d,
                                     constant SigPair &cp, thread float2x2 &m)
{
	return r.t.sample(r.s, r.origin) + r.transform[0] + r.pair.w + r.rows[1].x + v + d + cp.w + m[0][0];
}

// Pointers and references in each address space, to each kind of pointee: pointers to pointers,
// to void, to arrays and to scalars, packed vectors and matrices; an rvalue reference, and an
// lvalue reference to a pointer. Metal sizes a pointer to a pointer by the innermost pointee.
[[stitchable]] float sig_address_spaces(threadgroup float *t, threadgroup float &r, thread float &&rv,
                                        device float *device *pp, thread float **tp, device void *v,
                                        constant void *cv, threadgroup SigPair *tsp,
                                        device SigPair *device *psp, device float (*device *ppa)[3],
                                        thread device float *&rp, device bfloat *pbf,
                                        device packed_float3 *ppf, device bool *pb, device long *pl,
                                        device float3x3 *pm, device SigPair (&rsa)[2])
{
	return t[0] + r + rv + pp[0][0] + tp[0][0] + tsp->w + psp[0]->w + (*ppa[0])[1] + rp[0] + float(pbf[0]) +
	       ppf->x + float(pb[0]) + float(pl[0]) + (*pm)[0][0] + rsa[1].w;
}

// Atomics: Metal's library makes each the struct `metal::_atomic<T>` of one field; a float's is of
// no size and no field, as nothing in this file lays one out.
struct SigAtomics {
	atomic_int a;
	float b;
	atomic_uint c[2];
	device atomic_float *pf;
};

[[stitchable]] float sig_atomics(device atomic_int *a, device atomic_uint *b, device atomic_float *c,
                                 device SigAtomics *w, threadgroup atomic_int *ta)
{
	return w->b + float(atomic_load_explicit(a, memory_order_relaxed));
}

// Enums, of each underlying size, and bfloats.
enum class SigE : int { A, B };
enum class SigE8 : uchar { A };

struct SigEnums {
	SigE e;
	SigE8 f;
	SigE8 fs[2];
	bfloat bf;
	bfloat3 bf3;
	bfloat2 bfs[2];
	device SigE *pe;
	device bfloat *pbf;
};

[[stitchable]] SigE8 sig_enums(SigE e, SigE8 f, SigEnums w, device SigE *pe, bfloat bf, bfloat2 b2,
                               bfloat3 b3, bfloat4 b4)
{
	return e == SigE::A && float(bf) > 0 ? f : w.f;
}

[[stitchable]] bfloat3 sig_returns_bfloat(bfloat3 a) { return a; }

// Function tables, as arguments, fields, arrays and a pointee.
struct SigTables {
	visible_function_table<float(float)> t;
	intersection_function_table<triangle_data> it;
	visible_function_table<float(float)> ts[2];
	intersection_function_table<triangle_data> its[2];
	float y;
};

[[stitchable]] float sig_tables(visible_function_table<float(float)> t,
                                intersection_function_table<triangle_data> it, SigTables w,
                                array<visible_function_table<float(float)>, 2> ts,
                                device visible_function_table<float(float)> *pt, float x)
{
	return t[0](x) + w.y;
}

// Pointers to textures and samplers, and a struct's fields of each kind of pointee: Metal gives a
// pointed-to texture its pointer's address space as its access.
struct SigPointees {
	device texture2d<float> *pt;
	constant texture2d<float> *ptc;
	constant sampler *ps;
	texture2d<float, access::write> tw[2];
	device float *device *pp;
	device SigPair *device *psp;
	device void *device *ppv;
	device SigE *device *ppe;
	device float (*device *ppa)[3];
	device float4 *device *ppv4;
	device float2x2 *device *ppm;
	device texture2d<float> *device *ppt;
	device void *pv;
	device packed_half3 *pph3;
	device SigAtomics *pa;
	device float *device &rp;
	device void *pvs[2];
	device array<float2, 3> *pao;
	packed_char4 pc4;
	packed_int3 pi3;
};

[[stitchable]] float sig_pointees(device texture2d<float, access::read> *ptr,
                                  device texture2d<half, access::write> *ptw,
                                  constant texture2d<float> *ptc, threadgroup texture2d<float> *ptg,
                                  thread texture2d<float> &rt, device depth2d<float> *pd,
                                  device texture3d<int> *p3, constant sampler *ps,
                                  device array<float2, 3> *pao, SigPointees p)
{
	return p.rp[0];
}

// An unnamed argument: Metal names it "".
[[stitchable]] float sig_unnamed(float a, float4, int b) { return a + float(b); }

// An acceleration structure: Metal's own reflection of a signature that holds one ends the
// process, so the device answers none.
[[stitchable]] float sig_acceleration(instance_acceleration_structure s, float x) { return x; }

// Function constants of several types, vectors among them, and one the kernel tests with
// is_function_constant_defined, which is not required; a visible function that reads one.
constant float4 kTint [[function_constant(20)]];
constant ushort2 kSteps [[function_constant(21)]];
constant bool3 kMask [[function_constant(22)]];
constant half kGain [[function_constant(23)]];
constant int kOptional [[function_constant(24)]];
constant bool kHasOptional = is_function_constant_defined(kOptional);

kernel void every_constant(device float *out [[buffer(0)]], uint g [[thread_position_in_grid]])
{
	float v = kTint.x + float(kSteps.y) + (kMask.z ? 1.0f : 0.0f) + float(kGain);
	if (kHasOptional)
		v += float(kOptional);
	out[g] = v;
}

[[visible]] float tinted(float x) { return x * kTint.y; }
