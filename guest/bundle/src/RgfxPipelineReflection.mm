// RgfxPipelineReflection: see RgfxPipelineReflection.h. Our own ObjC++: Metal's reflection classes,
// their initialisers and their instance variables are named as the Objective-C runtime lists them
// (measured: macOS 26.7.1, AMD Radeon Pro 5300M and Intel UHD Graphics 630; tools/oracle
// oracle_reflection's class dump).
#import "RgfxPipelineReflection.h"

#import "RgfxMetalPrivate.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <cstring>
#include <set>
#include <string>

NS_ASSUME_NONNULL_BEGIN
// The initialisers and setters of Metal's reflection classes the answers are made through, beside
// RgfxMetalPrivate.h's (measured: the class dump, macOS 26.7.1 and 27).
@protocol RgfxPipelineReflectionCalls
- (nullable instancetype)initWithName:(NSString *)name
                                 type:(NSUInteger)type
                               access:(NSUInteger)access
                                index:(NSUInteger)index
                               active:(BOOL)active
                          arrayLength:(NSUInteger)arrayLength;
- (nullable instancetype)initWithName:(NSString *)name
                                 type:(NSUInteger)type
                               access:(NSUInteger)access
                             isActive:(BOOL)isActive
                        locationIndex:(NSUInteger)locationIndex
                            arraySize:(NSUInteger)arraySize
                             dataType:(NSUInteger)dataType
                          pixelFormat:(NSUInteger)pixelFormat
                              aluType:(NSUInteger)aluType
                     isConstantBuffer:(BOOL)isConstantBuffer
                             dataSize:(NSUInteger)dataSize
                            alignment:(NSUInteger)alignment;
- (nullable instancetype)initWithName:(NSString *)name
                                 type:(NSUInteger)type
                               access:(NSUInteger)access
                             isActive:(BOOL)isActive
                        locationIndex:(NSUInteger)locationIndex
                            arraySize:(NSUInteger)arraySize
                             dataType:(NSUInteger)dataType
                             dataSize:(NSUInteger)dataSize
                            alignment:(NSUInteger)alignment;
- (nullable instancetype)initWithName:(NSString *)name
                                 type:(NSUInteger)type
                               access:(NSUInteger)access
                               active:(BOOL)active
                                index:(NSUInteger)index
                             dataType:(NSUInteger)dataType
                          builtInType:(NSUInteger)builtInType;
- (nullable instancetype)initWithName:(NSString *)name
                                 type:(NSUInteger)type
                               access:(NSUInteger)access
                             isActive:(BOOL)isActive
                        locationIndex:(NSUInteger)locationIndex
                             dataType:(NSUInteger)dataType
                          pixelFormat:(NSUInteger)pixelFormat
                              aluType:(NSUInteger)aluType
                             dataSize:(NSUInteger)dataSize
                            alignment:(NSUInteger)alignment;
- (nullable instancetype)initWithName:(NSString *)name
                                 type:(NSUInteger)type
                               access:(NSUInteger)access
                               active:(BOOL)active
                                index:(NSUInteger)index
                             dataType:(NSUInteger)dataType
                          builtInType:(NSUInteger)builtInType
                 meshVertexStructType:(nullable id)vertexStruct
              meshPrimitiveStructType:(nullable id)primitiveStruct
                      meshVertexCount:(unsigned short)vertexCount
                   meshPrimitiveCount:(unsigned short)primitiveCount
                     meshTopologyType:(NSUInteger)topology;
- (nullable instancetype)initWithElementType:(NSUInteger)elementType
                      elementTypeDescription:(nullable id)elementTypeDescription
                                      access:(NSUInteger)access
                                   alignment:(NSUInteger)alignment
                                    dataSize:(NSUInteger)dataSize
             elementIsIndirectArgumentBuffer:(BOOL)elementIsIndirectArgumentBuffer
                            isConstantBuffer:(BOOL)isConstantBuffer
                                    doRetain:(BOOL)doRetain;
- (void)setIsIndirectArgumentBuffer:(BOOL)value;
- (void)setIndirectArgumentIndexStride:(NSUInteger)stride;
- (void)setVertexDescriptorBuffer:(BOOL)value;
- (void)setIsArgument:(BOOL)value;
- (void)setIsUsed:(BOOL)value;
@end
NS_ASSUME_NONNULL_END

namespace {

// What Metal's reflection answers where its public enums name no value (measured, both x86 Macs).
// A struct member's render target and raster order group where it has none, and a reflection's
// trace buffer index.
constexpr NSUInteger kNoIndex = 0xFFFFFFFF;
// The binding type of a built-in argument, of an argument buffer's constant member, and of a mesh.
constexpr NSUInteger kBuiltInBindingType = 15;
constexpr NSUInteger kIndirectConstantBindingType = 4;
constexpr NSUInteger kMeshBindingType = 35;
// The built-in type and data type of a mesh, and the data type of mesh_grid_properties.
constexpr NSUInteger kMeshBuiltInType = 24;
constexpr NSUInteger kMeshDataType = 157;
constexpr NSUInteger kMeshGridPropertiesDataType = 156;
// A constant buffer's alignment, and a descriptor buffer's.
constexpr NSUInteger kConstantBufferAlignment = 32;
constexpr NSUInteger kDescriptorBufferAlignment = 4;
// What a vertex function's post-vertex dump stride adds to its outputs' size (the AMD GPU's).
constexpr uint64_t kPostVertexDumpPadding = 12;
// The print style a vertex pipeline's reflection is of, and a mesh pipeline's.
constexpr NSUInteger kVertexPrintStyle = 3;
constexpr NSUInteger kMeshPrintStyle = 26;
// The most attributes a vertex or stage input descriptor has.
constexpr NSUInteger kMaxAttributes = 31;

// The built-in arguments whose answers were measured: the AIR's kind, and Metal's built-in type.
// A kernel's simdgroup_index_in_threadgroup is not listed by Metal (measured: k_more).
const struct {
	const char *kind;
	NSUInteger type; // 0: Metal lists it nowhere
} kBuiltIns[] = {
    {"air.vertex_id", 1},
    {"air.instance_id", 2},
    {"air.base_vertex", 3},
    {"air.base_instance", 4},
    {"air.thread_position_in_grid", 5},
    {"air.threads_per_grid", 6},
    {"air.thread_position_in_threadgroup", 7},
    {"air.thread_index_in_threadgroup", 8},
    {"air.threads_per_threadgroup", 9},
    {"air.threadgroup_position_in_grid", 10},
    {"air.threadgroups_per_grid", 11},
    {"air.thread_index_in_quadgroup", 17},
    {"air.thread_index_in_simdgroup", 18},
    {"air.threads_per_simdgroup", 20},
    {"air.mesh_grid_properties", 23},
    {"air.simdgroup_index_in_threadgroup", 0},
    {"air.simdgroups_per_threadgroup", 0},
    {"air.quadgroup_index_in_threadgroup", 0},
    {"air.quadgroups_per_threadgroup", 0},
    {"air.dispatch_threads_per_threadgroup", 0},
};

// A stage's nodes that are no binding and no built-in: a kernel's and a vertex function's stage
// input attributes.
bool isStageInput(const std::string &kind)
{
	return kind == "air.stage_in" || kind == "air.vertex_input";
}

// The classes the answers are made of, each looked up once with the initialiser it is made with.
struct Classes {
	Class type, structType, arrayType, pointerType, textureType, member, binding, bufferBinding, textureBinding,
	    threadgroupBinding, builtIn, indirectConstant, payload, meshBuiltIn, compute, render;
	NSString *missing; // the first class or initialiser that is not there, or nil
};

const Classes &classes()
{
	static Classes c;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		const struct {
			Class *cls;
			const char *name;
			SEL init;
		} kNeeded[] = {
		    {&c.type, "MTLTypeInternal", @selector(initWithDataType:)},
		    {&c.structType, "MTLStructTypeInternal", @selector(initWithMembers:count:typeName:)},
		    {&c.arrayType, "MTLArrayTypeInternal",
		     @selector(initWithArrayLength:elementType:stride:pixelFormat:aluType:details:)},
		    {&c.pointerType, "MTLPointerTypeInternal",
		     @selector(initWithElementType:elementTypeDescription:access:alignment:dataSize:
		                   elementIsIndirectArgumentBuffer:isConstantBuffer:doRetain:)},
		    {&c.textureType, "MTLTextureReferenceTypeInternal",
		     @selector(initWithDataType:textureType:access:isDepthTexture:)},
		    {&c.member, "MTLStructMemberInternal",
		     @selector(initWithName:offset:dataType:pixelFormat:aluType:indirectArgumentIndex:render_target:
		                   raster_order_group:details:)},
		    {&c.binding, "MTLBindingInternal", @selector(initWithName:type:access:index:active:arrayLength:)},
		    {&c.bufferBinding, "MTLBufferBindingInternal",
		     @selector(initWithName:type:access:isActive:locationIndex:arraySize:dataType:pixelFormat:aluType:
		                   isConstantBuffer:dataSize:alignment:)},
		    {&c.textureBinding, "MTLTextureBindingInternal",
		     @selector(initWithName:access:isActive:locationIndex:arraySize:dataType:textureType:isDepthTexture:)},
		    {&c.threadgroupBinding, "MTLThreadgroupMemoryBindingInternal",
		     @selector(initWithName:type:access:isActive:locationIndex:arraySize:dataType:dataSize:alignment:)},
		    {&c.builtIn, "MTLBuiltInArgument", @selector(initWithName:type:access:active:index:dataType:builtInType:)},
		    {&c.indirectConstant, "MTLIndirectConstantArgument",
		     @selector(initWithName:type:access:isActive:locationIndex:dataType:pixelFormat:aluType:dataSize:
		                   alignment:)},
		    {&c.payload, "MTLObjectPayloadBindingInternal",
		     @selector(initWithName:type:access:isActive:locationIndex:arraySize:dataType:dataSize:alignment:)},
		    {&c.meshBuiltIn, "MTLBuiltInMeshArgument",
		     @selector(initWithName:type:access:active:index:dataType:builtInType:meshVertexStructType:
		                   meshPrimitiveStructType:meshVertexCount:meshPrimitiveCount:meshTopologyType:)},
		    {&c.compute, "MTLComputePipelineReflectionInternal", @selector(init)},
		    {&c.render, "MTLRenderPipelineReflectionInternal", @selector(init)},
		};
		for (const auto &n : kNeeded) {
			*n.cls = NSClassFromString(@(n.name));
			if (*n.cls == Nil || ![*n.cls instancesRespondToSelector:n.init]) {
				c.missing = [NSString stringWithFormat:@"-[%s %s]", n.name, sel_getName(n.init)];
				return;
			}
		}
	});
	return c;
}

// The instance variable `name` of `o`'s class or a superclass, of an encoding of `type`'s first
// character; null where there is none.
Ivar ivarOf(id o, const char *name, char type)
{
	for (Class k = object_getClass(o); k != Nil; k = class_getSuperclass(k)) {
		Ivar v = class_getInstanceVariable(k, name);
		if (v != nullptr) {
			const char *e = ivar_getTypeEncoding(v);
			return e != nullptr && e[0] == type ? v : nullptr;
		}
	}
	return nullptr;
}


// What the AIR names the scalar of a type: its data type, its size, and whether Metal has matrices
// of it.
const struct {
	const char *name;
	NSUInteger type;
	uint32_t bytes;
	bool matrices;
} kScalars[] = {
    {"float", MTLDataTypeFloat, 4, true}, {"half", MTLDataTypeHalf, 2, true},     {"bfloat", MTLDataTypeBFloat, 2, false},
    {"int", MTLDataTypeInt, 4, false},    {"uint", MTLDataTypeUInt, 4, false},    {"short", MTLDataTypeShort, 2, false},
    {"ushort", MTLDataTypeUShort, 2, false}, {"char", MTLDataTypeChar, 1, false}, {"uchar", MTLDataTypeUChar, 1, false},
    {"bool", MTLDataTypeBool, 1, false},  {"long", MTLDataTypeLong, 8, false},    {"ulong", MTLDataTypeULong, 8, false},
};

// The scalar, vector or matrix the AIR names `name` (`float`, `half3`, `packed_float3`,
// `float4x4`): its data type and its natural alignment. False for another name.
bool valueType(const std::string &name, NSUInteger &type, uint32_t &align)
{
	const bool packed = name.compare(0, 7, "packed_") == 0;
	const std::string n = packed ? name.substr(7) : name;
	for (const auto &s : kScalars) {
		const size_t len = strlen(s.name);
		if (n.compare(0, len, s.name) != 0)
			continue;
		const std::string rest = n.substr(len);
		if (rest.empty() && !packed) {
			type = s.type;
			align = s.bytes;
			return true;
		}
		if (rest.size() == 1 && rest[0] >= '2' && rest[0] <= '4') {
			const uint32_t lanes = static_cast<uint32_t>(rest[0] - '0');
			type = s.type + lanes - 1;
			align = packed ? s.bytes : s.bytes * (lanes == 3 ? 4 : lanes);
			return true;
		}
		if (s.matrices && !packed && rest.size() == 3 && rest[1] == 'x' && rest[0] >= '2' && rest[0] <= '4' &&
		    rest[2] >= '2' && rest[2] <= '4') {
			const uint32_t rows = static_cast<uint32_t>(rest[2] - '0');
			type = (s.type == MTLDataTypeFloat ? MTLDataTypeFloat2x2 : MTLDataTypeHalf2x2) +
			       static_cast<NSUInteger>(rest[0] - '2') * 3 + (rows - 2);
			align = s.bytes * (rows == 3 ? 4 : rows);
			return true;
		}
		return false;
	}
	return false;
}

// A texture the AIR names `name` (`texture2d<float, read>`, `depthcube_array<float, sample>`, and
// an array of them, `array<texture2d<float, sample>, 3>`): its texture type, sample data type and
// whether it is a depth texture. False for another name.
bool textureType(const std::string &name, NSUInteger &type, NSUInteger &dataType, bool &depth)
{
	std::string n = name;
	if (n.compare(0, 6, "array<") == 0)
		n = n.substr(6);
	const size_t open = n.find('<'), comma = n.find(',', open);
	if (open == std::string::npos || comma == std::string::npos)
		return false;
	std::string shape = n.substr(0, open);
	depth = shape.compare(0, 5, "depth") == 0;
	if (depth)
		shape = "texture" + shape.substr(5);
	static const struct {
		const char *shape;
		MTLTextureType type;
	} kShapes[] = {
	    {"texture1d", MTLTextureType1D},
	    {"texture1d_array", MTLTextureType1DArray},
	    {"texture2d", MTLTextureType2D},
	    {"texture2d_array", MTLTextureType2DArray},
	    {"texture2d_ms", MTLTextureType2DMultisample},
	    {"texture2d_ms_array", MTLTextureType2DMultisampleArray},
	    {"texturecube", MTLTextureTypeCube},
	    {"texturecube_array", MTLTextureTypeCubeArray},
	    {"texture3d", MTLTextureType3D},
	    {"texture_buffer", MTLTextureTypeTextureBuffer},
	};
	bool found = false;
	for (const auto &s : kShapes) {
		if (shape == s.shape) {
			type = s.type;
			found = true;
		}
	}
	uint32_t align = 0;
	return found && valueType(n.substr(open + 1, comma - open - 1), dataType, align) &&
	       (dataType == MTLDataTypeFloat || dataType == MTLDataTypeHalf || dataType == MTLDataTypeInt ||
	        dataType == MTLDataTypeUInt || dataType == MTLDataTypeShort || dataType == MTLDataTypeUShort);
}

// A binding's access, as Metal numbers it: read and sample 0, read_write 1, write 2.
NSUInteger accessOf(RgfxAirAccess a)
{
	switch (a) {
	case RgfxAirAccess::ReadWrite: return MTLBindingAccessReadWrite;
	case RgfxAirAccess::Write: return MTLBindingAccessWriteOnly;
	default: return MTLBindingAccessReadOnly;
	}
}

// Makes the objects; the first thing it cannot make sets `why`.
struct Maker {
	const Classes &c;
	NSString *why = nil;

	bool fail(NSString *w)
	{
		if (why == nil)
			why = w;
		return false;
	}

	static id<RgfxMetalReflectionInitialisers, RgfxPipelineReflectionCalls> make(Class cls) { return [cls alloc]; }

	// Sets `o`'s object instance variable `name` (retained, as Metal's classes keep theirs).
	bool setObject(id o, const char *name, id value)
	{
		Ivar v = ivarOf(o, name, '@');
		if (v == nullptr)
			return fail([NSString stringWithFormat:@"Metal's %s has no %s", class_getName(object_getClass(o)), name]);
		object_setIvarWithStrongDefault(o, v, value);
		return true;
	}

	// Sets `o`'s unsigned instance variable `name`, of the encoding `type` ('Q', 'I', 'S' or 'c').
	bool setNumber(id o, const char *name, char type, uint64_t value)
	{
		Ivar v = ivarOf(o, name, type);
		if (v == nullptr)
			return fail([NSString stringWithFormat:@"Metal's %s has no %s of %c", class_getName(object_getClass(o)),
			                                       name, type]);
		uint8_t *at = reinterpret_cast<uint8_t *>((__bridge void *)o) + ivar_getOffset(v);
		switch (type) {
		case 'Q': *reinterpret_cast<uint64_t *>(at) = value; break;
		case 'I': *reinterpret_cast<uint32_t *>(at) = static_cast<uint32_t>(value); break;
		case 'S': *reinterpret_cast<uint16_t *>(at) = static_cast<uint16_t>(value); break;
		default: *at = static_cast<uint8_t>(value); break;
		}
		return true;
	}

	// A texture's type: `dataType` is its sample type (MTLDataTypeFloat for texture2d<float>); the
	// type's own data type is MTLDataTypeTexture.
	id texture(NSUInteger dataType, NSUInteger textureType, NSUInteger access, bool depth)
	{
		return [make(c.textureType) initWithDataType:dataType textureType:textureType access:access isDepthTexture:depth];
	}

	id pointer(NSUInteger elementType, id elementInfo, NSUInteger access, NSUInteger alignment, NSUInteger size,
	           bool argumentBuffer, bool constant)
	{
		return [make(c.pointerType) initWithElementType:elementType
		                         elementTypeDescription:elementInfo
		                                         access:access
		                                      alignment:alignment
		                                       dataSize:size
		                elementIsIndirectArgumentBuffer:argumentBuffer
		                               isConstantBuffer:constant
		                                       doRetain:YES];
	}

	id array(NSUInteger length, NSUInteger elementType, NSUInteger stride, id elementInfo)
	{
		id a = [make(c.arrayType) initWithArrayLength:length
		                                  elementType:elementType
		                                       stride:stride
		                                  pixelFormat:MTLPixelFormatInvalid
		                                      aluType:0
		                                      details:nil];
		if (a != nil && elementInfo != nil)
			setObject(a, "_elementTypeInfo", elementInfo);
		return a;
	}

	// A struct type of `members` (MTLStructMemberInternal), nameless as Metal's are.
	id structOf(NSArray *members, bool argumentBuffer)
	{
		std::vector<__unsafe_unretained id> list(members.count);
		for (NSUInteger i = 0; i < members.count; i++)
			list[i] = members[i];
		id s = [make(c.structType) initWithMembers:list.data() count:list.size() typeName:nil];
		if (argumentBuffer)
			[s setIsIndirectArgumentBuffer:YES];
		return s;
	}

	id member(const std::string &name, NSUInteger offset, NSUInteger dataType, NSUInteger argumentIndex,
	          NSUInteger renderTarget, NSUInteger rasterOrderGroup, id typeInfo, id details)
	{
		id m = [make(c.member) initWithName:@(name.c_str())
		                             offset:offset
		                           dataType:dataType
		                        pixelFormat:MTLPixelFormatInvalid
		                            aluType:0
		              indirectArgumentIndex:argumentIndex
		                      render_target:renderTarget
		                 raster_order_group:rasterOrderGroup
		                            details:nil];
		if (m == nil || !setObject(m, "_typeInfo", typeInfo) || !setObject(m, "_details", details))
			return nil;
		return m;
	}

	// The struct type of a layout: a struct's, or an argument buffer's (`argumentBuffer`).
	id structType(const std::vector<RgfxAirMember> &layout, bool argumentBuffer)
	{
		NSMutableArray *members = [NSMutableArray arrayWithCapacity:layout.size()];
		for (const RgfxAirMember &m : layout) {
			id one = argumentBuffer ? argumentMember(m) : plainMember(m);
			if (one == nil)
				return nil;
			[members addObject:one];
		}
		return structOf(members, argumentBuffer);
	}

	// A struct's member: a scalar, vector or matrix, a struct, or an array of either.
	id plainMember(const RgfxAirMember &m)
	{
		const NSUInteger rt = m.hasRenderTarget ? m.renderTarget : kNoIndex;
		const NSUInteger rog = m.hasRasterOrderGroup ? m.rasterOrderGroup : kNoIndex;
		if (m.indirect != nullptr || m.hasFirstId) {
			fail(@"a struct holds an argument buffer's member");
			return nil;
		}
		NSUInteger type = MTLDataTypeNone;
		uint32_t align = 0;
		id info = nil;
		if (m.layout != nullptr) {
			type = MTLDataTypeStruct;
			info = structType(*m.layout, false);
			if (info == nil)
				return nil;
		} else if (!valueType(m.typeName, type, align)) {
			fail([NSString stringWithFormat:@"a struct member's type %s was not measured", m.typeName.c_str()]);
			return nil;
		}
		if (m.arrayLength == 0)
			return member(m.name, m.offset, type, 0, rt, rog, info, nil);
		// An array is of its element's data type, its stride the element's size (measured: Inner,
		// float, float4, float4x4 and uint2).
		id a = array(m.arrayLength, type, m.size, info);
		return a == nil ? nil : member(m.name, m.offset, MTLDataTypeArray, 0, rt, rog, a, nil);
	}

	// An argument buffer's member: a texture (or an array of them), a buffer, a sampler or a
	// constant, its binding its details.
	id argumentMember(const RgfxAirMember &m)
	{
		if (m.indirect == nullptr || m.hasFirstId || m.arrayLength != 0) {
			fail(@"an argument buffer's member of a kind not measured");
			return nil;
		}
		const RgfxAirNode &n = *m.indirect;
		const NSUInteger rt = m.hasRenderTarget ? m.renderTarget : kNoIndex;
		const NSUInteger rog = m.hasRasterOrderGroup ? m.rasterOrderGroup : kNoIndex;
		id info = nil;
		id binding = resource(n, RgfxAirEntryStage::Kernel, &info, &m);
		if (binding == nil)
			return nil;
		NSUInteger type = MTLDataTypeNone;
		if (n.kind == "air.texture") {
			// The member's texture type is an object of its own, not its binding's (measured: the
			// instance variables' objects), as is an array's element type.
			NSUInteger shape = 0, dataType = 0;
			bool depth = false;
			textureType(n.typeName, shape, dataType, depth);
			info = texture(dataType, shape, accessOf(n.access), depth);
			type = MTLDataTypeTexture;
			if (n.count > 1) {
				id a = array(n.count, MTLDataTypeTexture, 0, info);
				if (a == nil)
					return nil;
				[a setIndirectArgumentIndexStride:1];
				info = a;
				type = MTLDataTypeArray;
			}
		} else if (n.kind == "air.buffer") {
			type = MTLDataTypePointer;
		} else if (n.kind == "air.sampler") {
			type = MTLDataTypeSampler;
		} else if (n.kind == "air.indirect_constant") {
			uint32_t align = 0;
			if (!valueType(n.typeName, type, align))
				return nil;
		}
		if (n.count > 1 && n.kind != "air.texture") {
			fail(@"an argument buffer's array of other than textures");
			return nil;
		}
		return member(m.name, m.offset, type, n.index, rt, rog, info, binding);
	}

	// The binding of a resource argument (or an argument buffer's member, `m` its layout entry),
	// and in `*info` its type object; nil for another kind.
	id resource(const RgfxAirNode &n, RgfxAirEntryStage stage, id __strong *info, const RgfxAirMember *m)
	{
		NSString *name = @(n.name.c_str());
		const BOOL used = !n.unused;
		*info = nil;
		if (n.kind == "air.texture") {
			NSUInteger type = 0, dataType = 0;
			bool depth = false;
			if (!textureType(n.typeName, type, dataType, depth)) {
				fail([NSString stringWithFormat:@"a texture of %s was not measured", n.typeName.c_str()]);
				return nil;
			}
			const NSUInteger access = accessOf(n.access);
			id t = texture(dataType, type, access, depth);
			id b = [make(c.textureBinding) initWithName:name
			                                    access:access
			                                  isActive:used
			                             locationIndex:n.index
			                                 arraySize:n.count
			                                  dataType:dataType
			                               textureType:type
			                            isDepthTexture:depth];
			// Its texture type: Metal's own object's, the binding's and its member's alike.
			if (b == nil || !setObject(b, "_typeInfo", t))
				return nil;
			*info = t;
			return argument(b, used);
		}
		if (n.kind == "air.sampler") {
			id b = [make(c.binding) initWithName:name
			                               type:MTLBindingTypeSampler
			                             access:MTLBindingAccessReadOnly
			                              index:n.index
			                             active:used
			                        arrayLength:n.count];
			return argument(b, used);
		}
		if (n.kind == "air.indirect_constant") {
			NSUInteger type = 0;
			uint32_t align = 0;
			if (m == nullptr || !valueType(n.typeName, type, align)) {
				fail(@"an argument buffer's constant of a type not measured");
				return nil;
			}
			id b = [make(c.indirectConstant) initWithName:name
			                                        type:kIndirectConstantBindingType
			                                      access:MTLBindingAccessReadOnly
			                                    isActive:used
			                               locationIndex:n.index
			                                    dataType:type
			                                 pixelFormat:MTLPixelFormatInvalid
			                                     aluType:0
			                                    dataSize:m->size
			                                   alignment:align];
			return argument(b, used);
		}
		if (n.kind == "air.payload") {
			if (n.layout == nullptr || !n.hasTypeSize || !n.hasTypeAlign) {
				fail(@"a payload of other than a struct");
				return nil;
			}
			id st = structType(*n.layout, false);
			const NSUInteger access =
			    stage == RgfxAirEntryStage::Object ? MTLBindingAccessReadWrite : MTLBindingAccessReadOnly;
			id p = st == nil ? nil : pointer(MTLDataTypeStruct, st, access, n.typeAlign, n.typeSize, false, false);
			id b = [make(c.payload) initWithName:name
			                               type:MTLBindingTypeObjectPayload
			                             access:access
			                           isActive:used
			                      locationIndex:kNoIndex
			                          arraySize:1
			                           dataType:MTLDataTypeStruct
			                           dataSize:n.typeSize
			                          alignment:n.typeAlign];
			if (p == nil || b == nil || !setObject(b, "_typeInfo", p))
				return nil;
			*info = p;
			return argument(b, used);
		}
		const bool argumentBuffer = n.kind == "air.indirect_buffer";
		if (n.kind != "air.buffer" && !argumentBuffer) {
			fail([NSString stringWithFormat:@"a binding of %s was not measured", n.kind.c_str()]);
			return nil;
		}
		if (!n.hasAddressSpace || !n.hasTypeSize || !n.hasTypeAlign) {
			fail(@"a buffer of no address space or pointee");
			return nil;
		}
		NSUInteger type = MTLDataTypeStruct;
		uint32_t align = 0;
		id st = nil;
		if (n.layout != nullptr) {
			st = structType(*n.layout, argumentBuffer);
			if (st == nil)
				return nil;
		} else if (argumentBuffer || !valueType(n.typeName, type, align)) {
			fail([NSString stringWithFormat:@"a buffer of %s was not measured", n.typeName.c_str()]);
			return nil;
		}
		const NSUInteger access = accessOf(n.access);
		if (n.addressSpace == 3) {
			if (argumentBuffer || m != nullptr)
				return nil;
			id p = pointer(type, st, access, n.typeAlign, n.typeSize, false, false);
			id b = [make(c.threadgroupBinding) initWithName:name
			                                          type:MTLBindingTypeThreadgroupMemory
			                                        access:access
			                                      isActive:used
			                                 locationIndex:n.index
			                                     arraySize:n.count
			                                      dataType:type
			                                      dataSize:n.typeSize
			                                     alignment:n.typeAlign];
			if (p == nil || b == nil || !setObject(b, "_typeInfo", p))
				return nil;
			*info = p;
			return argument(b, used);
		}
		if (n.addressSpace != 1 && n.addressSpace != 2) {
			fail(@"a buffer of an address space not measured");
			return nil;
		}
		const bool constant = n.addressSpace == 2;
		const NSUInteger alignment = constant ? kConstantBufferAlignment : n.typeAlign;
		id p = pointer(type, st, access, alignment, n.typeSize, argumentBuffer, constant);
		id b = [make(c.bufferBinding) initWithName:name
		                                     type:MTLBindingTypeBuffer
		                                   access:access
		                                 isActive:used
		                            locationIndex:n.index
		                                arraySize:n.count
		                                 dataType:type
		                              pixelFormat:MTLPixelFormatInvalid
		                                  aluType:0
		                         isConstantBuffer:constant
		                                 dataSize:n.typeSize
		                                alignment:alignment];
		if (p == nil || b == nil || !setObject(b, "_typeInfo", p))
			return nil;
		*info = p;
		return argument(b, used);
	}

	// `b`, an argument, used or not.
	id argument(id b, BOOL used)
	{
		if (b == nil)
			return nil;
		[b setIsArgument:YES];
		[b setIsUsed:used];
		return b;
	}

	// A descriptor's buffer, the `ordinal`th of its buffers in buffer index order.
	id descriptorBuffer(NSString *prefix, size_t ordinal, const RgfxDescriptorBuffer &d, bool used)
	{
		id p = pointer(MTLDataTypeStruct, nil, MTLBindingAccessReadOnly, kDescriptorBufferAlignment,
		               static_cast<NSUInteger>(d.stride), false, false);
		id b = [make(c.bufferBinding) initWithName:[NSString stringWithFormat:@"%@.%zu", prefix, ordinal]
		                                     type:MTLBindingTypeBuffer
		                                   access:MTLBindingAccessReadOnly
		                                 isActive:used
		                            locationIndex:d.index
		                                arraySize:1
		                                 dataType:MTLDataTypeStruct
		                              pixelFormat:MTLPixelFormatInvalid
		                                  aluType:0
		                         isConstantBuffer:NO
		                                 dataSize:static_cast<NSUInteger>(d.stride)
		                                alignment:kDescriptorBufferAlignment];
		if (p == nil || b == nil || !setObject(b, "_typeInfo", p))
			return nil;
		// No argument of the function's (measured: isArgument 0).
		[b setVertexDescriptorBuffer:YES];
		[b setIsUsed:used];
		return b;
	}

	// A built-in argument, or a mesh; nil with `*listed` false for one Metal lists nowhere.
	id builtIn(const RgfxAirNode &n, bool *listed)
	{
		*listed = true;
		NSString *name = @(n.name.c_str());
		if (n.kind == "air.mesh") {
			if (!n.hasMesh)
				return nil;
			NSUInteger topology = 0;
			if (n.meshTopology == "air.point")
				topology = MTLPrimitiveTopologyClassPoint;
			else if (n.meshTopology == "air.line")
				topology = MTLPrimitiveTopologyClassLine;
			else if (n.meshTopology == "air.triangle")
				topology = MTLPrimitiveTopologyClassTriangle;
			else {
				fail([NSString stringWithFormat:@"a mesh of %s was not measured", n.meshTopology.c_str()]);
				return nil;
			}
			id vertexStruct = dataStruct(n.meshVertexData);
			id primitiveStruct = dataStruct(n.meshPrimitiveData);
			if (vertexStruct == nil || primitiveStruct == nil)
				return nil;
			id b = [make(c.meshBuiltIn) initWithName:name
			                                   type:kMeshBindingType
			                                 access:MTLBindingAccessWriteOnly
			                                 active:!n.unused
			                                  index:0
			                               dataType:kMeshDataType
			                            builtInType:kMeshBuiltInType
			                   meshVertexStructType:vertexStruct
			                meshPrimitiveStructType:primitiveStruct
			                        meshVertexCount:static_cast<unsigned short>(n.meshVertices)
			                     meshPrimitiveCount:static_cast<unsigned short>(n.meshPrimitives)
			                       meshTopologyType:topology];
			return argument(b, !n.unused);
		}
		NSUInteger type = 0;
		bool known = false;
		for (const auto &k : kBuiltIns) {
			if (n.kind == k.kind) {
				type = k.type;
				known = true;
			}
		}
		if (!known) {
			fail([NSString stringWithFormat:@"a built-in %s was not measured", n.kind.c_str()]);
			return nil;
		}
		if (type == 0) {
			*listed = false;
			return nil;
		}
		NSUInteger dataType = 0;
		uint32_t align = 0;
		if (n.kind == "air.mesh_grid_properties")
			dataType = kMeshGridPropertiesDataType;
		else if (!valueType(n.typeName, dataType, align)) {
			fail([NSString stringWithFormat:@"a built-in of %s was not measured", n.typeName.c_str()]);
			return nil;
		}
		id b = [make(c.builtIn) initWithName:name
		                               type:kBuiltInBindingType
		                             access:MTLBindingAccessReadOnly
		                             active:!n.unused
		                              index:0
		                           dataType:dataType
		                        builtInType:type];
		return argument(b, !n.unused);
	}

	// A mesh's vertex or primitive data, as a struct of its members, each at offset 0.
	id dataStruct(const std::vector<RgfxAirOutput> &data)
	{
		NSMutableArray *members = [NSMutableArray arrayWithCapacity:data.size()];
		for (const RgfxAirOutput &o : data) {
			NSUInteger type = 0;
			uint32_t align = 0;
			if (!valueType(o.typeName, type, align)) {
				fail([NSString stringWithFormat:@"a mesh's data of %s was not measured", o.typeName.c_str()]);
				return nil;
			}
			id one = member(o.name, 0, type, 0, kNoIndex, kNoIndex, nil, nil);
			if (one == nil)
				return nil;
			[members addObject:one];
		}
		return structOf(members, false);
	}
};

// The value of the bool function constant `name` among the function's `values`: false where the
// module declares no such constant or the function has no bool value for it.
bool predicateValue(const RgfxAirEntries &module, const std::string &name, const RgfxFunctionConstant *values,
                    uint32_t count, bool &out)
{
	for (const auto &c : module.constants) {
		if (c.first != name)
			continue;
		for (uint32_t i = 0; i < count; i++) {
			if (values[i].index == c.second && values[i].data_type == MTLDataTypeBool) {
				out = values[i].value[0] != 0;
				return true;
			}
		}
		return false;
	}
	return false;
}

// The size of a vertex function's outputs: each one's value type's (a packed size: lanes times the
// scalar's), and nothing for a void function.
bool outputBytes(const std::vector<RgfxAirOutput> &outputs, uint64_t &out)
{
	out = 0;
	for (const RgfxAirOutput &o : outputs) {
		NSUInteger type = 0;
		uint32_t align = 0;
		if (!valueType(o.typeName, type, align))
			return false;
		for (const auto &s : kScalars) {
			if (type >= s.type && type <= s.type + 3 &&
			    o.typeName.compare(o.typeName.compare(0, 7, "packed_") == 0 ? 7 : 0, strlen(s.name), s.name) == 0) {
				out += static_cast<uint64_t>(s.bytes) * (type - s.type + 1);
				break;
			}
		}
	}
	return true;
}

// An immutable array of `a`'s objects; nil where it is empty.
NSArray *listOf(NSMutableArray *a)
{
	return a.count == 0 ? nil : [NSArray arrayWithArray:a];
}

} // namespace

std::vector<RgfxDescriptorBuffer> RgfxVertexDescriptorBuffers(MTLVertexDescriptor *descriptor)
{
	std::vector<RgfxDescriptorBuffer> out;
	if (descriptor == nil)
		return out;
	for (NSUInteger i = 0; i < kMaxAttributes; i++) {
		MTLVertexAttributeDescriptor *a = descriptor.attributes[i];
		if (a.format == MTLVertexFormatInvalid)
			continue;
		auto at = out.begin();
		while (at != out.end() && at->index < a.bufferIndex)
			++at;
		if (at == out.end() || at->index != a.bufferIndex) {
			RgfxDescriptorBuffer b;
			b.index = static_cast<uint32_t>(a.bufferIndex);
			b.stride = descriptor.layouts[a.bufferIndex].stride;
			at = out.insert(at, b);
		}
		at->attributes.push_back(static_cast<uint32_t>(i));
	}
	return out;
}

std::vector<RgfxDescriptorBuffer> RgfxStageInputBuffers(MTLStageInputOutputDescriptor *descriptor)
{
	std::vector<RgfxDescriptorBuffer> out;
	if (descriptor == nil)
		return out;
	for (NSUInteger i = 0; i < kMaxAttributes; i++) {
		MTLAttributeDescriptor *a = descriptor.attributes[i];
		if (a.format == MTLAttributeFormatInvalid)
			continue;
		auto at = out.begin();
		while (at != out.end() && at->index < a.bufferIndex)
			++at;
		if (at == out.end() || at->index != a.bufferIndex) {
			RgfxDescriptorBuffer b;
			b.index = static_cast<uint32_t>(a.bufferIndex);
			b.stride = descriptor.layouts[a.bufferIndex].stride;
			at = out.insert(at, b);
		}
		at->attributes.push_back(static_cast<uint32_t>(i));
	}
	return out;
}

bool RgfxStageReflectionOfEntry(const RgfxAirEntries &module, const RgfxAirEntry &entry,
                                const RgfxFunctionConstant *values, uint32_t count, NSString *prefix,
                                const std::vector<RgfxDescriptorBuffer> &buffers, RgfxStageReflection &out,
                                NSString *__strong *why)
{
	const Classes &c = classes();
	if (c.missing != nil) {
		*why = [NSString stringWithFormat:@"Metal has no %@", c.missing];
		return false;
	}
	Maker m{c};
	NSMutableArray *bindings = [NSMutableArray array];
	NSMutableArray *builtIns = [NSMutableArray array];
	// The descriptor's buffers, each used where the function reads an attribute it holds.
	std::set<uint32_t> read;
	for (const RgfxAirNode &n : entry.arguments)
		if (isStageInput(n.kind) && n.hasLocation && !n.unused)
			read.insert(n.index);
	for (size_t i = 0; i < buffers.size(); i++) {
		const RgfxDescriptorBuffer &d = buffers[i];
		bool used = false;
		for (uint32_t a : d.attributes)
			used = used || read.count(a) != 0;
		id b = m.descriptorBuffer(prefix, i, d, used);
		if (b == nil) {
			*why = m.why ?: @"Metal's descriptor buffer was not made";
			return false;
		}
		[bindings addObject:b];
	}
	for (const RgfxAirNode &n : entry.arguments) {
		if (isStageInput(n.kind))
			continue;
		if (n.predicated) {
			bool on = false;
			if (!predicateValue(module, n.predicate, values, count, on)) {
				*why = [NSString stringWithFormat:@"an argument's predicate %s is no constant of the function's values",
				                                  n.predicate.c_str()];
				return false;
			}
			if (!on)
				continue;
		}
		if (n.indexIsGlobal) {
			*why = @"an argument's binding index is computed from function constants";
			return false;
		}
		const bool resource = n.kind == "air.buffer" || n.kind == "air.indirect_buffer" || n.kind == "air.texture" ||
		                      n.kind == "air.sampler" || n.kind == "air.payload";
		if (resource) {
			id info = nil;
			id b = m.resource(n, entry.stage, &info, nullptr);
			if (b == nil) {
				*why = m.why ?: [NSString stringWithFormat:@"Metal's binding of %s was not made", n.kind.c_str()];
				return false;
			}
			[bindings addObject:b];
			continue;
		}
		// A fragment function's built-ins and inputs Metal lists nowhere.
		if (entry.stage == RgfxAirEntryStage::Fragment) {
			if (n.hasLocation) {
				*why = [NSString stringWithFormat:@"a fragment binding of %s was not measured", n.kind.c_str()];
				return false;
			}
			continue;
		}
		bool listed = true;
		id b = m.builtIn(n, &listed);
		if (b == nil && listed) {
			*why = m.why ?: [NSString stringWithFormat:@"Metal's built-in of %s was not made", n.kind.c_str()];
			return false;
		}
		if (b != nil)
			[builtIns addObject:b];
	}
	uint64_t bytes = 0;
	if (entry.stage == RgfxAirEntryStage::Vertex && !outputBytes(entry.outputs, bytes)) {
		*why = @"a vertex output of a type not measured";
		return false;
	}
	out.bindings = listOf(bindings);
	out.builtIns = listOf(builtIns);
	out.outputBytes = bytes;
	return true;
}

id RgfxComputePipelineReflectionMake(const RgfxStageReflection &kernel, NSString *__strong *why)
{
	const Classes &c = classes();
	if (c.missing != nil) {
		*why = [NSString stringWithFormat:@"Metal has no %@", c.missing];
		return nil;
	}
	Maker m{c};
	id r = [[c.compute alloc] init];
	if (r == nil || !m.setObject(r, "_arguments", kernel.bindings) ||
	    !m.setObject(r, "_bindings", kernel.bindings ? [NSArray arrayWithArray:kernel.bindings] : nil) ||
	    !m.setObject(r, "_builtInArguments", kernel.builtIns) || !m.setNumber(r, "_traceBufferIndex", 'I', kNoIndex)) {
		*why = m.why ?: @"Metal's compute pipeline reflection was not made";
		return nil;
	}
	return r;
}

id RgfxRenderPipelineReflectionMake(const RgfxStageReflection *vertex, const RgfxStageReflection *fragment,
                                    const RgfxStageReflection *object, const RgfxStageReflection *mesh,
                                    NSString *__strong *why)
{
	const Classes &c = classes();
	if (c.missing != nil) {
		*why = [NSString stringWithFormat:@"Metal has no %@", c.missing];
		return nil;
	}
	Maker m{c};
	id r = [[c.render alloc] init];
	bool ok = r != nil;
	const struct {
		const RgfxStageReflection *stage;
		const char *arguments, *bindings, *builtIns;
	} kStages[] = {
	    {vertex, "_vertexArguments", "_vertexBindings", "_vertexBuiltInArguments"},
	    {fragment, "_fragmentArguments", "_fragmentBindings", nullptr},
	    {object, "_objectArguments", "_objectBindings", "_objectBuiltInArguments"},
	    {mesh, "_meshArguments", "_meshBindings", "_meshBuiltInArguments"},
	};
	for (const auto &s : kStages) {
		if (!ok || s.stage == nullptr || s.stage->bindings == nil)
			continue;
		ok = m.setObject(r, s.arguments, s.stage->bindings) &&
		     m.setObject(r, s.bindings, [NSArray arrayWithArray:s.stage->bindings]);
	}
	for (const auto &s : kStages) {
		if (ok && s.stage != nullptr && s.stage->builtIns != nil && s.builtIns != nullptr)
			ok = m.setObject(r, s.builtIns, s.stage->builtIns);
	}
	const uint64_t stride = vertex != nullptr ? vertex->outputBytes + kPostVertexDumpPadding : 0;
	ok = ok && m.setNumber(r, "_postVertexDumpStride", 'Q', stride) &&
	     m.setNumber(r, "_traceBufferIndex", 'I', kNoIndex) &&
	     m.setNumber(r, "_printStyle", 'Q', mesh != nullptr ? kMeshPrintStyle : kVertexPrintStyle);
	if (!ok) {
		*why = m.why ?: @"Metal's render pipeline reflection was not made";
		return nil;
	}
	return r;
}
