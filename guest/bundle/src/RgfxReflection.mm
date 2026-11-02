// RgfxReflection: see RgfxReflection.h. Our own ObjC++: Metal's reflection classes and their
// initialisers are named as the Objective-C runtime lists them.
#import "RgfxReflection.h"

#import "RgfxMetalPrivate.h"

#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <vector>

namespace {

// What Metal's own reflection answers where its public enums name no value (measured: macOS 27,
// M3 Max; macOS 26.7.1, AMD Radeon Pro 5300M).
// The data type of a void return, and of a pointer's void pointee.
constexpr NSUInteger kVoidDataType = 61;
// The binding type of an argument that is a scalar, vector or matrix; a struct; a pointer.
constexpr NSUInteger kValueBindingType = 18;
constexpr NSUInteger kStructBindingType = 29;
constexpr NSUInteger kPointerBindingType = 28;
// A struct member's render target, where it is none.
constexpr NSUInteger kNoRenderTarget = 0xFFFFFFFF;
// The element type of an array of vectors, and of matrices, whatever their scalar and size.
constexpr NSUInteger kArrayVectorElement = 137;
constexpr NSUInteger kArrayMatrixElement = 136;

// Why a signature that holds an acceleration structure has no reflection: Metal's own reflection
// of one ends the process (measured: macOS 27, M3 Max, and macOS 26.7.1, AMD Radeon Pro 5300M,
// where the structure is an argument or a struct's field).
NSString *const kNoAccelerationStructures =
    @"it holds an acceleration structure, which Metal's own reflection does not answer";

// Whether `t` is or holds an acceleration structure.
bool holdsAccelerationStructure(const RgfxAirType &t)
{
	if (t.kind == RgfxAirType::Kind::AccelerationStructure)
		return true;
	for (const RgfxAirType &e : t.element)
		if (holdsAccelerationStructure(e))
			return true;
	for (const RgfxAirField &f : t.fields)
		if (holdsAccelerationStructure(f.type))
			return true;
	return false;
}

// Metal's reflection classes the answers are made of, looked up once, each with the initialiser
// it is made with (RgfxMetalReflectionInitialisers).
struct Classes {
	Class type, structType, arrayType, pointerType, textureType, member, binding, textureBinding;
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
			               elementIsIndirectArgumentBuffer:isConstantBuffer:)},
			{&c.textureType, "MTLTextureReferenceTypeInternal",
			 @selector(initWithDataType:textureType:access:isDepthTexture:)},
			{&c.member, "MTLStructMemberInternal",
			 @selector(initWithName:offset:dataType:pixelFormat:aluType:indirectArgumentIndex:render_target:
			               raster_order_group:details:)},
			{&c.binding, "MTLBindingInternal",
			 @selector(initWithName:type:access:index:active:arrayLength:typeDescription:)},
			{&c.textureBinding, "MTLTextureBindingInternal",
			 @selector(initWithName:access:isActive:locationIndex:arraySize:dataType:textureType:isDepthTexture:)},
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

MTLDataType scalarDataType(RgfxAirScalar s)
{
	switch (s) {
	case RgfxAirScalar::Float: return MTLDataTypeFloat;
	case RgfxAirScalar::Half: return MTLDataTypeHalf;
	case RgfxAirScalar::BFloat: return MTLDataTypeBFloat;
	case RgfxAirScalar::Int: return MTLDataTypeInt;
	case RgfxAirScalar::UInt: return MTLDataTypeUInt;
	case RgfxAirScalar::Short: return MTLDataTypeShort;
	case RgfxAirScalar::UShort: return MTLDataTypeUShort;
	case RgfxAirScalar::Char: return MTLDataTypeChar;
	case RgfxAirScalar::UChar: return MTLDataTypeUChar;
	case RgfxAirScalar::Bool: return MTLDataTypeBool;
	case RgfxAirScalar::Long: return MTLDataTypeLong;
	case RgfxAirScalar::ULong: return MTLDataTypeULong;
	}
	return MTLDataTypeNone;
}

// The data type of `t`: false for a vector or matrix Metal has no data type for, and for an
// acceleration structure. A metal::array's, an enum's and a function table's are none
// (MTLDataTypeNone).
bool dataType(const RgfxAirType &t, MTLDataType &out)
{
	switch (t.kind) {
	case RgfxAirType::Kind::Scalar:
		out = scalarDataType(t.scalar);
		return true;
	case RgfxAirType::Kind::Vector:
		// Each scalar's vectors of 2, 3 and 4 follow it in Metal's numbering.
		out = static_cast<MTLDataType>(scalarDataType(t.scalar) + t.count - 1);
		return t.count >= 2 && t.count <= 4;
	case RgfxAirType::Kind::Matrix: {
		// Metal's matrices: float and half, 2 to 4 columns of 2 to 4 rows, by columns then rows.
		if (t.count < 2 || t.count > 4 || t.rows < 2 || t.rows > 4 ||
		    (t.scalar != RgfxAirScalar::Float && t.scalar != RgfxAirScalar::Half))
			return false;
		const NSUInteger at = (t.count - 2) * 3 + (t.rows - 2);
		out = static_cast<MTLDataType>((t.scalar == RgfxAirScalar::Float ? MTLDataTypeFloat2x2 : MTLDataTypeHalf2x2) +
		                               at);
		return true;
	}
	case RgfxAirType::Kind::Array: out = MTLDataTypeArray; return true;
	case RgfxAirType::Kind::ArrayOf: out = MTLDataTypeNone; return true;
	case RgfxAirType::Kind::Struct: out = MTLDataTypeStruct; return true;
	case RgfxAirType::Kind::Pointer: out = MTLDataTypePointer; return true;
	case RgfxAirType::Kind::Texture: out = MTLDataTypeTexture; return true;
	case RgfxAirType::Kind::Sampler: out = MTLDataTypeSampler; return true;
	case RgfxAirType::Kind::Void: out = static_cast<MTLDataType>(kVoidDataType); return true;
	case RgfxAirType::Kind::Enum:
	case RgfxAirType::Kind::VisibleFunctionTable:
	case RgfxAirType::Kind::IntersectionFunctionTable: out = MTLDataTypeNone; return true;
	case RgfxAirType::Kind::AccelerationStructure: return false;
	}
	return false;
}

// The element type an array's type is of: its element's data type, but kArrayVectorElement for a
// vector, kArrayMatrixElement for a matrix, and a function table's own data type for a table.
bool arrayElementType(const RgfxAirType &e, MTLDataType &out)
{
	if (!dataType(e, out))
		return false;
	if (e.kind == RgfxAirType::Kind::Vector)
		out = static_cast<MTLDataType>(kArrayVectorElement);
	else if (e.kind == RgfxAirType::Kind::Matrix)
		out = static_cast<MTLDataType>(kArrayMatrixElement);
	else if (e.kind == RgfxAirType::Kind::VisibleFunctionTable)
		out = MTLDataTypeVisibleFunctionTable;
	else if (e.kind == RgfxAirType::Kind::IntersectionFunctionTable)
		out = MTLDataTypeIntersectionFunctionTable;
	return true;
}

// The size and alignment a pointer to `e` is of: a scalar's, vector's, matrix's, struct's or
// array's own, a pointer's pointee's where the pointee is a pointer, and none for another (void,
// an enum, a texture, a sampler, a function table), whatever its node says.
void pointeeExtent(const RgfxAirType &e, uint32_t &size, uint32_t &align)
{
	switch (e.kind) {
	case RgfxAirType::Kind::Scalar:
	case RgfxAirType::Kind::Vector:
	case RgfxAirType::Kind::Matrix:
	case RgfxAirType::Kind::Struct:
	case RgfxAirType::Kind::Array:
	case RgfxAirType::Kind::ArrayOf:
		size = e.size;
		align = e.align;
		return;
	case RgfxAirType::Kind::Pointer:
		pointeeExtent(e.element.at(0), size, align);
		return;
	default:
		size = 0;
		align = 0;
		return;
	}
}

MTLTextureType textureType(RgfxAirTextureShape s)
{
	switch (s) {
	case RgfxAirTextureShape::T1D: return MTLTextureType1D;
	case RgfxAirTextureShape::T1DArray: return MTLTextureType1DArray;
	case RgfxAirTextureShape::T2D: return MTLTextureType2D;
	case RgfxAirTextureShape::T2DArray: return MTLTextureType2DArray;
	case RgfxAirTextureShape::T2DMS: return MTLTextureType2DMultisample;
	case RgfxAirTextureShape::T2DMSArray: return MTLTextureType2DMultisampleArray;
	case RgfxAirTextureShape::Cube: return MTLTextureTypeCube;
	case RgfxAirTextureShape::CubeArray: return MTLTextureTypeCubeArray;
	case RgfxAirTextureShape::T3D: return MTLTextureType3D;
	case RgfxAirTextureShape::Buffer1D: return MTLTextureTypeTextureBuffer;
	}
	return MTLTextureType2D;
}

MTLBindingAccess textureAccess(RgfxAirAccess a)
{
	switch (a) {
	case RgfxAirAccess::Write: return MTLBindingAccessWriteOnly;
	case RgfxAirAccess::ReadWrite: return MTLBindingAccessReadWrite;
	default: return MTLBindingAccessReadOnly;
	}
}

// Builds the type objects; `ok` turns false at a type Metal has no data type for.
struct Builder {
	const Classes &c;
	bool ok = true;

	// A new object of `cls`, to initialise through the initialisers Metal's classes have.
	static id<RgfxMetalReflectionInitialisers> make(Class cls) { return [cls alloc]; }

	id plain(NSUInteger dataType) { return [make(c.type) initWithDataType:dataType]; }

	MTLDataType typeOf(const RgfxAirType &t)
	{
		MTLDataType dt = MTLDataTypeNone;
		if (!dataType(t, dt))
			ok = false;
		return dt;
	}

	id texture(NSUInteger dataType, MTLTextureType type, MTLBindingAccess access, bool depth)
	{
		return [make(c.textureType) initWithDataType:dataType
		                                 textureType:type
		                                      access:access
		                              isDepthTexture:depth ? YES : NO];
	}

	// The type object of a texture that is an array's element or a pointer's pointee: of data type
	// texture and texture type 0, whatever its sample type and shape, not a depth texture, and of
	// its address space's number as its access.
	id elementTexture(const RgfxAirType &e)
	{
		return texture(MTLDataTypeTexture, MTLTextureType1D, static_cast<MTLBindingAccess>(e.addressSpace), false);
	}

	// The type object of `t`, nil for a sampler. `inStruct` where it is a field's type, or its
	// type's: a struct type there is nameless, and a pointer's pointee is described otherwise.
	id type(const RgfxAirType &t, bool inStruct)
	{
		const MTLDataType dt = typeOf(t);
		if (!ok)
			return nil;
		switch (t.kind) {
		case RgfxAirType::Kind::Scalar:
		case RgfxAirType::Kind::Vector:
		case RgfxAirType::Kind::Matrix:
		case RgfxAirType::Kind::ArrayOf:
		case RgfxAirType::Kind::Void:
		case RgfxAirType::Kind::Enum:
		case RgfxAirType::Kind::VisibleFunctionTable:
		case RgfxAirType::Kind::IntersectionFunctionTable:
			return plain(dt);
		case RgfxAirType::Kind::AccelerationStructure:
			return nil; // typeOf failed (rgfxStitchableReflection refuses it first)
		case RgfxAirType::Kind::Sampler:
			return nil;
		case RgfxAirType::Kind::Texture:
			return texture(scalarDataType(t.scalar), textureType(t.shape), textureAccess(t.access), t.depth);
		case RgfxAirType::Kind::Array: {
			const RgfxAirType &e = t.element.at(0);
			MTLDataType et = MTLDataTypeNone;
			if (!arrayElementType(e, et)) {
				ok = false;
				return nil;
			}
			// The element's type object is of its element type alone; a texture's is
			// elementTexture's.
			id details = e.kind == RgfxAirType::Kind::Texture ? elementTexture(e) : plain(et);
			return [make(c.arrayType) initWithArrayLength:t.count
			                                 elementType:et
			                                      stride:t.align
			                                 pixelFormat:MTLPixelFormatInvalid
			                                     aluType:0
			                                     details:details];
		}
		case RgfxAirType::Kind::Pointer: {
			const RgfxAirType &e = t.element.at(0);
			const MTLDataType et = typeOf(e);
			id pointee = nil;
			if (e.kind == RgfxAirType::Kind::Struct) {
				pointee = type(e, inStruct);
			} else if (e.kind == RgfxAirType::Kind::Texture) {
				pointee = elementTexture(e);
			} else if (e.kind == RgfxAirType::Kind::Array) {
				// An array's type object of its data type alone; a field's of its element type.
				MTLDataType inner = MTLDataTypeArray;
				if (inStruct && !arrayElementType(e.element.at(0), inner))
					ok = false;
				pointee = plain(inner);
			} else if (e.kind == RgfxAirType::Kind::Pointer && inStruct) {
				// A field's pointer to a pointer: of its pointee's pointee's data type.
				pointee = plain(typeOf(e.element.at(0)));
			} else {
				// A field's pointee's type object is of no data type.
				pointee = plain(inStruct ? MTLDataTypeNone : et);
			}
			if (!ok)
				return nil;
			uint32_t size = 0, align = 0;
			pointeeExtent(e, size, align);
			return [make(c.pointerType) initWithElementType:et
			                         elementTypeDescription:pointee
			                                         access:MTLBindingAccessReadOnly
			                                      alignment:align
			                                       dataSize:size
			                elementIsIndirectArgumentBuffer:NO
			                               isConstantBuffer:NO];
		}
		case RgfxAirType::Kind::Struct: {
			NSMutableArray *members = [NSMutableArray arrayWithCapacity:t.fields.size()];
			for (const RgfxAirField &f : t.fields) {
				const MTLDataType ft = typeOf(f.type);
				// A struct's, array's, pointer's or texture's field has its type object; another
				// has none.
				id details = nil;
				if (f.type.kind == RgfxAirType::Kind::Struct || f.type.kind == RgfxAirType::Kind::Array ||
				    f.type.kind == RgfxAirType::Kind::Pointer || f.type.kind == RgfxAirType::Kind::Texture)
					details = type(f.type, true);
				if (!ok)
					return nil;
				id m = [make(c.member) initWithName:@(f.name.c_str())
				                             offset:f.offset
				                           dataType:ft
				                        pixelFormat:MTLPixelFormatInvalid
				                            aluType:0
				              indirectArgumentIndex:0
				                      render_target:kNoRenderTarget
				                 raster_order_group:0
				                            details:details];
				if (m == nil) {
					ok = false;
					return nil;
				}
				[members addObject:m];
			}
			std::vector<__unsafe_unretained id> list(members.count);
			for (NSUInteger i = 0; i < members.count; i++)
				list[i] = members[i];
			return [make(c.structType) initWithMembers:list.data()
			                                     count:list.size()
			                                  typeName:inStruct ? @"" : @(t.name.c_str())];
		}
		}
		ok = false;
		return nil;
	}

	id binding(const std::string &name, const RgfxAirType &t)
	{
		NSString *n = @(name.c_str());
		if (t.kind == RgfxAirType::Kind::Texture)
			return [make(c.textureBinding) initWithName:n
			                                     access:textureAccess(t.access)
			                                   isActive:NO
			                              locationIndex:0
			                                  arraySize:0
			                                   dataType:scalarDataType(t.scalar)
			                                textureType:textureType(t.shape)
			                             isDepthTexture:t.depth ? YES : NO];
		NSUInteger bindingType = kValueBindingType;
		if (t.kind == RgfxAirType::Kind::Struct)
			bindingType = kStructBindingType;
		else if (t.kind == RgfxAirType::Kind::Pointer)
			bindingType = kPointerBindingType;
		else if (t.kind == RgfxAirType::Kind::Sampler)
			bindingType = MTLBindingTypeSampler;
		else if (t.kind == RgfxAirType::Kind::Array)
			ok = false; // an argument is never an array of C's: a declared one is a pointer
		id description = type(t, false);
		if (!ok)
			return nil;
		return [make(c.binding) initWithName:n
		                                type:bindingType
		                              access:MTLBindingAccessReadOnly
		                               index:0
		                              active:NO
		                         arrayLength:0
		                     typeDescription:description];
	}
};

} // namespace

bool rgfxStitchableReflection(const RgfxAirVisible &fn, id __strong *returnType, NSArray *__strong *arguments,
                              NSString *__strong *why)
{
	const Classes &c = classes();
	if (c.missing != nil) {
		*why = [NSString stringWithFormat:@"Metal has no %@", c.missing];
		return false;
	}
	if (fn.signature != RgfxAirVisible::Signature::Read) {
		*why = @"its signature was not read";
		return false;
	}
	bool holds = !fn.returnsVoid && holdsAccelerationStructure(fn.returnType);
	for (const RgfxAirType &a : fn.argumentTypes)
		holds = holds || holdsAccelerationStructure(a);
	if (holds) {
		*why = kNoAccelerationStructures;
		return false;
	}
	Builder b{c};
	id ret = fn.returnsVoid ? b.plain(kVoidDataType) : b.type(fn.returnType, false);
	NSMutableArray *args = [NSMutableArray arrayWithCapacity:fn.argumentTypes.size()];
	for (size_t i = 0; b.ok && i < fn.argumentTypes.size(); i++) {
		id a = b.binding(fn.argumentNames.at(i), fn.argumentTypes[i]);
		if (a != nil)
			[args addObject:a];
		else
			b.ok = false;
	}
	if (!b.ok || ret == nil) {
		*why = @"it has a type Metal has no data type for";
		return false;
	}
	*returnType = ret;
	// An immutable array, as Metal's own is.
	std::vector<__unsafe_unretained id> list(args.count);
	for (NSUInteger i = 0; i < args.count; i++)
		list[i] = args[i];
	*arguments = [NSArray arrayWithObjects:list.data() count:list.size()];
	return true;
}
