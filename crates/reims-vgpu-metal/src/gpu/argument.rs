//! Argument-buffer layouts read off Metal's own `MTLArgumentEncoder`s ([`GpuDevice::argument_layout`]
//! and [`GpuDevice::function_argument_layout`]).
//!
//! Metal's encoder says its `encodedLength` and `alignment`, and where an argument index lands as
//! `constantDataAtIndex:` minus the start of the argument buffer it is set on; it does not say
//! what each argument is. That comes from the source:
//! - **Descriptors**: one member per descriptor, in the order given, its kind its `dataType`'s,
//!   at the offset Metal's encoder answers for its index, and an array's stride from its element
//!   1's offset.
//! - **A function's argument buffer**: the struct type `newArgumentEncoderWithBufferIndex:
//!   reflection:` reflects, member by member, at the offsets and argument indices the reflection
//!   states; an array at its reflected stride and argument index stride; a struct member's
//!   members at their own indices and offsets from the struct's; and a pointer to an argument
//!   buffer nesting the layout of the encoder `newArgumentEncoderForBufferAtIndex:` makes of it.
//!
//! A constant's size is its type's in the Metal Shading Language (a `float3` takes 16 bytes),
//! every other argument's 8, the word Metal writes. Nothing is kept: each encoder and the scratch
//! buffer its offsets are read over are released before the call returns.
//!
//! [`GpuDevice::argument_layout`]: reims_vgpu_core::platform::device::GpuDevice::argument_layout
//! [`GpuDevice::function_argument_layout`]:
//!     reims_vgpu_core::platform::device::GpuDevice::function_argument_layout

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTLArgumentDescriptor, MTLArgumentEncoder, MTLArrayType, MTLBindingAccess, MTLDataType,
    MTLDevice, MTLFunction, MTLPointerType, MTLResourceOptions, MTLStructType, MTLTextureType,
};
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::types::{ArgumentKind, ArgumentLayout, ArgumentMember};
use reims_vgpu_core::platform::values::ArgumentDesc;
use reims_vgpu_core::raise::RaiseResult;

use crate::msg::{self, Policy};
use crate::spi::methods::FunctionSpi;

/// Bytes of every argument that is no constant: an address or a resource ID.
const WORD: u64 = 8;

/// The deepest nesting of struct and pointer members read: deeper ones answer no layout.
const MAX_DEPTH: u32 = 16;

/// The layout of the encoder Metal makes of `arguments` (module doc). `None` where Metal makes
/// none.
pub(crate) fn of_descriptors(
    device: &ProtocolObject<dyn MTLDevice>,
    arguments: &[ArgumentDesc],
) -> RaiseResult<Option<ArgumentLayout>> {
    let descriptors: Vec<Retained<MTLArgumentDescriptor>> = arguments
        .iter()
        .map(|a| {
            let d = msg::get("new", MTLArgumentDescriptor::new);
            msg::get("setDataType:", || {
                d.setDataType(MTLDataType(a.data_type.to_usize()))
            });
            msg::get("setIndex:", || d.setIndex(a.index.to_usize()));
            msg::get("setArrayLength:", || {
                d.setArrayLength(a.array_length.to_usize())
            });
            msg::get("setAccess:", || {
                d.setAccess(MTLBindingAccess(a.access.to_usize()))
            });
            msg::get("setTextureType:", || {
                d.setTextureType(MTLTextureType(a.texture_type.to_usize()))
            });
            msg::get("setConstantBlockAlignment:", || {
                d.setConstantBlockAlignment(a.constant_block_alignment.to_usize())
            });
            d
        })
        .collect();
    let array = msg::value("arrayWithObjects:count:", || {
        NSArray::from_retained_slice(&descriptors)
    });
    let Some(encoder) = msg::send("newArgumentEncoderWithArguments:", Policy::Caught, || {
        device.newArgumentEncoderWithArguments(&array)
    })?
    else {
        return Ok(None);
    };
    let Some(offsets) = Offsets::over(device, &encoder)? else {
        return Ok(None);
    };
    let mut members = Vec::with_capacity(arguments.len());
    for a in arguments {
        let (Ok(id), Ok(count)) = (u32::try_from(a.index), u32::try_from(a.array_length.max(1)))
        else {
            return Ok(None);
        };
        let offset = offsets.at(&encoder, a.index)?;
        let size = size_of_type(a.data_type);
        let stride = if count > 1 {
            offsets.at(&encoder, a.index + 1)?.wrapping_sub(offset)
        } else {
            size
        };
        members.push(ArgumentMember {
            id,
            count,
            ids_per_element: 1,
            offset,
            stride,
            size,
            kind: kind_of(a.data_type),
            layout: None,
        });
    }
    Ok(Some(ArgumentLayout {
        encoded_length: msg::get("encodedLength", || encoder.encodedLength()).to_u64(),
        alignment: msg::get("alignment", || encoder.alignment()).to_u64(),
        members,
    }))
}

/// The layout of the encoder Metal makes of the argument buffer `function` takes at
/// `buffer_index` (module doc). `None` where Metal makes none, or reflects no struct type.
///
/// The `reflection:` form is deprecated, for `newArgumentEncoderWithBufferBinding:` of a
/// pipeline's reflected binding; it is the one that reflects the argument buffer of a function
/// alone, which is what the guest's `newArgumentEncoderWithBufferIndex:` names.
#[allow(deprecated)]
pub(crate) fn of_function(
    function: &ProtocolObject<dyn MTLFunction>,
    buffer_index: u64,
) -> RaiseResult<Option<ArgumentLayout>> {
    let mut reflection = None;
    let encoder = msg::send(
        "newArgumentEncoderWithBufferIndex:reflection:",
        Policy::Caught,
        || {
            function
                .newArgumentEncoderWithBufferIndex_reflection_or_nil(buffer_index, &mut reflection)
        },
    )?;
    let (Some(encoder), Some(reflection)) = (encoder, reflection) else {
        return Ok(None);
    };
    let Some(st) = msg::get("bufferStructType", || reflection.bufferStructType()) else {
        return Ok(None);
    };
    of_struct(&encoder, &st, 0)
}

/// The layout of `encoder`, whose arguments `st` reflects, at nesting `depth`.
fn of_struct(
    encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
    st: &MTLStructType,
    depth: u32,
) -> RaiseResult<Option<ArgumentLayout>> {
    let Some(members) = members(Some(encoder), st, depth)? else {
        return Ok(None);
    };
    Ok(Some(ArgumentLayout {
        encoded_length: msg::get("encodedLength", || encoder.encodedLength()).to_u64(),
        alignment: msg::get("alignment", || encoder.alignment()).to_u64(),
        members,
    }))
}

/// The members `st` reflects, each nested argument buffer's layout read off the encoder
/// `encoder` makes of it (none in a struct element, which no encoder holds: its pointers nest
/// no layout).
fn members(
    encoder: Option<&ProtocolObject<dyn MTLArgumentEncoder>>,
    st: &MTLStructType,
    depth: u32,
) -> RaiseResult<Option<Vec<ArgumentMember>>> {
    if depth > MAX_DEPTH {
        return Ok(None);
    }
    let mut out = Vec::new();
    for m in msg::get("members", || st.members().to_vec()) {
        let Ok(id) = u32::try_from(msg::get("argumentIndex", || m.argumentIndex())) else {
            return Ok(None);
        };
        let offset = msg::get("offset", || m.offset()).to_u64();
        let data_type = msg::get("dataType", || m.dataType());
        let member = if data_type == MTLDataType::Array {
            let Some(at) = msg::get("arrayType", || m.arrayType()) else {
                return Ok(None);
            };
            let Some(member) = array_member(encoder, &at, id, offset, depth)? else {
                return Ok(None);
            };
            member
        } else if data_type == MTLDataType::Struct {
            let Some(inner) = msg::get("structType", || m.structType()) else {
                return Ok(None);
            };
            let Some(layout) = struct_element(&inner, 0, depth)? else {
                return Ok(None);
            };
            let span = id_span(&layout.members);
            ArgumentMember {
                id,
                count: 1,
                ids_per_element: span.max(1),
                offset,
                stride: layout.encoded_length,
                size: layout.encoded_length,
                kind: ArgumentKind::Struct,
                layout: Some(Box::new(layout)),
            }
        } else if data_type == MTLDataType::Pointer {
            let pointer = msg::get("pointerType", || m.pointerType());
            let nested = match (encoder, pointer) {
                (Some(encoder), Some(pt)) => nested_of_pointer(encoder, id, &pt, depth)?,
                _ => None,
            };
            ArgumentMember {
                id,
                count: 1,
                ids_per_element: 1,
                offset,
                stride: WORD,
                size: WORD,
                kind: ArgumentKind::Buffer,
                layout: nested.map(Box::new),
            }
        } else {
            let raw = data_type.0.to_u64();
            let size = size_of_type(raw);
            ArgumentMember {
                id,
                count: 1,
                ids_per_element: 1,
                offset,
                stride: size,
                size,
                kind: kind_of(raw),
                layout: None,
            }
        };
        out.push(member);
    }
    Ok(Some(out))
}

/// An array member reflected as `at`, of element 0 at argument index `id` and byte `offset`.
fn array_member(
    encoder: Option<&ProtocolObject<dyn MTLArgumentEncoder>>,
    at: &MTLArrayType,
    id: u32,
    offset: u64,
    depth: u32,
) -> RaiseResult<Option<ArgumentMember>> {
    let (Ok(count), Ok(ids)) = (
        u32::try_from(msg::get("arrayLength", || at.arrayLength())),
        u32::try_from(msg::get("argumentIndexStride", || at.argumentIndexStride())),
    ) else {
        return Ok(None);
    };
    let stride = msg::get("stride", || at.stride()).to_u64();
    let element = msg::get("elementType", || at.elementType());
    let (kind, size, layout) = if element == MTLDataType::Struct {
        let Some(inner) = msg::get("elementStructType", || at.elementStructType()) else {
            return Ok(None);
        };
        let Some(layout) = struct_element(&inner, stride, depth)? else {
            return Ok(None);
        };
        (ArgumentKind::Struct, stride, Some(Box::new(layout)))
    } else if element == MTLDataType::Pointer {
        let pointer = msg::get("elementPointerType", || at.elementPointerType());
        let nested = match (encoder, pointer) {
            (Some(encoder), Some(pt)) => nested_of_pointer(encoder, id, &pt, depth)?,
            _ => None,
        };
        (ArgumentKind::Buffer, WORD, nested.map(Box::new))
    } else if element == MTLDataType::Array {
        return Ok(None); // an array of arrays, which Metal's encoder creation raises at
    } else {
        let raw = element.0.to_u64();
        (kind_of(raw), size_of_type(raw), None)
    };
    let ids_per_element = match &layout {
        Some(l) if kind == ArgumentKind::Struct && ids == 0 => id_span(&l.members).max(1),
        _ => ids.max(1),
    };
    Ok(Some(ArgumentMember {
        id,
        count: count.max(1),
        ids_per_element,
        offset,
        stride,
        size,
        kind,
        layout,
    }))
}

/// One element of a struct member reflected as `st`, of `size` bytes (0 for its members' end):
/// its members at indices and offsets from the element's, alignment 1.
fn struct_element(
    st: &MTLStructType,
    size: u64,
    depth: u32,
) -> RaiseResult<Option<ArgumentLayout>> {
    let Some(members) = members(None, st, depth + 1)? else {
        return Ok(None);
    };
    let end = members
        .iter()
        .map(|m| {
            m.offset
                .saturating_add(
                    m.stride
                        .saturating_mul(u64::from(m.count.saturating_sub(1))),
                )
                .saturating_add(m.size)
        })
        .max()
        .unwrap_or(0);
    Ok(Some(ArgumentLayout {
        encoded_length: if size != 0 { size } else { end },
        alignment: 1,
        members,
    }))
}

/// The layout of the argument buffer the pointer `pt` at argument index `id` of `encoder` points
/// to; `None` where it points to no argument buffer, or to one of no reflected struct.
fn nested_of_pointer(
    encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
    id: u32,
    pt: &MTLPointerType,
    depth: u32,
) -> RaiseResult<Option<ArgumentLayout>> {
    if !msg::get("elementIsArgumentBuffer", || pt.elementIsArgumentBuffer()) {
        return Ok(None);
    }
    match msg::get("elementStructType", || pt.elementStructType()) {
        Some(inner) => nested(encoder, id, &inner, depth),
        None => Ok(None),
    }
}

/// The layout of the argument buffer the pointer at argument index `id` of `encoder` points to,
/// whose arguments `st` reflects.
fn nested(
    encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
    id: u32,
    st: &MTLStructType,
    depth: u32,
) -> RaiseResult<Option<ArgumentLayout>> {
    let inner = msg::send(
        "newArgumentEncoderForBufferAtIndex:",
        Policy::Caught,
        || {
            // SAFETY: the index is one the reflection states for a pointer to an argument buffer.
            unsafe { encoder.newArgumentEncoderForBufferAtIndex(id.to_usize()) }
        },
    )?;
    match inner {
        Some(inner) => of_struct(&inner, st, depth + 1),
        None => Ok(None),
    }
}

/// One past the highest argument index `members` take.
fn id_span(members: &[ArgumentMember]) -> u32 {
    members
        .iter()
        .map(|m| {
            m.id.saturating_add(m.count.saturating_mul(m.ids_per_element.max(1)))
        })
        .max()
        .unwrap_or(0)
}

/// A scratch `Shared` buffer an encoder is set on, so `constantDataAtIndex:` answers offsets
/// from its contents.
struct Offsets {
    buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    base: usize,
}

impl Offsets {
    /// The scratch buffer of `encoder`'s length (at least one word), set on it. `None` where
    /// Metal makes no buffer.
    fn over(
        device: &ProtocolObject<dyn MTLDevice>,
        encoder: &ProtocolObject<dyn MTLArgumentEncoder>,
    ) -> RaiseResult<Option<Offsets>> {
        let length = msg::get("encodedLength", || encoder.encodedLength()).max(WORD.to_usize());
        let Some(buffer) = msg::send("newBufferWithLength:options:", Policy::Caught, || {
            device.newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        })?
        else {
            return Ok(None);
        };
        msg::send("setArgumentBuffer:offset:", Policy::Caught, || {
            // SAFETY: the buffer is Shared, holds the encoder's length, and outlives every
            // `constantDataAtIndex:` sent over it (it lives in `Offsets`).
            unsafe { encoder.setArgumentBuffer_offset(Some(&buffer), 0) }
        })?;
        let base =
            msg::get("contents", || objc2_metal::MTLBuffer::contents(&*buffer)).as_ptr() as usize;
        Ok(Some(Offsets { buffer, base }))
    }

    /// Where argument `index` lands, from the buffer's start.
    fn at(&self, encoder: &ProtocolObject<dyn MTLArgumentEncoder>, index: u64) -> RaiseResult<u64> {
        let _held = &self.buffer;
        let at = msg::send("constantDataAtIndex:", Policy::Caught, || {
            // SAFETY: the encoder is set on the scratch buffer (`over`).
            unsafe { encoder.constantDataAtIndex(index.to_usize()) }
        })?;
        Ok((at.as_ptr() as usize).wrapping_sub(self.base).to_u64())
    }
}

/// The kind of an argument of `MTLDataType` `data_type`.
fn kind_of(data_type: u64) -> ArgumentKind {
    match data_type {
        60 => ArgumentKind::Buffer,                       // Pointer
        58 => ArgumentKind::Texture,                      // Texture
        59 => ArgumentKind::Sampler,                      // Sampler
        78 => ArgumentKind::RenderPipelineState,          // RenderPipeline
        79 => ArgumentKind::ComputePipelineState,         // ComputePipeline
        80 => ArgumentKind::IndirectCommandBuffer,        // IndirectCommandBuffer
        115 => ArgumentKind::VisibleFunctionTable,        // VisibleFunctionTable
        116 => ArgumentKind::IntersectionFunctionTable,   // IntersectionFunctionTable
        117..=119 => ArgumentKind::AccelerationStructure, // Primitive, Instance, AS
        1 => ArgumentKind::Struct,
        _ => ArgumentKind::Constant,
    }
}

/// Bytes of an argument of `MTLDataType` `data_type`: a constant's size in the Metal Shading
/// Language (a three-lane vector takes four lanes, a matrix its columns of such vectors), and
/// every other argument's word.
fn size_of_type(data_type: u64) -> u64 {
    // Each family's scalar, then its two-, three- and four-lane vectors, numbered in a row
    // (Metal's MTLDataType.h); matrices as columns of the family's vectors.
    let vector = |lane: u64, scalar: u64| {
        let lanes = data_type - scalar + 1;
        lane * if lanes == 3 { 4 } else { lanes }
    };
    let matrix = |lane: u64, first: u64| {
        // float2x2, 2x3, 2x4, 3x2, 3x3, 3x4, 4x2, 4x3, 4x4 in a row.
        let n = data_type - first;
        let (columns, rows) = (n / 3 + 2, n % 3 + 2);
        columns * lane * if rows == 3 { 4 } else { rows }
    };
    match data_type {
        3..=6 => vector(4, 3),       // Float
        7..=15 => matrix(4, 7),      // Float2x2 ..= Float4x4
        16..=19 => vector(2, 16),    // Half
        20..=28 => matrix(2, 20),    // Half2x2 ..= Half4x4
        29..=32 => vector(4, 29),    // Int
        33..=36 => vector(4, 33),    // UInt
        37..=40 => vector(2, 37),    // Short
        41..=44 => vector(2, 41),    // UShort
        45..=48 => vector(1, 45),    // Char
        49..=52 => vector(1, 49),    // UChar
        53..=56 => vector(1, 53),    // Bool
        81..=84 => vector(8, 81),    // Long
        85..=88 => vector(8, 85),    // ULong
        121..=124 => vector(2, 121), // BFloat
        _ => WORD,
    }
}

#[cfg(test)]
mod tests {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLDevice as _, MTLLibrary as _};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::types::ArgumentKind;
    use reims_vgpu_core::platform::values::ArgumentDesc;

    use super::size_of_type;
    use crate::gpu::device::tests::dev;
    use crate::platform::Obj;

    fn arg(data_type: u64, index: u64, array_length: u64) -> ArgumentDesc {
        ArgumentDesc {
            data_type,
            index,
            array_length,
            texture_type: 2,
            ..ArgumentDesc::default()
        }
    }

    /// Descriptors are laid out as a C struct of their MSL types, in their order: a texture at
    /// 0, four pointers from 8 at a stride of 8, a `float4` at the next 16 bytes and a `float`
    /// after it; the encoder's length is the end rounded up to its largest alignment, 16
    /// (measured: Metal's own encoder, read back through `constantDataAtIndex:`)
    #[test]
    fn descriptors_lay_out_as_metals_encoder() {
        let d = dev();
        let layout = d
            .argument_layout(&[arg(58, 0, 0), arg(60, 1, 4), arg(6, 5, 0), arg(3, 6, 0)])
            .unwrap()
            .unwrap();
        let got: Vec<_> = layout
            .members
            .iter()
            .map(|m| (m.id, m.count, m.offset, m.stride, m.size, m.kind))
            .collect();
        assert_eq!(
            got,
            [
                (0, 1, 0, 8, 8, ArgumentKind::Texture),
                (1, 4, 8, 8, 8, ArgumentKind::Buffer),
                (5, 1, 48, 16, 16, ArgumentKind::Constant),
                (6, 1, 64, 4, 4, ArgumentKind::Constant),
            ]
        );
        assert_eq!((layout.encoded_length, layout.alignment), (80, 16));
    }

    /// The probe both platforms lay out alike.
    const SOURCE: &str = include_str!("../../../reims-vgpu-vulkan/src/gpu/argument_probe.metal");

    /// A function's argument buffer is laid out as its reflection and Metal's encoder say: each
    /// member at its offset, the `float4` at 32, and the pointer to a struct of arguments nesting
    /// that argument buffer's layout, while a pointer to plain floats nests none; a buffer index
    /// with no argument buffer makes none (measured: Metal's own encoder)
    #[test]
    fn a_functions_argument_buffer_lays_out_as_metals_encoder() {
        let d = dev();
        let source = NSString::from_str(SOURCE);
        let library = d
            .get()
            .newLibraryWithSource_options_error(&source, None)
            .expect("the probe compiles");
        let function = Obj::from_retained(
            library
                .newFunctionWithName(&NSString::from_str("reims_vgpu_argument_probe"))
                .unwrap(),
        );
        let layout = d.function_argument_layout(&function, 0).unwrap().unwrap();
        let got: Vec<_> = layout
            .members
            .iter()
            .map(|m| (m.id, m.offset, m.kind, m.layout.is_some()))
            .collect();
        assert_eq!(
            got,
            [
                (0, 0, ArgumentKind::Texture, false),
                (1, 8, ArgumentKind::Buffer, true),
                (2, 16, ArgumentKind::Sampler, false),
                (3, 32, ArgumentKind::Constant, false),
                (4, 48, ArgumentKind::Buffer, false),
            ]
        );
        assert_eq!((layout.encoded_length, layout.alignment), (64, 16));
        let inner = layout.members[1].layout.as_ref().unwrap();
        let got: Vec<_> = inner
            .members
            .iter()
            .map(|m| (m.id, m.offset, m.kind))
            .collect();
        assert_eq!(
            got,
            [(0, 0, ArgumentKind::Texture), (1, 8, ArgumentKind::Sampler)]
        );
        assert_eq!(inner.encoded_length, 16);
    }

    /// A constant takes its MSL size: a three-lane vector four lanes, a matrix its columns
    #[test]
    fn a_constant_takes_its_msl_size() {
        for (data_type, size) in [
            (3, 4),   // float
            (5, 16),  // float3
            (6, 16),  // float4
            (7, 16),  // float2x2
            (11, 48), // float3x3
            (15, 64), // float4x4
            (18, 8),  // half3
            (45, 1),  // char
            (84, 32), // long4
            (58, 8),  // texture
        ] {
            assert_eq!(size_of_type(data_type), size, "{data_type}");
        }
    }
}
