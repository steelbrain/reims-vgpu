//! Argument-buffer layouts on the rt platform: the translator's, which are Metal's encoder's.
//!
//! The guest's argument encoder writes the arguments itself, at the offsets the layout gives
//! (reims-vgpu's `ARGUMENT_LAYOUT`): a buffer's address plus its offset, every other object's
//! resource ID, 8 bytes each, and constants as the guest gives them. The layouts come from
//! metal2vulkan's translator, by Metal's rules ([`metal2vulkan::bind::ArgumentBufferLayout`]):
//!
//! - `newArgumentEncoderWithArguments:` ([`descriptors_layout`]) answers the layout of the
//!   descriptors in their order (`ArgumentBufferLayout::from_descriptors`).
//! - `newArgumentEncoderWithBufferIndex:` ([`function_layout`]) answers the layout of the argument
//!   buffer the function takes at the index, with its constant values
//!   (`Function::argument_buffer_layout`). It is the layout the translated function's own loads
//!   read, so what the guest writes at its offsets is what the function reads.
//!
//! Each translator member becomes core's [`ArgumentMember`] with the same index, count, offset,
//! stride, size and kind. A nested argument buffer or a struct carries its own layout.
//!
//! **Refusals.** A call answers nil (`None`) where Metal makes no encoder. These are Metal's
//! refusals, counted under `rt-refused-metal`:
//! - a raw value Metal has no value of (a data type, an access, a texture type, or an index,
//!   array length or block alignment past 32 bits);
//! - no descriptors;
//! - a data type no argument has;
//! - a constant block alignment on a constant that does not start its block;
//! - no argument buffer at the function's index;
//! - an array of arrays;
//! - a length past 64 bits.
//!
//! The translator's own failure to decode or specialise the function is counted under
//! `rt-refused-translation`, and an argument kind this platform names no kind of under
//! `rt-refused-other`. The first of each kind writes its line on stderr (`super::unbuilt`).

use metal2vulkan::Budget;
use metal2vulkan::bind::{
    ArgumentBufferLayout, ArgumentDescriptor, ArgumentLayoutError, ArgumentType,
};
use metal2vulkan::metal::{BindingAccess, DataType, TextureType};
use metal2vulkan_runtime as rt;

use crate::core::platform::types::{ArgumentKind, ArgumentLayout, ArgumentMember};
use crate::core::platform::values::ArgumentDesc;

use super::library::RtFunction;
use super::unbuilt::{RtRefusal, refused_as, rt_refused_create};

/// `newArgumentEncoderWithArguments:`.
const DESCRIPTORS: &str = "newArgumentEncoderWithArguments:";

/// `newArgumentEncoderWithBufferIndex:`.
const FUNCTION: &str = "newArgumentEncoderWithBufferIndex:";

/// The layout of the encoder Metal makes of `arguments`, in their order; `None`, counted, where
/// Metal makes none (the module's docs).
pub fn descriptors_layout(arguments: &[ArgumentDesc]) -> Option<ArgumentLayout> {
    let mut descriptors = Vec::with_capacity(arguments.len());
    for (position, a) in arguments.iter().enumerate() {
        let Some(d) = descriptor(a) else {
            refused_as(
                DESCRIPTORS,
                RtRefusal::Refused,
                &format_args!("descriptor {position} has a value Metal has none of: {a:?}"),
            );
            return None;
        };
        descriptors.push(d);
    }
    answer(
        DESCRIPTORS,
        ArgumentBufferLayout::from_descriptors(&descriptors),
    )
}

/// The layout of the encoder Metal makes of the argument buffer `function` takes at
/// `buffer_index`, its nested argument buffers' with it; `None`, counted, where Metal makes none
/// (the module's docs).
pub fn function_layout(function: &RtFunction, buffer_index: u64) -> Option<ArgumentLayout> {
    let Ok(index) = u32::try_from(buffer_index) else {
        refused_as(
            FUNCTION,
            RtRefusal::Refused,
            &format_args!("buffer index {buffer_index} is past 32 bits"),
        );
        return None;
    };
    let linked = function.linked();
    let layout =
        linked
            .function
            .argument_buffer_layout(index, &linked.constants, &Budget::standard());
    answer(FUNCTION, layout)
}

/// `a` as the translator's descriptor; `None` for a value Metal has none of.
fn descriptor(a: &ArgumentDesc) -> Option<ArgumentDescriptor> {
    let raw = |v: u64| u32::try_from(v).ok();
    let mut d = ArgumentDescriptor::new(DataType::try_from(raw(a.data_type)?).ok()?, raw(a.index)?);
    d.array_length = raw(a.array_length)?;
    d.access = BindingAccess::try_from(raw(a.access)?).ok()?;
    d.texture_type = TextureType::try_from(raw(a.texture_type)?).ok()?;
    d.constant_block_alignment = raw(a.constant_block_alignment)?;
    Some(d)
}

/// The translator's `layout` of the call `sel` as core's, or `None` with its refusal counted.
fn answer(
    sel: &'static str,
    layout: Result<ArgumentBufferLayout, ArgumentLayoutError>,
) -> Option<ArgumentLayout> {
    match layout {
        Ok(layout) => {
            let converted = convert(&layout);
            if converted.is_none() {
                refused_as(
                    sel,
                    RtRefusal::Other,
                    &"the layout holds an argument kind this platform names no kind of",
                );
            }
            converted
        }
        Err(ArgumentLayoutError::Compile(error)) => {
            rt_refused_create(sel, &rt::Error::Translation(error));
            None
        }
        Err(error) => {
            refused_as(sel, RtRefusal::Refused, &format_args!("{error:?}"));
            None
        }
    }
}

/// `layout` as core's: every member with its nested layout. `None` where a member's kind is one
/// core names none of.
fn convert(layout: &ArgumentBufferLayout) -> Option<ArgumentLayout> {
    let mut members = Vec::with_capacity(layout.members.len());
    for m in &layout.members {
        let nested = match &m.layout {
            Some(nested) => Some(Box::new(convert(nested)?)),
            None => None,
        };
        members.push(ArgumentMember {
            id: m.id,
            count: m.count,
            ids_per_element: m.ids_per_element,
            offset: m.offset,
            stride: m.stride,
            size: m.size,
            kind: kind(m.ty)?,
            layout: nested,
        });
    }
    Some(ArgumentLayout {
        encoded_length: layout.encoded_length,
        alignment: layout.alignment,
        members,
    })
}

/// Core's kind of the translator's `ty`; `None` for a kind core names none of.
fn kind(ty: ArgumentType) -> Option<ArgumentKind> {
    Some(match ty {
        ArgumentType::Buffer => ArgumentKind::Buffer,
        ArgumentType::Texture => ArgumentKind::Texture,
        ArgumentType::Sampler => ArgumentKind::Sampler,
        ArgumentType::Constant => ArgumentKind::Constant,
        ArgumentType::IndirectCommandBuffer => ArgumentKind::IndirectCommandBuffer,
        ArgumentType::ComputePipelineState => ArgumentKind::ComputePipelineState,
        ArgumentType::RenderPipelineState => ArgumentKind::RenderPipelineState,
        ArgumentType::VisibleFunctionTable => ArgumentKind::VisibleFunctionTable,
        ArgumentType::IntersectionFunctionTable => ArgumentKind::IntersectionFunctionTable,
        ArgumentType::AccelerationStructure => ArgumentKind::AccelerationStructure,
        ArgumentType::DepthStencilState => ArgumentKind::DepthStencilState,
        ArgumentType::Struct => ArgumentKind::Struct,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::resource::Library as _;
    use crate::core::raise::OnFault;

    /// The kernels of `tests/shaders/argument_buffers.metal`.
    const ARGUMENT_BUFFERS: &[u8] =
        include_bytes!("../../../../tests/shaders/argument_buffers.metallib");

    /// The count of `code` in the process's degradation counts.
    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// A descriptor of `data_type` at `index`, of `array_length` elements, with Metal's defaults.
    fn desc(data_type: u64, index: u64, array_length: u64) -> ArgumentDesc {
        ArgumentDesc {
            data_type,
            index,
            array_length,
            access: 0,
            texture_type: 2,
            constant_block_alignment: 0,
        }
    }

    /// A single member of `kind` at `id`, `offset`, of `size` bytes, holding no layout.
    fn member(id: u32, count: u32, offset: u64, size: u64, kind: ArgumentKind) -> ArgumentMember {
        ArgumentMember {
            id,
            count,
            ids_per_element: 1,
            offset,
            stride: size,
            size,
            kind,
            layout: None,
        }
    }

    /// Descriptors are laid out in their order as Metal's encoder lays them out: every handle
    /// kind 8 bytes aligned to 8, an array its elements in turn, a constant its MSL size and
    /// alignment, and the length the end rounded up to the largest alignment. Each translator kind
    /// is core's kind of the same name
    #[test]
    fn descriptors_lay_out_every_kind_as_metal_s_encoder() {
        use ArgumentKind as K;
        let arguments = [
            desc(58, 0, 0),   // Texture
            desc(60, 1, 4),   // Pointer, 4 elements
            desc(3, 5, 0),    // Float
            desc(6, 6, 0),    // Float4
            desc(59, 7, 0),   // Sampler
            desc(78, 8, 0),   // RenderPipeline
            desc(79, 9, 0),   // ComputePipeline
            desc(80, 10, 0),  // IndirectCommandBuffer
            desc(115, 11, 0), // VisibleFunctionTable
            desc(116, 12, 0), // IntersectionFunctionTable
            desc(117, 13, 0), // PrimitiveAccelerationStructure
            desc(118, 14, 0), // InstanceAccelerationStructure
            desc(139, 15, 0), // DepthStencilState
            desc(33, 16, 0),  // UInt
        ];
        let layout = descriptors_layout(&arguments).unwrap();
        assert_eq!(
            layout,
            ArgumentLayout {
                encoded_length: 144,
                alignment: 16,
                members: vec![
                    member(0, 1, 0, 8, K::Texture),
                    member(1, 4, 8, 8, K::Buffer),
                    member(5, 1, 40, 4, K::Constant),
                    member(6, 1, 48, 16, K::Constant),
                    member(7, 1, 64, 8, K::Sampler),
                    member(8, 1, 72, 8, K::RenderPipelineState),
                    member(9, 1, 80, 8, K::ComputePipelineState),
                    member(10, 1, 88, 8, K::IndirectCommandBuffer),
                    member(11, 1, 96, 8, K::VisibleFunctionTable),
                    member(12, 1, 104, 8, K::IntersectionFunctionTable),
                    member(13, 1, 112, 8, K::AccelerationStructure),
                    member(14, 1, 120, 8, K::AccelerationStructure),
                    member(15, 1, 128, 8, K::DepthStencilState),
                    member(16, 1, 136, 4, K::Constant),
                ],
            }
        );
    }

    /// What Metal makes no encoder of answers nil, counted as Metal's refusal: no descriptors, a
    /// data type no argument has (`MTLDataTypeNone`), a raw data type Metal has no value of, an
    /// index past 32 bits, and a block alignment on a constant that does not start its block
    #[test]
    fn descriptors_metal_makes_no_encoder_of_answer_nil() {
        let mut aligned = desc(3, 1, 0);
        aligned.constant_block_alignment = 16;
        let refused: [&[ArgumentDesc]; 5] = [
            &[],
            &[desc(0, 0, 0)],
            &[desc(0x7fff, 0, 0)],
            &[desc(60, 1 << 32, 0)],
            &[desc(3, 0, 0), aligned],
        ];
        for arguments in refused {
            let before = counted("rt-refused-metal");
            assert_eq!(descriptors_layout(arguments), None, "{arguments:?}");
            assert_eq!(counted("rt-refused-metal"), before + 1, "{arguments:?}");
        }
    }

    /// A function's argument buffer answers the layout its translation reads, the one Metal's
    /// compiler gave it: `sum_args`' two pointers at 0 and 8, 16 bytes aligned to 8 (as Metal's
    /// encoder answers it); `mixed_args`' array of pointers, constant, texture and pointer to a
    /// nested argument buffer, which carries its own layout. Its plain buffer at index 1 has none,
    /// which Metal refuses
    #[test]
    fn a_function_s_argument_buffer_answers_the_layout_its_translation_reads() {
        use ArgumentKind as K;
        let library = super::super::library::new_library(ARGUMENT_BUFFERS).unwrap();
        let function = |name| {
            library
                .new_function(OnFault::Return, name)
                .unwrap()
                .unwrap()
        };
        let sum = function("sum_args");
        assert_eq!(
            function_layout(&sum, 0),
            Some(ArgumentLayout {
                encoded_length: 16,
                alignment: 8,
                members: vec![member(0, 1, 0, 8, K::Buffer), member(1, 1, 8, 8, K::Buffer)],
            })
        );
        let inner = ArgumentLayout {
            encoded_length: 16,
            alignment: 8,
            members: vec![
                member(0, 1, 0, 8, K::Buffer),
                member(1, 1, 8, 4, K::Constant),
            ],
        };
        let mut nested = member(4, 1, 32, 8, K::Buffer);
        nested.layout = Some(Box::new(inner));
        assert_eq!(
            function_layout(&function("mixed_args"), 0),
            Some(ArgumentLayout {
                encoded_length: 40,
                alignment: 8,
                members: vec![
                    member(0, 2, 0, 8, K::Buffer),
                    member(2, 1, 16, 4, K::Constant),
                    member(3, 1, 24, 8, K::Texture),
                    nested,
                ],
            })
        );
        let before = counted("rt-refused-metal");
        assert_eq!(function_layout(&sum, 1), None);
        assert_eq!(function_layout(&sum, 1 << 32), None);
        assert_eq!(counted("rt-refused-metal"), before + 2);
    }
}
