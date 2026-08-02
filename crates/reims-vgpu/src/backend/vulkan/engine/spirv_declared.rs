//! Descriptor slots a SPIR-V module declares.
//!
//! The engine builds its descriptor-set layout from the resources a request
//! provides. A guest draw may name fewer than the shader reads — Metal answers
//! such a slot with zeros, so the guest issues it freely — and the shader then
//! indexes a descriptor the set never declared. That is undefined in Vulkan and
//! the ICDs disagree loudly: NVIDIA returns garbage, lavapipe dereferences null
//! and takes the process down. Scanning the module tells the layout builder
//! which slots must exist regardless of what the request carried.

use std::collections::HashMap;

const OP_DECORATE: u16 = 71;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLER: u16 = 26;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_POINTER: u16 = 32;
const OP_VARIABLE: u16 = 59;

const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;

const DIM_SUBPASS_DATA: u32 = 6;

const SPIRV_MAGIC: u32 = 0x0723_0203;
const SPIRV_HEADER_WORDS: usize = 5;

/// What a declared slot needs from the layout. Storage buffers are already
/// derived from the request in every path that has them, so only the image and
/// sampler kinds — the ones a draw legitimately omits — are reported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeclaredKind {
    SampledImage,
    Sampler,
}

/// `(set, binding, kind)` for every image or sampler variable the module
/// decorates with both a set and a binding. Returns nothing for a module that
/// is not valid SPIR-V rather than guessing — a caller that cannot parse the
/// shader has no business editing the layout it implies.
pub(crate) fn declared_descriptors(words: &[u32]) -> Vec<(u32, u32, DeclaredKind)> {
    if words.len() < SPIRV_HEADER_WORDS || words[0] != SPIRV_MAGIC {
        return Vec::new();
    }
    let mut binding_of: HashMap<u32, u32> = HashMap::new();
    let mut set_of: HashMap<u32, u32> = HashMap::new();
    let mut image_types: Vec<u32> = Vec::new();
    let mut sampler_types: Vec<u32> = Vec::new();
    // pointer result id -> pointee type id
    let mut pointee_of: HashMap<u32, u32> = HashMap::new();
    // variable result id -> its pointer type id
    let mut variables: Vec<(u32, u32)> = Vec::new();

    let mut i = SPIRV_HEADER_WORDS;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = (words[i] & 0xffff) as u16;
        // A zero word count cannot advance the cursor: stop rather than spin.
        if word_count == 0 || i + word_count > words.len() {
            break;
        }
        let operands = &words[i + 1..i + word_count];
        match opcode {
            OP_DECORATE if operands.len() >= 3 => match operands[1] {
                DECORATION_BINDING => {
                    binding_of.insert(operands[0], operands[2]);
                }
                DECORATION_DESCRIPTOR_SET => {
                    set_of.insert(operands[0], operands[2]);
                }
                _ => {}
            },
            // Dim (operand 2) SubpassData means an input attachment, which the
            // engine binds from the request's own color-input state — adding it
            // here would declare binding 96 twice with two different types.
            OP_TYPE_IMAGE if operands.len() >= 3 && operands[2] != DIM_SUBPASS_DATA => {
                image_types.push(operands[0]);
            }
            OP_TYPE_SAMPLED_IMAGE if !operands.is_empty() => {
                image_types.push(operands[0]);
            }
            OP_TYPE_SAMPLER if !operands.is_empty() => {
                sampler_types.push(operands[0]);
            }
            OP_TYPE_POINTER if operands.len() >= 3 => {
                pointee_of.insert(operands[0], operands[2]);
            }
            OP_VARIABLE if operands.len() >= 2 => {
                variables.push((operands[1], operands[0]));
            }
            _ => {}
        }
        i += word_count;
    }

    let mut out = Vec::new();
    for (var_id, ptr_type) in variables {
        let (Some(binding), Some(set)) = (binding_of.get(&var_id), set_of.get(&var_id)) else {
            continue;
        };
        let Some(pointee) = pointee_of.get(&ptr_type) else {
            continue;
        };
        let kind = if image_types.contains(pointee) {
            DeclaredKind::SampledImage
        } else if sampler_types.contains(pointee) {
            DeclaredKind::Sampler
        } else {
            continue;
        };
        out.push((*set, *binding, kind));
    }
    out
}
