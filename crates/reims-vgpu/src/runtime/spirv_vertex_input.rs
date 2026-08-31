//! How many components a translated vertex shader actually reads at one
//! stage-in `Location`.
//!
//! # Why anything needs to ask
//!
//! Vulkan requires only a subset of formats to be usable as vertex attributes,
//! and every three-component 8- and 16-bit format is outside it. A device that
//! declines one is handed its mandatory four-component sibling instead, which
//! [`crate::backend::vulkan::translate::support`] calls the widening fallback.
//!
//! That substitution is exact **only** for a shader that reads three components
//! or fewer, and the difference is not in the bytes — it is in what Vulkan does
//! with the components the format does not supply. A shader input declared
//! `vec4` over a three-component attribute takes `(x, y, z, 1.0)`, because
//! Vulkan fills a missing fourth component with the constant one. Over the
//! four-component substitute it takes `(x, y, z, <whatever those bytes hold>)`,
//! because the component is now supplied rather than defaulted, and what it
//! holds is the next attribute's data or the vertex's padding.
//!
//! So the fallback needs the shader's declared width, and this module reads it.
//!
//! # Read from the words, not from reflection
//!
//! metal2vulkan does emit a `ShaderReflection` carrying `vertex_attributes`
//! with a location and an AIR type name, and reading that would be shorter. It
//! is not what the engine is given: `DrawRequest` carries post-relocation SPIR-V
//! and nothing else, on purpose. Walking the module also answers about the
//! bytes that will actually be compiled rather than about a description of
//! them, which is the same reason [`crate::runtime::spirv_bind`] rewrites
//! bindings by walking types instead of trusting the translator's numbering.
//!
//! # The walk
//!
//! Structural, in the shape of [`crate::runtime::spirv_bind::image_format`]:
//! collect `OpDecorate <id> Location <n>`, the `OpVariable`s in the `Input`
//! storage class, the `OpTypePointer`s they point through, and the component
//! count of every scalar and vector type; then join them.
//!
//! **Filtering on the `Input` storage class is load-bearing, not tidiness.** A
//! vertex shader's *outputs* carry `Location` decorations from the same
//! numbering, so a module whose output 0 is a `vec4` varying and whose input 0
//! is a `float3` attribute answers 4 to a walk that only matches on `Location`.
//! That is the wrong answer in the one direction that costs a draw.

/// What a vertex shader declares at one stage-in `Location`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputWidth {
    /// An `Input` variable at this `Location` whose type reduces to a scalar or
    /// a vector of this many components. `1..=4` for anything SPIR-V can
    /// declare as a vertex input.
    Components(u32),
    /// No `Input` variable carries this `Location`. The vertex descriptor
    /// describes an attribute the shader does not read — routine, because the
    /// descriptor is built from the mesh and the shader from the material.
    Absent,
    /// An `Input` variable carries this `Location` and this walk could not
    /// reduce its type to a component count: a matrix, an array, a struct, or a
    /// module whose instruction stream did not parse. The distinction from
    /// [`Self::Absent`] is the whole point — "reads nothing" and "reads
    /// something this cannot measure" are opposite answers for any caller
    /// deciding whether a substitution is safe.
    Unreadable,
}

/// SPIR-V module header length in words.
const HEADER_WORDS: usize = 5;
/// SPIR-V `Decoration Location`.
const DECORATION_LOCATION: u32 = 30;
/// SPIR-V `StorageClass Input`.
const STORAGE_CLASS_INPUT: u32 = 1;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_POINTER: u16 = 32;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;

/// Every stage-in `Location` a module declares, and how wide each one is.
///
/// Built once per module rather than once per attribute: the walk is linear in
/// the module and a pipeline resolves several attributes against the same
/// shader, so asking per attribute would re-read the whole module per
/// attribute.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexInputWidths {
    /// `(location, width)` sorted by location. A `Vec` rather than a map
    /// because a vertex shader declares a handful of inputs and the lookup is
    /// a binary search over single digits.
    declared: Vec<(u32, InputWidth)>,
}

impl VertexInputWidths {
    /// Walk `words` and record every `Input` variable that carries a `Location`.
    ///
    /// A module that does not parse yields an empty set, which answers
    /// [`InputWidth::Absent`] for every location. That is the wrong direction
    /// for a caller that treats `Absent` as permission, so the parse failure is
    /// carried instead: see [`Self::at`].
    pub fn from_spirv(words: &[u32]) -> Self {
        let Some(bound) = words.get(3).map(|b| *b as usize) else {
            return Self::unparsable();
        };
        if words.len() < HEADER_WORDS || bound == 0 {
            return Self::unparsable();
        }

        // Indexed by SPIR-V result id, which is why `bound` is read first.
        let mut location = vec![None; bound];
        let mut input_variable_type = vec![None; bound];
        let mut pointer_pointee = vec![None; bound];
        let mut components = vec![None; bound];

        let mut i = HEADER_WORDS;
        while i < words.len() {
            let word0 = words[i];
            let word_count = (word0 >> 16) as usize;
            let opcode = (word0 & 0xffff) as u16;
            if word_count == 0 || i + word_count > words.len() {
                return Self::unparsable();
            }
            match opcode {
                OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_LOCATION => {
                    let id = words[i + 1] as usize;
                    if id < bound {
                        location[id] = Some(words[i + 3]);
                    }
                }
                // `OpVariable <result type> <result id> <storage class>`. Only
                // the `Input` class is recorded, so a vertex output at the same
                // Location cannot be mistaken for the attribute.
                OP_VARIABLE if word_count >= 4 && words[i + 3] == STORAGE_CLASS_INPUT => {
                    let id = words[i + 2] as usize;
                    if id < bound {
                        input_variable_type[id] = Some(words[i + 1] as usize);
                    }
                }
                // `OpTypePointer <result id> <storage class> <type>`.
                OP_TYPE_POINTER if word_count >= 4 => {
                    let id = words[i + 1] as usize;
                    if id < bound {
                        pointer_pointee[id] = Some(words[i + 3] as usize);
                    }
                }
                // `OpTypeVector <result id> <component type> <component count>`.
                OP_TYPE_VECTOR if word_count >= 4 => {
                    let id = words[i + 1] as usize;
                    if id < bound {
                        components[id] = Some(words[i + 3]);
                    }
                }
                // A scalar input is one component. `OpTypeInt` and
                // `OpTypeFloat` both name their result id first.
                OP_TYPE_INT | OP_TYPE_FLOAT if word_count >= 3 => {
                    let id = words[i + 1] as usize;
                    if id < bound {
                        components[id] = Some(1);
                    }
                }
                _ => {}
            }
            i += word_count;
        }

        let mut declared: Vec<(u32, InputWidth)> = (0..bound)
            .filter_map(|id| {
                let loc = location[id]?;
                let pointer = input_variable_type[id]?;
                let width = pointer_pointee
                    .get(pointer)
                    .copied()
                    .flatten()
                    .and_then(|pointee| components.get(pointee).copied().flatten())
                    .map_or(InputWidth::Unreadable, InputWidth::Components);
                Some((loc, width))
            })
            .collect();
        // Two `Input` variables at one Location is not something a well-formed
        // module contains, and if one arrives the narrower answer is the unsafe
        // one — so the widest declared read wins the tie, and `Unreadable`
        // beats every number.
        declared.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| rank(b.1).cmp(&rank(a.1))));
        declared.dedup_by_key(|(loc, _)| *loc);
        Self { declared }
    }

    /// The set that answers [`InputWidth::Unreadable`] for every location.
    fn unparsable() -> Self {
        Self {
            // A location that is present but unreadable cannot be enumerated,
            // so the marker is an empty set plus the sentinel below.
            declared: vec![(u32::MAX, InputWidth::Unreadable)],
        }
    }

    /// Whether the module failed to parse at all, in which case no location has
    /// a trustworthy answer.
    fn unreadable_module(&self) -> bool {
        self.declared == [(u32::MAX, InputWidth::Unreadable)]
    }

    /// What the shader declares at `location`.
    pub fn at(&self, location: u32) -> InputWidth {
        if self.unreadable_module() {
            return InputWidth::Unreadable;
        }
        match self.declared.binary_search_by_key(&location, |(l, _)| *l) {
            Ok(i) => self.declared[i].1,
            Err(_) => InputWidth::Absent,
        }
    }
}

/// Ordering for the widest-wins tie-break. `Unreadable` outranks every count.
fn rank(width: InputWidth) -> u32 {
    match width {
        InputWidth::Components(n) => n,
        InputWidth::Absent => 0,
        InputWidth::Unreadable => u32::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a module from instruction word-lists, with `bound` ids.
    fn module(bound: u32, instructions: &[Vec<u32>]) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, bound, 0];
        for instruction in instructions {
            let count = u32::try_from(instruction.len()).unwrap();
            words.push((count << 16) | instruction[0]);
            words.extend_from_slice(&instruction[1..]);
        }
        words
    }

    fn decorate_location(id: u32, loc: u32) -> Vec<u32> {
        vec![u32::from(OP_DECORATE), id, DECORATION_LOCATION, loc]
    }
    fn type_float(id: u32) -> Vec<u32> {
        vec![u32::from(OP_TYPE_FLOAT), id, 32]
    }
    fn type_vector(id: u32, component: u32, count: u32) -> Vec<u32> {
        vec![u32::from(OP_TYPE_VECTOR), id, component, count]
    }
    fn type_pointer(id: u32, class: u32, pointee: u32) -> Vec<u32> {
        vec![u32::from(OP_TYPE_POINTER), id, class, pointee]
    }
    fn variable(ptr_type: u32, id: u32, class: u32) -> Vec<u32> {
        vec![u32::from(OP_VARIABLE), ptr_type, id, class]
    }

    /// A `float3` input at Location 0 reads three components — the answer that
    /// licenses the widening fallback.
    #[test]
    fn a_three_component_input_reads_three() {
        let words = module(
            10,
            &[
                decorate_location(5, 0),
                type_float(1),
                type_vector(2, 1, 3),
                type_pointer(3, STORAGE_CLASS_INPUT, 2),
                variable(3, 5, STORAGE_CLASS_INPUT),
            ],
        );
        let widths = VertexInputWidths::from_spirv(&words);
        assert_eq!(widths.at(0), InputWidth::Components(3));
    }

    /// A `float4` input reads four, which is the case the fallback may not
    /// serve: Vulkan would hand it a fourth component from the vertex buffer
    /// where the guest's own format defaults it to 1.0.
    #[test]
    fn a_four_component_input_reads_four() {
        let words = module(
            10,
            &[
                decorate_location(5, 2),
                type_float(1),
                type_vector(2, 1, 4),
                type_pointer(3, STORAGE_CLASS_INPUT, 2),
                variable(3, 5, STORAGE_CLASS_INPUT),
            ],
        );
        assert_eq!(
            VertexInputWidths::from_spirv(&words).at(2),
            InputWidth::Components(4)
        );
    }

    /// A scalar `float` input is one component, not an unreadable type.
    #[test]
    fn a_scalar_input_reads_one() {
        let words = module(
            10,
            &[
                decorate_location(5, 1),
                type_float(1),
                type_pointer(3, STORAGE_CLASS_INPUT, 1),
                variable(3, 5, STORAGE_CLASS_INPUT),
            ],
        );
        assert_eq!(
            VertexInputWidths::from_spirv(&words).at(1),
            InputWidth::Components(1)
        );
    }

    /// The reason the walk filters on the storage class. This module declares a
    /// `vec4` *output* at Location 0 and a `float3` input at Location 1; a walk
    /// that matched on `Location` alone would report four components at 0 and
    /// refuse a widening that is perfectly safe.
    #[test]
    fn an_output_at_the_same_location_is_not_the_input() {
        const STORAGE_CLASS_OUTPUT: u32 = 3;
        let words = module(
            12,
            &[
                decorate_location(8, 0),
                decorate_location(9, 1),
                type_float(1),
                type_vector(2, 1, 4),
                type_vector(3, 1, 3),
                type_pointer(4, STORAGE_CLASS_OUTPUT, 2),
                type_pointer(5, STORAGE_CLASS_INPUT, 3),
                variable(4, 8, STORAGE_CLASS_OUTPUT),
                variable(5, 9, STORAGE_CLASS_INPUT),
            ],
        );
        let widths = VertexInputWidths::from_spirv(&words);
        assert_eq!(widths.at(0), InputWidth::Absent, "the vec4 is an output");
        assert_eq!(widths.at(1), InputWidth::Components(3));
    }

    /// A Location no input declares is `Absent`, not zero components — the
    /// vertex descriptor routinely describes attributes the shader ignores.
    #[test]
    fn an_undeclared_location_is_absent() {
        let words = module(
            10,
            &[
                decorate_location(5, 0),
                type_float(1),
                type_vector(2, 1, 3),
                type_pointer(3, STORAGE_CLASS_INPUT, 2),
                variable(3, 5, STORAGE_CLASS_INPUT),
            ],
        );
        assert_eq!(
            VertexInputWidths::from_spirv(&words).at(7),
            InputWidth::Absent
        );
    }

    /// A type the walk cannot reduce to a component count answers `Unreadable`
    /// and not `Absent`, because a caller treating `Absent` as permission would
    /// widen under a shader that might read the fourth component.
    #[test]
    fn a_type_that_is_neither_scalar_nor_vector_is_unreadable() {
        const OP_TYPE_MATRIX: u16 = 24;
        let words = module(
            10,
            &[
                decorate_location(5, 0),
                type_float(1),
                type_vector(2, 1, 3),
                vec![u32::from(OP_TYPE_MATRIX), 6, 2, 3],
                type_pointer(3, STORAGE_CLASS_INPUT, 6),
                variable(3, 5, STORAGE_CLASS_INPUT),
            ],
        );
        assert_eq!(
            VertexInputWidths::from_spirv(&words).at(0),
            InputWidth::Unreadable
        );
    }

    /// Every location of a module that does not parse is `Unreadable`. An empty
    /// answer would read as `Absent` everywhere, which is the permissive
    /// direction.
    #[test]
    fn a_module_that_does_not_parse_is_unreadable_everywhere() {
        for words in [
            vec![],
            vec![0x0723_0203, 0x0001_0000, 0, 0, 0],
            // A word count of zero would not advance the walk.
            {
                let mut w = vec![0x0723_0203, 0x0001_0000, 0, 8, 0];
                w.push(u32::from(OP_DECORATE));
                w
            },
            // An instruction claiming more words than the module holds.
            {
                let mut w = vec![0x0723_0203, 0x0001_0000, 0, 8, 0];
                w.push((99u32 << 16) | u32::from(OP_DECORATE));
                w.push(1);
                w
            },
        ] {
            let widths = VertexInputWidths::from_spirv(&words);
            assert_eq!(widths.at(0), InputWidth::Unreadable, "{words:?}");
            assert_eq!(widths.at(5), InputWidth::Unreadable, "{words:?}");
        }
    }

    /// The walk answers about several locations from one pass, which is why it
    /// is a set rather than a per-attribute query.
    #[test]
    fn one_walk_answers_every_location() {
        let words = module(
            16,
            &[
                decorate_location(10, 0),
                decorate_location(11, 1),
                decorate_location(12, 4),
                type_float(1),
                type_vector(2, 1, 2),
                type_vector(3, 1, 3),
                type_vector(4, 1, 4),
                type_pointer(5, STORAGE_CLASS_INPUT, 2),
                type_pointer(6, STORAGE_CLASS_INPUT, 3),
                type_pointer(7, STORAGE_CLASS_INPUT, 4),
                variable(5, 10, STORAGE_CLASS_INPUT),
                variable(6, 11, STORAGE_CLASS_INPUT),
                variable(7, 12, STORAGE_CLASS_INPUT),
            ],
        );
        let widths = VertexInputWidths::from_spirv(&words);
        assert_eq!(widths.at(0), InputWidth::Components(2));
        assert_eq!(widths.at(1), InputWidth::Components(3));
        assert_eq!(widths.at(2), InputWidth::Absent);
        assert_eq!(widths.at(4), InputWidth::Components(4));
    }

    /// Two inputs at one Location is malformed; the widest wins so the tie
    /// cannot license a substitution the narrower reading forbids.
    #[test]
    fn a_duplicated_location_answers_with_its_widest_reader() {
        let words = module(
            16,
            &[
                decorate_location(10, 3),
                decorate_location(11, 3),
                type_float(1),
                type_vector(2, 1, 2),
                type_vector(3, 1, 4),
                type_pointer(5, STORAGE_CLASS_INPUT, 2),
                type_pointer(6, STORAGE_CLASS_INPUT, 3),
                variable(5, 10, STORAGE_CLASS_INPUT),
                variable(6, 11, STORAGE_CLASS_INPUT),
            ],
        );
        assert_eq!(
            VertexInputWidths::from_spirv(&words).at(3),
            InputWidth::Components(4)
        );
    }
}
