//! How many components a translated vertex shader actually reads at one
//! stage-in `Location`.
//!
//! # Why anything needs to ask
//!
//! Vulkan requires only a subset of formats to be usable as vertex attributes,
//! and every three-component 8- and 16-bit format is outside it. A device that
//! declines one is handed its mandatory four-component sibling instead, which
//! [`crate::translate::support`] calls the widening fallback.
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
//! # Read from reflection
//!
//! metal2vulkan emits each stage-in location and AIR type in
//! `ShaderReflection::vertex_attributes`. The translated shader variant keeps
//! this projection beside its executable words, so pipeline construction does
//! not reconstruct a second answer from SPIR-V. A missing or unfamiliar type
//! is `Unreadable`, which refuses a format widening rather than guessing.
//!

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

/// Every stage-in `Location` a module declares, and how wide each one is.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexInputWidths {
    /// `(location, width)` sorted by location. A `Vec` rather than a map
    /// because a vertex shader declares a handful of inputs and the lookup is
    /// a binary search over single digits.
    declared: Vec<(u32, InputWidth)>,
}

impl VertexInputWidths {
    /// Conservative answer for a native module with no reflected source
    /// interface, such as an executor test fixture.
    pub fn unknown() -> Self {
        Self::unparsable()
    }

    /// Project the translator's vertex interface into the width answer needed
    /// by format substitution.
    pub fn from_reflection(attributes: &[metal2vulkan::reflect::VertexAttribute]) -> Self {
        let mut declared = attributes
            .iter()
            .map(|attribute| {
                let width = attribute
                    .type_name
                    .as_deref()
                    .and_then(air_scalar_or_vector_width)
                    .map_or(InputWidth::Unreadable, InputWidth::Components);
                (attribute.location, width)
            })
            .collect::<Vec<_>>();
        declared.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| rank(b.1).cmp(&rank(a.1))));
        declared.dedup_by_key(|(location, _)| *location);
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

    /// Whether no source reflection was available, in which case no location
    /// has a trustworthy answer.
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

fn air_scalar_or_vector_width(type_name: &str) -> Option<u32> {
    const SCALARS: &[&str] = &[
        "bool", "char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "half",
        "float", "double",
    ];
    let name = type_name.strip_prefix("packed_").unwrap_or(type_name);
    if SCALARS.contains(&name) {
        return Some(1);
    }
    let (scalar, width) = name.split_at(name.len().checked_sub(1)?);
    let width = width.parse::<u32>().ok()?;
    (SCALARS.contains(&scalar) && (2..=4).contains(&width)).then_some(width)
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

    #[test]
    fn reflected_air_types_supply_vertex_component_widths() {
        use metal2vulkan::reflect::VertexAttribute;
        let widths = VertexInputWidths::from_reflection(&[
            VertexAttribute {
                location: 3,
                type_name: Some("float3".into()),
                name: None,
            },
            VertexAttribute {
                location: 7,
                type_name: Some("packed_uchar4".into()),
                name: None,
            },
        ]);
        assert_eq!(widths.at(3), InputWidth::Components(3));
        assert_eq!(widths.at(7), InputWidth::Components(4));
        assert_eq!(widths.at(9), InputWidth::Absent);
    }

    #[test]
    fn missing_or_unfamiliar_reflected_types_cannot_authorize_widening() {
        use metal2vulkan::reflect::VertexAttribute;
        let widths = VertexInputWidths::from_reflection(&[
            VertexAttribute {
                location: 1,
                type_name: None,
                name: None,
            },
            VertexAttribute {
                location: 2,
                type_name: Some("float4x4".into()),
                name: None,
            },
        ]);
        assert_eq!(widths.at(1), InputWidth::Unreadable);
        assert_eq!(widths.at(2), InputWidth::Unreadable);
    }
}
