//! Backend-independent identities for renderable guest resources.

use reims_vgpu_protocol::TexelLayout;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
/// Protocol-derived render-target identity (resource state, not content hash).
pub enum TargetIdentity {
    Surface {
        id: u32,
        width: u32,
        height: u32,
        generation: u64,
        format: TexelLayout,
    },
    Texture {
        ref_: u32,
        width: u32,
        height: u32,
        generation: u64,
        stencil: bool,
    },
    Gva {
        gva: u64,
        width: u32,
        height: u32,
        generation: u64,
        format: TexelLayout,
    },
    Anonymous {
        slot: u64,
    },
}

impl Default for TargetIdentity {
    fn default() -> Self {
        Self::Anonymous { slot: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// First semantic field on which two target keys disagree.
pub enum TargetKeyDivergence {
    Absent,
    Namespace,
    Geometry,
    Generation,
    Other,
}

impl TargetKeyDivergence {
    pub fn label(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Namespace => "namespace",
            Self::Geometry => "geometry",
            Self::Generation => "generation",
            Self::Other => "other",
        }
    }
}

impl TargetIdentity {
    /// Geometry carried by identities that name guest-visible storage.
    /// Anonymous identities key backend-private storage whose geometry is
    /// supplied by the request that creates it.
    pub fn geometry(&self) -> Option<(u32, u32)> {
        match self {
            Self::Surface { width, height, .. }
            | Self::Texture { width, height, .. }
            | Self::Gva { width, height, .. } => Some((*width, *height)),
            Self::Anonymous { .. } => None,
        }
    }

    pub fn width(&self) -> u32 {
        self.geometry().map_or(0, |(width, _)| width)
    }
    pub fn height(&self) -> u32 {
        self.geometry().map_or(0, |(_, height)| height)
    }
    pub fn generation(&self) -> u64 {
        match self {
            Self::Surface { generation, .. }
            | Self::Texture { generation, .. }
            | Self::Gva { generation, .. } => *generation,
            Self::Anonymous { .. } => 0,
        }
    }
    pub fn namespaced_id(&self) -> (u8, u64) {
        match self {
            Self::Surface { id, .. } => (0, u64::from(*id)),
            Self::Texture { ref_, .. } => (1, u64::from(*ref_)),
            Self::Gva { gva, .. } => (2, *gva),
            Self::Anonymous { slot } => (3, *slot),
        }
    }
    pub fn diverges_from(&self, held: &Self) -> TargetKeyDivergence {
        if self.namespaced_id() != held.namespaced_id() {
            return TargetKeyDivergence::Namespace;
        }
        if (self.width(), self.height()) != (held.width(), held.height()) {
            return TargetKeyDivergence::Geometry;
        }
        if self.with_generation(held.generation()) == *held {
            TargetKeyDivergence::Generation
        } else {
            TargetKeyDivergence::Other
        }
    }
    pub fn with_generation(&self, generation: u64) -> Self {
        let mut next = self.clone();
        match &mut next {
            Self::Surface {
                generation: value, ..
            }
            | Self::Texture {
                generation: value, ..
            }
            | Self::Gva {
                generation: value, ..
            } => *value = generation,
            Self::Anonymous { .. } => {}
        }
        next
    }
    pub fn aliases(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Surface { id: a, .. }, Self::Surface { id: b, .. }) => a == b,
            (Self::Gva { gva: a, .. }, Self::Gva { gva: b, .. }) => a == b,
            (Self::Texture { ref_: a, .. }, Self::Texture { ref_: b, .. }) => a == b,
            (Self::Anonymous { slot: a }, Self::Anonymous { slot: b }) => a == b,
            _ => false,
        }
    }
    pub fn resident_layout(&self) -> TexelLayout {
        match self {
            Self::Surface { format, .. } | Self::Gva { format, .. } => *format,
            Self::Texture { .. } | Self::Anonymous { .. } => TexelLayout::Rgba8,
        }
    }
    pub fn is_bgra(&self) -> bool {
        self.resident_layout() == TexelLayout::Bgra8
    }
}

#[cfg(test)]
mod tests {
    use super::{TargetIdentity, TargetKeyDivergence};
    use reims_vgpu_protocol::TexelLayout;

    fn surface(generation: u64, width: u32) -> TargetIdentity {
        TargetIdentity::Surface {
            id: 7,
            width,
            height: 32,
            generation,
            format: TexelLayout::Bgra8,
        }
    }

    #[test]
    fn divergence_distinguishes_geometry_generation_and_namespace() {
        let held = surface(2, 64);
        assert_eq!(
            surface(2, 65).diverges_from(&held),
            TargetKeyDivergence::Geometry
        );
        assert_eq!(
            surface(3, 64).diverges_from(&held),
            TargetKeyDivergence::Generation
        );
        assert_eq!(
            TargetIdentity::Gva {
                gva: 7,
                width: 64,
                height: 32,
                generation: 2,
                format: TexelLayout::Bgra8,
            }
            .diverges_from(&held),
            TargetKeyDivergence::Namespace
        );
    }

    #[test]
    fn aliases_ignore_versioned_shape_but_not_namespace() {
        let a = surface(2, 64);
        assert!(a.aliases(&surface(9, 128)));
        assert!(!a.aliases(&TargetIdentity::Texture {
            ref_: 7,
            width: 64,
            height: 32,
            generation: 2,
            stencil: false,
        }));
    }

    #[test]
    fn resident_layout_is_semantic() {
        assert!(surface(0, 64).is_bgra());
        assert_eq!(
            TargetIdentity::Texture {
                ref_: 1,
                width: 8,
                height: 8,
                generation: 0,
                stencil: false,
            }
            .resident_layout(),
            TexelLayout::Rgba8
        );
    }
}
