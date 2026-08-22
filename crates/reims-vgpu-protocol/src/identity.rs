//! Non-interchangeable identities and quantities used by the semantic core.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

macro_rules! scalar_newtype {
    ($(#[$meta:meta])* $name:ident, $inner:ty) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name($inner);

        impl $name {
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl From<$inner> for $name {
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $inner {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

scalar_newtype!(TaskId, u32);
scalar_newtype!(ResourceNamespaceId, u32);
scalar_newtype!(MappingId, u32);
scalar_newtype!(
    /// Mapper-service lookup identity carried by mapper-backed IOSurface objects.
    ///
    /// This namespace is independent of GPU page-table mappings and registered
    /// surface backings. The wire producer and consumer both use all 64 bits; a
    /// low live value does not license narrowing the identity.
    MapperSurfaceRef,
    u64
);
scalar_newtype!(SurfaceId, u32);
scalar_newtype!(
    /// Task-visible surface/host-representation identity obtained by resolving
    /// a mapper-service reference.
    ///
    /// This is deliberately not [`MapperSurfaceRef`], a page-table
    /// [`MappingId`], or a canonical [`SurfaceBackingId`]. Adapters may still
    /// project it into a legacy integer-keyed table, but the relation is an
    /// explicit edge rather than numeric equivalence.
    MapperResolvedSurfaceId,
    u32
);
scalar_newtype!(SurfaceBackingId, u64);
scalar_newtype!(StorageId, u64);
scalar_newtype!(GuestVirtualAddress, u64);
scalar_newtype!(GuestPhysicalAddress, u64);
scalar_newtype!(ByteOffset, u64);
scalar_newtype!(ByteLength, u64);
scalar_newtype!(SubmissionId, u64);
scalar_newtype!(
    /// Executor-prepared shader identity.
    ///
    /// The semantic command carries this identity and its decoded interface;
    /// backend-native module bytes remain in the executor that prepared it.
    PreparedShaderId,
    u64
);
scalar_newtype!(BackingGeneration, u64);
scalar_newtype!(ContentVersion, u64);
scalar_newtype!(PlaneIndex, u32);
scalar_newtype!(TextureRotation, u8);

impl fmt::LowerHex for GuestVirtualAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::LowerHex for GuestPhysicalAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

/// A task-local slot in the kernel object-table namespace.
#[repr(transparent)]
pub struct ObjectTableRef<T> {
    value: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> ObjectTableRef<T> {
    pub const fn new(value: u32) -> Self {
        Self {
            value,
            marker: PhantomData,
        }
    }

    pub const fn get(self) -> u32 {
        self.value
    }
}

impl<T> Clone for ObjectTableRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ObjectTableRef<T> {}

impl<T> fmt::Debug for ObjectTableRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ObjectTableRef").field(&self.value).finish()
    }
}

impl<T> PartialEq for ObjectTableRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> Eq for ObjectTableRef<T> {}

impl<T> PartialOrd for ObjectTableRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ObjectTableRef<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T> Hash for ObjectTableRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

/// A slot in one serializer object's task-local, family-specific namespace.
///
/// It intentionally has no conversion to or from [`ObjectTableRef`]. The wire
/// integers can be equal while naming unrelated lifetimes.
///
/// ```compile_fail
/// use reims_vgpu_protocol::{ObjectTableRef, SamplerObject, SerializerRef};
/// fn delete_sampler(_: SerializerRef<SamplerObject>) {}
/// delete_sampler(ObjectTableRef::<SamplerObject>::new(7));
/// ```
#[repr(transparent)]
pub struct SerializerRef<T> {
    value: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> SerializerRef<T> {
    pub const fn new(value: u32) -> Self {
        Self {
            value,
            marker: PhantomData,
        }
    }

    pub const fn get(self) -> u32 {
        self.value
    }
}

impl<T> Clone for SerializerRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SerializerRef<T> {}

impl<T> fmt::Debug for SerializerRef<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SerializerRef").field(&self.value).finish()
    }
}

impl<T> PartialEq for SerializerRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> Eq for SerializerRef<T> {}

impl<T> PartialOrd for SerializerRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for SerializerRef<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T> Hash for SerializerRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

/// A generational internal identity for one typed resource lifetime.
pub struct ResourceId<T> {
    index: u32,
    generation: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> ResourceId<T> {
    pub const fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            marker: PhantomData,
        }
    }

    pub const fn index(self) -> u32 {
        self.index
    }

    pub const fn generation(self) -> u32 {
        self.generation
    }
}

impl<T> Clone for ResourceId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ResourceId<T> {}

impl<T> fmt::Debug for ResourceId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResourceId")
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

impl<T> PartialEq for ResourceId<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.index, self.generation) == (other.index, other.generation)
    }
}

impl<T> Eq for ResourceId<T> {}

impl<T> PartialOrd for ResourceId<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for ResourceId<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (self.index, self.generation).cmp(&(other.index, other.generation))
    }
}

impl<T> Hash for ResourceId<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    enum Buffer {}
    enum Texture {}

    #[test]
    fn typed_namespaces_do_not_share_a_runtime_representation_owner() {
        let buffer = ObjectTableRef::<Buffer>::new(7);
        let texture = ObjectTableRef::<Texture>::new(7);
        assert_eq!(buffer.get(), texture.get());

        let first = ResourceId::<Buffer>::new(3, 4);
        let reused = ResourceId::<Buffer>::new(3, 5);
        assert_ne!(first, reused);
    }
}
