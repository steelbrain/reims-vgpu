//! The arguments of one direct draw.

/// Bit `n` set = this device can execute `MTLPrimitiveType` `n`.
///
/// # This is an advertisement, and the guest reads it as permission
///
/// The guest driver answers `-[MTLDevice supportsPrimitiveType:]` by testing
/// bit `type` of the device-info value for
/// `crate::model::DEVICE_INFO_KEY_PRIMITIVE_TYPE_MASK`, for any `type <= 8`,
/// and falls back to `type < 5` when the key is absent. So the number this
/// device publishes decides which primitive types the guest is permitted to
/// build a draw out of — and a bit set for a type no backend can translate is a
/// draw this device refuses *after* the guest has committed to it.
///
/// The capture that table came from carried `1023`: bits 0..=9, authorising the
/// four non-public types 5..=8 on top of the public enum. Both backends refuse
/// those by name — `translate::raster::primitive_topology` answers
/// `UnknownPrimitiveType` and `backend::metal::mtl_enum::primitive_type` answers
/// `None` — so every one of those bits was a promise this device cannot keep.
/// Narrowing to what it can execute is the rule
/// `crate::model::device_info_caps` already applies to the GPU-dependent keys:
/// answering higher than the host can execute does not degrade gracefully.
///
/// Widening it again needs the *meaning* of 5..=8 first. They are not in the
/// public `MTLPrimitiveType` enum and nothing here has decoded one, so setting a
/// bit for one would be a number chosen to match a capture rather than a
/// contract. Each backend's translator carries the test that holds this constant
/// to the arms that actually exist.
pub const EXECUTABLE_PRIMITIVE_TYPES: u32 = 0b1_1111;

/// Whether `mtl` is a primitive type this device advertises and can execute.
#[inline]
pub const fn primitive_type_executable(mtl: u32) -> bool {
    mtl < u32::BITS && (EXECUTABLE_PRIMITIVE_TYPES >> mtl) & 1 == 1
}

/// What a `drawPrimitives` / `drawIndexedPrimitives` record asks for, as one
/// value.
///
/// Its own type for the same reason [`super::extent::Extent3`] is: the hazard is
/// at the call boundary, not at construction. These five were decoded into a
/// struct and then destructured back into loose `u32`s to cross two of them —
/// `draw::mrt_draw_request` took `(vertex_count, instance_count,
/// primitive_type, first_vertex, base_instance)` and
/// `backend::metal::render::render_core_mrt`, one call further down the same
/// draw, took the same five as `(vertex_count, first_vertex, instance_count,
/// base_instance, primitive_type)`. Two orders, both positional, both all-`u32`
/// or all-`usize`, so every one of the 120 permutations compiled at each site
/// and the two sites did not even agree with each other.
///
/// A transposition here does not fail: it draws a valid primitive of the wrong
/// shape, or the right vertices of the wrong instance, which nothing downstream
/// can distinguish from the draw the guest asked for.
///
/// What this does not close: the fields are still five `u32`s, so a *builder*
/// that names them wrongly compiles. That hazard is at construction, where the
/// field names are written out and a reader can check them against the decoder,
/// and it is not the one that has bitten.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DrawArgs {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub primitive_type: u32,
    pub first_vertex: u32,
    /// Metal `baseInstance` / Vulkan `firstInstance`.
    pub base_instance: u32,
}

/// The indirect draw argument blocks, as Metal lays them out in a buffer.
///
/// `drawPrimitives:indirectBuffer:indirectBufferOffset:` and its indexed
/// sibling put the counts in guest memory instead of in the record. What sits
/// at that offset is one of two structs from `MTLRenderCommandEncoder.h`, all
/// 32-bit fields in declaration order:
///
/// ```text
/// MTLDrawPrimitivesIndirectArguments        { vertexCount, instanceCount, vertexStart, baseInstance }
/// MTLDrawIndexedPrimitivesIndirectArguments { indexCount, instanceCount, indexStart, baseVertex, baseInstance }
/// ```
///
/// `baseVertex` is the one signed field, for the same reason
/// `crate::runtime::draw::IndexedDrawInfo::base_vertex` is: read as
/// unsigned, a negative one becomes a huge index rather than an error.
///
/// Nothing here is a `#[repr(C)]` view over guest bytes — the block is loaded
/// field by field out of a byte window. That is deliberate:
/// `reims-vgpu-wire`'s invariant 4 forbids a wire struct holding a field an
/// out-of-range guest value would make invalid, and five little-endian loads
/// cost nothing.
pub mod indirect {
    use crate::endian::ld32;

    /// Bytes `MTLDrawPrimitivesIndirectArguments` occupies: four `uint32_t`.
    pub const UNINDEXED_LEN: usize = 16;
    /// Bytes `MTLDrawIndexedPrimitivesIndirectArguments` occupies: five 32-bit
    /// fields, the fourth of them signed.
    pub const INDEXED_LEN: usize = 20;

    /// What an unindexed indirect draw's argument block says, as the same
    /// [`super::DrawArgs`] a direct record decodes to.
    ///
    /// `primitive_type` is *not* in the block — Metal takes it as an argument
    /// to the selector — so it comes off the record and is passed in here.
    /// `None` when the window is short, which is the caller's cue to refuse
    /// rather than to draw a zero.
    pub fn unindexed(block: &[u8], primitive_type: u32) -> Option<super::DrawArgs> {
        if block.len() < UNINDEXED_LEN {
            return None;
        }
        Some(super::DrawArgs {
            vertex_count: ld32(block),
            instance_count: ld32(&block[4..]),
            primitive_type,
            first_vertex: ld32(&block[8..]),
            base_instance: ld32(&block[12..]),
        })
    }

    /// What an indexed indirect draw's argument block says.
    ///
    /// The [`super::DrawArgs`] half carries `indexCount` in `vertex_count`,
    /// which is what every indexed path in this crate already does — a direct
    /// `drawIndexedPrimitives` record fills the same field the same way. The
    /// two trailing values are the index-buffer half: `(index_start,
    /// base_vertex)`.
    pub fn indexed(block: &[u8], primitive_type: u32) -> Option<(super::DrawArgs, u32, i32)> {
        if block.len() < INDEXED_LEN {
            return None;
        }
        let args = super::DrawArgs {
            vertex_count: ld32(block),
            instance_count: ld32(&block[4..]),
            primitive_type,
            // `indexStart` indexes the index buffer, not the vertex buffer, so
            // it is *not* `first_vertex`. It is returned beside the args and
            // applied to the index-buffer offset by the caller.
            first_vertex: 0,
            base_instance: ld32(&block[16..]),
        };
        Some((args, ld32(&block[8..]), ld32(&block[12..]) as i32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both indirect argument blocks read their fields in Metal's declared
    /// order, and a short window refuses instead of reading zeros.
    ///
    /// Field order is the whole hazard, exactly as it is for [`DrawArgs`]:
    /// every field is 32 bits, so a transposition compiles and draws the right
    /// vertices of the wrong instance. Each field gets a distinct value so no
    /// pair can swap and still read back correct.
    #[test]
    fn an_indirect_argument_block_reads_metals_field_order() {
        let mut block = Vec::new();
        for w in [11u32, 22, 33, 44, 55] {
            block.extend_from_slice(&w.to_le_bytes());
        }

        let un = indirect::unindexed(&block, 3).expect("16 bytes is a full block");
        assert_eq!(
            un,
            DrawArgs {
                vertex_count: 11,
                instance_count: 22,
                primitive_type: 3,
                first_vertex: 33,
                base_instance: 44,
            },
            "MTLDrawPrimitivesIndirectArguments order"
        );

        let (args, index_start, base_vertex) =
            indirect::indexed(&block, 4).expect("20 bytes is a full block");
        assert_eq!(
            args,
            DrawArgs {
                vertex_count: 11,
                instance_count: 22,
                primitive_type: 4,
                // `indexStart` is returned separately rather than here: it
                // offsets the index buffer, and putting it in `first_vertex`
                // would shift the vertex fetch instead.
                first_vertex: 0,
                base_instance: 55,
            },
            "MTLDrawIndexedPrimitivesIndirectArguments order"
        );
        assert_eq!(index_start, 33);
        assert_eq!(base_vertex, 44);

        // `baseVertex` is signed. Read as unsigned it becomes a vertex index
        // near four billion, which fetches out of every buffer rather than
        // failing.
        let mut negative = block.clone();
        negative[12..16].copy_from_slice(&(-7i32).to_le_bytes());
        assert_eq!(indirect::indexed(&negative, 3).unwrap().2, -7);

        // One byte short of each block refuses. Reading a partial block as
        // zeros is a draw of nothing that reports success.
        assert!(indirect::unindexed(&block[..indirect::UNINDEXED_LEN - 1], 3).is_none());
        assert!(indirect::indexed(&block[..indirect::INDEXED_LEN - 1], 3).is_none());
    }

    /// The mask names the public `MTLPrimitiveType` enum and nothing above it.
    ///
    /// Stated here as well as in each backend's translator test because this is
    /// the value that leaves the device: a bit added here without an arm behind
    /// it is a guest draw refused, and the backend tests cannot both run on one
    /// host.
    #[test]
    fn the_advertised_primitive_types_stop_at_the_public_enum() {
        for mtl in 0..5 {
            assert!(primitive_type_executable(mtl), "public type {mtl}");
        }
        for mtl in 5..=8 {
            assert!(
                !primitive_type_executable(mtl),
                "type {mtl} is not in the public enum and no arm decodes it"
            );
        }
        assert!(!primitive_type_executable(u32::MAX), "no shift overflow");
    }
}
