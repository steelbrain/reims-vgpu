//! `ARGUMENT_LAYOUT`: where an argument buffer's arguments lie, as the platform's
//! `MTLArgumentEncoder` lays them out, written into a block of the guest's.
//!
//! The guest's argument encoder writes the arguments itself: a buffer's GPU address plus its
//! offset, every other object's `gpuResourceID`, 8 bytes each, little endian, and constants as
//! the guest gives them. It needs only the layout, which the host GPU's compiler decides for a
//! function's argument buffer, and which Metal's rules decide for argument descriptors: the
//! platform makes the encoder Metal makes of the source, reads its layout, and keeps nothing.
//!
//! The answer is a tree flattened into records ([`RgfxArgumentLayoutReply`]): layout 0 is the
//! argument buffer's, each layout's members are consecutive, and a member that points to a nested
//! argument buffer, or is a struct, names its layout by its index. A layout's members follow every
//! member of the layouts before it, so the guest reads the records in one pass.

use reims_graphics::wire::{
    RGFX_ARGUMENT_ACCELERATION_STRUCTURE, RGFX_ARGUMENT_BUFFER, RGFX_ARGUMENT_COMPUTE_PIPELINE,
    RGFX_ARGUMENT_CONSTANT, RGFX_ARGUMENT_DEPTH_STENCIL, RGFX_ARGUMENT_INDIRECT_COMMAND_BUFFER,
    RGFX_ARGUMENT_INTERSECTION_FUNCTION_TABLE, RGFX_ARGUMENT_NO_LAYOUT,
    RGFX_ARGUMENT_RENDER_PIPELINE, RGFX_ARGUMENT_SAMPLER, RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
    RGFX_ARGUMENT_SOURCE_FUNCTION, RGFX_ARGUMENT_STRUCT, RGFX_ARGUMENT_TEXTURE,
    RGFX_ARGUMENT_VISIBLE_FUNCTION_TABLE, RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_NOT_FOUND,
    RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RGFX_STATUS_TOO_SMALL, RgfxArgumentDescriptor,
    RgfxArgumentLayoutQuery, RgfxArgumentLayoutRecord, RgfxArgumentLayoutReply,
    RgfxArgumentMemberRecord, RgfxMsgHeader,
};
use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::types::{ArgumentKind, ArgumentLayout};
use reims_vgpu_core::platform::values::ArgumentDesc;
use reims_vgpu_core::raise::RaiseResult;
use zerocopy::{FromBytes, IntoBytes};

use super::{GraphicsDecoder, ObjHandle, constants_tail, read, specialised, str_tail};

/// The flattened answer: its layouts, then its members ([`flatten`]).
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Flat {
    pub(crate) layouts: Vec<RgfxArgumentLayoutRecord>,
    pub(crate) members: Vec<RgfxArgumentMemberRecord>,
}

impl Flat {
    /// The answer's bytes: the reply head, the layouts, the members.
    pub(crate) fn bytes(&self) -> Option<Vec<u8>> {
        let layout_count = u32::try_from(self.layouts.len()).ok()?;
        let member_count = u32::try_from(self.members.len()).ok()?;
        let needed = size_of::<RgfxArgumentLayoutReply>()
            + self.layouts.len() * size_of::<RgfxArgumentLayoutRecord>()
            + self.members.len() * size_of::<RgfxArgumentMemberRecord>();
        let head = RgfxArgumentLayoutReply {
            needed: needed as u64,
            layout_count,
            member_count,
        };
        let mut bytes = Vec::with_capacity(needed);
        bytes.extend_from_slice(head.as_bytes());
        for l in &self.layouts {
            bytes.extend_from_slice(l.as_bytes());
        }
        for m in &self.members {
            bytes.extend_from_slice(m.as_bytes());
        }
        Some(bytes)
    }
}

/// `layout` as records: its own first, its members after the members of the layouts before it,
/// and each nested layout after its holder's. `None` past `u32` counts.
pub(crate) fn flatten(layout: &ArgumentLayout) -> Option<Flat> {
    let mut flat = Flat::default();
    push(&mut flat, layout)?;
    Some(flat)
}

/// Pushes `layout` and its members, then the layouts its members name; answers its index.
fn push(flat: &mut Flat, layout: &ArgumentLayout) -> Option<u32> {
    let index = u32::try_from(flat.layouts.len()).ok()?;
    let first = flat.members.len();
    flat.layouts.push(RgfxArgumentLayoutRecord {
        encoded_length: layout.encoded_length,
        alignment: layout.alignment,
        first_member: u32::try_from(first).ok()?,
        member_count: u32::try_from(layout.members.len()).ok()?,
    });
    for m in &layout.members {
        flat.members.push(RgfxArgumentMemberRecord {
            id: m.id,
            count: m.count,
            ids_per_element: m.ids_per_element,
            kind: kind(m.kind),
            offset: m.offset,
            stride: m.stride,
            size: m.size,
            layout: RGFX_ARGUMENT_NO_LAYOUT,
            reserved_2c: 0,
        });
    }
    for (i, m) in layout.members.iter().enumerate() {
        if let Some(nested) = &m.layout {
            flat.members[first + i].layout = push(flat, nested)?;
        }
    }
    Some(index)
}

/// The wire's `RGFX_ARGUMENT_*` of `kind`.
fn kind(kind: ArgumentKind) -> u32 {
    match kind {
        ArgumentKind::Buffer => RGFX_ARGUMENT_BUFFER,
        ArgumentKind::Texture => RGFX_ARGUMENT_TEXTURE,
        ArgumentKind::Sampler => RGFX_ARGUMENT_SAMPLER,
        ArgumentKind::Constant => RGFX_ARGUMENT_CONSTANT,
        ArgumentKind::IndirectCommandBuffer => RGFX_ARGUMENT_INDIRECT_COMMAND_BUFFER,
        ArgumentKind::ComputePipelineState => RGFX_ARGUMENT_COMPUTE_PIPELINE,
        ArgumentKind::RenderPipelineState => RGFX_ARGUMENT_RENDER_PIPELINE,
        ArgumentKind::VisibleFunctionTable => RGFX_ARGUMENT_VISIBLE_FUNCTION_TABLE,
        ArgumentKind::IntersectionFunctionTable => RGFX_ARGUMENT_INTERSECTION_FUNCTION_TABLE,
        ArgumentKind::AccelerationStructure => RGFX_ARGUMENT_ACCELERATION_STRUCTURE,
        ArgumentKind::DepthStencilState => RGFX_ARGUMENT_DEPTH_STENCIL,
        ArgumentKind::Struct => RGFX_ARGUMENT_STRUCT,
    }
}

/// The `count` descriptors at byte `offset` of `message`, or `None` where the message does not
/// cover them.
fn descriptors(message: &[u8], offset: u32, count: u32) -> Option<Vec<ArgumentDesc>> {
    let size = size_of::<RgfxArgumentDescriptor>();
    let start = offset as usize;
    let end = start.checked_add((count as usize).checked_mul(size)?)?;
    message
        .get(start..end)?
        .chunks_exact(size)
        .map(|b| {
            RgfxArgumentDescriptor::read_from_bytes(b)
                .ok()
                .map(|d| ArgumentDesc {
                    data_type: d.data_type.into(),
                    index: d.index.into(),
                    array_length: d.array_length.into(),
                    access: d.access.into(),
                    texture_type: d.texture_type.into(),
                    constant_block_alignment: d.constant_block_alignment.into(),
                })
        })
        .collect()
}

impl<P: Platform> GraphicsDecoder<P> {
    /// `ARGUMENT_LAYOUT`: the platform's layout of the source, written into the guest's block
    /// (module doc). The query's seqno (`None` for a message too short to carry one) and its
    /// `COMPLETE` status: invalid for a malformed message, a source the wire does not define or a
    /// block that is not the sender's; not found for a library the sender does not hold; refused
    /// where the platform makes no encoder or no function; too small where the answer does not fit
    /// the block, whose first 8 bytes then say how many it takes.
    pub(crate) fn argument_layout(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<(Option<u64>, u32)> {
        let seqno = read::<RgfxMsgHeader>(message).map(|h| h.seqno);
        let Some(m) = read::<RgfxArgumentLayoutQuery>(message) else {
            return Ok((seqno, RGFX_STATUS_INVALID_ARGUMENT));
        };
        if !m.reply_offset.is_multiple_of(8)
            || m.reply_length < 8
            || !self.in_own_block(m.reply_offset, m.reply_length)
        {
            return Ok((seqno, RGFX_STATUS_INVALID_ARGUMENT));
        }
        let layout = match m.source {
            RGFX_ARGUMENT_SOURCE_DESCRIPTORS => {
                let Some(arguments) = descriptors(message, m.descriptors_offset, m.count) else {
                    return Ok((seqno, RGFX_STATUS_INVALID_ARGUMENT));
                };
                device.argument_layout(&arguments)?
            }
            RGFX_ARGUMENT_SOURCE_FUNCTION => {
                let (Some(name), Some(constants)) = (
                    str_tail(message, m.name_offset, m.name_len),
                    constants_tail(message, m.constants_offset, m.constant_count),
                ) else {
                    return Ok((seqno, RGFX_STATUS_INVALID_ARGUMENT));
                };
                let Some(ObjHandle::Library(library)) = self.named(m.library_obj_id) else {
                    return Ok((seqno, RGFX_STATUS_NOT_FOUND));
                };
                let Some(function) = specialised::<P>(library, name, &constants)? else {
                    return Ok((seqno, RGFX_STATUS_REFUSED));
                };
                device.function_argument_layout(&function, m.buffer_index.into())?
            }
            _ => return Ok((seqno, RGFX_STATUS_INVALID_ARGUMENT)),
        };
        let Some(bytes) = layout.as_ref().and_then(flatten).and_then(|f| f.bytes()) else {
            return Ok((seqno, RGFX_STATUS_REFUSED));
        };
        if bytes.len() as u64 > m.reply_length {
            let written = self
                .region
                .write(m.reply_offset, &(bytes.len() as u64).to_le_bytes());
            return Ok((
                seqno,
                if written {
                    RGFX_STATUS_TOO_SMALL
                } else {
                    RGFX_STATUS_INVALID_ARGUMENT
                },
            ));
        }
        if !self.region.write(m.reply_offset, &bytes) {
            return Ok((seqno, RGFX_STATUS_INVALID_ARGUMENT));
        }
        Ok((seqno, RGFX_STATUS_OK))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_graphics::region::Region;
    use reims_graphics::wire::{
        RGFX_CMD_ARGUMENT_LAYOUT, RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN,
        RgfxFunctionConstant,
    };
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::testkit::FakePlatform;

    use super::super::{GraphicsDecoder, ObjHandle};
    use super::*;

    /// The decoder over a region with one MiB of the processes' blocks, and that block's offset.
    fn decoder() -> (
        Arc<FakePlatform>,
        Arc<Region>,
        GraphicsDecoder<FakePlatform>,
        u64,
    ) {
        let p = Arc::new(FakePlatform::new());
        let blocks = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
        let size = blocks + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let dec = GraphicsDecoder::<FakePlatform>::new(Arc::clone(&p), Arc::clone(&region));
        (p, region, dec, blocks + 0x2000)
    }

    /// A query of `source` answered into `reply_length` bytes at `reply`, with `tail` after the
    /// head: the descriptors for descriptors, the name and then the constants for a function.
    fn query(
        source: u32,
        reply: u64,
        reply_length: u64,
        library: u64,
        descriptors: &[RgfxArgumentDescriptor],
        name: &str,
        constants: &[RgfxFunctionConstant],
    ) -> Vec<u8> {
        let head = size_of::<RgfxArgumentLayoutQuery>();
        let mut tail = Vec::new();
        for d in descriptors {
            tail.extend_from_slice(d.as_bytes());
        }
        let name_offset = head + tail.len();
        tail.extend_from_slice(name.as_bytes());
        while tail.len() % 8 != 0 {
            tail.push(0);
        }
        let constants_offset = head + tail.len();
        for c in constants {
            tail.extend_from_slice(c.as_bytes());
        }
        let m = RgfxArgumentLayoutQuery {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_ARGUMENT_LAYOUT,
                size: (head + tail.len()) as u32,
                seqno: 9,
            },
            obj_id: 0x77,
            reply_offset: reply,
            reply_length,
            library_obj_id: library,
            source,
            count: descriptors.len() as u32,
            descriptors_offset: head as u32,
            buffer_index: 0,
            name_offset: name_offset as u32,
            name_len: name.len() as u32,
            constants_offset: constants_offset as u32,
            constant_count: constants.len() as u32,
        };
        let mut bytes = m.as_bytes().to_vec();
        bytes.extend_from_slice(&tail);
        bytes
    }

    fn descriptor(data_type: u32, index: u32, array_length: u32) -> RgfxArgumentDescriptor {
        RgfxArgumentDescriptor {
            data_type,
            index,
            array_length,
            texture_type: 2,
            ..RgfxArgumentDescriptor::default()
        }
    }

    /// The answer's records read back from the region.
    fn answer(
        region: &Region,
        at: u64,
    ) -> (
        RgfxArgumentLayoutReply,
        Vec<RgfxArgumentLayoutRecord>,
        Vec<RgfxArgumentMemberRecord>,
    ) {
        let head = RgfxArgumentLayoutReply::read_from_bytes(&region.read(at, 16).unwrap()).unwrap();
        let mut off = at + 16;
        let mut layouts = Vec::new();
        for _ in 0..head.layout_count {
            layouts.push(
                RgfxArgumentLayoutRecord::read_from_bytes(&region.read(off, 24).unwrap()).unwrap(),
            );
            off += 24;
        }
        let mut members = Vec::new();
        for _ in 0..head.member_count {
            members.push(
                RgfxArgumentMemberRecord::read_from_bytes(&region.read(off, 48).unwrap()).unwrap(),
            );
            off += 48;
        }
        (head, layouts, members)
    }

    /// Descriptors are laid out by the platform, in their order, and the layout is written into
    /// the guest's block: its head, its one layout and a record per descriptor; a block too small
    /// for it gets only the bytes it needs, and the status says so
    #[test]
    fn descriptors_answer_the_platforms_layout_into_the_block() {
        let (p, region, dec, reply) = decoder();
        let device = p.device();
        let descriptors = [
            descriptor(58, 0, 0),
            descriptor(60, 1, 4),
            descriptor(6, 5, 0),
        ];
        let m = query(
            RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
            reply,
            0x1000,
            0,
            &descriptors,
            "",
            &[],
        );
        assert_eq!(
            dec.argument_layout(&device, &m).unwrap(),
            (Some(9), RGFX_STATUS_OK)
        );
        let (head, layouts, members) = answer(&region, reply);
        assert_eq!(head.needed, 16 + 24 + 3 * 48);
        assert_eq!((head.layout_count, head.member_count), (1, 3));
        assert_eq!(layouts[0].encoded_length, 48);
        assert_eq!(layouts[0].alignment, 8);
        assert_eq!((layouts[0].first_member, layouts[0].member_count), (0, 3));
        let got: Vec<_> = members
            .iter()
            .map(|m| (m.id, m.count, m.offset, m.kind, m.layout))
            .collect();
        assert_eq!(
            got,
            [
                (0, 1, 0, RGFX_ARGUMENT_TEXTURE, RGFX_ARGUMENT_NO_LAYOUT),
                (1, 4, 8, RGFX_ARGUMENT_BUFFER, RGFX_ARGUMENT_NO_LAYOUT),
                (5, 1, 40, RGFX_ARGUMENT_CONSTANT, RGFX_ARGUMENT_NO_LAYOUT),
            ]
        );
        let calls = p.call_log().calls_to("newArgumentEncoderWithArguments:");
        assert_eq!(calls.len(), 1);

        // Too small: the bytes needed, and nothing past them.
        let small = reply + 0x4000;
        assert!(region.write(small + 8, &[0xee; 8]));
        let m = query(
            RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
            small,
            64,
            0,
            &descriptors,
            "",
            &[],
        );
        assert_eq!(
            dec.argument_layout(&device, &m).unwrap(),
            (Some(9), RGFX_STATUS_TOO_SMALL)
        );
        assert_eq!(region.read(small, 8).unwrap(), 184u64.to_le_bytes());
        assert_eq!(region.read(small + 8, 8).unwrap(), [0xee; 8]);
    }

    /// A function's argument buffer is the platform's layout of the library's function, nested
    /// argument buffers after their holders: the pointer at index 1 names layout 1, whose one
    /// member follows layout 0's two
    #[test]
    fn a_functions_argument_buffer_answers_its_nested_layouts() {
        let (p, region, dec, reply) = decoder();
        let device = p.device();
        let library = device.new_library(b"lib").unwrap().unwrap().unwrap();
        dec.store(
            0x10,
            super::super::tables::Entry::of(ObjHandle::Library(library)),
        );
        let m = query(
            RGFX_ARGUMENT_SOURCE_FUNCTION,
            reply,
            0x1000,
            0x10,
            &[],
            "kernel_main",
            &[],
        );
        assert_eq!(
            dec.argument_layout(&device, &m).unwrap(),
            (Some(9), RGFX_STATUS_OK)
        );
        let (head, layouts, members) = answer(&region, reply);
        assert_eq!((head.layout_count, head.member_count), (2, 3));
        assert_eq!(
            (
                layouts[0].encoded_length,
                layouts[0].first_member,
                layouts[0].member_count
            ),
            (16, 0, 2)
        );
        assert_eq!(
            (
                layouts[1].encoded_length,
                layouts[1].first_member,
                layouts[1].member_count
            ),
            (8, 2, 1)
        );
        assert_eq!(
            (members[0].kind, members[0].layout),
            (RGFX_ARGUMENT_TEXTURE, RGFX_ARGUMENT_NO_LAYOUT)
        );
        assert_eq!(
            (members[1].kind, members[1].offset, members[1].layout),
            (RGFX_ARGUMENT_BUFFER, 8, 1)
        );
        assert_eq!(members[2].kind, RGFX_ARGUMENT_SAMPLER);
        let calls = p.call_log().calls_to("newFunctionWithName:");
        assert_eq!(calls.len(), 1);
    }

    /// What is not answered: a source the wire does not define, a block outside the processes'
    /// blocks, a library the decoder does not hold, a function with no argument buffer at the
    /// index, and a message too short for its head
    #[test]
    fn a_query_the_host_cannot_answer_says_why() {
        let (p, _region, dec, reply) = decoder();
        let device = p.device();
        let library = device.new_library(b"lib").unwrap().unwrap().unwrap();
        dec.store(
            0x10,
            super::super::tables::Entry::of(ObjHandle::Library(library)),
        );
        let d = [descriptor(58, 0, 0)];
        let bad_source = query(7, reply, 0x1000, 0, &d, "", &[]);
        assert_eq!(
            dec.argument_layout(&device, &bad_source).unwrap().1,
            RGFX_STATUS_INVALID_ARGUMENT
        );
        let outside = query(
            RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
            0x100,
            0x1000,
            0,
            &d,
            "",
            &[],
        );
        assert_eq!(
            dec.argument_layout(&device, &outside).unwrap().1,
            RGFX_STATUS_INVALID_ARGUMENT
        );
        let no_library = query(
            RGFX_ARGUMENT_SOURCE_FUNCTION,
            reply,
            0x1000,
            0x11,
            &[],
            "f",
            &[],
        );
        assert_eq!(
            dec.argument_layout(&device, &no_library).unwrap().1,
            RGFX_STATUS_NOT_FOUND
        );
        let mut index1 = query(
            RGFX_ARGUMENT_SOURCE_FUNCTION,
            reply,
            0x1000,
            0x10,
            &[],
            "f",
            &[],
        );
        index1[0x3c..0x40].copy_from_slice(&1u32.to_le_bytes()); // buffer_index
        assert_eq!(
            dec.argument_layout(&device, &index1).unwrap().1,
            RGFX_STATUS_REFUSED
        );
        let short = &bad_source[..40];
        assert_eq!(
            dec.argument_layout(&device, short).unwrap(),
            (Some(9), RGFX_STATUS_INVALID_ARGUMENT)
        );
        let empty = query(
            RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
            reply,
            0x1000,
            0,
            &[],
            "",
            &[],
        );
        assert_eq!(
            dec.argument_layout(&device, &empty).unwrap().1,
            RGFX_STATUS_REFUSED
        );
    }
}
