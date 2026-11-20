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

use reims_vgpu_graphics::wire::{
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
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::types::{ArgumentKind, ArgumentLayout};
use reims_vgpu_vulkan::core::platform::values::ArgumentDesc;
use reims_vgpu_vulkan::core::raise::RaiseResult;
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

    use reims_vgpu_graphics::region::Region;
    use reims_vgpu_graphics::wire::{
        RGFX_CMD_ARGUMENT_LAYOUT, RGFX_CONTEXT_ALLOCATIONS_OFFSET, RGFX_CURSOR_SPAN,
        RgfxFunctionConstant,
    };
    use reims_vgpu_vulkan::core::platform::blit::ResourceRef;
    use reims_vgpu_vulkan::core::platform::device::GpuDevice;
    use reims_vgpu_vulkan::core::testkit::FakePlatform;

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

    /// The kernels of `tests/shaders/argument_buffers.metal`.
    const ARGUMENT_BUFFERS: &[u8] =
        include_bytes!("../../../../../tests/shaders/argument_buffers.metallib");

    type Rt = reims_vgpu_vulkan::RtPlatform;
    type RtBuffer = <Rt as Platform>::Buffer;

    /// The Vulkan platform, its device, the decoder over it, a region with one MiB of the processes' blocks, the
    /// reply's offset in it, and the argument-buffer kernels' library as object 0x10.
    fn on_vulkan() -> (
        Arc<Rt>,
        <Rt as Platform>::Device,
        Arc<Region>,
        GraphicsDecoder<Rt>,
        u64,
    ) {
        use crate::platform::ApiPlatform as _;
        let platform = Rt::process();
        let device = platform.metal_device(None).unwrap();
        let blocks = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
        let size = blocks + 0x10_0000 + RGFX_CURSOR_SPAN;
        let region = Arc::new(Region::new(size as usize).unwrap());
        let dec = GraphicsDecoder::<Rt>::new(Arc::clone(&platform), Arc::clone(&region));
        let library = device
            .new_library(ARGUMENT_BUFFERS)
            .unwrap()
            .unwrap()
            .unwrap();
        dec.store(
            0x10,
            super::super::tables::Entry::of(ObjHandle::Library(library)),
        );
        (platform, device, region, dec, blocks + 0x2000)
    }

    /// A Shared buffer of `words`, each `word(i)`.
    fn words(
        device: &<Rt as Platform>::Device,
        words: usize,
        word: impl Fn(u32) -> u32,
    ) -> RtBuffer {
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::raise::OnFault;
        let buffer = device
            .new_buffer_of_length(OnFault::Return, (words * 4).max(16) as u64, 0)
            .unwrap()
            .unwrap();
        let at = buffer.contents().as_ptr::<u32>();
        for i in 0..words {
            // SAFETY: the buffer is Shared and holds `words` words; the GPU has not started.
            unsafe { at.add(i).write(word(i as u32)) };
        }
        buffer
    }

    /// Writes `value`'s bytes at byte `offset` of `buffer`, as the guest's encoder writes an
    /// argument: an 8-byte word, or a constant's bytes.
    fn put(buffer: &RtBuffer, offset: u64, value: &[u8]) {
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        assert!(offset as usize + value.len() <= buffer.length() as usize);
        // SAFETY: the bytes lie in the buffer (above), which is Shared; the GPU has not started.
        unsafe {
            let at = buffer.contents().as_ptr::<u8>().add(offset as usize);
            std::ptr::copy_nonoverlapping(value.as_ptr(), at, value.len());
        };
    }

    /// Runs `kernel` of the library over `threads` threads, the argument buffer `args` at index
    /// 0 and `out` at 1, `resident` made resident for reads; answers `out`'s first `threads`
    /// words.
    fn run(
        device: &<Rt as Platform>::Device,
        kernel: &str,
        args: &RtBuffer,
        resident: &[ResourceRef<'_, Rt>],
        threads: u64,
    ) -> Vec<u32> {
        use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Library as _};
        use reims_vgpu_vulkan::core::platform::types::{MtlSize, NsRange};
        use reims_vgpu_vulkan::core::platform::values::ComputePassDesc;
        use reims_vgpu_vulkan::core::raise::OnFault;
        let library = device
            .new_library(ARGUMENT_BUFFERS)
            .unwrap()
            .unwrap()
            .unwrap();
        let function = library
            .new_function(OnFault::Return, kernel)
            .unwrap()
            .unwrap();
        let pipeline = device
            .new_compute_pipeline_state_for_function(Some(&function))
            .unwrap()
            .unwrap();
        let out = words(device, threads as usize, |_| 0xdead_beef);
        let resources: Vec<_> = resident.iter().map(|r| Some(*r)).collect();
        let queue = device.new_command_queue().unwrap();
        let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
        let enc = cb
            .compute_encoder_for_pass(&ComputePassDesc::default())
            .unwrap()
            .unwrap();
        for cmd in [
            ComputeCmd::SetComputePipelineState {
                state: Some(&pipeline),
            },
            ComputeCmd::SetBuffers {
                buffers: &[Some(args), Some(&out)],
                offsets: &[0, 0],
                range: NsRange {
                    location: 0,
                    length: 2,
                },
            },
            ComputeCmd::UseResources {
                resources: &resources,
                usage: 1, // MTLResourceUsageRead
            },
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(threads / 64, 1, 1),
                threads_per_threadgroup: MtlSize::new(64, 1, 1),
            },
            ComputeCmd::EndEncoding,
        ] {
            enc.encode(OnFault::Return, cmd).unwrap();
        }
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let at = out.contents().as_ptr::<u32>();
        // SAFETY: the GPU wrote the buffer's `threads` words, and is done.
        (0..threads as usize)
            .map(|i| unsafe { at.add(i).read() })
            .collect()
    }

    /// The members of the answer's layout `layout`: id, count, offset, stride, size, kind and the
    /// layout each names.
    fn members_of(
        layouts: &[RgfxArgumentLayoutRecord],
        members: &[RgfxArgumentMemberRecord],
        layout: usize,
    ) -> Vec<(u32, u32, u64, u64, u64, u32, u32)> {
        let l = &layouts[layout];
        members[l.first_member as usize..][..l.member_count as usize]
            .iter()
            .map(|m| (m.id, m.count, m.offset, m.stride, m.size, m.kind, m.layout))
            .collect()
    }

    /// On the Vulkan platform, the guest's probe shape reads what the guest wrote: `sum_args`'
    /// argument buffer answers two pointers at 0 and 8, 16 bytes aligned to 8, from the function
    /// and from the same two descriptors, as Metal's encoder answers it; the words the guest's
    /// encoder writes there (each buffer's GPU address plus its offset) are what the kernel reads,
    /// for two buffers, and for two places in one buffer
    #[test]
    fn on_vulkan_a_kernel_reads_the_pointers_the_guest_writes_at_its_argument_layout() {
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        let (_platform, device, region, dec, reply) = on_vulkan();
        let mut answers = Vec::new();
        for m in [
            query(
                RGFX_ARGUMENT_SOURCE_FUNCTION,
                reply,
                0x1000,
                0x10,
                &[],
                "sum_args",
                &[],
            ),
            query(
                RGFX_ARGUMENT_SOURCE_DESCRIPTORS,
                reply,
                0x1000,
                0,
                &[descriptor(60, 0, 0), descriptor(60, 1, 0)],
                "",
                &[],
            ),
        ] {
            assert_eq!(
                dec.argument_layout(&device, &m).unwrap(),
                (Some(9), RGFX_STATUS_OK)
            );
            answers.push(answer(&region, reply));
        }
        for (head, layouts, members) in &answers {
            assert_eq!((head.layout_count, head.member_count), (1, 2));
            assert_eq!((layouts[0].encoded_length, layouts[0].alignment), (16, 8));
            assert_eq!(
                members_of(layouts, members, 0),
                [
                    (0, 1, 0, 8, 8, RGFX_ARGUMENT_BUFFER, RGFX_ARGUMENT_NO_LAYOUT),
                    (1, 1, 8, 8, 8, RGFX_ARGUMENT_BUFFER, RGFX_ARGUMENT_NO_LAYOUT),
                ]
            );
        }
        let (_, layouts, members) = &answers[0];
        let (at_a, at_b) = (members[0].offset, members[1].offset);
        const N: usize = 4096;
        let a = words(&device, N + 256, |i| i * 3 + 1);
        let b = words(&device, N + 64, |i| i.wrapping_mul(0x9e37_79b9));
        let args = words(&device, layouts[0].encoded_length as usize / 4, |_| 0);
        put(&args, at_a, a.gpu_address().as_bytes());
        put(&args, at_b, (b.gpu_address() + 256).as_bytes());
        let got = run(
            &device,
            "sum_args",
            &args,
            &[ResourceRef::Buffer(&a), ResourceRef::Buffer(&b)],
            N as u64,
        );
        let want: Vec<u32> = (0..N as u32)
            .map(|i| (i * 3 + 1).wrapping_add((i + 64).wrapping_mul(0x9e37_79b9)))
            .collect();
        assert_eq!(first_wrong(&got, &want), None);
        // Both pointers into one buffer, 1 KiB apart.
        put(&args, at_b, (a.gpu_address() + 1024).as_bytes());
        let got = run(
            &device,
            "sum_args",
            &args,
            &[ResourceRef::Buffer(&a)],
            N as u64,
        );
        let want: Vec<u32> = (0..N as u32)
            .map(|i| (i * 3 + 1) + ((i + 256) * 3 + 1))
            .collect();
        assert_eq!(first_wrong(&got, &want), None);
    }

    /// The first index where `got` differs from `want`, with the two values, and how many differ.
    fn first_wrong(got: &[u32], want: &[u32]) -> Option<(usize, u32, u32, usize)> {
        assert_eq!(got.len(), want.len());
        let wrong = got.iter().zip(want).filter(|(g, w)| g != w).count();
        (got.iter().zip(want).enumerate())
            .find(|(_, (g, w))| g != w)
            .map(|(i, (g, w))| (i, *g, *w, wrong))
    }

    /// On the Vulkan platform, every kind a kernel reads through an argument buffer lands where
    /// the guest writes it: `mixed_args`' layout answers an array of two pointers at 0, a constant
    /// at 16, a texture at 24 and a pointer at 32 to a nested argument buffer (layout 1: a pointer
    /// at 0, a constant at 8), and the kernel reads each word the guest writes there: the buffers'
    /// addresses, the constant, the texture's resource ID, and the nested buffer's address and
    /// words
    #[test]
    fn on_vulkan_every_argument_kind_lands_where_the_guest_writes_it() {
        use reims_vgpu_vulkan::core::platform::host_range::HostRange;
        use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, Texture as _};
        use reims_vgpu_vulkan::core::platform::types::{MtlOrigin, MtlRegion, MtlSize};
        use reims_vgpu_vulkan::core::platform::values::TextureDesc;
        use reims_vgpu_vulkan::core::raise::OnFault;
        let (platform, device, region, dec, reply) = on_vulkan();
        let m = query(
            RGFX_ARGUMENT_SOURCE_FUNCTION,
            reply,
            0x1000,
            0x10,
            &[],
            "mixed_args",
            &[],
        );
        assert_eq!(
            dec.argument_layout(&device, &m).unwrap(),
            (Some(9), RGFX_STATUS_OK)
        );
        let (head, layouts, members) = answer(&region, reply);
        assert_eq!((head.layout_count, head.member_count), (2, 6));
        assert_eq!((layouts[0].encoded_length, layouts[0].alignment), (40, 8));
        assert_eq!((layouts[1].encoded_length, layouts[1].alignment), (16, 8));
        let none = RGFX_ARGUMENT_NO_LAYOUT;
        assert_eq!(
            members_of(&layouts, &members, 0),
            [
                (0, 2, 0, 8, 8, RGFX_ARGUMENT_BUFFER, none),
                (2, 1, 16, 4, 4, RGFX_ARGUMENT_CONSTANT, none),
                (3, 1, 24, 8, 8, RGFX_ARGUMENT_TEXTURE, none),
                (4, 1, 32, 8, 8, RGFX_ARGUMENT_BUFFER, 1),
            ]
        );
        assert_eq!(
            members_of(&layouts, &members, 1),
            [
                (0, 1, 0, 8, 8, RGFX_ARGUMENT_BUFFER, none),
                (1, 1, 8, 4, 4, RGFX_ARGUMENT_CONSTANT, none),
            ]
        );
        const N: usize = 256;
        let a = words(&device, N, |i| i + 1);
        let b = words(&device, N + 64, |i| i * 2);
        let c = words(&device, N, |i| i * 100);
        let texels: Vec<u32> = (0..N as u32).map(|i| i * 11).collect();
        let desc = platform.texture_descriptor(&TextureDesc {
            pixel_format: 53, // MTLPixelFormatR32Uint
            width: N as u64,
            height: 1,
            ..TextureDesc::default()
        });
        let texture = device.new_texture(OnFault::Return, &desc).unwrap().unwrap();
        texture
            .replace_region(
                MtlRegion {
                    origin: MtlOrigin::default(),
                    size: MtlSize::new(N as u64, 1, 1),
                },
                0,
                0,
                HostRange::of_slice(texels.as_bytes()),
                N as u64 * 4,
                0,
            )
            .unwrap();
        let inner = words(&device, layouts[1].encoded_length as usize / 4, |_| 0);
        let nested = members_of(&layouts, &members, 1);
        put(&inner, nested[0].2, c.gpu_address().as_bytes());
        put(&inner, nested[1].2, 5u32.as_bytes());
        let args = words(&device, layouts[0].encoded_length as usize / 4, |_| 0);
        let outer = members_of(&layouts, &members, 0);
        put(&args, outer[0].2, a.gpu_address().as_bytes());
        put(
            &args,
            outer[0].2 + outer[0].3,
            (b.gpu_address() + 256).as_bytes(),
        );
        put(&args, outer[1].2, 3u32.as_bytes());
        put(&args, outer[2].2, texture.gpu_resource_id().as_bytes());
        put(&args, outer[3].2, inner.gpu_address().as_bytes());
        let got = run(
            &device,
            "mixed_args",
            &args,
            &[
                ResourceRef::Buffer(&a),
                ResourceRef::Buffer(&b),
                ResourceRef::Buffer(&c),
                ResourceRef::Buffer(&inner),
                ResourceRef::Texture(&texture),
            ],
            N as u64,
        );
        let want: Vec<u32> = (0..N as u32)
            .map(|i| (i + 1) + (i + 64) * 2 * 3 + i * 11 + i * 100 + 5)
            .collect();
        assert_eq!(first_wrong(&got, &want), None);
    }
}
