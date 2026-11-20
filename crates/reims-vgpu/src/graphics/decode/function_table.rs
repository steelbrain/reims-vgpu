//! Function pointers: `FUNCTION_HANDLE_CREATE`, `FUNCTION_TABLE_CREATE` and `FUNCTION_TABLE_SET`,
//! a compute pipeline state's function handles and visible function tables, the platform's under
//! the guest's ids.
//!
//! A handle is the platform's answer to `functionHandleWithFunction:` of the function the message
//! names, made as a pipeline's linked function is (its library, name, constants, options and
//! `specializedName`, [`GraphicsDecoder::linked_tail`]), so the platform finds it among the
//! functions the state's kernel links; Metal's nil (a function it does not link) is answered not
//! found and makes nothing. A table is the platform's `newVisibleFunctionTableWithDescriptor:`,
//! and its create's answer carries what the guest's table answers of it: its `gpuResourceID`,
//! which argument buffers hold, its `allocatedSize` and its `resourceOptions`. A set writes the
//! entries through the platform's table, which the GPU reads as a dispatch runs, so an entry set
//! before the commit of a dispatch encoded earlier is the one it calls, as on Metal; ring 0 carries
//! a set out before the process's later command buffers.

use reims_vgpu_graphics::wire::{
    RGFX_CMD_FUNCTION_HANDLE_CREATE, RGFX_CMD_FUNCTION_TABLE_CREATE,
    RGFX_EVT_FUNCTION_POINTER_STATE, RGFX_LINKED_FUNCTIONS, RGFX_STATUS_INVALID_ARGUMENT,
    RGFX_STATUS_NOT_FOUND, RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RgfxFunctionHandleCreate,
    RgfxFunctionPointerState, RgfxFunctionTableCreate, RgfxFunctionTableSet, RgfxMsgHeader,
};
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::function_table::VisibleFunctionTable;
use reims_vgpu_vulkan::core::platform::resource::ComputePipeline;
use reims_vgpu_vulkan::core::platform::types::NsRange;
use reims_vgpu_vulkan::core::raise::RaiseResult;

use super::{DecodeError, GraphicsDecoder, ObjHandle, read};

/// The byte offset of a `FUNCTION_HANDLE_CREATE`'s function record.
const FUNCTION_AT: u32 = std::mem::offset_of!(RgfxFunctionHandleCreate, function) as u32;

/// The pipeline state id a `FUNCTION_HANDLE_CREATE` or `FUNCTION_TABLE_CREATE` names, which ring 0
/// waits for while it is being compiled; `None` for another message, or one too short.
pub(crate) fn named_pipeline(message: &[u8]) -> Option<u64> {
    let header = read::<RgfxMsgHeader>(message)?;
    match header.r#type {
        RGFX_CMD_FUNCTION_HANDLE_CREATE => {
            read::<RgfxFunctionHandleCreate>(message).map(|m| m.pipeline_obj_id)
        }
        RGFX_CMD_FUNCTION_TABLE_CREATE => {
            read::<RgfxFunctionTableCreate>(message).map(|m| m.pipeline_obj_id)
        }
        _ => None,
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// `FUNCTION_HANDLE_CREATE` or `FUNCTION_TABLE_CREATE`: the handle or table made under the
    /// message's id, and the answer; `None` for a message too short to carry its seqno. The id
    /// must be the sender's and the message whole, else invalid; a pipeline state or library the
    /// sender does not hold is not found, and so is a function the state's kernel does not link
    /// (Metal's nil handle); a state, library or function the host refused, or an object the
    /// platform makes none of, is refused. Nothing is made but on success.
    pub(crate) fn function_pointer_state(
        &self,
        message: &[u8],
    ) -> RaiseResult<Option<RgfxFunctionPointerState>> {
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(None);
        };
        let answer = |status| RgfxFunctionPointerState {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_FUNCTION_POINTER_STATE,
                size: size_of::<RgfxFunctionPointerState>() as u32,
                seqno: header.seqno,
            },
            status,
            ..RgfxFunctionPointerState::default()
        };
        let (obj_id, pipeline_obj_id) = match header.r#type {
            RGFX_CMD_FUNCTION_HANDLE_CREATE => match read::<RgfxFunctionHandleCreate>(message) {
                Some(m) => (m.obj_id, m.pipeline_obj_id),
                None => return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT))),
            },
            RGFX_CMD_FUNCTION_TABLE_CREATE => match read::<RgfxFunctionTableCreate>(message) {
                Some(m) => (m.obj_id, m.pipeline_obj_id),
                None => return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT))),
            },
            _ => return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT))),
        };
        if !self.owned(obj_id) {
            return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT)));
        }
        let pipeline = match self.named(pipeline_obj_id) {
            Some(ObjHandle::ComputePipeline(p)) => p,
            Some(ObjHandle::Refused(_)) => return Ok(Some(answer(RGFX_STATUS_REFUSED))),
            _ => return Ok(Some(answer(RGFX_STATUS_NOT_FOUND))),
        };
        if header.r#type == RGFX_CMD_FUNCTION_TABLE_CREATE {
            let Some(m) = read::<RgfxFunctionTableCreate>(message) else {
                return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT)));
            };
            let Some(table) = pipeline.new_visible_function_table(m.function_count)? else {
                return Ok(Some(answer(RGFX_STATUS_REFUSED)));
            };
            let made = RgfxFunctionPointerState {
                gpu_resource_id: table.gpu_resource_id(),
                allocated_size: table.allocated_size(),
                resource_options: table.resource_options(),
                ..answer(RGFX_STATUS_OK)
            };
            self.store(obj_id, super::Entry::of(ObjHandle::FunctionTable(table)));
            return Ok(Some(made));
        }
        let refs = match self.linked_tail(message, FUNCTION_AT, 1) {
            Ok(refs) => refs,
            Err(DecodeError::BadReference) => return Ok(Some(answer(RGFX_STATUS_NOT_FOUND))),
            Err(_) => return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT))),
        };
        let listed = read::<RgfxFunctionHandleCreate>(message)
            .is_some_and(|m| m.function.list == RGFX_LINKED_FUNCTIONS);
        if !listed {
            return Ok(Some(answer(RGFX_STATUS_INVALID_ARGUMENT)));
        }
        let function = match super::linked_functions::<P>(refs)? {
            Ok(mut made) => made.functions.pop(),
            Err(reason) => {
                log::warn!(
                    target: "reims_vgpu::graphics",
                    "function handle {obj_id:#x} of pipeline state {pipeline_obj_id:#x} not \
                     made: {reason}"
                );
                None
            }
        };
        let Some(function) = function else {
            return Ok(Some(answer(RGFX_STATUS_REFUSED)));
        };
        let Some(handle) = pipeline.function_handle(&function)? else {
            return Ok(Some(answer(RGFX_STATUS_NOT_FOUND)));
        };
        self.store(obj_id, super::Entry::of(ObjHandle::FunctionHandle(handle)));
        Ok(Some(answer(RGFX_STATUS_OK)))
    }

    /// `FUNCTION_TABLE_SET`: the entries the message names set through the table's platform
    /// object. A table or handle the sender does not hold is a bad reference, and a message
    /// shorter than its handles malformed, each setting nothing; a range past the table, or a
    /// handle the platform refuses, raises in the platform's call, which sets none either.
    pub(crate) fn function_table_set(
        &self,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let Some(m) = read::<RgfxFunctionTableSet>(message) else {
            return Ok(Err(DecodeError::Malformed));
        };
        let head = size_of::<RgfxFunctionTableSet>();
        let ids: Option<Vec<u64>> = usize::try_from(m.count)
            .ok()
            .and_then(|count| count.checked_mul(8))
            .and_then(|len| message.get(head..head.checked_add(len)?))
            .map(|bytes| {
                bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b))
                    .collect()
            });
        let Some(ids) = ids else {
            return Ok(Err(DecodeError::Malformed));
        };
        let Some(ObjHandle::FunctionTable(table)) = self.named(m.obj_id) else {
            return Ok(Err(DecodeError::BadReference));
        };
        let mut handles = Vec::with_capacity(ids.len());
        for &id in &ids {
            if id == 0 {
                handles.push(None);
                continue;
            }
            match self.named(id) {
                Some(ObjHandle::FunctionHandle(h)) => handles.push(Some(h)),
                _ => return Ok(Err(DecodeError::BadReference)),
            }
        }
        table.set_functions(&handles, NsRange::new(m.first, m.count))?;
        Ok(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_graphics::wire::{
        RGFX_CMD_FUNCTION_TABLE_SET, RGFX_CMD_LIBRARY_CREATE, RGFX_LIBRARY_BYTES_OFFSET,
        RGFX_LINKED_PRIVATE_FUNCTIONS, RGFX_PIPELINE_OK, RgfxComputePipelineCreate,
        RgfxLibraryCreate, RgfxLinkedFunction,
    };
    use reims_vgpu_vulkan::core::platform::Handle as _;
    use reims_vgpu_vulkan::core::testkit::FakePlatform;
    use zerocopy::IntoBytes;

    use super::super::tests::{append_linked, compute_create, edit_head, pipeline_decoder};
    use super::super::{DecodeError, GraphicsDecoder, ObjHandle, PipelineReply};
    use super::*;

    /// The compute pipeline state the fake tests make, of library 1's `cmain`.
    const PIPELINE: u64 = 2;

    /// A `FUNCTION_HANDLE_CREATE` under `obj_id` of library `library`'s `name`, in `list`, among
    /// the functions pipeline state `pipeline` links.
    fn handle_create(obj_id: u64, pipeline: u64, library: u64, name: &str, list: u32) -> Vec<u8> {
        let head = size_of::<RgfxFunctionHandleCreate>();
        let mut bytes = RgfxFunctionHandleCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_FUNCTION_HANDLE_CREATE,
                size: 0,
                seqno: 5,
            },
            obj_id,
            pipeline_obj_id: pipeline,
            function: RgfxLinkedFunction {
                library_obj_id: library,
                name_offset: head as u32,
                name_len: name.len() as u32,
                list,
                ..RgfxLinkedFunction::default()
            },
        }
        .as_bytes()
        .to_vec();
        bytes.extend_from_slice(name.as_bytes());
        bytes.resize(bytes.len().next_multiple_of(8), 0);
        let size = bytes.len() as u32;
        bytes[4..8].copy_from_slice(&size.to_le_bytes());
        bytes
    }

    /// A `FUNCTION_TABLE_CREATE` under `obj_id` of `count` entries of pipeline state `pipeline`.
    fn table_create(obj_id: u64, pipeline: u64, count: u64) -> Vec<u8> {
        RgfxFunctionTableCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_FUNCTION_TABLE_CREATE,
                size: size_of::<RgfxFunctionTableCreate>() as u32,
                seqno: 6,
            },
            obj_id,
            pipeline_obj_id: pipeline,
            function_count: count,
        }
        .as_bytes()
        .to_vec()
    }

    /// A `FUNCTION_TABLE_SET` of table `obj_id`'s entries from `first` to `handles`.
    fn table_set(obj_id: u64, first: u64, handles: &[u64]) -> Vec<u8> {
        let mut bytes = RgfxFunctionTableSet {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_FUNCTION_TABLE_SET,
                size: (size_of::<RgfxFunctionTableSet>() + handles.len() * 8) as u32,
                seqno: 0,
            },
            obj_id,
            first,
            count: handles.len() as u64,
        }
        .as_bytes()
        .to_vec();
        bytes.extend_from_slice(handles.as_bytes());
        bytes
    }

    /// The fake decoder with library 1 and pipeline state [`PIPELINE`] of its `cmain`.
    fn decoder() -> (Arc<FakePlatform>, GraphicsDecoder<FakePlatform>) {
        let (p, mut dec, _) = pipeline_decoder();
        let made = dec.decode(&p.device(), &compute_create(PIPELINE)).unwrap();
        assert_eq!(made, Ok(()));
        (p, dec)
    }

    /// A handle create makes the platform's handle of the function the message names, made as a
    /// pipeline's linked function is, through the pipeline state, and stores it under the
    /// guest's id; the answer is OK and carries nothing else.
    #[test]
    fn a_function_handle_create_is_the_pipeline_s_handle_of_its_function() {
        let (p, dec) = decoder();
        let m = p.call_log().mark();
        let answer = dec
            .function_pointer_state(&handle_create(
                0x40,
                PIPELINE,
                1,
                "fp_add1",
                RGFX_LINKED_FUNCTIONS,
            ))
            .unwrap()
            .unwrap();
        assert_eq!(answer.header.r#type, RGFX_EVT_FUNCTION_POINTER_STATE);
        assert_eq!((answer.header.seqno, answer.status), (5, RGFX_STATUS_OK));
        assert_eq!(answer.gpu_resource_id, 0);
        let sels = p.call_log().sels_since(m);
        assert_eq!(sels.last().copied(), Some("functionHandleWithFunction:"));
        assert!(matches!(
            dec.object(0x40),
            Some(ObjHandle::FunctionHandle(_))
        ));
    }

    /// A table create makes the platform's table of the pipeline state's, and answers its
    /// `gpuResourceID`, `allocatedSize` and `resourceOptions`.
    #[test]
    fn a_function_table_create_answers_the_platform_s_table() {
        let (p, dec) = decoder();
        let m = p.call_log().mark();
        let answer = dec
            .function_pointer_state(&table_create(0x50, PIPELINE, 3))
            .unwrap()
            .unwrap();
        assert_eq!((answer.header.seqno, answer.status), (6, RGFX_STATUS_OK));
        let Some(ObjHandle::FunctionTable(table)) = dec.object(0x50) else {
            panic!("no table under the guest's id");
        };
        assert_eq!(answer.gpu_resource_id, table.gpu_resource_id());
        assert_ne!(answer.gpu_resource_id, 0);
        assert_eq!(answer.allocated_size, 256 + 8 * 3);
        assert_eq!(answer.resource_options, 512);
        assert_eq!(
            p.call_log().sels_since(m),
            ["newVisibleFunctionTableWithDescriptor:"]
        );
    }

    /// A create of a pipeline state the sender does not hold is not found, of an id not the
    /// sender's or a message too short invalid; none makes anything.
    #[test]
    fn a_function_pointer_create_of_no_pipeline_makes_nothing() {
        let (_, mut dec) = decoder();
        let status = |dec: &GraphicsDecoder<FakePlatform>, m: &[u8]| {
            dec.function_pointer_state(m).unwrap().unwrap().status
        };
        assert_eq!(
            status(&dec, &table_create(0x50, 9, 3)),
            RGFX_STATUS_NOT_FOUND
        );
        let handle = handle_create(0x40, 9, 1, "fp_add1", RGFX_LINKED_FUNCTIONS);
        assert_eq!(status(&dec, &handle), RGFX_STATUS_NOT_FOUND);
        let short = &table_create(0x50, PIPELINE, 3)[..24];
        assert_eq!(status(&dec, short), RGFX_STATUS_INVALID_ARGUMENT);
        let grouped = handle_create(0x40, PIPELINE, 1, "fp_add1", RGFX_LINKED_PRIVATE_FUNCTIONS);
        assert_eq!(status(&dec, &grouped), RGFX_STATUS_INVALID_ARGUMENT);
        dec.set_context(Some(3));
        assert_eq!(
            status(&dec, &table_create(0x50, PIPELINE, 3)),
            RGFX_STATUS_INVALID_ARGUMENT
        );
        assert!(dec.object(0x40).is_none() && dec.object(0x50).is_none());
    }

    /// A set writes the entries through the platform's table, a 0 handle a nil entry; a handle
    /// or table the sender does not hold is a bad reference and a message shorter than its
    /// handles malformed, each setting nothing.
    #[test]
    fn a_function_table_set_sets_the_platform_s_entries() {
        let (p, dec) = decoder();
        for m in [
            handle_create(0x40, PIPELINE, 1, "fp_add1", RGFX_LINKED_FUNCTIONS),
            handle_create(0x41, PIPELINE, 1, "fp_mul2", RGFX_LINKED_FUNCTIONS),
            table_create(0x50, PIPELINE, 3),
        ] {
            let answer = dec.function_pointer_state(&m).unwrap().unwrap();
            assert_eq!(answer.status, RGFX_STATUS_OK);
        }
        let device = p.device();
        let mut dec = dec;
        assert_eq!(
            dec.decode(&device, &table_set(0x50, 1, &[0x41, 0]))
                .unwrap(),
            Ok(())
        );
        assert_eq!(
            dec.decode(&device, &table_set(0x50, 0, &[0x40])).unwrap(),
            Ok(())
        );
        let ids = |dec: &GraphicsDecoder<FakePlatform>| {
            let Some(ObjHandle::FunctionTable(table)) = dec.object(0x50) else {
                panic!("no table");
            };
            table.entries()
        };
        let handle_id = |dec: &GraphicsDecoder<FakePlatform>, id| match dec.object(id) {
            Some(ObjHandle::FunctionHandle(h)) => h.id(),
            _ => panic!("no handle {id:#x}"),
        };
        let (add1, mul2) = (handle_id(&dec, 0x40), handle_id(&dec, 0x41));
        assert_eq!(ids(&dec), [Some(add1), Some(mul2), None]);
        assert_eq!(
            dec.decode(&device, &table_set(0x50, 0, &[0x99])).unwrap(),
            Err(DecodeError::BadReference)
        );
        assert_eq!(
            dec.decode(&device, &table_set(0x51, 0, &[0x40])).unwrap(),
            Err(DecodeError::BadReference)
        );
        let mut short = table_set(0x50, 0, &[0x41, 0x41]);
        short.truncate(short.len() - 8);
        assert_eq!(
            dec.decode(&device, &short).unwrap(),
            Err(DecodeError::Malformed)
        );
        assert_eq!(ids(&dec), [Some(add1), Some(mul2), None]);
    }

    /// The visible functions and kernels of the tests' visible function tables (the Vulkan
    /// crate's `src/gpu/visible_tables.metal`).
    const VISIBLE_TABLES: &[u8] =
        include_bytes!("../../../../reims-vgpu-vulkan/metallibs/visible_tables.metallib");

    /// What `table_kernel` and `table_arg_kernel` write through the decoder on platform `P`, each
    /// linking `table_add1` and `table_mul2`, over six threads choosing entries 0, 1, 2, 1, 0, 2:
    /// the first through a table of `[add1, mul2, add1]` bound at buffer 0, the second through a
    /// table of `[mul2, add1, mul2]` an argument buffer holds by its resource id; with the state
    /// a table create answered, and the status of a handle of `table_kernel` itself, which no
    /// pipeline links.
    fn tables_through_the_decoder<P: crate::platform::ApiPlatform>()
    -> ([i32; 6], [i32; 6], RgfxFunctionPointerState, u32) {
        use reims_vgpu_vulkan::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_vulkan::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
        use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
        use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
        use reims_vgpu_vulkan::core::platform::types::{MtlSize, NsRange};
        use reims_vgpu_vulkan::core::platform::values::ComputePassDesc;
        use reims_vgpu_vulkan::core::raise::OnFault;
        let platform = P::process();
        let device = platform.metal_device(None).unwrap();
        let mut dec = GraphicsDecoder::<P>::new(
            Arc::clone(&platform),
            Arc::new(crate::platform::test_region(&*platform, 0x10000)),
        );
        let replies = Arc::new(reims_vgpu_vulkan::core::sync::Mutex::new(Vec::new()));
        let sink_replies = Arc::clone(&replies);
        dec.set_reflection_sink(Arc::new(move |r: PipelineReply| {
            sink_replies
                .lock()
                .push((r.reflection.obj_id, r.reflection.status));
        }));
        let mut library = RgfxLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_CREATE,
                size: (RGFX_LIBRARY_BYTES_OFFSET as usize + VISIBLE_TABLES.len())
                    .next_multiple_of(8) as u32,
                seqno: 1,
            },
            obj_id: 1,
            byte_length: VISIBLE_TABLES.len() as u64,
            ..RgfxLibraryCreate::default()
        }
        .as_bytes()
        .to_vec();
        library.extend_from_slice(VISIBLE_TABLES);
        library.resize(library.len().next_multiple_of(8), 0);
        assert_eq!(dec.decode(&device, &library).unwrap(), Ok(()));
        for (obj_id, kernel) in [(100, "table_kernel"), (101, "table_arg_kernel")] {
            let mut bytes = compute_create(obj_id);
            let at = bytes.len() as u32;
            bytes.extend_from_slice(kernel.as_bytes());
            bytes.resize(bytes.len().next_multiple_of(8), 0);
            let tail = append_linked(
                &mut bytes,
                1,
                &[
                    ("table_add1", RGFX_LINKED_FUNCTIONS, 0, "", &[], ""),
                    ("table_mul2", RGFX_LINKED_FUNCTIONS, 0, "", &[], ""),
                ],
            );
            edit_head(&mut bytes, |m: &mut RgfxComputePipelineCreate| {
                (m.name_offset, m.name_len) = (at, kernel.len() as u32);
                (m.linked_offset, m.linked_count) = tail;
            });
            assert_eq!(dec.decode(&device, &bytes).unwrap(), Ok(()), "{kernel}");
            assert_eq!(
                replies.lock().pop(),
                Some((obj_id, RGFX_PIPELINE_OK)),
                "{kernel}"
            );
        }
        let ask = |m: Vec<u8>| dec.function_pointer_state(&m).unwrap().unwrap();
        for (obj_id, pipeline, name) in [
            (0x40, 100, "table_add1"),
            (0x41, 100, "table_mul2"),
            (0x42, 101, "table_add1"),
            (0x43, 101, "table_mul2"),
        ] {
            let made = ask(handle_create(
                obj_id,
                pipeline,
                1,
                name,
                RGFX_LINKED_FUNCTIONS,
            ));
            assert_eq!(made.status, RGFX_STATUS_OK, "{name} of {pipeline}");
        }
        let unlinked = ask(handle_create(
            0x44,
            100,
            1,
            "table_kernel",
            RGFX_LINKED_FUNCTIONS,
        ))
        .status;
        let state = ask(table_create(0x50, 100, 3));
        assert_eq!(state.status, RGFX_STATUS_OK);
        let held = ask(table_create(0x51, 101, 3));
        assert_eq!(held.status, RGFX_STATUS_OK);
        assert_ne!(held.gpu_resource_id, state.gpu_resource_id);
        for m in [
            table_set(0x50, 0, &[0x40, 0x41]),
            table_set(0x50, 2, &[0x40]),
            table_set(0x51, 0, &[0x43, 0x42, 0x43]),
        ] {
            assert_eq!(dec.decode(&device, &m).unwrap(), Ok(()));
        }
        let buffer = |len| {
            device
                .new_buffer_of_length(OnFault::Return, len, 0)
                .unwrap()
                .unwrap()
        };
        let which = buffer(24);
        const WHICH: [u32; 6] = [0, 1, 2, 1, 0, 2];
        // SAFETY: the buffer is Shared and holds 24 bytes; no GPU work names it yet.
        unsafe {
            std::ptr::copy_nonoverlapping(WHICH.as_ptr(), which.contents().as_ptr::<u32>(), 6)
        };
        let arguments = buffer(8);
        // SAFETY: as above, 8 bytes: the argument buffer's table, by its resource id.
        unsafe {
            arguments
                .contents()
                .as_ptr::<u64>()
                .write(held.gpu_resource_id)
        };
        let run = |pipeline: u64, table: Option<u64>| -> [i32; 6] {
            let Some(ObjHandle::ComputePipeline(state)) = dec.object(pipeline) else {
                panic!("no pipeline {pipeline}");
            };
            let table = table.map(|id| match dec.object(id) {
                Some(ObjHandle::FunctionTable(t)) => t,
                _ => panic!("no table {id:#x}"),
            });
            let out = buffer(24);
            let queue = device.new_command_queue().unwrap();
            let cb = queue.command_buffer(OnFault::Return).unwrap().unwrap();
            let enc = cb
                .compute_encoder_for_pass(&ComputePassDesc::default())
                .unwrap()
                .unwrap();
            let bound = [Some(&which), Some(&out)];
            let through_arguments = [Some(&arguments), Some(&which), Some(&out)];
            let mut cmds = vec![ComputeCmd::SetComputePipelineState { state: Some(state) }];
            match table {
                Some(table) => cmds.extend([
                    ComputeCmd::SetVisibleFunctionTable {
                        function_table: Some(table),
                        buffer_index: 0,
                    },
                    ComputeCmd::SetBuffers {
                        buffers: &bound,
                        offsets: &[0, 0],
                        range: NsRange {
                            location: 1,
                            length: 2,
                        },
                    },
                ]),
                None => cmds.push(ComputeCmd::SetBuffers {
                    buffers: &through_arguments,
                    offsets: &[0, 0, 0],
                    range: NsRange {
                        location: 0,
                        length: 3,
                    },
                }),
            }
            cmds.extend([
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(6, 1, 1),
                    threads_per_threadgroup: MtlSize::new(6, 1, 1),
                },
                ComputeCmd::EndEncoding,
            ]);
            for cmd in cmds {
                enc.encode(OnFault::Return, cmd).unwrap();
            }
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            let mut got = [0; 6];
            // SAFETY: the GPU wrote the buffer's 24 bytes, and is done.
            unsafe {
                std::ptr::copy_nonoverlapping(out.contents().as_ptr::<i32>(), got.as_mut_ptr(), 6)
            };
            got
        };
        (run(100, Some(0x50)), run(101, None), state, unlinked)
    }

    /// On the Vulkan platform a kernel calls the functions its pipeline links through a table the
    /// guest made and set, bound at a buffer index or held by its resource id in an argument
    /// buffer, as the AMD Radeon Pro 5300M's kernels do (measured: macOS 26.7.1, oracle_fnptr:
    /// 11 22 13 26 15 16 and, with useResource:, 20 12 24 14 28 30); a handle of the kernel
    /// itself, which no pipeline links, is not found, Metal's nil.
    #[test]
    #[ignore = "rt refuses the table kernel: it reads an input the compute encoder does not supply yet (rt-refused-gap)"]
    fn on_vulkan_a_kernel_calls_through_the_guest_s_tables() {
        let (bound, held, state, unlinked) =
            tables_through_the_decoder::<reims_vgpu_vulkan::RtPlatform>();
        assert_eq!(bound, [11, 22, 13, 26, 15, 16]);
        assert_eq!(held, [20, 12, 24, 14, 28, 30]);
        assert_ne!(state.gpu_resource_id, 0);
        assert_ne!(state.allocated_size, 0);
        assert_eq!(unlinked, RGFX_STATUS_NOT_FOUND);
    }
}
