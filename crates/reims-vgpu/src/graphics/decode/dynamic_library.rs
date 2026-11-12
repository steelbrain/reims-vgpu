//! Dynamic libraries: `DYNAMIC_LIBRARY_CREATE`, the platform's dynamic library of a library the
//! guest made, under the guest's id, and the run of [`RgfxPipelineLibrary`] that ends a
//! `COMPUTE_PIPELINE_CREATE`: the dynamic libraries its kernel links, preloaded or loaded under an
//! install name.
//!
//! Metal loads the library an install name names from the file it names when it creates a
//! pipeline state; the guest reads that file in its own process and names each install name its
//! kernel's libraries depend on with the library it made of the file, so the platform's creation
//! finds each in the descriptor ([`DynamicLibrariesDesc`]).

use reims_vgpu_graphics::wire::{
    RGFX_STATUS_INVALID_ARGUMENT, RGFX_STATUS_NOT_FOUND, RGFX_STATUS_OK, RGFX_STATUS_REFUSED,
    RgfxComputePipelineCreate, RgfxDynamicLibraryCreate, RgfxMsgHeader, RgfxPipelineLibrary,
};
use reims_vgpu_vulkan::core::platform::device::GpuDevice;
use reims_vgpu_vulkan::core::platform::values::DynamicLibrariesDesc;
use reims_vgpu_vulkan::core::platform::{Handle, Platform};
use reims_vgpu_vulkan::core::raise::RaiseResult;
use zerocopy::FromBytes;

use super::{DecodeError, GraphicsDecoder, ObjHandle, read, str_tail};

impl<P: Platform> GraphicsDecoder<P> {
    /// `DYNAMIC_LIBRARY_CREATE` on `device`: the dynamic library made under the message's id, and
    /// the create's seqno with the status its `COMPLETE` answers; `None` for a message too short
    /// to carry its seqno. The id must be the sender's and the message whole, else invalid; a
    /// library the sender does not hold is not found; a library the host refused, or one the
    /// platform refuses as a dynamic library, is refused. A library still being made is waited
    /// for. Nothing is made but on success.
    pub(crate) fn dynamic_library_create(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Option<(u64, u32)>> {
        let Some(header) = read::<RgfxMsgHeader>(message) else {
            return Ok(None);
        };
        let answer = |status| Ok(Some((header.seqno, status)));
        let Some(m) = read::<RgfxDynamicLibraryCreate>(message) else {
            return answer(RGFX_STATUS_INVALID_ARGUMENT);
        };
        if !self.owned(m.obj_id) {
            return answer(RGFX_STATUS_INVALID_ARGUMENT);
        }
        let library = match self.library_ref(m.library_obj_id) {
            Ok(library) => library,
            Err(_) => {
                return answer(match self.named(m.library_obj_id) {
                    Some(ObjHandle::Refused(_)) => RGFX_STATUS_REFUSED,
                    _ => RGFX_STATUS_NOT_FOUND,
                });
            }
        };
        let Some(library) = library.resolve() else {
            return answer(RGFX_STATUS_REFUSED);
        };
        let made = match device.new_dynamic_library(&library)? {
            Ok(Some(made)) => made,
            Ok(None) => return answer(RGFX_STATUS_REFUSED),
            Err(e) => {
                log::info!(
                    target: "reims_vgpu::graphics",
                    "a dynamic library create (seqno {}, object {:#x}) of library {:#x} was \
                     refused: {}",
                    header.seqno,
                    m.obj_id,
                    m.library_obj_id,
                    e.description
                );
                return answer(RGFX_STATUS_REFUSED);
            }
        };
        self.store(m.obj_id, super::Entry::of(ObjHandle::DynamicLibrary(made)));
        answer(RGFX_STATUS_OK)
    }

    /// The dynamic libraries compute create `message` (whose head is `m`) names, in the run of
    /// `library_count` [`RgfxPipelineLibrary`] that ends it: each a dynamic library of the
    /// sender's, preloaded in the run's order, or loaded under its install name. A run past the
    /// head or the message, or a name the message does not cover or that is not UTF-8, is
    /// [`DecodeError::Malformed`]; an id that names no dynamic library of the sender's is
    /// [`DecodeError::BadReference`].
    pub(super) fn pipeline_libraries(
        &self,
        message: &[u8],
        m: &RgfxComputePipelineCreate,
    ) -> Result<DynamicLibrariesDesc<P>, DecodeError> {
        let mut out = DynamicLibrariesDesc::default();
        if m.library_count == 0 {
            return Ok(out);
        }
        let run = (m.library_count as usize)
            .checked_mul(size_of::<RgfxPipelineLibrary>())
            .ok_or(DecodeError::Malformed)?;
        let at = message
            .len()
            .checked_sub(run)
            .filter(|at| *at >= size_of::<RgfxComputePipelineCreate>())
            .ok_or(DecodeError::Malformed)?;
        let (records, _) = message[at..].as_chunks::<{ size_of::<RgfxPipelineLibrary>() }>();
        for bytes in records {
            let record = RgfxPipelineLibrary::read_from_bytes(bytes.as_slice())
                .map_err(|_| DecodeError::Malformed)?;
            let library = match self.named(record.obj_id) {
                Some(ObjHandle::DynamicLibrary(library)) if self.owned(record.obj_id) => {
                    library.clone_ref()
                }
                _ => return Err(DecodeError::BadReference),
            };
            if record.name_len == 0 {
                out.preloaded.push(library);
            } else {
                let name = str_tail(message, record.name_offset, record.name_len)
                    .ok_or(DecodeError::Malformed)?;
                out.loaded.push((name.to_owned(), library));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_graphics::wire::{RGFX_CMD_DYNAMIC_LIBRARY_CREATE, RGFX_PIPELINE_OK};
    use reims_vgpu_vulkan::core::platform::types::NsErrorText;
    use reims_vgpu_vulkan::core::testkit::{Arg, FakePlatform, Step};
    use zerocopy::IntoBytes;

    use super::super::tests::{compute_create, edit_head, pipeline_decoder};
    use super::super::{DecodeError, GraphicsDecoder};
    use super::*;

    /// A `DYNAMIC_LIBRARY_CREATE` under `obj_id` of library `library`.
    fn create(obj_id: u64, library: u64) -> Vec<u8> {
        RgfxDynamicLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_DYNAMIC_LIBRARY_CREATE,
                size: size_of::<RgfxDynamicLibraryCreate>() as u32,
                seqno: 7,
            },
            obj_id,
            library_obj_id: library,
        }
        .as_bytes()
        .to_vec()
    }

    /// The reflection replies the fake decoder's pipelines posted.
    type Replies = Arc<
        reims_vgpu_vulkan::core::sync::Mutex<
            Vec<reims_vgpu_graphics::wire::RgfxPipelineReflection>,
        >,
    >;

    /// The fake decoder with library 1, and its status for each create of `messages`.
    fn made(
        messages: &[Vec<u8>],
    ) -> (
        Arc<FakePlatform>,
        GraphicsDecoder<FakePlatform>,
        Replies,
        Vec<u32>,
    ) {
        let (p, dec, replies) = pipeline_decoder();
        let device = p.device();
        let statuses = messages
            .iter()
            .map(|m| {
                let (seqno, status) = dec.dynamic_library_create(&device, m).unwrap().unwrap();
                assert_eq!(seqno, 7);
                status
            })
            .collect();
        (p, dec, replies, statuses)
    }

    /// `compute_create(obj_id)` ending with the run of `libraries`, each an id and the install
    /// name it is loaded under (`None`: preloaded), the names before the run.
    fn compute_with(obj_id: u64, libraries: &[(u64, Option<&str>)]) -> Vec<u8> {
        let mut bytes = compute_create(obj_id);
        let mut records = Vec::new();
        for &(id, name) in libraries {
            let mut record = RgfxPipelineLibrary {
                obj_id: id,
                ..RgfxPipelineLibrary::default()
            };
            if let Some(name) = name {
                record.name_offset = bytes.len() as u32;
                record.name_len = name.len() as u32;
                bytes.extend_from_slice(name.as_bytes());
            }
            records.push(record);
        }
        bytes.resize(bytes.len().next_multiple_of(8), 0);
        bytes.extend_from_slice(records.as_bytes());
        let size = bytes.len() as u32;
        edit_head(&mut bytes, |m: &mut RgfxComputePipelineCreate| {
            m.header.size = size;
            m.library_count = libraries.len() as u32;
        });
        bytes
    }

    /// A create makes the platform's dynamic library of the library the message names, under the
    /// guest's id, and answers OK.
    #[test]
    fn a_dynamic_library_create_makes_the_platform_s_of_its_library() {
        let (p, dec) = {
            let (p, dec, _, _) = made(&[]);
            (p, dec)
        };
        let m = p.call_log().mark();
        let (seqno, status) = dec
            .dynamic_library_create(&p.device(), &create(0x60, 1))
            .unwrap()
            .unwrap();
        assert_eq!((seqno, status), (7, RGFX_STATUS_OK));
        assert_eq!(p.call_log().sels_since(m), ["newDynamicLibrary:error:"]);
        assert!(matches!(
            dec.object(0x60),
            Some(ObjHandle::DynamicLibrary(_))
        ));
    }

    /// A create of a library the sender does not hold is not found; of an id not the sender's or a
    /// message too short, invalid; one the platform refuses, refused, its error logged. None makes
    /// anything; a message too short to carry its seqno is not answered.
    #[test]
    fn a_dynamic_library_create_that_makes_nothing() {
        let (p, mut dec, _, statuses) = made(&[create(0x60, 9), create(0x61, 1)[..24].to_vec()]);
        assert_eq!(
            statuses,
            [RGFX_STATUS_NOT_FOUND, RGFX_STATUS_INVALID_ARGUMENT]
        );
        let device = p.device();
        p.script().push(
            "newDynamicLibrary:error:",
            Step::Error(NsErrorText {
                domain: "MTLDynamicLibraryDomain".to_owned(),
                code: 2,
                description: "reads a function constant".to_owned(),
            }),
        );
        let refused = dec
            .dynamic_library_create(&device, &create(0x62, 1))
            .unwrap()
            .unwrap();
        assert_eq!(refused.1, RGFX_STATUS_REFUSED);
        p.script().push("newDynamicLibrary:error:", Step::Nil);
        let nil = dec
            .dynamic_library_create(&device, &create(0x63, 1))
            .unwrap()
            .unwrap();
        assert_eq!(nil.1, RGFX_STATUS_REFUSED);
        assert_eq!(
            dec.dynamic_library_create(&device, &create(0x64, 1)[..8])
                .unwrap(),
            None
        );
        dec.set_context(Some(3));
        let foreign = dec
            .dynamic_library_create(&device, &create(0x65, 1))
            .unwrap()
            .unwrap();
        assert_eq!(foreign.1, RGFX_STATUS_INVALID_ARGUMENT);
        for id in 0x60..=0x65 {
            assert!(dec.object(id).is_none(), "{id:#x}");
        }
    }

    /// A compute create's run of dynamic libraries reaches the platform's creator: the preloaded
    /// ones in the run's order, and each loaded one under the install name the run gives it.
    #[test]
    fn a_compute_create_carries_its_dynamic_libraries() {
        let (p, mut dec, replies, statuses) = made(&[create(0x60, 1), create(0x61, 1)]);
        assert_eq!(statuses, [RGFX_STATUS_OK, RGFX_STATUS_OK]);
        let id = |dec: &GraphicsDecoder<FakePlatform>, obj| match dec.object(obj) {
            Some(ObjHandle::DynamicLibrary(library)) => library.id(),
            _ => panic!("no dynamic library {obj:#x}"),
        };
        let (a, b) = (id(&dec, 0x60), id(&dec, 0x61));
        let device = p.device();
        let bytes = compute_with(
            100,
            &[(0x61, None), (0x60, Some("@executable_path/liba.metallib"))],
        );
        assert_eq!(dec.decode(&device, &bytes).unwrap(), Ok(()));
        assert_eq!(replies.lock()[0].status, RGFX_PIPELINE_OK);
        let calls = p
            .call_log()
            .calls_to("newComputePipelineStateWithDescriptor:options:reflection:error:");
        let Arg::Val(logged) = &calls[0].args[0] else {
            panic!("{:?}", calls[0].args);
        };
        let libraries = format!(
            "libraries: DynamicLibrariesDesc {{ preloaded: [{b:?}], loaded: \
             [(\"@executable_path/liba.metallib\", {a:?})] }}"
        );
        assert!(logged.contains(&libraries), "{logged}\nlacks {libraries}");
        // One with none links none.
        assert_eq!(dec.decode(&device, &compute_create(101)).unwrap(), Ok(()));
        let calls = p
            .call_log()
            .calls_to("newComputePipelineStateWithDescriptor:options:reflection:error:");
        let Arg::Val(logged) = &calls[1].args[0] else {
            panic!("{:?}", calls[1].args);
        };
        assert!(
            logged.contains("libraries: DynamicLibrariesDesc { preloaded: [], loaded: [] }"),
            "{logged}"
        );
    }

    /// A run naming an object that is no dynamic library of the sender's is a bad reference; a
    /// run past the message's head, or a name past the message, malformed; none makes a pipeline.
    #[test]
    fn a_compute_create_s_bad_dynamic_libraries_make_nothing() {
        let (p, mut dec, _, _) = made(&[create(0x60, 1)]);
        let device = p.device();
        let creates = |p: &FakePlatform| {
            p.call_log()
                .calls_to("newComputePipelineStateWithDescriptor:options:reflection:error:")
                .len()
        };
        assert_eq!(
            dec.decode(&device, &compute_with(100, &[(1, None)]))
                .unwrap(),
            Err(DecodeError::BadReference)
        );
        assert_eq!(
            dec.decode(&device, &compute_with(101, &[(0x99, Some("x"))]))
                .unwrap(),
            Err(DecodeError::BadReference)
        );
        let mut long = compute_with(102, &[(0x60, None)]);
        edit_head(&mut long, |m: &mut RgfxComputePipelineCreate| {
            m.library_count = 1000
        });
        assert_eq!(
            dec.decode(&device, &long).unwrap(),
            Err(DecodeError::Malformed)
        );
        let mut past = compute_with(103, &[(0x60, Some("name"))]);
        let at = past.len() - size_of::<RgfxPipelineLibrary>();
        let mut record = RgfxPipelineLibrary::read_from_bytes(&past[at..]).unwrap();
        record.name_len = 0x1000;
        past[at..].copy_from_slice(record.as_bytes());
        assert_eq!(
            dec.decode(&device, &past).unwrap(),
            Err(DecodeError::Malformed)
        );
        assert_eq!(creates(&p), 0);
    }
}
