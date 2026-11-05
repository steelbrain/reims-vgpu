//! Test scaffolding: the manager as a recording fake over `FakePlatform` ([`FakeOps`]), the
//! guest memory the walks read and write, and the test layout and commands. The state kinds'
//! creator tests use it too.

use std::cell::RefCell;
use std::sync::Arc;

use zerocopy::{FromBytes, Immutable, IntoBytes};

use super::records::{
    ConcurrentDispatch, Dispatch, DrawPrimitives, NewIcbCommand, ObjectIcb,
    SerializedIcbDescriptor, Triple,
};
use super::{IcbWalkOps, ManagerOps};
use crate::guest::GuestPtr;
use crate::guest::records::ObjectEntry;
use crate::memory::cursor::{Cursor, WritableCursor};
use crate::platform::Handle;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Buffer;
use crate::platform::values::IcbDesc;
use crate::raise::RaiseResult;
use crate::raise::death::DeathSite;
use crate::raise::{OnFault, Raise};
use crate::resources::icb::{IndirectCommandBufferFlags, IndirectCommandLayout};
use crate::resources::manager::lookup::NoData;
use crate::resources::manager::records::{
    IndirectCommandBufferBind, OperationHeader, ResourceAndOffset,
};
use crate::rt::Hold;
use crate::testkit::calllog::Arg;
use crate::testkit::guest_ram::RawRegion;
use crate::testkit::{CallLog, FakePlatform, Fk, kind};

pub(crate) use crate::resources::resource::test_support::logged;

/// A scripted answer of one of the manager's lookups.
pub(crate) enum Lookup<T> {
    /// The object (nil when `None`).
    Found(Option<T>),
    /// The lookup's error.
    Missing(&'static str),
    /// A raise inside the lookup (a creation's, a resource with no buffer).
    Raise(&'static str),
}

impl<T: Handle> Lookup<T> {
    fn answer(&self, sel: &'static str) -> RaiseResult<Result<Option<T>, String>> {
        match self {
            Lookup::Found(o) => Ok(Ok(o.as_ref().map(Handle::clone_ref))),
            Lookup::Missing(s) => Ok(Err((*s).to_owned())),
            Lookup::Raise(r) => Err(crate::testkit::capture::foreign(OnFault::Return, sel, *r)),
        }
    }
}

/// What `new_icb_resource` was given.
#[derive(Debug)]
pub(crate) struct MadeIcb {
    pub descriptor: Vec<u8>,
    pub icb: Option<Fk<kind::Icb>>,
    pub layout: IndirectCommandLayout,
    pub flags: IndirectCommandBufferFlags,
}

/// A resource the address lookup finds at `start`; `buffer: None` is a resource whose buffer
/// getter raises (not a buffer).
pub(crate) struct Resident {
    pub start: u64,
    pub reference: u32,
    pub buffer: Option<Fk<kind::Buffer>>,
}

/// The manager as a recording fake: every call to it is noted in `calls`, in order.
pub(crate) struct FakeOps {
    pub p: FakePlatform,
    pub dev: Fk<kind::Device>,
    pub calls: RefCell<Vec<String>>,
    pub data: Option<Vec<u8>>,
    /// Why `get_data` gives nothing while `data` is `None`.
    pub no_data: NoData,
    pub register: Result<(), String>,
    pub buffers: Vec<(u32, Lookup<Fk<kind::Buffer>>)>,
    pub renders: Vec<(u32, Lookup<Fk<kind::RenderPipeline>>)>,
    pub computes: Vec<(u32, Lookup<Fk<kind::ComputePipeline>>)>,
    pub pipeline_refs: Vec<(u64, u32)>,
    pub resident: Vec<Resident>,
    pub functions: Vec<(u64, Fk<kind::Function>)>,
    /// Function references whose lookup raises (a task cursor raising).
    pub function_throws: Vec<u64>,
    pub icb_nil: bool,
}

impl FakeOps {
    pub(crate) fn new() -> FakeOps {
        let p = FakePlatform::new();
        let dev = p.device();
        FakeOps {
            p,
            dev,
            calls: RefCell::new(Vec::new()),
            data: None,
            no_data: NoData {
                hole: false,
                why: "no data".to_owned(),
            },
            register: Ok(()),
            buffers: Vec::new(),
            renders: Vec::new(),
            computes: Vec::new(),
            pipeline_refs: Vec::new(),
            resident: Vec::new(),
            functions: Vec::new(),
            function_throws: Vec::new(),
            icb_nil: false,
        }
    }

    pub(crate) fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }

    fn note(&self, s: String) {
        self.calls.borrow_mut().push(s);
    }

    pub(crate) fn log(&self) -> &CallLog {
        self.p.call_log()
    }

    pub(crate) fn buffer(&self, len: u64) -> Fk<kind::Buffer> {
        self.dev
            .new_buffer_of_length(OnFault::Die, len, 0)
            .unwrap()
            .unwrap()
    }

    /// A buffer found by reference and, at its GPU address, by address.
    pub(crate) fn resident_buffer(&mut self, reference: u32, len: u64) -> Fk<kind::Buffer> {
        let b = self.buffer(len);
        self.buffers
            .push((reference, Lookup::Found(Some(b.clone_ref()))));
        self.resident.push(Resident {
            start: b.gpu_address(),
            reference,
            buffer: Some(b.clone_ref()),
        });
        b
    }

    pub(crate) fn render_pipeline(&self) -> Fk<kind::RenderPipeline> {
        crate::testkit::impls::new_state::<kind::RenderPipeline>(self.p.world(), None)
    }

    pub(crate) fn compute_pipeline(&self) -> Fk<kind::ComputePipeline> {
        crate::testkit::impls::new_state::<kind::ComputePipeline>(self.p.world(), None)
    }

    pub(crate) fn icb(&self, size: u64) -> Fk<kind::Icb> {
        self.dev
            .new_indirect_command_buffer(&IcbDesc::default(), size, 0)
            .unwrap()
            .unwrap()
    }
}

impl IcbWalkOps<FakePlatform> for FakeOps {
    fn get_pipeline_ref(&self, identifier: u64) -> Result<u32, String> {
        self.note(format!("get_pipeline_ref({identifier})"));
        self.pipeline_refs
            .iter()
            .find(|(i, _)| *i == identifier)
            .map(|(_, r)| *r)
            .ok_or_else(|| "Unknown pipeline".to_owned())
    }

    fn get_buffer_object(
        &self,
        _hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<Fk<kind::Buffer>>, String>> {
        self.note(format!("get_buffer_object({reference})"));
        match self.buffers.iter().find(|(r, _)| *r == reference) {
            Some((_, l)) => l.answer("getBuffer"),
            None => Ok(Err("Invalid reference".to_owned())),
        }
    }

    fn get_render_pipeline_object(
        &self,
        _hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<Fk<kind::RenderPipeline>>, String>> {
        self.note(format!("get_render_pipeline_object({reference})"));
        match self.renders.iter().find(|(r, _)| *r == reference) {
            Some((_, l)) => l.answer("newRenderPipelineState"),
            None => Ok(Err("Invalid reference".to_owned())),
        }
    }

    fn get_compute_pipeline_object(
        &self,
        _hold: Hold<'_>,
        reference: u32,
    ) -> RaiseResult<Result<Option<Fk<kind::ComputePipeline>>, String>> {
        self.note(format!("get_compute_pipeline_object({reference})"));
        match self.computes.iter().find(|(r, _)| *r == reference) {
            Some((_, l)) => l.answer("newComputePipelineState"),
            None => Ok(Err("Invalid reference".to_owned())),
        }
    }

    fn resource_and_offset_at(
        &self,
        hold: Hold<'_>,
        address: u64,
    ) -> Result<ResourceAndOffset<FakePlatform>, String> {
        self.note(format!("resource_and_offset_at({address:#x})"));
        super::lookup::resource_and_offset_for_address::<FakePlatform, Option<Fk<kind::Buffer>>>(
            hold,
            address,
            |a| {
                self.resident
                    .iter()
                    .filter(|r| r.start <= a)
                    .max_by_key(|r| r.start)
                    .map(|r| (r.start, r.reference))
            },
            |reference| {
                self.resident
                    .iter()
                    .find(|r| r.reference == reference)
                    .map(|r| Some(r.buffer.as_ref().map(Handle::clone_ref)))
            },
            |b| match b {
                Some(b) => Ok(Some(b.clone_ref())),
                None => Err(not_a_buffer()),
            },
        )
    }
}

impl ManagerOps<FakePlatform> for FakeOps {
    type IcbResource = MadeIcb;

    fn platform(&self) -> &FakePlatform {
        &self.p
    }

    fn device(&self) -> Option<&Fk<kind::Device>> {
        Some(&self.dev)
    }

    fn function_with_function_ref(
        &self,
        _hold: Hold<'_>,
        function_ref: u64,
    ) -> RaiseResult<Option<Fk<kind::Function>>> {
        self.note(format!("function_with_function_ref({function_ref})"));
        if self.function_throws.contains(&function_ref) {
            return Err(crate::testkit::capture::raise(
                OnFault::Return,
                "function lookup",
                "function lookup",
            ));
        }
        Ok(self
            .functions
            .iter()
            .find(|(r, _)| *r == function_ref)
            .map(|(_, f)| f.clone_ref()))
    }

    fn get_data(&self, _hold: Hold<'_>, offset: u64, length: u64) -> Result<Vec<u8>, NoData> {
        self.note(format!("get_data({offset:#x}, {length:#x})"));
        self.data.clone().ok_or_else(|| self.no_data.clone())
    }

    fn register_pipeline(&self, identifier: u64, reference: u32) -> Result<(), String> {
        self.note(format!("register_pipeline({identifier}, {reference})"));
        self.register.clone()
    }

    fn new_icb_resource(
        &self,
        descriptor: &[u8],
        icb: Option<Fk<kind::Icb>>,
        layout: IndirectCommandLayout,
        flags: IndirectCommandBufferFlags,
    ) -> RaiseResult<Option<MadeIcb>> {
        self.note("new_icb_resource".to_owned());
        if self.icb_nil {
            return Ok(None);
        }
        Ok(Some(MadeIcb {
            descriptor: descriptor.to_vec(),
            icb,
            layout,
            flags,
        }))
    }
}

/// The raise of a resource that has no buffer.
pub(crate) fn not_a_buffer() -> Raise {
    crate::testkit::capture::foreign(OnFault::Return, "getBuffer", "not a buffer")
}

/// The site a death of `kind` records.
pub(crate) fn death(kind: super::DeathKind) -> DeathSite {
    kind.site()
}

/// A cursor over a copy of `bytes` (the region lives as long as the cursor).
pub(crate) fn cursor(bytes: &[u8]) -> Cursor {
    let r = RawRegion::new(bytes.len() as u64).unwrap();
    assert!(r.write(0, bytes));
    let addr = r.addr();
    Cursor::with_pointer(r, addr, bytes.len() as u64)
}

/// A writable cursor over `len` zero bytes, and the region to read back.
pub(crate) fn writable(len: u64) -> (Arc<RawRegion>, WritableCursor) {
    let r = RawRegion::new(len).unwrap();
    let addr = r.addr();
    (r.clone(), WritableCursor::with_pointer(r, addr, len))
}

/// A 16-byte placement (the resource index at 0, the resource ID at 8) and its pointer.
pub(crate) fn placement(index: u64, id: u64) -> (Arc<RawRegion>, GuestPtr) {
    let r = RawRegion::new(16).unwrap();
    assert!(r.write_u64(0, index));
    assert!(r.write_u64(8, id));
    let p = GuestPtr::new(r.addr());
    (r, p)
}

pub(crate) fn entry(ty: u32) -> ObjectEntry {
    ObjectEntry {
        type_and_length: ty,
        offset: 0,
    }
}

pub(crate) fn header(command_type: u32) -> Vec<u8> {
    OperationHeader {
        command_type,
        field4: 0,
    }
    .as_bytes()
    .to_vec()
}

pub(crate) fn bytes<T: IntoBytes + Immutable>(v: &T) -> Vec<u8> {
    v.as_bytes().to_vec()
}

pub(crate) fn read_back<T: FromBytes>(r: &RawRegion, off: u64) -> T {
    let v = r.read_vec(off, size_of::<T>() as u64).unwrap();
    T::read_from_bytes(&v).unwrap()
}

/// The selectors sent to objects of `class`, in order.
pub(crate) fn sels_of(log: &CallLog, class: &str) -> Vec<&'static str> {
    log.calls_of(class).into_iter().map(|c| c.sel).collect()
}

// ---- The walks' test layout and commands

/// Command types the tests use.
pub(crate) const DRAW: u32 = 1;
pub(crate) const DRAW_INDEXED: u32 = 2;
pub(crate) const DRAW_PATCHES: u32 = 4;
pub(crate) const DRAW_INDEXED_PATCHES: u32 = 8;
pub(crate) const CONCURRENT_DISPATCH: u32 = 0x20;
pub(crate) const CONCURRENT_DISPATCH_THREADS: u32 = 0x40;

/// The test layout: every record at its own offset of a 0x1a0-byte command.
pub(crate) fn layout() -> IndirectCommandLayout {
    IndirectCommandLayout {
        command_type_offset: 0,
        barrier_offset: 4,
        kernel_dispatch_arguments_offset: 0x160,
        tessellation_factor_offset: 0x70,
        pipeline_state_offset: 8,
        vertex_buffer_bind_offset: 0x10,
        fragment_buffer_bind_offset: 0x40,
        object_buffer_bind_offset: 0,
        mesh_buffer_bind_offset: 0,
        kernel_buffer_bind_offset: 0x100,
        attribute_stride_offset: 0x58,
        object_threadgroup_memory_length_offset: 0,
        threadgroup_memory_length_offset: 0x140,
        command_arguments_offset: 0x90,
        command_size: 0x1a0,
    }
}

/// The test layout's command size.
pub(crate) const CMD: usize = 0x1a0;

pub(crate) fn render_flags(bits: u32) -> IndirectCommandBufferFlags {
    IndirectCommandBufferFlags {
        bits,
        command_types: DRAW,
        max_vertex_buffer_bind_count: 2,
        max_fragment_buffer_bind_count: 1,
        max_kernel_buffer_bind_count: 0,
        max_kernel_threadgroup_memory_bind_count: 0,
    }
}

pub(crate) fn compute_flags(bits: u32) -> IndirectCommandBufferFlags {
    IndirectCommandBufferFlags {
        bits,
        command_types: CONCURRENT_DISPATCH,
        max_vertex_buffer_bind_count: 0,
        max_fragment_buffer_bind_count: 0,
        max_kernel_buffer_bind_count: 2,
        max_kernel_threadgroup_memory_bind_count: 2,
    }
}

/// `n` commands of the test layout, zeroed.
pub(crate) struct Cmds(pub Vec<u8>);

impl Cmds {
    pub(crate) fn new(n: usize) -> Cmds {
        Cmds(vec![0; n * CMD])
    }
    pub(crate) fn put<T: IntoBytes + Immutable>(&mut self, i: usize, off: u64, v: &T) -> &mut Cmds {
        let at = i * CMD + off as usize;
        let b = v.as_bytes();
        self.0[at..at + b.len()].copy_from_slice(b);
        self
    }
    pub(crate) fn ty(&mut self, i: usize, t: u32) -> &mut Cmds {
        self.put(i, 0, &t)
    }
    pub(crate) fn cursor(&self) -> Cursor {
        cursor(&self.0)
    }
}

/// The `Ok` of a lookup (`ResourceAndOffset` has no `Debug`).
pub(crate) fn ok<T>(r: Result<T, String>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("unexpected error: {e}"),
    }
}

/// The `Err` of a lookup.
pub(crate) fn err<T>(r: Result<T, String>) -> String {
    match r {
        Ok(_) => panic!("unexpected success"),
        Err(e) => e,
    }
}

pub(crate) fn bind(reference: u32, gpu_address: u64, offset: u64) -> IndirectCommandBufferBind {
    IndirectCommandBufferBind {
        buffer_reference: reference,
        gpu_address,
        offset,
    }
}

pub(crate) fn render_cmds(o: &FakeOps) -> Vec<&'static str> {
    sels_of(o.log(), "MTLIndirectRenderCommand")
}

pub(crate) fn compute_cmds(o: &FakeOps) -> Vec<&'static str> {
    sels_of(o.log(), "MTLIndirectComputeCommand")
}

pub(crate) fn last_args(o: &FakeOps, sel: &str) -> Vec<Arg> {
    o.log().calls_to(sel).last().unwrap().args.clone()
}

/// A draw at slot `i`: pipeline 9, two vertex binds (references 5, 6) with strides, one
/// fragment bind (reference 5), a barrier.
pub(crate) fn draw_command(c: &mut Cmds, i: usize) {
    c.ty(i, DRAW)
        .put(i, 4, &1u32)
        .put(i, 8, &9u32)
        .put(i, 0x10, &bind(5, 0, 0x10))
        .put(i, 0x24, &bind(6, 0, 0x20))
        .put(i, 0x40, &bind(5, 0, 8))
        .put(i, 0x58, &0x30u64)
        .put(i, 0x60, &0x40u64)
        .put(
            i,
            0x90,
            &DrawPrimitives {
                primitive_type: 3,
                vertex_start: 1,
                vertex_count: 2,
                instance_count: 4,
                base_instance: 5,
                field_22: [0; 2],
            },
        );
}

/// A compute command at slot `i` of type `ty`: pipeline 9, kernel binds (5, 6), strides, two
/// threadgroup lengths, a dispatch.
pub(crate) fn compute_command(c: &mut Cmds, i: usize, ty: u32) {
    c.ty(i, ty)
        .put(i, 8, &9u32)
        .put(i, 0x100, &bind(5, 0, 0x10))
        .put(i, 0x114, &bind(6, 0, 0x20))
        .put(i, 0x58, &0x30u64)
        .put(i, 0x60, &0x40u64)
        .put(i, 0x140, &0x100u64)
        .put(i, 0x148, &0x200u64)
        .put(
            i,
            0x90,
            &ConcurrentDispatch {
                grid: Triple { a: 1, b: 2, c: 3 },
                threads_per_threadgroup: Triple { a: 4, b: 5, c: 6 },
            },
        )
        .put(
            i,
            0x160,
            &Dispatch {
                stage_in_region_size: Triple { a: 7, b: 8, c: 9 },
                stage_in_region_origin: Triple { a: 1, b: 1, c: 0 },
                imageblock_width: 16,
                imageblock_height: 32,
            },
        );
}

/// A creation command with render types and the test layout.
pub(crate) fn icb_command(max_count: u32, options: u16) -> NewIcbCommand {
    NewIcbCommand {
        descriptor: SerializedIcbDescriptor {
            command_types: DRAW | DRAW_INDEXED,
            max_vertex_buffer_bind_count: 3,
            max_fragment_buffer_bind_count: 4,
            max_kernel_buffer_bind_count: 5,
            max_kernel_threadgroup_memory_bind_count: 6,
            // pipeline (0), stride (3) and ray tracing (2) set; buffers (1) clear.
            bits: 0b1101,
            ..Default::default()
        },
        command_layout: layout(),
        max_command_count: max_count,
        options,
        field_4e: [0; 2],
    }
}

/// The object record, the header and the command, as the creation reads them.
pub(crate) fn icb_object(command_type: u32, command: &NewIcbCommand) -> Vec<u8> {
    let mut v = bytes(&ObjectIcb {
        length: 0x1234,
        page: 7,
    });
    v.extend(header(command_type));
    v.extend(bytes(command));
    v
}
