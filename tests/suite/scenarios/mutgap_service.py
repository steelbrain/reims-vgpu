"""The service's and the tasks' range checks and paging at their edges.

Most of these lock an edge the other scenarios keep clear of: an object whose
pages lie past the task's length but over valid leaves (the service's range
checks are all that refuses it, since the page-table walk knows nothing of
the task's length), data that ends exactly at the task's end, data that
crosses a page, and buffers over read-only leaves."""
import os
import struct

from pgtest import abi
from pgtest.abi import PAGE
from pgtest.compute_objects import Compute as C, Rig, function_record
from pgtest.covgap_svc_util import point, punch
from pgtest.registry import scenario
from pgtest.resources_build import (RGBA8Unorm, RTask, b_copy_buf_buf, b_copy_tex_buf, b_fill_buf, buffer_rec,
                                    desc1, simple_dim, texture_rec)
from pgtest.scenario import FIXTURES, Scenario
from pgtest.task import seg
from pgtest.tasks_lib import exec3_payload, fault_and_resume, fill_rec

PATTERN = bytes((i * 7 + 3) & 0xFF for i in range(0x100))
TEXELS = bytes((i * 5 + 1) & 0xFF for i in range(4 * 4 * 4))
# The colour and alpha planes of the 550 texture a copy reads, unlike the
# bytes of the texture it writes.
XR10_SRC_COLOR = bytes((i * 53 + 29) & 0xFF for i in range(0x400))
XR10_SRC_ALPHA = bytes((i * 19 + 0x55) & 0xFF for i in range(0x400))
KERNEL = "shaders/kernel.metallib"
KERNEL_LEN = 3488  # tests/suite/fixtures/shaders/kernel.metallib


def past_end_task(s):
    """A task of 20 pages of leaves whose length is 16 pages: VA pages 16-19
    have valid leaves but lie past the task's end (RTask's layout: the
    object list at page 0, records 1-3, command buffers 4-7)."""
    return RTask(s, pages=20, length=16 * PAGE)


@scenario(setups=("isolate",))
def mutgap_svc_texture_past_end():
    """A normal texture (type 2, 4x4 RGBA8Unorm) whose guest pages are VA
    page 16 of a task whose length is 16 pages, where the page table still
    has a valid leaf. Paging it in makes a transfer buffer over those pages,
    whose mapping refuses a range past the task's end ("Address (...) out of
    range for task"), so the exec that copies the texture into a shared
    buffer faults and the buffer stays zero. A later exec on the task still
    runs: the service lives. Isolate only: the in-process tasks page textures
    through their own mapping, which the tasks scenarios lock."""
    s = Scenario("mutgap_svc_texture_past_end", setups=("isolate",), doc=mutgap_svc_texture_past_end.__doc__)
    s.bring_up()
    t = past_end_task(s)
    t.write(16 * PAGE, TEXELS)
    x = t.add(abi.T_TEXTURE, texture_rec(PAGE, 16, [simple_dim(4, 4)], desc1(fmt=RGBA8Unorm)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_tex_buf(x, b, 0, 16, 64, (4, 4, 1)), resources=[(x, 0), (b, 0)], fault=True,
           label="copy the texture past the task's end")
    t.dump(b_va, 0x40, "B")
    t.blit(b_fill_buf(b, 0x40, 0x10, 0x5A), resources=[(b, 0)], label="the task still runs")
    t.dump(b_va, 0x60, "B, after")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap_svc_buffer_past_end():
    """A shared buffer (type 1, flags 0) at VA page 17 of a task whose length
    is 16 pages, over a valid leaf holding a pattern. On this host the
    service backs buffers by detaching them and replacing their backing with
    the task's address ranges, and those ranges refuse a range past the
    task's end: the copy out of it faults and the probe stays zero. The task
    still runs after. Isolate only: in-process buffers are mapped by the
    task."""
    s = Scenario("mutgap_svc_buffer_past_end", setups=("isolate",), doc=mutgap_svc_buffer_past_end.__doc__)
    s.bring_up()
    t = past_end_task(s)
    t.write(17 * PAGE, PATTERN)
    a = t.add(abi.T_BUFFER, buffer_rec(PAGE, 17))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_copy_buf_buf(a, b, 0, 0, 0x100), resources=[(a, 0), (b, 0)], fault=True,
           label="copy out of a buffer past the task's end")
    t.dump(b_va, 0x100, "B")
    t.blit(b_fill_buf(b, 0x100, 0x10, 0x5B), resources=[(b, 0)], label="the task still runs")
    t.dump(b_va + 0x100, 0x10, "B+0x100")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",))
def mutgap_svc_cmdbuf_past_end():
    """A command buffer at VA 16 * PAGE of a task whose length is 16 pages,
    over a valid leaf holding a well-formed blit segment. The service's
    cursor refuses a range past the task's end ("Out of range cursor
    request") and returns a null cursor, which nothing checks
    (exec_null_cursor_past_end): the task dies (a death in its own code,
    which ends only the task), the exec faults and so does the next, and
    the fill never lands. Isolate only: in-process the
    range is copied, and the copy's range check refuses it the same way,
    which exec_null_cursor locks."""
    s = Scenario("mutgap_svc_cmdbuf_past_end", setups=("isolate",), doc=mutgap_svc_cmdbuf_past_end.__doc__)
    s.bring_up()
    t = past_end_task(s)
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0, 8, 0x11), resources=[(b, 0)], label="before")
    data = seg(abi.ENC_BLIT, fill_rec(b, 8, 8, 0x5E))
    t.write(16 * PAGE, data)
    t.exec3([(16 * PAGE, len(data))], resources=[(b, 0)], fault=True, label="a range past the task's end")
    fault_and_resume(s, "a range past the task's end")
    t.dump(b_va, 0x20, "B")
    t.blit(b_fill_buf(b, 0x10, 8, 0x22), resources=[(b, 0)], fault=True, label="the task's service is gone")
    s.events("end")
    return s


def function_case(name, doc, place, fault):
    """A Rig whose kernel.metallib is written where place(rig) says (a VA),
    a function record over it and a pipeline; one exec dispatches
    double_gid over 4 threads into `out`. fault: the exec faults (the
    function is nil and Metal aborts: a death), else out is dumped."""
    r = Rig(name, doc, pages=64)
    out, out_va = r.buffer()
    va = place(r)
    n = r.s.mem_file(r.t.gpa(va), KERNEL) if va % PAGE + KERNEL_LEN <= PAGE else None
    if n is None:
        # The metallib crosses a page: written in two pieces, one per page.
        data = open(os.path.join(FIXTURES, KERNEL), "rb").read()
        r.t.write(va, data)
        n = len(data)
    fn = r.raw(abi.T_FUNCTION, function_record(va, n))
    pso = r.pipeline(fn)
    r.ready()
    r.compute(C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)),
              fault=fault, label="dispatch")
    if fault:
        r.compute(C.set_buffers(0, [(out, 0)]), fault=True, label="the task's service is gone")
        r.s.events("end")
        return r.s
    r.dump(out_va, 0x20, "out")
    return r.end()


@scenario
def mutgap_svc_function_at_task_end():
    """A function whose metallib ends exactly at the task's end (VA 64 * PAGE
    - its length, in a 64-page task). Reading it copies from task memory, and
    that copy refuses a range whose end equals the task's length (end >=
    taskLength, in the service and in-process alike, one byte stricter than
    every other range check): "Failed to get function data", the function is
    nil, and Metal aborts building the pipeline. A death of the host in
    every setup, the isolate one included, since the abort is inside
    Metal; the next exec is never reached."""
    return function_case("mutgap_svc_function_at_task_end", mutgap_svc_function_at_task_end.__doc__,
                         lambda r: 64 * PAGE - KERNEL_LEN, fault=True)


@scenario
def mutgap_svc_function_across_pages():
    """A function whose metallib crosses a page boundary: it starts 0x100
    bytes before the end of a task page. The copy out of task memory walks
    both pages and appends each piece, so the library is whole, and
    double_gid writes out[gid] = 2 gid for 4 threads in every setup."""
    def place(r):
        va = r.t.reserve(2 * PAGE)
        return va + PAGE - 0x100
    return function_case("mutgap_svc_function_across_pages", mutgap_svc_function_across_pages.__doc__,
                         place, fault=False)


@scenario
def mutgap_svc_function_over_hole():
    """A function whose metallib sits on a task page the guest made a hole
    (its leaf 0) after the task was installed: IR the guest has not mapped
    yet. The exec's pipeline names the function; copying its IR
    walks into the hole, and the function's lookup raises ("function lookup
    failed"), so the exec faults in every setup, where a nil function would
    have Metal abort building the pipeline (contrast
    mutgap_svc_function_at_task_end). Nothing is cached: once the guest
    points the leaf back at its page, the same exec makes the function and
    the pipeline, and double_gid writes out[gid] = 2 gid for 4 threads."""
    r = Rig("mutgap_svc_function_over_hole", mutgap_svc_function_over_hole.__doc__, pages=64)
    out, out_va = r.buffer()
    va = r.t.reserve(PAGE)
    n = r.s.mem_file(r.t.gpa(va), KERNEL)
    fn = r.raw(abi.T_FUNCTION, function_record(va, n))
    pso = r.pipeline(fn)
    r.ready()
    vpn = va // PAGE
    punch(r.t, vpn)
    dispatch = (C.set_pipeline(pso), C.set_buffers(0, [(out, 0)]), C.dispatch_threads((4, 1, 1), (4, 1, 1)))
    r.compute(*dispatch, fault=True, label="the function's IR over a hole")
    point(r.t, vpn, r.s.pfn(r.t.data + vpn))
    r.compute(*dispatch, label="the function's IR mapped again")
    r.dump(out_va, 0x20, "out")
    return r.end()


@scenario
def mutgap_svc_readonly_leaf_buffer_write():
    """A shared buffer (type 1, flags 0) over a read-only leaf (bit 31
    clear), holding a pattern, and a blit fill into it. The service backs it
    with the task's address ranges and passes readOnly from the leaves.
    In-process the mapping of a range with a read-only page is read-only too.
    The buffer's guest page is dumped after the fill, and a copy out of it
    shows what the GPU then reads. The setups differ: in the blocks setup the
    host task's mapping is writable and the fill lands in guest RAM; in the
    map setup the read-only mapping and in isolate the read-only backing drop
    the GPU's write silently (no fault), and both the guest page and the copy
    show the pattern."""
    s = Scenario("mutgap_svc_readonly_leaf_buffer_write", doc=mutgap_svc_readonly_leaf_buffer_write.__doc__)
    s.bring_up()
    t = RTask(s, pages=12, writable={v for v in range(12) if v != 9})
    t.write(9 * PAGE, PATTERN)
    a = t.add(abi.T_BUFFER, buffer_rec(PAGE, 9))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(a, 0, 0x10, 0x3C), resources=[(a, 0), (b, 0)], label="fill the buffer")
    t.dump(9 * PAGE, 0x20, "A's page")
    t.blit(b_copy_buf_buf(a, b, 0, 0, 0x20), resources=[(a, 0), (b, 0)], label="copy it out")
    t.dump(b_va, 0x20, "B")
    t.delete()
    s.events("end")
    return s


# ---- tasks: synchronize, object and placement lists ----

def _xr10_dims(w=64, h=4):
    """Two planes of 4 bytes a texel, color at +0 and alpha at +0x4000, as
    textures_gaps' builder."""
    from pgtest.resources_build import dimension, mip
    return [dimension([mip(0, w * 4 * h, w * 4, w, h)], block=(1, 1, 1, 4)),
            dimension([mip(0, w * 4 * h, w * 4, w, h)], base=PAGE, block=(1, 1, 1, 4))]


@scenario(host=True)
def mutgap_svc_xr10_synchronize():
    """SynchronizeResources (53) naming only a compute-paged texture: a 64x4
    texture of pixel format 550 (type 12), written on the GPU by an exec with
    invalidateGuest only (flags 2), so its guest copy is invalid while the
    guest overwrites both planes' first 0x40 bytes with 0xAA. 53 sorts it
    into the compute list and, with no blit-paged resource named, opens a
    compute encoder for it alone: the BlitOutRGB_2P_XR10_A8 kernel writes both
    planes back over the 0xAA. A second 53 finds its guest copy valid and
    leaves a second 0xAA overwrite alone.

    The exec writes the texture twice. 303 copies texels 8-15 of row 1 of a
    second 550 texture, listed with flags 0 and paged in from other guest
    bytes, into texels 4-11 of row 0, inside the dumps. 321 fills texels
    60-63 of rows 2 and 3, outside them: Apple's fill of a 550 texture writes
    bytes that change with every host boot (docs/lessons.md). After the first
    53 the dumps (the first 16 texels of row 0 of each plane) hold what
    BlitOutRGB_2P_XR10_A8 wrote from the texture, so they show the page-out
    ran after the copy, from the texture it wrote. In the colour plane,
    texels 0-3 and 12-15 are the texture's own guest bytes as
    BlitInRGB_2P_XR10_A8 paged them in before the exec, and texels 4-11 the
    source's, each with its top two bits set. The alpha plane reads zero: the
    texture's own alpha plane is zero, and the copied texels' alpha reads
    zero too, not the source's. Keyed by the host: the planes hold the
    library's shaders' arithmetic."""
    from pgtest.resources_build import XR10_2P_A8, b_copy_tex_tex, b_fill_tex_color, cmd_synchronize
    s = Scenario("mutgap_svc_xr10_synchronize", host=True, doc=mutgap_svc_xr10_synchronize.__doc__)
    s.bring_up()
    t = RTask(s, pages=96)
    va = t.area(2 * PAGE, contents=PATTERN * 4)
    x = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, va // PAGE, _xr10_dims(),
                                                    desc1(fmt=XR10_2P_A8, w=64, h=4, usage=3)))
    y_va = t.area(2 * PAGE, contents=XR10_SRC_COLOR)
    t.write(y_va + PAGE, XR10_SRC_ALPHA)
    y = t.add(abi.T_DUAL_PLANE_TEXTURE, texture_rec(2 * PAGE, y_va // PAGE, _xr10_dims(),
                                                    desc1(fmt=XR10_2P_A8, w=64, h=4, usage=3)))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_tex_color(x, (4, 2, 1), (1.0, 0.5, 0.25, 1.0), XR10_2P_A8, origin=(60, 2, 0)),
           b_copy_tex_tex(y, x, (8, 1, 1), sorigin=(8, 1, 0), dorigin=(4, 0, 0)), b_fill_buf(b, 0, 0x10, 0x01),
           resources=[(x, 2), (y, 0), (b, 0)], label="fill it and copy into it, invalidateGuest only")
    t.fill(va, 0x40, 0xAA)
    t.fill(va + PAGE, 0x40, 0xAA)
    cmd_synchronize(s, 1, [x], label="53 naming it")
    t.dump(va, 0x40, "color plane after 53")
    t.dump(va + PAGE, 0x40, "alpha plane after 53")
    t.fill(va, 0x40, 0xAA)
    cmd_synchronize(s, 1, [x], label="53 again, its guest copy valid")
    t.dump(va, 0x40, "color plane after the second 53")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_svc_objlist_high_page():
    """An object list at VA page 12 of a 16-page task, naming a buffer at
    page 13. In the memory-map setups the list is mapped over the pages its
    bytes span, rounded up and counted from its offset: one page, inside the
    task, so the exec's fill lands. The blocks setup maps it through the
    task's own mapping; every setup completes."""
    from pgtest.task import ObjectList
    s = Scenario("mutgap_svc_objlist_high_page", doc=mutgap_svc_objlist_high_page.__doc__)
    s.bring_up()
    t = RTask(s, pages=16)
    t.ol = ObjectList(t, 12 * PAGE, PAGE)
    a = t.add(abi.T_BUFFER, buffer_rec(PAGE, 13))
    t.define()
    t.install()
    t.blit(b_fill_buf(a, 0, 0x10, 0x11), resources=[(a, 0)], label="a fill through a list at page 12")
    t.dump(13 * PAGE, 0x20, "the buffer")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_svc_placement_removed():
    """SetObjectAndPlacementList (68) with a placement list at VA page 4,
    then again with none (offset and length 0). With argument buffers (the
    service) the first maps the placement list writable; the second has no
    new list to copy the old one into and needs none, so it completes, and
    the task root keeps the object list and zero placement fields. In-process
    the manager has no argument buffers and records only the object list. An
    exec then runs through the object list."""
    from pgtest.tasks_lib import XGPUTask, maybe_fault
    s = Scenario("mutgap_svc_placement_removed", doc=mutgap_svc_placement_removed.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12)
    buf, buf_va = t.buffer(length=PAGE, va=8 * PAGE)
    t.define()
    n = t.ol.write()
    maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, 4 * PAGE, n, 16 * 2),
                "68: a placement list at page 4")
    s.dump(s.pa(t.root, 8), 0x20, "the task root's list fields")
    maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, 0, n, 0),
                "68: no placement list")
    s.dump(s.pa(t.root, 8), 0x20, "the task root's list fields, after")
    cb = t.cmdbuf(seg(abi.ENC_BLIT, fill_rec(buf, 0, 0x10, 0x31)))
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [cb], [(buf, 0)]), "an exec")
    t.dump(buf_va, 0x20, "the buffer")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_svc_placement_unmappable_empty_exec():
    """A task's first placement list set over a read-only leaf (68), as in
    service_gap_placement_unmappable, then an ExecIndirect3 with no command
    buffers and no resources, which looks up no object. In the service the
    exec must map the placement list writable first: "Failed to map placement
    list", and the exec faults although it names nothing. In-process there
    are no argument buffers and it completes."""
    from pgtest.tasks_lib import XGPUTask, maybe_fault
    s = Scenario("mutgap_svc_placement_unmappable_empty_exec",
                 doc=mutgap_svc_placement_unmappable_empty_exec.__doc__)
    s.bring_up()
    t = XGPUTask(s, tid=1, pages=12, writable={v for v in range(12) if v != 9})
    t.define()
    n = t.ol.write()
    maybe_fault(s, abi.OP_SET_OBJECT_AND_PLACEMENT_LIST, struct.pack("<IQQII", 1, 0, 9 * PAGE, n, 16 * 2),
                "68: the first placement list, over a read-only leaf")
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1), "an exec of nothing")
    maybe_fault(s, abi.OP_EXEC_INDIRECT3, exec3_payload(1, [t.cmdbuf(seg(abi.ENC_BLIT))]),
                "an exec of one empty blit segment")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_svc_exec2_sync_per_resource():
    """ExecIndirect2 (55) naming two normal textures (type 2, 4x4 RGBA8Unorm,
    private: paged in and out by blit copies in every setup), both filled on
    the GPU: the first with invalidateGuest and synchronize, the second with
    invalidateGuest only. Each resource's synchronize byte is its own (for
    the service too, through the conversion into ExecIndirect3 records and
    the service's completion of the exec): only the first texture's guest
    pages receive its fill. A 53 then synchronizes the second."""
    from pgtest.resources_build import b_fill_tex_color, cmd_synchronize
    s = Scenario("mutgap_svc_exec2_sync_per_resource", doc=mutgap_svc_exec2_sync_per_resource.__doc__)
    s.bring_up()
    t = RTask(s)
    x1_va = t.area(PAGE, contents=b"\xA1" * 0x40)
    x1 = t.add(abi.T_TEXTURE, texture_rec(PAGE, x1_va // PAGE, [simple_dim(4, 4)], desc1()))
    x2_va = t.area(PAGE, contents=b"\xA2" * 0x40)
    x2 = t.add(abi.T_TEXTURE, texture_rec(PAGE, x2_va // PAGE, [simple_dim(4, 4)], desc1()))
    t.define()
    t.install()
    t.reset_cmdbufs()
    cb = t.cmdbuf(seg(abi.ENC_BLIT, b_fill_tex_color(x1, (4, 1, 1), (1.0, 0.0, 0.0, 1.0), RGBA8Unorm)
                      + b_fill_tex_color(x2, (4, 1, 1), (0.0, 1.0, 0.0, 1.0), RGBA8Unorm)))
    t.exec2([cb], resources=[(x1, 0, 1, 1), (x2, 0, 1, 0)], label="55: X1 synchronized, X2 not")
    t.dump(x1_va, 0x20, "X1's guest pages")
    t.dump(x2_va, 0x20, "X2's guest pages")
    cmd_synchronize(s, 1, [x2], label="53 of X2")
    t.dump(x2_va, 0x20, "X2's guest pages after 53")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_svc_unmap_before_objlist():
    """An UnmapMemory (34) of the page just before the object list's, which
    ends exactly where the list starts, after the guest moved the list page's
    leaf to another guest page holding a different list. The unmap does not
    touch the list (the overlap test is strict), so in the memory-map setups
    the list stays mapped over its old guest page and reference 2, made after
    the move, is the old list's buffer at page 10. In the blocks setup the
    list is read through the host task's mapping, which the guest's leaf does
    not change either. The service keeps its own mapping likewise."""
    from pgtest.task import ObjectList
    s = Scenario("mutgap_svc_unmap_before_objlist", doc=mutgap_svc_unmap_before_objlist.__doc__)
    s.bring_up()
    t = RTask(s, pages=12)
    t.ol = ObjectList(t, 2 * PAGE, PAGE)
    a = t.ol.add(abi.T_BUFFER, buffer_rec(PAGE, 8))
    b = t.ol.add(abi.T_BUFFER, buffer_rec(PAGE, 10))
    t.define()
    t.install()
    other = s.alloc_page()
    alt = ObjectList(t, 0, PAGE)
    alt.entries = list(t.ol.entries)
    alt.entries[b] = (abi.T_BUFFER, 16, t.ol.records_va + 0x40)  # a record naming page 9
    t.write(t.ol.records_va + 0x40, buffer_rec(PAGE, 9))
    s.mem(s.pa(other), b"".join(abi.OBJ_ENTRY.pack(ty | (n << 8), va) for ty, n, va in alt.entries))
    t.blit(b_fill_buf(a, 0, 0x10, 0x11), resources=[(a, 0)], label="A, the list mapped")
    s.u32(s.pa(t.pt_root, 4 * 2), s.pfn(other) | abi.PTE_WRITABLE)  # VA page 2's leaf
    t.unmap(1 * PAGE, PAGE)
    t.blit(b_fill_buf(b, 0, 0x10, 0x22), resources=[(b, 0)], label="reference 2, after the move and the unmap")
    t.dump(8 * PAGE, 0x10, "page 8")
    t.dump(9 * PAGE, 0x10, "page 9")
    t.dump(10 * PAGE, 0x10, "page 10")
    t.delete()
    s.events("end")
    return s


@scenario(host=True)
def mutgap_svc_compute_info_list_unmappable():
    """GetComputeInfo (59) of a pipeline already made, after the task's
    object list was moved onto a hole in its page table (51 only records it).
    In-process the command maps the object lists first and throws "Failed to
    map object list" when it cannot, although the pipeline is cached and
    would be found without the list: the command faults and the reply keeps
    its 0xee. The service maps the lists first too, and faults the same way.
    With the list moved back, the same query writes the reply. Keyed by the
    host: the reply holds the host GPU's limits."""
    from pgtest.tasks_lib import maybe_fault, set_list_raw
    r = Rig("mutgap_svc_compute_info_list_unmappable", mutgap_svc_compute_info_list_unmappable.__doc__,
            host=True, pages=64)
    r.t.holes.add(40)
    reply = r.t.reserve(PAGE)
    pso = r.kernel()
    r.t.fill(reply, 0x100, 0xee)
    # VA page 40 is a hole: its leaf is cleared (an entry of 0 is a hole).
    r.s.u32(r.s.pa(r.t.pt_root, 4 * 40), 0)
    r.ready()

    def info(off):
        return struct.pack("<IIIIQ", r.t.tid, pso, 5, 4, reply + off)
    r.s.submit(abi.OP_GET_COMPUTE_INFO, info(0), label="59, the list mapped")
    set_list_raw(r.s, 1, 40, r.t.ol.length())
    maybe_fault(r.s, abi.OP_GET_COMPUTE_INFO, info(0x40), "59, the list on a hole")
    set_list_raw(r.s, 1, 0, r.t.ol.length())
    r.s.submit(abi.OP_GET_COMPUTE_INFO, info(0x80), label="59, the list moved back")
    r.dump(reply, 0xc0, "replies, 0x40 apart")
    return r.end()


@scenario
def mutgap_svc_info_prepares():
    """InfoIndirect (69) naming a private buffer (type 1, flags bit 0: paged
    in by a blit copy) never used before, with one empty info segment.
    InfoIndirect prepares its resources with no event to wait for (its
    prepare encodes a wait only for an event), so the paging command buffer
    is committed alone and 69 completes. An exec then copies the private
    buffer into a shared one: the guest bytes 69's prepare paged in."""
    s = Scenario("mutgap_svc_info_prepares", doc=mutgap_svc_info_prepares.__doc__)
    s.bring_up()
    t = RTask(s)
    p_va = t.area(PAGE, contents=PATTERN)
    p = t.add(abi.T_BUFFER, buffer_rec(PAGE, p_va // PAGE, flags=1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.reset_cmdbufs()
    t.exec3([t.cmdbuf(seg(abi.ENC_INFO))], resources=[(p, 0)], opcode=abi.OP_INFO_INDIRECT,
            label="69 naming a private buffer")
    t.blit(b_copy_buf_buf(p, b, 0, 0, 0x40), resources=[(p, 0), (b, 0)], label="copy it out")
    t.dump(b_va, 0x40, "B")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("map", "isolate"))
def mutgap_svc_mapper_below_v10():
    """A mapper-ref buffer (type 15) over a mapped surface when the guest
    wrote version 6, below 10, where the device does not set
    mapperIOSurfaces. In-process the task asserts the feature before asking
    the mapper service for the surface (docs/testing.md, section 7.4): the
    exec that copies out of the buffer (opcode 55, as 67 needs version 109)
    kills the host. In isolate the service checks no feature and asks the
    host instead: the buffer is made at version 6 and the copy shows the
    surface's page. Map and isolate setups only: the IOSurface host device
    needs a memory map."""
    from pgtest.iosfc_guest import CMD_MAP, IOSFC, desc_v2
    from pgtest.resources_build import mapper_ref_rec
    s = Scenario("mutgap_svc_mapper_below_v10", setups=("map", "isolate"), iosfc=True,
                 doc=mutgap_svc_mapper_below_v10.__doc__)
    s.bring_up(version=6)
    s.pg_version = 6
    d = IOSFC(s, v2=True)
    d.start()
    root, data = d.surface_pages(1)
    s.mem(s.pa(data), bytes((i * 3 + 1) & 0xFF for i in range(0x100)))
    d.put_desc(1, desc_v2(root, 64, 16, fmt=b"BGRA", bpe=4))
    d.submit((CMD_MAP, 1))
    t = RTask(s)
    m = t.add(abi.T_MAPPER_BUFFER, mapper_ref_rec(1))
    b, b_va = t.shared(PAGE)
    t.define()
    t.install()
    t.blit(b_fill_buf(b, 0x40, 0x10, 0x5C), resources=[(b, 0)], label="an exec without it")
    t.blit(b_copy_buf_buf(m, b, 0, 0, 0x40), resources=[(m, 0), (b, 0)], label="copy out of the mapper buffer")
    t.dump(b_va, 0x50, "B")
    t.delete()
    s.events("end")
    return s


@scenario
def mutgap_svc_present_not_a_backing():
    """A present (opcode 7) of a task-0 reference that is a normal texture,
    not an IOSurface backing. Finding the backing makes the texture (the
    object list lookup) and then refuses it by class before preparing it
    ("Failed to find backing"): the present is logged and its stamp signals,
    and no frame reaches the display. The texture was made but never
    prepared, so it is host-invalid: after the guest rewrites its pages, 52
    on task 0 marks its guest copy invalid and 53 finds nothing to
    synchronize (a resource needs a sync only when host-valid), and the
    rewritten pages stay. (Execs cannot name task 0.) Task 0 is in-process in
    every setup."""
    from pgtest.display_guest import DisplayGuest, display, p_present3
    from pgtest.resources_build import cmd_invalidate, cmd_synchronize
    s = Scenario("mutgap_svc_present_not_a_backing", displays=[display(0)],
                 doc=mutgap_svc_present_not_a_backing.__doc__)
    g = DisplayGuest(s)
    t = g.t0
    x_va = t.reserve(PAGE)
    t.write(x_va, TEXELS)
    x = t.ol.add(abi.T_TEXTURE, texture_rec(PAGE, x_va // PAGE, [simple_dim(4, 4)], desc1()))
    g.install()
    g.connect(0, enabled=0xF)
    g.ack(0, 1)
    s.submit(abi.OP_DISPLAY_TRANSACTION3, p_present3(0, x), label="present a normal texture")
    s.fault_state("the present")
    s.events("after the present")
    t.write(x_va, TEXELS[::-1])
    cmd_invalidate(s, 0, [(x, 0, 0, 1, 0)], label="52 on task 0: its guest copy invalid")
    cmd_synchronize(s, 0, [x], label="53 on task 0")
    t.dump(x_va, 0x40, "the texture's pages")
    s.api("present_counts", display=0)
    s.events("end")
    return s


def _placement_list(t, places):
    """The object list and a placement list on its own page, entry i from
    places.get(i, (0, 0)), set with 68 (as textures_placement's
    install_with_placements). Returns the placement list's VA."""
    n = len(t.ol.entries)
    va = t.reserve(PAGE)
    t.write(va, b"".join(abi.PLACEMENT.pack(*places.get(i, (0, 0))) for i in range(n)))
    t.ol.write()
    t.set_object_and_placement_list(0, va, t.ol.length(), 16 * n)
    return va


@scenario(setups=("isolate",), host=True)
def mutgap_svc_placement_reserved_index():
    """A placement list that already names a texture's place when the service
    first maps it: entry X0 = {resourceIndex 1, gpuResourceID 1} for a
    texture not yet made, as after a restore. The service's first mapping of
    the lists rebuilds the task: every texture entry with a nonzero ID has
    its index reserved in Metal. So a texture X1 made next with a zero entry
    gets another index than the first free one (textures_placement_written:
    the first texture made in a fresh service gets 1), and X0, made after
    with its forced index 1, is then made in the reserved slot. Both entries
    are dumped. Isolate only (argument buffers); host-keyed (Metal's
    indices)."""
    s = Scenario("mutgap_svc_placement_reserved_index", setups=("isolate",), host=True,
                 doc=mutgap_svc_placement_reserved_index.__doc__)
    s.bring_up()
    t = RTask(s, pages=32)
    x0 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    x1 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = _placement_list(t, {x0: (1, 1)})
    t.blit(b_copy_tex_buf(x1, b, 0, 16, 64, (4, 4, 1)), resources=[(x1, 0), (b, 0)], label="make X1")
    t.dump(pl + 16 * x0, 32, "X0's and X1's entries after X1")
    t.blit(b_copy_tex_buf(x0, b, 0x40, 16, 64, (4, 4, 1)), resources=[(x0, 0), (b, 0)],
           label="make X0 at its forced index")
    t.dump(pl + 16 * x0, 32, "X0's and X1's entries after X0")
    t.delete()
    s.events("end")
    return s


@scenario(setups=("isolate",), host=True)
def mutgap_svc_placement_unmapped_then_written():
    """UnmapMemory (34) of only the placement list's page, after an exec made
    a texture and wrote its entry. The service unmaps the placement list,
    which releases its mapping and forgets its address; the next exec maps it
    again, and a second texture's entry is written into the guest's page.
    Isolate only (argument buffers); host-keyed (Metal's indices)."""
    s = Scenario("mutgap_svc_placement_unmapped_then_written", setups=("isolate",), host=True,
                 doc=mutgap_svc_placement_unmapped_then_written.__doc__)
    s.bring_up()
    t = RTask(s, pages=32)
    x1 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    x2 = t.add(abi.T_TEXTURE, texture_rec(PAGE, t.area(PAGE) // PAGE, [simple_dim(4, 4)], desc1(usage=0x13)))
    b, b_va = t.shared(PAGE)
    t.define()
    pl = _placement_list(t, {})
    t.blit(b_copy_tex_buf(x1, b, 0, 16, 64, (4, 4, 1)), resources=[(x1, 0), (b, 0)], label="make X1")
    t.unmap(pl, PAGE, label="34 of the placement list's page")
    t.blit(b_copy_tex_buf(x2, b, 0x40, 16, 64, (4, 4, 1)), resources=[(x2, 0), (b, 0)], label="make X2")
    t.dump(pl + 16 * x1, 32, "X1's and X2's entries")
    t.delete()
    s.events("end")
    return s
