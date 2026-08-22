//! Guest function objects: loading the MTLB container, and carving the
//! wrapped AIR out of it.
//!
//! Each function descriptor names an MTLB container in guest memory;
//! metal2vulkan consumes the LLVM BitcodeWrapper (`0x0b17c0de`) record inside.
//! [`load_mtlb`](crate::runtime::mtlb::load_mtlb) does the first half
//! (object list → container bytes) and
//! [`extract_air`](crate::runtime::mtlb::extract_air) the second. Port of archive `reims-vgpu-backend-vulkan`
//! `mtlb.rs` (structural carve only — no guest scan).
//!
//! # One loader, two rails
//!
//! The compute rail and the draw rail both need a function's container, and
//! until this module took the loader they each had their own copy of it — the
//! same six steps in the same order, down to a verbatim-shared comment about the
//! guest's `blob_size` being authoritative. They had drifted in the way this
//! project's twin functions always drift: the compute copy named all six of its
//! failures in the fail log and the draw copy returned a bare `None` from every
//! one of them, so a draw that lost its shader said only `MissingMtlb` and never
//! which of six things went wrong. That is why the loader takes an
//! [`AirLoadRail`](crate::runtime::mtlb::AirLoadRail) rather than living in
//! either caller.
//!
//! # What the draw rail's new lines cost, measured
//!
//! Giving the per-frame rail six fail lines it never had is a volume question,
//! so it was measured rather than argued: a driven x86/Vulkan boot (Safari drag,
//! 2 685 posted events, ~35 Hz median present) ran **177 746 draws**, hence
//! ~355 000 calls to this loader — two per draw, vertex and fragment — and
//! emitted **zero** `draw_load_mtlb` lines.
//!
//! That zero is a healthy one rather than an unarmed detector, and the thing
//! that says so is independent of this module: the coarse `MissingMtlb`-class
//! declines its callers raise were *also* zero on that boot, and they predate
//! the emission. The loader was not failing quietly before; it was not failing.
//! So a `draw_load_mtlb` line is a real event, and the flood this could have
//! been does not exist on a healthy guest.

use crate::model::LoadedFunction;
use crate::runtime::decode::resource::{decode_function_descriptor, ObjectKind};
use crate::runtime::draw::host_alloc_len;
use crate::runtime::host::{HostMemory, HostOps};
use crate::runtime::Device;
use crate::runtime::{gva_mem, objects};
use std::sync::Arc;

/// LLVM BitcodeWrapperHeader magic `0x0b17c0de` LE.
///
/// Public because [`extract_air`] is, and everything that function does happens
/// *after* it finds this: a caller that wants to reach the wrapper-header
/// arithmetic must be able to name the magic rather than write the four bytes
/// out a second time.
pub const AIR_WRAP_MAGIC: [u8; 4] = [0xde, 0xc0, 0x17, 0x0b];
const WRAPPER_HEADER_LEN: usize = 0x14;

/// A structural refusal while locating the LLVM BitcodeWrapper inside an MTLB.
pub use reims_vgpu_core::MtlbDecline;

/// Which rail asked for a function's MTLB container.
///
/// The only thing that differs between the two is the event name the failure
/// lines carry. The reason slugs are deliberately bare — `ladder_slug!("", …)`
/// and friends — because the event name already says which load it was, which
/// is the convention [`crate::observe::ladder_slug`] documents; so the event
/// name is the one thing that has to travel in from the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AirLoadRail {
    /// Compute dispatch, its sessions, and compute ICB bodies.
    Compute,
    /// Render draws and render ICB bodies.
    Draw,
}

impl AirLoadRail {
    /// The fail-log event name for this rail's load failures.
    fn event(self) -> &'static str {
        match self {
            Self::Compute => "compute_load_mtlb",
            Self::Draw => "draw_load_mtlb",
        }
    }
}

/// Resolve a function object's immutable MTLB container.
///
/// The first resolution reads guest memory and retains the payload under the
/// function reference lifetime. Later callers receive the same bytes until an
/// explicit function delete or task teardown ends that lifetime.
///
/// `None` means the caller gets no shader; every reason for it but one is
/// written to the fail log under [`AirLoadRail::event`], because the callers all
/// collapse this into one coarse `MissingMtlb`-class decline and the reason is
/// the only thing that says which of six steps refused.
///
/// The exception is `func_ref == 0`, which is "no function bound" — a legitimate
/// state (a pipeline with no fragment stage, say) that `AGENTS.md` names as a
/// thing not to log. It stays silent.
pub fn load_mtlb<M: HostMemory + HostOps>(
    state: &Device,
    host: &M,
    task_id: u32,
    func_ref: u32,
    rail: AirLoadRail,
) -> Option<Arc<[u8]>> {
    if func_ref == 0 {
        return None;
    }
    if let Some(function) = state
        .task_objects
        .functions
        .get(task_id, reims_vgpu_protocol::SerializerRef::new(func_ref))
    {
        crate::runtime::drain::note_store_route("function_state_hit");
        return Some(Arc::clone(&function.mtlb));
    }
    crate::runtime::drain::note_store_route("function_state_miss");
    let report = crate::observe::RungReport::new(rail.event(), "func_ref");
    let miss = |reason: &str, detail: String| -> Option<Arc<[u8]>> {
        report.reason(task_id, func_ref, reason, &detail);
        None
    };
    let (_entry, desc) = match objects::resolve_descriptor(
        state,
        host,
        task_id,
        func_ref,
        &[ObjectKind::Function],
    ) {
        Ok(found) => found,
        Err(rung) => {
            report.rung(task_id, func_ref, rung);
            return None;
        }
    };
    let Ok(f) = decode_function_descriptor(&desc) else {
        return miss(
            crate::observe::ladder_slug!("", desc_decode),
            format!("desc_len={}", desc.len()),
        );
    };
    if f.blob_gva == 0 || f.blob_size < 4 {
        return miss(
            "bad_blob",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    // Guest blob_size is authoritative — no product 1 MiB MTLB ceiling.
    let Some(len) = host_alloc_len(f.blob_size as u64) else {
        return miss(
            "host_len",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    };
    let mut mtlb = vec![0u8; len];
    // Device page_shift (x86=12, arm64=14); the unshifted helper defaults to
    // arm and fails every load on the other geometry.
    if gva_mem::read_task_gva_by_id(
        host,
        &state.tasks,
        task_id,
        f.blob_gva,
        &mut mtlb,
        state.page_shift,
    )
    .is_err()
    {
        return miss(
            "gva_read",
            format!("blob_gva={:#x} blob_size={}", f.blob_gva, f.blob_size),
        );
    }
    let mtlb: Arc<[u8]> = mtlb.into();
    let function = Arc::new(LoadedFunction {
        mtlb: Arc::clone(&mtlb),
    });
    let retained = state.task_objects.functions.register(
        task_id,
        reims_vgpu_protocol::SerializerRef::new(func_ref),
        function,
    );
    Some(Arc::clone(&retained.mtlb))
}

/// Extract the wrapped-AIR blob from an MTLB container or bare wrapper.
pub fn extract_air(data: &[u8]) -> Result<&[u8], MtlbDecline> {
    let start = find_wrap_magic(data, 0).ok_or(MtlbDecline::WrappedAirMissing {
        data_len: data.len(),
    })?;
    blob_at(data, start)
}

fn find_wrap_magic(data: &[u8], from: usize) -> Option<usize> {
    if data.len() < WRAPPER_HEADER_LEN {
        return None;
    }
    (from..=data.len() - AIR_WRAP_MAGIC.len()).find(|&i| data[i..i + 4] == AIR_WRAP_MAGIC)
}

fn blob_at(data: &[u8], off: usize) -> Result<&[u8], MtlbDecline> {
    let header_end = off.saturating_add(WRAPPER_HEADER_LEN);
    if header_end > data.len() {
        return Err(MtlbDecline::WrapperHeaderTruncated {
            offset: off,
            data_len: data.len(),
        });
    }
    let bc_off = u32::from_le_bytes(data[off + 8..off + 12].try_into().unwrap());
    let bc_size = u32::from_le_bytes(data[off + 12..off + 16].try_into().unwrap());
    let blob_len = u64::from(bc_off) + u64::from(bc_size);
    let blob_end = usize::try_from(blob_len)
        .ok()
        .and_then(|len| off.checked_add(len));
    // Guest/header sizes are authoritative — no product MiB ceiling. Only require
    // the declared blob fits inside the MTLB buffer we already loaded.
    if blob_len < WRAPPER_HEADER_LEN as u64 || blob_end.is_none_or(|end| end > data.len()) {
        return Err(MtlbDecline::BlobOutOfBounds {
            offset: off,
            blob_len,
            data_len: data.len(),
        });
    }
    Ok(&data[off..blob_end.expect("bounds checked above")])
}

#[cfg(test)]
mod tests {
    use crate::runtime::decode::resource::OBJECT_TYPE_FUNCTION;

    use super::*;
    use crate::model::{DeviceId, PAGE_SHIFT_ARM64E};
    use crate::runtime::host::FakeHost;
    use reims_vgpu_core::endian::{st32, st64};
    use reims_vgpu_paging::geometry::{DIRECTORY_DEPTH, DIRECTORY_ROOT_PFN};

    /// Task 1 with a one-entry object list whose ref 1 holds `object_type`, and
    /// a descriptor blob of `desc` at GVA 0x40.
    fn task_with_one_object(host: &mut FakeHost, state: &mut Device, object_type: u8, desc: &[u8]) {
        let dir_gpa = 2u64 << PAGE_SHIFT_ARM64E;
        let root_gpa = 3u64 << PAGE_SHIFT_ARM64E;
        let data_gpa = 4u64 << PAGE_SHIFT_ARM64E;
        host.map_range(dir_gpa, 0x20, 0);
        host.map_range(root_gpa, 0x4000, 0);
        host.map_range(data_gpa, 0x200, 0);
        let mut d = [0u8; 8];
        st32(&mut d[DIRECTORY_ROOT_PFN as usize..], 3);
        st32(&mut d[DIRECTORY_DEPTH as usize..], 1);
        let _ = host.write_gpa(dir_gpa, &d);
        st32(&mut d[..4], 4);
        let _ = host.write_gpa(root_gpa, &d[..4]);

        state.define_task(1, 0x1000, 2);
        assert!(state.set_object_list(1, 0, 8));
        let mut entry = [0u8; 12];
        st32(
            &mut entry[0..],
            u32::from(object_type) | ((desc.len() as u32) << 8),
        );
        st64(&mut entry[4..12], 0x40);
        let _ = host.write_gpa(data_gpa + 12, &entry);
        let _ = host.write_gpa(data_gpa + 0x40, desc);
    }

    /// Both rails name the same refusal, each under its own event.
    ///
    /// The draw rail is the half this pins: it used to be a separate copy of
    /// this loader that returned a bare `None` from all six of its failure
    /// points, so a draw whose shader would not load reported `MissingMtlb` and
    /// nothing about which step refused. The compute half is here because the
    /// two events must stay distinguishable — one shared loader emitting one
    /// event name would make the fail log unable to say which rail lost work.
    #[test]
    fn both_rails_name_the_rung_that_refused_under_their_own_event() {
        let mut host = FakeHost::new();
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        // IOSurface wire tag 11 is not the function tag this loader accepts.
        task_with_one_object(&mut host, &mut state, 11, &[0u8; 0x20]);

        let cap = crate::observe::FailCapture::start();
        assert!(load_mtlb(&state, &host, 1, 1, AirLoadRail::Draw).is_none());
        let line = cap.one("draw_load_mtlb");
        assert!(
            line.contains(&format!(
                "reason={}",
                crate::observe::ladder_slug!("", wrong_type)
            )) && line.contains("ot=iosurface_texture"),
            "the draw rail must name the rung and the tag it found: {line}"
        );
        drop(cap);

        let cap = crate::observe::FailCapture::start();
        assert!(load_mtlb(&state, &host, 1, 1, AirLoadRail::Compute).is_none());
        assert!(
            cap.one("compute_load_mtlb")
                .contains("ot=iosurface_texture"),
            "the compute rail keeps its own event name"
        );
    }

    /// `func_ref == 0` is the one refusal that stays silent, because it is not
    /// one: a pipeline with no fragment stage binds no fragment function.
    #[test]
    fn an_unbound_function_ref_says_nothing() {
        let mut host = FakeHost::new();
        let mut state = Device::new(DeviceId(1), PAGE_SHIFT_ARM64E);
        task_with_one_object(&mut host, &mut state, OBJECT_TYPE_FUNCTION, &[0u8; 0x20]);

        for rail in [AirLoadRail::Draw, AirLoadRail::Compute] {
            let cap = crate::observe::FailCapture::start();
            assert!(load_mtlb(&state, &host, 1, 0, rail).is_none());
            assert!(
                cap.lines().is_empty(),
                "an unbound ref must spend no line on {rail:?}: {:?}",
                cap.lines()
            );
        }
    }

    #[test]
    fn extract_bare_wrapper() {
        // Minimal synthetic: magic + version + offset 0x14 + size 4 + cpu + 4 body bytes.
        let mut data = vec![0u8; 0x18];
        data[0..4].copy_from_slice(&AIR_WRAP_MAGIC);
        data[4..8].copy_from_slice(&0u32.to_le_bytes()); // version
        data[8..12].copy_from_slice(&0x14u32.to_le_bytes()); // BitcodeOffset
        data[12..16].copy_from_slice(&4u32.to_le_bytes()); // BitcodeSize
        data[0x14..0x18].copy_from_slice(&[1, 2, 3, 4]);
        let air = extract_air(&data).expect("air");
        assert_eq!(air.len(), 0x18);
    }

    #[test]
    fn malformed_wrappers_fire_typed_declines() {
        assert_eq!(
            extract_air(&[]).unwrap_err(),
            MtlbDecline::WrappedAirMissing { data_len: 0 }
        );
        assert_eq!(
            blob_at(&[0; 8], 0).unwrap_err(),
            MtlbDecline::WrapperHeaderTruncated {
                offset: 0,
                data_len: 8
            }
        );

        let mut data = vec![0u8; WRAPPER_HEADER_LEN];
        data[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        data[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        let expected = MtlbDecline::BlobOutOfBounds {
            offset: 0,
            blob_len: u64::from(u32::MAX) * 2,
            data_len: WRAPPER_HEADER_LEN,
        };
        assert_eq!(blob_at(&data, 0).unwrap_err(), expected);
    }

    #[test]
    fn mtlb_declines_have_distinct_log_safe_reasons() {
        use crate::observe::Decline as _;
        let cases = [
            MtlbDecline::WrappedAirMissing { data_len: 1 },
            MtlbDecline::WrapperHeaderTruncated {
                offset: 1,
                data_len: 2,
            },
            MtlbDecline::BlobOutOfBounds {
                offset: 1,
                blob_len: 2,
                data_len: 3,
            },
        ];
        let mut slugs = std::collections::HashSet::new();
        for decline in cases {
            assert!(slugs.insert(decline.slug()));
            for (_, value) in decline.fields() {
                assert!(!value.contains(char::is_whitespace));
            }
        }
    }
}
