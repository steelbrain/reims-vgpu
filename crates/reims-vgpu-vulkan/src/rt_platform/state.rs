//! The rt platform's sampler states ([`RtSampler`]) and depth-stencil states: rt's, made of
//! core's values ([`SamplerDesc`], [`DepthStencilDesc`]), each raw value as rt's type of it in the
//! value's order. A value rt's type has no value of is Metal's refusal, where Metal's validation
//! stops the creator: nil, counted (`rt-refused-metal`).
//!
//! - **A sampler state's index.** rt writes a sampler state into its sampler arrays at an index of
//!   a live sampler grant. A state the guest places (the value's `resourceIndex`, 0 for none)
//!   takes the guest's index, from the guest's reservation (`reserveResourceIndicesForResourceType:`
//!   of type 1); one it does not place takes one of the platform's own (`super::texture`'s
//!   `OwnIndices`, of the sampler class), as a texture does: the highest index no reservation holds
//!   and no state of the platform's took yet, then the one given back longest ago, a take while a
//!   command buffer whose encoders bound the index's last owner is unfinished counted for the boot
//!   diagnostics (`super::texture`'s `IndexUse`). Its `gpuResourceID` is rt's, the index and the
//!   translator's word of its descriptor, and its `resourceIndex` the index.
//!   `supportArgumentBuffers` changes nothing: every state has an index. rt refuses a cubic filter
//!   (`rt-refused-gap`), and a state past the device's samplers (`rt-refused-limit`): nil, counted.
//! - **A depth-stencil state** is rt's, which holds no Vulkan object: a draw under it sets its
//!   tests as dynamic state. Its `gpuResourceID` is 0 and its `resourceIndex` raises, as the
//!   host's stubs answer below Apple10 (measured on the M3 Max).

use std::sync::Arc;

use metal2vulkan_runtime as rt;

use crate::core::platform::Handle;
use crate::core::platform::resource::{DepthStencilState, SamplerState};
use crate::core::platform::values::{DepthStencilDesc, SamplerDesc, StencilDesc};
use crate::core::raise::{Raise, RaiseResult};

use super::RtObj;
use super::device::RtDevice;
use super::texture::{IndexUse, refused_metal};
use super::unbuilt::{RtRefusal, refused_as, rt_refused_create};

/// `newSamplerStateWithDescriptor:`.
const SAMPLER: &str = "newSamplerStateWithDescriptor:";

/// `newDepthStencilStateWithDescriptor:`.
const DEPTH_STENCIL: &str = "newDepthStencilStateWithDescriptor:";

/// `id<MTLSamplerState>` on metal2vulkan-runtime.
pub struct RtSampler {
    /// rt's state.
    rt: rt::SamplerState,
    /// The device that made it, whose own indices it may hold.
    device: RtObj<RtDevice>,
    /// Its index where it is one of the platform's own, which its drop gives back.
    own: Option<u32>,
    /// What reads it through that index (`super::texture::IndexUse`), which goes back with it;
    /// `None` where it has none of the platform's own.
    read: Option<Arc<IndexUse>>,
}

impl Drop for RtSampler {
    fn drop(&mut self) {
        if let Some(index) = self.own {
            self.device
                .sampler_indices()
                .give_back(index, self.read.take());
        }
    }
}

impl RtSampler {
    /// rt's state.
    pub fn rt(&self) -> &rt::SamplerState {
        &self.rt
    }

    /// What reads it through its index, where that is one of the platform's own
    /// (`super::texture::IndexUse`), which a command buffer whose work reads it so notes.
    pub(super) fn index_use(&self) -> Option<&Arc<IndexUse>> {
        self.read.as_ref()
    }
}

/// The raw value `raw` as rt's `T`, or the error naming it as `what`.
pub(super) fn value<T: TryFrom<u32>>(what: &str, raw: u64) -> Result<T, String> {
    let named = || format!("{what} {raw}");
    let raw = u32::try_from(raw).map_err(|_| named())?;
    T::try_from(raw).map_err(|_| named())
}

/// rt's sampler descriptor of `d`, each value set in the value's order and Metal's default for
/// each unset. `supportArgumentBuffers` and the resource index are not the descriptor's.
///
/// Err: the first value rt's descriptor has no value of, named.
pub fn rt_sampler_descriptor(d: &SamplerDesc) -> Result<rt::SamplerDescriptor, String> {
    let mut s = rt::SamplerDescriptor::new();
    if let Some(v) = d.min_filter {
        s.min_filter = value("min filter", v)?;
    }
    if let Some(v) = d.mag_filter {
        s.mag_filter = value("mag filter", v)?;
    }
    if let Some(v) = d.lod_average {
        s.lod_average = v;
    }
    if let Some(v) = d.compare_function {
        s.compare_function = value("compare function", v)?;
    }
    if let Some(v) = d.s_address_mode {
        s.s_address_mode = value("s address mode", v)?;
    }
    if let Some(v) = d.t_address_mode {
        s.t_address_mode = value("t address mode", v)?;
    }
    if let Some(v) = d.r_address_mode {
        s.r_address_mode = value("r address mode", v)?;
    }
    if let Some(v) = d.mip_filter {
        s.mip_filter = value("mip filter", v)?;
    }
    if let Some(v) = d.max_anisotropy {
        s.max_anisotropy = u32::try_from(v).map_err(|_| format!("max anisotropy {v}"))?;
    }
    if let Some(v) = d.normalized_coordinates {
        s.normalized_coordinates = v;
    }
    if let Some(v) = d.reduction_mode {
        s.reduction_mode = value("reduction mode", v)?;
    }
    if let Some(v) = d.lod_min_clamp {
        s.lod_min_clamp = v;
    }
    if let Some(v) = d.lod_max_clamp {
        s.lod_max_clamp = v;
    }
    if let Some(v) = d.lod_bias {
        s.lod_bias = v;
    }
    if let Some(v) = d.border_color {
        s.border_color = value("border colour", v)?;
    }
    Ok(s)
}

/// rt's stencil descriptor of a face's `s`, every value set in field order.
fn rt_stencil(s: &StencilDesc) -> Result<rt::StencilDescriptor, String> {
    let mut out = rt::StencilDescriptor::default();
    out.stencil_compare_function = value("stencil compare function", s.stencil_compare_function)?;
    out.stencil_failure_operation =
        value("stencil failure operation", s.stencil_failure_operation)?;
    out.depth_failure_operation = value("depth failure operation", s.depth_failure_operation)?;
    out.depth_stencil_pass_operation = value(
        "depth-stencil pass operation",
        s.depth_stencil_pass_operation,
    )?;
    out.read_mask = s.read_mask;
    out.write_mask = s.write_mask;
    Ok(out)
}

/// rt's depth-stencil descriptor of `d`, each value set in the value's order and Metal's default
/// for each unset, a nil face's among them.
///
/// Err: the first value rt's descriptor has no value of, named.
pub fn rt_depth_stencil_descriptor(
    d: &DepthStencilDesc,
) -> Result<rt::DepthStencilDescriptor, String> {
    let mut out = rt::DepthStencilDescriptor::default();
    if let Some(v) = d.depth_compare_function {
        out.depth_compare_function = value("depth compare function", v)?;
    }
    if let Some(v) = d.depth_write_enabled {
        out.depth_write_enabled = v;
    }
    if let Some(front) = &d.front {
        out.front_face_stencil = rt_stencil(front)?;
    }
    if let Some(back) = &d.back {
        out.back_face_stencil = rt_stencil(back)?;
    }
    Ok(out)
}

/// `newSamplerStateWithDescriptor:` on `device` (the module's docs): nil, counted, where a value
/// is one rt has none of, no index can be had, or rt refuses the state.
pub fn new_sampler_state(device: &RtObj<RtDevice>, desc: &SamplerDesc) -> Option<RtObj<RtSampler>> {
    let descriptor = match rt_sampler_descriptor(desc) {
        Ok(descriptor) => descriptor,
        Err(why) => return refused_metal(SAMPLER, &why),
    };
    let (index, own) = match desc.resource_index.unwrap_or(0) {
        0 => match device.sampler_indices().take(device.rt(), SAMPLER) {
            Ok(Some(index)) => (index, Some(index)),
            Ok(None) => {
                crate::core::platform::device::note_relayed_gap();
                let why = "no sampler index of the platform's own is left";
                refused_as(SAMPLER, RtRefusal::Limit, &why);
                return None;
            }
            Err(error) => {
                rt_refused_create(SAMPLER, &error);
                return None;
            }
        },
        placed => match u32::try_from(placed) {
            Ok(index) => (index, None),
            Err(_) => return refused_metal(SAMPLER, &format!("resource index {placed}")),
        },
    };
    match device.rt().new_sampler_state(&descriptor, index) {
        Ok(made) => {
            super::diag::sampler(&descriptor);
            Some(RtObj::new(RtSampler {
                rt: made,
                device: device.clone_ref(),
                own,
                read: own.map(|_| IndexUse::new(super::diag::OwnMade::sampler(SAMPLER))),
            }))
        }
        Err(error) => {
            if let Some(index) = own {
                device.sampler_indices().give_back(index, None);
            }
            rt_refused_create(SAMPLER, &error);
            None
        }
    }
}

/// `newDepthStencilStateWithDescriptor:` on `device` (the module's docs): nil, counted, where a
/// value is one rt has none of.
pub fn new_depth_stencil_state(
    device: &RtObj<RtDevice>,
    desc: &DepthStencilDesc,
) -> Option<RtObj<rt::DepthStencilState>> {
    match rt_depth_stencil_descriptor(desc) {
        Ok(descriptor) => Some(RtObj::new(device.rt().new_depth_stencil_state(&descriptor))),
        Err(why) => refused_metal(DEPTH_STENCIL, &why),
    }
}

/// rt's ID and its index (the module's docs).
impl SamplerState for RtObj<RtSampler> {
    fn gpu_resource_id(&self) -> u64 {
        crate::recorder::plain_resource_id(self, || self.rt.gpu_resource_id().to_bits())
    }

    fn resource_index(&self) -> u64 {
        u64::from(self.rt.gpu_resource_id().index)
    }
}

/// The host's stubs' answers (the module's docs).
impl DepthStencilState for RtObj<rt::DepthStencilState> {
    fn gpu_resource_id(&self) -> RaiseResult<u64> {
        crate::recorder::resource_id(self, || Ok(0))
    }

    fn resource_index(&self) -> RaiseResult<u64> {
        Err(Raise::platform("resourceIndex"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::GpuDevice as _;
    use crate::rt_platform::{RtPlatform, RtRefusal as Kind};

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// Every value of a sampler descriptor is rt's type of it; an unset one is Metal's default;
    /// a value rt has none of is named.
    #[test]
    fn sampler_values_are_rt_s() {
        let unset = rt_sampler_descriptor(&SamplerDesc::default());
        assert_eq!(unset, Ok(rt::SamplerDescriptor::new()));
        let d = SamplerDesc {
            min_filter: Some(1),
            mag_filter: Some(1),
            mip_filter: Some(2),
            s_address_mode: Some(2),
            t_address_mode: Some(3),
            r_address_mode: Some(4),
            border_color: Some(2),
            compare_function: Some(3),
            max_anisotropy: Some(8),
            lod_min_clamp: Some(1.0),
            lod_max_clamp: Some(4.0),
            lod_bias: Some(-0.5),
            ..SamplerDesc::default()
        };
        let s = rt_sampler_descriptor(&d).expect("rt's values");
        assert_eq!(s.min_filter, rt::SamplerMinMagFilter::Linear);
        assert_eq!(s.mip_filter, rt::SamplerMipFilter::Linear);
        assert_eq!(s.t_address_mode.as_raw(), 3);
        assert_eq!(s.border_color, rt::SamplerBorderColor::OpaqueWhite);
        assert_eq!(s.compare_function.as_raw(), 3);
        assert_eq!(
            (s.max_anisotropy, s.lod_max_clamp, s.lod_bias),
            (8, 4.0, -0.5)
        );
        let bad = SamplerDesc {
            mip_filter: Some(7),
            ..SamplerDesc::default()
        };
        assert_eq!(rt_sampler_descriptor(&bad), Err("mip filter 7".to_owned()));
        let wide = SamplerDesc {
            border_color: Some(1 << 32),
            ..SamplerDesc::default()
        };
        assert_eq!(
            rt_sampler_descriptor(&wide),
            Err(format!("border colour {}", 1u64 << 32))
        );
    }

    /// Every value of a depth-stencil descriptor is rt's type of it, a nil face Metal's default.
    #[test]
    fn depth_stencil_values_are_rt_s() {
        let unset = rt_depth_stencil_descriptor(&DepthStencilDesc::default());
        assert_eq!(unset, Ok(rt::DepthStencilDescriptor::default()));
        let face = StencilDesc {
            stencil_compare_function: 2,
            stencil_failure_operation: 1,
            depth_failure_operation: 3,
            depth_stencil_pass_operation: 7,
            read_mask: 0xf0,
            write_mask: 0x0f,
        };
        let d = DepthStencilDesc {
            depth_compare_function: Some(1),
            depth_write_enabled: Some(true),
            front: None,
            back: Some(face),
        };
        let out = rt_depth_stencil_descriptor(&d).expect("rt's values");
        assert_eq!(out.depth_compare_function, rt::CompareFunction::Less);
        assert!(out.depth_write_enabled);
        assert_eq!(out.front_face_stencil, rt::StencilDescriptor::default());
        let back = out.back_face_stencil;
        assert_eq!(back.stencil_compare_function, rt::CompareFunction::Equal);
        assert_eq!(
            back.depth_stencil_pass_operation,
            rt::StencilOperation::DecrementWrap
        );
        assert_eq!((back.read_mask, back.write_mask), (0xf0, 0x0f));
        let bad = DepthStencilDesc {
            front: Some(StencilDesc {
                depth_failure_operation: 8,
                ..face
            }),
            ..d
        };
        let named = rt_depth_stencil_descriptor(&bad);
        assert_eq!(named, Err("depth failure operation 8".to_owned()));
    }

    /// Sampler and depth-stencil states on the GPU the test holds: one the guest does not place
    /// takes the platform's highest sampler index, the next the one below; one the guest places
    /// at a reserved index takes it, and at an index not reserved is rt's refusal, nil, counted;
    /// a value rt has none of and a cubic filter nil, counted. A depth-stencil state's ID is 0
    /// and its index raises.
    #[test]
    fn states_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let sampler = |desc: &SamplerDesc| device.new_sampler_state(desc).expect("never a raise");
        let last = rt::IndexClass::Sampler.range().end - 1;

        let first = sampler(&SamplerDesc::default()).expect("a sampler state");
        assert_eq!(first.resource_index(), u64::from(last));
        assert_eq!(first.gpu_resource_id() & 0xffff_ffff, u64::from(last));
        let second = sampler(&SamplerDesc::default()).expect("a sampler state");
        assert_eq!(second.resource_index(), u64::from(last - 1));

        assert_eq!(device.reserve_resource_indices(1, &[5]), Ok(()));
        let placed = SamplerDesc {
            resource_index: Some(5),
            min_filter: Some(1),
            ..SamplerDesc::default()
        };
        let placed = sampler(&placed).expect("a placed sampler state");
        assert_eq!(placed.resource_index(), 5);
        assert_eq!(placed.gpu_resource_id() & 0xffff_ffff, 5);
        assert_ne!(
            placed.gpu_resource_id() >> 32,
            first.gpu_resource_id() >> 32
        );

        let metal = Kind::Refused.code();
        let before = counted(metal);
        let unplaced = SamplerDesc {
            resource_index: Some(9),
            ..SamplerDesc::default()
        };
        assert!(sampler(&unplaced).is_none());
        let unknown = SamplerDesc {
            s_address_mode: Some(99),
            ..SamplerDesc::default()
        };
        assert!(sampler(&unknown).is_none());
        assert_eq!(counted(metal), before + 2);
        let gap = Kind::Gap.code();
        let before = counted(gap);
        let cubic = SamplerDesc {
            min_filter: Some(2),
            ..SamplerDesc::default()
        };
        assert!(sampler(&cubic).is_none());
        assert_eq!(counted(gap), before + 1);

        let depth = device.new_depth_stencil_state(&DepthStencilDesc::default());
        let depth = depth
            .expect("never a raise")
            .expect("a depth-stencil state");
        assert_eq!(depth.gpu_resource_id(), Ok(0));
        assert!(depth.resource_index().is_err());
        let before = counted(metal);
        let unknown = DepthStencilDesc {
            depth_compare_function: Some(8),
            ..DepthStencilDesc::default()
        };
        let unknown = device.new_depth_stencil_state(&unknown);
        assert!(unknown.expect("never a raise").is_none());
        assert_eq!(counted(metal), before + 1);
    }
}
