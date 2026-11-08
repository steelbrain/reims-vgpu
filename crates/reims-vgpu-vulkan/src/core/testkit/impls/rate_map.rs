//! Fake: the `platform/rate_map.rs` traits over the fake models (architecture 13.2).
//!
//! A rate-map descriptor holds its value's fields and the layers set on it (child objects, by
//! index); a rate map holds what it was made from. The ordered calls (`setLayer:atIndex:`, the
//! map's copies, mappings and reset) are logged and scripted; the getters read the models.

use crate::core::testkit::script::Policy;
use std::sync::Arc;

use super::super::calllog::Arg;
use super::super::guest_ram::RawRegion;
use super::super::objects::{Fk, LayerModel, Model, ResourceCommon, World, kind};
use super::super::platform::FakePlatform;
use super::fatal;
use crate::core::host::HostPtr;
use crate::core::platform::Platform;
use crate::core::platform::rate_map::*;
use crate::core::platform::types::{MtlCoordinate2D, MtlSize, SizeAndAlign};
use crate::core::platform::values::RateMapDesc;
use crate::core::raise::{OnFault, RaiseResult};

/// A rasterization rate map's model.
#[derive(Debug)]
pub struct RateMapModel {
    /// Common resource state.
    pub common: ResourceCommon,
    /// `layerCount`: its descriptor's layers when made.
    pub layer_count: u64,
    /// Its descriptor's screen size, which `physicalSizeForLayer:` answers for every layer.
    pub screen_size: MtlSize,
    /// `mutability`.
    pub mutability: u64,
    /// `parameterBufferSizeAndAlign`.
    pub parameter_buffer: SizeAndAlign,
}

impl Default for RateMapModel {
    /// No layers, immutable, and a parameter buffer of 0x100 bytes aligned to 0x10.
    fn default() -> RateMapModel {
        RateMapModel {
            common: ResourceCommon::default(),
            layer_count: 0,
            screen_size: MtlSize::default(),
            mutability: 0,
            parameter_buffer: SizeAndAlign {
                size: 0x100,
                align: 0x10,
            },
        }
    }
}

impl RateMap<FakePlatform> for Fk<kind::RateMap> {
    fn copy_parameter_data(
        &self,
        buffer: &<FakePlatform as Platform>::Buffer,
        offset: u64,
    ) -> RaiseResult<()> {
        self.act(
            Policy::Caught,
            "copyParameterDataToBuffer:offset:",
            vec![Arg::obj(buffer), Arg::U64(offset)],
        )
    }

    fn layer_count(&self) -> u64 {
        self.with_model(|m: &RateMapModel| m.layer_count)
    }

    fn physical_to_screen(
        &self,
        physical_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        self.thr(
            Policy::Caught,
            "mapPhysicalToScreenCoordinates:forLayer:",
            vec![Arg::val(&physical_coordinates), Arg::U64(layer_index)],
            || physical_coordinates,
        )
    }

    fn screen_to_physical(
        &self,
        screen_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D> {
        self.thr(
            Policy::Caught,
            "mapScreenToPhysicalCoordinates:forLayer:",
            vec![Arg::val(&screen_coordinates), Arg::U64(layer_index)],
            || screen_coordinates,
        )
    }

    fn mutability(&self) -> u64 {
        self.with_model(|m: &RateMapModel| m.mutability)
    }

    fn parameter_buffer_size_and_align(&self) -> SizeAndAlign {
        self.with_model(|m: &RateMapModel| m.parameter_buffer)
    }

    fn physical_granularity(&self) -> MtlSize {
        MtlSize::new(1, 1, 1)
    }

    fn physical_size(&self, layer_index: u64) -> RaiseResult<MtlSize> {
        self.thr(
            Policy::Caught,
            "physicalSizeForLayer:",
            vec![Arg::U64(layer_index)],
            || self.with_model(|m: &RateMapModel| m.screen_size),
        )
    }

    fn reset(&self, descriptor: &<FakePlatform as Platform>::RateMapDescriptor) {
        fatal(self.act(
            Policy::Fatal,
            "resetUsingDescriptor:",
            vec![Arg::obj(descriptor)],
        ));
    }
}

impl Fk<kind::RateMap> {
    /// Makes `layerCount` answer `count`.
    pub fn set_layer_count(&self, count: u64) {
        self.with_model_mut(|m: &mut RateMapModel| m.layer_count = count);
    }

    /// Makes `mutability` answer `mutability`.
    pub fn set_mutability(&self, mutability: u64) {
        self.with_model_mut(|m: &mut RateMapModel| m.mutability = mutability);
    }

    /// Makes `parameterBufferSizeAndAlign` answer `size_and_align`.
    pub fn set_parameter_buffer_size_and_align(&self, size_and_align: SizeAndAlign) {
        self.with_model_mut(|m: &mut RateMapModel| m.parameter_buffer = size_and_align);
    }
}

/// A rasterization rate map descriptor's model; its layers are child objects.
#[derive(Debug, Default)]
pub struct RateMapDescriptorModel {
    /// The value it was made from.
    pub desc: RateMapDesc,
    /// What `layerCount` answers in place of the layers set ([`Fk::set_layer_count`]).
    pub layer_count: Option<u64>,
}

impl RateMapDescriptor<FakePlatform> for Fk<kind::RateMapDescriptor> {
    fn set_layer(
        &self,
        on: OnFault,
        layer: &<FakePlatform as Platform>::RateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()> {
        self.act(
            Policy::On(on),
            "setLayer:atIndex:",
            vec![Arg::obj(layer), Arg::U64(layer_index)],
        )?;
        self.set_child("layer", layer_index, layer.erase());
        Ok(())
    }

    /// The count [`set_layer_count`](Fk::set_layer_count) set, else one past the highest layer
    /// set.
    fn layer_count(&self) -> u64 {
        self.with_model(|m: &RateMapDescriptorModel| m.layer_count)
            .unwrap_or_else(|| self.layers_set())
    }

    /// The layer set at `layer_index`, nil when none is.
    fn layer(
        &self,
        _on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<<FakePlatform as Platform>::RateMapLayerDescriptor>> {
        Ok(self
            .get_child("layer", layer_index)
            .and_then(|h| h.cast::<kind::RateMapLayerDescriptor>()))
    }
}

impl Fk<kind::RateMapDescriptor> {
    /// Sets `desc`.
    pub(crate) fn apply(&self, desc: &RateMapDesc) {
        self.with_model_mut(|m: &mut RateMapDescriptorModel| m.desc = *desc);
    }

    /// The value the descriptor was made from.
    pub fn desc(&self) -> RateMapDesc {
        self.with_model(|m: &RateMapDescriptorModel| m.desc)
    }

    /// One past the highest layer set.
    pub(crate) fn layers_set(&self) -> u64 {
        self.state()
            .children
            .iter()
            .filter(|(k, _, _)| *k == "layer")
            .map(|(_, i, _)| (*i).wrapping_add(1))
            .max()
            .unwrap_or(0)
    }

    /// Makes `layerCount` answer `count`, whatever layers are set.
    pub fn set_layer_count(&self, count: u64) {
        self.with_model_mut(|m: &mut RateMapDescriptorModel| m.layer_count = Some(count));
    }
}

impl RateMapLayerDescriptor for Fk<kind::RateMapLayerDescriptor> {
    fn horizontal_sample_storage(&self) -> HostPtr {
        self.with_model(|l: &LayerModel| l.horizontal)
    }

    fn sample_count(&self) -> MtlSize {
        self.with_model(|l: &LayerModel| l.sample_count)
    }

    fn vertical_sample_storage(&self) -> HostPtr {
        self.with_model(|l: &LayerModel| l.vertical)
    }
}

impl Fk<kind::RateMapLayerDescriptor> {
    /// Makes `sampleCount` answer `count`; the storage stays.
    pub fn set_sample_count(&self, count: MtlSize) {
        self.with_model_mut(|l: &mut LayerModel| l.sample_count = count);
    }

    /// Makes `horizontalSampleStorage` answer `storage`.
    pub fn set_horizontal_sample_storage(&self, storage: HostPtr) {
        self.with_model_mut(|l: &mut LayerModel| l.horizontal = storage);
    }

    /// Makes `verticalSampleStorage` answer `storage`.
    pub fn set_vertical_sample_storage(&self, storage: HostPtr) {
        self.with_model_mut(|l: &mut LayerModel| l.vertical = storage);
    }
}

/// `[[MTLRasterizationRateLayerDescriptor alloc] initWithSampleCount:]`'s model: zeroed float
/// storage, `sample_count.width` horizontal and `sample_count.height` vertical samples. Nil when
/// either count is 0 (measured on macOS 26.7: `{0,0,0}` is nil) or when the storage cannot be
/// allocated (`{~0,~0,0}` is nil; `{100000,100000,0}` is an object).
pub fn new_layer(
    w: &Arc<World>,
    sample_count: MtlSize,
) -> Option<Fk<kind::RateMapLayerDescriptor>> {
    if sample_count.width == 0 || sample_count.height == 0 {
        return None;
    }
    let floats = |n: u64| RawRegion::new(n.checked_mul(4)?);
    let horizontal = floats(sample_count.width)?;
    let vertical = floats(sample_count.height)?;
    Some(w.new_obj(Model::Layer(LayerModel {
        sample_count,
        horizontal: horizontal.addr(),
        vertical: vertical.addr(),
        storage: vec![horizontal, vertical],
    })))
}
