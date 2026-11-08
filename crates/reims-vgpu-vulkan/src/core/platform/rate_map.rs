//! Platform traits for rasterization rate maps: `RateMap`, `RateMapDescriptor`,
//! `RateMapLayerDescriptor`.
//!
//! Conventions (architecture 4.2): one method per selector core sends; each doc line is
//! `sel: <selector> [SPI]`. Integer widths
//! follow the runtime's encodings (D41), as `method_getTypeEncoding` reads them on the AGX
//! classes (macOS 26.7), where a declaration in the headers differs. Methods with arguments,
//! creators and actions return `RaiseResult<T>`; zero-argument property getters and descriptor
//! property setters do not (a platform dies at their send on a raise). A handle argument core can
//! pass as nil is `Option<&P::X>`; one it never passes as nil is `&P::X`.

use super::types::{MtlCoordinate2D, MtlSize, SizeAndAlign};
use super::{Handle, Platform};
use crate::core::host::HostPtr;
use crate::core::raise::{OnFault, RaiseResult};

/// `MTLRasterizationRateMap` (with `MTLRasterizationRateMapSPI`).
pub trait RateMap<P: Platform>: Handle {
    /// sel: copyParameterDataToBuffer:offset:
    /// Core asserts the buffer non-nil first.
    fn copy_parameter_data(&self, buffer: &P::Buffer, offset: u64) -> RaiseResult<()>;
    /// sel: layerCount
    fn layer_count(&self) -> u64;
    /// sel: mapPhysicalToScreenCoordinates:forLayer:
    fn physical_to_screen(
        &self,
        physical_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D>;
    /// sel: mapScreenToPhysicalCoordinates:forLayer:
    fn screen_to_physical(
        &self,
        screen_coordinates: MtlCoordinate2D,
        layer_index: u64,
    ) -> RaiseResult<MtlCoordinate2D>;
    /// sel: mutability
    fn mutability(&self) -> u64;
    /// sel: parameterBufferSizeAndAlign
    fn parameter_buffer_size_and_align(&self) -> SizeAndAlign;
    /// sel: physicalGranularity
    fn physical_granularity(&self) -> MtlSize;
    /// sel: physicalSizeForLayer:
    fn physical_size(&self, layer_index: u64) -> RaiseResult<MtlSize>;
    /// sel: resetUsingDescriptor: SPI
    /// No path reaches it under a handler: a raise inside it dies at the send.
    fn reset(&self, descriptor: &P::RateMapDescriptor);
}

/// `MTLRasterizationRateMapDescriptor`, made from a `RateMapDesc` value
/// (`Platform::rate_map_descriptor`); its layers are set on it after.
///
/// Generic over `P` (architecture 4.1 lists the bound without it): `setLayer:atIndex:` names
/// another handle.
pub trait RateMapDescriptor<P: Platform>: Handle {
    /// sel: setLayer:atIndex:
    /// Reached both under a handler and with none: `on` says which this call is.
    fn set_layer(
        &self,
        on: OnFault,
        layer: &P::RateMapLayerDescriptor,
        layer_index: u64,
    ) -> RaiseResult<()>;
    /// sel: layerCount — Metal's count of the layers set.
    fn layer_count(&self) -> u64;
    /// Layer `layer_index`: `layers`, then `objectAtIndexedSubscript:`, which may raise; `None`
    /// for a nil array or layer.
    /// Reached both under a handler and with none: `on` says which this call is.
    fn layer(
        &self,
        on: OnFault,
        layer_index: u64,
    ) -> RaiseResult<Option<P::RateMapLayerDescriptor>>;
}

/// `MTLRasterizationRateLayerDescriptor`.
pub trait RateMapLayerDescriptor: Handle {
    /// sel: horizontalSampleStorage
    /// Metal's `float` array of `sample_count().width` entries, which core fills from the guest
    /// (`cursor->readBytes`); valid while the layer lives.
    fn horizontal_sample_storage(&self) -> HostPtr;
    /// sel: sampleCount
    fn sample_count(&self) -> MtlSize;
    /// sel: verticalSampleStorage
    /// Metal's `float` array of `sample_count().height` entries, which core fills from the guest
    /// (`cursor->readBytes`); valid while the layer lives.
    fn vertical_sample_storage(&self) -> HostPtr;
}
