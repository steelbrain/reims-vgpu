//! Host GPU capability classification — the single source of truth for what
//! the bound Vulkan device can do.
//!
//! Everything here is *measured* on the device at create time and consumed
//! either by a decision or by the one-shot `vk_caps` line. There is
//! deliberately no derived taxonomy on top of it: a classification nothing
//! branches on cannot be wrong in a way anyone notices, and the one that used to
//! live here was wrong. It was a `vk_caps` field naming which handoff the
//! present path would take, and it named an arm those boots never entered —
//! because the ladder that produced the string and the branch that picks the
//! path were separate pieces of code that had never been made to agree. A
//! capability line reports what was measured; the moment it becomes a second
//! implementation of a decision, it is free to disagree with the first and
//! nothing will catch it.
//!
//! * [`crate::memory::MemoryTopology`] — `Unified` vs `Discrete` selects an
//!   allocation *preference*, never a different observable result. Live: every
//!   allocation names a [`MemoryClass`] and this module turns it into flags.
//! * [`crate::device_features`] — which optional device features are queried and
//!   enabled, in one place, so no site can ask about one it did not request.
//! * [`crate::host_pointer::HostPointerImport`] — whether guest RAM can reach the GPU
//!   as a host-pointer import over a whole RAMBlock, and at what granularity.
//!   The one guest-memory rail on every host, because it is the only primitive
//!   Linux, Windows and macOS all have — dma-buf is a Linux kernel object with
//!   no equivalent on the other two. Same rule again: the
//!   import site branches on it, and a negative rung names the check that
//!   refused rather than reading as a slow host.
//! * [`DriverQuirk`] — the only place driver identity may change behavior.
//!
//! **Vulkan 1.2 is the baseline on every supported host.** See [`crate::api_floor`]
//! for why the API version is a floor check and nothing more. Nothing enforces
//! it automatically — a 1.3-core feature struct or promoted entry point must be
//! caught in review.
//!
//! # Rules for adding a capability gate
//!
//! 1. Gate on the **capability**, never on a driver name, vendor id, an API
//!    version, or `VK_KHR_portability_subset`. If a driver quirk genuinely needs
//!    keying on the driver, add a named [`DriverQuirk`] with the observed
//!    failure in its doc comment — so the next reader knows it is a workaround,
//!    not a design.
//! 2. Put the field on [`HostGpuCaps`] only if something reads it. A capability
//!    that only reaches a log line is a fact, and belongs in the format string
//!    at the site that measured it.

use crate::host_pointer::HostPointerCaps;
#[cfg(test)]
use crate::host_pointer::HostPointerImport;
use crate::memory::{MemoryClass, MemoryProfile};
use crate::push_descriptor::PushDescriptorCaps;

use ash::vk;

/// Driver-identity workarounds. Each variant documents the concrete failure it
/// works around and how to retire it. This is the ONLY place driver identity is
/// allowed to change behavior — see the module rules.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriverQuirk {
    /// MoltenVK reported `DEVICE_LOST` when a deferred draw batch was submitted
    /// by a later non-joinable draw after the target was already marked
    /// resident-ready. Retire by reproducing the batch submit on MoltenVK with
    /// validation on and fixing the ordering; the batching itself is portable.
    pub no_deferred_draw_batching: bool,
    /// GPU-only guest-visible content rails are held back where a device
    /// recreate drops registry residents and guest pages must stay
    /// authoritative. Retire once the device-loss source is closed.
    pub guest_pages_stay_authoritative: bool,
}

impl DriverQuirk {
    /// Quirks implied by a device advertising `VK_KHR_portability_subset`
    /// (in practice: MoltenVK).
    pub fn for_portability_subset(portability_subset: bool) -> Self {
        Self {
            no_deferred_draw_batching: portability_subset,
            guest_pages_stay_authoritative: portability_subset,
        }
    }
}

/// Everything the engine is allowed to know about the bound device's
/// capabilities, classified once at device create.
#[derive(Clone, Debug)]
pub struct HostGpuCaps {
    pub memory: MemoryProfile,
    /// `maxMemoryAllocationSize` — see
    /// [`crate::memory::reported_max_allocation_size`]. Read by the one memory-type
    /// entry point, which refuses an allocation past it rather than making a
    /// call the specification declares invalid.
    pub max_allocation_size: u64,
    pub quirks: DriverQuirk,
    /// Whether guest RAM may be imported as `VkDeviceMemory` through a host
    /// pointer over a whole RAMBlock, and at what granularity. Read by
    /// `runtime::guest_ram_map` through the granularity latch, by the import
    /// site, and by nothing else.
    pub host_pointer: HostPointerCaps,
    /// Whether descriptor writes may be recorded directly into a command
    /// buffer, and the maximum total descriptors in such a set layout.
    pub push_descriptor: PushDescriptorCaps,
    /// `VK_KHR_portability_subset` was advertised. Kept for the selection log
    /// line and for constructing [`DriverQuirk`] — never gate behavior on it
    /// directly.
    pub portability_subset: bool,
    /// Device `apiVersion` as reported, for the selection log line.
    pub device_api_version: u32,
    pub device_type: vk::PhysicalDeviceType,
}

impl HostGpuCaps {
    /// Flags to request for `class` on this device.
    pub fn memory_request(&self, class: MemoryClass) -> crate::memory::MemoryRequest {
        self.memory_policy().request(class)
    }

    /// The only behavior-bearing interpretation of the structural topology.
    pub const fn memory_policy(&self) -> crate::policy::MemoryPlacementPolicy {
        crate::policy::MemoryPlacementPolicy::new(self.memory.topology)
    }

    /// One-shot, fail-visible summary of the classification. Load-bearing for
    /// portability debugging: it names the memory topology, the signal that
    /// decided it, and the heap sizes that signal was read from. Every field is
    /// something the device reported.
    pub fn selection_line(&self, device_name: &str) -> String {
        format!(
            "vk_caps api={} baseline={} memory={} memory_signal={} device_local_mb={} host_visible_device_local_mb={} host_pointer_import={} host_pointer_handle={} host_pointer_align={} host_pointer_heap_mb={} max_alloc_mb={} push_descriptors={} portability_subset={} type={:?} name={device_name:?}",
            crate::api_floor::version_str(self.device_api_version),
            crate::api_floor::version_str(crate::api_floor::MIN_SUPPORTED_API),
            self.memory.topology.slug(),
            self.memory.signal.slug(),
            self.memory.device_local_bytes >> 20,
            self.memory.host_visible_device_local_bytes >> 20,
            self.host_pointer.rung.slug(),
            self.host_pointer.handle_slug(),
            self.host_pointer.min_alignment,
            self.host_pointer.heap_budget >> 20,
            self.max_allocation_size >> 20,
            self.push_descriptor.max_descriptors,
            self.portability_subset,
            self.device_type,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::fixtures;

    fn caps(api: u32, props: &vk::PhysicalDeviceMemoryProperties) -> HostGpuCaps {
        HostGpuCaps {
            memory: crate::memory::classify_memory(props),
            // A real reported limit rather than a sentinel: it is on the
            // selection line, and 4 GiB is the AMD APU value that made an
            // unchecked import invalid usage.
            max_allocation_size: 4 << 30,
            quirks: DriverQuirk::default(),
            host_pointer: HostPointerCaps {
                rung: HostPointerImport::Supported,
                handle_type: vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                min_alignment: 4096,
                heap_budget: 8 << 30,
                span_max: 4 << 30,
            },
            push_descriptor: PushDescriptorCaps::default(),
            portability_subset: false,
            device_api_version: api,
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
        }
    }

    /// The selection line names the topology, the signal that decided it, and the
    /// heap sizes that signal was read from — what a portability bug report needs.
    /// Every assertion here is a field a reader greps for.
    #[test]
    fn selection_line_carries_the_diagnosis() {
        let c = caps(vk::API_VERSION_1_3, &fixtures::nvidia_discrete());
        let line = c.selection_line("NVIDIA GeForce RTX 5080");
        assert!(line.contains("memory=discrete"), "{line}");
        assert!(line.contains("memory_signal=separate_host_heap"), "{line}");
        assert!(line.contains("device_local_mb=16384"), "{line}");
        // The baseline is stated on every line so no reader mistakes the
        // device's reported version for a requirement.
        assert!(line.contains("baseline=1.2"), "{line}");
    }

    /// The host-pointer rung and its granularity are on the same line, and the
    /// granularity is there because it is measured rather than assumed: a
    /// MoltenVK device reports Apple's page size and a Linux driver may report
    /// more, and "the import refused" and "the import ran at a granularity
    /// nobody expected" are two different bug reports.
    #[test]
    fn the_selection_line_carries_the_host_pointer_rung_and_its_granularity() {
        let line = caps(vk::API_VERSION_1_2, &fixtures::apple_m3_max()).selection_line("Apple");
        assert!(line.contains("host_pointer_import=supported"), "{line}");
        assert!(
            line.contains("host_pointer_handle=host_allocation"),
            "{line}"
        );
        assert!(line.contains("host_pointer_align=4096"), "{line}");
        // The heap ceiling is on the line because it is half of the reported
        // `radv`/`amdgpu` failure: an operator whose guest is larger than this
        // number has a host that cannot import it, and no other field says so.
        assert!(line.contains("host_pointer_heap_mb=8192"), "{line}");
        // `maxMemoryAllocationSize` is the other half of the same report: a
        // host whose limit is below its heap refuses imports the heap could
        // have held, and no other field on this line says so.
        assert!(line.contains("max_alloc_mb=4096"), "{line}");

        let mut refused = caps(vk::API_VERSION_1_2, &fixtures::apple_m3_max());
        refused.host_pointer = HostPointerCaps::default();
        let line = refused.selection_line("Apple");
        assert!(line.contains("host_pointer_import=unqueried"), "{line}");
        assert!(line.contains("host_pointer_handle=none"), "{line}");
        // A refused rung carries no granularity, so the line cannot suggest one
        // an import site could act on.
        assert!(line.contains("host_pointer_align=0"), "{line}");
        assert!(line.contains("host_pointer_heap_mb=0"), "{line}");
    }

    /// A host that cannot import guest RAM says so on the same line, naming the
    /// check that refused rather than just the absence. "This host is slow" and
    /// "this host cannot import guest RAM" are the same bug report, and this is
    /// the field that separates them without a second boot.
    #[test]
    fn the_selection_line_names_the_rung_that_refused_the_import() {
        let mut c = caps(vk::API_VERSION_1_2, &fixtures::apple_m3_max());
        c.host_pointer = HostPointerCaps {
            rung: HostPointerImport::NoHostPointerExtension,
            handle_type: vk::ExternalMemoryHandleTypeFlags::empty(),
            min_alignment: 0,
            heap_budget: 0,
            span_max: 0,
        };
        let line = c.selection_line("Apple M3 Max");
        assert!(
            line.contains("host_pointer_import=no_host_pointer_extension"),
            "{line}"
        );
    }

    /// The API version does not change the classification. Getting this wrong is
    /// how the retired tier axis smuggled a capability in under "is 1.3".
    #[test]
    fn the_api_version_does_not_change_the_classification() {
        let props = fixtures::intel_igpu();
        for api in [
            vk::API_VERSION_1_2,
            vk::API_VERSION_1_3,
            vk::make_api_version(0, 1, 4, 334),
        ] {
            let line = caps(api, &props).selection_line("dev");
            assert!(line.contains("memory=unified"), "{line}");
        }
    }

    /// A unified-memory host classifies as unified whatever else it reports, and
    /// the line is where "why is this host slow" starts.
    #[test]
    fn a_unified_memory_host_says_so() {
        let line =
            caps(vk::API_VERSION_1_2, &fixtures::apple_m3_max()).selection_line("Apple M3 Max");
        assert!(line.contains("memory=unified"), "{line}");
        assert!(line.contains("memory_signal="), "{line}");
    }

    /// Quirks are derived from portability-subset in ONE place, so no other
    /// site needs to know what that extension implies.
    #[test]
    fn quirks_derive_from_portability_subset_once() {
        let off = DriverQuirk::for_portability_subset(false);
        assert!(!off.no_deferred_draw_batching);
        assert!(!off.guest_pages_stay_authoritative);
        let on = DriverQuirk::for_portability_subset(true);
        assert!(on.no_deferred_draw_batching);
        assert!(on.guest_pages_stay_authoritative);
    }
}
