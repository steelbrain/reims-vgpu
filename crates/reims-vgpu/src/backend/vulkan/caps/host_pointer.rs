//! Whether this device can import a host pointer over guest RAM as
//! `VkDeviceMemory`, and the extension that answer implies.
//!
//! # Why a host pointer, and why it is bounded
//!
//! `VK_EXT_external_memory_host` is the one primitive that spans every host
//! this device targets: Linux, Windows, and macOS through MoltenVK, which
//! implements it over `newBufferWithBytesNoCopy` — the same call the
//! Metal-direct arm makes. dma-buf is a Linux kernel object and always will be,
//! and `VK_KHR_external_memory_win32` is not the Windows equivalent: it moves
//! NT handles for GPU-allocated or D3D resources, not arbitrary host pointers.
//!
//! What the extension does not carry is a bound. That is
//! [`crate::runtime::guest_ram`]'s job, and it is a type rather than a rule: an
//! import is sized to a RAMBlock exactly, and every reference inside it is a
//! `GuestSlice` that no call site can construct without the bounds check. Read
//! that module's doc before adding an import site; nothing here re-states it,
//! and nothing scans for violations of it.
//!
//! # Query and enable together
//!
//! Same rule as [`super::device_features`], for the same reason: the extension
//! is named here and nowhere else, so no site can bind an import the device was
//! never asked to support. [`HostPointerImport::required_extensions`] is the
//! only producer of that string.
//!
//! # Optional, and switchable off
//!
//! The extension is never a requirement. A host that does not advertise it
//! reaches [`HostPointerImport::NoHostPointerExtension`], asks for nothing at
//! `vkCreateDevice`, and runs every guest-memory rail through the copying path.
//! Those rails are not a legacy arm — they are the only arm on such a host, and
//! they are the arm a discrete GPU takes regardless, because there the copy into
//! VRAM is the point.
//!
//! [`crate::env::GUEST_IMPORT`] adds one more rung: an operator can take a host
//! that *is* capable down to [`HostPointerImport::DisabledByEnv`]. That is the
//! only way to exercise the copying rails on a machine where the import works,
//! so a regression in them is findable without hunting for hardware that lacks
//! the extension. It cannot run the other way — see [`crate::env`] for why no
//! variable may widen a measured capability.

use ash::vk;

/// Buffer usage every imported guest-memory buffer is created with, and
/// therefore the usage [`query`] asks the device about.
///
/// Importability is a property of the handle type **and the usage**, so asking
/// about a narrower set than the import site binds is a query that can answer
/// yes to a bind the driver then refuses. The set is the union of every
/// direction the rail runs in:
///
/// * `TRANSFER_SRC` — guest pages as the source of an upload into a
///   device-local image or buffer.
/// * `TRANSFER_DST` — guest pages as the destination of a render or compute
///   result, which is the writeback the deferred-flush rail otherwise stages
///   through the CPU.
/// * `VERTEX_BUFFER` / `INDEX_BUFFER` / `STORAGE_BUFFER` / `UNIFORM_BUFFER` —
///   guest pages bound directly to a draw, with no copy at all.
///
/// **Buffers only, deliberately.** An image has implementation-defined tiling,
/// and an optimally-tiled image backed by linear guest bytes is not a thing
/// this device may assume works on an unknown driver. A buffer's memory layout
/// is the bytes themselves on every implementation, which is why it is the
/// universal primitive here and why guest pages reach an image through a
/// `vkCmdCopyBufferToImage` from an imported buffer rather than by being one.
pub const GUEST_IMPORT_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw()
        | vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw(),
);

/// Whether guest RAM can reach this device as a host-pointer import, and when
/// it cannot, which check said so.
///
/// Rungs rather than a bool because every negative rung is a different host and
/// a different answer for a bug report: a Linux ICD without the extension, a
/// device that advertises it and still declines the handle type for this usage,
/// and a RAMBlock whose base cannot meet the device's own import granularity
/// are three separate findings, and only the last is about this guest's memory
/// rather than about the driver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HostPointerImport {
    /// The extension is advertised and the device reports the host-allocation
    /// handle type importable for [`GUEST_IMPORT_USAGE`]. The only rung on
    /// which the import path may run.
    Supported,
    /// No [`query`] has been run. The default, so a [`super::HostGpuCaps`] built
    /// without one never claims a capability nothing checked for.
    #[default]
    Unqueried,
    /// `VK_EXT_external_memory_host` absent. Without it there is no
    /// `vkGetMemoryHostPointerPropertiesEXT` to ask which memory types accept
    /// the pointer, and nothing to chain onto `vkAllocateMemory` to import
    /// through.
    NoHostPointerExtension,
    /// The extension is advertised, and the device still declines
    /// `HOST_ALLOCATION_EXT` as an importable handle type for
    /// [`GUEST_IMPORT_USAGE`].
    NotImportable,
    /// The device reported a `minImportedHostPointerAlignment` that no RAMBlock
    /// can satisfy — zero, or not a power of two. Both make the granularity
    /// arithmetic in [`crate::runtime::guest_ram`] meaningless, and a driver
    /// reporting either is broken rather than restrictive.
    ///
    /// A RAMBlock whose *base* merely needs rounding is not this rung: the
    /// import trims forward to the granule and refuses per region, which is a
    /// property of one block rather than of the device.
    AlignmentUnsatisfiable,
    /// [`crate::env::GUEST_IMPORT`] was set off. The only rung that is a
    /// statement about policy rather than about the host: this device may well
    /// be capable, and the operator asked for the copying rails anyway.
    /// Distinct from every rung above precisely so a log does not read as a
    /// hardware limitation when it is a switch someone left set.
    DisabledByEnv,
}

impl HostPointerImport {
    /// The one place the import path asks whether it may run.
    pub fn is_available(self) -> bool {
        matches!(self, Self::Supported)
    }

    /// Stable slug for the `reason=` field of a decline, and for the capability
    /// line. Named per rung so a log says which check refused.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Unqueried => "unqueried",
            Self::NoHostPointerExtension => "no_host_pointer_extension",
            Self::NotImportable => "not_importable",
            Self::AlignmentUnsatisfiable => "alignment_unsatisfiable",
            Self::DisabledByEnv => "disabled_by_env",
        }
    }

    /// Device extension names this rung requires. Only [`Self::Supported`]
    /// requires any: asking for an extension on a rung that cannot use it would
    /// fail device creation on the hosts that do not have it, which is every
    /// host the negative rungs describe.
    pub fn required_extensions(self) -> Vec<*const std::os::raw::c_char> {
        if self.is_available() {
            vec![vk::EXT_EXTERNAL_MEMORY_HOST_NAME.as_ptr()]
        } else {
            Vec::new()
        }
    }
}

/// What the device answered, together with the granularity an import must meet.
///
/// The pair travels together because they are one answer: a rung of
/// [`HostPointerImport::Supported`] with no granularity is not actionable, and
/// a granularity from a device that declined the handle type is a number
/// nothing may act on. Splitting them is how a site ends up importing at an
/// alignment the device never agreed to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostPointerCaps {
    /// Which check answered.
    pub rung: HostPointerImport,
    /// `minImportedHostPointerAlignment` as the device reported it. Both the
    /// imported pointer and the imported length must be multiples of it.
    ///
    /// Zero on every rung but [`HostPointerImport::Supported`], and never
    /// assumed to be 4096: MoltenVK reports Apple's page size, and a Linux
    /// driver may report more than either.
    pub min_alignment: u64,
}

impl HostPointerCaps {
    /// A negative answer with no granularity, for the rungs that have none.
    fn refused(rung: HostPointerImport) -> Self {
        Self {
            rung,
            min_alignment: 0,
        }
    }

    /// Whether the import path may run.
    pub fn is_available(self) -> bool {
        self.rung.is_available()
    }
}

/// What [`crate::env::GUEST_IMPORT`] says about running this rail at all.
///
/// `None` to go on and ask the device. `Some` short-circuits [`query`], which is
/// deliberate on two counts: the device is not asked about a handle type nothing
/// will import, and the extension is then not named at `vkCreateDevice`, so the
/// switch produces exactly the device a host without it would get rather than a
/// capable device with one gate closed.
///
/// [`crate::env::Switch::On`] is not a way to turn the rail on — no variable may
/// widen a measured capability — but it is not ignored either: an operator who
/// set it has stated an expectation, and if the device then refuses, the
/// `vk_caps` line names the rung that refused. Only the unrecognized case is
/// reported here, because that is the one an operator would otherwise read as
/// "the switch did nothing" with no way to tell a typo from a device that
/// declined.
fn env_override() -> Option<HostPointerImport> {
    match crate::env::read(crate::env::GUEST_IMPORT) {
        (crate::env::Switch::Off, _) => Some(HostPointerImport::DisabledByEnv),
        (crate::env::Switch::Unrecognized, value) => {
            crate::observe::fail(format!(
                "vk_guest_import_env_unrecognized var={} value={:?} (expected on|off; the rail is \
                 left to the device)",
                crate::env::GUEST_IMPORT,
                value.unwrap_or_default()
            ));
            None
        }
        (crate::env::Switch::On | crate::env::Switch::Unset, _) => None,
    }
}

/// Resolve host-pointer importability against one physical device.
///
/// `has_extension` is passed in rather than enumerated here because the caller
/// already enumerates device extensions for the other capability queries.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
) -> HostPointerCaps {
    if let Some(disabled) = env_override() {
        return HostPointerCaps::refused(disabled);
    }
    if !has_extension(vk::EXT_EXTERNAL_MEMORY_HOST_NAME) {
        return HostPointerCaps::refused(HostPointerImport::NoHostPointerExtension);
    }
    // `vkGetPhysicalDeviceExternalBufferProperties` is Vulkan 1.1 core and the
    // baseline is 1.2, so this is always answerable once the handle type itself
    // is spelled by an advertised extension.
    let info = vk::PhysicalDeviceExternalBufferInfo::default()
        .usage(GUEST_IMPORT_USAGE)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let mut props = vk::ExternalBufferProperties::default();
    unsafe { instance.get_physical_device_external_buffer_properties(pd, &info, &mut props) };
    if !props
        .external_memory_properties
        .external_memory_features
        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
    {
        return HostPointerCaps::refused(HostPointerImport::NotImportable);
    }

    // The granularity is the device's, not ours, and it is not 4096 everywhere:
    // MoltenVK reports Apple's page size and a Linux driver may report more.
    // Asking is the whole reason this query exists rather than a constant.
    let mut host_props = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut host_props);
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    let min_alignment = host_props.min_imported_host_pointer_alignment;
    if min_alignment == 0 || !min_alignment.is_power_of_two() {
        return HostPointerCaps::refused(HostPointerImport::AlignmentUnsatisfiable);
    }

    HostPointerCaps {
        rung: HostPointerImport::Supported,
        min_alignment,
    }
}

/// Which memory type an import of `host_ptr` must use, or `None` when the
/// device accepts none that also meet `req`.
///
/// `vkGetMemoryHostPointerPropertiesEXT` answers with a `memoryTypeBits` mask
/// that is a property of the *pointer*, not of the device, so it cannot be
/// resolved once at capability time. It is fed through
/// [`super::memory_topology::select_memory_type`] rather than being ranked here:
/// a second selection site is a second policy, and the two would diverge on the
/// first host where the ranking mattered.
///
/// # Safety
///
/// `device` must be the logical device the import will be made on, `ext` its
/// loaded `VK_EXT_external_memory_host` entry points, and `host_ptr` a live
/// mapping in this process aligned to the device's
/// [`HostPointerCaps::min_alignment`].
pub unsafe fn import_memory_type(
    ext: &ash::ext::external_memory_host::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    host_ptr: *const std::ffi::c_void,
    req: &super::memory_topology::MemoryRequest,
) -> Option<u32> {
    let mut ptr_props = vk::MemoryHostPointerPropertiesEXT::default();
    // `ash` 0.38 wraps this extension as raw function pointers only, so the
    // call goes through `fp()` rather than a checked method.
    //
    // SAFETY: the caller's contract supplies a live, correctly aligned mapping
    // and the extension's own device-loaded entry points; `ptr_props` is a
    // correctly initialized out-struct with its `sType` set by `default()`.
    let rc = unsafe {
        (ext.fp().get_memory_host_pointer_properties_ext)(
            ext.device(),
            vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
            host_ptr,
            &mut ptr_props,
        )
    };
    if rc != vk::Result::SUCCESS {
        return None;
    }
    super::memory_topology::select_memory_type(memory_props, ptr_props.memory_type_bits, req)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every rung, so a rung added without a test here fails to compile rather
    /// than quietly skipping the invariants below.
    const RUNGS: [HostPointerImport; 6] = [
        HostPointerImport::Supported,
        HostPointerImport::Unqueried,
        HostPointerImport::NoHostPointerExtension,
        HostPointerImport::NotImportable,
        HostPointerImport::AlignmentUnsatisfiable,
        HostPointerImport::DisabledByEnv,
    ];

    /// Only the supported rung asks for an extension string. Requesting
    /// `VK_EXT_external_memory_host` on a host that does not advertise it fails
    /// `vkCreateDevice` outright — so a negative rung that still named its
    /// extension would turn "no zero-copy here" into "no Vulkan here", on
    /// exactly the hosts the rung exists to describe.
    #[test]
    fn only_the_supported_rung_names_an_extension() {
        assert_eq!(HostPointerImport::Supported.required_extensions().len(), 1);
        for rung in RUNGS.into_iter().filter(|r| !r.is_available()) {
            assert!(
                rung.required_extensions().is_empty(),
                "{rung:?} must not request an extension it cannot use"
            );
            assert!(!rung.is_available(), "{rung:?} must not gate the rail open");
        }
    }

    /// The default rung is "nobody asked", not "no". Both refuse the rail, but
    /// only one of them is honest about a `HostGpuCaps` that was never queried —
    /// and the slug is what a reader greps to tell them apart.
    #[test]
    fn the_default_rung_says_it_was_never_queried() {
        assert_eq!(HostPointerImport::default(), HostPointerImport::Unqueried);
        assert_eq!(HostPointerImport::default().slug(), "unqueried");
        assert_eq!(HostPointerCaps::default().min_alignment, 0);
        assert!(!HostPointerCaps::default().is_available());
    }

    /// One slug per rung: two rungs sharing one would mean watching the slug
    /// fire in the log and still not knowing which check refused.
    #[test]
    fn every_rung_has_its_own_slug() {
        let mut slugs: Vec<_> = RUNGS.iter().map(|r| r.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two rungs share a slug");
        assert!(slugs
            .iter()
            .all(|s| s.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')));
    }

    /// A negative rung carries no granularity. A non-zero alignment beside a
    /// refusal is a number an import site could act on, and every one of those
    /// rungs describes a device that would reject the import.
    #[test]
    fn a_refused_rung_carries_no_granularity() {
        for rung in RUNGS.into_iter().filter(|r| !r.is_available()) {
            let caps = HostPointerCaps::refused(rung);
            assert_eq!(caps.min_alignment, 0, "{rung:?}");
            assert!(!caps.is_available(), "{rung:?}");
        }
    }

    /// The usage set queried is the usage set bound. Asking about a narrower set
    /// than the import site creates its buffer with is a query that can answer
    /// yes to a bind the driver then refuses — the failure would land at
    /// `vkCreateBuffer` on a guest frame rather than at capability time.
    #[test]
    fn the_queried_usage_covers_both_directions_of_the_rail() {
        // Guest pages as a GPU *source* — the upload the CPU otherwise gathers.
        assert!(GUEST_IMPORT_USAGE.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        // Guest pages as a GPU *destination* — the writeback the deferred-flush
        // rail otherwise stages through the CPU, and the larger half of the cost.
        assert!(GUEST_IMPORT_USAGE.contains(vk::BufferUsageFlags::TRANSFER_DST));
        // Bound straight to a draw, with no copy in either direction.
        for direct in [
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        ] {
            assert!(GUEST_IMPORT_USAGE.contains(direct));
        }
    }

    /// Set [`crate::env::GUEST_IMPORT`] to `value`, read the override, and
    /// restore. One test at a time: the variable is process-global.
    fn with_env(value: Option<&str>) -> Option<HostPointerImport> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock serializes every mutation of this variable in this
        // process, and the only reader is `env_override`, called below.
        unsafe {
            match value {
                Some(v) => std::env::set_var(crate::env::GUEST_IMPORT, v),
                None => std::env::remove_var(crate::env::GUEST_IMPORT),
            }
        }
        let out = env_override();
        unsafe { std::env::remove_var(crate::env::GUEST_IMPORT) };
        out
    }

    /// The switch turns the rail off, and lands on a rung that says a person did
    /// it. Reading `no_host_pointer_extension` for a host that has the extension
    /// would send the next bug report hunting for a driver problem.
    #[test]
    fn the_env_switch_takes_a_capable_host_down() {
        assert_eq!(with_env(Some("0")), Some(HostPointerImport::DisabledByEnv));
        assert_eq!(with_env(Some("off")), Some(HostPointerImport::DisabledByEnv));
        assert!(!HostPointerImport::DisabledByEnv.is_available());
        assert!(HostPointerImport::DisabledByEnv
            .required_extensions()
            .is_empty());
        assert_eq!(HostPointerImport::DisabledByEnv.slug(), "disabled_by_env");
    }

    /// The switch has no on direction. Setting it affirmatively hands the answer
    /// straight back to the device — which is the whole rule from [`crate::env`]:
    /// a variable may narrow what this device does and may never widen it,
    /// because binding an extension the host does not advertise fails
    /// `vkCreateDevice` and importing a handle type it declines is undefined
    /// behavior in the driver.
    #[test]
    fn the_env_switch_cannot_turn_the_rail_on() {
        for on in ["1", "on", "true", "yes"] {
            assert_eq!(with_env(Some(on)), None, "{on} must not preempt the query");
        }
        assert_eq!(with_env(None), None);
    }

    /// A misspelled value leaves the device to decide, rather than guessing at
    /// an intent. `env::read` keeps the raw value so the line above names it.
    #[test]
    fn an_unrecognized_value_does_not_change_the_answer() {
        assert_eq!(with_env(Some("maybe")), None);
    }
}
