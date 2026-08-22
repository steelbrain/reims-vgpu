//! Host GPU memory-topology classification and allocation policy.
//!
//! The engine supports **two** memory topologies as first-class targets:
//!
//! | Topology | Examples | Property that matters |
//! |---|---|---|
//! | [`MemoryTopology::Unified`] | Apple M-series, Intel/AMD iGPU, llvmpipe | GPU-local memory *is* cached CPU memory, so a staging hop is pure waste |
//! | [`MemoryTopology::Discrete`] | RTX 5080, GTX 750 Ti, Arc, RX 7900 | GPU-local memory is across PCIe, so bulk transfer must be a DMA, not a CPU map |
//!
//! ## The classification rule
//!
//! A device is `Unified` when **either** structural signal holds:
//!
//! * **A — no separate host heap:** every advertised memory heap is
//!   `DEVICE_LOCAL`. A discrete GPU always advertises the system-RAM heap
//!   without `DEVICE_LOCAL`; a single-pool device has no reason to.
//! * **B — cached direct access:** some memory type carries
//!   `DEVICE_LOCAL | HOST_VISIBLE | HOST_CACHED`. A PCIe BAR window into VRAM
//!   is mapped write-combining and is advertised `HOST_COHERENT` *without*
//!   `HOST_CACHED` (this is what distinguishes a resizable-BAR discrete GPU,
//!   which does expose a whole-VRAM host-visible type, from a real UMA part).
//!
//! Signal B exists so an AMD APU that advertises its GTT heap without
//! `DEVICE_LOCAL` is not mistaken for a discrete part.
//!
//! ## Misclassification is a performance bug, never a correctness bug
//!
//! Topology selects a *preference order* only. [`select_memory_type`] always
//! falls back to the required flags alone, so a device classified the wrong way
//! still allocates valid memory — it just pays a copy it did not need (or skips
//! one it wanted). Nothing in the engine may branch on topology in a way that
//! changes what the guest observes; that invariant is asserted by the tests at
//! the bottom of this file.
//!
//! ## Flags are not the whole selection: capacity is the other axis
//!
//! A preference expressed in flags cannot see how much of a pool there is, and
//! on a part whose `DEVICE_LOCAL` heap is a carve-out the two answers disagree.
//! The APU shape in the `amd_apu_host_heap` test fixture is the worked case: a 2 GiB
//! device-local carve-out beside 14 GiB of system RAM, classified `Unified`, so
//! `MemoryClass::Upload` prefers `DEVICE_LOCAL` and the whole-RAMBlock
//! guest-memory import — 16 GiB on a 16 GiB guest — is charged to the 2 GiB
//! pool. Nothing moves as a result: an imported host pointer's pages are where
//! the host mapping put them, and the memory type only decides which heap the
//! driver accounts them to and manages residency against.
//!
//! So [`select_memory_type`] takes the allocation's size and tries every
//! preference against heaps that could hold it, and where no heap can hold it at
//! all it refuses by name rather than nominating the roomiest one. That is not a
//! capacity policy this module invented: an `allocationSize` past its heap's
//! `size`, or past `maxMemoryAllocationSize`, is invalid usage, and one of the
//! two Mesa drivers this was reported on accepts the call and loses the device a
//! second later. [`MemoryTypeRefusal`] carries which bound was crossed.

use ash::vk;

/// How host and device memory relate on the bound physical device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryTopology {
    /// One physical pool shared by CPU and GPU (Apple M-series, Intel/AMD iGPU).
    Unified,
    /// Device memory is separate from system RAM and reached over a bus.
    Discrete,
}

impl MemoryTopology {
    /// Stable slug for logs, proxy lines, and the matrix cell name.
    pub fn slug(self) -> &'static str {
        match self {
            Self::Unified => "unified",
            Self::Discrete => "discrete",
        }
    }
}

/// What a piece of memory is *for*. The engine asks for a class; this module
/// turns the class into flags. No call site spells `MemoryPropertyFlags`
/// directly, so a topology decision is made in exactly one place.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    /// CPU writes once, GPU reads: vertex/index/uniform/staging uploads.
    Upload,
    /// GPU writes, CPU reads: present capture and target readback.
    Readback,
    /// GPU-only working set: render targets, sampled textures, scratch buffers.
    DeviceLocal,
    /// GPU-side working memory that *wants* to be device-local but must never
    /// fail to allocate — the host-window staging image, where a
    /// host-memory placement is slower but still correct. Distinct from
    /// [`MemoryClass::DeviceLocal`], which is a hard requirement.
    DeviceLocalPreferred,
}

/// A memory-type query: the flags that MUST be present, plus a ranked list of
/// bonus flag sets, best first. [`select_memory_type`] takes the first
/// candidate matching the highest-ranked bonus it can satisfy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRequest {
    pub required: vk::MemoryPropertyFlags,
    /// Best-first. An empty list means "required flags only".
    pub preferred: Vec<vk::MemoryPropertyFlags>,
}

// Placement consequences of `MemoryTopology` are owned by
// `policy::MemoryPlacementPolicy`. Unified upload prefers DEVICE_LOCAL while
// discrete upload avoids spending a scarce BAR window. Unified readback
// prefers DEVICE_LOCAL|HOST_CACHED while discrete readback prefers ordinary
// HOST_CACHED memory. Readback requires only HOST_VISIBLE: coherence remains a
// last preference, and a selected non-coherent type owes an explicit
// invalidate before CPU access.

/// Summary of the bound device's memory layout: the topology plus the sizes the
/// VRAM proxy reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryProfile {
    pub topology: MemoryTopology,
    /// Bytes in the largest `DEVICE_LOCAL` heap.
    pub device_local_bytes: u64,
    /// Bytes in the largest heap reachable through a `DEVICE_LOCAL|HOST_VISIBLE`
    /// memory type — 0 when the CPU cannot address device memory at all.
    pub host_visible_device_local_bytes: u64,
    /// Which structural signal decided the classification (for the one-shot
    /// selection log line; a misclassification report must say *why*).
    pub signal: TopologySignal,
}

/// Which rule fired in [`classify_memory`]. Recorded so a wrong call on an
/// unfamiliar driver can be diagnosed from one log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TopologySignal {
    /// Signal A: every heap is `DEVICE_LOCAL`.
    NoHostOnlyHeap,
    /// Signal B: a `DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED` type exists.
    CachedDeviceLocal,
    /// Neither signal fired.
    SeparateHostHeap,
}

impl TopologySignal {
    pub fn slug(self) -> &'static str {
        match self {
            Self::NoHostOnlyHeap => "no_host_only_heap",
            Self::CachedDeviceLocal => "cached_device_local",
            Self::SeparateHostHeap => "separate_host_heap",
        }
    }
}

/// Classify a device's memory layout. Pure over
/// `VkPhysicalDeviceMemoryProperties`, so every row of the support matrix is
/// testable without that GPU present.
pub fn classify_memory(props: &vk::PhysicalDeviceMemoryProperties) -> MemoryProfile {
    use vk::MemoryPropertyFlags as F;
    let heaps = &props.memory_heaps[..props.memory_heap_count as usize];
    let types = &props.memory_types[..props.memory_type_count as usize];

    // Signal A — no heap lacks DEVICE_LOCAL. An empty heap list cannot be
    // unified by omission, so require at least one heap.
    let no_host_only_heap = !heaps.is_empty()
        && heaps
            .iter()
            .all(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL));

    // Signal B — the CPU can *cache* a device-local mapping. A PCIe BAR window
    // is write-combining and never advertises HOST_CACHED.
    let cached_device_local = types.iter().any(|t| {
        t.property_flags
            .contains(F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_CACHED)
    });

    let (topology, signal) = if no_host_only_heap {
        (MemoryTopology::Unified, TopologySignal::NoHostOnlyHeap)
    } else if cached_device_local {
        (MemoryTopology::Unified, TopologySignal::CachedDeviceLocal)
    } else {
        (MemoryTopology::Discrete, TopologySignal::SeparateHostHeap)
    };

    let heap_bytes = |idx: u32| heaps.get(idx as usize).map_or(0, |h| h.size);
    let device_local_bytes = heaps
        .iter()
        .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|h| h.size)
        .max()
        .unwrap_or(0);
    let host_visible_device_local_bytes = types
        .iter()
        .filter(|t| t.property_flags.contains(F::DEVICE_LOCAL | F::HOST_VISIBLE))
        .map(|t| heap_bytes(t.heap_index))
        .max()
        .unwrap_or(0);

    MemoryProfile {
        topology,
        device_local_bytes,
        host_visible_device_local_bytes,
        signal,
    }
}

/// `maxMemoryAllocationSize` — the device-wide ceiling on one
/// `vkAllocateMemory`, which no memory type and no heap can widen
/// (VUID-vkAllocateMemory-pAllocateInfo-01713 bounds an allocation by its heap,
/// -01714 by this).
///
/// Vulkan 1.1 core and the baseline is 1.2, so every supported device answers
/// it. A device reporting zero has not filled the field in at all — the spec has
/// no zero-sized allocation limit — so this returns `None`. The composition
/// layer reports that omission and may then use the per-heap bound alone.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn reported_max_allocation_size(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> Option<u64> {
    let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
    let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut v11);
    unsafe { instance.get_physical_device_properties2(pd, &mut props) };
    (v11.max_memory_allocation_size != 0).then_some(v11.max_memory_allocation_size)
}

/// Query the device-wide allocation ceiling, treating an unreported limit as
/// heap-bounded while keeping that loss of precision visible.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn max_allocation_size(instance: &ash::Instance, pd: vk::PhysicalDevice) -> u64 {
    match unsafe { reported_max_allocation_size(instance, pd) } {
        Some(size) => size,
        None => {
            reims_vgpu_observe::fail(
                "vk_max_allocation_unreported reason=vk_max_allocation_unreported (the device \
                 reported maxMemoryAllocationSize=0; allocations are bounded by their heap alone)",
            );
            u64::MAX
        }
    }
}

/// A selected memory type, together with the heap behind it.
///
/// The heap travels with the index because the index alone cannot say whether
/// the allocation about to be made has anywhere to live: two types carrying
/// identical flags can sit on a 2 GiB carve-out and a 14 GiB pool, and picking
/// the wrong one is invisible in every log that prints only the index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryTypePick {
    /// Index into `VkPhysicalDeviceMemoryProperties::memoryTypes`.
    pub index: u32,
    /// The heap that type draws from.
    pub heap_index: u32,
    /// `VkMemoryHeap::size` for that heap, which is at least the bytes the pick
    /// was made for — see [`select_memory_type`], which cannot return a pick
    /// whose heap could not hold the allocation.
    pub heap_bytes: u64,
}

/// Which check refused to name a memory type for an allocation.
///
/// Two of the three are the two valid-usage statements Vulkan places on
/// `VkMemoryAllocateInfo::allocationSize`, and they are refusals rather than
/// warnings for the reason in [`select_memory_type`]'s doc: a driver is not
/// required to reject an allocation that violates them, and one that accepts it
/// has been asked for undefined behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryTypeRefusal {
    /// No type in `type_bits` carries the request's required flags. A pure
    /// flags answer: this device offers nothing of the kind asked for.
    NoTypeWithRequiredFlags { type_bits: u32 },
    /// `bytes` is past `maxMemoryAllocationSize`
    /// (VUID-vkAllocateMemory-pAllocateInfo-01714). A device-wide limit, so no
    /// memory type could have carried it.
    AboveDeviceMaximum { bytes: u64, max: u64 },
    /// Types with the required flags exist and every heap behind them is
    /// smaller than `bytes`
    /// (VUID-vkAllocateMemory-pAllocateInfo-01713). Carries the roomiest such
    /// heap, which is what says how far past this device the request was.
    EveryHeapTooSmall { bytes: u64, roomiest_heap: u64 },
}

impl MemoryTypeRefusal {
    /// Stable slug for the fail channel.
    pub fn slug(self) -> &'static str {
        match self {
            Self::NoTypeWithRequiredFlags { .. } => "vk_memory_no_type_with_required_flags",
            Self::AboveDeviceMaximum { .. } => "vk_memory_above_device_maximum",
            Self::EveryHeapTooSmall { .. } => "vk_memory_every_heap_too_small",
        }
    }
}

impl std::fmt::Display for MemoryTypeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::NoTypeWithRequiredFlags { type_bits } => {
                write!(f, "{} type_bits={type_bits:#x}", self.slug())
            }
            Self::AboveDeviceMaximum { bytes, max } => write!(
                f,
                "{} bytes_mb={} max_mb={}",
                self.slug(),
                bytes >> 20,
                max >> 20
            ),
            Self::EveryHeapTooSmall {
                bytes,
                roomiest_heap,
            } => write!(
                f,
                "{} bytes_mb={} roomiest_heap_mb={}",
                self.slug(),
                bytes >> 20,
                roomiest_heap >> 20
            ),
        }
    }
}

/// Pick a memory type satisfying `req` within `type_bits`, for an allocation of
/// `bytes` on a device whose `maxMemoryAllocationSize` is `max_allocation`.
///
/// Tries each `preferred` set best-first and then the required flags alone,
/// **restricted at every step to heaps that can hold `bytes`**. A topology
/// misclassification is still a performance bug rather than an allocation
/// failure: the fallback to the required flags alone is what guarantees that,
/// and it is untouched. What this will not do is name a type whose heap could
/// not hold the allocation — see below.
///
/// # Capacity is a validity rule, not a preference
///
/// Vulkan states both bounds on `allocationSize` as valid usage:
/// it must be at most `maxMemoryAllocationSize`, and at most the `size` of
/// `memoryHeaps[memoryTypes[memoryTypeIndex].heapIndex]`. An allocation past
/// either is an invalid call, and a driver is under no obligation to return an
/// error for it — the two behaviors seen on real hosts are Mesa ANV refusing
/// with `VK_ERROR_OUT_OF_DEVICE_MEMORY` and Mesa RADV returning `VK_SUCCESS` and
/// then losing the device a second later, taking the guest with it. So this
/// function refuses by name where it used to hand back the roomiest heap and
/// leave the call to the driver: relying on a refusal only one of two drivers
/// makes is not a bound.
///
/// The condition is not a slow path either. `VkMemoryHeap::size` is the heap's
/// total, so a heap that could not hold the allocation *empty* has nowhere to
/// put it however patient the caller is.
///
/// # Why the size is a parameter and not an afterthought
///
/// Flags say what a memory type *is*; they say nothing about how much of it
/// there is. An AMD APU advertises a `DEVICE_LOCAL|HOST_VISIBLE|HOST_CACHED`
/// type over a 2 GiB VRAM carve-out and a plain `HOST_VISIBLE|HOST_COHERENT`
/// type over 14 GiB of system RAM; classified `Unified`, `MemoryClass::Upload`
/// prefers `DEVICE_LOCAL` and lands every upload — including the whole-RAMBlock
/// guest-memory import, which on a 16 GiB guest is 16 GiB — in the 2 GiB pool.
/// The pages cannot move there: an imported host pointer's placement is fixed by
/// the host mapping, so the only thing the choice changes is which heap the
/// driver charges the allocation to and how it then manages residency against
/// everything else that needs that heap.
///
/// Passing `0` means "no size constraint", which is exactly true of a zero-byte
/// allocation and is what a caller asking a pure flags question wants.
pub fn select_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    req: &MemoryRequest,
    bytes: u64,
    max_allocation: u64,
) -> Result<MemoryTypePick, MemoryTypeRefusal> {
    let heap_index = |index: u32| props.memory_types[index as usize].heap_index;
    let heap_bytes = |index: u32| {
        props
            .memory_heaps
            .get(heap_index(index) as usize)
            .map_or(0, |h| h.size)
    };
    let carries = |index: u32, flags: vk::MemoryPropertyFlags| {
        (type_bits & (1u32 << index)) != 0
            && props.memory_types[index as usize]
                .property_flags
                .contains(flags)
    };
    let find_fitting = |flags: vk::MemoryPropertyFlags| {
        (0..props.memory_type_count)
            .find(|&index| carries(index, flags) && heap_bytes(index) >= bytes)
    };
    // The roomiest heap any candidate type draws from. Zero candidates and a
    // candidate on a zero-sized heap are told apart by the `Option`, because the
    // two are different refusals.
    let roomiest_required = (0..props.memory_type_count)
        .filter(|&index| carries(index, req.required))
        .map(heap_bytes)
        .max();
    let Some(roomiest_heap) = roomiest_required else {
        return Err(MemoryTypeRefusal::NoTypeWithRequiredFlags { type_bits });
    };
    if bytes > max_allocation {
        return Err(MemoryTypeRefusal::AboveDeviceMaximum {
            bytes,
            max: max_allocation,
        });
    }
    let index = req
        .preferred
        .iter()
        .find_map(|bonus| find_fitting(req.required | *bonus))
        .or_else(|| find_fitting(req.required))
        .ok_or(MemoryTypeRefusal::EveryHeapTooSmall {
            bytes,
            roomiest_heap,
        })?;
    Ok(MemoryTypePick {
        index,
        heap_index: heap_index(index),
        heap_bytes: heap_bytes(index),
    })
}

/// The largest heap this device can charge an allocation carrying `req`'s
/// required flags to.
///
/// The bound a caller sizing a whole family of allocations against this device
/// needs, and the one [`select_memory_type`] enforces per allocation. Derived
/// from the same `required` flags the selector filters on, so the two cannot
/// name different populations of memory types.
pub fn roomiest_heap_for(props: &vk::PhysicalDeviceMemoryProperties, req: &MemoryRequest) -> u64 {
    (0..props.memory_type_count)
        .filter(|&index| {
            props.memory_types[index as usize]
                .property_flags
                .contains(req.required)
        })
        .map(|index| {
            props
                .memory_heaps
                .get(props.memory_types[index as usize].heap_index as usize)
                .map_or(0, |h| h.size)
        })
        .max()
        .unwrap_or(0)
}

/// The host-side cost properties of a selected memory type, for a caller that
/// maps it.
///
/// A mapping is only free to read if it is cached, and only free of explicit
/// `vkInvalidate`/`vkFlushMappedMemoryRanges` if it is coherent. Those two are
/// independent, and [`MemoryClass::Readback`] deliberately ranks the first above
/// the second, so an allocation site cannot assume either. Reading the bits is
/// the topology policy's job (nothing outside `caps/` may name a
/// `MemoryPropertyFlags`), which is why this lives here rather than at the site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappedMemoryKind {
    pub cached: bool,
    pub coherent: bool,
}

impl MappedMemoryKind {
    pub fn of(props: &vk::PhysicalDeviceMemoryProperties, index: u32) -> Self {
        use vk::MemoryPropertyFlags as F;
        let flags = props.memory_types[index as usize].property_flags;
        Self {
            cached: flags.contains(F::HOST_CACHED),
            coherent: flags.contains(F::HOST_COHERENT),
        }
    }
}

#[cfg(any(test, feature = "test-fixtures"))]
#[doc(hidden)]
pub mod fixtures {
    //! Synthetic `VkPhysicalDeviceMemoryProperties` for every device family in
    //! the support matrix, so both memory rows are covered by unit tests on any
    //! machine. The Apple layout is transcribed from a live `vulkaninfo` on the
    //! M3 Max dev host; the rest follow each driver's documented layout.
    use ash::vk;

    const GIB: u64 = 1 << 30;

    pub fn build(
        heaps: &[(u64, vk::MemoryHeapFlags)],
        types: &[(u32, vk::MemoryPropertyFlags)],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_heap_count: heaps.len() as u32,
            memory_type_count: types.len() as u32,
            ..Default::default()
        };
        for (i, (size, flags)) in heaps.iter().enumerate() {
            p.memory_heaps[i] = vk::MemoryHeap {
                size: *size,
                flags: *flags,
            };
        }
        for (i, (heap, flags)) in types.iter().enumerate() {
            p.memory_types[i] = vk::MemoryType {
                heap_index: *heap,
                property_flags: *flags,
            };
        }
        p
    }

    /// Apple M3 Max / MoltenVK 1.4.1 — transcribed from `vulkaninfo` on the
    /// arm64 dev host: one 64 GiB DEVICE_LOCAL heap, three types.
    pub fn apple_m3_max() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[(64 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[
                (0, F::DEVICE_LOCAL),
                (
                    0,
                    F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                ),
                (0, F::DEVICE_LOCAL | F::LAZILY_ALLOCATED),
            ],
        )
    }

    /// Intel integrated (Mesa ANV): one DEVICE_LOCAL heap over system RAM.
    ///
    /// Transcribed from `vulkaninfo` on the x86 dev host (Intel ARL,
    /// 2026-07-30): one 70 GiB DEVICE_LOCAL heap and five types, `0x07` and
    /// `0x0b` each appearing twice with a PROTECTED type between them.
    ///
    /// **Nothing here carries `HOST_COHERENT` and `HOST_CACHED` together**, and
    /// that is the load-bearing property. The previous version of this fixture
    /// gave type 1 both bits — invented, not measured — which made every
    /// `MemoryClass::Readback` selection test pass while the real device fell
    /// through to uncached type 0 on every allocation. A fixture more capable
    /// than the hardware it stands for cannot fail the way the hardware does.
    pub fn intel_igpu() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        let coherent = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT;
        let cached = F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_CACHED;
        build(
            &[(70 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[
                (0, coherent),
                (0, cached),
                (0, F::DEVICE_LOCAL | F::PROTECTED),
                (0, coherent),
                (0, cached),
            ],
        )
    }

    /// AMD APU (RADV) advertising its GTT heap WITHOUT `DEVICE_LOCAL`. Signal A
    /// misses here; signal B is what keeps it out of the discrete row.
    pub fn amd_apu_host_heap() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[
                (2 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (14 * GIB, vk::MemoryHeapFlags::empty()),
            ],
            &[
                (0, F::DEVICE_LOCAL),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT),
                (
                    0,
                    F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
                ),
            ],
        )
    }

    /// NVIDIA discrete without resizable BAR: a 256 MiB host-visible window into
    /// 16 GiB of VRAM, plus the system-RAM heap.
    pub fn nvidia_discrete() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[
                (16 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (64 * GIB, vk::MemoryHeapFlags::empty()),
                (256 * 1024 * 1024, vk::MemoryHeapFlags::DEVICE_LOCAL),
            ],
            &[
                (0, F::DEVICE_LOCAL),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED),
                (2, F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT),
            ],
        )
    }

    /// NVIDIA discrete WITH resizable BAR: the whole 16 GiB of VRAM is
    /// host-visible, but write-combining — never `HOST_CACHED`. This is the
    /// fixture that would break a naive "has DEVICE_LOCAL|HOST_VISIBLE ⇒ UMA"
    /// rule.
    pub fn nvidia_discrete_rebar() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[
                (16 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL),
                (64 * GIB, vk::MemoryHeapFlags::empty()),
            ],
            &[
                (0, F::DEVICE_LOCAL),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT),
                (1, F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED),
                (0, F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT),
            ],
        )
    }

    /// Software rasterizer (llvmpipe): one heap, plain host memory.
    pub fn llvmpipe() -> vk::PhysicalDeviceMemoryProperties {
        use vk::MemoryPropertyFlags as F;
        build(
            &[(8 * GIB, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[(
                0,
                F::DEVICE_LOCAL | F::HOST_VISIBLE | F::HOST_COHERENT | F::HOST_CACHED,
            )],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    fn request(topology: MemoryTopology, class: MemoryClass) -> MemoryRequest {
        crate::policy::MemoryPlacementPolicy::new(topology).request(class)
    }

    /// The flags half of [`select_memory_type`], for the tests that are about
    /// which *properties* a class lands on. Zero bytes fits every heap, so the
    /// capacity stage is a no-op and these read as they did before it existed.
    fn pick_index(
        props: &vk::PhysicalDeviceMemoryProperties,
        type_bits: u32,
        req: &MemoryRequest,
    ) -> Option<u32> {
        select_memory_type(props, type_bits, req, 0, u64::MAX)
            .ok()
            .map(|p| p.index)
    }

    /// Every unified-memory device in the matrix classifies unified, and the
    /// signal that fired is recorded.
    #[test]
    fn unified_devices_classify_unified() {
        for (name, props, signal) in [
            ("apple", apple_m3_max(), TopologySignal::NoHostOnlyHeap),
            ("intel", intel_igpu(), TopologySignal::NoHostOnlyHeap),
            (
                "amd-apu",
                amd_apu_host_heap(),
                TopologySignal::CachedDeviceLocal,
            ),
            ("llvmpipe", llvmpipe(), TopologySignal::NoHostOnlyHeap),
        ] {
            let profile = classify_memory(&props);
            assert_eq!(
                profile.topology,
                MemoryTopology::Unified,
                "{name} must classify unified"
            );
            assert_eq!(profile.signal, signal, "{name} signal");
        }
    }

    /// Both discrete shapes classify discrete — including resizable BAR, where
    /// the whole of VRAM IS host-visible. That case is the reason the rule keys
    /// on `HOST_CACHED` rather than on `DEVICE_LOCAL|HOST_VISIBLE`.
    #[test]
    fn discrete_devices_classify_discrete_including_rebar() {
        for (name, props) in [
            ("nvidia", nvidia_discrete()),
            ("nvidia-rebar", nvidia_discrete_rebar()),
        ] {
            let profile = classify_memory(&props);
            assert_eq!(
                profile.topology,
                MemoryTopology::Discrete,
                "{name} must classify discrete"
            );
            assert_eq!(profile.signal, TopologySignal::SeparateHostHeap);
        }
    }

    /// The reported sizes back the VRAM proxy: device-local bytes is the largest
    /// device-local heap, and the host-visible-device-local figure exposes how
    /// small a non-resizable BAR window really is.
    #[test]
    fn profile_reports_heap_sizes() {
        let apple = classify_memory(&apple_m3_max());
        assert_eq!(apple.device_local_bytes, 64 << 30);
        assert_eq!(apple.host_visible_device_local_bytes, 64 << 30);

        let nv = classify_memory(&nvidia_discrete());
        assert_eq!(nv.device_local_bytes, 16 << 30);
        assert_eq!(nv.host_visible_device_local_bytes, 256 << 20);

        let rebar = classify_memory(&nvidia_discrete_rebar());
        assert_eq!(rebar.host_visible_device_local_bytes, 16 << 30);
    }

    /// An empty/absent memory layout must not be called unified by omission.
    #[test]
    fn empty_properties_are_not_unified() {
        let empty = vk::PhysicalDeviceMemoryProperties::default();
        let profile = classify_memory(&empty);
        assert_eq!(profile.topology, MemoryTopology::Discrete);
        assert_eq!(profile.device_local_bytes, 0);
    }

    /// Readback lands in cached memory on every device in the matrix — an
    /// uncached CPU read of a present frame is the difference between a copy and
    /// a stall (measured: 460 MB/s, 70-86 % of all draw time on an Intel iGPU).
    ///
    /// This asserts the **selected type**, not the request. The version it
    /// replaces checked that `preferred` mentioned `HOST_CACHED` and that
    /// `required` contained `HOST_VISIBLE | HOST_COHERENT` — both true, and
    /// together they guaranteed the opposite of what the test was named for on
    /// any device whose cached type is not coherent, because the preference is
    /// only ever tried *with* the requirement. A test that reads the query
    /// cannot see a preference that never matches.
    #[test]
    fn readback_lands_in_cached_memory_on_every_device() {
        let devices: [(&str, _, MemoryTopology); 6] = [
            ("apple", apple_m3_max(), MemoryTopology::Unified),
            ("intel", intel_igpu(), MemoryTopology::Unified),
            ("amd-apu", amd_apu_host_heap(), MemoryTopology::Unified),
            ("llvmpipe", llvmpipe(), MemoryTopology::Unified),
            ("nvidia", nvidia_discrete(), MemoryTopology::Discrete),
            (
                "nvidia-rebar",
                nvidia_discrete_rebar(),
                MemoryTopology::Discrete,
            ),
        ];
        for (name, props, topology) in devices {
            let req = request(topology, MemoryClass::Readback);
            let index = pick_index(&props, !0, &req).unwrap_or_else(|| panic!("{name}: no type"));
            let kind = MappedMemoryKind::of(&props, index);
            assert!(
                kind.cached,
                "{name}: readback took uncached type {index}, so every full-frame \
                 read on that host is an uncached copy"
            );
        }
    }

    /// Coherence is a preference, not a requirement, and the readback path is
    /// what owes the invalidate when it does not get it.
    ///
    /// Intel ANV is the device that forces the distinction: its cached type is
    /// not coherent and its coherent types are not cached, so requiring both
    /// silently selects the uncached one. Apple's single type carries both, so
    /// the same request needs no invalidate there — both rows are asserted
    /// because a change that "fixes" one by breaking the other would otherwise
    /// look green.
    #[test]
    fn a_readback_type_may_be_cached_without_being_coherent() {
        let intel = intel_igpu();
        let req = request(MemoryTopology::Unified, MemoryClass::Readback);
        let index = pick_index(&intel, !0, &req).expect("intel readback type");
        assert_eq!(index, 1, "the first cached type, not the coherent type 0");
        assert_eq!(
            MappedMemoryKind::of(&intel, index),
            MappedMemoryKind {
                cached: true,
                coherent: false,
            },
            "so `BufferSlot::coherent` is false and the reader must invalidate"
        );

        let apple = apple_m3_max();
        let index = pick_index(&apple, !0, &req).expect("apple readback type");
        assert_eq!(
            MappedMemoryKind::of(&apple, index),
            MappedMemoryKind {
                cached: true,
                coherent: true,
            },
            "a device with both bits still gets both, and pays no invalidate"
        );
    }

    /// A device with no cached type at all must still allocate, and the last
    /// preference is what keeps it coherent rather than leaving the choice to
    /// whatever type happens to sit at index 0.
    #[test]
    fn a_device_with_no_cached_type_falls_back_to_a_coherent_one() {
        use vk::MemoryPropertyFlags as F;
        // Type 0 is host-visible and NOT coherent — a shape no real driver is
        // required to avoid, and the one that would be picked by a bare
        // `HOST_VISIBLE` requirement with no coherence preference behind it.
        let props = build(
            &[(1 << 30, vk::MemoryHeapFlags::DEVICE_LOCAL)],
            &[
                (0, F::HOST_VISIBLE),
                (0, F::HOST_VISIBLE | F::HOST_COHERENT),
            ],
        );
        for topology in [MemoryTopology::Unified, MemoryTopology::Discrete] {
            let req = request(topology, MemoryClass::Readback);
            let index = pick_index(&props, !0, &req).expect("host-visible type exists");
            assert_eq!(
                MappedMemoryKind::of(&props, index),
                MappedMemoryKind {
                    cached: false,
                    coherent: true,
                },
                "{topology:?}: with nothing cached, coherence is the next best thing"
            );
        }
    }

    /// Discrete uploads must NOT prefer the scarce device-local BAR window;
    /// unified uploads should, since it is the same DRAM.
    #[test]
    fn upload_preference_follows_topology() {
        use vk::MemoryPropertyFlags as F;
        assert!(request(MemoryTopology::Unified, MemoryClass::Upload)
            .preferred
            .contains(&F::DEVICE_LOCAL));
        assert!(request(MemoryTopology::Discrete, MemoryClass::Upload)
            .preferred
            .is_empty());
    }

    /// Selection walks the preference list best-first on a unified device: the
    /// device-local cached type wins over the plain cached one.
    #[test]
    fn selection_takes_best_preference_first() {
        let props = apple_m3_max();
        let req = request(MemoryTopology::Unified, MemoryClass::Readback);
        assert_eq!(pick_index(&props, !0, &req), Some(1));
    }

    /// On a discrete device, readback lands in the HOST_CACHED system-RAM type
    /// (index 2), never in the write-combining BAR window (index 3).
    #[test]
    fn discrete_readback_avoids_the_bar_window() {
        let props = nvidia_discrete_rebar();
        let req = request(MemoryTopology::Discrete, MemoryClass::Readback);
        assert_eq!(pick_index(&props, !0, &req), Some(2));
    }

    /// THE load-bearing invariant: a topology misclassification never fails an
    /// allocation. Requesting either topology's flags against either device's
    /// memory layout always resolves to some valid type.
    #[test]
    fn misclassification_degrades_to_a_valid_type_never_a_failure() {
        let devices = [
            ("apple", apple_m3_max()),
            ("intel", intel_igpu()),
            ("amd-apu", amd_apu_host_heap()),
            ("nvidia", nvidia_discrete()),
            ("nvidia-rebar", nvidia_discrete_rebar()),
            ("llvmpipe", llvmpipe()),
        ];
        let classes = [
            MemoryClass::Upload,
            MemoryClass::Readback,
            MemoryClass::DeviceLocal,
            MemoryClass::DeviceLocalPreferred,
        ];
        for (name, props) in &devices {
            for topology in [MemoryTopology::Unified, MemoryTopology::Discrete] {
                for class in classes {
                    let req = request(topology, class);
                    let picked = pick_index(props, !0, &req);
                    assert!(
                        picked.is_some(),
                        "{name} under {topology:?} must resolve {class:?}"
                    );
                    let flags = props.memory_types[picked.unwrap() as usize].property_flags;
                    assert!(
                        flags.contains(req.required),
                        "{name}/{topology:?}/{class:?} must satisfy required flags"
                    );
                }
            }
        }
    }

    /// `type_bits` is honored: a driver that forbids a type for this resource
    /// must never have it selected, even when it is the best preference match.
    #[test]
    fn type_bits_mask_is_respected() {
        let props = apple_m3_max();
        let req = request(MemoryTopology::Unified, MemoryClass::Readback);
        // Mask out type 1 (the only host-visible type) → no candidate at all.
        assert_eq!(pick_index(&props, 0b101, &req), None);
        // Allow it again and it is chosen.
        assert_eq!(pick_index(&props, 0b010, &req), Some(1));
    }

    /// Every memory class resolves a type on every device family the support
    /// matrix names — including the discrete driver with no host-visible
    /// device-local heap and the software rasterizer. A class that resolves
    /// only on the dev host is a row that exists on paper: the allocation fails
    /// on the first machine that lacks the preferred layout, and nothing in the
    /// crate can see that from a single site.
    #[test]
    fn every_class_resolves_on_every_device_family() {
        for (name, props) in [
            ("apple_m3_max", apple_m3_max()),
            ("intel_igpu", intel_igpu()),
            ("amd_apu_host_heap", amd_apu_host_heap()),
            ("nvidia_discrete", nvidia_discrete()),
            ("nvidia_discrete_rebar", nvidia_discrete_rebar()),
            ("llvmpipe", llvmpipe()),
        ] {
            let profile = classify_memory(&props);
            for class in [
                MemoryClass::Upload,
                MemoryClass::Readback,
                MemoryClass::DeviceLocal,
                MemoryClass::DeviceLocalPreferred,
            ] {
                let req = request(profile.topology, class);
                assert!(
                    pick_index(&props, !0, &req).is_some(),
                    "{name}/{class:?} must resolve a memory type"
                );
            }
        }
    }

    /// A guest-sized allocation does not land in a device-local carve-out that
    /// could not hold it while a larger pool with the required flags exists.
    ///
    /// This is the whole-RAMBlock guest-memory import on an APU, and it is the
    /// shape behind "with the import on, the machine crawls unless VRAM is
    /// bigger than the guest". The APU fixture is `Unified` by signal B, so
    /// `MemoryClass::Upload` prefers `DEVICE_LOCAL` — and the only type carrying
    /// it draws from a 2 GiB carve-out. An imported host pointer's pages cannot
    /// move there: the choice does not place the memory, it only tells the
    /// driver which pool to charge and keep resident.
    ///
    /// Asserted at three sizes because the interesting behaviour is the
    /// crossover: under the carve-out the preference is still honoured, over it
    /// the larger pool wins, and past *every* pool it is a refusal naming the
    /// roomiest heap rather than a call the specification forbids.
    #[test]
    fn an_allocation_larger_than_a_heap_does_not_get_charged_to_it() {
        const GIB: u64 = 1 << 30;
        let props = amd_apu_host_heap();
        let req = request(classify_memory(&props).topology, MemoryClass::Upload);

        // Under the 2 GiB carve-out: the device-local preference still wins.
        let small = select_memory_type(&props, !0, &req, GIB, u64::MAX).expect("a type");
        assert_eq!(small.index, 2, "the DEVICE_LOCAL type is still preferred");
        assert_eq!(small.heap_index, 0);

        // Over it, and under the 14 GiB host heap: the preference loses to the
        // pool that can actually hold the allocation.
        let mid = select_memory_type(&props, !0, &req, 4 * GIB, u64::MAX).expect("a type");
        assert_eq!(
            mid.index, 1,
            "the 14 GiB host heap, not the 2 GiB carve-out"
        );
        assert_eq!(mid.heap_index, 1);

        // Larger than every heap — a 16 GiB guest on this part. There is no
        // legal allocation to make, so the answer is the refusal and the
        // roomiest heap it came up against.
        assert_eq!(
            select_memory_type(&props, !0, &req, 16 * GIB, u64::MAX),
            Err(MemoryTypeRefusal::EveryHeapTooSmall {
                bytes: 16 * GIB,
                roomiest_heap: 14 * GIB,
            }),
        );
    }

    /// `maxMemoryAllocationSize` refuses before any heap is consulted, because
    /// it is a device-wide bound no memory type can widen. The reported shape is
    /// an APU whose limit is 4 GiB and whose roomiest heap is far larger, where
    /// the heap check alone would have admitted the call.
    #[test]
    fn a_size_past_the_device_maximum_is_refused_whatever_the_heaps_hold() {
        const GIB: u64 = 1 << 30;
        let props = amd_apu_host_heap();
        let req = request(classify_memory(&props).topology, MemoryClass::Upload);

        assert!(
            select_memory_type(&props, !0, &req, 6 * GIB, u64::MAX).is_ok(),
            "the 14 GiB heap holds 6 GiB, so only the device limit can refuse it",
        );
        assert_eq!(
            select_memory_type(&props, !0, &req, 6 * GIB, 4 * GIB),
            Err(MemoryTypeRefusal::AboveDeviceMaximum {
                bytes: 6 * GIB,
                max: 4 * GIB,
            }),
        );
    }

    /// A size no heap can hold is refused on every device family and for every
    /// class, and the refusal is the capacity one rather than the flags one.
    ///
    /// This is the inverse of what this test asserted when
    /// [`select_memory_type`] nominated the roomiest heap instead: an
    /// `allocationSize` past its heap's `size` is invalid usage, one Mesa driver
    /// returns `VK_SUCCESS` for it and loses the device a second later, and a
    /// bound only the other driver enforces is not a bound. What must still
    /// never happen is a *flags* refusal — a misclassification stays a
    /// performance bug, which is what the variant assertion below pins.
    #[test]
    fn a_size_no_heap_can_hold_is_refused_by_capacity_and_never_by_flags() {
        for (name, props) in [
            ("apple_m3_max", apple_m3_max()),
            ("intel_igpu", intel_igpu()),
            ("amd_apu_host_heap", amd_apu_host_heap()),
            ("nvidia_discrete", nvidia_discrete()),
            ("nvidia_discrete_rebar", nvidia_discrete_rebar()),
            ("llvmpipe", llvmpipe()),
        ] {
            let profile = classify_memory(&props);
            for class in [
                MemoryClass::Upload,
                MemoryClass::Readback,
                MemoryClass::DeviceLocal,
                MemoryClass::DeviceLocalPreferred,
            ] {
                let req = request(profile.topology, class);
                // The roomiest heap any type with the required flags draws
                // from, which is the number the refusal must carry.
                let roomiest = roomiest_heap_for(&props, &req);
                assert_eq!(
                    select_memory_type(&props, !0, &req, u64::MAX, u64::MAX),
                    Err(MemoryTypeRefusal::EveryHeapTooSmall {
                        bytes: u64::MAX,
                        roomiest_heap: roomiest,
                    }),
                    "{name}/{class:?}",
                );
                // And every class still resolves at a size this device can
                // hold, so the capacity rule never stands in for a flags one.
                assert!(
                    select_memory_type(&props, !0, &req, 0, u64::MAX).is_ok(),
                    "{name}/{class:?} must still resolve a type",
                );
            }
        }
    }

    /// The pick names the heap the selected type actually draws from. The index
    /// alone is not a diagnosis on an unfamiliar device — two types with
    /// identical flags can sit on pools three orders of magnitude apart — and
    /// this is the field the `host_ram_import` and `vk_memory_type_pick` lines
    /// carry so a report from a machine nobody here owns is readable.
    #[test]
    fn the_pick_carries_the_heap_it_came_from() {
        let props = nvidia_discrete();
        let req = request(MemoryTopology::Discrete, MemoryClass::DeviceLocal);
        let pick =
            select_memory_type(&props, !0, &req, 1 << 20, u64::MAX).expect("a device-local type");
        let t = props.memory_types[pick.index as usize];
        assert_eq!(pick.heap_index, t.heap_index);
        assert_eq!(
            pick.heap_bytes,
            props.memory_heaps[t.heap_index as usize].size
        );
    }
}
