//! What the device makes on a platform that places a pinned buffer or heap only inside a block it
//! knows (`ApiPlatform::PINS_IN_BLOCKS`, metal2vulkan-runtime's), counted by the kind of make, so a
//! boot's counts line says which makes still lie elsewhere than the guest's pointers say, and
//! which import the host's memory.
//!
//! **Pins not kept.** On such a platform any other pinned object lies at the platform's own
//! address, which the guest's pointers to it do not hold. The platform counts each such object
//! once (metal2vulkan-runtime's `rt-gpu-address-unpinned`); the decoder, which knows what each
//! create makes, counts it again under its kind's code ([`Codes::unpinned`]), so the kinds' counts
//! sum to that total. A pin counts where the decoder passed it to the platform and the object made
//! answers another `gpuAddress` ([`made`]). A texture placed by the device at the guest's own pick
//! is counted apart, as [`TEXTURE_PICKS`]: no platform takes a texture's pin, so none is in the
//! total, and its `gpuAddress` is the guest's answer alone.
//!
//! **Objects made.** Each buffer or heap a guest's create makes there is counted by its kind
//! ([`Codes::made`]), with the most of the kind live at once: the decoder's entry of the object
//! holds a [`Live`], which counts it gone as the entry goes.
//!
//! **Imports.** Each make the decoder or the device asks of the platform runs as a make of its
//! kind on its thread (`reims_vgpu_vulkan::answer::making_as`, [`Codes::imports`]):
//! metal2vulkan-runtime counts each import of the host's memory it makes there under that kind,
//! with the most live at once, and under its total (`rt-imports`), so the kinds' counts sum to it.
//! Each refusal of rt's there is counted under the kind too
//! (`reims_vgpu_vulkan::answer::refusing_as`, [`Codes::refused`]), beside its own code
//! (`rt-refused-vulkan` and the others). The device's own makes that import are kinds of their own:
//! the views of pages blocks ([`Kind::PagesView`]), the views a device cache's copies use
//! ([`Kind::CacheView`]) and the display's scanout ([`Kind::Scanout`]).
//!
//! On a platform that places a pinned object anywhere (`ApiPlatform::PINS_IN_BLOCKS` false, the
//! default, the fake platform's) no pin or object is counted here, and none of its makes imports through a counted path.

use std::sync::atomic::{AtomicU32, Ordering};

use reims_vgpu_vulkan::answer::{MadeCodes, count_gone, count_kind, count_made};

/// The kind of a make, which names its counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A heap over a block of the guest's Shared pool (`HEAP_OVER_POOL`), whose slab buffers lie
    /// at its address plus their offsets.
    Pool,
    /// A Private buffer (`BUFFER_CREATE`) or a heap not Shared (`HEAP_CREATE`): in the context's
    /// Private blocks where the guest took `PRIVATE_BLOCKS`.
    Private,
    /// A `Managed` buffer (`MANAGED_BUFFER_CREATE`, `BUFFER_MANAGED_AT_OFFSET`, a `Managed`
    /// `BUFFER_OVER_HOST_MEMORY`) on a GPU of its own memory, whose GPU copy is at the pin.
    Managed,
    /// A `Managed` buffer on a GPU that shares the CPU's memory, which keeps one copy: its
    /// `gpuAddress` is its bytes', never a Private block's.
    ManagedUnified,
    /// A no-copy buffer over a pages block at the address its view was given
    /// (`BUFFER_OVER_PAGES`, `IMPORT_ADDRESSES`).
    Import,
    /// A heap over a pages block (`HEAP_OVER_PAGES`), or a no-copy buffer over one at the
    /// guest's own pick (`BUFFER_OVER_PAGES`): the guest's page slabs, and an app's no-copy
    /// buffer where its view was given no address.
    Pages,
    /// A Shared heap the device places (`HEAP_CREATE`).
    Heap,
    /// A buffer or heap over a block of the device's region (`BUFFER_OVER_HOST_MEMORY`,
    /// `HEAP_OVER_HOST_MEMORY`).
    Region,
    /// A buffer the device places that is not Private (`BUFFER_CREATE`).
    Buffer,
    /// The view of a pages block's pages and the buffer over the whole of it (an app's no-copy
    /// memory, an IOSurface's pages), and a buffer over an IOSurface's view that no device cache
    /// stands for (`BUFFER_OVER_SURFACE_PAGES`).
    PagesView,
    /// The buffer over an IOSurface's view that a device cache's copies use as their other end.
    CacheView,
    /// The display's scanout, a no-copy buffer over its pages in the region.
    Scanout,
}

/// Every kind, in the order of their codes.
#[cfg(test)]
pub(crate) const KINDS: [Kind; 12] = [
    Kind::Pool,
    Kind::Private,
    Kind::Managed,
    Kind::ManagedUnified,
    Kind::Import,
    Kind::Pages,
    Kind::Heap,
    Kind::Region,
    Kind::Buffer,
    Kind::PagesView,
    Kind::CacheView,
    Kind::Scanout,
];

/// A kind's codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Codes {
    /// A pin of the kind's the platform did not keep: metal2vulkan-runtime's total's code,
    /// `rt-gpu-address-unpinned`, and the kind.
    pub(crate) unpinned: &'static str,
    /// The kind's objects made, `rt-made-<kind>`, and the most live at once.
    pub(crate) made: MadeCodes,
    /// The platform's imports of the host's memory for the kind's makes, `rt-imports-<kind>`, and
    /// the most live at once.
    pub(crate) imports: MadeCodes,
    /// rt's refusals of the kind's makes, `rt-refused-making-<kind>`.
    pub(crate) refused: &'static str,
}

/// The codes of the kind named `$kind`.
macro_rules! codes {
    ($kind:literal) => {
        Codes {
            unpinned: concat!("rt-gpu-address-unpinned-", $kind),
            made: MadeCodes {
                made: concat!("rt-made-", $kind),
                live_most: concat!("rt-made-", $kind, "-live-most"),
            },
            imports: MadeCodes {
                made: concat!("rt-imports-", $kind),
                live_most: concat!("rt-imports-", $kind, "-live-most"),
            },
            refused: concat!("rt-refused-making-", $kind),
        }
    };
}

impl Kind {
    /// Its codes.
    pub(crate) const fn codes(self) -> Codes {
        match self {
            Kind::Pool => codes!("pool"),
            Kind::Private => codes!("private"),
            Kind::Managed => codes!("managed"),
            Kind::ManagedUnified => codes!("managed-unified"),
            Kind::Import => codes!("import"),
            Kind::Pages => codes!("pages"),
            Kind::Heap => codes!("heap"),
            Kind::Region => codes!("region"),
            Kind::Buffer => codes!("buffer"),
            Kind::PagesView => codes!("pages-view"),
            Kind::CacheView => codes!("cache-view"),
            Kind::Scanout => codes!("scanout"),
        }
    }

    /// A buffer of resource `options` the device places (`BUFFER_CREATE`): Private where its
    /// storage mode is.
    pub(crate) fn of_buffer(options: u64) -> Kind {
        match (options >> 4) & 0xf {
            STORAGE_PRIVATE => Kind::Private,
            _ => Kind::Buffer,
        }
    }

    /// A heap of resource `options` the device places (`HEAP_CREATE`): Shared, or Private (any
    /// other storage mode, which the platform refuses or places as Private).
    pub(crate) fn of_heap(options: u64) -> Kind {
        match (options >> 4) & 0xf {
            STORAGE_SHARED => Kind::Heap,
            _ => Kind::Private,
        }
    }

    /// A `Managed` buffer, on a GPU that shares the CPU's memory where `unified`.
    pub(crate) fn managed(unified: bool) -> Kind {
        match unified {
            true => Kind::ManagedUnified,
            false => Kind::Managed,
        }
    }

    /// Runs `make` as a make of the kind (`reims_vgpu_vulkan::answer::making_as`,
    /// `refusing_as`): the platform's imports and refusals for it are counted under the kind's
    /// codes.
    pub(crate) fn making<T>(self, make: impl FnOnce() -> T) -> T {
        let codes = self.codes();
        let refusing = || reims_vgpu_vulkan::answer::refusing_as(codes.refused, make);
        reims_vgpu_vulkan::answer::making_as(codes.imports, refusing)
    }
}

/// `MTLStorageModeShared`, in the resource options' bits 4 to 7.
const STORAGE_SHARED: u64 = 0;

/// `MTLStorageModePrivate`, in the resource options' bits 4 to 7.
const STORAGE_PRIVATE: u64 = 2;

/// The code of a texture the device placed at the guest's own pick of a GPU address
/// (`TEXTURE_CREATE` placed by the device, `MANAGED_TEXTURE_CREATE`), counted per texture: no
/// platform takes a texture's pin, so its `gpuAddress` is the guest's answer alone, which on
/// metal2vulkan-runtime may equal a live buffer's address there.
pub(crate) const TEXTURE_PICKS: &str = "rt-texture-address-guest-pick";

/// An object made of a kind, live while this lives ([`made`]): its decoder entry holds it, and
/// its drop counts the object gone.
#[derive(Debug)]
pub(crate) struct Live(MadeCodes);

impl Drop for Live {
    fn drop(&mut self) {
        count_gone(self.0);
    }
}

/// Counts an object of `kind` made at `made`, which the guest pinned at `pin` where the decoder
/// passed it to the platform, where `on` (the platform places pinned objects only in its known
/// blocks): one more made and live, and a pin the platform did not keep. The object's [`Live`]
/// where `on`, which its entry holds.
pub(crate) fn made(on: bool, kind: Kind, pin: Option<u64>, made: u64) -> Option<Live> {
    if !on {
        return None;
    }
    let codes = kind.codes();
    if let Some(pin) = pin.filter(|&pin| pin != made) {
        count_kind(codes.unpinned);
        unkept_said(kind, pin, made);
    }
    count_made(codes.made);
    Some(Live(codes.made))
}

/// The kinds whose first pin not kept was said ([`unkept_said`]), a bit each.
static UNKEPT_SAID: AtomicU32 = AtomicU32::new(0);

/// Says the process's first pin of `kind` the platform did not keep: where the guest pinned it
/// and where it lies.
fn unkept_said(kind: Kind, pin: u64, made: u64) {
    let bit = 1u32 << (kind as u32);
    if UNKEPT_SAID.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        let code = kind.codes().unpinned;
        log::warn!(
            target: "reims_vgpu::graphics",
            "an object the guest pinned at {pin:#x} lies at {made:#x}, the platform's address: \
             the first pin of its kind not kept ({code}), counted from here"
        );
    }
}

/// Counts a texture the device placed at the guest's pick `gpu_address` ([`TEXTURE_PICKS`]),
/// where `on` and the guest picked one.
pub(crate) fn count_texture(on: bool, gpu_address: u64) {
    if on && gpu_address != 0 {
        count_kind(TEXTURE_PICKS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each kind's codes are the platform's totals' with the kind, each its own; a buffer is
    /// Private by its storage mode, a heap Shared or Private, a `Managed` buffer by whether the
    /// GPU shares the CPU's memory.
    #[test]
    fn each_kind_counts_under_codes_of_its_own() {
        let mut codes: Vec<&str> = (KINDS.iter())
            .flat_map(|k| {
                let c = k.codes();
                [
                    c.unpinned,
                    c.made.made,
                    c.made.live_most,
                    c.imports.made,
                    c.imports.live_most,
                    c.refused,
                ]
            })
            .collect();
        for (kind, c) in KINDS.iter().map(|k| (k, k.codes())) {
            assert!(
                c.unpinned.starts_with("rt-gpu-address-unpinned-"),
                "{kind:?}"
            );
            assert!(c.made.made.starts_with("rt-made-"), "{kind:?}");
            assert!(c.imports.made.starts_with("rt-imports-"), "{kind:?}");
            assert!(c.refused.starts_with("rt-refused-making-"), "{kind:?}");
            assert_eq!(c.imports.live_most, format!("{}-live-most", c.imports.made));
            assert_eq!(c.made.live_most, format!("{}-live-most", c.made.made));
        }
        let all = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), all);
        assert!(!codes.contains(&"rt-imports-live-most"));
        assert_eq!(Kind::of_buffer(0x20), Kind::Private);
        assert_eq!(Kind::of_buffer(0x0), Kind::Buffer);
        assert_eq!(Kind::of_heap(0x0), Kind::Heap);
        assert_eq!(Kind::of_heap(0x20), Kind::Private);
        assert_eq!(Kind::managed(true), Kind::ManagedUnified);
        assert_eq!(Kind::managed(false), Kind::Managed);
    }
}
