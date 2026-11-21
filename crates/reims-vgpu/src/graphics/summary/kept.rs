//! What the host holds for the guest now, in the host's summary of each ten seconds (`super`), so
//! a boot's reader can tell the platform's textures and buffers a live guest id names from those
//! the host holds for none, by their holder, as the guest makes and releases its objects. One
//! part, `kept: NAME VALUE, ...` ([`line()`]), each a count as it is now:
//!
//! - **The platform's**: its live textures (each image and each view) and buffers, as the Vulkan
//!   platform's runtime counts them (`rt_textures`, `rt_buffers`;
//!   `reims_vgpu_vulkan::rt_platform::summary::live_textures_and_buffers`), left out on another
//!   platform.
//! - **The guest's**: the decoders' tables' textures and buffers ([`TableCounts`]): live, those
//!   removed and kept for an open view or a command buffer posted before the removal (`_gone`),
//!   and of the live ones the textures that are views of a device cache's shape's image
//!   (`shape_views`) or that the display made a view of its own of (`shown_views`), and the
//!   buffers that are a device cache's buffer itself (`over_cache`) or lie over a block's
//!   surface view (`over_view`); the heaps, and the buffers of `Managed` objects' CPU copies;
//!   and how many contexts were read, and those whose table was not had within [`BOUND`]
//!   (`contexts_unread`), whose objects are left out.
//! - **The IOSurfaces' device caches'** ([`CachesLive`]): the caches, the bytes the budget counts
//!   them holding, their buffers' bytes, those made for the host's video, their other buffers
//!   (over their pages' views, and wide homes); their shapes' images and bytes, those a live
//!   guest texture references and the references, those none does and their bytes, and of those
//!   the ones past the bound, which only the cache's next new shape sweeps
//!   ([`ShapesLive`]); the caches none of whose shapes a live texture references, with
//!   the bytes they hold, and those holding shapes past the bound.
//! - **The caches holding the most**, one by one, in a part of their own, `kept caches: block
//!   BLOCK held BYTES, ...; ...` ([`largest_line`]): the six caches holding the most bytes,
//!   the most first, each with its block, the bytes it holds, its buffer's, whether it was made
//!   for the host's video, and its shapes' counts as above.
//! - **The pages' views**: the surfaces' blocks' views, the pages blocks' and the idle ones the
//!   views' cache keeps (`decode::surface_pages`).
//! - **What no id names**: the platform's textures less the guest's (live and gone),
//!   `host_textures`; and of those, what neither a shape's image nor a display's view explains,
//!   `host_textures_other` (signed, as the counts are read one after another while the guest's
//!   work runs).
//!
//! Each decoder made while the summary is on is watched ([`watch`]), weakly, and read at each
//! period while it lives. A diagnostic: reading takes each table's lock for reading, the map of
//! contexts', the caches' and the views' locks, one after another, every ten seconds, and none
//! while the summary is off; which lines the host writes is free under the contract.

use std::sync::{Arc, Weak};
use std::time::Duration;

use reims_vgpu_vulkan::core::sync::Mutex;

use crate::graphics::decode::{CacheLive, CachesLive, ShapesLive, TableCounts};

/// How long a read waits for each lock it tries (the map of contexts' and each table's).
pub(crate) const BOUND: Duration = Duration::from_millis(20);

/// What a decoder holds now ([`Kept`]).
pub(crate) trait Source: Send + Sync {
    /// What it holds now.
    fn kept(&self) -> Kept;
}

/// What the decoders hold now (the module doc).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Kept {
    /// The contexts whose tables were read, and those not had in time (a busy map of contexts
    /// counts one).
    pub(crate) contexts: u64,
    pub(crate) unread: u64,
    pub(crate) tables: TableCounts,
    pub(crate) caches: CachesLive,
    /// The surfaces' blocks' views, the pages blocks' views and the idle views kept.
    pub(crate) views: (u64, u64, u64),
}

impl Kept {
    /// Adds `other`'s counts to these.
    fn add(&mut self, other: &Kept) {
        self.contexts += other.contexts;
        self.unread += other.unread;
        self.tables.sum(&other.tables);
        let (a, b) = (&mut self.caches, &other.caches);
        a.caches += b.caches;
        a.cache_bytes += b.cache_bytes;
        a.buffer_bytes += b.buffer_bytes;
        a.video += b.video;
        a.view_buffers += b.view_buffers;
        a.wide += b.wide;
        a.shapes.add(&b.shapes);
        a.unreferenced += b.unreferenced;
        a.unreferenced_bytes += b.unreferenced_bytes;
        a.past_bound += b.past_bound;
        for one in b.largest.iter().filter(|one| one.held != 0) {
            a.offer(*one);
        }
        self.views.0 += other.views.0;
        self.views.1 += other.views.1;
        self.views.2 += other.views.2;
    }
}

/// The decoders watched.
static SOURCES: Mutex<Vec<Weak<dyn Source>>> = Mutex::new(Vec::new());

/// Watches `source`, a decoder's, where the summary is on.
pub(crate) fn watch(source: Weak<dyn Source>) {
    if super::on() {
        SOURCES.lock().push(source);
    }
}

/// The parts of what the decoders watched hold now, with `rt`, the platform's live textures and
/// buffers where it counts them: the counts ([`line()`]) and the caches holding the most
/// ([`largest_line`]); `None` each where no decoder lives, and the second where no cache does.
pub(crate) fn take(rt: Option<(u64, u64)>) -> [Option<String>; 2] {
    let sources: Vec<Arc<dyn Source>> = {
        let mut sources = SOURCES.lock();
        sources.retain(|source| source.strong_count() > 0);
        sources.iter().filter_map(Weak::upgrade).collect()
    };
    if sources.is_empty() {
        return [None, None];
    }
    let mut kept = Kept::default();
    for source in &sources {
        kept.add(&source.kept());
    }
    [Some(line(&kept, rt)), largest_line(&kept)]
}

/// The part of the caches holding the most in `kept` (the module doc); `None` where no cache
/// holds a byte.
pub(crate) fn largest_line(kept: &Kept) -> Option<String> {
    let caches: Vec<String> = (kept.caches.largest.iter())
        .filter(|one| one.held != 0)
        .map(|one: &CacheLive| {
            let s = &one.shapes;
            format!(
                "block {:#x} held {}, buffer_bytes {}, video {}, shapes {}, shape_bytes {}, \
                 shapes_referenced {}, shape_references {}, shapes_unreferenced {}, \
                 shapes_unreferenced_bytes {}, shapes_past_bound {}, shapes_past_bound_bytes {}, \
                 plane_shapes {}",
                one.block,
                one.held,
                one.buffer_bytes,
                u8::from(one.video),
                s.shapes,
                s.bytes,
                s.referenced,
                s.references,
                s.unreferenced,
                s.unreferenced_bytes,
                s.past_bound,
                s.past_bound_bytes,
                s.planes,
            )
        })
        .collect();
    (!caches.is_empty()).then(|| format!("kept caches: {}", caches.join("; ")))
}

/// The part of `kept`, with `rt` (the module doc).
pub(crate) fn line(kept: &Kept, rt: Option<(u64, u64)>) -> String {
    let t = &kept.tables;
    let c = &kept.caches;
    let s: &ShapesLive = &c.shapes;
    let mut pairs: Vec<(&str, i128)> = Vec::new();
    if let Some((textures, buffers)) = rt {
        let host = i128::from(textures) - i128::from(t.textures) - i128::from(t.textures_gone);
        let other = host - i128::from(s.shapes) - i128::from(t.shown_views);
        pairs.extend([
            ("rt_textures", i128::from(textures)),
            ("rt_buffers", i128::from(buffers)),
            ("host_textures", host),
            ("host_textures_other", other),
        ]);
    }
    let counts = [
        ("guest_textures", t.textures),
        ("guest_textures_gone", t.textures_gone),
        ("shape_views", t.shape_views),
        ("shown_views", t.shown_views),
        ("guest_buffers", t.buffers),
        ("guest_buffers_gone", t.buffers_gone),
        ("over_cache", t.over_cache),
        ("over_view", t.over_view),
        ("heaps", t.heaps),
        ("managed_copies", t.managed_copies),
        ("contexts", kept.contexts),
        ("contexts_unread", kept.unread),
        ("caches", c.caches),
        ("cache_bytes", c.cache_bytes),
        ("cache_buffer_bytes", c.buffer_bytes),
        ("video_caches", c.video),
        ("cache_view_buffers", c.view_buffers),
        ("wide_homes", c.wide),
        ("shapes", s.shapes),
        ("shape_bytes", s.bytes),
        ("shapes_referenced", s.referenced),
        ("shape_references", s.references),
        ("shapes_unreferenced", s.unreferenced),
        ("shapes_unreferenced_bytes", s.unreferenced_bytes),
        ("shapes_past_bound", s.past_bound),
        ("shapes_past_bound_bytes", s.past_bound_bytes),
        ("plane_shapes", s.planes),
        ("caches_unreferenced", c.unreferenced),
        ("caches_unreferenced_bytes", c.unreferenced_bytes),
        ("caches_past_bound", c.past_bound),
        ("surface_views", kept.views.0),
        ("pages_views", kept.views.1),
        ("idle_views", kept.views.2),
    ];
    pairs.extend(counts.iter().map(|&(name, n)| (name, i128::from(n))));
    let named: Vec<String> = (pairs.iter())
        .map(|(name, value)| format!("{name} {value}"))
        .collect();
    format!("kept: {}", named.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphics::decode::surface_cache_largest as LARGEST;

    /// The part names each count, and tells the platform's textures no guest id names, and what
    /// neither a shape's image nor a display's view explains of them; with no platform's counts
    /// it leaves those out.
    #[test]
    fn the_kept_part_tells_the_textures_no_guest_id_names() {
        let mut kept = Kept::default();
        kept.tables.textures = 391;
        kept.tables.textures_gone = 2;
        kept.tables.shown_views = 3;
        kept.caches.caches = 40;
        kept.caches.shapes.shapes = 1100;
        kept.caches.shapes.past_bound = 7;
        kept.views = (0, 5, 1);
        let line = line(&kept, Some((1513, 675)));
        assert!(line.starts_with(
            "kept: rt_textures 1513, rt_buffers 675, host_textures 1120, \
             host_textures_other 17, guest_textures 391, guest_textures_gone 2, "
        ));
        assert!(line.contains(", shapes 1100, "));
        assert!(line.contains(", shapes_past_bound 7, "));
        assert!(line.ends_with(", surface_views 0, pages_views 5, idle_views 1"));
        let fewer = super::line(&Kept::default(), Some((1, 0)));
        assert!(fewer.contains("host_textures 1, host_textures_other 1,"));
        let off = super::line(&kept, None);
        assert!(off.starts_with("kept: guest_textures 391, "));
    }

    /// The caches holding the most are kept the most first, at most [`LARGEST`] of them, across
    /// the decoders too, each named by its block; with none holding a byte the part is left out.
    #[test]
    fn the_largest_caches_part_names_each_cache_the_most_first() {
        let one = |block: u64, held: u64| {
            let mut one = CacheLive {
                block,
                held,
                ..CacheLive::default()
            };
            one.shapes.shapes = block;
            one
        };
        let mut kept = Kept::default();
        assert_eq!(largest_line(&kept), None);
        for (block, held) in [(1, 10), (2, 30), (3, 20)] {
            kept.caches.offer(one(block, held));
        }
        let mut other = Kept::default();
        for block in 4..(4 + LARGEST as u64) {
            other.caches.offer(one(block, 5));
        }
        other.caches.offer(one(20, 40));
        kept.add(&other);
        let held: Vec<(u64, u64)> = (kept.caches.largest.iter())
            .map(|one| (one.block, one.held))
            .collect();
        assert_eq!(&held[..4], &[(20, 40), (2, 30), (3, 20), (1, 10)]);
        assert_eq!(held.len(), LARGEST);
        assert!(held[4..].iter().all(|&(_, held)| held == 5));
        let line = largest_line(&kept).unwrap();
        assert!(line.starts_with(
            "kept caches: block 0x14 held 40, buffer_bytes 0, video 0, shapes 20, shape_bytes 0, "
        ));
        assert!(line.contains("; block 0x2 held 30, "));
        assert_eq!(line.matches("; block ").count(), LARGEST - 1);
    }

    /// The decoders' counts are summed; a decoder gone is no longer read, and with none the part
    /// is left out.
    #[test]
    fn the_kept_part_sums_the_decoders_alive() {
        struct Fixed(Kept);
        impl Source for Fixed {
            fn kept(&self) -> Kept {
                self.0
            }
        }
        let mut one = Kept {
            contexts: 2,
            ..Kept::default()
        };
        one.tables.textures = 10;
        one.caches.shapes.shapes = 4;
        one.views = (1, 2, 3);
        let mut sum = Kept::default();
        sum.add(&one);
        sum.add(&one);
        assert_eq!((sum.contexts, sum.tables.textures), (4, 20));
        assert_eq!((sum.caches.shapes.shapes, sum.views), (8, (2, 4, 6)));
        let gone: Weak<dyn Source> = {
            let source: Arc<dyn Source> = Arc::new(Fixed(one));
            Arc::downgrade(&source)
        };
        assert!(gone.upgrade().is_none());
    }
}
