//! The pieces of the rt platform not built yet ([`Unbuilt`]), and their refusals.
//!
//! A call that reaches an unbuilt piece answers as Metal's call answers where it fails: a nil
//! object, a failed creation, zeros from a getter. Never a death or a panic. Each refusal is
//! counted under its piece's code, `rt-unbuilt-<piece>` ([`Unbuilt::code`]), in the process's
//! degradation counts ([`crate::answer::count_kind`], which the `reims-vgpu: degradation counts:`
//! line gives, docs/conventions.md). The first refusal of each kind in the process writes one line
//! on stderr naming the call, the piece and the code. A boot's counts line therefore lists every
//! unbuilt piece the boot reached, with how many times.
//!
//! [`Unbuilt::ALL`] is the list of what is still unbuilt. A piece leaves it when it is built.

use std::sync::atomic::{AtomicBool, Ordering};

/// A piece of the rt platform not built yet, which the calls that reach it refuse ([`refuse`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unbuilt {
    /// The device: `MTLCreateSystemDefaultDevice` and the device of a registry ID
    /// (`Platform::open_device`, reims-vgpu's `ApiPlatform::metal_device`). Each answers nil,
    /// and every object a device makes is unmade with it.
    Device,
    /// A render pass descriptor's attachments and properties (`RenderPassDescriptor`): its sets
    /// keep nothing, and its reads answer Metal's reading of a nil attachment.
    RenderPass,
    /// Rasterization rate maps, their descriptors and layers: rt has none. A layer's creation
    /// answers nil, and a descriptor keeps no layer.
    RateMap,
}

impl Unbuilt {
    /// Every piece, once.
    pub const ALL: [Unbuilt; 3] = [Unbuilt::Device, Unbuilt::RenderPass, Unbuilt::RateMap];

    /// The piece's code in the degradation counts.
    pub const fn code(self) -> &'static str {
        match self {
            Unbuilt::Device => "rt-unbuilt-device",
            Unbuilt::RenderPass => "rt-unbuilt-render-pass",
            Unbuilt::RateMap => "rt-unbuilt-rate-map",
        }
    }

    /// What the piece is, for the first refusal's line.
    pub const fn what(self) -> &'static str {
        match self {
            Unbuilt::Device => "a GPU device",
            Unbuilt::RenderPass => "a render pass descriptor's attachments",
            Unbuilt::RateMap => "a rasterization rate map",
        }
    }

    /// The piece's place in [`Unbuilt::ALL`].
    fn at(self) -> usize {
        match self {
            Unbuilt::Device => 0,
            Unbuilt::RenderPass => 1,
            Unbuilt::RateMap => 2,
        }
    }
}

/// Whether the process refused a call of each piece yet, in [`Unbuilt::ALL`]'s order.
static SAID: [AtomicBool; Unbuilt::ALL.len()] =
    [const { AtomicBool::new(false) }; Unbuilt::ALL.len()];

/// Refuses the call `sel` for the unbuilt `piece`: counts it under the piece's code, and the
/// process's first refusal of the piece writes its line on stderr. The caller answers as Metal's
/// call answers where it fails.
pub fn refuse(piece: Unbuilt, sel: &'static str) {
    crate::answer::count_kind(piece.code());
    let first = SAID
        .get(piece.at())
        .is_some_and(|said| !said.swap(true, Ordering::Relaxed));
    if first {
        let line = refusal_line(piece, sel);
        // One write, so the line never interleaves with another thread's.
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
    }
}

/// Refuses the creator `sel` for the unbuilt `piece` ([`refuse`]), which answers nil or an
/// `NSError` where Metal makes the object: noted for core on the creator's thread as a gap of the
/// platform's own (`core::platform::device::note_relayed_gap`), so core faults the work that names
/// the nil object where Metal would end the process at it.
pub fn refuse_create(piece: Unbuilt, sel: &'static str) {
    crate::core::platform::device::note_relayed_gap();
    refuse(piece, sel);
}

/// The line of the process's first refusal of `piece`, at `sel`.
fn refusal_line(piece: Unbuilt, sel: &'static str) -> String {
    format!(
        "reims-vgpu: {sel}: {} is not built on metal2vulkan-rt; refused ({}), the first of its \
         kind\n",
        piece.what(),
        piece.code()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::platform::device::relaying_gaps;

    /// The count of `code` in the process's degradation counts.
    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// Each piece is in the list once, at its place, with a code of its own that names it as
    /// unbuilt on rt, and no digit, so the logger's class of it is the whole code.
    #[test]
    fn every_unbuilt_piece_has_a_code_of_its_own() {
        let mut codes: Vec<&str> = Unbuilt::ALL.iter().map(|p| p.code()).collect();
        for (at, piece) in Unbuilt::ALL.iter().enumerate() {
            assert_eq!(piece.at(), at, "{piece:?}");
            let code = piece.code();
            assert!(code.starts_with("rt-unbuilt-"), "{code}");
            assert!(!code.chars().any(|c| c.is_ascii_digit()), "{code}");
        }
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), Unbuilt::ALL.len());
    }

    /// A refusal is counted under its piece's code in the degradation counts, every time; a
    /// creator's refusal is noted for core as a gap of the platform's.
    #[test]
    fn a_refusal_is_counted_under_its_piece_s_code() {
        let code = Unbuilt::RateMap.code();
        let before = counted(code);
        refuse(Unbuilt::RateMap, "test");
        refuse(Unbuilt::RateMap, "test");
        assert!(counted(code) >= before + 2, "{code}");
        let ((), relayed) = relaying_gaps(|| refuse_create(Unbuilt::RateMap, "test"));
        assert!(relayed);
        let ((), relayed) = relaying_gaps(|| refuse(Unbuilt::RateMap, "test"));
        assert!(!relayed);
    }

    /// The first refusal's line names the call, the piece and its code.
    #[test]
    fn the_first_refusal_s_line_names_the_call_and_the_code() {
        assert_eq!(
            refusal_line(Unbuilt::Device, "MTLCreateSystemDefaultDevice"),
            "reims-vgpu: MTLCreateSystemDefaultDevice: a GPU device is not built on \
             metal2vulkan-rt; refused (rt-unbuilt-device), the first of its kind\n"
        );
    }
}
