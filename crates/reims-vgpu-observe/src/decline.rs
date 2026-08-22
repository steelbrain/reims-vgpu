//! The crate-wide decline vocabulary: two traits.
//!
//! # What a decline is
//!
//! A **decline** is any path that rejects, drops, degrades or mis-executes a
//! decoded guest command. `AGENTS.md` requires each one to name a *specific*
//! reason — not a coarse status a dozen distinct checks collapse into. The
//! canonical failure this prevents is already in the ground rules: a 16 MiB cap
//! returning a bare `Unsupported` alongside six other checks, invisible for a
//! day because the log could not tell them apart.
//!
//! # Where the vocabulary lives
//!
//! In the `slug()` arms, and nowhere else. This module used to carry a 2 700-line
//! `#[cfg(test)]` `REGISTRY` restating every type's file, emission site and full
//! slug list, plus a source scanner in a `super::gate` module whose whole job was
//! checking that the restatement still matched the code. The table could only ever
//! agree or disagree with the arms it copied, so it added no invariant the arms do
//! not already carry — and it charged every deletion a second edit plus a
//! hand-bumped `(types, slugs)` baseline, which is what made shrinking this crate
//! expensive. Both went in `db80389`.
//!
//! One property is genuinely crate-wide and not visible from any single impl:
//! **no two checks share a slug**. A source scan over the `Decline`/`Refusal`
//! impls used to check it and is gone, so it is now the author's obligation —
//! prefix a slug with the rail that owns it and a collision stops being likely.
//!
//! # Adding a decline type
//!
//! 1. Implement [`Decline`] on it — one slug per distinct check, never shared.
//! 2. Emit it through [`super::emit::Emit`], which cannot render a line without
//!    a slug.

/// A typed refusal that can name itself in the always-on log.
///
/// Modelled on `TranslateReason`, which had the right shape before this trait
/// existed: the payload rides along with the variant, and rendering produces
/// `reason=<slug>` plus the values that caused it.
pub trait Decline {
    /// Stable snake_case slug for `reason=` in `/tmp/reims-vgpu-fail.log`.
    ///
    /// **One slug per distinct check.** Two checks sharing a slug is the exact
    /// defect this vocabulary exists to prevent: you grep the log, watch the
    /// slug fire, and still cannot tell which check refused.
    fn slug(&self) -> &'static str;

    /// The load-bearing values behind this decline — refs, dims, formats,
    /// offsets, sizes, caps. Rendered after `reason=` as `k=v` pairs.
    ///
    /// A decline that names only its class leaves the reader without the value
    /// that caused it, which is half a diagnostic. Allocation here is fine:
    /// declines are rare by construction, and a flood is a bug the sink's own
    /// detector will report.
    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// Give a [`Decline`] the `Display` every always-on line renders it through:
/// `reason=<slug>` followed by [`Decline::fields`] as ` k=v` pairs.
///
/// Rust cannot express this as `impl<T: Decline> Display for T` — the blanket
/// impl would claim `Display` for every foreign type — so each decline type
/// needs its own impl, and fourteen of them were the same nine lines. Spelling
/// it once means a change to the wire shape of a decline line lands in one
/// place instead of fourteen, which is the same argument the `Decline` trait
/// itself makes about slugs.
///
/// The `use` is inside the function body so the macro works whether or not the
/// trait is already in scope where it is invoked.
#[macro_export]
macro_rules! decline_display {
    ($ty:ty) => {
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use $crate::Decline as _;
                write!(f, "reason={}", self.slug())?;
                for (key, value) in self.fields() {
                    write!(f, " {key}={value}")?;
                }
                Ok(())
            }
        }
    };
}
pub use crate::decline_display;

/// A status enum that mixes success with refusal.
///
/// [`Decline`] is for a value that is *always* a refusal — a `DrawError`, a
/// `TranslateReason`. Most of this crate's older vocabulary is not shaped that
/// way: `DecodeStatus`, `BlitStatus`, `IcbStatus` and their siblings carry `Ok`
/// and `Done` alongside a dozen genuine refusals, and the difference matters
/// because I2's carve-out lives exactly there. A resolver answering "not ready
/// yet" every poll must **not** reach the log; a malformed length must.
///
/// That judgement is one no gate can make, so this trait makes it once, in an
/// exhaustive `match` the compiler forces you to revisit when a variant
/// appears. A new status variant cannot compile until its author has said which
/// side of the line it falls on.
pub trait Refusal {
    /// The registered slug when this value refused; `None` when it is control
    /// flow — success, "done", "not ready yet" — that must stay out of the log.
    fn refusal(&self) -> Option<&'static str>;

    /// The load-bearing values behind the refusal, as for [`Decline::fields`].
    /// Returning nothing is normal here: a status usually carries no payload and
    /// the failing site adds the refs and sizes with [`super::Emit::field`].
    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}
