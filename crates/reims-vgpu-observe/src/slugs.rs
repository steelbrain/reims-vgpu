//! Which type claimed each slug, and what happens when two of them claim one.
//!
//! # The property no single impl can see
//!
//! [`crate::decline`] states it and cannot enforce it: **no two checks share a
//! slug**. The failure is specific rather than aesthetic.
//! [`crate::emit::Emit::fail_once`] latches on `(slug, discriminant)` in one
//! process-global set, so two declines spelling one slug share a latch —
//! whichever fires first for a given discriminant silences the other for the
//! rest of the boot, and the log looks healthy while it happens. That is not
//! hypothetical: `mapping_gpa_span` had exactly this shape between its two
//! emitters, and the silence it produced had already been written up as a
//! finding about the device before anyone noticed the collision.
//!
//! # Why this is a runtime claim and not a scan
//!
//! The check used to be a source scan over every `slug()` body, next to a
//! 2 700-line restatement of the vocabulary. `AGENTS.md` forbids validating this
//! crate by reading its own source as text, and the restatement is what made
//! deleting a decline expensive, so both are gone and are not coming back.
//!
//! What is left is the observation itself. A slug reaches the log through
//! [`crate::emit::Emit`], and there the concrete decline type is known, so a line can
//! *claim* its slug on behalf of its type. A second type claiming a slug some
//! other type already holds is a collision, reported by name on the always-on
//! channel and — in a unit-test build, where one process runs the whole suite
//! and thousands of declines render — raised as a panic that names both types.
//!
//! This proves a collision when both sides emit; it cannot prove their absence.
//! That is a weaker claim than the scan made and a true one, which the scan's
//! restatement was not.
//!
//! # It selects nothing
//!
//! A claim answers no question a product path asks. It records a name, and on a
//! conflict it writes a line. No caller can branch on it, and a collision
//! changes neither the line that provoked it nor any decision behind it.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// The type currently holding each slug, by [`std::any::type_name`].
///
/// Deliberately never cleared, including by `forget_all_latches`: a claim is a
/// name, not a latch, and clearing it between unit tests would drop exactly the
/// cross-test accumulation that gives a one-process suite its reach.
static OWNERS: OnceLock<RwLock<HashMap<&'static str, &'static str>>> = OnceLock::new();

fn owners() -> &'static RwLock<HashMap<&'static str, &'static str>> {
    OWNERS.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Record that `owner` spells `slug`, and report it if another type already
/// does.
///
/// The read lock is the whole hot path: after the first line for a slug, every
/// later one finds its own owner already recorded and returns without taking
/// the write lock. That sits behind an allocation and a file write in every
/// caller, so it is not a cost this can be measured against.
pub(super) fn claim(slug: &'static str, owner: &'static str) {
    if let Ok(map) = owners().read() {
        match map.get(slug) {
            Some(held) if *held == owner => return,
            Some(held) => {
                let held = *held;
                drop(map);
                report(slug, held, owner);
                return;
            }
            None => {}
        }
    }
    let Ok(mut map) = owners().write() else {
        return;
    };
    let held = match map.entry(slug) {
        Entry::Vacant(slot) => {
            slot.insert(owner);
            return;
        }
        // The claim stays with whoever registered first, so the pair is
        // reported the same way round every time rather than flip-flopping.
        Entry::Occupied(slot) if *slot.get() == owner => return,
        Entry::Occupied(slot) => *slot.get(),
    };
    drop(map);
    report(slug, held, owner);
}

/// Name a collision once per claimant, on the always-on channel.
///
/// Latched through [`crate::emit::first_sight`] because the colliding pair emits as
/// often as the guest provokes it, and the second report says nothing the first
/// did not.
fn report(slug: &'static str, held: &'static str, claimant: &'static str) {
    if super::first_sight("observe_slug_collision", mix(slug, claimant)) {
        super::fail(format!(
            "observe_slug reason=observe_slug_collision slug={slug} holder={} claimant={}",
            compact(held),
            compact(claimant),
        ));
    }
    // A test build runs a whole suite in one process, so this is where a
    // collision is cheapest to find and loudest to ignore. `testing` as well as
    // `test`: the declines that collide belong to the crates above this one,
    // whose tests are a different compilation, and gating on `test` alone would
    // point the check at the handful of declines defined here. Production keeps
    // the line and no panic: an observability defect must not take the device
    // down.
    #[cfg(any(test, feature = "testing"))]
    panic!("two types spell the slug {slug:?}: {held} and {claimant}");
}

/// `type_name` renders `Foo<A, B>` with a space, and the log is parsed by
/// splitting on spaces. Dropping the space keeps the value one field.
fn compact(name: &'static str) -> String {
    name.chars().filter(|c| !c.is_whitespace()).collect()
}

/// A stable discriminant for one `(slug, claimant)` pair.
///
/// FNV-1a over both names rather than a `DefaultHasher`, which is seeded per
/// process: the latch only has to separate pairs within one boot, but a
/// reproducible value makes two boots' logs comparable.
fn mix(slug: &str, claimant: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in slug
        .bytes()
        .chain(b"/".iter().copied())
        .chain(claimant.bytes())
    {
        h ^= u64::from(byte);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slug reclaimed by its own owner is not a collision, however many lines
    /// it renders. This is the path every healthy emit takes.
    #[test]
    fn one_owner_may_claim_its_slug_repeatedly() {
        claim("slug_test_solo", "reims_vgpu::observe::slugs::tests::A");
        claim("slug_test_solo", "reims_vgpu::observe::slugs::tests::A");
        claim("slug_test_solo", "reims_vgpu::observe::slugs::tests::A");
    }

    /// Two owners for one slug is the defect, and in a test build it is a
    /// panic naming both. The message has to carry both type names: the point
    /// of the check is telling the author *which* pair to separate.
    #[test]
    #[should_panic(expected = "slug_test_shared")]
    fn two_owners_for_one_slug_is_a_collision() {
        claim("slug_test_shared", "reims_vgpu::observe::slugs::tests::A");
        claim("slug_test_shared", "reims_vgpu::observe::slugs::tests::B");
    }

    /// The claim is recorded under the first owner, so the pair reads the same
    /// way round on every later collision rather than swapping roles.
    #[test]
    fn the_first_claimant_keeps_the_slug() {
        claim(
            "slug_test_order",
            "reims_vgpu::observe::slugs::tests::First",
        );
        assert_eq!(
            owners().read().expect("owners").get("slug_test_order"),
            Some(&"reims_vgpu::observe::slugs::tests::First"),
        );
    }

    /// A generic type's name carries a space, and a value with a space in it
    /// splits one log field into two.
    #[test]
    fn a_generic_owner_name_stays_one_field() {
        assert_eq!(compact("Foo<A, B>"), "Foo<A,B>");
        assert!(!compact(std::any::type_name::<HashMap<u32, u64>>()).contains(' '));
    }
}
