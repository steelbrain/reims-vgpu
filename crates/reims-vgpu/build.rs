//! Product staticlib is pure Rust for protocol utilities.
//!
//! Pre-rewrite C packages lived under `host/utils/` (deleted; git history only).
//!
//! The one thing it computes is whether the wire crate's oracle fixtures are on
//! disk, so `tests/wire_fixtures_reach_the_decoders.rs` stands down as `ignored`
//! rather than as `ok` when they are not. See `../reims-vgpu-wire/oracle/fixture_presence.rs`
//! for why the answer is shared with the wire crate instead of decided twice.

include!("../reims-vgpu-wire/oracle/fixture_presence.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../reims-vgpu-wire/oracle/fixture_presence.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    probe_wire_fixtures(&format!("{manifest}/../reims-vgpu-wire/fixtures"));
}
