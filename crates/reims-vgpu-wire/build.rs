//! Tells the fixture tests whether Apple's captured records are on disk.
//!
//! See `oracle/fixture_presence.rs` for why the answer is a `cfg` rather than a
//! runtime check.

include!("oracle/fixture_presence.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=oracle/fixture_presence.rs");
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    probe_wire_fixtures(&format!("{manifest}/fixtures"));
}
