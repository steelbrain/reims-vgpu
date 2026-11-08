//! Containers named by the order they give.
//!
//! - [`LookupMap`]: maps that are only looked up. No iteration, and a fixed hasher, so its drop
//!   order is the same in every run.
//! - [`KeyHasher`]: the fixed hasher of [`LookupMap`] and of the maps `reims-vgpu`'s decoder
//!   keys by the guest's IDs.

pub mod hasher;
pub mod lookup;

pub use hasher::KeyHasher;
pub use lookup::LookupMap;
