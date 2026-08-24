//! Guest address resolution above the wire page-table format.
//!
//! `reims-vgpu-wire` owns the page-table bytes and the descent; the device
//! crate owns policy, host mapping and emission. This crate is the layer
//! between: the resolution algorithms that need to allocate but must not see a
//! host. The boundary is structural on both sides — `#![no_std]` + `alloc`
//! keeps host traits and the device's failure channel unreachable from here,
//! and the wire crate's no-allocation invariant keeps this crate's `Vec`s out
//! of there.
//!
//! Membership test for new code, applied in order:
//!
//! - needs a fixture or a format derivation → `reims-vgpu-wire`
//! - needs `alloc` but no host → here
//! - needs `HostOps`, the device model, or the failure channel → `reims-vgpu`
//!
//! The guest-memory seam is the wire crate's
//! [`GuestMemory`](reims_vgpu_wire::mem::GuestMemory) — one trait for both
//! crates, implemented by the device over its host access. An implementation
//! serves exactly one address space; see that trait's docs for why that is a
//! hard rule.

#![no_std]

extern crate alloc;

pub mod geometry;
pub mod mapper;
pub mod regions;
pub mod resolve;
pub mod runs;
pub mod span;
