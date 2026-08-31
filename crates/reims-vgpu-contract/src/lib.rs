//! Stable contract facts: layouts, formats, arithmetic, little-endian readers.
//!
//! Pure data and pure functions only — no guest state, no GPU, no QEMU.
//! Source of truth for numbers that come from the wire/SDK/`*_format.h`
//! contracts.
//!
//! # Backend-neutral, and now provably so
//!
//! This is the protocol vocabulary: decoded enums, descriptors, geometry, pixel
//! formats, pass actions, page arithmetic, and the exact refusals each check
//! names. What belongs *above* it is everything that knows how the host draws —
//! Vulkan handles, SPIR-V, memory placement, descriptors, queue families, image
//! layouts, host capability policy — and everything that knows the device is
//! attached to QEMU.
//!
//! That was a rule stated in this doc and held by habit while every module sat
//! in one crate. As a crate it is a fact: `ash`, Metal, QEMU and the device's
//! own state are not in scope here, so a contract check cannot reach one by
//! accident, and a reviewer does not have to notice that it did.
//!
//! The one dependency that looks like a device dependency and is not is
//! `reims_vgpu_observe`: a check that refuses has to be able to *name* its
//! refusal, and the [`Decline`](reims_vgpu_observe::Decline) vocabulary is that
//! name. It carries no policy and selects nothing.

pub mod checked;
pub mod dispatch;
pub mod draw;
pub mod endian;
pub mod extent;
pub mod fnv;
pub mod gva;
pub mod gva_resolve;
pub mod iosurface_pages;
pub mod mipmap;
pub mod pass_action;
pub mod pixel_format;
pub mod vertex_step;
pub mod visibility;

pub use checked::*;
