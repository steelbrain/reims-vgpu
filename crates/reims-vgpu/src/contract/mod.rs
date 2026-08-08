//! Stable contract facts: layouts, formats, arithmetic, little-endian readers.
//!
//! Pure data and pure functions only — no guest state, no GPU, no QEMU.
//! Source of truth for numbers that come from the wire/SDK/`*_format.h` contracts.

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

pub(crate) use checked::*;
