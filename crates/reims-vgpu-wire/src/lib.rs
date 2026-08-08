//! Zero-copy views over the Apple paravirtualized GPU wire format.
//!
//! The guest serializes Metal calls into a command stream with
//! `AppleParavirtGPUMetal.bundle`. This crate reads that stream by *mapping
//! structs onto it* — a length check and a pointer cast — rather than by
//! decoding fields into owned values.
//!
//! Its layouts are not inferred from captures. Apple's serializer runs on the
//! host and emits the same bytes on demand, so each layout is derived from the
//! Objective-C type encoding for field widths and pinned by perturbation for
//! field meaning. See `README.md` for how that works and `AGENTS.md` for the
//! procedure to extend it.
//!
//! # Shape
//!
//! - [`le`] — align-1 little-endian scalars, so wire structs are align-1 and no
//!   buffer offset can be misaligned.
//! - [`view`] — the checked cast, and the [`view::Wire`] contract a type must
//!   meet to be viewable.
//! - [`op`] — the `[opcode][length][payload]` record framing every command
//!   shares, and an iterator over a stream of them.
//! - [`ops`] — one module per operation family. [`ops::texture`] is the worked
//!   example.
//! - [`mem`] — the one thing the crate asks the device for: read this guest
//!   address. Needed because not every structure arrives as a buffer.
//! - [`page_table`] — the guest GPU page table and the walk that resolves a
//!   GVA through it, over [`mem::GuestMemory`]. Its layout comes from the device
//!   contract rather than from the serializer, so no fixture pins it.
//! - [`manifest`] — which selectors are covered, and how far that is from all
//!   of them.
//!
//! # Invariants
//!
//! - **No allocation.** The crate is `#![no_std]` and never allocates: a view
//!   that needs to allocate is a view that copied.
//! - **No unchecked casts.** The bytes are guest-controlled, so every
//!   constructor is fallible. See [`view`] for why that is not parsing.
//! - **No invented fields.** A field nothing has been made to move is named
//!   `unidentified_*` and carries the experiment that would settle it.
//!
//! # Relationship to `reims-vgpu`
//!
//! This crate is the layout authority for serializer records that
//! `reims_vgpu::runtime::decode` consumes. Decode maps wire views into its
//! product model; it does not re-declare covered opcodes or layouts. The two
//! still frame different levels: [`op::OpHeader`] is the serializer's operation
//! head, *not* the FIFO packet header in `decode::fifo`, whose fields collide
//! with it at offsets 4 and 8 while meaning something else.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

pub mod device_desc;
pub mod le;
pub mod manifest;
pub mod mem;
pub mod op;
pub mod ops;
pub mod page_table;
pub mod view;

pub use le::{F32le, F64le, U16le, U32le, U64le};
pub use mem::{GuestMemory, SliceMemory};
pub use op::{op, Op, OpHeader, OpStream, OP_HEADER_LEN};
pub use page_table::{Geometry, Walk, WalkError, WalkFailure};
pub use view::{split, view, view_at, view_slice, Wire, WireError};
