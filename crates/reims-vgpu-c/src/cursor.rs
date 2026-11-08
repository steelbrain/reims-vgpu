//! The guest's cursor on the graphics device's display, as the host is told it
//! (`RvgGraphicsCallbacks`' `cursor_*`): [`RvgCursorGlyph`], the three callback types, and the
//! position [`RvgDisplayCoord`] that `rvg_graphics_device_cursor_position` answers.

use std::ffi::c_void;
use std::sync::Arc;

use reims_vgpu::{CursorGlyph, CursorGlyphFn, CursorMoveFn, CursorShowFn};

use crate::ffi;
use crate::sized;
use crate::table::Table;

/// A position or size in pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgDisplayCoord {
    pub x: u16,
    pub y: u16,
}

/// The cursor's image, borrowed for the call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgCursorGlyph {
    /// sizeof(RvgCursorGlyph) as the library was compiled.
    pub struct_size: u32,
    pub width: u32,
    pub height: u32,
    /// The hot spot, in pixels from the top left.
    pub hot_spot: RvgDisplayCoord,
    /// `width * 4` bytes a row, `height` rows: straight-alpha RGBA.
    pub rgba: *const u8,
    pub rgba_length: usize,
}

/// `cursor_glyph(ctx, glyph)`: the guest set its cursor's image.
///
/// Thread: the library's display thread, a serial worker that runs every display callback of
/// the process one at a time, in the order the library queued them, and never inside a call of
/// the host's. After 5 s idle it exits, and the next callback starts a new one.
/// May call: every function of this header. A display callback may block (the host's big lock,
/// for example): the library never waits for this thread, and the next display callback waits
/// until it returns.
pub type RvgCursorGlyphFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, glyph: *const RvgCursorGlyph)>;

/// `cursor_show(ctx, show)`: whether the cursor shows changed: 1 shows it, 0 hides it.
///
/// Thread, and what it may call: as RvgCursorGlyphFn.
pub type RvgCursorShowFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, show: u8)>;

/// `cursor_move(ctx)`: the guest moved the cursor; rvg_graphics_device_cursor_position reads
/// where.
///
/// Thread, and what it may call: as RvgCursorGlyphFn.
pub type RvgCursorMoveFn = Option<unsafe extern "C-unwind" fn(ctx: *mut c_void)>;

/// The Rust API's closure over a table's `cursor_glyph`: it holds the table, and lends the host
/// the glyph for the call.
pub(crate) fn cursor_glyph(t: &Arc<Table>, f: RvgCursorGlyphFn) -> Option<CursorGlyphFn> {
    f.map(|f| {
        let t = t.clone();
        let c: CursorGlyphFn = Arc::new(move |g: &CursorGlyph<'_>| {
            let glyph = RvgCursorGlyph {
                struct_size: sized::size_u32::<RvgCursorGlyph>(),
                width: g.width,
                height: g.height,
                hot_spot: RvgDisplayCoord {
                    x: g.hot_spot.0,
                    y: g.hot_spot.1,
                },
                rgba: g.rgba.as_ptr(),
                rgba_length: g.rgba.len(),
            };
            let glyph = &raw const glyph;
            // SAFETY: the host's function over its ctx; the glyph and its pixels are live for the
            // call.
            ffi::host_call("cursor_glyph", || unsafe { f(t.ctx(), glyph) });
        });
        c
    })
}

/// The Rust API's closure over a table's `cursor_show`, as [`cursor_glyph`]'s.
pub(crate) fn cursor_show(t: &Arc<Table>, f: RvgCursorShowFn) -> Option<CursorShowFn> {
    f.map(|f| {
        let t = t.clone();
        let c: CursorShowFn = Arc::new(move |show| {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("cursor_show", || unsafe { f(t.ctx(), show) });
        });
        c
    })
}

/// The Rust API's closure over a table's `cursor_move`, as [`cursor_glyph`]'s.
pub(crate) fn cursor_move(t: &Arc<Table>, f: RvgCursorMoveFn) -> Option<CursorMoveFn> {
    f.map(|f| {
        let t = t.clone();
        let c: CursorMoveFn = Arc::new(move || {
            // SAFETY: the host's function over its ctx.
            ffi::host_call("cursor_move", || unsafe { f(t.ctx()) });
        });
        c
    })
}
