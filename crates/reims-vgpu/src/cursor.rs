//! The guest's cursor on a graphics device's display, as its host is told it
//! ([`GraphicsCallbacks`](crate::GraphicsCallbacks)' `cursor_*`).

use std::sync::Arc;

/// The cursor's image, borrowed for the callback.
#[derive(Clone, Copy, Debug)]
pub struct CursorGlyph<'a> {
    /// Pixels wide.
    pub width: u32,
    /// Pixels high.
    pub height: u32,
    /// The hot spot, in pixels from the top left.
    pub hot_spot: (u16, u16),
    /// `width * 4` bytes a row, `height` rows: straight-alpha RGBA.
    pub rgba: &'a [u8],
}

/// The guest set its cursor's image.
pub type CursorGlyphFn = Arc<dyn Fn(&CursorGlyph<'_>) + Send + Sync>;
/// Whether the cursor shows changed: 1 shows it, 0 hides it.
pub type CursorShowFn = Arc<dyn Fn(u8) + Send + Sync>;
/// The guest moved the cursor.
pub type CursorMoveFn = Arc<dyn Fn() + Send + Sync>;
