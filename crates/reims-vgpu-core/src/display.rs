//! Backend-independent display and cursor state.

/// Complete cursor position and visibility snapshot for a host adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorPosition {
    pub x: u16,
    pub y: u16,
    pub visible: bool,
}

/// Borrowed complete cursor glyph for a host adapter.
#[derive(Clone, Copy, Debug)]
pub struct CursorGlyph<'a> {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub pixels: &'a [u32],
}

/// Hardware cursor state with atomic glyph publication.
#[derive(Clone, Debug, Default)]
pub struct CursorState {
    show: bool,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    hot_x: u16,
    hot_y: u16,
    /// Host cursor pixels as `0xAARRGGBB`.
    pixels: Vec<u32>,
    glyph_ready: bool,
}

impl CursorState {
    pub fn initially_visible() -> Self {
        Self {
            show: true,
            ..Self::default()
        }
    }

    pub fn position(&self) -> CursorPosition {
        CursorPosition {
            x: self.x,
            y: self.y,
            visible: self.show,
        }
    }

    pub fn set_position(&mut self, x: u16, y: u16) {
        self.x = x;
        self.y = y;
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.show = visible;
    }

    pub fn publish_glyph(
        &mut self,
        width: u16,
        height: u16,
        hot_x: u16,
        hot_y: u16,
        pixels: Vec<u32>,
    ) {
        self.width = width;
        self.height = height;
        self.hot_x = hot_x;
        self.hot_y = hot_y;
        self.pixels = pixels;
        self.glyph_ready = true;
    }

    pub fn glyph(&self) -> Option<CursorGlyph<'_>> {
        (self.glyph_ready && !self.pixels.is_empty()).then_some(CursorGlyph {
            width: self.width,
            height: self.height,
            hot_x: self.hot_x,
            hot_y: self.hot_y,
            pixels: &self.pixels,
        })
    }
}

/// Guest display shared-state handshake and online-redrive progress.
#[derive(Clone, Debug, Default)]
pub struct DisplayHandshake {
    shared_gpa: u64,
    display_index: u32,
    online_acked: bool,
    online_tries: u32,
    poll_ctr: u32,
}

/// Published display shared page and the interrupt bit that belongs to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplaySharedPage {
    pub gpa: u64,
    pub display_index: u32,
}

/// One online-handshake poll after lifecycle admission and cadence decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayOnlinePoll {
    /// No shared page is published, or this generation was already acknowledged.
    Idle,
    /// This generation exhausted its permitted online pulses.
    Exhausted(DisplaySharedPage),
    /// Read the enable mask. `pulse_if_enabled` carries the retry cadence decision.
    Inspect {
        page: DisplaySharedPage,
        poll_count: u32,
        pulse_if_enabled: bool,
        first_pulse: bool,
    },
}

impl DisplayHandshake {
    /// Publish a new shared-state page and restart the online handshake.
    /// Returns whether the previous handshake had reached online.
    pub fn reinitialize(&mut self, display_index: u32, shared_gpa: u64) -> bool {
        let was_online = self.online_acked;
        self.display_index = display_index;
        self.shared_gpa = shared_gpa;
        self.online_acked = false;
        self.online_tries = 0;
        self.poll_ctr = 0;
        was_online
    }

    /// Return the currently published shared page, if any.
    pub fn shared_page(&self) -> Option<DisplaySharedPage> {
        (self.shared_gpa != 0).then_some(DisplaySharedPage {
            gpa: self.shared_gpa,
            display_index: self.display_index,
        })
    }

    pub fn is_online(&self) -> bool {
        self.online_acked
    }

    /// Complete the online handshake for the current shared-page generation.
    pub fn acknowledge_online(&mut self) {
        self.online_acked = true;
    }

    /// Advance one online poll and decide whether the caller may inspect/pulse.
    pub fn begin_online_poll(&mut self, max_tries: u32, retry_divisor: u32) -> DisplayOnlinePoll {
        let Some(page) = self.shared_page() else {
            return DisplayOnlinePoll::Idle;
        };
        if self.online_acked {
            return DisplayOnlinePoll::Idle;
        }
        if self.online_tries >= max_tries {
            return DisplayOnlinePoll::Exhausted(page);
        }

        self.poll_ctr = self.poll_ctr.wrapping_add(1);
        DisplayOnlinePoll::Inspect {
            page,
            poll_count: self.poll_ctr,
            pulse_if_enabled: self.online_tries == 0 || self.poll_ctr.is_multiple_of(retry_divisor),
            first_pulse: self.online_tries == 0,
        }
    }

    /// Record a pulse admitted by [`Self::begin_online_poll`].
    pub fn record_online_pulse(&mut self) {
        self.online_tries = self.online_tries.saturating_add(1);
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn online_progress(&self) -> (u32, u32) {
        (self.online_tries, self.poll_ctr)
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn online_tries(&self) -> u32 {
        self.online_tries
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn poll_count(&self) -> u32 {
        self.poll_ctr
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn set_online_progress(&mut self, online_tries: u32, poll_ctr: u32) {
        self.online_tries = online_tries;
        self.poll_ctr = poll_ctr;
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn set_online_tries(&mut self, online_tries: u32) {
        self.online_tries = online_tries;
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn set_poll_count(&mut self, poll_ctr: u32) {
        self.poll_ctr = poll_ctr;
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn withdraw_online_ack(&mut self) {
        self.online_acked = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glyph_publication_exposes_one_complete_snapshot() {
        let mut cursor = CursorState::initially_visible();
        assert!(cursor.position().visible);
        assert!(cursor.glyph().is_none());

        cursor.publish_glyph(2, 3, 1, 2, vec![0xff00_0000; 6]);
        let glyph = cursor.glyph().unwrap();
        assert_eq!((glyph.width, glyph.height), (2, 3));
        assert_eq!((glyph.hot_x, glyph.hot_y), (1, 2));
        assert_eq!(glyph.pixels.len(), 6);
    }

    #[test]
    fn shared_state_reinitialization_withdraws_every_prior_online_witness() {
        let mut display = DisplayHandshake::default();
        display.reinitialize(1, 0x1000);
        display.acknowledge_online();
        display.set_online_progress(3, 9);
        assert!(display.reinitialize(2, 0x4000));
        assert_eq!(
            display.shared_page(),
            Some(DisplaySharedPage {
                display_index: 2,
                gpa: 0x4000
            })
        );
        assert!(!display.is_online());
        assert_eq!(display.online_progress(), (0, 0));
    }

    #[test]
    fn online_poll_owns_admission_cadence_and_retry_progress() {
        let mut display = DisplayHandshake::default();
        assert_eq!(display.begin_online_poll(3, 4), DisplayOnlinePoll::Idle);

        display.reinitialize(2, 0x4000);
        assert_eq!(
            display.begin_online_poll(3, 4),
            DisplayOnlinePoll::Inspect {
                page: DisplaySharedPage {
                    gpa: 0x4000,
                    display_index: 2
                },
                poll_count: 1,
                pulse_if_enabled: true,
                first_pulse: true,
            }
        );
        display.record_online_pulse();
        assert!(matches!(
            display.begin_online_poll(3, 4),
            DisplayOnlinePoll::Inspect {
                pulse_if_enabled: false,
                first_pulse: false,
                ..
            }
        ));
        display.set_online_progress(3, 8);
        assert!(matches!(
            display.begin_online_poll(3, 4),
            DisplayOnlinePoll::Exhausted(_)
        ));
    }
}
