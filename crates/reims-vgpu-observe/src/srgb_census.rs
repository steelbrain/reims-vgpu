//! Always-on observation of a guest sRGB declaration being bound through a
//! linear image view.
//!
//! This is an instrument, not policy: it records a loss already selected by a
//! caller and never changes decode, execution, or presentation. It lives in the
//! shared observation crate because semantic planning and backend projection
//! can both discover the same loss and must share one first-sight latch.

use std::collections::BTreeSet;
use std::sync::Mutex;

pub const SRGB_DOWNGRADED_SLUG: &str = "srgb_downgraded";

pub mod site {
    /// A secondary render target whose transfer qualifier cannot be carried.
    pub const SECONDARY_COLOR_TARGET: &str = "secondary_color_target";
    /// Already-loaded bytes whose layout has no sRGB image-view spelling.
    pub const SAMPLED_BYTE_UPLOAD: &str = "sampled_byte_upload";

    pub const ALL: &[&str] = &[SECONDARY_COLOR_TARGET, SAMPLED_BYTE_UPLOAD];
}

static SEEN: Mutex<BTreeSet<(&'static str, u16)>> = Mutex::new(BTreeSet::new());

pub fn note_downgrade(site: &'static str, mtl: u16) {
    let first_sight = SEEN
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert((site, mtl));
    if first_sight {
        crate::fail(format!(
            "srgb_downgraded reason={SRGB_DOWNGRADED_SLUG} site={site} mtl={mtl:#x} \
             (bound the linear sibling; hardware will not apply the sRGB transfer \
             function on this rail)"
        ));
    }
}

#[cfg(test)]
fn reset_for_tests() {
    SEEN.lock()
        .unwrap_or_else(|error| error.into_inner())
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_site_and_format_pair_reports_once() {
        reset_for_tests();
        for _ in 0..64 {
            note_downgrade(site::SAMPLED_BYTE_UPLOAD, 0x51);
        }
        assert_eq!(SEEN.lock().unwrap().len(), 1);
        note_downgrade(site::SAMPLED_BYTE_UPLOAD, 0x47);
        note_downgrade(site::SECONDARY_COLOR_TARGET, 0x47);
        assert_eq!(SEEN.lock().unwrap().len(), 3);
        reset_for_tests();
    }

    #[test]
    fn site_names_are_distinct_and_log_safe() {
        let mut names = site::ALL.to_vec();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len());
        assert!(site::ALL.iter().all(|name| name
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')));
    }
}
