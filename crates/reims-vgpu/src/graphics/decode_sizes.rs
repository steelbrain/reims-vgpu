//! The sizes a host decoder takes, as its arm's own query of the decoder lists them
//! ([`DecodeSizes`]), and why it takes no picture of a size outside them. A VideoToolbox session
//! of a stream outside them is refused at its start (`GraphicsVideo::guest_declines`), as the
//! decoder would refuse its first picture.

/// The sizes of the pictures a decoder takes, as its arm's query lists them: each side from
/// `min` up to `max`, and at most `max_macroblocks` 16x16 blocks a picture. A bound the query
/// lists none of is 0, `u32::MAX` or `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecodeSizes {
    /// The narrowest and the shortest picture.
    pub min: (u32, u32),
    /// The widest and the tallest.
    pub max: (u32, u32),
    /// The most 16x16 blocks a picture holds.
    pub max_macroblocks: Option<u32>,
}

impl DecodeSizes {
    /// Why a decoder of these sizes takes no picture of `width` by `height`, its line led by
    /// `what` ("NVDEC decodes H.264"); `None` where it takes it.
    pub(super) fn declines(&self, what: &str, width: u32, height: u32) -> Option<String> {
        let ((min_width, min_height), (max_width, max_height)) = (self.min, self.max);
        let blocks = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
        let too_many = (self.max_macroblocks).is_some_and(|most| blocks > u64::from(most));
        if width < min_width || height < min_height {
            Some(format!(
                "{what} from {min_width}x{min_height}, not {width}x{height}"
            ))
        } else if width > max_width || height > max_height || too_many {
            Some(match self.max_macroblocks {
                Some(most) => format!(
                    "{what} up to {max_width}x{max_height} ({most} macroblocks), not \
                     {width}x{height}"
                ),
                None => format!("{what} up to {max_width}x{max_height}, not {width}x{height}"),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeSizes;

    /// A picture is taken from the least size up to the largest and the most blocks, each side
    /// and bound alone declining it
    #[test]
    fn a_picture_outside_any_bound_is_declined_naming_the_bounds() {
        let sizes = DecodeSizes {
            min: (48, 64),
            max: (4096, 4096),
            max_macroblocks: Some(65_536),
        };
        for (width, height) in [(48, 64), (4096, 4096), (1920, 1080)] {
            assert_eq!(sizes.declines("D", width, height), None, "{width}x{height}");
        }
        let from = "D from 48x64, not";
        let up_to = "D up to 4096x4096 (65536 macroblocks), not";
        for (width, height, why) in [
            (32, 64, format!("{from} 32x64")),
            (48, 48, format!("{from} 48x48")),
            (4112, 64, format!("{up_to} 4112x64")),
            (48, 4112, format!("{up_to} 48x4112")),
        ] {
            assert_eq!(sizes.declines("D", width, height), Some(why));
        }
        let fewer = DecodeSizes {
            max_macroblocks: Some(8160),
            ..sizes
        };
        assert_eq!(fewer.declines("D", 1920, 1088), None);
        assert_eq!(
            fewer.declines("D", 1920, 1104).as_deref(),
            Some("D up to 4096x4096 (8160 macroblocks), not 1920x1104")
        );
        let listed_none = DecodeSizes {
            min: (0, 0),
            max: (u32::MAX, u32::MAX),
            max_macroblocks: None,
        };
        for (width, height) in [(1, 1), (16, 16), (u32::MAX, u32::MAX)] {
            assert_eq!(listed_none.declines("D", width, height), None);
        }
        let unblocked = DecodeSizes {
            max_macroblocks: None,
            ..sizes
        };
        assert_eq!(
            unblocked.declines("D", 4097, 64).as_deref(),
            Some("D up to 4096x4096, not 4097x64")
        );
    }
}
