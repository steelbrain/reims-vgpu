//! What a `generateMipmapsForTexture:` request must satisfy before Metal sees it.
//!
//! Every check here is arithmetic over a guest format code, a width, a height, a
//! level count and a byte length. None of it names a Metal object, so none of it
//! needs a device — and the refusals it produces are the ones a malformed guest
//! request actually hits, which is why they are the half worth being able to run.
//!
//! # Why this is not under `backend/metal/`
//!
//! It was, and the whole of it was unreachable on any host without an Apple
//! linker: `src/backend/metal/` is `cfg`-ed out of the arm a Linux host can
//! build, so its tests are not skipped or ignored but simply absent, and a green
//! run there reads exactly like a clean tree. Five of that module's six tests
//! never reached `system_device()` — they walked this argument ladder and
//! checked which refusal came back — so they were testing portable arithmetic
//! from inside a gate that stopped anyone from running them.
//!
//! `backend::hash` is the worked example this follows, and the bar it set is the
//! one that matters: **this module names nothing from the `metal` crate.** The
//! one step that does — turning an accepted format code into an `MTLPixelFormat`
//! — stays in `backend::metal::mipmap`, which composes it onto
//! [`filterable_bpp`]. Move anything here that needs a `metal` type and the whole
//! file goes back behind the gate.
//!
//! That module is named in prose rather than linked because it does not exist on
//! the arm this file's own tests run on, which is the point being made.
//!
//! # Why the slugs still say `metal_`
//!
//! Because the contract being checked is Metal's. These refusals name Metal's
//! own preconditions for filtered mip generation — integer formats are not
//! filterable, a single-level texture has nothing to generate — and the slug is
//! the vocabulary a boot log is read in. Renaming them would move the strings a
//! log parser greps for in exchange for nothing. `contract::pixel_format` sets
//! the same precedent: it holds `MTL_FORMAT_*` constants and is not in the
//! backend either, because a number that comes from the wire and the SDK is a
//! contract fact wherever it is checked.

use crate::contract::pixel_format::{self, bytes_per_pixel};

/// Exact failed checks for the Metal mipmap path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalMipmapError {
    NoDevice,
    UnsupportedFormat {
        format: u16,
    },
    WidthZero,
    HeightZero,
    LevelCountTooSmall {
        levels: u32,
    },
    BaseSpanOverflow {
        width: u32,
        height: u32,
        bpp: u32,
    },
    Level0TooShort {
        len: usize,
        expected: u64,
    },
    LevelCountRejected {
        requested: u32,
        actual: u64,
    },
    /// `newTextureWithDescriptor:` returned nil — the device has no memory for
    /// this texture.
    ///
    /// Distinct from [`Self::LevelCountRejected`] on purpose, because that
    /// variant used to absorb this case and name it wrongly: an unchecked nil
    /// answers `mipmapLevelCount` with 0, and `0 < levels` reported a device
    /// that had refused the *level count* when what it had refused was the
    /// allocation.
    TextureAllocationFailed {
        width: u32,
        height: u32,
        levels: u32,
    },
    CommandBufferFailed,
    LevelSpanOverflow {
        level: u32,
        row_bytes: u64,
        height: u32,
    },
}

impl crate::observe::Decline for MetalMipmapError {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoDevice => "metal_mipmap_device_unavailable",
            Self::UnsupportedFormat { .. } => "metal_mipmap_format_unsupported",
            Self::WidthZero => "metal_mipmap_width_zero",
            Self::HeightZero => "metal_mipmap_height_zero",
            Self::LevelCountTooSmall { .. } => "metal_mipmap_level_count_too_small",
            Self::BaseSpanOverflow { .. } => "metal_mipmap_base_span_overflow",
            Self::Level0TooShort { .. } => "metal_mipmap_level0_too_short",
            Self::LevelCountRejected { .. } => "metal_mipmap_level_count_rejected",
            Self::TextureAllocationFailed { .. } => "metal_mipmap_texture_allocation_failed",
            Self::CommandBufferFailed => "metal_mipmap_command_buffer_failed",
            Self::LevelSpanOverflow { .. } => "metal_mipmap_level_span_overflow",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoDevice | Self::WidthZero | Self::HeightZero | Self::CommandBufferFailed => {
                Vec::new()
            }
            Self::UnsupportedFormat { format } => vec![("format", format.to_string())],
            Self::LevelCountTooSmall { levels } => vec![("levels", levels.to_string())],
            Self::BaseSpanOverflow { width, height, bpp } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("bpp", bpp.to_string()),
            ],
            Self::Level0TooShort { len, expected } => {
                vec![("len", len.to_string()), ("expected", expected.to_string())]
            }
            Self::LevelCountRejected { requested, actual } => vec![
                ("requested", requested.to_string()),
                ("actual", actual.to_string()),
            ],
            Self::TextureAllocationFailed {
                width,
                height,
                levels,
            } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("levels", levels.to_string()),
            ],
            Self::LevelSpanOverflow {
                level,
                row_bytes,
                height,
            } => vec![
                ("level", level.to_string()),
                ("row_bytes", row_bytes.to_string()),
                ("height", height.to_string()),
            ],
        }
    }
}

/// Bytes per pixel for a guest format Metal will filter, or `None` if it will not.
///
/// Filterability is not a property this device chooses: Metal's blit encoder
/// filters normalized and float formats and refuses integer ones, because there
/// is no defined average of two integer texels. So an integer format here is a
/// request Metal itself would reject, and returning `None` is what makes the
/// refusal name the reason instead of surfacing as a command-buffer error.
///
/// The set is spelled as an explicit match rather than as "anything
/// `bytes_per_pixel` knows", because those are different questions — this device
/// can size a `RGBA8_UINT` texel perfectly well and still must not filter it.
#[must_use]
pub fn filterable_bpp(format: u16) -> Option<u32> {
    let bpp = bytes_per_pixel(format)?;
    match format {
        pixel_format::MTL_FORMAT_A8_UNORM
        | pixel_format::MTL_FORMAT_R8_UNORM
        | pixel_format::MTL_FORMAT_RG8_UNORM
        | pixel_format::MTL_FORMAT_R16_FLOAT
        | pixel_format::MTL_FORMAT_RG16_FLOAT
        | pixel_format::MTL_FORMAT_RGBA8_UNORM
        | pixel_format::MTL_FORMAT_RGBA8_UNORM_SRGB
        | pixel_format::MTL_FORMAT_BGRA8_UNORM
        | pixel_format::MTL_FORMAT_BGRA8_UNORM_SRGB
        | pixel_format::MTL_FORMAT_RGBA16_FLOAT
        | pixel_format::MTL_FORMAT_RGBA32_FLOAT => Some(bpp),
        _ => None,
    }
}

/// The level-0 geometry a filtered mip generation will upload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level0Plan {
    /// Bytes per pixel of the guest format, from [`filterable_bpp`].
    pub bpp: u32,
    /// Tightly packed byte span of level 0 — `width * height * bpp`.
    pub tight_bytes: u64,
    /// Tightly packed row span of level 0 — `width * bpp`.
    pub bytes_per_row: u64,
}

/// Check a mip-generation request and size its level 0, or name what it failed.
///
/// # Why the order of these checks is the contract
///
/// Each refusal is a different sentence about the request, and a caller that
/// reordered them would report a different one for the same input — so the order
/// is pinned by a test rather than left to whichever check a future edit happens
/// to put first. It runs from the cheapest and most local outward: the two axes
/// separately (so a reading says *which* was zero), then the level count, then
/// the format, and only then the arithmetic that needs all four.
///
/// `level0_len` is a length rather than the slice, because nothing here reads a
/// byte of it and taking the slice would let a future edit start.
pub fn plan_level0(
    format: u16,
    width: u32,
    height: u32,
    levels: u32,
    level0_len: usize,
) -> Result<Level0Plan, MetalMipmapError> {
    if width == 0 {
        return Err(MetalMipmapError::WidthZero);
    }
    if height == 0 {
        return Err(MetalMipmapError::HeightZero);
    }
    if levels <= 1 {
        return Err(MetalMipmapError::LevelCountTooSmall { levels });
    }
    let bpp = filterable_bpp(format).ok_or(MetalMipmapError::UnsupportedFormat { format })?;
    let tight_bytes = (width as u64)
        .checked_mul(height as u64)
        .and_then(|v| v.checked_mul(bpp as u64))
        .ok_or(MetalMipmapError::BaseSpanOverflow { width, height, bpp })?;
    if (level0_len as u64) < tight_bytes {
        return Err(MetalMipmapError::Level0TooShort {
            len: level0_len,
            expected: tight_bytes,
        });
    }
    Ok(Level0Plan {
        bpp,
        tight_bytes,
        // Both factors are u32, so their product always fits in u64.
        bytes_per_row: (width as u64) * (bpp as u64),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::pixel_format::MTL_FORMAT_RGBA8_UNORM;
    use crate::observe::{Decline, Emit};

    #[test]
    fn filterable_accepts_unorm_rejects_uint() {
        assert!(filterable_bpp(MTL_FORMAT_RGBA8_UNORM).is_some());
        assert!(filterable_bpp(pixel_format::MTL_FORMAT_BGRA8_UNORM).is_some());
        assert!(filterable_bpp(pixel_format::MTL_FORMAT_RGBA8_UINT).is_none());
        assert!(filterable_bpp(pixel_format::MTL_FORMAT_RGBA16_UINT).is_none());
    }

    /// A format this device can size but Metal will not filter is still refused.
    ///
    /// The distinction the explicit match exists for. `bytes_per_pixel` answers
    /// for the integer formats above, so a `filterable_bpp` written as "whatever
    /// `bytes_per_pixel` knows" would accept every one of them and hand Metal a
    /// blit it cannot perform.
    #[test]
    fn a_sizeable_format_is_not_therefore_a_filterable_one() {
        for format in [
            pixel_format::MTL_FORMAT_RGBA8_UINT,
            pixel_format::MTL_FORMAT_RGBA16_UINT,
        ] {
            assert!(
                bytes_per_pixel(format).is_some(),
                "{format} is sizeable, which is what makes it the interesting case"
            );
            assert_eq!(filterable_bpp(format), None, "{format} is not filterable");
        }
    }

    #[test]
    fn rejects_single_level() {
        let error = plan_level0(MTL_FORMAT_RGBA8_UNORM, 1, 1, 1, 4).unwrap_err();
        assert_eq!(error, MetalMipmapError::LevelCountTooSmall { levels: 1 });
        assert_eq!(
            Emit::decline("metal_mipmap_test", &error).render(),
            "metal_mipmap_test reason=metal_mipmap_level_count_too_small levels=1"
        );
    }

    #[test]
    fn rejects_uint() {
        let error = plan_level0(pixel_format::MTL_FORMAT_RGBA8_UINT, 1, 1, 2, 4).unwrap_err();
        assert_eq!(
            error,
            MetalMipmapError::UnsupportedFormat {
                format: pixel_format::MTL_FORMAT_RGBA8_UINT
            }
        );
        assert_eq!(
            Emit::decline("metal_mipmap_test", &error).render(),
            format!(
                "metal_mipmap_test reason=metal_mipmap_format_unsupported format={}",
                pixel_format::MTL_FORMAT_RGBA8_UINT
            )
        );
    }

    #[test]
    fn reports_the_level_zero_byte_requirement() {
        let error = plan_level0(MTL_FORMAT_RGBA8_UNORM, 2, 2, 2, 15).unwrap_err();
        assert_eq!(
            error,
            MetalMipmapError::Level0TooShort {
                len: 15,
                expected: 16
            }
        );
        assert_eq!(
            Emit::decline("metal_mipmap_test", &error).render(),
            "metal_mipmap_test reason=metal_mipmap_level0_too_short len=15 expected=16"
        );
    }

    #[test]
    fn names_each_zero_axis_separately() {
        let width = plan_level0(MTL_FORMAT_RGBA8_UNORM, 0, 1, 2, 4).unwrap_err();
        let height = plan_level0(MTL_FORMAT_RGBA8_UNORM, 1, 0, 2, 4).unwrap_err();

        assert_eq!(width, MetalMipmapError::WidthZero);
        assert_eq!(height, MetalMipmapError::HeightZero);
        assert_eq!(
            Emit::decline("metal_mipmap_test", &width).render(),
            "metal_mipmap_test reason=metal_mipmap_width_zero"
        );
        assert_eq!(
            Emit::decline("metal_mipmap_test", &height).render(),
            "metal_mipmap_test reason=metal_mipmap_height_zero"
        );
    }

    /// An accepted request reports the two spans the upload is driven from.
    ///
    /// `bytes_per_row` is what the Metal texture upload is given as its stride,
    /// and `tight_bytes` is what the level-0 length was compared against, so a
    /// plan that returned one of them wrong would upload skewed rows rather than
    /// refuse anything — a wrong picture with no failure line.
    #[test]
    fn an_accepted_request_carries_both_level_zero_spans() {
        assert_eq!(
            plan_level0(MTL_FORMAT_RGBA8_UNORM, 4, 4, 3, 64),
            Ok(Level0Plan {
                bpp: 4,
                tight_bytes: 64,
                bytes_per_row: 16,
            })
        );
        // Exactly enough is enough; the comparison is `<`, not `<=`.
        assert!(plan_level0(MTL_FORMAT_RGBA8_UNORM, 2, 2, 2, 16).is_ok());
        assert!(plan_level0(MTL_FORMAT_RGBA8_UNORM, 2, 2, 2, 15).is_err());
        // A longer level 0 is accepted — the guest may hand over a padded buffer.
        assert!(plan_level0(MTL_FORMAT_RGBA8_UNORM, 2, 2, 2, 1024).is_ok());
    }

    /// The ladder's order, pinned: each input fails only its own check.
    ///
    /// Written as one request that is wrong in *every* way, then relaxed one
    /// step at a time. A reordering that let the format check run before the
    /// zero-axis checks would report `format_unsupported` for a zero-width
    /// request, and every individual assertion above would still pass.
    #[test]
    fn the_refusal_ladder_reports_the_outermost_failure_first() {
        let bad = pixel_format::MTL_FORMAT_RGBA8_UINT;
        let ladder = [
            (0, 0, 1, bad, 0, MetalMipmapError::WidthZero),
            (4, 0, 1, bad, 0, MetalMipmapError::HeightZero),
            (
                4,
                4,
                1,
                bad,
                0,
                MetalMipmapError::LevelCountTooSmall { levels: 1 },
            ),
            (
                4,
                4,
                2,
                bad,
                0,
                MetalMipmapError::UnsupportedFormat { format: bad },
            ),
            (
                4,
                4,
                2,
                MTL_FORMAT_RGBA8_UNORM,
                0,
                MetalMipmapError::Level0TooShort {
                    len: 0,
                    expected: 64,
                },
            ),
        ];
        for (w, h, levels, format, len, expected) in ladder {
            assert_eq!(
                plan_level0(format, w, h, levels, len),
                Err(expected),
                "w={w} h={h} levels={levels} format={format} len={len}"
            );
        }
    }

    /// Every refusal renders with a distinct slug.
    ///
    /// The four this module cannot reach — `NoDevice`, `CommandBufferFailed`,
    /// `LevelCountRejected` and `LevelSpanOverflow` — are Metal's own answers,
    /// raised in [`crate::backend::metal::mipmap`] where nobody on a non-Apple
    /// host can execute them. Their slugs and fields are pure data, so they are
    /// checked here, which is the only arm that runs.
    #[test]
    fn every_refusal_names_itself() {
        let all = [
            MetalMipmapError::NoDevice,
            MetalMipmapError::UnsupportedFormat { format: 1 },
            MetalMipmapError::WidthZero,
            MetalMipmapError::HeightZero,
            MetalMipmapError::LevelCountTooSmall { levels: 1 },
            MetalMipmapError::BaseSpanOverflow {
                width: 1,
                height: 2,
                bpp: 3,
            },
            MetalMipmapError::Level0TooShort {
                len: 1,
                expected: 2,
            },
            MetalMipmapError::LevelCountRejected {
                requested: 1,
                actual: 2,
            },
            MetalMipmapError::CommandBufferFailed,
            MetalMipmapError::LevelSpanOverflow {
                level: 1,
                row_bytes: 2,
                height: 3,
            },
        ];
        let mut slugs: Vec<&str> = all.iter().map(Decline::slug).collect();
        let before = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "two refusals share a slug: {slugs:?}");
        for error in &all {
            assert!(
                Emit::decline("t", error).render().contains(error.slug()),
                "{error:?} does not render its own slug"
            );
        }
    }
}
