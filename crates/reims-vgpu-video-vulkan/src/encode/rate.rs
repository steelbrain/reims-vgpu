//! Rate control: what the caller asks for ([`RateControl`], [`Vbv`]), and
//! the Vulkan rate control state it becomes ([`RatePlan`]), checked
//! against what the device takes. The plan is plain data, made and checked
//! with no GPU; the session builds `VkVideoEncodeRateControlInfoKHR` from
//! it for each command buffer.

use ash::vk;
use reims_vgpu_video_codec::encode::FrameRate;

use crate::encode::caps::EncodeCapabilities;
use crate::error::Error;

/// How the bitrate is controlled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RateControl {
    /// No rate control: a constant QP for each picture type
    /// (`VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR`).
    ConstantQp {
        /// For I and IDR pictures.
        intra: u8,
        /// For P pictures.
        predicted: u8,
        /// For B pictures.
        bidirectional: u8,
    },
    /// A constant bitrate, in bits a second (`CBR`).
    ConstantBitrate {
        /// The rate.
        bitrate: u64,
    },
    /// A variable bitrate, in bits a second (`VBR`): an average under a
    /// peak.
    VariableBitrate {
        /// The average.
        average: u64,
        /// The peak, at least the average.
        peak: u64,
    },
}

/// The HRD (VBV) buffer a bitrate-controlled stream is kept within, in
/// bits at the peak rate: its size, and how full it starts. Vulkan takes it
/// as the time the peak rate takes to fill it
/// (`virtualBufferSizeInMs`, `initialVirtualBufferSizeInMs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vbv {
    /// The buffer's size.
    pub size_bits: u64,
    /// How full it starts, at most its size.
    pub initial_bits: u64,
}

/// Ours: the buffer where none is asked for: a second at the peak,
/// starting three quarters full, as libva's users size and start it.
pub const DEFAULT_BUFFER_MS: u32 = 1000;

/// Ours: how full the default buffer starts, in milliseconds at the peak.
pub const DEFAULT_INITIAL_MS: u32 = 750;

/// The rate control state a session is given.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RatePlan {
    /// `rateControlMode`.
    pub mode: vk::VideoEncodeRateControlModeFlagsKHR,
    /// The average bitrate of the one layer; 0 for a constant QP.
    pub average: u64,
    /// Its peak.
    pub peak: u64,
    /// The frame rate, in lowest terms.
    pub frame_rate: FrameRate,
    /// `virtualBufferSizeInMs`: 0 for a constant QP.
    pub buffer_ms: u32,
    /// `initialVirtualBufferSizeInMs`, below `buffer_ms`.
    pub initial_ms: u32,
    /// Each picture type's QP (I, P, B), for a constant QP.
    pub constant_qp: Option<[i32; 3]>,
}

impl RatePlan {
    /// The rate control layers the state has: 0 for a constant QP, else 1.
    pub fn layers(&self) -> u32 {
        u32::from(self.constant_qp.is_none())
    }

    /// The QP of a picture of `kind`'s slices, for a constant QP; 0, which
    /// the rate control modes take, otherwise.
    pub fn qp(&self, kind: crate::encode::plan::PictureKind) -> i32 {
        use crate::encode::plan::PictureKind;
        let Some([intra, predicted, bidirectional]) = self.constant_qp else {
            return 0;
        };
        match kind {
            PictureKind::Idr | PictureKind::I => intra,
            PictureKind::P => predicted,
            PictureKind::B => bidirectional,
        }
    }

    /// The bitrate the stream's level and HRD must hold: the peak; none
    /// for a constant QP.
    pub fn peak_bits_per_second(&self) -> Option<u64> {
        self.constant_qp.is_none().then_some(self.peak)
    }
}

/// The state `rate` and `vbv` ask for at `frame_rate`, or why the device,
/// as `caps` answers, does not take it.
pub fn plan(
    rate: RateControl,
    vbv: Option<Vbv>,
    frame_rate: FrameRate,
    caps: &EncodeCapabilities,
) -> Result<RatePlan, Error> {
    if frame_rate.numerator == 0 || frame_rate.denominator == 0 {
        return Err(Error::Unsupported(format!(
            "a frame rate of {}/{}",
            frame_rate.numerator, frame_rate.denominator
        )));
    }
    let frame_rate = frame_rate.reduced();
    let (mode, average, peak) = match rate {
        RateControl::ConstantQp {
            intra,
            predicted,
            bidirectional,
        } => {
            let qps = [intra, predicted, bidirectional].map(i32::from);
            if let Some(qp) = qps.iter().find(|&&qp| qp < caps.min_qp || qp > caps.max_qp) {
                return Err(Error::Unsupported(format!(
                    "QP {qp}: the device takes {} to {}",
                    caps.min_qp, caps.max_qp
                )));
            }
            let mode = vk::VideoEncodeRateControlModeFlagsKHR::DISABLED;
            check_mode(caps, mode, "a constant QP")?;
            return Ok(RatePlan {
                mode,
                average: 0,
                peak: 0,
                frame_rate,
                buffer_ms: 0,
                initial_ms: 0,
                constant_qp: Some(qps),
            });
        }
        RateControl::ConstantBitrate { bitrate } => (
            vk::VideoEncodeRateControlModeFlagsKHR::CBR,
            bitrate,
            bitrate,
        ),
        RateControl::VariableBitrate { average, peak } => {
            (vk::VideoEncodeRateControlModeFlagsKHR::VBR, average, peak)
        }
    };
    check_mode(caps, mode, if peak == average { "CBR" } else { "VBR" })?;
    if average == 0 || peak < average {
        return Err(Error::Unsupported(format!(
            "an average of {average} bit/s under a peak of {peak}"
        )));
    }
    if caps.max_bitrate > 0 && peak > caps.max_bitrate {
        return Err(Error::Unsupported(format!(
            "a peak of {peak} bit/s: the device's is {}",
            caps.max_bitrate
        )));
    }
    let (buffer_ms, initial_ms) = match vbv {
        None => (DEFAULT_BUFFER_MS, DEFAULT_INITIAL_MS),
        Some(vbv) => buffer_ms(vbv, peak)?,
    };
    Ok(RatePlan {
        mode,
        average,
        peak,
        frame_rate,
        buffer_ms,
        initial_ms,
        constant_qp: None,
    })
}

fn check_mode(
    caps: &EncodeCapabilities,
    mode: vk::VideoEncodeRateControlModeFlagsKHR,
    name: &str,
) -> Result<(), Error> {
    if caps.takes(mode) {
        Ok(())
    } else {
        Err(Error::Unsupported(format!(
            "{name}: the device takes rate control modes {:#x}",
            caps.rate_control_modes.as_raw()
        )))
    }
}

/// `vbv` at `peak` bits a second as Vulkan takes it: the size in
/// milliseconds, rounded up so that the buffer holds at least what was
/// asked, and the initial fullness rounded down, below the size.
fn buffer_ms(vbv: Vbv, peak: u64) -> Result<(u32, u32), Error> {
    if vbv.size_bits == 0 || vbv.initial_bits > vbv.size_bits {
        return Err(Error::Unsupported(format!(
            "a buffer of {} bits starting with {}",
            vbv.size_bits, vbv.initial_bits
        )));
    }
    let ms = |bits: u64, up: bool| {
        let scaled = u128::from(bits) * 1000;
        let peak = u128::from(peak.max(1));
        let ms = if up {
            scaled.div_ceil(peak)
        } else {
            scaled / peak
        };
        u32::try_from(ms).unwrap_or(u32::MAX)
    };
    let size = ms(vbv.size_bits, true).max(1);
    let initial = ms(vbv.initial_bits, false).min(size - 1);
    Ok((size, initial))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::encode::caps::{EncodeProfile, QualityPreference};
    use crate::encode::plan::PictureKind;

    /// The 5080's H.264 answers (11-03): CQP, CBR and VBR, QP 0 to 51,
    /// 120 Mbit/s.
    pub(crate) fn caps() -> EncodeCapabilities {
        EncodeCapabilities {
            profile: EncodeProfile::H264_HIGH,
            max_level_idc: 52,
            min_coded_extent: vk::Extent2D {
                width: 160,
                height: 64,
            },
            max_coded_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            picture_access_granularity: vk::Extent2D {
                width: 16,
                height: 16,
            },
            input_granularity: vk::Extent2D {
                width: 16,
                height: 16,
            },
            max_dpb_slots: 17,
            max_active_reference_pictures: 16,
            min_bitstream_buffer_offset_alignment: 256,
            min_bitstream_buffer_size_alignment: 256,
            std_header_version: vk::ExtensionProperties::default(),
            flags: vk::VideoEncodeCapabilityFlagsKHR::empty(),
            rate_control_modes: vk::VideoEncodeRateControlModeFlagsKHR::DISABLED
                | vk::VideoEncodeRateControlModeFlagsKHR::CBR
                | vk::VideoEncodeRateControlModeFlagsKHR::VBR,
            max_rate_control_layers: 1,
            max_bitrate: 120_000_000,
            max_quality_levels: 5,
            feedback_flags: vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BUFFER_OFFSET
                | vk::VideoEncodeFeedbackFlagsKHR::BITSTREAM_BYTES_WRITTEN,
            max_slices: 64,
            max_p_l0: 7,
            max_b_l0: 7,
            max_l1: 1,
            min_qp: 0,
            max_qp: 51,
            requires_gop_remaining_frames: false,
            codec_flags: 0x270,
            std_syntax_flags: 0x39b3a,
            ctb_sizes: 0,
            transform_block_sizes: 0,
            quality_levels: vec![QualityPreference::default(); 5],
        }
    }

    #[test]
    fn a_constant_qp_has_no_layer_and_a_qp_per_type() {
        let rate = RateControl::ConstantQp {
            intra: 22,
            predicted: 25,
            bidirectional: 27,
        };
        let plan = plan(rate, None, FrameRate::new(30, 1), &caps()).unwrap();
        assert_eq!(
            plan.mode.as_raw(),
            vk::VideoEncodeRateControlModeFlagsKHR::DISABLED.as_raw()
        );
        assert_eq!(plan.layers(), 0);
        assert_eq!((plan.buffer_ms, plan.initial_ms), (0, 0));
        assert_eq!(plan.qp(PictureKind::Idr), 22);
        assert_eq!(plan.qp(PictureKind::I), 22);
        assert_eq!(plan.qp(PictureKind::P), 25);
        assert_eq!(plan.qp(PictureKind::B), 27);
        assert_eq!(plan.peak_bits_per_second(), None);
    }

    #[test]
    fn qps_past_the_devices_range_are_refused() {
        let mut caps = caps();
        caps.min_qp = 10;
        let rate = RateControl::ConstantQp {
            intra: 9,
            predicted: 25,
            bidirectional: 27,
        };
        assert!(plan(rate, None, FrameRate::new(30, 1), &caps).is_err());
        let rate = RateControl::ConstantQp {
            intra: 20,
            predicted: 52,
            bidirectional: 27,
        };
        assert!(plan(rate, None, FrameRate::new(30, 1), &caps).is_err());
    }

    #[test]
    fn cbr_peaks_at_its_rate_with_the_default_buffer() {
        let rate = RateControl::ConstantBitrate { bitrate: 4_000_000 };
        let plan = plan(rate, None, FrameRate::new(60_000, 1001), &caps()).unwrap();
        assert_eq!(
            plan.mode.as_raw(),
            vk::VideoEncodeRateControlModeFlagsKHR::CBR.as_raw()
        );
        assert_eq!((plan.average, plan.peak), (4_000_000, 4_000_000));
        assert_eq!(plan.layers(), 1);
        assert_eq!((plan.buffer_ms, plan.initial_ms), (1000, 750));
        assert_eq!(plan.frame_rate, FrameRate::new(60_000, 1001));
        assert_eq!(plan.qp(PictureKind::P), 0);
        assert_eq!(plan.peak_bits_per_second(), Some(4_000_000));
    }

    #[test]
    fn a_vbv_in_bits_becomes_milliseconds_at_the_peak() {
        // 2 Mbit at a 4 Mbit/s peak is 500 ms; 1.5 Mbit is 375.
        let vbv = Vbv {
            size_bits: 2_000_000,
            initial_bits: 1_500_000,
        };
        let rate = RateControl::VariableBitrate {
            average: 3_000_000,
            peak: 4_000_000,
        };
        let plan = plan(rate, Some(vbv), FrameRate::new(30, 1), &caps()).unwrap();
        assert_eq!((plan.buffer_ms, plan.initial_ms), (500, 375));
        // A size between two milliseconds rounds up, so the buffer holds
        // at least what was asked; the fullness rounds down.
        assert_eq!(
            buffer_ms(
                Vbv {
                    size_bits: 2_000_001,
                    initial_bits: 1_999_999
                },
                4_000_000
            )
            .unwrap(),
            (501, 499)
        );
        // A full start is kept below the size.
        assert_eq!(
            buffer_ms(
                Vbv {
                    size_bits: 4_000,
                    initial_bits: 4_000
                },
                4_000_000
            )
            .unwrap(),
            (1, 0)
        );
    }

    #[test]
    fn rates_the_device_does_not_take_are_refused() {
        let caps = caps();
        let fps = FrameRate::new(30, 1);
        let vbr = |average, peak| RateControl::VariableBitrate { average, peak };
        assert!(plan(vbr(5_000_000, 4_000_000), None, fps, &caps).is_err());
        assert!(plan(vbr(0, 4_000_000), None, fps, &caps).is_err());
        assert!(plan(vbr(5_000_000, 130_000_000), None, fps, &caps).is_err());
        assert!(plan(vbr(5_000_000, 120_000_000), None, fps, &caps).is_ok());
        let bad_vbv = Vbv {
            size_bits: 1000,
            initial_bits: 2000,
        };
        assert!(plan(vbr(5_000_000, 6_000_000), Some(bad_vbv), fps, &caps).is_err());
        assert!(
            plan(
                vbr(5_000_000, 6_000_000),
                None,
                FrameRate::new(30, 0),
                &caps
            )
            .is_err()
        );
        let mut no_cbr = caps.clone();
        no_cbr.rate_control_modes = vk::VideoEncodeRateControlModeFlagsKHR::VBR;
        let cbr = RateControl::ConstantBitrate { bitrate: 1_000_000 };
        assert!(plan(cbr, None, fps, &no_cbr).is_err());
        let cqp = RateControl::ConstantQp {
            intra: 20,
            predicted: 22,
            bidirectional: 24,
        };
        assert!(plan(cqp, None, fps, &no_cbr).is_err());
    }
}
