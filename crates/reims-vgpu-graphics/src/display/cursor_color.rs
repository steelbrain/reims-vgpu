//! The cursor's colour (`CURSOR_COLOR`, [`RgfxCursorColor`]): the chain a display's cursor plane
//! converts its pixels through, which an arm64 guest's WindowServer gives its cursor's layer and
//! the arm64e kext carries to the host as WindowServer gave it (its ICC blocks' curves and
//! matrices, `guest/kext/src/RgfxMobileICC.hpp`). The cursor's image is standard sRGB, and the
//! guest's desktop is drawn in the display's blend space, the space of the display's profile: a
//! display engine converts the cursor's layer from the one to the other as it blends the layers,
//! and the host does so to the image, before any screen shows it ([`CursorColor::convert`]), so
//! the host's cursor, QEMU's console and the window's drawn cursor all show the converted pixels.
//!
//! **The conversion.** Each pixel's straight colour, its channels in [0, 1], goes through the
//! source's curves channel by channel, the product of the two matrices (from XYZ to the blend
//! space, times the source's to XYZ), and the blend space's curves channel by channel; the result
//! is clamped to [0, 1] and rounded to the nearest of the 256 steps, and the alpha is kept. A
//! pixel's straight colour is converted, not its premultiplied one: the curves are not linear,
//! and a premultiplied colour would be converted as a darker one at the image's soft edges. The
//! work is done in `f64` on the CPU at each new image or chain, once per distinct colour of the
//! image (an image is at most `RGFX_CURSOR_MAX_SIZE` pixels each way). A chain whose curves are
//! all `Bypass` and whose matrices' product is the identity leaves every pixel as it is, and is
//! kept as no chain at all; the guest also sends `on` 0 for a chain that is the identity within
//! its own precision, so an identity costs nothing.
//!
//! **The curves** are the ICC's parametric curves in QuartzCore's forms (`RGFX_CURVE_*`); a power
//! of a base at or below 0 is 0, as the kext evaluates them, and so a curve of a negative input,
//! an out-of-gamut colour's after the matrices, ends at its linear part or 0, which the clamp
//! keeps.

use std::collections::HashMap;

use crate::wire::{
    RGFX_CURVE_BYPASS, RGFX_CURVE_TYPE0, RGFX_CURVE_TYPE1, RGFX_CURVE_TYPE2, RGFX_CURVE_TYPE3,
    RGFX_CURVE_TYPE4, RgfxCursorColor, RgfxCursorCurve,
};

/// Why a `CURSOR_COLOR` was dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dropped {
    /// A curve's form is none of `RGFX_CURVE_*`.
    Form,
    /// A parameter a curve's form reads, or a matrix's entry, is not a finite float.
    NotFinite,
}

/// A curve's form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// `y = x`.
    Bypass,
    /// `y = x^g`.
    Type0,
    /// `y = (a x + b)^g` where `a x + b >= 0`, else 0.
    Type1,
    /// `y = (a x + b)^g + c` where `a x + b >= 0`, else `c`.
    Type2,
    /// `y = (a x + b)^g` where `x >= d`, else `c x`.
    Type3,
    /// `y = (a x + b)^g + e` where `x >= d`, else `c x + f`.
    Type4,
}

impl TryFrom<u32> for Form {
    type Error = Dropped;

    fn try_from(form: u32) -> Result<Self, Dropped> {
        Ok(match form {
            RGFX_CURVE_BYPASS => Form::Bypass,
            RGFX_CURVE_TYPE0 => Form::Type0,
            RGFX_CURVE_TYPE1 => Form::Type1,
            RGFX_CURVE_TYPE2 => Form::Type2,
            RGFX_CURVE_TYPE3 => Form::Type3,
            RGFX_CURVE_TYPE4 => Form::Type4,
            _ => return Err(Dropped::Form),
        })
    }
}

/// One channel's curve: its form and its parameters `g`, `a`, `b`, `c`, `d`, `e` and `f` (0 where
/// the form does not read them).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Curve {
    form: Form,
    params: [f64; 7],
}

impl Curve {
    /// The parameters each form reads, by index in `g, a, b, c, d, e, f`.
    fn reads(form: Form) -> &'static [usize] {
        match form {
            Form::Bypass => &[],
            Form::Type0 => &[0],
            Form::Type1 => &[0, 1, 2],
            Form::Type2 => &[0, 1, 2, 3],
            Form::Type3 => &[0, 1, 2, 3, 4],
            Form::Type4 => &[0, 1, 2, 3, 4, 5, 6],
        }
    }

    /// The curve `wire` gives.
    fn read(wire: &RgfxCursorCurve) -> Result<Curve, Dropped> {
        let form = Form::try_from(wire.form)?;
        let mut params = [0.0; 7];
        for &i in Curve::reads(form) {
            let value = f32::from_bits(wire.params[i]);
            if !value.is_finite() {
                return Err(Dropped::NotFinite);
            }
            params[i] = f64::from(value);
        }
        Ok(Curve { form, params })
    }

    /// The curve at `x`.
    fn at(&self, x: f64) -> f64 {
        let [g, a, b, c, d, e, f] = self.params;
        // A power of a base at or below 0 is 0, as the kext's (`rgfxQ32Pow`).
        let pow = |base: f64| if base > 0.0 { base.powf(g) } else { 0.0 };
        match self.form {
            Form::Bypass => x,
            Form::Type0 => pow(x),
            Form::Type1 => pow(a * x + b),
            Form::Type2 => {
                let base = a * x + b;
                if base >= 0.0 { pow(base) + c } else { c }
            }
            Form::Type3 => {
                if x >= d {
                    pow(a * x + b)
                } else {
                    c * x
                }
            }
            Form::Type4 => {
                if x >= d {
                    pow(a * x + b) + e
                } else {
                    c * x + f
                }
            }
        }
    }
}

/// A display's cursor's colour chain, as a `CURSOR_COLOR` that turns it on sets it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CursorColor {
    /// The source's curves, red, green and blue.
    source: [Curve; 3],
    /// The blend space's matrix from XYZ times the source's to XYZ, row by row.
    matrix: [[f64; 3]; 3],
    /// The blend space's curves, red, green and blue.
    blend: [Curve; 3],
}

/// The matrix of `rows`, nine `f32`s as bits, row by row.
fn matrix(rows: &[u32; 9]) -> Result<[[f64; 3]; 3], Dropped> {
    let mut m = [[0.0; 3]; 3];
    for (i, bits) in rows.iter().enumerate() {
        let value = f32::from_bits(*bits);
        if !value.is_finite() {
            return Err(Dropped::NotFinite);
        }
        m[i / 3][i % 3] = f64::from(value);
    }
    Ok(m)
}

/// The product `a * b`.
fn product(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            *entry = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

/// The identity, row by row.
const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl CursorColor {
    /// The chain `command` sets: `None` where it turns the chain off (`on` 0), or where the chain
    /// leaves every pixel as it is (the module doc); refused where a curve's form is unknown or a
    /// value it reads is not finite, in which case the cursor keeps its chain.
    pub(crate) fn read(command: &RgfxCursorColor) -> Result<Option<CursorColor>, Dropped> {
        if command.on == 0 {
            return Ok(None);
        }
        let curves = |wire: &[RgfxCursorCurve; 3]| -> Result<[Curve; 3], Dropped> {
            Ok([
                Curve::read(&wire[0])?,
                Curve::read(&wire[1])?,
                Curve::read(&wire[2])?,
            ])
        };
        let source = curves(&command.source_curves)?;
        let blend = curves(&command.blend_curves)?;
        let matrix = product(&matrix(&command.from_xyz)?, &matrix(&command.to_xyz)?);
        let bypass = |c: &[Curve; 3]| c.iter().all(|c| c.form == Form::Bypass);
        if bypass(&source) && bypass(&blend) && matrix == IDENTITY {
            return Ok(None);
        }
        Ok(Some(CursorColor {
            source,
            matrix,
            blend,
        }))
    }

    /// The straight colour `rgb` converted (the module doc).
    pub(crate) fn pixel(&self, rgb: [u8; 3]) -> [u8; 3] {
        let linear: [f64; 3] =
            std::array::from_fn(|i| self.source[i].at(f64::from(rgb[i]) / 255.0));
        std::array::from_fn(|i| {
            let row = self.matrix[i];
            let mixed = row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2];
            step(self.blend[i].at(mixed))
        })
    }

    /// Converts straight-alpha RGBA `pixels` in place, each colour once (the module doc); a
    /// pixel's alpha is kept, and a transparent pixel left as it is.
    pub(crate) fn convert(&self, pixels: &mut [u8]) {
        let mut done: HashMap<[u8; 3], [u8; 3]> = HashMap::new();
        for [red, green, blue, alpha] in pixels.as_chunks_mut::<4>().0 {
            if *alpha == 0 {
                continue;
            }
            let rgb = [*red, *green, *blue];
            let [r, g, b] = *done.entry(rgb).or_insert_with(|| self.pixel(rgb));
            (*red, *green, *blue) = (r, g, b);
        }
    }
}

/// `value` clamped to [0, 1], as the nearest of 256 steps; 0 for a NaN.
fn step(value: f64) -> u8 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    if value >= 1.0 {
        return u8::MAX;
    }
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a value in [0, 255.5) rounds to a step that u8 holds"
    )]
    let step = (value * 255.0 + 0.5) as u8;
    step
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::RgfxMsgHeader;

    /// The sRGB curve, to linear, as an arm64 WindowServer gives its cursor's `Src2XYZ` curve
    /// (27.0.1, measured: iccmob's profile-switch capture).
    const SRGB_TO_LINEAR: (u32, [f32; 7]) = (
        RGFX_CURVE_TYPE3,
        [2.4, 0.947_867, 0.052_132_7, 0.077_399_4, 0.040_45, 0.0, 0.0],
    );
    /// The sRGB curve, from linear, as WindowServer gives the `XYZ2Blend` curve of a blend space
    /// of sRGB's transfer: the display's profile, Display P3.
    const LINEAR_TO_SRGB: (u32, [f32; 7]) = (
        RGFX_CURVE_TYPE4,
        [0.416_667, 1.137_12, 0.0, 12.92, 0.003_130_8, -0.055, 0.0],
    );
    /// Generic RGB's curve from linear, as WindowServer gives it (gamma 1.8, linear below).
    const LINEAR_TO_GENERIC: (u32, [f32; 7]) = (
        RGFX_CURVE_TYPE3,
        [0.555_556, 1.0, 0.0, 16.0, 0.001_953_12, 0.0, 0.0],
    );
    /// Standard sRGB to XYZ (D50), the cursor's `Src2XYZ` matrix.
    const SRGB_TO_XYZ: [f32; 9] = [
        0.43603,
        0.385_115,
        0.143_055,
        0.222_495,
        0.716_895,
        0.060_609_8,
        0.013_912_8,
        0.097_054_2,
        0.713_933,
    ];
    /// XYZ to Display P3: the cursor's `XYZ2Blend` once the display's profile is Display P3.
    const XYZ_TO_P3: [f32; 9] = [
        2.40398,
        -0.989_911,
        -0.397_637,
        -0.842_246,
        1.79888,
        0.016_023_5,
        0.048_221_1,
        -0.097_425_7,
        1.27401,
    ];
    /// XYZ to the display's own profile, from the primaries and white the device publishes:
    /// the cursor's `XYZ2Blend` at login.
    const XYZ_TO_DISPLAY: [f32; 9] = [
        3.12482,
        -1.60934,
        -0.489_281,
        -0.979_213,
        1.91654,
        0.033_485,
        0.071_869_4,
        -0.228_997,
        1.40587,
    ];
    /// XYZ to Generic RGB: the cursor's `XYZ2Blend` once the profile is Generic RGB.
    const XYZ_TO_GENERIC: [f32; 9] = [
        3.0287,
        -1.52389,
        -0.480_522,
        -1.09713,
        2.06033,
        -0.003_010_49,
        0.075_510_7,
        -0.22798,
        1.40038,
    ];

    fn curve((form, params): (u32, [f32; 7])) -> RgfxCursorCurve {
        RgfxCursorCurve {
            form,
            params: params.map(f32::to_bits),
        }
    }

    /// A `CURSOR_COLOR` that turns the chain on, of the source's and the blend space's curves
    /// and matrices.
    fn command(
        source: (u32, [f32; 7]),
        to_xyz: [f32; 9],
        from_xyz: [f32; 9],
        blend: (u32, [f32; 7]),
    ) -> RgfxCursorColor {
        RgfxCursorColor {
            header: RgfxMsgHeader::default(),
            display: 0,
            on: 1,
            source_curves: [curve(source); 3],
            to_xyz: to_xyz.map(f32::to_bits),
            from_xyz: from_xyz.map(f32::to_bits),
            blend_curves: [curve(blend); 3],
        }
    }

    /// The chain from sRGB to `from_xyz`'s space and `blend`'s curve.
    fn to(from_xyz: [f32; 9], blend: (u32, [f32; 7])) -> CursorColor {
        CursorColor::read(&command(SRGB_TO_LINEAR, SRGB_TO_XYZ, from_xyz, blend))
            .unwrap()
            .unwrap()
    }

    /// A cursor converted to Display P3 shows sRGB's saturated colours at their P3 values (red
    /// at (234, 51, 35), as a colour-managed Mac shows it), and black and white exactly
    #[test]
    fn a_p3_chain_converts_saturated_colours_and_keeps_black_and_white() {
        let p3 = to(XYZ_TO_P3, LINEAR_TO_SRGB);
        assert_eq!(p3.pixel([255, 0, 0]), [234, 51, 35]);
        assert_eq!(p3.pixel([0, 255, 0]), [117, 251, 76]);
        assert_eq!(p3.pixel([0, 0, 255]), [0, 0, 245]);
        assert_eq!(p3.pixel([0, 0, 0]), [0, 0, 0]);
        assert_eq!(p3.pixel([255, 255, 255]), [255, 255, 255]);
        // A grey stays grey, within a step: the two spaces share their white.
        for v in [1, 64, 128, 200, 254] {
            for c in p3.pixel([v, v, v]) {
                assert!(c.abs_diff(v) <= 1, "grey {v} became {c}");
            }
        }
    }

    /// The chains measured at login (the display's own profile, 0.0024 off sRGB's) and after a
    /// switch to Generic RGB keep black and white exactly too, and Generic RGB's gamma darkens
    /// mid-grey to its own encoding
    #[test]
    fn the_measured_chains_keep_black_and_white() {
        let display = to(XYZ_TO_DISPLAY, LINEAR_TO_SRGB);
        let generic = to(XYZ_TO_GENERIC, LINEAR_TO_GENERIC);
        for chain in [&display, &generic] {
            assert_eq!(chain.pixel([0, 0, 0]), [0, 0, 0]);
            assert_eq!(chain.pixel([255, 255, 255]), [255, 255, 255]);
        }
        // At login the matrices' product is 0.0024 off the identity at most, which sRGB's curve
        // makes up to 7 steps near black: green gains 7 steps of red and blue 1.
        assert_eq!(display.pixel([255, 0, 0]), [255, 0, 0]);
        assert_eq!(display.pixel([0, 255, 0]), [7, 255, 0]);
        assert_eq!(display.pixel([0, 0, 255]), [1, 0, 255]);
        assert_eq!(display.pixel([128, 64, 32]), [128, 64, 32]);
        // sRGB's mid-grey, 0.214 linear, is Generic RGB's 0.425.
        assert_eq!(generic.pixel([128, 128, 128]), [109, 109, 109]);
    }

    /// Converting an image keeps each pixel's alpha, leaves transparent pixels as they are, and
    /// converts each colour as `pixel` does
    #[test]
    fn converting_an_image_keeps_its_alpha() {
        let p3 = to(XYZ_TO_P3, LINEAR_TO_SRGB);
        let mut image = vec![
            255, 0, 0, 255, // opaque red
            255, 0, 0, 128, // half red
            9, 9, 9, 0, // transparent
            255, 255, 255, 200, // white
        ];
        p3.convert(&mut image);
        assert_eq!(
            image,
            [
                234, 51, 35, 255, 234, 51, 35, 128, 9, 9, 9, 0, 255, 255, 255, 200
            ]
        );
    }

    /// A chain turned off, and one that leaves every pixel as it is (curves of no form and
    /// matrices whose product is the identity), are no chain: the cursor shows its pixels as
    /// they are, at no cost
    #[test]
    fn an_identity_chain_is_no_chain() {
        const ONE: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        const BYPASS: (u32, [f32; 7]) = (RGFX_CURVE_BYPASS, [f32::NAN; 7]);
        let mut off = command(SRGB_TO_LINEAR, SRGB_TO_XYZ, XYZ_TO_P3, LINEAR_TO_SRGB);
        off.on = 0;
        assert_eq!(CursorColor::read(&off), Ok(None));
        assert_eq!(
            CursorColor::read(&command(BYPASS, ONE, ONE, BYPASS)),
            Ok(None)
        );
        let doubled = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0];
        let halved = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5];
        assert_eq!(
            CursorColor::read(&command(BYPASS, doubled, halved, BYPASS)),
            Ok(None)
        );
        assert!(matches!(
            CursorColor::read(&command(BYPASS, doubled, ONE, BYPASS)),
            Ok(Some(_))
        ));
    }

    /// A curve of a form there is none of, and a value a form reads or a matrix's entry that is
    /// not finite, drop the message; a parameter the form does not read is not looked at
    #[test]
    fn an_unknown_form_or_a_value_not_finite_is_dropped() {
        let mut unknown = command(SRGB_TO_LINEAR, SRGB_TO_XYZ, XYZ_TO_P3, LINEAR_TO_SRGB);
        unknown.blend_curves[2].form = 6;
        assert_eq!(CursorColor::read(&unknown), Err(Dropped::Form));
        let mut nan = command(SRGB_TO_LINEAR, SRGB_TO_XYZ, XYZ_TO_P3, LINEAR_TO_SRGB);
        nan.source_curves[1].params[4] = f32::NAN.to_bits();
        assert_eq!(CursorColor::read(&nan), Err(Dropped::NotFinite));
        let mut infinite = command(SRGB_TO_LINEAR, SRGB_TO_XYZ, XYZ_TO_P3, LINEAR_TO_SRGB);
        infinite.to_xyz[8] = f32::INFINITY.to_bits();
        assert_eq!(CursorColor::read(&infinite), Err(Dropped::NotFinite));
        // Type3 reads no `e` or `f`.
        let mut unread = command(SRGB_TO_LINEAR, SRGB_TO_XYZ, XYZ_TO_P3, LINEAR_TO_SRGB);
        unread.source_curves[0].params[5] = f32::NAN.to_bits();
        assert!(matches!(CursorColor::read(&unread), Ok(Some(_))));
    }

    /// Each form evaluates as QuartzCore's, a power of a base at or below 0 being 0
    #[test]
    fn each_form_evaluates_as_quartzcore_s() {
        let at = |form: Form, params: [f64; 7], x: f64| Curve { form, params }.at(x);
        let p = [2.0, 2.0, -0.5, 0.25, 0.5, 0.125, 0.0625];
        assert_eq!(at(Form::Bypass, p, 0.3), 0.3);
        assert_eq!(at(Form::Type0, p, 0.5), 0.25);
        assert_eq!(at(Form::Type0, p, -0.5), 0.0);
        assert_eq!(at(Form::Type1, p, 0.75), 1.0);
        assert_eq!(at(Form::Type1, p, 0.1), 0.0);
        assert_eq!(at(Form::Type2, p, 0.75), 1.25);
        assert_eq!(at(Form::Type2, p, 0.1), 0.25);
        assert_eq!(at(Form::Type3, p, 0.75), 1.0);
        assert_eq!(at(Form::Type3, p, 0.25), 0.0625);
        assert_eq!(at(Form::Type4, p, 0.75), 1.125);
        assert_eq!(at(Form::Type4, p, 0.25), 0.125);
    }
}
