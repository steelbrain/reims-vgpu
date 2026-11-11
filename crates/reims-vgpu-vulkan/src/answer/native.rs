//! Where a degradation of [`DIES`](super::DIES) is placed on a device that does not run on Metal.
//!
//! **The device class.** A device the runtime makes over MoltenVK runs on Metal: the kinds of
//! [`DIES`](super::DIES), [`HEARD`](super::HEARD) and `RELAYED_ONCE` were placed on it, measured
//! against the Mac GPUs, and stay as they are there. On macOS every device runs on Metal, whatever
//! its driver ([`DRIVERS_RUN_ON_METAL`]): the runtime loads Vulkan through the system's loader
//! there, which may list Mesa's KosmicKrisp, Vulkan on Metal too, ahead of MoltenVK, and where a
//! run dies on macOS is the contract's. Elsewhere any driver other than MoltenVK is a native
//! device: there a kind of [`NATIVE_RELAYED`] is relayed, counted, in place of dying. The class
//! is read when a device is adopted ([`device_made`], from the runtime's
//! `DeviceDescription::driver_id`, the fact the runtime itself tells MoltenVK by), and the
//! process's placement is native only while every device it adopted is native: before its first
//! device, and once any device runs on Metal or names no driver, every kind of
//! [`DIES`](super::DIES) dies, as before. The runtime's notices name no device, so the placement
//! is the process's.
//!
//! **A native relay** writes the runtime's line for the kind's first occurrence, as a relayed
//! kind's, and counts each occurrence the runtime announces ([`counts`]): every occurrence of
//! each kind of [`DIES`](super::DIES) in strict and census mode, and elsewhere its first on each
//! device (`announce_each`, `crate::platform`'s `announced_each`). Each kind stays on the list of
//! differences to bring to Metal's answer; the count is how often a run met it, as announced.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use metal2vulkan::vk::DriverId;
use metal2vulkan_runtime as rt;

/// One kind of [`DIES`](super::DIES) a native device relays, and why.
#[derive(Debug, Clone, Copy)]
pub struct NativeRelay {
    /// The kind.
    pub kind: rt::Degradation,
    /// Why it is relayed rather than a death on a native device.
    pub ground: &'static str,
}

/// The kinds of [`DIES`](super::DIES) a native device relays, each with its ground.
///
/// The first eight are relayed on the rule for native devices: as close to the Mac GPUs as the
/// hardware allows; a difference the hardware itself forces is logged as a typed
/// degradation, relayed once per kind and counted, with no interpreter-level fallback for
/// differences in the tenth decimal. Each is here on a fact of the device that its ground states:
/// the device's own interpolation, level arithmetic, reduction order, float controls, evaluation
/// position, reconvergence, memory granularity or line rasterizer's tie. The rest are the device's
/// maths and sampling, as a class of their own (below). "Not built", "not measured" and "a
/// guest or a suite case reaches it" are no grounds: a kind neither covers, whose exact form the
/// hardware allows or whose rule on Metal is not read, dies here as on MoltenVK, at the same op
/// and naming the kind, until the runtime builds that form or it is measured equal to Metal's
/// answer. Each relayed kind stays a difference to bring to Metal's answer, its magnitude for the
/// runtime team to confirm; one found visible is fixed toward Metal's answer, not relayed.
///
/// `DeviceInterpolation` and `GradFromBias` differ in texel values
/// alone, never in control flow, memory addresses or numbering. Their counts are a census-mode run
/// of 2026-10-10 on an RTX 5080 Laptop GPU (macOS 13, 15 and
/// 26 booted to an idle desktop, and macOS 14 and 12 through the built-in app sweep, with no
/// difference seen on screen): how often a run met each, not why it is relayed. A heap texture's
/// size the device's memory granularity
/// decides (`HeapTextureLargerThanMetal`) differs in the size query's answer, which reaches the
/// guest: a heap sized by it is larger, so the buffers and heaps made after it can lie at other
/// gpuAddresses than on Metal. It never differs in texels, nor in the range of Metal's space a
/// texture of its own memory takes.
///
/// Of those eight, the four after `HeapTextureLargerThanMetal` are the precision notes of a
/// pipeline's stage whose arithmetic or ordering can give other bits than Metal's, as the
/// runtime's documentation of each says: a note whose difference is the device's rounding,
/// evaluation position or reduction order, or the lanes its driver runs together after
/// divergence, is logged, not a death. The eighth, `DeviceLineTies`, is the pixel the device's
/// line rasterizer covers where a width-1 line runs through a vertex two pixels' diamonds share,
/// which Vulkan leaves to the driver with no control over it: it differs in which pixels a line
/// covers, never in control flow, memory addresses or numbering, and its only exact form is
/// Metal's line rule run in the fragment function, an interpreter's, as a depth test run there
/// is for `DepthDenormals`.
///
/// The rest are relayed as the class of the device's own maths and sampling: on a GPU that is not
/// Apple's, the device's maths and sampling are the default, their bounds recorded and announced,
/// and Metal's exact form is opt-in, which the runtime builds. Each is the device's arithmetic, its
/// maths or its sampling and filtering, and its ground states its bound where one is measured,
/// else that the runtime records it: `F16FmaDoubleRounding` and `F16IntConvDoubleRounding`,
/// `DeviceArithmetic`, `DeviceDenormals` and `DeviceNanEncoding` (maths, the last a NaN's bits,
/// whose native form the runtime takes and builds no exact form beside); `FloatSampleFilter`,
/// `HalfSampleFilter`, `CubeGradientFootprint`, `CompressedTexelDecode`, `ViewBaseTexelOffset`,
/// `SrgbTexelDecode`, `BicubicFilterArithmetic`, `HalfTexelConversion` and `DeviceSampleLevel`
/// (sampling). Then the
/// notes of the device's own answers off Apple's GPUs, which the runtime gives in place of Metal's,
/// each with the bound the runtime records: `NativeTranscendentals`, `NativeDivision`, `NativeFma`,
/// `NativeNanEncoding`, `NativeMinMax`, `NativeDot`, `NativeMix`, `NativeHalfConversion` and
/// `NativeHalfQuantize` (maths); `DeviceSampling`, `DeviceTexelConversion` and
/// `DecodedTexelPrecision` (sampling); `DeviceRoundedResolve` (the device's rounding of a resolve's
/// average); `DevicePackConversion` (its conversions of packed normalized values),
/// `DeviceVertexConversion` (its vertex input's conversion of a normalized attribute),
/// `DeviceAlphaCoverage` (its alpha to coverage) and `NativeLogicOperation` (its logic operation on
/// a 10-bit or signed normalized attachment's own codes). A device that runs on Metal (MoltenVK) is
/// an Apple GPU, where each still dies.
///
/// These die here, as on MoltenVK, as no class covers them: blending, storage, tessellation,
/// depth, layout, the band's rasterization, a pass's tile store and when a command buffer is seen
/// complete are neither maths nor sampling.
/// - those whose exact form the hardware allows, built or not: `BlendedWriteRounding` (a blended
///   float write to a half-float channel under `textureWriteRoundingMode` 2, whose exact form is
///   the blend in the function under the pixel interlock) and `RotatedBlendRounding` (a blend
///   into an `ARGB8Unorm` attachment in none of the forms measured to blend as Metal's, whose
///   exact form is a measured one or the blend in the function), and `DeviceSubpixelSnap` (the
///   band's corners, which the device's rasterizer snaps where Metal snaps a line's endpoints
///   first, not a fact of the device: the band is the runtime's own, and its endpoints snapped to
///   Metal's grid before it offsets them leave the rasterizer nothing to round, as the band's
///   depth bias, `WideEdgeDepthBias`, and the device's own wide lines, `DeviceWideLines`, die
///   here too);
/// - `AnisotropicFilter`, Metal's anisotropic filter emulated by the device's samples at
///   explicit levels: its level and its tap spacing past two taps
///   are a model fitted to Metal's answers, not Metal's rule, and each tap is the device's own
///   filter. Measured on `rgba8Unorm` alone, its linear filters lie within 0.68 of a code of an
///   Apple M4's on an RTX 3090, but its nearest filters up to 24.5 codes off. The class of the
///   device's maths and sampling does not name it;
/// - `TessellationFactorScale`, whose rule on Metal is not read (whether Metal's fractional
///   partitions take a scaled factor in `float` or rounded up to `half`): tessellation, not maths;
/// - four precision notes no fact of the device decides. `TessellationFactorInstance` (the
///   factors of a draw with a base instance past 0 read at an offset counted from it, where
///   Metal's count is not read) and `Fp8Value` (an 8-bit float's integer conversion from the
///   format's value, where Metal on an Apple M3 answers the carrier byte's integer) are other
///   values, of another meaning, not other bits of the same arithmetic.
///   `QuadsCountedBeforeDiscard` (a simdgroup operation other than a quad one before a discard,
///   in a shape where Apple's compiler's placement of it is not known, whose lanes include the
///   quads the discard drops) differs from Metal's where Apple's compiler moves the operation past
///   the discard: where that compiler places it decides the answer, not the device. So does
///   `UnmeasuredGroupSpeculation` (a quad or simdgroup operation in a branch arm of a shape no
///   probe says Metal's backend speculates, kept over the AIR's lanes): where that backend
///   speculates decides the answer;
/// - `AttachmentReadInPlace`, a draw's access to an attachment's memory inside its render pass
///   instance other than as an attachment, where the runtime serves no copy or snapshot: its
///   exact form, the copy or snapshot a sampled slot's binding reads, the hardware allows.
pub const NATIVE_RELAYED: [NativeRelay; 39] = [
    NativeRelay {
        kind: rt::Degradation::DeviceInterpolation,
        ground: "a fragment function's interpolated input, on every device but MoltenVK, where \
                 Metal interpolates (the runtime's note): Vulkan states no precision for \
                 interpolation, and against an M3 Max under macOS 27.0.1, 31% to 45% of the \
                 lanes RADV, ANV, lavapipe and NVIDIA's driver interpolated were Metal's bit for \
                 bit, the rest up to 1.2e-4 apart on a Radeon 780M, 7.6e-5 on an Iris Xe, \
                 1.5e-4 on an RTX 5080 and 3090, and 3e-4 on lavapipe, which snaps vertices \
                 otherwise. Announced at the first draw of any interpolated fragment function, \
                 so every guest's first present died of it",
    },
    NativeRelay {
        kind: rt::Degradation::GradFromBias,
        ground: "a biased sample emulated with gradients: the device's level arithmetic and \
                 `exp2` may pick another level or anisotropic footprint than Metal's (the \
                 runtime's note). Census: 1220, 175 and 102 occurrences on macOS 13, 15 and 26",
    },
    NativeRelay {
        kind: rt::Degradation::HeapTextureLargerThanMetal,
        ground: "a heap texture's size and alignment are the device's image's, coarser than \
                 Metal's: NVIDIA's bufferImageGranularity is 1024 bytes, so a 4x4 `Depth32Float`, \
                 `Depth16Unorm` or `Stencil8` texture answers 1024 bytes (0x400) at an alignment \
                 of 1024, where Metal answers 128 (0x80) at 128 (the runtime's measurement). Each \
                 texture keeps its texels, but the difference is not heap space alone: the size \
                 query's answer itself reaches the guest. A guest that sizes a heap by it makes a \
                 larger heap than on Metal, which takes a larger range of Metal's space where its \
                 size crosses one of the space's 32-KiB steps, so the buffers and heaps made after \
                 it sit at other gpuAddresses than on Metal, and the textures it places by the \
                 answers lie at other offsets. A texture of its own memory takes a range of \
                 Metal's space of the runtime's model of Metal's answer, Metal's",
    },
    NativeRelay {
        kind: rt::Degradation::ReductionOrder,
        ground: "a float reduction associates its operands in the device's order, where Metal's \
                 associates them in its own (the runtime's U3 note): a sum's last bits can \
                 differ",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceSignedZeroInfNan,
        ground: "float operations whose signed zeros, infinities and NaNs Metal's compiler keeps \
                 run without `SignedZeroInfNanPreserve`, which the device does not support at \
                 their width (the runtime's U3 and D57 note): the device's compiler may answer \
                 such an operand's result otherwise",
    },
    NativeRelay {
        kind: rt::Degradation::MultisamplePosition,
        ground: "a multisampled pipeline's fragment function reads at pixel rate a value Metal \
                 evaluates at the pixel's centre, which the device evaluates at a position \
                 within the fragment Vulkan leaves to the implementation (the runtime's D212 \
                 note): `[[position]]`'s z or w, a centre-sampled or centroid varying, \
                 `[[point_coord]]` or `[[barycentric_coord]]`; or one that runs per sample \
                 under sample shading reads one of them but a centre-sampled varying, which \
                 Vulkan evaluates for its sample where Metal reads it as the pixel's; each \
                 within its change over half the pixel's diagonal",
    },
    NativeRelay {
        kind: rt::Degradation::ImplementationDefinedReconvergence,
        ground: "a stage that reads simdgroup state after divergent control flow, on a device \
                 whose reconvergence is implementation-defined where Metal's is maximal: its \
                 simdgroup operations there can see other lanes active than Metal's (the \
                 runtime's note). A driver without `VK_KHR_shader_maximal_reconvergence` decides \
                 which lanes run together after divergence, so other sets of lanes than Metal's \
                 can be active together: a ballot or a sum can cover another set of lanes, and a \
                 `simd_is_first` election can pick several lanes. A fragment stage with \
                 simdgroup operations still dies, of `FragmentSimdgroupGrouping`, which the \
                 runtime records on every driver but MoltenVK",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceLineTies,
        ground: "a draw of lines, or of a triangle's edges in fill mode `Lines`, by the device's \
                 own lines at width 1, on a driver not measured to break Metal's tie (the \
                 runtime's note, `DriverFact::LineTiesAreMetals`): a line through a vertex two \
                 pixels' diamonds share exits neither diamond, and Vulkan's open diamond leaves \
                 the pixel to the driver, with no control over it, where Metal covers the lower \
                 one. Measured by the runtime: NVIDIA 615.71.09 on an RTX 5080 Laptop GPU covers \
                 the other pixel (a strip's diagonal a row higher), and lavapipe (Mesa 26.2.4) \
                 leaves out a strip's first pixel where it starts on such a vertex; RADV and ANV \
                 are not measured. Only the line's rule run in the fragment function, an \
                 interpreter's, is exact. Announced at every draw of lines at width 1 there, so \
                 a strict run would die at its first",
    },
    NativeRelay {
        kind: rt::Degradation::F16FmaDoubleRounding,
        ground: "class (e), owner 2026-10-11: the device's maths. A simdgroup matrix's `half` \
                 multiply-add steps computed in `float` round twice where Metal rounds once, at \
                 most one `half` unit apart at a rounding midpoint; the runtime records the \
                 device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::F16IntConvDoubleRounding,
        ground: "class (e), owner 2026-10-11: the device's maths. A `half` and integer \
                 conversion rounding twice; the runtime records the device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceArithmetic,
        ground: "class (e), owner 2026-10-11: the device's maths. A `float` division, \
                 remainder, square root, fused multiply-add or dot product at Vulkan's precision \
                 bounds (a division within 2.5 ulp, an `Fma` that may run unfused), where Metal \
                 answers the correctly rounded value; the runtime records the device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceDenormals,
        ground: "class (e), owner 2026-10-11: the device's maths. An `f32` denormal the device \
                 keeps or flushes by its own float controls, where Metal reads it as the zero of \
                 its sign; the runtime records the device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceNanEncoding,
        ground: "class (e): the device's maths, for a NaN's bits. The NaN the device's \
                 arithmetic, conversions, clears, copies and texel decodes give, its sign, \
                 payload and quietness the device's, where Metal answers its canonical NaN, a \
                 texel's own or a `half`'s own bits: a `half` colour output widened by the \
                 device's `OpFConvert` stores every NaN as `0x7fff` on NVIDIA 615.71.09 (an RTX \
                 5080 Laptop GPU, 8180 of 262144 channels of the runtime's lane, each a NaN) and \
                 as `0xfe00` on ANV and RADV. Every value but a NaN is unchanged. The runtime \
                 gives the device's NaN, its native form, and builds no path to Metal's bits \
                 where the device has one (its rule)",
    },
    NativeRelay {
        kind: rt::Degradation::FloatSampleFilter,
        ground: "class (e), owner 2026-10-11: the device's sampling. A filtered sample with a \
                 `float` result, whose footprint, weights and sum Vulkan bounds only by its \
                 precision rules: an RTX 3090's blend between mip levels lies 1.03 codes off \
                 Metal's at 8 bits (the runtime's D832); the runtime records the device's bound \
                 for the other formats and filters",
    },
    NativeRelay {
        kind: rt::Degradation::HalfSampleFilter,
        ground: "class (e), owner 2026-10-11: the device's sampling. A filtered sample with a \
                 `half` result, bounded as `float-sample-filter`'s: an RTX 3090's blend between \
                 mip levels lies 1.03 codes off Metal's at 8 bits (the runtime's D832); the \
                 runtime records the device's bound for the other formats and filters",
    },
    NativeRelay {
        kind: rt::Degradation::CubeGradientFootprint,
        ground: "class (e), owner 2026-10-11: the device's sampling. A cube sample's level and \
                 anisotropic filter from the gradients the runtime hands the device, Metal's \
                 footprint under macOS 27 (the face's two tangent components, an isotropic \
                 sampler's combined by Metal's length rule): the device takes its own logarithm \
                 of the footprint, where Metal's piecewise-linear one minifies only past \
                 1 + 3/2048 and blends levels in 256ths, and filters an anisotropic footprint \
                 with its own taps and weights; an implicit-level cube sample's level is the \
                 device's, where Metal's rule is not measured. The runtime records the device's \
                 bound",
    },
    NativeRelay {
        kind: rt::Degradation::CompressedTexelDecode,
        ground: "class (e), owner 2026-10-11: the device's sampling. A BC1 to BC5 texture's \
                 blocks as the device decodes them, where Metal interpolates with 6-bit weights: \
                 NVIDIA 615.71.09 up to 3/255 off for BC1 to BC3 and 7.3/255 for BC4 and BC5 \
                 (the runtime's D796); the runtime records other devices' bounds",
    },
    NativeRelay {
        kind: rt::Degradation::ViewBaseTexelOffset,
        ground: "class (e), owner 2026-10-11: the device's sampling. A constant texel offset on \
                 a repeating or mirrored axis of an argument buffer's view, placed by the view's \
                 own base level where Metal places it by its parent's: one texel or footprint \
                 away on a view whose first level lies above its parent's base; the runtime \
                 records the device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::SrgbTexelDecode,
        ground: "class (e), owner 2026-10-11: the device's sampling. An 8-bit sRGB texel read \
                 through the device's decode, where Metal decodes through its twelve-bit table; \
                 the runtime records the device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::BicubicFilterArithmetic,
        ground: "class (e), owner 2026-10-11: the device's sampling. Metal's bicubic filter \
                 emulated by fetches, whose sums differ from Metal's where its rule is not \
                 pinned (one grid step apart near a rounding tie); the runtime records the \
                 device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::HalfTexelConversion,
        ground: "class (e), owner 2026-10-11: the device's sampling. A `half` result of a \
                 texel the translation converts from the device's `f32` texel to nearest even, \
                 where Metal rounds a 32-bit float texel toward zero and a normalized one onto \
                 its grid first: one `half` unit apart; the runtime records the device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceSampleLevel,
        ground: "class (e), owner 2026-10-11: the device's sampling. A sample whose level the \
                 device's level arithmetic chooses, an LOD query's among them: Vulkan's level of detail may \
                 approximate its scale factor, and a nearest mip filter reads ceil(lambda + 1/2) \
                 - 1 where Metal rounds the level to half and truncates its fraction onto \
                 sixty-fourths, so near a boundary the sample may read another level or filter \
                 than Metal's; the runtime records the device's bound",
    },
    NativeRelay {
        kind: rt::Degradation::NativeTranscendentals,
        ground: "class (e), owner 2026-10-11: the device's maths, the device's own answer \
                 off Apple's GPUs. A math-library call the device's \
                 `GLSL.std.450` instruction answers, within Vulkan's precision table: `Exp2` and \
                 `Exp` 3 + 2|x| ulp, `Log2` and `Log` 3 ulp (2^-21 absolute in [0.5, 2]), \
                 `InverseSqrt` 2 ulp, `Sin` and `Cos` 2^-11 absolute in [-pi, pi], the inverse \
                 trigonometric functions 4096 ulp (the runtime's D907)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeDivision,
        ground: "class (e), owner 2026-10-11: the device's maths, the runtime's default off \
                 Apple's GPUs. A precise or fast division the device's `OpFDiv` answers: within \
                 2.5 ulp of the quotient, a `half` one within one unit in its last place (the \
                 runtime's D907)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeFma,
        ground: "class (e), owner 2026-10-11: the device's maths, the runtime's default off \
                 Apple's GPUs. A `half` fused multiply-add, or a `float` one without \
                 `OpFmaKHR`, the device's `Fma` answers: a `half` within one unit in its last \
                 place of Metal's, a `float` within half an ulp of the product plus half an ulp \
                 of the result (the runtime's D907)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeNanEncoding,
        ground: "class (e), owner 2026-10-11: the device's maths, the runtime's default off \
                 Apple's GPUs. An operation's NaN result in the device's sign and payload where \
                 Metal selects its canonical NaN or an operand's bits; every other result \
                 unchanged (the runtime's D914)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeMinMax,
        ground: "class (e), owner 2026-10-11: the device's maths, the runtime's default off \
                 Apple's GPUs. `fmin`, `fmax`, `min3`, `max3`, `clamp` and `saturate` as the \
                 device's `NMin`, `NMax` or `NClamp`: an operand, Metal's but at three ties (two \
                 NaNs, +0 and -0, an `f32` denormal beside a zero), where Metal answers the \
                 second (the runtime's D914)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeDot,
        ground: "class (e), owner 2026-10-11: the device's maths, the runtime's default off \
                 Apple's GPUs. `dot` as the device's `OpDot`: a `float` one of n lanes within \
                 (3n - 1) 2^-24 of the sum of the products' magnitudes of Metal's fused chain, a \
                 `half` one within (n + 3) 2^-11 of it plus (n + 2) 2^-25 (the runtime's D914)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeMix,
        ground: "class (e), owner 2026-10-11: the device's maths, the runtime's default off \
                 Apple's GPUs. `mix(x, y, t)` as the device's `FMix`: a `float` within 2^-21 \
                 (|x(1 - t)| + |y t|) of Metal's fused form, a `half` within 3 2^-10 of that sum \
                 plus 2^-23 (the runtime's D914)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeHalfConversion,
        ground: "class (e), owner 2026-10-11: the device's maths, the device's own answer \
                 off Apple's GPUs. A `half` result rounded by the \
                 device's own conversion to 16 bits: within one unit in its last place, toward \
                 zero on some devices (65504 from 65520 up where Metal answers infinity), a \
                 result below 2^-14 the zero of its sign where the device flushes, and a NaN's \
                 encoding and a zero's sign the device's (the runtime's D915)",
    },
    NativeRelay {
        kind: rt::Degradation::NativeHalfQuantize,
        ground: "class (e), owner 2026-10-11: the device's maths, the runtime's default off \
                 Apple's GPUs. A `half` result rounded by the device's `OpQuantizeToF16` on a \
                 device without `shaderFloat16`: within one unit in its last place, 65504 to \
                 65520 possibly infinity where Metal answers 65504, and a NaN's encoding the \
                 device's (the runtime's D915)",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceSampling,
        ground: "class (e), owner 2026-10-11: the device's sampling, the device's own answer \
                 off Apple's GPUs. Samples, gathers and comparisons \
                 the device's sampler answers: measured against an Apple M4, a linear filter \
                 within 0.919 of a code of Metal's on NVIDIA 615.71.09, 0.548 on RADV, 0.918 on \
                 ANV and 1.0 on lavapipe, an sRGB texture's within 1.832, 0.764, 0.912 and \
                 0.781, and a nearest sample's texel Metal's neighbour at 0.1% of coordinates \
                 at most (the runtime's D906); a two-plane YCbCr texture's linear filter \
                 within 1.18 codes of Metal's sum on an RTX 3090, \
                 0.88 on RADV and 1.06 on ANV; the runtime records the rest",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceTexelConversion,
        ground: "class (e), owner 2026-10-11: the device's sampling, the device's own answer \
                 off Apple's GPUs. A texel's codes converted by the \
                 device: an 8- or 16-bit normalized code within one `f32` ulp of Metal's, a 5- or \
                 6-bit channel within 0.18 of a code, and a half-float or small-float NaN in the \
                 device's encoding (the runtime's D905)",
    },
    NativeRelay {
        kind: rt::Degradation::DecodedTexelPrecision,
        ground: "class (e): the device's own answer off Apple's GPUs, for a texture of a format the device lacks \
                 held in a carrier, whose reads, samples and gathers read a decoded twin of \
                 compact codes: an ASTC LDR channel the 8-bit code nearest Metal's 12-bit \
                 value, at most 1/510 off it, and an EAC 11-bit channel the 16-bit normalized \
                 code nearest Metal's 11-bit value, at most half a 16-bit step off it (the \
                 runtime's record of the bound)",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceRoundedResolve,
        ground: "class (e): the device's own rounding of an average, the device's own answer \
                 off Apple's GPUs, for a pass's resolve of a multisampled \
                 colour attachment by the device's `AVERAGE`: a resolved channel up to one code \
                 off Metal's (an sRGB one two) near a rounding boundary, a float channel off in \
                 its last bits, and wholly where a sample is a NaN or an infinity, measured at 2 \
                 and 4 samples on RADV, ANV, NVIDIA and lavapipe, each rounding an 8-bit unorm \
                 tie down",
    },
    NativeRelay {
        kind: rt::Degradation::DevicePackConversion,
        ground: "class (e): the device's maths, for a conversion: its `GLSL.std.450` `Pack` and \
                 `Unpack` of a normalized format, and its clamp, multiply, `RoundEven`, \
                 reciprocal and `Pow` for the rest, where Metal rounds the exact product and \
                 takes an sRGB colour through its twelve-bit table: a packed code within one of \
                 Metal's, an unpacked value within 2.5 ulp of the correctly rounded quotient, an \
                 sRGB colour within half a twelve-bit step of Metal's (the runtime's record of \
                 the bound)",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceVertexConversion,
        ground: "class (e): the device's conversion of a normalized vertex attribute's codes by \
                 its fixed-function vertex input, as `DeviceTexelConversion` is its conversion of \
                 a texel's: within one `f32` ulp of Metal's where measured, and lavapipe's \
                 per-instance fetch reading the most negative signed code unclamped (the \
                 runtime's record of the bound)",
    },
    NativeRelay {
        kind: rt::Degradation::DeviceAlphaCoverage,
        ground: "class (e): the device's alpha to coverage, which Vulkan leaves to the device: \
                 its own algorithm where Metal covers by a 2x2 ordered dither, each pixel within \
                 one sample of Metal's on lavapipe, unmeasured elsewhere (the runtime's record). \
                 The runtime gives the device's answer with the note, as for its maths and \
                 sampling, with no exact form to take",
    },
    NativeRelay {
        kind: rt::Degradation::NativeLogicOperation,
        ground: "class (e): a render pipeline's private logic operation drawn by the device's \
                 `logicOp` into an `RGB10A2Unorm`, `BGR10A2Unorm` or signed normalized \
                 attachment, which operates on the channels' stored codes themselves, where Metal \
                 operates on a 10-bit or 2-bit code widened to 16 bits and narrows the result, \
                 and reads and stores a signed channel's lowest code as the code above it, so a \
                 channel may store another code (lavapipe's xor: green 938 where Metal's rule \
                 gives 937, and 0x60 over a signed 0x80 where Metal's gives 0x61; the runtime's \
                 record, with no bound, as the result follows the codes' bits). Every other \
                 format's logic operation is Metal's. The runtime gives the device's operation \
                 off Apple's GPUs, as for its maths and sampling, and draws it in the fragment \
                 function on MoltenVK, which has no `logicOp`",
    },
];

/// Where the process places a kind of [`DIES`](super::DIES).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// As measured against the Mac GPUs: every kind of [`DIES`](super::DIES) dies.
    OnMetal,
    /// On native devices: the kinds of [`NATIVE_RELAYED`] are relayed.
    Native,
}

/// Whether every Vulkan driver of this OS runs on Metal: on macOS, where MoltenVK and Mesa's
/// KosmicKrisp alike run Vulkan on Metal (module docs).
const DRIVERS_RUN_ON_METAL: bool = cfg!(target_os = "macos");

/// The class of a device whose driver is `driver`: on Metal on macOS, and elsewhere for MoltenVK
/// and for a driver the device does not name; native for any other.
pub fn class_of(driver: Option<DriverId>) -> Placement {
    class_on(DRIVERS_RUN_ON_METAL, driver)
}

/// [`class_of`] on an OS whose every driver runs on Metal where `drivers_run_on_metal` holds.
fn class_on(drivers_run_on_metal: bool, driver: Option<DriverId>) -> Placement {
    match driver {
        Some(d) if !drivers_run_on_metal && d != DriverId::MOLTENVK => Placement::Native,
        _ => Placement::OnMetal,
    }
}

/// The process's placement: no device yet.
const NO_DEVICE: u8 = 0;
/// Every device adopted is native.
const ALL_NATIVE: u8 = 1;
/// A device adopted runs on Metal, or names no driver.
const SOME_ON_METAL: u8 = 2;

/// The process's placement state.
static STATE: AtomicU8 = AtomicU8::new(NO_DEVICE);

/// The state after a device of `class` is adopted in `state`: on Metal for good once any device
/// is.
fn after(state: u8, class: Placement) -> u8 {
    match (state, class) {
        (_, Placement::OnMetal) | (SOME_ON_METAL, _) => SOME_ON_METAL,
        (_, Placement::Native) => ALL_NATIVE,
    }
}

/// The placement of the process in `state`.
fn placement_in(state: u8) -> Placement {
    if state == ALL_NATIVE {
        Placement::Native
    } else {
        Placement::OnMetal
    }
}

/// A device the process adopts, of these properties: its class joins the process's placement.
/// The first reads whether census mode is on, which says so then ([`super::census`]).
pub fn device_made(properties: &rt::DeviceProperties) {
    super::census::on();
    let class = class_of(properties.description.driver_id);
    let _ = STATE.try_update(Ordering::AcqRel, Ordering::Acquire, |s| {
        Some(after(s, class))
    });
}

/// The process's placement, but a test's own on its thread (the tests' `as_native`).
pub fn placement() -> Placement {
    #[cfg(test)]
    if let Some(placement) = TEST_PLACEMENT.with(std::cell::Cell::get) {
        return placement;
    }
    placement_in(STATE.load(Ordering::Acquire))
}

#[cfg(test)]
std::thread_local! {
    /// The placement a test runs its thread's notices under ([`as_native`]), in place of the
    /// process's.
    static TEST_PLACEMENT: std::cell::Cell<Option<Placement>> =
        const { std::cell::Cell::new(None) };
}

/// Runs `call` with the notices announced on this thread placed as on a native device, whatever
/// device the process adopted: a test's on the Mac, where every device runs on Metal. A Linux
/// host's device is native already.
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn as_native<T>(call: impl FnOnce() -> T) -> T {
    /// Puts back the outer placement on drop, an unwinding call's included.
    struct Restore(Option<Placement>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_PLACEMENT.with(|p| p.set(self.0));
        }
    }
    let _restore = Restore(TEST_PLACEMENT.with(|p| p.replace(Some(Placement::Native))));
    call()
}

/// Whether `kind` is one of [`NATIVE_RELAYED`].
pub fn relayed(kind: rt::Degradation) -> bool {
    NATIVE_RELAYED.iter().any(|r| r.kind == kind)
}

/// How many times the runtime announced each kind of [`NATIVE_RELAYED`] in this process, in its
/// order.
static COUNTS: [AtomicU64; NATIVE_RELAYED.len()] =
    [const { AtomicU64::new(0) }; NATIVE_RELAYED.len()];

/// Counts an occurrence of `kind`, one of [`NATIVE_RELAYED`]: whether it is the process's first,
/// which is relayed.
pub fn count(kind: rt::Degradation) -> bool {
    count_in(&COUNTS, kind)
}

/// Each kind of [`NATIVE_RELAYED`] this process met, with how many times the runtime announced it.
pub fn counts() -> Vec<(rt::Degradation, u64)> {
    counts_in(&COUNTS)
}

/// [`count`] in `counts`, one per kind of [`NATIVE_RELAYED`].
fn count_in(counts: &[AtomicU64], kind: rt::Degradation) -> bool {
    let Some(at) = NATIVE_RELAYED.iter().position(|r| r.kind == kind) else {
        return false;
    };
    counts
        .get(at)
        .is_some_and(|c| c.fetch_add(1, Ordering::Relaxed) == 0)
}

/// [`counts`] of `counts`.
fn counts_in(counts: &[AtomicU64]) -> Vec<(rt::Degradation, u64)> {
    NATIVE_RELAYED
        .iter()
        .zip(counts)
        .map(|(r, c)| (r.kind, c.load(Ordering::Relaxed)))
        .filter(|&(_, n)| n > 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Off macOS, MoltenVK, and a device that names no driver, run on Metal; NVIDIA's and Mesa's
    /// drivers are native
    #[test]
    fn the_class_is_the_driver_s() {
        let class = |driver| class_on(false, driver);
        assert_eq!(class(Some(DriverId::MOLTENVK)), Placement::OnMetal);
        assert_eq!(class(None), Placement::OnMetal);
        assert_eq!(class(Some(DriverId::NVIDIA_PROPRIETARY)), Placement::Native);
        assert_eq!(class(Some(DriverId::MESA_LLVMPIPE)), Placement::Native);
    }

    /// On macOS no driver is native: Mesa's KosmicKrisp is Vulkan on Metal too, whichever driver
    /// the system's loader lists first
    #[test]
    fn a_kosmickrisp_driver_on_macos_is_not_native() {
        let kosmickrisp = Some(DriverId::MESA_KOSMICKRISP);
        for driver in [
            kosmickrisp,
            Some(DriverId::MOLTENVK),
            Some(DriverId::NVIDIA_PROPRIETARY),
            None,
        ] {
            assert_eq!(class_on(true, driver), Placement::OnMetal, "{driver:?}");
        }
        #[cfg(target_os = "macos")]
        assert_eq!(class_of(kosmickrisp), Placement::OnMetal);
    }

    /// The process is native only while every device it adopted is: on Metal before its first
    /// device, and for good once one runs on Metal
    #[test]
    fn the_process_is_native_while_every_device_is() {
        use Placement::{Native, OnMetal};
        let run = |classes: &[Placement]| {
            placement_in(classes.iter().fold(NO_DEVICE, |s, &c| after(s, c)))
        };
        assert_eq!(run(&[]), OnMetal);
        assert_eq!(run(&[Native]), Native);
        assert_eq!(run(&[Native, Native]), Native);
        assert_eq!(run(&[Native, OnMetal]), OnMetal);
        assert_eq!(run(&[OnMetal, Native]), OnMetal);
    }

    /// The first occurrence of a kind is relayed and every one counted; a kind not listed is
    /// neither
    #[test]
    fn a_native_relay_is_counted() {
        let counts = [const { AtomicU64::new(0) }; NATIVE_RELAYED.len()];
        let kind = rt::Degradation::DeviceInterpolation;
        assert!(counts_in(&counts).is_empty());
        assert!(count_in(&counts, kind));
        assert!(!count_in(&counts, kind));
        assert!(!count_in(&counts, kind));
        assert_eq!(counts_in(&counts), [(kind, 3)]);
        assert!(!count_in(&counts, rt::Degradation::ReleasedCommandState));
        assert!(relayed(kind) && !relayed(rt::Degradation::ReleasedCommandState));
    }
}
