# The scaler

The M2 scaler device, 106b:1a0b: an x86_64 guest's IOSurfaceAccelerator,
and VideoToolbox's pixel transfers above it, scaling and converting on the
host's GPU. [devices.md](devices.md), "The M2 scaler device", is the
contract: what the guest sends and what Apple's host answers. This doc
says what Apple's scaler was measured to do, how this host does the same,
and how each part is checked. Each measured fact says how, where and when
it was measured.

## What is here

| what | where |
|---|---|
| The device's logic apart from its transport: requests, needs-resets, the capabilities reply, the side buffers through guest memory | `crates/reims-vgpu-video-core/src/scaler.rs` |
| The requests' layouts and flags | `scaler/request.rs` |
| The capabilities' layout | `scaler/capabilities.rs` |
| A transform resolved against its two surfaces: windows, orientation, filter, matrix, siting, alignment, refusals | `scaler/plan.rs` |
| The formats and their planes | `scaler/format.rs` |
| The bytes a transform writes of its destination | `scaler/coverage.rs` |
| `ScalerBackend` and `Accelerator`, and the CPU stand-in | `scaler/backend.rs` |
| The CPU reference every backend is checked against | `scaler/reference.rs` |
| The backend on a compute kernel of our own, on the GPU the library draws with ("The kernel") | `crates/reims-vgpu/src/scaler/gpu.rs`; what the kernel is told, `gpu/params.rs`; each platform's surfaces, `gpu/metal.rs` and `gpu/vulkan.rs` |
| The kernel, and each platform's running of it | `crates/reims-vgpu-metal/src/gpu/scaler.metal` and `scaler.rs`; `crates/reims-vgpu-vulkan/src/gpu/scaler.rs`, with the kernel compiled into `crates/reims-vgpu-vulkan/metallibs/scaler.metallib` |
| Its checks against the reference, on each GPU the host has | `crates/reims-vgpu/src/scaler/gpu/tests.rs` |
| The conformance suite's scaler cases, and its `siting` and `map` lines | `crates/reims-vgpu-video-conformance/src/scaler.rs`, `scaler/siting.rs`, `scaler/mapping.rs` |
| The QEMU device, the C API and the Rust glue | "The wire-up", below |

## Apple's scaler, measured

On the M3 Max (macOS 27.0.1, 26A434), 2026-10-24, by the scaler probe
("The probe", below). Its scaler is AppleM2ScalerCSCDriver, the kernel
driver that also serves an arm64 guest's requests through
`IOSurfaceAcceleratorParavirtProcessGuestData`. Each fact here is a
hypothesis for macOS 26.7's hosts, the contract's, until one is measured
there.

**The capabilities.** Selector 11 of its user client, which type 1 calls
([devices.md](devices.md), "Type 1, capabilities"), works from a user
process. It writes min(asked, 0x298) bytes and puts the count in the
output size: 0x288 for 0x288, 0x298 for 0x300 and for 0x1000, 0x200 for
0x200. The 0x288 bytes' values match the guest's field names (the same
section, the capabilities table, last column). The service publishes 139
keys, with 30 input and 28 output formats.

**The `Transform` bits.** Each of 1 to 8 on a 320x240 BGRA picture with
no left-right or up-down symmetry, compared with each of the eight
orientations of the reference, every one bit-exact (inf dB) for its
match:

| bits | result |
|---|---|
| 1 | flipped vertically |
| 2 | flipped horizontally |
| 3 | turned 180 degrees |
| 4 | turned 90 degrees clockwise |
| 5 | the anti-transpose (4, then flipped vertically) |
| 6 | the transpose (4, then flipped horizontally) |
| 7 | turned 90 degrees anticlockwise |
| 8 | as it was: bit 3 means nothing to this scaler |

So bit 2 turns first, and bits 0 and 1 flip the turned picture. A 420v
quarter turn is bit-exact in luma and 48.8 dB in chroma.

**The windows.**
- A zero width or height is the whole surface: `TransferSurface`, which
  leaves both windows zero, fills the destination.
- The source window is in the source's coordinates, before the turn. The
  destination window is in the destination's. A turned window fills its
  destination window, so a 160x240 source window turned 90 degrees fills a
  240x160 destination window.
- Outside the destination window the destination is untouched: the probe
  compares every byte, the rest left at its mid grey.

**The matrix.** With no matrix attached to either surface, Y'CbCr and RGB
meet through BT.601: 44.2 dB for 420v to BGRA as 601, the 709 reference
worse. With both surfaces tagged BT.709 (`kIOSurfaceYCbCrMatrix`), BT.709:
44.0 dB. VideoToolbox's transfer session tags its buffers, and its
conversions were 709 (45.3 dB).

**The conversions.** Every pair of 420v, 420f, x420, xf20, yuvs, 2vuy,
BGRA and 32ARGB (0x20), source by destination, at a copy (64x48), up
(64x48 to 96x72) and down (128x96 to 64x48), through TransformSurface,
TransferSurface and TransferSurface with FixUpscaling, measured with the
gap probe on 2026-10-25: a 2vuy or 32ARGB source or destination was
refused with kIOReturnUnsupported (0xe00002c7), each of the 252 such
calls; the 36 pairs of the other six succeeded, each of the 324 calls.

A conversion the scaler does not do is refused before any write, with
kIOReturnUnsupported, in under 0.1 ms.

**The conversions' arithmetic**, measured on 2026-10-25 by the gap probe
("The probe"), every byte of each of the conformance suite's 1:1
conversions between 420v, 420f, x420, xf20, yuvs and BGRA set beside a
model:
- A code is a fraction: luma's of `2^bits - 1`, chroma's of `2^bits`,
  with chroma's zero at a half and 1/2048, 128.125 at 8 bits and 512.5 at
  10. So 420v into x420 wrote luma `v · 1023 / 255` and chroma `4 c` on
  every sample, where `4 v` was what the reference wrote.
- Video range's levels are 8-bit ones at both depths, luma 16 to 235 and
  chroma 224 of 255, except where a 10-bit source goes into RGB: there
  they are its own, 64 to 940 and 896. A least-squares fit of x420 into
  BGRA at the chroma samples gave red 0.29104 a luma code, where 64 to 940
  gives 0.29110 and 8-bit levels 0.29024; and blue 0.50384 a Cb code,
  1023/1024 of BT.601's.
- Between its samples, chroma is the mean of the two or four codes about
  it, rounded down: 420v into yuvs wrote `(a + b) >> 1` on every odd row,
  and 420v into BGRA matched that, and `(a + b + c + d) >> 2` between
  four, where the exact mean did not.
- A 10-bit source written as 8 bits is dithered: of luma codes `4k + r`,
  a share `r / 4` came out `k + 1` and the rest `k` (x420 into 420v).
  The transforms set `DitherControl` 1 ([devices.md](devices.md), "The
  flags").

So modelled, 420v and 420f into BGRA matched 99.3% and 99.6% of samples
exactly, and every conversion within one code but xf20 into BGRA, two on
a few samples, where the dither is.

**Scaling.** Against a bilinear upscale and box downscale: 1280x720 to
640x360 41.4 dB, 320x180 to 1280x720 41.1 dB, a 640x360 crop to 320x180
42.3 dB, 1920x1080 420v to 1280x720 BGRA 43.9 dB, a quarter turn with a
320x240 to 160x120 scale 37.5 dB. Odd sizes, 321x241 to 161x121 in 420v:
luma 34.0 dB, chroma 29.3 dB.

**The filter**, measured on 2026-10-25 by the gap probe ("The probe")
with x420 luma impulses, each alone in its row or column, at every 32nd
of a pixel:
- **Small impulses** (24 and 32 codes, 8-bit) were filtered linearly, the
  same across and down, by taps that depend on the ratio `r` of source to
  destination pixels and on the sample point's distance from each source
  pixel in 32nds, the point rounded to a 32nd, halves up. An axis that
  enlarges, keeps its size or shrinks more than 4 times takes four taps,
  one table; one that shrinks up to 4 times takes nine, from one of six
  tables, the one `floor(8 / r)` names, from 7 to 2: either side of 8/7,
  4/3, 1.6, 2 and 8/3, and of 4 and 4.02, the impulses took the table the
  rule names. So 3 and 4 times filter alike, and past 4 times the scaler
  only interpolates. The tables are the reference's (`reference.rs`,
  `ENLARGING` and `REDUCING`).
- **A sharp step takes other taps.** Each pass, down the picture and then
  across over the first's output, unclamped, moves each sample from its
  taps' sum toward the bilinear sample at its point, by the five steps
  between the six source pixels about the point (`reference.rs`,
  `adaptive_blend`). A step of 0.45 of the full range or more next to the
  point moves an enlargement's sample 0.6 of the way, at any ratio from
  1.25 to 8, the nearest filter's too; a reduction's where it is the only
  such step of the five, 0.45 where it is one of two, and not at all where
  there are more or where any other step is 14.5 codes or more. Below it an
  enlargement's moves `(255 m - 32) / 500` of the way, `m` the largest
  step, a quarter at most, and a reduction's stays. The steps are in codes
  over the full range in luma, and in light in each of R, G and B; chroma
  is not moved. Measured on synthetic x420 and BGRA pictures: x420 luma
  impulses moved from 461 10-bit codes high, not at 460, and BGRA red
  impulses over backgrounds of 0 to 160 from 0.4508 in light, not at
  0.4498; every set of six neighbours raised by 112, 120 and 160 codes,
  doubled and halved, moved as the rule says at every sample; a green
  impulse of 16 beside a red one of 160 took the small taps, and so did
  Cb impulses of 96 to 160; impulses in both axes at once came within 1 to
  2 10-bit codes of the rule with the pass down first, 11 to 19 off with
  the pass across first. A moved sample matched 0.6 of the way to the
  bilinear sample within 0.002 of the impulse a tap. With the rule the
  suite's BGRA enlargements came from 36 to 41 codes off Apple's to 2 to
  11, and from 48.8 to 52.2 dB to 62.8 to 70.0; its reductions from 48.4 to
  63.3 dB to 55.6 to 70.6, at 36 to 46 codes off on a few samples. A
  reduction's isolated feature below the step is not modelled
  ([bugs.md](bugs.md), B83).
- **An RGB source scales in linear light**, into RGB or Y'CbCr: the suite's
  BGRA scales, 1.5 to 8 times up and 2 and 4 down, a turned one and a window,
  came within a code of the reference in red and green on 97% to 99% of
  samples, and 1 to 3 codes on the rest, with sRGB's curve taken there and
  back; scaled in codes they were up to 61 codes off. BGRA impulses of 32
  codes scaled 2 times up and 1.03 to 2 down, across and down, into x420 and
  420v (2026-10-25) matched within a code so, where scaling codes left x420's
  luma 4 to 6 codes off; the scaled light is clamped to black and white before
  it is converted, which took the test picture's worst x420 luma from 32 codes
  off to 14. Where the test picture's taps span under 24 codes, every luma
  sample matched within a code; the rest is the adaptive filter's (above).
  That the curve is sRGB's is a fit, not known.
- **The hints** `DisablePseudoLinearScaling`, `DirectionalScaling`,
  `SymmetricScaling`, `EdgeMirroring` and `EnhanceEnable` off changed no
  impulse. `PixelAveraging` made a halving the mean of each pair.
- **Edges**: the edge pixels repeat beyond a window. The reference so
  filtered matched Apple's luma within a code at 4 to 16 pixels a side.
- **The nearest filter** (`UseNearestFilter`) changes nothing: nine BGRA
  and 420v scales, 1.1 to 3 times up and 2 and 4 down, wrote the same
  bytes with it as without, and x420 impulses at every 32nd of a pixel,
  2 and 1.03 times up and 1.03 to 5.34 times down, took the smooth
  filter's taps with it, through TransformSurface and TransferSurface.

**The ratios.** This scaler offers `MaxHUpscale`, `MaxHDownscale` and
`MaxVUpscale` 8, and `MaxVDownscale` 4. A transform past one is refused
with kIOReturnBadArgument (0xe00002c2), the ratio taken after the
orientation (the probe's `limit` cases, BGRA):

| case | scale | status |
|---|---|---|
| 64x64 to 512x64 | 8 up across | 0 |
| 64x64 to 1024x64 | 16 up across | 0xe00002c2 |
| 1024x64 to 64x64 | 16 down across | 0xe00002c2 |
| 64x64 to 64x1024 | 16 up down the frame | 0xe00002c2 |
| 64x512 to 64x64 | 8 down the frame, past its 4 | 0xe00002c2 |
| 64x1024 to 64x64 | 16 down the frame | 0xe00002c2 |
| 512x32 to 32x512, a quarter turn | 1 after the turn, 16 down across before it | 0 |
| 32x32 to 512x32, a quarter turn | 16 up across after the turn | 0xe00002c2 |

**Speed**, single runs: 0.4 to 1.1 ms a transform up to 1280x720, 1.9 ms
for 1080p to 720p with a conversion. VideoToolbox's first transfer took
20 to 76 ms, its later ones 1.8 to 2.7 ms.

## The probe

Apple's scaler was measured with two probes of ours, written from scratch
and kept outside this repository, run as a process on the M3 Max and in
x86_64 guests; nothing of a run is committed. Both look up
IOSurfaceAccelerator's functions with `dlsym`, as no SDK has them.

- **The scaler probe** drives the scaler two ways:
  `IOSurfaceAcceleratorTransformSurface` and `…TransferSurface`, and
  VideoToolbox's `VTPixelTransferSession` and `VTPixelRotationSession`.
  - Each case fills its source with a picture drawn from scratch (ramps,
    a disc, a checkerboard), runs one call, and compares the whole
    destination with a CPU reference of its own: per-plane PSNR over
    every byte. It passes at 34 dB for a copy or a conversion, 30 dB when
    it scales, 28 dB where a size is odd.
  - Where the matrix or the orientation is the question, it makes each
    reference and names the best.
  - A refusal with kIOReturnUnsupported is counted apart: a scaler offers
    what its hardware does. A run in which no case passes fails, so that
    a scaler refusing everything does not pass.
  - It prints the class of the service it opened.
- **The siting probe** measures the chroma kernels of a conversion
  ("Chroma siting") and, in a second mode, where a scale takes each
  destination pixel from ("Where a scale samples").
- **The gap probe** runs the conformance suite's scaler cases, and sets
  of its own (impulses, odd windows, fractional windows, small sides),
  through `TransformSurface`, and through `TransferSurface` with and
  without the FixUpscaling option, on the sources the suite draws, and
  keeps every destination byte. Core's reference runs the same plan on
  the same bytes, and the two are compared plane by plane: PSNR, the
  largest difference, and the share of samples more than a code apart.
  On 2026-10-25, before the rules of "The conversions' arithmetic", "The
  filter" and "Where a scale samples" were modelled, 58 of the 104 cases
  Apple's scaler takes through TransformSurface were within a code of the
  reference in every plane; after, 87, and with the adaptive filter
  ("The filter") still 87, with the BGRA scales' PSNR from 48.4 to 63.3 dB
  to 55.6 to 70.6. Of the other 17, the BGRA scales' and RGB into 420v's
  few samples 2 to 46 codes off, and odd-yuvs-BGRA's 3 codes, are B83's
  and what the filter's rule leaves at sharp edges, and xf20 into BGRA's
  two codes the dither ("The conversions' arithmetic"). Through
  TransferSurface, with and without FixUpscaling, 65 of 80.

The kernel is measured the same ways by the conformance suite: its cases
as the scaler probe's, and its `siting` and `map` lines as the siting
probe's ("The kernel").

On the M3 Max, Apple's scaler writing into a surface read-only for its
clients panicked the kernel, and an x86_64 guest cannot make such a
surface ([devices.md](devices.md), "Surfaces read-only for their
clients").

## How this host does a transform

core's `ScalerDevice` takes each element from the transport, with the
guest's memory behind a `GuestMemory` trait and its surfaces behind
`Surfaces`, as the VideoToolbox device has them. `Plan::new` resolves the
transform against the two surfaces' layouts:

- **Windows**: zero means whole, as Apple's. A window that leaves its
  surface is kIOReturnBadArgument (ours: Apple's answer is not measured).
- **Orientation**: bits 0 to 2, as measured above. Bits 3 to 5 are
  ignored, as bit 3 is on the M3 Max.
- **Ratios**: past the `MaxHUpscale`, `MaxHDownscale`, `MaxVUpscale` or
  `MaxVDownscale` the backend offers, taken after the orientation, the
  transform is kIOReturnBadArgument with the destination untouched, as
  Apple's scaler answers (above, "The ratios"; `plan.rs`, `ratio_past`).
  A zero offers no bound.
- **Filter**: `UseNearestFilter` asks for nearest, which Apple's scaler
  ignores ("The filter"), so the reference, and the kernel with it,
  filters as without it.
- **Alignment**: each destination axis keeps pixel centres where it
  enlarges with `FixUpscaling` (bit 12), and puts the windows' first
  pixels on each other otherwise, as Apple's scaler does ("Where a scale
  samples"; `plan.rs`, `Alignment::of`).
- **Matrix**: BT.601. The guest's IOSurface mapper sends no colour
  attachments, so every surface reaches the host untagged, and BT.601 is
  what Apple's scaler uses for those.
- **Chroma siting**: the destination's, from flag bits 15 to 17, as
  Apple's scaler writes it; a source's is always read as Apple's scaler
  reads it ("Chroma siting", below; `plan.rs`, `Siting`).
- **Refusals**: an option that changes the output in a way this host
  does not (`BorderFill`, `AlphaOverride`, `HistogramBinMode`, `ZeroFill`,
  `ColorRemapMode`, `HDREnable`, `EnhanceEnable`, `WriteOnlyMode`,
  `FilmGrainEnable`, `BorderFillOddSizeYCbCrEnable`) is
  kIOReturnUnsupported, logged with its name. The other flags are hints,
  ignored (`plan.rs`, `refused_option`).
- **Formats**: 420v, 420f, x420, xf20, yuvs, 2vuy, BGRA and 32ARGB, as
  `format.rs` lays them out. Any other is kIOReturnUnsupported.
- **The conversions Apple's scaler refuses**: every transform from or into
  2vuy or 32ARGB, 28 of the 64 pairs, is kIOReturnUnsupported with the
  destination untouched, logged with the pair's names (`format.rs`,
  `apple_refuses`), as the M3 Max's scaler answers them (above, "The
  conversions"; [devices.md](devices.md), "The M2 scaler device").

The status goes to `R+4`. The differences from Apple are in
[devices.md](devices.md), "Where we differ from Apple".

**The reference.** `scaler/reference.rs` does every plan on the CPU: a
Y'CbCr source scaled in Y'CbCr, its chroma with taps of its own where
"Where a scale samples" puts it apart from its luma, and converted to RGB
after the scale where the destination is RGB; an RGB source scaled in
linear light, into RGB or Y'CbCr ("The
filter"), with BT.601 or 709, video or full range, its codes taken
and written as Apple's scaler takes them ("The conversions' arithmetic"),
rounded to the nearest where Apple's dithers. It turns a
Y'CbCr source's window as a picture of its own, reads its chroma on the
first pixels of the turned window's pairs and interpolates between, and
writes a destination's chroma as its siting says, all as Apple's scaler does ("Chroma siting"), so a copy or a turn
between surfaces of one format is exact. It filters with Apple's
scaler's small-impulse taps ("The filter"), each sample point where the
plan's alignment puts the pixel ("Where a scale samples"), and repeats a
window's edge pixels beyond it. So its luma came within a code of Apple's
on every 420v and x420 scale of the suite (down2-420v 66.2 dB, up4-420v
67.9, scale-x420-420v 52.6, odd-420v 66.4), where a bilinear and box
filter was 50 to 57 dB, up to 5 codes off. Its tests check every format's copy (over 50 dB a plane),
turns of 420v and x420 moving each plane's samples as they are, each
siting's chroma, two quarter turns making a half turn, a window's outside
untouched, scales sampling where the alignment puts them, and a halving
and a doubling against flat pictures. It is the `ReferenceBackend` the
device's tests run on, and the oracle the backends are checked against.

## Chroma siting

Where a 4:2:0 or 4:2:2 surface's chroma samples sit among its luma pixels.
A guest's IOSurfaceAccelerator reads the destination's chroma location
alone and sends it as transform flag bits 15 to 17 (read from Apple's
code): 1 left, 2 centre, 3 top left, 4 top, 5 bottom left, 6 bottom, 0
for none. The source's is not sent.

**Apple's scaler, measured** with the siting probe on the M3 Max's
AppleM2ScalerCSCDriver (macOS 27.0.1, 26A434), 2026-10-24, on ordinary
CVPixelBuffer surfaces only. Kernels are weights over the luma pixels
from a chroma sample's first less 1:

| conversion | kernel | so |
|---|---|---|
| 420v and 420f to BGRA, across and down; yuvs to BGRA across | 0.25 0.50 0.25 | each sample on its first column and row, interpolated between |
| the same, the source or the destination tagged with any of the six locations | the same | the source's own siting is not read, and the bits do not change it |
| BGRA to 420v, bits 0 or 3 | across 0 1 0, down 0 1 0 | the first pixel both ways, no filter |
| bits 1 (left) | across 0 1 0, down 0 0.5 0.5 | the first column, the mean of the two rows |
| bits 2 (centre) | 0 0.5 0.5 both ways | the mean of the two |
| bits 4 (top) | across 0 0.5 0.5, down 0 1 0 | |
| bits 5 (bottom left) | across 0 1 0, down 0 0 1 | the second row |
| bits 6 (bottom) | across 0 0.5 0.5, down 0 0 1 | |
| BGRA to yuvs, bits 0 | 0 1 0 | the first column |
| 2vuy in or out | refused, kIOReturnUnsupported | |

Through TransferSurface and TransformSurface alike. VideoToolbox's pixel
transfer on the same Mac gave the same kernels for every 420v, 420f and
yuvs case, with the locations as `kCVImageBufferChromaLocationTopFieldKey`
attachments, and replicated 2vuy's chroma (0.5 0.5) both ways.

**Turns and scales.** A 64x64 420v given each of the seven turns put a
raised chroma line 11 at 20.000 or 11.000 exactly, as a turn of the
plane as it is puts it, not at 20.5, where a turn that kept each sample
on its first pixel would. An odd surface turns its window, not itself:
x420 chroma impulses through 65x49 and 67x51 surfaces in each of the
seven turns (2026-10-25) landed whole on the destination's samples, a
flipped 65-wide surface's chroma column 4 on column 27, the window's
first 64 columns reversed, where the surface's 33 would put it between
27 and 28 (`reference.rs`, `decode_turned`); the transpose, which
reverses neither axis, is the same either way. Halved and doubled
through TransformSurface with no siting named, a raised chroma line 11
or 22 landed at 5.50 and 22.15, where a scale that keeps pixel centres
puts a sample sited on its first pixel at 5.375 and 22.25, and a centred
one at 5.25 and 22.5. Through TransferSurface, which sets no
`FixUpscaling` ([devices.md](devices.md), "Type 2, transform"), a
doubling put a raised luma column 22 at 44.0, where TransformSurface and
VideoToolbox put it at 44.5. Where a scale takes its pixels from says
why: a halving puts first pixels on each other on every path, and an
enlargement keeps centres only with bit 12 ("Where a scale samples").

**What this host does.** Core's plan carries the destination's siting
(`Siting::of_cosite`; 7, which the framework never sends, is taken as
none), and the reference does all of the above, its kernels held to
Apple's by the conformance suite's
`the_reference_sites_chroma_as_apples_scaler`. The kernel ("The kernel")
takes the reference's taps and sitings, so it reads and writes chroma
where the reference does. Its `siting` lines, the same on every GPU it
ran on ("The kernel"): 420v, 420f and yuvs to BGRA 0.25 0.50 0.25 each
way, the centre weight 0.51 across for 420v and yuvs, and BGRA to 420v
and yuvs each siting's kernel in the table above, to the hundredth
(2026-10-25).

## Where a scale samples

Where a scale takes each destination pixel from, along each axis of the
destination: pixel `d` of a destination window `D` long, from a source
window `W` long as turned, pixel `i`'s centre at `i`. Two rules matter:
the windows' pixel centres kept, `(d + 0.5) W / D - 0.5`, and their first
pixels on each other, `d W / D`. They differ by `(1 - W / D) / 2` source
pixels: a quarter of one for a doubling, half of one the other way for a
halving.

**Apple's scaler, measured** with the siting probe's second mode on the
M3 Max's AppleM2ScalerCSCDriver (macOS 27.0.1, 26A434), 2026-10-24, on
ordinary CVPixelBuffer surfaces only. The source rises by 6 a pixel over
teeth of 32 pixels along one axis, in BGRA's red and 420v's luma, and
falls so in green or in a second transform. Inside a tooth a filter that
keeps a straight line straight gives back the position each destination
sample was taken from, and the mean of the rising and falling readings
takes out the rounding. Each axis's positions are fitted, away from the
teeth's and the picture's edges, and set beside each rule: the measured
position less the rule's, and its spread. Each case ran four ways:
TransferSurface, which sets no `FixUpscaling`; TransferSurface with the
FixUpscaling option, which sets bit 12; TransformSurface, which sets it
always; and VideoToolbox's pixel transfer:

| the scale, 420v and BGRA | TransferSurface | with bit 12, and VideoToolbox |
|---|---|---|
| enlarging: 1.1 (100 to 110), 4/3, 1.5 (64 to 96, 720p to 1080p), 2 (32 and 64 a side; 321x241 and 33x33 BGRA), 2.5 (40 to 100, and 64 to 160 across alone), 2.67 by 2.25 (480p to 1080p), 3, 3.13 (31x31 BGRA to 97x97), 4 and 8 times | first pixels, `d W / D`, within 0.002 of a source pixel | centres, `(d + 0.5) W / D - 0.5`, within 0.002 |
| shrinking: 1.5 (96 to 64, 1080p to 720p), 2 (64 to 32, 65x65 BGRA to 33x33), 2.5 and 4 times | first pixels, within 0.003; 2.5 times 0.09, its spread 0.08 | the same |
| 999x555 BGRA to 1000x556 | either: the rules differ there by 0.0005 | either |

- **Bit 12 alone** makes the difference: TransferSurface with the
  FixUpscaling option gave every line TransformSurface gave, so
  TransferSurface's zero windows do not. VideoToolbox's lines were
  TransformSurface's, every unturned case to the thousandth (the probe
  gives VideoToolbox no turn): its pixel transfer is this scaler with the
  bit.
- **Each axis by its own ratio**: 64x32 into 32x64 put first pixels on each
  other across, and down the frame kept centres with bit 12 and first
  pixels without; 64x48 into 160x48 left the rows as they were.
- **After the turn**: the rule holds in the source as turned. A flipped
  doubling through TransferSurface took destination pixel 0 from the
  source's last pixel (`s = 31 - d / 2`), and a quarter turn applied the
  rule along the destination's axes.
- **Windows**: a window's origin is where the rule starts. (10, 6) 64x64
  into (8, 4) 32x32 took `x = 10 + 2 (dx - 8)` exactly, and (10, 6) 16x16
  into the same kept centres from (10, 6).
- **Chroma** followed luma's rule, both planes read and written on their
  first pixels ("Chroma siting"), but where an axis enlarged with bit 12
  and the source subsampled chroma along it. There it kept half the
  luma's offset (`plan.rs`, `Alignment::chroma_source_index`): the
  destination's chroma took `(d + 1/4) W / D - 1/4` in the source's luma
  pixels, `(1 - W / D) / 8` chroma samples from first pixels toward
  centres where the luma's rule would put it `(1 - W / D) / 4`; along an
  axis the source does not subsample, 4:2:2 down, it took the luma's
  place. Chroma impulses (2026-10-25) through x420, 420v and yuvs
  sources enlarged 1.03 to 8 times, across and down, into x420, 420v, yuvs
  and BGRA, quarter turns, a window and a destination window among them,
  fitted it: the impulses' centroids at 2, 2.5, 3, 4 and 8 times within
  0.008 of a chroma sample, and every picture within a code of the
  reference with it, where the luma's place left x420 chroma up to 18
  codes off. It scales a Y'CbCr source's chroma before it converts to
  RGB, so the rule holds into BGRA too: 420v doubled into BGRA, 16 codes
  off with chroma at the luma's place, came within a code.
- **Odd 4:2:0 sizes**: the scaler scaled the even part. 33x33 into 65x65
  took luma columns 0 to 31 into 0 to 63 (`s = d / 2`) and left column and
  row 64 of the destination as they were; 321x241 into 642x482 took 320
  columns at `320 / 642` a pixel, and 999x555 into 1000x556 was half a
  pixel off both rules. BGRA of the same sizes scaled whole.
- **Windows in even pairs** (the gap probe, 2026-10-25): along each axis a
  Y'CbCr surface subsamples chroma on, both 4:2:0 axes and 4:2:2's across,
  each window starts on the even pixel at or before its first, its
  fraction dropped, and spans its length rounded down to an even number of
  pixels. 420v and yuvs copies of a 128x96 picture with windows at
  (13, 9), (12, 8) and (13, 8), 63x47 and 64x48, at either end, wrote
  exactly what the windows so rounded copy and nothing outside them; a
  whole 65x49 copied its 64x48 and left the last column and row; a source
  window (10.5, 6.5) copied from (10, 6). A BGRA end took its window as
  given. Chroma is then read from the source window's own samples, the
  last repeated past it: into BGRA, a window whose last column took the
  next chroma sample was up to 75 codes off, and 1 to 3 without it.
- **Fractional windows**, through TransformSurface: the origin's fraction
  was kept, the ratio was that of the whole pixels the window touches, and
  the rule was as for a whole window. (3.75, 5.5) 60x45 into 40x30 took
  1.525 and 1.534 a pixel, 61/40 and 46/30; (20.25, 30.75) 40x30 into
  100x75 0.410 and 0.414, 41/100 and 31/75; the two (10.5, 6.25) windows
  within 0.01 of the same, and each origin within 0.12. Whole pictures of
  17 BGRA and 420v windows (the gap probe, 2026-10-25) fit a rule: a
  window that starts within a pixel takes the whole pixels it touches, one
  that starts on a pixel its length's whole pixels. Where those are the
  destination window's, it copies them from the first, byte for byte: (10.5,
  6.25) 64x48 into 65x49, (10, 6) 63.5x47.25 into 63x47. Otherwise it scales
  them from where the window starts, its fraction kept, red and green
  within a code of the reference on 97% to 100% of samples. A Y'CbCr
  window then pairs as above, so (11, 7) 63.5x47.5 of a 420v scaled (10, 6)
  62x46. A window that starts within a pixel and ends on one takes the
  pixel past its end too, and one that ends on the surface's last pixel
  its length and one pixel more, scaled from where it starts: red
  impulses through such windows (2026-10-25), the centroid of each fitted
  to a map from source to destination, gave (10.5, 0) 63.5 wide into 64,
  80 and 96, (0.5, 0) 63.5 into 64, and (20.25, 0) 43.75 into 44 and 50
  scaling 65, 65 and 45 pixels from their starts, and (0.5, 0) 127.5 wide
  of 128 into 64, 118, 127, 128, 129 and 256, (0.25, 0) 127.75, (0.75, 0)
  127.25, (1.5, 0) 126.5 and (10.5, 0) 117.5 scaling their length and one
  more, each within 0.1 of a pixel; with that rule the impulses' pictures
  came within 2 codes of the reference, and (0.5, 0.5) 127.5x95.5 into
  128x96, 30.8 dB against Apple's as a copy of the pixels it touches, to
  46.7 dB.

**What this host does.** Core's plan carries an alignment per destination
axis (`plan.rs`, `Alignment::of`): centres where the axis enlarges with
bit 12, first pixels otherwise, with the window's origin where it starts.
The reference centres each footprint there. The plan takes a source
window's whole pixels (`plan.rs`, `touched`) and a Y'CbCr window in even
pairs (`even_window`), so the odd 4:2:0 sizes scale their even part, and
the reference reads chroma within the source window. It takes a Y'CbCr
source's chroma where `Alignment::chroma_source_index` puts it, and
converts to RGB after the scale. The suite's cases set bit 12, as a
guest's TransformSurface does, except its `transfer-` cases: whole
windows without it, as TransferSurface sends them. Its `map` lines
measure a backend as the siting probe measures Apple's
(`crates/reims-vgpu-video-conformance/src/scaler/mapping.rs`); the
reference's own are within 0.02 of its plan
(`the_reference_samples_where_its_plan_says`).

**The kernel**, by the suite's `map` lines, samples on the plan: its 88
lines, the same on every GPU it ran on (2026-10-25, "The kernel"), are
each within 0.012 of a source pixel of the plan's position, with a spread
of at most 0.095, where the plan's two rules part by up to 1.5. That is a
TransferSurface enlargement and a fractional window included, which the
backends before it missed by up to 0.75 and 1.0 of a source pixel, as
they could place a window in whole pixels only.

## The kernel

`GpuScalerBackend` (`crates/reims-vgpu/src/scaler/gpu.rs`) is core's
`ScalerBackend` on compute kernels of our own
(`crates/reims-vgpu-metal/src/gpu/scaler.metal`), on the GPU the library
draws with, whatever GPU that is, a software rasterizer included:
- **macOS**: Metal's system default device. The kernels are compiled at
  run time with fast math off (`reims_vgpu_metal::gpu::scaler`).
- **Linux**: the Vulkan platform's runtime default device, the one the GPU
  device draws on, so a VM reims-ctl pins to a GPU through the loader's
  environment scales on that GPU (`reims_vgpu_vulkan::gpu::scaler`). The
  same source is compiled on a Mac with fast math off into
  `crates/reims-vgpu-vulkan/metallibs/scaler.metallib`, which the
  runtime translates.

Nothing chooses it. The device makes it at its first use, once a process
(`crates/reims-vgpu/src/scaler.rs`, `gpu_backend`), and logs `scaler
device: transforms on <the GPU's name>`.

**What it computes** is the reference's transform ("How this host does a
transform"). The CPU works out the plan's numbers (`gpu/params.rs`), each
from the reference's own functions: each destination plane's window, rows
and bytes; the source's planes and sample order; how a code reads and is
written (`luma_of`, `chroma_of`, `code_luma`, `code_chroma`); the matrix;
the turn as a map of the turned picture's pixels onto the source's; where
the turned source window's chroma samples sit, each on the source's
sample under its first pixel (`chroma_window`); whether the scale is in
linear light (`in_linear_light`); the destination's siting;
and the taps (`reference::taps`), each destination pixel's source pixels
and weights along each axis, Apple's filter's, up to nine and some
negative, merged by pixel and divided by their sum. The kernels run in
one command buffer, a dispatch a destination plane, one thread to a
32-bit word of the plane's rows, so that no two threads write one word.
For each of a word's samples they decode the source pixels the taps
take, a chroma sample's codes interpolated from the window's own samples
and the sum rounded down; resample them, an RGB source's in linear light
and a Y'CbCr source's chroma by its own taps where it has them, each pass
moved by the adaptive filter ("The filter") from the six pixels about its
sample point, which the CPU gives with the taps;
convert them; site the chroma as the destination's bits say;
and round half away from zero and clamp to the code range, as the
reference writes a sample. They compute in 32-bit floats, with fast math
off, where the reference computes in doubles. An RGB source's codes go
into linear light through a table of the 256 codes' values there, which
each threadgroup of a kernel that decodes the source works out first,
each by the kernels' `srgb_to_linear`, so that a decode is three loads,
not three powers at each tap, which fast math off keeps precise and slow
(the table took 29% off turn-aniso-BGRA's GPU time on the M3 Max,
"Speed"). A word the window covers only in part is read and written
back whole, its other bytes as they were.

Which kernels run is the CPU's choice (`gpu/params.rs`, `work`); each
sums in the reference's order:
- **A transform whose destination pixels each take more than one source
  pixel each way** is separable, and runs down, then across.
  `scaler_lines` first sums each column of the turned source the
  window's across taps take, down each destination row's taps, into a
  line of floats: four, or three for a Y'CbCr source, whose pixels are
  opaque. A threadgroup decodes each source pixel its rows take once,
  into threadgroup memory. Then a dispatch a plane sums the lines across:
  `scaler_across`, a thread a word, or, for a reduction across into a
  plane whose words hold more than one pixel, `scaler_across_shared`,
  whose threads work out the pixels side by side into threadgroup memory
  before they write the words. So a source pixel is decoded once a line,
  not once a destination pixel: at a reduction of up to 4, where Apple's
  filter takes nine source pixels a side, 18 taps an output rather than
  81. The lines take 16 MiB at most: a taller window runs a band of its
  rows at a time.
- **A copy of a Y'CbCr source into RGB** runs `scaler_quads`: a thread
  decodes two by two pixels, which share their chroma samples, into
  threadgroup memory, and the threadgroup writes its 32 words by 8 rows a
  row at a time, side by side. A discrete GPU writes guest memory across
  the bus, where stores with gaps between them crawl: on an RTX 5080
  Laptop GPU a 4K copy whose threads each wrote their own two by two
  pixels, so that each store's words lay 8 bytes apart, took 320 ms, and
  2.8 ms written side by side (2026-10-25,
  `crates/reims-vgpu/src/scaler/gpu.rs`, "Speed").
- **Anything else** runs `scaler_transform`, a thread a word, each pixel
  worked out from the source; where the adaptive filter moves either
  pass, `scaler_transform_adaptive`, the same with the filter, a kernel
  of its own so that the rest run, and are made, without its code
  (below); and for both planes of a 4:2:0 window that the filter does
  not move, `scaler_transform_both_planes`, a kernel of its own too
  (below).

**Making the kernels.** The Vulkan runtime makes a kernel's pipeline at
its first dispatch of each threadgroup size, so a process's first
transform on each kernel waits for the making, inside the 5 s a
transform is given before it fails (`device_wait`), unless the making
ahead (below) made it. How long the making takes follows how much code
the kernel has once its calls are inlined: each helper's code once for
every place it is called from, down the chain of calls. With the
filter's six pixels decoded at each place a tap read them, about 60
decodes a pixel inlined, the making of `scaler_transform` took longer
than 5 s on the RTX 5080, the RX 7900 XTX and MoltenVK on the M3 Max,
and the first transform failed. Each pass decodes its six in the same
loop as its taps, and the filter has a kernel of its own.

Each kernel also works a pixel out at one place. `write_word` takes the
pixels of its word's block in one loop, a row at a time, where it had a
place for each kind of plane and each pixel of a 4:2:0 word, and
`decoded_quad` takes its four pixels in a loop. A word of RGB, one
pixel, has a path of its own: on the Radeon 780M the loop's tests cost
a 1:1 RGB transform more than its pixels' work (1080p BGRA turned 90,
1.6 ms of GPU time through the loop against 1.16 alone). Keep it so: a
pixel worked out, or decoded, at each place a word or a tap takes it is
multiplied by every call site the inlining makes. Each kernel's blocks
(the SPIR-V the runtime translates it into, with the RTX 5080's
profile: its `OpLabel`s), and a process's first transform on it with
nothing made ahead, cold as below, before then after the kernels
worked a pixel out at one place; the outputs are the same bytes on
every GPU (wall ms, 2026-10-26, the RTX 5080's three processes within
1% of each other, two on the Radeons at a load average of 2 to 4 from
other work, the M3 Max's at 72 then 9):

| first transform (its kernels) | blocks | RTX 5080 | RX 7900 XTX | Radeon 780M | M3 Max, MoltenVK |
|---|---|---|---|---|---|
| 2160p 420v to 2160p BGRA (`scaler_quads`) | 748 to 216 | 233 to 104 | 85, 103 to 69, 76 | 87, 87 to 58, 47 | 488 to 255 |
| 1080p 420v to 720p BGRA (`scaler_lines` of 16 rows, `scaler_across`) | 829 to 478, 1448 to 466 | 321 to 165 | 145, 211 to 62, 85 | 109, 111 to 60, 58 | 735 to 320 |
| 2160p 420v to 1080p 420v (`scaler_across_shared`, unchanged) | 425 | 117 to 117 | 38, 40 to 38, 69 | 37, 38 to 42, 45 | 222 to 201 |
| 720p BGRA to 180p BGRA (`scaler_lines` of 8 rows) | 829 to 478 | 221 to 153 | 64, 74 to 56, 74 | 70, 71 to 57, 52 | 296 to 215 |
| 1080p BGRA turned 90 (`scaler_transform`) | 1824 to 560 | 724 to 183 | 278, 610 to 51, 95 | 196, 196 to 53, 50 | 1337 to 353 |
| 240p BGRA turned 90 to 160p (`scaler_transform_adaptive`) | 3696 to 1028 | 1530 to 348 | 656, 1788 to 83, 150 | 367, 376 to 73, 73 | 2508 to 577 |
| the making ahead, all seven | | 3.2 to 1.16, 1.18 s | 1.1 to 0.44, 0.33 s | 0.9 to 0.36, 0.29 s | 6.9 to 2.1 s |

Warm, the suite's benches took as long or less on the M3 Max, the RTX
5080 and the RX 7900 XTX, in alternated processes; on the Radeon 780M
1080p BGRA turned 90 went from 1.33 and 1.27 ms of GPU time to 1.18 and
1.13, and 720p BGRA to 720p 420v, one dispatch of both planes, from a
median of 0.26 ms to 0.31 over five processes of each. The table of
linear light ("What it computes") then took the decode's powers out of
the code as well: 453 blocks for the lines kernel, 537 for
`scaler_transform`, 169 for `scaler_quads` and 1005 for the adaptive
kernel, and the making ahead of all seven took 1.10 s on the RTX 5080,
0.31 on the RX 7900 XTX and 0.27 on the Radeon 780M, one process each;
the tables here and below are of the kernels before it. Once made, a
pipeline is kept in the runtime's shader cache and the driver's, and a
later process's first transform takes a few milliseconds.

Both planes of a 4:2:0 window then took a kernel of their own,
`scaler_transform_both_planes`. Its `both_planes` works out a word's
pixels in a loop over the two luma rows of a chroma row, a constant count.
The driver unrolls that loop into two copies of a pixel's work, which
run side by side. It is faster than `write_word`'s single loop over the
word's block, and it is more code, which `scaler_transform` no longer
holds. On the Radeon 780M (Mesa 26.2.6, 2026-11-07, six processes of
each build, alternated, 300 transforms each after a first, at a load
average of 0.6 to 1.2), 720p BGRA to 720p 420v went from a median of
0.230 ms of GPU time to 0.206, its 10th percentile from 0.200 to 0.186.
An earlier A/B on another Radeon 780M put `both_planes` inside
`scaler_transform` instead. It was as fast, 0.207 ms against 0.227, and
it made `scaler_transform` 31,216 bytes of code against 21,524
(`RADV_DEBUG=shaderstats`). The suite's other benches stayed within
their spread, `scaler_transform`'s 1080p BGRA turned 90 among them, at
0.752 and 0.762 ms. Cold, the making ahead of all eight took 298 and
297 ms against 274 and 274 ms for seven. With nothing made ahead, a
first 720p BGRA to 720p 420v took 27 ms, its own kernel's making,
against 47 to 53 ms for `scaler_transform`'s
(`scaler/gpu/tests/firsts.rs`).

**Made ahead.** When a process's first scaler device is made, a thread
of the library's dispatches each kernel at each threadgroup size the
transforms dispatch it at, eight in all (`params::DISPATCHED`), one
after another, with parameters of zeros, so that every thread returns
at its bounds check, on a command queue apart from the transforms',
binding only a 16-byte buffer of its own (`gpu::make_ahead`). Neither
the device's creation nor a transform waits for it. It starts no
dispatch while a transform runs, and a transform that comes first takes
its pipeline as it stands: made, in the making (it waits for the rest
of that making, which the making ahead's dispatch began), or not begun
(it makes its own). Where a dispatch is refused, fails or does not end
within 60 s, the kernels after it are made at their first transforms,
as without it.

A process's first transform on each kernel, with nothing made ahead and
then once the making ahead is done, cold: the runtime's shader cache new
(it lies under `HOME`, `.cache/reims-vgpu-vulkan`), the driver's own
off (`__GL_SHADER_DISK_CACHE=0`, `MESA_SHADER_CACHE_DISABLE=true`), and
each making shown cold by the runtime's counts: a pipeline made for each
miss the driver reported, no hit. On the M3 Max each run was a build of
its own, its kernels' code changed in a branch no pixel takes, so that
Metal's compiler cache, the host's, which cannot be turned off, held
none of it. Wall ms, of two processes each way where two are given
(2026-10-26, `scaler/gpu/tests/firsts.rs`; the RTX 5080 Laptop GPU at
driver 615.71.09, its first column three processes within 1% of each
other; the RX 7900 XTX and Radeon 780M at Mesa 26.2.4, their host at a
load average of 1 to 4 from other work; the M3 Max's at 8 to 9):

| first transform (its kernels) | RTX 5080 | RX 7900 XTX | Radeon 780M | M3 Max, MoltenVK | M3 Max, Metal |
|---|---|---|---|---|---|
| 2160p 420v to 2160p BGRA (`scaler_quads`) | 104 to 5, 6 | 69, 76 to 18, 10 | 58, 47 to 24, 10 | 255 to 6 | 6.0 to 2.5 |
| 1080p 420v to 720p BGRA (`scaler_lines` of 16 rows, `scaler_across`) | 165 to 3, 3 | 62, 85 to 6, 6 | 60, 58 to 9, 6 | 320 to 3 | 2.8 to 2.7 |
| 2160p 420v to 1080p 420v (`scaler_across_shared`) | 117 to 4, 4 | 38, 69 to 9, 8 | 42, 45 to 18, 14 | 201 to 4 | 3.3 to 4.1 |
| 720p BGRA to 180p BGRA (`scaler_lines` of 8 rows) | 153 to 1, 1 | 56, 74 to 2, 2 | 57, 52 to 2, 2 | 215 to 2 | 1.0 to 1.0 |
| 1080p BGRA turned 90 (`scaler_transform`) | 183 to 3, 3 | 51, 95 to 8, 4 | 53, 50 to 6, 3 | 353 to 2 | 1.3 to 1.5 |
| 240p BGRA turned 90 to 160p (`scaler_transform_adaptive`) | 348 to 1, 1 | 83, 150 to 1, 1 | 73, 73 to 1, 2 | 577 to 1 | 0.8 to 0.4 |
| the making ahead, all seven | 1.16, 1.18 s | 0.44, 0.33 s | 0.36, 0.29 s | 2.1 s | 6 ms |

With the transforms starting as the making ahead starts, a transform
whose kernel the making ahead had in hand waited for the rest of that
making, and one whose kernel it had not reached made its own beside it.
On the RTX 5080, in three processes, no first transform took longer
than with nothing made ahead (the adaptive kernel's 226 to 346 ms
against 348, 1080p 420v to 720p BGRA's 104 to 117 against 165). On the
Radeon pair, in two processes each, the making ahead was done in 0.4 s,
and no first transform took longer than 15 ms. Under heavier load from
other work (5 to 14 there, 24 to 72 on the M3 Max), a making took up to
three times as long in one process as in the next, made ahead or not.
A guest's first transform comes once its driver loads, long after its
device is made, when the making ahead is done.

Metal makes a kernel's pipeline where the device makes its pipeline
states, as it is made, which `rvg_scaler_device_scales` needs answered
(0.7 s on the M3 Max with the kernels new to its compiler cache, at
that load; 40 to 60 ms cached), so there the making ahead makes
nothing, and its dispatches, seven then, took 6 ms.

The guest sees nothing of it: it reads and writes no guest memory and no
surface, calls none of the host's functions, changes nothing of the C
API, and its dispatches, which do no work, run on a queue the
transforms' never wait on. What a process shows of it is its own: the
runtime's counts of its device (functions translated, pipelines made),
which neither the guest nor the host reads; a second command queue; a
log line at info once it is done, and one at debug each kernel. And on
Linux, a process that exits while one of its makings is in progress,
in the seconds after its first scaler device is made on a cold cache,
ends at once with the exit's status, its exit handlers registered
before the runtime's device was made not run, as the runtime ends a
process that exits during any making (its exit guard). It has no off
switch: it makes only what the guest's first transforms would make, and
changes nothing a guest or host is given.

**The guest's memory, in place.** A surface whose memory the host lends
a device (`DeviceMemory`) is bound where it is:
- on macOS, the IOSurface the host lends, as a buffer over it
  (`newBufferWithIOSurface:`), made for the transform and dropped before it
  returns: the buffer retains the IOSurface, a hold the guest sees. The
  buffer starts at the surface's first byte and is the allocation's size
  (measured on the M3 Max, macOS 27.0.1, 2026-10-25: the metal crate's
  `a_buffer_over_a_surface_starts_at_the_surfaces_first_byte`);
- on Linux, the host's view of the surface's pages, imported as the GPU
  device imports the guest's memory (`crate::os::host_memory`): the
  pointer, or dma-bufs of the same pages at stock `udmabuf` limits, bound
  into one sparse buffer where the pages need more than one. An import is
  kept by the mapping's identity until the mapping goes, 32 at most
  (`DeviceImports`), so a guest's later transforms of the same surfaces
  import nothing.

A destination is bound so only where the host lends it for writing,
never a surface the guest made read-only for its clients; where its
planes start on 4-byte boundaries and their rows are a multiple of 4
bytes; and where it is not the source's own memory. Anything else is
staged: the source copied in its read lend into a buffer of the GPU's
own, and the destination written into one and copied back in its write
lend, the bytes the window covers alone (`coverage.rs`). So a read-only
destination's lend refuses the write (kIOReturnNotWritable), and its
bytes stay as they were. No GPU work runs inside a lend.

**The wait** is core's deadline, 5 s (`device_wait`). Work not done by
then answers kIOReturnError, and is kept with its buffers and the
device's hold on both surfaces until the GPU is done (`Lingering`); the
accelerator answers kIOReturnError to every transform after it, until
the guest resets the device.

**What the guest is given**, on every GPU:

| capability | ours | why |
|---|---|---|
| formats | `2Planes420` (420v, 420f), `ARGB8888` (BGRA, 32ARGB), `Interleaved422` (yuvs, 2vuy), `2Planes42010` (x420, xf20) | the eight formats the plan lays out, each class as both an input and an output, as the guest publishes them ([devices.md](devices.md)) |
| sizes | 4 to 65536, source and destination | the M3 Max's |
| `MaxHUpscale`, `MaxHDownscale`, `MaxVUpscale`, `MaxVDownscale` | 8, 8, 8 and 4 | the M3 Max's; the device refuses past them as Apple's scaler does ("How this host does a transform") |
| `SourceXYOffsetsWithTiling`, `DestXYOffsetsWithTiling` | yes | windows go anywhere |
| `ScalerVersion` | 0 | No Apple hardware revision describes this scaler, and none of Apple's code that was read uses the key. What a guest does with another value is not measured |
| HDR, ASE, comm API, histogram, compression | none | core writes them 0 (`scaler/capabilities.rs`) |

**What it refuses**, with kIOReturnUnsupported: a side under 4 or over
65536, as the capabilities say, and a plane whose bytes lie past 4 GiB,
which the kernel's 32-bit offsets do not reach. The plan refuses the rest
before it ("How this host does a transform"). Metal or the runtime
failing the work is kIOReturnError, and is logged.

**Results** against core's reference, by the conformance suite's scaler
cases and its `siting` and `map` lines, on every GPU the kernel was run
on (2026-10-25; `the_whole_suite`, `crates/reims-vgpu/src/scaler/gpu/tests.rs`):

| GPU | platform | cases | exact | `map` lines off the plan |
|---|---|---|---|---|
| M3 Max, macOS 27.0.1 | Metal | 113 of 113, 28 refused | 98 | 0.012 at most |
| M3 Max | the Vulkan platform, on MoltenVK | 113 of 113, 28 refused | 99 | 0.012 |
| RX 7900 XTX, RADV (Mesa 26.2.4) | the Vulkan platform | 113 of 113, 28 refused | 98 | 0.012 |
| Radeon 780M, RADV (Mesa 26.2.4) | the Vulkan platform | 113 of 113, 28 refused | 98 | 0.012 |
| lavapipe (Mesa 26.2.4, LLVM 23.1.1) | the Vulkan platform | 113 of 113, 28 refused | 97 | 0.012 |
| RTX 5080 Laptop GPU, driver 615.71.09 | the Vulkan platform | 113 of 113, 28 refused | 97 | 0.012 |

- Every case's largest difference from the reference is a code. The
  worst planes are transfer-up2-420v's, 60.7 dB with 12,783 samples (2.8%)
  a code apart, and transfer-up4-420v's, 66.1 dB with 15,761. Apple's
  filter's weights are 512ths, so a 2x or 4x enlargement can put a value
  on half a code, which the kernels' floats and the reference's doubles
  round apart (a hypothesis). The other cases that differ, 13 or 14 a GPU,
  have 1 to 73 such samples each.
- Against `scaler_transform` alone, before the two passes, the same 7 or 8
  cases moved on each GPU by a few samples a code either side of the
  reference: odd-BGRA-420v from exact to one sample, up4-420v from 78 to
  73, transfer-up2-420v from 12,716 to 12,783.
- The 28 refusals are the pairs Apple's scaler refuses ("How this host
  does a transform").
- The `siting` lines are the same on every GPU, and so are the 88 `map`
  lines, each within 0.012 of a source pixel of the plan's position with a
  spread of 0.095 at most.

The backends the kernel replaced, by the same suite, against the
reference before Apple's filter, windows, code arithmetic and linear light
were modelled (2026-10-25):
- VideoToolbox, on the M3 Max: 104 of 106 cases, 35 refused (the nearest
  filter, sides under 4), 41 exact, a largest difference of 193 codes,
  and a worst plane of 31.3 dB. Against the reference as it is now
  (2026-10-25): 32.2 to 68.7 dB.
- VA-API, on the RX 7900 XTX and the Radeon 780M alike: 102 of 113 cases,
  28 refused. 4:2:2 output and the nearest filter were refused, and two
  cases under 4 a side scored 20.8 and 24.2 dB. 32 cases were exact, the
  largest difference was 211 codes, and the worst plane 20.8 dB.
- Their `map` lines missed the plan by up to 0.75 (VideoToolbox) and 1.0
  (VA-API) of a source pixel: they place a window in whole pixels only.

**Speed.** The suite's benches through the accelerator, the median of 20
after a first: wall time in ms, and the GPU's time in brackets, on
surfaces of the platform's reached in place (`bench_in_place`), after the
conformance suite's cases in the same process (2026-10-25):

| transform | M3 Max, Metal | M3 Max, MoltenVK | RX 7900 XTX | Radeon 780M | RTX 5080 | RTX 5080, first in its process | lavapipe |
|---|---|---|---|---|---|---|---|
| 1080p 420v to 720p BGRA | 2.0 (1.08) | 4.1 (2.20) | 1.6 (1.37) | 3.9 (3.59) | 0.9 (0.51) | 0.9 (0.49) | 30.7 (30.52) |
| 2160p 420v to 1080p 420v | 2.3 (1.07) | 4.4 (2.71) | 3.1 (2.66) | 7.2 (6.46) | 6.1 (5.64) | 1.5 (0.98) | 69.9 (69.65) |
| 720p BGRA to 720p 420v | 0.7 (0.16) | 1.4 (0.84) | 1.2 (1.00) | 0.4 (0.26) | 4.2 (3.81) | 1.0 (0.61) | 2.7 (2.65) |
| 1080p BGRA turned 90 | 1.5 (0.57) | 2.5 (1.12) | 2.1 (1.91) | 1.2 (1.06) | 2.2 (1.75) | 2.3 (1.88) | 16.2 (16.08) |
| 2160p 420v to 2160p BGRA | 1.8 (0.56) | 5.9 (2.79) | 6.6 (6.02) | 4.2 (3.96) | 12.6 (12.15) | 2.9 (2.47) | 35.5 (35.32) |

The M3 Max ran them under a load average of about 100 from other work.
On the RTX 5080 the times follow the process's past: its second column
is the same benches run before the conformance suite's cases, in a build
for that run alone, and after them in eight processes the three slow
ones took 6.0 to 11.3, 1.0 to 4.3 and 2.7 to 12.7 ms ([bugs.md](bugs.md),
B89).

MoltenVK's surfaces are lent, 42 lends a run, as the Vulkan platform on
a Mac copies an IOSurface; the rest are in place.

**Against each host's own scaler**, the kernels were held to a bar: within
1.5 times the faster native scaler's wall time on each GPU and at most
1 ms behind it, and a 4K 420v picture into BGRA within 7.3 ms. Each
case ran in one process, the kernels and the natives in turn on the same
surfaces, after a first to warm each: the median of 8 rounds of 3, wall
ms.
- **M3 Max**: Apple's scaler (TransformSurface through
  IOSurfaceAccelerator, A), `VTPixelTransferSession` (V) and MPS's
  `MPSImageLanczosScale` (L), as
  `crates/reims-vgpu/src/scaler/gpu/tests/native.rs` runs them
  (`the_kernel_against_the_macs_scalers`, ignored; `--run-ignored only`
  under the GPU lock). MPS's `MPSImageBilinearScale` runs too and is
  shown, not counted: two taps a side, it is the floor of a submission.
  The column is the build that landed, at a load average of 10 from
  other work, the quietest the Mac was that day. At a load of 27 its
  scale-csc-BGRA-420v took 2.25 ms against VideoToolbox's 1.28 (1.76
  times; its rounds 1.43 to 2.30 ms), where at 10 it took 1.31 against
  1.22.
- **RX 7900 XTX and Radeon 780M**: VA-API's video processing (radeonsi),
  as the backend the kernels replaced drove it, in a harness of that
  backend's code, not in the tree. The nearest filter's cases have no
  VA-API answer. The 780M's second run came within 0.4 ms of its first in
  every case.
- **RTX 5080 Laptop GPU**, which has no native scaler on Linux: the
  kernels before the two passes (`scaler_transform` alone) against them
  now, in four processes in turn, two of each.

| transform | M3 Max: kernels | fastest native | ratio | RX 7900 XTX: kernels | VA-API | Radeon 780M: kernels | VA-API | ratio | RTX 5080: before | kernels |
|---|---|---|---|---|---|---|---|---|---|---|
| 1080p 420v to 720p BGRA | 1.19 | 1.72 A | 0.69 | 1.27 | 2.95 | 3.62 | 2.54 | 1.43 | 2.45, 2.38 | 0.83, 0.82 |
| 2160p 420v to 1080p 420v | 1.72 | 2.38 L | 0.72 | 2.95 | 7.87 | 9.78 | 7.69 | 1.27 | 12.63, 12.39 | 1.40, 1.46 |
| 720p BGRA to 720p 420v | 0.62 | 0.81 V | 0.76 | 0.93 | 2.08 | 0.42 | 1.51 | 0.28 | 0.96, 0.81 | 1.02, 0.91 |
| 1080p BGRA turned 90 | 0.69 | 1.59 A | 0.43 | 2.34 | 6.36 | 1.48 | 5.97 | 0.25 | 2.37, 2.32 | 2.53, 2.52 |
| 2160p 420v to 2160p BGRA | 1.39 | 3.02 V | 0.46 | 6.73 | 18.97 | 4.45 | 14.20 | 0.31 | 2.77, 2.67 | 2.82, 2.89 |
| down2-420v | 0.55 | 0.42 L | 1.29 | 0.51 | 1.15 | 1.05 | 1.10 | 0.96 | 1.21, 1.07 | 0.42, 0.48 |
| down4-BGRA | 0.55 | 0.37 L | 1.50 | 0.81 | 1.45 | 0.90 | 1.10 | 0.82 | 0.88, 0.84 | 0.82, 0.58 |
| up4-420v | 0.47 | 0.36 L | 1.30 | 0.48 | 1.00 | 0.85 | 0.97 | 0.88 | 1.13, 1.13 | 0.68, 0.18 |
| up8-BGRA | 0.44 | 0.32 L | 1.39 | 0.81 | 1.29 | 0.84 | 1.04 | 0.81 | 0.93, 0.96 | 0.73, 0.35 |
| scale-csc-420v-BGRA | 1.38 | 1.70 V | 0.81 | 1.37 | 3.00 | 3.61 | 2.33 | 1.55 | 2.47, 2.40 | 0.93, 0.76 |
| scale-csc-BGRA-420v | 1.31 | 1.22 V | 1.08 | 2.00 | 4.14 | 3.20 | 3.44 | 0.93 | 18.32, 18.91 | 1.18, 1.02 |
| scale-x420-420v | 1.09 | 1.70 V | 0.64 | 1.38 | 3.72 | 2.44 | 3.13 | 0.78 | 6.87, 6.91 | 0.91, 0.67 |
| crop-420v | 0.30 | 0.31 L | 0.96 | 0.25 | 1.01 | 0.42 | 0.94 | 0.44 | 0.70, 0.69 | 0.56, 0.18 |
| crop-fraction-BGRA | 0.31 | 0.25 L | 1.25 | 0.24 | 0.64 | 0.31 | 0.56 | 0.55 | 0.56, 0.54 | 0.57, 0.09 |
| rect-scale-BGRA | 0.34 | 0.26 L | 1.32 | 0.36 | 1.36 | 0.41 | 0.92 | 0.45 | 0.55, 0.55 | 0.61, 0.16 |
| nearest-up-BGRA | 0.32 | 0.25 L | 1.29 | 0.32 | - | 0.34 | - | - | 0.43, 0.85 | 0.37, 0.10 |
| nearest-down-420v | 0.39 | 0.33 L | 1.18 | 0.28 | - | 0.53 | - | - | 1.00, 1.00 | 0.54, 0.15 |
| odd-420v | 0.33 | 0.27 L | 1.24 | 0.20 | 0.49 | 0.26 | 0.49 | 0.53 | 0.65, 0.66 | 0.47, 0.09 |
| odd-yuvs-BGRA | 0.30 | 0.32 A | 0.97 | 0.20 | 0.31 | 0.26 | 0.29 | 0.88 | 0.52, 0.63 | 0.61, 0.12 |
| turn-aniso-BGRA | 0.28 | 0.28 A | 1.01 | 0.21 | 0.71 | 0.21 | 0.82 | 0.25 | 0.68, 0.66 | 0.51, 0.09 |
| turn-1080-half-420v | 0.62 | 1.71 A | 0.36 | 0.93 | 2.88 | 2.12 | 3.54 | 0.60 | 41.01, 41.50 | 3.28, 0.81 |
| odd-crop-rotate-BGRA | 0.24 | 0.21 A | 1.15 | 0.13 | 0.65 | 0.14 | 0.65 | 0.21 | 0.19, 0.18 | 0.23, 0.06 |
| rotate-scale-BGRA | 0.28 | 0.26 A | 1.07 | 0.20 | 0.30 | 0.23 | 0.32 | 0.71 | 0.37, 0.21 | 0.53, 0.13 |
| transfer-up2-420v | 0.36 | 0.29 L | 1.25 | 0.24 | 0.62 | 0.42 | 0.76 | 0.56 | 0.59, 0.53 | 0.52, 0.12 |
| transfer-up2-BGRA | 0.29 | 0.24 L | 1.19 | 0.22 | 0.38 | 0.25 | 0.48 | 0.52 | 0.39, 0.38 | 0.46, 0.09 |
| transfer-up1.5-BGRA | 0.35 | 0.27 L | 1.33 | 0.32 | 0.50 | 0.38 | 0.57 | 0.66 | 0.41, 0.35 | 0.51, 0.11 |
| transfer-up4-420v | 0.49 | 0.37 L | 1.34 | 0.51 | 1.03 | 0.87 | 1.08 | 0.81 | 0.99, 1.31 | 0.88, 0.23 |
| transfer-up8-BGRA | 0.47 | 0.32 L | 1.46 | 0.91 | 1.28 | 0.89 | 0.98 | 0.91 | 0.85, 2.02 | 2.04, 0.38 |
| transfer-down2-BGRA | 0.38 | 0.28 L | 1.34 | 0.40 | 0.72 | 0.42 | 0.61 | 0.69 | 1.29, 0.74 | 0.94, 0.25 |
| transfer-csc-up-420v-BGRA | 0.64 | 0.89 A | 0.73 | 0.98 | 1.48 | 1.52 | 1.34 | 1.13 | 2.06, 1.20 | 0.78, 0.70 |
| transfer-turn-up-BGRA | 0.31 | 0.31 A | 0.98 | 0.20 | 0.70 | 0.26 | 0.68 | 0.38 | 0.31, 0.40 | 0.40, 0.09 |
| transfer-odd-up-BGRA | 0.41 | 0.27 L | 1.50 | 0.43 | 0.80 | 0.57 | 0.68 | 0.83 | 0.56, 0.74 | 0.63, 0.42 |

The table is of the kernels without the adaptive filter ("The filter"),
a Y'CbCr source's chroma taps of its own, and an RGB source's linear
light. In it every case meets the bar on the M3 Max and the RX 7900 XTX;
on the 780M three 420v reductions do not, 1.1 to 2.1 ms behind VA-API.
On the 5080 every case of a millisecond or more is faster than before, 2
to 50 times for the reductions, but three 1:1 transforms, 2 to 8% slower
in both pairs; its cases under a millisecond vary 2 to 8 times between
processes of one build.

**With the filter, the chroma's taps and linear light**, against the
kernels without them, in turn in one A/B (2026-10-26): on the M3 Max
four processes of each at a load average of 15 to 23, each against the
natives in its own process; on Linux two processes of 4 rounds each,
against VA-API's times in the table above. Six of the M3 Max's cases
leave the bar, each on its ratio alone, wall ms:

| transform | without | with | fastest native | ratio, without then with | behind | with, to Apple's scaler | with, to VideoToolbox |
|---|---|---|---|---|---|---|---|
| up4-420v | 0.50 | 0.76 | 0.45 L | 1.18, 1.70 | 0.31 | 0.84 | 0.85 |
| up8-BGRA | 0.48 | 0.67 | 0.39 L | 1.21, 1.57 | 0.24 | 0.74 | 0.74 |
| transfer-up8-BGRA | 0.55 | 0.64 | 0.39 L | 1.31, 1.70 | 0.26 | 0.70 | 0.71 |
| transfer-odd-up-BGRA | 0.47 | 0.51 | 0.33 L | 1.37, 1.56 | 0.18 | 1.05 | 1.11 |
| turn-aniso-BGRA | 0.36 | 0.44 | 0.29 A | 1.35, 1.52 | 0.16 | 1.52 | - |
| scale-csc-BGRA-420v | 1.88 | 2.07 | 1.27 V | 1.46, 1.65 | 0.81 | 1.61 | 1.63 |

MPS's Lanczos filter is linear and separable, with no adaptive step, so
it does less than Apple's scaler and the kernels do; against Apple's
scaler and VideoToolbox, which do the same work, the 4 and 8 times
enlargements are faster, and transfer-odd-up-BGRA takes 1.05 and 1.11
times their time. Two cases are well behind Apple's own scaler:
turn-aniso-BGRA, 1.52 times its time, and scale-csc-BGRA-420v, 1.61
times its time and 1.63 times VideoToolbox's. The work the latter gains
is its BGRA source's linear light, a conversion each tap, but its time
follows the Mac's load: without them its ratio went 0.82 to 1.56 in the
day's fourteen A/B runs, and at a load of 10 it took 1.31 ms against
VideoToolbox's 1.22. GPU time, without, then with the chroma's taps and
linear light alone, then with all three: up4-420v 0.25, 0.41, 0.47 ms;
up8-BGRA 0.22, 0.24, 0.35; down4-BGRA 0.33, 0.34, 0.38. Every other case
stays within the bar, down4-BGRA at 1.47 times MPS's, and the 1080p and
2160p ones at 0.70 to 1.08 of the fastest native.

**The two cases behind Apple's scaler**, in one A/B of processes in turn
(2026-10-26, three to six of each build, at a load average of 27 to 40
from other work): the kernels as they were above, then working each
pixel out at one place ("Making the kernels"), then decoding through the
table of linear light ("What it computes"). GPU ms, the median over the
processes, and the wall time's median ratio to the fastest native's, its
outputs the same bytes in all three:

| transform | GPU ms | ratio |
|---|---|---|
| turn-aniso-BGRA, to Apple's scaler | 0.136, 0.113, 0.080 | 1.82, 1.68, 1.48 |
| scale-csc-BGRA-420v, to VideoToolbox | 1.53, 1.53, 1.32 | 1.98, 1.84, 1.66 |
| 1080p BGRA turned 90 | 0.593, 0.492, 0.351 | 0.94, 0.89, 0.78 |
| 720p BGRA to 720p 420v | 0.296, 0.269, 0.202 | 1.15, 0.89, 0.79 |
| down4-BGRA | 0.375, 0.373, 0.268 | 1.44, 1.37, 1.23 |
| transfer-up8-BGRA | 0.347, 0.324, 0.301 | 1.56, 1.61, 1.33 |
| transfer-odd-up-BGRA | 0.224, 0.215, 0.184 | 1.66, 1.57, 1.40 |

Where their time goes:
- **turn-aniso-BGRA** turns 320x240 into 240x160 in `scaler_transform`,
  each pixel a sum of decoded source pixels, and the table took 29% off
  its GPU time. What is left is the Metal submission's: the case's wall
  time ran 0.21 to 0.36 ms past its GPU time, where Apple's scaler, which
  is not a Metal submission, took 0.21 to 0.30 ms in all, and MPS's
  bilinear filter, two taps a side, took 0.30 to 0.73 ms on 2160p 420v to
  1080p 420v. At a load of 4 to 12, the kernels working each pixel out
  at one place took 1.13 times Apple's scaler's time, the median of three
  processes.
- **scale-csc-BGRA-420v** is the lines and then both planes across, in
  two bands of rows. Built with one pass's kernel doing nothing, the
  kernels working each pixel out at one place took 0.52 to 1.11 ms of
  GPU time for the lines and 0.28 to 0.49 for the pass across; the table
  took 0.21 ms off the lines. Its GPU time is one of two in a process, as
  the 1080p and 2160p cases' are: about 1.5 ms or 0.85 for the same
  build, and 1080p 420v to 720p BGRA's 1.13 or 0.65. At a load of 4 to
  12 two of three processes took the faster, and the kernels working
  each pixel out at one place took 1.07 times VideoToolbox's time, the
  median of the three (0.91 to 1.17).

At a load of 8 to 24, in three processes of each, turn-aniso-BGRA took
1.02 to 1.05 times Apple's scaler's time with the table, against 1.16
to 1.41 without it, and scale-csc-BGRA-420v 1.09 to 1.36 times
VideoToolbox's, against 1.10 to 1.57, as its GPU time took one or the
other of its two.

The table takes the decode's powers out of every RGB source's transform
in linear light, and the kernels whose RGB decode has no powers in it
run faster on the M3 Max even where the transform does not go into
linear light: 1080p BGRA turned 90, one source pixel a destination
pixel, went from 0.492 to 0.351 ms of GPU time.

On the RX 7900 XTX every case stays at 0.26 to 0.72 of VA-API's time:
2160p 420v to 1080p 420v went from 2.47 to 2.67 ms of GPU time, and
1080p 420v to 720p BGRA from 1.14 to 1.19. On the 780M no case leaves
the bar: 2160p 420v to 1080p 420v went from 7.72 to 8.52 ms (7.09 to
7.71 of GPU time), 1080p 420v to 720p BGRA from 3.53 to 4.05 (3.35 to
3.77), and scale-csc-420v-BGRA from 4.48 to 4.39. On the RTX 5080 the
scaled cases took 0.02 to 0.10 ms more GPU time, and the 1:1 ones the
same or less. Where the 780M's time goes, what was tried, the M3 Max's
two cases slower than Apple's scaler and the 5080's slower transforms
are [bugs.md](bugs.md), B89.

## The wire-up

The scaler device goes into a VM the way the VideoToolbox device does: a
QEMU device in C, the C API it calls, and the Rust glue behind it.

| what | where |
|---|---|
| the QEMU device, `apple-m2scaler-pci` | `vendor/qemu/hw/display/apple-m2scaler-pci.c`, with `vendor/qemu/hw/display/Kconfig` (`MAC_PVG_SCALER_PCI`), `meson.build` and `trace-events` |
| the C API, `rvg_scaler_device_*` | `crates/reims-vgpu-c/src/scaler.rs`, in the committed `crates/reims-vgpu-c/include/reims_vgpu.h` |
| the Rust glue, `ScalerDevice`, over core's | `crates/reims-vgpu/src/scaler.rs`; its backend picked in `crates/reims-vgpu/src/video/backend_choice.rs` |
| a surface's bytes lent for reading | `crates/reims-vgpu/src/platform.rs` (`lend_surface_bytes_for_reading`), `crates/reims-vgpu-vulkan/src/os/surface.rs`, `crates/reims-vgpu-metal/src/os/surface_bytes.rs` |

**The QEMU device.** Vendor 0x106b, device 0x1a0b, written back over the
ID virtio-pci derives, as `apple-videotoolbox-pci` does; the subsystem
106b:1a0b and class 0x1200 are Apple's ([devices.md](devices.md), "The
two devices"). Modern only, no device features, no device configuration,
one queue of 256 entries (ours: Apple's size is not read), MSI-X with two
vectors, the capabilities linked in ascending address order. It has no
properties: the library picks its backend ([qemu.md](../qemu.md), "The M2
scaler device"). Each element's readable
part, at most 64 KiB, is copied whole into one buffer for
`rvg_scaler_device_process`, which answers it in place:
- `RVG_SCALER_REPLY`: the buffer goes into the writable part, with a used
  length of its size;
- `RVG_SCALER_NEEDS_RESET`: `virtio_error` with Apple's reason, which sets
  DEVICE_NEEDS_RESET and raises the configuration interrupt; the element
  is detached with no reply, and the queue is served no more until the
  guest resets the device;
- `RVG_SCALER_END_VM`: QEMU ends, as Apple's host ends its process.

A readable part over 64 KiB is a needs-reset before it is copied, ours
([devices.md](devices.md), "Where we differ from Apple").

DRIVER_OK calls `rvg_scaler_device_start`. Where it cannot make an
accelerator, a bottom half sets DEVICE_NEEDS_RESET once the status is
written: `virtio_set_status` writes the status after the device's hook
returns, so a needs-reset set inside the hook would be lost. The reset
calls `rvg_scaler_device_reset`, once the worker is done (below).

**The worker.** `rvg_scaler_device_process` runs on a thread of the
device's own, not on QEMU's main loop, since a transform takes the GPU
for as long as it takes (below, "How long a transform takes"):
- the queue's notification pops one element, copies its readable part and
  hands it to the worker;
- the worker has the library answer it, and hands the answer back through
  a bottom half, which on the main loop writes the reply and returns the
  element, or detaches it and sets DEVICE_NEEDS_RESET, and only then pops
  the next. So the elements are answered one at a time, in the guest's
  order, as Apple's handler answers them ([devices.md](devices.md), "The
  M2 scaler device"). An end of the VM aborts on the worker;
- a reset, and the device's unrealize, wait for the element with the
  worker and discard its answer: the queue is emptied after the reset, so
  nothing of it reaches the guest's new ring. The wait holds the BQL, and
  the worker never takes it, so it cannot deadlock; a reset during a long
  transform waits that transform out, the only stall left;
- DRIVER_OK's `rvg_scaler_device_start` stays on the main loop: it waits
  for core's accelerator lock, which a transform holds once its two
  surfaces are held, only where a guest sets DRIVER_OK again with no
  reset, an element still with the worker.

**How long a transform takes**, measured in an x86_64 guest ("In a
guest", below) with the scaler probe's timing mode, five runs a
transform, while a script polled QMP's `query-status` every 20 ms
(2026-10-24, the RX 7900 XTX, `renderD128`). The last column is of a
build that answered the elements on the main loop, where QMP's wait is
the host's part of each transform:

| transform, BGRA | the probe's runs | QMP's longest wait during them, answered on the main loop |
|---|---|---|
| 3840x2160 copied | 395, 143, 63, 57, 54 ms | 229 ms (the first), then 35 to 124 ms |
| 3840x2160 turned 90 degrees | 213, 74, 66, 56, 54 ms | 34 to 71 ms |
| 16384x16384 turned 90 degrees, the largest the capabilities offer | 11.3, 3.5, 3.3, 3.4, 3.4 s | 4.9, 3.5, 3.3, 3.4, 3.3 s |

A guest's first transform of a surface is the longest: of the 16384
turn's first 11.3 s, the host's part was the 4.9 s QMP waited, so the
rest went in the guest, which maps the surfaces' pages first. The later
runs took 3.3 to 3.5 s on the host. With the worker, the same runs: the
16384 turns took 10.6 s, then 3.8 to 4.0 s, and QMP answered within 23 ms
throughout them; the 3840x2160 ones took 59 to 197 ms, with no QMP wait
over 20 ms. Why the worker's 16384 runs were a tenth slower is not
measured: the GPU is shared. None of QMP's other waits fell during a
transform.

**The guest's memory.** The capabilities' address and the side buffers'
lists are guest-physical addresses, which the guest kext takes from
`pmap_find_phys` and `getPhysicalSegment` for a PCI device. The library
reaches them through three callbacks of the host's, `is_ram`, `read` and
`write`, and QEMU answers them on the worker, on the device's DMA address
space (`VirtIODevice.dma_as`): each piece of the range mapped with a
`MemoryRegionCache`, at most eight, every one RAM and not a RAM device or
none, and the bytes copied through the same caches, so what is checked is
what is copied and no access dispatches to a region that would take the
BQL. Memory that is not RAM (a device's registers) is refused, as Apple's
host refuses it: a needs-reset. An IOMMU's address space is not
translated: behind one every such request would be refused; the guests
have none. The video glue's way, `lend_surface_bytes`, reaches only an
IOSurface's pages, which the mapper recorded when the surface was made;
these addresses belong to no surface.

**The C API**, six functions after the video device's, so the header's
first 41 keep their places: `rvg_scaler_device_create`, `_release`,
`_scales`, `_start`, `_process` and `_reset`, with
`RvgScalerDeviceConfig`, `RvgScalerGuestMemory` and the `RVG_SCALER_*`
constants ([api.md](../api.md)). `_create` makes the device on every host;
`_scales` says whether it has a GPU to transform on, and QEMU's
`validate_features` refuses the guest's features where it has not ("A host
without a GPU").

**The Rust glue.** `ScalerDevice` holds core's scaler device over the
video device's `MapperSurfaces`, so a transform's two surfaces are held
by mapping ID through the config's IOSurface device, or the process's
latest where it names none, as QEMU's does, as a decoder's are. Its
backend is the library's kernel, made once a process at the device's
first use (`gpu_backend`) on the GPU the library draws with ("The
kernel"), whatever the video device's backend is. Core reads the
source through `GuestSurface::read`, which `MappedSurface` answers with
the platform's `lend_surface_bytes_for_reading`: the bytes as
`lend_surface_bytes` lends them, in place or gathered from the guest's
pages, with nothing written back, and a surface the guest made read-only
for its clients lent too. The destination is written through
`lend_surface_bytes`.

**Measured** on a Linux host with an RX 7900 XTX and a Radeon 780M,
2026-10-24, `x86_64-softmmu`, with no guest, on the VA-API backend the
kernel replaced:
- **Realized paused** on q35 under KVM, alone, and beside `apple-gfx-pci`,
  `apple-gfx-iosurface-pci` and `apple-videotoolbox-pci`: 106b:1a0b, class
  0x120000, subsystem 106b:1a0b, BAR 1 (4 KiB, MSI-X) and BAR 4 (16 KiB,
  64-bit prefetchable). The library opened VA-API on renderD128, and on
  renderD129 when told that node then, and logged the formats it reads
  (420v, 420f, x420, xf20, yuvs, 2vuy, BGRA, 32ARGB) and writes (the same
  but yuvs and 2vuy).
- **Through a qtest driver**, 29 of 29 checks: QEMU's transport features
  `0x101_3000_0000`, queue size 256; the capabilities answered in place
  across descriptors, 0x288 bytes written into guest RAM at an unaligned
  address with the count put back, `R+0x10` left 0, version 0, the
  classes 0x08 (`2Planes420`), 0x0d (`ARGB8888`) and 0x1e
  (`2Planes42010`), sizes to 16384, and no HDR, ASE or comm API; a
  transform from mapping IDs no IOSurface device has answered
  kIOReturnNotFound with its completion and both events cleared; side
  buffers gathered from two entries and scattered over two, the pair's
  addresses left the guest's; an unknown type replied unchanged, 64 KiB
  of one across two descriptors too; a short type 1, one naming the
  IOAPIC's page, and a readable part of 64 KiB and a byte (QEMU's
  `Request too large`, no element used) each set DEVICE_NEEDS_RESET with
  no reply, and the device served again after the driver's reset; a list
  of nine entries ended QEMU (SIGABRT, `Security Assertion Hit: `). Two
  are the worker's: three requests on one notification answered in their
  order, and a reset at once after a notification putting nothing on the
  new ring.

## In a guest

The x86_64 port's scaler kext goes in beside its video kext, bounded
25.6.0 to 25.99.99 as the video kext is, on a QEMU with both devices
([qemu.md](../qemu.md), "The M2 scaler device"). The guests measured here
are x86_64 macOS 26.7.1 (25G241) on q35 under KVM, on a Linux host with
an RX 7900 XTX and a Radeon 780M, 2026-10-24. The device transformed then
on the RX 7900 XTX's VA-API video processing (`renderD128`), the backend
the kernel replaced, which offered three classes, sides to 16384 and
ratios of 8, and did 2vuy in and 32ARGB out. No guest has run on the
kernel yet ([bugs.md](bugs.md), T18).

### A guest that hides its VM

IOSurfaceAccelerator picks its driver by whether it runs in a VM, read
from Apple's code: where `kern.hv_vmm_present` reads 1, and
`debug.iosa.simulated_model` is not set, `IOSurfaceAcceleratorCreate`
opens `AppleM2ScalerParavirtDriver`; otherwise it opens the hardware's
`AppleM2ScalerCSCDriver`. The guest's own x86_64 slice, read in 25G241's
x86_64 shared cache (2026-10-24), does the same: it holds both class
names, both `IOServiceMatchingfailed` messages and both sysctl names, and
`IOSurfaceAcceleratorCreate` tests the VM flag before each lookup.

reims-util's OpenCore, which makes reims-ctl's x86-64 VMs' images, hides
the VM by default: two kernel patches swap `kern.hv_vmm_present` with
`kern.hibernatecount`, so that iCloud and the App Store sign in. With the
kext's service named as Apple's is, such a guest never opens the scaler:
measured, the kext bound and the host served its capabilities, and
`IOSurfaceAcceleratorCreate` returned kIOReturnNoDevice with
`IOServiceMatchingfailed for: AppleM2ScalerCSCDriver` on stderr.

**So the x86_64 kext answers to both names.** Its driver's class is
`AppleM2ScalerCSCDriver`, ours, a subclass of
`AppleM2ScalerParavirtDriver` that adds nothing, and its personality's
`IOClass` names it. `IOServiceMatching` by either name finds the one
service. The alias is in the kext, not a patch of the framework's test,
which would rewrite Apple's code in every process, keyed to one build's
bytes.

**What the hidden path does differently**, read in IOSurfaceAccelerator
from create to finalize. Every other call is the same on both paths:

| where | the hidden path | the kext's user client |
|---|---|---|
| `IOSurfaceAcceleratorCreate` | looks the service up as `AppleM2ScalerCSCDriver` | found, through the alias |
| create | then calls selector 10, set client property, with `{2, 50000, 500000}`, 0x18 bytes, and prints `Failed to set client property` on an error | selector 10 takes 0x18 bytes. Apple's kext answers kIOReturnUnsupported; the x86_64 port answers success, as a VM's framework answers for it |
| `IOSurfaceAcceleratorSetProperty` | calls selector 10 with the parsed properties, and returns its status | the same seam |

The calls both paths make: selector 1, a transform, 0x1b0 bytes; 4, the
custom filter, 0x20 bytes; 5, the ID, no input and one scalar out; 7, the
histogram, 8 bytes, after reading `IOSurfaceAcceleratorCapabilitiesDict`
from the service; 8, the diagnostics, 8 bytes; 9, the estimation, 0x10
bytes. The kext's client table has each with the same size. It has no
method for selectors 0 (capture), 2 and 3 (the aborts) and 6 (kernel
tests), on either path, nor for 11, which only the host side calls.

**What else names the class**, in 25G241's x86_64 shared cache: three
images, each with one copy of the string (a search of the cache's files,
2026-10-24), IOSurfaceAccelerator and two more. What the alias changes
for the other two, read in their x86_64 code:
- **QuartzCore**, `CA::ScalerUtil::get_io_service`:
  `AppleM2ScalerCSCDriver`, then `AppleM2ScalerParavirtDriver`, with no
  VM test, so it finds the device's service whether the kext has the
  alias or not. `CA::OGL::MetalContext::copy_surface` tests that a
  service was found, and `CADeviceSupportsFilmGrain` reads
  `IOSurfaceAcceleratorCapabilitiesFilmGrain` from the service's
  `IOSurfaceAcceleratorCapabilitiesDict`, 0 in ours.
- **HDRProcessing**, `queryMSRCapabilities`: `AppleM2ScalerParavirtDriver`
  where `IsVirtualized()` (which reads `kern.hv_vmm_present`), else
  `AppleM2ScalerCSCDriver`, then bools and ints of its capabilities
  dictionary. Where the VM is hidden it reads ours through the alias, as
  it does where the VM shows: no HDR.

No kext in 25G241's x86_64 kernel collections, and no plist of its
payload, matches the class; the one that does, AppleM2ScalerCSC.kext, is
arm64's.

**Measured.** With the VM hidden (`kern.hv_vmm_present: 0`),
IOSurfaceAccelerator found the service by `AppleM2ScalerCSCDriver`,
selector 10 answered 0, and every case passed. With the VM shown, it
found the same service by `AppleM2ScalerParavirtDriver`, and the same
cases passed. The probe prints the service's class, so it says `service:
AppleM2ScalerCSCDriver` in both ("What the guest did", below).

### What the guest did

Measured in a guest that sees its VM:
- **The kext** loaded, matched 0x1a0b106b and published
  `IOSurfaceAcceleratorCapabilitiesDict`: 134 keys, the three classes
  this host offers as both input and output, version 0, rotation 1, sizes
  to 16384. The host served one type 1 request, at DRIVER_OK's start.
- **IOSurfaceAccelerator**: every probe case passed, 31 of 31 with none
  refused: copies, scales up and down, windows, conversions between 420v,
  420f, x420, BGRA and 32ARGB, yuvs and 2vuy read, odd sizes, all eight
  orientations. The worst plane was 41.6 dB (321x241 to 161x121), the rest
  46 dB or more. The host logged 31 type 2 requests, each status 0, one
  per case.
- **Conversions Apple's scaler refuses passed then**: 2vuy into 420v, and
  BGRA into 32ARGB (above, "Apple's scaler, measured"). The guest's
  framework does not hold a transform to the classes the capabilities
  offer, so the host's refusal is what the guest sees: the device now
  refuses both, as Apple's scaler does ("How this host does a
  transform").
- **The matrix** of 420v into BGRA was 601's, the surfaces tagged BT.709
  or not: the mapper sends no colour attachments ("How this host does a
  transform"), where the M3 Max's own scaler took 709 from the tags.
- **On the GPU.** The QEMU process's one amdgpu client on renderD128
  (Mesa's winsys shares one device per node, so the decoder and the
  scaler are one client) took 20.6 ms of compute engine time while the
  31 transforms ran, in steps with them, and no gfx time; its VCN time
  (`drm-engine-enc`) moved once in that window, for a 3840x2160 HEVC
  decoder the guest had opened on its own before the probes ran, as its
  Aerial wallpaper does.
- **VideoToolbox's pixel transfers do not use the device on x86_64.** The
  probe's four `VTPixelTransferSession` and `VTPixelRotationSession` cases
  passed (709's matrix for the conversion, VideoToolbox tagging its
  buffers), but the host saw no request while they ran: VideoToolbox does
  them in the guest. Whether arm64 VideoToolbox sends its transfers to
  the device, and whether x86_64's can be led to, is not measured.
- **Decode still passes beside it**: H.264 decode matched, 150 of 150
  frames on both streams.

Measured again with the kext's `AppleM2ScalerCSCDriver`, in a guest that
hides its VM and in one that shows it:
- **Both guests used the scaler**: 31 of 31 cases, every case's line but
  its time the same as above, and 39 type 2 requests on the host in each.
- **The ratios.** Of the probe's eight `limit` cases, the host refused
  the five past its offered 8 with kIOReturnBadArgument, which the guest
  saw as each transform's status, and did the three at 8 or less: 8 up
  across, 8 down the frame (where Apple's scaler, offering 4, refused),
  and a quarter turn at 1 after the turn. The quarter turn to 16 up
  across was refused as past `MaxHUpscale`, so the ratio was taken after
  the turn, as Apple's is ("The ratios").
- **H.264 decode** still matched, 150 of 150 on both streams, in both.

## A host without a GPU

The scaler device is made on every host, so a VM given it starts wherever
QEMU does. Its backend is the kernel on the GPU the library draws with
("The kernel"), on any GPU Metal or the Vulkan runtime opens, so an
NVIDIA GPU, which has no video processing of VA-API's, scales too. Where
Metal has no device, the Vulkan runtime opens none, or either refuses the
kernel (`gpu_backend`; the reason in words, as `Metal has no default
device`):
- `ScalerDevice::scales` is false (`rvg_scaler_device_scales` in C), and
  the library logs, at info, `scaler device: <the reason>: it refuses the
  guest's features, so the guest's driver does without it`;
- QEMU's `validate_features` refuses the guest's features (`-ENOTSUP`, and
  the trace event `apple_m2scaler_features_refused`), so virtio-pci leaves
  FEATURES_OK unset in the status the guest writes;
- the guest kext reads the status back, finds FEATURES_OK unset, writes
  FAILED and gives up its start before DRIVER_OK and before its type 1
  request, and registers no service, so IOSurfaceAccelerator finds no
  scaler, as in a VM without the device ([devices.md](devices.md), "Where
  we differ from Apple");
- a driver that went on to DRIVER_OK anyway would get a needs-reset from
  the start, as Apple's host gives one where it cannot make its
  accelerator, and every later request one too.

`crates/reims-vgpu/src/scaler/tests.rs` and the C API's tests hold the
device without a GPU (`ScalerDevice::without_gpu`); the C harness checks
its start fails. **Measured before the kernel**, when the device took
VA-API's video processing and NVIDIA's backend had none, on the RTX 5080
Laptop GPU (driver 615.71.09, 2026-10-25), with no guest: QEMU realized
the device alone and beside `apple-gfx-pci`, `apple-gfx-iosurface-pci`
and `apple-videotoolbox-pci`, paused under KVM, and the library logged
the line above. A qtest driver that went as the guest kext goes
(ACKNOWLEDGE, DRIVER, `VERSION_1`, then FEATURES_OK) read the status
back as 0x3, FEATURES_OK not kept, and QEMU traced
`apple_m2scaler_features_refused`; on the RX 7900 XTX the same driver
read 0xb, FEATURES_OK kept. In an x86_64 macOS 26.7.1 guest on the same
GPU (2026-10-25) with both devices and both kexts, QEMU started, the
library logged the line above, and QEMU traced
`apple_m2scaler_features_refused` once, the guest's kext writing the
status 0x3 and then 0x83 (FAILED) and logging `[IOSA][Boot ]
initializedDevice failed` on the kernel's console. The kext stayed
loaded with no `AppleM2ScalerParavirtDriver` in the registry, the scaler
probe found no accelerator, and the video device decoded and encoded
every stream on NVIDIA as it does without the scaler device. On the RX
7900 XTX and the Radeon 780M the same guest's kext wrote 0xb and then
0xf, and registered its service.

**Measured with the kernel** on the same RTX 5080 (2026-10-25), with no
guest: QEMU realized the device paused under KVM, alone and beside
`apple-videotoolbox-pci`, and the library logged `scaler device:
transforms on NVIDIA GeForce RTX 5080 Laptop GPU`. The same qtest driver
read the status back as 0xb: FEATURES_OK kept.

## Not yet here

- `IOSurfaceAcceleratorScalerVersion` is 0 here; what a guest does with
  another value is not measured.
- **Options a kernel could do, refused.** `ZeroFill`, `BorderFill`,
  `AlphaOverride`, `WriteOnlyMode` and `BorderFillOddSizeYCbCrEnable` ask
  for writes the kernel could make: a fill around the window, a constant
  alpha, a destination written without being read. They stay
  kIOReturnUnsupported until Apple's scaler's output with each is
  measured. `HDREnable`, `EnhanceEnable`, `FilmGrainEnable`,
  `ColorRemapMode` and the histogram need models of Apple's hardware that
  nothing here has.
- **A word written in part.** The kernel reads a word the destination
  window covers in part and writes it back whole, so a write of the
  guest's to its other bytes while the transform runs may be lost. How
  Apple's scaler writes next to a window, at that grain, is not measured.
- **No guest has used the kernel** ([bugs.md](bugs.md), T18).
