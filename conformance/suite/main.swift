// The battery’s running order. Every case invocation lives here and nowhere
// else: Swift permits top-level statements in `main.swift` alone, so a case that
// is not called from this file cannot run at all. The "a case nobody called
// reports nothing, and the totals do not notice" trap is closed by construction
// rather than by remembering.

import Metal
import Foundation
import IOSurface


// `library` is lazily initialised now that it lives in a declaration
// file, so force the build before the line that reports it.
_ = library
report("shader_compile", true, "runtime library built")

for f in formats {
    linearAliasCase(f, 64, 16, pitch: .tight, sampler: false)
    linearAliasCase(f, 64, 16, pitch: .tight, sampler: true)
}

// The label-layer geometries, from a driven boot's own census. Their pitches
// are the guest's, so a device whose `minimumLinearTextureAlignment` forbids
// one skips it -- Metal itself would reject the descriptor there, and the
// alignment differs by device (16 on an M-series host, 256 on Apple's
// paravirtual one), so a literal pitch is not portable and must not be a
// failure where it is simply not expressible.
linearAliasCase(formats[1], 54, 16, pitch: .exact(64), sampler: false)

linearAliasCase(formats[1], 54, 16, pitch: .exact(64), sampler: true)

linearAliasCase(formats[1], 218, 16, pitch: .exact(256), sampler: false)

linearAliasCase(formats[1], 85, 85, pitch: .exact(128), sampler: false)

linearAliasCase(formats[0], 128, 16, pitch: .exact(128), sampler: false)

linearAliasCase(formats[4], 60, 8, pitch: .exact(256), sampler: false)

// The same shapes with the padding expressed against whatever this device's
// alignment actually is, so every host runs a padded-pitch case whatever its
// limit. One alignment unit of padding beyond tight, and then two.
for f in formats {
    for rows in 1...2 {
        linearAliasCase(f, 54, 16, pitch: .padded(rows: rows), sampler: false)
    }
}

for f in formats {
    incrementalCase(f, viaFragment: false)
}

renderTargetCases()

mipCase()

mipBlitCase()

vertexBufferCase()

fragmentBufferCase()

if texPipeline == nil {
    report("fragsample_pipeline", false, "tex_fs pipeline would not build")
} else {
    fragmentAliasCase(formats[1], 54, 16, pitch: .exact(64))    // the label geometry
    fragmentAliasCase(formats[1], 218, 16, pitch: .exact(256))
    fragmentAliasCase(formats[1], 85, 85, pitch: .exact(128))
    fragmentAliasCase(formats[3], 64, 16, pitch: .tight)        // rgba8, tight
    fragmentAliasCase(formats[4], 60, 8, pitch: .exact(256))    // bgra8, padded
    // And once more with the padding derived from this device's own limit, so
    // a host whose alignment skipped the literal pitches above still runs a
    // padded fragment-sampled alias.
    for f in [formats[1], formats[4]] {
        fragmentAliasCase(f, 54, 16, pitch: .padded(rows: 1))
    }
}

// The glyph-atlas lifecycle from section B, now through the stage the type
// layer actually uses. This is the arm that can see a defect on the draw path,
// and for `a8Unorm` it is the only arm that produces a reading at all.
for f in formats {
    incrementalCase(f, viaFragment: true)
}

// The same lifecycle again, filled by `replaceRegion:` rather than through a
// buffer alias — section H says why the two are different rails. Declared
// there, invoked here, because the fragment arm needs `texPipeline`.
for f in formats {
    replaceRegionCase(f, viaFragment: true)
}

// Tight and padded, read both ways, so the pitch is the only thing that varies
// between a passing case and a failing one.
// 256 and 250 rather than the device's alignment and six under it: at one byte
// per texel both round up to a tight pitch of 256 on a device that aligns to 16
// and on one that aligns to 256, so the pair means "no padding" and "six bytes
// of padding" on either host and the two runs pair up by name.
for viaFragment in [false, true] {
    offsetOracleCase(256, 16, pitch: .tight, viaFragment: viaFragment)
    offsetOracleCase(250, 16, pitch: .tight, viaFragment: viaFragment)
    offsetOracleCase(218, 16, pitch: .tight, viaFragment: viaFragment)
    offsetOracleCase(54, 16, pitch: .padded(rows: 1), viaFragment: viaFragment)
}

for w in [256, 250, 218, 64, 60, 54, 63, 57] { targetWidthCase(w, 16) }

// A width that divides the threadgroup and one that does not, in both axes.
gridBoundsCase(64, 16, 8)

gridBoundsCase(218, 16, 8)

gridBoundsCase(54, 15, 8)

gridBoundsCase(57, 9, 8)

gridBoundsCase(31, 31, 16)

cpuWriteAfterRenderCase(64, 32, secondPass: false)

cpuWriteAfterRenderCase(64, 32, secondPass: true)

cpuWriteAfterRenderCase(256, 64, secondPass: false)

cpuWriteAfterRenderCase(256, 64, secondPass: true)

offsetAliasCase(formats[1], 54, 16, tiles: 4)

offsetAliasCase(formats[3], 32, 32, tiles: 3)

for f in formats {
    replaceRegionCase(f, viaFragment: false)
}

sharedTargetCases(60, 32)

sharedTargetCases(256, 64)

sharedTargetCases(1000, 40)

sharedTargetCpuSeedCase(60, 32)

sharedTargetCpuSeedCase(256, 64)

sharedTargetCpuSeedCase(1000, 40)

sharedTargetGlyphCase(60, 32)

sharedTargetGlyphCase(256, 64)

sharedTargetGlyphCase(1000, 40)

iosurfaceCases(60, 32)

iosurfaceCases(256, 64)

iosurfaceCases(1000, 40)

for (cw, ch) in [(60, 32), (256, 64), (1000, 40)] {
    cpuWriteThenSampleCase(cw, ch, iosurface: false)
    cpuWriteThenSampleCase(cw, ch, iosurface: true)
}

blitAfterRenderCase(1024, 768)

blitAfterRenderCase(1920, 1080)

blitPipelinedCase(1024, 768, frames: 8)

blitBufferBackedCase(512, 512)

print("SUMMARY cases=\(ran) failures=\(failures) skipped=\(skipped)")

print("DEVICE name=\(dev.name) unified=\(dev.hasUnifiedMemory)")

exit(failures == 0 ? 0 : 1)
