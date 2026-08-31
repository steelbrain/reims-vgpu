// A self-verifying Metal conformance battery.
//
// Why this exists. Every result this project has about Maps' missing type layer
// was scored by opening a screenshot, and a screenshot names no seam: it says
// "labels absent" for a wrong pitch, a wrong swizzle, a wrong render-target
// round trip and a dropped draw alike. This binary asks the API directly. Each
// case computes a value the CPU can predict exactly, the GPU produces it, and
// the CPU compares -- so a failure names the case, the expected bytes and the
// bytes that came back.
//
// It is built for two hosts and the comparison between them is the point: run
// it on a native macOS host to establish that a case's expectation is what
// Metal actually does, then run the same binary in the guest. A case that
// passes natively and fails in the guest is a named device defect. A case that
// fails on both is a wrong expectation in the suite, not a finding.
//
// Every case reports on one line:  CASE <name> PASS|FAIL <detail>
// and the process exits non-zero if any case failed.
//
// The tree, the two runners and what a case has to do to be a gate are in
// ../README.md.

import Metal
import Foundation
import IOSurface

// `MTLCreateSystemDefaultDevice` answers nil in a session with no window server
// attached -- which is every ssh session, and this battery is driven over ssh on
// purpose. `MTLCopyAllDevices` enumerates the same devices without that
// requirement, so the fallback is the contract and not a workaround.
//
// Initialised by a closure rather than bound by a top-level `guard` because only
// `main.swift` may hold top-level statements once the suite is more than one
// file. The diagnostics and the exit code are unchanged.
let dev: MTLDevice = {
    guard let d = MTLCreateSystemDefaultDevice() ?? MTLCopyAllDevices().first else {
        print("CASE device FAIL no Metal device")
        fflush(stdout)
        exit(2)
    }
    return d
}()

let queue: MTLCommandQueue = {
    guard let q = dev.makeCommandQueue() else {
        print("CASE queue FAIL \(dev.name) would not make a command queue")
        fflush(stdout)
        exit(2)
    }
    return q
}()


var failures = 0

var ran = 0

/// A case the device's own reported limits make inapplicable -- a pitch its
/// `minimumLinearTextureAlignment` forbids, say. Not a failure: Metal would
/// reject the descriptor on any host that reports the same limit, so there is
/// nothing here for a device to get wrong.
var skipped = 0

func skip(_ name: String, _ why: String) {
    skipped += 1
    print("CASE \(name) SKIP \(why)")
    fflush(stdout)
}

/// The device rails a case claims to move, named by the counters this device
/// reports them under.
///
/// A passing case that never reached the rail it was written for is the worst
/// reading this battery produces, because it is indistinguishable from coverage.
/// Three attempts at the regression case for the unordered host read passed on
/// the *broken* build, each for that reason: a `.bgra8Unorm` pair at 512x512 is
/// staged through a buffer by the guest driver, so the case exercised a rail
/// that was never in question and reported green. It took a temporary probe
/// inside the device to find out.
///
/// `verdict.py` reads these against the device's own fail log and says
/// NOT-COVERED where a claimed counter never moved. The check is per *run* and
/// not per case -- nothing carries a case name across into the device -- so it
/// answers "did anything in this run reach that rail", which is the question
/// that was being got wrong.
///
/// A case that claims nothing is unclaimed, not covered. Claiming a counter
/// nobody verified moves would make this table read as coverage while measuring
/// spelling, so claim one when it has been watched, and not before.
func claims(_ name: String, _ counters: String...) {
    for counter in counters {
        print("ROUTE \(name) \(counter)")
    }
    fflush(stdout)
}

func report(_ name: String, _ ok: Bool, _ detail: String) {
    ran += 1
    if !ok { failures += 1 }
    print("CASE \(name) \(ok ? "PASS" : "FAIL") \(detail)")
    fflush(stdout)
}

func finish() -> Never {
    print("SUMMARY cases=\(ran) failures=\(failures) skipped=\(skipped)")
    print("DEVICE name=\(dev.name) unified=\(dev.hasUnifiedMemory)")
    fflush(stdout)
    exit(failures == 0 ? 0 : 1)
}
