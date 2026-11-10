# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Compile a consumer against the SDK IOMobileFramebufferLegacy stand-in and check it.

The stand-in (sdk/include/graft/IOMobileFramebufferLegacy.h) is what an arm64 display
consumer subclasses; no other test in the repo compiles it. This builds a minimal concrete
subclass -- our own code, nothing fetched or taken from any Apple binary -- with the kext
flags for both architectures and checks two invariants the arm64 display synthesis rests on:

* The class reserves GRAFT_IOMFB_VTABLE_RESERVE trailing virtual slots, so a consumer's
  compiled vtable runs past the live IOMobileFramebufferLegacy's word count and Graft can
  fill it in place (docs/guides/synthesized-class.md). We count the reserved stubs.

* Overriding the class's methods emits exactly the selectors (name + parameter manglings)
  the live kernel names, so each override resolves to the right live slot. We freeze the 31
  manglings the stand-in's own declarations produce; a parameter-type edit that shifted a
  mangling would break the match with the live class, and this catches it. The strings are
  the compiler's deterministic output for our header, re-derived here, not Apple data.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "sdk/include/graft/IOMobileFramebufferLegacy.h"

# A concrete subclass overriding all 31 of the stand-in's methods, so each emits a
# Display-qualified selector. The bodies are trivial; synthesis overwrites the vtable at
# bind and never calls these.
SOURCE = """#include <graft/IOMobileFramebufferLegacy.h>
class Display : public IOMobileFramebufferLegacy {
    OSDeclareDefaultStructors(Display);

public:
    bool start(IOService *p) override;
    void stop(IOService *p) override;
    void free() override;
    graft_iomfb_word set_notify_tag_mask_gated(IOMFB_NotificationType, IOMFBNotifyRequest *,
                                               uint32_t) override;
    graft_iomfb_word supported_frame_info(uint32_t, uint32_t *, char (*)[64], uint32_t *) override;
    graft_iomfb_word factory_portal(void *, task *) override;
    graft_iomfb_word executeFactoryConfigurations_gated() override;
    graft_iomfb_word set_frame_info_notification_state_gated(IOMFBNotifyRequest *, bool) override;
    graft_iomfb_word do_swap_info_notify_gated(IOMFBNotifyRequest *, const SwapInfoBlob *) override;
    graft_iomfb_word notify_swap_done_gated(uint32_t, ::SwapInfoBlob *, uint32_t *,
                                            uint32_t) override;
    graft_iomfb_word notify_crc_ready_gated(uint32_t) override;
    bool are_timings_enabled() const override;
    graft_iomfb_word start_hardware() override;
    graft_iomfb_word stop_hardware_gated() override;
    graft_iomfb_word swap_begin_gated() override;
    graft_iomfb_word verify_swap(IOMFBSwapIORequest *) override;
    graft_iomfb_word swap_layer_map(IOMFBSwapIORequest *, uint32_t) override;
    graft_iomfb_word surface_layer_map(IOMFB::AP::DisplayedData *, uint32_t, IOSurface *) override;
    graft_iomfb_word surface_layer_unmap(IOMFB::AP::DisplayedData *, uint32_t) override;
    graft_iomfb_word get_dot_pitch(uint32_t *) override;
    graft_iomfb_word set_display_device(uint32_t) override;
    graft_iomfb_word set_digital_out_mode(uint32_t, uint32_t) override;
    graft_iomfb_word get_digital_out_state(uint32_t *) override;
    graft_iomfb_word get_protection_options(uint64_t *) override;
    graft_iomfb_word get_gamma_table(IOMFBGammaTable *) override;
    graft_iomfb_word set_gamma_table(IOMFBGammaTable *) override;
    graft_iomfb_word set_color_remap_mode(DisplayColorRemapMode) override;
    graft_iomfb_word get_color_remap_mode(DisplayColorRemapMode *) const override;
    graft_iomfb_word swap_set_color_matrix(IOMFBColorFixedMatrix *, IOMFBColorMatrixFunction,
                                           uint32_t) override;
    graft_iomfb_word set_parameter(IOMFBParameterName, const uint64_t *, uint32_t) override;
    graft_iomfb_word do_power_state_change_gated() override;
};
OSDefineMetaClassAndStructors(Display, IOMobileFramebufferLegacy);
bool Display::start(IOService *) { return true; }
void Display::stop(IOService *) {}
void Display::free() {}
graft_iomfb_word Display::set_notify_tag_mask_gated(IOMFB_NotificationType, IOMFBNotifyRequest *,
                                                    uint32_t) { return 0; }
graft_iomfb_word Display::supported_frame_info(uint32_t, uint32_t *, char (*)[64], uint32_t *)
                                                    { return 0; }
graft_iomfb_word Display::factory_portal(void *, task *) { return 0; }
graft_iomfb_word Display::executeFactoryConfigurations_gated() { return 0; }
graft_iomfb_word Display::set_frame_info_notification_state_gated(IOMFBNotifyRequest *, bool)
                                                    { return 0; }
graft_iomfb_word Display::do_swap_info_notify_gated(IOMFBNotifyRequest *, const SwapInfoBlob *)
                                                    { return 0; }
graft_iomfb_word Display::notify_swap_done_gated(uint32_t, ::SwapInfoBlob *, uint32_t *, uint32_t)
                                                    { return 0; }
graft_iomfb_word Display::notify_crc_ready_gated(uint32_t) { return 0; }
bool Display::are_timings_enabled() const { return false; }
graft_iomfb_word Display::start_hardware() { return 0; }
graft_iomfb_word Display::stop_hardware_gated() { return 0; }
graft_iomfb_word Display::swap_begin_gated() { return 0; }
graft_iomfb_word Display::verify_swap(IOMFBSwapIORequest *) { return 0; }
graft_iomfb_word Display::swap_layer_map(IOMFBSwapIORequest *, uint32_t) { return 0; }
graft_iomfb_word Display::surface_layer_map(IOMFB::AP::DisplayedData *, uint32_t, IOSurface *)
                                                    { return 0; }
graft_iomfb_word Display::surface_layer_unmap(IOMFB::AP::DisplayedData *, uint32_t) { return 0; }
graft_iomfb_word Display::get_dot_pitch(uint32_t *) { return 0; }
graft_iomfb_word Display::set_display_device(uint32_t) { return 0; }
graft_iomfb_word Display::set_digital_out_mode(uint32_t, uint32_t) { return 0; }
graft_iomfb_word Display::get_digital_out_state(uint32_t *) { return 0; }
graft_iomfb_word Display::get_protection_options(uint64_t *) { return 0; }
graft_iomfb_word Display::get_gamma_table(IOMFBGammaTable *) { return 0; }
graft_iomfb_word Display::set_gamma_table(IOMFBGammaTable *) { return 0; }
graft_iomfb_word Display::set_color_remap_mode(DisplayColorRemapMode) { return 0; }
graft_iomfb_word Display::get_color_remap_mode(DisplayColorRemapMode *) const { return 0; }
graft_iomfb_word Display::swap_set_color_matrix(IOMFBColorFixedMatrix *, IOMFBColorMatrixFunction,
                                                uint32_t) { return 0; }
graft_iomfb_word Display::set_parameter(IOMFBParameterName, const uint64_t *, uint32_t)
                                                { return 0; }
graft_iomfb_word Display::do_power_state_change_gated() { return 0; }
"""

# The 31 selectors the stand-in's declarations mangle to, qualified onto the test subclass
# "Display". Regenerated by compiling SOURCE; frozen so a signature edit that shifts a
# mangling fails the test. are_timings_enabled and get_color_remap_mode are const, so they
# carry the __ZNK prefix.
EXPECTED_OVERRIDES = {
    "__ZN7Display5startEP9IOService",
    "__ZN7Display4stopEP9IOService",
    "__ZN7Display4freeEv",
    "__ZN7Display25set_notify_tag_mask_gatedE22IOMFB_NotificationTypeP18IOMFBNotifyRequestj",
    "__ZN7Display20supported_frame_infoEjPjPA64_cS0_",
    "__ZN7Display14factory_portalEPvP4task",
    "__ZN7Display34executeFactoryConfigurations_gatedEv",
    "__ZN7Display39set_frame_info_notification_state_gatedEP18IOMFBNotifyRequestb",
    "__ZN7Display25do_swap_info_notify_gatedEP18IOMFBNotifyRequest"
    "PKN25IOMobileFramebufferLegacy12SwapInfoBlobE",
    "__ZN7Display22notify_swap_done_gatedEjP12SwapInfoBlobPjj",
    "__ZN7Display22notify_crc_ready_gatedEj",
    "__ZNK7Display19are_timings_enabledEv",
    "__ZN7Display14start_hardwareEv",
    "__ZN7Display19stop_hardware_gatedEv",
    "__ZN7Display16swap_begin_gatedEv",
    "__ZN7Display11verify_swapEP18IOMFBSwapIORequest",
    "__ZN7Display14swap_layer_mapEP18IOMFBSwapIORequestj",
    "__ZN7Display17surface_layer_mapEPN5IOMFB2AP13DisplayedDataEjP9IOSurface",
    "__ZN7Display19surface_layer_unmapEPN5IOMFB2AP13DisplayedDataEj",
    "__ZN7Display13get_dot_pitchEPj",
    "__ZN7Display18set_display_deviceEj",
    "__ZN7Display20set_digital_out_modeEjj",
    "__ZN7Display21get_digital_out_stateEPj",
    "__ZN7Display22get_protection_optionsEPy",
    "__ZN7Display15get_gamma_tableEP15IOMFBGammaTable",
    "__ZN7Display15set_gamma_tableEP15IOMFBGammaTable",
    "__ZN7Display20set_color_remap_modeE21DisplayColorRemapMode",
    "__ZNK7Display20get_color_remap_modeEP21DisplayColorRemapMode",
    "__ZN7Display21swap_set_color_matrixEP21IOMFBColorFixedMatrix24IOMFBColorMatrixFunctionj",
    "__ZN7Display13set_parameterE18IOMFBParameterNamePKyj",
    "__ZN7Display27do_power_state_change_gatedEv",
}

ARCHES = ("arm64e", "x86_64")


def sdk_path() -> str:
    return subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()


def compile_consumer(root: Path, arch: str, sdk: str) -> Path:
    """Compile SOURCE against the stand-in as a macOS 11 kext object for arch."""
    source = root / "display.cpp"
    source.write_text(SOURCE)
    obj = root / f"display-{arch}.o"
    command = [
        *("xcrun", "clang++", "-arch", arch, "-isysroot", sdk),
        *("-mmacosx-version-min=11.0", "-mkernel", "-nostdinc"),
        *("-isystem", f"{sdk}/System/Library/Frameworks/Kernel.framework/Headers"),
        *("-I", str(ROOT / "sdk/include")),
        *("-DKERNEL", "-std=c++17", "-fapple-kext", "-fno-rtti", "-fno-exceptions", "-O2"),
        *("-c", str(source), "-o", str(obj)),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return obj


def defined_symbols(obj: Path) -> list[str]:
    """The object's defined symbol names (nm -U omits undefined symbols)."""
    out = subprocess.check_output(["xcrun", "nm", "-U", str(obj)], text=True)
    return [line.split()[-1] for line in out.splitlines() if line.strip()]


def reserve_count() -> int:
    match = re.search(r"#define GRAFT_IOMFB_VTABLE_RESERVE (\d+)", HEADER.read_text())
    assert match, "GRAFT_IOMFB_VTABLE_RESERVE not found in the stand-in header"
    return int(match.group(1))


class SdkStandinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory()
        root = Path(cls._temp.name)
        sdk = sdk_path()
        # Compiling here is itself the check that the header builds for each architecture.
        cls.symbols = {arch: defined_symbols(compile_consumer(root, arch, sdk)) for arch in ARCHES}

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def test_reserves_the_padding_slots(self):
        reserve = reserve_count()
        for arch in ARCHES:
            stubs = [s for s in self.symbols[arch] if "graft_vtr_" in s]
            self.assertEqual(
                len(stubs),
                reserve,
                f"{arch}: expected {reserve} reserved vtable stubs, found {len(stubs)}",
            )

    def test_emits_the_override_selectors(self):
        for arch in ARCHES:
            defined = set(self.symbols[arch])
            missing = sorted(EXPECTED_OVERRIDES - defined)
            self.assertEqual(missing, [], f"{arch}: stand-in missing selectors: {missing}")


if __name__ == "__main__":
    unittest.main()
