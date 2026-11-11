// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The synthesized display's methods. Graft builds its vtable when it binds it
// and never calls these bodies in the tests.
#include "SynthDisplay.h"
#include <mach/mach_types.h>
OSDefineMetaClassAndStructors(SynthDisplay, IOMobileFramebufferLegacy);
bool SynthDisplay::start(IOService *) {
    return true;
}
void SynthDisplay::stop(IOService *) {}
void SynthDisplay::free() {}
graft_iomfb_word SynthDisplay::set_notify_tag_mask_gated(IOMFB_NotificationType,
                                                         IOMFBNotifyRequest *, uint32_t) {
    return 0;
}
graft_iomfb_word SynthDisplay::supported_frame_info(uint32_t, uint32_t *, char (*)[64],
                                                    uint32_t *) {
    return 0;
}
graft_iomfb_word SynthDisplay::factory_portal(void *, task *) {
    return 0;
}
graft_iomfb_word SynthDisplay::executeFactoryConfigurations_gated() {
    return 0;
}
graft_iomfb_word SynthDisplay::set_frame_info_notification_state_gated(IOMFBNotifyRequest *, bool) {
    return 0;
}
graft_iomfb_word SynthDisplay::do_swap_info_notify_gated(IOMFBNotifyRequest *,
                                                         const SwapInfoBlob *) {
    return 0;
}
graft_iomfb_word SynthDisplay::notify_swap_done_gated(uint32_t, ::SwapInfoBlob *, uint32_t *,
                                                      uint32_t) {
    return 0;
}
graft_iomfb_word SynthDisplay::notify_crc_ready_gated(uint32_t) {
    return 0;
}
bool SynthDisplay::are_timings_enabled() const {
    return false;
}
graft_iomfb_word SynthDisplay::start_hardware() {
    return 0;
}
graft_iomfb_word SynthDisplay::stop_hardware_gated() {
    return 0;
}
graft_iomfb_word SynthDisplay::swap_begin_gated() {
    return 0;
}
graft_iomfb_word SynthDisplay::verify_swap(IOMFBSwapIORequest *) {
    return 0;
}
graft_iomfb_word SynthDisplay::swap_layer_map(IOMFBSwapIORequest *, uint32_t) {
    return 0;
}
graft_iomfb_word SynthDisplay::surface_layer_map(IOMFB::AP::DisplayedData *, uint32_t,
                                                 IOSurface *) {
    return 0;
}
graft_iomfb_word SynthDisplay::surface_layer_unmap(IOMFB::AP::DisplayedData *, uint32_t) {
    return 0;
}
extern "C" kern_return_t display_start(kmod_info_t *, void *) {
    return KERN_SUCCESS;
}
extern "C" kern_return_t display_stop(kmod_info_t *, void *) {
    return KERN_SUCCESS;
}
extern "C" {
KMOD_EXPLICIT_DECL(ai.aneesiqbal.graft.test.Display, "1.0.0", display_start, display_stop)
}
