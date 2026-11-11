// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A synthesized display consumer: a subclass of the SDK's
// IOMobileFramebufferLegacy stand-in that overrides its nine pure methods and
// ten of the others.
#pragma once
#include <graft/IOMobileFramebufferLegacy.h>
class SynthDisplay : public IOMobileFramebufferLegacy {
    OSDeclareDefaultStructors(SynthDisplay);

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
};
