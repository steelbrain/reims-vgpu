// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// A stand-in declaration of IOMobileGraphicsFamily's IOMobileFramebufferLegacy,
// the class an arm64 WindowServer drives and which no Apple SDK declares. A
// Graft consumer subclasses it to be that display. It holds only what such a
// consumer overrides or calls: the class's own pure methods, the concrete
// methods a consumer overrides, start/stop/free, and note_vbl_ISR.
// Each carries the signature its symbol in the arm64 kernel collections of
// macOS 13.6 to 27.0.1 names; the parameter types are named only to produce
// those manglings and declared no further.
//
// No symbol encodes a return type, so where nothing says what a method returns
// it returns graft_iomfb_word, a 64-bit word that holds whatever the class's
// callers read back: an IOReturn, a bool or a pointer.
//
// A consumer calls none of the virtual methods declared here except start, stop
// and free. Not virtually, since Graft fills the consumer's vtable in the live
// order; and not as a qualified call such as
// IOMobileFramebufferLegacy::start_hardware(), which under -fapple-kext loads the
// method from the class's vtable at this stand-in's compiled slot, a slot the
// live class gives to another method. start, stop and free lie in IOService's
// part of the vtable, which every release keeps in the SDK's order, so a
// qualified call to those three reaches the right method.
//
// At bind Graft fills the consumer's OWN vtable in place, in the live
// IOMobileFramebufferLegacy's layout: it measures the live class, writes each
// inherited slot from the live parent and each override on the live slot of the
// same method identity (selector), so a release that renumbers slots still lands
// each override on the right method (notes/arm64-display.md, src/arm64e/
// Synthesize.h). For that in-place fill the compiled vtable must already be at
// least the live parent's word count, so the class reserves a run of virtual
// slots past its real methods (GRAFT_IOMFB_VTABLE_RESERVE below). The one field
// is field_reserve, inside which the live class's own fields lie at their real
// offsets, so the consumer's own fields follow rather than overlapping them;
// Graft refuses the bind where a release's live class is larger than IOService
// plus this reserve, or its vtable longer than the reserved slots. The live chain
// is IOMobileFramebufferLegacy : IOMobileFramebufferService : IOMobileFramebuffer
// : IOService; the intermediate IOMobileFramebufferService changes neither the
// manglings nor, since Graft sizes to the live Legacy, the layout, so the
// stand-in leaves it out.
#pragma once
#include <IOKit/IOService.h>

class IOSurface;
struct task;
struct IOMFBNotifyRequest;
struct IOMFBSwapIORequest;
struct SwapInfoBlob;
namespace IOMFB {
namespace AP {
struct DisplayedData;
}
} // namespace IOMFB
enum IOMFB_NotificationType : uint32_t;

// A 64-bit method word: see the file comment.
typedef uint64_t graft_iomfb_word;

// The reserve covers the live class's own fields. The live class is at most
// 0x1dd8 bytes across macOS 13.6 to 27.0.1 (Ambrosia's per-release measurement)
// and this stand-in's IOService base is 0x88, so 0x2000 clears the largest with
// room to spare; Graft refuses the bind where a future release's live class is
// larger still, rather than risk a field overlay.
#define GRAFT_IOMFB_LEGACY_RESERVE 0x2000

// GRAFT_IOMFB_VTABLE_RESERVE reserved virtual slots lengthen a consumer's vtable
// so its compiled __ZTV is at least the live IOMobileFramebufferLegacy's word
// count (at most 485 across macOS 13.6 to 27.0.1, with room for later releases).
// Graft fills the consumer's own vtable in place at bind (file comment), so the
// storage must already be long enough; it records the compiled vtable's length in
// the manifest and refuses the bind by name if a live release ever exceeds it. The
// reserved slots are never called: those within the live class's length are
// overwritten at bind, the rest stay as compiled, and the consumer never invokes
// any of them. So each is a trivial inline stub that exists only to lengthen the
// table (pure slots would make the class abstract; undefined ones would not
// link). GRAFT_IOMFB_VTABLE_PAD below realizes exactly this many, 8 * 64.
#define GRAFT_IOMFB_VTABLE_RESERVE 512
// A tabular macro expansion (8 * 64 = GRAFT_IOMFB_VTABLE_RESERVE slots); laid out
// by hand so clang-format leaves the table alone.
// clang-format off
#define GRAFT_IOMFB_VTR_1(p) virtual graft_iomfb_word p() { return 0; }
#define GRAFT_IOMFB_VTR_8(p) \
    GRAFT_IOMFB_VTR_1(p##0) GRAFT_IOMFB_VTR_1(p##1) GRAFT_IOMFB_VTR_1(p##2) GRAFT_IOMFB_VTR_1(p##3) \
    GRAFT_IOMFB_VTR_1(p##4) GRAFT_IOMFB_VTR_1(p##5) GRAFT_IOMFB_VTR_1(p##6) GRAFT_IOMFB_VTR_1(p##7)
#define GRAFT_IOMFB_VTR_64(p) \
    GRAFT_IOMFB_VTR_8(p##0) GRAFT_IOMFB_VTR_8(p##1) GRAFT_IOMFB_VTR_8(p##2) GRAFT_IOMFB_VTR_8(p##3) \
    GRAFT_IOMFB_VTR_8(p##4) GRAFT_IOMFB_VTR_8(p##5) GRAFT_IOMFB_VTR_8(p##6) GRAFT_IOMFB_VTR_8(p##7)
#define GRAFT_IOMFB_VTABLE_PAD \
    GRAFT_IOMFB_VTR_64(graft_vtr_0_) GRAFT_IOMFB_VTR_64(graft_vtr_1_) \
    GRAFT_IOMFB_VTR_64(graft_vtr_2_) GRAFT_IOMFB_VTR_64(graft_vtr_3_) \
    GRAFT_IOMFB_VTR_64(graft_vtr_4_) GRAFT_IOMFB_VTR_64(graft_vtr_5_) \
    GRAFT_IOMFB_VTR_64(graft_vtr_6_) GRAFT_IOMFB_VTR_64(graft_vtr_7_)
// clang-format on

class IOMobileFramebuffer : public IOService {
    OSDeclareAbstractStructors(IOMobileFramebuffer);
};

class IOMobileFramebufferLegacy : public IOMobileFramebuffer {
    OSDeclareAbstractStructors(IOMobileFramebufferLegacy);

public:
    struct SwapInfoBlob;

    bool start(IOService *provider) override;
    void stop(IOService *provider) override;
    void free() override;

    // The class's own pure methods: a subclass implements each.
    virtual graft_iomfb_word set_notify_tag_mask_gated(IOMFB_NotificationType type,
                                                       IOMFBNotifyRequest *request,
                                                       uint32_t mask) = 0;
    // macOS 14 and later; macOS 13's takes no fourth argument (Graft matches either).
    virtual graft_iomfb_word supported_frame_info(uint32_t index, uint32_t *count,
                                                  char (*names)[64], uint32_t *field_3) = 0;
    virtual graft_iomfb_word factory_portal(void *field_0, task *owner) = 0;
    virtual graft_iomfb_word executeFactoryConfigurations_gated() = 0;
    virtual graft_iomfb_word set_frame_info_notification_state_gated(IOMFBNotifyRequest *request,
                                                                     bool on) = 0;
    virtual graft_iomfb_word do_swap_info_notify_gated(IOMFBNotifyRequest *request,
                                                       const SwapInfoBlob *blob) = 0;
    virtual graft_iomfb_word notify_swap_done_gated(uint32_t field_0, ::SwapInfoBlob *blob,
                                                    uint32_t *field_2, uint32_t field_3) = 0;
    virtual graft_iomfb_word notify_crc_ready_gated(uint32_t field_0) = 0;
    virtual bool are_timings_enabled() const = 0;

    // What the class implements. Its own implementations do no display work, as
    // its hardware subclasses override them (start_hardware and verify_swap
    // return 0), so a display overrides each and never calls the class's.
    virtual graft_iomfb_word start_hardware();
    virtual graft_iomfb_word stop_hardware_gated();
    virtual graft_iomfb_word swap_begin_gated();
    virtual graft_iomfb_word verify_swap(IOMFBSwapIORequest *request);
    virtual graft_iomfb_word swap_layer_map(IOMFBSwapIORequest *request, uint32_t layer);
    virtual graft_iomfb_word surface_layer_map(IOMFB::AP::DisplayedData *data, uint32_t layer,
                                               IOSurface *surface);
    virtual graft_iomfb_word surface_layer_unmap(IOMFB::AP::DisplayedData *data, uint32_t layer);

    // What a consumer calls: a refresh of the display. Graft resolves its symbol
    // in the live collection, as it resolves the consumer's other kernel imports.
    void note_vbl_ISR(uint32_t field_0);

protected:
    // GRAFT_IOMFB_VTABLE_RESERVE reserved slots so the compiled vtable is at least
    // the live parent's word count for the in-place fill (file comment, and the
    // macro's own comment). Never called.
    GRAFT_IOMFB_VTABLE_PAD

private:
    UInt8 field_reserve[GRAFT_IOMFB_LEGACY_RESERVE];
};

#undef GRAFT_IOMFB_VTABLE_PAD
#undef GRAFT_IOMFB_VTR_64
#undef GRAFT_IOMFB_VTR_8
#undef GRAFT_IOMFB_VTR_1
