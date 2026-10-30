/*
 * The reims-graphics device's stage 2 wire protocol, as the guest's display
 * kext and option ROM use it: docs/graphics-kext/design.md, "Stage 2 wire
 * protocol". All values are little-endian.
 *
 * This header is written from that section's tables and stands in for the
 * host's generated one until it lands; then the guest code includes that one
 * and this file goes.
 *
 * Where the tables leave a layout open, this header takes the C layout: each
 * field at its natural alignment, so SET_CURSOR's offset sits at payload byte
 * 24 and VBLANK's count at payload byte 8.
 *
 * The defines before the structures are plain numbers, so an assembler
 * source can include this header for the PCI identity.
 */

#ifndef REIMS_GRAPHICS_WIRE_H_
#define REIMS_GRAPHICS_WIRE_H_

/* The PCI identity: the development placeholder ID, display controller,
   other. */
#define RG_PCI_VENDOR_ID   0x1234
#define RG_PCI_DEVICE_ID   0x5247
#define RG_PCI_CLASS_CODE  0x038000

/* BAR0: 4 KiB of 32-bit registers; the MSI-X table and PBA from 0x800. */
#define RG_BAR_REGISTERS          0
#define RG_REG_MAGIC              0x000  /* RO: RG_MAGIC */
#define RG_REG_VERSION            0x004  /* RO: RG_VERSION */
#define RG_REG_FEATURES_LO        0x008  /* RO: 0 in stage 2 */
#define RG_REG_FEATURES_HI        0x00c  /* RO */
#define RG_REG_STATUS             0x010  /* RW: 0 resets, 1 is driver ready */
#define RG_REG_REGION_SIZE_LO     0x014  /* RO: BAR2's size */
#define RG_REG_REGION_SIZE_HI     0x018  /* RO */
#define RG_REG_DOORBELL           0x020  /* WO: a ring's id wakes its consumer */
#define RG_REG_MSIX               0x800

#define RG_MAGIC                  0x58464752u  /* "RGFX" */
#define RG_VERSION                1u
#define RG_STATUS_RESET           0u
#define RG_STATUS_DRIVER_READY    1u

/* BAR2: the region, host memory mapped write-back. */
#define RG_BAR_REGION             2
#define RG_CONTROL_SUBMIT_OFFSET  0x00000  /* ring id 0, guest to host */
#define RG_CONTROL_EVENT_OFFSET   0x10000  /* host to guest, MSI-X vector 0 */
#define RG_CONTROL_RING_SLOT      0x10000  /* each control ring's 64 KiB */
#define RG_GUEST_ALLOC_OFFSET     0x20000  /* the guest's allocations, up to the end */
#define RG_CONTROL_RING_ID        0
#define RG_CONTROL_EVENT_VECTOR   0

/* A ring: a 256-byte header, each field in its own 64-byte line, then the
   bytes. The control rings' bytes are their slot's 64 KiB less the header. */
#define RG_RING_HEADER_SIZE       256
#define RG_RING_PRODUCER          0x00   /* u64, free-running, by the producer */
#define RG_RING_CONSUMER          0x40   /* u64, free-running, by the consumer */
#define RG_RING_CONSUMER_IDLE     0x80   /* u32, 1 while the consumer sleeps */
#define RG_RING_ARMED             0xc0   /* u32, 1 when the guest wants an interrupt */
#define RG_CONTROL_RING_BYTES     (RG_CONTROL_RING_SLOT - RG_RING_HEADER_SIZE)

/* Message types. Commands go guest to host on ring 0; events come back on
   the control completion ring. Type 0 pads to the ring's end. */
#define RG_MSG_PAD                0x000
#define RG_CMD_GET_DISPLAYS       0x001
#define RG_CMD_SET_SCANOUT        0x002
#define RG_CMD_SET_CURSOR         0x003
#define RG_CMD_MOVE_CURSOR        0x004
#define RG_CMD_SET_VBLANK         0x005
#define RG_EVT_COMPLETE           0x100
#define RG_EVT_DISPLAYS           0x101
#define RG_EVT_DISPLAY_CHANGED    0x102
#define RG_EVT_VBLANK             0x103

/* A message's size, header included, is a multiple of this. */
#define RG_MSG_ALIGN              8

#ifndef __ASSEMBLER__

#include <stdint.h>

/* Every message starts so. size counts these 16 bytes; the guest numbers its
   commands from 1, per ring. */
struct rg_msg_header {
	uint32_t type;
	uint32_t size;
	uint64_t seqno;
};

/* SET_SCANOUT: width 0 blanks the display. The pixels are 32-bit BGRA8. */
struct rg_set_scanout {
	uint32_t display;
	uint32_t width;
	uint32_t height;
	uint32_t stride;
	uint64_t offset;  /* into BAR2 */
};

/* SET_CURSOR: a BGRA8 premultiplied image of width x height, rows packed;
   width 0 removes the image. */
struct rg_set_cursor {
	uint32_t display;
	uint32_t width;
	uint32_t height;
	uint32_t hot_x;
	uint32_t hot_y;
	uint32_t reserved;
	uint64_t offset;  /* into BAR2 */
};

/* MOVE_CURSOR: x and y are the hotspot's place on the display. No answer. */
struct rg_move_cursor {
	uint32_t display;
	int32_t x;
	int32_t y;
	uint32_t visible;
};

struct rg_set_vblank {
	uint32_t display;
	uint32_t enabled;
};

/* COMPLETE: the seqno is the command's; status 0 is success. */
struct rg_complete {
	uint32_t status;
};

/* DISPLAYS: count, then per display an rg_display followed by its
   mode_count rg_modes. */
struct rg_displays {
	uint32_t count;
};

struct rg_display {
	uint32_t id;
	uint32_t connected;
	uint32_t preferred_mode;  /* an index into the display's modes */
	uint32_t mode_count;
};

struct rg_mode {
	uint32_t width;
	uint32_t height;
	uint32_t refresh_mhz;
};

/* DISPLAY_CHANGED: the guest asks GET_DISPLAYS again. */
struct rg_display_changed {
	uint32_t display;
};

struct rg_vblank {
	uint32_t display;
	uint32_t reserved;
	uint64_t count;
	uint64_t host_time_ns;
};

/* A message's size: header and payload, rounded up to RG_MSG_ALIGN. */
#define RG_MSG_SIZE(payload) \
	((uint32_t)((sizeof(struct rg_msg_header) + (payload) + RG_MSG_ALIGN - 1) & \
		~(uint32_t)(RG_MSG_ALIGN - 1)))

#endif /* __ASSEMBLER__ */

#endif /* REIMS_GRAPHICS_WIRE_H_ */
