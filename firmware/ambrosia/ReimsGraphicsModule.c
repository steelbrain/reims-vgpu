/** @file
  The reims-graphics device's driver module for Ambrosia, the boot firmware
  of arm64 macOS guests on QEMU's vmapple machine (AmbrosiaGraphics.h): an
  early console on the device, as the option ROM (../src) gives an x86-64
  guest's firmware one through the Graphics Output Protocol.

  The entry point finds the device on bus 0, gives BAR 0 an address at the
  PCIe window's start and BAR 2, the region, the first address after it
  aligned to its size, and turns memory decoding on. Then it does what the
  option ROM's Start and SetMode do: it resets the device, adopts the control
  rings' counters, asks the device for its displays, and points the preferred
  mode of the first connected display, or 1920x1080 when there is none, at the
  region's memory from RGFX_ALLOCATIONS_OFFSET with SET_SCANOUT. That memory
  is the framebuffer the module returns, which Ambrosia draws its log on and
  hands to the kernel. A command polls the completion ring for its answer.

  The PCIe window is device memory to the module, so it reads and writes the
  rings 8 bytes at a time with plain loads and stores, ordered by barriers,
  where the option ROM uses acquire and release atomics.
**/

#include "AmbrosiaGraphics.h"
#include "reims_graphics_wire.h"

//
// The mode used when the device names no connected display with a mode, and
// the alignment of a scanout row's bytes, as the option ROM has them
// (../src/ReimsGraphics.h), so the boot frame keeps its layout when the
// guest's display kext takes the display over.
//
#define REIMS_GRAPHICS_FALLBACK_WIDTH   1920
#define REIMS_GRAPHICS_FALLBACK_HEIGHT  1080
#define REIMS_GRAPHICS_STRIDE_ALIGN     256

//
// How long a command waits for room on the ring and for its answer: 2 s, as
// the option ROM's do.
//
#define REIMS_GRAPHICS_TIMEOUT_S  2

//
// The configuration space's command register, its memory decoding bit, and
// the first BAR.
//
#define PCI_COMMAND         0x04
#define PCI_COMMAND_MEMORY  0x0002
#define PCI_BARS            0x10

//
// The ring header's fields, by their offset in it, and the control ring's
// byte-ring size (a power of two, so an index wraps with a mask).
//
#define RING_PRODUCER_OFFSET       __builtin_offsetof (RgfxRingHeader, producer)
#define RING_CONSUMER_OFFSET       __builtin_offsetof (RgfxRingHeader, consumer)
#define RING_CONSUMER_IDLE_OFFSET  __builtin_offsetof (RgfxRingHeader, consumer_idle)
#define RING_MASK                  (RGFX_CONTROL_RING_DATA_SIZE - 1)
#define RING_MSG_ALIGN             8

typedef enum {
  ReimsGraphicsOk,
  ReimsGraphicsTimeout,       // no room on the ring, or no answer, in time
  ReimsGraphicsDeviceError,   // a malformed message on the completion ring
  ReimsGraphicsRefused        // a COMPLETE whose status is not RGFX_STATUS_OK
} REIMS_GRAPHICS_STATUS;

typedef struct {
  uint64_t                   Registers;   // BAR 0's address
  uint64_t                   Region;      // BAR 2's address
  uint64_t                   RegionSize;  // the size the device gives
  //
  // The control rings: the guest's producer count of the submission ring, its
  // consumer count of the completion ring, and the last command's number.
  //
  uint64_t                   SubmitProducer;
  uint64_t                   EventConsumer;
  uint64_t                   Seqno;
  //
  // The one mode set, on the display Display names.
  //
  uint32_t                   Display;
  uint32_t                   Width;
  uint32_t                   Height;
  uint32_t                   Stride;      // bytes
} REIMS_GRAPHICS;

//
// A command, and the 8-byte words it is copied to the ring in.
//
typedef union {
  RgfxMsgHeader      Header;
  RgfxGetDisplays    GetDisplays;
  RgfxSetScanout     SetScanout;
  uint64_t           Words[sizeof (RgfxSetScanout) / sizeof (uint64_t)];
} REIMS_GRAPHICS_COMMAND;

//
// A message's header, and the words it is read from and written to the ring
// in.
//
typedef union {
  RgfxMsgHeader    Header;
  uint64_t         Words[sizeof (RgfxMsgHeader) / sizeof (uint64_t)];
} REIMS_GRAPHICS_HEADER;

//
// An answer's payload, the message after its header, as 8-byte words.
//
typedef struct {
  uint64_t    Words[512];
} REIMS_GRAPHICS_ANSWER;

//
// One line of the log.
//
typedef struct {
  uint8_t    Text[128];
  size_t     Length;
} LOG_LINE;

/* ------ Device memory, barriers and the counter ------ */

//
// Each access to the device's memory is one instruction: a plain LDRH, STRH,
// LDR or STR at a base register's address, written out. A hypervisor that
// emulates the device's registers or its configuration space decodes a
// guest's access from its abort's syndrome, which the CPU gives only for a
// single load or store without writeback. A pre- or post-indexed, pair or
// SIMD&FP access aborts with no syndrome (ISV 0), and KVM hands that to QEMU,
// which decodes it only with its own instruction emulator. A volatile access
// fixes the access's size and count, not the instruction the compiler picks
// (BarSize's read of ECAM compiled to a pre-indexed LDR).
//

static inline uint16_t
Read16 (
  uint64_t  Address
  )
{
  uint16_t  Value;

  __asm__ volatile ("ldrh %w0, [%1]" : "=r" (Value) : "r" (Address) : "memory");
  return Value;
}

static inline void
Write16 (
  uint64_t  Address,
  uint16_t  Value
  )
{
  __asm__ volatile ("strh %w0, [%1]" : : "rZ" (Value), "r" (Address) : "memory");
}

static inline uint32_t
Read32 (
  uint64_t  Address
  )
{
  uint32_t  Value;

  __asm__ volatile ("ldr %w0, [%1]" : "=r" (Value) : "r" (Address) : "memory");
  return Value;
}

static inline void
Write32 (
  uint64_t  Address,
  uint32_t  Value
  )
{
  __asm__ volatile ("str %w0, [%1]" : : "rZ" (Value), "r" (Address) : "memory");
}

static inline uint64_t
Read64 (
  uint64_t  Address
  )
{
  uint64_t  Value;

  __asm__ volatile ("ldr %x0, [%1]" : "=r" (Value) : "r" (Address) : "memory");
  return Value;
}

static inline void
Write64 (
  uint64_t  Address,
  uint64_t  Value
  )
{
  __asm__ volatile ("str %x0, [%1]" : : "rZ" (Value), "r" (Address) : "memory");
}

/**
  Order every memory access before it against every one after it, for the
  device as for the CPU.
**/
static inline void
Barrier (
  void
  )
{
  __asm__ volatile ("dmb sy" ::: "memory");
}

/**
  The virtual counter, CNTVCT_EL0, which Ambrosia leaves readable.
**/
static inline uint64_t
Counter (
  void
  )
{
  uint64_t  Value;

  __asm__ volatile ("isb\n\tmrs %0, cntvct_el0" : "=r" (Value) :: "memory");
  return Value;
}

/**
  The counter's value REIMS_GRAPHICS_TIMEOUT_S from now.
**/
static uint64_t
Deadline (
  void
  )
{
  uint64_t  Frequency;

  __asm__ volatile ("mrs %0, cntfrq_el0" : "=r" (Frequency));
  return Counter () + Frequency * REIMS_GRAPHICS_TIMEOUT_S;
}

/* ------ The log ------ */

static void
Append (
  LOG_LINE    *Line,
  const char  *Text
  )
{
  while ((*Text != 0) && (Line->Length < sizeof (Line->Text))) {
    Line->Text[Line->Length++] = (uint8_t)*Text++;
  }
}

static void
AppendDecimal (
  LOG_LINE  *Line,
  uint64_t  Value
  )
{
  char    Digits[21];
  size_t  Index;

  Index         = sizeof (Digits) - 1;
  Digits[Index] = 0;
  do {
    Digits[--Index] = (char)('0' + Value % 10);
    Value          /= 10;
  } while (Value != 0);

  Append (Line, &Digits[Index]);
}

/**
  Append Value in lowercase hex, with at least MinDigits digits and no prefix.
**/
static void
AppendHex (
  LOG_LINE  *Line,
  uint64_t  Value,
  size_t    MinDigits
  )
{
  char    Digits[17];
  size_t  Index;

  Index         = sizeof (Digits) - 1;
  Digits[Index] = 0;
  do {
    Digits[--Index] = "0123456789abcdef"[Value % 16];
    Value          /= 16;
  } while ((Value != 0) || (sizeof (Digits) - 1 - Index < MinDigits));

  Append (Line, &Digits[Index]);
}

static void
Say (
  const AMBROSIA_SERVICES  *Services,
  const LOG_LINE           *Line
  )
{
  Services->Log (Services, Line->Text, Line->Length);
}

/**
  Log "reims-graphics: " and Text.
**/
static void
SayText (
  const AMBROSIA_SERVICES  *Services,
  const char               *Text
  )
{
  LOG_LINE  Line;

  Line.Length = 0;
  Append (&Line, "reims-graphics: ");
  Append (&Line, Text);
  Say (Services, &Line);
}

static const char *
StatusName (
  REIMS_GRAPHICS_STATUS  Status
  )
{
  switch (Status) {
    case ReimsGraphicsOk:
      return "done";
    case ReimsGraphicsTimeout:
      return "no answer in time";
    case ReimsGraphicsDeviceError:
      return "a malformed message on the completion ring";
    default:
      return "refused";
  }
}

/* ------ PCI ------ */

/**
  The configuration space of the device on bus 0, and its device and function
  numbers; 0 where there is none.
**/
static uint64_t
FindDevice (
  const AMBROSIA_SERVICES  *Services,
  uint32_t                 *Device,
  uint32_t                 *Function
  )
{
  uint64_t  Offset;
  uint64_t  Config;

  for (*Device = 0; *Device < 32; (*Device)++) {
    for (*Function = 0; *Function < 8; (*Function)++) {
      Offset = ((uint64_t)*Device << 15) | ((uint64_t)*Function << 12);
      if (Offset + 0x1000 > Services->EcamSize) {
        return 0;
      }

      Config = Services->EcamBase + Offset;
      if (Read32 (Config) == (((uint32_t)RGFX_PCI_DEVICE_ID << 16) | RGFX_PCI_VENDOR_ID)) {
        return Config;
      }
    }
  }

  return 0;
}

/**
  The size of memory BAR Index, and in Wide whether it is 64-bit; 0 where it is
  an I/O BAR or not there. Memory decoding must be off.
**/
static uint64_t
BarSize (
  uint64_t  Config,
  uint32_t  Index,
  int       *Wide
  )
{
  uint64_t  At;
  uint32_t  Low;
  uint32_t  High;
  uint64_t  Mask;

  At  = Config + PCI_BARS + 4 * Index;
  Low = Read32 (At);
  if ((Low & 1) != 0) {
    return 0;
  }

  *Wide = ((Low >> 1) & 3) == 2;
  Write32 (At, 0xffffffff);
  Mask = (Read32 (At) & ~(uint64_t)0xf) | 0xffffffff00000000;
  Write32 (At, Low);
  if (*Wide) {
    High = Read32 (At + 4);
    Write32 (At + 4, 0xffffffff);
    Mask = (Mask & 0xffffffff) | ((uint64_t)Read32 (At + 4) << 32);
    Write32 (At + 4, High);
  }

  if (*Wide ? (Mask == 0) : ((uint32_t)Mask == 0)) {
    return 0;
  }

  return ~Mask + 1;
}

static void
SetBar (
  uint64_t  Config,
  uint32_t  Index,
  int       Wide,
  uint64_t  Address
  )
{
  Write32 (Config + PCI_BARS + 4 * Index, (uint32_t)Address);
  if (Wide) {
    Write32 (Config + PCI_BARS + 4 * Index + 4, (uint32_t)(Address >> 32));
  }
}

/* ------ The control rings ------ */

/**
  Copy Size bytes, a multiple of 8, from the ring at Source, 8 bytes at a time.
**/
static void
CopyFromRing (
  uint64_t  *Destination,
  uint64_t  Source,
  uint32_t  Size
  )
{
  uint32_t  Index;

  for (Index = 0; Index < Size / sizeof (uint64_t); Index++) {
    Destination[Index] = Read64 (Source + Index * sizeof (uint64_t));
  }
}

/**
  Copy Size bytes, a multiple of 8, to the ring at Destination, 8 bytes at a
  time.
**/
static void
CopyToRing (
  uint64_t        Destination,
  const uint64_t  *Source,
  uint32_t        Size
  )
{
  uint32_t  Index;

  for (Index = 0; Index < Size / sizeof (uint64_t); Index++) {
    Write64 (Destination + Index * sizeof (uint64_t), Source[Index]);
  }
}

/**
  Put a command on the submission ring, padding to the ring's end first when
  it does not fit before it, and ring the doorbell if the device's consumer
  sleeps. Command's payload fields are filled; its header is stamped here.
  Size is a multiple of 8.
**/
static REIMS_GRAPHICS_STATUS
Submit (
  REIMS_GRAPHICS          *Gfx,
  uint32_t                Type,
  REIMS_GRAPHICS_COMMAND  *Command,
  uint32_t                Size,
  uint64_t                *Seqno
  )
{
  uint64_t               Ring;
  uint64_t               Bytes;
  uint64_t               Producer;
  uint64_t               Consumer;
  uint64_t               Until;
  uint32_t               Offset;
  uint32_t               Left;
  uint32_t               Need;
  REIMS_GRAPHICS_HEADER  Pad;

  Ring     = Gfx->Region + RGFX_CONTROL_SUBMIT_OFFSET;
  Bytes    = Ring + RGFX_RING_HEADER_SIZE;
  Producer = Gfx->SubmitProducer;
  Until    = Deadline ();
  for ( ; ; ) {
    Consumer = Read64 (Ring + RING_CONSUMER_OFFSET);
    Offset   = (uint32_t)Producer & RING_MASK;
    Left     = RGFX_CONTROL_RING_DATA_SIZE - Offset;
    Need     = Size + ((Left < Size) ? Left : 0);
    if (Producer - Consumer + Need <= RGFX_CONTROL_RING_DATA_SIZE) {
      break;
    }

    if (Counter () > Until) {
      return ReimsGraphicsTimeout;
    }
  }

  //
  // The consumer's count is read before the room it frees is written.
  //
  Barrier ();

  //
  // A message never wraps. The pad is a type 0 message of the bytes left;
  // with only 8 left, its type and size alone.
  //
  if (Left < Size) {
    Pad.Header.type  = RGFX_MSG_PAD;
    Pad.Header.size  = Left;
    Pad.Header.seqno = 0;
    CopyToRing (Bytes + Offset, Pad.Words, (Left < sizeof (Pad)) ? Left : sizeof (Pad));
    Producer += Left;
    Offset    = 0;
  }

  Command->Header.type  = Type;
  Command->Header.size  = Size;
  Command->Header.seqno = ++Gfx->Seqno;
  CopyToRing (Bytes + Offset, Command->Words, Size);
  Producer += Size;

  //
  // The message is written before the count that publishes it, and the count
  // before the consumer's idle flag is read.
  //
  Gfx->SubmitProducer = Producer;
  Barrier ();
  Write64 (Ring + RING_PRODUCER_OFFSET, Producer);
  Barrier ();
  if (Read32 (Ring + RING_CONSUMER_IDLE_OFFSET) == 1) {
    Write32 (Gfx->Registers + RGFX_REG_DOORBELL, RGFX_CONTROL_RING_ID);
  }

  *Seqno = Command->Header.seqno;
  return ReimsGraphicsOk;
}

/**
  Poll the completion ring for the answer to command Seqno: a COMPLETE, or a
  message of type Reply. Every other message is consumed and dropped. As much
  of the answer's payload as Answer holds goes there, and its type and payload
  size to Type and PayloadSize.
**/
static REIMS_GRAPHICS_STATUS
WaitFor (
  REIMS_GRAPHICS         *Gfx,
  uint64_t               Seqno,
  uint32_t               Reply,
  REIMS_GRAPHICS_ANSWER  *Answer,
  uint32_t               *Type,
  uint32_t               *PayloadSize
  )
{
  uint64_t               Ring;
  uint64_t               Bytes;
  uint64_t               Producer;
  uint64_t               Consumer;
  uint64_t               Until;
  uint32_t               Offset;
  uint32_t               Left;
  uint32_t               Copy;
  int                    Found;
  REIMS_GRAPHICS_HEADER  Message;

  Ring     = Gfx->Region + RGFX_CONTROL_COMPLETE_OFFSET;
  Bytes    = Ring + RGFX_RING_HEADER_SIZE;
  Consumer = Gfx->EventConsumer;
  Until    = Deadline ();
  Found    = 0;
  for ( ; ; ) {
    //
    // The count is read before the messages it publishes.
    //
    Producer = Read64 (Ring + RING_PRODUCER_OFFSET);
    Barrier ();
    while (!Found && (Consumer != Producer)) {
      Offset = (uint32_t)Consumer & RING_MASK;
      Left   = RGFX_CONTROL_RING_DATA_SIZE - Offset;
      if (Left < sizeof (Message)) {
        if (Left > Producer - Consumer) {
          return ReimsGraphicsDeviceError;
        }

        Consumer += Left;
        continue;
      }

      CopyFromRing (Message.Words, Bytes + Offset, sizeof (Message));
      if ((Message.Header.size < ((Message.Header.type == RGFX_MSG_PAD) ? RING_MSG_ALIGN : sizeof (Message))) ||
          ((Message.Header.size % RING_MSG_ALIGN) != 0) ||
          (Message.Header.size > Left) ||
          (Message.Header.size > Producer - Consumer))
      {
        return ReimsGraphicsDeviceError;
      }

      if ((Message.Header.type != RGFX_MSG_PAD) && (Message.Header.seqno == Seqno) &&
          ((Message.Header.type == RGFX_EVT_COMPLETE) || (Message.Header.type == Reply)))
      {
        *Type        = Message.Header.type;
        *PayloadSize = Message.Header.size - sizeof (Message);
        Copy         = (*PayloadSize < sizeof (*Answer)) ? *PayloadSize : sizeof (*Answer);
        CopyFromRing (Answer->Words, Bytes + Offset + sizeof (Message), Copy & ~(uint32_t)7);
        Found = 1;
      }

      Consumer += Message.Header.size;
    }

    //
    // The messages are read before the room they took is given back.
    //
    Gfx->EventConsumer = Consumer;
    Barrier ();
    Write64 (Ring + RING_CONSUMER_OFFSET, Consumer);
    if (Found) {
      return ReimsGraphicsOk;
    }

    if (Counter () > Until) {
      return ReimsGraphicsTimeout;
    }
  }
}

/**
  The 32-bit value at byte Offset, a multiple of 4, of an answer's payload.
**/
static uint32_t
AnswerU32 (
  const REIMS_GRAPHICS_ANSWER  *Answer,
  uint64_t                     Offset
  )
{
  return (uint32_t)(Answer->Words[Offset / 8] >> (8 * (Offset % 8)));
}

/**
  Reset the device, adopt the control rings' counters and tell the device the
  driver is ready.
**/
static void
OpenRings (
  REIMS_GRAPHICS  *Gfx
  )
{
  Write32 (Gfx->Registers + RGFX_REG_STATUS, RGFX_STATUS_RESET);
  Barrier ();
  Gfx->SubmitProducer = Read64 (Gfx->Region + RGFX_CONTROL_SUBMIT_OFFSET + RING_PRODUCER_OFFSET);
  Gfx->EventConsumer  = Read64 (Gfx->Region + RGFX_CONTROL_COMPLETE_OFFSET + RING_CONSUMER_OFFSET);
  Gfx->Seqno          = 0;
  Barrier ();
  Write32 (Gfx->Registers + RGFX_REG_STATUS, RGFX_STATUS_DRIVER_READY);
}

/**
  Ask GET_DISPLAYS for the first connected display and its preferred mode.
  The display stays the first one named, or 0, and the mode 0x0 when the
  answer names none.
**/
static void
ChooseMode (
  REIMS_GRAPHICS  *Gfx
  )
{
  REIMS_GRAPHICS_COMMAND  Command;
  REIMS_GRAPHICS_ANSWER   Answer;
  uint64_t                Seqno;
  uint32_t                Type;
  uint32_t                Size;
  uint32_t                Count;
  uint32_t                Index;
  uint64_t                Offset;
  uint64_t                Display;
  uint64_t                Mode;
  uint32_t                Preferred;
  uint32_t                Modes;

  Gfx->Display = 0;
  Gfx->Width   = 0;
  Gfx->Height  = 0;

  if ((Submit (Gfx, RGFX_CMD_GET_DISPLAYS, &Command, sizeof (Command.GetDisplays), &Seqno) != ReimsGraphicsOk) ||
      (WaitFor (Gfx, Seqno, RGFX_EVT_DISPLAYS, &Answer, &Type, &Size) != ReimsGraphicsOk) ||
      (Type != RGFX_EVT_DISPLAYS))
  {
    return;
  }

  //
  // The payload: the display count, then the display records from
  // RGFX_DISPLAYS_ENTRIES_OFFSET into the message, each followed by its modes.
  //
  Size = (Size < sizeof (Answer)) ? Size : sizeof (Answer);
  if (Size < sizeof (uint32_t)) {
    return;
  }

  Count  = AnswerU32 (&Answer, 0);
  Offset = RGFX_DISPLAYS_ENTRIES_OFFSET - sizeof (RgfxMsgHeader);
  for (Index = 0; Index < Count && Offset + sizeof (RgfxDisplayEntry) <= Size; Index++) {
    Display   = Offset;
    Offset   += sizeof (RgfxDisplayEntry);
    Preferred = AnswerU32 (&Answer, Display + __builtin_offsetof (RgfxDisplayEntry, preferred_mode));
    Modes     = AnswerU32 (&Answer, Display + __builtin_offsetof (RgfxDisplayEntry, mode_count));
    if (Index == 0) {
      Gfx->Display = AnswerU32 (&Answer, Display + __builtin_offsetof (RgfxDisplayEntry, id));
    }

    if ((AnswerU32 (&Answer, Display + __builtin_offsetof (RgfxDisplayEntry, connected)) != 0) &&
        (Preferred < Modes) &&
        (Offset + ((uint64_t)Preferred + 1) * sizeof (RgfxModeEntry) <= Size))
    {
      Mode = Offset + (uint64_t)Preferred * sizeof (RgfxModeEntry);
      if ((AnswerU32 (&Answer, Mode + __builtin_offsetof (RgfxModeEntry, width)) != 0) &&
          (AnswerU32 (&Answer, Mode + __builtin_offsetof (RgfxModeEntry, height)) != 0))
      {
        Gfx->Display = AnswerU32 (&Answer, Display + __builtin_offsetof (RgfxDisplayEntry, id));
        Gfx->Width   = AnswerU32 (&Answer, Mode + __builtin_offsetof (RgfxModeEntry, width));
        Gfx->Height  = AnswerU32 (&Answer, Mode + __builtin_offsetof (RgfxModeEntry, height));
        return;
      }
    }

    Offset += (uint64_t)Modes * sizeof (RgfxModeEntry);
  }
}

/**
  The mode's row bytes, and whether its framebuffer fits the scanout's room in
  the region, from RGFX_ALLOCATIONS_OFFSET.
**/
static int
ModeFits (
  REIMS_GRAPHICS  *Gfx
  )
{
  uint64_t  Stride;
  uint64_t  Room;

  Stride = ((uint64_t)Gfx->Width * 4 + REIMS_GRAPHICS_STRIDE_ALIGN - 1) &
           ~(uint64_t)(REIMS_GRAPHICS_STRIDE_ALIGN - 1);
  Room = Gfx->RegionSize - RGFX_ALLOCATIONS_OFFSET;
  Room = (Room < RGFX_SCANOUT_SPAN) ? Room : RGFX_SCANOUT_SPAN;
  if ((Gfx->Width == 0) || (Gfx->Height == 0) || (Stride * Gfx->Height > Room)) {
    return 0;
  }

  Gfx->Stride = (uint32_t)Stride;
  return 1;
}

/**
  Send SET_SCANOUT for the mode at RGFX_ALLOCATIONS_OFFSET, and wait for its
  COMPLETE.
**/
static REIMS_GRAPHICS_STATUS
SetScanout (
  REIMS_GRAPHICS  *Gfx
  )
{
  REIMS_GRAPHICS_COMMAND  Command;
  REIMS_GRAPHICS_ANSWER   Answer;
  REIMS_GRAPHICS_STATUS   Status;
  uint64_t                Seqno;
  uint32_t                Type;
  uint32_t                Size;

  Command.SetScanout.display = Gfx->Display;
  Command.SetScanout.width   = Gfx->Width;
  Command.SetScanout.height  = Gfx->Height;
  Command.SetScanout.stride  = Gfx->Stride;
  Command.SetScanout.offset  = RGFX_ALLOCATIONS_OFFSET;
  Status                     = Submit (Gfx, RGFX_CMD_SET_SCANOUT, &Command, sizeof (Command.SetScanout), &Seqno);
  if (Status == ReimsGraphicsOk) {
    Status = WaitFor (Gfx, Seqno, RGFX_EVT_COMPLETE, &Answer, &Type, &Size);
  }

  if ((Status == ReimsGraphicsOk) &&
      ((Size < sizeof (uint32_t)) || (AnswerU32 (&Answer, 0) != RGFX_STATUS_OK)))
  {
    Status = ReimsGraphicsRefused;
  }

  return Status;
}

/* ------ The entry point ------ */

/**
  Ambrosia's entry point of the module (AmbrosiaGraphics.h), the ELF's e_entry
  (Makefile). It returns 0 with the framebuffer filled when the device shows
  it, 1 when there is no device on bus 0, and 2 when the device can't be
  driven, saying why in the log.
**/
int32_t
ReimsGraphicsModuleEntry (
  const AMBROSIA_SERVICES  *Services,
  AMBROSIA_FRAMEBUFFER     *Framebuffer
  )
{
  REIMS_GRAPHICS         Gfx;
  REIMS_GRAPHICS_STATUS  Status;
  LOG_LINE               Line;
  uint64_t               Config;
  uint32_t               Device;
  uint32_t               Function;
  uint16_t               Command;
  uint64_t               RegistersSize;
  uint64_t               RegionBarSize;
  int                    RegistersWide;
  int                    RegionWide;
  uint32_t               Version;

  if (Services->Revision < AMBROSIA_GRAPHICS_REVISION) {
    return 2;
  }

  Config = FindDevice (Services, &Device, &Function);
  if (Config == 0) {
    SayText (Services, "no device on bus 0");
    return 1;
  }

  //
  // Size both BARs with decoding off, then place BAR 0 at the window's start
  // and BAR 2 at the first address after it aligned to its size, where the
  // kernel keeps them. A BAR's size is a power of two.
  //
  Command = Read16 (Config + PCI_COMMAND);
  Write16 (Config + PCI_COMMAND, (uint16_t)(Command & ~PCI_COMMAND_MEMORY));
  RegistersSize = BarSize (Config, RGFX_PCI_BAR_REGISTERS, &RegistersWide);
  RegionBarSize = BarSize (Config, RGFX_PCI_BAR_REGION, &RegionWide);
  if ((RegistersSize == 0) || (RegionBarSize == 0)) {
    SayText (Services, "BAR 0 or 2 is not memory");
    return 2;
  }

  Gfx.Registers = (Services->WindowBase + RegistersSize - 1) & ~(RegistersSize - 1);
  Gfx.Region    = (Gfx.Registers + RegistersSize + RegionBarSize - 1) & ~(RegionBarSize - 1);
  if (Gfx.Region + RegionBarSize > Services->WindowBase + Services->WindowSize) {
    SayText (Services, "its BARs don't fit the PCIe window");
    return 2;
  }

  SetBar (Config, RGFX_PCI_BAR_REGISTERS, RegistersWide, Gfx.Registers);
  SetBar (Config, RGFX_PCI_BAR_REGION, RegionWide, Gfx.Region);
  Write16 (Config + PCI_COMMAND, (uint16_t)(Command | PCI_COMMAND_MEMORY));

  if (Read32 (Gfx.Registers + RGFX_REG_MAGIC) != RGFX_MAGIC) {
    SayText (Services, "BAR 0 has no MAGIC");
    return 2;
  }

  //
  // The version names the region's layout and the messages, and a driver of
  // another version misreads them (docs/graphics-kext/design.md, "The wire's
  // version").
  //
  Version = Read32 (Gfx.Registers + RGFX_REG_VERSION);
  if (Version != RGFX_VERSION) {
    Line.Length = 0;
    Append (&Line, "reims-graphics: the device speaks wire version ");
    AppendDecimal (&Line, Version);
    Append (&Line, ", this module version ");
    AppendDecimal (&Line, RGFX_VERSION);
    Say (Services, &Line);
    return 2;
  }

  Gfx.RegionSize = Read32 (Gfx.Registers + RGFX_REG_REGION_SIZE_LO) |
                   ((uint64_t)Read32 (Gfx.Registers + RGFX_REG_REGION_SIZE_HI) << 32);
  if ((Gfx.RegionSize > RegionBarSize) || (Gfx.RegionSize <= RGFX_ALLOCATIONS_OFFSET)) {
    Line.Length = 0;
    Append (&Line, "reims-graphics: a region of 0x");
    AppendHex (&Line, Gfx.RegionSize, 1);
    Append (&Line, " bytes in a BAR of 0x");
    AppendHex (&Line, RegionBarSize, 1);
    Say (Services, &Line);
    return 2;
  }

  OpenRings (&Gfx);
  ChooseMode (&Gfx);
  if (!ModeFits (&Gfx)) {
    Gfx.Width  = REIMS_GRAPHICS_FALLBACK_WIDTH;
    Gfx.Height = REIMS_GRAPHICS_FALLBACK_HEIGHT;
    if (!ModeFits (&Gfx)) {
      SayText (Services, "no mode fits the region");
      return 2;
    }
  }

  Status = SetScanout (&Gfx);
  if (Status != ReimsGraphicsOk) {
    Line.Length = 0;
    Append (&Line, "reims-graphics: SET_SCANOUT: ");
    Append (&Line, StatusName (Status));
    Say (Services, &Line);
    return 2;
  }

  Line.Length = 0;
  Append (&Line, "reims-graphics at 00:");
  AppendHex (&Line, Device, 2);
  Append (&Line, ".");
  AppendHex (&Line, Function, 1);
  Append (&Line, ": ");
  AppendDecimal (&Line, Gfx.Width);
  Append (&Line, "x");
  AppendDecimal (&Line, Gfx.Height);
  Append (&Line, ", framebuffer 0x");
  AppendHex (&Line, Gfx.Region + RGFX_ALLOCATIONS_OFFSET, 1);
  Append (&Line, ", registers 0x");
  AppendHex (&Line, Gfx.Registers, 1);
  Say (Services, &Line);

  Framebuffer->Base     = Gfx.Region + RGFX_ALLOCATIONS_OFFSET;
  Framebuffer->Width    = Gfx.Width;
  Framebuffer->Height   = Gfx.Height;
  Framebuffer->Stride   = Gfx.Stride;
  Framebuffer->Reserved = 0;
  return 0;
}
