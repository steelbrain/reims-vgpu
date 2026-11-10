// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The manifest's records as bytes: each field little-endian at its offset in the
// shared declaration (<graft/ManifestFormat.h>), whatever the host's byte order.
#pragma once
#include "base.h"

#include <graft/ManifestFormat.h>

namespace graft::tool {

namespace manifest = graft::manifest;

// Every table of the header, in the order the header and the manifest hold them.
extern manifest::Table manifest::Header::*const ManifestTables[14];

// Append a record's bytes to out.
void encode(Bytes &out, const manifest::Header &record);
void encode(Bytes &out, const manifest::Library &record);
void encode(Bytes &out, const manifest::Import &record);
void encode(Bytes &out, const manifest::Site &record);
void encode(Bytes &out, const manifest::Class &record);
void encode(Bytes &out, const manifest::Slot &record);
void encode(Bytes &out, const manifest::Vtable &record);
void encode(Bytes &out, const manifest::Segment &record);
void encode(Bytes &out, const manifest::Fill &record);
void encode(Bytes &out, const manifest::SynthOverride &record);
void encode(Bytes &out, const manifest::SynthClass &record);
// A candidate, override or selector: one u32.
void encode(Bytes &out, uint32_t record);

// The record whose bytes start at at; the caller has bounded them.
manifest::Header decodeHeader(const uint8_t *at);
manifest::Site decodeSite(const uint8_t *at);

} // namespace graft::tool
