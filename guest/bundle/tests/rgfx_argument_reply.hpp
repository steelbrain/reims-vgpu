// An ARGUMENT_LAYOUT answer as the host writes one (RgfxArgumentLayoutReply and its records),
// built for the bundle's tests: layout() starts a layout, and member() adds a member to the
// last one started, so each layout's members are one run, as the host's are.
#ifndef RGFX_ARGUMENT_REPLY_HPP_
#define RGFX_ARGUMENT_REPLY_HPP_

#include <cstring>
#include <vector>

#include "reims_graphics_wire.h"

struct ArgReply {
	std::vector<RgfxArgumentLayoutRecord> layouts;
	std::vector<RgfxArgumentMemberRecord> members;

	ArgReply &layout(uint64_t encoded_length, uint64_t alignment)
	{
		RgfxArgumentLayoutRecord r = {};
		r.encoded_length = encoded_length;
		r.alignment = alignment;
		r.first_member = static_cast<uint32_t>(members.size());
		layouts.push_back(r);
		return *this;
	}

	ArgReply &member(uint32_t id, uint32_t kind, uint64_t offset, uint64_t size,
	                 uint32_t count = 1, uint64_t stride = 0, uint32_t ids_per_element = 1,
	                 uint32_t nested = RGFX_ARGUMENT_NO_LAYOUT)
	{
		RgfxArgumentMemberRecord m = {};
		m.id = id;
		m.count = count;
		m.ids_per_element = ids_per_element;
		m.kind = kind;
		m.offset = offset;
		m.stride = stride;
		m.size = size;
		m.layout = nested;
		members.push_back(m);
		layouts.back().member_count++;
		return *this;
	}

	uint64_t length() const
	{
		return sizeof(RgfxArgumentLayoutReply) + layouts.size() * sizeof(RgfxArgumentLayoutRecord) +
		       members.size() * sizeof(RgfxArgumentMemberRecord);
	}

	std::vector<uint8_t> bytes() const
	{
		RgfxArgumentLayoutReply head = {};
		head.needed = length();
		head.layout_count = static_cast<uint32_t>(layouts.size());
		head.member_count = static_cast<uint32_t>(members.size());
		std::vector<uint8_t> out(static_cast<size_t>(length()));
		uint8_t *p = out.data();
		std::memcpy(p, &head, sizeof(head));
		p += sizeof(head);
		if (!layouts.empty())
			std::memcpy(p, layouts.data(), layouts.size() * sizeof(layouts[0]));
		p += layouts.size() * sizeof(layouts[0]);
		if (!members.empty())
			std::memcpy(p, members.data(), members.size() * sizeof(members[0]));
		return out;
	}
};

// The layout Metal gives the probe kernel's argument buffer (measured on the M3 Max, macOS 27;
// MoltenVK gives the same): a texture at 0; a buffer at 8 whose argument buffer holds a texture
// at 0 and a sampler at 8 (16 bytes); a sampler at 16; a float4 at 32; a plain buffer at 48.
inline ArgReply probeReply()
{
	ArgReply r;
	r.layout(64, 16)
	    .member(0, RGFX_ARGUMENT_TEXTURE, 0, 8)
	    .member(1, RGFX_ARGUMENT_BUFFER, 8, 8, 1, 0, 1, 1)
	    .member(2, RGFX_ARGUMENT_SAMPLER, 16, 8)
	    .member(3, RGFX_ARGUMENT_CONSTANT, 32, 16)
	    .member(4, RGFX_ARGUMENT_BUFFER, 48, 8);
	r.layout(16, 8)
	    .member(0, RGFX_ARGUMENT_TEXTURE, 0, 8)
	    .member(1, RGFX_ARGUMENT_SAMPLER, 8, 8);
	return r;
}

#endif // RGFX_ARGUMENT_REPLY_HPP_
