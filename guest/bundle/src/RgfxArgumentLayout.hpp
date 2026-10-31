/*
 * RgfxArgumentLayout: an argument buffer's layout as the host answers ARGUMENT_LAYOUT
 * (RgfxArgumentLayoutReply in the wire header): its encodedLength and alignment, and where each
 * argument index lands, with the layouts of nested argument buffers and of struct elements.
 *
 * The reply is a flat list of layouts, the first the argument buffer's, and of members, which
 * each layout holds a consecutive run of; a member names a further layout by its index. Parsing
 * builds the tree once and checks every index and run against the reply's length, so a layout
 * here is one the host sent whole. `place(i)` is where argument index `i` lands: in the last
 * member whose indices hold it, at its element's offset, and inside a struct element as that
 * element's layout says, as Metal's encoder resolves an index two members claim (the later one).
 *
 * Header-only and free of Metal's headers, so the serialiser's native tests build it as the
 * bundle does. Our own code.
 */
#ifndef RGFX_ARGUMENT_LAYOUT_HPP_
#define RGFX_ARGUMENT_LAYOUT_HPP_

#include <stdint.h>

#include <cstring>
#include <memory>
#include <vector>

#include "reims_graphics_wire.h"

struct RgfxArgLayout;

/// One member of a layout: an argument, or an array of them (RgfxArgumentMemberRecord).
struct RgfxArgMember {
	uint32_t id = 0;
	uint32_t count = 1;
	uint32_t idsPerElement = 1;
	uint32_t kind = RGFX_ARGUMENT_CONSTANT;
	uint64_t offset = 0;
	uint64_t stride = 0;
	uint64_t size = 0;
	/// A nested argument buffer's layout (RGFX_ARGUMENT_BUFFER), or one struct element's
	/// (RGFX_ARGUMENT_STRUCT); null otherwise.
	std::shared_ptr<const RgfxArgLayout> layout;
};

/// Where an argument index lands.
struct RgfxArgPlace {
	uint32_t kind;
	uint64_t offset;
	uint64_t size;
	std::shared_ptr<const RgfxArgLayout> nested;
};

struct RgfxArgLayout {
	uint64_t encodedLength = 0;
	uint64_t alignment = 1;
	std::vector<RgfxArgMember> members;

	/// Where argument `index` lands, from the arguments' start; false where no member holds it.
	bool place(uint32_t index, RgfxArgPlace *out) const { return placeFrom(index, 0, out, 0); }

	/// One past the highest argument index the members take (0 for none).
	uint32_t idSpan() const
	{
		uint64_t span = 0;
		for (const RgfxArgMember &m : members) {
			const uint64_t end =
			    uint64_t(m.id) + uint64_t(m.count) * (m.idsPerElement ? m.idsPerElement : 1);
			if (end > span)
				span = end;
		}
		return span > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(span);
	}

	/// The layout the `length`-byte reply at `bytes` gives; null where it is not whole (a count,
	/// a run or an index past the reply, a layout that holds itself).
	static std::shared_ptr<const RgfxArgLayout> parse(const uint8_t *bytes, uint64_t length)
	{
		RgfxArgumentLayoutReply head;
		if (bytes == nullptr || length < sizeof(head))
			return nullptr;
		std::memcpy(&head, bytes, sizeof(head));
		const uint64_t layoutsAt = sizeof(head);
		const uint64_t membersAt =
		    layoutsAt + uint64_t(head.layout_count) * sizeof(RgfxArgumentLayoutRecord);
		const uint64_t end =
		    membersAt + uint64_t(head.member_count) * sizeof(RgfxArgumentMemberRecord);
		if (head.layout_count == 0 || end > length || head.needed > length)
			return nullptr;
		std::vector<RgfxArgumentLayoutRecord> layouts(head.layout_count);
		std::memcpy(layouts.data(), bytes + layoutsAt, layouts.size() * sizeof(layouts[0]));
		std::vector<RgfxArgumentMemberRecord> members(head.member_count);
		if (!members.empty())
			std::memcpy(members.data(), bytes + membersAt, members.size() * sizeof(members[0]));
		std::vector<std::shared_ptr<const RgfxArgLayout>> built(layouts.size());
		std::vector<uint8_t> state(layouts.size(), 0); // 0 unbuilt, 1 building, 2 built
		return build(0, layouts, members, built, state, 0);
	}

private:
	/// Bounds the struct nesting `place` recurses through, as the host's reply bounds it.
	static constexpr unsigned kMaxDepth = 64;

	bool placeFrom(uint32_t index, uint64_t base, RgfxArgPlace *out, unsigned depth) const
	{
		if (depth > kMaxDepth)
			return false;
		for (size_t i = members.size(); i-- > 0;) {
			const RgfxArgMember &m = members[i];
			if (index < m.id)
				continue;
			const uint32_t rel = index - m.id;
			const uint32_t per = m.idsPerElement ? m.idsPerElement : 1;
			const uint32_t element = rel / per;
			if (element >= m.count)
				continue;
			const uint64_t offset = base + m.offset + m.stride * element;
			if (m.kind == RGFX_ARGUMENT_STRUCT) {
				if (m.layout && m.layout->placeFrom(rel % per, offset, out, depth + 1))
					return true;
				continue; // the element does not hold it: the members before may
			}
			*out = {m.kind, offset, m.size, m.layout};
			return true;
		}
		return false;
	}

	static std::shared_ptr<const RgfxArgLayout>
	build(uint32_t at, const std::vector<RgfxArgumentLayoutRecord> &layouts,
	      const std::vector<RgfxArgumentMemberRecord> &members,
	      std::vector<std::shared_ptr<const RgfxArgLayout>> &built, std::vector<uint8_t> &state,
	      unsigned depth)
	{
		if (at >= layouts.size() || depth > kMaxDepth || state[at] == 1)
			return nullptr;
		if (state[at] == 2)
			return built[at];
		state[at] = 1;
		const RgfxArgumentLayoutRecord &r = layouts[at];
		if (uint64_t(r.first_member) + r.member_count > members.size()) {
			state[at] = 0;
			return nullptr;
		}
		auto layout = std::make_shared<RgfxArgLayout>();
		layout->encodedLength = r.encoded_length;
		layout->alignment = r.alignment ? r.alignment : 1;
		layout->members.reserve(r.member_count);
		for (uint32_t i = 0; i < r.member_count; i++) {
			const RgfxArgumentMemberRecord &w = members[r.first_member + i];
			RgfxArgMember m;
			m.id = w.id;
			m.count = w.count;
			m.idsPerElement = w.ids_per_element;
			m.kind = w.kind;
			m.offset = w.offset;
			m.stride = w.stride;
			m.size = w.size;
			if (w.layout != RGFX_ARGUMENT_NO_LAYOUT) {
				m.layout = build(w.layout, layouts, members, built, state, depth + 1);
				if (!m.layout) {
					state[at] = 0;
					return nullptr;
				}
			}
			layout->members.push_back(std::move(m));
		}
		built[at] = layout;
		state[at] = 2;
		return layout;
	}
};

#endif // RGFX_ARGUMENT_LAYOUT_HPP_
