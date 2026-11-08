/*
 * RgfxVideoBits: an RBSP written bit by bit and made a NAL unit, for the video driver's parameter
 * set writers (RgfxVideoH264.hpp, RgfxVideoH265.hpp), and a NAL unit's payload read bit by bit,
 * for the slice header reader (RgfxVideoH265.hpp, currentReferences).
 */

#ifndef RGFX_VIDEO_BITS_HPP_
#define RGFX_VIDEO_BITS_HPP_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace rgfx::video {

// An RBSP, bit by bit, made a NAL unit's payload with its emulation prevention at the end.
class BitWriter {
public:
	void u(uint32_t bits, uint32_t value)
	{
		for (uint32_t i = bits; i-- > 0;)
			bit((value >> i) & 1);
	}

	void ue(uint32_t value)
	{
		uint64_t v = (uint64_t)value + 1;
		uint32_t length = 0;
		while ((v >> length) > 1)
			length++;
		u(length, 0);
		for (uint32_t i = length + 1; i-- > 0;)
			bit((uint32_t)((v >> i) & 1));
	}

	void se(int32_t value)
	{
		ue(value > 0 ? (uint32_t)(2 * (int64_t)value - 1) : (uint32_t)(-2 * (int64_t)value));
	}

	void flag(uint32_t value) { bit(value ? 1 : 0); }

	// Appends the start code, the NAL unit header's bytes and the payload, its trailing bits
	// added and every 0x000000 to 0x000003 run escaped.
	void appendNal(std::initializer_list<uint8_t> header, std::vector<uint8_t> &out)
	{
		bit(1);
		while (filled_ != 0)
			bit(0);
		out.insert(out.end(), {0, 0, 0, 1});
		out.insert(out.end(), header);
		uint32_t zeros = 0;
		for (uint8_t b : bytes_) {
			if (zeros >= 2 && b <= 3) {
				out.push_back(3);
				zeros = 0;
			}
			out.push_back(b);
			zeros = b == 0 ? zeros + 1 : 0;
		}
	}

private:
	void bit(uint32_t b)
	{
		if (filled_ == 0)
			bytes_.push_back(0);
		bytes_.back() |= (uint8_t)(b << (7 - filled_));
		filled_ = (filled_ + 1) % 8;
	}

	std::vector<uint8_t> bytes_;
	uint32_t filled_ = 0;
};

// A NAL unit's payload, read bit by bit with its emulation prevention bytes skipped. A read past
// the end, or an Exp-Golomb code longer than 32 bits, reads 0 and marks the reader failed, which
// every read after keeps.
class BitReader {
public:
	BitReader(const uint8_t *bytes, size_t size) : bytes_(bytes), size_(size) {}

	uint32_t u(uint32_t bits)
	{
		uint32_t v = 0;
		for (uint32_t i = 0; i < bits; i++)
			v = v << 1 | bit();
		return failed_ ? 0 : v;
	}

	uint32_t ue()
	{
		uint32_t zeros = 0;
		while (!failed_ && bit() == 0)
			if (++zeros > 31)
				failed_ = true;
		if (failed_)
			return 0;
		const uint64_t v = ((uint64_t)1 << zeros) - 1 + u(zeros);
		return failed_ || v > UINT32_MAX ? 0 : (uint32_t)v;
	}

	bool flag() { return bit() != 0; }

	bool failed() const { return failed_; }

private:
	uint32_t bit()
	{
		if (left_ == 0) {
			// 0x000003: the 3 is not the payload's.
			if (at_ < size_ && zeros_ >= 2 && bytes_[at_] == 3) {
				at_++;
				zeros_ = 0;
			}
			if (at_ >= size_) {
				failed_ = true;
				return 0;
			}
			byte_ = bytes_[at_++];
			zeros_ = byte_ == 0 ? zeros_ + 1 : 0;
			left_ = 8;
		}
		left_--;
		return (byte_ >> left_) & 1;
	}

	const uint8_t *bytes_;
	size_t size_;
	size_t at_ = 0;
	uint32_t zeros_ = 0;
	uint8_t byte_ = 0;
	uint32_t left_ = 0;
	bool failed_ = false;
};

} // namespace rgfx::video

#endif
