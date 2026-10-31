// RgfxPipeClients: the display pipe's clients a framebuffer serves at once (ReimsGraphics), and
// what the display shows through them. Header-only and free of the kernel's headers, so its
// native test (tests/pipe_clients_test.cpp) builds it as the kext does.
//
// IOPresentment opens more than one pipe client on the accelerator for a display and associates
// each with it (measured on a macOS 26 guest: two per WindowServer, and another for a batch of
// transactions), and any of them may submit a transaction. So the framebuffer keeps every
// associated client, not the last one: the host's answer to a PRESENT_SURFACE goes to each, and
// the one whose transaction it answers takes it (the others hold no such seqno); a VBLANK goes
// to each client registered for them.
//
// While the pipe shows an IOSurface, the host draws it in place of the scanout. When the last
// client goes (its process closed it, or died), nothing would replace that surface, and the
// display would stay on the dead process's last frame while a successor draws into the scanout:
// so the framebuffer then asks the host to show the scanout again (PRESENT_SURFACE 0).
//
// Our own code.
#ifndef RGFX_PIPE_CLIENTS_HPP_
#define RGFX_PIPE_CLIENTS_HPP_

#include <stdint.h>

template <typename Client, uint32_t Max> class RgfxPipeClients {
public:
	// Adds `client` (again: nothing changes). False where `Max` are held.
	bool add(Client client)
	{
		if (find(client) != Max)
			return true;
		for (uint32_t i = 0; i < Max; i++) {
			if (!clients_[i]) {
				clients_[i] = client;
				vblank_[i] = false;
				return true;
			}
		}
		return false;
	}

	// Removes `client`: true where it was held. `*showScanout` is set where it was the last and
	// the display shows a surface, which is forgotten: the caller asks for the scanout.
	bool remove(Client client, bool *showScanout)
	{
		*showScanout = false;
		uint32_t i = find(client);
		if (i == Max)
			return false;
		clients_[i] = Client();
		vblank_[i] = false;
		if (count() == 0 && shown_ != 0) {
			shown_ = 0;
			*showScanout = true;
		}
		return true;
	}

	// Whether `client` is held.
	bool holds(Client client) const { return find(client) != Max; }

	// Whether `client` wants VBLANKs (its registration for them).
	void setVblank(Client client, bool wanted)
	{
		uint32_t i = find(client);
		if (i != Max)
			vblank_[i] = wanted;
	}

	// Whether any client wants VBLANKs.
	bool anyVblank() const
	{
		for (uint32_t i = 0; i < Max; i++) {
			if (clients_[i] && vblank_[i])
				return true;
		}
		return false;
	}

	// The display shows IOSurface `surface` (0: the scanout), as a PRESENT_SURFACE asked.
	void shows(uint32_t surface) { shown_ = surface; }
	uint32_t shown() const { return shown_; }

	// The clients, and those wanting VBLANKs, into `out` (room for `Max`): their count.
	uint32_t all(Client *out) const { return collect(out, false); }
	uint32_t wantingVblank(Client *out) const { return collect(out, true); }

	uint32_t count() const
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < Max; i++) {
			if (clients_[i])
				n++;
		}
		return n;
	}

private:
	uint32_t find(Client client) const
	{
		if (!client)
			return Max;
		for (uint32_t i = 0; i < Max; i++) {
			if (clients_[i] == client)
				return i;
		}
		return Max;
	}

	uint32_t collect(Client *out, bool vblankOnly) const
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < Max; i++) {
			if (clients_[i] && (!vblankOnly || vblank_[i]))
				out[n++] = clients_[i];
		}
		return n;
	}

	Client clients_[Max] = {};
	bool vblank_[Max] = {};
	uint32_t shown_ = 0;
};

#endif // RGFX_PIPE_CLIENTS_HPP_
