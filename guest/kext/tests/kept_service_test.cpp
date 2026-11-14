// The native test of RgfxKeptService.hpp: the accelerator keeps the framebuffer its lookup found
// started, hands it out again only while it is started and not terminated, and lets go of one
// that stopped or was terminated, so the lookup is made again and finds the framebuffer that
// replaced it: a stale framebuffer is never handed out, and every reference taken is let go.
#include "RgfxKeptService.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "kept_service_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

struct Fb {
	bool started = false;
	bool inactive = false;
	int refs = 1; // the registry's
};

struct Ops {
	bool live(Fb *fb) { return fb->started && !fb->inactive; }
	void retain(Fb *fb) { fb->refs++; }
};

void release(Fb *fb)
{
	if (fb)
		fb->refs--;
}

// copyFramebuffer as the kext makes it: the kept one where it is live, else the lookup (`attached`,
// the PCI device's client, null where none is), offered to the keep. Retained for the caller.
Fb *copy(Fb *&kept, Fb *attached, int *lookups)
{
	Ops ops;
	Fb *dropped = nullptr;
	Fb *fb = rgfxKeptTake(kept, ops, &dropped);
	release(dropped);
	if (fb)
		return fb;
	(*lookups)++;
	if (attached == nullptr)
		return nullptr;
	ops.retain(attached);
	rgfxKeptOffer(kept, ops, attached);
	return attached;
}
} // namespace

int main()
{
	Ops ops;
	// Nothing kept: no lookup's answer kept until it has started.
	{
		Fb *kept = nullptr;
		Fb fb;
		int lookups = 0;
		Fb *got = copy(kept, &fb, &lookups);
		check(got == &fb && kept == nullptr && lookups == 1, "an attached framebuffer not yet "
		                                                     "started is handed out, not kept");
		release(got);
		got = copy(kept, &fb, &lookups);
		check(lookups == 2, "and looked up again");
		release(got);
		check(fb.refs == 1, "every reference let go");
		check(copy(kept, nullptr, &lookups) == nullptr && lookups == 3, "none attached, none");
	}
	// Kept once started, then handed out with no lookup.
	{
		Fb *kept = nullptr;
		Fb fb;
		fb.started = true;
		int lookups = 0;
		Fb *got = copy(kept, &fb, &lookups);
		check(got == &fb && kept == &fb && lookups == 1, "the started framebuffer is kept");
		check(fb.refs == 3, "the caller's reference and the keep's");
		release(got);
		for (int i = 0; i < 4; i++) {
			got = copy(kept, &fb, &lookups);
			check(got == &fb && lookups == 1, "then handed out from the keep");
			release(got);
		}
		check(fb.refs == 2, "the keep's reference alone stays");
		// Its stop begins: let go, looked up again, and the stopped one handed out by the lookup
		// as before the keep, never kept.
		fb.started = false;
		got = copy(kept, &fb, &lookups);
		check(kept == nullptr && lookups == 2, "a stopped framebuffer is let go of the keep");
		release(got);
		check(fb.refs == 1, "the keep's reference let go");
	}
	// Terminated while kept, a new framebuffer attached in its place (a restart): the new one is
	// found and kept, the old one never handed out again.
	{
		Fb *kept = nullptr;
		Fb old, fresh;
		old.started = fresh.started = true;
		int lookups = 0;
		release(copy(kept, &old, &lookups));
		check(kept == &old, "kept");
		old.inactive = true;
		Fb *got = copy(kept, &fresh, &lookups);
		check(got == &fresh && kept == &fresh && lookups == 2,
		      "a terminated framebuffer is let go, and its successor found and kept");
		release(got);
		check(old.refs == 1 && fresh.refs == 2, "the old one's keep let go, the new one's held");
		// Gone with nothing in its place: none handed out.
		fresh.inactive = true;
		check(copy(kept, nullptr, &lookups) == nullptr && kept == nullptr && fresh.refs == 1,
		      "a terminated framebuffer with no successor: none");
	}
	// The offer keeps nothing where one is kept already, and nothing that is not live.
	{
		Fb *kept = nullptr;
		Fb a, b;
		a.started = b.started = true;
		check(rgfxKeptOffer(kept, ops, &a) && kept == &a && a.refs == 2, "the first offer is kept");
		check(!rgfxKeptOffer(kept, ops, &b) && kept == &a && b.refs == 1,
		      "a second offer is not kept");
		Fb *none = nullptr;
		check(!rgfxKeptOffer(none, ops, static_cast<Fb *>(nullptr)), "no answer, nothing kept");
		b.inactive = true;
		check(!rgfxKeptOffer(none, ops, &b) && none == nullptr, "a terminated one is not kept");
	}
	std::printf("kept_service_test: %d checks passed\n", checks);
	return 0;
}
