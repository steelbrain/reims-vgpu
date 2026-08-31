# This crate's half of observability

The vocabulary and the delivery path are the `reims-vgpu-observe` crate:

- `sink.rs` — the always-on `/tmp/reims-vgpu-fail.log` writer, background queue,
  flood self-detector, and the `verbose`/`when_verbose` entry points a hot path
  hands diagnostic work to instead of asking whether the log is open.
- `decline.rs` — `Decline` and `Refusal`. The vocabulary lives in the `slug()`
  arms and nowhere else; the 2 700-line `REGISTRY` that used to restate every
  type's file, emission site and slug list was removed, because a copy of the
  arms can only ever agree or disagree with them.
- `emit.rs` — the only reason-bearing line builder. `Emit::decline` requires a
  typed decline; `Emit::refusal` makes the successful status of a mixed status
  enum unrepresentable as a failure line.
- `slugs.rs` — crate-wide slug uniqueness, the one property no single `impl` can
  see and the one that decides whether `Emit::fail_once`'s latch silences a
  second check. Every rendered line claims its slug for the type that spelled
  it; a second claimant is reported by name, and panics in a test build.

It is a crate so the layers below the device can name their own refusal type
without depending on the device, and so nothing in it can reach back up into
`runtime`, `model`, or a backend.

What stays here is the two emitters that are *about this crate's* types:

- `ladder.rs` — the four object-list resolution rungs.
- `panic.rs` — a `catch_unwind` at a `reims_vgpu_qemu_*` entry point.

Both name `runtime` types, which is why they did not move. `mod.rs` re-exports
the crate's surface under the paths callers already write, so
`crate::observe::fail(…)` and `crate::observe::Decline` mean what they always
did.
