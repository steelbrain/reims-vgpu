-- reposition.applescript — move a window along a path as fast as Apple Events
-- allow, for a fixed duration.
--
-- This is the fallback motion, and it is not a pointer drag. A real drag needs
-- CGEventPost (see drag.c), whose events this guest silently discards because
-- the posting process is not trusted for Accessibility and TCC.db cannot be
-- written here: there is no passwordless sudo and SIP's Filesystem Protections
-- are on.
--
-- It drives the target application through its **own** Standard Suite rather
-- than through System Events' accessibility objects. That is not a style
-- choice. The `System Events` route needs assistive access, which TCC grants
-- only by hand in System Preferences and which exactly one rail of six has;
-- every other rail answers `osascript is not allowed assistive access (-25211)`
-- and the probe reads that as an app with no windows. An application's own
-- `bounds of front window` needs only Apple Events consent, and that panel has
-- a default button the harness can press over QMP.
--
-- What that costs is stated rather than glossed: the window server sees a
-- sequence of window moves instead of a pointer held across a title bar, so any
-- work specific to a drag session is missing. What it keeps is the part the
-- device sees — a large window changing position at high rate, and everything
-- behind it recomposited.
--
-- Runs for a duration rather than a step count, and cannot be paced: each
-- `set bounds` is a synchronous round trip, so the rate is whatever that costs
-- and `--hz` has no meaning in this mode. Taking a step count instead was a real
-- defect — asking for 15 s at 12 Hz ran 180 steps flat out and finished in
-- 1.9 s, so the run measured a third of the intended window and reported it as
-- if it were the whole one.
--
-- `bounds` is {left, top, right, bottom}, so moving without resizing means
-- carrying the width and height across every step. They are read once up front
-- rather than per step, which keeps the loop to one round trip.
--
-- Returns the number of steps performed; the caller times the run, because
-- AppleScript's `current date` has one-second resolution and cannot.
on run argv
	set appName to item 1 of argv
	set x0 to (item 2 of argv) as integer
	set y0 to (item 3 of argv) as integer
	set secs to (item 4 of argv) as integer
	set ampX to (item 5 of argv) as integer
	set ampY to (item 6 of argv) as integer

	set done to 0
	tell application appName
		set b to bounds of front window
		set w to (item 3 of b) - (item 1 of b)
		set h to (item 4 of b) - (item 2 of b)
		set t0 to current date
		repeat with i from 1 to 1000000
			if ((current date) - t0) ≥ secs then exit repeat
			-- Two incommensurate periods, so the path is not a straight slide,
			-- which is the one motion a compositor may coalesce. Integer
			-- arithmetic only: AppleScript has no cheap sine and the point is
			-- the motion, not its exact shape.
			set dx to ((i * 7) mod (2 * ampX)) - ampX
			set dy to ((i * 11) mod (2 * ampY)) - ampY
			set nx to x0 + dx
			set ny to y0 + dy
			set bounds of front window to {nx, ny, nx + w, ny + h}
			set done to done + 1
		end repeat
	end tell
	return done
end run
