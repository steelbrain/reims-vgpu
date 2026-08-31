# menu-close-probe

Photographs the dock context menu's **close-in animation** and says whether the
region it vacates shows the desktop or shows black.

The reported defect: right-click a dock icon, then left-click to dismiss, and a
black rectangle is left behind for the duration of the close animation, where
the desktop should show through. It is intermittent across boots, so a run that
does not see it is not evidence that it is gone — see *Reading the result*.

## Running it

```sh
scripts/menu-close-probe/menu-close-probe.sh [TRIALS]
```

| Variable | Meaning |
|---|---|
| `QPID` | the guest to drive; defaults to the running `qemu-system-x86_64` |
| `OUT` | where frames land; defaults to `$TMPDIR/menu-close-probe` |
| `ICON_X`, `ICON_Y` | the dock icon to right-click, in **guest** pixels |
| `AWAY_X`, `AWAY_Y` | where to click to dismiss, in **guest** pixels |

Exit codes are the verdict, and the third one matters as much as the other two:

| Exit | Verdict | Meaning |
|---|---|---|
| 0 | `CLEAN` | the animation was sampled and the vacated region was not black |
| 1 | `BLACK_RECTANGLE` | the animation was sampled and the region went black |
| 2 | — | the run **did not sample the animation** and says nothing about the device |

## Why it aims the capture instead of chasing the animation

The host capture costs about **1 430 ms** on the reference rig (six timed
captures, 8 562 ms). The close animation is about **250 ms**. A loop of captures
taken after the dismiss click cannot sample it — every frame lands after it is
over, and every trial reports the same settled mean. The probe's first version
did exactly that and never produced a valid reading.

The capture's cost is nearly all setup: the frame is grabbed **late**, roughly
500 ms in. That is measurable — start a capture and open the menu 150 ms later,
and the menu is in the resulting frame — and it is what makes the animation
reachable. Start the capture, wait, *then* dismiss, and the grab lands inside
the fade.

The delay is calibrated per run rather than assumed, because it is a property of
the host compositor and not of the guest. A scan over five delays produces the
fade as a ramp; on the reference rig:

```text
menu open  147.4      no menu  96.0
dismiss at 0.34 s      88.4    already closed
dismiss at 0.38 s      92.7
dismiss at 0.42 s     106.9
dismiss at 0.46 s     131.7    <- inside the fade
dismiss at 0.50 s     147.4    still open
```

If no delay lands strictly between the two settled states, the probe exits 2
rather than reporting a verdict on frames that never contained the animation.

## Why the menu rectangle is anchored, not differenced

Differencing a menu-open capture against a menu-closed one sounds like the way
to find the menu, and it does not work here. The two captures are more than a
second apart on a live desktop that repaints damage rectangles continuously, so
the changed-pixel bounding box comes back as the whole screen (measured:
`1279x704+0+15`). On a boot with the blank-desktop defect the background field
is itself churning between the two frames.

The menu's position is not actually unknown: it opens directly above the icon
this script right-clicked. The box is derived from that icon and the capture's
own dimensions, and then validated — if the box did not change when the menu
opened, the right-click did not open a menu and the probe exits 2.

## Why the score subtracts the no-menu frame

The score is the fraction of the menu box that is near-black in the animation
frame **and was not near-black with no menu on screen**. A wallpaper with dark
foliage in it reads near-black honestly; subtracting the settled no-menu frame
is what keeps the probe from reporting the desktop as the defect.

On the reference rig that residual sits at 0.031 and varies by under 0.001
across trials, so the 0.10 threshold has a wide margin under it.

## Reading the result

`CLEAN` means *this boot, this many trials*. The defect is reported as
intermittent across boots, and a single clean run does not establish a rate.
Repeat across boots, and quote the trial count and the worst fraction, not just
the verdict.
