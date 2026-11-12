# reims-vgpu-window features

What someone at the host sees and does with a guest display's window. Every
entry was put in on purpose. Removing or changing one is a regression unless
it is done deliberately, in its own commit, with this file updated and the
reason given.

Each entry ends with its guard: `test: <module>::<name>`, a unit test in this
crate (`features::features_md_guards_exist` fails when one is gone), or
`manual`, a check on a VM boot. A change to this crate walks the `manual`
entries it can reach. A bare `test` is a guard not yet written.

The sections up to "On Linux" are the macOS window's, AppKit's. "On Linux"
is the window on Linux, SDL3's on Wayland or X11, entry by entry against
them; its `manual` checks are on a Linux host's guest boot.

## Switches

| variable | effect |
|---|---|
| `REIMS_VGPU_WINDOW=0` | QEMU stays headless under `-display none` (QEMU reads it) |
| `REIMS_VGPU_WINDOW_BORDERLESS=1` | borderless windows that fill their screens |
| `REIMS_VGPU_WINDOW_SCREENS=1,0` | display i opens on the host screen the list's i-th entry names |
| `resolution=2560x1600` | the mode the window opens for, and the guest is offered until the window's pixels are known (QEMU's reims-graphics-pci property; 1920x1080 by default) |
| `REIMS_VGPU_WINDOW_IMPL=sdl` | macOS, built with the `macos-sdl` feature (on by default): SDL3's window in place of AppKit's ("On macOS through SDL3" below) |

Unset or unparsable values give the defaults below.
`test: env::unset_or_unparsable_values_are_the_defaults`,
`test: env::the_window_is_appkits_unless_sdl_is_named`,
`test: env::borderless_takes_the_usual_words`,
`test: env::the_screen_list_is_all_or_nothing`

## Opening

- QEMU run with `-display none` on macOS opens one window per guest display
  by itself; a display created without a window config opens none. `manual`
- The VM is an app: Dock icon, ⌘Tab entry, a menu bar with Quit, Hide and
  Hide Others, and a Window menu with Minimize, Zoom, Full Screen and Release
  Input. `manual`
- The title is "Reims vGPU", with " (port N)" for a display on port N above
  0. `test: handle::a_window_s_title_is_reims_vgpu_and_its_port`
- A new window shows its mode a guest pixel per host point, shrunk to fit
  the screen's visible area, on whole pixels, a multiple of 16 wide and even
  high (the steps of the guest's modes, below): on a Retina screen the
  default 1920×1080 opens at 1920×1080 points, 3840×2160 pixels, where they
  fit, not at half size.
  `test: geometry::a_new_window_shows_a_guest_point_per_host_point`,
  `test: geometry::a_new_window_shrinks_to_the_visible_area`
- Display i opens on host screen i, or on the main screen when there is no
  screen i. `test: screens::display_i_opens_on_screen_i`,
  `test: screens::a_missing_screen_is_the_main_one`,
  `test: screens::the_list_names_each_display_its_screen`
- Two windows on one screen open offset from each other, not one exactly
  over the other: 24 points right and down, made smaller where the visible
  area has no room for the whole window, and kept so while following the
  guest's mode. `test: native::macos::shell::windows_on_one_screen_cascade`,
  `test: native::macos::shell::a_cascade_stays_on_the_screen`,
  `test: native::macos::shell::a_cascaded_window_shrinks_rather_than_overlap`

## The picture

- One GPU pass from the guest's frame into the window's drawable, with the
  display's gamma and colour matrix; no CPU copy and no GPU copy. On the
  Vulkan platform the pass draws into the image of a swapchain over the
  window's layer, which is the drawable, as on Linux. `manual`
- Each guest frame is drawn as it arrives, up to the screen's refresh rate
  (120 Hz on ProMotion). `manual`
- Colours are sRGB, not stretched over a P3 screen's gamut. `manual`
- The window resizes while input is not captured (Capture), its content by
  16 pixels wide and 2 high at a time (AppKit's content resize increments,
  in points at the scale of the screen it opened on): the steps of the
  guest's modes, so a window the person resizes shows the guest's whole
  mode, with no black beside it. Zoom, full screen and tiling take no
  increments, and a window moved to a screen of another scale can be 8
  pixels off a step. The image keeps the guest mode's aspect ratio, centred
  on whole pixels between black bars.
  `test: native::macos::shell::a_windowed_window_resizes_by_a_mode_s_steps`,
  `test: geometry::keeps_the_guest_ratio`,
  `test: geometry::centres_the_image`,
  `test: geometry::the_image_is_on_whole_pixels`
- Until the person resizes it (an edge, tiling, zoom, full screen), a
  windowed window follows the guest's mode: each change resizes it as it
  would open, top left corner kept, inside the screen's visible area.
  `test: geometry::a_resize_keeps_the_top_left_corner`,
  `test: geometry::a_resize_stays_in_the_visible_area`, `manual`
- After that, and always when borderless, a guest mode change re-fits the
  image at once; the window keeps its size.
  `test: geometry::a_mode_change_refits_the_image`, `manual`
- A window that keeps its size (the reims-graphics display's, whose guest
  takes the window's size in pixels as its mode) never follows the guest's
  mode: it opens sized for the first mode offered, and from then on a guest
  mode change re-fits the image, as after the person's resize. `manual`
- That guest is offered the window's pixels with the width rounded down to
  a multiple of 16 and the height to even, and a window of other pixels (a
  resize without steps, a zoom or tiling, or a borderless window at a
  fractional scale: 3841 × 2161
  pixels are offered 3840 × 2160, 3662 × 2039 are offered 3648 × 2038)
  shows that mode unscaled at its top left, beside up to 15 columns and a
  row of black. 16 BGRA8 pixels are a 64-byte line of the GPU's memory: on
  Intel's Vulkan driver a guest surface over the guest's RAM is drawn in
  place only where each row starts and ends on a line, and the guest's
  desktop surfaces are the mode's width (on homelab-ms01's Iris Xe the
  3662-pixel desktop stayed a copy, 2.54 ms of GPU time a plain frame and
  3.94 ms a frame that reads its own pixels; the owner's ruling of
  2026-11-07 removes the cause). The guest's kext offers a mode HiDPI only
  where its pixels halve exactly, and a guest given a 3841 × 2161 mode
  never brought its display up, HiDPI or not.
  `test: geometry::a_drawable_offers_a_mode_a_multiple_of_16_wide_drawn_unscaled`
- That guest is told the scale of the screen the window was last drawn on,
  and its display is as dense as that screen, 110 pixels per inch a point
  of it. An arm64 guest runs at 1x on a screen at 1x and HiDPI on one above
  1x, at every window size. An x86-64 guest starts HiDPI on a screen at 1.5x
  or more at every window size from 2048 × 1536 pixels, and at 1x below that
  and on a screen at 1.45x or less, up to 3840 × 2160 pixels, from which
  macOS starts any display HiDPI. Each window size is a display of its own
  to an x86-64 guest: a mode the person chooses is kept for that size,
  through restarts, and comes back when the window is that size again, and a
  resize to another size gives that size's mode. So on a screen above 1x and
  under 1.5x a chosen HiDPI mode does not follow a resize. A window moved to
  a screen of another scale gives the guest a new mode.
  `test: geometry::the_layout_carries_the_screen_s_scale`, `manual`
- Sharp on every screen: one drawable pixel per screen pixel, following the
  window across screens of different scale. `manual`
- Native full screen (the green button, ⌃⌘F) keeps the ratio. `manual`
- Nothing is drawn while the window is fully hidden; it redraws when it
  shows again. `test: native::macos::render::a_hidden_window_is_drawn_when_it_shows`,
  `manual` (occlusion)
- While an x86-64 guest's display sleeps (`pmset displaysleepnow`, Energy
  Saver's timer), the image is black and the guest's cursor does not show,
  as on a monitor whose panel is off; at the guest's wake, the image shows
  the guest's desktop again, with its cursor. `manual`

## Matching the host's screen

- The guest's mode runs at the refresh rate of the screen the window opens
  on, the one the screen is set to (its current mode's): a ProMotion screen
  set to 60 Hz gives the guest 60, not the 120 it offers; where the current
  mode gives no rate, the highest of the screen's modes of its size, else
  60 Hz. With no host screen (as for a VM started from an ssh shell, with no
  window system to reach), 60 Hz.
  `test: screens::a_screen_runs_at_its_current_mode_s_rate`,
  `manual` (a VM boot from an ssh shell)

## Borderless

- Each window covers its whole screen, with no title bar. `manual`
- The menu bar and Dock hide while the VM is frontmost, and return when
  another app is. `manual`
- On a screen with a camera housing, the image fits below it.
  `test: geometry::the_image_fits_below_a_camera_housing`
- The window takes keys and clicks as a titled one does. `manual`
- When screens change (rearranged, resized, unplugged), each window re-fills
  its screen, or the main screen when its own is gone. `manual`

## Pointer

- Absolute: the guest's pointer is where the host's is over the image, and
  it holds at the nearest edge outside the image.
  `test: geometry::the_pointer_maps_to_the_guest_pixel_under_it`,
  `test: geometry::the_pointer_holds_at_the_nearest_edge`
- Left, right, middle and both side buttons click and drag; a drag ends in
  the guest wherever its button comes up, held at the image's edge outside
  it. `test: geometry::buttons_map_to_the_five_bits`,
  `test: geometry::the_pointer_holds_at_the_nearest_edge`,
  `test: native::macos::input::a_click_in_the_image_is_sent_with_its_button`,
  `test: native::macos::input::a_drag_keeps_going_outside_the_window`
- The click that brings the window forward is not sent to the guest.
  `test: native::macos::input::the_click_that_makes_the_window_key_is_not_sent`, `manual`

## Keyboard

- Keys reach the guest by position, so the guest's own layout applies.
  `test: keymap::keys_map_by_position`,
  `test: keymap::fn_and_unknown_codes_have_no_usage`
- On ISO keyboards, the key left of 1 and the key right of left Shift arrive
  as on a real ISO keyboard, and a key goes up as it went down though
  another keyboard was typed on between. `test: keymap::iso_swaps_section_and_grave`,
  `test: native::macos::input::a_key_goes_up_as_it_went_down_across_a_keyboard_change`
- Left and right Shift, Control, Option and Command are told apart.
  `test: keys::left_and_right_modifiers_are_told_apart`,
  `test: keys::a_synthesized_modifier_takes_its_side_from_the_key`
- Caps Lock toggles as on a Mac keyboard. `manual`
- Held keys repeat at the guest's own rate; host repeats are dropped.
  `test: keys::host_repeats_are_dropped`,
  `test: native::macos::input::host_repeats_are_dropped_at_the_view`
- No stuck keys: losing focus, releasing capture or closing sends a key-up
  for every key still down. `test: keys::release_all_lifts_every_held_key`,
  `test: keys::a_missed_flags_changed_is_made_good`,
  `test: native::macos::input::losing_focus_lifts_every_key_and_releases`,
  `test: native::macos::input::releasing_capture_lifts_every_key`,
  `test: native::macos::input::closing_lifts_every_key`
- Key-ups under Command reach the guest.
  `test: native::macos::input::key_ups_under_command_reach_the_guest`, `manual`

## Capture

- A click into the image captures the keyboard and the pointer together: ⌘
  shortcuts and system keys (⌘Q, ⌘W, ⌘Tab, ⌘Space) go to the guest, and the
  pointer stays over the image. Neither is captured without the other.
  `test: capture::a_click_in_the_image_captures`,
  `test: capture::captured_keys_go_to_the_guest`,
  `test: native::macos::input::captured_command_keys_reach_the_guest_not_the_menu`
  (routing), `manual` (system keys)
- Captured, the host's pointer is held over the image (the window's
  `mouseConfinementRect`), so every click is the guest's: it cannot leave
  for the bars, the title bar or the host's other windows, menu bar or Dock,
  and the guest's own edges and corners (its menu bar, its Dock, its hot
  corners) are in reach. Nor can the window be resized while captured, so a
  click or a drag where the image meets the window's edges is the guest's
  too, never a resize of the window. Released, the pointer goes where the
  person moves it, and the window resizes again. In native full screen an
  image that reaches the screen's top or bottom edge takes the pointer there,
  where macOS slides its own menu bar or Dock in.
  `test: capture::captured_input_holds_the_pointer_over_the_image`,
  `test: capture::a_captured_window_cannot_be_resized`,
  `test: native::macos::shell::a_windowed_window_is_not_resizable_while_captured`,
  `manual` (the hold, and edge clicks)
- Pressing ⌃⌥⌘ together releases capture, of the keyboard and the pointer.
  `test: capture::control_option_command_releases`,
  `test: native::macos::input::the_release_chord_lifts_every_key_and_releases`
- Losing focus releases capture and gives the host back its system keys and
  its pointer.
  `test: capture::losing_focus_releases`,
  `test: native::macos::input::losing_focus_lifts_every_key_and_releases`,
  `test: native::macos::hotkeys::the_hot_keys_come_back_when_the_last_holder_lets_go`
- The title says whether input is captured: " — click to capture input"
  while not, and how to release it while it is.
  `test: capture::the_title_says_whether_input_is_captured`, `manual`
- Uncaptured, ⌘ shortcuts drive the host's menu, and other keys go to the
  guest. `test: capture::uncaptured_command_goes_to_the_host`,
  `test: native::macos::input::uncaptured_command_keys_go_to_the_menu`

## Scrolling

Where the host takes smooth scrolling (QEMU's reims-graphics-pci), and the
guest's pointing device takes it (vmapple's `usb-vmapple-tablet`):

- Trackpads and the Magic Mouse scroll the guest as far as the host reports,
  event by event, with the gesture's phases and its momentum, so the guest
  scrolls and coasts as the host does.
  `test: native::macos::input::smooth_scrolling_sends_the_gesture_to_its_end`,
  `test: scroll::smooth_scrolling_is_in_256ths_of_a_point`,
  `test: native::macos::input::appkit_phases_are_the_guests`, `manual`
- Horizontal scrolling reaches the guest.
  `test: scroll::smooth_scrolling_is_in_256ths_of_a_point`, `manual`
- A mouse wheel scrolls the guest the lines the host gives it, accelerated
  as the host accelerates them, at 10 points a line.
  `test: scroll::smooth_scrolling_is_in_256ths_of_a_point`
- A gesture that begins over the image scrolls the guest to its end,
  momentum included, wherever the pointer goes meanwhile; one that begins
  over the bars never does. It begins where the pointer is as it first
  scrolls, or where the fingers rested before that while the pointer is
  still there. An end the window never saw (fingers that lifted, a gesture
  that ended, out of its sight) carries nothing over to the next gesture.
  `test: scroll::a_gesture_begun_over_the_image_is_sent_to_its_end`,
  `test: scroll::a_gesture_begun_off_the_image_is_not_sent`,
  `test: scroll::an_end_unseen_leaves_nothing_to_the_next_gesture`,
  `test: scroll::an_event_of_no_phase_goes_where_the_pointer_is`
- The guest gets the physical direction here too.
  `test: scroll::smooth_scrolling_is_in_256ths_of_a_point`

Otherwise, as wheel notches (a host built without smooth scrolling, or
QEMU's `usb-tablet`, for which the host gets each event's notches too,
`test: native::macos::input::smooth_scrolling_sends_the_gesture_to_its_end`):

- A mouse wheel scrolls one guest notch per wheel notch.
  `test: scroll::a_notched_wheel_gives_a_notch_per_line`
- Trackpads and the Magic Mouse scroll one notch per 40 points of movement,
  momentum included, restarting with each new gesture.
  `test: scroll::precise_scrolling_gives_a_notch_per_40_points`,
  `test: scroll::a_new_gesture_restarts_the_count`,
  `test: scroll::a_direction_change_restarts_the_count`
- The guest gets the physical direction, so the guest's own natural
  scrolling setting decides it, not the host's.
  `test: scroll::the_guest_gets_the_physical_direction`

## Cursor

- Over the image, the host's cursor takes the guest's shape and hot spot, at
  the size the guest draws it, scaling with the window.
  `test: geometry::the_cursor_scales_with_the_image`,
  `test: native::macos::cursor::the_cursor_is_the_glyph_at_the_images_scale`
- It hides when the guest hides its cursor.
  `test: native::macos::cursor::a_hidden_guest_cursor_is_invisible`, `manual`
- Where the guest's cursor position changes with no pointer move of the
  window's in the second before (a mouse passed through to the guest with
  QEMU's `usb-host`, which the host's pointer never sees, or the guest placing
  its own), the window draws the guest's cursor into the image where the guest
  puts it, at the image's scale and clipped to it, and the host's cursor is
  invisible over the image. A show, a hide or a glyph change that leaves the
  cursor where it was moves nothing. The window's next pointer move gives the
  cursor back to the host's pointer.
  `test: handle::the_guest_leads_its_cursor_when_nothing_of_the_window_s_moved_it`,
  `test: handle::a_doorbell_reads_the_guest_s_cursor_through_the_sink`,
  `test: handle::a_cursor_the_guest_moves_on_its_own_is_the_window_s_to_draw`,
  `manual`
- Over the bars and the title bar, it is the host's arrow. `manual`

## Leaving

- Closing any window, or ⌘Q while uncaptured, powers the VM off, and QEMU
  exits cleanly. `manual`
- The VM keeps full speed with its windows hidden or minimised (no App Nap).
  `manual`
- `screendump` keeps working with windows open. `manual`

## On Linux

The window is SDL3's, on Wayland or X11, drawn through Vulkan on the
runtime's own device. The switches above apply, and `resolution=` too.

- Opening: QEMU run with `-display none` opens one window per guest display,
  as on macOS. SDL runs on a thread of the library's own, `reims-vgpu.ui`:
  QEMU's main thread is never taken or waited on, and the window callbacks
  run on that thread, one at a time. SDL installs no signal handlers, so
  Ctrl+C in QEMU's terminal ends QEMU as before. Without a Wayland
  compositor or an X server, no window opens, the guest runs headless, and
  the reason is logged once. `manual`
- The title, the first size (a guest point per host point, shrunk to fit
  the screen less 48 points for the title bar and a panel, on whole pixels)
  and the screen a display opens on are as above. The size is asked for in
  SDL's sizes: points on Wayland, and on X11 pixels, the desktop's content
  scale of them to the point, so an X11 desktop at 200% does not open the
  window at half its size. The size is made smaller to the largest that
  comes to a multiple of 16 pixels wide and an even number high, the steps
  of the guest's modes, as SDL rounds it: on Wayland the points times the
  scale in 120ths, rounded half away from zero (at a fractional scale such
  as 2.1, about one size in 16 comes to a multiple of 16 pixels); on X11
  the pixels themselves. SDL takes no resize increments, so a window the
  person resizes, or the compositor sizes otherwise, shows the guest's mode
  beside up to 15 columns and a row of black.
  `test: sdl::units::a_window_opens_a_multiple_of_16_pixels_wide_at_fractional_scales`,
  `test: handle::a_window_s_title_is_reims_vgpu_and_its_port`,
  `test: geometry::a_new_window_shows_a_guest_point_per_host_point`,
  `test: sdl::units::a_window_opens_a_guest_point_per_host_point_in_sdl_s_sizes`,
  `test: screens::display_i_opens_on_screen_i`, `manual` (X11 at a content
  scale above 1 among them)
- Wayland lets no client place a windowed window: the compositor does.
  Borderless windows fill the screen named, over its panels. At a
  fractional scale a borderless window is the compositor's points for the
  screen rounded to whole points, so its pixels can pass the screen's by one
  (3841 × 2161 on a 3840 × 2160 screen at 2.1); the reims-graphics guest is
  offered the screen's own 3840 × 2160 (above). `manual`
- The host's screens are SDL's: the primary first, then top to bottom and
  left to right, each at its scale (fractional ones included) and the
  refresh rate it is set to (its current mode's, as above); a change of
  screens lays every window out again. `manual`
- The picture: one GPU pass from the guest's frame into the swapchain's
  image, no copy, with the display's gamma and colour matrix, in sRGB.
  `manual`
- It never tears and never makes the guest wait: MAILBOX with at least four
  images where the compositor offers it, else FIFO; the render thread alone
  waits for an image, and a window the compositor stops showing slows only
  its own drawing. `manual`
- Sharp at any scale, fractional ones included: one drawable pixel per
  screen pixel, the swapchain made again whenever the window's pixels
  change, across screens and scale changes. `manual`
- The window resizes freely, keeping the guest's ratio between black bars,
  and follows the guest's mode until the person resizes it (a compositor's
  rounding by a point is not the person's).
  `test: geometry::keeps_the_guest_ratio`,
  `test: sdl::shell::a_resize_within_a_point_of_the_asked_size_is_the_library_s`,
  `manual`
- A window that keeps its size (the reims-graphics display's) never follows
  the guest's mode, as on macOS. `manual`
- Nothing is drawn while the window is hidden or minimized, nor before it
  and its layout exist; it redraws when it shows.
  `test: native::linux::render::a_hidden_window_is_drawn_when_it_shows`,
  `test: native::linux::render::nothing_is_drawn_before_the_window_and_layout`,
  `test: native::linux::render::what_asks_for_a_draw`
- Nothing is presented while the compositor suspends the window (Wayland's
  suspended state, as KWin and Mutter set it for a covered window or one on
  another desktop): from SDL's occluded event until it is exposed again, the
  last of a batch of SDL's events deciding (a size change of a suspended
  window sends an exposure and an occlusion together), as
  on macOS; a draw waiting for an image is not made again meanwhile. A
  suspended window's FIFO present waits for a frame callback that does not
  come, on the window's presenting queue, which the guest's frames go through
  too where the device has no queue to spare. The guest's frames and present
  callbacks go on as before: the window draws the display's current frame
  when it shows, and holds none back.
  `test: sdl::shell::a_covered_window_stops_drawing_until_exposed`,
  `test: sdl::shell::the_last_show_or_hide_of_a_batch_decides`,
  `test: native::linux::render::a_hidden_window_takes_no_retry`, `manual`
- Pointer: absolute, held at the image's nearest edge; five buttons; a
  drag ends in the guest; the click that focuses the window is not sent;
  nothing is sent before the guest has a mode.
  `test: native::linux::input::sdl_buttons_are_the_hosts_bits`,
  `test: native::linux::input::a_drag_ends_in_the_guest_outside_the_window`,
  `test: native::linux::input::the_click_that_focuses_the_window_is_not_sent`,
  `test: native::linux::input::before_the_guest_has_a_mode_the_pointer_sends_nothing`,
  `test: native::linux::input::moves_are_sent_where_the_guest_pixel_changes`
- Keys by position, as evdev reports them, so the guest's layout applies and
  ISO keys arrive as on a real ISO keyboard; media keys are dropped.
  `test: native::linux::keymap::scancodes_are_usb_usages`,
  `test: native::linux::keymap::keys_off_the_keyboard_page_are_dropped`
- Held keys repeat at the guest's rate; host repeats are dropped. No stuck
  keys: losing the keyboard, releasing capture or closing lifts every key.
  `test: native::linux::input::host_repeats_are_dropped`,
  `test: native::linux::input::losing_the_keyboard_leaves_nothing_stuck`,
  `test: native::linux::input::closing_leaves_nothing_stuck`,
  `test: native::linux::sdl::a_close_the_person_asks_for_lifts_every_key_and_ends_capture`,
  `test: sdl::shell::a_close_sends_the_key_ups_and_ends_capture_before_the_host_hears_of_it`
- Capture: a click in the image captures the keyboard and the pointer, and
  the window then grabs the keyboard, so the compositor's own shortcuts
  (Super, Alt+Tab) reach the guest, and holds the pointer over the image, as
  on macOS: SDL's mouse rect, in whole points rounded out, which the
  compositor's pointer constraints keep on Wayland and XFixes barriers on
  X11 (where SDL has no mouse rect, its mouse grab holds the pointer in the
  whole window); the window stays resizable, its resize edges being the
  compositor's, outside its surface and beyond the hold; Ctrl+Alt+Super
  releases both; losing the keyboard releases; the window's end releases
  both, and the grab and the hold go with the capture before it hides; the
  title says whether input is captured, as on macOS, with Ctrl+Alt+Super to
  release.
  `test: native::linux::input::a_click_in_the_image_captures_and_is_sent`,
  `test: native::linux::input::ctrl_alt_super_releases_capture`,
  `test: sdl::shell::the_end_lets_go_of_the_grab_and_the_holds`,
  `test: sdl::shell::the_pointer_is_held_over_the_image_in_whole_points`,
  `manual` (the grab and the hold)
- Scrolling: the wheel's notches, or, where the host takes smooth
  scrolling, its lines at 10 points each, in no gesture (SDL gives no
  phases); the guest gets the device's own direction.
  `test: native::linux::input::the_wheel_sends_the_devices_own_direction`
- Cursor: over the image the guest's glyph at the image's scale, with an
  image at the screen's own scale for HiDPI; hidden when the guest hides
  its own; the host's arrow over the bars and the title bar. Where the guest's
  cursor position changes with no pointer move of the window's in the second
  before, the window draws the cursor into the image where the guest puts it,
  and hides the host's over the image, as on macOS.
  `test: sdl::cursor::a_glyph_is_resampled_by_nearest_pixel`,
  `test: sdl::cursor::a_short_or_empty_glyph`,
  `test: handle::a_cursor_the_guest_moves_on_its_own_is_the_window_s_to_draw`,
  `manual`
- With several displays' windows open, the cursor is set only by the window
  the pointer is in: SDL's cursor is the process's, so another window's guest
  changing its cursor, or that window's layout changing, leaves the cursor
  over the image the pointer is in as it is.
  `test: sdl::shell::a_window_the_pointer_is_not_in_leaves_the_cursor`,
  `manual` (a VM with two displays)
- Leaving: closing a window asks the host to end the VM; the window closes
  at the display's release, once its swapchain is gone. `manual`

Not on Linux: a menu bar, Dock icon or ⌘Tab entry (the compositor's task
bar shows the window, as `reims-vgpu`), the camera-housing fit, and trackpad
gesture phases and momentum.

## On macOS through SDL3

With `REIMS_VGPU_WINDOW_IMPL=sdl`, in a build with reims-vgpu-window's
`macos-sdl` feature (on by default), the window is SDL3's over Cocoa, and
every entry up to "On Linux" is meant to hold as for AppKit's window, which
stays the default until this one's manual walk on a vmapple boot passes. It
runs in the same application (its Dock icon, ⌘Tab entry, menu and Quit, App
Nap held off), is drawn by the same render thread into the layer of SDL's
Metal view, opens and follows the guest's mode in the frames AppKit's window
takes, is borderless as AppKit's window is (the camera housing's room
included), and gives AppKit's events to AppKit's window's input, so keys,
capture, the system's hot keys and scrolling's phases and momentum are the
same. The cursor is SDL's, with an image at the screen's scale, the
pointer's hold while captured is SDL's mouse rect, which SDL keeps with the
same `mouseConfinementRect`, and the window is held from resizing while
captured through SDL's resizable flag. `manual`

## Not provided

- Horizontal scrolling with QEMU's `usb-tablet`, which has no horizontal
  wheel.
- Fn/Globe: QEMU's USB keyboard cannot carry it.
- HDR output.
