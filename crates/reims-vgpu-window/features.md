# reims-vgpu-window features

What someone at the host sees and does with a guest display's window. Every
entry was put in on purpose. Removing or changing one is a regression unless
it is done deliberately, in its own commit, with this file updated and the
reason given.

Each entry ends with its guard: `test: <module>::<name>`, a unit test in this
crate; `test: <crate>::<module>::<name>`, a unit test in the workspace's crate
in `crates/<crate>`, such as reims-vgpu's display tests, which drive the
window from the guest's device; or `manual`, a check on a VM boot.
`features::features_md_guards_exist` fails when a named test is gone. A change
to this crate walks the `manual` entries it can reach. A bare `test` is a
guard not yet written.

The sections up to "On Linux" are the macOS window's, AppKit's. "On Linux"
is the window on Linux, SDL3's on Wayland or X11, entry by entry against
them; its `manual` checks are on a Linux host's guest boot.

## Switches

| variable | effect |
|---|---|
| `REIMS_VGPU_WINDOW=0` | QEMU stays headless under `-display none` (QEMU reads it) |
| `REIMS_VGPU_WINDOW_BORDERLESS=1` | borderless windows that fill their screens |
| `REIMS_VGPU_WINDOW_SCREENS=1,0` | display i opens on the host screen the list's i-th entry names |
| `resolution=2560x1600` | the mode the window opens for, and the guest is offered until the window's pixels are known (QEMU's reims-graphics-pci property; 1920x1080 by default); with several displays, every window's |
| `displays=2` | the reims-graphics device's displays, 1 to 4, a window each (QEMU's reims-graphics-pci property; 1 by default) |
| `REIMS_VGPU_WINDOW_IMPL=sdl` | macOS, built with the `macos-sdl` feature (on by default): SDL3's window in place of AppKit's ("On macOS through SDL3" below) |
| `REIMS_VGPU_WINDOW_CAPTURE=0` | capture off: no window ever captures the keyboard or the pointer ("Capture off" below); reims-ctl sets it for a VM whose capture setting is off |

Unset or unparsable values give the defaults below.
`test: env::unset_or_unparsable_values_are_the_defaults`,
`test: env::the_window_is_appkits_unless_sdl_is_named`,
`test: env::borderless_takes_the_usual_words`,
`test: env::the_screen_list_is_all_or_nothing`,
`test: env::capture_is_on_unless_turned_off`

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
- With several reims-graphics displays, each window is its own display's:
  its guest display is offered the window's mode, at the refresh rate and
  scale of the screen the window is on, whatever the other windows are, and
  a resize changes only its own display's mode. Each display is offered at
  most the largest mode of its window's ratio that its share of the device's
  128 MiB of scanout holds (64 MiB each with two displays, so an 8K window
  is offered 5440 × 3060, which it shows scaled to fit; 32 MiB each with
  four, 3840 × 2160 at 16:9), as reims-vgpu-graphics' tests
  `display::tests::a_display_offers_the_largest_mode_of_its_ratio_its_room_holds`
  and `device::tests::displays::a_guest_that_took_multi_display_is_told_every_display_s_own_mode`
  hold. `manual` (a VM with two displays)
- A window whose display the guest does not drive shows black, as a monitor
  with no signal does, and so does QEMU's screendump of its head: a guest
  whose driver does not take the device's `MULTI_DISPLAY` drives display 0
  alone, and the option ROM drives only display 0, so during boot the other
  windows stay black. reims-vgpu's test
  `graphics::tests::displays::each_display_reads_back_and_one_past_the_count_is_refused`
  holds the black read-back. `manual` (a VM with two displays and a kext
  that drives one)

## The picture

- One GPU pass from the guest's frame into the window's drawable, with the
  display's gamma and colour matrix, and no CPU copy. With the colour at
  identity the pass is a GPU copy into the drawable, of the guest's scanout
  or of a presented surface in the drawable's format (BGRA8, with or without
  sRGB); otherwise it is a draw of the display's present or colour shaders.
  The window makes no other copy of the frame: the pass writes the image of a
  swapchain over the window's layer, which is the drawable, as on Linux.
  `manual`
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
- With several reims-graphics displays, the pointer over a window reaches
  that window's display: moving the host's pointer from one window into
  another moves the guest's cursor onto that window's display, where the
  pointer is, and the pointer then drives the cursor over that display.
  macOS scales the one USB tablet over the display its cursor is on, so the
  window has the guest move its cursor first (`POINTER_OVER`,
  docs/graphics-kext/design.md, "Multiple displays"), and holds the
  pointer's positions over the new display until the guest's cursor is
  there, for at most a second, so none lands on the display the cursor left; a
  button going down or up is sent at once. The cursor is on the display that
  shows its hot spot: the part of its image another display shows over a
  shared edge (an arm64 guest's WindowServer gives each display its part)
  does not count. Captured, the pointer is held
  over its window's image (Capture), so on macOS a drag stays on the display
  it began on, and the pointer reaches another display's window once capture
  is released (on Linux it goes on past the edge: "On Linux").
  `test: reims-vgpu::graphics::tests::displays::a_window_s_pointer_tells_the_guest_its_display_and_waits_for_its_cursor`,
  `test: reims-vgpu-graphics::device::tests::displays::the_pointer_over_another_display_tells_the_guest_once`,
  `test: reims-vgpu-graphics::device::tests::displays::a_crossing_holds_the_pointer_until_the_guest_s_cursor_is_there`,
  `test: reims-vgpu-graphics::device::tests::displays::the_image_over_the_edge_leaves_the_cursor_on_the_display_of_its_hot_spot`,
  `manual` (a VM with two displays)
- In a game: captured, once the guest has hidden its cursor on every display
  for 50 ms (a hide while the guest has yet to show the cursor on the display
  a crossing moves it to counts from a second after the crossing began), as
  a game does that turns its camera by the mouse's motion, the
  pointer drives the guest's own mouse (`POINTER_MOTION`,
  docs/graphics-kext/design.md, "A pointer for games"), so the camera turns
  however far the mouse moves, past the image's edge. The window holds the
  host's pointer still (`CGAssociateMouseAndMouseCursorPosition`) and hands
  the guest's mouse its unaccelerated motion, the buttons pressed meanwhile
  and a mouse wheel's notches; the guest accelerates the motion once, as its
  own mouse settings say. Once the guest shows its cursor again (the game's
  menu, or the next move after typing hid it), the pointer is the absolute
  one again, the host's pointer let go over the guest's cursor where it shows
  on the window's display, else where it was held. A button comes up by the
  route it went down by, so one held across either switch is never left
  down. Releasing capture, losing the keyboard and closing let the pointer go
  where it was held. On the desktop, and with a guest whose kext does not
  take `POINTER_MOTION` (one built before it, the option ROM, Ambrosia's
  module), the pointer is the absolute one only.
  `test: game::the_guest_s_mouse_is_driven_once_its_cursor_stays_hidden`,
  `test: game::moves_go_to_the_guest_s_mouse_until_it_shows_its_cursor`,
  `test: game::a_button_comes_up_by_the_route_it_went_down_by`,
  `test: game::the_wheel_goes_to_the_driven_mouse`,
  `test: game::capture_s_end_and_a_refused_lock_give_the_pointer_back_where_held`,
  `test: native::macos::input::the_tablet_s_events_are_unchanged_outside_a_game`,
  `test: native::macos::input::a_hidden_cursor_s_motion_drives_the_guest_s_mouse_until_it_shows`,
  `test: native::macos::input::a_button_held_across_the_switch_comes_up_by_its_own_route`,
  `test: native::macos::input::letting_go_of_a_game_gives_the_pointer_back_where_it_was_held`,
  `test: reims-vgpu::graphics::tests::displays::a_window_s_device_offers_pointer_motion_and_drives_the_guest_s_mouse`,
  `test: reims-vgpu-graphics::device::tests::motion::a_crossing_s_hide_is_not_a_game_s`,
  `test: reims-vgpu-graphics::device::tests::motion::a_guest_that_did_not_take_pointer_motion_gets_none`,
  `test: reims-vgpu-graphics::device::tests::motion::the_cursor_is_hidden_once_every_display_it_sees_hides_it`,
  `test: reims-vgpu-graphics::device::tests::motion::owed_motion_joins_and_a_press_never_does`,
  `manual` (a game, or a probe that hides the cursor and reads its deltas:
  the camera turns past the image's edge; a drag on the desktop lands where
  the pointer is; typing in TextEdit hides the cursor, and the next move
  shows it where the guest moved it, the host's pointer over it)

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
  pointer stays over the image. Neither is captured without the other. With
  capture off, nothing is ("Capture off", below).
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
  where macOS slides its own menu bar or Dock in. While the pointer drives
  the guest's mouse in a game (Pointer), the host's pointer is held still
  instead, and capture's end lets it go where it was held.
  `test: native::macos::input::letting_go_of_a_game_gives_the_pointer_back_where_it_was_held`,
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

## Capture off

- With capture off (`REIMS_VGPU_WINDOW_CAPTURE=0`, which reims-ctl sets for a
  VM whose capture setting is off), no window ever captures, on any host: a
  click into the image goes to the guest but captures nothing, so the window
  never takes the host's system keys, never holds or locks the host's
  pointer, and never holds its own size, and there is nothing to release.
  The pointer reaches the guest over the image alone, as the absolute
  tablet; outside the image only a drag that began in it goes on, held at
  the image's edge, and ends in the guest. Keys go to the guest while the
  window has the keyboard: on macOS ⌘ shortcuts too, taken before the
  window's menu as while captured (⌘Q among them; the menu's Quit and the
  close button still end the VM, once the person chooses Turn Off at the
  question Leaving describes), with the system's own (⌘Tab, ⌘Space) left
  to macOS; on Linux and Windows what the compositor or Windows keeps for
  itself (Super, Alt+Tab, the Windows key) stays its own. A game gets the
  absolute pointer alone: the guest's mouse is never driven, so a camera
  that turns by the mouse's motion stops where the pointer reaches the
  image's edge. The title ends " — input capture off" on every host.
  `test: capture::capture_off_never_captures`,
  `test: capture::capture_off_sends_the_pointer_outside_the_image_only_in_a_drag`,
  `test: env::capture_is_on_unless_turned_off`,
  `test: native::sdl_thread::input::capture_off_captures_nothing_and_sends_the_pointer_over_the_image_alone`,
  `test: native::macos::input::capture_off_sends_keys_and_the_pointer_over_the_image_and_holds_nothing`,
  `manual` (a VM with capture off: the title says so; clicks, drags and
  keys reach the guest; the pointer leaves the window freely, and Alt+Tab
  or ⌘Tab switches away, with no release chord)

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
  `test: handle::a_cursor_the_guest_moves_on_its_own_is_the_window_s_to_draw`,
  `test: reims-vgpu::graphics::tests::the_window_draws_the_cursor_the_guest_moves_on_its_own`,
  `test: reims-vgpu::graphics::sink::the_guest_s_cursor_is_drawn_over_the_frame_in_a_pass_of_its_own`,
  `test: reims-vgpu::graphics::sink::on_vulkan_the_cursor_s_pass_blends_the_glyph_over_the_frame`,
  `manual`
- Over the bars and the title bar, it is the host's arrow. `manual`

## Leaving

- Closing any window (its close button), or Quit (⌘Q while uncaptured with
  capture on, the application menu's Quit or the Dock's; with capture off
  ⌘Q is the guest's, and the menu's Quit or the close button asks), first
  lets every key and button go and ends any capture, the key-ups reaching
  the guest, then asks in an alert, a sheet on the window, which comes
  forward: "Turn off the VM?", "The VM turns off at once, as if its power
  were cut. Unsaved work in it is lost.", with Turn Off, marked
  destructive, and Cancel, the default. Turn Off powers the VM off, and
  QEMU exits cleanly. Cancel leaves the window and the VM running, nothing
  held in the guest and the input released. Quit asks once for every
  window. A close while the question is up adds none; once Turn Off is
  chosen, a later close asks nothing. No question where the close would end
  nothing (the host gave no close callback) or where the VM is already
  ending (the guest shut down, QEMU quitting, reims-ctl stop): the display's
  release closes the window, and takes an open question's sheet off first.
  `test: close::a_close_asks_only_where_the_host_would_hear_of_it`,
  `test: close::only_turn_off_tells_the_host`,
  `test: native::macos::input::a_cancelled_close_leaves_nothing_held_and_the_input_goes_on`,
  `manual` (the sheet, its words and buttons; Cancel, then Turn Off)
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
  image, with the display's gamma and colour matrix, in sRGB: a GPU copy
  where the colour is at identity and the frame is the guest's scanout or a
  surface in the image's format, else a draw, as above; the window makes no
  other copy of it where the compositor composites on the guest's GPU.
  Where it composites on another GPU (a GPU in an eGPU enclosure or an x4
  slot beside the screen's), the swapchain is still the guest's GPU's, and
  each image presented crosses to the compositor's GPU through the window
  system, once a present (an RTX 3090 on PCIe 3.0 x4 spends 16.4 ms of its
  link on a 3648 × 2039 present; reims-vgpu-vulkan's `swapchain`). `manual`
- It never tears and never makes the guest wait: MAILBOX with at least four
  images where the compositor offers it, else FIFO; the render thread alone
  waits for an image, and a window the compositor stops showing slows only
  its own drawing. `manual`
- Sharp at any scale, fractional ones included: one drawable pixel per
  screen pixel, the swapchain made again whenever the window's pixels
  change, across screens and scale changes. `manual`
- The reims-graphics guest is told the scale of the screen the window is
  on, as on macOS (The picture): on Wayland the window's pixels a point; on
  X11 and on a Windows host, where SDL's sizes are pixels and a window has
  one pixel to each of them at any scale, the screen's content scale, SDL's
  display scale of the window (on X11 the desktop's `Xft.dpi` over 96, else
  GDK's scale; on Windows the monitor's scale setting, 150% telling the
  guest 180 in 120ths). The drawable, the image and the pointer stay in the
  window's pixels. So on a Windows screen at 150% an x86-64 guest starts
  HiDPI at every window size from 2048 × 1536 pixels, as on macOS (a
  2560 × 1600 screen's borderless window at 1280 × 800 points). A change of
  the screen's scale, or a move to a screen of another, tells the guest the
  new one.
  `test: sdl::units::a_window_s_scale_is_its_density_where_sizes_are_points_else_its_screen_s`,
  `test: geometry::the_layout_carries_the_screen_s_scale`, `manual` (a
  Windows host at 150%, X11 at a content scale above 1)
- The window resizes freely, keeping the guest's ratio between black bars,
  and follows the guest's mode until the person resizes it (a compositor's
  rounding by a point is not the person's).
  `test: geometry::keeps_the_guest_ratio`,
  `test: sdl::shell::a_resize_within_a_point_of_the_asked_size_is_the_library_s`,
  `manual`
- A window that keeps its size (the reims-graphics display's) never follows
  the guest's mode, as on macOS. `manual`
- With several reims-graphics displays, each window is its own display's,
  offered its own mode, rate and scale and capped by its share of the
  scanout, and a window whose display the guest does not drive shows black,
  as on macOS (Opening). `manual` (a VM with two displays)
- Nothing is drawn while the window is hidden or minimized, nor before it
  and its layout exist; it redraws when it shows.
  `test: native::sdl_thread::render::a_hidden_window_is_drawn_when_it_shows`,
  `test: native::sdl_thread::render::nothing_is_drawn_before_the_window_and_layout`,
  `test: native::sdl_thread::render::what_asks_for_a_draw`
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
  `test: native::sdl_thread::render::a_hidden_window_takes_no_retry`, `manual`
- Pointer: absolute, held at the image's nearest edge; five buttons; a
  drag ends in the guest; the click that focuses the window is not sent;
  nothing is sent before the guest has a mode.
  `test: native::sdl_thread::input::sdl_buttons_are_the_hosts_bits`,
  `test: native::sdl_thread::input::a_drag_ends_in_the_guest_outside_the_window`,
  `test: native::sdl_thread::input::the_click_that_focuses_the_window_is_not_sent`,
  `test: native::sdl_thread::input::before_the_guest_has_a_mode_the_pointer_sends_nothing`,
  `test: native::sdl_thread::input::moves_are_sent_where_the_guest_pixel_changes`
- With several reims-graphics displays, the pointer over a window reaches
  that window's display, as on macOS.
  `test: reims-vgpu::graphics::tests::displays::a_window_s_pointer_tells_the_guest_its_display_and_waits_for_its_cursor`,
  `test: reims-vgpu-graphics::device::tests::displays::a_crossing_holds_the_pointer_until_the_guest_s_cursor_is_there`,
  `test: reims-vgpu-graphics::device::tests::displays::the_image_over_the_edge_leaves_the_cursor_on_the_display_of_its_hot_spot`,
  `manual` (a VM with two displays)
- With several reims-graphics displays, a captured pointer moving toward an
  edge of the image where the guest placed another display (System
  Settings, Displays, Arrange), within 32 points of it, goes on onto that
  display, as a Mac's cursor does, so a drag carries a guest window across
  with its button held. It leaves before the edge, as KWin holds a confined
  pointer short of it, and the guest's cursor goes on from where it is. The
  window locks the host's pointer where it was (SDL's relative mode,
  the compositor's pointer lock on Wayland), and its motion, at the
  desktop's own acceleration, moves the guest's cursor across the guest's
  desktop, onto whichever display it reaches, through that display's window,
  which draws the guest's cursor. The pointer is the window's again where it
  comes back over the window's own display, out of reach of that edge, the
  host's pointer let go there, over the guest's cursor. Releasing capture
  (Ctrl+Alt+Super), losing the keyboard or closing lets the host's pointer
  go where it was held. Where the compositor
  refuses the lock, and with a guest that does not place its displays
  (`DISPLAY_BOUNDS`: a kext built before it, or an arm64 guest before its
  cursor first went to the display past the edge), the pointer stays held
  at the edge, as on macOS.
  `test: reims-vgpu::graphics::tests::displays::a_captured_pointer_past_the_shared_edge_goes_on_through_the_other_display_s_window`,
  `test: reims-vgpu-graphics::device::tests::displays::a_pointer_past_a_shared_edge_goes_onto_the_display_there`,
  `test: reims-vgpu-graphics::device::tests::displays::the_guest_s_arrangement_decides_which_edge_meets_which_display`,
  `test: reims-vgpu-graphics::device::tests::displays::only_a_guest_that_placed_its_displays_carries_the_pointer_past_an_edge`,
  `test: beyond::the_pointer_near_a_shared_edge_goes_on_onto_the_other_display`,
  `test: beyond::letting_go_while_away_gives_the_pointer_back_where_it_was_held`,
  `test: beyond::the_edge_is_within_reach_of_its_last_pixel`,
  `test: native::sdl_thread::input::the_captured_pointer_at_a_shared_edge_goes_on_past_it`,
  `test: native::sdl_thread::input::letting_go_while_away_gives_the_pointer_back_where_it_was_held`,
  `test: handle::a_pointer_from_beyond_reaches_the_host_and_the_window_draws_the_cursor`,
  `manual` (a VM with two displays: a Finder window dragged across and back,
  display 1 right of display 0 and left of it)
- In a game, as on macOS (Pointer): the window locks the host's pointer
  (SDL's relative mode, the compositor's pointer lock on Wayland) and hands
  the guest's mouse its unaccelerated motion (`SDL_MOUSE_RELATIVE_SYSTEM_SCALE`
  0 while it does: relative-pointer's unaccelerated motion on Wayland, XInput2's
  raw motion on X11), its buttons pressed meanwhile and the wheel's notches,
  until the guest shows its cursor, where the host's pointer is let go over
  it; a button comes up by the route it went down by; releasing capture
  (Ctrl+Alt+Super), losing the keyboard and closing let it go where it was
  held. Gone past a shared edge when the guest hides its cursor, the pointer
  drives the guest's mouse from there. Where the compositor refuses the lock,
  the pointer stays the absolute one until capture changes.
  `test: native::sdl_thread::input::the_tablet_s_events_are_unchanged_outside_a_game`,
  `test: native::sdl_thread::input::a_hidden_cursor_s_motion_drives_the_guest_s_mouse_until_it_shows`,
  `test: native::sdl_thread::input::a_button_held_across_the_switch_comes_up_by_its_own_route`,
  `test: native::sdl_thread::input::letting_go_of_a_game_gives_the_pointer_back_where_it_was_held`,
  `test: reims-vgpu-graphics::device::tests::motion::motion_reaches_a_guest_that_took_it`,
  `test: reims-vgpu-graphics::device::tests::motion::a_crossing_s_hide_is_not_a_game_s`,
  `manual` (as on macOS, on a Linux host's guest boot)
- Keys by position, as evdev reports them, so the guest's layout applies and
  ISO keys arrive as on a real ISO keyboard; media keys are dropped.
  `test: native::sdl_thread::keymap::scancodes_are_usb_usages`,
  `test: native::sdl_thread::keymap::keys_off_the_keyboard_page_are_dropped`
- Held keys repeat at the guest's rate; host repeats are dropped. No stuck
  keys: losing the keyboard, releasing capture or closing lifts every key.
  `test: native::sdl_thread::input::host_repeats_are_dropped`,
  `test: native::sdl_thread::input::losing_the_keyboard_leaves_nothing_stuck`,
  `test: native::sdl_thread::input::closing_leaves_nothing_stuck`,
  `test: native::sdl_thread::sdl::a_close_the_person_asks_for_lifts_every_key_and_ends_capture`,
  `test: sdl::shell::a_close_sends_the_key_ups_and_ends_capture_before_it_asks`
- Capture: a click in the image captures the keyboard and the pointer, and
  the window then grabs the keyboard, so the compositor's own shortcuts
  (Super, Alt+Tab) reach the guest, and holds the pointer over the image, as
  on macOS: SDL's mouse rect, in whole points rounded out, which the
  compositor's pointer constraints keep on Wayland and XFixes barriers on
  X11 (where SDL has no mouse rect, its mouse grab holds the pointer in the
  whole window), except past an edge where the guest placed another display
  and while it drives the guest's mouse in a game (Pointer, above); the
  window stays resizable, its resize edges being the
  compositor's, outside its surface and beyond the hold; Ctrl+Alt+Super
  releases both; losing the keyboard releases; the window's end releases
  both, and the grab and the hold go with the capture before it hides; the
  title says whether input is captured, as on macOS, with Ctrl+Alt+Super to
  release. On a Windows host, where the same SDL window runs, the Windows
  key is that Super key (SDL's GUI key), and the title names the chord
  Ctrl+Alt+Win. With capture off, nothing is captured, grabbed or held
  ("Capture off", above).
  `test: native::sdl_thread::input::a_click_in_the_image_captures_and_is_sent`,
  `test: native::sdl_thread::input::ctrl_alt_super_releases_capture`,
  `test: capture::the_title_names_the_release_chord_in_the_host_s_keys`,
  `test: sdl::shell::the_end_lets_go_of_the_grab_and_the_holds`,
  `test: sdl::shell::the_pointer_is_held_over_the_image_in_whole_points`,
  `manual` (the grab and the hold)
- Scrolling: the wheel's notches, or, where the host takes smooth
  scrolling, its lines at 10 points each, in no gesture (SDL gives no
  phases); the guest gets the device's own direction.
  `test: native::sdl_thread::input::the_wheel_sends_the_devices_own_direction`
- Cursor: over the image the guest's glyph at the image's scale, the size
  the guest draws it: on Wayland with an image at the screen's own scale
  for HiDPI, and on X11 and a Windows host at the window's pixels, which
  SDL draws as they are (on a Windows screen at 150%, a guest shown a
  pixel to a screen pixel has its cursor at its own pixels, not half as
  large again); hidden when the guest hides its own; the host's arrow over
  the bars and the title bar. Where the guest's
  cursor position changes with no pointer move of the window's in the second
  before, the window draws the cursor into the image where the guest puts it,
  and hides the host's over the image, as on macOS.
  `test: sdl::cursor::a_cursor_is_handed_sdl_at_the_guest_s_size_in_sdl_s_units`,
  `test: native::sdl_thread::ui::sdl_draws_a_cursor_at_the_size_it_is_given`,
  `test: sdl::cursor::a_glyph_is_resampled_by_nearest_pixel`,
  `test: sdl::cursor::a_short_or_empty_glyph`,
  `test: handle::a_cursor_the_guest_moves_on_its_own_is_the_window_s_to_draw`,
  `test: reims-vgpu::graphics::tests::the_window_draws_the_cursor_the_guest_moves_on_its_own`,
  `test: reims-vgpu::graphics::sink::on_vulkan_the_cursor_s_pass_blends_the_glyph_over_the_frame`,
  `manual` (a Windows host at 150%)
- With several displays' windows open, the cursor is set only by the window
  the pointer is in: SDL's cursor is the process's, so another window's guest
  changing its cursor, or that window's layout changing, leaves the cursor
  over the image the pointer is in as it is.
  `test: sdl::shell::a_window_the_pointer_is_not_in_leaves_the_cursor`,
  `manual` (a VM with two displays)
- Leaving: closing a window (its close button, Alt+F4 or the window
  system's close, the task bar's on Windows) lets every key and button go
  and ends capture, the key-ups reaching the guest first, then asks, with
  macOS's words and buttons (Leaving, above) in ASCII, Cancel the default
  for Return and Escape. On Linux the library asks in the window itself,
  with nothing installed on the host and the same on Wayland and X11 and
  every desktop: a dark panel centred over the frame, Cancel left of Turn
  Off, in an 8 by 8 pixel font each of whose pixels is a whole square of
  the screen's, a line about 12 points high at every scale, fractional
  ones and 4K included; Tab and the arrows move the focus, Space answers
  the focused button, a click answers the button it went down and up on,
  the host's arrow shows over the window, and a minimized window is
  restored and raised. Where the window draws no more (its GPU lost, or no
  swapchain), the question shows in SDL's message box instead: zenity's
  where zenity is installed, else SDL's own X11 dialog. On Windows a task
  dialog the window owns, Turn Off left of Cancel. Turn Off asks the host
  to end the VM; Cancel, or the question dismissed, leaves the window and
  the VM running, nothing held in the guest and the input released. From
  the close to the answer nothing of the window's input reaches the guest:
  not a key that goes down after the close let the keys go (F4 of Alt+F4
  on Windows, which SDL hands over after the close), nor the keys and
  clicks the question takes, nor their ups. Where no question can show,
  the close keeps the VM running and logs why; the VM is then turned off
  by the guest's own shutdown or the host's stop (reims-ctl stop). The one
  case known to reach it is a Linux host whose window draws no more (its
  GPU lost, or no swapchain) in a Wayland session without zenity, or with
  no X server. While a dialog is up, the library's UI thread waits for it:
  the windows' frames keep drawing, and their events wait. When to ask is
  as on macOS. The window closes at the display's release, once its
  swapchain is gone.
  `test: close::a_close_asks_only_where_the_host_would_hear_of_it`,
  `test: close::only_turn_off_tells_the_host`,
  `test: sdl::shell::a_close_asks_where_the_host_hears_and_tells_it_once_turned_off`,
  `test: native::sdl_thread::input::a_cancelled_close_leaves_nothing_held_and_the_input_goes_on`,
  `test: native::sdl_thread::input::a_key_down_after_the_close_s_lift_never_reaches_the_guest`,
  `test: native::sdl_thread::question::the_question_s_words_are_macos_s_in_the_font`,
  `test: native::sdl_thread::question::the_font_scales_with_the_density`,
  `test: native::sdl_thread::question::return_and_escape_cancel_and_space_answers_the_focus`,
  `test: native::sdl_thread::question::a_click_answers_the_button_it_went_down_and_up_on`,
  `test: native::sdl_thread::render::an_overlay_is_drawn_and_told_unshown_once`,
  `test: reims-vgpu::graphics::sink::the_window_s_overlay_is_drawn_over_everything_in_a_pass_of_its_own`,
  `test: native::sdl_thread::sdl::the_question_s_default_is_cancel`,
  `test: native::sdl_thread::sdl::an_answer_is_turn_off_only_by_its_button`,
  `manual` (the question on Wayland without zenity, X11 and Windows;
  Cancel, then Turn Off)
- A GPU the driver reports lost, which the library makes no new device for:
  the window stays open and keeps the last frame it showed, and draws no
  more. Its swapchain makes no Vulkan call on the lost device from then on,
  and none is made again; a window opened after the loss stays empty. Only
  the destroys at the window's end reach the device.
  `test: present::a_lost_device_s_presenter_is_kept_and_asked_no_more`,
  `test: reims-vgpu-vulkan::rt_platform::swapchain::a_lost_device_s_swapchain_makes_no_vulkan_call_but_its_end_s_destroys_on_vulkan`,
  `test: reims-vgpu::graphics::rt_window::on_vulkan_a_lost_device_s_sink_asks_for_no_surface`,
  `manual` (a boot whose rt device is lost: the Windows mini's UHD 630)

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
