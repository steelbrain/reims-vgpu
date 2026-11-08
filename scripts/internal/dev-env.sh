# The environment a developer's builds run in. Source it from the worktree's root, which builds in
# its own target/ (no CARGO_TARGET_DIR), before each cargo command:
#
#   . scripts/internal/dev-env.sh && cargo nextest run -p reims-vgpu-vulkan -E 'test(/^core::/)'
#
# Builds are incremental, as cargo's dev and test profiles are. Where kache is on the PATH, rustc
# runs through it (RUSTC_WRAPPER=kache): a compiler cache that stores each unit's outputs on this
# machine, keyed by the unit's inputs, and restores them as clones of the stored files, so a
# worktree builds only the units no tree on the machine built before. A unit being edited compiles
# incrementally instead, with kache's private incremental state, which a new worktree seeds from a
# compatible one's (the host's configuration turns both on: docs/development.md). kache stays local,
# whatever its configuration says (KACHE_LOCAL_ONLY=1). It runs each build script with its OUT_DIR
# in a directory of its store named by the script's inputs, which the target directory's OUT_DIR
# links to (KACHE_BUILD_SCRIPT_HERMETIC=1), so a script's run is shared between worktrees even where
# what it writes names its OUT_DIR, as SDL3's library does. Without kache the build is cargo's own:
# nothing in the repository needs kache, and it is never Cargo's configured wrapper
# (docs/development.md, "Building through kache").
# A shell that sourced an earlier version keeps no incremental switch of its own.
unset CARGO_INCREMENTAL KACHE_ADAPTIVE_INCREMENTAL
if command -v kache > /dev/null 2>&1; then
  export RUSTC_WRAPPER=kache KACHE_LOCAL_ONLY=1 KACHE_BUILD_SCRIPT_HERMETIC=1
fi
