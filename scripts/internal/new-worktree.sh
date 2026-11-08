#!/bin/bash
# A new worktree on a new branch. Its builds run in its own target/ (no CARGO_TARGET_DIR), in the
# environment of scripts/internal/dev-env.sh, which builds through kache where it is installed
# (docs/development.md).
#
#   scripts/internal/new-worktree.sh PATH BRANCH [START]
#
# BRANCH is a new branch at START (default: the main checkout's HEAD). The worktree is added from
# the main checkout: a new worktree takes the sparse-checkout settings of the one it is added from.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: scripts/internal/new-worktree.sh PATH BRANCH [START]" >&2; exit 2; }
path=$1 branch=$2
main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')
start=${3:-$(git -C "$main" rev-parse HEAD)}
[[ ! -e $path ]] || { echo "new-worktree: $path exists" >&2; exit 1; }
git -C "$main" worktree add -q -b "$branch" "$path" "$start"
echo "new-worktree: $(cd "$path" && pwd) on $branch at $(git -C "$path" rev-parse --short HEAD)"
