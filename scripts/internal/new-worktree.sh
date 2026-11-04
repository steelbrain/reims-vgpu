#!/bin/bash
# A new worktree whose large tracked files are APFS clones of the main checkout's, so git writes
# only the files that differ. Its builds run in its own target/ (no CARGO_TARGET_DIR), in the
# environment of scripts/internal/dev-env.sh, which builds through kache where it is installed
# (docs/development.md).
#
#   scripts/internal/new-worktree.sh PATH BRANCH [START]
#
# BRANCH is a new branch at START (default: the main checkout's HEAD). tests/suite holds 257 of
# the tree's 278 MiB (its expected and generated files), and a plain `git worktree add` writes
# every byte of it again. Here the main checkout's tests/suite is cloned into PATH first (on
# macOS cp -c, which writes nothing until a file changes; on Linux cp --reflink=auto, a copy
# where the filesystem has no reflinks). git then takes each clone that matches START's blob as
# clean and writes the rest, and the clone's files START does not track are removed, so the
# worktree holds START's tree exactly, whatever the main checkout's files are.
#
# The worktree is added from the main checkout: a new worktree takes the sparse-checkout
# settings of the one it is added from.
set -euo pipefail
[[ $# -ge 2 ]] || { echo "usage: scripts/internal/new-worktree.sh PATH BRANCH [START]" >&2; exit 2; }
path=$1 branch=$2
main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')
start=${3:-$(git -C "$main" rev-parse HEAD)}
big=tests/suite
[[ ! -e $path ]] || { echo "new-worktree: $path exists" >&2; exit 1; }
git -C "$main" worktree add -q --no-checkout -b "$branch" "$path" "$start"
path=$(cd "$path" && pwd)
mkdir -p "$path/$(dirname "$big")"
if [[ $(uname) == Darwin ]]; then
  cp -c -p -R "$main/$big" "$path/$big"
else
  cp -a --reflink=auto "$main/$big" "$path/$big"
fi
git -C "$path" reset -q
git -C "$path" update-index -q --refresh || true
git -C "$path" checkout -q -- .
git -C "$path" clean -q -f -d -x -- "$big"
echo "new-worktree: $path on $branch at $(git -C "$path" rev-parse --short HEAD)"
