#!/bin/bash
# A throwaway commit of a directory's files for the lane gate's potluck jobs, in its own worktree:
#
#   ship-commit.sh DIR PARENT BRANCH WORKTREE MESSAGE
#
# The commit's tree is DIR's files exactly (their executable bits kept; DIR holds no symbolic
# links), its parent PARENT (the candidate a job builds from, whose history potluck's mirrors then
# carry), and BRANCH points at it, checked out in WORKTREE (removed first where it exists), from
# which the jobs are dispatched. Run from a checkout of the repository; prints the commit.
# What it commits is build output for a run (our own scripts, kexts, bundle and guest tools; never
# an Apple binary or anything from a guest), on a branch that never merges and goes after the run.
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: ship-commit.sh DIR PARENT BRANCH WORKTREE MESSAGE" >&2; exit 2; }
S=$(cd "$1" && pwd) parent=$2 branch=$3 wt=$4 msg=$5
gd=$(git rev-parse --path-format=absolute --git-common-dir)
[[ -z $(cd "$S" && find . -type l -print -quit) ]] || { echo "ship-commit: symbolic links in $S" >&2; exit 1; }
tmp=$(mktemp -d /tmp/lg-ship.XXXXXX)
# bash 3.2, macOS's, exits 0 from a set -u error where an EXIT trap is set, with $? 0 in the
# trap too, so an aborted run would read as a pass (docs/lessons.md): the trap fails a run
# that exits 0 short of its end, which sets `ended`.
ended=0
trap 'rc=$?; rm -rf "$tmp"; [[ $rc != 0 || $ended == 1 ]] || rc=1; exit $rc' EXIT
(cd "$S" && find . -type f | sed 's#^\./##' | LC_ALL=C sort) > "$tmp/files"
sed "s#^#$S/#" "$tmp/files" | git --git-dir="$gd" hash-object -w --stdin-paths > "$tmp/blobs"
while IFS= read -r f; do if [[ -x $S/$f ]]; then echo 100755; else echo 100644; fi; done < "$tmp/files" > "$tmp/modes"
paste -d' ' "$tmp/modes" "$tmp/blobs" | paste -d'\t' - "$tmp/files" |
  GIT_INDEX_FILE=$tmp/index git --git-dir="$gd" update-index --add --index-info
tree=$(GIT_INDEX_FILE=$tmp/index git --git-dir="$gd" write-tree)
commit=$(git --git-dir="$gd" commit-tree "$tree" -p "$parent" -m "$msg")
[[ ! -e $wt ]] || git worktree remove --force "$wt"
git branch -f -q "$branch" "$commit"
mkdir -p "$(dirname "$wt")"
git worktree add -q "$wt" "$branch"
ended=1
echo "$commit"
