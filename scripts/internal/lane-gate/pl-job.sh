#!/bin/bash
# Another lane's command on the lane host as a potluck job, with the gate's host scripts beside it:
#
#   pl-job.sh NAME [DISPATCH-OPTION...] -- COMMAND-LINE
#
# Dispatches COMMAND-LINE (a shell command line, its words joined, run by bash in the workspace's
# root, so $PWD there is the workspace) from a throwaway commit whose lg/bin
# holds the gate's host scripts (lg/bin/msi-launch.sh is the gate's VM launch, for a lane's own boot
# script: LAUNCH=$PWD/lg/bin/msi-launch.sh), waits for it with its output, brings the workspace's
# results/ back (--keep results) into LANE_GATE_HOME/pljobs/NAME/, and removes the workspace, the
# worktree and the branch. DISPATCH-OPTIONs go to potluck dispatch as they are (--timeout is
# required; --exclusive --gpu 5080 for a boot); --on node=msi-laptop is added. Exits with the job's
# exit.
set -uo pipefail
L=$(cd "$(dirname "$0")" && pwd)
GH=${LANE_GATE_HOME:-$HOME/.cache/rgk-lane-gate}
name=${1:?usage: pl-job.sh NAME [DISPATCH-OPTION...] -- COMMAND-LINE}
shift
opts=()
while (($#)) && [[ $1 != -- ]]; do opts+=("$1"); shift; done
[[ ${1:-} == -- ]] || { echo "pl-job: no -- CMD" >&2; exit 2; }
shift
(($#)) || { echo "pl-job: no CMD" >&2; exit 2; }
stamp=$(date +%Y%m%d-%H%M%S)
S=$(mktemp -d /tmp/lg-pljob.XXXXXX)
mkdir -p "$S/lg/bin"
find "$L" -maxdepth 1 -type f -exec cp -p {} "$S/lg/bin/" \;
printf '%s\n' "$*" > "$S/lg/cmd"
branch=steelbrain/lane-gate-job-$name-$stamp wt=$HOME/.cache/lane-gate-ship/job-$name-$stamp
commit=$(cd "$L" && bash "$L/ship-commit.sh" "$S" "$(git -C "$L" rev-parse HEAD)" "$branch" "$wt" \
  ":construction: WIP lane-gate job $name $stamp (throwaway)") || { rm -rf "$S"; exit 1; }
rm -rf "$S"
echo "pl-job: $name from $commit on $branch: $*"
out=$(cd "$wt" && potluck dispatch --json --on "node=${POTLUCK_MSI_NODE:-msi-laptop}" --label "lane-gate=job-$name" \
  --keep results "${opts[@]}" -- bash -c 'eval "$(cat lg/cmd)"') || { echo "pl-job: potluck dispatch: $out" >&2; exit 1; }
job=$(python3 -I -c 'import json, sys; print(json.load(sys.stdin)["job"])' <<< "$out")
echo "pl-job: potluck job $job"
(cd "$wt" && potluck wait "$job")
rc=$?
ws=$(potluck status --json | python3 -I -c '
import json, sys
d = json.load(sys.stdin)
for k in ("running", "recent", "queued"):
    for j in d.get(k) or []:
        if j.get("id") == sys.argv[1] and j.get("workspace"):
            print(j["workspace"]); sys.exit(0)' "$job" || true)
# results/ as the job's --keep brought it back, else from its workspace (a job that ended
# without its keep).
k=${POTLUCK_STATE:-$HOME/.local/state/potluck}/jobs/$job/keep/results
mkdir -p "$GH/pljobs"
rm -rf "$GH/pljobs/$name"
if [[ -d $k ]]; then
  cp -R "$k" "$GH/pljobs/$name" && echo "pl-job: results in $GH/pljobs/$name"
elif [[ -n $ws ]]; then
  potluck cp "$ws:results" "$GH/pljobs/$name" > /dev/null 2>&1 && echo "pl-job: results in $GH/pljobs/$name"
fi
[[ -z $ws ]] || potluck release "$ws" > /dev/null 2>&1 || true
git -C "$L" worktree remove --force "$wt" 2> /dev/null || true
git -C "$L" branch -q -D "$branch" 2> /dev/null || true
echo "pl-job: $name: job $job exit $rc"
exit $rc
