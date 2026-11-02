#!/usr/bin/env bash
# The Vulkan Video H.264 decoder against a software reference.
#
#   crates/reims-vgpu-video-vulkan/tools/h264-check.sh
#
# Runs on Linux with a GPU that decodes H.264 through Vulkan Video, and x264
# and ffmpeg. The streams and their references are the VA-API backend's H.264
# check's (crates/reims-vgpu-video-vaapi/tools/h264-check.sh, whose recipes
# below are copied from it and kept in step with it), made the same way into
# the same directory, RVGV_H264_OUT (build/h264-check by default): a directory
# that check filled is taken as it is, and one this fills serves it. x264 and
# ffmpeg are test oracles only, and nothing they write is committed.
#
# Each stream is decoded by examples/vulkan-h264-framemd5 on the device
# VideoDevice::create picks, from its MP4 (the avcC record and length-prefixed
# samples) and from its Annex B form, in display order: every frame's NV12 MD5
# must equal ffmpeg's. One stream is decoded again with its pictures copied
# out of the DPB on the graphics queue, and again on a lent queue (each copy
# submitted once the host sees its decode done), and three are landed in
# buffers. Decode order must give the same frames, as a set. The
# crafted streams (long-term references, memory management operations 2 to 6,
# frame number gaps, 1- to 3-byte lengths), the SPS switch mid-stream and the
# lost reference are the VA-API check's, as are their verdicts. A stream whose
# SPS allows fields must be refused with "interlaced".
#
# The streams are made first, with no GPU; the decodes then run holding every
# GPU's lock (scripts/internal/gpu-locks.sh --all), as the script runs itself
# again under it. Each run's frames and log go to RVGV_VULKAN_H264_OUT
# (build/vulkan-h264-check). The exit status is 1 where any check fails.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../../reims-vgpu-video-vaapi/tools/lib.sh"

out=${RVGV_H264_OUT:-$root/build/h264-check}
runs=${RVGV_VULKAN_H264_OUT:-$root/build/vulkan-h264-check}
mkdir -p "$out" "$runs"
tool=$CARGO_TARGET_DIR/release/examples/vulkan-h264-framemd5
info=$CARGO_TARGET_DIR/release/examples/h264-info
poc1=$CARGO_TARGET_DIR/release/examples/h264-poc1
craft=$CARGO_TARGET_DIR/release/examples/h264-craft

# An asymmetric scaling matrix, so that a list in the wrong scan order or
# the wrong slot changes the picture.
cqm=$out/asymmetric.cqm

# name size frames qpfile-lines x264-options... (the VA-API check's)
streams=(
  "baseline-720p|1280x720|90|30 i,60 I|--profile baseline --preset fast --slices 4 --ref 3 --keyint 300 --min-keyint 151"
  "baseline-1080p|1920x1080|45|20 i|--profile baseline --preset fast --slices 2 --ref 2 --keyint 300 --min-keyint 151"
  "main-720p|1280x720|120|40 i,80 I|--profile main --preset medium --bframes 3 --b-pyramid normal --ref 4 --slices 2 --weightp 2 --keyint 300 --min-keyint 151"
  "main-1080p-opengop|1920x1080|90||--profile main --preset medium --bframes 3 --open-gop --keyint 30 --slices 3 --ref 3"
  "high-720p-cqm|1280x720|60|30 i|--profile high --preset medium --bframes 3 --b-pyramid strict --ref 6 --8x8dct --cqmfile $cqm --slices 3 --keyint 300 --min-keyint 151"
  "high-1080p|1920x1080|120|50 i,100 I|--profile high --preset medium --bframes 3 --b-pyramid normal --ref 5 --slices 4 --weightb --direct auto --cqm jvt --keyint 300 --min-keyint 151"
  "high-1080p-deep|1920x1080|60||--profile high --preset slower --bframes 8 --b-pyramid normal --ref 16 --keyint 300 --min-keyint 151"
  "main-ref1-720p|1280x720|60||--profile main --preset fast --bframes 0 --ref 1 --weightp 0 --keyint 300 --min-keyint 151"
  "baseline-ref1-360p|640x360|60||--profile baseline --preset fast --ref 1 --keyint 300 --min-keyint 151"
  "main-gapsrc-720p|1280x720|60|30 i|--profile main --preset fast --bframes 0 --ref 1 --weightp 0 --keyint 300 --min-keyint 151"
)

# Crafted streams: name|source|reference (ffmpeg, or the source's
# decode)|h264-craft's edits (the VA-API check's).
lt="--max-refs 3 --mark 0:lt --mark 10:4:2,1:1,6:1 --mark 20:2:0,3:0:0 --mark 30:4:1 --mark 40:5 --mark 50:4:1,1:2,6:0"
crafted=(
  "ltr-mmco-cabac|main-ref1-720p|ffmpeg|$lt"
  "ltr-mmco-cavlc|baseline-ref1-360p|ffmpeg|$lt"
  "gaps5-ref1|main-gapsrc-720p|source|--gaps-allowed --gap 30:5"
  "gaps2-ref4|main-gapsrc-720p|source|--gaps-allowed --max-refs 4 --gap 30:2"
  "gaps9-ref4|main-gapsrc-720p|source|--gaps-allowed --max-refs 4 --gap 30:9"
  "len2-baseline-360p|baseline-ref1-360p|source|--length-size 2"
  "len3-baseline-360p|baseline-ref1-360p|source|--length-size 3"
)

# The stream NAME, as Annex B and as an MP4, as the VA-API check makes it.
encode() {
  local name=$1 size=$2 frames=$3 qp=$4 opts=$5
  if [[ -s $out/$name.mp4 && -s $out/$name.264 ]]; then
    return
  fi
  rm -f "$out/$name.mp4" "$out/$name.264" "$out/$name.ref"
  : > "$out/$name.qp"
  if [[ -n $qp ]]; then
    tr ',' '\n' <<<"$qp" | while read -r frame type; do echo "$frame $type -1"; done > "$out/$name.qp"
  fi
  # shellcheck disable=SC2086 # $opts is a list of options
  ffmpeg -nostdin -hide_banner -loglevel error -f lavfi \
      -i "testsrc2=size=$size:rate=30,noise=alls=14:allf=t+u" \
      -frames:v "$frames" -pix_fmt yuv420p -f yuv4mpegpipe - |
    x264 --quiet --demuxer y4m $opts --qpfile "$out/$name.qp" -o "$out/$name.part.264" - 2>/dev/null
  mv "$out/$name.part.264" "$out/$name.264"
  mux "$name"
}

# NAME's Annex B stream as an MP4.
mux() {
  ffmpeg -nostdin -hide_banner -loglevel error -y -r 30 -i "$out/$1.264" -c copy "$out/$1.part.mp4"
  mv "$out/$1.part.mp4" "$out/$1.mp4"
}

# ffmpeg's NV12 MD5 of each frame of NAME's Annex B stream.
reference() {
  local name=$1
  if [[ ! -s $out/$name.ref ]]; then
    hashes "$out/$name.ref" -hwaccel none -i "$out/$name.264" -fps_mode passthrough -pix_fmt nv12
  fi
}

# run INPUT LABEL [OPTION]...: `$tool INPUT OPTION...` into
# `$runs/LABEL.frames` and `.log`, and each frame's MD5 into
# `$runs/LABEL.md5`. A run that fails is a FAIL, counted in `failed`, and
# returns 1.
run() {
  local input=$1 label=$2
  shift 2
  local log=$runs/$label.log
  if "$tool" "$input" "$@" > "$runs/$label.frames" 2> "$log"; then
    awk '{ print $NF }' "$runs/$label.frames" > "$runs/$label.md5"
    echo "     $(tail -n1 "$log")"
    return 0
  fi
  echo "FAIL $label: $(tail -n1 "$log")"
  failed=1
  return 1
}

# refused LABEL INPUT PATTERN: PASS where `$tool INPUT` fails with a line
# matching PATTERN (grep -E) in its log; FAIL otherwise.
refused() {
  local label=$1 input=$2 pattern=$3
  local log=$runs/$label.log
  if "$tool" "$input" > /dev/null 2> "$log"; then
    echo "FAIL $label: decoded, should be refused"
    failed=1
  elif grep -Eq -- "$pattern" "$log"; then
    echo "PASS $label: refused: $(tail -n1 "$log")"
  else
    echo "FAIL $label: $(tail -n1 "$log")"
    failed=1
  fi
}

# three NAME LABEL: NAME's avcC and Annex B forms in display order, and its
# avcC form in decode order as a set, against NAME.ref.
three() {
  local name=$1
  if run "$out/$name.mp4" "$name.avcc"; then
    check "$name (avcC, display order)" "$out/$name.ref" "$runs/$name.avcc.md5"
  fi
  if run "$out/$name.264" "$name.annexb"; then
    check "$name (Annex B, display order)" "$out/$name.ref" "$runs/$name.annexb.md5"
  fi
  if run "$out/$name.mp4" "$name.decode" --decode-order; then
    sort "$out/$name.ref" > "$runs/$name.ref.sorted"
    sort "$runs/$name.decode.md5" > "$runs/$name.decode.sorted"
    check "$name (avcC, decode order, as a set)" "$runs/$name.ref.sorted" "$runs/$name.decode.sorted"
  fi
}

if [[ -z ${RVG_GPU_LOCKS_HELD:-} ]]; then
  # The streams, the references and the tools, with no GPU.
  (cd "$root" && cargo build --quiet --release -p reims-vgpu-video-vulkan --example vulkan-h264-framemd5)
  build reims-vgpu-video-vaapi h264-info h264-poc1 h264-craft
  {
    for name in INTRA4X4_LUMA INTRA4X4_CHROMAU INTRA4X4_CHROMAV INTER4X4_LUMA INTER4X4_CHROMAU INTER4X4_CHROMAV; do
      echo "$name ="
      for r in 0 1 2 3; do
        echo "$(for c in 0 1 2 3; do printf '%d,' $(( 8 + (r * 5 + c * 3 + ${#name}) % 23 )); done | sed 's/,$//')"
      done
    done
    for name in INTRA8X8_LUMA INTER8X8_LUMA; do
      echo "$name ="
      for r in 0 1 2 3 4 5 6 7; do
        echo "$(for c in 0 1 2 3 4 5 6 7; do printf '%d,' $(( 10 + (r * 7 + c * 2 + ${#name}) % 29 )); done | sed 's/,$//')"
      done
    done
  } > "$cqm"
  for entry in "${streams[@]}"; do
    IFS='|' read -r name size frames qp opts <<<"$entry"
    encode "$name" "$size" "$frames" "$qp" "$opts"
    reference "$name"
  done
  name=baseline-720p-poc1
  if [[ ! -s $out/$name.mp4 ]]; then
    "$poc1" "$out/baseline-720p.264" "$out/$name.264" 2>/dev/null
    mux "$name"
  fi
  reference "$name"
  for entry in "${crafted[@]}"; do
    IFS='|' read -r name source ref edits <<<"$entry"
    # shellcheck disable=SC2086 # $edits is a list of options
    "$craft" "$out/$source.264" "$out/$name" $edits 2> "$out/$name.craft.log"
    if [[ $ref == source ]]; then
      cp "$out/$source.ref" "$out/$name.ref"
    else
      rm -f "$out/$name.ref"
      reference "$name"
    fi
  done
  name=switch-720p-1080p
  cat "$out/baseline-720p.264" "$out/high-1080p.264" > "$out/$name.264"
  cat "$out/baseline-720p.ref" "$out/high-1080p.ref" > "$out/$name.ref"
  "$craft" "$out/main-gapsrc-720p.264" "$out/drop20-ref1" --drop 20 --check > "$out/drop20-ref1.check" 2>&1
  encode high-1080p-interlaced 1920x1080 10 "" "--profile high --preset fast --tff"
  encode high-1080p-fakeint 1920x1080 30 "" "--profile high --preset fast --fake-interlaced --bframes 2"
  exec bash "$root/scripts/internal/gpu-locks.sh" --all -- bash "$0" "$@"
fi

echo "vulkan h264-check, $(date -Is), $(hostname)"
"$tool" --capabilities
for entry in "${streams[@]}"; do
  IFS='|' read -r name _ <<<"$entry"
  echo "---- $name: $("$info" "$out/$name.264")"
  three "$name"
done

# Order count type 1: baseline-720p's SPS rewritten from type 2 to an
# equivalent type 1, which must not change ffmpeg's reference either.
name=baseline-720p-poc1
echo "---- $name: $("$info" "$out/$name.264")"
check "$name (ffmpeg's reference against type 2's)" "$out/baseline-720p.ref" "$out/$name.ref"
if run "$out/$name.mp4" "$name.avcc"; then
  check "$name (avcC, display order)" "$out/$name.ref" "$runs/$name.avcc.md5"
fi
if run "$out/$name.264" "$name.annexb"; then
  check "$name (Annex B, display order)" "$out/$name.ref" "$runs/$name.annexb.md5"
fi

for entry in "${crafted[@]}"; do
  IFS='|' read -r name source _ edits <<<"$entry"
  echo "---- $name (crafted from $source: $edits): $("$info" "$out/$name.264")"
  three "$name"
done

# The caller-facing options, on one stream each.
if run "$out/high-1080p.mp4" high-1080p.len2 --length-size 2; then
  check "high-1080p (2-byte NAL lengths)" "$out/high-1080p.ref" "$runs/high-1080p.len2.md5"
fi
if run "$out/main-720p.mp4" main-720p.len3 --length-size 3; then
  check "main-720p (3-byte NAL lengths)" "$out/main-720p.ref" "$runs/main-720p.len3.md5"
fi
if run "$out/main-720p.mp4" main-720p.hold --hold 24; then
  check "main-720p (24 frames held by the caller)" "$out/main-720p.ref" "$runs/main-720p.hold.md5"
fi
# Pictures copied out of the DPB on a queue of their own, behind a
# semaphore wait, as on a device whose decode family does not copy.
if run "$out/high-1080p.mp4" high-1080p.graphics --copy-on-graphics; then
  check "high-1080p (copied out on the graphics queue)" "$out/high-1080p.ref" "$runs/high-1080p.graphics.md5"
fi
# Pictures landed in buffers, cropped, at offsets and pitches of their own:
# a cropped frame, a frame number wrap, an open GOP, and the copy queue.
for name in high-1080p baseline-720p main-1080p-opengop; do
  if run "$out/$name.mp4" "$name.buffer" --buffer; then
    check "$name (landed in buffers)" "$out/$name.ref" "$runs/$name.buffer.md5"
  fi
done
if run "$out/high-1080p.mp4" high-1080p.buffer-graphics --buffer --copy-on-graphics; then
  check "high-1080p (landed in buffers on the graphics queue)" "$out/high-1080p.ref" "$runs/high-1080p.buffer-graphics.md5"
fi
# Pictures copied out on a queue another submitter lends, each submitted
# once the host sees its decode done, into images and into buffers.
if run "$out/high-1080p.mp4" high-1080p.lent --copy-lent; then
  check "high-1080p (copied out on a lent queue)" "$out/high-1080p.ref" "$runs/high-1080p.lent.md5"
fi
if run "$out/high-1080p.mp4" high-1080p.buffer-lent --buffer --copy-lent; then
  check "high-1080p (landed in buffers on a lent queue)" "$out/high-1080p.ref" "$runs/high-1080p.buffer-lent.md5"
fi

# A new SPS at an IDR picture, with another size and profile, mid-stream.
name=switch-720p-1080p
if run "$out/$name.264" "$name.annexb"; then
  check "$name (Annex B, display order)" "$out/$name.ref" "$runs/$name.annexb.md5"
fi

# A lost reference: main-gapsrc-720p without frame 20, whose SPS allows no
# gap. Its 59 frames decode, the 20 before the loss and the 30 from the I
# picture at 30 on as the source's.
name=drop20-ref1
echo "---- $name (crafted from main-gapsrc-720p: --drop 20): $("$info" "$out/$name.264")"
if grep -q refused "$out/$name.check"; then
  echo "FAIL $name: the stand-in refused a sample, so it goes to no GPU: $(grep -m1 refused "$out/$name.check")"
  failed=1
elif run "$out/$name.mp4" "$name.avcc"; then
  { sed -n '1,20p' "$out/main-gapsrc-720p.ref"; sed -n '31,60p' "$out/main-gapsrc-720p.ref"; } > "$runs/$name.kept.ref"
  { sed -n '1,20p' "$runs/$name.avcc.md5"; sed -n '30,59p' "$runs/$name.avcc.md5"; } > "$runs/$name.kept.md5"
  n=$(wc -l < "$runs/$name.avcc.md5")
  if [[ $n != 59 ]]; then
    echo "FAIL $name: $n frames, not 59"
    failed=1
  else
    check "$name (59 frames; the 50 the loss does not reach)" "$runs/$name.kept.ref" "$runs/$name.kept.md5"
  fi
fi

# Interlaced coding, MBAFF and fake-interlaced alike, is refused before any
# slice reaches the GPU.
for name in high-1080p-interlaced high-1080p-fakeint; do
  refused "$name" "$out/$name.mp4" '^vulkan-h264-framemd5: interlaced: '
done

exit "$failed"
