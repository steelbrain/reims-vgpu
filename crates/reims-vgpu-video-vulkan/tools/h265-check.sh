#!/usr/bin/env bash
# The Vulkan Video H.265 decoder against a software reference.
#
#   crates/reims-vgpu-video-vulkan/tools/h265-check.sh
#
# Runs on Linux with a GPU that decodes H.265 through Vulkan Video, and
# x265, kvazaar, ffmpeg and, where installed, libde265's dec265. The streams
# and their references are the VA-API backend's HEVC check's
# (crates/reims-vgpu-video-vaapi/tools/hevc-check.sh, whose recipes below
# are copied from it and kept in step with it), made the same way into the
# same directory, RVGV_HEVC_OUT (build/hevc-check by default): a directory
# that check filled is taken as it is, and one this fills serves it. x265,
# kvazaar, dec265 and ffmpeg are test oracles only, and nothing they write
# is committed. kvazaar is found as RVGV_KVAZAAR names it, else on the PATH;
# where there is none, each of its streams (tiles, dependent slice segments
# and WPP with slices, low-delay and eight-reference GOPs) is a FAIL that
# says so, and nothing else changes.
#
# The streams cover Main and Main 10, B-frames with deep reorder, open GOPs
# whose CRA pictures lead RASL ones, RADL pictures, slices, dependent slice
# segments, tiles, WPP, scaling lists (the defaults and an asymmetric
# matrix), weighted prediction, temporal layers, lossless and transform
# skip, CTBs of 16, 32 and 64, a cropped 1366x766, 1080p and 4K;
# hevc-craft's copies add reference list modification, short-term sets
# predicted from others in the SPS and in slice headers, and long-term
# pictures.
#
# Each stream is decoded by examples/vulkan-h265-framemd5 on the device
# VideoDevice::create picks, from its MP4 (the hvcC record and
# length-prefixed samples) and from its Annex B form, in display order:
# every frame's MD5, over NV12 for Main and P010 for Main 10, must equal the
# reference's (ffmpeg's software decode, or the encoder's reconstruction
# where the two differ, as the VA-API check takes it). Decode order must
# give the same frames, as a set. Then:
# - a seek: the open-GOP streams cut at their second CRA picture, whose
#   RASL pictures are skipped, from Annex B and from the MP4 by --skip; and
#   the same CRA picture made a BLA one mid-stream;
# - the caller-facing options: 2- and 3-byte NAL lengths, frames held by
#   the caller;
# - copies out of the DPB on the graphics queue, behind a semaphore wait;
# - pictures landed in buffers, cropped, at offsets and pitches of their
#   own, 8- and 10-bit, and through the graphics queue;
# - access units read in place, from one host-visible allocation where
#   each lies as the guest's video driver places it: every picture must be
#   read in place, none copied;
# - a new SPS at an IDR picture, Main 720p to Main 10 1080p, mid-stream;
# - the refusals: 4:2:2, 4:4:4 and 12-bit streams, tiles narrower than Main
#   allows, and a picture of more than 8 current references, which Vulkan
#   Video's std structure holds, after the frames before it;
# - with the Khronos validation layer, where it is installed, a stream of
#   each kind into images and buffers, on the decode and graphics queues,
#   and in place: the frames must match and the layer say nothing.
#
# The streams are made first, with no GPU; the decodes then run holding
# every GPU's lock (scripts/internal/gpu-locks.sh --all), as the script runs
# itself again under it. Each run's frames and log go to
# RVGV_VULKAN_H265_OUT (build/vulkan-h265-check). The exit status is 1
# where any check fails.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../../reims-vgpu-video-vaapi/tools/lib.sh"

out=${RVGV_HEVC_OUT:-$root/build/hevc-check}
runs=${RVGV_VULKAN_H265_OUT:-$root/build/vulkan-h265-check}
mkdir -p "$out" "$runs"
tool=$CARGO_TARGET_DIR/release/examples/vulkan-h265-framemd5
info=$CARGO_TARGET_DIR/release/examples/hevc-info
cut=$CARGO_TARGET_DIR/release/examples/hevc-cut
craft=$CARGO_TARGET_DIR/release/examples/hevc-craft
kvazaar=${RVGV_KVAZAAR:-kvazaar}
have_kvazaar=0
if command -v "$kvazaar" > /dev/null; then
  have_kvazaar=1
fi

# The VA-API check's asymmetric scaling lists in HM's format, so that a
# list in the wrong scan order, transposed, in the wrong slot or with the
# wrong DC changes the picture. Each chroma V list equals its U list, which
# x265 codes as a prediction from it; the two 32x32 lists differ, since
# x265 4.3 writes a pred_matrix_id_delta ffmpeg rejects when they are equal
# (2026-10-24).
cqm=$out/asymmetric.cqm

noise="noise=alls=14:allf=t+u"
# name|size|frames|bits|source filter|x265 options (the VA-API check's)
streams=(
  "main-720p|1280x720|120|8|testsrc2|--preset medium --bframes 4 --b-pyramid --ref 4 --keyint 48 --min-keyint 48"
  "main-1080p-slices|1920x1080|60|8|testsrc2|--preset fast --slices 4 --bframes 3 --ref 3 --keyint 300"
  "main-720p-closed|1280x720|90|8|mandelbrot|--preset fast --no-open-gop --no-wpp --keyint 30 --min-keyint 30 --bframes 3"
  "main-720p-radl|1280x720|90|8|testsrc2|--preset fast --no-open-gop --radl 2 --keyint 30 --min-keyint 30 --bframes 3"
  "main-720p-cqm-default|1280x720|60|8|testsrc2|--preset fast --scaling-list default --bframes 3"
  "main-720p-cqm-file|1280x720|60|8|testsrc2|--preset fast --scaling-list $cqm --bframes 3 --ctu 32"
  "main-720p-fade|1280x720|120|8|fade|--preset medium --weightp --weightb --bframes 3 --ref 3"
  "main-1366x766-ctu16|1366x766|60|8|testsrc2|--preset fast --ctu 16 --bframes 2 --tskip --amp --rect"
  "main-720p-deep|1280x720|90|8|testsrc2|--preset slower --bframes 8 --ref 6 --b-pyramid --keyint 300"
  "main-720p-temporal|1280x720|60|8|testsrc2|--preset fast --temporal-layers 3 --bframes 3"
  "main-360p-lossless|640x360|20|8|testsrc2|--preset fast --lossless --bframes 2"
  "main-4k|3840x2160|30|8|testsrc2|--preset faster --bframes 3"
  "mainstill-1080p|1920x1080|1|8|testsrc2|--preset medium --profile mainstillpicture --keyint 1 --frames 1"
  "main10-720p|1280x720|120|10|testsrc2|--preset medium --bframes 4 --b-pyramid --ref 4 --keyint 48 --min-keyint 48"
  "main10-1080p-slices-cqm|1920x1080|60|10|mandelbrot|--preset fast --slices 3 --scaling-list default --weightb --bframes 3"
  "main10-720p-fade|1280x720|90|10|fade|--preset fast --weightp --weightb --bframes 3"
  "main10-1366x766-ctu16|1366x766|45|10|testsrc2|--preset fast --ctu 16 --bframes 2"
  "main10-4k|3840x2160|30|10|testsrc2|--preset faster --bframes 3"
  "main-720p-ref1|1280x720|60|8|testsrc2|--preset fast --bframes 0 --ref 1 --no-temporal-mvp --keyint 300"
)

# kvazaar's streams (the VA-API check's): name|size|frames|source
# filter|kvazaar options, after --preset, whose GOP and references the
# options then replace. Every one also gets --hash md5.
kvazaar_streams=(
  "kvz-720p-tiles|1280x720|60|testsrc2|--preset medium --gop 8 --level 4.1 --tiles 3x3"
  "kvz-720p-tiles-explicit|1280x720|60|testsrc2|--preset fast --gop 8 --level 4.1 --tiles-width-split 256,640,960 --tiles-height-split 128,512"
  "kvz-1080p-tiles-slices|1920x1080|30|testsrc2|--preset fast --gop 8 --level 5.1 --tiles 2x2 --slices tiles"
  "kvz-1366x766-tiles|1366x766|30|testsrc2|--preset fast --gop 8 --level 4.1 --tiles 2x3 --slices tiles"
  "kvz-4k-tiles-slices|3840x2160|20|mandelbrot|--preset ultrafast --gop 8 --level 5.1 --tiles 4x4 --slices tiles"
  "kvz-720p-dependent|1280x720|60|testsrc2|--preset fast --gop 8 --level 4.1 --wpp --slices wpp"
  "kvz-1080p-dependent-cra|1920x1080|70|testsrc2|--preset fast --gop 16 --period 32 --open-gop --level 5.1 --wpp --slices wpp"
  "kvz-720p-lowdelay|1280x720|60|testsrc2|--preset fast --gop lp-g8d4t2 --ref 4 --level 4.1"
  "kvz-720p-ref8|1280x720|30|testsrc2|--preset fast --gop 0 --ref 8 --level 4.1"
  "kvz-720p-fields|1280x720|30|testsrc2|--preset fast --gop lp-g4d3t1 --level 4.1 --source-scan-type tff"
  "kvz-720p-tools|1280x720|30|fade|--preset fast --gop 8 --level 4.1 --signhide --transform-skip --amp --smp --vaq 5 --deblock -3:2 --sao band --aud"
  "kvz-1080p-ref1|1920x1080|30|testsrc2|--preset fast --gop 0 --ref 1 --no-tmvp --level 4.1"
)

# hevc-craft's streams (the VA-API check's): name|source|its edits.
crafted=(
  "main-720p-lists|main-720p|--lists"
  "main-720p-sets-sps|main-720p|--sets sps"
  "main-720p-sets-slice|main-720p|--sets slice"
  "main-720p-closed-all|main-720p-closed|--lists --sets slice --long-term"
  "main-1080p-slices-lists-sps|main-1080p-slices|--lists --sets sps"
  "main-1080p-slices-all|main-1080p-slices|--lists --sets slice --long-term"
  "main10-720p-all|main10-720p|--lists --sets slice --long-term"
  "main-720p-ref1-long-term|main-720p-ref1|--long-term"
  "main-720p-ref1-all|main-720p-ref1|--lists --sets sps --long-term"
  "kvz-1080p-dependent-cra-lists|kvz-1080p-dependent-cra|--lists"
  "kvz-1080p-dependent-cra-all|kvz-1080p-dependent-cra|--lists --sets slice --long-term"
  "kvz-720p-lowdelay-lists-sps|kvz-720p-lowdelay|--lists --sets sps"
  "kvz-720p-lowdelay-all|kvz-720p-lowdelay|--lists --sets slice --long-term"
  "kvz-1080p-ref1-long-term|kvz-1080p-ref1|--long-term --sets slice"
)

# What the decoder refuses, cleanly, before any slice reaches the GPU
# (the VA-API check's): x265's name|size|frames|bits|chroma|options, and
# kvazaar's name|size|frames|source filter|options|what the refusal says.
refusals=(
  "main422-10|640x360|10|10|422|--preset fast --profile main422-10"
  "main444-8|640x360|10|8|444|--preset fast --profile main444-8"
  "main12|640x360|10|12|420|--preset fast --profile main12"
)
kvazaar_refusals=(
  "kvz-720p-tiles-narrow|1280x720|10|testsrc2|--preset fast --gop 8 --level 4.1 --tiles-width-split 256,448,960|ColumnWidthInLumaSamples is out of range"
)
# A stream whose pictures name more than 8 current references from its
# tenth on, which Vulkan Video's StdVideoDecodeH265PictureInfo does not
# hold: name|size|frames|source filter|options|the frames before it|what
# the refusal says. The frames before it must match, and it must be
# refused before it reaches the GPU.
kvazaar_partial=(
  "kvz-720p-ref15|1280x720|30|testsrc2|--preset fast --gop 0 --ref 15 --level 4.1|9|Vulkan Video takes 8"
)

# What runs again under the validation layer: name|options.
validated=(
  "main-720p|"
  "main10-720p|--buffer"
  "main-720p-ref1-long-term|--copy-on-graphics"
  "main10-1366x766-ctu16|--buffer --copy-on-graphics"
  "kvz-1080p-dependent-cra-all|"
  "kvz-720p-tiles-explicit|--buffer"
  "main-1080p-slices|--in-place"
  "main10-1080p-slices-cqm|--in-place --buffer"
)

source_filter() {
  local kind=$1 size=$2
  case $kind in
    testsrc2) echo "testsrc2=size=$size:rate=30,$noise" ;;
    mandelbrot) echo "mandelbrot=size=$size:rate=30,$noise" ;;
    # Brightness and contrast that change from frame to frame, for
    # weighted prediction to model.
    fade) echo "testsrc2=size=$size:rate=30,eq=brightness='0.25*sin(2*PI*t/3)':contrast='1+0.3*cos(2*PI*t/2)':eval=frame,$noise" ;;
  esac
}

# x265's stream NAME, its reconstruction's MD5s and its MP4 (the VA-API
# check's); 1 where one is not made, which leaves no stream a later run
# would take. A stream made again drops the reference of the one before.
encode() {
  local name=$1 size=$2 frames=$3 bits=$4 kind=$5 opts=$6 csp=${7:-420}
  if [[ -s $out/$name.mp4 && -s $out/$name.265 ]]; then
    return
  fi
  rm -f "$out/$name.mp4" "$out/$name.265" "$out/$name.ref"
  local pix=yuv${csp}p depth=()
  if (( bits > 8 )); then
    pix=yuv${csp}p${bits}le
    depth=(--input-depth "$bits" --output-depth "$bits")
  fi
  # shellcheck disable=SC2086 # $opts is a list of options
  if ! ffmpeg -nostdin -hide_banner -loglevel error -f lavfi -i "$(source_filter "$kind" "$size")" \
      -frames:v "$frames" -pix_fmt "$pix" -strict -1 -f yuv4mpegpipe - 2> /dev/null |
    nice -n 10 x265 --input - --y4m "${depth[@]}" $opts --recon "$out/$name.recon.y4m" \
        -o "$out/$name.part.265" 2> "$out/$name.x265.log"; then
    rm -f "$out/$name.part.265" "$out/$name.recon.y4m"
    echo "FAIL $name: x265: $(grep -m1 error "$out/$name.x265.log")"
    return 1
  fi
  local pix_out=nv12
  if (( bits > 8 )); then
    pix_out=p010le
  fi
  hashes "$out/$name.recon" -i "$out/$name.recon.y4m" -pix_fmt "$pix_out" || return 1
  rm -f "$out/$name.recon.y4m"
  mux "$name" x265
}

# NAME's stream, from `$out/NAME.part.265`, as `$out/NAME.265` and an MP4;
# 1 where ffmpeg cannot read what ENCODER wrote.
mux() {
  local name=$1 encoder=$2
  if ! ffmpeg -nostdin -hide_banner -loglevel error -y -r 30 -i "$out/$name.part.265" -c copy \
      -tag:v hvc1 "$out/$name.part.mp4" 2> "$out/$name.mux.log"; then
    rm -f "$out/$name.part.265" "$out/$name.part.mp4"
    echo "FAIL $name: ffmpeg cannot read $encoder's stream: $(head -n1 "$out/$name.mux.log")"
    return 1
  fi
  mv "$out/$name.part.mp4" "$out/$name.mp4"
  mv "$out/$name.part.265" "$out/$name.265"
}

# kvazaar's stream, its reconstruction's MD5s (but for field pictures,
# whose --debug output is not the decoded fields) and its MP4 (the VA-API
# check's). Without kvazaar, a stream not made yet is missing: 1.
encode_kvazaar() {
  local name=$1 size=$2 frames=$3 kind=$4 opts=$5
  if [[ -s $out/$name.mp4 && -s $out/$name.265 ]]; then
    return
  fi
  if (( !have_kvazaar )); then
    echo "FAIL $name: missing: no kvazaar ($kvazaar) to make it"
    return 1
  fi
  rm -f "$out/$name.mp4" "$out/$name.265" "$out/$name.ref"
  local recon=(--debug "$out/$name.recon.yuv")
  if [[ $opts == *--source-scan-type* ]]; then
    recon=()
  fi
  ffmpeg -nostdin -hide_banner -loglevel error -y -f lavfi -i "$(source_filter "$kind" "$size")" \
      -frames:v "$frames" -pix_fmt yuv420p -f yuv4mpegpipe "$out/$name.src.y4m" || return 1
  # shellcheck disable=SC2086 # $opts is a list of options
  if ! nice -n 10 "$kvazaar" -i "$out/$name.src.y4m" --input-fps 30 $opts --hash md5 \
      ${recon[@]+"${recon[@]}"} -o "$out/$name.part.265" > "$out/$name.kvazaar.log" 2>&1 < /dev/null; then
    rm -f "$out/$name.part.265" "$out/$name.recon.yuv" "$out/$name.src.y4m"
    echo "FAIL $name: kvazaar: $(tail -n1 "$out/$name.kvazaar.log")"
    return 1
  fi
  rm -f "$out/$name.src.y4m" "$out/$name.recon"
  if (( ${#recon[@]} )); then
    hashes "$out/$name.recon" -f rawvideo -pix_fmt yuv420p -s "$size" -i "$out/$name.recon.yuv" \
      -pix_fmt nv12 || return 1
    rm -f "$out/$name.recon.yuv"
  fi
  mux "$name" kvazaar
}

# How many pictures' MD5 SEI ffmpeg's software decode checked, and how many
# planes of them it found different (the VA-API check's).
picture_hashes() {
  local log
  log=$(ffmpeg -nostdin -hide_banner -loglevel debug -err_detect crccheck -hwaccel none \
      -i "$out/$1.265" -f null - 2>&1 || true)
  echo "$(grep -c 'Verifying checksum' <<<"$log") $(grep -c 'mismatching checksum' <<<"$log")"
}

reference() {
  local name=$1 bits=$2
  local pix=nv12
  if (( bits > 8 )); then
    pix=p010le
  fi
  if [[ ! -s $out/$name.ref ]]; then
    hashes "$out/$name.ref" -hwaccel none -i "$out/$name.265" -fps_mode passthrough -pix_fmt "$pix"
  fi
}

# libde265's decode of NAME's Annex B stream, each frame's MD5 in NV12 or
# P010, into NAME.de265; 1 where it fails (the VA-API check's).
de265() {
  local name=$1 bits=$2 size pix=yuv420p packed=nv12
  if (( bits > 8 )); then
    pix=yuv420p10le packed=p010le
  fi
  rm -f "$out/$name.de265"
  size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \
    "$out/$name.265" < /dev/null)
  if ! dec265 -q -o "$out/$name.de265.yuv" "$out/$name.265" > "$out/$name.de265.log" 2>&1 < /dev/null; then
    rm -f "$out/$name.de265.yuv"
    return 1
  fi
  hashes "$out/$name.de265" -f rawvideo -pix_fmt "$pix" -s "$size" -i "$out/$name.de265.yuv" \
    -pix_fmt "$packed" || return 1
  rm -f "$out/$name.de265.yuv"
}

bits_of() {
  local entry n b
  for entry in "${streams[@]}"; do
    IFS='|' read -r n _ _ b _ _ <<<"$entry"
    if [[ $n == "$1" ]]; then
      echo "$b"
      return
    fi
  done
  # kvazaar's streams are all 8-bit.
  echo 8
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

# all_in_place LABEL: PASS where LABEL's run read every picture in place;
# FAIL otherwise.
all_in_place() {
  local label=$1 line
  line=$(tail -n1 "$runs/$label.log")
  if [[ $line =~ ([0-9]+)\ pictures\ read\ in\ place\ and\ ([0-9]+)\ copied ]] &&
      (( BASH_REMATCH[1] > 0 && BASH_REMATCH[2] == 0 )); then
    echo "PASS $label: ${BASH_REMATCH[1]} pictures read in place, none copied"
  else
    echo "FAIL $label: not every picture read in place: $line"
    failed=1
  fi
}

# three NAME: NAME's hvcC and Annex B forms in display order, and its hvcC
# form in decode order as a set, against NAME.expect.
three() {
  local name=$1
  if run "$out/$name.mp4" "$name.hvcc"; then
    check "$name (hvcC, display order)" "$out/$name.expect" "$runs/$name.hvcc.md5"
  fi
  if run "$out/$name.265" "$name.annexb"; then
    check "$name (Annex B, display order)" "$out/$name.expect" "$runs/$name.annexb.md5"
  fi
  if run "$out/$name.mp4" "$name.decode" --decode-order; then
    sort "$out/$name.expect" > "$runs/$name.expect.sorted"
    sort "$runs/$name.decode.md5" > "$runs/$name.decode.sorted"
    check "$name (hvcC, decode order, as a set)" "$runs/$name.expect.sorted" "$runs/$name.decode.sorted"
  fi
}

if [[ -z ${RVG_GPU_LOCKS_HELD:-} ]]; then
  # The streams, the references and the tools, with no GPU.
  (cd "$root" && cargo build --quiet --release -p reims-vgpu-video-vulkan --example vulkan-h265-framemd5)
  build reims-vgpu-video-vaapi hevc-info hevc-cut hevc-craft
  {
    for size in 4 8 16 32; do
      n=$(( size == 4 ? 4 : 8 ))
      if (( size == 32 )); then
        kinds=(INTRA32X32_LUMA INTER32X32_LUMA)
      else
        kinds=()
        for mode in INTRA INTER; do
          for plane in LUMA CHROMAU CHROMAV; do kinds+=("${mode}${size}X${size}_$plane"); done
        done
      fi
      for kind in "${kinds[@]}"; do
        inter=0
        if [[ $kind == INTER* ]]; then
          inter=1
        fi
        echo "$kind ="
        for (( r = 0; r < n; r++ )); do
          row=""
          for (( c = 0; c < n; c++ )); do
            row+="$(( 6 + (r * 7 + c * 3 + ${#kind} * 5 + size + inter * 11) % 41 )),"
          done
          echo "${row%,}"
        done
        if (( size >= 16 )); then
          echo "${kind}_DC ="
          echo "$(( 9 + (${#kind} * 3 + size + inter * 5) % 23 ))"
        fi
      done
    done
  } > "$cqm"

  echo "vulkan h265-check streams, $(date -Is), $(hostname), $(x265 --version 2>&1 | grep -m1 -o 'version [^ ]*'), $(if (( have_kvazaar )); then "$kvazaar" --version 2>&1 | head -n1; else echo "no kvazaar"; fi)"
  # Each checked stream goes into `made` as name|bits, kept in a file for
  # the run under the locks.
  made=()
  for entry in "${streams[@]}"; do
    IFS='|' read -r name size frames bits kind opts <<<"$entry"
    if ! encode "$name" "$size" "$frames" "$bits" "$kind" "$opts"; then
      failed=1
      continue
    fi
    reference "$name" "$bits"
    cp "$out/$name.ref" "$out/$name.expect"
    if ! cmp -s "$out/$name.ref" "$out/$name.recon"; then
      echo "NOTE $name: ffmpeg's decode differs from x265's reconstruction from frame" \
        "$(paste "$out/$name.ref" "$out/$name.recon" | awk '$1 != $2 { print NR - 1; exit }'):" \
        "checking against the reconstruction"
      cp "$out/$name.recon" "$out/$name.expect"
    fi
    made+=("$name|$bits")
  done
  for entry in "${kvazaar_streams[@]}"; do
    IFS='|' read -r name size frames kind opts <<<"$entry"
    if ! encode_kvazaar "$name" "$size" "$frames" "$kind" "$opts"; then
      failed=1
      continue
    fi
    reference "$name" 8
    read -r hashed mismatched <<<"$(picture_hashes "$name")"
    pictures=$(wc -l < "$out/$name.ref")
    cp "$out/$name.ref" "$out/$name.expect"
    if [[ -s $out/$name.recon ]] && ! cmp -s "$out/$name.ref" "$out/$name.recon"; then
      echo "NOTE $name: ffmpeg's decode differs from kvazaar's reconstruction from frame" \
        "$(paste "$out/$name.ref" "$out/$name.recon" | awk '$1 != $2 { print NR - 1; exit }'):" \
        "checking against the reconstruction"
      cp "$out/$name.recon" "$out/$name.expect"
    elif [[ ! -s $out/$name.recon ]] && (( hashed < pictures || mismatched != 0 )); then
      echo "FAIL $name: no reference: $hashed of $pictures pictures' MD5s checked, $mismatched planes differ"
      failed=1
      continue
    fi
    echo "     $name: ffmpeg checked $hashed MD5s of its $pictures pictures, $mismatched planes differ"
    made+=("$name|8")
  done
  # The crafted streams, made again each run from their sources as they
  # are (the VA-API check's).
  for entry in "${crafted[@]}"; do
    IFS='|' read -r name source edits <<<"$entry"
    if [[ ! -s $out/$source.265 || ! -s $out/$source.expect ]]; then
      echo "FAIL $name: no $source to craft it from"
      failed=1
      continue
    fi
    bits=$(bits_of "$source")
    rm -f "$out/$name.265" "$out/$name.mp4" "$out/$name.ref" "$out/$name.expect" "$out/$name.craft"
    # shellcheck disable=SC2086 # $edits is a list of options
    if ! "$craft" "$out/$source.265" "$out/$name.part.265" $edits --check \
        > "$out/$name.check" 2> "$out/$name.craft.log"; then
      rm -f "$out/$name.part.265"
      echo "FAIL $name: hevc-craft: $(tail -n1 "$out/$name.craft.log")"
      failed=1
      continue
    fi
    mux "$name" hevc-craft || { failed=1; continue; }
    echo "     $name ($edits): $(sed 's/^wrote [^:]*: //' "$out/$name.craft.log")"
    reference "$name" "$bits"
    if ! cmp -s "$out/$source.ref" "$out/$name.ref"; then
      echo "FAIL $name: ffmpeg decodes it otherwise than $source, from frame" \
        "$(paste "$out/$source.ref" "$out/$name.ref" | awk '$1 != $2 { print NR - 1; exit }')"
      failed=1
      continue
    fi
    if command -v dec265 > /dev/null; then
      if ! de265 "$source" "$bits" || ! de265 "$name" "$bits"; then
        echo "FAIL $name: libde265 decodes no reference; see $out/$source.de265.log and $out/$name.de265.log"
        failed=1
        continue
      fi
      if ! cmp -s "$out/$source.de265" "$out/$name.de265"; then
        echo "FAIL $name: libde265 decodes it otherwise than $source, from frame" \
          "$(paste "$out/$source.de265" "$out/$name.de265" | awk '$1 != $2 { print NR - 1; exit }')"
        failed=1
        continue
      fi
    else
      echo "NOTE $name: no dec265 here; ffmpeg's decode is $source's, $(wc -l < "$out/$name.ref") frames"
    fi
    echo "$source $edits" > "$out/$name.craft"
    cp "$out/$source.expect" "$out/$name.expect"
    made+=("$name|$bits")
  done
  for entry in "${refusals[@]}"; do
    IFS='|' read -r name size frames bits csp opts <<<"$entry"
    encode "$name" "$size" "$frames" "$bits" testsrc2 "$opts" "$csp" || failed=1
  done
  for entry in "${kvazaar_refusals[@]}" "${kvazaar_partial[@]}"; do
    IFS='|' read -r name size frames kind opts _ _ <<<"$entry"
    if encode_kvazaar "$name" "$size" "$frames" "$kind" "$opts"; then
      reference "$name" 8
    else
      failed=1
    fi
  done
  # The seeks and splices: each open-GOP stream cut at its second CRA
  # picture, whose RASL pictures reference what came before the cut and
  # are not output (NoRaslOutputFlag, H.265 8.1.3), so that ffmpeg's decode
  # of the cut stream is the reference; and the same CRA picture made a BLA
  # one, mid-stream. Each goes into `seeks` as name|bits|cut sample.
  seeks=()
  for name in main-720p main10-720p kvz-1080p-dependent-cra; do
    [[ -s $out/$name.265 ]] || continue
    bits=$(bits_of "$name")
    at=$("$cut" list "$out/$name.265" | awk '$2 == "CRA" { n++; if (n == 2) { print $1; exit } }')
    if [[ -z $at ]]; then
      echo "FAIL $name: no second CRA picture to cut at"
      failed=1
      continue
    fi
    "$cut" from "$out/$name.265" "$out/$name-from-cra$at.265" "$at"
    reference "$name-from-cra$at" "$bits"
    "$cut" bla "$out/$name.265" "$out/$name-bla$at.265" "$at"
    reference "$name-bla$at" "$bits"
    seeks+=("$name|$bits|$at")
  done
  name=switch-main-main10
  if [[ -s $out/main-720p-closed.265 && -s $out/main10-1080p-slices-cqm.265 ]]; then
    cat "$out/main-720p-closed.265" "$out/main10-1080p-slices-cqm.265" > "$out/$name.265"
    cat "$out/main-720p-closed.expect" "$out/main10-1080p-slices-cqm.expect" > "$out/$name.expect"
  fi
  printf '%s\n' "${made[@]}" > "$runs/made"
  printf '%s\n' "${seeks[@]}" > "$runs/seeks"
  echo "$failed" > "$runs/made.failed"
  exec bash "$root/scripts/internal/gpu-locks.sh" --all -- bash "$0" "$@"
fi

failed=$(cat "$runs/made.failed")
mapfile -t made < "$runs/made"
mapfile -t seeks < "$runs/seeks"
echo "vulkan h265-check, $(date -Is), $(hostname)"
"$tool" --capabilities
for entry in "${made[@]}"; do
  [[ -n $entry ]] || continue
  IFS='|' read -r name bits <<<"$entry"
  echo "---- $name: $("$info" "$out/$name.265")"
  three "$name"
done

for entry in "${seeks[@]}"; do
  [[ -n $entry ]] || continue
  IFS='|' read -r name bits at <<<"$entry"
  cutname=$name-from-cra$at
  echo "---- $cutname: $("$info" "$out/$cutname.265")"
  if run "$out/$cutname.265" "$cutname.annexb"; then
    check "$cutname (Annex B, display order)" "$out/$cutname.ref" "$runs/$cutname.annexb.md5"
  fi
  if run "$out/$name.mp4" "$cutname.hvcc" --skip "$at"; then
    check "$cutname (hvcC from sample $at, display order)" "$out/$cutname.ref" "$runs/$cutname.hvcc.md5"
  fi
  blaname=$name-bla$at
  echo "---- $blaname: $("$info" "$out/$blaname.265")"
  if run "$out/$blaname.265" "$blaname.annexb"; then
    check "$blaname (Annex B, display order)" "$out/$blaname.ref" "$runs/$blaname.annexb.md5"
  fi
done

# The caller-facing options, on one stream each.
if run "$out/main-1080p-slices.mp4" main-1080p-slices.len2 --length-size 2; then
  check "main-1080p-slices (2-byte NAL lengths)" "$out/main-1080p-slices.expect" "$runs/main-1080p-slices.len2.md5"
fi
if run "$out/main10-720p.mp4" main10-720p.len3 --length-size 3; then
  check "main10-720p (3-byte NAL lengths)" "$out/main10-720p.expect" "$runs/main10-720p.len3.md5"
fi
if run "$out/main-720p.mp4" main-720p.hold --hold 24; then
  check "main-720p (24 frames held by the caller)" "$out/main-720p.expect" "$runs/main-720p.hold.md5"
fi

# Pictures copied out of the DPB on a queue of their own, behind a
# semaphore wait, as on a device whose decode family does not copy.
for name in main-720p-deep main10-1080p-slices-cqm; do
  if run "$out/$name.mp4" "$name.graphics" --copy-on-graphics; then
    check "$name (copied out on the graphics queue)" "$out/$name.expect" "$runs/$name.graphics.md5"
  fi
done
# Pictures landed in buffers, cropped, at offsets and pitches of their own:
# 8 and 10 bits, cropped frames, an open GOP, tiles.
for name in main-1366x766-ctu16 main10-1366x766-ctu16 main-720p main10-4k kvz-720p-tiles-explicit; do
  [[ -s $out/$name.mp4 ]] || continue
  if run "$out/$name.mp4" "$name.buffer" --buffer; then
    check "$name (landed in buffers)" "$out/$name.expect" "$runs/$name.buffer.md5"
  fi
done
if run "$out/main10-720p-fade.mp4" main10-720p-fade.buffer-graphics --buffer --copy-on-graphics; then
  check "main10-720p-fade (landed in buffers on the graphics queue)" "$out/main10-720p-fade.expect" "$runs/main10-720p-fade.buffer-graphics.md5"
fi
# Access units read in place, as the guest's video driver lays them out,
# into images and buffers: every picture must be read in place.
for entry in "main-1080p-slices|" "main10-720p|" "main-4k|--buffer" "kvz-1080p-dependent-cra|" "main-720p-ref1-all|--buffer" "kvz-1080p-tiles-slices|"; do
  IFS='|' read -r name opts <<<"$entry"
  [[ -s $out/$name.265 ]] || continue
  label=$name.in-place${opts:+${opts// /}}
  # shellcheck disable=SC2086 # $opts is a list of options
  if run "$out/$name.265" "$label" --in-place $opts; then
    check "$name (read in place${opts:+, $opts})" "$out/$name.expect" "$runs/$label.md5"
    all_in_place "$label"
  fi
done

# A new SPS at an IDR picture, from Main at 720p to Main 10 at 1080p,
# mid-stream: the decoder outputs what it holds, makes a new session in
# the new format and goes on.
name=switch-main-main10
if [[ -s $out/$name.265 ]] && run "$out/$name.265" "$name.annexb"; then
  check "$name (Annex B, display order)" "$out/$name.expect" "$runs/$name.annexb.md5"
fi

# What the decoder refuses before any slice reaches the GPU.
for entry in "${refusals[@]}"; do
  IFS='|' read -r name _ <<<"$entry"
  [[ -s $out/$name.mp4 ]] || continue
  refused "$name" "$out/$name.mp4" '^vulkan-h265-framemd5: unsupported: profiles outside Main'
done
for entry in "${kvazaar_refusals[@]}"; do
  IFS='|' read -r name _ _ _ _ why <<<"$entry"
  [[ -s $out/$name.mp4 ]] || continue
  for input in "$out/$name.mp4" "$out/$name.265"; do
    refused "$name.${input##*.}" "$input" "$why"
  done
done
for entry in "${kvazaar_partial[@]}"; do
  IFS='|' read -r name _ _ _ _ good why <<<"$entry"
  [[ -s $out/$name.mp4 ]] || continue
  echo "---- $name: $("$info" "$out/$name.265")"
  head -n "$good" "$out/$name.ref" > "$runs/$name.expect"
  for input in "$out/$name.mp4" "$out/$name.265"; do
    label=$name.${input##*.}
    if "$tool" "$input" > "$runs/$label.frames" 2> "$runs/$label.log"; then
      echo "FAIL $label: decoded whole, should be refused at frame $good"
      failed=1
    elif ! grep -q "$why" "$runs/$label.log"; then
      echo "FAIL $label: $(tail -n1 "$runs/$label.log")"
      failed=1
    else
      awk '{ print $NF }' "$runs/$label.frames" > "$runs/$label.md5"
      check "$name (${input##*.}, the $good frames before the refusal)" "$runs/$name.expect" "$runs/$label.md5"
      echo "     $(tail -n1 "$runs/$label.log")"
    fi
  done
done

# Under the validation layer: the frames as before, and nothing from it
# in its log. The layer says on stdout, once, where it logs, which shows it
# ran; any other line there that is not a frame is a FAIL. On NVIDIA's
# 615.71.09 the layer's handle wrapping crashes the driver's vkQueueSubmit2
# of a video decode, so it runs without it.
if [[ -n $(find /usr/share/vulkan /etc/vulkan "$HOME/.local/share/vulkan" -name '*validation*.json' 2> /dev/null) ]]; then
  for entry in "${validated[@]}"; do
    IFS='|' read -r name opts <<<"$entry"
    [[ -s $out/$name.mp4 ]] || continue
    input=$out/$name.mp4
    if [[ $opts == *--in-place* ]]; then
      input=$out/$name.265
    fi
    label=$name.validated${opts:+${opts// /}}
    vvl=$runs/$label.vvl
    rm -f "$vvl"
    # shellcheck disable=SC2086 # $opts is a list of options
    if VK_LOADER_LAYERS_ENABLE='*validation' VK_KHRONOS_VALIDATION_UNIQUE_HANDLES=false \
        VK_KHRONOS_VALIDATION_LOG_FILENAME="$vvl" \
        run "$input" "$label" $opts; then
      ran="Validation Layer Info - Logging validation error to $vvl"
      grep '^frame ' "$runs/$label.frames" | awk '{ print $NF }' > "$runs/$label.md5"
      check "$name (validation layer${opts:+, $opts})" "$out/$name.expect" "$runs/$label.md5"
      other=$(grep -v -e '^frame ' -e "^$ran\$" "$runs/$label.frames" || true)
      if ! grep -qx -- "$ran" "$runs/$label.frames"; then
        echo "FAIL $label: the validation layer did not run"
        failed=1
      elif [[ -s $vvl || -n $other ]]; then
        echo "FAIL $label: the validation layer says: $(head -c 400 "$vvl" 2> /dev/null; head -n1 <<<"$other")"
        failed=1
      else
        echo "PASS $label: the validation layer says nothing"
      fi
    fi
  done
else
  echo "FAIL validation: no Khronos validation layer installed"
  failed=1
fi

exit "$failed"
