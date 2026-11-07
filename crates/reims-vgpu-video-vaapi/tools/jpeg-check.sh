#!/usr/bin/env bash
# The JPEG engine's VA-API decoder against libjpeg and ffmpeg.
#
#   crates/reims-vgpu-video-vaapi/tools/jpeg-check.sh [--large] [NODE]...
#
# Makes its JPEGs with cjpeg and jpegtran from ffmpeg's testsrc2, cropped to
# each size. Each one that should decode gets two references: libjpeg's
# raw output with its islow IDCT (jpeg-planes.c), and ffmpeg's mjpeg
# decode in its own planar format. Then the conformance suite's jpeg-check
# (vaapi-conformance NODE jpeg-check) decodes each on every render node (every /dev/dri/renderD* when none is named) through
# the backend's JPEG decoder, and compares. A case passes when every plane
# is within the tolerance of both references (RVGV_JPEG_TOLERANCE, 1 by
# default), or when it is refused with the error it should be, or when its
# stream is outside what the backend answers it decodes (its samplings and
# largest picture) and refused as unsupported. The last line counts each.
# A host with no render node fails.
#
# --large adds images of 8192x8192 (4:2:0 and 4:2:2) and 16384x16000 (several
# GB of scratch).
#
# Everything goes to build/jpeg-check/ (RVGV_JPEG_OUT): img/, ref/,
# runs/<node>/<case>.txt, and summary.txt, one line a case and node, with
# the digest of what each decode wrote, so that two runs' summaries show
# whether they wrote the same bytes. Nothing it writes is committed.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

out=${RVGV_JPEG_OUT:-$root/build/jpeg-check}
tolerance=${RVGV_JPEG_TOLERANCE:-1}

large=0
nodes=()
for arg in "$@"; do
  case $arg in
    --large) large=1 ;;
    *) nodes+=("$arg") ;;
  esac
done
if [[ ${#nodes[@]} -eq 0 ]]; then
  shopt -s nullglob
  nodes=(/dev/dri/renderD*)
  shopt -u nullglob
fi
if [[ ${#nodes[@]} -eq 0 ]]; then
  echo "jpeg-check: no render node" >&2
  exit 1
fi

mkdir -p "$out/img" "$out/ref" "$out/src"
cc -O2 -Wall -Wextra -o "$out/jpeg-planes" "$root/crates/reims-vgpu-video-vaapi/tools/jpeg-planes.c" -ljpeg
# RVGV_JPEG_CHECK names another backend's conformance example (such as
# nvidia-conformance), whose jpeg-check runs instead; VA-API's is built
# otherwise.
if [[ -n ${RVGV_JPEG_CHECK:-} ]]; then
  check=$RVGV_JPEG_CHECK
else
  build reims-vgpu-video-vaapi vaapi-conformance
  check=$CARGO_TARGET_DIR/release/examples/vaapi-conformance
fi

# A testsrc2 frame of exactly WxH. testsrc2 rounds an odd size down, and
# crop rounds to its 4:2:0 frame's even sizes, so the frame is made even and
# at least 64x64, then turned to RGB and cropped.
source_ppm() {
  local size=$1 w h
  w=${size%x*} h=${size#*x}
  local file=$out/src/$size.ppm
  if [[ ! -e $file ]]; then
    local fw=$(((w + 1) / 2 * 2)) fh=$(((h + 1) / 2 * 2))
    ffmpeg -nostdin -loglevel error -y -f lavfi -i "testsrc2=size=$((fw < 64 ? 64 : fw))x$((fh < 64 ? 64 : fh))" \
      -frames:v 1 -vf "format=rgb24,crop=$w:$h:0:0" "$file"
  fi
  echo "$file"
}

# The ffmpeg reference's planar format for a stream: its decoder's own.
ffmpeg_format() {
  ffprobe -loglevel error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "$1"
}

cases=()
# case NAME SIZE "CJPEG OPTIONS" ["CHECK OPTIONS"]: a stream cjpeg makes.
case_cjpeg() {
  local name=$1 size=$2 options=$3 extra=${4:-}
  # shellcheck disable=SC2086 # the options are words
  cjpeg $options -outfile "$out/img/$name.jpg" "$(source_ppm "$size")"
  cases+=("$name|$extra")
}

references() {
  local name=$1
  local jpg=$out/img/$name.jpg
  "$out/jpeg-planes" islow "$jpg" "$out/ref/$name.libjpeg.yuv"
  local format
  format=$(ffmpeg_format "$jpg")
  ffmpeg -nostdin -loglevel error -y -i "$jpg" -f rawvideo -pix_fmt "$format" "$out/ref/$name.ffmpeg.yuv"
}

# Qualities: -baseline keeps the tables 8-bit, as VA-API needs.
for q in 10 25 50 75 90 100; do
  case_cjpeg "420-q$q" 640x480 "-baseline -quality $q -sample 2x2"
done
# Each subsampling.
case_cjpeg 422-q90 640x480 "-quality 90 -sample 2x1"
case_cjpeg 444-q90 640x480 "-quality 90 -sample 1x1"
case_cjpeg grey-q90 640x480 "-quality 90 -grayscale"
case_cjpeg 411-q90 640x480 "-quality 90 -sample 4x1" "--expect-error UnsupportedStream"
case_cjpeg 440-q90 640x480 "-quality 90 -sample 1x2" "--expect-error SubsamplingMismatch"
# Restart intervals: every MCU, every third, every MCU row.
for subsampling in 2x2 2x1 1x1; do
  case_cjpeg "restart-1mcu-$subsampling" 333x211 "-quality 80 -sample $subsampling -restart 1B"
  case_cjpeg "restart-3mcu-$subsampling" 333x211 "-quality 80 -sample $subsampling -restart 3B"
  case_cjpeg "restart-1row-$subsampling" 333x211 "-quality 80 -sample $subsampling -restart 1"
done
case_cjpeg restart-2mcu-grey 333x211 "-quality 80 -grayscale -restart 2B"
# Odd and small sizes, below the driver's 64x64 surfaces among them.
case_cjpeg odd-333x211-420 333x211 "-quality 85 -sample 2x2"
case_cjpeg odd-17x9-420 17x9 "-quality 85 -sample 2x2"
case_cjpeg odd-1x1-420 1x1 "-quality 85 -sample 2x2"
case_cjpeg odd-1x1-grey 1x1 "-quality 85 -grayscale"
case_cjpeg odd-65x33-422 65x33 "-quality 85 -sample 2x1"
case_cjpeg odd-15x15-444 15x15 "-quality 85 -sample 1x1"
case_cjpeg odd-4095x7-420 4095x7 "-quality 85 -sample 2x2"
case_cjpeg odd-7x4095-grey 7x4095 "-quality 85 -grayscale"
case_cjpeg big-1920x1080-420 1920x1080 "-quality 90 -sample 2x2" "--repeat 20"
case_cjpeg big-4000x3000-420 4000x3000 "-quality 90 -sample 2x2" "--repeat 10"
case_cjpeg big-4000x3000-444 4000x3000 "-quality 90 -sample 1x1" "--repeat 10"
# Huffman tables: the stream's own, and none (Motion JPEG's), whose
# defaults must be cjpeg's Annex K tables.
case_cjpeg optimized-420 640x480 "-quality 90 -sample 2x2 -optimize"
case_cjpeg annexk-420 640x480 "-quality 90 -sample 2x2" "--annex-k"
case_cjpeg nodht-420 640x480 "-quality 90 -sample 2x2" "--strip-dht"
case_cjpeg nodht-422 333x211 "-quality 90 -sample 2x1 -restart 2B" "--strip-dht"
case_cjpeg nodht-grey 333x211 "-quality 90 -grayscale" "--strip-dht"
# Crops, as CMPhoto's CropRect gives them in MCUs, at the edge among them.
case_cjpeg crop-420-middle 640x480 "-quality 90 -sample 2x2" "--crop 1,1,10,10"
case_cjpeg crop-420-edge 333x211 "-quality 90 -sample 2x2" "--crop 20,13,1,1"
case_cjpeg crop-422-row 333x211 "-quality 90 -sample 2x1" "--crop 0,5,21,3"
case_cjpeg crop-444-corner 15x15 "-quality 90 -sample 1x1" "--crop 1,1,1,1"
case_cjpeg crop-grey-edge 333x211 "-quality 90 -grayscale" "--crop 41,26,1,1"
# The other layouts.
case_cjpeg layout-422-yuvs 333x211 "-quality 90 -sample 2x1" "--format yuvs"
case_cjpeg layout-422-2vuy 333x211 "-quality 90 -sample 2x1" "--format 2vuy"
case_cjpeg layout-grey-422f 333x211 "-quality 90 -grayscale" "--format 422f"
case_cjpeg layout-grey-444f 333x211 "-quality 90 -grayscale" "--format 444f"
case_cjpeg layout-grey-yuvs 333x211 "-quality 90 -grayscale" "--format yuvs"
case_cjpeg layout-grey-2vuy 333x211 "-quality 90 -grayscale" "--format 2vuy"
# Layouts of another sampling, and BGRA, through the driver's video
# processing: compared with core's scaler reference, by PSNR.
for format in 422f yuvs 2vuy BGRA; do
  case_cjpeg "convert-420-$format" 640x480 "-quality 90 -sample 2x2" "--format $format"
done
for format in 420f BGRA; do
  case_cjpeg "convert-422-$format" 640x480 "-quality 90 -sample 2x1" "--format $format"
done
for format in 420f 422f yuvs BGRA; do
  case_cjpeg "convert-444-$format" 640x480 "-quality 90 -sample 1x1" "--format $format"
done
case_cjpeg convert-grey-BGRA 640x480 "-quality 90 -grayscale" "--format BGRA"
case_cjpeg convert-odd-420-2vuy 333x211 "-quality 90 -sample 2x2" "--format 2vuy"
case_cjpeg convert-odd-444-BGRA 333x211 "-quality 90 -sample 1x1" "--format BGRA"
# 444f from 4:2:0 and 4:2:2, which Apple's hardware refuses too.
case_cjpeg refuse-420-444f 640x480 "-quality 90 -sample 2x2" "--format 444f --expect-error UnsupportedFormat"
case_cjpeg refuse-422-444f 640x480 "-quality 90 -sample 2x1" "--format 444f --expect-error UnsupportedFormat"
# Scales of 1/2, 1/4 and 1/8, as the options' DecodeWidth and DecodeHeight
# ask for them, into each kind of layout.
for scale in 2 4 8; do
  case_cjpeg "scale-420-$scale" 640x480 "-quality 90 -sample 2x2" "--scale $scale"
done
case_cjpeg scale-420-2-BGRA 640x480 "-quality 90 -sample 2x2" "--scale 2 --format BGRA"
case_cjpeg scale-420-2-yuvs 640x480 "-quality 90 -sample 2x2" "--scale 2 --format yuvs"
case_cjpeg scale-422-2 640x480 "-quality 90 -sample 2x1" "--scale 2"
case_cjpeg scale-422-4-420f 640x480 "-quality 90 -sample 2x1" "--scale 4 --format 420f"
case_cjpeg scale-422-4-BGRA 640x480 "-quality 90 -sample 2x1" "--scale 4 --format BGRA"
case_cjpeg scale-444-2 640x480 "-quality 90 -sample 1x1" "--scale 2"
case_cjpeg scale-444-8-420f 640x480 "-quality 90 -sample 1x1" "--scale 8 --format 420f"
case_cjpeg scale-444-8-BGRA 640x480 "-quality 90 -sample 1x1" "--scale 8 --format BGRA"
case_cjpeg scale-444-4-2vuy 640x480 "-quality 90 -sample 1x1" "--scale 4 --format 2vuy"
case_cjpeg scale-grey-2 640x480 "-quality 90 -grayscale" "--scale 2"
case_cjpeg scale-grey-4-444f 640x480 "-quality 90 -grayscale" "--scale 4 --format 444f"
case_cjpeg scale-grey-8-BGRA 640x480 "-quality 90 -grayscale" "--scale 8 --format BGRA"
case_cjpeg scale-crop-420 640x480 "-quality 90 -sample 2x2" "--crop 1,1,10,10 --scale 2"
case_cjpeg scale-crop-420-BGRA 640x480 "-quality 90 -sample 2x2" "--crop 3,2,20,16 --scale 4 --format BGRA"
case_cjpeg scale-crop-444-edge 333x211 "-quality 90 -sample 1x1" "--crop 30,20,20,20 --scale 2"
case_cjpeg scale-odd-420 333x211 "-quality 90 -sample 2x2" "--scale 2"
case_cjpeg scale-odd-420-2vuy 333x211 "-quality 90 -sample 2x2" "--scale 2 --format 2vuy"
case_cjpeg scale-big-4000x3000-420 4000x3000 "-quality 90 -sample 2x2" "--scale 8 --repeat 10"
# What a baseline decoder does not take.
case_cjpeg refuse-progressive 640x480 "-quality 90 -progressive" "--expect-error UnsupportedStream"
case_cjpeg refuse-arithmetic 640x480 "-quality 90 -arithmetic" "--expect-error UnsupportedStream"
case_cjpeg refuse-12bit 640x480 "-quality 90 -precision 12" "--expect-error UnsupportedStream"
case_cjpeg refuse-lossless 640x480 "-lossless 1" "--expect-error UnsupportedStream"
case_cjpeg refuse-rgb 640x480 "-quality 90 -rgb" "--expect-error UnsupportedStream"
case_cjpeg refuse-16bit-tables 640x480 "-quality 5 -sample 2x2" "--expect-error UnsupportedStream"
case_cjpeg refuse-over-max 16400x64 "-quality 50 -sample 2x2" "--expect-error UnsupportedStream"
# Sequential, but one scan a component.
printf '0;\n1;\n2;\n' > "$out/src/scans.txt"
jpegtran -scans "$out/src/scans.txt" -outfile "$out/img/refuse-three-scans.jpg" "$out/img/420-q90.jpg"
cases+=("refuse-three-scans|--expect-error UnsupportedStream")
# Cut short: the EOI is missing.
head -c 20000 "$out/img/420-q90.jpg" > "$out/img/refuse-no-eoi.jpg"
cases+=("refuse-no-eoi|--expect-error BadData")
# The driver's largest width and height, a strip each.
case_cjpeg max-16384x64-422 16384x64 "-quality 75 -sample 2x1"
case_cjpeg max-64x16384-444 64x16384 "-quality 75 -sample 1x1"
if [[ $large -eq 1 ]]; then
  # ffmpeg refuses a 16384x16384 frame, and a 16384x16382 one, but makes
  # and decodes 16384x16000 (ffmpeg n9.0.2, 2026-10-24).
  case_cjpeg large-8192x8192-420 8192x8192 "-quality 90 -sample 2x2"
  case_cjpeg large-8192x8192-422 8192x8192 "-quality 90 -sample 2x1"
  case_cjpeg large-16384x16000-420 16384x16000 "-quality 75 -sample 2x2"
  case_cjpeg large-16384x16000-grey 16384x16000 "-quality 75 -grayscale"
fi
# Bytes of the scan's data overwritten, away from any marker: the stream
# still parses, and the walk of its scan refuses it before the driver,
# which decodes such data without an error.
for offset in 5000 15000 29000; do
  name=refuse-damaged-at-$offset
  cp "$out/img/420-q90.jpg" "$out/img/$name.jpg"
  printf '\x5a\x5a\x5a\x5a\x5a\x5a\x5a\x5a' |
    dd of="$out/img/$name.jpg" bs=1 seek=$offset conv=notrunc status=none
  cases+=("$name|--expect-error BadData")
done
# Cut short and closed with an EOI: the scan ends before its last MCU.
{ head -c 20000 "$out/img/420-q90.jpg"; printf '\xff\xd9'; } > "$out/img/refuse-cut-with-eoi.jpg"
cases+=("refuse-cut-with-eoi|--expect-error ScanEndsEarly")

locked=()
for node in "${nodes[@]}"; do locked+=(--node "$node"); done
lock_gpu "${locked[@]}"

summary=$out/summary.txt
: > "$summary"
for entry in "${cases[@]}"; do
  name=${entry%%|*}
  extra=${entry#*|}
  jpg=$out/img/$name.jpg
  refs=()
  if [[ $extra != *--expect-error* ]]; then
    references "$name"
    refs=(--reference "libjpeg=$out/ref/$name.libjpeg.yuv" --reference "ffmpeg=$out/ref/$name.ffmpeg.yuv")
  fi
  for node in "${nodes[@]}"; do
    dir=$out/runs/$(basename "$node")
    mkdir -p "$dir"
    log=$dir/$name.txt
    # shellcheck disable=SC2086 # the extra options are words
    if timeout 120 "$check" "$node" jpeg-check --tolerance "$tolerance" ${refs[@]+"${refs[@]}"} $extra "$jpg" \
        > "$log" 2>&1; then
      verdict=pass
    else
      verdict=FAIL
      failed=1
    fi
    line="$(basename "$node") $name: $verdict; $(grep -E '^(decode|digest|compare)' "$log" | tr '\n' ';' || true)"
    echo "$line" | tee -a "$summary"
  done
done
echo "jpeg-check: $(wc -l <"$summary") runs, $(grep -c ': pass;' "$summary") pass ($(grep -c 'decode refused as answered' "$summary") refused as the backend answers), $(grep -c ': FAIL;' "$summary") FAIL"
exit "$failed"
