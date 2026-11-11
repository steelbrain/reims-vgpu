#!/usr/bin/env bash
# The VA-API HEVC decoder against a software reference.
#
#   crates/reims-vgpu-video/tools/hevc-check.sh [render node]
#
# Runs on Linux with libva, x265, kvazaar and ffmpeg. x265,
# kvazaar and ffmpeg are test oracles only: x265 and kvazaar write the test
# streams and ffmpeg decodes them in software (-hwaccel none) for the
# reference MD5 of each frame. None is linked into anything, and nothing
# they write is committed: it all stays under build/. kvazaar writes what
# x265 does not: tiles, dependent slice segments, field pictures and
# short-term sets in every slice header.
#
# Each stream is decoded by examples/hevc-framemd5 from its MP4 (the hvcC
# record and length-prefixed samples, as VideoToolbox hands them over) and
# from its Annex B form, in display order; every frame's MD5, over NV12
# for Main and P010 for Main 10, must equal ffmpeg's. Decode order must
# give the same frames. Streams the decoder must refuse (4:2:2, 4:4:4,
# 12-bit) must fail with "unsupported".
#
# x265's own reconstruction (--recon) is a second reference. ffmpeg n9.0.2's
# software decode differs from it on some CTB-16 streams with SAO and
# deblocking on, where the RX 7900 XTX, the Radeon 780M and ffmpeg's VA-API
# decode agree with x265. Where the two references differ, the script says so and checks
# against the reconstruction. kvazaar's reconstruction (--debug) is the
# second reference for its streams, but for field pictures, whose --debug
# output is not the decoded fields (2026-10-24); each kvazaar stream also
# carries an MD5 of every decoded picture (--hash md5), which ffmpeg checks
# its decode against (-err_detect crccheck).
#
# What neither encoder writes, examples/hevc-craft writes into copies of
# their streams, every picture decoding as the source's: reference list
# modification, short-term sets predicted from others in the SPS and in
# slice headers, and long-term pictures. hevc-craft's --check decodes each
# through the codec crate's stand-in first, against its source; ffmpeg's
# and libde265's decodes (dec265, where it is installed) of it must then
# equal their decodes of the source, and it is checked as any stream is,
# against its source's reference.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

node=${1:-/dev/dri/renderD128}
out=${RVGV_HEVC_OUT:-$root/build/hevc-check}
mkdir -p "$out"

build reims-vgpu-video hevc-framemd5 hevc-info hevc-cut hevc-craft
tool=$CARGO_TARGET_DIR/release/examples/hevc-framemd5
info=$CARGO_TARGET_DIR/release/examples/hevc-info
cut=$CARGO_TARGET_DIR/release/examples/hevc-cut
craft=$CARGO_TARGET_DIR/release/examples/hevc-craft

# Asymmetric scaling lists in HM's format, so that a list in the wrong
# scan order, transposed, in the wrong slot or with the wrong DC changes
# the picture. Each chroma V list equals its U list, which x265 codes as a
# prediction from it; the two 32x32 lists differ, since x265 4.3 writes a
# pred_matrix_id_delta ffmpeg rejects when they are equal (2026-10-24).
cqm=$out/asymmetric.cqm
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
    for name in "${kinds[@]}"; do
      inter=0
      if [[ $name == INTER* ]]; then
        inter=1
      fi
      echo "$name ="
      for (( r = 0; r < n; r++ )); do
        row=""
        for (( c = 0; c < n; c++ )); do
          row+="$(( 6 + (r * 7 + c * 3 + ${#name} * 5 + size + inter * 11) % 41 )),"
        done
        echo "${row%,}"
      done
      if (( size >= 16 )); then
        echo "${name}_DC ="
        echo "$(( 9 + (${#name} * 3 + size + inter * 5) % 23 ))"
      fi
    done
  done
} > "$cqm"

noise="noise=alls=14:allf=t+u"
# name|size|frames|bits|source filter|x265 options
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
  # One reference, no B-frames and no temporal motion vector prediction,
  # for hevc-craft's long-term pictures: each P picture predicts from one
  # picture alone, which may then be long-term with no change to the
  # picture.
  "main-720p-ref1|1280x720|60|8|testsrc2|--preset fast --bframes 0 --ref 1 --no-temporal-mvp --keyint 300"
)

# kvazaar 2.3.2's streams, 8 bits only as CachyOS builds it, for what x265
# does not write: name|size|frames|source filter|kvazaar options, after
# --preset, whose GOP and references the options then replace. Every one
# also gets --hash md5. A preset's options come first, so that the ones
# after it win.
kvazaar_streams=(
  "kvz-720p-tiles|1280x720|60|testsrc2|--preset medium --gop 8 --level 4.1 --tiles 3x3"
  "kvz-720p-tiles-explicit|1280x720|60|testsrc2|--preset fast --gop 8 --level 4.1 --tiles-width-split 256,640,960 --tiles-height-split 128,512"
  # Under the Main profiles' bounds (H.265 A.3.2 and A.3.3), which the
  # codec crate takes but for columns under 128 (MIN_TILE_COLUMN_WIDTH):
  # a column of 192 samples, then rows of 704 and 16.
  "kvz-720p-tiles-narrow|1280x720|10|testsrc2|--preset fast --gop 8 --level 4.1 --tiles-width-split 256,448,960"
  "kvz-720p-tiles-short|1280x720|10|testsrc2|--preset fast --gop 8 --level 4.1 --tiles-height-split 704"
  "kvz-1080p-tiles-slices|1920x1080|30|testsrc2|--preset fast --gop 8 --level 5.1 --tiles 2x2 --slices tiles"
  "kvz-1366x766-tiles|1366x766|30|testsrc2|--preset fast --gop 8 --level 4.1 --tiles 2x3 --slices tiles"
  "kvz-4k-tiles-slices|3840x2160|20|mandelbrot|--preset ultrafast --gop 8 --level 5.1 --tiles 4x4 --slices tiles"
  "kvz-720p-dependent|1280x720|60|testsrc2|--preset fast --gop 8 --level 4.1 --wpp --slices wpp"
  "kvz-1080p-dependent-cra|1920x1080|70|testsrc2|--preset fast --gop 16 --period 32 --open-gop --level 5.1 --wpp --slices wpp"
  "kvz-720p-lowdelay|1280x720|60|testsrc2|--preset fast --gop lp-g8d4t2 --ref 4 --level 4.1"
  "kvz-720p-ref8|1280x720|30|testsrc2|--preset fast --gop 0 --ref 8 --level 4.1"
  "kvz-720p-fields|1280x720|30|testsrc2|--preset fast --gop lp-g4d3t1 --level 4.1 --source-scan-type tff"
  "kvz-720p-tools|1280x720|30|fade|--preset fast --gop 8 --level 4.1 --signhide --transform-skip --amp --smp --vaq 5 --deblock -3:2 --sao band --aud"
  # main-720p-ref1's, for long-term pictures.
  "kvz-1080p-ref1|1920x1080|30|testsrc2|--preset fast --gop 0 --ref 1 --no-tmvp --level 4.1"
)

# hevc-craft's streams: name|source|its edits. x265's and kvazaar's, at
# 720p and 1080p, with B-frames and open GOPs and without, Main and Main 10;
# with list modification alone, the SPS's sets alone and the headers' own
# sets alone, then everything, and the one-reference streams' long-term
# pictures.
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

# x265's stream NAME, its reconstruction's MD5s and its MP4; 1 where one is
# not made, which leaves no stream a later run would take. A stream made
# again drops the reference of the one before.
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
# whose --debug output is not the decoded fields) and its MP4.
encode_kvazaar() {
  local name=$1 size=$2 frames=$3 kind=$4 opts=$5
  if [[ -s $out/$name.mp4 && -s $out/$name.265 ]]; then
    return
  fi
  rm -f "$out/$name.mp4" "$out/$name.265" "$out/$name.ref"
  local recon=(--debug "$out/$name.recon.yuv")
  if [[ $opts == *--source-scan-type* ]]; then
    recon=()
  fi
  ffmpeg -nostdin -hide_banner -loglevel error -y -f lavfi -i "$(source_filter "$kind" "$size")" \
      -frames:v "$frames" -pix_fmt yuv420p -f yuv4mpegpipe "$out/$name.src.y4m" || return 1
  # shellcheck disable=SC2086 # $opts is a list of options
  if ! nice -n 10 kvazaar -i "$out/$name.src.y4m" --input-fps 30 $opts --hash md5 \
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
# planes of them it found different: kvazaar's own reconstruction against
# ffmpeg's decode, picture by picture. ffmpeg decodes the first picture
# twice, once as it probes the stream, so it checks one more than there
# are.
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
# P010, into NAME.de265; 1 where it fails, its reason in NAME.de265.log.
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
  local entry
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

# What the decoder refuses, cleanly, before any slice reaches the GPU:
# x265's name|size|frames|bits|chroma|options, and kvazaar's
# name|size|frames|source filter|options|what the refusal says. kvazaar
# takes tile columns of 64 samples, one CTB, which radeonsi decodes wrongly
# with no error, and the codec crate refuses columns under 128.
refusals=(
  "main422-10|640x360|10|10|422|--preset fast --profile main422-10"
  "main444-8|640x360|10|8|444|--preset fast --profile main444-8"
  "main12|640x360|10|12|420|--preset fast --profile main12"
)
kvazaar_refusals=(
  "kvz-720p-tiles-col64|1280x720|10|testsrc2|--preset fast --gop 8 --level 5.1 --tiles-width-split 64,128,1216|ColumnWidthInLumaSamples is out of range"
)
# A stream whose pictures name more than 8 current references from its
# tenth on, where va.h says the sets hold 8 at most and radeonsi decodes
# such a picture wrongly: name|size|frames|source filter|options|the
# frames before it|what the refusal says. The frames before it must match,
# and it must be refused before it reaches the GPU.
kvazaar_partial=(
  "kvz-720p-ref15|1280x720|30|testsrc2|--preset fast --gop 0 --ref 15 --level 4.1|9|VA-API takes 8"
)

echo "hevc-check on $node, $(date -Is), $(hostname), $(x265 --version 2>&1 | grep -m1 -o 'version [^ ]*'), $(kvazaar --version 2>&1 | head -n1)"

# Every stream and its reference first, then the GPU. Each checked stream
# goes into `made` as name|bits.
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
# The crafted streams, made again each run from their sources as they are.
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
  echo "     $name: $(cat "$out/$name.check")"
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
    if ! cmp -s "$out/$source.de265" "$out/$source.expect"; then
      echo "NOTE $name: libde265's decode of $source differs from its reference, from frame" \
        "$(paste "$out/$source.de265" "$out/$source.expect" | awk '$1 != $2 { print NR - 1; exit }')"
    fi
    echo "     $name: ffmpeg's and libde265's decodes are $source's, $(wc -l < "$out/$name.ref") frames"
  else
    echo "NOTE $name: no dec265 here; ffmpeg's decode is $source's, $(wc -l < "$out/$name.ref") frames"
  fi
  # What the VideoToolbox check crafts again, to send this very stream
  # only once the codec crate's --check has passed on it there.
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

for entry in "${made[@]}"; do
  IFS='|' read -r name bits <<<"$entry"
  echo "---- $name: $("$info" "$out/$name.265")"
  if run "$out/$name.mp4" "$name.hvcc"; then
    check "$name (hvcC, display order)" "$out/$name.expect" "$out/$name.hvcc.md5"
  fi
  if run "$out/$name.265" "$name.annexb"; then
    check "$name (Annex B, display order)" "$out/$name.expect" "$out/$name.annexb.md5"
  fi
  if run "$out/$name.mp4" "$name.decode" --decode-order; then
    sort "$out/$name.expect" > "$out/$name.expect.sorted"
    sort "$out/$name.decode.md5" > "$out/$name.decode.sorted"
    check "$name (hvcC, decode order, as a set)" "$out/$name.expect.sorted" "$out/$name.decode.sorted"
  fi
done

# A seek: each open-GOP stream cut at its second CRA picture, whose RASL
# pictures reference what came before the cut and are not output
# (NoRaslOutputFlag, H.265 8.1.3). ffmpeg drops them too, so its decode of
# the cut stream is the reference. The MP4 is the same cut, by --skip.
for name in main-720p main10-720p kvz-1080p-dependent-cra; do
  bits=$(bits_of "$name")
  at=$("$cut" list "$out/$name.265" | awk '$2 == "CRA" { n++; if (n == 2) { print $1; exit } }')
  if [[ -z $at ]]; then
    echo "FAIL $name: no second CRA picture to cut at"
    failed=1
    continue
  fi
  cutname=$name-from-cra$at
  "$cut" from "$out/$name.265" "$out/$cutname.265" "$at"
  reference "$cutname" "$bits"
  echo "---- $cutname: $("$info" "$out/$cutname.265")"
  if run "$out/$cutname.265" "$cutname.annexb"; then
    check "$cutname (Annex B, display order)" "$out/$cutname.ref" "$out/$cutname.annexb.md5"
  fi
  if run "$out/$name.mp4" "$cutname.hvcc" --skip "$at"; then
    check "$cutname (hvcC from sample $at, display order)" "$out/$cutname.ref" "$out/$cutname.hvcc.md5"
  fi
  # A splice: the same CRA picture made a BLA one, mid-stream.
  blaname=$name-bla$at
  "$cut" bla "$out/$name.265" "$out/$blaname.265" "$at"
  reference "$blaname" "$bits"
  echo "---- $blaname: $("$info" "$out/$blaname.265")"
  if run "$out/$blaname.265" "$blaname.annexb"; then
    check "$blaname (Annex B, display order)" "$out/$blaname.ref" "$out/$blaname.annexb.md5"
  fi
done

# The caller-facing options, on one stream each.
if run "$out/main-1080p-slices.mp4" main-1080p-slices.len2 --length-size 2; then
  check "main-1080p-slices (2-byte NAL lengths)" "$out/main-1080p-slices.expect" "$out/main-1080p-slices.len2.md5"
fi
if run "$out/main10-720p.mp4" main10-720p.len3 --length-size 3; then
  check "main10-720p (3-byte NAL lengths)" "$out/main10-720p.expect" "$out/main10-720p.len3.md5"
fi
if run "$out/main10-1366x766-ctu16.mp4" main10-1366x766-ctu16.stride --stride-pad 192; then
  check "main10-1366x766-ctu16 (padded strides)" "$out/main10-1366x766-ctu16.expect" "$out/main10-1366x766-ctu16.stride.md5"
fi
if run "$out/main-720p.mp4" main-720p.hold --hold 24; then
  check "main-720p (24 frames held by the caller)" "$out/main-720p.expect" "$out/main-720p.hold.md5"
fi

# A new SPS at an IDR picture, from Main at 720p to Main 10 at 1080p,
# mid-stream: the decoder outputs what it holds, makes a new session in
# the new format and goes on. Each half starts with an IDR picture, so the
# reference is the two halves'.
name=switch-main-main10
cat "$out/main-720p-closed.265" "$out/main10-1080p-slices-cqm.265" > "$out/$name.265"
cat "$out/main-720p-closed.expect" "$out/main10-1080p-slices-cqm.expect" > "$out/$name.expect"
if run "$out/$name.265" "$name.annexb"; then
  check "$name (Annex B, display order)" "$out/$name.expect" "$out/$name.annexb.md5"
fi

# A stream to refuse that was not made has failed already.
for entry in "${refusals[@]}"; do
  IFS='|' read -r name _ <<<"$entry"
  [[ -s $out/$name.mp4 ]] || continue
  refused "$name" "$out/$name.mp4" '^hevc-framemd5: unsupported: profiles outside Main'
done
for entry in "${kvazaar_refusals[@]}"; do
  IFS='|' read -r name _ _ _ _ why <<<"$entry"
  [[ -s $out/$name.mp4 ]] || continue
  for input in "$out/$name.mp4" "$out/$name.265"; do
    refused "$name (${input##*.})" "$input" "$why"
  done
done
for entry in "${kvazaar_partial[@]}"; do
  IFS='|' read -r name _ _ _ _ good why <<<"$entry"
  [[ -s $out/$name.mp4 ]] || continue
  echo "---- $name: $("$info" "$out/$name.265")"
  head -n "$good" "$out/$name.ref" > "$out/$name.expect"
  for input in "$out/$name.mp4" "$out/$name.265"; do
    label=$name.${input##*.}
    if "$tool" "$node" "$input" > "$out/$label.frames" 2> "$out/$label.log"; then
      echo "FAIL $label: decoded whole, should be refused at frame $good"
      failed=1
    elif ! grep -q "$why" "$out/$label.log"; then
      echo "FAIL $label: $(tail -n1 "$out/$label.log")"
      failed=1
    else
      awk '{ print $NF }' "$out/$label.frames" > "$out/$label.md5"
      check "$name (${input##*.}, the $good frames before the refusal)" "$out/$name.expect" "$out/$label.md5"
      echo "     $(tail -n1 "$out/$label.log")"
    fi
  done
done

exit "$failed"
