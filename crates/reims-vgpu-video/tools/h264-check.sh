#!/usr/bin/env bash
# The VA-API H.264 decoder against a software reference.
#
#   crates/reims-vgpu-video/tools/h264-check.sh [render node]
#
# Runs on Linux with libva, x264 and ffmpeg. x264 and ffmpeg are test
# oracles only: x264 writes the test streams and ffmpeg decodes them in
# software (-hwaccel none) for the reference MD5 of each frame. Neither is
# linked into anything, and nothing they write is committed: it all stays
# under build/.
#
# Each stream is decoded by examples/h264-framemd5 from its MP4 (the avcC
# record and length-prefixed samples, as VideoToolbox hands them over) and
# from its Annex B form, in display order; every frame's NV12 MD5 must
# equal ffmpeg's. Decode order must give the same frames. A stream the
# decoder must refuse (any whose SPS allows fields, as Apple's decoder
# refuses them) must fail with "interlaced".
#
# Long-term references, memory management operations 2 to 6, frame number
# gaps and 2-byte lengths, which x264 does not write, come from
# examples/h264-craft rewriting x264's one-reference streams; each must
# match ffmpeg's decode of it or, for a gap, its source's, which the gap
# does not change. A lost reference, where the SPS allows no gap, must
# decode the frames after it from a copy of the last reference;
# h264-craft's --check shows what would reach the GPU first, and the
# stream goes to it only if no sample is refused.
set -euo pipefail
source "$(dirname "$0")/lib.sh"

node=${1:-/dev/dri/renderD128}
out=${RVGV_H264_OUT:-$root/build/h264-check}
mkdir -p "$out"

build reims-vgpu-video h264-framemd5 h264-info h264-poc1 h264-craft
tool=$CARGO_TARGET_DIR/release/examples/h264-framemd5
info=$CARGO_TARGET_DIR/release/examples/h264-info
poc1=$CARGO_TARGET_DIR/release/examples/h264-poc1
craft=$CARGO_TARGET_DIR/release/examples/h264-craft

# An asymmetric scaling matrix, so that a list in the wrong scan order or
# the wrong slot changes the picture.
cqm=$out/asymmetric.cqm
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

# name size frames qpfile-lines x264-options...
# A qpfile line "N i" makes frame N a non-IDR I picture, "N I" an IDR; a
# min-keyint above N keeps x264 from promoting the I picture to an IDR.
streams=(
  "baseline-720p|1280x720|90|30 i,60 I|--profile baseline --preset fast --slices 4 --ref 3 --keyint 300 --min-keyint 151"
  "baseline-1080p|1920x1080|45|20 i|--profile baseline --preset fast --slices 2 --ref 2 --keyint 300 --min-keyint 151"
  "main-720p|1280x720|120|40 i,80 I|--profile main --preset medium --bframes 3 --b-pyramid normal --ref 4 --slices 2 --weightp 2 --keyint 300 --min-keyint 151"
  "main-1080p-opengop|1920x1080|90||--profile main --preset medium --bframes 3 --open-gop --keyint 30 --slices 3 --ref 3"
  "high-720p-cqm|1280x720|60|30 i|--profile high --preset medium --bframes 3 --b-pyramid strict --ref 6 --8x8dct --cqmfile $cqm --slices 3 --keyint 300 --min-keyint 151"
  "high-1080p|1920x1080|120|50 i,100 I|--profile high --preset medium --bframes 3 --b-pyramid normal --ref 5 --slices 4 --weightb --direct auto --cqm jvt --keyint 300 --min-keyint 151"
  "high-1080p-deep|1920x1080|60||--profile high --preset slower --bframes 8 --b-pyramid normal --ref 16 --keyint 300 --min-keyint 151"
  # One reference and no weighted prediction, so that h264-craft's
  # markings below leave each P slice its one reference.
  "main-ref1-720p|1280x720|60||--profile main --preset fast --bframes 0 --ref 1 --weightp 0 --keyint 300 --min-keyint 151"
  "baseline-ref1-360p|640x360|60||--profile baseline --preset fast --ref 1 --keyint 300 --min-keyint 151"
  "main-gapsrc-720p|1280x720|60|30 i|--profile main --preset fast --bframes 0 --ref 1 --weightp 0 --keyint 300 --min-keyint 151"
)

# The stream NAME, as Annex B and as an MP4; a stream that is not made ends
# the script, and leaves no file a later run would take. A stream made
# again drops the reference of the one before.
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

echo "h264-check on $node, $(date -Is), $(hostname)"
for entry in "${streams[@]}"; do
  IFS='|' read -r name size frames qp opts <<<"$entry"
  encode "$name" "$size" "$frames" "$qp" "$opts"
  reference "$name"
  echo "---- $name: $("$info" "$out/$name.264")"
  if run "$out/$name.mp4" "$name.avcc"; then
    check "$name (avcC, display order)" "$out/$name.ref" "$out/$name.avcc.md5"
  fi
  if run "$out/$name.264" "$name.annexb"; then
    check "$name (Annex B, display order)" "$out/$name.ref" "$out/$name.annexb.md5"
  fi
  if run "$out/$name.mp4" "$name.decode" --decode-order; then
    sort "$out/$name.ref" > "$out/$name.ref.sorted"
    sort "$out/$name.decode.md5" > "$out/$name.decode.sorted"
    check "$name (avcC, decode order, as a set)" "$out/$name.ref.sorted" "$out/$name.decode.sorted"
  fi
done

# Order count type 1, which x264 does not write: baseline-720p's SPS
# rewritten from type 2 to an equivalent type 1. The samples are the same,
# so ffmpeg's reference must not change either.
name=baseline-720p-poc1
if [[ ! -s $out/$name.mp4 ]]; then
  "$poc1" "$out/baseline-720p.264" "$out/$name.264" 2>/dev/null
  mux "$name"
fi
reference "$name"
echo "---- $name: $("$info" "$out/$name.264")"
check "$name (ffmpeg's reference against type 2's)" "$out/baseline-720p.ref" "$out/$name.ref"
if run "$out/$name.mp4" "$name.avcc"; then
  check "$name (avcC, display order)" "$out/$name.ref" "$out/$name.avcc.md5"
fi
if run "$out/$name.264" "$name.annexb"; then
  check "$name (Annex B, display order)" "$out/$name.ref" "$out/$name.annexb.md5"
fi

# Crafted streams: name|source|reference (ffmpeg, or the source's
# decode)|h264-craft's edits. The long-term stream marks the IDR picture
# long-term, then uses operations 4 and 6 (frame 10), 2 and 3 (20), 4 (30),
# 5 (40), and 4, 1 and 6 (50), with room for three references; Apple's
# hardware decoded both and matched its software decoder. A gap of 5 or 9
# frames before an I picture changes no pixel, and ffmpeg n9.0.2 leaves
# frames out after one, so a gap stream's reference is its source's.
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
for entry in "${crafted[@]}"; do
  IFS='|' read -r name source reference edits <<<"$entry"
  # shellcheck disable=SC2086 # $edits is a list of options
  "$craft" "$out/$source.264" "$out/$name" $edits 2> "$out/$name.craft.log"
  if [[ $reference == source ]]; then
    cp "$out/$source.ref" "$out/$name.ref"
  else
    rm -f "$out/$name.ref"
    reference "$name"
  fi
  echo "---- $name (crafted from $source: $edits): $("$info" "$out/$name.264")"
  if run "$out/$name.mp4" "$name.avcc"; then
    check "$name (avcC, display order)" "$out/$name.ref" "$out/$name.avcc.md5"
  fi
  if run "$out/$name.264" "$name.annexb"; then
    check "$name (Annex B, display order)" "$out/$name.ref" "$out/$name.annexb.md5"
  fi
  if run "$out/$name.mp4" "$name.decode" --decode-order; then
    sort "$out/$name.ref" > "$out/$name.ref.sorted"
    sort "$out/$name.decode.md5" > "$out/$name.decode.sorted"
    check "$name (avcC, decode order, as a set)" "$out/$name.ref.sorted" "$out/$name.decode.sorted"
  fi
done

# The caller-facing options, on one stream each.
if run "$out/high-1080p.mp4" high-1080p.len2 --length-size 2; then
  check "high-1080p (2-byte NAL lengths)" "$out/high-1080p.ref" "$out/high-1080p.len2.md5"
fi
if run "$out/main-720p.mp4" main-720p.len3 --length-size 3; then
  check "main-720p (3-byte NAL lengths)" "$out/main-720p.ref" "$out/main-720p.len3.md5"
fi
if run "$out/high-1080p.mp4" high-1080p.stride --stride-pad 192; then
  check "high-1080p (padded strides)" "$out/high-1080p.ref" "$out/high-1080p.stride.md5"
fi
if run "$out/main-720p.mp4" main-720p.hold --hold 24; then
  check "main-720p (24 frames held by the caller)" "$out/main-720p.ref" "$out/main-720p.hold.md5"
fi

# A new SPS at an IDR picture, with another size and profile, mid-stream:
# the decoder outputs what it holds, makes a new session and goes on. Each
# half starts with an IDR picture, so the reference is the two halves'.
name=switch-720p-1080p
cat "$out/baseline-720p.264" "$out/high-1080p.264" > "$out/$name.264"
cat "$out/baseline-720p.ref" "$out/high-1080p.ref" > "$out/$name.ref"
if run "$out/$name.264" "$name.annexb"; then
  check "$name (Annex B, display order)" "$out/$name.ref" "$out/$name.annexb.md5"
fi

# A lost reference: main-gapsrc-720p without frame 20, whose SPS allows no
# gap. Its 59 frames decode, the 20 before the loss and the 30 from the I
# picture at 30 on as the source's.
name=drop20-ref1
"$craft" "$out/main-gapsrc-720p.264" "$out/$name" --drop 20 --check > "$out/$name.check" 2>&1
echo "---- $name (crafted from main-gapsrc-720p: --drop 20): $("$info" "$out/$name.264")"
if grep -q refused "$out/$name.check"; then
  echo "FAIL $name: the stand-in refused a sample, so it goes to no GPU: $(grep -m1 refused "$out/$name.check")"
  failed=1
elif run "$out/$name.mp4" "$name.avcc"; then
  { sed -n '1,20p' "$out/main-gapsrc-720p.ref"; sed -n '31,60p' "$out/main-gapsrc-720p.ref"; } > "$out/$name.kept.ref"
  { sed -n '1,20p' "$out/$name.avcc.md5"; sed -n '30,59p' "$out/$name.avcc.md5"; } > "$out/$name.kept.md5"
  n=$(wc -l < "$out/$name.avcc.md5")
  if [[ $n != 59 ]]; then
    echo "FAIL $name: $n frames, not 59"
    failed=1
  else
    check "$name (59 frames; the 50 the loss does not reach)" "$out/$name.kept.ref" "$out/$name.kept.md5"
  fi
fi

# Interlaced coding, MBAFF and fake-interlaced (frames alone, of an SPS
# that allows fields) alike, is refused, cleanly, before any slice reaches
# the GPU.
encode high-1080p-interlaced 1920x1080 10 "" "--profile high --preset fast --tff"
encode high-1080p-fakeint 1920x1080 30 "" "--profile high --preset fast --fake-interlaced --bframes 2"
for name in high-1080p-interlaced high-1080p-fakeint; do
  refused "$name" "$out/$name.mp4" '^h264-framemd5: interlaced: '
done

exit "$failed"
