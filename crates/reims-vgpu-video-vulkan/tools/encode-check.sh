#!/usr/bin/env bash
# The Vulkan Video H.264 and H.265 encoders against ffmpeg and against this
# crate's own decoders.
#
#   crates/reims-vgpu-video-vulkan/tools/encode-check.sh [NAME...]
#
# Runs on Linux with a GPU that encodes H.264 and H.265 through Vulkan
# Video, and ffmpeg. The sources are ffmpeg's testsrc2 and mandelbrot, raw
# NV12 and P010, at 720p, 1080p, 4K and an odd size, made into
# RVGV_VULKAN_ENC_OUT (build/vulkan-encode-check), where they are kept for
# the next run. ffmpeg is the oracle only, and nothing it writes is
# committed.
#
# Each stream below is encoded by examples/vulkan-encode on the device
# VideoDevice::create picks, under the Khronos validation layer, and must:
# - (a) decode in ffmpeg with no error, to as many frames as went in;
# - (b) keep the PSNR against its source, mean and lowest frame, above the
#   stream's floor (measured on the RTX 5080, 11-03, and set 1 dB under);
# - (c) with a bitrate, keep the HRD buffer (the encoder's default: a
#   second at the peak, three quarters full at the start) from underflowing,
#   by a leaky bucket over the frames' bytes in decode order at the peak;
#   and, over 300 frames or more (10 s), come within 10% of a CBR target,
#   and under the peak and over half the average of a VBR one. A shorter
#   stream's bitrate is shown only: its first IDR picture, which the
#   buffer's start pays for, is a large part of it;
# - (d) decode with this crate's decoder (vulkan-h264-framemd5,
#   vulkan-h265-framemd5) to the frames ffmpeg decodes, bit for bit;
# - (e) have an IDR picture at each frame forced IDR, in its log and as
#   ffprobe's key frame there;
# - (f) read its sources as asked: images in place, or host-visible planes
#   through one GPU copy each, and write its bitstream into the buffer its
#   caller gave, host-visible slots among them; with no copy on the CPU;
# - (H.264) bound motion vectors in its SPS no tighter than 2^15 quarter
#   samples, as ffmpeg reads the SPS: the device's own bound is raised
#   where it is tighter (NVIDIA's 615.71.09 writes 2^1);
# - and leave the validation layer silent.
#
# Then each encoder is timed, with no validation layer, at 1080p and 4K,
# its sources in images read in place and in host-visible planes copied on
# the GPU: the GPU time of each picture's work, and of the copy within it.
#
# The sources are made first, with no GPU; the encodes then run holding
# every GPU's lock (scripts/internal/gpu-locks.sh --all), as the script runs
# itself again under it. The exit status is 1 where any check fails.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../../reims-vgpu-video-vaapi/tools/lib.sh"

out=${RVGV_VULKAN_ENC_OUT:-$root/build/vulkan-encode-check}
mkdir -p "$out"
tool=$CARGO_TARGET_DIR/release/examples/vulkan-encode
only=" $* "

# name|lavfi source|size|frames|pixel format
sources=(
  "t720|testsrc2=size=1280x720:rate=30,noise=alls=8:allf=t+u|1280x720|300|nv12"
  "t1080|testsrc2=size=1920x1080:rate=30|1920x1080|120|nv12"
  "m4k|mandelbrot=size=3840x2160:rate=30|3840x2160|60|nv12"
  "odd|mandelbrot=size=1918x1078:rate=30|1918x1078|40|nv12"
  "t720p10|testsrc2=size=1280x720:rate=30,noise=alls=8:allf=t+u|1280x720|300|p010le"
  "t1080p10|testsrc2=size=1920x1080:rate=30|1920x1080|120|p010le"
  "m4kp10|mandelbrot=size=3840x2160:rate=30|3840x2160|60|p010le"
)

# name|source|PSNR floor (mean,lowest)|rate (cbr:BITS, vbr:AVERAGE,PEAK or
# cqp)|forced IDR frames|vulkan-encode's options
streams=(
  "h264-baseline-720p|t720|37,37|cbr:3000000|45,100|--codec h264 --profile baseline --refs 2 --slices 4 --idr 60"
  "h264-main-720p-b3|t720|38,37|vbr:4000000,6000000|77|--codec h264 --profile main --b 3 --refs 2 --intra 30 --idr 120 --in-band"
  "h264-main-720p-cavlc|t720|37,37|cbr:2500000||--codec h264 --profile main --cavlc --refs 3 --idr none --quality 4"
  "h264-high-720p-wrap|t720|38,37|cbr:3000000||--codec h264 --profile high --idr none --refs 2"
  "h264-high-720p-switch|t720|38,36|cbr:2000000||--codec h264 --profile high --switch-rc 150:cbr:6000000"
  "h264-high-1080p|t1080|45,41|vbr:8000000,12000000|30|--codec h264 --profile high --b 2 --refs 3 --slices 2"
  "h264-high-1080p-cqp|t1080|45,44|cqp||--codec h264 --profile high --b 1 --rc cqp:24,26,28 --tuning hq"
  "h264-main-1080p-host|t1080|46,41|vbr:8000000,12000000|60|--codec h264 --profile main --b 2 --in-band --source buffer --bitstream import"
  "h264-high-4k|m4k|44,39|cbr:25000000||--codec h264 --profile high --b 2 --refs 2 --tuning ll"
  "h264-high-odd|odd|42,42|cqp||--codec h264 --profile high --rc cqp:26,28,30 --source buffer"
  "h265-main-720p|t720|37,36|cbr:2500000|50,77|--codec h265 --b 3 --refs 3 --intra 30 --idr 120 --slices 3 --in-band"
  "h265-main-720p-p|t720|38,38|vbr:3000000,4500000||--codec h265 --refs 4 --idr none --quality 6"
  "h265-main-1080p|t1080|43,39|vbr:6000000,9000000|40|--codec h265 --b 2 --refs 2"
  "h265-main-1080p-host|t1080|43,41|cbr:6000000||--codec h265 --b 2 --in-band --source buffer --bitstream import"
  "h265-main-4k|m4k|43,41|cqp||--codec h265 --rc cqp:26,28,30 --b 2"
  "h265-main-odd|odd|39,37|cqp||--codec h265 --rc cqp:30,32,34 --b 1 --source buffer"
  "h265-main10-720p|t720p10|38,36|cbr:3000000|100|--codec h265 --profile main10 --b 2 --refs 2 --idr 60"
  "h265-main10-1080p|t1080p10|47,43|vbr:8000000,12000000||--codec h265 --profile main10 --b 2 --slices 2"
  "h265-main10-1080p-host|t1080p10|47,45|vbr:8000000,12000000|30|--codec h265 --profile main10 --in-band --source buffer --bitstream import"
  "h265-main10-4k|m4kp10|44,38|cbr:20000000||--codec h265 --profile main10 --b 2"
)

# label|source|options: the timed runs.
timed=(
  "h264-1080p-image|t1080|--codec h264 --profile high --rc vbr:8000000,12000000"
  "h264-1080p-buffer|t1080|--codec h264 --profile high --rc vbr:8000000,12000000 --source buffer"
  "h264-4k-image|m4k|--codec h264 --profile high --rc vbr:25000000,37500000"
  "h264-4k-buffer|m4k|--codec h264 --profile high --rc vbr:25000000,37500000 --source buffer"
  "h265-1080p-image|t1080|--codec h265 --rc vbr:6000000,9000000"
  "h265-1080p-buffer|t1080|--codec h265 --rc vbr:6000000,9000000 --source buffer"
  "h265-4k-image|m4k|--codec h265 --rc vbr:20000000,30000000"
  "h265-4k-buffer|m4k|--codec h265 --rc vbr:20000000,30000000 --source buffer"
  "h265-main10-4k-image|m4kp10|--codec h265 --profile main10 --rc vbr:20000000,30000000"
  "h265-main10-4k-buffer|m4kp10|--codec h265 --profile main10 --rc vbr:20000000,30000000 --source buffer"
)

# source NAME: the source's fields.
source_of() {
  local entry
  for entry in "${sources[@]}"; do
    if [[ $entry == "$1|"* ]]; then
      echo "$entry"
      return
    fi
  done
  return 1
}

# fail LABEL WHY: a FAIL, counted.
fail() {
  echo "FAIL $1: $2"
  failed=1
}

# psnr STREAM SIZE PIX SOURCE: mean and lowest PSNR over the frames.
psnr() {
  local stream=$1 size=$2 pix=$3 src=$4
  ffmpeg -nostdin -hide_banner -loglevel error -i "$stream" -f rawvideo -pix_fmt "$pix" - |
    ffmpeg -nostdin -hide_banner -loglevel error -f rawvideo -pix_fmt "$pix" -s "$size" -r 30 -i - \
      -f rawvideo -pix_fmt "$pix" -s "$size" -r 30 -i "$src" \
      -lavfi "[0:v][1:v]psnr=stats_file=$stream.psnr" -f null -
  awk '{ split($6, a, ":"); s += a[2]; if (NR == 1 || a[2] < min) min = a[2] }
       END { printf "%.2f %.2f %d\n", s / NR, min, NR }' "$stream.psnr"
}

# vbv LOG RATE: the HRD buffer's lowest fill, in bits, over LOG's frames in
# decode order, at RATE bits a second and 30 frames a second, a second of
# it in size and three quarters of that at the start: below 0 is an
# underflow.
vbv() {
  awk -v rate="$2" '/^frame / {
      if (!started) { size = rate; fill = size * 3 / 4; low = fill; started = 1 }
      fill -= ($18 + $20) * 8
      if (fill < low) low = fill
      fill += rate / 30
      if (fill > size) fill = size
    }
    END { printf "%.0f\n", low }' "$1"
}

# encode NAME: the stream NAME, checked.
encode() {
  local name=$1 src_name floor rate forced opts
  IFS='|' read -r _ src_name floor rate forced opts <<<"$(printf '%s\n' "${streams[@]}" | grep "^$1|")"
  local entry lavfi size frames pix
  entry=$(source_of "$src_name")
  IFS='|' read -r _ lavfi size frames pix <<<"$entry"
  local src=$out/$src_name.yuv codec=${name%%-*}
  local stream=$out/$name.bit log=$out/$name.log vvl=$out/$name.vvl
  local rate_opts=()
  case $rate in
    cbr:* | vbr:*) rate_opts=(--rc "$rate") ;;
  esac
  local force=()
  [[ -z $forced ]] || force=(--force-idr "$forced")
  echo "---- $name ($src_name, $size, $frames frames, $rate${forced:+, IDR forced at $forced}): $opts"
  rm -f "$vvl" "$stream"
  # shellcheck disable=SC2086 # $opts is a list of options
  if ! VK_LOADER_LAYERS_ENABLE='*validation' VK_KHRONOS_VALIDATION_UNIQUE_HANDLES=false \
      VK_KHRONOS_VALIDATION_LOG_FILENAME="$vvl" \
      "$tool" --size "$size" --input "$src" --output "$stream" "${rate_opts[@]}" "${force[@]}" \
        $opts > "$out/$name.frames" 2> "$log"; then
    fail "$name" "the encode failed: $(tail -n1 "$log")"
    return
  fi
  echo "     $(head -n1 "$log")"
  # The layer says once, on stdout, where it logs, which shows it ran.
  local ran="Validation Layer Info - Logging validation error to $vvl"
  local other
  other=$(grep -v -e '^frame ' -e "^$ran\$" "$out/$name.frames" || true)
  if ! grep -qx -- "$ran" "$out/$name.frames"; then
    fail "$name" "the validation layer did not run"
  elif [[ -s $vvl || -n $other ]]; then
    fail "$name" "the validation layer says: $(head -c 400 "$vvl" 2> /dev/null; head -n1 <<<"$other")"
  else
    echo "PASS $name: the validation layer says nothing"
  fi

  # (a) ffmpeg decodes it, all of it.
  local errors decoded
  errors=$(ffmpeg -nostdin -hide_banner -loglevel error -i "$stream" -f null - 2>&1 || echo "exit $?")
  decoded=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$stream")
  if [[ -n $errors ]]; then
    fail "$name" "ffmpeg's decode says: $(head -n1 <<<"$errors")"
  elif [[ $decoded != "$frames" ]]; then
    fail "$name" "ffmpeg decodes $decoded frames, not $frames"
  else
    echo "PASS $name: ffmpeg decodes $decoded frames, with no error"
  fi

  # (b) PSNR above the floor.
  local mean low n
  read -r mean low n <<<"$(psnr "$stream" "$size" "$pix" "$src")"
  if awk -v m="$mean" -v l="$low" -v f="$floor" 'BEGIN { split(f, a, ","); exit !(m >= a[1] && l >= a[2]) }'; then
    echo "PASS $name: PSNR mean $mean dB, lowest $low dB over $n frames (floor $floor)"
  else
    fail "$name" "PSNR mean $mean dB, lowest $low dB over $n frames, under the floor $floor"
  fi

  # (c) the bitrate, and the HRD buffer at the peak.
  local got peak
  got=$(awk '/^frame / { b += ($18 + $20) * 8; n++ } END { printf "%.0f", b * 30 / n }' "$out/$name.frames")
  case $rate in
    cbr:*)
      local target=${rate#cbr:}
      if [[ $opts == *--switch-rc* ]]; then
        # The second half at the rate switched to.
        local at=${opts#*--switch-rc } second
        target=${at#*:cbr:}
        target=${target%% *}
        at=${at%%:*}
        second=$(awk -v at="$at" '/^frame / && $2 >= at { b += ($18 + $20) * 8; n++ } END { printf "%.0f", b * 30 / n }' "$out/$name.frames")
        got=$second
      fi
      peak=$target
      if ((frames < 300)); then
        echo "     $name: $got bit/s, CBR $target ($(awk -v g="$got" -v t="$target" 'BEGIN { printf "%+.1f%%", (g / t - 1) * 100 }')) over $frames frames"
      elif awk -v g="$got" -v t="$target" 'BEGIN { exit !(g >= t * 0.9 && g <= t * 1.1) }'; then
        echo "PASS $name: $got bit/s, CBR $target ($(awk -v g="$got" -v t="$target" 'BEGIN { printf "%+.1f%%", (g / t - 1) * 100 }'))"
      else
        fail "$name" "$got bit/s, CBR $target: past 10%"
      fi
      ;;
    vbr:*)
      local average=${rate#vbr:}
      peak=${average#*,}
      average=${average%,*}
      if ((frames < 300)); then
        echo "     $name: $got bit/s, VBR $average under $peak ($(awk -v g="$got" -v t="$average" 'BEGIN { printf "%+.1f%%", (g / t - 1) * 100 }')) over $frames frames"
      elif awk -v g="$got" -v a="$average" -v p="$peak" 'BEGIN { exit !(g <= p && g >= a / 2) }'; then
        echo "PASS $name: $got bit/s, VBR $average under $peak ($(awk -v g="$got" -v t="$average" 'BEGIN { printf "%+.1f%%", (g / t - 1) * 100 }'))"
      else
        fail "$name" "$got bit/s, VBR $average under $peak: past the peak or under half the average"
      fi
      ;;
    *)
      peak=
      echo "     $name: $got bit/s at a constant QP"
      ;;
  esac
  if [[ -n $peak && $opts != *--switch-rc* ]]; then
    local lowest
    lowest=$(vbv "$out/$name.frames" "$peak")
    if ((lowest >= 0)); then
      echo "PASS $name: the HRD buffer at $peak bit/s never underflows (lowest fill $lowest bits)"
    else
      fail "$name" "the HRD buffer at $peak bit/s underflows by $((-lowest)) bits"
    fi
  fi

  # (d) this crate's decoder, bit for bit against ffmpeg.
  local md5pix=nv12 decoder=$CARGO_TARGET_DIR/release/examples/vulkan-$codec-framemd5
  [[ $pix == nv12 ]] || md5pix=p010le
  if hashes "$out/$name.ref" -i "$stream" -fps_mode passthrough -pix_fmt "$md5pix"; then
    if "$decoder" "$stream" > "$out/$name.own" 2> "$out/$name.own.log"; then
      awk '{ print $NF }' "$out/$name.own" > "$out/$name.own.md5"
      check "$name (own decoder against ffmpeg)" "$out/$name.ref" "$out/$name.own.md5"
    else
      fail "$name" "the own decoder failed: $(tail -n1 "$out/$name.own.log")"
    fi
  else
    fail "$name" "ffmpeg's framemd5 failed"
  fi

  # (e) IDR pictures where forced.
  if [[ -n $forced ]]; then
    local keys missing=
    keys=$(ffprobe -v error -select_streams v:0 -show_entries frame=key_frame -of csv=p=0 "$stream")
    for f in ${forced//,/ }; do
      if ! grep -q "^frame $f pts .* type IDR " "$out/$name.frames" ||
        [[ $(sed -n "$((f + 1))p" <<<"$keys") != 1* ]]; then
        missing+=" $f"
      fi
    done
    if [[ -z $missing ]]; then
      echo "PASS $name: IDR pictures at the frames forced ($forced)"
    else
      fail "$name" "no IDR picture at frames$missing"
    fi
  fi

  # The SPS's motion vector bounds.
  if [[ $codec == h264 ]]; then
    local bounds
    bounds=$(ffmpeg -nostdin -hide_banner -i "$stream" -c copy -bsf:v trace_headers -frames:v 1 -f null - 2>&1 |
      awk '/log2_max_mv_length_(horizontal|vertical)/ && n++ < 2 { printf "%s ", $NF }')
    if [[ $bounds == "15 15 " ]]; then
      echo "PASS $name: the SPS bounds motion vectors at 2^15 quarter samples"
    else
      fail "$name" "the SPS's log2_max_mv_length_horizontal and _vertical are '$bounds', not 15 15"
    fi
  fi

  # (f) where the sources were read, and no copy on the CPU.
  local counts want
  counts=$(grep -o 'sources in_place [0-9]* gpu_copies [0-9]* cpu_copies [0-9]*' "$log")
  if [[ $opts == *"--source buffer"* ]]; then
    want="sources in_place 0 gpu_copies $frames cpu_copies 0"
  else
    want="sources in_place $frames gpu_copies 0 cpu_copies 0"
  fi
  local into=
  [[ $opts != *"--bitstream import"* ]] || into=", each bitstream written into a slot of one host-visible allocation"
  if [[ $counts == "$want" ]]; then
    echo "PASS $name: $counts$into"
  else
    fail "$name" "$counts, not $want"
  fi
}

if [[ -z ${RVG_GPU_LOCKS_HELD:-} ]]; then
  # The sources and the tools, with no GPU.
  build reims-vgpu-video-vulkan vulkan-encode vulkan-h264-framemd5 vulkan-h265-framemd5
  for entry in "${sources[@]}"; do
    IFS='|' read -r name lavfi size frames pix <<<"$entry"
    if [[ ! -s $out/$name.yuv ]]; then
      ffmpeg -nostdin -hide_banner -loglevel error -y -f lavfi -i "$lavfi" -frames:v "$frames" \
        -pix_fmt "$pix" -f rawvideo "$out/$name.part.yuv"
      mv "$out/$name.part.yuv" "$out/$name.yuv"
    fi
  done
  exec bash "$root/scripts/internal/gpu-locks.sh" --all -- bash "$0" "$@"
fi

echo "vulkan encode-check, $(date -Is), $(hostname)"
"$tool" --capabilities
for entry in "${streams[@]}"; do
  IFS='|' read -r name _ <<<"$entry"
  if wanted "$name"; then
    encode "$name"
  fi
done

# GPU time a picture, with no validation layer.
if wanted timing; then
  echo "---- timing: each picture's work on the GPU, and its source's copy within it"
  for entry in "${timed[@]}"; do
    IFS='|' read -r label src_name opts <<<"$entry"
    IFS='|' read -r _ _ size _ _ <<<"$(source_of "$src_name")"
    # shellcheck disable=SC2086 # $opts is a list of options
    if "$tool" --size "$size" --input "$out/$src_name.yuv" --output "$out/$label.bit" --timing \
        $opts > "$out/$label.frames" 2> "$out/$label.log"; then
      echo "     $label ($size): $(grep '^gpu ' "$out/$label.log"); $(grep '^copy ' "$out/$label.log")"
    else
      fail "$label" "$(tail -n1 "$out/$label.log")"
    fi
  done
fi

exit "$failed"
