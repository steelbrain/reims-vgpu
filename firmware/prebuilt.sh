#!/bin/sh
# The option ROM reims-ctl carries for reims-vgpu's own device, and the record of what it was
# built from:
#
#   sh prebuilt.sh BUILD     make prebuilt, after the ROM's build into BUILD, on a Mac
#   sh prebuilt.sh --check   make prebuilt-check, anywhere
#
# The first copies BUILD/ReimsGraphics.rom to prebuilt/ReimsGraphics.rom and writes
# prebuilt/ReimsGraphics.toml: the ROM's sha256, and that of each file it was built from, by its
# path in the repository (the Makefile, the ROM's assembler, and every file the compiler's
# dependency lists name). It runs on a Mac alone: the ROM's bytes are the compiler's, and the
# committed one's are Xcode's clang's and Homebrew's lld's. reims-ctl embeds the ROM
# (crates/reims-ctl/src/prebuilt.rs).
#
# The second fails where the committed ROM or a file the record names is not the record's: a
# change to what the ROM is built from, without the ROM built again. cargo xtask check and
# reims-ctl's tests run it.
set -eu
cd "$(dirname "$0")"
root=$(cd .. && pwd)
record=prebuilt/ReimsGraphics.toml
sha() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

if [ "${1:-}" = --check ]; then
  [ -f "$record" ] || {
    echo "prebuilt: firmware/$record is missing" >&2
    exit 1
  }
  stale=0
  want=$(sed -n 's/^rom = "\(.*\)"$/\1/p' "$record")
  if [ ! -f prebuilt/ReimsGraphics.rom ] || [ "$(sha prebuilt/ReimsGraphics.rom)" != "$want" ]; then
    echo "prebuilt: firmware/prebuilt/ReimsGraphics.rom is not the one its record names" >&2
    stale=1
  fi
  sed -n 's/^"\(.*\)" = "\(.*\)"$/\1 \2/p' "$record" | {
    changed=0
    while read -r f want; do
      if [ ! -f "$root/$f" ] || [ "$(sha "$root/$f")" != "$want" ]; then
        echo "prebuilt: $f changed since firmware/prebuilt/ReimsGraphics.rom was built" >&2
        changed=1
      fi
    done
    exit $changed
  } || stale=1
  if [ $stale = 1 ]; then
    echo "prebuilt: run make -C firmware prebuilt on a Mac, and commit the ROM and its record" >&2
    exit 1
  fi
  echo "prebuilt: firmware/prebuilt/ReimsGraphics.rom is built from the tree's sources"
  exit 0
fi

[ "$(uname)" = Darwin ] || {
  echo "prebuilt: build the committed ROM on a Mac: its bytes are the compiler's" >&2
  exit 1
}
build=$1
sources=$(
  {
    echo Makefile src/ReimsGraphicsRom.S src/OptionRom.S
    cat "$build"/obj/ReimsGraphics.d "$build"/obj/ReimsGraphicsOutput.d "$build"/obj/ReimsGraphicsName.d
  } | tr -d '\\' | tr ' \t' '\n\n' | grep -v -e ':$' -e '^$' -e '\.obj$' |
    while read -r f; do
      d=$(cd "$(dirname "$f")" && pwd)
      echo "${d#"$root"/}/$(basename "$f")"
    done | LC_ALL=C sort -u
)
mkdir -p prebuilt
cp "$build/ReimsGraphics.rom" prebuilt/ReimsGraphics.rom
chmod 644 prebuilt/ReimsGraphics.rom
{
  echo "# prebuilt/ReimsGraphics.rom, the option ROM reims-ctl carries for reims-vgpu's own device,"
  echo "# and the sha256 of each file it was built from: make -C firmware prebuilt writes both on a"
  echo "# Mac, and make -C firmware prebuilt-check checks them against the tree. Built with:"
  echo "#   $(clang --version | head -1)"
  echo "#   $(ld.lld --version | head -1)"
  echo "rom = \"$(sha prebuilt/ReimsGraphics.rom)\""
  echo
  echo "[sources]"
  for f in $sources; do
    echo "\"$f\" = \"$(sha "$root/$f")\""
  done
} >"$record"
echo "prebuilt: firmware/prebuilt/ReimsGraphics.rom and firmware/$record"
