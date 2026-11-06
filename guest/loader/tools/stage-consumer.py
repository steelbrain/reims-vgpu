#!/usr/bin/env python3
"""Stage a consumer kext and its shims for the loader (README.md).

    python3 -I -B tools/stage-consumer.py KEXT SHIMS OUT

KEXT is the consumer as its own build makes it, SHIMS the directory
gen-shims.py made from its executable (its shims and fixups.json). OUT,
which must not exist, gets:
- <Executable>.kext, a copy of KEXT whose OSBundleLibraries name the shims
  in their libraries' place and which has no IOKitPersonalities: OpenCore
  injects it dormant, and the loader publishes its personalities once it has
  bound it;
- <Executable>.personality.plist, those personalities, each with the
  consumer's CFBundleIdentifier;
- the shims;
- <Executable>.bridge, a copy of SHIMS, so the loader's tables are made
  from exactly this build's table (gen-tables.py).
Refuses a table made from another executable, a library the consumer does
not name, and a consumer with no personalities. Only our build output is
read and copied. Ours.
"""

import hashlib
import json
from pathlib import Path
import plistlib
import shutil
import sys


def stage(kext, shims, out):
    info = plistlib.loads((kext / 'Contents/Info.plist').read_bytes())
    name = info['CFBundleExecutable']
    if not (shims / 'fixups.json').is_file():
        raise SystemExit(f'{shims}: no fixups.json (gen-shims.py makes it)')
    table = json.loads((shims / 'fixups.json').read_text())
    executable = kext / 'Contents/MacOS' / name
    if table['consumer'] != name or \
            hashlib.sha256(executable.read_bytes()).hexdigest() != table['consumer_sha256']:
        raise SystemExit(f'{kext}: the fixup table describes another executable')
    if out.exists():
        raise SystemExit(f'{out}: exists; staging starts from nothing')
    out.mkdir(parents=True)
    libraries = info['OSBundleLibraries']
    for provider in table['providers']:
        lib, shim = provider['library'], provider['shim']
        if lib not in libraries or not provider['version'] or not provider['sites']:
            raise SystemExit(f'{kext}: invalid bridge library {lib}')
        src = shims / (provider['binary'] + 'Bridge.kext')
        shim_info = plistlib.loads((src / 'Contents/Info.plist').read_bytes())
        if shim_info['CFBundleIdentifier'] != shim:
            raise SystemExit(f'{src}: wrong shim identifier')
        del libraries[lib]
        libraries[shim] = shim_info['CFBundleVersion']
        shutil.copytree(src, out / src.name)
    personalities = info.pop('IOKitPersonalities', None)
    if not personalities:
        raise SystemExit(f'{kext}: no personalities to publish')
    for personality in personalities.values():
        personality['CFBundleIdentifier'] = info['CFBundleIdentifier']
    dst = out / (name + '.kext')
    shutil.copytree(kext, dst)
    (dst / 'Contents/Info.plist').write_bytes(plistlib.dumps(info))
    (out / (name + '.personality.plist')).write_bytes(plistlib.dumps(personalities))
    shutil.copytree(shims, out / (name + '.bridge'))
    print(f'{name}: staged without personalities; '
          f'{sum(len(p["sites"]) for p in table["providers"])} sites')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    stage(*(Path(p) for p in sys.argv[1:]))
