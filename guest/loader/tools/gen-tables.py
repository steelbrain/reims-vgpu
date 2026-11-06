#!/usr/bin/env python3
"""Generate the loader's tables from a staged set (README.md).

    python3 -I -B tools/gen-tables.py STAGED OUT.h CONSUMER...

STAGED is what stage-consumer.py stages for each CONSUMER: its dormant
kext, its personality side file and its bridge directory. OUT.h is the C
header Loader.cpp compiles in, from our build output only: the System KC
libraries the consumers bind to, each consumer's personalities and sites,
and the library classes it derives from as the table's KDK lays them out
(bridge.py, layouts): each one's size and vtable, which the loader checks at
load (Loader.cpp, layoutsMatch). The kernel methods those vtables hold are
listed once, in kernelSymbols. No library version is compiled in: the
OpenCore entries' MinKernel and MaxKernel bound the releases the set is
injected on.

Refuses a staged consumer the command does not name, or one it names that
is not staged, a consumer that still has personalities, a table made from
another executable, a library still named in place of its shim, one shim
two consumers name, unsupported or overlapping sites, a consumer whose
OSBundleLibraries name another consumer (each binds on its own), and a
consumer with no library class, or a class without a size or vtable. Ours.
"""
import hashlib
import json
from pathlib import Path
import plistlib
import sys

EXTENSIONS = '/System/Library/Extensions/'


def c_string(text):
    """A C string literal. JSON's escapes are C's for the characters allowed."""
    bad = [c for c in text if not (c.isascii() and (c.isprintable() or c in '\t\n'))]
    if bad:
        raise SystemExit(f'unexpected character {bad[0]!r} in a generated string')
    return json.dumps(text)


def read_consumer(staged, name):
    """The staged consumer's plist, table and personalities, once they agree."""
    kext = staged / (name + '.kext')
    info = plistlib.loads((kext / 'Contents/Info.plist').read_bytes())
    if info.get('IOKitPersonalities'):
        raise SystemExit(f'{name}: consumer is not dormant')
    table = json.loads((staged / (name + '.bridge/fixups.json')).read_text())
    binary = kext / 'Contents/MacOS' / name
    if table['consumer'] != name or \
            hashlib.sha256(binary.read_bytes()).hexdigest() != table['consumer_sha256']:
        raise SystemExit(f'{name}: table describes another executable')
    personalities = plistlib.loads((staged / (name + '.personality.plist')).read_bytes())
    if not personalities or any(p.get('CFBundleIdentifier') != info['CFBundleIdentifier']
                                for p in personalities.values()):
        raise SystemExit(f'{name}: invalid personalities')
    for p in table['providers']:
        if p['shim'] not in info['OSBundleLibraries'] or p['library'] in info['OSBundleLibraries']:
            raise SystemExit(f'{name}: libraries were not substituted')
    return info, table, personalities


def site_rows(name, providers, first):
    """The consumer's sites, each naming its library's index in `libraries`."""
    rows, taken = [], set()
    for i, p in enumerate(providers):
        for s in p['sites']:
            offset, width = s['offset'], s['width']
            if width not in (4, 8) or taken & set(range(offset, offset + width)):
                raise SystemExit(f'{name}: unsupported or overlapping site at {offset:#x}')
            taken.update(range(offset, offset + width))
            rows.append(f'    {{ {offset:#x}, {width}, {first + i}, {s.get("opcode", 0)}, '
                        f'{c_string(s["kind"])}, {c_string(s["symbol"])}, '
                        f'{s["placeholder"]:#x}, {s["addend"]} }},')
    return rows


def layout_rows(name, providers, first, kernel):
    """The consumer's library classes and their vtables' slots, each kernel
    method by its index in `kernel` (which grows), each class by its
    library's index in `libraries`."""
    layouts, slots = [], []
    for i, p in enumerate(providers):
        for c in p.get('classes', ()):
            if not c.get('size') or not c.get('slots') or c['vtable'] != f'__ZTV{len(c["name"])}{c["name"]}':
                raise SystemExit(f'{name}: {c.get("name")} has no size, vtable or slots')
            layouts.append(f'    {{ {c_string(c["name"])}, {c_string(c["vtable"])}, {first + i}, '
                           f'{c["size"]}, {len(slots)}, {len(c["slots"])} }},')
            for slot in c['slots']:
                index = -1
                if slot['kernel']:
                    index = kernel.setdefault(slot['symbol'], len(kernel))
                slots.append(f'    {{ {c_string(slot["symbol"])}, {index} }},')
    if not layouts:
        raise SystemExit(f'{name}: no library class to check')
    return layouts, slots


def main(staged, out, names):
    tables = {p.name[:-len('.bridge')] for p in staged.glob('*.bridge')}
    if tables != set(names) or len(names) != len(set(names)):
        raise SystemExit(f'consumers {names} and staged tables {sorted(tables)} differ')
    read = [read_consumer(staged, name) for name in names]
    ids = {info['CFBundleIdentifier'] for info, _, _ in read}
    libraries, body, consumers, kernel = [], [], [], {}
    for ci, (name, (info, table, personalities)) in enumerate(zip(names, read)):
        if ids & set(info['OSBundleLibraries']):
            raise SystemExit(f'{name}: links against another consumer; each binds on its own')
        first = len(libraries)
        for p in table['providers']:
            if any(lib['shim'] == p['shim'] for lib in libraries):
                raise SystemExit(f'{name}: shim {p["shim"]} is already another consumer\'s')
            libraries.append({'id': p['library'], 'path': EXTENSIONS + p['path'], 'shim': p['shim']})
        xml = plistlib.dumps(personalities).decode()
        layouts, slots = layout_rows(name, table['providers'], first, kernel)
        body += [f'static constexpr char personality{ci}[] = {c_string(xml)};',
                 f'static constexpr Site sites{ci}[] = {{',
                 *site_rows(name, table['providers'], first), '};',
                 f'static constexpr Layout layouts{ci}[] = {{', *layouts, '};',
                 f'static constexpr Slot slots{ci}[] = {{', *slots, '};']
        consumers.append(f'    {{ {c_string(name)}, {c_string(info["CFBundleIdentifier"])}, '
                         f'personality{ci}, {first}, {len(table["providers"])}, sites{ci}, '
                         f'arrsize(sites{ci}), layouts{ci}, arrsize(layouts{ci}), '
                         f'slots{ci}, arrsize(slots{ci}) }},')
    lines = ['// Generated by guest/loader/tools/gen-tables.py from a staged set; do not edit.',
             'static constexpr Library libraries[] = {',
             *(f'    {{ {", ".join(c_string(lib[k]) for k in ("id", "path", "shim"))} }},'
               for lib in libraries),
             '};',
             'static constexpr const char *kernelSymbols[] = {',
             *(f'    {c_string(name)},' for name in kernel), '};', *body,
             'static constexpr Consumer consumers[] = {', *consumers, '};', '']
    out.write_text('\n'.join(lines))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
