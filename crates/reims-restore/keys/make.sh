#!/bin/bash
# Makes the project's LocalPolicy keys, which reims-restore builds in (src/anchors.rs, docs/design.md
# "The LocalPolicy"): three P-384 private keys in PKCS#8 PEM, for the chain's root, intermediate
# and leaf. The certificates are not made here: each restore builds them from these keys at the
# lengths of the guest's own, read from its kernel collection (anchors::chain), and has Ambrosia
# swap them in for the guest's anchors at boot.
#
# The keys are committed, private halves too. Whoever can write a VM's disk can write its NVRAM and
# its Ambrosia bundle, so a secret key protects nothing; trust comes from the anchor swap, as the
# guest's own trust comes from Apple's Hacktivation key, which ships in every restore OS.
#
# Usage: keys/make.sh DIR. It writes DIR/project-{root,intermediate,leaf}.key, refusing a key that
# exists, so the project's keys are made once.
set -euo pipefail
out=$1
mkdir -p "$out"
for n in root intermediate leaf; do
    k=$out/project-$n.key
    [[ ! -e $k ]] || { echo "make.sh: $k exists" >&2; exit 1; }
    openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:secp384r1 -outform PEM -out "$k"
done
