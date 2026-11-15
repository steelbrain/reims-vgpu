#!/bin/bash
# Makes the LocalPolicy signer reims-restore builds in (src/policy.rs, docs/design.md "The
# LocalPolicy"): a P-384 root, an intermediate and a leaf, the intermediate and leaf shaped as
# Image4 manifest signers are: CA:FALSE, the key usage below, and the Image4 constraints
# extension (1.2.840.113635.100.6.1.15), which names the manifest properties (MANP) and object
# properties (OBJP) a manifest may carry.
#
# The intermediate is CA:FALSE, as Apple's manifest signers are, so a generic X.509 verifier
# rejects the chain; Image4 checks it its own way.
#
# Usage: keys/make.sh DIR. It writes DIR/{root,int,leaf}.key, which stay out of the repository,
# and the files to commit: policy-leaf.key and policy-{root,intermediate,leaf}.der.
set -euo pipefail
out=$1
mkdir -p "$out"
cd "$out"
# The constraints extension's DER: SET { [MANP] SEQ { "MANP", SET { [name] SEQ { name, [0] { NULL } } } },
# [OBJP] SEQ { "OBJP", SET { [DGST] SEQ { "DGST", [0] { NULL } }, [name] SEQ { name, [1] { NULL } } } } },
# each [name] a private-class constructed tag whose number is the name's four bytes.
ext=$(python3 -I - <<'PY'
def der(tag, body):
    n = len(body)
    ln = bytes([n]) if n < 0x80 else bytes([0x80 | ((n.bit_length() + 7) // 8)]) + n.to_bytes((n.bit_length() + 7) // 8, 'big')
    return tag + ln + body
def private(name):
    num = int.from_bytes(name.encode(), 'big')
    groups = []
    while True:
        groups.append(num & 0x7f)
        num >>= 7
        if not num:
            break
    groups.reverse()
    return bytes([0xff]) + bytes([g | 0x80 for g in groups[:-1]] + [groups[-1]])
def ia5(s): return der(b'\x16', s.encode())
def entry(name, kind): return der(private(name), der(b'\x30', ia5(name) + der(bytes([0xa0 | kind]), der(b'\x05', b''))))
def group(name, entries): return der(private(name), der(b'\x30', ia5(name) + der(b'\x31', b''.join(entries))))
manp = group('MANP', [entry(n, 0) for n in ['BORD', 'CEPO', 'CHIP', 'CPRO', 'CSEC', 'ECID', 'SDOM']])
objp = group('OBJP', [entry('DGST', 0)] + [entry(n, 1) for n in ['DPRO', 'DSEC', 'EPRO', 'ESEC']])
print(der(b'\x31', manp + objp).hex().upper())
PY
)
cat > ext.cnf <<CNF
[req]
distinguished_name=dn
[dn]
[root]
basicConstraints=critical,CA:TRUE
keyUsage=critical,digitalSignature,keyCertSign,cRLSign
[img4]
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,nonRepudiation,keyEncipherment,dataEncipherment
1.2.840.113635.100.6.1.15=critical,DER:$ext
CNF
for n in root int leaf; do
    [ -f $n.key ] || openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:secp384r1 -out $n.key
done
dates=(-not_before 20200101000000Z -not_after 20560101000000Z)
openssl req -new -x509 -key root.key -sha384 -subj "/CN=reims-restore LocalPolicy Root/O=reims-restore" \
    "${dates[@]}" -set_serial 1 -config ext.cnf -extensions root -out root.pem
for n in int:root:Intermediate leaf:int:Leaf; do
    IFS=: read -r me ca cn <<< "$n"
    openssl req -new -key $me.key -subj "/CN=reims-restore LocalPolicy $cn/O=reims-restore" -config ext.cnf -out $me.csr
    openssl x509 -req -in $me.csr -CA $ca.pem -CAkey $ca.key -set_serial 1 -sha384 "${dates[@]}" \
        -extfile ext.cnf -extensions img4 -out $me.pem 2>/dev/null
done
openssl x509 -in root.pem -outform DER -out policy-root.der
openssl x509 -in int.pem -outform DER -out policy-intermediate.der
openssl x509 -in leaf.pem -outform DER -out policy-leaf.der
openssl pkcs8 -topk8 -nocrypt -in leaf.key -out policy-leaf.key
