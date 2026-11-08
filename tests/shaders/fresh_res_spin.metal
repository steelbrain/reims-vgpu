#include <metal_stdlib>
using namespace metal;
// One thread that runs until the guest releases it: it polls c[0], a word
// of guest memory the scenario writes, and ends when it reads nonzero, so
// the guest, not the GPU's speed, decides how long it runs. c[1] caps the
// polls (each after 4096 steps of busy work), so a scenario that never
// releases it still ends. On exit c[2] says which ended it: 0x600d the
// guest's release, 0xdead the cap. c[3] keeps the busy work's result,
// which depends on the poll count and so on timing: scenarios never read it.
kernel void fresh_res_spin(device atomic_uint *c [[buffer(0)]]) {
    uint cap = atomic_load_explicit(&c[1], memory_order_relaxed);
    uint a = 7;
    uint n = 0;
    while (atomic_load_explicit(&c[0], memory_order_relaxed) == 0 && n < cap) {
        for (uint j = 0; j < 4096; j++) {
            a = a * 1664525u + 1013904223u;
        }
        n++;
    }
    atomic_store_explicit(&c[3], a, memory_order_relaxed);
    atomic_store_explicit(&c[2], n < cap ? 0x600du : 0xdeadu, memory_order_relaxed);
}
