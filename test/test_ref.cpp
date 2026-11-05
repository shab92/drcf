// Plain ASCON-128 vs official known-answer vectors.
#include <cstdio>
#include "ascon_ref.h"
#include "kat.h"
int main() {
    int ok = 0;
    for (auto& v : KATS) {
        auto c = ascon::encrypt(v.key.data(), v.nonce.data(), v.ad, v.pt);
        auto p = ascon::decrypt(v.key.data(), v.nonce.data(), v.ad, v.ct);
        auto bad = v.ct; bad[0] ^= 1;
        if (c == v.ct && p && *p == v.pt && !ascon::decrypt(v.key.data(), v.nonce.data(), v.ad, bad)) ok++;
    }
    printf("ascon reference vs KAT: %d/%zu\n", ok, KATS.size());
    return ok == (int)KATS.size() ? 0 : 1;
}
