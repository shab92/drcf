// Two-process test of 2-party Schnorr on secp256k1. Run: ./run.sh build/test_tsig
#include <chrono>
#include <cstdio>
#include "tsig.h"
using namespace emp;
using clk = std::chrono::steady_clock;
int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    tsig::Curve C;
    auto sk = tsig::schnorr_keygen(C, io.get(), party);
    int pass = 0; const int N = 20; double ms = 0;
    for (int i = 0; i < N; i++) {
        tsig::bytes m = {'C', 'Y', (uint8_t)i};
        auto t0 = clk::now();
        auto s = tsig::schnorr_sign(C, io.get(), party, sk, m);
        ms += std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        tsig::bytes m2 = m; m2[0] ^= 1;
        if (tsig::schnorr_verify(C, sk.pk.get(), m, s) && !tsig::schnorr_verify(C, sk.pk.get(), m2, s)) pass++;
    }
    printf("[%s] 2-party Schnorr: %d/%d valid, tampered rejected, %.2f ms/sig\n", party == ALICE ? "P1" : "P2", pass, N, ms / N);
    return pass == N ? 0 : 1;
}
