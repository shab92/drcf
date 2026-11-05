// Two-process test of 2-party Schnorr and 2-party ECDSA. Run: ./run.sh build/test_tsig
#include <chrono>
#include <cstdio>
#include "tsig.h"
using namespace emp;
using clk = std::chrono::steady_clock;
int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    IKNP ot(party, io.get(), false);
    tsig::Curve C;
    int pass = 0, total = 0;
    auto sk = tsig::schnorr_keygen(C, io.get(), party);
    auto ek = tsig::ecdsa_keygen(C, io.get(), party);
    double ts = 0, te = 0; uint64_t bs = 0, be = 0; const int N = 20;
    for (int i = 0; i < N; i++) {
        tsig::bytes m = {'C', 'Y', (uint8_t)i, 0x42};
        uint64_t c0 = io->send_counter + io->recv_counter; auto t0 = clk::now();
        auto s1 = tsig::schnorr_sign(C, io.get(), party, sk, m);
        auto t1 = clk::now(); uint64_t c1 = io->send_counter + io->recv_counter;
        auto s2 = tsig::ecdsa_sign(C, io.get(), &ot, party, ek, m);
        auto t2 = clk::now(); uint64_t c2 = io->send_counter + io->recv_counter;
        ts += std::chrono::duration<double, std::milli>(t1 - t0).count(); te += std::chrono::duration<double, std::milli>(t2 - t1).count();
        bs += c1 - c0; be += c2 - c1;
        tsig::bytes m2 = m; m2[0] ^= 1;
        bool ok = tsig::schnorr_verify(C, sk.pk.get(), m, s1) && !tsig::schnorr_verify(C, sk.pk.get(), m2, s1)
               && tsig::ecdsa_verify(C, ek.pk.get(), m, s2) && !tsig::ecdsa_verify(C, ek.pk.get(), m2, s2);
        total++; if (ok) pass++;
    }
    printf("[%s] threshold signatures: %d/%d valid (OpenSSL-verified), tampered rejected\n", party == ALICE ? "B1" : "B2", pass, total);
    if (party == ALICE) printf("  Schnorr2: %.2f ms/sig, %.1f KiB/sig\n  ECDSA2  : %.2f ms/sig, %.1f KiB/sig\n", ts / N, bs / 1024.0 / N, te / N, be / 1024.0 / N);
    return pass == total ? 0 : 1;
}
