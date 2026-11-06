// Two-process test of 2-party DKLs23 threshold ECDSA (Protocol 3.4, n = t = 2) on secp256k1.
// Run: ./run.sh build/test_dkls23
#include <chrono>
#include <cstdio>
#include "dkls23.h"
using namespace emp;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    tsig::Curve C;   // secp256k1
    auto t0 = clk::now();
    dkls23::OTPair ots(party, io.get());
    auto key = dkls23::keygen(C, io.get(), party);
    double setup_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    const char* me = party == ALICE ? "P1" : "P2";

    // honest signing
    const int N = 20; int good = 0; double ms = 0; uint64_t bytes = 0;
    for (int i = 0; i < N; i++) {
        tsig::bytes m = {'C', '_', 'Y', (uint8_t)i};
        uint64_t c0 = io->send_counter + io->recv_counter; auto a = clk::now();
        auto r = dkls23::sign_and_check(C, io.get(), ots, party, key, m);
        ms += std::chrono::duration<double, std::milli>(clk::now() - a).count(); bytes += io->send_counter + io->recv_counter - c0;
        tsig::bytes m2 = m; m2[0] ^= 1;
        if (r.ok && dkls23::verify(C, key.pk.get(), m, r.sig) && !dkls23::verify(C, key.pk.get(), m2, r.sig)) good++;
        else printf("[%s] honest run %d failed: %s\n", me, i, r.failure.c_str());
    }
    printf("[%s] honest signing: %d/%d valid ECDSA signatures (OpenSSL, secp256k1), tampered message rejected\n", me, good, N);

    // P1 commits to an R_1 that is not r_1*G -> P2's step-8 check must fail
    auto c1 = dkls23::sign(C, io.get(), ots, party, key, {'x'}, party == ALICE ? dkls23::Cheat::InconsistentR : dkls23::Cheat::None);
    // P2 sends pk_2 != sk_2*G -> P1's step-8 check must fail
    auto c2 = dkls23::sign(C, io.get(), ots, party, key, {'y'}, party == BOB ? dkls23::Cheat::InconsistentPk : dkls23::Cheat::None);
    bool det1 = party == BOB ? (!c1.ok && c1.failure == "consistency check on R_j failed") : !c1.ok;
    bool det2 = party == ALICE ? (!c2.ok && c2.failure == "consistency check on pk_j failed") : !c2.ok;
    printf("[%s] cheating P1 (bad R): %s | cheating P2 (bad pk_i): %s\n", me, c1.ok ? "NOT detected" : c1.failure.c_str(), c2.ok ? "NOT detected" : c2.failure.c_str());

    // after aborted sessions, honest signing still works
    auto again = dkls23::sign_and_check(C, io.get(), ots, party, key, {'o', 'k'});
    if (party == ALICE) printf("  setup (OT + keygen) %.1f ms | sign %.2f ms, %.1f KiB per signature (both directions)\n", setup_ms, ms / N, bytes / 1024.0 / N);
    bool pass = good == N && det1 && det2 && again.ok;
    printf("[%s] %s\n", me, pass ? "ALL PASS" : "FAIL");
    // dump one signature for an independent check
    if (party == ALICE && again.ok) {
        FILE* f = fopen("dkls23_sig.txt", "w");
        auto hx = [&](const tsig::bytes& b) { std::string s; char t[3]; for (auto x : b) { snprintf(t, 3, "%02x", x); s += t; } return s; };
        unsigned char pub[65]; EC_POINT_point2oct(C.g, key.pk.get(), POINT_CONVERSION_UNCOMPRESSED, pub, 65, C.ctx);
        fprintf(f, "%s\n%s\n%s\n%s\n", hx(tsig::bytes(pub, pub + 65)).c_str(), "6f6b", hx(again.sig.r).c_str(), hx(again.sig.s).c_str());
        fclose(f);
    }
    return pass ? 0 : 1;
}
