// Two-process test of DistAEAD. Run: ./test_distaead 1 & ./test_distaead 2
// The harness gives both processes the same seed so they agree on the test data,
// but each broker only passes ITS OWN key share into the protocol. The harness then
// sends B1's share to B2 (outside the protocol) purely to check M1 ^ M2 == M.
#include <cstdio>
#include <random>
#include "distaead.h"
#include "ascon_ref.h"
#include "kat.h"
using namespace emp;
using bytes = std::vector<uint8_t>;

static bytes rnd(std::mt19937_64& g, size_t n) { bytes b(n); for (auto& x : b) x = (uint8_t)g(); return b; }
static bytes xr(const bytes& a, const bytes& b) { bytes o(a.size()); for (size_t i = 0; i < a.size(); i++) o[i] = a[i] ^ b[i]; return o; }

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    std::mt19937_64 g(2026);
    int pass = 0, total = 0;
    for (auto& v : KATS) {
        // DistAEAD.Setup for publisher P and subscriber S
        bytes KP = v.key, KP1 = rnd(g, 16), KP2 = xr(KP, KP1);
        bytes KS = rnd(g, 16), KS1 = rnd(g, 16), KS2 = xr(KS, KS1);
        const bytes& myKP = party == ALICE ? KP1 : KP2;
        const bytes& myKS = party == ALICE ? KS1 : KS2;

        // 1. publisher's ciphertext (official vector) -> brokers decrypt into shares
        auto d = distaead::dist_dec(s, myKP, v.nonce.data(), v.ad, v.ct);
        bool dec_ok = d.ok;
        bytes other(d.share.size());
        if (party == ALICE) { io->send_data(d.share.data(), d.share.size()); io->flush(); }
        else { io->recv_data(other.data(), other.size()); dec_ok = dec_ok && xr(d.share, other) == v.pt && (v.pt.empty() || d.share != v.pt); }

        // 2. tampered ciphertext -> ⊥ for both
        bytes bad = v.ct; bad[bad.size() / 2] ^= 0x20;
        bool rej = !distaead::dist_dec(s, myKP, v.nonce.data(), v.ad, bad).ok;

        // 3. brokers re-encrypt their shares for the subscriber
        bytes n2 = rnd(g, 16), ad2 = {'r', 'e', 'q'};
        auto ct2 = distaead::dist_enc(s, myKS, n2.data(), ad2, d.share);
        auto back = ascon::decrypt(KS.data(), n2.data(), ad2, ct2);
        bool enc_ok = back && *back == v.pt && ct2 == ascon::encrypt(KS.data(), n2.data(), ad2, v.pt);

        total++;
        if (party == ALICE) dec_ok = true;  // B1 can't check without B2's share
        if (dec_ok && rej && enc_ok) pass++;
        else printf("[party %d] FAIL ad=%zu pt=%zu dec=%d rej=%d enc=%d\n", party, v.ad.size(), v.pt.size(), dec_ok, rej, enc_ok);
    }
    printf("[%s] DistAEAD end-to-end: %d/%d\n", party == ALICE ? "B1" : "B2", pass, total);
    return pass == total ? 0 : 1;
}
