// Two-process test of share conversions (GC) and HE cross terms (SEAL). Run: ./run.sh build/test_compute
#include <chrono>
#include <cstdio>
#include <random>
#include "convert.h"
#include "he.h"
using namespace emp;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    he::Party P;
    const uint64_t t = P.t;
    std::mt19937_64 g(99);                       // shared seed: test data only
    const size_t G = 200, p = 4, n = G * p;
    std::vector<int32_t> x(n); std::vector<int64_t> beta(n);
    for (auto& v : x) v = (int32_t)(g() % 20001) - 10000;
    for (auto& v : beta) v = (int64_t)(g() % 4001) - 2000;
    // XOR shares of x (as from DistAEAD.Dec) and additive shares of beta
    std::vector<uint32_t> xs(n); std::vector<uint64_t> bs(n);
    for (size_t i = 0; i < n; i++) {
        uint32_t m = (uint32_t)g(); xs[i] = party == ALICE ? m : (uint32_t)x[i] ^ m;
        uint64_t r = g() % t; bs[i] = party == ALICE ? r : (convert::enc_signed(beta[i], t) + t - r) % t;
    }
    int bad = 0;
    auto t0 = clk::now(); uint64_t c0 = io->send_counter + io->recv_counter;
    auto xa = convert::b2a(s, xs, t);
    double tb = ms(t0); uint64_t cb = io->send_counter + io->recv_counter - c0;
    auto t1 = clk::now(); c0 = io->send_counter + io->recv_counter;
    auto ys = he::inner_products(P, io.get(), party, xa, bs, p);
    double th = ms(t1); uint64_t ch = io->send_counter + io->recv_counter - c0;
    auto t2 = clk::now(); c0 = io->send_counter + io->recv_counter;
    auto yx = convert::a2b_signed(s, ys, t);
    double ta = ms(t2); uint64_t ca = io->send_counter + io->recv_counter - c0;
    // threshold on y with theta
    const uint64_t theta = 5000000;
    auto al = convert::threshold(s, ys, t, theta);
    {   // debug: B2A and HE stages
        std::vector<uint64_t> o(n), oy2(G);
        if (party == ALICE) { io->send_data(xa.data(), 8*n); io->send_data(ys.data(), 8*G); io->flush(); }
        else { io->recv_data(o.data(), 8*n); io->recv_data(oy2.data(), 8*G); int e1=0,e2=0;
          for (size_t i=0;i<n;i++) if (convert::dec_signed((o[i]+xa[i])%t,t)!=x[i]) e1++;
          for (size_t k=0;k<G;k++){ int64_t w=0; for(size_t j=0;j<p;j++) w+=(int64_t)x[k*p+j]*beta[k*p+j]; if (convert::dec_signed((oy2[k]+ys[k])%t,t)!=w) e2++; }
          printf("debug: B2A errors %d/%zu, HE errors %d/%zu\n", e1, n, e2, G); }
    }
    // check (harness exchanges shares outside the protocol)
    std::vector<uint64_t> oy(G); std::vector<uint8_t> oa(G);
    if (party == ALICE) { io->send_data(yx.data(), 8 * G); io->send_data(al.data(), G); io->flush(); }
    else {
        io->recv_data(oy.data(), 8 * G); io->recv_data(oa.data(), G);
        for (size_t k = 0; k < G; k++) {
            int64_t want = 0; for (size_t j = 0; j < p; j++) want += (int64_t)x[k * p + j] * beta[k * p + j];
            int64_t got = (int64_t)(yx[k] ^ oy[k]);
            bool alert = (al[k] ^ oa[k]) & 1;
            if (got != want || alert != ((want < 0 ? -want : want) > (int64_t)theta)) bad++;
        }
        printf("  HE noise budget left: %d bits\n", he::last_budget);
        printf("[B2] B2A + HE inner products + A2B + threshold: %zu/%zu correct\n", G - bad, G);
    }
    if (party == ALICE)
        printf("  t = %llu (%d bits), %zu values, groups of %zu\n  B2A: %.1f ms, %.1f KiB\n  HE cross terms: %.1f ms, %.1f KiB\n  A2B: %.1f ms, %.1f KiB\n",
               (unsigned long long)t, 64 - __builtin_clzll(t), n, p, tb, cb / 1024.0, th, ch / 1024.0, ta, ca / 1024.0);
    return bad;
}
