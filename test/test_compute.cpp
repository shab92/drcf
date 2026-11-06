// Two-process test of DRCF.Compute building blocks: B2A (GC) -> CKKS cross terms (SEAL) -> A2B / threshold (GC).
// Run: ./run.sh build/test_compute
#include <chrono>
#include <cmath>
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
    const bool p1 = party == ALICE;
    std::mt19937_64 g(99);                                   // shared seed: test data only
    const size_t G = 400, p = 4, n = G * p;
    const double Q = 1000.0;                                 // fixed point used only when entering the GC
    std::vector<int32_t> x(n); std::vector<double> beta(n);
    std::uniform_real_distribution<double> ub(-1.0, 1.0);
    for (auto& v : x) v = (int32_t)(g() % 4001);             // e.g. Wh readings 0..4000
    for (auto& v : beta) v = ub(g);                          // real coefficients
    std::vector<uint32_t> xs(n); std::vector<double> bs(n);
    std::uniform_real_distribution<double> mb(-64.0, 64.0);  // beta shares
    for (size_t i = 0; i < n; i++) {
        uint32_t m = (uint32_t)g(); xs[i] = p1 ? m : (uint32_t)x[i] ^ m;
        double r = mb(g); bs[i] = p1 ? r : beta[i] - r;
    }
    int bad = 0;
    auto t0 = clk::now(); uint64_t c0 = io->send_counter + io->recv_counter;
    auto xa = convert::b2a(s, xs, 16);
    double tb = ms(t0); uint64_t cb = io->send_counter + io->recv_counter - c0;
    std::vector<double> xd(xa.begin(), xa.end());
    he::Params P;
    auto t1 = clk::now(); c0 = io->send_counter + io->recv_counter;
    auto y = he::inner_products(io.get(), party, xd, bs, p, P);
    double th = ms(t1); uint64_t ch = io->send_counter + io->recv_counter - c0;
    auto hs = he::last;
    // each broker rounds its real share to fixed point before the GC
    std::vector<int64_t> yi(G); for (size_t k = 0; k < G; k++) yi[k] = std::llround(y[k] * Q);
    auto t2 = clk::now(); c0 = io->send_counter + io->recv_counter;
    auto yx = convert::a2b(s, yi);
    double ta = ms(t2); uint64_t ca = io->send_counter + io->recv_counter - c0;
    const uint64_t theta = 1000000;                          // |y| > 1000 (x Q)
    auto al = convert::threshold(s, yi, theta);
    // harness: exchange outputs to check against the clear computation
    std::vector<uint64_t> oy(G); std::vector<uint8_t> oa(G); std::vector<double> oyr(G);
    if (p1) { io->send_data(yx.data(), 8 * G); io->send_data(al.data(), G); io->send_data(y.data(), 8 * G); io->flush(); }
    else {
        io->recv_data(oy.data(), 8 * G); io->recv_data(oa.data(), G); io->recv_data(oyr.data(), 8 * G);
        double maxerr = 0; int near = 0;
        for (size_t k = 0; k < G; k++) {
            double want = 0; for (size_t j = 0; j < p; j++) want += x[k * p + j] * beta[k * p + j];
            double got = (int64_t)(yx[k] ^ oy[k]) / Q;
            maxerr = std::max(maxerr, std::fabs(got - want));
            bool alert = (al[k] ^ oa[k]) & 1, ref = std::fabs(want) * Q > theta;
            if (std::fabs(got - want) > 0.01) bad++;
            if (alert != ref) { if (std::fabs(std::fabs(want) - theta / Q) < 0.01) near++; else bad++; }
        }
        printf("[B2] CKKS pipeline: %zu/%zu results within 0.01 of the exact value, max error %.2e; alert mismatches at the boundary: %d\n",
               G - bad, G, maxerr, near);
    }
    if (p1)
        printf("  CKKS N=%zu, coeff {60,36,60}, scale 2^36, fresh keys %.1f ms\n  B2A: %.1f ms, %.0f KiB\n  CKKS cross terms: %.1f ms, %.0f KiB; ciphertext out %zu B for %zu B plaintext (%.1fx)\n  A2B: %.1f ms, %.0f KiB\n",
               P.N, hs.keygen_ms, tb, cb / 1024.0, th, ch / 1024.0, hs.ct_bytes_out, hs.plain_bytes, (double)hs.ct_bytes_out / hs.plain_bytes, ta, ca / 1024.0);
    return bad;
}
