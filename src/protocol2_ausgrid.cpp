// DRCF Table A1, Protocol II: Pi_ML-AnomalyDetect on the Ausgrid solar-home data, with the anomaly model of
// Jaiswal et al. (AICCC 2021): Prophet (configured as in their notebook), anomaly if the reading lies outside the
// model interval or equals 0.
//
//   ./run.sh build/protocol2_ausgrid <csv> [customer-id | all] [k]
//   k: interval half-width in sigmas; Table A1 uses 3 (default). Prophet's 99% interval is about 2.576.
//
// Two-party flow (B1 = ALICE, B2 = BOB), input D(i) = additive shares of the half-hourly readings (Wh):
//   1. Model fit on shares (local, no interaction): theta_i = G^{-1} A^T y_i, yhat_i = A theta_i  (trend + seasonality + holidays)
//   2. Residual variance: e_i = y_i - yhat_i; sum (e_1 + e_2)^2 via CKKS cross terms (he.h), each broker gets a share
//   3. GC: sigma = isqrt(var), bound = 3 sigma (Table A1); for every reading r = [ |y - yhat| > bound  or  y == 0 ]
//   4. Output: XOR shares of r for every reading
// Harness only: both processes read the same CSV and split it with a shared seed (standing in for DRCF.Publish),
// and at the end B2 sends its output shares to B1, which compares them with (a) the same model computed in the
// clear and (b) the anomaly labels Jaiswal et al. published.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include "batch.h"
#include "he.h"
#include "prophet_lite.h"

using namespace emp;
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }
static const double Q = 16.0;   // fixed-point scale (1/16 Wh) when real shares enter the GC

struct Row { prophet_lite::Stamp ts; int64_t wh; int label; };

static std::map<int, std::vector<Row>> load(const std::string& path) {
    std::ifstream f(path); std::string line; std::getline(f, line);
    std::map<int, std::vector<Row>> by;
    while (std::getline(f, line)) {
        std::stringstream ss(line); std::string c; std::vector<std::string> v;
        while (std::getline(ss, c, ',')) v.push_back(c);
        if (v.size() < 10) continue;
        Row r; int cust = std::stoi(v[1]);
        r.wh = std::llround(std::stod(v[2]) * 1000.0);   // kWh with 3 decimals -> exact Wh
        r.ts = {std::stoi(v[8]), std::stoi(v[7]), std::stoi(v[6]), std::stoi(v[3]), std::stoi(v[4]), std::stoi(v[5])};
        r.label = std::stoi(v[9]);
        by[cust].push_back(r);
    }
    return by;
}

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    std::string csv = argc > 2 ? argv[2] : "Anomaly labelled dataset.csv";
    std::string which = argc > 3 ? argv[3] : "7";
    const double K = argc > 4 ? std::stod(argv[4]) : 3.0;
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    const bool b1 = party == ALICE;
    auto data = load(csv);
    he::Params hp; hp.mask_range = std::pow(2.0, 50);    // masks on cross terms of 2^40-sized products
    if (b1) printf("%-6s %7s | %8s %8s %8s %8s | %7s %7s %7s | %6s %6s %6s %6s %6s %6s\n", "cust", "N", "fit ms", "CKKS ms", "GC ms", "MiB",
                   "agree", "flagB", "flagR", "TP", "FP", "FN", "prec", "recall", "Gmean");
    long tot_tp = 0, tot_fp = 0, tot_fn = 0, tot_tn = 0;
    for (auto& [cust, rows] : data) {
        if (which != "all" && std::to_string(cust) != which) continue;
        const size_t N = rows.size();
        std::vector<prophet_lite::Stamp> ts(N); for (size_t i = 0; i < N; i++) ts[i] = rows[i].ts;
        // ---- harness: additive shares y = y1 + y2 (B1's share uniform in [-2^20, 2^20] Wh) ----
        std::mt19937_64 g(1000 + cust);
        std::uniform_int_distribution<int64_t> mask(-(1 << 20), 1 << 20);
        std::vector<int64_t> y(N);
        for (size_t i = 0; i < N; i++) { int64_t r = mask(g); y[i] = b1 ? r : rows[i].wh - r; }
        uint64_t c0 = io->send_counter + io->recv_counter;

        // ---- 1. model fit on shares (local) ----
        auto t1 = clk::now();
        prophet_lite::Model M; M.build(ts);
        std::vector<double> yd(y.begin(), y.end());
        auto yh = M.predict(M.fit(yd));
        std::vector<double> e(N); for (size_t i = 0; i < N; i++) e[i] = yd[i] - yh[i];
        double fit_ms = ms_since(t1);

        // ---- 2. residual variance via CKKS cross terms: share of sum (e1 + e2)^2 ----
        auto t2 = clk::now();
        double sse = 0;
        const size_t C = 4096;
        for (size_t off = 0; off < N; off += C) {
            size_t n = std::min(C, N - off);
            std::vector<double> chunk(e.begin() + off, e.begin() + off + n);
            sse += he::inner_products(io.get(), party, chunk, chunk, n, hp)[0];
        }
        double var_share = sse / N;
        double ckks_ms = ms_since(t2);

        // ---- 3. GC: bound = 3 sigma, per-reading decision ----
        auto t3 = clk::now();
        std::vector<int64_t> vq = {std::llround(var_share * Q * Q)};
        auto V = batch::input_shared(s, vq, 1)[0];
        auto sd = batch::isqrt(V);
        // bound = K * sigma, K as a public fixed-point constant (x256)
        auto bound = (sd * batch::U64::constant(s.ctx(), (uint64_t)std::llround(K * 256))) >> 8;
        auto& c = s.ctx();
        batch::U64 zero = batch::U64::constant(c, 0);
        std::vector<uint8_t> flags(N);
        for (size_t off = 0; off < N; off += C) {
            size_t n = std::min(C, N - off);
            std::vector<int64_t> d(n), yy(y.begin() + off, y.begin() + off + n);
            for (size_t i = 0; i < n; i++) d[i] = std::llround(e[off + i] * Q);
            auto D = batch::input_shared(s, d, n), Y = batch::input_shared(s, yy, n);
            std::vector<batch::Bit> r; r.reserve(n);
            for (size_t i = 0; i < n; i++) r.push_back((batch::abs64(D[i]) > bound) | (Y[i] == zero));
            auto sh = batch::reveal(s, r, emp::XOR);
            std::copy(sh.begin(), sh.end(), flags.begin() + off);
        }
        io->flush();
        double gc_ms = ms_since(t3);
        double mib = (io->send_counter + io->recv_counter - c0) / 1048576.0;

        // ---- harness: B2 hands its output shares to B1 for checking ----
        if (!b1) { io->send_data(flags.data(), N); io->flush(); continue; }
        std::vector<uint8_t> other(N); io->recv_data(other.data(), N);
        // reference: the same model in the clear
        std::vector<double> yc(N); for (size_t i = 0; i < N; i++) yc[i] = (double)rows[i].wh;
        auto yhc = M.predict(M.fit(yc));
        double v = 0; for (size_t i = 0; i < N; i++) v += (yc[i] - yhc[i]) * (yc[i] - yhc[i]);
        double sdc = std::sqrt(v / N);
        long agree = 0, flagB = 0, flagR = 0, tp = 0, fp = 0, fn = 0, tn = 0;
        for (size_t i = 0; i < N; i++) {
            int fb = (flags[i] ^ other[i]) & 1;
            int fr = (std::fabs(yc[i] - yhc[i]) > K * sdc) || rows[i].wh == 0;
            agree += fb == fr; flagB += fb; flagR += fr;
            int lab = rows[i].label;
            tp += fb && lab; fp += fb && !lab; fn += !fb && lab; tn += !fb && !lab;
        }
        tot_tp += tp; tot_fp += fp; tot_fn += fn; tot_tn += tn;
        double prec = tp + fp ? (double)tp / (tp + fp) : 0, rec = tp + fn ? (double)tp / (tp + fn) : 0, spec = tn + fp ? (double)tn / (tn + fp) : 0;
        printf("%-6d %7zu | %8.0f %8.0f %8.0f %8.1f | %6.2f%% %7ld %7ld | %6ld %6ld %6ld %6.3f %6.3f %6.3f\n", cust, N, fit_ms, ckks_ms, gc_ms, mib,
               100.0 * agree / N, flagB, flagR, tp, fp, fn, prec, rec, std::sqrt(rec * spec));
        fflush(stdout);
    }
    if (b1 && which == "all") {
        double prec = (double)tot_tp / (tot_tp + tot_fp), rec = (double)tot_tp / (tot_tp + tot_fn), spec = (double)tot_tn / (tot_tn + tot_fp);
        printf("all customers, k = %.3f, vs Jaiswal et al. labels: precision %.3f, recall %.3f, G-mean %.3f\n", K, prec, rec, std::sqrt(rec * spec));
    }
    return 0;
}
