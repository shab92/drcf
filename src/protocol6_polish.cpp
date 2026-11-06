// DRCF Table A1, Protocol VI: Pi_ML-PriceForecast on the Polish day-ahead market, with the SVRARIMA model of
// Che & Wang (Energy Conversion and Management 51(10), 2010). Model trained by scripts/train_protocol6.py.
//
//   ./run.sh build/protocol6_polish prices_p6.txt model_p6.txt reference_p6.txt
//
// Two-party flow (B1 = ALICE, B2 = BOB). Inputs: additive shares of hourly prices (cents) and of the model
// (support vectors, alpha, b, normalisation min/range, ARIMA weights, owner's training residuals).
//   GC   : x_norm = (x - min) / range                                    (Eq. 3.1)
//          N_t   = b + sum_k alpha_k exp(-||x_t - sv_k||^2 / (2 delta^2)), x_t = (x_{t-1}, x_{t-2}, x_{t-3})   (Eq. 2.6, 2.9)
//          e_t   = x_norm_t - N_t                                         (Eq. 2.12)
//   CKKS : L_t   = c' + sum_j w_j e_{t-j}   (ARIMA on residuals, one-step predictor; HE cross terms, Table A1)
//   GC   : price_t = (N_t + L_t) * range + min                            (Eq. 2.11)
// Output: additive shares of the 168 hourly forecasts per week.
// Harness only: the model owner's and publishers' shares are dealt from a shared seed; at the end B2 sends its
// output shares to B1, which compares them with the plaintext reference and with the actual prices.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include "he.h"
#include "svr_gc.h"

using namespace emp;
using namespace svr_gc;
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct WeekModel {
    std::string start; int type = 0; long t0 = 0, tr0 = 0;
    double lo = 0, hi = 0, gamma = 0, b = 0;
    std::vector<double> alpha; std::vector<std::array<double, 3>> sv;
    int L = 0; double cp = 0; std::vector<double> w, resid;
};

static std::vector<WeekModel> load_models(const std::string& path) {
    std::ifstream f(path); std::string tag; std::vector<WeekModel> out;
    while (f >> tag) {
        if (tag == "WEEK") { out.emplace_back(); auto& m = out.back(); f >> m.start >> m.type >> m.t0 >> m.tr0; }
        else if (tag == "NORM") f >> out.back().lo >> out.back().hi;
        else if (tag == "SVR") {
            auto& m = out.back(); int n; f >> m.gamma >> m.b >> n;
            m.alpha.resize(n); m.sv.resize(n);
            for (int i = 0; i < n; i++) f >> m.alpha[i] >> m.sv[i][0] >> m.sv[i][1] >> m.sv[i][2];
        } else if (tag == "ARIMA") { auto& m = out.back(); f >> m.L >> m.cp; m.w.resize(m.L); for (auto& v : m.w) f >> v; }
        else if (tag == "RESID") { auto& m = out.back(); int n; f >> n; m.resid.resize(n); for (auto& v : m.resid) f >> v; }
    }
    return out;
}

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    std::string pf = argc > 2 ? argv[2] : "prices_p6.txt", mf = argc > 3 ? argv[3] : "model_p6.txt", rf = argc > 4 ? argv[4] : "reference_p6.txt";
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    const bool b1 = party == ALICE;
    std::vector<double> price; { std::ifstream f(pf); double v; while (f >> v) price.push_back(v); }
    auto models = load_models(mf);
    std::ifstream rfs(rf);
    if (b1) printf("%-11s %4s | %8s %7s %8s %9s %9s | %9s %7s | %9s %7s | %8s\n", "week", "nSV", "GC1 ms", "CKKS ms", "GC2 ms", "AND (M)", "MiB",
                   "RMSE", "MAPE%", "ref RMSE", "MAPE%", "max|d|");
    for (auto& M : models) {
        const long t0 = M.t0; const int L = M.L, H = 168, nsv = (int)M.alpha.size();
        std::mt19937_64 g(std::hash<std::string>{}(M.start));
        auto split = [&](int64_t v, int bits) { std::uniform_int_distribution<int64_t> d(-(int64_t(1) << bits), int64_t(1) << bits); int64_t r = d(g); return b1 ? r : v - r; };
        auto splitd = [&](double v, double range) { std::uniform_real_distribution<double> d(-range, range); double r = d(g); return b1 ? r : v - r; };
        auto q14 = [](double v) { return (int64_t)std::llround(v * ONE); };
        // ---- harness: shares from the model owner and the publishers ----
        std::vector<int64_t> m_in;   // sv (3 per SV), alpha, b, min, range (cents)
        for (int k = 0; k < nsv; k++) for (int c = 0; c < 3; c++) m_in.push_back(split(q14(M.sv[k][c]), 20));
        for (int k = 0; k < nsv; k++) m_in.push_back(split(q14(M.alpha[k]), 20));
        m_in.push_back(split(q14(M.b), 20));
        const int64_t lo_c = std::llround(M.lo * 100), rg_c = std::llround((M.hi - M.lo) * 100);
        m_in.push_back(split(lo_c, 30)); m_in.push_back(split(rg_c, 30));
        std::vector<int64_t> x_in;   // prices (cents) for hours t0-3 .. t0+H-1
        for (long t = t0 - 3; t < t0 + H; t++) x_in.push_back(split(std::llround(price[t] * 100), 30));
        std::vector<double> w_sh(L), r_sh(L);
        for (int j = 0; j < L; j++) w_sh[j] = splitd(M.w[j], 16.0);
        for (int j = 0; j < L; j++) r_sh[j] = (double)split(q14(M.resid[j]), 28) / ONE;   // owner's residuals e_{t0-L} .. e_{t0-1}
        double cp_sh = splitd(M.cp, 16.0);
        uint64_t c0 = io->send_counter + io->recv_counter, a0 = s.num_and();

        // ---- GC phase 1: normalise, SVR, residuals ----
        auto t1 = clk::now();
        auto mv = input_shared(s, m_in);
        auto xv = input_shared(s, x_in);
        auto& c = s.ctx();
        svr_gc::Model SM; SM.gc = (uint64_t)std::llround(M.gamma * std::log2(std::exp(1.0)) * 4294967296.0);
        for (int k = 0; k < nsv; k++) SM.sv.push_back({low32(mv[3 * k]), low32(mv[3 * k + 1]), low32(mv[3 * k + 2])});
        for (int k = 0; k < nsv; k++) SM.alpha.push_back(low32(mv[3 * nsv + k]));
        SM.b = low32(mv[4 * nsv]);
        I64 LO = mv[4 * nsv + 1], RG = mv[4 * nsv + 2];
        std::vector<I32> xn;
        for (auto& x : xv) xn.push_back(clamp(low32(((x - LO) << FRAC) / RG), -(7 << FRAC), 7 << FRAC));
        std::vector<I32> Nt; std::vector<I64> et;
        for (int h = 0; h < H; h++) {
            int i = h + 3;                                         // index of hour t0+h in xn
            I32 n = eval(SM, {xn[i - 1], xn[i - 2], xn[i - 3]});
            Nt.push_back(n); et.push_back(sext64(xn[i] - n));
        }
        auto e_sh = to_additive(s, et, 28);
        double gc1 = ms_since(t1);

        // ---- CKKS: ARIMA one-step predictor on residuals ----
        auto t2 = clk::now();
        std::vector<double> ehist(r_sh);                           // e_{t0-L} ... e_{t0+H-1}
        for (auto v : e_sh) ehist.push_back((double)v / ONE);
        const int per = (int)(4096 / L);
        std::vector<double> Lt;
        for (int h0 = 0; h0 < H; h0 += per) {
            int n = std::min(per, H - h0);
            std::vector<double> x, wv;
            for (int h = h0; h < h0 + n; h++) for (int j = 1; j <= L; j++) { x.push_back(ehist[L + h - j]); wv.push_back(w_sh[j - 1]); }
            auto part = he::inner_products(io.get(), party, x, wv, L);
            for (auto v : part) Lt.push_back(v + cp_sh);
        }
        double ck = ms_since(t2);

        // ---- GC phase 2: forecast = (N + L) * range + min ----
        auto t3 = clk::now();
        std::vector<int64_t> lq; for (auto v : Lt) lq.push_back(std::llround(v * ONE));
        auto LV = input_shared(s, lq);
        std::vector<I64> pc;
        for (int h = 0; h < H; h++) pc.push_back((((sext64(Nt[h]) + LV[h]) * RG) >> FRAC) + LO);
        auto out = to_additive(s, pc, 40);
        io->flush();
        double gc2 = ms_since(t3);
        double mib = (io->send_counter + io->recv_counter - c0) / 1048576.0, ands = (s.num_and() - a0) / 1e6;

        // ---- harness: check ----
        std::string tag, wk; rfs >> tag >> wk;
        std::vector<double> act(H), rs(H), rh(H);
        for (int h = 0; h < H; h++) rfs >> act[h] >> rs[h] >> rh[h];
        if (!b1) { io->send_data(out.data(), 8 * H); io->flush(); continue; }
        std::vector<int64_t> o2(H); io->recv_data(o2.data(), 8 * H);
        double se = 0, ape = 0, ser = 0, aper = 0, md = 0;
        for (int h = 0; h < H; h++) {
            double f = (double)(out[h] + o2[h]) / 100.0;
            se += (f - act[h]) * (f - act[h]); ape += std::fabs((f - act[h]) / act[h]);
            ser += (rh[h] - act[h]) * (rh[h] - act[h]); aper += std::fabs((rh[h] - act[h]) / act[h]);
            md = std::max(md, std::fabs(f - rh[h]));
        }
        printf("%-11s %4d | %8.0f %7.0f %8.0f %9.1f %9.1f | %9.3f %7.3f | %9.3f %7.3f | %8.3f\n", M.start.c_str(), nsv, gc1, ck, gc2, ands, mib,
               std::sqrt(se / H), 100 * ape / H, std::sqrt(ser / H), 100 * aper / H, md);
        fflush(stdout);
    }
    return 0;
}
