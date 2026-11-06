// Generic DRCF.Compute engine for the prototype (drcf.py). Runs one job: a series, a task and a trained model.
//
//   ./run.sh build/drcf_secure job.txt result.txt
//
// Model families (covering Table A1):
//   linear  : y_hat_t = sum_j beta_j y_{t-l_j} + sum_i c_i u_i(t)     secret lags x shared beta via CKKS cross terms,
//             public features u (intercept, calendar, Fourier, weather) x shared c locally        (Protocols I, III, V)
//   prophet : additive trend + seasonality fitted on each broker's share (linear), sigma via CKKS, bound in GC (Protocol II)
//   svr     : RBF SVR on lagged, min-max normalised values in a garbled circuit, optional ARIMA on its residuals
//             via CKKS cross terms                                                                     (Protocols IV, VI)
// Tasks: forecast (output = shares of y_hat) or anomaly (output = shares of [ |y - y_hat| > threshold ]).
// The job file is plain text written by drcf.py; result.txt is written by B1 after B2 hands over its output
// shares (harness only, for the report).
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
#include "svr_gc.h"

using namespace emp;
using clk = std::chrono::steady_clock;

struct Job {
    std::string task, model;
    double scale = 100;                  // values * scale -> integers in the GC
    std::vector<double> y;
    long start = 0, count = 0;
    std::vector<int> lags; std::vector<double> coef;              // linear
    int rowk = 0; std::vector<int> rowlag; std::vector<double> rowcoef, rowc;   // linear, one row per test point
    int npub = 0; std::vector<double> pubrows, pubcoef;
    double thresh = 0;                                            // anomaly (absolute, value units)
    std::vector<prophet_lite::Stamp> stamps; double k = 3; bool zero_rule = false, au = false; double yscale = 0;   // prophet
    double lo = 0, hi = 1, gamma = 0, b = 0; std::vector<double> alpha; std::vector<std::vector<double>> sv;      // svr
    int L = 0; double cp = 0; std::vector<double> w, resid;
};

static Job load(const std::string& path) {
    std::ifstream f(path); Job j; std::string tag;
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    while (f >> tag) {
        if (tag[0] == '#') { std::string rest; std::getline(f, rest); }
        else if (tag == "TASK") f >> j.task;
        else if (tag == "MODEL") f >> j.model;
        else if (tag == "SCALE") f >> j.scale;
        else if (tag == "SERIES") { long n; f >> n; j.y.resize(n); for (auto& v : j.y) f >> v; }
        else if (tag == "TEST") f >> j.start >> j.count;
        else if (tag == "LAGS") { int n; f >> n; j.lags.resize(n); for (auto& v : j.lags) f >> v; }
        else if (tag == "COEF") { int n; f >> n; j.coef.resize(n); for (auto& v : j.coef) f >> v; }
        else if (tag == "PUBLIC") { f >> j.npub; j.pubcoef.resize(j.npub); for (auto& v : j.pubcoef) f >> v; j.pubrows.resize((size_t)j.npub * j.count); for (auto& v : j.pubrows) f >> v; }
        else if (tag == "THRESH") f >> j.thresh;
        else if (tag == "ROWS") {   // per test point: intercept, then k (lag, coef) pairs
            f >> j.rowk; j.rowc.resize(j.count); j.rowlag.resize((size_t)j.count * j.rowk); j.rowcoef.resize((size_t)j.count * j.rowk);
            for (long h = 0; h < j.count; h++) { f >> j.rowc[h]; for (int q = 0; q < j.rowk; q++) f >> j.rowlag[(size_t)h * j.rowk + q] >> j.rowcoef[(size_t)h * j.rowk + q]; }
        }
        else if (tag == "STAMPS") { long n; f >> n; j.stamps.resize(n); for (auto& s : j.stamps) f >> s.y >> s.mo >> s.d >> s.h >> s.mi >> s.dow; }
        else if (tag == "PROPHET") { int z, a; f >> j.k >> z >> a >> j.yscale; j.zero_rule = z; j.au = a; }
        else if (tag == "NORM") f >> j.lo >> j.hi;
        else if (tag == "SVR") { int n; f >> j.gamma >> j.b >> n; j.alpha.resize(n); j.sv.assign(n, std::vector<double>(j.lags.size())); for (int i = 0; i < n; i++) { f >> j.alpha[i]; for (auto& v : j.sv[i]) f >> v; } }
        else if (tag == "ARIMA") { f >> j.L >> j.cp; j.w.resize(j.L); for (auto& v : j.w) f >> v; }
        else if (tag == "RESID") { int n; f >> n; j.resid.resize(n); for (auto& v : j.resid) f >> v; }
        else { fprintf(stderr, "unknown tag %s\n", tag.c_str()); exit(1); }
    }
    return j;
}

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    if (argc < 4) { fprintf(stderr, "usage: drcf_secure <party> job.txt result.txt\n"); return 1; }
    Job J = load(argv[2]);
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    const bool b1 = party == ALICE;
    const long N = (long)J.y.size(), T0 = J.start, C = J.count;
    const bool anomaly = J.task == "anomaly";
    std::mt19937_64 g(424242);
    auto split_i = [&](int64_t v, int bits) { std::uniform_int_distribution<int64_t> d(-(int64_t(1) << bits), int64_t(1) << bits); int64_t r = d(g); return b1 ? r : v - r; };
    auto split_d = [&](double v, double range) { std::uniform_real_distribution<double> d(-range, range); double r = d(g); return b1 ? r : v - r; };
    // ---- harness: publishers' shares of the series (integers at J.scale) ----
    std::vector<int64_t> yi(N);
    for (long t = 0; t < N; t++) yi[t] = split_i(std::llround(J.y[t] * J.scale), 30);
    auto t_start = clk::now(); uint64_t c0 = io->send_counter + io->recv_counter, a0 = s.num_and();
    std::vector<int64_t> out_i;     // forecasts (x scale) or flags, this broker's additive/XOR share
    std::vector<int> ref_flags;      // prophet only: plaintext reference computed by B1 from the harness copy of the series
    bool out_xor = false;
    he::Params hp; hp.mask_range = std::pow(2.0, 40);

    // shared helper: anomaly decision in GC given additive int shares of y and y_hat (x scale)
    auto decide = [&](const std::vector<int64_t>& ys, const std::vector<int64_t>& yh, int64_t tau_share) {
        auto Y = batch::input_shared(s, ys, ys.size()), Yh = batch::input_shared(s, yh, yh.size());
        auto TAU = batch::input_shared(s, {tau_share}, 1)[0];
        std::vector<batch::Bit> r;
        for (size_t i = 0; i < ys.size(); i++) r.push_back(batch::abs64(Y[i] - Yh[i]) > TAU);
        auto f = batch::reveal(s, r, emp::XOR);
        return std::vector<int64_t>(f.begin(), f.end());
    };

    if (J.model == "linear" && J.rowk > 0) {
        // per-point linear predictors (e.g. multi-step ARIMA forecasts): y_hat_h = c_h + sum_q coef_hq * y(t_h - lag_hq)
        const int K = J.rowk;
        std::vector<double> x, bv, c_sh(C);
        for (long h = 0; h < C; h++) {
            c_sh[h] = split_d(J.rowc[h], 1 << 20);
            for (int q = 0; q < K; q++) { x.push_back((double)yi[T0 + h - J.rowlag[(size_t)h * K + q]] / J.scale); bv.push_back(split_d(J.rowcoef[(size_t)h * K + q], 16.0)); }
        }
        auto yh = he::inner_products(io.get(), party, x, bv, K, hp);
        std::vector<int64_t> yhq(C); for (long h = 0; h < C; h++) yhq[h] = std::llround((yh[h] + c_sh[h]) * J.scale);
        if (anomaly) {
            std::vector<int64_t> ys(yi.begin() + T0, yi.begin() + T0 + C);
            out_i = decide(ys, yhq, split_i(std::llround(J.thresh * J.scale), 30)); out_xor = true;
        } else out_i = yhq;
    } else if (J.model == "linear") {
        const int K = (int)J.lags.size();
        std::vector<double> beta_sh(K), pc_sh(J.npub);
        for (int j = 0; j < K; j++) beta_sh[j] = split_d(J.coef[j], 16.0);
        for (int j = 0; j < J.npub; j++) pc_sh[j] = split_d(J.pubcoef[j], 16.0);
        std::vector<double> yh(C, 0.0);
        if (K > 0) {
            std::vector<double> x, bv;
            for (long h = 0; h < C; h++) for (int j = 0; j < K; j++) { x.push_back((double)yi[T0 + h - J.lags[j]] / J.scale); bv.push_back(beta_sh[j]); }
            yh = he::inner_products(io.get(), party, x, bv, K, hp);
        }
        for (long h = 0; h < C; h++) for (int j = 0; j < J.npub; j++) yh[h] += J.pubrows[(size_t)h * J.npub + j] * pc_sh[j];
        std::vector<int64_t> yhq(C); for (long h = 0; h < C; h++) yhq[h] = std::llround(yh[h] * J.scale);
        if (anomaly) {
            std::vector<int64_t> ys(yi.begin() + T0, yi.begin() + T0 + C);
            out_i = decide(ys, yhq, split_i(std::llround(J.thresh * J.scale), 30)); out_xor = true;
        } else out_i = yhq;
    } else if (J.model == "prophet") {
        prophet_lite::Model M; M.cfg.au_holidays = J.au; if (J.yscale > 0) M.cfg.y_scale = J.yscale;
        M.build(J.stamps);
        std::vector<double> yd(N); for (long t = 0; t < N; t++) yd[t] = (double)yi[t] / J.scale;
        auto yh = M.predict(M.fit(yd));
        std::vector<double> e(N); for (long t = 0; t < N; t++) e[t] = yd[t] - yh[t];
        double sse = 0;
        for (long off = 0; off < N; off += 4096) {
            long n = std::min(4096L, N - off);
            std::vector<double> ch(e.begin() + off, e.begin() + off + n);
            sse += he::inner_products(io.get(), party, ch, ch, n, hp)[0];
        }
        const double Q = 16.0;
        auto V = batch::input_shared(s, {std::llround(sse / N * Q * Q * J.scale * J.scale)}, 1)[0];
        auto sd = batch::isqrt(V);                                                  // sigma * Q * scale
        auto bound = (sd * batch::U64::constant(s.ctx(), (uint64_t)std::llround(J.k * 256))) >> 8;
        std::vector<int64_t> d(C), ys(yi.begin() + T0, yi.begin() + T0 + C);
        for (long h = 0; h < C; h++) d[h] = std::llround(e[T0 + h] * Q * J.scale);
        auto D = batch::input_shared(s, d, C), Y = batch::input_shared(s, ys, C);
        batch::U64 zero = batch::U64::constant(s.ctx(), 0);
        std::vector<batch::Bit> r;
        for (long h = 0; h < C; h++) { auto f = batch::abs64(D[h]) > bound; r.push_back(J.zero_rule ? (f | (Y[h] == zero)) : f); }
        auto fl = batch::reveal(s, r, emp::XOR);
        out_i.assign(fl.begin(), fl.end()); out_xor = true;
        if (b1) {   // reference: same model on the clear series
            auto yhc = M.predict(M.fit(J.y)); double v = 0;
            for (long t = 0; t < N; t++) v += (J.y[t] - yhc[t]) * (J.y[t] - yhc[t]);
            double sdc = std::sqrt(v / N);
            for (long h = 0; h < C; h++) ref_flags.push_back(std::fabs(J.y[T0 + h] - yhc[T0 + h]) > J.k * sdc || (J.zero_rule && J.y[T0 + h] == 0));
        }
    } else if (J.model == "svr") {
        using namespace svr_gc;
        const int m = (int)J.lags.size(), nsv = (int)J.alpha.size();
        int maxlag = 0; for (int l : J.lags) maxlag = std::max(maxlag, l);
        auto q14 = [](double v) { return (int64_t)std::llround(v * ONE); };
        std::vector<int64_t> m_in;
        for (int k = 0; k < nsv; k++) for (int c = 0; c < m; c++) m_in.push_back(split_i(q14(J.sv[k][c]), 20));
        for (int k = 0; k < nsv; k++) m_in.push_back(split_i(q14(J.alpha[k]), 20));
        m_in.push_back(split_i(q14(J.b), 20));
        m_in.push_back(split_i(std::llround(J.lo * J.scale), 30)); m_in.push_back(split_i(std::llround((J.hi - J.lo) * J.scale), 30));
        std::vector<double> w_sh(J.L), r_sh(J.L);
        for (int j = 0; j < J.L; j++) w_sh[j] = split_d(J.w[j], 16.0);
        for (int j = 0; j < J.L; j++) r_sh[j] = (double)split_i(q14(J.resid[j]), 28) / ONE;
        double cp_sh = split_d(J.cp, 16.0);
        auto mv = input_shared(s, m_in);
        std::vector<int64_t> x_in(yi.begin() + T0 - maxlag, yi.begin() + T0 + C);
        auto xv = input_shared(s, x_in);
        svr_gc::Model SM; SM.gc = (uint64_t)std::llround(J.gamma * std::log2(std::exp(1.0)) * 4294967296.0);
        for (int k = 0; k < nsv; k++) { std::vector<I32> r; for (int c = 0; c < m; c++) r.push_back(low32(mv[(size_t)m * k + c])); SM.sv.push_back(r); }
        for (int k = 0; k < nsv; k++) SM.alpha.push_back(low32(mv[(size_t)m * nsv + k]));
        SM.b = low32(mv[(size_t)(m + 1) * nsv]);
        I64 LO = mv[(size_t)(m + 1) * nsv + 1], RG = mv[(size_t)(m + 1) * nsv + 2];
        std::vector<I32> xn;
        for (auto& x : xv) xn.push_back(clamp(low32(((x - LO) << FRAC) / RG), -(7 << FRAC), 7 << FRAC));
        std::vector<I32> Nt; std::vector<I64> et;
        for (long h = 0; h < C; h++) {
            long i = h + maxlag; std::vector<I32> x; for (int l : J.lags) x.push_back(xn[i - l]);
            I32 nv = eval(SM, x); Nt.push_back(nv); et.push_back(sext64(xn[i] - nv));
        }
        std::vector<I64> fc;
        if (J.L > 0) {
            auto e_sh = to_additive(s, et, 28);
            std::vector<double> eh(r_sh); for (auto v : e_sh) eh.push_back((double)v / ONE);
            const long per = std::max<long>(1, 4096 / J.L);
            std::vector<double> Lt;
            for (long h0 = 0; h0 < C; h0 += per) {
                long n = std::min(per, C - h0); std::vector<double> x, wv;
                for (long h = h0; h < h0 + n; h++) for (int j = 1; j <= J.L; j++) { x.push_back(eh[J.L + h - j]); wv.push_back(w_sh[j - 1]); }
                auto part = he::inner_products(io.get(), party, x, wv, J.L);
                for (auto v : part) Lt.push_back(v + cp_sh);
            }
            std::vector<int64_t> lq; for (auto v : Lt) lq.push_back(std::llround(v * ONE));
            auto LV = input_shared(s, lq);
            for (long h = 0; h < C; h++) fc.push_back((((sext64(Nt[h]) + LV[h]) * RG) >> FRAC) + LO);
        } else for (long h = 0; h < C; h++) fc.push_back(((sext64(Nt[h]) * RG) >> FRAC) + LO);
        if (anomaly) {
            std::vector<int64_t> ys(yi.begin() + T0, yi.begin() + T0 + C);
            auto Y = batch::input_shared(s, ys, C);
            auto TAU = batch::input_shared(s, {split_i(std::llround(J.thresh * J.scale), 30)}, 1)[0];
            std::vector<batch::Bit> r;
            for (long h = 0; h < C; h++) r.push_back(batch::abs64(Y[h] - fc[h].as_unsigned()) > TAU);
            auto f = batch::reveal(s, r, emp::XOR); out_i.assign(f.begin(), f.end()); out_xor = true;
        } else out_i = to_additive(s, fc, 40);
    } else { fprintf(stderr, "unknown model %s\n", J.model.c_str()); return 1; }

    io->flush();
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t_start).count();
    double mib = (io->send_counter + io->recv_counter - c0) / 1048576.0, ands = (double)(s.num_and() - a0);
    // ---- harness: B2 hands its output shares to B1, which writes the result file ----
    if (!b1) { io->send_data(out_i.data(), 8 * out_i.size()); io->flush(); return 0; }
    std::vector<int64_t> o2(out_i.size()); io->recv_data(o2.data(), 8 * o2.size());
    std::ofstream f(argv[3]);
    f << "COST " << ms << " " << ands << " " << mib << "\nOUT " << out_i.size() << "\n";
    for (size_t i = 0; i < out_i.size(); i++) {
        if (out_xor) f << ((out_i[i] ^ o2[i]) & 1) << "\n";
        else f << (double)(out_i[i] + o2[i]) / J.scale << "\n";
    }
    if (!ref_flags.empty()) { f << "REF " << ref_flags.size() << "\n"; for (int v : ref_flags) f << v << "\n"; }
    return 0;
}
