// DRCF Table A1, Protocol IV: Pi_ML-LoadForecast with the daily-peak SVR model of
// Chen, Chang & Lin (IEEE TPWRS 19(4), 2004). Model file written by drcf.py --protocol 4.
//
//   ./run.sh build/protocol4_daily model_p4.txt [result.txt]
//
// Two-party flow (B1 = ALICE, B2 = BOB). Inputs: additive shares of every client's hourly load for the 7 days before
// the forecast month (one share vector per publisher), and of the model (support vectors, alpha, b, normalisation min/range).
//   local: system load per hour = sum of the clients' shares (each broker adds its own shares)
//   GC   : daily maximum over the 24 hours of each day; x = (max - min) / range
//          31 days, recursively: N_d = b + sum_k alpha_k exp(-gamma ||f_d - sv_k||^2),
//          f_d = (x_{d-1}, ..., x_{d-7}, weekday bits, holiday bit); x_d = N_d feeds the next day
//          forecast_d = N_d * range + min
// Calendar bits are public. Output: additive shares of the 31 daily maxima, for the subscriber.
// Harness only: shares are dealt from a fixed seed; B2 sends its output shares to B1 for the check.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include "svr_gc.h"

using namespace emp;
using namespace svr_gc;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    std::string mf = argc > 2 ? argv[2] : "model_p4.txt";
    std::ifstream f(mf); std::string tag;
    double lo = 0, hi = 0, gamma = 0, b = 0, scale = 1; int nsv = 0, m = 14, ndays = 0, ncl = 0, nh = 0;
    std::vector<double> alpha, act, ref; std::vector<std::vector<double>> sv; std::vector<std::vector<int>> cal; std::vector<std::vector<int64_t>> hist;
    while (f >> tag) {
        if (tag == "NORM") f >> lo >> hi;
        else if (tag == "SCALE") f >> scale;   // HIST readings are integers at this scale
        else if (tag == "SVR") { f >> gamma >> b >> nsv; alpha.resize(nsv); sv.assign(nsv, std::vector<double>(m)); for (int k = 0; k < nsv; k++) { f >> alpha[k]; for (auto& v : sv[k]) f >> v; } }
        else if (tag == "CAL") { f >> ndays; cal.assign(ndays, std::vector<int>(7)); for (auto& r : cal) for (auto& v : r) f >> v; }
        else if (tag == "HIST") { f >> ncl >> nh; hist.assign(ncl, std::vector<int64_t>(nh)); for (auto& r : hist) for (auto& v : r) f >> v; }
        else if (tag == "ACT") { int n; f >> n; act.resize(n); for (auto& v : act) f >> v; }
        else if (tag == "REF") { int n; f >> n; ref.resize(n); for (auto& v : ref) f >> v; }
    }
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    const bool b1 = party == ALICE;
    std::mt19937_64 g(4044);
    auto split = [&](int64_t v, int bits) { std::uniform_int_distribution<int64_t> d(-(int64_t(1) << bits), int64_t(1) << bits); int64_t r = d(g); return b1 ? r : v - r; };
    auto q14 = [](double v) { return (int64_t)std::llround(v * ONE); };

    // ---- harness: shares from the model owner and the 321 publishers ----
    std::vector<int64_t> m_in;
    for (int k = 0; k < nsv; k++) for (int c = 0; c < m; c++) m_in.push_back(split(q14(sv[k][c]), 20));
    for (int k = 0; k < nsv; k++) m_in.push_back(split(q14(alpha[k]), 20));
    m_in.push_back(split(q14(b), 20));
    m_in.push_back(split(std::llround(lo * scale), 40)); m_in.push_back(split(std::llround((hi - lo) * scale), 40));
    std::vector<int64_t> load(nh, 0);                       // each broker: sum of its shares of the clients' readings
    for (int c = 0; c < ncl; c++) for (int h = 0; h < nh; h++) load[h] += split(hist[c][h], 30);

    auto t0 = clk::now(); uint64_t c0 = io->send_counter + io->recv_counter, a0 = s.num_and();
    auto mv = input_shared(s, m_in);
    auto lv = input_shared(s, load);
    auto& c = s.ctx();
    Model SM; SM.gc = (uint64_t)std::llround(gamma * std::log2(std::exp(1.0)) * 4294967296.0);
    for (int k = 0; k < nsv; k++) { std::vector<I32> r; for (int j = 0; j < m; j++) r.push_back(low32(mv[(size_t)m * k + j])); SM.sv.push_back(r); }
    for (int k = 0; k < nsv; k++) SM.alpha.push_back(low32(mv[(size_t)m * nsv + k]));
    SM.b = low32(mv[(size_t)(m + 1) * nsv]);
    I64 LO = mv[(size_t)(m + 1) * nsv + 1], RG = mv[(size_t)(m + 1) * nsv + 2];
    std::vector<I32> x;                                       // scaled daily maxima, oldest first
    for (int d = 0; d < nh / 24; d++) {
        I64 mx = lv[24 * d];
        for (int h = 1; h < 24; h++) mx = mx.select(lv[24 * d + h] > mx, lv[24 * d + h]);
        x.push_back(clamp(low32(((mx - LO) << FRAC) / RG), -(7 << FRAC), 7 << FRAC));
    }
    std::vector<I64> fc;
    for (int d = 0; d < ndays; d++) {
        std::vector<I32> feat;
        for (int j = 1; j <= 7; j++) feat.push_back(x[x.size() - j]);
        for (int j = 0; j < 7; j++) feat.push_back(I32::constant(c, cal[d][j] ? (int32_t)ONE : 0));
        I32 n = clamp(eval(SM, feat), -(7 << FRAC), 7 << FRAC);
        x.push_back(n);
        fc.push_back(((sext64(n) * RG) >> FRAC) + LO);
    }
    auto out = to_additive(s, fc, 40);
    io->flush();
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    double mib = (io->send_counter + io->recv_counter - c0) / 1048576.0, ands = (s.num_and() - a0) / 1e6;

    if (!b1) { io->send_data(out.data(), 8 * ndays); io->flush(); return 0; }
    std::vector<int64_t> o2(ndays); io->recv_data(o2.data(), 8 * ndays);
    if (argc > 3) {   // result file for drcf.py
        std::ofstream r(argv[3]); r << "COST " << ms << " " << ands * 1e6 << " " << mib << "\nOUT " << ndays << "\n";
        for (int d = 0; d < ndays; d++) r << (double)(out[d] + o2[d]) / scale << "\n";
    }
    if ((int)act.size() < ndays || (int)ref.size() < ndays) return 0;
    double ape = 0, aper = 0, md = 0, mr = 0;
    for (int d = 0; d < ndays; d++) {
        double v = (double)(out[d] + o2[d]) / scale;
        ape += std::fabs((v - act[d]) / act[d]); aper += std::fabs((ref[d] - act[d]) / act[d]);
        md = std::max(md, std::fabs(v - ref[d])); mr = std::max(mr, std::fabs(v - ref[d]) / ref[d]);
    }
    printf("Protocol IV: %d clients, %d SVs, %d-day recursive forecast of the daily maximum load\n", ncl, nsv, ndays);
    printf("cost: %.0f ms, %.1f M AND gates, %.1f MiB\n", ms, ands, mib);
    printf("secure MAPE %.3f%%   plaintext reference MAPE %.3f%%   max |secure - reference| %.0f kW (%.3f%%)\n",
           100 * ape / ndays, 100 * aper / ndays, md, 100 * mr);
    return 0;
}
