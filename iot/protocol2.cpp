// Protocol II (Table A1): ML-based anomaly detection on Ausgrid smart-meter data, two brokers.
//   ./run.sh build/protocol2 iot/data/p2_c7            (add --no-aead to skip the DistAEAD step)
//
// Model: Prophet as configured by Jaiswal et al. 2021 (see iot/p2_prophet_prep.py). With Gaussian priors
// the fit is linear in the readings, w = A y, so it runs on shares:
//   1. Publish: each half-hourly reading is ASCON-encrypted by its meter; DistAEAD.Dec gives XOR shares.
//   2. B2A (GC): XOR shares -> additive shares y = y1 + y2.
//   3. Local: w_i = A y_i, yhat_i = P w_i, r_i = y_i - yhat_i     (A and P depend only on timestamps)
//   4. CKKS cross terms: shares of r^2 per point (r1*r2 encrypted, as in Table A1), sigma^2 = mean(r^2).
//   5. GC: anomaly = [ r^2 > z^2 sigma^2 ] OR [ y == 0 ]  (outside the 99% interval, or a zero reading).
// The harness (not part of the protocol) simulates the meters inside both processes and, at the end,
// sends B2's output shares to B1 only to check them against the plaintext model and the paper's labels.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include "ascon_ref.h"
#include "convert.h"
#include "distaead.h"
#include "he.h"

using namespace emp;
using clk = std::chrono::steady_clock;
using bytes = std::vector<uint8_t>;

template <class T> static std::vector<T> load(const std::string& f, size_t n) {
    std::vector<T> v(n); std::ifstream in(f, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(v.data()), n * sizeof(T))) { fprintf(stderr, "cannot read %s\n", f.c_str()); exit(1); }
    return v;
}
static std::string meta(const std::string& dir, const std::string& key) {
    std::ifstream in(dir + "/meta.txt"); std::string k, v;
    while (in >> k) { std::getline(in, v); if (k == key) return v.substr(1); }
    fprintf(stderr, "meta.txt: missing %s\n", key.c_str()); exit(1);
}

struct Stage { std::string name; double ms; double mib; };

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    std::string dir = argc > 2 ? argv[2] : "iot/data/p2_c7";
    bool aead = !(argc > 3 && std::string(argv[3]) == "--no-aead");
    const bool b1 = party == ALICE;
    const size_t N = std::stoul(meta(dir, "N")), K = std::stoul(meta(dir, "K"));
    const double z = std::stod(meta(dir, "z"));
    auto y = load<int32_t>(dir + "/readings.i32", N);           // harness: the meters' readings
    auto P = load<double>(dir + "/design.f64", N * K);          // public: depends on timestamps only
    auto A = load<double>(dir + "/A.f64", K * N);

    auto io = b1 ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    std::vector<Stage> stages;
    auto t0 = clk::now(); uint64_t c0 = io->send_counter + io->recv_counter;
    auto mark = [&](const std::string& n) {
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        double mib = (io->send_counter + io->recv_counter - c0) / 1048576.0;
        stages.push_back({n, ms, mib}); t0 = clk::now(); c0 = io->send_counter + io->recv_counter;
    };

    // ---- 1. Publish + DistAEAD.Dec ----
    std::mt19937_64 g(7);                                       // harness: shared randomness for the simulated meter
    bytes K_meter(16), K1(16); for (auto& b : K_meter) b = (uint8_t)g(); for (auto& b : K1) b = (uint8_t)g();
    bytes K2(16); for (int i = 0; i < 16; i++) K2[i] = K_meter[i] ^ K1[i];
    std::vector<uint32_t> xs(N);
    size_t tags_ok = 0;
    for (size_t i = 0; i < N; i++) {
        if (aead) {
            bytes nonce(16); for (auto& b : nonce) b = (uint8_t)g();
            std::string ads = "ausgrid/" + meta(dir, "customer") + "/" + std::to_string(i);
            bytes ad(ads.begin(), ads.end()), m(4);
            for (int k = 0; k < 4; k++) m[k] = (uint8_t)((uint32_t)y[i] >> (8 * k));
            bytes ct = ascon::encrypt(K_meter.data(), nonce.data(), ad, m);
            auto d = distaead::dist_dec(s, b1 ? K1 : K2, nonce.data(), ad, ct);
            tags_ok += d.ok;
            xs[i] = d.share[0] | d.share[1] << 8 | d.share[2] << 16 | (uint32_t)d.share[3] << 24;
        } else {                                                 // --no-aead: hand out XOR shares directly
            uint32_t m = (uint32_t)g(); xs[i] = b1 ? m : ((uint32_t)y[i] ^ m); tags_ok++;
        }
    }
    mark(aead ? "DistAEAD.Dec (ASCON in GC)" : "XOR shares (DistAEAD skipped)");

    // ---- 2. B2A ----
    auto ya = convert::b2a(s, xs, 20);
    mark("B2A (GC)");

    // ---- 3. local linear fit on shares ----
    std::vector<double> yi(ya.begin(), ya.end()), w(K, 0.0), r(N);
    for (size_t k = 0; k < K; k++) { double acc = 0; const double* a = &A[k * N]; for (size_t n = 0; n < N; n++) acc += a[n] * yi[n]; w[k] = acc; }
    for (size_t n = 0; n < N; n++) { double acc = 0; const double* p = &P[n * K]; for (size_t k = 0; k < K; k++) acc += p[k] * w[k]; r[n] = yi[n] - acc; }
    mark("Local fit w = A·y, residuals");

    // ---- 4. CKKS: shares of r^2, sigma^2 ----
    auto rsq = he::products(io.get(), party, r, r);
    auto hs = he::last;
    double s2 = 0; for (double v : rsq) s2 += v; s2 /= N;
    mark("CKKS cross terms r1·r2");

    // ---- 5. GC decision ----
    std::vector<int64_t> a(N), b(N);
    for (size_t n = 0; n < N; n++) { a[n] = std::llround(rsq[n]); b[n] = std::llround(z * z * s2); }
    auto bits = convert::greater_or_zero(s, a, b, ya);
    mark("GC: r² > z²σ² or y = 0");

    // ---- harness: check ----
    if (!b1) {
        io->send_data(bits.data(), N); io->send_data(rsq.data(), 8 * N); io->send_data(&s2, 8); io->flush();
        return 0;
    }
    std::vector<uint8_t> ob(N); std::vector<double> orsq(N); double os2;
    io->recv_data(ob.data(), N); io->recv_data(orsq.data(), 8 * N); io->recv_data(&os2, 8);
    auto ref = load<uint8_t>(dir + "/ref_anomaly.u8", N), lab = load<uint8_t>(dir + "/paper_label.u8", N);
    // plaintext model for the error check
    std::vector<double> wc(K, 0.0); double sc = 0, maxe = 0;
    for (size_t k = 0; k < K; k++) { double acc = 0; for (size_t n = 0; n < N; n++) acc += A[k * N + n] * y[n]; wc[k] = acc; }
    for (size_t n = 0; n < N; n++) {
        double acc = 0; for (size_t k = 0; k < K; k++) acc += P[n * K + k] * wc[k];
        double rc = y[n] - acc; sc += rc * rc;
        maxe = std::max(maxe, std::fabs(rsq[n] + orsq[n] - rc * rc));
    }
    sc /= N;
    size_t same_ref = 0, same_lab = 0, tp = 0, fp = 0, fn = 0, flagged = 0;
    for (size_t n = 0; n < N; n++) {
        uint8_t v = (bits[n] ^ ob[n]) & 1; flagged += v;
        same_ref += v == ref[n]; same_lab += v == lab[n];
        tp += v && lab[n]; fp += v && !lab[n]; fn += !v && lab[n];
    }
    double total_ms = 0, total_mib = 0;
    printf("Protocol II, %s: %zu half-hourly readings (%s to %s), %zu model terms\n", meta(dir, "customer").c_str(), N,
           meta(dir, "start").c_str(), meta(dir, "end").c_str(), K);
    for (auto& st : stages) { printf("  %-32s %9.1f ms %9.1f MiB\n", st.name.c_str(), st.ms, st.mib); total_ms += st.ms; total_mib += st.mib; }
    printf("  %-32s %9.1f ms %9.1f MiB\n", "total", total_ms, total_mib);
    printf("  tags valid %zu/%zu | CKKS: fresh ct %zu B for %zu slots | max |r² error| %.3g Wh², sigma secure %.3f vs plain %.3f Wh\n",
           tags_ok, N, hs.fresh_ct_bytes, hs.slots, maxe, std::sqrt(s2 + os2), std::sqrt(sc));
    printf("  secure vs plaintext model: %zu/%zu identical (%zu flagged)\n", same_ref, N, flagged);
    printf("  secure vs paper labels:    agreement %.4f, precision %.3f, recall %.3f\n", (double)same_lab / N,
           (double)tp / std::max<size_t>(tp + fp, 1), (double)tp / std::max<size_t>(tp + fn, 1));
    std::ofstream o(dir + "/secure_anomaly.u8", std::ios::binary);
    for (size_t n = 0; n < N; n++) o.put((char)((bits[n] ^ ob[n]) & 1));
    std::ofstream st(dir + "/secure_stats.txt");
    for (auto& x : stages) st << x.name << "\t" << x.ms << "\t" << x.mib << "\n";
    st << "same_ref\t" << same_ref << "\nsame_label\t" << same_lab << "\nN\t" << N << "\ntp\t" << tp << "\nfp\t" << fp << "\nfn\t" << fn << "\n";
    return same_ref == N ? 0 : 2;
}
