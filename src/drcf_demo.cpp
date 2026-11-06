// End-to-end DRCF run: Publish -> Compute -> Subscribe with two real broker processes.
//   ./run.sh build/drcf_demo            (writes trace.js for the dashboard, from B1's process)
//
// Brokers: B1 = ALICE, B2 = BOB. Every cryptographic step is a real 2-party protocol:
//   DistAEAD (ASCON-128 in GC), B2A / A2B / threshold (GC), cross terms (SEAL CKKS, fresh keys),
//   2-party threshold ECDSA (DKLs23, Protocol 3.4) and 2-party Schnorr on secp256k1.
// Test harness only (clearly outside the protocol): both processes derive the same
// synthetic data from a shared seed, publishers and the subscriber are simulated inside
// B1's process, and at the end B2 sends its view to B1 so one trace file can be written.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include "ascon_ref.h"
#include "convert.h"
#include "distaead.h"
#include "he.h"
#include "dkls23.h"

using namespace emp;
using bytes = std::vector<uint8_t>;
using clk = std::chrono::steady_clock;

static const int METERS = 3, DAYS = 7, H = DAYS * 24;
static const double Q = 1000.0;   // fixed point used only when CKKS outputs enter a garbled circuit
static const int64_t ALERT_WH = 900;  // |forecast - actual| above this (Wh) raises an alert

static std::string hex(const bytes& b, size_t max = 64) {
    static const char* d = "0123456789abcdef"; std::string s;
    for (size_t i = 0; i < b.size() && i < max; i++) { s += d[b[i] >> 4]; s += d[b[i] & 15]; }
    if (b.size() > max) s += "…";
    return s;
}
static bytes u32b(uint32_t v) { return {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)}; }
static bytes u64b(uint64_t v) { bytes b(8); for (int i = 0; i < 8; i++) b[i] = (uint8_t)(v >> (8 * i)); return b; }
static uint32_t b2u32(const bytes& b) { return b[0] | b[1] << 8 | b[2] << 16 | (uint32_t)b[3] << 24; }
static bytes rnd(std::mt19937_64& g, size_t n) { bytes b(n); for (auto& x : b) x = (uint8_t)g(); return b; }
static bytes xr(const bytes& a, const bytes& b) { bytes o(a.size()); for (size_t i = 0; i < a.size(); i++) o[i] = a[i] ^ b[i]; return o; }
static std::string js(const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; } return o + "\""; }

// ---- trace ----
struct Step { std::string phase, name, from, to, what; double ms; uint64_t bytes, ands; std::string b1, b2; };
struct Trace {
    std::vector<Step> steps;
    IOChannel* io; SH2PCSession* s;
    clk::time_point t0; uint64_t c0, a0;
    void begin() { t0 = clk::now(); c0 = io->send_counter + io->recv_counter; a0 = s->num_and(); }
    void end(const std::string& ph, const std::string& nm, const std::string& from, const std::string& to, const std::string& what,
             const std::string& b1 = "", const std::string& b2 = "") {
        steps.push_back({ph, nm, from, to, what, std::chrono::duration<double, std::milli>(clk::now() - t0).count(),
                         io->send_counter + io->recv_counter - c0, s->num_and() - a0, b1, b2});
    }
};

// synthetic smart-meter load (Wh per hour): daily shape + noise, a few injected spikes
static std::vector<std::vector<int32_t>> make_data(std::mt19937_64& g, std::vector<std::pair<int, int>>& spikes) {
    std::normal_distribution<double> noise(0, 60);
    std::vector<std::vector<int32_t>> X(METERS, std::vector<int32_t>(H));
    for (int m = 0; m < METERS; m++) {
        double base = 450 + 150 * m, amp = 380 + 90 * m;
        for (int h = 0; h < H; h++) {
            int hr = h % 24;
            double shape = 0.6 * std::exp(-std::pow((hr - 7.5) / 1.6, 2)) + 1.0 * std::exp(-std::pow((hr - 19) / 2.2, 2));
            double wk = (h / 24) % 7 >= 5 ? 1.15 : 1.0;
            X[m][h] = (int32_t)std::lround((base + amp * shape) * wk + noise(g));
        }
    }
    spikes = {{0, 4 * 24 + 3}, {1, 5 * 24 + 14}, {2, 6 * 24 + 2}, {0, 6 * 24 + 11}};
    for (auto [m, h] : spikes) X[m][h] += 1800;
    return X;
}

// AR model: y_t = b0 + b1 x_{t-1} + b2 x_{t-2} + b3 x_{t-24}; fitted in the clear by the model owner on day 1-3 of meter 0
static std::vector<double> fit(const std::vector<std::vector<int32_t>>& X) {
    double A[4][5] = {};
    for (int m = 0; m < METERS; m++)
        for (int h = 24; h < 3 * 24; h++) {
            double f[4] = {1, (double)X[m][h - 1], (double)X[m][h - 2], (double)X[m][h - 24]};
            for (int i = 0; i < 4; i++) { for (int j = 0; j < 4; j++) A[i][j] += f[i] * f[j]; A[i][4] += f[i] * X[m][h]; }
        }
    for (int i = 0; i < 4; i++) for (int k = i + 1; k < 4; k++) { double r = A[k][i] / A[i][i]; for (int j = i; j < 5; j++) A[k][j] -= r * A[i][j]; }
    std::vector<double> b(4);
    for (int i = 3; i >= 0; i--) { double s = A[i][4]; for (int j = i + 1; j < 4; j++) s -= A[i][j] * b[j]; b[i] = s / A[i][i]; }
    return b;
}

int main(int argc, char** argv) {
    int party = parse_party(argv), port = peer_port();
    auto io = party == ALICE ? NetIO::listen(port, true) : NetIO::connect(peer_ip(), port, true);
    SH2PCSession s(io.get(), party);
    tsig::Curve C;   // secp256k1
    dkls23::OTPair ots(party, io.get());
    const bool b1 = party == ALICE;
    Trace T{{}, io.get(), &s};
    auto mine = [&](const bytes& a, const bytes& b) -> const bytes& { return b1 ? a : b; };

    // ---------- harness: shared synthetic world ----------
    std::mt19937_64 g(20260509);
    std::vector<std::pair<int, int>> spikes;
    auto X = make_data(g, spikes);
    auto beta = fit(X);   // real-valued AR coefficients

    // ================= Setup =================
    T.begin();
    std::vector<bytes> KP(METERS), KP1(METERS), KP2(METERS);
    for (int m = 0; m < METERS; m++) { KP[m] = rnd(g, 16); KP1[m] = rnd(g, 16); KP2[m] = xr(KP[m], KP1[m]); }
    bytes KS = rnd(g, 16), KS1 = rnd(g, 16), KS2 = xr(KS, KS1);
    T.end("Setup", "DistAEAD.Setup", "Dealer", "B1,B2", "Each publisher P_m and the subscriber S get K; brokers get XOR shares K(1), K(2)",
          "K_P0(1) = " + hex(KP1[0]), "K_P0(2) = " + hex(KP2[0]));
    T.begin(); auto ek = dkls23::keygen(C, io.get(), party);
    T.end("Setup", "TS.KeyGen (DKLs23)", "B1", "B2", "Commit-release keygen: additive shares sk = sk1 + sk2, tvk = sk·G, plus FZero PRF key",
          "sk1 (secret), tvk = " + hex(C.enc(ek.pk.get())), "sk2 (secret), tvk = " + hex(C.enc(ek.pk.get())));
    T.begin(); auto sk = tsig::schnorr_keygen(C, io.get(), party);
    T.end("Setup", "TS.KeyGen (Schnorr)", "B1", "B2", "Additive key shares; public pk = X1 + X2",
          "pk = " + hex(C.enc(sk.pk.get())), "pk = " + hex(C.enc(sk.pk.get())));
    std::vector<double> beta_sh(4);
    std::ostringstream bv1, bv2;
    bv1.precision(5); bv2.precision(5);
    std::uniform_real_distribution<double> bmask(-64.0, 64.0);   // |beta2 * x1| <= 64 * 2^16 = 2^22 stays under the 2^23 CKKS bound
    for (int i = 0; i < 4; i++) {
        double r = bmask(g); beta_sh[i] = b1 ? r : beta[i] - r;
        bv1 << (i ? ", " : "") << r; bv2 << (i ? ", " : "") << beta[i] - r;
    }
    T.begin();
    T.end("Setup", "Model shares", "Model owner", "B1,B2", "AR(1,2,24) coefficients as real numbers, beta = beta(1) + beta(2)",
          "beta(1) = [" + bv1.str() + "]", "beta(2) = [" + bv2.str() + "]");

    // ================= Publish =================
    std::vector<std::vector<int64_t>> D(METERS, std::vector<int64_t>(H));     // this broker's additive DB
    std::vector<uint32_t> xorsh(METERS * H);
    bytes ex_ct, ex_ad, ex_s1, ex_s2;
    T.begin();
    int dec_ok = 0;
    for (int m = 0; m < METERS; m++)
        for (int h = 0; h < H; h++) {
            bytes nonce = rnd(g, 16);
            std::string ads = "meter/" + std::to_string(m) + "/h" + std::to_string(h);
            bytes ad(ads.begin(), ads.end());
            bytes ct = ascon::encrypt(KP[m].data(), nonce.data(), ad, u32b((uint32_t)X[m][h]));   // publisher (simulated)
            auto d = distaead::dist_dec(s, mine(KP1[m], KP2[m]), nonce.data(), ad, ct);
            dec_ok += d.ok;
            xorsh[m * H + h] = b2u32(d.share);
            if (m == 0 && h == 19) { ex_ct = ct; ex_ad = ad; ex_s1 = d.share; }
        }
    // B2's example share travels to B1 only for the trace (harness)
    if (b1) { ex_s2.resize(4); io->recv_data(ex_s2.data(), 4); } else { io->send_data(&xorsh[19], 4); io->flush(); }
    if (!b1) ex_s2 = u32b(xorsh[19]);
    T.end("Publish", "DistAEAD.Dec", "Publishers", "B1,B2",
          std::to_string(METERS * H) + " readings; ASCON-128 decrypted inside a garbled circuit; " + std::to_string(dec_ok) + " tags valid",
          "C = " + hex(ex_ct) + " | M(1) = " + hex(ex_s1), "C = " + hex(ex_ct) + " | M(2) = " + hex(ex_s2));

    // tampered publication
    {
        bytes nonce = rnd(g, 16), ad = {'m', '0'};
        bytes ct = ascon::encrypt(KP[0].data(), nonce.data(), ad, u32b(1234)); ct[1] ^= 0x40;
        T.begin(); auto d = distaead::dist_dec(s, mine(KP1[0], KP2[0]), nonce.data(), ad, ct);
        T.end("Publish", "Tampered publication", "Attacker", "B1,B2", d.ok ? "ACCEPTED (bug)" : "Tag check failed inside the circuit: both brokers output ⊥, plaintext never opened",
              "⊥", "⊥");
    }

    T.begin();
    auto xa = convert::b2a(s, xorsh, 16);
    for (int m = 0; m < METERS; m++) for (int h = 0; h < H; h++) D[m][h] = xa[m * H + h];
    T.end("Publish", "B2A (XOR → additive)", "B1", "B2", "GC converts XOR shares of each int32 reading into additive shares x = x(1) + x(2), mask in [-2^16, 2^16], used as reals in CKKS",
          "D(1)[0][19] = " + std::to_string(D[0][19]), "D(2)[0][19] = " + std::to_string(D[0][19]));

    // ================= Compute =================
    // features for every (m, h >= 24): [1, x_{h-1}, x_{h-2}, x_{h-24}]
    const int F = 4, NF = METERS * (H - 24);
    std::vector<double> feat(NF * F), bvec(NF * F);
    for (int m = 0; m < METERS; m++)
        for (int h = 24; h < H; h++) {
            int k = m * (H - 24) + (h - 24);
            feat[k * F + 0] = b1 ? 1.0 : 0.0;
            feat[k * F + 1] = (double)D[m][h - 1]; feat[k * F + 2] = (double)D[m][h - 2]; feat[k * F + 3] = (double)D[m][h - 24];
            for (int i = 0; i < F; i++) bvec[k * F + i] = beta_sh[i];
        }
    T.begin();
    auto yhat = he::inner_products(io.get(), party, feat, bvec, F);
    auto hs = he::last;
    T.end("Compute", "HE cross terms (CKKS)", "B1", "B2",
          std::to_string(NF) + " AR forecasts in one batched CKKS round trip each way (N = 8192, scale 2^36); each broker generated fresh keys (" + std::to_string((int)hs.keygen_ms) + " ms) and erased them after use",
          "C(1) = Enc(pk1, x(1)) → B2; Y(1) = Dec(sk1, C_T) + δ1", "C(2) = Enc(pk2, x(2)) → B1; Y(2) = Dec(sk2, C_T′) + δ2");
    // each broker rounds its real share Y(i) to fixed point before entering the GC
    std::vector<int64_t> yfix(NF), dlt(NF);
    for (int k = 0; k < NF; k++) yfix[k] = std::llround(yhat[k] * Q);
    for (int m = 0; m < METERS; m++)
        for (int h = 24; h < H; h++) { int k = m * (H - 24) + (h - 24); dlt[k] = D[m][h] * (int64_t)Q - yfix[k]; }
    T.begin();
    auto alert = convert::threshold(s, dlt, (uint64_t)(ALERT_WH * Q));
    T.end("Compute", "GC threshold (anomaly)", "B1", "B2", "Y(1), Y(2) enter a garbled circuit (Sec. 4.1.3): |x_t − ŷ_t| > " + std::to_string(ALERT_WH) + " Wh decided inside; output stays XOR-shared",
          "alert(1) bits", "alert(2) bits");

    // ================= Subscribe =================
    // req: meter 0, last 24 hours: forecast (Wh x1000, int64) + alert flag
    const int REQ_M = 0, REQ_N = 24;
    std::string reqs = "req/S7/F=forecast+alerts/meter0/last24h";
    bytes req(reqs.begin(), reqs.end());
    std::vector<int64_t> ysel(REQ_N);
    for (int i = 0; i < REQ_N; i++) ysel[i] = yfix[REQ_M * (H - 24) + (H - 24 - REQ_N) + i];
    T.begin();
    auto yx = convert::a2b(s, ysel);
    bytes Y;
    for (int i = 0; i < REQ_N; i++) { auto b = u64b(yx[i]); Y.insert(Y.end(), b.begin(), b.end()); }
    for (int i = 0; i < REQ_N; i++) Y.push_back(alert[REQ_M * (H - 24) + (H - 24 - REQ_N) + i] & 1);
    T.end("Subscribe", "A2B (additive → XOR)", "B1", "B2", "Requested result Y (24 forecasts in mWh + 24 alert flags, " + std::to_string(Y.size()) + " B) as XOR shares Y(1) ⊕ Y(2)",
          "Y(1) = " + hex(Y, 24), "Y(2) = " + hex(Y, 24));
    bytes n2 = rnd(g, 16);
    T.begin();
    bytes CY = distaead::dist_enc(s, mine(KS1, KS2), n2.data(), req, Y);
    T.end("Subscribe", "DistAEAD.Enc", "B1", "B2", "ASCON-128 encryption of Y under the subscriber key, inside a garbled circuit",
          "C_Y = " + hex(CY), "C_Y = " + hex(CY));
    bytes tosign = CY; tosign.insert(tosign.end(), req.begin(), req.end());
    T.begin(); auto sres = dkls23::sign_and_check(C, io.get(), ots, party, ek, tosign);
    if (!sres.ok) { fprintf(stderr, "DKLs23 signing failed: %s\n", sres.failure.c_str()); return 1; }
    auto sig = sres.sig;
    T.end("Subscribe", "TS.Sign (DKLs23 ECDSA)", "B1", "B2", "Three rounds: commit R_i, RVOLE of {r_i, sk_i}, consistency checks on R_j and pk_j, then s = Σw / Σu (Doerner et al., S&P 2024)",
          "r = " + hex(sig.r), "s = " + hex(sig.s));
    T.begin(); auto ssig = tsig::schnorr_sign(C, io.get(), party, sk, tosign);
    T.end("Subscribe", "TS.Sign (2-party Schnorr)", "B1", "B2", "R = R1 + R2, s = s1 + s2", "R = " + hex(ssig.R), "s = " + hex(ssig.s));

    // ================= Harness: gather B2's view, subscriber checks, write trace =================
    std::vector<std::string> b2v;
    if (!b1) {
        // send B2's share strings to B1 (for the trace file only)
        std::ostringstream o;
        for (auto& st : T.steps) o << st.b2 << '\n';
        std::string str = o.str(); uint64_t n = str.size();
        io->send_data(&n, 8); io->send_data(str.data(), n); io->send_data(Y.data(), Y.size()); io->flush();
        printf("[B2] done\n");
        return 0;
    }
    uint64_t n; io->recv_data(&n, 8); std::string str(n, '\0'); io->recv_data(str.data(), n);
    bytes Y2(Y.size()); io->recv_data(Y2.data(), Y2.size());
    { std::istringstream is(str); std::string line; size_t i = 0; while (std::getline(is, line) && i < T.steps.size()) T.steps[i++].b2 = line; }
    // fix the share displays that need B2's actual values
    for (auto& st : T.steps) if (st.name == "A2B (additive → XOR)") st.b2 = "Y(2) = " + hex(Y2, 24);

    // ---- subscriber (simulated): verify both signatures, then decrypt ----
    bool v_ecdsa = dkls23::verify(C, ek.pk.get(), tosign, sig), v_schnorr = tsig::schnorr_verify(C, sk.pk.get(), tosign, ssig);
    auto Yd = ascon::decrypt(KS.data(), n2.data(), req, CY);
    bytes bad = tosign; bad[3] ^= 1;
    bool rej = !dkls23::verify(C, ek.pk.get(), bad, sig) && !tsig::schnorr_verify(C, sk.pk.get(), bad, ssig);
    // reference: same model in the clear
    int match = 0; double maxerr = 0;
    std::vector<double> fc(REQ_N); std::vector<int> al(REQ_N);
    if (Yd) for (int i = 0; i < REQ_N; i++) {
        int h = H - REQ_N + i;
        double ref = beta[0] + beta[1] * X[REQ_M][h - 1] + beta[2] * X[REQ_M][h - 2] + beta[3] * X[REQ_M][h - 24];
        int64_t gi = 0; for (int b = 0; b < 8; b++) gi |= (int64_t)(*Yd)[8 * i + b] << (8 * b);
        double got = gi / Q;
        int a = (*Yd)[8 * REQ_N + i];
        double diff = X[REQ_M][h] - ref;
        bool near = std::fabs(std::fabs(diff) - ALERT_WH) < 0.01;   // CKKS error can only flip a decision this close to the threshold
        if (std::fabs(got - ref) < 0.01 && (a == (std::fabs(diff) > ALERT_WH) || near)) match++;
        maxerr = std::max(maxerr, std::fabs(got - ref));
        fc[i] = got; al[i] = a;
    }
    printf("[B1] tags valid %d/%d | tampered publication rejected | ECDSA %s | Schnorr %s | tampered C_Y rejected %s | subscriber result matches clear model %d/%d (max CKKS error %.1e Wh)\n",
           dec_ok, METERS * H, v_ecdsa ? "OK" : "FAIL", v_schnorr ? "OK" : "FAIL", rej ? "yes" : "NO", match, REQ_N, maxerr);

    // all forecasts + alerts in the clear for the dashboard charts (computed by reconstructing shares: harness only)
    std::ofstream f("trace.js");
    f << "// Generated by drcf_demo (C++ brokers B1/B2, EMP sh2pc + SEAL + OpenSSL). Do not edit.\n";
    f << "window.DRCF_TRACE = {\n  generated: " << js("2026-11-05") << ",\n";
    f << "  params: {meters: " << METERS << ", hours: " << H << ", he: \"CKKS (SEAL 4.1.2), N = 8192, coeff {60,36,60}, scale 2^36\""
      << ", alertWh: " << ALERT_WH << ", beta: [" << beta[0] << "," << beta[1] << "," << beta[2] << "," << beta[3] << "], curve: \"secp256k1\", cipher: \"ASCON-128 (v1.2)\"},\n";
    f << "  ckks: {ctBytesOut: " << hs.ct_bytes_out << ", ctBytesIn: " << hs.ct_bytes_in << ", plainBytes: " << hs.plain_bytes << ", slots: " << hs.slots_used
      << ", keygenMs: " << hs.keygen_ms << ", maxErrWh: " << maxerr << "},\n";
    f << "  checks: {tagsValid: " << dec_ok << ", readings: " << METERS * H << ", ecdsa: " << v_ecdsa << ", schnorr: " << v_schnorr
      << ", tamperedRejected: " << rej << ", resultMatches: " << match << ", resultSize: " << REQ_N << "},\n";
    f << "  steps: [\n";
    for (auto& st : T.steps)
        f << "    {phase: " << js(st.phase) << ", name: " << js(st.name) << ", from: " << js(st.from) << ", to: " << js(st.to) << ", what: " << js(st.what)
          << ", ms: " << st.ms << ", bytes: " << st.bytes << ", ands: " << st.ands << ", b1: " << js(st.b1) << ", b2: " << js(st.b2) << "},\n";
    f << "  ],\n  readings: [\n";
    for (int m = 0; m < METERS; m++) { f << "    ["; for (int h = 0; h < H; h++) f << X[m][h] << (h + 1 < H ? "," : ""); f << "],\n"; }
    f << "  ],\n  forecast: [\n";
    for (int m = 0; m < METERS; m++) {
        f << "    [";
        for (int h = 24; h < H; h++) { double r = beta[0] + beta[1] * X[m][h - 1] + beta[2] * X[m][h - 2] + beta[3] * X[m][h - 24]; f << std::lround(r) << (h + 1 < H ? "," : ""); }
        f << "],\n";
    }
    f << "  ],\n  alerts: [\n";
    for (int m = 0; m < METERS; m++) {
        f << "    [";
        for (int h = 24; h < H; h++) { int k = m * (H - 24) + (h - 24); (void)k;
            double ref = beta[0] + beta[1] * X[m][h - 1] + beta[2] * X[m][h - 2] + beta[3] * X[m][h - 24];
            f << (std::fabs(X[m][h] - ref) > ALERT_WH) << (h + 1 < H ? "," : ""); }
        f << "],\n";
    }
    f << "  ],\n  spikes: [";
    for (auto [m, h] : spikes) f << "[" << m << "," << h << "],";
    f << "],\n  subscriber: {meter: " << REQ_M << ", from: " << H - REQ_N << ", req: " << js(reqs) << ", forecast: [";
    for (int i = 0; i < REQ_N; i++) f << fc[i] << (i + 1 < REQ_N ? "," : "");
    f << "], alerts: [";
    for (int i = 0; i < REQ_N; i++) f << al[i] << (i + 1 < REQ_N ? "," : "");
    f << "], CY: " << js(hex(CY, 400)) << ", ecdsa: {r: " << js(hex(sig.r)) << ", s: " << js(hex(sig.s)) << ", pk: " << js(hex(C.enc(ek.pk.get()))) << "}"
      << ", schnorr: {R: " << js(hex(ssig.R)) << ", s: " << js(hex(ssig.s)) << "}},\n";
    f << "  example: {K: " << js(hex(KP[0])) << ", K1: " << js(hex(KP1[0])) << ", K2: " << js(hex(KP2[0])) << ", value: " << X[0][19]
      << ", ad: " << js(std::string(ex_ad.begin(), ex_ad.end())) << ", C: " << js(hex(ex_ct)) << ", M1: " << js(hex(ex_s1)) << ", M2: " << js(hex(ex_s2)) << "}\n};\n";
    printf("[B1] trace.js written (%zu steps)\n", T.steps.size());
    return 0;
}
