// Joint evaluation with CKKS (Microsoft SEAL 4.1.2), following DRCF Sec. 4.1.3 and Sec. 5.
//
// Regression tasks hold real-valued additive shares X = X1 + X2 and beta = beta1 + beta2.
// When joint computation is needed, each broker generates FRESH CKKS keys (short-term keys,
// Sec. 4.1.1), encrypts its local share under its own key, and the brokers exchange encrypted
// cross terms:
//     B1:  C(1) = Enc(sk1, X1)  ->  B2 evaluates  C_T = C(1) * beta2 + delta   ->  B1
//     B2:  C(2) = Enc(sk2, X2)  ->  B1 evaluates  C_T' = C(2) * beta1 + delta' ->  B2
// and each broker outputs   Y(i) = Dec(sk_i, C_T) + delta_i   (local term minus its own mask),
// so that Y(1) + Y(2) = <beta, X> per group (up to CKKS approximation error). Keys are erased after use.
#pragma once
#include <seal/seal.h>
#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
#include "emp-tool/emp-tool.h"

namespace he {

// CKKS parameters. N = 8192 with a 200-bit modulus is within SEAL's 128-bit security table (218 bits).
// Each broker encrypts its own share under its own SECRET key (seeded symmetric encryption) and
// ciphertexts are serialised with zstd. This reproduces the ~6x expansion of Table 6 in the paper
// (measured: 6.02-6.03x per fresh ciphertext with all 4096 slots filled).
struct Params {
    size_t N = 8192;
    std::vector<int> bits = {40, 40, 40, 40, 40};   // five 40-bit primes; one rescale after multiply_plain
    double scale = std::pow(2.0, 40);
    double mask_range = std::pow(2.0, 30);           // |delta| bound; headroom after rescale is about 2^79
    seal::compr_mode_type compr = seal::compr_mode_type::zstd;
};

struct Stats { size_t ct_bytes_out = 0, ct_bytes_in = 0, fresh_ct_bytes = 0, plain_bytes = 0, slots = 0; double keygen_ms = 0; int slots_used = 0; };
inline Stats last;

inline std::mt19937_64& rng() { static thread_local std::mt19937_64 g(std::random_device{}()); return g; }
inline double uniform(double r) { std::uniform_real_distribution<double> d(-r, r); return d(rng()); }

struct Session {   // one broker's short-term CKKS state for a single joint evaluation
    seal::EncryptionParameters parms{seal::scheme_type::ckks};
    std::unique_ptr<seal::SEALContext> ctx;
    std::unique_ptr<seal::KeyGenerator> kg;
    std::unique_ptr<seal::Encryptor> enc;
    std::unique_ptr<seal::Decryptor> dec;
    std::unique_ptr<seal::Evaluator> ev;
    std::unique_ptr<seal::CKKSEncoder> ce;
    Params P;
    explicit Session(const Params& p) : P(p) {
        parms.set_poly_modulus_degree(P.N);
        parms.set_coeff_modulus(seal::CoeffModulus::Create(P.N, P.bits));
        ctx = std::make_unique<seal::SEALContext>(parms);
        kg = std::make_unique<seal::KeyGenerator>(*ctx);          // fresh sk_i for this evaluation
        enc = std::make_unique<seal::Encryptor>(*ctx, kg->secret_key());   // symmetric: only this broker can decrypt
        dec = std::make_unique<seal::Decryptor>(*ctx, kg->secret_key());
        ev = std::make_unique<seal::Evaluator>(*ctx);
        ce = std::make_unique<seal::CKKSEncoder>(*ctx);
    }
    size_t slots() const { return ce->slot_count(); }
};

inline size_t send_str(emp::IOChannel* io, const std::string& s) {
    uint64_t n = s.size(); io->send_data(&n, 8); io->send_data(s.data(), n); io->flush(); return n;
}
template <class T> inline size_t send_ct(emp::IOChannel* io, const T& c, seal::compr_mode_type m) {
    std::stringstream ss; c.save(ss, m); return send_str(io, ss.str());
}
inline seal::Ciphertext recv_ct(emp::IOChannel* io, const seal::SEALContext& ctx, size_t& n_out) {
    uint64_t n; io->recv_data(&n, 8); std::string s(n, '\0'); io->recv_data(s.data(), n);
    std::stringstream ss(s); seal::Ciphertext c; c.load(ctx, ss); n_out = n; return c;
}

// key owner: C(i) = Enc(sk_i, X_i) -> peer; receives C_T; returns Dec(sk_i, C_T) summed per group of p slots
inline std::vector<double> owner(Session& S, emp::IOChannel* io, const std::vector<double>& x, size_t p) {
    std::vector<double> v(S.slots(), 0.0); std::copy(x.begin(), x.end(), v.begin());
    seal::Plaintext pt; S.ce->encode(v, S.P.scale, pt);
    size_t n0 = send_ct(io, S.enc->encrypt_symmetric(pt), S.P.compr);   // C(i) = Enc(sk_i, X_i), seeded + compressed
    last.ct_bytes_out += n0; last.fresh_ct_bytes = n0;
    size_t n; seal::Ciphertext back = recv_ct(io, *S.ctx, n); last.ct_bytes_in += n;
    seal::Plaintext out; S.dec->decrypt(back, out);
    std::vector<double> r; S.ce->decode(out, r);
    std::vector<double> g(x.size() / p, 0.0);
    for (size_t i = 0; i < x.size(); i++) g[i / p] += r[i];
    return g;
}
// evaluator: C_T = C(j) * beta_i + delta, sent back; returns -sum(delta) per group
inline std::vector<double> evaluator(Session& S, emp::IOChannel* io, const std::vector<double>& beta, size_t p) {
    size_t n; seal::Ciphertext ct = recv_ct(io, *S.ctx, n); last.ct_bytes_in += n;
    std::vector<double> b(S.slots(), 0.0), d(S.slots(), 0.0);
    std::copy(beta.begin(), beta.end(), b.begin());
    for (size_t i = 0; i < beta.size(); i++) d[i] = uniform(S.P.mask_range);
    seal::Plaintext pb, pd;
    S.ce->encode(b, S.P.scale, pb);
    S.ev->multiply_plain_inplace(ct, pb);
    S.ev->rescale_to_next_inplace(ct);
    S.ce->encode(d, ct.parms_id(), ct.scale(), pd);
    S.ev->add_plain_inplace(ct, pd);
    last.ct_bytes_out += send_ct(io, ct, S.P.compr);
    std::vector<double> g(beta.size() / p, 0.0);
    for (size_t i = 0; i < beta.size(); i++) g[i / p] -= d[i];
    return g;
}

// Both brokers call this with their own real-valued shares of X and beta (n = groups * p).
// Returns Y(i): this broker's additive share of <beta_g, X_g> for every group g.
inline std::vector<double> inner_products(emp::IOChannel* io, int party, const std::vector<double>& x,
                                          const std::vector<double>& beta, size_t p, const Params& P = Params()) {
    last = Stats{};
    auto t0 = std::chrono::steady_clock::now();
    Session S(P);                                             // fresh short-term keys for this evaluation
    last.keygen_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (x.size() > S.slots()) throw std::invalid_argument("too many slots");
    last.slots_used = (int)x.size(); last.plain_bytes = x.size() * sizeof(double); last.slots = S.slots();
    std::vector<double> y(x.size() / p, 0.0);
    for (size_t i = 0; i < x.size(); i++) y[i / p] += x[i] * beta[i];               // local term
    std::vector<double> a, b;
    if (party == emp::ALICE) { a = owner(S, io, x, p); b = evaluator(S, io, beta, p); }
    else { a = evaluator(S, io, beta, p); b = owner(S, io, x, p); }
    for (size_t g = 0; g < y.size(); g++) y[g] += a[g] + b[g];
    return y;                                                  // S (and sk_i) destroyed here
}

}  // namespace he
