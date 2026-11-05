// Encrypted cross-terms with Microsoft SEAL (BFV, batching), for DRCF.Compute linear models.
// Each broker has its OWN BFV key pair (Sec. 4.1.3). For x = x1 + x2 and beta = beta1 + beta2 (mod t):
//   sum_i beta_i x_i = <b1,x1> + <b2,x2> + <b2,x1> + <b1,x2>
// The two cross terms go through HE:  B1 sends Enc_1(x1); B2 returns Enc_1(b2*x1 + rho); B1 decrypts.
// Output: additive shares (mod t) of one inner product per group of `p` slots.
#pragma once
#include <seal/seal.h>
#include <sstream>
#include "emp-tool/emp-tool.h"
#include "convert.h"

namespace he {

inline int last_budget = -1;   // noise budget (bits) left in the last decrypted cross term

struct Party {
    seal::EncryptionParameters parms{seal::scheme_type::bfv};
    std::unique_ptr<seal::SEALContext> ctx;
    std::unique_ptr<seal::KeyGenerator> kg;
    seal::PublicKey pk;
    std::unique_ptr<seal::Encryptor> enc;
    std::unique_ptr<seal::Decryptor> dec;
    std::unique_ptr<seal::Evaluator> ev;
    std::unique_ptr<seal::BatchEncoder> be;
    uint64_t t;
    size_t slots;
    explicit Party(size_t N = 8192, int t_bits = 41) {
        parms.set_poly_modulus_degree(N);
        parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(N));
        parms.set_plain_modulus(seal::PlainModulus::Batching(N, t_bits));
        ctx = std::make_unique<seal::SEALContext>(parms);
        kg = std::make_unique<seal::KeyGenerator>(*ctx);
        kg->create_public_key(pk);
        enc = std::make_unique<seal::Encryptor>(*ctx, pk);
        dec = std::make_unique<seal::Decryptor>(*ctx, kg->secret_key());
        ev = std::make_unique<seal::Evaluator>(*ctx);
        be = std::make_unique<seal::BatchEncoder>(*ctx);
        t = parms.plain_modulus().value();
        slots = be->slot_count();
    }
};

inline void send_ct(emp::IOChannel* io, const seal::Ciphertext& c) {
    std::stringstream ss; c.save(ss, seal::compr_mode_type::none);
    std::string s = ss.str(); uint64_t n = s.size();
    io->send_data(&n, 8); io->send_data(s.data(), n); io->flush();
}
inline seal::Ciphertext recv_ct(emp::IOChannel* io, const seal::SEALContext& ctx) {
    uint64_t n; io->recv_data(&n, 8); std::string s(n, '\0'); io->recv_data(s.data(), n);
    std::stringstream ss(s); seal::Ciphertext c; c.load(ctx, ss); return c;
}

// key owner: encrypts own x, gets back masked cross term, returns sum per group
inline std::vector<uint64_t> cross_owner(Party& P, emp::IOChannel* io, const std::vector<uint64_t>& x, size_t p) {
    std::vector<uint64_t> v(P.slots, 0); std::copy(x.begin(), x.end(), v.begin());
    seal::Plaintext pt; P.be->encode(v, pt);
    seal::Ciphertext ct; P.enc->encrypt(pt, ct);
    send_ct(io, ct);
    seal::Ciphertext back = recv_ct(io, *P.ctx);
    last_budget = P.dec->invariant_noise_budget(back);
    seal::Plaintext out; P.dec->decrypt(back, out);
    std::vector<uint64_t> r; P.be->decode(out, r);
    std::vector<uint64_t> g(x.size() / p, 0);
    for (size_t i = 0; i < x.size(); i++) g[i / p] = (g[i / p] + r[i]) % P.t;
    return g;
}
// helper: multiplies the other party's ciphertext by own beta, adds a fresh mask, returns -sum(mask) per group
inline std::vector<uint64_t> cross_helper(Party& P, emp::IOChannel* io, const std::vector<uint64_t>& beta, size_t p) {
    seal::Ciphertext ct = recv_ct(io, *P.ctx);
    std::vector<uint64_t> b(P.slots, 0), rho(P.slots, 0);
    std::copy(beta.begin(), beta.end(), b.begin());
    for (size_t i = 0; i < beta.size(); i++) rho[i] = convert::rand_mod(P.t);
    seal::Plaintext pb, pr; P.be->encode(b, pb); P.be->encode(rho, pr);
    P.ev->multiply_plain_inplace(ct, pb);
    P.ev->add_plain_inplace(ct, pr);
    P.ev->mod_switch_to_next_inplace(ct);   // drop one prime: smaller reply, budget still positive
    send_ct(io, ct);
    std::vector<uint64_t> g(beta.size() / p, 0);
    for (size_t i = 0; i < beta.size(); i++) g[i / p] = (g[i / p] + P.t - rho[i]) % P.t;
    return g;
}

// Both brokers call this with their own shares of x and beta (same length n = groups * p).
// Returns this broker's additive share of <beta_g, x_g> for every group g.
inline std::vector<uint64_t> inner_products(Party& P, emp::IOChannel* io, int party, const std::vector<uint64_t>& x,
                                            const std::vector<uint64_t>& beta, size_t p) {
    if (x.size() > P.slots) throw std::invalid_argument("too many slots");
    const uint64_t t = P.t;
    std::vector<uint64_t> share(x.size() / p, 0);
    for (size_t i = 0; i < x.size(); i++) share[i / p] = (share[i / p] + convert::mulmod(x[i], beta[i], t)) % t;  // local term
    std::vector<uint64_t> a, b;
    if (party == emp::ALICE) { a = cross_owner(P, io, x, p); b = cross_helper(P, io, beta, p); }    // b2*x1 via B1 key, b1*x2 via B2 key
    else { a = cross_helper(P, io, beta, p); b = cross_owner(P, io, x, p); }
    for (size_t g = 0; g < share.size(); g++) share[g] = (share[g] + a[g] + b[g]) % t;
    return share;
}

}  // namespace he
