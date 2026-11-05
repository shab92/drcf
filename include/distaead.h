// 2-party DistAEAD (DRCF, Definition 5) on EMP sh2pc. ALICE = broker B1 (garbler),
// BOB = broker B2 (evaluator). Each broker calls the same function with its own shares.
//
//   dist_dec(K^(i), N, AD, C||tag)  -> {ok, M^(i)} with M^(1) ^ M^(2) = M, or ok = false (⊥)
//   dist_enc(K^(i), N, AD, Y^(i))   -> C||tag of Y = Y^(1) ^ Y^(2), known to both
//
// K = K^(1) ^ K^(2) only ever exists inside the garbled circuit.
#pragma once
#include <stdexcept>
#include "emp-sh2pc/emp-sh2pc.h"
#include "ascon_circuit.h"

namespace distaead {

using Ctx = emp::SH2PCSession::ctx_t;
using B = emp::Bit_T<Ctx>;
using Wc = ascon_gc::W<Ctx>;
using Bt = ascon_gc::Byte<Ctx>;
using bytes = std::vector<uint8_t>;

// Both parties call this. Only `owner`'s bytes are used; the other passes anything of length n.
inline std::vector<Bt> input_bytes(emp::SH2PCSession& s, int owner, const bytes& mine, size_t n) {
    std::unique_ptr<bool[]> bits(new bool[8 * n]());
    if (s.party() == owner) {
        if (mine.size() != n) throw std::invalid_argument("input length mismatch");
        for (size_t i = 0; i < 8 * n; i++) bits[i] = (mine[i / 8] >> (i % 8)) & 1;
    }
    auto w = s.input_bits(owner, bits.get(), 8 * n);
    std::vector<Bt> out(n);
    for (size_t i = 0; i < 8 * n; i++) out[i / 8][i % 8] = Wc::s(B(s.ctx(), w[i]));
    return out;
}

// recipient: emp::PUBLIC, emp::ALICE, emp::BOB, or emp::XOR (each party gets its share)
inline bytes reveal_bytes(emp::SH2PCSession& s, const std::vector<Bt>& v, int recipient) {
    std::vector<emp::block> w;
    for (auto& by : v) for (auto& x : by) if (!x.is_const()) w.push_back(x.b.w);
    std::unique_ptr<bool[]> o(new bool[w.size() + 1]());
    if (!w.empty()) s.reveal_bits(o.get(), recipient, w.data(), w.size());
    bytes out(v.size(), 0);
    size_t k = 0;
    for (size_t i = 0; i < v.size(); i++)
        for (int j = 0; j < 8; j++) {
            bool bit;
            if (v[i][j].is_const()) bit = (recipient == emp::XOR && s.party() == emp::BOB) ? false : v[i][j].c;
            else bit = o[k++];
            out[i] |= (uint8_t)bit << j;
        }
    return out;
}

inline std::vector<Bt> shared_key(emp::SH2PCSession& s, const bytes& my_key_share) {
    auto k1 = input_bytes(s, emp::ALICE, my_key_share, 16);
    auto k2 = input_bytes(s, emp::BOB, my_key_share, 16);
    std::vector<Bt> k(16);
    for (int i = 0; i < 16; i++) for (int j = 0; j < 8; j++) k[i][j] = k1[i][j] ^ k2[i][j];
    return k;
}

struct DecResult { bool ok; bytes share; };

inline DecResult dist_dec(emp::SH2PCSession& s, const bytes& key_share, const uint8_t nonce[16], const bytes& ad, const bytes& ct_tag) {
    if (ct_tag.size() < 16) return {false, {}};
    auto K = shared_key(s, key_share);
    Wc ok;
    auto pt = ascon_gc::decrypt<Ctx>(K, nonce, ad, ct_tag, ok);
    auto okb = reveal_bytes(s, {Bt{ok, Wc::k(0), Wc::k(0), Wc::k(0), Wc::k(0), Wc::k(0), Wc::k(0), Wc::k(0)}}, emp::PUBLIC);
    if (!(okb[0] & 1)) return {false, {}};               // both brokers output ⊥; plaintext never opened
    return {true, reveal_bytes(s, pt, emp::XOR)};        // M^(1) at B1, M^(2) at B2
}

inline bytes dist_enc(emp::SH2PCSession& s, const bytes& key_share, const uint8_t nonce[16], const bytes& ad, const bytes& y_share) {
    auto K = shared_key(s, key_share);
    size_t n = y_share.size();
    auto y1 = input_bytes(s, emp::ALICE, y_share, n);
    auto y2 = input_bytes(s, emp::BOB, y_share, n);
    std::vector<Bt> y(n);
    for (size_t i = 0; i < n; i++) for (int j = 0; j < 8; j++) y[i][j] = y1[i][j] ^ y2[i][j];
    return reveal_bytes(s, ascon_gc::encrypt<Ctx>(K, nonce, ad, y), emp::PUBLIC);
}

}  // namespace distaead
