// ASCON-128 as a boolean circuit over EMP Bit_T<Ctx>.
// Public values (IV, nonce, AD, ciphertext, tag, round constants) are folded as
// constants, so they create no gates. XOR / NOT are free under free-XOR; only AND
// costs (2 ciphertexts with half-gates). One permutation round = 320 AND gates.
#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "emp-tool/emp-tool.h"

namespace ascon_gc {

// A wire is either a known constant (c = 0/1) or a secret EMP bit (c = -1).
template <class Ctx>
struct W {
    using B = emp::Bit_T<Ctx>;
    int8_t c = 0;
    B b{};
    static W k(bool v) { W w; w.c = v; return w; }
    static W s(const B& x) { W w; w.c = -1; w.b = x; return w; }
    bool is_const() const { return c >= 0; }
};

template <class Ctx> W<Ctx> operator^(const W<Ctx>& x, const W<Ctx>& y) {
    if (x.is_const() && y.is_const()) return W<Ctx>::k(x.c ^ y.c);
    if (x.is_const()) return x.c ? W<Ctx>::s(!y.b) : y;
    if (y.is_const()) return y.c ? W<Ctx>::s(!x.b) : x;
    return W<Ctx>::s(x.b ^ y.b);
}
template <class Ctx> W<Ctx> operator&(const W<Ctx>& x, const W<Ctx>& y) {
    if (x.is_const() && y.is_const()) return W<Ctx>::k(x.c & y.c);
    if (x.is_const()) return x.c ? y : W<Ctx>::k(0);
    if (y.is_const()) return y.c ? x : W<Ctx>::k(0);
    return W<Ctx>::s(x.b & y.b);
}
template <class Ctx> W<Ctx> operator!(const W<Ctx>& x) {
    return x.is_const() ? W<Ctx>::k(!x.c) : W<Ctx>::s(!x.b);
}

// 64-bit word: index i = bit i (LSB = 0). Bytes: 8 wires, LSB first.
template <class Ctx> using Word = std::array<W<Ctx>, 64>;
template <class Ctx> using Byte = std::array<W<Ctx>, 8>;

template <class Ctx> Word<Ctx> kword(uint64_t x) {
    Word<Ctx> w; for (int i = 0; i < 64; i++) w[i] = W<Ctx>::k((x >> i) & 1); return w;
}
template <class Ctx> Byte<Ctx> kbyte(uint8_t x) {
    Byte<Ctx> b; for (int i = 0; i < 8; i++) b[i] = W<Ctx>::k((x >> i) & 1); return b;
}
// big-endian: byte j of the block sits at bits (7-j)*8 .. (7-j)*8+7
template <class Ctx> Word<Ctx> word(const Byte<Ctx>* bytes8) {
    Word<Ctx> w;
    for (int j = 0; j < 8; j++) for (int k = 0; k < 8; k++) w[(7 - j) * 8 + k] = bytes8[j][k];
    return w;
}
template <class Ctx> Byte<Ctx> word_byte(const Word<Ctx>& w, int j) {
    Byte<Ctx> b; for (int k = 0; k < 8; k++) b[k] = w[(7 - j) * 8 + k]; return b;
}
template <class Ctx> Word<Ctx> operator^(const Word<Ctx>& a, const Word<Ctx>& b) {
    Word<Ctx> o; for (int i = 0; i < 64; i++) o[i] = a[i] ^ b[i]; return o;
}
template <class Ctx> Word<Ctx> rotr(const Word<Ctx>& w, int n) {
    Word<Ctx> o; for (int i = 0; i < 64; i++) o[i] = w[(i + n) % 64]; return o;
}

template <class Ctx> void permute(std::array<Word<Ctx>, 5>& S, int rounds) {
    static const int L[5][2] = {{19, 28}, {61, 39}, {1, 6}, {10, 17}, {7, 41}};
    for (int r = 12 - rounds; r < 12; r++) {
        S[2] = S[2] ^ kword<Ctx>((uint64_t)(0xf0 - r * 0x10 + r));
        S[0] = S[0] ^ S[4]; S[4] = S[4] ^ S[3]; S[2] = S[2] ^ S[1];
        std::array<Word<Ctx>, 5> T;
        for (int i = 0; i < 5; i++) for (int k = 0; k < 64; k++) T[i][k] = (!S[i][k]) & S[(i + 1) % 5][k];
        for (int i = 0; i < 5; i++) S[i] = S[i] ^ T[(i + 1) % 5];
        S[1] = S[1] ^ S[0]; S[0] = S[0] ^ S[4]; S[3] = S[3] ^ S[2];
        for (int k = 0; k < 64; k++) S[2][k] = !S[2][k];
        for (int i = 0; i < 5; i++) S[i] = S[i] ^ rotr(S[i], L[i][0]) ^ rotr(S[i], L[i][1]);
    }
}

inline uint64_t load_be(const uint8_t* p, size_t n = 8) {
    uint64_t x = 0; for (size_t i = 0; i < 8; i++) x = (x << 8) | (i < n ? p[i] : 0); return x;
}

// key: 16 secret bytes (already K1 ^ K2)
template <class Ctx>
std::array<Word<Ctx>, 5> init(const std::vector<Byte<Ctx>>& key, const uint8_t nonce[16], const std::vector<uint8_t>& ad,
                              Word<Ctx>& K0, Word<Ctx>& K1) {
    K0 = word<Ctx>(&key[0]); K1 = word<Ctx>(&key[8]);
    std::array<Word<Ctx>, 5> S = {kword<Ctx>(0x80400c0600000000ULL), K0, K1, kword<Ctx>(load_be(nonce)), kword<Ctx>(load_be(nonce + 8))};
    permute(S, 12);
    S[3] = S[3] ^ K0; S[4] = S[4] ^ K1;
    if (!ad.empty()) {
        std::vector<uint8_t> a(ad); a.push_back(0x80); while (a.size() % 8) a.push_back(0);
        for (size_t i = 0; i < a.size(); i += 8) { S[0] = S[0] ^ kword<Ctx>(load_be(&a[i])); permute(S, 6); }
    }
    S[4] = S[4] ^ kword<Ctx>(1);
    return S;
}

// returns 16 tag bytes (as wires)
template <class Ctx>
std::vector<Byte<Ctx>> final_tag(std::array<Word<Ctx>, 5>& S, const Word<Ctx>& K0, const Word<Ctx>& K1) {
    S[1] = S[1] ^ K0; S[2] = S[2] ^ K1;
    permute(S, 12);
    Word<Ctx> t0 = S[3] ^ K0, t1 = S[4] ^ K1;
    std::vector<Byte<Ctx>> t;
    for (int j = 0; j < 8; j++) t.push_back(word_byte(t0, j));
    for (int j = 0; j < 8; j++) t.push_back(word_byte(t1, j));
    return t;
}

// Encrypt secret message bytes. Returns ct || tag wires (to be revealed).
template <class Ctx>
std::vector<Byte<Ctx>> encrypt(const std::vector<Byte<Ctx>>& key, const uint8_t nonce[16], const std::vector<uint8_t>& ad,
                               const std::vector<Byte<Ctx>>& msg) {
    Word<Ctx> K0, K1;
    auto S = init(key, nonce, ad, K0, K1);
    std::vector<Byte<Ctx>> p(msg);
    p.push_back(kbyte<Ctx>(0x80)); while (p.size() % 8) p.push_back(kbyte<Ctx>(0));
    std::vector<Byte<Ctx>> out;
    for (size_t i = 0; i < p.size(); i += 8) {
        S[0] = S[0] ^ word<Ctx>(&p[i]);
        for (size_t j = 0; j < 8 && i + j < msg.size(); j++) out.push_back(word_byte(S[0], (int)j));
        if (i + 8 < p.size()) permute(S, 6);
    }
    auto t = final_tag(S, K0, K1);
    out.insert(out.end(), t.begin(), t.end());
    return out;
}

// Decrypt public ct||tag. Returns plaintext wires; ok = tag-valid bit.
template <class Ctx>
std::vector<Byte<Ctx>> decrypt(const std::vector<Byte<Ctx>>& key, const uint8_t nonce[16], const std::vector<uint8_t>& ad,
                               const std::vector<uint8_t>& ct_tag, W<Ctx>& ok) {
    size_t n = ct_tag.size() - 16, last = n % 8, full = n - last;
    Word<Ctx> K0, K1;
    auto S = init(key, nonce, ad, K0, K1);
    std::vector<Byte<Ctx>> pt;
    for (size_t i = 0; i < full; i += 8) {
        for (int j = 0; j < 8; j++) { auto sb = word_byte(S[0], j), cb = kbyte<Ctx>(ct_tag[i + j]); Byte<Ctx> o; for (int k = 0; k < 8; k++) o[k] = sb[k] ^ cb[k]; pt.push_back(o); }
        S[0] = kword<Ctx>(load_be(&ct_tag[i]));
        permute(S, 6);
    }
    Byte<Ctx> nb[8];
    for (size_t j = 0; j < 8; j++) {
        auto sb = word_byte(S[0], (int)j);
        if (j < last) { auto cb = kbyte<Ctx>(ct_tag[full + j]); Byte<Ctx> o; for (int k = 0; k < 8; k++) o[k] = sb[k] ^ cb[k]; pt.push_back(o); nb[j] = cb; }
        else if (j == last) { auto pb = kbyte<Ctx>(0x80); for (int k = 0; k < 8; k++) nb[j][k] = sb[k] ^ pb[k]; }
        else nb[j] = sb;
    }
    S[0] = word<Ctx>(nb);
    auto t = final_tag(S, K0, K1);
    // ok = AND over all 128 bits of (t == tag); balanced tree, 127 AND gates
    std::vector<W<Ctx>> eq;
    for (int j = 0; j < 16; j++) for (int k = 0; k < 8; k++) eq.push_back(((ct_tag[n + j] >> k) & 1) ? t[j][k] : !t[j][k]);
    while (eq.size() > 1) {
        std::vector<W<Ctx>> nx;
        for (size_t i = 0; i < eq.size(); i += 2) nx.push_back(i + 1 < eq.size() ? (eq[i] & eq[i + 1]) : eq[i]);
        eq.swap(nx);
    }
    ok = eq[0];
    return pt;
}

}  // namespace ascon_gc
