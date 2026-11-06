// Batched garbled-circuit helpers for DRCF.Compute protocols that touch many readings at once.
// One OT batch per owner for inputs, one reveal for outputs, instead of one round per value.
#pragma once
#include <memory>
#include <vector>
#include "emp-sh2pc/emp-sh2pc.h"

namespace batch {

using Ctx = emp::SH2PCSession::ctx_t;
using U64 = emp::UInt_T<Ctx, 64>;
using Bit = emp::Bit_T<Ctx>;

// Both parties call with the same n; only `owner`'s values are read.
inline std::vector<U64> input(emp::SH2PCSession& s, int owner, const std::vector<int64_t>& v, size_t n) {
    std::unique_ptr<bool[]> bits(new bool[64 * n]());
    if (s.party() == owner)
        for (size_t i = 0; i < n; i++) for (int b = 0; b < 64; b++) bits[64 * i + b] = ((uint64_t)v[i] >> b) & 1;
    auto w = s.input_bits(owner, bits.get(), 64 * n);
    std::vector<U64> out; out.reserve(n);
    for (size_t i = 0; i < n; i++) out.push_back(U64::from_wires(s.ctx(), &w[64 * i]));
    return out;
}
// x = x1 + x2 (mod 2^64) for every element
inline std::vector<U64> input_shared(emp::SH2PCSession& s, const std::vector<int64_t>& mine, size_t n) {
    auto a = input(s, emp::ALICE, mine, n), b = input(s, emp::BOB, mine, n);
    for (size_t i = 0; i < n; i++) a[i] = a[i] + b[i];
    return a;
}
// recipient: emp::XOR gives each party its share of every bit
inline std::vector<uint8_t> reveal(emp::SH2PCSession& s, const std::vector<Bit>& bits, int recipient) {
    std::vector<typename Ctx::Wire> w; w.reserve(bits.size());
    for (auto& b : bits) w.push_back(b.w);
    std::unique_ptr<bool[]> o(new bool[bits.size() + 1]());
    if (!w.empty()) s.reveal_bits(o.get(), recipient, w.data(), w.size());
    return std::vector<uint8_t>(o.get(), o.get() + bits.size());
}
// additive int64 shares of a secret U64 value: B1 inputs a random mask, B2 learns value - mask
inline int64_t to_additive(emp::SH2PCSession& s, const U64& v) {
    int64_t r = 0;
    if (s.party() == emp::ALICE) { emp::PRG prg; prg.random_data(&r, 8); }
    U64 R = input(s, emp::ALICE, {r}, 1)[0];
    auto w = s.reveal(v - R, emp::BOB);
    return s.party() == emp::ALICE ? r : (int64_t)*w;
}

inline U64 abs64(const U64& d) { U64 z = U64::constant(*d.context(), 0); return d.select(d[63], z - d); }

// floor(sqrt(v)) for v < 2^62, bit-by-bit (32 rounds, about 10k AND gates)
inline U64 isqrt(const U64& v0) {
    auto& c = *v0.context();
    U64 v = v0, res = U64::constant(c, 0);
    for (int k = 31; k >= 0; k--) {
        U64 bit = U64::constant(c, 1ULL << (2 * k));
        U64 t = res + bit;
        Bit ge = v >= t;
        v = v.select(ge, v - t);
        res = (res >> 1).select(ge, (res >> 1) + bit);
    }
    return res;
}

}  // namespace batch
