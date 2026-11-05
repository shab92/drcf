// Share conversions and decision logic in garbled circuits (EMP sh2pc), for DRCF.Compute.
//   b2a        : XOR shares of int32 (from DistAEAD.Dec)   -> additive shares mod t (for HE / linear ops)
//   a2b_signed : additive shares mod t                     -> XOR shares of the signed int64 value
//   threshold  : additive shares of d mod t, public theta  -> XOR shares of [ |d| > theta ]
// t is the BFV plaintext modulus (a prime < 2^62). Values are centred: v in (-t/2, t/2).
#pragma once
#include <random>
#include "emp-sh2pc/emp-sh2pc.h"

namespace convert {

using Ctx = emp::SH2PCSession::ctx_t;
using U64 = emp::UInt_T<Ctx, 64>;
using U32 = emp::UInt_T<Ctx, 32>;
using B = emp::Bit_T<Ctx>;

inline uint64_t rand_mod(uint64_t t) {
    static thread_local std::mt19937_64 g(std::random_device{}());
    std::uniform_int_distribution<uint64_t> d(0, t - 1); return d(g);
}

// (a + b) mod t for a, b in [0, t)
inline U64 add_mod(const U64& a, const U64& b, const U64& T) { U64 s = a + b; return s.select(s >= T, s - T); }

// Every party passes its own XOR share; returns its additive share.
inline std::vector<uint64_t> b2a(emp::SH2PCSession& s, const std::vector<uint32_t>& xs, uint64_t t) {
    auto& c = s.ctx();
    U64 T = U64::constant(c, t), off = U64::constant(c, t - (1ULL << 32));
    std::vector<uint64_t> out(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        U32 v = s.input<U32>(emp::ALICE, xs[i]) ^ s.input<U32>(emp::BOB, xs[i]);
        U64 vz = v.zext<64>();
        U64 vm = vz.select(v[31], vz + off);                 // int32 -> [0, t)
        uint64_t r = s.party() == emp::ALICE ? rand_mod(t) : 0;
        U64 R = s.input<U64>(emp::ALICE, r);
        U64 w = add_mod(vm, T - R, T);                       // v - r mod t
        auto w2 = s.reveal(w, emp::BOB);
        out[i] = s.party() == emp::ALICE ? r : *w2;
    }
    return out;
}

inline U64 centred(const U64& y, const U64& T) {        // [0, t) -> two's complement int64
    U64 half = T >> 1;
    return y.select(y > half, y - T);
}

inline std::vector<uint64_t> a2b_signed(emp::SH2PCSession& s, const std::vector<uint64_t>& ys, uint64_t t) {
    auto& c = s.ctx(); U64 T = U64::constant(c, t);
    std::vector<uint64_t> out(ys.size());
    for (size_t i = 0; i < ys.size(); i++) {
        U64 y = add_mod(s.input<U64>(emp::ALICE, ys[i]), s.input<U64>(emp::BOB, ys[i]), T);
        out[i] = *s.reveal(centred(y, T), emp::XOR);
    }
    return out;
}

inline std::vector<uint8_t> threshold(emp::SH2PCSession& s, const std::vector<uint64_t>& ds, uint64_t t, uint64_t theta) {
    auto& c = s.ctx(); U64 T = U64::constant(c, t), TH = U64::constant(c, theta);
    std::vector<uint8_t> out(ds.size());
    for (size_t i = 0; i < ds.size(); i++) {
        U64 d = add_mod(s.input<U64>(emp::ALICE, ds[i]), s.input<U64>(emp::BOB, ds[i]), T);
        U64 a = d.select(d > (T >> 1), T - d);                // |d|
        out[i] = *s.reveal(a > TH, emp::XOR);
    }
    return out;
}

// helpers on clear additive shares
inline uint64_t enc_signed(int64_t v, uint64_t t) { return v >= 0 ? (uint64_t)v % t : t - ((uint64_t)(-v) % t); }
inline int64_t dec_signed(uint64_t v, uint64_t t) { return v > t / 2 ? -(int64_t)(t - v) : (int64_t)v; }
inline uint64_t mulmod(uint64_t a, uint64_t b, uint64_t t) { return (unsigned __int128)a * b % t; }

}  // namespace convert
