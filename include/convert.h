// Share conversions and decision logic in garbled circuits (EMP sh2pc), for DRCF.Compute.
// Arithmetic shares are signed 64-bit integers (two's complement, sums wrap mod 2^64).
//   b2a        : XOR shares of an int32 (from DistAEAD.Dec) -> additive shares x = x1 + x2 with
//                x1 drawn from [-2^k, 2^k], so the shares can be used as real numbers in CKKS.
//   a2b        : additive int64 shares -> XOR shares of the value (to feed DistAEAD.Enc).
//   threshold  : additive int64 shares of d, public theta -> XOR shares of [ |d| > theta ].
// After CKKS, each broker rounds its real share Y(i) to fixed point before entering the GC (Sec. 4.1.3).
#pragma once
#include <random>
#include "emp-sh2pc/emp-sh2pc.h"

namespace convert {

using Ctx = emp::SH2PCSession::ctx_t;
using U64 = emp::UInt_T<Ctx, 64>;
using U32 = emp::UInt_T<Ctx, 32>;

inline int64_t rand_range(int bits) {   // uniform in [-2^bits, 2^bits]
    static thread_local std::mt19937_64 g(std::random_device{}());
    std::uniform_int_distribution<int64_t> d(-(int64_t(1) << bits), int64_t(1) << bits); return d(g);
}

// Every party passes its own XOR share; returns its additive share (B1 gets the mask, B2 gets x - mask).
inline std::vector<int64_t> b2a(emp::SH2PCSession& s, const std::vector<uint32_t>& xs, int mask_bits) {
    auto& c = s.ctx();
    U64 hi = U64::constant(c, 0xFFFFFFFF00000000ULL);
    std::vector<int64_t> out(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        U32 v = s.input<U32>(emp::ALICE, xs[i]) ^ s.input<U32>(emp::BOB, xs[i]);
        U64 vz = v.zext<64>();
        U64 vs = vz.select(v[31], vz + hi);                       // sign-extend int32 -> int64
        int64_t r = s.party() == emp::ALICE ? rand_range(mask_bits) : 0;
        U64 R = s.input<U64>(emp::ALICE, (uint64_t)r);
        auto w = s.reveal(vs - R, emp::BOB);
        out[i] = s.party() == emp::ALICE ? r : (int64_t)*w;
    }
    return out;
}

inline std::vector<uint64_t> a2b(emp::SH2PCSession& s, const std::vector<int64_t>& ys) {
    std::vector<uint64_t> out(ys.size());
    for (size_t i = 0; i < ys.size(); i++) {
        U64 y = s.input<U64>(emp::ALICE, (uint64_t)ys[i]) + s.input<U64>(emp::BOB, (uint64_t)ys[i]);
        out[i] = *s.reveal(y, emp::XOR);
    }
    return out;
}

inline std::vector<uint8_t> threshold(emp::SH2PCSession& s, const std::vector<int64_t>& ds, uint64_t theta) {
    auto& c = s.ctx();
    U64 TH = U64::constant(c, theta), Z = U64::constant(c, 0);
    std::vector<uint8_t> out(ds.size());
    for (size_t i = 0; i < ds.size(); i++) {
        U64 d = s.input<U64>(emp::ALICE, (uint64_t)ds[i]) + s.input<U64>(emp::BOB, (uint64_t)ds[i]);
        U64 a = d.select(d[63], Z - d);                           // |d|
        out[i] = *s.reveal(a > TH, emp::XOR);
    }
    return out;
}

// Protocol II decision: [ a > b ] OR [ y == 0 ], all three as additive int64 shares. Output XOR-shared.
inline std::vector<uint8_t> greater_or_zero(emp::SH2PCSession& s, const std::vector<int64_t>& a, const std::vector<int64_t>& b,
                                            const std::vector<int64_t>& y) {
    auto& c = s.ctx();
    U64 Z = U64::constant(c, 0);
    std::vector<uint8_t> out(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        U64 A = s.input<U64>(emp::ALICE, (uint64_t)a[i]) + s.input<U64>(emp::BOB, (uint64_t)a[i]);
        U64 B = s.input<U64>(emp::ALICE, (uint64_t)b[i]) + s.input<U64>(emp::BOB, (uint64_t)b[i]);
        U64 Y = s.input<U64>(emp::ALICE, (uint64_t)y[i]) + s.input<U64>(emp::BOB, (uint64_t)y[i]);
        U64 d = B - A;                                             // a > b  <=>  b - a < 0
        out[i] = *s.reveal(d[63] | (Y == Z), emp::XOR);
    }
    return out;
}

}  // namespace convert
