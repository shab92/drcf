// RBF support vector regression evaluated inside a garbled circuit (EMP sh2pc), fixed point Q14.
//   f(x) = b + sum_k alpha_k * exp(-gamma * ||x - sv_k||^2)
// exp(-z) = 2^-(z log2 e): split y = z log2 e into integer k and fraction f, 2^-f by a degree-4 polynomial
// (max error 3.5e-6 with Q14 coefficients, below the Q14 step), then an arithmetic shift right by k.
// Data, support vectors, alpha and b are all secret (shared) inputs; gamma is a public hyperparameter.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>
#include "batch.h"

namespace svr_gc {

using Ctx = emp::SH2PCSession::ctx_t;
using I32 = emp::Int_T<Ctx, 32>;
using I64 = emp::Int_T<Ctx, 64>;
using U32 = emp::UInt_T<Ctx, 32>;
using U64 = emp::UInt_T<Ctx, 64>;
constexpr int FRAC = 14;
constexpr double ONE = 1 << FRAC;

inline I64 to_i64(const U64& u) { return I64::from_wires(*u.context(), u.data()); }
inline I32 low32(const I64& v) { return I32::from_wires(*v.context(), v.data()); }
inline I64 sext64(const I32& v) { return v.template sext<64>(); }

// additive int64 shares -> secret I64 (both parties call)
inline std::vector<I64> input_shared(emp::SH2PCSession& s, const std::vector<int64_t>& mine) {
    auto u = batch::input_shared(s, mine, mine.size());
    std::vector<I64> out; out.reserve(u.size());
    for (auto& x : u) out.push_back(to_i64(x));
    return out;
}
// secret I64 -> additive shares; B1's mask is uniform in [-2^mask_bits, 2^mask_bits]
inline std::vector<int64_t> to_additive(emp::SH2PCSession& s, const std::vector<I64>& v, int mask_bits) {
    const size_t n = v.size();
    std::vector<int64_t> r(n, 0);
    if (s.party() == emp::ALICE) {
        emp::PRG prg; std::vector<uint64_t> raw(n); prg.random_data(raw.data(), 8 * n);
        const uint64_t span = (2ULL << mask_bits) + 1;
        for (size_t i = 0; i < n; i++) r[i] = (int64_t)(raw[i] % span) - (int64_t(1) << mask_bits);
    }
    auto R = input_shared(s, r);
    std::vector<typename Ctx::Wire> w; w.reserve(64 * n);
    for (size_t i = 0; i < n; i++) { I64 d = v[i] - R[i]; for (int b = 0; b < 64; b++) w.push_back(d.w[b]); }
    std::unique_ptr<bool[]> o(new bool[64 * n + 1]());
    s.reveal_bits(o.get(), emp::BOB, w.data(), w.size());
    if (s.party() == emp::ALICE) return r;
    std::vector<int64_t> out(n, 0);
    for (size_t i = 0; i < n; i++) { uint64_t x = 0; for (int b = 0; b < 64; b++) x |= (uint64_t)o[64 * i + b] << b; out[i] = (int64_t)x; }
    return out;
}

// 2^-y for y >= 0 given in Q28 (as I64); result Q14
inline I32 exp2neg(const I64& yq28) {
    auto& c = *yq28.context();
    I64 k = yq28 >> 28;
    I64 f = (yq28 & I64::constant(c, (int64_t(1) << 28) - 1)) >> 14;     // fraction, Q14
    I32 fx = low32(f);
    static const int32_t C[5] = {112, -872, 3923, -11355, 16384};       // 2^-f, highest degree first (Q14)
    I32 p = I32::constant(c, C[0]);
    for (int i = 1; i < 5; i++) p = ((p * fx) >> FRAC) + I32::constant(c, C[i]);
    U32 sh = low32(k).as_unsigned();
    return p >> sh;                                                      // shifts >= 32 give 0 (p > 0)
}

struct Model {        // secret model, held as GC wires after input
    std::vector<std::array<I32, 3>> sv;
    std::vector<I32> alpha;
    I32 b;
    uint64_t gc;      // public: round(gamma * log2(e) * 2^32)
};

// x: three normalised inputs (Q14), clamped by the caller to [-7, 7]. Differences are squared at Q12 so that
// |d| <= 8 gives d^2 <= 2^30 and the sum of three stays below 2^32.
inline I32 eval(const Model& M, const std::array<I32, 3>& x) {
    auto& c = *x[0].context();
    I32 acc = M.b;
    U64 G = U64::constant(c, M.gc);
    for (size_t k = 0; k < M.sv.size(); k++) {
        U64 d2 = U64::constant(c, 0);
        for (int m = 0; m < 3; m++) { I32 d = (x[m] - M.sv[k][m]) >> 2; d2 = d2 + (d * d).as_unsigned().template zext<64>(); }   // Q24, < 2^32
        I64 y = to_i64((d2 * G) >> 28);                                   // gamma log2e ||x - sv||^2, Q28
        acc = acc + ((M.alpha[k] * exp2neg(y)) >> FRAC);
    }
    return acc;
}

inline I32 clamp(const I32& v, int32_t lo, int32_t hi) {
    auto& c = *v.context();
    I32 L = I32::constant(c, lo), H = I32::constant(c, hi);
    I32 r = v.select(v < L, L);
    return r.select(r > H, H);
}

}  // namespace svr_gc
