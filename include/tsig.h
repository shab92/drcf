// Curve helpers (OpenSSL, secp256k1 by default) and 2-party Schnorr, for DRCF.Subscribe.
//   Schnorr2: additive key x = x1 + x2, R = R1 + R2, s = s1 + s2   (matches Sec. 4.1.4 structure)
// The threshold ECDSA used by DRCF (Doerner-Kondi-Lee-shelat, S&P 2024) is in dkls23.h.
// Party 1 = ALICE = B1, party 2 = BOB = B2. All messages go over the given IOChannel.
#pragma once
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/obj_mac.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include "emp-ot/emp-ot.h"

namespace tsig {

using bytes = std::vector<uint8_t>;
struct BNf { void operator()(BIGNUM* b) const { BN_clear_free(b); } };
struct ECPf { void operator()(EC_POINT* p) const { EC_POINT_free(p); } };
using BNp = std::unique_ptr<BIGNUM, BNf>;
using Pt = std::unique_ptr<EC_POINT, ECPf>;

struct Curve {
    EC_GROUP* g; BN_CTX* ctx; BIGNUM* q;
    int nid;
    explicit Curve(int nid_ = NID_secp256k1) : g(EC_GROUP_new_by_curve_name(nid_)), ctx(BN_CTX_new()), q(BN_new()), nid(nid_) { EC_GROUP_get_order(g, q, ctx); }
    ~Curve() { BN_free(q); BN_CTX_free(ctx); EC_GROUP_free(g); }
    BNp bn() { return BNp(BN_new()); }
    Pt pt() { return Pt(EC_POINT_new(g)); }
    BNp rand_scalar() { BNp x = bn(); do BN_rand_range(x.get(), q); while (BN_is_zero(x.get())); return x; }
    Pt mulG(const BIGNUM* k) { Pt p = pt(); EC_POINT_mul(g, p.get(), k, nullptr, nullptr, ctx); return p; }
    Pt mul(const EC_POINT* P, const BIGNUM* k) { Pt p = pt(); EC_POINT_mul(g, p.get(), nullptr, P, k, ctx); return p; }
    Pt add(const EC_POINT* a, const EC_POINT* b) { Pt p = pt(); EC_POINT_add(g, p.get(), a, b, ctx); return p; }
    bool eq(const EC_POINT* a, const EC_POINT* b) { return EC_POINT_cmp(g, a, b, ctx) == 0; }
    bytes enc(const EC_POINT* P) { bytes b(33); EC_POINT_point2oct(g, P, POINT_CONVERSION_COMPRESSED, b.data(), 33, ctx); return b; }
    Pt dec(const bytes& b) { Pt p = pt(); if (!EC_POINT_oct2point(g, p.get(), b.data(), b.size(), ctx)) throw std::runtime_error("bad point"); return p; }
    bytes s2b(const BIGNUM* x) { bytes b(32); BN_bn2binpad(x, b.data(), 32); return b; }
    BNp b2s(const bytes& b) { BNp x = bn(); BN_bin2bn(b.data(), (int)b.size(), x.get()); BN_mod(x.get(), x.get(), q, ctx); return x; }
    BNp hash(const std::vector<bytes>& parts) {
        SHA256_CTX c; SHA256_Init(&c);
        for (auto& p : parts) { uint32_t n = (uint32_t)p.size(); SHA256_Update(&c, &n, 4); SHA256_Update(&c, p.data(), p.size()); }
        bytes h(32); SHA256_Final(h.data(), &c); return b2s(h);
    }
};

inline void send(emp::IOChannel* io, const bytes& b) { io->send_data(b.data(), b.size()); io->flush(); }
inline bytes recv(emp::IOChannel* io, size_t n) { bytes b(n); io->recv_data(b.data(), n); return b; }
inline bytes sha256(const bytes& m) { bytes h(32); SHA256(m.data(), m.size(), h.data()); return h; }

// ======================= 2-party Schnorr =======================
struct SchnorrKey { BNp x_i; Pt pk; };
struct SchnorrSig { bytes R; bytes s; };

inline SchnorrKey schnorr_keygen(Curve& C, emp::IOChannel* io, int party) {
    SchnorrKey k{C.rand_scalar(), nullptr};
    Pt X = C.mulG(k.x_i.get());
    bytes mine = C.enc(X.get()), other;
    if (party == emp::ALICE) { send(io, mine); other = recv(io, 33); } else { other = recv(io, 33); send(io, mine); }
    k.pk = C.add(X.get(), C.dec(other).get());
    return k;
}

inline SchnorrSig schnorr_sign(Curve& C, emp::IOChannel* io, int party, const SchnorrKey& key, const bytes& msg) {
    BNp k = C.rand_scalar(); Pt Ri = C.mulG(k.get()); bytes mine = C.enc(Ri.get()), other;
    // B1 commits to R1 first so neither side can pick its nonce after seeing the other's
    if (party == emp::ALICE) { send(io, sha256(mine)); other = recv(io, 33); send(io, mine); }
    else { bytes com = recv(io, 32); send(io, mine); other = recv(io, 33); if (sha256(other) != com) throw std::runtime_error("nonce commitment mismatch"); }
    Pt R = C.add(Ri.get(), C.dec(other).get());
    bytes Rb = C.enc(R.get());
    BNp e = C.hash({Rb, C.enc(key.pk.get()), msg});
    BNp s = C.bn(); BN_mod_mul(s.get(), e.get(), key.x_i.get(), C.q, C.ctx); BN_mod_add(s.get(), s.get(), k.get(), C.q, C.ctx);
    bytes sm = C.s2b(s.get()), so;
    if (party == emp::ALICE) { send(io, sm); so = recv(io, 32); } else { so = recv(io, 32); send(io, sm); }
    BN_mod_add(s.get(), s.get(), C.b2s(so).get(), C.q, C.ctx);
    return {Rb, C.s2b(s.get())};
}

// public verification: s*G == R + e*pk
inline bool schnorr_verify(Curve& C, const EC_POINT* pk, const bytes& msg, const SchnorrSig& sig) {
    try {
        Pt R = C.dec(sig.R);
        BNp e = C.hash({sig.R, C.enc(pk), msg});
        Pt lhs = C.mulG(C.b2s(sig.s).get()), rhs = C.add(R.get(), C.mul(pk, e.get()).get());
        return C.eq(lhs.get(), rhs.get());
    } catch (...) { return false; }
}

}  // namespace tsig
