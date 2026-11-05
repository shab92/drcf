// 2-party threshold signatures on P-256 (OpenSSL), semi-honest, for DRCF.Subscribe.
//   Schnorr2: additive key x = x1 + x2, R = R1 + R2, s = s1 + s2   (matches Sec. 4.1.4 structure)
//   ECDSA2  : multiplicative key x = x1 * x2, nonce k = k1 * k2, products via Gilboa OT-MtA
//             over EMP IKNP. Output is a standard ECDSA signature verified with OpenSSL.
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
    Curve() : g(EC_GROUP_new_by_curve_name(NID_X9_62_prime256v1)), ctx(BN_CTX_new()), q(BN_new()) { EC_GROUP_get_order(g, q, ctx); }
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

// ======================= Gilboa MtA over OT =======================
// Sender (B1) holds a, receiver (B2) holds b. Output alpha (B1), beta (B2), alpha + beta = a*b mod q.
// 256 OTs on random 128-bit keys (EMP IKNP), then 32-byte one-time pads derived with SHA-256.
inline BNp mta_sender(Curve& C, emp::IOChannel* io, emp::OT* ot, const BIGNUM* a) {
    const int L = 256;
    std::vector<emp::block> k0(L), k1(L);
    emp::PRG prg; prg.random_block(k0.data(), L); prg.random_block(k1.data(), L);
    ot->send(k0.data(), k1.data(), L);
    BNp alpha = C.bn(); BN_zero(alpha.get());
    BNp r = C.bn(), m1 = C.bn(), p2 = C.bn();
    bytes buf; buf.reserve(64 * L);
    for (int j = 0; j < L; j++) {
        BN_rand_range(r.get(), C.q);
        BN_set_word(p2.get(), 0); BN_set_bit(p2.get(), j); BN_mod_mul(m1.get(), p2.get(), a, C.q, C.ctx);
        BN_mod_add(m1.get(), m1.get(), r.get(), C.q, C.ctx);              // m1 = r + 2^j a
        BN_mod_sub(alpha.get(), alpha.get(), r.get(), C.q, C.ctx);        // alpha = -sum r
        bytes c0 = C.s2b(r.get()), c1 = C.s2b(m1.get());
        bytes pad0 = sha256(bytes((uint8_t*)&k0[j], (uint8_t*)&k0[j] + 16)), pad1 = sha256(bytes((uint8_t*)&k1[j], (uint8_t*)&k1[j] + 16));
        for (int i = 0; i < 32; i++) { c0[i] ^= pad0[i]; c1[i] ^= pad1[i]; }
        buf.insert(buf.end(), c0.begin(), c0.end()); buf.insert(buf.end(), c1.begin(), c1.end());
    }
    send(io, buf);
    return alpha;
}
inline BNp mta_receiver(Curve& C, emp::IOChannel* io, emp::OT* ot, const BIGNUM* b) {
    const int L = 256;
    std::unique_ptr<bool[]> ch(new bool[L]);
    for (int j = 0; j < L; j++) ch[j] = BN_is_bit_set(b, j);
    std::vector<emp::block> k(L);
    ot->recv(k.data(), ch.get(), L);
    bytes buf = recv(io, 64 * L);
    BNp beta = C.bn(); BN_zero(beta.get());
    for (int j = 0; j < L; j++) {
        bytes c(buf.begin() + 64 * j + 32 * ch[j], buf.begin() + 64 * j + 32 * ch[j] + 32);
        bytes pad = sha256(bytes((uint8_t*)&k[j], (uint8_t*)&k[j] + 16));
        for (int i = 0; i < 32; i++) c[i] ^= pad[i];
        BN_mod_add(beta.get(), beta.get(), C.b2s(c).get(), C.q, C.ctx);
    }
    return beta;
}

// ======================= 2-party ECDSA =======================
struct EcdsaKey { BNp x_i; Pt pk; };
struct EcdsaSig { bytes r, s; };

inline EcdsaKey ecdsa_keygen(Curve& C, emp::IOChannel* io, int party) {
    EcdsaKey k{C.rand_scalar(), nullptr};
    Pt X = C.mulG(k.x_i.get());
    bytes mine = C.enc(X.get()), other;
    if (party == emp::ALICE) { send(io, mine); other = recv(io, 33); } else { other = recv(io, 33); send(io, mine); }
    k.pk = C.mul(C.dec(other).get(), k.x_i.get());   // pk = x1 x2 G
    return k;
}

inline EcdsaSig ecdsa_sign(Curve& C, emp::IOChannel* io, emp::OT* ot, int party, const EcdsaKey& key, const bytes& msg) {
    BNp z = C.b2s(sha256(msg));
    BNp k = C.rand_scalar(); Pt Ki = C.mulG(k.get());
    bytes mine = C.enc(Ki.get()), other;
    if (party == emp::ALICE) { send(io, sha256(mine)); other = recv(io, 33); send(io, mine); }
    else { bytes com = recv(io, 32); send(io, mine); other = recv(io, 33); if (sha256(other) != com) throw std::runtime_error("nonce commitment mismatch"); }
    Pt R = C.mul(C.dec(other).get(), k.get());       // R = k1 k2 G
    BNp rx = C.bn(), r = C.bn();
    EC_POINT_get_affine_coordinates(C.g, R.get(), rx.get(), nullptr, C.ctx);
    BN_nnmod(r.get(), rx.get(), C.q, C.ctx);
    BNp kinv = C.bn(); BN_mod_inverse(kinv.get(), k.get(), C.q, C.ctx);
    BNp xk = C.bn(); BN_mod_mul(xk.get(), key.x_i.get(), kinv.get(), C.q, C.ctx);
    // t1 + t2 = 1/(k1 k2),  u1 + u2 = x1 x2 / (k1 k2)
    BNp t = party == emp::ALICE ? mta_sender(C, io, ot, kinv.get()) : mta_receiver(C, io, ot, kinv.get());
    BNp u = party == emp::ALICE ? mta_sender(C, io, ot, xk.get()) : mta_receiver(C, io, ot, xk.get());
    BNp s = C.bn(), tmp = C.bn();
    BN_mod_mul(s.get(), z.get(), t.get(), C.q, C.ctx); BN_mod_mul(tmp.get(), r.get(), u.get(), C.q, C.ctx); BN_mod_add(s.get(), s.get(), tmp.get(), C.q, C.ctx);
    bytes sm = C.s2b(s.get()), so;
    if (party == emp::ALICE) { send(io, sm); so = recv(io, 32); } else { so = recv(io, 32); send(io, sm); }
    BN_mod_add(s.get(), s.get(), C.b2s(so).get(), C.q, C.ctx);
    // low-s normalisation
    BNp half = C.bn(); BN_rshift1(half.get(), C.q);
    if (BN_cmp(s.get(), half.get()) > 0) BN_sub(s.get(), C.q, s.get());
    return {C.s2b(r.get()), C.s2b(s.get())};
}

// standard ECDSA verification with OpenSSL
inline bool ecdsa_verify(Curve& C, const EC_POINT* pk, const bytes& msg, const EcdsaSig& sig) {
    EC_KEY* ek = EC_KEY_new_by_curve_name(NID_X9_62_prime256v1);
    EC_KEY_set_public_key(ek, pk);
    ECDSA_SIG* es = ECDSA_SIG_new();
    ECDSA_SIG_set0(es, BN_bin2bn(sig.r.data(), 32, nullptr), BN_bin2bn(sig.s.data(), 32, nullptr));
    bytes h = sha256(msg);
    int ok = ECDSA_do_verify(h.data(), 32, es, ek);
    ECDSA_SIG_free(es); EC_KEY_free(ek);
    return ok == 1;
}

}  // namespace tsig
