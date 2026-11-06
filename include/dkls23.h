// Two-party threshold ECDSA following Doerner, Kondi, Lee, shelat,
// "Threshold ECDSA in Three Rounds", IEEE S&P 2024 (DKLs23), Protocol 3.4 with n = t = 2.
// Curve: secp256k1 (as in the paper's implementation). Party P1 = emp::ALICE, P2 = emp::BOB.
//
// Mapping to the paper:
//   FRelaxedKeyGen  : commit-release of X_i = x_i*G; pk = X_1 + X_2 (2-of-2, so Lagrange coefficients are 1).
//   FCom            : salted SHA-256 commitment (the folklore ROM realisation the paper cites).
//   FZero(Zq, 2)    : PRF key agreed at setup; zeta_1 = +H(key, sigid), zeta_2 = -H(key, sigid).
//   FRVOLE(q, 2)    : Bob samples chi; Alice inputs {r_i, sk_i}; outputs c + d = a * chi.
//                     Realised here with OT (EMP SoftSpokenOT, malicious-secure OT extension) and
//                     Bob's randomised gadget encoding of chi (length kappa + 2*lambda_s = 416), as in
//                     the Doerner et al. OT-multiplication line of work. The paper's own optimised
//                     two-round VOLE and its proof are in its full version and are not reproduced here.
//   Steps 5-10      : r_i, phi_i, commitments to R_i, check-adjust message (Gamma^u, Gamma^v, psi, pk_i),
//                     the two consistency checks of step 8, u_i, v_i, w_i, and s = sum(w) / sum(u),
//                     output only if ECDSAVerify passes.
#pragma once
#include <string>
#include "tsig.h"
#include "emp-ot/emp-ot.h"

namespace dkls23 {

using tsig::bytes; using tsig::BNp; using tsig::Pt; using tsig::Curve;
constexpr int KAPPA = 256, LAMBDA_S = 80, XI = KAPPA + 2 * LAMBDA_S;   // gadget length

// ordered exchange: P1 sends first, P2 receives first (no deadlock on any channel)
inline bytes xchg(emp::IOChannel* io, int party, const bytes& mine, size_t n) {
    bytes other;
    if (party == emp::ALICE) { tsig::send(io, mine); other = tsig::recv(io, n); }
    else { other = tsig::recv(io, n); tsig::send(io, mine); }
    return other;
}
inline void put(bytes& b, const bytes& x) { b.insert(b.end(), x.begin(), x.end()); }
inline bytes take(const bytes& b, size_t& off, size_t n) { bytes x(b.begin() + off, b.begin() + off + n); off += n; return x; }
inline bytes rand_bytes(size_t n) { bytes b(n); RAND_bytes(b.data(), (int)n); return b; }

// ---------- FCom: salted hash commitment ----------
struct Opening { bytes salt, data; };
inline bytes commit_of(const Opening& o) { bytes m = {'c', 'o', 'm'}; put(m, o.salt); put(m, o.data); return tsig::sha256(m); }
inline Opening make_opening(const bytes& data) { return {rand_bytes(32), data}; }

// ---------- OT channels: one per direction ----------
// ot_p1s: P1 is OT sender (P2 receiver). ot_p2s: P2 is OT sender (P1 receiver).
struct OTPair {
    std::unique_ptr<emp::OT> ot_p1s, ot_p2s;
    OTPair(int party, emp::IOChannel* io) {
        ot_p1s.reset(new emp::SoftSpoken<8>(party, io, true));
        ot_p2s.reset(new emp::SoftSpoken<8>(party == emp::ALICE ? emp::BOB : emp::ALICE, io, true));
    }
};

// ---------- Key ----------
struct Key {
    BNp x;          // additive share of sk
    Pt pk;          // joint public key
    bytes zero_key; // FZero PRF key
    uint64_t next_sigid = 0;
};

// FRelaxedKeyGen for n = t = 2: commit to (X_i, zero-seed_i), then release.
inline Key keygen(Curve& C, emp::IOChannel* io, int party) {
    Key k{C.rand_scalar(), nullptr, {}};
    Pt X = C.mulG(k.x.get());
    bytes seed = rand_bytes(32), data = C.enc(X.get()); put(data, seed);
    Opening op = make_opening(data);
    bytes other_com = xchg(io, party, commit_of(op), 32);
    bytes mine = op.salt; put(mine, op.data);
    bytes other = xchg(io, party, mine, 32 + 33 + 32);
    Opening oo{bytes(other.begin(), other.begin() + 32), bytes(other.begin() + 32, other.end())};
    if (commit_of(oo) != other_com) throw std::runtime_error("keygen: commitment does not open");
    Pt Xo = C.dec(bytes(oo.data.begin(), oo.data.begin() + 33));
    k.pk = C.add(X.get(), Xo.get());
    if (EC_POINT_is_at_infinity(C.g, k.pk.get())) throw std::runtime_error("keygen: pk at infinity");
    bytes s1 = party == emp::ALICE ? seed : bytes(oo.data.begin() + 33, oo.data.end());
    bytes s2 = party == emp::ALICE ? bytes(oo.data.begin() + 33, oo.data.end()) : seed;
    bytes zk = {'z', 'e', 'r', 'o'}; put(zk, s1); put(zk, s2);
    k.zero_key = tsig::sha256(zk);
    return k;
}

// ---------- FRVOLE(q, 2) over OT ----------
// gadget vector: g_i = 2^i for i < 256, then public pseudo-random scalars tied to the session
inline std::vector<BNp> gadget(Curve& C, const bytes& sid) {
    std::vector<BNp> g;
    for (int i = 0; i < XI; i++) {
        BNp x = C.bn();
        if (i < KAPPA) { BN_zero(x.get()); BN_set_bit(x.get(), i); }
        else { bytes m = {'g'}; put(m, sid); m.push_back((uint8_t)(i >> 8)); m.push_back((uint8_t)i); x = C.b2s(tsig::sha256(m)); }
        g.push_back(std::move(x));
    }
    return g;
}
inline bytes pad(const emp::block& k, int i, int comp, const bytes& sid) {
    bytes m((uint8_t*)&k, (uint8_t*)&k + 16); m.push_back((uint8_t)(i >> 8)); m.push_back((uint8_t)i); m.push_back((uint8_t)comp); put(m, sid);
    return tsig::sha256(m);
}

struct VoleBob { BNp chi; std::unique_ptr<bool[]> beta; std::vector<emp::block> keys; };
struct VoleAlice { std::vector<emp::block> k0, k1; };

// Round 1, Bob side: sample chi, encode it, run OT as receiver with the encoding as choice bits.
inline VoleBob vole_bob_sample(Curve& C, emp::OT* ot, const std::vector<BNp>& g) {
    VoleBob b{C.rand_scalar(), std::unique_ptr<bool[]>(new bool[XI]), std::vector<emp::block>(XI)};
    BNp chip = C.bn(); BN_copy(chip.get(), b.chi.get());
    BNp t = C.bn();
    for (int i = KAPPA; i < XI; i++) {
        uint8_t r; RAND_bytes(&r, 1); b.beta[i] = r & 1;
        if (b.beta[i]) BN_mod_sub(chip.get(), chip.get(), g[i].get(), C.q, C.ctx);
    }
    for (int i = 0; i < KAPPA; i++) b.beta[i] = BN_is_bit_set(chip.get(), i);   // sum_i g_i beta_i = chi
    ot->recv(b.keys.data(), b.beta.get(), XI);
    return b;
}
// Round 1, Alice side: OT as sender on random keys.
inline VoleAlice vole_alice_setup(emp::OT* ot) {
    VoleAlice a{std::vector<emp::block>(XI), std::vector<emp::block>(XI)};
    emp::PRG prg; prg.random_block(a.k0.data(), XI); prg.random_block(a.k1.data(), XI);
    ot->send(a.k0.data(), a.k1.data(), XI);
    return a;
}
// Round 2, Alice: input a = {a0, a1}; returns her shares c and the message for Bob.
inline bytes vole_alice_multiply(Curve& C, const VoleAlice& A, const std::vector<BNp>& g, const BIGNUM* a0, const BIGNUM* a1,
                                 const bytes& sid, BNp& c0, BNp& c1) {
    const BIGNUM* a[2] = {a0, a1};
    BNp* c[2] = {&c0, &c1};
    for (int k = 0; k < 2; k++) { *c[k] = C.bn(); BN_zero(c[k]->get()); }
    bytes msg; msg.reserve((size_t)XI * 128);
    BNp rho = C.bn(), m1 = C.bn();
    for (int i = 0; i < XI; i++)
        for (int k = 0; k < 2; k++) {
            BN_rand_range(rho.get(), C.q);
            BN_mod_mul(m1.get(), g[i].get(), a[k], C.q, C.ctx); BN_mod_add(m1.get(), m1.get(), rho.get(), C.q, C.ctx);
            BN_mod_sub(c[k]->get(), c[k]->get(), rho.get(), C.q, C.ctx);
            bytes e0 = C.s2b(rho.get()), e1 = C.s2b(m1.get()), p0 = pad(A.k0[i], i, k, sid), p1 = pad(A.k1[i], i, k, sid);
            for (int j = 0; j < 32; j++) { e0[j] ^= p0[j]; e1[j] ^= p1[j]; }
            put(msg, e0); put(msg, e1);
        }
    return msg;
}
// Round 2, Bob: decode his shares d with c + d = a * chi.
inline void vole_bob_receive(Curve& C, const VoleBob& B, const bytes& msg, const bytes& sid, BNp& d0, BNp& d1) {
    BNp* d[2] = {&d0, &d1};
    for (int k = 0; k < 2; k++) { *d[k] = C.bn(); BN_zero(d[k]->get()); }
    for (int i = 0; i < XI; i++)
        for (int k = 0; k < 2; k++) {
            size_t off = ((size_t)i * 2 + k) * 64 + 32 * B.beta[i];
            bytes e(msg.begin() + off, msg.begin() + off + 32), p = pad(B.keys[i], i, k, sid);
            for (int j = 0; j < 32; j++) e[j] ^= p[j];
            BN_mod_add(d[k]->get(), d[k]->get(), C.b2s(e).get(), C.q, C.ctx);
        }
}

// ---------- Signing (Protocol 3.4, steps 4-10) ----------
enum class Cheat { None, InconsistentR, InconsistentPk };   // test hooks: deviate in round 2
struct Sig { bytes r, s; };
struct SignResult { bool ok; Sig sig; std::string failure; };

inline SignResult sign(Curve& C, emp::IOChannel* io, OTPair& ots, int party, Key& key, const bytes& msg, Cheat cheat = Cheat::None) {
    const bool p1 = party == emp::ALICE;
    uint64_t sigid = key.next_sigid++;
    bytes sid = {'s', 'i', 'g'}; for (int i = 0; i < 8; i++) sid.push_back((uint8_t)(sigid >> (8 * i)));
    auto g = gadget(C, sid);
    BNp z = C.b2s(tsig::sha256(msg));   // SHA2(m) as a scalar

    // ---- step 5-6 (round 1): instance key, inversion mask, commit to R_i, sample RVOLE, FZero ----
    BNp r = C.rand_scalar(), phi = C.rand_scalar();
    Pt Ri = C.mulG(r.get());
    if (cheat == Cheat::InconsistentR) Ri = C.add(Ri.get(), C.mulG(BN_value_one()).get());   // commits to R_i != r_i * G
    Opening opR = make_opening(C.enc(Ri.get()));
    bytes comR_other = xchg(io, party, commit_of(opR), 32);
    // RVOLE instances: (Bob = P1, Alice = P2) on ot_p2s, (Bob = P2, Alice = P1) on ot_p1s
    VoleAlice va; VoleBob vb;
    if (p1) { va = vole_alice_setup(ots.ot_p1s.get()); vb = vole_bob_sample(C, ots.ot_p2s.get(), g); }
    else { vb = vole_bob_sample(C, ots.ot_p1s.get(), g); va = vole_alice_setup(ots.ot_p2s.get()); }
    bytes zm = key.zero_key; put(zm, sid);
    BNp zeta = C.b2s(tsig::sha256(zm));
    if (!p1) BN_mod_sub(zeta.get(), C.q, zeta.get(), C.q, C.ctx);

    // ---- step 7 (round 2): sk_i, multiply {r_i, sk_i}, Gamma, psi, pk_i, decommit ----
    BNp ski = C.bn(); BN_mod_add(ski.get(), key.x.get(), zeta.get(), C.q, C.ctx);
    BNp cu, cv;
    bytes vmsg = vole_alice_multiply(C, va, g, r.get(), ski.get(), sid, cu, cv);
    Pt Gu = C.mulG(cu.get()), Gv = C.mulG(cv.get()), pki = C.mulG(ski.get());
    if (cheat == Cheat::InconsistentPk) pki = C.add(pki.get(), C.mulG(BN_value_one()).get());
    BNp psi = C.bn(); BN_mod_sub(psi.get(), phi.get(), vb.chi.get(), C.q, C.ctx);
    bytes r2; put(r2, opR.salt); put(r2, opR.data); put(r2, C.enc(Gu.get())); put(r2, C.enc(Gv.get())); put(r2, C.s2b(psi.get())); put(r2, C.enc(pki.get())); put(r2, vmsg);
    bytes o2 = xchg(io, party, r2, r2.size());

    // ---- step 8 (round 3): open R_j, decode RVOLE, consistency checks, fragments ----
    size_t off = 0;
    Opening oR{take(o2, off, 32), take(o2, off, 33)};
    Pt Guj = C.dec(take(o2, off, 33)), Gvj = C.dec(take(o2, off, 33));
    BNp psij = C.b2s(take(o2, off, 32));
    Pt pkj = C.dec(take(o2, off, 33));
    bytes vmsg_j = take(o2, off, (size_t)XI * 128);
    BNp du, dv; vole_bob_receive(C, vb, vmsg_j, sid, du, dv);
    std::string fail;
    Pt Rj;
    if (commit_of(oR) != comR_other) fail = "R_j does not match its commitment";
    else {
        Rj = C.dec(oR.data);
        // chi_{i,j} * R_j - Gamma^u_{j,i} == d^u_{i,j} * G   and   chi_{i,j} * pk_j - Gamma^v_{j,i} == d^v_{i,j} * G
        Pt lu = C.mul(Rj.get(), vb.chi.get()), lv = C.mul(pkj.get(), vb.chi.get());
        EC_POINT_invert(C.g, Guj.get(), C.ctx); EC_POINT_invert(C.g, Gvj.get(), C.ctx);
        if (!C.eq(C.add(lu.get(), Guj.get()).get(), C.mulG(du.get()).get())) fail = "consistency check on R_j failed";
        else if (!C.eq(C.add(lv.get(), Gvj.get()).get(), C.mulG(dv.get()).get())) fail = "consistency check on pk_j failed";
        else if (!C.eq(C.add(pki.get(), pkj.get()).get(), key.pk.get())) fail = "pk_i + pk_j != pk";
    }
    bytes r3(1 + 64, 0);
    BNp u = C.bn(), w = C.bn();
    if (fail.empty()) {
        Pt R = C.add(Ri.get(), Rj.get());
        BNp rxf = C.bn(), rx = C.bn();
        EC_POINT_get_affine_coordinates(C.g, R.get(), rxf.get(), nullptr, C.ctx);
        BN_nnmod(rx.get(), rxf.get(), C.q, C.ctx);
        BNp m = C.bn(), v = C.bn(), t = C.bn();
        BN_mod_add(m.get(), phi.get(), psij.get(), C.q, C.ctx);                     // phi_i + psi_{j,i}
        BN_mod_mul(u.get(), r.get(), m.get(), C.q, C.ctx); BN_mod_add(u.get(), u.get(), cu.get(), C.q, C.ctx); BN_mod_add(u.get(), u.get(), du.get(), C.q, C.ctx);
        BN_mod_mul(v.get(), ski.get(), m.get(), C.q, C.ctx); BN_mod_add(v.get(), v.get(), cv.get(), C.q, C.ctx); BN_mod_add(v.get(), v.get(), dv.get(), C.q, C.ctx);
        BN_mod_mul(w.get(), z.get(), phi.get(), C.q, C.ctx); BN_mod_mul(t.get(), rx.get(), v.get(), C.q, C.ctx); BN_mod_add(w.get(), w.get(), t.get(), C.q, C.ctx);
        r3[0] = 1; bytes wb = C.s2b(w.get()), ub = C.s2b(u.get());
        std::copy(wb.begin(), wb.end(), r3.begin() + 1); std::copy(ub.begin(), ub.end(), r3.begin() + 33);
        // ---- step 10: assemble ----
        bytes o3 = xchg(io, party, r3, r3.size());
        if (!o3[0]) return {false, {}, "peer reported failure"};
        BNp W = C.bn(), U = C.bn(), Ui = C.bn(), s = C.bn();
        BN_mod_add(W.get(), w.get(), C.b2s(bytes(o3.begin() + 1, o3.begin() + 33)).get(), C.q, C.ctx);
        BN_mod_add(U.get(), u.get(), C.b2s(bytes(o3.begin() + 33, o3.end())).get(), C.q, C.ctx);
        BN_mod_inverse(Ui.get(), U.get(), C.q, C.ctx);
        BN_mod_mul(s.get(), W.get(), Ui.get(), C.q, C.ctx);
        Sig sig{C.s2b(rx.get()), C.s2b(s.get())};
        return {true, sig, ""};
    }
    xchg(io, party, r3, r3.size());   // step 8 failure: send (fail) instead of a fragment
    return {false, {}, fail};
}

// standard ECDSA verification (OpenSSL)
inline bool verify(Curve& C, const EC_POINT* pk, const bytes& msg, const Sig& sig) {
    EC_KEY* ek = EC_KEY_new_by_curve_name(C.nid);
    EC_KEY_set_public_key(ek, pk);
    ECDSA_SIG* es = ECDSA_SIG_new();
    ECDSA_SIG_set0(es, BN_bin2bn(sig.r.data(), 32, nullptr), BN_bin2bn(sig.s.data(), 32, nullptr));
    bytes h = tsig::sha256(msg);
    int ok = ECDSA_do_verify(h.data(), 32, es, ek);
    ECDSA_SIG_free(es); EC_KEY_free(ek);
    return ok == 1;
}

// Step 10 as the paper states it: output the signature only if ECDSAVerify passes.
inline SignResult sign_and_check(Curve& C, emp::IOChannel* io, OTPair& ots, int party, Key& key, const bytes& msg, Cheat cheat = Cheat::None) {
    auto r = sign(C, io, ots, party, key, msg, cheat);
    if (r.ok && !verify(C, key.pk.get(), msg, r.sig)) return {false, {}, "assembled signature does not verify"};
    return r;
}

}  // namespace dkls23
