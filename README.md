# DRCF in C++: two-broker pipeline (EMP-toolkit, SEAL, OpenSSL)

2-party distributed AEAD from DRCF (ACM TCPS 2026, Def. 5). ASCON-128 (v1.2) runs inside a
Yao garbled circuit on EMP sh2pc. B1 = ALICE (garbler), B2 = BOB (evaluator).
K = K1 ^ K2 and the plaintext never exist in either broker.

- `dist_dec(K_i, N, AD, C||tag)` -> `{ok, M_i}` with M_1 ^ M_2 = M. On a bad tag, both get ⊥ and the plaintext is never opened.
- `dist_enc(K_i, N, AD, Y_i)` -> C||tag of Y = Y_1 ^ Y_2, known to both.

## Components
| Paper step | File | How |
|---|---|---|
| DistAEAD.Enc / Dec | `include/distaead.h`, `include/ascon_circuit.h` | ASCON-128 inside a Yao GC (EMP sh2pc, half-gates) |
| XOR ↔ additive shares, thresholds | `include/convert.h` | GC (EMP `UInt_T`), signed 64-bit additive shares |
| Joint evaluation (regression) | `include/he.h` | CKKS (SEAL 4.1.2), fresh keys per broker per evaluation, encrypted cross terms, Y(i) = Dec(sk_i, C_T) + δ_i (Sec. 4.1.3) |
| TS.KeyGen / TS.Sign | `include/dkls23.h`, `include/tsig.h` | Threshold ECDSA per Doerner, Kondi, Lee, shelat (S&P 2024) Protocol 3.4 with n = t = 2, and 2-party Schnorr, secp256k1 |
| Full pipeline | `src/drcf_demo.cpp` | Publish → Compute → Subscribe, writes `trace.js` for the dashboard |

## Files
- `include/ascon_ref.h`: plain ASCON-128 (publishers and subscribers).
- `include/ascon_circuit.h`: ASCON as a circuit over `emp::Bit_T`. Public bits are folded as constants, so they cost no gates.
- `include/distaead.h`: `dist_dec`, `dist_enc` over `SH2PCSession`. Shares come out via EMP's XOR reveal.
- `test/kat.h`: 61 known-answer vectors from the official Python reference (pyascon).
- `test/test_ref.cpp`, `test/test_distaead.cpp`, `src/bench.cpp`.

## Build
Needs EMP-toolkit (emp-tool, emp-ot, emp-sh2pc; current master, C++20 `SH2PCSession` API), Microsoft SEAL 4.1 and OpenSSL 3.

## CKKS joint evaluation (Sec. 4.1.3)
`include/he.h` follows the paper: real-valued shares X = X1 + X2 and β = β1 + β2. For each joint evaluation every broker generates fresh CKKS keys, encrypts its own share, and the brokers exchange encrypted cross terms (C(1)·β2 + δ, C(2)·β1 + δ′). Each broker outputs Y(i) = Dec(sk_i, C_T) + δ_i; the keys are erased afterwards. For thresholds, each broker rounds Y(i) to fixed point and the brokers continue in a garbled circuit.

Parameters: N = 8192, coeff modulus {60, 36, 60}, scale 2^36 (128-bit security per SEAL's tables). Values must stay below q0 / (2·scale) = 2^23; the masks (δ up to 2^20, reading shares up to 2^16, β shares up to 64) keep inside that: 64·2^16 + 2^20 < 2^23. Measured error: at most 1e-3 on 400 inner products (test_compute) and 9e-4 Wh on the subscriber's result (drcf_demo), which is the mWh rounding before the GC.

## Threshold ECDSA (DKLs23)
`include/dkls23.h` follows Protocol 3.4 of "Threshold ECDSA in Three Rounds" (Doerner, Kondi, Lee, shelat, IEEE S&P 2024) for two parties:
- Setup: commit-release key generation (FRelaxedKeyGen for n = t = 2), additive shares sk = sk1 + sk2, FZero PRF key.
- Round 1: instance key r_i, inversion mask φ_i, commitment to R_i, RVOLE instance toward the other party.
- Round 2: sk_i rerandomised with the FZero mask, RVOLE multiplication of {r_i, sk_i}, check-adjust message (Γu, Γv, ψ, pk_i), decommit R_i.
- Round 3: consistency checks χ·R_j − Γu = du·G and χ·pk_j − Γv = dv·G, Σpk = pk, then u_i, v_i, w_i; s = Σw / Σu, output only if ECDSA verification passes.

Realisation choices: FCom is a salted SHA-256 commitment. FRVOLE(q, 2) is built from EMP's SoftSpokenOT (malicious-secure OT extension) with Bob's randomised gadget encoding of χ (length κ + 2λs = 416). The paper's own optimised two-round VOLE is in its full version and is not reproduced. The test (`test_dkls23`) also checks that a party committing to a wrong R_i, or sending a wrong pk_i, is caught by the step-8 checks.
```
# EMP (once)
for r in emp-tool emp-ot emp-sh2pc; do
  git clone https://github.com/emp-toolkit/$r.git
  cmake -S $r -B $r/build -DCMAKE_INSTALL_PREFIX=$HOME/emp -DCMAKE_PREFIX_PATH=$HOME/emp -DCMAKE_BUILD_TYPE=Release
  cmake --build $r/build -j && cmake --install $r/build
done
# SEAL (once)
git clone --branch v4.1.2 https://github.com/microsoft/SEAL.git
cmake -S SEAL -B SEAL/build -DCMAKE_INSTALL_PREFIX=$HOME/seal -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF
cmake --build SEAL/build -j && cmake --install SEAL/build
# this project
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/emp;$HOME/seal" && cmake --build build -j
```

## Run
```
./build/test_ref                     # 61/61 vs KAT
./run.sh build/test_distaead         # B1 + B2: publisher -> dist_dec -> dist_enc -> subscriber, tamper check
./run.sh build/bench 10              # cost table
./run.sh build/test_dkls23           # DKLs23 threshold ECDSA: honest runs + two cheating parties caught
./run.sh build/test_tsig             # 2-party Schnorr
./run.sh build/test_compute          # B2A, HE cross terms, A2B, threshold
cd build && ../run.sh ./drcf_demo    # whole DRCF pipeline, writes trace.js
```
For two machines, set `EMP_PEER_IP` and `EMP_PORT` and start party 1 and party 2 separately.

## Results (localhost, 2 cores, 10 reps, AD = 5 bytes)
| op | msg bytes | AND gates | comm (KiB) | ms/op |
|---|---|---|---|---|
| Enc | 8 | 11,264 | 352 | 1.8 |
| Dec | 8 | 11,391 | 356 | 1.6 |
| Enc | 64 | 24,704 | 799 | 4.2 |
| Dec | 64 | 24,830 | 776 | 3.8 |
| Enc | 256 | 70,784 | 2,242 | 7.6 |
| Dec | 256 | 70,910 | 2,216 | 7.5 |

Each permutation round is 320 AND gates. With padding, a = ⌊|AD|/8⌋+1 AD blocks (0 if AD is empty) and m = ⌊|M|/8⌋+1 message blocks,
ANDs = (24 + 6a + 6(m − 1))·320 − 256 (the first round partly folds, since IV and nonce are public), plus 127 for the tag check in Dec.
Check: 8-byte message, 5-byte AD: (24 + 6 + 6)·320 − 256 = 11,264.

## Full pipeline (drcf_demo, 3 meters × 168 h, localhost)
504 publications, 432 forecasts, 1 subscriber request. About 1 s end to end, 195 MiB between the brokers, 6.3 M AND gates.
DistAEAD.Dec is 84% of the time. CKKS cross terms for all 432 forecasts: 36 ms including fresh key generation (12 ms). DKLs23 ECDSA: about 13 ms and 135 KiB per signature after setup. 2-party Schnorr: 0.5 ms.

## Caveats
- Semi-honest (as in the paper). Not audited. Not constant-time. Localhost numbers have no network latency.
- CKKS: shares and masks are real numbers with bounded range, so hiding is statistical and limited by CKKS headroom: about 4 bits of margin for readings (mask 2^16 vs readings up to about 3,000 Wh) and about 6 to 8 bits for the AR coefficients (mask 64), but under 1 bit for the intercept (45.6). Larger N or a bigger q0 buys more margin. The evaluator's reply has no noise flooding, so a key owner who inspects CKKS noise could learn a little about the other's β share. Results are approximate (see above).
- ECDSA: the DKLs23 consistency checks are implemented; the VOLE is OT-based but is not the paper's exact two-round VOLE (full version).
- The paper's Sec. 4.1.4 describes a Schnorr-style threshold signature while Sec. 5 implements threshold ECDSA (DKLs23). Both are here.
- drcf_demo's harness gives both processes the same seed (synthetic data) and simulates publishers and the subscriber inside B1's process. Brokers only feed their own shares into the protocols.
