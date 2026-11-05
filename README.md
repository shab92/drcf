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
| XOR ↔ additive shares, thresholds | `include/convert.h` | GC (EMP `UInt_T`), arithmetic mod the BFV plaintext modulus t |
| Linear models (AR / regression) | `include/he.h` | SEAL BFV, own keys per broker, masked cross terms β₂x₁, β₁x₂ |
| TS.KeyGen / TS.Sign | `include/tsig.h` | 2-party ECDSA (multiplicative shares, Gilboa OT-MtA on EMP IKNP) and 2-party Schnorr, P-256 |
| Full pipeline | `src/drcf_demo.cpp` | Publish → Compute → Subscribe, writes `trace.js` for the dashboard |

## Files
- `include/ascon_ref.h`: plain ASCON-128 (publishers and subscribers).
- `include/ascon_circuit.h`: ASCON as a circuit over `emp::Bit_T`. Public bits are folded as constants, so they cost no gates.
- `include/distaead.h`: `dist_dec`, `dist_enc` over `SH2PCSession`. Shares come out via EMP's XOR reveal.
- `test/kat.h`: 61 known-answer vectors from the official Python reference (pyascon).
- `test/test_ref.cpp`, `test/test_distaead.cpp`, `src/bench.cpp`.

## Build
Needs EMP-toolkit (emp-tool, emp-ot, emp-sh2pc; current master, C++20 `SH2PCSession` API), Microsoft SEAL 4.1 and OpenSSL 3.
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
./run.sh build/test_tsig             # 2-party ECDSA + Schnorr, verified with OpenSSL
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
DistAEAD.Dec is 84% of the time. HE cross terms for all 432 forecasts: 28 ms. 2-party ECDSA: 27 ms. 2-party Schnorr: 0.3 ms.

## Caveats
- Semi-honest (as in the paper). Not audited. Not constant-time. Localhost numbers have no network latency.
- HE: the helper's reply is masked with fresh ρ, but there's no noise flooding. For circuit privacy against a key owner who inspects noise, add flooding or switch to a scheme with circuit privacy.
- ECDSA: semi-honest MtA. Malicious security needs the DKLs checks (Doerner et al.) or Paillier proofs.
- The paper's Sec. 4.1.4 describes a Schnorr-style threshold signature while Sec. 5 implements threshold ECDSA. Both are here.
- drcf_demo's harness gives both processes the same seed (synthetic data) and simulates publishers and the subscriber inside B1's process. Brokers only feed their own shares into the protocols.
