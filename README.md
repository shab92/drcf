# DRCF: A Privacy-Enhanced Distributed Publish-Subscribe System for Secure Data Computation in Untrusted Environments

*Two-broker privacy-preserving publish-subscribe, reference implementation in C++.*

Reference implementation of **DRCF**, a publish-subscribe system in which the broker role is split between two non-colluding brokers, B1 and B2. Publishers' data, keys, intermediate results and the subscriber's result are only ever held as shares; neither broker alone learns anything about them.

> S. Saderi Oskouei, K. Mandal, A. A. Ghorbani. *DRCF: A Privacy-Enhanced Distributed Publish-Subscribe System for Secure Data Computation in Untrusted Environments.* ACM Transactions on Cyber-Physical Systems 10(3), Article 36, 2026. [doi:10.1145/3805045](https://doi.org/10.1145/3805045)

**Live dashboard:** https://shab92.github.io/drcf/ (replay of a real two-broker run)

**Status: research code.** Not audited, not constant-time, not for production use. See [Security model and limitations](#security-model-and-limitations).

## What is inside

| DRCF step | File | Realisation |
|---|---|---|
| Publish: DistAEAD.Dec | `include/distaead.h`, `include/ascon_circuit.h` | ASCON-128 (v1.2) evaluated in a Yao garbled circuit (EMP sh2pc, half-gates). K = K1 ⊕ K2 never exists. A bad tag gives ⊥ to both brokers before any plaintext share is released. |
| Share conversion | `include/convert.h` | Garbled circuits: XOR shares → additive int64 shares (B2A), additive → XOR (A2B). |
| Compute: joint evaluation | `include/he.h` | CKKS (Microsoft SEAL 4.1.2). For each evaluation every broker generates fresh keys, encrypts its own share, and the brokers exchange masked encrypted cross terms. Each broker ends with Y(i) = Dec(sk_i, C_T) + δ_i. |
| Compute: decisions | `include/convert.h` | Thresholds and anomaly flags in a garbled circuit; outputs stay XOR-shared. |
| Subscribe: DistAEAD.Enc | `include/distaead.h` | ASCON-128 encryption of the shared result under the subscriber's key, in a garbled circuit. |
| Subscribe: TS.Sign | `include/dkls23.h` | 2-party threshold ECDSA on secp256k1 following Doerner, Kondi, Lee, shelat, *Threshold ECDSA in Three Rounds*, IEEE S&P 2024 (Protocol 3.4, n = t = 2). |
| Alternative signature | `include/tsig.h` | 2-party Schnorr on secp256k1. |
| End to end | `src/drcf_demo.cpp` | Publish → Compute → Subscribe on synthetic smart-meter data (load forecasting and anomaly alerts). Writes `trace.js` for the dashboard. |
| Dashboard | `docs/index.html`, `scripts/make_dashboard.py` | Self-contained page that replays a recorded run: protocol flow, what each broker sees, forecasts and alerts, costs. |

## Build

Requirements: a C++20 compiler, CMake ≥ 3.16, OpenSSL 3, [EMP-toolkit](https://github.com/emp-toolkit) (emp-tool, emp-ot, emp-sh2pc; current master with the `SH2PCSession` API) and [Microsoft SEAL 4.1.2](https://github.com/microsoft/SEAL) built with zstd.

```sh
# EMP-toolkit (once)
for r in emp-tool emp-ot emp-sh2pc; do
  git clone https://github.com/emp-toolkit/$r.git
  cmake -S $r -B $r/build -DCMAKE_INSTALL_PREFIX=$HOME/emp -DCMAKE_PREFIX_PATH=$HOME/emp -DCMAKE_BUILD_TYPE=Release
  cmake --build $r/build -j && cmake --install $r/build
done

# SEAL (once)
git clone --branch v4.1.2 https://github.com/microsoft/SEAL.git
cmake -S SEAL -B SEAL/build -DCMAKE_INSTALL_PREFIX=$HOME/seal -DCMAKE_BUILD_TYPE=Release \
      -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=ON -DSEAL_USE_ZSTD=ON
cmake --build SEAL/build -j && cmake --install SEAL/build

# DRCF
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/emp;$HOME/seal"
cmake --build build -j
```

## Run

Every two-party program takes the party number (1 = B1, 2 = B2). `run.sh` starts both locally.

```sh
./build/test_ref                      # ASCON-128 vs 61 known-answer vectors
./run.sh build/test_distaead          # DistAEAD: publisher -> dist_dec -> dist_enc -> subscriber, tamper check
./run.sh build/test_compute           # B2A, CKKS cross terms, A2B, threshold
./run.sh build/test_dkls23            # threshold ECDSA: honest runs and two cheating parties caught
./run.sh build/test_tsig              # 2-party Schnorr
./run.sh build/bench 10               # DistAEAD cost vs message size
cd build && ../run.sh ./drcf_demo     # full pipeline, writes trace.js
```

On two machines, set `EMP_PEER_IP` and `EMP_PORT`, then start party 1 and party 2 separately.

## Dashboard

`docs/index.html` replays a recorded `drcf_demo` run in the browser: the protocol flow between publishers, B1, B2 and the subscriber; what each broker sees at every step (the page XORs the two shares to show they recover the real reading); attack scenarios (tampered publication, tampered result, a compromised broker); forecasts and alerts per meter; and the cost of each step. It is a single file with no server and opens offline.

To rebuild it from your own run:

```sh
cd build && ../run.sh ./drcf_demo && cd ..
python3 scripts/make_dashboard.py build/trace.js    # writes docs/index.html
```

## Protocol notes

**DistAEAD.** ASCON-128 is written as a boolean circuit; public values (IV, nonce, associated data, ciphertext, tag) are folded as constants and cost no gates. One permutation round is 320 AND gates. With a = ⌊|AD|/8⌋+1 associated-data blocks (0 if AD is empty) and m = ⌊|M|/8⌋+1 message blocks, encryption costs (24 + 6a + 6(m − 1))·320 − 256 AND gates; decryption adds 127 for the in-circuit tag comparison. Decryption first opens only the tag bit; plaintext shares are released (as XOR shares) only if it is 1.

**CKKS joint evaluation.** Shares are real numbers: X = X1 + X2, β = β1 + β2. B1 sends C(1) = Enc(sk1, X1); B2 returns C_T = C(1)·β2 + δ with a fresh random mask δ; B1 decrypts. The symmetric exchange gives B2 the other cross term. Adding the local term X_i·β_i and subtracting its own mask, each broker holds a share Y(i) of ⟨β, X⟩. Keys are generated per evaluation and discarded. Parameters: N = 8192, coefficient modulus {40, 40, 40, 40, 40}, scale 2^40, seeded symmetric encryption, zstd serialisation. A fresh ciphertext holding 4096 values is about 6.0× the plaintext size, as reported in the paper (Table 6). Cross-term error is below 1e-5; values entering a garbled circuit are rounded to fixed point (×1000).

**Threshold ECDSA.** Follows Protocol 3.4 of DKLs23 for two parties: commit-release key generation, per-signature instance keys r_i and inversion masks φ_i, a randomised VOLE on {r_i, sk_i}, the step-8 consistency checks (χ·R_j − Γᵘ = dᵘ·G and χ·pk_j − Γᵛ = dᵛ·G), and s = Σw / Σu, released only if ECDSA verification passes. F_Com is a salted SHA-256 commitment; F_Zero uses a PRF key agreed at setup. The VOLE is built from EMP's SoftSpokenOT (malicious-secure OT extension) with a randomised gadget encoding of χ (length 416); it is not the optimised two-round VOLE from the full version of DKLs23.

## Results

Localhost, 2 vCPUs. Timings depend on hardware and network.

DistAEAD (10 repetitions, 5-byte AD):

| op | message bytes | AND gates | traffic (KiB) | ms |
|---|---|---|---|---|
| Enc | 8 | 11,264 | 352 | 1.8 |
| Dec | 8 | 11,391 | 356 | 1.6 |
| Enc | 64 | 24,704 | 799 | 4.2 |
| Dec | 64 | 24,830 | 776 | 3.8 |
| Enc | 256 | 70,784 | 2,242 | 7.6 |
| Dec | 256 | 70,910 | 2,216 | 7.5 |

Full pipeline (`drcf_demo`: 3 meters × 168 hours, 504 publications, 432 forecasts, one subscriber request): about 1 s end to end and about 195 MiB between the brokers. DistAEAD.Dec takes most of the time. CKKS cross terms for all 432 forecasts take about 70 ms, including key generation. Threshold ECDSA: about 13 ms and 135 KiB per signature after setup. Schnorr: about 0.5 ms.

## Security model and limitations

- **Adversary.** Two brokers that do not collude. Garbled circuits, share conversions and CKKS evaluation are secure against semi-honest brokers, as in the paper. Threshold signing additionally detects a broker that deviates (DKLs23 consistency checks over malicious-secure OT).
- **CKKS shares are statistically hidden.** Real-valued shares and masks have bounded range, so hiding is statistical: about 8 bits of margin for meter readings (mask 2^20), about 10 bits for model coefficients (mask 2^10), less for large coefficients such as an intercept. The evaluator's reply has no noise flooding. Results are approximate.
- **Not audited, not constant-time.** Do not use with real keys or data.
- **Demo harness.** `drcf_demo` gives both broker processes the same seed to generate synthetic data and simulates the publishers and the subscriber inside B1's process; brokers only feed their own shares into the protocols.
- **Not yet included.** MQTT/Mosquitto integration, dataset loaders, and the six analytics protocols of the paper (Table A1).

## Citation

Please cite the DRCF paper if you use this code. GitHub shows a "Cite this repository" button generated from `CITATION.cff`.

```bibtex
@article{SaderiOskouei2026DRCF,
  author  = {Saderi Oskouei, Shabnam and Mandal, Kalikinkar and Ghorbani, Ali A.},
  title   = {{DRCF}: A Privacy-Enhanced Distributed Publish-Subscribe System for Secure Data Computation in Untrusted Environments},
  journal = {ACM Transactions on Cyber-Physical Systems},
  volume  = {10},
  number  = {3},
  articleno = {36},
  year    = {2026},
  doi     = {10.1145/3805045}
}
```

## License

MIT, see [LICENSE](LICENSE). Dependencies keep their own licenses: EMP-toolkit (Apache-2.0), Microsoft SEAL (MIT), OpenSSL 3 (Apache-2.0). The ASCON test vectors in `test/kat.h` were generated with the reference implementation [pyascon](https://github.com/meichlseder/pyascon) (CC0).
