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
| Table A1 protocols on your data | `drcf.py`, `drcf_models.py`, `src/drcf_secure.cpp`, `src/protocol4_daily.cpp`, `include/svr_gc.h`, `include/prophet_lite.h` | Model owner in Python (paper model, fitted in the clear); the two brokers in C++ on secret-shared data. |
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

## IoT protocols (Table A1)

Every protocol of the paper's Table A1 runs on your own data with one command. Give a CSV with a time column and a value column; `drcf.py` uses the model and settings of the paper Table A1 cites, plays the model owner (fits in the clear), runs the two brokers on secret-shared data, and reports accuracy, how closely the secure result matches the plaintext model, and the cost. Every paper setting can be overridden (`python3 drcf.py --help`).

```sh
pip install pandas numpy scipy scikit-learn statsmodels holidays   # front end only
python3 drcf.py --protocol 1 --data meter.csv --time ts --value kwh --temp temperature --holidays CA
python3 drcf.py --protocol 2 --data meter.csv --time ts --value kwh --label anomaly
python3 drcf.py --protocol 3 --data load.csv  --time ts --value kw --test 720
python3 drcf.py --protocol 4 --data load.csv  --time ts --value kw --series client --holidays PT
python3 drcf.py --protocol 5 --data price.csv --time ts --value price
python3 drcf.py --protocol 6 --data price.csv --time ts --value price
```

| | Task | Model (cited paper) | Paper settings used | Secure realisation |
|---|---|---|---|---|
| I | anomaly detection | PARX (Ardakanian et al. 2014) | one OLS model per hour × (weekday, weekend/holiday); lags = same hour on the previous 3 days of that type; temperature terms (T−20)⁺, (16−T)⁺, (5−T)⁺; flag if \|y − ŷ\| > 3σ of the season | CKKS cross terms, decision in a GC |
| II | anomaly detection | Prophet (Jaiswal et al. 2021) | 25 changepoints, daily/yearly/weekday/weekend seasonality, 99% interval, zero readings flagged | fit on each broker's share, σ via CKKS, decision in a GC |
| III | load forecasting | SWHSA (Alberg & Last 2018) | SARIMA(0,0,1)(0,1,1)₂₄, 96-day window, 24 h ahead, slide 24 h, refit when MAPE_val / MAPE_tr > 1 | per-hour linear predictors, CKKS |
| IV | load forecasting | RBF SVR (Chen, Chang & Lin 2004) | daily peak, 31 days ahead recursively; 7 previous peaks + weekday bits + holiday bit; winter training months; C, γ by validation | daily peaks and all SVR steps in a GC |
| V | price forecasting | ARIMA (Contreras et al. 2003) | Eq. (3): multiplicative AR (1–5)(23,24,47,48,72,96,120,144)(168,336,504), MA (1,2)(24)(168,336,504) on log prices, 24 h ahead each day | per-hour linear predictors, CKKS |
| VI | price forecasting | SVRARIMA (Che & Wang 2010) | SVR on lags 1–3, C = 1, δ = 20, ε = 0.001, ARIMA(24,0,1) on its residuals, 7-day training window | SVR in a GC, ARIMA via CKKS |

Notes:
- **I.** The occupancy terms of Ardakanian et al. depend on the reading being checked, so the anomaly model leaves them out. `--temp` is optional (weather is public data). `--holidays` takes a country code.
- **II.** Jaiswal et al. use z = 2.576 (99%); Table A1 writes 3σ. Use `--k 3` for the latter.
- **III and V.** The (S)ARIMA forecasts are written as linear predictors in past readings (AR(∞) form, up to `--truncate` 2304 lags; multi-step recursion folded into the weights), so the brokers only need CKKS; the coefficients stay secret-shared. Any factored ARIMA can be given with `--ar`, `--ma`, `--diff`, `--const`, `--log`, `--window`, `--horizon`, `--refit`. For Eq. (4) of Contreras et al. (California) use `--ar "1,2;23,24,47,48,72,96,120,144;167,168,169,192" --diff 1,24,168 --ma "1,2;24,48,72,96;144;168,336,504" --const 0`. With `--log` each publisher takes the log of its own reading before sharing and the subscriber exponentiates what it receives.
- **IV.** With `--series`, every series is a publisher; each broker adds its shares to get the system load. `--months` sets the training months (paper: Jan–Mar, Oct–Dec).
- Protocols I, III and V use hourly data; other spacings are resampled to hourly means.
- Fixed point for CKKS is chosen from the data: shares keep about 13 bits of statistical hiding above the readings and the masks about 13 bits above the products they hide. The report prints the scale and mask used.
- The model owner always trains in the clear and shares the model, as Table A1 assumes.

Sanity runs on public data (localhost):

| Protocol | data | secure / plaintext | cost |
|---|---|---|---|
| I | UCI household power, last 720 h | flags 100% equal | 0.1 s, 13 MiB |
| II | Ausgrid customer 7, 35,088 readings, with DistAEAD (`iot/protocol2.cpp`) | 35,088 / 35,088 equal; precision 0.967, recall 0.956 vs Jaiswal et al. labels | 55 s |
| III | UCI household power, last 720 h | MAPE 66.18% / 66.16% | 8 s, 339 MiB |
| IV | ElectricityLoadDiagrams, 40 clients, last 31 days | MAPE 9.64% / 9.37% | 3 s, 2.7 GB |
| V | Polish day-ahead prices, last week | MAPE 7.35% / 7.34% | 4 s, 158 MiB |
| VI | Polish day-ahead prices, last week | MAPE 5.91% / 5.93% | 14 s, 364 M AND gates |

Fixed-point differences: under 0.6% of the forecast for V (CKKS over 2304-lag sums) and IV (Q14 kernel through a 31-step recursion).

### Paper reproductions

Two protocols also have scripts that redo the cited paper's own experiment.

**Protocol II** on the Jaiswal et al. labelled Ausgrid data (`iot/p2_prophet_prep.py`, `iot/protocol2.cpp`, every reading through DistAEAD first):

```sh
python3 iot/p2_prophet_prep.py "<path to Anomaly labelled dataset.csv>" 7,29,30,64,155,160,184,202,206,215
./run.sh build/protocol2 iot/data/p2_c7
```

Results (35,088 half-hourly readings per customer, two years):

| Customer | secure = plaintext model | agreement with the paper's labels | precision | recall | time (incl. DistAEAD) |
|---|---|---|---|---|---|
| 7 | 35,088 / 35,088 | 99.73% | 0.967 | 0.956 | 55 s |
| 29 | 35,088 / 35,088 | 98.74% | 0.845 | 0.755 | 62 s |
| 30 | 35,088 / 35,088 | 99.73% | 0.968 | 0.938 | 64 s |
| 64 | 35,088 / 35,088 | 98.92% | 0.873 | 0.850 | 63 s |
| 155 | 35,088 / 35,088 | 99.69% | 0.970 | 0.936 | 62 s |
| 160 | 35,088 / 35,088 | 99.62% | 0.945 | 0.925 | 64 s |
| 184 | 35,088 / 35,088 | 98.92% | 0.889 | 0.820 | 65 s |
| 202 | 35,088 / 35,088 | 99.10% | 0.916 | 0.829 | 62 s |
| 206 | 35,088 / 35,088 | 99.68% | 0.977 | 0.918 | 63 s |
| 215 | 35,088 / 35,088 | 99.40% | 0.951 | 0.891 | 62 s |

The remaining gap to the paper's labels comes from Prophet's per-customer tuned hyperparameters, which were not published (defaults are used here), and from replacing Prophet's Laplace changepoint prior by a Gaussian. Without DistAEAD the protocol takes about 6 s per customer; CKKS error on r² is about 0.002 Wh².

**Protocol VI** on the paper's test weeks of the Polish market (`scripts/train_protocol6.py`, `src/protocol6_polish.cpp`):

```sh
python3 scripts/train_protocol6.py <polish repo>/data/data.xlsx build
cd build && ../run.sh ./protocol6_polish prices_p6.txt model_p6.txt reference_p6.txt
```

| week | MAPE, secure | MAPE, plaintext reference | SVR alone (plaintext) | max \|secure − reference\| |
|---|---|---|---|---|
| 2018-02-26 | 12.10% | 12.11% | 18.24% | 0.26 PLN |
| 2019-02-26 | 5.94% | 5.96% | 14.31% | 0.24 PLN |
| 2020-02-26 | 11.41% | 11.41% | 21.61% | 0.35 PLN |
| 2018-04-01 | 7.35% | 7.35% | 26.87% | 0.30 PLN |
| 2019-04-01 | 3.55% | 3.54% | 10.54% | 0.37 PLN |
| 2020-04-01 | 5.18% | 5.18% | 17.10% | 0.38 PLN |

As in the paper, the hybrid beats SVR alone every week.

## Dashboard

`docs/index.html` replays a recorded `drcf_demo` run in the browser: the protocol flow between publishers, B1, B2 and the subscriber; what each broker sees at every step (the page XORs the two shares to show they recover the real reading); attack scenarios (tampered publication, tampered result, a compromised broker); forecasts and alerts per meter; and the cost of each step. It is a single file with no server and opens offline.

To rebuild it from your own run:

```sh
cd build && ../run.sh ./drcf_demo && cd ..
python3 scripts/make_dashboard.py build/trace.js    # writes docs/index.html
```

## Generic models

Besides the six protocols, `drcf.py --model ...` runs the building blocks with your own choices:

| `--model` | what it computes | secure realisation |
|---|---|---|
| `linear` | ŷ = Σ βⱼ·y(t − lagⱼ) + public features (intercept, hour, weekday, `--exog` columns) | CKKS cross terms |
| `arima` | any factored (S)ARIMA, sliding window, multi-step | CKKS |
| `parx` | PARX with your `--p`, temperature and holidays | CKKS, GC decision |
| `prophet` | trend + seasonality, anomaly if \|y − ŷ\| > k·σ | shares + CKKS + GC |
| `svr` | RBF SVR on lags, optional `--arima p,d,q` on its residuals | GC + CKKS |
| `svr-daily` | daily-peak SVR, recursive | GC |

```sh
python3 drcf.py --data load.csv --time ts --value kwh --task forecast --model linear \
                --lags 1,2,24,168 --public intercept,hour,weekday --exog temperature
python3 drcf.py --data load.csv --time ts --value kw --task forecast --model arima \
                --ar "1,2" --ma "24" --diff 24 --window 1344 --horizon 24 --refit every
```

Tasks: `forecast` or `anomaly` (threshold = k × training-residual standard deviation). Cost guide: CKKS models take seconds for tens of thousands of points; `svr` costs about 2,100 AND gates per test point × support vector.

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
- **Not yet included.** MQTT/Mosquitto integration and Protocols I and V of Table A1 (their models are linear predictors, so they can be run on your own data with `drcf.py --model linear`).
- **Model training.** For every protocol except II the model owner trains in the clear on its own historical data and shares the model, as Table A1 assumes; training on shared data is not implemented.

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
