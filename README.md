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

## IoT protocols (Table A1)

Each protocol of the paper's Table A1 is implemented with the model from the paper it cites, on the same dataset (paper Table 3). A Python script builds the plaintext model and the public inputs; a C++ program runs the protocol between the two brokers and checks the result.

| Protocol | Task | Model (paper) | Dataset | Status |
|---|---|---|---|---|
| I | Anomaly detection, regression | PARX (Ardakanian et al. 2014) | synthetic smart meter (Liu) | waiting for the paper's synthetic dataset |
| II | Anomaly detection, ML | Prophet as configured by Jaiswal et al. 2021 | Ausgrid, 10 customers (Jaiswal) | done |
| III | Load forecasting, regression | sliding-window (S)ARIMA (Alberg & Last 2018) | UCI household power | done |
| IV | Load forecasting, ML | RBF SVR (Chen, Chang & Lin 2004) | UCI ElectricityLoadDiagrams | done |
| V | Price forecasting, regression | ARIMA (Contreras et al. 2003) | electricity price (Kaggle) | waiting for the dataset |
| VI | Price forecasting, ML | SVRARIMA (Che & Wang 2010) | Polish day-ahead market | done |

**Protocol II** (`iot/p2_prophet_prep.py`, `iot/protocol2.cpp`). Prophet's configuration from the authors' notebook: piecewise-linear trend with 25 changepoints, daily (order 4) and yearly (order 10) seasonality, weekday/weekend weekly seasonality (order 3), Australian holidays, 99% interval; an anomaly is a reading outside the interval or equal to 0. With Gaussian priors the fit is linear in the readings (w = A·y with A public), so each broker fits on its share locally; the squared residuals need the cross term r1·r2, computed with CKKS; the decision runs in a garbled circuit. Every reading goes through DistAEAD first.

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

Interval width: the code uses Prophet's 99% interval (z = 2.576, `Z` in `iot/p2_prophet_prep.py`), as Jaiswal et al. did. Table A1 of the DRCF paper writes the bound as 3σ; with z = 3 the protocol flags fewer readings and recall against the published labels drops to about 0.68 (precision about 0.98).

**Protocol III** (Alberg & Last, 2018, on the UCI household data).

`scripts/train_protocol3.py` (model owner), secure part in the generic engine `src/drcf_secure.cpp` (`ROWS` jobs).

SWHSA, the paper's best hourly algorithm: SARIMA(0,0,1)(0,1,1)₂₄ (their 9.19% model), training window 96 days (2304 h), validation window 24 h, the window slides by 24 h, and the model is re-induced when MAPE_val / MAPE_tr > Th. The paper does not give Th; the default is 1 (a parameter of the script). The data is the hourly mean of Global_active_power.

Secure flow: the owner fits each window in the clear and writes the 24 forecasts of the next day as linear predictors in past readings (AR(∞) form of the SARIMA, 2304 lags, recursion for hours 2–24 folded into the weights). The brokers evaluate them on the shared readings with CKKS cross terms; the coefficients stay secret-shared.

```sh
python3 examples/prepare_uci_household.py household_power_consumption.txt > build/uci_hourly.csv
python3 scripts/train_protocol3.py build/uci_hourly.csv build/job_p3.txt build/reference_p3.txt 365
cd build && ../run.sh ./drcf_secure job_p3.txt result_p3.txt
```

Results, last 365 days of the data (8,760 hourly forecasts, 194 refits):

| | MAPE |
|---|---|
| secure | 63.26% |
| plaintext linear predictor | 63.26% (max difference 0.008 kW) |
| statsmodels SARIMA forecast | 64.68% |
| naive hourly baseline, y(t−24) | 68.74% |

Cost: 92 s and 4.1 GB of localhost traffic for the year (CKKS only, no garbled circuit). The truncated predictor matches statsmodels to 0.002 kW in the median; in about 7% of the hours it differs by more than 0.1 kW, in windows where the fitted seasonal MA term is close to −1 and 2304 lags are not enough. The MAPE is much higher than the paper's because this is one household (spiky, low night load) and theirs were six district meters; the gain over the naive baseline is similar. CKKS mask is 2^34 here (job tag `MASK`) at scale 10⁴: the products being hidden are below 2^21, and a 2^40 mask over 1758-term sums costs about 0.3 kW of precision.

**Protocol IV** (Chen, Chang & Lin, 2004, on ElectricityLoadDiagrams).

`scripts/train_protocol4.py` (model owner), `src/protocol4_eld.cpp` (brokers).

The EUNITE 2001 winning model, on the system load (sum of the 321 clients of the hourly Lai et al. version): daily maximum load, 31-day recursive forecast; 14 attributes = maxima of the previous 7 days scaled to [0, 1], six weekday bits (Sunday all zero), one holiday bit; no temperature; winter months only (Jan–Mar, Oct–Dec); RBF ε-SVR with ε = 0.1 (LIBSVM default); C and γ chosen by recursive prediction of a January validation month (January 2013); holiday flags removed in the forecast month, as the paper did. Test month: January 2014.

Secure flow: every client sends shares of its hourly readings for the 7 days before the month; each broker adds its shares (system load); the garbled circuit takes the daily maxima, scales them, and runs the 31 SVR steps, each prediction feeding the next, with support vectors, α, b, min and range secret. Calendar bits are public.

```sh
python3 scripts/train_protocol4.py electricity.txt.gz build/model_p4.txt
cd build && ../run.sh ./protocol4_eld model_p4.txt
```

| | MAPE, January 2014 |
|---|---|
| secure | 5.85% |
| plaintext reference | 5.98% |

23 support vectors (C = 1, γ = 0.5). Cost: 1.1 s, 18 M AND gates, 558 MiB. Fixed-point error grows through the recursion and stays below 0.26% of the forecast. The paper reports 1.95% on the EUNITE data (one regional utility); here the target is the sum of 321 Portuguese clients.

**Protocol VI** (Che & Wang, 2010, on the Polish day-ahead market).

`src/protocol6_polish.cpp`, garbled-circuit SVR in `include/svr_gc.h`, model owner in `scripts/train_protocol6.py`.

SVRARIMA as in the paper: min-max normalisation on the training window (Eq. 3.1); SVR on the phase-space reconstruction with m = 3, τ = 1 (Eq. 2.1–2.3); RBF kernel exp(−‖x − x′‖²/2δ²) with C = 1, δ = 20, ε = 0.001 (Figs. 4, 7); ARIMA on the SVR residuals with p = (24), d = (0), q = (1) for February weeks and p = (1, 24), d = (24), q = (1) for April weeks; training on the 7 days before each forecast week; one-hour-ahead forecasts; RMSE and MAPE. Test weeks are the paper's calendar weeks (Feb 26 – Mar 3, Apr 1 – 7) in 2018, 2019 and 2020.

The model owner trains in the clear and secret-shares the model (Table A1 input β). Secure flow on shared prices:
1. Garbled circuit: normalisation, then N_t = b + Σ α_k exp(−γ‖x_t − sv_k‖²) with support vectors, α and b all secret. exp is fixed point: 2^−y with y split into integer and fraction, a degree-4 polynomial for the fraction (error 3.5e-6) and a secret shift for the integer part. Residuals e_t = x_t − N_t leave the circuit as additive shares.
2. CKKS cross terms: the ARIMA one-step forecast of the residual, written as its linear predictor Σ w_j e_{t−j} (AR(∞) form truncated at 144 lags).
3. Garbled circuit: forecast = (N_t + L_t)·range + min, output as additive shares.

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

As in the paper, the hybrid beats SVR alone every week. Cost per week (168 forecasts, 164 support vectors): about 13 s, 360 M AND gates and 11 GB of localhost traffic, almost all of it the 27,500 kernel evaluations in the garbled circuit.



## Dashboard

`docs/index.html` replays a recorded `drcf_demo` run in the browser: the protocol flow between publishers, B1, B2 and the subscriber; what each broker sees at every step (the page XORs the two shares to show they recover the real reading); attack scenarios (tampered publication, tampered result, a compromised broker); forecasts and alerts per meter; and the cost of each step. It is a single file with no server and opens offline.

To rebuild it from your own run:

```sh
cd build && ../run.sh ./drcf_demo && cd ..
python3 scripts/make_dashboard.py build/trace.js    # writes docs/index.html
```

## Try it on your own data

`drcf.py` lets you run DRCF on any time series and decide which model suits it. It plays the model owner (trains in the clear on a training window), runs the real two-broker computation on secret-shared data with `build/drcf_secure`, and reports accuracy, how closely the secure result matches the plaintext model, and the cost.

Input: a CSV with a time column and a value column (optionally a series id column, public covariates such as temperature, and a 0/1 anomaly label column for scoring). The last `--test` points are evaluated.

```sh
pip install pandas numpy scikit-learn statsmodels   # front end only

# forecasting with a linear model on lags plus public features
python3 drcf.py --data load.csv --time timestamp --value kwh --task forecast --model linear \
                --lags 1,2,24,168 --public intercept,hour,weekday --exog temperature

# anomaly detection, Prophet-style model (Protocol II), scored against labels
python3 drcf.py --data meter.csv --time ts --value kwh --label anomaly --task anomaly --model prophet --k 3

# RBF SVR with ARIMA on its residuals (Protocol VI)
python3 drcf.py --data prices.csv --time date --value price --task forecast --model svr \
                --lags 1,2,3 --svr-C 1 --svr-delta 20 --svr-epsilon 0.001 --arima 24,0,1 --train 168
```

| `--model` | what it computes | secure realisation | covers |
|---|---|---|---|
| `linear` | ŷ = Σ βⱼ·y(t − lagⱼ) + public features (intercept, hour, weekday, `--exog` columns) | CKKS cross terms for secret lags × shared β | AR, ARIMA predictors, regression, PARX with weather columns (Protocols I, III, V) |
| `prophet` | trend + daily, yearly, weekday and weekend seasonality; anomaly if \|y − ŷ\| > k·σ (`--zero-rule` adds y = 0) | fit on each broker's share, σ via CKKS, decision in a GC | Protocol II |
| `svr` | RBF SVR on min-max normalised lags, optional `--arima p,d,q` on its residuals | exact kernel in a GC, ARIMA via CKKS | Protocols IV, VI |

Tasks: `forecast` (one step ahead for every test point) or `anomaly` (flag test points; for `linear` and `svr` the threshold is k times the training-residual standard deviation). Run `python3 drcf.py --help` for every option. `examples/prepare_ausgrid.py` and `examples/prepare_polish.py` turn the two public datasets into the CSV format above.

Example runs (localhost):

| data | task, model | accuracy (secure / plaintext) | secure = plaintext | cost |
|---|---|---|---|---|
| Polish prices, last week to 2019-03-04 | forecast, `linear` lags 1,2,24,168 + hour, weekday, demand, wind | MAPE 2.918% / 2.918% | max diff 0.01 PLN | 0.04 s, 0.9 MiB |
| same week | forecast, `svr` lags 1–3 + ARIMA(24,0,1) | MAPE 5.91% / 5.93% | max diff 0.24 PLN | 14 s, 364 M AND gates |
| Ausgrid customer 7, 35,088 points | anomaly, `prophet` k = 2.576, zero rule | F1 0.961 vs Jaiswal et al. labels | 99.99% of flags | 1.2 s, 487 MiB |
| Ausgrid customer 7, last 17,520 points | anomaly, `linear` lags 1,2,48,336 + hour, weekday | F1 0.507 vs labels | 99.99% of flags | 1.1 s, 254 MiB |

Cost guide: `linear` and `prophet` take seconds for tens of thousands of points. `svr` costs about 2,100 AND gates per test point × support vector, so keep that product around 30,000 or below.

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
- **Not yet included.** MQTT/Mosquitto integration and Protocols I, III, IV and V of Table A1 (Protocols II and VI are included).
- **Model training.** For Protocols III, IV and VI the model owner trains in the clear on its own historical data and shares the model, as Table A1 assumes; training on shared data is not implemented.

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
